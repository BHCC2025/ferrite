//! Speculative greedy decode driver: the DSpark drafter proposes k tokens, ONE (k+1)-row `Engine::step_rows` verifies them
//! (rows = the last committed token + the drafts), the greedy-matching prefix and the bonus token are committed, the rejected
//! rows are rolled back (`truncate`) and the kept rows' taps are absorbed into the drafter. Uses only the engine's public API
//! (`step_rows`, `truncate`, `taps`, `ids_seen`) plus the context handle for the drafter's rebind.
//! Invariant between rounds: the engine has fed ids[..n-1] (n = committed tokens), the drafter has absorbed the taps of those
//! positions (`n_abs == n - 1`), and `ids[n-1]` is the last committed token, not yet fed.
use super::draft::Drafter;
use super::{argmax, prefill_rows, Engine, RMAX};
use anyhow::{ensure, Result};
use fe_cuda::memcpy_dtoh;
use std::time::Instant;

pub const TAP: usize = 3 * 5120;

#[derive(Clone, Debug, Default)]
pub struct SpecStats {
    pub rounds: usize,
    pub tokens: usize,
    /// rounds by accepted drafts 0..=5
    pub hist: [usize; 16],
    /// rounds whose drafts were copied from the context, and the tokens they committed
    pub copy_rounds: usize,
    pub copy_tokens: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub absorb_ms: f64,
    pub total_ms: f64,
    /// drafts verified (sum of k over rounds)
    pub drafted: usize,
}

/// Plain greedy decode with single-row steps: feed ids[..P], then `n` tokens.
pub fn plain(e: &mut Engine, ids: &[u32], n: usize) -> Result<(Vec<u32>, f64)> {
    let mut lg = None;
    for (i, &t) in ids.iter().enumerate() {
        lg = e.step(t, i + 1 == ids.len())?;
    }
    let t0 = Instant::now();
    let mut out = vec![];
    for _ in 0..n {
        let next = argmax(lg.as_ref().unwrap());
        out.push(next);
        lg = e.step(next, true)?;
    }
    Ok((out, t0.elapsed().as_secs_f64() * 1e3))
}

/// Cost model for the early stop: ms of one verify pass by rows (index = rows - 1) and the drafter pass.
#[derive(Clone, Debug)]
pub struct Cost {
    pub round_ms: Vec<f64>,
    pub draft_ms: f64,
}

/// The k in 0..=kmax with the most expected tokens a millisecond (fe-dspark depth.rs choose_k for one stream; the first best wins).
pub fn choose_k(conf: &[f32], kmax: usize, cost: &Cost) -> usize {
    choose_k_v(conf, kmax, cost).0
}

impl Cost {
    /// Verify ms of a pass of `rows` rows: measured up to the table's length, a linear extension past it.
    pub fn at(&self, rows: usize) -> f64 {
        let n = self.round_ms.len();
        if rows <= n {
            self.round_ms[rows.max(1) - 1]
        } else {
            let slope = if n >= 7 { (self.round_ms[n - 1] - self.round_ms[5]) / (n - 6) as f64 } else { 3.0 };
            self.round_ms[n - 1] + slope.max(0.5) * (rows - n) as f64
        }
    }
}

/// (k, expected tokens a ms) of the best k.
pub fn choose_k_v(conf: &[f32], kmax: usize, cost: &Cost) -> (usize, f64) {
    let (mut best, mut best_v) = (0usize, -1.0f64);
    let (mut surv, mut exp_tokens) = (1.0f64, 1.0f64);
    for kk in 0..=kmax.min(conf.len()) {
        if kk > 0 {
            surv *= 1.0 / (1.0 + (-(conf[kk - 1] as f64)).exp());
            exp_tokens += surv;
        }
        let v = exp_tokens / (cost.at(kk + 1) + cost.draft_ms);
        if v > best_v {
            best = kk;
            best_v = v;
        }
    }
    (best, best_v)
}

/// Copy drafting: an index of the 3-grams of the context (their last positions), and the proposal of the longest earlier match of the context's
/// suffix: the tokens that followed it. Verbatim spans of the prompt (edits, quotes, repeated code) are accepted at a rate the drafter cannot reach.
pub struct CopyIndex {
    cx: Vec<u32>,
    map: std::collections::HashMap<u64, Vec<u32>>,
    /// running estimate of the per-token acceptance of copied drafts, by match length (4-7, 8-15, 16+)
    pub s: [f64; 3],
}

impl CopyIndex {
    pub fn new(ids: &[u32]) -> CopyIndex {
        let mut c = CopyIndex { cx: vec![], map: Default::default(), s: [0.7, 0.85, 0.93] };
        c.push(ids);
        c
    }
    fn key(a: u32, b: u32, c: u32) -> u64 {
        ((a as u64) << 42) ^ ((b as u64) << 21) ^ (c as u64) ^ ((a as u64) << 7)
    }
    pub fn push(&mut self, toks: &[u32]) {
        for &t in toks {
            self.cx.push(t);
            let n = self.cx.len();
            if n >= 3 {
                let k = Self::key(self.cx[n - 3], self.cx[n - 2], self.cx[n - 1]);
                self.map.entry(k).or_default().push((n - 1) as u32);
            }
        }
    }
    /// Drop everything after position `n`.
    pub fn len(&self) -> usize {
        self.cx.len()
    }
    /// (tokens that followed the longest earlier match of the suffix, the match length), at most `lmax` tokens.
    pub fn propose(&self, lmax: usize) -> Option<(Vec<u32>, usize)> {
        let n = self.cx.len();
        if n < 4 {
            return None;
        }
        let k = Self::key(self.cx[n - 3], self.cx[n - 2], self.cx[n - 1]);
        let ends = self.map.get(&k)?;
        let mut best: Option<(usize, usize)> = None;
        for &e in ends.iter().rev().take(24) {
            let e = e as usize;
            if e + 1 >= n {
                continue;
            }
            let mut m = 0;
            while m < 64 && m <= e && self.cx[e - m] == self.cx[n - 1 - m] {
                m += 1;
            }
            if m >= 3 && best.map_or(true, |(_, bm)| m > bm) {
                best = Some((e, m));
            }
        }
        let (e, m) = best?;
        let l = lmax.min(n - 1 - e);
        Some((self.cx[e + 1..e + 1 + l].to_vec(), m))
    }
    fn bucket(m: usize) -> usize {
        if m >= 16 { 2 } else if m >= 8 { 1 } else { 0 }
    }
    /// Per-token acceptance estimate for a match of `m` tokens.
    pub fn surv(&self, m: usize) -> f64 {
        self.s[Self::bucket(m)].clamp(0.3, 0.99)
    }
    pub fn learn(&mut self, m: usize, accepted: usize, proposed: usize) {
        let fails = (accepted < proposed) as usize;
        let obs = accepted as f64 / (accepted + fails).max(1) as f64;
        let b = Self::bucket(m);
        self.s[b] = 0.9 * self.s[b] + 0.1 * obs;
    }
}

/// One lane's speculative decode state between passes (the resumable form of `spec_stream`'s loop): the scheduler asks it for this round's drafts
/// (`plan`), feeds [last, drafts...] in a shared pass, and hands the logits rows back (`commit`).
pub struct SpecRun {
    pub last: u32,
    pub st: SpecStats,
    zero_run: usize,
    cpi: CopyIndex,
    copy_on: bool,
    lmax: usize,
    kmax: usize,
    plan: Vec<u32>,
    plan_copy: bool,
    plan_cml: usize,
}

impl SpecRun {
    /// `ids`: prompt + the first generated token (its last token is `last`, not yet fed).
    pub fn new(ids: &[u32], kmax: usize, copy_on: bool) -> SpecRun {
        let lmax: usize = std::env::var("DSC_COPY_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(15).min(RMAX - 1).min(15);
        SpecRun { last: ids[ids.len() - 1], st: SpecStats::default(), zero_run: 0, cpi: CopyIndex::new(ids), copy_on, lmax, kmax, plan: vec![], plan_copy: false, plan_cml: 0 }
    }

    /// The drafts of this round (at most `cap`): copy proposals or the drafter's (the drafter's lane must be active); `pos` = the lane's fed length.
    pub fn plan(&mut self, dr: &mut Drafter, pos: usize, cost: Option<&Cost>, cap: usize) -> Result<&[u32]> {
        let kmax = self.kmax.min(cap);
        self.plan_copy = false;
        self.plan_cml = 0;
        if cap == 0 {
            self.plan.clear();
            return Ok(&self.plan);
        }
        let mut d: Option<super::draft::DraftOut> = None;
        if self.copy_on {
            if let (Some((prop, ml)), Some(c)) = (self.cpi.propose(self.lmax.min(cap)), cost) {
                if ml >= 4 && !prop.is_empty() {
                    self.plan_cml = ml;
                    let sv = self.cpi.surv(ml);
                    let (mut bl, mut bv, mut et, mut sp) = (0usize, -1.0f64, 1.0f64, 1.0f64);
                    for l in 1..=prop.len() {
                        sp *= sv;
                        et += sp;
                        let v = et / c.at(l + 1);
                        if v > bv {
                            bl = l;
                            bv = v;
                        }
                    }
                    let take = if ml >= 16 {
                        true
                    } else {
                        let t1 = Instant::now();
                        let dd = dr.draft(self.last, pos, self.kmax.max(1).min(5))?;
                        self.st.draft_ms += t1.elapsed().as_secs_f64() * 1e3;
                        let (_, vd) = choose_k_v(&dd.conf, kmax.min(dd.conf.len()), c);
                        d = Some(dd);
                        bv > vd * 1.2
                    };
                    if take {
                        self.plan = prop[..bl].to_vec();
                        self.plan_copy = true;
                        return Ok(&self.plan);
                    }
                }
            }
        }
        let dd = match d.take() {
            Some(dd) => dd,
            None => {
                let t1 = Instant::now();
                let dd = dr.draft(self.last, pos, self.kmax.max(1).min(5))?;
                self.st.draft_ms += t1.elapsed().as_secs_f64() * 1e3;
                dd
            }
        };
        let mut k = match cost {
            Some(c) => choose_k(&dd.conf, kmax.min(dd.conf.len()), c),
            None => kmax,
        };
        self.zero_run = if k == 0 { self.zero_run + 1 } else { 0 };
        if self.zero_run >= 24 && kmax >= 2 {
            k = 2;
            self.zero_run = 0;
        }
        self.plan = dd.ids[..k].to_vec();
        Ok(&self.plan)
    }

    /// The pass fed [last] + the plan: `lgs` are its logits rows. Returns (accepted drafts, next token, the committed tokens).
    pub fn commit(&mut self, lgs: &[Vec<f32>], pick: Option<&mut dyn FnMut(&[Vec<f32>], &[u32]) -> (usize, u32)>) -> (usize, u32, Vec<u32>) {
        let k = self.plan.len();
        let (m, nxt) = match pick {
            Some(pk) => pk(lgs, &self.plan),
            None => {
                let g: Vec<u32> = lgs.iter().map(|l| argmax(l)).collect();
                let m = (0..k).take_while(|&j| self.plan[j] == g[j]).count();
                (m, g[m])
            }
        };
        let mut out: Vec<u32> = self.plan[..m].to_vec();
        out.push(nxt);
        if self.plan_copy {
            self.cpi.learn(self.plan_cml, m, k);
            self.st.copy_rounds += 1;
            self.st.copy_tokens += out.len();
        }
        self.cpi.push(&out);
        self.last = nxt;
        self.st.rounds += 1;
        self.st.hist[m.min(15)] += 1;
        self.st.drafted += k;
        self.st.tokens += out.len();
        (m, nxt, out)
    }
}

/// A lane's draft candidates for this round: tokens and the cumulative probability that the first j are all accepted.
pub struct Cand {
    pub ids: Vec<u32>,
    pub surv: Vec<f64>,
    pub copy: bool,
    pub cml: usize,
}

impl SpecRun {
    /// The candidates of this round (copy proposal or the drafter's, whichever promises more per ms), up to `cap` tokens; the drafter's lane
    /// must be active. The scheduler then picks how many of them ride in the pass (`set_plan`).
    pub fn candidates(&mut self, dr: &mut Drafter, pos: usize, cost: Option<&Cost>, cap: usize) -> Result<Cand> {
        let drafter = |this: &mut SpecRun, dr: &mut Drafter| -> Result<Cand> {
            let t1 = Instant::now();
            let dd = dr.draft(this.last, pos, this.kmax.max(1).min(5))?;
            this.st.draft_ms += t1.elapsed().as_secs_f64() * 1e3;
            let mut surv = vec![];
            let mut sv = 1.0f64;
            for c in dd.conf.iter() {
                sv *= 1.0 / (1.0 + (-(*c as f64)).exp());
                surv.push(sv);
            }
            let n = dd.ids.len().min(cap);
            Ok(Cand { ids: dd.ids[..n].to_vec(), surv: surv[..n].to_vec(), copy: false, cml: 0 })
        };
        if self.copy_on {
            if let (Some((prop, ml)), Some(c)) = (self.cpi.propose(self.lmax.min(cap)), cost) {
                if ml >= 4 && !prop.is_empty() {
                    let sv = self.cpi.surv(ml);
                    let mut surv = vec![];
                    let mut sp = 1.0;
                    for _ in 0..prop.len() {
                        sp *= sv;
                        surv.push(sp);
                    }
                    let cp = Cand { ids: prop, surv, copy: true, cml: ml };
                    if ml >= 16 {
                        return Ok(cp);
                    }
                    let dc = drafter(self, dr)?;
                    // tokens a ms of each (single lane view) decides
                    let val = |x: &Cand| -> f64 {
                        let (mut best, mut et) = (0f64, 1.0f64);
                        for l in 0..=x.ids.len() {
                            if l > 0 {
                                et += x.surv[l - 1];
                            }
                            best = best.max(et / (c.at(l + 1) + if x.copy { 0.0 } else { c.draft_ms }));
                        }
                        best
                    };
                    return Ok(if val(&cp) > val(&dc) * 1.2 { cp } else { dc });
                }
            }
        }
        drafter(self, dr)
    }

    /// Does this round need the drafter? false: a long verbatim copy proposal rides alone (as `candidates` decides).
    pub fn wants_draft(&self, cost: Option<&Cost>, cap: usize) -> bool {
        if self.copy_on {
            if let (Some((prop, ml)), Some(_)) = (self.cpi.propose(self.lmax.min(cap)), cost) {
                if ml >= 16 && !prop.is_empty() {
                    return false;
                }
            }
        }
        true
    }

    /// Tokens a drafter call makes for this run.
    pub fn draft_k(&self) -> usize {
        self.kmax.max(1).min(5)
    }

    /// `candidates` with the drafter's output made by the caller (one drafter pass for several lanes): `dd` is this lane's draft, None
    /// only when `wants_draft` said the round needs none. `ms` = this lane's share of the drafter's time.
    pub fn candidates_from(&mut self, dd: Option<super::draft::DraftOut>, ms: f64, cost: Option<&Cost>, cap: usize) -> Result<Cand> {
        self.st.draft_ms += ms;
        let from = |dd: &super::draft::DraftOut| -> Cand {
            let mut surv = vec![];
            let mut sv = 1.0f64;
            for c in dd.conf.iter() {
                sv *= 1.0 / (1.0 + (-(*c as f64)).exp());
                surv.push(sv);
            }
            let n = dd.ids.len().min(cap);
            Cand { ids: dd.ids[..n].to_vec(), surv: surv[..n].to_vec(), copy: false, cml: 0 }
        };
        if self.copy_on {
            if let (Some((prop, ml)), Some(c)) = (self.cpi.propose(self.lmax.min(cap)), cost) {
                if ml >= 4 && !prop.is_empty() {
                    let sv = self.cpi.surv(ml);
                    let mut surv = vec![];
                    let mut sp = 1.0;
                    for _ in 0..prop.len() {
                        sp *= sv;
                        surv.push(sp);
                    }
                    let cp = Cand { ids: prop, surv, copy: true, cml: ml };
                    if ml >= 16 {
                        return Ok(cp);
                    }
                    let Some(dd) = dd.as_ref() else { return Ok(cp) };
                    let dc = from(dd);
                    let val = |x: &Cand| -> f64 {
                        let (mut best, mut et) = (0f64, 1.0f64);
                        for l in 0..=x.ids.len() {
                            if l > 0 {
                                et += x.surv[l - 1];
                            }
                            best = best.max(et / (c.at(l + 1) + if x.copy { 0.0 } else { c.draft_ms }));
                        }
                        best
                    };
                    return Ok(if val(&cp) > val(&dc) * 1.2 { cp } else { dc });
                }
            }
        }
        let dd = dd.ok_or_else(|| anyhow::anyhow!("candidates_from: no draft for a round that needs one"))?;
        Ok(from(&dd))
    }

    pub fn set_plan(&mut self, c: &Cand, k: usize) {
        self.plan = c.ids[..k].to_vec();
        self.plan_copy = c.copy && k > 0;
        self.plan_cml = c.cml;
    }

    /// This round carries no drafts.
    pub fn clear_plan(&mut self) {
        self.plan.clear();
        self.plan_copy = false;
        self.plan_cml = 0;
    }

    /// The round's verdict when the request decides what it keeps: `m` drafts were accepted, `kept` is the prefix of (accepted drafts,
    /// the token after them) the request took. `commit` without the pick, for a scheduler whose requests can give tokens back.
    pub fn commit_kept(&mut self, m: usize, kept: &[u32]) {
        let k = self.plan.len();
        if self.plan_copy {
            self.cpi.learn(self.plan_cml, m, k);
            self.st.copy_rounds += 1;
            self.st.copy_tokens += kept.len();
        }
        self.cpi.push(kept);
        if let Some(&t) = kept.last() {
            self.last = t;
        }
        self.st.rounds += 1;
        self.st.hist[m.min(15)] += 1;
        self.st.drafted += k;
        self.st.tokens += kept.len();
    }

    /// Tokens that joined the sequence outside a draft round (the engine's forced token, the pick after it).
    pub fn push_tokens(&mut self, toks: &[u32]) {
        self.cpi.push(toks);
        if let Some(&t) = toks.last() {
            self.last = t;
        }
        self.st.tokens += toks.len();
    }

    /// The sequence was rewound to `seq` (its last token is not fed).
    pub fn reseed(&mut self, seq: &[u32]) {
        self.cpi = CopyIndex::new(seq);
        self.last = seq[seq.len() - 1];
        self.clear_plan();
        self.zero_run = 0;
    }
}

/// Feed the prompt except its last token in chunks, absorbing the taps of every fed row into the drafter.
fn prefill(e: &mut Engine, dr: &mut Drafter, ids: &[u32], host: &mut [u16]) -> Result<()> {
    let np = ids.len() - 1;
    let mut i = 0;
    while i < np {
        let c = e.chunk_for(np - i);
        e.step_rows(&ids[i..i + c], false)?;
        if c > 8 {
            let mut big = vec![0u16; c * TAP];
            memcpy_dtoh(&mut big, e.taps.ptr())?;
            let mut j = 0;
            while j < c {
                let cc = (c - j).min(8);
                dr.absorb(&big[j * TAP..(j + cc) * TAP], cc, i + j)?;
                j += cc;
            }
        } else {
            memcpy_dtoh(&mut host[..c * TAP], e.taps.ptr())?;
            dr.absorb(&host[..c * TAP], c, i)?;
        }
        i += c;
    }
    Ok(())
}

/// Speculative greedy decode of `n` tokens after the prompt `ids` (>= 2 tokens), drafting `kmax` (<= 5) and verifying all of
/// them (`cost` None) or the confidence-chosen prefix (`cost` Some). Returns exactly n tokens (the last round is cut).
pub fn spec(e: &mut Engine, dr: &mut Drafter, ids: &[u32], n: usize, kmax: usize, cost: Option<&Cost>) -> Result<(Vec<u32>, SpecStats, f64)> {
    ensure!(ids.len() >= 2 && kmax >= 1 && kmax < RMAX && kmax <= 5, "spec: prompt {} kmax {kmax}", ids.len());
    dr.set_back(e._ctx.clone());
    let mut host = vec![0u16; RMAX * TAP];
    prefill(e, dr, ids, &mut host)?;
    let mut last = ids[ids.len() - 1];
    let mut out: Vec<u32> = vec![];
    let mut st = SpecStats::default();
    let t0 = Instant::now();
    while out.len() < n {
        let pos = e.ids_seen.len();
        let t1 = Instant::now();
        let d = dr.draft(last, pos, kmax)?;
        st.draft_ms += t1.elapsed().as_secs_f64() * 1e3;
        let k = match cost {
            Some(c) => choose_k(&d.conf, kmax, c),
            None => kmax,
        };
        let mut rows = vec![last];
        rows.extend_from_slice(&d.ids[..k]);
        let t2 = Instant::now();
        let lgs = e.step_rows(&rows, true)?.unwrap();
        st.verify_ms += t2.elapsed().as_secs_f64() * 1e3;
        let g: Vec<u32> = lgs.iter().map(|l| argmax(l)).collect();
        let m = (0..k).take_while(|&j| d.ids[j] == g[j]).count();
        out.extend_from_slice(&d.ids[..m]);
        out.push(g[m]);
        // keep rows 0..=m (last + m accepted drafts), roll back the rest
        e.truncate(pos + m + 1);
        let t3 = Instant::now();
        memcpy_dtoh(&mut host[..(m + 1) * TAP], e.taps.ptr())?;
        dr.absorb(&host[..(m + 1) * TAP], m + 1, pos)?;
        st.absorb_ms += t3.elapsed().as_secs_f64() * 1e3;
        last = g[m];
        st.rounds += 1;
        st.hist[m] += 1;
        st.drafted += k;
    }
    out.truncate(n);
    st.tokens = n;
    st.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    let ms = st.total_ms;
    Ok((out, st, ms))
}

/// One serving sequence kept between requests (prefix cache): the engine's caches plus the host copy of every fed position's DSpark
/// taps, so the drafter's window can be rebuilt after a rollback. Invariant: `hist` holds the taps of exactly `e.ids_seen`.
#[derive(Default)]
pub struct Session {
    pub hist: Vec<u16>,
}

impl Session {
    /// Roll the engine back to the longest common prefix with `ids` (keeping at least the last prompt token unfed) when the rollback
    /// is within `Engine::max_rewind`, else start from scratch. Returns the number of prompt positions already fed (the cached prefix).
    pub fn begin(&mut self, e: &mut Engine, ids: &[u32]) -> Result<usize> {
        let seen = &e.ids_seen;
        let lcp = seen.iter().zip(ids).take_while(|(a, b)| a == b).count();
        let p = lcp.min(ids.len() - 1).min(self.hist.len() / TAP);
        if p == 0 || seen.len() - p > e.max_rewind() {
            e.reset()?;
            self.hist.clear();
            return Ok(0);
        }
        e.truncate(p);
        self.hist.truncate(p * TAP);
        Ok(p)
    }

    /// `Engine::step_rows` plus the host copy of the rows' taps.
    pub fn feed(&mut self, e: &mut Engine, tokens: &[u32], want: bool) -> Result<Option<Vec<Vec<f32>>>> {
        let r = e.step_rows(tokens, want)?;
        let n = self.hist.len();
        self.hist.resize(n + tokens.len() * TAP, 0);
        memcpy_dtoh(&mut self.hist[n..], e.taps.ptr())?;
        Ok(r)
    }

    /// Feed ids[start .. len-1] in RMAX-row passes (the last prompt token stays unfed).
    pub fn prefill(&mut self, e: &mut Engine, ids: &[u32], start: usize) -> Result<()> {
        let np = ids.len() - 1;
        let mut i = start;
        while i < np {
            let c = e.chunk_for(np - i);
            self.feed(e, &ids[i..i + c], false)?;
            i += c;
        }
        Ok(())
    }

    /// Rebuild the drafter's window (its ring holds the last 256 positions) from the host taps of positions < n.
    pub fn sync_drafter(&self, dr: &mut Drafter, n: usize) -> Result<()> {
        dr.reset();
        let mut i = n.saturating_sub(256);
        while i < n {
            let c = (n - i).min(8);
            dr.absorb(&self.hist[i * TAP..(i + c) * TAP], c, i)?;
            i += c;
        }
        Ok(())
    }
}

/// Streaming speculative greedy decode for a server, on a `Session`: the engine has fed ids[..start] (from `Session::begin`); feeds the
/// rest of the prompt, then drafts / verifies until `emit` returns false (it gets each round's committed tokens). `cost` Some: the
/// confidence-chosen number of drafts a round (0 = a plain one-row step); None: kmax. Same bits as plain greedy.
pub fn spec_stream(e: &mut Engine, dr: &mut Drafter, ss: &mut Session, ids: &[u32], start: usize, kmax: usize, mut cost: Option<&mut Cost>, mut pick: Option<&mut dyn FnMut(&[Vec<f32>], &[u32]) -> (usize, u32)>, emit: &mut dyn FnMut(&[u32]) -> bool) -> Result<(SpecStats, f64)> {
    ensure!(ids.len() >= 2 && kmax >= 1 && kmax < RMAX && kmax <= 5, "spec_stream: prompt {} kmax {kmax}", ids.len());
    dr.set_back(e._ctx.clone());
    let tp = Instant::now();
    ss.prefill(e, ids, start)?;
    ss.sync_drafter(dr, ids.len() - 1)?;
    let prefill_ms = tp.elapsed().as_secs_f64() * 1e3;
    let mut last = ids[ids.len() - 1];
    let mut st = SpecStats::default();
    let t0 = Instant::now();
    let mut zero_run = 0usize;
    let copy_on = cost.is_some() && std::env::var("DSC_COPY").map_or(true, |v| v != "0");
    let lmax: usize = std::env::var("DSC_COPY_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(15).min(RMAX - 1).min(15);
    let mut cpi = CopyIndex::new(ids);
    loop {
        let pos = e.ids_seen.len();
        // copy proposal from the context (long verbatim matches skip the drafter altogether)
        let mut copy: Option<Vec<u32>> = None;
        let mut d: Option<super::draft::DraftOut> = None;
        let mut copy_surv = 0.0;
        let mut cml = 0usize;
        if copy_on {
            if let (Some((prop, ml)), Some(c)) = (cpi.propose(lmax), cost.as_deref()) {
                if ml >= 4 && !prop.is_empty() {
                    cml = ml;
                    let sv = cpi.surv(ml);
                    let (mut bl, mut bv, mut et) = (0usize, -1.0f64, 1.0f64);
                    let mut sp = 1.0;
                    for l in 1..=prop.len() {
                        sp *= sv;
                        et += sp;
                        let v = et / c.at(l + 1);
                        if v > bv {
                            bl = l;
                            bv = v;
                        }
                    }
                    let take = if ml >= 16 {
                        true
                    } else {
                        let t1 = Instant::now();
                        let dd = dr.draft(last, pos, kmax)?;
                        st.draft_ms += t1.elapsed().as_secs_f64() * 1e3;
                        let (_, vd) = choose_k_v(&dd.conf, kmax, c);
                        d = Some(dd);
                        bv > vd * 1.2
                    };
                    if take {
                        copy = Some(prop[..bl].to_vec());
                        copy_surv = sv;
                    }
                }
            }
        }
        let _ = copy_surv;
        let mut conf_log: Vec<f32> = vec![];
        let (dids, k): (Vec<u32>, usize) = match &copy {
            Some(c) => (c.clone(), c.len()),
            None => {
                let dd = match d.take() {
                    Some(dd) => dd,
                    None => {
                        let t1 = Instant::now();
                        let dd = dr.draft(last, pos, kmax)?;
                        st.draft_ms += t1.elapsed().as_secs_f64() * 1e3;
                        dd
                    }
                };
                let mut k = match cost.as_deref() {
                    Some(c) => choose_k(&dd.conf, kmax, c),
                    None => kmax,
                };
                // the cost model only learns the rows it sees: after a long run of plain steps try a 3-row pass again
                zero_run = if k == 0 { zero_run + 1 } else { 0 };
                if zero_run >= 24 && kmax >= 2 {
                    k = 2;
                    zero_run = 0;
                }
                conf_log = dd.conf.clone();
                (dd.ids[..k].to_vec(), k)
            }
        };
        let mut rows = vec![last];
        rows.extend_from_slice(&dids);
        let t2 = Instant::now();
        let lgs = ss.feed(e, &rows, true)?.unwrap();
        let vms = t2.elapsed().as_secs_f64() * 1e3;
        st.verify_ms += vms;
        // the cost model follows the real pass times (the start-up calibration is noisy: clocks are still ramping, other work on the box)
        if let Some(c) = cost.as_deref_mut() {
            let i = rows.len() - 1;
            if i < c.round_ms.len() {
                c.round_ms[i] = 0.9 * c.round_ms[i] + 0.1 * vms;
            }
        }
        // greedy: the drafts that match the argmax; sampled (`pick`): the lossless accept / reject of the sampler
        let (m, nxt) = match pick.as_deref_mut() {
            Some(pk) => pk(&lgs, &dids),
            None => {
                let g: Vec<u32> = lgs.iter().map(|l| argmax(l)).collect();
                let m = (0..k).take_while(|&j| dids[j] == g[j]).count();
                (m, g[m])
            }
        };
        if let Ok(path) = std::env::var("DSC_DRAFT_LOG") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let g: Vec<u32> = lgs.iter().map(|l| argmax(l)).collect();
                // top-5 of the target at the first rejected row
                let rej = if m < k { let mut idx: Vec<usize> = (0..lgs[m].len()).collect(); idx.select_nth_unstable_by(4, |&a, &b| lgs[m][b].partial_cmp(&lgs[m][a]).unwrap()); idx.truncate(5); idx.sort_by(|&a, &b| lgs[m][b].partial_cmp(&lgs[m][a]).unwrap()); idx } else { vec![] };
                let _ = writeln!(f, "{} {} copy={} conf={:?} d={:?} g={:?} top5={:?} ctx={:?}", pos, m, copy.is_some() as u8, conf_log, dids, g, rej, &e.ids_seen[e.ids_seen.len().saturating_sub(9)..e.ids_seen.len() - rows.len()]);
            }
        }
        let mut out: Vec<u32> = dids[..m].to_vec();
        out.push(nxt);
        e.truncate(pos + m + 1);
        ss.hist.truncate((pos + m + 1) * TAP);
        let t3 = Instant::now();
        let mut a = 0;
        while a < m + 1 {
            let c = (m + 1 - a).min(8);
            dr.absorb(&ss.hist[(pos + a) * TAP..], c, pos + a)?;
            a += c;
        }
        st.absorb_ms += t3.elapsed().as_secs_f64() * 1e3;
        if copy.is_some() {
            cpi.learn(cml, m, k);
            st.copy_rounds += 1;
            st.copy_tokens += out.len();
        }
        cpi.push(&out);
        last = nxt;
        st.rounds += 1;
        st.hist[m.min(15)] += 1;
        st.drafted += k;
        st.tokens += out.len();
        if !emit(&out) {
            break;
        }
    }
    st.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok((st, prefill_ms))
}

/// What follows a run the request took (see [`drive`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Next {
    /// decode on
    Go,
    /// the request is done
    Stop,
    /// the engine writes this token itself as the next one (the end of a reasoning block); the model goes on after it
    Force(u32),
    /// the driver took back the last n tokens it had accepted (none was sent on): rewind the sequence by n and decode on
    Rewind(usize),
}

/// The request's side of [`drive`]: it accepts / rejects the drafts and decides, token by token, what it keeps of a committed run.
pub trait Driver {
    /// May the next round carry drafts? false: a plain one-row step (a sampler with penalties, the first token of a reply).
    fn draft_ok(&mut self) -> bool;
    /// The accepted drafts and the token after them from the verify rows (`lgs[j]` = the logits after `drafts[..j]`): greedy takes the
    /// matching prefix and the argmax after it, a sampler its lossless accept / reject. None: stop here (the request records why).
    fn pick(&mut self, lgs: &[Vec<f32>], drafts: &[u32]) -> Option<(usize, u32)>;
    /// A committed run (the accepted drafts and the token after them, or the engine's own forced token): how many of its tokens the
    /// request keeps, and what follows. The tokens after the kept ones are given back (their rows are rolled back), so a request acts
    /// at exactly the token where it decides to, however many rows the round verified.
    fn take(&mut self, out: &[u32]) -> (usize, Next);
    /// The loop is busy without tokens for the driver (a long rewind).
    fn ping(&mut self) {}
}

/// The serving decode loop on a `Session`: `ids` = the prompt and the tokens generated so far; the engine has fed all of them but the
/// last one. Each round drafts (the DSpark drafter or a copy proposal, as `spec_stream`) when the driver allows it, verifies in one pass,
/// lets the driver pick and then take the committed run token by token; the rows of tokens the driver does not keep are rolled back
/// and the drafter absorbs the kept rows only. `force`: a token the engine writes itself right after the last one. A request whose
/// driver keeps everything decodes exactly as `spec_stream` does.
#[allow(clippy::too_many_arguments)]
pub fn drive(e: &mut Engine, dr: &mut Drafter, ss: &mut Session, ids: &[u32], kmax: usize, mut cost: Option<&mut Cost>, mut force: Option<u32>, drv: &mut dyn Driver) -> Result<SpecStats> {
    ensure!(ids.len() >= 2 && kmax >= 1 && kmax < RMAX && kmax <= 5, "drive: prompt {} kmax {kmax}", ids.len());
    ensure!(e.ids_seen.len() + 1 == ids.len(), "drive: the engine has fed {} of {} tokens", e.ids_seen.len(), ids.len());
    dr.set_back(e._ctx.clone());
    // the drafter's window is rebuilt from the host taps when the first drafting round comes (a request that never drafts skips it)
    let mut synced = false;
    let mut last = ids[ids.len() - 1];
    let mut st = SpecStats::default();
    let t0 = Instant::now();
    let mut zero_run = 0usize;
    let copy_on = cost.is_some() && std::env::var("DSC_COPY").map_or(true, |v| v != "0");
    let lmax: usize = std::env::var("DSC_COPY_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(15).min(RMAX - 1).min(15);
    let mut cpi = CopyIndex::new(ids);
    fn absorb(dr: &mut Drafter, ss: &Session, pos: usize, n: usize) -> Result<()> {
        let mut a = 0;
        while a < n {
            let c = (n - a).min(8);
            dr.absorb(&ss.hist[(pos + a) * TAP..], c, pos + a)?;
            a += c;
        }
        Ok(())
    }
    loop {
        let pos = e.ids_seen.len();
        if let Some(t) = force.take() {
            // the engine's own token after `last`: both rows in one pass; the logits after the forced token choose the next one
            let t2 = Instant::now();
            let lgs = ss.feed(e, &[last, t], true)?.unwrap();
            st.verify_ms += t2.elapsed().as_secs_f64() * 1e3;
            let (keep, next) = drv.take(&[t]);
            if keep == 0 || next != Next::Go {
                // the request ended at the forced token (its limit): `last` stays fed, the forced row is given back unless it was kept
                e.truncate(pos + 1 + keep.min(1));
                ss.hist.truncate((pos + 1 + keep.min(1)) * TAP);
                if synced {
                    absorb(dr, ss, pos, 1 + keep.min(1))?;
                }
                st.tokens += keep.min(1);
                break;
            }
            if synced {
                absorb(dr, ss, pos, 2)?;
            }
            cpi.push(&[t]);
            st.tokens += 1;
            st.rounds += 1;
            st.hist[0] += 1;
            let Some((_, nxt)) = drv.pick(&lgs[1..], &[]) else { break };
            let (keep, next) = drv.take(&[nxt]);
            if keep == 0 {
                break;
            }
            cpi.push(&[nxt]);
            st.tokens += 1;
            last = nxt;
            match next {
                Next::Go => {}
                Next::Stop => break,
                Next::Force(f) => force = Some(f),
                Next::Rewind(_) => {}
            }
            continue;
        }
        let drafting = drv.draft_ok();
        if drafting && !synced {
            ss.sync_drafter(dr, pos)?;
            synced = true;
        }
        // copy proposal from the context (long verbatim matches skip the drafter altogether)
        let mut copy: Option<Vec<u32>> = None;
        let mut d: Option<super::draft::DraftOut> = None;
        let mut cml = 0usize;
        if drafting && copy_on {
            if let (Some((prop, ml)), Some(c)) = (cpi.propose(lmax), cost.as_deref()) {
                if ml >= 4 && !prop.is_empty() {
                    cml = ml;
                    let sv = cpi.surv(ml);
                    let (mut bl, mut bv, mut et) = (0usize, -1.0f64, 1.0f64);
                    let mut sp = 1.0;
                    for l in 1..=prop.len() {
                        sp *= sv;
                        et += sp;
                        let v = et / c.at(l + 1);
                        if v > bv {
                            bl = l;
                            bv = v;
                        }
                    }
                    let take = if ml >= 16 {
                        true
                    } else {
                        let t1 = Instant::now();
                        let dd = dr.draft(last, pos, kmax)?;
                        st.draft_ms += t1.elapsed().as_secs_f64() * 1e3;
                        let (_, vd) = choose_k_v(&dd.conf, kmax, c);
                        d = Some(dd);
                        bv > vd * 1.2
                    };
                    if take {
                        copy = Some(prop[..bl].to_vec());
                    }
                }
            }
        }
        let mut conf_log: Vec<f32> = vec![];
        let (dids, k): (Vec<u32>, usize) = match &copy {
            Some(c) => (c.clone(), c.len()),
            None if !drafting => (vec![], 0),
            None => {
                let dd = match d.take() {
                    Some(dd) => dd,
                    None => {
                        let t1 = Instant::now();
                        let dd = dr.draft(last, pos, kmax)?;
                        st.draft_ms += t1.elapsed().as_secs_f64() * 1e3;
                        dd
                    }
                };
                let mut k = match cost.as_deref() {
                    Some(c) => choose_k(&dd.conf, kmax, c),
                    None => kmax,
                };
                // the cost model only learns the rows it sees: after a long run of plain steps try a 3-row pass again
                zero_run = if k == 0 { zero_run + 1 } else { 0 };
                if zero_run >= 24 && kmax >= 2 {
                    k = 2;
                    zero_run = 0;
                }
                conf_log = dd.conf.clone();
                (dd.ids[..k].to_vec(), k)
            }
        };
        let mut rows = vec![last];
        rows.extend_from_slice(&dids);
        let t2 = Instant::now();
        let lgs = ss.feed(e, &rows, true)?.unwrap();
        let vms = t2.elapsed().as_secs_f64() * 1e3;
        st.verify_ms += vms;
        // the cost model follows the real pass times (the start-up calibration is noisy: clocks are still ramping, other work on the box)
        if let Some(c) = cost.as_deref_mut() {
            let i = rows.len() - 1;
            if i < c.round_ms.len() {
                c.round_ms[i] = 0.9 * c.round_ms[i] + 0.1 * vms;
            }
        }
        let Some((m, nxt)) = drv.pick(&lgs, &dids) else {
            e.truncate(pos + 1);
            ss.hist.truncate((pos + 1) * TAP);
            break;
        };
        if let Ok(path) = std::env::var("DSC_DRAFT_LOG") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let g: Vec<u32> = lgs.iter().map(|l| argmax(l)).collect();
                let _ = writeln!(f, "{} {} copy={} conf={:?} d={:?} g={:?}", pos, m, copy.is_some() as u8, conf_log, dids, g);
            }
        }
        let mut out: Vec<u32> = dids[..m].to_vec();
        out.push(nxt);
        let (keep, next) = drv.take(&out);
        // row 0 (the last token) stays; rows 1.. are the kept drafts; the last kept token is not fed yet
        let rows_kept = keep.clamp(1, m + 1);
        e.truncate(pos + rows_kept);
        ss.hist.truncate((pos + rows_kept) * TAP);
        let t3 = Instant::now();
        if synced {
            absorb(dr, ss, pos, rows_kept)?;
        }
        st.absorb_ms += t3.elapsed().as_secs_f64() * 1e3;
        if copy.is_some() {
            cpi.learn(cml, m, k);
            st.copy_rounds += 1;
            st.copy_tokens += keep;
        }
        cpi.push(&out[..keep]);
        st.rounds += 1;
        st.hist[m.min(15)] += 1;
        st.drafted += k;
        st.tokens += keep;
        if keep == 0 {
            break;
        }
        last = out[keep - 1];
        match next {
            Next::Go => {}
            Next::Stop => break,
            Next::Force(t) => force = Some(t),
            Next::Rewind(nb) => {
                // the sequence so far is the fed positions plus `last`: drop its last nb tokens (never into the prompt)
                let total = e.ids_seen.len() + 1;
                let keep_len = total.saturating_sub(nb).max(ids.len());
                let mut seq: Vec<u32> = e.ids_seen.clone();
                seq.push(last);
                seq.truncate(keep_len);
                if e.ids_seen.len() - (keep_len - 1) <= e.max_rewind() {
                    e.truncate(keep_len - 1);
                    ss.hist.truncate((keep_len - 1) * TAP);
                } else {
                    // past the window ring: the kept sequence is fed again (the prompt path), the request kept alive meanwhile
                    e.reset()?;
                    ss.hist.clear();
                    let np = keep_len - 1;
                    let mut i = 0;
                    while i < np {
                        let c = e.chunk_for(np - i);
                        ss.feed(e, &seq[i..i + c], false)?;
                        i += c;
                        drv.ping();
                    }
                }
                last = seq[keep_len - 1];
                cpi = CopyIndex::new(&seq);
                synced = false;
            }
        }
    }
    st.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok(st)
}

/// Measure the verify pass by rows (1..=6) on the live sequence (rows are rolled back after) and one drafter pass.
pub fn calibrate(e: &mut Engine, dr: &mut Drafter, ids: &[u32]) -> Result<Cost> {
    dr.set_back(e._ctx.clone());
    let mut host = vec![0u16; RMAX * TAP];
    prefill(e, dr, ids, &mut host)?;
    let np = ids.len() - 1;
    let last = ids[np];
    let mut round_ms = vec![];
    for r in 1..=16usize {
        let rows = vec![last; r];
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            e.step_rows(&rows, true)?;
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            e.truncate(np);
        }
        round_ms.push(best);
    }
    let mut dms = f64::MAX;
    for _ in 0..3 {
        dms = dms.min(dr.draft(last, np, 5)?.ms as f64);
    }
    e.reset()?;
    dr.reset();
    Ok(Cost { round_ms, draft_ms: dms })
}
