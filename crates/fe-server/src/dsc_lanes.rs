//! `--engine dsc` with DSC_LANES > 1: several requests decode at once. Every lane is a sequence slot of the engine (all caches are [slots][...]
//! and the row metadata carries (slot, position)), so one pass carries the rows of every lane: the dense and expert weights are read once for
//! all of them, each row attends to its own lane's caches (`Engine::step_multi`; a lane's logits equal the same request run alone).
//!
//! Scheduling, one pass at a time:
//! - arrivals are drained (a short window after an idle start, so near-simultaneous requests join the same pass) and given lanes: the free lane
//!   with the longest cached prefix; a prefix shared with another lane is forked from it (device copy of the slot) once that lane is at the
//!   share point, so a prompt prefix common to several requests is prefilled once (the leader stops at the share point; followers wait);
//! - decoding lanes bring their verify rows (last token + drafts: the drafter's or copied from the context); prefilling lanes fill the
//!   remaining row budget in small chunks, so a long prompt never stalls the others for more than one pass; a lone prefilling lane (nobody
//!   decoding) takes the big-chunk path;
//! - the draft budget per lane follows the lanes' count (rows total <= 32).
//!
//! Protections: every lane carries the request state of the single-stream path (crate::dsc::RunState: the sampling of the phase, the
//! guard with the reasoning cap, the loop breaker and the repeated-sentence rule, the first-token stop ban, the finite check, the
//! tool-call recovery, the constrained opening). A lane acts at exactly the token where its guard answers: the rows a pass verified
//! past that token are rolled back for that lane only (`truncate_lane`), a forced `</think>` rides in the lane's next pass as a second
//! row, a rewind truncates the lane (or feeds the kept sequence again when it is past the engine's window).
use std::collections::BTreeSet;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::Result;
use fe_dsc::engine::{argmax, draft::Drafter, spec, spec::Next, spec::Session, spec::SpecRun, Engine as DscCore};

use crate::dsc::RunState;
use crate::engine::{Finish, FinishReason, GenRequest, Timings, TokenSink};

#[derive(PartialEq)]
enum Phase {
    /// feeding `feed` (positions fed .. feed.len() - 1)
    Prefill,
    /// waiting for lane `.0` to reach the share point `.1`, then forked from it
    WaitFork(usize, usize),
    /// the prompt's last token is fed next (the first generated token; the tokens of a constrained opening one pass each), then rounds
    First,
    Rounds,
}

/// What a lane's rows in a pass are.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Prefill,
    First,
    /// [last, the engine's forced token]
    Force,
    /// [last] + drafts (none: a plain step)
    Round,
}

struct Lane {
    req: GenRequest,
    sink: TokenSink,
    /// sampling of the phase, protections, what the client got (as the single-stream path)
    st: RunState,
    ss: Session,
    prompt: Vec<u32>,
    /// what the lane feeds before it decodes: the prompt, or (after a rewind past the engine's window) the kept sequence
    feed: Vec<u32>,
    /// `feed` is a kept sequence: rounds go on after it (no first-token pass)
    resume: bool,
    cached: usize,
    phase: Phase,
    reason: Option<FinishReason>,
    t0: Instant,
    t_first: f64,
    t_gen: Instant,
    sp: Option<SpecRun>,
    /// the drafter's window holds this lane's rows (rebuilt from the taps at the first drafting round and after a rewind)
    synced: bool,
    /// the engine writes this token itself after the last one (the end of the reasoning block)
    force: Option<u32>,
    /// constrained opening (tool_choice required with several functions): the sequences still possible, tokens chosen so far
    cands: Vec<Vec<u32>>,
    depth: usize,
    open: Vec<u32>,
    /// prompt + opening + first token: a rewind never goes below it
    base: usize,
    /// ascending prompt positions at which followers are forked from this lane
    stops: BTreeSet<usize>,
    /// draft rows planned for this pass
    plan: Vec<u32>,
    drafted: u64,
    accepted: u64,
}

impl Lane {
    fn end(&mut self) {
        self.reason = Some(self.st.fin.clone().unwrap_or(FinishReason::Cancelled));
    }

    /// `feed` is in the caches (all but its last token): what comes next
    fn fed(&mut self) {
        if self.resume {
            self.resume = false;
            self.phase = Phase::Rounds;
        } else {
            self.phase = Phase::First;
        }
    }
}

/// What a lane keeps after its request finished: the tokens its caches hold (the prefix cache), the taps history.
struct Idle {
    ss: Session,
}

enum Slot {
    Free(Option<Idle>),
    Busy(Box<Lane>),
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The drafter takes the taps of `n` kept rows from position `pos` (its lane active).
fn absorb(dr: &mut Drafter, ss: &Session, pos: usize, n: usize) -> Result<()> {
    let mut a = 0;
    while a < n {
        let c = (n - a).min(8);
        dr.absorb(&ss.hist[(pos + a) * spec::TAP..], c, pos + a)?;
        a += c;
    }
    Ok(())
}

pub fn run_lanes(e: &mut DscCore, dr: &mut Drafter, cost0: Option<spec::Cost>, rx: mpsc::Receiver<(GenRequest, TokenSink)>, kmax: usize, use_spec: bool, ctx: usize) -> Result<()> {
    let nl = e.n_lanes();
    dr.alloc_lanes(nl)?;
    tracing::info!("dsc lanes: {nl} lanes, the protections of the single-stream path apply per lane");
    // the calibrated verify cost (code, resident experts) scaled by the observed ratio of real passes to it: lanes' rows cost more (cold experts, other texts)
    let mut scale = 1.0f64;
    let mut cost: Option<spec::Cost> = cost0.clone();
    let mut cost1: Option<spec::Cost> = cost0.clone();
    let mut slots: Vec<Slot> = (0..nl).map(|_| Slot::Free(None)).collect();
    let copy_on = std::env::var("DSC_COPY").map_or(true, |v| v != "0");
    let arrival = Duration::from_millis(std::env::var("DSC_ARRIVAL_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(25));
    let row_budget: usize = std::env::var("DSC_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(32).clamp(4, 32);
    let pf_min_rows: usize = 4;
    // rows of one pass a prefilling lane may take while others decode (the rest is the decode lanes' guaranteed share)
    let pf_cap: usize = std::env::var("DSC_PF_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(row_budget / 4).max(pf_min_rows);
    let mut pending: Vec<(GenRequest, TokenSink)> = vec![];
    let mut host_taps: Vec<u16> = vec![0u16; 32 * spec::TAP];
    let mut passes = 0u64;
    let mut rr_pf = 0usize;
    // DSC_DRAFT_BATCH=0: a drafter pass a lane (the old way); DSC_DRAFT_CHECK=1: every batched draft is compared with the lane's own draft
    let batch_draft = std::env::var("DSC_DRAFT_BATCH").map_or(true, |v| v != "0");
    let draft_check = std::env::var("DSC_DRAFT_CHECK").is_ok_and(|v| v == "1");
    let mut draft_same = 0u64;
    let mut draft_ema = vec![0f64; nl + 1];
    let (mut t_plan, mut t_pass, mut t_post, mut t_rows) = (0f64, 0f64, 0f64, 0usize);
    loop {
        let busy = slots.iter().any(|s| matches!(s, Slot::Busy(_)));
        // ---- arrivals
        if !busy && pending.is_empty() {
            match rx.recv() {
                Ok(r) => pending.push(r),
                Err(_) => return Ok(()),
            }
            std::thread::sleep(arrival);
        }
        while let Ok(r) = rx.try_recv() {
            pending.push(r);
        }
        // admit what fits
        let mut i = 0;
        while i < pending.len() {
            if pending[i].0.cancel.is_cancelled() || pending[i].1.is_closed() {
                let (_, sink) = pending.remove(i);
                sink.finish(Finish::new(FinishReason::Cancelled));
                continue;
            }
            let free = slots.iter().position(|s| matches!(s, Slot::Free(_)));
            let Some(_) = free else { break };
            let (req, sink) = pending.remove(i);
            admit(e, &mut slots, req, sink, use_spec, ctx, kmax)?;
        }
        // ---- fork followers whose leader has reached the share point
        for f in 0..nl {
            let (ld, at) = match &slots[f] {
                Slot::Busy(l) => match l.phase {
                    Phase::WaitFork(ld, at) => (ld, at),
                    _ => continue,
                },
                _ => continue,
            };
            let leader_alive = matches!(slots[ld], Slot::Busy(_));
            let have = e.lane_len(ld);
            if leader_alive && have < at {
                continue;
            }
            // the leader's lane must still hold this prefix (it may have been given another request)
            let same = match (&slots[f], &slots[ld]) {
                (Slot::Busy(fl), Slot::Busy(ll)) => lcp(&ll.prompt, &fl.prompt) >= at,
                (Slot::Busy(fl), Slot::Free(_)) => lcp(e.lane_tokens(ld), &fl.prompt) >= at,
                _ => false,
            };
            let hist = match &slots[ld] {
                Slot::Busy(l) => l.ss.hist.clone(),
                Slot::Free(Some(i)) => i.ss.hist.clone(),
                Slot::Free(None) => vec![],
            };
            if same && have >= at && have - at <= e.max_rewind() && hist.len() >= at * spec::TAP {
                e.fork_lane(ld, f, at)?;
                if let Slot::Busy(l) = &mut slots[f] {
                    l.ss.hist = hist[..at * spec::TAP].to_vec();
                    l.cached = at;
                    l.phase = Phase::Prefill;
                }
            } else if let Slot::Busy(l) = &mut slots[f] {
                // the leader is gone or moved on: prefill alone
                l.phase = Phase::Prefill;
                e.activate(f);
                e.truncate(0);
                l.ss.hist.clear();
                l.cached = 0;
            }
            if let Slot::Busy(l) = &mut slots[ld] {
                l.stops.remove(&at);
            }
        }
        // ---- cancelled lanes
        for s in slots.iter_mut() {
            if let Slot::Busy(l) = s {
                if l.req.cancel.is_cancelled() || l.sink.is_closed() {
                    l.reason = Some(FinishReason::Cancelled);
                }
            }
        }
        for li in 0..nl {
            if matches!(&slots[li], Slot::Busy(l) if l.reason.is_some()) {
                finish_lane(e, &mut slots, li);
            }
        }
        // ---- plan the pass
        let dec: Vec<usize> = (0..nl).filter(|&i| matches!(&slots[i], Slot::Busy(l) if matches!(l.phase, Phase::First | Phase::Rounds))).collect();
        let pre: Vec<usize> = (0..nl).filter(|&i| matches!(&slots[i], Slot::Busy(l) if l.phase == Phase::Prefill)).collect();
        if dec.is_empty() && pre.is_empty() {
            // only followers waiting for a leader that is not there, or nothing
            continue;
        }
        // a lone prefilling lane (nobody decoding): the big-chunk path
        if dec.is_empty() && !pre.is_empty() {
            // nobody decoding: one lane's big chunk at a time (lanes take turns), never small mixed passes
            rr_pf = rr_pf.wrapping_add(1);
            let li = pre[rr_pf % pre.len()];
            if let Slot::Busy(l) = &mut slots[li] {
                e.activate(li);
                let fed = e.ids_seen.len();
                let end = l.feed.len() - 1;
                if fed < end {
                    let mut left = end - fed;
                    if let Some(&stop) = l.stops.range(fed + 1..).next() {
                        left = left.min(stop - fed);
                    }
                    // chunks of at most 1024 rows keep the host responsive (cancel, arrivals) between passes
                    let c = e.chunk_for(left).max(1).min(left);
                    let toks = l.feed[fed..fed + c].to_vec();
                    l.ss.feed(e, &toks, false)?;
                    if l.resume {
                        let _ = l.sink.ping();
                    }
                }
                if e.ids_seen.len() >= end {
                    l.fed();
                }
            }
            continue;
        }
        if let (Some(b), Some(c)) = (cost0.as_ref(), cost.as_mut()) {
            // the one-row cost is the fixed part (weights); the rows beyond it cost `scale` times the calibrated increment
            let b1 = b.round_ms[0];
            for (x, y) in c.round_ms.iter_mut().zip(&b.round_ms) {
                *x = b1 + (*y - b1) * scale;
            }
        }
        // decode rows: every lane that may draft drafts (the candidates), then the pass's draft rows go to the drafts with the best marginal gain per ms
        let t_pl = Instant::now();
        let mut drafting = vec![false; nl];
        let mut base_rows = 0usize;
        for &li in &dec {
            if let Slot::Busy(l) = &mut slots[li] {
                let l: &mut Lane = &mut **l;
                drafting[li] = l.phase == Phase::Rounds && l.force.is_none() && l.st.draft_ok(&l.req);
                base_rows += if l.phase == Phase::Rounds && l.force.is_some() { 2 } else { 1 };
            }
        }
        let single = dec.len() == 1 && pre.is_empty() && drafting[dec[0]];
        let mut cands: Vec<(usize, Option<spec::Cand>)> = vec![];
        let n_pf = if pre.is_empty() { 0 } else { pf_cap };
        // the lanes that need the drafter this round (a long verbatim copy proposal rides alone), their windows in step
        let mut want: Vec<(usize, u32, usize)> = vec![];
        let mut kd = 1usize;
        for &li in &dec {
            if let Slot::Busy(l) = &mut slots[li] {
                let l: &mut Lane = &mut **l;
                if !drafting[li] {
                    continue;
                }
                let pos = e.lane_len(li);
                if !l.synced {
                    // the drafter's window from the taps (positions fed so far)
                    dr.activate(li);
                    l.ss.sync_drafter(dr, pos)?;
                    l.synced = true;
                }
                let sp = l.sp.as_ref().unwrap();
                kd = sp.draft_k();
                if !single && sp.wants_draft(cost.as_ref(), 15) {
                    want.push((li, sp.last, pos));
                }
            }
        }
        // one drafter pass for all of them
        let mut drafts: Vec<Option<fe_dsc::engine::draft::DraftOut>> = (0..nl).map(|_| None).collect();
        let mut share_ms = 0f64;
        if !want.is_empty() {
            let t_d = Instant::now();
            let outs = if batch_draft {
                dr.draft_multi(&want, kd)?
            } else {
                let mut o = vec![];
                for &(li, last, pos) in &want {
                    dr.activate(li);
                    o.push(dr.draft(last, pos, kd)?);
                }
                o
            };
            let ms = t_d.elapsed().as_secs_f64() * 1e3;
            if draft_check {
                for (&(li, last, pos), o) in want.iter().zip(&outs) {
                    dr.activate(li);
                    let one = dr.draft(last, pos, kd)?;
                    if one.ids != o.ids || one.conf.iter().zip(&o.conf).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        tracing::error!("dsc lanes: batched draft differs from the lane's own draft (lane {li}, pos {pos}): {:?} {:?} vs {:?} {:?}", o.ids, o.conf, one.ids, one.conf);
                    } else {
                        draft_same += 1;
                    }
                }
            }
            draft_ema[want.len()] = if draft_ema[want.len()] == 0.0 { ms } else { 0.9 * draft_ema[want.len()] + 0.1 * ms };
            share_ms = ms / want.len() as f64;
            for (&(li, _, _), o) in want.iter().zip(outs) {
                drafts[li] = Some(o);
            }
        }
        for &li in &dec {
            if let Slot::Busy(l) = &mut slots[li] {
                let l: &mut Lane = &mut **l;
                if !drafting[li] {
                    cands.push((li, None));
                    continue;
                }
                let sp = l.sp.as_mut().unwrap();
                if single {
                    // one decoding lane: exactly the single-stream policy (its own cost model with per-entry corrections)
                    dr.activate(li);
                    let pos = e.lane_len(li);
                    let d = sp.plan(dr, pos, cost1.as_ref(), 15)?.to_vec();
                    cands.push((li, Some(spec::Cand { ids: d.clone(), surv: vec![1.0; d.len()], copy: false, cml: 0 })));
                } else {
                    let dd = drafts[li].take();
                    let ms = if dd.is_some() { share_ms } else { 0.0 };
                    cands.push((li, Some(sp.candidates_from(dd, ms, cost.as_ref(), 15)?)));
                }
            }
        }
        // greedy allocation by marginal expected tokens per ms
        let mut ks = vec![0usize; cands.len()];
        let mut r_tot = base_rows + n_pf;
        if single {
            ks[0] = cands[0].1.as_ref().map_or(0, |c| c.ids.len());
        } else if let Some(c) = cost.as_ref() {
            // the drafter's time this round (one pass for all lanes: measured, by lane count)
            let dms = if want.is_empty() { 0.0 } else if draft_ema[want.len()] > 0.0 { draft_ema[want.len()] } else { c.draft_ms };
            let mut tok = dec.len() as f64;
            loop {
                let mut bi: Option<(usize, f64)> = None;
                for (i, (_, cd)) in cands.iter().enumerate() {
                    if let Some(cd) = cd {
                        if ks[i] < cd.ids.len() && bi.map_or(true, |(_, g)| cd.surv[ks[i]] > g) {
                            bi = Some((i, cd.surv[ks[i]]));
                        }
                    }
                }
                let Some((i, g)) = bi else { break };
                if r_tot + 1 > row_budget {
                    break;
                }
                if (tok + g) / (c.at(r_tot + 1) + dms) <= tok / (c.at(r_tot) + dms) {
                    break;
                }
                ks[i] += 1;
                r_tot += 1;
                tok += g;
            }
        }
        let mut parts_owned: Vec<(usize, Vec<u32>)> = vec![];
        let mut row_of: Vec<(usize, usize, usize, Kind)> = vec![]; // lane, first row, rows, what they are
        let mut rows = 0usize;
        for (ci, (li, cd)) in cands.iter().enumerate() {
            let li = *li;
            if let Slot::Busy(l) = &mut slots[li] {
                let l: &mut Lane = &mut **l;
                let mut toks = vec![];
                let kind;
                if l.phase == Phase::First {
                    toks.push(*l.open.last().unwrap_or(l.prompt.last().unwrap()));
                    l.plan.clear();
                    kind = Kind::First;
                } else if let Some(t) = l.force {
                    toks.push(l.sp.as_ref().unwrap().last);
                    toks.push(t);
                    l.plan.clear();
                    kind = Kind::Force;
                } else {
                    let sp = l.sp.as_mut().unwrap();
                    toks.push(sp.last);
                    match cd {
                        Some(cd) => {
                            if !single {
                                sp.set_plan(cd, ks[ci]);
                            }
                            l.plan = cd.ids[..ks[ci]].to_vec();
                        }
                        None => {
                            sp.clear_plan();
                            l.plan.clear();
                        }
                    }
                    toks.extend_from_slice(&l.plan);
                    kind = Kind::Round;
                }
                row_of.push((li, rows, toks.len(), kind));
                rows += toks.len();
                parts_owned.push((li, toks));
            }
        }
        // prefill rows
        if !pre.is_empty() {
            let mut left = row_budget.saturating_sub(rows).max(pf_min_rows);
            if !dec.is_empty() {
                left = left.min(pf_cap);
            }
            let per = (left / pre.len()).max(1);
            for &li in &pre {
                if let Slot::Busy(l) = &mut slots[li] {
                    let fed = e.lane_len(li);
                    let end = l.feed.len() - 1;
                    let mut n = (end - fed).min(per).min(left);
                    if let Some(&stop) = l.stops.range(fed + 1..).next() {
                        n = n.min(stop - fed);
                    }
                    if n == 0 {
                        if fed >= end {
                            l.fed();
                        }
                        continue;
                    }
                    let toks = l.feed[fed..fed + n].to_vec();
                    row_of.push((li, rows, n, Kind::Prefill));
                    rows += n;
                    left = left.saturating_sub(n);
                    parts_owned.push((li, toks));
                }
            }
        }
        if rows == 0 {
            continue;
        }
        t_plan += t_pl.elapsed().as_secs_f64() * 1e3;
        let want = !dec.is_empty();
        let parts: Vec<(usize, &[u32])> = parts_owned.iter().map(|(l, t)| (*l, &t[..])).collect();
        let t_pass0 = Instant::now();
        let lgs = e.step_multi(&parts, want)?;
        let vms = t_pass0.elapsed().as_secs_f64() * 1e3;
        passes += 1;
        t_pass += vms;
        t_rows += rows;
        let t_po = Instant::now();
        if single {
            if let Some(c) = cost1.as_mut() {
                let i = rows - 1;
                if i < c.round_ms.len() {
                    c.round_ms[i] = 0.9 * c.round_ms[i] + 0.1 * vms;
                }
            }
        } else if let Some(b) = cost0.as_ref() {
            if !dec.is_empty() && rows >= 2 {
                let b1 = b.round_ms[0];
                let inc = (b.at(rows) - b1).max(1.0);
                scale = (0.93 * scale + 0.07 * (vms - b1) / inc).clamp(0.6, 5.0);
            }
        }
        // taps of the rows (one device read), split to the lanes' histories
        e.taps_to_host(rows, &mut host_taps)?;
        for &(li, r0, n, _) in &row_of {
            if let Slot::Busy(l) = &mut slots[li] {
                l.ss.hist.extend_from_slice(&host_taps[r0 * spec::TAP..(r0 + n) * spec::TAP]);
            }
        }
        // ---- results
        let lgs = lgs.unwrap_or_default();
        for &(li, r0, n, kind) in &row_of {
            let Slot::Busy(l) = &mut slots[li] else { continue };
            let l: &mut Lane = &mut **l;
            if kind == Kind::Prefill {
                if l.resume {
                    let _ = l.sink.ping();
                }
                if e.lane_len(li) >= l.feed.len() - 1 {
                    l.fed();
                }
                continue;
            }
            let rows_lg = &lgs[r0..r0 + n];
            // position of the lane's first row of this pass
            let pos = e.lane_len(li) - n;
            match kind {
                Kind::Prefill => {}
                Kind::First => {
                    let lg = &rows_lg[0];
                    // constrained opening: the reply starts with one of the offered sequences, chosen token by token from the model's
                    // logits among the candidates still possible; each chosen token is one more plain row
                    if !l.cands.is_empty() && !l.cands.iter().any(|c| c.len() == l.depth) {
                        let mut masked = vec![f32::NEG_INFINITY; lg.len()];
                        for c in &l.cands {
                            let t = c[l.depth] as usize;
                            if t < lg.len() {
                                masked[t] = lg[t];
                            }
                        }
                        let t = if l.st.greedy { argmax(&masked) } else { l.st.sampler.sample(&mut masked) };
                        let d = l.depth;
                        l.cands.retain(|c| c[d] == t);
                        if !l.cands.is_empty() {
                            l.depth += 1;
                            if l.depth == 1 {
                                l.t_first = l.t0.elapsed().as_secs_f64() * 1e3;
                                l.t_gen = Instant::now();
                            }
                            let (keep, nx) = l.st.take(&l.req, &l.sink, &[t]);
                            if keep == 0 || nx != Next::Go {
                                l.end();
                            } else {
                                l.open.push(t);
                            }
                            continue;
                        }
                    }
                    // the first free token: a plain one-row pick (the stop ban applies to the first token of a reply)
                    let Some(t1) = l.st.pick1(&l.req, lg) else {
                        l.end();
                        continue;
                    };
                    if l.depth == 0 {
                        l.t_first = l.t0.elapsed().as_secs_f64() * 1e3;
                        l.t_gen = Instant::now();
                    }
                    let (keep, nx) = l.st.take(&l.req, &l.sink, &[t1]);
                    if keep != 1 {
                        l.end();
                        continue;
                    }
                    match nx {
                        Next::Go => {}
                        Next::Force(f) => l.force = Some(f),
                        _ => {
                            l.end();
                            continue;
                        }
                    }
                    let mut ids2 = l.prompt.clone();
                    ids2.extend_from_slice(&l.open);
                    ids2.push(t1);
                    l.base = ids2.len();
                    l.sp = Some(SpecRun::new(&ids2, kmax, copy_on));
                    l.synced = false;
                    l.phase = Phase::Rounds;
                }
                Kind::Force => {
                    // the engine's own token after `last`: both rows rode in this pass; the logits after the forced token choose the next one
                    let t = l.force.take().unwrap();
                    let (keep, next) = l.st.take(&l.req, &l.sink, &[t]);
                    if keep == 0 || next != Next::Go {
                        // the request ended at the forced token (its limit): `last` stays fed, the forced row is given back unless it was kept
                        let kk = keep.min(1);
                        e.truncate_lane(li, pos + 1 + kk);
                        l.ss.hist.truncate((pos + 1 + kk) * spec::TAP);
                        l.end();
                        continue;
                    }
                    if l.synced {
                        dr.activate(li);
                        absorb(dr, &l.ss, pos, 2)?;
                    }
                    l.sp.as_mut().unwrap().push_tokens(&[t]);
                    let Some((_, nxt)) = l.st.pick(&l.req, &rows_lg[1..], &[]) else {
                        l.end();
                        continue;
                    };
                    let (keep, next) = l.st.take(&l.req, &l.sink, &[nxt]);
                    if keep == 0 {
                        l.end();
                        continue;
                    }
                    l.sp.as_mut().unwrap().push_tokens(&[nxt]);
                    match next {
                        Next::Go | Next::Rewind(_) => {}
                        Next::Stop => l.end(),
                        Next::Force(f) => l.force = Some(f),
                    }
                }
                Kind::Round => {
                    let k = l.plan.len();
                    let Some((m, nxt)) = l.st.pick(&l.req, rows_lg, &l.plan) else {
                        e.truncate_lane(li, pos + 1);
                        l.ss.hist.truncate((pos + 1) * spec::TAP);
                        l.end();
                        continue;
                    };
                    let mut out: Vec<u32> = l.plan[..m].to_vec();
                    out.push(nxt);
                    let (keep, next) = l.st.take(&l.req, &l.sink, &out);
                    // row 0 (the last token) stays; rows 1.. are the kept drafts; the last kept token is not fed yet
                    let rows_kept = keep.clamp(1, m + 1);
                    e.truncate_lane(li, pos + rows_kept);
                    l.ss.hist.truncate((pos + rows_kept) * spec::TAP);
                    if l.synced {
                        dr.activate(li);
                        absorb(dr, &l.ss, pos, rows_kept)?;
                    }
                    let sp = l.sp.as_mut().unwrap();
                    sp.commit_kept(m, &out[..keep]);
                    l.drafted += k as u64;
                    l.accepted += m.min(keep) as u64;
                    if keep == 0 {
                        l.end();
                        continue;
                    }
                    match next {
                        Next::Go => {}
                        Next::Stop => l.end(),
                        Next::Force(t) => l.force = Some(t),
                        Next::Rewind(nb) => {
                            // the sequence so far is the fed positions plus `last`: drop its last nb tokens (never into the prompt)
                            let have = e.lane_len(li);
                            let keep_len = (have + 1).saturating_sub(nb).max(l.base);
                            let mut seq: Vec<u32> = e.lane_tokens(li).to_vec();
                            seq.push(sp.last);
                            seq.truncate(keep_len);
                            if have - (keep_len - 1) <= e.max_rewind() {
                                e.truncate_lane(li, keep_len - 1);
                                l.ss.hist.truncate((keep_len - 1) * spec::TAP);
                            } else {
                                // past the window ring: the kept sequence is fed again (the prompt path), then rounds go on
                                e.activate(li);
                                e.truncate(0);
                                l.ss.hist.clear();
                                l.feed = seq.clone();
                                l.resume = true;
                                l.phase = Phase::Prefill;
                            }
                            sp.reseed(&seq);
                            l.synced = false;
                        }
                    }
                }
            }
        }
        for li in 0..nl {
            if matches!(&slots[li], Slot::Busy(l) if l.reason.is_some()) {
                finish_lane(e, &mut slots, li);
            }
        }
        t_post += t_po.elapsed().as_secs_f64() * 1e3;
        if passes % 100 == 0 {
            let n = 100.0;
            let h = std::mem::take(&mut e.zc_hist);
            let cold: u64 = h.iter().map(|x| x.0 as u64).sum();
            tracing::info!("dsc lanes: 100 passes: {:.1} rows, plan (drafts) {:.1} ms, pass {:.1} ms, post (taps, absorb, emit) {:.1} ms, cold pairs a pass {:.1}", t_rows as f64 / n, t_plan / n, t_pass / n, t_post / n, cold as f64 / n);
            if draft_check {
                tracing::info!("dsc lanes: draft check: {draft_same} batched drafts equal to the lane's own so far");
            }
            t_plan = 0.0; t_pass = 0.0; t_post = 0.0; t_rows = 0;
        }
    }
}

/// Give request `req` a lane (a free one: the longest cached prefix of its own, or a fork of another lane's prefix).
fn admit(e: &mut DscCore, slots: &mut Vec<Slot>, req: GenRequest, sink: TokenSink, use_spec: bool, ctx: usize, _kmax: usize) -> Result<()> {
    let nl = slots.len();
    let ids = req.prompt_ids.clone();
    if ids.is_empty() || ids.len() >= ctx {
        sink.finish(Finish::new(FinishReason::Error(format!("prompt of {} tokens does not fit the {ctx}-token context", ids.len()))));
        return Ok(());
    }
    let mut p = req.params.clone();
    p.max_tokens = p.max_tokens.min(ctx - ids.len());
    let req = GenRequest { params: p.clone(), ..req };
    // the free lane whose cache shares the longest prefix; else the first free one
    let mut best: Option<(usize, usize)> = None;
    for i in 0..nl {
        if let Slot::Free(idle) = &slots[i] {
            let have = idle.as_ref().map_or(0, |x| x.ss.hist.len() / spec::TAP).min(e.lane_len(i));
            let l = lcp(e.lane_tokens(i), &ids).min(ids.len() - 1).min(have);
            if best.map_or(true, |(_, bl)| l > bl) {
                best = Some((i, l));
            }
        }
    }
    let (lane, own) = best.unwrap();
    let mut ss = match std::mem::replace(&mut slots[lane], Slot::Free(None)) {
        Slot::Free(Some(i)) => i.ss,
        _ => Session::default(),
    };
    e.activate(lane);
    // own cache prefix
    let mut start = 0usize;
    if own > 0 && e.lane_len(lane) - own <= e.max_rewind() {
        e.truncate(own);
        ss.hist.truncate(own * spec::TAP);
        start = own;
    } else {
        e.truncate(0);
        ss.hist.clear();
    }
    // a prefix shared with another lane: fork it (if that lane is at the share point) or wait for it
    let mut phase = Phase::Prefill;
    let mut leader_stop: Option<(usize, usize)> = None;
    if ids.len() > 64 {
        let mut bestf: Option<(usize, usize)> = None;
        for o in 0..nl {
            if o == lane {
                continue;
            }
            let target: Vec<u32> = match &slots[o] {
                Slot::Busy(l) => l.prompt.clone(),
                Slot::Free(_) => e.lane_tokens(o).to_vec(),
            };
            let mut share = lcp(&target, &ids).min(ids.len() - 1);
            // on the big-chunk path the prompt is cut at multiples of its chunk: fork only at such a boundary, so the follower's state is the one a solo run has
            let bc = e.pf_chunk();
            if bc > 32 && std::env::var("DSC_FORK_EXACT").map_or(false, |v| v == "1") {
                share = share / bc * bc;
            }
            if share >= 64 && share > start + 32 && bestf.map_or(true, |(_, s)| share > s) {
                bestf = Some((o, share));
            }
        }
        if let Some((o, share)) = bestf {
            let have = e.lane_len(o);
            let hist_ok = match &slots[o] {
                Slot::Busy(l) => l.ss.hist.len() / spec::TAP >= share.min(have),
                Slot::Free(Some(i)) => i.ss.hist.len() / spec::TAP >= share.min(have),
                _ => false,
            };
            if have >= share && have - share <= e.max_rewind() && hist_ok {
                e.fork_lane(o, lane, share)?;
                ss.hist = match &slots[o] {
                    Slot::Busy(l) => l.ss.hist[..share * spec::TAP].to_vec(),
                    Slot::Free(Some(i)) => i.ss.hist[..share * spec::TAP].to_vec(),
                    _ => vec![],
                };
                start = share;
            } else if have < share && matches!(&slots[o], Slot::Busy(l) if l.phase == Phase::Prefill || matches!(l.phase, Phase::WaitFork(..))) {
                phase = Phase::WaitFork(o, share);
                leader_stop = Some((o, share));
            }
        }
    }
    // the request's state: sampling of the phase the reply starts in, the protections (as the single-stream path)
    let st = RunState::new(&req, use_spec && ids.len() >= 2 && p.min_tokens == 0);
    let cands: Vec<Vec<u32>> = p.prefix_choices.iter().filter(|c| !c.is_empty()).cloned().collect();
    let lane_state = Lane {
        req,
        sink,
        st,
        ss,
        prompt: ids.clone(),
        feed: ids.clone(),
        resume: false,
        cached: start,
        phase,
        reason: if p.max_tokens == 0 { Some(FinishReason::Length) } else { None },
        t0: Instant::now(),
        t_first: 0.0,
        t_gen: Instant::now(),
        sp: None,
        synced: false,
        force: None,
        cands,
        depth: 0,
        open: vec![],
        base: ids.len() + 1,
        stops: BTreeSet::new(),
        plan: vec![],
        drafted: 0,
        accepted: 0,
    };
    slots[lane] = Slot::Busy(Box::new(lane_state));
    if let Some((o, share)) = leader_stop {
        if let Slot::Busy(l) = &mut slots[o] {
            l.stops.insert(share);
        }
    }
    // a lane that already has the whole prompt but its last token goes straight to the first-token pass
    if let Slot::Busy(l) = &mut slots[lane] {
        if l.phase == Phase::Prefill && e.lane_len(lane) >= ids.len() - 1 {
            l.phase = Phase::First;
        }
    }
    Ok(())
}

fn finish_lane(e: &mut DscCore, slots: &mut Vec<Slot>, li: usize) {
    let Slot::Busy(l) = std::mem::replace(&mut slots[li], Slot::Free(None)) else { return };
    let mut l = *l;
    // whatever was held back while a tool-call cycle was building goes to the client
    l.st.flush_held(&l.sink);
    let mut f = Finish::new(l.reason.clone().unwrap_or(FinishReason::Cancelled));
    f.timings = Some(Timings {
        prompt_n: l.prompt.len() - l.cached,
        cache_n: l.cached,
        prompt_ms: l.t_first,
        predicted_n: l.st.produced,
        predicted_ms: l.t_gen.elapsed().as_secs_f64() * 1e3,
        draft: if l.drafted > 0 { Some((l.drafted, l.accepted)) } else { None },
    });
    f.cached_prompt_tokens = l.cached;
    l.st.report(&l.req, &mut f);
    if let Some(sp) = &l.sp {
        let st = &sp.st;
        if st.rounds > 0 {
            tracing::info!("dsc lane {li}: {} tokens in {} rounds ({:.2} tok/round, k avg {:.2}), copy rounds {}, drafter {:.1} ms/round", st.tokens, st.rounds, st.tokens as f64 / st.rounds as f64, st.drafted as f64 / st.rounds as f64, st.copy_rounds, st.draft_ms / st.rounds as f64);
        }
    }
    let _ = e;
    l.sink.finish(f);
    slots[li] = Slot::Free(Some(Idle { ss: l.ss }));
}
