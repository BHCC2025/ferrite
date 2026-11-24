//! Speculative decoding with the model's own draft head: the loop and the policy that picks how many drafts to verify.
//!
//! Per pass: the head drafts 7 tokens after the anchor (the last emitted token); the engine verifies `1 + d` rows (anchor + the first d drafts) as one
//! speculative chunk (`Ink::spec`), the longest accepted prefix is kept (`Ink::rollback`), one more token comes from the target (bonus / correction).
//! Greedy output is identical to plain decode (verify rows have the same bits as decode rows). Sampled output too: every verified row draws its
//! token with the next uniform exactly as plain decode draws one, a draft is accepted when the drawn token is the draft (probability p(d): lossless),
//! and one uniform is used a token emitted, so a seed gives the same tokens with drafting on or off and whatever the draft length.
//! `d` adapts to the running acceptance: cost model pass = c0 + c1 * d ms, value = 1 + sum_j prod_{i<=j} q_i.
use crate::draft::{Draft, NDRAFT};
use crate::sampler::PenHist;
use crate::engine::Ink;
use crate::*;
use anyhow::Result;
use fe_cuda::{DevPtr, DeviceBuffer};
use std::path::Path;

pub const EOS: u32 = 200006;
/// drafting is suspended while its measured speed is below this fraction of plain decode
const SUSPEND_MARGIN: f64 = 0.99;
/// passes between measurements of the plain decode time at long contexts, and the context from which it is measured at all (a plain token costs about
/// the constant `plain_ms` starts from below that)
const CALIB_EVERY: usize = 128;
const CALIB_MIN_CTX: usize = 16384;
const STAGE_ROWS: usize = 64;

pub use crate::sampler::SampleCfg;

/// Kept for callers that only know a temperature.
pub type Sampling = SampleCfg;

#[derive(Default, Clone, Debug)]
pub struct SpecStats {
    pub passes: usize,
    pub accepted: usize,
    pub rows: usize,
    pub tokens: usize,
    pub hist: [usize; 8],
    /// seconds: draft, verify (forward + host sampling), rollback + context extension
    pub t_draft: f64,
    pub t_verify: f64,
    pub t_commit: f64,
    /// tokens produced by plain decode while drafting was suspended, and how often it was suspended
    pub plain_tokens: usize,
    pub suspensions: usize,
}

pub struct Spec {
    pub draft: Draft,
    q: [f64; NDRAFT],
    /// device pointer of the taps [4][2304] of the last processed position (a row of `Ink::dtaps` or of the staging buffer)
    last_ptr: DevPtr,
    /// staging buffer for the taps of the tokens decoded while drafting is suspended
    stage: DeviceBuffer<u8>,
    staged: usize,
    /// drafting suspended for `susp_left` more tokens (plain decode, same bits)
    susp_left: usize,
    susp_k: usize,
    probing: usize,
    passes_run: usize,
    acc_tok: f64,
    acc_ms: f64,
    acc_n: usize,
    calib: bool,
    calib_pending: bool,
    since_calib: usize,
    /// running throughput of spec passes (tokens per ms) and the measured time of a plain decode token (ms)
    spec_rate: f64,
    pub plain_ms: f64,
    /// suspend drafting when it is slower than plain decode (default on)
    pub suspend: bool,
    /// fixed number of verified drafts (None = adaptive)
    pub fixed: Option<usize>,
    cfg: SampleCfg,
    /// token -> (output count, seen in prompt or output): the penalty history
    hist: PenHist,
    /// pass cost model ms = a + b * d, fitted online (exponentially forgotten regression with a weak prior): sums w, wd, wy, wdd, wdy
    reg: [f64; 5],
    drafted_d: usize,
    pub stats: SpecStats,
    /// multi-lane passes in a row this lane ran without drafts (`multi_d`)
    multi_plain: usize,
    /// lanes: the fields above (but `draft`, `suspend`, `fixed` and the pass cost model `reg`, a property of the card) belong to lane `cur`; the other lanes' are parked here (None at `cur`)
    cur: usize,
    parked: Vec<Option<SpecSt>>,
}

pub fn rng_next(s: &mut u64) -> f64 {
    // xorshift64*
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    let r = s.wrapping_mul(0x2545F4914F6CDD1D);
    (r >> 11) as f64 / (1u64 << 53) as f64
}

fn bf(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// softmax(softcap(logits) / T) over one row.
pub fn sample_probs(row: &[u16], t: f32) -> Vec<f32> {
    let mut z: Vec<f32> = row.iter().map(|&h| 30.0 * (bf(h) / 30.0).tanh() / t).collect();
    let mx = z.iter().cloned().fold(f32::MIN, f32::max);
    let mut s = 0.0f64;
    for x in z.iter_mut() {
        *x = (*x - mx).exp();
        s += *x as f64;
    }
    let inv = (1.0 / s) as f32;
    for x in z.iter_mut() {
        *x *= inv;
    }
    z
}

/// Draw from p, optionally with token `skip` removed (renormalised).
pub fn draw_from(p: &[f32], rng: &mut u64) -> u32 {
    draw(p, None, rng)
}

fn draw(p: &[f32], skip: Option<usize>, rng: &mut u64) -> u32 {
    let tot: f64 = p.iter().enumerate().map(|(i, &x)| if Some(i) == skip { 0.0 } else { x as f64 }).sum();
    let mut u = rng_next(rng) * tot;
    let mut last = 0;
    for (i, &x) in p.iter().enumerate() {
        if Some(i) == skip {
            continue;
        }
        last = i;
        u -= x as f64;
        if u <= 0.0 {
            return i as u32;
        }
    }
    last as u32
}

pub fn argmax(row: &[u16]) -> u32 {
    let mut b = (0usize, f32::MIN);
    for (i, &h) in row.iter().enumerate() {
        let x = bf(h);
        if x > b.1 {
            b = (i, x);
        }
    }
    b.0 as u32
}

/// Draft head precision (INK_DRAFT_FP8): 0 = bf16, 1 = FP8 layer matrices, 2 = also the Markov head's first layer and the fold table (0.45 GiB instead of 0.75 GiB).
/// Default: 2 on a card with less than 20 GB (the 16 GB layout), else 0.
fn fp8_level(ink: &Ink) -> u8 {
    if let Some(v) = std::env::var("INK_DRAFT_FP8").ok().and_then(|v| v.parse().ok()) {
        return v;
    }
    match ink.ctx.mem_info() {
        Ok((_, total)) if total < (20usize << 30) => 2,
        _ => 0,
    }
}

impl Spec {
    /// `model_dir`: release dir (mtp.safetensors); `fold`: fold.safetensors.
    pub fn load(ink: &mut Ink, model_dir: &Path, fold: &Path) -> Result<Spec> {
        // on the 16 GB layout the head's memory (0.45 GiB + a growth step) comes from the experts: park enough of them first
        if ink.slot_dev_bytes() > 0 {
            ink.fit_host_experts(640 << 20, usize::MAX / 4)?;
        }
        let draft = Draft::load(ink.ctx.clone(), &ink.s, model_dir, fold, ink.cap, fp8_level(ink))?;
        ink.draft_taps = true;
        // INK_DRAFT_FIXED=n (tests): always verify n drafts and never suspend, so drafting does not depend on measured times (seeded sampled
        // requests with the draft head then give the same tokens run to run)
        let fixed: Option<usize> = std::env::var("INK_DRAFT_FIXED").ok().and_then(|v| v.trim().parse().ok()).filter(|&n: &usize| n >= 1);
        let mut sp = Spec { draft, q: [0.8, 0.75, 0.83, 0.8, 0.75, 0.83, 0.8], last_ptr: DevPtr(0), stage: DeviceBuffer::<u8>::alloc(STAGE_ROWS * 4 * D * 2)?, staged: 0, susp_left: 0,
            susp_k: 32, probing: 0, passes_run: 0, acc_tok: 0.0, acc_ms: 0.0, acc_n: 0, calib: false, calib_pending: true, since_calib: 0, spec_rate: 0.0, plain_ms: 13.5, suspend: true, fixed: None, cfg: SampleCfg::greedy(), hist: PenHist::default(), reg: [4.0, 16.0, 89.6, 80.0, 2.0 * (2.0 * 20.0 + 6.0 * 24.8)], drafted_d: 0, stats: SpecStats::default(), multi_plain: 0, cur: 0, parked: vec![None] };
        if let Some(n) = fixed {
            sp.fixed = Some(n.min(NDRAFT));
            sp.suspend = false;
        }
        Ok(sp)
    }

    pub fn reset(&mut self, ink: &mut Ink) -> Result<()> {
        ink.reset()?;
        self.draft.reset(&ink.s)?;
        self.reset_policy();
        Ok(())
    }

    /// The per-request part of [`Spec::reset`] (acceptance estimates, statistics, suspension) without touching the engine or the draft
    /// context: a request that resumes from a prompt-cache snapshot starts here.
    pub fn reset_policy(&mut self) {
        self.multi_plain = 0;
        self.q = [0.8, 0.75, 0.83, 0.8, 0.75, 0.83, 0.8];
        self.stats = SpecStats::default();
        self.last_ptr = DevPtr(0);
        self.staged = 0;
        self.susp_left = 0;
        self.susp_k = 32;
        self.probing = 0;
        self.passes_run = 0;
        self.calib = false;
        self.calib_pending = true;
        self.since_calib = 0;
        self.acc_tok = 0.0;
        self.acc_ms = 0.0;
        self.acc_n = 0;
        self.spec_rate = 0.0;
    }

    /// The taps of the last processed prompt position (a row of `Ink::dtaps`): the draft head's input for the first pass.
    pub fn set_last_taps(&mut self, p: DevPtr) {
        self.last_ptr = p;
        self.staged = 0;
    }

    /// Run prompt rows (any count; chunks of tmax) from position `ink.len`, feeding the draft context. Returns the bf16 logits of the last row.
    pub fn prefill(&mut self, ink: &mut Ink, ids: &[u32]) -> Result<Vec<u16>> {
        ink.draft_taps = true;
        let prompt = ink.prompt;
        ink.prompt = true;
        let mut pos = 0;
        let mut lg = vec![];
        while pos < ids.len() {
            let t = (ids.len() - pos).min(ink.tmax);
            let p0 = ink.len;
            lg = ink.forward(&ids[pos..pos + t], 1, None, None)?;
            self.draft.extend(ink, ink.dtaps.ptr(), t, p0)?;
            self.last_ptr = ink.dtaps.ptr().byte_add((t - 1) * 4 * D * 2);
            self.staged = 0;
            pos += t;
        }
        ink.prompt = prompt;
        Ok(lg)
    }

    /// Start a request: its sampling configuration and the prompt (penalty history). Call after `reset`, before `prefill`.
    pub fn begin(&mut self, cfg: SampleCfg, prompt: &[u32]) {
        self.cfg = cfg;
        self.hist.begin(&cfg, prompt);
    }

    fn push(&mut self, t: u32) {
        self.hist.push(t);
    }

    fn table(&self) -> Vec<[u32; 3]> {
        self.hist.table()
    }

    /// First generated token from the last prompt row's logits (host row); sampled requests use the same device distribution as the verify rows.
    pub fn first_token(&mut self, ink: &Ink, lg: &[u16], rng: &mut u64) -> Result<u32> {
        let t = if self.cfg.plain_greedy() {
            argmax(lg)
        } else {
            let p = self.draft.upload_row(&ink.s, lg)?;
            let words = self.cfg.words();
            let tab = self.table();
            self.draft.sample_rows_filtered(&ink.s, p, &[], &words, &tab, || rng_next(rng))?.1
        };
        self.push(t);
        Ok(t)
    }

    /// `first_token` from logits already on the device (`lg` = that row; `greedy` = its argmax).
    pub fn first_token_dev(&mut self, ink: &Ink, lg: DevPtr, greedy: u32, rng: &mut u64) -> Result<u32> {
        let t = if self.cfg.plain_greedy() {
            greedy
        } else {
            let words = self.cfg.words();
            let tab = self.table();
            self.draft.sample_rows_filtered(&ink.s, lg, &[], &words, &tab, || rng_next(rng))?.1
        };
        self.push(t);
        Ok(t)
    }

    /// (a, b) of the pass cost model a + b * d (ms)
    pub fn cost_model(&self) -> (f64, f64) {
        let [w, wd, wy, wdd, wdy] = self.reg;
        let det = w * wdd - wd * wd;
        let mut b = if det > 1e-6 { (w * wdy - wd * wy) / det } else { 1.2 };
        b = b.clamp(0.8, 6.0);
        let a = (wy - b * wd) / w;
        (a, b)
    }

    fn observe_cost(&mut self, d: usize, ms: f64) {
        for x in self.reg.iter_mut() {
            *x *= 0.96;
        }
        let d = d as f64;
        self.reg[0] += 1.0;
        self.reg[1] += d;
        self.reg[2] += ms;
        self.reg[3] += d * d;
        self.reg[4] += d * ms;
    }

    fn pick_d(&self) -> usize {
        if let Some(f) = self.fixed {
            return f.clamp(1, NDRAFT);
        }
        // a probe after a suspension verifies 2 rows only (cheap)
        if self.probing > 0 {
            return 1;
        }
        // first passes: spread the verify length so the cost slope is identified (it differs a lot between layouts)
        if self.stats.passes < 8 {
            return [2, 6, 4, 1, 7, 3, 5, 2][self.stats.passes];
        }
        if self.stats.passes % 24 == 23 {
            return if (self.stats.passes / 24) % 2 == 0 { NDRAFT } else { 2 };
        }
        let (mut best, mut cum, mut pj) = ((NDRAFT, 0.0f64), 1.0, 1.0);
        for d in 1..=NDRAFT {
            pj *= self.q[d - 1];
            cum += pj;
            let (ca, cb) = self.cost_model();
            let v = cum / (ca + cb * d as f64);
            if v > best.1 {
                best = (d, v);
            }
        }
        best.0
    }

    /// One pass. `anchor` = the last emitted token (not yet processed by the target). Returns the newly emitted tokens (at least one); the caller stops at EOS / limits.
    pub fn step(&mut self, ink: &mut Ink, anchor: u32, rng: &mut u64) -> Result<Vec<u32>> {
        let (plan, rows) = self.plan(ink, anchor, NDRAFT)?;
        let am = match &plan {
            Plan::Plain { .. } | Plan::Forced => vec![ink.decode_argmax(rows[0])?],
            Plan::Spec { .. } => ink.spec_argmax(&rows)?,
        };
        self.finish(ink, plan, 0, 0, &am, rng)
    }

    /// The first half of a pass (the current lane in `ink` and in the head): the rows to run, `[anchor]` (plain, drafting suspended) or
    /// `[anchor] + d drafts` (a speculative chunk). The caller runs them (alone: `step`; with other lanes: `Ink::forward_multi`) and calls `finish`.
    pub fn plan(&mut self, ink: &mut Ink, anchor: u32, dmax: usize) -> Result<(Plan, Vec<u32>)> {
        if self.suspend && self.fixed.is_none() && self.susp_left == 0 && self.probing == 0 && ink.len >= CALIB_MIN_CTX && (self.calib_pending || self.since_calib >= CALIB_EVERY) {
            self.calib = true;
            self.susp_left = if self.calib_pending { 3 } else { 2 };
            self.calib_pending = false;
            self.since_calib = 0;
        }
        if self.susp_left > 0 {
            let t0 = std::time::Instant::now();
            if self.staged == STAGE_ROWS {
                self.flush_stage(ink)?;
            }
            return Ok((Plan::Plain { t0 }, vec![anchor]));
        }
        if self.staged > 0 {
            self.flush_stage(ink)?;
        }
        let t0 = std::time::Instant::now();
        // passes that try other draft lengths on purpose (see pick_d) say nothing about the speed of the best choice: they do not enter the rate
        let explore = self.stats.passes < 8 || self.stats.passes % 24 == 23;
        let before = ink.len;
        let dr = self.draft.draft(ink, anchor, self.last_ptr)?;
        let t1 = std::time::Instant::now();
        let d = self.pick_d().min(dmax.max(1));
        self.drafted_d = d;
        let mut rows = vec![anchor];
        rows.extend_from_slice(&dr[..d]);
        Ok((Plan::Spec { t0, t1, before, dr, d, explore, multi: false }, rows))
    }

    /// Drafts for this lane in a multi-lane pass: the d that adds the most tokens beyond what the pass time it costs would give the plain
    /// rows of the pass (`rate` tokens a ms), with `draft_ms` for the head and `row_ms` a verify row; 0 = no drafts this pass. A lane that
    /// went 16 passes without drafts probes 2 (its acceptance estimates only learn from drafted passes).
    pub fn multi_d(&mut self, rate: f64, draft_ms: f64, row_ms: f64) -> usize {
        if let Some(f) = self.fixed {
            return f.clamp(1, NDRAFT);
        }
        if self.stats.passes < 4 {
            return 3;
        }
        if self.multi_plain >= 16 {
            return 2;
        }
        let (mut best, mut cum, mut pj) = ((0usize, 0.0f64), 0.0, 1.0);
        for d in 1..=NDRAFT {
            pj *= self.q[d - 1];
            cum += pj;
            let gain = cum - rate * (draft_ms + row_ms * d as f64);
            if gain > best.1 {
                best = (d, gain);
            }
        }
        best.0
    }

    /// `plan` for a multi-lane pass with `d` drafts chosen by `multi_d` (0 = the plain row only); the single-lane suspension rule is not used here.
    pub fn plan_multi(&mut self, ink: &mut Ink, anchor: u32, d: usize) -> Result<(Plan, Vec<u32>)> {
        if d == 0 {
            if self.staged == STAGE_ROWS {
                self.flush_stage(ink)?;
            }
            return Ok((Plan::Forced, vec![anchor]));
        }
        if self.staged > 0 {
            self.flush_stage(ink)?;
        }
        let t0 = std::time::Instant::now();
        let before = ink.len;
        let dr = self.draft.draft(ink, anchor, self.last_ptr)?;
        let t1 = std::time::Instant::now();
        let d = d.min(NDRAFT);
        self.drafted_d = d;
        let mut rows = vec![anchor];
        rows.extend_from_slice(&dr[..d]);
        Ok((Plan::Spec { t0, t1, before, dr, d, explore: true, multi: true }, rows))
    }

    /// The second half of a pass: `lrow` = the lane's first row among the logits the forward left on the device, `trow` = its first row in
    /// `Ink::dtaps` (the forward's row), `am` = the greedy tokens of its rows. Accepts / rejects, rolls the lane back to the kept rows, extends
    /// the draft context; returns the emitted tokens.
    pub fn finish(&mut self, ink: &mut Ink, plan: Plan, lrow: usize, trow: usize, am: &[u32], rng: &mut u64) -> Result<Vec<u32>> {
        let lp = ink.logits_ptr().byte_add(lrow * VOCAB * 2);
        let row0 = trow;
        let tap_b = 4 * D * 2;
        match plan {
            Plan::Plain { t0 } => {
                // one plain decode token (same bits) while drafting is suspended: the taps of the processed token are staged for the draft context
                let tok = if self.cfg.plain_greedy() {
                    am[0]
                } else {
                    let words = self.cfg.words();
                    let tab = self.table();
                    self.draft.sample_rows_filtered(&ink.s, lp, &[], &words, &tab, || rng_next(rng))?.1
                };
                ink.s.memcpy_dtod(self.stage.ptr().byte_add(self.staged * tap_b), ink.dtaps.ptr().byte_add(row0 * tap_b), tap_b)?;
                self.staged += 1;
                self.push(tok);
                self.stats.plain_tokens += 1;
                self.stats.tokens += 1;
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                self.plain_ms = if self.calib { ms } else { 0.8 * self.plain_ms + 0.2 * ms };
                self.susp_left -= 1;
                if self.susp_left == 0 {
                    self.flush_stage(ink)?;
                    if self.calib {
                        self.calib = false;
                    } else {
                        self.probing = 3;
                        self.passes_run = 1;
                        self.acc_tok = 0.0;
                        self.acc_ms = 0.0;
                        self.acc_n = 0;
                    }
                }
                Ok(vec![tok])
            }
            Plan::Forced => {
                let tok = if self.cfg.plain_greedy() {
                    am[0]
                } else {
                    let words = self.cfg.words();
                    let tab = self.table();
                    self.draft.sample_rows_filtered(&ink.s, lp, &[], &words, &tab, || rng_next(rng))?.1
                };
                ink.s.memcpy_dtod(self.stage.ptr().byte_add(self.staged * tap_b), ink.dtaps.ptr().byte_add(row0 * tap_b), tap_b)?;
                self.staged += 1;
                self.push(tok);
                self.stats.plain_tokens += 1;
                self.stats.tokens += 1;
                self.multi_plain += 1;
                Ok(vec![tok])
            }
            Plan::Spec { t0, t1, before, dr, d, explore, multi } => {
                let mut out: Vec<u32> = vec![];
                let k;
                if self.cfg.plain_greedy() {
                    let mut kk = 0;
                    while kk < d && dr[kk] == am[kk] {
                        kk += 1;
                    }
                    k = kk;
                    out.extend_from_slice(&dr[..k]);
                    out.push(am[k]);
                } else {
                    // verify rows leave their logits on the device; accept / reject and the draw run there
                    // the filtered sampler for every sampled request (temperature-only too): its rows draw exactly as plain decode draws, so the
                    // tokens of a seed do not depend on drafting; history of row j = base + drafted prefix (the kernel adds the prefix; the base
                    // excludes the anchor's successors, includes the anchor, already pushed)
                    let (kk, tok) = {
                        let words = self.cfg.words();
                        let tab = self.table();
                        self.draft.sample_rows_filtered(&ink.s, lp, &dr[..d], &words, &tab, || rng_next(rng))?
                    };
                    k = kk;
                    out.extend_from_slice(&dr[..k]);
                    out.push(tok);
                }
                let t2 = std::time::Instant::now();
                ink.rollback(k + 1)?;
                self.draft.extend(ink, ink.dtaps.ptr().byte_add(row0 * tap_b), k + 1, before)?;
                self.last_ptr = ink.dtaps.ptr().byte_add((row0 + k) * tap_b);
                let t3 = std::time::Instant::now();
                for &t in &out {
                    self.push(t);
                }
                for i in 1..=d {
                    if k + 1 >= i {
                        self.q[i - 1] = 0.9 * self.q[i - 1] + 0.1 * if k >= i { 1.0 } else { 0.0 };
                    }
                }
                let st = &mut self.stats;
                st.passes += 1;
                st.accepted += k;
                st.rows += d + 1;
                st.tokens += k + 1;
                st.hist[k] += 1;
                st.t_draft += (t1 - t0).as_secs_f64();
                st.t_verify += (t2 - t1).as_secs_f64();
                st.t_commit += (t3 - t2).as_secs_f64();
                if multi {
                    self.multi_plain = 0;
                } else {
                    self.after_pass(out.len(), t0.elapsed().as_secs_f64() * 1e3, explore);
                }
                Ok(out)
            }
        }
    }

    /// Rate bookkeeping and the suspension rule after a speculative pass of `n` tokens in `ms`.
    fn after_pass(&mut self, n: usize, ms: f64, explore: bool) {
        let rate = n as f64 / ms;
        if self.passes_run >= 2 {
            self.observe_cost(self.drafted_d, ms);
        }
        // the first passes after a start / resume include warm-up (kernel loads, graph capture): not counted
        if self.passes_run >= 2 {
            self.spec_rate = if self.passes_run == 2 { rate } else { 0.85 * self.spec_rate + 0.15 * rate };
        }
        // steady state: the speed of the best draft length over many passes (the passes that try other lengths on purpose and the probes are left out)
        if self.passes_run >= 2 && !explore && self.probing == 0 {
            self.acc_tok += n as f64;
            self.acc_ms += ms;
            self.acc_n += 1;
            if self.acc_n >= 64 {
                self.acc_tok *= 0.5;
                self.acc_ms *= 0.5;
                self.acc_n = 32;
            }
        }
        self.passes_run += 1;
        self.since_calib += 1;
        if self.suspend && self.fixed.is_none() {
            let slow = if self.probing > 0 {
                self.passes_run >= 2 && self.spec_rate < SUSPEND_MARGIN / self.plain_ms
            } else {
                self.acc_n >= 12 && self.acc_tok / self.acc_ms < SUSPEND_MARGIN / self.plain_ms
            };
            if self.probing > 0 {
                self.probing -= 1;
                if self.probing == 0 && slow {
                    self.susp_k = (self.susp_k * 4).min(1024);
                    self.susp_left = self.susp_k;
                    self.stats.suspensions += 1;
                } else if self.probing == 0 {
                    self.susp_k = 32;
                }
            } else if self.passes_run >= 10 && slow {
                self.susp_left = self.susp_k;
                self.stats.suspensions += 1;
            }
        }
    }

    fn flush_stage(&mut self, ink: &mut Ink) -> Result<()> {
        if self.staged > 0 {
            let at = self.draft.ctx_len;
            self.draft.extend(ink, self.stage.ptr(), self.staged, at)?;
            self.last_ptr = self.stage.ptr().byte_add((self.staged - 1) * 4 * D * 2);
            self.staged = 0;
        }
        Ok(())
    }

    /// Give the head `n` lanes (each its own draft context, policy state, sampler history and staging rows).
    pub fn add_lanes(&mut self, n: usize) -> Result<()> {
        self.draft.add_lanes(n)?;
        while self.parked.len() < n {
            self.parked.push(Some(SpecSt::fresh()?));
        }
        Ok(())
    }

    pub fn n_lanes(&self) -> usize {
        self.parked.len()
    }

    /// Make lane `i` current (its policy state and draft context); select the same lane in the engine (`Ink::select_lane`).
    pub fn select_lane(&mut self, i: usize) -> Result<()> {
        anyhow::ensure!(i < self.parked.len(), "spec lane {i} of {}", self.parked.len());
        if i == self.cur {
            return Ok(());
        }
        self.draft.select_lane(i)?;
        let mut st = self.parked[i].take().unwrap();
        macro_rules! sw { ($($f:ident),*) => { $( std::mem::swap(&mut self.$f, &mut st.$f); )* } }
        sw!(q, last_ptr, stage, staged, susp_left, susp_k, probing, passes_run, acc_tok, acc_ms, acc_n, calib, calib_pending, since_calib, spec_rate,
            plain_ms, cfg, hist, drafted_d, stats, multi_plain);
        self.parked[self.cur] = Some(st);
        self.cur = i;
        Ok(())
    }
}

/// The first half of a pass (see `Spec::plan`).
pub enum Plan {
    Plain { t0: std::time::Instant },
    Spec { t0: std::time::Instant, t1: std::time::Instant, before: usize, dr: [u32; NDRAFT], d: usize, explore: bool, multi: bool },
    /// a multi-lane pass without drafts by choice (`Spec::multi_d`): the taps are staged as while suspended
    Forced,
}

/// A lane's per-request drafting state while another lane is current (see `Spec::select_lane`).
struct SpecSt {
    q: [f64; NDRAFT],
    last_ptr: DevPtr,
    stage: DeviceBuffer<u8>,
    staged: usize,
    susp_left: usize,
    susp_k: usize,
    probing: usize,
    passes_run: usize,
    acc_tok: f64,
    acc_ms: f64,
    acc_n: usize,
    calib: bool,
    calib_pending: bool,
    since_calib: usize,
    spec_rate: f64,
    plain_ms: f64,
    cfg: SampleCfg,
    hist: PenHist,
    drafted_d: usize,
    stats: SpecStats,
    multi_plain: usize,
}

impl SpecSt {
    /// The state `Spec::load` starts with.
    fn fresh() -> Result<SpecSt> {
        Ok(SpecSt { q: [0.8, 0.75, 0.83, 0.8, 0.75, 0.83, 0.8], last_ptr: DevPtr(0), stage: DeviceBuffer::<u8>::alloc(STAGE_ROWS * 4 * D * 2)?, staged: 0, susp_left: 0,
            susp_k: 32, probing: 0, passes_run: 0, acc_tok: 0.0, acc_ms: 0.0, acc_n: 0, calib: false, calib_pending: true, since_calib: 0, spec_rate: 0.0, plain_ms: 13.5,
            cfg: SampleCfg::greedy(), hist: PenHist::default(), drafted_d: 0, stats: SpecStats::default(), multi_plain: 0 })
    }
}
