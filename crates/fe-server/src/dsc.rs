//! `--engine dsc`: DeepSeek-V4.1-Flash Coder (ISTA GGUF) on three cards (main + two expert cards) through fe-dsc. One engine thread owns
//! the cards; with DSC_LANES=1 (the launcher default) it serves ONE request at a time (the server queues the rest). The sequence stays
//! between requests (prefix cache: the longest common prefix is kept, the rest is fed in chunks).
//!
//! Decode: one loop for every request (fe_dsc::engine::spec::drive). Greedy requests and sampled ones without penalties verify drafts
//! (the DSpark drafter, or copies of the context) in one multi-row pass: greedy keeps the argmax-matching prefix (bit-identical to plain
//! greedy), a sampler its lossless accept / reject; penalties or min_tokens take plain one-row steps.
//!
//! Protections: the code the other engines use (crate::guard::PhaseGuard over crate::loopdet::PhaseTracker). The guard is fed the tokens
//! the client gets one at a time, accepted tokens only, and the engine acts at exactly the token where it answers: the tokens a verify
//! accepted past it are given back (those rows are rolled back, the drafter absorbs the kept rows only), so what a protection does
//! never depends on how many rows a round verified, and a request on which none fires gets the tokens it got before. The reasoning cap
//! and a reasoning cycle close the block: the engine writes `</think>` itself (one two-row pass: the last token and the forced one) and
//! drafting goes on after it. An answer / tool-call cycle boosts the sampling for a stretch, then ends the request if it comes back.
//! The first token of a reply is never a stop token. A pick from a non-finite logit row ends its request with an error
//! (FE_FINITE_CHECK; every row a pick reads is on the host, so greedy rows are checked too).
//! Lanes (DSC_LANES>1, dsc_lanes.rs): the same request state (RunState) and the same protections, applied per lane.
//! Environment: DSC_MODEL (first GGUF split), DSC_DRAFT_MODEL (drafter GGUF), DSC_DRAFT_CARD, DSC_XCARDS, DSC_KMAX (drafts per round,
//! default 5), DSC_SPEC=0 (no drafts), DSC_SPEC_SAMPLED=0 (sampled requests take plain steps), DSC_COPY=0 (no copy drafts).
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use fe_dsc::engine::{argmax, draft::Drafter, spec, spec::Session, Engine as DscCore};
use fe_dsc::{config::DscConfig, engram_path, open_model};

use crate::engine::{Engine, EngineInfo, Finish, FinishReason, GenRequest, SamplingParams, Timings, TokenSink};
use crate::guard::{Act, PhaseGuard};
use crate::sampler::Sampler;

pub struct DscEngine {
    tx: Mutex<mpsc::Sender<(GenRequest, TokenSink)>>,
    ctx: usize,
    dead: Arc<Mutex<Option<String>>>,
}

impl DscEngine {
    pub fn start(ctx: usize) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<(GenRequest, TokenSink)>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
        let dead = Arc::new(Mutex::new(None));
        let dead2 = dead.clone();
        std::thread::Builder::new().name("dsc-engine".into()).spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(ctx, rx, &ready_tx)));
            let why = match r {
                Ok(Ok(())) => "engine thread ended".to_string(),
                Ok(Err(e)) => format!("{e:#}"),
                Err(_) => "engine thread panicked".to_string(),
            };
            let _ = ready_tx.send(Err(anyhow::anyhow!("{why}")));
            *dead2.lock().unwrap() = Some(why);
        })?;
        ready_rx.recv()??;
        Ok(Self { tx: Mutex::new(tx), ctx, dead })
    }
}

impl Engine for DscEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "dsc".into(), max_batch: std::env::var("DSC_LANES").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize).clamp(1, 8), max_context: self.ctx }
    }
    fn submit(&self, req: GenRequest, sink: TokenSink) {
        let _ = self.tx.lock().unwrap().send((req, sink));
    }
    fn dead(&self) -> Option<String> {
        self.dead.lock().unwrap().clone()
    }
}

fn run(ctx: usize, rx: mpsc::Receiver<(GenRequest, TokenSink)>, ready: &mpsc::Sender<Result<()>>) -> Result<()> {
    let path = fe_dsc::model_path();
    let eng = engram_path(&path);
    let g = open_model(&path, eng.as_deref(), false)?;
    let c = DscConfig::from_gguf(&g)?;
    // rows of the last spec round sit past the context the request may use
    // a multiple of 1024: the indexer scan range is max_ctx rounded up to 1024 and must not pass the score rows (illegal address otherwise)
    let max_ctx = (ctx + 16).div_ceil(1024) * 1024;
    let mut e = DscCore::new(&g, &c, max_ctx)?;
    let dpath = fe_dsc::draft_path();
    let dg = fe_gguf::Gguf::open(&dpath)?;
    let card: i32 = std::env::var("DSC_DRAFT_CARD").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut dr = Drafter::new(&dg, &g, &c, max_ctx, card)?;
    dr.set_back(e.ctx());
    e.ctx().bind()?;
    // warm-up: graphs, page faults (a short spec request and a plain one)
    let t = Instant::now();
    // real text (code) for the warm-up and the cost calibration: random token ids route to cold experts and make every pass look slower
    let warm: Vec<u32> = match fe_dsc::tok::build(&g, &c).and_then(|t| fe_dsc::tok::encode(&t, WARM_TEXT)) {
        Ok(mut v) if v.len() >= 64 => {
            v.insert(0, 0);
            v.truncate(160);
            v
        }
        _ => (0..100u32).map(|i| 1000 + i * 37).collect(),
    };
    let mut ss = Session::default();
    let _ = e.prefill_chunked(&warm, true)?;
    e.reset()?;
    // the big-prompt path once, over every layer's experts: the pinned host copies are populated by their first DMA (the first prompts after a
    // start ran at a third of the speed without this)
    {
        let mut big = vec![0u32];
        while big.len() < 4300 {
            big.extend_from_slice(&warm[1..]);
        }
        let _ = e.prefill_chunked(&big, false)?;
        e.reset()?;
    }
    dr.reset();
    let _ = spec::spec_stream(&mut e, &mut dr, &mut ss, &warm, 0, 5, None, None, &mut { let mut n = 0; move |t: &[u32]| { n += t.len(); n < 12 } })?;
    // cost model for the adaptive number of drafts (verify ms by rows, drafter ms): measured here on a 100-token context
    let mut cost = if std::env::var("DSC_ADAPT").map_or(true, |v| v != "0") {
        // repeat until two rounds agree (the first ones are slow while the clocks ramp and the kernel is still reclaiming memory after the load)
        let mut c = spec::calibrate(&mut e, &mut dr, &warm)?;
        for _ in 0..14 {
            let c2 = spec::calibrate(&mut e, &mut dr, &warm)?;
            // every entry must agree (the clocks ramp and the kernel reclaims memory for a while after the load): keep the faster of the two
            let stable = c.round_ms.iter().zip(&c2.round_ms).all(|(a, b)| (a - b).abs() < 0.06 * a.min(*b));
            for (a, b) in c.round_ms.iter_mut().zip(&c2.round_ms) {
                *a = a.min(*b);
            }
            c.draft_ms = c.draft_ms.min(c2.draft_ms);
            if stable {
                break;
            }
        }
        // a pass of more rows never costs less than one of fewer
        for i in 1..c.round_ms.len() {
            c.round_ms[i] = c.round_ms[i].max(c.round_ms[i - 1]);
        }
        tracing::info!("dsc: cost model verify ms by rows {:?}, draft {:.1} ms", c.round_ms.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>(), c.draft_ms);
        Some(c)
    } else {
        None
    };
    e.reset()?;
    dr.reset();
    ss.hist.clear();
    tracing::info!("dsc: model loaded, warm-up {:.1} s, context {ctx}, max rewind {}", t.elapsed().as_secs_f32(), e.max_rewind());
    let _ = ready.send(Ok(()));
    let kmax: usize = std::env::var("DSC_KMAX").ok().and_then(|v| v.parse().ok()).unwrap_or(5).clamp(1, 5);
    let use_spec = std::env::var("DSC_SPEC").map_or(true, |v| v != "0");
    if e.n_lanes() > 1 {
        return crate::dsc_lanes::run_lanes(&mut e, &mut dr, cost, rx, kmax, use_spec, ctx);
    }
    let died = AtomicBool::new(false);
    while let Ok((req, sink)) = rx.recv() {
        if req.cancel.is_cancelled() || sink.is_closed() {
            sink.finish(Finish::new(FinishReason::Cancelled));
            continue;
        }
        match serve_one(&mut e, &mut dr, &mut ss, cost.as_mut(), &req, &sink, kmax, use_spec, ctx) {
            Ok(f) => sink.finish(f),
            Err(err) => {
                tracing::error!("dsc: request {} failed: {err:#}", req.id);
                sink.finish(Finish::new(FinishReason::Error(format!("{err:#}"))));
                // a CUDA error is sticky: let the supervisor restart the server
                died.store(true, Ordering::Relaxed);
                return Err(err);
            }
        }
    }
    Ok(())
}

/// A phase's sampling as this engine keeps it: the request's parameters with that phase's values (crate::guard).
#[derive(Clone, PartialEq)]
pub(crate) struct Mode(SamplingParams);

/// FE_FINITE_CHECK (default on): a pick from a logit row that holds an infinity or a NaN ends its request with an error; nothing is
/// sampled from it. Every row a pick reads is on the host here (greedy rows too), so every pick is checked.
fn finite_check() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FE_FINITE_CHECK").map(|v| v != "0").unwrap_or(true))
}

/// FE_TEST_NONFINITE=n: pick n of this process counts as non-finite (tests the path).
fn test_nonfinite() -> bool {
    static N: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    static PICKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = *N.get_or_init(|| std::env::var("FE_TEST_NONFINITE").ok().and_then(|v| v.parse().ok()));
    let k = PICKS.fetch_add(1, Ordering::Relaxed) + 1;
    n == Some(k)
}

const NONFINITE: &str = "the model produced non-finite logits";

fn rows_finite(lgs: &[Vec<f32>]) -> bool {
    if test_nonfinite() {
        return false;
    }
    !finite_check() || lgs.iter().all(|l| l.iter().all(|x| x.is_finite()))
}

/// One request while it decodes: the sampling of its current phase, its protections (crate::guard::PhaseGuard: reasoning cap, loop
/// breaker, repeated-sentence rule) and what the client got so far. The guard sees the tokens the client gets one at a time, accepted
/// tokens only (the engine's own forced `</think>` included); `take` cuts a committed run at the token where the guard, a stop token
/// or the token limit acts, and the decode loop (fe_dsc::engine::spec::drive, or the lane scheduler in dsc_lanes.rs) gives the rest back.
pub(crate) struct RunState {
    /// the sampling of the current phase (reasoning / answer / boosted stretch)
    pub sampler: Sampler,
    /// the current phase decodes plain greedy (argmax of the verify rows)
    pub greedy: bool,
    pub guard: Option<PhaseGuard<Mode>>,
    /// the tokens produced so far (a phase's sampler starts from them)
    pub hist: Vec<u32>,
    pub produced: usize,
    /// drafting allowed at all for this request (the drafter is on, no min_tokens)
    pub spec_ok: bool,
    pub sampled_spec: bool,
    pub fin: Option<FinishReason>,
    /// accepted tokens not sent yet: a cycle is building inside a tool call (guard.holding()); sent when it breaks, dropped at a rewind
    pub held: Vec<u32>,
}

impl RunState {
    /// The state a request starts with: its guard, the sampling of the phase the reply starts in, the tool-call recovery
    /// (loop_rules.call_recover; DSC_CALL_RECOVER=1 / 0 forces it on / off).
    pub fn new(req: &GenRequest, spec_ok: bool) -> RunState {
        let p = &req.params;
        let (guard, mode) = match PhaseGuard::new(p, |q| Mode(q.clone())) {
            Some((g, m)) => (Some(g), m),
            None => (None, Mode(p.clone())),
        };
        let sampler = Sampler::new(&mode.0, &req.prompt_ids);
        let greedy = sampler.is_plain_greedy();
        if let Some(g) = guard.as_ref() {
            match std::env::var("DSC_CALL_RECOVER").ok().as_deref() {
                Some("0") => {}
                Some(_) => g.enable_recover(true),
                None => g.enable_recover(false),
            }
        }
        RunState { sampler, greedy, guard, hist: vec![], produced: 0, spec_ok, sampled_spec: std::env::var("DSC_SPEC_SAMPLED").map_or(true, |v| v != "0"), fin: None, held: vec![] }
    }

    /// The next token is the first of the reply: never a stop token (an empty reply is never wanted; this model often ends a reply at
    /// once after `</think>` on a code prompt).
    pub fn ban(&self, req: &GenRequest) -> bool {
        self.produced < req.params.min_tokens.max(1)
    }

    fn set_mode(&mut self, req: &GenRequest, m: Mode) {
        let mut s = Sampler::new(&m.0, &req.prompt_ids);
        for &t in &self.hist {
            s.accept(t);
        }
        self.greedy = s.is_plain_greedy();
        self.sampler = s;
    }

    /// The next token from one logit row.
    pub fn pick1(&mut self, req: &GenRequest, lg: &[f32]) -> Option<u32> {
        if test_nonfinite() || (finite_check() && !lg.iter().all(|x| x.is_finite())) {
            return self.nonfinite(req);
        }
        let ban = self.ban(req);
        if self.greedy {
            let t = argmax(lg);
            if !(ban && req.params.stop_token_ids.contains(&t)) {
                return Some(t);
            }
        }
        let mut lg = lg.to_vec();
        if ban {
            for &t in &req.params.stop_token_ids {
                if let Some(l) = lg.get_mut(t as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        Some(if self.greedy { argmax(&lg) } else { self.sampler.sample(&mut lg) })
    }

    /// Whatever is still held back goes to the client (the request ends, or the decode loop returned).
    pub fn flush_held(&mut self, sink: &TokenSink) {
        if !self.held.is_empty() {
            let h = std::mem::take(&mut self.held);
            let _ = sink.send_tokens(h);
        }
    }

    fn nonfinite(&mut self, req: &GenRequest) -> Option<u32> {
        tracing::error!("dsc: {}: non-finite logits at output token {}: ending the request", req.id, self.produced);
        self.fin = Some(FinishReason::Error(NONFINITE.into()));
        None
    }

    /// May the next round carry drafts?
    pub fn draft_ok(&mut self, req: &GenRequest) -> bool {
        self.spec_ok && !self.ban(req) && (self.greedy || (self.sampled_spec && !self.sampler.has_penalties()))
    }

    /// The accepted drafts and the token after them from the verify rows (spec::Driver::pick).
    pub fn pick(&mut self, req: &GenRequest, lgs: &[Vec<f32>], drafts: &[u32]) -> Option<(usize, u32)> {
        if drafts.is_empty() {
            return self.pick1(req, &lgs[0]).map(|t| (0, t));
        }
        if self.greedy {
            let g: Vec<u32> = lgs.iter().map(|l| argmax(l)).collect();
            let m = (0..drafts.len()).take_while(|&j| drafts[j] == g[j]).count();
            if !rows_finite(&lgs[..=m]) {
                return self.nonfinite(req).map(|t| (0, t));
            }
            Some((m, g[m]))
        } else {
            if !rows_finite(lgs) {
                return self.nonfinite(req).map(|t| (0, t));
            }
            Some(self.sampler.spec_accept(lgs, drafts))
        }
    }

    /// A committed run: how many of its tokens the request keeps, and what follows (spec::Driver::take).
    pub fn take(&mut self, req: &GenRequest, sink: &TokenSink, out: &[u32]) -> (usize, spec::Next) {
        use spec::Next;
        let p = &req.params;
        let mut n = 0;
        let mut next = Next::Go;
        let mut send: Vec<u32> = Vec::new();
        for &t in out {
            if self.produced >= p.min_tokens && p.stop_token_ids.contains(&t) {
                self.fin = Some(FinishReason::Stop(t));
                next = Next::Stop;
                break;
            }
            self.sampler.accept(t);
            self.hist.push(t);
            self.produced += 1;
            n += 1;
            let hold_t = self.guard.as_ref().is_some_and(|g| g.holding());
            let act = match self.guard.as_mut() {
                Some(g) => g.feed(t),
                None => Act::None,
            };
            if hold_t { self.held.push(t) } else { send.push(t) }
            if !matches!(act, Act::Rewind(..)) && !self.held.is_empty() && !self.guard.as_ref().is_some_and(|g| g.holding()) {
                // the cycle broke: the held tokens are ordinary output
                send.append(&mut self.held);
            }
            if self.produced >= p.max_tokens {
                self.fin = Some(FinishReason::Length);
                next = Next::Stop;
                break;
            }
            match act {
                Act::None => continue,
                // an answer-phase cycle that came back after the boost: end cleanly (as the token limit would)
                Act::End => {
                    self.fin = Some(FinishReason::Length);
                    next = Next::Stop;
                }
                // the engine closes the reasoning block: `</think>` follows this token
                Act::Force(f) => next = Next::Force(f),
                Act::Sampling(m) => self.set_mode(req, m),
                // a tool-call cycle: the held tokens are taken back (none of them was sent), the stretch after it is sampled
                Act::Rewind(nb, nheld, m) => {
                    if nheld != self.held.len() {
                        tracing::warn!("dsc: {}: rewind with {nheld} tokens held by the guard, {} here", req.id, self.held.len());
                    }
                    let nh = nheld.min(self.held.len());
                    self.held.truncate(self.held.len() - nh);
                    send.append(&mut self.held);
                    // the model's sequence goes back to the start of the cycle; the client keeps what it got
                    let nb = nb.min(self.hist.len());
                    self.hist.truncate(self.hist.len() - nb);
                    self.produced -= nh;
                    self.set_mode(req, m);
                    next = Next::Rewind(nb);
                }
            }
            break;
        }
        if next == Next::Stop {
            send.append(&mut self.held);
        }
        let sent = if send.is_empty() { self.held.is_empty() || sink.ping() } else { sink.send_tokens(send) };
        if self.fin.is_none() && (!sent || req.cancel.is_cancelled()) {
            self.fin = Some(FinishReason::Cancelled);
            next = Next::Stop;
        }
        (n, next)
    }

    /// What the protections did, into the request's finish record (and the log when something fired).
    pub fn report(&self, req: &GenRequest, f: &mut Finish) {
        if let Some(g) = &self.guard {
            f.events = g.events();
            let ev = &f.events;
            if ev.reasoning_capped || ev.loop_breaks > 0 || ev.sentence_breaks > 0 {
                tracing::info!("dsc: {}: protections fired: reasoning cap {}, loop breaker {}, sentence rule {} ({} reasoning tokens)", req.id, ev.reasoning_capped, ev.loop_breaks, ev.sentence_breaks, ev.reasoning_tokens);
            }
        }
    }
}

/// A request on the single-stream path: its state with the request and its sink (the driver of fe_dsc::engine::spec::drive).
struct Run<'a> {
    req: &'a GenRequest,
    sink: &'a TokenSink,
    st: RunState,
}

impl std::ops::Deref for Run<'_> {
    type Target = RunState;
    fn deref(&self) -> &RunState {
        &self.st
    }
}

impl std::ops::DerefMut for Run<'_> {
    fn deref_mut(&mut self) -> &mut RunState {
        &mut self.st
    }
}

impl Run<'_> {
    fn pick1(&mut self, lg: &[f32]) -> Option<u32> {
        self.st.pick1(self.req, lg)
    }

    fn flush_held(&mut self) {
        self.st.flush_held(self.sink)
    }
}

impl spec::Driver for Run<'_> {
    fn draft_ok(&mut self) -> bool {
        self.st.draft_ok(self.req)
    }

    fn pick(&mut self, lgs: &[Vec<f32>], drafts: &[u32]) -> Option<(usize, u32)> {
        self.st.pick(self.req, lgs, drafts)
    }

    fn ping(&mut self) {
        let _ = self.sink.ping();
    }

    fn take(&mut self, out: &[u32]) -> (usize, spec::Next) {
        self.st.take(self.req, self.sink, out)
    }
}

fn serve_one(e: &mut DscCore, dr: &mut Drafter, ss: &mut Session, cost: Option<&mut spec::Cost>, req: &GenRequest, sink: &TokenSink, kmax: usize, use_spec: bool, ctx: usize) -> Result<Finish> {
    use spec::{Driver, Next};
    let mut p = req.params.clone();
    let ids = &req.prompt_ids;
    anyhow::ensure!(!ids.is_empty() && ids.len() < ctx, "prompt of {} tokens does not fit the {ctx}-token context", ids.len());
    p.max_tokens = p.max_tokens.min(ctx - ids.len());
    let req = &GenRequest { params: p.clone(), ..req.clone() };
    // prefix cache: roll back to the longest common prefix with the previous sequence, feed only the rest
    let start = if std::env::var("DSC_PCACHE").is_ok_and(|v| v == "0") { e.reset()?; ss.hist.clear(); 0 } else { ss.begin(e, ids)? };
    let t0 = Instant::now();
    let mut t_first = 0.0f64;
    let mut t_gen = Instant::now();
    // the protections and the sampling of the phase the reply starts in (crate::guard, as the other engines)
    let mut run = Run { req, sink, st: RunState::new(req, use_spec && ids.len() >= 2 && p.min_tokens == 0) };
    let mut draft = None;
    if p.max_tokens == 0 {
        run.fin = Some(FinishReason::Length);
    } else {
        ss.prefill(e, ids, start)?;
        let mut lg = ss.feed(e, &ids[ids.len() - 1..], true)?.unwrap().remove(0);
        let mut ids_all = ids.clone();
        let mut next = Next::Go;
        // Constrained opening (tool_choice required with several functions): the reply starts with one of the offered sequences, chosen
        // token by token from the model's logits among the candidates still possible; each chosen token is one more plain row.
        let mut cands: Vec<Vec<u32>> = p.prefix_choices.iter().filter(|c| !c.is_empty()).cloned().collect();
        let mut depth = 0;
        while next == Next::Go && !cands.is_empty() && !cands.iter().any(|c| c.len() == depth) {
            let mut masked = vec![f32::NEG_INFINITY; lg.len()];
            for c in &cands {
                let t = c[depth] as usize;
                if t < lg.len() {
                    masked[t] = lg[t];
                }
            }
            let t = if run.greedy { argmax(&masked) } else { run.sampler.sample(&mut masked) };
            cands.retain(|c| c[depth] == t);
            if cands.is_empty() {
                break;
            }
            depth += 1;
            if depth == 1 {
                t_first = t0.elapsed().as_secs_f64() * 1e3;
                t_gen = Instant::now();
            }
            let (keep, nx) = run.take(&[t]);
            next = nx;
            if keep == 0 || next != Next::Go {
                next = Next::Stop;
                break;
            }
            ids_all.push(t);
            lg = ss.feed(e, &[t], true)?.unwrap().remove(0);
        }
        // the first free token: a plain one-row pick (the stop ban applies to the first token of a reply)
        let mut force = None;
        if next == Next::Go {
            match run.pick1(&lg) {
                Some(t1) => {
                    if depth == 0 {
                        t_first = t0.elapsed().as_secs_f64() * 1e3;
                        t_gen = Instant::now();
                    }
                    let (keep, nx) = run.take(&[t1]);
                    if keep == 1 {
                        ids_all.push(t1);
                    }
                    match nx {
                        Next::Force(f) if keep == 1 => force = Some(f),
                        Next::Go if keep == 1 => {}
                        _ => next = Next::Stop,
                    }
                }
                None => next = Next::Stop,
            }
        }
        if next == Next::Go && ids_all.len() >= 2 {
            let st = spec::drive(e, dr, ss, &ids_all, kmax, cost, force, &mut run)?;
            run.flush_held();
            if st.rounds > 0 {
                tracing::info!("dsc: {} tokens in {} rounds ({:.2} tok/round, k avg {:.2}), per round: verify {:.1} ms + draft {:.1} + absorb {:.1} = {:.1} ms; accepted-by-round hist {:?}, copy rounds {} ({} tokens)",
                    st.tokens, st.rounds, st.tokens as f64 / st.rounds as f64, st.drafted as f64 / st.rounds as f64, st.verify_ms / st.rounds as f64, st.draft_ms / st.rounds as f64, st.absorb_ms / st.rounds as f64, st.total_ms / st.rounds as f64, st.hist, st.copy_rounds, st.copy_tokens);
            }
            let acc: usize = st.hist.iter().enumerate().map(|(m, n)| m * n).sum();
            if st.drafted > 0 {
                draft = Some((st.drafted as u64, acc as u64));
            }
        }
    }
    {
        // cold pairs a token (host-backed experts not yet in a slot ring) over the request: whole, first 100 tokens, last 100 tokens
        e.heat_flush();
        let h = std::mem::take(&mut e.zc_hist);
        let tok: u64 = h.iter().map(|x| x.1 as u64).sum();
        if tok > 0 {
            let cold: u64 = h.iter().map(|x| x.0 as u64).sum();
            let (mut a, mut at, mut b, mut bt) = (0u64, 0u64, 0u64, 0u64);
            for &(c, r) in &h {
                if at < 100 { a += c as u64; at += r as u64; }
            }
            for &(c, r) in h.iter().rev() {
                if bt < 100 { b += c as u64; bt += r as u64; }
            }
            tracing::info!("zc: cold pairs a token {:.1} over {} tokens (first 100: {:.1}, last 100: {:.1})", cold as f64 / tok as f64, tok, a as f64 / at.max(1) as f64, b as f64 / bt.max(1) as f64);
        }
    }
    let mut f = Finish::new(run.fin.clone().unwrap_or(FinishReason::Cancelled));
    f.timings = Some(Timings {
        prompt_n: ids.len() - start,
        cache_n: start,
        prompt_ms: t_first,
        predicted_n: run.produced,
        predicted_ms: t_gen.elapsed().as_secs_f64() * 1e3,
        draft,
    });
    f.cached_prompt_tokens = start;
    run.st.report(req, &mut f);
    Ok(f)
}

const WARM_TEXT: &str = "def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals.\"\"\"\n    if not intervals:\n        return []\n    intervals.sort(key=lambda x: x[0])\n    merged = [intervals[0]]\n    for start, end in intervals[1:]:\n        last_start, last_end = merged[-1]\n        if start <= last_end:\n            merged[-1] = (last_start, max(last_end, end))\n        else:\n            merged.append((start, end))\n    return merged\n\n\nclass Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        return self.items.pop()\n";
