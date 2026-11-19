//! `--engine fnext`: Qwen3.8-Flash-Next (GGUF, qwen4exp) on one or two cards (FNEXT_GPUS=2 layer split) through
//! fe-fnext's fast path (cargo feature `fnext`). An engine thread owns the CUDA contexts and runs up to FNEXT_SLOTS
//! requests at once on lanes (continuous batching: a new request prefills between steps, then joins the batched
//! decode). Each lane keeps its sequence between requests: a prompt that extends it (a chat follow-up) only runs its
//! new tokens. Greedy requests decode with MTP drafts (FNEXT_MTP=1, default on), several lanes at once through
//! spec_lanes; while any live request samples, every live lane steps one token at a time.
//!
//! Protections: the code the 27B engine uses (crate::guard::PhaseGuard over crate::loopdet::PhaseTracker). The guard
//! is fed the tokens the client gets one at a time, accepted tokens only, and the engine acts at exactly the token
//! where it answers: the tokens a verify accepted past it are given back (the lane keeps fewer rows), so what a
//! protection does never depends on how many rows a step verified, and a request on which none fires gets the
//! tokens it got before. The reasoning cap and a reasoning cycle close the block (the engine writes `</think>`
//! itself, as one verify row without drafts; the MTP block absorbs it and the next step drafts again). An answer /
//! tool-call cycle boosts the sampling for a stretch, then ends the request if it comes back. A sampled stretch of
//! a greedy request (that boost, or a reasoning block sampled under greedy_floor) decodes one token a step; a lane
//! that decodes alone keeps its MTP block in step through it and drafts again afterwards, a lane with neighbours
//! takes batched plain steps and stays on them. The first token of a reply is never a stop token (an empty reply
//! is never wanted). A pick from a non-finite logit row ends its request with an error (FE_FINITE_CHECK).
use std::sync::mpsc;
use std::time::Instant;

use anyhow::Result;
use fe_fnext::fast::{Fast, Spec};

use crate::engine::{Engine, EngineInfo, Finish, FinishReason, GenRequest, SamplingParams, Timings, TokenSink};
use crate::guard::{Act, PhaseGuard};
use crate::sampler::Sampler;

pub struct FnextEngine {
    tx: std::sync::Mutex<mpsc::Sender<(GenRequest, TokenSink)>>,
    ctx: usize,
    lanes: usize,
}

fn envu(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

impl FnextEngine {
    pub fn start(ctx: usize) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<(GenRequest, TokenSink)>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<usize>>();
        std::thread::Builder::new().name("fnext-engine".into()).spawn(move || {
            if std::env::var("FNEXT_MTP").is_err() {
                // SAFETY: set before the model reads it, on this thread, before any other engine thread exists
                unsafe { std::env::set_var("FNEXT_MTP", "1") };
            }
            let counts = std::env::var("FNEXT_COUNTS").ok().and_then(|p| fe_fnext::fast::load_counts(&p).ok());
            let mib = std::env::var("FNEXT_EXPERT_MIB").ok().and_then(|v| v.parse().ok());
            let m = Fast::load(&fe_fnext::default_model(), ctx, counts.as_ref(), mib, envu("FNEXT_THREADS", 4));
            let mut m = match m {
                Ok(mut m) => {
                    // capture the step graphs (plain rows, verify rows, MTP rows, commits) before the first request
                    let t = Instant::now();
                    if let Err(e) = warmup(&mut m) {
                        tracing::warn!("fnext: warm-up failed: {e:#}");
                    }
                    let _ = m.reset();
                    tracing::info!("fnext: warm-up {:.1} s", t.elapsed().as_secs_f32());
                    let _ = ready_tx.send(Ok(m.n_lanes()));
                    m
                }
                Err(e) => {
                    // a raw CUDA OOM says nothing about what to change: on the card (out of memory), or pinned host
                    // buffers past the memory limit (cuMemHostAlloc fails, often as CUDA_ERROR_UNKNOWN)
                    let cu = e.chain().find_map(|c| c.downcast_ref::<fe_fnext::CuError>()).cloned();
                    let slots = envu("FNEXT_SLOTS", 1);
                    let e = match cu {
                        Some(c) if c.call.contains("HostAlloc") || c.call.contains("HostRegister") => e.context(format!(
                            "pinned host memory ran out while loading Flash-Next ({ctx}-token context, {slots} request \
                            slots): lower FERRITE_CTX or FERRITE_SLOTS, or give the server more memory")),
                        Some(c) if c.code == 2 => e.context(format!("the card ran out of memory while loading Flash-Next \
                            ({ctx}-token context, {slots} request slots): lower FERRITE_CTX or FERRITE_SLOTS, or stop \
                            what else uses the card (nvidia-smi)")),
                        _ => e,
                    };
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            schedule(&mut m, rx);
        })?;
        let lanes = ready_rx.recv()??;
        Ok(Self { tx: std::sync::Mutex::new(tx), ctx, lanes })
    }
}

impl Engine for FnextEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "fnext".into(), max_batch: self.lanes, max_context: self.ctx }
    }
    fn submit(&self, req: GenRequest, sink: TokenSink) {
        let _ = self.tx.lock().unwrap().send((req, sink));
    }
}

fn warmup(m: &mut Fast) -> Result<()> {
    // every graph a request can use: plain rows 1..8, verify rows 1..8, commits 1..8, MTP rows 1..8, the prompt path.
    // Step graphs come in two kinds (with / without the logits host copy, see Fast::want_logits): the greedy kind for
    // everything, the sampled kind for plain rows 1..8 (a sampled request never verifies).
    let toks: Vec<u32> = (0..200u32).map(|i| 1000 + i * 37).collect();
    m.want_logits = true;
    for r in 1..=8 {
        m.step(&toks[r * 8..r * 8 + r])?;
    }
    m.reset()?;
    m.want_logits = false;
    if m.has_mtp() {
        m.prefill_mtp(&toks[..100])?;
    } else {
        m.prefill(&toks[..100])?;
    }
    m.reset()?;
    let mut t = 0;
    for r in 1..=8 {
        t = m.step(&toks[r * 8..r * 8 + r])?;
    }
    for r in 1..=8 {
        let rows: Vec<u32> = std::iter::once(t).chain(toks[100..100 + r - 1].iter().copied()).collect();
        m.verify(&rows)?;
        m.commit(&rows, r)?;
    }
    // the batched graphs of every lane set of two or more (FNEXT_WARM_LANES=0: captured when first used)
    let n = m.n_lanes();
    if n > 1 && std::env::var("FNEXT_WARM_LANES").map_or(true, |v| v != "0") {
        let t = Instant::now();
        let a0 = rss_anon_mib();
        let sets: Vec<Vec<usize>> = (1u32..(1 << n)).filter(|b| b.count_ones() >= 2)
            .map(|b| (0..n).filter(|&i| b >> i & 1 == 1).collect()).collect();
        let made = m.warm_lanes(&sets, envu("FNEXT_DRAFTS", 4) + 1)?;
        let da = rss_anon_mib().saturating_sub(a0);
        tracing::info!("fnext: {made} batched lane graphs for {} lane sets in {:.1} s, +{da} MiB host memory ({:.1} MiB a graph)",
            sets.len(), t.elapsed().as_secs_f32(), da as f64 / made.max(1) as f64);
    }
    if m.has_mtp() {
        m.reset()?;
        let (t, d) = m.prefill_mtp(&toks[..40])?;
        let mut sp = Spec { cur: t, drafts: vec![d] };
        for _ in 0..8 {
            m.spec_step(&mut sp, 7)?;
        }
    }
    m.reset()?;
    Ok(())
}

/// A phase's sampling as this engine keeps it: the request's parameters with that phase's values (crate::guard).
#[derive(Clone, PartialEq)]
struct Mode(SamplingParams);

/// Greedy with no penalties: the device argmax decides (the MTP path).
fn plain_greedy(p: &SamplingParams) -> bool {
    p.temperature <= 0.0 && p.repetition_penalty == 1.0 && p.presence_penalty == 0.0 && p.frequency_penalty == 0.0
}

/// FE_FINITE_CHECK (default on): a pick from a logit row that holds an infinity or a NaN ends its request with an
/// error; nothing is sampled from it. Sampled rows are scanned on the host. A greedy row never comes to the host:
/// under FNEXT_TP=2 its best logit and probability do (fe_fnext::fast::Fast::bad_rows); without tensor parallel a
/// greedy row is not checked.
fn finite_check() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FE_FINITE_CHECK").map(|v| v != "0").unwrap_or(true))
}

/// FE_TEST_NONFINITE=n: pick n of this process counts as non-finite (tests the path).
fn test_nonfinite() -> bool {
    static N: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    static PICKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = *N.get_or_init(|| std::env::var("FE_TEST_NONFINITE").ok().and_then(|v| v.parse().ok()));
    let k = PICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    n == Some(k)
}

const NONFINITE: &str = "the model produced non-finite logits";

/// A request decoding on a lane.
struct Active {
    req: GenRequest,
    sink: TokenSink,
    /// the sampling of the current phase (reasoning / answer / boosted stretch)
    sampler: Sampler,
    /// the current phase decodes plain greedy (the device argmax; MTP drafts)
    greedy: bool,
    /// the answer phase decodes plain greedy: a sampled stretch (a sampled reasoning block, the boosted stretch after
    /// an answer loop) can return to drafting
    answer_greedy: bool,
    /// reasoning cap, loop breaker, repeated-sentence rule and the sampling of each phase; None when the request has
    /// neither a reasoning block nor the breaker
    guard: Option<PhaseGuard<Mode>>,
    /// MTP drafting state (a greedy phase whose MTP block is in step)
    spec: Option<Spec>,
    /// the MTP block has seen every position of this lane's sequence, so drafting can start (again); lost for good
    /// when the lane takes a batched plain step
    mtp_live: bool,
    /// the tokens produced so far (a phase's sampler starts from them)
    hist: Vec<u32>,
    last: u32,
    produced: usize,
    cached: usize,
    prompt_n: usize,
    t_first: f64,
    t_gen: Instant,
    d0: (u64, u64),
    /// a token the engine writes itself as the next one instead of the model's (the forced `</think>`)
    force: Option<u32>,
}

impl Active {
    /// The next token needs the logit row on the host (a sampled phase).
    fn needs_logits(&self) -> bool {
        !self.greedy
    }

    /// The next token is the first of the reply: never a stop token (an empty reply is never wanted).
    fn ban(&self) -> bool {
        self.produced < self.req.params.min_tokens.max(1)
    }

    fn set_mode(&mut self, m: Mode) {
        let mut s = Sampler::new(&m.0, &self.req.prompt_ids);
        for &t in &self.hist {
            s.accept(t);
        }
        self.greedy = s.is_plain_greedy();
        self.sampler = s;
    }
}

/// What the request takes of a run of tokens the engine offers.
struct Scan {
    /// the tokens the client gets
    out: Vec<u32>,
    fin: Option<FinishReason>,
    /// what the guard asked for at the last token of `out`
    act: Act<Mode>,
}

/// Take the tokens of `cand` the client will get: up to a stop token or the token limit, and up to the token where
/// the request's guard acts. The guard is fed one token at a time; the tokens after the one it acts on are cut
/// from `cand` (the engine gives them back). A stop token or the limit ends the request and leaves `cand` alone.
fn scan(a: &mut Active, cand: &mut Vec<u32>) -> Scan {
    let mut sc = Scan { out: Vec::with_capacity(cand.len()), fin: None, act: Act::None };
    for j in 0..cand.len() {
        let t = cand[j];
        let p = &a.req.params;
        if a.produced >= p.min_tokens && p.stop_token_ids.contains(&t) {
            sc.fin = Some(FinishReason::Stop(t));
            break;
        }
        a.sampler.accept(t);
        a.hist.push(t);
        sc.out.push(t);
        a.produced += 1;
        a.last = t;
        let full = a.produced >= a.req.params.max_tokens;
        if let Some(g) = a.guard.as_mut() {
            sc.act = g.feed(t);
        }
        if full {
            sc.fin = Some(FinishReason::Length);
        }
        if sc.act != Act::None {
            cand.truncate(j + 1);
        }
        if full || sc.act != Act::None {
            break;
        }
    }
    sc
}

/// Send what `scan` took and do what the guard asked for; Some(reason) when the request is done.
fn apply(a: &mut Active, sc: Scan) -> Option<FinishReason> {
    let sent = sc.out.is_empty() || a.sink.send_tokens(sc.out);
    if sc.fin.is_some() {
        return sc.fin;
    }
    if !sent || a.req.cancel.is_cancelled() {
        return Some(FinishReason::Cancelled);
    }
    match sc.act {
        Act::None => {}
        // an answer-phase cycle that came back after the boost: end cleanly (as the token limit would)
        Act::End => return Some(FinishReason::Length),
        // the engine closes the reasoning block: `</think>` follows this token
        Act::Force(t) => a.force = Some(t),
        Act::Sampling(m) => {
            a.set_mode(m);
            if !a.greedy {
                // a sampled stretch: no drafts (the MTP block stays in step while the lane decodes alone)
                a.spec = None;
            }
        }
    }
    None
}

/// Emit `toks`; Some(reason) when the request is done.
fn emit(a: &mut Active, toks: &[u32]) -> Option<FinishReason> {
    let mut v = toks.to_vec();
    let sc = scan(a, &mut v);
    apply(a, sc)
}

/// The next token from a host logit row (plain greedy: the engine's argmax unless the stop ban removes it).
fn pick(a: &mut Active, lg: &[f32], greedy_tok: u32) -> u32 {
    let ban = a.ban();
    if a.greedy && !(ban && a.req.params.stop_token_ids.contains(&greedy_tok)) {
        return greedy_tok;
    }
    let mut lg = lg.to_vec();
    if ban {
        for &t in &a.req.params.stop_token_ids {
            if let Some(l) = lg.get_mut(t as usize) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
    if a.greedy {
        return crate::sampler::argmax(&lg);
    }
    a.sampler.sample(&mut lg)
}

/// Is the row the next pick of `a` reads finite? `lg`: the host row (when the step copied one), `dev_bad`: the
/// engine's own flag for the lane's greedy row.
fn pick_finite(a: &Active, lg: &[f32], greedy_tok: u32, dev_bad: bool) -> bool {
    if test_nonfinite() {
        return false;
    }
    if !finite_check() {
        return true;
    }
    let host = !a.greedy || (a.ban() && a.req.params.stop_token_ids.contains(&greedy_tok));
    if host { lg.iter().all(|x| x.is_finite()) } else { !dev_bad }
}

fn done(m: &Fast, a: Active, reason: FinishReason) {
    let mut f = Finish::new(reason);
    f.cached_prompt_tokens = a.cached;
    let d1s = m.spec_stats;
    if let Some(g) = &a.guard {
        f.events = g.events();
    }
    f.timings = Some(Timings {
        prompt_n: a.prompt_n,
        cache_n: a.cached,
        prompt_ms: a.t_first,
        predicted_n: a.produced,
        predicted_ms: a.t_gen.elapsed().as_secs_f64() * 1000.0,
        draft: if a.spec.is_some() { Some((d1s.0 - a.d0.0, d1s.1 - a.d0.1)) } else { None },
    });
    a.sink.finish(f);
}

/// End a request whose next pick would read a non-finite row: an error for the client, a log line, the lane reset.
fn fail_nonfinite(m: &mut Fast, lane: usize, a: Active) {
    tracing::error!("fnext: {}: non-finite logits at output token {}: ending the request", a.req.id, a.produced);
    done(m, a, FinishReason::Error(NONFINITE.into()));
    if m.select_lane(lane).is_ok() {
        let _ = m.reset();
    }
}

/// Prefill a new request on the current lane (prompt cache: resume at the longest snapshot of the lane's sequence,
/// prefill cut at the request's snapshot points) and emit its first token. None when it already finished.
fn start(m: &mut Fast, req: GenRequest, sink: TokenSink, alone: bool) -> Result<Option<Active>> {
    let t0 = Instant::now();
    let prompt = req.prompt_ids.clone();
    if req.cancel.is_cancelled() {
        // the client left while the request waited for a lane
        sink.finish(Finish::new(FinishReason::Cancelled));
        return Ok(None);
    }
    if prompt.is_empty() {
        sink.finish(Finish::new(FinishReason::Error("empty prompt".into())));
        return Ok(None);
    }
    let cache_on = std::env::var("FNEXT_CACHE").map_or(true, |v| v != "0");
    let min = envu("FNEXT_CACHE_MIN", 1024);
    let lcp = m.tokens.iter().zip(prompt.iter()).take_while(|(a, b)| a == b).count();
    let best = if cache_on {
        m.snaps.iter().filter(|s| s.pos <= lcp && s.pos < prompt.len() && s.pos >= min).map(|s| s.pos).max()
    } else {
        None
    };
    let cached = match best {
        Some(pos) => {
            let i = m.snaps.iter().position(|s| s.pos == pos).unwrap();
            let sn = m.snaps.remove(i);
            let r = m.snap_restore(&sn);
            m.snaps.insert(i, sn);
            r?;
            m.snaps.retain(|s| s.pos <= pos);
            pos
        }
        None => {
            m.reset()?;
            0
        }
    };
    let rest = &prompt[cached..];
    // the protections and the sampling of the phase the reply starts in (crate::guard, as the 27B engine)
    let (guard, mode) = match PhaseGuard::new(&req.params, |q| Mode(q.clone())) {
        Some((g, s)) => (Some(g), s),
        None => (None, Mode(req.params.clone())),
    };
    let sampler = Sampler::new(&mode.0, &prompt);
    let greedy = sampler.is_plain_greedy();
    let answer_greedy = plain_greedy(&req.params);
    // MTP for every greedy request (spec_lanes batches several), and for one whose answer is greedy after a sampled
    // reasoning block; `alone` only matters to the long-request path
    let _ = alone;
    let spec = (greedy || answer_greedy) && m.has_mtp() && std::env::var("FNEXT_SPEC").map_or(true, |v| v != "0");
    // the prompt's last step always hands its logit row to the host: a sampled request draws its first token from it, a
    // greedy one needs it when the model's first choice is a stop token (see pick). The step graphs are keyed by this flag.
    m.want_logits = true;
    let t_pre = Instant::now();
    let mut t_snap = 0f64;
    // a client that goes away stops the prompt between layers (within ~0.2 s), see Fast::cancel
    m.cancel = Some(req.cancel.flag());
    let res = m.prefill_cut(rest, &req.snapshots, spec, |m| {
        if cache_on {
            let ts = Instant::now();
            let sn = m.snap_save()?;
            t_snap += ts.elapsed().as_secs_f64();
            if m.snaps.last().is_none_or(|l| l.pos < sn.pos) {
                m.snaps.push(sn);
            }
            // keep the first (the shared system prompt) and the latest two
            while m.snaps.len() > 3 {
                m.snaps.remove(1);
            }
        }
        Ok(())
    });
    if std::env::var("FNEXT_VERBOSE").is_ok() {
        eprintln!("fnext: lane {} resume+restore {:.0} ms, prefill {} tokens {:.0} ms (of it snapshots {:.0} ms)", m.lane(),
            (t_pre - t0).as_secs_f64() * 1e3, rest.len(), t_pre.elapsed().as_secs_f64() * 1e3, t_snap * 1e3);
    }
    m.cancel = None;
    let (first, d1) = match res {
        Ok(v) => v,
        Err(e) if e.is::<fe_fnext::fast::Cancelled>() => {
            // the lane holds a partly computed prompt: back to the latest snapshot this prefill (or an earlier
            // request) left, so a retry of the same conversation resumes there; else an empty lane
            let t_stop = t0.elapsed().as_secs_f64();
            let restored = match m.snaps.iter().filter(|s| s.pos <= m.tokens.len()).map(|s| s.pos).max() {
                Some(pos) => {
                    let i = m.snaps.iter().position(|s| s.pos == pos).unwrap();
                    let sn = m.snaps.remove(i);
                    let r = m.snap_restore(&sn);
                    m.snaps.insert(i, sn);
                    m.snaps.retain(|s| s.pos <= pos);
                    r.map(|_| pos)
                }
                None => m.reset().map(|_| 0),
            };
            match restored {
                Ok(pos) => tracing::info!("fnext: lane {} prompt stopped after {:.2} s (client gone), lane kept at {pos} tokens",
                    m.lane(), t_stop),
                Err(e) => {
                    tracing::warn!("fnext: lane {}: restore after a cancelled prompt: {e:#}", m.lane());
                    let _ = m.reset();
                }
            }
            sink.finish(Finish::new(FinishReason::Cancelled));
            return Ok(None);
        }
        Err(e) => {
            sink.finish(Finish::new(FinishReason::Error(format!("{e:#}"))));
            return Err(e);
        }
    };
    let mut a = Active { req, sink, sampler, greedy, answer_greedy, guard, spec: None, mtp_live: spec, hist: vec![], last: first,
        produced: 0, cached, prompt_n: rest.len(), t_first: 0.0, t_gen: Instant::now(), d0: m.spec_stats, force: None };
    // Constrained opening (tool_choice required): the reply starts with one of the offered sequences, chosen token by token from
    // the model's logits among the candidates still possible. Each chosen token is fed as one more prompt row (the MTP block stays
    // in step), so the free generation after it starts exactly like any other request.
    let (mut first, mut d1) = (first, d1);
    let mut opening: Vec<u32> = vec![];
    let mut cands: Vec<Vec<u32>> = a.req.params.prefix_choices.iter().filter(|c| !c.is_empty()).cloned().collect();
    while !cands.is_empty() && !cands.iter().any(|c| c.len() == opening.len()) {
        let depth = opening.len();
        let mut lg = vec![f32::NEG_INFINITY; m.logits().len()];
        for c in &cands {
            let t = c[depth] as usize;
            if t < lg.len() {
                lg[t] = m.logits()[t];
            }
        }
        let t = if a.greedy { crate::sampler::argmax(&lg) } else { a.sampler.sample(&mut lg) };
        cands.retain(|c| c[depth] == t);
        if cands.is_empty() {
            break;
        }
        opening.push(t);
        m.want_logits = true;
        match m.prefill_cut(&[t], &[], spec, |_| Ok(())) {
            Ok(v) => (first, d1) = v,
            Err(e) => {
                a.sink.finish(Finish::new(FinishReason::Error(format!("{e:#}"))));
                return Err(e);
            }
        }
    }
    a.t_first = t0.elapsed().as_secs_f64() * 1000.0;
    a.t_gen = Instant::now();
    if !opening.is_empty() {
        if let Some(r) = emit(&mut a, &opening) {
            done(m, a, r);
            return Ok(None);
        }
    }
    if !pick_finite(&a, m.logits(), first, m.bad_rows & 1 != 0) {
        let lane = m.lane();
        fail_nonfinite(m, lane, a);
        return Ok(None);
    }
    let engine_first = first;
    let first = match a.force.take() {
        Some(f) => f,
        None => pick(&mut a, m.logits(), first),
    };
    if spec && a.greedy {
        // a first token the engine did not choose (the stop ban's, a forced one): the draft was made for the model's
        // own choice, so the MTP block drafts again for the token that was written
        let d = if first == engine_first { d1.unwrap() } else { m.mtp_draft_for(first)? };
        a.spec = Some(Spec { cur: first, drafts: vec![d] });
    }
    if let Some(r) = emit(&mut a, &[first]) {
        done(m, a, r);
        return Ok(None);
    }
    Ok(Some(a))
}

/// The engine loop: up to n_lanes requests at once. New requests are admitted between steps into a free lane (the
/// one whose cached sequence shares the most with the prompt); every step then decodes all live lanes in one batched
/// graph (a lone greedy request that was never batched drafts with MTP). A request longer than a lane's share of
/// the KV pool waits until the others finish and runs alone with the whole pool.
fn schedule(m: &mut Fast, rx: mpsc::Receiver<(GenRequest, TokenSink)>) {
    let n = m.n_lanes();
    let pool = m.ctx_cap;
    let cap = if n > 1 { (pool / n / 8192 * 8192).max(8192) } else { pool };
    if n > 1 {
        tracing::info!("fnext: {n} lanes, {cap} cells each (one {pool}-cell pool for a request that runs alone)");
    }
    let mut act: Vec<Option<Active>> = (0..n).map(|_| None).collect();
    let mut queue: std::collections::VecDeque<(GenRequest, TokenSink)> = Default::default();
    let mut solo = false;
    // parked conversations (FNEXT_PARK_GB of host memory, default 4; least recently parked out first)
    let park_budget = (envu("FNEXT_PARK_GB", 4) as usize) << 30;
    let min = envu("FNEXT_CACHE_MIN", 1024);
    let mut parked: Vec<fe_fnext::fast::Parked> = vec![];
    // drafts per multi-lane speculative step, learned online (FNEXT_DRAFTS = the most, and a lone lane's count)
    let mut tuner = fe_fnext::fast::DraftTuner::new(envu("FNEXT_DRAFTS", 4));
    let mut tuner_log = Instant::now();
    let mut mem_log = Instant::now();
    loop {
        // host memory held by the prompt cache, every 10 minutes (parked conversations are capped by FNEXT_PARK_GB)
        if mem_log.elapsed().as_secs() >= envu("FNEXT_MEM_LOG_S", 600) as u64 {
            mem_log = Instant::now();
            let (pn, pb) = fe_fnext::fast::snap_pool_bytes();
            let lane_snaps: usize = (0..n).map(|i| m.lane_snap_bytes(i)).sum();
            tracing::info!("fnext: host memory: {} parked conversations {} MiB (cap {} MiB; KV rows {} MiB), lane \
                snapshots {} MiB, {pn} free snapshot buffers {} MiB, {} step graphs; RssAnon {} MiB (Rust heap {} MiB)", parked.len(),
                parked.iter().map(|p| p.bytes()).sum::<usize>() >> 20, park_budget >> 20,
                parked.iter().map(|p| p.kv_bytes()).sum::<usize>() >> 20, lane_snaps >> 20, pb >> 20, m.n_lane_graphs(), rss_anon_mib(), crate::rust_heap_bytes() >> 20);
        }
        if act.iter().all(|a| a.is_none()) && queue.is_empty() {
            match rx.recv() {
                Ok(x) => queue.push_back(x),
                Err(_) => return,
            }
        }
        while let Ok(x) = rx.try_recv() {
            queue.push_back(x);
        }
        // ---- admission ----
        while !solo && !queue.is_empty() {
            let free: Vec<usize> = (0..n).filter(|&i| act[i].is_none()).collect();
            if free.is_empty() {
                break;
            }
            let need = { let (r, _) = queue.front().unwrap(); r.prompt_ids.len() + r.params.max_tokens + 64 };
            let busy = act.iter().any(|a| a.is_some());
            let long = n > 1 && need > cap;
            if long && busy {
                break;
            }
            let lane = {
                let (r, _) = queue.front().unwrap();
                *free.iter().max_by_key(|&&i| m.lane_tokens(i).iter().zip(&r.prompt_ids).take_while(|(a, b)| a == b).count())
                    .unwrap()
            };
            let (req, sink) = queue.pop_front().unwrap();
            let id = req.id.clone();
            if let Err(e) = m.select_lane(lane) {
                tracing::error!("fnext: lane {lane}: {e:#}");
                sink.finish(Finish::new(FinishReason::Error(format!("{e:#}"))));
                continue;
            }
            if long {
                for i in 0..n {
                    if i != lane {
                        let _ = m.trim_lane(i);
                    }
                }
                solo = true;
            }
            m.ctx_cap = if long { pool } else { cap };
            // the lane's own conversation is no use to this request: park it; a parked one that is, comes back
            let lcp = |t: &[u32]| t.iter().zip(&req.prompt_ids).take_while(|(a, b)| a == b).count();
            let own = lcp(&m.tokens);
            if !long && park_budget > 0 {
                if own < min && !m.tokens.is_empty() {
                    match m.park() {
                        Ok(Some(p)) => {
                            parked.push(p);
                            let mut tot: usize = parked.iter().map(|p| p.bytes()).sum();
                            while tot > park_budget && parked.len() > 1 {
                                tot -= parked.remove(0).bytes();
                            }
                        }
                        Ok(None) => {}
                        Err(e) => tracing::warn!("fnext: park: {e:#}"),
                    }
                }
                let own = lcp(&m.tokens);
                let best = parked.iter().enumerate().map(|(i, p)| (i, lcp(&p.tokens))).max_by_key(|x| x.1);
                if let Some((i, _)) = best.filter(|&(_, l)| l >= min && l > own) {
                    let p = parked.remove(i);
                    if let Err(e) = m.unpark(p) {
                        tracing::warn!("fnext: unpark: {e:#}");
                        let _ = m.reset();
                    }
                }
            }
            let alone = !busy && queue.is_empty();
            match start(m, req, sink, alone) {
                Ok(Some(a)) => act[lane] = Some(a),
                Ok(None) => {}
                Err(e) => {
                    tracing::error!("fnext: request {id}: {e:#}");
                    let _ = m.reset();
                }
            }
            if long {
                break;
            }
        }
        // ---- one step ----
        let live: Vec<usize> = (0..n).filter(|&i| act[i].is_some()).collect();
        if live.is_empty() {
            solo = false;
            continue;
        }
        let all_spec = live.iter().all(|&i| act[i].as_ref().unwrap().spec.is_some());
        // a lone lane in a sampled stretch whose answer is greedy (a sampled reasoning block, the boosted stretch after
        // an answer loop): it steps with the MTP block kept in step, so drafting resumes when the stretch ends
        let keep_mtp = !all_spec && live.len() == 1 && {
            let a = act[live[0]].as_ref().unwrap();
            a.mtp_live && a.answer_greedy && m.has_mtp()
        };
        if all_spec {
            // greedy lanes with MTP: one speculative step over all of them (one lane: spec_step); never the graphs that
            // copy logits (a sampled request that started and ended at its first token leaves the flag up)
            m.want_logits = false;
            let nd = tuner.pick(live.len());
            let t_step = Instant::now();
            let mut taken: Vec<Spec> = live.iter().map(|&i| act[i].as_mut().unwrap().spec.take().unwrap()).collect();
            // a lane whose guard asked for a token of the engine's own (the forced `</think>`) verifies its last token
            // alone in this step and drafts again in the next
            let force: Vec<Option<u32>> = live.iter().map(|&i| act[i].as_mut().unwrap().force.take()).collect();
            let mut scans: Vec<Option<Scan>> = live.iter().map(|_| None).collect();
            let mut bad = vec![false; live.len()];
            let r = {
                let mut items: Vec<(usize, &mut Spec)> = live.iter().copied().zip(taken.iter_mut()).collect();
                // every lane's guard sees its tokens one at a time; what it cuts is given back (the lane keeps fewer rows)
                m.spec_lanes_ctl(&mut items, nd, &force, &mut |k, out, finite| {
                    if test_nonfinite() || (finite_check() && !finite) {
                        bad[k] = true;
                        out.truncate(1);
                        return;
                    }
                    scans[k] = Some(scan(act[live[k]].as_mut().unwrap(), out));
                })
            };
            for (k, sp) in taken.into_iter().enumerate() {
                act[live[k]].as_mut().unwrap().spec = Some(sp);
            }
            match r {
                Ok(got) => {
                    tuner.record(got.iter().map(|g| g.len()).sum(), t_step.elapsed().as_secs_f64());
                    if tuner_log.elapsed().as_secs() >= 300 && live.len() > 1 {
                        tracing::info!("fnext: draft tuner (lanes, drafts, tok/s): {:?}", tuner.summary());
                        tuner_log = Instant::now();
                    }
                    for (k, &i) in live.iter().enumerate() {
                        if bad[k] {
                            let a = act[i].take().unwrap();
                            fail_nonfinite(m, i, a);
                            continue;
                        }
                        let Some(sc) = scans[k].take() else { continue };
                        let a = act[i].as_mut().unwrap();
                        if let Some(reason) = apply(a, sc) {
                            let a = act[i].take().unwrap();
                            done(m, a, reason);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("fnext: speculative step over lanes {live:?}: {e:#}");
                    for &i in &live {
                        let a = act[i].take().unwrap();
                        a.sink.finish(Finish::new(FinishReason::Error(format!("{e:#}"))));
                        if m.select_lane(i).is_ok() {
                            let _ = m.reset();
                        }
                    }
                }
            }
        } else if keep_mtp {
            let i = live[0];
            let last = act[i].as_ref().unwrap().last;
            m.want_logits = true;
            let r = m.select_lane(i).and_then(|_| m.prefill_mtp(&[last]));
            match r {
                Ok((tg, d1)) => {
                    let a = act[i].as_mut().unwrap();
                    if a.force.is_none() && !pick_finite(a, m.logits(), tg, m.bad_rows & 1 != 0) {
                        let a = act[i].take().unwrap();
                        fail_nonfinite(m, i, a);
                    } else {
                        let t = match a.force.take() {
                            Some(f) => f,
                            None => pick(a, m.logits(), tg),
                        };
                        match emit(a, &[t]) {
                            Some(reason) => {
                                let a = act[i].take().unwrap();
                                done(m, a, reason);
                            }
                            None if a.greedy && a.force.is_none() => {
                                // the stretch is over: draft again from the token just written
                                let d = if t == tg { Ok(d1) } else { m.mtp_draft_for(t) };
                                match d {
                                    Ok(d) => a.spec = Some(Spec { cur: t, drafts: vec![d] }),
                                    Err(e) => {
                                        tracing::warn!("fnext: lane {i}: no draft after a sampled stretch ({e:#}): plain steps");
                                        a.mtp_live = false;
                                    }
                                }
                            }
                            None => {}
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("fnext: step on lane {i}: {e:#}");
                    let a = act[i].take().unwrap();
                    a.sink.finish(Finish::new(FinishReason::Error(format!("{e:#}"))));
                    if m.select_lane(i).is_ok() {
                        let _ = m.reset();
                    }
                }
            }
        } else {
            // a sampled lane among several, or a lane without the MTP block: every live lane steps one token at a
            // time in one batched graph, and the MTP block of each falls behind (no drafts for the rest of those requests)
            for &i in &live {
                let a = act[i].as_mut().unwrap();
                a.spec = None;
                a.mtp_live = false;
            }
            m.want_logits = live.iter().any(|&i| act[i].as_ref().unwrap().needs_logits());
            let items: Vec<(usize, u32)> = live.iter().map(|&i| (i, act[i].as_ref().unwrap().last)).collect();
            match m.step_lanes(&items) {
                Ok(r) => {
                    let dev_bad = m.bad_rows;
                    for (k, &i) in live.iter().enumerate() {
                        let lg = if m.want_logits {
                            if live.len() > 1 { m.lane_logits(k).to_vec() } else { m.logits().to_vec() }
                        } else {
                            vec![]
                        };
                        let a = act[i].as_mut().unwrap();
                        // (a greedy lane beside a sampled one has its row on the host too; its pick stays the device argmax)
                        if a.force.is_none() && !pick_finite(a, &lg, r[k], dev_bad >> k & 1 != 0) {
                            let a = act[i].take().unwrap();
                            fail_nonfinite(m, i, a);
                            continue;
                        }
                        let t = match a.force.take() {
                            Some(f) => f,
                            None => pick(a, &lg, r[k]),
                        };
                        if let Some(reason) = emit(a, &[t]) {
                            let a = act[i].take().unwrap();
                            done(m, a, reason);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("fnext: batched step over lanes {live:?}: {e:#}");
                    for &i in &live {
                        let a = act[i].take().unwrap();
                        a.sink.finish(Finish::new(FinishReason::Error(format!("{e:#}"))));
                        if m.select_lane(i).is_ok() {
                            let _ = m.reset();
                        }
                    }
                }
            }
        }
        if solo && act.iter().all(|a| a.is_none()) {
            solo = false;
        }
    }
}

/// This process's resident anonymous memory (MiB), from /proc/self/status.
fn rss_anon_mib() -> usize {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|t| t.lines().find(|l| l.starts_with("RssAnon:"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok())).map_or(0, |k| k >> 10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{CancelToken, GenEvent};

    const TE: u32 = 900;
    const STOP: u32 = 7;

    /// A request as `start` sets it up, without an engine.
    fn active(p: SamplingParams) -> (Active, tokio::sync::mpsc::UnboundedReceiver<GenEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let req = GenRequest { id: "t".into(), prompt_ids: vec![1, 2, 3], snapshots: vec![], params: p, cancel: CancelToken::new() };
        let (guard, mode) = match PhaseGuard::new(&req.params, |q| Mode(q.clone())) {
            Some((g, s)) => (Some(g), s),
            None => (None, Mode(req.params.clone())),
        };
        let sampler = Sampler::new(&mode.0, &req.prompt_ids);
        let greedy = sampler.is_plain_greedy();
        let answer_greedy = plain_greedy(&req.params);
        let a = Active { req, sink: TokenSink::new(tx), sampler, greedy, answer_greedy, guard, spec: None, mtp_live: false, hist: vec![],
            last: 0, produced: 0, cached: 0, prompt_n: 3, t_first: 0.0, t_gen: Instant::now(), d0: (0, 0), force: None };
        (a, rx)
    }

    fn greedy() -> SamplingParams {
        SamplingParams { temperature: 0.0, max_tokens: 100, stop_token_ids: vec![STOP], ..Default::default() }
    }

    #[test]
    fn a_request_without_a_guard_takes_the_run_up_to_a_stop_token_or_the_limit() {
        let (mut a, _rx) = active(greedy());
        let mut cand = vec![10, 11, 12];
        let sc = scan(&mut a, &mut cand);
        assert!(sc.out == vec![10, 11, 12] && sc.fin.is_none() && sc.act == Act::None && cand.len() == 3);
        // a stop token ends the request; it is not sent and the run is not cut for the engine
        let mut cand = vec![13, STOP, 14];
        let sc = scan(&mut a, &mut cand);
        assert!(sc.out == vec![13] && sc.fin == Some(FinishReason::Stop(STOP)) && cand.len() == 3);
        // the first token of a reply is not taken as a stop by `scan` only from min_tokens on
        let (mut a, _rx) = active(SamplingParams { max_tokens: 2, ..greedy() });
        let mut cand = vec![20, 21, 22];
        let sc = scan(&mut a, &mut cand);
        assert!(sc.out == vec![20, 21] && sc.fin == Some(FinishReason::Length) && a.produced == 2 && a.last == 21);
    }

    #[test]
    fn the_guard_is_fed_one_token_at_a_time_and_the_rest_of_the_run_is_given_back() {
        // reasoning cap 3: of five verified tokens the request takes three, the engine writes the block end next
        let (mut a, mut rx) = active(SamplingParams { think_end: Some(TE), max_reasoning: 3, ..greedy() });
        let mut cand = vec![10, 11, 12, 13, 14];
        let sc = scan(&mut a, &mut cand);
        assert!(sc.out == vec![10, 11, 12] && sc.act == Act::Force(TE) && sc.fin.is_none());
        assert_eq!(cand, vec![10, 11, 12]);
        assert!(apply(&mut a, sc).is_none() && a.force == Some(TE) && a.last == 12 && a.produced == 3);
        assert_eq!(rx.try_recv().ok(), Some(GenEvent::Tokens(vec![10, 11, 12])));
        // the forced token is fed like any other; the same sampling in both phases: nothing else happens
        assert!(emit(&mut a, &[TE]).is_none() && a.greedy);
        let ev = a.guard.as_ref().unwrap().events();
        assert!(ev.reasoning_capped && ev.reasoning_tokens == 4);
        // the same tokens one step at a time: the same answer at the same token (no dependence on the rows of a step)
        let (mut b, _rx) = active(SamplingParams { think_end: Some(TE), max_reasoning: 3, ..greedy() });
        let acts: Vec<bool> = [10u32, 11, 12].iter().map(|&t| { let sc = scan(&mut b, &mut vec![t]); sc.act == Act::Force(TE) }).collect();
        assert_eq!(acts, vec![false, false, true]);
    }

    #[test]
    fn a_sampled_reasoning_block_hands_over_to_a_greedy_answer_at_the_block_end() {
        let mut p = SamplingParams { think_end: Some(TE), ..greedy() };
        let mut r = p.clone();
        r.temperature = 1.0;
        p.reasoning = Some(Box::new(r));
        let (mut a, _rx) = active(p);
        assert!(!a.greedy && a.answer_greedy && a.needs_logits());
        let mut cand = vec![10, TE, 11];
        let sc = scan(&mut a, &mut cand);
        assert!(sc.out == vec![10, TE] && matches!(sc.act, Act::Sampling(_)) && cand.len() == 2);
        assert!(apply(&mut a, sc).is_none() && a.greedy && a.hist == vec![10, TE]);
    }

    #[test]
    fn an_answer_cycle_boosts_a_greedy_request_then_ends_it() {
        let mut p = SamplingParams { loop_breaker: true, max_tokens: 10_000, ..greedy() };
        p.loop_rules.answer = crate::loopdet::LoopRule { min_repeats: 4, min_repeated_tokens: 150 };
        let (mut a, _rx) = active(p);
        a.spec = Some(Spec { cur: 0, drafts: vec![] });
        let mut ended = None;
        let mut boosted = None;
        for i in 0..2000u32 {
            let r = emit(&mut a, &[10 + i % 5]);
            if boosted.is_none() && !a.greedy {
                // the boosted stretch samples: no drafts
                assert!(a.spec.is_none());
                boosted = Some(i);
            }
            if r.is_some() {
                ended = Some((i, r.unwrap()));
                break;
            }
        }
        assert!(boosted.is_some_and(|i| i < 200));
        assert!(matches!(ended, Some((_, FinishReason::Length))));
        assert_eq!(a.guard.as_ref().unwrap().events().loop_breaks, 2);
    }

    #[test]
    fn the_first_token_is_never_a_stop_token() {
        let (mut a, _rx) = active(greedy());
        let mut lg = vec![0f32; 32];
        lg[STOP as usize] = 9.0;
        lg[20] = 5.0;
        assert_eq!(pick(&mut a, &lg, STOP), 20);
        assert_eq!(pick(&mut a, &lg, 21), 21);
        emit(&mut a, &[20]);
        // after the first token the model's own choice stands
        assert_eq!(pick(&mut a, &lg, STOP), STOP);
        // a row with a NaN or an infinity is not picked from
        let (a, _rx) = active(SamplingParams { temperature: 0.7, ..greedy() });
        assert!(pick_finite(&a, &lg, 3, false));
        lg[5] = f32::NAN;
        assert!(!pick_finite(&a, &lg, 3, false));
        lg[5] = f32::INFINITY;
        assert!(!pick_finite(&a, &lg, 3, false));
    }
}
