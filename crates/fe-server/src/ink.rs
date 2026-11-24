//! `--engine ink`: Inkling-2-Mini on one NVIDIA card (crates/fe-ink). One engine thread owns the model; requests run one at a
//! time (max_batch 1). A prompt that starts with tokens an earlier request prefilled resumes from the prompt cache (fe_ink::pcache;
//! INK_PCACHE=0 turns it off) and prefills only the rest. Prompt rows go through the prompt (tensor-core) family in chunks, generated tokens through the decode
//! graph. Greedy requests take the argmax of the bf16 logits (the final softcap is monotone), sampled ones the softcapped f32
//! row through `crate::sampler`.
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use fe_ink::engine::Ink;
use fe_ink::pcache::PCache;
use fe_ink::sampler::{DevSampler, PenHist};
use fe_ink::spec::{argmax, rng_next, SampleCfg, Spec};

use crate::engine::{Engine, EngineInfo, Finish, FinishReason, GenRequest, Timings, TokenSink};

pub(crate) struct Job {
    pub(crate) req: GenRequest,
    pub(crate) sink: TokenSink,
}

pub struct InkEngine {
    tx: Mutex<mpsc::Sender<Job>>,
    ctx: usize,
    lanes: usize,
    dead: Arc<Mutex<Option<String>>>,
}

impl InkEngine {
    /// Load the model on the visible card (CUDA_VISIBLE_DEVICES) and start the engine thread. `ctx` = context window.
    pub fn start(model_dir: PathBuf, ctx: usize) -> Result<InkEngine> {
        let (tx, rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<usize, String>>();
        let dead = Arc::new(Mutex::new(None));
        let dead2 = dead.clone();
        let chunk: usize = std::env::var("INK_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
        std::thread::Builder::new().name("ink-engine".into()).spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                let mut ink = match Ink::load(&model_dir, 0, ctx, chunk) {
                    Ok(i) => i,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("{e:#}")));
                        return Ok(());
                    }
                };
                // INK_CHUNK=0: the chunk was sized from the free memory after the weights
                let chunk = ink.tmax;
                // INK_DRAFT_MAX_CTX=n: the draft head is only resident for requests whose prompt + max_tokens fit n tokens (on the 16 GB card it costs ~0.45 GiB plus
                // 4.5 KiB a position of context and pays only up to ~48K tokens); a longer request unloads it (its memory goes back to the experts through
                // fit_host_experts), the next short request loads it again.
                let max_head: usize = std::env::var("INK_DRAFT_MAX_CTX").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(usize::MAX);
                let head_allowed = !fe_ink::env_flag("INK_NODRAFT");
                let load_head = |ink: &mut Ink| -> Option<Spec> {
                    let fold = match fe_ink::fold::ensure(&model_dir) {
                        Ok(f) => f,
                        Err(e) => { tracing::warn!("ink engine: draft head fold tables unavailable ({e:#}), plain decode"); ink.draft_taps = false; return None; }
                    };
                    match Spec::load(ink, &model_dir, &fold) {
                        Ok(sp) => Some(sp),
                        Err(e) => { tracing::warn!("ink engine: draft head not loaded ({e:#}), plain decode"); ink.draft_taps = false; None }
                    }
                };
                // the model's own draft head (lossless: greedy and seeded sampled requests give the tokens plain decode gives); INK_NODRAFT=1 turns it off
                let mut spec = if head_allowed { load_head(&mut ink) } else { None };
                let mut head_failed = head_allowed && spec.is_none();
                // device sampler for plain (non-drafted) decode: the same distribution code as the draft loop, no host softmax
                let mut dsamp = DevSampler::new(ink.ctx.clone())?;
                let mut pc = PCache::new(&ink);
                if pc.on {
                    tracing::info!("ink engine: prompt cache on: snapshots of {:.0} MiB in pinned host memory, up to {} ({:.2} GiB, INK_PCACHE_GB), marks at the prompt end and every {} tokens (INK_PCACHE_EVERY); INK_PCACHE=0 turns it off",
                        pc.snap_bytes() as f64 / 1048576.0, pc.max_snaps, (pc.max_snaps * pc.snap_bytes()) as f64 / 1073741824.0, pc.every);
                } else {
                    tracing::info!("ink engine: prompt cache off (INK_PCACHE=0)");
                }
                // INK_LANES > 1: several requests at once (crate::ink_lanes); the draft head stays as loaded
                if ink.n_lanes() > 1 {
                    let _ = ready_tx.send(Ok(ink.n_lanes()));
                    return crate::ink_lanes::run_lanes(&mut ink, spec.as_mut(), &mut dsamp, &rx);
                }
                let _ = ready_tx.send(Ok(1));
                while let Ok(job) = rx.recv() {
                    let need = job.req.prompt_ids.len() + job.req.params.max_tokens;
                    if spec.is_some() && need > max_head {
                        tracing::info!("ink engine: request of up to {need} tokens: unloading the draft head");
                        spec = None;
                        ink.draft_taps = false;
                        let _ = ink.s.sync();
                        if let Err(e) = ink.reset() { tracing::warn!("ink engine: reset after unloading the head: {e:#}"); }
                        pc.clear();
                    } else if spec.is_none() && head_allowed && !head_failed && need <= max_head {
                        tracing::info!("ink engine: request of up to {need} tokens: loading the draft head");
                        spec = load_head(&mut ink);
                        head_failed = spec.is_none();
                        // the cached snapshots carry no draft context: start over
                        pc.clear();
                    }
                    let _ = chunk;
                    run(&mut ink, spec.as_mut(), &mut dsamp, &mut pc, job);
                }
                Ok(())
            }));
            let why = match r {
                Ok(Ok(())) => "engine thread ended".to_string(),
                Ok(Err(e)) => format!("engine error: {e:#}"),
                Err(_) => "engine thread panicked".to_string(),
            };
            *dead2.lock().unwrap() = Some(why);
        })?;
        let lanes = match ready_rx.recv() {
            Ok(Ok(n)) => n,
            Ok(Err(e)) => anyhow::bail!("ink engine: {e}"),
            Err(_) => anyhow::bail!("ink engine thread died while loading"),
        };
        Ok(InkEngine { tx: Mutex::new(tx), ctx, lanes, dead })
    }
}

impl Engine for InkEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "ink (Inkling-2-Mini)".into(), max_batch: self.lanes, max_context: self.ctx }
    }

    fn submit(&self, req: GenRequest, sink: TokenSink) {
        if let Err(e) = self.tx.lock().unwrap().send(Job { req, sink }) {
            e.0.sink.finish(Finish::new(FinishReason::Error("the ink engine is not running".into())));
        }
    }

    fn dead(&self) -> Option<String> {
        self.dead.lock().unwrap().clone()
    }
}

fn run(ink: &mut Ink, mut spec: Option<&mut Spec>, dsamp: &mut DevSampler, pc: &mut PCache, job: Job) {
    let Job { req, sink } = job;
    let p = &req.params;
    let fin0 = |reason: FinishReason, t: Option<Timings>| Finish { reason, cached_prompt_tokens: 0, timings: t, events: Default::default() };
    let n = req.prompt_ids.len();
    if n == 0 || n >= ink.cap {
        return sink.finish(fin0(FinishReason::Error(format!("prompt of {n} tokens does not fit the context {}", ink.cap)), None));
    }
    let t0 = Instant::now();
    // speculative decoding for every request except min_tokens (banned stop ids): greedy is bit-identical to plain decode, sampled requests (temperature,
    // top-p / top-k / min-p, penalties) give the tokens plain decode draws for the seed (each verified row draws as plain decode does, fe_ink::spec)
    let use_spec = spec.is_some() && p.min_tokens == 0 && n + p.max_tokens + 16 < ink.cap;
    let mut rng: u64 = p.seed.unwrap_or_else(crate::util::random_u64) | 1;
    let cfg = SampleCfg { temperature: p.temperature, top_k: p.top_k as u32, top_p: p.top_p, min_p: p.min_p, rep: p.repetition_penalty, pres: p.presence_penalty,
        freq: p.frequency_penalty };
    // the draft context is fed by every prefill while the head is loaded and the cache is on (so every snapshot can resume a drafted
    // request), else only by drafted requests (as before the cache)
    let feed = use_spec || (pc.on && spec.is_some());
    let from = match pc.begin(ink, if feed { spec.as_deref_mut() } else { None }, &req.prompt_ids) {
        Ok(f) => f,
        Err(e) => return sink.finish(fin0(FinishReason::Error(format!("{e:#}")), None)),
    };
    let fin = |reason: FinishReason, t: Option<Timings>| Finish { reason, cached_prompt_tokens: from, timings: t, events: Default::default() };
    if !feed {
        ink.draft_taps = false;
    }
    if use_spec {
        spec.as_mut().unwrap().begin(cfg, &req.prompt_ids);
    }
    let mut first_chunk = true;
    let stop = || {
        if req.cancel.is_cancelled() || sink.is_closed() {
            return true;
        }
        if !first_chunk {
            sink.ping();
        }
        first_chunk = false;
        false
    };
    let logits: Vec<u16> = match pc.prefill(ink, if feed { spec.as_deref_mut() } else { None }, &req.prompt_ids, from, stop) {
        Ok(Some(l)) => l,
        Ok(None) => return sink.finish(fin(FinishReason::Cancelled, None)),
        Err(e) => return sink.finish(fin(FinishReason::Error(format!("{e:#}")), None)),
    };
    let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
    if pc.on {
        let st = &pc.last;
        tracing::info!("ink pcache: {from} of {n} prompt tokens from the cache (common prefix {}, restore {:.1} ms), {} prefilled in {:.0} ms; {} snapshots taken ({:.1} ms), {} held, {:.2} GiB pinned host",
            st.common, st.restore_ms, n - from, prompt_ms, st.snaps_taken, st.snap_ms, pc.len(), pc.host_bytes() as f64 / 1073741824.0);
    }
    if !use_spec {
        ink.draft_taps = false;
    }
    let t1 = Instant::now();
    let mut generated = 0usize;
    ink.prompt = false;
    if use_spec {
        let sp = spec.as_mut().unwrap();
        let mut cur = match sp.first_token(ink, &logits, &mut rng) {
            Ok(t) => t,
            Err(e) => return sink.finish(fin(FinishReason::Error(format!("{e:#}")), None)),
        };
        let mut pending: Vec<u32> = vec![cur];
        loop {
            // this request's drafting so far (the Spec's stats restart with each request): draft_n = drafted positions verified, draft_n_accepted = accepted ones
            let st = sp.stats.clone();
            let ds = Some(((st.rows - st.passes.min(st.rows)) as u64, st.accepted as u64));
            let timings = |g: usize| Timings { prompt_n: n - from, cache_n: from, prompt_ms, predicted_n: g, predicted_ms: t1.elapsed().as_secs_f64() * 1e3, draft: ds };
            let log_spec = |why: &str| tracing::info!("ink spec ({why}): {} passes, {} tokens ({:.2} a pass), {} drafts accepted of {} verified, {} plain tokens, {} suspensions", st.passes, st.tokens - st.plain_tokens,
                (st.tokens - st.plain_tokens) as f64 / st.passes.max(1) as f64, st.accepted, st.rows - st.passes.min(st.rows), st.plain_tokens, st.suspensions);
            let mut batch = vec![];
            for &tok in &pending {
                if p.stop_token_ids.contains(&tok) {
                    if !batch.is_empty() { let _ = sink.send_tokens(std::mem::take(&mut batch)); }
                    log_spec("stop");
                    return sink.finish(fin(FinishReason::Stop(tok), Some(timings(generated))));
                }
                batch.push(tok);
                generated += 1;
                cur = tok;
                if generated >= p.max_tokens { break; }
            }
            if !sink.send_tokens(batch) || req.cancel.is_cancelled() {
                log_spec("cancelled");
                return sink.finish(fin(FinishReason::Cancelled, Some(timings(generated))));
            }
            if generated >= p.max_tokens || n + generated + 16 >= ink.cap {
                log_spec("length");
                return sink.finish(fin(FinishReason::Length, Some(timings(generated))));
            }
            match sp.step(ink, cur, &mut rng) {
                Ok(v) => pending = v,
                Err(e) => return sink.finish(fin(FinishReason::Error(format!("{e:#}")), Some(timings(generated)))),
            }
        }
    }
    // plain decode (no drafting: min_tokens, INK_NODRAFT, draft head missing): CUDA-graph decode rows, tokens drawn on the device
    let mut hist = PenHist::default();
    hist.begin(&cfg, &req.prompt_ids);
    let banned_now = |generated: usize| -> Vec<u32> { if generated < p.min_tokens { p.stop_token_ids.clone() } else { vec![] } };
    let draw = |ink: &mut Ink, dsamp: &mut DevSampler, hist: &PenHist, rng: &mut u64, banned: &[u32], argmax_tok: u32| -> Result<u32> {
        if cfg.plain_greedy() && banned.is_empty() {
            Ok(argmax_tok)
        } else {
            dsamp.sample_one(&ink.s, ink.logits_ptr(), &cfg, hist, banned, || rng_next(rng))
        }
    };
    let mut tok = {
        let banned = banned_now(0);
        let r = if cfg.plain_greedy() && banned.is_empty() {
            Ok(argmax(&logits))
        } else {
            dsamp.upload_row(&ink.s, &logits).and_then(|ptr| dsamp.sample_one(&ink.s, ptr, &cfg, &hist, &banned, || rng_next(&mut rng)))
        };
        match r {
            Ok(t) => t,
            Err(e) => return sink.finish(fin(FinishReason::Error(format!("{e:#}")), None)),
        }
    };
    loop {
        let timings = |g: usize| Timings { prompt_n: n - from, cache_n: from, prompt_ms, predicted_n: g, predicted_ms: t1.elapsed().as_secs_f64() * 1e3, draft: None };
        if p.stop_token_ids.contains(&tok) {
            return sink.finish(fin(FinishReason::Stop(tok), Some(timings(generated))));
        }
        hist.push(tok);
        generated += 1;
        if !sink.send_tokens(vec![tok]) || req.cancel.is_cancelled() {
            return sink.finish(fin(FinishReason::Cancelled, Some(timings(generated))));
        }
        if generated >= p.max_tokens || n + generated >= ink.cap {
            return sink.finish(fin(FinishReason::Length, Some(timings(generated))));
        }
        let next = match ink.decode_argmax(tok) {
            Ok(g) => draw(ink, dsamp, &hist, &mut rng, &banned_now(generated), g),
            Err(e) => Err(e),
        };
        match next {
            Ok(t) => tok = t,
            Err(e) => return sink.finish(fin(FinishReason::Error(format!("{e:#}")), Some(timings(generated)))),
        }
    }
}
