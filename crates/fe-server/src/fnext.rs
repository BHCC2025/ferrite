//! `--engine fnext`: Qwen3.8-Flash-Next (GGUF, qwen4exp) on one or two cards (FNEXT_GPUS=2 layer split) through
//! fe-fnext's fast path (cargo feature `fnext`). An engine thread owns the CUDA contexts and runs up to FNEXT_SLOTS
//! requests at once on lanes (continuous batching: a new request prefills between steps, then joins the batched
//! decode). Each lane keeps its sequence between requests: a prompt that extends it (a chat follow-up) only runs its
//! new tokens. Greedy requests decode with MTP drafts (FNEXT_MTP=1, default on), several lanes at once through
//! spec_lanes; while any live request samples, every live lane steps one token at a time.
use std::sync::mpsc;
use std::time::Instant;

use anyhow::Result;
use fe_fnext::fast::{Fast, Spec};

use crate::engine::{Engine, EngineInfo, Finish, FinishReason, GenRequest, Timings, TokenSink};
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
    // every graph a request can use: plain rows 1..8, verify rows 1..8, commits 1..8, MTP rows 1..8, the prompt path
    let toks: Vec<u32> = (0..200u32).map(|i| 1000 + i * 37).collect();
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

/// A request decoding on a lane.
struct Active {
    req: GenRequest,
    sink: TokenSink,
    sampler: Sampler,
    greedy: bool,
    /// MTP drafting state; dropped for good once the lane takes a plain step (its MTP caches then miss positions)
    spec: Option<Spec>,
    last: u32,
    produced: usize,
    cached: usize,
    prompt_n: usize,
    t_first: f64,
    t_gen: Instant,
    d0: (u64, u64),
}

/// Emit `toks`; Some(reason) when the request is done.
fn emit(a: &mut Active, toks: &[u32]) -> Option<FinishReason> {
    let p = &a.req.params;
    let mut out = vec![];
    for &t in toks {
        if a.produced >= p.min_tokens && p.stop_token_ids.contains(&t) {
            if !out.is_empty() {
                a.sink.send_tokens(std::mem::take(&mut out));
            }
            return Some(FinishReason::Stop(t));
        }
        a.sampler.accept(t);
        out.push(t);
        a.produced += 1;
        a.last = t;
        if a.produced >= p.max_tokens {
            a.sink.send_tokens(out);
            return Some(FinishReason::Length);
        }
    }
    if !a.sink.send_tokens(out) || a.req.cancel.is_cancelled() {
        return Some(FinishReason::Cancelled);
    }
    None
}

/// The next token from a logit row (greedy: the engine's argmax).
fn pick(a: &mut Active, lg: &[f32], greedy_tok: u32) -> u32 {
    if a.greedy {
        return greedy_tok;
    }
    let p = &a.req.params;
    let mut lg = lg.to_vec();
    if a.produced < p.min_tokens {
        for &t in &p.stop_token_ids {
            if let Some(l) = lg.get_mut(t as usize) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
    a.sampler.sample(&mut lg)
}

fn done(m: &Fast, a: Active, reason: FinishReason) {
    let mut f = Finish::new(reason);
    f.cached_prompt_tokens = a.cached;
    let d1s = m.spec_stats;
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
    let sampler = Sampler::new(&req.params, &prompt);
    let greedy = sampler.is_plain_greedy();
    // MTP for every greedy request (spec_lanes batches several); `alone` only matters to the long-request path
    let _ = alone;
    let spec = greedy && m.has_mtp() && std::env::var("FNEXT_SPEC").map_or(true, |v| v != "0");
    m.want_logits = !greedy;
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
    let mut a = Active { req, sink, sampler, greedy, spec: None, last: first, produced: 0, cached,
        prompt_n: rest.len(), t_first: 0.0, t_gen: Instant::now(), d0: m.spec_stats };
    let lg = m.logits().to_vec();
    let first = pick(&mut a, &lg, first);
    a.t_first = t0.elapsed().as_secs_f64() * 1000.0;
    a.t_gen = Instant::now();
    if spec {
        a.spec = Some(Spec { cur: first, drafts: vec![d1.unwrap()] });
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
        if all_spec {
            // greedy lanes with MTP: one speculative step over all of them (one lane: spec_step)
            let nd = tuner.pick(live.len());
            let t_step = Instant::now();
            let mut taken: Vec<Spec> = live.iter().map(|&i| act[i].as_mut().unwrap().spec.take().unwrap()).collect();
            let r = {
                let mut items: Vec<(usize, &mut Spec)> = live.iter().copied().zip(taken.iter_mut()).collect();
                m.spec_lanes(&mut items, nd)
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
                        let a = act[i].as_mut().unwrap();
                        if let Some(reason) = emit(a, &got[k]) {
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
        } else {
            for &i in &live {
                act[i].as_mut().unwrap().spec = None;
            }
            m.want_logits = live.iter().any(|&i| !act[i].as_ref().unwrap().greedy);
            let items: Vec<(usize, u32)> = live.iter().map(|&i| (i, act[i].as_ref().unwrap().last)).collect();
            match m.step_lanes(&items) {
                Ok(r) => {
                    for (k, &i) in live.iter().enumerate() {
                        let lg = if m.want_logits {
                            if live.len() > 1 { m.lane_logits(k).to_vec() } else { m.logits().to_vec() }
                        } else {
                            vec![]
                        };
                        let a = act[i].as_mut().unwrap();
                        let t = pick(a, &lg, r[k]);
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
