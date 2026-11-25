//! `--engine ink` with INK_LANES > 1: several requests decode at once. Every lane is a sequence of the engine (own K/V, rings, conv
//! tails, draft context, drafting policy, sampler state and prompt cache), so one forward carries the rows of every lane
//! (`Ink::forward_multi`): the dense and expert weights are read once for all of them, each lane's rows attend to its own caches, and
//! every row has the bits of the same row run alone.
//!
//! Scheduling, one forward at a time:
//! - requests are given lanes in arrival order: the free lane whose prompt cache resumes the longest prefix, else the least recently used
//!   free lane (its cached conversation is replaced). A request waits when no lane is free or when the K/V it may need (prompt +
//!   max_tokens) does not fit next to the other lanes' (idle lanes' cached conversations are dropped first, least recently used first);
//!   with every other lane idle a request always starts, so one stream keeps the whole context of the single-lane engine;
//! - decoding lanes bring their rows each pass (the last token, plus the drafts when the lane drafts); one prefilling lane (the oldest)
//!   adds a chunk of at most INK_LANE_PREFILL_ROWS rows (128; 256 on an integrated GPU) to the same forward, so a long prompt delays the others by one chunk per
//!   pass, not by its whole prefill; with nobody decoding it takes the full chunks of the single-lane path;
//! - a lone decoding lane takes the single-lane path (CUDA graph), so one stream runs at the single-lane speed.
//! Per lane the request runs exactly as in `crate::ink::run`: the same prompt cache, the same first token, the same drafting and
//! sampling code (`Spec::plan` / `Spec::finish` around the shared forward), the same stop / length / cancel rules.
use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::Result;
use fe_ink::engine::Ink;
use fe_ink::pcache::{Chunk, PCache, PrefillRun};
use fe_ink::sampler::{DevSampler, PenHist};
use fe_ink::spec::{argmax, rng_next, Plan, SampleCfg, Spec};

use crate::engine::{Finish, FinishReason, Timings};
use crate::ink::Job;

enum Phase {
    Prefill(PrefillRun),
    Decode,
}

struct Act {
    job: Job,
    n: usize,
    from: usize,
    use_spec: bool,
    feed: bool,
    cfg: SampleCfg,
    rng: u64,
    hist: PenHist,
    phase: Phase,
    t0: Instant,
    prompt_ms: f64,
    t1: Instant,
    generated: usize,
    cur: u32,
    /// tokens reserved for this request's K/V (prompt + max_tokens, capped by the context)
    need: usize,
}

struct Lane {
    pc: PCache,
    act: Option<Act>,
    used: u64,
}

fn env_usize(n: &str) -> Option<usize> {
    std::env::var(n).ok().and_then(|v| v.trim().parse().ok())
}

fn select(ink: &mut Ink, spec: &mut Option<&mut Spec>, l: usize) -> Result<()> {
    ink.select_lane(l)?;
    if let Some(sp) = spec.as_deref_mut() {
        sp.select_lane(l)?;
    }
    Ok(())
}

impl Act {
    fn fin(&self, reason: FinishReason, t: Option<Timings>) -> Finish {
        Finish { reason, cached_prompt_tokens: self.from, timings: t, events: Default::default() }
    }

    fn timings(&self, spec: &Option<&mut Spec>) -> Timings {
        let draft = if self.use_spec {
            spec.as_deref().map(|sp| {
                let st = &sp.stats;
                ((st.rows - st.passes.min(st.rows)) as u64, st.accepted as u64)
            })
        } else {
            None
        };
        Timings { prompt_n: self.n - self.from, cache_n: self.from, prompt_ms: self.prompt_ms, predicted_n: self.generated,
            predicted_ms: self.t1.elapsed().as_secs_f64() * 1e3, draft }
    }

    /// Hand newly sampled tokens to the client (the rules of `crate::ink::run`); Some(finish) when the request ends. The current lane is
    /// this request's (the drafting statistics in the log line are its own).
    fn emit(&mut self, toks: Vec<u32>, cap: usize, spec: &Option<&mut Spec>) -> Option<Finish> {
        let p = &self.job.req.params;
        let mut batch = vec![];
        for tok in toks {
            if p.stop_token_ids.contains(&tok) {
                if !batch.is_empty() {
                    let _ = self.job.sink.send_tokens(std::mem::take(&mut batch));
                }
                self.log_spec(spec, "stop");
                return Some(self.fin(FinishReason::Stop(tok), Some(self.timings(spec))));
            }
            batch.push(tok);
            if !self.use_spec {
                self.hist.push(tok);
            }
            self.generated += 1;
            self.cur = tok;
            if self.generated >= p.max_tokens {
                break;
            }
        }
        if !self.job.sink.send_tokens(batch) || self.job.req.cancel.is_cancelled() {
            self.log_spec(spec, "cancelled");
            return Some(self.fin(FinishReason::Cancelled, Some(self.timings(spec))));
        }
        let margin = if self.use_spec { 16 } else { 0 };
        if self.generated >= p.max_tokens || self.n + self.generated + margin >= cap {
            self.log_spec(spec, "length");
            return Some(self.fin(FinishReason::Length, Some(self.timings(spec))));
        }
        None
    }

    fn log_spec(&self, spec: &Option<&mut Spec>, why: &str) {
        if !self.use_spec {
            return;
        }
        if let Some(sp) = spec.as_deref() {
            let st = &sp.stats;
            tracing::info!("ink spec ({why}): {} passes, {} tokens ({:.2} a pass), {} drafts accepted of {} verified, {} plain tokens, {} suspensions", st.passes,
                st.tokens - st.plain_tokens, (st.tokens - st.plain_tokens) as f64 / st.passes.max(1) as f64, st.accepted, st.rows - st.passes.min(st.rows), st.plain_tokens,
                st.suspensions);
        }
    }

    fn banned(&self) -> Vec<u32> {
        let p = &self.job.req.params;
        if self.generated < p.min_tokens { p.stop_token_ids.clone() } else { vec![] }
    }
}

pub(crate) fn run_lanes(ink: &mut Ink, mut spec: Option<&mut Spec>, dsamp: &mut DevSampler, images: &mut fe_ink::images::Images, rx: &mpsc::Receiver<Job>) -> Result<()> {
    let nl = ink.n_lanes();
    if let Some(sp) = spec.as_deref_mut() {
        sp.add_lanes(nl)?;
    }
    // INK_LANE_PREFILL_ROWS: 128 on a discrete card; 256 on an integrated GPU (DGX Spark), where its 2048-row prompt chunks make a prefilling
    // lane beside decoding ones relatively slower (a 30K prompt beside 3 decoding streams: 53 s instead of 80 s; the streams 4-6 instead of 6-7 tok/s)
    let pre_default = if ink.ctx.attribute(18).unwrap_or(0) != 0 { 256 } else { 128 };
    let pre_rows = env_usize("INK_LANE_PREFILL_ROWS").unwrap_or(pre_default).clamp(16, ink.tmax);
    // the K/V budget in tokens over all lanes: what the card has free now over the per-token cost (global K/V of 8 applications + the draft
    // head's full layer); the 16 GB layout moves experts to host memory as K/V grows, so there the lanes share one context's worth
    let per_tok = if ink.kvq6 { 832 * 16 } else { 2048 * 16 } + if spec.is_some() { 4096 } else { 0 };
    let budget = if ink.slot_dev_bytes() > 0 {
        ink.cap
    } else {
        let (free, _) = ink.ctx.mem_info()?;
        ((free.saturating_sub(768 << 20)) / per_tok).max(ink.cap)
    };
    let mut lanes: Vec<Lane> = (0..nl).map(|_| {
        let mut pc = PCache::new(ink);
        pc.set_lanes(nl);
        Lane { pc, act: None, used: 0 }
    }).collect();
    tracing::info!("ink engine: {nl} lanes, K/V budget {budget} tokens over all lanes ({:.1} KiB a token), prefill chunks of {pre_rows} rows beside decoding lanes, {} prompt-cache snapshots a lane",
        per_tok as f64 / 1024.0, lanes[0].pc.max_snaps);
    let cap = ink.cap;
    let mut queue: VecDeque<Job> = VecDeque::new();
    let mut tick: u64 = 0;
    let mut cost = Cost::new();
    // the park pool (INK_PARK_GB): conversations a lane gives up wait there for their next turn
    let mut park = fe_ink::park::Park::new(ink);
    match &park {
        Some(p) => tracing::info!("ink engine: park pool {:.1} GiB ({}; INK_PARK_GB, INK_PARK_RESIDENT), conversations of {} tokens and more",
            p.budget() as f64 / 1073741824.0, if p.resident { "kept in device memory, mapped back on return" } else { "pinned host memory, copied back on return" }, p.min_tokens),
        None => tracing::info!("ink engine: park pool off (INK_PARK_GB=0)"),
    }
    loop {
        // 1. arrivals
        if lanes.iter().all(|l| l.act.is_none()) && queue.is_empty() {
            match rx.recv() {
                Ok(j) => queue.push_back(j),
                Err(_) => return Ok(()),
            }
        }
        while let Ok(j) = rx.try_recv() {
            queue.push_back(j);
        }
        // 2. queued requests whose client is gone
        let mut keep = VecDeque::new();
        for j in queue.drain(..) {
            if j.req.cancel.is_cancelled() || j.sink.is_closed() {
                j.sink.finish(Finish::new(FinishReason::Cancelled));
            } else {
                keep.push_back(j);
            }
        }
        queue = keep;
        // 3. lanes for waiting requests (arrival order)
        while let Some(job) = queue.front() {
            let n = job.req.prompt_ids.len();
            if n == 0 || n >= cap {
                let j = queue.pop_front().unwrap();
                j.sink.finish(Finish::new(FinishReason::Error(format!("prompt of {n} tokens does not fit the context {cap}"))));
                continue;
            }
            let free: Vec<usize> = (0..nl).filter(|&l| lanes[l].act.is_none()).collect();
            if free.is_empty() {
                break;
            }
            let need_draft = spec.is_some();
            let mut best = free[0];
            let mut best_r = (lanes[best].pc.resume_point(&job.req.prompt_ids, need_draft), u64::MAX - lanes[best].used);
            for &l in &free[1..] {
                let r = (lanes[l].pc.resume_point(&job.req.prompt_ids, need_draft), u64::MAX - lanes[l].used);
                if r > best_r {
                    best = l;
                    best_r = r;
                }
            }
            // a parked conversation that resumes further than every free lane: it comes back into a free lane (an empty one, else the
            // least recently used, whose own conversation is parked)
            let from_park = park.as_ref().and_then(|p| p.best(&job.req.prompt_ids, need_draft)).filter(|&(_, r)| r > best_r.0).is_some();
            if from_park {
                best = *free.iter().min_by_key(|&&l| (ink.lane_len(l) > 0 || !lanes[l].pc.is_empty(), lanes[l].used)).unwrap();
            }
            let need = (n + job.req.params.max_tokens + 16).min(cap);
            let held = |lanes: &Vec<Lane>, ink: &Ink, l: usize| -> usize {
                match &lanes[l].act {
                    Some(a) => a.need,
                    None => ink.lane_len(l),
                }
            };
            let mut others: usize = (0..nl).filter(|&l| l != best).map(|l| held(&lanes, ink, l)).sum();
            if others + need > budget {
                // drop idle lanes' cached conversations, least recently used first
                let mut idle: Vec<usize> = (0..nl).filter(|&l| l != best && lanes[l].act.is_none() && ink.lane_len(l) > 0).collect();
                idle.sort_by_key(|&l| lanes[l].used);
                for l in idle {
                    if others + need <= budget {
                        break;
                    }
                    select(ink, &mut spec, l)?;
                    park_or_drop(ink, &mut spec, &mut park, &mut lanes[l].pc, l)?;
                    others = (0..nl).filter(|&l| l != best).map(|l| held(&lanes, ink, l)).sum();
                    tracing::info!("ink lanes: lane {l}'s cached conversation moved out to make room ({others} + {need} tokens of {budget})");
                }
            }
            let busy_others = (0..nl).any(|l| l != best && lanes[l].act.is_some());
            if others + need > budget && busy_others {
                break;
            }
            let job = queue.pop_front().unwrap();
            tick += 1;
            // the lane's own conversation, if this request does not continue it: parked (or dropped) before the lane is reused
            let continues = !from_park && lanes[best].pc.resume_point(&job.req.prompt_ids, need_draft) > 0;
            if !continues && (ink.lane_len(best) > 0 || !lanes[best].pc.is_empty()) {
                select(ink, &mut spec, best)?;
                park_or_drop(ink, &mut spec, &mut park, &mut lanes[best].pc, best)?;
            }
            if from_park {
                select(ink, &mut spec, best)?;
                let p = park.as_mut().unwrap();
                // (parking the lane's conversation may have evicted entries: look the conversation up again)
                if let Some((pi, _)) = p.best(&job.req.prompt_ids, need_draft) {
                    let mv = p.unpark(pi, ink, spec.as_deref_mut(), &mut lanes[best].pc)?;
                    tracing::info!("ink park: conversation of {} tokens back into lane {best} ({:.0} MiB, {:.1} ms); {} parked, {:.2} GiB",
                        mv.mark, mv.bytes as f64 / 1048576.0, mv.ms, p.len(), p.used_bytes() as f64 / 1073741824.0);
                }
            }
            lanes[best].used = tick;
            match start(ink, &mut spec, &mut lanes[best].pc, images, best, job, need) {
                Ok(Some(a)) => lanes[best].act = Some(a),
                Ok(None) => {}
                Err(e) => tracing::warn!("ink lanes: lane {best}: {e:#}"),
            }
        }
        // 4. one forward
        let dec: Vec<usize> = (0..nl).filter(|&l| matches!(lanes[l].act.as_ref().map(|a| &a.phase), Some(Phase::Decode))).collect();
        let pre: Option<usize> = (0..nl).filter(|&l| matches!(lanes[l].act.as_ref().map(|a| &a.phase), Some(Phase::Prefill(_))))
            .min_by_key(|&l| lanes[l].act.as_ref().unwrap().t0);
        if dec.is_empty() && pre.is_none() {
            continue;
        }
        // a prefilling lane whose client is gone stops before its next chunk (its snapshots stay)
        if let Some(l) = pre {
            let a = lanes[l].act.as_ref().unwrap();
            if a.job.req.cancel.is_cancelled() || a.job.sink.is_closed() {
                let a = lanes[l].act.take().unwrap();
                let f = a.fin(FinishReason::Cancelled, None);
                a.job.sink.finish(f);
                continue;
            }
        }
        let r = if dec.is_empty() {
            prefill_alone(ink, &mut spec, &mut lanes, pre.unwrap(), dsamp)
        } else if dec.len() == 1 && pre.is_none() {
            decode_alone(ink, &mut spec, &mut lanes, dec[0], dsamp)
        } else {
            pass_multi(ink, &mut spec, &mut lanes, &dec, pre, pre_rows, dsamp, &mut cost)
        };
        if let Err(e) = r {
            // an engine error: end every request, forget every lane's cache
            tracing::warn!("ink lanes: {e:#}");
            for l in 0..nl {
                lanes[l].pc.clear();
                if let Some(a) = lanes[l].act.take() {
                    let f = a.fin(FinishReason::Error(format!("{e:#}")), None);
                    a.job.sink.finish(f);
                }
                let _ = select(ink, &mut spec, l).and_then(|_| ink.reset());
            }
        }
    }
}

/// Give lane `l` the request: prompt cache restore (or reset), sampling state, the prefill run.
fn start(ink: &mut Ink, spec: &mut Option<&mut Spec>, pc: &mut PCache, images: &mut fe_ink::images::Images, l: usize, job: Job, need: usize) -> Result<Option<Act>> {
    select(ink, spec, l)?;
    let p = &job.req.params;
    let n = job.req.prompt_ids.len();
    // INK_DRAFT_MAX_CTX (16 GB layout): beyond it drafting costs more than it gains; the lanes keep the head loaded and decode such a request plainly
    let max_head: usize = std::env::var("INK_DRAFT_MAX_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let use_spec = spec.is_some() && p.min_tokens == 0 && n + p.max_tokens + 16 < ink.cap && n + p.max_tokens <= max_head;
    let feed = use_spec || (pc.on && spec.is_some());
    let cfg = SampleCfg { temperature: p.temperature, top_k: p.top_k as u32, top_p: p.top_p, min_p: p.min_p, rep: p.repetition_penalty, pres: p.presence_penalty,
        freq: p.frequency_penalty };
    let rng: u64 = p.seed.unwrap_or_else(crate::util::random_u64) | 1;
    let t0 = Instant::now();
    let from = match pc.begin(ink, if feed { spec.as_deref_mut() } else { None }, &job.req.prompt_ids) {
        Ok(f) => f,
        Err(e) => {
            job.sink.finish(Finish::new(FinishReason::Error(format!("{e:#}"))));
            return Ok(None);
        }
    };
    // the pictures of the prompt (the encoder runs here, between forwards, for those the prefill reaches)
    if let Err(e) = crate::ink::place_images(ink, images, &job.req, from) {
        job.sink.finish(Finish { reason: FinishReason::Error(format!("{e:#}")), cached_prompt_tokens: from, timings: None, events: Default::default() });
        return Ok(None);
    }
    if use_spec {
        spec.as_deref_mut().unwrap().begin(cfg, &job.req.prompt_ids);
    }
    let mut hist = PenHist::default();
    hist.begin(&cfg, &job.req.prompt_ids);
    let run = match pc.prefill_start(ink, if feed { spec.as_deref() } else { None }, n, from) {
        Ok(r) => r,
        Err(e) => {
            job.sink.finish(Finish { reason: FinishReason::Error(format!("{e:#}")), cached_prompt_tokens: from, timings: None, events: Default::default() });
            return Ok(None);
        }
    };
    Ok(Some(Act { job, n, from, use_spec, feed, cfg, rng, hist, phase: Phase::Prefill(run), t0, prompt_ms: 0.0, t1: Instant::now(), generated: 0, cur: 0, need }))
}

/// After a lane's last prompt chunk: the first token from that row's logits (host copy), the prompt-cache log line, decoding starts.
fn first_token(ink: &mut Ink, spec: &mut Option<&mut Spec>, lane: &mut Lane, lg: &[u16], dsamp: &mut DevSampler) -> Result<Option<Finish>> {
    let a = lane.act.as_mut().unwrap();
    a.prompt_ms = a.t0.elapsed().as_secs_f64() * 1e3;
    if lane.pc.on {
        let st = &lane.pc.last;
        tracing::info!("ink pcache (lane {}): {} of {} prompt tokens from the cache (common prefix {}, restore {:.1} ms), {} prefilled in {:.0} ms; {} snapshots taken ({:.1} ms), {} held, {:.2} GiB pinned host",
            ink.lane(), a.from, a.n, st.common, st.restore_ms, a.n - a.from, a.prompt_ms, st.snaps_taken, st.snap_ms, lane.pc.len(), lane.pc.host_bytes() as f64 / 1073741824.0);
    }
    ink.prompt = false;
    if !a.use_spec {
        ink.draft_taps = false;
    }
    a.t1 = Instant::now();
    a.phase = Phase::Decode;
    let tok = if a.use_spec {
        let mut rng = a.rng;
        let t = spec.as_deref_mut().unwrap().first_token(ink, lg, &mut rng)?;
        a.rng = rng;
        t
    } else {
        let banned = a.banned();
        if a.cfg.plain_greedy() && banned.is_empty() {
            argmax(lg)
        } else {
            let ptr = dsamp.upload_row(&ink.s, lg)?;
            let mut rng = a.rng;
            let t = dsamp.sample_one(&ink.s, ptr, &a.cfg, &a.hist, &banned, || rng_next(&mut rng))?;
            a.rng = rng;
            t
        }
    };
    Ok(a.emit(vec![tok], ink.cap, spec))
}

fn done(lane: &mut Lane, f: Option<Finish>) {
    if let Some(f) = f {
        let a = lane.act.take().unwrap();
        a.job.sink.finish(f);
    }
}

/// Nobody decodes: one full prompt chunk of lane `l` (the single-lane path).
fn prefill_alone(ink: &mut Ink, spec: &mut Option<&mut Spec>, lanes: &mut [Lane], l: usize, dsamp: &mut DevSampler) -> Result<()> {
    select(ink, spec, l)?;
    let lane = &mut lanes[l];
    let a = lane.act.as_mut().unwrap();
    let Phase::Prefill(run) = &mut a.phase else { unreachable!() };
    let ch = lane.pc.next_chunk(run, ink, ink.tmax);
    let pos = run.pos;
    ink.prompt = true;
    ink.draft_taps = a.feed;
    ink.spec = ch.spec;
    let r = ink.forward(&a.job.req.prompt_ids[pos..pos + ch.t], 1, None, None);
    ink.spec = false;
    let lg = r?;
    let feed = a.feed;
    lane.pc.chunk_done(run, ink, if feed { spec.as_deref_mut() } else { None }, ch, 0)?;
    if run.pos < run.n {
        a.job.sink.ping();
        return Ok(());
    }
    let f = first_token(ink, spec, lane, &lg, dsamp)?;
    done(lane, f);
    Ok(())
}

/// One decoding lane, nobody prefilling: the single-lane decode step (CUDA graph; drafted or plain).
fn decode_alone(ink: &mut Ink, spec: &mut Option<&mut Spec>, lanes: &mut [Lane], l: usize, dsamp: &mut DevSampler) -> Result<()> {
    select(ink, spec, l)?;
    let lane = &mut lanes[l];
    let a = lane.act.as_mut().unwrap();
    ink.prompt = false;
    let toks = if a.use_spec {
        ink.draft_taps = true;
        let mut rng = a.rng;
        let v = spec.as_deref_mut().unwrap().step(ink, a.cur, &mut rng)?;
        a.rng = rng;
        v
    } else {
        ink.draft_taps = false;
        let g = ink.decode_argmax(a.cur)?;
        let banned = a.banned();
        let t = if a.cfg.plain_greedy() && banned.is_empty() {
            g
        } else {
            let mut rng = a.rng;
            let t = dsamp.sample_one(&ink.s, ink.logits_ptr(), &a.cfg, &a.hist, &banned, || rng_next(&mut rng))?;
            a.rng = rng;
            t
        };
        vec![t]
    };
    let f = a.emit(toks, ink.cap, spec);
    done(lane, f);
    Ok(())
}

/// Several lanes in one forward: one prefilling lane's chunk (if any) first, then every decoding lane's rows.
#[allow(clippy::too_many_arguments)]
fn pass_multi(ink: &mut Ink, spec: &mut Option<&mut Spec>, lanes: &mut [Lane], dec: &[usize], pre: Option<usize>, pre_rows: usize,
    dsamp: &mut DevSampler, cost: &mut Cost) -> Result<()> {
    // drafts a lane may verify: on the 16 GB layout the forward keeps logits for 16 rows
    let tp0 = Instant::now();
    let host_cap = ink.slot_dev_bytes() > 0;
    let dmax = if host_cap { (16 / dec.len().max(1)).saturating_sub(1).max(1) } else { fe_ink::draft::NDRAFT };
    let mut plans: Vec<(usize, Option<Plan>, Vec<u32>)> = vec![];
    // what plain rows earn in this pass (tokens a ms), from the measured pass cost
    let nd = dec.len() as f64;
    let rate = nd / cost.pass_ms(nd);
    let mut drafting = 0usize;
    for &l in dec {
        select(ink, spec, l)?;
        let a = lanes[l].act.as_mut().unwrap();
        if a.use_spec {
            ink.draft_taps = true;
            let sp = spec.as_deref_mut().unwrap();
            let d = if cost.policy { sp.multi_d(rate, cost.draft_ms, cost.row_ms()).min(dmax) } else { dmax };
            let (plan, rows) = if cost.policy { sp.plan_multi(ink, a.cur, d)? } else { sp.plan(ink, a.cur, dmax)? };
            drafting += matches!(plan, Plan::Spec { .. }) as usize;
            plans.push((l, Some(plan), rows));
        } else {
            plans.push((l, None, vec![a.cur]));
        }
    }
    let tp_plan = tp0.elapsed().as_secs_f64() * 1e3;
    let dec_rows: usize = plans.iter().map(|p| p.2.len()).sum();
    // the prefilling lane's chunk
    let mut pch: Option<(usize, Chunk, Vec<u32>)> = None;
    if let Some(l) = pre {
        let room = ink.tmax.saturating_sub(dec_rows).min(pre_rows);
        let logits_room = if host_cap { 16usize.saturating_sub(dec_rows) } else { usize::MAX };
        if room > 0 && logits_room > 0 {
            select(ink, spec, l)?;
            let a = lanes[l].act.as_ref().unwrap();
            let Phase::Prefill(run) = &a.phase else { unreachable!() };
            let ch = lanes[l].pc.next_chunk(run, ink, room);
            let ids = a.job.req.prompt_ids[run.pos..run.pos + ch.t].to_vec();
            pch = Some((l, ch, ids));
        }
    }
    let mut segs: Vec<(usize, &[u32], bool)> = vec![];
    let pre_final = match &pch {
        Some((l, ch, ids)) => {
            segs.push((*l, ids.as_slice(), ch.spec));
            let Phase::Prefill(run) = &lanes[*l].act.as_ref().unwrap().phase else { unreachable!() };
            run.pos + ch.t == run.n
        }
        None => false,
    };
    for p in &plans {
        segs.push((p.0, p.2.as_slice(), p.1.as_ref().is_some_and(|pl| matches!(pl, Plan::Spec { .. }))));
    }
    let t_all: usize = segs.iter().map(|s| s.1.len()).sum();
    let nlog = dec_rows + pre_final as usize;
    ink.draft_taps = spec.is_some();
    let tp1 = Instant::now();
    let am = ink.forward_multi(&segs, nlog, true)?;
    let tp2 = Instant::now();
    cost.observe(t_all, (tp2 - tp1).as_secs_f64() * 1e3, drafting, tp_plan);
    let skip = t_all - nlog;
    // the prefilling lane
    if let Some((l, ch, _)) = pch {
        select(ink, spec, l)?;
        let lane = &mut lanes[l];
        let a = lane.act.as_mut().unwrap();
        let feed = a.feed;
        let Phase::Prefill(run) = &mut a.phase else { unreachable!() };
        lane.pc.chunk_done(run, ink, if feed { spec.as_deref_mut() } else { None }, ch, 0)?;
        if pre_final {
            let lg = ink.logits_rows(0, 1)?;
            let f = first_token(ink, spec, lane, &lg, dsamp)?;
            done(lane, f);
        } else {
            a.job.sink.ping();
        }
    }
    // the decoding lanes
    let mut off = t_all - dec_rows;
    for (l, plan, rows) in plans {
        select(ink, spec, l)?;
        let lane = &mut lanes[l];
        let a = lane.act.as_mut().unwrap();
        let lrow = off - skip;
        let toks = match plan {
            Some(plan) => {
                let mut rng = a.rng;
                let v = spec.as_deref_mut().unwrap().finish(ink, plan, lrow, off, &am[lrow..lrow + rows.len()], &mut rng)?;
                a.rng = rng;
                v
            }
            None => {
                let banned = a.banned();
                let t = if a.cfg.plain_greedy() && banned.is_empty() {
                    am[lrow]
                } else {
                    let mut rng = a.rng;
                    let t = dsamp.sample_one(&ink.s, ink.logits_ptr().byte_add(lrow * fe_ink::VOCAB * 2), &a.cfg, &a.hist, &banned, || rng_next(&mut rng))?;
                    a.rng = rng;
                    t
                };
                vec![t]
            }
        };
        off += rows.len();
        let f = a.emit(toks, ink.cap, spec);
        done(lane, f);
    }
    if prof_on() {
        prof_add([(tp1 - tp0).as_secs_f64(), (tp2 - tp1).as_secs_f64(), tp2.elapsed().as_secs_f64()], t_all, dec.len());
    }
    Ok(())
}

fn prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| fe_ink::env_flag("INK_LANES_PROF"))
}

/// INK_LANES_PROF=1: every 200 multi-lane passes, the mean time of planning (draft heads), the shared forward and finishing (accept,
/// rollback, draft extension, sampling), with the mean rows and decoding lanes of a pass.
fn prof_add(t: [f64; 3], rows: usize, lanes: usize) {
    static ACC: std::sync::Mutex<([f64; 3], usize, usize, usize)> = std::sync::Mutex::new(([0.0; 3], 0, 0, 0));
    let mut a = ACC.lock().unwrap();
    for i in 0..3 {
        a.0[i] += t[i];
    }
    a.1 += rows;
    a.2 += lanes;
    a.3 += 1;
    if a.3 == 200 {
        let n = a.3 as f64;
        tracing::info!("ink lanes prof: {:.0} passes: plan {:.2} ms, forward {:.2} ms, finish {:.2} ms; {:.1} rows, {:.2} decoding lanes a pass",
            n, a.0[0] / n * 1e3, a.0[1] / n * 1e3, a.0[2] / n * 1e3, a.1 as f64 / n, a.2 as f64 / n);
        *a = ([0.0; 3], 0, 0, 0);
    }
}

/// Online cost of a multi-lane pass: forward ms = a + b * rows (least squares with forgetting), draft head ms a drafting lane. INK_LANE_DRAFT=all
/// drafts every drafting lane as the single-lane policy would (no multi-lane choice).
pub(crate) struct Cost {
    reg: [f64; 5],
    pub draft_ms: f64,
    pub policy: bool,
}

impl Cost {
    fn new() -> Cost {
        // prior: ~20 ms a pass + 1.3 ms a row (RTX PRO 4000, 4 plain rows ~23 ms), 3 ms a draft head
        let mut c = Cost { reg: [0.0; 5], draft_ms: 3.0, policy: std::env::var("INK_LANE_DRAFT").ok().as_deref() != Some("all") };
        for (x, y) in [(4.0, 25.2), (16.0, 40.8)] {
            c.add(x, y, 2.0);
        }
        c
    }
    fn add(&mut self, x: f64, y: f64, w: f64) {
        self.reg[0] += w;
        self.reg[1] += w * x;
        self.reg[2] += w * y;
        self.reg[3] += w * x * x;
        self.reg[4] += w * x * y;
    }
    fn ab(&self) -> (f64, f64) {
        let [w, wx, wy, wxx, wxy] = self.reg;
        let det = w * wxx - wx * wx;
        let b = if det > 1e-6 { ((w * wxy - wx * wy) / det).clamp(0.2, 10.0) } else { 1.3 };
        ((wy - b * wx) / w, b)
    }
    pub fn row_ms(&self) -> f64 {
        self.ab().1
    }
    pub fn pass_ms(&self, rows: f64) -> f64 {
        let (a, b) = self.ab();
        (a + b * rows).max(1.0)
    }
    fn observe(&mut self, rows: usize, ms: f64, drafting: usize, plan_ms: f64) {
        for x in self.reg.iter_mut() {
            *x *= 0.98;
        }
        self.add(rows as f64, ms, 1.0);
        if drafting > 0 {
            self.draft_ms = 0.9 * self.draft_ms + 0.1 * (plan_ms / drafting as f64);
        }
    }
}

/// Lane `l` (selected) gives up its conversation: into the park pool when there is one, else dropped. The lane is reset either way.
fn park_or_drop(ink: &mut Ink, spec: &mut Option<&mut Spec>, park: &mut Option<fe_ink::park::Park>, pc: &mut PCache, l: usize) -> Result<()> {
    match park.as_mut() {
        Some(p) => {
            if let Some(mv) = p.park(ink, spec.as_deref_mut(), pc)? {
                tracing::info!("ink park: lane {l}'s conversation of {} tokens parked ({:.0} MiB, {:.1} ms); {} parked, {:.2} GiB", mv.mark,
                    mv.bytes as f64 / 1048576.0, mv.ms, p.len(), p.used_bytes() as f64 / 1073741824.0);
            }
        }
        None => {
            pc.clear();
            match spec.as_deref_mut() {
                Some(sp) => sp.reset(ink)?,
                None => ink.reset()?,
            }
        }
    }
    Ok(())
}
