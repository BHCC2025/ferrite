//! `BatchEngine`: an [`Engine`] that runs continuous batching over a slot-level model.
//!
//! The model side is [`StepModel`], the shape of `fe_model::model::Engine` (core branch, M1): a fixed
//! number of cache slots, `forward(segments, logit_rows)` over any mix of (slot, tokens) segments,
//! `argmax` / `logits_host` for the requested rows, `reset_slot`. One scheduler thread owns the model
//! (CUDA context stays on that thread) and every step:
//! 1. admits queued requests into free slots: a slot whose cached history is a prefix of the new prompt
//!    is reused (prompt cache: only the rest is prefilled), otherwise a slot is reset;
//! 2. builds one forward: one decode row per generating request, then prefill chunks for requests still
//!    in their prompt (`prefill_chunk` rows each, within the model's `max_rows`);
//! 3. samples the requested rows (device argmax when every row is plain greedy, else bf16 logits to the
//!    host and [`Sampler`]), and streams tokens, stopping on stop ids / `max_tokens` / cancellation.
use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Result, anyhow};

use crate::engine::{Engine, EngineInfo, Finish, FinishReason, GenRequest, TokenSink};
use crate::sampler::Sampler;

/// A batched, slot-based forward pass (implemented for `fe_model::model::Engine` in the workspace).
pub trait StepModel {
    /// Number of independent cache slots (= max concurrent requests).
    fn n_slots(&self) -> usize;
    /// Tokens per slot.
    fn max_ctx(&self) -> usize;
    /// Rows (tokens) per forward.
    fn max_rows(&self) -> usize;
    fn vocab(&self) -> usize;
    /// Empty a slot (its next token is position 0).
    fn reset_slot(&mut self, slot: usize) -> Result<()>;
    /// Append `segs` = (slot, tokens) to their slots; compute logits for the listed rows (indices into
    /// the concatenation of all segments' tokens), in that order.
    fn forward(&mut self, segs: &[(usize, &[u32])], logit_rows: &[usize]) -> Result<()>;
    /// Greedy token for each of the first `n` logit rows of the last forward.
    fn argmax(&mut self, n: usize) -> Result<Vec<u32>>;
    /// The first `n` logit rows of the last forward as bf16 bits, `[n][vocab]`.
    fn logits_host(&mut self, n: usize, dst: &mut [u16]) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub name: String,
    /// Prefill rows per request per step (0 = the model's `max_rows`).
    pub prefill_chunk: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self { name: "ferrite".into(), prefill_chunk: 0 }
    }
}

struct Job {
    req: GenRequest,
    sink: TokenSink,
}

pub struct BatchEngine {
    tx: mpsc::Sender<Job>,
    info: EngineInfo,
}

impl BatchEngine {
    /// Start the scheduler thread; `factory` builds the model on that thread (so a CUDA context created
    /// there stays current). Returns once the model is up, or with the factory's error.
    pub fn spawn<M, F>(cfg: BatchConfig, factory: F) -> Result<Self>
    where
        M: StepModel + 'static,
        F: FnOnce() -> Result<M> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<Job>();
        let (info_tx, info_rx) = mpsc::channel::<Result<EngineInfo, String>>();
        std::thread::Builder::new().name("fe-batch".into()).spawn(move || {
            let model = match factory() {
                Ok(m) => m,
                Err(e) => {
                    let _ = info_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            let info = EngineInfo { name: cfg.name.clone(), max_batch: model.n_slots(), max_context: model.max_ctx() };
            let _ = info_tx.send(Ok(info));
            Scheduler::new(model, cfg).run(rx);
        })?;
        let info = info_rx.recv().map_err(|_| anyhow!("batch engine thread died"))?.map_err(|e| anyhow!(e))?;
        Ok(Self { tx, info })
    }
}

impl Engine for BatchEngine {
    fn info(&self) -> EngineInfo {
        self.info.clone()
    }

    fn submit(&self, req: GenRequest, sink: TokenSink) {
        if let Err(mpsc::SendError(job)) = self.tx.send(Job { req, sink }) {
            job.sink.finish(Finish::new(FinishReason::Error("the batch engine has stopped".into())));
        }
    }
}

struct Slot {
    busy: bool,
    /// Tokens currently in this slot's cache (everything forwarded since the last reset).
    hist: Vec<u32>,
    last_used: u64,
}

struct Active {
    id: String,
    prompt: Vec<u32>,
    params: crate::engine::SamplingParams,
    cancel: crate::engine::CancelToken,
    sink: TokenSink,
    sampler: Sampler,
    slot: usize,
    /// Prompt tokens already in the slot.
    fed: usize,
    cached: usize,
    generated: usize,
    /// Sampled but not yet forwarded (decode phase).
    next: Option<u32>,
    t_start: Instant,
}

struct Scheduler<M: StepModel> {
    m: M,
    chunk: usize,
    slots: Vec<Slot>,
    active: Vec<Active>,
    pending: VecDeque<Job>,
    buf: Vec<u16>,
    clock: u64,
}

impl<M: StepModel> Scheduler<M> {
    fn new(m: M, cfg: BatchConfig) -> Self {
        let chunk = if cfg.prefill_chunk == 0 { m.max_rows() } else { cfg.prefill_chunk.min(m.max_rows()) };
        let slots = (0..m.n_slots()).map(|_| Slot { busy: false, hist: Vec::new(), last_used: 0 }).collect();
        Self { m, chunk: chunk.max(1), slots, active: Vec::new(), pending: VecDeque::new(), buf: Vec::new(), clock: 0 }
    }

    fn run(mut self, rx: mpsc::Receiver<Job>) {
        loop {
            if self.active.is_empty() && self.pending.is_empty() {
                match rx.recv() {
                    Ok(j) => self.pending.push_back(j),
                    Err(_) => return, // engine dropped
                }
            }
            while let Ok(j) = rx.try_recv() {
                self.pending.push_back(j);
            }
            self.reap_cancelled();
            self.admit();
            if self.active.is_empty() {
                continue;
            }
            // a panic in a step (model glue, sampler) must not take the scheduler thread down with it
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.step()))
                .unwrap_or_else(|p| {
                    let what = p.downcast_ref::<&str>().map(|s| s.to_string())
                        .or_else(|| p.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "panic".into());
                    Err(anyhow!("panic: {what}"))
                });
            if let Err(e) = res {
                let msg = format!("engine step failed: {e:#}");
                tracing::error!("{msg}");
                for i in (0..self.active.len()).rev() {
                    self.finish(i, FinishReason::Error(msg.clone()));
                }
                for s in &mut self.slots {
                    s.hist.clear(); // cache contents unknown now: force resets
                    s.hist.push(u32::MAX);
                }
            }
        }
    }

    fn reap_cancelled(&mut self) {
        for i in (0..self.active.len()).rev() {
            let a = &self.active[i];
            if a.cancel.is_cancelled() || a.sink.is_closed() {
                self.finish(i, FinishReason::Cancelled);
            }
        }
    }

    fn admit(&mut self) {
        while !self.pending.is_empty() && self.slots.iter().any(|s| !s.busy) {
            let Job { req, sink } = self.pending.pop_front().unwrap();
            if req.cancel.is_cancelled() || sink.is_closed() {
                sink.finish(Finish::new(FinishReason::Cancelled));
                continue;
            }
            let n = req.prompt_ids.len();
            if n == 0 || n >= self.m.max_ctx() {
                let msg = format!("prompt of {n} tokens does not fit the {}-token context", self.m.max_ctx());
                sink.finish(Finish::new(FinishReason::Error(msg)));
                continue;
            }
            // best free slot: the longest cached history that is a strict prefix of the prompt; on a tie an
            // empty slot, then the least recently used one
            let prompt = &req.prompt_ids;
            let best = self
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.busy)
                .map(|(i, s)| {
                    let reuse = if s.hist.len() < n && prompt.starts_with(&s.hist) { s.hist.len() } else { 0 };
                    (i, reuse, s.hist.is_empty(), std::cmp::Reverse(s.last_used))
                })
                .max_by_key(|&(i, reuse, empty, lru)| (reuse, empty, lru, std::cmp::Reverse(i)))
                .map(|(i, reuse, _, _)| (i, reuse));
            let (slot, reuse) = best.unwrap();
            if reuse == 0 {
                if let Err(e) = self.m.reset_slot(slot) {
                    sink.finish(Finish::new(FinishReason::Error(format!("reset_slot: {e:#}"))));
                    continue;
                }
                self.slots[slot].hist.clear();
            }
            self.clock += 1;
            self.slots[slot].busy = true;
            self.slots[slot].last_used = self.clock;
            tracing::debug!("{} -> slot {slot}, prompt {n}, reused {reuse}", req.id);
            self.active.push(Active {
                sampler: Sampler::new(&req.params, &req.prompt_ids),
                id: req.id,
                prompt: req.prompt_ids,
                params: req.params,
                cancel: req.cancel,
                sink,
                slot,
                fed: reuse,
                cached: reuse,
                generated: 0,
                next: None,
                t_start: Instant::now(),
            });
        }
    }

    fn step(&mut self) -> Result<()> {
        let budget = self.m.max_rows();
        // (active index, slot, tokens, wants logits)
        let mut plan: Vec<(usize, usize, Vec<u32>, bool)> = Vec::new();
        let mut rows = 0;
        for (i, a) in self.active.iter().enumerate() {
            if let Some(t) = a.next {
                plan.push((i, a.slot, vec![t], true));
                rows += 1;
            }
        }
        for (i, a) in self.active.iter().enumerate() {
            if a.next.is_some() || rows >= budget {
                continue;
            }
            let left = a.prompt.len() - a.fed;
            let take = left.min(self.chunk).min(budget - rows);
            if take == 0 {
                continue;
            }
            plan.push((i, a.slot, a.prompt[a.fed..a.fed + take].to_vec(), take == left));
            rows += take;
        }
        if plan.is_empty() {
            return Ok(());
        }
        let mut logit_rows = Vec::new();
        let mut owners = Vec::new();
        let mut row = 0;
        for (i, _, toks, wants) in &plan {
            row += toks.len();
            if *wants {
                logit_rows.push(row - 1);
                owners.push(*i);
            }
        }
        {
            let segs: Vec<(usize, &[u32])> = plan.iter().map(|(_, s, t, _)| (*s, t.as_slice())).collect();
            self.m.forward(&segs, &logit_rows)?;
        }
        for (i, slot, toks, _) in &plan {
            self.slots[*slot].hist.extend_from_slice(toks);
            let a = &mut self.active[*i];
            if a.next.is_some() {
                a.next = None;
            } else {
                a.fed += toks.len();
            }
        }
        if owners.is_empty() {
            return Ok(());
        }
        let n = owners.len();
        // stop ids are banned until min_tokens (that needs the host path, as does any non-greedy row)
        let banning = |a: &Active| a.generated < a.params.min_tokens && !a.params.stop_token_ids.is_empty();
        let host = owners.iter().any(|&i| !self.active[i].sampler.is_plain_greedy() || banning(&self.active[i]));
        let toks: Vec<u32> = if host {
            let v = self.m.vocab();
            self.buf.resize(n * v, 0);
            self.m.logits_host(n, &mut self.buf)?;
            let mut toks = Vec::with_capacity(n);
            for (k, &i) in owners.iter().enumerate() {
                let a = &mut self.active[i];
                let banned: &[u32] = if banning(a) { &a.params.stop_token_ids } else { &[] };
                toks.push(a.sampler.sample_bf16_banning(&self.buf[k * v..(k + 1) * v], banned));
            }
            toks
        } else {
            self.m.argmax(n)?
        };
        let max_ctx = self.m.max_ctx();
        let mut done: Vec<(usize, FinishReason)> = Vec::new();
        for (&i, &t) in owners.iter().zip(&toks) {
            let a = &mut self.active[i];
            if a.generated == 0 {
                tracing::debug!("{} first token after {:.3}s", a.id, a.t_start.elapsed().as_secs_f64());
            }
            if a.params.stop_token_ids.contains(&t) {
                // (before min_tokens stop ids are banned in sampling, so this is a real stop)
                done.push((i, FinishReason::Stop(t)));
                continue;
            }
            a.sampler.accept(t);
            a.generated += 1;
            if !a.sink.send_tokens(vec![t]) {
                done.push((i, FinishReason::Cancelled));
            } else if a.generated >= a.params.max_tokens || self.slots[a.slot].hist.len() + 1 > max_ctx {
                done.push((i, FinishReason::Length));
            } else {
                a.next = Some(t);
            }
        }
        done.sort_by(|a, b| b.0.cmp(&a.0));
        for (i, reason) in done {
            self.finish(i, reason);
        }
        Ok(())
    }

    fn finish(&mut self, i: usize, reason: FinishReason) {
        let a = self.active.swap_remove(i);
        self.slots[a.slot].busy = false;
        tracing::debug!("{} finished in slot {}: {:?} after {} tokens", a.id, a.slot, reason, a.generated);
        a.sink.finish(Finish { reason, cached_prompt_tokens: a.cached, timings: None });
    }
}

/// A CPU stand-in for the slot-level model: `--engine fake` (live tests of the real-engine path without a
/// GPU) and the unit tests. Its next token is a hash of the slot's whole history, so a wrong cache reuse
/// or a mixed-up slot changes the output; it checks how it is driven (rows, one segment per slot, context).
pub mod fake {
    use super::*;
    use std::sync::{Arc, Mutex};

    pub struct FakeModel {
        pub n_slots: usize,
        pub max_ctx: usize,
        pub max_rows: usize,
        pub vocab: usize,
        /// generated ids are drawn from [2, text_vocab) (keep special tokens out of the output)
        pub text_vocab: usize,
        pub eos: u32,
        /// after 40 tokens of history, eos comes when sum % eos_period == 0 (about 1 in eos_period tokens)
        pub eos_period: u64,
        /// per slot: (sum of ids, number of ids)
        state: Vec<(u64, usize)>,
        logits: Vec<(u32, u32)>,
        /// every forward's (slot, n_tokens) segments
        pub log: Arc<Mutex<Vec<Vec<(usize, usize)>>>>,
        /// simulated time per forward
        pub delay: std::time::Duration,
        /// simulated extra time per row (prefill cost)
        pub row_delay: std::time::Duration,
    }

    impl FakeModel {
        pub fn new(n_slots: usize, max_ctx: usize, max_rows: usize, vocab: usize, eos: u32) -> Self {
            Self {
                n_slots,
                max_ctx,
                max_rows,
                vocab,
                text_vocab: vocab,
                eos,
                eos_period: 23,
                state: vec![(0, 0); n_slots],
                logits: Vec::new(),
                log: Default::default(),
                delay: Default::default(),
                row_delay: Default::default(),
            }
        }
    }

    /// The fake's next token after a history with this sum and length.
    pub fn next_token(sum: u64, len: usize, text_vocab: usize, eos: u32, eos_period: u64) -> u32 {
        if len > 40 && sum % eos_period == 0 {
            return eos;
        }
        ((sum.wrapping_mul(31) + len as u64) % (text_vocab as u64 - 2) + 2) as u32
    }

    /// What a solo greedy run of the fake produces.
    pub fn reference(prompt: &[u32], max_tokens: usize, text_vocab: usize, eos: u32, eos_period: u64) -> (Vec<u32>, FinishReason) {
        let (mut sum, mut len) = (prompt.iter().map(|&t| t as u64).sum::<u64>(), prompt.len());
        let mut out = Vec::new();
        loop {
            let t = next_token(sum, len, text_vocab, eos, eos_period);
            if t == eos {
                return (out, FinishReason::Stop(eos));
            }
            out.push(t);
            if out.len() >= max_tokens {
                return (out, FinishReason::Length);
            }
            sum += t as u64;
            len += 1;
        }
    }

    impl StepModel for FakeModel {
        fn n_slots(&self) -> usize {
            self.n_slots
        }
        fn max_ctx(&self) -> usize {
            self.max_ctx
        }
        fn max_rows(&self) -> usize {
            self.max_rows
        }
        fn vocab(&self) -> usize {
            self.vocab
        }
        fn reset_slot(&mut self, slot: usize) -> Result<()> {
            self.state[slot] = (0, 0);
            Ok(())
        }
        fn forward(&mut self, segs: &[(usize, &[u32])], logit_rows: &[usize]) -> Result<()> {
            let rows: usize = segs.iter().map(|s| s.1.len()).sum();
            anyhow::ensure!(rows > 0 && rows <= self.max_rows, "rows {rows}");
            let mut seen = std::collections::HashSet::new();
            let mut row_next: Vec<u32> = Vec::with_capacity(rows);
            for &(slot, toks) in segs {
                anyhow::ensure!(seen.insert(slot), "slot {slot} twice in one forward");
                let st = &mut self.state[slot];
                anyhow::ensure!(st.1 + toks.len() <= self.max_ctx, "context overflow");
                for &t in toks {
                    anyhow::ensure!((t as usize) < self.vocab, "token {t} out of range");
                    st.0 += t as u64;
                    st.1 += 1;
                    row_next.push(next_token(st.0, st.1, self.text_vocab, self.eos, self.eos_period));
                }
            }
            self.logits = logit_rows
                .iter()
                .map(|&r| {
                    let want = row_next[r];
                    (want, (want + 1) % self.vocab as u32)
                })
                .collect();
            self.log.lock().unwrap().push(segs.iter().map(|s| (s.0, s.1.len())).collect());
            std::thread::sleep(self.delay + self.row_delay * rows as u32);
            Ok(())
        }
        fn argmax(&mut self, n: usize) -> Result<Vec<u32>> {
            Ok(self.logits[..n].iter().map(|l| l.0).collect())
        }
        fn logits_host(&mut self, n: usize, dst: &mut [u16]) -> Result<()> {
            let v = self.vocab;
            for (k, &(a, b)) in self.logits[..n].iter().enumerate() {
                let row = &mut dst[k * v..(k + 1) * v];
                row.fill(0);
                row[a as usize] = (8.0f32.to_bits() >> 16) as u16;
                row[b as usize] = (6.0f32.to_bits() >> 16) as u16;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::fake::FakeModel;
    use super::*;
    use crate::engine::{CancelToken, GenEvent, SamplingParams};
    use std::sync::{Arc, Mutex};
    use tokio::sync::mpsc as tmpsc;

    pub const EOS: u32 = 1;

    fn reference(prompt: &[u32], max_tokens: usize, vocab: usize) -> (Vec<u32>, FinishReason) {
        super::fake::reference(prompt, max_tokens, vocab, EOS, 23)
    }

    fn params(max_tokens: usize) -> SamplingParams {
        SamplingParams { max_tokens, temperature: 0.0, stop_token_ids: vec![EOS], ..Default::default() }
    }

    pub fn generate(e: &BatchEngine, prompt: Vec<u32>, p: SamplingParams, cancel: CancelToken) -> (Vec<u32>, Finish) {
        let (tx, mut rx) = tmpsc::unbounded_channel();
        e.submit(GenRequest { id: "t".into(), prompt_ids: prompt, snapshots: Vec::new(), params: p, cancel }, TokenSink::new(tx));
        let mut out = Vec::new();
        loop {
            match rx.blocking_recv() {
                Some(GenEvent::Tokens(t)) => out.extend(t),
                Some(GenEvent::Finished(f)) => return (out, f),
                None => panic!("sink dropped without finish"),
            }
        }
    }

    fn engine(n_slots: usize, max_rows: usize, chunk: usize) -> (BatchEngine, Arc<Mutex<Vec<Vec<(usize, usize)>>>>) {
        engine_with_delay(n_slots, max_rows, chunk, 0)
    }

    fn engine_with_delay(n_slots: usize, max_rows: usize, chunk: usize, ms: u64) -> (BatchEngine, Arc<Mutex<Vec<Vec<(usize, usize)>>>>) {
        let mut m = FakeModel::new(n_slots, 4096, max_rows, 512, EOS);
        m.delay = std::time::Duration::from_millis(ms);
        let log = m.log.clone();
        let e = BatchEngine::spawn(BatchConfig { name: "fake".into(), prefill_chunk: chunk }, move || Ok(m)).unwrap();
        (e, log)
    }

    fn prompt(seed: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| (i * 7 + seed * 13) % 500 + 2).collect()
    }

    #[test]
    fn solo_greedy_matches_reference_with_chunked_prefill() {
        let (e, log) = engine(4, 64, 16);
        let p = prompt(1, 100);
        let (out, f) = generate(&e, p.clone(), params(30), CancelToken::new());
        let (want, reason) = reference(&p, 30, 512);
        assert_eq!(out, want);
        assert_eq!(f.reason, reason);
        // 100-token prompt in chunks of 16: 7 prefill forwards
        let log = log.lock().unwrap();
        assert_eq!(log.iter().take(7).map(|s| s[0].1).collect::<Vec<_>>(), [16, 16, 16, 16, 16, 16, 4]);
    }

    #[test]
    fn concurrent_requests_match_solo_runs() {
        let (e, log) = engine_with_delay(4, 48, 20, 2);
        let e = Arc::new(e);
        let hs: Vec<_> = (0..7)
            .map(|i| {
                let e = e.clone();
                std::thread::spawn(move || {
                    let p = prompt(i, 30 + 11 * i as usize);
                    let max = 20 + 5 * i as usize;
                    let (out, f) = generate(&e, p.clone(), params(max), CancelToken::new());
                    assert_eq!((out, f.reason), reference(&p, max, 512), "request {i}");
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        let log = log.lock().unwrap();
        assert!(log.iter().any(|segs| segs.len() >= 3), "requests never shared a forward");
        assert!(log.iter().all(|segs| segs.iter().map(|s| s.1).sum::<usize>() <= 48));
    }

    #[test]
    fn slot_history_is_reused_as_prompt_cache() {
        let (e, log) = engine(2, 256, 0);
        let p1 = prompt(3, 50);
        let (out1, f1) = generate(&e, p1.clone(), params(10), CancelToken::new());
        assert_eq!(f1.cached_prompt_tokens, 0);
        assert_eq!(f1.reason, FinishReason::Length, "pick a prompt that runs to max_tokens");
        // next turn: the slot holds the prompt + all generated tokens but the last (never forwarded)
        let mut p2 = p1.clone();
        p2.extend_from_slice(&out1);
        p2.extend_from_slice(&[7, 8, 9]);
        let before = log.lock().unwrap().len();
        let (out2, f2) = generate(&e, p2.clone(), params(10), CancelToken::new());
        assert_eq!(f2.cached_prompt_tokens, 50 + out1.len() - 1);
        assert_eq!((out2, f2.reason), reference(&p2, 10, 512), "reuse changed the output");
        let first = log.lock().unwrap()[before].clone();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].1, 4, "only the new tail is prefilled");
        // an unrelated prompt does not reuse
        let (_, f3) = generate(&e, prompt(9, 20), params(3), CancelToken::new());
        assert_eq!(f3.cached_prompt_tokens, 0);
    }

    #[test]
    fn cancellation_and_limits() {
        let (e, _) = engine(1, 64, 0);
        let cancel = CancelToken::new();
        cancel.cancel();
        let (out, f) = generate(&e, prompt(1, 10), params(1000), cancel);
        assert!(out.is_empty());
        assert_eq!(f.reason, FinishReason::Cancelled);
        let (_, f) = generate(&e, vec![5; 5000], params(10), CancelToken::new());
        assert!(matches!(f.reason, FinishReason::Error(_)));
        // the single slot is free again
        let (out, f) = generate(&e, prompt(2, 10), params(5), CancelToken::new());
        assert_eq!((out, f.reason), reference(&prompt(2, 10), 5, 512));
    }

    #[test]
    fn min_tokens_bans_stop_ids() {
        let (e, _) = engine(1, 64, 0);
        // find a prompt whose reference stops early, then require more tokens than that
        let (p, n) = (0..200u32)
            .map(|s| (prompt(s, 45), reference(&prompt(s, 45), 100, 512)))
            .find_map(|(p, (out, r))| (r == FinishReason::Stop(EOS) && out.len() < 30).then_some((p, out.len())))
            .expect("some prompt stops early");
        let p2 = SamplingParams { min_tokens: n + 10, ..params(n + 20) };
        let (out, f) = generate(&e, p.clone(), p2, CancelToken::new());
        assert!(out.len() >= n + 10, "stopped at {} < min_tokens {}", out.len(), n + 10);
        assert!(!out.contains(&EOS));
        assert!(matches!(f.reason, FinishReason::Length | FinishReason::Stop(_)));
    }

    #[test]
    fn sampled_requests_are_seeded() {
        let (e, _) = engine(2, 64, 0);
        let p = SamplingParams { temperature: 1.0, top_k: 2, seed: Some(42), ..params(40) };
        let a = generate(&e, prompt(4, 12), p.clone(), CancelToken::new()).0;
        let b = generate(&e, prompt(4, 12), p.clone(), CancelToken::new()).0;
        assert_eq!(a, b);
        assert_ne!(a, reference(&prompt(4, 12), 40, 512).0, "temperature 1 over 8.0/6.0 logits should deviate");
    }
}
