//! The B70 engine behind fe-server (cargo feature `xe`).
//!
//! `--engine xe`: [`XeEngine`], arc-engine's serving shape: one sequence at a time (FIFO), a prefix cache of GDN-state checkpoints
//! (resume inside a shared prefix: the 8192 grid, a few tokens before each prompt end, the request's snapshot hints and the end of
//! every reply), a side pool (a second, 16K KV cache for requests that would otherwise evict a long kept sequence: titles, summaries,
//! sub-agents next to a long conversation) and speculative decoding (MTP chain k <= 7 with the adaptive cost table, context copy, ranked int8 draft head),
//! exact for greedy and for sampling (each window row is sampled from its own logits in order; a draft is kept when it equals the
//! sample). Env: FE_XE_K (0 = no drafting), FE_XE_CKPT_SLOTS (8), FE_XE_CKPT_EVERY (8192), FE_XE_CKPT_TAIL (8), FE_XE_HINTS (1),
//! FE_XE_SIDE_CTX (16384; 0 = no side pool),
//! MTP_DHEAD (the DHQ8 file; "none" = no draft head). Prompt cuts change the prefill numerics (like arc-engine's), so the lossless
//! gate against a harness run without cuts needs FE_XE_CKPT_TAIL=0 FE_XE_HINTS=0 (arc-engine: --ckpt_tail 0).
//!
//! `--engine xe-batch`: [`XeModel`], fe-xe's forward as a [`StepModel`] for fe-server's [`crate::batch::BatchEngine`] (one slot,
//! prompt windows, plain one-row decode, reuse only when the cached sequence is a prefix of the next prompt).
//!
//! Kernels: fe-xe's sources (FE_XE_KERNELS / FE_XE_RUST_SPV; the reference SPIR-V only in the test-only `xe-ref` build).
use std::path::Path;

use anyhow::{ensure, Context, Result};
use fe_xe::engine::{Engine, KEEP_HIDDEN, MAX_ROWS};

use crate::batch::StepModel;

pub struct XeModel {
    e: Engine,
    rcap: usize,
    ctx: usize,
    am: Vec<i32>,
    lg: Vec<f32>,
}

impl XeModel {
    /// Load on the engine thread (BatchEngine::spawn's factory). `weights_dir`: an XQ checkpoint (xq_config.json + shards).
    pub fn load(weights_dir: &Path, ctx: usize, rcap: usize) -> Result<Self> {
        let mut e = fe_xe::engine::open(weights_dir, ctx as u32, rcap as u32)?;
        e.bf16_logits = true; // arc-engine's head: bf16-rounded logits for the argmax and the sampler
        let v = e.cfg.vocab;
        Ok(Self { e, rcap: rcap.max(MAX_ROWS as usize), ctx, am: vec![0; MAX_ROWS as usize], lg: vec![0.0; MAX_ROWS as usize * v] })
    }
}

impl StepModel for XeModel {
    fn n_slots(&self) -> usize {
        1
    }
    fn max_ctx(&self) -> usize {
        self.ctx
    }
    fn max_rows(&self) -> usize {
        self.rcap
    }
    fn vocab(&self) -> usize {
        self.e.cfg.vocab
    }
    fn reset_slot(&mut self, _slot: usize) -> Result<()> {
        self.e.reset()
    }
    fn forward(&mut self, segs: &[(usize, &[u32])], logit_rows: &[usize]) -> Result<()> {
        ensure!(segs.len() == 1 && segs[0].0 == 0, "the B70 engine has one slot");
        let toks = segs[0].1;
        let m = toks.len();
        match logit_rows {
            [] => self.e.forward_rows(toks, 0, true),
            [r] if *r + 1 == m => self.e.forward_rows(toks, 1, true),
            rows => {
                // other rows: keep the normalised hidden rows, then one head batch over the span the rows cover
                ensure!(rows.len() <= MAX_ROWS as usize && rows.windows(2).all(|w| w[1] == w[0] + 1), "logit rows {rows:?}");
                self.e.forward_rows(toks, KEEP_HIDDEN, true)?;
                self.e.head_batch(rows[0] as u32, rows.len() as u32)
            }
        }
    }
    fn argmax(&mut self, n: usize) -> Result<Vec<u32>> {
        self.e.argmax(&mut self.am)?;
        Ok(self.am[..n].iter().map(|&t| t as u32).collect())
    }
    fn logits_host(&mut self, n: usize, dst: &mut [u16]) -> Result<()> {
        let v = self.e.cfg.vocab;
        self.e.fetch_logits(&mut self.lg)?;
        for (d, &x) in dst[..n * v].iter_mut().zip(&self.lg[..n * v]) {
            *d = fe_xe::eval::to_bf16(x);
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------------------------------------------
// XeEngine

use std::cell::Cell;
use std::sync::mpsc;
use std::time::Instant;

use fe_xe::engine::Engine as XeDev;
use fe_xe::fe_ze::DeviceBuffer;
use fe_xe::mtp::Mtp;
use fe_xe::specgen::{Pick, Spec, SpecCfg, SpecStats};
use fe_xdf::draft::DFlash;
use fe_xdf::spec::{DfCfg, DfSpec};

use crate::engine::{EngineInfo, Finish, FinishReason, GenRequest, Timings, TokenSink};
use crate::sampler::Sampler;

/// Options of [`XeEngine`].
#[derive(Clone, Debug)]
pub struct XeOpts {
    pub ctx: usize,
    /// prompt window rows (4096)
    pub rcap: usize,
    /// GDN-state checkpoint slots (~154 MB each)
    pub ckpt_slots: usize,
    /// checkpoint the prompt every N positions (8192; 0 = off)
    pub ckpt_every: usize,
    /// checkpoint N tokens before the prompt end (8; 0 = off): the next turn of a chat re-renders the last few tokens
    pub ckpt_tail: usize,
    pub spec: SpecCfg,
    /// also cut (checkpoint) the prompt at the request's snapshot hints (fe-server: reply starts, the shared system prompt end)
    pub hints: bool,
    /// the ranked int8 draft head (DHQ8); None = MTP chain only
    pub dhead: Option<String>,
    /// side pool positions (16384; 0 = none)
    pub side_ctx: usize,
    /// DFlash2 drafting (the default; FE_XE_DRAFT=mtp selects arc-engine's MTP policy above): the drafter checkpoint (FE_DRAFT_DIR).
    /// In DFlash2 mode the MTP head and its draft head are not loaded.
    pub dflash: Option<String>,
}

impl Default for XeOpts {
    fn default() -> Self {
        XeOpts {
            ctx: 131072,
            rcap: 4096,
            ckpt_slots: 8,
            ckpt_every: 8192,
            ckpt_tail: 8,
            hints: true,
            side_ctx: 16384,
            spec: SpecCfg::default(),
            dhead: None,
            dflash: None,
        }
    }
}

impl XeOpts {
    /// Defaults with the env overrides (FE_XE_K, FE_XE_CKPT_SLOTS, FE_XE_CKPT_EVERY, FE_XE_CKPT_TAIL, FE_XE_SIDE_CTX, FE_XE_HINTS,
    /// MTP_DHEAD).
    pub fn from_env(ctx: usize, rcap: usize) -> Result<Self> {
        let mut o = XeOpts { ctx, rcap, ..Default::default() };
        if let Ok(k) = std::env::var("FE_XE_K") {
            o.spec.k = k.parse().context("FE_XE_K")?;
        }
        if let Ok(n) = std::env::var("FE_XE_CKPT_SLOTS") {
            o.ckpt_slots = n.parse().context("FE_XE_CKPT_SLOTS")?;
        }
        if let Ok(n) = std::env::var("FE_XE_CKPT_EVERY") {
            o.ckpt_every = n.parse().context("FE_XE_CKPT_EVERY")?;
        }
        if let Ok(n) = std::env::var("FE_XE_CKPT_TAIL") {
            o.ckpt_tail = n.parse().context("FE_XE_CKPT_TAIL")?;
        }
        if let Ok(n) = std::env::var("FE_XE_SIDE_CTX") {
            o.side_ctx = n.parse().context("FE_XE_SIDE_CTX")?;
        }
        if let Ok(v) = std::env::var("FE_XE_HINTS") {
            o.hints = v != "0";
        }
        o.dflash = match std::env::var("FE_XE_DRAFT").as_deref() {
            Ok("mtp") => None,
            Ok("dflash") | Err(_) => Some(default_draft_dir()?),
            Ok(v) => anyhow::bail!("FE_XE_DRAFT={v}: dflash (default) or mtp"),
        };
        if let Ok(d) = std::env::var("MTP_DHEAD") {
            o.dhead = if d.is_empty() || d == "none" { None } else { Some(d) };
        }
        Ok(o)
    }
}

/// FE_DRAFT_DIR: the z-lab Qwen3.8-27B-DFlash2 checkpoint folder (required for DFlash2 drafting).
fn default_draft_dir() -> Result<String> {
    std::env::var("FE_DRAFT_DIR").map_err(|_| anyhow::anyhow!("FE_DRAFT_DIR is not set: the Qwen3.8-27B-DFlash2 drafter folder (or FE_XE_DRAFT=mtp for MTP drafting)"))
}

/// One GDN-state checkpoint: the recurrent state after `pos` tokens of the cached sequence (the KV caches are positional).
struct Ckpt {
    pos: usize,
    /// the KV pool whose sequence it belongs to (0 main, 1 side)
    pool: u8,
    used: bool,
    stamp: u64,
    buf: DeviceBuffer<u8>,
    /// the DFlash2 draft context at `pos` (DFlash::save_window), with the drafter
    dbuf: Option<DeviceBuffer<u8>>,
}

/// Snapshot the engine state at `pos` of pool `pool`'s sequence into the slot already holding it, else a free slot, else the pool's own
/// least recently used one, else the least recently used one.
fn take_ckpt(e: &XeDev, df: Option<&DFlash>, ckpts: &mut [Ckpt], clock: &mut u64, pool: u8, pos: usize) -> Result<()> {
    if ckpts.is_empty() || pos == 0 {
        return Ok(());
    }
    *clock += 1;
    let lru = |f: &dyn Fn(&Ckpt) -> bool| ckpts.iter().enumerate().filter(|(_, c)| f(c)).min_by_key(|(_, c)| c.stamp).map(|(i, _)| i);
    let i = ckpts
        .iter()
        .position(|c| c.used && c.pool == pool && c.pos == pos)
        .or_else(|| ckpts.iter().position(|c| !c.used))
        .or_else(|| lru(&|c| c.pool == pool))
        .or_else(|| lru(&|_| true))
        .unwrap();
    e.snapshot(&ckpts[i].buf)?;
    if let (Some(d), Some(db)) = (df, ckpts[i].dbuf.as_ref()) {
        d.save_window(e, pos, db.ptr())?;
    }
    let c = &mut ckpts[i];
    c.pos = pos;
    c.pool = pool;
    c.used = true;
    c.stamp = *clock;
    Ok(())
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

struct Worker {
    e: XeDev,
    /// arc-engine's MTP head (None in DFlash2 mode)
    mt: Option<Mtp>,
    df: Option<DFlash>,
    opts: XeOpts,
    /// the current pool's sequence: its first `e.pos` tokens are in the KV caches (the last token is not fed yet)
    hist: Vec<u32>,
    ckpts: Vec<Ckpt>,
    clock: u64,
    /// the pool the engine points at (0 main, 1 side); the other pool's caches, sequence and fed position
    cur: u8,
    other: Option<fe_xe::engine::KvPool>,
    other_hist: Vec<u32>,
    other_pos: usize,
    promotions: u64,
}

struct XeJob {
    req: GenRequest,
    sink: TokenSink,
}

/// The shaped like arc-engine's server B70 engine (see the module docs).
pub struct XeEngine {
    tx: mpsc::Sender<XeJob>,
    info: EngineInfo,
}

impl XeEngine {
    /// Loads the model on the engine thread; returns when it is ready (or failed).
    pub fn start(weights_dir: &Path, opts: XeOpts) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<XeJob>();
        let (info_tx, info_rx) = mpsc::channel::<std::result::Result<EngineInfo, String>>();
        let dir = weights_dir.to_path_buf();
        std::thread::Builder::new().name("fe-xe".into()).spawn(move || {
            let w = match Worker::load(&dir, opts) {
                Ok(w) => w,
                Err(e) => {
                    let _ = info_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            let _ = info_tx.send(Ok(EngineInfo { name: "xe".into(), max_batch: 1, max_context: w.opts.ctx }));
            w.run(rx);
        })?;
        let info = info_rx.recv().map_err(|_| anyhow::anyhow!("the xe engine thread died while loading"))?.map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self { tx, info })
    }
}

impl crate::engine::Engine for XeEngine {
    fn info(&self) -> EngineInfo {
        self.info.clone()
    }
    fn submit(&self, req: GenRequest, sink: TokenSink) {
        if let Err(mpsc::SendError(job)) = self.tx.send(XeJob { req, sink }) {
            job.sink.finish(Finish::new(FinishReason::Error("the xe engine has stopped".into())));
        }
    }
}

impl Worker {
    fn load(dir: &Path, mut opts: XeOpts) -> Result<Worker> {
        let mut e = fe_xe::engine::open(dir, opts.ctx as u32, opts.rcap as u32)?;
        e.bf16_logits = true; // arc-engine's head: bf16-rounded logits for the argmax and the sampler
        let ck = fe_xe::fe_xq::Checkpoint::open(dir)?;
        let mt = if opts.dflash.is_none() { Some(Mtp::load(&e, &ck, opts.dhead.as_deref())?) } else { None };
        opts.spec.rows = opts.rcap;
        let df = match &opts.dflash {
            Some(d) => Some(DFlash::load(&mut e, Path::new(d))?),
            None => None,
        };
        let bytes = e.ckpt_bytes();
        let mut ckpts = Vec::new();
        for _ in 0..opts.ckpt_slots {
            let dbuf = match &df {
                Some(d) => Some(e.dev.alloc::<u8>(d.window_bytes())?),
                None => None,
            };
            ckpts.push(Ckpt { pos: 0, pool: 0, used: false, stamp: 0, buf: e.dev.alloc::<u8>(bytes)?, dbuf });
        }
        let other = if opts.side_ctx > 0 && opts.side_ctx < opts.ctx { Some(e.alloc_pool(opts.side_ctx as u32, mt.is_some())?) } else { None };
        tracing::info!(
            "xe: ctx {} | side pool {} | prompt windows {} rows | {} checkpoint slots of {:.0} MB | spec k {} adaptive {} copy {} draft head {} | drafter {} | {:.2} GB on the device",
            opts.ctx,
            other.as_ref().map_or(0, |p| p.ctx),
            opts.rcap,
            opts.ckpt_slots,
            bytes as f64 / 1e6,
            opts.spec.k,
            opts.spec.adaptive,
            opts.spec.hybrid,
            mt.as_ref().is_some_and(|m| m.dhead.is_some()),
            if df.is_some() { "DFlash2" } else { "MTP" },
            e.dev.bytes.get() as f64 / 1e9
        );
        Ok(Worker { e, mt, df, opts, hist: Vec::new(), ckpts, clock: 0, cur: 0, other, other_hist: Vec::new(), other_pos: 0, promotions: 0 })
    }

    fn run(mut self, rx: mpsc::Receiver<XeJob>) {
        while let Ok(XeJob { req, sink }) = rx.recv() {
            if req.cancel.is_cancelled() || sink.is_closed() {
                sink.finish(Finish::new(FinishReason::Cancelled));
                continue;
            }
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.serve(&req, &sink)));
            let msg = match res {
                Ok(Ok((reason, cached, timings))) => {
                    sink.finish(Finish { reason, cached_prompt_tokens: cached, timings: Some(timings) });
                    continue;
                }
                Ok(Err(e)) => format!("xe engine: {e:#}"),
                Err(_) => "xe engine: panic".to_string(),
            };
            tracing::error!("{}: {msg}", req.id);
            self.forget_all();
            sink.finish(Finish::new(FinishReason::Error(msg)));
        }
    }

    /// Drops the current pool's sequence and checkpoints (the other pool's stay).
    fn forget(&mut self) {
        self.hist.clear();
        for c in &mut self.ckpts {
            if c.pool == self.cur {
                c.used = false;
            }
        }
    }

    /// The model state is unknown (an error mid-request): drop both sequences and every checkpoint.
    fn forget_all(&mut self) {
        self.hist.clear();
        self.other_hist.clear();
        self.other_pos = 0;
        for c in &mut self.ckpts {
            c.used = false;
        }
    }

    /// Points the engine at pool `p`'s caches and sequence (its GDN state comes back through `resume`: every request ends with a
    /// checkpoint).
    fn switch_to(&mut self, p: u8) {
        if p == self.cur {
            return;
        }
        let other = self.other.as_mut().expect("side pool");
        self.e.swap_kv(self.mt.as_mut(), other);
        std::mem::swap(&mut self.hist, &mut self.other_hist);
        let pos = self.e.pos as usize;
        self.e.pos = self.other_pos as u32;
        self.other_pos = pos;
        self.cur = p;
    }

    /// The side sequence outgrew the side cache: its KV / MTP cache prefix moves into the main buffers, its checkpoints and history
    /// become the main pool's (the main sequence is dropped).
    fn promote(&mut self) -> Result<()> {
        if self.cur == 1 {
            self.switch_to(0);
        }
        // now: main caches in the engine, the side ones in `other`, other_hist / other_pos = the side sequence
        let n = self.other_pos.min(self.other_hist.len());
        self.e.copy_kv_from(self.mt.as_ref(), self.other.as_ref().unwrap(), n)?;
        self.e.dev.sync()?;
        for c in &mut self.ckpts {
            if c.pool == 0 {
                c.used = false;
            } else {
                c.pool = 0;
            }
        }
        std::mem::swap(&mut self.hist, &mut self.other_hist);
        self.e.pos = self.other_pos as u32;
        self.other_hist.clear();
        self.other_pos = 0;
        self.promotions += 1;
        Ok(())
    }

    /// arc-engine's choosePool: the main pool unless there is a side pool, the request fits it, and running it in the main pool would cost
    /// the kept main sequence more re-prefill later (its tokens past the shared prefix, at least 2048) than the side pool costs now. A
    /// request continuing the side sequence that no longer fits the side cache promotes that sequence.
    fn choose_pool(&mut self, ids: &[u32], max_new: usize) -> Result<u8> {
        let Some(other) = &self.other else { return Ok(0) };
        let side_cap = if self.cur == 1 { self.e.ctx as usize } else { other.ctx as usize };
        let fed = self.e.pos as usize;
        let (mh, mp, sh, sp) = if self.cur == 0 {
            (&self.hist, fed, &self.other_hist, self.other_pos)
        } else {
            (&self.other_hist, self.other_pos, &self.hist, fed)
        };
        let (main_len, side_len) = (mh.len().min(mp), sh.len().min(sp));
        let l_main = lcp(mh, ids).min(main_len);
        let l_side = lcp(sh, ids).min(side_len);
        if ids.len() + max_new + 20 > side_cap {
            if l_side > l_main && l_side >= 256 {
                self.promote()?;
            }
            return Ok(0);
        }
        let lost = main_len - l_main;
        if lost < 2048 {
            return Ok(0);
        }
        Ok(if ids.len() - l_side < lost + (ids.len() - l_main) { 1 } else { 0 })
    }

    /// Restores the latest usable checkpoint for `ids` and returns its position (0 = cold start after a reset). Usable: inside the
    /// prefix shared with the cached sequence, one token short of its end (the MTP cache entry at p-1 was made with token p).
    fn resume(&mut self, ids: &[u32]) -> Result<usize> {
        let l = lcp(&self.hist, ids);
        let cur = self.cur;
        for c in &mut self.ckpts {
            if c.used && c.pool == cur && c.pos > l {
                c.used = false;
            }
        }
        let limit = l.saturating_sub(1);
        let best = self.ckpts.iter().enumerate().filter(|(_, c)| c.used && c.pool == cur && c.pos <= limit).max_by_key(|(_, c)| c.pos).map(|(i, _)| i);
        match best {
            Some(i) => {
                self.clock += 1;
                let c = &mut self.ckpts[i];
                c.stamp = self.clock;
                self.e.restore(&c.buf, c.pos as u32)?;
                if let (Some(d), Some(db)) = (self.df.as_ref(), c.dbuf.as_ref()) {
                    d.restore_window(&self.e, c.pos, db.ptr())?;
                }
                self.hist.truncate(c.pos);
                Ok(c.pos)
            }
            None => {
                self.e.reset()?;
                self.forget();
                Ok(0)
            }
        }
    }

    /// One request: (finish reason, cached prompt tokens, timings: prompt = from the request's start on the engine thread to the
    /// first token and its drafts, generation = the steps after it, as arc-engine's worker reports them).
    fn serve(&mut self, req: &GenRequest, sink: &TokenSink) -> Result<(FinishReason, usize, Timings)> {
        let spec_k = self.opts.spec.k;
        let ids = &req.prompt_ids;
        let n = ids.len();
        let p = &req.params;
        ensure!(n >= 2 && n + 32 < self.opts.ctx, "a prompt of {n} tokens does not fit the {}-token context", self.opts.ctx);
        let max_new = p.max_tokens.min(self.opts.ctx - n - 32);
        if max_new == 0 {
            return Ok((FinishReason::Length, 0, Timings::default()));
        }
        let t0 = Instant::now();
        let pool = self.choose_pool(ids, max_new)?;
        self.switch_to(pool);
        let c0 = self.resume(ids)?;
        // checkpoint cuts: the grid, a few tokens before the prompt end, the request's snapshot hints
        let mut cuts: Vec<usize> = Vec::new();
        if self.opts.ckpt_every > 0 {
            let mut c = (c0 / self.opts.ckpt_every + 1) * self.opts.ckpt_every;
            while c < n {
                cuts.push(c);
                c += self.opts.ckpt_every;
            }
        }
        // a snapshot hint near the prompt end (the reply start of a generation prompt that ends in an empty think block) is where the
        // next turn resumes: it replaces the tail cut (each cut costs a window: a full pass over the weights)
        let hints: Vec<usize> = if self.opts.hints { req.snapshots.iter().copied().filter(|&s| s > c0 && s < n).collect() } else { Vec::new() };
        let tail = self.opts.ckpt_tail;
        if tail > 0 && n - c0 > 4 * tail && !hints.iter().any(|&h| h + 4 * tail >= n) {
            cuts.push(n - tail);
        }
        cuts.extend(hints);
        cuts.sort_unstable();
        cuts.dedup();

        // host sampling (any non-greedy request, and greedy ones while min_tokens bans the stop ids); else the device argmax
        let mut sampler = Sampler::new(p, ids);
        let greedy = sampler.is_plain_greedy();
        let stops = p.stop_token_ids.clone();
        let picked = Cell::new(0usize);
        let host = |picked: usize| !greedy || (picked < p.min_tokens && !stops.is_empty());
        let mut row16 = vec![0u16; self.e.cfg.vocab];
        let mut pick_fn = |row: &[f32]| -> u32 {
            for (d, &x) in row16.iter_mut().zip(row) {
                *d = fe_xe::eval::to_bf16(x);
            }
            let banned: &[u32] = if picked.get() < p.min_tokens { &stops } else { &[] };
            let t = sampler.sample_bf16_banning(&row16, banned);
            sampler.accept(t);
            picked.set(picked.get() + 1);
            t
        };
        let Worker { e, mt, df, ckpts, clock, opts, cur, .. } = self;
        let cur = *cur;
        let mut spec = match df {
            Some(_) => Drafting::Df(DfSpec::new(DfCfg { rows: opts.rcap, k: opts.spec.k, ..Default::default() })),
            None => Drafting::Mtp(Spec::new(opts.spec.clone())),
        };
        let first = {
            let pick: Pick = if host(0) { Some(&mut pick_fn) } else { None };
            match (&mut spec, df.as_mut()) {
                (Drafting::Df(sp), Some(d)) => {
                    let dp: *const DFlash = d;
                    sp.prefill(e, d, ids, c0, &cuts, &mut |e, pos| take_ckpt(e, Some(unsafe { &*dp }), ckpts, clock, cur, pos), pick)?
                }
                (Drafting::Mtp(sp), _) => {
                    let mt = mt.as_mut().context("MTP drafting without the MTP head")?;
                    sp.prefill(e, mt, ids, c0, &cuts, &mut |e, pos| take_ckpt(e, None, ckpts, clock, cur, pos), pick)?
                }
                _ => unreachable!(),
            }
        };
        if !host(0) {
            picked.set(1);
        }
        let prefill_s = t0.elapsed().as_secs_f64();
        let mut reason = FinishReason::Length;
        if stops.contains(&first) {
            reason = FinishReason::Stop(first);
        } else if !sink.send_tokens(vec![first]) {
            reason = FinishReason::Cancelled;
        } else {
            let mut generated = 1;
            while generated < max_new {
                if req.cancel.is_cancelled() || sink.is_closed() {
                    reason = FinishReason::Cancelled;
                    break;
                }
                let before = picked.get();
                let (kept, send, hit) = {
                    let pick: Pick = if host(before) { Some(&mut pick_fn) } else { None };
                    match (&mut spec, df.as_mut()) {
                        (Drafting::Df(sp), Some(d)) => sp.step(e, d, max_new - generated, &stops, pick)?,
                        (Drafting::Mtp(sp), _) => sp.step(e, mt.as_mut().context("MTP drafting without the MTP head")?, max_new - generated, &stops, pick)?,
                        _ => unreachable!(),
                    }
                };
                if picked.get() == before {
                    picked.set(before + kept.len());
                }
                let mut out: Vec<u32> = kept[..send].to_vec();
                if hit {
                    let t = out.pop().unwrap();
                    if !out.is_empty() {
                        sink.send_tokens(out);
                    }
                    reason = FinishReason::Stop(t);
                    break;
                }
                generated += out.len();
                if !sink.send_tokens(out) {
                    reason = FinishReason::Cancelled;
                    break;
                }
            }
        }
        // keep the sequence and checkpoint its end: the next turn of a conversation resumes here
        let pos = e.pos as usize;
        self.hist = std::mem::take(spec.hist_mut());
        debug_assert_eq!(self.hist.len(), pos + 1);
        take_ckpt(&self.e, self.df.as_ref(), &mut self.ckpts, &mut self.clock, cur, pos)?;
        self.e.dev.sync()?;
        let s = spec.st();
        tracing::info!(
            "{}: prompt {n} (cached {c0}{}) {:.0} tok/s (draft-head rows {} in {:.1} ms) | {} tokens {:.1} tok/s | windows {} plain {} drafted {} accepted {} copies {} vetoed {} | {:?}",
            req.id,
            if cur == 1 { ", side pool" } else { "" },
            (n - c0) as f64 / prefill_s.max(1e-9),
            self.mt.as_ref().and_then(|m| m.dhead.as_ref()).map_or(0, |d| d.last_rows),
            self.mt.as_ref().and_then(|m| m.dhead.as_ref()).map_or(0.0, |d| d.last_ms),
            s.tokens,
            s.tokens.saturating_sub(1) as f64 / s.gen_s.max(1e-9),
            s.windows,
            s.plain,
            s.drafted,
            s.accepted,
            s.copies,
            s.vetoed,
            reason
        );
        let timings = Timings {
            prompt_n: n - c0,
            cache_n: c0,
            prompt_ms: prefill_s * 1e3,
            predicted_n: s.tokens,
            predicted_ms: s.gen_s * 1e3,
            draft: (spec_k > 0 || self.df.is_some()).then_some((s.drafted, s.accepted)),
        };
        // FE_XE_PROFILE=1: device time per kernel of this request (fe-xe Kx profile), then reset
        if let Some(r) = self.e.kx.profile_report()? {
            tracing::info!("{}: {r}", req.id);
            self.e.kx.profile_reset()?;
        }
        Ok((reason, c0, timings))
    }
}

/// The drafting policy of a request: arc-engine's MTP chain (+ context copy, draft head) or DFlash2 blocks (+ context copy).
enum Drafting {
    Mtp(Spec),
    Df(DfSpec),
}

impl Drafting {
    fn hist_mut(&mut self) -> &mut Vec<u32> {
        match self {
            Drafting::Mtp(s) => &mut s.hist,
            Drafting::Df(s) => &mut s.hist,
        }
    }
    fn st(&self) -> &SpecStats {
        match self {
            Drafting::Mtp(s) => &s.st,
            Drafting::Df(s) => &s.st,
        }
    }
}
