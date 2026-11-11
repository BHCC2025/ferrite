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
    /// lanes (FE_XE_LANES, default 1): requests served at once, each on KV caches of its own. Lane 0 has `ctx` positions, the
    /// others `lane_ctx`. With more than one lane the side pool is not used and drafting is DFlash2.
    pub lanes: usize,
    /// positions of lanes 1.. (FE_XE_LANE_CTX, 28672)
    pub lane_ctx: usize,
    /// decode windows a lane runs before the next lane's turn (FE_XE_QUANTUM, 8)
    pub quantum: usize,
    /// rows a lane gets in a shared verify window by the number of lanes decoding (index 2..8; FE_XE_LANE_ROWS="r2,r3,..")
    pub lane_rows: [usize; 9],
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
            lanes: 1,
            lane_ctx: 28672,
            quantum: 8,
            lane_rows: [8, 8, 8, 5, 4, 3, 2, 2, 2],
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
        if let Ok(n) = std::env::var("FE_XE_LANES") {
            o.lanes = n.parse::<usize>().context("FE_XE_LANES")?.clamp(1, 8);
        }
        if let Ok(n) = std::env::var("FE_XE_LANE_CTX") {
            o.lane_ctx = n.parse().context("FE_XE_LANE_CTX")?;
        }
        if let Ok(n) = std::env::var("FE_XE_QUANTUM") {
            o.quantum = n.parse::<usize>().context("FE_XE_QUANTUM")?.max(1);
        }
        if let Ok(v) = std::env::var("FE_XE_LANE_ROWS") {
            for (i, x) in v.split(',').enumerate().take(7) {
                let r = x.trim().parse::<usize>().context("FE_XE_LANE_ROWS")?;
                ensure!(r >= 1 && r * (i + 2) <= 16, "FE_XE_LANE_ROWS: {r} rows a lane at {} lanes (16 rows a window)", i + 2);
                o.lane_rows[i + 2] = r;
            }
        }
        if o.lanes > 1 {
            ensure!(o.dflash.is_some(), "FE_XE_LANES > 1 needs DFlash2 drafting (FE_XE_DRAFT=dflash)");
            // lanes share a prefix at prompt-window boundaries: checkpoint every window unless told otherwise
            if std::env::var("FE_XE_CKPT_EVERY").is_err() {
                o.ckpt_every = rcap;
            }
            if std::env::var("FE_XE_CKPT_SLOTS").is_err() {
                o.ckpt_slots = 10;
            }
            o.side_ctx = 0;
        }
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
    /// lanes: the state is what any prompt with these first `pos` tokens and the hints `hints` below `pos` computes from
    /// nothing (a prompt-window boundary of a run that started cold), so another lane may start from it
    clean: bool,
    hints: Vec<usize>,
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
    c.clean = false;
    c.hints.clear();
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
    /// FE_XE_LANES > 1: the lanes and the one the engine points at
    lanes: Vec<Lane>,
    lcur: usize,
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
            let _ = info_tx.send(Ok(EngineInfo { name: "xe".into(), max_batch: w.opts.lanes.max(1), max_context: w.opts.ctx }));
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
        let mut df = match &opts.dflash {
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
            ckpts.push(Ckpt { pos: 0, pool: 0, used: false, stamp: 0, buf: e.dev.alloc::<u8>(bytes)?, dbuf, clean: false, hints: Vec::new() });
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
        let mut lanes = Vec::new();
        if opts.lanes > 1 {
            for i in 0..opts.lanes {
                let (kv, cap) = if i == 0 { (None, opts.ctx) } else { (Some(e.alloc_pool(opts.lane_ctx as u32, false)?), opts.lane_ctx) };
                // lane i's draft context is the drafter's context i
                if let (Some(d), true) = (df.as_mut(), i > 0) {
                    d.add_ring(&e)?;
                }
                lanes.push(Lane { kv, cap, hist: Vec::new(), pos: 0, live: e.dev.alloc::<u8>(bytes)?, saved: false, stamp: 0, run: None });
            }
            tracing::info!("xe: {} lanes: lane 0 of {} positions, the others of {} | {} decode windows a turn | {:.2} GB on the device",
                opts.lanes, opts.ctx, opts.lane_ctx, opts.quantum, e.dev.bytes.get() as f64 / 1e9);
        }
        Ok(Worker { e, mt, df, opts, hist: Vec::new(), ckpts, clock: 0, cur: 0, other, other_hist: Vec::new(), other_pos: 0, promotions: 0, lanes, lcur: 0 })
    }

    fn run(mut self, rx: mpsc::Receiver<XeJob>) {
        if self.opts.lanes > 1 {
            return self.run_lanes(rx);
        }
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


// ------------------------------------------------------------------------------------------------------------------------------
// Lanes (FE_XE_LANES > 1): several requests served at once, each on its own KV caches.
//
// The engine holds one sequence. A lane is a set of KV caches plus, while its request is not on the engine, a copy of the
// engine state the request needs (GDN state, draft context). The worker gives the running lanes turns of `quantum` decode
// windows; switching lanes swaps the cache pointers and copies those states, which reproduces the engine state exactly, so a
// lane's tokens are the tokens of the same request served alone.
//
// Shared prefix: a request may start from another lane's checkpoint when that checkpoint is `clean`: taken at a prompt-window
// boundary (a multiple of the window rows) of a run that started from nothing or from a clean checkpoint, before that run's
// tail cut, with the same hints below it. Up to such a point every prompt with those tokens is computed in the same windows,
// so the copied caches and state are what the request would have computed itself. A request waits while another one that
// shares more than a window with it has not processed its prompt yet.

struct Lane {
    /// this lane's caches while the engine points at another lane (None: the engine holds them)
    kv: Option<fe_xe::engine::KvPool>,
    cap: usize,
    /// an idle lane's kept sequence (its first `pos` tokens are in the caches)
    hist: Vec<u32>,
    pos: usize,
    /// the engine state of this lane's request while another lane is on the engine
    live: DeviceBuffer<u8>,
    /// the lane's GDN state is in `live` (not in the engine): set when the engine moves to another lane, and while the lane
    /// decodes in shared windows, which work on `live` in place
    saved: bool,
    stamp: u64,
    run: Option<Box<Run>>,
}

struct Run {
    req: GenRequest,
    sink: TokenSink,
    sp: DfSpec,
    sampler: Sampler,
    row16: Vec<u16>,
    picked: usize,
    greedy: bool,
    stops: Vec<u32>,
    max_new: usize,
    generated: usize,
    c0: usize,
    cuts: Vec<usize>,
    /// the request's hints (all of them, empty when hints are off) and its tail cut
    hints: Vec<usize>,
    tailcut: Option<usize>,
    /// started from nothing or from a clean checkpoint
    clean: bool,
    started: bool,
    t0: Instant,
    prefill_s: f64,
}

impl Worker {
    fn run_lanes(mut self, rx: mpsc::Receiver<XeJob>) {
        let mut pending: std::collections::VecDeque<XeJob> = Default::default();
        let mut next = 0usize;
        loop {
            if pending.is_empty() && self.lanes.iter().all(|l| l.run.is_none()) {
                match rx.recv() {
                    Ok(j) => pending.push_back(j),
                    Err(_) => return,
                }
                // requests sent together (the lanes of one job) are a few milliseconds apart: let them queue
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            while let Ok(j) = rx.try_recv() {
                pending.push_back(j);
            }
            // admit what can start now, in arrival order; the rest keeps its place
            let mut wait = std::collections::VecDeque::new();
            while let Some(j) = pending.pop_front() {
                if let Some(j) = self.admit(j) {
                    wait.push_back(j);
                }
            }
            pending = wait;
            let n = self.lanes.len();
            // a request that has not processed its prompt yet goes first (one a pass), then the decoding lanes get their windows
            let fresh = (0..n).map(|k| (next + k) % n).find(|&i| self.lanes[i].run.as_ref().is_some_and(|r| !r.started));
            if let Some(li) = fresh {
                next = li + 1;
                self.turn(li);
            }
            let dec: Vec<usize> = (0..n).filter(|&i| self.lanes[i].run.as_ref().is_some_and(|r| r.started)).collect();
            if dec.is_empty() {
                if fresh.is_none() && !pending.is_empty() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                continue;
            }
            let waiting = self.lanes.iter().any(|l| l.run.as_ref().is_some_and(|r| !r.started));
            let turns = if waiting { self.opts.quantum } else { 1 };
            for _ in 0..turns {
                let go = if dec.len() == 1 { self.turn(dec[0]) } else { self.forest_turn(&dec) };
                if !go || !pending.is_empty() {
                    break;
                }
            }
        }
    }

    /// One call of [`Worker::advance`] for lane `li`; false when the request has ended.
    fn turn(&mut self, li: usize) -> bool {
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.advance(li)));
        match res {
            Ok(Ok(None)) => true,
            Ok(Ok(Some(reason))) => {
                if let Err(e) = self.finish_run(li, reason) {
                    tracing::error!("xe lane {li}: {e:#}");
                    self.fail_lane(li, format!("xe engine: {e:#}"));
                }
                false
            }
            Ok(Err(e)) => {
                self.fail_lane(li, format!("xe engine: {e:#}"));
                false
            }
            Err(_) => {
                self.fail_lane(li, "xe engine: panic".to_string());
                false
            }
        }
    }

    /// One shared window of the decoding lanes `ls`; false when one of them has ended.
    fn forest_turn(&mut self, ls: &[usize]) -> bool {
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.forest_step(ls)));
        let msg = match res {
            Ok(Ok(done)) => {
                let go = done.is_empty();
                for (li, reason) in done {
                    if let Err(e) = self.finish_run(li, reason) {
                        tracing::error!("xe lane {li}: {e:#}");
                        self.fail_lane(li, format!("xe engine: {e:#}"));
                    }
                }
                return go;
            }
            Ok(Err(e)) => format!("xe engine: {e:#}"),
            Err(_) => "xe engine: panic".to_string(),
        };
        // a shared window failed: none of its lanes can be trusted
        for &li in ls {
            self.fail_lane(li, msg.clone());
        }
        false
    }

    /// The decoding lanes `ls` (two or more) run one verify window together: each drafts its tree on its own draft context, one
    /// window verifies all trees (the weights are read once for all rows), each lane keeps its path. Returns the lanes that ended.
    fn forest_step(&mut self, ls: &[usize]) -> Result<Vec<(usize, FinishReason)>> {
        let t0 = Instant::now();
        let mut done = Vec::new();
        let mut act = Vec::new();
        for &li in ls {
            let run = self.lanes[li].run.as_ref().context("no request on the lane")?;
            if run.req.cancel.is_cancelled() || run.sink.is_closed() {
                done.push((li, FinishReason::Cancelled));
            } else if run.generated >= run.max_new {
                done.push((li, FinishReason::Length));
            } else {
                act.push(li);
            }
        }
        if act.is_empty() {
            return Ok(done);
        }
        // the shared window works on each lane's state in its live buffer
        let cur = self.lcur;
        if act.contains(&cur) && !self.lanes[cur].saved {
            self.lanes[cur].pos = self.e.pos as usize;
            self.e.snapshot(&self.lanes[cur].live)?;
            self.lanes[cur].saved = true;
        }
        let per = self.opts.lane_rows[act.len().min(8)].max(1);
        let Worker { e, df, lanes, .. } = self;
        let d = df.as_mut().context("lanes need the DFlash2 drafter")?;
        let mut plans = Vec::new();
        let mut tokens: Vec<u32> = Vec::new();
        let mut seqs = Vec::new();
        let mut row0s = Vec::new();
        for &li in &act {
            ensure!(lanes[li].saved, "lane {li}: its state is not in its live buffer");
            d.select_ring(li);
            let pos = lanes[li].pos;
            let run = lanes[li].run.as_mut().unwrap();
            let plan = run.sp.plan(e, d, pos, run.max_new - run.generated, per)?;
            row0s.push(tokens.len());
            tokens.extend_from_slice(&plan.win);
            seqs.push(fe_xe::engine::ForestSeq { pos: pos as u32, ctx: lanes[li].cap as u32, par: plan.par.clone(),
                gdn: e.gdn_view(Some(&lanes[li].live)), kv: e.kv_view(lanes[li].kv.as_ref()) });
            plans.push(plan);
        }
        e.taps = Some(d.taps(0));
        let r = e.forward_forest(&tokens, &seqs);
        e.taps = None;
        r?;
        let mut am = [0i32; 16];
        e.argmax(&mut am)?;
        for (k, &li) in act.iter().enumerate() {
            d.select_ring(li);
            let lane = &mut lanes[li];
            let lpos = &mut lane.pos;
            let run = lane.run.as_mut().unwrap();
            let Run { req, sink, sp, sampler, row16, picked, greedy, stops, max_new, generated, .. } = &mut **run;
            let p = &req.params;
            let greedy = *greedy;
            let stops: &[u32] = stops;
            let pk = Cell::new(*picked);
            let before = pk.get();
            let host = !greedy || (before < p.min_tokens && !stops.is_empty());
            let mut pick_fn = |row: &[f32]| -> u32 {
                for (d, &x) in row16.iter_mut().zip(row) {
                    *d = fe_xe::eval::to_bf16(x);
                }
                let banned: &[u32] = if pk.get() < p.min_tokens { stops } else { &[] };
                let t = sampler.sample_bf16_banning(row16, banned);
                sampler.accept(t);
                pk.set(pk.get() + 1);
                t
            };
            let (row0, rows) = (row0s[k], plans[k].win.len());
            let seq = &seqs[k];
            let mut commit = |e: &mut XeDev, path: &[usize]| e.commit_forest(k, seq, path);
            let (kept, send, hit) = {
                let pick: Pick = if host { Some(&mut pick_fn) } else { None };
                sp.settle(e, d, &plans[k], row0, &am[row0..row0 + rows], *max_new - *generated, stops, pick, &mut commit, 0.0)?
            };
            if pk.get() == before {
                pk.set(before + kept.len());
            }
            *picked = pk.get();
            *lpos += kept.len();
            let mut out: Vec<u32> = kept[..send].to_vec();
            if hit {
                let t = out.pop().unwrap();
                if !out.is_empty() {
                    sink.send_tokens(out);
                }
                done.push((li, FinishReason::Stop(t)));
            } else {
                *generated += out.len();
                if !sink.send_tokens(out) {
                    done.push((li, FinishReason::Cancelled));
                } else if *generated >= *max_new {
                    done.push((li, FinishReason::Length));
                }
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        for &li in &act {
            if let Some(run) = lanes[li].run.as_mut() {
                run.sp.st.gen_s += secs;
            }
        }
        Ok(done)
    }

    /// The request on lane `li` ends with an error: its sequence and checkpoints are dropped (the other lanes keep theirs).
    fn fail_lane(&mut self, li: usize, msg: String) {
        if let Some(run) = self.lanes[li].run.take() {
            tracing::error!("{}: {msg}", run.req.id);
            run.sink.finish(Finish::new(FinishReason::Error(msg)));
        }
        let l = &mut self.lanes[li];
        l.hist.clear();
        l.pos = 0;
        l.saved = false;
        for c in &mut self.ckpts {
            if c.pool == li as u8 {
                c.used = false;
            }
        }
        if self.lcur == li {
            let _ = self.e.reset();
        }
    }

    /// Points the engine (and the drafter) at lane `li`: the lane it leaves keeps its request's state in its live buffer.
    fn to_lane(&mut self, li: usize) -> Result<()> {
        if let Some(d) = self.df.as_mut() {
            d.select_ring(li);
        }
        let cur = self.lcur;
        if li != cur {
            if !self.lanes[cur].saved {
                self.lanes[cur].pos = self.e.pos as usize;
                if self.lanes[cur].run.is_some() {
                    self.e.snapshot(&self.lanes[cur].live)?;
                    self.lanes[cur].saved = true;
                }
            }
            let mut pool = self.lanes[li].kv.take().context("lane caches")?;
            self.e.swap_kv(None, &mut pool);
            self.lanes[cur].kv = Some(pool);
            self.lcur = li;
            self.e.pos = self.lanes[li].pos as u32;
        }
        if self.lanes[li].saved {
            let pos = self.lanes[li].pos as u32;
            self.e.restore(&self.lanes[li].live, pos)?;
            self.lanes[li].saved = false;
        }
        Ok(())
    }

    /// Gives the request a lane and sets the engine state it starts from, or returns it to wait (no free lane that fits it, or
    /// a request it shares a prefix with has not processed its prompt yet).
    fn admit(&mut self, job: XeJob) -> Option<XeJob> {
        if job.req.cancel.is_cancelled() || job.sink.is_closed() {
            job.sink.finish(Finish::new(FinishReason::Cancelled));
            return None;
        }
        let n = job.req.prompt_ids.len();
        let cap0 = self.lanes[0].cap;
        if n < 2 || n + 32 >= cap0 {
            job.sink.finish(Finish::new(FinishReason::Error(format!("a prompt of {n} tokens does not fit the {cap0}-token context"))));
            return None;
        }
        let need = n + job.req.params.max_tokens.min(cap0) + 32;
        let elig: Vec<usize> = (0..self.lanes.len()).filter(|&i| self.lanes[i].run.is_none() && (i == 0 || need <= self.lanes[i].cap)).collect();
        if elig.is_empty() {
            return Some(job);
        }
        let rcap = self.opts.rcap;
        if self.lanes.iter().any(|l| l.run.as_ref().is_some_and(|r| !r.started && lcp(&r.req.prompt_ids, &job.req.prompt_ids) > rcap)) {
            return Some(job);
        }
        match self.place(job, &elig) {
            Ok(()) => None,
            Err((job, e)) => {
                tracing::error!("{}: {e:#}", job.req.id);
                job.sink.finish(Finish::new(FinishReason::Error(format!("xe engine: {e:#}"))));
                None
            }
        }
    }

    fn place(&mut self, job: XeJob, elig: &[usize]) -> std::result::Result<(), (XeJob, anyhow::Error)> {
        let ids = &job.req.prompt_ids;
        let n = ids.len();
        let (rcap, tail) = (self.opts.rcap, self.opts.ckpt_tail);
        let hints_all: Vec<usize> = if self.opts.hints { job.req.snapshots.iter().copied().filter(|&s| s < n).collect() } else { Vec::new() };
        // the latest checkpoint of each eligible lane's own sequence
        let own = |w: &Worker, i: usize| -> usize {
            let l = lcp(&w.lanes[i].hist, ids).saturating_sub(1);
            w.ckpts.iter().filter(|c| c.used && c.pool == i as u8 && c.pos <= l).map(|c| c.pos).max().unwrap_or(0)
        };
        // lanes 1.. before lane 0 (the long one), least recently used first
        let order = |w: &Worker, i: usize| (i == 0, w.lanes[i].stamp);
        let best_own = elig.iter().copied().max_by_key(|&i| (own(self, i), std::cmp::Reverse(order(self, i)))).unwrap();
        let own_pos = own(self, best_own);
        // a clean checkpoint of another lane's sequence
        let mut cross: Option<(usize, usize)> = None; // (checkpoint index, position)
        for (ci, c) in self.ckpts.iter().enumerate() {
            if !(c.used && c.clean && c.pos > own_pos && c.pos % rcap == 0 && c.pos + 4 * tail.max(1) <= n) {
                continue;
            }
            let lane = &self.lanes[c.pool as usize];
            let seq: &[u32] = match &lane.run {
                Some(r) => &r.req.prompt_ids,
                None => &lane.hist,
            };
            if c.pos < lcp(seq, ids) && c.hints.iter().copied().eq(hints_all.iter().copied().filter(|&h| h < c.pos)) && cross.is_none_or(|x| c.pos > x.1) {
                cross = Some((ci, c.pos));
            }
        }
        let (t, c0, clean) = match cross {
            Some((ci, pos)) => {
                let src = self.ckpts[ci].pool as usize;
                let t = elig.iter().copied().filter(|&i| i != src).min_by_key(|&i| order(self, i));
                match t {
                    Some(t) => {
                        let r = (|| -> Result<()> {
                            self.to_lane(t)?;
                            for c in &mut self.ckpts {
                                if c.pool == t as u8 {
                                    c.used = false;
                                }
                            }
                            let sp = self.lanes[src].kv.as_ref().context("source lane caches")?;
                            self.e.copy_kv_from(None, sp, pos)?;
                            self.clock += 1;
                            let c = &mut self.ckpts[ci];
                            c.stamp = self.clock;
                            self.e.restore(&c.buf, pos as u32)?;
                            if let (Some(d), Some(db)) = (self.df.as_ref(), c.dbuf.as_ref()) {
                                d.restore_window(&self.e, pos, db.ptr())?;
                            }
                            Ok(())
                        })();
                        if let Err(e) = r {
                            return Err((job, e));
                        }
                        (t, pos, true)
                    }
                    None => (usize::MAX, 0, false),
                }
            }
            None => (usize::MAX, 0, false),
        };
        let (t, c0, clean) = if t != usize::MAX {
            (t, c0, clean)
        } else {
            let t = best_own;
            let r = (|| -> Result<(usize, bool)> {
                self.to_lane(t)?;
                let l = lcp(&self.lanes[t].hist, ids);
                for c in &mut self.ckpts {
                    if c.used && c.pool == t as u8 && c.pos > l {
                        c.used = false;
                    }
                }
                let limit = l.saturating_sub(1);
                let best = self.ckpts.iter().enumerate().filter(|(_, c)| c.used && c.pool == t as u8 && c.pos <= limit).max_by_key(|(_, c)| c.pos).map(|(i, _)| i);
                match best {
                    Some(i) => {
                        self.clock += 1;
                        let c = &mut self.ckpts[i];
                        c.stamp = self.clock;
                        self.e.restore(&c.buf, c.pos as u32)?;
                        if let (Some(d), Some(db)) = (self.df.as_ref(), c.dbuf.as_ref()) {
                            d.restore_window(&self.e, c.pos, db.ptr())?;
                        }
                        Ok((c.pos, c.clean))
                    }
                    None => {
                        self.e.reset()?;
                        for c in &mut self.ckpts {
                            if c.pool == t as u8 {
                                c.used = false;
                            }
                        }
                        Ok((0, true))
                    }
                }
            })();
            match r {
                Ok((c0, clean)) => (t, c0, clean),
                Err(e) => return Err((job, e)),
            }
        };
        let cap = self.lanes[t].cap;
        let p = &job.req.params;
        let max_new = p.max_tokens.min(cap - n - 32);
        // cuts: the window grid, a few tokens before the prompt end, the hints (as the one-lane engine)
        let mut cuts: Vec<usize> = Vec::new();
        if self.opts.ckpt_every > 0 {
            let mut c = (c0 / self.opts.ckpt_every + 1) * self.opts.ckpt_every;
            while c < n {
                cuts.push(c);
                c += self.opts.ckpt_every;
            }
        }
        let hints: Vec<usize> = hints_all.iter().copied().filter(|&s| s > c0).collect();
        let mut tailcut = None;
        if tail > 0 && n - c0 > 4 * tail && !hints.iter().any(|&h| h + 4 * tail >= n) {
            cuts.push(n - tail);
            tailcut = Some(n - tail);
        }
        cuts.extend(hints);
        cuts.sort_unstable();
        cuts.dedup();
        let sampler = Sampler::new(p, ids);
        let greedy = sampler.is_plain_greedy();
        let stops = p.stop_token_ids.clone();
        let sp = DfSpec::new(DfCfg { rows: self.opts.rcap, k: self.opts.spec.k, ..Default::default() });
        self.clock += 1;
        let l = &mut self.lanes[t];
        l.hist.clear();
        l.pos = c0;
        l.saved = false;
        l.stamp = self.clock;
        let row16 = vec![0u16; self.e.cfg.vocab];
        let XeJob { req, sink } = job;
        l.run = Some(Box::new(Run { req, sink, sp, sampler, row16, picked: 0, greedy, stops, max_new, generated: 0, c0, cuts, hints: hints_all, tailcut, clean,
            started: false, t0: Instant::now(), prefill_s: 0.0 }));
        Ok(())
    }

    /// The request on lane `li` processes its prompt (the first call) or runs one decode window. Some(reason): it has ended.
    fn advance(&mut self, li: usize) -> Result<Option<FinishReason>> {
        self.to_lane(li)?;
        let rcap = self.opts.rcap;
        let Worker { e, df, ckpts, clock, lanes, .. } = self;
        let d = df.as_mut().context("lanes need the DFlash2 drafter")?;
        let run = lanes[li].run.as_mut().context("no request on the lane")?;
        let Run { req, sink, sp, sampler, row16, picked, greedy, stops, max_new, generated, c0, cuts, hints, tailcut, clean, started, t0, prefill_s } = &mut **run;
        if *max_new == 0 {
            return Ok(Some(FinishReason::Length));
        }
        let p = &req.params;
        let greedy = *greedy;
        let stops: &[u32] = stops;
        let pk = Cell::new(*picked);
        let host = |picked: usize| !greedy || (picked < p.min_tokens && !stops.is_empty());
        let mut pick_fn = |row: &[f32]| -> u32 {
            for (d, &x) in row16.iter_mut().zip(row) {
                *d = fe_xe::eval::to_bf16(x);
            }
            let banned: &[u32] = if pk.get() < p.min_tokens { stops } else { &[] };
            let t = sampler.sample_bf16_banning(row16, banned);
            sampler.accept(t);
            pk.set(pk.get() + 1);
            t
        };
        let out = if !*started {
            let ids = &req.prompt_ids;
            let (run_clean, tc, pool) = (*clean, *tailcut, li as u8);
            let hints: &[usize] = hints;
            let dp: *const DFlash = d;
            let mut on_cut = |e: &mut XeDev, pos: usize| -> Result<()> {
                take_ckpt(e, Some(unsafe { &*dp }), ckpts, clock, pool, pos)?;
                if run_clean && pos % rcap == 0 && tc.is_none_or(|t| t >= pos) {
                    if let Some(c) = ckpts.iter_mut().find(|c| c.used && c.pool == pool && c.pos == pos) {
                        c.clean = true;
                        c.hints = hints.iter().copied().filter(|&h| h < pos).collect();
                    }
                }
                Ok(())
            };
            let first = {
                let pick: Pick = if host(0) { Some(&mut pick_fn) } else { None };
                sp.prefill(e, d, ids, *c0, cuts, &mut on_cut, pick)?
            };
            if !host(0) {
                pk.set(1);
            }
            *started = true;
            *prefill_s = t0.elapsed().as_secs_f64();
            if stops.contains(&first) {
                Some(FinishReason::Stop(first))
            } else if !sink.send_tokens(vec![first]) {
                Some(FinishReason::Cancelled)
            } else {
                *generated = 1;
                (*generated >= *max_new).then_some(FinishReason::Length)
            }
        } else if req.cancel.is_cancelled() || sink.is_closed() {
            Some(FinishReason::Cancelled)
        } else {
            let before = pk.get();
            let (kept, send, hit) = {
                let pick: Pick = if host(before) { Some(&mut pick_fn) } else { None };
                sp.step(e, d, *max_new - *generated, stops, pick)?
            };
            if pk.get() == before {
                pk.set(before + kept.len());
            }
            let mut out: Vec<u32> = kept[..send].to_vec();
            if hit {
                let t = out.pop().unwrap();
                if !out.is_empty() {
                    sink.send_tokens(out);
                }
                Some(FinishReason::Stop(t))
            } else {
                *generated += out.len();
                if !sink.send_tokens(out) {
                    Some(FinishReason::Cancelled)
                } else {
                    (*generated >= *max_new).then_some(FinishReason::Length)
                }
            }
        };
        *picked = pk.get();
        Ok(out)
    }

    /// The request on lane `li` has ended: the lane keeps its sequence and a checkpoint of its end.
    fn finish_run(&mut self, li: usize, reason: FinishReason) -> Result<()> {
        let mut run = self.lanes[li].run.take().context("no request on the lane")?;
        let sink = run.sink;
        if !run.started {
            // ended before its prompt was processed (cancelled, or nothing to generate): nothing new is in the caches
            sink.finish(Finish { reason, cached_prompt_tokens: 0, timings: None });
            return Ok(());
        }
        let r = (|| -> Result<()> {
            self.to_lane(li)?;
            let pos = self.e.pos as usize;
            self.lanes[li].hist = std::mem::take(&mut run.sp.hist);
            self.lanes[li].pos = pos;
            take_ckpt(&self.e, self.df.as_ref(), &mut self.ckpts, &mut self.clock, li as u8, pos)?;
            self.e.dev.sync()
        })();
        let s = &run.sp.st;
        let n = run.req.prompt_ids.len();
        tracing::info!(
            "{}: lane {li}: prompt {n} (cached {}) {:.0} tok/s | {} tokens {:.1} tok/s | windows {} plain {} drafted {} accepted {} copies {} vetoed {} | {:?}",
            run.req.id, run.c0, (n - run.c0) as f64 / run.prefill_s.max(1e-9), s.tokens, s.tokens.saturating_sub(1) as f64 / s.gen_s.max(1e-9),
            s.windows, s.plain, s.drafted, s.accepted, s.copies, s.vetoed, reason
        );
        let timings = Timings { prompt_n: n - run.c0, cache_n: run.c0, prompt_ms: run.prefill_s * 1e3, predicted_n: s.tokens, predicted_ms: s.gen_s * 1e3,
            draft: Some((s.drafted, s.accepted)) };
        match r {
            Ok(()) => {
                sink.finish(Finish { reason, cached_prompt_tokens: run.c0, timings: Some(timings) });
                Ok(())
            }
            Err(e) => {
                sink.finish(Finish::new(FinishReason::Error(format!("xe engine: {e:#}"))));
                Err(e)
            }
        }
    }
}
