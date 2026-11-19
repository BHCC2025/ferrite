//! Server-facing scheduler: thread-safe `submit`, one engine thread doing continuous batching over the
//! model engine's slots (prefill chunks and decode rows share each forward), host-side sampling, and a
//! per-slot prompt cache. See ENGINE_API.md.
use crate::model::{Engine, MtpRow, MtpSrc};
use anyhow::Result;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct SchedOptions {
    pub n_slots: usize,
    pub max_ctx: usize,
    /// prompt rows a request feeds per step (its chunk)
    pub prefill_chunk: usize,
    /// speculative depth cap (0 = off; MTP needs an MTP head in the checkpoint). Default FE_SPEC or 8.
    pub spec_depth: usize,
    /// an external block drafter (e.g. fe-dflash's DFlash2), built on the engine thread; replaces the MTP head
    pub drafter: Option<DrafterFactory>,
}
impl Default for SchedOptions {
    fn default() -> Self {
        let spec = std::env::var("FE_SPEC").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        SchedOptions { n_slots: 4, max_ctx: 131072, prefill_chunk: 2048, spec_depth: spec, drafter: None }
    }
}
impl SchedOptions {
    /// FE_POOL=1 (tools/serve-5080.sh with SLOTS > 1): the slots' contexts are one pool. A request that fits a slot
    /// (`max_ctx` tokens with its reply) runs beside others; a larger one takes slot 0 with every slot's context, up
    /// to `n_slots * max_ctx` tokens, and runs alone (the others wait in the queue; conversations in its way are
    /// parked first).
    pub fn pool(&self) -> bool {
        self.n_slots > 1 && std::env::var("FE_POOL").is_ok_and(|v| v == "1")
    }
    /// tokens a request may hold
    pub fn pool_ctx(&self) -> usize {
        if self.pool() { self.n_slots * self.max_ctx } else { self.max_ctx }
    }
    /// Rows a forward may carry (the engine's max_rows): FE_BATCH_ROWS, default the chunks of up to 4 prompts, so
    /// requests that arrive together prefill in one forward (M-wide prompt GEMMs) instead of a chunk a step.
    pub fn batch_rows(&self) -> usize {
        std::env::var("FE_BATCH_ROWS").ok().and_then(|v| v.parse().ok())
            .unwrap_or(self.prefill_chunk * self.n_slots.min(4)).max(self.prefill_chunk).max(self.n_slots)
    }
}
impl std::fmt::Debug for SchedOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SchedOptions {{ n_slots: {}, max_ctx: {}, prefill_chunk: {}, spec_depth: {}, drafter: {} }}",
            self.n_slots, self.max_ctx, self.prefill_chunk, self.spec_depth, self.drafter.is_some())
    }
}

/// A drafter that reads target feature taps (DFlash2-style): the scheduler absorbs every kept target row into it
/// and asks it for drafts after each stream's pending token.
pub trait Drafter: Send {
    /// target layers whose outputs it reads (passed to Engine::set_feature_taps)
    fn layers(&self) -> Vec<usize>;
    /// most drafts one call returns
    fn max_drafts(&self) -> usize;
    /// absorb `rows` target rows of `slot` at positions pos0.. (`feats` = their feature taps, contiguous)
    fn absorb(&mut self, e: &Engine, slot: usize, feats: fe_cuda::DevPtr, rows: usize, pos0: usize) -> Result<()>;
    /// up to max_drafts() greedy drafts following `anchor` (the pending token at position `pos`); may use e.logits
    fn draft(&mut self, e: &Engine, slot: usize, anchor: u32, pos: usize) -> Result<Vec<u32>>;
    /// a verify tree of up to `budget` drafted nodes following `anchor`: (tokens, parents), parent -1 = the anchor,
    /// otherwise an earlier node's index. None: this drafter drafts chains only.
    fn draft_tree(&mut self, _e: &Engine, _slot: usize, _anchor: u32, _pos: usize, _budget: usize)
        -> Result<Option<(Vec<u32>, Vec<i32>)>> {
        Ok(None)
    }
    /// split across the TP ranks: rank 0's draft / absorb calls are mirrored to rank 1 (Call::Draft / Call::Absorb),
    /// whose drafter half runs them through follow_draft / absorb_many
    fn tp_split(&self) -> bool {
        false
    }
    /// TP rank 1's side of rank 0's draft_many / draft_tree_many over the same (slot, anchor, pos, budget) list
    fn follow_draft(&mut self, _e: &Engine, _reqs: &[(usize, u32, usize, usize)]) -> Result<()> {
        Ok(())
    }
    /// `absorb` for several streams (slot, feats, rows, pos0) at once (a drafter may batch them into one pass)
    fn absorb_many(&mut self, e: &Engine, items: &[(usize, fe_cuda::DevPtr, usize, usize)]) -> Result<()> {
        for &(slot, feats, rows, pos0) in items {
            self.absorb(e, slot, feats, rows, pos0)?;
        }
        Ok(())
    }
    /// `draft` for several streams (slot, anchor, pos) at once (a drafter may batch them into one pass)
    fn draft_many(&mut self, e: &Engine, reqs: &[(usize, u32, usize)]) -> Result<Vec<Vec<u32>>> {
        reqs.iter().map(|&(slot, anchor, pos)| self.draft(e, slot, anchor, pos)).collect()
    }
    /// `draft_tree` for several streams (slot, anchor, pos, budget) at once; None: chains only
    /// Prompt cache (docs/prompt-cache.md): bytes of a slot's drafting state at a position (0: nothing to keep), and its
    /// save to / restore from device memory `dst` / `src` (a pool block's drafter part). Restore may target any slot.
    fn state_bytes(&self) -> usize {
        0
    }
    fn save_state(&self, _e: &Engine, _slot: usize, _pos: usize, _dst: fe_cuda::DevPtr) -> Result<()> {
        Ok(())
    }
    fn restore_state(&mut self, _e: &Engine, _slot: usize, _pos: usize, _src: fe_cuda::DevPtr) -> Result<()> {
        Ok(())
    }
    fn draft_tree_many(&mut self, e: &Engine, reqs: &[(usize, u32, usize, usize)])
        -> Result<Option<Vec<(Vec<u32>, Vec<i32>)>>> {
        let mut out = Vec::with_capacity(reqs.len());
        for &(slot, anchor, pos, budget) in reqs {
            match self.draft_tree(e, slot, anchor, pos, budget)? {
                Some(t) => out.push(t),
                None => return Ok(None),
            }
        }
        Ok(Some(out))
    }
}

/// The verify segment's row parents for a drafted tree (row 0 = the anchor; node j is row j + 1).
fn tree_rows(par: &[i32]) -> Vec<i32> {
    std::iter::once(-1).chain(par.iter().map(|&p| p + 1)).collect()
}

/// Greedy walk of a verified tree from the root: follow the child whose token is the target's argmax of the current
/// row. Returns the kept rows (root first) and the bonus token (the argmax after the last kept row).
pub fn tree_walk(toks: &[u32], par: &[i32], am: &[u32]) -> (Vec<usize>, u32) {
    let mut path = vec![0usize];
    loop {
        let node = *path.last().unwrap();
        let t = am[node];
        match (node + 1..toks.len()).find(|&c| par[c] == node as i32 && toks[c] == t) {
            Some(c) => path.push(c),
            None => return (path, t),
        }
    }
}
/// Builds the drafter on the engine thread (after the model is loaded).
pub type DrafterFactory = Arc<dyn Fn(&Engine, &SchedOptions) -> Result<Box<dyn Drafter>> + Send + Sync>;

/// Same meaning as fe_server::engine::SamplingParams.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
    pub stop_token_ids: Vec<u32>,
}
impl Default for Sampling {
    fn default() -> Self {
        Sampling { max_tokens: 256, min_tokens: 0, temperature: 0.0, top_p: 1.0, top_k: 0, min_p: 0.0,
            presence_penalty: 0.0, frequency_penalty: 0.0, repetition_penalty: 1.0, seed: None,
            stop_token_ids: Vec::new() }
    }
}
impl Sampling {
    fn greedy(&self) -> bool {
        self.temperature <= 0.0 && self.presence_penalty == 0.0 && self.frequency_penalty == 0.0
            && self.repetition_penalty == 1.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Finish {
    Stop(u32),
    Length,
    Cancelled,
    Error(String),
}

#[derive(Clone, Debug)]
pub enum Event {
    Tokens(Vec<u32>),
    Finished { reason: Finish, cached_prompt_tokens: usize },
}

/// What a request's guard asks for after an accepted token.
#[derive(Clone, Debug, PartialEq)]
pub enum GuardAct {
    None,
    /// the engine writes this token itself as the next one (the end of the reasoning block); the model goes on after it
    Force(u32),
    /// end the request here (finish reason Length)
    End,
    /// the tokens after this one are drawn with this sampling (reasoning -> answer, a boosted stretch and its end)
    Sampling(Box<Sampling>),
}

/// Per-request protections (the server's reasoning cap and loop breaker). The scheduler feeds it every token the
/// client gets, one at a time and in order, ACCEPTED tokens only (drafts that were verified, the engine's own forced
/// token included), and acts at exactly the token where it answers: tokens a verify accepted past that point are
/// given back (the verify keeps fewer rows), so what a guard does never depends on how many rows a step verified,
/// and a request whose guard never acts gets the tokens it would get without one. Drafting stays on throughout: a
/// forced token is absorbed by the drafter like any other kept row.
pub trait Guard: Send {
    fn feed(&mut self, tok: u32) -> GuardAct;
}

pub struct Request {
    pub id: String,
    /// protections fed with the accepted tokens (None: none)
    pub guard: Option<Box<dyn Guard>>,
    /// constrained opening: the reply starts with one of these token sequences, chosen token by token from the
    /// model's logits among the candidates still possible (tool_choice required with several functions); empty: free
    pub prefix_choices: Vec<Vec<u32>>,
    pub prompt_ids: Vec<u32>,
    /// snapshot points: ascending positions in prompt_ids where the prompt cache keeps a resumable state (the
    /// server's chat-template boundaries: each reply start, the end of the shared system prompt); empty: none
    pub snapshots: Vec<usize>,
    pub sampling: Sampling,
    pub cancel: Arc<AtomicBool>,
}

pub type Sink = Box<dyn FnMut(Event) -> bool + Send>;

enum Cmd {
    Submit(Request, Sink),
    Shutdown,
}

pub struct Scheduler {
    tx: mpsc::Sender<Cmd>,
    max_batch: usize,
    max_ctx: usize,
    thread: Option<std::thread::JoinHandle<()>>,
    /// why the engine thread stopped serving (a panic, a TP2 peer lost, repeated step failures); None while it serves
    dead: Arc<std::sync::Mutex<Option<String>>>,
    step_t0: Arc<std::sync::atomic::AtomicU64>,
    step_timeout: u64,
    /// speculative depth: a prompt needs len + 1 + spec <= the context (admit)
    spec: usize,
}

/// Consecutive failed steps after which the engine counts as broken (a sticky CUDA error fails every step).
const STEP_FAILS_FATAL: usize = 3;

impl Scheduler {
    /// Load the model on a new engine thread; returns once it is ready.
    pub fn start(model_dir: &Path, opts: SchedOptions) -> Result<Scheduler> {
        let (tx, rx) = mpsc::channel::<Cmd>();
        let (rtx, rrx) = mpsc::channel::<Result<(usize, usize)>>();
        let dir: PathBuf = model_dir.to_path_buf();
        let o = opts.clone();
        let dead = Arc::new(std::sync::Mutex::new(None::<String>));
        let dead_t = dead.clone();
        let step_t0 = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let step_t0_w = step_t0.clone();
        let thread = std::thread::Builder::new().name("ferrite-engine".into()).spawn(move || {
            let built = (|| -> Result<(Arc<fe_cuda::Context>, Engine)> {
                // whole model, or TP rank 0 (FE_TP_*: rank 1 runs `ferrite <model> follow`)
                // copy rows verify longer segments than the drafter's: the GDN replay keeps rows for them
                let hist = if o.spec_depth > 0 && copy_rows() > 0 { o.spec_depth.max(copy_rows() + 1).max(copy_hist_min(o.drafter.is_some(), o.spec_depth)) }
                    else { o.spec_depth };
                crate::tp::load_engine(&dir, o.n_slots, o.max_ctx, o.batch_rows(), hist, false)
            })();
            match built {
                Ok((_ctx, mut e)) => {
                    let df = match &o.drafter {
                        Some(f) => match f(&e, &o).and_then(|d| {
                            e.set_feature_taps(&d.layers())?;
                            Ok(d)
                        }) {
                            Ok(d) => Some(d),
                            Err(err) => {
                                let _ = rtx.send(Err(err));
                                return;
                            }
                        },
                        None => None,
                    };
                    let mut w = Worker::new(e, o);
                    w.step_t0 = step_t0_w;
                    w.df = df;
                    if let Err(err) = w.cache_init() {
                        let _ = rtx.send(Err(err));
                        return;
                    }
                    // requests it decodes at once: every slot, but with the block drafter alone (no MTP head to fall
                    // back on) one draft pass serves DF_PASS_STREAMS streams
                    let batch = if w.df.is_some() && !w.e.has_mtp() { w.o.n_slots.min(DF_PASS_STREAMS) } else { w.o.n_slots };
                    let _ = rtx.send(Ok((batch, w.spec)));
                    // a panic must not leave a server that accepts requests for an engine that is gone: record why
                    // (Scheduler::dead), so the server can report it and exit
                    let why = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || w.run(rx))) {
                        Ok(why) => why,
                        Err(p) => Some(format!("engine thread panicked: {}", p.downcast_ref::<&str>().map(|s| s.to_string())
                            .or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into()))),
                    };
                    if let Some(why) = why {
                        eprintln!("ferrite: the engine stopped: {why}");
                        *dead_t.lock().unwrap_or_else(|e| e.into_inner()) = Some(why);
                    }
                }
                Err(err) => {
                    let _ = rtx.send(Err(err));
                }
            }
        })?;
        let (max_batch, spec) = rrx.recv()??;
        Ok(Scheduler { tx, max_batch, max_ctx: opts.pool_ctx(), thread: Some(thread), dead, spec, step_t0, step_timeout: step_timeout_ms() })
    }
    /// Some(reason) once the engine thread has stopped serving for good (it then fails every request): the server
    /// reports it on /health and exits so that a supervisor can restart it.
    /// Positions of the context a prompt may not use: a prompt of max_context - prompt_reserve() tokens or more is
    /// refused (the first reply token and a full verify segment must fit after it).
    pub fn prompt_reserve(&self) -> usize {
        self.spec
    }
    pub fn dead(&self) -> Option<String> {
        if let Some(why) = self.dead.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            return Some(why);
        }
        let t0 = self.step_t0.load(Ordering::Relaxed);
        if t0 > 0 && now_ms().saturating_sub(t0) > self.step_timeout {
            return Some(format!("an engine step has run for {} s (FE_STEP_TIMEOUT {} s): the engine is hung",
                (now_ms() - t0) / 1000, self.step_timeout / 1000));
        }
        self.thread.as_ref().is_some_and(|t| t.is_finished()).then(|| "the engine thread exited".to_string())
    }
    /// (max_batch, max_context): the requests the loaded engine decodes at once (the slots; at most
    /// DF_PASS_STREAMS when DFlash2 is the only drafter) and the tokens a request may hold
    pub fn info(&self) -> (usize, usize) {
        (self.max_batch, self.max_ctx)
    }
    /// Non-blocking. `sink` runs on the engine thread; returning false cancels the request.
    pub fn submit(&self, req: Request, sink: Sink) {
        if let Err(mpsc::SendError(Cmd::Submit(_, mut sink))) = self.tx.send(Cmd::Submit(req, sink)) {
            sink(Event::Finished { reason: Finish::Error("engine stopped".into()), cached_prompt_tokens: 0 });
        }
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct Active {
    req: Request,
    sink: Sink,
    slot: usize,
    /// tokens its slot may hold: the slot's own context, or the whole pool for a wide request (SchedOptions::pool)
    cap: usize,
    /// prompt tokens already in the cache
    fed: usize,
    cached: usize,
    /// prompt cache: where this prompt is cut (ascending): the request's snapshot points (cut_points) and `share`
    cuts: Vec<usize>,
    /// share points (Worker::share_wait): positions on the 32-row grid where requests waiting in the queue resume;
    /// a shared-prefix entry is stored when the prefill reaches one
    share: Vec<usize>,
    out: Vec<u32>,
    /// next token to feed (sampled, not yet in the cache)
    pending: Option<u32>,
    /// MTP drafts following `pending`
    drafts: Vec<u32>,
    /// a drafted tree: row parents of the verify segment [pending, drafts..] (tree_rows)
    tree: Option<Vec<i32>>,
    /// FE_COPY_ROWS: the stream's suffix lookup (greedy streams), and this step's copy rows (first row, count)
    copier: Option<crate::copy::Copier>,
    copied: (usize, usize),
    /// row of the newest absorbed position in the head's last output (chained drafts read it)
    out_idx: usize,
    /// speculative depth (adaptive) and its acceptance average
    depth: usize,
    acc_ema: f32,
    rounds: usize,
    accepted: usize,
    rng: u64,
    t0: Instant,
    /// seconds from admission to the first token
    ttft: Option<f64>,
    /// a token the engine writes itself as the next one (GuardAct::Force)
    force: Option<u32>,
    /// constrained opening still running: (tokens of it written, the candidates that match them; all longer)
    opening: Option<(usize, Vec<Vec<u32>>)>,
}

impl Active {
    /// The next token is not the model's free choice (a forced token, a constrained opening): this step it decodes
    /// one plain row, without drafts.
    fn special(&self) -> bool {
        self.force.is_some() || self.opening.is_some()
    }
    /// The tokens a constrained opening allows next.
    fn opening_cands(&self) -> Option<Vec<u32>> {
        self.opening.as_ref().map(|(d, c)| c.iter().map(|s| s[*d]).collect())
    }
    fn opening_step(&mut self, t: u32) {
        if let Some((d, c)) = self.opening.as_mut() {
            c.retain(|s| s[*d] == t);
            *d += 1;
            if c.is_empty() || c.iter().any(|s| s.len() == *d) {
                self.opening = None;
            }
        }
    }
}

/// bf16 bits of -infinity (a masked logit).
const BF16_NEG_INF: u16 = 0xff80;

/// No infinity or NaN in a bf16 logit row (all exponent bits set).
fn row_finite(row: &[u16]) -> bool {
    !row.iter().any(|&h| h & 0x7f80 == 0x7f80)
}

/// Mask a host logit row: only the tokens of `only` stay (when given), the tokens of `ban` go.
fn mask_row(row: &mut [u16], only: Option<&[u32]>, ban: &[u32]) {
    if let Some(c) = only {
        let keep: Vec<(usize, u16)> = c.iter().filter(|&&t| (t as usize) < row.len()).map(|&t| (t as usize, row[t as usize])).collect();
        row.fill(BF16_NEG_INF);
        for (i, x) in keep {
            row[i] = x;
        }
    }
    for &t in ban {
        if let Some(x) = row.get_mut(t as usize) {
            *x = BF16_NEG_INF;
        }
    }
}

/// The first largest logit of a host row.
fn argmax_row(row: &[u16]) -> u32 {
    row.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &h)| {
        let x = f32::from_bits((h as u32) << 16);
        if x > b.1 { (i, x) } else { b }
    }).0 as u32
}

/// What a guard asked for, applied after its token went out; true: the request ends here.
fn apply_act(a: &mut Active, act: GuardAct, copy_hi: usize) -> bool {
    match act {
        GuardAct::None => false,
        GuardAct::End => true,
        GuardAct::Force(t) => {
            a.force = Some(t);
            false
        }
        GuardAct::Sampling(s) => {
            a.req.sampling = *s;
            // copy rows are for greedy streams (verified as trees)
            if !a.req.sampling.greedy() {
                a.copier = None;
            } else if a.copier.is_none() && copy_hi > 0 {
                a.copier = Some(crate::copy::Copier::new(COPY_LO, copy_hi));
            }
            false
        }
    }
}

/// Milliseconds since the Unix epoch (the step watchdog's clock).
fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// FE_STEP_TIMEOUT (seconds, default 120; the longest step, a 2K-row prompt chunk at 128K, takes a few seconds): a step that runs longer counts as a hung engine (a TP2 collective whose
/// peer is gone never returns); Scheduler::dead then reports it and the server exits.
fn step_timeout_ms() -> u64 {
    std::env::var("FE_STEP_TIMEOUT").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(120).max(10) * 1000
}

struct Worker {
    steps_run: u64,
    test_panic_step: Option<u64>,
    /// when the running step started (now_ms), 0 between steps (Scheduler::dead's hang check)
    step_t0: Arc<std::sync::atomic::AtomicU64>,
    e: Engine,
    o: SchedOptions,
    spec: usize,
    slot_tokens: Vec<Vec<u32>>,
    slot_busy: Vec<bool>,
    slot_used: Vec<u64>,
    tick: u64,
    waiting: VecDeque<(Request, Sink)>,
    active: Vec<Active>,
    host: Vec<u16>,
    /// external drafter (replaces the MTP head when set)
    df: Option<Box<dyn Drafter>>,
    /// decode/prefill fairness: decayed wall time of steps with prompt rows (t_pre) and without (t_dec) while
    /// some stream is decoding
    t_pre: f64,
    t_dec: f64,
    /// FE_DECODE_SHARE (default 0.3; 0 = off): least fraction of wall time spent in decode-only steps while
    /// prompts are pending and some stream decodes; FE_PREFILL_CAP (default 2048): prompt rows a step then.
    /// TP1, 1 decoding stream + 4 x 30K prompts: off = 3.4 tok/s during prefill (11% of solo), all prefilled in
    /// 56.7 s; 0.3/2048 = 15.9 (55%), 78.8 s; 0.3/1024 = 18.3 (61%), 86.5 s; 0.5/1024 = 26.9 (93%), 118.7 s.
    decode_share: f64,
    prefill_cap: usize,
    /// rows a forward may carry (SchedOptions::batch_rows, the engine's max_rows)
    batch_rows: usize,
    /// FE_ARRIVAL_MS (default 5, 0 = off): an idle worker woken by a request waits until no other has arrived for
    /// this long (at most 4x in all), so requests sent together start (prefill) in the same forward
    arrival: std::time::Duration,
    /// FE_TREE (default 0 until the GDN tree kernels land): greedy streams verify the drafter's trees
    tree: bool,
    /// FE_SCHED_PROF=1: wall time of the step's phases (draft, target forward + argmax, commit, drafter absorb,
    /// the rest) and the step count, printed and reset when a request finishes
    prof: Option<([f64; 5], usize)>,
    /// FE_SCHED_PROF=2: also a line a step (decode streams' slots, verify rows, accepted drafts and accepted node
    /// indices, prompt rows, phases): the trees are grown best-first, so a budget of B nodes is their first B and
    /// a round's path tells what any smaller budget would have accepted
    prof_steps: bool,
    /// FE_VERIFY_ROWS (default 32): the block drafter's verify rows a step, all streams together, split by
    /// recent acceptance (the EXL3 GEMV costs little more at 32 rows than at 16)
    verify_rows: usize,
    /// FE_VERIFY_ROWS is set: the block drafter's verify rows do not follow the decoding streams
    verify_fixed: bool,
    /// FE_COPY_ROWS (copy_rows) and FE_COPY_AGREE (default 3): a copy is grafted onto a tree only past this many
    /// rows of it the tree already holds
    copy: usize,
    copy_agree: usize,
    /// FE_CACHE=1 (docs/prompt-cache.md): snapshots at the request's snapshot points, longest-matching-snapshot resume;
    /// FE_CACHE_LIVE=1 also keeps today's full-history hit (not bitwise a prefill of the same tokens)
    cache: bool,
    cache_live: bool,
    /// FE_CACHE_RESTORE=0: cut and snapshot as with the cache, never resume (the uncached reference for the gates)
    cache_restore: bool,
    /// FE_CACHE_TRACE=1: a line per cache decision (resume source, save, park, prefix entry)
    cache_trace: bool,
    /// FE_CACHE_SAVE=0: cut at the points but take no snapshots (diagnostics). Points within DECODE_ROWS of the
    /// prompt end are ignored unless FE_CACHE_TAILCUT=1: their last segment (thinking off: `<think>\n\n</think>\n\n`
    /// after the generation prompt point) would take the decode GEMV path, and the clean-prose set then echoed its
    /// filler 14 of 16 times (4 of 16 without the cut). Cost: thinking-off turns resume one turn further back.
    cache_save_on: bool,
    tailcut: bool,
    /// per slot: its snapshots (ascending positions; rows [0, pos) of the slot's KV), and the free state blocks
    slot_snaps: Vec<Vec<crate::pcache::Snap>>,
    blocks: crate::pcache::Blocks,
    /// (slot, position, also a shared-prefix entry: the request's first snapshot point or a share point) snapshots to
    /// take after this step's drafter absorb
    snap_due: Vec<(usize, usize, bool)>,
    /// parked conversations and shared-prefix entries (FE_PARK_GB on the engine side), at most FE_PREFIX_ENTRIES (4)
    /// prefix entries
    park: crate::pcache::Park,
    prefix_max: usize,
    /// FE_CACHE_LEAD_MIN (lead_min): snapshot points from this position on are moved LEAD rows earlier (0 = off)
    lead_min: usize,
    /// FE_CACHE_TRACE: the queued requests whose wait for a shared prefix was already logged
    wait_said: std::collections::HashSet<String>,
    /// one host logit row for the picks that need it on a greedy stream (first-token stop ban, constrained opening)
    scratch: Vec<u16>,
    /// FE_FINITE_CHECK (default 1): a pick from a logit row that holds an infinity or a NaN ends its request with an
    /// error instead (sampled rows are scanned on the host; a greedy row never comes to the host: the value at the
    /// argmax of the row whose token no draft vouches for is read back, one bf16 a stream a step)
    finite: bool,
    /// picks so far, and FE_TEST_NONFINITE=n: pick n counts as non-finite (tests the path)
    picks: u64,
    test_nonfinite: Option<u64>,
}

/// FE_COPY_ROWS (default 0 = off): the most copy rows a greedy DFlash2 stream verifies (crate::copy). A copy starts
/// at COPY_LO rows grafted onto the drafter's tree past the rows it agrees with (a tree segment holds TREE_ROWS rows
/// at most); while copies land whole it doubles, and a lone stream then verifies the copy alone as a chain of up to
/// FE_COPY_ROWS + 1 rows (47: 48 rows, below the 49-row GEMV pass) without drafting; when most of one misses it
/// halves, and four that land nothing in a row silence the lookup for 16 asks.
fn copy_rows() -> usize {
    std::env::var("FE_COPY_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(0usize).min(crate::model::DEC_MAX - 1)
}

/// streams one DFlash2 draft pass serves (the launch scripts refuse more slots than this with DRAFT=dflash)
const DF_PASS_STREAMS: usize = 4;
/// GDN replay rows a slot keeps so that copy rows can be verified: a whole tree segment with the block drafter's
/// trees; with the MTP head a chain of `spec` drafts and a first copy (COPY_LO rows) behind the row they start from.
/// Every verify with copy rows is also bounded by the engine's hist_rows, so fewer rows only shorten copies.
fn copy_hist_min(block_drafter: bool, spec: usize) -> usize {
    if block_drafter { TREE_ROWS } else { (COPY_LO + 1 + spec).min(TREE_ROWS) }
}

/// shared context pool: the reply length a request is assumed to need at most when it is given one slot (a longer
/// reply ends at the slot's context; a request whose prompt leaves less room than this, or than its max_tokens, is
/// given the whole pool)
const WIDE_REPLY: usize = 4608;

/// rows a verify tree segment holds at most (the engine's tree mask is 32 bits)
const TREE_ROWS: usize = 32;
/// copy rows a proposal starts with: a 16-row tree and its copy fill one tree segment
const COPY_LO: usize = 16;
/// the verify's cost steps up past multiples of this many rows (the EXL3 GEMV's row tiles; TP2, one verify segment: 16 rows
/// 37.8 ms, 17 rows 45.9, 32 rows 53.2, 33 rows 58.4)
const COPY_TIER: usize = 16;
/// a match this long is trusted to cross a tier before its stream's copies have landed (TensorFold's confident_match)
const COPY_SURE: usize = 24;

/// Rows the EXL3 GEMV takes in one call: verify rows of all streams share it.
const DECODE_ROWS: usize = crate::model::X3_ROWS;
/// Prompt chunks that do not finish their prompt end at a multiple of this many positions: the chunked GDN kernel
/// works in 32-row steps from each segment's first row, so any other cut changes a prompt's bits (and through near
/// ties its text) with whatever else shares the steps. Cuts also never leave a tail of <= X3_ROWS rows (those take
/// the decode GEMV / split attention, not the prompt kernels).
const PROMPT_ALIGN: usize = 32;
/// Shared-prefix entries hold at least this many tokens (a shorter prefix is cheaper to compute than to store).
const PREFIX_MIN: usize = 1024;
/// A waiting request uses a leader's existing share point up to this many rows before its own (Worker::share_wait),
/// and a leader takes at most SHARE_MAX share points (each stores a prefix entry).
const SHARE_NEAR: usize = 256;
const SHARE_MAX: usize = 4;

/// Rows of the next chunk of a prompt of `len` ids with `fed` already in, given `avail` rows (0: wait a step).
/// `base`: where this request's prefill started (0, or the length of a reused cached prefix): cuts sit on the
/// 32-row grid from there, so a cache hit's prompt is also cut-invariant (its first segment starts at `base`).
pub fn prompt_cut(len: usize, base: usize, fed: usize, avail: usize) -> usize {
    let left = len - fed;
    if left <= avail {
        return left;
    }
    let mut end = base + (fed + avail - base) / PROMPT_ALIGN * PROMPT_ALIGN;
    while end > fed && len - end <= crate::model::X3_ROWS {
        end -= PROMPT_ALIGN;
    }
    end.saturating_sub(fed)
}

/// Where a queued prompt `p` (cut points `mine`) can share the prefill of an active prompt `q` (cut points `theirs`,
/// `fed` rows in, share points `share`): the end of their common tokens, at least DECODE_ROWS + 1 rows before either
/// prompt's end, moved back onto the 32-row grid from the last cut point of both before it; or an existing share
/// point up to SHARE_NEAR rows before that. None: nothing to share (the common part is shorter than PREFIX_MIN, the
/// active prompt is already past it, their cut points differ, or it has SHARE_MAX share points).
pub fn share_point(p: &[u32], mine: &[usize], q: &[u32], theirs: &[usize], fed: usize, share: &[usize]) -> Option<usize> {
    let l = p.iter().zip(q).take_while(|(x, y)| x == y).count();
    let lim = l.min(p.len().saturating_sub(DECODE_ROWS + 1)).min(q.len().saturating_sub(DECODE_ROWS + 1));
    // the last cut point of each inside the common text: the same position, or their grids differ
    let last = |pts: &[usize]| pts.iter().rev().find(|&&x| x <= lim).copied().unwrap_or(0);
    let base = last(theirs);
    if last(mine) != base {
        return None;
    }
    let pos = base + (lim - base) / PROMPT_ALIGN * PROMPT_ALIGN;
    let pos = share.iter().copied().filter(|&c| c > fed && c <= pos && pos - c <= SHARE_NEAR).max().unwrap_or(pos);
    // too short to store, or the active prompt is past it (its state there was not kept)
    (pos >= PREFIX_MIN && pos > fed && (share.contains(&pos) || share.len() < SHARE_MAX)).then_some(pos)
}

/// MemAvailable from /proc/meminfo, in bytes.
fn mem_available() -> Option<usize> {
    let m = std::fs::read_to_string("/proc/meminfo").ok()?;
    let l = m.lines().find(|l| l.starts_with("MemAvailable:"))?;
    l.split_whitespace().nth(1)?.parse::<usize>().ok().map(|kb| kb * 1024)
}

/// The request's snapshot points the prompt cuts and snapshots use (`tailcut` false: without those within
/// DECODE_ROWS before the prompt end, whose last segment would take the decode GEMV; the end itself stays).
fn cut_points(r: &Request, tailcut: bool) -> Vec<usize> {
    let len = r.prompt_ids.len();
    r.snapshots.iter().copied().filter(|&p| tailcut || p >= len || len - p > DECODE_ROWS).collect()
}

/// FE_CACHE_LEAD_MIN (default 4096 on small-memory devices, else 0 = off): snapshot points at or past this many
/// tokens move back onto the prompt's chunk grid. The point of a turn's generation prompt lies within DECODE_ROWS of
/// the prompt end, where nothing is cut or saved, so the next turn could only resume a turn further back and computed
/// a long first prompt twice. Moved back to the last whole chunk before it (counted from the point before), it is
/// where a forward ends anyway: the state is saved there at no cost, and the next turn resumes from it, computing at
/// most one chunk of the old prompt again. The points stay a function of the prompt and the chunk size alone, so a
/// resumed reply is still the full computation's.
fn lead_min() -> usize {
    std::env::var("FE_CACHE_LEAD_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(if crate::lowmem::on() { 4096 } else { 0 })
}

/// `points` (ascending) with every point p >= `min` replaced by the end of the last whole `chunk` before it, counted
/// from the point before it (0 for the first) and leaving more than DECODE_ROWS rows up to p; dropped when no whole
/// chunk fits.
pub fn lead_points(points: &[usize], min: usize, chunk: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(points.len());
    for &p in points {
        let base = out.last().copied().unwrap_or(0);
        let q = if p >= min && chunk > 0 {
            base + p.saturating_sub(DECODE_ROWS + 1).saturating_sub(base) / chunk * chunk
        } else {
            p
        };
        if q > base {
            out.push(q);
        }
    }
    out
}

/// The snapshot points every request is cut and saved at (in every mode): each point moves back to the end of the
/// last whole grid step before it, counted from the point before it (0 for the first) and leaving more than
/// DECODE_ROWS rows up to it; the step is the prompt chunk for points at or past `min` (FE_CACHE_LEAD_MIN, when on)
/// and PROMPT_ALIGN (32 rows) for the others. A turn's generation-prompt point (within DECODE_ROWS of its prompt
/// end, where nothing could be cut) thus lands at most 48 rows back, where the next turn resumes; and since the
/// points are a function of the points alone, the next turn cuts at the same place: a resumed reply is the full
/// computation's (tools/bench/agentturns.py, cache on vs off).
pub fn turn_points(points: &[usize], min: usize, chunk: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(points.len());
    for &p in points {
        let base = out.last().copied().unwrap_or(0);
        let step = if min > 0 && p >= min && chunk > 0 { chunk } else { PROMPT_ALIGN };
        let q = base + p.saturating_sub(DECODE_ROWS + 1).saturating_sub(base) / step * step;
        if q > base {
            out.push(q);
        }
    }
    out
}

/// Host rows for the sampled segments of a step: segment i owns logit rows lr_first[i]..lr_first[i + 1] (the last
/// up to `nl`); each sampled segment with logit rows gets the next free host rows, the others usize::MAX.
/// Returns (first host row per segment, host rows needed).
fn sampled_host_rows(lr_first: &[usize], nl: usize, sampled: &[bool]) -> (Vec<usize>, usize) {
    let mut out = vec![usize::MAX; lr_first.len()];
    let mut hr = 0;
    for i in 0..lr_first.len() {
        let k = lr_first.get(i + 1).copied().unwrap_or(nl) - lr_first[i];
        if k > 0 && sampled[i] {
            out[i] = hr;
            hr += k;
        }
    }
    (out, hr)
}

/// FE_VERIFY_ROWS (default 48: one g6 GEMV pass; 4-stream code TP2 286 vs 264 at 32, TP1 208 vs 183; 64 gains
/// nothing, its 49-64 rows run as two 32-row passes), kept within the engine's decode-row limit.
fn verify_rows(n_slots: usize) -> usize {
    std::env::var("FE_VERIFY_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(3 * DECODE_ROWS)
        .clamp(n_slots.max(2), crate::model::DEC_MAX)
}

impl Worker {
    fn new(e: Engine, o: SchedOptions) -> Worker {
        let n = o.n_slots;
        let v = e.cfg.vocab;
        // the engine may keep more history rows than the drafter's depth (copy rows)
        let spec = o.spec_depth.min(e.hist_rows);
        let copy = if spec > 0 && e.hist_rows >= copy_hist_min(o.drafter.is_some(), o.spec_depth) { copy_rows().min(e.hist_rows - 1) } else { 0 };
        // prompt cache: DFlash2 / MTP / plain (FE_CACHE_MTP=0 turns it off with the MTP head loaded): the head's KV
        // rows are parked and restored with the target's, and the one row a resume leaves it without only costs a draft
        let want_cache = std::env::var("FE_CACHE").is_ok_and(|v| v == "1");
        let cache = want_cache && (!e.has_mtp() || std::env::var("FE_CACHE_MTP").map(|v| v != "0").unwrap_or(true));
        if want_cache && !cache {
            eprintln!("ferrite: FE_CACHE off with the MTP head loaded (FE_CACHE_MTP=0): no prompt reuse");
        }
        Worker { steps_run: 0, test_panic_step: std::env::var("FE_TEST_PANIC_STEP").ok().and_then(|v| v.parse().ok()), step_t0: Arc::new(std::sync::atomic::AtomicU64::new(0)), e, spec, slot_tokens: vec![Vec::new(); n], slot_busy: vec![false; n], slot_used: vec![0; n], tick: 0,
            waiting: VecDeque::new(), active: Vec::new(), host: vec![0; verify_rows(n).max(DECODE_ROWS).max(n) * v], df: None,
            t_pre: 0.0, t_dec: 0.0,
            decode_share: std::env::var("FE_DECODE_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.3f64).clamp(0.0, 0.95),
            prefill_cap: std::env::var("FE_PREFILL_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(2048usize).max(4 * PROMPT_ALIGN),
            batch_rows: o.batch_rows(),
            arrival: std::time::Duration::from_micros(
                (std::env::var("FE_ARRIVAL_MS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(5.0) * 1e3) as u64),
            tree: std::env::var("FE_TREE").is_ok_and(|v| v == "1"),
            prof: std::env::var("FE_SCHED_PROF").is_ok_and(|v| v == "1" || v == "2").then_some(([0.0; 5], 0)),
            prof_steps: std::env::var("FE_SCHED_PROF").is_ok_and(|v| v == "2"), verify_rows: verify_rows(n), verify_fixed: std::env::var("FE_VERIFY_ROWS").is_ok(),
            copy,
            copy_agree: std::env::var("FE_COPY_AGREE").ok().and_then(|v| v.parse().ok()).unwrap_or(3),
            cache,
            cache_live: std::env::var("FE_CACHE_LIVE").is_ok_and(|v| v == "1"),
            cache_restore: std::env::var("FE_CACHE_RESTORE").map(|v| v != "0").unwrap_or(true),
            cache_trace: std::env::var("FE_CACHE_TRACE").is_ok_and(|v| v == "1"),
            cache_save_on: std::env::var("FE_CACHE_SAVE").map(|v| v != "0").unwrap_or(true),
            tailcut: std::env::var("FE_CACHE_TAILCUT").is_ok_and(|v| v == "1"),
            slot_snaps: vec![Vec::new(); n], blocks: crate::pcache::Blocks::new(0), snap_due: Vec::new(),
            park: crate::pcache::Park::new(0),
            prefix_max: std::env::var("FE_PREFIX_ENTRIES").ok().and_then(|v| v.parse().ok()).unwrap_or(4),
            lead_min: lead_min(),
            wait_said: std::collections::HashSet::new(),
            scratch: Vec::new(),
            finite: std::env::var("FE_FINITE_CHECK").map(|v| v != "0").unwrap_or(true),
            picks: 0,
            test_nonfinite: std::env::var("FE_TEST_NONFINITE").ok().and_then(|v| v.parse().ok()),
            o }
    }

    /// Serve until Shutdown (None) or until the engine is broken (Some(reason)).
    fn run(mut self, rx: mpsc::Receiver<Cmd>) -> Option<String> {
        let mut fails = 0;
        loop {
            // intake
            let mut cmds = Vec::new();
            if self.active.is_empty() && self.waiting.is_empty() {
                match rx.recv() {
                    Ok(c) => cmds.push(c),
                    Err(_) => return None,
                }
                // woken from idle: give the requests sent together a moment to arrive (one prefill forward)
                let cap = Instant::now() + 4 * self.arrival;
                while !self.arrival.is_zero() {
                    let left = self.arrival.min(cap.saturating_duration_since(Instant::now()));
                    match rx.recv_timeout(left) {
                        Ok(c) => cmds.push(c),
                        Err(_) => break,
                    }
                }
            }
            while let Ok(c) = rx.try_recv() {
                cmds.push(c);
            }
            for c in cmds {
                match c {
                    Cmd::Submit(mut r, s) => {
                        // the cut points are a function of the prompt alone, whatever the cache setting (greedy text
                        // must not depend on it)
                        r.snapshots = turn_points(&r.snapshots, self.lead_min, self.o.prefill_chunk);
                        self.waiting.push_back((r, s))
                    }
                    Cmd::Shutdown => return None,
                }
            }
            self.admit();
            if self.active.is_empty() {
                continue;
            }
            self.step_t0.store(now_ms(), Ordering::Relaxed);
            self.steps_run += 1;
            // FE_TEST_PANIC_STEP=n: panic in step n (tests that a dead engine is reported and the server exits)
            if self.test_panic_step == Some(self.steps_run) {
                panic!("FE_TEST_PANIC_STEP {}", self.steps_run);
            }
            let r = self.step();
            self.step_t0.store(0, Ordering::Relaxed);
            match r {
                Ok(()) => fails = 0,
                Err(err) => {
                    let msg = format!("{err:#}");
                    eprintln!("ferrite: step failed: {msg}");
                    fails += 1;
                    // TP2: rank 1 may have run a call rank 0 did not finish (or be gone): the ranks no longer agree
                    let fatal = self.e.tp_rank().is_some() || fails >= STEP_FAILS_FATAL;
                    for mut a in self.active.drain(..) {
                        (a.sink)(Event::Finished { reason: Finish::Error(msg.clone()), cached_prompt_tokens: a.cached });
                        self.slot_busy[a.slot] = false;
                        self.slot_tokens[a.slot].clear();
                        let b: Vec<usize> = self.slot_snaps[a.slot].drain(..).map(|x| x.blk).collect();
                        self.blocks.release(b);
                    }
                    if fatal {
                        for (_, mut sink) in self.waiting.drain(..) {
                            sink(Event::Finished { reason: Finish::Error(format!("engine stopped: {msg}")), cached_prompt_tokens: 0 });
                        }
                        return Some(format!("step failed: {msg}"));
                    }
                }
            }
        }
    }

    /// Give waiting requests a slot: the free slot whose cache is the longest proper prefix of the prompt,
    /// else the least recently used free slot (reset).
    fn admit(&mut self) {
        // a prompt group is being taken in (a few short steps): its forwards must be the group's chunks alone
        if self.e.group_slot().is_some() {
            return;
        }
        // requests waiting for another one to prefill their shared prefix (put back in order at the end)
        let mut held: Vec<(Request, Sink)> = Vec::new();
        self.admit_inner(&mut held);
        for r in held.into_iter().rev() {
            self.waiting.push_front(r);
        }
    }

    /// The longest resume point the caches hold for prompt `p` right now (a free slot's snapshot, a parked
    /// conversation or a shared-prefix entry); 0: none.
    fn cached_pos(&self, p: &[u32]) -> usize {
        let slot = (0..self.o.n_slots).filter(|&s| !self.slot_busy[s])
            .filter_map(|s| crate::pcache::best_snap(&self.slot_tokens[s], &self.slot_snaps[s], p, self.min_tail()))
            .map(|x| x.pos).max();
        slot.max(self.park.best(p, self.min_tail()).map(|(_, x)| x.pos)).unwrap_or(0)
    }

    /// Shared prefix, computed once (docs/prompt-cache.md). A request waits in the queue while an active request is
    /// still prefilling text both start with, further than anything cached covers: the active request (the leader)
    /// stores a shared-prefix entry when its prefill gets there, and the waiting one resumes from it. Requests that
    /// arrive together and start with the same text (parallel lanes of one job, agents with one system prompt)
    /// therefore compute that text once, even though no cache held it when they were queued.
    ///
    /// Where they share: the end of their common tokens, moved back onto the 32-row grid from the last snapshot
    /// point before it (and at least DECODE_ROWS + 1 rows before either prompt's end). A cut on that grid does not
    /// change a prompt's rows (PROMPT_ALIGN), and the grid after it stays where it was, so the leader and the
    /// waiting request both produce exactly what they produce alone. The position becomes one of the leader's cut
    /// points (`Active::share`); a leader's share point within SHARE_NEAR rows before it is used instead, so lanes
    /// whose tails start alike share one entry.
    /// Returns the position the request waits for, or None when it should be admitted now.
    fn share_wait(&mut self, req: &Request) -> Option<usize> {
        let p = &req.prompt_ids;
        if p.len() <= PREFIX_MIN + DECODE_ROWS {
            return None;
        }
        let mine = cut_points(req, self.tailcut);
        let mut best: Option<(usize, usize)> = None; // (active index, position)
        for (ai, a) in self.active.iter().enumerate() {
            let q = &a.req.prompt_ids;
            if a.pending.is_some() || a.fed >= q.len() {
                continue; // its prompt is in
            }
            let Some(pos) = share_point(p, &mine, q, &cut_points(&a.req, self.tailcut), a.fed, &a.share) else { continue };
            if best.is_none_or(|b| pos > b.1) {
                best = Some((ai, pos));
            }
        }
        let (ai, pos) = best.filter(|b| b.1 > self.cached_pos(p))?;
        let a = &mut self.active[ai];
        if !a.share.contains(&pos) {
            a.share.push(pos);
        }
        if let Err(i) = a.cuts.binary_search(&pos) {
            a.cuts.insert(i, pos);
        }
        if self.cache_trace && self.wait_said.insert(req.id.clone()) {
            eprintln!("cache: {} waits for {}'s prefill to reach {pos} (shared prefix)", req.id, self.active[ai].req.id);
        }
        Some(pos)
    }

    /// a wide request (the whole context pool in slot 0) is running
    fn wide_active(&self) -> bool {
        self.active.iter().any(|a| a.cap > self.o.max_ctx)
    }

    fn admit_inner(&mut self, held: &mut Vec<(Request, Sink)>) {
        while let Some((req, mut sink)) = self.waiting.pop_front() {
            if self.cache && self.cache_restore && self.cache_save_on && self.prefix_max > 0 && self.share_wait(&req).is_some() {
                held.push((req, sink));
                continue;
            }
            self.wait_said.remove(&req.id);
            let p = &req.prompt_ids;
            if p.is_empty() || p.len() + 1 + self.spec > self.o.pool_ctx() {
                sink(Event::Finished { reason: Finish::Error(format!("prompt of {} tokens (max {})", p.len(),
                    self.o.pool_ctx() - 1 - self.spec)), cached_prompt_tokens: 0 });
                continue;
            }
            // shared context pool: a request that does not fit one slot with its reply takes them all, alone
            let wide = self.o.pool() && p.len() + 1 + self.spec + req.sampling.max_tokens.min(WIDE_REPLY) > self.o.max_ctx;
            if self.o.pool() && (self.wide_active() || (wide && !self.active.is_empty())) {
                self.waiting.push_front((req, sink));
                return;
            }
            if wide {
                // every other slot's conversation goes to the park pool: this one's rows will run through them
                let mut failed = None;
                for s in 1..self.o.n_slots {
                    if let Err(err) = self.park_slot(s).and_then(|_| self.e.reset_slot(s)) {
                        failed = Some(err);
                        break;
                    }
                    self.slot_tokens[s].clear();
                }
                if let Some(err) = failed {
                    sink(Event::Finished { reason: Finish::Error(format!("{err:#}")), cached_prompt_tokens: 0 });
                    continue;
                }
            } else if self.o.pool() && self.e.slot_len[0] > self.o.max_ctx {
                // slot 0 still holds a wide conversation: park it before the other slots are written again
                if let Err(err) = self.park_slot(0).and_then(|_| self.e.reset_slot(0)) {
                    sink(Event::Finished { reason: Finish::Error(format!("{err:#}")), cached_prompt_tokens: 0 });
                    continue;
                }
                self.slot_tokens[0].clear();
            }
            let free: Vec<usize> = (0..self.o.n_slots).filter(|&s| !self.slot_busy[s] && (!wide || s == 0)).collect();
            if free.is_empty() {
                self.waiting.push_front((req, sink));
                return;
            }
            let mut best: Option<(usize, usize)> = None;
            // a slot's live history (FE_CACHE_LIVE=1) holds the previous reply's rows as decoded, not as a prompt
            // computes them: resuming there changes the text, so it is opt-in (diagnostics), cache or not
            if self.cache_live {
                for &s in &free {
                    let c = &self.slot_tokens[s];
                    if !c.is_empty() && c.len() < p.len() && p[..c.len()] == c[..] && best.is_none_or(|b| c.len() > b.1) {
                        best = Some((s, c.len()));
                    }
                }
            }
            // prompt cache: the longest matching snapshot of a free slot, if it beats the live-history hit
            let mut snap: Option<(usize, crate::pcache::Snap)> = None;
            if self.cache && self.cache_restore {
                for &s in &free {
                    if let Some(x) = crate::pcache::best_snap(&self.slot_tokens[s], &self.slot_snaps[s], p, self.min_tail()) {
                        if snap.is_none_or(|(_, y)| x.pos > y.pos) && best.is_none_or(|b| x.pos > b.1) {
                            snap = Some((s, x));
                        }
                    }
                }
            }
            // a parked conversation or a shared-prefix entry that matches longer than any free slot: restored into
            // the least recently used free slot (whose own conversation is parked first)
            let parked = if self.cache && self.cache_restore { self.park.best(p, self.min_tail()) } else { None };
            let from_park = parked.filter(|(_, x)| snap.is_none_or(|(_, y)| x.pos > y.pos) && best.is_none_or(|b| x.pos > b.1));
            let resumed = if let Some((ei, x)) = from_park {
                let s = *free.iter().min_by_key(|&&s| self.slot_used[s]).unwrap();
                match self.park_restore(s, ei, x) {
                    Ok(true) => Some((s, x.pos)),
                    // the entry had to make room for the slot's own conversation: a cache miss, computed in full
                    Ok(false) => match self.e.reset_slot(s) {
                        Ok(()) => {
                            self.slot_tokens[s].clear();
                            Some((s, 0))
                        }
                        Err(err) => {
                            sink(Event::Finished { reason: Finish::Error(format!("{err:#}")), cached_prompt_tokens: 0 });
                            continue;
                        }
                    },
                    Err(err) => {
                        sink(Event::Finished { reason: Finish::Error(format!("{err:#}")), cached_prompt_tokens: 0 });
                        continue;
                    }
                }
            } else {
                match snap {
                    Some((s, x)) => match self.cache_restore(s, x) {
                        Ok(()) => Some((s, x.pos)),
                        Err(err) => {
                            sink(Event::Finished { reason: Finish::Error(format!("{err:#}")), cached_prompt_tokens: 0 });
                            continue;
                        }
                    },
                    None => None,
                }
            };
            if self.cache_trace {
                let src = match (from_park, snap, resumed) {
                    (Some((ei, x)), _, Some(_)) => format!("park entry {} ({}, {} tokens, snap blk {}) at {}", ei,
                        if self.park.entries.get(ei).is_some_and(|e| e.prefix) { "prefix" } else { "conv" },
                        self.park.entries.get(ei).map_or(0, |e| e.tokens.len()), x.blk, x.pos),
                    (None, Some((s, x)), Some(_)) => format!("own slot {s} snap blk {} at {}", x.blk, x.pos),
                    _ => "no snapshot".into(),
                };
                eprintln!("cache: {} ({} tokens): {src}", req.id, p.len());
            }
            let (slot, cached) = match resumed.or(best) {
                Some(b) => b,
                None => {
                    let s = *free.iter().min_by_key(|&&s| self.slot_used[s]).unwrap();
                    if let Err(err) = self.park_slot(s).and_then(|_| self.e.reset_slot(s)) {
                        sink(Event::Finished { reason: Finish::Error(format!("{err:#}")), cached_prompt_tokens: 0 });
                        continue;
                    }
                    self.slot_tokens[s].clear();
                    (s, 0)
                }
            };
            self.slot_busy[slot] = true;
            self.tick += 1;
            self.slot_used[slot] = self.tick;
            let seed = req.sampling.seed.unwrap_or(0x9e37_79b9_7f4a_7c15 ^ self.tick);
            let depth = if self.df.is_some() { self.spec } else { self.spec.min(4) };
            // copy rows: greedy streams of the block drafter's trees, or of the MTP head (its chain and the copy
            // share a verify tree)
            let copier = (self.copy > 0 && req.sampling.greedy() && ((self.tree && self.df.is_some()) || (self.df.is_none() && self.e.has_mtp())))
                .then(|| crate::copy::Copier::new(COPY_LO, self.copy.max(COPY_LO)));
            // every mode cuts a prompt at its snapshot points (the cache only decides whether a state is saved there):
            // the chunked GDN grid restarts at each cut, so a cut taken in one mode and not in another changes the bits
            let cuts = cut_points(&req, self.tailcut);
            let cap = if wide { self.o.pool_ctx() } else { self.o.max_ctx };
            let cands: Vec<Vec<u32>> = req.prefix_choices.iter().filter(|c| !c.is_empty()).cloned().collect();
            let opening = (!cands.is_empty()).then_some((0, cands));
            self.active.push(Active { req, sink, slot, cap, fed: cached, cached, cuts, share: Vec::new(), out: Vec::new(), pending: None,
                drafts: Vec::new(), tree: None, copier, copied: (0, 0), out_idx: 0, depth, acc_ema: depth as f32, rounds: 0,
                accepted: 0, rng: seed | 1, t0: Instant::now(), ttft: None, force: None, opening });
        }
    }

    fn step(&mut self) -> Result<()> {
        // drop cancelled requests first
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].req.cancel.load(Ordering::Acquire) {
                let a = self.active.swap_remove(i);
                self.finish(a, Finish::Cancelled);
            } else {
                i += 1;
            }
        }
        if self.active.is_empty() {
            return Ok(());
        }
        let spec = self.spec > 0;
        // a stream whose next token is forced or constrained decodes one plain row this step (the draft drawn after
        // its last step is dropped; the step's kept row is absorbed as always, so drafting goes on right after)
        for a in self.active.iter_mut().filter(|a| a.pending.is_some() && a.special()) {
            a.drafts.clear();
            a.tree = None;
            a.copied = (0, 0);
        }
        // copy rows a stream may be given when a guard switches it to greedy sampling (0: none in this configuration)
        let copy_hi = if self.copy > 0 && ((self.tree && self.df.is_some()) || (self.df.is_none() && self.e.has_mtp())) {
            self.copy.max(COPY_LO) } else { 0 };
        // FE_SCHED_PROF=2: this step's phases (from the totals) and per decode stream (rows, accepted, nodes, slot)
        let t_before = self.prof.map(|p| p.0);
        let mut trace: Vec<(usize, usize, Vec<usize>, usize, (usize, usize))> = Vec::new();
        let mut tp = Instant::now();
        let mut lap = |prof: &mut Option<([f64; 5], usize)>, i: usize| {
            if let Some((t, _)) = prof.as_mut() {
                t[i] += tp.elapsed().as_secs_f64();
                tp = Instant::now();
            }
        };
        // With both drafters loaded (FE_DRAFT=dflash and the MTP head), the block drafter serves steps with few
        // decoding streams (its drafts run one stream at a time) and MTP the rest (FE_DF_STREAMS, default 2).
        let ndec_now = self.active.iter().filter(|a| a.pending.is_some()).count();
        let df_streams = std::env::var("FE_DF_STREAMS").ok().and_then(|v| v.parse().ok()).unwrap_or(2usize);
        let use_df = self.df.is_some() && (!self.e.has_mtp() || ndec_now <= df_streams);
        // 1a. external drafter: one block of drafts per decoding stream from its pending token
        if spec && use_df {
            // verify rows a stream: what its recent acceptance can use (FE_VERIFY_GAIN (1.0) x acc_ema + 2, so a
            // prose stream accepting ~1.3 asks for ~3 rows and a code stream ~9)
            // By decoding streams, when neither is set: three or more share 32 verify rows at gain 8 (each asks for
            // its whole block, and one 32-row GEMV pass carries them), fewer keep 48 rows at gain 1. Total tokens/s
            // of 3 / 4 parallel code streams on one GB10: 145 / 168 against 133 / 149; one and two streams are
            // unchanged. The rows a stream verifies never change its tokens.
            // One node only: on two nodes (TP2) the split policy measured 444 / 426 tok/s for 4 code / edit streams
            // against 470 / 473 with 48 rows at gain 1 (each rank's half GEMV is cheap, the bigger trees pay), so TP2
            // keeps the fixed policy.
            let many = ndec_now >= 3 && self.e.tp_rank().is_none();
            let gain: f32 = std::env::var("FE_VERIFY_GAIN").ok().and_then(|v| v.parse().ok()).unwrap_or(if many { 8.0 } else { 1.0 });
            if !self.verify_fixed {
                self.verify_rows = if many { 2 * DECODE_ROWS } else { 3 * DECODE_ROWS }.clamp(self.o.n_slots.max(2), crate::model::DEC_MAX);
            }
            let want_rows = |a: &Active| (2.0 + gain * a.acc_ema.max(0.0)).round().clamp(2.0, (self.spec + 1) as f32);
            let wsum: f32 = self.active.iter().filter(|a| a.pending.is_some()).map(want_rows).sum();
            // up to DECODE_ROWS (16) rows a step cost the verify GEMV next to nothing, so a small demand is
            // stretched to fill them (one stream gets its whole block); above that the streams get what they ask for,
            // shrunk to FE_VERIFY_ROWS
            let scale = if wsum < DECODE_ROWS as f32 { DECODE_ROWS as f32 / wsum.max(1.0) }
                else { (self.verify_rows as f32 / wsum).min(1.0) };
            // copy rows: a lone stream whose copies keep landing whole verifies its copy alone, without drafting
            if self.copy > 0 && ndec_now == 1 {
                for a in self.active.iter_mut().filter(|a| a.pending.is_some() && !a.special()) {
                    let Some(c) = a.copier.as_mut() else { continue };
                    if c.extra <= COPY_LO {
                        continue;
                    }
                    // 32 or 48 rows (the GEMV's cost steps up past 16-row multiples)
                    let n = if c.extra >= self.copy { self.copy } else { c.extra.min(TREE_ROWS - 1) }
                        .min(self.verify_rows - 1).min(self.e.hist_rows - 1)
                        .min(a.req.sampling.max_tokens.saturating_sub(a.out.len()))
                        .min(a.cap.saturating_sub(self.e.slot_len[a.slot] + 3));
                    c.sync(&a.req.prompt_ids, &a.out);
                    let (copy, _) = c.propose(n);
                    if !copy.is_empty() {
                        a.copied = (1, copy.len());
                        a.drafts = copy;
                        a.tree = None;
                    }
                }
            }
            let df = self.df.as_mut().unwrap();
            // (active index, slot, anchor, pos, cap, tree budget) of every stream that drafts this step
            let mut want = Vec::new();
            for (ai, a) in self.active.iter().enumerate().filter(|(_, a)| a.copied.1 == 0 && !a.special()) {
                if let Some(t) = a.pending {
                    let rows = ((want_rows(a) * scale) as usize).max(2);
                    let cap = (rows - 1).min(self.spec);
                    let pos = self.e.slot_len[a.slot];
                    if a.cap.saturating_sub(pos + 2) >= df.max_drafts() + 1 {
                        // trees: greedy streams only; the tree's rows must fit the GDN replay (hist_rows)
                        let budget = if self.tree && a.req.sampling.greedy() { cap.min(self.spec.saturating_sub(1)) } else { 0 };
                        want.push((ai, a.slot, t, pos, cap, budget));
                    }
                }
            }
            // TP split drafter: rank 1 runs the same pass (one per call below)
            if !want.is_empty() && df.tp_split() {
                self.e.mirror_call(crate::follow::Call::Draft { reqs: want.iter().map(|w| (w.1, w.2, w.3, w.5)).collect() })?;
            }
            // one batched pass for all of them: trees where every stream takes one, else chains
            let trees = if !want.is_empty() && want.iter().all(|w| w.5 >= 2) {
                df.draft_tree_many(&self.e, &want.iter().map(|w| (w.1, w.2, w.3, w.5)).collect::<Vec<_>>())?
            } else {
                None
            };
            match trees {
                Some(ts) => {
                    for (w, (d, par)) in want.iter().zip(ts) {
                        anyhow::ensure!(d.len() == par.len() && d.len() <= w.5, "drafter tree: {} nodes, {} parents",
                            d.len(), par.len());
                        let a = &mut self.active[w.0];
                        a.tree = Some(tree_rows(&par));
                        a.drafts = d;
                    }
                    // copy rows: grafted onto each tree past the rows it agrees with, within the step's verify rows.
                    // The verify's cost steps up past multiples of COPY_TIER rows: a copy crosses into the next tier
                    // only when its stream's copies keep landing whole or its match is long (COPY_SURE tokens).
                    if self.copy > 0 {
                        let mut total: usize = self.active.iter().filter(|a| a.pending.is_some()).map(|a| 1 + a.drafts.len())
                            .sum();
                        let mut left = self.verify_rows.saturating_sub(total);
                        for w in &want {
                            let a = &mut self.active[w.0];
                            let (Some(c), Some(par), Some(t)) = (a.copier.as_mut(), a.tree.as_mut(), a.pending) else { continue };
                            let rows = par.len();
                            let room = c.extra.min(TREE_ROWS.saturating_sub(rows)).min(left)
                                .min(a.req.sampling.max_tokens.saturating_sub(a.out.len()))
                                .min(a.cap.saturating_sub(self.e.slot_len[a.slot] + rows + 3));
                            if room == 0 {
                                continue;
                            }
                            c.sync(&a.req.prompt_ids, &a.out);
                            // the tree's depth at most, then the copy rows
                            let (copy, matched) = c.propose(rows - 1 + room);
                            let room = if c.extra > COPY_LO || matched >= COPY_SURE { room }
                                else { room.min(total.next_multiple_of(COPY_TIER) - total) };
                            if copy.is_empty() || room == 0 {
                                continue;
                            }
                            let mut depth = vec![0usize; rows];
                            for r in 1..rows {
                                depth[r] = depth[par[r] as usize] + 1;
                            }
                            let mut toks = Vec::with_capacity(rows + room);
                            toks.push(t);
                            toks.extend_from_slice(&a.drafts);
                            let mut p2 = par.clone();
                            let (agree, added) = crate::copy::graft(&mut toks, &mut p2, &copy, room);
                            if added > 0 && agree >= self.copy_agree.min(depth.iter().copied().max().unwrap_or(0)).max(1) {
                                *par = p2;
                                a.drafts = toks.split_off(1);
                                a.copied = (rows, added);
                                left -= added;
                                total += added;
                            }
                        }
                    }
                }
                None if !want.is_empty() => {
                    let ds = df.draft_many(&self.e, &want.iter().map(|w| (w.1, w.2, w.3)).collect::<Vec<_>>())?;
                    for (w, mut d) in want.iter().zip(ds) {
                        let a = &mut self.active[w.0];
                        d.truncate(w.4.min(a.depth.max(1)));
                        a.drafts = d;
                    }
                }
                None => {}
            }
        }
        // 1b. chain MTP drafts up to each decoding stream's depth (the verify rows of all streams share one GEMV call)
        if spec && !use_df && self.e.has_mtp() {
            let ndec = self.active.iter().filter(|a| a.pending.is_some()).count().max(1);
            let cap = (DECODE_ROWS / ndec).saturating_sub(1).min(self.spec);
            // copy rows: a lone stream whose copies keep landing whole verifies its copy alone as one chain, without
            // chaining MTP drafts (as with the block drafter)
            if self.copy > 0 && ndec_now == 1 {
                for a in self.active.iter_mut().filter(|a| a.pending.is_some() && !a.special()) {
                    let Some(c) = a.copier.as_mut() else { continue };
                    if c.extra <= COPY_LO {
                        continue;
                    }
                    let n = if c.extra >= self.copy { self.copy } else { c.extra.min(TREE_ROWS - 1) }
                        .min(self.verify_rows - 1).min(self.e.hist_rows - 1)
                        .min(a.req.sampling.max_tokens.saturating_sub(a.out.len()))
                        .min(a.cap.saturating_sub(self.e.slot_len[a.slot] + 3));
                    c.sync(&a.req.prompt_ids, &a.out);
                    let (copy, _) = c.propose(n);
                    if !copy.is_empty() {
                        a.copied = (1, copy.len());
                        a.drafts = copy;
                        a.tree = None;
                    }
                }
            }
            loop {
                let mut rows = Vec::new();
                let mut who = Vec::new();
                for (ai, a) in self.active.iter().enumerate() {
                    let want = a.depth.min(cap).min(a.cap.saturating_sub(self.e.slot_len[a.slot] + 2));
                    if a.pending.is_some() && a.copied.1 == 0 && !a.drafts.is_empty() && a.drafts.len() < want {
                        let k = a.drafts.len();
                        rows.push(MtpRow { slot: a.slot, pos: self.e.slot_len[a.slot] + k - 1, token: a.drafts[k - 1],
                            src: MtpSrc::Out(a.out_idx) });
                        who.push(ai);
                    }
                }
                if rows.is_empty() {
                    break;
                }
                let draw: Vec<usize> = (0..rows.len()).collect();
                let d = self.e.mtp_forward(&rows, &draw)?;
                for (j, &ai) in who.iter().enumerate() {
                    self.active[ai].drafts.push(d[j]);
                    self.active[ai].out_idx = j;
                }
            }
            for a in self.active.iter_mut().filter(|a| a.copied.1 == 0) {
                a.drafts.truncate(cap);
            }
            // copy rows grafted onto each stream's MTP chain: one verify tree holds both (the copy follows the chain
            // while they agree, then branches), within the step's verify rows. A copy joins when it agrees with the
            // chain's first draft or its match is long (COPY_SURE), and crosses a COPY_TIER row boundary only when
            // its stream's copies keep landing whole or the match is long.
            if self.copy > 0 {
                let mut total: usize = self.active.iter().filter(|a| a.pending.is_some()).map(|a| 1 + a.drafts.len()).sum();
                let mut left = self.verify_rows.saturating_sub(total);
                for a in self.active.iter_mut() {
                    if a.special() {
                        continue;
                    }
                    let (Some(c), Some(t)) = (a.copier.as_mut(), a.pending) else { continue };
                    if a.copied.1 > 0 {
                        continue;
                    }
                    let rows = 1 + a.drafts.len();
                    let room = c.extra.min(TREE_ROWS.saturating_sub(rows)).min(left).min(self.e.hist_rows.saturating_sub(rows))
                        .min(a.req.sampling.max_tokens.saturating_sub(a.out.len()))
                        .min(a.cap.saturating_sub(self.e.slot_len[a.slot] + rows + 3));
                    if room == 0 {
                        continue;
                    }
                    c.sync(&a.req.prompt_ids, &a.out);
                    let (copy, matched) = c.propose(rows - 1 + room);
                    let room = if c.extra > COPY_LO || matched >= COPY_SURE { room }
                        else { room.min(total.next_multiple_of(COPY_TIER) - total) };
                    if copy.is_empty() || room == 0 {
                        continue;
                    }
                    let mut toks = Vec::with_capacity(rows + room);
                    toks.push(t);
                    toks.extend_from_slice(&a.drafts);
                    let mut par: Vec<i32> = (0..rows as i32).map(|r| r - 1).collect();
                    let (agree, added) = crate::copy::graft(&mut toks, &mut par, &copy, room);
                    if added > 0 && (agree >= 1 || rows == 1 || matched >= COPY_SURE) {
                        a.tree = Some(par);
                        a.drafts = toks.split_off(1);
                        a.copied = (rows, added);
                        left -= added;
                        total += added;
                    }
                }
            }
        }
        lap(&mut self.prof, 0);
        // 2. the target forward: decode/verify rows first, then prefill chunks within the row budget
        let mut segs: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut owners: Vec<(usize, bool)> = Vec::new(); // (active index, is a decode segment)
        let mut rows = 0;
        for (ai, a) in self.active.iter().enumerate() {
            if let Some(t) = a.pending {
                let mut v = vec![t];
                v.extend_from_slice(&a.drafts);
                rows += v.len();
                segs.push((a.slot, v));
                owners.push((ai, true));
            }
        }
        let mut budget = self.batch_rows;
        // fairness: while streams decode, prompt chunks are capped and only get (1 - share) of the wall time
        let decoding = rows > 0;
        if !decoding {
            self.t_pre = 0.0;
            self.t_dec = 0.0;
        } else if self.decode_share > 0.0 {
            budget = budget.min(rows + self.prefill_cap);
            if self.t_pre * self.decode_share > (1.0 - self.decode_share) * self.t_dec {
                budget = rows; // a decode-only step
            }
        }
        let step_t0 = Instant::now();
        // a long prompt alone on a device that streams its prompt weights: the layers of its next chunks run as one
        // group (layer by layer, the weights cross the bus once); the chunks then go through the steps below as
        // always, cut the same way (at every snapshot point too), and only add the head
        if rows == 0 && self.active.len() == 1 && self.e.group_slot().is_none() && self.e.group_capable() {
            let a = &self.active[0];
            let len = a.req.prompt_ids.len();
            if a.pending.is_none() && a.fed < len {
                let lim = self.o.prefill_chunk.min(budget);
                let cap = self.e.group_rows();
                let (mut f, mut ends) = (a.fed, Vec::new());
                while f < len {
                    let n = prompt_cut_at(len, &a.cuts, f, lim);
                    if n <= crate::model::X3_ROWS || f + n - a.fed > cap {
                        break;
                    }
                    f += n;
                    ends.push(f);
                }
                // what fits the device beside its reserve (the parked rows, the state at snapshot points inside)
                let inside = |ends: &[usize]| ends[..ends.len() - 1].iter().filter(|e| self.cache && a.cuts.binary_search(e).is_ok()).count();
                // (with the GDN overlap's buffers where it is on; without them when not even two chunks fit beside)
                let all = ends.clone();
                for overlap in [true, false] {
                    ends = all.clone();
                    while ends.len() > 1 && !self.e.group_fits(ends[ends.len() - 1] - a.fed, ends.len(), inside(&ends), lim, overlap) {
                        ends.pop();
                    }
                    if ends.len() > 1 {
                        break;
                    }
                }
                if ends.len() > 1 {
                    let mut from = a.fed;
                    let chunks: Vec<&[u32]> = ends.iter().map(|&e| {
                        let c = &a.req.prompt_ids[from..e];
                        from = e;
                        c
                    }).collect();
                    let snaps: Vec<bool> = ends.iter().map(|e| self.cache && a.cuts.binary_search(e).is_ok()).collect();
                    if !self.e.group_prefill(a.slot, &chunks, &snaps, &a.req.cancel)? {
                        // cancelled part way: finish resets the slot (its state is between layers)
                        let a = self.active.swap_remove(0);
                        self.finish(a, Finish::Cancelled);
                        return Ok(());
                    }
                }
            }
        }
        let first_prompt = segs.len();
        for (ai, a) in self.active.iter().enumerate() {
            if a.pending.is_none() && a.fed < a.req.prompt_ids.len() && rows < budget {
                let lim = self.o.prefill_chunk.min(budget - rows);
                let mut n = prompt_cut_at(a.req.prompt_ids.len(), &a.cuts, a.fed, lim);
                if n == 0 && self.o.prefill_chunk < 4 * PROMPT_ALIGN {
                    n = (a.req.prompt_ids.len() - a.fed).min(lim); // tiny test chunks: unaligned, as before
                }
                if n == 0 {
                    continue; // < 64 rows left in this step's budget: the prompt waits for the next step
                }
                segs.push((a.slot, a.req.prompt_ids[a.fed..a.fed + n].to_vec()));
                owners.push((ai, false));
                rows += n;
            }
        }
        // short prompt chunks right after the decode rows: the engine runs them through the decode GEMV as when
        // they are alone (Prepared::lin_dec), so a prompt's numerics do not depend on what else is in the step
        let mut tail: Vec<_> = segs.drain(first_prompt..).zip(owners.drain(first_prompt..)).collect();
        tail.sort_by_key(|(sg, _)| sg.1.len() > crate::model::X3_ROWS);
        for (sg, o) in tail {
            segs.push(sg);
            owners.push(o);
        }
        let mut logit_rows = Vec::new();
        let mut seg_row0 = Vec::new();
        let mut row = 0;
        let mut lr_first = Vec::new(); // first logit index of each segment
        for (i, sg) in segs.iter().enumerate() {
            seg_row0.push(row);
            lr_first.push(logit_rows.len());
            let a = &self.active[owners[i].0];
            if owners[i].1 {
                logit_rows.extend(row..row + sg.1.len());
            } else if a.fed + sg.1.len() == a.req.prompt_ids.len() {
                logit_rows.push(row + sg.1.len() - 1);
            }
            row += sg.1.len();
        }
        let starts: Vec<usize> = segs.iter().map(|(sl, _)| self.e.slot_len[*sl]).collect();
        let seg_refs: Vec<(usize, &[u32])> = segs.iter().map(|(s, t)| (*s, t.as_slice())).collect();
        let hist = spec && segs.iter().zip(&owners).any(|(sg, o)| o.1 && sg.1.len() > 1);
        let dec_rows: usize = segs.iter().zip(&owners).filter(|(_, o)| o.1).map(|(sg, _)| sg.1.len()).sum();
        let trees: Vec<Option<&[i32]>> = owners.iter().map(|o| if o.1 { self.active[o.0].tree.as_deref() } else { None })
            .collect();
        self.e.forward_tree(&seg_refs, &trees, &logit_rows, hist, dec_rows.min(crate::model::DEC_MAX))?;
        // 3. pick tokens: greedy rows by the GPU argmax, sampled rows on the host
        let nl = logit_rows.len();
        let greedy_all = owners.iter().all(|o| self.active[o.0].req.sampling.greedy());
        let am = if nl > 0 { self.e.argmax(nl)? } else { Vec::new() };
        // host rows: only the logit rows of sampled sequences come to the host, packed in segment order (a step can
        // carry more logit rows than any fixed buffer: every verify row plus the last row of each ending prompt)
        let mut host_row = vec![usize::MAX; segs.len()];
        if !greedy_all && nl > 0 {
            let v = self.e.cfg.vocab;
            let sampled: Vec<bool> = owners.iter().map(|o| !self.active[o.0].req.sampling.greedy()).collect();
            let hr;
            (host_row, hr) = sampled_host_rows(&lr_first, nl, &sampled);
            if self.host.len() < hr * v {
                self.host.resize(hr * v, 0);
            }
            for si in 0..segs.len() {
                if host_row[si] != usize::MAX {
                    let k = lr_first.get(si + 1).copied().unwrap_or(nl) - lr_first[si];
                    let h0 = host_row[si] * v;
                    self.e.logits_host_rows(lr_first[si], k, &mut self.host[h0..h0 + k * v])?;
                }
            }
        }
        lap(&mut self.prof, 1);
        let v = self.e.cfg.vocab;
        let mut done: Vec<(usize, Finish)> = Vec::new();
        let mut absorb: Vec<(usize, Vec<MtpRow>, bool)> = Vec::new(); // (active, rows, draw last)
        for (si, sg) in segs.iter().enumerate() {
            let (ai, is_dec) = owners[si];
            let a = &mut self.active[ai];
            let start = starts[si];
            let n = sg.1.len();
            let r0 = seg_row0[si];
            if !is_dec {
                // prefill chunk: into the slot's bookkeeping; the last chunk samples the first token
                self.slot_tokens[a.slot].extend_from_slice(&sg.1);
                a.fed += n;
                if self.cache && self.cache_save_on && a.cuts.binary_search(&a.fed).is_ok() {
                    self.snap_due.push((a.slot, a.fed, a.req.snapshots.first() == Some(&a.fed) || a.share.contains(&a.fed)));
                }
                let complete = a.fed == a.req.prompt_ids.len();
                let mut first = None;
                let mut bad = false;
                if complete {
                    let li = lr_first[si];
                    let hi = host_row[si];
                    // the first token of a reply is never a stop token (an empty reply is never wanted): the best other
                    // one; a constrained opening picks among its candidates
                    let cands = a.opening_cands();
                    let t = if a.req.sampling.greedy() {
                        let t = am[li];
                        if cands.is_some() || a.req.sampling.stop_token_ids.contains(&t) {
                            self.scratch.resize(v, 0);
                            self.e.logits_host_rows(li, 1, &mut self.scratch)?;
                            bad = self.finite && !row_finite(&self.scratch);
                            mask_row(&mut self.scratch, cands.as_deref(), &a.req.sampling.stop_token_ids);
                            argmax_row(&self.scratch)
                        } else {
                            if self.finite {
                                bad = self.e.logit_bits(li, t)? & 0x7f80 == 0x7f80;
                            }
                            t
                        }
                    } else {
                        let row = &mut self.host[hi * v..(hi + 1) * v];
                        bad = self.finite && !row_finite(row);
                        mask_row(row, cands.as_deref(), &a.req.sampling.stop_token_ids);
                        let t = sample(row, &a.req.sampling, &a.req.prompt_ids, &a.out, &mut a.rng);
                        // (a draw can only land on a masked token through rounding at the tail)
                        if row[t as usize] == BF16_NEG_INF { argmax_row(row) } else { t }
                    };
                    self.picks += 1;
                    bad |= self.test_nonfinite == Some(self.picks);
                    if bad {
                        eprintln!("ferrite: {}: non-finite logits for the first output token: ending the request", a.req.id);
                        done.push((ai, Finish::Error("the model produced non-finite logits".into())));
                    } else {
                        a.opening_step(t);
                        first = Some(t);
                    }
                }
                if spec {
                    let p = &a.req.prompt_ids;
                    let rows: Vec<MtpRow> = (0..n).map(|i| {
                        let pos = start + i;
                        let token = if pos + 1 < p.len() { p[pos + 1] } else { first.unwrap_or(0) };
                        MtpRow { slot: a.slot, pos, token, src: MtpSrc::Tap(r0 + i) }
                    }).collect();
                    absorb.push((ai, rows, complete && !bad));
                }
                if let Some(t) = first {
                    let act = Self::guard_scan(a, &[t]).map_or(GuardAct::None, |x| x.1);
                    match Self::emit(a, &[t], self.e.slot_len[a.slot], a.cap) {
                        Some(f) => done.push((ai, f)),
                        None => {
                            a.pending = Some(t);
                            if apply_act(a, act, copy_hi) {
                                done.push((ai, Finish::Length));
                            }
                        }
                    }
                }
                continue;
            }
            // decode / verify segment: rows [pending, d1..dn-1]
            let li = lr_first[si];
            let drafts = a.drafts.clone();
            let mut emitted: Vec<u32> = Vec::new();
            let mut acc = 0;
            let mut path = None;
            let mut bad = false;
            let forced = a.force.take();
            let cands = if forced.is_none() { a.opening_cands() } else { None };
            let free = forced.is_none() && cands.is_none();
            if let Some(f) = forced {
                // the engine's own token (the end of the reasoning block): this row's logits are not used
                emitted.push(f);
            } else if let Some(c) = cands {
                // constrained opening: the best (sampled request: a drawn) candidate
                let t = if a.req.sampling.greedy() {
                    self.scratch.resize(v, 0);
                    self.e.logits_host_rows(li, 1, &mut self.scratch)?;
                    bad = self.finite && !row_finite(&self.scratch);
                    mask_row(&mut self.scratch, Some(&c), &[]);
                    argmax_row(&self.scratch)
                } else {
                    let hi = host_row[si];
                    let row = &mut self.host[hi * v..(hi + 1) * v];
                    bad = self.finite && !row_finite(row);
                    mask_row(row, Some(&c), &[]);
                    let t = sample(row, &a.req.sampling, &a.req.prompt_ids, &a.out, &mut a.rng);
                    if row[t as usize] == BF16_NEG_INF { argmax_row(row) } else { t }
                };
                a.opening_step(t);
                emitted.push(t);
            } else if let Some(par) = a.tree.take() {
                // tree verify (greedy only): walk from the root, then the kept rows become consecutive rows
                let (p, bonus) = tree_walk(&sg.1, &par, &am[li..li + n]);
                acc = p.len() - 1;
                emitted.extend(p[1..].iter().map(|&r| sg.1[r]));
                emitted.push(bonus);
                path = Some(p);
            } else if a.req.sampling.greedy() {
                while acc < drafts.len() && drafts[acc] == am[li + acc] {
                    acc += 1;
                }
                emitted.extend_from_slice(&drafts[..acc]);
                emitted.push(am[li + acc]);
            } else {
                // speculative sampling with deterministic drafts: accept d with prob p(d), else sample p without d
                let hi = host_row[si];
                bad = self.finite && (0..n).any(|j| !row_finite(&self.host[(hi + j) * v..(hi + j + 1) * v]));
                let mut hist_out = a.out.clone();
                let mut bonus = None;
                for (j, &d) in drafts.iter().enumerate() {
                    let row = &self.host[(hi + j) * v..(hi + j + 1) * v];
                    let (idx, pr) = dist(row, &a.req.sampling, &a.req.prompt_ids, &hist_out);
                    let pd = idx.iter().position(|&t| t == d).map_or(0.0, |k| pr[k]);
                    let u = (splitmix(&mut a.rng) >> 40) as f32 / (1u64 << 24) as f32;
                    if u < pd {
                        acc += 1;
                        hist_out.push(d);
                    } else {
                        bonus = Some(draw_excluding(&idx, &pr, d, &mut a.rng));
                        break;
                    }
                }
                emitted.extend_from_slice(&drafts[..acc]);
                let b = match bonus {
                    Some(b) => b,
                    None => {
                        let row = &self.host[(hi + acc) * v..(hi + acc + 1) * v];
                        sample(row, &a.req.sampling, &a.req.prompt_ids, &hist_out, &mut a.rng)
                    }
                };
                emitted.push(b);
            }
            // finite check of a greedy pick (its rows never come to the host): the logit at the argmax of the row
            // whose token no draft vouches for
            if self.finite && free && a.req.sampling.greedy() {
                let brow = path.as_ref().map_or(acc, |p: &Vec<usize>| *p.last().unwrap());
                bad = self.e.logit_bits(li + brow, *emitted.last().unwrap())? & 0x7f80 == 0x7f80;
            }
            self.picks += 1;
            bad |= self.test_nonfinite == Some(self.picks);
            if bad {
                // never sample garbage: keep the fed row only (a consistent slot) and end the request below
                eprintln!("ferrite: {}: non-finite logits at output token {}: ending the request", a.req.id, a.out.len());
                acc = 0;
                emitted.truncate(1);
                if let Some(p) = path.as_mut() {
                    p.truncate(1);
                }
            }
            // the guard sees the tokens the client will get, one at a time; where it acts the rest is given back
            let mut act = GuardAct::None;
            if !bad {
                if let Some((keep, g)) = Self::guard_scan(a, &emitted) {
                    if keep < emitted.len() {
                        emitted.truncate(keep);
                        acc = keep - 1;
                        if let Some(p) = path.as_mut() {
                            p.truncate(keep);
                        }
                    }
                    act = g;
                }
            }
            lap(&mut self.prof, 4);
            match &path {
                Some(p) => {
                    self.e.commit_tree(a.slot, start, n, p, self.df.is_some().then_some(r0))?;
                    self.slot_tokens[a.slot].extend(p.iter().map(|&r| sg.1[r]));
                }
                None => {
                    if n > 1 {
                        self.e.commit(a.slot, start, n, acc + 1)?;
                    }
                    self.slot_tokens[a.slot].extend_from_slice(&sg.1[..acc + 1]);
                }
            }
            lap(&mut self.prof, 2);
            let copied = std::mem::take(&mut a.copied);
            if copied.1 > 0 {
                // copy rows kept: the path's rows from the first copy row on (a chain's are its accepted drafts)
                let landed = match &path { Some(p) => p.iter().filter(|&&r| r >= copied.0).count(), None => acc };
                if let Some(c) = a.copier.as_mut() {
                    c.observe(copied.1, landed);
                }
            }
            if self.prof_steps {
                // accepted drafted nodes (row - 1, best-first order; copy rows after the tree's)
                trace.push((n, acc, match &path { Some(p) => p[1..].iter().map(|r| r - 1).collect(), None => (0..acc).collect() },
                    a.slot, copied));
            }
            if !drafts.is_empty() {
                a.rounds += 1;
                a.accepted += acc;
                a.acc_ema = 0.7 * a.acc_ema + 0.3 * acc as f32;
                a.depth = ((a.acc_ema.ceil() as usize) + 1).clamp(1, self.spec);
            }
            a.drafts.clear();
            a.pending = None;
            if spec {
                // the kept rows' taps: consecutive rows of a chain, the path's rows of a tree
                let rows: Vec<MtpRow> = (0..=acc).map(|i| MtpRow { slot: a.slot, pos: start + i, token: emitted[i],
                    src: MtpSrc::Tap(r0 + path.as_ref().map_or(i, |p| p[i])) }).collect();
                absorb.push((ai, rows, true));
            }
            let slot_len = self.e.slot_len[a.slot];
            if bad {
                done.push((ai, Finish::Error("the model produced non-finite logits".into())));
                continue;
            }
            match Self::emit(a, &emitted, slot_len, a.cap) {
                Some(f) => done.push((ai, f)),
                None => {
                    a.pending = emitted.last().copied();
                    if apply_act(a, act, copy_hi) {
                        done.push((ai, Finish::Length));
                    }
                }
            }
        }
        // 4a. external drafter: absorb the kept rows (their feature taps) per stream
        if spec && self.df.is_some() && !absorb.is_empty() {
            let feats = self.e.feature_taps().ok_or_else(|| anyhow::anyhow!("drafter without feature taps"))?;
            let df = self.df.as_mut().unwrap();
            let fw = df.layers().len() * self.e.cfg.hidden;
            let rows_of: Vec<(usize, usize, usize, usize)> = absorb.iter().filter(|(_, rows, _)| !rows.is_empty())
                .map(|(ai, rows, _)| {
                    let r0 = match rows[0].src { MtpSrc::Tap(i) | MtpSrc::Out(i) => i };
                    (self.active[*ai].slot, r0, rows.len(), rows[0].pos)
                }).collect();
            if df.tp_split() {
                self.e.mirror_call(crate::follow::Call::Absorb { items: rows_of.clone() })?;
            }
            let items: Vec<(usize, fe_cuda::DevPtr, usize, usize)> =
                rows_of.iter().map(|&(slot, r0, n, pos)| (slot, feats.add::<u16>(r0 * fw), n, pos)).collect();
            df.absorb_many(&self.e, &items)?;
            if self.prof.is_some() {
                self.e.stream.sync()?;
            }
            lap(&mut self.prof, 3);
        }
        // 4a'. prompt cache: snapshot the slots whose prefill chunk ended on a snapshot point (state after the chunk's
        // forward and its drafter absorb)
        for (slot, pos, first) in std::mem::take(&mut self.snap_due) {
            self.cache_save(slot, pos)?;
            if first {
                self.prefix_entry(slot, pos)?;
            }
        }
        // 4b. absorb the kept rows into the MTP head and draw each continuing stream's first draft (MTP steps only:
        // a block-drafter step leaves those positions out of the MTP cache, which only costs its acceptance)
        if spec && !use_df && self.e.has_mtp() && !absorb.is_empty() {
            let finished: Vec<usize> = done.iter().map(|d| d.0).collect();
            let mut rows = Vec::new();
            let mut draws = Vec::new();
            let mut who = Vec::new();
            for (ai, r, draw) in absorb {
                if r.is_empty() {
                    continue;
                }
                rows.extend_from_slice(&r);
                if draw && !finished.contains(&ai) {
                    draws.push(rows.len() - 1);
                    who.push(ai);
                }
            }
            if rows.len() <= self.batch_rows {
                let d = self.e.mtp_forward(&rows, &draws)?;
                for (j, &ai) in who.iter().enumerate() {
                    let a = &mut self.active[ai];
                    if a.pending.is_some() {
                        a.drafts = vec![d[j]];
                        a.out_idx = draws[j];
                    }
                }
            }
        }
        if decoding && self.decode_share > 0.0 {
            let dt = step_t0.elapsed().as_secs_f64();
            let had_prompt = owners.iter().any(|o| !o.1);
            // decay so the share tracks the last few seconds
            self.t_pre *= 0.97;
            self.t_dec *= 0.97;
            if had_prompt {
                self.t_pre += dt;
            } else {
                self.t_dec += dt;
            }
        }
        lap(&mut self.prof, 4);
        if let Some((_, n)) = self.prof.as_mut() {
            *n += 1;
        }
        if let (true, Some(t0), Some((t1, n))) = (self.prof_steps, t_before, self.prof) {
            let prompt: usize = segs.iter().zip(&owners).filter(|(_, o)| !o.1).map(|(sg, _)| sg.1.len()).sum();
            let d: Vec<f64> = (0..5).map(|i| 1e3 * (t1[i] - t0[i])).collect();
            eprintln!("step {n}: dec {} rows {:?} acc {:?} prompt {prompt} | draft {:.1} fwd {:.1} commit {:.1} absorb {:.1} \
                rest {:.1} ms | slots {:?} paths {:?} copies {:?}", trace.len(), trace.iter().map(|t| t.0).collect::<Vec<_>>(),
                trace.iter().map(|t| t.1).collect::<Vec<_>>(), d[0], d[1], d[2], d[3], d[4],
                trace.iter().map(|t| t.3).collect::<Vec<_>>(), trace.iter().map(|t| &t.2).collect::<Vec<_>>(),
                trace.iter().map(|t| t.4).collect::<Vec<_>>());
        }
        done.sort_by(|x, y| y.0.cmp(&x.0));
        for (ai, f) in done {
            let a = self.active.swap_remove(ai);
            self.finish(a, f);
        }
        Ok(())
    }

    /// Feed the request's guard the tokens of `toks` the client will get (as emit: up to a stop token or the token
    /// limit), one at a time; Some((tokens to keep, what it asks for)) when it acts.
    fn guard_scan(a: &mut Active, toks: &[u32]) -> Option<(usize, GuardAct)> {
        let g = a.req.guard.as_mut()?;
        let sp = &a.req.sampling;
        for (i, &t) in toks.iter().enumerate() {
            let n = a.out.len() + i;
            if sp.stop_token_ids.contains(&t) && n >= sp.min_tokens {
                return None;
            }
            let act = g.feed(t);
            if act != GuardAct::None {
                return Some((i + 1, act));
            }
            if n + 1 >= sp.max_tokens {
                return None;
            }
        }
        None
    }

    /// Deliver sampled tokens in order; returns the finish reason if the request ends on one of them.
    fn emit(a: &mut Active, toks: &[u32], slot_len: usize, max_ctx: usize) -> Option<Finish> {
        if a.ttft.is_none() && !toks.is_empty() {
            a.ttft = Some(a.t0.elapsed().as_secs_f64());
        }
        for &t in toks {
            let sp = &a.req.sampling;
            if sp.stop_token_ids.contains(&t) && a.out.len() >= sp.min_tokens {
                return Some(Finish::Stop(t));
            }
            a.out.push(t);
            if !(a.sink)(Event::Tokens(vec![t])) {
                return Some(Finish::Cancelled);
            }
            if a.out.len() >= a.req.sampling.max_tokens {
                return Some(Finish::Length);
            }
        }
        if slot_len + 2 + a.depth >= max_ctx {
            return Some(Finish::Length);
        }
        None
    }

    /// Prompt cache: save `slot`'s state at `pos` (it holds rows [0, pos)) into a free state block (engine GDN part,
    /// TP2-mirrored, then the drafter's window); keeps the slot's first snapshot and its latest KEEP_LAST.
    fn cache_save(&mut self, slot: usize, pos: usize) -> Result<()> {
        if self.slot_snaps[slot].last().is_some_and(|x| x.pos >= pos) {
            return Ok(());
        }
        let Some(blk) = self.state_block() else { return Ok(()) }; // pool full: no snapshot
        let off = self.e.cache_save(slot, pos, blk)?;
        if let Some(df) = self.df.as_ref() {
            df.save_state(&self.e, slot, pos, self.e.park_ptr(off)?)?;
        }
        if self.cache_trace {
            eprintln!("cache: save slot {slot} at {pos} -> blk {blk}");
        }
        let freed = crate::pcache::push(&mut self.slot_snaps[slot], crate::pcache::Snap { pos, blk });
        self.blocks.release(freed);
        Ok(())
    }

    /// FE_CACHE: one pool a rank (docs/prompt-cache.md, park pool): state blocks for the slots' own snapshots (first + KEEP_LAST
    /// each) + 32 for parked conversations, the rest of FE_PARK_GB (default 24) as KV blocks of KV_BLOCK rows.
    fn cache_init(&mut self) -> Result<()> {
        if !self.cache {
            return Ok(());
        }
        let extra = self.df.as_ref().map_or(0, |d| d.state_bytes());
        anyhow::ensure!(extra == 0 || !crate::lowmem::flag("FE_PARK_HOST"),
            "prompt cache: a drafter with saved state needs the park pool on the device (FE_PARK_HOST=0)");
        // the pool: FE_PARK_GB (default 24) but never closer than FE_PARK_RESERVE_GB (default 20) to running out of
        // available memory (MemAvailable, what memguard and the OOM killer see; TP1 128K slots leave ~31 GB)
        let gb: f64 = std::env::var("FE_PARK_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(24.0);
        let reserve: f64 = std::env::var("FE_PARK_RESERVE_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(20.0);
        let avail = mem_available().unwrap_or(usize::MAX);
        let budget = ((gb * 1e9) as usize).min(avail.saturating_sub((reserve * 1e9) as usize));
        let state_b = (self.e.park_gdn_bytes() + extra).next_multiple_of(4096);
        let kv_b = crate::pcache::KV_BLOCK * self.e.park_kv_bytes();
        // the slots' own snapshots always; up to 32 more for parked conversations within half the budget
        let base = self.o.n_slots * (1 + crate::pcache::KEEP_LAST);
        let n_state = base + (budget / 2 / state_b).saturating_sub(base).min(32);
        let n_kv = budget.saturating_sub(n_state * state_b) / kv_b;
        let bytes = self.e.cache_pool(n_state, extra, n_kv)?;
        self.blocks = crate::pcache::Blocks::new(n_state);
        self.park = crate::pcache::Park::new(n_kv);
        eprintln!("ferrite: prompt cache: {n_state} state blocks ({:.1} MB) + {n_kv} KV blocks of {} rows ({:.1} MB) = \
            {:.1} GB a rank (FE_PARK_GB {gb}, {:.1} GB available before it)", state_b as f64 / 1e6, crate::pcache::KV_BLOCK,
            kv_b as f64 / 1e6, bytes as f64 / 1e9, avail as f64 / 1e9);
        Ok(())
    }

    /// Rows a resume point must leave before the prompt end: a point the request itself would not cut at (within
    /// DECODE_ROWS of the end, FE_CACHE_TAILCUT off) is no resume point either (cached == uncached cuts).
    fn min_tail(&self) -> usize {
        if self.tailcut { 0 } else { DECODE_ROWS }
    }

    /// A free state block, dropping least recently used park entries if needed.
    fn state_block(&mut self) -> Option<usize> {
        loop {
            if let Some(b) = self.blocks.alloc() {
                return Some(b);
            }
            let freed = self.park.evict_lru()?;
            self.blocks.release(freed);
        }
    }

    /// `n` free KV park blocks, dropping least recently used park entries if needed.
    fn kv_blocks(&mut self, n: usize) -> Option<Vec<usize>> {
        while self.park.kv.available() < n {
            let freed = self.park.evict_lru()?;
            self.blocks.release(freed);
        }
        Some((0..n).map(|_| self.park.kv.alloc().unwrap()).collect())
    }

    /// Before `slot` is reset for another conversation: park its conversation (KV rows up to its latest snapshot and
    /// its snapshots' state blocks) so a later request can restore it; without room it is dropped.
    fn park_slot(&mut self, slot: usize) -> Result<()> {
        let Some(len) = self.slot_snaps[slot].last().map(|x| x.pos) else { return Ok(()) };
        let snaps: Vec<crate::pcache::Snap> = self.slot_snaps[slot].drain(..).collect();
        let tokens = self.slot_tokens[slot][..len.min(self.slot_tokens[slot].len())].to_vec();
        let Some(kv) = (tokens.len() == len).then(|| self.kv_blocks(len.div_ceil(crate::pcache::KV_BLOCK))).flatten() else {
            self.blocks.release(snaps.into_iter().map(|x| x.blk));
            return Ok(());
        };
        self.e.kv_out(slot, len, &kv)?;
        if self.cache_trace {
            eprintln!("cache: park slot {slot}: {len} tokens, kv {kv:?}, snaps {snaps:?}");
        }
        self.tick += 1;
        self.park.entries.push(crate::pcache::Entry { tokens, kv, snaps, used: self.tick, prefix: false });
        Ok(())
    }

    /// Restore park entry `ei` at its snapshot `x` into free slot `slot` (its own conversation parked first).
    /// Ok(false): parking the slot's conversation needed the entry's blocks (the pool held nothing else to evict),
    /// so there is nothing to restore: the caller treats it as a cache miss.
    fn park_restore(&mut self, slot: usize, ei: usize, x: crate::pcache::Snap) -> Result<bool> {
        // the entry about to be restored is the most recently used: parking evicts every other entry before it
        self.tick += 1;
        self.park.entries[ei].used = self.tick;
        self.park_slot(slot)?;
        // parking may evict entries (swap_remove moves the others): find the entry again by its snapshot block (state
        // blocks are not handed out while parking, so a freed block cannot reappear under another entry)
        let Some(ei) = (0..self.park.entries.len()).find(|&i| self.park.entries[i].snaps.contains(&x)) else {
            if self.cache_trace {
                eprintln!("cache: park entry with snap blk {} evicted while parking slot {slot}: computed in full", x.blk);
            }
            return Ok(false);
        };
        self.e.reset_slot(slot)?;
        // the entry's blocks back with the rows they were written with (a block's layout depends on its row count:
        // reading fewer rows than kv_out wrote misplaces every layer past the first); rows past x.pos are rewritten
        let (kv, len) = (self.park.entries[ei].kv.clone(), self.park.entries[ei].tokens.len());
        self.e.kv_in(slot, len, &kv)?;
        let off = self.e.cache_restore(slot, x.pos, x.blk)?;
        if let Some(df) = self.df.as_mut() {
            df.restore_state(&self.e, slot, x.pos, self.e.park_ptr(off)?)?;
        }
        self.tick += 1;
        let en = &mut self.park.entries[ei];
        en.used = self.tick;
        self.slot_tokens[slot] = en.tokens[..x.pos].to_vec();
        Ok(true)
    }

    /// The first snapshot point of a request (the end of the shared system prompt) or one of its share points: keep
    /// it as a shared-prefix entry (KV rows [0, pos) + its state) unless one with these tokens exists; at most
    /// `prefix_max` (LRU).
    fn prefix_entry(&mut self, slot: usize, pos: usize) -> Result<()> {
        if pos < PREFIX_MIN || self.prefix_max == 0 || self.park.has_prefix(&self.slot_tokens[slot][..pos]) {
            return Ok(());
        }
        if self.park.prefixes() >= self.prefix_max {
            if let Some(i) = (0..self.park.entries.len()).filter(|&i| self.park.entries[i].prefix)
                .min_by_key(|&i| self.park.entries[i].used) {
                let e = self.park.entries.swap_remove(i);
                self.park.kv.release(e.kv);
                self.blocks.release(e.snaps.into_iter().map(|x| x.blk));
            }
        }
        let Some(blk) = self.state_block() else { return Ok(()) };
        let Some(kv) = self.kv_blocks(pos.div_ceil(crate::pcache::KV_BLOCK)) else {
            self.blocks.release([blk]);
            return Ok(());
        };
        let off = self.e.cache_save(slot, pos, blk)?;
        if let Some(df) = self.df.as_ref() {
            df.save_state(&self.e, slot, pos, self.e.park_ptr(off)?)?;
        }
        self.e.kv_out(slot, pos, &kv)?;
        if self.cache_trace {
            eprintln!("cache: prefix entry from slot {slot}: {pos} tokens, kv {kv:?}, blk {blk}");
        }
        self.tick += 1;
        self.park.entries.push(crate::pcache::Entry { tokens: self.slot_tokens[slot][..pos].to_vec(), kv,
            snaps: vec![crate::pcache::Snap { pos, blk }], used: self.tick, prefix: true });
        Ok(())
    }

    /// Prompt cache: resume `slot` at snapshot `x` (its state block back into the slot, slot_len = x.pos); the
    /// slot's later snapshots and history past x.pos are dropped (about to be rewritten).
    fn cache_restore(&mut self, slot: usize, x: crate::pcache::Snap) -> Result<()> {
        let off = self.e.cache_restore(slot, x.pos, x.blk)?;
        if let Some(df) = self.df.as_mut() {
            df.restore_state(&self.e, slot, x.pos, self.e.park_ptr(off)?)?;
        }
        self.slot_tokens[slot].truncate(x.pos);
        let freed = crate::pcache::truncate(&mut self.slot_snaps[slot], x.pos);
        self.blocks.release(freed);
        Ok(())
    }

    fn finish(&mut self, mut a: Active, reason: Finish) {
        let dt = a.t0.elapsed().as_secs_f64();
        let spec = if a.rounds > 0 {
            format!(", {:.2} drafts accepted/round over {} rounds", a.accepted as f64 / a.rounds as f64, a.rounds)
        } else {
            String::new()
        };
        eprintln!("ferrite: {} done: {:?}, prompt {} (cached {}), out {} in {:.2}s (first token {:.3}s){spec}", a.req.id,
            reason, a.req.prompt_ids.len(), a.cached, a.out.len(), dt, a.ttft.unwrap_or(0.0));
        if let Some((t, n)) = self.prof.as_mut() {
            let ms = |x: f64| 1e3 * x / (*n).max(1) as f64;
            eprintln!("ferrite: per step over {n} steps: draft {:.2} ms, forward + argmax {:.2}, commit {:.2}, absorb {:.2}, \
                rest {:.2}", ms(t[0]), ms(t[1]), ms(t[2]), ms(t[3]), ms(t[4]));
            *t = [0.0; 5];
            *n = 0;
        }
        // (an engine error: the slot's state is not trusted as a prompt cache)
        let bad = matches!(reason, Finish::Error(_));
        (a.sink)(Event::Finished { reason, cached_prompt_tokens: a.cached });
        self.slot_busy[a.slot] = false;
        // the slot keeps prompt + generated (minus an unfed last token) as its prompt cache
        // (a prompt group left pending by a cancelled request: the slot's state is past its length)
        let group = self.e.group_slot() == Some(a.slot);
        if group {
            self.e.group_abort();
        }
        if group || bad || self.slot_tokens[a.slot].len() != self.e.slot_len[a.slot] {
            self.slot_tokens[a.slot].clear();
            let b: Vec<usize> = self.slot_snaps[a.slot].drain(..).map(|x| x.blk).collect();
            self.blocks.release(b);
            let _ = self.e.reset_slot(a.slot);
        }
    }
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The filtered target distribution of one bf16 logits row: candidate ids (descending) and probabilities.
pub fn dist(row: &[u16], sp: &Sampling, prompt: &[u32], out: &[u32]) -> (Vec<u32>, Vec<f32>) {
    let mut x: Vec<f32> = row.iter().map(|&h| f32::from_bits((h as u32) << 16)).collect();
    penalize(&mut x, sp, prompt, out);
    let t = if sp.temperature > 0.0 { sp.temperature } else { 1e-4 };
    let mut idx: Vec<u32> = (0..x.len() as u32).collect();
    let k = if sp.top_k > 0 { sp.top_k.min(idx.len()) } else { 256.min(idx.len()) };
    idx.select_nth_unstable_by(k - 1, |&a, &b| x[b as usize].total_cmp(&x[a as usize]));
    idx.truncate(k);
    idx.sort_unstable_by(|&a, &b| x[b as usize].total_cmp(&x[a as usize]));
    let m = x[idx[0] as usize];
    let mut p: Vec<f32> = idx.iter().map(|&i| ((x[i as usize] - m) / t).exp()).collect();
    let z: f32 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= z);
    let mut keep = p.len();
    if sp.top_p < 1.0 {
        let mut c = 0.0;
        for (i, &v) in p.iter().enumerate() {
            c += v;
            if c >= sp.top_p {
                keep = i + 1;
                break;
            }
        }
    }
    if sp.min_p > 0.0 {
        let thr = sp.min_p * p[0];
        keep = keep.min(p.iter().take_while(|&&v| v >= thr).count().max(1));
    }
    idx.truncate(keep);
    p.truncate(keep);
    let z: f32 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= z);
    (idx, p)
}

/// Sample from (idx, p) with token `d` removed (the speculative-sampling residual for a point-mass draft).
fn draw_excluding(idx: &[u32], p: &[f32], d: u32, rng: &mut u64) -> u32 {
    let z: f32 = idx.iter().zip(p).filter(|(t, _)| **t != d).map(|(_, v)| *v).sum();
    if z <= 0.0 {
        return idx[0];
    }
    let u = (splitmix(rng) >> 40) as f32 / (1u64 << 24) as f32 * z;
    let mut c = 0.0;
    for (t, v) in idx.iter().zip(p) {
        if *t == d {
            continue;
        }
        c += v;
        if u < c {
            return *t;
        }
    }
    *idx.iter().rfind(|&&t| t != d).unwrap_or(&idx[0])
}

fn penalize(x: &mut [f32], sp: &Sampling, prompt: &[u32], out: &[u32]) {
    if sp.repetition_penalty != 1.0 || sp.presence_penalty != 0.0 || sp.frequency_penalty != 0.0 {
        let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for &t in out {
            *counts.entry(t).or_default() += 1;
        }
        if sp.repetition_penalty != 1.0 {
            let mut seen: std::collections::HashSet<u32> = prompt.iter().copied().collect();
            seen.extend(out.iter().copied());
            for t in seen {
                let v = &mut x[t as usize];
                *v = if *v > 0.0 { *v / sp.repetition_penalty } else { *v * sp.repetition_penalty };
            }
        }
        for (&t, &c) in &counts {
            x[t as usize] -= sp.frequency_penalty * c as f32 + sp.presence_penalty;
        }
    }
}

/// Host sampling from one bf16 logits row.
pub fn sample(row: &[u16], sp: &Sampling, prompt: &[u32], out: &[u32], rng: &mut u64) -> u32 {
    if sp.temperature <= 0.0 {
        let mut x: Vec<f32> = row.iter().map(|&h| f32::from_bits((h as u32) << 16)).collect();
        penalize(&mut x, sp, prompt, out);
        return x.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;
    }
    let (idx, p) = dist(row, sp, prompt, out);
    let u = (splitmix(rng) >> 40) as f32 / (1u64 << 24) as f32;
    let mut c = 0.0;
    for (i, &v) in p.iter().enumerate() {
        c += v;
        if u < c {
            return idx[i];
        }
    }
    idx[idx.len() - 1]
}

/// prompt_cut with resume points (snapshot positions the prompt cache may restore at, the request's prefill start
/// among them, ascending): a chunk never crosses a point, and the 32-row grid restarts at each one, so a prompt
/// resumed at any point is cut exactly as when it is computed uncached (same segments from that point on).
pub fn prompt_cut_at(len: usize, points: &[usize], fed: usize, avail: usize) -> usize {
    let base = points.iter().rev().find(|&&p| p <= fed).copied().unwrap_or(0);
    let stop = points.iter().find(|&&p| p > fed && p < len).copied().unwrap_or(len);
    prompt_cut(stop, base, fed, avail)
}

#[cfg(test)]
mod share_tests {
    use super::{prompt_cut_at, share_point, DECODE_ROWS, PROMPT_ALIGN};

    /// two prompts with `common` equal tokens, then different tails
    fn pair(common: usize, tail_a: usize, tail_b: usize) -> (Vec<u32>, Vec<u32>) {
        let a: Vec<u32> = (0..common as u32).chain((0..tail_a as u32).map(|i| 1_000_000 + i)).collect();
        let b: Vec<u32> = (0..common as u32).chain((0..tail_b as u32).map(|i| 2_000_000 + i)).collect();
        (a, b)
    }

    #[test]
    fn the_common_prefix_on_the_grid() {
        let (a, b) = pair(5000, 300, 420);
        // no snapshot points: the grid starts at 0
        assert_eq!(share_point(&b, &[], &a, &[], 0, &[]), Some(4992));
        // a snapshot point at 301 (the system prompt's end) in both: the grid starts there
        assert_eq!(share_point(&b, &[301], &a, &[301], 0, &[]), Some(301 + (5000 - 301) / 32 * 32));
        // the leader is part way in: still ahead of it
        assert_eq!(share_point(&b, &[301], &a, &[301], 2349, &[]), Some(4973));
        // the leader is at or past the point: its state there is gone
        assert_eq!(share_point(&b, &[301], &a, &[301], 4973, &[]), None);
        assert_eq!(share_point(&b, &[301], &a, &[301], 5300, &[]), None);
        // a snapshot point exactly at the end of the common text is the point itself
        assert_eq!(share_point(&b, &[5000], &a, &[5000], 0, &[]), Some(5000));
    }

    #[test]
    fn no_point_when_sharing_would_change_a_prompt() {
        // under 1024 common tokens: computed by each
        let (a, b) = pair(1000, 300, 300);
        assert_eq!(share_point(&b, &[], &a, &[], 0, &[]), None);
        // the prompts' cut points before the common end differ (one prompt's point is too close to its end to cut at)
        let (a, b) = pair(5000, 300, 300);
        assert_eq!(share_point(&b, &[], &a, &[301], 0, &[]), None);
        // a waiting prompt that ends within DECODE_ROWS rows of the common text keeps a tail for the prompt kernels
        let (a, b) = pair(4992, 300, 5);
        let pos = share_point(&b, &[], &a, &[], 0, &[]).unwrap();
        assert!(b.len() - pos > DECODE_ROWS && pos % PROMPT_ALIGN == 0, "{pos}");
        // identical prompts: the same rule against the prompt's own end
        let pos = share_point(&a, &[], &a, &[], 0, &[]).unwrap();
        assert!(a.len() - pos > DECODE_ROWS && pos % PROMPT_ALIGN == 0, "{pos}");
    }

    #[test]
    fn lanes_share_one_point() {
        // lane 3's tail starts like lane 1's for 40 more tokens than lane 2's: it takes the point lane 2 asked for
        let (a, b) = pair(5000, 300, 300);
        let first = share_point(&b, &[], &a, &[], 0, &[]).unwrap();
        let c: Vec<u32> = a[..5040].iter().copied().chain(3_000_000..3_000_200).collect();
        assert_eq!(share_point(&c, &[], &a, &[], 0, &[first]), Some(first));
        // far beyond it: a point of its own, until the leader holds four
        let d: Vec<u32> = a[..5290].iter().copied().chain(3_000_000..3_000_200).collect();
        assert_eq!(share_point(&d, &[], &a, &[], 0, &[first]), Some(5280));
        assert_eq!(share_point(&d, &[], &a, &[], 0, &[1024, 2048, 3072, first]), None);
        assert_eq!(share_point(&c, &[], &a, &[], 0, &[1024, 2048, 3072, first]), Some(first));
    }

    /// The chunk ends of a prompt of `len` tokens cut at `points`, with `avail` rows a step, from `from` on.
    fn chunks(len: usize, points: &[usize], from: usize, avail: usize) -> Vec<usize> {
        let (mut fed, mut ends) = (from, Vec::new());
        while fed < len {
            let n = prompt_cut_at(len, points, fed, avail);
            assert!(n > 0);
            fed += n;
            ends.push(fed);
        }
        ends
    }

    #[test]
    fn a_share_point_keeps_every_cut_on_the_prompts_grid() {
        // a prompt alone (cut points: the system prompt's end), as the leader (cut at the share point too) and as the
        // waiting request (resumed at the share point): at any row budget, every chunk ends on the 32-row grid the
        // prompt has alone, and no chunk before the last is short enough to leave the prompt kernels
        let (a, b) = pair(5000, 333, 181);
        for sys in [0usize, 301, 1500] {
            let pts: Vec<usize> = if sys > 0 { vec![sys] } else { vec![] };
            let pos = share_point(&b, &pts, &a, &pts, 0, &[]).unwrap();
            let mut with = pts.clone();
            with.push(pos);
            for avail in [64usize, 300, 1000, 2048, 8192] {
                let on_grid = |ends: &[usize], len: usize, from: usize| {
                    let mut prev = from;
                    for &e in ends {
                        let origin = if e > sys { sys } else { 0 };
                        assert!(e == len || e == sys || (e - origin) % PROMPT_ALIGN == 0, "end {e} sys {sys} avail {avail}");
                        assert!(e == len || e - prev > DECODE_ROWS, "short chunk {prev}..{e}");
                        prev = e;
                    }
                };
                on_grid(&chunks(a.len(), &pts, 0, avail), a.len(), 0); // alone
                let lead = chunks(a.len(), &with, 0, avail);
                assert!(lead.contains(&pos));
                on_grid(&lead, a.len(), 0);
                on_grid(&chunks(b.len(), &pts, pos, avail), b.len(), pos); // resumed at the share point
                assert!(b.len() - pos > DECODE_ROWS && a.len() - pos > DECODE_ROWS);
            }
        }
    }
}

#[cfg(test)]
mod prompt_cut_tests {
    use super::{prompt_cut, prompt_cut_at};
    #[test]
    fn turn_points_sit_on_a_grid() {
        use super::{turn_points, DECODE_ROWS, PROMPT_ALIGN};
        // one node (lead off): every point on the 32-row grid of the point before, > DECODE_ROWS before it
        let t = turn_points(&[6561, 7842], 0, 2048);
        assert_eq!(t[0], (6561 - DECODE_ROWS - 1) / PROMPT_ALIGN * PROMPT_ALIGN);
        assert!(7842 - t[1] > DECODE_ROWS && 7842 - t[1] <= DECODE_ROWS + PROMPT_ALIGN && (t[1] - t[0]) % PROMPT_ALIGN == 0);
        // the same conversation one turn later cuts there again: a function of the points alone
        assert_eq!(&turn_points(&[6561, 7842, 9000], 0, 2048)[..2], &t[..]);
        // small-memory devices: points past FE_CACHE_LEAD_MIN on the chunk grid as lead_points
        assert_eq!(turn_points(&[20000, 20300, 23000], 4096, 2240), [17920, 20160, 22400]);
    }

    #[test]
    fn lead_points_sit_on_the_chunk_grid() {
        // short prompts: untouched
        assert_eq!(super::lead_points(&[100, 3000], 4096, 2240), [100, 3000]);
        // a 20000-token first turn: saved at 8 whole chunks; the next turn's point adds whole chunks from there
        assert_eq!(super::lead_points(&[20000], 4096, 2240), [17920]);
        assert_eq!(super::lead_points(&[20000, 20100], 4096, 2240), [17920]);
        assert_eq!(super::lead_points(&[20000, 20300, 23000], 4096, 2240), [17920, 20160, 22400]);
        // a short system prompt stays; the turn point counts chunks from it; a point exactly at a chunk end keeps a tail
        assert_eq!(super::lead_points(&[1500, 9000], 4096, 2240), [1500, 8220]);
        assert_eq!(super::lead_points(&[4480], 4096, 2240), [2240]);
        assert_eq!(super::lead_points(&[4497], 4096, 2240), [4480]);
        // no whole chunk before the point: dropped
        assert_eq!(super::lead_points(&[5000, 5100], 4096, 4096), [4096]);
    }

    #[test]
    fn cuts_stop_at_resume_points() {
        // uncached: points 0 and 700 (a turn boundary); budget 2048: [0, 700) then [700, 1800)
        assert_eq!(prompt_cut_at(1800, &[0, 700], 0, 2048), 700);
        assert_eq!(prompt_cut_at(1800, &[0, 700], 700, 2048), 1100);
        // a tight budget: on the grid from 0 below 700, from 700 above it
        assert_eq!(prompt_cut_at(1800, &[0, 700], 0, 300), 288);
        assert_eq!(prompt_cut_at(1800, &[0, 700], 288, 2048), 412);
        assert_eq!(prompt_cut_at(1800, &[0, 700], 700, 300), 288); // 988 = 700 + 9 * 32
        // never a <= 16-row tail before a point
        assert_eq!(prompt_cut_at(1800, &[0, 700], 0, 690), 672); // leaves 28 before the point
        assert_eq!(prompt_cut_at(1800, &[0, 650], 0, 645), 608); // 640 would leave 10 <= 16 before the point
        // resumed at 700 (cached): the same segments as uncached from 700 on
        assert_eq!(prompt_cut_at(1800, &[700], 700, 2048), 1100);
        assert_eq!(prompt_cut_at(1800, &[700], 700, 300), 288);
    }
    #[test]
    fn cuts_on_the_grid_from_base() {
        assert_eq!(prompt_cut(1800, 0, 0, 2048), 1800); // fits
        assert_eq!(prompt_cut(1800, 0, 0, 248), 224); // 248 -> 224
        assert_eq!(prompt_cut(1800, 0, 224, 2000), 1576); // the rest
        assert_eq!(prompt_cut(1800, 0, 0, 1790), 1760); // tail 40 > 16
        assert_eq!(prompt_cut(1800, 0, 0, 1795), 1760); // 1792 would leave 8: back to 1760
        assert_eq!(prompt_cut(1800, 0, 0, 20), 0); // < one step: wait
        assert_eq!(prompt_cut(1800, 100, 100, 248), 224); // cache hit at 100: cut at 324 = 100 + 7 * 32
        assert_eq!(prompt_cut(1800, 100, 324, 300), 288); // next cut 612 = 100 + 16 * 32
    }

    #[test]
    fn sampled_rows_fit_any_step() {
        // regression: 3 greedy 16-row verify segments + a sampled 16-row one + 2 ending prompts (1 sampled):
        // 66 logit rows, more than the old fixed 48-row host buffer; only the 17 sampled rows go to the host
        let lr_first = [0, 16, 32, 48, 64, 65];
        let sampled = [false, true, false, false, true, false];
        let (rows, n) = super::sampled_host_rows(&lr_first, 66, &sampled);
        assert_eq!(n, 17);
        assert_eq!(rows, vec![usize::MAX, 0, usize::MAX, usize::MAX, 16, usize::MAX]);
        // segments without logit rows (mid-prompt chunks) get none, even when sampled
        let (rows, n) = super::sampled_host_rows(&[0, 0, 1], 1, &[true, true, true]);
        assert_eq!((rows, n), (vec![usize::MAX, 0, usize::MAX], 1));
    }
}

#[cfg(test)]
mod guard_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn active(sp: Sampling, guard: Option<Box<dyn Guard>>, choices: Vec<Vec<u32>>) -> Active {
        let req = Request { id: "t".into(), guard, prefix_choices: choices.clone(), prompt_ids: vec![1, 2, 3], snapshots: Vec::new(), sampling: sp,
            cancel: Arc::new(AtomicBool::new(false)) };
        let opening = (!choices.is_empty()).then_some((0, choices));
        Active { req, sink: Box::new(|_| true), slot: 0, cap: 1000, fed: 3, cached: 0, cuts: Vec::new(), share: Vec::new(), out: Vec::new(),
            pending: None, drafts: Vec::new(), tree: None, copier: None, copied: (0, 0), out_idx: 0, depth: 4, acc_ema: 4.0, rounds: 0, accepted: 0,
            rng: 1, t0: Instant::now(), ttft: None, force: None, opening }
    }

    /// Counts what it is fed and acts at the n-th token.
    struct At(Arc<AtomicUsize>, usize, GuardAct);
    impl Guard for At {
        fn feed(&mut self, _t: u32) -> GuardAct {
            let n = self.0.fetch_add(1, Ordering::Relaxed) + 1;
            if n == self.1 { self.2.clone() } else { GuardAct::None }
        }
    }

    fn bits(x: f32) -> u16 {
        (x.to_bits() >> 16) as u16
    }

    #[test]
    fn the_guard_sees_exactly_the_tokens_that_go_out() {
        let fed = Arc::new(AtomicUsize::new(0));
        let sp = Sampling { max_tokens: 100, stop_token_ids: vec![9], ..Default::default() };
        // no action: every token up to the stop token is fed, none after it
        let mut a = active(sp.clone(), Some(Box::new(At(fed.clone(), 99, GuardAct::End))), Vec::new());
        assert_eq!(Worker::guard_scan(&mut a, &[5, 6, 9, 7, 8]), None);
        assert_eq!(fed.load(Ordering::Relaxed), 2);
        // an action at the third token of a five-token verify: three tokens are kept, the rest is given back
        let fed = Arc::new(AtomicUsize::new(0));
        let mut a = active(sp.clone(), Some(Box::new(At(fed.clone(), 3, GuardAct::Force(42)))), Vec::new());
        assert_eq!(Worker::guard_scan(&mut a, &[5, 6, 7, 8, 5]), Some((3, GuardAct::Force(42))));
        assert_eq!(fed.load(Ordering::Relaxed), 3);
        // the token limit: tokens past it never reach the guard
        let fed = Arc::new(AtomicUsize::new(0));
        let mut a = active(Sampling { max_tokens: 2, ..sp.clone() }, Some(Box::new(At(fed.clone(), 3, GuardAct::End))), Vec::new());
        assert_eq!(Worker::guard_scan(&mut a, &[5, 6, 7, 8]), None);
        assert_eq!(fed.load(Ordering::Relaxed), 2);
        // min_tokens: a stop token before it is content and is fed
        let fed = Arc::new(AtomicUsize::new(0));
        let mut a = active(Sampling { min_tokens: 2, ..sp.clone() }, Some(Box::new(At(fed.clone(), 99, GuardAct::End))), Vec::new());
        assert_eq!(Worker::guard_scan(&mut a, &[9, 5, 9, 6]), None);
        assert_eq!(fed.load(Ordering::Relaxed), 2);
        // no guard: nothing to do
        let mut a = active(sp, None, Vec::new());
        assert_eq!(Worker::guard_scan(&mut a, &[5, 6]), None);
    }

    #[test]
    fn guard_actions() {
        let greedy = Sampling::default();
        let sampled = Sampling { temperature: 0.7, ..Default::default() };
        let mut a = active(greedy.clone(), None, Vec::new());
        assert!(!a.special());
        assert!(!apply_act(&mut a, GuardAct::None, 47));
        assert!(apply_act(&mut a, GuardAct::End, 47));
        // a forced token: the next step is a plain row
        assert!(!apply_act(&mut a, GuardAct::Force(7), 47));
        assert!(a.special() && a.force == Some(7));
        // a switch to sampling drops the copy rows, a switch back to greedy brings them back (where the configuration has them)
        a.copier = Some(crate::copy::Copier::new(COPY_LO, 47));
        assert!(!apply_act(&mut a, GuardAct::Sampling(Box::new(sampled.clone())), 47));
        assert!(a.copier.is_none() && a.req.sampling == sampled);
        assert!(!apply_act(&mut a, GuardAct::Sampling(Box::new(greedy.clone())), 47));
        assert!(a.copier.is_some() && a.req.sampling == greedy);
        a.copier = None;
        apply_act(&mut a, GuardAct::Sampling(Box::new(greedy)), 0);
        assert!(a.copier.is_none());
    }

    #[test]
    fn a_constrained_opening_narrows_to_one_candidate() {
        let mut a = active(Sampling::default(), None, vec![vec![1, 2, 3], vec![1, 4], vec![5, 6]]);
        assert!(a.special());
        assert_eq!(a.opening_cands(), Some(vec![1, 1, 5]));
        a.opening_step(1);
        assert_eq!(a.opening_cands(), Some(vec![2, 4]));
        a.opening_step(2);
        assert_eq!(a.opening_cands(), Some(vec![3]));
        a.opening_step(3);
        assert!(a.opening.is_none() && !a.special());
        // a candidate that is complete ends the opening at once
        let mut a = active(Sampling::default(), None, vec![vec![1, 2, 3], vec![1, 4]]);
        a.opening_step(1);
        a.opening_step(4);
        assert!(a.opening.is_none());
    }

    #[test]
    fn host_row_helpers() {
        let mut row: Vec<u16> = [0.5f32, 3.0, -1.0, 2.0, 7.0].iter().map(|&x| bits(x)).collect();
        assert!(row_finite(&row));
        assert_eq!(argmax_row(&row), 4);
        // the first-token stop ban: the best other token
        let mut r = row.clone();
        mask_row(&mut r, None, &[4]);
        assert_eq!(argmax_row(&r), 1);
        // a constrained pick: only the candidates, the ban on top
        let mut r = row.clone();
        mask_row(&mut r, Some(&[0, 2, 3]), &[3]);
        assert_eq!(argmax_row(&r), 0);
        assert_eq!(r[1], BF16_NEG_INF);
        // sampling from a masked row never returns a masked token
        let sp = Sampling { temperature: 1.0, top_k: 20, top_p: 0.95, ..Default::default() };
        let mut r = row.clone();
        mask_row(&mut r, Some(&[2, 3]), &[]);
        let mut rng = 1u64;
        for _ in 0..200 {
            let t = sample(&r, &sp, &[], &[], &mut rng);
            assert!(t == 2 || t == 3, "{t}");
        }
        // an infinity or a NaN anywhere fails the finite check
        row[2] = bits(f32::NAN);
        assert!(!row_finite(&row));
        row[2] = bits(f32::INFINITY);
        assert!(!row_finite(&row));
        row[2] = BF16_NEG_INF;
        assert!(!row_finite(&row));
    }
}
