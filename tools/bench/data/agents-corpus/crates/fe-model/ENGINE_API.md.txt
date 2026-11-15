# ferrite engine API (fe-model)

Status: implemented; the server (fe-server `--engine ferrite`) runs on it.

Two layers. The server talks to the **scheduler**; the scheduler drives the **model engine**.

## 1. Model engine (`fe_model::model::Engine`)

Single-threaded, owns the CUDA stream, the weights and all per-slot state. Not `Sync`.

```rust
let ctx = fe_cuda::Context::new(0)?;                         // bind on the engine thread (also sets oom_score_adj 500)
let cfg = Config::load(&Config::locate(model_dir))?;
let k = Kernels::load(&Module::load(fe_model::K_CORE_CUBIN)?)?;
let w = Weights::load(&cfg, model_dir, &LoadOpts::from_env(), ctx.sm_count()? as u32)?;  // BF16 or EXL3
let mut e = Engine::with_spec(cfg, k, w, n_slots, max_ctx, max_rows, spec_depth)?;     // spec_depth 0 = no spec

e.reset_slot(slot)?;                                         // empty a slot (GDN + conv state zeroed)
e.forward(&[(slot_a, &prompt_chunk), (slot_b, &[tok]), ...], &logit_rows)?;
e.forward_ex(&segs, &logit_rows, hist, dec_rows)?;           // verify forward: hist = keep per-row GDN states
e.commit(slot, start_len, rows, keep)?;                      // keep the first `keep` verify rows, roll GDN back
e.forward_tree(&segs, &[None, Some(&parents)], &logit_rows, true, dec_rows)?;  // segment 1 is a verify tree
e.commit_tree(slot, start_len, rows, &path, taps_row0)?;     // keep a root path of the tree (see below)
let d = e.mtp_forward(&[MtpRow { slot, pos, token, src: MtpSrc::Tap(i) /* or Out(i) */ }], &draw)?;
let next = e.argmax(logit_rows.len())?;                      // greedy, or
e.logits_host(n, &mut buf)?;                                 // bf16 logits rows for host-side sampling
e.slot_len[slot]                                             // tokens currently cached in that slot
```

- `forward` takes segments `(slot, tokens)`: a long segment is prefill (chunked by the caller to `max_rows`),
  length-1 segments are decode rows; prefill and decode rows can share one call (continuous batching). Each
  slot's cache advances by its segment length.
- `dec_rows`: the leading rows that are decode / verify rows. They always run the EXL3 GEMV (16-row chunks),
  whatever the batch size; the remaining (prompt) rows run the prompt weights (NVFP4/FP8 with FE_PROMPT=fp4, else
  bf16). `forward` counts the leading length-1 segments. Logits rows always use the EXL3 head.
- `logit_rows` are indices into the concatenated rows; only those rows go through the head. The final-normed
  hidden of every row is kept (`taps`) for the MTP head.
- Batches of short segments (<= 16 rows, every segment < 16) replay a CUDA graph keyed by the segment lengths.
- Slots are fixed: `n_slots` x `max_ctx` tokens of KV (16 attention layers + the MTP layer, bf16) plus one GDN
  recurrent state (48 layers x 48 heads x 128 x 128 f32, ~151 MB) and conv state per slot; `spec_depth` adds
  that many rows of GDN state history per slot. 128K x 4 slots with the hybrid weights: ~80 GB used.
- The GDN state is a running state: a slot can be extended, and cut back only to a row of the last verify
  (`commit`, `commit_tree`).
- Verify trees (`forward_tree`): `parents[i]` makes segment i a tree of <= 32 rows (row 0 = the root = the pending
  token, parent -1; every other row's parent comes before it). Each row attends to the cache before the tree plus
  its own ancestors, at RoPE position start + depth (cache index start + row); the GDN layers run each row on the
  state along its own root path (read-only, the conv taps from its ancestors). A tree needs GDN replay: `hist`,
  rows <= `spec_depth`, leading decode rows. Trees, and decode / verify chains of any length, take fe_attn_vk in
  pieces of up to 16 rows, so every verify row is bitwise its 1-row decode.
- `commit_tree(slot, start, rows, path, taps)`: `path` = the kept root path (tree rows, path[0] = 0, increasing).
  One launch moves the kept rows' K/V (and the saved GDN replay inputs, and with `taps` = the segment's first row
  in the last forward, the drafter feature-tap rows) to consecutive rows, then it commits like `commit` with
  keep = path.len(). A path that is a chain prefix is exactly `commit`. Under TP the follower receives the parents
  with the forward and a CommitTree call. Prompt-cache reuse is "new prompt starts with everything already in the slot" (multi-turn chat).
- Env (LoadOpts::from_env): FE_PROMPT=fp4 (+ FE_PROMPT_WEIGHTS, default the ModelOpt NVFP4 checkpoint),
  FE_BF16=0/1, FE_EXL3_DECODE=0, FE_MTP=0, FE_DRAFT_VOCAB (64K), FE_GRAPHS=0, FE_SWIGLU=0, FE_GDN_CHUNK=0.

## 2. Scheduler (`fe_model::sched`) — the server-facing API

Thread-safe front, one engine thread behind it. No tokio dependency; the server adapts it to its own
`fe_server::engine::Engine` trait (`submit(GenRequest, TokenSink)`), which maps 1:1:

```rust
pub struct SchedOptions { pub n_slots: usize, pub max_ctx: usize, pub prefill_chunk: usize }

pub struct Sampling {                // same meaning as fe_server::engine::SamplingParams
    pub max_tokens: usize, pub min_tokens: usize,
    pub temperature: f32, pub top_p: f32, pub top_k: usize, pub min_p: f32,
    pub presence_penalty: f32, pub frequency_penalty: f32, pub repetition_penalty: f32,
    pub seed: Option<u64>, pub stop_token_ids: Vec<u32>,
}

pub enum Event {
    Tokens(Vec<u32>),                                        // newly sampled ids (stop id never included)
    Finished { reason: Finish, cached_prompt_tokens: usize },
}
pub enum Finish { Stop(u32), Length, Cancelled, Error(String) }

pub struct Request {
    pub id: String,
    pub prompt_ids: Vec<u32>,
    pub sampling: Sampling,
    pub cancel: Arc<AtomicBool>,
}

impl Scheduler {
    /// Loads the model on a new engine thread; returns when it is ready.
    pub fn start(model_dir: &Path, opts: SchedOptions) -> Result<Scheduler>;
    pub fn info(&self) -> (/*max_batch*/ usize, /*max_context*/ usize);
    /// Non-blocking. `sink` is called on the engine thread; return false to cancel.
    /// Exactly one Event::Finished is delivered last.
    pub fn submit(&self, req: Request, sink: Box<dyn FnMut(Event) -> bool + Send>);
}
```

Server adapter (lives in fe-server):

```rust
impl fe_server::engine::Engine for FerriteEngine {
    fn info(&self) -> EngineInfo { let (b, c) = self.0.info(); EngineInfo { name: "ferrite".into(), max_batch: b, max_context: c } }
    fn submit(&self, r: GenRequest, sink: TokenSink) {
        let mut sink = Some(sink);
        self.0.submit(Request { id: r.id, prompt_ids: r.prompt_ids, sampling: r.params.into(), cancel: r.cancel.flag() },
            Box::new(move |ev| match ev {
                Event::Tokens(t) => sink.as_ref().map_or(false, |s| s.send_tokens(t)),
                Event::Finished { reason, cached_prompt_tokens } => {
                    if let Some(s) = sink.take() { s.finish(Finish { reason: reason.into(), cached_prompt_tokens }) }
                    false
                }
            }));
    }
}
```

### 2.1 Scheduling

- Each step: MTP chain drafts for every decoding stream (depth adaptive per stream, all verify rows <= 16 so
  they share one GEMV pass), then one target forward with every decoding stream's [pending, drafts...] and
  prefill chunks of the others (round-robin, total rows <= `prefill_chunk`), then acceptance + rollback, then
  one MTP absorb of the kept rows (and of prompt chunks) that also draws each stream's next first draft.
- Greedy requests accept drafts while they match the argmax; sampled requests use speculative sampling with
  point-mass drafts (accept d with p(d), else sample p without d), so the output distribution is the target's.
- Trees (FE_TREE=1, greedy requests, a drafter whose `Drafter::draft_tree` returns one): the verify segment is the
  drafter's tree (up to min(16 / decoding streams - 1, spec_depth - 1) nodes); the walk from the root follows the
  child whose token is the current row's argmax, the bonus is the argmax after the last kept row, and
  `commit_tree` keeps that path (the drafter then absorbs consecutive tap rows as for a chain).
  `ferrite <model> tree-test "<prompt>" [n] [nodes]` gates the engine side with oracle trees against plain greedy.
- Copy rows (FE_COPY_ROWS, the servers' default 47; tree streams only; the idea follows TensorFold's copy rows):
  a suffix lookup over the stream's prompt + reply (crate::copy) proposes what followed the latest longest earlier
  match of its last tokens. The copy is grafted
  onto the drafter's tree past the rows it agrees with (FE_COPY_AGREE, 3), up to a 32-row tree, within the step's
  verify rows and its 16-row cost tier unless the stream's copies keep landing whole or the match is >= 24 tokens;
  a lone stream whose copies land whole verifies its copy alone as a 32- or 48-row chain without drafting. TP2
  server, copies off -> on: an edit (a module returned with a rename) 140 -> 563 tok/s, a reply echoing its prompt
  323 -> 569; prose without repeats unchanged; seeded code benchmarks 1-2% slower.
- Host sampling from the bf16 logits rows (temperature / top-k / top-p / min-p / penalties / seed).
- Cancellation is checked between steps; a cancelled or finished request frees its slot immediately.
- SchedOptions { n_slots, max_ctx, prefill_chunk, spec_depth (FE_SPEC, default 8; 0 = off) }.

### 2.2 Limits

`max_batch = n_slots` (4 by default), `max_context = max_ctx` (131072 by default). Prompts longer than
`max_ctx - 1` finish with `Error`.

### 2.3 Prompt cache

Per slot the engine remembers the token ids in its cache. A new request takes the free slot with the
longest cached sequence that is a full prefix of its prompt and prefills only the remainder
(`cached_prompt_tokens` = that length); otherwise the least recently used free slot is reset.
Planned: GDN-state snapshots at prompt boundaries so a shared system prompt can be reused by any slot.
