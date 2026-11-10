//! The boundary between the HTTP server and the model engine.
//!
//! The server does everything text-shaped (chat template, tokenization, detokenization, stop strings,
//! reasoning / tool-call parsing, the OpenAI wire format). The engine only sees token ids:
//!
//! 1. the server calls [`Engine::submit`] with the full prompt token ids, the sampling parameters and a
//!    [`CancelToken`], plus a [`TokenSink`] to stream results into;
//! 2. the engine returns immediately and generates on its own threads, calling
//!    [`TokenSink::send_tokens`] for every newly sampled token (or several at once when speculative
//!    decoding accepts a run), and finally [`TokenSink::finish`] exactly once;
//! 3. the engine stops early when [`CancelToken::is_cancelled`] turns true or `send_tokens` returns
//!    `false` (the client went away, or the server hit a stop string / `max_tokens`) and then calls
//!    `finish` with [`FinishReason::Cancelled`].
//!
//! Several requests are submitted concurrently (up to [`EngineInfo::max_batch`] at a time; the server
//! queues the rest), so `submit` must be callable from many tasks at once (`Engine: Send + Sync`).
//! Dropping the sink without calling `finish` is reported to the client as an engine error.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;

/// Per-request sampling parameters, already resolved against the model's `generation_config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    /// Hard cap on generated tokens (the stop token, if any, is not counted here).
    pub max_tokens: usize,
    /// Do not stop on a stop token before this many tokens have been generated.
    pub min_tokens: usize,
    /// 0.0 means greedy (argmax); then top_p / top_k / min_p / seed are irrelevant.
    pub temperature: f32,
    /// Nucleus sampling; 1.0 = off.
    pub top_p: f32,
    /// 0 = off.
    pub top_k: usize,
    /// 0.0 = off.
    pub min_p: f32,
    /// OpenAI-style additive penalties; 0.0 = off.
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// HF-style multiplicative penalty over prompt + output tokens; 1.0 = off.
    pub repetition_penalty: f32,
    /// RNG seed for reproducible sampling; `None` = engine-chosen.
    pub seed: Option<u64>,
    /// Token ids that end generation (the model's eos ids plus any request `stop_token_ids`).
    /// When one is sampled the engine finishes with [`FinishReason::Stop`] and does NOT send it
    /// through [`TokenSink::send_tokens`].
    pub stop_token_ids: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            min_tokens: 0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: None,
            stop_token_ids: Vec::new(),
        }
    }
}

/// Shared cancellation flag. Cheap to clone; the server sets it, the engine polls it between steps.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    /// The shared flag itself, for engines that take a plain `Arc<AtomicBool>`.
    pub fn flag(&self) -> Arc<AtomicBool> {
        self.0.clone()
    }
    pub fn from_flag(flag: Arc<AtomicBool>) -> Self {
        Self(flag)
    }
}

/// One generation request as the engine sees it.
#[derive(Clone, Debug)]
pub struct GenRequest {
    /// Request id (`chatcmpl-...` / `cmpl-...`), for logs only.
    pub id: String,
    /// The full rendered prompt, tokenized (no special tokens added beyond what the template wrote).
    /// Engines with a prefix / prompt cache match on these ids.
    pub prompt_ids: Vec<u32>,
    /// Snapshot points (crate::snapshots): ascending positions in prompt_ids where an engine with a prompt cache
    /// keeps a resumable state (a turn's reply start, the end of the shared system prompt). Empty: none.
    pub snapshots: Vec<usize>,
    pub params: SamplingParams,
    pub cancel: CancelToken,
}

/// Why generation ended.
#[derive(Clone, Debug, PartialEq)]
pub enum FinishReason {
    /// A stop token (one of `stop_token_ids`) was sampled; it is not included in the tokens sent.
    Stop(u32),
    /// `max_tokens` tokens were generated.
    Length,
    /// The cancel token was set or the sink was closed.
    Cancelled,
    /// The engine failed; the message goes to the client.
    Error(String),
}

/// Final event of a generation.
#[derive(Clone, Debug, PartialEq)]
pub struct Finish {
    pub reason: FinishReason,
    /// How many prompt tokens came from the engine's prompt / prefix cache (0 if none or unknown).
    pub cached_prompt_tokens: usize,
    /// Engine-side timings for the responses' llama-server-style `timings` object; None (the default) = no such key.
    pub timings: Option<Timings>,
}

impl Finish {
    pub fn new(reason: FinishReason) -> Self {
        Self { reason, cached_prompt_tokens: 0, timings: None }
    }
}

/// Engine-side timings of one request, as llama-server (and arc-engine) report them in `timings`: tools such as quickbench,
/// gate.py and long.py prefer them to client-side estimates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timings {
    /// prompt tokens run (the cached ones not counted)
    pub prompt_n: usize,
    /// prompt tokens taken from the prompt cache
    pub cache_n: usize,
    /// prompt processing up to the first token
    pub prompt_ms: f64,
    /// generated tokens, the first one included
    pub predicted_n: usize,
    /// generation after the first token
    pub predicted_ms: f64,
    /// speculative decoding: (draft tokens proposed, accepted); None without speculation
    pub draft: Option<(u64, u64)>,
}

impl Timings {
    /// The `timings` object (rates as llama-server / arc-engine: prompt_n / prompt time, (predicted_n - 1) / generation time).
    pub fn to_json(&self) -> serde_json::Value {
        let rate = |n: f64, ms: f64| if ms > 0.0 { n * 1000.0 / ms } else { 0.0 };
        let mut t = serde_json::json!({
            "prompt_n": self.prompt_n,
            "cache_n": self.cache_n,
            "prompt_ms": self.prompt_ms,
            "prompt_per_second": rate(self.prompt_n as f64, self.prompt_ms),
            "predicted_n": self.predicted_n,
            "predicted_ms": self.predicted_ms,
            "predicted_per_second": if self.predicted_n > 1 { rate((self.predicted_n - 1) as f64, self.predicted_ms) } else { 0.0 },
        });
        if let Some((n, a)) = self.draft {
            t["draft_n"] = n.into();
            t["draft_n_accepted"] = a.into();
        }
        t
    }
}

/// What travels from the engine to the server.
#[derive(Clone, Debug, PartialEq)]
pub enum GenEvent {
    /// Newly generated token ids, in order.
    Tokens(Vec<u32>),
    /// Generation ended; nothing follows.
    Finished(Finish),
}

/// The engine's handle for streaming one request's output back to the server.
///
/// Plain synchronous calls: usable from any thread, never blocks (unbounded channel).
/// It also holds the request's concurrency slot, released when the sink is dropped, so an engine
/// must drop (or `finish`) the sink as soon as the request is done.
pub struct TokenSink {
    tx: mpsc::UnboundedSender<GenEvent>,
    _slot: Option<Box<dyn Send + Sync>>,
}

impl TokenSink {
    pub fn new(tx: mpsc::UnboundedSender<GenEvent>) -> Self {
        Self { tx, _slot: None }
    }

    /// A sink that also owns `slot` until it is dropped (the server passes its semaphore permit).
    pub fn with_slot(tx: mpsc::UnboundedSender<GenEvent>, slot: Box<dyn Send + Sync>) -> Self {
        Self { tx, _slot: Some(slot) }
    }

    /// Send newly generated tokens. Returns `false` when the server no longer wants output
    /// (client disconnected / request finished): the engine should stop and `finish(Cancelled)`.
    pub fn send_tokens(&self, ids: Vec<u32>) -> bool {
        if ids.is_empty() {
            return !self.tx.is_closed();
        }
        self.tx.send(GenEvent::Tokens(ids)).is_ok()
    }

    /// True once the receiving side is gone.
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }

    /// End the request. Consumes the sink, which releases the concurrency slot.
    pub fn finish(self, finish: Finish) {
        let _ = self.tx.send(GenEvent::Finished(finish));
    }
}

/// Static facts the server needs about the engine.
#[derive(Clone, Debug)]
pub struct EngineInfo {
    /// Shown in logs.
    pub name: String,
    /// How many requests the engine decodes at once; the server runs at most this many and queues the rest.
    pub max_batch: usize,
    /// Maximum prompt + generated tokens per request.
    pub max_context: usize,
}

/// A token-in / token-out generation engine.
pub trait Engine: Send + Sync + 'static {
    fn info(&self) -> EngineInfo;

    /// Start generating for `req`; must return quickly (do the work on engine threads).
    /// Results go to `sink`: zero or more `send_tokens`, then exactly one `finish`.
    fn submit(&self, req: GenRequest, sink: TokenSink);
}
