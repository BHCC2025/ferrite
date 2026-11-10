//! fe-server: the OpenAI-compatible HTTP front end of ferrite.
//!
//! The server owns everything text-shaped (chat template, tokenizer, streaming detokenization, stop
//! strings, reasoning and tool-call parsing, OpenAI JSON / SSE); the model engine behind
//! [`engine::Engine`] only maps prompt token ids to generated token ids.
pub mod api;
pub mod batch;
pub mod engine;
#[cfg(feature = "ferrite")]
pub mod ferrite;
pub mod mock;
pub mod parser;
pub mod pyjson;
pub mod sampler;
pub mod server;
pub mod snapshots;
pub mod stop;
pub mod template;
pub mod tokenizer;
pub mod toolcall;
pub mod util;
#[cfg(feature = "xe")]
pub mod xe;
