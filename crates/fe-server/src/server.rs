//! The axum app: routes, request preparation, the token → text → deltas pipeline, SSE / JSON output,
//! concurrency slots and cancellation.
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use axum::Json as AxJson;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Map, Value as Json, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::api::{ApiError, ChatRequest, CompletionRequest, SamplingRequest, parse_body, wants_logprobs};
use crate::engine::{CancelToken, Engine, FinishReason, GenEvent, GenRequest, ImageInput, SamplingParams, Timings, TokenSink};
use crate::parser::{Delta, ForcedCall, OutputParser, ParserConfig, Parsed};
use crate::stop::StopMatcher;
use crate::template::{ChatTemplate, normalize_messages_with};
use crate::tokenizer::{IncrementalDecoder, Tok};
use crate::toolcall::ToolSet;
use crate::util::{random_id, unix_now};

/// Sampling defaults (from `generation_config.json`) and stop ids.
#[derive(Clone, Debug)]
pub struct GenDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub eos_ids: Vec<u32>,
    pub phase: PhaseDefaults,
    /// max_tokens for a request that sends none (FE_DEFAULT_MAX_TOKENS, 0 = no cap; the Inkling template gets 32768 unless the env says otherwise, see
    /// `apply_template`): the model card advises limiting output because reasoning can loop; the context's end is the only other limit
    pub default_max_tokens: Option<usize>,
    /// reasoning effort for a request that sends none (FE_DEFAULT_REASONING_EFFORT: a name or a number; Inkling template only); unset = the template's own 0.9
    pub default_reasoning_effort: Option<Json>,
}

/// Reasoning tokens after which the engine closes the block itself when neither sampling.json nor FE_MAX_REASONING_TOKENS says otherwise.
pub const DEFAULT_MAX_REASONING: usize = 8192;

/// Thinking-phase behaviour from `sampling.json` in the model dir (env FE_THINK_TEMP / FE_THINK_TOP_P / FE_MAX_REASONING_TOKENS /
/// FE_LOOP_BREAKER / FE_LOOP_RULES override): the reasoning block is sampled at these settings when the client asks for greedy (or says nothing and the
/// model default is greedy); the answer / tool-call phase keeps the client's values.
/// One mode's sampling profile (`profiles.thinking` / `profiles.non_thinking` in sampling.json): the values a request of that mode gets
/// for the fields its client does not send. A field the profile leaves out falls back to generation_config.json (penalties: off).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Profile {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

impl Profile {
    pub fn from_json(v: Option<&Json>) -> Self {
        let f = |k: &str| v.and_then(|v| v.get(k)).and_then(Json::as_f64).map(|x| x as f32);
        Self {
            temperature: f("temperature"),
            top_p: f("top_p"),
            top_k: v.and_then(|v| v.get("top_k")).and_then(Json::as_u64).map(|x| x as usize),
            min_p: f("min_p"),
            presence_penalty: f("presence_penalty"),
            frequency_penalty: f("frequency_penalty"),
            repetition_penalty: f("repetition_penalty"),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PhaseDefaults {
    /// per-mode sampling profiles: a request that thinks / one that does not. A field the client sends always wins.
    pub profile_thinking: Profile,
    pub profile_plain: Profile,
    /// `greedy_floor` in sampling.json (env FE_GREEDY_FLOOR=1/0 overrides; default off): a client temperature of 0 is replaced by the
    /// thinking profile's temperature inside the reasoning block only (the answer / tool call stays greedy)
    pub greedy_floor: bool,
    /// `greedy_keeps_fast_path` in sampling.json (env FE_GREEDY_FAST_PATH=1/0 overrides; default on): a client that sends
    /// temperature 0 asks for greedy decoding, so the profile adds no penalty to it (presence / frequency / repetition penalty stay
    /// at the model defaults unless the client sends them): the request keeps the device argmax and its copy rows. Off: the
    /// profile fills every field the client omits, penalties included. A client that omits temperature gets the whole profile
    /// either way.
    pub greedy_keeps_fast_path: bool,
    /// watchdog (`watchdog.no_token_timeout_s`, default 60, env FE_NO_TOKEN_TIMEOUT; `watchdog.reply_wall_limit_s`, default 0 = off,
    /// env FE_REPLY_WALL_LIMIT), seconds; 0 = off
    pub no_token_timeout: f64,
    pub reply_wall_limit: f64,
    pub max_reasoning: usize,
    pub loop_breaker: bool,
    /// loop-breaker thresholds: `loop_rules` in sampling.json, env FE_LOOP_RULES (the same JSON object) overrides it
    pub loop_rules: crate::loopdet::LoopCfg,
    pub think_end: Option<u32>,
    pub call_marks: Vec<Vec<u32>>,
}

impl PhaseDefaults {
    pub fn from_dir(dir: &Path, tok: &Tok) -> Self {
        // sampling.json of the model dir; FE_SAMPLING_JSON=<file> reads another one instead (A/B runs of profiles and thresholds)
        let path = std::env::var_os("FE_SAMPLING_JSON").filter(|p| !p.is_empty()).map_or_else(|| dir.join("sampling.json"), PathBuf::from);
        let cfg: Json = match std::fs::read_to_string(&path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
                tracing::warn!("{}: not valid JSON ({e}): ignored, built-in defaults apply", path.display());
                Json::Null
            }),
            Err(_) => Json::Null,
        };
        let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
        // profiles.thinking / profiles.non_thinking; the older top-level `thinking` object (temperature, top_p) and FE_THINK_TEMP /
        // FE_THINK_TOP_P still fill the thinking profile
        let profiles = cfg.get("profiles");
        let mut profile_thinking = Profile::from_json(profiles.and_then(|p| p.get("thinking")).or(cfg.get("thinking")));
        if let Some(t) = envf("FE_THINK_TEMP") {
            profile_thinking.temperature = Some(t as f32);
        }
        if let Some(t) = envf("FE_THINK_TOP_P") {
            profile_thinking.top_p = Some(t as f32);
        }
        let profile_plain = Profile::from_json(profiles.and_then(|p| p.get("non_thinking")));
        let greedy_floor = match std::env::var("FE_GREEDY_FLOOR").ok().as_deref() {
            Some("0") => false,
            Some(_) => true,
            None => cfg.get("greedy_floor").and_then(Json::as_bool).unwrap_or(false),
        };
        let greedy_keeps_fast_path = match std::env::var("FE_GREEDY_FAST_PATH").ok().as_deref() {
            Some("0") => false,
            Some(_) => true,
            None => cfg.get("greedy_keeps_fast_path").and_then(Json::as_bool).unwrap_or(true),
        };
        let wd = cfg.get("watchdog");
        let wdf = |k: &str| wd.and_then(|w| w.get(k)).and_then(Json::as_f64);
        let no_token_timeout = envf("FE_NO_TOKEN_TIMEOUT").or(wdf("no_token_timeout_s")).unwrap_or(60.0).max(0.0);
        let reply_wall_limit = envf("FE_REPLY_WALL_LIMIT").or(wdf("reply_wall_limit_s")).unwrap_or(0.0).max(0.0);
        // default 8192; 0 = no cap
        let max_reasoning = envf("FE_MAX_REASONING_TOKENS").or(cfg.get("max_reasoning_tokens").and_then(Json::as_f64)).unwrap_or(DEFAULT_MAX_REASONING as f64) as usize;
        let loop_breaker = match std::env::var("FE_LOOP_BREAKER").ok().as_deref() {
            Some("0") => false,
            Some(_) => true,
            None => cfg.get("loop_breaker").and_then(Json::as_bool).unwrap_or(false),
        };
        let env_rules: Option<Json> = std::env::var("FE_LOOP_RULES").ok().and_then(|s| match serde_json::from_str(&s) {
            Ok(j) => Some(j),
            Err(e) => {
                tracing::warn!("FE_LOOP_RULES is not valid JSON ({e}): ignored");
                None
            }
        });
        let mut loop_rules = crate::loopdet::LoopCfg::from_json(env_rules.as_ref().or(cfg.get("loop_rules")));
        // the stretch after a tool-call rewind samples with the thinking profile's values
        if let Some(t) = profile_thinking.temperature.filter(|&t| t > 0.0) {
            loop_rules.recover_temperature = t;
        }
        if let Some(t) = profile_thinking.top_p {
            loop_rules.recover_top_p = t;
        }
        // the token that ends the reasoning block (`think_end` in sampling.json, default `</think>`); None when the vocabulary has no such token
        let think_end = tok.token_id(cfg.get("think_end").and_then(Json::as_str).unwrap_or("</think>"));
        let mut call_marks = Vec::new();
        if let Some(arr) = cfg.get("call_marks").and_then(Json::as_array) {
            for m in arr.iter().filter_map(Json::as_str) {
                if let Ok(ids) = tok.encode(m) {
                    if !ids.is_empty() {
                        call_marks.push(ids);
                    }
                }
            }
        }
        Self { profile_thinking, profile_plain, greedy_floor, greedy_keeps_fast_path, no_token_timeout, reply_wall_limit, max_reasoning, loop_breaker, loop_rules, think_end, call_marks }
    }
}

impl GenDefaults {
    /// Reads `generation_config.json` (if present) and adds the tokenizer's `<|im_end|>` / eos ids.
    pub fn from_dir(dir: &Path, tok: &Tok) -> Self {
        let cfg: Json = std::fs::read_to_string(dir.join("generation_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Json::Null);
        let f = |k: &str, d: f32| cfg.get(k).and_then(Json::as_f64).map(|v| v as f32).unwrap_or(d);
        let mut eos_ids: Vec<u32> = match cfg.get("eos_token_id") {
            Some(Json::Number(n)) => n.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
            Some(Json::Array(a)) => a.iter().filter_map(Json::as_u64).map(|v| v as u32).collect(),
            _ => Vec::new(),
        };
        let tok_eos: Option<String> = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Json>(&s).ok())
            .and_then(|c| c.get("eos_token").and_then(Json::as_str).map(str::to_string));
        for t in tok_eos.iter().map(String::as_str).chain(["<|im_end|>"]) {
            if let Some(id) = tok.token_id(t) {
                if !eos_ids.contains(&id) {
                    eos_ids.push(id);
                }
            }
        }
        let do_sample = cfg.get("do_sample").and_then(Json::as_bool).unwrap_or(true);
        // FE_DEFAULT_TEMPERATURE: the temperature of a request that sends none (default: generation_config.json, i.e. the publisher's)
        let env_temp = std::env::var("FE_DEFAULT_TEMPERATURE").ok().and_then(|v| v.trim().parse::<f32>().ok()).filter(|t| t.is_finite() && *t >= 0.0);
        let default_max_tokens = std::env::var("FE_DEFAULT_MAX_TOKENS").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&n| n > 0);
        let default_reasoning_effort = std::env::var("FE_DEFAULT_REASONING_EFFORT").ok().map(|v| v.trim().to_lowercase()).filter(|v| !v.is_empty())
            .map(|v| v.parse::<f64>().map(|f| json!(f)).unwrap_or_else(|_| json!(v)));
        Self {
            default_max_tokens,
            default_reasoning_effort,
            temperature: env_temp.unwrap_or(if do_sample { f("temperature", 1.0) } else { 0.0 }),
            top_p: f("top_p", 1.0),
            top_k: cfg.get("top_k").and_then(Json::as_u64).unwrap_or(0) as usize,
            min_p: f("min_p", 0.0),
            repetition_penalty: f("repetition_penalty", 1.0),
            eos_ids,
            phase: PhaseDefaults::from_dir(dir, tok),
        }
    }
}

impl GenDefaults {
    /// Defaults that depend on the chat template: the Inkling model gets a max_tokens cap of 32768 for requests that send none (FE_DEFAULT_MAX_TOKENS
    /// overrides it, 0 turns it off).
    pub fn apply_template(&mut self, t: &ChatTemplate) {
        if t.inkling && self.default_max_tokens.is_none() && std::env::var_os("FE_DEFAULT_MAX_TOKENS").is_none() {
            self.default_max_tokens = Some(INK_DEFAULT_MAX_TOKENS);
        }
    }
}

pub const INK_DEFAULT_MAX_TOKENS: usize = 32768;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// The model id reported everywhere (requests may name any model).
    pub model_id: String,
    /// Requests generating at once; the rest wait in a FIFO queue. `/health` reports it as `max_batch_size`;
    /// [`AppState::new`] keeps it within the engine's `max_batch`.
    pub max_concurrent: usize,
    /// Prompt + output token limit per request (`/health`: `max_model_len`); [`AppState::new`] keeps it within the
    /// engine's `max_context`.
    pub max_context: usize,
    pub defaults: GenDefaults,
}

pub struct AppState {
    pub engine: Arc<dyn Engine>,
    pub tok: Arc<Tok>,
    pub template: Arc<ChatTemplate>,
    pub cfg: ServerConfig,
    /// the chat template's snapshot markers (crate::snapshots)
    marks: crate::snapshots::Marks,
    slots: Arc<Semaphore>,
    queued: Arc<AtomicUsize>,
    running: Arc<AtomicUsize>,
    created: u64,
    /// Request capture (env FE_DUMP_REQUESTS=<dir>, off when unset): every chat-completions body as received in `<id>.json`, and
    /// `<id>.meta.json` when the request completes. Set once at start (tests set it directly).
    pub dump_dir: OnceLock<PathBuf>,
    /// Image input (the ink engine with the Inkling template): the limits of a request. None: image parts become a text placeholder.
    #[cfg(feature = "ink")]
    image_limits: Option<fe_ink::images::Limits>,
}

/// Write `<dir>/<name>` (a capture file); failures are logged, never returned (capture must not break a request).
fn capture_write(dir: &Path, name: &str, bytes: &[u8]) {
    if let Err(e) = std::fs::create_dir_all(dir).and_then(|_| std::fs::write(dir.join(name), bytes)) {
        tracing::warn!("request capture: cannot write {}: {e}", dir.join(name).display());
    }
}

impl AppState {
    /// The body of a chat-completions request exactly as received (replayable with tools/conformance/run.py --replay).
    fn capture_body(&self, id: &str, body: &[u8]) {
        if let Some(dir) = self.dump_dir.get() {
            capture_write(dir, &format!("{id}.json"), body);
        }
    }

    /// The sidecar written when the request completes (or is rejected / cancelled).
    fn capture_meta(&self, id: &str, meta: Json) {
        if let Some(dir) = self.dump_dir.get() {
            capture_write(dir, &format!("{id}.meta.json"), serde_json::to_string_pretty(&meta).unwrap_or_default().as_bytes());
        }
    }

    /// `cfg.max_concurrent` and `cfg.max_context` are capped at what the engine reports ([`Engine::info`]): the
    /// server never runs, and `/health` never claims, more requests at once or a longer request than the loaded
    /// engine handles.
    pub fn new(engine: Arc<dyn Engine>, tok: Arc<Tok>, template: Arc<ChatTemplate>, mut cfg: ServerConfig) -> Arc<Self> {
        let info = engine.info();
        cfg.max_concurrent = cfg.max_concurrent.clamp(1, info.max_batch.max(1));
        cfg.max_context = cfg.max_context.min(info.max_context);
        {
            // the loop breaker's log line shows the cycle as text
            let t = tok.clone();
            crate::loopdet::set_detok(Box::new(move |ids| t.decode(ids, false).unwrap_or_default()));
        }
        let dump_dir = OnceLock::new();
        if let Some(d) = std::env::var_os("FE_DUMP_REQUESTS").filter(|d| !d.is_empty()) {
            tracing::info!("request capture on: chat-completions bodies and .meta.json sidecars go to {}", Path::new(&d).display());
            let _ = dump_dir.set(PathBuf::from(d));
        }
        #[cfg(feature = "ink")]
        let image_limits = (engine.accepts_images() && template.inkling).then(fe_ink::images::Limits::from_env);
        #[cfg(feature = "ink")]
        if let Some(l) = &image_limits {
            let over = if l.drop_oldest {
                format!("a request with more pictures or image tokens leaves out its oldest pictures, {} at a time (INK_IMAGE_OVERFLOW=drop)", l.overflow_batch())
            } else {
                "a request with more pictures or image tokens is refused (INK_IMAGE_OVERFLOW=reject)".to_string()
            };
            tracing::info!("image input on: PNG and JPEG in data: URLs, up to {} pictures and {} image tokens a request, {} image tokens a picture (a larger picture is shrunk to fit) and {} MiB a file (INK_IMAGE_MAX, INK_IMAGE_TOKENS, INK_IMAGE_TOKENS_EACH, INK_IMAGE_MB); {over}",
                l.max_images, l.max_tokens, l.max_tokens_each, l.max_bytes >> 20);
        }
        Arc::new(Self {
            #[cfg(feature = "ink")]
            image_limits,
            dump_dir,
            slots: Arc::new(Semaphore::new(cfg.max_concurrent.max(1))),
            marks: crate::snapshots::Marks::new(&tok)
                .unwrap_or_else(|_| crate::snapshots::Marks::from_ids(Vec::new(), Vec::new(), Vec::new())),
            engine,
            tok,
            template,
            cfg,
            queued: Arc::new(AtomicUsize::new(0)),
            running: Arc::new(AtomicUsize::new(0)),
            created: unix_now(),
        })
    }

    pub fn running(&self) -> usize {
        self.running.load(Ordering::Relaxed)
    }

    pub fn queued(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }

    /// The engine reads pictures: image parts of a chat request are decoded and placed in the prompt.
    pub fn accepts_images(&self) -> bool {
        #[cfg(feature = "ink")]
        return self.image_limits.is_some();
        #[cfg(not(feature = "ink"))]
        false
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    // pictures arrive inside the request body (base64)
    let body_limit = if state.accepts_images() { IMAGE_BODY_LIMIT } else { BODY_LIMIT };
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(model_one))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        // a 128K-token prompt is about 0.5 MB of text; larger bodies are rejected before any tokenizing
        .layer(DefaultBodyLimit::max(body_limit))
        .with_state(state)
}

/// `GET /health` (and `/v1/health`). Clients size themselves from it, so the field names are a contract
/// (tests/health_contract.rs): `max_batch_size` = requests the loaded engine decodes at once (never more than the
/// engine reports), `max_model_len` = prompt + output tokens a request may use, `running` = requests holding a
/// slot now, `queued` = requests waiting for one.
async fn health(State(s): State<Arc<AppState>>) -> Response {
    if let Some(why) = s.engine.dead() {
        // the engine is gone (the process exits in a moment): a load balancer or supervisor must see it
        return (axum::http::StatusCode::SERVICE_UNAVAILABLE, AxJson(json!({
            "status": "error",
            "error": format!("the engine stopped: {why}"),
            "model": s.cfg.model_id,
            "engine": s.engine.info().name,
        }))).into_response();
    }
    AxJson(json!({
        "status": "ok",
        "model": s.cfg.model_id,
        "engine": s.engine.info().name,
        "version": env!("CARGO_PKG_VERSION"),
        "max_batch_size": s.cfg.max_concurrent,
        "max_model_len": s.cfg.max_context,
        "running": s.running(),
        "queued": s.queued(),
    })).into_response()
}

/// Request bodies above this many bytes get a 413 (axum's DefaultBodyLimit).
pub const BODY_LIMIT: usize = 32 << 20;
/// The same for a server whose engine reads pictures (they come base64-encoded in the body).
pub const IMAGE_BODY_LIMIT: usize = 256 << 20;

fn model_card(s: &AppState) -> Json {
    json!({
        "id": s.cfg.model_id,
        "object": "model",
        "created": s.created,
        "owned_by": "ferrite",
        "root": s.cfg.model_id,
        "parent": null,
        "max_model_len": s.cfg.max_context,
        "context_length": s.cfg.max_context,
    })
}

async fn models(State(s): State<Arc<AppState>>) -> AxJson<Json> {
    AxJson(json!({"object": "list", "data": [model_card(&s)]}))
}

async fn model_one(State(s): State<Arc<AppState>>) -> AxJson<Json> {
    AxJson(model_card(&s))
}

// ---------------------------------------------------------------------------------------------
// request preparation

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Chat,
    Completion,
}

/// Everything needed to run one request, decided before it is queued.
struct Prepared {
    kind: Kind,
    id: String,
    created: u64,
    prompt_ids: Vec<u32>,
    /// the pictures of the prompt (engines with image input)
    images: Vec<ImageInput>,
    /// pictures of the request left out of the prompt (image overflow; `usage.prompt_tokens_details.images_omitted` when not 0)
    images_omitted: usize,
    /// snapshot points in prompt_ids (crate::snapshots)
    snapshots: Vec<usize>,
    params: SamplingParams,
    stops: Vec<String>,
    include_stop: bool,
    skip_special: bool,
    /// `None` for raw completions (no reasoning / tool parsing).
    parser: Option<ParserConfig>,
    stream: bool,
    include_usage: bool,
    /// Completions `echo`: the prompt text to put in front of the output.
    echo: Option<String>,
    /// the sampling this request runs with and where each value came from (client / profile / default): logged, and sent in the
    /// `x-sampling-effective` response header
    effective: Json,
    /// Engine::incoming's token, held from the request's arrival until it is submitted (start)
    incoming: std::sync::Mutex<Option<Box<dyn std::any::Any + Send + Sync>>>,
}

/// An f32 setting as JSON without the float32 tail (0.95, not 0.949999988079071).
fn jf(x: f32) -> Json {
    json!((x as f64 * 1e6).round() / 1e6)
}

#[derive(Debug, PartialEq)]
enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

fn parse_tool_choice(v: Option<&Json>) -> Result<ToolChoice, ApiError> {
    match v {
        None | Some(Json::Null) => Ok(ToolChoice::Auto),
        Some(Json::String(s)) => match s.as_str() {
            "auto" => Ok(ToolChoice::Auto),
            "none" => Ok(ToolChoice::None),
            "required" | "any" => Ok(ToolChoice::Required),
            other => Err(ApiError::bad_request(format!("unknown tool_choice '{other}'"))),
        },
        Some(obj @ Json::Object(_)) => obj
            .get("function")
            .and_then(|f| f.get("name"))
            .or_else(|| obj.get("name"))
            .and_then(Json::as_str)
            .map(|n| ToolChoice::Named(n.to_string()))
            .ok_or_else(|| ApiError::bad_request("tool_choice object needs function.name")),
        Some(_) => Err(ApiError::bad_request("tool_choice must be a string or an object")),
    }
}

/// Add an instruction to the system message (creating one if needed): the best-effort way to make
/// the model call a tool for `tool_choice: required` / a named function.
fn add_system_hint(messages: &mut Vec<Json>, hint: &str) {
    if let Some(first) = messages.first_mut() {
        if first.get("role").and_then(Json::as_str) == Some("system") {
            match first.get_mut("content") {
                Some(Json::String(s)) => {
                    s.push_str("\n\n");
                    s.push_str(hint);
                }
                Some(Json::Array(parts)) => parts.push(json!({"type": "text", "text": format!("\n\n{hint}")})),
                _ => {
                    first["content"] = Json::String(hint.to_string());
                }
            }
            return;
        }
    }
    messages.insert(0, json!({"role": "system", "content": hint}));
}

/// Thinking is "open" at the end of the prompt if the last assistant turn has a `<think>` without a
/// matching `</think>` after it.
fn prompt_opens_thinking(prompt: &str) -> bool {
    let tail = prompt.rfind("<|im_start|>").map(|i| &prompt[i..]).unwrap_or(prompt);
    match (tail.rfind("<think>"), tail.rfind("</think>")) {
        (Some(open), Some(close)) => open > close,
        (Some(_), None) => true,
        _ => false,
    }
}

fn resolve_sampling(
    s: &AppState,
    r: &SamplingRequest,
    prompt_len: usize,
    default_max: Option<usize>,
    // chat requests: does the reply think (picks the sampling profile); None: no profile (raw completions)
    thinking: Option<bool>,
) -> Result<(SamplingParams, Vec<String>, bool, bool, Json), ApiError> {
    let ctx = s.cfg.max_context;
    // the engine may keep the context's last positions for its own rows: such a prompt is the client's to shorten
    let reserve = s.engine.prompt_reserve();
    if prompt_len + reserve >= ctx {
        let room = if reserve > 0 { format!(" (a prompt may use at most {} of them)", ctx.saturating_sub(reserve + 1)) } else { String::new() };
        return Err(ApiError::bad_request(format!(
            "the prompt is {prompt_len} tokens; this model's context window is {ctx} tokens{room}"
        )));
    }
    let room = ctx - prompt_len;
    let requested = r.max_completion_tokens.or(r.max_tokens).map(|v| v as usize);
    if requested == Some(0) {
        return Err(ApiError::bad_request("max_tokens must be at least 1"));
    }
    let cap = requested.is_none().then(|| default_max.or(s.cfg.defaults.default_max_tokens)).flatten();
    let max_tokens = requested.or(cap).unwrap_or(room).min(room);
    let max_from = if requested.is_some() { "client" } else if cap.is_some() { "default" } else { "context" };
    if let Some(t) = r.temperature {
        if !(0.0..=100.0).contains(&t) {
            return Err(ApiError::bad_request("temperature must be >= 0"));
        }
    }
    if let Some(p) = r.top_p {
        if !(p > 0.0 && p <= 1.0) {
            return Err(ApiError::bad_request("top_p must be in (0, 1]"));
        }
    }
    if let Some(p) = r.min_p {
        if !(0.0..=1.0).contains(&p) {
            return Err(ApiError::bad_request("min_p must be in [0, 1]"));
        }
    }
    if let Some(p) = r.repetition_penalty {
        if !(p > 0.0 && p.is_finite()) {
            return Err(ApiError::bad_request("repetition_penalty must be > 0"));
        }
    }
    for (name, v) in [("presence_penalty", r.presence_penalty), ("frequency_penalty", r.frequency_penalty)] {
        if v.is_some_and(|v| !v.is_finite()) {
            return Err(ApiError::bad_request(format!("{name} must be a finite number")));
        }
    }
    let d = &s.cfg.defaults;
    let mut stop_ids: Vec<u32> = if r.ignore_eos.unwrap_or(false) { Vec::new() } else { d.eos_ids.clone() };
    for id in r.stop_token_ids.iter().flatten() {
        if !stop_ids.contains(id) {
            stop_ids.push(*id);
        }
    }
    // every field: the client's value if it sent one, else the profile of the request's mode, else the model default
    let none = Profile::default();
    let prof = match thinking {
        Some(true) => &d.phase.profile_thinking,
        Some(false) => &d.phase.profile_plain,
        None => &none,
    };
    let mut from = Map::new();
    // greedy_keeps_fast_path: the client sent temperature 0 itself (greedy, as below). The profile's penalties are then not
    // added: a penalty the client does not send stays at the model default, and the header names why ("greedy_fast_path").
    // top_p / top_k / min_p still come from the profile: no engine reads them at temperature 0 (Sampler::is_plain_greedy).
    let keep_greedy = d.phase.greedy_keeps_fast_path && thinking.is_some() && r.temperature.is_some_and(|t| t < 1e-5);
    let mut pick_from = |name: &str, client: Option<f64>, profile: Option<f32>, default: f32, withheld: bool| -> f32 {
        let (v, src) = match (client, profile) {
            (Some(c), _) => (c as f32, "client"),
            (None, Some(_)) if withheld => (default, "greedy_fast_path"),
            (None, Some(p)) => (p, "profile"),
            (None, None) => (default, "default"),
        };
        from.insert(name.to_string(), json!(src));
        v
    };
    let mut pick = |name: &str, client: Option<f64>, profile: Option<f32>, default: f32| pick_from(name, client, profile, default, false);
    let temperature = match pick("temperature", r.temperature, prof.temperature, d.temperature) {
        t if t < 1e-5 => 0.0, // greedy, as vLLM treats temperatures below 1e-5
        t => t,
    };
    let top_p = pick("top_p", r.top_p, prof.top_p, d.top_p);
    let top_k = pick("top_k", r.top_k.map(|k| k.max(0) as f64), prof.top_k.map(|k| k as f32), d.top_k as f32) as usize;
    let min_p = pick("min_p", r.min_p, prof.min_p, d.min_p);
    let presence_penalty = pick_from("presence_penalty", r.presence_penalty, prof.presence_penalty, 0.0, keep_greedy);
    let frequency_penalty = pick_from("frequency_penalty", r.frequency_penalty, prof.frequency_penalty, 0.0, keep_greedy);
    let repetition_penalty = pick_from("repetition_penalty", r.repetition_penalty, prof.repetition_penalty, d.repetition_penalty, keep_greedy);
    let params = SamplingParams {
        max_tokens,
        min_tokens: (r.min_tokens.unwrap_or(0) as usize).min(max_tokens),
        temperature,
        top_p,
        top_k,
        min_p,
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
        seed: r.seed.map(|v| v as u64),
        stop_token_ids: stop_ids,
        ..Default::default()
    };
    from.insert("max_tokens".to_string(), json!(max_from));
    let effective = json!({
        "max_tokens": max_tokens,
        "mode": match thinking { Some(true) => "thinking", Some(false) => "non_thinking", None => "none" },
        "temperature": jf(temperature), "top_p": jf(top_p), "top_k": top_k, "min_p": jf(min_p),
        "presence_penalty": jf(presence_penalty), "frequency_penalty": jf(frequency_penalty), "repetition_penalty": jf(repetition_penalty),
        "from": from,
        "greedy_keeps_fast_path": d.phase.greedy_keeps_fast_path,
    });
    let stops = r.stop.clone().map(|s| s.into_vec()).unwrap_or_default();
    Ok((params, stops, r.include_stop_str_in_output.unwrap_or(false), r.skip_special_tokens.unwrap_or(true), effective))
}

/// The text an image part becomes when its picture is left out of the prompt (see [`take_pictures`]).
#[cfg(feature = "ink")]
const PICTURE_OMITTED: &str = "[earlier picture omitted]";

/// The `data:` URL of an image part (`image_url` with a `data:` URL; also `input_image` and `image` parts that carry one); None for
/// a part that is not an image part.
#[cfg(feature = "ink")]
fn image_part_url<'a>(p: &'a Json, at: &str) -> Result<Option<&'a str>, String> {
    let ty = p.get("type").and_then(Json::as_str).unwrap_or("");
    if !matches!(ty, "image_url" | "input_image" | "image") {
        return Ok(None);
    }
    let url = match p.get("image_url") {
        Some(Json::String(u)) => Some(u.as_str()),
        Some(Json::Object(o)) => o.get("url").and_then(Json::as_str),
        _ => None,
    }
    .or_else(|| p.get("url").and_then(Json::as_str));
    match url {
        Some(u) => Ok(Some(u)),
        None => Err(format!("{at}: an image part needs image_url.url with a data: URL (data:image/png;base64,... or data:image/jpeg;base64,...)")),
    }
}

/// Image input: decode the pictures of the image parts, in the order the template will place them, and leave
/// `{"type": "image_url"}` in each part's place. The limits of `lim` apply. A request with more pictures than `lim.max_images` or
/// more image tokens than `lim.max_tokens` is not refused (unless `lim.drop_oldest` is off): its oldest pictures (message order,
/// then part order) are left out, each image part becoming the text part "[earlier picture omitted]". How many:
/// `fe_ink::images::omit_count`, in whole batches, so that the prompt in front of the kept pictures stays the same from turn to
/// turn and the prompt cache resumes it. Returns the pictures kept and the number left out.
#[cfg(feature = "ink")]
fn take_pictures(messages: &mut [Json], lim: &fe_ink::images::Limits) -> Result<(Vec<fe_ink::images::Picture>, usize), String> {
    let mut pics = Vec::new();
    if !lim.drop_oldest {
        // INK_IMAGE_OVERFLOW=reject: a request beyond the limits gets HTTP 400
        let mut tokens = 0usize;
        for (i, m) in messages.iter_mut().enumerate() {
            let Some(Json::Array(parts)) = m.get_mut("content") else { continue };
            for (j, p) in parts.iter_mut().enumerate() {
                let at = format!("messages[{i}].content[{j}]");
                let Some(url) = image_part_url(p, &at)? else { continue };
                if pics.len() >= lim.max_images {
                    return Err(format!("{at}: more than {} pictures in one request (INK_IMAGE_MAX)", lim.max_images));
                }
                let bytes = fe_ink::images::data_url_bytes(url, lim.max_bytes).map_err(|e| format!("{at}: {e}"))?;
                let pic = fe_ink::images::intake(&bytes, lim).map_err(|e| format!("{at}: {e}"))?;
                tokens += pic.tokens();
                if tokens > lim.max_tokens {
                    return Err(format!("{at}: the pictures of this request are more than {} image tokens (one per 40 x 40 pixels; INK_IMAGE_TOKENS). Send fewer or smaller pictures.", lim.max_tokens));
                }
                pics.push(pic);
                *p = json!({"type": "image_url"});
            }
        }
        return Ok((pics, 0));
    }
    // every picture file and its image tokens (from the file header; nothing is decoded yet), oldest first
    let mut files: Vec<Vec<u8>> = Vec::new();
    let mut tokens: Vec<usize> = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let Some(Json::Array(parts)) = m.get("content") else { continue };
        for (j, p) in parts.iter().enumerate() {
            let at = format!("messages[{i}].content[{j}]");
            let Some(url) = image_part_url(p, &at)? else { continue };
            let bytes = fe_ink::images::data_url_bytes(url, lim.max_bytes).map_err(|e| format!("{at}: {e}"))?;
            tokens.push(fe_ink::images::header_tokens(&bytes, lim).map_err(|e| format!("{at}: {e}"))?);
            files.push(bytes);
        }
    }
    let omit = fe_ink::images::omit_count(&tokens, lim);
    let mut n = 0usize;
    for (i, m) in messages.iter_mut().enumerate() {
        let Some(Json::Array(parts)) = m.get_mut("content") else { continue };
        for (j, p) in parts.iter_mut().enumerate() {
            let at = format!("messages[{i}].content[{j}]");
            if image_part_url(p, &at)?.is_none() {
                continue;
            }
            if n < omit {
                *p = json!({"type": "text", "text": PICTURE_OMITTED});
            } else {
                pics.push(fe_ink::images::intake(&files[n], lim).map_err(|e| format!("{at}: {e}"))?);
                *p = json!({"type": "image_url"});
            }
            files[n] = Vec::new();
            n += 1;
        }
    }
    if omit > 0 {
        let b = lim.overflow_batch();
        let kept: usize = tokens[omit..].iter().sum();
        if omit % b != 0 {
            tracing::warn!("image overflow: the limits (INK_IMAGE_MAX {}, INK_IMAGE_TOKENS {}) are smaller than one batch of this request's pictures; the count left out is not a multiple of the batch", lim.max_images, lim.max_tokens);
        }
        tracing::info!("image overflow: {omit} of {n} pictures left out, oldest first (k = {omit}, batch B = {b}); kept {} pictures, {kept} image tokens (INK_IMAGE_MAX {}, INK_IMAGE_TOKENS {})",
            n - omit, lim.max_images, lim.max_tokens);
    }
    Ok((pics, omit))
}

/// Image input: every placeholder token the template wrote (one a picture, in order) becomes the picture's image tokens, which carry
/// the picture's id (fe_ink::images) so that the prompt cache tells pictures apart.
#[cfg(feature = "ink")]
fn place_pictures(ids: Vec<u32>, pics: Vec<fe_ink::images::Picture>) -> Result<(Vec<u32>, Vec<ImageInput>), ApiError> {
    let ph = fe_ink::vision::IMAGE_TOKEN;
    let n = ids.iter().filter(|&&t| t == ph).count();
    if n != pics.len() {
        return Err(ApiError::bad_request(format!("the chat template placed {n} of the {} pictures of this request (pictures belong in user and tool messages)", pics.len())));
    }
    let mut out = Vec::with_capacity(ids.len() + pics.iter().map(|p| p.tokens()).sum::<usize>());
    let mut images = Vec::with_capacity(pics.len());
    let mut pics = pics.into_iter();
    for t in ids {
        if t != ph {
            out.push(t);
            continue;
        }
        let p = pics.next().expect("counted above");
        let (pos, id, tokens) = (out.len(), p.id(), p.tokens());
        out.resize(pos + tokens, id);
        images.push(ImageInput { pos, key: p.key, width: p.width, height: p.height, rgb: Arc::new(p.rgb) });
    }
    Ok((out, images))
}

fn prepare_chat(s: &AppState, req: ChatRequest) -> Result<Prepared, ApiError> {
    #[cfg(feature = "ink")]
    let mut req = req;
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must be a non-empty list"));
    }
    if req.n.unwrap_or(1) > 1 {
        return Err(ApiError::bad_request("n > 1 is not supported"));
    }
    if wants_logprobs(&req.logprobs) || wants_logprobs(&req.top_logprobs) {
        return Err(ApiError::bad_request("logprobs are not supported"));
    }
    #[cfg(feature = "ink")]
    let (pictures, images_omitted) = match &s.image_limits {
        Some(lim) => take_pictures(&mut req.messages, lim).map_err(ApiError::bad_request)?,
        None => (Vec::new(), 0),
    };
    #[cfg(not(feature = "ink"))]
    let images_omitted = 0usize;
    let mut messages = normalize_messages_with(&req.messages, s.accepts_images()).map_err(ApiError::bad_request)?;
    let tools: Vec<Json> = req.tools.clone().unwrap_or_default();
    let choice = parse_tool_choice(req.tool_choice.as_ref())?;
    let template_tools: Option<Vec<Json>> = match &choice {
        _ if tools.is_empty() => {
            if matches!(choice, ToolChoice::Required | ToolChoice::Named(_)) {
                return Err(ApiError::bad_request("tool_choice requires tools"));
            }
            None
        }
        ToolChoice::None => None,
        ToolChoice::Auto => Some(tools.clone()),
        ToolChoice::Required => {
            add_system_hint(&mut messages, "You must call at least one of the provided functions in this reply.");
            Some(tools.clone())
        }
        ToolChoice::Named(name) => {
            let chosen: Vec<Json> = tools
                .iter()
                .filter(|t| t.get("function").unwrap_or(t).get("name").and_then(Json::as_str) == Some(name.as_str()))
                .cloned()
                .collect();
            if chosen.is_empty() {
                return Err(ApiError::bad_request(format!("tool_choice names '{name}', which is not in tools")));
            }
            add_system_hint(&mut messages, &format!("You must call the function {name} in this reply."));
            Some(chosen)
        }
    };

    let mut kwargs: Map<String, Json> = req.chat_template_kwargs.clone().unwrap_or_default();
    if let Some(on) = req.enable_thinking {
        kwargs.entry("enable_thinking").or_insert(Json::Bool(on));
    }
    // Inkling-2: the template reads `reasoning_effort` (names none..max, or a number 0..0.99; default 0.9); thinking on / off switches map onto it
    // (off = "none", which the model card says still reasons a little), and the reply is split by `crate::inkling`, not by a `<think>` convention
    let ink = s.template.inkling;
    if ink {
        let off = matches!(kwargs.get("enable_thinking"), Some(Json::Bool(false)));
        kwargs.remove("enable_thinking");
        if !kwargs.contains_key("reasoning_effort") {
            if let Some(e) = &req.reasoning_effort {
                let e = e.trim().to_lowercase();
                let v = e.parse::<f64>().map(|f| json!(f)).unwrap_or_else(|_| json!(e));
                kwargs.insert("reasoning_effort".into(), v);
            } else if off {
                kwargs.insert("reasoning_effort".into(), json!("none"));
            } else if let Some(d) = &s.cfg.defaults.default_reasoning_effort {
                // the launcher's default (FE_DEFAULT_REASONING_EFFORT); without one the template's own 0.9 applies
                kwargs.insert("reasoning_effort".into(), d.clone());
            }
        }
    }
    // tool_choice required / named: the reply IS the call. The generation prompt is prefilled with the call opener (below), which only
    // works outside a reasoning block: such a reply does not think.
    let continue_final = req.continue_final_message.unwrap_or(false);
    let add_gen = req.add_generation_prompt.unwrap_or(true) && !continue_final;
    let offered = ToolSet::from_openai(template_tools.as_deref().unwrap_or(&[]));
    let mut forced: Option<ForcedCall> = match &choice {
        _ if ink => None,
        _ if template_tools.is_none() || !add_gen => None,
        ToolChoice::Required => {
            // one function offered: that is the call; several: the name is the model's choice among them (constrained below)
            let names: Vec<&str> = offered.names().collect();
            if names.len() == 1 { Some(ForcedCall::Named(names[0].to_string())) } else { Some(ForcedCall::Any) }
        }
        ToolChoice::Named(n) => Some(ForcedCall::Named(n.clone())),
        _ => None,
    };
    if forced.is_some() {
        kwargs.insert("enable_thinking".into(), Json::Bool(false));
    }
    if let Some(effort) = req.reasoning_effort.as_ref().filter(|_| !ink) {
        match effort.to_lowercase().as_str() {
            "none" => {
                kwargs.entry("enable_thinking").or_insert(Json::Bool(false));
            }
            e if !kwargs.contains_key("reasoning_effort") => {
                let mapped = match e {
                    "minimal" | "low" => "low",
                    "medium" => "medium",
                    _ => "xhigh",
                };
                kwargs.insert("reasoning_effort".into(), json!(mapped));
            }
            _ => {}
        }
    }
    let mut template_tools = template_tools;
    if ink {
        // text in messages and tool definitions must not be able to spell the model's control tokens (see crate::inkling::ESC)
        let is_markup = |t: &str| s.tok.is_markup(t);
        for m in messages.iter_mut() {
            crate::inkling::escape_json(m, &is_markup);
        }
        if let Some(ts) = template_tools.as_mut() {
            for t in ts.iter_mut() {
                crate::inkling::escape_json(t, &is_markup);
            }
        }
    }
    let mut prompt = s
        .template
        .render(&messages, template_tools.as_deref(), add_gen, &kwargs)
        .map_err(|e| ApiError::bad_request(e.0))?;
    if continue_final {
        let trimmed = prompt.trim_end_matches('\n');
        if let Some(t) = trimmed.strip_suffix("<|im_end|>") {
            prompt = t.to_string();
        }
    }
    let thinking = if ink {
        !matches!(kwargs.get("reasoning_effort"), Some(Json::String(e)) if e == "none") && !matches!(kwargs.get("reasoning_effort"), Some(Json::Number(n)) if n.as_f64() == Some(0.0))
    } else {
        prompt_opens_thinking(&prompt)
    };
    if thinking {
        // a template that opens a reasoning block regardless: the system hint is all there is
        forced = None;
    }
    if let Some(f) = &forced {
        prompt.push_str(&f.opener_for(s.template.dsml));
    }
    if tracing::enabled!(tracing::Level::DEBUG) {
        let tail: String = prompt.chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect();
        tracing::debug!("rendered prompt ({} chars, thinking {thinking}), tail: {tail:?}", prompt.len());
    }
    let prompt_ids = if ink && prompt.contains(crate::inkling::ESC) {
        // pieces: the template's own control tokens are parsed, the disarmed spellings in message text are not
        let mut ids = Vec::new();
        for piece in prompt.split(crate::inkling::ESC) {
            ids.extend(s.tok.encode(piece).map_err(|e| ApiError::internal(e.to_string()))?);
        }
        ids
    } else {
        s.tok.encode(&prompt).map_err(|e| ApiError::internal(e.to_string()))?
    };
    #[cfg(feature = "ink")]
    let (prompt_ids, images) = if pictures.is_empty() { (prompt_ids, Vec::new()) } else { place_pictures(prompt_ids, pictures)? };
    #[cfg(not(feature = "ink"))]
    let images: Vec<ImageInput> = Vec::new();
    if !images.is_empty() {
        let sizes: Vec<String> = images.iter().map(|i| format!("{} x {}", i.width, i.height)).collect();
        let tokens: usize = images.iter().map(|i| i.width.div_ceil(40) * i.height.div_ceil(40)).sum();
        tracing::info!("chat request with {} picture(s) ({}): {tokens} image tokens of {} prompt tokens", images.len(), sizes.join(", "), prompt_ids.len());
    }
    let (mut params, stops, include_stop, skip_special, mut effective) = resolve_sampling(s, &req.sampling, prompt_ids.len(), None, Some(thinking))?;
    let ph = &s.cfg.defaults.phase;
    // the loop breaker watches every chat reply (answer and tool-call rules); the reasoning rule and cap need the block's end token
    params.loop_breaker = ph.loop_breaker;
    params.loop_rules = ph.loop_rules.clone();
    params.call_marks = ph.call_marks.clone();
    params.call_open = forced.is_some();
    if forced == Some(ForcedCall::Any) {
        // the reply continues `<function=`: only an offered name may follow
        // (DSML: the reply continues `<｜DSML｜ invoke name="`)
        let close = if s.template.dsml { "\">\n" } else { ">\n" };
        params.prefix_choices = offered.names().filter_map(|n| s.tok.encode(&format!("{n}{close}")).ok()).filter(|c| !c.is_empty()).collect();
    }
    if thinking {
        if let Some(te) = ph.think_end {
            params.think_end = Some(te);
            params.max_reasoning = ph.max_reasoning;
            // greedy_floor: a greedy request (temperature 0) thinks at the thinking profile's temperature; its answer / tool call stays greedy
            if ph.greedy_floor && params.temperature <= 0.0 {
                if let Some(t) = ph.profile_thinking.temperature.filter(|&t| t > 0.0) {
                    let mut r = params.clone();
                    r.temperature = t;
                    r.think_end = None;
                    r.reasoning = None;
                    params.reasoning = Some(Box::new(r));
                    effective["reasoning_temperature"] = jf(t);
                }
            }
            effective["max_reasoning_tokens"] = json!(ph.max_reasoning);
        }
    }
    if ink {
        effective["reasoning_effort"] = kwargs.get("reasoning_effort").cloned().unwrap_or_else(|| json!("template default 0.9"));
    }
    effective["greedy_floor"] = json!(ph.greedy_floor);
    effective["loop_breaker"] = json!(ph.loop_breaker);
    let parser = ParserConfig {
        thinking: thinking && !ink,
        tools: template_tools.as_ref().map(|t| ToolSet::from_openai(t)),
        max_tool_calls: (req.parallel_tool_calls == Some(false)).then_some(1),
        dsml: s.template.dsml,
        forced,
        inkling: ink,
    };
    let stream = req.stream.unwrap_or(false);
    Ok(Prepared {
        kind: Kind::Chat,
        id: random_id("chatcmpl-", 24),
        created: unix_now(),
        snapshots: s.marks.points(&prompt_ids),
        prompt_ids,
        images,
        images_omitted,
        params,
        stops,
        include_stop,
        skip_special,
        parser: Some(parser),
        stream,
        include_usage: stream && req.stream_options.and_then(|o| o.include_usage).unwrap_or(false),
        echo: None,
        effective,
        incoming: std::sync::Mutex::new(None),
    })
}

fn prepare_completion(s: &AppState, req: CompletionRequest) -> Result<Prepared, ApiError> {
    if req.n.unwrap_or(1) > 1 {
        return Err(ApiError::bad_request("n > 1 is not supported"));
    }
    if req.suffix.as_deref().is_some_and(|x| !x.is_empty()) {
        return Err(ApiError::bad_request("suffix is not supported"));
    }
    if wants_logprobs(&req.logprobs) {
        return Err(ApiError::bad_request("logprobs are not supported"));
    }
    let ids_of = |a: &[Json]| -> Option<Vec<u32>> { a.iter().map(|v| v.as_u64().map(|x| x as u32)).collect() };
    let (text, prompt_ids) = match req.prompt.as_ref() {
        Some(Json::String(p)) => (p.clone(), None),
        Some(Json::Array(a)) if a.len() == 1 && a[0].is_string() => (a[0].as_str().unwrap_or("").to_string(), None),
        Some(Json::Array(a)) if a.len() == 1 && a[0].is_array() => {
            let ids = ids_of(a[0].as_array().map(Vec::as_slice).unwrap_or(&[]))
                .ok_or_else(|| ApiError::bad_request("prompt token ids must be integers"))?;
            (String::new(), Some(ids))
        }
        Some(Json::Array(a)) if !a.is_empty() && a.iter().all(Json::is_number) => {
            let ids = ids_of(a).ok_or_else(|| ApiError::bad_request("prompt token ids must be integers"))?;
            (String::new(), Some(ids))
        }
        Some(Json::Array(_)) => return Err(ApiError::bad_request("only one prompt per request is supported")),
        _ => return Err(ApiError::bad_request("prompt is required (a string or a list of token ids)")),
    };
    let prompt_ids = match prompt_ids {
        Some(ids) => ids,
        None => s.tok.encode(&text).map_err(|e| ApiError::internal(e.to_string()))?,
    };
    if prompt_ids.is_empty() {
        return Err(ApiError::bad_request("prompt is empty"));
    }
    let vocab = s.tok.vocab_size();
    if let Some(bad) = prompt_ids.iter().find(|&&t| t as usize >= vocab) {
        return Err(ApiError::bad_request(format!("token id {bad} is out of range")));
    }
    let (params, stops, include_stop, skip_special, effective) = resolve_sampling(s, &req.sampling, prompt_ids.len(), Some(16), None)?;
    let echo = if req.echo.unwrap_or(false) {
        Some(if text.is_empty() { s.tok.decode(&prompt_ids, skip_special).unwrap_or_default() } else { text })
    } else {
        None
    };
    let stream = req.stream.unwrap_or(false);
    Ok(Prepared {
        kind: Kind::Completion,
        id: random_id("cmpl-", 24),
        created: unix_now(),
        snapshots: s.marks.points(&prompt_ids),
        prompt_ids,
        images: Vec::new(),
        images_omitted: 0,
        params,
        stops,
        include_stop,
        skip_special,
        parser: None,
        stream,
        include_usage: stream && req.stream_options.and_then(|o| o.include_usage).unwrap_or(false),
        echo,
        effective,
        incoming: std::sync::Mutex::new(None),
    })
}

// ---------------------------------------------------------------------------------------------
// running a generation

/// Counts a waiting request while it holds this guard.
struct Counted(Arc<AtomicUsize>);

impl Counted {
    fn new(c: &Arc<AtomicUsize>) -> Self {
        c.fetch_add(1, Ordering::Relaxed);
        Self(c.clone())
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A concurrency slot, owned by the engine's `TokenSink` until the engine lets go of the request.
struct Slot {
    _permit: OwnedSemaphorePermit,
    _running: Counted,
}

/// Engine output for one request; dropping it cancels the generation.
struct Running {
    rx: mpsc::UnboundedReceiver<GenEvent>,
    cancel: CancelToken,
}

impl Drop for Running {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

async fn start(s: &Arc<AppState>, p: &Prepared) -> Running {
    let waiting = Counted::new(&s.queued);
    let permit = s.slots.clone().acquire_owned().await.expect("slot semaphore is never closed");
    drop(waiting);
    let slot = Slot { _permit: permit, _running: Counted::new(&s.running) };
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancelToken::new();
    let req = GenRequest { id: p.id.clone(), prompt_ids: p.prompt_ids.clone(), snapshots: p.snapshots.clone(),
        params: p.params.clone(), cancel: cancel.clone(), images: p.images.clone() };
    // given back before the submit: the worker this request wakes must not wait for the request itself
    drop(p.incoming.lock().unwrap_or_else(|e| e.into_inner()).take());
    s.engine.submit(req, TokenSink::with_slot(tx, Box::new(slot)));
    Running { rx, cancel }
}

/// Token ids → text (UTF-8 safe) → stop strings → reasoning / content / tool-call deltas.
struct Pipeline {
    dec: IncrementalDecoder,
    stop: StopMatcher,
    parser: Option<OutputParser>,
    /// the Inkling reply markup: its special tokens go to the parser as sentinel characters (`crate::inkling::sentinel`)
    ink: bool,
    /// RUST_LOG=fe_server=debug: the raw reply with the markup spelled out, logged at the end
    trace: Option<String>,
    max_tokens: usize,
    completion_tokens: usize,
    /// Set when the server ends the request itself (stop string or max_tokens).
    server_finish: Option<&'static str>,
    closed: bool,
    /// tokens generated while inside the reasoning block / the cap on them (FE_MAX_REASONING_TOKENS, 0 = no cap)
    reasoning_tokens: usize,
    max_reasoning: usize,
    /// the server-side reasoning backstop ended the request
    capped: bool,
}

impl Pipeline {
    fn new(p: &Prepared) -> Self {
        Self {
            dec: IncrementalDecoder::new(p.skip_special),
            stop: StopMatcher::new(p.stops.clone(), p.include_stop),
            parser: p.parser.clone().map(OutputParser::new),
            ink: p.parser.as_ref().is_some_and(|c| c.inkling),
            trace: tracing::enabled!(tracing::Level::DEBUG).then(String::new),
            max_tokens: p.params.max_tokens,
            completion_tokens: 0,
            server_finish: None,
            closed: false,
            reasoning_tokens: 0,
            capped: false,
            // the engine closes the block itself at max_reasoning (graceful); this is the backstop for engines that do not
            max_reasoning: if p.params.max_reasoning > 0 { p.params.max_reasoning * 2 + 1024 } else { std::env::var("FE_MAX_REASONING_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(16384) },
        }
    }

    fn parse(&mut self, text: &str) -> Vec<Delta> {
        match &mut self.parser {
            Some(p) => p.feed(text),
            None if text.is_empty() => Vec::new(),
            None => vec![Delta::Content(text.to_string())],
        }
    }

    fn tokens(&mut self, tok: &Tok, ids: &[u32]) -> Result<Vec<Delta>, String> {
        // token by token: a stop string or max_tokens that lands inside a batch (an accepted speculative-decoding block, a chunk of tokens)
        // cuts at exactly that token; the tokens after it are neither counted nor parsed
        let mut deltas = Vec::new();
        for &id in ids {
            if self.completion_tokens >= self.max_tokens {
                self.server_finish = Some("length");
                break;
            }
            self.completion_tokens += 1;
            let text = match tok.markup(id).and_then(crate::inkling::sentinel).filter(|_| self.ink) {
                Some(mark) => {
                    // a reply marker (never part of a character): what is pending is complete, then the marker
                    let mut t = self.dec.flush(tok).map_err(|e| e.to_string())?;
                    t.push_str(mark);
                    t
                }
                None => self.dec.push(tok, id).map_err(|e| e.to_string())?,
            };
            if let Some(t) = &mut self.trace {
                for c in text.chars() {
                    match c {
                        '\u{E000}' => t.push_str("<|message_model|>"),
                        '\u{E001}' => t.push_str("<|content_thinking|>"),
                        '\u{E002}' => t.push_str("<|content_text|>"),
                        '\u{E003}' => t.push_str("<|content_invoke_tool_json|>"),
                        '\u{E004}' => t.push_str("<|end_message|>"),
                        '\u{E005}' => t.push_str("<|content_model_end_sampling|>"),
                        '\u{E006}' => t.push_str("<|content_other|>"),
                        c => t.push(c),
                    }
                }
            }
            let (emit, hit) = self.stop.push(&text);
            let was = self.parser.as_ref().is_some_and(|p| p.in_reasoning());
            deltas.extend(self.parse(&emit));
            if self.parser.as_ref().is_some_and(|p| p.halted()) && self.server_finish.is_none() {
                self.server_finish = Some("stop");
                if hit {
                    break;
                }
            }
            if was || self.parser.as_ref().is_some_and(|p| p.in_reasoning()) {
                self.reasoning_tokens += 1;
            }
            if hit {
                self.server_finish = Some("stop");
                break;
            }
            if self.server_finish.is_some() {
                break;
            }
        }
        if self.completion_tokens >= self.max_tokens && self.server_finish.is_none() {
            self.server_finish = Some("length");
        }
        if self.parser.as_ref().is_some_and(|p| p.in_reasoning()) {
            if self.max_reasoning > 0 && self.reasoning_tokens > self.max_reasoning && self.server_finish.is_none() {
                // a reasoning runaway (a 48K-token reasoning block with no answer was seen): end the request as length instead of burning the whole budget
                tracing::warn!("reasoning cap reached ({} tokens): ending the request as length", self.reasoning_tokens);
                self.server_finish = Some("length");
                self.capped = true;
            }
        }
        if self.server_finish.is_some() {
            deltas.extend(self.close(tok)?);
        }
        Ok(deltas)
    }

    /// End of output: flush the decoder, the stop matcher and the parser.
    fn close(&mut self, tok: &Tok) -> Result<Vec<Delta>, String> {
        if self.closed {
            return Ok(Vec::new());
        }
        self.closed = true;
        let mut deltas = Vec::new();
        if !self.stop.stopped() {
            let tail = self.dec.flush(tok).map_err(|e| e.to_string())?;
            let (mut emit, hit) = self.stop.push(&tail);
            if hit {
                self.server_finish = Some("stop");
            } else {
                emit.push_str(&self.stop.flush());
            }
            deltas.extend(self.parse(&emit));
        }
        if let Some(p) = &mut self.parser {
            deltas.extend(p.finish());
        }
        if let Some(t) = &self.trace {
            tracing::debug!("raw reply ({} tokens): {t:?}", self.completion_tokens);
        }
        Ok(deltas)
    }

    fn tool_calls(&self) -> usize {
        self.parser.as_ref().map(|p| p.tool_calls()).unwrap_or(0)
    }

    fn malformed(&self) -> bool {
        self.parser.as_ref().is_some_and(|p| p.malformed())
    }
}

struct Outcome {
    finish_reason: &'static str,
    /// generated tokens that were inside the reasoning block (reported in `usage.completion_tokens_details`; they are part of completion_tokens)
    reasoning_tokens: usize,
    error: Option<String>,
    completion_tokens: usize,
    cached_prompt_tokens: usize,
    /// the engine's timings, if it reports them (`timings` in the response; absent otherwise)
    timings: Option<Timings>,
}

enum Step {
    Deltas(Vec<Delta>),
    Final(Vec<Delta>, Outcome),
}

/// Drives one request: engine events in, deltas out, request log line at the end.
struct Driver {
    state: Arc<AppState>,
    running: Running,
    pipe: Pipeline,
    id: String,
    prompt_tokens: usize,
    t_start: Instant,
    t_first: Option<Instant>,
    finished: bool,
    stream: bool,
    /// what the engine's reasoning cap / loop breaker did (from its `Finish`)
    events: crate::loopdet::PhaseEvents,
    /// when the request went to the engine / when its latest tokens arrived (the watchdog's clocks)
    t_engine: Instant,
    t_last: Option<Instant>,
    /// why the watchdog ended the request
    watchdog: Option<&'static str>,
}

impl Drop for Driver {
    fn drop(&mut self) {
        if !self.finished {
            tracing::info!(
                "{} cancelled: client disconnected after {} output tokens",
                self.id,
                self.pipe.completion_tokens
            );
            self.capture(None, Some("client disconnected"), self.pipe.completion_tokens, 0);
        }
    }
}

impl Driver {
    fn new(state: Arc<AppState>, running: Running, p: &Prepared, t_start: Instant) -> Self {
        Self {
            state,
            running,
            pipe: Pipeline::new(p),
            id: p.id.clone(),
            prompt_tokens: p.prompt_ids.len(),
            t_start,
            t_first: None,
            finished: false,
            stream: p.stream,
            events: Default::default(),
            t_engine: Instant::now(),
            t_last: None,
            watchdog: None,
        }
    }

    /// The request-capture sidecar (`<id>.meta.json`); a no-op unless FE_DUMP_REQUESTS is set.
    fn capture(&self, finish_reason: Option<&str>, error: Option<&str>, completion_tokens: usize, cached: usize) {
        if self.state.dump_dir.get().is_none() {
            return;
        }
        let ev = &self.events;
        self.state.capture_meta(&self.id, json!({
            "id": self.id,
            "finish_reason": finish_reason,
            "error": error,
            "stream": self.stream,
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": cached,
            "completion_tokens": completion_tokens,
            // counted by the server's parser (text inside the reasoning block) / by the engine (0 for engines without phase tracking)
            "reasoning_tokens": self.pipe.reasoning_tokens,
            "engine_reasoning_tokens": ev.reasoning_tokens,
            "tool_calls": self.pipe.tool_calls(),
            "malformed_tool_call": self.pipe.malformed(),
            // the engine closed the reasoning block at the cap (the request went on), or the server-side backstop ended the request
            "reasoning_cap_fired": ev.reasoning_capped || self.pipe.capped,
            "reasoning_cap_by": if ev.reasoning_capped { json!("engine") } else if self.pipe.capped { json!("server") } else { Json::Null },
            "loop_breaker_fired": ev.loop_breaks > 0,
            "loop_breaker_firings": ev.loop_breaks,
            "sentence_loop_firings": ev.sentence_breaks,
            "loop_rewinds": ev.rewinds,
            "watchdog": self.watchdog,
            "seconds": (self.t_start.elapsed().as_secs_f64() * 1000.0).round() / 1000.0,
        }));
    }

    async fn next(&mut self) -> Step {
        let tok = self.state.tok.clone();
        // watchdog: no token for no_token_timeout seconds since the last accepted one (from the first token on: a prompt has the
        // keep-alive and the engine's own step watchdog), or the reply's wall-clock limit (from the hand-over to the engine)
        let (quiet, wall) = { let ph = &self.state.cfg.defaults.phase; (ph.no_token_timeout, ph.reply_wall_limit) };
        let mut limit: Option<(Instant, &'static str)> = None;
        if let (true, Some(t)) = (quiet > 0.0, self.t_last) {
            limit = Some((t + std::time::Duration::from_secs_f64(quiet), "no token within the no-token timeout"));
        }
        if wall > 0.0 {
            let at = self.t_engine + std::time::Duration::from_secs_f64(wall);
            if limit.is_none_or(|l| at < l.0) {
                limit = Some((at, "the reply reached its wall-clock limit"));
            }
        }
        let ev = match limit {
            None => self.running.rx.recv().await,
            Some((at, why)) => match tokio::time::timeout_at(tokio::time::Instant::from_std(at), self.running.rx.recv()).await {
                Ok(ev) => ev,
                Err(_) => {
                    tracing::warn!("watchdog: {}: {why} ({} s since the last token, {} s in all): ending the request after {} output tokens",
                        self.id, self.t_last.map_or(0, |t| t.elapsed().as_secs()), self.t_engine.elapsed().as_secs(), self.pipe.completion_tokens);
                    self.watchdog = Some(why);
                    self.running.cancel.cancel();
                    let (cached, timings) = self.engine_finish().await;
                    let d = self.pipe.close(&tok).unwrap_or_default();
                    return Step::Final(d, self.outcome("length", None, 0, cached).with_timings(timings));
                }
            },
        };
        match ev {
            Some(GenEvent::Tokens(ids)) => {
                self.t_last = Some(Instant::now());
                // an empty event is the engine's keep-alive ping during a long prefill, not the first token
                if self.t_first.is_none() && !ids.is_empty() {
                    self.t_first = self.t_last;
                }
                match self.pipe.tokens(&tok, &ids) {
                    Ok(d) if self.pipe.server_finish.is_none() => Step::Deltas(d),
                    Ok(d) => {
                        // the server ended it (stop string / max_tokens): stop the engine, and take its
                        // final event for the prompt-cache count
                        self.running.cancel.cancel();
                        let (cached, timings) = self.engine_finish().await;
                        let reason = self.pipe.server_finish.unwrap_or("stop");
                        Step::Final(d, self.outcome(reason, None, 0, cached).with_timings(timings))
                    }
                    Err(e) => {
                        self.running.cancel.cancel();
                        Step::Final(Vec::new(), self.outcome("stop", Some(e), 0, 0))
                    }
                }
            }
            Some(GenEvent::Finished(f)) => {
                self.events = f.events.clone();
                let d = self.pipe.close(&tok).unwrap_or_default();
                let (reason, err, extra) = match (self.pipe.server_finish, &f.reason) {
                    (Some(r), _) => (r, None, 0),
                    (None, FinishReason::Stop(_)) => ("stop", None, 1), // the stop token counts as output
                    (None, FinishReason::Length) => ("length", None, 0),
                    (None, FinishReason::Cancelled) => ("stop", None, 0),
                    (None, FinishReason::Error(e)) => ("stop", Some(e.clone()), 0),
                };
                Step::Final(d, self.outcome(reason, err, extra, f.cached_prompt_tokens).with_timings(f.timings.clone()))
            }
            None => {
                let d = self.pipe.close(&tok).unwrap_or_default();
                Step::Final(d, self.outcome("stop", Some("the engine dropped the request".into()), 0, 0))
            }
        }
    }

    /// Wait (briefly) for the engine's `Finished` after a server-side stop; returns its cached-prompt count and timings.
    async fn engine_finish(&mut self) -> (usize, Option<Timings>) {
        let deadline = tokio::time::sleep(std::time::Duration::from_secs(2));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                ev = self.running.rx.recv() => match ev {
                    Some(GenEvent::Finished(f)) => {
                        self.events = f.events;
                        return (f.cached_prompt_tokens, f.timings);
                    }
                    Some(GenEvent::Tokens(_)) => continue,
                    None => return (0, None),
                },
                _ = &mut deadline => return (0, None),
            }
        }
    }

    fn outcome(&mut self, reason: &'static str, error: Option<String>, extra: usize, cached: usize) -> Outcome {
        self.finished = true;
        let reason = if reason == "stop" && self.pipe.tool_calls() > 0 { "tool_calls" } else { reason };
        let completion_tokens = self.pipe.completion_tokens + extra;
        if self.pipe.malformed() {
            tracing::warn!("{} malformed tool call: DSML closing tags with no call block (answer passed as content, {} tokens)", self.id, completion_tokens);
        }
        let now = Instant::now();
        let ttft = self.t_first.map(|t| t - self.t_start).unwrap_or(now - self.t_start);
        let gen_s = self.t_first.map(|t| (now - t).as_secs_f64()).unwrap_or(0.0);
        let tps = if gen_s > 0.0 { (completion_tokens.saturating_sub(1)) as f64 / gen_s } else { 0.0 };
        match &error {
            Some(e) => tracing::warn!("{} failed after {} tokens: {e}", self.id, completion_tokens),
            None => tracing::info!(
                "{} done: prompt {} (cached {cached}), output {} [{reason}], first token {:.3}s, {:.1} tok/s",
                self.id,
                self.prompt_tokens,
                completion_tokens,
                ttft.as_secs_f64(),
                tps
            ),
        }
        self.capture(Some(reason), error.as_deref(), completion_tokens, cached);
        Outcome { finish_reason: reason, reasoning_tokens: self.pipe.reasoning_tokens, error, completion_tokens, cached_prompt_tokens: cached, timings: None }
    }
}

impl Outcome {
    fn with_timings(mut self, t: Option<Timings>) -> Outcome {
        self.timings = t;
        self
    }
}

// ---------------------------------------------------------------------------------------------
// output formatting

/// `images_omitted`: pictures of the request left out of the prompt (image overflow); reported as
/// `prompt_tokens_details.images_omitted` when not 0 (no such key otherwise: the JSON of every other request is unchanged).
fn usage(prompt: usize, out: &Outcome, images_omitted: usize) -> Json {
    let mut u = json!({
        "prompt_tokens": prompt,
        "completion_tokens": out.completion_tokens,
        "total_tokens": prompt + out.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": out.cached_prompt_tokens},
        "completion_tokens_details": {"reasoning_tokens": out.reasoning_tokens},
    });
    if images_omitted > 0 {
        u["prompt_tokens_details"]["images_omitted"] = json!(images_omitted);
    }
    u
}

fn delta_json(d: &Delta) -> Json {
    match d {
        Delta::Reasoning(s) => json!({"reasoning_content": s}),
        Delta::Content(s) => json!({"content": s}),
        Delta::ToolStart { index, id, name } => json!({"tool_calls": [{
            "index": index, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}]}),
        Delta::ToolArgs { index, arguments } => json!({"tool_calls": [{
            "index": index, "function": {"arguments": arguments}}]}),
    }
}

impl Prepared {
    fn object(&self, stream: bool) -> &'static str {
        match (self.kind, stream) {
            (Kind::Chat, true) => "chat.completion.chunk",
            (Kind::Chat, false) => "chat.completion",
            (Kind::Completion, _) => "text_completion",
        }
    }

    /// One streamed chunk carrying `choice` (or no choices, for the usage chunk).
    fn chunk(&self, model: &str, choice: Option<Json>, usage: Option<Json>) -> String {
        self.chunk_t(model, choice, usage, None)
    }

    /// A chunk, with the engine's `timings` when it reports them (no key otherwise: the JSON is then exactly `chunk`'s).
    fn chunk_t(&self, model: &str, choice: Option<Json>, usage: Option<Json>, timings: Option<&Timings>) -> String {
        let mut c = json!({
            "id": self.id,
            "object": self.object(true),
            "created": self.created,
            "model": model,
            "choices": choice.map(|c| vec![c]).unwrap_or_default(),
        });
        if let Some(u) = usage {
            c["usage"] = u;
        } else if self.include_usage {
            c["usage"] = Json::Null;
        }
        if let Some(t) = timings {
            c["timings"] = t.to_json();
        }
        c.to_string()
    }

    fn delta_chunk(&self, model: &str, d: &Delta) -> String {
        let choice = match self.kind {
            Kind::Chat => json!({"index": 0, "delta": delta_json(d), "logprobs": null, "finish_reason": null}),
            Kind::Completion => {
                let text = match d {
                    Delta::Content(s) | Delta::Reasoning(s) => s.clone(),
                    _ => String::new(),
                };
                json!({"index": 0, "text": text, "logprobs": null, "finish_reason": null})
            }
        };
        self.chunk(model, Some(choice), None)
    }

    fn final_chunk(&self, model: &str, reason: &str, timings: Option<&Timings>) -> String {
        let choice = match self.kind {
            Kind::Chat => json!({"index": 0, "delta": {}, "logprobs": null, "finish_reason": reason}),
            Kind::Completion => json!({"index": 0, "text": "", "logprobs": null, "finish_reason": reason}),
        };
        self.chunk_t(model, Some(choice), None, timings)
    }
}

fn sse(data: String) -> Result<Event, Infallible> {
    Ok(Event::default().data(data))
}

/// The `x-sampling-effective` response header: the sampling the request runs with (compact JSON).
fn effective_header(resp: &mut Response, effective: &Json) {
    if let Ok(v) = axum::http::HeaderValue::from_str(&effective.to_string()) {
        resp.headers_mut().insert("x-sampling-effective", v);
    }
}

fn stream_response(state: Arc<AppState>, p: Prepared, t_start: Instant) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(256);
    let effective = p.effective.clone();
    tokio::spawn(async move {
        let model = state.cfg.model_id.clone();
        let running = tokio::select! {
            r = start(&state, &p) => r,
            _ = tx.closed() => {
                tracing::info!("{} cancelled while queued (client gone)", p.id);
                state.capture_meta(&p.id, json!({"id": p.id, "finish_reason": null, "cancelled": "while queued", "completion_tokens": 0, "reasoning_cap_fired": false, "loop_breaker_fired": false}));
                return;
            }
        };
        let mut driver = Driver::new(state.clone(), running, &p, t_start);
        if p.kind == Kind::Chat {
            let first = json!({"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null});
            if tx.send(sse(p.chunk(&model, Some(first), None))).await.is_err() {
                return;
            }
        } else if let Some(echo) = &p.echo {
            if tx.send(sse(p.delta_chunk(&model, &Delta::Content(echo.clone())))).await.is_err() {
                return;
            }
        }
        loop {
            let step = tokio::select! {
                s = driver.next() => s,
                _ = tx.closed() => return, // dropping the driver cancels the engine request (and logs it)
            };
            let (deltas, outcome) = match step {
                Step::Deltas(d) => (d, None),
                Step::Final(d, o) => (d, Some(o)),
            };
            for d in &deltas {
                if tx.send(sse(p.delta_chunk(&model, d))).await.is_err() {
                    return;
                }
            }
            if let Some(o) = outcome {
                if let Some(e) = &o.error {
                    let _ = tx.send(sse(ApiError::internal(e.clone()).body().to_string())).await;
                } else {
                    let _ = tx.send(sse(p.final_chunk(&model, o.finish_reason, o.timings.as_ref()))).await;
                    if p.include_usage {
                        let _ = tx.send(sse(p.chunk_t(&model, None, Some(usage(p.prompt_ids.len(), &o, p.images_omitted)), o.timings.as_ref()))).await;
                    }
                }
                let _ = tx.send(sse("[DONE]".to_string())).await;
                return;
            }
        }
    });
    // SSE comment lines (`: keep-alive`) every 10 s of silence keep idle-timeout clients alive through queueing and long prefills
    let mut resp = Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(10)).text("keep-alive"))
        .into_response();
    // SSE is UTF-8 by definition, but clients that go by the header (python requests: text/* without a charset is ISO-8859-1)
    // would decode every multi-byte character into mojibake: say it
    resp.headers_mut().insert(axum::http::header::CONTENT_TYPE, axum::http::HeaderValue::from_static("text/event-stream; charset=utf-8"));
    effective_header(&mut resp, &effective);
    resp
}

async fn full_response(state: Arc<AppState>, p: Prepared, t_start: Instant) -> Response {
    // If the client disconnects, hyper drops this future, which drops `Running` and cancels the engine.
    let running = start(&state, &p).await;
    let mut driver = Driver::new(state.clone(), running, &p, t_start);
    let mut acc = Parsed::default();
    let outcome = loop {
        match driver.next().await {
            Step::Deltas(d) => acc.absorb(&d),
            Step::Final(d, o) => {
                acc.absorb(&d);
                break o;
            }
        }
    };
    if let Some(e) = outcome.error {
        return ApiError::internal(e).into_response();
    }
    let model = &state.cfg.model_id;
    let usage = usage(p.prompt_ids.len(), &outcome, p.images_omitted);
    let body = match p.kind {
        Kind::Chat => {
            let mut message = json!({"role": "assistant", "content": null});
            let content = if acc.tool_calls.is_empty() { acc.content.clone() } else { acc.content.trim().to_string() };
            if !content.is_empty() || acc.tool_calls.is_empty() {
                message["content"] = Json::String(content);
            }
            if !acc.reasoning.is_empty() {
                message["reasoning_content"] = Json::String(acc.reasoning.clone());
            }
            if !acc.tool_calls.is_empty() {
                message["tool_calls"] = Json::Array(
                    acc.tool_calls
                        .iter()
                        .map(|(id, name, args)| {
                            json!({"id": id, "type": "function", "function": {"name": name, "arguments": if args.is_empty() { "{}" } else { args }}})
                        })
                        .collect(),
                );
            }
            json!({
                "id": p.id, "object": p.object(false), "created": p.created, "model": model,
                "choices": [{"index": 0, "message": message, "logprobs": null, "finish_reason": outcome.finish_reason}],
                "usage": usage,
            })
        }
        Kind::Completion => {
            let text = format!("{}{}", p.echo.clone().unwrap_or_default(), acc.content);
            json!({
                "id": p.id, "object": p.object(false), "created": p.created, "model": model,
                "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": outcome.finish_reason}],
                "usage": usage,
            })
        }
    };
    let mut body = body;
    if let Some(t) = &outcome.timings {
        body["timings"] = t.to_json();
    }
    let mut resp = AxJson(body).into_response();
    effective_header(&mut resp, &p.effective);
    resp
}

async fn chat_completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let t_start = Instant::now();
    let incoming = state.engine.incoming();
    let prepared = parse_body::<ChatRequest>(&body).and_then(|r| prepare_chat(&state, r)).map(|p| Prepared { incoming: std::sync::Mutex::new(incoming), ..p });
    if state.dump_dir.get().is_some() {
        match &prepared {
            Ok(p) => state.capture_body(&p.id, &body),
            Err(e) => {
                // a rejected request has no id of its own
                let id = random_id("rejected-", 24);
                state.capture_body(&id, &body);
                state.capture_meta(&id, json!({"id": id, "finish_reason": null, "rejected": true, "error": e.message, "completion_tokens": 0, "reasoning_cap_fired": false, "loop_breaker_fired": false}));
            }
        }
    }
    run(state, prepared, t_start).await
}

async fn completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let t_start = Instant::now();
    let incoming = state.engine.incoming();
    let prepared = parse_body::<CompletionRequest>(&body).and_then(|r| prepare_completion(&state, r)).map(|p| Prepared { incoming: std::sync::Mutex::new(incoming), ..p });
    run(state, prepared, t_start).await
}

async fn run(state: Arc<AppState>, prepared: Result<Prepared, ApiError>, t_start: Instant) -> Response {
    let p = match prepared {
        Ok(p) => p,
        Err(e) => {
            tracing::info!("rejected request: {}", e.message);
            return e.into_response();
        }
    };
    if let Some(why) = state.engine.dead() {
        return ApiError::unavailable(format!("the engine stopped: {why}")).into_response();
    }
    tracing::info!(
        "{} queued: prompt {} tokens, max_tokens {}, temp {}, stream {}",
        p.id,
        p.prompt_ids.len(),
        p.params.max_tokens,
        p.params.temperature,
        p.stream
    );
    // (worded so that counting the warning lines "loop breaker:" / "reasoning cap:" in a log never counts this per-request line)
    tracing::info!("{} sampling: {}", p.id, p.effective);
    if p.stream { stream_response(state, p, t_start) } else { full_response(state, p, t_start).await }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ink")]
    #[test]
    fn pictures_take_the_place_of_the_placeholders() {
        use fe_ink::images::{IMG_ID_BASE, Picture};
        let pic = |key: u32, w: usize, h: usize| Picture { key, width: w, height: h, rgb: vec![0; w * h * 3] };
        let ph = fe_ink::vision::IMAGE_TOKEN;
        // 80 x 40 pixels = 2 image tokens, 41 x 81 = 2 x 3 = 6
        let (ids, images) = place_pictures(vec![5, ph, 6, 7, ph, 8], vec![pic(3, 80, 40), pic(9, 41, 81)]).unwrap();
        let (a, b) = (IMG_ID_BASE | 3, IMG_ID_BASE | 9);
        assert_eq!(ids, vec![5, a, a, 6, 7, b, b, b, b, b, b, 8]);
        assert_eq!(images.iter().map(|i| (i.pos, i.key, i.width, i.height)).collect::<Vec<_>>(), vec![(1, 3, 80, 40), (5, 9, 41, 81)]);
        // a placeholder count that differs from the pictures is refused
        assert!(place_pictures(vec![5, ph], vec![pic(3, 80, 40), pic(9, 41, 81)]).is_err());
        // image parts are taken out in message order; other URLs are refused
        let lim = fe_ink::images::Limits { max_images: 2, max_tokens: 100, max_tokens_each: 100, max_bytes: 1 << 20, drop_oldest: true };
        let mut m = vec![json!({"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]})];
        let e = take_pictures(&mut m, &lim).unwrap_err();
        assert!(e.starts_with("messages[0].content[1]:") && e.contains("does not fetch"), "{e}");
        let mut m = vec![json!({"role": "user", "content": [{"type": "image_url"}]})];
        assert!(take_pictures(&mut m, &lim).unwrap_err().contains("data: URL"));
        let mut m = vec![json!({"role": "user", "content": "no parts"}), json!({"role": "user", "content": [{"type": "text", "text": "hi"}]})];
        assert!(take_pictures(&mut m, &lim).unwrap().0.is_empty());
    }

    /// An image part with a grey PNG of w x h pixels (stored, not compressed); `shade` tells the pictures apart.
    #[cfg(feature = "ink")]
    fn png_part(w: u32, h: u32, shade: u8) -> Json {
        fn crc(d: &[u8]) -> u32 {
            let mut c = !0u32;
            for &b in d {
                c ^= b as u32;
                for _ in 0..8 {
                    c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
                }
            }
            !c
        }
        fn chunk(out: &mut Vec<u8>, ty: &[u8; 4], d: &[u8]) {
            out.extend((d.len() as u32).to_be_bytes());
            let mut td = ty.to_vec();
            td.extend(d);
            out.extend(&td);
            out.extend(crc(&td).to_be_bytes());
        }
        let mut raw = Vec::new();
        for _ in 0..h {
            raw.push(0);
            raw.extend(std::iter::repeat(shade).take(3 * w as usize));
        }
        assert!(raw.len() < 65536, "one stored block");
        let (mut a, mut b) = (1u32, 0u32);
        for &x in &raw {
            a = (a + x as u32) % 65521;
            b = (b + a) % 65521;
        }
        let mut z = vec![0x78, 0x01, 0x01];
        z.extend((raw.len() as u16).to_le_bytes());
        z.extend((!(raw.len() as u16)).to_le_bytes());
        z.extend(&raw);
        z.extend(((b << 16) | a).to_be_bytes());
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend(w.to_be_bytes());
        ihdr.extend(h.to_be_bytes());
        ihdr.extend([8, 2, 0, 0, 0]);
        chunk(&mut png, b"IHDR", &ihdr);
        chunk(&mut png, b"IDAT", &z);
        chunk(&mut png, b"IEND", &[]);
        const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::from("data:image/png;base64,");
        for c in png.chunks(3) {
            let v = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
            s.push(T[(v >> 18) as usize & 63] as char);
            s.push(T[(v >> 12) as usize & 63] as char);
            s.push(if c.len() > 1 { T[(v >> 6) as usize & 63] as char } else { '=' });
            s.push(if c.len() > 2 { T[v as usize & 63] as char } else { '=' });
        }
        json!({"type": "image_url", "image_url": {"url": s}})
    }

    #[cfg(feature = "ink")]
    #[test]
    fn the_oldest_pictures_of_a_request_beyond_the_limits_are_left_out() {
        // a conversation of n turns, one 80 x 40 picture (2 image tokens) a turn, the first one in a tool message
        let convo = |n: usize| -> Vec<Json> {
            (0..n)
                .flat_map(|i| {
                    let role = if i == 0 { "tool" } else { "user" };
                    [json!({"role": role, "content": [{"type": "text", "text": format!("turn {i}")}, png_part(80, 40, i as u8)]}), json!({"role": "assistant", "content": "ok"})]
                })
                .collect()
        };
        let shape = |m: &[Json]| -> Vec<&'static str> {
            m.iter()
                .filter_map(|m| m["content"].as_array())
                .map(|p| {
                    if p[1]["type"] == "image_url" {
                        assert_eq!(p[1], json!({"type": "image_url"}));
                        "pic"
                    } else {
                        assert_eq!(p[1], json!({"type": "text", "text": "[earlier picture omitted]"}));
                        "gone"
                    }
                })
                .collect()
        };
        let lim = fe_ink::images::Limits { max_images: 8, max_tokens: 1000, max_tokens_each: 100, max_bytes: 1 << 20, drop_oldest: true };
        assert_eq!(lim.overflow_batch(), 2);
        // within the limits: nothing is left out
        let mut m = convo(8);
        let (pics, omitted) = take_pictures(&mut m, &lim).unwrap();
        assert_eq!((pics.len(), omitted), (8, 0));
        assert_eq!(pics[0].tokens(), 2);
        // 9 and 10 pictures: the same 2 left out, in place; the kept ones are the newest, in order
        let mut keys = Vec::new();
        for n in [9, 10] {
            let mut m = convo(n);
            let (pics, omitted) = take_pictures(&mut m, &lim).unwrap();
            assert_eq!((pics.len(), omitted), (n - 2, 2), "n {n}");
            let s = shape(&m);
            assert!(s[..2].iter().all(|x| *x == "gone") && s[2..].iter().all(|x| *x == "pic"), "{s:?}");
            keys.push(pics.iter().map(|p| p.key).collect::<Vec<_>>());
        }
        // the prompt cache's view: the pictures of the 9-picture request lead the 10-picture request
        assert_eq!(keys[0][..], keys[1][..7]);
        assert_eq!(take_pictures(&mut convo(11), &lim).unwrap().1, 4);
        // the token limit: 2 tokens a picture, 5 fit in 11 tokens; 6 pictures lose 2
        let tight = fe_ink::images::Limits { max_tokens: 11, ..lim };
        assert_eq!(take_pictures(&mut convo(5), &tight).unwrap().1, 0);
        let (pics, omitted) = take_pictures(&mut convo(6), &tight).unwrap();
        assert_eq!((pics.len(), omitted), (4, 2));
        // a bad picture is still refused, also among the oldest
        let mut m = convo(9);
        m[0]["content"][1] = json!({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}});
        assert!(take_pictures(&mut m, &lim).unwrap_err().contains("does not fetch"));
        // INK_IMAGE_OVERFLOW=reject: HTTP 400 as before
        let reject = fe_ink::images::Limits { drop_oldest: false, ..lim };
        assert_eq!(take_pictures(&mut convo(8), &reject).unwrap().0.len(), 8);
        let e = take_pictures(&mut convo(9), &reject).unwrap_err();
        assert!(e.starts_with("messages[16].content[1]:") && e.contains("more than 8 pictures in one request (INK_IMAGE_MAX)"), "{e}");
        let e = take_pictures(&mut convo(6), &fe_ink::images::Limits { drop_oldest: false, ..tight }).unwrap_err();
        assert!(e.contains("more than 11 image tokens") && e.contains("INK_IMAGE_TOKENS"), "{e}");
    }

    #[test]
    fn usage_names_omitted_pictures_only_when_there_are_some() {
        let o = Outcome { finish_reason: "stop", reasoning_tokens: 0, error: None, completion_tokens: 2, cached_prompt_tokens: 3, timings: None };
        let plain = usage(10, &o, 0);
        assert_eq!(plain["prompt_tokens_details"], json!({"cached_tokens": 3}));
        let u = usage(10, &o, 16);
        assert_eq!(u["prompt_tokens_details"], json!({"cached_tokens": 3, "images_omitted": 16}));
        assert_eq!(u["prompt_tokens"], plain["prompt_tokens"]);
    }

    fn prepared(include_usage: bool) -> Prepared {
        Prepared {
            kind: Kind::Chat,
            id: "chatcmpl-x".into(),
            created: 1,
            prompt_ids: vec![1, 2, 3],
            images: Vec::new(),
            images_omitted: 0,
            snapshots: Vec::new(),
            params: SamplingParams::default(),
            stops: Vec::new(),
            include_stop: false,
            skip_special: true,
            parser: None,
            stream: true,
            include_usage,
            echo: None,
            effective: Json::Null,
            incoming: std::sync::Mutex::new(None),
        }
    }

    #[test]
    fn timings_absent_leaves_the_json_unchanged() {
        // engines without timings (the CUDA engines, the mock): the chunks are byte for byte what they were
        for iu in [false, true] {
            let p = prepared(iu);
            let choice = json!({"index": 0, "delta": {}, "logprobs": null, "finish_reason": "stop"});
            assert_eq!(p.final_chunk("m", "stop", None), p.chunk("m", Some(choice), None));
            let u = json!({"prompt_tokens": 3});
            assert_eq!(p.chunk_t("m", None, Some(u.clone()), None), p.chunk("m", None, Some(u)));
            assert!(!p.final_chunk("m", "stop", None).contains("timings"));
        }
    }

    #[test]
    fn timings_present_as_llama_server() {
        let p = prepared(true);
        let t = Timings { prompt_n: 2000, cache_n: 26, prompt_ms: 1600.0, predicted_n: 501, predicted_ms: 5000.0, draft: Some((700, 400)) };
        let c: Json = serde_json::from_str(&p.final_chunk("m", "stop", Some(&t))).unwrap();
        let tj = &c["timings"];
        assert_eq!(tj["prompt_n"], 2000);
        assert_eq!(tj["cache_n"], 26);
        assert_eq!(tj["prompt_per_second"], 1250.0);
        assert_eq!(tj["predicted_n"], 501);
        assert_eq!(tj["predicted_per_second"], 100.0);
        assert_eq!(tj["draft_n"], 700);
        assert_eq!(tj["draft_n_accepted"], 400);
        let none = Timings { draft: None, ..t };
        assert!(none.to_json().get("draft_n").is_none());
        // the rest of the chunk is the plain one plus the key
        let mut plain: Json = serde_json::from_str(&p.final_chunk("m", "stop", None)).unwrap();
        plain["timings"] = tj.clone();
        assert_eq!(c, plain);
    }

    #[test]
    fn thinking_detection() {
        assert!(prompt_opens_thinking("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"));
        assert!(!prompt_opens_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(!prompt_opens_thinking("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"));
        assert!(!prompt_opens_thinking("<|im_start|>assistant\n<think>\nr\n</think>\n\nI was say"));
    }

    #[test]
    fn tool_choice_parsing() {
        assert_eq!(parse_tool_choice(None).unwrap(), ToolChoice::Auto);
        assert_eq!(parse_tool_choice(Some(&json!("none"))).unwrap(), ToolChoice::None);
        assert_eq!(parse_tool_choice(Some(&json!("required"))).unwrap(), ToolChoice::Required);
        assert_eq!(
            parse_tool_choice(Some(&json!({"type": "function", "function": {"name": "f"}}))).unwrap(),
            ToolChoice::Named("f".into())
        );
        assert!(parse_tool_choice(Some(&json!("sometimes"))).is_err());
    }

    #[test]
    fn system_hint() {
        let mut m = vec![json!({"role": "user", "content": "x"})];
        add_system_hint(&mut m, "H");
        assert_eq!(m[0], json!({"role": "system", "content": "H"}));
        let mut m = vec![json!({"role": "system", "content": "S"}), json!({"role": "user", "content": "x"})];
        add_system_hint(&mut m, "H");
        assert_eq!(m[0]["content"], "S\n\nH");
    }
}
