//! The axum app: routes, request preparation, the token → text → deltas pipeline, SSE / JSON output,
//! concurrency slots and cancellation.
use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;
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

use crate::api::{ApiError, ChatRequest, CompletionRequest, SamplingRequest, parse_body};
use crate::engine::{CancelToken, Engine, FinishReason, GenEvent, GenRequest, SamplingParams, TokenSink};
use crate::parser::{Delta, OutputParser, ParserConfig, Parsed};
use crate::stop::StopMatcher;
use crate::template::{ChatTemplate, normalize_messages};
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
        Self {
            temperature: if do_sample { f("temperature", 1.0) } else { 0.0 },
            top_p: f("top_p", 1.0),
            top_k: cfg.get("top_k").and_then(Json::as_u64).unwrap_or(0) as usize,
            min_p: f("min_p", 0.0),
            repetition_penalty: f("repetition_penalty", 1.0),
            eos_ids,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// The model id reported everywhere (requests may name any model).
    pub model_id: String,
    /// Requests generating at once; the rest wait in a FIFO queue.
    pub max_concurrent: usize,
    /// Prompt + output token limit per request.
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
}

impl AppState {
    pub fn new(engine: Arc<dyn Engine>, tok: Arc<Tok>, template: Arc<ChatTemplate>, cfg: ServerConfig) -> Arc<Self> {
        Arc::new(Self {
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
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(model_one))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(DefaultBodyLimit::max(256 << 20))
        .with_state(state)
}

async fn health(State(s): State<Arc<AppState>>) -> AxJson<Json> {
    AxJson(json!({
        "status": "ok",
        "model": s.cfg.model_id,
        "engine": s.engine.info().name,
        "max_batch_size": s.cfg.max_concurrent,
        "max_model_len": s.cfg.max_context,
        "running": s.running(),
        "queued": s.queued(),
    }))
}

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
) -> Result<(SamplingParams, Vec<String>, bool, bool), ApiError> {
    let ctx = s.cfg.max_context;
    if prompt_len >= ctx {
        return Err(ApiError::bad_request(format!(
            "the prompt is {prompt_len} tokens; this model's context window is {ctx} tokens"
        )));
    }
    let room = ctx - prompt_len;
    let requested = r.max_completion_tokens.or(r.max_tokens).map(|v| v as usize);
    if requested == Some(0) {
        return Err(ApiError::bad_request("max_tokens must be at least 1"));
    }
    let max_tokens = requested.or(default_max).unwrap_or(room).min(room);
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
    let params = SamplingParams {
        max_tokens,
        min_tokens: (r.min_tokens.unwrap_or(0) as usize).min(max_tokens),
        temperature: match r.temperature.map(|t| t as f32).unwrap_or(d.temperature) {
            t if t < 1e-5 => 0.0, // greedy, as vLLM treats temperatures below 1e-5
            t => t,
        },
        top_p: r.top_p.map(|t| t as f32).unwrap_or(d.top_p),
        top_k: match r.top_k {
            Some(k) if k > 0 => k as usize,
            Some(_) => 0,
            None => d.top_k,
        },
        min_p: r.min_p.map(|t| t as f32).unwrap_or(d.min_p),
        presence_penalty: r.presence_penalty.unwrap_or(0.0) as f32,
        frequency_penalty: r.frequency_penalty.unwrap_or(0.0) as f32,
        repetition_penalty: r.repetition_penalty.map(|t| t as f32).unwrap_or(d.repetition_penalty),
        seed: r.seed.map(|v| v as u64),
        stop_token_ids: stop_ids,
    };
    let stops = r.stop.clone().map(|s| s.into_vec()).unwrap_or_default();
    Ok((params, stops, r.include_stop_str_in_output.unwrap_or(false), r.skip_special_tokens.unwrap_or(true)))
}

fn prepare_chat(s: &AppState, req: ChatRequest) -> Result<Prepared, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must be a non-empty list"));
    }
    if req.n.unwrap_or(1) > 1 {
        return Err(ApiError::bad_request("n > 1 is not supported"));
    }
    let mut messages = normalize_messages(&req.messages).map_err(ApiError::bad_request)?;
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
    if let Some(effort) = &req.reasoning_effort {
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
    let continue_final = req.continue_final_message.unwrap_or(false);
    let add_gen = req.add_generation_prompt.unwrap_or(true) && !continue_final;
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
    let thinking = prompt_opens_thinking(&prompt);
    if tracing::enabled!(tracing::Level::DEBUG) {
        let tail: String = prompt.chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect();
        tracing::debug!("rendered prompt ({} chars, thinking {thinking}), tail: {tail:?}", prompt.len());
    }
    let prompt_ids = s.tok.encode(&prompt).map_err(|e| ApiError::internal(e.to_string()))?;
    let (params, stops, include_stop, skip_special) = resolve_sampling(s, &req.sampling, prompt_ids.len(), None)?;
    let parser = ParserConfig {
        thinking,
        tools: template_tools.as_ref().map(|t| ToolSet::from_openai(t)),
        max_tool_calls: (req.parallel_tool_calls == Some(false)).then_some(1),
    };
    let stream = req.stream.unwrap_or(false);
    Ok(Prepared {
        kind: Kind::Chat,
        id: random_id("chatcmpl-", 24),
        created: unix_now(),
        snapshots: s.marks.points(&prompt_ids),
        prompt_ids,
        params,
        stops,
        include_stop,
        skip_special,
        parser: Some(parser),
        stream,
        include_usage: stream && req.stream_options.and_then(|o| o.include_usage).unwrap_or(false),
        echo: None,
    })
}

fn prepare_completion(s: &AppState, req: CompletionRequest) -> Result<Prepared, ApiError> {
    if req.n.unwrap_or(1) > 1 {
        return Err(ApiError::bad_request("n > 1 is not supported"));
    }
    if req.suffix.as_deref().is_some_and(|x| !x.is_empty()) {
        return Err(ApiError::bad_request("suffix is not supported"));
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
    let (params, stops, include_stop, skip_special) = resolve_sampling(s, &req.sampling, prompt_ids.len(), Some(16))?;
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
        params,
        stops,
        include_stop,
        skip_special,
        parser: None,
        stream,
        include_usage: stream && req.stream_options.and_then(|o| o.include_usage).unwrap_or(false),
        echo,
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
        params: p.params.clone(), cancel: cancel.clone() };
    s.engine.submit(req, TokenSink::with_slot(tx, Box::new(slot)));
    Running { rx, cancel }
}

/// Token ids → text (UTF-8 safe) → stop strings → reasoning / content / tool-call deltas.
struct Pipeline {
    dec: IncrementalDecoder,
    stop: StopMatcher,
    parser: Option<OutputParser>,
    max_tokens: usize,
    completion_tokens: usize,
    /// Set when the server ends the request itself (stop string or max_tokens).
    server_finish: Option<&'static str>,
    closed: bool,
}

impl Pipeline {
    fn new(p: &Prepared) -> Self {
        Self {
            dec: IncrementalDecoder::new(p.skip_special),
            stop: StopMatcher::new(p.stops.clone(), p.include_stop),
            parser: p.parser.clone().map(OutputParser::new),
            max_tokens: p.params.max_tokens,
            completion_tokens: 0,
            server_finish: None,
            closed: false,
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
        let mut text = String::new();
        for &id in ids {
            if self.completion_tokens >= self.max_tokens {
                self.server_finish = Some("length");
                break;
            }
            self.completion_tokens += 1;
            text.push_str(&self.dec.push(tok, id).map_err(|e| e.to_string())?);
        }
        if self.completion_tokens >= self.max_tokens && self.server_finish.is_none() {
            self.server_finish = Some("length");
        }
        let (emit, hit) = self.stop.push(&text);
        if hit {
            self.server_finish = Some("stop");
        }
        let mut deltas = self.parse(&emit);
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
        Ok(deltas)
    }

    fn tool_calls(&self) -> usize {
        self.parser.as_ref().map(|p| p.tool_calls()).unwrap_or(0)
    }
}

struct Outcome {
    finish_reason: &'static str,
    error: Option<String>,
    completion_tokens: usize,
    cached_prompt_tokens: usize,
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
}

impl Drop for Driver {
    fn drop(&mut self) {
        if !self.finished {
            tracing::info!(
                "{} cancelled: client disconnected after {} output tokens",
                self.id,
                self.pipe.completion_tokens
            );
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
        }
    }

    async fn next(&mut self) -> Step {
        let ev = self.running.rx.recv().await;
        let tok = self.state.tok.clone();
        match ev {
            Some(GenEvent::Tokens(ids)) => {
                if self.t_first.is_none() {
                    self.t_first = Some(Instant::now());
                }
                match self.pipe.tokens(&tok, &ids) {
                    Ok(d) if self.pipe.server_finish.is_none() => Step::Deltas(d),
                    Ok(d) => {
                        // the server ended it (stop string / max_tokens): stop the engine, and take its
                        // final event for the prompt-cache count
                        self.running.cancel.cancel();
                        let cached = self.engine_finish().await;
                        let reason = self.pipe.server_finish.unwrap_or("stop");
                        Step::Final(d, self.outcome(reason, None, 0, cached))
                    }
                    Err(e) => {
                        self.running.cancel.cancel();
                        Step::Final(Vec::new(), self.outcome("stop", Some(e), 0, 0))
                    }
                }
            }
            Some(GenEvent::Finished(f)) => {
                let d = self.pipe.close(&tok).unwrap_or_default();
                let (reason, err, extra) = match (self.pipe.server_finish, &f.reason) {
                    (Some(r), _) => (r, None, 0),
                    (None, FinishReason::Stop(_)) => ("stop", None, 1), // the stop token counts as output
                    (None, FinishReason::Length) => ("length", None, 0),
                    (None, FinishReason::Cancelled) => ("stop", None, 0),
                    (None, FinishReason::Error(e)) => ("stop", Some(e.clone()), 0),
                };
                Step::Final(d, self.outcome(reason, err, extra, f.cached_prompt_tokens))
            }
            None => {
                let d = self.pipe.close(&tok).unwrap_or_default();
                Step::Final(d, self.outcome("stop", Some("the engine dropped the request".into()), 0, 0))
            }
        }
    }

    /// Wait (briefly) for the engine's `Finished` after a server-side stop; returns its cached-prompt count.
    async fn engine_finish(&mut self) -> usize {
        let deadline = tokio::time::sleep(std::time::Duration::from_secs(2));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                ev = self.running.rx.recv() => match ev {
                    Some(GenEvent::Finished(f)) => return f.cached_prompt_tokens,
                    Some(GenEvent::Tokens(_)) => continue,
                    None => return 0,
                },
                _ = &mut deadline => return 0,
            }
        }
    }

    fn outcome(&mut self, reason: &'static str, error: Option<String>, extra: usize, cached: usize) -> Outcome {
        self.finished = true;
        let reason = if reason == "stop" && self.pipe.tool_calls() > 0 { "tool_calls" } else { reason };
        let completion_tokens = self.pipe.completion_tokens + extra;
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
        Outcome { finish_reason: reason, error, completion_tokens, cached_prompt_tokens: cached }
    }
}

// ---------------------------------------------------------------------------------------------
// output formatting

fn usage(prompt: usize, out: &Outcome) -> Json {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": out.completion_tokens,
        "total_tokens": prompt + out.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": out.cached_prompt_tokens},
    })
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

    fn final_chunk(&self, model: &str, reason: &str) -> String {
        let choice = match self.kind {
            Kind::Chat => json!({"index": 0, "delta": {}, "logprobs": null, "finish_reason": reason}),
            Kind::Completion => json!({"index": 0, "text": "", "logprobs": null, "finish_reason": reason}),
        };
        self.chunk(model, Some(choice), None)
    }
}

fn sse(data: String) -> Result<Event, Infallible> {
    Ok(Event::default().data(data))
}

fn stream_response(state: Arc<AppState>, p: Prepared, t_start: Instant) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(256);
    tokio::spawn(async move {
        let model = state.cfg.model_id.clone();
        let running = tokio::select! {
            r = start(&state, &p) => r,
            _ = tx.closed() => {
                tracing::info!("{} cancelled while queued (client gone)", p.id);
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
                    let _ = tx.send(sse(p.final_chunk(&model, o.finish_reason))).await;
                    if p.include_usage {
                        let _ = tx.send(sse(p.chunk(&model, None, Some(usage(p.prompt_ids.len(), &o))))).await;
                    }
                }
                let _ = tx.send(sse("[DONE]".to_string())).await;
                return;
            }
        }
    });
    // SSE comment lines every 15 s keep idle-timeout clients alive through queueing and long prefills
    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response()
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
    let usage = usage(p.prompt_ids.len(), &outcome);
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
    AxJson(body).into_response()
}

async fn chat_completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let t_start = Instant::now();
    let prepared = parse_body::<ChatRequest>(&body).and_then(|r| prepare_chat(&state, r));
    run(state, prepared, t_start).await
}

async fn completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let t_start = Instant::now();
    let prepared = parse_body::<CompletionRequest>(&body).and_then(|r| prepare_completion(&state, r));
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
    tracing::info!(
        "{} queued: prompt {} tokens, max_tokens {}, temp {}, stream {}",
        p.id,
        p.prompt_ids.len(),
        p.params.max_tokens,
        p.params.temperature,
        p.stream
    );
    if p.stream { stream_response(state, p, t_start) } else { full_response(state, p, t_start).await }
}

#[cfg(test)]
mod tests {
    use super::*;

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
