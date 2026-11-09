//! OpenAI request types (lenient: unknown fields ignored, nulls allowed) and error responses.
use axum::Json as AxJson;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Map, Value as Json, json};

#[derive(Deserialize, Debug, Default, Clone)]
#[serde(default)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Stop::One(s) => vec![s],
            Stop::Many(v) => v,
        }
    }
}

/// Sampling fields shared by chat and completions (OpenAI + the vLLM extensions people use).
#[derive(Deserialize, Debug, Default, Clone)]
#[serde(default)]
pub struct SamplingRequest {
    pub max_tokens: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub min_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub stop: Option<Stop>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub ignore_eos: Option<bool>,
    pub include_stop_str_in_output: Option<bool>,
    pub skip_special_tokens: Option<bool>,
}

#[derive(Deserialize, Debug, Default, Clone)]
#[serde(default)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<Json>,
    pub tools: Option<Vec<Json>>,
    pub tool_choice: Option<Json>,
    pub parallel_tool_calls: Option<bool>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    #[serde(flatten)]
    pub sampling: SamplingRequest,
    pub chat_template_kwargs: Option<Map<String, Json>>,
    /// OpenAI-style top-level effort: none / minimal / low / medium / high (mapped onto the template).
    pub reasoning_effort: Option<String>,
    /// Qwen-API-style top-level switch (same as `chat_template_kwargs.enable_thinking`).
    pub enable_thinking: Option<bool>,
    pub add_generation_prompt: Option<bool>,
    pub continue_final_message: Option<bool>,
    pub n: Option<u32>,
}

#[derive(Deserialize, Debug, Default, Clone)]
#[serde(default)]
pub struct CompletionRequest {
    pub model: Option<String>,
    /// A string, `[string]`, `[token ids]` or `[[token ids]]` (one prompt only).
    pub prompt: Option<Json>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    #[serde(flatten)]
    pub sampling: SamplingRequest,
    pub echo: Option<bool>,
    pub n: Option<u32>,
    pub suffix: Option<String>,
}

/// An OpenAI-shaped error response.
#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, message: msg.into() }
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, message: msg.into() }
    }
    pub fn body(&self) -> Json {
        let kind = if self.status.is_client_error() { "invalid_request_error" } else { "server_error" };
        json!({"error": {"message": self.message, "type": kind, "param": null, "code": self.status.as_u16()}})
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, AxJson(self.body())).into_response()
    }
}

pub fn parse_body<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError::bad_request(format!("invalid request body: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lenient_chat_request() {
        let r: ChatRequest = parse_body(
            br#"{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":true,
                 "stream_options":{"include_usage":true},"max_tokens":5,"temperature":null,"stop":"END",
                 "top_k":-1,"chat_template_kwargs":{"enable_thinking":false},"logprobs":false,"whatever":1}"#,
        )
        .unwrap();
        assert_eq!(r.messages.len(), 1);
        assert_eq!(r.sampling.max_tokens, Some(5));
        assert_eq!(r.sampling.temperature, None);
        assert_eq!(r.sampling.top_k, Some(-1));
        assert!(matches!(r.sampling.stop, Some(Stop::One(ref s)) if s == "END"));
        assert_eq!(r.stream_options.unwrap().include_usage, Some(true));
        let e = parse_body::<ChatRequest>(b"{nope").unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
    }
}
