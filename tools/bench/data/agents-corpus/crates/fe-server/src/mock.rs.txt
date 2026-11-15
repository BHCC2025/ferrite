//! `MockEngine`: a deterministic stand-in for the real engine (`--mock`).
//!
//! It only sees prompt token ids, like the real engine. It decodes them back to text and picks a canned
//! reply from what the rendered chat prompt shows:
//! - tools offered (`<tools>` block) and the last user message names one (e.g. "weather" for
//!   `get_weather`) → reasoning (if thinking is on) + a short sentence + one `<tool_call>` block per
//!   matching tool, arguments made up from the tool's JSON schema;
//! - the last turn is a tool response → a sentence quoting the tool result;
//! - otherwise → reasoning (if thinking is on) + an answer that echoes the user and contains
//!   multi-byte UTF-8. If the user text contains "write" or "long" (quickbench's prompts do) it then
//!   keeps generating numbered filler lines until `max_tokens`.
//! Raw completions (no chat markup) get a continuation sentence (plus filler under the same rule).
//!
//! Tokens are produced one at a time (or `burst` at a time, like speculative decoding) with a fixed
//! delay, honouring `max_tokens`, `min_tokens`, stop token ids and cancellation.
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value as Json;

use crate::engine::{Engine, EngineInfo, Finish, FinishReason, GenRequest, TokenSink};
use crate::tokenizer::Tok;

#[derive(Clone, Debug)]
pub struct MockConfig {
    /// Delay per decode step.
    pub delay: Duration,
    /// Tokens per step (>1 simulates speculative decoding).
    pub burst: usize,
    /// Simulated prefill speed in tokens/s (0 = instant).
    pub prefill_tps: f64,
    pub max_batch: usize,
    pub max_context: usize,
    /// Prompt tokens reported as served from a prompt cache (at most the prompt length - 1); 0 = none.
    pub cached_prompt_tokens: usize,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self { delay: Duration::from_millis(5), burst: 1, prefill_tps: 0.0, max_batch: 4, max_context: 131_072, cached_prompt_tokens: 0 }
    }
}

pub struct MockEngine {
    tok: Arc<Tok>,
    cfg: MockConfig,
}

impl MockEngine {
    pub fn new(tok: Arc<Tok>, cfg: MockConfig) -> Self {
        Self { tok, cfg }
    }
}

impl Engine for MockEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "mock".into(), max_batch: self.cfg.max_batch, max_context: self.cfg.max_context }
    }

    fn submit(&self, req: GenRequest, sink: TokenSink) {
        let tok = self.tok.clone();
        let cfg = self.cfg.clone();
        let spawned = std::thread::Builder::new().name("mock-gen".into()).spawn(move || run(&tok, &cfg, req, sink));
        if let Err(e) = spawned {
            tracing::error!("mock engine: cannot spawn a thread: {e}");
        }
    }
}

fn run(tok: &Tok, cfg: &MockConfig, req: GenRequest, sink: TokenSink) {
    let p = &req.params;
    let cached = cfg.cached_prompt_tokens.min(req.prompt_ids.len().saturating_sub(1));
    let done = |reason: FinishReason| Finish { reason, cached_prompt_tokens: cached, timings: None };
    let prompt = match tok.decode(&req.prompt_ids, false) {
        Ok(s) => s,
        Err(e) => return sink.finish(done(FinishReason::Error(format!("mock: {e}")))),
    };
    let plan = plan_response(&prompt);
    let canned = match tok.encode(&plan.text) {
        Ok(ids) => ids,
        Err(e) => return sink.finish(done(FinishReason::Error(format!("mock: {e}")))),
    };
    if cfg.prefill_tps > 0.0 {
        let total = Duration::from_secs_f64(req.prompt_ids.len() as f64 / cfg.prefill_tps);
        // sleep in slices so cancellation during "prefill" is noticed
        let slice = Duration::from_millis(20);
        let mut left = total;
        while !left.is_zero() {
            if req.cancel.is_cancelled() || sink.is_closed() {
                return sink.finish(done(FinishReason::Cancelled));
            }
            let d = left.min(slice);
            std::thread::sleep(d);
            left -= d;
        }
    }
    // the "eos" this mock samples once the canned text is done
    let eos = p.stop_token_ids.first().copied();
    let mut pending: std::collections::VecDeque<u32> = canned.into();
    let mut filler_line = 0usize;
    let mut generated = 0usize;
    loop {
        if req.cancel.is_cancelled() || sink.is_closed() {
            return sink.finish(done(FinishReason::Cancelled));
        }
        let mut step = Vec::with_capacity(cfg.burst);
        let mut stop = None;
        while step.len() < cfg.burst.max(1) {
            if generated + step.len() >= p.max_tokens {
                break;
            }
            if pending.is_empty() {
                let more = plan.long || eos.is_none() || generated + step.len() < p.min_tokens;
                if more {
                    filler_line += 1;
                    let line = format!("Line {filler_line}: the quick brown fox jumps over the lazy dog, café ✓.\n");
                    pending.extend(tok.encode(&line).unwrap_or_default());
                } else {
                    stop = eos;
                    break;
                }
            }
            let t = pending.pop_front().unwrap_or(0);
            if p.stop_token_ids.contains(&t) {
                if generated + step.len() >= p.min_tokens {
                    stop = Some(t);
                    break;
                }
                continue; // banned before min_tokens
            }
            step.push(t);
        }
        if !cfg.delay.is_zero() {
            std::thread::sleep(cfg.delay);
        }
        generated += step.len();
        if !step.is_empty() && !sink.send_tokens(step) {
            return sink.finish(done(FinishReason::Cancelled));
        }
        if let Some(t) = stop {
            return sink.finish(done(FinishReason::Stop(t)));
        }
        if generated >= p.max_tokens {
            return sink.finish(done(FinishReason::Length));
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub text: String,
    /// Keep generating filler until max_tokens after `text`.
    pub long: bool,
}

/// Choose the canned reply for a rendered prompt.
pub fn plan_response(prompt: &str) -> Plan {
    let wants_long = |s: &str| {
        let l = s.to_lowercase();
        l.contains("write") || l.contains("long")
    };
    if !prompt.contains("<|im_start|>") {
        let tail: String = prompt.chars().rev().take(40).collect::<Vec<_>>().into_iter().rev().collect();
        return Plan {
            text: format!(" and so the mock engine continues after \"{}\". Ünïcödé: 日本語 🦀.", tail.trim().replace('\n', " ")),
            long: wants_long(prompt),
        };
    }
    let thinking = prompt.ends_with("<think>\n");
    let tools = tools_in_prompt(prompt);
    let user = last_user_turn(prompt);
    let short: String = user.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(60).collect();
    let think = |r: &str| if thinking { format!("{r}\n</think>\n\n") } else { String::new() };

    if user.trim_start().starts_with("<tool_response>") {
        let result = between(&user, "<tool_response>", "</tool_response>").unwrap_or_default();
        let result: String = result.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(120).collect();
        return Plan {
            text: format!(
                "{}Based on the tool result: {result}",
                think("The tool answered; summarize its result for the user.")
            ),
            long: false,
        };
    }

    let lower = user.to_lowercase();
    let matching: Vec<&(String, Json)> = tools
        .iter()
        .filter(|(name, _)| {
            let n = name.to_lowercase();
            let last = n.rsplit('_').next().unwrap_or(&n);
            lower.contains(&n) || lower.contains(&n.replace('_', " ")) || (last.len() >= 4 && lower.contains(last))
        })
        .collect();
    if !matching.is_empty() {
        let names: Vec<&str> = matching.iter().map(|(n, _)| n.as_str()).collect();
        let mut text = think(&format!("The user needs {}. I will call the tool.", names.join(" and ")));
        text.push_str(&format!("I'll use {} for this.\n\n", names.join(" and ")));
        for (i, (name, params)) in matching.iter().enumerate() {
            if i > 0 {
                text.push('\n');
            }
            text.push_str(&format!("<tool_call>\n<function={name}>\n"));
            for (arg, value) in mock_arguments(params, &user) {
                text.push_str(&format!("<parameter={arg}>\n{value}\n</parameter>\n"));
            }
            text.push_str("</function>\n</tool_call>");
        }
        return Plan { text, long: false };
    }

    let reasoning = format!("The user wrote: \"{short}\". This is the mock engine, so I give my fixed reply.");
    Plan {
        text: format!(
            "{}Hello from the ferrite mock engine! You said: \"{short}\". Some UTF-8 to exercise streaming: café, naïve, Zürich, 日本語のテキスト, emoji 🦀🚀✨.",
            think(&reasoning)
        ) + if wants_long(&user) { "\n\n" } else { "" },
        long: wants_long(&user),
    }
}

/// The tools listed in the system prompt's `<tools>` block: (name, parameters schema).
fn tools_in_prompt(prompt: &str) -> Vec<(String, Json)> {
    let Some(block) = between(prompt, "<tools>", "</tools>") else { return Vec::new() };
    block
        .lines()
        .filter_map(|l| serde_json::from_str::<Json>(l.trim()).ok())
        .filter_map(|t| {
            let f = t.get("function").cloned().unwrap_or(t);
            let name = f.get("name")?.as_str()?.to_string();
            Some((name, f.get("parameters").cloned().unwrap_or(Json::Null)))
        })
        .collect()
}

fn last_user_turn(prompt: &str) -> String {
    const START: &str = "<|im_start|>user\n";
    match prompt.rfind(START) {
        Some(i) => {
            let rest = &prompt[i + START.len()..];
            rest.split("<|im_end|>").next().unwrap_or(rest).to_string()
        }
        None => String::new(),
    }
}

fn between(s: &str, a: &str, b: &str) -> Option<String> {
    let i = s.find(a)? + a.len();
    let j = s[i..].find(b)? + i;
    Some(s[i..j].trim().to_string())
}

/// Plausible argument values from a JSON schema (required parameters, or all if none are marked).
fn mock_arguments(params: &Json, user: &str) -> Vec<(String, String)> {
    let Some(props) = params.get("properties").and_then(Json::as_object) else { return Vec::new() };
    let required: Vec<&str> = params
        .get("required")
        .and_then(Json::as_array)
        .map(|r| r.iter().filter_map(Json::as_str).collect())
        .unwrap_or_default();
    // a place: the capitalized word after "in"/"for"/"at", else "Paris"
    let words: Vec<&str> = user.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let place = words
        .windows(2)
        .find(|w| matches!(w[0], "in" | "for" | "at") && w[1].chars().next().is_some_and(char::is_uppercase))
        .map(|w| w[1])
        .unwrap_or("Paris")
        .to_string();
    let mut out = Vec::new();
    for (name, schema) in props {
        if !required.is_empty() && !required.contains(&name.as_str()) {
            continue;
        }
        if let Some(first) = schema.get("enum").and_then(Json::as_array).and_then(|e| e.first()) {
            out.push((name.clone(), first.as_str().map(str::to_string).unwrap_or_else(|| first.to_string())));
            continue;
        }
        let ty = schema.get("type").and_then(Json::as_str).unwrap_or("string");
        let v = match ty {
            "integer" => "3".to_string(),
            "number" => "0.5".to_string(),
            "boolean" => "true".to_string(),
            "array" => "[\"a\", \"b\"]".to_string(),
            "object" => "{\"key\": \"value\"}".to_string(),
            _ => {
                let n = name.to_lowercase();
                if n.contains("location") || n.contains("city") || n.contains("place") {
                    place.clone()
                } else if n.contains("path") || n.contains("file") {
                    // a file-looking word from the request, e.g. "notes.txt"
                    user.split_whitespace()
                        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != '_' && c != '-' && c != '.'))
                        .map(|w| w.trim_end_matches('.'))
                        .find(|w| w.contains('.') && w.len() > 2)
                        .unwrap_or("README.md")
                        .to_string()
                } else if n.contains("query") || n.contains("text") || n.contains("prompt") {
                    user.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(60).collect()
                } else {
                    format!("mock {name}")
                }
            }
        };
        out.push((name.clone(), v));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans() {
        let p = plan_response("<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert!(p.text.contains("</think>\n\nHello from the ferrite mock engine! You said: \"Hi there\"."), "{}", p.text);
        assert!(!p.long);
        let p = plan_response("<|im_start|>user\nwrite a story<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert!(!p.text.contains("</think>") && p.long);
        let tools = "<|im_start|>system\n# Tools\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"celsius\"]}}, \"required\": [\"location\"]}}}\n</tools><|im_end|>\n";
        let p = plan_response(&format!("{tools}<|im_start|>user\nWhat is the weather in Paris?<|im_end|>\n<|im_start|>assistant\n<think>\n"));
        assert!(p.text.ends_with("<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>"), "{}", p.text);
        let p = plan_response(&format!("{tools}<|im_start|>user\n<tool_response>\n18 C\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"));
        assert!(p.text.ends_with("Based on the tool result: 18 C"), "{}", p.text);
    }
}
