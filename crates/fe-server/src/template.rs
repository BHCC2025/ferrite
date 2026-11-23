//! Chat-template rendering with minijinja, matching HF `apply_chat_template` byte for byte.
//!
//! HF renders with jinja2 `trim_blocks=True, lstrip_blocks=True`, no autoescape, `tojson` =
//! `json.dumps(ensure_ascii=False)`, `raise_exception(msg)`, plus the tokenizer's special tokens as
//! variables. Python string methods (`startswith`, `strip`, ...) come from minijinja-contrib's pycompat.
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use minijinja::value::{Kwargs, Serde};
use minijinja::{AutoEscape, Environment, Error, ErrorKind, Value};
use serde_json::{Map, Value as Json};

use crate::pyjson;

const TEMPLATE_NAME: &str = "chat_template";

pub struct ChatTemplate {
    /// The template writes DeepSeek DSML tool calls (`<｜DSML｜ calls>`)
    pub dsml: bool,
    /// The template writes Inkling-2 typed messages (`<|content_thinking|>`, `<|content_invoke_tool_json|>`): the reply is read by `crate::inkling`
    pub inkling: bool,
    env: Environment<'static>,
    /// Special-token variables HF passes to every render (`bos_token`, `eos_token`, ...).
    special_tokens: Map<String, Json>,
}

impl ChatTemplate {
    /// `chat_template.jinja` from the model dir, else `chat_template` in `tokenizer_config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config: Json = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Json::Null);
        let source = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(s) => s,
            Err(_) => match config.get("chat_template") {
                Some(Json::String(s)) => s.clone(),
                _ => return Err(anyhow!("no chat_template.jinja or tokenizer_config.json chat_template in {}", dir.display())),
            },
        };
        let mut special_tokens = Map::new();
        for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
            let v = match config.get(key) {
                Some(Json::String(s)) => Json::String(s.clone()),
                Some(Json::Object(o)) => o.get("content").cloned().unwrap_or(Json::Null),
                _ => Json::Null,
            };
            special_tokens.insert(key.to_string(), v);
        }
        Self::from_source(source, special_tokens)
    }

    pub fn from_source(source: String, special_tokens: Map<String, Json>) -> Result<Self> {
        // HF training-mask tags (`{% generation %}`): they only mark the assistant spans for loss masks and render nothing
        let source = ["{%- generation -%}", "{%- endgeneration -%}", "{% generation %}", "{% endgeneration %}"]
            .iter().fold(source, |s, t| s.replace(t, ""));
        let mut env = Environment::new();
        let syntax = minijinja::syntax::SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .map_err(|e| anyhow!("syntax config: {e}"))?;
        env.set_syntax(syntax);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_filter("tojson", tojson);
        env.add_function("strftime_now", strftime_now);
        let dsml = source.contains("｜DSML｜");
        let inkling = source.contains("<|content_thinking|>") && source.contains("<|content_invoke_tool_json|>");
        env.add_template_owned(TEMPLATE_NAME, source).context("compiling chat template")?;
        Ok(Self { dsml, inkling, env, special_tokens })
    }

    /// Render a conversation. `messages` / `tools` are OpenAI-format JSON, already normalized
    /// (see [`normalize_messages`]); `extra` are `chat_template_kwargs` (e.g. `enable_thinking`).
    pub fn render(
        &self,
        messages: &[Json],
        tools: Option<&[Json]>,
        add_generation_prompt: bool,
        extra: &Map<String, Json>,
    ) -> Result<String, TemplateError> {
        let mut ctx = self.special_tokens.clone();
        for (k, v) in extra {
            ctx.insert(k.clone(), v.clone());
        }
        ctx.insert("messages".into(), Json::Array(messages.to_vec()));
        match tools {
            Some(t) if !t.is_empty() => {
                ctx.insert("tools".into(), Json::Array(t.to_vec()));
            }
            _ => {
                ctx.insert("tools".into(), Json::Null);
            }
        }
        ctx.insert("add_generation_prompt".into(), Json::Bool(add_generation_prompt));
        let tmpl = self.env.get_template(TEMPLATE_NAME).map_err(|e| TemplateError(e.to_string()))?;
        tmpl.render(Value::from(Serde(&Json::Object(ctx)))).map_err(|e| {
            // raise_exception messages come back verbatim; other errors with their detail
            let mut msg = e.detail().map(str::to_string).unwrap_or_else(|| e.to_string());
            if e.kind() != ErrorKind::InvalidOperation {
                msg = format!("chat template error: {e}");
            }
            TemplateError(msg)
        })
    }
}

/// A template failure, reported to the client as a 400.
#[derive(Debug, Clone)]
pub struct TemplateError(pub String);

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn raise_exception(msg: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

fn tojson(value: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    // HF's `tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)` = json.dumps with those arguments
    let indent: Option<usize> = kwargs.get("indent")?;
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let separators: Option<Value> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    let separators = match separators {
        Some(v) if !v.is_none() => {
            let items: Vec<String> = v
                .try_iter()
                .map_err(|_| Error::new(ErrorKind::InvalidOperation, "tojson: separators must be a pair of strings"))?
                .map(|x| x.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "tojson: separators must be a pair of strings"))?;
            if items.len() != 2 {
                return Err(Error::new(ErrorKind::InvalidOperation, "tojson: separators must be a pair of strings"));
            }
            Some((items[0].clone(), items[1].clone()))
        }
        _ => None,
    };
    let json = serde_json::to_value(value)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
    let opts = pyjson::Opts { indent, separators, sort_keys: sort_keys.unwrap_or(false), ensure_ascii: ensure_ascii.unwrap_or(false) };
    Ok(Value::from_safe_string(pyjson::dumps_with(&json, &opts)))
}

fn strftime_now(fmt: String) -> Result<Value, Error> {
    // Only the formats chat templates use in practice ("%d %b %Y", "%Y-%m-%d").
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const FULL: [&str; 12] = [
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
        "November", "December",
    ];
    let out = fmt
        .replace("%Y", &format!("{y:04}"))
        .replace("%m", &format!("{m:02}"))
        .replace("%d", &format!("{d:02}"))
        .replace("%b", MONTHS[(m - 1) as usize])
        .replace("%B", FULL[(m - 1) as usize]);
    Ok(Value::from(out))
}

/// Days since 1970-01-01 to (year, month, day), Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Bring OpenAI-format messages into the shape HF chat templates expect (what vLLM does):
/// - assistant `tool_calls[].function.arguments` JSON strings are parsed into objects
///   (invalid JSON becomes `{}`), empty `tool_calls` lists are dropped;
/// - an assistant `reasoning` field is accepted as `reasoning_content`;
/// - the `developer` role becomes `system`.
///
/// - non-text content parts (images, audio, ...) become a text placeholder: the model is text-only
///   and an agent loop should keep going (e.g. a tool that returned a screenshot) instead of failing.
pub fn normalize_messages(messages: &[Json]) -> Result<Vec<Json>, String> {
    let mut out = Vec::with_capacity(messages.len());
    for (i, m) in messages.iter().enumerate() {
        let Json::Object(obj) = m else {
            return Err(format!("messages[{i}] must be an object"));
        };
        let mut obj = obj.clone();
        match obj.get("role").and_then(Json::as_str) {
            None => return Err(format!("messages[{i}].role is required")),
            Some("developer") => {
                obj.insert("role".into(), Json::String("system".into()));
            }
            Some("system" | "user" | "assistant" | "tool") => {}
            Some(r) => return Err(format!("messages[{i}].role '{r}' is not one of system, developer, user, assistant, tool")),
        }
        if let Some(Json::Array(parts)) = obj.get_mut("content") {
            for p in parts.iter_mut() {
                let ty = p.get("type").and_then(Json::as_str).unwrap_or("");
                let is_text = p.get("text").is_some_and(Json::is_string) && (ty == "text" || ty.is_empty());
                if !is_text {
                    let kind = if ty.is_empty() { "content" } else { ty.trim_end_matches("_url") };
                    tracing::warn!("messages[{i}]: replacing a '{ty}' content part with a placeholder (text-only model)");
                    *p = serde_json::json!({"type": "text", "text": format!("[{kind} omitted: this model only reads text]")});
                }
            }
        }
        if obj.get("role").and_then(Json::as_str) == Some("assistant") {
            if !obj.contains_key("reasoning_content") {
                if let Some(r @ Json::String(_)) = obj.get("reasoning").cloned() {
                    obj.insert("reasoning_content".into(), r);
                }
            }
            let mut drop_calls = false;
            if let Some(calls) = obj.get_mut("tool_calls") {
                match calls {
                    Json::Array(list) if list.is_empty() => drop_calls = true,
                    Json::Array(list) => {
                        for call in list.iter_mut() {
                            let Some(f) = call.get_mut("function").and_then(Json::as_object_mut) else {
                                return Err(format!("messages[{i}].tool_calls entries need a 'function' object"));
                            };
                            let parsed = match f.get("arguments") {
                                Some(Json::String(s)) if s.trim().is_empty() => Json::Object(Map::new()),
                                Some(Json::String(s)) => match serde_json::from_str::<Json>(s) {
                                    Ok(v @ Json::Object(_)) => v,
                                    _ => Json::Object(Map::new()),
                                },
                                Some(v @ Json::Object(_)) => v.clone(),
                                _ => Json::Object(Map::new()),
                            };
                            f.insert("arguments".into(), parsed);
                        }
                    }
                    Json::Null => drop_calls = true,
                    _ => return Err(format!("messages[{i}].tool_calls must be a list")),
                }
            }
            if drop_calls {
                obj.remove("tool_calls");
            }
        }
        out.push(Json::Object(obj));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn civil_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_760), (2026, 11, 3));
    }

    #[test]
    fn normalizes_tool_call_arguments() {
        let msgs = vec![json!({"role": "assistant", "content": null, "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}},
            {"id": "d", "type": "function", "function": {"name": "g", "arguments": "not json"}}]}),
            json!({"role": "assistant", "content": "x", "tool_calls": []}),
            json!({"role": "developer", "content": "d"})];
        let out = normalize_messages(&msgs).unwrap();
        assert_eq!(out[0]["tool_calls"][0]["function"]["arguments"], json!({"a": 1}));
        assert_eq!(out[0]["tool_calls"][1]["function"]["arguments"], json!({}));
        assert!(out[1].get("tool_calls").is_none());
        assert_eq!(out[2]["role"], "system");
        let out = normalize_messages(&[json!({"role": "user", "content": [{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {}}]})]).unwrap();
        assert_eq!(out[0]["content"][1], json!({"type": "text", "text": "[image omitted: this model only reads text]"}));
    }
}
