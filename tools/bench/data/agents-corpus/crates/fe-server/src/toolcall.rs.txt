//! Qwen3 XML tool calls → OpenAI `tool_calls`, with schema-typed arguments (as vLLM's qwen3 parser).
//!
//! The model writes
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=ARG>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//! Values are raw text; each is converted using the tool's JSON-schema type for that parameter
//! (integer / number / boolean / null / object / array are parsed, strings stay strings).
use serde_json::{Map, Value as Json};

use crate::pyjson;

/// The function tools offered in a request: name → `parameters.properties`.
#[derive(Clone, Debug, Default)]
pub struct ToolSet {
    tools: Vec<(String, Map<String, Json>)>,
}

impl ToolSet {
    /// From OpenAI `tools` (`[{"type": "function", "function": {"name", "parameters"}}]`).
    pub fn from_openai(tools: &[Json]) -> Self {
        let mut out = Vec::new();
        for t in tools {
            let f = t.get("function").unwrap_or(t);
            let Some(name) = f.get("name").and_then(Json::as_str) else { continue };
            let props = f
                .get("parameters")
                .and_then(|p| p.get("properties"))
                .and_then(Json::as_object)
                .cloned()
                .unwrap_or_default();
            out.push((name.to_string(), props));
        }
        Self { tools: out }
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|(n, _)| n.as_str())
    }

    pub fn properties(&self, name: &str) -> Option<&Map<String, Json>> {
        self.tools.iter().find(|(n, _)| n == name).map(|(_, p)| p)
    }
}

/// Parse the body of a `<function=NAME>` block (everything between `>` and `</function>`) into the
/// JSON arguments string. `partial` = the block was cut off (end of generation).
pub fn convert_arguments(raw: &str, tool: Option<&Map<String, Json>>) -> String {
    let mut args = Map::new();
    for (name, value) in parse_parameters(raw) {
        let v = match tool.and_then(|p| p.get(&name)) {
            Some(schema) => coerce_to_schema_type(&value, &extract_types_from_schema(schema)),
            None => Json::String(value),
        };
        args.insert(name, v);
    }
    pyjson::dumps(&Json::Object(args), None)
}

/// `<parameter=NAME>VALUE</parameter>` pairs; a parameter also ends at the next `<parameter=` or at
/// the end of the text (cut-off output). One leading and one trailing newline are template markup.
pub fn parse_parameters(raw: &str) -> Vec<(String, String)> {
    const START: &str = "<parameter=";
    const END: &str = "</parameter>";
    let mut out: Vec<(String, String)> = Vec::new();
    let mut rest = raw;
    while let Some(s) = rest.find(START) {
        let after = &rest[s + START.len()..];
        let Some(gt) = after.find('>') else { break };
        let name = after[..gt].trim().to_string();
        let body = &after[gt + 1..];
        let end = body.find(END);
        let next = body.find(START);
        let (value, consumed) = match (end, next) {
            (Some(e), Some(n)) if n < e => (&body[..n], n),
            (Some(e), _) => (&body[..e], e + END.len()),
            (None, Some(n)) => (&body[..n], n),
            (None, None) => (body, body.len()),
        };
        if !name.is_empty() {
            let v = trim_wrapping_newlines(value).to_string();
            if let Some(slot) = out.iter_mut().find(|(n, _)| *n == name) {
                slot.1 = v;
            } else {
                out.push((name, v));
            }
        }
        rest = &body[consumed..];
    }
    out
}

fn trim_wrapping_newlines(v: &str) -> &str {
    let v = v.strip_prefix('\n').unwrap_or(v);
    v.strip_suffix('\n').unwrap_or(v)
}

/// All JSON-schema type names a schema allows (`type`, `enum` inference, `anyOf`/`oneOf`/`allOf`);
/// `["string"]` when nothing is declared.
pub fn extract_types_from_schema(schema: &Json) -> Vec<String> {
    let mut types: Vec<String> = Vec::new();
    let add = |t: &str, types: &mut Vec<String>| {
        if !types.iter().any(|x| x == t) {
            types.push(t.to_string());
        }
    };
    let Json::Object(s) = schema else { return vec!["string".into()] };
    match s.get("type") {
        Some(Json::String(t)) => add(t, &mut types),
        Some(Json::Array(ts)) => {
            for t in ts.iter().filter_map(Json::as_str) {
                add(t, &mut types);
            }
        }
        _ => {}
    }
    if let Some(Json::Array(vals)) = s.get("enum") {
        for v in vals {
            let t = match v {
                Json::Null => "null",
                Json::Bool(_) => "boolean",
                Json::Number(n) if n.is_i64() || n.is_u64() => "integer",
                Json::Number(_) => "number",
                Json::String(_) => "string",
                Json::Array(_) => "array",
                Json::Object(_) => "object",
            };
            add(t, &mut types);
        }
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(Json::Array(choices)) = s.get(key) {
            for c in choices {
                for t in extract_types_from_schema(c) {
                    add(&t, &mut types);
                }
            }
        }
    }
    if types.is_empty() { vec!["string".into()] } else { types }
}

fn normalize_type(t: &str) -> String {
    let t = t.trim().to_lowercase();
    match t.as_str() {
        "str" | "text" | "varchar" | "char" | "enum" => "string",
        "int" | "int32" | "int64" | "uint" | "uint32" | "uint64" | "long" | "short" | "unsigned" => "integer",
        "float" | "float32" | "float64" | "double" => "number",
        "bool" => "boolean",
        "dict" => "object",
        "arr" | "list" | "sequence" => "array",
        _ => return t,
    }
    .to_string()
}

/// Best-effort conversion of a raw value to one of `types`, in vLLM's priority order
/// null > integer > number > boolean > object > array > string; falls back to JSON, then the string.
pub fn coerce_to_schema_type(value: &str, types: &[String]) -> Json {
    let types: Vec<String> = types.iter().map(|t| normalize_type(t)).collect();
    let has = |t: &str| types.iter().any(|x| x == t);
    for cand in ["null", "integer", "number", "boolean", "object", "array", "string"] {
        if !has(cand) {
            continue;
        }
        match cand {
            "null" => {
                if value.eq_ignore_ascii_case("null") {
                    return Json::Null;
                }
            }
            "string" => return Json::String(value.to_string()),
            "integer" => {
                if let Some(i) = parse_py_int(value) {
                    return Json::from(i);
                }
            }
            "number" => {
                if let Some(f) = parse_py_float(value) {
                    if f.is_finite() {
                        if f == f.trunc() && f.abs() < 9.007_199_254_740_992e15 {
                            return Json::from(f as i64);
                        }
                        if let Some(n) = serde_json::Number::from_f64(f) {
                            return Json::Number(n);
                        }
                    }
                }
            }
            "boolean" => match value.trim().to_lowercase().as_str() {
                "true" | "1" => return Json::Bool(true),
                "false" | "0" => return Json::Bool(false),
                _ => {}
            },
            "object" | "array" => {
                if let Ok(v) = serde_json::from_str::<Json>(value) {
                    return v;
                }
            }
            _ => {}
        }
    }
    serde_json::from_str::<Json>(value).unwrap_or_else(|_| Json::String(value.to_string()))
}

/// Python `int(str)`: surrounding whitespace, optional sign, digits with single underscores.
fn parse_py_int(v: &str) -> Option<i64> {
    let t = v.trim();
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return None;
    }
    if !digits.chars().all(|c| c.is_ascii_digit() || c == '_') {
        return None;
    }
    t.replace('_', "").parse().ok()
}

/// Python `float(str)` for the usual spellings.
fn parse_py_float(v: &str) -> Option<f64> {
    let t = v.trim();
    if t.is_empty() || t.contains(char::is_whitespace) {
        return None;
    }
    let lower = t.to_lowercase();
    let body = lower.trim_start_matches(['+', '-']);
    if matches!(body, "inf" | "infinity" | "nan") {
        return None; // non-finite: treated as not a number
    }
    if !body.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | '+' | '-' | '_')) {
        return None;
    }
    t.replace('_', "").parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn search_props() -> Map<String, Json> {
        json!({
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
            "safe": {"type": "boolean"},
            "filters": {"type": "object"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "threshold": {"type": "number"},
            "maybe": {"type": ["string", "null"]},
            "choice": {"enum": [1, 2, 3]},
            "any": {"anyOf": [{"type": "integer"}, {"type": "string"}]}
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn typed_parameters() {
        let raw = "\n<parameter=query>\ncafés in Paris\n</parameter>\n<parameter=max_results>\n3\n</parameter>\n\
                   <parameter=safe>\nTrue\n</parameter>\n<parameter=filters>\n{\"site\": \"example.com\"}\n</parameter>\n\
                   <parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=threshold>\n0.5\n</parameter>\n\
                   <parameter=maybe>\nnull\n</parameter>\n<parameter=choice>\n2\n</parameter>\n\
                   <parameter=any>\nabc\n</parameter>\n<parameter=unknown>\n42\n</parameter>\n";
        let args = convert_arguments(raw, Some(&search_props()));
        let v: Json = serde_json::from_str(&args).unwrap();
        assert_eq!(
            v,
            json!({"query": "cafés in Paris", "max_results": 3, "safe": true, "filters": {"site": "example.com"},
                   "tags": ["a", "b"], "threshold": 0.5, "maybe": null, "choice": 2, "any": "abc", "unknown": "42"})
        );
        // Python json.dumps formatting
        assert!(args.starts_with("{\"query\": \"cafés in Paris\", \"max_results\": 3,"));
    }

    #[test]
    fn strings_stay_strings_and_keep_inner_newlines() {
        let raw = "\n<parameter=query>\nline 1\nline 2\n\n</parameter>\n<parameter=max_results>\n007\n</parameter>\n";
        let v: Json = serde_json::from_str(&convert_arguments(raw, Some(&search_props()))).unwrap();
        assert_eq!(v, json!({"query": "line 1\nline 2\n", "max_results": 7}));
    }

    #[test]
    fn number_and_fallbacks() {
        let t = |v: &str, ty: &[&str]| coerce_to_schema_type(v, &ty.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(t("3.0", &["number"]), json!(3));
        assert_eq!(t("2.5", &["number"]), json!(2.5));
        assert_eq!(t("abc", &["integer"]), json!("abc"));
        assert_eq!(t("[1,2]", &["integer"]), json!([1, 2]));
        assert_eq!(t("inf", &["number"]), json!("inf"));
        assert_eq!(t(" 12 ", &["int"]), json!(12));
        assert_eq!(t("yes", &["boolean"]), json!("yes"));
    }

    #[test]
    fn no_schema_keeps_strings() {
        let v: Json = serde_json::from_str(&convert_arguments("<parameter=a>\n1\n</parameter>", None)).unwrap();
        assert_eq!(v, json!({"a": "1"}));
    }

    #[test]
    fn unterminated_parameters() {
        let p = parse_parameters("<parameter=a>\nx\n<parameter=b>\ny");
        assert_eq!(p, vec![("a".into(), "x".into()), ("b".into(), "y".into())]);
    }
}
