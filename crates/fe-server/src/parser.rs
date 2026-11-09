//! Streaming split of model output into reasoning, content and tool calls (Qwen3 format).
//!
//! Follows vLLM's qwen3 parser: with thinking on, output starts as reasoning and `</think>` switches to
//! content; `<tool_call>` (in content, or in reasoning as an implicit end of thinking) opens a tool call
//! in the `<function=NAME><parameter=..>..</parameter></function>` format; a bare `<function=` is
//! accepted too, and so is a JSON body (`{"name": .., "arguments": {..}}`) inside `<tool_call>`.
//!
//! Text is fed in arbitrary chunks; anything that could still be the start of a marker is held back.
//! Leading whitespace of reasoning / content and whitespace right before a marker are dropped.
use crate::toolcall::{ToolSet, convert_arguments};
use crate::util::random_id;

#[derive(Clone, Debug, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    /// A tool call began (its name is known). `index` counts calls in this response from 0.
    ToolStart { index: usize, id: String, name: String },
    /// The complete JSON arguments of call `index` (sent once, when the call closes).
    ToolArgs { index: usize, arguments: String },
}

#[derive(Clone, Debug, Default)]
pub struct ParserConfig {
    /// Output begins inside `<think>` (the generation prompt ended with `<think>\n`).
    pub thinking: bool,
    /// Parse tool calls against these tools; `None` = tool calls are plain content.
    pub tools: Option<ToolSet>,
    /// `parallel_tool_calls: false` → `Some(1)`: later calls are parsed but not reported.
    pub max_tool_calls: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
enum Mode {
    Reasoning,
    Content,
    /// Inside `<tool_call>`, before `<function=`.
    Preamble,
    /// After `<function=`, before `>`.
    Name,
    /// Inside `<function=NAME>` until `</function>`.
    Args { index: usize, name: String, emit: bool },
    /// After `</function>`: `</tool_call>`, another `<function=` or a new `<tool_call>` may follow.
    Between,
}

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_START: &str = "<tool_call>";
const TOOL_END: &str = "</tool_call>";
const FUNC_START: &str = "<function=";
const FUNC_END: &str = "</function>";

pub struct OutputParser {
    mode: Mode,
    buf: String,
    tools: Option<ToolSet>,
    max_calls: Option<usize>,
    calls: usize,
    reported_calls: usize,
    reasoning_ws: String,
    content_ws: String,
    reasoning_started: bool,
    content_started: bool,
    /// Raw text consumed since `<tool_call>` opened, given back as content if it turns out not to be a call.
    raw_tool: String,
}

impl OutputParser {
    pub fn new(cfg: ParserConfig) -> Self {
        Self {
            mode: if cfg.thinking { Mode::Reasoning } else { Mode::Content },
            buf: String::new(),
            tools: cfg.tools,
            max_calls: cfg.max_tool_calls,
            calls: 0,
            reported_calls: 0,
            reasoning_ws: String::new(),
            content_ws: String::new(),
            reasoning_started: false,
            content_started: false,
            raw_tool: String::new(),
        }
    }

    /// Number of tool calls reported so far.
    pub fn tool_calls(&self) -> usize {
        self.reported_calls
    }

    pub fn feed(&mut self, text: &str) -> Vec<Delta> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        while self.step(&mut out, false) {}
        merge(out)
    }

    /// End of output: flush everything held back.
    pub fn finish(&mut self) -> Vec<Delta> {
        let mut out = Vec::new();
        while self.step(&mut out, true) {}
        if !self.buf.is_empty() {
            let rest = std::mem::take(&mut self.buf);
            match self.mode {
                Mode::Reasoning => self.emit_reasoning(&rest, &mut out),
                _ => self.emit_content(&rest, &mut out),
            }
        }
        if self.mode == Mode::Content && self.reported_calls == 0 && !self.content_ws.is_empty() {
            let ws = std::mem::take(&mut self.content_ws);
            out.push(Delta::Content(ws));
        }
        merge(out)
    }

    fn tools_on(&self) -> bool {
        self.tools.is_some()
    }

    /// One parsing step; returns true if it made progress.
    fn step(&mut self, out: &mut Vec<Delta>, eof: bool) -> bool {
        match self.mode.clone() {
            Mode::Reasoning => {
                let mut markers = vec![THINK_END, THINK_START];
                if self.tools_on() {
                    markers.push(TOOL_START);
                }
                match find_first(&self.buf, &markers) {
                    Some((i, m)) => {
                        let before = self.buf[..i].to_string();
                        self.buf.drain(..i + m.len());
                        self.emit_reasoning(&before, out);
                        match m {
                            THINK_END => {
                                self.reasoning_ws.clear();
                                self.mode = Mode::Content;
                            }
                            TOOL_START => {
                                self.reasoning_ws.clear();
                                self.raw_tool = TOOL_START.to_string();
                                self.mode = Mode::Preamble;
                            }
                            _ => {} // a stray <think>: drop it
                        }
                        true
                    }
                    None => {
                        let hold = if eof { 0 } else { holdback(&self.buf, &markers) };
                        let cut = self.buf.len() - hold;
                        if cut > 0 {
                            let text: String = self.buf.drain(..cut).collect();
                            self.emit_reasoning(&text, out);
                        }
                        false
                    }
                }
            }
            Mode::Content => {
                // the model may open thinking itself, but only as the very first thing it writes
                if !self.content_started && !self.reasoning_started {
                    let lead = self.buf.len() - self.buf.trim_start().len();
                    let t = &self.buf[lead..];
                    if t.starts_with(THINK_START) {
                        self.buf.drain(..lead + THINK_START.len());
                        self.mode = Mode::Reasoning;
                        return true;
                    }
                    if !eof && (t.is_empty() || THINK_START.starts_with(t)) {
                        return false;
                    }
                }
                let mut markers = vec![THINK_END];
                if self.tools_on() {
                    markers.push(TOOL_START);
                    markers.push(FUNC_START);
                }
                match find_first(&self.buf, &markers) {
                    Some((i, m)) => {
                        let before = self.buf[..i].to_string();
                        self.buf.drain(..i + m.len());
                        self.emit_content(&before, out);
                        match m {
                            TOOL_START => {
                                self.raw_tool = std::mem::take(&mut self.content_ws) + TOOL_START;
                                self.mode = Mode::Preamble;
                            }
                            FUNC_START => {
                                self.content_ws.clear();
                                self.mode = Mode::Name;
                            }
                            _ => {} // a duplicate </think>: drop it
                        }
                        true
                    }
                    None => {
                        let hold = if eof { 0 } else { holdback(&self.buf, &markers) };
                        let cut = self.buf.len() - hold;
                        if cut > 0 {
                            let text: String = self.buf.drain(..cut).collect();
                            self.emit_content(&text, out);
                        }
                        false
                    }
                }
            }
            Mode::Preamble => {
                let ws = self.buf.len() - self.buf.trim_start().len();
                let drained: String = self.buf.drain(..ws).collect();
                self.raw_tool.push_str(&drained);
                if self.buf.starts_with(FUNC_START) {
                    self.buf.drain(..FUNC_START.len());
                    self.raw_tool.clear();
                    self.mode = Mode::Name;
                    return true;
                }
                if self.buf.starts_with(TOOL_END) {
                    self.buf.drain(..TOOL_END.len());
                    self.raw_tool.clear();
                    self.mode = Mode::Content;
                    return true;
                }
                if !eof && (self.buf.is_empty() || FUNC_START.starts_with(&self.buf) || TOOL_END.starts_with(&self.buf)) {
                    return false;
                }
                // not the XML format: maybe a JSON body, up to </tool_call>
                let (body, consumed, closed) = match self.buf.find(TOOL_END) {
                    Some(i) => (self.buf[..i].to_string(), i + TOOL_END.len(), true),
                    None if eof => (self.buf.clone(), self.buf.len(), false),
                    None => return false,
                };
                self.buf.drain(..consumed);
                self.mode = Mode::Content;
                let raw = std::mem::take(&mut self.raw_tool);
                if !self.emit_json_call(&body, out) {
                    // not a call after all: give the text back unchanged
                    let end = if closed { TOOL_END } else { "" };
                    self.emit_content(&format!("{raw}{body}{end}"), out);
                }
                true
            }
            Mode::Name => match self.buf.find('>') {
                Some(i) => {
                    let name = self.buf[..i].trim().to_string();
                    self.buf.drain(..=i);
                    let index = self.calls;
                    self.calls += 1;
                    let emit = self.max_calls.is_none_or(|m| self.reported_calls < m) && !name.is_empty();
                    if emit {
                        self.reported_calls += 1;
                        out.push(Delta::ToolStart { index: self.reported_calls - 1, id: random_id("call_", 24), name: name.clone() });
                    }
                    let index = if emit { self.reported_calls - 1 } else { index };
                    self.mode = Mode::Args { index, name, emit };
                    true
                }
                None => {
                    if eof {
                        self.buf.clear(); // a cut-off function name: nothing usable
                        self.mode = Mode::Content;
                        return true;
                    }
                    false
                }
            },
            Mode::Args { index, name, emit } => {
                let end = find_first(&self.buf, &[FUNC_END, TOOL_END]);
                let (raw, consumed, next) = match end {
                    Some((i, FUNC_END)) => (self.buf[..i].to_string(), i + FUNC_END.len(), Mode::Between),
                    Some((i, _)) => (self.buf[..i].to_string(), i + TOOL_END.len(), Mode::Content),
                    None if eof => (self.buf.clone(), self.buf.len(), Mode::Content),
                    None => return false,
                };
                self.buf.drain(..consumed);
                if emit {
                    let props = self.tools.as_ref().and_then(|t| t.properties(&name));
                    out.push(Delta::ToolArgs { index, arguments: convert_arguments(&raw, props) });
                }
                self.mode = next;
                true
            }
            Mode::Between => {
                let ws = self.buf.len() - self.buf.trim_start().len();
                self.buf.drain(..ws);
                for (m, next) in [(TOOL_END, Mode::Content), (FUNC_START, Mode::Name), (TOOL_START, Mode::Preamble)] {
                    if self.buf.starts_with(m) {
                        self.buf.drain(..m.len());
                        self.mode = next;
                        return true;
                    }
                }
                if !eof
                    && (self.buf.is_empty()
                        || [TOOL_END, FUNC_START, TOOL_START].iter().any(|m| m.starts_with(self.buf.as_str())))
                {
                    return false;
                }
                self.mode = Mode::Content;
                !self.buf.is_empty()
            }
        }
    }

    /// A hermes-style `{"name": "...", "arguments": {...}}` body; true if it was one.
    fn emit_json_call(&mut self, body: &str, out: &mut Vec<Delta>) -> bool {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(body.trim()) else { return false };
        let Some(name) = v.get("name").and_then(|n| n.as_str()) else { return false };
        self.calls += 1;
        if self.max_calls.is_some_and(|m| self.reported_calls >= m) {
            return true;
        }
        let args = match v.get("arguments").or_else(|| v.get("parameters")) {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(a) => crate::pyjson::dumps(a, None),
            None => "{}".to_string(),
        };
        let index = self.reported_calls;
        self.reported_calls += 1;
        out.push(Delta::ToolStart { index, id: random_id("call_", 24), name: name.to_string() });
        out.push(Delta::ToolArgs { index, arguments: args });
        true
    }

    fn emit_reasoning(&mut self, text: &str, out: &mut Vec<Delta>) {
        if let Some(s) = emit_trimmed(text, &mut self.reasoning_ws, &mut self.reasoning_started) {
            out.push(Delta::Reasoning(s));
        }
    }

    fn emit_content(&mut self, text: &str, out: &mut Vec<Delta>) {
        if let Some(s) = emit_trimmed(text, &mut self.content_ws, &mut self.content_started) {
            out.push(Delta::Content(s));
        }
    }
}

/// Append `text` to the held-back whitespace; return everything up to the trailing whitespace
/// (which stays held back), dropping leading whitespace before the first visible character.
fn emit_trimmed(text: &str, ws: &mut String, started: &mut bool) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    let mut s = std::mem::take(ws);
    s.push_str(text);
    if !*started {
        let lead = s.len() - s.trim_start().len();
        s.drain(..lead);
    }
    let body = s.trim_end().len();
    *ws = s.split_off(body);
    if s.is_empty() {
        return None;
    }
    *started = true;
    Some(s)
}

/// Earliest occurrence of any marker (ties: the longer marker).
fn find_first<'a>(buf: &str, markers: &[&'a str]) -> Option<(usize, &'a str)> {
    let mut best: Option<(usize, &'a str)> = None;
    for m in markers {
        if let Some(i) = buf.find(m) {
            if best.is_none_or(|(b, bm)| i < b || (i == b && m.len() > bm.len())) {
                best = Some((i, m));
            }
        }
    }
    best
}

/// Length of the longest suffix of `buf` that is a proper prefix of a marker.
fn holdback(buf: &str, markers: &[&str]) -> usize {
    let mut best = 0;
    for m in markers {
        let max = (m.len() - 1).min(buf.len());
        for k in (best + 1..=max).rev() {
            let start = buf.len() - k;
            if buf.is_char_boundary(start) && m.as_bytes()[..k] == buf.as_bytes()[start..] {
                best = k;
                break;
            }
        }
    }
    best
}

/// Merge adjacent reasoning / content deltas.
fn merge(deltas: Vec<Delta>) -> Vec<Delta> {
    let mut out: Vec<Delta> = Vec::with_capacity(deltas.len());
    for d in deltas {
        match (out.last_mut(), d) {
            (Some(Delta::Reasoning(a)), Delta::Reasoning(b)) => a.push_str(&b),
            (Some(Delta::Content(a)), Delta::Content(b)) => a.push_str(&b),
            (_, d) => out.push(d),
        }
    }
    out
}

/// Everything a parser produced, accumulated (non-streaming responses and tests).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Parsed {
    pub reasoning: String,
    pub content: String,
    /// (id, name, arguments)
    pub tool_calls: Vec<(String, String, String)>,
}

impl Parsed {
    pub fn absorb(&mut self, deltas: &[Delta]) {
        for d in deltas {
            match d {
                Delta::Reasoning(s) => self.reasoning.push_str(s),
                Delta::Content(s) => self.content.push_str(s),
                Delta::ToolStart { id, name, .. } => self.tool_calls.push((id.clone(), name.clone(), String::new())),
                Delta::ToolArgs { index, arguments } => {
                    if let Some(c) = self.tool_calls.get_mut(*index) {
                        c.2 = arguments.clone();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> ToolSet {
        ToolSet::from_openai(&[
            json!({"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
                "properties": {"location": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}}}}}),
            json!({"type": "function", "function": {"name": "search", "parameters": {"type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "number"}}}}}),
        ])
    }

    /// Parse `text` whole, char by char and in 3-byte-ish chunks; all must agree.
    fn parse(cfg: ParserConfig, text: &str) -> Parsed {
        let mut results = Vec::new();
        for chunk in [usize::MAX, 1, 3, 7] {
            let mut p = OutputParser::new(cfg.clone());
            let mut acc = Parsed::default();
            let chars: Vec<char> = text.chars().collect();
            let step = chunk.min(chars.len().max(1));
            for piece in chars.chunks(step) {
                let s: String = piece.iter().collect();
                acc.absorb(&p.feed(&s));
            }
            acc.absorb(&p.finish());
            for c in acc.tool_calls.iter_mut() {
                assert!(c.0.starts_with("call_") && c.0.len() == 29, "{}", c.0);
                c.0 = "id".into();
            }
            results.push(acc);
        }
        for r in &results[1..] {
            assert_eq!(r, &results[0], "chunking changed the result for {text:?}");
        }
        results.remove(0)
    }

    fn cfg(thinking: bool, tools_on: bool) -> ParserConfig {
        ParserConfig { thinking, tools: tools_on.then(tools), max_tool_calls: None }
    }

    fn call(name: &str, args: serde_json::Value) -> (String, String, String) {
        ("id".into(), name.into(), crate::pyjson::dumps(&args, None))
    }

    #[test]
    fn reasoning_then_content() {
        let p = parse(cfg(true, false), "Let me think.\nOK.\n</think>\n\nThe answer is 4.");
        assert_eq!(p.reasoning, "Let me think.\nOK.");
        assert_eq!(p.content, "The answer is 4.");
        assert!(p.tool_calls.is_empty());
    }

    #[test]
    fn thinking_off_is_all_content() {
        let p = parse(cfg(false, false), "Hello </think> there");
        assert_eq!(p.reasoning, "");
        assert_eq!(p.content, "Hello  there");
    }

    #[test]
    fn unterminated_reasoning_stays_reasoning() {
        let p = parse(cfg(true, false), "thinking forever and ever");
        assert_eq!(p.reasoning, "thinking forever and ever");
        assert_eq!(p.content, "");
    }

    #[test]
    fn model_opened_think_itself() {
        let p = parse(cfg(false, false), "<think>\nhmm\n</think>\n\nanswer");
        assert_eq!(p.reasoning, "hmm");
        assert_eq!(p.content, "answer");
    }

    #[test]
    fn content_keeps_inner_whitespace_and_angle_brackets() {
        let p = parse(cfg(true, false), "r</think>\n\n```rust\nfn a() -> Vec<u8> {\n    vec![]\n}\n```\n");
        assert_eq!(p.content, "```rust\nfn a() -> Vec<u8> {\n    vec![]\n}\n```\n");
        let p = parse(cfg(false, true), "a <tool and <function but not a call");
        assert_eq!(p.content, "a <tool and <function but not a call");
    }

    #[test]
    fn single_tool_call() {
        let text = "I should check.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let p = parse(cfg(true, true), text);
        assert_eq!(p.reasoning, "I should check.");
        assert_eq!(p.content, "");
        assert_eq!(p.tool_calls, vec![call("get_weather", json!({"location": "Paris", "days": 3}))]);
    }

    #[test]
    fn text_before_and_parallel_calls() {
        let text = "Let me look both up.\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nOslo\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=search>\n<parameter=query>\nbest café in Oslo\n</parameter>\n<parameter=limit>\n2.5\n</parameter>\n</function>\n</tool_call>";
        let p = parse(cfg(false, true), text);
        assert_eq!(p.content, "Let me look both up.");
        assert_eq!(
            p.tool_calls,
            vec![
                call("get_weather", json!({"location": "Oslo", "metric": true})),
                call("search", json!({"query": "best café in Oslo", "limit": 2.5})),
            ]
        );
    }

    #[test]
    fn tool_call_inside_reasoning_ends_reasoning() {
        let p = parse(cfg(true, true), "need weather\n<tool_call>\n<function=get_weather>\n<parameter=location>\nRome\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(p.reasoning, "need weather");
        assert_eq!(p.tool_calls, vec![call("get_weather", json!({"location": "Rome"}))]);
    }

    #[test]
    fn tools_off_leaves_markup_in_content() {
        let text = "<tool_call>\n<function=search>\n<parameter=query>\nx\n</parameter>\n</function>\n</tool_call>";
        let p = parse(cfg(false, false), text);
        assert_eq!(p.content, text);
        assert!(p.tool_calls.is_empty());
    }

    #[test]
    fn bare_function_and_two_functions_in_one_block() {
        let p = parse(cfg(false, true), "<function=search>\n<parameter=query>\na\n</parameter>\n</function>\nmore text");
        assert_eq!(p.tool_calls, vec![call("search", json!({"query": "a"}))]);
        assert_eq!(p.content, "more text");
        let p = parse(
            cfg(false, true),
            "<tool_call>\n<function=search>\n<parameter=query>\na\n</parameter>\n</function>\n<function=search>\n<parameter=query>\nb\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(p.tool_calls, vec![call("search", json!({"query": "a"})), call("search", json!({"query": "b"}))]);
    }

    #[test]
    fn cut_off_tool_call_reports_partial_arguments() {
        let p = parse(cfg(false, true), "<tool_call>\n<function=search>\n<parameter=query>\nhalf a que");
        assert_eq!(p.tool_calls, vec![call("search", json!({"query": "half a que"}))]);
    }

    #[test]
    fn json_body_tool_call() {
        let p = parse(cfg(false, true), "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": \"q\"}}\n</tool_call>");
        assert_eq!(p.tool_calls, vec![call("search", json!({"query": "q"}))]);
    }

    #[test]
    fn parallel_calls_off_keeps_first() {
        let text = "<tool_call>\n<function=search>\n<parameter=query>\na\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=search>\n<parameter=query>\nb\n</parameter>\n</function>\n</tool_call>";
        let p = parse(ParserConfig { thinking: false, tools: Some(tools()), max_tool_calls: Some(1) }, text);
        assert_eq!(p.tool_calls, vec![call("search", json!({"query": "a"}))]);
    }

    #[test]
    fn think_only_opens_at_the_start_of_content() {
        // regardless of chunking: later <think> is text (a later </think> is still dropped)
        let p = parse(cfg(false, false), "OK<think>x</think>y");
        assert_eq!((p.reasoning.as_str(), p.content.as_str()), ("", "OK<think>xy"));
        let p = parse(cfg(false, false), "  <think>x</think>y");
        assert_eq!((p.reasoning.as_str(), p.content.as_str()), ("x", "y"));
    }

    #[test]
    fn non_call_tool_block_is_given_back_verbatim() {
        let p = parse(cfg(false, true), "Use <tool_call>\nnot json here</tool_call> ok");
        assert_eq!(p.content, "Use <tool_call>\nnot json here</tool_call> ok");
        assert!(p.tool_calls.is_empty());
        let p = parse(cfg(false, true), "Use <tool_call>\nunclosed");
        assert_eq!(p.content, "Use <tool_call>\nunclosed");
    }

    #[test]
    fn fuzz_chunking_never_changes_the_result() {
        const PIECES: &[&str] = &[
            "hi", " ", "\n", "é🦀", "<", ">", "<think>", "</think>", "<tool_call>", "</tool_call>",
            "<function=search>", "<function=get_weather>", "<parameter=query>", "<parameter=days>",
            "</parameter>", "</function>", "{\"name\": \"search\", \"arguments\": {}}", "3", "<tool", "</fun",
        ];
        let mut rng = crate::sampler::Rng::new(3);
        for case in 0..3000 {
            let n = 1 + (rng.next_u64() % 14) as usize;
            let text: String = (0..n).map(|_| PIECES[(rng.next_u64() % PIECES.len() as u64) as usize]).collect();
            let c = match case % 4 {
                0 => cfg(true, true),
                1 => cfg(false, true),
                2 => cfg(true, false),
                _ => cfg(false, false),
            };
            parse(c.clone(), &text); // whole vs 1 / 3 / 7 char chunks
            // and random chunk sizes
            let whole = parse(c.clone(), &text);
            let chars: Vec<char> = text.chars().collect();
            let mut p = OutputParser::new(c);
            let mut acc = Parsed::default();
            let mut i = 0;
            while i < chars.len() {
                let k = 1 + (rng.next_u64() % 5) as usize;
                let piece: String = chars[i..(i + k).min(chars.len())].iter().collect();
                acc.absorb(&p.feed(&piece));
                i += k;
            }
            acc.absorb(&p.finish());
            for c in acc.tool_calls.iter_mut() {
                c.0 = "id".into();
            }
            assert_eq!(acc, whole, "random chunking changed the result for {text:?}");
        }
    }

    #[test]
    fn streaming_emits_name_before_arguments() {
        let mut p = OutputParser::new(cfg(false, true));
        let d = p.feed("Checking.\n\n<tool_call>\n<function=search>\n<parameter=query>\nab");
        assert!(matches!(&d[..], [Delta::Content(c), Delta::ToolStart { index: 0, name, .. }] if c == "Checking." && name == "search"), "{d:?}");
        let d = p.feed("c\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(d, vec![Delta::ToolArgs { index: 0, arguments: "{\"query\": \"abc\"}".into() }]);
        assert!(p.finish().is_empty());
    }
}
