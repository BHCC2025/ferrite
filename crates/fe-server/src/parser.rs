//! Streaming split of model output into reasoning, content and tool calls (Qwen3 format).
//!
//! Follows vLLM's qwen3 parser: with thinking on, output starts as reasoning and `</think>` switches to
//! content; `<tool_call>` (in content, or in reasoning as an implicit end of thinking) opens a tool call
//! in the `<function=NAME><parameter=..>..</parameter></function>` format; a bare `<function=` is
//! accepted too, and so is a JSON body (`{"name": .., "arguments": {..}}`) inside `<tool_call>`.
//!
//! Text is fed in arbitrary chunks; anything that could still be the start of a marker is held back.
//! Leading whitespace of reasoning / content and whitespace right before a marker are dropped.
use crate::toolcall::{ToolSet, coerce_to_schema_type, extract_types_from_schema, is_string_only};
use crate::util::random_id;

#[derive(Clone, Debug, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    /// A tool call began (its name is known). `index` counts calls in this response from 0.
    ToolStart { index: usize, id: String, name: String },
    /// A piece of the JSON arguments of call `index`; the pieces concatenate to the complete arguments (a long string value arrives in many pieces).
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
    /// DeepSeek DSML tool calls (`<｜DSML｜ calls>` ... `</｜DSML｜ calls>`) instead of the Qwen XML ones.
    pub dsml: bool,
    /// The prompt ended with a call opener (tool_choice required / named): the output starts inside the call.
    pub forced: Option<ForcedCall>,
}

/// What the generation prompt was prefilled with to force a tool call (Qwen XML format).
#[derive(Clone, Debug, PartialEq)]
pub enum ForcedCall {
    /// `<tool_call>\n<function=`: the output starts with the function name
    Any,
    /// `<tool_call>\n<function=NAME>\n`: the output starts with the parameters of NAME
    Named(String),
}

impl ForcedCall {
    /// The text appended to the generation prompt.
    pub fn opener(&self) -> String {
        match self {
            ForcedCall::Any => format!("{TOOL_START}\n{FUNC_START}"),
            ForcedCall::Named(n) => format!("{TOOL_START}\n{FUNC_START}{n}>\n"),
        }
    }
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
    /// Inside `<｜DSML｜ calls>`: between invokes.
    Dsml,
    /// Inside `<｜DSML｜ invoke name=..>` until its closing tag.
    DsmlInvoke { name: String },
}

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_START: &str = "<tool_call>";
const TOOL_END: &str = "</tool_call>";
const FUNC_START: &str = "<function=";
const FUNC_END: &str = "</function>";
const DS_CALLS: &str = "<｜DSML｜ calls>";
const DS_CALLS_END: &str = "</｜DSML｜ calls>";
const DS_INVOKE: &str = "<｜DSML｜ invoke name=\"";
const DS_INVOKE_END: &str = "</｜DSML｜ invoke>";
const DS_PARAM: &str = "<｜DSML｜ parameter name=\"";
const DS_PARAM_END: &str = "</｜DSML｜ parameter>";

/// JSON arguments of a DSML invoke body: `<｜DSML｜ parameter name="p" string="true|false">VALUE</｜DSML｜ parameter>` each; string="true" values are
/// strings as written, others are JSON (a value that does not parse stays a string).
#[cfg(test)]
fn dsml_arguments(body: &str) -> String {
    let mut map = serde_json::Map::new();
    let mut rest = body;
    while let Some(i) = rest.find(DS_PARAM) {
        rest = &rest[i + DS_PARAM.len()..];
        let Some(q) = rest.find('"') else { break };
        let pname = rest[..q].to_string();
        rest = &rest[q + 1..];
        let Some(gt) = rest.find('>') else { break };
        let attrs = rest[..gt].to_string();
        rest = &rest[gt + 1..];
        let Some(e) = rest.find(DS_PARAM_END) else { break };
        let raw = &rest[..e];
        rest = &rest[e + DS_PARAM_END.len()..];
        let is_string = !attrs.contains("string=\"false\"");
        let v = if is_string { serde_json::Value::String(raw.to_string()) } else { serde_json::from_str(raw.trim()).unwrap_or_else(|_| serde_json::Value::String(raw.to_string())) };
        map.insert(pname, v);
    }
    crate::pyjson::dumps(&serde_json::Value::Object(map), None)
}

/// Streaming state of one DSML invoke: its arguments JSON is emitted piece by piece (long string values stream as they are written).
/// How a DSML parameter value is delivered: streamed as a JSON string, or collected and typed at its end.
#[derive(Clone, Debug)]
enum Kind {
    Str,
    /// collected, then parsed as JSON; the schema types (empty = no schema: JSON if it parses, else string)
    Raw(Vec<String>),
}

#[derive(Default)]
struct Inv {
    /// schema of the tool being called
    tool: String,
    emit: bool,
    index: usize,
    nparams: usize,
    /// inside a parameter value
    val: Option<Kind>,
    /// a non-string value is collected whole and parsed at its closing tag
    raw: String,
    /// XML calls: the value's first character was seen (one newline after `>` is template markup)
    started: bool,
    /// XML calls: a parameter without a name, its value is dropped
    skip: bool,
    /// XML calls: something of the current value was consumed already (a tag at the buffer start is not at the value's start)
    consumed: bool,
    /// XML calls: call markup opened INSIDE the value and not yet closed (`<parameter=`, `<function=`, `<tool_call>`): text about
    /// tool calls being written as a value; the matching closing tags belong to the value too
    nest: [usize; 3],
}

/// Drop `,` before `]` / `}` outside strings.
fn strip_trailing_commas(t: &str) -> String {
    let b: Vec<char> = t.chars().collect();
    let mut out = String::with_capacity(t.len());
    let (mut in_str, mut esc) = (false, false);
    for (i, &c) in b.iter().enumerate() {
        if in_str {
            out.push(c);
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        if c == '"' {
            in_str = true;
        } else if c == ',' {
            let next = b[i + 1..].iter().find(|x| !x.is_whitespace());
            if matches!(next, Some(']') | Some('}')) {
                continue;
            }
        }
        out.push(c);
    }
    out
}

/// JSON from a model-written value: plain, inside a code fence, with trailing commas, or double-encoded (a JSON string holding JSON).
fn lenient_json(raw: &str) -> Option<serde_json::Value> {
    let mut t = raw.trim();
    if t.starts_with("```") {
        t = t.split_once('\n').map_or("", |x| x.1).trim();
        t = t.strip_suffix("```").unwrap_or(t).trim();
    }
    let v = serde_json::from_str::<serde_json::Value>(t).ok().or_else(|| serde_json::from_str(&strip_trailing_commas(t)).ok())?;
    if let serde_json::Value::String(s) = &v {
        let st = s.trim_start();
        if st.starts_with('[') || st.starts_with('{') {
            if let Some(inner) = lenient_json(s) {
                return Some(inner);
            }
        }
    }
    Some(v)
}

/// A collected DSML value typed by the schema types (empty = no schema: JSON if it parses, else the text as a string).
fn typed_value(raw: &str, types: &[String]) -> serde_json::Value {
    use serde_json::Value;
    if types.is_empty() {
        return lenient_json(raw).unwrap_or_else(|| Value::String(raw.to_string()));
    }
    let has = |t: &str| types.iter().any(|x| x.eq_ignore_ascii_case(t));
    if has("array") || has("object") {
        if let Some(v) = lenient_json(raw) {
            if (v.is_array() && has("array")) || (v.is_object() && has("object")) {
                return v;
            }
        }
    }
    let v = crate::toolcall::coerce_to_schema_type(raw.trim(), types);
    if let Value::String(_) = &v {
        if !has("string") {
            tracing::warn!("dsml: a {types:?} parameter did not parse as JSON ({} chars); passed as text", raw.len());
        }
    }
    v
}

pub struct OutputParser {
    inv: Inv,
    /// DSML closing tags met with no call before them: the reply is not a valid tool call (see `malformed`)
    malformed: bool,
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
    dsml: bool,
    /// deltas owed before any output is parsed (the ToolStart of a call the prompt opened by name)
    pending: Vec<Delta>,
    /// the prompt ended with `<function=`: a model that writes the opener again (`<tool_call>`, `<function=`) has it dropped
    reopen: bool,
}

impl OutputParser {
    pub fn new(cfg: ParserConfig) -> Self {
        let mut inv = Inv::default();
        let mut pending = Vec::new();
        let mut calls = 0;
        let mode = match &cfg.forced {
            Some(ForcedCall::Any) => Mode::Name,
            Some(ForcedCall::Named(n)) => {
                calls = 1;
                pending.push(Delta::ToolStart { index: 0, id: random_id("call_", 24), name: n.clone() });
                inv = Inv { tool: n.clone(), emit: true, index: 0, ..Default::default() };
                Mode::Args { index: 0, name: n.clone(), emit: true }
            }
            None if cfg.thinking => Mode::Reasoning,
            None => Mode::Content,
        };
        Self {
            reopen: cfg.forced == Some(ForcedCall::Any),
            inv,
            pending,
            malformed: false,
            mode,
            buf: String::new(),
            tools: cfg.tools,
            max_calls: cfg.max_tool_calls,
            calls,
            reported_calls: calls,
            reasoning_ws: String::new(),
            content_ws: String::new(),
            reasoning_started: false,
            content_started: false,
            raw_tool: String::new(),
            dsml: cfg.dsml,
        }
    }

    /// True when DSML closing tags came with no call (the model wrote its answer as text and then only the end of a call block): the reply is passed
    /// as content, the server marks it so a client can retry.
    pub fn malformed(&self) -> bool {
        self.malformed
    }

    /// True while the parser is inside the model's reasoning block.
    pub fn in_reasoning(&self) -> bool {
        self.mode == Mode::Reasoning
    }

    /// Number of tool calls reported so far.
    pub fn tool_calls(&self) -> usize {
        self.reported_calls
    }

    pub fn feed(&mut self, text: &str) -> Vec<Delta> {
        self.buf.push_str(text);
        let mut out = std::mem::take(&mut self.pending);
        while self.step(&mut out, false) {}
        merge(out)
    }

    /// End of output: flush everything held back.
    pub fn finish(&mut self) -> Vec<Delta> {
        let mut out = std::mem::take(&mut self.pending);
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

    fn inv_emit(&self, out: &mut Vec<Delta>, text: String) {
        if self.inv.emit && !text.is_empty() {
            out.push(Delta::ToolArgs { index: self.inv.index, arguments: text });
        }
    }

    /// Inside a DSML invoke: the arguments JSON is emitted as the parameters arrive (`{"p": "` ... escaped pieces of the value ... `"`, `}`), so a long string value
    /// streams to the client as it is written. Values are JSON-escaped exactly as `dsml_arguments` would write them.
    fn dsml_invoke_step(&mut self, out: &mut Vec<Delta>, eof: bool) -> bool {
        let mut progressed = false;
        loop {
            if let Some(kind) = self.inv.val.clone() {
                // a value ends at its own closing tag, or (the model forgot it) at the next parameter, the invoke's end or the calls end; the tags never belong to it
                let ends: [&str; 4] = [DS_PARAM_END, DS_PARAM, DS_INVOKE_END, DS_CALLS_END];
                let found = find_first(&self.buf, &ends);
                let own = matches!(found, Some((_, m)) if m == DS_PARAM_END);
                let (take, done) = match found {
                    Some((i, _)) => (i, true),
                    None if eof => (self.buf.len(), true),
                    None => {
                        let mut t = self.buf.len() - holdback(&self.buf, &ends);
                        // trailing whitespace is held until it is known whether the value ends at its own tag (kept) or not (trimmed)
                        while t > 0 && self.buf.as_bytes()[t - 1].is_ascii_whitespace() {
                            t -= 1;
                        }
                        (t, false)
                    }
                };
                let mut piece: String = self.buf.drain(..take).collect();
                if done && !own {
                    piece.truncate(piece.trim_end().len());
                }
                if !piece.is_empty() {
                    progressed = true;
                }
                match &kind {
                    Kind::Str => {
                        let j = crate::pyjson::dumps(&serde_json::Value::String(piece), None);
                        self.inv_emit(out, j[1..j.len() - 1].to_string());
                    }
                    Kind::Raw(_) => self.inv.raw.push_str(&piece),
                }
                if !done {
                    return progressed;
                }
                if own {
                    self.buf.drain(..DS_PARAM_END.len());
                }
                match kind {
                    Kind::Str => self.inv_emit(out, "\"".into()),
                    Kind::Raw(types) => {
                        let raw = std::mem::take(&mut self.inv.raw);
                        let v = typed_value(&raw, &types);
                        self.inv_emit(out, crate::pyjson::dumps(&v, None));
                    }
                }
                self.inv.val = None;
                progressed = true;
                continue;
            }
            let ws = self.buf.len() - self.buf.trim_start().len();
            if ws > 0 {
                self.buf.drain(..ws);
                progressed = true;
            }
            if self.buf.starts_with(DS_INVOKE_END) {
                self.buf.drain(..DS_INVOKE_END.len());
                self.close_invoke(out);
                return true;
            }
            // the model forgot the invoke's closing tag: the calls end or the next invoke closes it
            if self.buf.starts_with(DS_CALLS_END) || self.buf.starts_with(DS_INVOKE) {
                self.close_invoke(out);
                return true;
            }
            if self.buf.starts_with(DS_PARAM) {
                let rest = &self.buf[DS_PARAM.len()..];
                let Some(q) = rest.find('"') else { return progressed };
                let Some(gt) = rest[q + 1..].find('>') else { return progressed };
                let pname = rest[..q].to_string();
                let attrs = rest[q + 1..q + 1 + gt].to_string();
                let consumed = DS_PARAM.len() + q + 1 + gt + 1;
                // the request's schema types the value (a string="true" array parameter is still an array, a string="false" string parameter still a string);
                // without a schema the attribute decides
                let schema_types = self.tools.as_ref().and_then(|t| t.properties(&self.inv.tool)).and_then(|p| p.get(&pname)).map(crate::toolcall::extract_types_from_schema);
                let kind = match schema_types {
                    Some(t) if t.iter().all(|x| x.eq_ignore_ascii_case("string")) => Kind::Str,
                    Some(t) => Kind::Raw(t),
                    None if attrs.contains("string=\"false\"") => Kind::Raw(vec![]),
                    None => Kind::Str,
                };
                self.buf.drain(..consumed);
                let head = format!("{}{}: {}", if self.inv.nparams == 0 { "{" } else { ", " }, crate::pyjson::dumps(&serde_json::Value::String(pname), None), if matches!(kind, Kind::Str) { "\"" } else { "" });
                self.inv_emit(out, head);
                self.inv.nparams += 1;
                self.inv.val = Some(kind);
                progressed = true;
                continue;
            }
            if self.buf.is_empty() {
                if eof {
                    // a call cut off at a parameter boundary: close what was written
                    self.close_invoke(out);
                    return true;
                }
                return progressed;
            }
            if !eof && [DS_INVOKE_END, DS_CALLS_END, DS_INVOKE, DS_PARAM].iter().any(|m| m.starts_with(self.buf.as_str())) {
                return progressed;
            }
            // text between parameters that is no tag: skipped up to the next tag
            let mut skip = self.buf[1..].find('<').map_or(self.buf.len(), |i| i + 1);
            while !self.buf.is_char_boundary(skip) {
                skip += 1;
            }
            self.buf.drain(..skip);
            progressed = true;
        }
    }

    /// Inside `<function=NAME>` (Qwen XML): the arguments JSON is emitted as the parameters arrive, exactly the text
    /// `toolcall::convert_arguments` writes for the whole block (`{"p": "` ... escaped pieces of the value ... `"`, `}`): a string value
    /// streams to the client while the model writes it; a value the schema types as anything else is collected and converted at its end.
    /// A value ends at `</parameter>`, at a `<parameter=` that starts a line (the model forgot the end tag), at `</function>` or
    /// `</tool_call>`; one newline after `>` and one before the end are template markup. Call markup written inside a line of the value
    /// (a file that talks about tool calls) is content: `<parameter=` inside a line, `<function=` and `<tool_call>` open a nested level
    /// and the closing tags that match them stay in the value.
    fn xml_args_step(&mut self, out: &mut Vec<Delta>, eof: bool) -> bool {
        const P_START: &str = "<parameter=";
        const P_END: &str = "</parameter>";
        let mut progressed = false;
        loop {
            if let Some(kind) = self.inv.val.clone() {
                if !self.inv.started {
                    if self.buf.is_empty() && !eof {
                        return progressed;
                    }
                    if self.buf.starts_with('\n') {
                        self.buf.drain(..1);
                    }
                    self.inv.started = true;
                    progressed = true;
                }
                // the first tag that really ends the value: tags that open or close a nested level are skipped over
                let all: [&str; 6] = [P_END, P_START, FUNC_END, TOOL_END, FUNC_START, TOOL_START];
                let mut from = 0;
                let mut found: Option<(usize, &str)> = None;
                while let Some((i, m)) = find_first(&self.buf[from..], &all) {
                    let i = from + i;
                    let line_start = if i > 0 { self.buf.as_bytes()[i - 1] == b'\n' } else { !self.inv.consumed };
                    let n = &mut self.inv.nest;
                    let nested = n.iter().any(|&x| x > 0);
                    let content = match m {
                        P_START if nested || !line_start => { n[0] += 1; true }
                        FUNC_START => { n[1] += 1; true }
                        TOOL_START => { n[2] += 1; true }
                        P_END if n[0] > 0 => { n[0] -= 1; true }
                        FUNC_END if n[1] > 0 => { n[1] -= 1; true }
                        TOOL_END if n[2] > 0 => { n[2] -= 1; true }
                        _ => false,
                    };
                    if !content {
                        found = Some((i, m));
                        break;
                    }
                    // counted once: everything up to and including the tag is value text, emitted now
                    let upto = i + m.len();
                    let piece: String = self.buf.drain(..upto).collect();
                    self.inv.consumed = true;
                    progressed = true;
                    match &kind {
                        Kind::Str => {
                            let j = crate::pyjson::dumps(&serde_json::Value::String(piece), None);
                            self.inv_emit(out, j[1..j.len() - 1].to_string());
                        }
                        Kind::Raw(_) => self.inv.raw.push_str(&piece),
                    }
                    from = 0;
                }
                let (take, done) = match found {
                    Some((i, _)) => (i, true),
                    None if eof => (self.buf.len(), true),
                    None => {
                        let mut t = self.buf.len() - holdback(&self.buf, &all);
                        // the newline before the end tag is markup: the last one waits until it is known what follows
                        if t > 0 && self.buf.as_bytes()[t - 1] == b'\n' {
                            t -= 1;
                        }
                        (t, false)
                    }
                };
                let mut piece: String = self.buf.drain(..take).collect();
                if done && piece.ends_with('\n') {
                    piece.pop();
                }
                if !piece.is_empty() {
                    progressed = true;
                    self.inv.consumed = true;
                }
                match &kind {
                    Kind::Str => {
                        let j = crate::pyjson::dumps(&serde_json::Value::String(piece), None);
                        self.inv_emit(out, j[1..j.len() - 1].to_string());
                    }
                    Kind::Raw(_) => self.inv.raw.push_str(&piece),
                }
                if !done {
                    return progressed;
                }
                if matches!(found, Some((_, m)) if m == P_END) {
                    self.buf.drain(..P_END.len());
                }
                self.inv.consumed = false;
                self.inv.nest = [0; 3];
                match kind {
                    Kind::Str => self.inv_emit(out, "\"".into()),
                    Kind::Raw(types) => {
                        let raw = std::mem::take(&mut self.inv.raw);
                        if !self.inv.skip {
                            self.inv_emit(out, crate::pyjson::dumps(&coerce_to_schema_type(&raw, &types), None));
                        }
                    }
                }
                self.inv.val = None;
                self.inv.skip = false;
                self.inv.started = false;
                progressed = true;
                continue;
            }
            // between parameters: text that is no tag is ignored
            let marks: [&str; 3] = [P_START, FUNC_END, TOOL_END];
            match find_first(&self.buf, &marks) {
                Some((i, m)) if m == P_START => {
                    let after = i + P_START.len();
                    let gt = self.buf[after..].find('>');
                    // the block ending before the name does: a cut-off parameter tag, nothing usable
                    if let Some((e, m2)) = find_first(&self.buf[after..], &[FUNC_END, TOOL_END]) {
                        if gt.is_none_or(|g| e < g) {
                            self.buf.drain(..after + e + m2.len());
                            self.close_args(out, if m2 == FUNC_END { Mode::Between } else { Mode::Content });
                            return true;
                        }
                    }
                    let Some(gt) = gt else {
                        if eof {
                            self.buf.clear();
                            self.close_args(out, Mode::Content);
                            return true;
                        }
                        if i > 0 {
                            self.buf.drain(..i);
                            progressed = true;
                        }
                        return progressed;
                    };
                    let pname = self.buf[after..after + gt].trim().to_string();
                    self.buf.drain(..after + gt + 1);
                    if pname.is_empty() {
                        self.inv.skip = true;
                        self.inv.val = Some(Kind::Raw(Vec::new()));
                    } else {
                        // the request's schema types the value; a parameter the schema does not know is a string
                        let types = self.tools.as_ref().and_then(|t| t.properties(&self.inv.tool)).and_then(|p| p.get(&pname)).map(extract_types_from_schema);
                        let kind = match types {
                            Some(t) if !is_string_only(&t) => Kind::Raw(t),
                            _ => Kind::Str,
                        };
                        let head = format!("{}{}: {}", if self.inv.nparams == 0 { "{" } else { ", " }, crate::pyjson::dumps(&serde_json::Value::String(pname), None),
                            if matches!(kind, Kind::Str) { "\"" } else { "" });
                        self.inv_emit(out, head);
                        self.inv.nparams += 1;
                        self.inv.val = Some(kind);
                    }
                    self.inv.started = false;
                    progressed = true;
                }
                Some((i, m)) => {
                    self.buf.drain(..i + m.len());
                    self.close_args(out, if m == FUNC_END { Mode::Between } else { Mode::Content });
                    return true;
                }
                None => {
                    if eof {
                        self.buf.clear();
                        self.close_args(out, Mode::Content);
                        return true;
                    }
                    let cut = self.buf.len() - holdback(&self.buf, &marks);
                    if cut > 0 {
                        self.buf.drain(..cut);
                        progressed = true;
                    }
                    return progressed;
                }
            }
        }
    }

    /// The end of an XML call's parameters: close the arguments object.
    fn close_args(&mut self, out: &mut Vec<Delta>, next: Mode) {
        let close = if self.inv.nparams == 0 { "{}" } else { "}" };
        self.inv_emit(out, close.into());
        self.inv = Inv::default();
        self.mode = next;
    }

    fn close_invoke(&mut self, out: &mut Vec<Delta>) {
        let close = if self.inv.nparams == 0 { "{}" } else { "}" };
        self.inv_emit(out, close.into());
        self.calls += 1;
        self.inv = Inv::default();
        self.mode = Mode::Dsml;
    }

    /// One parsing step; returns true if it made progress.
    fn step(&mut self, out: &mut Vec<Delta>, eof: bool) -> bool {
        match self.mode.clone() {
            Mode::Reasoning => {
                let mut markers = vec![THINK_END, THINK_START];
                if self.tools_on() {
                    if self.dsml {
                        markers.push(DS_CALLS);
                        markers.push(DS_INVOKE);
                    } else {
                        markers.push(TOOL_START);
                    }
                }
                match find_first(&self.buf, &markers) {
                    Some((i, m)) if m == TOOL_START && i + m.len() == self.buf.len() && !eof => {
                        // what follows the tag decides (below): wait for it
                        if i == 0 {
                            return false;
                        }
                        let before: String = self.buf.drain(..i).collect();
                        self.emit_reasoning(&before, out);
                        true
                    }
                    Some((i, m)) if m == TOOL_START && !self.buf[i + m.len()..].starts_with(['\n', '\r']) && i + m.len() < self.buf.len() => {
                        // The model's own call opener is `<tool_call>` + newline (the template's format). The tag with text right
                        // after it, inside the reasoning, is the model thinking ABOUT such markup (quoting a file or the request):
                        // reasoning text, not the end of the block.
                        let text: String = self.buf.drain(..i + m.len()).collect();
                        self.emit_reasoning(&text, out);
                        true
                    }
                    Some((i, m)) => {
                        let before = self.buf[..i].to_string();
                        self.buf.drain(..i + m.len());
                        self.emit_reasoning(&before, out);
                        match m {
                            THINK_END => {
                                self.reasoning_ws.clear();
                                self.mode = Mode::Content;
                            }
                            DS_CALLS => {
                                self.reasoning_ws.clear();
                                self.mode = Mode::Dsml;
                            }
                            DS_INVOKE => {
                                // the model sometimes leaves out the opening calls tag
                                self.reasoning_ws.clear();
                                self.buf.insert_str(0, DS_INVOKE);
                                self.mode = Mode::Dsml;
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
                    if self.dsml {
                        markers.push(DS_CALLS);
                        markers.push(DS_INVOKE);
                        markers.push(DS_CALLS_END);
                        markers.push(DS_INVOKE_END);
                        markers.push(DS_PARAM_END);
                    } else {
                        markers.push(TOOL_START);
                        markers.push(FUNC_START);
                    }
                }
                match find_first(&self.buf, &markers) {
                    Some((i, m)) => {
                        let before = self.buf[..i].to_string();
                        self.buf.drain(..i + m.len());
                        self.emit_content(&before, out);
                        match m {
                            DS_CALLS => {
                                self.content_ws.clear();
                                self.mode = Mode::Dsml;
                            }
                            DS_INVOKE => {
                                // the model sometimes leaves out the opening calls tag (a real 14K-token write call did): the invoke is the start of the block
                                self.content_ws.clear();
                                self.buf.insert_str(0, DS_INVOKE);
                                self.mode = Mode::Dsml;
                            }
                            DS_CALLS_END | DS_INVOKE_END | DS_PARAM_END => {
                                // closing tags with no opening block: dropped after a call, otherwise kept as text and the reply flagged
                                if self.calls == 0 {
                                    self.malformed = true;
                                    self.emit_content(m, out);
                                }
                            }
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
            Mode::Name if self.reopen => {
                // a function name never starts with `<`
                let lead = self.buf.len() - self.buf.trim_start().len();
                let t = &self.buf[lead..];
                if let Some(m) = [TOOL_START, FUNC_START].into_iter().find(|m| t.starts_with(m)) {
                    self.buf.drain(..lead + m.len());
                    return true;
                }
                if !eof && (t.is_empty() || TOOL_START.starts_with(t) || FUNC_START.starts_with(t)) {
                    return false;
                }
                self.reopen = false;
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
                    self.inv = Inv { tool: name.clone(), emit, index, ..Default::default() };
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
            Mode::Args { .. } => self.xml_args_step(out, eof),
            Mode::Dsml => {
                let ws = self.buf.len() - self.buf.trim_start().len();
                self.buf.drain(..ws);
                if self.buf.starts_with(DS_CALLS_END) {
                    self.buf.drain(..DS_CALLS_END.len());
                    self.mode = Mode::Content;
                    return true;
                }
                if self.buf.starts_with(DS_INVOKE) {
                    match self.buf[DS_INVOKE.len()..].find("\">") {
                        Some(i) => {
                            let name = self.buf[DS_INVOKE.len()..DS_INVOKE.len() + i].trim().to_string();
                            self.buf.drain(..DS_INVOKE.len() + i + 2);
                            let emit = !name.is_empty() && self.max_calls.is_none_or(|m| self.reported_calls < m);
                            self.inv = Inv { emit, index: self.reported_calls, tool: name.clone(), ..Default::default() };
                            if emit {
                                self.reported_calls += 1;
                                out.push(Delta::ToolStart { index: self.inv.index, id: random_id("call_", 24), name: name.clone() });
                            }
                            self.mode = Mode::DsmlInvoke { name };
                            return true;
                        }
                        None if eof => {
                            self.buf.clear();
                            self.mode = Mode::Content;
                            return true;
                        }
                        None => return false,
                    }
                }
                if !eof && (self.buf.is_empty() || DS_CALLS_END.starts_with(self.buf.as_str()) || DS_INVOKE.starts_with(self.buf.as_str()) || self.buf.starts_with(DS_INVOKE)) {
                    return false;
                }
                // not a call after all (or the end of a cut-off block): back to content
                self.mode = Mode::Content;
                !self.buf.is_empty()
            }
            Mode::DsmlInvoke { name: _ } => self.dsml_invoke_step(out, eof),
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

/// Merge adjacent reasoning / content deltas and adjacent argument pieces of one call.
fn merge(deltas: Vec<Delta>) -> Vec<Delta> {
    let mut out: Vec<Delta> = Vec::with_capacity(deltas.len());
    for d in deltas {
        match (out.last_mut(), d) {
            (Some(Delta::Reasoning(a)), Delta::Reasoning(b)) => a.push_str(&b),
            (Some(Delta::Content(a)), Delta::Content(b)) => a.push_str(&b),
            (Some(Delta::ToolArgs { index: i, arguments: a }), Delta::ToolArgs { index: j, arguments: b }) if *i == j => a.push_str(&b),
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
                        c.2.push_str(arguments);
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
        ParserConfig { thinking, tools: tools_on.then(tools), max_tool_calls: None, dsml: false, forced: None }
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

    /// Reasoning that quotes call markup (no newline after the tag: not the model's own opener) stays reasoning.
    #[test]
    fn quoted_call_markup_inside_reasoning_is_reasoning() {
        let quoted = "The content is `<tool_call><function=x><parameter=y>1</parameter></function></tool_call> <div>` literally.";
        let p = parse(cfg(true, true), &format!("{quoted}\n</think>\n\n<tool_call>\n<function=search>\n<parameter=query>\nq\n</parameter>\n</function>\n</tool_call>"));
        assert_eq!(p.reasoning, quoted);
        assert_eq!(p.content, "");
        assert_eq!(p.tool_calls, vec![call("search", json!({"query": "q"}))]);
        // cut off right after the tag: as before, the block ends there (nothing follows that could tell)
        let p = parse(cfg(true, true), "thinking <tool_call>");
        assert_eq!(p.reasoning, "thinking");
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
        let p = parse(ParserConfig { thinking: false, tools: Some(tools()), max_tool_calls: Some(1), dsml: false, forced: None }, text);
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
        // the name first, then the arguments as they are written (a string value streams)
        assert!(matches!(&d[..], [Delta::Content(c), Delta::ToolStart { index: 0, name, .. }, Delta::ToolArgs { index: 0, arguments }]
            if c == "Checking." && name == "search" && arguments == "{\"query\": \"ab"), "{d:?}");
        let d = p.feed("c\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(d, vec![Delta::ToolArgs { index: 0, arguments: "c\"}".into() }]);
        assert!(p.finish().is_empty());
    }

    fn dcfg() -> ParserConfig {
        ParserConfig { thinking: false, tools: Some(tools()), max_tool_calls: None, dsml: true, forced: None }
    }

    fn dcall(name: &str, body: &str) -> String {
        format!("<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"{name}\">\n{body}</｜DSML｜ invoke>\n</｜DSML｜ calls>")
    }

    fn sp(name: &str, v: &str) -> String {
        format!("<｜DSML｜ parameter name=\"{name}\" string=\"true\">{v}</｜DSML｜ parameter>\n")
    }

    #[test]
    fn dsml_basic_and_preamble() {
        let text = format!("I will write it.\n\n{}", dcall("write", &(sp("path", "a.html") + &sp("content", "<p>hi</p>\n"))));
        let p = parse(dcfg(), &text);
        assert_eq!(p.content, "I will write it.");
        assert_eq!(p.tool_calls, vec![call("write", json!({"path": "a.html", "content": "<p>hi</p>\n"}))]);
    }

    #[test]
    fn dsml_missing_opening_calls_tag() {
        // the real failure: invoke without the opening calls tag, closed with the closing one
        let text = "Plan:\n- a\n\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">c.html</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"content\" string=\"true\"><html></html>\n</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let p = parse(dcfg(), text);
        assert_eq!(p.content, "Plan:\n- a");
        assert_eq!(p.tool_calls, vec![call("write", json!({"path": "c.html", "content": "<html></html>\n"}))]);
        // same thing while "thinking"
        let mut c = dcfg();
        c.thinking = true;
        let p = parse(c, &format!("thoughts</think>{text}"));
        assert_eq!(p.tool_calls.len(), 1);
        assert!(!p.content.contains("DSML"));
    }

    #[test]
    fn dsml_values_with_markup_and_lookalikes() {
        let html = "<!DOCTYPE html>\n<script>if (a < b && c > d) { x = \"</p>\"; }</script>\n<!-- parameter name=\"x\" -->\n```html\n<div name=\"parameter\"></div>\n```\n</DSML calls> not really\n</｜DSML para\r\n\\ \" \u{1F600} é";
        let p = parse(dcfg(), &dcall("write", &(sp("path", "p") + &sp("content", html))));
        assert_eq!(p.tool_calls.len(), 1);
        let v: serde_json::Value = serde_json::from_str(&p.tool_calls[0].2).unwrap();
        assert_eq!(v["content"], html);
    }

    #[test]
    fn dsml_big_value_and_parameter_order_and_nonstring() {
        let big = "<div class=\"row\">line</div>\n".repeat(4000); // 120K characters
        let body = sp("content", &big) + "<｜DSML｜ parameter name=\"limit\" string=\"false\">5</｜DSML｜ parameter>\n" + &sp("query", "q");
        let p = parse(dcfg(), &dcall("search", &body));
        let v: serde_json::Value = serde_json::from_str(&p.tool_calls[0].2).unwrap();
        assert_eq!(v["content"], big);
        assert_eq!(v["limit"], 5);
        assert_eq!(v["query"], "q");
        assert_eq!(p.tool_calls[0].2.find("\"content\"").unwrap() < p.tool_calls[0].2.find("\"limit\"").unwrap(), true);
    }

    #[test]
    fn dsml_crlf_multiple_calls_and_empty_args() {
        let text = format!("{}\r\n{}", dcall("search", &sp("query", "a\r\nb")).replace("\n", "\r\n").replace("\r\r\n", "\r\n"), dcall("get_weather", ""));
        let p = parse(dcfg(), &text);
        assert_eq!(p.tool_calls.len(), 2);
        let v: serde_json::Value = serde_json::from_str(&p.tool_calls[0].2).unwrap();
        assert_eq!(v["query"], "a\r\nb");
        assert_eq!(p.tool_calls[1].2, "{}");
    }

    #[test]
    fn dsml_streams_long_values_incrementally() {
        let mut p = OutputParser::new(dcfg());
        let mut args = Vec::new();
        let mut started = false;
        let text = dcall("write", &(sp("path", "x") + &sp("content", &"abc\n".repeat(500))));
        let chars: Vec<char> = text.chars().collect();
        for piece in chars.chunks(40) {
            for d in p.feed(&piece.iter().collect::<String>()) {
                match d {
                    Delta::ToolStart { .. } => started = true,
                    Delta::ToolArgs { arguments, .. } => {
                        assert!(started);
                        args.push(arguments);
                    }
                    _ => {}
                }
            }
        }
        for d in p.finish() {
            if let Delta::ToolArgs { arguments, .. } = d {
                args.push(arguments);
            }
        }
        assert!(args.len() > 20, "{} pieces", args.len());
        let v: serde_json::Value = serde_json::from_str(&args.concat()).unwrap();
        assert_eq!(v["content"], "abc\n".repeat(500));
    }

    #[test]
    fn dsml_cut_off_call_closes_json() {
        let text = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"content\" string=\"true\">half a fi";
        let p = parse(dcfg(), text);
        let v: serde_json::Value = serde_json::from_str(&p.tool_calls[0].2).unwrap();
        assert_eq!(v["content"], "half a fi");
    }

    #[test]
    fn dsml_closing_tags_without_a_block_flag_the_reply() {
        let mut p = OutputParser::new(pcfg());
        let mut acc = Parsed::default();
        acc.absorb(&p.feed("The file:\nbody {}\n</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"));
        acc.absorb(&p.finish());
        assert!(p.malformed());
        assert!(acc.content.contains("body {}"));
        assert!(acc.tool_calls.is_empty());
        let mut p = OutputParser::new(pcfg());
        p.feed("plain answer");
        p.finish();
        assert!(!p.malformed());
    }

    #[test]
    fn dsml_stray_closing_tag_without_call_is_text() {
        let p = parse(dcfg(), "see </｜DSML｜ calls> here");
        assert_eq!(p.content, "see </｜DSML｜ calls> here");
        assert!(p.tool_calls.is_empty());
    }

    fn pi_tools() -> ToolSet {
        ToolSet::from_openai(&[
            json!({"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}}),
            json!({"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}),
            json!({"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object", "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}}}}}, "required": ["path", "edits"]}}}),
            json!({"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}}, "required": ["command"]}}}),
        ])
    }

    fn pcfg() -> ParserConfig {
        ParserConfig { thinking: false, tools: Some(pi_tools()), max_tool_calls: None, dsml: true, forced: None }
    }

    fn args_of(p: &Parsed, i: usize) -> serde_json::Value {
        serde_json::from_str(&p.tool_calls[i].2).unwrap_or_else(|e| panic!("bad json {e}: {}", p.tool_calls[i].2))
    }

    #[test]
    fn dsml_missing_parameter_closing_tag_before_invoke_end() {
        // the live failure: the closing parameter tag is missing, the closing tags used to end up inside the value
        let new_text = "</style>\n<style>\n.a { content: \"x\\\"y\"; }\n</style>\n".repeat(60);
        let edits = serde_json::to_string(&json!([{"oldText": "</style>", "newText": new_text}])).unwrap();
        let text = format!("<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"edit\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">contact.html</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"edits\" string=\"true\">{edits}</｜DSML｜ invoke>\n</｜DSML｜ calls>");
        let p = parse(pcfg(), &text);
        assert_eq!(p.tool_calls.len(), 1);
        let v = args_of(&p, 0);
        assert_eq!(v["path"], "contact.html");
        assert!(v["edits"].is_array(), "{}", p.tool_calls[0].2);
        assert_eq!(v["edits"][0]["newText"], new_text);
        assert!(!p.tool_calls[0].2.contains("DSML"));
    }

    #[test]
    fn dsml_missing_closing_tags_everywhere() {
        // parameter closing missing before the next parameter, invoke closing missing before the calls closing
        let text = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a.txt\n<｜DSML｜ parameter name=\"content\" string=\"true\">hello\nworld\n</｜DSML｜ calls>";
        let p = parse(pcfg(), text);
        assert_eq!(args_of(&p, 0), json!({"path": "a.txt", "content": "hello\nworld"}));
        // two invokes, the first without its closing tag
        let text = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a</｜DSML｜ parameter>\n<｜DSML｜ invoke name=\"bash\">\n<｜DSML｜ parameter name=\"command\" string=\"true\">ls</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let p = parse(pcfg(), text);
        assert_eq!(p.tool_calls.len(), 2);
        assert_eq!(args_of(&p, 1), json!({"command": "ls"}));
    }

    #[test]
    fn dsml_schema_types_win_over_the_string_attribute() {
        // string="true" on an array parameter: still an array; string="false" on a string parameter: still a string; numbers by schema
        let text = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"edit\">\n<｜DSML｜ parameter name=\"path\" string=\"false\">123</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"edits\" string=\"true\">[{\"oldText\": \"a\", \"newText\": \"b\"},]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">x</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"limit\" string=\"true\">50</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        let p = parse(pcfg(), text);
        assert_eq!(args_of(&p, 0), json!({"path": "123", "edits": [{"oldText": "a", "newText": "b"}]}));
        assert_eq!(args_of(&p, 1), json!({"path": "x", "limit": 50}));
    }

    #[test]
    fn dsml_array_value_forms_fence_double_encoded_unparsable() {
        let wrap = |v: &str| format!("<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"edit\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">f</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"edits\" string=\"true\">{v}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>");
        let want = json!([{"oldText": "a", "newText": "b"}]);
        let p = parse(pcfg(), &wrap("```json\n[{\"oldText\": \"a\", \"newText\": \"b\"}]\n```"));
        assert_eq!(args_of(&p, 0)["edits"], want);
        let p = parse(pcfg(), &wrap("\"[{\\\"oldText\\\": \\\"a\\\", \\\"newText\\\": \\\"b\\\"}]\""));
        assert_eq!(args_of(&p, 0)["edits"], want);
        // not JSON at all: the raw text is passed (the client's own validation reports it), never dropped
        let p = parse(pcfg(), &wrap("replace a with b"));
        assert_eq!(args_of(&p, 0)["edits"], "replace a with b");
    }

    #[test]
    fn dsml_write_then_edits_pi_session() {
        let page = "<!DOCTYPE html>\n<html><head><style>\nbody { margin: 0; }\n</style></head>\n<body>\n".to_string() + &"<p class=\"x\">line</p>\n".repeat(900) + "</body></html>\n";
        let t1 = format!("I will write the page.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n{}{}</｜DSML｜ invoke>\n</｜DSML｜ calls>", sp("path", "contact.html"), sp("content", &page));
        let p = parse(pcfg(), &t1);
        assert_eq!(p.content, "I will write the page.");
        assert_eq!(args_of(&p, 0)["content"], page);
        for k in 0..3 {
            let nt = format!("/* edit {k} */\n").to_string() + &"a { color: \"red\"; }\n".repeat(150);
            let edits = serde_json::to_string(&json!([{"oldText": "body { margin: 0; }", "newText": nt}])).unwrap();
            let t = format!("<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"edit\">\n{}{}</｜DSML｜ invoke>\n</｜DSML｜ calls>", sp("path", "contact.html"), sp("edits", &edits));
            let p = parse(pcfg(), &t);
            assert_eq!(args_of(&p, 0)["edits"][0]["newText"], nt);
        }
    }

    #[test]
    fn dsml_trailing_whitespace_kept_with_own_tag_trimmed_without() {
        let p = parse(pcfg(), "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"content\" string=\"true\">x  \n\n</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>");
        assert_eq!(args_of(&p, 0)["content"], "x  \n\n");
        let p = parse(pcfg(), "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"content\" string=\"true\">x  \n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>");
        assert_eq!(args_of(&p, 0)["content"], "x");
    }
    /// The streamed Qwen XML arguments are, concatenated, exactly what `convert_arguments` writes for the whole block: typed and string
    /// values, markup newlines, a value that ends at the next parameter, a cut-off value, a nameless parameter, text between tags.
    #[test]
    fn xml_streamed_arguments_equal_the_whole_block_conversion() {
        let bodies = [
            "\n<parameter=location>\nParis, France\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n",
            "<parameter=location>no newlines</parameter><parameter=days>x7</parameter>",
            "\n<parameter=location>\n\nline one\n\nline \"two\" \\ and a tab\t café 日本語 😀\n\n</parameter>\n",
            "\n<parameter=location>\nvalue without its end tag\n<parameter=days>\n 12 \n</parameter>\n",
            "\n<parameter=>\ndropped\n</parameter>\n<parameter=unknown>\n{\"kept\": \"as text\"}\n</parameter>\n",
            "\n",
            "noise <parameter=location>\n</parameter> more noise <parameter=days>\n\n</parameter>",
            "\n<parameter=location>\na < b && c > d </param not a tag\n</parameter>\n",
        ];
        for body in bodies {
            let want = crate::toolcall::convert_arguments(body, tools().properties("get_weather"));
            let p = parse(cfg(false, true), &format!("<tool_call>\n<function=get_weather>{body}</function>\n</tool_call>"));
            assert_eq!(p.tool_calls, vec![("id".to_string(), "get_weather".to_string(), want.clone())], "{body:?}");
            assert!(serde_json::from_str::<serde_json::Value>(&want).is_ok());
        }
        // text ABOUT tool calls as a value: balanced markup inside a line of the value is content, whole
        let about = "Docs: <tool_call><function=x><parameter=y>1</parameter></function></tool_call> <div class=\"a\">&amp;</div>\nsecond line with <parameter=z>2</parameter> inline";
        let p = parse(cfg(false, true), &format!("<tool_call>\n<function=search>\n<parameter=query>\n{about}\n</parameter>\n<parameter=limit>\n3\n</parameter>\n</function>\n</tool_call>"));
        assert_eq!(p.tool_calls, vec![call("search", json!({"query": about, "limit": 3}))]);
        assert_eq!(p.content, "");
        // cut off inside a value / inside a parameter tag / right after the name
        for body in ["\n<parameter=location>\nPar", "\n<parameter=location>\nParis\n</parameter>\n<parameter=da", ""] {
            let want = crate::toolcall::convert_arguments(body, tools().properties("get_weather"));
            let p = parse(cfg(false, true), &format!("<tool_call>\n<function=get_weather>{body}"));
            assert_eq!(p.tool_calls, vec![("id".to_string(), "get_weather".to_string(), want)], "{body:?}");
        }
    }

    /// A long string value reaches the client while it is written: many argument deltas, valid JSON once joined.
    #[test]
    fn xml_streams_long_values_incrementally() {
        let content: String = (0..400).map(|i| format!("<p class=\"row\">line {i} \\ end</p>\n")).collect();
        let text = format!("<tool_call>\n<function=search>\n<parameter=query>\n{content}\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>");
        let mut p = OutputParser::new(cfg(false, true));
        let (mut pieces, mut args, mut first_at) = (0, String::new(), None);
        let chars: Vec<char> = text.chars().collect();
        for (k, ch) in chars.chunks(16).enumerate() {
            for d in p.feed(&ch.iter().collect::<String>()) {
                if let Delta::ToolArgs { arguments, .. } = d {
                    pieces += 1;
                    first_at.get_or_insert(k);
                    args.push_str(&arguments);
                }
            }
        }
        for d in p.finish() {
            if let Delta::ToolArgs { arguments, .. } = d {
                args.push_str(&arguments);
            }
        }
        assert!(pieces > 200, "{pieces} argument deltas");
        assert!(first_at.unwrap() < 8, "first argument delta only at chunk {first_at:?}");
        let v: serde_json::Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v["query"], content);
        assert_eq!(v["limit"], 5);
    }

    /// tool_choice required / named: the prompt ended with the call opener, the output starts inside the call.
    #[test]
    fn forced_call_openers() {
        let any = ParserConfig { forced: Some(ForcedCall::Any), ..cfg(false, true) };
        assert_eq!(ForcedCall::Any.opener(), "<tool_call>\n<function=");
        let p = parse(any.clone(), "get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(p.tool_calls, vec![call("get_weather", json!({"location": "Paris"}))]);
        assert_eq!(p.content, "");
        // a model that writes the opener itself again
        let p = parse(ParserConfig { forced: Some(ForcedCall::Any), ..cfg(false, true) }, "<tool_call>\n<function=search>\n<parameter=limit>\n4\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(p.tool_calls, vec![call("search", json!({"limit": 4}))]);
        // a second call after the forced one
        let p = parse(any, "search>\n<parameter=query>\nq\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(p.tool_calls, vec![call("search", json!({"query": "q"})), call("get_weather", json!({"days": 2}))]);
        let named = ParserConfig { forced: Some(ForcedCall::Named("get_weather".into())), ..cfg(false, true) };
        assert_eq!(ForcedCall::Named("get_weather".into()).opener(), "<tool_call>\n<function=get_weather>\n");
        let p = parse(named.clone(), "<parameter=location>\nOslo\n</parameter>\n<parameter=metric>\nfalse\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(p.tool_calls, vec![call("get_weather", json!({"location": "Oslo", "metric": false}))]);
        // the name is reported before any output, and a reply that ends at once still is a call with empty arguments
        let mut q = OutputParser::new(named.clone());
        assert!(matches!(&q.feed("")[..], [Delta::ToolStart { index: 0, name, .. }] if name == "get_weather"));
        assert_eq!(q.tool_calls(), 1);
        let p = parse(named, "");
        assert_eq!(p.tool_calls, vec![call("get_weather", json!({}))]);
    }
}
