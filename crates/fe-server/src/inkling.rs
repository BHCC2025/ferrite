//! Inkling-2 output markup -> reasoning / content / tool-call deltas (the model's own message format; see its chat template).
//!
//! The model writes its turn as typed messages: `<|message_model|>` + a content type + body + `<|end_message|>`, and ends the turn with
//! `<|content_model_end_sampling|>` (the EOS token, which the engine does not forward):
//!   `<|message_model|><|content_thinking|>REASONING<|end_message|>`
//!   `<|message_model|><|content_text|>ANSWER<|end_message|>`
//!   `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`   (one message per call)
//! The text fed in is the decoded stream in which every markup special token is ONE sentinel character (`sentinel`, below), so text that merely
//! spells a marker is never taken for one; a marker may arrive in any chunk (nothing is held back but the sentinel-free text itself).
//!
//! Tool-call arguments are streamed as they are written: the JSON of `args` is checked character by character by [`JsonCheck`], a pending
//! comma is held until the next token so a cut-off can always be closed into valid JSON, and the call's start is held back until either the
//! arguments are complete or [`HOLD`] bytes of valid arguments are in, so that a call that is broken early is returned as plain content
//! instead (and flagged `malformed`). A body that does not begin with `{"name":"..","args":` (other key order, `arguments`, a string-encoded
//! args value) is read whole at its end. A call cut off by max_tokens is closed into valid JSON.
use serde_json::Value;

use crate::parser::Delta;
use crate::toolcall::ToolSet;
use crate::util::random_id;

/// The markers as the pipeline hands them over: the model's special tokens become one private-use character each (see [`sentinel`]), so that the
/// SAME text written as ordinary tokens (a file that talks about this very format) stays ordinary text.
pub const MSG: &str = "\u{E000}";
pub const THINK: &str = "\u{E001}";
pub const TEXT: &str = "\u{E002}";
pub const INVOKE: &str = "\u{E003}";
pub const END: &str = "\u{E004}";
pub const EOT: &str = "\u{E005}";
/// any other typed content (`<|content_xml|>`, ...): read as text
pub const OTHER: &str = "\u{E006}";

/// The sentinel for a special token's literal text, or None for a special token that is not part of the reply markup.
pub fn sentinel(literal: &str) -> Option<&'static str> {
    Some(match literal {
        "<|message_model|>" => MSG,
        "<|content_thinking|>" => THINK,
        "<|content_text|>" => TEXT,
        "<|content_invoke_tool_json|>" => INVOKE,
        "<|end_message|>" => END,
        "<|content_model_end_sampling|>" => EOT,
        l if l.starts_with("<|content_") => OTHER,
        _ => return None,
    })
}

/// Markup written with the literal special-token strings -> the sentinel form (tests, tools).
pub fn from_markup(s: &str) -> String {
    let mut out = s.to_string();
    for lit in ["<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>", "<|end_message|>", "<|content_model_end_sampling|>"] {
        out = out.replace(lit, sentinel(lit).unwrap());
    }
    out
}
/// bytes of valid arguments held back before a call is announced to the client
const HOLD: usize = 512;

// ---------------------------------------------------------------------------------------------------------------------------------------------
// incremental JSON checker for the `args` object

#[derive(Clone, Debug, PartialEq)]
enum V {
    Value,
    ObjFirst,
    ObjKey,
    ObjColon,
    ObjNext,
    ArrFirst,
    ArrNext,
    Str { key: bool, esc: bool, hex: u8 },
    Num(String),
    Lit(&'static str),
    Done,
}

/// Strict JSON, one character at a time; the top-level value must be an object.
#[derive(Clone, Debug)]
pub struct JsonCheck {
    st: V,
    stack: Vec<u8>,
    first: bool,
}

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\n' | '\r' | '\t')
}

fn valid_number(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    if i < b.len() && b[i] == b'-' {
        i += 1;
    }
    match b.get(i) {
        Some(b'0') => i += 1,
        Some(c) if c.is_ascii_digit() => {
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        _ => return false,
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let s0 = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == s0 {
            return false;
        }
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let s0 = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == s0 {
            return false;
        }
    }
    i == b.len()
}

impl JsonCheck {
    pub fn new() -> Self {
        Self { st: V::Value, stack: Vec::new(), first: true }
    }

    fn after_value(&mut self) {
        self.st = match self.stack.last() {
            None => V::Done,
            Some(b'{') => V::ObjNext,
            _ => V::ArrNext,
        };
    }

    /// Feed one character; `Ok(true)` when the object is complete, `Err` when the text is not valid JSON.
    pub fn push(&mut self, c: char) -> Result<bool, ()> {
        // a reply marker (sentinel) inside the arguments: the model wrote a special token into a value
        if ('\u{E000}'..='\u{E006}').contains(&c) {
            return Err(());
        }
        loop {
            match &mut self.st {
                V::Done => return if is_ws(c) { Ok(true) } else { Err(()) },
                V::Str { key, esc, hex } => {
                    if *hex > 0 {
                        if !c.is_ascii_hexdigit() {
                            return Err(());
                        }
                        *hex -= 1;
                    } else if *esc {
                        match c {
                            '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {}
                            'u' => *hex = 4,
                            _ => return Err(()),
                        }
                        *esc = false;
                    } else if c == '\\' {
                        *esc = true;
                    } else if c == '"' {
                        let key = *key;
                        if key {
                            self.st = V::ObjColon;
                        } else {
                            self.after_value();
                        }
                    } else if (c as u32) < 0x20 {
                        return Err(());
                    }
                    return Ok(false);
                }
                V::Num(t) => {
                    if c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-') {
                        t.push(c);
                        return Ok(false);
                    }
                    if !valid_number(t) {
                        return Err(());
                    }
                    self.after_value();
                    continue; // reprocess c in the new state
                }
                V::Lit(rest) => {
                    let r: &'static str = *rest;
                    if !r.starts_with(c) {
                        return Err(());
                    }
                    let r = &r[c.len_utf8()..];
                    if r.is_empty() {
                        self.after_value();
                    } else {
                        self.st = V::Lit(r);
                    }
                    return Ok(self.st == V::Done);
                }
                st => {
                    if is_ws(c) {
                        return Ok(false);
                    }
                    match (st.clone(), c) {
                        (V::Value, c) if self.first && c != '{' => return Err(()),
                        (V::Value, '{') => {
                            self.first = false;
                            self.stack.push(b'{');
                            self.st = V::ObjFirst;
                        }
                        (V::Value, '[') => {
                            self.stack.push(b'[');
                            self.st = V::ArrFirst;
                        }
                        (V::Value, '"') => self.st = V::Str { key: false, esc: false, hex: 0 },
                        (V::Value, c) if c == '-' || c.is_ascii_digit() => self.st = V::Num(c.to_string()),
                        (V::Value, 't') => self.st = V::Lit("rue"),
                        (V::Value, 'f') => self.st = V::Lit("alse"),
                        (V::Value, 'n') => self.st = V::Lit("ull"),
                        (V::ObjFirst, '}') | (V::ObjNext, '}') => {
                            self.stack.pop();
                            self.after_value();
                        }
                        (V::ObjFirst, '"') | (V::ObjKey, '"') => self.st = V::Str { key: true, esc: false, hex: 0 },
                        (V::ObjColon, ':') => self.st = V::Value,
                        (V::ObjNext, ',') => self.st = V::ObjKey,
                        (V::ArrFirst, ']') | (V::ArrNext, ']') => {
                            self.stack.pop();
                            self.after_value();
                        }
                        (V::ArrFirst, _) => {
                            self.st = V::Value;
                            continue;
                        }
                        (V::ArrNext, ',') => self.st = V::Value,
                        _ => return Err(()),
                    }
                    return Ok(self.st == V::Done);
                }
            }
        }
    }

    /// True when `c` would be a structural comma (held back by the streamer).
    fn is_comma(&self, c: char) -> bool {
        c == ',' && matches!(self.st, V::ObjNext | V::ArrNext)
    }

    /// The text that turns what has been fed so far into valid JSON (the call was cut off).
    pub fn closers(&self) -> String {
        let mut s = String::new();
        match &self.st {
            V::Str { key, esc, hex } => {
                if *hex > 0 {
                    for _ in 0..*hex {
                        s.push('0');
                    }
                } else if *esc {
                    s.push('\\');
                }
                s.push('"');
                if *key {
                    s.push_str(":null");
                }
            }
            V::Num(t) => {
                if !valid_number(t) {
                    // `-`, `1.`, `1e`, `1e-`: finish with a digit
                    s.push('0');
                }
            }
            V::Lit(rest) => s.push_str(rest),
            V::ObjColon => s.push_str(":null"),
            V::Value if !self.stack.is_empty() => s.push_str("null"),
            _ => {}
        }
        for open in self.stack.iter().rev() {
            s.push(if *open == b'{' { '}' } else { ']' });
        }
        s
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// the message parser

enum Phase {
    /// before `{"name":"..","args":`
    Prefix,
    /// inside the args object
    Args,
    /// args complete, waiting for the message end
    After,
    /// the body is not in the usual shape: read whole at the end
    Fallback,
    /// malformed after the call was announced, or a refused call: the rest of the message is dropped
    Drop,
    /// malformed before the call was announced: the rest of the message is passed as content
    AsContent,
}

struct Tool {
    header: String,
    phase: Phase,
    /// the whole body so far (fallback / content)
    raw: String,
    name: Option<String>,
    id: String,
    index: usize,
    report: bool,
    started: bool,
    held: String,
    pending: String,
    check: JsonCheck,
}

enum St {
    /// between messages (or before the first one)
    Start,
    /// after `<|message_model|>`: header text up to the content type
    Header,
    Thinking,
    Text,
    Tool(Box<Tool>),
    Done,
}

pub struct InklingParser {
    st: St,
    buf: String,
    tools: Option<ToolSet>,
    max_calls: Option<usize>,
    reported: usize,
    seen_calls: usize,
    malformed: bool,
    /// the model started a tool call but no tools were offered: nothing it writes after that is an answer, the request should end
    halted: bool,
}

fn find_first(s: &str, marks: &[&str]) -> Option<(usize, usize)> {
    marks.iter().enumerate().filter_map(|(k, m)| s.find(m).map(|i| (i, k))).min()
}

/// Length of the longest suffix of `s` that is a proper prefix of one of the markers.
fn holdback(s: &str, marks: &[&str]) -> usize {
    let mut best = 0;
    for m in marks {
        let max = m.len().saturating_sub(1).min(s.len());
        for l in (1..=max).rev() {
            if l > best && s.is_char_boundary(s.len() - l) && m.starts_with(&s[s.len() - l..]) {
                best = l;
                break;
            }
        }
    }
    best
}

impl InklingParser {
    pub fn new(tools: Option<ToolSet>, max_calls: Option<usize>) -> Self {
        Self { st: St::Start, buf: String::new(), tools, max_calls, reported: 0, seen_calls: 0, malformed: false, halted: false }
    }

    pub fn malformed(&self) -> bool {
        self.malformed
    }

    pub fn in_reasoning(&self) -> bool {
        matches!(self.st, St::Thinking)
    }

    /// True once the model has started a tool call in a request that offered no tools (the call is dropped, the rest of the output is ignored).
    pub fn halted(&self) -> bool {
        self.halted
    }

    pub fn tool_calls(&self) -> usize {
        self.reported
    }

    pub fn feed(&mut self, text: &str) -> Vec<Delta> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        while self.step(&mut out, false) {}
        merge(out)
    }

    pub fn finish(&mut self) -> Vec<Delta> {
        let mut out = Vec::new();
        while self.step(&mut out, true) {}
        // whatever is left of an open message
        match std::mem::replace(&mut self.st, St::Done) {
            St::Tool(t) => self.end_tool(*t, true, &mut out),
            St::Thinking | St::Text | St::Start | St::Header | St::Done => {}
        }
        merge(out)
    }

    fn step(&mut self, out: &mut Vec<Delta>, eof: bool) -> bool {
        match &mut self.st {
            St::Done => {
                self.buf.clear();
                false
            }
            St::Start => {
                let marks = [MSG, EOT];
                match find_first(&self.buf, &marks) {
                    Some((i, k)) => {
                        let before: String = self.buf.drain(..i).collect();
                        if !before.trim().is_empty() {
                            out.push(Delta::Content(before));
                        }
                        self.buf.drain(..marks[k].len());
                        self.st = if k == 0 { St::Header } else { St::Done };
                        true
                    }
                    None => {
                        let keep = if eof { 0 } else { holdback(&self.buf, &marks) };
                        let take = self.buf.len() - keep;
                        let t: String = self.buf.drain(..take).collect();
                        if !t.trim().is_empty() {
                            out.push(Delta::Content(t));
                        }
                        false
                    }
                }
            }
            St::Header => {
                // the content type marker, or (an empty message) the end of it
                let hit = find_first(&self.buf, &[THINK, TEXT, INVOKE, OTHER, END, MSG, EOT]);
                match hit {
                    Some((i, k)) if k <= 3 => {
                        let header: String = self.buf.drain(..i).collect();
                        self.buf.drain(..[THINK, TEXT, INVOKE, OTHER][k].len());
                        self.st = match k {
                            0 => St::Thinking,
                            2 if self.tools.is_none() => {
                                // a call to a tool that was never offered (a model trained on agent harnesses reaches for Bash on its own, and what follows is
                                // often a degenerate loop): drop it and end the turn
                                self.malformed = true;
                                self.halted = true;
                                self.buf.clear();
                                St::Done
                            }
                            2 => {
                                let seen = self.seen_calls;
                                self.seen_calls += 1;
                                let report = self.max_calls.map(|m| seen < m).unwrap_or(true);
                                let as_content = false;
                                St::Tool(Box::new(Tool {
                                    header: header.trim().to_string(),
                                    phase: if as_content { Phase::AsContent } else { Phase::Prefix },
                                    raw: String::new(),
                                    name: None,
                                    id: random_id("call_", 24),
                                    index: self.reported,
                                    report: report || as_content,
                                    started: false,
                                    held: String::new(),
                                    pending: String::new(),
                                    check: JsonCheck::new(),
                                }))
                            }
                            _ => St::Text, // content_text and any other typed content
                        };
                        true
                    }
                    Some((i, 4)) => {
                        self.buf.drain(..i + END.len());
                        self.st = St::Start;
                        true
                    }
                    Some((i, 5)) => {
                        self.buf.drain(..i + MSG.len());
                        true
                    }
                    Some((i, _)) => {
                        self.buf.drain(..i + EOT.len());
                        self.st = St::Done;
                        true
                    }
                    None => {
                        if eof {
                            self.buf.clear();
                        }
                        false
                    }
                }
            }
            St::Thinking | St::Text => {
                let thinking = matches!(self.st, St::Thinking);
                let marks = [END, MSG, EOT];
                let emit = |mut t: String, out: &mut Vec<Delta>| {
                    // typed-content markers in the middle of a body are dropped
                    t.retain(|c| !('\u{E000}'..='\u{E006}').contains(&c));
                    if !t.is_empty() {
                        out.push(if thinking { Delta::Reasoning(t) } else { Delta::Content(t) });
                    }
                };
                match find_first(&self.buf, &marks) {
                    Some((i, k)) => {
                        let t: String = self.buf.drain(..i).collect();
                        emit(t, out);
                        self.buf.drain(..marks[k].len());
                        self.st = match k {
                            0 => St::Start,
                            1 => St::Header,
                            _ => St::Done,
                        };
                        true
                    }
                    None => {
                        let keep = if eof { 0 } else { holdback(&self.buf, &marks) };
                        let take = self.buf.len() - keep;
                        let t: String = self.buf.drain(..take).collect();
                        emit(t, out);
                        false
                    }
                }
            }
            St::Tool(_) => {
                let St::Tool(mut t) = std::mem::replace(&mut self.st, St::Done) else { unreachable!() };
                let marks = [END, MSG, EOT];
                // the message ends at its end marker; args are consumed up to there
                let (body_len, ended) = match find_first(&self.buf, &marks) {
                    Some((i, k)) => (i, Some(k)),
                    None => (self.buf.len() - if eof { 0 } else { holdback(&self.buf, &marks) }, None),
                };
                // an unfinished prefix needs more text: with the end marker in sight use what there is
                let body: String = self.buf.drain(..body_len).collect();
                let progressed = !body.is_empty();
                let more = self.tool_text(&mut t, &body, ended.is_some() || eof, out);
                match ended {
                    Some(k) => {
                        self.buf.drain(..marks[k].len());
                        self.end_tool(*t, false, out);
                        self.st = match k {
                            0 => St::Start,
                            1 => St::Header,
                            _ => St::Done,
                        };
                        true
                    }
                    None => {
                        self.st = St::Tool(t);
                        let _ = more;
                        progressed && !eof
                    }
                }
            }
        }
    }

    /// Body text of a tool message. `last`: the message ends with this text (no more to come).
    fn tool_text(&mut self, t: &mut Tool, text: &str, last: bool, out: &mut Vec<Delta>) -> bool {
        t.raw.push_str(text);
        match t.phase {
            Phase::Drop => {}
            Phase::AsContent => {
                if !text.is_empty() {
                    out.push(Delta::Content(text.to_string()));
                }
            }
            Phase::Fallback => {}
            Phase::After => {}
            Phase::Prefix | Phase::Args => {
                let mut rest = text.to_string();
                if let Phase::Prefix = t.phase {
                    // `raw` holds the whole body so far
                    match match_prefix(&t.raw, last) {
                        Prefix::More => return false,
                        Prefix::No => {
                            t.phase = Phase::Fallback;
                            return false;
                        }
                        Prefix::Ok { name, consumed } => {
                            t.name = Some(name);
                            t.phase = Phase::Args;
                            rest = t.raw[consumed..].to_string();
                            if rest.is_empty() {
                                return true;
                            }
                        }
                    }
                }
                for c in rest.chars() {
                    if !matches!(t.phase, Phase::Args) {
                        break;
                    }
                    let comma = t.check.is_comma(c);
                    match t.check.push(c) {
                        Err(()) => {
                            self.malformed = true;
                            if t.started {
                                // already announced: close what was sent into valid JSON and drop the rest
                                let cl = t.check.closers();
                                self.flush_call(t, out, Some(cl));
                                t.phase = Phase::Drop;
                            } else {
                                // never announced: the call becomes content
                                if t.report {
                                    out.push(Delta::Content(t.raw.clone()));
                                }
                                t.phase = Phase::AsContent;
                            }
                            return true;
                        }
                        Ok(done) => {
                            if comma || (!t.pending.is_empty() && is_ws(c)) {
                                t.pending.push(c);
                            } else {
                                let p = std::mem::take(&mut t.pending);
                                t.held.push_str(&p);
                                t.held.push(c);
                            }
                            if done {
                                self.flush_call(t, out, None);
                                t.phase = Phase::After;
                            }
                        }
                    }
                }
                if matches!(t.phase, Phase::Args) && (t.started || t.held.len() >= HOLD) {
                    self.flush_call(t, out, None);
                }
            }
        }
        true
    }

    /// Announce the call (once) and send the arguments held so far (+ `tail`).
    fn flush_call(&mut self, t: &mut Tool, out: &mut Vec<Delta>, tail: Option<String>) {
        if !t.report {
            t.held.clear();
            return;
        }
        if !t.started {
            t.started = true;
            self.reported += 1;
            out.push(Delta::ToolStart { index: t.index, id: t.id.clone(), name: t.name.clone().unwrap_or_default() });
        }
        let mut s = std::mem::take(&mut t.held);
        if let Some(tail) = tail {
            s.push_str(&tail);
        }
        if !s.is_empty() {
            out.push(Delta::ToolArgs { index: t.index, arguments: s });
        }
    }

    /// The message ended (or the output did: `cut`).
    fn end_tool(&mut self, mut t: Tool, cut: bool, out: &mut Vec<Delta>) {
        match t.phase {
            Phase::Args => {
                // cut off inside the arguments: close them into valid JSON
                self.malformed = true;
                let cl = t.check.closers();
                if cut || !t.report {
                    self.flush_call(&mut t, out, Some(cl));
                } else {
                    self.flush_call(&mut t, out, Some(cl));
                }
            }
            Phase::Prefix => {
                // cut off before the arguments began
                self.malformed = true;
                if !t.raw.trim().is_empty() && t.report {
                    out.push(Delta::Content(t.raw.clone()));
                }
            }
            Phase::Fallback => match parse_whole(&t.raw, &t.header) {
                Some((name, args)) if t.report => {
                    self.reported += 1;
                    out.push(Delta::ToolStart { index: t.index, id: t.id.clone(), name });
                    out.push(Delta::ToolArgs { index: t.index, arguments: args });
                }
                Some(_) => {}
                None => {
                    self.malformed = true;
                    if t.report && !t.raw.trim().is_empty() {
                        out.push(Delta::Content(t.raw.clone()));
                    }
                }
            },
            Phase::After | Phase::Drop | Phase::AsContent => {}
        }
    }
}

enum Prefix {
    More,
    No,
    Ok { name: String, consumed: usize },
}

/// `{"name":"NAME","args":` (whitespace allowed between tokens; `arguments` accepted for `args`), stopping before the `{` of the args.
fn match_prefix(s: &str, last: bool) -> Prefix {
    let b = s.as_bytes();
    let mut i = 0;
    let more = || if last { Prefix::No } else { Prefix::More };
    macro_rules! skip_ws {
        () => {
            while i < b.len() && (b[i] as char).is_ascii_whitespace() {
                i += 1;
            }
        };
    }
    macro_rules! lit {
        ($l:expr) => {{
            let l: &str = $l;
            let n = l.len().min(b.len() - i);
            if s[i..i + n] != l[..n] {
                return Prefix::No;
            }
            if n < l.len() {
                return more();
            }
            i += l.len();
        }};
    }
    skip_ws!();
    if i == b.len() {
        return more();
    }
    lit!("{");
    skip_ws!();
    lit!("\"name\"");
    skip_ws!();
    lit!(":");
    skip_ws!();
    // the name: a JSON string
    if i == b.len() {
        return more();
    }
    if b[i] != b'"' {
        return Prefix::No;
    }
    let start = i;
    i += 1;
    let mut esc = false;
    loop {
        if i >= b.len() {
            return more();
        }
        if esc {
            esc = false;
        } else if b[i] == b'\\' {
            esc = true;
        } else if b[i] == b'"' {
            i += 1;
            break;
        }
        i += 1;
    }
    let name: String = match serde_json::from_str::<String>(&s[start..i]) {
        Ok(n) => n,
        Err(_) => return Prefix::No,
    };
    skip_ws!();
    lit!(",");
    skip_ws!();
    if i == b.len() {
        return more();
    }
    if s[i..].starts_with("\"arguments\"") {
        i += "\"arguments\"".len();
    } else {
        lit!("\"args\"");
    }
    skip_ws!();
    lit!(":");
    skip_ws!();
    if i == b.len() {
        return more();
    }
    if b[i] != b'{' {
        return Prefix::No;
    }
    Prefix::Ok { name, consumed: i }
}

/// A whole tool message body: `{"name": .., "args" | "arguments" | "parameters": {..}}` in any key order; args may be a JSON string holding the object.
fn parse_whole(body: &str, header: &str) -> Option<(String, String)> {
    let v: Value = serde_json::from_str(body.trim()).ok()?;
    let o = v.as_object()?;
    let name = o.get("name").and_then(Value::as_str).map(str::to_string).or_else(|| (!header.is_empty()).then(|| header.to_string()))?;
    let args = o.get("args").or_else(|| o.get("arguments")).or_else(|| o.get("parameters")).cloned().unwrap_or(Value::Object(Default::default()));
    let args = match args {
        Value::String(s) => serde_json::from_str::<Value>(&s).ok().filter(Value::is_object)?,
        v @ Value::Object(_) => v,
        _ => return None,
    };
    Some((name, serde_json::to_string(&args).ok()?))
}

/// Adjacent deltas of one kind joined (what the client would reassemble anyway).
fn merge(v: Vec<Delta>) -> Vec<Delta> {
    let mut out: Vec<Delta> = Vec::new();
    for d in v {
        match (out.last_mut(), &d) {
            (Some(Delta::Reasoning(a)), Delta::Reasoning(b)) => a.push_str(b),
            (Some(Delta::Content(a)), Delta::Content(b)) => a.push_str(b),
            (Some(Delta::ToolArgs { index: i, arguments: a }), Delta::ToolArgs { index: j, arguments: b }) if i == j => a.push_str(b),
            _ => out.push(d),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parsed;

    fn tools() -> Option<ToolSet> {
        Some(ToolSet::from_openai(&[serde_json::json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}})]))
    }

    /// parse `text` fed in chunks of `step` bytes (char boundaries respected)
    fn run(text: &str, step: usize, tools: Option<ToolSet>, max: Option<usize>) -> (Parsed, InklingParser) {
        let text = &from_markup(text);
        let mut p = InklingParser::new(tools, max);
        let mut acc = Parsed::default();
        let mut i = 0;
        while i < text.len() {
            let mut j = (i + step).min(text.len());
            while !text.is_char_boundary(j) {
                j += 1;
            }
            acc.absorb(&p.feed(&text[i..j]));
            i = j;
        }
        acc.absorb(&p.finish());
        (acc, p)
    }

    const FULL: &str = "<|message_model|><|content_thinking|>I should add.<|end_message|><|message_model|><|content_text|>Let me call é.<|end_message|><|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1,\"b\":[true,null,\"x\\\"y\"],\"c\":{\"d\":-2.5e3}}}<|end_message|><|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}<|end_message|>";

    #[test]
    fn splits_everything_at_any_chunking() {
        for step in [1, 2, 3, 5, 7, 11, 64, 1000] {
            let (a, p) = run(FULL, step, tools(), None);
            assert_eq!(a.reasoning, "I should add.", "step {step}");
            assert_eq!(a.content, "Let me call é.", "step {step}");
            assert_eq!(a.tool_calls.len(), 2, "step {step}");
            assert_eq!(a.tool_calls[0].1, "f");
            assert_eq!(a.tool_calls[0].2, "{\"a\":1,\"b\":[true,null,\"x\\\"y\"],\"c\":{\"d\":-2.5e3}}", "step {step}");
            assert_eq!(a.tool_calls[1].2, "{}");
            assert_eq!(p.tool_calls(), 2);
            assert!(!p.malformed());
        }
    }

    #[test]
    fn long_arguments_stream_in_pieces() {
        let big = "x".repeat(3000);
        let text = format!("<|message_model|>f<|content_invoke_tool_json|>{{\"name\":\"f\",\"args\":{{\"content\":\"{big}\",\"n\":1}}}}<|end_message|>");
        let text = from_markup(&text);
        let mut p = InklingParser::new(tools(), None);
        let mut pieces = 0;
        let mut args = String::new();
        let mut starts = 0;
        for chunk in text.as_bytes().chunks(40) {
            for d in p.feed(std::str::from_utf8(chunk).unwrap()) {
                match d {
                    Delta::ToolStart { .. } => starts += 1,
                    Delta::ToolArgs { arguments, .. } => {
                        pieces += 1;
                        args.push_str(&arguments);
                    }
                    _ => {}
                }
            }
        }
        for d in p.finish() {
            if let Delta::ToolArgs { arguments, .. } = d {
                args.push_str(&arguments);
            }
        }
        assert_eq!(starts, 1);
        assert!(pieces > 10, "{pieces} pieces");
        assert_eq!(serde_json::from_str::<Value>(&args).unwrap()["content"].as_str().unwrap().len(), 3000);
    }

    #[test]
    fn broken_early_becomes_content() {
        let text = "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":oops}}<|end_message|>";
        let (a, p) = run(text, 3, tools(), None);
        assert!(a.tool_calls.is_empty());
        assert!(p.malformed());
        assert!(a.content.contains("oops"), "{:?}", a.content);
    }

    #[test]
    fn cut_off_call_is_closed_into_valid_json() {
        for cut in [
            "{\"a\":\"abc", "{\"a\":1,\"b\":[1,", "{\"a\":", "{\"a\":tr", "{\"a\":1.", "{\"k", "{\"a\":{\"b\":\"x\\", "{\"a\":\"\\u00",
        ] {
            let mut padded = String::from("{\"name\":\"f\",\"args\":");
            padded.push_str(cut);
            let text = format!("<|message_model|>f<|content_invoke_tool_json|>{padded}");
            let (a, _) = run(&text, 5, tools(), None);
            assert_eq!(a.tool_calls.len(), 1, "{cut}");
            assert!(serde_json::from_str::<Value>(&a.tool_calls[0].2).is_ok(), "{cut}: {:?}", a.tool_calls[0].2);
        }
    }

    #[test]
    fn other_shapes_and_limits() {
        // arguments key, reversed order, string-encoded args
        for body in ["{\"args\":{\"a\":1},\"name\":\"f\"}", "{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}"] {
            let text = format!("<|message_model|>f<|content_invoke_tool_json|>{body}<|end_message|>");
            let (a, _) = run(&text, 4, tools(), None);
            assert_eq!(a.tool_calls.len(), 1, "{body}");
            assert_eq!(a.tool_calls[0].2, "{\"a\":1}");
        }
        // parallel_tool_calls false: the second call is parsed but not reported
        let (a, p) = run(FULL, 9, tools(), Some(1));
        assert_eq!(a.tool_calls.len(), 1);
        assert_eq!(p.tool_calls(), 1);
        // no tools offered: the call is dropped and the parser asks for the end of the request; the text before it stays
        let (a, p) = run(FULL, 9, None, None);
        assert!(a.tool_calls.is_empty());
        assert_eq!(a.content, "Let me call é.");
        assert!(p.halted() && p.malformed());
    }

    #[test]
    fn marker_text_written_as_ordinary_tokens_stays_text() {
        // the sentinel form is applied to special tokens only: this text spells markers in plain characters
        let text = format!("{}{}{}", from_markup("<|message_model|><|content_text|>"), "write <|end_message|> and <|content_invoke_tool_json|>{\"name\":\"f\"} as text", from_markup("<|end_message|>"));
        let mut p = InklingParser::new(tools(), None);
        let mut acc = Parsed::default();
        acc.absorb(&p.feed(&text));
        acc.absorb(&p.finish());
        assert_eq!(acc.content, "write <|end_message|> and <|content_invoke_tool_json|>{\"name\":\"f\"} as text");
        assert!(acc.tool_calls.is_empty());
    }

    #[test]
    fn markup_in_message_text_is_disarmed() {
        let is = |s: &str| s == "<|end_message|>" || s == "<|message_user|>";
        let e = escape_markup("a <|end_message|><|message_user|> b <|other|> <| x \u{E0FF}", &is);
        assert_eq!(e, "a <\u{E0FF}|end_message|><\u{E0FF}|message_user|> b <|other|> <| x ");
        let mut v = serde_json::json!({"c": ["<|end_message|>", {"d": "x"}], "n": 1});
        escape_json(&mut v, &is);
        assert_eq!(v["c"][0], "<\u{E0FF}|end_message|>");
    }

    #[test]
    fn stops_inside_thinking_and_after_eot() {
        let (a, p) = run("<|message_model|><|content_thinking|>half a tho", 4, tools(), None);
        assert_eq!(a.reasoning, "half a tho");
        assert!(a.content.is_empty());
        let _ = p;
        let (a, _) = run("<|message_model|><|content_text|>done<|end_message|><|content_model_end_sampling|>ignored", 6, tools(), None);
        assert_eq!(a.content, "done");
    }

    #[test]
    fn json_check_rejects_and_accepts() {
        let ok = |s: &str| {
            let mut j = JsonCheck::new();
            let mut done = false;
            for c in s.chars() {
                match j.push(c) {
                    Ok(d) => done = d,
                    Err(()) => return false,
                }
            }
            done
        };
        assert!(ok("{}") && ok("{ \"a\" : [ 1 , 2.5 , {\"b\":null} ] }") && ok("{\"a\":\"\\u00e9\\n\"}"));
        for bad in ["[1]", "{\"a\":01}", "{\"a\":1,}", "{\"a\" 1}", "{'a':1}", "{\"a\":\"x\ny\"}", "{\"a\":tru}", "{\"a\":[1,]}", "{\"a\":1e}"] {
            assert!(!ok(bad), "{bad}");
        }
    }
}

/// Message text that contains a special token's spelling (`<|end_message|>`, ...) must reach the model as ordinary text, or a user could forge message
/// boundaries and the model would copy real control tokens into its answers: the `<` of such a spelling gets this private-use character after it (the
/// prompt is tokenized in pieces split here, see [`escape_markup`]); [`ESC`] itself is removed from input text first.
pub const ESC: char = '\u{E0FF}';

/// `s` with every special-token spelling disarmed (see [`ESC`]); `is_markup` says whether a `<|...|>` string is a special token.
pub fn escape_markup(s: &str, is_markup: &dyn Fn(&str) -> bool) -> String {
    let s: std::borrow::Cow<str> = if s.contains(ESC) { s.replace(ESC, "").into() } else { s.into() };
    if !s.contains("<|") {
        return s.into_owned();
    }
    let mut out = String::with_capacity(s.len() + 8);
    let mut rest: &str = &s;
    while let Some(i) = rest.find("<|") {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        match tail.find("|>") {
            Some(j) if is_markup(&tail[..j + 2]) => {
                out.push('<');
                out.push(ESC);
                out.push_str(&tail[1..j + 2]);
                rest = &tail[j + 2..];
            }
            _ => {
                out.push_str("<|");
                rest = &tail[2..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// [`escape_markup`] on every string of a JSON value (messages, tools).
pub fn escape_json(v: &mut Value, is_markup: &dyn Fn(&str) -> bool) {
    match v {
        Value::String(s) => *s = escape_markup(s, is_markup),
        Value::Array(a) => a.iter_mut().for_each(|x| escape_json(x, is_markup)),
        Value::Object(o) => o.iter_mut().for_each(|(_, x)| escape_json(x, is_markup)),
        _ => {}
    }
}
