//! Token-cycle loop breaker and the per-request phase bookkeeping (reasoning cap, breaker) shared by the thinking-aware engines.
//!
//! A loop here is an EXACT token cycle that reaches the current position: the tail of the output is periodic. A rule fires only when the
//! periodic tail holds both a minimum number of full repeats of the cycle and a minimum number of repeated tokens (tokens beyond the first
//! occurrence), so the repeat count a period needs scales inversely with its length: a 5-token cycle needs 81 repeats at the reasoning
//! default (400 repeated tokens), a 70-token cycle 7, a 300-token cycle 4. Text that says a span two or three times and moves on, spans that
//! are similar but not identical, and lines that differ by a counter never match (the last is the reasoning cap's job).
use crate::engine::SamplingParams;
use serde_json::Value as Json;
use std::collections::HashMap;
use std::sync::OnceLock;

/// One threshold pair: the periodic tail must hold `min_repeats` full occurrences of the cycle AND `min_repeated_tokens` tokens beyond the first one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoopRule {
    pub min_repeats: usize,
    pub min_repeated_tokens: usize,
}

/// Loop-breaker thresholds (`loop_rules` in the model dir's sampling.json; env FE_LOOP_RULES = the same JSON object overrides it).
#[derive(Clone, Debug, PartialEq)]
pub struct LoopCfg {
    /// cycle lengths considered, in tokens
    pub min_period: usize,
    pub max_period: usize,
    /// inside the reasoning block: firing closes the block
    pub reasoning: LoopRule,
    /// in the answer outside a tool call: the first firing boosts the sampling, the second ends the request
    pub answer: LoopRule,
    /// once a tool call has opened (argument values: files with many identical rows are legitimate)
    pub call: LoopRule,
    /// the repeated-sentence rule on the decoded text (wandering loops the exact cycle misses)
    pub sentence: SentCfg,
    /// tool-call rule, first firing, engines that can take tokens back (the DeepSeek engine): the tokens generated since the cycle was
    /// first seen are held back from the client; at the firing they are dropped, the engine rewinds to that point and samples with
    /// `recover_temperature` / `recover_top_p` (the profile's values) for a bounded stretch. A second firing ends the request as before.
    pub call_recover: bool,
    pub recover_temperature: f32,
    pub recover_top_p: f32,
}

/// Repeated-sentence rule (`loop_rules.sentence` in sampling.json). The decoded text is cut into units (lines, and sentences inside a
/// line), normalised (lower case, punctuation and spacing dropped). The rule looks at the last `window` units of at least
/// UNIT_MIN_CHARS letters or digits: it is saturated when at least `min_fraction` of them are units of `min_chars` letters or more that
/// occur `min_count` times or more inside the window (short units never count as recurring, but they are in the window: an
/// enumeration whose items differ only by a short head is not saturated), and it fires when it has stayed saturated over `min_tokens`
/// tokens in the reasoning block (`answer_min_tokens` in the answer). Text inside a fenced code block and everything after a tool
/// call has opened is never looked at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SentCfg {
    pub enabled: bool,
    pub window: usize,
    pub min_chars: usize,
    pub min_count: usize,
    pub min_fraction: f32,
    pub min_tokens: usize,
    pub answer_min_tokens: usize,
}

impl Default for SentCfg {
    fn default() -> Self {
        // tuned on tests/fixtures: the wandering transcript (12 recurring sentences under changing labels, short rejections between
        // them) reaches a recurring fraction of 0.89 and fires ~600 words in; a 700-line page (HTML + CSS + JS) and a table-heavy
        // answer written by the model stay at 0.05 / 0.09 with their fences removed; an enumeration with two identical remarks an
        // item stays at 0.67
        Self { enabled: true, window: 64, min_chars: 16, min_count: 3, min_fraction: 0.7, min_tokens: 600, answer_min_tokens: 1200 }
    }
}

impl SentCfg {
    pub fn from_json(v: Option<&Json>) -> Self {
        let mut c = Self::default();
        let Some(v) = v else { return c };
        let num = |k: &str| v.get(k).and_then(Json::as_u64).map(|x| x as usize);
        if let Some(b) = v.get("enabled").and_then(Json::as_bool) {
            c.enabled = b;
        }
        if let Some(x) = num("window") {
            c.window = x.clamp(4, 512);
        }
        if let Some(x) = num("min_chars") {
            c.min_chars = x.max(1);
        }
        if let Some(x) = num("min_count") {
            c.min_count = x.max(2);
        }
        if let Some(x) = v.get("min_fraction").and_then(Json::as_f64) {
            c.min_fraction = (x as f32).clamp(0.1, 1.0);
        }
        if let Some(x) = num("min_tokens") {
            c.min_tokens = x;
        }
        if let Some(x) = num("answer_min_tokens") {
            c.answer_min_tokens = x;
        }
        c
    }
}

/// What the repeated-sentence rule found.
#[derive(Clone, Debug, PartialEq)]
pub struct SentHit {
    /// distinct recurring units in the window
    pub distinct: usize,
    /// tokens the window has stayed saturated over
    pub tokens: usize,
    /// the most frequent unit (normalised, first 60 characters)
    pub sample: String,
}

/// The repeated-sentence detector: text in, a hit when the same few sentences keep coming back (see [`SentCfg`]).
pub struct SentDet {
    cfg: SentCfg,
    /// the unit being written, and the token count when it started
    line: String,
    line_tok0: usize,
    fence: bool,
    /// the last `window` units: (hash, token count at the unit's start, normalised head, long enough to count as recurring)
    win: std::collections::VecDeque<(u64, usize, String, bool)>,
    tokens: usize,
    /// token count at the start of the window when it became saturated
    sat_since: Option<usize>,
}

impl SentDet {
    /// letters or digits a unit needs to enter the window at all
    const UNIT_MIN_CHARS: usize = 4;

    pub fn new(cfg: SentCfg) -> Self {
        Self { cfg, line: String::new(), line_tok0: 0, fence: false, win: Default::default(), tokens: 0, sat_since: None }
    }

    /// Forget everything (a phase change, a firing): the pattern must build up again from here.
    pub fn restart(&mut self) {
        self.line.clear();
        self.line_tok0 = self.tokens;
        self.fence = false;
        self.win.clear();
        self.sat_since = None;
    }

    /// Inside a fenced code block.
    pub fn in_fence(&self) -> bool {
        self.fence
    }

    /// Lower-case letters and digits, everything else one space.
    fn normalise(unit: &str) -> String {
        let mut out = String::with_capacity(unit.len());
        for ch in unit.chars() {
            if ch.is_alphanumeric() {
                out.extend(ch.to_lowercase());
            } else if !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
        }
        out.truncate(out.trim_end().len());
        out
    }

    /// A unit without a short label before a colon (`Option:`, `Try again:`), when enough text follows: the same candidate under
    /// another label is the same unit. A label that holds a number stays (`Handler 12:`, `Step 3:`): lines that differ by a counter
    /// are an enumeration making progress, never a match (a runaway of those is the reasoning cap's job, as for the exact rule).
    fn strip_lead(t: &str, min_chars: usize) -> &str {
        if let Some(i) = t.find(':') {
            let rest = t[i + 1..].trim_start();
            if i <= 24 && !t[..i].contains(|c: char| c.is_numeric()) && rest.chars().filter(|c| c.is_alphanumeric()).count() >= min_chars {
                return rest;
            }
        }
        t
    }

    /// The current unit is complete.
    fn unit(&mut self) {
        let unit = std::mem::take(&mut self.line);
        let tok0 = std::mem::replace(&mut self.line_tok0, self.tokens);
        let t = unit.trim();
        if t.starts_with("```") || t.starts_with("~~~") {
            self.fence = !self.fence;
            return;
        }
        if self.fence {
            return;
        }
        let norm = Self::normalise(Self::strip_lead(t, self.cfg.min_chars));
        let chars = norm.chars().filter(|c| *c != ' ').count();
        if chars < Self::UNIT_MIN_CHARS {
            return;
        }
        let mut h = 0xcbf29ce484222325u64;
        for b in norm.bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
        self.win.push_back((h, tok0, norm.chars().take(60).collect(), chars >= self.cfg.min_chars));
        while self.win.len() > self.cfg.window {
            self.win.pop_front();
        }
    }

    /// (units of the window that occur min_count times or more in it, how many distinct ones, the most frequent)
    fn recurring(&self) -> (usize, usize, &str) {
        let mut counts: HashMap<u64, (usize, &str)> = HashMap::new();
        for (h, _, s, _) in self.win.iter().filter(|w| w.3) {
            counts.entry(*h).or_insert((0, s.as_str())).0 += 1;
        }
        let rec: Vec<&(usize, &str)> = counts.values().filter(|c| c.0 >= self.cfg.min_count).collect();
        let best = rec.iter().max_by_key(|c| c.0).map_or("", |c| c.1);
        (rec.iter().map(|c| c.0).sum(), rec.len(), best)
    }

    /// Text of `ntok` more tokens. Some(hit) when the rule's window has stayed saturated over `min_tokens` tokens.
    pub fn push_text(&mut self, text: &str, ntok: usize, min_tokens: usize) -> Option<SentHit> {
        self.tokens += ntok;
        let before = self.win.len();
        let mut pushed = false;
        for ch in text.chars() {
            if ch == '\n' {
                self.unit();
                pushed = true;
            } else if ch.is_whitespace() && self.line.trim_end().ends_with(['.', '!', '?']) && !self.line.trim_start().starts_with("```") {
                self.unit();
                pushed = true;
            } else {
                self.line.push(ch);
            }
        }
        if !pushed || (self.win.len() == before && self.win.len() < self.cfg.window) {
            return None;
        }
        if self.win.len() < self.cfg.window {
            self.sat_since = None;
            return None;
        }
        let (n, distinct, best) = self.recurring();
        let best = best.to_string();
        if (n as f32) < self.cfg.min_fraction * self.cfg.window as f32 {
            self.sat_since = None;
            return None;
        }
        let since = *self.sat_since.get_or_insert(self.win.front().map_or(self.tokens, |w| w.1));
        let span = self.tokens - since;
        (span >= min_tokens).then(|| SentHit { distinct, tokens: span, sample: best })
    }
}

impl Default for LoopCfg {
    fn default() -> Self {
        Self {
            min_period: 1,
            max_period: 512,
            reasoning: LoopRule { min_repeats: 4, min_repeated_tokens: 400 },
            answer: LoopRule { min_repeats: 8, min_repeated_tokens: 400 },
            call: LoopRule { min_repeats: 30, min_repeated_tokens: 3000 },
            sentence: SentCfg::default(),
            call_recover: false,
            recover_temperature: 0.7,
            recover_top_p: 0.95,
        }
    }
}

impl LoopCfg {
    /// Defaults overlaid with the keys present in `v` (`{"min_period", "max_period", "reasoning" | "answer" | "call": {"min_repeats", "min_repeated_tokens"}}`).
    pub fn from_json(v: Option<&Json>) -> Self {
        let mut c = Self::default();
        let Some(v) = v else { return c };
        let num = |j: &Json, k: &str| j.get(k).and_then(Json::as_u64).map(|x| x as usize);
        if let Some(x) = num(v, "min_period") {
            c.min_period = x.max(1);
        }
        if let Some(x) = num(v, "max_period") {
            c.max_period = x.max(c.min_period);
        }
        for (k, r) in [("reasoning", &mut c.reasoning), ("answer", &mut c.answer), ("call", &mut c.call)] {
            if let Some(j) = v.get(k) {
                if let Some(x) = num(j, "min_repeats") {
                    r.min_repeats = x.max(2);
                }
                if let Some(x) = num(j, "min_repeated_tokens") {
                    r.min_repeated_tokens = x;
                }
            }
        }
        c.sentence = SentCfg::from_json(v.get("sentence"));
        if let Some(b) = v.get("call_recover").and_then(Json::as_bool) {
            c.call_recover = b;
        }
        c
    }

    fn fires(&self, r: &LoopRule, c: &Cycle) -> bool {
        (self.min_period..=self.max_period).contains(&c.period) && c.repeats >= r.min_repeats && c.repeated_tokens >= r.min_repeated_tokens
    }
}

/// The periodic tail of the token stream: `repeated_tokens` tail tokens each equal the token `period` positions before it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cycle {
    pub period: usize,
    /// full occurrences of the cycle in the periodic tail (the first one included)
    pub repeats: usize,
    /// tokens of the periodic tail beyond the first occurrence
    pub repeated_tokens: usize,
}

/// Finds the period of the tail. A hash of the last 12 tokens remembers where that 12-gram last ended, which proposes a period; the proposed
/// period is verified by comparing tokens, then tracked incrementally (one comparison a token) for as long as the tail stays periodic.
/// A 12-gram that occurs twice inside one cycle proposes a distance that is not a period: it fails the comparison and the tracked one stays.
pub struct LoopDet {
    toks: Vec<u32>,
    idx: HashMap<u64, usize>,
    /// (period, run): the last `run` tokens each equal the token `period` before
    cand: Option<(usize, usize)>,
}

impl Default for LoopDet {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopDet {
    const N: usize = 12;
    /// longest period ever proposed (rules cap it lower)
    const MAX_D: usize = 4096;

    pub fn new() -> Self {
        Self { toks: Vec::new(), idx: HashMap::new(), cand: None }
    }

    /// Forget everything: a later cycle must build up again from here.
    pub fn restart(&mut self) {
        self.toks.clear();
        self.idx.clear();
        self.cand = None;
    }

    fn run_of(&self, d: usize) -> usize {
        let n = self.toks.len();
        let mut k = 0;
        while n - k > d && self.toks[n - 1 - k] == self.toks[n - 1 - k - d] {
            k += 1;
        }
        k
    }

    pub fn push(&mut self, t: u32) {
        self.toks.push(t);
        let n = self.toks.len();
        if let Some((d, run)) = self.cand {
            self.cand = if self.toks[n - 1] == self.toks[n - 1 - d] { Some((d, run + 1)) } else { None };
        }
        if n < Self::N {
            return;
        }
        let mut h = 0xcbf29ce484222325u64;
        for &x in &self.toks[n - Self::N..] {
            h = (h ^ x as u64).wrapping_mul(0x100000001b3);
        }
        if let Some(pe) = self.idx.insert(h, n) {
            let d = n - pe;
            if d <= Self::MAX_D && self.cand.map_or(true, |(cd, _)| cd != d) {
                let run = self.run_of(d);
                // the longer periodic tail wins; a proposal that is not a period has a short run and loses to a tracked cycle
                if run > 0 && self.cand.map_or(true, |(_, crun)| run > crun) {
                    self.cand = Some((d, run));
                }
            }
        }
    }

    /// The periodic tail, once the cycle has occurred at least twice in full.
    pub fn cycle(&self) -> Option<Cycle> {
        let (d, run) = self.cand?;
        (run >= d).then(|| Cycle { period: d, repeats: run / d + 1, repeated_tokens: run })
    }

    /// Token ids of the first occurrence of the cycle in the periodic tail.
    pub fn cycle_ids(&self) -> &[u32] {
        match self.cand {
            Some((d, run)) => {
                let s = self.toks.len() - run - d;
                &self.toks[s..s + d]
            }
            None => &[],
        }
    }
}

static DETOK: OnceLock<Box<dyn Fn(&[u32]) -> String + Send + Sync>> = OnceLock::new();

/// The server registers its detokenizer once so the breaker's log line can show the cycle as text (engines only see token ids).
pub fn set_detok(f: Box<dyn Fn(&[u32]) -> String + Send + Sync>) {
    let _ = DETOK.set(f);
}

/// First 80 characters of the cycle, for the log line.
fn cycle_text(ids: &[u32]) -> String {
    match DETOK.get() {
        Some(f) => f(ids).chars().take(80).collect(),
        None => format!("token ids {:?}", &ids[..ids.len().min(16)]),
    }
}

/// What the engine must do after a batch of tokens.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Trig {
    None,
    /// close the reasoning block (cap or cycle) and let the model answer
    Close,
    /// a tool-call cycle, first firing, with the recovery on: rewind the sequence by the first number of tokens (to where the cycle
    /// began); the second number is how many of them were held back from the client (the rest, the two occurrences that showed the
    /// cycle, were sent: the text after them continues as the text before them did, so what follows the rewind joins on)
    Rewind(usize, usize),
    /// an answer-phase cycle that came back after the sampling boost: end the request cleanly
    EndLoop,
}

/// What fired during a request (reported in `Finish` and in the request-capture meta file).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhaseEvents {
    /// tokens generated inside the reasoning block (the closing token included)
    pub reasoning_tokens: usize,
    /// the reasoning cap closed the block
    pub reasoning_capped: bool,
    /// loop-breaker firings (reasoning close, answer boost, answer end)
    pub loop_breaks: u32,
    /// of them, firings of the repeated-sentence rule
    pub sentence_breaks: u32,
    /// tool-call cycles answered by taking the held tokens back
    pub rewinds: u32,
}

/// A detokenizer for one tracker (tests; the server's is registered once with [`set_detok`]).
pub type Detok = std::sync::Arc<dyn Fn(&[u32]) -> String + Send + Sync>;

/// Does `toks` hold one of the marks in its last 64 tokens?
pub fn has_call_mark(toks: &[u32], marks: &[Vec<u32>]) -> bool {
    let w = &toks[toks.len().saturating_sub(64)..];
    marks.iter().any(|m| !m.is_empty() && w.windows(m.len()).any(|x| x == &m[..]))
}

/// Tokens of boosted sampling after the first answer-phase firing.
pub const BOOST_TOKENS: usize = 96;

/// Tokens the repeated-sentence rule decodes at a time.
const SENT_BATCH: usize = 16;

/// Per-request phase state: where the generation is (reasoning / answer / tool call), the reasoning cap and the loop breaker.
/// Engine-independent: feed it every generated token (the engine's own forced `</think>` included) and act on the returned [`Trig`].
pub struct PhaseTracker {
    think_end: Option<u32>,
    cap: usize,
    breaker: bool,
    rules: LoopCfg,
    call_marks: Vec<Vec<u32>>,
    in_reasoning: bool,
    in_call: bool,
    generated: Vec<u32>,
    boost_until: usize,
    boosts: usize,
    /// tool-call recovery: switched on by the engine, `generated.len()` when the cycle was first seen (tokens after it are held back),
    /// the stretch after the rewind
    recover_on: bool,
    hold_from: Option<usize>,
    recovering: bool,
    det: LoopDet,
    /// the repeated-sentence rule (None: off, or no detokenizer), the tokens whose text is not complete yet (a split UTF-8
    /// sequence), a hit not acted on yet
    sent: Option<SentDet>,
    pend: Vec<u32>,
    sent_hit: Option<SentHit>,
    detok: Option<Detok>,
    pub events: PhaseEvents,
}

impl PhaseTracker {
    pub fn new(p: &SamplingParams) -> Self {
        Self {
            think_end: p.think_end,
            cap: p.max_reasoning,
            breaker: p.loop_breaker,
            rules: p.loop_rules.clone(),
            call_marks: p.call_marks.clone(),
            in_reasoning: p.think_end.is_some(),
            in_call: p.call_open,
            generated: Vec::new(),
            boost_until: 0,
            boosts: 0,
            recover_on: false,
            hold_from: None,
            recovering: false,
            det: LoopDet::new(),
            sent: (p.loop_breaker && p.loop_rules.sentence.enabled && DETOK.get().is_some()).then(|| SentDet::new(p.loop_rules.sentence)),
            pend: Vec::new(),
            sent_hit: None,
            detok: None,
            events: PhaseEvents::default(),
        }
    }

    /// A tracker with its own detokenizer (the repeated-sentence rule then runs without the server's).
    pub fn with_detok(p: &SamplingParams, detok: Detok) -> Self {
        let mut t = Self::new(p);
        t.sent = (p.loop_breaker && p.loop_rules.sentence.enabled).then(|| SentDet::new(p.loop_rules.sentence));
        t.detok = Some(detok);
        t
    }

    /// One more token for the repeated-sentence rule (outside tool calls only). The text is decoded SENT_BATCH tokens at a time
    /// (a decode call per token cost ~0.5% of a 525 tok/s edit stream): the rule is then judged every SENT_BATCH tokens of the
    /// reply, at token counts that do not depend on how the engine batches its steps.
    fn sentence_token(&mut self, t: u32) {
        let Some(sd) = self.sent.as_mut() else { return };
        self.pend.push(t);
        if self.pend.len() < SENT_BATCH {
            return;
        }
        let text = match (&self.detok, DETOK.get()) {
            (Some(f), _) => f(&self.pend),
            (None, Some(f)) => f(&self.pend),
            (None, None) => return,
        };
        // a character split across tokens: wait for the rest
        if text.ends_with('\u{fffd}') && self.pend.len() < SENT_BATCH + 8 {
            return;
        }
        let n = self.pend.len();
        self.pend.clear();
        let min = if self.in_reasoning { self.rules.sentence.min_tokens } else { self.rules.sentence.answer_min_tokens };
        if let Some(hit) = sd.push_text(&text, n, min) {
            self.sent_hit = Some(hit);
        }
    }

    /// An answer-phase firing: the first boosts the sampling (the pattern must build up again), the second ends the request.
    fn answer_fire(&mut self, what: &str, sentence: bool) -> Trig {
        self.events.loop_breaks += 1;
        if sentence {
            self.events.sentence_breaks += 1;
        }
        if self.boosts == 0 {
            self.boosts = 1;
            self.boost_until = self.generated.len() + BOOST_TOKENS;
            // the cycle has to build up again from here before the request is ended
            self.det.restart();
            if let Some(sd) = self.sent.as_mut() {
                sd.restart();
            }
            tracing::warn!("loop breaker: {what}: boosting the sampling for {BOOST_TOKENS} tokens");
            Trig::None
        } else {
            tracing::warn!("loop breaker: {what}, again after the boost: ending the request");
            Trig::EndLoop
        }
    }

    pub fn in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    /// The next answer token is sampled with the boosted settings.
    pub fn boosting(&self) -> bool {
        !self.in_reasoning && self.generated.len() < self.boost_until
    }

    pub fn in_call(&self) -> bool {
        self.in_call
    }

    /// The engine can hold tokens back and rewind: use the tool-call recovery (`force`: whatever loop_rules.call_recover says).
    pub fn enable_recover(&mut self, force: bool) {
        self.recover_on = self.breaker && (force || self.rules.call_recover);
    }

    /// The tokens fed from now on must not go to the client yet (a cycle is building inside a tool call).
    pub fn holding(&self) -> bool {
        self.hold_from.is_some()
    }

    /// The stretch after a rewind (sampled with the recovery values).
    pub fn recovering(&self) -> bool {
        self.recovering && self.boosting()
    }

    /// Count a batch of generated tokens; the rules are judged on the state after the whole batch.
    pub fn feed(&mut self, toks: &[u32]) -> Trig {
        for &t in toks {
            self.generated.push(t);
            if self.in_reasoning {
                self.events.reasoning_tokens += 1;
                if Some(t) == self.think_end {
                    self.in_reasoning = false;
                    // the answer is judged on its own text
                    self.pend.clear();
                    self.sent_hit = None;
                    if let Some(sd) = self.sent.as_mut() {
                        sd.restart();
                    }
                    if self.breaker {
                        self.det.push(t);
                    }
                    continue;
                }
            }
            if self.breaker {
                self.det.push(t);
                if self.recover_on && !self.in_reasoning && self.boosts == 0 {
                    if self.in_call && self.det.cycle().is_some() {
                        if self.hold_from.is_none() {
                            self.hold_from = Some(self.generated.len());
                        }
                    } else {
                        self.hold_from = None;
                    }
                }
                if self.sent.is_some() {
                    // (a mark can only have just completed when this token ends one)
                    if !self.in_reasoning && !self.in_call && self.call_marks.iter().any(|m| m.last() == Some(&t))
                        && has_call_mark(&self.generated, &self.call_marks) {
                        self.in_call = true;
                        self.sent_hit = None;
                        self.pend.clear();
                    }
                    if !self.in_call {
                        self.sentence_token(t);
                    }
                }
            }
        }
        if self.in_reasoning {
            if self.cap > 0 && self.events.reasoning_tokens >= self.cap {
                tracing::warn!("reasoning cap: {} reasoning tokens, closing the block", self.events.reasoning_tokens);
                self.events.reasoning_capped = true;
                return Trig::Close;
            }
            if self.breaker {
                if let Some(c) = self.det.cycle().filter(|c| self.rules.fires(&self.rules.reasoning, c)) {
                    tracing::warn!(
                        "loop breaker: reasoning cycle, period {} tokens x {} repeats ({} repeated tokens) after {} reasoning tokens: closing the block; cycle: {:?}",
                        c.period, c.repeats, c.repeated_tokens, self.events.reasoning_tokens, cycle_text(self.det.cycle_ids())
                    );
                    self.events.loop_breaks += 1;
                    return Trig::Close;
                }
                if let Some(hit) = self.sent_hit.take() {
                    tracing::warn!(
                        "loop breaker: reasoning wanders, {} sentences keep recurring over {} tokens after {} reasoning tokens: closing the block; one of them: {:?}",
                        hit.distinct, hit.tokens, self.events.reasoning_tokens, hit.sample
                    );
                    self.events.loop_breaks += 1;
                    self.events.sentence_breaks += 1;
                    return Trig::Close;
                }
            }
        } else if self.breaker {
            if !self.in_call && has_call_mark(&self.generated, &self.call_marks) {
                self.in_call = true;
            }
            let rule = if self.in_call { self.rules.call } else { self.rules.answer };
            if let Some(c) = self.det.cycle().filter(|c| self.rules.fires(&rule, c)) {
                let place = if self.in_call { "tool-call" } else { "answer" };
                let text = cycle_text(self.det.cycle_ids());
                self.sent_hit = None;
                if self.in_call && self.boosts == 0 {
                    if let Some(h) = self.hold_from.take().filter(|&h| h < self.generated.len()) {
                        let held = self.generated.len() - h;
                        // back to the start of the cycle's first occurrence
                        let mut n = (c.repeated_tokens + c.period).min(self.generated.len()).max(held);
                        // a short unit repeated many times right before the cycle (the cycle is made of it: a selector or a path that
                        // repeats one piece without end) is part of the loop: go back to where that run began
                        let at0 = self.generated.len() - n;
                        let mut lead = 0usize;
                        for p in 1..=16usize.min(at0) {
                            let mut k = 0usize;
                            while at0 - k > p && self.generated[at0 - 1 - k] == self.generated[at0 - 1 - k - p] {
                                k += 1;
                            }
                            if k >= 8 * p && k + p > lead {
                                lead = k + p;
                            }
                        }
                        n += lead;
                        let win = (4 * c.period + 256).clamp(512, 2048);
                        self.events.loop_breaks += 1;
                        self.events.rewinds += 1;
                        self.boosts = 1;
                        self.recovering = true;
                        let at = self.generated.len() - n;
                        self.generated.truncate(at);
                        self.boost_until = at + win;
                        self.det.restart();
                        tracing::warn!("loop breaker: tool-call cycle, period {} tokens x {} repeats ({} repeated tokens), cycle {text:?}: rewinding {n} tokens to the start of the cycle ({held} of them were held back, {lead} are a repeated lead-in before it) and sampling for {win} tokens", c.period, c.repeats, c.repeated_tokens);
                        return Trig::Rewind(n, held);
                    }
                }
                return self.answer_fire(&format!("{place} cycle, period {} tokens x {} repeats ({} repeated tokens), cycle {text:?}", c.period, c.repeats, c.repeated_tokens), false);
            }
            if let Some(hit) = self.sent_hit.take().filter(|_| !self.in_call) {
                return self.answer_fire(&format!("answer wanders, {} sentences keep recurring over {} tokens, one of them {:?}", hit.distinct, hit.tokens, hit.sample), true);
            }
        }
        Trig::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THINK_END: u32 = 1_000_000;
    const CALL_MARK: [u32; 3] = [1_000_001, 1_000_002, 1_000_003];

    /// A stand-in tokenizer for the fixtures: a word with its leading space is one token, digits and punctuation one token each, as a
    /// byte-pair vocabulary does for common English and code. (The real tokenizer is used by `real_tokenizer_captures`, run by hand.)
    struct Vocab(HashMap<String, u32>);

    impl Vocab {
        fn new() -> Self {
            Self(HashMap::new())
        }
        fn enc(&mut self, text: &str) -> Vec<u32> {
            let mut out = Vec::new();
            let mut cur = String::new();
            let flush = |cur: &mut String, map: &mut HashMap<String, u32>, out: &mut Vec<u32>| {
                if !cur.is_empty() {
                    let n = map.len() as u32;
                    out.push(*map.entry(std::mem::take(cur)).or_insert(n));
                }
            };
            for ch in text.chars() {
                if ch.is_alphabetic() {
                    if !cur.is_empty() && !cur.chars().last().unwrap().is_alphabetic() && cur != " " {
                        flush(&mut cur, &mut self.0, &mut out);
                    }
                    cur.push(ch);
                } else {
                    flush(&mut cur, &mut self.0, &mut out);
                    cur.push(ch);
                    if ch != ' ' {
                        flush(&mut cur, &mut self.0, &mut out);
                    }
                }
            }
            flush(&mut cur, &mut self.0, &mut out);
            out
        }
    }

    fn params(cap: usize) -> SamplingParams {
        SamplingParams { think_end: Some(THINK_END), max_reasoning: cap, loop_breaker: true, call_marks: vec![CALL_MARK.to_vec()], ..Default::default() }
    }

    /// Feed one token at a time (the strictest schedule); the index of the token at which something fired, and what.
    fn run(tr: &mut PhaseTracker, toks: &[u32]) -> Option<(usize, Trig)> {
        for (i, &t) in toks.iter().enumerate() {
            let g = tr.feed(&[t]);
            if g != Trig::None {
                return Some((i + 1, g));
            }
        }
        None
    }

    /// The earlier rule in reasoning: the tail periodic over max(3, ceil(48 / period)) repeats, periods 4..=512.
    fn old_rule_fires(toks: &[u32]) -> Option<usize> {
        let mut idx: HashMap<u64, usize> = HashMap::new();
        for n in 12..=toks.len() {
            let mut h = 0xcbf29ce484222325u64;
            for &x in &toks[n - 12..n] {
                h = (h ^ x as u64).wrapping_mul(0x100000001b3);
            }
            if let Some(pe) = idx.insert(h, n) {
                let d = n - pe;
                let reps = 3usize.max(48usize.div_ceil(d));
                if (4..=512).contains(&d) && n >= d * reps && (n - (reps - 1) * d..n).all(|i| toks[i] == toks[i - d]) {
                    return Some(n);
                }
            }
        }
        None
    }

    const PREAMBLE: &str = "We need to review app.js for bugs. Look at the copy handler first: it calls toast and returns the next state. Then the validation path, where an early return skips the status line. The theme toggle writes the class on the body and then updates the status. Let me walk through each handler and note what could go wrong before choosing the three fixes to apply with the edit tool.\n\n";
    /// The live loop (the three paragraphs that repeated verbatim for 8,800+ tokens), rebuilt from the fragments in the incident notes.
    const REAL_CYCLE: &str = "Potential issue: toast function called from copy; status not updated. Fine.\n\nPotential issue: toast function called from validation failure; status updates. Good.\n\nPotential issue: toast function called from theme toggle; status updates. Good.\n\n";

    #[test]
    fn real_three_paragraph_cycle_is_caught_within_a_few_hundred_tokens() {
        let mut v = Vocab::new();
        let pre = v.enc(PREAMBLE);
        let cyc = v.enc(REAL_CYCLE);
        let mut toks = pre.clone();
        for _ in 0..200 {
            toks.extend_from_slice(&cyc);
        }
        assert!(toks.len() - pre.len() > 8800, "the fixture is as long as the live loop");
        let mut tr = PhaseTracker::new(&params(8192));
        let (at, trig) = run(&mut tr, &toks).expect("the cycle is caught");
        let wasted = at - pre.len() - cyc.len();
        println!("real cycle: period {} tokens, fired after {} tokens of the loop ({} repeated, {} reasoning tokens in all)", cyc.len(), at - pre.len(), wasted, at);
        assert_eq!(trig, Trig::Close);
        assert_eq!(tr.events.loop_breaks, 1);
        assert!(!tr.events.reasoning_capped);
        assert_eq!(tr.det.cycle().unwrap().period, cyc.len(), "the true period, not a sub-span of it");
        // 400 repeated tokens (the reasoning default); a few less when the text before the loop ends like the cycle does (here: the blank line)
        assert!((400 - 8..=400 + LoopDet::N).contains(&wasted), "wasted {wasted}");
        // the batch schedule of speculative decoding (several tokens a round) fires within one round of the same place
        let mut tr = PhaseTracker::new(&params(8192));
        let mut fed = 0;
        let at5 = loop {
            let n = 5.min(toks.len() - fed);
            fed += n;
            if tr.feed(&toks[fed - n..fed]) != Trig::None {
                break fed;
            }
        };
        assert!(at5 >= at && at5 < at + 5);
        // after the forced close the tracker is in the answer phase and quiet
        assert_eq!(tr.feed(&[THINK_END]), Trig::None);
        assert!(!tr.in_reasoning());
    }

    #[test]
    fn reasoning_that_revisits_a_span_does_not_trigger() {
        let mut v = Vocab::new();
        let pre = v.enc(PREAMBLE);
        let tail = v.enc("So the three fixes are clear now. First, update the status after a copy. Second, guard the lookup. Third, keep the count a number. Now write the edit call with exact old text.");
        // 1. a span said exactly three times, then the reasoning moves on (the old rule fired here)
        let span = v.enc("Check handler: getElementById may return null; the early return keeps state. The count uses a default of zero, fine. toast is called only for copy events.\n");
        assert!(span.len() >= 30);
        let mut a = pre.clone();
        for _ in 0..3 {
            a.extend_from_slice(&span);
        }
        a.extend_from_slice(&tail);
        assert!(old_rule_fires(&a).is_some(), "the 19fd2a1 rule fires on a span repeated three times");
        let mut tr = PhaseTracker::new(&params(0));
        assert_eq!(run(&mut tr, &a), None);
        // 2. a long span (several hundred tokens) three times: still three, still not a loop
        let long: Vec<u32> = (0..5).flat_map(|i| v.enc(&format!("Step {i}: {PREAMBLE}"))).collect();
        assert!(long.len() > 300 && long.len() <= 512);
        let mut b = pre.clone();
        for _ in 0..3 {
            b.extend_from_slice(&long);
        }
        b.extend_from_slice(&tail);
        let mut tr = PhaseTracker::new(&params(0));
        assert_eq!(run(&mut tr, &b), None);
        // 3. near-identical paragraphs, forty of them: only the handler number changes
        let mut c = pre.clone();
        for i in 0..40 {
            c.extend(v.enc(&format!("Potential issue: handler{i} reads item-{i} and may get null; status not updated after copy. Needs a guard and a status update.\n\n")));
        }
        c.extend_from_slice(&tail);
        assert!(c.len() > 800);
        let mut tr = PhaseTracker::new(&params(0));
        assert_eq!(run(&mut tr, &c), None);
        // 4. a short line many times but under the repeated-token floor (a drafted table of 20 identical rows)
        let mut d = pre.clone();
        for _ in 0..20 {
            d.extend(v.enc("| ok | ok | ok |\n"));
        }
        d.extend_from_slice(&tail);
        assert!(old_rule_fires(&d).is_some());
        let mut tr = PhaseTracker::new(&params(0));
        assert_eq!(run(&mut tr, &d), None);
        // 5. the same span at three separate places with other text between (never contiguous)
        let mut e = pre.clone();
        for i in 0..3 {
            e.extend_from_slice(&span);
            e.extend(v.enc(&format!("That was pass number {i}; next look at a different part of the file before coming back to it.\n")));
        }
        let mut tr = PhaseTracker::new(&params(0));
        assert_eq!(run(&mut tr, &e), None);
        assert_eq!(tr.events.loop_breaks, 0);
    }

    #[test]
    fn repeats_scale_inversely_with_the_period() {
        // (period, tokens of loop needed to fire) at the reasoning default: max(4 occurrences, first occurrence + 400 repeated tokens)
        for (period, want) in [(1usize, 401usize), (5, 405), (20, 420), (70, 470), (150, 600), (300, 1200), (512, 2048)] {
            let mut toks: Vec<u32> = (0..200).map(|i| 50_000 + i).collect();
            let cyc: Vec<u32> = (0..period as u32).map(|i| 7 + i).collect();
            for _ in 0..(3000 / period + 8) {
                toks.extend_from_slice(&cyc);
            }
            let mut tr = PhaseTracker::new(&params(0));
            let (at, trig) = run(&mut tr, &toks).expect("fires");
            assert_eq!(trig, Trig::Close);
            let need = at - 200;
            assert!(need >= want && need <= want + LoopDet::N, "period {period}: fired after {need} loop tokens, expected {want}");
        }
        // a period above max_period is left to the reasoning cap
        let mut toks: Vec<u32> = Vec::new();
        let cyc: Vec<u32> = (0..600).collect();
        for _ in 0..10 {
            toks.extend_from_slice(&cyc);
        }
        let mut tr = PhaseTracker::new(&params(0));
        assert_eq!(run(&mut tr, &toks), None);
        let mut tr = PhaseTracker::new(&params(4000));
        assert_eq!(run(&mut tr, &toks), Some((4000, Trig::Close)));
        assert!(tr.events.reasoning_capped && tr.events.loop_breaks == 0);
    }

    #[test]
    fn counting_runaway_is_left_to_the_reasoning_cap() {
        // the 48,000-token reasoning runaway of the edit turn: class names that differ only by an incrementing number
        let mut v = Vocab::new();
        let mut toks = v.enc(PREAMBLE);
        for i in 1..4000 {
            toks.extend(v.enc(&format!(".u-font-size-{i}xl\n")));
        }
        assert!(toks.len() > 30_000);
        // breaker alone: silent over the whole runaway
        let mut tr = PhaseTracker::new(&params(0));
        assert_eq!(run(&mut tr, &toks), None);
        assert_eq!(tr.events.loop_breaks, 0);
        // with the cap: closed at exactly the cap, by the cap
        let mut tr = PhaseTracker::new(&params(8192));
        assert_eq!(run(&mut tr, &toks), Some((8192, Trig::Close)));
        assert!(tr.events.reasoning_capped);
        assert_eq!(tr.events.loop_breaks, 0);
        assert_eq!(tr.events.reasoning_tokens, 8192);
        assert_eq!(tr.feed(&[THINK_END]), Trig::None);
        assert!(!tr.in_reasoning());
        assert_eq!(tr.events.reasoning_tokens, 8193);
    }

    /// reasoning, `</think>`, a preamble, the call mark: the stream up to the start of a tool-call argument value
    fn call_prefix(v: &mut Vocab) -> Vec<u32> {
        let mut toks = v.enc("Plan: write the stylesheet with the write tool.");
        toks.push(THINK_END);
        toks.extend(v.enc("Writing the file now.\n\n"));
        toks.extend_from_slice(&CALL_MARK);
        toks.extend(v.enc("write\">\npath: styles.css\ncontent:\n:root { --ink: #1f2d24; }\n"));
        toks
    }

    #[test]
    fn repetition_inside_a_tool_call_argument_does_not_trigger() {
        let mut v = Vocab::new();
        // 50 near-identical rules, then 50 identical ones, then the file goes on
        let mut toks = call_prefix(&mut v);
        let start = toks.len();
        for i in 0..50 {
            toks.extend(v.enc(&format!(".u-m-{i} {{ margin: {i}px; }}\n")));
        }
        let same = v.enc(".grid > .cell { border: 1px solid var(--line); padding: 4px 8px; }\n");
        for _ in 0..50 {
            toks.extend_from_slice(&same);
        }
        toks.extend(v.enc("footer { color: var(--muted); }\n"));
        println!("tool-call value: {} tokens, the identical rule is {} tokens x 50", toks.len() - start, same.len());
        assert!(same.len() * 50 >= 800, "enough identical text to trip the answer rule");
        let mut tr = PhaseTracker::new(&params(8192));
        assert_eq!(run(&mut tr, &toks), None);
        assert!(tr.in_call());
        assert_eq!(tr.events.loop_breaks, 0);
        assert!(!tr.boosting());
        // the same 50 identical rules in a plain answer (no call opened) do reach the answer rule: boost first, never an immediate end
        let mut plain = v.enc("Plan.");
        plain.push(THINK_END);
        for _ in 0..50 {
            plain.extend_from_slice(&same);
        }
        let mut tr = PhaseTracker::new(&params(8192));
        let mut first = None;
        for &t in &plain {
            let g = tr.feed(&[t]);
            if tr.events.loop_breaks == 1 {
                first = Some(g);
                break;
            }
        }
        assert_eq!(first, Some(Trig::None));
        assert!(tr.boosting());
    }

    /// The recovery (engines that hold tokens back): nothing is held while the text does not repeat; a short repeat is held and released;
    /// a runaway is taken back to where the cycle was first seen, sampled for a stretch, and ends the request if it builds up again.
    #[test]
    fn tool_call_recovery_holds_takes_back_then_ends() {
        let mut v = Vocab::new();
        let mut toks = call_prefix(&mut v);
        let same = v.enc(".grid > .cell { border: 1px solid var(--line); padding: 4px 8px; }\n");
        // three identical rules, then different text: held for a while, released, nothing fires
        let mut legit = toks.clone();
        for _ in 0..3 {
            legit.extend_from_slice(&same);
        }
        legit.extend(v.enc(".footer { margin: 0 auto; color: var(--ink); }\n"));
        let mut tr = PhaseTracker::new(&params(8192));
        tr.enable_recover(true);
        let mut held_some = false;
        for &t in &legit {
            assert_eq!(tr.feed(&[t]), Trig::None);
            held_some |= tr.holding();
        }
        assert!(held_some && !tr.holding() && tr.events.loop_breaks == 0);
        // the runaway
        let start = toks.len();
        for _ in 0..700 {
            toks.extend_from_slice(&same);
        }
        let mut tr = PhaseTracker::new(&params(8192));
        tr.enable_recover(true);
        let mut fed = 0usize;
        let mut hold_at = None;
        let mut rewound = None;
        let mut ended = false;
        for (i, &t) in toks.iter().enumerate() {
            let was = tr.holding();
            let g = tr.feed(&[t]);
            fed += 1;
            if !was && tr.holding() && hold_at.is_none() {
                hold_at = Some(fed);
            }
            match g {
                Trig::Rewind(n, held) => {
                    assert!(rewound.is_none(), "one rewind only");
                    // held: exactly the tokens fed since the hold began; the rewind goes on to the start of the cycle
                    assert_eq!(held, fed - hold_at.unwrap());
                    assert!(n > held && fed - n <= start && fed - n + 8 >= start, "rewind to {} (cycle starts at {start})", fed - n);
                    assert!(i + 1 - start >= 3000 && !tr.holding() && tr.boosting() && tr.recovering());
                    rewound = Some(n);
                    fed -= n;
                }
                Trig::EndLoop => {
                    ended = true;
                    break;
                }
                g => assert_eq!(g, Trig::None),
            }
        }
        let n = rewound.expect("rewind");
        assert!(hold_at.unwrap() - start <= 2 * same.len() + LoopDet::N, "hold began at {:?}", hold_at);
        assert!(n >= 3000 - same.len() && ended && !tr.recovering());
        assert_eq!((tr.events.loop_breaks, tr.events.rewinds), (2, 1));
        // without the engine's opt-in nothing changes
        let mut tr = PhaseTracker::new(&params(8192));
        assert!(toks.iter().all(|&t| { let g = tr.feed(&[t]); !tr.holding() && !matches!(g, Trig::Rewind(..)) && g != Trig::Close }));
    }

    #[test]
    fn runaway_inside_a_tool_call_boosts_then_ends() {
        let mut v = Vocab::new();
        let mut toks = call_prefix(&mut v);
        let start = toks.len();
        let same = v.enc(".grid > .cell { border: 1px solid var(--line); padding: 4px 8px; }\n");
        for _ in 0..700 {
            toks.extend_from_slice(&same);
        }
        let mut tr = PhaseTracker::new(&params(8192));
        let mut boosted_at = None;
        let mut ended_at = None;
        for (i, &t) in toks.iter().enumerate() {
            let g = tr.feed(&[t]);
            if boosted_at.is_none() && tr.boosting() {
                boosted_at = Some(i + 1 - start);
            }
            if g == Trig::EndLoop {
                ended_at = Some(i + 1 - start);
                break;
            }
            assert_eq!(g, Trig::None);
        }
        let (b, e) = (boosted_at.expect("boost"), ended_at.expect("end"));
        // 3000 repeated tokens before the boost, and the cycle has to build up in full again (not one token later) before the end
        assert!(b >= 3000 && b <= 3000 + same.len() + LoopDet::N, "boost at {b}");
        assert!(e >= b + 3000, "end at {e}, boost at {b}");
        assert_eq!(tr.events.loop_breaks, 2);
    }

    #[test]
    fn sub_span_repeats_inside_a_cycle_do_not_hide_it() {
        // every 12-gram of the cycle body occurs twice inside one cycle (A A B): the hash proposes the short distance on most tokens
        let a: Vec<u32> = (100..140).collect();
        let b: Vec<u32> = (200..215).collect();
        let cyc: Vec<u32> = [&a[..], &a[..], &b[..]].concat();
        let mut toks: Vec<u32> = (9000..9100).collect();
        for _ in 0..12 {
            toks.extend_from_slice(&cyc);
        }
        let mut d = LoopDet::new();
        let mut seen = None;
        for (i, &t) in toks.iter().enumerate() {
            d.push(t);
            if let Some(c) = d.cycle() {
                if c.period == cyc.len() && c.repeats >= 4 && seen.is_none() {
                    seen = Some(i + 1 - 100);
                }
            }
        }
        assert_eq!(seen, Some(4 * cyc.len()), "four occurrences are seen as soon as the fourth is complete");
        assert_eq!(d.cycle_ids(), &cyc[..]);
    }

    #[test]
    fn thresholds_come_from_json() {
        let j: Json = serde_json::from_str(r#"{"max_period": 1024, "reasoning": {"min_repeats": 3, "min_repeated_tokens": 600}, "call": {"min_repeated_tokens": 10000}}"#).unwrap();
        let c = LoopCfg::from_json(Some(&j));
        assert_eq!(c.max_period, 1024);
        assert_eq!(c.reasoning, LoopRule { min_repeats: 3, min_repeated_tokens: 600 });
        assert_eq!(c.answer, LoopCfg::default().answer);
        assert_eq!(c.call, LoopRule { min_repeats: 30, min_repeated_tokens: 10000 });
        assert_eq!(LoopCfg::from_json(None), LoopCfg::default());
        // the configured rule is the one applied
        let mut p = params(0);
        p.loop_rules = c;
        let cyc: Vec<u32> = (0..70).collect();
        let mut toks = Vec::new();
        for _ in 0..40 {
            toks.extend_from_slice(&cyc);
        }
        let (at, _) = run(&mut PhaseTracker::new(&p), &toks).unwrap();
        assert!((670..=670 + LoopDet::N).contains(&at), "fired at {at}");
        // breaker off: nothing fires
        let mut p = params(0);
        p.loop_breaker = false;
        assert_eq!(run(&mut PhaseTracker::new(&p), &toks), None);
    }

    #[test]
    fn call_marks_found_in_the_recent_window() {
        let marks = vec![vec![7u32, 8, 9]];
        assert!(has_call_mark(&[1, 2, 7, 8, 9, 4], &marks));
        assert!(!has_call_mark(&[1, 2, 7, 8, 4, 9], &marks));
    }

    // ---- the repeated-sentence rule ----

    const WANDER: &str = include_str!("../tests/fixtures/wander_reasoning.txt");
    const PAGE: &str = include_str!("../tests/fixtures/model_page.html.txt");
    const TABLES: &str = include_str!("../tests/fixtures/model_tables.md.txt");

    /// Feed text a word at a time (a word with its leading space as one token, as the stand-in vocabulary does); the token count
    /// at which the rule fired.
    fn sent_run(cfg: SentCfg, text: &str, min_tokens: usize) -> Option<(usize, SentHit)> {
        let mut sd = SentDet::new(cfg);
        let mut n = 0;
        let mut word = String::new();
        let mut flush = |w: &mut String, n: &mut usize| -> Option<SentHit> {
            if w.is_empty() {
                return None;
            }
            *n += 1;
            let h = sd.push_text(w, 1, min_tokens);
            w.clear();
            h
        };
        for ch in text.chars() {
            if ch.is_whitespace() {
                if let Some(h) = flush(&mut word, &mut n) {
                    return Some((n, h));
                }
            }
            word.push(ch);
        }
        flush(&mut word, &mut n).map(|h| (n, h))
    }

    #[test]
    fn wandering_candidates_are_caught_and_the_exact_cycle_rule_misses_them() {
        // candidate lines tried and rejected in shifting order, under changing labels, for ~2,900 words: no exact cycle
        let mut v = Vocab::new();
        let toks = v.enc(WANDER);
        assert!(toks.len() > 2500, "{}", toks.len());
        let mut p = params(0);
        p.loop_rules.sentence.enabled = false;
        assert_eq!(run(&mut PhaseTracker::new(&p), &toks), None, "the exact tail-cycle rule must not see a cycle in it");
        // the sentence rule: saturated from the first rounds, fires once it has lasted min_tokens
        let cfg = SentCfg::default();
        let (at, hit) = sent_run(cfg, WANDER, cfg.min_tokens).expect("the wandering transcript is caught");
        assert!(at <= 1000, "fired after {at} words");
        assert!(hit.distinct >= 3 && hit.distinct <= 16, "{hit:?}");
    }

    #[test]
    fn real_code_and_tables_never_trigger_even_unfenced() {
        // a long page (HTML + CSS + JS) and a table-heavy markdown answer, both written by the model (tests/fixtures)
        let cfg = SentCfg::default();
        assert!(PAGE.len() > 20_000 && TABLES.len() > 6_000, "{} {}", PAGE.len(), TABLES.len());
        for (name, text) in [("page", PAGE), ("tables", TABLES)] {
            // as an answer (fences honoured), with the fences removed (every line judged), and at a third of the token threshold
            let bare: String = text.lines().filter(|l| !l.trim_start().starts_with("```")).map(|l| format!("{l}\n")).collect();
            for (kind, t) in [("fenced", text.to_string()), ("unfenced", bare)] {
                for min in [cfg.answer_min_tokens, cfg.min_tokens, cfg.min_tokens / 3] {
                    assert_eq!(sent_run(cfg, &t, min), None, "{name} ({kind}, min_tokens {min})");
                }
            }
        }
    }

    #[test]
    fn fenced_repetition_is_exempt_and_prose_after_it_is_judged() {
        let cfg = SentCfg::default();
        let rows: String = (0..400).map(|i| format!("    <li class=\"item\"><span>entry</span><b>value {}</b></li>\n", i % 3)).collect();
        // 400 near-identical rows: the rule would fire on them as text ...
        assert!(sent_run(cfg, &rows, cfg.min_tokens).is_some());
        // ... never inside a fence (``` or ~~~, with a language tag)
        for fence in ["```", "~~~"] {
            let fenced = format!("Here is the list.\n\n{fence}html\n{rows}{fence}\n\nThat is the whole file.\n");
            assert_eq!(sent_run(cfg, &fenced, cfg.min_tokens), None);
            // and the text after the fence is judged again
            assert!(sent_run(cfg, &format!("{fenced}{WANDER}"), cfg.min_tokens).is_some());
        }
    }

    #[test]
    fn ordinary_reasoning_and_short_lines_do_not_trigger() {
        let cfg = SentCfg::default();
        // prose that says a few things two or three times while moving on
        let mut text = String::from(PREAMBLE);
        for i in 0..60 {
            text.push_str(&format!("Step {i}: handler number {i} reads the state and writes field {} back. That looks correct. Let me check the next one.\n", i * 7 % 13));
            if i % 9 == 0 {
                text.push_str(REAL_CYCLE);
            }
        }
        assert_eq!(sent_run(cfg, &text, cfg.min_tokens), None);
        // an enumeration: the same finding for every numbered item differs by its counter and is left to the reasoning cap
        let walk: String = (0..300).map(|i| format!("Handler {i}: the status line is not updated after the copy, which is a bug.\nSame problem in the next handler as well.\n")).collect();
        assert_eq!(sent_run(cfg, &walk, cfg.min_tokens), None);
        // ... also when the item's own part is short and two identical remarks follow each one
        let walk: String = (0..300).map(|i| format!("Handler {i}: fine. No change is needed in this one. Moving on to the next handler now.\n")).collect();
        assert_eq!(sent_run(cfg, &walk, cfg.min_tokens), None);
        // short lines (under min_chars letters) never count, however often they repeat
        let short: String = (0..600).map(|i| if i % 2 == 0 { "No.\n" } else { "Not quite. Wait.\n" }).collect();
        assert_eq!(sent_run(cfg, &short, 10), None);
    }

    #[test]
    fn sentence_thresholds_come_from_json() {
        let j: Json = serde_json::from_str(r#"{"sentence": {"enabled": false, "window": 48, "min_chars": 20, "min_count": 4, "min_fraction": 0.6, "min_tokens": 900, "answer_min_tokens": 2000}}"#).unwrap();
        let c = LoopCfg::from_json(Some(&j));
        assert_eq!(c.sentence, SentCfg { enabled: false, window: 48, min_chars: 20, min_count: 4, min_fraction: 0.6, min_tokens: 900, answer_min_tokens: 2000 });
        assert_eq!(LoopCfg::from_json(Some(&serde_json::json!({}))).sentence, SentCfg::default());
        assert_eq!(c.reasoning, LoopCfg::default().reasoning);
    }

    /// The stand-in vocabulary as a detokenizer.
    fn detok_of(v: &Vocab) -> Detok {
        let mut rev = vec![String::new(); v.0.len()];
        for (s, &i) in &v.0 {
            rev[i as usize] = s.clone();
        }
        std::sync::Arc::new(move |ids: &[u32]| ids.iter().map(|&i| rev.get(i as usize).cloned().unwrap_or_default()).collect())
    }

    #[test]
    fn tracker_closes_a_wandering_reasoning_block_and_boosts_then_ends_an_answer() {
        let mut v = Vocab::new();
        let toks = v.enc(WANDER);
        let d = detok_of(&v);
        // reasoning: the block is closed
        let p = params(0);
        let mut tr = PhaseTracker::with_detok(&p, d.clone());
        let (at, trig) = run(&mut tr, &toks).expect("fires");
        assert_eq!(trig, Trig::Close);
        assert!(at <= 1200, "fired at token {at}");
        assert!(tr.events.loop_breaks == 1 && tr.events.sentence_breaks == 1 && !tr.events.reasoning_capped);
        // answer (no reasoning block): the first firing boosts, the second ends the request
        let mut pa = params(0);
        pa.think_end = None;
        let mut tr = PhaseTracker::with_detok(&pa, d.clone());
        let mut fired = None;
        let mut boost_at = None;
        for (i, &t) in toks.iter().cycle().take(3 * toks.len()).enumerate() {
            let g = tr.feed(&[t]);
            if boost_at.is_none() && tr.boosting() {
                boost_at = Some(i + 1);
            }
            if g != Trig::None {
                fired = Some((i + 1, g));
                break;
            }
        }
        let boost_at = boost_at.expect("the answer rule boosts first");
        let (end_at, trig) = fired.expect("then ends");
        assert_eq!(trig, Trig::EndLoop);
        assert!(boost_at >= pa.loop_rules.sentence.answer_min_tokens && end_at >= boost_at + pa.loop_rules.sentence.answer_min_tokens, "{boost_at} {end_at}");
        assert_eq!((tr.events.loop_breaks, tr.events.sentence_breaks), (2, 2));
        // the rule off: nothing fires
        let mut off = params(0);
        off.loop_rules.sentence.enabled = false;
        assert_eq!(run(&mut PhaseTracker::with_detok(&off, d.clone()), &toks), None);
        // inside a tool call (argument text): never
        let mut pc = params(0);
        pc.think_end = None;
        let mut call = CALL_MARK.to_vec();
        call.extend_from_slice(&toks);
        let mut tr = PhaseTracker::with_detok(&pc, d);
        assert_eq!(run(&mut tr, &call), None);
        assert!(tr.in_call());
    }

    /// Run by hand with the model's tokenizer and the saved captures:
    /// `FE_TEST_MODEL_DIR=<model dir> FE_TEST_CAPTURE=<reasoning text file> cargo test -p fe-server real_tokenizer_captures -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_tokenizer_captures() {
        let dir = std::env::var("FE_TEST_MODEL_DIR").expect("FE_TEST_MODEL_DIR");
        let tok = crate::tokenizer::Tok::from_dir(std::path::Path::new(&dir)).unwrap();
        let te = tok.token_id("</think>").expect("</think> id");
        // thresholds, cap and call marks as the server reads them from the model dir's sampling.json
        let ph = crate::server::PhaseDefaults::from_dir(std::path::Path::new(&dir), &tok);
        println!("REAL TOKENIZER sampling.json: cap {}, breaker {}, rules {:?}, {} call marks", ph.max_reasoning, ph.loop_breaker, ph.loop_rules, ph.call_marks.len());
        let p = SamplingParams { think_end: Some(te), max_reasoning: ph.max_reasoning, loop_breaker: true, loop_rules: ph.loop_rules.clone(), call_marks: ph.call_marks.clone(), ..Default::default() };
        // the live cycle
        let pre = tok.encode(PREAMBLE).unwrap();
        let cyc_len = tok.encode(&REAL_CYCLE.repeat(3)).unwrap().len() - tok.encode(&REAL_CYCLE.repeat(2)).unwrap().len();
        let toks = tok.encode(&format!("{PREAMBLE}{}", REAL_CYCLE.repeat(200))).unwrap();
        let mut tr = PhaseTracker::new(&p);
        let (at, trig) = run(&mut tr, &toks).expect("the cycle is caught");
        println!("REAL TOKENIZER live cycle: period {cyc_len} tokens ({} in the stream), fired {trig:?} after {} tokens of the loop, loop total {} tokens; cycle {:?}", tr.det.cycle().unwrap().period, at - pre.len(), toks.len() - pre.len(), tok.decode(tr.det.cycle_ids(), false).unwrap());
        assert_eq!(trig, Trig::Close);
        assert!(tr.events.loop_breaks == 1 && !tr.events.reasoning_capped);
        // a saved reasoning capture (the counting runaway): the breaker stays silent, the cap closes it
        if let Ok(f) = std::env::var("FE_TEST_CAPTURE") {
            let text = std::fs::read_to_string(&f).unwrap();
            let toks = tok.encode(&text).unwrap();
            let mut off = p.clone();
            off.max_reasoning = 0;
            let mut tr = PhaseTracker::new(&off);
            let r = run(&mut tr, &toks);
            println!("REAL TOKENIZER capture {f}: {} tokens, breaker alone: {r:?}, old rule fires at {:?}", toks.len(), old_rule_fires(&toks));
            let mut tr = PhaseTracker::new(&p);
            let r2 = run(&mut tr, &toks);
            println!("REAL TOKENIZER capture {f}: with the cap: {r2:?}, capped {}, breaker firings {}", tr.events.reasoning_capped, tr.events.loop_breaks);
        }
    }
}
