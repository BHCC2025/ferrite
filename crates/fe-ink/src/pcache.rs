//! Prompt cache of the single-slot Inkling-2-Mini engine: a request whose prompt starts with tokens the engine prefilled before
//! resumes there and prefills only the rest.
//!
//! What a resume at position m reuses in place: the global layers' K/V (5, 11, 17, 23: indexed by position and only appended to, so rows
//! below m stay valid until a later request writes those positions again) and the full draft layer's cache (same reasoning). What it
//! cannot reuse in place: the sliding layers' K/V rings (1024 + chunk rows, reused modulo), the three-row conv tails of every layer
//! application and the draft head's two sliding rings, which the rows after m overwrite. Those are copied to pinned host memory
//! ("snapshots", [`PCache::snap_bytes`] each) at marks while a prompt is prefilled:
//! - prompt end - 1: in chat use the next prompt repeats this whole prompt (the template has no generation prompt; the reply and the new
//!   message follow), and one row before the end also serves an identical request, which must recompute its last row for the logits;
//! - every `every` positions (INK_PCACHE_EVERY, default 8192): a prompt that differs from an earlier one inside its history resumes from
//!   the last mark before the difference.
//! A mark inside a prefill chunk costs no extra forward: that chunk runs as a speculative chunk (its conv inputs are kept after the state
//! rows, `Ink::spec`), so the conv tails at any of its rows can be read, and the rings still hold the 1024 positions before every row of it.
//! The draft context is extended up to each mark, copied, then extended further.
//!
//! Only prompt rows are ever reused: generated rows were computed by the decode family (bf16 MoE activations instead of the prompt
//! family's quantised ones), so the reply that the template re-renders into the next prompt is prefilled again like any new text.
//! The result is bit-identical to a fresh engine: prompt-family rows do not depend on the chunking (`ink-run selftest`), and the restored
//! state is a byte copy of the state a fresh prefill holds at that position (`ink-run pcache` checks both).
//!
//! Invariant: every snapshot's mark is <= the number of leading tokens of `toks` whose K/V sits in the global caches. A request that
//! resumes at m drops the snapshots above m (it overwrites those positions); a request that cannot resume resets the engine (which
//! unmaps the lazily mapped K/V) and drops them all.
use crate::draft::DRAFT_SNAP_BYTES;
use crate::engine::Ink;
use crate::spec::Spec;
use crate::D;
use anyhow::{ensure, Result};
use fe_cuda::HostBuffer;
use std::time::Instant;

struct Snap {
    mark: usize,
    /// the draft context was fed up to `mark` too (the head was loaded while this prompt was prefilled)
    draft: bool,
    buf: HostBuffer<u8>,
}

/// What the last request did.
#[derive(Default, Clone, Debug)]
pub struct Stats {
    /// prompt tokens reused (the resume position; 0 = full prefill)
    pub resumed_at: usize,
    /// leading tokens shared with the previous prompt
    pub common: usize,
    pub restore_ms: f64,
    pub snaps_taken: usize,
    pub snap_ms: f64,
}

pub struct PCache {
    pub on: bool,
    /// periodic marks (positions); 0 = only the prompt-end mark
    pub every: usize,
    /// snapshots held at most (INK_PCACHE_GB of pinned host memory)
    pub max_snaps: usize,
    toks: Vec<u32>,
    snaps: Vec<Snap>,
    free: Vec<HostBuffer<u8>>,
    ink_bytes: usize,
    bytes: usize,
    pub last: Stats,
}

fn env_usize(n: &str) -> Option<usize> {
    std::env::var(n).ok().and_then(|v| v.parse().ok())
}

impl PCache {
    /// INK_PCACHE=0 (or false / off / no) turns the cache off (every request resets the engine and prefills its whole prompt, as before);
    /// INK_PCACHE_EVERY=n periodic marks (default 8192, 0 = none); INK_PCACHE_GB=x pinned host memory for snapshots (default 2).
    pub fn new(ink: &Ink) -> PCache {
        let on = std::env::var("INK_PCACHE").map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off" | "no")).unwrap_or(true);
        let every = env_usize("INK_PCACHE_EVERY").unwrap_or(8192);
        let ink_bytes = ink.snap_bytes();
        let bytes = ink_bytes + DRAFT_SNAP_BYTES;
        let gb: f64 = std::env::var("INK_PCACHE_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0);
        let max_snaps = ((gb * 1073741824.0) as usize / bytes).max(2);
        PCache { on, every, max_snaps, toks: vec![], snaps: vec![], free: vec![], ink_bytes, bytes, last: Stats::default() }
    }

    /// Bytes of one snapshot (engine part + draft part).
    pub fn snap_bytes(&self) -> usize {
        self.bytes
    }

    /// Pinned host memory allocated for snapshots (held + free for reuse).
    pub fn host_bytes(&self) -> usize {
        (self.snaps.len() + self.free.len()) * self.bytes
    }

    /// Snapshots held.
    pub fn len(&self) -> usize {
        self.snaps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snaps.is_empty()
    }

    /// Marks of the snapshots held (ascending).
    pub fn marks(&self) -> Vec<usize> {
        self.snaps.iter().map(|s| s.mark).collect()
    }

    /// Forget everything (after `Ink::reset` from outside: the global K/V is gone). The pinned buffers are kept for reuse.
    pub fn clear(&mut self) {
        for s in self.snaps.drain(..) {
            self.free.push(s.buf);
        }
        self.toks.clear();
    }

    /// Prepare the engine for `prompt`: restore the snapshot at the longest usable mark (returns that position: the prompt tokens that
    /// need no prefill), or reset the engine (returns 0). `spec`: the draft head when this request's prefill feeds its context (it is then
    /// restored with the engine, or reset with it); None = the engine alone (`Ink::reset`, as a request without the head).
    pub fn begin(&mut self, ink: &mut Ink, spec: Option<&mut Spec>, prompt: &[u32]) -> Result<usize> {
        let r = self.begin_inner(ink, spec, prompt);
        if r.is_err() {
            self.clear();
        }
        r
    }

    fn begin_inner(&mut self, ink: &mut Ink, spec: Option<&mut Spec>, prompt: &[u32]) -> Result<usize> {
        let t0 = Instant::now();
        self.last = Stats::default();
        let n = prompt.len();
        ensure!(n > 0, "empty prompt");
        let common = self.toks.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        self.last.common = common;
        // at least the last prompt row is computed again (its logits give the first token)
        let limit = common.min(n - 1);
        let need_draft = spec.is_some();
        let pick = if self.on { self.snaps.iter().rposition(|s| s.mark > 0 && s.mark <= limit && (s.draft || !need_draft)) } else { None };
        let Some(i) = pick else {
            self.clear();
            match spec {
                Some(sp) => sp.reset(ink)?,
                None => ink.reset()?,
            }
            self.toks = prompt.to_vec();
            return Ok(0);
        };
        // the rows above the resume point are written again by this request: those snapshots go
        let dropped: Vec<Snap> = self.snaps.drain(i + 1..).collect();
        for s in dropped {
            self.free.push(s.buf);
        }
        let s = &self.snaps[i];
        let mark = s.mark;
        ink.restore(&s.buf.as_slice()[..self.ink_bytes], mark)?;
        if let Some(sp) = spec {
            if std::env::var("INK_PCACHE_PLANT").ok().as_deref() != Some("draft") {
                sp.draft.restore(&ink.s, &s.buf.as_slice()[self.ink_bytes..], mark)?;
            } else {
                // planted fault for the gate's control run: the draft rings are NOT restored (ctx_len only)
                sp.draft.restore_len_only(mark);
            }
            sp.reset_policy();
        }
        self.toks = prompt.to_vec();
        self.last.resumed_at = mark;
        self.last.restore_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(mark)
    }

    /// A buffer for one more snapshot (`pending` = snapshots of the current chunk not pushed yet): evicts one when the budget is full
    /// (the one whose removal leaves the smallest gap between marks; never the newest), reuses a freed buffer or allocates pinned memory.
    fn take_buf(&mut self, pending: usize) -> Option<HostBuffer<u8>> {
        while self.snaps.len() + pending + 1 > self.max_snaps && self.snaps.len() > 1 {
            let k = self.snaps.len();
            let mut best = (usize::MAX, 0usize);
            for i in 0..k - 1 {
                let prev = if i == 0 { 0 } else { self.snaps[i - 1].mark };
                let gap = self.snaps[i + 1].mark - prev;
                if gap < best.0 {
                    best = (gap, i);
                }
            }
            let s = self.snaps.remove(best.1);
            self.free.push(s.buf);
        }
        if self.snaps.len() + pending + 1 > self.max_snaps {
            return None;
        }
        if let Some(b) = self.free.pop() {
            return Some(b);
        }
        match HostBuffer::<u8>::alloc(self.bytes) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("ink pcache: pinned host buffer of {} MiB not allocated ({e:#}): snapshot skipped", self.bytes >> 20);
                None
            }
        }
    }

    /// Prefill `prompt[from..]` (the engine at position `from`, as left by [`PCache::begin`]) in chunks of `ink.tmax` prompt rows, feeding the
    /// draft context when `spec` is given and taking the snapshots at the marks. `stop()` is called before every chunk; true = stop
    /// (cancelled, client gone): Ok(None), the snapshots taken so far stay valid. Returns the bf16 logits of the last prompt row.
    pub fn prefill(&mut self, ink: &mut Ink, spec: Option<&mut Spec>, prompt: &[u32], from: usize, stop: impl FnMut() -> bool) -> Result<Option<Vec<u16>>> {
        let r = self.prefill_inner(ink, spec, prompt, from, stop);
        ink.spec = false;
        if r.is_err() {
            self.clear();
        }
        r
    }

    fn prefill_inner(&mut self, ink: &mut Ink, mut spec: Option<&mut Spec>, prompt: &[u32], from: usize, mut stop: impl FnMut() -> bool) -> Result<Option<Vec<u16>>> {
        let n = prompt.len();
        ensure!(from < n && ink.len == from, "prefill from {from}: engine at {}, prompt of {n}", ink.len);
        let mut marks: Vec<usize> = vec![];
        if self.on {
            if self.every > 0 {
                let mut m = (from / self.every + 1) * self.every;
                while m < n - 1 {
                    marks.push(m);
                    m += self.every;
                }
            }
            if n - 1 > from {
                marks.push(n - 1);
            }
        }
        ink.prompt = true;
        if let Some(sp) = spec.as_deref() {
            ensure!(sp.draft.ctx_len == from, "draft context at {} for a prefill from {from}", sp.draft.ctx_len);
            ink.draft_taps = true;
        }
        let (mut pos, mut lg, mut mi) = (from, vec![], 0usize);
        let row_b = 4 * D * 2;
        while pos < n {
            if stop() {
                return Ok(None);
            }
            let t = (n - pos).min(ink.tmax);
            let ms: Vec<usize> = marks[mi..].iter().take_while(|&&m| m <= pos + t).copied().collect();
            mi += ms.len();
            ink.spec = !ms.is_empty();
            let r = ink.forward(&prompt[pos..pos + t], 1, None, None);
            ink.spec = false;
            lg = r?;
            let ts = Instant::now();
            let mut bufs: Vec<Option<HostBuffer<u8>>> = vec![];
            for k in 0..ms.len() {
                bufs.push(self.take_buf(k));
            }
            if let Some(sp) = spec.as_deref_mut() {
                let mut d0 = pos;
                for (k, &m) in ms.iter().enumerate() {
                    if m > d0 {
                        sp.draft.extend(ink, ink.dtaps.ptr().byte_add((d0 - pos) * row_b), m - d0, d0)?;
                        d0 = m;
                    }
                    if let Some(b) = bufs[k].as_mut() {
                        let ib = self.ink_bytes;
                        sp.draft.snapshot(&ink.s, &mut b.as_mut_slice()[ib..])?;
                    }
                }
                if pos + t > d0 {
                    sp.draft.extend(ink, ink.dtaps.ptr().byte_add((d0 - pos) * row_b), pos + t - d0, d0)?;
                }
                sp.set_last_taps(ink.dtaps.ptr().byte_add((t - 1) * row_b));
            }
            for (k, &m) in ms.iter().enumerate() {
                if let Some(mut b) = bufs[k].take() {
                    let ib = self.ink_bytes;
                    ink.snapshot(&mut b.as_mut_slice()[..ib], m)?;
                    self.snaps.push(Snap { mark: m, draft: spec.is_some(), buf: b });
                    self.last.snaps_taken += 1;
                }
            }
            if !ms.is_empty() {
                self.last.snap_ms += ts.elapsed().as_secs_f64() * 1e3;
            }
            if ink.pending_rows() > 0 {
                // keep every row of the chunk: the conv tails as of its last row
                ink.rollback(t)?;
            }
            pos += t;
        }
        Ok(Some(lg))
    }
}
