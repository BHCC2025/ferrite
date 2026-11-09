//! TP2 serving: rank 0 drives the engine (scheduler / server / CLI), rank 1 follows. Every state-changing
//! Engine call on rank 0 (forward_tree / forward_ex, commit / commit_tree, reset_slot, mtp_forward) is sent over the bootstrap link before
//! rank 0 runs it; rank 1's `Engine::follow` loop replays the calls in order, so both ranks keep identical
//! caches and meet in every reduction. Read-only calls (argmax, logits_host) stay on rank 0.
use anyhow::{bail, Result};

use crate::model::{MtpRow, MtpSrc};

pub enum Call {
    /// `parents[i]`: segment i is a verify tree (Engine::forward_tree)
    Forward { segs: Vec<(usize, Vec<u32>)>, logit_rows: Vec<usize>, hist: bool, dec: usize, parents: Vec<Option<Vec<i32>>> },
    Commit { slot: usize, start: usize, n: usize, keep: usize },
    CommitTree { slot: usize, start: usize, n: usize, path: Vec<usize>, taps: Option<usize> },
    Reset { slot: usize },
    Mtp { rows: Vec<MtpRow>, draw: Vec<usize> },
    /// TP-split block drafter: rank 1's half of a draft over (slot, anchor, pos, tree budget) streams
    Draft { reqs: Vec<(usize, u32, usize, usize)> },
    /// TP-split block drafter: absorb (slot, feature-tap row, rows, first position) of each stream
    Absorb { items: Vec<(usize, usize, usize, usize)> },
    Stop,
    /// prompt cache: (re)allocate the park pool (Engine::park_pool)
    ParkPool { bytes: usize },
    /// prompt cache: a slot's KV tokens [a, b) (+ recurrent state) to the park pool at `off` (Engine::park_save)
    ParkSave { slot: usize, a: usize, b: usize, gdn: bool, off: usize },
    /// prompt cache: back from the pool, the slot's length set to `len` (Engine::park_restore)
    ParkRestore { slot: usize, a: usize, b: usize, gdn: bool, off: usize, len: usize },
    /// prompt cache: slot to slot (Engine::copy_slot)
    CopySlot { src: usize, dst: usize, a: usize, b: usize, gdn: bool, len: usize },
    /// prompt cache: a snapshot (the slot's recurrent state at `pos`) into the state block at byte `off`
    /// (Engine::cache_save); the follower's drafter saves its part at `off + park_gdn_bytes()` too
    CacheSave { slot: usize, pos: usize, off: usize },
    /// prompt cache: the reverse (Engine::cache_restore), the slot's length set to `pos`
    CacheRestore { slot: usize, pos: usize, off: usize },
    /// prompt cache: KV rows [0, rows) out to fixed-stride KV blocks at byte offsets `offs` (Engine::kv_out)
    KvOut { slot: usize, rows: usize, offs: Vec<usize> },
    /// prompt cache: the reverse, any rows <= what the blocks hold (Engine::kv_in)
    KvIn { slot: usize, rows: usize, offs: Vec<usize> },
}

impl Call {
    /// The variant's index, in declaration order (Engine::follow's FE_FOLLOW_PROF table).
    pub fn kind(&self) -> usize {
        match self {
            Call::Forward { .. } => 0,
            Call::Commit { .. } => 1,
            Call::CommitTree { .. } => 2,
            Call::Reset { .. } => 3,
            Call::Mtp { .. } => 4,
            Call::Draft { .. } => 5,
            Call::Absorb { .. } => 6,
            Call::Stop => 7,
            Call::ParkPool { .. } => 8,
            Call::ParkSave { .. } => 9,
            Call::ParkRestore { .. } => 10,
            Call::CopySlot { .. } => 11,
            Call::CacheSave { .. } => 12,
            Call::CacheRestore { .. } => 13,
            Call::KvOut { .. } => 14,
            Call::KvIn { .. } => 15,
        }
    }
}

struct W(Vec<u8>);
impl W {
    fn u(&mut self, v: usize) {
        self.0.extend_from_slice(&(v as u32).to_le_bytes());
    }
    fn us(&mut self, v: &[usize]) {
        self.u(v.len());
        for &x in v {
            self.u(x);
        }
    }
    fn toks(&mut self, v: &[u32]) {
        self.u(v.len());
        for &x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
}

struct R<'a>(&'a [u8], usize);
impl R<'_> {
    fn u(&mut self) -> Result<usize> {
        if self.1 + 4 > self.0.len() {
            bail!("follow: truncated message");
        }
        let v = u32::from_le_bytes(self.0[self.1..self.1 + 4].try_into()?) as usize;
        self.1 += 4;
        Ok(v)
    }
    fn us(&mut self) -> Result<Vec<usize>> {
        let n = self.u()?;
        (0..n).map(|_| self.u()).collect()
    }
    fn u64(&mut self) -> Result<u64> {
        if self.1 + 8 > self.0.len() {
            bail!("follow: truncated message");
        }
        let v = u64::from_le_bytes(self.0[self.1..self.1 + 8].try_into()?);
        self.1 += 8;
        Ok(v)
    }
    fn toks(&mut self) -> Result<Vec<u32>> {
        let n = self.u()?;
        (0..n).map(|_| self.u().map(|v| v as u32)).collect()
    }
}

impl Call {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::with_capacity(64));
        match self {
            Call::Forward { segs, logit_rows, hist, dec, parents } => {
                w.0.push(1);
                w.u(*hist as usize);
                w.u(*dec);
                w.u(segs.len());
                for (s, t) in segs {
                    w.u(*s);
                    w.toks(t);
                }
                w.us(logit_rows);
                // trees: per segment 0 (none) or 1 + its parents (-1 as u32::MAX)
                w.u(parents.len());
                for p in parents {
                    w.u(p.is_some() as usize);
                    if let Some(p) = p {
                        w.toks(&p.iter().map(|&x| x as u32).collect::<Vec<_>>());
                    }
                }
            }
            Call::CommitTree { slot, start, n, path, taps } => {
                w.0.push(5);
                for v in [slot, start, n] {
                    w.u(*v);
                }
                w.us(path);
                w.u(taps.map_or(u32::MAX as usize, |t| t));
            }
            Call::Commit { slot, start, n, keep } => {
                w.0.push(2);
                for v in [slot, start, n, keep] {
                    w.u(*v);
                }
            }
            Call::Reset { slot } => {
                w.0.push(3);
                w.u(*slot);
            }
            Call::Mtp { rows, draw } => {
                w.0.push(4);
                w.u(rows.len());
                for r in rows {
                    let (kind, idx) = match r.src {
                        MtpSrc::Tap(i) => (0, i),
                        MtpSrc::Out(i) => (1, i),
                    };
                    for v in [r.slot, r.pos, r.token as usize, kind, idx] {
                        w.u(v);
                    }
                }
                w.us(draw);
            }
            Call::Draft { reqs } => {
                w.0.push(6);
                w.u(reqs.len());
                for &(slot, anchor, pos, budget) in reqs {
                    for v in [slot, anchor as usize, pos, budget] {
                        w.u(v);
                    }
                }
            }
            Call::Absorb { items } => {
                w.0.push(7);
                w.u(items.len());
                for &(slot, row, rows, pos) in items {
                    for v in [slot, row, rows, pos] {
                        w.u(v);
                    }
                }
            }
            Call::Stop => w.0.push(0),
            Call::ParkPool { bytes } => {
                w.0.push(8);
                w.0.extend_from_slice(&(*bytes as u64).to_le_bytes());
            }
            Call::ParkSave { slot, a, b, gdn, off } => {
                w.0.push(9);
                for v in [*slot, *a, *b, *gdn as usize] {
                    w.u(v);
                }
                w.0.extend_from_slice(&(*off as u64).to_le_bytes());
            }
            Call::ParkRestore { slot, a, b, gdn, off, len } => {
                w.0.push(10);
                for v in [*slot, *a, *b, *gdn as usize, *len] {
                    w.u(v);
                }
                w.0.extend_from_slice(&(*off as u64).to_le_bytes());
            }
            Call::CopySlot { src, dst, a, b, gdn, len } => {
                w.0.push(11);
                for v in [*src, *dst, *a, *b, *gdn as usize, *len] {
                    w.u(v);
                }
            }
            Call::KvOut { slot, rows, offs } | Call::KvIn { slot, rows, offs } => {
                w.0.push(if matches!(self, Call::KvOut { .. }) { 14 } else { 15 });
                w.u(*slot);
                w.u(*rows);
                w.u(offs.len());
                for &o in offs {
                    w.0.extend_from_slice(&(o as u64).to_le_bytes());
                }
            }
            Call::CacheSave { slot, pos, off } | Call::CacheRestore { slot, pos, off } => {
                w.0.push(if matches!(self, Call::CacheSave { .. }) { 12 } else { 13 });
                w.u(*slot);
                w.u(*pos);
                w.0.extend_from_slice(&(*off as u64).to_le_bytes());
            }
        }
        w.0
    }

    pub fn decode(b: &[u8]) -> Result<Call> {
        let Some(&tag) = b.first() else { bail!("follow: empty message") };
        let mut r = R(b, 1);
        Ok(match tag {
            0 => Call::Stop,
            1 => {
                let hist = r.u()? != 0;
                let dec = r.u()?;
                let n = r.u()?;
                let mut segs = Vec::with_capacity(n);
                for _ in 0..n {
                    let s = r.u()?;
                    segs.push((s, r.toks()?));
                }
                let logit_rows = r.us()?;
                let np = r.u()?;
                let mut parents = Vec::with_capacity(np);
                for _ in 0..np {
                    parents.push(if r.u()? != 0 { Some(r.toks()?.into_iter().map(|x| x as i32).collect()) } else { None });
                }
                Call::Forward { segs, logit_rows, hist, dec, parents }
            }
            2 => Call::Commit { slot: r.u()?, start: r.u()?, n: r.u()?, keep: r.u()? },
            3 => Call::Reset { slot: r.u()? },
            5 => {
                let (slot, start, n, path) = (r.u()?, r.u()?, r.u()?, r.us()?);
                let taps = Some(r.u()?).filter(|&t| t != u32::MAX as usize);
                Call::CommitTree { slot, start, n, path, taps }
            }
            4 => {
                let n = r.u()?;
                let mut rows = Vec::with_capacity(n);
                for _ in 0..n {
                    let (slot, pos, token, kind, idx) = (r.u()?, r.u()?, r.u()? as u32, r.u()?, r.u()?);
                    let src = if kind == 0 { MtpSrc::Tap(idx) } else { MtpSrc::Out(idx) };
                    rows.push(MtpRow { slot, pos, token, src });
                }
                Call::Mtp { rows, draw: r.us()? }
            }
            6 => {
                let n = r.u()?;
                let mut reqs = Vec::with_capacity(n);
                for _ in 0..n {
                    reqs.push((r.u()?, r.u()? as u32, r.u()?, r.u()?));
                }
                Call::Draft { reqs }
            }
            7 => {
                let n = r.u()?;
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    items.push((r.u()?, r.u()?, r.u()?, r.u()?));
                }
                Call::Absorb { items }
            }
            8 => Call::ParkPool { bytes: r.u64()? as usize },
            9 => {
                let (slot, a, b, gdn) = (r.u()?, r.u()?, r.u()?, r.u()? != 0);
                Call::ParkSave { slot, a, b, gdn, off: r.u64()? as usize }
            }
            10 => {
                let (slot, a, b, gdn, len) = (r.u()?, r.u()?, r.u()?, r.u()? != 0, r.u()?);
                Call::ParkRestore { slot, a, b, gdn, off: r.u64()? as usize, len }
            }
            11 => {
                let (src, dst, a, b, gdn, len) = (r.u()?, r.u()?, r.u()?, r.u()?, r.u()? != 0, r.u()?);
                Call::CopySlot { src, dst, a, b, gdn, len }
            }
            12 => Call::CacheSave { slot: r.u()?, pos: r.u()?, off: r.u64()? as usize },
            13 => Call::CacheRestore { slot: r.u()?, pos: r.u()?, off: r.u64()? as usize },
            14 | 15 => {
                let (slot, rows, n) = (r.u()?, r.u()?, r.u()?);
                let offs = (0..n).map(|_| r.u64().map(|o| o as usize)).collect::<Result<Vec<_>>>()?;
                if tag == 14 { Call::KvOut { slot, rows, offs } } else { Call::KvIn { slot, rows, offs } }
            }
            t => bail!("follow: unknown call {t}"),
        })
    }
}
