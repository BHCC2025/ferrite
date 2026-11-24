//! Park pool for the lanes' prompt caches: when a lane must take another conversation, the one it held is parked here instead of being
//! dropped, and a later request with that conversation's prefix brings it back into a lane (`unpark`) and resumes from it.
//!
//! A parked conversation is its newest prompt-cache snapshot (the sliding K/V window, conv tails and draft rings at its mark, already in
//! pinned host memory) plus what stays in place in a lane: the global layers' K/V and the full draft layer's cache for positions [0, mark).
//! - Host mode (discrete cards: the parked K/V must leave the card): those prefixes are copied into pinned host blocks (`INK_PARK_GB`, capped
//!   so that `INK_PARK_RESERVE_GB` of host memory stays available), and copied back into the lane on unpark. Blocks are reused, not freed.
//! - Resident mode (unified memory, `INK_PARK_RESIDENT`, default on integrated GPUs such as GB10): nothing is copied; the lane's physical
//!   K/V memory is unmapped and kept (`VmBuffer::detach`), the lane gets fresh memory, and unpark maps the kept memory back at the lane's
//!   addresses (`attach`), so the lane's CUDA graphs stay valid.
//! Least recently used conversations leave the pool when it is full. Every byte that comes back is the byte that left, so a resumed
//! request has the bits of the same request prefilled from 0 (the lanes gate checks it).
use crate::engine::Ink;
use crate::pcache::{PCache, ParkedCache};
use crate::spec::Spec;
use anyhow::{ensure, Result};
use fe_cuda::{DevPtr, HostBuffer, VmHandles};
use std::time::Instant;

const BLOCK: usize = 32 << 20;

enum Store {
    /// pinned host blocks, filled in order with the buffers' prefixes (lens per buffer: the 16 global K/V buffers, then the draft's)
    Host { blocks: Vec<usize>, lens: Vec<usize>, n_ink: usize },
    Resident { ink: Vec<VmHandles>, draft: Option<(Vec<VmHandles>, usize)> },
}

struct Entry {
    cache: ParkedCache,
    store: Store,
    bytes: usize,
    used: u64,
}

/// What a park or unpark did.
#[derive(Clone, Debug, Default)]
pub struct Moved {
    pub mark: usize,
    pub bytes: usize,
    pub ms: f64,
}

pub struct Park {
    pub resident: bool,
    budget: usize,
    reserve: usize,
    blocks: Vec<HostBuffer<u8>>,
    free: Vec<usize>,
    entries: Vec<Entry>,
    tick: u64,
    /// conversations shorter than this are not parked (prefilling them again is cheap)
    pub min_tokens: usize,
}

fn mem_available() -> usize {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
        s.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<usize>().ok())
    }).map_or(usize::MAX, |kb| kb * 1024)
}

impl Park {
    /// INK_PARK_GB (budget; default 8 on discrete cards, 16 on unified memory; 0 = off), INK_PARK_RESERVE_GB (host memory that stays
    /// available, host mode; default 16), INK_PARK_RESIDENT (0 / 1; default: integrated GPU), INK_PARK_MIN (tokens; default 512).
    pub fn new(ink: &Ink) -> Option<Park> {
        let integrated = ink.ctx.attribute(18).unwrap_or(0) != 0;
        let resident = std::env::var("INK_PARK_RESIDENT").ok().map_or(integrated, |v| !matches!(v.trim(), "0" | "false" | "off" | "no"));
        let gb: f64 = std::env::var("INK_PARK_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(if resident { 16.0 } else { 8.0 });
        if gb <= 0.0 {
            return None;
        }
        let reserve: f64 = std::env::var("INK_PARK_RESERVE_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(16.0);
        let min_tokens = std::env::var("INK_PARK_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        Some(Park { resident, budget: (gb * 1073741824.0) as usize, reserve: (reserve * 1073741824.0) as usize, blocks: vec![], free: vec![],
            entries: vec![], tick: 0, min_tokens })
    }

    pub fn budget(&self) -> usize {
        self.budget
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Bytes the parked conversations hold (host blocks in use / kept device memory, plus their snapshots).
    pub fn used_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    /// The parked conversation that resumes `prompt` furthest: (entry, resume point).
    pub fn best(&self, prompt: &[u32], need_draft: bool) -> Option<(usize, usize)> {
        self.entries.iter().enumerate().map(|(i, e)| (i, e.cache.resume(prompt, need_draft))).filter(|x| x.1 > 0).max_by_key(|x| x.1)
    }

    fn evict_lru(&mut self) -> bool {
        let Some(i) = (0..self.entries.len()).min_by_key(|&i| self.entries[i].used) else { return false };
        let e = self.entries.remove(i);
        if let Store::Host { blocks, .. } = e.store {
            self.free.extend(blocks);
        }
        true
    }

    /// `n` host blocks: free ones, new ones while the budget and the host's available memory allow, least recently used entries evicted.
    fn take_blocks(&mut self, n: usize, other: usize) -> Option<Vec<usize>> {
        loop {
            let in_use = self.used_bytes();
            let new = n.saturating_sub(self.free.len());
            let grow_ok = new == 0 || (self.blocks.len() + new) * BLOCK <= self.budget && mem_available() > self.reserve + new * BLOCK;
            if in_use + n * BLOCK + other <= self.budget && grow_ok {
                break;
            }
            if !self.evict_lru() {
                return None;
            }
        }
        let mut v = vec![];
        while v.len() < n {
            if let Some(b) = self.free.pop() {
                v.push(b);
            } else {
                match HostBuffer::<u8>::alloc(BLOCK) {
                    Ok(b) => {
                        self.blocks.push(b);
                        v.push(self.blocks.len() - 1);
                    }
                    Err(_) => {
                        self.free.extend(v);
                        return None;
                    }
                }
            }
        }
        Some(v)
    }

    /// Park the current lane's conversation (lane selected in `ink` and in the head; `pc` = that lane's cache). The lane is reset (its K/V
    /// released or replaced). None when there was nothing worth parking or it does not fit (it is then dropped).
    pub fn park(&mut self, ink: &mut Ink, spec: Option<&mut Spec>, pc: &mut PCache) -> Result<Option<Moved>> {
        let t0 = Instant::now();
        let mut spec = spec;
        let Some(cache) = pc.take_for_park() else {
            reset_lane(ink, spec.as_deref_mut())?;
            return Ok(None);
        };
        let m = cache.mark();
        if m < self.min_tokens || ink.lane_len(ink.lane()) < m {
            reset_lane(ink, spec.as_deref_mut())?;
            return Ok(None);
        }
        let with_draft = cache.draft() && spec.is_some();
        let store;
        let bytes;
        if self.resident {
            let need = ink.kv_prefix(m).iter().map(|b| b.1).sum::<usize>() + cache.bytes();
            while self.used_bytes() + need > self.budget {
                if !self.evict_lru() {
                    break;
                }
            }
            if self.used_bytes() + need > self.budget {
                reset_lane(ink, spec.as_deref_mut())?;
                return Ok(None);
            }
            let kvh = ink.kv_detach()?;
            let dh = match (with_draft, spec.as_deref_mut()) {
                (true, Some(sp)) => Some(sp.draft.full_detach(&ink.s)?),
                _ => None,
            };
            bytes = kvh.iter().map(|h| h.bytes()).sum::<usize>() + dh.as_ref().map_or(0, |d| d.0.iter().map(|h| h.bytes()).sum::<usize>()) + cache.bytes();
            store = Store::Resident { ink: kvh, draft: dh };
        } else {
            let mut bufs: Vec<(DevPtr, usize)> = ink.kv_prefix(m);
            let n_ink = bufs.len();
            if with_draft {
                bufs.extend(spec.as_deref().unwrap().draft.full_prefix(m));
            }
            let total: usize = bufs.iter().map(|b| b.1).sum();
            let Some(blocks) = self.take_blocks(total.div_ceil(BLOCK), cache.bytes()) else {
                reset_lane(ink, spec.as_deref_mut())?;
                return Ok(None);
            };
            ink.s.sync()?;
            let (mut bi, mut off) = (0usize, 0usize);
            for &(p, len) in &bufs {
                let mut done = 0;
                while done < len {
                    let c = (len - done).min(BLOCK - off);
                    let dst = &mut self.blocks[blocks[bi]].as_mut_slice()[off..off + c];
                    unsafe { ink.s.memcpy_dtoh(dst, p.byte_add(done))? };
                    done += c;
                    off += c;
                    if off == BLOCK {
                        bi += 1;
                        off = 0;
                    }
                }
            }
            ink.s.sync()?;
            bytes = blocks.len() * BLOCK + cache.bytes();
            store = Store::Host { blocks, lens: bufs.iter().map(|b| b.1).collect(), n_ink };
        }
        reset_lane(ink, spec.as_deref_mut())?;
        self.tick += 1;
        self.entries.push(Entry { cache, store, bytes, used: self.tick });
        Ok(Some(Moved { mark: m, bytes, ms: t0.elapsed().as_secs_f64() * 1e3 }))
    }

    /// Bring parked conversation `i` into the current lane (selected in `ink` and in the head; `pc` = that lane's cache, its conversation
    /// already parked or dropped): the lane is reset, the K/V of [0, mark) put back, the snapshot handed to `pc` (the request then resumes
    /// through `PCache::begin`).
    pub fn unpark(&mut self, i: usize, ink: &mut Ink, spec: Option<&mut Spec>, pc: &mut PCache) -> Result<Moved> {
        let t0 = Instant::now();
        let mut spec = spec;
        let e = self.entries.remove(i);
        let m = e.cache.mark();
        pc.clear();
        reset_lane(ink, spec.as_deref_mut())?;
        let plant = std::env::var("INK_PARK_PLANT").ok().as_deref() == Some("kv");
        match e.store {
            Store::Resident { ink: kvh, draft } => {
                ink.kv_attach(kvh)?;
                // planted fault for the gate's control run (resident mode): the K of one global application comes back zeroed
                if plant {
                    if let Some(&(p, l)) = ink.kv_prefix(m).get(2) {
                        ink.s.memset_d8(p, 0, l)?;
                        ink.s.sync()?;
                    }
                }
                if let Some((h, rows)) = draft {
                    let sp = spec.as_deref_mut().ok_or_else(|| anyhow::anyhow!("unpark: a draft context without the head"))?;
                    sp.draft.full_attach(&ink.s, h, rows)?;
                }
            }
            Store::Host { blocks, lens, n_ink } => {
                ink.ensure_kv(m)?;
                let mut bufs = ink.kv_prefix(m);
                ensure!(bufs.len() == n_ink, "unpark: {} K/V buffers, {} parked", bufs.len(), n_ink);
                if lens.len() > n_ink {
                    let sp = spec.as_deref_mut().ok_or_else(|| anyhow::anyhow!("unpark: a draft context without the head"))?;
                    sp.draft.ensure_capacity(&ink.s, m + crate::draft::BLOCK)?;
                    bufs.extend(sp.draft.full_prefix(m));
                }
                ensure!(bufs.len() == lens.len() && bufs.iter().zip(&lens).all(|(b, &l)| b.1 >= l), "unpark: buffer sizes changed");
                let (mut bi, mut off) = (0usize, 0usize);
                for (k, (&(p, _), &len)) in bufs.iter().zip(&lens).enumerate() {
                    let mut done = 0;
                    while done < len {
                        let c = (len - done).min(BLOCK - off);
                        // planted fault for the gate's control run: the K of one global application is not put back
                        if !(plant && k == 2) {
                            let src = &self.blocks[blocks[bi]].as_slice()[off..off + c];
                            unsafe { ink.s.memcpy_htod(p.byte_add(done), src)? };
                        }
                        done += c;
                        off += c;
                        if off == BLOCK {
                            bi += 1;
                            off = 0;
                        }
                    }
                }
                ink.s.sync()?;
                self.free.extend(blocks);
            }
        }
        pc.give_parked(e.cache);
        Ok(Moved { mark: m, bytes: e.bytes, ms: t0.elapsed().as_secs_f64() * 1e3 })
    }
}

fn reset_lane(ink: &mut Ink, spec: Option<&mut Spec>) -> Result<()> {
    match spec {
        Some(sp) => sp.reset(ink),
        None => ink.reset(),
    }
}
