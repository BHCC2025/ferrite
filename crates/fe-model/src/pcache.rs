//! Prompt cache policy (host side; docs/prompt-cache.md): per-slot snapshots at the server's snapshot points (the chat
//! template boundaries), longest-matching-snapshot resume, and the state-block allocator. The device side (saving /
//! restoring a slot's GDN + drafter state into a block, TP2 mirrored) is the engine's (Engine::cache_save /
//! cache_restore); this module only decides which block, where and when.

/// One resumable point of a slot: its state at `pos` (rows [0, pos) in the slot's KV) saved in state block `blk`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snap {
    pub pos: usize,
    pub blk: usize,
}

/// Snapshots a slot keeps: its first one (the end of the shared system prompt) and the latest KEEP_LAST others
/// (turn k + 1 resumes at turn k's generation prompt; one more for a resent turn).
pub const KEEP_LAST: usize = 2;

/// The longest snapshot of a slot whose tokens match `prompt`: the largest `pos` < prompt.len() (at least one
/// token is left to prefill, for the first logits; more than `min_tail` when short last segments are not cut) with
/// slot_tokens[..pos] == prompt[..pos].
pub fn best_snap(slot_tokens: &[u32], snaps: &[Snap], prompt: &[u32], min_tail: usize) -> Option<Snap> {
    let lcp = slot_tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    snaps.iter().copied().filter(|s| s.pos <= lcp && s.pos + min_tail < prompt.len() && s.pos > 0).max_by_key(|s| s.pos)
}

/// After resuming a slot at `pos`: the snapshots past it no longer describe the slot (its rows from `pos` on are
/// about to be rewritten). Returns the blocks to free.
pub fn truncate(snaps: &mut Vec<Snap>, pos: usize) -> Vec<usize> {
    let (keep, drop): (Vec<Snap>, Vec<Snap>) = snaps.iter().partition(|s| s.pos <= pos);
    *snaps = keep;
    drop.into_iter().map(|s| s.blk).collect()
}

/// Record a new snapshot (ascending positions) and apply the keep policy (the first + the latest KEEP_LAST).
/// Returns the blocks to free.
pub fn push(snaps: &mut Vec<Snap>, s: Snap) -> Vec<usize> {
    debug_assert!(snaps.last().is_none_or(|l| l.pos < s.pos));
    snaps.push(s);
    let mut freed = Vec::new();
    while snaps.len() > 1 + KEEP_LAST {
        freed.push(snaps.remove(1).blk);
    }
    freed
}

/// Fixed pool of state blocks (allocated once by the engine; rank 1 holds the same layout, the ids travel in Calls).
pub struct Blocks {
    free: Vec<usize>,
}

impl Blocks {
    pub fn new(n: usize) -> Blocks {
        Blocks { free: (0..n).rev().collect() }
    }
    pub fn alloc(&mut self) -> Option<usize> {
        self.free.pop()
    }
    pub fn release(&mut self, b: impl IntoIterator<Item = usize>) {
        self.free.extend(b);
    }
    pub fn available(&self) -> usize {
        self.free.len()
    }
}

/// KV rows a park block holds (32 KB a row a rank at TP2: 64 MB a block).
pub const KV_BLOCK: usize = crate::model::KV_BLOCK;

/// A parked conversation (its slot was taken) or a shared-prefix entry: token ids [0, len), the KV rows [0, len) in
/// park blocks, and its snapshots (state blocks, positions <= len).
pub struct Entry {
    pub tokens: Vec<u32>,
    pub kv: Vec<usize>,
    pub snaps: Vec<Snap>,
    pub used: u64,
    pub prefix: bool,
}

/// The park pool (FE_PARK_GB): entries, LRU, and the free KV blocks.
pub struct Park {
    pub entries: Vec<Entry>,
    pub kv: Blocks,
}

impl Park {
    pub fn new(kv_blocks: usize) -> Park {
        Park { entries: Vec::new(), kv: Blocks::new(kv_blocks) }
    }
    /// The entry and snapshot with the longest match for `prompt` (ties: the most recently used).
    pub fn best(&self, prompt: &[u32], min_tail: usize) -> Option<(usize, Snap)> {
        let mut best: Option<(usize, Snap)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if let Some(x) = best_snap(&e.tokens, &e.snaps, prompt, min_tail) {
                if best.is_none_or(|(j, y)| x.pos > y.pos || (x.pos == y.pos && e.used > self.entries[j].used)) {
                    best = Some((i, x));
                }
            }
        }
        best
    }
    /// Drop the least recently used entry: its KV blocks go back to the pool, its state blocks are returned.
    pub fn evict_lru(&mut self) -> Option<Vec<usize>> {
        let i = (0..self.entries.len()).min_by_key(|&i| self.entries[i].used)?;
        let e = self.entries.swap_remove(i);
        self.kv.release(e.kv);
        Some(e.snaps.into_iter().map(|x| x.blk).collect())
    }
    pub fn has_prefix(&self, tokens: &[u32]) -> bool {
        self.entries.iter().any(|e| e.prefix && e.tokens == tokens)
    }
    pub fn prefixes(&self) -> usize {
        self.entries.iter().filter(|e| e.prefix).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_snap_is_the_longest_match() {
        let slot: Vec<u32> = (0..100).collect();
        let snaps = [Snap { pos: 10, blk: 0 }, Snap { pos: 40, blk: 1 }, Snap { pos: 90, blk: 2 }];
        // a prompt that diverges at 60: the 40 snapshot
        let mut p: Vec<u32> = (0..120).collect();
        p[60] = 999;
        assert_eq!(best_snap(&slot, &snaps, &p, 0), Some(snaps[1]));
        // identical through 100 and longer: the 90 snapshot
        let p2: Vec<u32> = (0..150).collect();
        assert_eq!(best_snap(&slot, &snaps, &p2, 0), Some(snaps[2]));
        // a snapshot at the prompt's full length is not usable (nothing left to prefill)
        let p3: Vec<u32> = (0..90).collect();
        assert_eq!(best_snap(&slot, &snaps, &p3, 0), Some(snaps[1]));
        // diverges before every snapshot
        let p4: Vec<u32> = vec![5; 50];
        assert_eq!(best_snap(&slot, &snaps, &p4, 0), None);
        // short last segments not cut: a snapshot within min_tail of the prompt end is not a resume point
        let p5: Vec<u32> = (0..100).collect();
        assert_eq!(best_snap(&slot, &snaps, &p5, 16), Some(snaps[1]));
    }

    #[test]
    fn keep_first_and_last_two() {
        let mut s = Vec::new();
        let mut freed = Vec::new();
        for (i, pos) in [100, 200, 300, 400].iter().enumerate() {
            freed.extend(push(&mut s, Snap { pos: *pos, blk: i }));
        }
        assert_eq!(s.iter().map(|x| x.pos).collect::<Vec<_>>(), vec![100, 300, 400]);
        assert_eq!(freed, vec![1]);
        assert_eq!(truncate(&mut s, 300), vec![3]);
        assert_eq!(s.iter().map(|x| x.pos).collect::<Vec<_>>(), vec![100, 300]);
    }

    #[test]
    fn park_best_and_lru() {
        let mut pk = Park::new(4);
        let t: Vec<u32> = (0..100).collect();
        pk.entries.push(Entry { tokens: t[..50].to_vec(), kv: vec![0], snaps: vec![Snap { pos: 50, blk: 7 }], used: 1,
            prefix: true });
        pk.entries.push(Entry { tokens: t[..90].to_vec(), kv: vec![1], snaps: vec![Snap { pos: 50, blk: 8 },
            Snap { pos: 80, blk: 9 }], used: 2, prefix: false });
        let p: Vec<u32> = (0..120).collect();
        assert_eq!(pk.best(&p, 0), Some((1, Snap { pos: 80, blk: 9 })));
        let mut q = p.clone();
        q[60] = 7;
        assert_eq!(pk.best(&q, 0), Some((1, Snap { pos: 50, blk: 8 }))); // tie at 50: the more recent entry
        assert_eq!(pk.evict_lru(), Some(vec![7]));
        assert_eq!(pk.kv.available(), 5);
        assert!(!pk.has_prefix(&t[..50]));
    }

    #[test]
    fn blocks() {
        let mut b = Blocks::new(3);
        let x = b.alloc().unwrap();
        let y = b.alloc().unwrap();
        assert_ne!(x, y);
        b.release([x]);
        assert_eq!(b.available(), 2);
    }
}
