//! Copy rows: verify rows a greedy stream adds from its own history (prompt + reply), for replies that repeat what
//! they have seen (a quoted passage, code carried over from the prompt, a loop). A suffix lookup finds the latest
//! longest earlier occurrence of the history's last tokens and proposes what followed it; the scheduler grafts that
//! onto the drafter's tree past the rows it agrees with, or verifies it alone as a long chain once copies keep
//! landing whole. The idea follows TensorFold's copy rows (native/qwen27_slots.zig, core/lanes/proposer.zig
//! SuffixLookup).
use std::collections::HashMap;

/// tokens a lookup key spans
const KEY: usize = 3;
/// a match counts from this many tokens (the key included)
pub const MIN_MATCH: usize = 5;
/// a match is measured back this far at most
const MAX_MATCH: usize = 64;
/// latest key occurrences a lookup examines
const SCAN: usize = 512;
/// proposals landing nothing in a row before the copier goes silent, and the asks it then skips
const MISSES: usize = 4;
const SILENCE: usize = 16;

/// A stream's suffix index over its history and the adaptive state of its proposals.
pub struct Copier {
    hist: Vec<u32>,
    /// reply tokens appended so far
    seen: usize,
    /// key (the KEY tokens before p) -> positions p, ascending
    index: HashMap<[u32; KEY], Vec<u32>>,
    indexed: usize,
    /// copy rows of the next proposal (grows while copies land whole, shrinks when most miss)
    pub extra: usize,
    lo: usize,
    hi: usize,
    misses: usize,
    silent: usize,
}

impl Copier {
    /// `lo`: the copy rows a proposal starts with and never goes below; `hi`: the most it grows to.
    pub fn new(lo: usize, hi: usize) -> Copier {
        Copier { hist: Vec::new(), seen: 0, index: HashMap::new(), indexed: 0, extra: lo, lo, hi: hi.max(lo), misses: 0,
            silent: 0 }
    }

    /// Brings the history up to prompt + reply (the reply ends with the stream's pending token) and indexes it.
    pub fn sync(&mut self, prompt: &[u32], out: &[u32]) {
        if self.hist.is_empty() && self.seen == 0 {
            self.hist.extend_from_slice(prompt);
        }
        if out.len() > self.seen {
            self.hist.extend_from_slice(&out[self.seen..]);
            self.seen = out.len();
        }
        while self.indexed < self.hist.len() {
            self.indexed += 1;
            let p = self.indexed;
            if p >= KEY {
                let k: [u32; KEY] = self.hist[p - KEY..p].try_into().unwrap();
                self.index.entry(k).or_default().push(p as u32);
            }
        }
    }

    /// Up to `max` tokens that followed the latest longest earlier match (>= MIN_MATCH tokens) of the history's end,
    /// with the match length; empty while silent. A continuation that reaches the history's end goes on with the
    /// copy itself (a loop of period < max continues).
    pub fn propose(&mut self, max: usize) -> (Vec<u32>, usize) {
        let h = &self.hist;
        let n = h.len();
        if max == 0 || n <= KEY {
            return (Vec::new(), 0);
        }
        if self.silent > 0 {
            self.silent -= 1;
            return (Vec::new(), 0);
        }
        let k: [u32; KEY] = h[n - KEY..].try_into().unwrap();
        let Some(pos) = self.index.get(&k) else { return (Vec::new(), 0) };
        let (mut best, mut best_len) = (0usize, 0usize);
        for &p in pos.iter().rev().take(SCAN) {
            let p = p as usize;
            if p >= n {
                continue;
            }
            let mut l = 0;
            while l < MAX_MATCH && l < p && h[p - 1 - l] == h[n - 1 - l] {
                l += 1;
            }
            if l > best_len {
                best = p;
                best_len = l;
                if l == MAX_MATCH {
                    break;
                }
            }
        }
        if best_len < MIN_MATCH {
            return (Vec::new(), best_len);
        }
        let mut out = Vec::with_capacity(max);
        for j in 0..max {
            let i = best + j;
            out.push(if i < n { h[i] } else { out[i - n] });
        }
        (out, best_len)
    }

    /// A proposal's outcome: `proposed` copy rows verified, `landed` of them kept.
    pub fn observe(&mut self, proposed: usize, landed: usize) {
        if proposed == 0 {
            return;
        }
        if landed == proposed {
            self.extra = self.hi.min(self.extra * 2);
        } else if 2 * landed < proposed {
            self.extra = self.lo.max(self.extra / 2);
        }
        if landed == 0 {
            self.misses += 1;
            if self.misses >= MISSES {
                self.misses = 0;
                self.silent = SILENCE;
                self.extra = self.lo;
            }
        } else {
            self.misses = 0;
        }
    }
}

/// Grafts `copy` (the tokens after the segment's root) onto a verify tree of `toks` (row 0 = the root) with row
/// parents `par` (-1 = the root): follows the copy down the rows that already hold it, then appends up to `room` of
/// the rest as a chain from the deepest of them. Returns (rows followed, rows added).
pub fn graft(toks: &mut Vec<u32>, par: &mut Vec<i32>, copy: &[u32], room: usize) -> (usize, usize) {
    let mut r = 0usize;
    let mut i = 0;
    while i < copy.len() {
        match (r + 1..toks.len()).find(|&c| par[c] == r as i32 && toks[c] == copy[i]) {
            Some(c) => {
                r = c;
                i += 1;
            }
            None => break,
        }
    }
    let add = (copy.len() - i).min(room);
    for &t in &copy[i..i + add] {
        par.push(r as i32);
        toks.push(t);
        r = toks.len() - 1;
    }
    (i, add)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_what_followed_the_latest_longest_match() {
        let mut c = Copier::new(4, 8);
        // ... 1 2 3 4 5 [9] ... 7 2 3 4 5 [6] ... 1 2 3 4 5 -> 9 follows the longer match (1 2 3 4 5)
        c.sync(&[1, 2, 3, 4, 5, 9, 8, 7, 2, 3, 4, 5, 6, 0], &[1, 2, 3, 4, 5]);
        assert_eq!(c.propose(3), (vec![9, 8, 7], 5));
        // shorter than MIN_MATCH: nothing
        let mut c = Copier::new(4, 8);
        c.sync(&[2, 3, 4, 9, 0], &[7, 2, 3, 4]);
        assert_eq!(c.propose(3).0, Vec::<u32>::new());
    }

    #[test]
    fn a_loop_continues_past_the_history_end() {
        let mut c = Copier::new(4, 8);
        c.sync(&[0], &[5, 6, 7, 8, 5, 6, 7, 8, 5, 6]);
        assert_eq!(c.propose(7).0, vec![7, 8, 5, 6, 7, 8, 5]);
    }

    #[test]
    fn reply_tokens_are_indexed_as_they_arrive() {
        let mut c = Copier::new(4, 8);
        c.sync(&[1, 2, 3, 4, 5, 6], &[]);
        assert!(c.propose(2).0.is_empty());
        c.sync(&[1, 2, 3, 4, 5, 6], &[1, 2, 3, 4, 5]);
        assert_eq!(c.propose(2).0, vec![6, 1]);
    }

    #[test]
    fn extra_adapts_and_misses_silence() {
        let mut c = Copier::new(4, 16);
        c.observe(4, 4);
        assert_eq!(c.extra, 8);
        c.observe(8, 8);
        c.observe(16, 16);
        assert_eq!(c.extra, 16);
        c.observe(16, 3);
        assert_eq!(c.extra, 8);
        for _ in 0..MISSES {
            c.observe(8, 0);
        }
        assert_eq!(c.extra, 4);
        c.sync(&[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5], &[]);
        for _ in 0..SILENCE {
            assert!(c.propose(2).0.is_empty());
        }
        assert_eq!(c.propose(2).0, vec![6, 1]);
    }

    #[test]
    fn graft_follows_the_tree_then_chains() {
        // root 10; nodes 11 (row 1, parent 0), 12 (row 2, parent 1), 13 (row 3, parent 0)
        let mut toks = vec![10, 11, 12, 13];
        let mut par = vec![-1, 0, 1, 0];
        assert_eq!(graft(&mut toks, &mut par, &[11, 12, 14, 15, 16], 2), (2, 2));
        assert_eq!(toks, vec![10, 11, 12, 13, 14, 15]);
        assert_eq!(par, vec![-1, 0, 1, 0, 2, 4]);
        // a copy the tree does not start with branches from the root
        let mut toks = vec![10, 11];
        let mut par = vec![-1, 0];
        assert_eq!(graft(&mut toks, &mut par, &[20, 21], 8), (0, 2));
        assert_eq!(par, vec![-1, 0, 0, 2]);
    }
}
