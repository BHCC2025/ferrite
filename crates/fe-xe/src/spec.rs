//! Host side of the speculation policy (arc-engine `qwen_spec.zig`), ported decision for decision so fe-xe drafts exactly what arc-engine
//! drafts on the same tokens (same windows, same acceptance statistics, comparable speed):
//! - [`draft_n`] / [`GramIndex`]: context copy: the longest earlier occurrence of the context's suffix (min_n..max_n tokens, most recent
//!   among equals; the index looks at most 512 candidates), and the tokens that followed it;
//! - [`Adapt`]: the adaptive MTP chain length (expected accepted tokens per unit of verify cost; f32 arithmetic as in Zig, strict float
//!   mode on both sides, so the choices are identical);
//! - [`hybrid_fires`]: when a context copy replaces the MTP chain (HybridDrafter.observe).
//! None of it changes the output (the target verifies every draft); it changes which tokens are proposed. Checked against the Zig trace
//! a trace written by arc-engine (FE_XE_SPECREF names the file; the test skips without it).

/// A copy match: tokens copied and the matched suffix length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Match {
    pub cnt: usize,
    pub n: usize,
}

/// Copies up to k tokens that followed the most recent earlier occurrence of the longest suffix (max_n down to min_n tokens) of ctx.
pub fn draft_n(ctx: &[u32], k: usize, max_n: usize, min_n: usize, out: &mut [u32]) -> Match {
    let len = ctx.len();
    if len < 2 || k == 0 {
        return Match::default();
    }
    let mut n = max_n.min(len - 1);
    while n >= min_n && n > 0 {
        let suffix = &ctx[len - n..];
        let mut p = len - n; // a match must leave at least one following token
        while p > 0 {
            p -= 1;
            if &ctx[p..p + n] != suffix {
                continue;
            }
            let start = p + n;
            let cnt = k.min(len - start);
            out[..cnt].copy_from_slice(&ctx[start..start + cnt]);
            return Match { cnt, n };
        }
        n -= 1;
    }
    Match::default()
}

/// Index of the context's g-grams by end position (hash chains), so a copy match costs the candidate chain, not the history length.
#[derive(Default)]
pub struct GramIndex {
    g: usize,
    head: Vec<u32>,
    prev: Vec<u32>,
    next: usize,
}

const NONE: u32 = u32::MAX;
const BITS: u32 = 18;
const MAX_CHAIN: usize = 512;

fn gram_hash(t: &[u32]) -> usize {
    let mut h: u32 = 0x9e37_79b9;
    for &x in t {
        h = (h ^ x).wrapping_mul(0x85eb_ca6b);
    }
    ((h ^ (h >> 15)) >> (32 - BITS)) as usize
}

impl GramIndex {
    fn reset(&mut self) {
        *self = GramIndex::default();
    }

    /// Indexes the grams ending at ctx.len - 2 or earlier (the suffix itself is the query).
    fn update(&mut self, ctx: &[u32], g: usize) {
        if self.g != g || ctx.len() < self.next {
            self.reset();
        }
        if self.head.is_empty() {
            self.head = vec![NONE; 1 << BITS];
            self.g = g;
            self.next = g - 1;
        }
        let end = ctx.len() - 1;
        if self.prev.len() < end {
            self.prev.resize(end, 0);
        }
        while self.next < end {
            let e = self.next;
            let h = gram_hash(&ctx[e + 1 - g..e + 1]);
            self.prev[e] = self.head[h];
            self.head[h] = e as u32;
            self.next += 1;
        }
    }

    pub fn draft(&mut self, ctx: &[u32], k: usize, max_n: usize, min_n: usize, out: &mut [u32]) -> Match {
        let len = ctx.len();
        if len < 2 || k == 0 || min_n == 0 || min_n > len - 1 {
            return Match::default();
        }
        self.update(ctx, min_n);
        let (mut best_e, mut best_n) = (0usize, 0usize);
        let mut e = self.head[gram_hash(&ctx[len - min_n..])];
        let mut steps = 0;
        let full = max_n.min(len - 1);
        while e != NONE && steps < MAX_CHAIN {
            let eu = e as usize;
            let cap = full.min(eu + 1);
            let mut l = 0;
            while l < cap && ctx[eu - l] == ctx[len - 1 - l] {
                l += 1;
            }
            if l >= min_n && l > best_n {
                best_n = l;
                best_e = eu;
                if l == full {
                    break;
                }
            }
            e = self.prev[eu];
            steps += 1;
        }
        if best_n == 0 {
            return Match::default();
        }
        let start = best_e + 1;
        let cnt = k.min(len - start);
        out[..cnt].copy_from_slice(&ctx[start..start + cnt]);
        Match { cnt, n: best_n }
    }
}

/// Adaptive MTP chain length (`qwen_spec.Adapt`): per draft depth j an estimate p[j] of P(draft j accepted | the drafts before it were),
/// a verify-cost table cost[R] (window of R rows, in plain steps), a draft step cost; each window drafts the k <= kmax maximising
/// (1 + E[accepted](k)) / (cost[k + 1] + k step + absorb (1 + E)).
#[derive(Clone, Debug)]
pub struct Adapt {
    pub kmax: usize,
    pub p: [f32; 16],
    pub cost: [f32; 17],
    pub step: f32,
    pub absorb: f32,
    pub rate: f32,
    pub prior: f32,
    pub hist: [u64; 16],
}

impl Adapt {
    /// The defaults (the B70 EXL3 SC 4.00 verify-cost curve at 2K); serving sets `step` from SPEC_DSTEP (0.035).
    pub fn new(kmax: usize) -> Adapt {
        Adapt {
            kmax: kmax.min(15),
            p: [0.6; 16],
            cost: [0.0, 1.00, 1.05, 1.08, 1.10, 1.12, 1.15, 1.16, 1.19, 1.38, 1.41, 1.43, 1.45, 1.48, 1.51, 1.53, 1.56],
            step: 0.049,
            absorb: 0.018,
            rate: 0.15,
            prior: 0.6,
            hist: [0; 16],
        }
    }

    /// A window verified kd MTP drafts and accepted a of them.
    pub fn update(&mut self, kd: usize, a: usize) {
        for j in 0..16 {
            if j < a.min(kd) {
                self.p[j] += self.rate * (1.0 - self.p[j]);
            } else if j == a && a < kd {
                self.p[j] += self.rate * (0.0 - self.p[j]);
            } else {
                self.p[j] += 0.02 * (self.prior - self.p[j]);
            }
        }
    }

    pub fn choose(&mut self) -> usize {
        let (mut best_k, mut best) = (1usize, 0f32);
        let (mut e, mut reach) = (0f32, 1f32);
        for k in 1..=self.kmax {
            reach *= self.p[k - 1];
            e += reach;
            let v = (1.0 + e) / (self.cost[k + 1] + k as f32 * self.step + self.absorb * (1.0 + e));
            if v > best {
                best = v;
                best_k = k;
            }
        }
        self.hist[best_k] += 1;
        best_k
    }
}

/// HybridDrafter's decision after the kept rows are absorbed and the MTP head drafted `mtp_first` from the last one: a context copy of
/// `copy_cnt` tokens replaces the chain when it is longer than `min_copy` (0: longer than the chain, `chain_k`) and, with `agree`, starts
/// with the head's own draft.
pub fn hybrid_fires(copy_cnt: usize, copy_first: u32, mtp_first: u32, min_copy: usize, chain_k: usize, agree: bool) -> bool {
    let fire = copy_cnt > if min_copy > 0 { min_copy } else { chain_k };
    fire && (!agree || copy_first == mtp_first)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(x: &mut u64) -> u64 {
        *x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x
    }

    /// The trace of src/xpu/specref_test.zig, line for line.
    fn trace() -> Vec<String> {
        let mut lines = Vec::new();
        let mut a = Adapt::new(7);
        a.step = 0.035;
        let mut x = 12345u64;
        for i in 0..200 {
            let k = a.choose();
            let acc = ((lcg(&mut x) >> 33) % (k as u64 + 1)) as usize;
            a.update(k, acc);
            lines.push(format!("adapt {i} k {k} acc {acc} p0 {:x} p3 {:x}", a.p[0].to_bits(), a.p[3].to_bits()));
        }
        let mut ctx = vec![0u32; 4000];
        let mut gi = GramIndex::default();
        let (mut out, mut out2) = ([0u32; 16], [0u32; 16]);
        for t in 0..4000usize {
            lcg(&mut x);
            ctx[t] = if t > 50 && (t / 50) % 2 == 1 { ctx[t - 37] } else { ((x >> 33) % 23) as u32 };
            if t >= 2 && t % 13 == 0 {
                let m = gi.draft(&ctx[..t + 1], 15, 64, 8, &mut out);
                let m2 = draft_n(&ctx[..t + 1], 15, 64, 8, &mut out2);
                lines.push(format!("gram {} cnt {} n {} first {} | draftN cnt {} n {}", t + 1, m.cnt, m.n, if m.cnt > 0 { out[0] } else { 0 }, m2.cnt, m2.n));
            }
        }
        lines
    }

    #[test]
    fn index_equals_brute_force() {
        for line in trace().iter().filter(|l| l.starts_with("gram")) {
            let f: Vec<&str> = line.split_whitespace().collect();
            assert_eq!((f[3], f[5]), (f[11], f[13]), "{line}");
        }
    }

    #[test]
    fn matches_zig_trace() {
        let p = std::env::var("FE_XE_SPECREF").unwrap_or_default();
        let Ok(want) = std::fs::read_to_string(&p) else {
            eprintln!("SKIP: no {p:?}");
            return;
        };
        let want: Vec<&str> = want.lines().collect();
        let got = trace();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g, w);
        }
    }

    #[test]
    fn hybrid_rule() {
        assert!(hybrid_fires(15, 7, 7, 0, 7, true));
        assert!(!hybrid_fires(15, 7, 8, 0, 7, true)); // vetoed by the head
        assert!(!hybrid_fires(7, 7, 7, 0, 7, true)); // not longer than the chain
        assert!(hybrid_fires(3, 1, 2, 2, 7, false));
    }
}
