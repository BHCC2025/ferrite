//! Verify trees from a draft pass (the CUDA drafter's `grow_tree`, TensorFold's best-first greedy grow), and an offline
//! simulator: with every round's draft candidates recorded during a chain run and the run's (lossless) output, how many
//! tokens a tree of a given node budget would have accepted at each round, against the chain the selector walked.
use crate::draft::{RANK, TOPK};

/// Best-first verify tree: each row's top-16 candidates after a predecessor scored as log-softmax((logit + 0.6 edge) / 1.5),
/// edge = pred_cb[pred] . gate[row] . succ_cb[cand]; the anchor and every taken node open their 4 best children on the next
/// row; the open node of least path cost (- sum of log-chances; ties by parent, token, row) is taken next, up to `most`
/// nodes. Returns (tokens, parents), parent -1 = the anchor. tv / ti [rows][16], gate [rows][256] (block rows 1..).
pub fn grow_tree(tv: &[f32], ti: &[u32], gate: &[u16], cb: &(Vec<u16>, Vec<u16>), anchor: u32, rows: usize, most: usize)
    -> (Vec<u32>, Vec<i32>) {
    grow_tree_p(tv, ti, gate, cb, anchor, rows, most, TreeParams::default())
}

/// The tree-growth constants (TensorFold's: branch 4, edge 0.6, scale 1.5; FE_DF_TREE_BRANCH / _EDGE / _SCALE).
#[derive(Clone, Copy, Debug)]
pub struct TreeParams {
    pub branch: usize,
    pub edge: f64,
    pub scale: f64,
}

impl Default for TreeParams {
    fn default() -> Self {
        let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        TreeParams { branch: env("FE_DF_TREE_BRANCH", 4.0) as usize, edge: env("FE_DF_TREE_EDGE", 0.6), scale: env("FE_DF_TREE_SCALE", 1.5) }
    }
}

#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn grow_tree_p(tv: &[f32], ti: &[u32], gate: &[u16], cb: &(Vec<u16>, Vec<u16>), anchor: u32, rows: usize, most: usize,
    tp: TreeParams) -> (Vec<u32>, Vec<i32>) {
    let (BRANCH, EDGE, SCALE) = (tp.branch, tp.edge, tp.scale);
    let bf = |h: u16| f32::from_bits((h as u32) << 16) as f64;
    let chances = |r: usize, pred: u32| -> [f64; TOPK] {
        let g = &gate[r * RANK..(r + 1) * RANK];
        let a = &cb.0[pred as usize * RANK..(pred as usize + 1) * RANK];
        let mut out = [0f64; TOPK];
        let mut most = f64::NEG_INFINITY;
        for (j, o) in out.iter_mut().enumerate() {
            let c = ti[r * TOPK + j] as usize;
            let b = &cb.1[c * RANK..(c + 1) * RANK];
            let edge: f64 = (0..RANK).map(|k| bf(a[k]) * bf(g[k]) * bf(b[k])).sum();
            *o = (tv[r * TOPK + j] as f64 + EDGE * edge) / SCALE;
            most = most.max(*o);
        }
        let lt = most + out.iter().map(|v| (v - most).exp()).sum::<f64>().ln();
        for o in out.iter_mut() {
            *o -= lt;
        }
        out
    };
    #[derive(Clone, Copy)]
    struct Open {
        cost: f64,
        parent: i32,
        token: u32,
        row: usize,
    }
    let (mut tokens, mut parents) = (Vec::new(), Vec::new());
    let mut open: Vec<Open> = Vec::new();
    let (mut pred, mut parent, mut row, mut path) = (anchor, -1i32, 0usize, 0f64);
    loop {
        if row < rows {
            let lc = chances(row, pred);
            let mut order: Vec<usize> = (0..TOPK).collect();
            order.sort_by(|&x, &y| lc[y].partial_cmp(&lc[x]).unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(&y)));
            for &j in order.iter().take(BRANCH) {
                if lc[j].is_finite() {
                    open.push(Open { cost: path - lc[j], parent, token: ti[row * TOPK + j], row });
                }
            }
        }
        if tokens.len() == most || open.is_empty() {
            break;
        }
        let mut best = 0;
        for (i, o) in open.iter().enumerate().skip(1) {
            let b = open[best];
            if o.cost < b.cost || (o.cost == b.cost && (o.parent, o.token, o.row) < (b.parent, b.token, b.row)) {
                best = i;
            }
        }
        let node = open.swap_remove(best);
        tokens.push(node.token);
        parents.push(node.parent);
        pred = node.token;
        parent = (tokens.len() - 1) as i32;
        row = node.row + 1;
        path = node.cost;
    }
    (tokens, parents)
}

/// Accepted depth of a tree against the true continuation `truth` (truth[0] = the token after the anchor): follow the child
/// equal to the next true token while there is one.
pub fn tree_accept(tokens: &[u32], parents: &[i32], truth: &[u32]) -> usize {
    let mut at = -1i32;
    let mut depth = 0;
    while depth < truth.len() {
        let next = (0..tokens.len()).find(|&j| parents[j] == at && tokens[j] == truth[depth]);
        match next {
            Some(j) => {
                at = j as i32;
                depth += 1;
            }
            None => break,
        }
    }
    depth
}

/// One recorded draft round: the anchor's position and token, the chain drafted, and the rows' candidates.
pub struct Round {
    pub pos: usize,
    pub anchor: u32,
    pub chain: Vec<u32>,
    pub tv: Vec<f32>,
    pub ti: Vec<u32>,
    pub gate: Vec<u16>,
}

/// For each budget: (mean accepted of the chain (first `budget` drafts), mean accepted of a `budget`-node tree), over the
/// rounds, given the final sequence `seq` (prompt + output; position p holds seq[p]).
pub fn simulate(rounds: &[Round], seq: &[u32], cb: &(Vec<u16>, Vec<u16>), budgets: &[usize]) -> Vec<(usize, f64, f64)> {
    simulate_p(rounds, seq, cb, budgets, TreeParams::default())
}

pub fn simulate_p(rounds: &[Round], seq: &[u32], cb: &(Vec<u16>, Vec<u16>), budgets: &[usize], tp: TreeParams) -> Vec<(usize, f64, f64)> {
    let mut out = Vec::new();
    for &b in budgets {
        let (mut ca, mut ta, mut n) = (0usize, 0usize, 0usize);
        for r in rounds {
            let rows = r.ti.len() / TOPK;
            let hi = (r.pos + 1 + rows).min(seq.len());
            if r.pos + 1 >= hi {
                continue;
            }
            let truth = &seq[r.pos + 1..hi];
            let k = b.min(r.chain.len());
            ca += r.chain[..k].iter().zip(truth).take_while(|(x, y)| x == y).count();
            let (tk, pa) = grow_tree_p(&r.tv, &r.ti, &r.gate, cb, r.anchor, rows, b, tp);
            ta += tree_accept(&tk, &pa, truth);
            n += 1;
        }
        out.push((b, ca as f64 / n.max(1) as f64, ta as f64 / n.max(1) as f64));
    }
    out
}

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}

impl Rd<'_> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes([self.b[self.o], self.b[self.o + 1], self.b[self.o + 2], self.b[self.o + 3]]);
        self.o += 4;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.b[self.o], self.b[self.o + 1]]);
        self.o += 2;
        v
    }
}

/// Rounds + the final sequence to a file (FE_DF_TREESIM_DUMP), for offline tree studies (`xdf treesim`).
pub fn dump(path: &std::path::Path, rounds: &[Round], seq: &[u32]) -> std::io::Result<()> {
    fn w(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    let mut v: Vec<u8> = Vec::new();
    w(&mut v, seq.len() as u32);
    seq.iter().for_each(|&t| w(&mut v, t));
    w(&mut v, rounds.len() as u32);
    for r in rounds {
        w(&mut v, r.pos as u32);
        w(&mut v, r.anchor);
        w(&mut v, r.chain.len() as u32);
        r.chain.iter().for_each(|&t| w(&mut v, t));
        w(&mut v, r.ti.len() as u32);
        r.ti.iter().zip(&r.tv).for_each(|(&i, &x)| {
            w(&mut v, i);
            w(&mut v, x.to_bits());
        });
        w(&mut v, r.gate.len() as u32);
        r.gate.iter().for_each(|x| v.extend_from_slice(&x.to_le_bytes()));
    }
    std::fs::write(path, v)
}

pub fn load_dump(path: &std::path::Path) -> std::io::Result<(Vec<Round>, Vec<u32>)> {
    let b = std::fs::read(path)?;
    let mut r = Rd { b: &b, o: 0 };
    let n = r.u32() as usize;
    let seq: Vec<u32> = (0..n).map(|_| r.u32()).collect();
    let nr = r.u32() as usize;
    let mut rounds = Vec::with_capacity(nr);
    for _ in 0..nr {
        let pos = r.u32() as usize;
        let anchor = r.u32();
        let nc = r.u32() as usize;
        let chain: Vec<u32> = (0..nc).map(|_| r.u32()).collect();
        let nt = r.u32() as usize;
        let (mut ti, mut tv) = (Vec::with_capacity(nt), Vec::with_capacity(nt));
        for _ in 0..nt {
            ti.push(r.u32());
            tv.push(f32::from_bits(r.u32()));
        }
        let ng = r.u32() as usize;
        let gate: Vec<u16> = (0..ng).map(|_| r.u16()).collect();
        rounds.push(Round { pos, anchor, chain, tv, ti, gate });
    }
    Ok((rounds, seq))
}
