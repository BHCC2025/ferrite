//! CPU f32 reference forward of DeepSeek-V4.1-Flash from the GGUF (the M2 oracle), layer-major: every row of a call
//! goes through layer l before layer l + 1, weights are dequantised (refq) one tensor at a time and dropped, and each
//! routed expert is dequantised once into a reused buffer and applied to all rows routed to it. Activations are f32
//! throughout (no bf16 rounding); the model's own KV formats (FP8 window rows, FP4 compressed / indexer entries) are
//! applied, as the reference does. Semantics are those of the ds-model kernels (k-ds): hc mixes with the previous
//! sublayer's `pre`, MQA latent attention with sinks, compressor, indexer, candidate pool, sqrt-softplus router, Engram.
//! A call can start mid-sequence: the per-layer caches persist (`Oracle::forward` appends rows).
use crate::config::DscConfig;
use crate::engram::Engram;
use anyhow::{anyhow, Result};
use fe_ggq::refq;
use fe_gguf::{GgmlType, Gguf};
use rayon::prelude::*;

// ---------------------------------------------------------------- small math

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            s[i] += x[i] * y[i];
        }
    }
    let mut t = ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
    for (x, y) in ra.iter().zip(rb) {
        t += x * y;
    }
    t
}

/// y[n][o] = sum_k x[n][k] w[o][k]; w [o][k] row-major f32.
pub fn matmul(w: &[f32], o: usize, k: usize, x: &[f32], n: usize, par: bool) -> Vec<f32> {
    debug_assert!(w.len() == o * k && x.len() == n * k);
    let mut yt = vec![0f32; o * n];
    let f = |(oi, row): (usize, &mut [f32])| {
        let wr = &w[oi * k..(oi + 1) * k];
        for i in 0..n {
            row[i] = dot(wr, &x[i * k..(i + 1) * k]);
        }
    };
    if par {
        yt.par_chunks_mut(n).enumerate().for_each(f);
    } else {
        yt.chunks_mut(n).enumerate().for_each(f);
    }
    let mut y = vec![0f32; n * o];
    for oi in 0..o {
        for i in 0..n {
            y[i * o + oi] = yt[oi * n + i];
        }
    }
    y
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| w * (v * inv)).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

// FP8 E4M3 / FP4 E2M1 grids (exactly k-ds q.rs)
fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}
fn pow2_ceil(v: f32) -> f32 {
    let b = v.to_bits();
    pow2(((b >> 23) & 0xff) as i32 - 127 + ((b & 0x7f_ffff) != 0) as i32)
}
fn rni(x: f32) -> f32 {
    x.round_ties_even()
}
fn e4m3(x: f32) -> f32 {
    let a = x.abs();
    let step = if a < 0.015625 { 0.001953125 } else { pow2(((a.to_bits() >> 23) & 0xff) as i32 - 127 - 3) };
    rni(x / step) * step
}
fn e2m1(v: f32) -> f32 {
    let a = v.abs();
    let g = if a <= 0.25 { 0.0 } else if a < 0.75 { 0.5 } else if a <= 1.25 { 1.0 } else if a < 1.75 { 1.5 }
        else if a <= 2.5 { 2.0 } else if a < 3.5 { 3.0 } else if a <= 5.0 { 4.0 } else { 6.0 };
    if v < 0.0 { -g } else { g }
}
/// FP8 quant-dequant in blocks of 32 (power-of-two scale, amax floored at 1e-4).
fn fp8_blocks(x: &mut [f32]) {
    for b in x.chunks_mut(32) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = pow2_ceil(amax.max(1e-4) / 448.0);
        for v in b {
            *v = e4m3((*v / s).clamp(-448.0, 448.0)) * s;
        }
    }
}
/// FP4 quant-dequant in blocks of `blk`; `e4m3_scale` selects the compressed-KV scale (E4M3) vs power of two.
fn fp4_blocks(x: &mut [f32], blk: usize, e4m3_scale: bool) {
    for b in x.chunks_mut(blk) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = if e4m3_scale { e4m3(amax.max(6.0 * 0.001953125) / 6.0) } else { pow2_ceil(amax.max(6.0 * 1.1754944e-38) / 6.0) };
        for v in b {
            *v = e2m1((*v / s).clamp(-6.0, 6.0)) * s;
        }
    }
}

/// cos / sin table [pos][32][2] for the last 64 dims; YaRN ramp when `orig_len` > 0 (no attention mscale).
pub fn rope_table(c: &DscConfig, base: f64, orig_len: usize, max_pos: usize) -> Vec<f32> {
    let dim = c.n_rot;
    let half = dim / 2;
    let mut freqs: Vec<f32> = (0..half).map(|i| 1.0f32 / (base as f32).powf((2 * i) as f32 / dim as f32)).collect();
    if orig_len > 0 {
        let corr = |rot: f64| dim as f64 * (orig_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
        let low = corr(c.yarn_beta_fast as f64).floor().max(0.0);
        let high = corr(c.yarn_beta_slow as f64).ceil().min(dim as f64 - 1.0);
        let span = (high - low).max(1e-3) as f32;
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - low as f32) / span).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / c.yarn_factor * (1.0 - smooth) + *f * smooth;
        }
    }
    let mut t = vec![0f32; max_pos * dim];
    for p in 0..max_pos {
        for (i, &f) in freqs.iter().enumerate() {
            let a = (p as f32 * f) as f64;
            t[p * dim + 2 * i] = a.cos() as f32;
            t[p * dim + 2 * i + 1] = a.sin() as f32;
        }
    }
    t
}

/// Rotate the adjacent pairs of the 64 values of `x` by the table row (inverse: conjugate).
fn rope64(x: &mut [f32], tab: &[f32], pos: usize, inverse: bool) {
    for i in 0..32 {
        let (cs, mut sn) = (tab[pos * 64 + 2 * i], tab[pos * 64 + 2 * i + 1]);
        if inverse {
            sn = -sn;
        }
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        x[2 * i] = a * cs - b * sn;
        x[2 * i + 1] = a * sn + b * cs;
    }
}

// ---------------------------------------------------------------- weights

/// Dequantise a whole tensor (or the `[row0, row0 + nrows)` rows) to f32, in parallel over row groups.
pub fn load(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name).ok_or_else(|| anyhow!("tensor {name} missing"))?;
    dequant_bytes(t.ty, g.data(t), t.n_elements() as usize, t.ne0())
}

pub fn dequant_bytes(ty: GgmlType, src: &[u8], n: usize, ne0: usize) -> Result<Vec<f32>> {
    if !refq::supported(ty) {
        return Err(anyhow!("refq has no dequantiser for {}", ty.name()));
    }
    let rb = ty.row_bytes(ne0);
    let rows = n / ne0;
    let mut y = vec![0f32; n];
    let per = rows.div_ceil(rayon::current_num_threads() * 4).max(1);
    y.par_chunks_mut(per * ne0).zip(src.par_chunks(per * rb)).for_each(|(yy, ss)| refq::dequant(ty, ss, yy));
    Ok(y)
}

/// y = x . w^T as the engine computes it for a quantised weight: each activation row is q8_1-quantised in blocks of 32
/// (fe-ggq `quantize_q8_1`: d = amax/127, q = round(x/d), s = f16-rounded float sum), the product is taken with the
/// dequantised weight and the dequantised activation q*d, and for Q4_K / Q5_K the min term uses the float sum s instead of
/// d*sum(q) (`refq::dot_q8`). `raw` = the weight rows as stored (o rows of k).
pub fn mm_q8(ty: GgmlType, raw: &[u8], w: &[f32], o: usize, k: usize, x: &[f32], n: usize, par: bool) -> Vec<f32> {
    let nb = k / 32;
    let mut xq = vec![0f32; n * k];
    let mut corr = vec![0f32; n * nb]; // d * sum(q) - s per block (Q4_K / Q5_K min term)
    for i in 0..n {
        let (q, d, s) = refq::quantize_q8_1(&x[i * k..(i + 1) * k]);
        for b in 0..nb {
            let mut sq = 0i32;
            for j in 0..32 {
                xq[i * k + b * 32 + j] = q[b * 32 + j] as f32 * d[b];
                sq += q[b * 32 + j] as i32;
            }
            corr[i * nb + b] = d[b] * sq as f32 - s[b];
        }
    }
    let mut y = matmul(w, o, k, &xq, n, par);
    if matches!(ty, GgmlType::Q4K | GgmlType::Q5K) {
        let rb = ty.row_bytes(k);
        let (_, by) = ty.block();
        let fix = |(oi, yo): (usize, &mut [f32])| {
            let mut mins = vec![0f32; nb];
            for blk in 0..k / 256 {
                let m = refq::kq_mins(ty, &raw[oi * rb + blk * by..oi * rb + (blk + 1) * by]).unwrap();
                mins[blk * 8..blk * 8 + 8].copy_from_slice(&m);
            }
            for i in 0..n {
                let mut a = 0f32;
                for b in 0..nb {
                    a += mins[b] * corr[i * nb + b];
                }
                yo[i] += a;
            }
        };
        let mut yt = vec![0f32; o * n];
        for oi in 0..o { for i in 0..n { yt[oi * n + i] = y[i * o + oi]; } }
        if par { yt.par_chunks_mut(n).enumerate().for_each(fix); } else { yt.chunks_mut(n).enumerate().for_each(fix); }
        for oi in 0..o { for i in 0..n { y[i * o + oi] = yt[oi * n + i]; } }
    }
    y
}

fn is_q8_weight(ty: GgmlType) -> bool {
    !matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

// ---------------------------------------------------------------- state

#[derive(Default)]
struct LayerState {
    /// per position: FP8-gridded, roped, normed latent (512)
    win: Vec<Vec<f32>>,
    /// compressor inputs per position: (kvc, sc)
    raw: Vec<(Vec<f32>, Vec<f32>)>,
    /// compressed plane entries (FP4, roped) and indexer keys
    plane: Vec<Vec<f32>>,
    ikeys: Vec<Vec<f32>>,
}

/// One routed-expert use, for the routing trace.
#[derive(Clone, Debug)]
pub struct Route {
    pub layer: u8,
    pub row: u32,
    pub ids: [u16; 6],
    pub w: [f32; 6],
}

pub struct Oracle<'a> {
    pub g: &'a Gguf,
    pub c: &'a DscConfig,
    eng: Engram,
    tab_plain: Vec<f32>,
    tab_comp: Vec<f32>,
    pub ids: Vec<u32>,
    layers: Vec<LayerState>,
    /// FP8 / FP4 KV grids (default on, ORACLE_NOQUANT=1 turns them off)
    pub quant_kv: bool,
    /// q8_1 activation mode (ORACLE_Q8=1): activations are quantised exactly where the engine feeds a quantised-weight
    /// GEMV / mmq (see `mm_q8`), so the oracle models the engine numerics; off = f32 activations (the absolute oracle).
    pub q8: bool,
    /// per layer, the last row's streams after the layer (f32 [4*d]), when set to Some(vec![]) (the engine gate's layer dumps)
    pub dump_h: Option<Vec<Vec<f32>>>,
    /// hc-rule ablation (ORACLE_HC): own | first | ffnprev | headmean | head0 | combT; empty = the chosen rule
    pub hcv: String,
    pub trace: Vec<Route>,
    pub verbose: bool,
    pool: rayon::ThreadPool,
}

struct Mix {
    pre: [f32; 4],
    post: [f32; 4],
    comb: [f32; 16],
}

impl<'a> Oracle<'a> {
    pub fn new(g: &'a Gguf, c: &'a DscConfig, max_pos: usize) -> Result<Oracle<'a>> {
        let nthreads: usize = std::env::var("ORACLE_EXPERT_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        Ok(Oracle {
            g, c,
            eng: Engram::new(g, c)?,
            tab_plain: rope_table(c, c.rope_base as f64, 0, max_pos),
            tab_comp: rope_table(c, c.rope_compress_base as f64, c.yarn_orig_ctx, max_pos),
            ids: vec![],
            layers: (0..c.n_layer).map(|_| LayerState::default()).collect(),
            hcv: std::env::var("ORACLE_HC").unwrap_or_default(),
            dump_h: None,
            q8: std::env::var("ORACLE_Q8").is_ok_and(|v| v == "1"),
            quant_kv: std::env::var("ORACLE_NOQUANT").map_or(true, |v| v != "1"),
            trace: vec![],
            verbose: std::env::var("ORACLE_VERBOSE").is_ok(),
            pool: rayon::ThreadPoolBuilder::new().num_threads(nthreads).build()?,
        })
    }

    fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        load(self.g, name)
    }

    fn hc_mix(&self, h: &[f32], fnw: &[f32], scale: &[f32], base: &[f32]) -> Mix {
        let c = self.c;
        let d4 = c.hc_dim();
        let ss: f32 = h.iter().map(|v| v * v).sum::<f32>();
        let rs = 1.0 / (ss / d4 as f32 + c.eps).sqrt();
        let m: Vec<f32> = (0..24).map(|k| dot(h, &fnw[k * d4..(k + 1) * d4]) * rs).collect();
        let mut pre = [0f32; 4];
        let mut post = [0f32; 4];
        for k in 0..4 {
            pre[k] = sigmoid(m[k] * scale[0] + base[k]) + c.hc_eps;
            post[k] = 2.0 * sigmoid(m[4 + k] * scale[1] + base[4 + k]);
        }
        let mut cb = [0f32; 16];
        for i in 0..4 {
            let v: Vec<f32> = (0..4).map(|j| m[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j]).collect();
            let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = v.iter().map(|x| (x - mx).exp()).collect();
            let s: f32 = e.iter().sum();
            for j in 0..4 {
                cb[i * 4 + j] = e[j] / s + c.hc_eps;
            }
        }
        let col = |c4: &mut [f32; 16]| {
            for j in 0..4 {
                let s = c4[j] + c4[4 + j] + c4[8 + j] + c4[12 + j] + c.hc_eps;
                for i in 0..4 {
                    c4[i * 4 + j] /= s;
                }
            }
        };
        let row = |c4: &mut [f32; 16]| {
            for i in 0..4 {
                let s = c4[i * 4] + c4[i * 4 + 1] + c4[i * 4 + 2] + c4[i * 4 + 3] + c.hc_eps;
                for j in 0..4 {
                    c4[i * 4 + j] /= s;
                }
            }
        };
        col(&mut cb);
        for _ in 1..c.hc_iters {
            row(&mut cb);
            col(&mut cb);
        }
        Mix { pre, post, comb: cb }
    }

    /// Matmul against the (already dequantised) weight `w` of tensor `name`; q8 mode applies [`mm_q8`] for quantised types.
    fn mq(&self, name: &str, w: &[f32], o: usize, k: usize, x: &[f32], n: usize) -> Vec<f32> {
        if self.q8 {
            if let Some(t) = self.g.tensor(name) {
                if is_q8_weight(t.ty) {
                    return mm_q8(t.ty, self.g.data(t), w, o, k, x, n, true);
                }
            }
        }
        matmul(w, o, k, x, n, true)
    }

    fn collapse(&self, h: &[f32], pre: &[f32; 4]) -> Vec<f32> {
        let d = self.c.n_embd;
        (0..d).map(|i| pre[0] * h[i] + pre[1] * h[d + i] + pre[2] * h[2 * d + i] + pre[3] * h[3 * d + i]).collect()
    }

    fn post(&self, h: &[f32], y: &[f32], m: &Mix) -> Vec<f32> {
        let d = self.c.n_embd;
        let mut o = vec![0f32; 4 * d];
        for j in 0..4 {
            for i in 0..d {
                let mut s = m.post[j] * y[i];
                for k in 0..4 {
                    s += if self.hcv == "combT" { m.comb[j * 4 + k] } else { m.comb[k * 4 + j] } * h[k * d + i];
                }
                o[j * d + i] = s;
            }
        }
        o
    }

    /// Append `tokens` to the sequence and return the logits of the rows in `logit_rows` (indices into `tokens`).
    pub fn forward(&mut self, tokens: &[u32], logit_rows: &[usize]) -> Result<Vec<Vec<f32>>> {
        let c = self.c;
        let d = c.n_embd;
        let n = tokens.len();
        let start = self.ids.len();
        self.ids.extend_from_slice(tokens);
        // embedding: 4 identical streams
        let emb = self.g.tensor("token_embd.weight").ok_or_else(|| anyhow!("token_embd"))?;
        let rb = emb.row_bytes();
        let ed = self.g.data(emb);
        let mut h = vec![0f32; n * 4 * d];
        for (i, &t) in tokens.iter().enumerate() {
            let mut row = vec![0f32; d];
            refq::dequant(emb.ty, &ed[t as usize * rb..(t as usize + 1) * rb], &mut row);
            for s in 0..4 {
                h[(i * 4 + s) * d..(i * 4 + s + 1) * d].copy_from_slice(&row);
            }
        }
        let mut pre: Vec<[f32; 4]> = vec![[1.0, 0.0, 0.0, 0.0]; n];
        // selection state shared by the layers of a call: per row, the latest index source's top-k and the candidate mask
        let mut sel: Vec<Vec<i32>> = vec![vec![]; n];
        let mut cand: Vec<Vec<bool>> = vec![vec![]; n];
        for l in 0..c.n_layer {
            let t0 = std::time::Instant::now();
            self.layer(l, start, n, &mut h, &mut pre, &mut sel, &mut cand)?;
            if let Some(dv) = self.dump_h.as_mut() {
                dv.push(h[(n - 1) * 4 * d..n * 4 * d].to_vec());
            }
            if self.verbose {
                eprintln!("oracle layer {l} {:.2}s", t0.elapsed().as_secs_f32());
            }
        }
        if logit_rows.is_empty() {
            return Ok(vec![]);
        }
        let hp = |i: usize| -> [f32; 4] {
            match self.hcv.as_str() {
                "headmean" => [0.25; 4],
                "head0" => [1.0, 0.0, 0.0, 0.0],
                _ => pre[i],
            }
        };
        // head: collapse with the last mix's pre, norm, output
        let onorm = self.f32s("output_norm.weight")?;
        let xs: Vec<f32> = logit_rows.iter().flat_map(|&i| rmsnorm(&self.collapse(&h[i * 4 * d..(i + 1) * 4 * d], &hp(i)), &onorm, c.eps)).collect();
        let ow = self.f32s("output.weight")?;
        let lg = self.mq("output.weight", &ow, c.n_vocab, d, &xs, logit_rows.len());
        Ok(lg.chunks(c.n_vocab).map(|v| v.to_vec()).collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(&mut self, l: usize, start: usize, n: usize, h: &mut [f32], pre: &mut [[f32; 4]], sel: &mut [Vec<i32>], cand: &mut [Vec<bool>]) -> Result<()> {
        let c = self.c;
        let d = c.n_embd;
        let role = c.role(l);
        let p = |s: &str| format!("blk.{l}.{s}.weight");
        // ---- Engram
        if let Some(li) = c.engram_layers.iter().position(|&x| x == l) {
            let kvw = self.f32s(&p("engram_kv"))?;
            let qk = self.f32s(&p("engram_gate"))?;
            let w = (c.hc + 1) * d;
            let vecs: Vec<f32> = (0..n).map(|i| self.eng.vector(&self.ids, start + i, li)).collect::<Result<Vec<_>>>()?.concat();
            let kv = matmul(&kvw, w, c.engram_width(), &vecs, n, true);
            for i in 0..n {
                let kvr = &kv[i * w..(i + 1) * w];
                for s in 0..c.hc {
                    let hr = &mut h[(i * 4 + s) * d..(i * 4 + s + 1) * d];
                    let key = &kvr[s * d..(s + 1) * d];
                    let q = &qk[s * d..(s + 1) * d];
                    let sh: f32 = hr.iter().map(|v| v * v).sum();
                    let sk: f32 = key.iter().map(|v| v * v).sum();
                    let sd: f32 = (0..d).map(|j| hr[j] * q[j] * key[j]).sum();
                    let df = d as f32;
                    let rstd = 1.0 / (sh / df + c.eps).sqrt() / (sk / df + c.eps).sqrt();
                    let dt = sd * rstd / df.sqrt();
                    let mag = dt.abs().max(1e-6).sqrt();
                    let sg = if dt < 0.0 { -mag } else { mag };
                    let gate = 1.0 / (1.0 + (-sg).exp());
                    let val = &kvr[c.hc * d..(c.hc + 1) * d];
                    for j in 0..d {
                        hr[j] += gate * val[j];
                    }
                }
            }
        }
        // ---- attention sublayer
        let fnw = self.f32s(&p("hc_attn_fn"))?;
        let (sc_, ba_) = (self.f32s(&p("hc_attn_scale"))?, self.f32s(&p("hc_attn_base"))?);
        let attn_norm = self.f32s(&p("attn_norm"))?;
        let mixes: Vec<Mix> = (0..n).map(|i| self.hc_mix(&h[i * 4 * d..(i + 1) * 4 * d], &fnw, &sc_, &ba_)).collect();
        let cur = std::env::var("ORACLE_PRE").is_ok_and(|v| v == "cur") || self.hcv == "own"; // experiment: collapse with the sublayer's own mix
        let first_own = self.hcv == "first" && l == 0;
        let pre_in: Vec<[f32; 4]> = pre.to_vec();
        let xs: Vec<f32> = (0..n).flat_map(|i| rmsnorm(&self.collapse(&h[i * 4 * d..(i + 1) * 4 * d], if cur || first_own { &mixes[i].pre } else { &pre[i] }), &attn_norm, c.eps)).collect();
        let tab = if role.ratio > 0 { &self.tab_comp } else { &self.tab_plain };
        let hd = c.head_dim;
        let wq_a = self.f32s(&p("attn_q_a"))?;
        let qa_norm = self.f32s(&p("attn_q_a_norm"))?;
        let qr_raw = self.mq(&p("attn_q_a"), &wq_a, c.q_lora, d, &xs, n);
        drop(wq_a);
        let qr: Vec<f32> = qr_raw.chunks(c.q_lora).flat_map(|r| rmsnorm(r, &qa_norm, c.eps)).collect();
        let wq_b = self.f32s(&p("attn_q_b"))?;
        let mut q = self.mq(&p("attn_q_b"), &wq_b, c.q_dim(), c.q_lora, &qr, n);
        drop(wq_b);
        for i in 0..n {
            for hh in 0..c.n_head {
                let o = i * c.q_dim() + hh * hd + hd - 64;
                rope64(&mut q[o..o + 64], tab, start + i, false);
            }
        }
        let wkv = self.f32s(&p("attn_kv"))?;
        let kv_norm = self.f32s(&p("attn_kv_a_norm"))?;
        let kvr = self.mq(&p("attn_kv"), &wkv, hd, d, &xs, n);
        drop(wkv);
        for i in 0..n {
            let mut k = rmsnorm(&kvr[i * hd..(i + 1) * hd], &kv_norm, c.eps);
            rope64(&mut k[hd - 64..], tab, start + i, false);
            if self.quant_kv {
                fp8_blocks(&mut k);
            }
            self.layers[l].win.push(k);
        }
        // compressor + indexer keys
        if role.kv_source {
            let wkv_c = load(self.g, &p("attn_compressor_kv"))?;
            let kvc = matmul(&wkv_c, hd, d, &xs, n, true);
            let scs = if role.ratio == 2 { matmul(&load(self.g, &p("attn_compressor_gate"))?, hd, d, &xs, n, true) } else { vec![] };
            let cnorm = self.f32s(&p("attn_compressor_norm"))?;
            let ix = if c.index_sources.contains(&l) && self.g.tensor(&format!("blk.{l}.indexer.attn_k.weight")).is_some() {
                Some((self.f32s(&p("indexer.attn_k"))?, self.f32s(&p("indexer.k_norm"))?))
            } else { None };
            for i in 0..n {
                let pos = start + i;
                let sc = if role.ratio == 2 { scs[i * hd..(i + 1) * hd].to_vec() } else { vec![] };
                self.layers[l].raw.push((kvc[i * hd..(i + 1) * hd].to_vec(), sc));
                if (pos + 1) % role.ratio as usize != 0 {
                    continue;
                }
                let e = pos / role.ratio as usize;
                let lat: Vec<f32> = if role.ratio == 1 {
                    self.layers[l].raw[pos].0.clone()
                } else {
                    let (a, b) = (&self.layers[l].raw[pos - 1], &self.layers[l].raw[pos]);
                    (0..hd).map(|j| {
                        let m = a.1[j].max(b.1[j]);
                        let (ea, eb) = ((a.1[j] - m).exp(), (b.1[j] - m).exp());
                        let s = ea + eb;
                        a.0[j] * (ea / s) + b.0[j] * (eb / s)
                    }).collect()
                };
                let latn = rmsnorm(&lat, &cnorm, c.eps);
                let mut pe = latn.clone();
                rope64(&mut pe[hd - 64..], tab, e * role.ratio as usize, false);
                if self.quant_kv {
                    fp4_blocks(&mut pe, 16, true);
                }
                debug_assert_eq!(self.layers[l].plane.len(), e);
                self.layers[l].plane.push(pe);
                if let Some((wk, kn)) = &ix {
                    let ik_raw = matmul(wk, c.idx_dim, hd, &latn, 1, false);
                    let mut ik = rmsnorm(&ik_raw, kn, c.eps);
                    rope64(&mut ik[c.idx_dim - 64..], tab, e * role.ratio as usize, false);
                    if self.quant_kv {
                        fp4_blocks(&mut ik, 32, false);
                    }
                    self.layers[l].ikeys.push(ik);
                }
            }
        }
        // selection
        let ratio = role.ratio as usize;
        if role.index_source {
            let src = role.kv_from.unwrap();
            let wqb = self.f32s(&p("indexer.attn_q_b"))?;
            let iq = matmul(&wqb, c.idx_heads * c.idx_dim, c.q_lora, &qr, n, true);
            let proj = self.f32s(&p("indexer.proj"))?;
            let iw = matmul(&proj, c.idx_heads, d, &xs, n, true);
            let sc = (c.idx_dim as f32).powf(-0.5) * (c.idx_heads as f32).powf(-0.5);
            for i in 0..n {
                let pos = start + i;
                let len = (pos + 1) / ratio;
                let mut qq = iq[i * c.idx_heads * c.idx_dim..(i + 1) * c.idx_heads * c.idx_dim].to_vec();
                for hh in 0..c.idx_heads {
                    let o = hh * c.idx_dim;
                    rope64(&mut qq[o + c.idx_dim - 64..o + c.idx_dim], tab, pos, false);
                    if self.quant_kv {
                        fp4_blocks(&mut qq[o..o + c.idx_dim], 32, false);
                    }
                }
                let w: Vec<f32> = iw[i * c.idx_heads..(i + 1) * c.idx_heads].iter().map(|v| v * sc).collect();
                let keys = &self.layers[src].ikeys;
                let mask = &cand[i];
                let use_mask = role.cand_masked && !mask.is_empty();
                let scores: Vec<f32> = (0..len).map(|e| {
                    if use_mask && !mask[e / c.cand_block] {
                        return f32::NEG_INFINITY;
                    }
                    (0..c.idx_heads).map(|hh| w[hh] * dot(&qq[hh * c.idx_dim..(hh + 1) * c.idx_dim], &keys[e]).max(0.0)).sum()
                }).collect();
                if role.cand_source {
                    let nb = len.div_ceil(c.cand_block);
                    let mut keep = vec![false; nb];
                    if nb <= c.cand_blocks {
                        keep.iter_mut().for_each(|k| *k = true);
                    } else {
                        let mut bm: Vec<f32> = (0..nb).map(|b| scores[b * c.cand_block..((b + 1) * c.cand_block).min(len)].iter().cloned().fold(f32::NEG_INFINITY, f32::max)).collect();
                        bm[(len - 1) / c.cand_block] = f32::INFINITY;
                        let mut idx: Vec<usize> = (0..nb).collect();
                        idx.sort_by(|&a, &b| bm[b].partial_cmp(&bm[a]).unwrap().then(a.cmp(&b)));
                        for &b in &idx[..c.cand_blocks] {
                            keep[b] = true;
                        }
                    }
                    cand[i] = keep;
                }
                // exact top-k of the finite scores, ties to the lower index, ascending ids
                let mut idx: Vec<usize> = (0..len).filter(|&e| scores[e].is_finite()).collect();
                idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
                idx.truncate(c.idx_top_k);
                idx.sort();
                sel[i] = idx.into_iter().map(|e| e as i32).collect();
            }
        }
        // attention core
        let kv_from = role.kv_from;
        let scale = (hd as f32).powf(-0.5);
        let sinks = self.f32s(&p("attn_sinks"))?;
        let mut o = vec![0f32; n * c.q_dim()];
        for i in 0..n {
            let pos = start + i;
            let lo = (pos + 1).saturating_sub(c.window);
            let st = &self.layers[l];
            let plane = kv_from.filter(|_| ratio > 0).map(|s| &self.layers[s].plane);
            for hh in 0..c.n_head {
                let qh = &q[i * c.q_dim() + hh * hd..i * c.q_dim() + (hh + 1) * hd];
                let mut keys: Vec<&Vec<f32>> = (lo..=pos).map(|j| &st.win[j]).collect();
                if let Some(pl) = plane {
                    for &e in &sel[i] {
                        keys.push(&pl[e as usize]);
                    }
                }
                let s: Vec<f32> = keys.iter().map(|k| dot(qh, k) * scale).collect();
                let m = s.iter().cloned().fold(sinks[hh], f32::max);
                let ex: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
                let den: f32 = ex.iter().sum::<f32>() + (sinks[hh] - m).exp();
                let out = &mut o[i * c.q_dim() + hh * hd..i * c.q_dim() + (hh + 1) * hd];
                for (k, e) in keys.iter().zip(&ex) {
                    let pw = e / den;
                    for j in 0..hd {
                        out[j] += pw * k[j];
                    }
                }
                rope64(&mut out[hd - 64..], tab, pos, true);
            }
        }
        drop(q);
        // wo_a (8 groups: group g reads heads 8g..8g+7 = 4096 values, writes 1024) and wo_b
        let woa = self.f32s(&p("attn_output_a"))?;
        let gin = c.q_dim() / c.o_groups;
        let mut u = vec![0f32; n * c.o_groups * c.o_lora];
        for g in 0..c.o_groups {
            let wg = &woa[g * c.o_lora * gin..(g + 1) * c.o_lora * gin];
            let xin: Vec<f32> = (0..n).flat_map(|i| o[i * c.q_dim() + g * gin..i * c.q_dim() + (g + 1) * gin].to_vec()).collect();
            let y = if self.q8 {
                let t = self.g.tensor(&p("attn_output_a")).unwrap();
                let rb = t.ty.row_bytes(gin);
                mm_q8(t.ty, &self.g.data(t)[g * c.o_lora * rb..(g + 1) * c.o_lora * rb], wg, c.o_lora, gin, &xin, n, true)
            } else { matmul(wg, c.o_lora, gin, &xin, n, true) };
            for i in 0..n {
                u[i * c.o_groups * c.o_lora + g * c.o_lora..i * c.o_groups * c.o_lora + (g + 1) * c.o_lora].copy_from_slice(&y[i * c.o_lora..(i + 1) * c.o_lora]);
            }
        }
        drop(woa);
        let wob = self.f32s(&p("attn_output_b"))?;
        let ya = self.mq(&p("attn_output_b"), &wob, d, c.o_groups * c.o_lora, &u, n);
        drop(wob);
        let mut hmid = vec![0f32; n * 4 * d];
        for i in 0..n {
            hmid[i * 4 * d..(i + 1) * 4 * d].copy_from_slice(&self.post(&h[i * 4 * d..(i + 1) * 4 * d], &ya[i * d..(i + 1) * d], &mixes[i]));
            pre[i] = mixes[i].pre;
        }
        // ---- MoE sublayer (its collapse uses the attention mix's pre, set above)
        let fnf = self.f32s(&p("hc_ffn_fn"))?;
        let (scf, bf) = (self.f32s(&p("hc_ffn_scale"))?, self.f32s(&p("hc_ffn_base"))?);
        let ffn_norm = self.f32s(&p("ffn_norm"))?;
        let mixf: Vec<Mix> = (0..n).map(|i| self.hc_mix(&hmid[i * 4 * d..(i + 1) * 4 * d], &fnf, &scf, &bf)).collect();
        let xf: Vec<f32> = (0..n).flat_map(|i| rmsnorm(&self.collapse(&hmid[i * 4 * d..(i + 1) * 4 * d], if cur { &mixf[i].pre } else if self.hcv == "ffnprev" { &pre_in[i] } else { &pre[i] }), &ffn_norm, c.eps)).collect();
        let y = self.moe(l, &xf, n)?;
        for i in 0..n {
            h[i * 4 * d..(i + 1) * 4 * d].copy_from_slice(&self.post(&hmid[i * 4 * d..(i + 1) * 4 * d], &y[i * d..(i + 1) * d], &mixf[i]));
            pre[i] = mixf[i].pre;
        }
        Ok(())
    }

    /// Router + shared expert + routed experts for rows `x` [n][d]; returns [n][d].
    fn moe(&mut self, l: usize, x: &[f32], n: usize) -> Result<Vec<f32>> {
        let c = self.c;
        let d = c.n_embd;
        let p = |s: &str| format!("blk.{l}.{s}.weight");
        let gw = self.f32s(&p("ffn_gate_inp"))?;
        let bias = self.f32s(&format!("blk.{l}.exp_probs_b.bias"))?;
        let logits = matmul(&gw, c.n_expert, d, x, n, true);
        let mut picks: Vec<[(usize, f32); 6]> = vec![];
        for i in 0..n {
            let sc: Vec<f32> = logits[i * c.n_expert..(i + 1) * c.n_expert].iter().map(|v| softplus(*v).sqrt()).collect();
            let mut idx: Vec<usize> = (0..c.n_expert).collect();
            idx.sort_by(|&a, &b| (sc[b] + bias[b]).partial_cmp(&(sc[a] + bias[a])).unwrap().then(a.cmp(&b)));
            let top = &idx[..c.n_expert_used];
            let sum: f32 = top.iter().map(|&e| sc[e]).sum::<f32>() + 1e-20;
            let mut pk = [(0usize, 0f32); 6];
            for (k, &e) in top.iter().enumerate() {
                pk[k] = (e, sc[e] / sum * c.weights_scale);
            }
            let mut ids = [0u16; 6];
            let mut w = [0f32; 6];
            for k in 0..6 {
                ids[k] = pk[k].0 as u16;
                w[k] = pk[k].1;
            }
            self.trace.push(Route { layer: l as u8, row: (self.ids.len() - n + i) as u32, ids, w });
            picks.push(pk);
        }
        let mut y = vec![0f32; n * d];
        // shared expert (weight 1)
        {
            let lim = c.clamp_shexp[l];
            let (wg, wu, wd) = (self.f32s(&p("ffn_gate_shexp"))?, self.f32s(&p("ffn_up_shexp"))?, self.f32s(&p("ffn_down_shexp"))?);
            let g = self.mq(&p("ffn_gate_shexp"), &wg, c.n_ff_exp, d, x, n);
            let u = self.mq(&p("ffn_up_shexp"), &wu, c.n_ff_exp, d, x, n);
            let hh: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(g.min(lim)) * u.clamp(-lim, lim)).collect();
            let o = self.mq(&p("ffn_down_shexp"), &wd, d, c.n_ff_exp, &hh, n);
            y.copy_from_slice(&o);
        }
        // routed: per expert, the rows that picked it
        let mut by_e: Vec<Vec<(usize, f32)>> = vec![vec![]; c.n_expert];
        for (i, pk) in picks.iter().enumerate() {
            for &(e, w) in pk {
                by_e[e].push((i, w));
            }
        }
        let (tg, tu, td) = (self.g.tensor(&p("ffn_gate_exps")).unwrap(), self.g.tensor(&p("ffn_up_exps")).unwrap(), self.g.tensor(&p("ffn_down_exps")).unwrap());
        let lim = c.clamp_exp[l];
        let work: Vec<usize> = (0..c.n_expert).filter(|&e| !by_e[e].is_empty()).collect();
        let g = self.g;
        let q8 = self.q8;
        let res: Vec<(usize, Vec<f32>)> = self.pool.install(|| {
            work.par_iter().map_init(|| (vec![0f32; c.n_ff_exp * d], vec![0f32; c.n_ff_exp * d], vec![0f32; c.n_ff_exp * d]), |(bg, bu, bd), &e| {
                refq::dequant(tg.ty, g.slice(tg, e), bg);
                refq::dequant(tu.ty, g.slice(tu, e), bu);
                refq::dequant(td.ty, g.slice(td, e), bd);
                let rows = &by_e[e];
                let xs: Vec<f32> = rows.iter().flat_map(|&(i, _)| x[i * d..(i + 1) * d].to_vec()).collect();
                if q8 {
                    let gg = mm_q8(tg.ty, g.slice(tg, e), bg, c.n_ff_exp, d, &xs, rows.len(), false);
                    let uu = mm_q8(tu.ty, g.slice(tu, e), bu, c.n_ff_exp, d, &xs, rows.len(), false);
                    // h is quantised unweighted (as the GPU mmq_moe_gu epilogue does); the route weight scales the down output
                    let hh: Vec<f32> = gg.iter().zip(&uu).map(|(g, u)| silu(g.min(lim)) * u.clamp(-lim, lim)).collect();
                    let mut o = mm_q8(td.ty, g.slice(td, e), bd, d, c.n_ff_exp, &hh, rows.len(), false);
                    for (r, row) in o.chunks_mut(d).enumerate() {
                        row.iter_mut().for_each(|v| *v *= rows[r].1);
                    }
                    return (e, o);
                }
                let gg = matmul(bg, c.n_ff_exp, d, &xs, rows.len(), false);
                let uu = matmul(bu, c.n_ff_exp, d, &xs, rows.len(), false);
                let hh: Vec<f32> = gg.iter().zip(&uu).enumerate().map(|(j, (g, u))| silu(g.min(lim)) * u.clamp(-lim, lim) * rows[j / c.n_ff_exp].1).collect();
                (e, matmul(bd, d, c.n_ff_exp, &hh, rows.len(), false))
            }).collect()
        });
        // combine in expert-id order per row (deterministic)
        let mut res = res;
        res.sort_by_key(|r| r.0);
        for (e, o) in res {
            for (k, &(i, _)) in by_e[e].iter().enumerate() {
                for j in 0..d {
                    y[i * d + j] += o[k * d + j];
                }
            }
        }
        Ok(y)
    }
}

/// log-softmax probability of `tok` in `logits`.
pub fn logprob(logits: &[f32], tok: usize) -> f64 {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = logits.iter().map(|&v| (v as f64 - m).exp()).sum();
    logits[tok] as f64 - m - z.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grids() {
        assert_eq!(e4m3(1.0), 1.0);
        assert_eq!(e4m3(0.3), 0.3125);
        assert_eq!(e4m3(448.0), 448.0);
        assert_eq!(e4m3(1e-4), 0.0);
        assert_eq!(e2m1(2.4), 2.0);
        assert_eq!(e2m1(2.6), 3.0);
        assert_eq!(e2m1(-0.25), 0.0);
        assert_eq!(e2m1(5.1), 6.0);
        assert_eq!(pow2_ceil(0.3), 0.5);
        assert_eq!(pow2_ceil(0.5), 0.5);
        // fp8 block: values on the grid are fixed points
        let mut x: Vec<f32> = (0..32).map(|i| i as f32 / 8.0).collect();
        let y = {
            let mut t = x.clone();
            fp8_blocks(&mut t);
            t
        };
        fp8_blocks(&mut x);
        assert_eq!(x, y);
        let mut once = x.clone();
        fp8_blocks(&mut once);
        assert_eq!(once, x, "fp8 quant-dequant is idempotent");
        let mut q: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.37).collect();
        fp4_blocks(&mut q, 16, true);
        let mut q2 = q.clone();
        fp4_blocks(&mut q2, 16, true);
        assert_eq!(q, q2, "fp4 quant-dequant is idempotent");
    }

    #[test]
    fn rope_inverse_roundtrip() {
        let c_dim = 64;
        let mut tab = vec![0f32; 4 * c_dim];
        for p in 0..4 {
            for i in 0..32 {
                let a = (p as f32 + 1.0) * (i as f32 + 1.0) * 0.013;
                tab[p * 64 + 2 * i] = a.cos();
                tab[p * 64 + 2 * i + 1] = a.sin();
            }
        }
        let x0: Vec<f32> = (0..64).map(|i| (i as f32 * 0.7).sin()).collect();
        let mut x = x0.clone();
        rope64(&mut x, &tab, 3, false);
        assert!(x.iter().zip(&x0).any(|(a, b)| (a - b).abs() > 1e-3));
        rope64(&mut x, &tab, 3, true);
        for (a, b) in x.iter().zip(&x0) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn matmul_matches_naive() {
        let (o, k, n) = (13, 37, 5);
        let w: Vec<f32> = (0..o * k).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1).collect();
        let x: Vec<f32> = (0..n * k).map(|i| ((i * 3 % 13) as f32 - 6.0) * 0.1).collect();
        for par in [false, true] {
            let y = matmul(&w, o, k, &x, n, par);
            for i in 0..n {
                for j in 0..o {
                    let r: f32 = (0..k).map(|t| w[j * k + t] * x[i * k + t]).sum();
                    assert!((y[i * o + j] - r).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn q8_mode_matches_the_engine_dot() {
        // mm_q8 (dequantised weight x dequantised q8_1 activation, Q5_K min term on the float sum) == refq::dot_q8
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (ty, bs) in [(GgmlType::Q5K, 176usize), (GgmlType::Q8_0, 34)] {
            let (o, k, n) = (5usize, 512usize, 3usize);
            let rb = ty.row_bytes(k);
            let mut raw = vec![0u8; o * rb];
            for r in 0..o * rb / bs {
                let b = &mut raw[r * bs..(r + 1) * bs];
                for v in b.iter_mut() {
                    *v = rnd() as u8;
                }
                b[..2].copy_from_slice(&refq::f32_to_f16(0.01).to_le_bytes());
                if ty == GgmlType::Q5K {
                    b[2..4].copy_from_slice(&refq::f32_to_f16(0.02).to_le_bytes());
                }
            }
            let w = dequant_bytes(ty, &raw, o * k, k).unwrap();
            let x: Vec<f32> = (0..n * k).map(|_| (rnd() % 20001) as f32 / 10000.0 - 1.0).collect();
            let y = mm_q8(ty, &raw, &w, o, k, &x, n, false);
            for i in 0..n {
                let (q, d, s) = refq::quantize_q8_1(&x[i * k..(i + 1) * k]);
                for oi in 0..o {
                    let e = refq::dot_q8(ty, &raw[oi * rb..(oi + 1) * rb], &q, &d, &s);
                    let g = y[i * o + oi] as f64;
                    assert!((g - e).abs() < 1e-3 * (1.0 + e.abs()), "{ty:?} row {i} out {oi}: {g} vs {e}");
                }
            }
        }
    }
}
