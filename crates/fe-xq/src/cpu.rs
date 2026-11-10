//! CPU reference for XQ (the target the Rust B70 kernels are checked against, next to arc-engine's own outputs).
//!
//! - [`prep`]: the activation prepass of `xq_prep` (`kernels/xpu/xq.cl`) bit for bit: `x~ = fp16_rte(FWHT128(f32(x) * f32(su)) * HAD_SCALE)`
//!   with the kernel's butterfly order (index bits 4, 5, 6 in registers, then 0..3 across lanes; `a + b` / `a - b`, lower index first),
//!   and the per-32 sums `xs` of the fp16 values (lane pair sums, then a sub-group reduction; see [`subgroup_sum16`]).
//! - [`dequant`] / [`gemv_f64`]: the format's semantics, `y[r][n] = sum_k x~[r][k] * s[n][k / g] * q[n][k]` in float64.
//! - [`gemv_model`]: an fp32 model of the decode matvec `xq_mv_*` (per group a DPAS chain `c = sum x~ (1024 + u)`, then
//!   `acc = fma(s, fma(-CORR, xs_g, c), acc)`, K split partials added in split order). DPAS internals (product summation order and rounding
//!   inside one k16 step) are hardware-defined, so this model is a tight tolerance target, not a bitwise one; bitwise equality is checked
//!   GPU against GPU (Rust kernel vs `xq{3,4,5,6,8}.spv`).
use crate::half::{bf16_to_f32, f16_to_f32, f32_to_bf16, f64_to_f16};
use crate::pack::{self, b_dword, chunk_dwords, lane_dwords, split_range};

/// 1 / sqrt(128) as the kernels' fp32 literal `0.08838834764831845f`.
#[allow(clippy::excessive_precision)]
pub const HAD_SCALE: f32 = 0.08838834764831845_f32; // the same decimal as the kernel literal, rounded to f32 the same way

/// Activation rows as the prepass reads them (`ld` in xq.cl: dt 0 = fp16, 1 = bf16, 2 = fp32).
#[derive(Clone, Copy)]
pub enum Act<'a> {
    F16(&'a [u16]),
    Bf16(&'a [u16]),
    F32(&'a [f32]),
}

impl Act<'_> {
    pub fn get(&self, i: usize) -> f32 {
        match self {
            Act::F16(v) => f16_to_f32(v[i]),
            Act::Bf16(v) => bf16_to_f32(v[i]),
            Act::F32(v) => v[i],
        }
    }
}

/// fp32 -> fp16 bits, round to nearest even (`convert_half_rte`). Exact: f32 -> f64 is lossless, then one rounding.
pub fn f32_to_f16(x: f32) -> u16 {
    f64_to_f16(x as f64)
}

/// The unscaled Walsh-Hadamard transform of one 128-block in `fwht8`'s operation order (element `lane + 16 i`).
pub fn fwht128_xq(v: &mut [f32; 128]) {
    // bits 4..6 (the 8 values a lane holds), lower index first
    for m in [1usize, 2, 4] {
        for lane in 0..16 {
            for i in 0..8 {
                if i & m == 0 {
                    let (a, b) = (v[lane + 16 * i], v[lane + 16 * (i | m)]);
                    v[lane + 16 * i] = a + b;
                    v[lane + 16 * (i | m)] = a - b;
                }
            }
        }
    }
    // bits 0..3 by shuffle_xor: v = (lane & m) ? o - v : v + o
    for m in [1usize, 2, 4, 8] {
        let old = *v;
        for lane in 0..16 {
            for i in 0..8 {
                let x = old[lane + 16 * i];
                let o = old[(lane ^ m) + 16 * i];
                v[lane + 16 * i] = if lane & m != 0 { o - x } else { x + o };
            }
        }
    }
}

/// Model of `sub_group_reduce_add` over 16 lanes: a halving tree (lane i + lane i + 8, then + 4, + 2, + 1).
/// The hardware order is IGC's choice; [`prep`] results are checked against device dumps of xs (oracle `xq_prep_*`).
pub fn subgroup_sum16(v: &[f32; 16]) -> f32 {
    let mut a = *v;
    let mut w = 8;
    while w >= 1 {
        for i in 0..w {
            a[i] += a[i + w];
        }
        w /= 2;
    }
    a[0]
}

/// The prepass result in row-major form: `xt` fp16 bits [m][k], `xs` fp32 [m][k / 32].
#[derive(Clone, Debug)]
pub struct Prepped {
    pub m: usize,
    pub k: usize,
    pub xt: Vec<u16>,
    pub xs: Vec<f32>,
}

/// `xq_prep` for m rows of x (row stride `xstride` elements), rotated with `su` (fp16 bits [k]) or not.
pub fn prep(x: Act, xstride: usize, m: usize, k: usize, su: Option<&[u16]>) -> Prepped {
    assert!(k % 128 == 0, "prep: k {k} not a multiple of 128");
    let mut xt = vec![0u16; m * k];
    let mut xs = vec![0f32; m * k / 32];
    for r in 0..m {
        for blk in 0..k / 128 {
            let mut v = [0f32; 128];
            for (j, vj) in v.iter_mut().enumerate() {
                let a = x.get(r * xstride + blk * 128 + j);
                *vj = match su {
                    Some(su) => a * f16_to_f32(su[blk * 128 + j]),
                    None => a,
                };
            }
            if su.is_some() {
                fwht128_xq(&mut v);
                for vj in v.iter_mut() {
                    *vj *= HAD_SCALE;
                }
            }
            let mut h = [0f32; 128];
            for j in 0..128 {
                let b = f32_to_f16(v[j]);
                xt[r * k + blk * 128 + j] = b;
                h[j] = f16_to_f32(b);
            }
            // xs of 32-block jj: lane l adds its values l + 32 jj and l + 32 jj + 16, then the sub-group sum
            for jj in 0..4 {
                let mut lanes = [0f32; 16];
                for (l, lv) in lanes.iter_mut().enumerate() {
                    *lv = h[32 * jj + l] + h[32 * jj + 16 + l];
                }
                xs[r * (k / 32) + blk * 4 + jj] = subgroup_sum16(&lanes);
            }
        }
    }
    Prepped { m, k, xt, xs }
}

/// The prepass exactly as the device scratch holds it after `xq_prep` of m rows: `(xt [NR][K/16][MW][16] fp16, xs [NR][K/32][MW] fp32)`.
/// The padding rows m..MW*NR run through the same arithmetic on x = 0, so with a rotation they hold signed zeros (0 * negative su = -0.0,
/// which the butterflies partly keep): bit-exact only this way.
pub fn prep_device(x: Act, xstride: usize, m: usize, k: usize, su: Option<&[u16]>) -> (Vec<u16>, Vec<f32>) {
    let (mw, nr) = pack::VARIANTS[pack::variant_for(m as u32).unwrap()];
    let mpad = (mw * nr) as usize;
    let mut xp = vec![0f32; mpad * k];
    for r in 0..m {
        for c in 0..k {
            xp[r * k + c] = x.get(r * xstride + c);
        }
    }
    let mut p = prep(Act::F32(&xp), k, mpad, k, su);
    let (xt, xs) = (p.xt_device_rows(m as u32), p.xs_device_rows(m as u32));
    p.m = m;
    (xt, xs)
}

impl Prepped {
    /// The device scratch layout of xt for a call of `m` rows: `[row group][K/16][MW][16]` fp16, rows m..MW*NR zero (see [`prep_device`]
    /// for the device's exact padding rows).
    pub fn xt_device(&self) -> Vec<u16> {
        self.xt_device_rows(self.m as u32)
    }

    /// xt in the layout of an m-row call, from as many of this prepass's rows as the layout holds (all MW*NR when computed padded).
    fn xt_device_rows(&self, m: u32) -> Vec<u16> {
        let (mw, nr) = pack::VARIANTS[pack::variant_for(m).unwrap()];
        let (mw, nr, k) = (mw as usize, nr as usize, self.k);
        let mut o = vec![0u16; nr * k * mw];
        for row in 0..self.m.min(mw * nr) {
            let (rg, r) = (row / mw, row % mw);
            for kk in 0..k {
                o[((rg * (k / 16) + kk / 16) * mw + r) * 16 + kk % 16] = self.xt[row * k + kk];
            }
        }
        o
    }

    /// The device layout of xs: `[row group][K/32][MW]` fp32.
    pub fn xs_device(&self) -> Vec<f32> {
        self.xs_device_rows(self.m as u32)
    }

    fn xs_device_rows(&self, m: u32) -> Vec<f32> {
        let (mw, nr) = pack::VARIANTS[pack::variant_for(m).unwrap()];
        let (mw, nr, k) = (mw as usize, nr as usize, self.k);
        let mut o = vec![0f32; nr * (k / 32) * mw];
        for row in 0..self.m.min(mw * nr) {
            let (rg, r) = (row / mw, row % mw);
            for j in 0..k / 32 {
                o[(rg * (k / 32) + j) * mw + r] = self.xs[row * (k / 32) + j];
            }
        }
        o
    }
}

/// Effective weights `s[n][k / g] * q[n][k]` (exact in f32: an fp16 scale times a code of at most 8 bits) as [n][k].
pub fn dequant(g: u32, n: usize, k: usize, q: &[i8], s: &[u16]) -> Vec<f32> {
    let g = g as usize;
    let mut w = vec![0f32; n * k];
    for r in 0..n {
        for c in 0..k {
            w[r * k + c] = f16_to_f32(s[r * (k / g) + c / g]) * q[r * k + c] as f32;
        }
    }
    w
}

/// `y[r][n] = sum_k x~[r][k] * s[n][k/g] * q[n][k]` in float64 (x~ fp16 bits [m][k]) -> [m][n].
pub fn gemv_f64(g: u32, n: usize, k: usize, q: &[i8], s: &[u16], xt: &[u16], m: usize) -> Vec<f64> {
    let g = g as usize;
    let xf: Vec<f64> = xt.iter().map(|&b| f16_to_f32(b) as f64).collect();
    let mut y = vec![0f64; m * n];
    for o in 0..n {
        let qr = &q[o * k..(o + 1) * k];
        let sr: Vec<f64> = s[o * (k / g)..(o + 1) * (k / g)].iter().map(|&b| f16_to_f32(b) as f64).collect();
        for r in 0..m {
            let xr = &xf[r * k..(r + 1) * k];
            let mut acc = 0f64;
            for gi in 0..k / g {
                let mut part = 0f64;
                for c in gi * g..(gi + 1) * g {
                    part += xr[c] * qr[c] as f64;
                }
                acc += part * sr[gi];
            }
            y[r * n + o] = acc;
        }
    }
    y
}

/// fp32 model of `xq_mv_{g}_*` on packed weights `w` (the device layout) and a prepass `p`: [m][n] fp32.
/// `splits` = [`pack::split_for`] of the shape (what the kernel's work-group size is set to).
pub fn gemv_model(bits: u32, g: u32, n: usize, k: usize, w: &[u32], p: &Prepped, splits: u32) -> Vec<f32> {
    let gu = g as usize;
    let ng = k / gu;
    let ch = chunk_dwords(bits, g);
    let dl = lane_dwords(bits, g);
    let corr = 1024.0f32 + (1u32 << (bits - 1)) as f32;
    let mut y = vec![0f32; p.m * n];
    for rb in 0..n / 16 {
        for lane in 0..16 {
            let o = rb * 16 + lane;
            for r in 0..p.m {
                let xr = &p.xt[r * k..(r + 1) * k];
                let mut parts = Vec::with_capacity(splits as usize);
                for sp in 0..splits {
                    let (g0, g1) = split_range(ng as u32, sp, splits);
                    let mut acc = 0f32;
                    for gi in g0 as usize..g1 as usize {
                        let c0 = &w[(rb * ng + gi) * ch..(rb * ng + gi + 1) * ch];
                        let words: Vec<u32> = (0..dl).map(|i| c0[i * 16 + lane]).collect();
                        let sc = f16_to_f32((c0[16 * dl + lane / 2] >> (16 * (lane % 2))) as u16);
                        let mut c = 0f32;
                        for t in 0..gu / 16 {
                            // one k16 DPAS step: exact products, summed in f64, one rounding into the fp32 accumulator (model)
                            let mut sum = c as f64;
                            for d in 0..8 {
                                let b = b_dword(bits, g, &words, t, d);
                                for e in 0..2 {
                                    let bv = f16_to_f32(((b >> (16 * e)) & 0xffff) as u16) as f64;
                                    sum += f16_to_f32(xr[gi * gu + 16 * t + 2 * d + e]) as f64 * bv;
                                }
                            }
                            c = sum as f32;
                        }
                        let xsg = xs_group(gu, &p.xs[r * (k / 32)..(r + 1) * (k / 32)], gi);
                        acc = sc.mul_add((-corr).mul_add(xsg, c), acc);
                    }
                    parts.push(acc);
                }
                y[r * n + o] = if splits > 1 { parts.iter().fold(0f32, |v, &a| v + a) } else { parts[0] };
            }
        }
    }
    y
}

/// The xs sum of group gi: G / 32 partial sums added pairwise in a fixed order (`xs_group` in xq.cl).
pub fn xs_group(g: usize, xs_row: &[f32], gi: usize) -> f32 {
    let p = &xs_row[gi * (g / 32)..];
    match g {
        32 => p[0],
        64 => p[0] + p[1],
        _ => (p[0] + p[1]) + (p[2] + p[3]),
    }
}

/// fp32 -> bf16 bits as the kernels store (`to_bf`: round to nearest even).
pub fn to_bf(v: f32) -> u16 {
    f32_to_bf16(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_is_scaled_orthogonal() {
        // H H = 128 I: transforming twice gives 128 x (exact for small integers)
        let mut v = [0f32; 128];
        for (i, x) in v.iter_mut().enumerate() {
            *x = (i as f32 - 60.0) * 0.5;
        }
        let orig = v;
        fwht128_xq(&mut v);
        fwht128_xq(&mut v);
        for i in 0..128 {
            assert_eq!(v[i], 128.0 * orig[i]);
        }
    }

    #[test]
    fn fwht_matches_sylvester_hadamard() {
        // H[i][j] = (-1)^popcount(i & j), any butterfly order gives this matrix
        let mut v = [0f32; 128];
        v[5] = 1.0;
        fwht128_xq(&mut v);
        for (i, x) in v.iter().enumerate() {
            assert_eq!(*x, if (i & 5).count_ones() % 2 == 0 { 1.0 } else { -1.0 });
        }
    }

    #[test]
    fn model_close_to_f64() {
        let (bits, g, n, k, m) = (5u32, 128u32, 32usize, 512usize, 3usize);
        let mut seed = 1u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as u32
        };
        let q: Vec<i8> = (0..n * k).map(|_| ((rnd() % 32) as i32 - 16) as i8).collect();
        let s: Vec<u16> = (0..n * k / 128).map(|_| f32_to_f16(0.001 + (rnd() % 100) as f32 * 1e-4)).collect();
        let su: Vec<u16> = (0..k).map(|_| f32_to_f16(if rnd() % 2 == 0 { 1.0 } else { -1.0 })).collect();
        let x: Vec<f32> = (0..m * k).map(|_| (rnd() % 2000) as f32 / 500.0 - 2.0).collect();
        let p = prep(Act::F32(&x), k, m, k, Some(&su));
        let w = pack::pack(bits, g, n, k, &q, &s).unwrap();
        let y = gemv_model(bits, g, n, k, &w, &p, pack::split_for(n, k, g, None));
        let r = gemv_f64(g, n, k, &q, &s, &p.xt, m);
        let mx = r.iter().fold(0f64, |a, &b| a.max(b.abs()));
        let err = y.iter().zip(&r).fold(0f64, |a, (&u, &v)| a.max((u as f64 - v).abs()));
        assert!(err <= 1e-4 * mx, "model err {err} of max {mx}");
    }
}
