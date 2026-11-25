//! Inkling-2-Mini vision encoder kernels (the ViT backbone and the projector that turn an image into prompt embeddings).
//! Activations are f32, weights are read as shipped (bf16): the encoder's output is compared against the fp32 reference, and
//! bf16 activations do not hold that comparison (a bf16 run of the reference itself differs from its fp32 run by more than
//! the check allows). The order of every sum is fixed, so a row's bits do not depend on the other rows of a call.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

extern crate kcommon;

use kcommon::math::{exp_det, fma, sqrt};
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

#[inline(always)]
fn bf(h: u16) -> f32 {
    kcommon::bf16::to_f32(h)
}

/// GELU (the exact erf form, torch `F.gelu`): x * Phi(x), absolute error below 5e-7.
/// z = |x| / sqrt(2). z < 1.25: erf(z) = 2/sqrt(pi) * exp(-z^2) * sum_n 2^n z^(2n+1) / (1 * 3 * ... * (2n+1)) (all terms positive);
/// z >= 1.25: erfc(z) = exp(-z^2) / sqrt(pi) / (z + (1/2) / (z + 1 / (z + (3/2) / (z + ...)))) (60 levels, evaluated backwards).
#[inline(always)]
fn gelu(x: f32) -> f32 {
    let z = x.abs() * core::f32::consts::FRAC_1_SQRT_2;
    let z2 = z * z;
    let e = exp_det(-z2);
    let phi = if z < 1.25 {
        let tz = 2.0 * z2;
        let (mut term, mut s) = (z, z);
        let mut n = 0u32;
        while n < 28 {
            term = term * tz / (2 * n + 3) as f32;
            s += term;
            n += 1;
        }
        let erf = 1.128_379_2 * e * s;
        if x >= 0.0 { 0.5 * (1.0 + erf) } else { 0.5 * (1.0 - erf) }
    } else {
        let mut f = z;
        let mut k = 60u32;
        while k > 0 {
            f = z + (k as f32 * 0.5) / f;
            k -= 1;
        }
        let erfc = 0.564_189_6 * e / f;
        if x >= 0.0 { 1.0 - 0.5 * erfc } else { 0.5 * erfc }
    };
    x * phi
}

const T: u32 = 64; // output tile side (rows and columns)
const BK: u32 = 16; // k values a stage
/// Dynamic shared memory of `inkv_gemm`: the A and the W tile of one stage, f32 [16][64] each.
pub const GEMM_SMEM: u32 = 2 * BK * T * 4;

/// flags of `inkv_gemm`
pub const W_BF16: u32 = 1; // W is bf16 (a shipped weight); else f32 (an activation)
pub const W_TR: u32 = 2; // W is stored transposed: element (n, k) at w[k * ldw + n]
pub const ACT_GELU: u32 = 4; // GELU after the bias
pub const ACCUM: u32 = 8; // C = C + ls[n] * value (ls bf16, 1 when null): the residual update with layer scale

/// C[M][N] (f32, stride ldc) = A[M][K] (f32, stride lda) . W[N][K]^T (stride ldw; see W_BF16 / W_TR), f32 products added in
/// ascending k (fma); then + bias[n] (bf16, may be null), GELU (ACT_GELU), and the residual form (ACCUM).
/// grid = (ceil(N / 64), ceil(M / 64)), block = 256, smem = GEMM_SMEM. A thread owns a 4 x 4 block of the tile.
/// Element indexes are computed in 32 bits and widened once (the caller checks that they fit): ptxas 13.0, the DGX Spark
/// toolkit, can drop the high half of an address built from a widened index times a stride inside an unrolled loop (see
/// k-inkm `ink_sconv`).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn inkv_gemm(c: *mut f32, a: *const f32, w: *const u8, bias: *const u16, ls: *const u16, m: u32,
    n: u32, k: u32, lda: u32, ldw: u32, ldc: u32, flags: u32) {
    let sa: Shared<f32> = dyn_base();
    let sw: Shared<f32> = sa.add(BK * T);
    let t = tid_x();
    let n0 = bid_x() * T;
    let m0 = bid_y() * T;
    let tx = t & 15;
    let ty = t >> 4;
    let wbf = flags & W_BF16 != 0;
    let wtr = flags & W_TR != 0;
    let mut acc = [[0.0f32; 4]; 4];
    let mut k0 = 0u32;
    while k0 < k {
        kcommon::unroll!(i in 4 => {
            let idx = t + 256 * i as u32;
            let r = idx >> 4;
            let kk = idx & 15;
            let v = if m0 + r < m && k0 + kk < k { *a.add(((m0 + r) * lda + (k0 + kk)) as usize) } else { 0.0 };
            sa.set(kk * T + r, v);
        });
        kcommon::unroll!(i in 4 => {
            let idx = t + 256 * i as u32;
            let (r, kk) = if wtr { (idx & 63, idx >> 6) } else { (idx >> 4, idx & 15) };
            let v = if n0 + r < n && k0 + kk < k {
                let off = (if wtr { (k0 + kk) * ldw + (n0 + r) } else { (n0 + r) * ldw + (k0 + kk) }) as usize;
                if wbf { bf(*(w as *const u16).add(off)) } else { *(w as *const f32).add(off) }
            } else {
                0.0
            };
            sw.set(kk * T + r, v);
        });
        sync_threads();
        let mut kk = 0u32;
        while kk < BK {
            let av: [f32; 4] = sa.add(kk * T + ty * 4).cast::<[f32; 4]>().ld();
            let wv: [f32; 4] = sw.add(kk * T + tx * 4).cast::<[f32; 4]>().ld();
            kcommon::unroll!(i in 4 => {
                kcommon::unroll!(j in 4 => {
                    acc[i][j] = fma(av[i], wv[j], acc[i][j]);
                });
            });
            kk += 1;
        }
        sync_threads();
        k0 += BK;
    }
    kcommon::unroll!(i in 4 => {
        let row = m0 + ty * 4 + i as u32;
        if row < m {
            kcommon::unroll!(j in 4 => {
                let col = n0 + tx * 4 + j as u32;
                if col < n {
                    let mut v = acc[i][j];
                    if !bias.is_null() {
                        v += bf(*bias.add(col as usize));
                    }
                    if flags & ACT_GELU != 0 {
                        v = gelu(v);
                    }
                    let p = c.add((row * ldc + col) as usize);
                    if flags & ACCUM != 0 {
                        let s = if ls.is_null() { 1.0 } else { bf(*ls.add(col as usize)) };
                        v = *p + s * v;
                    }
                    *p = v;
                }
            });
        }
    });
}

/// LayerNorm of a row: y = (x - mean) / sqrt(var + eps) * w + b (biased variance; w, b bf16). y may be x.
/// grid = rows, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn inkv_ln(y: *mut f32, x: *const f32, w: *const u16, b: *const u16, dim: u32, ldx: u32, ldy: u32,
    eps: f32) {
    let red: Shared<f32> = static_shared!(inkv_ln_red, f32, 128);
    let r = bid_x() as usize;
    let xr = x.add(r * ldx as usize);
    let yr = y.add(r * ldy as usize);
    let mut s = 0.0f32;
    let mut i = tid_x();
    while i < dim {
        s += *xr.add(i as usize);
        i += bdim_x();
    }
    let mean = warp::block_sum(s, red) / dim as f32;
    let mut ss = 0.0f32;
    let mut i = tid_x();
    while i < dim {
        let d = *xr.add(i as usize) - mean;
        ss += d * d;
        i += bdim_x();
    }
    let var = warp::block_sum(ss, red) / dim as f32;
    let sd = sqrt(var + eps);
    let mut i = tid_x();
    while i < dim {
        let d = *xr.add(i as usize) - mean;
        *yr.add(i as usize) = d / sd * bf(*w.add(i as usize)) + bf(*b.add(i as usize));
        i += bdim_x();
    }
}

/// Softmax of a row of n scores in place (max subtracted, exp from IEEE operations only).
/// grid = rows, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn inkv_softmax(s: *mut f32, n: u32, ld: u32) {
    let red: Shared<f32> = static_shared!(inkv_sm_red, f32, 128);
    let row = s.add(bid_x() as usize * ld as usize);
    let mut mx = f32::NEG_INFINITY;
    let mut i = tid_x();
    while i < n {
        mx = mx.max(*row.add(i as usize));
        i += bdim_x();
    }
    let mx = warp::block_max(mx, red);
    let mut tot = 0.0f32;
    let mut i = tid_x();
    while i < n {
        let e = exp_det(*row.add(i as usize) - mx);
        *row.add(i as usize) = e;
        tot += e;
        i += bdim_x();
    }
    let tot = warp::block_sum(tot, red);
    let mut i = tid_x();
    while i < n {
        *row.add(i as usize) = *row.add(i as usize) / tot;
        i += bdim_x();
    }
}

/// Axial 2-D rotary embedding in place on the query and key rows (f32 [rows][heads * 64]): each head has 32 channel pairs,
/// pair p is turned by the angle whose cos / sin are cs[row][p] / cs[row][32 + p] (the host fills the table: pairs 0..16 the
/// column angles, 16..32 the row angles). The query is also multiplied by `qscale` (head_dim^-0.5).
/// grid = rows, block = 256; heads * 32 pairs a row.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn inkv_rope(q: *mut f32, kx: *mut f32, cs: *const f32, heads: u32, ld: u32, qscale: f32) {
    let r = bid_x() as usize;
    let qr = q.add(r * ld as usize);
    let kr = kx.add(r * ld as usize);
    let cr = cs.add(r * 64);
    let mut i = tid_x();
    while i < heads * 32 {
        let p = (i & 31) as usize;
        let (co, si) = (*cr.add(p), *cr.add(32 + p));
        let e = 2 * i as usize;
        let (a, b) = (*qr.add(e), *qr.add(e + 1));
        *qr.add(e) = (a * co - b * si) * qscale;
        *qr.add(e + 1) = (b * co + a * si) * qscale;
        let (a, b) = (*kr.add(e), *kr.add(e + 1));
        *kr.add(e) = a * co - b * si;
        *kr.add(e + 1) = b * co + a * si;
        i += bdim_x();
    }
}
