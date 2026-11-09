//! CPU reference: trellis decode (mul1), rotated-basis W_hat, original-basis W, and a reference linear.
//! Everything after the fp16 codebook output is computed in f64, so this is the ground truth the GPU
//! kernels are checked against.

use std::sync::OnceLock;

use crate::format::{value_col, value_row, Exl3Tensor, MUL1};
use crate::half::{f16_fma, f16_to_f64, f64_to_bf16_torch};

pub const HAD_SCALE: f64 = 0.088_388_347_648_318_44; // 1 / sqrt(128)

/// The 16-bit trellis state of position `t` (0..256) of a tile: stream bits [(t+1)*bits - 16, (t+1)*bits),
/// cyclic over the tile's 256 * bits bits, most significant first (u32 words, MSB = earliest bit).
pub fn tile_state(tile: &[u32], bits: u32, t: usize) -> u32 {
    let len = 256 * bits as usize;
    let start = ((t + 1) * bits as usize + len - 16) % len;
    let mut s = 0u32;
    for b in 0..16 {
        let p = (start + b) % len;
        s = (s << 1) | ((tile[p / 32] >> (31 - (p % 32))) & 1);
    }
    s
}

fn mul1_table() -> &'static [u16; 1021] {
    static T: OnceLock<[u16; 1021]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u16; 1021];
        for (s, v) in t.iter_mut().enumerate() {
            // 0x6400 + byte sum as fp16 is 1024 + sum; times 1/147.7, minus 10.39, fp16 fma
            *v = f16_fma(0x6400 + s as u16, 0x1eee, 0xc931);
        }
        t
    })
}

/// mul1 codebook value (fp16 bits) of a 16-bit state.
#[inline]
pub fn decode_mul1(state: u32) -> u16 {
    let x = state.wrapping_mul(MUL1);
    let s = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
    mul1_table()[s as usize]
}

/// A 16x16 tile as fp16 bits, [row = k][col = n].
pub fn decode_tile(tile: &[u32], bits: u32) -> [[u16; 16]; 16] {
    let mut out = [[0u16; 16]; 16];
    for lane in 0..32 {
        for j in 0..8 {
            out[value_row(lane, j)][value_col(lane, j)] = decode_mul1(tile_state(tile, bits, lane * 8 + j));
        }
    }
    out
}

/// Rotated-basis W_hat columns n0..n0+ns as fp16 bits, row-major [k][ns] (exllamav3's `reconstruct`).
pub fn w_hat_f16(t: &Exl3Tensor, n0: usize, ns: usize) -> Vec<u16> {
    assert!(n0 % 16 == 0 && ns % 16 == 0 && n0 + ns <= t.n);
    let mut w = vec![0u16; t.k * ns];
    for kt in 0..t.k / 16 {
        for nt in 0..ns / 16 {
            let d = decode_tile(t.tile(kt, n0 / 16 + nt), t.bits);
            for r in 0..16 {
                w[(kt * 16 + r) * ns + nt * 16..][..16].copy_from_slice(&d[r]);
            }
        }
    }
    w
}

/// Unnormalized natural-order (Sylvester) Walsh-Hadamard transform of 128 values.
pub fn fwht128(v: &mut [f64]) {
    debug_assert_eq!(v.len(), 128);
    let mut h = 1;
    while h < 128 {
        for i in (0..128).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// Original-basis W rows r0..r0+128 (one Hadamard block), columns n0..n0+ns, f64 [128][ns].
pub fn weight_block_f64(t: &Exl3Tensor, r0: usize, n0: usize, ns: usize) -> Vec<f64> {
    assert!(r0 % 128 == 0 && n0 % 128 == 0 && ns % 128 == 0 && n0 + ns <= t.n);
    let mut w = vec![0f64; 128 * ns];
    for ktl in 0..8 {
        for nt in 0..ns / 16 {
            let d = decode_tile(t.tile(r0 / 16 + ktl, n0 / 16 + nt), t.bits);
            for r in 0..16 {
                for c in 0..16 {
                    w[(ktl * 16 + r) * ns + nt * 16 + c] = f16_to_f64(d[r][c]);
                }
            }
        }
    }
    // H over n within each row's 128-column blocks
    for row in w.chunks_exact_mut(ns) {
        for blk in row.chunks_exact_mut(128) {
            fwht128(blk);
        }
    }
    // H over k: butterflies between whole rows
    let mut h = 1;
    while h < 128 {
        for i in (0..128).step_by(2 * h) {
            for j in i..i + h {
                for c in 0..ns {
                    let (a, b) = (w[j * ns + c], w[(j + h) * ns + c]);
                    w[j * ns + c] = a + b;
                    w[(j + h) * ns + c] = a - b;
                }
            }
        }
        h *= 2;
    }
    for r in 0..128 {
        let su = f16_to_f64(t.suh[r0 + r]) * HAD_SCALE * HAD_SCALE;
        for c in 0..ns {
            w[r * ns + c] *= su * f16_to_f64(t.svh[n0 + c]);
        }
    }
    w
}

/// Output layout of a reconstructed weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// [k][n]: `y = x @ W` (exllamav3's layout)
    KN,
    /// [n][k]: the HF `weight` layout, out_features rows of in_features
    NK,
}

/// Original-basis W columns n0..n0+ns as bf16 bits (f64 math, rounded the way torch rounds f64 -> bf16).
pub fn weight_bf16(t: &Exl3Tensor, n0: usize, ns: usize, layout: Layout) -> Vec<u16> {
    let mut out = vec![0u16; t.k * ns];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let blocks = t.k / 128;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, Vec<u16>)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if b >= blocks {
                            break;
                        }
                        let w = weight_block_f64(t, b * 128, n0, ns);
                        mine.push((b, w.iter().map(|&v| f64_to_bf16_torch(v)).collect()));
                    }
                    mine
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (b, w) in parts.into_iter().flatten() {
        for r in 0..128 {
            for c in 0..ns {
                let v = w[r * ns + c];
                match layout {
                    Layout::KN => out[(b * 128 + r) * ns + c] = v,
                    Layout::NK => out[c * t.k + b * 128 + r] = v,
                }
            }
        }
    }
    out
}

/// Reference linear in f64: `y[m][n] = sum_k x[m][k] W[k][n]` via the factored form
/// `H(H(x * suh) @ W_hat) * svh / 128`. `x` is f64 [m][k], result f64 [m][n].
pub fn linear_f64(t: &Exl3Tensor, x: &[f64], m: usize) -> Vec<f64> {
    let (k, n) = (t.k, t.n);
    assert_eq!(x.len(), m * k);
    let mut xh = x.to_vec();
    for row in xh.chunks_exact_mut(k) {
        for (i, v) in row.iter_mut().enumerate() {
            *v *= f16_to_f64(t.suh[i]);
        }
        for blk in row.chunks_exact_mut(128) {
            fwht128(blk);
        }
    }
    let mut z = vec![0f64; m * n];
    for kt in 0..k / 16 {
        for nt in 0..n / 16 {
            let d = decode_tile(t.tile(kt, nt), t.bits);
            for r in 0..16 {
                for c in 0..16 {
                    let wv = f16_to_f64(d[r][c]);
                    for mi in 0..m {
                        z[mi * n + nt * 16 + c] += xh[mi * k + kt * 16 + r] * wv;
                    }
                }
            }
        }
    }
    for row in z.chunks_exact_mut(n) {
        for blk in row.chunks_exact_mut(128) {
            fwht128(blk);
        }
        for (i, v) in row.iter_mut().enumerate() {
            *v *= f16_to_f64(t.svh[i]) * HAD_SCALE * HAD_SCALE;
        }
    }
    z
}

/// The rotated input as the GPU stores it: fp16(H(x * suh) / sqrt(128)), f64 math, [m][k] fp16 bits.
pub fn rotate_input_ref(t: &Exl3Tensor, x: &[f64], m: usize) -> Vec<u16> {
    use crate::half::f64_to_f16;
    let k = t.k;
    assert_eq!(x.len(), m * k);
    let mut xh = x.to_vec();
    let mut out = Vec::with_capacity(m * k);
    for row in xh.chunks_exact_mut(k) {
        for (i, v) in row.iter_mut().enumerate() {
            *v *= f16_to_f64(t.suh[i]);
        }
        for blk in row.chunks_exact_mut(128) {
            fwht128(blk);
        }
        out.extend(row.iter().map(|v| f64_to_f16(*v * HAD_SCALE)));
    }
    out
}

/// Reference GEMV in f64 on fp16 rotated inputs `xh` [m][k] and a decoded W_hat (fp16 bits [k][nc]: the first
/// nc columns, from `w_hat_f16`): `y = H(xh @ W_hat) * svh / sqrt(128)`, [m][nc].
pub fn gemv_ref(t: &Exl3Tensor, w_hat: &[u16], nc: usize, xh: &[u16], m: usize) -> Vec<f64> {
    let (k, n) = (t.k, nc);
    assert_eq!(xh.len(), m * k);
    assert_eq!(w_hat.len(), k * n);
    let wf: Vec<f64> = w_hat.iter().map(|&h| f16_to_f64(h)).collect();
    let mut z = vec![0f64; m * n];
    for mi in 0..m {
        let zr = &mut z[mi * n..(mi + 1) * n];
        for kk in 0..k {
            let xv = f16_to_f64(xh[mi * k + kk]);
            if xv == 0.0 {
                continue;
            }
            let wr = &wf[kk * n..(kk + 1) * n];
            for (zv, wv) in zr.iter_mut().zip(wr) {
                *zv += xv * wv;
            }
        }
    }
    for row in z.chunks_exact_mut(n) {
        for blk in row.chunks_exact_mut(128) {
            fwht128(blk);
        }
        for (i, v) in row.iter_mut().enumerate() {
            *v *= f16_to_f64(t.svh[i]) * HAD_SCALE;
        }
    }
    z
}
