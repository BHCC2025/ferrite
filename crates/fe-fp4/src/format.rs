//! NVFP4 / FP8 number formats and the CPU reference: conversions, activation quantizers (bit-exact with
//! the k-fp4 kernels), scale tiling, dequantization and a reference GEMM.

/// e2m1 code -> value (bit 3 = sign).
pub const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
pub fn f32_to_bf16(f: f32) -> u16 {
    let b = f.to_bits();
    if f.is_nan() {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Exact e4m3 (fn) -> f32; 0x7f / 0xff are NaN.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let e = ((b >> 3) & 15) as u32;
    let m = (b & 7) as u32;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let v = if e == 0 { m as f32 / 512.0 } else { f32::from_bits(((e + 120) << 23) | (m << 20)) };
    if b & 0x80 != 0 { -v } else { v }
}

/// Nearest code in a sorted table of non-negative values, ties to the even code.
fn nearest(t: &[f32], a: f32) -> usize {
    let mut i = t.partition_point(|&v| v < a);
    if i == t.len() {
        return t.len() - 1;
    }
    if i > 0 {
        let (lo, hi) = (a - t[i - 1], t[i] - a);
        if lo < hi || (lo == hi && (i - 1) % 2 == 0) {
            i -= 1;
        }
    }
    i
}

fn e4m3_pos_table() -> &'static [f32; 127] {
    use std::sync::OnceLock;
    static T: OnceLock<[f32; 127]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0.0f32; 127];
        for (i, v) in t.iter_mut().enumerate() {
            *v = e4m3_to_f32(i as u8);
        }
        t
    })
}

/// f32 -> e4m3, round to nearest even, saturating to +-448 (`cvt.rn.satfinite.e4m3x2.f32`).
pub fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let c = nearest(e4m3_pos_table(), x.abs()) as u8;
    if x.is_sign_negative() && c != 0 { c | 0x80 } else if x.is_sign_negative() { 0x80 } else { c }
}

/// f32 -> e2m1 code, round to nearest even, saturating to +-6 (`cvt.rn.satfinite.e2m1x2.f32`).
pub fn f32_to_e2m1(x: f32) -> u8 {
    let c = nearest(&E2M1[..8], x.abs()) as u8;
    if x.is_sign_negative() { c | 8 } else { c }
}

/// NVFP4 quantization of one row (same arithmetic as `fe_fp4_quant`): returns (codes [k/2], scales [k/16]).
pub fn quant_fp4_row(x: &[u16], g: f32) -> (Vec<u8>, Vec<u8>) {
    let k = x.len();
    let mut codes = vec![0u8; k / 2];
    let mut sf = vec![0u8; k / 16];
    for b in 0..k / 16 {
        let f: Vec<f32> = x[b * 16..b * 16 + 16].iter().map(|&h| bf16_to_f32(h)).collect();
        let amax = f.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let s8 = f32_to_e4m3(amax * (1.0 / 6.0) * g);
        let sv = e4m3_to_f32(s8);
        let mul = if sv > 0.0 { g / sv } else { 0.0 };
        for j in 0..8 {
            let lo = f32_to_e2m1(f[2 * j] * mul);
            let hi = f32_to_e2m1(f[2 * j + 1] * mul);
            codes[b * 8 + j] = lo | (hi << 4);
        }
        sf[b] = s8;
    }
    (codes, sf)
}

/// FP8 quantization of one row (same arithmetic as `fe_fp8_quant`).
pub fn quant_fp8_row(x: &[u16], g: f32) -> Vec<u8> {
    x.iter().map(|&h| f32_to_e4m3(bf16_to_f32(h) * g)).collect()
}

/// Row-major scales [rows][kb] -> ferrite tiled layout ([rows/128][kb/8][128][8], rows padded to 128).
pub fn tile_scales(src: &[u8], rows: usize, kb: usize) -> Vec<u8> {
    assert!(kb % 8 == 0 && src.len() == rows * kb);
    let rp = rows.div_ceil(128) * 128;
    let mut out = vec![0u8; rp * kb];
    for r in 0..rows {
        for b in 0..kb {
            out[tiled_offset(r, b, kb)] = src[r * kb + b];
        }
    }
    out
}
pub fn tiled_offset(r: usize, b: usize, kb: usize) -> usize {
    ((r / 128) * (kb / 8) + b / 8) * 1024 + (r % 128) * 8 + b % 8
}
/// Inverse of [`tile_scales`] for the first `rows` rows.
pub fn untile_scales(t: &[u8], rows: usize, kb: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows * kb];
    for r in 0..rows {
        for b in 0..kb {
            out[r * kb + b] = t[tiled_offset(r, b, kb)];
        }
    }
    out
}
/// Row-major scales -> cuBLASLt VEC16_UE4M3 layout (128 x 4 tiles, rows padded to 128, kb to 4).
pub fn cublaslt_scales(src: &[u8], rows: usize, kb: usize) -> Vec<u8> {
    let rp = rows.div_ceil(128) * 128;
    let ct = kb.div_ceil(4);
    let mut out = vec![0u8; rp * ct * 4];
    for r in 0..rows {
        for b in 0..kb {
            out[((r / 128) * ct + b / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + b % 4] = src[r * kb + b];
        }
    }
    out
}

/// Dequantize one NVFP4 row (codes [k/2], scales [k/16]) with global scale `gs` into f32.
pub fn dequant_fp4_row(codes: &[u8], sf: &[u8], gs: f32, out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        let c = codes[i / 2];
        let q = if i % 2 == 0 { c & 15 } else { c >> 4 };
        *o = E2M1[q as usize] * e4m3_to_f32(sf[i / 16]) * gs;
    }
}
pub fn dequant_fp8_row(codes: &[u8], gs: f32, out: &mut [f32]) {
    for (o, &c) in out.iter_mut().zip(codes) {
        *o = e4m3_to_f32(c) * gs;
    }
}

/// Reference: y[r][c] = sum_k a[r][k] * w[c][k] (f64 accumulate) for every row of `a`, all `n` columns of
/// `w` ([n][k] f32), multi-threaded over columns. Returns [a.len()][n].
pub fn ref_gemm(a: &[Vec<f32>], w: &[f32], n: usize, k: usize) -> Vec<Vec<f32>> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(20);
    let chunk = n.div_ceil(threads);
    let parts: Vec<Vec<Vec<f32>>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let (c0, c1) = ((t * chunk).min(n), ((t + 1) * chunk).min(n));
                    a.iter()
                        .map(|ar| {
                            (c0..c1)
                                .map(|c| {
                                    let wr = &w[c * k..(c + 1) * k];
                                    ar.iter().zip(wr).map(|(x, y)| (*x as f64) * (*y as f64)).sum::<f64>() as f32
                                })
                                .collect()
                        })
                        .collect()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = vec![Vec::with_capacity(n); a.len()];
    for p in parts {
        for (r, v) in p.into_iter().enumerate() {
            out[r].extend(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn e4m3_roundtrip() {
        for b in 0..=255u8 {
            let v = e4m3_to_f32(b);
            if v.is_nan() {
                continue;
            }
            let c = f32_to_e4m3(v);
            assert_eq!(e4m3_to_f32(c), v, "code {b:#x}");
        }
        assert_eq!(f32_to_e4m3(1000.0), 0x7e);
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        // tie between 1.0 (0x38) and 1.125 (0x39) -> even
        assert_eq!(f32_to_e4m3(1.0625), 0x38);
    }
    #[test]
    fn e2m1_rounding() {
        assert_eq!(f32_to_e2m1(0.25), 0);
        assert_eq!(f32_to_e2m1(0.75), 2);
        assert_eq!(f32_to_e2m1(2.5), 4);
        assert_eq!(f32_to_e2m1(5.0), 6);
        assert_eq!(f32_to_e2m1(7.0), 7);
        assert_eq!(f32_to_e2m1(-3.2), 8 | 5);
    }
    #[test]
    fn tiling_roundtrip() {
        let (rows, kb) = (200, 24);
        let src: Vec<u8> = (0..rows * kb).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(untile_scales(&tile_scales(&src, rows, kb), rows, kb), src);
    }
}
