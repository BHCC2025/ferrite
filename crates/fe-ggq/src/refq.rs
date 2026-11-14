//! Host reference dequantization (scalar ports of ggml-quants.c `dequantize_row_*`), the q8_1 activation
//! quantizer of ggml-cuda (quantize.cu), and f16 helpers. This is the oracle every GPU/CPU kernel is checked
//! against; it is itself checked against libggml (an external test harness).
use crate::tables::*;
use fe_gguf::GgmlType;

/// IEEE half -> f32 (subnormals, inf, nan exact).
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal: m * 2^-24
            let f = m as f32 * (1.0 / 16777216.0);
            return if s != 0 { -f } else { f };
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> IEEE half, round to nearest even (as `__float2half_rn` / F16C).
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32;
    let m = x & 0x7f_ffff;
    if e == 255 {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return sign | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return sign;
        }
        let mm = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = mm >> shift;
        let rem = mm & ((1 << shift) - 1);
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e2 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    sign | r as u16
}

#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
/// f32 -> bf16 bits, round to nearest even (nan kept quiet).
#[inline]
pub fn f32_to_bf16(f: f32) -> u16 {
    let x = f.to_bits();
    if f.is_nan() {
        return ((x >> 16) | 0x40) as u16;
    }
    ((x + 0x7fff + ((x >> 16) & 1)) >> 16) as u16
}

#[inline]
fn rd16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn h(b: &[u8], o: usize) -> f32 {
    f16_to_f32(rd16(b, o))
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Dequantize one block (`ty.block().1` bytes) into `ty.block().0` floats.
pub fn dequant_block(ty: GgmlType, b: &[u8], y: &mut [f32]) {
    use GgmlType::*;
    match ty {
        F32 => y[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        F16 => y[0] = h(b, 0),
        BF16 => y[0] = bf16_to_f32(rd16(b, 0)),
        Q8_0 => {
            let d = h(b, 0);
            for j in 0..32 {
                y[j] = d * (b[2 + j] as i8) as f32;
            }
        }
        Q2_0 => {
            let d = h(b, 0);
            for j in 0..64 {
                let q = (b[2 + j / 4] >> ((j % 4) * 2)) & 3;
                y[j] = (q as i32 - 1) as f32 * d;
            }
        }
        IQ4NL => {
            let d = h(b, 0);
            for j in 0..16 {
                y[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0xf) as usize] as f32;
                y[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
            }
        }
        IQ4XS => {
            let d = h(b, 0);
            let sh = rd16(b, 2);
            let sl = &b[4..8];
            let qs = &b[8..136];
            for ib in 0..8 {
                let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4);
                let dl = d * (ls - 32) as f32;
                for j in 0..16 {
                    y[ib * 32 + j] = dl * KVALUES_IQ4NL[(qs[ib * 16 + j] & 0xf) as usize] as f32;
                    y[ib * 32 + j + 16] = dl * KVALUES_IQ4NL[(qs[ib * 16 + j] >> 4) as usize] as f32;
                }
            }
        }
        Q4K => {
            let d = h(b, 0);
            let mn = h(b, 2);
            let sc = &b[4..16];
            let q = &b[16..144];
            for jj in 0..4 {
                let (s1, m1) = scale_min_k4(2 * jj, sc);
                let (s2, m2) = scale_min_k4(2 * jj + 1, sc);
                let (d1, mm1, d2, mm2) = (d * s1 as f32, mn * m1 as f32, d * s2 as f32, mn * m2 as f32);
                for l in 0..32 {
                    y[jj * 64 + l] = d1 * (q[jj * 32 + l] & 0xf) as f32 - mm1;
                    y[jj * 64 + 32 + l] = d2 * (q[jj * 32 + l] >> 4) as f32 - mm2;
                }
            }
        }
        Q5K => {
            let d = h(b, 0);
            let mn = h(b, 2);
            let sc = &b[4..16];
            let qh = &b[16..48];
            let ql = &b[48..176];
            for jj in 0..4 {
                let (s1, m1) = scale_min_k4(2 * jj, sc);
                let (s2, m2) = scale_min_k4(2 * jj + 1, sc);
                let (d1, mm1, d2, mm2) = (d * s1 as f32, mn * m1 as f32, d * s2 as f32, mn * m2 as f32);
                let (u1, u2) = (1u8 << (2 * jj), 2u8 << (2 * jj));
                for l in 0..32 {
                    let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                    let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                    y[jj * 64 + l] = d1 * ((ql[jj * 32 + l] & 0xf) + hi1) as f32 - mm1;
                    y[jj * 64 + 32 + l] = d2 * ((ql[jj * 32 + l] >> 4) + hi2) as f32 - mm2;
                }
            }
        }
        Q6K => {
            let d = h(b, 208);
            for n in 0..2 {
                let ql = &b[n * 64..];
                let qh = &b[128 + n * 32..];
                let sc = &b[192 + n * 8..];
                let yo = n * 128;
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    y[yo + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                    y[yo + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                    y[yo + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                    y[yo + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                }
            }
        }
        IQ2S => {
            let d = h(b, 0);
            let qs = &b[2..34];
            let signs = &b[34..66];
            let qh = &b[66..74];
            let scales = &b[74..82];
            for ib in 0..8 {
                let db = [d * (0.5 + (scales[ib] & 0xf) as f32) * 0.25, d * (0.5 + (scales[ib] >> 4) as f32) * 0.25];
                for l in 0..4 {
                    let dl = db[l / 2];
                    let idx = qs[ib * 4 + l] as usize | ((qh[ib] as usize) << (8 - 2 * l) & 0x300);
                    let g = IQ2S_GRID[idx].to_le_bytes();
                    let s = signs[ib * 4 + l];
                    for j in 0..8 {
                        y[ib * 32 + l * 8 + j] = dl * g[j] as f32 * if s & (1 << j) != 0 { -1.0 } else { 1.0 };
                    }
                }
            }
        }
        IQ3XXS => {
            let d = h(b, 0);
            let qs = &b[2..66];
            let ss = &b[66..98];
            for ib in 0..8 {
                let aux = u32::from_le_bytes([ss[4 * ib], ss[4 * ib + 1], ss[4 * ib + 2], ss[4 * ib + 3]]);
                let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                    let g1 = IQ3XXS_GRID[qs[ib * 8 + 2 * l] as usize].to_le_bytes();
                    let g2 = IQ3XXS_GRID[qs[ib * 8 + 2 * l + 1] as usize].to_le_bytes();
                    for j in 0..4 {
                        y[ib * 32 + l * 8 + j] = db * g1[j] as f32 * if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                        y[ib * 32 + l * 8 + j + 4] =
                            db * g2[j] as f32 * if signs & (1 << (j + 4)) != 0 { -1.0 } else { 1.0 };
                    }
                }
            }
        }
        IQ3S => {
            let d = h(b, 0);
            let qs = &b[2..66];
            let qh = &b[66..74];
            let signs = &b[74..106];
            let scales = &b[106..110];
            for ib in 0..8 {
                let sc = scales[ib / 2];
                let db = d * (1 + 2 * if ib % 2 == 0 { sc & 0xf } else { sc >> 4 }) as f32;
                let qhb = qh[ib] as usize;
                for l in 0..4 {
                    let i1 = qs[ib * 8 + 2 * l] as usize | ((qhb << (8 - 2 * l)) & 256);
                    let i2 = qs[ib * 8 + 2 * l + 1] as usize | ((qhb << (7 - 2 * l)) & 256);
                    let g1 = IQ3S_GRID[i1].to_le_bytes();
                    let g2 = IQ3S_GRID[i2].to_le_bytes();
                    let s = signs[ib * 4 + l];
                    for j in 0..4 {
                        y[ib * 32 + l * 8 + j] = db * g1[j] as f32 * if s & (1 << j) != 0 { -1.0 } else { 1.0 };
                        y[ib * 32 + l * 8 + j + 4] = db * g2[j] as f32 * if s & (1 << (j + 4)) != 0 { -1.0 } else { 1.0 };
                    }
                }
            }
        }
        t => panic!("refq: dequant of {} not implemented", t.name()),
    }
}

/// Whether [`dequant_block`] (and the GPU/CPU kernels) handle this type.
pub fn supported(ty: GgmlType) -> bool {
    use GgmlType::*;
    matches!(ty, F32 | F16 | BF16 | Q8_0 | Q2_0 | IQ4NL | IQ4XS | Q4K | Q5K | Q6K | IQ2S | IQ3XXS | IQ3S)
}

/// Dequantize `n` values (a whole number of blocks) from `src`.
pub fn dequant(ty: GgmlType, src: &[u8], y: &mut [f32]) {
    let (bs, by) = ty.block();
    assert_eq!(y.len() % bs, 0);
    for (i, out) in y.chunks_exact_mut(bs).enumerate() {
        dequant_block(ty, &src[i * by..(i + 1) * by], out);
    }
}

/// One q8_1 block (ggml-cuda `quantize_q8_1`): d = amax / 127, q = round(x / d) (half away from zero), and
/// s = f16-rounded sum of the FLOAT inputs (ggml's `ds.y`; used for the Q4_K/Q5_K min terms). The sum is taken in the
/// GPU's warp-butterfly order (xor 16, 8, 4, 2, 1), so host, CPU and GPU agree bit for bit. d stays f32.
pub fn q8_block(xs: &[f32]) -> ([i8; 32], f32, f32) {
    q8_block_opt(xs, false)
}
/// [`q8_block`] with the sum kept in f32 when `sum_f32`.
pub fn q8_block_opt(xs: &[f32], sum_f32: bool) -> ([i8; 32], f32, f32) {
    let amax = xs[..32].iter().fold(0f32, |a, v| a.max(v.abs()));
    let d = amax / 127.0;
    let mut q = [0i8; 32];
    for j in 0..32 {
        q[j] = if amax == 0.0 { 0 } else { (xs[j] / d).round() as i8 };
    }
    let mut v = [0f32; 32];
    v.copy_from_slice(&xs[..32]);
    for off in [16usize, 8, 4, 2, 1] {
        let w = v;
        for j in 0..32 {
            v[j] = w[j] + w[j ^ off];
        }
    }
    (q, d, if sum_f32 { v[0] } else { f16_to_f32(f32_to_f16(v[0])) })
}

/// q8_1 rows as the GPU kernel stores them: (q, d per 32, s per 32), see [`q8_block`].
pub fn quantize_q8_1(x: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    quantize_q8_1_opt(x, false)
}
pub fn quantize_q8_1_opt(x: &[f32], sum_f32: bool) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    let nb = x.len() / 32;
    let mut q = vec![0i8; x.len()];
    let mut ds = vec![0f32; nb];
    let mut ss = vec![0f32; nb];
    for b in 0..nb {
        let (qb, d, sm) = q8_block_opt(&x[b * 32..b * 32 + 32], sum_f32);
        q[b * 32..b * 32 + 32].copy_from_slice(&qb);
        ds[b] = d;
        ss[b] = sm;
    }
    (q, ds, ss)
}

/// Q4_K / Q5_K: the min of each 32-value sub-block of a 256 block (value = scale * q - min); None for other types.
pub fn kq_mins(ty: GgmlType, b: &[u8]) -> Option<[f32; 8]> {
    if !matches!(ty, GgmlType::Q4K | GgmlType::Q5K) {
        return None;
    }
    let dmin = h(b, 2);
    let mut m = [0f32; 8];
    for j in 0..8 {
        m[j] = dmin * scale_min_k4(j, &b[4..16]).1 as f32;
    }
    Some(m)
}

/// Exact model of the GEMV kernels for one weight row and one q8 activation row (f64): the weight's scaled part
/// times the quantized activations, minus each sub-block's min times the float-input sum `s`.
pub fn dot_q8(ty: GgmlType, row: &[u8], q: &[i8], d: &[f32], s: &[f32]) -> f64 {
    let k = q.len();
    let mut w = vec![0f32; k];
    dequant(ty, row, &mut w);
    let mut acc = 0f64;
    for i in 0..k {
        acc += w[i] as f64 * q[i] as f64 * d[i / 32] as f64;
    }
    if matches!(ty, GgmlType::Q4K | GgmlType::Q5K) {
        let (_, by) = ty.block();
        for blk in 0..k / 256 {
            let m = kq_mins(ty, &row[blk * by..(blk + 1) * by]).unwrap();
            for j in 0..8 {
                let sb = blk * 8 + j;
                let sq: i32 = q[sb * 32..sb * 32 + 32].iter().map(|&v| v as i32).sum();
                // the dequantized value already carries -m * (d * sum q); replace it by -m * s
                acc += m[j] as f64 * (d[sb] as f64 * sq as f64 - s[sb] as f64);
            }
        }
    }
    acc
}
