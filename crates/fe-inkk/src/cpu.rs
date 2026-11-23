//! CPU reference for the routed experts: decodes the release bytes exactly like `Inkling2LoopDequantizeExperts`
//! (f32 `(code * scale) * factor`, then bf16 for NVFP4; `value * 2^(s-127)` for MXFP8) and runs the reference
//! expert (bf16 gate|up, softcapped SwiGLU, bf16 down) with f64 dot products.

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
pub fn f32_to_bf16(f: f32) -> u16 {
    let b = f.to_bits();
    if f.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
pub fn round_bf16(f: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(f))
}
pub fn e4m3_to_f32(b: u8) -> f32 {
    let e = ((b >> 3) & 15) as u32;
    let m = (b & 7) as u32;
    let v = if e == 0 { m as f32 / 512.0 } else { f32::from_bits(((e + 120) << 23) | (m << 20)) };
    if b & 0x80 != 0 { -v } else { v }
}
pub const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// One NVFP4 weight row (`k` values) as bf16-rounded f32. `codes` = k / 2 bytes, `scales` = k / 16 e4m3 bytes.
pub fn dec_fp4_row(codes: &[u8], scales: &[u8], factor: f32, out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        let byte = codes[i / 2];
        let nib = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
        let s = e4m3_to_f32(scales[i / 16]);
        *o = round_bf16((E2M1[nib as usize] * s) * factor);
    }
}

/// One MXFP8 weight row as bf16-rounded f32 (exact). `vals` = k e4m3 bytes, `scales` = k / 32 ue8m0 bytes.
pub fn dec_fp8_row(vals: &[u8], scales: &[u8], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        let sc = 2.0f32.powi(scales[i / 32] as i32 - 127);
        let sc = if scales[i / 32] == 0 { f32::from_bits(0x0040_0000) } else { sc };
        *o = round_bf16(e4m3_to_f32(vals[i]) * sc);
    }
}

pub fn dot(w: &[f32], x: &[f32]) -> f32 {
    let mut a = 0.0f64;
    for (p, q) in w.iter().zip(x) {
        a += *p as f64 * *q as f64;
    }
    a as f32
}

pub fn swiglu(g: f32, u: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-g).exp());
    round_bf16((g / 4.0).tanh() * 4.0 * sig * (u / 25.0).tanh() * 25.0)
}

/// f32 -> e4m3 code, round to nearest even, saturating at 448 (cvt.rn.satfinite.e4m3x2).
pub fn f32_to_e4m3(x: f32) -> u8 {
    let neg = x.is_sign_negative();
    let a = x.abs();
    let mut best = 0u8;
    let mut bd = f32::INFINITY;
    for c in 0u8..=0x7e {
        let v = e4m3_to_f32(c);
        let d = (v - a).abs();
        if d < bd || (d == bd && c % 2 == 0) {
            bd = d;
            best = c;
        }
    }
    best | if neg { 0x80 } else { 0 }
}

/// f32 -> e2m1 code (sign in bit 3), round to nearest even, saturating at 6.
pub fn f32_to_e2m1(x: f32) -> u8 {
    let a = x.abs();
    let m = if a <= 0.25 { 0 } else if a < 0.75 { 1 } else if a <= 1.25 { 2 } else if a < 1.75 { 3 }
        else if a <= 2.5 { 4 } else if a < 3.5 { 5 } else if a <= 5.0 { 6 } else { 7 };
    m | if x.is_sign_negative() { 8 } else { 0 }
}

/// The release's NVFP4 activation rule (SGLang `quantize_part`): returns (codes K/2 bytes, e4m3 scale bits K/16, row amax).
pub fn quant_row_nvfp4(x: &[f32]) -> (Vec<u8>, Vec<u8>, f32) {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let encoder = (1536.0f32 / amax).min(f32::MAX);
    let decoder = 1.0f32 / encoder;
    let outer = amax * (1.0f32 / 1536.0f32);
    let mut codes = vec![0u8; x.len() / 2];
    let mut sf = vec![0u8; x.len() / 16];
    for b in 0..x.len() / 16 {
        let blk = &x[b * 16..b * 16 + 16];
        let bm = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let b6 = f32_to_e4m3((bm * (encoder * (1.0f32 / 6.0f32))).min(448.0));
        let b4 = f32_to_e4m3((bm * (encoder * 0.25f32)).min(448.0));
        let (s6, s4) = (e4m3_to_f32(b6), e4m3_to_f32(b4));
        let inv6 = (1.0f32 / (s6 * decoder)).min(f32::MAX);
        let inv4 = (1.0f32 / (s4 * decoder)).min(f32::MAX);
        let mut c6 = [0u8; 16];
        let mut c4 = [0u8; 16];
        let mut e6 = [0f32; 16];
        let mut e4 = [0f32; 16];
        for i in 0..16 {
            c6[i] = f32_to_e2m1(blk[i] * inv6);
            c4[i] = f32_to_e2m1(blk[i] * inv4);
            e6[i] = ((E2M1[c6[i] as usize] * s6) * outer - blk[i]).abs();
            e4[i] = ((E2M1[c4[i] as usize] * s4) * outer - blk[i]).abs();
        }
        let mut w = 8;
        while w >= 1 {
            for i in 0..w {
                e6[i] = e6[2 * i] + e6[2 * i + 1];
                e4[i] = e4[2 * i] + e4[2 * i + 1];
            }
            w /= 2;
        }
        let (c, bits) = if e4[0] < e6[0] { (c4, b4) } else { (c6, b6) };
        sf[b] = bits;
        for i in 0..8 {
            codes[b * 8 + i] = c[2 * i] | c[2 * i + 1] << 4;
        }
    }
    (codes, sf, amax)
}

/// Tiled-scale byte offset (k-ink `sf_tiled`).
pub fn sf_tiled(row: usize, b: usize, kb: usize) -> usize {
    ((row >> 7) * (kb >> 3) + (b >> 3)) * 1024 + (row & 127) * 8 + (b & 7)
}

/// The release's MXFP8 activation rule (SGLang `mxfp8_half_block`): returns (e4m3 bytes K, ue8m0 exponent bytes K/32).
pub fn quant_row_mxfp8(x: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let mut vals = vec![0u8; x.len()];
    let mut ex = vec![0u8; x.len() / 32];
    let min_scale = f32::from_bits(0x0040_0000);
    for b in 0..x.len() / 32 {
        let blk = &x[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let raw = amax * (1.0f32 / 448.0f32);
        let mut e = ((raw.to_bits().wrapping_add(0x007F_FFFF)) & 0x7F80_0000) >> 23;
        e = if raw <= min_scale { 0 } else { e.min(254) };
        let scale = f32::from_bits(e << 23).max(min_scale);
        let inv = if e == 0 { f32::from_bits(0x7F00_0000) } else { 1.0f32 / scale };
        for i in 0..32 {
            vals[b * 32 + i] = f32_to_e4m3((blk[i] * inv).max(-448.0).min(448.0));
        }
        ex[b] = e as u8;
    }
    (vals, ex)
}

/// 2^(e - 127) as f32 (the f32 denormal 2^-127 for e = 0).
pub fn ue8m0(e: u8) -> f32 {
    if e == 0 { f32::from_bits(0x0040_0000) } else { f32::from_bits((e as u32) << 23) }
}

/// MXFP8 scale byte offset (k-ink `sf8_tiled`), kb = K / 32.
pub fn sf8_tiled(row: usize, b: usize, kb: usize) -> usize {
    ((row >> 7) * (kb >> 1) + (b >> 1)) * 256 + (row & 127) * 2 + (b & 1)
}
