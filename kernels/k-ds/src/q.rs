//! Scalar helpers: exact FP8 E4M3 / FP4 E2M1 rounding in f32 (RNE, as torch's casts), power-of-two scales.
use core::arch::asm;

/// Round to the nearest integer, ties to even (`cvt.rni`).
#[inline(always)]
pub fn rni(x: f32) -> f32 {
    let r: f32;
    unsafe { asm!("cvt.rni.f32.f32 {r}, {x};", r = out(reg32) r, x = in(reg32) x, options(pure, nomem, nostack)) }
    r
}

/// 2^e for a normal exponent.
#[inline(always)]
pub fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// 2 ** ceil(log2(v)) of a positive normal f32 (DeepSeek's bit trick: exponent + (mantissa != 0)).
#[inline(always)]
pub fn pow2_ceil(v: f32) -> f32 {
    let b = v.to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127 + ((b & 0x7f_ffff) != 0) as i32;
    pow2(e)
}

/// Round to the FP8 E4M3 (fn) grid, ties to even; |x| <= 448 expected (callers clamp).
#[inline(always)]
pub fn e4m3(x: f32) -> f32 {
    let a = x.abs();
    let step = if a < 0.015625 {
        0.001953125 // subnormal spacing 2^-9
    } else {
        let e = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
        pow2(e - 3)
    };
    rni(x / step) * step
}

/// Round |v| <= 6 to the FP4 E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the even code.
#[inline(always)]
pub fn e2m1(v: f32) -> f32 {
    // midpoints 0.25 .75 1.25 1.75 2.5 3.5 5; a tie goes to the even code (0, 1, 2, 4)
    let a = v.abs();
    let g = if a <= 0.25 {
        0.0
    } else if a < 0.75 {
        0.5
    } else if a <= 1.25 {
        1.0
    } else if a < 1.75 {
        1.5
    } else if a <= 2.5 {
        2.0
    } else if a < 3.5 {
        3.0
    } else if a <= 5.0 {
        4.0
    } else {
        6.0
    };
    if v < 0.0 { -g } else { g }
}

/// FP8 E4M3 quant-dequant of one block (amax over the block): returns the scale.
#[inline(always)]
pub fn fp8_scale(amax: f32) -> f32 {
    pow2_ceil(amax.max(1e-4) / 448.0)
}

/// FP4 scale of a block: E4M3-rounded (compressed KV) or power-of-two (indexer).
#[inline(always)]
pub fn fp4_scale(amax: f32, e4m3_scale: bool) -> f32 {
    if e4m3_scale {
        e4m3(amax.max(6.0 * 0.001953125) / 6.0)
    } else {
        pow2_ceil(amax.max(6.0 * 1.1754944e-38) / 6.0)
    }
}

#[inline(always)]
pub fn fp4_qd(v: f32, s: f32) -> f32 {
    e2m1((v / s).max(-6.0).min(6.0)) * s
}

#[inline(always)]
pub fn fp8_qd(v: f32, s: f32) -> f32 {
    e4m3((v / s).max(-448.0).min(448.0)) * s
}

/// fp16 bits -> f32.
#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let r: f32;
    unsafe { asm!("cvt.f32.f16 {r}, {h};", r = out(reg32) r, h = in(reg16) h, options(pure, nomem, nostack)) }
    r
}
