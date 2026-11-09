//! EXL3-specific device helpers kcommon does not have: fp16 conversions and fma, dp4a, funnel shifts,
//! the mul1 codebook decode and the 128-point Walsh-Hadamard transform across a warp (both after exllamav3, MIT;
//! see NOTICE).

use core::arch::asm;

#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let r: f32;
    unsafe { asm!("cvt.f32.f16 {0}, {1};", out(reg32) r, in(reg16) h, options(pure, nomem, nostack)) };
    r
}

/// Two fp16 halves of a u32 (low half first) to f32.
#[inline(always)]
pub fn h2_to_f32(h: u32) -> (f32, f32) {
    (f16_to_f32(h as u16), f16_to_f32((h >> 16) as u16))
}

/// (lo, hi) f32 to a packed fp16x2 (lo in the low half), round to nearest even.
#[inline(always)]
pub fn f32x2_to_h2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe { asm!("cvt.rn.f16x2.f32 {0}, {1}, {2};", out(reg32) r, in(reg32) hi, in(reg32) lo, options(pure, nomem, nostack)) };
    r
}

/// (lo, hi) f32 to a packed bf16x2 (lo in the low half), round to nearest even.
#[inline(always)]
pub fn f32x2_to_bf2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe { asm!("cvt.rn.bf16x2.f32 {0}, {1}, {2};", out(reg32) r, in(reg32) hi, in(reg32) lo, options(pure, nomem, nostack)) };
    r
}

/// fp16x2 fma, one rounding per half (fma.rn.f16x2).
#[inline(always)]
pub fn hfma2(a: u32, b: u32, c: u32) -> u32 {
    let r: u32;
    unsafe { asm!("fma.rn.f16x2 {0}, {1}, {2}, {3};", out(reg32) r, in(reg32) a, in(reg32) b, in(reg32) c, options(pure, nomem, nostack)) };
    r
}

/// Unsigned byte dot product: sum of the 4 bytes of a times those of b, plus c.
#[inline(always)]
pub fn dp4a_u(a: u32, b: u32, c: u32) -> u32 {
    let r: u32;
    unsafe { asm!("dp4a.u32.u32 {0}, {1}, {2}, {3};", out(reg32) r, in(reg32) a, in(reg32) b, in(reg32) c, options(pure, nomem, nostack)) };
    r
}

/// Byte permute: bytes of {b:a} picked by the selector's nibbles.
#[inline(always)]
pub fn prmt(a: u32, b: u32, sel: u32) -> u32 {
    let r: u32;
    unsafe { asm!("prmt.b32 {0}, {1}, {2}, {3};", out(reg32) r, in(reg32) a, in(reg32) b, in(reg32) sel, options(pure, nomem, nostack)) };
    r
}

/// Low 32 bits of the 64-bit hi:lo shifted right by `s` (0..32).
#[inline(always)]
pub fn funnel_r(lo: u32, hi: u32, s: u32) -> u32 {
    let r: u32;
    unsafe { asm!("shf.r.clamp.b32 {0}, {1}, {2}, {3};", out(reg32) r, in(reg32) lo, in(reg32) hi, in(reg32) s, options(pure, nomem, nostack)) };
    r
}

/// High 32 bits of hi:lo shifted left by `s` (0..32).
#[inline(always)]
pub fn funnel_l(lo: u32, hi: u32, s: u32) -> u32 {
    let r: u32;
    unsafe { asm!("shf.l.clamp.b32 {0}, {1}, {2}, {3};", out(reg32) r, in(reg32) lo, in(reg32) hi, in(reg32) s, options(pure, nomem, nostack)) };
    r
}

/// Two mul1 codebook values from two 16-bit states, as fp16x2 (s0 in the low half).
#[inline(always)]
pub fn decode2_mul1(s0: u32, s1: u32) -> u32 {
    let x0 = s0.wrapping_mul(0x83DC_D12D);
    let x1 = s1.wrapping_mul(0x83DC_D12D);
    let h0 = dp4a_u(x0, 0x0101_0101, 0x6400); // fp16 bits of 1024 + byte sum
    let h1 = dp4a_u(x1, 0x0101_0101, 0x6400);
    let hh = prmt(h0, h1, 0x5410); // low halves of h0 and h1 (one PRMT, not an IMAD)
    hfma2(hh, 0x1eee_1eee, 0xc931_c931) // * 1/147.7 - 10.39, fp16
}

/// Unscaled natural-order Walsh-Hadamard transform of 128 f32, 4 a lane (element 4 * lane + q).
#[inline(always)]
pub fn fwht128(v: &mut [f32; 4], lane: u32) {
    let (a, b, c, d) = (v[0] + v[1], v[0] - v[1], v[2] + v[3], v[2] - v[3]);
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
    let mut m = 1u32;
    while m < 32 {
        for x in v.iter_mut() {
            let o = kcommon::warp::shfl_xor(*x, m);
            *x = if lane & m != 0 { o - *x } else { *x + o };
        }
        m <<= 1;
    }
}

pub const HAD_SCALE: f32 = 0.088_388_347_648_318_44; // 1 / sqrt(128)

/// Read-only, L1-no-allocate global load of a weight-stream word (read once per call).
#[inline(always)]
pub unsafe fn ldg_stream(p: *const u32) -> u32 {
    let r: u32;
    asm!("ld.global.nc.L1::no_allocate.u32 {0}, [{1}];", out(reg32) r, in(reg64) p, options(nostack, readonly, pure));
    r
}

/// Read-only global load through the non-coherent cache.
#[inline(always)]
pub unsafe fn ldg_u32(p: *const u32) -> u32 {
    let r: u32;
    asm!("ld.global.nc.u32 {0}, [{1}];", out(reg32) r, in(reg64) p, options(nostack, readonly, pure));
    r
}
