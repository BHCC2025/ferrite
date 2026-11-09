//! bf16 is carried as raw `u16` bits (Rust has no device bf16 type); pairs as `u32` (`lo` = element 0).
use core::arch::asm;

#[inline(always)]
pub fn to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
/// Round-to-nearest-even (`cvt.rn.bf16.f32`).
#[inline(always)]
pub fn from_f32(f: f32) -> u16 {
    let h: u16;
    unsafe { asm!("cvt.rn.bf16.f32 {h}, {f};", h = out(reg16) h, f = in(reg32) f, options(pure, nomem, nostack)) }
    h
}
/// Pack two floats into a bf16x2 word: `lo` in bits 0..16 (element 0), `hi` in bits 16..32.
#[inline(always)]
pub fn pack2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe {
        asm!("cvt.rn.bf16x2.f32 {r}, {hi}, {lo};", r = out(reg32) r, hi = in(reg32) hi, lo = in(reg32) lo,
            options(pure, nomem, nostack))
    }
    r
}
#[inline(always)]
pub fn lo(w: u32) -> f32 {
    f32::from_bits(w << 16)
}
#[inline(always)]
pub fn hi(w: u32) -> f32 {
    f32::from_bits(w & 0xffff_0000)
}
#[inline(always)]
pub fn unpack2(w: u32) -> (f32, f32) {
    (lo(w), hi(w))
}
/// Round a float through bf16 and back (models a bf16 store/load).
#[inline(always)]
pub fn round(f: f32) -> f32 {
    to_f32(from_f32(f))
}
/// Unpack 8 bf16 from a 128-bit load.
#[inline(always)]
pub fn unpack8(v: [u32; 4]) -> [f32; 8] {
    [lo(v[0]), hi(v[0]), lo(v[1]), hi(v[1]), lo(v[2]), hi(v[2]), lo(v[3]), hi(v[3])]
}
