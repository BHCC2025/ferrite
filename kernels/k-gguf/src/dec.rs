//! Block decoders. `decN_<ty>(blk, c)` returns values `8c .. 8c+8` of the block at `blk` as f32, with exactly the
//! arithmetic (and operation order) of ggml's `dequantize_row_*`, so the result is bit-identical to the host
//! reference (crates/fe-ggq/src/refq.rs). Block bases are only 2-byte aligned for most types (82/98/110/210-byte
//! blocks), so reads are u8/u16.
use crate::tables::*;
use core::arch::asm;

#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let r: f32;
    unsafe { asm!("cvt.f32.f16 {r}, {h};", r = out(reg32) r, h = in(reg16) h, options(pure, nomem, nostack)) }
    r
}
#[inline(always)]
pub unsafe fn ld_u8(p: *const u8) -> u8 {
    *p
}
#[inline(always)]
pub unsafe fn ld_u16(p: *const u8) -> u16 {
    *(p as *const u16)
}
#[inline(always)]
pub unsafe fn ld_h(p: *const u8) -> f32 {
    f16_to_f32(ld_u16(p))
}
/// u32 from a 2-byte aligned address.
#[inline(always)]
pub unsafe fn ld_u32_a2(p: *const u8) -> u32 {
    (ld_u16(p) as u32) | ((ld_u16(p.add(2)) as u32) << 16)
}
#[inline(always)]
pub unsafe fn ld_u64_a2(p: *const u8) -> u64 {
    (ld_u32_a2(p) as u64) | ((ld_u32_a2(p.add(4)) as u64) << 32)
}
#[inline(always)]
fn sgn(s: u32, j: u32) -> f32 {
    if s & (1 << j) != 0 { -1.0 } else { 1.0 }
}
#[inline(always)]
fn mul(a: f32, b: f32) -> f32 {
    // no contraction: plain mul.rn
    let r: f32;
    unsafe { asm!("mul.rn.f32 {r}, {a}, {b};", r = out(reg32) r, a = in(reg32) a, b = in(reg32) b, options(pure, nomem, nostack)) }
    r
}
#[inline(always)]
fn sub(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe { asm!("sub.rn.f32 {r}, {a}, {b};", r = out(reg32) r, a = in(reg32) a, b = in(reg32) b, options(pure, nomem, nostack)) }
    r
}
#[inline(always)]
fn add(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe { asm!("add.rn.f32 {r}, {a}, {b};", r = out(reg32) r, a = in(reg32) a, b = in(reg32) b, options(pure, nomem, nostack)) }
    r
}

#[inline(always)]
pub unsafe fn dec_q8_0(b: *const u8, c: u32) -> [f32; 8] {
    let d = ld_h(b);
    let mut y = [0f32; 8];
    for j in 0..8 {
        y[j] = mul(d, (ld_u8(b.add(2 + 8 * c as usize + j)) as i8) as f32);
    }
    y
}
#[inline(always)]
pub unsafe fn dec_q2_0(b: *const u8, c: u32) -> [f32; 8] {
    let d = ld_h(b);
    let w = ld_u16(b.add(2 + 2 * c as usize)) as u32;
    let mut y = [0f32; 8];
    for j in 0..8 {
        y[j] = mul((((w >> (2 * j)) & 3) as i32 - 1) as f32, d);
    }
    y
}
#[inline(always)]
pub unsafe fn kv4(i: u32) -> f32 {
    (*KVALUES_IQ4NL.as_ptr().add(i as usize)) as f32
}
/// 8 nibbles (low or high) of 8 consecutive qs bytes.
#[inline(always)]
unsafe fn nib8(qs: *const u8, hi: bool, d: f32) -> [f32; 8] {
    let lo = ld_u32_a2(qs);
    let h2 = ld_u32_a2(qs.add(4));
    let mut y = [0f32; 8];
    for j in 0..8 {
        let byte = if j < 4 { lo >> (8 * j) } else { h2 >> (8 * (j - 4)) } & 0xff;
        let q = if hi { byte >> 4 } else { byte & 0xf };
        y[j] = mul(d, kv4(q));
    }
    y
}
#[inline(always)]
pub unsafe fn dec_iq4_nl(b: *const u8, c: u32) -> [f32; 8] {
    let d = ld_h(b);
    nib8(b.add(2 + 8 * (c as usize & 1)), c >= 2, d)
}
#[inline(always)]
pub unsafe fn dec_iq4_xs(b: *const u8, c: u32) -> [f32; 8] {
    let d = ld_h(b);
    let ib = (c / 4) as usize;
    let sh = ld_u16(b.add(2)) as u32;
    let sl = ld_u8(b.add(4 + ib / 2)) as u32;
    let ls = ((sl >> (4 * (ib as u32 % 2))) & 0xf) | (((sh >> (2 * ib as u32)) & 3) << 4);
    let dl = mul(d, (ls as i32 - 32) as f32);
    let q = c % 4;
    nib8(b.add(8 + ib * 16 + 8 * (q as usize & 1)), q >= 2, dl)
}
#[inline(always)]
unsafe fn scale_min_k4(j: usize, q: *const u8) -> (u32, u32) {
    if j < 4 {
        ((ld_u8(q.add(j)) & 63) as u32, (ld_u8(q.add(j + 4)) & 63) as u32)
    } else {
        let a = ld_u8(q.add(j + 4)) as u32;
        ((a & 0xf) | ((ld_u8(q.add(j - 4)) as u32 >> 6) << 4), (a >> 4) | ((ld_u8(q.add(j)) as u32 >> 6) << 4))
    }
}
#[inline(always)]
pub unsafe fn dec_q4_k(b: *const u8, c: u32) -> [f32; 8] {
    let d = ld_h(b);
    let mn = ld_h(b.add(2));
    let y0 = c * 8;
    let jj = (y0 / 64) as usize;
    let half = ((y0 % 64) / 32) as usize;
    let l = (y0 % 32) as usize;
    let (s, m) = scale_min_k4(2 * jj + half, b.add(4));
    let (dd, mm) = (mul(d, s as f32), mul(mn, m as f32));
    let q = b.add(16 + jj * 32 + l);
    let mut y = [0f32; 8];
    for j in 0..8 {
        let v = ld_u8(q.add(j)) as u32;
        let v = if half == 1 { v >> 4 } else { v & 0xf };
        y[j] = sub(mul(dd, v as f32), mm);
    }
    y
}
#[inline(always)]
pub unsafe fn dec_q5_k(b: *const u8, c: u32) -> [f32; 8] {
    let d = ld_h(b);
    let mn = ld_h(b.add(2));
    let y0 = c * 8;
    let jj = (y0 / 64) as usize;
    let half = ((y0 % 64) / 32) as usize;
    let l = (y0 % 32) as usize;
    let (s, m) = scale_min_k4(2 * jj + half, b.add(4));
    let (dd, mm) = (mul(d, s as f32), mul(mn, m as f32));
    let u = 1u32 << (2 * jj + half);
    let ql = b.add(48 + jj * 32 + l);
    let qh = b.add(16 + l);
    let mut y = [0f32; 8];
    for j in 0..8 {
        let v = ld_u8(ql.add(j)) as u32;
        let v = if half == 1 { v >> 4 } else { v & 0xf };
        let hb = if (ld_u8(qh.add(j)) as u32) & u != 0 { 16 } else { 0 };
        y[j] = sub(mul(dd, (v + hb) as f32), mm);
    }
    y
}
#[inline(always)]
pub unsafe fn dec_q6_k(b: *const u8, c: u32) -> [f32; 8] {
    let d = ld_h(b.add(208));
    let y0 = c * 8;
    let n = (y0 / 128) as usize;
    let r = y0 % 128;
    let qd = (r / 32) as usize;
    let l = (r % 32) as usize;
    let ql = b.add(n * 64 + l + if qd & 1 == 1 { 32 } else { 0 });
    let qh = b.add(128 + n * 32 + l);
    let sc = (ld_u8(b.add(192 + n * 8 + l / 16 + 2 * qd)) as i8) as f32;
    let ds = mul(d, sc);
    let mut y = [0f32; 8];
    for j in 0..8 {
        let lo = ld_u8(ql.add(j)) as u32;
        let lo = if qd >= 2 { lo >> 4 } else { lo & 0xf };
        let hb = ((ld_u8(qh.add(j)) as u32) >> (2 * qd)) & 3;
        y[j] = mul(ds, (((lo | (hb << 4)) as i32) - 32) as f32);
    }
    y
}
#[inline(always)]
pub unsafe fn dec_iq2_s(b: *const u8, c: u32) -> [f32; 8] {
    let d = ld_h(b);
    let ib = (c / 4) as usize;
    let l = (c % 4) as usize;
    let sc = ld_u8(b.add(74 + ib)) as u32;
    let s4 = if l < 2 { sc & 0xf } else { sc >> 4 };
    let dl = mul(mul(d, add(0.5, s4 as f32)), 0.25);
    let qh = ld_u8(b.add(66 + ib)) as usize;
    let idx = ld_u8(b.add(2 + ib * 4 + l)) as usize | ((qh << (8 - 2 * l)) & 0x300);
    let g = *IQ2S_GRID.as_ptr().add(idx);
    let s = ld_u8(b.add(34 + ib * 4 + l)) as u32;
    let mut y = [0f32; 8];
    for j in 0..8 {
        y[j] = mul(mul(dl, ((g >> (8 * j)) & 0xff) as f32), sgn(s, j as u32));
    }
    y
}
#[inline(always)]
pub unsafe fn dec_iq3_xxs(b: *const u8, c: u32) -> [f32; 8] {
    let d = ld_h(b);
    let ib = (c / 4) as usize;
    let l = (c % 4) as usize;
    let aux = ld_u32_a2(b.add(66 + 4 * ib));
    let db = mul(mul(d, add(0.5, (aux >> 28) as f32)), 0.5);
    let signs = *KSIGNS_IQ2XS.as_ptr().add(((aux >> (7 * l)) & 127) as usize) as u32;
    let g1 = *IQ3XXS_GRID.as_ptr().add(ld_u8(b.add(2 + ib * 8 + 2 * l)) as usize);
    let g2 = *IQ3XXS_GRID.as_ptr().add(ld_u8(b.add(2 + ib * 8 + 2 * l + 1)) as usize);
    let mut y = [0f32; 8];
    for j in 0..4 {
        y[j] = mul(mul(db, ((g1 >> (8 * j)) & 0xff) as f32), sgn(signs, j as u32));
        y[j + 4] = mul(mul(db, ((g2 >> (8 * j)) & 0xff) as f32), sgn(signs, j as u32 + 4));
    }
    y
}
#[inline(always)]
pub unsafe fn dec_iq3_s(b: *const u8, c: u32) -> [f32; 8] {
    let d = ld_h(b);
    let ib = (c / 4) as usize;
    let l = (c % 4) as usize;
    let sc = ld_u8(b.add(106 + ib / 2)) as u32;
    let s4 = if ib % 2 == 0 { sc & 0xf } else { sc >> 4 };
    let db = mul(d, (1 + 2 * s4) as f32);
    let qh = ld_u8(b.add(66 + ib)) as usize;
    let i1 = ld_u8(b.add(2 + ib * 8 + 2 * l)) as usize | ((qh << (8 - 2 * l)) & 256);
    let i2 = ld_u8(b.add(2 + ib * 8 + 2 * l + 1)) as usize | ((qh << (7 - 2 * l)) & 256);
    let g1 = *IQ3S_GRID.as_ptr().add(i1);
    let g2 = *IQ3S_GRID.as_ptr().add(i2);
    let s = ld_u8(b.add(74 + ib * 4 + l)) as u32;
    let mut y = [0f32; 8];
    for j in 0..4 {
        y[j] = mul(mul(db, ((g1 >> (8 * j)) & 0xff) as f32), sgn(s, j as u32));
        y[j + 4] = mul(mul(db, ((g2 >> (8 * j)) & 0xff) as f32), sgn(s, j as u32 + 4));
    }
    y
}
/// Plain 8-value chunks of f32 / f16 / bf16 tensors ("block" = 8 values).
#[inline(always)]
pub unsafe fn dec_f32(b: *const u8, c: u32) -> [f32; 8] {
    let p = (b as *const f32).add(8 * c as usize);
    let mut y = [0f32; 8];
    for j in 0..8 {
        y[j] = *p.add(j);
    }
    y
}
#[inline(always)]
pub unsafe fn dec_f16(b: *const u8, c: u32) -> [f32; 8] {
    let p = (b as *const u16).add(8 * c as usize);
    let mut y = [0f32; 8];
    for j in 0..8 {
        y[j] = f16_to_f32(*p.add(j));
    }
    y
}
#[inline(always)]
pub unsafe fn dec_bf16(b: *const u8, c: u32) -> [f32; 8] {
    let p = (b as *const u16).add(8 * c as usize);
    let mut y = [0f32; 8];
    for j in 0..8 {
        y[j] = f32::from_bits((*p.add(j) as u32) << 16);
    }
    y
}
