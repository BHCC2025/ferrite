//! Integer sub-block decoders for the dp4a paths (GEMV, grouped-expert GEMV). A "sub-block" is 32 consecutive
//! values of a row (the q8_1 activation block size). `Sb` holds them as 8 packed int8x4 words plus the float
//! scales: value[i] = base * ia * w[i] for i < 16, base * ib * w[i] for i >= 16 (integer half-block multipliers; both
//! 1 when the type has one scale per 32, `Vq::SAME`), minus `m` per unit of activation
//! (dot = d8 * base * (ia * dot(w[0..4], a) + ib * dot(w[4..8], a)) - m * s8, where s8 = the f16-rounded float sum of the
//! activation block).
use crate::dec::{f16_to_f32, ld_u16, ld_u32_a2, ld_u8};
use crate::tables::*;
use core::arch::asm;

pub struct Sb {
    pub w: [u32; 8],
    pub base: f32,
    pub ia: i32,
    pub ib: i32,
    pub m: f32,
}

/// The integer dot of a sub-block with 32 int8 activations (`a` = values 0..16, `c` = 16..32), half-block
/// multipliers applied, as f32 (multiply by d8 * base).
#[inline(always)]
pub fn sb_dot<V: Vq>(s: &Sb, a: &[u32; 4], c: &[u32; 4]) -> f32 {
    if V::SAME {
        let t = dp4a(s.w[3], a[3], dp4a(s.w[2], a[2], dp4a(s.w[1], a[1], dp4a(s.w[0], a[0], 0))));
        (dp4a(s.w[7], c[3], dp4a(s.w[6], c[2], dp4a(s.w[5], c[1], dp4a(s.w[4], c[0], t))))) as f32
    } else {
        let lo = dp4a(s.w[3], a[3], dp4a(s.w[2], a[2], dp4a(s.w[1], a[1], dp4a(s.w[0], a[0], 0))));
        let hi = dp4a(s.w[7], c[3], dp4a(s.w[6], c[2], dp4a(s.w[5], c[1], dp4a(s.w[4], c[0], 0))));
        (lo * s.ia + hi * s.ib) as f32
    }
}

#[inline(always)]
pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
    let r: i32;
    unsafe {
        asm!("dp4a.s32.s32 {r}, {a}, {b}, {c};", r = out(reg32) r, a = in(reg32) a, b = in(reg32) b, c = in(reg32) c,
            options(pure, nomem, nostack))
    }
    r
}
#[inline(always)]
pub fn prmt(a: u32, b: u32, s: u32) -> u32 {
    let r: u32;
    unsafe {
        asm!("prmt.b32 {r}, {a}, {b}, {s};", r = out(reg32) r, a = in(reg32) a, b = in(reg32) b, s = in(reg32) s,
            options(pure, nomem, nostack))
    }
    r
}
#[inline(always)]
pub fn popc(a: u32) -> u32 {
    let r: u32;
    unsafe { asm!("popc.b32 {r}, {a};", r = out(reg32) r, a = in(reg32) a, options(pure, nomem, nostack)) }
    r
}
/// Negate the bytes of `g` (all bytes nonzero and < 128) whose bit is set in the low 4 bits of `s`.
#[inline(always)]
pub fn sign4(g: u32, s: u32) -> u32 {
    let ones = ((s & 0xf).wrapping_mul(0x0020_4081)) & 0x0101_0101;
    let mask = ones.wrapping_mul(0xff);
    (g ^ mask).wrapping_add(ones)
}
/// Per-byte `v - k` for bytes v in 0..128, k <= 127, as int8 (no borrow between bytes).
#[inline(always)]
pub fn sub_bytes(v: u32, k4: u32) -> u32 {
    ((v | 0x8080_8080).wrapping_sub(k4)) ^ 0x8080_8080
}
#[inline(always)]
unsafe fn ldh(p: *const u8) -> f32 {
    f16_to_f32(ld_u16(p))
}
#[inline(always)]
unsafe fn ld_u64_a8(p: *const u8) -> u64 {
    *(p as *const u64)
}

/// IQ4_NL/IQ4_XS: 8 nibbles of `q` (bytes b0..b3, lo/hi) -> (lo-nibble values, hi-nibble values) as int8x4.
#[inline(always)]
pub unsafe fn iq4_lut(q: u32) -> (u32, u32) {
    let t = KVALUES_IQ4NL.as_ptr() as *const u32;
    let (t0, t1, t2, t3) = (*t, *t.add(1), *t.add(2), *t.add(3));
    let sel = 0x3210_3210 | ((q & 0x8888_8888) >> 1);
    let q7 = q & 0x7777_7777;
    let lo0 = prmt(t0, t1, q7);
    let hi0 = prmt(t2, t3, q7);
    let a = prmt(lo0, hi0, sel & 0xffff);
    let lo1 = prmt(t0, t1, q7 >> 16);
    let hi1 = prmt(t2, t3, q7 >> 16);
    let b = prmt(lo1, hi1, (sel >> 16) & 0xffff);
    (prmt(a, b, 0x6420), prmt(a, b, 0x7531))
}

pub trait Vq {
    /// Values per quant block and bytes per block.
    const QK: u32;
    const BYTES: u32;
    /// One scale for all 32 values (ia = ib = 1): one dp4a chain.
    const SAME: bool = true;
    /// Whether `m` (a min term, Q4_K/Q5_K) is used.
    const HAS_M: bool = false;
    /// Decode sub-block `j` (32 values) of the row starting at `row`.
    unsafe fn sb(row: *const u8, j: u32) -> Sb;
}

pub struct Q8_0;
impl Vq for Q8_0 {
    const QK: u32 = 32;
    const BYTES: u32 = 34;
    #[inline(always)]
    unsafe fn sb(row: *const u8, j: u32) -> Sb {
        let b = row.add(34 * j as usize);
        let d = ldh(b);
        let mut w = [0u32; 8];
        for k in 0..8 {
            w[k] = ld_u32_a2(b.add(2 + 4 * k));
        }
        Sb { w, base: d, ia: 1, ib: 1, m: 0.0 }
    }
}
pub struct Q2_0;
impl Vq for Q2_0 {
    const QK: u32 = 64;
    const BYTES: u32 = 18;
    #[inline(always)]
    unsafe fn sb(row: *const u8, j: u32) -> Sb {
        let b = row.add(18 * (j / 2) as usize);
        let d = ldh(b);
        let q = b.add(2 + 8 * (j & 1) as usize);
        let lo = ld_u32_a2(q);
        let hi = ld_u32_a2(q.add(4));
        let mut w = [0u32; 8];
        for k in 0..8 {
            let byte = if k < 4 { lo >> (8 * k) } else { hi >> (8 * (k - 4)) } & 0xff;
            let t = (byte | (byte << 6) | (byte << 12) | (byte << 18)) & 0x0303_0303;
            w[k] = sub_bytes(t, 0x0101_0101);
        }
        Sb { w, base: d, ia: 1, ib: 1, m: 0.0 }
    }
}
pub struct Iq4Nl;
impl Vq for Iq4Nl {
    const QK: u32 = 32;
    const BYTES: u32 = 18;
    #[inline(always)]
    unsafe fn sb(row: *const u8, j: u32) -> Sb {
        let b = row.add(18 * j as usize);
        let d = ldh(b);
        let mut w = [0u32; 8];
        for k in 0..4 {
            let (lo, hi) = iq4_lut(ld_u32_a2(b.add(2 + 4 * k)));
            w[k] = lo;
            w[k + 4] = hi;
        }
        Sb { w, base: d, ia: 1, ib: 1, m: 0.0 }
    }
}
pub struct Iq4Xs;
impl Vq for Iq4Xs {
    const QK: u32 = 256;
    const BYTES: u32 = 136;
    #[inline(always)]
    unsafe fn sb(row: *const u8, j: u32) -> Sb {
        let b = row.add(136 * (j / 8) as usize);
        let ib = j % 8;
        let d = ldh(b);
        let sh = ld_u16(b.add(2)) as u32;
        let sl = ld_u8(b.add(4 + (ib / 2) as usize)) as u32;
        let ls = ((sl >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        let dl = d * (ls as i32 - 32) as f32;
        let q = b.add(8 + 16 * ib as usize);
        let v0 = ld_u64_a8(q);
        let v1 = ld_u64_a8(q.add(8));
        let qs = [v0 as u32, (v0 >> 32) as u32, v1 as u32, (v1 >> 32) as u32];
        let mut w = [0u32; 8];
        for k in 0..4 {
            let (lo, hi) = iq4_lut(qs[k]);
            w[k] = lo;
            w[k + 4] = hi;
        }
        Sb { w, base: dl, ia: 1, ib: 1, m: 0.0 }
    }
}
#[inline(always)]
unsafe fn k4_scale_min(sc: *const u8, j: u32) -> (u32, u32) {
    let j = j as usize;
    if j < 4 {
        ((ld_u8(sc.add(j)) & 63) as u32, (ld_u8(sc.add(j + 4)) & 63) as u32)
    } else {
        let a = ld_u8(sc.add(j + 4)) as u32;
        ((a & 0xf) | ((ld_u8(sc.add(j - 4)) as u32 >> 6) << 4), (a >> 4) | ((ld_u8(sc.add(j)) as u32 >> 6) << 4))
    }
}
pub struct Q4K;
impl Vq for Q4K {
    const QK: u32 = 256;
    const HAS_M: bool = true;
    const BYTES: u32 = 144;
    #[inline(always)]
    unsafe fn sb(row: *const u8, j: u32) -> Sb {
        let b = row.add(144 * (j / 8) as usize);
        let ib = j % 8;
        let d = ldh(b);
        let dmin = ldh(b.add(2));
        let (s, m) = k4_scale_min(b.add(4), ib);
        let q = b.add(16 + 32 * (ib / 2) as usize) as *const u32;
        let sh = 4 * (ib & 1);
        let mut w = [0u32; 8];
        for k in 0..8 {
            w[k] = (*q.add(k) >> sh) & 0x0f0f_0f0f;
        }
        let ds = d * s as f32;
        Sb { w, base: ds, ia: 1, ib: 1, m: dmin * m as f32 }
    }
}
pub struct Q5K;
impl Vq for Q5K {
    const QK: u32 = 256;
    const HAS_M: bool = true;
    const BYTES: u32 = 176;
    #[inline(always)]
    unsafe fn sb(row: *const u8, j: u32) -> Sb {
        let b = row.add(176 * (j / 8) as usize);
        let ib = j % 8;
        let d = ldh(b);
        let dmin = ldh(b.add(2));
        let (s, m) = k4_scale_min(b.add(4), ib);
        let ql = b.add(48 + 32 * (ib / 2) as usize) as *const u32;
        let qh = b.add(16) as *const u32;
        let sh = 4 * (ib & 1);
        let mut w = [0u32; 8];
        for k in 0..8 {
            w[k] = ((*ql.add(k) >> sh) & 0x0f0f_0f0f) | (((*qh.add(k) >> ib) & 0x0101_0101) << 4);
        }
        let ds = d * s as f32;
        Sb { w, base: ds, ia: 1, ib: 1, m: dmin * m as f32 }
    }
}
pub struct Q6K;
impl Vq for Q6K {
    const QK: u32 = 256;
    const SAME: bool = false;
    const BYTES: u32 = 210;
    #[inline(always)]
    unsafe fn sb(row: *const u8, j: u32) -> Sb {
        let b = row.add(210 * (j / 8) as usize);
        let ib = j % 8;
        let n = (ib / 4) as usize;
        let qd = ib % 4;
        let d = ldh(b.add(208));
        let ql = b.add(n * 64 + 32 * (qd & 1) as usize);
        let qh = b.add(128 + n * 32);
        let lsh = 4 * (qd >> 1);
        let hsh = 2 * qd;
        let mut w = [0u32; 8];
        for k in 0..8 {
            let lo = (ld_u32_a2(ql.add(4 * k)) >> lsh) & 0x0f0f_0f0f;
            let hi = ((ld_u32_a2(qh.add(4 * k)) >> hsh) & 0x0303_0303) << 4;
            w[k] = sub_bytes(lo | hi, 0x2020_2020);
        }
        let sc = b.add(192 + n * 8 + 2 * qd as usize);
        let s0 = (ld_u8(sc) as i8) as i32;
        let s1 = (ld_u8(sc.add(1)) as i8) as i32;
        Sb { w, base: d, ia: s0, ib: s1, m: 0.0 }
    }
}
pub struct Iq2S;
impl Vq for Iq2S {
    const QK: u32 = 256;
    const SAME: bool = false;
    const BYTES: u32 = 82;
    #[inline(always)]
    unsafe fn sb(row: *const u8, j: u32) -> Sb {
        let b = row.add(82 * (j / 8) as usize);
        let ib = (j % 8) as usize;
        let d = ldh(b);
        let qs = ld_u32_a2(b.add(2 + 4 * ib));
        let sg = ld_u32_a2(b.add(34 + 4 * ib));
        let qh = ld_u8(b.add(66 + ib)) as u32;
        let sc = ld_u8(b.add(74 + ib)) as u32;
        let grid = IQ2S_GRID.as_ptr();
        let mut w = [0u32; 8];
        for l in 0..4u32 {
            let idx = ((qs >> (8 * l)) & 0xff) | ((qh << (8 - 2 * l)) & 0x300);
            let s = sg >> (8 * l);
            let g = *grid.add(idx as usize); // one 64-bit load
            w[2 * l as usize] = sign4(g as u32, s);
            w[2 * l as usize + 1] = sign4((g >> 32) as u32, s >> 4);
        }
        // d * (0.5 + ls) * 0.25 = d / 8 * (2 ls + 1)
        Sb { w, base: d * 0.125, ia: (2 * (sc & 0xf) + 1) as i32, ib: (2 * (sc >> 4) + 1) as i32, m: 0.0 }
    }
}
pub struct Iq3Xxs;
impl Vq for Iq3Xxs {
    const QK: u32 = 256;
    const BYTES: u32 = 98;
    #[inline(always)]
    unsafe fn sb(row: *const u8, j: u32) -> Sb {
        let b = row.add(98 * (j / 8) as usize);
        let ib = (j % 8) as usize;
        let d = ldh(b);
        let q0 = ld_u32_a2(b.add(2 + 8 * ib));
        let q1 = ld_u32_a2(b.add(6 + 8 * ib));
        let aux = ld_u32_a2(b.add(66 + 4 * ib));
        let grid = IQ3XXS_GRID.as_ptr();
        let mut w = [0u32; 8];
        for l in 0..4u32 {
            let v = (aux >> (7 * l)) & 127;
            let s = v | ((popc(v) & 1) << 7);
            let qq = if l < 2 { q0 >> (16 * l) } else { q1 >> (16 * (l - 2)) };
            w[2 * l as usize] = sign4(*grid.add((qq & 0xff) as usize), s);
            w[2 * l as usize + 1] = sign4(*grid.add(((qq >> 8) & 0xff) as usize), s >> 4);
        }
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        Sb { w, base: db, ia: 1, ib: 1, m: 0.0 }
    }
}
pub struct Iq3S;
impl Vq for Iq3S {
    const QK: u32 = 256;
    const BYTES: u32 = 110;
    #[inline(always)]
    unsafe fn sb(row: *const u8, j: u32) -> Sb {
        let b = row.add(110 * (j / 8) as usize);
        let ib = (j % 8) as usize;
        let d = ldh(b);
        let q0 = ld_u32_a2(b.add(2 + 8 * ib));
        let q1 = ld_u32_a2(b.add(6 + 8 * ib));
        let qh = ld_u8(b.add(66 + ib)) as u32;
        let sg = ld_u32_a2(b.add(74 + 4 * ib));
        let sc = ld_u8(b.add(106 + ib / 2)) as u32;
        let ls = if ib % 2 == 0 { sc & 0xf } else { sc >> 4 };
        let grid = IQ3S_GRID.as_ptr();
        let mut w = [0u32; 8];
        for l in 0..4u32 {
            let qq = if l < 2 { q0 >> (16 * l) } else { q1 >> (16 * (l - 2)) };
            let i1 = (qq & 0xff) | ((qh << (8 - 2 * l)) & 256);
            let i2 = ((qq >> 8) & 0xff) | ((qh << (7 - 2 * l)) & 256);
            let s = sg >> (8 * l);
            w[2 * l as usize] = sign4(*grid.add(i1 as usize), s);
            w[2 * l as usize + 1] = sign4(*grid.add(i2 as usize), s >> 4);
        }
        let db = d * (1 + 2 * ls) as f32;
        Sb { w, base: db, ia: 1, ib: 1, m: 0.0 }
    }
}
