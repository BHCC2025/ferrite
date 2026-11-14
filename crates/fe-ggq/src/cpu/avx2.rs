//! AVX2 (+FMA) sub-block decoders and the B-row dot. Every decoder turns 32 values of a row into `Dq`:
//! unsigned magnitudes `wa` (u8, < 128 or the unsigned K-quant values), a sign source `sg` (activation bytes are
//! multiplied by sign(sg); unused when `SIGNED == false`), and float scales `sa` (values 0..16), `sb` (16..32)
//! plus a min coefficient `m` (Q4_K/Q5_K: value = scale * w - m). The dot of one sub-block with activation row b is
//! d8_b * (sa * sum(w[0..16] a[0..16]) + sb * sum(w[16..32] a[16..32])) - m * s8_b; `maddubs(wa, sign(a, sg))`
//! forms the products (|w| <= 127 and |a| <= 127: no int16 saturation).
#![allow(clippy::missing_safety_doc, unsafe_op_in_unsafe_fn)]
use crate::tables::*;
use std::arch::x86_64::*;

pub struct Dq {
    pub wa: __m256i,
    pub sg: __m256i,
    /// int16 multipliers of the maddubs pair sums: values 0..16 (low 128 bits) and 16..32 (high 128 bits)
    pub sc: __m256i,
    /// float factor of the sub-block (times the activation d8)
    pub base: f32,
    /// the integer scale when it is the same for both halves (`Cq::SAME`; folding it into `base` was tried and
    /// did not pay on the 285K)
    pub isc: i32,
    pub m: f32,
}
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn sc1() -> __m256i {
    _mm256_set1_epi16(1)
}
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn sc2(a: i32, b: i32) -> __m256i {
    _mm256_set_m128i(_mm_set1_epi16(b as i16), _mm_set1_epi16(a as i16))
}

#[inline(always)]
unsafe fn rd16(p: *const u8) -> u16 {
    (p as *const u16).read_unaligned()
}
#[inline(always)]
unsafe fn rd32(p: *const u8) -> u32 {
    (p as *const u32).read_unaligned()
}
#[inline(always)]
unsafe fn rd64(p: *const u8) -> u64 {
    (p as *const u64).read_unaligned()
}
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn h(p: *const u8) -> f32 {
    // F16C (on every AVX2 CPU): branch-free
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(rd16(p) as i32)))
}

/// 32 sign bits (bit j = value j negative) -> 0xff / 0x01 per byte (a `sign_epi8` source: -1 or +1).
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn sign_bits(s: u32) -> __m256i {
    let v = _mm256_set1_epi32(s as i32);
    // byte j <- byte j/8 of s
    let idx = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
    let b = _mm256_shuffle_epi8(v, idx);
    let bit = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
    let m = _mm256_cmpeq_epi8(_mm256_and_si256(b, bit), bit);
    _mm256_or_si256(m, _mm256_set1_epi8(1))
}

/// IQ codebook lookups as AVX2 gathers instead of insert chains (see `init_gather`).
#[inline]
pub fn gather() -> bool {
    GATHER.load(std::sync::atomic::Ordering::Relaxed)
}
pub static SIGNS_MEM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn signs_mem() -> bool {
    SIGNS_MEM.load(std::sync::atomic::Ordering::Relaxed)
}
pub static GATHER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Read FE_GGQ_CPU_GATHER once (0 = vector inserts); called by `cpu::has_avx2`.
pub fn init_gather() {
    // AVX2 gathers: fast on AMD Zen, slow on Intel cores with the GDS (Downfall) mitigation microcode (on the 285K
    // the insert chains win). Default: on for AMD, off otherwise; FE_GGQ_CPU_GATHER=0/1 overrides.
    let v = std::arch::x86_64::__cpuid(0);
    let is_amd = v.ebx == 0x6874_7541; // "Auth"enticAMD
    let on = std::env::var("FE_GGQ_CPU_GATHER").map(|v| v == "1").unwrap_or(is_amd);
    GATHER.store(on, std::sync::atomic::Ordering::Relaxed);
    let sm = std::env::var("FE_GGQ_CPU_SIGNMUL").map(|v| v == "1").unwrap_or(false);
    SIGNS_MEM.store(sm, std::sync::atomic::Ordering::Relaxed);
}

/// 8 codebook entries (u32) as one vector: a gather when `gather()`, else scalar loads + inserts.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn grid_vec32<const G: bool>(g: *const u32, idx: impl Fn(usize) -> usize) -> __m256i {
    if G {
        let iv = _mm256_setr_epi32(idx(0) as i32, idx(1) as i32, idx(2) as i32, idx(3) as i32, idx(4) as i32,
            idx(5) as i32, idx(6) as i32, idx(7) as i32);
        _mm256_i32gather_epi32::<4>(g as *const i32, iv)
    } else {
        _mm256_set_epi32(*g.add(idx(7)) as i32, *g.add(idx(6)) as i32, *g.add(idx(5)) as i32, *g.add(idx(4)) as i32,
            *g.add(idx(3)) as i32, *g.add(idx(2)) as i32, *g.add(idx(1)) as i32, *g.add(idx(0)) as i32)
    }
}

/// [`sign_bits`] of the u32 at `p`: broadcast straight from memory + variable shifts + a multiply (no p5 shuffles).
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn sign_bits_mem(p: *const u8) -> __m256i {
    let v = _mm256_set1_epi32(rd32(p) as i32);
    let x = _mm256_and_si256(_mm256_srlv_epi32(v, _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28)), _mm256_set1_epi32(0xf));
    let y = _mm256_and_si256(_mm256_mullo_epi32(x, _mm256_set1_epi32(0x0020_4081)), _mm256_set1_epi32(0x0101_0101));
    _mm256_or_si256(_mm256_cmpeq_epi8(y, _mm256_set1_epi32(0x0101_0101)), _mm256_set1_epi8(1))
}

pub trait Cq {
    const QK: usize;
    const BYTES: usize;
    /// Signed codes small enough (|w| <= 63) for maddubs(a + 128, w) without int16 saturation: from B >= 3 the
    /// kernels apply the signs to the weights once and use offset activations (one op fewer per row).
    const OFFSET_OK: bool = false;
    /// One integer scale for the whole sub-block (`Dq::isc`); false for IQ2_S and Q6_K (two halves).
    const SAME: bool = true;
    /// false: `wa` holds the values themselves (unsigned 0..31), no sign step.
    const SIGNED: bool = true;
    const HAS_M: bool = false;
    /// The GPU's integer sub-block form (k-gguf vq.rs) leaves this type's integer sub-block scale out of the integer
    /// sum and folds it into the float base: [`row_dot_gpu`] then sums with unit scales and uses [`Cq::gpu_base`].
    const GPU_ISC_IN_BASE: bool = false;
    /// The GPU's float base from this decoder's (base, isc), with the GPU's exact operations.
    #[inline(always)]
    fn gpu_base(base: f32, isc: i32) -> f32 {
        base * isc as f32
    }
    unsafe fn dq(row: *const u8, j: usize) -> Dq;
    /// Sub-blocks j .. j + 4 (j % 4 == 0): superblock types override it to read the block header once.
    #[inline(always)]
    unsafe fn dq4(row: *const u8, j: usize) -> [Dq; 4] {
        [Self::dq(row, j), Self::dq(row, j + 1), Self::dq(row, j + 2), Self::dq(row, j + 3)]
    }
}

#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn signed(w: __m256i) -> (__m256i, __m256i) {
    (_mm256_sign_epi8(w, w), w)
}

pub struct Q8_0;
impl Cq for Q8_0 {
    const QK: usize = 32;
    const BYTES: usize = 34;
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq(row: *const u8, j: usize) -> Dq {
        let b = row.add(34 * j);
        let d = h(b);
        let (wa, sg) = signed(_mm256_loadu_si256(b.add(2) as *const __m256i));
        Dq { wa, sg, sc: sc1(), base: d, isc: 1, m: 0.0 }
    }
}
pub struct Q2_0;
impl Cq for Q2_0 {
    const QK: usize = 64;
    const BYTES: usize = 18;
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq(row: *const u8, j: usize) -> Dq {
        let b = row.add(18 * (j / 2));
        let d = h(b);
        let q = rd64(b.add(2 + 8 * (j & 1)));
        let v = _mm256_set1_epi64x(q as i64);
        // byte j <- source byte j/4
        let idx = _mm256_set_epi64x(0x0707070706060606, 0x0505050504040404, 0x0303030302020202, 0x0101010100000000);
        let v = _mm256_shuffle_epi8(v, idx);
        let hv = _mm256_and_si256(_mm256_srli_epi16(v, 4), _mm256_set1_epi8(0x0f));
        let m1 = _mm256_set1_epi32(0x0000_0c03);
        let m2 = _mm256_set1_epi32(0x0c03_0000);
        let x = _mm256_or_si256(_mm256_and_si256(v, m1), _mm256_and_si256(hv, m2));
        // crumb value (masked, possibly <<2) -> c - 1
        let lut = _mm256_setr_epi8(-1, 0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, -1, 0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0,
            2, 0, 0, 0);
        let (wa, sg) = signed(_mm256_shuffle_epi8(lut, x));
        Dq { wa, sg, sc: sc1(), base: d, isc: 1, m: 0.0 }
    }
}
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn iq4_lut() -> __m256i {
    let t = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    _mm256_set_m128i(t, t)
}
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn iq4_vals(qs: *const u8) -> __m256i {
    let q = _mm_loadu_si128(qs as *const __m128i);
    let lo = _mm_and_si128(q, _mm_set1_epi8(0x0f));
    let hi = _mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0x0f));
    _mm256_shuffle_epi8(iq4_lut(), _mm256_set_m128i(hi, lo))
}
pub struct Iq4Nl;
impl Cq for Iq4Nl {
    const QK: usize = 32;
    const BYTES: usize = 18;
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq(row: *const u8, j: usize) -> Dq {
        let b = row.add(18 * j);
        let d = h(b);
        let (wa, sg) = signed(iq4_vals(b.add(2)));
        Dq { wa, sg, sc: sc1(), base: d, isc: 1, m: 0.0 }
    }
}
pub struct Iq4Xs;
impl Cq for Iq4Xs {
    const GPU_ISC_IN_BASE: bool = true;
    const QK: usize = 256;
    const BYTES: usize = 136;
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq(row: *const u8, j: usize) -> Dq {
        let b = row.add(136 * (j / 8));
        let ib = j % 8;
        let d = h(b);
        let sh = rd16(b.add(2)) as u32;
        let sl = *b.add(4 + ib / 2) as u32;
        let ls = ((sl >> (4 * (ib as u32 % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        let (wa, sg) = signed(iq4_vals(b.add(8 + 16 * ib)));
        let l = ls as i32 - 32;
        Dq { wa, sg, sc: sc2(l, l), base: d, isc: l, m: 0.0 }
    }
}
#[inline(always)]
unsafe fn k4_scale_min(sc: *const u8, j: usize) -> (u32, u32) {
    if j < 4 {
        ((*sc.add(j) & 63) as u32, (*sc.add(j + 4) & 63) as u32)
    } else {
        let a = *sc.add(j + 4) as u32;
        ((a & 0xf) | ((*sc.add(j - 4) as u32 >> 6) << 4), (a >> 4) | ((*sc.add(j) as u32 >> 6) << 4))
    }
}
pub struct Q4K;
impl Cq for Q4K {
    const GPU_ISC_IN_BASE: bool = true;
    const QK: usize = 256;
    const BYTES: usize = 144;
    const SIGNED: bool = false;
    const HAS_M: bool = true;
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq(row: *const u8, j: usize) -> Dq {
        let b = row.add(144 * (j / 8));
        let ib = j % 8;
        let d = h(b);
        let dmin = h(b.add(2));
        let (s, m) = k4_scale_min(b.add(4), ib);
        let q = _mm256_loadu_si256(b.add(16 + 32 * (ib / 2)) as *const __m256i);
        let q = if ib & 1 == 1 { _mm256_srli_epi16(q, 4) } else { q };
        let wa = _mm256_and_si256(q, _mm256_set1_epi8(0x0f));
        Dq { wa, sg: wa, sc: sc2(s as i32, s as i32), base: d, isc: s as i32, m: dmin * m as f32 }
    }
}
pub struct Q5K;
impl Cq for Q5K {
    const GPU_ISC_IN_BASE: bool = true;
    const QK: usize = 256;
    const BYTES: usize = 176;
    const SIGNED: bool = false;
    const HAS_M: bool = true;
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq(row: *const u8, j: usize) -> Dq {
        let b = row.add(176 * (j / 8));
        let ib = j % 8;
        let d = h(b);
        let dmin = h(b.add(2));
        let (s, m) = k4_scale_min(b.add(4), ib);
        let q = _mm256_loadu_si256(b.add(48 + 32 * (ib / 2)) as *const __m256i);
        let q = if ib & 1 == 1 { _mm256_srli_epi16(q, 4) } else { q };
        let lo = _mm256_and_si256(q, _mm256_set1_epi8(0x0f));
        let qh = _mm256_loadu_si256(b.add(16) as *const __m256i);
        let hb = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(ib as i32)), _mm256_set1_epi8(1));
        let wa = _mm256_or_si256(lo, _mm256_slli_epi16(hb, 4));
        Dq { wa, sg: wa, sc: sc2(s as i32, s as i32), base: d, isc: s as i32, m: dmin * m as f32 }
    }
}
pub struct Q6K;
impl Cq for Q6K {
    const QK: usize = 256;
    const SAME: bool = false;
    const BYTES: usize = 210;
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq(row: *const u8, j: usize) -> Dq {
        let b = row.add(210 * (j / 8));
        let ib = j % 8;
        let n = ib / 4;
        let qd = ib % 4;
        let d = h(b.add(208));
        let ql = _mm256_loadu_si256(b.add(n * 64 + 32 * (qd & 1)) as *const __m256i);
        let ql = if qd >= 2 { _mm256_srli_epi16(ql, 4) } else { ql };
        let lo = _mm256_and_si256(ql, _mm256_set1_epi8(0x0f));
        let qh = _mm256_loadu_si256(b.add(128 + n * 32) as *const __m256i);
        let hb = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(2 * qd as i32)), _mm256_set1_epi8(3));
        let w = _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hb, 4)), _mm256_set1_epi8(32));
        let sc = b.add(192 + n * 8 + 2 * qd);
        let (wa, sg) = signed(w);
        Dq { wa, sg, sc: sc2((*sc as i8) as i32, (*sc.add(1) as i8) as i32), base: d, isc: 1, m: 0.0 }
    }
}
pub struct Iq2S<const G: bool = false>;
impl<const G: bool> Iq2S<G> {
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq_at(b: *const u8, d: f32, ib: usize) -> Dq {
        let qs = b.add(2 + 4 * ib);
        let qh = *b.add(66 + ib) as usize;
        let sc = *b.add(74 + ib) as u32;
        let g = IQ2S_GRID.as_ptr();
        let q4 = rd32(qs) as usize;
        let idx = [(q4 & 0xff) | ((qh << 8) & 0x300), ((q4 >> 8) & 0xff) | ((qh << 6) & 0x300),
            ((q4 >> 16) & 0xff) | ((qh << 4) & 0x300), (q4 >> 24) | ((qh << 2) & 0x300)];
        let wa = if G {
            _mm256_i32gather_epi64::<8>(g as *const i64, _mm_setr_epi32(idx[0] as i32, idx[1] as i32, idx[2] as i32,
                idx[3] as i32))
        } else {
            _mm256_set_epi64x(*g.add(idx[3]) as i64, *g.add(idx[2]) as i64, *g.add(idx[1]) as i64, *g.add(idx[0]) as i64)
        };
        let sg = sign_bits(rd32(b.add(34 + 4 * ib)));
        // d * (0.5 + ls) * 0.25 = d / 8 * (2 ls + 1)
        Dq { wa, sg, sc: sc2(2 * (sc & 0xf) as i32 + 1, 2 * (sc >> 4) as i32 + 1), base: d * 0.125, isc: 1, m: 0.0 }
    }
}
impl<const G: bool> Cq for Iq2S<G> {
    const QK: usize = 256;
    const SAME: bool = false;
    const OFFSET_OK: bool = true;
    const BYTES: usize = 82;
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq(row: *const u8, j: usize) -> Dq {
        let b = row.add(82 * (j / 8));
        Self::dq_at(b, h(b), j % 8)
    }
    #[inline(always)]
    unsafe fn dq4(row: *const u8, j: usize) -> [Dq; 4] {
        let b = row.add(82 * (j / 8));
        let d = h(b);
        let ib = j % 8;
        [Self::dq_at(b, d, ib), Self::dq_at(b, d, ib + 1), Self::dq_at(b, d, ib + 2), Self::dq_at(b, d, ib + 3)]
    }
}
pub struct Iq3Xxs<const G: bool = false>;
impl<const G: bool> Iq3Xxs<G> {
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq_at(b: *const u8, d: f32, ib: usize) -> Dq {
        let q = b.add(2 + 8 * ib);
        let aux = rd32(b.add(66 + 4 * ib));
        let g = IQ3XXS_GRID.as_ptr();
        let wa = grid_vec32::<G>(g, |l| *q.add(l) as usize);
        // ksigns without the table: 4 x 7 sign bits spread to bytes (pdep), bit 7 = parity of the byte's 7 bits
        let v = _pdep_u32(aux, 0x7f7f_7f7f);
        let mut p = v ^ (v >> 4);
        p ^= p >> 2;
        p ^= p >> 1;
        let s = v | ((p & 0x0101_0101) << 7);
        // d * (0.5 + ls) * 0.5 = d / 4 * (2 ls + 1)
        let l = 2 * (aux >> 28) as i32 + 1;
        Dq { wa, sg: sign_bits(s), sc: sc2(l, l), base: d * 0.25, isc: l, m: 0.0 }
    }
}
impl<const G: bool> Cq for Iq3Xxs<G> {
    const GPU_ISC_IN_BASE: bool = true;
    /// GPU: d * (0.5 + ls) * 0.5 with this decoder's base = d / 4, isc = 2 ls + 1
    #[inline(always)]
    fn gpu_base(base: f32, isc: i32) -> f32 {
        (base * 4.0) * (0.5 + ((isc - 1) / 2) as f32) * 0.5
    }
    const QK: usize = 256;
    const OFFSET_OK: bool = true;
    const BYTES: usize = 98;
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq(row: *const u8, j: usize) -> Dq {
        let b = row.add(98 * (j / 8));
        Self::dq_at(b, h(b), j % 8)
    }
    #[inline(always)]
    unsafe fn dq4(row: *const u8, j: usize) -> [Dq; 4] {
        let b = row.add(98 * (j / 8));
        let d = h(b);
        let ib = j % 8;
        [Self::dq_at(b, d, ib), Self::dq_at(b, d, ib + 1), Self::dq_at(b, d, ib + 2), Self::dq_at(b, d, ib + 3)]
    }
}
pub struct Iq3S<const G: bool = false>;
impl<const G: bool> Iq3S<G> {
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq_at(b: *const u8, d: f32, ib: usize) -> Dq {
        let q = b.add(2 + 8 * ib);
        let qh = *b.add(66 + ib) as usize;
        let sc = *b.add(106 + ib / 2) as u32;
        let ls = if ib % 2 == 0 { sc & 0xf } else { sc >> 4 };
        let g = IQ3S_GRID.as_ptr();
        let wa = grid_vec32::<G>(g, |l| *q.add(l) as usize | ((qh << (8 - l)) & 256));
        let l = (1 + 2 * ls) as i32;
        let sg = sign_bits(rd32(b.add(74 + 4 * ib)));
        Dq { wa, sg, sc: sc2(l, l), base: d, isc: l, m: 0.0 }
    }
}
impl<const G: bool> Cq for Iq3S<G> {
    const GPU_ISC_IN_BASE: bool = true;
    const QK: usize = 256;
    const OFFSET_OK: bool = true;
    const BYTES: usize = 110;
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c,bmi2")]
    unsafe fn dq(row: *const u8, j: usize) -> Dq {
        let b = row.add(110 * (j / 8));
        Self::dq_at(b, h(b), j % 8)
    }
    #[inline(always)]
    unsafe fn dq4(row: *const u8, j: usize) -> [Dq; 4] {
        let b = row.add(110 * (j / 8));
        let d = h(b);
        let ib = j % 8;
        [Self::dq_at(b, d, ib), Self::dq_at(b, d, ib + 1), Self::dq_at(b, d, ib + 2), Self::dq_at(b, d, ib + 3)]
    }
}

/// q8 activation rows for the CPU kernels: int8 `q` [rows][k], f32 `d` [rows][k/32], `s` = f16-rounded float sum
/// per block (see `refq::q8_block`).
#[derive(Clone, Default)]
pub struct Q8Rows {
    pub k: usize,
    pub rows: usize,
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    pub s: Vec<f32>,
    /// q + 128 as u8 (the offset form used by the IQ kernels at B >= 3)
    pub qu: Vec<u8>,
}
impl Q8Rows {
    /// Quantize `rows` rows of `k` floats (same arithmetic as the GPU `ggq_q8_1`).
    pub fn quantize(x: &[f32], rows: usize, k: usize) -> Q8Rows {
        let mut r = Q8Rows::default();
        r.set(x, rows, k);
        r
    }
    pub fn set(&mut self, x: &[f32], rows: usize, k: usize) {
        self.k = k;
        self.rows = rows;
        self.q.resize(rows * k, 0);
        self.d.resize(rows * k / 32, 0.0);
        self.s.resize(rows * k / 32, 0.0);
        if super::has_avx2() {
            self.qu.resize(rows * k, 0);
            for bi in 0..rows * k / 32 {
                let (d, sm) = unsafe {
                    q8_block_fast(x.as_ptr().add(bi * 32), self.q.as_mut_ptr().add(bi * 32), self.qu.as_mut_ptr().add(bi * 32))
                };
                self.d[bi] = d;
                self.s[bi] = sm;
            }
            return;
        }
        for (bi, xs) in x[..rows * k].chunks_exact(32).enumerate() {
            let (q, d, sm) = crate::refq::q8_block(xs);
            self.q[bi * 32..bi * 32 + 32].copy_from_slice(&q);
            self.d[bi] = d;
            self.s[bi] = sm;
        }
        self.fill_offset();
    }
    /// q8 rows from their parts (e.g. the GPU's `Q8Act` copied to the host): codes [rows][k], scales and sums
    /// [rows][k / 32]; the offset codes are derived.
    pub fn from_parts(k: usize, rows: usize, q: &[i8], d: &[f32], s: &[f32]) -> Q8Rows {
        let mut r = Q8Rows { k, rows, q: q[..rows * k].to_vec(), d: d[..rows * k / 32].to_vec(), s: s[..rows * k / 32].to_vec(),
            qu: vec![] };
        r.fill_offset();
        r
    }
    /// Copy row `src_row` of `src` (same k) into row `dst_row` of self (q, qu, d, s).
    pub fn copy_row(&mut self, dst_row: usize, src: &Q8Rows, src_row: usize) {
        let (k, nb) = (self.k, self.k / 32);
        self.q[dst_row * k..(dst_row + 1) * k].copy_from_slice(&src.q[src_row * k..(src_row + 1) * k]);
        self.qu[dst_row * k..(dst_row + 1) * k].copy_from_slice(&src.qu[src_row * k..(src_row + 1) * k]);
        self.d[dst_row * nb..(dst_row + 1) * nb].copy_from_slice(&src.d[src_row * nb..(src_row + 1) * nb]);
        self.s[dst_row * nb..(dst_row + 1) * nb].copy_from_slice(&src.s[src_row * nb..(src_row + 1) * nb]);
    }
    /// Size for `rows` rows of `k` (contents unspecified).
    pub fn shape(&mut self, rows: usize, k: usize) {
        self.k = k;
        self.rows = rows;
        self.q.resize(rows * k, 0);
        self.qu.resize(rows * k, 0);
        self.d.resize(rows * k / 32, 0.0);
        self.s.resize(rows * k / 32, 0.0);
    }
    /// Recompute `qu` from `q` (call after writing `q` directly).
    pub fn fill_offset(&mut self) {
        self.qu.resize(self.q.len(), 0);
        for (u, &q) in self.qu.iter_mut().zip(&self.q) {
            *u = (q as u8) ^ 0x80;
        }
    }
}

/// `y[b] = sum_j W[row] . a[b]` over the whole row (`nsb` sub-blocks) for B activation rows. Groups of 4
/// sub-blocks: integer sums (scales folded into the madd) reduced to one lane per sub-block, then one f32 multiply
/// by the activation scales and one FMA by the weight factors per group and row.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn row_dot<C: Cq, const B: usize>(row: *const u8, nsb: usize, a: &Q8Rows) -> [f32; B] {
    let mut acc = [_mm_setzero_ps(); B];
    let mut macc = [_mm_setzero_ps(); B];
    let k = a.k;
    let nb = k / 32;
    let mut j = 0;
    while j + 4 <= nsb {
        let [d0, d1, d2, d3] = C::dq4(row, j);
        let base = _mm_setr_ps(d0.base, d1.base, d2.base, d3.base);
        let mv = if C::HAS_M { _mm_setr_ps(d0.m, d1.m, d2.m, d3.m) } else { _mm_setzero_ps() };
        if C::OFFSET_OK && B >= OFFSET_MIN_B {
            // signed weights once; per row maddubs(a + 128, w) - 128 * sum(w) (the correction reduced once a group)
            let off = _mm256_set1_epi8(-128i8);
            let ws = [_mm256_sign_epi8(d0.wa, d0.sg), _mm256_sign_epi8(d1.wa, d1.sg), _mm256_sign_epi8(d2.wa, d2.sg),
                _mm256_sign_epi8(d3.wa, d3.sg)];
            let sc = [d0.sc, d1.sc, d2.sc, d3.sc];
            let cr = |i: usize| _mm256_madd_epi16(_mm256_maddubs_epi16(off, ws[i]), sc[i]);
            let ch = _mm256_hadd_epi32(_mm256_hadd_epi32(cr(0), cr(1)), _mm256_hadd_epi32(cr(2), cr(3)));
            let corr = _mm_add_epi32(_mm256_castsi256_si128(ch), _mm256_extracti128_si256(ch, 1));
            for b in 0..B {
                let ap = a.qu.as_ptr().add(b * k + 32 * j);
                let pp = |i: usize| {
                    let av = _mm256_loadu_si256(ap.add(32 * i) as *const __m256i);
                    _mm256_madd_epi16(_mm256_maddubs_epi16(av, ws[i]), sc[i])
                };
                let h = red4(pp(0), pp(1), pp(2), pp(3));
                let x = _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)), corr);
                let d8 = _mm_loadu_ps(a.d.as_ptr().add(b * nb + j));
                acc[b] = _mm_fmadd_ps(_mm_mul_ps(_mm_cvtepi32_ps(x), d8), base, acc[b]);
            }
            j += 4;
            continue;
        }
        for b in 0..B {
            let ap = a.q.as_ptr().add(b * k + 32 * j);
            macro_rules! p {
                ($d:ident, $o:expr) => {{
                    let av = _mm256_loadu_si256(ap.add($o) as *const __m256i);
                    let av = if C::SIGNED { _mm256_sign_epi8(av, $d.sg) } else { av };
                    _mm256_madd_epi16(_mm256_maddubs_epi16($d.wa, av), $d.sc)
                }};
            }
            let (p0, p1, p2, p3) = (p!(d0, 0), p!(d1, 32), p!(d2, 64), p!(d3, 96));
            let h = red4(p0, p1, p2, p3);
            let x = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
            let d8 = _mm_loadu_ps(a.d.as_ptr().add(b * nb + j));
            acc[b] = _mm_fmadd_ps(_mm_mul_ps(_mm_cvtepi32_ps(x), d8), base, acc[b]);
            if C::HAS_M {
                macc[b] = _mm_fmadd_ps(mv, _mm_loadu_ps(a.s.as_ptr().add(b * nb + j)), macc[b]);
            }
        }
        j += 4;
    }
    let mut out = [0f32; B];
    for b in 0..B {
        let s = _mm_sub_ps(acc[b], macc[b]);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        out[b] = _mm_cvtss_f32(s);
    }
    // tail sub-blocks (K % 128 != 0)
    while j < nsb {
        let dq = C::dq(row, j);
        for b in 0..B {
            let av = _mm256_loadu_si256(a.q.as_ptr().add(b * k + 32 * j) as *const __m256i);
            let av = if C::SIGNED { _mm256_sign_epi8(av, dq.sg) } else { av };
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(dq.wa, av), dq.sc);
            let x = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
            let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0x4e));
            let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0xb1));
            out[b] += _mm_cvtsi128_si32(x) as f32 * *a.d.get_unchecked(b * nb + j) * dq.base;
            if C::HAS_M {
                out[b] -= dq.m * *a.s.get_unchecked(b * nb + j);
            }
        }
        j += 1;
    }
    out
}

/// GEMV over a row range: y[b * ld + r] = W[r] . a[b] for r in rows.
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn gemv_rows<C: Cq, const B: usize>(w: *const u8, row_bytes: usize, rows: std::ops::Range<usize>,
    a: &Q8Rows, y: *mut f32, ld: usize) {
    let nsb = a.k / 32;
    for r in rows {
        let v = row_dot::<C, B>(w.add(r * row_bytes), nsb, a);
        for b in 0..B {
            *y.add(b * ld + r) = v[b];
        }
    }
}

/// [`crate::refq::q8_block`] with AVX2, bit-identical: q[j] = round-half-away(x[j] / d) with d = amax / 127 (an
/// IEEE division, as the scalar code), s = the same butterfly sum tree rounded to f16 (RNE) and back. Writes the
/// codes to `q` and the offset codes (q + 128) to `qu`; returns (d, s).
///
/// # Safety
/// `x` 32 floats, `q` / `qu` 32 bytes; AVX2 + F16C.
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn q8_block_fast(x: *const f32, q: *mut i8, qu: *mut u8) -> (f32, f32) {
    let v = [_mm256_loadu_ps(x), _mm256_loadu_ps(x.add(8)), _mm256_loadu_ps(x.add(16)), _mm256_loadu_ps(x.add(24))];
    let absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let m = _mm256_max_ps(_mm256_max_ps(_mm256_and_ps(v[0], absm), _mm256_and_ps(v[1], absm)),
        _mm256_max_ps(_mm256_and_ps(v[2], absm), _mm256_and_ps(v[3], absm)));
    let m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    let m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    let m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    let amax = _mm_cvtss_f32(m4);
    let d = amax / 127.0;
    let mut qi = [_mm256_setzero_si256(); 4];
    if amax != 0.0 {
        let dv = _mm256_set1_ps(d);
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        for i in 0..4 {
            let y = _mm256_div_ps(v[i], dv);
            let a = _mm256_and_ps(y, absm);
            let t = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(a);
            let up = _mm256_and_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(_mm256_sub_ps(a, t), half), one);
            let r = _mm256_or_ps(_mm256_add_ps(t, up), _mm256_andnot_ps(absm, y));
            qi[i] = _mm256_cvttps_epi32(r);
        }
    }
    // i32 x 32 -> i8 x 32 in order
    let p01 = _mm256_packs_epi32(qi[0], qi[1]);
    let p23 = _mm256_packs_epi32(qi[2], qi[3]);
    let p = _mm256_packs_epi16(p01, p23);
    let p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256(q as *mut __m256i, p);
    _mm256_storeu_si256(qu as *mut __m256i, _mm256_xor_si256(p, _mm256_set1_epi8(-128i8)));
    // sum: v[j] + v[j ^ 16], then ^8, ^4, ^2, ^1 (lane 0)
    let a16 = _mm256_add_ps(v[0], v[2]); // lanes 0..8
    let b16 = _mm256_add_ps(v[1], v[3]); // lanes 8..16
    let r8 = _mm256_add_ps(a16, b16);
    let r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
    let r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    let r1 = _mm_add_ss(r2, _mm_shuffle_ps(r2, r2, 1));
    let h = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(r1);
    let s = _mm_cvtss_f32(_mm_cvtph_ps(h));
    (d, s)
}

/// Row count from which the |w| <= 63 types use the offset form (signed weights once, a + 128 activations).
const OFFSET_MIN_B: usize = 3;

/// AVX-VNNI available (and not disabled with FE_GGQ_CPU_VNNI=0).
/// Runtime detection only (`is_x86_feature_detected!`): CPUs without AVX-VNNI (e.g. Zen 3) take the AVX2 kernels.
pub fn vnni() -> bool {
    match VNNI_STATE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = is_x86_feature_detected!("avxvnni") && std::env::var("FE_GGQ_CPU_VNNI").map_or(true, |v| v != "0");
            VNNI_STATE.store(if on { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
            on
        }
    }
}
static VNNI_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
/// Tests: force the VNNI kernels off (false) or back to runtime detection (true; never on without the CPU feature).
pub fn set_vnni(allow: bool) {
    let on = allow && is_x86_feature_detected!("avxvnni") && std::env::var("FE_GGQ_CPU_VNNI").map_or(true, |v| v != "0");
    VNNI_STATE.store(if on { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
}

/// [`row_dot`] for types with one float scale per 32 values and signed codes up to |w| <= 127 (IQ4_NL, Q8_0), with
/// AVX-VNNI: sum((a + 128) * w) - 128 * sum(w) per sub-block in one vpdpbusd each (no int16 step, no sign op per
/// row), the correction once per sub-block for all rows. Integer sums are exact, so the result is bit-identical to
/// [`row_dot`]. `nsb % 4 == 0`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2,avxvnni")]
pub unsafe fn row_dot_vnni<C: Cq, const B: usize>(row: *const u8, nsb: usize, a: &Q8Rows) -> [f32; B] {
    let mut acc = [_mm_setzero_ps(); B];
    let k = a.k;
    let nb = k / 32;
    let z = _mm256_setzero_si256();
    let x80 = _mm256_set1_epi8(-128i8);
    let mut j = 0;
    while j < nsb {
        let [d0, d1, d2, d3] = C::dq4(row, j);
        let base = _mm_setr_ps(d0.base, d1.base, d2.base, d3.base);
        let ws = [d0.sg, d1.sg, d2.sg, d3.sg];
        let cr = |i: usize| _mm256_dpbusd_avx_epi32(z, x80, ws[i]);
        let ch = _mm256_hadd_epi32(_mm256_hadd_epi32(cr(0), cr(1)), _mm256_hadd_epi32(cr(2), cr(3)));
        let corr = _mm_add_epi32(_mm256_castsi256_si128(ch), _mm256_extracti128_si256(ch, 1));
        for b in 0..B {
            let ap = a.qu.as_ptr().add(b * k + 32 * j);
            let pp = |i: usize| _mm256_dpbusd_avx_epi32(z, _mm256_loadu_si256(ap.add(32 * i) as *const __m256i), ws[i]);
            let h = red4(pp(0), pp(1), pp(2), pp(3));
            let x = _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)), corr);
            let d8 = _mm_loadu_ps(a.d.as_ptr().add(b * nb + j));
            acc[b] = _mm_fmadd_ps(_mm_mul_ps(_mm_cvtepi32_ps(x), d8), base, acc[b]);
        }
        j += 4;
    }
    let mut out = [0f32; B];
    for b in 0..B {
        let s = acc[b];
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        out[b] = _mm_cvtss_f32(s);
    }
    out
}

/// [`gemv_rows`] through [`row_dot_vnni`].
#[target_feature(enable = "avx2,fma,f16c,bmi2,avxvnni")]
pub unsafe fn gemv_rows_vnni<C: Cq, const B: usize>(w: *const u8, row_bytes: usize, rows: std::ops::Range<usize>,
    a: &Q8Rows, y: *mut f32, ld: usize) {
    let nsb = a.k / 32;
    for r in rows {
        let v = row_dot_vnni::<C, B>(w.add(r * row_bytes), nsb, a);
        for b in 0..B {
            *y.add(b * ld + r) = v[b];
        }
    }
}

/// Per 128-bit lane: [sum of p0's 4 lanes, p1's, p2's, p3's] (what two levels of hadd give), as unpack + add
/// (hadd is 3 uops with two shuffles; integer sums, so any order is exact). FE_GGQ_CPU_HADD=1: the hadd form.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn red4(p0: __m256i, p1: __m256i, p2: __m256i, p3: __m256i) -> __m256i {
    if RED_HADD {
        return _mm256_hadd_epi32(_mm256_hadd_epi32(p0, p1), _mm256_hadd_epi32(p2, p3));
    }
    let s01 = _mm256_add_epi32(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1));
    let s23 = _mm256_add_epi32(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3));
    _mm256_add_epi32(_mm256_unpacklo_epi64(s01, s23), _mm256_unpackhi_epi64(s01, s23))
}
const RED_HADD: bool = false;

/// [`row_dot`] for the offset-form types with one integer scale per sub-block (IQ3_XXS, IQ3_S) with AVX-VNNI:
/// isc * (sum((a + 128) * w) - 128 * sum(w)) per sub-block, the sums by vpdpbusd (no int16 step), the scale as
/// one integer multiply per 4 sub-blocks. Same integers as [`row_dot`]'s offset form: bit-identical. `nsb % 4 == 0`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2,avxvnni")]
pub unsafe fn row_dot_vnni_off<C: Cq, const B: usize>(row: *const u8, nsb: usize, a: &Q8Rows) -> [f32; B] {
    let mut acc = [_mm_setzero_ps(); B];
    let k = a.k;
    let nb = k / 32;
    let z = _mm256_setzero_si256();
    let x80 = _mm256_set1_epi8(-128i8);
    let mut j = 0;
    while j < nsb {
        let [d0, d1, d2, d3] = C::dq4(row, j);
        let base = _mm_setr_ps(d0.base, d1.base, d2.base, d3.base);
        let isc = _mm_setr_epi32(d0.isc, d1.isc, d2.isc, d3.isc);
        let ws = [_mm256_sign_epi8(d0.wa, d0.sg), _mm256_sign_epi8(d1.wa, d1.sg), _mm256_sign_epi8(d2.wa, d2.sg),
            _mm256_sign_epi8(d3.wa, d3.sg)];
        let cr = |i: usize| _mm256_dpbusd_avx_epi32(z, x80, ws[i]);
        let ch = red4(cr(0), cr(1), cr(2), cr(3));
        let corr = _mm_add_epi32(_mm256_castsi256_si128(ch), _mm256_extracti128_si256(ch, 1));
        for b in 0..B {
            let ap = a.qu.as_ptr().add(b * k + 32 * j);
            let pp = |i: usize| _mm256_dpbusd_avx_epi32(z, _mm256_loadu_si256(ap.add(32 * i) as *const __m256i), ws[i]);
            let h = red4(pp(0), pp(1), pp(2), pp(3));
            let x = _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)), corr);
            let x = _mm_mullo_epi32(x, isc);
            let d8 = _mm_loadu_ps(a.d.as_ptr().add(b * nb + j));
            acc[b] = _mm_fmadd_ps(_mm_mul_ps(_mm_cvtepi32_ps(x), d8), base, acc[b]);
        }
        j += 4;
    }
    let mut out = [0f32; B];
    for b in 0..B {
        let s = acc[b];
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        out[b] = _mm_cvtss_f32(s);
    }
    out
}

/// [`gemv_rows`] through [`row_dot_vnni_off`].
#[target_feature(enable = "avx2,fma,f16c,bmi2,avxvnni")]
pub unsafe fn gemv_rows_vnni_off<C: Cq, const B: usize>(w: *const u8, row_bytes: usize, rows: std::ops::Range<usize>,
    a: &Q8Rows, y: *mut f32, ld: usize) {
    let nsb = a.k / 32;
    for r in rows {
        let v = row_dot_vnni_off::<C, B>(w.add(r * row_bytes), nsb, a);
        for b in 0..B {
            *y.add(b * ld + r) = v[b];
        }
    }
}

/// The GPU's per-row float order (k-gguf gemv.rs `row_dot_r` + `reduce_lanes`) for B activation rows: lane i of
/// `lpr` takes sub-blocks i, i + lpr, ... and accumulates v = (d8 * base) * isum (- m * s8) in that order, then the
/// lanes are summed by the xor butterfly (lpr / 2, ..., 1). The integer sub-block sums are exact either way, so
/// the result has the GPU's bits. lpr in {4, 8, 16, 32}, nsb % 4 == 0.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn row_dot_gpu<C: Cq, const B: usize, const NL: usize>(row: *const u8, nsb: usize, a: &Q8Rows) -> [f32; B] {
    let lpr = 4 * NL;
    let mut part = [[_mm_setzero_ps(); NL]; B];
    let k = a.k;
    let nb = k / 32;
    let mut j = 0;
    while j < nsb {
        for li in 0..NL {
            if j >= nsb {
                break;
            }
            let [mut d0, mut d1, mut d2, mut d3] = C::dq4(row, j);
            if C::GPU_ISC_IN_BASE {
                for d in [&mut d0, &mut d1, &mut d2, &mut d3] {
                    d.base = C::gpu_base(d.base, d.isc);
                    d.sc = sc1();
                }
            }
            let base = _mm_setr_ps(d0.base, d1.base, d2.base, d3.base);
            let mv = if C::HAS_M { _mm_setr_ps(d0.m, d1.m, d2.m, d3.m) } else { _mm_setzero_ps() };
            if C::OFFSET_OK && B >= OFFSET_MIN_B {
                let off = _mm256_set1_epi8(-128i8);
                let ws = [_mm256_sign_epi8(d0.wa, d0.sg), _mm256_sign_epi8(d1.wa, d1.sg), _mm256_sign_epi8(d2.wa, d2.sg),
                    _mm256_sign_epi8(d3.wa, d3.sg)];
                let sc = [d0.sc, d1.sc, d2.sc, d3.sc];
                let cr = |i: usize| _mm256_madd_epi16(_mm256_maddubs_epi16(off, ws[i]), sc[i]);
                let ch = red4(cr(0), cr(1), cr(2), cr(3));
                let corr = _mm_add_epi32(_mm256_castsi256_si128(ch), _mm256_extracti128_si256(ch, 1));
                for b in 0..B {
                    let ap = a.qu.as_ptr().add(b * k + 32 * j);
                    let pp = |i: usize| {
                        let av = _mm256_loadu_si256(ap.add(32 * i) as *const __m256i);
                        _mm256_madd_epi16(_mm256_maddubs_epi16(av, ws[i]), sc[i])
                    };
                    let h = red4(pp(0), pp(1), pp(2), pp(3));
                    let x = _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)), corr);
                    let d8 = _mm_loadu_ps(a.d.as_ptr().add(b * nb + j));
                    let mut v = _mm_mul_ps(_mm_mul_ps(d8, base), _mm_cvtepi32_ps(x));
                    if C::HAS_M {
                        v = _mm_sub_ps(v, _mm_mul_ps(mv, _mm_loadu_ps(a.s.as_ptr().add(b * nb + j))));
                    }
                    part[b][li] = _mm_add_ps(part[b][li], v);
                }
            } else {
                for b in 0..B {
                    let ap = a.q.as_ptr().add(b * k + 32 * j);
                    macro_rules! p {
                        ($d:ident, $o:expr) => {{
                            let av = _mm256_loadu_si256(ap.add($o) as *const __m256i);
                            let av = if C::SIGNED { _mm256_sign_epi8(av, $d.sg) } else { av };
                            _mm256_madd_epi16(_mm256_maddubs_epi16($d.wa, av), $d.sc)
                        }};
                    }
                    let (p0, p1, p2, p3) = (p!(d0, 0), p!(d1, 32), p!(d2, 64), p!(d3, 96));
                    let h = red4(p0, p1, p2, p3);
                    let x = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
                    let d8 = _mm_loadu_ps(a.d.as_ptr().add(b * nb + j));
                    let mut v = _mm_mul_ps(_mm_mul_ps(d8, base), _mm_cvtepi32_ps(x));
                    if C::HAS_M {
                        v = _mm_sub_ps(v, _mm_mul_ps(mv, _mm_loadu_ps(a.s.as_ptr().add(b * nb + j))));
                    }
                    part[b][li] = _mm_add_ps(part[b][li], v);
                }
            }
            j += 4;
        }
    }
    let mut out = [0f32; B];
    for b in 0..B {
        let mut l = [0f32; 32];
        for i in 0..NL {
            _mm_storeu_ps(l.as_mut_ptr().add(4 * i), part[b][i]);
        }
        let mut m = lpr / 2;
        while m > 0 {
            let w = l;
            for i in 0..lpr {
                l[i] = w[i] + w[i ^ m];
            }
            m /= 2;
        }
        out[b] = l[0];
    }
    out
}

/// [`gemv_rows`] in the GPU's float order ([`row_dot_gpu`]).
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn gemv_rows_gpu<C: Cq, const B: usize>(w: *const u8, row_bytes: usize, rows: std::ops::Range<usize>,
    a: &Q8Rows, y: *mut f32, ld: usize, lpr: usize) {
    let nsb = a.k / 32;
    for r in rows {
        let rp = w.add(r * row_bytes);
        let v = match lpr {
            4 => row_dot_gpu::<C, B, 1>(rp, nsb, a),
            8 => row_dot_gpu::<C, B, 2>(rp, nsb, a),
            16 => row_dot_gpu::<C, B, 4>(rp, nsb, a),
            _ => row_dot_gpu::<C, B, 8>(rp, nsb, a),
        };
        for b in 0..B {
            *y.add(b * ld + r) = v[b];
        }
    }
}

/// [`row_dot_gpu`] for one-scale types with |w| <= 127 (IQ4_NL, Q8_0, Q2_0) with AVX-VNNI integer sums (exact:
/// same bits). nsb % 4 == 0.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2,avxvnni")]
pub unsafe fn row_dot_gpu_vnni<C: Cq, const B: usize, const NL: usize>(row: *const u8, nsb: usize, a: &Q8Rows) -> [f32; B] {
    let lpr = 4 * NL;
    let mut part = [[_mm_setzero_ps(); NL]; B];
    let k = a.k;
    let nb = k / 32;
    let z = _mm256_setzero_si256();
    let x80 = _mm256_set1_epi8(-128i8);
    let mut j = 0;
    while j < nsb {
        for li in 0..NL {
            if j >= nsb {
                break;
            }
            let [d0, d1, d2, d3] = C::dq4(row, j);
            let base = _mm_setr_ps(d0.base, d1.base, d2.base, d3.base);
            let ws = [d0.sg, d1.sg, d2.sg, d3.sg];
            let cr = |i: usize| _mm256_dpbusd_avx_epi32(z, x80, ws[i]);
            let ch = red4(cr(0), cr(1), cr(2), cr(3));
            let corr = _mm_add_epi32(_mm256_castsi256_si128(ch), _mm256_extracti128_si256(ch, 1));
            for b in 0..B {
                let ap = a.qu.as_ptr().add(b * k + 32 * j);
                let pp = |i: usize| _mm256_dpbusd_avx_epi32(z, _mm256_loadu_si256(ap.add(32 * i) as *const __m256i), ws[i]);
                let h = red4(pp(0), pp(1), pp(2), pp(3));
                let x = _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)), corr);
                let d8 = _mm_loadu_ps(a.d.as_ptr().add(b * nb + j));
                let v = _mm_mul_ps(_mm_mul_ps(d8, base), _mm_cvtepi32_ps(x));
                part[b][li] = _mm_add_ps(part[b][li], v);
            }
            j += 4;
        }
    }
    let mut out = [0f32; B];
    for b in 0..B {
        let mut l = [0f32; 32];
        for i in 0..NL {
            _mm_storeu_ps(l.as_mut_ptr().add(4 * i), part[b][i]);
        }
        let mut m = lpr / 2;
        while m > 0 {
            let w = l;
            for i in 0..lpr {
                l[i] = w[i] + w[i ^ m];
            }
            m /= 2;
        }
        out[b] = l[0];
    }
    out
}

/// [`gemv_rows_gpu`] through [`row_dot_gpu_vnni`].
#[target_feature(enable = "avx2,fma,f16c,bmi2,avxvnni")]
pub unsafe fn gemv_rows_gpu_vnni<C: Cq, const B: usize>(w: *const u8, row_bytes: usize, rows: std::ops::Range<usize>,
    a: &Q8Rows, y: *mut f32, ld: usize, lpr: usize) {
    let nsb = a.k / 32;
    for r in rows {
        let rp = w.add(r * row_bytes);
        let v = match lpr {
            4 => row_dot_gpu_vnni::<C, B, 1>(rp, nsb, a),
            8 => row_dot_gpu_vnni::<C, B, 2>(rp, nsb, a),
            16 => row_dot_gpu_vnni::<C, B, 4>(rp, nsb, a),
            _ => row_dot_gpu_vnni::<C, B, 8>(rp, nsb, a),
        };
        for b in 0..B {
            *y.add(b * ld + r) = v[b];
        }
    }
}
