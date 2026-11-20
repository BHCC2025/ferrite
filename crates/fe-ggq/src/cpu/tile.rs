//! Many-row CPU expert path (prompt chunks: 8..256+ token rows per expert). Each weight row is decoded ONCE per
//! call into int8 + per-sub-block scales (thread-local buffer, L2-resident), then reused by every activation row in
//! a 4 (activation rows) x 2 (weight rows) register-blocked micro-kernel. Same numerics as the GEMV kernels:
//! q8 activations (per-32 scales, ggml float-sum min term), exact integer sub-block dots, f32 scaling and sums.
//!
//! Two inner forms: weights with |w| <= 63 (IQ2_S, IQ3_*, Q2_0, Q4_K, Q5_K, Q6_K) multiply offset activations
//! (a + 128 as u8) with the signed weights (maddubs without saturation) and subtract 128 * sum(w) per sub-block
//! afterwards; IQ4_NL / IQ4_XS / Q8_0 (|w| up to 127) use maddubs(|w|, sign(a, w)).
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]
use super::avx2::{Cq, Q8Rows};
use std::arch::x86_64::*;

/// Decoded weight rows of one matrix slice.
#[derive(Default)]
pub struct DecRows {
    pub k: usize,
    pub nsb: usize,
    pub rows: usize,
    /// offset form: signed w; sign form: |w|
    pub w: Vec<u8>,
    /// sign form only: the sign source (w itself)
    pub sg: Vec<u8>,
    /// integer half-block scales per sub-block (ia, ib)
    pub sc: Vec<[i16; 2]>,
    /// float factor per sub-block
    pub base: Vec<f32>,
    /// both halves share the scale (ia == ib) for every sub-block of every row
    pub same: bool,
    /// per sub-block: base * 128 * sum(w * sc) (offset form) and the min coefficient (Q4_K/Q5_K)
    pub corr: Vec<f32>,
    pub m: Vec<f32>,
    /// the GPU kernels' integer form (`Cq::GPU_ISC_IN_BASE`): half-block scales of the integer sum, float base, and
    /// (offset form) the integer correction 128 * sum(w * scale) per sub-block
    pub gsc: Vec<[i16; 2]>,
    pub gbase: Vec<f32>,
    pub icorr: Vec<i32>,
    pub has_m: bool,
}

/// Whether a type's integer codes fit the offset form (|w| <= 63).
pub fn offset_ok(ty: fe_gguf::GgmlType) -> bool {
    use fe_gguf::GgmlType::*;
    matches!(ty, IQ2S | IQ3XXS | IQ3S | IQ2XXS | IQ1M | IQ2XS | RepIq1M | RepHXxs | RepR3Xxs | RepR3Xs | RepR3Iq1M | Q2_0 | Q4K | Q5K | Q6K)
}

/// Decode rows `r0 .. r0 + n` (row stride `row_bytes`, `k` values) into rows `at ..` of `d`.
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn decode_rows<C: Cq>(w: *const u8, row_bytes: usize, r0: usize, n: usize, k: usize, off: bool, at: usize,
    d: &mut DecRows) {
    let nsb = k / 32;
    d.k = k;
    d.nsb = nsb;
    let tot = at + n;
    d.rows = tot;
    d.w.resize(tot * k, 0);
    if !off {
        d.sg.resize(tot * k, 0);
    }
    d.sc.resize(tot * nsb, [0; 2]);
    d.base.resize(tot * nsb, 0.0);
    d.same = C::SAME;
    d.corr.resize(tot * nsb, 0.0);
    d.m.resize(tot * nsb, 0.0);
    d.gsc.resize(tot * nsb, [0; 2]);
    d.gbase.resize(tot * nsb, 0.0);
    d.icorr.resize(tot * nsb, 0);
    d.has_m = C::HAS_M;
    let x80 = _mm256_set1_epi8(-128i8);
    for rr in 0..n {
        let r = at + rr;
        let row = w.add((r0 + rr) * row_bytes);
        for j in 0..nsb {
            let q = C::dq(row, j);
            let ws = if C::SIGNED { _mm256_sign_epi8(q.wa, q.sg) } else { q.wa };
            let o = r * k + 32 * j;
            if off {
                _mm256_storeu_si256(d.w.as_mut_ptr().add(o) as *mut __m256i, ws);
                let c = _mm256_madd_epi16(_mm256_maddubs_epi16(x80, ws), q.sc);
                d.corr[r * nsb + j] = q.base * hsum_i(c) as f32;
            } else {
                _mm256_storeu_si256(d.w.as_mut_ptr().add(o) as *mut __m256i, _mm256_sign_epi8(ws, ws));
                _mm256_storeu_si256(d.sg.as_mut_ptr().add(o) as *mut __m256i, ws);
                d.corr[r * nsb + j] = 0.0;
            }
            let lo = _mm256_extract_epi16::<0>(q.sc) as i16;
            let hi = _mm256_extract_epi16::<8>(q.sc) as i16;
            d.sc[r * nsb + j] = [lo, hi];
            d.base[r * nsb + j] = q.base;
            d.m[r * nsb + j] = if C::HAS_M { q.m } else { 0.0 };
            let (gsv, gb) = if C::GPU_ISC_IN_BASE { (_mm256_set1_epi16(1), C::gpu_base(q.base, q.isc)) } else { (q.sc, q.base) };
            d.gsc[r * nsb + j] = [_mm256_extract_epi16::<0>(gsv) as i16, _mm256_extract_epi16::<8>(gsv) as i16];
            d.gbase[r * nsb + j] = gb;
            d.icorr[r * nsb + j] = if off { hsum_i(_mm256_madd_epi16(_mm256_maddubs_epi16(x80, ws), gsv)) } else { 0 };
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn hsum_i(v: __m256i) -> i32 {
    let x = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0x4e));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0xb1));
    _mm_cvtsi128_si32(x)
}
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn hsum_f(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// out[b][e] = W[wr[e]] . a[ar[b]] for b < na (<= 4), e < 2.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn micro<const OFF: bool, const CORR: bool, const SAME: bool>(d: &DecRows, wr: [usize; 2], a: &Q8Rows,
    ar: [usize; 4], na: usize) -> [[f32; 2]; 4] {
    let (k, nsb) = (d.k, d.nsb);
    let mut acc = [[_mm256_setzero_ps(); 2]; 4];
    let w0p = d.w.as_ptr().add(wr[0] * k);
    let w1p = d.w.as_ptr().add(wr[1] * k);
    let s0p = d.sg.as_ptr().wrapping_add(wr[0] * k);
    let s1p = d.sg.as_ptr().wrapping_add(wr[1] * k);
    let ap: [*const u8; 4] = core::array::from_fn(|b| {
        if OFF { a.qu.as_ptr().add(ar[b.min(na - 1)] * k) } else { a.q.as_ptr().add(ar[b.min(na - 1)] * k) as *const u8 }
    });
    let dp: [*const f32; 4] = core::array::from_fn(|b| a.d.as_ptr().add(ar[b.min(na - 1)] * nsb));
    for j in 0..nsb {
        let w0 = _mm256_loadu_si256(w0p.add(32 * j) as *const __m256i);
        let w1 = _mm256_loadu_si256(w1p.add(32 * j) as *const __m256i);
        let s0 = *d.sc.get_unchecked(wr[0] * nsb + j);
        let s1 = *d.sc.get_unchecked(wr[1] * nsb + j);
        let (sc0, sc1) = if SAME {
            (_mm256_set1_epi16(s0[0]), _mm256_set1_epi16(s1[0]))
        } else {
            (_mm256_set_m128i(_mm_set1_epi16(s0[1]), _mm_set1_epi16(s0[0])),
                _mm256_set_m128i(_mm_set1_epi16(s1[1]), _mm_set1_epi16(s1[0])))
        };
        let b0 = _mm256_set1_ps(*d.base.get_unchecked(wr[0] * nsb + j));
        let b1 = _mm256_set1_ps(*d.base.get_unchecked(wr[1] * nsb + j));
        let (g0, g1) = if OFF {
            (w0, w1)
        } else {
            (_mm256_loadu_si256(s0p.add(32 * j) as *const __m256i), _mm256_loadu_si256(s1p.add(32 * j) as *const __m256i))
        };
        macro_rules! row {
            ($b:literal) => {
                {
                    let av = _mm256_loadu_si256(ap[$b].add(32 * j) as *const __m256i);
                    let (p0, p1) = if OFF {
                        (_mm256_maddubs_epi16(av, w0), _mm256_maddubs_epi16(av, w1))
                    } else {
                        (_mm256_maddubs_epi16(w0, _mm256_sign_epi8(av, g0)), _mm256_maddubs_epi16(w1, _mm256_sign_epi8(av, g1)))
                    };
                    let d8 = _mm256_set1_ps(*dp[$b].add(j));
                    let f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p0, sc0)), d8);
                    let f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p1, sc1)), d8);
                    acc[$b][0] = _mm256_fmadd_ps(f0, b0, acc[$b][0]);
                    acc[$b][1] = _mm256_fmadd_ps(f1, b1, acc[$b][1]);
                }
            };
        }
        row!(0);
        row!(1);
        row!(2);
        row!(3);
    }
    let mut out = [[0f32; 2]; 4];
    for b in 0..na {
        let ab = ar[b];
        let (dd, ss) = (a.d.as_ptr().add(ab * nsb), a.s.as_ptr().add(ab * nsb));
        for e in 0..2 {
            if !CORR {
                out[b][e] = hsum_f(acc[b][e]);
                continue;
            }
            // offset correction and min term: sum over the sub-blocks of d8 * corr + s * m (vectorized)
            let (cp, mp) = (d.corr.as_ptr().add(wr[e] * nsb), d.m.as_ptr().add(wr[e] * nsb));
            let mut cv = _mm256_setzero_ps();
            let mut jj = 0;
            while jj + 8 <= nsb {
                cv = _mm256_fmadd_ps(_mm256_loadu_ps(dd.add(jj)), _mm256_loadu_ps(cp.add(jj)), cv);
                cv = _mm256_fmadd_ps(_mm256_loadu_ps(ss.add(jj)), _mm256_loadu_ps(mp.add(jj)), cv);
                jj += 8;
            }
            if jj < nsb {
                // tail of 4 (nsb = 20 for the down GEMVs)
                let c4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(dd.add(jj)), _mm_loadu_ps(cp.add(jj))),
                    _mm_mul_ps(_mm_loadu_ps(ss.add(jj)), _mm_loadu_ps(mp.add(jj))));
                cv = _mm256_add_ps(cv, _mm256_insertf128_ps::<0>(_mm256_setzero_ps(), c4));
                jj += 4;
            }
            debug_assert!(jj == nsb);
            out[b][e] = hsum_f(acc[b][e]) - hsum_f(cv);
        }
    }
    out
}

/// [`micro`] with the forms picked at run time (offset form, correction epilogue, one scale per sub-block).
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn micro_any(d: &DecRows, wr: [usize; 2], a: &Q8Rows, ar: [usize; 4], na: usize, off: bool, corr: bool)
    -> [[f32; 2]; 4] {
    match (off, corr, d.same) {
        (true, _, true) => micro::<true, true, true>(d, wr, a, ar, na),
        (true, _, false) => micro::<true, true, false>(d, wr, a, ar, na),
        (false, true, true) => micro::<false, true, true>(d, wr, a, ar, na),
        (false, true, false) => micro::<false, true, false>(d, wr, a, ar, na),
        (false, false, true) => micro::<false, false, true>(d, wr, a, ar, na),
        (false, false, false) => micro::<false, false, false>(d, wr, a, ar, na),
    }
}

/// Per 128-bit lane [sum of p0's 4 i32, p1's, p2's, p3's] (as cpu::avx2 `red4`).
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn red4t(p0: __m256i, p1: __m256i, p2: __m256i, p3: __m256i) -> __m128i {
    let s01 = _mm256_add_epi32(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1));
    let s23 = _mm256_add_epi32(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3));
    let h = _mm256_add_epi32(_mm256_unpacklo_epi64(s01, s23), _mm256_unpackhi_epi64(s01, s23));
    _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1))
}

/// [`micro`] in the GPU decode kernels' float order (bit-identical to them): per sub-block the exact integer sum in
/// the GPU's form, v = (d8 * base) * isum (- m * s8), accumulated into lane partial (j / 4) % NL (lpr = 4 NL) and
/// summed by the xor butterfly. out[b][e] = W[wr[e]] . a[ar[b]], b < na (<= 4). nsb % 4 == 0.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn micro_gpu<const OFF: bool, const NL: usize>(d: &DecRows, wr: [usize; 2], a: &Q8Rows, ar: [usize; 4], na: usize)
    -> [[f32; 2]; 4] {
    let (k, nsb) = (d.k, d.nsb);
    let mut part = [[[_mm_setzero_ps(); NL]; 2]; 4];
    let ap: [*const u8; 4] = core::array::from_fn(|b| {
        if OFF { a.qu.as_ptr().add(ar[b.min(na - 1)] * k) } else { a.q.as_ptr().add(ar[b.min(na - 1)] * k) as *const u8 }
    });
    let dp: [*const f32; 4] = core::array::from_fn(|b| a.d.as_ptr().add(ar[b.min(na - 1)] * nsb));
    let sp: [*const f32; 4] = core::array::from_fn(|b| a.s.as_ptr().add(ar[b.min(na - 1)] * nsb));
    let mut j = 0;
    while j < nsb {
        for li in 0..NL {
            if j >= nsb {
                break;
            }
            for e in 0..2 {
                let wrow = wr[e];
                let wp = d.w.as_ptr().add(wrow * k + 32 * j);
                let gp = d.sg.as_ptr().wrapping_add(wrow * k + 32 * j);
                let si = wrow * nsb + j;
                let sc: [__m256i; 4] = core::array::from_fn(|i| {
                    let s2 = *d.gsc.get_unchecked(si + i);
                    _mm256_set_m128i(_mm_set1_epi16(s2[1]), _mm_set1_epi16(s2[0]))
                });
                let w: [__m256i; 4] = core::array::from_fn(|i| _mm256_loadu_si256(wp.add(32 * i) as *const __m256i));
                let base = _mm_loadu_ps(d.gbase.as_ptr().add(si));
                let mv = _mm_loadu_ps(d.m.as_ptr().add(si));
                let ic = _mm_loadu_si128(d.icorr.as_ptr().add(si) as *const __m128i);
                for b in 0..4 {
                    if b >= na {
                        break;
                    }
                    let pb = ap[b].add(32 * j);
                    let pp = |i: usize| {
                        let av = _mm256_loadu_si256(pb.add(32 * i) as *const __m256i);
                        let p = if OFF {
                            _mm256_maddubs_epi16(av, w[i])
                        } else {
                            let g = _mm256_loadu_si256(gp.add(32 * i) as *const __m256i);
                            _mm256_maddubs_epi16(w[i], _mm256_sign_epi8(av, g))
                        };
                        _mm256_madd_epi16(p, sc[i])
                    };
                    let mut x = red4t(pp(0), pp(1), pp(2), pp(3));
                    if OFF {
                        x = _mm_sub_epi32(x, ic);
                    }
                    let d8 = _mm_loadu_ps(dp[b].add(j));
                    let mut v = _mm_mul_ps(_mm_mul_ps(d8, base), _mm_cvtepi32_ps(x));
                    if d.has_m {
                        v = _mm_sub_ps(v, _mm_mul_ps(mv, _mm_loadu_ps(sp[b].add(j))));
                    }
                    part[b][e][li] = _mm_add_ps(part[b][e][li], v);
                }
            }
            j += 4;
        }
    }
    let lpr = 4 * NL;
    let mut out = [[0f32; 2]; 4];
    for b in 0..na {
        for e in 0..2 {
            let mut l = [0f32; 32];
            for i in 0..NL {
                _mm_storeu_ps(l.as_mut_ptr().add(4 * i), part[b][e][i]);
            }
            let mut m = lpr / 2;
            while m > 0 {
                let w = l;
                for i in 0..lpr {
                    l[i] = w[i] + w[i ^ m];
                }
                m /= 2;
            }
            out[b][e] = l[0];
        }
    }
    out
}

/// [`micro_gpu`] with the form and lane count picked at run time.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn micro_gpu_any(d: &DecRows, wr: [usize; 2], a: &Q8Rows, ar: [usize; 4], na: usize, off: bool, lpr: usize)
    -> [[f32; 2]; 4] {
    match (off, lpr) {
        (true, 4) => micro_gpu::<true, 1>(d, wr, a, ar, na),
        (true, 8) => micro_gpu::<true, 2>(d, wr, a, ar, na),
        (true, 16) => micro_gpu::<true, 4>(d, wr, a, ar, na),
        (true, _) => micro_gpu::<true, 8>(d, wr, a, ar, na),
        (false, 4) => micro_gpu::<false, 1>(d, wr, a, ar, na),
        (false, 8) => micro_gpu::<false, 2>(d, wr, a, ar, na),
        (false, 16) => micro_gpu::<false, 4>(d, wr, a, ar, na),
        (false, _) => micro_gpu::<false, 8>(d, wr, a, ar, na),
    }
}

#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
unsafe fn tr4(r: [__m128; 4]) -> [__m128; 4] {
    let t0 = _mm_unpacklo_ps(r[0], r[1]);
    let t1 = _mm_unpacklo_ps(r[2], r[3]);
    let t2 = _mm_unpackhi_ps(r[0], r[1]);
    let t3 = _mm_unpackhi_ps(r[2], r[3]);
    [_mm_movelh_ps(t0, t1), _mm_movehl_ps(t1, t0), _mm_movelh_ps(t2, t3), _mm_movehl_ps(t3, t2)]
}

/// [`micro`] in the prompt MMQ's float order (k-gguf mmq.rs, bit-identical to it): per output one accumulator over
/// the sub-blocks in K order, acc += (d8 * base) * isum (- m * s8 for min types), with the GPU integer form.
/// out[b][e] = W[wr[e]] . a[ar[b]], b < na (<= 4). nsb % 4 == 0.
#[inline]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn micro_mmq<const OFF: bool>(d: &DecRows, wr: [usize; 2], a: &Q8Rows, ar: [usize; 4], na: usize) -> [[f32; 2]; 4] {
    let (k, nsb) = (d.k, d.nsb);
    let mut acc = [_mm_setzero_ps(); 2]; // lanes = activation rows b
    let ap: [*const u8; 4] = core::array::from_fn(|b| {
        if OFF { a.qu.as_ptr().add(ar[b.min(na - 1)] * k) } else { a.q.as_ptr().add(ar[b.min(na - 1)] * k) as *const u8 }
    });
    let dp: [*const f32; 4] = core::array::from_fn(|b| a.d.as_ptr().add(ar[b.min(na - 1)] * nsb));
    let sp: [*const f32; 4] = core::array::from_fn(|b| a.s.as_ptr().add(ar[b.min(na - 1)] * nsb));
    let mut j = 0;
    while j < nsb {
        // d8 / s8 as [sub-block][row b]
        let d8t = tr4(core::array::from_fn(|b| _mm_loadu_ps(dp[b].add(j))));
        let s8t = if d.has_m { tr4(core::array::from_fn(|b| _mm_loadu_ps(sp[b].add(j)))) } else { [_mm_setzero_ps(); 4] };
        for e in 0..2 {
            let wrow = wr[e];
            let wp = d.w.as_ptr().add(wrow * k + 32 * j);
            let gp = d.sg.as_ptr().wrapping_add(wrow * k + 32 * j);
            let si = wrow * nsb + j;
            let sc: [__m256i; 4] = core::array::from_fn(|i| {
                let s2 = *d.gsc.get_unchecked(si + i);
                _mm256_set_m128i(_mm_set1_epi16(s2[1]), _mm_set1_epi16(s2[0]))
            });
            let w: [__m256i; 4] = core::array::from_fn(|i| _mm256_loadu_si256(wp.add(32 * i) as *const __m256i));
            let ic = _mm_loadu_si128(d.icorr.as_ptr().add(si) as *const __m128i);
            let mut fx = [_mm_setzero_ps(); 4];
            for b in 0..4 {
                if b >= na {
                    break;
                }
                let pb = ap[b].add(32 * j);
                let pp = |i: usize| {
                    let av = _mm256_loadu_si256(pb.add(32 * i) as *const __m256i);
                    let p = if OFF {
                        _mm256_maddubs_epi16(av, w[i])
                    } else {
                        let g = _mm256_loadu_si256(gp.add(32 * i) as *const __m256i);
                        _mm256_maddubs_epi16(w[i], _mm256_sign_epi8(av, g))
                    };
                    _mm256_madd_epi16(p, sc[i])
                };
                let mut x = red4t(pp(0), pp(1), pp(2), pp(3));
                if OFF {
                    x = _mm_sub_epi32(x, ic);
                }
                fx[b] = _mm_cvtepi32_ps(x);
            }
            let ft = tr4(fx); // [sub-block][row b]
            for jj in 0..4 {
                let base = _mm_set1_ps(*d.gbase.get_unchecked(si + jj));
                let mut v = _mm_mul_ps(_mm_mul_ps(d8t[jj], base), ft[jj]);
                if d.has_m {
                    v = _mm_sub_ps(v, _mm_mul_ps(_mm_set1_ps(*d.m.get_unchecked(si + jj)), s8t[jj]));
                }
                acc[e] = _mm_add_ps(acc[e], v);
            }
        }
        j += 4;
    }
    let mut out = [[0f32; 2]; 4];
    let mut l0 = [0f32; 4];
    let mut l1 = [0f32; 4];
    _mm_storeu_ps(l0.as_mut_ptr(), acc[0]);
    _mm_storeu_ps(l1.as_mut_ptr(), acc[1]);
    for b in 0..na {
        out[b] = [l0[b], l1[b]];
    }
    out
}
