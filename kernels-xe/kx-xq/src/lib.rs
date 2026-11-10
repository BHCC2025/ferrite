//! XQ weights (symmetric integer codes x per-group fp16 scales, input-side 128-block Hadamard rotation) on the Arc Pro
//! B70: the activation prepasses and the decode matvec for 1..16 activation rows on the matrix engine. Rust port of
//! arc-engine `kernels/xpu/xq.cl`, operation for operation, so the outputs are bit-identical to `xq{3,4,5,6,8}.spv`.
//! Built once per code width (cargo feature b3 | b4 | b5 | b6 | b8 -> kx_xq{N}.spv); kernel names as in xq.cl.
//!
//! y[r][n] = sum_g s[n][g] * (sum_{k in g} x~[r][k] * q[n][k]),  x~ = fp16(FWHT128(su * x) / sqrt(128)) (no rotation: fp16(x)).
//! A code is stored unsigned, u = q + 2^(b-1); the fp16 bits 0x6400 | u are exactly 1024 + u, so a B fragment is built
//! with shifts, masks and one or. Per group a fresh DPAS chain gives c = sum x~ (1024 + u) (fp32), and the offset comes
//! back out with the group sum of x~ from the prepass: acc = fma(s, fma(-(1024 + 2^(b-1)), xs, c), acc).
//! Row invariance: a row depends on its own x~ row, the weights and a fixed chain order only (DPAS rows are independent,
//! the K split S is a function of the shape, the S partials are added in split order through SLM).
//!
//! Floating point: every product/sum here is either explicit `fma` (where xq.cl writes fma()) or two roundings (Rust
//! never contracts; the module has ContractionOff, as xq4.spv has on every kernel). `a / b` in xq.cl is OpenCL's
//! 2.5-ulp division -> `math::div_ocl`.
#![no_std]
#![feature(abi_gpu_kernel, portable_simd, f16)]
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
use kxcommon::sg::Shuffle;
use kxcommon::v::*;
use kxcommon::{attr, bits, blk, cvt, dpas, id, math, sg, slm, unroll};

#[cfg(feature = "b3")]
const BITS: usize = 3;
#[cfg(feature = "b4")]
const BITS: usize = 4;
#[cfg(feature = "b5")]
const BITS: usize = 5;
#[cfg(feature = "b6")]
const BITS: usize = 6;
#[cfg(feature = "b8")]
const BITS: usize = 8;

const HAD_SCALE: f32 = 0.08838834764831845; // the literal of xq.cl, rounded to f32 the same way
const CORR: f32 = 1024.0 + (1u32 << (BITS - 1)) as f32;

#[inline(always)]
fn bf(v: u16) -> f32 {
    cvt::bf16_to_f32(v)
}
#[inline(always)]
fn to_bf(f: f32) -> u16 {
    cvt::f32_to_bf16(f)
}

/// x[i] of dtype dt (0 fp16, 1 bf16, 2 fp32) as fp32.
#[inline(always)]
unsafe fn ld(p: *const u8, dt: u32, i: usize) -> f32 {
    if dt == 0 {
        cvt::f16_bits_to_f32(*(p as *const u16).add(i))
    } else if dt == 1 {
        bf(*(p as *const u16).add(i))
    } else {
        *(p as *const f32).add(i)
    }
}

/// Unscaled Walsh-Hadamard of 128 values, lane l holding l + 16 i (bits 4-6 in registers, bits 0-3 by shuffle, a - b
/// with the lower index first): xq.cl `fwht8`.
#[inline(always)]
fn fwht8(v: &mut [f32; 8], lane: u32) {
    unroll!(4, 3, |mi| {
        let m = 1usize << mi;
        unroll!(8, 8, |i| {
            if i & m == 0 {
                let (a, b) = (v[i], v[i | m]);
                v[i] = a + b;
                v[i | m] = a - b;
            }
        });
    });
    unroll!(4, 4, |mi| {
        let m = 1u32 << mi;
        unroll!(8, 8, |i| {
            let o = v[i].shuffle_xor(m);
            v[i] = if lane & m != 0 { o - v[i] } else { v[i] + o };
        });
    });
}

// ---- activation prepass -------------------------------------------------------------------------------------------
// x [M rows, stride xstride] -> xt [row group][K/16 steps][MW][16] fp16 (the DPAS A operand of a step is one block read)
// and xs [row group][K/32][MW] fp32 sums of the fp16 x~ over 32 inputs. Rows M .. Mpad are zero. rot = 0: no rotation.

/// The rotation and the stores of one 128 block of row `row` (v: the 8 values of the lane, su already applied).
#[inline(always)]
unsafe fn prep_store(v: &mut [f32; 8], rot: u32, lane: u32, row: u32, blk_: u32, xt: *mut u16, xs: *mut f32, k: u32, mw: u32) {
    if rot != 0 {
        fwht8(v, lane);
        unroll!(8, 8, |i| v[i] *= HAD_SCALE);
    }
    let (rg, r, t) = (row / mw, row % mw, k / 16);
    let mut h = [0f32; 8];
    unroll!(8, 8, |i| {
        let hv = cvt::f16_rte(v[i]);
        h[i] = hv as f32;
        // index types as xq.cl: (((ulong)rg * T + blk * 8 + i) * MW + r) * 16 + lane (32-bit terms stay 32-bit)
        let at = (((rg as u64 * t as u64 + (blk_ * 8 + i as u32) as u64) * mw as u64 + r as u64) * 16 + lane as u64) as usize;
        *xt.add(at) = bits::<f16, u16>(hv);
    });
    unroll!(4, 4, |j| {
        let s = sg::reduce_add_f32(h[2 * j] + h[2 * j + 1]);
        if lane == 0 {
            *xs.add(((rg as u64 * (k / 32) as u64 + (blk_ * 4 + j as u32) as u64) * mw as u64 + r as u64) as usize) = s;
        }
    });
}

/// Grid (K/128, Mpad), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_prep(
    x: *const u8,
    xdt: u32,
    xstride: u32,
    su: *const u16,
    rot: u32,
    xt: *mut u16,
    xs: *mut f32,
    k: u32,
    m: u32,
    mw: u32,
) {
    let (lane, row, blk_) = (id::lane(), id::group_id(1), id::group_id(0));
    let k0 = blk_ * 128 + lane;
    let mut v = [0f32; 8];
    unroll!(8, 8, |i| {
        let a = if row < m { ld(x, xdt, (row as u64 * xstride as u64 + (k0 + 16 * i as u32) as u64) as usize) } else { 0.0 };
        v[i] = if rot != 0 { a * cvt::f16_bits_to_f32(*su.add(k0 as usize + 16 * i)) } else { a };
    });
    prep_store(&mut v, rot, lane, row, blk_, xt, xs, k, mw);
}

/// act = bf16(silu(g) * u) for M rows [M][K] (qwen_basic.cl swiglu's expression, so act has its bits) and the prepass
/// of act, in one launch. Grid (K/128, Mpad), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_prep_swiglu(
    g: *const u16,
    u: *const u16,
    act: *mut u16,
    su: *const u16,
    rot: u32,
    xt: *mut u16,
    xs: *mut f32,
    k: u32,
    m: u32,
    mw: u32,
) {
    let (lane, row, blk_) = (id::lane(), id::group_id(1), id::group_id(0));
    let k0 = blk_ * 128 + lane;
    let mut v = [0f32; 8];
    unroll!(8, 8, |i| {
        let mut val = 0.0f32;
        if row < m {
            let at = (row as u64 * k as u64 + (k0 + 16 * i as u32) as u64) as usize;
            let a = bf(*g.add(at));
            let r = to_bf(math::div_ocl(a, 1.0 + math::exp(-a)) * bf(*u.add(at)));
            *act.add(at) = r;
            val = bf(r);
        }
        v[i] = if rot != 0 { val * cvt::f16_bits_to_f32(*su.add(k0 as usize + 16 * i)) } else { val };
    });
    prep_store(&mut v, rot, lane, row, blk_, xt, xs, k, mw);
}

/// x = bf16(x + d); y = bf16(x * rsqrt(mean(x^2) + eps) * w) for one row of 5120 (qwen_basic.cl add_rmsnorm, same
/// operation order), then the prepass of y (one row, MW = 1). One work-group of 512 (32 sub-groups; sub-group j rotates
/// blocks j, j + 32).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_add_rms_prep(
    x: *mut u16,
    d: *const u16,
    w: *const u16,
    y: *mut u16,
    eps: f32,
    add: u32,
    su: *const u16,
    rot: u32,
    xt: *mut u16,
    xs: *mut f32,
) {
    attr::reqd_work_group_size::<512, 1, 1>();
    let part = slm::alloc::<f32, 4, 0>();
    let vals = slm::alloc::<f32, 5120, 1>();
    let lid = id::local_id(0) as usize;
    let mut xa = [0u16; 10];
    let mut da = [0u16; 10];
    let mut wa = [0u16; 10];
    unroll!(16, 10, |k| {
        xa[k] = *x.add(lid + 512 * k);
        da[k] = if add != 0 { *d.add(lid + 512 * k) } else { 0 };
        wa[k] = *w.add(lid + 512 * k);
    });
    unroll!(16, 10, |k| {
        if add != 0 {
            let v = to_bf(bf(xa[k]) + bf(da[k]));
            *x.add(lid + 512 * k) = v;
            *vals.add(lid + 512 * k) = bf(v);
        } else {
            *vals.add(lid + 512 * k) = bf(xa[k]);
        }
    });
    slm::barrier();
    // sumsq_5120 of qwen_basic.cl: lane L of the first 64 items adds L, L + 64, ... in order, sub-group reduce, four
    // parts left to right
    let mut ss = 0.0f32;
    if lid < 64 {
        let mut k = 0;
        while k < 80 {
            let f = *vals.add(lid + 64 * k);
            ss = math::fma(f, f, ss);
            k += 1;
        }
        ss = sg::reduce_add_f32(ss);
        if id::lane() == 0 {
            *part.add(id::sg_id() as usize) = ss;
        }
    }
    slm::barrier();
    let scale = math::rsqrt(math::div_ocl(*part + *part.add(1) + *part.add(2) + *part.add(3), 5120.0) + eps);
    let mut ya = [0u16; 10];
    unroll!(16, 10, |k| {
        ya[k] = to_bf(*vals.add(lid + 512 * k) * scale * bf(wa[k]));
        *y.add(lid + 512 * k) = ya[k];
    });
    slm::barrier();
    unroll!(16, 10, |k| *vals.add(lid + 512 * k) = bf(ya[k]));
    slm::barrier();
    let (lane, sgi) = (id::lane(), id::sg_id());
    let mut b = sgi;
    while b < 40 {
        let mut v = [0f32; 8];
        unroll!(8, 8, |i| {
            let kk = (b * 128 + lane) as usize + 16 * i;
            v[i] = if rot != 0 { *vals.add(kk) * cvt::f16_bits_to_f32(*su.add(kk)) } else { *vals.add(kk) };
        });
        prep_store(&mut v, rot, lane, 0, b, xt, xs, 5120, 1);
        b += 32;
    }
}

// ---- decode matvec ------------------------------------------------------------------------------------------------
// Weight layout (xq.zig pack / fe-xq pack): 16-row blocks; chunk (block rb, group gi) at dword (rb * NG + gi) * (16 D + 8),
// D = G * BITS / 32 dwords a lane: [D dwords][16 lanes] (lane = row rb * 16 + lane), then 16 fp16 scales. A lane's
// dwords: the low-nibble plane (G / 8 dwords), then the high plane (5: 1 bit, 6: 2 bits, 8: 4 bits a code). BITS 3: a
// 2-bit plane (G/16 dwords) then a 1-bit plane (G/32 dwords).

/// Dwords a lane per group.
const fn lane_dwords(g: usize) -> usize {
    g * BITS / 32
}

/// Moves bit sh to bit 4 (and sh + 16 to 20).
#[inline(always)]
fn shl4(v: u32, sh: usize) -> u32 {
    if sh >= 4 { v >> (sh - 4) } else { v << (4 - sh) }
}
/// Moves bit sh to bit 2 (and sh + 16 to 18).
#[inline(always)]
fn shl2(v: u32, sh: usize) -> u32 {
    if sh >= 2 { v >> (sh - 2) } else { v << (2 - sh) }
}

/// B fragment dword d of step t: the fp16 pair (1024 + u[2d], 1024 + u[2d + 1]). `w`: the lane's D dwords of the group.
#[inline(always)]
fn bdw<const G: usize>(w: &[u32], t: usize, d: usize) -> i32 {
    let (h, j) = (d >> 2, d & 3);
    if BITS == 3 {
        return (((w[t] >> (2 * d)) & 0x0003_0003) | (shl2(w[G / 16 + (t >> 1)], (t & 1) * 8 + d) & 0x0004_0004) | 0x6400_6400) as i32;
    }
    let mut v = (w[2 * t + h] >> (4 * j)) & 0x000F_000F;
    if BITS == 5 {
        v |= shl4(w[G / 8 + (t >> 1)], (t & 1) * 8 + d) & 0x0010_0010;
    }
    if BITS == 6 {
        v |= shl4(w[G / 8 + t], 2 * d) & 0x0030_0030;
    }
    if BITS == 8 {
        v |= shl4(w[G / 8 + 2 * t + h], 4 * j) & 0x00F0_00F0;
    }
    (v | 0x6400_6400) as i32
}

/// One DPAS of the M live rows (M = 1, 2, 4, 8): the narrow forms keep the idle accumulator rows out of the chain.
#[inline(always)]
fn dpas_m(m: usize, a: &[u16], b: I32x8, c: F32x8) -> F32x8 {
    let mut c = c;
    match m {
        1 => c[0] = dpas::f16_m1(a[0] as i16, b, c[0]),
        2 => {
            let r = dpas::f16_m2(I16x2::from_array([a[0] as i16, a[1] as i16]), b, F32x2::from_array([c[0], c[1]])).to_array();
            c[0] = r[0];
            c[1] = r[1];
        }
        4 => {
            let r = dpas::f16_m4(
                I16x4::from_array([a[0] as i16, a[1] as i16, a[2] as i16, a[3] as i16]),
                b,
                F32x4::from_array([c[0], c[1], c[2], c[3]]),
            )
            .to_array();
            unroll!(4, 4, |i| c[i] = r[i]);
        }
        _ => {
            let av = I16x8::from_array([
                a[0] as i16, a[1] as i16, a[2] as i16, a[3] as i16, a[4] as i16, a[5] as i16, a[6] as i16, a[7] as i16,
            ]);
            c = dpas::f16_m8(av, b, c);
        }
    }
    c
}

/// The xs sum of group gi (G / 32 partial sums, added pairwise in a fixed order) for the MW rows of row group rg. The MW
/// sums of a 32-block are contiguous (one vector load each in xq.cl).
#[inline(always)]
unsafe fn xs_group<const G: usize, const MW: usize>(xs: *const f32, k: u32, rg: usize, gi: u32, o: &mut [f32; 8]) {
    // ((ulong)rg * (K / 32) + gi * (G / 32)) * MW, the group term in 32 bits as in xq.cl
    let p = xs.add(((rg as u64 * (k / 32) as u64 + (gi * (G / 32) as u32) as u64) * MW as u64) as usize);
    let ld_mw = |q: *const f32| -> [f32; 8] {
        let mut v = [0f32; 8];
        unroll!(8, MW, |r| v[r] = *q.add(r));
        v
    };
    let a = ld_mw(p);
    let b = if G >= 64 { ld_mw(p.add(MW)) } else { [0.0; 8] };
    let (c, d) = if G == 128 { (ld_mw(p.add(2 * MW)), ld_mw(p.add(3 * MW))) } else { ([0.0; 8], [0.0; 8]) };
    unroll!(8, MW, |r| {
        o[r] = if G == 32 {
            a[r]
        } else if G == 64 {
            a[r] + b[r]
        } else {
            (a[r] + b[r]) + (c[r] + d[r])
        };
    });
}

/// The chunk of group gi for NB blocks: w [NB][D] dwords a lane, sc [NB] scales.
#[inline(always)]
unsafe fn ld_chunk<const G: usize, const NB: usize>(wt: *const u32, ng: u32, rb0: u32, gi: u32, w: &mut [u32; 64], sc: &mut [f32; 2]) {
    let d = lane_dwords(G);
    let ch = 16 * d + 8;
    unroll!(2, NB, |nb| {
        let cp = wt.add(((rb0 as usize + nb) * ng as usize + gi as usize) * ch);
        blk::global::read_u32_run(d, cp, &mut w[nb * d..nb * d + d]);
        sc[nb] = cvt::f16_bits_to_f32(blk::global::read_u16(cp.add(16 * d) as *const u16));
    });
}

/// One group: per (block, row group) a fresh DPAS chain over the group's steps in order, then acc += s * (c - CORR xs).
/// NR = 1: the A operand of the group is read in one run of block reads. NR = 2: A is read for both row groups in batches
/// of up to 4 steps and each B fragment is decoded once for both.
#[inline(always)]
unsafe fn grp<const G: usize, const MW: usize, const NR: usize, const NB: usize>(
    w: &[u32; 64],
    sc: &[f32; 2],
    xt: *const u16,
    xs: *const f32,
    k: u32,
    gi: u32,
    acc: &mut [F32x8; 4],
) {
    let t_n = G / 16;
    let d_n = lane_dwords(G);
    let tb = if NR == 1 { t_n } else if t_n < 4 { t_n } else { 4 };
    let mut c = [F32x8::splat(0.0); 4];
    unroll!(2, t_n / tb, |ti| {
        let t0 = ti * tb;
        let mut a = [[0u16; 64]; 2];
        unroll!(2, NR, |q| {
            // (((ulong)q * (K / 16) + gi * T + t0) * MW) * 16: gi * T + t0 in 32 bits (64-bit integer math costs two
            // instructions an operation on Xe2)
            let p = xt.add((((q as u64 * (k / 16) as u64 + (gi * t_n as u32 + t0 as u32) as u64) * MW as u64) * 16) as usize);
            blk::global::read_u16_run(tb * MW, p, &mut a[q]);
        });
        unroll!(8, tb, |t| {
            unroll!(2, NB, |nb| {
                let wl = &w[nb * d_n..nb * d_n + d_n];
                let mut bv = [0i32; 8];
                unroll!(8, 8, |d| bv[d] = bdw::<G>(wl, t0 + t, d));
                let b = I32x8::from_array(bv);
                unroll!(2, NR, |q| c[nb * NR + q] = dpas_m(MW, &a[q][t * MW..], b, c[nb * NR + q]));
            });
        });
    });
    unroll!(2, NR, |q| {
        let mut xsg = [0f32; 8];
        xs_group::<G, MW>(xs, k, q, gi, &mut xsg);
        unroll!(2, NB, |nb| {
            let i = nb * NR + q;
            let mut av = acc[i];
            unroll!(8, MW, |r| av[r] = math::fma(sc[nb], math::fma(-CORR, xsg[r], c[i][r]), av[r]));
            acc[i] = av;
        });
    });
}

/// Work-group = S sub-groups (one K range each) x NB 16-row blocks; grid (N / (16 NB)). red: dynamic SLM of
/// S * NB * MW * NR * 16 floats (S > 1). The chunk of the next group is loaded while the current one is decoded (two
/// register buffers, loop unrolled by two).
#[inline(always)]
unsafe fn xq_mv<const G: usize, const MW: usize, const NR: usize, const NB: usize>(
    xt: *const u16,
    xs: *const f32,
    wt: *const u32,
    y: *mut u8,
    k: u32,
    m: u32,
    y_off: u32,
    y_stride: u32,
    f32out: u32,
    red: *mut f32,
) {
    let red = attr::local_arg(red);
    let (lane, s, ns) = (id::lane() as usize, id::sg_id(), id::num_sg());
    let ng = k / G as u32;
    let (g0, g1) = (s * ng / ns, (s + 1) * ng / ns);
    let rb0 = id::group_id(0) * NB as u32;
    let mut acc = [F32x8::splat(0.0); 4];
    let mut wa = [0u32; 64];
    let mut wb = [0u32; 64];
    let mut sa = [0f32; 2];
    let mut sb = [0f32; 2];
    if g0 < g1 {
        ld_chunk::<G, NB>(wt, ng, rb0, g0, &mut wa, &mut sa);
    }
    let mut gi = g0;
    while gi < g1 {
        if gi + 1 < g1 {
            ld_chunk::<G, NB>(wt, ng, rb0, gi + 1, &mut wb, &mut sb);
        }
        grp::<G, MW, NR, NB>(&wa, &sa, xt, xs, k, gi, &mut acc);
        if gi + 1 >= g1 {
            break;
        }
        if gi + 2 < g1 {
            ld_chunk::<G, NB>(wt, ng, rb0, gi + 2, &mut wa, &mut sa);
        }
        grp::<G, MW, NR, NB>(&wb, &sb, xt, xs, k, gi + 1, &mut acc);
        gi += 2;
    }
    let mt = MW * NR;
    let store = |rr: usize, nb: usize, v: f32| {
        let at = (rr as u64 * y_stride as u64 + (y_off + (rb0 + nb as u32) * 16 + lane as u32) as u64) as usize;
        if f32out != 0 {
            *(y as *mut f32).add(at) = v;
        } else {
            *(y as *mut u16).add(at) = to_bf(v);
        }
    };
    if ns > 1 {
        unroll!(2, NB, |nb| {
            unroll!(2, NR, |q| {
                let av = acc[nb * NR + q];
                unroll!(8, MW, |r| *red.add(((s as usize * NB + nb) * mt + q * MW + r) * 16 + lane) = av[r]);
            })
        });
        slm::barrier();
        let mut p = s as usize;
        while p < NB * mt {
            let (nb, rr) = (p / mt, p % mt);
            if rr < m as usize {
                let mut v = 0.0f32;
                let mut j = 0;
                while j < ns as usize {
                    v += *red.add(((j * NB + nb) * mt + rr) * 16 + lane);
                    j += 1;
                }
                store(rr, nb, v);
            }
            p += ns as usize;
        }
    } else {
        unroll!(2, NB, |nb| {
            unroll!(2, NR, |q| {
                let av = acc[nb * NR + q];
                unroll!(8, MW, |r| {
                    let rr = q * MW + r;
                    if rr < m as usize {
                        store(rr, nb, av[r]);
                    }
                });
            })
        });
    }
}

/// [`xq_mv`] for NR row groups with one sub-group per (K split, row group): sub-group q * S + s runs split s of row group q
/// (grp with NR = 1 on that row group's A and xs). Each row group's DPAS chains, K ranges and the SLM reduction (same layout,
/// splits summed in order) are those of xq_mv's NR form, so the output is bit-identical to it, with NR times the threads
/// (B decoded once per sub-group). For 16-row windows on low-split shapes (gate / up: split 1), where xq_mv 8x2 runs short of
/// threads to hide latency. Grid (N / (16 NB)), local NR S 16; red: S NB MW NR 16 floats.
#[inline(always)]
unsafe fn xq_mv_p<const G: usize, const MW: usize, const NR: usize, const NB: usize>(
    xt: *const u16,
    xs: *const f32,
    wt: *const u32,
    y: *mut u8,
    k: u32,
    m: u32,
    y_off: u32,
    y_stride: u32,
    f32out: u32,
    red: *mut f32,
) {
    let red = attr::local_arg(red);
    let (lane, sg_all, ns_all) = (id::lane() as usize, id::sg_id(), id::num_sg());
    let ns = ns_all / NR as u32;
    let (q, s) = (sg_all / ns, sg_all % ns);
    let xtq = xt.add((q as u64 * (k / 16) as u64 * MW as u64 * 16) as usize);
    let xsq = xs.add((q as u64 * (k / 32) as u64 * MW as u64) as usize);
    let ng = k / G as u32;
    let (g0, g1) = (s * ng / ns, (s + 1) * ng / ns);
    let rb0 = id::group_id(0) * NB as u32;
    let mut acc = [F32x8::splat(0.0); 4];
    let mut wa = [0u32; 64];
    let mut wb = [0u32; 64];
    let mut sa = [0f32; 2];
    let mut sb = [0f32; 2];
    if g0 < g1 {
        ld_chunk::<G, NB>(wt, ng, rb0, g0, &mut wa, &mut sa);
    }
    let mut gi = g0;
    while gi < g1 {
        if gi + 1 < g1 {
            ld_chunk::<G, NB>(wt, ng, rb0, gi + 1, &mut wb, &mut sb);
        }
        grp::<G, MW, 1, NB>(&wa, &sa, xtq, xsq, k, gi, &mut acc);
        if gi + 1 >= g1 {
            break;
        }
        if gi + 2 < g1 {
            ld_chunk::<G, NB>(wt, ng, rb0, gi + 2, &mut wa, &mut sa);
        }
        grp::<G, MW, 1, NB>(&wb, &sb, xtq, xsq, k, gi + 1, &mut acc);
        gi += 2;
    }
    let mt = MW * NR;
    unroll!(2, NB, |nb| {
        let av = acc[nb];
        unroll!(8, MW, |r| *red.add(((s as usize * NB + nb) * mt + q as usize * MW + r) * 16 + lane) = av[r]);
    });
    slm::barrier();
    let mut p = sg_all as usize;
    while p < NB * mt {
        let (nb, rr) = (p / mt, p % mt);
        if rr < m as usize {
            let mut v = 0.0f32;
            let mut j = 0;
            while j < ns as usize {
                v += *red.add(((j * NB + nb) * mt + rr) * 16 + lane);
                j += 1;
            }
            let at = (rr as u64 * y_stride as u64 + (y_off + (rb0 + nb as u32) * 16 + lane as u32) as u64) as usize;
            if f32out != 0 {
                *(y as *mut f32).add(at) = v;
            } else {
                *(y as *mut u16).add(at) = to_bf(v);
            }
        }
        p += ns_all as usize;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_mv_128_8x2p_1(
    xt: *const u16,
    xs: *const f32,
    w: *const u32,
    y: *mut u8,
    k: u32,
    _n: u32,
    m: u32,
    y_off: u32,
    y_stride: u32,
    f32out: u32,
    red: *mut f32,
) {
    xq_mv_p::<128, 8, 2, 1>(xt, xs, w, y, k, m, y_off, y_stride, f32out, red)
}

macro_rules! xqmv {
    ($name:ident, $g:literal, $mw:literal, $nr:literal, $nb:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            xt: *const u16,
            xs: *const f32,
            w: *const u32,
            y: *mut u8,
            k: u32,
            _n: u32,
            m: u32,
            y_off: u32,
            y_stride: u32,
            f32out: u32,
            red: *mut f32,
        ) {
            xq_mv::<$g, $mw, $nr, $nb>(xt, xs, w, y, k, m, y_off, y_stride, f32out, red)
        }
    };
}

xqmv!(xq_mv_32_1x1_1, 32, 1, 1, 1);
xqmv!(xq_mv_32_2x1_1, 32, 2, 1, 1);
xqmv!(xq_mv_32_4x1_1, 32, 4, 1, 1);
xqmv!(xq_mv_32_8x1_1, 32, 8, 1, 1);
xqmv!(xq_mv_32_8x2_1, 32, 8, 2, 1);
xqmv!(xq_mv_32_1x1_2, 32, 1, 1, 2);
xqmv!(xq_mv_32_2x1_2, 32, 2, 1, 2);
xqmv!(xq_mv_32_4x1_2, 32, 4, 1, 2);
xqmv!(xq_mv_32_8x1_2, 32, 8, 1, 2);
xqmv!(xq_mv_32_8x2_2, 32, 8, 2, 2);
xqmv!(xq_mv_64_1x1_1, 64, 1, 1, 1);
xqmv!(xq_mv_64_2x1_1, 64, 2, 1, 1);
xqmv!(xq_mv_64_4x1_1, 64, 4, 1, 1);
xqmv!(xq_mv_64_8x1_1, 64, 8, 1, 1);
xqmv!(xq_mv_64_8x2_1, 64, 8, 2, 1);
xqmv!(xq_mv_64_1x1_2, 64, 1, 1, 2);
xqmv!(xq_mv_64_2x1_2, 64, 2, 1, 2);
xqmv!(xq_mv_64_4x1_2, 64, 4, 1, 2);
xqmv!(xq_mv_64_8x1_2, 64, 8, 1, 2);
xqmv!(xq_mv_64_8x2_2, 64, 8, 2, 2);
xqmv!(xq_mv_128_1x1_1, 128, 1, 1, 1);
xqmv!(xq_mv_128_2x1_1, 128, 2, 1, 1);
xqmv!(xq_mv_128_4x1_1, 128, 4, 1, 1);
xqmv!(xq_mv_128_8x1_1, 128, 8, 1, 1);
xqmv!(xq_mv_128_8x2_1, 128, 8, 2, 1);
xqmv!(xq_mv_128_1x1_2, 128, 1, 1, 2);
xqmv!(xq_mv_128_2x1_2, 128, 2, 1, 2);
xqmv!(xq_mv_128_4x1_2, 128, 4, 1, 2);
xqmv!(xq_mv_128_8x1_2, 128, 8, 1, 2);
xqmv!(xq_mv_128_8x2_2, 128, 8, 2, 2);

/// GDN in_proj_b + in_proj_a (48 channels each, G = 128, NB = 1) and kx-core gdn_gates_r in one launch (decode fusion F2).
/// Work-group = 2 S sub-groups for one 16-channel block: sub-groups 0 .. S compute b's block, S .. 2S a's block, each half
/// exactly as [`xq_mv`] with S sub-groups (same K ranges, same SLM reduction order), so wb / wa are bit-identical; then for
/// each window row and channel h: beta = 1 / (1 + exp(-b)), g = exp(-exp(A_log) * softplus(a + dt_bias)) on the bf16 b / a, the
/// expressions of gdn_gates_r. Writes wb / wa (bf16 [m][48]; the replay inputs) and beta / g (fp32 [m][48]).
/// Grid (3), local 32 S; red: dynamic SLM of 2 S MW NR 16 floats.
#[inline(always)]
unsafe fn xq_mv_ba<const MW: usize, const NR: usize>(
    xt: *const u16,
    xs: *const f32,
    wbp: *const u32,
    wap: *const u32,
    yb: *mut u16,
    ya: *mut u16,
    k: u32,
    m: u32,
    red: *mut f32,
    a_log: *const f32,
    dt_bias: *const f32,
    beta_o: *mut f32,
    g_o: *mut f32,
) {
    const G: usize = 128;
    const NB: usize = 1;
    let red = attr::local_arg(red);
    let fin = slm::alloc::<f32, { 2 * 16 * 16 }, 2>(); // [half][row][lane]: the reduced b / a values
    let (lane, sg_all, ns_all) = (id::lane() as usize, id::sg_id(), id::num_sg());
    let ns = ns_all / 2;
    let half = sg_all / ns; // 0: b, 1: a
    let s = sg_all % ns;
    let wt = if half == 0 { wbp } else { wap };
    let ng = k / G as u32;
    let (g0, g1) = (s * ng / ns, (s + 1) * ng / ns);
    let rb0 = id::group_id(0) * NB as u32;
    let mut acc = [F32x8::splat(0.0); 4];
    let mut wa = [0u32; 64];
    let mut wb = [0u32; 64];
    let mut sa = [0f32; 2];
    let mut sb = [0f32; 2];
    if g0 < g1 {
        ld_chunk::<G, NB>(wt, ng, rb0, g0, &mut wa, &mut sa);
    }
    let mut gi = g0;
    while gi < g1 {
        if gi + 1 < g1 {
            ld_chunk::<G, NB>(wt, ng, rb0, gi + 1, &mut wb, &mut sb);
        }
        grp::<G, MW, NR, NB>(&wa, &sa, xt, xs, k, gi, &mut acc);
        if gi + 1 >= g1 {
            break;
        }
        if gi + 2 < g1 {
            ld_chunk::<G, NB>(wt, ng, rb0, gi + 2, &mut wa, &mut sa);
        }
        grp::<G, MW, NR, NB>(&wb, &sb, xt, xs, k, gi + 1, &mut acc);
        gi += 2;
    }
    let mt = MW * NR;
    let rh = red.add(half as usize * ns as usize * NB * mt * 16);
    unroll!(2, NR, |q| {
        let av = acc[q];
        unroll!(8, MW, |r| *rh.add(((s as usize * NB) * mt + q * MW + r) * 16 + lane) = av[r]);
    });
    slm::barrier();
    let y = if half == 0 { yb } else { ya };
    let mut p = s as usize;
    while p < NB * mt {
        let rr = p % mt;
        if rr < m as usize {
            let mut v = 0.0f32;
            let mut j = 0;
            while j < ns as usize {
                v += *rh.add((j * NB * mt + rr) * 16 + lane);
                j += 1;
            }
            let o = to_bf(v);
            *y.add(rr * 48 + (rb0 as usize) * 16 + lane) = o;
            *fin.add((half as usize * 16 + rr) * 16 + lane) = bf(o);
        }
        p += ns as usize;
    }
    slm::barrier();
    // gdn_gates_r for this block's 16 channels: sub-group j of the b half takes rows j, j + S, ..
    if half == 0 {
        let h = rb0 as usize * 16 + lane;
        let mut rr = s as usize;
        while rr < m as usize {
            let bv = *fin.add(rr * 16 + lane);
            let avv = *fin.add((16 + rr) * 16 + lane);
            *beta_o.add(rr * 48 + h) = math::div_ocl(1.0, 1.0 + math::exp(-bv));
            let av = avv + *dt_bias.add(h);
            let sp = if av > 20.0 { av } else { math::log1p(math::exp(av)) };
            *g_o.add(rr * 48 + h) = math::exp(-math::exp(*a_log.add(h)) * sp);
            rr += ns as usize;
        }
    }
}

macro_rules! xqmv_ba {
    ($name:ident, $mw:literal, $nr:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            xt: *const u16,
            xs: *const f32,
            wb: *const u32,
            wa: *const u32,
            yb: *mut u16,
            ya: *mut u16,
            k: u32,
            m: u32,
            red: *mut f32,
            a_log: *const f32,
            dt_bias: *const f32,
            beta_o: *mut f32,
            g_o: *mut f32,
        ) {
            xq_mv_ba::<$mw, $nr>(xt, xs, wb, wa, yb, ya, k, m, red, a_log, dt_bias, beta_o, g_o)
        }
    };
}
xqmv_ba!(xq_mv_ba_1x1, 1, 1);
xqmv_ba!(xq_mv_ba_2x1, 2, 1);
xqmv_ba!(xq_mv_ba_4x1, 4, 1);
xqmv_ba!(xq_mv_ba_8x1, 8, 1);
xqmv_ba!(xq_mv_ba_8x2, 8, 2);

// ---- prompt-window path (arc-engine xq_pfd.cl): weight decode to the pre-scaled fp16 scratch, activation prep ----------

/// XQ prefill weight decode to the pre-scaled fp16 scratch W16 [N][K] (xq_pfd.cl `xq_pfdec16x_128`, the f16out values of
/// xq_pf.cl's xq_pfdec_G bit for bit) with coalesced 16-byte stores: lane l decodes k 8l .. 8l + 7 of all 16 channels of the
/// block (its low word l and high word 16 + hi(l), 64 contiguous bytes each) and the 16 lanes store 256 contiguous bytes
/// of a row at a time. w16 = fp16(float(u - 2^(b-1)) * scale). Grid (N / 16, K / 128 / 8), local (16, 8). G = 128 only.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfdec16x_128(w: *const u32, w16: *mut u16, k: u32, _n: u32) {
    use kxcommon::mem::global::vload16_u32;
    const D: usize = 128 * BITS / 32;
    let (lane, rb, gi) = (id::lane(), id::group_id(0), id::global_id(1) as u32);
    let ng = k / 128;
    let cp = w.add(((rb as u64 * ng as u64 + gi as u64) * (16 * D as u64 + 8)) as usize);
    let l = lane as usize;
    // BITS 3: 2-bit plane word l/2 (bits 16e + 8(l%2) + 2(i/2)), 1-bit plane word 8 + l/4 (bit 16e + 8((l/2)%2) + 4(l%2) + i/2)
    let lo = if BITS == 3 { vload16_u32(l / 2, cp) } else { vload16_u32(l, cp) };
    let hi = match BITS {
        3 => vload16_u32(8 + l / 4, cp),
        5 => vload16_u32(16 + l / 4, cp),
        6 => vload16_u32(16 + l / 2, cp),
        8 => vload16_u32(16 + l, cp),
        _ => U32x16::splat(0),
    };
    let lsh = 8 * (lane & 1);
    let hsh = match BITS {
        3 | 5 => 8 * ((lane >> 1) & 1) + 4 * (lane & 1),
        6 => 8 * (lane & 1),
        _ => 0,
    };
    // the 16 fp16 scales as 8 dwords: one 32-byte message (a ushort vload16 became eight dword loads)
    let scw = kxcommon::mem::global::vload8_u32(0, cp.add(16 * D));
    let off = 1i32 << (BITS - 1);
    let o = w16.add((((rb * 16) as u64) * k as u64) as usize).add((gi * 128 + lane * 8) as usize);
    unroll!(16, 16, |c| {
        let sf = cvt::f16_bits_to_f32((scw[c / 2] >> (16 * (c % 2))) as u16);
        let mut v = [0u16; 8];
        unroll!(8, 8, |i| {
            let i = i as u32;
            let u = if BITS == 3 {
                ((lo[c] >> (16 * (i % 2) + lsh + 2 * (i / 2))) & 3) | (((hi[c] >> (16 * (i % 2) + hsh + i / 2)) & 1) << 2)
            } else {
                let mut u = (lo[c] >> (4 * (i / 2) + 16 * (i % 2))) & 15;
                if BITS == 5 {
                    u |= ((hi[c] >> (16 * (i % 2) + hsh + i / 2)) & 1) << 4;
                } else if BITS == 6 {
                    u |= ((hi[c] >> (16 * (i % 2) + hsh + 2 * (i / 2))) & 3) << 4;
                } else if BITS == 8 {
                    u |= ((hi[c] >> (16 * (i % 2) + 4 * (i / 2))) & 15) << 4;
                }
                u
            };
            v[i as usize] = cvt::f16_bits_rte(((u as i32 - off) as f32) * sf);
        });
        // one 16-byte store a lane (vstore8 on a ushort pointer would split into four 4-byte stores)
        *(o.add(c * k as usize) as *mut U32x4) = bits::<U16x8, U32x4>(U16x8::from_array(v));
    });
}

/// The rotated values of row `row` at k = blk * 128 + lane + 16 i (xq_prep's arithmetic, xq_pfd.cl `rot_block`).
#[inline(always)]
unsafe fn rot_block(x: *const u8, xdt: u32, xstride: u32, su: *const u16, rot: u32, row: u32, m: u32, blk_: u32, lane: u32) -> [f32; 8] {
    let k0 = blk_ * 128 + lane;
    let mut v = [0f32; 8];
    unroll!(8, 8, |i| {
        let a = if row < m { ld(x, xdt, (row as u64 * xstride as u64 + (k0 + 16 * i as u32) as u64) as usize) } else { 0.0 };
        v[i] = if rot != 0 { a * cvt::f16_bits_to_f32(*su.add(k0 as usize + 16 * i)) } else { a };
    });
    if rot != 0 {
        fwht8(&mut v, lane);
        unroll!(8, 8, |i| v[i] *= HAD_SCALE);
    }
    v
}

/// Prepared rows, k-major for the VNNI-transform 2D loads of the prompt GEMM (xq_pfd.cl `xq_pfprep_k16`): xt [K][Rp] fp16
/// (x~ rounded as the decode prepass) or int8 (round(x~ * 127 / amax) per row and 128 block, sa [Rp][K/128] = amax / 127).
/// A work-group of 32 sub-groups rotates 32 rows of one 128 block into SLM and writes 128 k x 32 rows as contiguous runs
/// (one 16-byte store a lane, 8 bytes for int8). Rows >= M are zero. Grid (K/128, Rp/32), local 512; Rp a multiple of 32.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfprep_k16(
    x: *const u8,
    xdt: u32,
    xstride: u32,
    su: *const u16,
    rot: u32,
    xt: *mut u8,
    sa: *mut f32,
    k: u32,
    m: u32,
    rp: u32,
    i8: u32,
) {
    use kxcommon::mem::local::{vload8_u16, vload8_u8};
    attr::reqd_work_group_size::<512, 1, 1>();
    let tile = slm::alloc::<u16, 5120, 0>(); // fp16 [k][32 rows + 8]; int8 uses it as bytes [k][32 rows + 16]
    let t8 = tile as *mut u8;
    let (lane, sgi) = (id::lane(), id::sg_id());
    let (row, blk_) = (id::group_id(1) * 32 + sgi, id::group_id(0));
    let v = rot_block(x, xdt, xstride, su, rot, row, m, blk_, lane);
    if i8 != 0 {
        let mut mx = 0.0f32;
        unroll!(8, 8, |i| mx = math::fmax(mx, math::fabs(v[i])));
        mx = sg::reduce_max_f32(mx);
        let inv = if mx > 0.0 { math::div_ocl(127.0, mx) } else { 0.0 };
        unroll!(8, 8, |i| *t8.add(((lane + 16 * i as u32) * 48 + sgi) as usize) = cvt::i8_rte(v[i] * inv) as u8);
        if lane == 0 {
            *sa.add((row as u64 * (k / 128) as u64 + blk_ as u64) as usize) = math::div_ocl(mx, 127.0);
        }
    } else {
        unroll!(8, 8, |i| *tile.add(((lane + 16 * i as u32) * 40 + sgi) as usize) = cvt::f16_bits_rte(v[i]));
    }
    slm::barrier();
    let t = id::local_id(0);
    let (kk, part) = (t >> 2, t & 3); // 128 k x 4 runs of 8 rows
    let o = ((blk_ * 128 + kk) as u64 * rp as u64 + (id::group_id(1) * 32 + part * 8) as u64) as usize;
    if i8 != 0 {
        *(xt.add(o) as *mut U32x2) = bits::<U8x8, U32x2>(vload8_u8(0, t8.add((kk * 48 + part * 8) as usize)));
    } else {
        *((xt as *mut u16).add(o) as *mut U32x4) = bits::<U16x8, U32x4>(vload8_u16(0, tile.add((kk * 40 + part * 8) as usize)));
    }
}

/// Code u of element k of a group from a lane's D dwords (xq_pf.cl `code_u`).
#[inline(always)]
fn code_u<const G: usize>(w: &[u32], k: usize) -> u32 {
    let (t, d, e) = (k / 16, (k % 16) / 2, k % 2);
    if BITS == 3 {
        return ((w[t] >> (16 * e + 2 * d)) & 3) | (((w[G / 16 + t / 2] >> (16 * e + 8 * (t % 2) + d)) & 1) << 2);
    }
    let mut u = (w[2 * t + d / 4] >> (4 * (d % 4) + 16 * e)) & 15;
    if BITS == 5 {
        u |= ((w[G / 8 + t / 2] >> (16 * e + 8 * (t % 2) + d)) & 1) << 4;
    }
    if BITS == 6 {
        u |= ((w[G / 8 + t] >> (16 * e + 2 * d)) & 3) << 4;
    }
    if BITS == 8 {
        u |= ((w[G / 8 + 2 * t + d / 4] >> (16 * e + 4 * (d % 4))) & 15) << 4;
    }
    u
}

/// XQ prefill weight decode (xq_pf.cl `xq_pfdec_G`): w8 [N][K] int8 codes q (row-major) + sw [K/G][N] fp32 scales, or with
/// f16out w16 [N][K] = fp16(q * s). A lane decodes its channel's group into SLM, then every store writes one row's
/// contiguous run. Grid (N/16, K/G), local 16.
#[inline(always)]
unsafe fn pfdec<const G: usize>(wt: *const u32, wq: *mut u32, sw: *mut f32, k: u32, n: u32, f16out: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    let tile = slm::alloc::<u32, { 16 * 65 }, 0>(); // 16 * (G / 2 + 1) dwords at most (G = 128)
    let d_n = lane_dwords(G);
    let (lane, rb, gi) = (id::lane(), id::group_id(0), id::group_id(1));
    let ng = k / G as u32;
    let cp = wt.add(((rb as u64 * ng as u64 + gi as u64) * (16 * d_n as u64 + 8)) as usize);
    let mut w = [0u32; 32];
    unroll!(32, d_n, |j| w[j] = *cp.add(j * 16 + lane as usize));
    let sbits = *(cp.add(16 * d_n) as *const u16).add(lane as usize);
    let nn = rb * 16 + lane;
    *sw.add((gi as u64 * n as u64 + nn as u64) as usize) = cvt::f16_bits_to_f32(sbits);
    let off = 1i32 << (BITS - 1);
    let row = G / 2 + 1;
    let dw = if f16out != 0 { G / 2 } else { G / 4 };
    if f16out != 0 {
        let sf = cvt::f16_bits_to_f32(sbits);
        unroll!(64, G / 2, |h| {
            let a = cvt::f16_bits_rte(((code_u::<G>(&w, 2 * h) as i32 - off) as f32) * sf);
            let b = cvt::f16_bits_rte(((code_u::<G>(&w, 2 * h + 1) as i32 - off) as f32) * sf);
            *tile.add(lane as usize * row + h) = a as u32 | ((b as u32) << 16);
        });
    } else {
        unroll!(32, G / 4, |q4| {
            let mut v = 0u32;
            unroll!(4, 4, |j| v |= (((code_u::<G>(&w, 4 * q4 + j) as i32 - off) as u32) & 0xFF) << (8 * j));
            *tile.add(lane as usize * row + q4) = v;
        });
    }
    slm::barrier();
    let per: u64 = if f16out != 0 { 2 } else { 4 }; // codes a dword
    let mut r = 0u32;
    while r < 16 {
        let o = wq.add((((rb * 16 + r) as u64 * k as u64 + (gi * G as u32) as u64) / per) as usize);
        let mut c = lane as usize;
        while c < dw {
            *o.add(c) = *tile.add(r as usize * row + c);
            c += 16;
        }
        r += 1;
    }
}

macro_rules! pfdec_g {
    ($name:ident, $g:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(w: *const u32, wq: *mut u32, sw: *mut f32, k: u32, n: u32, f16out: u32) {
            pfdec::<$g>(w, wq, sw, k, n, f16out)
        }
    };
}
pfdec_g!(xq_pfdec_32, 32);
pfdec_g!(xq_pfdec_64, 64);
pfdec_g!(xq_pfdec_128, 128);

/// m-row [`xq_add_rms_prep`]: row get_group_id(0) of x [m][5120]: x = bf16(x + d) when `add`, y = bf16(x * rsqrt(mean(x^2) + eps)
/// * w) (kx-core add_rmsnorm_r / rmsnorm_r: same operation order, same bits), then the prepass of y in xq_prep's MW-row-group
/// layout; rows m .. Mpad write xq_prep's padding blocks (0 * su, so -0.0 where su < 0, as xq_prep does). A row's bits do not
/// depend on m. Replaces add_rmsnorm_r + xq_prep (decode / verify windows, decode fusion F1). Grid (Mpad = MW * NR), local 512.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_add_rms_prep_r(
    x: *mut u16,
    d: *const u16,
    w: *const u16,
    y: *mut u16,
    eps: f32,
    add: u32,
    su: *const u16,
    rot: u32,
    xt: *mut u16,
    xs: *mut f32,
    m: u32,
    mw: u32,
) {
    attr::reqd_work_group_size::<512, 1, 1>();
    let part = slm::alloc::<f32, 4, 0>();
    let vals = slm::alloc::<f32, 5120, 1>();
    let row = id::group_id(0);
    let lid = id::local_id(0) as usize;
    let (lane, sgi) = (id::lane(), id::sg_id());
    if row >= m {
        // xq_prep's padding rows: a = 0, v = a * su (rot), the rotation, the stores
        let mut b = sgi;
        while b < 40 {
            let mut v = [0f32; 8];
            unroll!(8, 8, |i| {
                let kk = (b * 128 + lane) as usize + 16 * i;
                v[i] = if rot != 0 { 0.0 * cvt::f16_bits_to_f32(*su.add(kk)) } else { 0.0 };
            });
            prep_store(&mut v, rot, lane, row, b, xt, xs, 5120, mw);
            b += 32;
        }
        return;
    }
    let base = row as usize * 5120;
    let mut xa = [0u16; 10];
    let mut da = [0u16; 10];
    let mut wa = [0u16; 10];
    unroll!(16, 10, |k| {
        xa[k] = *x.add(base + lid + 512 * k);
        da[k] = if add != 0 { *d.add(base + lid + 512 * k) } else { 0 };
        wa[k] = *w.add(lid + 512 * k);
    });
    unroll!(16, 10, |k| {
        if add != 0 {
            let v = to_bf(bf(xa[k]) + bf(da[k]));
            *x.add(base + lid + 512 * k) = v;
            *vals.add(lid + 512 * k) = bf(v);
        } else {
            *vals.add(lid + 512 * k) = bf(xa[k]);
        }
    });
    slm::barrier();
    let mut ss = 0.0f32;
    if lid < 64 {
        let mut k = 0;
        while k < 80 {
            let f = *vals.add(lid + 64 * k);
            ss = math::fma(f, f, ss);
            k += 1;
        }
        ss = sg::reduce_add_f32(ss);
        if id::lane() == 0 {
            *part.add(id::sg_id() as usize) = ss;
        }
    }
    slm::barrier();
    let scale = math::rsqrt(math::div_ocl(*part + *part.add(1) + *part.add(2) + *part.add(3), 5120.0) + eps);
    let mut ya = [0u16; 10];
    unroll!(16, 10, |k| {
        ya[k] = to_bf(*vals.add(lid + 512 * k) * scale * bf(wa[k]));
        *y.add(base + lid + 512 * k) = ya[k];
    });
    slm::barrier();
    unroll!(16, 10, |k| *vals.add(lid + 512 * k) = bf(ya[k]));
    slm::barrier();
    let mut b = sgi;
    while b < 40 {
        let mut v = [0f32; 8];
        unroll!(8, 8, |i| {
            let kk = (b * 128 + lane) as usize + 16 * i;
            v[i] = if rot != 0 { *vals.add(kk) * cvt::f16_bits_to_f32(*su.add(kk)) } else { *vals.add(kk) };
        });
        prep_store(&mut v, rot, lane, row, b, xt, xs, 5120, mw);
        b += 32;
    }
}

// ---- X3: group-free int8 prompt path (per-channel int8 weights, per-token int8 activations) ---------------------------------
//
// For kx-xqpf `xq_pfgemm_i8c2_cCtT`: y[t][n] = float(sum_k W8[n][k] X8[k][t]) * sn[n] * st[t], the int32 sum exact.
// Weights: XQ codes q with fp16 group scales s_g, requantized per channel: W8 = sat_rte(q * s_g * invn), invn = 127 / (max_g |s_g|
// * 2^(BITS-1)) (an upper bound of max |q s_g|), sn = 1 / invn. Activations: the Hadamard-rotated prompt rows (the fp16 prepass) with
// one scale a token: tinv = 127 / max_k |x_k|, X8 = sat_rte(x * tinv), st = max / 127. Every rounding step reads its multiplier
// from memory (invn, tinv), so a host check recomputes W8 and X8 exactly.

/// The fp16 prompt prepass (`xq_pfprep_k16` with i8 = 0, same output bit for bit) that also writes each (row, 128-block)'s
/// largest |x| after rotation into sa [Rp][K/128] (for [`xq_pftok`]). Grid (K/128, Rp/32), local 512.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfprep_k16m(
    x: *const u8,
    xdt: u32,
    xstride: u32,
    su: *const u16,
    rot: u32,
    xt: *mut u16,
    sa: *mut f32,
    k: u32,
    m: u32,
    rp: u32,
) {
    use kxcommon::mem::local::vload8_u16;
    attr::reqd_work_group_size::<512, 1, 1>();
    let tile = slm::alloc::<u16, 5120, 0>(); // fp16 [k][32 rows + 8]
    let (lane, sgi) = (id::lane(), id::sg_id());
    let (row, blk_) = (id::group_id(1) * 32 + sgi, id::group_id(0));
    let v = rot_block(x, xdt, xstride, su, rot, row, m, blk_, lane);
    let mut mx = 0.0f32;
    unroll!(8, 8, |i| mx = math::fmax(mx, math::fabs(v[i])));
    mx = sg::reduce_max_f32(mx);
    if lane == 0 {
        *sa.add((row as u64 * (k / 128) as u64 + blk_ as u64) as usize) = mx;
    }
    unroll!(8, 8, |i| *tile.add(((lane + 16 * i as u32) * 40 + sgi) as usize) = cvt::f16_bits_rte(v[i]));
    slm::barrier();
    let t = id::local_id(0);
    let (kk, part) = (t >> 2, t & 3);
    let o = ((blk_ * 128 + kk) as u64 * rp as u64 + (id::group_id(1) * 32 + part * 8) as u64) as usize;
    *(xt.add(o) as *mut U32x4) = bits::<U16x8, U32x4>(vload8_u16(0, tile.add((kk * 40 + part * 8) as usize)));
}

/// Per-token scale from the block maxima of [`xq_pfprep_k16m`]: st[t] = max / 127 (the GEMM's token scale), tinv[t] = 127 / max
/// (0 for an all-zero row). Grid (Rp/16), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pftok(sa: *const f32, st: *mut f32, tinv: *mut f32, kb: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    let t = id::group_id(0) * 16 + id::lane();
    let p = sa.add((t as u64 * kb as u64) as usize);
    let mut mx = 0.0f32;
    let mut g = 0u32;
    while g < kb {
        mx = math::fmax(mx, *p.add(g as usize));
        g += 1;
    }
    *st.add(t as usize) = math::div_ocl(mx, 127.0);
    *tinv.add(t as usize) = if mx > 0.0 { math::div_ocl(127.0, mx) } else { 0.0 };
}

/// fp16 Xt [K][Rp] -> int8 X8 [K][Rp] with the token's tinv: X8 = sat_rte(x * tinv). A lane: 8 consecutive tokens, 32 k.
/// Grid (ceil(Rp/128), K/32), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfq8t(xt: *const u16, tinv: *const f32, x8: *mut u8, rp: u32) {
    use kxcommon::mem::global::{vload8_f32, vload8_u16};
    attr::reqd_work_group_size::<16, 1, 1>();
    let t0 = id::group_id(0) * 128 + id::lane() * 8;
    let k0 = id::group_id(1) * 32;
    if t0 >= rp {
        return; // Rp is a multiple of 64
    }
    let inv = vload8_f32(0, tinv.add(t0 as usize)).to_array();
    let mut kk = 0u32;
    while kk < 32 {
        let o = ((k0 + kk) as u64 * rp as u64 + t0 as u64) as usize;
        let h = vload8_u16(0, xt.add(o)).to_array();
        let mut b = [0u8; 8];
        unroll!(8, 8, |j| b[j] = cvt::i8_sat_rte(cvt::f16_bits_to_f32(h[j]) * inv[j]) as u8);
        *(x8.add(o) as *mut U32x2) = bits::<U8x8, U32x2>(U8x8::from_array(b));
        kk += 1;
    }
}

/// Per-channel int8 scale of an XQ tensor (G = 128): sn[n] = m / 127 and invn[n] = 127 / m, m = max_g |s_g| * 2^(BITS-1) (0 -> 0).
/// Grid (N/16), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfscale_c(wt: *const u32, sn: *mut f32, invn: *mut f32, k: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    const G: usize = 128;
    let d_n = lane_dwords(G);
    let (lane, rb) = (id::lane(), id::group_id(0));
    let ng = k / G as u32;
    let mut mx = 0.0f32;
    let mut gi = 0u32;
    while gi < ng {
        let cp = wt.add(((rb as u64 * ng as u64 + gi as u64) * (16 * d_n as u64 + 8)) as usize);
        mx = math::fmax(mx, math::fabs(cvt::f16_bits_to_f32(*(cp.add(16 * d_n) as *const u16).add(lane as usize))));
        gi += 1;
    }
    let m = mx * (1u32 << (BITS - 1)) as f32;
    let nn = (rb * 16 + lane) as usize;
    *sn.add(nn) = math::div_ocl(m, 127.0);
    *invn.add(nn) = if m > 0.0 { math::div_ocl(127.0, m) } else { 0.0 };
}

/// XQ weights (G = 128) -> per-channel int8 W8 [N][K]: W8 = sat_rte(float(q) * s_g * invn[n]) (q * s_g exact in fp32). The
/// [`pfdec`] structure (a lane decodes its channel's group into SLM, then each store writes one row's run). Grid (N/16, K/128),
/// local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfdec_c128(wt: *const u32, wq: *mut u32, invn: *const f32, k: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    const G: usize = 128;
    let tile = slm::alloc::<u32, { 16 * 65 }, 0>();
    let d_n = lane_dwords(G);
    let (lane, rb, gi) = (id::lane(), id::group_id(0), id::group_id(1));
    let ng = k / G as u32;
    let cp = wt.add(((rb as u64 * ng as u64 + gi as u64) * (16 * d_n as u64 + 8)) as usize);
    let mut w = [0u32; 32];
    unroll!(32, d_n, |j| w[j] = *cp.add(j * 16 + lane as usize));
    let sf = cvt::f16_bits_to_f32(*(cp.add(16 * d_n) as *const u16).add(lane as usize));
    let iv = *invn.add((rb * 16 + lane) as usize);
    let off = 1i32 << (BITS - 1);
    let row = G / 2 + 1;
    unroll!(32, G / 4, |q4| {
        let mut v = 0u32;
        unroll!(4, 4, |j| {
            let q = (code_u::<G>(&w, 4 * q4 + j) as i32 - off) as f32;
            v |= ((cvt::i8_sat_rte(q * sf * iv) as u8) as u32) << (8 * j);
        });
        *tile.add(lane as usize * row + q4) = v;
    });
    slm::barrier();
    let dw = G / 4;
    let mut r = 0u32;
    while r < 16 {
        let o = wq.add((((rb * 16 + r) as u64 * k as u64 + (gi * G as u32) as u64) / 4) as usize);
        let mut c = lane as usize;
        while c < dw {
            *o.add(c) = *tile.add(r as usize * row + c);
            c += 16;
        }
        r += 1;
    }
}

/// Diagnostics (X3 KL attribution, not an engine path): sa [Rp][kb] = st[t] for every group (per-token activation scales in the
/// grouped GEMM's layout). Grid (Rp/16), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pf_fill_tok(sa: *mut f32, st: *const f32, kb: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    let t = id::group_id(0) * 16 + id::lane();
    let v = *st.add(t as usize);
    let mut g = 0u32;
    while g < kb {
        *sa.add((t as u64 * kb as u64 + g as u64) as usize) = v;
        g += 1;
    }
}

/// Diagnostics: sw [kb][N] = sn[n] for every group (per-channel weight scales in the grouped GEMM's layout). Grid (N/16, kb),
/// local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pf_fill_chan(sw: *mut f32, sn: *const f32, n: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    let nn = id::group_id(0) * 16 + id::lane();
    *sw.add((id::group_id(1) as u64 * n as u64 + nn as u64) as usize) = *sn.add(nn as usize);
}

/// [`xq_pfdec_c128`]'s values (W8 = sat_rte(float(q) * s_g * invn[n]), bit for bit) in [`xq_pfdec16x_128`]'s structure: lane l
/// decodes k 8l .. 8l + 7 of the block's 16 channels and stores 8 bytes a channel row, the 16 lanes 128 contiguous bytes; 8
/// sub-groups a work-group (8 groups). Grid (N / 16, K / 128 / 8), local (16, 8).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfdec16x_c128(w: *const u32, w8: *mut u8, invn: *const f32, k: u32, _n: u32) {
    use kxcommon::mem::global::{vload16_u32, vload8_f32};
    const D: usize = 128 * BITS / 32;
    let (lane, rb, gi) = (id::lane(), id::group_id(0), id::global_id(1) as u32);
    let ng = k / 128;
    let cp = w.add(((rb as u64 * ng as u64 + gi as u64) * (16 * D as u64 + 8)) as usize);
    let l = lane as usize;
    let lo = if BITS == 3 { vload16_u32(l / 2, cp) } else { vload16_u32(l, cp) };
    let hi = match BITS {
        3 => vload16_u32(8 + l / 4, cp),
        5 => vload16_u32(16 + l / 4, cp),
        6 => vload16_u32(16 + l / 2, cp),
        8 => vload16_u32(16 + l, cp),
        _ => U32x16::splat(0),
    };
    let lsh = 8 * (lane & 1);
    let hsh = match BITS {
        3 | 5 => 8 * ((lane >> 1) & 1) + 4 * (lane & 1),
        6 => 8 * (lane & 1),
        _ => 0,
    };
    let scw = kxcommon::mem::global::vload8_u32(0, cp.add(16 * D));
    let iva = vload8_f32(0, invn.add((rb * 16) as usize)).to_array();
    let ivb = vload8_f32(1, invn.add((rb * 16) as usize)).to_array();
    let off = 1i32 << (BITS - 1);
    let o = w8.add((((rb * 16) as u64) * k as u64) as usize).add((gi * 128 + lane * 8) as usize);
    unroll!(16, 16, |c| {
        let sf = cvt::f16_bits_to_f32((scw[c / 2] >> (16 * (c % 2))) as u16);
        let iv = if c < 8 { iva[c % 8] } else { ivb[c % 8] };
        let mut v = [0u8; 8];
        unroll!(8, 8, |i| {
            let i = i as u32;
            let u = if BITS == 3 {
                ((lo[c] >> (16 * (i % 2) + lsh + 2 * (i / 2))) & 3) | (((hi[c] >> (16 * (i % 2) + hsh + i / 2)) & 1) << 2)
            } else {
                let mut u = (lo[c] >> (4 * (i / 2) + 16 * (i % 2))) & 15;
                if BITS == 5 {
                    u |= ((hi[c] >> (16 * (i % 2) + hsh + i / 2)) & 1) << 4;
                } else if BITS == 6 {
                    u |= ((hi[c] >> (16 * (i % 2) + hsh + 2 * (i / 2))) & 3) << 4;
                } else if BITS == 8 {
                    u |= ((hi[c] >> (16 * (i % 2) + 4 * (i / 2))) & 15) << 4;
                }
                u
            };
            v[i as usize] = cvt::i8_sat_rte(((u as i32 - off) as f32) * sf * iv) as u8;
        });
        *(o.add(c * k as usize) as *mut U32x2) = bits::<U8x8, U32x2>(U8x8::from_array(v));
    });
}

/// Diagnostics: per-(token, slice of `sb` 128-blocks) activation scales from the block maxima of [`xq_pfprep_k16m`], written per
/// block: sa_out[t][g] = max / 127, tinv[t][g] = 127 / max over g's slice. Grid (Rp/16), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pftok_s(sa: *const f32, sa_out: *mut f32, tinv: *mut f32, kb: u32, sb: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    let t = id::group_id(0) * 16 + id::lane();
    let base = (t as u64 * kb as u64) as usize;
    let mut g0 = 0u32;
    while g0 < kb {
        let g1 = (g0 + sb).min(kb);
        let mut mx = 0.0f32;
        let mut g = g0;
        while g < g1 {
            mx = math::fmax(mx, *sa.add(base + g as usize));
            g += 1;
        }
        let (sc, iv) = (math::div_ocl(mx, 127.0), if mx > 0.0 { math::div_ocl(127.0, mx) } else { 0.0 });
        g = g0;
        while g < g1 {
            *sa_out.add(base + g as usize) = sc;
            *tinv.add(base + g as usize) = iv;
            g += 1;
        }
        g0 = g1;
    }
}

/// Diagnostics: fp16 Xt [K][Rp] -> int8 with a multiplier per (token, 128-block): X8 = sat_rte(x * tinv[t][k / 128]). Grid
/// (ceil(Rp/128), K/32), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfq8s(xt: *const u16, tinv: *const f32, x8: *mut u8, rp: u32, kb: u32) {
    use kxcommon::mem::global::vload8_u16;
    attr::reqd_work_group_size::<16, 1, 1>();
    let t0 = id::group_id(0) * 128 + id::lane() * 8;
    let k0 = id::group_id(1) * 32;
    if t0 >= rp {
        return;
    }
    let g = (k0 / 128) as usize;
    let mut inv = [0f32; 8];
    unroll!(8, 8, |j| inv[j] = *tinv.add((t0 as usize + j) * kb as usize + g));
    let mut kk = 0u32;
    while kk < 32 {
        let o = ((k0 + kk) as u64 * rp as u64 + t0 as u64) as usize;
        let h = vload8_u16(0, xt.add(o)).to_array();
        let mut b = [0u8; 8];
        unroll!(8, 8, |j| b[j] = cvt::i8_sat_rte(cvt::f16_bits_to_f32(h[j]) * inv[j]) as u8);
        *(x8.add(o) as *mut U32x2) = bits::<U8x8, U32x2>(U8x8::from_array(b));
        kk += 1;
    }
}

// ---- X3: K-sliced int8 (per-(channel, slice) weight scales, per-(token, slice) activation scales) --------------------------
// Slices of `sb` 128-blocks; the scales are stored per block ([K/128][N] weights, [Rp][K/128] activations, the slice's value in
// each of its blocks) so the activation side is xq_pftok_s + xq_pfq8s and the GEMM reads block j * sb of slice j.

/// Per-(channel, slice) int8 weight scale: m = max |s_g| over the slice's groups * 2^(BITS-1), sn = m / 127, invn = 127 / m (0 ->
/// 0), written to each block of the slice: sn / invn [K/128][N]. Grid (N/16, ceil(K/128 / sb)), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfscale_s(wt: *const u32, sn: *mut f32, invn: *mut f32, k: u32, n: u32, sb: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    const G: usize = 128;
    let d_n = lane_dwords(G);
    let (lane, rb, j) = (id::lane(), id::group_id(0), id::group_id(1));
    let ng = k / G as u32;
    let (g0, g1) = (j * sb, ((j + 1) * sb).min(ng));
    let mut mx = 0.0f32;
    let mut gi = g0;
    while gi < g1 {
        let cp = wt.add(((rb as u64 * ng as u64 + gi as u64) * (16 * d_n as u64 + 8)) as usize);
        mx = math::fmax(mx, math::fabs(cvt::f16_bits_to_f32(*(cp.add(16 * d_n) as *const u16).add(lane as usize))));
        gi += 1;
    }
    let m = mx * (1u32 << (BITS - 1)) as f32;
    let (a, b) = (math::div_ocl(m, 127.0), if m > 0.0 { math::div_ocl(127.0, m) } else { 0.0 });
    let nn = (rb * 16 + lane) as u64;
    gi = g0;
    while gi < g1 {
        *sn.add((gi as u64 * n as u64 + nn) as usize) = a;
        *invn.add((gi as u64 * n as u64 + nn) as usize) = b;
        gi += 1;
    }
}

/// [`xq_pfdec16x_c128`] with the multiplier per (channel, block): W8 = sat_rte(float(q) * s_g * invn[g][n]). Grid (N / 16, K /
/// 128 / 8), local (16, 8).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfdec16x_s128(w: *const u32, w8: *mut u8, invn: *const f32, k: u32, n: u32) {
    use kxcommon::mem::global::{vload16_u32, vload8_f32};
    const D: usize = 128 * BITS / 32;
    let (lane, rb, gi) = (id::lane(), id::group_id(0), id::global_id(1) as u32);
    let ng = k / 128;
    let cp = w.add(((rb as u64 * ng as u64 + gi as u64) * (16 * D as u64 + 8)) as usize);
    let l = lane as usize;
    let lo = if BITS == 3 { vload16_u32(l / 2, cp) } else { vload16_u32(l, cp) };
    let hi = match BITS {
        3 => vload16_u32(8 + l / 4, cp),
        5 => vload16_u32(16 + l / 4, cp),
        6 => vload16_u32(16 + l / 2, cp),
        8 => vload16_u32(16 + l, cp),
        _ => U32x16::splat(0),
    };
    let lsh = 8 * (lane & 1);
    let hsh = match BITS {
        3 | 5 => 8 * ((lane >> 1) & 1) + 4 * (lane & 1),
        6 => 8 * (lane & 1),
        _ => 0,
    };
    let scw = kxcommon::mem::global::vload8_u32(0, cp.add(16 * D));
    let ip = invn.add((gi as u64 * n as u64 + (rb * 16) as u64) as usize);
    let (iva, ivb) = (vload8_f32(0, ip).to_array(), vload8_f32(1, ip).to_array());
    let off = 1i32 << (BITS - 1);
    let o = w8.add((((rb * 16) as u64) * k as u64) as usize).add((gi * 128 + lane * 8) as usize);
    unroll!(16, 16, |c| {
        let sf = cvt::f16_bits_to_f32((scw[c / 2] >> (16 * (c % 2))) as u16);
        let iv = if c < 8 { iva[c % 8] } else { ivb[c % 8] };
        let mut v = [0u8; 8];
        unroll!(8, 8, |i| {
            let i = i as u32;
            let u = if BITS == 3 {
                ((lo[c] >> (16 * (i % 2) + lsh + 2 * (i / 2))) & 3) | (((hi[c] >> (16 * (i % 2) + hsh + i / 2)) & 1) << 2)
            } else {
                let mut u = (lo[c] >> (4 * (i / 2) + 16 * (i % 2))) & 15;
                if BITS == 5 {
                    u |= ((hi[c] >> (16 * (i % 2) + hsh + i / 2)) & 1) << 4;
                } else if BITS == 6 {
                    u |= ((hi[c] >> (16 * (i % 2) + hsh + 2 * (i / 2))) & 3) << 4;
                } else if BITS == 8 {
                    u |= ((hi[c] >> (16 * (i % 2) + 4 * (i / 2))) & 15) << 4;
                }
                u
            };
            v[i as usize] = cvt::i8_sat_rte(((u as i32 - off) as f32) * sf * iv) as u8;
        });
        *(o.add(c * k as usize) as *mut U32x2) = bits::<U8x8, U32x2>(U8x8::from_array(v));
    });
}

// ---- X3: rescaled-int32 K-sliced int8 (xq_pfgemm_i8r): compact per-slice scales [nsl][N] / [Rp][nsl] and the boundary ratios ---
// A slice whose maximum is 0 keeps the previous slice's scale (1.0 for the first), so every ratio is finite and an all-zero
// slice does not change the accumulator's units.

/// Per-(channel, slice of sb groups) weight scale sn [nsl][N] = m / 127 (m = max |s_g| 2^(BITS-1)), multiplier invn [nsl][N] =
/// 127 / m (0 when m = 0), boundary ratio rn [nsl][N] = sn[j-1] / sn[j] (rn[0] = 1); codes != 0: sn = max |s_g|, invn = 1 / max
/// |s_g| (sb = 1: W8 = q exactly, the grouped numerics). Grid (N/16), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfscale_r(wt: *const u32, sn: *mut f32, invn: *mut f32, rn: *mut f32, k: u32, n: u32, sb: u32, codes: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    const G: usize = 128;
    let d_n = lane_dwords(G);
    let (lane, rb) = (id::lane(), id::group_id(0));
    let ng = k / G as u32;
    let nn = (rb * 16 + lane) as u64;
    let mut prev = 1.0f32;
    let mut j = 0u32;
    let mut g0 = 0u32;
    while g0 < ng {
        let g1 = (g0 + sb).min(ng);
        let mut mx = 0.0f32;
        let mut gi = g0;
        while gi < g1 {
            let cp = wt.add(((rb as u64 * ng as u64 + gi as u64) * (16 * d_n as u64 + 8)) as usize);
            mx = math::fmax(mx, math::fabs(cvt::f16_bits_to_f32(*(cp.add(16 * d_n) as *const u16).add(lane as usize))));
            gi += 1;
        }
        // codes: the scale is the largest group scale itself, so with sb = 1 W8 = the XQ codes exactly
        let m = if codes != 0 { mx } else { mx * (1u32 << (BITS - 1)) as f32 };
        let d = if codes != 0 { 1.0 } else { 127.0 };
        let sc = if m > 0.0 { math::div_ocl(m, d) } else { prev };
        let o = (j as u64 * n as u64 + nn) as usize;
        *sn.add(o) = sc;
        *invn.add(o) = if m > 0.0 { math::div_ocl(d, m) } else { 0.0 };
        *rn.add(o) = if j == 0 { 1.0 } else { math::div_ocl(prev, sc) };
        prev = sc;
        j += 1;
        g0 = g1;
    }
}

/// [`xq_pfdec16x_c128`] with the multiplier per (channel, slice): W8 = sat_rte(float(q) * s_g * invn[g / sb][n]). Grid (N / 16,
/// K / 128 / 8), local (16, 8).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfdec16x_r128(w: *const u32, w8: *mut u8, invn: *const f32, k: u32, n: u32, sb: u32) {
    use kxcommon::mem::global::{vload16_u32, vload8_f32};
    const D: usize = 128 * BITS / 32;
    let (lane, rb, gi) = (id::lane(), id::group_id(0), id::global_id(1) as u32);
    let ng = k / 128;
    let cp = w.add(((rb as u64 * ng as u64 + gi as u64) * (16 * D as u64 + 8)) as usize);
    let l = lane as usize;
    let lo = if BITS == 3 { vload16_u32(l / 2, cp) } else { vload16_u32(l, cp) };
    let hi = match BITS {
        3 => vload16_u32(8 + l / 4, cp),
        5 => vload16_u32(16 + l / 4, cp),
        6 => vload16_u32(16 + l / 2, cp),
        8 => vload16_u32(16 + l, cp),
        _ => U32x16::splat(0),
    };
    let lsh = 8 * (lane & 1);
    let hsh = match BITS {
        3 | 5 => 8 * ((lane >> 1) & 1) + 4 * (lane & 1),
        6 => 8 * (lane & 1),
        _ => 0,
    };
    let scw = kxcommon::mem::global::vload8_u32(0, cp.add(16 * D));
    let ip = invn.add(((gi / sb) as u64 * n as u64 + (rb * 16) as u64) as usize);
    let (iva, ivb) = (vload8_f32(0, ip).to_array(), vload8_f32(1, ip).to_array());
    let off = 1i32 << (BITS - 1);
    let o = w8.add((((rb * 16) as u64) * k as u64) as usize).add((gi * 128 + lane * 8) as usize);
    unroll!(16, 16, |c| {
        let sf = cvt::f16_bits_to_f32((scw[c / 2] >> (16 * (c % 2))) as u16);
        let iv = if c < 8 { iva[c % 8] } else { ivb[c % 8] };
        let mut v = [0u8; 8];
        unroll!(8, 8, |i| {
            let i = i as u32;
            let u = if BITS == 3 {
                ((lo[c] >> (16 * (i % 2) + lsh + 2 * (i / 2))) & 3) | (((hi[c] >> (16 * (i % 2) + hsh + i / 2)) & 1) << 2)
            } else {
                let mut u = (lo[c] >> (4 * (i / 2) + 16 * (i % 2))) & 15;
                if BITS == 5 {
                    u |= ((hi[c] >> (16 * (i % 2) + hsh + i / 2)) & 1) << 4;
                } else if BITS == 6 {
                    u |= ((hi[c] >> (16 * (i % 2) + hsh + 2 * (i / 2))) & 3) << 4;
                } else if BITS == 8 {
                    u |= ((hi[c] >> (16 * (i % 2) + 4 * (i / 2))) & 15) << 4;
                }
                u
            };
            v[i as usize] = cvt::i8_sat_rte(((u as i32 - off) as f32) * sf * iv) as u8;
        });
        *(o.add(c * k as usize) as *mut U32x2) = bits::<U8x8, U32x2>(U8x8::from_array(v));
    });
}

/// Per-(token, slice) activation scales from the block maxima of [`xq_pfprep_k16m`]: st [Rp][nsl] = max / 127 (carried when 0),
/// tinv [Rp][nsl] = 127 / max (0 when 0), rt [Rp][nsl] = st[j-1] / st[j] (rt[0] = 1). Grid (Rp/16), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pftok_r(sa: *const f32, st: *mut f32, tinv: *mut f32, rt: *mut f32, kb: u32, sb: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    let t = id::group_id(0) * 16 + id::lane();
    let nsl = kb.div_ceil(sb);
    let p = sa.add((t as u64 * kb as u64) as usize);
    let o = (t as u64 * nsl as u64) as usize;
    let mut prev = 1.0f32;
    let mut j = 0u32;
    while j < nsl {
        let (g0, g1) = (j * sb, ((j + 1) * sb).min(kb));
        let mut mx = 0.0f32;
        let mut g = g0;
        while g < g1 {
            mx = math::fmax(mx, *p.add(g as usize));
            g += 1;
        }
        let sc = if mx > 0.0 { math::div_ocl(mx, 127.0) } else { prev };
        *st.add(o + j as usize) = sc;
        *tinv.add(o + j as usize) = if mx > 0.0 { math::div_ocl(127.0, mx) } else { 0.0 };
        *rt.add(o + j as usize) = if j == 0 { 1.0 } else { math::div_ocl(prev, sc) };
        prev = sc;
        j += 1;
    }
}

/// fp16 Xt [K][Rp] -> int8 with the (token, slice) multiplier: X8 = sat_rte(x * tinv[t][(k / 128) / sb]). Grid (ceil(Rp/128),
/// K/32), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn xq_pfq8r(xt: *const u16, tinv: *const f32, x8: *mut u8, rp: u32, nsl: u32, sb: u32) {
    use kxcommon::mem::global::vload8_u16;
    attr::reqd_work_group_size::<16, 1, 1>();
    let t0 = id::group_id(0) * 128 + id::lane() * 8;
    let k0 = id::group_id(1) * 32;
    if t0 >= rp {
        return;
    }
    let j = (k0 / 128) / sb;
    let mut inv = [0f32; 8];
    unroll!(8, 8, |i| inv[i] = *tinv.add((t0 as usize + i) * nsl as usize + j as usize));
    let mut kk = 0u32;
    while kk < 32 {
        let o = ((k0 + kk) as u64 * rp as u64 + t0 as u64) as usize;
        let h = vload8_u16(0, xt.add(o)).to_array();
        let mut b = [0u8; 8];
        unroll!(8, 8, |i| b[i] = cvt::i8_sat_rte(cvt::f16_bits_to_f32(h[i]) * inv[i]) as u8);
        *(x8.add(o) as *mut U32x2) = bits::<U8x8, U32x2>(U8x8::from_array(b));
        kk += 1;
    }
}
