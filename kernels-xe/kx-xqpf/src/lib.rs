//! XQ prompt-prefill GEMM with work-group tiling: Rust port of arc-engine `kernels/xpu/xq_pfg.cl` `GEMM_F16` (the
//! `xq_pfgemm_f16_cCtT` kernels, PFD = 1), same arguments, same arithmetic, so every output is bit-identical.
//!
//! W [N][K] fp16 (the pre-scaled decoded weights, kx-xq `xq_pfdec16x_128`), Xt [K][Rp] fp16 (k-major prepared rows,
//! kx-xq `xq_pfprep_k16`), y [R][ldy] at y_off, bf16 or fp32. A sub-group computes 32 output channels x 64 tokens (16
//! accumulators of 8 x 16), one ascending-k fp32 DPAS chain an output. SGC x SGT sub-groups a work-group cover 32 SGC
//! channels x 64 SGT tokens, so sub-groups that share a channel (token) block load the same W (Xt) fragments together and
//! most loads hit in L1. Fragments: W 32 channels x 32 k (one 2D read 16b 32r16x2c), Xt 32 k x 32 tokens per VNNI-transform
//! read (16b 32r16x2c), 2 a k step; the next k step's fragments are 2D-prefetched (PFD = 1).
//! Launch: grid (ceil(R / (64 SGT)), ceil(N / (32 SGC))), local 16 SGC SGT; build the module with
//! -cl-intel-256-GRF-per-thread (128 accumulator registers a lane).
#![no_std]
#![feature(abi_gpu_kernel, portable_simd)]
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
use core::mem::MaybeUninit;
use core::simd::num::SimdInt;
use kxcommon::ffi::{
    _Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt as read_w,
    _Z46intel_sub_group_2d_block_prefetch_16b_32r16x2cPU3AS1viiiDv2_i as prefetch,
    _Z52intel_sub_group_2d_block_read_transform_16b_32r16x2cPU3AS1viiiDv2_iPj as read_xt,
};
use kxcommon::mem::global::{vstore4_f32, vstore8_u16};
use kxcommon::v::*;
use kxcommon::{attr, bits, cvt, dpas, id, math, sg, unroll};

/// 2D prefetch distance in k steps of 32 (xq_pfg.cl PFD).
const PFD: i32 = 1;

/// Private arrays the 2D reads write (OpenCL: `__attribute__((aligned(16)))`), read back as DPAS operand vectors.
#[repr(C, align(64))]
struct Al<T>(T);

/// Dword offset of the B tile (k half kh, token block h) in a VNNI-transform 16b 32r16x2c read: block-major (block 0: 16
/// dwords = k rows 0..31 as 16 VNNI pairs, then block 1), the first 8 dwords of a block are k 0..15 (xq_pfg.cl `XB`).
const fn xb(kh: usize, h: usize) -> usize {
    h * 16 + kh * 8
}

#[inline(always)]
unsafe fn gemm_f16<const SGC: u32, const SGT: u32>(
    w: *const u16,
    xt: *const u16,
    y: *mut u8,
    r: u32,
    rp: u32,
    k: u32,
    n: u32,
    ldy: u32,
    y_off: u32,
    f32out: u32,
) {
    let (lane, sgi) = (id::lane(), id::sg_id());
    let n0 = ((id::group_id(1) * SGC + sgi % SGC) * 32) as i32;
    let t0 = ((id::group_id(0) * SGT + sgi / SGC) * 64) as i32;
    let mut acc = [[F32x8::splat(0.0); 4]; 4];
    let (wb, wpitch) = (w as *const u8, (k * 2) as i32);
    let (xb_, xpitch) = (xt as *const u8, (rp * 2) as i32);
    let mut k0 = 0i32;
    while k0 < k as i32 {
        if PFD > 0 && k0 + 32 * PFD < k as i32 {
            prefetch(wb, wpitch, n as i32, wpitch, I32x2::from_array([k0 + 32 * PFD, n0]));
            prefetch(xb_, xpitch, k as i32, xpitch, I32x2::from_array([t0, k0 + 32 * PFD]));
            prefetch(xb_, xpitch, k as i32, xpitch, I32x2::from_array([t0 + 32, k0 + 32 * PFD]));
        }
        let mut ab = MaybeUninit::<Al<[u16; 64]>>::uninit();
        read_w(wb, wpitch, n as i32, wpitch, I32x2::from_array([k0, n0]), ab.as_mut_ptr() as *mut u16);
        let mut bb = MaybeUninit::<Al<[[u32; 32]; 2]>>::uninit();
        let bp = bb.as_mut_ptr() as *mut u32;
        unroll!(2, 2, |u2| read_xt(xb_, xpitch, k as i32, xpitch, I32x2::from_array([t0 + u2 as i32 * 32, k0]), bp.add(u2 * 32)));
        let ap = ab.as_ptr() as *const u16;
        unroll!(2, 2, |kh| {
            unroll!(2, 2, |u2| {
                unroll!(2, 2, |h| {
                    let b = bits::<U32x8, I32x8>(*(bp.add(u2 * 32 + xb(kh, h)) as *const U32x8));
                    unroll!(4, 4, |g| {
                        let a = bits::<U16x8, I16x8>(*(ap.add(kh * 32 + g * 8) as *const U16x8));
                        acc[g][u2 * 2 + h] = dpas::f16_m8(a, b, acc[g][u2 * 2 + h]);
                    });
                });
            });
        });
        k0 += 32;
    }
    unroll!(4, 4, |u| {
        let tok = t0 as u32 + u as u32 * 16 + lane;
        if tok < r {
            unroll!(4, 4, |g| {
                if n0 + (g as i32) * 8 < n as i32 {
                    let at = (tok as u64 * ldy as u64 + (y_off + n0 as u32 + g as u32 * 8) as u64) as usize;
                    let a = acc[g][u].to_array();
                    if f32out != 0 {
                        let yf = y as *mut f32;
                        vstore4_f32(F32x4::from_array([a[0], a[1], a[2], a[3]]), 0, yf.add(at));
                        vstore4_f32(F32x4::from_array([a[4], a[5], a[6], a[7]]), 1, yf.add(at));
                    } else {
                        let mut o = [0u16; 8];
                        unroll!(8, 8, |rr| o[rr] = cvt::f32_to_bf16(a[rr]));
                        let o = U16x8::from_array(o);
                        let yh = (y as *mut u16).add(at);
                        if ((ldy | y_off) & 7) == 0 {
                            *(yh as *mut U32x4) = bits::<U16x8, U32x4>(o); // one 16-byte store
                        } else {
                            vstore8_u16(o, 0, yh);
                        }
                    }
                }
            });
        }
    });
}

macro_rules! gemm {
    ($name:ident, $c:literal, $t:literal, $wg:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            w: *const u16,
            xt: *const u16,
            y: *mut u8,
            r: u32,
            rp: u32,
            k: u32,
            n: u32,
            ldy: u32,
            y_off: u32,
            f32out: u32,
        ) {
            attr::reqd_work_group_size::<$wg, 1, 1>();
            gemm_f16::<$c, $t>(w, xt, y, r, rp, k, n, ldy, y_off, f32out)
        }
    };
}
gemm!(xq_pfgemm_f16_c4t2, 4, 2, 128);
gemm!(xq_pfgemm_f16_c2t4, 2, 4, 128);
gemm!(xq_pfgemm_f16_c4t4, 4, 4, 256);
gemm!(xq_pfgemm_f16_c8t2, 8, 2, 256);
gemm!(xq_pfgemm_f16_c1t1, 1, 1, 16);
gemm!(xq_pfgemm_f16_c8t4, 8, 4, 512);
gemm!(xq_pfgemm_f16_c4t8, 4, 8, 512);

// ---- int8 prompt GEMMs (X3 groundwork) ---------------------------------------------------------------------------------
use kxcommon::ffi::{
    _Z41intel_sub_group_2d_block_read_8b_32r32x1cPU3AS1viiiDv2_iPt as read_w8,
    _Z45intel_sub_group_2d_block_prefetch_8b_32r32x1cPU3AS1viiiDv2_i as prefetch8,
    _Z51intel_sub_group_2d_block_read_transform_8b_32r16x1cPU3AS1viiiDv2_iPj as read_x8_1,
    _Z51intel_sub_group_2d_block_read_transform_8b_32r16x2cPU3AS1viiiDv2_iPj as read_x8_2,
    _Z41intel_sub_group_2d_block_read_8b_32r32x2cPU3AS1viiiDv2_iPt as read_w8x2,
    _Z51intel_sub_group_2d_block_read_transform_8b_32r16x4cPU3AS1viiiDv2_iPj as read_x8_4,
    _Z45intel_sub_group_2d_block_prefetch_8b_32r32x2cPU3AS1viiiDv2_i as prefetch8x2,
};
use kxcommon::mem::global::vload8_f32;

/// bf16 / fp32 store of an 8-channel column of one token (xq_pf.cl `store_tile` / xq_pfg.cl `STORE_TILE`).
#[inline(always)]
unsafe fn store8(y: *mut u8, at: usize, a: [f32; 8], f32out: u32, ldy: u32, y_off: u32, wide16: bool) {
    if f32out != 0 {
        let yf = y as *mut f32;
        vstore4_f32(F32x4::from_array([a[0], a[1], a[2], a[3]]), 0, yf.add(at));
        vstore4_f32(F32x4::from_array([a[4], a[5], a[6], a[7]]), 1, yf.add(at));
    } else {
        let mut o = [0u16; 8];
        unroll!(8, 8, |rr| o[rr] = cvt::f32_to_bf16(a[rr]));
        let o = U16x8::from_array(o);
        let yh = (y as *mut u16).add(at);
        if wide16 && ((ldy | y_off) & 7) == 0 {
            *(yh as *mut U32x4) = bits::<U16x8, U32x4>(o);
        } else {
            vstore8_u16(o, 0, yh);
        }
    }
}

/// The exact XQ_PF=i8 numerics (xq_pf.cl `xq_pfgemm_i8{,w}_G`): per weight group an int8 DPAS chain into int32, then
/// acc = fma(float(int32), s_w * s_a, acc), groups in order. A sub-group: 32 channels x 16 U tokens. w8 [N][K] int8,
/// sw [K/G][N] fp32, xt [K][Rp] int8 (k-major), sa [Rp][K/128] fp32. Grid (ceil(R / 16U), N / 32), local 16.
#[inline(always)]
unsafe fn gemm_i8<const G: usize, const U: usize>(
    w8: *const i8,
    sw: *const f32,
    xt: *const u8,
    sa: *const f32,
    y: *mut u8,
    r: u32,
    rp: u32,
    k: u32,
    n: u32,
    ldy: u32,
    y_off: u32,
    f32out: u32,
) {
    let lane = id::lane();
    let (t0, n0) = (id::group_id(0) * (16 * U as u32), id::group_id(1) * 32);
    let mut acc = [F32x8::splat(0.0); 16];
    let (wb, xb_) = (w8 as *const u8, xt);
    let mut gi = 0u32;
    while gi < k / G as u32 {
        let mut ci = [[I32x8::splat(0); 4]; 4]; // U <= 3 used (4: the unroll! bound)
        unroll!(4, G / 32, |ks| {
            let k0 = (gi * G as u32 + ks as u32 * 32) as i32;
            let mut wa = MaybeUninit::<Al<[u16; 32]>>::uninit();
            read_w8(wb, k as i32, n as i32, k as i32, I32x2::from_array([k0, n0 as i32]), wa.as_mut_ptr() as *mut u16);
            let mut bb = MaybeUninit::<Al<[u32; 24]>>::uninit();
            let bp = bb.as_mut_ptr() as *mut u32;
            read_x8_2(xb_, rp as i32, k as i32, rp as i32, I32x2::from_array([t0 as i32, k0]), bp);
            if U == 3 {
                read_x8_1(xb_, rp as i32, k as i32, rp as i32, I32x2::from_array([t0 as i32 + 32, k0]), bp.add(16));
            }
            let wp = wa.as_ptr() as *const u16;
            unroll!(4, U, |u| {
                let b = bits::<U32x8, I32x8>(*(bp.add(8 * u) as *const U32x8));
                unroll!(4, 4, |a| {
                    let av = bits::<U16x8, I16x8>(kxcommon::mem::private::vload8_u16(wp.add(8 * a)));
                    ci[a][u] = dpas::i8_m8(av, b, ci[a][u]);
                });
            });
        });
        let mut s_w = [0f32; 32];
        unroll!(4, 4, |a| {
            let v = vload8_f32(0, sw.add((gi as u64 * n as u64 + (n0 + 8 * a as u32) as u64) as usize));
            unroll!(8, 8, |e| s_w[8 * a + e] = v[e]);
        });
        unroll!(4, U, |u| {
            let tok = (t0 + u as u32 * 16 + lane).min(rp - 1);
            let s_a = *sa.add((tok as u64 * (k / 128) as u64 + ((gi * G as u32) / 128) as u64) as usize);
            unroll!(4, 4, |a| {
                let c = ci[a][u].to_array();
                let mut v = acc[a * U + u];
                unroll!(8, 8, |e| v[e] = math::fma(c[e] as f32, s_w[8 * a + e] * s_a, v[e]));
                acc[a * U + u] = v;
            });
        });
        gi += 1;
    }
    unroll!(4, U, |u| {
        let tok = t0 + u as u32 * 16 + lane;
        if tok < r {
            unroll!(4, 4, |a| {
                let at = (tok as u64 * ldy as u64 + (y_off + n0 + a as u32 * 8) as u64) as usize;
                store8(y, at, acc[a * U + u].to_array(), f32out, ldy, y_off, false);
            });
        }
    });
}

macro_rules! gemm_i8 {
    ($name:ident, $g:literal, $u:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            w8: *const i8,
            sw: *const f32,
            xt: *const u8,
            sa: *const f32,
            y: *mut u8,
            r: u32,
            rp: u32,
            k: u32,
            n: u32,
            ldy: u32,
            y_off: u32,
            f32out: u32,
        ) {
            gemm_i8::<$g, $u>(w8, sw, xt, sa, y, r, rp, k, n, ldy, y_off, f32out)
        }
    };
}
gemm_i8!(xq_pfgemm_i8_32, 32, 2);
gemm_i8!(xq_pfgemm_i8_64, 64, 2);
gemm_i8!(xq_pfgemm_i8_128, 128, 2);
gemm_i8!(xq_pfgemm_i8w_32, 32, 3);
gemm_i8!(xq_pfgemm_i8w_64, 64, 3);
gemm_i8!(xq_pfgemm_i8w_128, 128, 3);

/// The same numerics with work-group tiling (xq_pfg.cl `xq_pfgemm_i8g_G_cCtT`): SGC x SGT sub-groups of 32 channels x
/// 32 tokens, PFD = 1. Grid (ceil(R / (32 SGT)), ceil(N / (32 SGC))), local 16 SGC SGT.
#[inline(always)]
unsafe fn gemm_i8g<const G: usize, const SGC: u32, const SGT: u32>(
    w8: *const i8,
    sw: *const f32,
    xt: *const u8,
    sa: *const f32,
    y: *mut u8,
    r: u32,
    rp: u32,
    k: u32,
    n: u32,
    ldy: u32,
    y_off: u32,
    f32out: u32,
) {
    let (lane, sgi) = (id::lane(), id::sg_id());
    let t0 = (id::group_id(0) * SGT + sgi / SGC) * 32;
    let n0 = (id::group_id(1) * SGC + sgi % SGC) * 32;
    let mut acc = [F32x8::splat(0.0); 8];
    let (wb, xb_) = (w8 as *const u8, xt);
    let mut gi = 0u32;
    while gi < k / G as u32 {
        let mut ci = [[I32x8::splat(0); 2]; 4];
        unroll!(4, G / 32, |ks| {
            let k0 = (gi * G as u32 + ks as u32 * 32) as i32;
            if PFD > 0 && k0 + 32 * PFD < k as i32 {
                prefetch8(wb, k as i32, n as i32, k as i32, I32x2::from_array([k0 + 32 * PFD, n0 as i32]));
                prefetch8(xb_, rp as i32, k as i32, rp as i32, I32x2::from_array([t0 as i32, k0 + 32 * PFD]));
            }
            let mut wa = MaybeUninit::<Al<[u16; 32]>>::uninit();
            read_w8(wb, k as i32, n as i32, k as i32, I32x2::from_array([k0, n0 as i32]), wa.as_mut_ptr() as *mut u16);
            let mut bb = MaybeUninit::<Al<[u32; 16]>>::uninit();
            let bp = bb.as_mut_ptr() as *mut u32;
            read_x8_2(xb_, rp as i32, k as i32, rp as i32, I32x2::from_array([t0 as i32, k0]), bp);
            let wp = wa.as_ptr() as *const u16;
            unroll!(2, 2, |u| {
                let b = bits::<U32x8, I32x8>(*(bp.add(8 * u) as *const U32x8));
                unroll!(4, 4, |a| {
                    let av = bits::<U16x8, I16x8>(kxcommon::mem::private::vload8_u16(wp.add(8 * a)));
                    ci[a][u] = dpas::i8_m8(av, b, ci[a][u]);
                });
            });
        });
        let mut s_w = [0f32; 32];
        unroll!(4, 4, |a| {
            let v = vload8_f32(0, sw.add((gi as u64 * n as u64 + (n0 + 8 * a as u32).min(n - 8) as u64) as usize));
            unroll!(8, 8, |e| s_w[8 * a + e] = v[e]);
        });
        unroll!(2, 2, |u| {
            let tok = (t0 + u as u32 * 16 + lane).min(rp - 1);
            let s_a = *sa.add((tok as u64 * (k / 128) as u64 + ((gi * G as u32) / 128) as u64) as usize);
            unroll!(4, 4, |a| {
                let c = ci[a][u].to_array();
                let mut v = acc[a * 2 + u];
                unroll!(8, 8, |e| v[e] = math::fma(c[e] as f32, s_w[8 * a + e] * s_a, v[e]));
                acc[a * 2 + u] = v;
            });
        });
        gi += 1;
    }
    unroll!(2, 2, |u| {
        let tok = t0 + u as u32 * 16 + lane;
        if tok < r {
            unroll!(4, 4, |a| {
                if n0 + a as u32 * 8 < n {
                    let at = (tok as u64 * ldy as u64 + (y_off + n0 + a as u32 * 8) as u64) as usize;
                    store8(y, at, acc[a * 2 + u].to_array(), f32out, ldy, y_off, true);
                }
            });
        }
    });
}

macro_rules! gemm_i8g {
    ($name:ident, $c:literal, $t:literal, $wg:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            w8: *const i8,
            sw: *const f32,
            xt: *const u8,
            sa: *const f32,
            y: *mut u8,
            r: u32,
            rp: u32,
            k: u32,
            n: u32,
            ldy: u32,
            y_off: u32,
            f32out: u32,
        ) {
            attr::reqd_work_group_size::<$wg, 1, 1>();
            gemm_i8g::<128, $c, $t>(w8, sw, xt, sa, y, r, rp, k, n, ldy, y_off, f32out)
        }
    };
}
gemm_i8g!(xq_pfgemm_i8g_128_c4t4, 4, 4, 256);
gemm_i8g!(xq_pfgemm_i8g_128_c4t8, 4, 8, 512);
gemm_i8g!(xq_pfgemm_i8g_128_c8t4, 8, 4, 512);
gemm_i8g!(xq_pfgemm_i8g_128_c2t8, 2, 8, 256);
gemm_i8g!(xq_pfgemm_i8g_128_c1t1, 1, 1, 16);

/// int8 without group logic (xq_pfg.cl `xq_pfgemm_i8c_cCtT`): W8 [N][K] with one fp32 scale a channel (sn [N]), Xt8 [K][Rp]
/// with one a token (st [Rp]); y = float(sum_k W8 Xt8) * sn * st, the int32 sum exact. 32 channels x 64 tokens a
/// sub-group, same tiling as the fp16 GEMM.
#[inline(always)]
unsafe fn gemm_i8c<const SGC: u32, const SGT: u32>(
    w: *const i8,
    sn: *const f32,
    xt: *const u8,
    st: *const f32,
    y: *mut u8,
    r: u32,
    rp: u32,
    k: u32,
    n: u32,
    ldy: u32,
    y_off: u32,
    f32out: u32,
) {
    let (lane, sgi) = (id::lane(), id::sg_id());
    let n0 = ((id::group_id(1) * SGC + sgi % SGC) * 32) as i32;
    let t0 = ((id::group_id(0) * SGT + sgi / SGC) * 64) as i32;
    let mut ia = [[I32x8::splat(0); 4]; 4];
    let (wb, xb_) = (w as *const u8, xt);
    let mut k0 = 0i32;
    while k0 < k as i32 {
        if PFD > 0 && k0 + 32 * PFD < k as i32 {
            prefetch8(wb, k as i32, n as i32, k as i32, I32x2::from_array([k0 + 32 * PFD, n0]));
            prefetch8(xb_, rp as i32, k as i32, rp as i32, I32x2::from_array([t0, k0 + 32 * PFD]));
            prefetch8(xb_, rp as i32, k as i32, rp as i32, I32x2::from_array([t0 + 32, k0 + 32 * PFD]));
        }
        let mut wa = MaybeUninit::<Al<[u16; 32]>>::uninit();
        read_w8(wb, k as i32, n as i32, k as i32, I32x2::from_array([k0, n0]), wa.as_mut_ptr() as *mut u16);
        let mut bb = MaybeUninit::<Al<[[u32; 16]; 2]>>::uninit();
        let bp = bb.as_mut_ptr() as *mut u32;
        unroll!(2, 2, |u2| read_x8_2(xb_, rp as i32, k as i32, rp as i32, I32x2::from_array([t0 + u2 as i32 * 32, k0]), bp.add(16 * u2)));
        let wp = wa.as_ptr() as *const u16;
        unroll!(4, 4, |u| {
            let b = bits::<U32x8, I32x8>(*(bp.add(16 * (u >> 1) + 8 * (u & 1)) as *const U32x8));
            unroll!(4, 4, |g| {
                let av = bits::<U16x8, I16x8>(*(wp.add(8 * g) as *const U16x8));
                ia[g][u] = dpas::i8_m8(av, b, ia[g][u]);
            });
        });
        k0 += 32;
    }
    let mut acc = [[F32x8::splat(0.0); 4]; 4];
    unroll!(4, 4, |u| {
        let s_t = *st.add((t0 as u32 + u as u32 * 16 + lane).min(rp - 1) as usize);
        unroll!(4, 4, |g| {
            let s_n = vload8_f32(0, sn.add((n0 + g as i32 * 8).min(n as i32 - 8) as usize));
            let c = ia[g][u].to_array();
            let mut v = [0f32; 8];
            unroll!(8, 8, |e| v[e] = c[e] as f32 * s_n[e] * s_t);
            acc[g][u] = F32x8::from_array(v);
        });
    });
    unroll!(4, 4, |u| {
        let tok = t0 as u32 + u as u32 * 16 + lane;
        if tok < r {
            unroll!(4, 4, |g| {
                if n0 + (g as i32) * 8 < n as i32 {
                    let at = (tok as u64 * ldy as u64 + (y_off + n0 as u32 + g as u32 * 8) as u64) as usize;
                    store8(y, at, acc[g][u].to_array(), f32out, ldy, y_off, true);
                }
            });
        }
    });
}

macro_rules! gemm_i8c {
    ($name:ident, $c:literal, $t:literal, $wg:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            w: *const i8,
            sn: *const f32,
            xt: *const u8,
            st: *const f32,
            y: *mut u8,
            r: u32,
            rp: u32,
            k: u32,
            n: u32,
            ldy: u32,
            y_off: u32,
            f32out: u32,
        ) {
            attr::reqd_work_group_size::<$wg, 1, 1>();
            gemm_i8c::<$c, $t>(w, sn, xt, st, y, r, rp, k, n, ldy, y_off, f32out)
        }
    };
}
gemm_i8c!(xq_pfgemm_i8c_c4t2, 4, 2, 128);
gemm_i8c!(xq_pfgemm_i8c_c4t4, 4, 4, 256);
gemm_i8c!(xq_pfgemm_i8c_c1t1, 1, 1, 16);
gemm_i8c!(xq_pfgemm_i8c_c8t2, 8, 2, 256);
gemm_i8c!(xq_pfgemm_i8c_c8t4, 8, 4, 512);

// ---- X3: int8 prompt GEMMs beyond the OpenCL reference ----------------------------------------------------------------

/// Group-free int8 GEMM (the [`gemm_i8c`] numerics, so bit-identical to it: the int32 sums are exact) with 64 k a loop
/// step instead of 32: one 8b 32r32x2c W read (32 channels x 64 k) and two 8b 32r16x4c transform Xt reads (32 k x 64 tokens
/// each) feed 32 int8 DPAS, the shape of the fp16 GEMM's step (3 messages, 32 DPAS). At 32 k a step the int8 tile has half
/// the DPAS work to hide each step's load latency behind, and i8c reaches ~54% of the int8 peak where the fp16 GEMM
/// reaches ~77% of its own. Grid (ceil(R / (64 SGT)), ceil(N / (32 SGC))), local 16 SGC SGT, K % 64 == 0, 256 GRF.
#[inline(always)]
unsafe fn gemm_i8c2<const SGC: u32, const SGT: u32>(
    w: *const i8,
    sn: *const f32,
    xt: *const u8,
    st: *const f32,
    y: *mut u8,
    r: u32,
    rp: u32,
    k: u32,
    n: u32,
    ldy: u32,
    y_off: u32,
    f32out: u32,
) {
    let (lane, sgi) = (id::lane(), id::sg_id());
    let n0 = ((id::group_id(1) * SGC + sgi % SGC) * 32) as i32;
    let t0 = ((id::group_id(0) * SGT + sgi / SGC) * 64) as i32;
    let mut ia = [[I32x8::splat(0); 4]; 4];
    let (wb, xb_) = (w as *const u8, xt);
    let (ki, ri) = (k as i32, rp as i32);
    let mut k0 = 0i32;
    while k0 < ki {
        if k0 + 64 < ki {
            prefetch8x2(wb, ki, n as i32, ki, I32x2::from_array([k0 + 64, n0]));
            prefetch8x2(xb_, ri, ki, ri, I32x2::from_array([t0, k0 + 64]));
            prefetch8x2(xb_, ri, ki, ri, I32x2::from_array([t0, k0 + 96]));
        }
        let mut wa = MaybeUninit::<Al<[u16; 64]>>::uninit();
        read_w8x2(wb, ki, n as i32, ki, I32x2::from_array([k0, n0]), wa.as_mut_ptr() as *mut u16);
        let mut bb = MaybeUninit::<Al<[u32; 64]>>::uninit();
        let bp = bb.as_mut_ptr() as *mut u32;
        unroll!(2, 2, |kh| read_x8_4(xb_, ri, ki, ri, I32x2::from_array([t0, k0 + 32 * kh as i32]), bp.add(32 * kh)));
        let wp = wa.as_ptr() as *const u16;
        unroll!(2, 2, |kh| {
            unroll!(4, 4, |u| {
                let b = bits::<U32x8, I32x8>(*(bp.add(32 * kh + 8 * u) as *const U32x8));
                unroll!(4, 4, |g| {
                    let av = bits::<U16x8, I16x8>(*(wp.add(32 * kh + 8 * g) as *const U16x8));
                    ia[g][u] = dpas::i8_m8(av, b, ia[g][u]);
                });
            });
        });
        k0 += 64;
    }
    let mut acc = [[F32x8::splat(0.0); 4]; 4];
    unroll!(4, 4, |u| {
        let s_t = *st.add((t0 as u32 + u as u32 * 16 + lane).min(rp - 1) as usize);
        unroll!(4, 4, |g| {
            let s_n = vload8_f32(0, sn.add((n0 + g as i32 * 8).min(n as i32 - 8) as usize));
            let c = ia[g][u].to_array();
            let mut v = [0f32; 8];
            unroll!(8, 8, |e| v[e] = c[e] as f32 * s_n[e] * s_t);
            acc[g][u] = F32x8::from_array(v);
        });
    });
    unroll!(4, 4, |u| {
        let tok = t0 as u32 + u as u32 * 16 + lane;
        if tok < r {
            unroll!(4, 4, |g| {
                if n0 + (g as i32) * 8 < n as i32 {
                    let at = (tok as u64 * ldy as u64 + (y_off + n0 as u32 + g as u32 * 8) as u64) as usize;
                    store8(y, at, acc[g][u].to_array(), f32out, ldy, y_off, true);
                }
            });
        }
    });
}

macro_rules! gemm_i8c2 {
    ($name:ident, $c:literal, $t:literal, $wg:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            w: *const i8,
            sn: *const f32,
            xt: *const u8,
            st: *const f32,
            y: *mut u8,
            r: u32,
            rp: u32,
            k: u32,
            n: u32,
            ldy: u32,
            y_off: u32,
            f32out: u32,
        ) {
            attr::reqd_work_group_size::<$wg, 1, 1>();
            gemm_i8c2::<$c, $t>(w, sn, xt, st, y, r, rp, k, n, ldy, y_off, f32out)
        }
    };
}
gemm_i8c2!(xq_pfgemm_i8c2_c4t4, 4, 4, 256);
gemm_i8c2!(xq_pfgemm_i8c2_c4t2, 4, 2, 128);
gemm_i8c2!(xq_pfgemm_i8c2_c2t4, 2, 4, 128);
gemm_i8c2!(xq_pfgemm_i8c2_c8t2, 8, 2, 256);
gemm_i8c2!(xq_pfgemm_i8c2_c8t4, 8, 4, 512);
gemm_i8c2!(xq_pfgemm_i8c2_c4t8, 4, 8, 512);

/// The exact grouped int8 numerics of [`gemm_i8g`] (G = 128: per group an int32 DPAS chain, then acc = fma(float(int32),
/// s_w * s_a, acc)), so bit-identical to it, with 64 k a loop step: one 8b 32r32x2c W read and two 8b 32r16x2c transform Xt
/// reads feed 16 DPAS (i8g: 8). A sub-group is 32 channels x 32 tokens: the group logic keeps an int32 and an fp32
/// accumulator set live, 2 x 64 registers, so the 32 x 64 tile of the group-free kernels does not fit in 256.
/// Grid (ceil(R / (32 SGT)), ceil(N / (32 SGC))), local 16 SGC SGT.
#[inline(always)]
unsafe fn gemm_i8g2<const SGC: u32, const SGT: u32>(
    w8: *const i8,
    sw: *const f32,
    xt: *const u8,
    sa: *const f32,
    y: *mut u8,
    r: u32,
    rp: u32,
    k: u32,
    n: u32,
    ldy: u32,
    y_off: u32,
    f32out: u32,
) {
    const G: u32 = 128;
    let (lane, sgi) = (id::lane(), id::sg_id());
    let t0 = (id::group_id(0) * SGT + sgi / SGC) * 32;
    let n0 = (id::group_id(1) * SGC + sgi % SGC) * 32;
    let mut acc = [F32x8::splat(0.0); 8];
    let (wb, xb_) = (w8 as *const u8, xt);
    let (ki, ri) = (k as i32, rp as i32);
    let mut gi = 0u32;
    while gi < k / G {
        let mut ci = [[I32x8::splat(0); 2]; 4];
        unroll!(2, 2, |ks| {
            let k0 = (gi * G + ks as u32 * 64) as i32;
            if k0 + 64 < ki {
                prefetch8x2(wb, ki, n as i32, ki, I32x2::from_array([k0 + 64, n0 as i32]));
                prefetch8(xb_, ri, ki, ri, I32x2::from_array([t0 as i32, k0 + 64]));
                prefetch8(xb_, ri, ki, ri, I32x2::from_array([t0 as i32, k0 + 96]));
            }
            let mut wa = MaybeUninit::<Al<[u16; 64]>>::uninit();
            read_w8x2(wb, ki, n as i32, ki, I32x2::from_array([k0, n0 as i32]), wa.as_mut_ptr() as *mut u16);
            let mut bb = MaybeUninit::<Al<[u32; 32]>>::uninit();
            let bp = bb.as_mut_ptr() as *mut u32;
            unroll!(2, 2, |kh| read_x8_2(xb_, ri, ki, ri, I32x2::from_array([t0 as i32, k0 + 32 * kh as i32]), bp.add(16 * kh)));
            let wp = wa.as_ptr() as *const u16;
            unroll!(2, 2, |kh| {
                unroll!(2, 2, |u| {
                    let b = bits::<U32x8, I32x8>(*(bp.add(16 * kh + 8 * u) as *const U32x8));
                    unroll!(4, 4, |a| {
                        let av = bits::<U16x8, I16x8>(*(wp.add(32 * kh + 8 * a) as *const U16x8));
                        ci[a][u] = dpas::i8_m8(av, b, ci[a][u]);
                    });
                });
            });
        });
        let mut s_w = [0f32; 32];
        unroll!(4, 4, |a| {
            let v = vload8_f32(0, sw.add((gi as u64 * n as u64 + (n0 + 8 * a as u32).min(n - 8) as u64) as usize));
            unroll!(8, 8, |e| s_w[8 * a + e] = v[e]);
        });
        unroll!(2, 2, |u| {
            let tok = (t0 + u as u32 * 16 + lane).min(rp - 1);
            let s_a = *sa.add((tok as u64 * (k / 128) as u64 + gi as u64) as usize);
            unroll!(4, 4, |a| {
                let c = ci[a][u].to_array();
                let mut v = acc[a * 2 + u];
                unroll!(8, 8, |e| v[e] = math::fma(c[e] as f32, s_w[8 * a + e] * s_a, v[e]));
                acc[a * 2 + u] = v;
            });
        });
        gi += 1;
    }
    unroll!(2, 2, |u| {
        let tok = t0 + u as u32 * 16 + lane;
        if tok < r {
            unroll!(4, 4, |a| {
                if n0 + a as u32 * 8 < n {
                    let at = (tok as u64 * ldy as u64 + (y_off + n0 + a as u32 * 8) as u64) as usize;
                    store8(y, at, acc[a * 2 + u].to_array(), f32out, ldy, y_off, true);
                }
            });
        }
    });
}

macro_rules! gemm_i8g2 {
    ($name:ident, $c:literal, $t:literal, $wg:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            w8: *const i8,
            sw: *const f32,
            xt: *const u8,
            sa: *const f32,
            y: *mut u8,
            r: u32,
            rp: u32,
            k: u32,
            n: u32,
            ldy: u32,
            y_off: u32,
            f32out: u32,
        ) {
            attr::reqd_work_group_size::<$wg, 1, 1>();
            gemm_i8g2::<$c, $t>(w8, sw, xt, sa, y, r, rp, k, n, ldy, y_off, f32out)
        }
    };
}
gemm_i8g2!(xq_pfgemm_i8g2_c4t4, 4, 4, 256);
gemm_i8g2!(xq_pfgemm_i8g2_c4t8, 4, 8, 512);
gemm_i8g2!(xq_pfgemm_i8g2_c8t4, 8, 4, 512);
gemm_i8g2!(xq_pfgemm_i8g2_c2t8, 2, 8, 256);
gemm_i8g2!(xq_pfgemm_i8g2_c8t2, 8, 2, 256);

/// K-sliced int8 GEMM: per slice of `sb` 128-blocks an exact int32 DPAS chain in the [`gemm_i8c2`] step (64 k, 32 DPAS, the
/// 32 x 64 tile), then acc = fma(float(int32), sn[g0][n] * st[t][g0], acc) (g0 the slice's first block), slices in order. The
/// fp32 tile lives in SLM (16 x 128 dwords a sub-group, zeroed first, sub-group block reads / writes; no barrier: a sub-group
/// touches only its own region, lane l the dwords it wrote; y is stored from it after the last slice), so the registers hold only the int32 tile: per-slice scales at the i8c2 tile size.
/// W8 [N][K] int8, sn [K/128][N], X8 [K][Rp] int8, st [Rp][K/128]. Grid (ceil(R / (64 SGT)), ceil(N / (32 SGC))), local 16 SGC
/// SGT, SLM 8 KB a sub-group, K % 64 == 0.
#[inline(always)]
unsafe fn gemm_i8s<const SGC: u32, const SGT: u32, const SLMF: usize>(
    w: *const i8,
    sn: *const f32,
    xt: *const u8,
    st: *const f32,
    y: *mut u8,
    r: u32,
    rp: u32,
    k: u32,
    n: u32,
    ldy: u32,
    y_off: u32,
    f32out: u32,
    sb: u32,
) {
    use kxcommon::blk::local::{read_u32x8, write_u32x8};
    let tile = kxcommon::slm::alloc::<u32, SLMF, 0>();
    let (lane, sgi) = (id::lane(), id::sg_id());
    let mine = tile.add(sgi as usize * 2048);
    let n0 = ((id::group_id(1) * SGC + sgi % SGC) * 32) as i32;
    let t0 = ((id::group_id(0) * SGT + sgi / SGC) * 64) as i32;
    let (wb, xb_) = (w as *const u8, xt);
    let (ki, ri) = (k as i32, rp as i32);
    let kb = k / 128;
    let nsl = kb.div_ceil(sb);
    unroll!(16, 16, |i| write_u32x8(mine.add(i * 128), U32x8::splat(0)));
    let mut k0 = 0i32;
    let mut j = 0u32;
    while j < nsl {
        let g0 = j * sb;
        let kend = (((j + 1) * sb).min(kb) * 128) as i32;
        let mut ia = [[I32x8::splat(0); 4]; 4];
        while k0 < kend {
            if k0 + 64 < ki {
                prefetch8x2(wb, ki, n as i32, ki, I32x2::from_array([k0 + 64, n0]));
                prefetch8x2(xb_, ri, ki, ri, I32x2::from_array([t0, k0 + 64]));
                prefetch8x2(xb_, ri, ki, ri, I32x2::from_array([t0, k0 + 96]));
            }
            let mut wa = MaybeUninit::<Al<[u16; 64]>>::uninit();
            read_w8x2(wb, ki, n as i32, ki, I32x2::from_array([k0, n0]), wa.as_mut_ptr() as *mut u16);
            let mut bb = MaybeUninit::<Al<[u32; 64]>>::uninit();
            let bp = bb.as_mut_ptr() as *mut u32;
            unroll!(2, 2, |kh| read_x8_4(xb_, ri, ki, ri, I32x2::from_array([t0, k0 + 32 * kh as i32]), bp.add(32 * kh)));
            let wp = wa.as_ptr() as *const u16;
            unroll!(2, 2, |kh| {
                unroll!(4, 4, |u| {
                    let b = bits::<U32x8, I32x8>(*(bp.add(32 * kh + 8 * u) as *const U32x8));
                    unroll!(4, 4, |g| {
                        let av = bits::<U16x8, I16x8>(*(wp.add(32 * kh + 8 * g) as *const U16x8));
                        ia[g][u] = dpas::i8_m8(av, b, ia[g][u]);
                    });
                });
            });
            k0 += 64;
        }
        unroll!(4, 4, |u| {
            let tok = t0 as u32 + u as u32 * 16 + lane;
            let s_t = *st.add((tok.min(rp - 1) as u64 * kb as u64 + g0 as u64) as usize);
            unroll!(4, 4, |g| {
                let s_n = vload8_f32(0, sn.add((g0 as u64 * n as u64 + (n0 + g as i32 * 8).min(n as i32 - 8) as u64) as usize));
                let c = ia[g][u].to_array();
                let p = mine.add((u * 4 + g) * 128);
                let mut v = bits::<U32x8, F32x8>(read_u32x8(p)).to_array();
                unroll!(8, 8, |e| v[e] = math::fma(c[e] as f32, s_n[e] * s_t, v[e]));
                write_u32x8(p, bits::<F32x8, U32x8>(F32x8::from_array(v)));
            });
        });
        j += 1;
    }
    unroll!(4, 4, |u| {
        let tok = t0 as u32 + u as u32 * 16 + lane;
        if tok < r {
            unroll!(4, 4, |g| {
                if n0 + (g as i32) * 8 < n as i32 {
                    let v = bits::<U32x8, F32x8>(read_u32x8(mine.add((u * 4 + g) * 128))).to_array();
                    let at = (tok as u64 * ldy as u64 + (y_off + n0 as u32 + g as u32 * 8) as u64) as usize;
                    store8(y, at, v, f32out, ldy, y_off, true);
                }
            });
        }
    });
}

macro_rules! gemm_i8s {
    ($name:ident, $c:literal, $t:literal, $wg:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            w: *const i8,
            sn: *const f32,
            xt: *const u8,
            st: *const f32,
            y: *mut u8,
            r: u32,
            rp: u32,
            k: u32,
            n: u32,
            ldy: u32,
            y_off: u32,
            f32out: u32,
            sb: u32,
        ) {
            attr::reqd_work_group_size::<$wg, 1, 1>();
            gemm_i8s::<$c, $t, { $c * $t * 2048 }>(w, sn, xt, st, y, r, rp, k, n, ldy, y_off, f32out, sb)
        }
    };
}
gemm_i8s!(xq_pfgemm_i8s_c4t4, 4, 4, 256);
gemm_i8s!(xq_pfgemm_i8s_c4t2, 4, 2, 128);
gemm_i8s!(xq_pfgemm_i8s_c2t4, 2, 4, 128);
gemm_i8s!(xq_pfgemm_i8s_c8t2, 8, 2, 256);

/// K-sliced int8 with a rescaled int32 accumulator (no fp32 tile): per slice of `sb` 128-blocks the exact int32 DPAS chain of
/// [`gemm_i8c2`] (64 k a step, the 32 x 64 tile), accumulating in units of the slice's channel x token scale; at each slice
/// boundary the accumulator changes units, ia = sat_rte(float(ia) * (rn[j][n] * rt[t][j])) (the old scale over the new, at most
/// half a product quantum lost a boundary), and y = (float(ia) * sn[last][n]) * st[t][last]. Registers and loads as i8c2.
/// W8 [N][K] int8, sn / rn [nsl][N], X8 [K][Rp] int8, st / rt [Rp][nsl]. Grid (ceil(R / (64 SGT)), ceil(N / (32 SGC))), local
/// 16 SGC SGT, K % 64 == 0, sb * 128 % 64 == 0.
#[inline(always)]
unsafe fn gemm_i8r<const SGC: u32, const SGT: u32, const U: usize>(
    w: *const i8,
    sn: *const f32,
    rn: *const f32,
    xt: *const u8,
    st: *const f32,
    rt: *const f32,
    y: *mut u8,
    r: u32,
    rp: u32,
    k: u32,
    n: u32,
    ldy: u32,
    y_off: u32,
    f32out: u32,
    sb: u32,
) {
    let (lane, sgi) = (id::lane(), id::sg_id());
    let n0 = ((id::group_id(1) * SGC + sgi % SGC) * 32) as i32;
    let t0 = ((id::group_id(0) * SGT + sgi / SGC) * (16 * U as u32)) as i32;
    let mut ia = [[I32x8::splat(0); 4]; 4];
    let (wb, xb_) = (w as *const u8, xt);
    let (ki, ri) = (k as i32, rp as i32);
    let kb = k / 128;
    let nsl = kb.div_ceil(sb);
    let mut k0 = 0i32;
    let mut j = 0u32;
    while j < nsl {
        // unconditional (j = 0: ratios 1 on a zero tile): no branch around the accumulators, which made IGC copy and spill
        // them in the loop header
        let rbase = j as u64 * n as u64;
        let rv0 = *rn.add((rbase + (n0 as u32).min(n - 16) as u64 + lane as u64) as usize);
        let rv1 = *rn.add((rbase + (n0 as u32 + 16).min(n - 16) as u64 + lane as u64) as usize);
        let mut r_t = [0f32; 4];
        unroll!(4, U, |u| {
            let tok = (t0 as u32 + u as u32 * 16 + lane).min(rp - 1);
            r_t[u] = *rt.add((tok as u64 * nsl as u64 + j as u64) as usize);
        });
        unroll!(4, 4, |g| {
            let rv = if g < 2 { rv0 } else { rv1 };
            let mut rn8 = [0f32; 8];
            unroll!(8, 8, |e| rn8[e] = sg::broadcast_f32(rv, (8 * (g % 2) + e) as u32));
            let rnv = F32x8::from_array(rn8);
            unroll!(4, U, |u| {
                ia[g][u] = cvt::i32x8_sat_rte(ia[g][u].cast::<f32>() * (rnv * F32x8::splat(r_t[u])));
            });
        });
        let kend = (((j + 1) * sb).min(kb) * 128) as i32;
        while k0 < kend {
            if k0 + 64 < ki {
                prefetch8x2(wb, ki, n as i32, ki, I32x2::from_array([k0 + 64, n0]));
                prefetch8x2(xb_, ri, ki, ri, I32x2::from_array([t0, k0 + 64]));
                prefetch8x2(xb_, ri, ki, ri, I32x2::from_array([t0, k0 + 96]));
            }
            let mut wa = MaybeUninit::<Al<[u16; 64]>>::uninit();
            read_w8x2(wb, ki, n as i32, ki, I32x2::from_array([k0, n0]), wa.as_mut_ptr() as *mut u16);
            let mut bb = MaybeUninit::<Al<[u32; 64]>>::uninit();
            let bp = bb.as_mut_ptr() as *mut u32;
            unroll!(2, 2, |kh| {
                if U == 4 {
                    read_x8_4(xb_, ri, ki, ri, I32x2::from_array([t0, k0 + 32 * kh as i32]), bp.add(32 * kh));
                } else {
                    read_x8_2(xb_, ri, ki, ri, I32x2::from_array([t0, k0 + 32 * kh as i32]), bp.add(32 * kh));
                    read_x8_1(xb_, ri, ki, ri, I32x2::from_array([t0 + 32, k0 + 32 * kh as i32]), bp.add(32 * kh + 16));
                }
            });
            let wp = wa.as_ptr() as *const u16;
            unroll!(2, 2, |kh| {
                unroll!(4, U, |u| {
                    let b = bits::<U32x8, I32x8>(*(bp.add(32 * kh + 8 * u) as *const U32x8));
                    unroll!(4, 4, |g| {
                        let av = bits::<U16x8, I16x8>(*(wp.add(32 * kh + 8 * g) as *const U16x8));
                        ia[g][u] = dpas::i8_m8(av, b, ia[g][u]);
                    });
                });
            });
            k0 += 64;
        }
        j += 1;
    }
    let last = nsl - 1;
    unroll!(4, U, |u| {
        let tok = t0 as u32 + u as u32 * 16 + lane;
        let s_t = *st.add((tok.min(rp - 1) as u64 * nsl as u64 + last as u64) as usize);
        if tok < r {
            unroll!(4, 4, |g| {
                if n0 + (g as i32) * 8 < n as i32 {
                    let s_n = vload8_f32(0, sn.add((last as u64 * n as u64 + (n0 + g as i32 * 8) as u64) as usize));
                    let c = ia[g][u].to_array();
                    let mut v = [0f32; 8];
                    unroll!(8, 8, |e| v[e] = c[e] as f32 * s_n[e] * s_t);
                    let at = (tok as u64 * ldy as u64 + (y_off + n0 as u32 + g as u32 * 8) as u64) as usize;
                    store8(y, at, v, f32out, ldy, y_off, true);
                }
            });
        }
    });
}

macro_rules! gemm_i8r {
    ($name:ident, $c:literal, $t:literal, $u:literal, $wg:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            w: *const i8,
            sn: *const f32,
            rn: *const f32,
            xt: *const u8,
            st: *const f32,
            rt: *const f32,
            y: *mut u8,
            r: u32,
            rp: u32,
            k: u32,
            n: u32,
            ldy: u32,
            y_off: u32,
            f32out: u32,
            sb: u32,
        ) {
            attr::reqd_work_group_size::<$wg, 1, 1>();
            gemm_i8r::<$c, $t, $u>(w, sn, rn, xt, st, rt, y, r, rp, k, n, ldy, y_off, f32out, sb)
        }
    };
}
gemm_i8r!(xq_pfgemm_i8r_c4t4, 4, 4, 4, 256);
gemm_i8r!(xq_pfgemm_i8r3_c4t4, 4, 4, 3, 256);
gemm_i8r!(xq_pfgemm_i8r3_c8t4, 8, 4, 3, 512);
gemm_i8r!(xq_pfgemm_i8r3_c4t2, 4, 2, 3, 128);
