//! Inkling-2-Mini kernels: routed-expert MoE on the release's native weight bytes, decode path (1-16 rows).
//!
//! Weights are read as stored: NVFP4 (layers 0-19: u8 codes, even element in the low nibble, one e4m3 scale per 16
//! values along K, a per-expert f32 factor `amax / outer_scale_denominator` applied to every value) and MXFP8
//! (layers 20-23: e4m3 values, one ue8m0 scale `2^(s-127)` per 32 values along K). Both decode exactly like
//! `Inkling2LoopDequantizeExperts`: value = bf16((code * scale) * factor), so the weights the kernels multiply with are
//! the reference's bf16 weights bit for bit; products accumulate in f32, outputs round to bf16 where the reference's
//! `F.linear` does.
//!
//! Every (row, pick) pair is computed on its own, with a fixed summation order (4 lanes per output element, then a
//! fixed butterfly), so a row's bits never depend on the number of rows in the batch. The 8 picks of a row are summed
//! in ascending expert-id order in f32 (the reference's `index_add_` order).
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

extern crate kcommon;

use core::arch::asm;
use kcommon::*;

/// Programmatic dependent launch: wait for the previous grid (its outputs are then visible), and let the next grid start.
/// Discipline of every kernel here: [prologue touching only weights or data from two or more kernels back] -> `pdl_wait()` ->
/// `pdl_trigger()` -> work. A kernel triggers only after its own wait, so when a successor starts, every kernel before its
/// predecessor has completed. No-ops under plain launches.
#[inline(always)]
pub(crate) fn pdl_wait() {
    unsafe { asm!("griddepcontrol.wait;", options(nostack)) }
}
#[inline(always)]
pub(crate) fn pdl_trigger() {
    unsafe { asm!("griddepcontrol.launch_dependents;", options(nostack)) }
}

pub const HID: u32 = 2304;
pub const FFN: u32 = 1152;
pub const TOPK: u32 = 8;
const GATE_CAP: f32 = 4.0;
const UP_CAP: f32 = 25.0;

/// Exact e4m3 -> f32.
#[inline(always)]
fn e4m3_to_f32(b: u32) -> f32 {
    let e = (b >> 3) & 15;
    let m = b & 7;
    let v = if e == 0 { m as f32 * (1.0 / 512.0) } else { f32::from_bits(((e + 120) << 23) | (m << 20)) };
    if b & 0x80 != 0 { -v } else { v }
}

/// 8 e2m1 nibbles of `w` (element 0 in the low nibble) -> 8 exact f32.
#[inline(always)]
fn e2m1x8(w: u32) -> [f32; 8] {
    let (a0, a1, a2, a3, a4, a5, a6, a7): (f32, f32, f32, f32, f32, f32, f32, f32);
    unsafe {
        asm!("{{\n\t.reg .b8 q0, q1, q2, q3;\n\t.reg .b32 h0, h1, h2, h3;\n\t.reg .b16 l, u;\n\t\
              mov.b32 {{q0, q1, q2, q3}}, {w};\n\t\
              cvt.rn.f16x2.e2m1x2 h0, q0;\n\tcvt.rn.f16x2.e2m1x2 h1, q1;\n\t\
              cvt.rn.f16x2.e2m1x2 h2, q2;\n\tcvt.rn.f16x2.e2m1x2 h3, q3;\n\t\
              mov.b32 {{l, u}}, h0;\n\tcvt.f32.f16 {a0}, l;\n\tcvt.f32.f16 {a1}, u;\n\t\
              mov.b32 {{l, u}}, h1;\n\tcvt.f32.f16 {a2}, l;\n\tcvt.f32.f16 {a3}, u;\n\t\
              mov.b32 {{l, u}}, h2;\n\tcvt.f32.f16 {a4}, l;\n\tcvt.f32.f16 {a5}, u;\n\t\
              mov.b32 {{l, u}}, h3;\n\tcvt.f32.f16 {a6}, l;\n\tcvt.f32.f16 {a7}, u;\n\t}}",
            w = in(reg32) w, a0 = out(reg32) a0, a1 = out(reg32) a1, a2 = out(reg32) a2, a3 = out(reg32) a3,
            a4 = out(reg32) a4, a5 = out(reg32) a5, a6 = out(reg32) a6, a7 = out(reg32) a7,
            options(pure, nomem, nostack));
    }
    [a0, a1, a2, a3, a4, a5, a6, a7]
}

/// 4 e4m3 bytes of `w` (element 0 in the low byte) -> 4 exact f32.
#[inline(always)]
fn e4m3x4(w: u32) -> [f32; 4] {
    let (a0, a1, a2, a3): (f32, f32, f32, f32);
    unsafe {
        asm!("{{\n\t.reg .b16 p, q, l, u;\n\t.reg .b32 h0, h1;\n\t\
              mov.b32 {{p, q}}, {w};\n\t\
              cvt.rn.f16x2.e4m3x2 h0, p;\n\tcvt.rn.f16x2.e4m3x2 h1, q;\n\t\
              mov.b32 {{l, u}}, h0;\n\tcvt.f32.f16 {a0}, l;\n\tcvt.f32.f16 {a1}, u;\n\t\
              mov.b32 {{l, u}}, h1;\n\tcvt.f32.f16 {a2}, l;\n\tcvt.f32.f16 {a3}, u;\n\t}}",
            w = in(reg32) w, a0 = out(reg32) a0, a1 = out(reg32) a1, a2 = out(reg32) a2, a3 = out(reg32) a3,
            options(pure, nomem, nostack));
    }
    [a0, a1, a2, a3]
}

/// 2^(s-127) as f32 (s = 0 gives the f32 denormal 2^-127, like `torch.exp2`).
#[inline(always)]
fn ue8m0(s: u32) -> f32 {
    if s == 0 { f32::from_bits(0x0040_0000) } else { f32::from_bits(s << 23) }
}

/// Deterministic tanh from IEEE ops only (exp_det; Taylor series near zero).
#[inline(always)]
fn tanh_det(x: f32) -> f32 {
    let a = x.abs();
    if a < 0.1 {
        let x2 = x * x;
        let mut p = -17.0 / 315.0;
        p = math::fma(p, x2, 2.0 / 15.0);
        p = math::fma(p, x2, -1.0 / 3.0);
        return math::fma(x * x2, p, x);
    }
    let e = math::exp_det(2.0 * a);
    let t = 1.0 - 2.0 / (e + 1.0);
    if x < 0.0 { -t } else { t }
}

/// The reference's softcapped SwiGLU on bf16-rounded gate / up: cap(g) * sigmoid(g) * cap(u), rounded to bf16.
#[inline(always)]
fn swiglu(g: f32, u: f32) -> u16 {
    let sig = 1.0 / (1.0 + math::exp_det(-g));
    let a = tanh_det(g / GATE_CAP) * GATE_CAP;
    let b = tanh_det(u / UP_CAP) * UP_CAP;
    bf16::from_f32(a * sig * b)
}

/// SwiGLU of the tensor-core path (the native kernels' `capped_swiglu`): ((cap(g) * sigmoid(g)) * cap(u)) * gamma, one
/// bf16 rounding; gamma is the routing weight (folded in before the W2 input is quantized).
#[inline(always)]
fn swiglu_gamma(g: f32, u: f32, gamma: f32) -> u16 {
    let sig = 1.0 / (1.0 + math::exp_det(-g));
    let a = GATE_CAP * tanh_det(g / GATE_CAP);
    let b = UP_CAP * tanh_det(u / UP_CAP);
    bf16::from_f32(((a * sig) * b) * gamma)
}

/// 32 bf16 x values (64 bytes) as 4 x 128-bit loads.
#[inline(always)]
unsafe fn ld_x32(x: *const u16) -> [[u32; 4]; 4] {
    let p = x as *const u32;
    [ldg_nc_v4(p), ldg_nc_v4(p.add(4)), ldg_nc_v4(p.add(8)), ldg_nc_v4(p.add(12))]
}

/// acc += sum_i bf16((c[i] * s) * f) * x[i] over 16 NVFP4 values (8 bytes = two words), x as 2 loaded 128-bit groups.
#[inline(always)]
fn dot16_fp4(mut acc: f32, w0: u32, w1: u32, s: f32, f: f32, xa: [u32; 4], xb: [u32; 4]) -> f32 {
    let c0 = e2m1x8(w0);
    let c1 = e2m1x8(w1);
    let x0 = bf16::unpack8(xa);
    let x1 = bf16::unpack8(xb);
    kcommon::unroll!(i in 8 => {
        acc = math::fma(bf16::round((c0[i] * s) * f), x0[i], acc);
    });
    kcommon::unroll!(i in 8 => {
        acc = math::fma(bf16::round((c1[i] * s) * f), x1[i], acc);
    });
    acc
}

/// acc += sum over 16 MXFP8 values (4 words) of (v * sc) * x.
#[inline(always)]
fn dot16_fp8(mut acc: f32, w: [u32; 4], sc: f32, xa: [u32; 4], xb: [u32; 4]) -> f32 {
    let x0 = bf16::unpack8(xa);
    let x1 = bf16::unpack8(xb);
    kcommon::unroll!(q in 2 => {
        let v0 = e4m3x4(w[q * 2]);
        let v1 = e4m3x4(w[q * 2 + 1]);
        let xs = if q == 0 { x0 } else { x1 };
        kcommon::unroll!(i in 4 => {
            acc = math::fma(v0[i] * sc, xs[i], acc);
        });
        kcommon::unroll!(i in 4 => {
            acc = math::fma(v1[i] * sc, xs[4 + i], acc);
        });
    });
    acc
}

/// NVFP4 row dot: `row` points at K/2 code bytes, `srow` at K/16 e4m3 scales. 4 lanes (`lq`) share the row, lane lq
/// takes 32-value granules lq, lq + 4, ...; NIT = K / 128 granules a lane. Returns this lane's partial sum.
#[inline(always)]
unsafe fn row_fp4<const NIT: usize>(row: *const u8, srow: *const u8, f: f32, x: *const u16, lq: u32) -> f32 {
    let mut acc = 0.0f32;
    let mut c = 0;
    while c < NIT / 3 {
        let mut wv = [[0u32; 4]; 3];
        let mut sv = [0u32; 3];
        kcommon::unroll!(i in 3 => {
            let q = lq as usize + 4 * (c * 3 + i);
            wv[i] = ldg_stream_v4(row.add(q * 16) as *const u32);
            sv[i] = *(srow.add(q * 2) as *const u16) as u32;
        });
        kcommon::unroll!(i in 3 => {
            let q = lq as usize + 4 * (c * 3 + i);
            let xv = ld_x32(x.add(q * 32));
            let s0 = e4m3_to_f32(sv[i] & 0xff);
            let s1 = e4m3_to_f32(sv[i] >> 8);
            acc = dot16_fp4(acc, wv[i][0], wv[i][1], s0, f, xv[0], xv[1]);
            acc = dot16_fp4(acc, wv[i][2], wv[i][3], s1, f, xv[2], xv[3]);
        });
        c += 1;
    }
    acc
}

/// MXFP8 row dot: `row` points at K value bytes, `srow` at K/32 ue8m0 scales; lane lq takes 32-value granules
/// lq, lq + 4, ... (NIT = K / 128 of them).
#[inline(always)]
unsafe fn row_fp8<const NIT: usize>(row: *const u8, srow: *const u8, x: *const u16, lq: u32) -> f32 {
    let mut acc = 0.0f32;
    let mut c = 0;
    while c < NIT / 3 {
        let mut wv = [[[0u32; 4]; 2]; 3];
        let mut sv = [0u32; 3];
        kcommon::unroll!(i in 3 => {
            let q = lq as usize + 4 * (c * 3 + i);
            wv[i][0] = ldg_stream_v4(row.add(q * 32) as *const u32);
            wv[i][1] = ldg_stream_v4(row.add(q * 32 + 16) as *const u32);
            sv[i] = *srow.add(q) as u32;
        });
        kcommon::unroll!(i in 3 => {
            let q = lq as usize + 4 * (c * 3 + i);
            let xv = ld_x32(x.add(q * 32));
            let sc = ue8m0(sv[i]);
            acc = dot16_fp8(acc, wv[i][0], sc, xv[0], xv[1]);
            acc = dot16_fp8(acc, wv[i][1], sc, xv[2], xv[3]);
        });
        c += 1;
    }
    acc
}

/// Sum over the 4 lanes of a row: ((l0 + l2) + (l1 + l3)), the same for every row of every batch.
#[inline(always)]
fn red4(v: f32) -> f32 {
    warp::sum_width::<4>(v)
}

/// Gate|up + softcapped SwiGLU, NVFP4 experts. h[pair][1152] bf16, pair = row * 8 + pick; ids[pair] = expert (< 0:
/// the pair is skipped and h left alone). w13 = [128][2304][1152] code bytes (row 2j = gate j, row 2j + 1 = up j, as released),
/// s13 = [128][2304][144] e4m3 scales, fac13 = [128] f32 `amax / outer_scale_denominator`.
/// grid = (1152 / 64, rows * 8), block = 256 (4 lanes per output column, 64 columns a block).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_moe_gu_nvfp4(h: *mut u16, x: *const u16, ldx: u32, ids: *const i32,
    w13: *const u8, s13: *const u8, fac13: *const f32) {
    pdl_wait();
    pdl_trigger();
    let pair = bid_y();
    let e = *ids.add(pair as usize);
    if e < 0 {
        return;
    }
    let e = e as usize;
    let j = (bid_x() * 64 + tid_x() / 4) as usize;
    let lq = tid_x() & 3;
    let xr = x.add((pair / TOPK) as usize * ldx as usize);
    // the release interleaves gate / up along the rows: row 2j = gate j, row 2j + 1 = up j (the loader's Interleave(dim=1))
    let rg = w13.add((e * 2304 + 2 * j) * 1152);
    let ru = w13.add((e * 2304 + 2 * j + 1) * 1152);
    let sg = s13.add((e * 2304 + 2 * j) * 144);
    let su = s13.add((e * 2304 + 2 * j + 1) * 144);
    let f = *fac13.add(e);
    let g = red4(row_fp4::<18>(rg, sg, f, xr, lq));
    let u = red4(row_fp4::<18>(ru, su, f, xr, lq));
    if lq == 0 {
        *h.add(pair as usize * 1152 + j) = swiglu(bf16::round(g), bf16::round(u));
    }
}

/// Down projection, NVFP4 experts: y[pair][2304] bf16 = h[pair][:] . w2[e][n][:]. w2 = [128][2304][576] code bytes,
/// s2 = [128][2304][72] scales, fac2 = [128]. grid = (2304 / 64, rows * 8), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_moe_down_nvfp4(y: *mut u16, h: *const u16, ids: *const i32,
    w2: *const u8, s2: *const u8, fac2: *const f32) {
    pdl_wait();
    pdl_trigger();
    let pair = bid_y();
    let e = *ids.add(pair as usize);
    if e < 0 {
        return;
    }
    let e = e as usize;
    let n = (bid_x() * 64 + tid_x() / 4) as usize;
    let lq = tid_x() & 3;
    let hr = h.add(pair as usize * 1152);
    let a = red4(row_fp4::<9>(w2.add((e * 2304 + n) * 576), s2.add((e * 2304 + n) * 72), *fac2.add(e), hr, lq));
    if lq == 0 {
        *y.add(pair as usize * 2304 + n) = bf16::from_f32(a);
    }
}

/// As `ink_moe_gu_nvfp4` for MXFP8 experts: w13 = [128][2304][2304] e4m3 bytes, s13 = [128][2304][72] ue8m0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_moe_gu_mxfp8(h: *mut u16, x: *const u16, ldx: u32, ids: *const i32,
    w13: *const u8, s13: *const u8) {
    pdl_wait();
    pdl_trigger();
    let pair = bid_y();
    let e = *ids.add(pair as usize);
    if e < 0 {
        return;
    }
    let e = e as usize;
    let j = (bid_x() * 64 + tid_x() / 4) as usize;
    let lq = tid_x() & 3;
    let xr = x.add((pair / TOPK) as usize * ldx as usize);
    let g = red4(row_fp8::<18>(w13.add((e * 2304 + 2 * j) * 2304), s13.add((e * 2304 + 2 * j) * 72), xr, lq));
    let u = red4(row_fp8::<18>(w13.add((e * 2304 + 2 * j + 1) * 2304), s13.add((e * 2304 + 2 * j + 1) * 72), xr, lq));
    if lq == 0 {
        *h.add(pair as usize * 1152 + j) = swiglu(bf16::round(g), bf16::round(u));
    }
}

/// As `ink_moe_down_nvfp4` for MXFP8 experts: w2 = [128][2304][1152] e4m3 bytes, s2 = [128][2304][36] ue8m0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_moe_down_mxfp8(y: *mut u16, h: *const u16, ids: *const i32,
    w2: *const u8, s2: *const u8) {
    pdl_wait();
    pdl_trigger();
    let pair = bid_y();
    let e = *ids.add(pair as usize);
    if e < 0 {
        return;
    }
    let e = e as usize;
    let n = (bid_x() * 64 + tid_x() / 4) as usize;
    let lq = tid_x() & 3;
    let hr = h.add(pair as usize * 1152);
    let a = red4(row_fp8::<9>(w2.add((e * 2304 + n) * 1152), s2.add((e * 2304 + n) * 36), hr, lq));
    if lq == 0 {
        *y.add(pair as usize * 2304 + n) = bf16::from_f32(a);
    }
}

/// out[row][c] (bf16) = bf16( sum over the row's picks in ascending expert-id order of f32(y[row * 8 + k][c]) * wts[row * 8 + k] ),
/// f32 accumulation from 0 (the reference's `index_add_` into a zero tensor). Picks with id < 0 are skipped.
/// grid = (ceil(2304 / 256), rows), block = 256. `ldo` = out row stride in elements.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_moe_combine(out: *mut u16, ldo: u32, y: *const u16, ids: *const i32,
    wts: *const f32) {
    pdl_wait();
    pdl_trigger();
    let row = bid_y() as usize;
    let c = (bid_x() * 256 + tid_x()) as usize;
    if c >= 2304 {
        return;
    }
    let mut id = [0u32; 8];
    let mut yv = [0.0f32; 8];
    let mut wv = [0.0f32; 8];
    kcommon::unroll!(k in 8 => {
        id[k] = *ids.add(row * 8 + k) as u32; // negative ids sort last
        yv[k] = bf16::to_f32(*y.add((row * 8 + k) * 2304 + c));
        wv[k] = *wts.add(row * 8 + k);
    });
    let mut rank = [0u32; 8];
    kcommon::unroll!(k in 8 => {
        let mut r = 0u32;
        kcommon::unroll!(j in 8 => {
            if id[j] < id[k] || (id[j] == id[k] && j < k) {
                r += 1;
            }
        });
        rank[k] = r;
    });
    let mut acc = 0.0f32;
    kcommon::unroll!(r in 8 => {
        kcommon::unroll!(k in 8 => {
            if rank[k] == r as u32 && (id[k] as i32) >= 0 {
                acc += yv[k] * wv[k];
            }
        });
    });
    *out.add(row * ldo as usize + c) = bf16::from_f32(acc);
}

mod moe_gemm;
mod plan;
mod quant;

/// f32 -> e4m3 (round to nearest even, saturate to +-448), in the low byte.
#[inline(always)]
fn to_e4m3(x: f32) -> u32 {
    let h: u16;
    unsafe {
        asm!("cvt.rn.satfinite.e4m3x2.f32 {h}, {z}, {x};", h = out(reg16) h, z = in(reg32) 0.0f32, x = in(reg32) x,
            options(pure, nomem, nostack));
    }
    (h & 0xff) as u32
}

/// Eight f32 -> eight e2m1 nibbles (round to nearest even, saturate to +-6), x[0] in the low nibble.
#[inline(always)]
fn to_e2m1x8(x: [f32; 8]) -> u32 {
    let r: u32;
    unsafe {
        asm!("{{\n\t.reg .b8 q0, q1, q2, q3;\n\t\
              cvt.rn.satfinite.e2m1x2.f32 q0, {x1}, {x0};\n\t\
              cvt.rn.satfinite.e2m1x2.f32 q1, {x3}, {x2};\n\t\
              cvt.rn.satfinite.e2m1x2.f32 q2, {x5}, {x4};\n\t\
              cvt.rn.satfinite.e2m1x2.f32 q3, {x7}, {x6};\n\t\
              mov.b32 {r}, {{q0, q1, q2, q3}};\n\t}}",
            r = out(reg32) r, x0 = in(reg32) x[0], x1 = in(reg32) x[1], x2 = in(reg32) x[2], x3 = in(reg32) x[3],
            x4 = in(reg32) x[4], x5 = in(reg32) x[5], x6 = in(reg32) x[6], x7 = in(reg32) x[7],
            options(pure, nomem, nostack));
    }
    r
}

/// Byte offset of NVFP4 scale (row, 16-block b) in the tiled layout (kb = K/16): for row tile rt (128 rows) and
/// k-tile kt (8 blocks) one contiguous 1 KB block [128 rows][8 scales].
#[inline(always)]
fn sf_tiled(row: u32, b: u32, kb: u32) -> usize {
    (((row >> 7) * (kb >> 3) + (b >> 3)) as usize) * 1024 + ((row & 127) * 8 + (b & 7)) as usize
}

mod dense;

mod requant;

mod moe_gemm_sw;

mod dense_s8;

mod router;

mod glue;
