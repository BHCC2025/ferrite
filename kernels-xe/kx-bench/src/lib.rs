//! X0 ceilings, kernel for kernel the arc-engine probes (same access pattern, same launch geometry), so the Rust ->
//! SPIR-V route is compared with ocloc's on equal terms:
//! * `bw_probe` = ggml_quant.cl `bw_probe` (DRAM streaming read: 16-byte lane loads, 4 in flight; `xpu-ggml-test --probe`);
//!   `bw_blk` streams with 512-byte sub-group block reads instead (the XQ weight-read pattern).
//! * `lb_*` = qwen_lbench.cl (load-primitive throughput from a cache-resident region).
//! * `me_*` = qwen_mebench.cl (matrix-engine ceilings, random register operands).
//! * `dpas_loop` / `dpas_tile` = qwen_dpasbench.cl.
#![no_std]
#![feature(abi_gpu_kernel, portable_simd)]
#![allow(clippy::missing_safety_doc)]
use kxcommon::v::*;
use kxcommon::{bits, blk, blk2d, dpas, id, unroll};

// ---- DRAM streaming ----------------------------------------------------------------------------

/// Every sub-group streams `units_per_sg` consecutive 16-byte units, 4 units in flight a lane, xor-folds them.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bw_probe(w: *const U32x4, out: *mut u32, units_per_sg: u32) {
    let sg = id::group_id(0) * 4 + id::sg_id();
    let lane = id::lane() as usize;
    let p = w.add(sg as usize * units_per_sg as usize);
    let mut acc = 0u32;
    let mut i = 0usize;
    while i < units_per_sg as usize {
        let a = *p.add(i + lane);
        let b = *p.add(i + 16 + lane);
        let c = *p.add(i + 32 + lane);
        let d = *p.add(i + 48 + lane);
        let x = a ^ b ^ c ^ d;
        acc = acc.wrapping_add(x[0] ^ x[1] ^ x[2] ^ x[3]);
        i += 64;
    }
    if acc == 0x1234_5678 {
        *out = acc;
    }
}

/// As `bw_probe` with sub-group block reads of 8 dwords (512 B a message), 4 messages in flight a sub-group.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bw_blk(w: *const u32, out: *mut u32, dwords_per_sg: u32) {
    let sg = id::group_id(0) * 4 + id::sg_id();
    let p = w.add(sg as usize * dwords_per_sg as usize);
    let mut acc = U32x8::splat(0);
    let mut i = 0usize;
    while i < dwords_per_sg as usize {
        let a = blk::global::read_u32x8(p.add(i));
        let b = blk::global::read_u32x8(p.add(i + 128));
        let c = blk::global::read_u32x8(p.add(i + 256));
        let d = blk::global::read_u32x8(p.add(i + 384));
        acc ^= a ^ b ^ c ^ d;
        i += 512;
    }
    let x = acc.to_array();
    let s = x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[4] ^ x[5] ^ x[6] ^ x[7];
    if s == 0x1234_5678 {
        *out = s;
    }
}

// ---- load primitives (qwen_lbench.cl) ----------------------------------------------------------

/// 512 B a load.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn lb_blk8(p: *const u32, out: *mut u32, iters: u32, mask: u32) {
    let g = id::group_id(0);
    let mut x = 0u32;
    for i in 0..iters {
        let r = blk::global::read_u32x8(p.add(((g.wrapping_mul(7).wrapping_add(i.wrapping_mul(3))) & mask) as usize * 128)).to_array();
        x ^= r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4] ^ r[5] ^ r[6] ^ r[7];
    }
    *out.add(id::global_id(0)) = x;
}
/// 256 B a load (16 B a lane).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn lb_vl8(p: *const u16, out: *mut u32, iters: u32, mask: u32) {
    let g = id::group_id(0);
    let lane = id::lane() as usize;
    let mut x = 0u32;
    let q = p as *const U16x8;
    for i in 0..iters {
        let r = (*q.add(((g.wrapping_mul(7).wrapping_add(i.wrapping_mul(3))) & mask) as usize * 16 + lane)).to_array();
        x ^= (r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4] ^ r[5] ^ r[6] ^ r[7]) as u32;
    }
    *out.add(id::global_id(0)) = x;
}
/// 1 KB a load (64 B a lane).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn lb_vl16(p: *const u32, out: *mut u32, iters: u32, mask: u32) {
    let g = id::group_id(0);
    let lane = id::lane() as usize;
    let mut x = 0u32;
    let q = p as *const U32x16;
    for i in 0..iters {
        let r = (*q.add(((g.wrapping_mul(7).wrapping_add(i.wrapping_mul(3))) & mask) as usize * 16 + lane)).to_array();
        x ^= r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4] ^ r[5] ^ r[6] ^ r[7] ^ r[8] ^ r[15];
    }
    *out.add(id::global_id(0)) = x;
}
/// 8 rows x 32 cols x 16 bit = 512 B.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn lb_2d_8r16x2c(p: *const u16, out: *mut u32, iters: u32, mask: u32, w: i32, h: i32, pitch: i32) {
    let g = id::group_id(0);
    let s = blk2d::Surface { base: p as *const u8, width_bytes: w, height: h, pitch_bytes: pitch };
    let mut x = 0u32;
    for i in 0..iters {
        let t = g.wrapping_mul(7).wrapping_add(i.wrapping_mul(3)) & mask;
        let v = blk2d::read_16b_8r16x2c(s, ((t & 63) * 32) as i32, ((t >> 6) * 8) as i32);
        x ^= (v[0] ^ v[1] ^ v[2] ^ v[3] ^ v[4] ^ v[5] ^ v[6] ^ v[7] ^ v[8] ^ v[15]) as u32;
    }
    *out.add(id::global_id(0)) = x;
}
/// 32 rows x 32 cols x 16 bit = 2 KB.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn lb_2d_32r16x2c(p: *const u16, out: *mut u32, iters: u32, mask: u32, w: i32, h: i32, pitch: i32) {
    let g = id::group_id(0);
    let s = blk2d::Surface { base: p as *const u8, width_bytes: w, height: h, pitch_bytes: pitch };
    let mut x = 0u32;
    for i in 0..iters {
        let t = g.wrapping_mul(7).wrapping_add(i.wrapping_mul(3)) & mask;
        let v = blk2d::read_16b_32r16x2c(s, ((t & 63) * 32) as i32, ((t >> 6) * 32) as i32);
        x ^= (v[0] ^ v[1] ^ v[7] ^ v[8] ^ v[15] ^ v[16] ^ v[31]) as u32;
    }
    *out.add(id::global_id(0)) = x;
}
/// VNNI transform, 16 rows x 32 cols x 16 bit = 1 KB.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn lb_2d_tr16x2c(p: *const u16, out: *mut u32, iters: u32, mask: u32, w: i32, h: i32, pitch: i32) {
    let g = id::group_id(0);
    let s = blk2d::Surface { base: p as *const u8, width_bytes: w, height: h, pitch_bytes: pitch };
    let mut x = 0u32;
    for i in 0..iters {
        let t = g.wrapping_mul(7).wrapping_add(i.wrapping_mul(3)) & mask;
        let v = blk2d::read_transform_16b_16r16x2c(s, ((t & 63) * 32) as i32, ((t >> 6) * 16) as i32);
        x ^= v[0] ^ v[1] ^ v[2] ^ v[3] ^ v[4] ^ v[5] ^ v[6] ^ v[7] ^ v[8] ^ v[15];
    }
    *out.add(id::global_id(0)) = x;
}

// ---- matrix engine (qwen_mebench.cl) -----------------------------------------------------------

macro_rules! me_body {
    ($src:ident, $out:ident, $iters:ident, $mad:path, $acc0:expr, $sum:expr) => {{
        let lane = id::lane() as usize;
        let sgi = id::global_id(0) / 16;
        let mut a = [I16x8::splat(0); 4];
        let mut b = [I32x8::splat(0); 4];
        unroll!(4, 4, |i| {
            a[i] = bits(*($src.add(((sgi * 8 + i) & 4095) * 256 + lane * 8) as *const U32x4));
            b[i] = bits(*($src.add(((sgi * 8 + 4 + i) & 4095) * 256 + lane * 8) as *const U32x8));
        });
        let mut acc = [[$acc0; 4]; 4];
        for _ in 0..$iters {
            unroll!(4, 4, |i| unroll!(4, 4, |j| acc[i][j] = $mad(a[i], b[j], acc[i][j])));
        }
        let mut s = 0.0f32;
        unroll!(4, 4, |i| unroll!(4, 4, |j| s += $sum(acc[i][j])));
        *$out.add(id::global_id(0)) = s;
    }};
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn me_f16(src: *const u32, out: *mut f32, iters: u32) {
    me_body!(src, out, iters, dpas::f16_m8, F32x8::splat(0.0), |v: F32x8| v[0] + v[7])
}
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn me_bf16(src: *const u32, out: *mut f32, iters: u32) {
    me_body!(src, out, iters, dpas::bf16_m8, F32x8::splat(0.0), |v: F32x8| v[0] + v[7])
}
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn me_i8(src: *const u32, out: *mut f32, iters: u32) {
    me_body!(src, out, iters, dpas::i8_m8, I32x8::splat(0), |v: I32x8| v[0] as f32 + v[7] as f32)
}

// ---- DPAS ceilings (qwen_dpasbench.cl) ---------------------------------------------------------

/// 8 independent accumulators, operands in registers (pure matrix-engine rate).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn dpas_loop(out: *mut f32, iters: u32) {
    let lane = id::lane();
    let a = I16x8::splat((lane + 0x3f00) as i16);
    let b = I32x8::splat((0x3f80_3f80u32 + lane) as i32);
    let mut acc = [F32x8::splat(0.0); 8];
    for _ in 0..iters {
        unroll!(8, 8, |j| acc[j] = dpas::bf16_m8(a, b, acc[j]));
    }
    let mut s = 0.0f32;
    unroll!(8, 8, |j| s += acc[j][0] + acc[j][7]);
    *out.add(id::global_id(0)) = s;
}

/// A 32 x 64 output micro-tile per sub-group (4 x 4 DPAS tiles of 8 x 16), operands from a small cache-resident buffer.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn dpas_tile(a_: *const u16, b_: *const u16, out: *mut f32, iters: u32) {
    let lane = id::lane() as usize;
    let wg = id::group_id(0);
    let mut acc = [[F32x8::splat(0.0); 4]; 4];
    let bp = b_ as *const u32;
    for k in 0..iters {
        let o = (k.wrapping_add(wg) & 63) as usize;
        let mut a = [I16x8::splat(0); 4];
        let mut b = [I32x8::splat(0); 4];
        unroll!(4, 4, |i| a[i] = *((a_ as *const i16).add(((o * 4 + i) * 16 + lane) * 8) as *const I16x8));
        unroll!(4, 4, |j| b[j] = bits(*(bp.add((o * 4 + j) * 128 + lane * 8) as *const U32x8)));
        unroll!(4, 4, |i| unroll!(4, 4, |j| acc[i][j] = dpas::bf16_m8(a[i], b[j], acc[i][j])));
    }
    let mut s = 0.0f32;
    unroll!(4, 4, |i| unroll!(4, 4, |j| s += acc[i][j][0] + acc[i][j][7]));
    *out.add(id::global_id(0)) = s;
}
