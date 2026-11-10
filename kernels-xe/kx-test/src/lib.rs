//! X0 feature proofs: one kernel per hardware feature, each checked against a CPU reference by
//! `crates/fe-kx-tests` (`kx-feature-test`). Sub-group size 16 throughout.
#![no_std]
#![feature(abi_gpu_kernel, portable_simd)]
#![allow(clippy::missing_safety_doc)]
use kxcommon::sg::Shuffle;
use kxcommon::v::*;
use kxcommon::{attr, blk, blk2d, cvt, dpas, id, math, sg, slm};

/// ids: out[g * 8 ..] = global id, local id, group id, lane, sub-group id, sub-groups, sub-group size, local size.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn t_ids(out: *mut u32) {
    let g = id::global_id(0);
    let o = out.add(g * 8);
    *o = g as u32;
    *o.add(1) = id::local_id(0);
    *o.add(2) = id::group_id(0);
    *o.add(3) = id::lane();
    *o.add(4) = id::sg_id();
    *o.add(5) = id::num_sg();
    *o.add(6) = id::sg_size();
    *o.add(7) = id::local_size(0);
}

/// Sub-group shuffles, broadcast, reductions and scans (float and uint).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn t_sg(x: *const f32, u: *const u32, out: *mut f32, outu: *mut u32) {
    let g = id::global_id(0);
    let lane = id::lane();
    let v = *x.add(g);
    let w = *u.add(g);
    let o = out.add(g * 10);
    *o = v.shuffle((lane * 3) & 15);
    *o.add(1) = v.shuffle_xor(5);
    *o.add(2) = sg::broadcast_f32(v, 7);
    *o.add(3) = sg::reduce_add_f32(v);
    *o.add(4) = sg::reduce_max_f32(v);
    *o.add(5) = sg::reduce_min_f32(v);
    *o.add(6) = sg::scan_incl_add_f32(v);
    *o.add(7) = sg::scan_excl_add_f32(v);
    *o.add(8) = sg::shuffle_down_f32(v, v + 100.0, 3);
    *o.add(9) = sg::shuffle_up_f32(v - 100.0, v, 3);
    let ou = outu.add(g * 5);
    *ou = w.shuffle(15 - lane);
    *ou.add(1) = w.shuffle_xor(3);
    *ou.add(2) = sg::reduce_add_u32(w);
    *ou.add(3) = sg::scan_incl_add_u32(w);
    *ou.add(4) = sg::any(w == 7) as u32 + 2 * sg::all(w < 1_000_000) as u32;
}

/// SLM: a static region and a dynamic (`__local` parameter) region, a barrier, a split barrier; work-group 256.
/// out[g] = (s[255 - l] + d[(l + 1) & 255]) of the lane l ^ 1, where s = x of the group and d = 2 x.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn t_slm(x: *const f32, out: *mut f32, dynbuf: *mut f32) {
    attr::reqd_work_group_size::<256, 1, 1>();
    let d = attr::local_arg(dynbuf);
    let s = slm::alloc::<f32, 256, 0>();
    let l = id::local_id(0) as usize;
    let g = id::global_id(0);
    let xv = *x.add(g);
    *s.add(l) = xv;
    *d.add(l) = xv * 2.0;
    slm::barrier();
    let a = *s.add(255 - l) + *d.add((l + 1) & 255);
    slm::barrier_arrive(slm::LOCAL);
    slm::barrier_wait(slm::LOCAL);
    *s.add(l) = a;
    slm::barrier();
    *out.add(g) = *s.add(l ^ 1);
}

/// Block reads / writes, one sub-group a work-group, 128 dwords (and 128 ushorts) a sub-group:
/// copy[gi] = block_write8(block_read8(src[gi])) (identity); chk[gi][lane][7] = read1, read2, read4 of the run;
/// copy16 likewise for ushort x8 and chk16[gi][lane] = read_us; lcopy[gi] = the run through SLM with local block IO.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn t_blk(
    src: *const u32,
    src16: *const u16,
    copy: *mut u32,
    chk: *mut u32,
    copy16: *mut u16,
    chk16: *mut u16,
    lcopy: *mut u32,
) {
    attr::reqd_work_group_size::<16, 1, 1>();
    let gi = id::group_id(0) as usize;
    let lane = id::lane() as usize;
    let p = src.add(gi * 128);
    let a8 = blk::global::read_u32x8(p);
    blk::global::write_u32x8(copy.add(gi * 128), a8);
    let a1 = blk::global::read_u32(p);
    let a2 = blk::global::read_u32x2(p).to_array();
    let a4 = blk::global::read_u32x4(p).to_array();
    let c = chk.add((gi * 16 + lane) * 7);
    *c = a1;
    *c.add(1) = a2[0];
    *c.add(2) = a2[1];
    for i in 0..4 {
        *c.add(3 + i) = a4[i];
    }
    let p16 = src16.add(gi * 128);
    blk::global::write_u16x8(copy16.add(gi * 128), blk::global::read_u16x8(p16));
    *chk16.add(gi * 16 + lane) = blk::global::read_u16(p16);
    // SLM round trip: lane writes its 8 dwords of the run with a local block write, reads them back as a block
    let s = slm::alloc::<u32, 128, 0>();
    blk::local::write_u32x8(s, a8);
    slm::barrier();
    let r = blk::local::read_u32x8(s);
    blk::global::write_u32x8(lcopy.add(gi * 128), r);
}

/// DPAS fp16 / bf16 (M = 8, 4, 2, 1) and int8 (M = 8, 1), one sub-group.
/// a: [8][16] 16-bit (row r, lane = k), b: [8 dwords][16 lanes] VNNI (dword d of lane n = B[2d][n] | B[2d+1][n] << 16),
/// c: [M][16] fp32 per form (8 + 4 + 2 + 1 rows) for fp16 then bf16. a8: [8][16] shorts (2 int8 each), b8: [8][16] dwords
/// (4 int8 each, B[4d + j][n] in byte j), c8: [8 + 1][16] int32.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn t_dpas(a: *const u16, b: *const u32, ab: *const u16, bb: *const u32, c: *mut f32, a8: *const u16, b8: *const u32, c8: *mut i32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    let lane = id::lane() as usize;
    let ld_a = |p: *const u16| -> [i16; 8] { core::array::from_fn(|r| *p.add(r * 16 + lane) as i16) };
    let ld_b = |p: *const u32| -> I32x8 { I32x8::from_array(core::array::from_fn(|d| *p.add(d * 16 + lane) as i32)) };
    let mut row = 0usize;
    for (pa, pb, bf) in [(a, b, false), (ab, bb, true)] {
        let av = ld_a(pa);
        let bv = ld_b(pb);
        let r8 = if bf { dpas::bf16_m8(I16x8::from_array(av), bv, F32x8::splat(0.0)) } else { dpas::f16_m8(I16x8::from_array(av), bv, F32x8::splat(0.0)) };
        let r4 = if bf {
            dpas::bf16_m4(I16x4::from_array([av[0], av[1], av[2], av[3]]), bv, F32x4::splat(0.0))
        } else {
            dpas::f16_m4(I16x4::from_array([av[0], av[1], av[2], av[3]]), bv, F32x4::splat(0.0))
        };
        let r2 = if bf { dpas::bf16_m2(I16x2::from_array([av[0], av[1]]), bv, F32x2::splat(0.0)) } else { dpas::f16_m2(I16x2::from_array([av[0], av[1]]), bv, F32x2::splat(0.0)) };
        let r1 = if bf { dpas::bf16_m1(av[0], bv, 0.0) } else { dpas::f16_m1(av[0], bv, 0.0) };
        let r8 = r8.to_array();
        let r4 = r4.to_array();
        let r2 = r2.to_array();
        for r in 0..8 {
            *c.add((row + r) * 16 + lane) = r8[r];
        }
        for r in 0..4 {
            *c.add((row + 8 + r) * 16 + lane) = r4[r];
        }
        for r in 0..2 {
            *c.add((row + 12 + r) * 16 + lane) = r2[r];
        }
        *c.add((row + 14) * 16 + lane) = r1;
        row += 15;
    }
    let av = ld_a(a8);
    let bv = ld_b(b8);
    let r8 = dpas::i8_m8(I16x8::from_array(av), bv, I32x8::splat(0)).to_array();
    for r in 0..8 {
        *c8.add(r * 16 + lane) = r8[r];
    }
    *c8.add(8 * 16 + lane) = dpas::i8_m1(av[0], bv, 0);
}

/// 2D block IO on a 64 x 64 surface of 16-bit values (pitch 128 B) at (x0, y0):
/// out[0..16][16] = read_16b_16r16x1c, out[16..32] = read_16b_8r16x2c, out32[0..8][16] = read_transform_16b_16r16x1c
/// (as dwords), then the 8 x 16 block at (x0, y0) is written with write_16b_8r16x1c to `dst` (same surface shape).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn t_blk2d(src: *const u16, out: *mut u16, out32: *mut u32, dst: *mut u16, x0: i32, y0: i32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    let lane = id::lane() as usize;
    let s = blk2d::Surface { base: src as *const u8, width_bytes: 128, height: 64, pitch_bytes: 128 };
    blk2d::prefetch_16b_16r16x2c(s, x0, y0);
    let v = blk2d::read_16b_16r16x1c(s, x0, y0);
    for r in 0..16 {
        *out.add(r * 16 + lane) = v[r];
    }
    let w = blk2d::read_16b_8r16x2c(s, x0, y0);
    for r in 0..16 {
        *out.add((16 + r) * 16 + lane) = w[r];
    }
    let t = blk2d::read_transform_16b_16r16x1c(s, x0, y0).to_array();
    for d in 0..8 {
        *out32.add(d * 16 + lane) = t[d];
    }
    let mut row = [0u16; 8];
    row.copy_from_slice(&v[..8]);
    blk2d::write_16b_8r16x1c(dst as *mut u8, 128, 64, 128, x0, y0, U16x8::from_array(row));
}

/// Conversions: per element h = fp16(x) RTE bits, b = bf16(x) by integer RNE, bh = bf16(x) hardware, back[0] = fp32(h),
/// back[1] = fp32(b).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn t_cvt(x: *const f32, h: *mut u16, b: *mut u16, bh: *mut u16, back: *mut f32) {
    let g = id::global_id(0);
    let v = *x.add(g);
    let hv = cvt::f16_bits_rte(v);
    let bv = cvt::f32_to_bf16(v);
    *h.add(g) = hv;
    *b.add(g) = bv;
    *bh.add(g) = cvt::f32_to_bf16_hw(v);
    *back.add(2 * g) = cvt::f16_bits_to_f32(hv);
    *back.add(2 * g + 1) = cvt::bf16_to_f32(bv);
}

/// Rounding control: out[g][0] = a * b + c (two roundings), [1] = fma, [2] = mad, [3] = a / b (IEEE), [4] = OpenCL a / b,
/// [5] = exp(a), [6] = rsqrt(|b|), [7] = (a * b) + (c * a) (no contraction across either product).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn t_fp(a: *const f32, b: *const f32, c: *const f32, out: *mut f32) {
    let g = id::global_id(0);
    let (x, y, z) = (*a.add(g), *b.add(g), *c.add(g));
    let o = out.add(g * 8);
    *o = x * y + z;
    *o.add(1) = math::fma(x, y, z);
    *o.add(2) = math::mad(x, y, z);
    *o.add(3) = x / y;
    *o.add(4) = math::div_ocl(x, y);
    *o.add(5) = math::exp(x);
    *o.add(6) = math::rsqrt(math::fabs(y));
    *o.add(7) = x * y + z * x;
}
