//! NVFP4 activation quantization into the grouped, tiled layout the MoE GEMMs read, and the load-time weight repack.
//! The block rule lives in `quant_block` (one place).
use super::*;
use super::plan::NONE;

const FLT_MAX: f32 = 3.4028234663852886e38;

/// f32 -> (e4m3 bits, value) with the native kernel's `e4m3_round`.
#[inline(always)]
fn e4m3_round(v: f32) -> (u32, f32) {
    let b = to_e4m3(v);
    (b, e4m3_to_f32(b))
}

/// e2m1 round of 2 values (low, high) -> (packed byte, decoded low, decoded high).
#[inline(always)]
fn e2m1_pair(lo: f32, hi: f32) -> (u32, f32, f32) {
    let (p, h): (u32, u32);
    unsafe {
        asm!("{{\n\t.reg .b8 b;\n\tcvt.rn.satfinite.e2m1x2.f32 b, {hi}, {lo};\n\tcvt.u32.u8 {p}, b;\n\tcvt.rn.f16x2.e2m1x2 {h}, b;\n\t}}",
            p = out(reg32) p, h = out(reg32) h, lo = in(reg32) lo, hi = in(reg32) hi, options(pure, nomem, nostack));
    }
    let (l, u): (f32, f32);
    unsafe {
        asm!("{{\n\t.reg .b16 l, u;\n\tmov.b32 {{l, u}}, {h};\n\tcvt.f32.f16 {a}, l;\n\tcvt.f32.f16 {b}, u;\n\t}}",
            h = in(reg32) h, a = out(reg32) l, b = out(reg32) u, options(pure, nomem, nostack));
    }
    (p & 0xff, l, u)
}

/// The release's NVFP4 activation rule (SGLang `quantize_part`, `fouroversix_mae`), one 16-value block per call:
/// `amax` = the row's max |x|, `bm` the block's. Returns (e4m3 scale bits, 16 codes as 2 words). Every op is an IEEE
/// f32 op in the order of the native kernel; the 16 absolute errors are summed as an adjacent-pair tree.
#[inline(always)]
pub(crate) fn quant_block(x: [f32; 16], amax: f32, bm: f32) -> (u32, [u32; 2]) {
    let encoder = (1536.0f32 / amax).min(FLT_MAX);
    let decoder = 1.0f32 / encoder;
    let outer = amax * (1.0f32 / 1536.0f32);
    let (bits6, scale6) = e4m3_round((bm * (encoder * (1.0f32 / 6.0f32))).min(448.0));
    let (bits4, scale4) = e4m3_round((bm * (encoder * 0.25f32)).min(448.0));
    let inv6 = (1.0f32 / (scale6 * decoder)).min(FLT_MAX);
    let inv4 = (1.0f32 / (scale4 * decoder)).min(FLT_MAX);
    let mut p6 = [0u32; 2];
    let mut p4 = [0u32; 2];
    let mut e6 = [0.0f32; 16];
    let mut e4 = [0.0f32; 16];
    kcommon::unroll!(i in 8 => {
        let (c6, r0, r1) = e2m1_pair(x[2 * i] * inv6, x[2 * i + 1] * inv6);
        p6[i / 4] |= c6 << (8 * (i % 4));
        e6[2 * i] = ((r0 * scale6) * outer - x[2 * i]).abs();
        e6[2 * i + 1] = ((r1 * scale6) * outer - x[2 * i + 1]).abs();
        let (c4, q0, q1) = e2m1_pair(x[2 * i] * inv4, x[2 * i + 1] * inv4);
        p4[i / 4] |= c4 << (8 * (i % 4));
        e4[2 * i] = ((q0 * scale4) * outer - x[2 * i]).abs();
        e4[2 * i + 1] = ((q1 * scale4) * outer - x[2 * i + 1]).abs();
    });
    // adjacent-pair tree, in place like the native kernel
    kcommon::unroll!(i in 8 => {
        e6[i] = e6[2 * i] + e6[2 * i + 1];
        e4[i] = e4[2 * i] + e4[2 * i + 1];
    });
    kcommon::unroll!(i in 4 => {
        e6[i] = e6[2 * i] + e6[2 * i + 1];
        e4[i] = e4[2 * i] + e4[2 * i + 1];
    });
    kcommon::unroll!(i in 2 => {
        e6[i] = e6[2 * i] + e6[2 * i + 1];
        e4[i] = e4[2 * i] + e4[2 * i + 1];
    });
    e6[0] = e6[0] + e6[1];
    e4[0] = e4[0] + e4[1];
    if e4[0] < e6[0] { (bits4, p4) } else { (bits6, p6) }
}

/// bf16 rows -> NVFP4 codes [pos][K/2], tiled scales (kb = K/16, rows padded to 128) and the row's amax arow[pos] (the
/// outer scale input), one warp per (row, pick) pair p: destination position pos_of_pair[p] (NONE: skipped); gather = 1 ->
/// source row p / 8 (x into the experts), gather = 0 -> source row = the destination position (the SwiGLU output, already
/// in grouped order). grid = ceil(npairs / 8), block = 256. K % 128 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_quant_nvfp4(codes: *mut u8, sf: *mut u8, arow: *mut f32, x: *const u16, ldx: u32,
    pos_of_pair: *const u32, gather: u32, npairs: u32, k: u32) {
    pdl_wait();
    pdl_trigger();
    let pr = bid_x() * 8 + warp_id();
    if pr >= npairs {
        return;
    }
    let d = *pos_of_pair.add(pr as usize);
    if d == NONE {
        return;
    }
    let src = if gather != 0 { pr / 8 } else { d };
    let xr = x.add(src as usize * ldx as usize);
    let kb = k >> 4;
    let l = lane();
    let mut amax = 0.0f32;
    let mut b = l;
    while b < kb {
        let p = xr.add(b as usize * 16) as *const u32;
        let f0 = bf16::unpack8(ldg_nc_v4(p));
        let f1 = bf16::unpack8(ldg_nc_v4(p.add(4)));
        kcommon::unroll!(j in 8 => {
            amax = amax.max(f0[j].abs()).max(f1[j].abs());
        });
        b += 32;
    }
    amax = warp::max(amax);
    b = l;
    while b < kb {
        let p = xr.add(b as usize * 16) as *const u32;
        let f0 = bf16::unpack8(ldg_nc_v4(p));
        let f1 = bf16::unpack8(ldg_nc_v4(p.add(4)));
        let mut v = [0.0f32; 16];
        let mut bm = 0.0f32;
        kcommon::unroll!(j in 8 => {
            v[j] = f0[j];
            v[8 + j] = f1[j];
            bm = bm.max(f0[j].abs()).max(f1[j].abs());
        });
        let (s8, q) = quant_block(v, amax, bm);
        let w = codes.add(d as usize * (k as usize >> 1) + b as usize * 8) as *mut u32;
        *w = q[0];
        *w.add(1) = q[1];
        *sf.add(sf_tiled(d, b, kb)) = s8 as u8;
        b += 32;
    }
    if l == 0 {
        *arow.add(d as usize) = amax;
    }
}

/// Byte offset of 16-byte chunk `i` of weight row `r` in the tile-contiguous code layout used by the grouped GEMMs:
/// [row tile (128 rows)][k tile (64 bytes of every row)][128 rows][64 bytes], so one pipeline stage of the weights is a
/// single contiguous 8 KB read. `rowbytes` = bytes per row (K / 2 for NVFP4, K for MXFP8).
#[inline(always)]
pub(crate) fn tiled_code_off(r: u32, i: u32, rowbytes: u32) -> usize {
    let ktiles = rowbytes >> 6;
    (((r >> 7) * ktiles + (i >> 2)) as usize) * 8192 + ((r & 127) * 64 + (i & 3) * 16) as usize
}

/// Load-time retile of weight codes only (w2: rows keep their order). src [128][rows][rowbytes] -> dst tile-contiguous per
/// expert. grid = (rows, 128), block = 128. rowbytes % 64 == 0, rows % 128 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_retile_codes(dst: *mut u8, src: *const u8, rows: u32, rowbytes: u32) {
    let r = bid_x();
    let e = bid_y() as usize;
    let mut i = tid_x();
    while i < rowbytes / 16 {
        let v = *(src.add((e * rows as usize + r as usize) * rowbytes as usize + i as usize * 16) as *const [u32; 4]);
        *(dst.add(e * rows as usize * rowbytes as usize + tiled_code_off(r, i, rowbytes)) as *mut [u32; 4]) = v;
        i += 128;
    }
}

/// Load-time repack of NVFP4 w13 for the grouped GEMM: rows interleaved in blocks of 8 (gate f..f+7, up f..f+7, ...)
/// and the scales tiled. src = the release's [128][2304][1152] codes / [128][2304][144] scales; dst codes
/// [128][2304][1152], dst scales tiled per expert (2304 rows x 144 = 331776 bytes each).
/// grid = (2304, 128), block = 128.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_repack_w13_nvfp4(dcodes: *mut u8, dsf: *mut u8, scodes: *const u8, ssf: *const u8) {
    let r = bid_x(); // destination row
    let e = bid_y() as usize;
    let q = r / 16;
    let j = r % 16;
    // released rows: 2f = gate f, 2f + 1 = up f; destination: blocks of 8 gate rows then 8 up rows
    let srow = if j < 8 { 2 * (8 * q + j) } else { 2 * (8 * q + j - 8) + 1 };
    let t = tid_x();
    let s = scodes.add((e * 2304 + srow as usize) * 1152);
    if t < 72 {
        *(dcodes.add(e * 2304 * 1152 + tiled_code_off(r, t, 1152)) as *mut [u32; 4]) = *(s.add(t as usize * 16) as *const [u32; 4]);
    }
    let mut b = t;
    while b < 144 {
        let v = *ssf.add((e * 2304 + srow as usize) * 144 + b as usize);
        *dsf.add(e * 331776 + sf_tiled(r, b, 144)) = v;
        b += 128;
    }
}

/// Load-time scale retile only (w2: codes are used as stored). src [128][rows][kb], dst tiled per expert.
/// grid = (rows, 128), block = 128. rows % 128 == 0, kb % 8 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_retile_sf(dsf: *mut u8, ssf: *const u8, rows: u32, kb: u32) {
    let r = bid_x();
    let e = bid_y() as usize;
    let t = tid_x();
    let mut b = t;
    while b < kb {
        let v = *ssf.add((e * rows as usize + r as usize) * kb as usize + b as usize);
        *dsf.add(e * rows as usize * kb as usize + sf_tiled(r, b, kb)) = v;
        b += 128;
    }
}

// ---- MXFP8 (layers 20-23): block 32, ue8m0 power-of-two scale, no outer scale -------------------------------------------

/// Byte offset of MXFP8 scale (row, 32-block b) in the tiled layout: row tile rt (128 rows) x k-tile kt (2 blocks = 64 values)
/// = one 256-byte block [128 rows][2 scales]. `kb` = K / 32.
#[inline(always)]
pub(crate) fn sf8_tiled(row: u32, b: u32, kb: u32) -> usize {
    (((row >> 7) * (kb >> 1) + (b >> 1)) as usize) * 256 + ((row & 127) * 2 + (b & 1)) as usize
}

/// The native MXFP8 rule (`mxfp8_half_block`): returns (ue8m0 exponent, 32 e4m3 bytes as 8 words).
#[inline(always)]
pub(crate) fn mxfp8_block(x: [f32; 32]) -> (u32, [u32; 8]) {
    let mut amax = 0.0f32;
    kcommon::unroll!(i in 32 => {
        amax = amax.max(x[i].abs());
    });
    let min_scale = f32::from_bits(0x0040_0000);
    let raw = amax * (1.0f32 / 448.0f32);
    let mut exponent = ((raw.to_bits().wrapping_add(0x007F_FFFF)) & 0x7F80_0000) >> 23;
    exponent = if raw <= min_scale { 0 } else if exponent > 254 { 254 } else { exponent };
    if raw != raw {
        exponent = 255;
    }
    let scale = f32::from_bits(exponent << 23).max(min_scale);
    let mut inverse = if exponent == 0 { f32::from_bits(0x7F00_0000) } else { math::rcp(scale) };
    if exponent == 255 {
        inverse = 0.0;
    }
    let mut w = [0u32; 8];
    kcommon::unroll!(i in 16 => {
        let lo = (x[2 * i] * inverse).max(-448.0).min(448.0);
        let hi = (x[2 * i + 1] * inverse).max(-448.0).min(448.0);
        let pair: u16;
        unsafe {
            asm!("cvt.rn.satfinite.e4m3x2.f32 {p}, {hi}, {lo};", p = out(reg16) pair, hi = in(reg32) hi, lo = in(reg32) lo,
                options(pure, nomem, nostack));
        }
        w[i / 2] |= (pair as u32) << (16 * (i % 2));
    });
    (exponent, w)
}

/// bf16 rows -> MXFP8 values [pos][K] (e4m3 bytes) and tiled ue8m0 scales (kb = K / 32). Same pair / position convention
/// as `ink_quant_nvfp4`. One lane a 32-value block, one warp a pair: grid = ceil(npairs / 8), block = 256. K % 64 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_quant_mxfp8(vals: *mut u8, sf: *mut u8, x: *const u16, ldx: u32,
    pos_of_pair: *const u32, gather: u32, npairs: u32, k: u32) {
    pdl_wait();
    pdl_trigger();
    let pr = bid_x() * 8 + warp_id();
    if pr >= npairs {
        return;
    }
    let d = *pos_of_pair.add(pr as usize);
    if d == NONE {
        return;
    }
    let src = if gather != 0 { pr / 8 } else { d };
    let xr = x.add(src as usize * ldx as usize);
    let kb = k >> 5;
    let mut b = lane();
    while b < kb {
        let p = xr.add(b as usize * 32) as *const u32;
        let mut v = [0.0f32; 32];
        kcommon::unroll!(q in 4 => {
            let f = bf16::unpack8(ldg_nc_v4(p.add(q * 4)));
            kcommon::unroll!(j in 8 => {
                v[q * 8 + j] = f[j];
            });
        });
        let (e, w) = mxfp8_block(v);
        let o = vals.add(d as usize * k as usize + b as usize * 32) as *mut [u32; 4];
        *o = [w[0], w[1], w[2], w[3]];
        *o.add(1) = [w[4], w[5], w[6], w[7]];
        *sf.add(sf8_tiled(d, b, kb)) = e as u8;
        b += 32;
    }
}

/// Load-time repack of MXFP8 w13: rows interleaved in blocks of 8 (released rows 2f = gate f, 2f + 1 = up f), scales tiled.
/// src: [128][2304][2304] values, [128][2304][72] scales. grid = (2304, 128), block = 128.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_repack_w13_mxfp8(dvals: *mut u8, dsf: *mut u8, svals: *const u8, ssf: *const u8) {
    let r = bid_x();
    let e = bid_y() as usize;
    let q = r / 16;
    let j = r % 16;
    let srow = if j < 8 { 2 * (8 * q + j) } else { 2 * (8 * q + j - 8) + 1 };
    let t = tid_x();
    let s = svals.add((e * 2304 + srow as usize) * 2304);
    let mut i = t;
    while i < 144 {
        *(dvals.add(e * 2304 * 2304 + tiled_code_off(r, i, 2304)) as *mut [u32; 4]) = *(s.add(i as usize * 16) as *const [u32; 4]);
        i += 128;
    }
    if t < 72 {
        let v = *ssf.add((e * 2304 + srow as usize) * 72 + t as usize);
        *dsf.add(e * 165888 + sf8_tiled(r, t, 72)) = v;
    }
}

/// Load-time retile of MXFP8 scales only (w2). src [128][rows][kb] -> tiled per expert. grid = (rows, 128), block = 128.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_retile_sf8(dsf: *mut u8, ssf: *const u8, rows: u32, kb: u32) {
    let r = bid_x();
    let e = bid_y() as usize;
    let mut b = tid_x();
    while b < kb {
        let v = *ssf.add((e * rows as usize + r as usize) * kb as usize + b as usize);
        *dsf.add(e * rows as usize * kb as usize + sf8_tiled(r, b, kb)) = v;
        b += 128;
    }
}
