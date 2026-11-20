//! RMSNorm (DeepSeek: plain weight), RoPE from host tables, and the cache writers: window KV (RoPE + FP8 grid,
//! ring store), compressed entries (RoPE + FP4 / E4M3 per 16, plane store) and indexer keys / queries (RoPE + FP4
//! / power-of-two per 32).
use crate::q;
use kcommon::smem::Shared;
use kcommon::*;

/// y[r] = bf16(w * (x * rsqrt(mean(x^2) + eps))), f32 math. x bf16 (`x_f32` = 0) or f32 (1; DeepSeek rounds the
/// linear output to bf16 first, so f32 inputs are rounded to bf16 before the norm unless `x_f32` = 2).
/// Row strides ldx / ldy in elements. grid = rows, block = 256 (or 128 for small dims); dim <= 8192.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_rmsnorm(y: *mut u16, x: *const u8, x_f32: u32, w: *const u16, dim: u32, eps: f32,
    ldx: u32, ldy: u32) {
    let red: Shared<f32> = kcommon::static_shared!(rn_red, f32, 128);
    let buf: Shared<f32> = kcommon::static_shared!(rn_buf, f32, 32768);
    let r = bid_x() as usize;
    let d = dim as usize;
    let mut ss = 0.0f32;
    let mut i = tid_x() as usize;
    while i < d {
        let v = if x_f32 == 0 {
            bf16::to_f32(*(x as *const u16).add(r * ldx as usize + i))
        } else {
            let f = *(x as *const f32).add(r * ldx as usize + i);
            if x_f32 == 1 { bf16::round(f) } else { f }
        };
        buf.set(i as u32, v);
        ss = math::fma(v, v, ss);
        i += bdim_x() as usize;
    }
    let tot = warp::block_sum(ss, red);
    let inv = math::rsqrt(tot / dim as f32 + eps);
    let mut i = tid_x() as usize;
    while i < d {
        *y.add(r * ldy as usize + i) = bf16::from_f32(bf16::to_f32(*w.add(i)) * (buf.at(i as u32) * inv));
        i += bdim_x() as usize;
    }
}

/// Rotate the 32 adjacent pairs of `x[off .. off + 64]` (bf16) by the table row of `pos`: tab[pos][i] = (cos, sin)
/// f32; inverse rotates by the conjugate. Result rounded to bf16.
#[inline(always)]
unsafe fn rope64(x: *mut u16, tab: *const f32, pos: u32, i: usize, inverse: bool) {
    let t = tab.add(pos as usize * 64 + 2 * i);
    let (c, s) = (*t, *t.add(1));
    let s = if inverse { -s } else { s };
    let p = x as *mut u32;
    let w = *p.add(i);
    let (a, b) = bf16::unpack2(w);
    *p.add(i) = bf16::pack2(a * c - b * s, a * s + b * c);
}

/// RoPE on `heads` heads of each row: head h of row r starts at x + r*ld + h*hstride; the rotated part is
/// [off, off + 64). pos = meta[r].1. grid = (rows, heads), block = 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_rope(x: *mut u16, ld: u32, hstride: u32, off: u32, meta: *const u32,
    tab: *const f32, inverse: u32) {
    let r = bid_x() as usize;
    let h = bid_y() as usize;
    let pos = *meta.add(2 * r + 1);
    let base = x.add(r * ld as usize + h * hstride as usize + off as usize);
    rope64(base, tab, pos, tid_x() as usize, inverse != 0);
}

/// Window KV: kv[r] (bf16 [512], normed) gets RoPE on its last 64 at pos, then the FP8 E4M3 grid with a
/// power-of-two scale per 32 (amax floored at 1e-4), and is written back and to ring[slot][pos % ring_len].
/// grid = rows, block = 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_kv_window(kv: *mut u16, ring: *mut u16, slot_stride: u32, ring_len: u32,
    meta: *const u32, tab: *const f32, dim: u32) {
    kv_window_row(kv, ring, slot_stride, ring_len, meta, tab, dim, bid_x() as usize);
}

#[inline(always)]
unsafe fn kv_window_row(kv: *mut u16, ring: *mut u16, slot_stride: u32, ring_len: u32, meta: *const u32,
    tab: *const f32, dim: u32, r: usize) {
    let slot = *meta.add(2 * r) as usize;
    let pos = *meta.add(2 * r + 1);
    let row = kv.add(r * dim as usize);
    let l = lane() as usize;
    rope64(row.add(dim as usize - 64), tab, pos, l, false);
    sync_warp();
    // FP8 blocks of 32: lane handles elements l, l+32, ... and the warp reduces each block's amax
    let nb = dim as usize / 32;
    let dst = ring.add(slot * slot_stride as usize + (pos % ring_len) as usize * dim as usize);
    if nb == 16 {
        // all 16 blocks' loads first (dim 512), then the same per-block math
        let mut vs = [0.0f32; 16];
        let mut b = 0;
        while b < 16 {
            vs[b] = bf16::to_f32(*row.add(b * 32 + l));
            b += 1;
        }
        let mut b = 0;
        while b < 16 {
            let s = q::fp8_scale(warp::max(vs[b].abs()));
            let o = bf16::from_f32(q::fp8_qd(vs[b], s));
            *row.add(b * 32 + l) = o;
            *dst.add(b * 32 + l) = o;
            b += 1;
        }
        return;
    }
    let mut b = 0;
    while b < nb {
        let v = bf16::to_f32(*row.add(b * 32 + l));
        let s = q::fp8_scale(warp::max(v.abs()));
        let o = bf16::from_f32(q::fp8_qd(v, s));
        *row.add(b * 32 + l) = o;
        *dst.add(b * 32 + l) = o;
        b += 1;
    }
}

/// Compressed entry store: for rows that close a group ((pos + 1) % ratio == 0, entry e = pos / ratio), copy the
/// latent (bf16 [dim]), RoPE its last 64 at position e * ratio, FP4 E2M1 with an E4M3 scale per 16, into
/// plane[slot][e]. grid = rows, block = 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_comp_store(plane: *mut u16, slot_stride: u32, latent: *const u16,
    meta: *const u32, ratio: u32, tab: *const f32, dim: u32) {
    let r = bid_x() as usize;
    let slot = *meta.add(2 * r) as usize;
    let pos = *meta.add(2 * r + 1);
    if (pos + 1) % ratio != 0 {
        return;
    }
    let e = (pos / ratio) as usize;
    let d = dim as usize;
    let dst = plane.add(slot * slot_stride as usize + e * d);
    let l = lane() as usize;
    let mut i = l;
    while i < d {
        *dst.add(i) = *latent.add(r * d + i);
        i += 32;
    }
    sync_warp();
    rope64(dst.add(d - 64), tab, e as u32 * ratio, l, false);
    sync_warp();
    quant_blocks(dst, d, 16, true);
}

/// Max over each half-warp (lanes 0-15, 16-31).
#[inline(always)]
fn max16(mut v: f32) -> f32 {
    let mut m = 8;
    while m > 0 {
        v = v.max(warp::shfl_xor(v, m));
        m >>= 1;
    }
    v
}

/// FP4 quant-dequant of `n` bf16 values in blocks of `blk` (16 or 32) by one warp, in place.
#[inline(always)]
unsafe fn quant_blocks(p: *mut u16, n: usize, blk: usize, e4m3_scale: bool) {
    let l = lane() as usize;
    // each lane owns elements l, l + 32, ...; a block of 16 spans half a warp, of 32 the whole warp
    let mut i = 0;
    while i < n {
        let v = bf16::to_f32(*p.add(i + l));
        let a = v.abs();
        let amax = if blk == 16 { max16(a) } else { warp::max(a) };
        let s = q::fp4_scale(amax, e4m3_scale);
        *p.add(i + l) = bf16::from_f32(q::fp4_qd(v, s));
        i += 32;
    }
}

/// Indexer key store: k[r] (bf16 [128], normed) of rows closing a group gets RoPE on its last 64 at e * ratio
/// and FP4 with a power-of-two scale per 32, into ikeys[slot][e]. grid = rows, block = 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_ikey_store(ikeys: *mut u16, slot_stride: u32, k: *const u16, meta: *const u32,
    ratio: u32, tab: *const f32) {
    let r = bid_x() as usize;
    let slot = *meta.add(2 * r) as usize;
    let pos = *meta.add(2 * r + 1);
    if (pos + 1) % ratio != 0 {
        return;
    }
    let e = (pos / ratio) as usize;
    let dst = ikeys.add(slot * slot_stride as usize + e * 128);
    let l = lane() as usize;
    let mut i = l;
    while i < 128 {
        *dst.add(i) = *k.add(r * 128 + i);
        i += 32;
    }
    sync_warp();
    rope64(dst.add(64), tab, e as u32 * ratio, l, false);
    sync_warp();
    quant_blocks(dst, 128, 32, false);
}

/// Indexer queries: q[r][h] (bf16 [heads][128]) gets RoPE on its last 64 at pos, FP4 / power-of-two per 32,
/// out as f32 [rows][heads][128]. grid = (rows, heads), block = 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_idx_query(out: *mut f32, qb: *mut u16, meta: *const u32, tab: *const f32,
    heads: u32) {
    let r = bid_x() as usize;
    let h = bid_y() as usize;
    let pos = *meta.add(2 * r + 1);
    let row = qb.add((r * heads as usize + h) * 128);
    let l = lane() as usize;
    rope64(row.add(64), tab, pos, l, false);
    sync_warp();
    quant_blocks(row, 128, 32, false);
    sync_warp();
    let o = out.add((r * heads as usize + h) * 128);
    let mut i = l;
    while i < 128 {
        *o.add(i) = bf16::to_f32(*row.add(i));
        i += 32;
    }
}

/// ds_rmsnorm (dim 512, as its 128-thread launch: thread t takes t, t + 128, ..., the block sum over 4 warps) and then
/// ds_kv_window on the result, on one warp: lane l plays threads l, l + 32, l + 64, l + 96, so the bits are the two
/// launches'. kv = the normed, roped, FP8-gridded row (also written to the ring). grid = rows, block = 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_kv_norm_window(kv: *mut u16, kv_raw: *const u16, w: *const u16, eps: f32,
    ring: *mut u16, slot_stride: u32, ring_len: u32, meta: *const u32, tab: *const f32) {
    const D: usize = 512;
    let r = bid_x() as usize;
    let l = lane() as usize;
    let x = kv_raw.add(r * D);
    let mut v = [[0.0f32; 4]; 4]; // [virtual warp][j]
    // all 16 loads first, then the sums (the same values and order)
    let mut vw = 0;
    while vw < 4 {
        let mut j = 0;
        while j < 4 {
            v[vw][j] = bf16::to_f32(*x.add(l + 32 * vw + 128 * j));
            j += 1;
        }
        vw += 1;
    }
    let mut s = [0.0f32; 4];
    let mut vw = 0;
    while vw < 4 {
        let mut ss = 0.0f32;
        let mut j = 0;
        while j < 4 {
            ss = math::fma(v[vw][j], v[vw][j], ss);
            j += 1;
        }
        s[vw] = warp::sum(ss);
        vw += 1;
    }
    let t2 = if l == 0 { s[0] } else if l == 1 { s[1] } else if l == 2 { s[2] } else if l == 3 { s[3] } else { 0.0 };
    let tot = warp::sum(t2);
    let inv = math::rsqrt(tot / D as f32 + eps);
    let row = kv.add(r * D);
    let mut vw = 0;
    while vw < 4 {
        let mut j = 0;
        while j < 4 {
            let i = l + 32 * vw + 128 * j;
            *row.add(i) = bf16::from_f32(bf16::to_f32(*w.add(i)) * (v[vw][j] * inv));
            j += 1;
        }
        vw += 1;
    }
    sync_warp();
    kv_window_row(kv, ring, slot_stride, ring_len, meta, tab, D as u32, r);
}
