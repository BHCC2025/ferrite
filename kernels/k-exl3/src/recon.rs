//! Whole-tensor reconstruct to the original basis, run once at model load:
//!   1. exl3_recon_hat_f32: trellis -> W_hat (f32 scratch [k][ns], rotated basis)
//!   2. exl3_had_rows:      H over n (each row's 128-column blocks), * svh / sqrt(128), in place
//!   3. exl3_had_cols_bf16: H over k (128-row blocks), * suh / sqrt(128), -> bf16 in [k][n] or [n][k]
//! Bits are a runtime argument: this is load-time code, simplicity over speed.

use crate::util::*;
use kcommon::{bid_x, bid_y, static_shared, sync_threads, tid_x};

/// The 16-bit state of position p (0..256) of a tile with `bits` bits a value.
#[inline(always)]
unsafe fn tile_state(tile: *const u32, bits: u32, p: u32) -> u32 {
    let words = 8 * bits;
    let b1 = (p + 1) * bits + 256 * bits; // one past the window's last bit (+ one tile so b1 - 16 >= 0)
    let b0 = b1 - 16;
    let i0 = b0 / 32;
    let i1 = (b1 - 1) / 32;
    let s = (i1 + 1) * 32 - b1;
    let a = *tile.add((i0 % words) as usize);
    let b = *tile.add((i1 % words) as usize);
    funnel_r(b, a, s) & 0xffff
}

/// grid (ns / 128, k / 16), 256 threads: warp w decodes tile (ctaid.y, nt0 + 8 ctaid.x + w).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn exl3_recon_hat_f32(
    out: *mut f32,
    trellis: *const u32,
    n_tiles: u32,
    nt0: u32,
    ns: u32,
    bits: u32,
) {
    let warp = tid_x() / 32;
    let lane = tid_x() % 32;
    let kt = bid_y();
    let ntl = bid_x() * 8 + warp;
    let tile = trellis.add(((kt * n_tiles + nt0 + ntl) * 8 * bits) as usize);
    let mut j = 0;
    while j < 8 {
        let s0 = tile_state(tile, bits, lane * 8 + j);
        let s1 = tile_state(tile, bits, lane * 8 + j + 1);
        let (v0, v1) = h2_to_f32(decode2_mul1(s0, s1));
        for (jj, v) in [(j, v0), (j + 1, v1)] {
            let row = 2 * (lane & 3) + (jj & 1) + 8 * ((jj >> 1) & 1);
            let col = (lane >> 2) + 8 * (jj >> 2);
            *out.add(((kt * 16 + row) * ns + ntl * 16 + col) as usize) = v;
        }
        j += 2;
    }
}

/// grid (ns / 128, k), 32 threads: row ctaid.y, columns 128 ctaid.x.., H over n, * svh[n0 + col] / sqrt(128).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn exl3_had_rows(buf: *mut f32, ns: u32, svh: *const u16, n0: u32) {
    let lane = tid_x();
    let c = bid_x() * 128 + 4 * lane;
    let p = buf.add((bid_y() * ns + c) as usize);
    let mut v = [*p, *p.add(1), *p.add(2), *p.add(3)];
    fwht128(&mut v, lane);
    for q in 0..4 {
        *p.add(q) = v[q] * HAD_SCALE * f16_to_f32(*svh.add((n0 + c) as usize + q));
    }
}

/// grid (ns / 32, k / 128), 128 threads: rows 128 ctaid.y.., columns 32 ctaid.x.. of the scratch; H over k,
/// * suh / sqrt(128), bf16 out. nk != 0: out is [n][k] (HF weight layout, ld = k), else [k][n] (ld = n).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn exl3_had_cols_bf16(
    buf: *const f32,
    out: *mut u16,
    ns: u32,
    suh: *const u16,
    n0: u32,
    ld: u32,
    nk: u32,
) {
    const PITCH: u32 = 33; // floats a row of the 128 x 32 tile (bank spread)
    let sm = static_shared!(sm_cols, f32, 128 * 33 * 4);
    let warp = tid_x() / 32;
    let lane = tid_x() % 32;
    let r0 = bid_y() * 128;
    let c0 = bid_x() * 32;
    let mut i = 0;
    while i < 32 {
        let r = warp * 32 + i;
        sm.set(r * PITCH + lane, *buf.add(((r0 + r) * ns + c0 + lane) as usize));
        i += 1;
    }
    sync_threads();
    let mut cc = 0;
    while cc < 8 {
        let c = warp * 8 + cc;
        let mut v = [0f32; 4];
        for q in 0..4 {
            v[q] = sm.at((4 * lane + q as u32) * PITCH + c);
        }
        fwht128(&mut v, lane);
        for q in 0..4 {
            let r = 4 * lane + q as u32;
            sm.set(r * PITCH + c, v[q] * HAD_SCALE * f16_to_f32(*suh.add((r0 + r) as usize)));
        }
        cc += 1;
    }
    sync_threads();
    if nk != 0 {
        let mut cc = 0;
        while cc < 8 {
            let c = warp * 8 + cc;
            let mut v = [0f32; 4];
            for q in 0..4 {
                v[q] = sm.at((4 * lane + q as u32) * PITCH + c);
            }
            let p = out.add(((n0 + c0 + c) * ld + r0 + 4 * lane) as usize) as *mut u32;
            *p = f32x2_to_bf2(v[0], v[1]);
            *p.add(1) = f32x2_to_bf2(v[2], v[3]);
            cc += 1;
        }
    } else {
        let mut i = 0;
        while i < 32 {
            let r = warp * 32 + i;
            let v = sm.at(r * PITCH + lane);
            *out.add(((r0 + r) * ld + n0 + c0 + lane) as usize) = f32x2_to_bf2(v, 0.0) as u16;
            i += 1;
        }
    }
}

/// Rotated-basis W_hat straight to fp16 [k][ns] (exllamav3's `reconstruct`), same grid as exl3_recon_hat_f32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn exl3_recon_hat_f16(
    out: *mut u16,
    trellis: *const u32,
    n_tiles: u32,
    nt0: u32,
    ns: u32,
    bits: u32,
) {
    let warp = tid_x() / 32;
    let lane = tid_x() % 32;
    let kt = bid_y();
    let ntl = bid_x() * 8 + warp;
    let tile = trellis.add(((kt * n_tiles + nt0 + ntl) * 8 * bits) as usize);
    let mut j = 0;
    while j < 8 {
        let s0 = tile_state(tile, bits, lane * 8 + j);
        let s1 = tile_state(tile, bits, lane * 8 + j + 1);
        let h = decode2_mul1(s0, s1);
        for (jj, v) in [(j, h as u16), (j + 1, (h >> 16) as u16)] {
            let row = 2 * (lane & 3) + (jj & 1) + 8 * ((jj >> 1) & 1);
            let col = (lane >> 2) + 8 * (jj >> 2);
            *out.add(((kt * 16 + row) * ns + ntl * 16 + col) as usize) = v;
        }
        j += 2;
    }
}
