//! Prompt MoE: sort the (token, expert) pairs by expert on the GPU, stably (pair order inside an expert), for the
//! segmented int8 GEMM. Three launches, deterministic:
//! 1. `ggq_moe_sort_count`: per chunk of 4096 pairs, a histogram over experts.
//! 2. `ggq_moe_sort_scan` (one block): `seg_off[e]` (exclusive prefix of the totals, `seg_off[nexp]` = np), each
//!    chunk's start per expert, and the TILE-row tile list `tiles[2i] = e, tiles[2i+1] = first row`, `ntiles[0]`.
//! 3. `ggq_moe_sort_scatter`: one warp per chunk walks its pairs in order (`match.any` ranks equal experts):
//!    sorted row r gets `xidx[r]` = token (pair / topk) and `yidx[r]` = pair.
use kcommon::smem::Shared;
use kcommon::*;

pub const CHUNK: u32 = 4096;
pub const MAXE: u32 = 512;
/// Rows per tile of the tile list (= the int8 GEMM's BM).
pub const TILE: u32 = 128;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_moe_sort_count(counts: *mut u32, ids: *const i32, np: u32, nexp: u32) {
    let h: Shared<u32> = static_shared!(ggq_sort_hist, u32, 512 * 4);
    let c = bid_x();
    let mut e = tid_x();
    while e < nexp {
        h.add(e).st(0);
        e += bdim_x();
    }
    sync_threads();
    let mut q = c * CHUNK + tid_x();
    let end = ((c + 1) * CHUNK).min(np);
    while q < end {
        let ex = *ids.add(q as usize) as u32;
        core::arch::asm!("atom.shared.add.u32 _, [{a}], 1;", a = in(reg32) h.add(ex).addr(), options(nostack));
        q += bdim_x();
    }
    sync_threads();
    let mut e = tid_x();
    while e < nexp {
        *counts.add((c * nexp + e) as usize) = h.add(e).ld();
        e += bdim_x();
    }
}

/// One block of 512 threads. `chunk_off[c][e]` = first sorted row of chunk c's pairs of expert e.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_moe_sort_scan(counts: *const u32, nchunk: u32, nexp: u32, chunk_off: *mut u32,
    seg_off: *mut u32, tiles: *mut u32, ntiles: *mut u32) {
    let tot: Shared<u32> = static_shared!(ggq_sort_tot, u32, 512 * 4);
    let ntl: Shared<u32> = static_shared!(ggq_sort_ntl, u32, 512 * 4);
    let e = tid_x();
    let mut t = 0u32;
    if e < nexp {
        for c in 0..nchunk {
            t += *counts.add((c * nexp + e) as usize);
        }
    }
    tot.add(e).st(t);
    ntl.add(e).st(t.div_ceil(TILE));
    sync_threads();
    if e == 0 {
        // serial exclusive scans (512 entries)
        let mut a = 0u32;
        let mut b = 0u32;
        for i in 0..nexp {
            let x = tot.add(i).ld();
            tot.add(i).st(a);
            a += x;
            let y = ntl.add(i).ld();
            ntl.add(i).st(b);
            b += y;
        }
        *seg_off.add(nexp as usize) = a;
        *ntiles = b;
    }
    sync_threads();
    if e < nexp {
        let s0 = tot.add(e).ld();
        *seg_off.add(e as usize) = s0;
        let mut o = s0;
        for c in 0..nchunk {
            *chunk_off.add((c * nexp + e) as usize) = o;
            o += *counts.add((c * nexp + e) as usize);
        }
        // tiles of expert e
        let mut ti = ntl.add(e).ld();
        let mut r = s0;
        while r < o {
            *tiles.add(2 * ti as usize) = e;
            *tiles.add(2 * ti as usize + 1) = r;
            ti += 1;
            r += TILE;
        }
    }
}

/// Grid: nchunk blocks of 32 threads.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_moe_sort_scatter(ids: *const i32, np: u32, nexp: u32, topk: u32,
    chunk_off: *const u32, xidx: *mut u32, yidx: *mut u32) {
    let cur: Shared<u32> = static_shared!(ggq_sort_cur, u32, 512 * 4);
    let c = bid_x();
    let mut e = lane();
    while e < nexp {
        cur.add(e).st(*chunk_off.add((c * nexp + e) as usize));
        e += 32;
    }
    sync_warp();
    let mut base = c * CHUNK;
    let end = ((c + 1) * CHUNK).min(np);
    while base < end {
        let q = base + lane();
        let ok = q < end;
        let ex: u32 = if ok { *ids.add(q as usize) as u32 } else { 0xffff_ffff };
        let mask: u32;
        core::arch::asm!("match.any.sync.b32 {m}, {v}, 0xffffffff;", m = out(reg32) mask, v = in(reg32) ex, options(nostack));
        let lt = (1u32 << lane()) - 1;
        let rank = (mask & lt).count_ones();
        let pos = if ok { cur.add(ex).ld() + rank } else { 0 };
        sync_warp();
        if ok {
            *xidx.add(pos as usize) = q / topk;
            *yidx.add(pos as usize) = q;
            if rank == 0 {
                cur.add(ex).st(pos + mask.count_ones());
            }
        }
        sync_warp();
        base += 32;
    }
}
