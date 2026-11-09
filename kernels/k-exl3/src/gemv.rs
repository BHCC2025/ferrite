//! Decode GEMV straight from the trellis: y[m][n] = x[m][k] @ W for 1..16+ rows (16-row passes).
//!
//! Two launches per linear:
//!   exl3_rot_in:  xh = fp16(H(x * suh) / sqrt(128))       (x bf16 or f32 [m][k], xh fp16 [m][k])
//!   exl3_gemv_bB_wW: one program per (128-column block, K split), W warps over contiguous K ranges.
//!     Each lane decodes its 8 values of each 16x16 tile straight into mma B fragments (the trellis is stored
//!     in fragment order), x rows are the A fragment, fp32 accumulators. The warps' sums meet in shared memory;
//!     with one split the program finishes the block (H over its 128 columns, * svh / sqrt(128), store);
//!     with several, splits write fp32 partials and the last to arrive (atomic counter) adds them in split
//!     order and finishes, then re-arms the counter (graph-safe, counters zeroed once at allocation).
//! Port of the TensorFold EXL3 linear (zig/kernels/cuda/exl3_linear.cu, exl3_linear_wc.cuh, exl3_decode.cuh;
//! Apache-2.0, itself after exllamav3, MIT), rewritten in Rust; see NOTICE.

use core::arch::asm;

use crate::util::*;
use kcommon::mma::mma_f16_16816;
use kcommon::smem::{dyn_base, Shared};
use kcommon::{bid_x, bid_y, lane, static_shared, sync_threads, tid_x, warp_id};

/// Full unrolling (Rust has no `#pragma unroll`): the body is pasted once per index, so arrays indexed by
/// it stay in registers.
macro_rules! unroll_pub {
    (2, $i:ident, $b:block) => {{ { let $i = 0usize; $b } { let $i = 1usize; $b } }};
    (3, $i:ident, $b:block) => {{ { let $i = 0usize; $b } { let $i = 1usize; $b } { let $i = 2usize; $b } }};
    (4, $i:ident, $b:block) => {{ { let $i = 0usize; $b } { let $i = 1usize; $b } { let $i = 2usize; $b } { let $i = 3usize; $b } }};
    (8, $i:ident, $b:block) => {{
        { let $i = 0usize; $b } { let $i = 1usize; $b } { let $i = 2usize; $b } { let $i = 3usize; $b }
        { let $i = 4usize; $b } { let $i = 5usize; $b } { let $i = 6usize; $b } { let $i = 7usize; $b }
    }};
}

pub(crate) use unroll_pub;
pub(crate) use unroll_pub as unroll;

#[inline(always)]
fn min_u(a: u32, b: u32) -> u32 {
    if a < b { a } else { b }
}

/// Programmatic dependent launch: wait until the previous grid has finished and its writes are visible
/// (a no-op when the launch is not programmatic).
#[inline(always)]
fn pdl_wait() {
    unsafe { asm!("griddepcontrol.wait;", options(nostack)) };
}

/// Let the next (programmatic) launch start once every program of this grid has passed here.
#[inline(always)]
fn pdl_trigger() {
    unsafe { asm!("griddepcontrol.launch_dependents;", options(nostack)) };
}

/// Pull this warp's first `steps` k steps (8 tiles each) into L2 (no registers), before `pdl_wait`.
#[inline(always)]
unsafe fn prefetch_steps<const BITS: u32>(tile: *const u32, stride_k: u32, steps: u32, ln: u32) {
    let lines = 2 * BITS; // 128-byte lines a step (256 * BITS bytes)
    let mut l = ln;
    while l < steps * lines {
        let p = (tile as *const u8).add(((l / lines) * stride_k * 4 + (l % lines) * 128) as usize);
        asm!("prefetch.global.L2 [{0}];", in(reg64) p, options(nostack, readonly));
        l += 32;
    }
}

#[inline(always)]
fn fence_gpu() {
    unsafe { asm!("fence.acq_rel.gpu;", options(nostack)) };
}

#[inline(always)]
unsafe fn atomic_add_u32(p: *mut u32, v: u32) -> u32 {
    let r: u32;
    asm!("atom.add.acq_rel.gpu.u32 {0}, [{1}], {2};", out(reg32) r, in(reg64) p, in(reg32) v, options(nostack));
    r
}

/// L2-only (bypass L1) vector load of split partials written by other programs.
#[inline(always)]
unsafe fn ldcg_v4(p: *const f32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    asm!("ld.global.cg.v4.f32 {{{0}, {1}, {2}, {3}}}, [{4}];", out(reg32) a, out(reg32) b, out(reg32) c,
         out(reg32) d, in(reg64) p, options(nostack, readonly));
    [a, b, c, d]
}

#[inline(always)]
unsafe fn stg_v4f(p: *mut f32, v: [f32; 4]) {
    asm!("st.global.v4.f32 [{0}], {{{1}, {2}, {3}, {4}}};", in(reg64) p, in(reg32) v[0], in(reg32) v[1],
         in(reg32) v[2], in(reg32) v[3], options(nostack));
}

#[inline(always)]
unsafe fn stg_v2u(p: *mut u32, a: u32, b: u32) {
    asm!("st.global.v2.u32 [{0}], {{{1}, {2}}};", in(reg64) p, in(reg32) a, in(reg32) b, options(nostack));
}

#[inline(always)]
unsafe fn ldg_v2u(p: *const u32) -> (u32, u32) {
    let (a, b): (u32, u32);
    asm!("ld.global.nc.v2.u32 {{{0}, {1}}}, [{2}];", out(reg32) a, out(reg32) b, in(reg64) p, options(nostack, readonly, pure));
    (a, b)
}

/// Load 4 consecutive activations as f32: bf16 (x_f32 == 0) or f32.
#[inline(always)]
unsafe fn load4(x: *const u8, x_f32: u32, i: usize) -> [f32; 4] {
    if x_f32 != 0 {
        let p = (x as *const f32).add(i);
        [*p, *p.add(1), *p.add(2), *p.add(3)]
    } else {
        let (a, b) = ldg_v2u((x as *const u16).add(i) as *const u32);
        [
            f32::from_bits(a << 16),
            f32::from_bits(a & 0xffff_0000),
            f32::from_bits(b << 16),
            f32::from_bits(b & 0xffff_0000),
        ]
    }
}

/// Store 4 consecutive outputs. Mode (the kernels' `y_f32` argument): 0 bf16, 1 f32, 2 bf16 y += v
/// (residual add: bf16(f32(y) + v)), 3 f32 y += v.
#[inline(always)]
unsafe fn store4(y: *mut u8, mode: u32, i: usize, v: [f32; 4]) {
    if mode & 1 != 0 {
        let p = (y as *mut f32).add(i);
        let v = if mode & 2 != 0 { [v[0] + *p, v[1] + *p.add(1), v[2] + *p.add(2), v[3] + *p.add(3)] } else { v };
        stg_v4f(p, v);
    } else {
        let p = (y as *mut u16).add(i) as *mut u32;
        let v = if mode & 2 != 0 {
            let (a, b) = (*p, *p.add(1));
            [
                v[0] + f32::from_bits(a << 16),
                v[1] + f32::from_bits(a & 0xffff_0000),
                v[2] + f32::from_bits(b << 16),
                v[3] + f32::from_bits(b & 0xffff_0000),
            ]
        } else {
            v
        };
        stg_v2u(p, f32x2_to_bf2(v[0], v[1]), f32x2_to_bf2(v[2], v[3]));
    }
}

/// xh[row][k] = fp16(H(x * suh) / sqrt(128)); grid (k / 512, m), 128 threads (one warp a 128-column block).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn exl3_rot_in(x: *const u8, x_f32: u32, suh: *const u16, xh: *mut u16, k: u32) {
    pdl_trigger(); // the GEMV may launch now and prefetch its weights
    pdl_wait();
    let blk = bid_x() * 4 + warp_id();
    if blk * 128 >= k {
        return;
    }
    let ln = lane();
    let row = bid_y();
    let c = blk * 128 + 4 * ln;
    let mut v = load4(x, x_f32, (row * k + c) as usize);
    let (s0, s1) = ldg_v2u(suh.add(c as usize) as *const u32);
    let (a, b) = h2_to_f32(s0);
    let (cc, d) = h2_to_f32(s1);
    v[0] *= a;
    v[1] *= b;
    v[2] *= cc;
    v[3] *= d;
    fwht128(&mut v, ln);
    let p = xh.add((row * k + c) as usize) as *mut u32;
    stg_v2u(p, f32x2_to_h2(v[0] * HAD_SCALE, v[1] * HAD_SCALE), f32x2_to_h2(v[2] * HAD_SCALE, v[3] * HAD_SCALE));
}

/// The output H over a row's 128 columns (4 a lane), * svh / sqrt(128), stored.
#[inline(always)]
unsafe fn finish(mut v: [f32; 4], ln: u32, svh: *const u16, col: u32, y: *mut u8, y_f32: u32, at: usize) {
    fwht128(&mut v, ln);
    let (s0, s1) = ldg_v2u(svh.add(col as usize) as *const u32);
    let (a, b) = h2_to_f32(s0);
    let (c, d) = h2_to_f32(s1);
    store4(y, y_f32, at, [v[0] * HAD_SCALE * a, v[1] * HAD_SCALE * b, v[2] * HAD_SCALE * c, v[3] * HAD_SCALE * d]);
}

/// 16-bit window starting `d` bits into the 96-bit big-endian stream hi:mid:lo (d constant after unrolling).
#[inline(always)]
fn window16(hi: u32, mid: u32, lo: u32, d: u32) -> u32 {
    if d <= 16 {
        (hi >> (16 - d)) & 0xffff
    } else if d < 32 {
        funnel_l(mid, hi, d) >> 16
    } else if d <= 48 {
        (mid >> (48 - d)) & 0xffff
    } else {
        funnel_l(lo, mid, d - 32) >> 16
    }
}

/// This lane's two B fragments of one tile (columns 0-7 in b0, 8-15 in b1) from its LW stream words.
#[inline(always)]
fn decode_lane<const BITS: u32, const LW: usize>(w: &[u32; 3], off: u32) -> ([u32; 2], [u32; 2]) {
    let w2 = if LW > 2 { w[2] } else { 0 };
    let hi = funnel_l(w[1], w[0], off);
    let mid = funnel_l(w2, w[1], off);
    let lo = w2 << off;
    let s = |j: u32| window16(hi, mid, lo, j * BITS);
    (
        [decode2_mul1(s(0), s(1)), decode2_mul1(s(2), s(3))],
        [decode2_mul1(s(4), s(5)), decode2_mul1(s(6), s(7))],
    )
}

/// One k step's words for this lane: the step's 8 tiles back to back, LW words of each.
#[inline(always)]
unsafe fn load_step<const BITS: u32, const LW: usize>(step: *const u32, word: u32) -> [[u32; 3]; 8] {
    let tw = 8 * BITS;
    let mut r = [[0u32; 3]; 8];
    unroll!(8, j, {
        unroll!(3, q, {
            if q < LW {
                // L1-allocating on purpose: neighbouring lanes and the q + 1 load reread the same sectors
                r[j][q] = ldg_u32(step.add((j as u32 * tw + (word + q as u32) % tw) as usize));
            }
        });
    });
    r
}

#[inline(always)]
unsafe fn gemv_body<const BITS: u32, const WK: u32, const LW: usize, const DEPTH: u32>(
    xh: *const u16,
    trellis: *const u32,
    svh: *const u16,
    y: *mut u8,
    z: *mut f32,
    counters: *mut u32,
    n: u32,
    k: u32,
    m: u32,
    sk: u32,
    y_f32: u32,
    stride_k: u32,
    stride_nb: u32,
) {
    let tw = 8 * BITS;
    let red: Shared<f32> = dyn_base();
    let last = static_shared!(gemv_last, u32, 4);
    let warp = warp_id();
    let ln = lane();
    let g = ln >> 2;
    let t = ln & 3;
    let nbs = n / 128;
    let nb = bid_x() % nbs;
    let split = bid_x() / nbs;
    let per_warp = (k / 16) / sk / WK;
    let kt0 = split * per_warp * WK + warp * per_warp;
    // words between k steps and between 128-column blocks: stored layout [k/16][n/16][tw] gives
    // ((n / 16) tw, 8 tw), the "strips" layout [n/128][k/16][8][tw] gives (8 tw, (k / 16) 8 tw)
    let tiles = trellis.add((nb * stride_nb) as usize);
    let col0 = nb * 128;
    let rh = min_u(m, 8);
    // the lane's first stream bit: end of value 8 lane's window - 16, plus one tile so it is non-negative
    let first = (8 * ln + 1) * BITS + 256 * BITS - 16;
    let word = (first >> 5) % tw;
    let off = first & 31;

    let mut m0 = 0;
    let mut pass = 0;
    while m0 < m {
        let rr = min_u(16, m - m0);
        let mut acc = [[[0f32; 4]; 2]; 8];
        let r0 = m0 + min_u(g, rr - 1);
        let r1 = m0 + min_u(g + 8, rr - 1);
        let x0 = xh.add((r0 * k) as usize) as *const u32;
        let x1 = xh.add((r1 * k) as usize) as *const u32;
        let tile = tiles.add((kt0 * stride_k) as usize);
        if pass == 0 {
            prefetch_steps::<BITS>(tile, stride_k, min_u(per_warp, 6), ln);
        }
        // DEPTH k steps of weight words in flight in registers ahead of the one being decoded
        let mut cur = load_step::<BITS, LW>(tile, word);
        if pass == 0 {
            pdl_wait(); // the weights do not depend on the previous kernel; xh, y, z, counters do
        }
        let mut nx1 = if DEPTH > 1 && per_warp > 1 {
            load_step::<BITS, LW>(tile.add(stride_k as usize), word)
        } else {
            cur
        };
        let mut i = 0;
        while i < per_warp {
            let kt = kt0 + i;
            let xo = (kt * 8 + t) as usize; // u32 (fp16 pair) index of k = 16 kt + 2t
            let u = [ldg_u32(x0.add(xo)), ldg_u32(x1.add(xo)), ldg_u32(x0.add(xo + 4)), ldg_u32(x1.add(xo + 4))];
            let nxt = if i + DEPTH < per_warp {
                load_step::<BITS, LW>(tile.add(((i + DEPTH) * stride_k) as usize), word)
            } else if DEPTH > 1 {
                nx1
            } else {
                cur // (not nx1: with one step in flight nx1 would stay live through the loop)
            };
            unroll!(8, j, {
                let (b0, b1) = decode_lane::<BITS, LW>(&cur[j], off);
                mma_f16_16816(&mut acc[j][0], u, b0);
                mma_f16_16816(&mut acc[j][1], u, b1);
            });
            if DEPTH > 1 {
                cur = nx1;
                nx1 = nxt;
            } else {
                cur = nxt;
            }
            i += 1;
        }
        if pass == 0 {
            pdl_trigger(); // only the reduction is left
        }

        // the warps' sums meet in shared memory: rows 0-7 of the pass, then rows 8-15
        let mut rlo = 0;
        while rlo < rr {
            let rn = min_u(rr - rlo, 8);
            sync_threads();
            if g < rh {
                unroll!(8, j, {
                    unroll!(2, h, {
                        let col = j as u32 * 16 + h as u32 * 8 + 2 * t;
                        let (a, b) = if rlo != 0 { (acc[j][h][2], acc[j][h][3]) } else { (acc[j][h][0], acc[j][h][1]) };
                        let p = red.add((warp * rh + g) * 128 + col);
                        p.st(a);
                        p.add(1).st(b);
                    });
                });
            }
            sync_threads();
            if sk == 1 {
                let mut r = warp;
                while r < rn {
                    let mut v = red.add(r * 128 + 4 * ln).cast::<[f32; 4]>().ld();
                    let mut w = 1;
                    while w < WK {
                        let q = red.add((w * rh + r) * 128 + 4 * ln).cast::<[f32; 4]>().ld();
                        v = [v[0] + q[0], v[1] + q[1], v[2] + q[2], v[3] + q[3]];
                        w += 1;
                    }
                    finish(v, ln, svh, col0 + 4 * ln, y, y_f32, ((m0 + rlo + r) * n + col0 + 4 * ln) as usize);
                    r += WK;
                }
            } else {
                let mut idx = tid_x();
                while idx < rn * 32 {
                    let r = idx >> 5;
                    let c = 4 * (idx & 31);
                    let mut v = red.add(r * 128 + c).cast::<[f32; 4]>().ld();
                    let mut w = 1;
                    while w < WK {
                        let q = red.add((w * rh + r) * 128 + c).cast::<[f32; 4]>().ld();
                        v = [v[0] + q[0], v[1] + q[1], v[2] + q[2], v[3] + q[3]];
                        w += 1;
                    }
                    stg_v4f(z.add(((split * m + m0 + rlo + r) * n + col0 + c) as usize), v);
                    idx += WK * 32;
                }
            }
            rlo += 8;
        }
        if sk > 1 {
            fence_gpu();
            sync_threads();
            if tid_x() == 0 {
                let old = atomic_add_u32(counters.add((pass * nbs + nb) as usize), 1);
                last.st((old == sk - 1) as u32);
            }
            sync_threads();
            if last.ld() != 0 {
                fence_gpu();
                let mut r = warp;
                while r < rr {
                    let at = ((m0 + r) * n + col0 + 4 * ln) as usize;
                    let mut v = ldcg_v4(z.add(at));
                    let mut q = 1;
                    while q < sk {
                        let o = ldcg_v4(z.add((q * m * n) as usize + at));
                        v = [v[0] + o[0], v[1] + o[1], v[2] + o[2], v[3] + o[3]];
                        q += 1;
                    }
                    finish(v, ln, svh, col0 + 4 * ln, y, y_f32, at);
                    r += WK;
                }
                if tid_x() == 0 {
                    *counters.add((pass * nbs + nb) as usize) = 0;
                }
            }
        }
        sync_threads();
        m0 += 16;
        pass += 1;
    }
}


/// This lane's tile as the mma A operand (W transposed: 16 n x 16 k): the trellis fragment order puts value j
/// at (k = 2t + (j & 1) + 8 ((j >> 1) & 1), n = g + 8 (j >> 2)), i.e. a0 = (n g, k 2t..), a1 = (n g+8, k 2t..),
/// a2 = (n g, k 2t+8..), a3 = (n g+8, k 2t+8..).
#[inline(always)]
fn decode_lane_a<const BITS: u32, const LW: usize>(w: &[u32; 3], off: u32) -> [u32; 4] {
    let (b0, b1) = decode_lane::<BITS, LW>(w, off);
    [b0[0], b1[0], b0[1], b1[1]]
}

/// v8: W^T as the A operand, x rows as B (8 rows a group, G groups a pass): one mma a tile and group, 32
/// accumulators a group. A row's bits do not depend on G or on the other rows (each group's mma chain is the
/// same), so 1..8 rows (G = 1), 9..16 (G = 2) and 17+ (G = 4, 32-row passes) give a row identical results.
#[inline(always)]
unsafe fn gemv8_body<const BITS: u32, const WK: u32, const LW: usize, const G: usize, const DEPTH: u32>(
    xh: *const u16,
    trellis: *const u32,
    svh: *const u16,
    y: *mut u8,
    z: *mut f32,
    counters: *mut u32,
    n: u32,
    k: u32,
    m: u32,
    sk: u32,
    y_f32: u32,
    stride_k: u32,
    stride_nb: u32,
) {
    let tw = 8 * BITS;
    let red: Shared<f32> = dyn_base();
    let last = static_shared!(gemv8_last, u32, 4);
    let warp = warp_id();
    let ln = lane();
    let g = ln >> 2;
    let t = ln & 3;
    let nbs = n / 128;
    let nb = bid_x() % nbs;
    let split = bid_x() / nbs;
    let per_warp = (k / 16) / sk / WK;
    let kt0 = split * per_warp * WK + warp * per_warp;
    let tiles = trellis.add((nb * stride_nb) as usize);
    let col0 = nb * 128;
    let rh = min_u(m, 8);
    let first = (8 * ln + 1) * BITS + 256 * BITS - 16;
    let word = (first >> 5) % tw;
    let off = first & 31;
    let rows_pass = 8 * G as u32;

    let mut m0 = 0;
    let mut pass = 0;
    while m0 < m {
        let rr = min_u(rows_pass, m - m0);
        let mut acc = [[[0f32; 4]; G]; 8];
        // B columns: x row 8 gi + g of the pass, clamped inside the buffer (extra rows are dropped)
        let mut xp = [xh as *const u32; G];
        unroll!(4, gi, {
            if gi < G {
                let r = m0 + min_u(8 * gi as u32 + g, rr - 1);
                xp[gi] = xh.add((r * k) as usize) as *const u32;
            }
        });
        let tile = tiles.add((kt0 * stride_k) as usize);
        if pass == 0 {
            prefetch_steps::<BITS>(tile, stride_k, min_u(per_warp, 6), ln);
        }
        let mut cur = load_step::<BITS, LW>(tile, word);
        let mut nx1 = if DEPTH > 1 && per_warp > 1 { load_step::<BITS, LW>(tile.add(stride_k as usize), word) } else { cur };
        if pass == 0 {
            pdl_wait(); // the weights do not depend on the previous kernel; xh, y, z, counters do
        }
        let mut i = 0;
        while i < per_warp {
            let xo = ((kt0 + i) * 8 + t) as usize;
            let mut bx = [[0u32; 2]; G];
            unroll!(4, gi, {
                if gi < G {
                    bx[gi] = [ldg_u32(xp[gi].add(xo)), ldg_u32(xp[gi].add(xo + 4))];
                }
            });
            let nxt = if i + DEPTH < per_warp {
                load_step::<BITS, LW>(tile.add(((i + DEPTH) * stride_k) as usize), word)
            } else if DEPTH > 1 {
                nx1
            } else {
                cur // (not nx1: with one step in flight nx1 would stay live through the loop)
            };
            unroll!(8, j, {
                let a = decode_lane_a::<BITS, LW>(&cur[j], off);
                unroll!(4, gi, {
                    if gi < G {
                        mma_f16_16816(&mut acc[j][gi], a, bx[gi]);
                    }
                });
            });
            if DEPTH > 1 {
                cur = nx1;
                nx1 = nxt;
            } else {
                cur = nxt;
            }
            i += 1;
        }
        if pass == 0 {
            pdl_trigger(); // only the reduction is left
        }

        // D of tile j, group gi: c0 (n 16j + g, row 8gi + 2t), c1 (.., 2t + 1), c2 (n 16j + g + 8, 2t), c3
        unroll!(4, gi, {
            if gi < G && (8 * gi as u32) < rr {
                let rlo = 8 * gi as u32;
                let rn = min_u(rr - rlo, 8);
                sync_threads();
                unroll!(8, j, {
                    let col = j as u32 * 16 + g;
                    let r = 2 * t;
                    if r < rn {
                        red.add((warp * rh + r) * 128 + col).st(acc[j][gi][0]);
                        red.add((warp * rh + r) * 128 + col + 8).st(acc[j][gi][2]);
                    }
                    if r + 1 < rn {
                        red.add((warp * rh + r + 1) * 128 + col).st(acc[j][gi][1]);
                        red.add((warp * rh + r + 1) * 128 + col + 8).st(acc[j][gi][3]);
                    }
                });
                sync_threads();
                if sk == 1 {
                    let mut r = warp;
                    while r < rn {
                        let mut v = red.add(r * 128 + 4 * ln).cast::<[f32; 4]>().ld();
                        let mut w = 1;
                        while w < WK {
                            let q = red.add((w * rh + r) * 128 + 4 * ln).cast::<[f32; 4]>().ld();
                            v = [v[0] + q[0], v[1] + q[1], v[2] + q[2], v[3] + q[3]];
                            w += 1;
                        }
                        finish(v, ln, svh, col0 + 4 * ln, y, y_f32, ((m0 + rlo + r) * n + col0 + 4 * ln) as usize);
                        r += WK;
                    }
                } else {
                    let mut idx = tid_x();
                    while idx < rn * 32 {
                        let r = idx >> 5;
                        let c = 4 * (idx & 31);
                        let mut v = red.add(r * 128 + c).cast::<[f32; 4]>().ld();
                        let mut w = 1;
                        while w < WK {
                            let q = red.add((w * rh + r) * 128 + c).cast::<[f32; 4]>().ld();
                            v = [v[0] + q[0], v[1] + q[1], v[2] + q[2], v[3] + q[3]];
                            w += 1;
                        }
                        stg_v4f(z.add(((split * m + m0 + rlo + r) * n + col0 + c) as usize), v);
                        idx += WK * 32;
                    }
                }
            }
        });
        if sk > 1 {
            fence_gpu();
            sync_threads();
            if tid_x() == 0 {
                let old = atomic_add_u32(counters.add((pass * nbs + nb) as usize), 1);
                last.st((old == sk - 1) as u32);
            }
            sync_threads();
            if last.ld() != 0 {
                fence_gpu();
                let mut r = warp;
                while r < rr {
                    let at = ((m0 + r) * n + col0 + 4 * ln) as usize;
                    let mut v = ldcg_v4(z.add(at));
                    let mut q = 1;
                    while q < sk {
                        let o = ldcg_v4(z.add((q * m * n) as usize + at));
                        v = [v[0] + o[0], v[1] + o[1], v[2] + o[2], v[3] + o[3]];
                        q += 1;
                    }
                    finish(v, ln, svh, col0 + 4 * ln, y, y_f32, at);
                    r += WK;
                }
                if tid_x() == 0 {
                    *counters.add((pass * nbs + nb) as usize) = 0;
                }
            }
        }
        sync_threads();
        m0 += rows_pass;
        pass += 1;
    }
}

/// One row group's reduction (the tail of `gemv8_body`'s group loop): the warps' partial sums for rows
/// rlo..rlo+rn go through `red` and are added in warp order, then finished (one split) or written to the
/// split partials `z`. `v[j]` is this lane's D fragment of tile j: c0 (n 16j + g, row 2t), c1 (row 2t + 1),
/// c2 (n 16j + g + 8, row 2t), c3.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn group8_reduce<const WK: u32>(
    v: &[[f32; 4]; 8],
    red: Shared<f32>,
    rh: u32,
    rlo: u32,
    rn: u32,
    m0: u32,
    m: u32,
    n: u32,
    col0: u32,
    split: u32,
    sk: u32,
    svh: *const u16,
    y: *mut u8,
    y_f32: u32,
    z: *mut f32,
) {
    let warp = warp_id();
    let ln = lane();
    let g = ln >> 2;
    let t = ln & 3;
    sync_threads();
    unroll!(8, j, {
        let col = j as u32 * 16 + g;
        let r = 2 * t;
        if r < rn {
            red.add((warp * rh + r) * 128 + col).st(v[j][0]);
            red.add((warp * rh + r) * 128 + col + 8).st(v[j][2]);
        }
        if r + 1 < rn {
            red.add((warp * rh + r + 1) * 128 + col).st(v[j][1]);
            red.add((warp * rh + r + 1) * 128 + col + 8).st(v[j][3]);
        }
    });
    sync_threads();
    if sk == 1 {
        let mut r = warp;
        while r < rn {
            let mut s = red.add(r * 128 + 4 * ln).cast::<[f32; 4]>().ld();
            let mut w = 1;
            while w < WK {
                let q = red.add((w * rh + r) * 128 + 4 * ln).cast::<[f32; 4]>().ld();
                s = [s[0] + q[0], s[1] + q[1], s[2] + q[2], s[3] + q[3]];
                w += 1;
            }
            finish(s, ln, svh, col0 + 4 * ln, y, y_f32, ((m0 + rlo + r) * n + col0 + 4 * ln) as usize);
            r += WK;
        }
    } else {
        let mut idx = tid_x();
        while idx < rn * 32 {
            let r = idx >> 5;
            let c = 4 * (idx & 31);
            let mut s = red.add(r * 128 + c).cast::<[f32; 4]>().ld();
            let mut w = 1;
            while w < WK {
                let q = red.add((w * rh + r) * 128 + c).cast::<[f32; 4]>().ld();
                s = [s[0] + q[0], s[1] + q[1], s[2] + q[2], s[3] + q[3]];
                w += 1;
            }
            stg_v4f(z.add(((split * m + m0 + rlo + r) * n + col0 + c) as usize), s);
            idx += WK * 32;
        }
    }
}

/// v8 with 48-row passes (`g6`, GS = 2) or 40-row passes (`g5`, GS = 1). The A fragments of a k step are decoded
/// once for 4 + GS row groups: groups 0-3 accumulate in registers exactly as `g4` (128 accumulators a lane, at the
/// register cap), groups 4.. in shared memory: each of their mmas loads its C fragment, runs the same `mma_f16_16816` and stores it back,
/// the same instruction on the same values as a register chain. The k ranges, the warp order of the sums and
/// the split order are those of g1 / g2 / g4, so a row's bits do not depend on the form or on how many rows
/// the call has. Dynamic smem: the warps' partial sums (WK x 8 rows x 512 B) then the shared accumulators
/// (WK x 2 groups x 8 tiles x 32 lanes x 16 B = WK x 8 KB): 96 KB at 8 warps.
#[inline(always)]
unsafe fn gemv8s_body<const BITS: u32, const WK: u32, const LW: usize, const GS: u32>(
    xh: *const u16,
    trellis: *const u32,
    svh: *const u16,
    y: *mut u8,
    z: *mut f32,
    counters: *mut u32,
    n: u32,
    k: u32,
    m: u32,
    sk: u32,
    y_f32: u32,
    stride_k: u32,
    stride_nb: u32,
) {
    const GR: usize = 4; // register groups; GS (1 or 2): shared-memory groups
    let tw = 8 * BITS;
    let red: Shared<f32> = dyn_base();
    let last = static_shared!(gemv8s_last, u32, 4);
    let warp = warp_id();
    let ln = lane();
    let g = ln >> 2;
    let t = ln & 3;
    let nbs = n / 128;
    let nb = bid_x() % nbs;
    let split = bid_x() / nbs;
    let per_warp = (k / 16) / sk / WK;
    let kt0 = split * per_warp * WK + warp * per_warp;
    let tiles = trellis.add((nb * stride_nb) as usize);
    let col0 = nb * 128;
    let rh = min_u(m, 8);
    // this lane's shared accumulators, after the partial-sum area (sized for 8 rows): [warp][group][tile j]
    // blocks of 32 lanes x 4 floats (one conflict-free v4 access a warp)
    let sacc: Shared<[f32; 4]> = red.add(WK * 8 * 128).cast::<[f32; 4]>().add(warp * GS * 8 * 32 + ln);
    let first = (8 * ln + 1) * BITS + 256 * BITS - 16;
    let word = (first >> 5) % tw;
    let off = first & 31;
    let rows_pass = 8 * (GR as u32 + GS);

    let mut m0 = 0;
    let mut pass = 0;
    while m0 < m {
        let rr = min_u(rows_pass, m - m0);
        // groups 4-5 hold rows of this pass (a 64-row call's second pass has 16)
        let wide = rr > 8 * GR as u32;
        let mut acc = [[[0f32; 4]; GR]; 8];
        if wide {
            unroll!(2, s, {
                if (s as u32) < GS {
                    unroll!(8, j, {
                        sacc.add(((s * 8 + j) * 32) as u32).st([0f32; 4]);
                    });
                }
            });
        }
        // B columns: x row 8 gi + g of the pass, clamped inside the buffer (extra rows are dropped)
        let mut xp = [xh as *const u32; GR];
        unroll!(4, gi, {
            let r = m0 + min_u(8 * gi as u32 + g, rr - 1);
            xp[gi] = xh.add((r * k) as usize) as *const u32;
        });
        let mut xs = [0u32; 2]; // u32 index of the shared groups' x rows
        unroll!(2, s, {
            let r = m0 + min_u(8 * (GR as u32 + s as u32) + g, rr - 1);
            xs[s] = r * (k / 2);
        });
        let tile = tiles.add((kt0 * stride_k) as usize);
        if pass == 0 {
            prefetch_steps::<BITS>(tile, stride_k, min_u(per_warp, 6), ln);
        }
        let mut cur = load_step::<BITS, LW>(tile, word);
        if pass == 0 {
            pdl_wait(); // the weights do not depend on the previous kernel; xh, y, z, counters do
        }
        let mut i = 0;
        while i < per_warp {
            let xo = ((kt0 + i) * 8 + t) as usize;
            let mut bx = [[0u32; 2]; GR];
            unroll!(4, gi, {
                bx[gi] = [ldg_u32(xp[gi].add(xo)), ldg_u32(xp[gi].add(xo + 4))];
            });
            let nxt = if i + 1 < per_warp {
                load_step::<BITS, LW>(tile.add(((i + 1) * stride_k) as usize), word)
            } else {
                cur
            };
            if wide {
                let mut bs = [[0u32; 2]; 2];
                unroll!(2, s, {
                    if (s as u32) < GS {
                        let p = (xh as *const u32).add(xs[s] as usize + xo);
                        bs[s] = [ldg_u32(p), ldg_u32(p.add(4))];
                    }
                });
                unroll!(8, j, {
                    let a = decode_lane_a::<BITS, LW>(&cur[j], off);
                    unroll!(4, gi, {
                        mma_f16_16816(&mut acc[j][gi], a, bx[gi]);
                    });
                    unroll!(2, s, {
                        if (s as u32) < GS {
                            let p = sacc.add(((s * 8 + j) * 32) as u32);
                            let mut c = p.ld();
                            mma_f16_16816(&mut c, a, bs[s]);
                            p.st(c);
                        }
                    });
                });
            } else {
                unroll!(8, j, {
                    let a = decode_lane_a::<BITS, LW>(&cur[j], off);
                    unroll!(4, gi, {
                        mma_f16_16816(&mut acc[j][gi], a, bx[gi]);
                    });
                });
            }
            cur = nxt;
            i += 1;
        }
        if pass == 0 {
            pdl_trigger(); // only the reduction is left
        }

        unroll!(4, gi, {
            if (8 * gi as u32) < rr {
                let v = [acc[0][gi], acc[1][gi], acc[2][gi], acc[3][gi], acc[4][gi], acc[5][gi], acc[6][gi], acc[7][gi]];
                let rlo = 8 * gi as u32;
                group8_reduce::<WK>(&v, red, rh, rlo, min_u(rr - rlo, 8), m0, m, n, col0, split, sk, svh, y, y_f32, z);
            }
        });
        unroll!(2, s, {
            let rlo = 8 * (GR as u32 + s as u32);
            if (s as u32) < GS && rlo < rr {
                let mut v = [[0f32; 4]; 8];
                unroll!(8, j, {
                    v[j] = sacc.add(((s * 8 + j) * 32) as u32).ld();
                });
                group8_reduce::<WK>(&v, red, rh, rlo, min_u(rr - rlo, 8), m0, m, n, col0, split, sk, svh, y, y_f32, z);
            }
        });
        if sk > 1 {
            fence_gpu();
            sync_threads();
            if tid_x() == 0 {
                let old = atomic_add_u32(counters.add((pass * nbs + nb) as usize), 1);
                last.st((old == sk - 1) as u32);
            }
            sync_threads();
            if last.ld() != 0 {
                fence_gpu();
                let mut r = warp;
                while r < rr {
                    let at = ((m0 + r) * n + col0 + 4 * ln) as usize;
                    let mut v = ldcg_v4(z.add(at));
                    let mut q = 1;
                    while q < sk {
                        let o = ldcg_v4(z.add((q * m * n) as usize + at));
                        v = [v[0] + o[0], v[1] + o[1], v[2] + o[2], v[3] + o[3]];
                        q += 1;
                    }
                    finish(v, ln, svh, col0 + 4 * ln, y, y_f32, at);
                    r += WK;
                }
                if tid_x() == 0 {
                    *counters.add((pass * nbs + nb) as usize) = 0;
                }
            }
        }
        sync_threads();
        m0 += rows_pass;
        pass += 1;
    }
}

macro_rules! gemv_kernel {
    ($name:ident, $bits:literal, $wk:literal, $lw:literal, $depth:literal) => {
        /// grid (n / 128 * sk), WK * 32 threads, dynamic smem WK * min(m, 8) * 512 bytes.
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            xh: *const u16,
            trellis: *const u32,
            svh: *const u16,
            y: *mut u8,
            z: *mut f32,
            counters: *mut u32,
            n: u32,
            k: u32,
            m: u32,
            sk: u32,
            y_f32: u32,
            stride_k: u32,
            stride_nb: u32,
        ) {
            gemv_body::<$bits, $wk, $lw, $depth>(xh, trellis, svh, y, z, counters, n, k, m, sk, y_f32, stride_k, stride_nb);
        }
    };
}

// exl3_gemv{_d2}_b{bits}_w{warps}[_o{programs an SM}]: d2 = two k steps prefetched; the `_o` suffix becomes
// `.maxntid` / `.minnctapersm` in the PTX (crates/fe-exl3/build.rs), i.e. CUDA's __launch_bounds__.
macro_rules! gemv_family {
    ($b:literal, $lw:literal, $d:literal, $w4:ident, $w4o3:ident, $w4o4:ident, $w8:ident, $w8o2:ident) => {
        gemv_kernel!($w4, $b, 4, $lw, $d);
        gemv_kernel!($w4o3, $b, 4, $lw, $d);
        gemv_kernel!($w4o4, $b, 4, $lw, $d);
        gemv_kernel!($w8, $b, 8, $lw, $d);
        gemv_kernel!($w8o2, $b, 8, $lw, $d);
    };
}
gemv_family!(2, 2, 1, exl3_gemv_b2_w4, exl3_gemv_b2_w4_o3, exl3_gemv_b2_w4_o4, exl3_gemv_b2_w8, exl3_gemv_b2_w8_o2);
gemv_family!(3, 2, 1, exl3_gemv_b3_w4, exl3_gemv_b3_w4_o3, exl3_gemv_b3_w4_o4, exl3_gemv_b3_w8, exl3_gemv_b3_w8_o2);
gemv_family!(4, 2, 1, exl3_gemv_b4_w4, exl3_gemv_b4_w4_o3, exl3_gemv_b4_w4_o4, exl3_gemv_b4_w8, exl3_gemv_b4_w8_o2);
gemv_family!(5, 3, 1, exl3_gemv_b5_w4, exl3_gemv_b5_w4_o3, exl3_gemv_b5_w4_o4, exl3_gemv_b5_w8, exl3_gemv_b5_w8_o2);
gemv_family!(2, 2, 2, exl3_gemv_d2_b2_w4, exl3_gemv_d2_b2_w4_o3, exl3_gemv_d2_b2_w4_o4, exl3_gemv_d2_b2_w8, exl3_gemv_d2_b2_w8_o2);
gemv_family!(3, 2, 2, exl3_gemv_d2_b3_w4, exl3_gemv_d2_b3_w4_o3, exl3_gemv_d2_b3_w4_o4, exl3_gemv_d2_b3_w8, exl3_gemv_d2_b3_w8_o2);
gemv_family!(4, 2, 2, exl3_gemv_d2_b4_w4, exl3_gemv_d2_b4_w4_o3, exl3_gemv_d2_b4_w4_o4, exl3_gemv_d2_b4_w8, exl3_gemv_d2_b4_w8_o2);
gemv_family!(5, 3, 2, exl3_gemv_d2_b5_w4, exl3_gemv_d2_b5_w4_o3, exl3_gemv_d2_b5_w4_o4, exl3_gemv_d2_b5_w8, exl3_gemv_d2_b5_w8_o2);

macro_rules! gemv8_kernel {
    ($name:ident, $bits:literal, $wk:literal, $lw:literal, $g:literal) => {
        gemv8_kernel!($name, $bits, $wk, $lw, $g, 1);
    };
    ($name:ident, $bits:literal, $wk:literal, $lw:literal, $g:literal, $d:literal) => {
        /// grid (n / 128 * sk), WK * 32 threads, dynamic smem WK * min(m, 8) * 512 bytes; passes of 8 G rows.
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            xh: *const u16,
            trellis: *const u32,
            svh: *const u16,
            y: *mut u8,
            z: *mut f32,
            counters: *mut u32,
            n: u32,
            k: u32,
            m: u32,
            sk: u32,
            y_f32: u32,
            stride_k: u32,
            stride_nb: u32,
        ) {
            gemv8_body::<$bits, $wk, $lw, $g, $d>(xh, trellis, svh, y, z, counters, n, k, m, sk, y_f32, stride_k, stride_nb);
        }
    };
}

// exl3_gemv8_b{bits}_w{warps}_g{groups}[_o{programs an SM}]
macro_rules! gemv8_family {
    ($b:literal, $lw:literal, $g:literal, $w4:ident, $w4o4:ident, $w4o5:ident, $w8:ident, $w8o2:ident) => {
        gemv8_kernel!($w4, $b, 4, $lw, $g);
        gemv8_kernel!($w4o4, $b, 4, $lw, $g);
        gemv8_kernel!($w4o5, $b, 4, $lw, $g);
        gemv8_kernel!($w8, $b, 8, $lw, $g);
        gemv8_kernel!($w8o2, $b, 8, $lw, $g);
    };
}
gemv8_family!(2, 2, 1, exl3_gemv8_b2_w4_g1, exl3_gemv8_b2_w4_g1_o4, exl3_gemv8_b2_w4_g1_o5, exl3_gemv8_b2_w8_g1, exl3_gemv8_b2_w8_g1_o2);
gemv8_family!(3, 2, 1, exl3_gemv8_b3_w4_g1, exl3_gemv8_b3_w4_g1_o4, exl3_gemv8_b3_w4_g1_o5, exl3_gemv8_b3_w8_g1, exl3_gemv8_b3_w8_g1_o2);
gemv8_family!(4, 2, 1, exl3_gemv8_b4_w4_g1, exl3_gemv8_b4_w4_g1_o4, exl3_gemv8_b4_w4_g1_o5, exl3_gemv8_b4_w8_g1, exl3_gemv8_b4_w8_g1_o2);
gemv8_family!(5, 3, 1, exl3_gemv8_b5_w4_g1, exl3_gemv8_b5_w4_g1_o4, exl3_gemv8_b5_w4_g1_o5, exl3_gemv8_b5_w8_g1, exl3_gemv8_b5_w8_g1_o2);
gemv8_family!(2, 2, 2, exl3_gemv8_b2_w4_g2, exl3_gemv8_b2_w4_g2_o4, exl3_gemv8_b2_w4_g2_o5, exl3_gemv8_b2_w8_g2, exl3_gemv8_b2_w8_g2_o2);
gemv8_family!(3, 2, 2, exl3_gemv8_b3_w4_g2, exl3_gemv8_b3_w4_g2_o4, exl3_gemv8_b3_w4_g2_o5, exl3_gemv8_b3_w8_g2, exl3_gemv8_b3_w8_g2_o2);
gemv8_family!(4, 2, 2, exl3_gemv8_b4_w4_g2, exl3_gemv8_b4_w4_g2_o4, exl3_gemv8_b4_w4_g2_o5, exl3_gemv8_b4_w8_g2, exl3_gemv8_b4_w8_g2_o2);
gemv8_family!(5, 3, 2, exl3_gemv8_b5_w4_g2, exl3_gemv8_b5_w4_g2_o4, exl3_gemv8_b5_w4_g2_o5, exl3_gemv8_b5_w8_g2, exl3_gemv8_b5_w8_g2_o2);
gemv8_family!(2, 2, 4, exl3_gemv8_b2_w4_g4, exl3_gemv8_b2_w4_g4_o4, exl3_gemv8_b2_w4_g4_o5, exl3_gemv8_b2_w8_g4, exl3_gemv8_b2_w8_g4_o2);
gemv8_family!(3, 2, 4, exl3_gemv8_b3_w4_g4, exl3_gemv8_b3_w4_g4_o4, exl3_gemv8_b3_w4_g4_o5, exl3_gemv8_b3_w8_g4, exl3_gemv8_b3_w8_g4_o2);
gemv8_family!(4, 2, 4, exl3_gemv8_b4_w4_g4, exl3_gemv8_b4_w4_g4_o4, exl3_gemv8_b4_w4_g4_o5, exl3_gemv8_b4_w8_g4, exl3_gemv8_b4_w8_g4_o2);
gemv8_family!(5, 3, 4, exl3_gemv8_b5_w4_g4, exl3_gemv8_b5_w4_g4_o4, exl3_gemv8_b5_w4_g4_o5, exl3_gemv8_b5_w8_g4, exl3_gemv8_b5_w8_g4_o2);
// 24-row passes (17..24 rows: g4 would run a fourth group of clamped rows)
gemv8_family!(2, 2, 3, exl3_gemv8_b2_w4_g3, exl3_gemv8_b2_w4_g3_o4, exl3_gemv8_b2_w4_g3_o5, exl3_gemv8_b2_w8_g3, exl3_gemv8_b2_w8_g3_o2);
gemv8_family!(3, 2, 3, exl3_gemv8_b3_w4_g3, exl3_gemv8_b3_w4_g3_o4, exl3_gemv8_b3_w4_g3_o5, exl3_gemv8_b3_w8_g3, exl3_gemv8_b3_w8_g3_o2);
gemv8_family!(4, 2, 3, exl3_gemv8_b4_w4_g3, exl3_gemv8_b4_w4_g3_o4, exl3_gemv8_b4_w4_g3_o5, exl3_gemv8_b4_w8_g3, exl3_gemv8_b4_w8_g3_o2);
gemv8_family!(5, 3, 3, exl3_gemv8_b5_w4_g3, exl3_gemv8_b5_w4_g3_o4, exl3_gemv8_b5_w4_g3_o5, exl3_gemv8_b5_w8_g3, exl3_gemv8_b5_w8_g3_o2);

// depth-2 prefetch forms of the 8- and 16-row kernels: exl3_gemv8d_b{bits}_w{warps}_g{1,2}[_o{occ}]
macro_rules! gemv8d_family {
    ($b:literal, $lw:literal, $g:literal, $w4:ident, $w4o4:ident, $w8:ident, $w8o2:ident) => {
        gemv8_kernel!($w4, $b, 4, $lw, $g, 2);
        gemv8_kernel!($w4o4, $b, 4, $lw, $g, 2);
        gemv8_kernel!($w8, $b, 8, $lw, $g, 2);
        gemv8_kernel!($w8o2, $b, 8, $lw, $g, 2);
    };
}
gemv8d_family!(2, 2, 1, exl3_gemv8d_b2_w4_g1, exl3_gemv8d_b2_w4_g1_o4, exl3_gemv8d_b2_w8_g1, exl3_gemv8d_b2_w8_g1_o2);
gemv8d_family!(3, 2, 1, exl3_gemv8d_b3_w4_g1, exl3_gemv8d_b3_w4_g1_o4, exl3_gemv8d_b3_w8_g1, exl3_gemv8d_b3_w8_g1_o2);
gemv8d_family!(4, 2, 1, exl3_gemv8d_b4_w4_g1, exl3_gemv8d_b4_w4_g1_o4, exl3_gemv8d_b4_w8_g1, exl3_gemv8d_b4_w8_g1_o2);
gemv8d_family!(5, 3, 1, exl3_gemv8d_b5_w4_g1, exl3_gemv8d_b5_w4_g1_o4, exl3_gemv8d_b5_w8_g1, exl3_gemv8d_b5_w8_g1_o2);
gemv8d_family!(2, 2, 2, exl3_gemv8d_b2_w4_g2, exl3_gemv8d_b2_w4_g2_o4, exl3_gemv8d_b2_w8_g2, exl3_gemv8d_b2_w8_g2_o2);
gemv8d_family!(3, 2, 2, exl3_gemv8d_b3_w4_g2, exl3_gemv8d_b3_w4_g2_o4, exl3_gemv8d_b3_w8_g2, exl3_gemv8d_b3_w8_g2_o2);
gemv8d_family!(4, 2, 2, exl3_gemv8d_b4_w4_g2, exl3_gemv8d_b4_w4_g2_o4, exl3_gemv8d_b4_w8_g2, exl3_gemv8d_b4_w8_g2_o2);
gemv8d_family!(5, 3, 2, exl3_gemv8d_b5_w4_g2, exl3_gemv8d_b5_w4_g2_o4, exl3_gemv8d_b5_w8_g2, exl3_gemv8d_b5_w8_g2_o2);

macro_rules! gemv8s_kernel {
    ($name:ident, $bits:literal, $wk:literal, $lw:literal, $gs:literal) => {
        /// grid (n / 128 * sk), WK * 32 threads, dynamic smem WK * (4 + 4 GS) KB (`Plan::smem`); passes of 32 + 8 GS
        /// rows.
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            xh: *const u16,
            trellis: *const u32,
            svh: *const u16,
            y: *mut u8,
            z: *mut f32,
            counters: *mut u32,
            n: u32,
            k: u32,
            m: u32,
            sk: u32,
            y_f32: u32,
            stride_k: u32,
            stride_nb: u32,
        ) {
            gemv8s_body::<$bits, $wk, $lw, $gs>(xh, trellis, svh, y, z, counters, n, k, m, sk, y_f32, stride_k, stride_nb);
        }
    };
}

// 48-row passes, groups 4-5 accumulating in shared memory: exl3_gemv8_b{bits}_w{warps}_g6 (never launch-bounded)
gemv8s_kernel!(exl3_gemv8_b2_w4_g6, 2, 4, 2, 2);
gemv8s_kernel!(exl3_gemv8_b2_w8_g6, 2, 8, 2, 2);
gemv8s_kernel!(exl3_gemv8_b3_w4_g6, 3, 4, 2, 2);
gemv8s_kernel!(exl3_gemv8_b3_w8_g6, 3, 8, 2, 2);
gemv8s_kernel!(exl3_gemv8_b4_w4_g6, 4, 4, 2, 2);
gemv8s_kernel!(exl3_gemv8_b4_w8_g6, 4, 8, 2, 2);
gemv8s_kernel!(exl3_gemv8_b5_w4_g6, 5, 4, 3, 2);
gemv8s_kernel!(exl3_gemv8_b5_w8_g6, 5, 8, 3, 2);
// 40-row passes (33..40 rows), group 4 in shared memory: exl3_gemv8_b{bits}_w{warps}_g5
gemv8s_kernel!(exl3_gemv8_b2_w4_g5, 2, 4, 2, 1);
gemv8s_kernel!(exl3_gemv8_b2_w8_g5, 2, 8, 2, 1);
gemv8s_kernel!(exl3_gemv8_b3_w4_g5, 3, 4, 2, 1);
gemv8s_kernel!(exl3_gemv8_b3_w8_g5, 3, 8, 2, 1);
gemv8s_kernel!(exl3_gemv8_b4_w4_g5, 4, 4, 2, 1);
gemv8s_kernel!(exl3_gemv8_b4_w8_g5, 4, 8, 2, 1);
gemv8s_kernel!(exl3_gemv8_b5_w4_g5, 5, 4, 3, 1);
gemv8s_kernel!(exl3_gemv8_b5_w8_g5, 5, 8, 3, 1);
