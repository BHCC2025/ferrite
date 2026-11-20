//! Prompt-path int8 GEMM ("MMQ"): `y[yrow(i)][n] (= or +=) W[n] . x[xrow(i)]` with q8 activations (the decode
//! GEMV's `ggq_q8_1` layout: int8 [rows][K], f32 d and s per 32) and the weights decoded to int8 sub-blocks
//! (`vq`, the exact integer codes + per-sub-block scales), multiplied with `mma.sync` m16n8k32 s8 -> s32 (4x the
//! bf16 tensor rate on sm_120), scaled per 32-value sub-block in f32. Same numerics as the decode GEMV up to the
//! f32 summation order.
//!
//! Tile BM (rows, 128) x 64 (n) x 128 (k = 4 sub-blocks) per stage, BM / 16 warps (32 x 32 each), 2 stages.
//! Row tiles: dense (`tiles` null: tile y = bid_y, rows y*64.., weights `w`), or a tile list from
//! `ggq_moe_sort` (`tiles[2t]` = segment/expert, `tiles[2t+1]` = first row; `seg_off` bounds, `segw[seg]` = that
//! expert's weights, 0 = skip; `xidx` / `yidx` = activation / output row of each sorted row).
use crate::vq::*;
use kcommon::mma::*;
use kcommon::smem::*;
use kcommon::*;

/// Default tile rows (the Flash-Next launches); the grouped prompt path can also use 64 and 32 (`ggq_mmq_*_bm64`, `_bm32`):
/// every output element is summed in the same k order whatever BM, so the bits do not depend on the tile rows.
const BM: u32 = 128;
/// Bench-only ablation switches (see `mmq`); 0 in normal builds.
const DBG: u32 = 0;
/// Canonical per-output order (shared with the decode GEMV and the CPU): v_j = (d8 * base) * isum (- m * s8) per
/// 32-value sub-block, summed per 4-sub-block chunk from 0, chunks added to the output in K order.
const CANON: bool = false;
const BN: u32 = 64;
const SB: u32 = 4; // sub-blocks per stage
const BK: u32 = 32 * SB; // 128
const LDB: u32 = BK + 16; // bytes per smem row (conflict-free ldmatrix)
const TB: u32 = BN * LDB;
const NP: u32 = 128; // producer threads (4 warps)
const TSB: u32 = BN * SB * 4; // f32 [BN][SB] weight scales
/// stage layout: A | B | ad | as | base | m | iab (2 x i32); A and ad / as scale with the tile rows
const fn off_ad(bm: u32) -> u32 {
    bm * LDB + TB
}
const fn off_as(bm: u32) -> u32 {
    off_ad(bm) + bm * SB * 4
}
const fn off_base(bm: u32) -> u32 {
    off_as(bm) + bm * SB * 4
}
const fn off_m(bm: u32) -> u32 {
    off_base(bm) + TSB
}
const fn off_iab(bm: u32) -> u32 {
    off_m(bm) + TSB
}
pub const fn stage_bytes(bm: u32) -> u32 {
    off_iab(bm) + 2 * TSB
}
/// Stage buffers of the producer/consumer pipeline.
const NSTAGE: u32 = 2;
pub const fn nstage(_bm: u32) -> u32 {
    NSTAGE
}
pub const fn mmq_smem(bm: u32) -> u32 {
    nstage(bm) * stage_bytes(bm)
}
pub const MMQ_SMEM: u32 = mmq_smem(128);

#[inline(always)]
pub fn mma_s8_16832(d: &mut [i32; 4], a: [u32; 4], b: [u32; 2]) {
    unsafe {
        core::arch::asm!("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], options(nomem, nostack));
    }
}
#[inline(always)]
pub fn mma_s8_16816(d: &mut [i32; 4], a0: u32, a1: u32, b0: u32) {
    unsafe {
        core::arch::asm!("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}}}, {{{b0}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a0, a1 = in(reg32) a1, b0 = in(reg32) b0, options(nomem, nostack));
    }
}
/// Exact int -> f32 for |v| < 2^22 with full-rate ops (no I2F): bits(v + 0x4b400000) - 1.5 * 2^23.
#[inline(always)]
fn i2f(v: i32) -> f32 {
    f32::from_bits((v + 0x4b40_0000) as u32) - 12582912.0
}

struct Ctx {
    aq: *const i8,
    ad: *const f32,
    as_: *const f32,
    k: u32,
    nsb: u32,
}

#[inline(always)]
#[allow(non_snake_case)]
pub unsafe fn mmq_bm<V: Vq, const BM: u32>(y: *mut f32, ldy: u32, aq: *const i8, ad: *const f32, as_: *const f32, w: *const u8,
    row_bytes: u64, m: u32, k: u32, xidx: *const u32, yidx: *const u32, segw: *const u64, seg_off: *const u32,
    tiles: *const u32, ntiles: *const u32, add: u32, segw2: *const u64, hq: *mut i8, wts: *const f32, lim: f32) {
    // weighted token accumulation (wts non-null, expert tiles): output row of sorted row r is the token yidx[r] / m
    // (m = topk), value scaled by wts[yidx[r]] and added with red.global.add (the MoE combine fused into down).
    // gate/up fusion (hq non-null, expert tiles only): the 64-row weight tile is 32 gate rows + the same 32 up rows
    // (`segw2`); the epilogue forms h = silu(g) * u for the tile's 32 h columns and writes them as one q8 block per
    // activation row: q to hq [yrow][nff], d to `y` [yrow][nff / 32], s to `w` (as f32) [yrow][nff / 32]; nff = ldy.
    // two stages; barriers 1, 2 = stage full, 3, 4 = stage free, 5 = the gate/up exchange
    let NT: u32 = 2 * BM; // consumer threads (BM / 32 x 2 warps of 32 x 32)
    let (TA, OFF_AD, OFF_AS, OFF_BASE, OFF_M, OFF_IAB, STAGE) = (BM * LDB, off_ad(BM), off_as(BM), off_base(BM), off_m(BM), off_iab(BM), stage_bytes(BM));
    let gu = !hq.is_null();
    let (hd, hs, nff) = (y, w as *mut f32, ldy);
    // ablation bits (bench only, compile-time): 1 no weight decode, 2 no float epilogue, 4 no mma
    let dbg = DBG;
    let add = add & 1;
    let (r0, r1, wb, wb2) = if tiles.is_null() {
        (bid_y() * BM, m, w, w)
    } else {
        if bid_y() >= *ntiles {
            return;
        }
        let sg = *tiles.add(2 * bid_y() as usize) as usize;
        let wp = *segw.add(sg);
        let wp2 = if gu { *segw2.add(sg) } else { 1 };
        if wp == 0 || wp2 == 0 {
            return;
        }
        (*tiles.add(2 * bid_y() as usize + 1), *seg_off.add(sg + 1), wp as *const u8, wp2 as *const u8)
    };
    if r0 >= r1 {
        return;
    }
    let n0 = bid_x() * BN;
    let t = tid_x();
    let c = Ctx { aq, ad, as_, k, nsb: k / 32 };
    let base = dyn_base::<u8>();
    let stg = [base, base.byte_add(STAGE), base.byte_add(2 * STAGE), base.byte_add(3 * STAGE)];
    let nst = k / BK;
    // Warp specialization: warps 0..8 (NT threads) compute, warps 8..12 (NP threads) produce the stages (A tile +
    // scales by cp.async, the weight tile decoded to int8). Two stage buffers; named barriers 1/2 = stage b full
    // (producers arrive, consumers wait), 3/4 = stage b free again (consumers arrive, producers wait).
    if t >= NT {
        let tp = t - NT;
        let src_row = |gi: u32| -> (u32, bool) {
            if gi < r1 { (if xidx.is_null() { gi } else { *xidx.add(gi as usize) }, true) } else { (0, false) }
        };
        let mut arow = [0u32; 8];
        let mut aok = [false; 8];
        unroll!(i in 8 => {
            if (i as u32) * (NP / 8) < BM {
                let (r, ok) = src_row(r0 + tp / 8 + (NP / 8) * i as u32);
                arow[i] = r;
                aok[i] = ok;
            }
        });
        let (srow, sok) = src_row(r0 + tp % BM);
        let wrow = if gu {
            // tile rows 0..32 = gate rows n0/2 .., 32..64 = up rows n0/2 ..
            let h0 = n0 / 2;
            [wb.add((h0 + tp / SB) as usize * row_bytes as usize), wb2.add((h0 + (tp + NP) / SB - 32) as usize * row_bytes as usize)]
        } else {
            [wb.add((n0 + tp / SB) as usize * row_bytes as usize), wb.add((n0 + (tp + NP) / SB) as usize * row_bytes as usize)]
        };
        // two stage buffers, unrolled by hand so every barrier id is a compile-time constant
        macro_rules! pstep {
            ($kt:expr, $bsel:literal) => {{
                let kt: u32 = $kt;
                if kt >= 2 {
                    bar_sync_c::<{ 3 + $bsel }>(NT + NP);
                }
            if BM < 128 {
                // small tiles are latency bound (few rows to hide the weight loads behind): pull the weight rows' lines of
                // the next stages into L2 ahead of their decode
                let off = (((kt + 6) * SB) as u64 * V::BYTES as u64 * 32 / V::QK as u64) as usize;
                if kt + 6 < nst {
                    core::arch::asm!("prefetch.global.L2 [{a}];", a = in(reg64) wrow[0].add(off), options(nostack, readonly));
                    core::arch::asm!("prefetch.global.L2 [{a}];", a = in(reg64) wrow[1].add(off), options(nostack, readonly));
                }
            }
            produce::<V, BM>(stg[$bsel as usize], &c, kt * SB, &arow, &aok, srow, sok, &wrow);
            cp_async_wait_all();
                bar_arrive_c::<{ 1 + $bsel }>(NT + NP);
            }};
        }
        let mut kt0 = 0;
        while kt0 < nst {
            pstep!(kt0, 0);
            if kt0 + 1 < nst {
                pstep!(kt0 + 1, 1);
            }
            kt0 += 2;
        }
        return;
    }
    let wm = (warp_id() / 2) * 32;
    let wn = (warp_id() % 2) * 32;
    let l = lane();
    let g = l / 4;
    let tq = l % 4;
    let mut acc = [[[0f32; 4]; 4]; 2];
    let dbg = DBG;
    let _ = dbg;
    macro_rules! cstage {
        ($kt:expr, $bsel:literal) => {{
            let kt: u32 = $kt;
            bar_sync_c::<{ 1 + $bsel }>(NT + NP);
        let st = stg[$bsel as usize];
        // canonical order: the stage's 4 sub-blocks summed into a chunk partial from 0, then added to the total
        let mut tmp = [[[0f32; 4]; 4]; 2];
        unroll!(jli in 4 => {
            let jl = jli as u32;
            let mut af = [[0u32; 4]; 2];
            unroll!(mi in 2 => {
                let a = st.byte_add((wm + mi as u32 * 16 + l % 16) * LDB + 32 * jl + (l / 16) * 16);
                af[mi] = ldmatrix_x4(a.addr());
            });
            // per-row activation scales (rows g, g + 8 of each m16) and per-col weight scales (cols 2t, 2t+1)
            let mut d8 = [0f32; 4];
            let mut s8 = [0f32; 4];
            unroll!(mh in 4 => {
                let r = wm + (mh / 2) as u32 * 16 + g + 8 * (mh % 2) as u32;
                d8[mh] = st.byte_add(OFF_AD + (r * SB + jl) * 4).cast::<f32>().ld();
                if V::HAS_M {
                    s8[mh] = st.byte_add(OFF_AS + (r * SB + jl) * 4).cast::<f32>().ld();
                }
            });
            unroll!(ni in 4 => {
                let b = st.byte_add(TA + (wn + ni as u32 * 8 + l % 8) * LDB + 32 * jl + ((l / 8) % 2) * 16);
                let bf = ldmatrix_x2(b.addr());
                let mut wb2 = [0f32; 2];
                let mut wm2 = [0f32; 2];
                let mut iab = [[1i32; 2]; 2];
                unroll!(e in 2 => {
                    let col = wn + ni as u32 * 8 + 2 * tq + e as u32;
                    wb2[e] = st.byte_add(OFF_BASE + (col * SB + jl) * 4).cast::<f32>().ld();
                    if V::HAS_M {
                        wm2[e] = st.byte_add(OFF_M + (col * SB + jl) * 4).cast::<f32>().ld();
                    }
                    if !V::SAME {
                        let v = st.byte_add(OFF_IAB + (col * SB + jl) * 8).cast::<[u32; 2]>().ld();
                        iab[e] = [v[0] as i32, v[1] as i32];
                    }
                });
                unroll!(mi in 2 => {
                    let mut isum = [0i32; 4];
                    if V::SAME {
                        if dbg & 4 == 0 { mma_s8_16832(&mut isum, af[mi], bf); } else { isum[0] = (af[mi][0] ^ bf[0]) as i32; }
                    } else {
                        let mut lo = [0i32; 4];
                        let mut hi = [0i32; 4];
                        mma_s8_16816(&mut lo, af[mi][0], af[mi][1], bf[0]);
                        mma_s8_16816(&mut hi, af[mi][2], af[mi][3], bf[1]);
                        unroll!(q in 4 => {
                            isum[q] = lo[q] * iab[q & 1][0] + hi[q] * iab[q & 1][1];
                        });
                    }
                    unroll!(q in 4 => {
                        let mut v = d8[2 * mi + q / 2] * wb2[q & 1] * i2f(isum[q]);
                        if V::HAS_M {
                            v -= wm2[q & 1] * s8[2 * mi + q / 2];
                        }
                        if CANON {
                            tmp[mi][ni][q] += v;
                        } else {
                            acc[mi][ni][q] += v;
                        }
                    });
                });
            });
        });
        if CANON {
            unroll!(mi in 2 => { unroll!(ni in 4 => { unroll!(q in 4 => { acc[mi][ni][q] += tmp[mi][ni][q]; }); }); });
        }
            if kt + 2 < nst {
                bar_arrive_c::<{ 3 + $bsel }>(NT + NP);
            }
        }};
    }
    let mut kt0 = 0;
    while kt0 < nst {
        cstage!(kt0, 0);
        if kt0 + 1 < nst {
            cstage!(kt0 + 1, 1);
        }
        kt0 += 2;
    }
    if gu {
        // consumers only: stage buffers are free (every stage consumed); exchange through smem as f32 [BM][65]
        bar_sync_c::<5>(NT);
        let ex = base.cast::<f32>();
        unroll!(mh in 4 => {
            let (mi, h) = (mh / 2, mh % 2);
            let r = wm + mi as u32 * 16 + g + 8 * h as u32;
            unroll!(ni in 4 => {
                let col = wn + ni as u32 * 8 + 2 * tq;
                ex.add(r * 65 + col).st(acc[mi][ni][2 * h]);
                ex.add(r * 65 + col + 1).st(acc[mi][ni][2 * h + 1]);
            });
        });
        bar_sync_c::<5>(NT);
        // two threads per activation row, 16 h values each
        let r = t / 2;
        let half = t % 2;
        let gi = r0 + r;
        let mut hv = [0f32; 16];
        let mut amax = 0f32;
        let mut sum = 0f32;
        unroll!(i in 16 => {
            let c = half * 16 + i as u32;
            let gv = ex.add(r * 65 + c).ld();
            let uv = ex.add(r * 65 + 32 + c).ld();
            hv[i] = crate::moe::swiglu(gv, uv, lim);
            amax = amax.max(hv[i].abs());
            sum += hv[i];
        });
        amax = amax.max(warp::shfl_xor(amax, 1));
        sum += warp::shfl_xor(sum, 1);
        if gi < r1 {
            let yrow = if yidx.is_null() { gi } else { *yidx.add(gi as usize) };
            let d = amax / 127.0;
            let c0 = n0 / 2; // h column of the block
            let mut qw = [0u32; 4];
            unroll!(i in 16 => {
                let q = if amax == 0.0 { 0i32 } else { roundf(hv[i] / d) as i32 };
                qw[i / 4] |= ((q as u32) & 0xff) << (8 * (i % 4));
            });
            stg_v4(hq.add(yrow as usize * nff as usize + (c0 + half * 16) as usize) as *mut u32, qw);
            if half == 0 {
                let nb = (nff / 32) as usize;
                *hd.add(yrow as usize * nb + (c0 / 32) as usize) = d;
                *hs.add(yrow as usize * nb + (c0 / 32) as usize) = f16_round(sum);
            }
        }
        return;
    }
    unroll!(mh in 4 => {
        let (mi, h) = (mh / 2, mh % 2);
        let gi = r0 + wm + mi as u32 * 16 + g + 8 * h as u32;
        if gi < r1 {
            let yrow = if yidx.is_null() { gi } else { *yidx.add(gi as usize) };
            if !wts.is_null() {
                // deterministic: fixed point i64 (2^40 = 1 unit) accumulated with integer atomics (order-free)
                let wv = *wts.add(yrow as usize);
                let yp = (y as *mut u64).add((yrow / m) as usize * ldy as usize);
                let status = w as *mut u32;
                unroll!(ni in 4 => {
                    let col = n0 + wn + ni as u32 * 8 + 2 * tq;
                    unroll!(e in 2 => {
                        let v = wv * acc[mi][ni][2 * h + e];
                        if !(v.abs() < FX_MAX) {
                            // out of range (or NaN): flag it; the value is dropped
                            atomic_add_u32(status, 1);
                        } else {
                            red_add_u64(yp.add(col as usize + e), to_fx(v));
                        }
                    });
                });
            } else {
            let yp = y.add(yrow as usize * ldy as usize);
            unroll!(ni in 4 => {
                let col = n0 + wn + ni as u32 * 8 + 2 * tq;
                let v0 = acc[mi][ni][2 * h];
                let v1 = acc[mi][ni][2 * h + 1];
                let o = yp.add(col as usize);
                if add != 0 {
                    *o += v0;
                    *o.add(1) += v1;
                } else {
                    *o = v0;
                    *o.add(1) = v1;
                }
            });
            }
        }
    });
}

#[inline(always)]
pub unsafe fn mmq<V: Vq>(y: *mut f32, ldy: u32, aq: *const i8, ad: *const f32, as_: *const f32, w: *const u8,
    row_bytes: u64, m: u32, k: u32, xidx: *const u32, yidx: *const u32, segw: *const u64, seg_off: *const u32,
    tiles: *const u32, ntiles: *const u32, add: u32, segw2: *const u64, hq: *mut i8, wts: *const f32, lim: f32) {
    mmq_bm::<V, 128>(y, ldy, aq, ad, as_, w, row_bytes, m, k, xidx, yidx, segw, seg_off, tiles, ntiles, add, segw2, hq, wts, lim)
}

/// Producer side of one stage (NP threads): A tile rows tp / 8 + 16 i (8 per thread), d and s of row tp, and 2 of the
/// BN * SB weight sub-blocks decoded.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
unsafe fn produce<V: Vq, const BM: u32>(st: Shared<u8>, c: &Ctx, j0: u32, arow: &[u32; 8], aok: &[bool; 8], srow: u32, sok: bool,
    wrow: &[*const u8; 2]) {
    let NT: u32 = 2 * BM;
    let (TA, OFF_AD, OFF_AS, OFF_BASE, OFF_M, OFF_IAB) = (BM * LDB, off_ad(BM), off_as(BM), off_base(BM), off_m(BM), off_iab(BM));
    let tp = tid_x() - NT;
    unroll!(i in 8 => {
        if (i as u32) * (NP / 8) < BM {
            let r = tp / 8 + (NP / 8) * i as u32;
            let ch = tp % 8;
            let dst = st.byte_add(r * LDB + ch * 16).addr();
            let src = c.aq.add(arow[i] as usize * c.k as usize + (32 * j0 + 16 * ch) as usize) as *const u8;
            cp_async_16_zfill(dst, src, if aok[i] { 16 } else { 0 });
        }
    });
    {
        let r = tp % BM;
        let sd = c.ad.add(srow as usize * c.nsb as usize + j0 as usize) as *const u8;
        cp_async_16_zfill(st.byte_add(OFF_AD + r * SB * 4).addr(), sd, if sok { 16 } else { 0 });
        let ss = c.as_.add(srow as usize * c.nsb as usize + j0 as usize) as *const u8;
        cp_async_16_zfill(st.byte_add(OFF_AS + r * SB * 4).addr(), ss, if sok { 16 } else { 0 });
    }
    cp_async_commit();
    unroll!(i in 2 => {
        let idx = tp + NP * i as u32;
        let r = idx / SB;
        let jl = idx % SB;
        let sb = if DBG & 1 != 0 { Sb { w: [idx; 8], base: 1.0, ia: 1, ib: 1, m: 0.0 } } else { V::sb(wrow[i], j0 + jl) };
        let o = st.byte_add(TA + r * LDB + 32 * jl);
        o.cast::<[u32; 4]>().st([sb.w[0], sb.w[1], sb.w[2], sb.w[3]]);
        o.byte_add(16).cast::<[u32; 4]>().st([sb.w[4], sb.w[5], sb.w[6], sb.w[7]]);
        st.byte_add(OFF_BASE + (r * SB + jl) * 4).cast::<f32>().st(sb.base);
        if V::HAS_M {
            st.byte_add(OFF_M + (r * SB + jl) * 4).cast::<f32>().st(sb.m);
        }
        if !V::SAME {
            st.byte_add(OFF_IAB + (r * SB + jl) * 8).cast::<[u32; 2]>().st([sb.ia as u32, sb.ib as u32]);
        }
    });
}

/// Fixed point of the deterministic MoE combine: value * 2^40 as i64 (range +-2^22 per term, 9e-13 resolution).
pub const FX_SHIFT: i32 = 40;
const FX_MAX: f32 = 4_194_304.0; // 2^22
#[inline(always)]
fn to_fx(v: f32) -> u64 {
    // v * 2^40 is exact in f32 (power-of-two scale), and so is its conversion: no f64 (slow on GeForce)
    let r: i64;
    let d = v * (1u64 << FX_SHIFT) as f32;
    unsafe { core::arch::asm!("cvt.rni.s64.f32 {r}, {d};", r = out(reg64) r, d = in(reg32) d, options(pure, nomem, nostack)) }
    r as u64
}
#[inline(always)]
unsafe fn red_add_u64(p: *mut u64, v: u64) {
    core::arch::asm!("red.global.add.u64 [{p}], {v};", p = in(reg64) p, v = in(reg64) v, options(nostack));
}

/// dst[i] += src[i] over i64 fixed-point accumulators (the second card's partial sums), src re-zeroed. Grid ceil(n_total / 256).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_fx_add(dst: *mut i64, src: *mut i64, n_total: u32) {
    let i = bid_x() * bdim_x() + tid_x();
    if i < n_total {
        *dst.add(i as usize) += *src.add(i as usize);
        *src.add(i as usize) = 0;
    }
}

/// i64 fixed-point accumulator -> f32 (out (= or +=) acc * 2^-40), then acc = 0. Grid ceil(n_total / 256).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_fx_finish(out: *mut f32, acc: *mut i64, n_total: u32, add: u32) {
    let i = bid_x() * bdim_x() + tid_x();
    if i < n_total {
        let a = *acc.add(i as usize);
        let af: f32;
        core::arch::asm!("cvt.rn.f32.s64 {f}, {a};", f = out(reg32) af, a = in(reg64) a, options(pure, nomem, nostack));
        let v = af * (1.0 / (1u64 << FX_SHIFT) as f32);
        let o = out.add(i as usize);
        if add != 0 { *o += v } else { *o = v }
        *acc.add(i as usize) = 0;
    }
}

#[inline(always)]
fn roundf(v: f32) -> f32 {
    let a = v.abs();
    let mut r = (a + 0.5) as i32 as f32;
    if a < 0.5 {
        r = 0.0;
    }
    if v < 0.0 { -r } else { r }
}
#[inline(always)]
fn f16_round(x: f32) -> f32 {
    let h: u16;
    unsafe { core::arch::asm!("cvt.rn.f16.f32 {h}, {x};", h = out(reg16) h, x = in(reg32) x, options(pure, nomem, nostack)) }
    crate::dec::f16_to_f32(h)
}

/// Named barriers with compile-time ids: a runtime id makes ptxas reserve all 16 barriers for the CTA (1 CTA per SM).
#[inline(always)]
fn bar_sync_c<const ID: u32>(n: u32) {
    unsafe { core::arch::asm!("bar.sync {i}, {n};", i = const ID, n = in(reg32) n) }
}
#[inline(always)]
fn bar_arrive_c<const ID: u32>(n: u32) {
    unsafe { core::arch::asm!("bar.arrive {i}, {n};", i = const ID, n = in(reg32) n, options(nostack)) }
}
#[inline(always)]
fn bar_arrive(id: u32, n: u32) {
    unsafe { core::arch::asm!("bar.arrive {i}, {n};", i = in(reg32) id, n = in(reg32) n, options(nostack)) }
}

macro_rules! mmq_kernel {
    ($v:ty, $name:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(y: *mut f32, ldy: u32, aq: *const i8, ad: *const f32, as_: *const f32,
            w: *const u8, row_bytes: u64, m: u32, k: u32, xidx: *const u32, yidx: *const u32, segw: *const u64,
            seg_off: *const u32, tiles: *const u32, ntiles: *const u32, add: u32, segw2: *const u64, hq: *mut i8,
            wts: *const f32, lim: f32) {
            mmq::<$v>(y, ldy, aq, ad, as_, w, row_bytes, m, k, xidx, yidx, segw, seg_off, tiles, ntiles, add, segw2, hq, wts, lim)
        }
    };
}
mmq_kernel!(Q8_0, ggq_mmq_q8_0);
mmq_kernel!(Q2_0, ggq_mmq_q2_0);
mmq_kernel!(Iq4Nl, ggq_mmq_iq4_nl);
mmq_kernel!(Iq4Xs, ggq_mmq_iq4_xs);
mmq_kernel!(Q4K, ggq_mmq_q4_k);
mmq_kernel!(Q5K, ggq_mmq_q5_k);
mmq_kernel!(Q6K, ggq_mmq_q6_k);
mmq_kernel!(Iq2S, ggq_mmq_iq2_s);
mmq_kernel!(Iq3Xxs, ggq_mmq_iq3_xxs);
mmq_kernel!(Iq3S, ggq_mmq_iq3_s);
mmq_kernel!(Iq2Xxs, ggq_mmq_iq2_xxs);
mmq_kernel!(Iq1M, ggq_mmq_iq1_m);
mmq_kernel!(Iq2Xs, ggq_mmq_iq2_xs);
mmq_kernel!(Mxfp4R, ggq_mmq_mxfp4r);
macro_rules! mmq_kernel_bm {
    ($v:ty, $name:ident, $bm:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(y: *mut f32, ldy: u32, aq: *const i8, ad: *const f32, as_: *const f32,
            w: *const u8, row_bytes: u64, m: u32, k: u32, xidx: *const u32, yidx: *const u32, segw: *const u64,
            seg_off: *const u32, tiles: *const u32, ntiles: *const u32, add: u32, segw2: *const u64, hq: *mut i8,
            wts: *const f32, lim: f32) {
            mmq_bm::<$v, $bm>(y, ldy, aq, ad, as_, w, row_bytes, m, k, xidx, yidx, segw, seg_off, tiles, ntiles, add, segw2, hq, wts, lim)
        }
    };
}
mmq_kernel_bm!(Q8_0, ggq_mmq_q8_0_bm64, 64);
mmq_kernel_bm!(Q8_0, ggq_mmq_q8_0_bm32, 32);
mmq_kernel_bm!(Q2_0, ggq_mmq_q2_0_bm64, 64);
mmq_kernel_bm!(Q2_0, ggq_mmq_q2_0_bm32, 32);
mmq_kernel_bm!(Iq4Nl, ggq_mmq_iq4_nl_bm64, 64);
mmq_kernel_bm!(Iq4Nl, ggq_mmq_iq4_nl_bm32, 32);
mmq_kernel_bm!(Iq4Xs, ggq_mmq_iq4_xs_bm64, 64);
mmq_kernel_bm!(Iq4Xs, ggq_mmq_iq4_xs_bm32, 32);
mmq_kernel_bm!(Q4K, ggq_mmq_q4_k_bm64, 64);
mmq_kernel_bm!(Q4K, ggq_mmq_q4_k_bm32, 32);
mmq_kernel_bm!(Q5K, ggq_mmq_q5_k_bm64, 64);
mmq_kernel_bm!(Q5K, ggq_mmq_q5_k_bm32, 32);
mmq_kernel_bm!(Q6K, ggq_mmq_q6_k_bm64, 64);
mmq_kernel_bm!(Q6K, ggq_mmq_q6_k_bm32, 32);
mmq_kernel_bm!(Iq2S, ggq_mmq_iq2_s_bm64, 64);
mmq_kernel_bm!(Iq2S, ggq_mmq_iq2_s_bm32, 32);
mmq_kernel_bm!(Iq3Xxs, ggq_mmq_iq3_xxs_bm64, 64);
mmq_kernel_bm!(Iq3Xxs, ggq_mmq_iq3_xxs_bm32, 32);
mmq_kernel_bm!(Iq3S, ggq_mmq_iq3_s_bm64, 64);
mmq_kernel_bm!(Iq3S, ggq_mmq_iq3_s_bm32, 32);
mmq_kernel_bm!(Iq2Xxs, ggq_mmq_iq2_xxs_bm64, 64);
mmq_kernel_bm!(Iq2Xxs, ggq_mmq_iq2_xxs_bm32, 32);
mmq_kernel_bm!(Iq1M, ggq_mmq_iq1_m_bm64, 64);
mmq_kernel_bm!(Iq1M, ggq_mmq_iq1_m_bm32, 32);
mmq_kernel_bm!(Iq2Xs, ggq_mmq_iq2_xs_bm64, 64);
mmq_kernel_bm!(Iq2Xs, ggq_mmq_iq2_xs_bm32, 32);
mmq_kernel_bm!(Mxfp4R, ggq_mmq_mxfp4r_bm64, 64);
mmq_kernel_bm!(Mxfp4R, ggq_mmq_mxfp4r_bm32, 32);
