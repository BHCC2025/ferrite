//! Grouped-expert GEMVs for MoE decode (1..8 token rows x top-k experts), weights addressed through per-layer slot
//! tables (`gtab/utab/dtab[expert]` = device address of that expert's [rows x cols] slice, 0 = not on the card).
//!
//! Pairs: `p = token * topk + k`, `ids[p]` = expert. Every block of pair `p` first finds all pairs with the same
//! expert; only the lowest one (the leader) runs, for all tokens that picked the expert, so each expert's weights
//! are read once per launch however many tokens share it.
//!
//! - `ggq_moe_gu_<ty>`: h[p][r] = silu(gate_e[r] . x[tok]) * (up_e[r] . x[tok]); gate and up rows share every
//!   activation load. A block owns 32 consecutive rows (`lpr` warps), so it also quantizes its h block to q8
//!   (`hq/hd/hs`, the down GEMV's activations) in place; optional f32 `h` too.
//! - `ggq_moe_down_<ty>`: y[p][r] = down_e[r] . h[p], f32 [P][n_embd].
//! - `ggq_moe_misses`: the pairs whose expert is not on the card (table entry 0), in pair order, and each pair's
//!   slot in that list (-1 = on the card): one small D2H read tells the host what to run on the CPU.
//! - `ggq_moe_combine`: out[t][n] (= or +=) sum_k w[t][k] * y_or_cpu[t * topk + k][n], fixed order (deterministic);
//!   missed pairs read their row from the CPU buffer at their slot.
use crate::gemv::{reduce_lanes, row_dot_r};
use crate::vq::*;
use kcommon::smem::Shared;
use kcommon::*;

/// Pairs sharing pair `p`'s expert: (is_leader, count, their tokens, their pair indices). Warp-uniform.
#[inline(always)]
unsafe fn group(ids: *const i32, p: u32, np: u32, topk: u32) -> (bool, u32, [u32; 8], [u32; 8]) {
    let e = *ids.add(p as usize);
    let mut cnt = 0u32;
    let mut tok = [0u32; 8];
    let mut prs = [0u32; 8];
    let mut leader = true;
    let mut base = 0u32;
    while base < np {
        let q = base + lane();
        let hit = q < np && *ids.add(q as usize) == e;
        let mut m = ballot(hit);
        while m != 0 {
            let bit = m.trailing_zeros();
            m &= m - 1;
            let q = base + bit;
            if q < p {
                leader = false;
            }
            // constant indices only (no local-memory array)
            unroll!(i in 8 => {
                if cnt == i as u32 {
                    tok[i] = q / topk;
                    prs[i] = q;
                }
            });
            cnt += 1;
        }
        base += 32;
    }
    (leader, cnt.min(8), tok, prs)
}

#[inline(always)]
pub fn ballot(p: bool) -> u32 {
    let r: u32;
    unsafe {
        core::arch::asm!("{{ .reg .pred q; setp.ne.u32 q, {v}, 0; vote.sync.ballot.b32 {r}, q, 0xffffffff; }}",
            r = out(reg32) r, v = in(reg32) p as u32, options(nostack));
    }
    r
}

/// Up to this many tokens, moe_gu runs gate and up as separate row passes.
const GU_SPLIT_MAX_B: usize = 0;

/// DeepSeek's clamped SwiGLU: silu(min(g, lim)) * clamp(u, -lim, lim); `lim` = +inf leaves g and u (and the bits) untouched.
#[inline(always)]
pub fn swiglu(g: f32, u: f32, lim: f32) -> f32 {
    let g = if g > lim { lim } else { g };
    let u = if u > lim { lim } else if u < -lim { -lim } else { u };
    silu(g) * u
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    // deterministic exp: the CPU miss path (fe-ggq cpu) computes the same bits
    kcommon::math::silu_det(x)
}

/// gate (row 0) and up (row 1) of one h row for B tokens; returns silu(g) * u per token (lane-reduced).
#[inline(always)]
unsafe fn gu_body<V: Vq, const B: usize>(row_g: *const u8, row_u: *const u8, nsb: u32, lpr: u32, aq: *const i8,
    ad: *const f32, as_: *const f32, k: u32, tok: &[u32; 8], valid: bool, lim: f32) -> [f32; 8] {
    let ar: [u32; B] = core::array::from_fn(|b| tok[b]);
    let mut acc = [[0f32; B]; 2];
    if valid {
        if B <= GU_SPLIT_MAX_B {
            // few tokens: gate and up one after the other (fewer registers, better occupancy)
            let mut a0 = [[0f32; B]; 1];
            let mut a1 = [[0f32; B]; 1];
            row_dot_r::<V, B, 1>(&[row_g], lane() % lpr, lpr, nsb, aq, ad, as_, k, &ar, &mut a0);
            row_dot_r::<V, B, 1>(&[row_u], lane() % lpr, lpr, nsb, aq, ad, as_, k, &ar, &mut a1);
            acc = [a0[0], a1[0]];
        } else {
            row_dot_r::<V, B, 2>(&[row_g, row_u], lane() % lpr, lpr, nsb, aq, ad, as_, k, &ar, &mut acc);
        }
    }
    let mut h = [0f32; 8];
    unroll!(b in 8 => {
        if B > b {
            let g = reduce_lanes(*acc[0].get_unchecked(b), lpr);
            let u = reduce_lanes(*acc[1].get_unchecked(b), lpr);
            h[b] = swiglu(g, u, lim);
        }
    });
    h
}

#[inline(always)]
fn f16_round(x: f32) -> f32 {
    let h: u16;
    unsafe { core::arch::asm!("cvt.rn.f16.f32 {h}, {x};", h = out(reg16) h, x = in(reg32) x, options(pure, nomem, nostack)) }
    crate::dec::f16_to_f32(h)
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

/// Fused gate/up/SwiGLU/q8. Block = `lpr` warps = 32 rows of h (row = bid_x * 32 + ...). n_ff % 32 == 0.
#[inline(always)]
pub unsafe fn moe_gu<V: Vq>(hq: *mut i8, hd: *mut f32, hs: *mut f32, hf: *mut f32, gtab: *const u64,
    utab: *const u64, row_bytes: u64, ids: *const i32, np: u32, topk: u32, aq: *const i8, ad: *const f32,
    as_: *const f32, k: u32, nff: u32, lpr: u32, sum_mode: u32, lim: f32) {
    let p = bid_y();
    let (leader, cnt, tok, prs) = group(ids, p, np, topk);
    if !leader {
        return;
    }
    let e = *ids.add(p as usize) as usize;
    let gw = *gtab.add(e);
    let uw = *utab.add(e);
    if gw == 0 || uw == 0 {
        return;
    }
    let rpw = 32 / lpr;
    let rib = warp_id() * rpw + lane() / lpr; // row in block, 0..32
    let row = bid_x() * 32 + rib;
    let valid = row < nff;
    let off = row as usize * row_bytes as usize;
    let (rg, ru) = ((gw as usize + off) as *const u8, (uw as usize + off) as *const u8);
    let nsb = k / 32;
    let h = match cnt {
        1 => gu_body::<V, 1>(rg, ru, nsb, lpr, aq, ad, as_, k, &tok, valid, lim),
        2 => gu_body::<V, 2>(rg, ru, nsb, lpr, aq, ad, as_, k, &tok, valid, lim),
        3 => gu_body::<V, 3>(rg, ru, nsb, lpr, aq, ad, as_, k, &tok, valid, lim),
        4 => gu_body::<V, 4>(rg, ru, nsb, lpr, aq, ad, as_, k, &tok, valid, lim),
        5 => gu_body::<V, 5>(rg, ru, nsb, lpr, aq, ad, as_, k, &tok, valid, lim),
        6 => gu_body::<V, 6>(rg, ru, nsb, lpr, aq, ad, as_, k, &tok, valid, lim),
        7 => gu_body::<V, 7>(rg, ru, nsb, lpr, aq, ad, as_, k, &tok, valid, lim),
        _ => gu_body::<V, 8>(rg, ru, nsb, lpr, aq, ad, as_, k, &tok, valid, lim),
    };
    let hsm: Shared<f32> = static_shared!(ggq_moe_hs, f32, 8 * 32 * 4);
    if lane() % lpr == 0 {
        unroll!(b in 8 => {
            if (b as u32) < cnt {
                hsm.add(b as u32 * 32 + rib).st(h[b]);
                if !hf.is_null() && valid {
                    *hf.add(prs[b] as usize * nff as usize + row as usize) = h[b];
                }
            }
        });
    }
    if hq.is_null() {
        return;
    }
    sync_threads();
    // warp w quantizes tokens w, w + nwarps, ... (one q8 block of h each)
    let mut w = warp_id();
    while w < cnt && bid_x() * 32 < nff {
        let mut pr = 0u32;
        unroll!(b in 8 => {
            if w == b as u32 {
                pr = prs[b];
            }
        });
        let v = hsm.add(w * 32 + lane()).ld();
        let amax = warp::max(v.abs());
        let d = amax / 127.0;
        let q = if amax == 0.0 { 0i32 } else { roundf(v / d) as i32 };
        *hq.add(pr as usize * nff as usize + (bid_x() * 32 + lane()) as usize) = q as i8;
        let sx = match sum_mode {
            1 => warp::sum(v),
            2 => d * warp::sum(q as f32),
            _ => f16_round(warp::sum(v)),
        };
        if lane() == 0 {
            let nb = (nff / 32) as usize;
            *hd.add(pr as usize * nb + bid_x() as usize) = d;
            *hs.add(pr as usize * nb + bid_x() as usize) = sx;
        }
        w += nwarps();
    }
}

#[inline(always)]
unsafe fn down_body<V: Vq, const B: usize, const R: usize>(rows: &[*const u8; R], nsb: u32, lpr: u32, aq: *const i8,
    ad: *const f32, as_: *const f32, k: u32, prs: &[u32; 8], valid: bool) -> [[f32; 8]; R] {
    let ar: [u32; B] = core::array::from_fn(|b| prs[b]);
    let mut acc = [[0f32; B]; R];
    if valid {
        row_dot_r::<V, B, R>(rows, lane() % lpr, lpr, nsb, aq, ad, as_, k, &ar, &mut acc);
    }
    let mut o = [[0f32; 8]; R];
    unroll!(r in 4 => {
        if R > r {
            unroll!(b in 8 => {
                if B > b {
                    *o.get_unchecked_mut(r).get_unchecked_mut(b) = reduce_lanes(*acc.get_unchecked(r).get_unchecked(b), lpr);
                }
            });
        }
    });
    o
}

/// Down GEMV for the pairs; R output rows per lane group (a warp covers R * 32 / lpr rows).
#[inline(always)]
pub unsafe fn moe_down<V: Vq, const R: usize>(y: *mut f32, dtab: *const u64, row_bytes: u64, ids: *const i32, np: u32,
    topk: u32, aq: *const i8, ad: *const f32, as_: *const f32, k: u32, nout: u32, lpr: u32) {
    let p = bid_y();
    let (leader, cnt, _tok, prs) = group(ids, p, np, topk);
    if !leader {
        return;
    }
    let e = *ids.add(p as usize) as usize;
    let dw = *dtab.add(e);
    if dw == 0 {
        return;
    }
    let rpw = 32 / lpr;
    let row0 = ((bid_x() * nwarps() + warp_id()) * rpw + lane() / lpr) * R as u32;
    let valid = row0 < nout;
    let rows: [*const u8; R] =
        core::array::from_fn(|r| (dw as usize + (row0 + r as u32).min(nout - 1) as usize * row_bytes as usize) as *const u8);
    let nsb = k / 32;
    let o = match cnt {
        1 => down_body::<V, 1, R>(&rows, nsb, lpr, aq, ad, as_, k, &prs, valid),
        2 => down_body::<V, 2, R>(&rows, nsb, lpr, aq, ad, as_, k, &prs, valid),
        3 => down_body::<V, 3, R>(&rows, nsb, lpr, aq, ad, as_, k, &prs, valid),
        4 => down_body::<V, 4, R>(&rows, nsb, lpr, aq, ad, as_, k, &prs, valid),
        5 => down_body::<V, 5, R>(&rows, nsb, lpr, aq, ad, as_, k, &prs, valid),
        6 => down_body::<V, 6, R>(&rows, nsb, lpr, aq, ad, as_, k, &prs, valid),
        7 => down_body::<V, 7, R>(&rows, nsb, lpr, aq, ad, as_, k, &prs, valid),
        _ => down_body::<V, 8, R>(&rows, nsb, lpr, aq, ad, as_, k, &prs, valid),
    };
    if valid && lane() % lpr == 0 {
        unroll!(r in 4 => {
            let row = row0 + r as u32;
            if R > r && row < nout {
                unroll!(b in 8 => {
                    if (b as u32) < cnt {
                        *y.add(prs[b] as usize * nout as usize + row as usize) = *o.get_unchecked(r).get_unchecked(b);
                    }
                });
            }
        });
    }
}

macro_rules! moe_kernels {
    ($v:ty, $gu:ident, $down:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $gu(hq: *mut i8, hd: *mut f32, hs: *mut f32, hf: *mut f32, gtab: *const u64,
            utab: *const u64, row_bytes: u64, ids: *const i32, np: u32, topk: u32, aq: *const i8, ad: *const f32,
            as_: *const f32, k: u32, nff: u32, lpr: u32, sum_mode: u32, lim: f32) {
            moe_gu::<$v>(hq, hd, hs, hf, gtab, utab, row_bytes, ids, np, topk, aq, ad, as_, k, nff, lpr, sum_mode, lim)
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $down(y: *mut f32, dtab: *const u64, row_bytes: u64, ids: *const i32,
            np: u32, topk: u32, aq: *const i8, ad: *const f32, as_: *const f32, k: u32, nout: u32, lpr: u32, rpl: u32) {
            match rpl {
                4 => moe_down::<$v, 4>(y, dtab, row_bytes, ids, np, topk, aq, ad, as_, k, nout, lpr),
                2 => moe_down::<$v, 2>(y, dtab, row_bytes, ids, np, topk, aq, ad, as_, k, nout, lpr),
                _ => moe_down::<$v, 1>(y, dtab, row_bytes, ids, np, topk, aq, ad, as_, k, nout, lpr),
            }
        }
    };
}
moe_kernels!(Q8_0, ggq_moe_gu_q8_0, ggq_moe_down_q8_0);
moe_kernels!(Q2_0, ggq_moe_gu_q2_0, ggq_moe_down_q2_0);
moe_kernels!(Iq4Nl, ggq_moe_gu_iq4_nl, ggq_moe_down_iq4_nl);
moe_kernels!(Iq4Xs, ggq_moe_gu_iq4_xs, ggq_moe_down_iq4_xs);
moe_kernels!(Q4K, ggq_moe_gu_q4_k, ggq_moe_down_q4_k);
moe_kernels!(Q5K, ggq_moe_gu_q5_k, ggq_moe_down_q5_k);
moe_kernels!(Q6K, ggq_moe_gu_q6_k, ggq_moe_down_q6_k);
moe_kernels!(Iq2S, ggq_moe_gu_iq2_s, ggq_moe_down_iq2_s);
moe_kernels!(Iq3Xxs, ggq_moe_gu_iq3_xxs, ggq_moe_down_iq3_xxs);
moe_kernels!(Iq3S, ggq_moe_gu_iq3_s, ggq_moe_down_iq3_s);
moe_kernels!(Iq2Xxs, ggq_moe_gu_iq2_xxs, ggq_moe_down_iq2_xxs);
moe_kernels!(Iq1M, ggq_moe_gu_iq1_m, ggq_moe_down_iq1_m);
moe_kernels!(Iq2Xs, ggq_moe_gu_iq2_xs, ggq_moe_down_iq2_xs);
moe_kernels!(Mxfp4R, ggq_moe_gu_mxfp4r, ggq_moe_down_mxfp4r);

/// Pairs whose expert has table entry 0. One warp: `count[0]` = n missed, `pairs[0..n]` = their pair indices in
/// order, `slot[p]` = rank of p in that list or -1. Grid 1 x 32 threads.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_moe_misses(count: *mut u32, pairs: *mut u32, slot: *mut i32, tab: *const u64,
    ids: *const i32, np: u32) {
    let mut n = 0u32;
    let mut base = 0u32;
    while base < np {
        let q = base + lane();
        let miss = q < np && *tab.add(*ids.add(q as usize) as usize) == 0;
        let m = ballot(miss);
        let rank = n + (m & ((1u32 << lane()) - 1)).count_ones();
        if q < np {
            *slot.add(q as usize) = if miss { rank as i32 } else { -1 };
            if miss {
                *pairs.add(rank as usize) = q;
            }
        }
        n += m.count_ones();
        base += 32;
    }
    if lane() == 0 {
        *count = n;
    }
}

/// out[t][i] (= or +=) sum_k w[t * topk + k] * row(t * topk + k)[i], row(p) = ycpu[slot[p]] when `slot` is non-null
/// and slot[p] >= 0, else y[p]. Grid: x = ceil(n / 256), y = token.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_moe_combine(out: *mut f32, y: *const f32, w: *const f32, topk: u32, n: u32,
    add: u32, ycpu: *const f32, slot: *const i32) {
    let i = bid_x() * bdim_x() + tid_x();
    if i >= n {
        return;
    }
    let t = bid_y() as usize;
    let mut acc = 0f32;
    for kk in 0..topk as usize {
        let p = t * topk as usize + kk;
        let sl = if slot.is_null() { -1 } else { *slot.add(p) };
        let v = if sl >= 0 { *ycpu.add(sl as usize * n as usize + i as usize) } else { *y.add(p * n as usize + i as usize) };
        acc += *w.add(p) * v;
    }
    let o = out.add(t * n as usize + i as usize);
    if add != 0 { *o += acc } else { *o = acc }
}

/// dst[e] = src[e] + off where src[e] != 0 (an expert table shifted to a block of rows). grid ceil(n / 256), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_tab_offset(dst: *mut u64, src: *const u64, n: u32, off: u64) {
    let i = bid_x() * bdim_x() + tid_x();
    if i < n {
        let v = *src.add(i as usize);
        *dst.add(i as usize) = if v == 0 { 0 } else { v + off };
    }
}

/// `ggq_moe_combine` with strides: out[t * ldo + i] (= or +=) sum_k w[t][k] * y[(t * topk + k) * ldy + i], i < n,
/// fixed slot order (the same arithmetic as ggq_moe_combine). grid (ceil(n / 256), n_tok), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_moe_combine_ld(out: *mut f32, ldo: u32, y: *const f32, ldy: u32, w: *const f32,
    topk: u32, n: u32, add: u32) {
    let i = bid_x() * bdim_x() + tid_x();
    if i >= n {
        return;
    }
    let t = bid_y() as usize;
    let mut acc = 0f32;
    for kk in 0..topk as usize {
        let p = t * topk as usize + kk;
        acc += *w.add(p) * *y.add(p * ldy as usize + i as usize);
    }
    let o = out.add(t * ldo as usize + i as usize);
    if add != 0 { *o += acc } else { *o = acc }
}

/// Device-side pair routing for the zero-host layer chain: `slot[q]` = -1 when the pair's expert belongs to this (the main) card
/// (`owner` 0), else the row `(owner - 1) * stride + q` of the expert cards' result buffer. One warp.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_slot_owner(slot: *mut i32, owner: *const u8, ids: *const i32, np: u32, stride: u32) {
    let mut q = lane();
    while q < np {
        let o = *owner.add(*ids.add(q as usize) as usize) as u32;
        *slot.add(q as usize) = if o == 0 { -1 } else { ((o - 1) * stride + q) as i32 };
        q += 32;
    }
}

/// Cold-expert staging, step 1: for every pair whose expert lives in pinned host memory (table pointer in [lo, hi)), claim a staging slot, remember
/// the host pointers and point the tables at the slot (the copy kernel fills it before the MoE kernels read). One block per pair, thread 0 works.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_cold_claim(ids: *const i32, np: u32, gtab: *mut u64, utab: *mut u64, dtab: *mut u64, lo: u64, hi: u64,
    counter: *mut u32, slot_e: *mut u32, slot_orig: *mut u64, stage: *mut u8, slot_bytes: u64, nslots: u32, szg: u64, szu: u64) {
    if lane() != 0 {
        return;
    }
    let p = bid_x();
    if p >= np {
        return;
    }
    let e = *ids.add(p as usize);
    let mut q = 0u32;
    while q < p {
        if *ids.add(q as usize) == e {
            return;
        }
        q += 1;
    }
    let eu = e as usize;
    let (gw, uw, dw) = (*gtab.add(eu), *utab.add(eu), *dtab.add(eu));
    if gw >= lo && gw < hi && uw >= lo && uw < hi && dw >= lo && dw < hi {
        let slot = atomic_add_u32(counter, 1);
        if slot < nslots {
            *slot_e.add(slot as usize) = e as u32;
            *slot_orig.add(3 * slot as usize) = gw;
            *slot_orig.add(3 * slot as usize + 1) = uw;
            *slot_orig.add(3 * slot as usize + 2) = dw;
            let base = stage as u64 + slot as u64 * slot_bytes;
            *gtab.add(eu) = base;
            *utab.add(eu) = base + szg;
            *dtab.add(eu) = base + szg + szu;
        }
    }
}

/// Step 2: copy the claimed experts from pinned host memory into their slots with many 16-byte loads in flight (the link, not the MoE kernels'
/// few loads, sets the speed). Grid (nslots, chunks), 256 threads.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_cold_copy(counter: *const u32, slot_orig: *const u64, stage: *mut u8, slot_bytes: u64, nslots: u32, szg: u64, szu: u64, szd: u64, chunks: u32) {
    let slot = bid_x();
    let n = (*counter).min(nslots);
    if slot >= n {
        return;
    }
    let total = szg + szu + szd;
    let per = (total / 16).div_ceil(chunks as u64);
    let start = bid_y() as u64 * per;
    let end = (start + per).min(total / 16);
    let dst = stage.add((slot as u64 * slot_bytes) as usize);
    let (g, u, d) = (*slot_orig.add(3 * slot as usize), *slot_orig.add(3 * slot as usize + 1), *slot_orig.add(3 * slot as usize + 2));
    let mut i = start + tid_x() as u64;
    let step = 256u64;
    while i < end {
        let off = i * 16;
        let src = if off < szg { g + off } else if off < szg + szu { u + (off - szg) } else { d + (off - szg - szu) };
        let v = ldg_stream_v4(src as *const u32);
        *(dst.add(off as usize) as *mut [u32; 4]) = v;
        i += step;
    }
}

/// Step 3 (after the MoE kernels): the tables point at the host copies again. One block per slot.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_cold_restore(counter: *const u32, slot_e: *const u32, slot_orig: *const u64, gtab: *mut u64, utab: *mut u64, dtab: *mut u64, nslots: u32) {
    let slot = bid_x();
    let n = (*counter).min(nslots);
    if slot >= n || tid_x() != 0 {
        return;
    }
    let e = *slot_e.add(slot as usize) as usize;
    *gtab.add(e) = *slot_orig.add(3 * slot as usize);
    *utab.add(e) = *slot_orig.add(3 * slot as usize + 1);
    *dtab.add(e) = *slot_orig.add(3 * slot as usize + 2);
}
