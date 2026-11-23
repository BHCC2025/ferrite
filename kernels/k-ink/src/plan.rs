//! MoE routing plan for the grouped GEMMs: (row, pick) pairs sorted (stably) by expert into 128-row-aligned groups,
//! a tile table for the grid, and the permutations between pair index and grouped position.
use super::*;

pub const NONE: u32 = 0xffff_ffff;

/// counts[e] = number of pairs routed to expert e. grid = 128 (one block per expert), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_plan_count(counts: *mut u32, ids: *const i32, npairs: u32, use_cnt: *mut u32) {
    pdl_wait();
    pdl_trigger();
    let scratch = static_shared!(plan_cnt_red, f32, 128);
    let e = bid_x() as i32;
    let mut c = 0.0f32;
    let mut i = tid_x();
    while i < npairs {
        if *ids.add(i as usize) == e {
            c += 1.0;
        }
        i += 256;
    }
    let t = warp::block_sum(c, scratch);
    if tid_x() == 0 {
        *counts.add(e as usize) = t as u32;
        if !use_cnt.is_null() {
            *use_cnt.add(e as usize) += t as u32;
        }
    }
}

/// Group offsets (each group padded to a multiple of 128 rows) and the tile table: tab[0] = number of tiles,
/// tab[1 + 3t ..] = (expert, first grouped row, valid rows). goff[128] = total padded rows. 1 block, 128 threads.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_plan_build(tab: *mut u32, goff: *mut u32, counts: *const u32) {
    pdl_wait();
    pdl_trigger();
    if tid_x() != 0 {
        return;
    }
    let mut off = 0u32;
    let mut nt = 0u32;
    let mut e = 0u32;
    while e < 128 {
        let c = *counts.add(e as usize);
        *goff.add(e as usize) = off;
        let t = c.div_ceil(128);
        let mut i = 0;
        while i < t {
            let p = tab.add(1 + 3 * (nt + i) as usize);
            *p = e;
            *p.add(1) = off + i * 128;
            *p.add(2) = if c - i * 128 < 128 { c - i * 128 } else { 128 };
            i += 1;
        }
        nt += t;
        off += t * 128;
        e += 1;
    }
    *tab = nt;
    *goff.add(128) = off;
}

/// Stable placement: pos_of_pair[p] = grouped row of pair p (NONE if its id < 0), pair_of_pos[pos] = p. The caller
/// fills pair_of_pos with NONE first (padding rows). grid = 128, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_plan_perm(pair_of_pos: *mut u32, pos_of_pair: *mut u32, ids: *const i32,
    goff: *const u32, npairs: u32) {
    pdl_wait();
    pdl_trigger();
    let wtot = static_shared!(plan_perm_w, u32, 128);
    let e = bid_x() as i32;
    let base = *goff.add(e as usize);
    let l = lane();
    let w = warp_id();
    let mut running = 0u32;
    let mut c = 0u32;
    while c * 256 < npairs {
        let i = c * 256 + tid_x();
        let id = if i < npairs { *ids.add(i as usize) } else { -2 };
        if e == 0 && id < 0 && i < npairs {
            *pos_of_pair.add(i as usize) = NONE;
        }
        let flag = if id == e { 1u32 } else { 0 };
        // inclusive warp scan
        let mut v = flag;
        let mut d = 1;
        while d < 32 {
            let u = warp::shfl_up_u32(v, d);
            if l >= d {
                v += u;
            }
            d <<= 1;
        }
        if l == 31 {
            wtot.set(w, v);
        }
        sync_threads();
        let mut woff = 0u32;
        let mut q = 0;
        while q < w {
            woff += wtot.at(q);
            q += 1;
        }
        let mut blk = 0u32;
        q = 0;
        while q < 8 {
            blk += wtot.at(q);
            q += 1;
        }
        if flag == 1 {
            let pos = base + running + woff + v - 1;
            *pair_of_pos.add(pos as usize) = i;
            *pos_of_pair.add(i as usize) = pos;
        }
        running += blk;
        sync_threads();
        c += 1;
    }
}

/// The whole routing plan in one block for up to 1024 (row, pick) pairs (decode and short chunks): same outputs as
/// `ink_plan_count` + `ink_plan_build` + `ink_plan_perm` (stable order inside an expert, groups padded to 128 rows).
/// 1 block, 1024 threads.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_plan_small(tab: *mut u32, goff: *mut u32, pair_of_pos: *mut u32,
    pos_of_pair: *mut u32, ids: *const i32, npairs: u32, use_cnt: *mut u32) {
    pdl_wait();
    pdl_trigger();
    let sid = static_shared!(plan_small_ids, u32, 4096); // 1024 ids
    let scnt = static_shared!(plan_small_cnt, u32, 512); // 128 counts
    let sgo = static_shared!(plan_small_goff, u32, 512);
    let t = tid_x();
    let id = if t < npairs { *ids.add(t as usize) } else { 0x7fff_ffff };
    sid.set(t, id as u32);
    sync_threads();
    if t < 128 {
        let mut c = 0u32;
        let mut q = 0;
        while q < npairs {
            if sid.at(q) == t {
                c += 1;
            }
            q += 1;
        }
        scnt.set(t, c);
        if !use_cnt.is_null() {
            *use_cnt.add(t as usize) += c;
        }
    }
    sync_threads();
    if t == 0 {
        let mut off = 0u32;
        let mut nt = 0u32;
        let mut e = 0u32;
        while e < 128 {
            let c = scnt.at(e);
            sgo.set(e, off);
            *goff.add(e as usize) = off;
            let tl = c.div_ceil(128);
            let mut i = 0;
            while i < tl {
                let p = tab.add(1 + 3 * (nt + i) as usize);
                *p = e;
                *p.add(1) = off + i * 128;
                *p.add(2) = if c - i * 128 < 128 { c - i * 128 } else { 128 };
                i += 1;
            }
            nt += tl;
            off += tl * 128;
            e += 1;
        }
        *tab = nt;
        *goff.add(128) = off;
    }
    sync_threads();
    if t < npairs {
        if (id as i32) < 0 {
            *pos_of_pair.add(t as usize) = NONE;
        } else {
            let mut rank = 0u32;
            let mut q = 0;
            while q < t {
                if sid.at(q) == id as u32 {
                    rank += 1;
                }
                q += 1;
            }
            let pos = sgo.at(id as u32) + rank;
            *pos_of_pair.add(t as usize) = pos;
            *pair_of_pos.add(pos as usize) = t;
        }
    }
}

/// out[row][c] = bf16(sum over the row's picks of f32(y[pos_of_pair[row * 8 + k]][c]) [* wts[row * 8 + k]]), f32 from 0.
/// flags bit 0: sum in slot order k = 0..7 (the native engine), else ascending expert id; bit 1: multiply by wts.
/// The grouped-position twin of `ink_moe_combine`. grid = (ceil(2304 / 256), rows), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_moe_combine_pos(out: *mut u16, ldo: u32, y: *const u16, pos: *const u32,
    ids: *const i32, wts: *const f32, flags: u32) {
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
        id[k] = *ids.add(row * 8 + k) as u32;
        let p = *pos.add(row * 8 + k);
        yv[k] = if p == NONE { 0.0 } else { bf16::to_f32(*y.add(p as usize * 2304 + c)) };
        wv[k] = if flags & 2 != 0 { *wts.add(row * 8 + k) } else { 1.0 };
    });
    let mut rank = [0u32; 8];
    kcommon::unroll!(k in 8 => {
        let mut r = 0u32;
        kcommon::unroll!(j in 8 => {
            if id[j] < id[k] || (id[j] == id[k] && j < k) {
                r += 1;
            }
        });
        rank[k] = if flags & 1 != 0 { k as u32 } else { r };
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

/// Decode tail of a layer application, in one launch: the 8 slots of each row summed in slot order (as `ink_moe_combine_pos`,
/// bf16), then the MLP short convolution (kernel 4, residual, / sqrt 5, 3-row f32 state, rows in order) and the final add
/// out = bf16(resid + conv). Same arithmetic as `ink_sconv` + `ink_final_add` after the combine.
/// grid = ceil(2304 / 256), block = 256; one thread a channel walks the rows.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_moe_tail(out: *mut u16, resid: *const f32, y: *const u16, pos: *const u32, ids: *const i32,
    w: *const u16, state: *mut f32, nt: u32, spec: u32) {
    pdl_wait();
    pdl_trigger();
    let ch = (bid_x() * 256 + tid_x()) as usize;
    if ch >= 2304 {
        return;
    }
    let w0 = bf16::to_f32(*w.add(ch * 4));
    let w1 = bf16::to_f32(*w.add(ch * 4 + 1));
    let w2 = bf16::to_f32(*w.add(ch * 4 + 2));
    let w3 = bf16::to_f32(*w.add(ch * 4 + 3));
    let mut r0 = *state.add(ch);
    let mut r1 = *state.add(2304 + ch);
    let mut r2 = *state.add(2 * 2304 + ch);
    let mut t = 0usize;
    while t < nt as usize {
        let mut acc = 0.0f32;
        kcommon::unroll!(k in 8 => {
            let p = *pos.add(t * 8 + k);
            if (*ids.add(t * 8 + k)) >= 0 && p != NONE {
                acc += bf16::to_f32(*y.add(p as usize * 2304 + ch));
            }
        });
        let xv = bf16::round(acc);
        if spec != 0 {
            *state.add((3 + t) * 2304 + ch) = xv;
        }
        let cv = math::fma(w3, xv, math::fma(w2, r2, math::fma(w1, r1, w0 * r0)));
        let mc = bf16::from_f32((cv + xv) / 2.236068);
        *out.add(t * 2304 + ch) = bf16::from_f32(*resid.add(t * 2304 + ch) + bf16::to_f32(mc));
        r0 = r1;
        r1 = r2;
        r2 = xv;
        t += 1;
    }
    if spec == 0 {
        *state.add(ch) = r0;
        *state.add(2304 + ch) = r1;
        *state.add(2 * 2304 + ch) = r2;
    }
}
