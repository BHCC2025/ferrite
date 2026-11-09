//! Gated full attention (Qwen3.5): q/k RMSNorm + partial RoPE + KV-cache write, split-K attention over
//! the cache for every row (decode and prefill rows alike), and a combine pass that applies the
//! sigmoid output gate. Head dim 256, 24 q heads, 4 kv heads (GQA group 6), rotary dims 64 (rotate-half).
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

pub const HD: u32 = 256;
pub const NQ: u32 = crate::dims::NQ; // 24 (12 a TP2 rank)
pub const NKV: u32 = crate::dims::NKV; // 4 (2 a TP2 rank)
pub const GRP: u32 = NQ / NKV; // 6
/// grid = rows, block = 1024 (32 warps: 24 q heads, 4 k heads, 4 v heads).
/// Inputs are the projection outputs: `qg` rows of [q0 | gate0 | q1 | gate1 ...] (24 x 512, stride ld_qg),
/// `kx` rows of 4 x 256 (stride ld_k), `vx` rows of 4 x 256 (stride ld_v).
/// q_out[r][h][256]: normed + roped q. K/V caches: [slot][ctx][4][256] (slot stride `slot_stride` elements).
/// rope: [pos][64] bf16 = cos[32] | sin[32]. row_cpos: the cache index each row's K/V goes to.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_attn_prep(q_out: *mut u16, kc: *mut u16, vc: *mut u16, qg: *const u16, ld_qg: u32,
    kx: *const u16, ld_k: u32, vx: *const u16, ld_v: u32, qn_w: *const u16, kn_w: *const u16, rope: *const u16,
    row_slot: *const u32, row_pos: *const u32, row_cpos: *const u32, slot_stride: u64, eps: f32) {
    let r = bid_x() as usize;
    let w = warp_id();
    let l = lane();
    let slot = *row_slot.add(r) as u64;
    // RoPE at the row's sequence position; K/V written at its cache index (they differ for verify-tree rows:
    // position = start + depth, cache index = start + tree row)
    let pos = *row_pos.add(r) as u64;
    let cpos = *row_cpos.add(r) as u64;
    let cache_off = (slot * slot_stride + cpos * (NKV * HD) as u64) as usize;
    if w >= NQ + NKV {
        // v: plain copy into the cache
        let h = w - NQ - NKV;
        let src = vx.add(r * ld_v as usize + (h * HD + l * 8) as usize) as *const u32;
        let dst = vc.add(cache_off + (h * HD + l * 8) as usize) as *mut u32;
        stg_v4(dst, ldg_nc_v4(src));
        return;
    }
    let (src, nw, dst) = if w < NQ {
        (qg.add(r * ld_qg as usize + (w * 2 * HD) as usize), qn_w, q_out.add(r * (NQ * HD) as usize + (w * HD) as usize))
    } else {
        let h = w - NQ;
        (kx.add(r * ld_k as usize + (h * HD) as usize), kn_w, kc.add(cache_off + (h * HD) as usize))
    };
    let x = bf16::unpack8(ldg_nc_v4(src.add((l * 8) as usize) as *const u32));
    let wv = bf16::unpack8(ldg_nc_v4(nw.add((l * 8) as usize) as *const u32));
    let mut ss = 0.0;
    let mut j = 0;
    while j < 8 {
        ss += x[j] * x[j];
        j += 1;
    }
    let inv = math::rsqrt(warp::sum(ss) / HD as f32 + eps);
    let mut y = [0.0f32; 8];
    let mut j = 0;
    while j < 8 {
        y[j] = bf16::round(x[j] * inv * (1.0 + wv[j]));
        j += 1;
    }
    // rotate-half RoPE over dims 0..64: lanes 0..3 hold dims 0..32, lanes 4..7 hold 32..64.
    let mut p = [0.0f32; 8];
    let mut j = 0;
    while j < 8 {
        p[j] = warp::shfl_xor(y[j], 4);
        j += 1;
    }
    if l < 8 {
        let fi = (l & 3) * 8;
        let cs = rope.add((pos * 64) as usize);
        let c = bf16::unpack8(ldg_nc_v4(cs.add(fi as usize) as *const u32));
        let s = bf16::unpack8(ldg_nc_v4(cs.add((32 + fi) as usize) as *const u32));
        let mut j = 0;
        while j < 8 {
            let a = bf16::round(y[j] * c[j]);
            let b = if l < 4 { bf16::round(-p[j] * s[j]) } else { bf16::round(p[j] * s[j]) };
            y[j] = a + b;
            j += 1;
        }
    }
    let o = [bf16::pack2(y[0], y[1]), bf16::pack2(y[2], y[3]), bf16::pack2(y[4], y[5]), bf16::pack2(y[6], y[7])];
    stg_v4(dst.add((l * 8) as usize) as *mut u32, o);
}

/// Splits of a row with `nkeys` keys: about 512 keys a split, 16..=64, a power of two. A row's partition depends
/// only on its own key count, so a verify row (any batch, any tree) sums exactly as 1-row decode at its position.
#[inline(always)]
pub fn row_splits(nkeys: u32) -> u32 {
    let s = nkeys.div_ceil(512).next_power_of_two();
    if s < 16 { 16 } else if s > 64 { 64 } else { s }
}

/// Split-K attention for rows row_base.. over their slot's cache: the row at RoPE position pos sees pos + 1 keys
/// in decode order. A chain row's key v is cache[v]; a tree row (keys from tbase on are its ancestors, tmask over
/// the tree rows) reads its key tbase + d from its depth-d ancestor, so after commit_tree's gather its keys are
/// the same cache rows, in the same order, as 1-row decode at that position. grid = (rows, 4, nsplit), block =
/// 128: a row splits its keys into row_splits(pos + 1) parts (at most the grid's; the rest stay empty), so it
/// sums exactly as decode does whatever else is in the launch. Writes unnormalised partials:
/// part_o[r][kvh][s][6][256] f32 and part_ml[r][kvh][s][6][2] (max, sum) in log2 units. Dynamic smem = 4 warps
/// x (6 x 256 + 12) f32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_attn_split(part_o: *mut f32, part_ml: *mut f32, q: *const u16, kc: *const u16,
    vc: *const u16, row_slot: *const u32, row_pos: *const u32, row_tbase: *const u32, row_tmask: *const u32,
    slot_stride: u64, scale_log2: f32, row_base: u32) {
    let lr = bid_x(); // row within this launch (partials index)
    let r = row_base + lr;
    let kvh = bid_y();
    let s = bid_z();
    let ns = gdim_z();
    let w = warp_id();
    let l = lane();
    let slot = *row_slot.add(r as usize) as u64;
    let pos = *row_pos.add(r as usize);
    // verify trees: key tbase + d is the ancestor at depth d = the d-th set bit of tmask (chain rows: tbase > pos)
    let tbase = *row_tbase.add(r as usize);
    let tmask = *row_tmask.add(r as usize);
    let nkeys = pos + 1;
    let rs = { let x = row_splits(nkeys); if x < ns { x } else { ns } };
    let split_len = nkeys.div_ceil(rs);
    let k0 = if s < rs { s * split_len } else { nkeys };
    let k1 = if k0 + split_len < nkeys { k0 + split_len } else { nkeys };

    // q for the 6 heads of this group, this lane's 8 dims, pre-scaled for exp2.
    let mut qv = [[0.0f32; 8]; 6];
    kcommon::unroll!(h in 6 => {
        let qp = q.add((r * NQ * HD + (kvh * GRP + h as u32) * HD + l * 8) as usize) as *const u32;
        let v = bf16::unpack8(ldg_nc_v4(qp));
        kcommon::unroll!(j in 8 => {
            qv[h][j] = v[j] * scale_log2;
        });
    });
    let mut m = [f32::NEG_INFINITY; 6];
    let mut lsum = [0.0f32; 6];
    let mut acc = [[0.0f32; 8]; 6];
    let base = (slot * slot_stride) as usize + (kvh * HD + l * 8) as usize;
    let mut key = k0 + w;
    while key < k1 {
        let mut ck = key;
        if key >= tbase {
            let mut m = tmask;
            let mut d = key - tbase;
            while d > 0 {
                m &= m - 1;
                d -= 1;
            }
            ck = tbase + m.trailing_zeros();
        }
        let off = base + ck as usize * (NKV * HD) as usize;
        let kv = bf16::unpack8(ldg_nc_v4(kc.add(off) as *const u32));
        let vv = bf16::unpack8(ldg_nc_v4(vc.add(off) as *const u32));
        kcommon::unroll!(h in 6 => {
            let mut d = 0.0;
            kcommon::unroll!(j in 8 => {
                d = math::fma(qv[h][j], kv[j], d);
            });
            let sc = warp::sum(d);
            let mn = if sc > m[h] { sc } else { m[h] };
            let corr = math::exp2(m[h] - mn);
            let p = math::exp2(sc - mn);
            lsum[h] = lsum[h] * corr + p;
            kcommon::unroll!(j in 8 => {
                acc[h][j] = math::fma(p, vv[j], acc[h][j] * corr);
            });
            m[h] = mn;
        });
        key += 4;
    }
    // cross-warp combine through shared memory
    let sm: Shared<f32> = dyn_base();
    let wbase = sm.add(w * (6 * 256 + 12));
    kcommon::unroll!(h in 6 => {
        kcommon::unroll!(j in 8 => {
            wbase.set(h as u32 * 256 + l * 8 + j as u32, acc[h][j]);
        });
    });
    if l == 0 {
        kcommon::unroll!(h in 6 => {
            wbase.set(1536 + h as u32, m[h]);
            wbase.set(1536 + 6 + h as u32, lsum[h]);
        });
    }
    sync_threads();
    let t = tid_x();
    let pbase = ((lr * NKV + kvh) * ns + s) as usize;
    let mut i = t;
    while i < 1536 {
        let h = i / 256;
        let mut mm = f32::NEG_INFINITY;
        let mut ww = 0;
        while ww < 4 {
            let mw = sm.at(ww * (1536 + 12) + 1536 + h);
            mm = if mw > mm { mw } else { mm };
            ww += 1;
        }
        let mut o = 0.0;
        let mut ls = 0.0;
        if mm != f32::NEG_INFINITY {
            let mut ww = 0;
            while ww < 4 {
                let wb = ww * (1536 + 12);
                let mw = sm.at(wb + 1536 + h);
                if mw != f32::NEG_INFINITY {
                    let f = math::exp2(mw - mm);
                    o += sm.at(wb + i) * f;
                    ls += sm.at(wb + 1536 + 6 + h) * f;
                }
                ww += 1;
            }
        }
        *part_o.add(pbase * 1536 + i as usize) = o;
        if (i & 255) == 0 {
            *part_ml.add(pbase * 12 + (h * 2) as usize) = mm;
            *part_ml.add(pbase * 12 + (h * 2 + 1) as usize) = ls;
        }
        i += bdim_x();
    }
}

/// Merge splits and apply the output gate: out[r][h*256+d] = bf16(bf16(o) * bf16(sigmoid(gate))).
/// grid = (rows, 24), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_attn_combine(out: *mut u16, part_o: *const f32, part_ml: *const f32,
    qg: *const u16, ld_qg: u32, nsplit: u32, row_base: u32) {
    let lr = bid_x();
    let r = row_base + lr;
    let h = bid_y();
    let d = tid_x();
    let kvh = h / GRP;
    let hh = h % GRP;
    let mut mm = f32::NEG_INFINITY;
    let mut s = 0;
    while s < nsplit {
        let pb = ((lr * NKV + kvh) * nsplit + s) as usize;
        let mw = *part_ml.add(pb * 12 + (hh * 2) as usize);
        mm = if mw > mm { mw } else { mm };
        s += 1;
    }
    let mut o = 0.0;
    let mut ls = 0.0;
    let mut s = 0;
    while s < nsplit {
        let pb = ((lr * NKV + kvh) * nsplit + s) as usize;
        let mw = *part_ml.add(pb * 12 + (hh * 2) as usize);
        if mw != f32::NEG_INFINITY {
            let f = math::exp2(mw - mm);
            o += *part_o.add(pb * 1536 + (hh * 256 + d) as usize) * f;
            ls += *part_ml.add(pb * 12 + (hh * 2 + 1) as usize) * f;
        }
        s += 1;
    }
    let ov = bf16::round(o / ls);
    let g = bf16::to_f32(*qg.add(r as usize * ld_qg as usize + (h * 2 * HD + HD + d) as usize));
    let sg = bf16::round(math::sigmoid(g));
    *out.add((r * NQ * HD + h * HD + d) as usize) = bf16::from_f32(ov * sg);
}

// ---- Short segments on tensor cores (decode rows and verify segments alike) ----
// fe_attn_vk: every row of a run of short segments (1-row decode, chain or tree verifies of <= 16 rows) over its
// keys in decode's virtual order (as fe_attn_split: key v is cache[v] below tbase, then the depth-ordered
// ancestors). A row's partition depends only on its own key count: chunks of vk_chunk(n) keys, split s = chunk s
// (VK_SPLITS splits, the rest empty), 16-key tiles in order inside a chunk. A K/V tile is loaded once and shared
// by every row of the segment whose keys in it are the same cache rows (the committed prefix); a tree row's tail
// tiles are loaded for it alone. The per-row arithmetic is the same either way (rows of an mma are independent;
// a row left out of a tile is an exact no-op), so a verify row is bitwise its later 1-row decode, and decode and
// verify share one kernel. QK^T on bf16 tensor cores, probabilities through f16 (scaled by 2^14 so long contexts
// stay normal) into P V on f16 tensor cores; partials in fe_attn_split's layout, merged by fe_attn_combine with
// nsplit = VK_SPLITS.

const SB: u32 = 16; // keys a tile
const SROWB: u32 = HD * 2; // 512 bytes a row
/// log2 of the probability scale (f16 P, max 2^14 < 65504).
const P_SHIFT: f32 = 14.0;
/// Splits of every row (grid z, the combine's nsplit).
pub const VK_SPLITS: u32 = 48;
/// K/V tiles in flight (cp.async stages).
const VK_STAGES: u32 = 3;
/// Q [GRP][16][256] + VK_STAGES x (K + V) [16][256], bf16 (98304 bytes; opt-in).
pub const SEG_SMEM: u32 = GRP * 16 * SROWB + VK_STAGES * 2 * SB * SROWB;

/// Chunk (= split) length for a row with `n` keys: whole tiles, VK_SPLITS chunks or fewer (a launch of one
/// 16-row segment and each 1-row decode is then about 2 x 48 / NKV waves of one block an SM at any context).
#[inline(always)]
pub fn vk_chunk(n: u32) -> u32 {
    SB * n.div_ceil(SB * VK_SPLITS)
}

/// Byte offset of 16-byte chunk `c` (0..32) of row `r` in a [rows][256] tile, XOR-swizzled (as flash.rs).
#[inline(always)]
fn ssw(r: u32, c: u32) -> u32 {
    r * SROWB + ((c ^ (r & 7)) << 4)
}

/// Two floats to an f16x2 word (`lo` = element 0).
#[inline(always)]
fn f16x2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe {
        core::arch::asm!("cvt.rn.f16x2.f32 {r}, {hi}, {lo};", r = out(reg32) r, hi = in(reg32) hi, lo = in(reg32) lo,
            options(pure, nomem, nostack));
    }
    r
}

/// x rounded to f16 and back.
#[inline(always)]
fn f16r(x: f32) -> f32 {
    let y: f32;
    unsafe {
        core::arch::asm!("{{ .reg .f16 h; cvt.rn.f16.f32 h, {x}; cvt.f32.f16 {y}, h; }}", y = out(reg32) y, x = in(reg32) x,
            options(pure, nomem, nostack));
    }
    y
}

/// Cache row of virtual key v for a row with tree base `tb` and ancestor mask `tm`.
#[inline(always)]
fn vk_phys(v: u32, tb: u32, tm: u32) -> u32 {
    if v < tb {
        return v;
    }
    let mut m = tm;
    let mut d = v - tb;
    while d > 0 {
        m &= m - 1;
        d -= 1;
    }
    tb + m.trailing_zeros()
}

/// grid = (segments of the run, NKV, VK_SPLITS), block = 32 x GRP (warp w: q head kvh * GRP + w), dyn smem =
/// SEG_SMEM. Segment seg0 + x has rows seg_row0[..] .. + seg_len[..] (<= 16, one slot); partial row index = row -
/// run_row0. q: [rows][NQ][256] (normed, roped); row_pos: RoPE position (keys = pos + 1).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_attn_vk(part_o: *mut f32, part_ml: *mut f32, q: *const u16, kc: *const u16,
    vc: *const u16, row_slot: *const u32, row_pos: *const u32, row_tbase: *const u32, row_tmask: *const u32,
    seg_row0: *const u32, seg_len: *const u32, seg0: u32, run_row0: u32, slot_stride: u64, scale_log2: f32) {
    use kcommon::mma::*;
    let meta = static_shared!(vk_meta, u32, 4 * 64);
    let smem: Shared<u8> = dyn_base();
    let sq = smem;
    let skv = smem.byte_add(GRP * 16 * SROWB);
    let seg = (seg0 + bid_x()) as usize;
    let row0 = *seg_row0.add(seg);
    let n = *seg_len.add(seg);
    let kvh = bid_y();
    let s = bid_z();
    let w = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    let nt = bdim_x();
    let tid = tid_x();
    if tid < n {
        let r = (row0 + tid) as usize;
        let nk = *row_pos.add(r) + 1;
        meta.set(tid, nk);
        meta.set(16 + tid, vk_chunk(nk));
        meta.set(32 + tid, *row_tbase.add(r));
        meta.set(48 + tid, *row_tmask.add(r));
    }
    sync_threads();
    let nk_of = |rr: u32| meta.at(rr);
    let c_of = |rr: u32| meta.at(16 + rr);
    let tb_of = |rr: u32| meta.at(32 + rr);
    let tm_of = |rr: u32| meta.at(48 + rr);
    let pbase_of = |rr: u32| ((((row0 - run_row0 + rr) * NKV + kvh) * VK_SPLITS) + s) as usize;

    // this split's work: per row keys [s c, min((s+1) c, nk))
    let mut any = false;
    let mut rr = 0;
    while rr < n {
        any |= s * c_of(rr) < nk_of(rr);
        rr += 1;
    }
    if !any {
        let mut i = tid;
        while i < n * GRP {
            let pb = pbase_of(i / GRP);
            *part_ml.add(pb * 12 + ((i % GRP) * 2) as usize) = f32::NEG_INFINITY;
            *part_ml.add(pb * 12 + ((i % GRP) * 2 + 1) as usize) = 0.0;
            i += nt;
        }
        return;
    }
    let slot = *row_slot.add(row0 as usize) as usize;
    // Q: the group's heads of the segment's rows, zero rows past n
    let mut i = tid;
    while i < GRP * 16 * 32 {
        let hh = i >> 9;
        let r = (i >> 5) & 15;
        let c = i & 31;
        let d = sq.byte_add(hh * 16 * SROWB + ssw(r, c)).addr();
        if r < n {
            cp_async_16(d, q.add((((row0 + r) * NQ + kvh * GRP + hh) * HD + c * 8) as usize) as *const u8);
        } else {
            cp_async_16_zfill(d, q as *const u8, 0);
        }
        i += nt;
    }
    cp_async_commit();

    // Work items, in the same order on every thread: chunk groups (rows with the same chunk length, increasing),
    // then tiles t of the group's split in increasing order, then for each tile the shared pass (rows whose keys
    // in it are cache rows t*16..) and one pass for each tree row with ancestors in it. An item is (cg, t, who):
    // who = 16 shared, < 16 that row alone.
    let need_own = |rr: u32, t: u32| {
        let nk = nk_of(rr);
        let end = if t * SB + SB < nk { t * SB + SB } else { nk };
        t * SB < nk && tb_of(rr) < end
    };
    // the next chunk length > cg present among the rows (u32::MAX: none)
    let next_group = |cg: u32| {
        let mut best = u32::MAX;
        let mut rr = 0;
        while rr < n {
            let c = c_of(rr);
            if c > cg && c < best && s * c < nk_of(rr) {
                best = c;
            }
            rr += 1;
        }
        best
    };
    // tile range of group cg in this split: [s cg / 16, ceil(min((s+1) cg, max nk) / 16))
    let tiles_of = |cg: u32| {
        let mut mx = 0;
        let mut rr = 0;
        while rr < n {
            if c_of(rr) == cg && nk_of(rr) > mx {
                mx = nk_of(rr);
            }
            rr += 1;
        }
        let end = if (s + 1) * cg < mx { (s + 1) * cg } else { mx };
        (s * cg / SB, end.div_ceil(SB))
    };
    // the item after (cg, t, who); (u32::MAX, ..) at the end
    let next_item_slow = |cg: u32, t: u32, who: u32| -> (u32, u32, u32) {
        // the next tree row of this tile needing its own pass
        let mut nw = if who == 16 { 0 } else { who + 1 };
        while nw < n && !(c_of(nw) == cg && need_own(nw, t)) {
            nw += 1;
        }
        if nw < n {
            return (cg, t, nw);
        }
        // the next tile's shared pass, else the next group's first tile
        let (_, t1) = tiles_of(cg);
        if t + 1 < t1 {
            return (cg, t + 1, 16);
        }
        let cn = next_group(cg);
        if cn == u32::MAX {
            return (u32::MAX, 0, 0);
        }
        (cn, tiles_of(cn).0, 16)
    };
    let first = {
        let cg = next_group(0);
        let (t0, _) = tiles_of(cg);
        (cg, t0, 16u32)
    };
    // fast path for the first group's leading tiles that no tree row needs on its own (all of a decode row's,
    // a verify segment's committed prefix): its key limit and the end of that tile range
    let cg1 = first.0;
    let (mx1, tmid1) = {
        let (mut mx, mut tb) = (0u32, u32::MAX);
        let mut rr = 0;
        while rr < n {
            if c_of(rr) == cg1 {
                if nk_of(rr) > mx {
                    mx = nk_of(rr);
                }
                if tb_of(rr) < tb {
                    tb = tb_of(rr);
                }
            }
            rr += 1;
        }
        let (_, t1) = tiles_of(cg1);
        (mx, if tb / SB < t1 { tb / SB } else { t1 })
    };
    let next_item = |cg: u32, t: u32, who: u32| -> (u32, u32, u32) {
        if cg == cg1 && who == 16 && t + 1 < tmid1 {
            return (cg, t + 1, 16);
        }
        next_item_slow(cg, t, who)
    };
    let cbase = slot * slot_stride as usize + (kvh * HD) as usize;
    let kv_stride = (NKV * HD) as usize;
    // K / V tile of an item: virtual keys t*16.. of the group's rows (shared) or of row `who` (its tree tail)
    let load_item = |it: (u32, u32, u32), buf: u32| {
        let (cg, t, who) = it;
        let k0 = t * SB;
        let (lim, tb, tm) = if who < 16 {
            (nk_of(who), tb_of(who), tm_of(who))
        } else if cg == cg1 {
            (mx1, u32::MAX, 0)
        } else {
            let mut mx = 0;
            let mut rr = 0;
            while rr < n {
                if c_of(rr) == cg && nk_of(rr) > mx {
                    mx = nk_of(rr);
                }
                rr += 1;
            }
            (mx, u32::MAX, 0)
        };
        let st = skv.byte_add(buf * 2 * SB * SROWB);
        let mut i = tid;
        while i < 2 * SB * 32 {
            let which = i / (SB * 32);
            let r = (i >> 5) & (SB - 1);
            let c = i & 31;
            let d = st.byte_add(which * SB * SROWB + ssw(r, c)).addr();
            let src = if which == 0 { kc } else { vc };
            let v = k0 + r;
            if v < lim {
                let p = vk_phys(v, tb, tm);
                cp_async_16(d, src.add(cbase + p as usize * kv_stride + (c * 8) as usize) as *const u8);
            } else {
                cp_async_16_zfill(d, src as *const u8, 0);
            }
            i += nt;
        }
    };
    // VK_STAGES - 1 items in flight ahead of the one computed
    let mut ring = [first, (u32::MAX, 0, 0), (u32::MAX, 0, 0)];
    load_item(first, 0);
    cp_async_commit();
    let mut k = 1;
    while k < VK_STAGES - 1 {
        let pv = ring[(k - 1) as usize];
        if pv.0 != u32::MAX {
            let nx = next_item(pv.0, pv.1, pv.2);
            ring[k as usize] = nx;
            if nx.0 != u32::MAX {
                load_item(nx, k);
            }
        }
        cp_async_commit();
        k += 1;
    }

    let sqw = sq.byte_add(w * 16 * SROWB);
    // this lane's rows g and g + 8 of the segment
    let rows = [g, g + 8];
    let lim1 = [
        if g < n && c_of(g) == cg1 { nk_of(g) } else { 0 },
        if g + 8 < n && c_of(g + 8) == cg1 { nk_of(g + 8) } else { 0 },
    ];
    let mut o = [[0.0f32; 4]; 32];
    let mut mrow = [f32::NEG_INFINITY; 2];
    let mut lrow = [0.0f32; 2];
    let mut buf = 0u32;
    let mut it = first;
    while it.0 != u32::MAX {
        // the item VK_STAGES - 1 ahead into the buffer last used
        let last = ring[((buf + VK_STAGES - 2) % VK_STAGES) as usize];
        let ahead = if last.0 != u32::MAX { next_item(last.0, last.1, last.2) } else { (u32::MAX, 0, 0) };
        let slot_b = (buf + VK_STAGES - 1) % VK_STAGES;
        ring[slot_b as usize] = ahead;
        if ahead.0 != u32::MAX {
            load_item(ahead, slot_b);
        }
        cp_async_commit();
        cp_async_wait::<{ VK_STAGES - 1 }>();
        sync_threads();
        let sk = skv.byte_add(buf * 2 * SB * SROWB);
        let sv = sk.byte_add(SB * SROWB);
        let sv32: Shared<u32> = sv.cast();
        let mut i = tid;
        while i < SB * HD / 2 {
            let x = sv32.at(i);
            sv32.set(i, f16x2(f32::from_bits(x << 16), f32::from_bits(x & 0xffff_0000)));
            i += nt;
        }
        sync_threads();
        let (cg, t, who) = it;
        let k0 = t * SB;
        // which of this lane's rows take part, and their key limits
        let lim = if cg == cg1 && who == 16 && t < tmid1 {
            lim1
        } else {
            let mut lim = [0u32; 2];
            kcommon::unroll!(hf in 2 => {
                let rr = rows[hf];
                let part = rr < n && c_of(rr) == cg && if who < 16 { rr == who } else { !need_own(rr, t) };
                lim[hf] = if part { nk_of(rr) } else { 0 };
            });
            lim
        };
        let mut sc = [[0.0f32; 4]; 2];
        kcommon::unroll!(ks in 16 => {
            let a = ldmatrix_x4(sqw.byte_add(ssw(l & 15, ks as u32 * 2 + (l >> 4))).addr());
            let bb = ldmatrix_x4(sk.byte_add(ssw((l >> 4) * 8 + (l & 7), ks as u32 * 2 + ((l >> 3) & 1))).addr());
            mma_bf16_16816(&mut sc[0], a, [bb[0], bb[1]]);
            mma_bf16_16816(&mut sc[1], a, [bb[2], bb[3]]);
        });
        let mut mx = mrow;
        kcommon::unroll!(tn in 2 => {
            let key = k0 + tn as u32 * 8 + 2 * t4;
            sc[tn][0] = if key < lim[0] { sc[tn][0] * scale_log2 } else { f32::NEG_INFINITY };
            sc[tn][1] = if key + 1 < lim[0] { sc[tn][1] * scale_log2 } else { f32::NEG_INFINITY };
            sc[tn][2] = if key < lim[1] { sc[tn][2] * scale_log2 } else { f32::NEG_INFINITY };
            sc[tn][3] = if key + 1 < lim[1] { sc[tn][3] * scale_log2 } else { f32::NEG_INFINITY };
            mx[0] = mx[0].max(sc[tn][0]).max(sc[tn][1]);
            mx[1] = mx[1].max(sc[tn][2]).max(sc[tn][3]);
        });
        kcommon::unroll!(hf in 2 => {
            mx[hf] = mx[hf].max(warp::shfl_xor(mx[hf], 1));
            mx[hf] = mx[hf].max(warp::shfl_xor(mx[hf], 2));
        });
        let base = [if mx[0] == f32::NEG_INFINITY { 0.0 } else { mx[0] }, if mx[1] == f32::NEG_INFINITY { 0.0 } else { mx[1] }];
        let cr = [math::exp2(mrow[0] - base[0]), math::exp2(mrow[1] - base[1])];
        mrow = mx;
        let mut ps = [0.0f32; 2];
        let mut pa = [0u32; 4];
        kcommon::unroll!(tn in 2 => {
            let e0 = f16r(math::exp2(sc[tn][0] - base[0] + P_SHIFT));
            let e1 = f16r(math::exp2(sc[tn][1] - base[0] + P_SHIFT));
            let e2 = f16r(math::exp2(sc[tn][2] - base[1] + P_SHIFT));
            let e3 = f16r(math::exp2(sc[tn][3] - base[1] + P_SHIFT));
            ps[0] += e0 + e1;
            ps[1] += e2 + e3;
            pa[tn * 2] = f16x2(e0, e1);
            pa[tn * 2 + 1] = f16x2(e2, e3);
        });
        lrow[0] = lrow[0] * cr[0] + ps[0];
        lrow[1] = lrow[1] * cr[1] + ps[1];
        kcommon::unroll!(dt in 32 => {
            o[dt][0] *= cr[0];
            o[dt][1] *= cr[0];
            o[dt][2] *= cr[1];
            o[dt][3] *= cr[1];
        });
        kcommon::unroll!(dp in 16 => {
            let bb = ldmatrix_x4_trans(sv.byte_add(ssw((l & 7) + ((l >> 3) & 1) * 8, dp as u32 * 2 + (l >> 4))).addr());
            mma_f16_16816(&mut o[2 * dp], pa, [bb[0], bb[1]]);
            mma_f16_16816(&mut o[2 * dp + 1], pa, [bb[2], bb[3]]);
        });
        sync_threads();
        buf = (buf + 1) % VK_STAGES;
        it = ring[buf as usize];
    }
    cp_async_wait_all();

    kcommon::unroll!(hf in 2 => {
        lrow[hf] += warp::shfl_xor(lrow[hf], 1);
        lrow[hf] += warp::shfl_xor(lrow[hf], 2);
    });
    kcommon::unroll!(hf in 2 => {
        let rr = rows[hf];
        if rr < n {
            let pb = pbase_of(rr);
            let po = part_o.add(pb * 1536 + (w * HD) as usize);
            kcommon::unroll!(dt in 32 => {
                let d = (dt as u32 * 8 + 2 * t4) as usize;
                *po.add(d) = o[dt][2 * hf];
                *po.add(d + 1) = o[dt][2 * hf + 1];
            });
            if t4 == 0 {
                *part_ml.add(pb * 12 + (w * 2) as usize) = mrow[hf] - P_SHIFT; // -inf stays -inf
                *part_ml.add(pb * 12 + (w * 2 + 1) as usize) = lrow[hf];
            }
        }
    });
}
