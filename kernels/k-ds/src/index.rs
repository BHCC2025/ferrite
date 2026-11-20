//! The compressor (ratio 1 / 2 latents), indexer scores and the exact top-k selection.
use kcommon::smem::Shared;
use kcommon::*;

/// Compressor, step 1: stash each row's kvc / sc (f32 [dim] each, the compressor linears' outputs) in the tail
/// ring tail[slot][pos % ring_len][2 * dim]; then for rows closing a group ((pos + 1) % ratio == 0) write the
/// pre-norm latent: ratio 1: bf16(kvc); ratio 2: bf16(sum_t kvc_t * softmax_t(sc_t)) over the pair (pos - 1,
/// pos), per channel. Other rows get zeros. out bf16 [rows][dim]. The ring must hold a group's earlier rows
/// (written by this or an earlier launch). grid = rows, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_comp_stash(tail: *mut f32, tail_stride: u32, ring_len: u32, kvc: *const f32,
    sc: *const f32, meta: *const u32, ratio: u32, dim: u32) {
    let r = bid_x() as usize;
    let slot = *meta.add(2 * r) as usize;
    let pos = *meta.add(2 * r + 1);
    let d = dim as usize;
    let dst = tail.add(slot * tail_stride as usize + (pos % ring_len) as usize * 2 * d);
    let mut i = tid_x() as usize;
    while i < d {
        *dst.add(i) = *kvc.add(r * d + i);
        if ratio > 1 {
            *dst.add(d + i) = *sc.add(r * d + i);
        }
        i += bdim_x() as usize;
    }
}

/// Compressor, step 2 (after every row's stash): the pre-norm latent of rows closing a group, zeros elsewhere.
/// grid = rows, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_comp_pair(out: *mut u16, tail: *const f32, tail_stride: u32, ring_len: u32,
    meta: *const u32, ratio: u32, dim: u32) {
    let r = bid_x() as usize;
    let slot = *meta.add(2 * r) as usize;
    let pos = *meta.add(2 * r + 1);
    let d = dim as usize;
    let o = out.add(r * d);
    let closes = (pos + 1) % ratio == 0;
    let tb = tail.add(slot * tail_stride as usize);
    let mut i = tid_x() as usize;
    while i < d {
        let v = if !closes {
            0.0
        } else if ratio == 1 {
            *tb.add((pos % ring_len) as usize * 2 * d + i)
        } else {
            // ratio 2: tokens pos - 1, pos
            let a = tb.add(((pos - 1) % ring_len) as usize * 2 * d);
            let b = tb.add((pos % ring_len) as usize * 2 * d);
            let (ka, sa, kb, sb) = (*a.add(i), *a.add(d + i), *b.add(i), *b.add(d + i));
            let m = sa.max(sb);
            let ea = math::exp(sa - m);
            let eb = math::exp(sb - m);
            let s = ea + eb;
            ka * (ea / s) + kb * (eb / s)
        };
        *o.add(i) = bf16::from_f32(v);
        i += bdim_x() as usize;
    }
}

/// Indexer scores: score[r][e] = sum_h w[r][h] * relu(q[r][h] . k[e]) for e < len(r) = (pos + 1) / ratio (and,
/// with a candidate mask, only entries whose 8-block is kept: mask[r][e / 8] != 0), else -inf. q f32 [rows][32][128]
/// (FP4 grid), w f32 [rows][32], keys bf16 ikeys[slot][e][128]. score rows have stride `ld`.
/// grid = (ceil(ld / 128), rows), block = 128.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_idx_score(score: *mut f32, ld: u32, q: *const f32, w: *const f32,
    ikeys: *const u16, key_stride: u32, meta: *const u32, ratio: u32, mask: *const u8, mask_ld: u32) {
    let sq: Shared<f32> = kcommon::static_shared!(is_q, f32, 16384);
    let sw: Shared<f32> = kcommon::static_shared!(is_w, f32, 128);
    let r = bid_y() as usize;
    let slot = *meta.add(2 * r) as usize;
    let pos = *meta.add(2 * r + 1);
    let len = ((pos + 1) / ratio) as usize;
    let t = tid_x() as usize;
    let mut i = t;
    while i < 4096 {
        sq.set(i as u32, *q.add(r * 4096 + i));
        i += 128;
    }
    if t < 32 {
        sw.set(t as u32, *w.add(r * 32 + t));
    }
    sync_threads();
    let e = bid_x() as usize * 128 + t;
    if e >= ld as usize {
        return;
    }
    let mut out = f32::NEG_INFINITY;
    let keep = e < len && (mask.is_null() || *mask.add(r * mask_ld as usize + e / 8) != 0);
    if keep {
        let kp = ikeys.add(slot * key_stride as usize + e * 128) as *const u32;
        let mut kv = [0.0f32; 128];
        let mut j = 0;
        while j < 16 {
            let v = bf16::unpack8(ldg_nc_v4(kp.add(j * 4)));
            let mut c = 0;
            while c < 8 {
                kv[j * 8 + c] = v[c];
                c += 1;
            }
            j += 1;
        }
        let mut tot = 0.0f32;
        let mut hh = 0u32;
        while hh < 32 {
            let mut dot = 0.0f32;
            let mut c = 0u32;
            while c < 128 {
                dot = math::fma(sq.at(hh * 128 + c), kv[c as usize], dot);
                c += 1;
            }
            tot += dot.max(0.0) * sw.at(hh);
            hh += 1;
        }
        out = tot;
    }
    *score.add(r * ld as usize + e) = out;
}

/// Monotonic u32 key of an f32 (larger float -> larger key); -inf is the smallest finite-order key.
#[inline(always)]
fn fkey(f: f32) -> u32 {
    let b = f.to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}

/// Exact top-k of each row's scores (k <= 1024; ties go to the lower index), out as ascending entry ids padded
/// with -1: sel[r][0..k]. Entries scored -inf never count (n_valid = number of finite scores, capped at k).
/// grid = rows, block = 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_topk(sel: *mut i32, score: *const f32, ld: u32, n: u32, k: u32) {
    let r = bid_x() as usize;
    topk_body(sel.add(r * k as usize), score.add(r * ld as usize), n, k);
}

/// Two-stage exact top-k for long rows, stage 1: block (c, r) takes chunk c ([c * chunk, (c + 1) * chunk) of row r's
/// first n scores) and keeps its exact top-k (ties to the lower index): cidx / cval [rows][chunks][k] hold the kept
/// entries' row indices (ascending) and scores, padded with -1 / -inf. Any entry of the row's top-k is in its chunk's
/// top-k, so stage 2 over the candidates gives ds_topk's selection exactly. grid = (chunks, rows), block = 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_topk_part(cidx: *mut i32, cval: *mut f32, score: *const f32, ld: u32, n: u32,
    k: u32, chunk: u32) {
    let c = bid_x();
    let r = bid_y() as usize;
    let nch = gdim_x() as usize;
    let base = c * chunk;
    let nc = if base < n { (n - base).min(chunk) } else { 0 };
    let row = score.add(r * ld as usize + base as usize);
    let o = (r * nch + c as usize) * k as usize;
    topk_body(cidx.add(o), row, nc, k);
    sync_threads();
    let mut j = tid_x();
    while j < k {
        let e = *cidx.add(o + j as usize);
        if e >= 0 {
            *cval.add(o + j as usize) = *row.add(e as usize);
            *cidx.add(o + j as usize) = e + base as i32;
        } else {
            *cval.add(o + j as usize) = f32::NEG_INFINITY;
        }
        j += bdim_x();
    }
}

/// Two-stage top-k, stage 2: the exact top-k of row r's chunks x k candidates (their positions ascend with the row
/// index, so ties still go to the lower index), written as row indices, ascending, -1 padded. grid = rows, block = 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_topk_fin(sel: *mut i32, cidx: *const i32, cval: *const f32, nch: u32, k: u32) {
    let r = bid_x() as usize;
    let nk = nch as usize * k as usize;
    let out = sel.add(r * k as usize);
    topk_body(out, cval.add(r * nk), nk as u32, k);
    sync_threads();
    let mut j = tid_x();
    while j < k {
        let p = *out.add(j as usize);
        if p >= 0 {
            *out.add(j as usize) = *cidx.add(r * nk + p as usize);
        }
        j += bdim_x();
    }
}

/// ds_topk's selection of one row: out[0..k] = the indices of the top min(k, valid) of row[0..n], ascending, -1 padded.
#[inline(always)]
unsafe fn topk_body(out: *mut i32, row: *const f32, n: u32, k: u32) {
    let hist: Shared<u32> = kcommon::static_shared!(tk_hist, u32, 1024);
    let wsum: Shared<u32> = kcommon::static_shared!(tk_ws, u32, 256);
    let sv: Shared<u32> = kcommon::static_shared!(tk_sv, u32, 16);
    let t = tid_x();
    let neg = fkey(f32::NEG_INFINITY);
    // count valid
    let mut cnt = 0u32;
    let mut i = t;
    while i < n {
        cnt += (fkey(*row.add(i as usize)) > neg) as u32;
        i += 1024;
    }
    let nvalid = block_sum_u32(cnt, wsum);
    let kk = k.min(nvalid);
    // threshold key T = the kk-th largest valid key (radix select, 8 bits a pass); `need` of the T ties are kept
    let mut prefix = 0u32;
    let mut want = kk; // rank (1-based) among keys matching the prefix, counted from the top
    let mut shift = 32u32;
    while shift > 0 && kk > 0 {
        shift -= 8;
        if t < 256 {
            hist.set(t, 0);
        }
        sync_threads();
        let hmask = if shift == 24 { 0 } else { !0u32 << (shift + 8) };
        let mut i = t;
        while i < n {
            let key = fkey(*row.add(i as usize));
            if key > neg && (key & hmask) == (prefix & hmask) {
                atomic_add_shared(hist.add((key >> shift) & 0xff), 1);
            }
            i += 1024;
        }
        sync_threads();
        if t == 0 {
            let mut b = 256u32;
            let mut acc = 0u32;
            while b > 0 {
                b -= 1;
                let c = hist.at(b);
                if acc + c >= want {
                    prefix |= b << shift;
                    want -= acc;
                    break;
                }
                acc += c;
            }
            sv.set(0, prefix);
            sv.set(1, want);
        }
        sync_threads();
        prefix = sv.at(0);
        want = sv.at(1);
        sync_threads();
    }
    let thr = prefix;
    let need_eq = want; // keys == thr to keep, lowest index first
    // compaction in index order, tiles of 1024
    let mut base_out = 0u32;
    let mut eq_seen = 0u32;
    let mut i0 = 0u32;
    while i0 < n && kk > 0 {
        let i = i0 + t;
        let key = if i < n { fkey(*row.add(i as usize)) } else { 0 };
        let valid = i < n && key > neg;
        let is_eq = valid && key == thr;
        let eq_rank = eq_seen + block_excl_scan(is_eq as u32, wsum, sv);
        let eq_tot = sv.at(2);
        let keep = valid && (key > thr || (is_eq && eq_rank < need_eq));
        let at = base_out + block_excl_scan(keep as u32, wsum, sv);
        let kept = sv.at(2);
        if keep {
            *out.add(at as usize) = i as i32;
        }
        base_out += kept;
        eq_seen += eq_tot;
        sync_threads();
        i0 += 1024;
    }
    let mut j = kk + t;
    while j < k {
        *out.add(j as usize) = -1;
        j += 1024;
    }
}

#[inline(always)]
unsafe fn atomic_add_shared(p: Shared<u32>, v: u32) {
    core::arch::asm!("red.shared.add.u32 [{a}], {v};", a = in(reg32) p.addr(), v = in(reg32) v, options(nostack));
}

#[inline(always)]
fn warp_incl_scan(mut v: u32) -> u32 {
    let l = lane();
    let mut d = 1;
    while d < 32 {
        let o = warp::shfl_up_u32(v, d);
        if l >= d {
            v += o;
        }
        d <<= 1;
    }
    v
}

/// Block-wide sum of u32 (block 1024), scratch >= 32 words.
#[inline(always)]
fn block_sum_u32(v: u32, ws: Shared<u32>) -> u32 {
    let mut s = v;
    let mut m = 16;
    while m > 0 {
        s += warp::shfl_xor_u32(s, m);
        m >>= 1;
    }
    if lane() == 0 {
        ws.set(warp_id(), s);
    }
    sync_threads();
    let mut t = if lane() < nwarps() { ws.at(lane()) } else { 0 };
    let mut m = 16;
    while m > 0 {
        t += warp::shfl_xor_u32(t, m);
        m >>= 1;
    }
    sync_threads();
    t
}

/// Exclusive block scan (block 1024); the block total lands in sv[2]. Contains barriers.
#[inline(always)]
fn block_excl_scan(v: u32, ws: Shared<u32>, sv: Shared<u32>) -> u32 {
    let inc = warp_incl_scan(v);
    if lane() == 31 {
        ws.set(warp_id(), inc);
    }
    sync_threads();
    if warp_id() == 0 {
        let wv = if lane() < nwarps() { ws.at(lane()) } else { 0 };
        let wi = warp_incl_scan(wv);
        if lane() < nwarps() {
            ws.set(32 + lane(), wi - wv);
        }
        if lane() == 31 {
            sv.set(2, wi);
        }
    }
    sync_threads();
    let r = ws.at(32 + warp_id()) + inc - v;
    sync_threads();
    r
}

/// Candidate pool of the source layer: per row, block maxima of its scores over `bsize`-entry blocks; the block
/// holding the row's last valid entry is forced in; keep the top `nblocks` blocks (ties: lower index). mask[r][b]
/// = 1 kept, 0 not. Rows with <= nblocks valid blocks keep all of them. bmax scratch f32 [rows][mask_ld].
/// grid = rows, block = 1024. Uses ds_topk's selection on the block maxima.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_block_max(bmax: *mut f32, mask_ld: u32, score: *const f32, ld: u32,
    meta: *const u32, ratio: u32, bsize: u32) {
    let r = bid_x() as usize;
    let pos = *meta.add(2 * r + 1);
    let len = (pos + 1) / ratio;
    let nb = len.div_ceil(bsize);
    let last = if len > 0 { (len - 1) / bsize } else { u32::MAX };
    let mut b = tid_x();
    while b < mask_ld {
        let mut m = f32::NEG_INFINITY;
        if b < nb {
            let mut j = 0;
            while j < bsize {
                let e = b * bsize + j;
                if e < len {
                    m = m.max(*score.add(r * ld as usize + e as usize));
                }
                j += 1;
            }
            if b == last {
                m = f32::INFINITY;
            }
        }
        *bmax.add(r * mask_ld as usize + b as usize) = m;
        b += bdim_x();
    }
}

/// mask[r][b] = 1 for the blocks in sel[r][0..k] (ascending, -1 padded), others 0. grid = rows, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_sel_to_mask(mask: *mut u8, mask_ld: u32, sel: *const i32, k: u32) {
    let r = bid_x() as usize;
    let mut b = tid_x();
    while b < mask_ld {
        *mask.add(r * mask_ld as usize + b as usize) = 0;
        b += bdim_x();
    }
    sync_threads();
    let mut j = tid_x();
    while j < k {
        let e = *sel.add(r * k as usize + j as usize);
        if e >= 0 {
            *mask.add(r * mask_ld as usize + e as usize) = 1;
        }
        j += bdim_x();
    }
}

/// Tensor-core indexer scores (same contract as ds_idx_score, but q bf16 [rows][32][128] on the FP4 grid, exact in
/// bf16): one row a block, 128 entries a block. The 32 heads x 128 entries x 128 dims run as bf16 mma (f32
/// accumulation), then relu, the head weights and the sum over heads in a fixed order: each row's bits are its own
/// (any batch, any slot mix). Entries past `ecap` (the slot's plane size) are not read.
/// grid = (ceil(nmax / 128), rows), block = 256 (8 warps x 16 entries).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_idx_score_tc(score: *mut f32, ld: u32, q: *const u16, w: *const f32,
    ikeys: *const u16, key_stride: u32, ecap: u32, meta: *const u32, ratio: u32, mask: *const u8, mask_ld: u32) {
    const KP: u32 = 136; // padded row (bf16): conflict-free fragment loads
    let ks: Shared<u32> = kcommon::static_shared!(itc_k, u32, 34816); // [128 entries][136] bf16
    let qs: Shared<u32> = kcommon::static_shared!(itc_q, u32, 8704); // [32 heads][136] bf16
    let sw: Shared<f32> = kcommon::static_shared!(itc_w, f32, 128);
    let r = bid_y() as usize;
    let slot = *meta.add(2 * r) as usize;
    let pos = *meta.add(2 * r + 1);
    let len = (pos + 1) / ratio;
    let e0 = bid_x() * 128;
    let t = tid_x();
    // q row: 32 x 128 bf16 = 2048 words
    let qr = q.add(r * 4096) as *const u32;
    let mut i = t;
    while i < 2048 {
        let (h, c) = (i / 64, i % 64);
        qs.set(h * (KP / 2) + c, *qr.add(i as usize));
        i += 256;
    }
    if t < 32 {
        sw.set(t, *w.add(r * 32 + t as usize));
    }
    // keys: 128 entries x 128 bf16 (zero past ecap)
    let kb = ikeys.add(slot * key_stride as usize) as *const u32;
    let mut i = t;
    while i < 128 * 64 {
        let (n, c) = (i / 64, i % 64);
        let e = e0 + n;
        let v = if e < ecap { *kb.add(e as usize * 64 + c as usize) } else { 0 };
        ks.set(n * (KP / 2) + c, v);
        i += 256;
    }
    sync_threads();
    let wid = warp_id();
    let l = lane();
    let (g, tq) = (l / 4, l % 4);
    let mut acc = [[[0.0f32; 4]; 2]; 2]; // [m tile][n tile][frag]
    let mut k0 = 0u32;
    while k0 < 128 {
        let kw = k0 / 2 + tq; // word of (k0 + 2t)
        let mut a = [[0u32; 4]; 2];
        let mut mt = 0;
        while mt < 2 {
            let h = 16 * mt as u32 + g;
            a[mt] = [qs.at(h * (KP / 2) + kw), qs.at((h + 8) * (KP / 2) + kw), qs.at(h * (KP / 2) + kw + 4),
                qs.at((h + 8) * (KP / 2) + kw + 4)];
            mt += 1;
        }
        let mut nt = 0;
        while nt < 2 {
            let n = wid * 16 + 8 * nt as u32 + g;
            let b = [ks.at(n * (KP / 2) + kw), ks.at(n * (KP / 2) + kw + 4)];
            mma::mma_bf16_16816(&mut acc[0][nt], a[0], b);
            mma::mma_bf16_16816(&mut acc[1][nt], a[1], b);
            nt += 1;
        }
        k0 += 16;
    }
    // relu x head weight, summed over the 32 heads: m tiles, then the two halves of a tile, then lanes g = 0..7
    let mut nt = 0;
    while nt < 2 {
        let mut s = [0.0f32; 2];
        let mut c = 0;
        while c < 2 {
            let mut v = 0.0f32;
            let mut mt = 0;
            while mt < 2 {
                let h = 16 * mt as u32 + g;
                v += acc[mt][nt][c].max(0.0) * sw.at(h) + acc[mt][nt][c + 2].max(0.0) * sw.at(h + 8);
                mt += 1;
            }
            v += warp::shfl_xor(v, 4);
            v += warp::shfl_xor(v, 8);
            v += warp::shfl_xor(v, 16);
            s[c] = v;
            c += 1;
        }
        if g == 0 {
            let mut c = 0;
            while c < 2 {
                let e = e0 + wid * 16 + 8 * nt as u32 + 2 * tq + c as u32;
                if e < ld {
                    let keep = e < len && (mask.is_null() || *mask.add(r * mask_ld as usize + e as usize / 8) != 0);
                    *score.add(r * ld as usize + e as usize) = if keep { s[c] } else { f32::NEG_INFINITY };
                }
                c += 1;
            }
        }
        nt += 1;
    }
}
