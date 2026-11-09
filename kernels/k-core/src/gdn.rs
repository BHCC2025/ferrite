//! Gated DeltaNet (Qwen3.5 linear attention): causal depthwise conv1d (k=4) + SiLU with carried state,
//! and the delta-rule recurrence with fp32 state fused with the gated RMSNorm.
//!
//! Work is described by segments: segment i covers rows [seg_row0[i], seg_row0[i] + seg_len[i]) of the
//! batch, all belonging to slot seg_slot[i], in order. Decode = segments of length 1.
use kcommon::smem::Shared;
use kcommon::*;

pub const NKH: u32 = crate::dims::NKH; // 16 (8 a TP2 rank)
pub const NVH: u32 = crate::dims::NVH; // 48 (24 a TP2 rank)
pub const CONV_DIM: u32 = 2 * NKH * 128 + NVH * 128; // q 2048 | k 2048 | v 6144
pub const DK: u32 = 128;
pub const DV: u32 = 128;
/// saved node states a verify tree's recurrence may hold at once (Engine's tree_steps checks it)
pub const TREE_SLOTS: usize = 8;
/// rows fe_gdn_rec2 normalises together before running their recurrence (a warp each for a row's q and k: 16 warps)
const REC_CHUNK: usize = 8;
const K_OFF: u32 = NKH * DK; // 2048
const V_OFF: u32 = 2 * NKH * DK; // 4096

/// grid = (CONV_DIM / 256, nseg), block = 256.
/// x: projection rows (conv channels in columns 0..10240, row stride ldx); w: [10240][4] bf16;
/// state: [slot][3][10240] bf16 raw inputs (oldest first); out: [rows][10240] bf16 = silu(conv).
/// hist (optional): [slot][t][3][10240] = the window after row t of the segment (speculative rollback).
/// seg_tree (optional): per-segment flags; a flagged segment is a verify tree (<= 32 rows): row t's taps are its
/// ancestors' inputs (row_par: its last three ancestors, segment-local, packed by Engine's prepare_tree) and its
/// history is the window along its own root path; the state ends as the last row's window (Engine::commit_tree
/// restores the kept path's from the history).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gdn_conv(out: *mut u16, x: *const u16, ldx: u32, w: *const u16, state: *mut u16,
    seg_slot: *const u32, seg_row0: *const u32, seg_len: *const u32, hist: *mut u16, hist_rows: u32,
    row_par: *const u32, seg_tree: *const u32) {
    let c = bid_x() * 256 + tid_x();
    let sg = bid_y() as usize;
    let slot = *seg_slot.add(sg) as usize;
    let r0 = *seg_row0.add(sg) as usize;
    let n = *seg_len.add(sg) as usize;
    let st = state.add(slot * 3 * CONV_DIM as usize + c as usize);
    let wv = *(w.add(c as usize * 4) as *const [u32; 2]);
    let (w0, w1, w2, w3) = (bf16::lo(wv[0]), bf16::hi(wv[0]), bf16::lo(wv[1]), bf16::hi(wv[1]));
    let mut s0 = bf16::to_f32(*st);
    let mut s1 = bf16::to_f32(*st.add(CONV_DIM as usize));
    let mut s2 = bf16::to_f32(*st.add(2 * CONV_DIM as usize));
    if !seg_tree.is_null() && *seg_tree.add(sg) != 0 {
        // row t's window = the inputs of its last three ancestors (row_par: p1 | p2 << 8 | p3 << 16, 0xff above
        // the root, where the state's entries continue the path); every row's input stays in shared memory, read
        // back only by the thread that stored it
        let xs: Shared<u16> = kcommon::static_shared!(conv_tree_x, u16, 32 * 256 * 2);
        let xa = |r: u32| bf16::to_f32(xs.at(r * 256 + tid_x()));
        let mut last = (s0, s1, s2);
        let mut t = 0;
        while t < n && t < 32 {
            let an = *row_par.add(r0 + t);
            let (p1, p2, p3) = (an & 0xff, (an >> 8) & 0xff, (an >> 16) & 0xff);
            let (a0, a1, a2) = if p1 == 0xff { (s0, s1, s2) } else if p2 == 0xff { (s1, s2, xa(p1)) } else {
                (if p3 == 0xff { s2 } else { xa(p3) }, xa(p2), xa(p1))
            };
            let xb = *x.add((r0 + t) * ldx as usize + c as usize);
            xs.set(t as u32 * 256 + tid_x(), xb);
            let xv = bf16::to_f32(xb);
            let y = bf16::round(w0 * a0 + w1 * a1 + w2 * a2 + w3 * xv);
            *out.add((r0 + t) * CONV_DIM as usize + c as usize) = bf16::from_f32(math::silu(y));
            last = (a1, a2, xv);
            if !hist.is_null() && (t as u32) < hist_rows {
                let hp = hist.add((slot * hist_rows as usize + t) * 3 * CONV_DIM as usize + c as usize);
                *hp = bf16::from_f32(a1);
                *hp.add(CONV_DIM as usize) = bf16::from_f32(a2);
                *hp.add(2 * CONV_DIM as usize) = bf16::from_f32(xv);
            }
            t += 1;
        }
        (s0, s1, s2) = last;
        *st = bf16::from_f32(s0);
        *st.add(CONV_DIM as usize) = bf16::from_f32(s1);
        *st.add(2 * CONV_DIM as usize) = bf16::from_f32(s2);
        return;
    }
    let mut t = 0;
    while t < n {
        // load up to 4 rows first (the stores below could alias as far as the compiler knows)
        let mut xs = [0.0f32; 4];
        kcommon::unroll!(u in 4 => {
            if t + u < n {
                xs[u] = bf16::to_f32(*x.add((r0 + t + u) * ldx as usize + c as usize));
            }
        });
        kcommon::unroll!(u in 4 => {
            if t + u < n {
                let row = r0 + t + u;
                let xv = xs[u];
                let y = bf16::round(w0 * s0 + w1 * s1 + w2 * s2 + w3 * xv);
                *out.add(row * CONV_DIM as usize + c as usize) = bf16::from_f32(math::silu(y));
                s0 = s1;
                s1 = s2;
                s2 = xv;
                if !hist.is_null() && ((t + u) as u32) < hist_rows {
                    let hp = hist.add((slot * hist_rows as usize + t + u) * 3 * CONV_DIM as usize + c as usize);
                    *hp = bf16::from_f32(s0);
                    *hp.add(CONV_DIM as usize) = bf16::from_f32(s1);
                    *hp.add(2 * CONV_DIM as usize) = bf16::from_f32(s2);
                }
            }
        });
        t += 4;
    }
    *st = bf16::from_f32(s0);
    *st.add(CONV_DIM as usize) = bf16::from_f32(s1);
    *st.add(2 * CONV_DIM as usize) = bf16::from_f32(s2);
}

/// Delta-rule recurrence + gated RMSNorm. grid = (48, nseg), block = 512 (hist: optional per-row states,
/// [slot][t][48][128][128], for speculative rollback):
/// thread t owns dv column j = t / 4 and dk rows (t % 4) * 32 .. +32 of the head's fp32 state S[dk][dv].
/// conv: [rows][10240] (post conv+silu); z rows (stride ldz) of 48 x 128; a and b rows (stride ldab) of 48;
/// state: [slot][48][128][128] f32; y: [rows][48*128] bf16.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gdn_rec(y: *mut u16, conv: *const u16, zx: *const u16, ldz: u32, ax: *const u16,
    bx: *const u16, ldab: u32, a_log: *const u16, dt_bias: *const u16, norm_w: *const u16, state: *mut f32,
    seg_slot: *const u32, seg_row0: *const u32, seg_len: *const u32, eps: f32, hist: *mut f32, hist_rows: u32) {
    let h = bid_x();
    let sg = bid_y() as usize;
    let t = tid_x();
    let j = t >> 2;
    let qtr = t & 3;
    let l = lane();
    let wid = warp_id();
    let slot = *seg_slot.add(sg) as usize;
    let r0 = *seg_row0.add(sg) as usize;
    let n = *seg_len.add(sg) as usize;
    let qs: Shared<f32> = kcommon::static_shared!(gdn_q, f32, 512);
    let ks: Shared<f32> = kcommon::static_shared!(gdn_k, f32, 512);
    let red: Shared<f32> = kcommon::static_shared!(gdn_red, f32, 128);

    let sp = state.add(((slot * NVH as usize + h as usize) * (DK * DV) as usize) + j as usize);
    let mut s = [0.0f32; 32];
    kcommon::unroll!(i in 32 => {
        s[i] = *sp.add(((qtr * 32 + i as u32) * DV) as usize);
    });
    let a_neg = -math::exp(bf16::to_f32(*a_log.add(h as usize)));
    let dtb = bf16::to_f32(*dt_bias.add(h as usize));
    let nw = bf16::to_f32(*norm_w.add(j as usize));
    let kh = h / (NVH / NKH);

    let mut tt = 0;
    while tt < n {
        let row = r0 + tt;
        let crow = conv.add(row * CONV_DIM as usize);
        // q (threads 0..128) and k (threads 128..256): l2-normalise in fp32.
        let mut v = 0.0f32;
        if t < 128 {
            v = bf16::to_f32(*crow.add((kh * DK + t) as usize));
        } else if t < 256 {
            v = bf16::to_f32(*crow.add((K_OFF + kh * DK + t - 128) as usize));
        }
        let ws = warp::sum(v * v);
        if l == 0 {
            red.set(wid, ws);
        }
        sync_threads();
        if t < 256 {
            let mut tot = 0.0;
            let base = if t < 128 { 0 } else { 4 };
            let mut q = 0;
            while q < 4 {
                tot += red.at(base + q);
                q += 1;
            }
            let inv = math::rsqrt(tot + 1e-6);
            if t < 128 {
                qs.set(t, v * inv * 0.088388347648318440550f32); // 1/sqrt(128)
            } else {
                ks.set(t - 128, v * inv);
            }
        }
        let bb = bf16::to_f32(*bx.add(row * ldab as usize + h as usize));
        let aa = bf16::to_f32(*ax.add(row * ldab as usize + h as usize));
        let beta = bf16::round(math::sigmoid(bb));
        let decay = math::exp(a_neg * math::softplus(aa + dtb));
        let vj = bf16::to_f32(*crow.add((V_OFF + h * DV + j) as usize));
        sync_threads();

        let mut kv = 0.0f32;
        kcommon::unroll!(i in 32 => {
            s[i] *= decay;
            kv = math::fma(s[i], ks.at(qtr * 32 + i as u32), kv);
        });
        kv += warp::shfl_xor(kv, 1);
        kv += warp::shfl_xor(kv, 2);
        let delta = (vj - kv) * beta;
        let mut o = 0.0f32;
        kcommon::unroll!(i in 32 => {
            s[i] = math::fma(ks.at(qtr * 32 + i as u32), delta, s[i]);
            o = math::fma(s[i], qs.at(qtr * 32 + i as u32), o);
        });
        o += warp::shfl_xor(o, 1);
        o += warp::shfl_xor(o, 2);
        // gated RMSNorm over the 128 dv columns: o -> bf16, normalise, * w (bf16), * silu(z)
        let ob = bf16::round(o);
        let sq = warp::sum(if qtr == 0 { ob * ob } else { 0.0 });
        if l == 0 {
            red.set(16 + wid, sq);
        }
        sync_threads();
        if qtr == 0 {
            let mut tot = 0.0;
            let mut q = 0;
            while q < 16 {
                tot += red.at(16 + q);
                q += 1;
            }
            let inv = math::rsqrt(tot / DV as f32 + eps);
            let xn = bf16::round(bf16::round(ob * inv) * nw);
            let z = bf16::to_f32(*zx.add(row * ldz as usize + (h * DV + j) as usize));
            *y.add(row * (NVH * DV) as usize + (h * DV + j) as usize) = bf16::from_f32(xn * math::silu(z));
        }
        if !hist.is_null() && (tt as u32) < hist_rows {
            // the state after this row: [slot][t][48][128][128]
            let hp = hist.add(((slot * hist_rows as usize + tt) * NVH as usize + h as usize) * (DK * DV) as usize + j as usize);
            kcommon::unroll!(i in 32 => {
                *hp.add(((qtr * 32 + i as u32) * DV) as usize) = s[i];
            });
        }
        tt += 1;
    }
    kcommon::unroll!(i in 32 => {
        *sp.add(((qtr * 32 + i as u32) * DV) as usize) = s[i];
    });
}

/// Delta-rule recurrence, split over value columns for parallelism. grid = (48 heads * 4 column slices, nseg),
/// block = 512: thread t owns value column j = slice * 32 + t / 16 and key rows (t % 16) * 8 .. +8 of the fp32
/// state. Writes the raw per-head output o (bf16, = HF's core_attn_out) to `o`; fe_gdn_norm applies the gated
/// RMSNorm. hist as in fe_gdn_rec. `ro` (null = none): per-segment u32 flags; a flagged segment (a speculative
/// verify in the engine's replay mode) leaves the state untouched and writes no history: Engine::commit replays
/// its kept rows from the saved inputs with this same kernel. `seg_tree` (null = none): per-segment flags; a
/// flagged segment (always read-only) is a verify tree of <= 32 rows, run in the order of `tree_steps` (per step:
/// row | leaf << 5 | (save slot + 1) << 6 | (restore slot + 1) << 10, Engine's tree_steps: depth first, a node's
/// leaf children straight from its state without advancing it, its state saved / restored only around several
/// inner children).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gdn_rec2(o: *mut u16, conv: *const u16, ax: *const u16, bx: *const u16, ldab: u32,
    a_log: *const u16, dt_bias: *const u16, state: *mut f32, seg_slot: *const u32, seg_row0: *const u32,
    seg_len: *const u32, hist: *mut f32, hist_rows: u32, ro: *const u32, tree_steps: *const u32, seg_tree: *const u32) {
    let sg = bid_y() as usize;
    let sm = (kcommon::static_shared!(gdn2_q, f32, REC_CHUNK * 128 * 4), kcommon::static_shared!(gdn2_k, f32, REC_CHUNK * 128 * 4),
        kcommon::static_shared!(gdn2_steps, u32, 32 * 4));
    if !seg_tree.is_null() && *seg_tree.add(sg) != 0 {
        rec2_body::<true>(o, conv, ax, bx, ldab, a_log, dt_bias, state, seg_slot, seg_row0, seg_len, hist, hist_rows,
            ro, tree_steps, sg, sm);
    } else {
        rec2_body::<false>(o, conv, ax, bx, ldab, a_log, dt_bias, state, seg_slot, seg_row0, seg_len, hist, hist_rows,
            ro, core::ptr::null(), sg, sm);
    }
}

/// Engine::commit's GDN replay for every GDN layer in one launch: fe_gdn_rec2 over one segment (seg_* [0]) with
/// layer z's [conv, a|b (stride ldab, b at + NVH), a_log, dt_bias, state] pointers from tab[z * 5..], no output,
/// no history. grid = (48 heads * 4 column slices, 1, layers), block = 512.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gdn_replay(tab: *const u64, ldab: u32, seg_slot: *const u32, seg_row0: *const u32,
    seg_len: *const u32) {
    let p = tab.add(5 * bid_z() as usize);
    let ab = *p.add(1) as *const u16;
    let sm = (kcommon::static_shared!(gdn2r_q, f32, REC_CHUNK * 128 * 4),
        kcommon::static_shared!(gdn2r_k, f32, REC_CHUNK * 128 * 4), Shared::<u32>::from_addr(0));
    rec2_body::<false>(core::ptr::null_mut(), *p as *const u16, ab, ab.add(NVH as usize), ldab, *p.add(2) as *const u16,
        *p.add(3) as *const u16, *p.add(4) as *mut f32, seg_slot, seg_row0, seg_len, core::ptr::null_mut(), 0,
        core::ptr::null(), core::ptr::null(), 0, sm);
}

#[inline(always)]
unsafe fn rec2_body<const TREE: bool>(o: *mut u16, conv: *const u16, ax: *const u16, bx: *const u16, ldab: u32, a_log: *const u16,
    dt_bias: *const u16, state: *mut f32, seg_slot: *const u32, seg_row0: *const u32, seg_len: *const u32,
    hist: *mut f32, hist_rows: u32, ro: *const u32, tsteps: *const u32, sg: usize,
    (qs, ks, steps): (Shared<f32>, Shared<f32>, Shared<u32>)) {
    let h = bid_x() >> 2;
    let slice = bid_x() & 3;
    let t = tid_x();
    let j = slice * 32 + (t >> 4);
    let part = t & 15;
    let l = lane();
    let wid = warp_id();
    let slot = *seg_slot.add(sg) as usize;
    let r0 = *seg_row0.add(sg) as usize;
    let n = *seg_len.add(sg) as usize;
    // a tree (TREE, read-only) runs its rows in the host's step order; saved states: a stack of node states
    // (uninitialised: zeroing it would be a stack write per thread and launch). The chain copy compiles as before.
    let tree = TREE;
    let keep_state = !tree && (ro.is_null() || *ro.add(sg) == 0);
    let mut saved = core::mem::MaybeUninit::<[[f32; 8]; TREE_SLOTS]>::uninit();
    let saved = saved.as_mut_ptr() as *mut [f32; 8];
    if tree {
        if (t as usize) < n.min(32) {
            steps.set(t, *tsteps.add(r0 + t as usize));
        }
        sync_threads();
    }
    let step_row = |k: usize| -> usize { if tree { r0 + (steps.at(k as u32) & 31) as usize } else { r0 + k } };

    let sp = state.add(((slot * NVH as usize + h as usize) * (DK * DV) as usize) + j as usize);
    let mut s = [0.0f32; 8];
    kcommon::unroll!(i in 8 => {
        s[i] = *sp.add(((part * 8 + i as u32) * DV) as usize);
    });
    let a_neg = -math::exp(bf16::to_f32(*a_log.add(h as usize)));
    let dtb = bf16::to_f32(*dt_bias.add(h as usize));
    let kh = h / (NVH / NKH);

    // rows go in chunks of REC_CHUNK: first every row's l2-normalised q and k into shared memory (warp 2c + 0/1:
    // row c's q / k, lane l holding elements l, 32 + l, 64 + l, 96 + l; the four 32-element sums and their total
    // in the order a warp per 32 elements gave them), then the chunk's recurrence without block barriers
    let qk_base = if wid & 1 == 0 { kh * DK } else { K_OFF + kh * DK };
    // this thread's per-row inputs: b, a, v_j (loaded a row ahead to hide latency)
    let load = |row: usize| -> (u16, u16, u16) {
        (*bx.add(row * ldab as usize + h as usize), *ax.add(row * ldab as usize + h as usize),
         *conv.add(row * CONV_DIM as usize + (V_OFF + h * DV + j) as usize))
    };
    let mut nxt = if n > 0 { load(step_row(0)) } else { (0, 0, 0) };
    let mut c0 = 0;
    while c0 < n {
        let m = if n - c0 < REC_CHUNK { n - c0 } else { REC_CHUNK };
        let c = (wid >> 1) as usize;
        if c < m {
            let crow = conv.add(step_row(c0 + c) * CONV_DIM as usize + qk_base as usize);
            let mut x = [0.0f32; 4];
            kcommon::unroll!(g in 4 => {
                x[g] = bf16::to_f32(*crow.add(g * 32 + l as usize));
            });
            let mut gs = [0.0f32; 4];
            kcommon::unroll!(g in 4 => {
                gs[g] = warp::sum(x[g] * x[g]);
            });
            let inv = math::rsqrt(gs[0] + gs[1] + gs[2] + gs[3] + 1e-6);
            let dst = if wid & 1 == 0 { qs } else { ks }.add((c * 128) as u32 + l);
            kcommon::unroll!(g in 4 => {
                let y = if wid & 1 == 0 { x[g] * inv * 0.088388347648318440550f32 } else { x[g] * inv }; // 1/sqrt(128)
                dst.set(g as u32 * 32, y);
            });
        }
        sync_threads();
        let mut ci = 0;
        while ci < m {
            let tt = c0 + ci;
            let row = step_row(tt);
            let st = if tree { steps.at(tt as u32) } else { 0 };
            let cur = nxt;
            if tt + 1 < n {
                nxt = load(step_row(tt + 1));
            }
            let (sv, rs) = ((st >> 6) & 15, (st >> 10) & 15);
            if sv != 0 {
                *saved.add(sv as usize - 1) = s;
            }
            if rs != 0 {
                s = *saved.add(rs as usize - 1);
            }
            let bb = bf16::to_f32(cur.0);
            let aa = bf16::to_f32(cur.1);
            let vj = bf16::to_f32(cur.2);
            let beta = bf16::round(math::sigmoid(bb));
            let decay = math::exp(a_neg * math::softplus(aa + dtb));
            let kv4 = ks.add((ci * 128) as u32 + part * 8).cast::<[f32; 4]>();
            let qv4 = qs.add((ci * 128) as u32 + part * 8).cast::<[f32; 4]>();
            let (k0, k1) = (kv4.ld(), kv4.add(1).ld());
            let kk = [k0[0], k0[1], k0[2], k0[3], k1[0], k1[1], k1[2], k1[3]];
            // u = the state after this row; a tree leaf leaves s (its parent's) for its siblings
            let mut u = s;
            let mut kv = 0.0f32;
            kcommon::unroll!(i in 8 => {
                u[i] *= decay;
                kv = math::fma(u[i], kk[i], kv);
            });
            kv += warp::shfl_xor(kv, 1);
            kv += warp::shfl_xor(kv, 2);
            kv += warp::shfl_xor(kv, 4);
            kv += warp::shfl_xor(kv, 8);
            let delta = (vj - kv) * beta;
            let (q0, q1) = (qv4.ld(), qv4.add(1).ld());
            let qq = [q0[0], q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3]];
            let mut ov = 0.0f32;
            kcommon::unroll!(i in 8 => {
                u[i] = math::fma(kk[i], delta, u[i]);
                ov = math::fma(u[i], qq[i], ov);
            });
            if (st >> 5) & 1 == 0 {
                s = u;
            }
            ov += warp::shfl_xor(ov, 1);
            ov += warp::shfl_xor(ov, 2);
            ov += warp::shfl_xor(ov, 4);
            ov += warp::shfl_xor(ov, 8);
            if part == 0 && !o.is_null() {
                *o.add(row * (NVH * DV) as usize + (h * DV + j) as usize) = bf16::from_f32(ov);
            }
            if keep_state && !hist.is_null() && (tt as u32) < hist_rows {
                let hp = hist.add(((slot * hist_rows as usize + tt) * NVH as usize + h as usize) * (DK * DV) as usize + j as usize);
                kcommon::unroll!(i in 8 => {
                    *hp.add(((part * 8 + i as u32) * DV) as usize) = s[i];
                });
            }
            ci += 1;
        }
        // the next chunk's q / k overwrite shared memory only after every warp finished this one
        sync_threads();
        c0 += m;
    }
    if keep_state {
        kcommon::unroll!(i in 8 => {
            *sp.add(((part * 8 + i as u32) * DV) as usize) = s[i];
        });
    }
}

/// Gated RMSNorm of the GDN output (HF Qwen3_5RMSNormGated): y = bf16(bf16(w * bf16(o * rsqrt(mean(o^2)+eps)))
/// * silu(z)). grid = (rows, 2), block = 24 * 32 (warp per head: 4 values a lane).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gdn_norm(y: *mut u16, o: *const u16, zx: *const u16, ldz: u32, norm_w: *const u16,
    eps: f32) {
    let r = bid_x() as usize;
    let h = bid_y() * (NVH / 2) + warp_id();
    let l = lane();
    let base = r * (NVH * DV) as usize + (h * DV + l * 4) as usize;
    let ov = *(o.add(base) as *const [u32; 2]);
    let x = [bf16::lo(ov[0]), bf16::hi(ov[0]), bf16::lo(ov[1]), bf16::hi(ov[1])];
    let ss = warp::sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
    let inv = math::rsqrt(ss / DV as f32 + eps);
    let wv = *(norm_w.add((l * 4) as usize) as *const [u32; 2]);
    let w = [bf16::lo(wv[0]), bf16::hi(wv[0]), bf16::lo(wv[1]), bf16::hi(wv[1])];
    let zv = *(zx.add(r * ldz as usize + (h * DV + l * 4) as usize) as *const [u32; 2]);
    let z = [bf16::lo(zv[0]), bf16::hi(zv[0]), bf16::lo(zv[1]), bf16::hi(zv[1])];
    let mut out = [0.0f32; 4];
    kcommon::unroll!(i in 4 => {
        let xn = bf16::round(bf16::round(x[i] * inv) * w[i]);
        out[i] = xn * math::silu(z[i]);
    });
    *(y.add(base) as *mut [u32; 2]) = [bf16::pack2(out[0], out[1]), bf16::pack2(out[2], out[3])];
}
