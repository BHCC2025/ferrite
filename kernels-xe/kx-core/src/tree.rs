//! Tree verify windows (draft trees, DFlash2): the GDN and helper kernels for a window whose rows form a tree instead of
//! a chain. Rows are in topological order (par[t] < t; row 0 the root, par = -1: it continues the committed sequence),
//! row t sits at position pos0 + depth[t]. Every output of row t is bit-identical to decoding the root -> t path alone,
//! one row at a time, with the linear window kernels (rows.rs): the same arithmetic in the same order, only the inputs
//! (conv taps, delta-rule state) come from the row's ancestors instead of the previous row.
//!
//! The tree table (host-built, fe-kxcore `TreeTable`): u32 [rows][4] =
//!   [0] par as i32 (-1: the committed state), [1] depth, [2] ancestor mask over window rows (bit t = row t, incl. itself),
//!   [3] delta-rule state slots for gdn_step_tr: bits 0..7 = slot to save the state after this row into (0xff: none),
//!       bits 8..15 = where the state before this row comes from (0xff: the registers, i.e. par == t - 1; 0xfe: s_in, the
//!       committed state; else a slot saved earlier).
use crate::ext::{dot4, vload8, vstore8};
use crate::{bf, bfr, silu, to_bf};
use kxcommon::{attr, id, math, sg, slm, unroll};

const DK: usize = 128;
const DV: usize = 128;
const KH: usize = 16;
const VH: usize = 48;
/// bytes of one position of a q8 KV cache (4 kv heads x 272)
const RW: usize = 4 * 272;
/// keys of a row's tail window (kv_tree_tail): from kbase = pos0 & !31, covers depths up to 32
pub const TAIL_KEYS: usize = 64;

#[inline(always)]
unsafe fn par_of(tree: *const u32, t: u32) -> i32 {
    *tree.add(4 * t as usize) as i32
}

/// Extended-input index of the k-th predecessor along the branch: ext(0..2) = the conv state rows (oldest first),
/// ext(3 + r) = window row r. pred(3 + r) = 3 + par[r] (2 when par = -1); pred(j) = j - 1 for a state row.
#[inline(always)]
unsafe fn pred(tree: *const u32, e: u32) -> u32 {
    if e >= 3 {
        let p = par_of(tree, e - 3);
        if p >= 0 { 3 + p as u32 } else { 2 }
    } else {
        e - 1
    }
}

#[inline(always)]
unsafe fn ext_at(qkv: *const u16, state: *const u16, cd: u32, ch: u32, j: u32) -> u16 {
    if j < 3 { *state.add((j * cd + ch) as usize) } else { *qkv.add((j - 3) as usize * cd as usize + ch as usize) }
}

/// Conv + silu of channel ch for tree row t: taps over [3rd, 2nd, 1st predecessor, row t] along the branch (the
/// arithmetic of rows.rs conv_t).
#[inline(always)]
unsafe fn conv_tree(qkv: *const u16, state: *const u16, cw: *const u16, cd: u32, ch: u32, e: [u32; 4]) -> u16 {
    let w = cw.add(4 * ch as usize);
    let (w0, w1, w2, w3) = (*w, *w.add(1), *w.add(2), *w.add(3));
    let mut acc = bf(ext_at(qkv, state, cd, ch, e[0])) * bf(w0);
    acc = math::mad(bf(ext_at(qkv, state, cd, ch, e[1])), bf(w1), acc);
    acc = math::mad(bf(ext_at(qkv, state, cd, ch, e[2])), bf(w2), acc);
    acc = math::mad(bf(ext_at(qkv, state, cd, ch, e[3])), bf(w3), acc);
    to_bf(silu(acc))
}

#[inline(always)]
pub(crate) unsafe fn taps(tree: *const u32, t: u32) -> [u32; 4] {
    let e3 = 3 + t;
    let e2 = pred(tree, e3);
    let e1 = pred(tree, e2);
    let e0 = pred(tree, e1);
    [e0, e1, e2, e3]
}

/// gdn_conv_prep_r for a tree window (q / k channels of row get_group_id(1)). Grid (16, rows), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_conv_prep_tr(qkv: *const u16, state: *const u16, cw: *const u16, qn: *mut f32, kn: *mut f32, cd: u32, tree: *const u32) {
    let h = id::group_id(0);
    let t = id::group_id(1);
    crate::gdnpre::prep_head(qkv, state, cw, cd, taps(tree, t), h, t, qn, kn);
}

/// gdn_vconv_r for a tree window. Grid (rows, 96), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_vconv_tr(qkv: *const u16, state: *const u16, cw: *const u16, vout: *mut u16, cd: u32, tree: *const u32) {
    attr::sg_size::<32>();
    let t = id::group_id(0);
    let c = id::group_id(1) * 64 + id::local_id(0);
    if c < (VH * DV) as u32 {
        let e = taps(tree, t);
        *vout.add(t as usize * (VH * DV) + c as usize) = conv_tree(qkv, state, cw, cd, (2 * KH * DK) as u32 + c, e);
    }
}

#[inline(always)]
fn lo(v: &[f32; 8]) -> [f32; 4] {
    [v[0], v[1], v[2], v[3]]
}
#[inline(always)]
fn hi(v: &[f32; 8]) -> [f32; 4] {
    [v[4], v[5], v[6], v[7]]
}

/// Delta rule over a tree window (rows.rs gdn_step_r2x4 with the state of row t = the state after par[t]): rows in
/// order; the state before a row comes from the registers (par == t - 1), from s_in (par == -1), or from a slot of the
/// scratch sstate [slots][48][128][128] fp32 saved after an earlier row (tree table [3]). Writes y only: a tree commit
/// replays the accepted path with the linear kernels. Grid (48, 8), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_step_tr(
    qn: *const f32,
    kn: *const f32,
    vconv: *const u16,
    beta_a: *const f32,
    g_a: *const f32,
    s_in: *const f32,
    sscr: *mut f32,
    y: *mut u16,
    tiled: u32,
    rows: u32,
    tree: *const u32,
) {
    const NR: usize = 4;
    let h = id::group_id(0) as usize;
    let row0 = (id::group_id(1) * 4 + id::sg_id()) as usize * NR;
    let lane = id::lane() as usize;
    let kh = if tiled != 0 { h % KH } else { h / (VH / KH) };
    let off = (h * DV + row0) * DK;
    let slot_sz = VH * DV * DK;
    let mut s = [[0f32; 8]; NR];
    unroll!(4, NR, |r| s[r] = vload8(lane, s_in.add(off + r * DK)));
    let mut t = 0u32;
    while t < rows {
        let tb = t as usize;
        let f3 = *tree.add(4 * tb + 3);
        let restore = (f3 >> 8) & 0xff;
        if restore == 0xfe {
            unroll!(4, NR, |r| s[r] = vload8(lane, s_in.add(off + r * DK)));
        } else if restore != 0xff {
            let base = sscr.add(restore as usize * slot_sz + off);
            unroll!(4, NR, |r| s[r] = vload8(lane, base.add(r * DK)));
        }
        let k8 = vload8(lane, kn.add(tb * (KH * DK) + kh * DK));
        let q8 = vload8(lane, qn.add(tb * (KH * DK) + kh * DK));
        let g = *g_a.add(tb * VH + h);
        let beta = *beta_a.add(tb * VH + h);
        let mut vr = [0f32; NR];
        unroll!(4, NR, |r| vr[r] = bf(*vconv.add(tb * (VH * DV) + h * DV + row0 + r)));
        let mut kv = [0f32; NR];
        unroll!(4, NR, |r| {
            unroll!(8, 8, |j| s[r][j] *= g);
            kv[r] = sg::reduce_add_f32(dot4(lo(&s[r]), lo(&k8)) + dot4(hi(&s[r]), hi(&k8)));
        });
        unroll!(4, NR, |r| {
            let dl = (vr[r] - kv[r]) * beta;
            unroll!(8, 8, |j| s[r][j] = math::mad(k8[j], dl, s[r][j]));
        });
        let mut o = [0f32; NR];
        unroll!(4, NR, |r| o[r] = sg::reduce_add_f32(dot4(lo(&s[r]), lo(&q8)) + dot4(hi(&s[r]), hi(&q8))));
        unroll!(4, NR, |r| {
            if lane == 0 {
                *y.add(tb * (VH * DV) + h * DV + row0 + r) = to_bf(o[r]);
            }
        });
        let save = f3 & 0xff;
        if save != 0xff {
            let base = sscr.add(save as usize * slot_sz + off);
            unroll!(4, NR, |r| vstore8(s[r], lane, base.add(r * DK)));
        }
        t += 1;
    }
}

/// Per-row tail windows of the q8 K and V caches for the tree decode attention: for row t (get_group_id(1)), keys
/// kbase .. kbase + 63: logical key k < pos0 -> the cache position k; pos0 <= k <= pos0 + depth[t] -> the cache slot
/// pos0 + (row of the ancestor of t at depth k - pos0); later keys -> zeros (as the 2D loads read past a row's end).
/// tk / tv [rows][64][4 kv heads][272 B]. One work-item a dword of a position. Grid (64 keys, rows), local 272.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kv_tree_tail(kc: *const u32, vc: *const u32, tk: *mut u32, tv: *mut u32, tree: *const u32, pos0: u32, kbase: u32) {
    let j = id::group_id(0);
    let t = id::group_id(1);
    let d = id::local_id(0) as usize; // dword of the position (272 = 1088 B)
    let k = kbase + j;
    let src = if k < pos0 {
        k as i64
    } else {
        let e = k - pos0;
        let depth = *tree.add(4 * t as usize + 1);
        if e > depth {
            -1
        } else {
            // the e-th set bit of the ancestor mask (ancestors are in increasing row order along the branch)
            let mut mask = *tree.add(4 * t as usize + 2);
            let mut n = 0u32;
            let mut r = 0i64;
            let mut b = 0u32;
            while b < 32 {
                if mask & 1 != 0 {
                    if n == e {
                        r = b as i64;
                    }
                    n += 1;
                }
                mask >>= 1;
                b += 1;
            }
            pos0 as i64 + r
        }
    };
    let dst = (t as usize * TAIL_KEYS + j as usize) * (RW / 4) + d;
    if src >= 0 {
        *tk.add(dst) = *kc.add(src as usize * (RW / 4) + d);
        *tv.add(dst) = *vc.add(src as usize * (RW / 4) + d);
    } else {
        *tk.add(dst) = 0;
        *tv.add(dst) = 0;
    }
}

/// After a tree verify: the accepted path p_0 = 0, p_1 .. p_{n-1} (window rows) -> cache positions pos0 .. pos0 + n - 1,
/// for the K and the V cache (records copied whole; every source is read before any write). One work-group.
/// Grid (1), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kv_tree_commit(kc: *mut u32, vc: *mut u32, path: *const u32, n: u32, pos0: u32) {
    attr::reqd_work_group_size::<256, 1, 1>();
    let buf = slm::alloc::<u32, { 2 * 16 * (RW / 4) }, 0>(); // n <= 16
    let lid = id::local_id(0) as usize;
    let tot = n as usize * (RW / 4);
    let mut i = lid;
    while i < tot {
        let (e, d) = (i / (RW / 4), i % (RW / 4));
        let s = (pos0 as usize + *path.add(e) as usize) * (RW / 4) + d;
        *buf.add(i) = *kc.add(s);
        *buf.add(16 * (RW / 4) + i) = *vc.add(s);
        i += 256;
    }
    slm::barrier_all();
    let mut i = lid;
    while i < tot {
        let dd = pos0 as usize * (RW / 4) + i;
        *kc.add(dd) = *buf.add(i);
        *vc.add(dd) = *buf.add(16 * (RW / 4) + i);
        i += 256;
    }
}

/// Row gather of 16-bit rows: dst[r] = src[idx[r]] (row_elems u16 a row), e.g. the saved GDN inputs of an accepted tree
/// path before the commit replay. Grid (rows, ceil(row_elems / 256)), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rows_gather_u16(src: *const u16, dst: *mut u16, idx: *const u32, row_elems: u32) {
    let r = id::group_id(0) as usize;
    let c = (id::group_id(1) * 256 + id::local_id(0)) as usize;
    if c < row_elems as usize {
        *dst.add(r * row_elems as usize + c) = *src.add(*idx.add(r) as usize * row_elems as usize + c);
    }
}

/// attn_long_merge for a tree window: row z has pos0 + depth[z] + 1 keys. Grid (24, rows), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn attn_long_merge_tr(po: *const f32, pm: *const f32, pl: *const f32, qg: *const u16, out: *mut u16, pos0: u32, maxc: u32, tree: *const u32) {
    attr::sg_size::<32>();
    attr::reqd_work_group_size::<256, 1, 1>();
    let z = id::group_id(1);
    let len = pos0 + *tree.add(4 * z as usize + 1) + 1;
    crate::long::merge_row(po, pm, pl, qg, out, len, z, maxc);
}
