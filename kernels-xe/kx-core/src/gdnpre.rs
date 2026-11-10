//! The GDN window chain before the delta rule, faster with the same bits as rows.rs / tree.rs:
//! * the q / k conv + RMS prep of a sub-group (one key head, 8 channels a lane) with vector loads: the 4 taps of the
//!   lane's 8 channels as four 16-byte vload8 and its 32 conv weights as two vload16, instead of 64 single 16-bit loads
//!   (the arithmetic per channel is conv_t's: tap 0 product, then three mads, silu, bf16; the sum of squares and the norm
//!   as gdn_conv_prep_r);
//! * gdn_pre_r / gdn_pre_tr: q / k prep + v conv + the gates of a window in ONE launch (three before), work-groups by
//!   role: x < 4 the q / k prep of key heads 4 x .. 4 x + 3 (a sub-group each), 4 <= x < 100 64 v channels, x == 100
//!   the 48 heads' beta and g (launch 100 groups to leave the gates to another kernel).
use crate::ext::{log1p, vload16_u16, vload8_u16};
use crate::{bf, bfr, silu, to_bf};
use kxcommon::{id, math, sg, unroll};

const DK: usize = 128;
const DV: usize = 128;
const KH: usize = 16;
const VH: usize = 48;

#[inline(always)]
unsafe fn ext_row(qkv: *const u16, state: *const u16, cd: u32, e: u32) -> *const u16 {
    if e < 3 { state.add((e * cd) as usize) } else { qkv.add((e - 3) as usize * cd as usize) }
}

/// The q / k prep of key head h for one window row (its taps e: ext indices oldest first; rows.rs ext numbering):
/// qn / kn rows at `t`.
#[inline(always)]
pub(crate) unsafe fn prep_head(qkv: *const u16, state: *const u16, cw: *const u16, cd: u32, e: [u32; 4], h: u32, t: u32, qn: *mut f32, kn: *mut f32) {
    let lane = id::lane();
    let rp = [ext_row(qkv, state, cd, e[0]), ext_row(qkv, state, cd, e[1]), ext_row(qkv, state, cd, e[2]), ext_row(qkv, state, cd, e[3])];
    unroll!(2, 2, |wi| {
        let which = wi as u32;
        let base = (which * (KH * DK) as u32 + h * DK as u32 + lane * 8) as usize;
        let mut x = [[0u16; 8]; 4];
        unroll!(4, 4, |k| x[k] = vload8_u16(0, rp[k].add(base)));
        let w0 = vload16_u16(0, cw.add(4 * base));
        let w1 = vload16_u16(1, cw.add(4 * base));
        let mut v = [0f32; 8];
        let mut ss = 0.0f32;
        unroll!(8, 8, |j| {
            let w = |k: usize| -> u16 { if 4 * j + k < 16 { w0[4 * j + k] } else { w1[4 * j + k - 16] } };
            let mut acc = bf(x[0][j]) * bf(w(0));
            acc = math::mad(bf(x[1][j]), bf(w(1)), acc);
            acc = math::mad(bf(x[2][j]), bf(w(2)), acc);
            acc = math::mad(bf(x[3][j]), bf(w(3)), acc);
            v[j] = bf(to_bf(silu(acc)));
            ss = math::mad(v[j], v[j], ss);
        });
        ss = sg::reduce_add_f32(ss);
        let r = math::rsqrt(math::div_ocl(ss, DK as f32) + 1e-6);
        let sc = if which == 0 { 1.0 / DK as f32 } else { math::div_ocl(1.0, math::sqrt(DK as f32)) };
        let dst = if which == 0 { qn } else { kn }.add(t as usize * (KH * DK) + h as usize * DK + lane as usize * 8);
        unroll!(8, 8, |j| *dst.add(j) = bfr(v[j] * r * sc));
    });
}

/// v conv of channel 4096 + c for one window row (taps e), rows.rs conv_t arithmetic.
#[inline(always)]
unsafe fn vconv_one(qkv: *const u16, state: *const u16, cw: *const u16, cd: u32, e: [u32; 4], c: u32) -> u16 {
    let ch = (2 * KH * DK) as u32 + c;
    let w = cw.add(4 * ch as usize);
    let at = |k: usize| -> f32 { bf(*ext_row(qkv, state, cd, e[k]).add(ch as usize)) };
    let mut acc = at(0) * bf(*w);
    acc = math::mad(at(1), bf(*w.add(1)), acc);
    acc = math::mad(at(2), bf(*w.add(2)), acc);
    acc = math::mad(at(3), bf(*w.add(3)), acc);
    to_bf(silu(acc))
}

/// beta / g of window row t, head h (rows.rs gdn_gates_r).
#[inline(always)]
unsafe fn gates_one(bp: *const u16, ap: *const u16, a_log: *const f32, dt_bias: *const f32, beta_o: *mut f32, g_o: *mut f32, t: usize, h: usize) {
    *beta_o.add(t * VH + h) = math::div_ocl(1.0, 1.0 + math::exp(-bf(*bp.add(t * VH + h))));
    let av = bf(*ap.add(t * VH + h)) + *dt_bias.add(h);
    let sp = if av > 20.0 { av } else { log1p(math::exp(av)) };
    *g_o.add(t * VH + h) = math::exp(-math::exp(*a_log.add(h)) * sp);
}

#[inline(always)]
unsafe fn pre_body(
    qkv: *const u16,
    state: *const u16,
    cw: *const u16,
    qn: *mut f32,
    kn: *mut f32,
    vout: *mut u16,
    cd: u32,
    bp: *const u16,
    ap: *const u16,
    a_log: *const f32,
    dt_bias: *const f32,
    beta_o: *mut f32,
    g_o: *mut f32,
    t: u32,
    e: [u32; 4],
) {
    let x = id::group_id(0);
    if x < 4 {
        prep_head(qkv, state, cw, cd, e, 4 * x + id::sg_id(), t, qn, kn);
    } else if x < 100 {
        let c = (x - 4) * 64 + id::local_id(0);
        *vout.add(t as usize * (VH * DV) + c as usize) = vconv_one(qkv, state, cw, cd, e, c);
    } else if id::local_id(0) < VH as u32 {
        gates_one(bp, ap, a_log, dt_bias, beta_o, g_o, t as usize, id::local_id(0) as usize);
    }
}

/// gdn_conv_prep_r + gdn_vconv_r + gdn_gates_r of a linear window in one launch. Grid (101, rows) (100: no gates),
/// local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_pre_r(
    qkv: *const u16,
    state: *const u16,
    cw: *const u16,
    qn: *mut f32,
    kn: *mut f32,
    vout: *mut u16,
    cd: u32,
    bp: *const u16,
    ap: *const u16,
    a_log: *const f32,
    dt_bias: *const f32,
    beta_o: *mut f32,
    g_o: *mut f32,
) {
    let t = id::group_id(1);
    pre_body(qkv, state, cw, qn, kn, vout, cd, bp, ap, a_log, dt_bias, beta_o, g_o, t, [t, t + 1, t + 2, t + 3]);
}

/// gdn_pre_r for a tree window (taps along each row's predecessor chain, tree.rs). Grid (101, rows), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_pre_tr(
    qkv: *const u16,
    state: *const u16,
    cw: *const u16,
    qn: *mut f32,
    kn: *mut f32,
    vout: *mut u16,
    cd: u32,
    bp: *const u16,
    ap: *const u16,
    a_log: *const f32,
    dt_bias: *const f32,
    beta_o: *mut f32,
    g_o: *mut f32,
    tree: *const u32,
) {
    let t = id::group_id(1);
    pre_body(qkv, state, cw, qn, kn, vout, cd, bp, ap, a_log, dt_bias, beta_o, g_o, t, crate::tree::taps(tree, t));
}
