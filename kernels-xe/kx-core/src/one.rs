//! One-token (plain decode step / MTP step) forms: Rust port of arc-engine `kernels/xpu/qwen_gdn.cl` (gdn_conv_prep,
//! gdn_step with the gates and the v-channel conv fused in, gdn_gate_norm), `qwen_attn.cl` attn_prep, and the one-row
//! norms of `qwen_basic.cl` (add_rmsnorm, rmsnorm512 with their static 5120 path, add_bf16). Bit-identical to the
//! reference kernels and, row for row, to the window forms of rows.rs (same arithmetic, same reduction order).
//!
//! Contraction sites of the reference (qwen_gdn.spv, qwen_attn.spv, qwen_basic.spv): conv_ch three `mad`; the q/k and
//! gated-norm sums of squares `mad`; gdn_step `mad(k8, delta, s)`; attn_prep RoPE `mad` x 2; sumsq_5120 / sumsq_ordered
//! `fma` (source fma).
use crate::ext::{dot4, log1p, vload8, vstore8};
use crate::{bf, bfr, silu, to_bf};
use kxcommon::{attr, id, math, sg, slm, unroll};

const DK: usize = 128;
const DV: usize = 128;
const KH: usize = 16;
const VH: usize = 48;
const HD: usize = 256;
const NH: usize = 24;
const RD: usize = 32;

// ---- Gated DeltaNet decode step ----

/// One conv channel over [state0, state1, state2, qkv[ch]] (tap 0 = oldest), silu, bf16 bits.
#[inline(always)]
unsafe fn conv_ch(qkv: *const u16, state: *const u16, cw: *const u16, cd: u32, ch: u32) -> u16 {
    let w = cw.add(4 * ch as usize);
    let (w0, w1, w2, w3) = (*w, *w.add(1), *w.add(2), *w.add(3));
    let (c, d) = (ch as usize, cd as usize);
    let mut acc = bf(*state.add(c)) * bf(w0);
    acc = math::mad(bf(*state.add(d + c)), bf(w1), acc);
    acc = math::mad(bf(*state.add(2 * d + c)), bf(w2), acc);
    acc = math::mad(bf(*qkv.add(c)), bf(w3), acc);
    to_bf(silu(acc))
}

/// Shift the conv state of one channel by the new input.
#[inline(always)]
unsafe fn conv_shift(qkv: *const u16, state: *mut u16, cd: u32, ch: u32) {
    let (c, d) = (ch as usize, cd as usize);
    *state.add(c) = *state.add(d + c);
    *state.add(d + c) = *state.add(2 * d + c);
    *state.add(2 * d + c) = *qkv.add(c);
}

/// Conv + silu of the q and k channels, q = bf16(rms(q) / Dk), k = bf16(rms(k) / sqrt(Dk)) per key head, widened to
/// fp32 into qn / kn [2048]; shifts the conv state of those channels. Grid (16), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_conv_prep(qkv: *const u16, state: *mut u16, cw: *const u16, qn: *mut f32, kn: *mut f32, cd: u32) {
    let h = id::group_id(0);
    let lane = id::lane();
    unroll!(2, 2, |wi| {
        let which = wi as u32;
        let base = which * (KH * DK) as u32 + h * DK as u32 + lane * 8;
        let mut v = [0f32; 8];
        let mut ss = 0.0f32;
        unroll!(8, 8, |j| {
            v[j] = bf(conv_ch(qkv, state, cw, cd, base + j as u32));
            ss = math::mad(v[j], v[j], ss);
        });
        unroll!(8, 8, |j| conv_shift(qkv, state, cd, base + j as u32));
        ss = sg::reduce_add_f32(ss);
        let r = math::rsqrt(math::div_ocl(ss, DK as f32) + 1e-6);
        let sc = if which == 0 { 1.0 / DK as f32 } else { math::div_ocl(1.0, math::sqrt(DK as f32)) };
        let dst = if which == 0 { qn } else { kn }.add(h as usize * DK + lane as usize * 8);
        unroll!(8, 8, |j| *dst.add(j) = bfr(v[j] * r * sc));
    });
}

#[inline(always)]
fn lo(v: &[f32; 8]) -> [f32; 4] {
    [v[0], v[1], v[2], v[3]]
}
#[inline(always)]
fn hi(v: &[f32; 8]) -> [f32; 4] {
    [v[4], v[5], v[6], v[7]]
}

/// One delta-rule step for value head get_group_id(0), rows 4 * get_group_id(1) + sub-group: beta and g from b / a, the
/// v channel of the row by lane 0 (conv + silu, state shift), S = S g; kv = S k; S += k (v - kv) beta; y = S q.
/// Grid (48, 32), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_step(
    qn: *const f32,
    kn: *const f32,
    qkv: *const u16,
    cstate: *mut u16,
    cw: *const u16,
    bp: *const u16,
    ap: *const u16,
    a_log: *const f32,
    dt_bias: *const f32,
    state: *mut f32,
    y: *mut u16,
    tiled: u32,
    cd: u32,
) {
    let h = id::group_id(0) as usize;
    let row = (id::group_id(1) * 4 + id::sg_id()) as usize;
    let lane = id::lane();
    let kh = if tiled != 0 { h % KH } else { h / (VH / KH) };
    let beta = math::div_ocl(1.0, 1.0 + math::exp(-bf(*bp.add(h))));
    let av = bf(*ap.add(h)) + *dt_bias.add(h);
    let sp = if av > 20.0 { av } else { log1p(math::exp(av)) };
    let g = math::exp(-math::exp(*a_log.add(h)) * sp);
    let k8 = vload8(lane as usize, kn.add(kh * DK));
    let q8 = vload8(lane as usize, qn.add(kh * DK));
    let srow = state.add((h * DV + row) * DK);
    let mut s = vload8(lane as usize, srow);
    unroll!(8, 8, |j| s[j] *= g);
    let kv = sg::reduce_add_f32(dot4(lo(&s), lo(&k8)) + dot4(hi(&s), hi(&k8)));
    let mut vc = 0.0f32;
    if lane == 0 {
        let ch = (2 * KH * DK + h * DV + row) as u32;
        vc = bf(conv_ch(qkv, cstate, cw, cd, ch));
        conv_shift(qkv, cstate, cd, ch);
    }
    let vc = sg::broadcast_f32(vc, 0);
    let delta = (vc - kv) * beta;
    unroll!(8, 8, |j| s[j] = math::mad(k8[j], delta, s[j]));
    vstore8(s, lane as usize, srow);
    let o = sg::reduce_add_f32(dot4(lo(&s), lo(&q8)) + dot4(hi(&s), hi(&q8)));
    if lane == 0 {
        *y.add(h * DV + row) = to_bf(o);
    }
}

/// out = bf16(silu(z) * (y * rsqrt(mean(y^2) + eps) * w)) for value head get_group_id(0). Grid (48), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_gate_norm(y: *const u16, z: *const u16, w: *const u16, out: *mut u16, eps: f32) {
    let h = id::group_id(0) as usize;
    let lane = id::lane() as usize;
    let mut v = [0f32; 8];
    let mut ss = 0.0f32;
    unroll!(8, 8, |j| {
        v[j] = bf(*y.add(h * DV + lane * 8 + j));
        ss = math::mad(v[j], v[j], ss);
    });
    ss = sg::reduce_add_f32(ss);
    let r = math::rsqrt(math::div_ocl(ss, DV as f32) + eps);
    unroll!(8, 8, |j| {
        let d = lane * 8 + j;
        *out.add(h * DV + d) = to_bf(silu(bf(*z.add(h * DV + d))) * (v[j] * r * bf(*w.add(d))));
    });
}

// ---- full attention ----

/// q / k RMSNorm + RoPE of one token: heads 0..23 from qg[head][0..255] (rows of 512: query then gate), 24..27 from
/// kraw; rope = [cos(32) | sin(32)] of the position; queries to qo [24][256], keys to kc[k_off ..]. Grid (28), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn attn_prep(
    qg: *const u16,
    kraw: *const u16,
    qw: *const u16,
    kw: *const u16,
    rope: *const f32,
    qo: *mut u16,
    kc: *mut u16,
    k_off: u32,
    eps: f32,
) {
    let part = slm::alloc::<f32, 16, 0>();
    let nv = slm::alloc::<f32, 256, 1>();
    let head = id::group_id(0) as usize;
    let i = id::local_id(0) as usize;
    let is_q = head < NH;
    let x = if is_q { bf(*qg.add(head * 2 * HD + i)) } else { bf(*kraw.add((head - NH) * HD + i)) };
    let ss = sg::reduce_add_f32(x * x);
    if id::lane() == 0 {
        *part.add(id::sg_id() as usize) = ss;
    }
    slm::barrier();
    let mut total = 0.0f32;
    unroll!(16, 16, |j| total += *part.add(j));
    let r = math::rsqrt(math::div_ocl(total, HD as f32) + eps);
    *nv.add(i) = bfr(x * r * bf(if is_q { *qw.add(i) } else { *kw.add(i) }));
    slm::barrier();
    let mut o = *nv.add(i);
    if i < RD {
        o = math::mad(*nv.add(i), *rope.add(i), -(*nv.add(i + RD) * *rope.add(RD + i)));
    } else if i < 2 * RD {
        o = math::mad(*nv.add(i), *rope.add(i - RD), *nv.add(i - RD) * *rope.add(RD + i - RD));
    }
    if is_q {
        *qo.add(head * HD + i) = to_bf(o);
    } else {
        *kc.add(k_off as usize + (head - NH) * HD + i) = to_bf(o);
    }
}

// ---- one-row norms (qwen_basic.cl) ----

/// Sum of squares of vals[0..5120) in the 64-item order with static trip counts (qwen_basic.cl sumsq_5120): lane L of
/// items 0..63 adds L, L + 64, .. (80 terms) with fma, the sub-group reduce, the four parts left to right.
#[inline(always)]
pub(crate) unsafe fn sumsq_5120(vals: *const f32, part: *mut f32) -> f32 {
    let lid = id::local_id(0) as usize;
    if lid < 64 {
        let mut ss = 0.0f32;
        let mut k = 0;
        while k < 80 {
            unroll!(16, 16, |u| {
                let f = *vals.add(lid + 64 * (k + u));
                ss = math::fma(f, f, ss);
            });
            k += 16;
        }
        ss = sg::reduce_add_f32(ss);
        if id::lane() == 0 {
            *part.add(id::sg_id() as usize) = ss;
        }
    }
    slm::barrier();
    *part + *part.add(1) + *part.add(2) + *part.add(3)
}

/// qwen_basic.cl sumsq_ordered (any n <= 5120).
#[inline(always)]
unsafe fn sumsq_ordered(vals: *const f32, n: u32, part: *mut f32) -> f32 {
    let lid = id::local_id(0);
    if lid < 64 {
        let mut ss = 0.0f32;
        let mut i = lid;
        while i < n {
            let f = *vals.add(i as usize);
            ss = math::fma(f, f, ss);
            i += 64;
        }
        ss = sg::reduce_add_f32(ss);
        if id::lane() == 0 {
            *part.add(id::sg_id() as usize) = ss;
        }
    }
    slm::barrier();
    *part + *part.add(1) + *part.add(2) + *part.add(3)
}

/// One row: x = bf16(x + d); y = bf16(x * rsqrt(mean(x^2) + eps) * w) (n <= 5120). Grid (1), local 512.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn add_rmsnorm(x: *mut u16, d: *const u16, w: *const u16, y: *mut u16, n: u32, eps: f32) {
    attr::reqd_work_group_size::<512, 1, 1>();
    let part = slm::alloc::<f32, 4, 0>();
    let vals = slm::alloc::<f32, 5120, 1>();
    let lid = id::local_id(0) as usize;
    if n == 5120 {
        // static trip counts: every load of a work-item is issued before it is used (same values, same sums)
        let (mut xa, mut da, mut wa) = ([0u16; 10], [0u16; 10], [0u16; 10]);
        unroll!(16, 10, |k| {
            xa[k] = *x.add(lid + 512 * k);
            da[k] = *d.add(lid + 512 * k);
            wa[k] = *w.add(lid + 512 * k);
        });
        unroll!(16, 10, |k| {
            let v = to_bf(bf(xa[k]) + bf(da[k]));
            *x.add(lid + 512 * k) = v;
            *vals.add(lid + 512 * k) = bf(v);
        });
        slm::barrier();
        let scale = math::rsqrt(math::div_ocl(sumsq_5120(vals, part), 5120.0) + eps);
        unroll!(16, 10, |k| *y.add(lid + 512 * k) = to_bf(*vals.add(lid + 512 * k) * scale * bf(wa[k])));
        return;
    }
    let mut i = lid as u32;
    while i < n {
        let v = to_bf(bf(*x.add(i as usize)) + bf(*d.add(i as usize)));
        *x.add(i as usize) = v;
        *vals.add(i as usize) = bf(v);
        i += 512;
    }
    slm::barrier();
    let scale = math::rsqrt(math::div_ocl(sumsq_ordered(vals, n, part), n as f32) + eps);
    let mut i = lid as u32;
    while i < n {
        *y.add(i as usize) = to_bf(*vals.add(i as usize) * scale * bf(*w.add(i as usize)));
        i += 512;
    }
}

/// RMSNorm of one row (n <= 5120), no residual add. Grid (1), local 512.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rmsnorm512(x: *mut u16, w: *const u16, y: *mut u16, n: u32, eps: f32) {
    attr::reqd_work_group_size::<512, 1, 1>();
    let part = slm::alloc::<f32, 4, 0>();
    let vals = slm::alloc::<f32, 5120, 1>();
    let lid = id::local_id(0) as usize;
    if n == 5120 {
        let (mut xa, mut wa) = ([0u16; 10], [0u16; 10]);
        unroll!(16, 10, |k| {
            xa[k] = *x.add(lid + 512 * k);
            wa[k] = *w.add(lid + 512 * k);
        });
        unroll!(16, 10, |k| *vals.add(lid + 512 * k) = bf(xa[k]));
        slm::barrier();
        let scale = math::rsqrt(math::div_ocl(sumsq_5120(vals, part), 5120.0) + eps);
        unroll!(16, 10, |k| *y.add(lid + 512 * k) = to_bf(*vals.add(lid + 512 * k) * scale * bf(wa[k])));
        return;
    }
    let mut i = lid as u32;
    while i < n {
        *vals.add(i as usize) = bf(*x.add(i as usize));
        i += 512;
    }
    slm::barrier();
    let scale = math::rsqrt(math::div_ocl(sumsq_ordered(vals, n, part), n as f32) + eps);
    let mut i = lid as u32;
    while i < n {
        *y.add(i as usize) = to_bf(*vals.add(i as usize) * scale * bf(*w.add(i as usize)));
        i += 512;
    }
}

/// Residual add in place: x = bf16(x + d). Grid (n / 64), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn add_bf16(x: *mut u16, d: *const u16, n: u32) {
    attr::sg_size::<32>(); // the reference compiles at SIMD32 (no required sub-group size)
    let i = id::global_id(0);
    if i < n as usize {
        *x.add(i) = to_bf(bf(*x.add(i)) + bf(*d.add(i)));
    }
}
