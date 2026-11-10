//! Window (1..16 rows, and prompt windows) building blocks: Rust port of arc-engine `kernels/xpu/qwen_rows.cl`.
//! A row's bits never depend on how many rows share the launch: norms, conv, gates, gated norm and attention prep are
//! row-parallel; the delta rule runs the window's rows in order inside the kernel with the state in registers.
//!
//! Contraction sites of the reference (`spirv-dis qwen_rows.spv`): sumsq_ordered `fma(f, f, ss)` (source fma);
//! conv_t three `mad` (acc += a * w); gdn_conv_prep_r / gdn_gate_norm_r `mad(v, v, ss)`; gdn_step `mad(k8, delta, s)`;
//! attn_prep_r RoPE `mad(n_i, c_i, -(n_{i+32} s_i))` and `mad(n_i, c_{i-32}, n_{i-32} s_{i-32})`. Everything else is
//! unfused.
use crate::ext::{dot4, vload8, vstore8};
use crate::{bf, bfr, silu, to_bf};
use kxcommon::{attr, id, math, sg, slm, unroll};

const DK: usize = 128;
const DV: usize = 128;
const KH: usize = 16;
const VH: usize = 48;
const HD: usize = 256;
const NH: usize = 24;
const NKV: usize = 4;
const RD: usize = 32;

/// Sum of squares of vals[0..n) in the order of the 64-item rmsnorm (qwen_rows.cl `sumsq_ordered`): items 0..63, lane L
/// adds L, L + 64, .. with fma, the sub-group reduce, the four parts left to right. Needs a barrier after vals is written.
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

/// Row get_group_id(0) of x [rows][n] (n <= 5120): x = bf16(x + d); y = bf16(x * rsqrt(mean(x^2) + eps) * w).
/// Grid (rows), local 512.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn add_rmsnorm_r(x: *mut u16, d: *const u16, w: *const u16, y: *mut u16, n: u32, eps: f32) {
    attr::reqd_work_group_size::<512, 1, 1>();
    let part = slm::alloc::<f32, 4, 0>();
    let vals = slm::alloc::<f32, 5120, 1>();
    let lid = id::local_id(0);
    let base = id::group_id(0) as usize * n as usize;
    if n == 5120 {
        // the static path of qwen_basic.cl add_rmsnorm (same values, same sums: sumsq_5120 == sumsq_ordered bit for
        // bit): every load of an item issued before use, the 80 squares in batches of 16 (4.8x faster at 1 row)
        let (x, d, y) = (x.add(base), d.add(base), y.add(base));
        let l = lid as usize;
        let (mut xa, mut da, mut wa) = ([0u16; 10], [0u16; 10], [0u16; 10]);
        unroll!(16, 10, |k| {
            xa[k] = *x.add(l + 512 * k);
            da[k] = *d.add(l + 512 * k);
            wa[k] = *w.add(l + 512 * k);
        });
        unroll!(16, 10, |k| {
            let v = to_bf(bf(xa[k]) + bf(da[k]));
            *x.add(l + 512 * k) = v;
            *vals.add(l + 512 * k) = bf(v);
        });
        slm::barrier();
        let scale = math::rsqrt(math::div_ocl(crate::one::sumsq_5120(vals, part), 5120.0) + eps);
        unroll!(16, 10, |k| *y.add(l + 512 * k) = to_bf(*vals.add(l + 512 * k) * scale * bf(wa[k])));
        return;
    }
    let mut i = lid;
    while i < n {
        let v = to_bf(bf(*x.add(base + i as usize)) + bf(*d.add(base + i as usize)));
        *x.add(base + i as usize) = v;
        *vals.add(i as usize) = bf(v);
        i += 512;
    }
    slm::barrier();
    let scale = math::rsqrt(math::div_ocl(sumsq_ordered(vals, n, part), n as f32) + eps);
    let mut i = lid;
    while i < n {
        *y.add(base + i as usize) = to_bf(*vals.add(i as usize) * scale * bf(*w.add(i as usize)));
        i += 512;
    }
}

/// RMSNorm of row get_group_id(0) of x [rows][n] (no residual add). Grid (rows), local 512.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rmsnorm_r(x: *mut u16, w: *const u16, y: *mut u16, n: u32, eps: f32) {
    attr::reqd_work_group_size::<512, 1, 1>();
    let part = slm::alloc::<f32, 4, 0>();
    let vals = slm::alloc::<f32, 5120, 1>();
    let lid = id::local_id(0);
    let base = id::group_id(0) as usize * n as usize;
    if n == 5120 {
        // static path of qwen_basic.cl rmsnorm512 (same bits as the loop below)
        let (x, y) = (x.add(base), y.add(base));
        let l = lid as usize;
        let (mut xa, mut wa) = ([0u16; 10], [0u16; 10]);
        unroll!(16, 10, |k| {
            xa[k] = *x.add(l + 512 * k);
            wa[k] = *w.add(l + 512 * k);
        });
        unroll!(16, 10, |k| *vals.add(l + 512 * k) = bf(xa[k]));
        slm::barrier();
        let scale = math::rsqrt(math::div_ocl(crate::one::sumsq_5120(vals, part), 5120.0) + eps);
        unroll!(16, 10, |k| *y.add(l + 512 * k) = to_bf(*vals.add(l + 512 * k) * scale * bf(wa[k])));
        return;
    }
    let mut i = lid;
    while i < n {
        *vals.add(i as usize) = bf(*x.add(base + i as usize));
        i += 512;
    }
    slm::barrier();
    let scale = math::rsqrt(math::div_ocl(sumsq_ordered(vals, n, part), n as f32) + eps);
    let mut i = lid;
    while i < n {
        *y.add(base + i as usize) = to_bf(*vals.add(i as usize) * scale * bf(*w.add(i as usize)));
        i += 512;
    }
}

// ---- Gated DeltaNet window ----

/// Extended conv input of one channel: ext(0..2) = the three state rows, ext(3 + t) = qkv row t.
#[inline(always)]
unsafe fn ext_at(qkv: *const u16, state: *const u16, cd: u32, ch: u32, j: u32) -> u16 {
    if j < 3 { *state.add((j * cd + ch) as usize) } else { *qkv.add((j - 3) as usize * cd as usize + ch as usize) }
}

/// Conv + silu of channel ch at window row t: taps cw[ch][0..3] (tap 0 = oldest); bf16 bits.
#[inline(always)]
unsafe fn conv_t(qkv: *const u16, state: *const u16, cw: *const u16, cd: u32, ch: u32, t: u32) -> u16 {
    let w = cw.add(4 * ch as usize);
    let (w0, w1, w2, w3) = (*w, *w.add(1), *w.add(2), *w.add(3));
    let mut acc = bf(ext_at(qkv, state, cd, ch, t)) * bf(w0);
    acc = math::mad(bf(ext_at(qkv, state, cd, ch, t + 1)), bf(w1), acc);
    acc = math::mad(bf(ext_at(qkv, state, cd, ch, t + 2)), bf(w2), acc);
    acc = math::mad(bf(ext_at(qkv, state, cd, ch, t + 3)), bf(w3), acc);
    to_bf(silu(acc))
}

/// Conv state after c accepted rows for channels [ch0, ch0 + n) (one item a channel): state = ext(c .. c + 2), all
/// three old values read before writing. Grid ((n + 63) / 64), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_conv_commit(qkv: *const u16, state: *mut u16, cd: u32, c: u32, ch0: u32, n: u32) {
    attr::sg_size::<32>(); // the reference compiles at SIMD32 (no required sub-group size)
    let i = id::global_id(0) as u32;
    if i < n {
        let ch = ch0 + i;
        let n0 = ext_at(qkv, state, cd, ch, c);
        let n1 = ext_at(qkv, state, cd, ch, c + 1);
        let n2 = ext_at(qkv, state, cd, ch, c + 2);
        *state.add(ch as usize) = n0;
        *state.add((cd + ch) as usize) = n1;
        *state.add((2 * cd + ch) as usize) = n2;
    }
}

/// q and k channels of window row get_group_id(1): conv + silu, then q = bf16(rms(q) / Dk), k = bf16(rms(k) / sqrt(Dk))
/// per key head, widened to fp32 into qn / kn [rows][2048]. One sub-group a (key head, row). Grid (16, rows), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_conv_prep_r(qkv: *const u16, state: *const u16, cw: *const u16, qn: *mut f32, kn: *mut f32, cd: u32) {
    let h = id::group_id(0);
    let t = id::group_id(1);
    // vector loads of the taps and weights (gdnpre.rs); same arithmetic as conv_t + the norm
    crate::gdnpre::prep_head(qkv, state, cw, cd, [t, t + 1, t + 2, t + 3], h, t, qn, kn);
}

/// Value-channel conv + silu of window row get_group_id(0): vout[row][6144] = conv of channels 4096 + c (bf16), reading
/// the old state like gdn_conv_prep_r. Grid (rows, 96), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_vconv_r(qkv: *const u16, state: *const u16, cw: *const u16, vout: *mut u16, cd: u32) {
    attr::sg_size::<32>(); // the reference compiles at SIMD32 (no required sub-group size)
    let t = id::group_id(0);
    let c = id::group_id(1) * 64 + id::local_id(0);
    if c < (VH * DV) as u32 {
        *vout.add(t as usize * (VH * DV) + c as usize) = conv_t(qkv, state, cw, cd, (2 * KH * DK) as u32 + c, t);
    }
}

/// beta = sigmoid(b) and the decay g = exp(-exp(A_log) * softplus(a + dt_bias)) of window row get_group_id(0), head
/// get_local_id(0). Grid (rows), local 48.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_gates_r(bp: *const u16, ap: *const u16, a_log: *const f32, dt_bias: *const f32, beta_o: *mut f32, g_o: *mut f32) {
    attr::sg_size::<32>(); // the reference compiles at SIMD32 (no required sub-group size)
    let t = id::group_id(0) as usize;
    let h = id::local_id(0) as usize;
    *beta_o.add(t * VH + h) = math::div_ocl(1.0, 1.0 + math::exp(-bf(*bp.add(t * VH + h))));
    let av = bf(*ap.add(t * VH + h)) + *dt_bias.add(h);
    let sp = if av > 20.0 { av } else { crate::ext::log1p(math::exp(av)) };
    *g_o.add(t * VH + h) = math::exp(-math::exp(*a_log.add(h)) * sp);
}

#[inline(always)]
fn lo(v: &[f32; 8]) -> [f32; 4] {
    [v[0], v[1], v[2], v[3]]
}
#[inline(always)]
fn hi(v: &[f32; 8]) -> [f32; 4] {
    [v[4], v[5], v[6], v[7]]
}

/// Delta rule over `rows` window rows for value head get_group_id(0), NR value rows a 16-lane sub-group (8 Dk values a
/// lane, state in registers across the window); per-row inputs (q, k, the conv'd v, beta, g) precomputed row-parallel.
/// State to s_out (== s_in for in place). tiled: GGUF value-head order. Every row does the arithmetic and reduction
/// order of qwen_rows.cl gdn_step_r2, so state and y are bit-identical for every NR.
#[inline(always)]
unsafe fn gdn_step_nr<const NR: usize>(
    qn: *const f32,
    kn: *const f32,
    vconv: *const u16,
    beta_a: *const f32,
    g_a: *const f32,
    s_in: *const f32,
    s_out: *mut f32,
    y: *mut u16,
    tiled: u32,
    rows: u32,
) {
    let h = id::group_id(0) as usize;
    let row0 = (id::group_id(1) * 4 + id::sg_id()) as usize * NR;
    let lane = id::lane() as usize;
    let kh = if tiled != 0 { h % KH } else { h / (VH / KH) };
    let off = (h * DV + row0) * DK;
    let mut s = [[0f32; 8]; NR];
    unroll!(4, NR, |r| s[r] = vload8(lane, s_in.add(off + r * DK)));
    let mut t = 0u32;
    while t < rows {
        let tb = t as usize;
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
        t += 1;
    }
    unroll!(4, NR, |r| vstore8(s[r], lane, s_out.add(off + r * DK)));
}

/// gdn_step with 4 value rows a sub-group. Grid (48, 8), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_step_r2x4(
    qn: *const f32,
    kn: *const f32,
    vconv: *const u16,
    beta_a: *const f32,
    g_a: *const f32,
    s_in: *const f32,
    s_out: *mut f32,
    y: *mut u16,
    tiled: u32,
    rows: u32,
) {
    gdn_step_nr::<4>(qn, kn, vconv, beta_a, g_a, s_in, s_out, y, tiled, rows)
}

/// gdn_step with 1 value row a sub-group (gdn_step_r2). Grid (48, 32), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_step_r2(
    qn: *const f32,
    kn: *const f32,
    vconv: *const u16,
    beta_a: *const f32,
    g_a: *const f32,
    s_in: *const f32,
    s_out: *mut f32,
    y: *mut u16,
    tiled: u32,
    rows: u32,
) {
    gdn_step_nr::<1>(qn, kn, vconv, beta_a, g_a, s_in, s_out, y, tiled, rows)
}

/// out = bf16(silu(z) * (y * rsqrt(mean(y^2) + eps) * w)) for window row get_group_id(1), value head get_group_id(0).
/// Grid (48, rows), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gdn_gate_norm_r(y: *const u16, z: *const u16, w: *const u16, out: *mut u16, eps: f32) {
    let h = id::group_id(0) as usize;
    let rb = id::group_id(1) as usize * (VH * DV);
    let lane = id::lane() as usize;
    let mut v = [0f32; 8];
    let mut ss = 0.0f32;
    unroll!(8, 8, |j| {
        v[j] = bf(*y.add(rb + h * DV + lane * 8 + j));
        ss = math::mad(v[j], v[j], ss);
    });
    ss = sg::reduce_add_f32(ss);
    let r = math::rsqrt(math::div_ocl(ss, DV as f32) + eps);
    unroll!(8, 8, |j| {
        let d = lane * 8 + j;
        *out.add(rb + h * DV + d) = to_bf(silu(bf(*z.add(rb + h * DV + d))) * (v[j] * r * bf(*w.add(d))));
    });
}

// ---- full attention window ----

/// Window row get_group_id(1): heads 0..23 are queries from qg[row][head][0..255] (qg rows are [head][512]: query then
/// gate), heads 24..27 keys from kraw[row]. RMSNorm (bf16 weights), then rotate-half RoPE on the first 64 dims with
/// rope [row][cos(32) | sin(32)]; queries to qo [row][24][256], keys to kc[k_off + row * 1024 ..]. Grid (28, rows), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn attn_prep_r(
    qg: *const u16,
    kraw: *const u16,
    qw: *const u16,
    kw: *const u16,
    rope: *const f32,
    qo: *mut u16,
    kc: *mut u16,
    k_off: u64,
    eps: f32,
) {
    let part = slm::alloc::<f32, 16, 0>();
    let nv = slm::alloc::<f32, 256, 1>();
    let head = id::group_id(0) as usize;
    let row = id::group_id(1) as usize;
    let i = id::local_id(0) as usize;
    let is_q = head < NH;
    let x = if is_q { bf(*qg.add(row * (NH * 2 * HD) + head * 2 * HD + i)) } else { bf(*kraw.add(row * (NKV * HD) + (head - NH) * HD + i)) };
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
    let rp = rope.add(row * 64);
    let mut o = *nv.add(i);
    if i < RD {
        o = math::mad(*nv.add(i), *rp.add(i), -(*nv.add(i + RD) * *rp.add(RD + i)));
    } else if i < 2 * RD {
        o = math::mad(*nv.add(i), *rp.add(i - RD), *nv.add(i - RD) * *rp.add(RD + i - RD));
    }
    if is_q {
        *qo.add(row * (NH * HD) + head * HD + i) = to_bf(o);
    } else {
        *kc.add(k_off as usize + row * (NKV * HD) + (head - NH) * HD + i) = to_bf(o);
    }
}
