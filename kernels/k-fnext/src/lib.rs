//! Qwen3.8-Flash-Next kernels (f32 activations): hyper-connection mixer / combine, grouped norms, PLE, GDN step,
//! attention, GEMV over f16 / bf16 weights. Loaded by crates/fe-fnext. Correctness first: simple, row-generic.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

extern crate kcommon;
pub mod fast;
pub mod gdnc;
pub mod gdntc;
pub mod idxtc;
pub mod pattn;
use kcommon::smem::Shared;
use kcommon::*;

#[inline(always)]
pub fn h2f_pub(h: u16) -> f32 {
    h2f(h)
}
#[inline(always)]
fn h2f(h: u16) -> f32 {
    let r: f32;
    unsafe { core::arch::asm!("cvt.f32.f16 {r}, {h};", r = out(reg32) r, h = in(reg16) h, options(pure, nomem, nostack)) }
    r
}
#[inline(always)]
fn w8(v: [u32; 4], bf: bool) -> [f32; 8] {
    if bf {
        bf16::unpack8(v)
    } else {
        let mut o = [0.0f32; 8];
        unroll!(i in 4 => {
            o[2 * i] = h2f(v[i] as u16);
            o[2 * i + 1] = h2f((v[i] >> 16) as u16);
        });
        o
    }
}
#[inline(always)]
unsafe fn ld8(p: *const f32) -> [f32; 8] {
    let a = ldg_nc_v4(p as *const u32);
    let b = ldg_nc_v4(p.add(4) as *const u32);
    [f32::from_bits(a[0]), f32::from_bits(a[1]), f32::from_bits(a[2]), f32::from_bits(a[3]), f32::from_bits(b[0]),
        f32::from_bits(b[1]), f32::from_bits(b[2]), f32::from_bits(b[3])]
}

/// y[r][j] = (acc ? y[r][j] : 0) + scale * sum_k w[j][k] x[r][k]. w: [n][k] f16 (bf = 0) or bf16 (bf = 1).
/// grid = (ceil(n / 8), ceil(rows / 4)), block 256: a warp per output, 4 rows per block. k % 8 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_gemv_h(y: *mut f32, x: *const f32, w: *const u16, k: u32, n: u32, rows: u32,
    scale: f32, bf: u32, acc: u32) {
    let j = bid_x() * 8 + warp_id();
    if j >= n {
        return;
    }
    let r0 = bid_y() * 4;
    let nr = if rows - r0 < 4 { rows - r0 } else { 4 };
    let wr = w.add(j as usize * k as usize);
    let mut a = [0.0f32; 4];
    let mut i = lane() * 8;
    while i < k {
        let wv = w8(ldg_nc_v4(wr.add(i as usize) as *const u32), bf != 0);
        unroll!(r in 4 => {
            if (r as u32) < nr {
                let xv = ld8(x.add((r0 as usize + r) * k as usize + i as usize));
                let mut s = 0.0f32;
                unroll!(e in 8 => { s = math::fma(wv[e], xv[e], s); });
                a[r] += s;
            }
        });
        i += 256;
    }
    unroll!(r in 4 => {
        let t = warp::sum(a[r]);
        if lane() == 0 && (r as u32) < nr {
            let p = y.add((r0 as usize + r) * n as usize + j as usize);
            *p = if acc != 0 { *p + scale * t } else { scale * t };
        }
    });
}

/// Grouped RMSNorm: y[g][i] = x[g][i] * rsqrt(mean(x[g]^2) + eps) * w[(g % ngw) * d + i] (w null: 1).
/// Groups of d contiguous floats; grid = groups, block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_rmsnorm(y: *mut f32, x: *const f32, w: *const f32, d: u32, ngw: u32, eps: f32) {
    let red: Shared<f32> = kcommon::static_shared!(rn_red, f32, 128);
    let g = bid_x() as usize;
    let xr = x.add(g * d as usize);
    let yr = y.add(g * d as usize);
    let mut ss = 0.0f32;
    let mut i = tid_x();
    while i < d {
        let v = *xr.add(i as usize);
        ss += v * v;
        i += bdim_x();
    }
    let tot = warp::block_sum(ss, red);
    let inv = 1.0 / math::sqrt(tot / d as f32 + eps);
    let wo = (g % ngw.max(1) as usize) * d as usize;
    let mut i = tid_x();
    while i < d {
        let wv = if w.is_null() { 1.0 } else { *w.add(wo + i as usize) };
        *yr.add(i as usize) = *xr.add(i as usize) * inv * wv;
        i += bdim_x();
    }
}

/// x = silu(x), n elements; grid = ceil(n/256), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_silu(x: *mut f32, n: u32) {
    let i = bid_x() * 256 + tid_x();
    if i < n {
        let v = *x.add(i as usize);
        *x.add(i as usize) = v / (1.0 + math::exp(-v));
    }
}

/// out = silu(g) * u (n elements; out may alias g); grid = ceil(n/256), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_swiglu(out: *mut f32, g: *const f32, u: *const f32, n: u32) {
    let i = bid_x() * 256 + tid_x();
    if i < n {
        let v = *g.add(i as usize);
        *out.add(i as usize) = v / (1.0 + math::exp(-v)) * *u.add(i as usize);
    }
}

/// res[r][s][:] = e[r][:] for the hc streams; n = rows * d; grid = ceil(n/256), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_hc_init(res: *mut f32, e: *const f32, d: u32, hc: u32, n: u32) {
    let idx = bid_x() * 256 + tid_x();
    if idx < n {
        let (r, i) = (idx / d, idx % d);
        let v = *e.add(idx as usize);
        let mut s = 0;
        while s < hc {
            *res.add(((r * hc + s) * d + i) as usize) = v;
            s += 1;
        }
    }
}

/// HC mixer output: mixed[r][i] = mean_s xn[r][s][i] * sigmoid(gate[r][s][i]); n = rows * d.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_hc_mix(mixed: *mut f32, xn: *const f32, gate: *const f32, d: u32, hc: u32, n: u32) {
    let idx = bid_x() * 256 + tid_x();
    if idx < n {
        let (r, i) = (idx / d, idx % d);
        let mut acc = 0.0f32;
        let mut s = 0;
        while s < hc {
            let o = ((r * hc + s) * d + i) as usize;
            let gv = *gate.add(o);
            acc += *xn.add(o) * (1.0 / (1.0 + math::exp(-gv)));
            s += 1;
        }
        *mixed.add(idx as usize) = acc / hc as f32;
    }
}

/// HC combine: res[r][s][i] += out[r][i] * 2 sigmoid(inj[r * ldi + s]); n = rows * hc * d.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_hc_combine(res: *mut f32, out: *const f32, inj: *const f32, d: u32, hc: u32, ldi: u32,
    n: u32) {
    let idx = bid_x() * 256 + tid_x();
    if idx < n {
        let i = idx % d;
        let rs = idx / d;
        let (r, s) = (rs / hc, rs % hc);
        let w = 2.0 / (1.0 + math::exp(-*inj.add((r * ldi + s) as usize)));
        *res.add(idx as usize) += *out.add((r * d + i) as usize) * w;
    }
}

/// PLE gate per (row, stream): s = dot(key_n, q_n) / sqrt(d); gate = sigmoid(sign(s) sqrt(max(|s|, 1e-6)));
/// gated[r][s][:] = value[r][:] * gate. grid = rows * hc, block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_ple_gate(gated: *mut f32, key_n: *const f32, q_n: *const f32, value: *const f32,
    d: u32, hc: u32) {
    let red: Shared<f32> = kcommon::static_shared!(pg_red, f32, 128);
    let g = bid_x() as usize;
    let r = g / hc as usize;
    let kr = key_n.add(g * d as usize);
    let qr = q_n.add(g * d as usize);
    let mut acc = 0.0f32;
    let mut i = tid_x();
    while i < d {
        acc += *kr.add(i as usize) * *qr.add(i as usize);
        i += bdim_x();
    }
    let s = warp::block_sum(acc, red) / math::sqrt(d as f32);
    let mag = math::sqrt(math::maxf(if s < 0.0 { -s } else { s }, 1e-6));
    let sg = if s > 0.0 { 1.0 } else if s < 0.0 { -1.0 } else { 0.0 };
    let gate = 1.0 / (1.0 + math::exp(-(sg * mag)));
    let mut i = tid_x();
    while i < d {
        *gated.add(g * d as usize + i as usize) = *value.add(r * d as usize + i as usize) * gate;
        i += bdim_x();
    }
}

/// PLE dilated causal depthwise conv + residual: conv[t][c] = sum_k w[c][k] xn[t - (K-1-k) dil][c] (history: hist
/// [(K-1) dil][C], oldest first); res[t][c] += gated[t][c] + silu(conv). Then hist = the last (K-1) dil inputs.
/// K = 4. co (optional): the silu(conv) rows. grid = ceil(C/256), block 256 (thread per channel, rows in order).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_ple_conv(res: *mut f32, gated: *const f32, xn: *const f32, w: *const f32,
    hist: *mut f32, rows: u32, ch: u32, dil: u32, co: *mut f32, ro: u32) {
    let c = bid_x() * 256 + tid_x();
    if c >= ch {
        return;
    }
    let hl = 3 * dil; // history length
    let wv = [*w.add(c as usize * 4), *w.add(c as usize * 4 + 1), *w.add(c as usize * 4 + 2), *w.add(c as usize * 4 + 3)];
    // value at sequence index q of concat(hist, xn)
    let at = |q: u32| -> f32 {
        if q < hl { *hist.add((q * ch + c) as usize) } else { *xn.add(((q - hl) * ch + c) as usize) }
    };
    let mut t = 0;
    while t < rows {
        let q = t + hl;
        let mut acc = 0.0f32;
        let mut k = 0u32;
        while k < 4 {
            acc += wv[k as usize] * at(q - (3 - k) * dil);
            k += 1;
        }
        let o = (t * ch + c) as usize;
        let sv = acc / (1.0 + math::exp(-acc));
        if !co.is_null() {
            *co.add(o) = sv;
        }
        *res.add(o) += *gated.add(o) + sv;
        t += 1;
    }
    let mut j = 0;
    while j < hl && ro == 0 {
        let v = at(rows + j);
        *hist.add((j * ch + c) as usize) = v;
        j += 1;
    }
}

/// GDN causal conv (kernel 4) + SiLU with carried state [3][C] (oldest first): out[t][c] = silu(w0 x[t-3] + w1 x[t-2]
/// + w2 x[t-1] + w3 x[t]); x rows of stride ldx. grid = ceil(C/256), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_gdn_conv(out: *mut f32, x: *const f32, ldx: u32, w: *const f32, state: *mut f32,
    rows: u32, ch: u32, ro: u32) {
    let c = bid_x() * 256 + tid_x();
    if c >= ch {
        return;
    }
    let (w0, w1, w2, w3) = (*w.add(c as usize * 4), *w.add(c as usize * 4 + 1), *w.add(c as usize * 4 + 2), *w.add(c as usize * 4 + 3));
    let mut s0 = *state.add(c as usize);
    let mut s1 = *state.add((ch + c) as usize);
    let mut s2 = *state.add((2 * ch + c) as usize);
    let mut t = 0;
    while t < rows {
        let xv = *x.add((t * ldx + c) as usize);
        let y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * xv;
        *out.add((t * ch + c) as usize) = y / (1.0 + math::exp(-y));
        s0 = s1;
        s1 = s2;
        s2 = xv;
        t += 1;
    }
    if ro == 0 {
        *state.add(c as usize) = s0;
        *state.add((ch + c) as usize) = s1;
        *state.add((2 * ch + c) as usize) = s2;
    }
}

/// Gated delta rule, rows in order. grid = (nvh, 512 / block), block 512 or 128 (value columns split over grid.y): thread t owns value column j = t / 4 and key rows
/// (t % 4) * 32 .. +32 of the head's fp32 state S[i][j] (state [h][128][128], i = key dim).
/// conv: [rows][C] post conv+silu (q: nkh heads at 0, k at nkh*128, v: nvh heads at 2 nkh*128); v head h reads
/// q/k head h % nkh. ab: rows of stride ldab, alpha at 0, beta at nvh. o: [rows][nvh][128].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_gdn_rec(o: *mut f32, conv: *const f32, ab: *const f32, ldab: u32, dt_bias: *const f32,
    ssm_a: *const f32, state: *mut f32, rows: u32, nkh: u32, nvh: u32, eps: f32, ro: u32) {
    let qs: Shared<f32> = kcommon::static_shared!(gr_q, f32, 512);
    let ks: Shared<f32> = kcommon::static_shared!(gr_k, f32, 512);
    let red: Shared<f32> = kcommon::static_shared!(gr_red, f32, 16);
    let h = bid_x();
    let t = tid_x();
    // value columns split over grid.y slices: blockDim / 4 columns each (512 threads x 1 slice, or 128 x 4)
    let j = bid_y() * (bdim_x() >> 2) + (t >> 2);
    let part = t & 3;
    let kh = h % nkh;
    let ch = 2 * nkh * 128 + nvh * 128;
    let sp = state.add((h * 128 * 128) as usize);
    let mut s = [0.0f32; 32];
    unroll!(ii in 32 => { s[ii] = *sp.add(((part * 32 + ii as u32) * 128 + j) as usize); });
    let dtb = *dt_bias.add(h as usize);
    let sa = *ssm_a.add(h as usize);
    let mut r = 0;
    while r < rows {
        let cr = conv.add((r * ch) as usize);
        // q / k of this head into shared, l2-normalised (warp 0: q, warp 1: k)
        let mut i = t;
        while i < 256 {
            let (src, dst) = if i < 128 { (cr.add((kh * 128 + i) as usize), qs.add(i)) } else {
                (cr.add((nkh * 128 + kh * 128 + i - 128) as usize), ks.add(i - 128))
            };
            dst.st(*src);
            i += bdim_x();
        }
        sync_threads();
        if t < 64 {
            let w = t >> 5; // 0: q, 1: k
            let buf = if w == 0 { qs } else { ks };
            let l = lane();
            let mut ss = 0.0f32;
            unroll!(e in 4 => { let v = buf.at(l * 4 + e as u32); ss += v * v; });
            let tot = warp::sum(ss);
            if l == 0 {
                red.set(w, 1.0 / math::sqrt(tot + eps));
            }
        }
        sync_threads();
        let qn = red.at(0) / math::sqrt(128.0);
        let kn = red.at(1);
        let abr = ab.add((r * ldab) as usize);
        let g = math::softplus(*abr.add(h as usize) + dtb) * sa;
        let beta = 1.0 / (1.0 + math::exp(-*abr.add((nvh + h) as usize)));
        let decay = math::exp(g);
        let vj = *cr.add((2 * nkh * 128 + h * 128 + j) as usize);
        let mut sk = 0.0f32;
        unroll!(ii in 32 => {
            s[ii] *= decay;
            sk += s[ii] * ks.at(part * 32 + ii as u32) * kn;
        });
        sk += warp::shfl_xor(sk, 1);
        sk += warp::shfl_xor(sk, 2);
        let delta = (vj - sk) * beta;
        let mut oq = 0.0f32;
        unroll!(ii in 32 => {
            s[ii] += ks.at(part * 32 + ii as u32) * kn * delta;
            oq += s[ii] * qs.at(part * 32 + ii as u32) * qn;
        });
        oq += warp::shfl_xor(oq, 1);
        oq += warp::shfl_xor(oq, 2);
        if part == 0 && !o.is_null() {
            *o.add(((r * nvh + h) * 128 + j) as usize) = oq;
        }
        sync_threads();
        r += 1;
    }
    if ro == 0 {
        unroll!(ii in 32 => { *sp.add(((part * 32 + ii as u32) * 128 + j) as usize) = s[ii]; });
    }
}

/// GDN gated norm: y[r][h][:] = rmsnorm(o[r][h][:]) * w * sigmoid(z[r][h][:]) (head dim 128, warp per head).
/// grid = (rows, nvh / 8), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_gdn_gnorm(y: *mut f32, o: *const f32, z: *const f32, ldz: u32, w: *const f32, nvh: u32,
    eps: f32) {
    let r = bid_x();
    let h = bid_y() * 8 + warp_id();
    if h >= nvh {
        return;
    }
    let l = lane();
    let base = ((r * nvh + h) * 128 + l * 4) as usize;
    let mut x = [0.0f32; 4];
    let mut ss = 0.0f32;
    unroll!(e in 4 => { x[e] = *o.add(base + e); ss += x[e] * x[e]; });
    let inv = 1.0 / math::sqrt(warp::sum(ss) / 128.0 + eps);
    unroll!(e in 4 => {
        let zv = *z.add((r * ldz + h * 128 + l * 4) as usize + e);
        *y.add(base + e) = x[e] * inv * *w.add((l * 4) as usize + e) * (1.0 / (1.0 + math::exp(-zv)));
    });
}

/// Attention prep: q heads (from qf rows of stride ldq, head h at h * qstride) and k heads RMS-normed, roped
/// (neox pairs i, i + nrot/2 over the first nrot dims; cos / sin tables [rows][nrot/2]), k and v stored into the
/// f32 caches at cell pos0 + r ([cell][nkv][hd]). grid = (rows, nq + nkv), block = hd (256).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_attn_prep(qo: *mut f32, kc: *mut f32, vc: *mut f32, qf: *const f32, ldq: u32,
    qstride: u32, kx: *const f32, vx: *const f32, ldkv: u32, qnw: *const f32, knw: *const f32, cs: *const f32,
    sn: *const f32, nq: u32, nkv: u32, nrot: u32, pos0: u32, eps: f32) {
    let red: Shared<f32> = kcommon::static_shared!(ap_red, f32, 128);
    let xs: Shared<f32> = kcommon::static_shared!(ap_x, f32, 2048);
    let r = bid_x();
    let h = bid_y();
    let i = tid_x();
    let hd = bdim_x();
    let isq = h < nq;
    let src = if isq { qf.add((r * ldq + h * qstride + i) as usize) } else { kx.add((r * ldkv + (h - nq) * hd + i) as usize) };
    let v = *src;
    let tot = warp::block_sum(v * v, red);
    let wv = if isq { *qnw.add(i as usize) } else { *knw.add(i as usize) };
    let xn = v * (1.0 / math::sqrt(tot / hd as f32 + eps)) * wv;
    xs.set(i, xn);
    sync_threads();
    let half = nrot / 2;
    let mut out = xn;
    if i < nrot {
        let p = i % half;
        let c = *cs.add((r * half + p) as usize);
        let s = *sn.add((r * half + p) as usize);
        out = if i < half { xn * c - xs.at(i + half) * s } else { xn * c + xs.at(i - half) * s };
    }
    if isq {
        *qo.add(((r * nq + h) * hd + i) as usize) = out;
    } else {
        let kh = h - nq;
        let cell = (pos0 + r) as usize;
        *kc.add((cell * nkv as usize + kh as usize) * hd as usize + i as usize) = out;
        *vc.add((cell * nkv as usize + kh as usize) * hd as usize + i as usize) = *vx.add((r * ldkv + kh * hd + i) as usize);
    }
}

/// Causal attention over the f32 caches for rows at cells pos0 + r (cells 0 ..= pos0 + r, or the seln[r] cells
/// sel[r][..] when sel is not null: the QSA selection), output gated by
/// sigmoid(gate[r][h * gstride + d]). q [rows][nq][256]; out [rows][nq][256]. grid = (rows, nq), block 256
/// (8 warps over the cells, lane = 8 dims).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_attn(out: *mut f32, q: *const f32, kc: *const f32, vc: *const f32, gate: *const f32,
    ldg: u32, gstride: u32, nq: u32, nkv: u32, pos0: u32, scale: f32, sel: *const u32, seln: *const u32, ldsel: u32) {
    let sm: Shared<f32> = kcommon::static_shared!(at_m, f32, 64);
    let sl: Shared<f32> = kcommon::static_shared!(at_l, f32, 64);
    let sacc: Shared<f32> = kcommon::static_shared!(at_acc, f32, 8192);
    let r = bid_x();
    let h = bid_y();
    let kh = h / (nq / nkv);
    let w = warp_id();
    let l = lane();
    let qv = ld8(q.add(((r * nq + h) * 256 + l * 8) as usize));
    // dense: cells 0 ..= pos0 + r; QSA: the row's selected cells
    let dense = sel.is_null();
    let n = if dense { pos0 + r + 1 } else { *seln.add(r as usize) };
    let mut m = f32::NEG_INFINITY;
    let mut ls = 0.0f32;
    let mut acc = [0.0f32; 8];
    let mut pi = w;
    while pi < n {
        let p = if dense { pi } else { *sel.add((r * ldsel + pi) as usize) };
        let kv = ld8(kc.add(((p * nkv + kh) * 256 + l * 8) as usize));
        let mut d = 0.0f32;
        unroll!(e in 8 => { d = math::fma(qv[e], kv[e], d); });
        let sc = warp::sum(d) * scale;
        let mn = math::maxf(m, sc);
        let a = math::exp(m - mn);
        let b = math::exp(sc - mn);
        let vv = ld8(vc.add(((p * nkv + kh) * 256 + l * 8) as usize));
        unroll!(e in 8 => { acc[e] = acc[e] * a + b * vv[e]; });
        ls = ls * a + b;
        m = mn;
        pi += 8;
    }
    if l == 0 {
        sm.set(w, m);
        sl.set(w, ls);
    }
    unroll!(e in 8 => { sacc.set(w * 256 + l * 8 + e as u32, acc[e]); });
    sync_threads();
    // thread t combines dim t over the 8 warps
    let t = tid_x();
    let mut mm = f32::NEG_INFINITY;
    unroll!(k in 8 => { mm = math::maxf(mm, sm.at(k as u32)); });
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    unroll!(k in 8 => {
        let mk = sm.at(k as u32);
        if mk != f32::NEG_INFINITY {
            let f = math::exp(mk - mm);
            num += sacc.at(k as u32 * 256 + t) * f;
            den += sl.at(k as u32) * f;
        }
    });
    let gv = *gate.add((r * ldg + h * gstride + t) as usize);
    *out.add(((r * nq + h) * 256 + t) as usize) = num / den * (1.0 / (1.0 + math::exp(-gv)));
}

/// out[r][i] = moe[r][i] + sh[r][i] * sigmoid(glog[r]); n = rows * d.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_moe_out(out: *mut f32, moe: *const f32, sh: *const f32, glog: *const f32, d: u32, n: u32) {
    let idx = bid_x() * 256 + tid_x();
    if idx < n {
        let g = 1.0 / (1.0 + math::exp(-*glog.add((idx / d) as usize)));
        *out.add(idx as usize) = *moe.add(idx as usize) + *sh.add(idx as usize) * g;
    }
}

/// In-place neox rope over the first nrot dims of `nh` heads of `hd` per row (cos / sin tables [rows][nrot/2]).
/// grid = (rows, nh), block = hd.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_rope_rows(x: *mut f32, cs: *const f32, sn: *const f32, nh: u32, nrot: u32) {
    let xs: Shared<f32> = kcommon::static_shared!(rr_x, f32, 2048);
    let r = bid_x();
    let h = bid_y();
    let i = tid_x();
    let hd = bdim_x();
    let p = x.add(((r * nh + h) * hd + i) as usize);
    let v = *p;
    xs.set(i, v);
    sync_threads();
    let half = nrot / 2;
    if i < nrot {
        let k = i % half;
        let c = *cs.add((r * half + k) as usize);
        let s = *sn.add((r * half + k) as usize);
        *p = if i < half { v * c - xs.at(i + half) * s } else { v * c + xs.at(i - half) * s };
    }
}

/// Mean of each block of `kp` raw indexer keys: out[j][:] = mean_k raw[(b0 + j) kp + k][:]. grid = nb, block = d.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_idx_pool(out: *mut f32, raw: *const f32, b0: u32, kp: u32) {
    let j = bid_x();
    let i = tid_x();
    let d = bdim_x();
    let mut acc = 0.0f32;
    let mut k = 0;
    while k < kp {
        acc += *raw.add((((b0 + j) * kp + k) * d + i) as usize);
        k += 1;
    }
    *out.add((j * d + i) as usize) = acc * (1.0 / kp as f32);
}

/// QSA block scores: score[r][b] = scale * sum_h relu(q[r][h] . pooled[b]) for the blocks visible to row r
/// (b < (pos0 + r + 1) / kp), -inf for the rest up to ld. q [rows][nh][128]; pooled [b][128].
/// grid = (rows, ceil(ld / 8)), block 256 (warp per block).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_idx_score(score: *mut f32, q: *const f32, pooled: *const f32, nh: u32, pos0: u32,
    kp: u32, ld: u32, scale: f32) {
    let r = bid_x();
    let b = bid_y() * 8 + warp_id();
    if b >= ld {
        return;
    }
    let nv = (pos0 + r + 1) / kp;
    let l = lane();
    let out = score.add((r * ld + b) as usize);
    if b >= nv {
        if l == 0 {
            *out = f32::NEG_INFINITY;
        }
        return;
    }
    let kv = [*pooled.add((b * 128 + l * 4) as usize), *pooled.add((b * 128 + l * 4 + 1) as usize),
        *pooled.add((b * 128 + l * 4 + 2) as usize), *pooled.add((b * 128 + l * 4 + 3) as usize)];
    let mut tot = 0.0f32;
    let mut h = 0;
    while h < nh {
        let qp = q.add(((r * nh + h) * 128 + l * 4) as usize);
        let d = warp::sum(*qp * kv[0] + *qp.add(1) * kv[1] + *qp.add(2) * kv[2] + *qp.add(3) * kv[3]);
        tot += math::maxf(d, 0.0);
        h += 1;
    }
    if l == 0 {
        *out = tot * scale;
    }
}

/// Numerics emulation of the reference (checks only): mode 1 = q8_1 round trip per 32 values (d = amax / 127 held
/// as f16, q = round(x / d)), mode 2 = round to bf16. grid = ceil(n / 256), block 256 (n % 32 == 0).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_emu_round(y: *mut f32, x: *const f32, n: u32, mode: u32) {
    let i = bid_x() * 256 + tid_x();
    if i >= n {
        return;
    }
    let v = *x.add(i as usize);
    if mode == 2 {
        *y.add(i as usize) = bf16::round(v);
        return;
    }
    let a = warp::max(if v < 0.0 { -v } else { v });
    let d = h2f(f2h(a / 127.0));
    let q = if a == 0.0 { 0.0 } else { rintf(v / (a / 127.0)) };
    *y.add(i as usize) = q * d;
}

#[inline(always)]
fn f2h(f: f32) -> u16 {
    let h: u16;
    unsafe { core::arch::asm!("cvt.rn.f16.f32 {h}, {f};", h = out(reg16) h, f = in(reg32) f, options(pure, nomem, nostack)) }
    h
}
#[inline(always)]
fn rintf(f: f32) -> f32 {
    // roundf: half away from zero
    let a = if f < 0.0 { -f } else { f };
    let r: f32;
    unsafe { core::arch::asm!("cvt.rmi.f32.f32 {r}, {f};", r = out(reg32) r, f = in(reg32) a + 0.5, options(pure, nomem, nostack)) }
    if f < 0.0 { -r } else { r }
}

/// Commit a causal-conv history after a speculative verify: state [hl][C] (oldest first) becomes the last hl rows of
/// concat(state, x[0..n]) (x rows of stride ldx). grid = ceil(C / 256), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fn_conv_shift(state: *mut f32, x: *const f32, ldx: u32, n: u32, ch: u32, hl: u32) {
    let c = bid_x() * 256 + tid_x();
    if c >= ch {
        return;
    }
    let mut j = 0;
    while j < hl {
        let q = n + j; // index into concat(state, x)
        let v = if q < hl { *state.add((q * ch + c) as usize) } else { *x.add(((q - hl) * ldx + c) as usize) };
        *state.add((j * ch + c) as usize) = v;
        j += 1;
    }
}
