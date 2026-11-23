//! Inkling-2-Mini draft head (mtp.safetensors) kernels: tap mixing + RMS norm, per-head RMS norm + NeoX RoPE, the small
//! block attention (8 queries over the draft K/V cache + the block), residual add, SwiGLU, and the Markov head
//! (state-update GEMV, gates, bias GEMV + soft-capped argmax over the vocabulary). Reference-grade, rounding points follow
//! the publisher's reference (bf16 at every op it rounds).
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

extern crate kcommon;

use kcommon::smem::Shared;
use kcommon::*;

#[inline(always)]
fn bf(h: u16) -> f32 {
    kcommon::bf16::to_f32(h)
}
#[inline(always)]
fn tb(f: f32) -> u16 {
    kcommon::bf16::from_f32(f)
}
#[inline(always)]
fn rb(f: f32) -> f32 {
    kcommon::bf16::round(f)
}
#[inline(always)]
fn ex(x: f32) -> f32 {
    math::exp_det(x)
}
#[inline(always)]
fn sigm(x: f32) -> f32 {
    1.0 / (1.0 + ex(-x))
}
#[inline(always)]
fn th(x: f32) -> f32 {
    let x = if x > 20.0 { 20.0 } else if x < -20.0 { -20.0 } else { x };
    1.0 - 2.0 / (ex(2.0 * x) + 1.0)
}

/// out[r] = bf16( rms( sum_t a_t * taps[r][t] ) * w ), taps bf16 [rows][4][2304] with row stride ldt elements. grid = rows, block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_mix(out: *mut u16, taps: *const u16, w: *const u16, a0: f32, a1: f32, a2: f32, a3: f32, ldt: u32) {
    let red: Shared<f32> = static_shared!(drmix_red, f32, 128);
    let r = bid_x() as usize;
    let base = taps.add(r * ldt as usize);
    let mut v = [0.0f32; 9];
    let mut ss = 0.0f32;
    let mut j = 0;
    while j < 9 {
        let d = tid_x() as usize + 256 * j;
        let mut m = 0.0f32;
        m += bf(*base.add(d)) * a0;
        m += bf(*base.add(2304 + d)) * a1;
        m += bf(*base.add(2 * 2304 + d)) * a2;
        m += bf(*base.add(3 * 2304 + d)) * a3;
        v[j] = m;
        ss += m * m;
        j += 1;
    }
    let tot = warp::block_sum(ss, red);
    let inv = math::rsqrt(tot / 2304.0 + 1e-6);
    j = 0;
    while j < 9 {
        let d = tid_x() as usize + 256 * j;
        *out.add(r * 2304 + d) = tb(v[j] * inv * bf(*w.add(d)));
        j += 1;
    }
}

/// Per-head weighted RMS norm (128, eps 1e-6, output rounded to bf16) then NeoX RoPE (cos / sin tables [pos][64], f32).
/// x bf16 [nrows][nh * 128] (row stride ldx); out row (row_out0 + r) with stride ldo; position of row r = pos0 + r. One warp a (row, head).
/// grid = ceil(nrows * nh / 8), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_hnr(out: *mut u16, x: *const u16, w: *const u16, cos: *const f32, sin: *const f32, nh: u32, nrows: u32,
    ldx: u32, ldo: u32, row_out0: u32, pos0: u32, ring: u32) {
    let wi = bid_x() * 8 + warp_id();
    if wi >= nrows * nh {
        return;
    }
    let r = wi / nh;
    let h = wi % nh;
    let l = lane();
    let xin = x.add((r * ldx + h * 128) as usize);
    let mut v = [0.0f32; 4];
    let mut ss = 0.0f32;
    let mut i = 0;
    while i < 4 {
        v[i] = bf(*xin.add((l + 32 * i as u32) as usize));
        ss += v[i] * v[i];
        i += 1;
    }
    let tot = warp::sum(ss);
    let inv = math::rsqrt(tot / 128.0 + 1e-6);
    i = 0;
    while i < 4 {
        v[i] = rb(v[i] * inv * bf(*w.add((l + 32 * i as u32) as usize)));
        i += 1;
    }
    let p = (pos0 + r) as usize;
    let orow = if ring > 0 { (row_out0 + r) % ring } else { row_out0 + r };
    let dst = out.add((orow * ldo + h * 128) as usize);
    // pairs (d, d + 64): d = l and d = l + 32
    let mut k = 0;
    while k < 2 {
        let d = l + 32 * k as u32;
        let c = *cos.add(p * 64 + d as usize);
        let s = *sin.add(p * 64 + d as usize);
        let x1 = v[k];
        let x2 = v[k + 2];
        let a = x1 * c;
        let b = x2 * s;
        let o1 = a - b;
        let a2 = x2 * c;
        let b2 = x1 * s;
        let o2 = a2 + b2;
        *dst.add(d as usize) = tb(o1);
        *dst.add(d as usize + 64) = tb(o2);
        k += 1;
    }
}

/// Block attention: 8 queries x 24 heads over [draft K/V cache rows lo..m) + the 8 block tokens, GQA 3:1, f32 online softmax.
/// q bf16 [8][24][128]; kc / vc cache [pos][8][128]; kb / vb block [8][8][128]; out bf16 [8][3072].
/// window > 0: ctx position p visible to query i iff (qpos0 + i) - p < window. causal != 0: block key j visible iff j <= i.
/// grid (24, 8), block 256 (8 warps, a warp takes every 8th key).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_attn(out: *mut u16, q: *const u16, kc: *const u16, vc: *const u16, kb: *const u16, vb: *const u16, m: u32,
    qpos0: u32, window: u32, causal: u32, ring: u32) {
    let sh_m: Shared<f32> = static_shared!(dra_m, f32, 32);
    let sh_l: Shared<f32> = static_shared!(dra_l, f32, 32);
    let sh_a: Shared<f32> = static_shared!(dra_a, f32, 4096);
    let head = bid_x();
    let qi = bid_y();
    let g = head / 3;
    let l = lane();
    let w = warp_id();
    let qrow = q.add(((qi * 24 + head) * 128) as usize);
    let mut qv = [0.0f32; 4];
    let mut e = 0;
    while e < 4 {
        qv[e] = bf(*qrow.add((4 * l + e as u32) as usize));
        e += 1;
    }
    let qpos = qpos0 + qi;
    let lo = if window > 0 && qpos + 1 > window { qpos + 1 - window } else { 0 };
    let nctx = if m > lo { m - lo } else { 0 };
    let nblk = if causal != 0 { qi + 1 } else { 8 };
    let nk = nctx + nblk;
    let scale: f32 = 0.088388346;
    let mut mx = -3.0e38f32;
    let mut lsum = 0.0f32;
    let mut acc = [0.0f32; 4];
    let mut t = w;
    while t < nk {
        let (kp, vp) = if t < nctx {
            let mut pos = (lo + t) as usize;
            if ring > 0 {
                pos %= ring as usize;
            }
            (kc.add((pos * 8 + g as usize) * 128), vc.add((pos * 8 + g as usize) * 128))
        } else {
            let j = (t - nctx) as usize;
            (kb.add((j * 8 + g as usize) * 128), vb.add((j * 8 + g as usize) * 128))
        };
        let kw = *(kp.add(4 * l as usize) as *const u64);
        let vw = *(vp.add(4 * l as usize) as *const u64);
        let mut dot = 0.0f32;
        let mut vv = [0.0f32; 4];
        e = 0;
        while e < 4 {
            let kk = bf(((kw >> (16 * e as u64)) & 0xffff) as u16);
            vv[e] = bf(((vw >> (16 * e as u64)) & 0xffff) as u16);
            dot += qv[e] * kk;
            e += 1;
        }
        let s = warp::sum(dot) * scale;
        let mnew = if s > mx { s } else { mx };
        let corr = if mx < -1.0e38 { 0.0 } else { ex(mx - mnew) };
        let p = ex(s - mnew);
        lsum = lsum * corr + p;
        e = 0;
        while e < 4 {
            acc[e] = acc[e] * corr + p * vv[e];
            e += 1;
        }
        mx = mnew;
        t += 8;
    }
    if l == 0 {
        sh_m.set(w, mx);
        sh_l.set(w, lsum);
    }
    e = 0;
    while e < 4 {
        sh_a.set(w * 128 + 4 * l + e as u32, acc[e]);
        e += 1;
    }
    sync_threads();
    let d = tid_x();
    if d < 128 {
        let mut mm = -3.0e38f32;
        let mut i = 0;
        while i < 8 {
            let v = sh_m.at(i);
            if v > mm {
                mm = v;
            }
            i += 1;
        }
        let mut ll = 0.0f32;
        let mut aa = 0.0f32;
        i = 0;
        while i < 8 {
            let v = sh_m.at(i);
            if v > -1.0e38 {
                let f = ex(v - mm);
                ll += sh_l.at(i) * f;
                aa += sh_a.at(i * 128 + d) * f;
            }
            i += 1;
        }
        *out.add((qi * 3072 + head * 128 + d) as usize) = tb(aa / ll);
    }
}

/// Split-K block attention for the FULL draft layer (all queries see all `m` context keys): grid (nsplit, 8 kv heads), block 256, one warp a query
/// (its 3 heads) so the K/V of a kv head are read once for the 24 (query, head) rows instead of once per row. A split takes a range of 32-key chunks
/// (cp.async double buffer, K/V tiles in smem), f32 online softmax per row; split 0 also takes the block's own 8 keys (causal: key j visible to query j' >= j).
/// part: [split][kv head][row = 3 * query + head offset][132] = (m, l, 2 pad, acc[128]). Dynamic shared memory 50176 B. The draft is verified by the target,
/// so the summation order here need not match `dr_attn`; it is deterministic (fixed split / key order).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_attn_split(part: *mut f32, q: *const u16, kc: *const u16, vc: *const u16, kb: *const u16, vb: *const u16, m: u32, causal: u32,
    nsplit: u32) {
    use kcommon::mma::*;
    const QS: u32 = 0;
    const KT: u32 = 12288;
    const TB: u32 = 32 * 272;
    const VT: u32 = KT + 2 * TB;
    const PS: u32 = VT + 2 * TB;
    let smem: Shared<u8> = kcommon::smem::dyn_base();
    let g = bid_y();
    let sp = bid_x();
    let w = warp_id();
    let l = lane();
    let t = tid_x();
    let qs: Shared<f32> = smem.byte_add(QS).cast();
    let ps: Shared<f32> = smem.byte_add(PS).cast();
    let mut i = t;
    while i < 24 * 128 {
        let r = i >> 7;
        let d = i & 127;
        qs.set(i, bf(*q.add((((r / 3) * 24 + 3 * g + r % 3) * 128 + d) as usize)));
        i += 256;
    }
    sync_threads();
    let chunks = (m + 31) / 32;
    let cps = (chunks + nsplit - 1) / nsplit;
    let c0 = if sp * cps < chunks { sp * cps } else { chunks };
    let c1 = if c0 + cps < chunks { c0 + cps } else { chunks };
    let extra = if sp == 0 { 1 } else { 0 };
    let nit = (c1 - c0) + extra;
    let scale: f32 = 0.088388346;
    let mut mx = [-3.0e38f32; 3];
    let mut ls = [0.0f32; 3];
    let mut acc = [[0.0f32; 4]; 3];
    // chunk loader: ctx chunk c (rows pos * 8 + g) or the block chunk (rows j * 8 + g); 32 rows x 16 pieces of 16 B for K and for V
    let fetch = |it: u32, buf: u32| {
        let blk = it >= c1 - c0;
        let c = c0 + it;
        let mut j = t;
        while j < 512 {
            let r = j >> 4;
            let cc = j & 15;
            let (ksrc, vsrc, ok) = if !blk {
                let pos = c * 32 + r;
                (kc.add(((pos * 8 + g) * 128 + cc * 8) as usize), vc.add(((pos * 8 + g) * 128 + cc * 8) as usize), pos < m)
            } else {
                (kb.add(((r * 8 + g) * 128 + cc * 8) as usize), vb.add(((r * 8 + g) * 128 + cc * 8) as usize), r < 8)
            };
            let kd = smem.byte_add(KT + buf * TB + r * 272 + cc * 16).addr();
            let vd = smem.byte_add(VT + buf * TB + r * 272 + cc * 16).addr();
            if ok {
                cp_async_16(kd, ksrc as *const u8);
                cp_async_16(vd, vsrc as *const u8);
            } else {
                cp_async_16_zfill(kd, kc as *const u8, 0);
                cp_async_16_zfill(vd, vc as *const u8, 0);
            }
            j += 256;
        }
    };
    if nit > 0 {
        fetch(0, 0);
    }
    cp_async_commit();
    let mut it = 0;
    while it < nit {
        let buf = it & 1;
        if it + 1 < nit {
            fetch(it + 1, 1 - buf);
        }
        cp_async_commit();
        cp_async_wait::<1>();
        sync_threads();
        let blk = it >= c1 - c0;
        let key = if blk { l } else { (c0 + it) * 32 + l };
        let valid = if blk { l < 8 && (causal == 0 || l <= w) } else { key < m };
        // scores of this lane's key for the warp's 3 rows
        let kr: Shared<u8> = smem.byte_add(KT + buf * TB + l * 272);
        let mut dot = [0.0f32; 3];
        let mut d = 0;
        while d < 128 {
            let kw = <[u32; 4] as kcommon::smem::SmemIo>::lds(kr.byte_add(2 * d).addr());
            let kf = [bf16::lo(kw[0]), bf16::hi(kw[0]), bf16::lo(kw[1]), bf16::hi(kw[1]), bf16::lo(kw[2]), bf16::hi(kw[2]), bf16::lo(kw[3]), bf16::hi(kw[3])];
            let mut r = 0;
            while r < 3 {
                let q0 = <[f32; 4] as kcommon::smem::SmemIo>::lds(qs.add((w * 3 + r) * 128 + d).addr());
                let q1 = <[f32; 4] as kcommon::smem::SmemIo>::lds(qs.add((w * 3 + r) * 128 + d + 4).addr());
                let mut a = dot[r as usize];
                a = math::fma(q0[0], kf[0], a);
                a = math::fma(q0[1], kf[1], a);
                a = math::fma(q0[2], kf[2], a);
                a = math::fma(q0[3], kf[3], a);
                a = math::fma(q1[0], kf[4], a);
                a = math::fma(q1[1], kf[5], a);
                a = math::fma(q1[2], kf[6], a);
                a = math::fma(q1[3], kf[7], a);
                dot[r as usize] = a;
                r += 1;
            }
            d += 8;
        }
        let mut r = 0;
        while r < 3 {
            let sv = if valid { dot[r] * scale } else { -3.0e38 };
            let cmax = warp::max(sv);
            let mnew = if cmax > mx[r] { cmax } else { mx[r] };
            let corr = if mx[r] < -1.0e38 { 0.0 } else { ex(mx[r] - mnew) };
            let p = if valid { ex(sv - mnew) } else { 0.0 };
            let csum = warp::sum(p);
            ls[r] = ls[r] * corr + csum;
            mx[r] = mnew;
            ps.set((w * 3 + r as u32) * 32 + l, p);
            let mut e = 0;
            while e < 4 {
                acc[r][e] *= corr;
                e += 1;
            }
            r += 1;
        }
        sync_warp();
        let vt: Shared<u8> = smem.byte_add(VT + buf * TB);
        let mut k = 0;
        while k < 32 {
            let vw = <[u32; 2] as kcommon::smem::SmemIo>::lds(vt.byte_add(k * 272 + 8 * l).addr());
            let vv = [bf16::lo(vw[0]), bf16::hi(vw[0]), bf16::lo(vw[1]), bf16::hi(vw[1])];
            let mut r = 0;
            while r < 3 {
                let p = ps.at((w * 3 + r as u32) * 32 + k);
                let mut e = 0;
                while e < 4 {
                    acc[r][e] = math::fma(p, vv[e], acc[r][e]);
                    e += 1;
                }
                r += 1;
            }
            k += 1;
        }
        sync_threads();
        it += 1;
    }
    let mut r = 0;
    while r < 3 {
        let base = part.add(((((sp * 8 + g) * 24) + w * 3 + r as u32) * 132) as usize);
        if l == 0 {
            *base = mx[r];
            *base.add(1) = ls[r];
        }
        let o = base.add(4 + 4 * l as usize);
        *o = acc[r][0];
        *o.add(1) = acc[r][1];
        *o.add(2) = acc[r][2];
        *o.add(3) = acc[r][3];
        r += 1;
    }
}

/// Combine the splits: grid (24 heads, 8 queries), block 128 (a channel each): out[query][head * 128 + d] = bf16(sum / l), splits without keys skipped.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_attn_comb(out: *mut u16, part: *const f32, nsplit: u32) {
    let head = bid_x();
    let qi = bid_y();
    let d = tid_x();
    let g = head / 3;
    let row = qi * 3 + head % 3;
    let mut mm = -3.0e38f32;
    let mut sp = 0;
    while sp < nsplit {
        let v = *part.add((((sp * 8 + g) * 24 + row) * 132) as usize);
        if v > mm {
            mm = v;
        }
        sp += 1;
    }
    let mut ll = 0.0f32;
    let mut aa = 0.0f32;
    sp = 0;
    while sp < nsplit {
        let b = part.add((((sp * 8 + g) * 24 + row) * 132) as usize);
        let v = *b;
        if v > -1.0e38 {
            let f = ex(v - mm);
            ll += *b.add(1) * f;
            aa += *b.add(4 + d as usize) * f;
        }
        sp += 1;
    }
    *out.add((qi * 3072 + head * 128 + d) as usize) = tb(aa / ll);
}

/// e4m3 (sign, 4 exponent bits bias 7, 3 mantissa bits) to f32.
#[inline(always)]
fn e4m3(b: u32) -> f32 {
    let e = (b >> 3) & 15;
    let m = b & 7;
    let v = if e == 0 { m as f32 * (1.0 / 512.0) } else { (1.0 + m as f32 * 0.125) * f32::from_bits((e + 120) << 23) };
    if (b >> 7) & 1 == 1 { -v } else { v }
}

/// out = bf16(a + b), n elements.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_add(out: *mut u16, a: *const u16, b: *const u16, n: u32) {
    let mut i = bid_x() * bdim_x() + tid_x();
    while i < n {
        *out.add(i as usize) = tb(bf(*a.add(i as usize)) + bf(*b.add(i as usize)));
        i += gdim_x() * bdim_x();
    }
}

/// out = bf16(silu(g) * u), g and u bf16 rows (f32 math), n elements.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_silu_mul(out: *mut u16, g: *const u16, u: *const u16, n: u32) {
    let mut i = bid_x() * bdim_x() + tid_x();
    while i < n {
        let gv = bf(*g.add(i as usize));
        let s = gv / (1.0 + ex(-gv));
        *out.add(i as usize) = tb(s * bf(*u.add(i as usize)));
        i += gdim_x() * bdim_x();
    }
}

/// inp[5120] = [state (256) | markov_w1[tok] (256) | y (2304) | f_rnn (2304)], bf16. grid 1, block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_rnn_in(out: *mut u16, state: *const u16, w1: *const u16, tok: *const u32, y: *const u16, f: *const u16) {
    let t = *tok as usize;
    let mut i = tid_x() as usize;
    while i < 5120 {
        let v = if i < 256 { *state.add(i) } else if i < 512 { *w1.add(t * 256 + i - 256) } else if i < 2816 { *y.add(i - 512) } else { *f.add(i - 2816) };
        *out.add(i) = v;
        i += 256;
    }
}

/// FP8 variant of `dr_rnn_in`: the markov_w1 row is e4m3 bytes with one f32 scale per row; the value is bf16(scale * e4m3).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_rnn_in8(out: *mut u16, state: *const u16, w1q: *const u8, w1s: *const f32, tok: *const u32, y: *const u16, f: *const u16) {
    let t = *tok as usize;
    let sc = *w1s.add(t);
    let mut i = tid_x() as usize;
    while i < 5120 {
        let v = if i < 256 { *state.add(i) } else if i < 512 { tb(sc * e4m3(*w1q.add(t * 256 + i - 256) as u32)) } else if i < 2816 { *y.add(i - 512) } else { *f.add(i - 2816) };
        *out.add(i) = v;
        i += 256;
    }
}

/// out[o] = bf16(sum_k inp[k] * w[o][k] + b[o]) as f32, w bf16 [n][k]. grid = n, block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_gemv_bias(out: *mut f32, inp: *const u16, w: *const u16, b: *const u16, k: u32) {
    let red: Shared<f32> = static_shared!(drg_red, f32, 128);
    let o = bid_x() as usize;
    let mut acc = 0.0f32;
    let mut i = tid_x();
    while i < k {
        acc += bf(*inp.add(i as usize)) * bf(*w.add(o * k as usize + i as usize));
        i += 256;
    }
    let tot = warp::block_sum(acc, red);
    if tid_x() == 0 {
        *out.add(o) = rb(tot + bf(*b.add(o)));
    }
}

/// Markov state update. proj f32 [768] = (gate_raw | cand | out_raw) rounded to bf16 already. grid 1, block 256.
/// gate = bf16(sigmoid); kept = bf16(gate * s); blend = bf16(bf16(1 - gate) * bf16(tanh(cand))); s' = bf16(kept + blend); latent = bf16(tanh(out)).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_rnn_gates(state: *mut u16, latent: *mut u16, proj: *const f32) {
    let r = tid_x() as usize;
    if r < 256 {
        let gate = rb(sigm(*proj.add(r)));
        let cand = *proj.add(256 + r);
        let outr = *proj.add(512 + r);
        let s = bf(*state.add(r));
        let kept = rb(gate * s);
        let blend = rb(rb(1.0 - gate) * rb(th(cand)));
        *state.add(r) = tb(kept + blend);
        *latent.add(r) = tb(th(outr));
    }
}

/// One Markov step, vocabulary part: bias[row] = bf16(latent . fw[row] + fb[row]); val = bf16(30 * tanh((base[row] + bias) / 30));
/// per block the best (val, lowest row) -> pv / pi. fw bf16 [V][256]. One warp a row; grid = ceil(V / 8), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_bias_step(pv: *mut f32, pi: *mut u32, latent: *const u16, fw: *const u16, fb: *const u16, base: *const u16, nv: u32) {
    let sh_v: Shared<f32> = static_shared!(drb_v, f32, 32);
    let sh_i: Shared<u32> = static_shared!(drb_i, u32, 32);
    let w = warp_id();
    let l = lane();
    let row = bid_x() * 8 + w;
    let mut lat = [0.0f32; 8];
    let mut e = 0;
    while e < 8 {
        lat[e] = bf(*latent.add((l * 8 + e as u32) as usize));
        e += 1;
    }
    let mut val = -3.0e38f32;
    if row < nv {
        let wv = bf16::unpack8(ldg_nc_v4(fw.add((row * 256 + l * 8) as usize) as *const u32));
        let mut acc = 0.0f32;
        e = 0;
        while e < 8 {
            acc += lat[e] * wv[e];
            e += 1;
        }
        let tot = warp::sum(acc);
        let bias = rb(tot + bf(*fb.add(row as usize)));
        val = rb(th((bf(*base.add(row as usize)) + bias) / 30.0) * 30.0);
    }
    if l == 0 {
        sh_v.set(w, val);
        sh_i.set(w, row);
    }
    sync_threads();
    if tid_x() == 0 {
        let mut bv = sh_v.at(0);
        let mut bi = sh_i.at(0);
        let mut i = 1;
        while i < 8 {
            let v = sh_v.at(i);
            if v > bv {
                bv = v;
                bi = sh_i.at(i);
            }
            i += 1;
        }
        *pv.add(bid_x() as usize) = bv;
        *pi.add(bid_x() as usize) = bi;
    }
}

/// FP8 variant of `dr_bias_step`: fold weight rows as e4m3 bytes [V][256] with one f32 scale per row (effective weight bf16(scale * e4m3)).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_bias_step8(pv: *mut f32, pi: *mut u32, latent: *const u16, fwq: *const u8, fws: *const f32, fb: *const u16, base: *const u16, nv: u32) {
    let sh_v: Shared<f32> = static_shared!(drb8_v, f32, 32);
    let sh_i: Shared<u32> = static_shared!(drb8_i, u32, 32);
    let w = warp_id();
    let l = lane();
    let row = bid_x() * 8 + w;
    let mut lat = [0.0f32; 8];
    let mut e = 0;
    while e < 8 {
        lat[e] = bf(*latent.add((l * 8 + e as u32) as usize));
        e += 1;
    }
    let mut val = -3.0e38f32;
    if row < nv {
        let q = *(fwq.add((row * 256 + l * 8) as usize) as *const u64);
        let sc = *fws.add(row as usize);
        let mut acc = 0.0f32;
        e = 0;
        while e < 8 {
            let b = ((q >> (8 * e as u64)) & 0xff) as u32;
            acc += lat[e] * rb(sc * e4m3(b));
            e += 1;
        }
        let tot = warp::sum(acc);
        let bias = rb(tot + bf(*fb.add(row as usize)));
        val = rb(th((bf(*base.add(row as usize)) + bias) / 30.0) * 30.0);
    }
    if l == 0 {
        sh_v.set(w, val);
        sh_i.set(w, row);
    }
    sync_threads();
    if tid_x() == 0 {
        let mut bv = sh_v.at(0);
        let mut bi = sh_i.at(0);
        let mut i = 1;
        while i < 8 {
            let v = sh_v.at(i);
            if v > bv {
                bv = v;
                bi = sh_i.at(i);
            }
            i += 1;
        }
        *pv.add(bid_x() as usize) = bv;
        *pi.add(bid_x() as usize) = bi;
    }
}

/// tokbuf[step + 1] = argmax over the np per-block winners (largest value, lowest row on ties). grid 1, block 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_argmax(tokbuf: *mut u32, step: u32, pv: *const f32, pi: *const u32, np: u32) {
    let sh_v: Shared<f32> = static_shared!(dam_v, f32, 128);
    let sh_i: Shared<u32> = static_shared!(dam_i, u32, 128);
    let mut bv = -3.0e38f32;
    let mut bi = 0xffff_ffffu32;
    let mut i = tid_x();
    while i < np {
        let v = *pv.add(i as usize);
        let ix = *pi.add(i as usize);
        if v > bv || (v == bv && ix < bi) {
            bv = v;
            bi = ix;
        }
        i += 1024;
    }
    // warp reduce (value, index)
    let mut o = 16;
    while o > 0 {
        let v2 = warp::shfl_xor(bv, o);
        let i2 = warp::shfl_xor_u32(bi, o);
        if v2 > bv || (v2 == bv && i2 < bi) {
            bv = v2;
            bi = i2;
        }
        o >>= 1;
    }
    if lane() == 0 {
        sh_v.set(warp_id(), bv);
        sh_i.set(warp_id(), bi);
    }
    sync_threads();
    if tid_x() == 0 {
        let mut b = sh_v.at(0);
        let mut ix = sh_i.at(0);
        let mut w = 1;
        while w < 32 {
            let v = sh_v.at(w);
            let j = sh_i.at(w);
            if v > b || (v == b && j < ix) {
                b = v;
                ix = j;
            }
            w += 1;
        }
        *tokbuf.add(step as usize + 1) = ix;
    }
}

/// out[r] = index of the largest bf16 logit of row r (lowest index on ties). logits bf16 [rows][nv]. grid = rows, block 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_argmax_rows(out: *mut u32, logits: *const u16, nv: u32) {
    let sh_v: Shared<f32> = static_shared!(dar_v, f32, 128);
    let sh_i: Shared<u32> = static_shared!(dar_i, u32, 128);
    let r = bid_x() as usize;
    let row = logits.add(r * nv as usize);
    let mut bv = -3.0e38f32;
    let mut bi = 0xffff_ffffu32;
    let mut i = tid_x();
    while i < nv {
        let v = bf(*row.add(i as usize));
        if v > bv {
            bv = v;
            bi = i;
        }
        i += 1024;
    }
    let mut o = 16;
    while o > 0 {
        let v2 = warp::shfl_xor(bv, o);
        let i2 = warp::shfl_xor_u32(bi, o);
        if v2 > bv || (v2 == bv && i2 < bi) {
            bv = v2;
            bi = i2;
        }
        o >>= 1;
    }
    if lane() == 0 {
        sh_v.set(warp_id(), bv);
        sh_i.set(warp_id(), bi);
    }
    sync_threads();
    if tid_x() == 0 {
        let mut b = sh_v.at(0);
        let mut ix = sh_i.at(0);
        let mut w = 1;
        while w < 32 {
            let v = sh_v.at(w);
            let j = sh_i.at(w);
            if v > b || (v == b && j < ix) {
                b = v;
                ix = j;
            }
            w += 1;
        }
        *out.add(r) = ix;
    }
}

/// Sampling statistics of speculative rows: per row r of bf16 logits [rows][nv]: z = 30 * tanh(l / 30) * inv_t; mx[r] = max z; zs[r] = sum exp(z - mx);
/// pd[r] = exp(z[draft[r]] - mx) / zs when draft[r] < nv (0 otherwise). grid = rows, block 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_rows_stats(pd: *mut f32, mx_out: *mut f32, zs_out: *mut f32, logits: *const u16, nv: u32, inv_t: f32, draft: *const u32) {
    let sh: Shared<f32> = static_shared!(drs_a, f32, 128);
    let r = bid_x() as usize;
    let row = logits.add(r * nv as usize);
    let mut m = -3.0e38f32;
    let mut i = tid_x();
    while i < nv {
        let z = 30.0 * th(bf(*row.add(i as usize)) / 30.0) * inv_t;
        if z > m {
            m = z;
        }
        i += 1024;
    }
    m = warp::max(m);
    if lane() == 0 {
        sh.set(warp_id(), m);
    }
    sync_threads();
    let mut mm = sh.at(0);
    let mut w = 1;
    while w < 32 {
        let v = sh.at(w);
        if v > mm {
            mm = v;
        }
        w += 1;
    }
    sync_threads();
    let mut s = 0.0f32;
    i = tid_x();
    while i < nv {
        let z = 30.0 * th(bf(*row.add(i as usize)) / 30.0) * inv_t;
        s += ex(z - mm);
        i += 1024;
    }
    s = warp::sum(s);
    if lane() == 0 {
        sh.set(warp_id(), s);
    }
    sync_threads();
    if tid_x() == 0 {
        let mut tot = 0.0f32;
        let mut w = 0;
        while w < 32 {
            tot += sh.at(w);
            w += 1;
        }
        *mx_out.add(r) = mm;
        *zs_out.add(r) = tot;
        let d = *draft.add(r);
        *pd.add(r) = if d < nv {
            let z = 30.0 * th(bf(*row.add(d as usize)) / 30.0) * inv_t;
            ex(z - mm) / tot
        } else {
            0.0
        };
    }
}

/// Draw one token from softmax(z) of row `row` with token `skip` removed (skip >= nv: none): inverse CDF at u in [0,1). grid 1, block 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_sample_row(tok_out: *mut u32, logits: *const u16, row: u32, nv: u32, inv_t: f32, mx: *const f32, zs: *const f32, skip: u32, u: f32) {
    let sh: Shared<f32> = static_shared!(drm_s, f32, 4096);
    let sh_c: Shared<u32> = static_shared!(drm_c, u32, 16);
    let sh_r: Shared<f32> = static_shared!(drm_r, f32, 16);
    let lg = logits.add(row as usize * nv as usize);
    let m = *mx.add(row as usize);
    let cs = (nv + 1023) / 1024;
    let t = tid_x();
    let lo = t * cs;
    let hi = if lo + cs < nv { lo + cs } else { nv };
    let mut s = 0.0f32;
    let mut i = lo;
    while i < hi {
        if i != skip {
            let z = 30.0 * th(bf(*lg.add(i as usize)) / 30.0) * inv_t;
            s += ex(z - m);
        }
        i += 1;
    }
    sh.set(t, s);
    sync_threads();
    if t == 0 {
        let mut tot = 0.0f32;
        let mut j = 0;
        while j < 1024 {
            tot += sh.at(j);
            j += 1;
        }
        let mut target = u * tot;
        let mut c = 1023u32;
        j = 0;
        while j < 1024 {
            let v = sh.at(j);
            if target <= v {
                c = j;
                break;
            }
            target -= v;
            j += 1;
        }
        sh_c.set(0, c);
        sh_r.set(0, target);
    }
    sync_threads();
    if t == sh_c.at(0) {
        let mut target = sh_r.at(0);
        let mut last = hi - 1;
        let mut i = lo;
        let mut found = 0xffff_ffffu32;
        while i < hi {
            if i != skip {
                last = i;
                let z = 30.0 * th(bf(*lg.add(i as usize)) / 30.0) * inv_t;
                target -= ex(z - m);
                if target <= 0.0 {
                    found = i;
                    break;
                }
            }
            i += 1;
        }
        *tok_out = if found != 0xffff_ffff { found } else { last };
    }
}

#[inline(always)]
unsafe fn cfgf(cfg: *const u32, i: usize) -> f32 {
    f32::from_bits(*cfg.add(i))
}

#[inline(always)]
unsafe fn pen_apply(z: *mut f32, tok: u32, do_rep: bool, rep: f32, sub: f32) {
    let mut v = *z.add(tok as usize);
    if do_rep {
        v = if v > 0.0 { v / rep } else { v * rep };
    }
    *z.add(tok as usize) = v - sub;
}

#[derive(Clone, Copy)]
struct Kset {
    hk: f32,
    lk: f32,
    ck: u32,
    kon: bool,
    cut: f32,
    hp: f32,
    lp: f32,
    cp: u32,
    pon: bool,
}

/// Membership in the kept set, with the server Sampler's tie rule: top-k / top-p keep the tokens above the boundary value and, among the tokens AT the boundary
/// (the tie group [l, h)), the lowest indices (the Sampler sorts by (logit desc, token asc) and truncates).
#[inline(always)]
fn kept(i: u32, s: f32, k: &Kset, with_p: bool) -> bool {
    if s < -20.7 || s < k.cut {
        return false;
    }
    if k.kon && !(s >= k.hk || (s >= k.lk && i <= k.ck)) {
        return false;
    }
    if with_p && k.pon && !(s >= k.hp || (s >= k.lp && i <= k.cp)) {
        return false;
    }
    true
}

/// sum over kept-without-p tokens with s >= tau of (count mode: 1, mass mode: exp(s)).
unsafe fn weight_ge(sr: *const f32, nv: u32, tau: f32, k: &Kset, mass: bool, red: Shared<f32>) -> f32 {
    let mut c = 0.0f32;
    let mut i = tid_x();
    while i < nv {
        let v = *sr.add(i as usize);
        if v >= tau && kept(i, v, k, false) {
            c += if mass { ex(v) } else { 1.0 };
        }
        i += 1024;
    }
    warp::block_sum(c, red)
}

/// Boundary of the prefix of kept-without-p tokens (sorted by s descending, index ascending) holding `target` weight: returns (h, l, cidx) with tokens s >= h all inside,
/// the tie group G = [l, h) contributing its lowest-index tokens up to and including `cidx`.
unsafe fn find_cut(sr: *const f32, nv: u32, k: &Kset, mass: bool, target: f32, red: Shared<f32>, pre: Shared<f32>, sc: Shared<u32>, sr2: Shared<f32>) -> (f32, f32, u32) {
    let (h, l);
    let w0 = weight_ge(sr, nv, 0.0, k, mass, red);
    if w0 >= target {
        h = 1.0f32;
        l = 0.0f32;
    } else {
        let mut a = -20.7f32;
        let mut b = 0.0f32;
        let mut it = 0;
        while it < 24 {
            let mid = 0.5 * (a + b);
            if weight_ge(sr, nv, mid, k, mass, red) >= target {
                a = mid;
            } else {
                b = mid;
            }
            it += 1;
        }
        l = a;
        h = b;
    }
    let whi = if h > 0.5 { 0.0 } else { weight_ge(sr, nv, h, k, mass, red) };
    let extra = target - whi;
    // lowest-index prefix of the group G = kept tokens with l <= s < h
    let cs = (nv + 1023) / 1024;
    let t = tid_x();
    let lo = t * cs;
    let hi = if lo + cs < nv { lo + cs } else { nv };
    let mut sum = 0.0f32;
    let mut i = lo;
    while i < hi {
        let v = *sr.add(i as usize);
        if v >= l && v < h && kept(i, v, k, false) {
            sum += if mass { ex(v) } else { 1.0 };
        }
        i += 1;
    }
    pre.set(t, sum);
    sync_threads();
    if t == 0 {
        let mut rem = extra;
        let mut c = 0xffff_ffffu32;
        let mut lastnz = 0xffff_ffffu32;
        let mut j = 0;
        while j < 1024 {
            let v = pre.at(j);
            if v > 0.0 {
                lastnz = j;
                if rem <= v {
                    c = j;
                    break;
                }
                rem -= v;
            }
            j += 1;
        }
        if c == 0xffff_ffff {
            c = lastnz;
        }
        sc.set(0, c);
        sr2.set(0, rem);
    }
    sync_threads();
    let c = sc.at(0);
    let mut cidx = nv; // default: all of G
    if c != 0xffff_ffff && t == c {
        let mut rem = sr2.at(0);
        let mut i = lo;
        let mut found = hi.wrapping_sub(1);
        while i < hi {
            let v = *sr.add(i as usize);
            if v >= l && v < h && kept(i, v, k, false) {
                found = i;
                rem -= if mass { ex(v) } else { 1.0 };
                if rem <= 0.0 {
                    break;
                }
            }
            i += 1;
        }
        // publish through shared
        sr2.set(1, found as f32);
        cidx = found;
    }
    sync_threads();
    if c != 0xffff_ffff {
        cidx = sr2.at(1) as u32;
    }
    sync_threads();
    (h, l, cidx)
}

/// Final sampling distribution of speculative rows (the server's Sampler semantics): per row r of bf16 logits: z = 30 tanh(l/30); penalties (repetition over the
/// seen set, presence / frequency over the output counts: base table `tab` [ntab][tok, count, seen] + the row's own drafted-prefix extras `ext` [row][7][tok, add, base_seen,
/// base_count]); s = (z - max) / T; kept set = floor -20.7, top-k, min-p, top-p in that order with the Sampler's tie rule (see `kept`).
/// Writes s (f32 [rows][nv], scratch for the draw), outf[r*8..]: hk, lk, cut, hp, lp, Z, pd (kept probability of draft[r]); outu[r*4..]: lowest argmax, ck, cp, flags (1 = top-k
/// active, 2 = top-p active). Greedy (cfg greedy != 0): the kept set is the lowest-index argmax only.
/// cfg words: inv_t, greedy, top_k, top_p, min_p, rep, pres, freq (f32 bits). grid = rows, block 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_filter_prep(s: *mut f32, logits: *const u16, nv: u32, cfg: *const u32, tab: *const u32, ntab: u32, ext: *const u32,
    next: *const u32, draft: *const u32, outf: *mut f32, outu: *mut u32, ban: *const u32, nban: u32) {
    let red: Shared<f32> = static_shared!(drf_red, f32, 128);
    let pre: Shared<f32> = static_shared!(drf_pre, f32, 4096);
    let sc: Shared<u32> = static_shared!(drf_sc, u32, 16);
    let sr2: Shared<f32> = static_shared!(drf_sr, f32, 16);
    let r = bid_x() as usize;
    let sr = s.add(r * nv as usize);
    let row = logits.add(r * nv as usize);
    let inv_t = cfgf(cfg, 0);
    let greedy = *cfg.add(1) != 0;
    let top_k = *cfg.add(2);
    let top_p = cfgf(cfg, 3);
    let min_p = cfgf(cfg, 4);
    let rep = cfgf(cfg, 5);
    let pres = cfgf(cfg, 6);
    let freq = cfgf(cfg, 7);
    let mut i = tid_x();
    while i < nv {
        *sr.add(i as usize) = 30.0 * th(bf(*row.add(i as usize)) / 30.0);
        i += 1024;
    }
    sync_threads();
    if rep != 1.0 || pres != 0.0 || freq != 0.0 {
        let mut e = tid_x();
        while e < ntab {
            let tok = *tab.add(3 * e as usize);
            let cnt = *tab.add(3 * e as usize + 1);
            let seen = *tab.add(3 * e as usize + 2);
            let sub = if cnt > 0 { pres + freq * cnt as f32 } else { 0.0 };
            pen_apply(sr, tok, rep != 1.0 && seen != 0, rep, sub);
            e += 1024;
        }
        sync_threads();
        let ne = *next.add(r);
        if tid_x() < ne {
            let o = (r * 7 + tid_x() as usize) * 4;
            let tok = *ext.add(o);
            let add = *ext.add(o + 1);
            let bseen = *ext.add(o + 2);
            let bcnt = *ext.add(o + 3);
            let sub = freq * add as f32 + if bcnt == 0 { pres } else { 0.0 };
            pen_apply(sr, tok, rep != 1.0 && bseen == 0, rep, sub);
        }
        sync_threads();
    }
    if tid_x() < nban {
        *sr.add(*ban.add(tid_x() as usize) as usize) = -1.0e30;
    }
    sync_threads();
    let mut m = -3.0e38f32;
    i = tid_x();
    while i < nv {
        let v = *sr.add(i as usize);
        if v > m {
            m = v;
        }
        i += 1024;
    }
    let mx = warp::block_max(m, red);
    let mut lo = 3.0e38f32;
    i = tid_x();
    while i < nv {
        if *sr.add(i as usize) == mx {
            lo = i as f32;
            break;
        }
        i += 1024;
    }
    let amin = -warp::block_max(-lo, red);
    let d = *draft.add(r);
    if greedy {
        if tid_x() == 0 {
            *outf.add(r * 8 + 6) = if d < nv && (d as f32) == amin { 1.0 } else { 0.0 };
            *outu.add(r * 4) = amin as u32;
        }
        return;
    }
    i = tid_x();
    while i < nv {
        *sr.add(i as usize) = (*sr.add(i as usize) - mx) * inv_t;
        i += 1024;
    }
    sync_threads();
    let mut ks = Kset { hk: 1.0, lk: 0.0, ck: 0, kon: false, cut: -20.7, hp: 1.0, lp: 0.0, cp: 0, pon: false };
    if top_k > 0 && top_k < nv {
        let kf = top_k as f32;
        let cnt = weight_ge(sr, nv, -20.7, &ks, false, red);
        if cnt > kf {
            let (h, l, c) = find_cut(sr, nv, &ks, false, kf, red, pre, sc, sr2);
            ks.hk = h;
            ks.lk = l;
            ks.ck = c;
            ks.kon = true;
        }
    }
    if min_p > 0.0 {
        let mp = if min_p > 1.0 { 1.0 } else { min_p };
        let mut cut = math::ln(mp);
        if cut > 0.0 {
            cut = 0.0;
        }
        if cut > ks.cut {
            ks.cut = cut;
        }
    }
    let mut total = weight_ge(sr, nv, -20.7, &ks, true, red);
    if top_p < 1.0 {
        let target = top_p * total;
        let (h, l, c) = find_cut(sr, nv, &ks, true, target, red, pre, sc, sr2);
        ks.hp = h;
        ks.lp = l;
        ks.cp = c;
        ks.pon = true;
        let mut z1 = 0.0f32;
        i = tid_x();
        while i < nv {
            let v = *sr.add(i as usize);
            if kept(i, v, &ks, true) {
                z1 += ex(v);
            }
            i += 1024;
        }
        total = warp::block_sum(z1, red);
    }
    if tid_x() == 0 {
        let o = r * 8;
        *outf.add(o) = ks.hk;
        *outf.add(o + 1) = ks.lk;
        *outf.add(o + 2) = ks.cut;
        *outf.add(o + 3) = ks.hp;
        *outf.add(o + 4) = ks.lp;
        *outf.add(o + 5) = total;
        let mut p = 0.0f32;
        if d < nv {
            let v = *sr.add(d as usize);
            if kept(d, v, &ks, true) {
                p = ex(v) / total;
            }
        }
        *outf.add(o + 6) = p;
        *outu.add(r * 4) = amin as u32;
        *outu.add(r * 4 + 1) = ks.ck;
        *outu.add(r * 4 + 2) = ks.cp;
        *outu.add(r * 4 + 3) = ks.kon as u32 | ((ks.pon as u32) << 1);
    }
}

/// Draw from the kept set of row `row` with token `skip` removed, inverse CDF at u; greedy: the stored lowest argmax. grid 1, block 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dr_sample_filtered(tok_out: *mut u32, s: *const f32, row: u32, nv: u32, outf: *const f32, outu: *const u32, skip: u32, u: f32, greedy: u32) {
    let sh: Shared<f32> = static_shared!(drq_s, f32, 4096);
    let sh_c: Shared<u32> = static_shared!(drq_c, u32, 16);
    let sh_r: Shared<f32> = static_shared!(drq_r, f32, 16);
    let sr = s.add(row as usize * nv as usize);
    let am = *outu.add(row as usize * 4);
    if greedy != 0 {
        if tid_x() == 0 {
            *tok_out = am;
        }
        return;
    }
    let o = row as usize * 8;
    let fl = *outu.add(row as usize * 4 + 3);
    let ks = Kset { hk: *outf.add(o), lk: *outf.add(o + 1), ck: *outu.add(row as usize * 4 + 1), kon: fl & 1 != 0, cut: *outf.add(o + 2), hp: *outf.add(o + 3),
        lp: *outf.add(o + 4), cp: *outu.add(row as usize * 4 + 2), pon: fl & 2 != 0 };
    let cs = (nv + 1023) / 1024;
    let t = tid_x();
    let lo = t * cs;
    let hi = if lo + cs < nv { lo + cs } else { nv };
    let mut sum = 0.0f32;
    let mut i = lo;
    while i < hi {
        let v = *sr.add(i as usize);
        if i != skip && kept(i, v, &ks, true) {
            sum += ex(v);
        }
        i += 1;
    }
    sh.set(t, sum);
    sync_threads();
    if t == 0 {
        let mut tot = 0.0f32;
        let mut j = 0;
        while j < 1024 {
            tot += sh.at(j);
            j += 1;
        }
        let mut target = u * tot;
        let mut c = 0xffff_ffffu32;
        let mut lastnz = 0xffff_ffffu32;
        j = 0;
        while j < 1024 {
            let v = sh.at(j);
            if v > 0.0 {
                lastnz = j;
                if target <= v {
                    c = j;
                    break;
                }
                target -= v;
            }
            j += 1;
        }
        if c == 0xffff_ffff {
            c = lastnz;
        }
        sh_c.set(0, c);
        sh_r.set(0, target);
    }
    sync_threads();
    let c = sh_c.at(0);
    if c == 0xffff_ffff {
        if t == 0 {
            *tok_out = am;
        }
        return;
    }
    if t == c {
        let mut target = sh_r.at(0);
        let mut last = am;
        let mut found = 0xffff_ffffu32;
        let mut i = lo;
        while i < hi {
            let v = *sr.add(i as usize);
            if i != skip && kept(i, v, &ks, true) {
                last = i;
                target -= ex(v);
                if target <= 0.0 {
                    found = i;
                    break;
                }
            }
            i += 1;
        }
        *tok_out = if found != 0xffff_ffff { found } else { last };
    }
}
