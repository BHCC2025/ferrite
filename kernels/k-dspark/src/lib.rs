//! DeepSeek-V4.1 DSpark drafting kernels: the Markov head loop (bias rows, scoring under torch.argmax's order, the
//! cross-rank pick) and the confidence head. Built to one PTX module (`k_dspark`) by crates/fe-kbuild; loaded by
//! fe-dspark's `gpu` module. Host references: fe-dspark's `markov` module.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

extern crate kcommon;

use core::arch::asm;
use kcommon::*;

#[inline(always)]
fn f16_to_f32(h: u16) -> f32 {
    let f: f32;
    unsafe { asm!("cvt.f32.f16 {f}, {h};", f = out(reg32) f, h = in(reg16) h, options(pure, nomem, nostack)) }
    f
}

#[inline(always)]
fn f32_to_f16(f: f32) -> u16 {
    let h: u16;
    unsafe { asm!("cvt.rn.f16.f32 {h}, {f};", h = out(reg16) h, f = in(reg32) f, options(pure, nomem, nostack)) }
    h
}

/// (v1, i1) before (v2, i2) in torch.argmax's order: NaN above everything, then the larger value, ties (and NaN
/// against NaN) to the lower index. A total order: any reduction tree gives the same best.
#[inline(always)]
fn better(v1: f32, i1: u32, v2: f32, i2: u32) -> bool {
    let (n1, n2) = (v1 != v1, v2 != v2);
    if n1 || n2 {
        return if n1 && n2 { i1 < i2 } else { n1 };
    }
    v1 > v2 || (v1 == v2 && i1 < i2)
}

#[inline(always)]
fn warp_best(mut v: f32, mut i: u32) -> (f32, u32) {
    let mut m = 16;
    while m > 0 {
        let ov = warp::shfl_xor(v, m);
        let oi = warp::shfl_xor_u32(i, m);
        if better(ov, oi, v, i) {
            v = ov;
            i = oi;
        }
        m >>= 1;
    }
    (v, i)
}

/// Bias rows of the tokens without a cached row (the GGUF drafter: head and emb are BF16): out[r][v] = fp16(sum_k head[v][k] * fp16(emb[tok[r]][k])), fp32
/// accumulation (each lane 8 consecutive k, then a fixed xor tree). grid (blocks, rows), 256 threads; a warp a
/// vocabulary entry at a time. `rank` = 256 (32 lanes x 8). Rows whose slot[r] >= 0 (cached) do nothing.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn markov_bias(
    out: *mut u16,
    tok: *const u32,
    slot: *const i32,
    emb: *const u16,
    head: *const u16,
    n: u32,
) {
    let r = bid_y() as usize;
    if *slot.add(r) >= 0 {
        return;
    }
    const RANK: usize = 256;
    let t = *tok.add(r) as usize;
    let l = lane() as usize;
    // this lane's 8 embedding values, through fp16 (torch: e.to(float16))
    let ev = ldg_nc_v4(emb.add(t * RANK + l * 8) as *const u32);
    let e = bf16::unpack8(ev);
    let mut ef = [0f32; 8];
    for k in 0..8 {
        ef[k] = f16_to_f32(f32_to_f16(e[k]));
    }
    let warps = (gdim_x() * nwarps()) as usize;
    let mut v = (bid_x() * nwarps() + warp_id()) as usize;
    let o = out.add(r * n as usize);
    while v < n as usize {
        let hv = ldg_nc_v4(head.add(v * RANK + l * 8) as *const u32);
        let mut acc = 0f32;
        for k in 0..4 {
            let w = hv[k];
            acc = math::fma(f32::from_bits(w << 16), ef[2 * k], acc);
            acc = math::fma(f32::from_bits(w & 0xffff_0000), ef[2 * k + 1], acc);
        }
        let s = warp::sum(acc);
        if l == 0 {
            *o.add(v) = f32_to_f16(s);
        }
        v += warps;
    }
}

/// Each row's best (value, global index) of logits + bias over this rank's `n` columns (global ids col0 + c), the bias
/// the token's cached row (slot >= 0) or its staged row. grid (rows, parts), any block size: block (r, p) takes the
/// p-th of `parts` equal column slices and writes pair[p][r] = (value bits, idx) ([parts][rows][2]); markov_pick
/// reduces the parts as it reduces the ranks (the order is a strict total order: the same best whatever the split).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn markov_score(
    pair: *mut u32,
    logits: *const f32,
    ld: u32,
    stage: *const u16,
    cache: *const u16,
    slot: *const i32,
    n: u32,
    col0: u32,
) {
    let red = static_shared!(ms_red, u32, 64 * 4);
    let r = bid_x() as usize;
    let (p, parts) = (bid_y(), gdim_y());
    let chunk = n.div_ceil(parts);
    let end = n.min((p + 1) * chunk);
    let s = *slot.add(r);
    let bias = if s >= 0 { cache.add(s as usize * n as usize) } else { stage.add(r * n as usize) };
    let lg = logits.add(r * ld as usize);
    let (mut bv, mut bi) = (f32::NEG_INFINITY, u32::MAX);
    let mut c = p * chunk + tid_x();
    while c < end {
        let v = *lg.add(c as usize) + f16_to_f32(*bias.add(c as usize));
        let i = col0 + c;
        if bi == u32::MAX || better(v, i, bv, bi) {
            bv = v;
            bi = i;
        }
        c += bdim_x();
    }
    let (wv, wi) = warp_best(bv, bi);
    let w = warp_id();
    if lane() == 0 {
        red.add(2 * w).st(wv.to_bits());
        red.add(2 * w + 1).st(wi);
    }
    sync_threads();
    if w == 0 {
        let nw = nwarps();
        let l = lane();
        let (mut v, mut i) = if l < nw { (f32::from_bits(red.add(2 * l).ld()), red.add(2 * l + 1).ld()) }
            else { (f32::NEG_INFINITY, u32::MAX) };
        if i == u32::MAX {
            v = f32::NEG_INFINITY;
        }
        let (fv, fi) = warp_best(v, i);
        if l == 0 {
            let o = 2 * (p as usize * gdim_x() as usize + r);
            *pair.add(o) = fv.to_bits();
            *pair.add(o + 1) = fi;
        }
    }
}

/// The step's pick from every rank's pairs ([world][rows][2], rank order; with markov_score's column parts `world` is
/// ranks x parts; the lower index wins a tie, so rank 0's):
/// out[r * stride + step + 1] = best, tok[r] = best, slot[r] = slot_of[best]. One block, a thread a row.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn markov_pick(
    out: *mut u32,
    tok: *mut u32,
    slot: *mut i32,
    pairs: *const u32,
    slot_of: *const i32,
    world: u32,
    rows: u32,
    stride: u32,
    step: u32,
) {
    let r = tid_x();
    if r >= rows {
        return;
    }
    let mut bv = f32::from_bits(*pairs.add(2 * r as usize));
    let mut bi = *pairs.add(2 * r as usize + 1);
    let mut k = 1;
    while k < world {
        let p = pairs.add(2 * (k * rows + r) as usize);
        let (v, i) = (f32::from_bits(*p), *p.add(1));
        if better(v, i, bv, bi) {
            bv = v;
            bi = i;
        }
        k += 1;
    }
    *out.add((r * stride + step + 1) as usize) = bi;
    *tok.add(r as usize) = bi;
    *slot.add(r as usize) = *slot_of.add(bi as usize);
}

/// Start of a loop: tok[r] = out[r * stride] (the anchors), slot[r] = slot_of[tok]. One block, a thread a row.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn markov_begin(out: *const u32, tok: *mut u32, slot: *mut i32, slot_of: *const i32,
    rows: u32, stride: u32) {
    let r = tid_x();
    if r < rows {
        let t = *out.add((r * stride) as usize);
        *tok.add(r as usize) = t;
        *slot.add(r as usize) = *slot_of.add(t as usize);
    }
}

/// The confidence head: conf[j] = [xc row, emb[prev]] . w, f32 (w f32 [d + rank], the GGUF conf_proj.weight), for rows j = s * steps + i of
/// `streams` x `steps`: xc row s * block + i, prev = out[s * stride + i] (the token draft i + 1 followed). grid (rows),
/// 256 threads.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn conf_head(
    conf: *mut f32,
    xc: *const u16,
    emb: *const u16,
    out: *const u32,
    w: *const f32,
    d: u32,
    steps: u32,
    block: u32,
    stride: u32,
) {
    let red = static_shared!(ch_red, f32, 32 * 4);
    let j = bid_x();
    let (s, i) = (j / steps, j % steps);
    let x = xc.add(((s * block + i) * d) as usize);
    let prev = *out.add((s * stride + i) as usize) as usize;
    let e = emb.add(prev * 256);
    let mut acc = 0f32;
    let mut c = tid_x();
    while c < d + 256 {
        let v = if c < d { bf16::to_f32(*x.add(c as usize)) } else { bf16::to_f32(*e.add((c - d) as usize)) };
        acc = math::fma(v, *w.add(c as usize), acc);
        c += bdim_x();
    }
    let t = warp::block_sum(acc, red);
    if tid_x() == 0 {
        *conf.add(j as usize) = t;
    }
}

// ---- the drafter pass (DSpark blocks) ---------------------------------------------------------------------------

/// The taps: out[r * ld + c] = bf16((((h0 + h1) + h2) + h3) / 4) of the 4 residual streams h [rows][4][dim] (the mean
/// over the streams of a tap layer's input, TensorFold's _tap order). grid = rows, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dspark_tap(out: *mut u16, ld: u32, h: *const u16, dim: u32) {
    let r = bid_x() as usize;
    let d = dim as usize;
    let hr = h.add(r * 4 * d);
    let mut c = tid_x() as usize;
    while c < d {
        let s = ((bf16::to_f32(*hr.add(c)) + bf16::to_f32(*hr.add(d + c))) + bf16::to_f32(*hr.add(2 * d + c)))
            + bf16::to_f32(*hr.add(3 * d + c));
        *out.add(r * ld as usize + c) = bf16::from_f32(s / 4.0);
        c += bdim_x() as usize;
    }
}

/// The pre-weighted collapse alone: xc[r] = bf16(sum_j pre[r][j] * h[r][j]) (fe-ds's hc_pre_norm before its norm;
/// the confidence head reads it). `pre` rows of stride pre_ld. grid = rows, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dspark_collapse(xc: *mut u16, h: *const u16, pre: *const f32, pre_ld: u32, dim: u32) {
    let r = bid_x() as usize;
    let p = pre.add(r * pre_ld as usize);
    let (p0, p1, p2, p3) = (*p, *p.add(1), *p.add(2), *p.add(3));
    let d = dim as usize;
    let hr = h.add(r * 4 * d);
    let mut i = tid_x() as usize;
    while i < d {
        let v = p0 * bf16::to_f32(*hr.add(i)) + p1 * bf16::to_f32(*hr.add(d + i)) + p2 * bf16::to_f32(*hr.add(2 * d + i))
            + p3 * bf16::to_f32(*hr.add(3 * d + i));
        *xc.add(r * d + i) = bf16::from_f32(v);
        i += bdim_x() as usize;
    }
}

const AD: usize = 512;

/// The drafter blocks' attention (dspark.py _attention): row r (a stream's block row j at position q0 + j) sees its
/// slot's ring positions max(0, wpos + 1 - window) ..= wpos (wpos = q0 - 1: the newest absorbed) and the stream's
/// `block` block keys kvb[b0 ..= b0 + block - 1] (no mask inside the block), with the head's sink in the denominator;
/// then the inverse RoPE of the output's last 64 at the row's position. meta[r] = (slot, pos, wpos + 1, b0).
/// Output in fe-ds's attention layout: o[(h / 8) * o_gstride + r * 4096 + (h % 8) * 512 + d].
/// grid = (heads, rows), block = 128 (4 warps).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dspark_attn(o: *mut u16, o_gstride: u32, q: *const u16, q_ld: u32, ring: *const u16,
    ring_stride: u32, ring_len: u32, window: u32, kvb: *const u16, block: u32, meta: *const u32, sink: *const f32,
    scale: f32, tab: *const f32) {
    let sm = static_shared!(da_m, f32, 32);
    let sl = static_shared!(da_l, f32, 32);
    let sacc = static_shared!(da_acc, f32, 8192); // [4][512]
    let h = bid_x() as usize;
    let r = bid_y() as usize;
    let w = warp_id() as usize;
    let l = lane() as usize;
    let slot = *meta.add(4 * r) as usize;
    let pos = *meta.add(4 * r + 1);
    let wend = *meta.add(4 * r + 2); // newest absorbed position + 1
    let b0 = *meta.add(4 * r + 3) as usize;
    let qp = q.add(r * q_ld as usize + h * AD + l * 16) as *const u32;
    let q0 = bf16::unpack8(ldg_nc_v4(qp));
    let q1 = bf16::unpack8(ldg_nc_v4(qp.add(4)));
    let mut qv = [0.0f32; 16];
    let mut i = 0;
    while i < 8 {
        qv[i] = q0[i];
        qv[8 + i] = q1[i];
        i += 1;
    }
    let first = if wend > window { wend - window } else { 0 };
    let nwin = (wend - first) as usize;
    let ntot = nwin + block as usize;
    let ringb = ring.add(slot * ring_stride as usize);
    let mut m = f32::NEG_INFINITY;
    let mut den = 0.0f32;
    let mut acc = [0.0f32; 16];
    let mut k = w;
    while k < ntot {
        let kp: *const u16 = if k < nwin {
            ringb.add(((first + k as u32) % ring_len) as usize * AD)
        } else {
            kvb.add((b0 + (k - nwin)) * AD)
        };
        let kp = kp.add(l * 16) as *const u32;
        let k0 = bf16::unpack8(ldg_nc_v4(kp));
        let k1 = bf16::unpack8(ldg_nc_v4(kp.add(4)));
        let mut kv = [0.0f32; 16];
        let mut i = 0;
        while i < 8 {
            kv[i] = k0[i];
            kv[8 + i] = k1[i];
            i += 1;
        }
        let mut dot = 0.0f32;
        let mut i = 0;
        while i < 16 {
            dot = math::fma(qv[i], kv[i], dot);
            i += 1;
        }
        let s = warp::sum(dot) * scale;
        let mn = m.max(s);
        let corr = if m == f32::NEG_INFINITY { 0.0 } else { math::exp(m - mn) };
        let p = math::exp(s - mn);
        den = den * corr + p;
        let mut i = 0;
        while i < 16 {
            acc[i] = acc[i] * corr + p * kv[i];
            i += 1;
        }
        m = mn;
        k += 4;
    }
    if l == 0 {
        sm.add(w as u32).st(m);
        sl.add(w as u32).st(den);
    }
    let mut i = 0;
    while i < 16 {
        sacc.add((w * AD + l * 16 + i) as u32).st(acc[i]);
        i += 1;
    }
    sync_threads();
    let sk = *sink.add(h);
    let mut mm = sk;
    let mut j = 0;
    while j < 4 {
        mm = mm.max(sm.add(j).ld());
        j += 1;
    }
    let mut cw = [0.0f32; 4];
    let mut tden = math::exp(sk - mm);
    let mut j = 0;
    while j < 4 {
        let mj = sm.add(j as u32).ld();
        cw[j] = if mj == f32::NEG_INFINITY { 0.0 } else { math::exp(mj - mm) };
        tden += sl.add(j as u32).ld() * cw[j];
        j += 1;
    }
    let t = tid_x() as usize;
    let mut ov = [0.0f32; 4];
    let mut c = 0;
    while c < 4 {
        let d = 4 * t + c;
        let v = sacc.add(d as u32).ld() * cw[0] + sacc.add((AD + d) as u32).ld() * cw[1]
            + sacc.add((2 * AD + d) as u32).ld() * cw[2] + sacc.add((3 * AD + d) as u32).ld() * cw[3];
        ov[c] = bf16::round(v / tden);
        c += 1;
    }
    if 4 * t >= AD - 64 {
        let mut c = 0;
        while c < 4 {
            let pi = (4 * t + c - (AD - 64)) / 2;
            let tb = tab.add(pos as usize * 64 + 2 * pi);
            let (cs, sn) = (*tb, -*tb.add(1));
            let (a, b) = (ov[c], ov[c + 1]);
            ov[c] = a * cs - b * sn;
            ov[c + 1] = a * sn + b * cs;
            c += 2;
        }
    }
    let dst = o.add((h / 8) * o_gstride as usize + r * 4096 + (h % 8) * AD + 4 * t) as *mut u32;
    *dst = bf16::pack2(ov[0], ov[1]);
    *dst.add(1) = bf16::pack2(ov[2], ov[3]);
}

/// F32 router logits (the drafter keeps ffn_gate_inp in F32): y[r][o] = sum_k w[o][k] * x[r][k], f32 fma, each lane
/// k = lane, lane + 32, ... then the fixed warp tree: the same bits whatever the row count. A warp an output;
/// grid = (ceil(n / warps), rows), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dspark_gate32(y: *mut f32, x: *const f32, ldx: u32, w: *const f32, k: u32, n: u32) {
    let r = bid_y() as usize;
    let o = (bid_x() * nwarps() + warp_id()) as usize;
    if o >= n as usize {
        return;
    }
    let l = lane() as usize;
    let xr = x.add(r * ldx as usize);
    let wr = w.add(o * k as usize);
    let mut acc = 0f32;
    let mut i = l;
    while i < k as usize {
        acc = math::fma(*wr.add(i), *xr.add(i), acc);
        i += 32;
    }
    let s = warp::sum(acc);
    if l == 0 {
        *y.add(r * n as usize + o) = s;
    }
}
