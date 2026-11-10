//! DFlash2 drafter kernels for the Arc Pro B70 (z-lab Qwen3.8-27B-DFlash2; host side crates/fe-xdf): the two-tap grouped
//! dynamic convolution, per-head RMSNorm + RoPE with the draft K/V ring write, block attention over the draft cache
//! (sliding window, non-causal inside the block; split-K partials + merge), per-row top-16 (parts + merge), the
//! candidate-selector walk, and the target feature taps (a layer's output rows into the drafter's context buffer).
//! The drafter's linears are XQ tensors on kx-xq's GEMV; norms, the embedding and SwiGLU are kx-core's.
//! Ports of the CUDA kernels in kernels/k-dflash (same arithmetic per element; nothing here has to match another kernel
//! bit for bit: the drafter only proposes tokens, the target verifies every one).
#![no_std]
#![feature(abi_gpu_kernel, portable_simd, f16)]
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use kxcommon::mem::global::vload8_u16;
use kxcommon::sg::Shuffle;
use kxcommon::v::{F32x8, I16x8, I32x8, U16x8};
use kxcommon::{attr, bits, blk, cvt, dpas, id, math, sg, slm, unroll};

const H: u32 = 5120;
const GROUPS: u32 = H / 16; // 320 conv groups of 16 channels
const HD: u32 = 128;
const NQ: u32 = 32;
const NKV: u32 = 8;
const KVD: u32 = NKV * HD; // 1024 bf16 a position in the draft cache

#[inline(always)]
fn bf(v: u16) -> f32 {
    cvt::bf16_to_f32(v)
}
#[inline(always)]
fn to_bf(f: f32) -> u16 {
    cvt::f32_to_bf16(f)
}
#[inline(always)]
fn bfr(f: f32) -> f32 {
    bf(to_bf(f))
}

/// Grouped dynamic conv over a block (two taps, causal inside the block of `blk` rows):
/// y[t][j] = (base[0][j] + c[t][g]) x[t][j] + (t % blk >= 1) (base[1][j] + c[t][GROUPS + g]) x[t-1][j], g = j / 16, with
/// c = coef + coff (row stride ldc; one side's two taps of the kernel projection), base = one side's [2][5120] base kernel.
/// With `res` non-null: res[t][j] = bf16(res + bf16(y)) (the residual add), else out[t][j] = bf16(y).
/// Grid (rows, 5120 / 256), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn df_dynconv(out: *mut u16, x: *const u16, coef: *const u16, ldc: u32, coff: u32, base: *const u16,
    res: *mut u16, blk: u32) {
    attr::reqd_work_group_size::<256, 1, 1>();
    let t = id::group_id(0);
    let j = id::group_id(1) * 256 + id::local_id(0);
    let g = j / 16;
    let c = coef.add((t * ldc + coff) as usize);
    let w0 = bf(*base.add(j as usize)) + bf(*c.add(g as usize));
    let mut y = w0 * bf(*x.add((t * H + j) as usize));
    if t % blk != 0 {
        let w1 = bf(*base.add((H + j) as usize)) + bf(*c.add((GROUPS + g) as usize));
        y += w1 * bf(*x.add(((t - 1) * H + j) as usize));
    }
    if res.is_null() {
        *out.add((t * H + j) as usize) = to_bf(y);
    } else {
        let r = res.add((t * H + j) as usize);
        *r = to_bf(bf(*r) + bfr(y));
    }
}

/// Per-head RMSNorm (weight w[128], eps) then rotate-half RoPE, in place on x[row][head * 128] (row stride ldx).
/// rope: [rows][128] f32 = cos[64] | sin[64] of each row's position (host table). With `cache` non-null the result also
/// goes to cache[((pos0 + row) & mask) * 1024 + head * 128] (a draft K ring), and v[row][head * 128] (row stride ldv) to
/// vcache likewise. Grid (rows, heads), local 16 (lane l holds dims l + 16 i).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn df_norm_rope(x: *mut u16, ldx: u32, w: *const u16, eps: f32, rope: *const f32, pos0: u32,
    cache: *mut u16, vcache: *mut u16, v: *const u16, ldv: u32, mask: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    let r = id::group_id(0);
    let hh = id::group_id(1);
    let l = id::lane();
    let p = x.add((r * ldx + hh * HD) as usize);
    let mut e = [0f32; 8];
    unroll!(8, 8, |i| e[i] = bf(*p.add((l + 16 * i as u32) as usize)));
    let mut ss = 0f32;
    unroll!(8, 8, |i| ss += e[i] * e[i]);
    let ss = sg::reduce_add_f32(ss);
    let inv = math::rsqrt(ss / HD as f32 + eps);
    unroll!(8, 8, |i| e[i] = bfr(bf(*w.add((l + 16 * i as u32) as usize)) * bfr(e[i] * inv)));
    let rp = rope.add((r * HD) as usize);
    let mut y = [0f32; 8];
    // dims d = l + 16 i (i < 4) pair with d + 64 (i + 4)
    unroll!(4, 4, |i| {
        let d = (l + 16 * i as u32) as usize;
        let (c, s) = (*rp.add(d), *rp.add(64 + d));
        y[i] = e[i] * c - e[i + 4] * s;
        y[i + 4] = e[i + 4] * c + e[i] * s;
    });
    let at = (((pos0 + r) & mask) * KVD + hh * HD) as usize;
    unroll!(8, 8, |i| {
        let b = to_bf(y[i]);
        let d = (l + 16 * i as u32) as usize;
        *p.add(d) = b;
        if !cache.is_null() {
            *cache.add(at + d) = b;
        }
    });
    if !vcache.is_null() {
        unroll!(8, 8, |i| {
            let d = (l + 16 * i as u32) as usize;
            *vcache.add(at + d) = *v.add((r * ldv + hh * HD) as usize + d);
        });
    }
}

/// Merges df_attn_dpas's partials of (q head, row): out[row][qh * 128 ..] bf16. Grid (32, rows), local 16.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn df_attn_comb(out: *mut u16, part: *const f32, rows: u32, nsplit: u32) {
    attr::reqd_work_group_size::<16, 1, 1>();
    let qh = id::group_id(0);
    let r = id::group_id(1);
    let l = id::lane();
    let mut mx = -1.0e30f32;
    let mut sp = 0;
    while sp < nsplit {
        mx = math::fmax(mx, *part.add((((sp * rows + r) * NQ + qh) * 130) as usize));
        sp += 1;
    }
    let mut tot = 0f32;
    let mut acc = [0f32; 8];
    let mut sp = 0;
    while sp < nsplit {
        let p = part.add((((sp * rows + r) * NQ + qh) * 130) as usize);
        let w = math::exp(*p - mx);
        tot += *p.add(1) * w;
        unroll!(8, 8, |i| acc[i] += *p.add(2 + 8 * l as usize + i) * w);
        sp += 1;
    }
    let inv = 1.0 / tot;
    let o = out.add((r * NQ * HD + qh * HD + 8 * l) as usize);
    unroll!(8, 8, |i| *o.add(i) = to_bf(acc[i] * inv));
}

/// (v, i) ordered before (bv, bi): larger value, then lower id.
#[inline(always)]
fn better(v: f32, i: u32, bv: f32, bi: u32) -> bool {
    v > bv || (v == bv && i < bi)
}

/// Top-16 of a part of a logits row: work-group (row, p) scans logits[row][p * per .. min(vocab, (p + 1) * per)) (fp32,
/// row stride ld) and writes its 16 best (value desc, id asc) to tv / ti [row][parts][16]. Grid (rows, parts), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn df_topk_part(tv: *mut f32, ti: *mut u32, logits: *const f32, ld: u32, vocab: u32, per: u32) {
    attr::reqd_work_group_size::<256, 1, 1>();
    let cv = slm::alloc::<f32, 4096, 0>();
    let ci = slm::alloc::<u32, 4096, 1>();
    let sv = slm::alloc::<f32, 16, 2>();
    let si = slm::alloc::<u32, 16, 3>();
    let r = id::group_id(0);
    let p = id::group_id(1);
    let parts = id::num_groups(1);
    let t = id::local_id(0);
    let row = logits.add(r as usize * ld as usize);
    let mut lv = [f32::NEG_INFINITY; 16];
    let mut li = [u32::MAX; 16];
    let lo = p * per;
    let hi = vocab.min(lo + per);
    let mut i = lo + t;
    while i < hi {
        let v = *row.add(i as usize);
        // a thread's ids ascend, so `>` keeps equal values in id order
        if v > lv[15] {
            let (mut cv_, mut ci_) = (v, i);
            unroll!(16, 16, |j| {
                if cv_ > lv[j] {
                    let (a, b) = (lv[j], li[j]);
                    lv[j] = cv_;
                    li[j] = ci_;
                    cv_ = a;
                    ci_ = b;
                }
            });
        }
        i += 256;
    }
    unroll!(16, 16, |j| {
        *cv.add((t * 16) as usize + j) = lv[j];
        *ci.add((t * 16) as usize + j) = li[j];
    });
    slm::barrier();
    let (mut pv, mut pi) = (f32::INFINITY, 0u32);
    let mut n = 0u32;
    while n < 16 {
        let (mut bv, mut bi) = (f32::NEG_INFINITY, u32::MAX);
        let mut e = t;
        while e < 4096 {
            let (v, idv) = (*cv.add(e as usize), *ci.add(e as usize));
            let after = v < pv || (v == pv && idv > pi);
            if after && better(v, idv, bv, bi) {
                bv = v;
                bi = idv;
            }
            e += 256;
        }
        unroll!(4, 4, |k| {
            let d = 8u32 >> k;
            let ov = bv.shuffle_xor(d);
            let oi = bi.shuffle_xor(d);
            if better(ov, oi, bv, bi) {
                bv = ov;
                bi = oi;
            }
        });
        if id::lane() == 0 {
            *sv.add(id::sg_id() as usize) = bv;
            *si.add(id::sg_id() as usize) = bi;
        }
        slm::barrier();
        let (mut v, mut idv) = (*sv, *si);
        let mut w = 1;
        while w < 16 {
            let (ov, oi) = (*sv.add(w), *si.add(w));
            if better(ov, oi, v, idv) {
                v = ov;
                idv = oi;
            }
            w += 1;
        }
        if t == 0 {
            let o = ((r * parts + p) * 16 + n) as usize;
            *tv.add(o) = v;
            *ti.add(o) = idv;
        }
        pv = v;
        pi = idv;
        slm::barrier();
        n += 1;
    }
}

/// Merges df_topk_part's lists (parts * 16 <= 256 candidates a row) into the row's top-16: vals / ids [row][16].
/// Each work-item takes one candidate and counts the better ones (ids are distinct, so ranks are). Grid (rows), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn df_topk_merge(vals: *mut f32, ids: *mut u32, tv: *const f32, ti: *const u32, parts: u32) {
    attr::reqd_work_group_size::<256, 1, 1>();
    let cv = slm::alloc::<f32, 256, 0>();
    let ci = slm::alloc::<u32, 256, 1>();
    let r = id::group_id(0);
    let t = id::local_id(0);
    let n = parts * 16;
    let (v, i) = if t < n {
        (*tv.add((r * n + t) as usize), *ti.add((r * n + t) as usize))
    } else {
        (f32::NEG_INFINITY, u32::MAX)
    };
    *cv.add(t as usize) = v;
    *ci.add(t as usize) = i;
    slm::barrier();
    if t < n && i != u32::MAX {
        let mut rank = 0u32;
        let mut e = 0;
        while e < n {
            if better(*cv.add(e as usize), *ci.add(e as usize), v, i) {
                rank += 1;
            }
            e += 1;
        }
        if rank < 16 {
            *vals.add((r * 16 + rank) as usize) = v;
            *ids.add((r * 16 + rank) as usize) = i;
        }
    }
}

/// The candidate selector's greedy walk: for i in 0..rows, from predecessor `pred` (the anchor, then the previous pick),
/// score each of row i's 16 candidates c as vals[i][c] + sum_k a_cb[pred][k] gate[i][k] b_cb[cand][k] (rank 256) and
/// take the best (first on ties); out[i] = its id. One work-group of 16 sub-groups (sub-group c = candidate c, lane l
/// holds rank dims 16 l .. 16 l + 16). Grid (1), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn df_select(out: *mut u32, vals: *const f32, ids: *const u32, gate: *const u16, a_cb: *const u16,
    b_cb: *const u16, anchor: u32, rows: u32) {
    attr::reqd_work_group_size::<256, 1, 1>();
    let sc = slm::alloc::<f32, 16, 0>();
    let sp = slm::alloc::<u32, 1, 1>();
    let c = id::sg_id();
    let l = id::lane();
    if id::local_id(0) == 0 {
        *sp = anchor;
    }
    slm::barrier();
    let mut i = 0u32;
    while i < rows {
        let pred = *sp;
        let cand = *ids.add((i * 16 + c) as usize);
        let mut s = 0f32;
        unroll!(2, 2, |h| {
            let off = (16 * l + 8 * h as u32) as usize;
            let a = vload8_u16(0, a_cb.add(pred as usize * 256 + off));
            let b = vload8_u16(0, b_cb.add(cand as usize * 256 + off));
            let g = vload8_u16(0, gate.add((i * 256) as usize + off));
            unroll!(8, 8, |x| s = math::fma(bf(a[x]) * bf(g[x]), bf(b[x]), s));
        });
        let s = sg::reduce_add_f32(s);
        if l == 0 {
            *sc.add(c as usize) = *vals.add((i * 16 + c) as usize) + s;
        }
        slm::barrier();
        if id::local_id(0) == 0 {
            let mut best = 0usize;
            let mut bv = *sc;
            let mut k = 1usize;
            while k < 16 {
                if *sc.add(k) > bv {
                    bv = *sc.add(k);
                    best = k;
                }
                k += 1;
            }
            let id_ = *ids.add((i * 16) as usize + best);
            *out.add(i as usize) = id_;
            *sp = id_;
        }
        slm::barrier();
        i += 1;
    }
}

/// Target feature tap: rows row0 .. of a layer's output bf16(x + d) (x the residual stream, d the pending block output,
/// both [rows][5120]) to taps[(r - row0) * ld + j * 5120 ..] (the drafter's context features of tap layer j).
/// Grid (rows - row0, 5120 / 256), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn df_tap(x: *const u16, d: *const u16, taps: *mut u16, row0: u32, j: u32, ld: u32) {
    attr::reqd_work_group_size::<256, 1, 1>();
    let r = id::group_id(0);
    let i = id::group_id(1) * 256 + id::local_id(0);
    let at = ((row0 + r) * H + i) as usize;
    *taps.add(r as usize * ld as usize + (j * H + i) as usize) = to_bf(bf(*x.add(at)) + bf(*d.add(at)));
}

/// Draft attention on the matrix engine: work-group (kv head, split), sub-group s = 8 queries (query qi = row * 4 + head of
/// the kv head; qi = 8 s .. 8 s + 7), keys 16 at a time. Scores S[8 q][16 keys] = sum over 8 k-steps of DPAS(bf16 Q rows,
/// K^T: lane = key, dims 16 t .. as VNNI dwords straight from the cache row), online softmax per query (component), then
/// O[8 q][16 dims] += DPAS(P as bf16, V^T: lane = dim, staged transposed in SLM per 16 keys). Partials as df_attn_part.
/// q rows ldq elements apart (the fused q | k | v rows). Partials part[((sp * rows + r) * 32 + qh) * 130] = (max, sum, acc[128]).
/// Grid (8, nsplit), local 16 * ceil(4 rows / 8) (rows <= 16).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn df_attn_dpas(part: *mut f32, q: *const u16, kc: *const u16, vc: *const u16, rows: u32, pos0: u32,
    win: u32, scale: f32, mask: u32, nsplit: u32, ldq: u32) {
    let vt = slm::alloc::<u16, { 128 * 16 }, 0>();
    let kvh = id::group_id(0);
    let sp = id::group_id(1);
    let sgi = id::sg_id();
    let l = id::lane();
    let lid = id::local_id(0);
    let nthr = id::local_size(0);
    let nq = 4 * rows;
    // A operands of the scores: component c = query 8 sgi + c, lane = dim 16 t + l
    let mut qa = [I16x8::splat(0); 8];
    let mut lo_c = [0u32; 8];
    unroll!(8, 8, |c| {
        let qi = 8 * sgi + c as u32;
        let (r, h) = (qi / 4, qi % 4);
        let pos = pos0 + r;
        lo_c[c] = if qi >= nq { u32::MAX } else if pos + 1 > win { pos + 1 - win } else { 0 };
    });
    unroll!(8, 8, |t| {
        let mut a = [0i16; 8];
        unroll!(8, 8, |c| {
            let qi = 8 * sgi + c as u32;
            if qi < nq {
                let (r, h) = (qi / 4, qi % 4);
                a[c] = *q.add((r * ldq + (kvh * 4 + h) * HD + 16 * t as u32 + l) as usize) as i16;
            }
        });
        qa[t] = I16x8::from_array(a);
    });
    let lo_u = if pos0 + 1 > win { pos0 + 1 - win } else { 0 };
    let hi = pos0 + rows;
    let len = hi - lo_u;
    let c0 = lo_u + len * sp / nsplit;
    let c1 = lo_u + len * (sp + 1) / nsplit;
    let mut m = [-1.0e30f32; 8];
    let mut ll = [0f32; 8];
    let mut acc = [F32x8::splat(0.0); 8];
    let mut kb = c0;
    while kb < c1 {
        let nk = (c1 - kb).min(16);
        // V^T tile of 16 keys: vt[dim][key]
        let mut e = lid;
        while e < 16 * 16 {
            let (j, cch) = (e / 16, e % 16);
            let src = (((kb + j) & mask) * KVD + kvh * HD + 8 * cch) as usize;
            let v8 = vload8_u16(0, vc.add(src));
            unroll!(8, 8, |i| *vt.add(((8 * cch + i as u32) * 16 + j) as usize) = v8[i]);
            e += nthr;
        }
        slm::barrier();
        // scores: lane = key kb + l
        let kp = kb + l;
        let krow = kc.add(((kp & mask) * KVD + kvh * HD) as usize) as *const u32;
        let mut s = F32x8::splat(0.0);
        unroll!(8, 8, |t| {
            let b: I32x8 = bits(kxcommon::mem::global::vload8_u32(0, krow.add(8 * t)));
            s = dpas::bf16_m8(qa[t], b, s);
        });
        let sa = s.to_array();
        let mut p = [0f32; 8];
        let mut corr = [1f32; 8];
        unroll!(8, 8, |c| {
            let ok = l < nk && kp >= lo_c[c];
            let sc = if ok { sa[c] * scale } else { f32::NEG_INFINITY };
            let mx = math::fmax(m[c], sg::reduce_max_f32(sc));
            corr[c] = math::exp(m[c] - mx);
            p[c] = math::exp(sc - mx);
            ll[c] = ll[c] * corr[c] + p[c];
            m[c] = mx;
        });
        let cv = F32x8::from_array(corr);
        unroll!(8, 8, |bk| acc[bk] *= cv);
        let mut pa = [0i16; 8];
        unroll!(8, 8, |c| pa[c] = cvt::f32_to_bf16(p[c]) as i16);
        let pa = I16x8::from_array(pa);
        unroll!(8, 8, |bk| {
            let b: I32x8 = bits(kxcommon::mem::local::vload8_u32(0, vt.add(((16 * bk as u32 + l) * 16) as usize) as *const u32));
            acc[bk] = dpas::bf16_m8(pa, b, acc[bk]);
        });
        slm::barrier();
        kb += 16;
    }
    unroll!(8, 8, |c| {
        let qi = 8 * sgi + c as u32;
        let tot = sg::reduce_add_f32(ll[c]);
        if qi < nq {
            let (r, h) = (qi / 4, qi % 4);
            let pp = part.add((((sp * rows + r) * NQ + kvh * 4 + h) * 130) as usize);
            if l == 0 {
                *pp = m[c];
                *pp.add(1) = tot;
            }
            unroll!(8, 8, |bk| *pp.add(2 + 16 * bk + l as usize) = acc[bk][c]);
        }
    });
}

/// act[r][i] = bf16(silu(g) * u) from a fused gate | up row ([rows][2 x 17408] -> [rows][17408]; silu = x / (1 + exp(-x)) with
/// the OpenCL division, as kx-core's swiglu). Grid (rows, 17408 / 256), local 256.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn df_swiglu2(gu: *const u16, act: *mut u16) {
    attr::reqd_work_group_size::<256, 1, 1>();
    const INTER: u32 = 17408;
    let r = id::group_id(0);
    let i = id::group_id(1) * 256 + id::local_id(0);
    let g = bf(*gu.add((r * 2 * INTER + i) as usize));
    let u = bf(*gu.add((r * 2 * INTER + INTER + i) as usize));
    *act.add((r * INTER + i) as usize) = to_bf(math::div_ocl(g, 1.0 + math::exp(-g)) * u);
}
