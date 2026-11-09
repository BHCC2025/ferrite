//! DFlash2 drafter kernels (z-lab Qwen3.8-27B-DFlash2): the two-tap grouped dynamic convolution, per-head
//! RMSNorm + RoPE with the draft KV-cache write, block attention over the draft cache (sliding window,
//! non-causal inside the block), per-row top-16 and the candidate-selector walk. Linears, norms, embedding
//! and SwiGLU reuse k-core.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

extern crate kcommon;

use kcommon::smem::Shared;
use kcommon::*;

const H: u32 = 5120;
const GROUP: u32 = 16;
const GROUPS: u32 = H / GROUP; // 320
const HD: u32 = 128;
const NQ: u32 = 32;
const NKV: u32 = 8;
const KVD: u32 = NKV * HD; // 1024 per position in the draft cache

/// Grouped dynamic causal conv over a block (two taps, causal inside the block):
/// y[t][j] = (base[0][j] + c[t][g]) x[t][j] + (t >= 1) (base[1][j] + c[t][GROUPS + g]) x[t-1][j], g = j / 16,
/// with c = coef + coff (stride ldc; one side's two taps of the kernel projection) and base = one side's
/// [2][5120] base kernel. With `res` non-null: res[t][j] = bf16(res + y) (the residual add), else out = y.
/// grid = (rows, 5120 / 256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_dynconv(out: *mut u16, x: *const u16, coef: *const u16, ldc: u32, coff: u32,
    base: *const u16, res: *mut u16, blk: u32) {
    let t = bid_x();
    let j = bid_y() * 256 + tid_x();
    let g = j / GROUP;
    let c = coef.add((t * ldc + coff) as usize);
    let w0 = bf16::to_f32(*base.add(j as usize)) + bf16::to_f32(*c.add(g as usize));
    let mut y = w0 * bf16::to_f32(*x.add((t * H + j) as usize));
    if t % blk != 0 {
        let w1 = bf16::to_f32(*base.add((H + j) as usize)) + bf16::to_f32(*c.add((GROUPS + g) as usize));
        y += w1 * bf16::to_f32(*x.add(((t - 1) * H + j) as usize));
    }
    if res.is_null() {
        *out.add((t * H + j) as usize) = bf16::from_f32(y);
    } else {
        let r = res.add((t * H + j) as usize);
        *r = bf16::from_f32(bf16::to_f32(*r) + bf16::round(y));
    }
}

/// Per-head RMSNorm (weight w[128], eps) then rotate-half RoPE at position pos0 + row, in place on
/// x[row][head * 128] (stride ldx). rope: [pos][128] f32 = cos[64] | sin[64]. With `cache` non-null the result
/// row also goes to cache[((pos0 + row) & mask) * ldcache + head * 128] (the draft K cache, a ring of mask + 1
/// positions), and `v` (stride ldv, same heads) is copied to vcache likewise. grid = (rows, heads), block = 32
/// (lane holds dims l, l+32, l+64, l+96).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_norm_rope(x: *mut u16, ldx: u32, w: *const u16, eps: f32, rope: *const f32,
    pos0: u32, cache: *mut u16, vcache: *mut u16, ldcache: u32, v: *const u16, ldv: u32, mask: u32, blk: u32,
    posv: *const u32, cso: *const u32) {
    let r = bid_x();
    // batched streams (posv non-null): row r is row r % blk of stream r / blk, at posv[s] + r % blk, its cache
    // cso[s] elements in
    let (pos, co) = if posv.is_null() { (pos0 + r, 0usize) } else {
        let st = (r / blk) as usize;
        (*posv.add(st) + r % blk, *cso.add(st) as usize)
    };
    let hh = bid_y();
    let l = lane();
    let p = x.add((r * ldx + hh * HD) as usize);
    let mut e = [0.0f32; 4];
    kcommon::unroll!(i in 4 => {
        e[i] = bf16::to_f32(*p.add((l + 32 * i as u32) as usize));
    });
    let ss = warp::sum(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3]);
    let inv = math::rsqrt(ss / HD as f32 + eps);
    kcommon::unroll!(i in 4 => {
        e[i] = bf16::round(bf16::to_f32(*w.add((l + 32 * i as u32) as usize)) * bf16::round(e[i] * inv));
    });
    let rp = rope.add((pos * HD) as usize);
    // pairs (l, l + 64) and (l + 32, l + 96)
    let (c0, s0) = (*rp.add(l as usize), *rp.add((64 + l) as usize));
    let (c1, s1) = (*rp.add((32 + l) as usize), *rp.add((96 + l) as usize));
    let y = [e[0] * c0 - e[2] * s0, e[1] * c1 - e[3] * s1, e[2] * c0 + e[0] * s0, e[3] * c1 + e[1] * s1];
    kcommon::unroll!(i in 4 => {
        let b = bf16::from_f32(y[i]);
        *p.add((l + 32 * i as u32) as usize) = b;
        if !cache.is_null() {
            *cache.add(co + ((pos & mask) * ldcache + hh * HD + l + 32 * i as u32) as usize) = b;
        }
    });
    if !vcache.is_null() {
        kcommon::unroll!(i in 4 => {
            *vcache.add(co + ((pos & mask) * ldcache + hh * HD + l + 32 * i as u32) as usize) =
                *v.add((r * ldv + hh * HD + l + 32 * i as u32) as usize);
        });
    }
}

/// Block attention of the drafter: query row r (position pos0 + r) of q head qh attends to every cached key
/// in [max(0, pos0 + r - (win - 1)), pos0 + rows) (sliding window on the left, the whole block on the right),
/// GQA 32 / 8. q [rows][4096] (normed + roped), caches [pos & mask][1024] (rings of mask + 1 >= win + rows
/// positions), out [rows][4096]. grid = 32 (q heads), block = 32 * rows (warp = query row; lane holds dims 4l..4l+4).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_attn(out: *mut u16, q: *const u16, kc: *const u16, vc: *const u16, rows: u32,
    pos0: u32, win: u32, scale: f32, mask: u32) {
    let qh = bid_x();
    let r = warp_id();
    let l = lane();
    if r >= rows {
        return;
    }
    let kvh = qh / (NQ / NKV);
    let qv = bf16::unpack2(*(q.add((r * NQ * HD + qh * HD + 4 * l) as usize) as *const u32));
    let qw = bf16::unpack2(*(q.add((r * NQ * HD + qh * HD + 4 * l + 2) as usize) as *const u32));
    let qq = [qv.0 * scale, qv.1 * scale, qw.0 * scale, qw.1 * scale];
    let pos = pos0 + r;
    let lo = if pos + 1 > win { pos + 1 - win } else { 0 };
    let hi = pos0 + rows;
    let mut m = -1.0e30f32;
    let mut sum = 0.0f32;
    let mut acc = [0.0f32; 4];
    let mut kp = lo;
    while kp < hi {
        let base = ((kp & mask) * KVD + kvh * HD + 4 * l) as usize;
        let k2 = *(kc.add(base) as *const [u32; 2]);
        let (k0, k1) = bf16::unpack2(k2[0]);
        let (k2a, k3) = bf16::unpack2(k2[1]);
        let s = warp::sum(qq[0] * k0 + qq[1] * k1 + qq[2] * k2a + qq[3] * k3);
        let v2 = *(vc.add(base) as *const [u32; 2]);
        let (v0, v1) = bf16::unpack2(v2[0]);
        let (v2a, v3) = bf16::unpack2(v2[1]);
        let mn = m.max(s);
        let corr = math::exp(m - mn);
        let pe = math::exp(s - mn);
        sum = sum * corr + pe;
        acc[0] = acc[0] * corr + pe * v0;
        acc[1] = acc[1] * corr + pe * v1;
        acc[2] = acc[2] * corr + pe * v2a;
        acc[3] = acc[3] * corr + pe * v3;
        m = mn;
        kp += 1;
    }
    let inv = 1.0 / sum;
    let o = out.add((r * NQ * HD + qh * HD + 4 * l) as usize) as *mut u32;
    *o = bf16::pack2(acc[0] * inv, acc[1] * inv);
    *o.add(1) = bf16::pack2(acc[2] * inv, acc[3] * inv);
}

/// Split-K form of fe_df_attn (nq query heads a row, draft-cache rows of kvd: a TP rank holds half of the heads):
/// block (qh, sp) walks the sp-th of `nsplit` equal parts of each query row's key
/// range and writes its online-softmax partial (max, sum, acc[128]) to part[((sp * rows + r) * 32 + qh) * 130];
/// fe_df_attn_comb merges them. grid = (32, nsplit), block = 32 * rows.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_attn_part(part: *mut f32, q: *const u16, kc: *const u16, vc: *const u16,
    rows: u32, pos0: u32, win: u32, scale: f32, mask: u32, nsplit: u32, posv: *const u32, cso: *const u32, nq: u32,
    kvd: u32) {
    let qh = bid_x();
    let sp = bid_y();
    let r = warp_id();
    let l = lane();
    if r >= rows {
        return;
    }
    // stream z (grid z) of a batch: its block's rows, position and draft cache (posv / cso non-null)
    let z = bid_z();
    let (pos0, kc, vc) = if posv.is_null() { (pos0, kc, vc) } else {
        let co = *cso.add(z as usize) as usize;
        (*posv.add(z as usize), kc.add(co), vc.add(co))
    };
    let q = q.add((z * rows * nq * HD) as usize);
    let part = part.add((z * nsplit * rows * nq * 130) as usize);
    let kvh = qh / (NQ / NKV); // GQA 4: the same for a TP rank's heads
    let qv = bf16::unpack2(*(q.add((r * nq * HD + qh * HD + 4 * l) as usize) as *const u32));
    let qw = bf16::unpack2(*(q.add((r * nq * HD + qh * HD + 4 * l + 2) as usize) as *const u32));
    let qq = [qv.0 * scale, qv.1 * scale, qw.0 * scale, qw.1 * scale];
    let pos = pos0 + r;
    let lo = if pos + 1 > win { pos + 1 - win } else { 0 };
    let len = pos0 + rows - lo;
    let c0 = lo + len * sp / nsplit;
    let c1 = lo + len * (sp + 1) / nsplit;
    let mut m = -1.0e30f32;
    let mut sum = 0.0f32;
    let mut acc = [0.0f32; 4];
    let mut kp = c0;
    while kp < c1 {
        let base = ((kp & mask) * kvd + kvh * HD + 4 * l) as usize;
        let k2 = *(kc.add(base) as *const [u32; 2]);
        let (k0, k1) = bf16::unpack2(k2[0]);
        let (k2a, k3) = bf16::unpack2(k2[1]);
        let s = warp::sum(qq[0] * k0 + qq[1] * k1 + qq[2] * k2a + qq[3] * k3);
        let v2 = *(vc.add(base) as *const [u32; 2]);
        let (v0, v1) = bf16::unpack2(v2[0]);
        let (v2a, v3) = bf16::unpack2(v2[1]);
        let mn = m.max(s);
        let corr = math::exp(m - mn);
        let pe = math::exp(s - mn);
        sum = sum * corr + pe;
        acc[0] = acc[0] * corr + pe * v0;
        acc[1] = acc[1] * corr + pe * v1;
        acc[2] = acc[2] * corr + pe * v2a;
        acc[3] = acc[3] * corr + pe * v3;
        m = mn;
        kp += 1;
    }
    let p = part.add((((sp * rows + r) * nq + qh) * 130) as usize);
    if l == 0 {
        *p = m;
        *p.add(1) = sum;
    }
    kcommon::unroll!(i in 4 => {
        *p.add(2 + 4 * l as usize + i) = acc[i];
    });
}

/// fe_df_attn_mma: keys a tile, bytes a K / V / Q row, cp.async stages, log2 of the f16 probability scale.
const AM_SB: u32 = 16;
const AM_ROWB: u32 = HD * 2;
const AM_STAGES: u32 = 3;
const AM_PSHIFT: f32 = 14.0;
/// q heads sharing a kv head (one warp each)
const AM_GRP: u32 = NQ / NKV;

/// Byte offset of 16-byte chunk `c` (0..16) of row `r` in a [rows][128] bf16 tile, XOR-swizzled.
#[inline(always)]
fn am_sw(r: u32, c: u32) -> u32 {
    r * AM_ROWB + ((c ^ (r & 7)) << 4)
}

/// Two floats to an f16x2 word (`lo` = element 0).
#[inline(always)]
fn am_f16x2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe {
        core::arch::asm!("cvt.rn.f16x2.f32 {r}, {hi}, {lo};", r = out(reg32) r, hi = in(reg32) hi, lo = in(reg32) lo,
            options(pure, nomem, nostack));
    }
    r
}

/// fe_df_attn_part on tensor cores: block (kvh, sp, z) takes the AM_GRP q heads of draft kv head kvh (one warp
/// each) for all `rows` (<= 16) block rows, over split sp of the union of the rows' key windows ([lo of row 0,
/// pos0 + rows), cut into `nsplit` runs of 16-key tiles); a row's keys below its own window start are masked. K / V
/// tiles (ring rows, key & mask) go through shared memory once for the whole group (cp.async, AM_STAGES deep); QK^T
/// on bf16 mma, probabilities through f16 (scaled by 2^AM_PSHIFT) into P V on f16 mma (V converted in shared memory).
/// Partials in fe_df_attn_part's layout (max in natural-log units, sum, acc[128]) for fe_df_attn_comb.
/// scale_log2 = softmax scale x log2(e). grid = (nq / AM_GRP, nsplit, streams), block = 32 * AM_GRP.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_attn_mma(part: *mut f32, q: *const u16, kc: *const u16, vc: *const u16,
    rows: u32, pos0: u32, win: u32, scale_log2: f32, mask: u32, nsplit: u32, posv: *const u32, cso: *const u32,
    nq: u32, kvd: u32) {
    use kcommon::mma::*;
    let sq: Shared<u8> = kcommon::static_shared!(df_am_q, u8, AM_GRP * 16 * AM_ROWB);
    let skv: Shared<u8> = kcommon::static_shared!(df_am_kv, u8, AM_STAGES * 2 * AM_SB * AM_ROWB);
    let kvh = bid_x();
    let sp = bid_y();
    let z = bid_z();
    let w = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    let tid = tid_x();
    let nt = bdim_x();
    let (pos0, kc, vc) = if posv.is_null() { (pos0, kc, vc) } else {
        let co = *cso.add(z as usize) as usize;
        (*posv.add(z as usize), kc.add(co), vc.add(co))
    };
    let q = q.add((z * rows * nq * HD) as usize);
    let part = part.add((z * nsplit * rows * nq * 130) as usize);
    let end = pos0 + rows;
    let lo_of = |r: u32| if pos0 + r + 1 > win { pos0 + r + 1 - win } else { 0 };
    let u0 = lo_of(0);
    let ntiles = (end - u0).div_ceil(AM_SB);
    let (t0, t1) = (ntiles * sp / nsplit, ntiles * (sp + 1) / nsplit);
    // Q: the group's heads of the block rows, zero rows past `rows`
    let mut i = tid;
    while i < AM_GRP * 16 * 16 {
        let hh = i >> 8;
        let r = (i >> 4) & 15;
        let c = i & 15;
        let d = sq.byte_add(hh * 16 * AM_ROWB + am_sw(r, c)).addr();
        if r < rows {
            cp_async_16(d, q.add(((r * nq + kvh * AM_GRP + hh) * HD + c * 8) as usize) as *const u8);
        } else {
            cp_async_16_zfill(d, q as *const u8, 0);
        }
        i += nt;
    }
    cp_async_commit();
    let kvo = (kvh * HD) as usize;
    let load = |t: u32, buf: u32| {
        let k0 = u0 + t * AM_SB;
        let st = skv.byte_add(buf * 2 * AM_SB * AM_ROWB);
        let mut i = tid;
        while i < 2 * AM_SB * 16 {
            let which = i / (AM_SB * 16);
            let r = (i >> 4) & (AM_SB - 1);
            let c = i & 15;
            let d = st.byte_add(which * AM_SB * AM_ROWB + am_sw(r, c)).addr();
            let src = if which == 0 { kc } else { vc };
            let key = k0 + r;
            if key < end {
                cp_async_16(d, src.add(((key & mask) * kvd) as usize + kvo + (c * 8) as usize) as *const u8);
            } else {
                cp_async_16_zfill(d, src as *const u8, 0);
            }
            i += nt;
        }
    };
    // AM_STAGES - 1 tiles in flight ahead of the one computed (an empty group keeps the count)
    kcommon::unroll!(k in 2 => {
        if t0 + (k as u32) < t1 {
            load(t0 + k as u32, k as u32);
        }
        cp_async_commit();
    });
    let sqw = sq.byte_add(w * 16 * AM_ROWB);
    // this lane's rows g, g + 8: their window starts (rows past `rows` take no keys)
    let lo = [if g < rows { lo_of(g) } else { u32::MAX }, if g + 8 < rows { lo_of(g + 8) } else { u32::MAX }];
    let mut o = [[0.0f32; 4]; 16];
    let mut mrow = [f32::NEG_INFINITY; 2];
    let mut lrow = [0.0f32; 2];
    let mut t = t0;
    while t < t1 {
        let buf = (t - t0) % AM_STAGES;
        if t + 2 < t1 {
            load(t + 2, (t + 2 - t0) % AM_STAGES);
        }
        cp_async_commit();
        cp_async_wait::<2>();
        sync_threads();
        let sk = skv.byte_add(buf * 2 * AM_SB * AM_ROWB);
        let sv = sk.byte_add(AM_SB * AM_ROWB);
        let sv32: Shared<u32> = sv.cast();
        let mut i = tid;
        while i < AM_SB * HD / 2 {
            let x = sv32.at(i);
            sv32.set(i, am_f16x2(f32::from_bits(x << 16), f32::from_bits(x & 0xffff_0000)));
            i += nt;
        }
        sync_threads();
        let k0 = u0 + t * AM_SB;
        let mut sc = [[0.0f32; 4]; 2];
        kcommon::unroll!(ks in 8 => {
            let a = ldmatrix_x4(sqw.byte_add(am_sw(l & 15, ks as u32 * 2 + (l >> 4))).addr());
            let bb = ldmatrix_x4(sk.byte_add(am_sw((l >> 4) * 8 + (l & 7), ks as u32 * 2 + ((l >> 3) & 1))).addr());
            mma_bf16_16816(&mut sc[0], a, [bb[0], bb[1]]);
            mma_bf16_16816(&mut sc[1], a, [bb[2], bb[3]]);
        });
        let mut mx = mrow;
        kcommon::unroll!(tn in 2 => {
            let key = k0 + tn as u32 * 8 + 2 * t4;
            sc[tn][0] = if key >= lo[0] && key < end { sc[tn][0] * scale_log2 } else { f32::NEG_INFINITY };
            sc[tn][1] = if key + 1 >= lo[0] && key + 1 < end { sc[tn][1] * scale_log2 } else { f32::NEG_INFINITY };
            sc[tn][2] = if key >= lo[1] && key < end { sc[tn][2] * scale_log2 } else { f32::NEG_INFINITY };
            sc[tn][3] = if key + 1 >= lo[1] && key + 1 < end { sc[tn][3] * scale_log2 } else { f32::NEG_INFINITY };
            mx[0] = mx[0].max(sc[tn][0]).max(sc[tn][1]);
            mx[1] = mx[1].max(sc[tn][2]).max(sc[tn][3]);
        });
        kcommon::unroll!(hf in 2 => {
            mx[hf] = mx[hf].max(warp::shfl_xor(mx[hf], 1));
            mx[hf] = mx[hf].max(warp::shfl_xor(mx[hf], 2));
        });
        let base = [if mx[0] == f32::NEG_INFINITY { 0.0 } else { mx[0] }, if mx[1] == f32::NEG_INFINITY { 0.0 } else { mx[1] }];
        let cr = [math::exp2(mrow[0] - base[0]), math::exp2(mrow[1] - base[1])];
        mrow = mx;
        let mut ps = [0.0f32; 2];
        let mut pa = [0u32; 4];
        kcommon::unroll!(tn in 2 => {
            let e0 = am_f16r(math::exp2(sc[tn][0] - base[0] + AM_PSHIFT));
            let e1 = am_f16r(math::exp2(sc[tn][1] - base[0] + AM_PSHIFT));
            let e2 = am_f16r(math::exp2(sc[tn][2] - base[1] + AM_PSHIFT));
            let e3 = am_f16r(math::exp2(sc[tn][3] - base[1] + AM_PSHIFT));
            ps[0] += e0 + e1;
            ps[1] += e2 + e3;
            pa[tn * 2] = am_f16x2(e0, e1);
            pa[tn * 2 + 1] = am_f16x2(e2, e3);
        });
        lrow[0] = lrow[0] * cr[0] + ps[0];
        lrow[1] = lrow[1] * cr[1] + ps[1];
        kcommon::unroll!(dt in 16 => {
            o[dt][0] *= cr[0];
            o[dt][1] *= cr[0];
            o[dt][2] *= cr[1];
            o[dt][3] *= cr[1];
        });
        kcommon::unroll!(dp in 8 => {
            let bb = ldmatrix_x4_trans(sv.byte_add(am_sw((l & 7) + ((l >> 3) & 1) * 8, dp as u32 * 2 + (l >> 4))).addr());
            mma_f16_16816(&mut o[2 * dp], pa, [bb[0], bb[1]]);
            mma_f16_16816(&mut o[2 * dp + 1], pa, [bb[2], bb[3]]);
        });
        sync_threads();
        t += 1;
    }
    cp_async_wait_all();
    kcommon::unroll!(hf in 2 => {
        lrow[hf] += warp::shfl_xor(lrow[hf], 1);
        lrow[hf] += warp::shfl_xor(lrow[hf], 2);
    });
    let qh = kvh * AM_GRP + w;
    kcommon::unroll!(hf in 2 => {
        let r = g + 8 * hf as u32;
        if r < rows {
            let p = part.add((((sp * rows + r) * nq + qh) * 130) as usize);
            if t4 == 0 {
                // natural-log max of the scaled probabilities (-inf: no keys of this row in the split)
                *p = if mrow[hf] == f32::NEG_INFINITY { f32::NEG_INFINITY } else { (mrow[hf] - AM_PSHIFT) * core::f32::consts::LN_2 };
                *p.add(1) = lrow[hf];
            }
            kcommon::unroll!(dt in 16 => {
                let d = (dt as u32 * 8 + 2 * t4) as usize;
                *p.add(2 + d) = o[dt][2 * hf];
                *p.add(3 + d) = o[dt][2 * hf + 1];
            });
        }
    });
}

/// x rounded to f16 and back (the sums take the probabilities the P V products use).
#[inline(always)]
fn am_f16r(x: f32) -> f32 {
    let y: f32;
    unsafe {
        core::arch::asm!("{{ .reg .f16 h; cvt.rn.f16.f32 h, {x}; cvt.f32.f16 {y}, h; }}", y = out(reg32) y, x = in(reg32) x,
            options(pure, nomem, nostack));
    }
    y
}

/// Merge fe_df_attn_part's partials (fixed split order) into out [rows][4096] bf16. grid = (32, rows), block = 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_attn_comb(out: *mut u16, part: *const f32, rows: u32, nsplit: u32, nq: u32) {
    let qh = bid_x();
    let r = bid_y();
    let l = lane();
    // stream z (grid z) of a batch
    let z = bid_z();
    let part = part.add((z * nsplit * rows * nq * 130) as usize);
    let out = out.add((z * rows * nq * HD) as usize);
    let mut mx = -1.0e30f32;
    let mut sp = 0;
    while sp < nsplit {
        mx = mx.max(*part.add((((sp * rows + r) * nq + qh) * 130) as usize));
        sp += 1;
    }
    let mut tot = 0.0f32;
    let mut acc = [0.0f32; 4];
    let mut sp = 0;
    while sp < nsplit {
        let p = part.add((((sp * rows + r) * nq + qh) * 130) as usize);
        let w = math::exp(*p - mx);
        tot += *p.add(1) * w;
        kcommon::unroll!(i in 4 => {
            acc[i] += *p.add(2 + 4 * l as usize + i) * w;
        });
        sp += 1;
    }
    let inv = 1.0 / tot;
    let o = out.add((r * nq * HD + qh * HD + 4 * l) as usize) as *mut u32;
    *o = bf16::pack2(acc[0] * inv, acc[1] * inv);
    *o.add(1) = bf16::pack2(acc[2] * inv, acc[3] * inv);
}

/// Per-row top-16 of bf16 logits [rows][ld] over the first `vocab` entries, in descending order (ties: lower
/// id first): vals [rows][16] f32, ids [rows][16]. grid = rows, block = 256: each thread keeps a sorted top-16
/// of its stripe in registers, then 16 block-wide argmax passes over the 4096 survivors in shared memory.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_topk16(vals: *mut f32, ids: *mut u32, logits: *const u16, ld: u32, vocab: u32) {
    let r = bid_x();
    let t = tid_x();
    let row = logits.add((r * ld) as usize);
    let cv: Shared<f32> = kcommon::static_shared!(df_tk_cv, f32, 4096 * 4);
    let ci: Shared<u32> = kcommon::static_shared!(df_tk_ci, u32, 4096 * 4);
    let sv: Shared<f32> = kcommon::static_shared!(df_tk_v, f32, 32);
    let si: Shared<u32> = kcommon::static_shared!(df_tk_i, u32, 32);
    let mut lv = [-f32::INFINITY; 16];
    let mut li = [u32::MAX; 16];
    // insert (v, i) keeping lv descending; a thread's ids ascend, so equal values stay in id order
    let mut ins = |v: f32, i: u32| {
        if v > lv[15] {
            let (mut cvv, mut cii) = (v, i);
            kcommon::unroll!(j in 16 => {
                if cvv > lv[j] {
                    let (tv, ti) = (lv[j], li[j]);
                    lv[j] = cvv;
                    li[j] = cii;
                    cvv = tv;
                    cii = ti;
                }
            });
        }
    };
    // 8 consecutive logits a 16-byte load, two loads in flight (the scalar loop was bound by load latency)
    let n8 = if ld % 8 == 0 { vocab / 8 } else { 0 };
    let mut c = t;
    while c + 256 < n8 {
        let a = bf16::unpack8(ldg_nc_v4(row.add((c * 8) as usize) as *const u32));
        let b = bf16::unpack8(ldg_nc_v4(row.add(((c + 256) * 8) as usize) as *const u32));
        kcommon::unroll!(j in 8 => {
            ins(a[j], c * 8 + j as u32);
        });
        kcommon::unroll!(j in 8 => {
            ins(b[j], (c + 256) * 8 + j as u32);
        });
        c += 512;
    }
    if c < n8 {
        let a = bf16::unpack8(ldg_nc_v4(row.add((c * 8) as usize) as *const u32));
        kcommon::unroll!(j in 8 => {
            ins(a[j], c * 8 + j as u32);
        });
    }
    let mut i = n8 * 8 + t;
    while i < vocab {
        ins(bf16::to_f32(*row.add(i as usize)), i);
        i += 256;
    }
    kcommon::unroll!(j in 16 => {
        cv.set(t * 16 + j as u32, lv[j]);
        ci.set(t * 16 + j as u32, li[j]);
    });
    sync_threads();
    let (mut pv, mut pi) = (f32::INFINITY, 0u32);
    let mut n = 0;
    while n < 16 {
        let mut bv = -f32::INFINITY;
        let mut bi = u32::MAX;
        let mut e = t;
        while e < 4096 {
            let (v, id) = (cv.at(e), ci.at(e));
            let after = v < pv || (v == pv && id > pi);
            if after && (v > bv || (v == bv && id < bi)) {
                bv = v;
                bi = id;
            }
            e += 256;
        }
        kcommon::unroll!(d in [16, 8, 4, 2, 1] => {
            let ov = warp::shfl_xor(bv, d as u32);
            let oi = warp::shfl_xor_u32(bi, d as u32);
            if ov > bv || (ov == bv && oi < bi) {
                bv = ov;
                bi = oi;
            }
        });
        if lane() == 0 {
            sv.set(warp_id(), bv);
            si.set(warp_id(), bi);
        }
        sync_threads();
        let (mut v, mut id) = (sv.at(0), si.at(0));
        let mut w = 1;
        while w < 8 {
            let (ov, oi) = (sv.at(w), si.at(w));
            if ov > v || (ov == v && oi < id) {
                v = ov;
                id = oi;
            }
            w += 1;
        }
        if t == 0 {
            *vals.add((r * 16 + n) as usize) = v;
            *ids.add((r * 16 + n) as usize) = id;
        }
        pv = v;
        pi = id;
        sync_threads();
        n += 1;
    }
}

/// TP2 draft head: a rank's per-row top-16 (vals / ids, ids + id_off) packed as [rows][32] u32 (16 f32 bits, 16 ids)
/// for the fabric all_gather. grid = rows, block = 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_topk_pack(out: *mut u32, vals: *const f32, ids: *const u32, id_off: u32) {
    let (r, l) = (bid_x() as usize, lane() as usize);
    if l < 16 {
        *out.add(r * 32 + l) = (*vals.add(r * 16 + l)).to_bits();
        *out.add(r * 32 + 16 + l) = *ids.add(r * 16 + l) + id_off;
    }
}

/// Merge the two ranks' packed top-16 lists ([2][rows][32], from fe_df_topk_pack + all_gather) into vals / ids
/// [rows][16]: descending, ties to the lower id (as fe_df_topk16). grid = rows, block = 32 (lane 0 merges).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_topk_merge(vals: *mut f32, ids: *mut u32, packed: *const u32, rows: u32) {
    let r = bid_x() as usize;
    if lane() != 0 {
        return;
    }
    let a = packed.add(r * 32);
    let b = packed.add((rows as usize + r) * 32);
    let (mut i, mut j) = (0usize, 0usize);
    let mut n = 0usize;
    while n < 16 {
        let (va, ia) = (f32::from_bits(*a.add(i)), *a.add(16 + i));
        let (vb, ib) = (f32::from_bits(*b.add(j)), *b.add(16 + j));
        let take_a = j >= 16 || (i < 16 && (va > vb || (va == vb && ia < ib)));
        if take_a {
            *vals.add(r * 16 + n) = va;
            *ids.add(r * 16 + n) = ia;
            i += 1;
        } else {
            *vals.add(r * 16 + n) = vb;
            *ids.add(r * 16 + n) = ib;
            j += 1;
        }
        n += 1;
    }
}

/// Candidate-selector walk (dflash CandidateSelector): pred = anchor; for each row i:
/// score[c] = vals[i][c] + sum_r A[pred][r] * gate[i][r] * B[ids[i][c]][r]; pred = ids[i][argmax c] -> out[i].
/// gate [rows][256] bf16 (hidden_projection of the row's final-normed state), A / B codebooks [vocab][256]
/// bf16. grid = 1, block = 512 (warp c scores candidate c; lane covers 8 ranks).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_select(out: *mut u32, vals: *const f32, ids: *const u32, gate: *const u16,
    a_cb: *const u16, b_cb: *const u16, anchor: u32, rows: u32) {
    let c = warp_id();
    let l = lane();
    let sc: Shared<f32> = kcommon::static_shared!(df_sel_s, f32, 64);
    let sp: Shared<u32> = kcommon::static_shared!(df_sel_p, u32, 16);
    if tid_x() == 0 {
        sp.set(0, anchor);
    }
    sync_threads();
    let mut i = 0;
    while i < rows {
        let pred = sp.at(0);
        let cand = *ids.add((i * 16 + c) as usize);
        let a = bf16::unpack8(ldg_nc_v4(a_cb.add((pred as usize) * 256 + 8 * l as usize) as *const u32));
        let b = bf16::unpack8(ldg_nc_v4(b_cb.add((cand as usize) * 256 + 8 * l as usize) as *const u32));
        let gg = bf16::unpack8(ldg_nc_v4(gate.add((i * 256 + 8 * l) as usize) as *const u32));
        let mut s = 0.0f32;
        kcommon::unroll!(x in 8 => {
            s = math::fma(a[x] * gg[x], b[x], s);
        });
        s = warp::sum(s);
        if l == 0 {
            sc.set(c, *vals.add((i * 16 + c) as usize) + s);
        }
        sync_threads();
        if tid_x() == 0 {
            let mut best = 0;
            let mut bv = sc.at(0);
            let mut k = 1;
            while k < 16 {
                if sc.at(k) > bv {
                    bv = sc.at(k);
                    best = k;
                }
                k += 1;
            }
            let id = *ids.add((i * 16 + best) as usize);
            *out.add(i as usize) = id;
            sp.set(0, id);
        }
        sync_threads();
        i += 1;
    }
}

/// Qwen3 RMSNorm (plain weight, unlike the target's Qwen3.5 1 + w): y = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))).
/// x / y rows of 5120 (stride 5120). grid = rows, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_rmsnorm(y: *mut u16, x: *const u16, w: *const u16, eps: f32) {
    let red: Shared<f32> = kcommon::static_shared!(df_rms_red, f32, 128);
    let r = bid_x() as usize;
    let xr = x.add(r * H as usize) as *const u32;
    let yr = y.add(r * H as usize) as *mut u32;
    let wr = w as *const u32;
    let nv = H / 8;
    let mut ss = 0.0f32;
    let mut i = tid_x();
    while i < nv {
        let v = bf16::unpack8(ldg_nc_v4(xr.add(i as usize * 4)));
        kcommon::unroll!(j in 8 => {
            ss += v[j] * v[j];
        });
        i += 256;
    }
    let tot = warp::block_sum(ss, red);
    let inv = math::rsqrt(tot / H as f32 + eps);
    let mut i = tid_x();
    while i < nv {
        let v = bf16::unpack8(ldg_nc_v4(xr.add(i as usize * 4)));
        let wv = bf16::unpack8(ldg_nc_v4(wr.add(i as usize * 4)));
        let mut o = [0u32; 4];
        kcommon::unroll!(j in 4 => {
            o[j] = bf16::pack2(wv[2 * j] * bf16::round(v[2 * j] * inv), wv[2 * j + 1] * bf16::round(v[2 * j + 1] * inv));
        });
        stg_v4(yr.add(i as usize * 4), o);
        i += 256;
    }
}

/// W4A16 GEMV for draft rows: y[r][col] = sum_k x[r][k] * (q[col][k] * s[col][k/64] + b[col][k/64]) (+ res[r][col]),
/// r < m <= 8. q: 8 nibbles a u32, k ascending, low nibble first ([n][k/8]); s, b: bf16 [n][k/64] (affine,
/// group 64). x rows stride ldx, y / res stride ldy. k % 64 == 0. grid = ceil(n/8), block = 256 (warp per column).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_gemv4(y: *mut u16, x: *const u16, wq: *const u32, sc: *const u16, bi: *const u16,
    res: *const u16, m: u32, n: u32, k: u32, ldx: u32, ldy: u32) {
    let col = bid_x() * 8 + warp_id();
    if col >= n {
        return;
    }
    let l = lane();
    let wr = wq.add(col as usize * (k as usize / 8));
    let gs = col as usize * (k as usize / 64);
    let mut acc = [0.0f32; 8];
    let mut c = l; // chunk of 32 weights (16 bytes)
    while c < k / 32 {
        let w4 = ldg_stream_v4(wr.add(c as usize * 4));
        let g = gs + (c / 2) as usize;
        let s = bf16::to_f32(*sc.add(g));
        let b = bf16::to_f32(*bi.add(g));
        let mut wv = [0.0f32; 32];
        kcommon::unroll!(u in 4 => {
            kcommon::unroll!(j in 8 => {
                wv[u * 8 + j] = math::fma(((w4[u] >> (4 * j)) & 15) as f32, s, b);
            });
        });
        kcommon::unroll!(r in 8 => {
            if (r as u32) < m {
                let xp = x.add(r * ldx as usize + c as usize * 32) as *const u32;
                let mut a = acc[r];
                kcommon::unroll!(h in 4 => {
                    let xv = bf16::unpack8(ldg_nc_v4(xp.add(h * 4)));
                    kcommon::unroll!(j in 8 => {
                        a = math::fma(wv[h * 8 + j], xv[j], a);
                    });
                });
                acc[r] = a;
            }
        });
        c += 32;
    }
    kcommon::unroll!(r in 8 => {
        if (r as u32) < m {
            let v = warp::sum(acc[r]);
            if l == 0 {
                let off = r * ldy as usize + col as usize;
                let mut o = v;
                if !res.is_null() {
                    o = bf16::round(o) + bf16::to_f32(*res.add(off));
                }
                *y.add(off) = bf16::from_f32(o);
            }
        }
    });
}

/// One affine-4 nibble pair (bits sh..sh+4 and sh+4..sh+8 of w) dequantized to a bf16x2 A-fragment word.
#[inline(always)]
fn dq2(w: u32, sh: u32, s: f32, b: f32) -> u32 {
    let q0 = f32::from_bits(0x4B00_0000 | ((w >> sh) & 15)) - 8_388_608.0;
    let q1 = f32::from_bits(0x4B00_0000 | ((w >> (sh + 4)) & 15)) - 8_388_608.0;
    bf16::pack2(math::fma(q0, s, b), math::fma(q1, s, b))
}

/// W4A16 GEMV for draft rows on tensor cores (mma m16n8k16 bf16: A = 16 weight columns, B = the <= 8 x rows):
/// y[r][col] = sum_k x[r][k] * (q * s + b) (+ res[r][col]), r < m <= 8, same affine-4 g64 values as
/// fe_df_gemv4 but repacked at load (fe-dflash `pack4t`) so every load is a coalesced 16 B per lane:
/// wp [n/16][k/128][2][32 lanes][4] u32 — lane (g, t) word s (step s of 8 in a 128-k chunk) holds the nibbles
/// of (col g, k 32t+4s+0..1), (col g+8, same k), (col g, 32t+4s+2..3), (col g+8, same) = a0..a3, so its x is the
/// contiguous x[g][32t..32t+32] of the chunk; sp [n/16][k/128][32 lanes] u32 = (s, b) bf16 of lane (g, t)'s
/// (col g + 8 (t & 1), group 2 chunk + t / 2), shuffled to the lanes that use it. Each block: 32 columns
/// (2 tiles); the 8 warps split K (chunks w, w + 8, ...) and are summed in a fixed order through shared memory.
/// yf32 != 0: y is f32 [m][ldy] and gets the raw sums (no residual). n % 32 == 0, k % 128 == 0. grid = n / 32,
/// block = 256. fe_df_gemv4t4: 64 columns (4 tiles) a block, half the
/// x traffic per weight byte; n % 64 == 0, grid = n / 64. fe_df_gemv4t16: m <= 16 (two 8-row B fragments share
/// each dequantized A fragment), grid = n / 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_gemv4t(y: *mut u16, x: *const u16, wp: *const u32, sp: *const u32,
    res: *const u16, m: u32, n: u32, k: u32, ldx: u32, ldy: u32, yf32: u32) {
    let _ = n;
    let red = static_shared!(df_gemv4t_red, f32, 8 * 256 * 4);
    gemv4t_body::<2, 1>(red, bid_x() * 2, y, x, wp, sp, res, m, k, ldx, ldy, yf32);
}
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_gemv4t4(y: *mut u16, x: *const u16, wp: *const u32, sp: *const u32,
    res: *const u16, m: u32, n: u32, k: u32, ldx: u32, ldy: u32, yf32: u32) {
    let _ = n;
    let red = static_shared!(df_gemv4t4_red, f32, 8 * 512 * 4);
    gemv4t_body::<4, 1>(red, bid_x() * 4, y, x, wp, sp, res, m, k, ldx, ldy, yf32);
}
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_gemv4t16(y: *mut u16, x: *const u16, wp: *const u32, sp: *const u32,
    res: *const u16, m: u32, n: u32, k: u32, ldx: u32, ldy: u32, yf32: u32) {
    let _ = n;
    let red = static_shared!(df_gemv4t16_red, f32, 8 * 512 * 4);
    gemv4t_body::<2, 2>(red, bid_x() * 2, y, x, wp, sp, res, m, k, ldx, ldy, yf32);
}
/// fe_df_gemv4t16 over more than 16 rows: block b takes rows 16 (b % chunks).. of column block b / chunks, so the
/// chunks of a column block run side by side and share its weights through L2. grid = n / 32 * chunks.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_df_gemv4t16c(y: *mut u16, x: *const u16, wp: *const u32, sp: *const u32,
    res: *const u16, m: u32, n: u32, k: u32, ldx: u32, ldy: u32, chunks: u32, yf32: u32) {
    let _ = n;
    let red = static_shared!(df_gemv4t16c_red, f32, 8 * 512 * 4);
    let (cb, ch) = (bid_x() / chunks, bid_x() % chunks);
    let r0 = 16 * ch;
    if r0 >= m {
        return;
    }
    let res = if res.is_null() { res } else { res.add((r0 * ldy) as usize) };
    let y = if yf32 != 0 { (y as *mut f32).add((r0 * ldy) as usize) as *mut u16 } else { y.add((r0 * ldy) as usize) };
    gemv4t_body::<2, 2>(red, cb * 2, y, x.add((r0 * ldx) as usize), wp, sp, res, (m - r0).min(16), k, ldx, ldy, yf32);
}

/// T 16-column tiles a block, R groups of 8 rows.
#[inline(always)]
unsafe fn gemv4t_body<const T: usize, const R: usize>(red: Shared<f32>, tile0: u32, y: *mut u16, x: *const u16,
    wp: *const u32, sp: *const u32, res: *const u16, m: u32, k: u32, ldx: u32, ldy: u32, yf32: u32) {
    let l = lane();
    let (g, t) = (l / 4, l % 4);
    let w = warp_id();
    let nch = k / 128;
    let mut acc = [[[0.0f32; 4]; T]; R];
    let mut c = w;
    while c < nch {
        let mut xs = [[0u32; 16]; R];
        kcommon::unroll!(gr in 2 => {
            if gr < R && g + 8 * (gr as u32) < m {
                let xr = x.add(((g + 8 * gr as u32) * ldx + 32 * t) as usize) as *const u32;
                kcommon::unroll!(h in 4 => {
                    let v = ldg_nc_v4(xr.add(c as usize * 64 + h * 4));
                    kcommon::unroll!(j in 4 => {
                        xs[gr][h * 4 + j] = v[j];
                    });
                });
            }
        });
        let mut wv = [[0u32; 8]; T];
        let mut sbv = [0u32; T];
        kcommon::unroll!(tt in 4 => {
            if tt < T {
                let blk = (tile0 + tt as u32) * nch + c;
                let base = wp.add(blk as usize * 256 + l as usize * 4);
                let w0 = ldg_stream_v4(base);
                let w1 = ldg_stream_v4(base.add(128));
                kcommon::unroll!(j in 4 => {
                    wv[tt][j] = w0[j];
                    wv[tt][4 + j] = w1[j];
                });
                sbv[tt] = *sp.add(blk as usize * 32 + l as usize);
            }
        });
        kcommon::unroll!(tt in 4 => {
            if tt < T {
                let sbg = warp::shfl_idx_u32(sbv[tt], g * 4 + (t & 2));
                let sb8 = warp::shfl_idx_u32(sbv[tt], g * 4 + (t & 2) + 1);
                let (sg, bg) = (bf16::lo(sbg), bf16::hi(sbg));
                let (s8, b8) = (bf16::lo(sb8), bf16::hi(sb8));
                kcommon::unroll!(s in 8 => {
                    let wd = wv[tt][s];
                    let a = [dq2(wd, 0, sg, bg), dq2(wd, 8, s8, b8), dq2(wd, 16, sg, bg), dq2(wd, 24, s8, b8)];
                    kcommon::unroll!(gr in 2 => {
                        if gr < R {
                            mma::mma_bf16_16816(&mut acc[gr][tt], a, [xs[gr][2 * s], xs[gr][2 * s + 1]]);
                        }
                    });
                });
            }
        });
        c += 8;
    }
    let per = (T * R) as u32 * 128;
    kcommon::unroll!(gr in 2 => {
        if gr < R {
            kcommon::unroll!(tt in 4 => {
                if tt < T {
                    kcommon::unroll!(e in 4 => {
                        red.set(w * per + ((gr * T + tt) as u32) * 128 + l * 4 + e as u32, acc[gr][tt][e]);
                    });
                }
            });
        }
    });
    sync_threads();
    let mut i = tid_x();
    while i < per {
        let mut v = 0.0f32;
        kcommon::unroll!(ww in 8 => {
            v += red.at(ww as u32 * per + i);
        });
        // i -> (row group, tile, lane', element): c0/c1 = (col g', rows 2t', 2t'+1), c2/c3 = (col g'+8, same rows)
        let (gr, rem) = (i / (T as u32 * 128), i % (T as u32 * 128));
        let (tt, li, e) = (rem / 128, (rem % 128) / 4, rem % 4);
        let col = (tile0 + tt) * 16 + li / 4 + 8 * (e >> 1);
        let row = 8 * gr + 2 * (li % 4) + (e & 1);
        if row < m {
            let off = (row * ldy + col) as usize;
            if yf32 != 0 {
                // f32 partial (a TP rank's row-parallel slice, summed by the fabric)
                *(y as *mut f32).add(off) = v;
            } else {
                let mut o = v;
                if !res.is_null() {
                    o = bf16::round(o) + bf16::to_f32(*res.add(off));
                }
                *y.add(off) = bf16::from_f32(o);
            }
        }
        i += 256;
    }
}
