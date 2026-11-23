//! Q / rel-state preparation, short convolutions, K / V head norms.
use crate::*;
use kcommon::*;

/// Scaled L2 clip (+ optional log scaling) of heads: one warp a (token, head); dh <= 128.
/// y = bf16(v * min(thr / (|v| + 1e-6), 1)), v = x * scale; with `log`: y = bf16(bf16(y) * tau(pos)).
/// grid = ceil(tokens * heads / 8), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_headclip(out: *mut u16, x: *const u16, ld: u32, scale: *const u16, heads: u32,
    dh: u32, thr: f32, log: u32, pos: *const u32, ntok: u32, n_floor: f32, alpha: f32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let pos0 = *pos;
    let w = bid_x() * 8 + warp_id();
    if w >= ntok * heads {
        return;
    }
    let t = w / heads;
    let h = w % heads;
    let l = lane();
    let mut v = [0.0f32; 4];
    let mut ss = 0.0f32;
    let mut i = 0;
    while i < 4 {
        let e = l + 32 * i as u32;
        if e < dh {
            v[i] = bf(*x.add(t as usize * ld as usize + (h * dh + e) as usize)) * bf(*scale.add((h * dh + e) as usize));
            ss += v[i] * v[i];
        }
        i += 1;
    }
    let norm = math::sqrt(warp::sum(ss));
    let mut f = thr / (norm + 1e-6);
    if f > 1.0 {
        f = 1.0;
    }
    let mut tau = 1.0f32;
    if log != 0 {
        let mut a = (pos0 + t + 1) as f32 / n_floor;
        if a < 1.0 {
            a = 1.0;
        }
        tau = 1.0 + alpha * math::ln(a);
    }
    i = 0;
    while i < 4 {
        let e = l + 32 * i as u32;
        if e < dh {
            let mut y = v[i] * f;
            if log != 0 {
                y = rb(y) * tau;
            }
            *out.add((t * heads * dh + h * dh + e) as usize) = tb(y);
        }
        i += 1;
    }
}

/// Static short convolution (kernel 4) + residual, /sqrt(5), with a 3-row f32 state ([3][C], oldest first). One thread
/// a channel walks the tokens. out/x bf16 [T][C]. grid = ceil(C/256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_sconv(out: *mut u16, ldo: u32, x: *const u16, ldx: u32, w: *const u16,
    state: *mut f32, nt: u32, c: u32, spec: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let ch = bid_x() * 256 + tid_x();
    if ch >= c {
        return;
    }
    let w0 = bf(*w.add(ch as usize * 4));
    let w1 = bf(*w.add(ch as usize * 4 + 1));
    let w2 = bf(*w.add(ch as usize * 4 + 2));
    let w3 = bf(*w.add(ch as usize * 4 + 3));
    let mut r0 = *state.add(ch as usize);
    let mut r1 = *state.add((c + ch) as usize);
    let mut r2 = *state.add((2 * c + ch) as usize);
    const S5: f32 = 2.236068;
    let mut t = 0;
    while t < nt {
        // 32-bit element index, widened once (as for `state` below): with `t as usize * ld` ptxas 13.0 (V13.0.88, the
        // DGX Spark toolkit) folds the unrolled `base + ld * 2` into a uniform ULEA without the ULEA.HI.X and drops the
        // high half of the address (CUDA error 700 on the first prompt); ptxas 13.3 emits it. t * ld < 2^32 here.
        let xv = bf(*x.add((t * ldx + ch) as usize));
        if spec != 0 {
            *state.add(((3 + t) * c + ch) as usize) = xv;
        }
        let acc = math::fma(w3, xv, math::fma(w2, r2, math::fma(w1, r1, w0 * r0)));
        *out.add((t * ldo + ch) as usize) = tb((acc + xv) / S5);
        r0 = r1;
        r1 = r2;
        r2 = xv;
        t += 1;
    }
    if spec == 0 {
        *state.add(ch as usize) = r0;
        *state.add((c + ch) as usize) = r1;
        *state.add((2 * c + ch) as usize) = r2;
    }
}

/// Dynamic short convolution (full-attention K / V): taps = static[lag][c] + dyn[t][head(c) * 4 + lag], tap 0 weights the
/// current token; (sum + x) / 3. `dynw` bf16 [T][32] (ld = 32); head(c) = c / 128. Same state as ink_sconv.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_sconv_dyn(out: *mut u16, ldo: u32, x: *const u16, ldx: u32,
    stat: *const u16, dynw: *const u16, state: *mut f32, nt: u32, c: u32, spec: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let ch = bid_x() * 256 + tid_x();
    if ch >= c {
        return;
    }
    let head = ch / 128;
    let s0 = bf(*stat.add(ch as usize));
    let s1 = bf(*stat.add((c + ch) as usize));
    let s2 = bf(*stat.add((2 * c + ch) as usize));
    let s3 = bf(*stat.add((3 * c + ch) as usize));
    let mut r0 = *state.add(ch as usize);
    let mut r1 = *state.add((c + ch) as usize);
    let mut r2 = *state.add((2 * c + ch) as usize);
    let mut t = 0;
    while t < nt {
        let xv = bf(*x.add(t as usize * ldx as usize + ch as usize));
        if spec != 0 {
            *state.add(((3 + t) * c + ch) as usize) = xv;
        }
        let dp = dynw.add(t as usize * 32 + head as usize * 4);
        // lag 0 = xv, lag 1 = r2, lag 2 = r1, lag 3 = r0
        let acc = math::fma(r0, s3 + bf(*dp.add(3)), math::fma(r1, s2 + bf(*dp.add(2)),
            math::fma(r2, s1 + bf(*dp.add(1)), xv * (s0 + bf(*dp)))));
        *out.add(t as usize * ldo as usize + ch as usize) = tb((acc + xv) / 3.0);
        r0 = r1;
        r1 = r2;
        r2 = xv;
        t += 1;
    }
    if spec == 0 {
        *state.add(ch as usize) = r0;
        *state.add((c + ch) as usize) = r1;
        *state.add((2 * c + ch) as usize) = r2;
    }
}

/// K / V head normalisation into the cache: mode 0 = L2 clip (thr), mode 1 = RMS without weight (eps 1e-6).
/// x bf16 [T][1024]; out row (pos0 + t) of the cache (stride 1024). One warp a (token, kv head); grid = ceil(T * 8 / 8).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_headnorm(cache: *mut u16, x: *const u16, mode: u32, thr: f32, pos: *const u32, ntok: u32,
    ring: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let pos0 = *pos;
    let w = bid_x() * 8 + warp_id();
    if w >= ntok * 8 {
        return;
    }
    let t = w / 8;
    let h = w % 8;
    let l = lane();
    let mut v = [0.0f32; 4];
    let mut ss = 0.0f32;
    let mut i = 0;
    while i < 4 {
        v[i] = bf(*x.add((t * 1024 + h * 128 + l + 32 * i as u32) as usize));
        ss += v[i] * v[i];
        i += 1;
    }
    let tot = warp::sum(ss);
    let f = if mode == 0 {
        let mut f = thr / (math::sqrt(tot) + 1e-6);
        if f > 1.0 {
            f = 1.0;
        }
        f
    } else {
        math::rsqrt(tot / 128.0 + 1e-6)
    };
    i = 0;
    while i < 4 {
        *cache.add((((pos0 + t) % ring) * 1024 + h * 128 + l + 32 * i as u32) as usize) = tb(v[i] * f);
        i += 1;
    }
}

/// Keep the first `keep` rows of a speculative chunk: rows [keep, keep + 3) of the state (3 previous inputs, then the chunk's inputs)
/// become the state rows 0..3. grid = ceil(C / 256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_conv_commit(state: *mut f32, c: u32, keep: u32) {
    let ch = bid_x() * 256 + tid_x();
    if ch >= c {
        return;
    }
    let a = *state.add((keep * c + ch) as usize);
    let b = *state.add(((keep + 1) * c + ch) as usize);
    let d = *state.add(((keep + 2) * c + ch) as usize);
    *state.add(ch as usize) = a;
    *state.add((c + ch) as usize) = b;
    *state.add((2 * c + ch) as usize) = d;
}

/// 6-bit K/V record of one kv head (128 dims) = 104 bytes: 4 groups of 32 dims, each 24 bytes of 6-bit codes (little-endian bit stream,
/// code k at bit 6k) then 4 bf16 group scales at byte 96. Groups are rotated by H32/sqrt(32), scaled by their absmax (bf16, rounded up) and stored on
/// the midpoint grid value = (2 code + 1 - 64) / 64 x scale. Quantises `ntok` staged bf16 rows (stg [t][1024]) into rows pos0 + t of `packed`
/// ([pos][8][104]). One warp a (token, kv head); grid = ceil(ntok * 8 / 8), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_kvq_pack(packed: *mut u8, stg: *const u16, pos: *const u32, ntok: u32) {
    let w = bid_x() * 8 + warp_id();
    if w >= ntok * 8 {
        return;
    }
    let t = w / 8;
    let h = w % 8;
    let l = lane();
    let pp = *pos + t;
    let rec = packed.add(((((pp >> 6) * 8 + h) * 64 + (pp & 63)) * 104) as usize);
    const R: f32 = 0.176_776_7;
    let mut i = 0;
    while i < 4 {
        let mut x = bf(*stg.add((t * 1024 + h * 128 + 32 * i as u32 + l) as usize));
        let mut m = 1;
        while m < 32 {
            let p = warp::shfl_xor(x, m);
            x = if l & m == 0 { x + p } else { p - x };
            m <<= 1;
        }
        x *= R;
        let am = warp::max(if x < 0.0 { -x } else { x });
        let mut sb = tb(am);
        if bf(sb) < am {
            sb += 1;
        }
        let sc = bf(sb);
        let mut c = 0i32;
        if sc > 0.0 {
            c = ((x / sc + 1.0) * 32.0) as i32;
            if c < 0 { c = 0; }
            if c > 63 { c = 63; }
        }
        // lane j < 8 packs the codes of lanes 4j .. 4j + 3 into 3 bytes
        let c0 = warp::shfl_idx_u32(c as u32, (l & 7) * 4);
        let c1 = warp::shfl_idx_u32(c as u32, (l & 7) * 4 + 1);
        let c2 = warp::shfl_idx_u32(c as u32, (l & 7) * 4 + 2);
        let c3 = warp::shfl_idx_u32(c as u32, (l & 7) * 4 + 3);
        if l < 8 {
            let v = c0 | (c1 << 6) | (c2 << 12) | (c3 << 18);
            let o = rec.add(24 * i + 3 * l as usize);
            *o = v as u8;
            *o.add(1) = (v >> 8) as u8;
            *o.add(2) = (v >> 16) as u8;
        }
        if l == 0 {
            *(rec.add(96 + 2 * i) as *mut u16) = sb;
        }
        i += 1;
    }
}
