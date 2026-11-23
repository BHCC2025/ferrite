//! Decode router front: post-attention RMS norm + router logits in one multi-block kernel (16 blocks a row instead of 4, and one
//! launch less), with the arithmetic of `ink_rms` (256-thread strided sums, `warp::block_sum`) and
//! `ink_router_logits` (lane-strided fma over bf16 pairs, `warp::sum`) so the bits are unchanged.
use super::*;
use kcommon::smem::Shared;

/// xm[t] = bf16(w * (src[t] * rsqrt(mean(src[t]^2) + 1e-6))) (written by the y == 0 block), logits[t][e] = xm[t] . wg[e] (f32).
/// src f32 [T][2304], w / wg bf16, grid = (T, 16), block = 256 (each warp one expert: e = bid_y * 8 + warp).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_router_fused(xm: *mut u16, logits: *mut f32, src: *const f32, w: *const u16, wg: *const u16) {
    pdl_wait();
    pdl_trigger();
    const D: u32 = 2304;
    let red = static_shared!(rf_red, f32, 128);
    let xs: Shared<f32> = static_shared!(rf_x, f32, 2304 * 4);
    let t = bid_x() as usize;
    let mut ss = 0.0f32;
    let mut i = tid_x();
    while i < D {
        let v = *src.add(t * D as usize + i as usize);
        ss += v * v;
        i += 256;
    }
    let tot = warp::block_sum(ss, red);
    let inv = math::rsqrt(tot / D as f32 + 1e-6);
    i = tid_x();
    while i < D {
        let v = *src.add(t * D as usize + i as usize);
        let o = bf16::from_f32(bf16::to_f32(*w.add(i as usize)) * (v * inv));
        xs.set(i, bf16::to_f32(o));
        if bid_y() == 0 {
            *xm.add(t * D as usize + i as usize) = o;
        }
        i += 256;
    }
    sync_threads();
    let l = lane();
    let e = bid_y() * 8 + warp_id();
    let row = wg.add(e as usize * D as usize) as *const u32;
    let mut s = 0.0f32;
    let mut k = l;
    while k < D / 2 {
        let (a, b) = bf16::unpack2(*row.add(k as usize));
        s = math::fma(a, xs.at(2 * k), s);
        s = math::fma(b, xs.at(2 * k + 1), s);
        k += 32;
    }
    let s = warp::sum(s);
    if l == 0 {
        *logits.add(t * 128 + e as usize) = s;
    }
}

/// Attention-side tail of a decode layer application in one launch: the static short convolution (kernel 4, residual, / sqrt 5,
/// 3-row f32 state, rows in order, as `ink_sconv`), then `ink_resid` (resid = conv [+ f32(hs)] when `mode` is 1) and,
/// when `copy_dst` is not null, the block-state copy `copy_dst[t * ldd + c] = hs[t * 2304 + c]`.
/// One thread a channel walks the rows. grid = ceil(2304 / 256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_sconv_resid(resid: *mut f32, x: *const u16, w: *const u16, state: *mut f32, hs: *const u16,
    mode: u32, copy_dst: *mut u16, ldd: u32, nt: u32, spec: u32) {
    pdl_wait();
    pdl_trigger();
    let ch = (bid_x() * 256 + tid_x()) as usize;
    if ch >= 2304 {
        return;
    }
    let w0 = bf16::to_f32(*w.add(ch * 4));
    let w1 = bf16::to_f32(*w.add(ch * 4 + 1));
    let w2 = bf16::to_f32(*w.add(ch * 4 + 2));
    let w3 = bf16::to_f32(*w.add(ch * 4 + 3));
    let mut r0 = *state.add(ch);
    let mut r1 = *state.add(2304 + ch);
    let mut r2 = *state.add(2 * 2304 + ch);
    let mut t = 0usize;
    while t < nt as usize {
        let xv = bf16::to_f32(*x.add(t * 2304 + ch));
        if spec != 0 {
            *state.add((3 + t) * 2304 + ch) = xv;
        }
        let acc = math::fma(w3, xv, math::fma(w2, r2, math::fma(w1, r1, w0 * r0)));
        let a = bf16::round((acc + xv) / 2.236068);
        let h = *hs.add(t * 2304 + ch);
        *resid.add(t * 2304 + ch) = if mode == 1 { a + bf16::to_f32(h) } else { a };
        if !copy_dst.is_null() {
            *copy_dst.add(t * ldd as usize + ch) = h;
        }
        r0 = r1;
        r1 = r2;
        r2 = xv;
        t += 1;
    }
    if spec == 0 {
        *state.add(ch) = r0;
        *state.add(2304 + ch) = r1;
        *state.add(2 * 2304 + ch) = r2;
    }
}
