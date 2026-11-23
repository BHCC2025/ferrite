//! Loop message gates and the MoE router.
use crate::*;
use kcommon::smem::Shared;
use kcommon::*;

/// Message gate, step 1: z = bf16(Wd . bf16([norm(h), norm(m)])) -> f32 [T][64]. grid = (tokens, 16), block = 128 (4 warps,
/// one output each). Wd bf16 [64][4608]. h / m f32 with token strides ldh / ldm.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gate_z(z: *mut f32, h: *const f32, ldh: u32, m: *const f32, ldm: u32, wd: *const u16) {
    crate::pdl_wait();
    crate::pdl_trigger();
    const D: u32 = 2304;
    let red: Shared<f32> = static_shared!(gz_red, f32, 128);
    let t = bid_x() as usize;
    let hp = h.add(t * ldh as usize);
    let mp = m.add(t * ldm as usize);
    let mut sh = 0.0f32;
    let mut sm = 0.0f32;
    let mut d = tid_x();
    while d < D {
        let a = *hp.add(d as usize);
        let b = *mp.add(d as usize);
        sh += a * a;
        sm += b * b;
        d += 128;
    }
    let ih = math::rsqrt(warp::block_sum(sh, red) / D as f32 + 1e-6);
    let im = math::rsqrt(warp::block_sum(sm, red) / D as f32 + 1e-6);
    let j = bid_y() * 4 + warp_id();
    let row = wd.add(j as usize * 4608);
    let l = lane();
    let mut s = 0.0f32;
    let mut k = l;
    while k < D {
        s = math::fma(bf(*row.add(k as usize)), rb(*hp.add(k as usize) * ih), s);
        s = math::fma(bf(*row.add((D + k) as usize)), rb(*mp.add(k as usize) * im), s);
        k += 32;
    }
    let s = warp::sum(s);
    if l == 0 {
        *z.add(t * 64 + j as usize) = rb(s);
    }
}

/// Message gate, step 2: out = (1 - tanh(Wf z)) * h + tanh(Wg z) * m. grid = (tokens, 9), block = 256 (one channel each).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gate_out(out: *mut f32, ldo: u32, h: *const f32, ldh: u32, m: *const f32, ldm: u32,
    z: *const f32, wf: *const u16, wg: *const u16) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let t = bid_x() as usize;
    let d = (bid_y() * 256 + tid_x()) as usize;
    let zs = z.add(t * 64);
    let mut sf = 0.0f32;
    let mut sg = 0.0f32;
    let mut j = 0;
    while j < 64 {
        let zj = *zs.add(j);
        sf = math::fma(bf(*wf.add(d * 64 + j)), zj, sf);
        sg = math::fma(bf(*wg.add(d * 64 + j)), zj, sg);
        j += 1;
    }
    *out.add(t * ldo as usize + d) = (1.0 - th(sf)) * *h.add(t * ldh as usize + d) + th(sg) * *m.add(t * ldm as usize + d);
}

/// Message gate, step 1 (batched form): g[t] = bf16([norm(h), norm(m)]) -> bf16 [T][4608]. grid = tokens, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gate_g(g: *mut u16, h: *const f32, ldh: u32, m: *const f32, ldm: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    const D: u32 = 2304;
    let red: Shared<f32> = static_shared!(gg_red, f32, 128);
    let t = bid_x() as usize;
    let hp = h.add(t * ldh as usize);
    let mp = m.add(t * ldm as usize);
    let mut sh = 0.0f32;
    let mut sm = 0.0f32;
    let mut d = tid_x();
    while d < D {
        let a = *hp.add(d as usize);
        let b = *mp.add(d as usize);
        sh += a * a;
        sm += b * b;
        d += 256;
    }
    let ih = math::rsqrt(warp::block_sum(sh, red) / D as f32 + 1e-6);
    let im = math::rsqrt(warp::block_sum(sm, red) / D as f32 + 1e-6);
    d = tid_x();
    while d < D {
        *g.add(t * 4608 + d as usize) = tb(*hp.add(d as usize) * ih);
        *g.add(t * 4608 + (D + d) as usize) = tb(*mp.add(d as usize) * im);
        d += 256;
    }
}

/// Router logits: grid = (tokens, 4), block = 256; 8 warps x 4 experts each, f32. x bf16 [T][2304].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_router_logits(logits: *mut f32, x: *const u16, wg: *const u16) {
    crate::pdl_wait();
    crate::pdl_trigger();
    const D: u32 = 2304;
    let xs: Shared<f32> = static_shared!(rl_x, f32, 2304 * 4);
    let t = bid_x() as usize;
    let mut d = tid_x();
    while d < D {
        xs.set(d, bf(*x.add(t * D as usize + d as usize)));
        d += 256;
    }
    sync_threads();
    let l = lane();
    let mut ee = 0;
    while ee < 4 {
        let e = bid_y() * 32 + warp_id() * 4 + ee;
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
        ee += 1;
    }
}

/// Router pick: scores = sigmoid(logits), top-8 of scores + bias (lowest index on ties), weights = scores[idx] /
/// (sum + 1e-20) * route_scale; picks in ascending expert order. `forced` (i32 [T][8], ascending) replaces the picks when
/// non-null. grid = tokens, block = 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_router_pick(picks: *mut i32, wts: *mut f32, logits: *const f32, bias: *const f32,
    forced: *const i32, route_scale: f32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let t = bid_x() as usize;
    let l = lane();
    let mut sc = [0.0f32; 4];
    let mut sb = [0.0f32; 4];
    kcommon::unroll!(i in 4 => {
        let e = l as usize + 32 * i;
        sc[i] = sigm(*logits.add(t * 128 + e));
        sb[i] = sc[i] + *bias.add(e);
    });
    let mut idx = [0i32; 8];
    if !forced.is_null() {
        let mut i = 0;
        while i < 8 {
            idx[i] = *forced.add(t * 8 + i);
            i += 1;
        }
    } else {
        kcommon::unroll!(r in 8 => {
            // lane-local best, then warp argmax (value, then lowest index)
            let mut bv = -1.0e30f32;
            let mut bi = 1000u32;
            kcommon::unroll!(i in 4 => {
                if sb[i] > bv {
                    bv = sb[i];
                    bi = l + 32 * i as u32;
                }
            });
            let m = warp::max(bv);
            let cand = if bv == m { bi } else { 1000 };
            let mut c = cand;
            let mut off = 16;
            while off > 0 {
                let o = warp::shfl_xor_u32(c, off);
                if o < c {
                    c = o;
                }
                off >>= 1;
            }
            idx[r] = c as i32;
            // retire the pick
            kcommon::unroll!(i in 4 => {
                if l + 32 * i as u32 == c {
                    sb[i] = -2.0e30;
                }
            });
        });
        let mut a = 1;
        while a < 8 {
            let v = idx[a];
            let mut b = a;
            while b > 0 && idx[b - 1] > v {
                idx[b] = idx[b - 1];
                b -= 1;
            }
            idx[b] = v;
            a += 1;
        }
    }
    if l == 0 {
        let mut sum = 0.0f32;
        let mut s8 = [0.0f32; 8];
        let mut i = 0;
        while i < 8 {
            s8[i] = sigm(*logits.add(t * 128 + idx[i] as usize));
            sum += s8[i];
            i += 1;
        }
        i = 0;
        while i < 8 {
            *picks.add(t * 8 + i) = idx[i];
            *wts.add(t * 8 + i) = s8[i] / (sum + 1e-20) * route_scale;
            i += 1;
        }
    }
}

/// Message gate, step 2 on tensor cores (prompt rows): out = (1 - tanh(z . Wf^T)) * h + tanh(z . Wg^T) * m, z bf16 [T][64], Wf / Wg
/// bf16 [2304][64] (exact in the products, f32 accumulate). Block = 16 tokens x 32 channels, one warp an 8-channel slice.
/// grid = (ceil(T / 16), 2304 / 32), block = 128.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gate_mix(out: *mut f32, ldo: u32, h: *const f32, ldh: u32, m: *const f32, ldm: u32,
    z: *const u16, wf: *const u16, wg: *const u16, ntok: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    use kcommon::mma::mma_bf16_16816;
    let t0 = bid_x() * 16;
    let c0 = bid_y() * 32 + warp_id() * 8;
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    let (ra, rb_) = (t0 + g, t0 + g + 8);
    let mut accf = [0.0f32; 4];
    let mut accg = [0.0f32; 4];
    kcommon::unroll!(ks in 4 => {
        let k0 = ks as u32 * 16;
        let ld = |r: u32, k: u32| -> u32 { if r < ntok { *(z.add(r as usize * 64 + k as usize) as *const u32) } else { 0 } };
        let af = [ld(ra, k0 + 2 * t4), ld(rb_, k0 + 2 * t4), ld(ra, k0 + 8 + 2 * t4), ld(rb_, k0 + 8 + 2 * t4)];
        let wrow = (c0 + g) as usize * 64;
        let bf_ = [*(wf.add(wrow + (k0 + 2 * t4) as usize) as *const u32), *(wf.add(wrow + (k0 + 8 + 2 * t4) as usize) as *const u32)];
        let bg_ = [*(wg.add(wrow + (k0 + 2 * t4) as usize) as *const u32), *(wg.add(wrow + (k0 + 8 + 2 * t4) as usize) as *const u32)];
        mma_bf16_16816(&mut accf, af, bf_);
        mma_bf16_16816(&mut accg, af, bg_);
    });
    kcommon::unroll!(e in 4 => {
        let row = if e < 2 { ra } else { rb_ };
        let col = c0 + 2 * t4 + (e as u32 & 1);
        if row < ntok {
            let hv = *h.add(row as usize * ldh as usize + col as usize);
            let mv = *m.add(row as usize * ldm as usize + col as usize);
            *out.add(row as usize * ldo as usize + col as usize) = (1.0 - th(accf[e])) * hv + th(accg[e]) * mv;
        }
    });
}
