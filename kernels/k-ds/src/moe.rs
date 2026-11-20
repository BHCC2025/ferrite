//! Router (sqrt-softplus scores, selection bias, normalized top-k x scale) and the per-expert MoE fallback glue.
use crate::q;
use kcommon::smem::Shared;
use kcommon::*;

/// scores[e] = sqrt(softplus(x . gw[e])) (x bf16 [dim], gw fp16 [n_exp][dim], f32 math); pick the top `topk` of
/// scores + bias (fp16 or f32 bias, `bias_f32`), ties to the lower id, in descending order; weights = picked scores /
/// (sum + 1e-20) * route_scale. ids u32 [rows][topk], wts f32 [rows][topk], scores_out f32 [rows][n_exp] (may be
/// null). grid = rows, block = 512 (16 warps); n_exp <= 512, dim <= 8192, dim % 256 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_router(ids: *mut u32, wts: *mut f32, scores_out: *mut f32, x: *const u16,
    gw: *const u16, bias: *const u8, bias_f32: u32, dim: u32, n_exp: u32, topk: u32, route_scale: f32) {
    let sx: Shared<f32> = kcommon::static_shared!(rt_x, f32, 32768);
    let ss: Shared<f32> = kcommon::static_shared!(rt_s, f32, 2048);
    let sb: Shared<f32> = kcommon::static_shared!(rt_b, f32, 2048);
    let r = bid_x() as usize;
    let d = dim as usize;
    let mut i = tid_x() as usize;
    while i < d {
        sx.set(i as u32, bf16::to_f32(*x.add(r * d + i)));
        i += bdim_x() as usize;
    }
    sync_threads();
    let w = warp_id();
    let l = lane() as usize;
    let mut e = w;
    while e < n_exp {
        let row = gw.add(e as usize * d);
        let mut acc = 0.0f32;
        let mut k = l * 8;
        while k < d {
            let v = ldg_nc_v4(row.add(k) as *const u32);
            let mut j = 0;
            while j < 4 {
                let lo = (v[j] & 0xffff) as u16;
                let hi = (v[j] >> 16) as u16;
                acc = math::fma(sx.at((k + 2 * j) as u32), q::f16_to_f32(lo), acc);
                acc = math::fma(sx.at((k + 2 * j + 1) as u32), q::f16_to_f32(hi), acc);
                j += 1;
            }
            k += 256;
        }
        let logit = warp::sum(acc);
        if l == 0 {
            let s = math::sqrt(softplus(logit));
            ss.set(e, s);
            let b = if bias_f32 != 0 {
                *(bias as *const f32).add(e as usize)
            } else {
                q::f16_to_f32(*(bias as *const u16).add(e as usize))
            };
            sb.set(e, s + b);
            if !scores_out.is_null() {
                *scores_out.add(r * n_exp as usize + e as usize) = s;
            }
        }
        e += nwarps();
    }
    sync_threads();
    if warp_id() == 0 {
        let mut sum = 0.0f32;
        let mut picked = [0u32; 16];
        let mut k = 0u32;
        while k < topk {
            // warp argmax over sb (lowest id on ties), then knock it out
            let mut best = f32::NEG_INFINITY;
            let mut bi = u32::MAX;
            let mut e = l as u32;
            while e < n_exp {
                let v = sb.at(e);
                if v > best || (v == best && e < bi) {
                    best = v;
                    bi = e;
                }
                e += 32;
            }
            let mut m = 16;
            while m > 0 {
                let ov = warp::shfl_xor(best, m);
                let oi = warp::shfl_xor_u32(bi, m);
                if ov > best || (ov == best && oi < bi) {
                    best = ov;
                    bi = oi;
                }
                m >>= 1;
            }
            picked[k as usize] = bi;
            sum += ss.at(bi);
            sync_warp();
            if l == 0 {
                sb.set(bi, f32::NEG_INFINITY);
            }
            sync_warp();
            k += 1;
        }
        if l == 0 {
            let inv = if topk > 1 { 1.0 / (sum + 1e-20) } else { 1.0 };
            let mut k = 0;
            while k < topk as usize {
                *ids.add(r * topk as usize + k) = picked[k];
                *wts.add(r * topk as usize + k) = ss.at(picked[k]) * inv * route_scale;
                k += 1;
            }
        }
    }
}

/// torch softplus (beta 1, threshold 20) with a log1p-accurate tail.
#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        let e = math::exp(x);
        // log1p(e): for small e, ln(1 + e) loses digits; use e - e^2/2 + e^3/3 below 1e-3
        if e < 1e-3 { e * (1.0 - e * (0.5 - e / 3.0)) } else { math::ln(1.0 + e) }
    }
}

/// out[i] = x[idx[i]] (bf16 rows of dim). grid = n, block = 256; dim % 8 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_gather_rows(out: *mut u16, x: *const u16, idx: *const u32, dim: u32) {
    let i = bid_x() as usize;
    let src = x.add(*idx.add(i) as usize * dim as usize) as *const u32;
    let dst = out.add(i * dim as usize) as *mut u32;
    let mut j = tid_x() as usize;
    while j < dim as usize / 8 {
        stg_v4(dst.add(j * 4), ldg_nc_v4(src.add(j * 4)));
        j += bdim_x() as usize;
    }
}

/// h[i] = bf16(silu(min(gate, limit)) * clamp(up, -limit, limit) * w[i]) (w null: 1); gate / up bf16 [n][inter] as
/// the linears return them. grid = (ceil(inter / 256), n), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_swiglu(h: *mut u16, gate: *const u16, up: *const u16, w: *const f32, inter: u32,
    limit: f32) {
    let i = bid_y() as usize;
    let c = (bid_x() * bdim_x() + tid_x()) as usize;
    if c >= inter as usize {
        return;
    }
    let at = i * inter as usize + c;
    let g = bf16::to_f32(*gate.add(at)).min(limit);
    let u = bf16::to_f32(*up.add(at)).max(-limit).min(limit);
    let mut v = g / (1.0 + math::exp(-g)) * u;
    if !w.is_null() {
        v *= *w.add(i);
    }
    *h.add(at) = bf16::from_f32(v);
}

/// y[idx[i]] += f32(o[i]) (f32 rows of dim; o bf16, or f32 when `o_f32`), rows of one call are distinct.
/// grid = (ceil(dim / 256), n), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_scatter_add(y: *mut f32, o: *const u8, o_f32: u32, idx: *const u32, dim: u32) {
    let i = bid_y() as usize;
    let c = (bid_x() * bdim_x() + tid_x()) as usize;
    if c >= dim as usize {
        return;
    }
    let v = if o_f32 != 0 {
        *(o as *const f32).add(i * dim as usize + c)
    } else {
        bf16::to_f32(*(o as *const u16).add(i * dim as usize + c))
    };
    let dst = y.add(*idx.add(i) as usize * dim as usize + c);
    *dst += v;
}

/// Router, part 1: logits[r][e] = x[r] . gw[e] (f32 math, fp16 weights). grid = (ceil(n_exp / 16), rows), block = 512
/// (warp w: expert 16 * bid_x + w); dim % 256 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_router_logits(logits: *mut f32, x: *const u16, gw: *const u16, dim: u32, n_exp: u32) {
    let r = bid_y() as usize;
    let e = bid_x() * 16 + warp_id();
    if e >= n_exp {
        return;
    }
    let l = lane() as usize;
    let d = dim as usize;
    let xr = x.add(r * d) as *const u32;
    let row = gw.add(e as usize * d) as *const u32;
    let mut acc = 0.0f32;
    let mut k = l * 8;
    // four iterations' loads in flight; the fma order is the plain loop's
    while k + 768 < d {
        let mut xs = [[0u32; 4]; 4];
        let mut ws = [[0u32; 4]; 4];
        let mut u = 0;
        while u < 4 {
            xs[u] = ldg_nc_v4(xr.add((k + 256 * u) / 2));
            ws[u] = ldg_nc_v4(row.add((k + 256 * u) / 2));
            u += 1;
        }
        let mut u = 0;
        while u < 4 {
            let xv = bf16::unpack8(xs[u]);
            let v = ws[u];
            let mut j = 0;
            while j < 4 {
                acc = math::fma(xv[2 * j], q::f16_to_f32((v[j] & 0xffff) as u16), acc);
                acc = math::fma(xv[2 * j + 1], q::f16_to_f32((v[j] >> 16) as u16), acc);
                j += 1;
            }
            u += 1;
        }
        k += 1024;
    }
    while k < d {
        let xv = bf16::unpack8(ldg_nc_v4(xr.add(k / 2)));
        let v = ldg_nc_v4(row.add(k / 2));
        let mut j = 0;
        while j < 4 {
            acc = math::fma(xv[2 * j], q::f16_to_f32((v[j] & 0xffff) as u16), acc);
            acc = math::fma(xv[2 * j + 1], q::f16_to_f32((v[j] >> 16) as u16), acc);
            j += 1;
        }
        k += 256;
    }
    let s = warp::sum(acc);
    if l == 0 {
        *logits.add(r * n_exp as usize + e as usize) = s;
    }
}

/// ds_router_logits for up to 8 rows a block: each warp streams its expert's gate row once and keeps 8 row
/// accumulators, each summed in ds_router_logits's order (the same bits). grid = (ceil(n_exp / 16), ceil(rows / 8)),
/// block = 512.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_router_logits8(logits: *mut f32, x: *const u16, gw: *const u16, dim: u32, n_exp: u32,
    rows: u32) {
    let e = bid_x() * 16 + warp_id();
    if e >= n_exp {
        return;
    }
    let r0 = bid_y() as usize * 8;
    let nr = (rows as usize - r0).min(8);
    let l = lane() as usize;
    let d = dim as usize;
    let row = gw.add(e as usize * d) as *const u32;
    let mut acc = [0.0f32; 8];
    let mut k = l * 8;
    while k < d {
        let v = ldg_nc_v4(row.add(k / 2));
        let mut wv = [0.0f32; 8];
        let mut j = 0;
        while j < 4 {
            wv[2 * j] = q::f16_to_f32((v[j] & 0xffff) as u16);
            wv[2 * j + 1] = q::f16_to_f32((v[j] >> 16) as u16);
            j += 1;
        }
        let mut rr = 0;
        while rr < 8 {
            if rr < nr {
                let xv = bf16::unpack8(ldg_nc_v4((x.add((r0 + rr) * d) as *const u32).add(k / 2)));
                let mut j = 0;
                while j < 8 {
                    acc[rr] = math::fma(xv[j], wv[j], acc[rr]);
                    j += 1;
                }
            }
            rr += 1;
        }
        k += 256;
    }
    let mut rr = 0;
    while rr < 8 {
        if rr < nr {
            let s = warp::sum(acc[rr]);
            if l == 0 {
                *logits.add((r0 + rr) * n_exp as usize + e as usize) = s;
            }
        }
        rr += 1;
    }
}

/// Router, part 2: scores = sqrt(softplus(logits)), top-k of scores + bias (ties to the lower id), weights normalized
/// x route_scale (as ds_router). grid = rows, block = 32 or 128 (the scores on every thread, the pick on warp 0).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_router_pick(ids: *mut u32, wts: *mut f32, scores_out: *mut f32, logits: *const f32,
    bias: *const u8, bias_f32: u32, n_exp: u32, topk: u32, route_scale: f32) {
    let ss: Shared<f32> = kcommon::static_shared!(rp_s, f32, 2048);
    let sb: Shared<f32> = kcommon::static_shared!(rp_b, f32, 2048);
    let r = bid_x() as usize;
    let l = lane();
    let mut e = tid_x();
    while e < n_exp {
        let s = math::sqrt(softplus(*logits.add(r * n_exp as usize + e as usize)));
        ss.set(e, s);
        let b = if bias_f32 != 0 { *(bias as *const f32).add(e as usize) } else { q::f16_to_f32(*(bias as *const u16).add(e as usize)) };
        sb.set(e, s + b);
        if !scores_out.is_null() {
            *scores_out.add(r * n_exp as usize + e as usize) = s;
        }
        e += bdim_x();
    }
    sync_threads();
    if warp_id() != 0 {
        return;
    }
    let mut sum = 0.0f32;
    let mut picked = [0u32; 16];
    let mut k = 0u32;
    while k < topk {
        let mut best = f32::NEG_INFINITY;
        let mut bi = u32::MAX;
        let mut e = l;
        while e < n_exp {
            let v = sb.at(e);
            if v > best || (v == best && e < bi) {
                best = v;
                bi = e;
            }
            e += 32;
        }
        let mut m = 16;
        while m > 0 {
            let ov = warp::shfl_xor(best, m);
            let oi = warp::shfl_xor_u32(bi, m);
            if ov > best || (ov == best && oi < bi) {
                best = ov;
                bi = oi;
            }
            m >>= 1;
        }
        picked[k as usize] = bi;
        sum += ss.at(bi);
        sync_warp();
        if l == 0 {
            sb.set(bi, f32::NEG_INFINITY);
        }
        sync_warp();
        k += 1;
    }
    if l == 0 {
        let inv = if topk > 1 { 1.0 / (sum + 1e-20) } else { 1.0 };
        let mut k = 0;
        while k < topk as usize {
            *ids.add(r * topk as usize + k) = picked[k];
            *wts.add(r * topk as usize + k) = ss.at(picked[k]) * inv * route_scale;
            k += 1;
        }
    }
}
