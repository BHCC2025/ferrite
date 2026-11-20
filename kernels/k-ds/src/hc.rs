//! Hyper-connections (hc = 4 streams of `dim`): embedding into 4 streams, the mixes GEMV + Sinkhorn, the
//! pre-weighted collapse + RMSNorm, and the post / comb update.
use kcommon::smem::Shared;
use kcommon::*;

/// h[r][j][:] = table[ids[r]][:] for j = 0..4 (bf16). grid = rows, block = 256; dim % 8 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_embed4(h: *mut u16, table: *const u16, ids: *const u32, dim: u32) {
    let r = bid_x() as usize;
    let id = *ids.add(r) as usize;
    let src = table.add(id * dim as usize) as *const u32;
    let nv = (dim / 8) as usize;
    let mut i = tid_x() as usize;
    while i < nv {
        let v = ldg_nc_v4(src.add(i * 4));
        let mut j = 0;
        while j < 4 {
            stg_v4((h.add((r * 4 + j) * dim as usize) as *mut u32).add(i * 4), v);
            j += 1;
        }
        i += bdim_x() as usize;
    }
}

/// The mixes of one sublayer and their Sinkhorn split (DeepSeek's hc_pre / hc_split_sinkhorn):
/// x = h[r] flattened (4*dim, bf16), mixes[k] = (x . fnw[k]) * rsqrt(mean(x^2) + eps), k < 24, then
/// pre = sigmoid(m[0..4] s0 + b) + hc_eps, post = 2 sigmoid(m[4..8] s1 + b),
/// comb = softmax_j(m[8..24] s2 + b)[i][j] + hc_eps, column-normalized, then (iters - 1) x (row, column).
/// out[r] = [pre 4 | post 4 | comb 16 (i * 4 + j: stream i into stream j)] f32.
/// grid = rows, block = 768 (24 warps, warp k computes mix k); dim4 % 256 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_hc_mix(out: *mut f32, h: *const u16, fnw: *const f32, scale: *const f32,
    base: *const f32, dim4: u32, eps: f32, hc_eps: f32, iters: u32) {
    let mx: Shared<f32> = kcommon::static_shared!(hc_mx, f32, 128);
    let r = bid_x() as usize;
    let w = warp_id() as usize;
    let l = lane() as usize;
    let xr = h.add(r * dim4 as usize) as *const u32;
    let fr = fnw.add(w * dim4 as usize) as *const u32;
    let mut dot = 0.0f32;
    let mut ss = 0.0f32;
    let nv = (dim4 / 8) as usize;
    let mut i = l;
    while i < nv {
        let v = bf16::unpack8(ldg_nc_v4(xr.add(i * 4)));
        let f0 = ldg_nc_v4(fr.add(i * 8));
        let f1 = ldg_nc_v4(fr.add(i * 8 + 4));
        let f = [f0[0], f0[1], f0[2], f0[3], f1[0], f1[1], f1[2], f1[3]];
        let mut j = 0;
        while j < 8 {
            dot = math::fma(v[j], f32::from_bits(f[j]), dot);
            ss = math::fma(v[j], v[j], ss);
            j += 1;
        }
        i += 32;
    }
    dot = warp::sum(dot);
    ss = warp::sum(ss); // every warp sums the same elements in the same order: identical in all warps
    if l == 0 {
        mx.set(w as u32, dot * math::rsqrt(ss / dim4 as f32 + eps));
    }
    sync_threads();
    if tid_x() == 0 {
        let s0 = *scale;
        let s1 = *scale.add(1);
        let s2 = *scale.add(2);
        let o = out.add(r * 24);
        let mut k = 0u32;
        while k < 4 {
            *o.add(k as usize) = math::sigmoid(mx.at(k) * s0 + *base.add(k as usize)) + hc_eps;
            *o.add(4 + k as usize) = 2.0 * math::sigmoid(mx.at(4 + k) * s1 + *base.add(4 + k as usize));
            k += 1;
        }
        let mut c = [0.0f32; 16];
        let mut i = 0;
        while i < 4 {
            let mut m = f32::NEG_INFINITY;
            let mut j = 0;
            while j < 4 {
                let v = mx.at((8 + i * 4 + j) as u32) * s2 + *base.add(8 + i * 4 + j);
                c[i * 4 + j] = v;
                m = m.max(v);
                j += 1;
            }
            let mut sum = 0.0f32;
            let mut j = 0;
            while j < 4 {
                let e = math::exp(c[i * 4 + j] - m);
                c[i * 4 + j] = e;
                sum += e;
                j += 1;
            }
            let mut j = 0;
            while j < 4 {
                c[i * 4 + j] = c[i * 4 + j] / sum + hc_eps;
                j += 1;
            }
            i += 1;
        }
        col_norm(&mut c, hc_eps);
        let mut it = 1;
        while it < iters {
            row_norm(&mut c, hc_eps);
            col_norm(&mut c, hc_eps);
            it += 1;
        }
        let mut k = 0;
        while k < 16 {
            *o.add(8 + k) = c[k];
            k += 1;
        }
    }
}

/// comb / (sum over i (dim -2) + eps)
#[inline(always)]
fn col_norm(c: &mut [f32; 16], eps: f32) {
    let mut j = 0;
    while j < 4 {
        let s = c[j] + c[4 + j] + c[8 + j] + c[12 + j] + eps;
        let mut i = 0;
        while i < 4 {
            c[i * 4 + j] /= s;
            i += 1;
        }
        j += 1;
    }
}

/// comb / (sum over j (dim -1) + eps)
#[inline(always)]
fn row_norm(c: &mut [f32; 16], eps: f32) {
    let mut i = 0;
    while i < 4 {
        let s = c[i * 4] + c[i * 4 + 1] + c[i * 4 + 2] + c[i * 4 + 3] + eps;
        let mut j = 0;
        while j < 4 {
            c[i * 4 + j] /= s;
            j += 1;
        }
        i += 1;
    }
}

/// y[r] = rms_norm(bf16(sum_j pre[r][j] * h[r][j]), w): the collapse rounds to bf16 (DeepSeek's hc_pre
/// returns the input dtype), the norm runs in f32 on that and rounds once. `pre` rows have stride `pre_ld`
/// (24 for a mix buffer; 0 broadcasts one row). grid = rows, block = 256; dim % 8 == 0, dim <= 8192.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_hc_pre_norm(y: *mut u16, h: *const u16, pre: *const f32, pre_ld: u32,
    w: *const u16, dim: u32, eps: f32) {
    pre_norm_body(y, h, pre, pre_ld, w, dim, eps, bid_x() as usize);
}

#[inline(always)]
unsafe fn pre_norm_body(y: *mut u16, h: *const u16, pre: *const f32, pre_ld: u32, w: *const u16, dim: u32, eps: f32,
    r: usize) {
    let red: Shared<f32> = kcommon::static_shared!(hpn_red, f32, 128);
    let buf: Shared<f32> = kcommon::static_shared!(hpn_buf, f32, 32768);
    let p = pre.add(r * pre_ld as usize);
    let (p0, p1, p2, p3) = (*p, *p.add(1), *p.add(2), *p.add(3));
    let d = dim as usize;
    let hr = h.add(r * 4 * d);
    let nt = bdim_x() as usize;
    let t0 = tid_x() as usize;
    // 1. the collapse, element by element, on every thread of the block (its value does not depend on the thread)
    let mut i = t0;
    while i + 3 * nt < d {
        let mut hv = [[0u16; 4]; 4];
        let mut u = 0;
        while u < 4 {
            let q = i + u * nt;
            hv[u] = [*hr.add(q), *hr.add(d + q), *hr.add(2 * d + q), *hr.add(3 * d + q)];
            u += 1;
        }
        let mut u = 0;
        while u < 4 {
            let v = p0 * bf16::to_f32(hv[u][0]) + p1 * bf16::to_f32(hv[u][1]) + p2 * bf16::to_f32(hv[u][2])
                + p3 * bf16::to_f32(hv[u][3]);
            buf.set((i + u * nt) as u32, bf16::round(v));
            u += 1;
        }
        i += 4 * nt;
    }
    while i < d {
        let v = p0 * bf16::to_f32(*hr.add(i)) + p1 * bf16::to_f32(*hr.add(d + i)) + p2 * bf16::to_f32(*hr.add(2 * d + i))
            + p3 * bf16::to_f32(*hr.add(3 * d + i));
        buf.set(i as u32, bf16::round(v));
        i += nt;
    }
    sync_threads();
    // 2. the sum of squares as the 256-thread form adds it (thread t: t, t + 256, ...; then the block sum: other
    // threads add 0), so the bits are that form's
    let bd = 256usize;
    let mut ss = 0.0f32;
    if t0 < bd {
        let mut i = t0;
        while i < d {
            let v = buf.at(i as u32);
            ss = math::fma(v, v, ss);
            i += bd;
        }
    }
    let tot = warp::block_sum(ss, red);
    let inv = math::rsqrt(tot / dim as f32 + eps);
    // 3. the output, element by element, on every thread
    let mut i = t0;
    while i < d {
        *y.add(r * d + i) = bf16::from_f32(bf16::to_f32(*w.add(i)) * (buf.at(i as u32) * inv));
        i += nt;
    }
}

/// h_out[r][j] = bf16(post[j] * y[r] + sum_i comb[i][j] * res[r][i]); y is f32 (a cross-rank sum) rounded to bf16
/// first, as the sublayer output is in DeepSeek's code. `mix` rows: [pre 4 | post 4 | comb 16] (stride 24).
/// grid = (ceil(dim / 256), rows), block = 256. h_out may alias res.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_hc_post(h_out: *mut u16, y: *const f32, res: *const u16, mix: *const f32,
    dim: u32) {
    let r = bid_y() as usize;
    let c = (bid_x() * bdim_x() + tid_x()) as usize;
    let d = dim as usize;
    if c >= d {
        return;
    }
    let m = mix.add(r * 24);
    let yv = bf16::round(*y.add(r * d + c));
    let base = r * 4 * d + c;
    let r0 = bf16::to_f32(*res.add(base));
    let r1 = bf16::to_f32(*res.add(base + d));
    let r2 = bf16::to_f32(*res.add(base + 2 * d));
    let r3 = bf16::to_f32(*res.add(base + 3 * d));
    let mut j = 0usize;
    let mut o = [0u16; 4];
    while j < 4 {
        let v = *m.add(4 + j) * yv
            + (*m.add(8 + j) * r0 + *m.add(12 + j) * r1 + *m.add(16 + j) * r2 + *m.add(20 + j) * r3);
        o[j] = bf16::from_f32(v);
        j += 1;
    }
    let mut j = 0usize;
    while j < 4 {
        *h_out.add(base + j * d) = o[j];
        j += 1;
    }
}

/// DSpark tap: out[m(r)][off + i] = bf16(mean over the 4 streams of h[r][j][i]) (f32 sum / 4, as torch's mean),
/// m(r) = map[r] when `map` is non-null (a replay cut compacted the rows: back to the forward's original rows), else
/// r. Rows of `out` have stride `ld`. grid = (ceil(dim / 256), rows), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_tap(out: *mut u16, ld: u32, off: u32, h: *const u16, dim: u32, map: *const u32) {
    let r = bid_y() as usize;
    let i = (bid_x() * bdim_x() + tid_x()) as usize;
    let d = dim as usize;
    if i >= d {
        return;
    }
    let b = r * 4 * d + i;
    let s = bf16::to_f32(*h.add(b)) + bf16::to_f32(*h.add(b + d)) + bf16::to_f32(*h.add(b + 2 * d))
        + bf16::to_f32(*h.add(b + 3 * d));
    let ro = if map.is_null() { r } else { *map.add(r) as usize };
    *out.add(ro * ld as usize + off as usize + i) = bf16::from_f32(s / 4.0);
}

/// Split-K mixes, part 1: block (ks, r) sums its slice [ks * slice, (ks + 1) * slice) of x = h[r] (flattened) for the
/// 24 mixes and the sum of squares: part[r][ks][0..24] dots, part[r][ks][24] = sum x^2. grid = (KS, rows),
/// block = 256 (warp w: mixes 3w..3w+3); slice % 256 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_hc_mix_part(part: *mut f32, h: *const u16, fnw: *const f32, dim4: u32, slice: u32) {
    mix_part_body(part, h, fnw, dim4, slice, bid_x() as usize, gdim_x() as usize, bid_y() as usize);
}

/// ds_hc_mix_part's blocks plus one more a row (the last, bid_x == KS) doing ds_hc_pre_norm of that row with `pre`
/// (the previous mixes: they must not be the ones this launch's partials feed). Same bits as the two launches.
/// grid = (KS + 1, rows), block = 256 or 768 (24 warps: one mix a warp; the pre-norm on the first 256 threads).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_hc_mix_part_pn(part: *mut f32, h: *const u16, fnw: *const f32, dim4: u32,
    slice: u32, y: *mut u16, pre: *const f32, pre_ld: u32, w: *const u16, dim: u32, eps: f32) {
    let nks = gdim_x() as usize - 1;
    let r = bid_y() as usize;
    if (bid_x() as usize) < nks {
        mix_part_body(part, h, fnw, dim4, slice, bid_x() as usize, nks, r);
    } else {
        pre_norm_body(y, h, pre, pre_ld, w, dim, eps, r);
    }
}

/// ds_hc_mix_part_pn for up to 8 rows a block group: blocks bid_x < KS take chunk bid_x of every row of the group
/// (each fnw slice read once for the group), blocks KS.. the pre-norm of row bid_x - KS of the group. 24 warps a
/// block (a mix each). Every row's sums are ds_hc_mix_part's, so the bits are the same.
/// grid = (KS + 8, ceil(rows / 8)), block = 768.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_hc_mix_part_pn8(part: *mut f32, h: *const u16, fnw: *const f32, dim4: u32,
    slice: u32, y: *mut u16, pre: *const f32, pre_ld: u32, w: *const u16, dim: u32, eps: f32, rows: u32, nks: u32) {
    let r0 = bid_y() as usize * 8;
    let nr = (rows as usize - r0).min(8);
    let bx = bid_x() as usize;
    let nks = nks as usize;
    if bx >= nks {
        if bx - nks < nr {
            pre_norm_body(y, h, pre, pre_ld, w, dim, eps, r0 + bx - nks);
        }
        return;
    }
    let ks = bx;
    let wp = warp_id() as usize; // the mix
    let l = lane() as usize;
    let k0 = ks * slice as usize;
    let fr = fnw.add(wp * dim4 as usize + k0) as *const u32;
    let mut dot = [0.0f32; 8];
    let mut ss = [0.0f32; 8];
    let nv = slice as usize / 8;
    let mut i = l;
    while i < nv {
        let f0 = ldg_nc_v4(fr.add(i * 8));
        let f1 = ldg_nc_v4(fr.add(i * 8 + 4));
        let f = [f0[0], f0[1], f0[2], f0[3], f1[0], f1[1], f1[2], f1[3]];
        let mut rr = 0;
        while rr < 8 {
            if rr < nr {
                let xr = h.add((r0 + rr) * dim4 as usize + k0) as *const u32;
                let v = bf16::unpack8(ldg_nc_v4(xr.add(i * 4)));
                let mut j = 0;
                while j < 8 {
                    dot[rr] = math::fma(v[j], f32::from_bits(f[j]), dot[rr]);
                    j += 1;
                }
                if wp == 0 {
                    let mut j = 0;
                    while j < 8 {
                        ss[rr] = math::fma(v[j], v[j], ss[rr]);
                        j += 1;
                    }
                }
            }
            rr += 1;
        }
        i += 32;
    }
    let mut rr = 0;
    while rr < 8 {
        if rr < nr {
            let o = part.add(((r0 + rr) * nks + ks) * 25);
            let sd = warp::sum(dot[rr]);
            if l == 0 {
                *o.add(wp) = sd;
            }
            if wp == 0 {
                let sq = warp::sum(ss[rr]);
                if l == 0 {
                    *o.add(24) = sq;
                }
            }
        }
        rr += 1;
    }
}

#[inline(always)]
unsafe fn mix_part_body(part: *mut f32, h: *const u16, fnw: *const f32, dim4: u32, slice: u32, ks: usize, nks: usize,
    r: usize) {
    let w = warp_id() as usize;
    let l = lane() as usize;
    let per = 24 / nwarps() as usize; // mixes a warp: 3 (8 warps) or 1 (24 warps); each mix's sum is the same
    let k0 = ks * slice as usize;
    let xr = h.add(r * dim4 as usize + k0) as *const u32;
    let mut dot = [0.0f32; 3];
    let mut ss = 0.0f32;
    let nv = slice as usize / 8;
    let mut i = l;
    while i < nv {
        let v = bf16::unpack8(ldg_nc_v4(xr.add(i * 4)));
        let mut m = 0;
        while m < per {
            let fr = fnw.add((per * w + m) * dim4 as usize + k0) as *const u32;
            let f0 = ldg_nc_v4(fr.add(i * 8));
            let f1 = ldg_nc_v4(fr.add(i * 8 + 4));
            let f = [f0[0], f0[1], f0[2], f0[3], f1[0], f1[1], f1[2], f1[3]];
            let mut j = 0;
            while j < 8 {
                dot[m] = math::fma(v[j], f32::from_bits(f[j]), dot[m]);
                j += 1;
            }
            m += 1;
        }
        if w == 0 {
            let mut j = 0;
            while j < 8 {
                ss = math::fma(v[j], v[j], ss);
                j += 1;
            }
        }
        i += 32;
    }
    let o = part.add((r * nks + ks) * 25);
    let mut m = 0;
    while m < per {
        let s = warp::sum(dot[m]);
        if l == 0 {
            *o.add(per * w + m) = s;
        }
        m += 1;
    }
    if w == 0 {
        let s = warp::sum(ss);
        if l == 0 {
            *o.add(24) = s;
        }
    }
}

/// Split-K mixes, part 2: sum the KS partials in order, then the Sinkhorn split exactly as ds_hc_mix.
/// grid = rows, block = 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_hc_mix_fin(out: *mut f32, part: *const f32, nks: u32, scale: *const f32,
    base: *const f32, dim4: u32, eps: f32, hc_eps: f32, iters: u32) {
    let mx: Shared<f32> = kcommon::static_shared!(hcf_mx, f32, 128);
    let r = bid_x() as usize;
    let l = lane();
    if l < 25 {
        let mut s = 0.0f32;
        let mut k = 0usize;
        while k < nks as usize {
            s += *part.add((r * nks as usize + k) * 25 + l as usize);
            k += 1;
        }
        mx.set(l, s);
    }
    sync_warp();
    let inv = math::rsqrt(mx.at(24) / dim4 as f32 + eps);
    let m = if l < 24 { mx.at(l) * inv } else { 0.0 };
    sinkhorn_lanes(out.add(r * 24), m, scale, base, hc_eps, iters);
}

/// sinkhorn_out on a warp: lane k < 24 holds mix k (normalised); lanes 0-3 write pre, 4-7 post, and comb element t
/// lives on lane t (t < 16, row t / 4, column t % 4; lanes 16-31 mirror 0-15). Every sum adds in sinkhorn_out's order,
/// so the bits are the same.
#[inline(always)]
unsafe fn sinkhorn_lanes(o: *mut f32, m: f32, scale: *const f32, base: *const f32, hc_eps: f32, iters: u32) {
    let l = lane();
    if l < 4 {
        *o.add(l as usize) = math::sigmoid(m * *scale + *base.add(l as usize)) + hc_eps;
    } else if l < 8 {
        *o.add(l as usize) = 2.0 * math::sigmoid(m * *scale.add(1) + *base.add(l as usize));
    }
    let t = l & 15;
    let (ri, cj) = (t & !3, t & 3);
    // comb mixes 8..24 to lanes 0..16
    let mc = warp::shfl_idx(m, 8 + t);
    let v = mc * *scale.add(2) + *base.add(8 + t as usize);
    // row softmax: max (exact in any order), exp, sum in column order
    let mut mxr = f32::NEG_INFINITY;
    let mut j = 0u32;
    while j < 4 {
        mxr = mxr.max(warp::shfl_idx(v, ri + j));
        j += 1;
    }
    let e = math::exp(v - mxr);
    let mut sum = 0.0f32;
    let mut j = 0u32;
    while j < 4 {
        sum += warp::shfl_idx(e, ri + j);
        j += 1;
    }
    let mut c = e / sum + hc_eps;
    let colsum = |c: f32| -> f32 {
        warp::shfl_idx(c, cj) + warp::shfl_idx(c, 4 + cj) + warp::shfl_idx(c, 8 + cj) + warp::shfl_idx(c, 12 + cj)
    };
    let rowsum = |c: f32| -> f32 {
        warp::shfl_idx(c, ri) + warp::shfl_idx(c, ri + 1) + warp::shfl_idx(c, ri + 2) + warp::shfl_idx(c, ri + 3)
    };
    c /= colsum(c) + hc_eps;
    let mut it = 1;
    while it < iters {
        c /= rowsum(c) + hc_eps;
        c /= colsum(c) + hc_eps;
        it += 1;
    }
    if l < 16 {
        *o.add(8 + l as usize) = c;
    }
}

/// pre / post / comb of 24 mixes in shared memory (the second half of ds_hc_mix), one thread.
#[inline(always)]
unsafe fn sinkhorn_out(o: *mut f32, mx: Shared<f32>, scale: *const f32, base: *const f32, hc_eps: f32, iters: u32) {
    let s0 = *scale;
    let s1 = *scale.add(1);
    let s2 = *scale.add(2);
    let mut k = 0u32;
    while k < 4 {
        *o.add(k as usize) = math::sigmoid(mx.at(k) * s0 + *base.add(k as usize)) + hc_eps;
        *o.add(4 + k as usize) = 2.0 * math::sigmoid(mx.at(4 + k) * s1 + *base.add(4 + k as usize));
        k += 1;
    }
    let mut c = [0.0f32; 16];
    let mut i = 0;
    while i < 4 {
        let mut m = f32::NEG_INFINITY;
        let mut j = 0;
        while j < 4 {
            let v = mx.at((8 + i * 4 + j) as u32) * s2 + *base.add(8 + i * 4 + j);
            c[i * 4 + j] = v;
            m = m.max(v);
            j += 1;
        }
        let mut sum = 0.0f32;
        let mut j = 0;
        while j < 4 {
            let e = math::exp(c[i * 4 + j] - m);
            c[i * 4 + j] = e;
            sum += e;
            j += 1;
        }
        let mut j = 0;
        while j < 4 {
            c[i * 4 + j] = c[i * 4 + j] / sum + hc_eps;
            j += 1;
        }
        i += 1;
    }
    col_norm(&mut c, hc_eps);
    let mut it = 1;
    while it < iters {
        row_norm(&mut c, hc_eps);
        col_norm(&mut c, hc_eps);
        it += 1;
    }
    let mut k = 0;
    while k < 16 {
        *o.add(8 + k) = c[k];
        k += 1;
    }
}
