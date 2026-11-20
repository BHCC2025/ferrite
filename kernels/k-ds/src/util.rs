//! Glue: a plain bf16-weight linear (fallback for EXL3 bit widths without a GEMV form), copies, converts.
use kcommon::smem::Shared;
use kcommon::*;

/// y[r][c] = sum_k x[r][k] * W[k][c] (W [k][n] row-major, bf16 or f32 when `w_f32`; x bf16 [rows][k] or f32 when
/// `x_f32`), f32 accumulation in k order; y f32 (`y_bf16` = 0) or bf16 (1). 8 rows a block.
/// grid = (ceil(n / 256), ceil(rows / 8)), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_linear(y: *mut u8, y_bf16: u32, x: *const u8, x_f32: u32, wt: *const u8, w_f32: u32,
    rows: u32, k: u32, n: u32) {
    let sx: Shared<f32> = kcommon::static_shared!(lb_x, f32, 32768); // [8][1024]
    let c = (bid_x() * bdim_x() + tid_x()) as usize;
    let r0 = bid_y() as usize * 8;
    let nr = (rows as usize - r0).min(8);
    let mut acc = [0.0f32; 8];
    let mut k0 = 0usize;
    while k0 < k as usize {
        let kc = (k as usize - k0).min(1024);
        sync_threads();
        let mut i = tid_x() as usize;
        while i < 8 * kc {
            let rr = i / kc;
            let kk = i % kc;
            let v = if rr < nr {
                let at = (r0 + rr) * k as usize + k0 + kk;
                if x_f32 != 0 { *(x as *const f32).add(at) } else { bf16::to_f32(*(x as *const u16).add(at)) }
            } else {
                0.0
            };
            sx.set((rr * 1024 + kk) as u32, v);
            i += bdim_x() as usize;
        }
        sync_threads();
        if c < n as usize {
            let mut kk = 0usize;
            while kk < kc {
                let at = (k0 + kk) * n as usize + c;
                let wv = if w_f32 != 0 { *(wt as *const f32).add(at) } else { bf16::to_f32(*(wt as *const u16).add(at)) };
                let mut rr = 0;
                while rr < 8 {
                    acc[rr] = math::fma(sx.at((rr * 1024 + kk) as u32), wv, acc[rr]);
                    rr += 1;
                }
                kk += 1;
            }
        }
        k0 += kc;
    }
    if c >= n as usize {
        return;
    }
    let mut rr = 0;
    while rr < nr {
        let at = (r0 + rr) * n as usize + c;
        if y_bf16 != 0 {
            *(y as *mut u16).add(at) = bf16::from_f32(acc[rr]);
        } else {
            *(y as *mut f32).add(at) = acc[rr];
        }
        rr += 1;
    }
}

/// dst[r][0..cols] = src[r][0..cols] (bf16, row strides in elements). grid = (rows, ceil(cols / 256)), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_copy_rows(dst: *mut u16, ldd: u32, src: *const u16, lds: u32, cols: u32) {
    let r = bid_x() as usize;
    let c = (bid_y() * bdim_x() + tid_x()) as usize;
    if c < cols as usize {
        *dst.add(r * ldd as usize + c) = *src.add(r * lds as usize + c);
    }
}

/// y[i] = bf16(x[i]) (f32 -> bf16), n elements. grid = ceil(n / 256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_f32_to_bf16(y: *mut u16, x: *const f32, n: u32) {
    let i = (bid_x() * bdim_x() + tid_x()) as usize;
    if i < n as usize {
        *y.add(i) = bf16::from_f32(*x.add(i));
    }
}

/// y[i] = f32(x[i]) (bf16 -> f32, exact), n elements. grid = ceil(n / 256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_bf16_to_f32(y: *mut f32, x: *const u16, n: u32) {
    let i = (bid_x() * bdim_x() + tid_x()) as usize;
    if i < n as usize {
        *y.add(i) = bf16::to_f32(*x.add(i));
    }
}

/// x[i] *= s (bf16 in place, f32 math, one rounding). grid = ceil(n / 256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_scale_bf16(x: *mut u16, s: f32, n: u32) {
    let i = (bid_x() * bdim_x() + tid_x()) as usize;
    if i < n as usize {
        *x.add(i) = bf16::from_f32(bf16::to_f32(*x.add(i)) * s);
    }
}

/// x[i] = bf16(bf16(x[i]) * s) as f32 in place (a bf16 linear output times a scalar, as torch rounds it).
/// grid = ceil(n / 256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_round_scale_f32(x: *mut f32, s: f32, n: u32) {
    let i = (bid_x() * bdim_x() + tid_x()) as usize;
    if i < n as usize {
        *x.add(i) = bf16::round(bf16::round(*x.add(i)) * s);
    }
}

/// Zero n f32. grid = ceil(n / 256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_zero_f32(x: *mut f32, n: u32) {
    let i = (bid_x() * bdim_x() + tid_x()) as usize;
    if i < n as usize {
        *x.add(i) = 0.0;
    }
}

/// Indexer head weights, split-K part (32 heads): part[r][ks][c] = sum_k x[r][k] * w[k][c] over block ks's slice
/// [ks * slice, (ks + 1) * slice) (x bf16 [rows][dim], w f32 [dim][32]). Warp j takes k = k0 + j, k0 + j + 8, ... in
/// order (lane c = column, coalesced weight rows); the 8 warp sums add in warp order. Each row's bits are its own.
/// grid = (dim / slice, rows), block = 256; slice % 8 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_iw_part(part: *mut f32, x: *const u16, w: *const f32, dim: u32, slice: u32) {
    let red: Shared<f32> = kcommon::static_shared!(iw_red, f32, 1024);
    let ks = bid_x() as usize;
    let r = bid_y() as usize;
    let nks = gdim_x() as usize;
    let wp = warp_id() as usize;
    let c = lane() as usize;
    let k0 = ks * slice as usize;
    let xr = x.add(r * dim as usize);
    let mut acc = 0.0f32;
    let mut k = k0 + wp;
    while k < k0 + slice as usize {
        acc = math::fma(bf16::to_f32(*xr.add(k)), *w.add(k * 32 + c), acc);
        k += 8;
    }
    red.set((wp * 32 + c) as u32, acc);
    sync_threads();
    if wp == 0 {
        let mut s = red.at(c as u32);
        let mut j = 1usize;
        while j < 8 {
            s += red.at((j * 32 + c) as u32);
            j += 1;
        }
        *part.add((r * nks + ks) * 32 + c) = s;
    }
}

/// Indexer head weights, finish: out[r][c] = bf16(bf16(sum_ks part[r][ks][c]) * s) (in ks order; the oracle's bf16
/// linear output times its bf16-rounded scale product). grid = rows, block = 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_iw_fin(out: *mut f32, part: *const f32, nks: u32, s: f32) {
    let r = bid_x() as usize;
    let c = lane() as usize;
    let mut v = 0.0f32;
    let mut k = 0usize;
    while k < nks as usize {
        v += *part.add((r * nks as usize + k) * 32 + c);
        k += 1;
    }
    *out.add(r * 32 + c) = bf16::round(bf16::round(v) * s);
}
