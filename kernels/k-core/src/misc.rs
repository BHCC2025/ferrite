//! Elementwise and row kernels: embedding gather, zero-centred RMSNorm, SwiGLU, row gather, argmax.
use kcommon::smem::{Shared};
use kcommon::*;

/// out[r][:] = table[ids[r]][:]; grid = rows, block = 256. `dim` % 8 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_embed(out: *mut u16, table: *const u16, ids: *const u32, dim: u32) {
    let r = bid_x();
    let id = *ids.add(r as usize);
    let src = table.add(id as usize * dim as usize) as *const u32;
    let dst = out.add(r as usize * dim as usize) as *mut u32;
    let mut i = tid_x();
    while i < dim / 8 {
        stg_v4(dst.add(i as usize * 4), ldg_nc_v4(src.add(i as usize * 4)));
        i += bdim_x();
    }
}

/// Qwen3.5 RMSNorm: y = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)), all math in f32.
/// grid = rows, block = 256 (8 warps), `dim` % 8 == 0. Row strides ldx / ldy (elements).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_rmsnorm(y: *mut u16, x: *const u16, w: *const u16, dim: u32, eps: f32, ldx: u32,
    ldy: u32) {
    let red: Shared<f32> = kcommon::static_shared!(rms_red, f32, 128);
    let r = bid_x() as usize;
    let xr = x.add(r * ldx as usize) as *const u32;
    let yr = y.add(r * ldy as usize) as *mut u32;
    let wr = w as *const u32;
    let nv = dim / 8;
    let mut ss = 0.0f32;
    let mut i = tid_x();
    while i < nv {
        let v = bf16::unpack8(ldg_nc_v4(xr.add(i as usize * 4)));
        let mut j = 0;
        while j < 8 {
            ss += v[j] * v[j];
            j += 1;
        }
        i += bdim_x();
    }
    let tot = warp::block_sum(ss, red);
    let inv = math::rsqrt(tot / dim as f32 + eps);
    let mut i = tid_x();
    while i < nv {
        let v = bf16::unpack8(ldg_nc_v4(xr.add(i as usize * 4)));
        let wv = bf16::unpack8(ldg_nc_v4(wr.add(i as usize * 4)));
        let mut o = [0u32; 4];
        let mut j = 0;
        while j < 4 {
            o[j] = bf16::pack2(v[2 * j] * inv * (1.0 + wv[2 * j]), v[2 * j + 1] * inv * (1.0 + wv[2 * j + 1]));
            j += 1;
        }
        stg_v4(yr.add(i as usize * 4), o);
        i += bdim_x();
    }
}

/// SwiGLU: out[r][c] = bf16(bf16(silu(gate[r][c])) * up[r][c]); gate/up rows share stride `ld`.
/// grid = (ceil(inter/2/256), rows).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_silu_mul(out: *mut u16, gate: *const u16, up: *const u16, ld: u32, inter: u32) {
    let r = bid_y() as usize;
    let c2 = bid_x() * bdim_x() + tid_x(); // pair index
    if c2 * 2 >= inter {
        return;
    }
    let g = *(gate.add(r * ld as usize + c2 as usize * 2) as *const u32);
    let u = *(up.add(r * ld as usize + c2 as usize * 2) as *const u32);
    let s0 = bf16::round(math::silu(bf16::lo(g)));
    let s1 = bf16::round(math::silu(bf16::hi(g)));
    *(out.add(r * inter as usize + c2 as usize * 2) as *mut u32) = bf16::pack2(s0 * bf16::lo(u), s1 * bf16::hi(u));
}

/// out[i][:] = x[idx[i]][:] (bf16 rows of `dim`); grid = n, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gather_rows(out: *mut u16, x: *const u16, idx: *const u32, dim: u32) {
    let i = bid_x() as usize;
    let src = x.add(*idx.add(i) as usize * dim as usize) as *const u32;
    let dst = out.add(i * dim as usize) as *mut u32;
    let mut j = tid_x();
    while j < dim / 8 {
        stg_v4(dst.add(j as usize * 4), ldg_nc_v4(src.add(j as usize * 4)));
        j += bdim_x();
    }
}

/// Greedy argmax over bf16 logits rows: out[r] = argmax_j x[r][j]. grid = rows, block = 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_argmax(out: *mut u32, x: *const u16, n: u32) {
    let sv: Shared<f32> = kcommon::static_shared!(am_v, f32, 128);
    let si: Shared<u32> = kcommon::static_shared!(am_i, u32, 128);
    let r = bid_x() as usize;
    let xr = x.add(r * n as usize);
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0u32;
    let mut j = tid_x();
    while j < n {
        let v = bf16::to_f32(*xr.add(j as usize));
        if v > best {
            best = v;
            bi = j;
        }
        j += bdim_x();
    }
    // warp argmax (lowest index on ties)
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
    if lane() == 0 {
        sv.set(warp_id(), best);
        si.set(warp_id(), bi);
    }
    sync_threads();
    if warp_id() == 0 {
        let nw = nwarps();
        let l = lane();
        best = if l < nw { sv.at(l) } else { f32::NEG_INFINITY };
        bi = if l < nw { si.at(l) } else { u32::MAX };
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
        if l == 0 {
            *out.add(r) = bi;
        }
    }
}

/// bf16 -> f32 copy (for host-side sampling). grid = ceil(n/256), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_bf16_to_f32(out: *mut f32, x: *const u16, n: u32) {
    let i = bid_x() * bdim_x() + tid_x();
    if i < n {
        *out.add(i as usize) = bf16::to_f32(*x.add(i as usize));
    }
}

/// In-place residual add: x[i] = bf16(x[i] + y[i]) over n elements (n even). grid = ceil(n/512), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_add_bf16(x: *mut u16, y: *const u16, n: u32) {
    let i = bid_x() * bdim_x() + tid_x();
    if i * 2 < n {
        let p = x.add(i as usize * 2) as *mut u32;
        let a = *p;
        let b = *(y.add(i as usize * 2) as *const u32);
        *p = bf16::pack2(bf16::lo(a) + bf16::lo(b), bf16::hi(a) + bf16::hi(b));
    }
}

/// dst[r][0..cols] = src[r][0..cols] for bf16 rows (cols even). grid = (rows, ceil(cols/2/256)), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_copy2d(dst: *mut u16, ldd: u32, src: *const u16, lds: u32, cols: u32) {
    let r = bid_x() as usize;
    let c = (bid_y() * bdim_x() + tid_x()) * 2;
    if c < cols {
        *(dst.add(r * ldd as usize + c as usize) as *mut u32) = *(src.add(r * lds as usize + c as usize) as *const u32);
    }
}

/// Verify-tree commit: rows path[j] of a buffer move to row j, j = 1..n, in order (K/V at the verify start, the saved
/// GDN replay inputs, the drafter's feature taps). `tab` holds one (base pointer, row elements) pair per buffer, one
/// block each. Path indices strictly increase, so a source is never a lower destination; every thread owns the
/// same elements of every row, which keeps each element's read-before-overwrite order. Rows are multiples of 8
/// elements, 16-byte aligned. block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_rows_gather(tab: *const u64, path: *const u32, n: u32) {
    let base = *tab.add(2 * bid_x() as usize) as *mut u16;
    let row = *tab.add(2 * bid_x() as usize + 1) as usize;
    let mut j = 1;
    while j < n {
        let p = *path.add(j as usize);
        if p != j {
            let src = base.add(p as usize * row) as *const [u32; 4];
            let dst = base.add(j as usize * row) as *mut [u32; 4];
            let mut i = tid_x() as usize;
            while i < row / 8 {
                *dst.add(i) = *src.add(i);
                i += bdim_x() as usize;
            }
        }
        j += 1;
    }
}
