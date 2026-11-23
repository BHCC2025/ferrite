//! Row norms and the attention-residual aggregation.
use crate::*;
use kcommon::smem::Shared;
use kcommon::*;

/// y[r] = bf16(w * (x[r] * rsqrt(mean(x[r]^2) + eps))); x is bf16 (in_f32 == 0) or f32; w may be null (no weight).
/// grid = rows, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_rms(y: *mut u16, x: *const u8, in_f32: u32, ldx: u32, w: *const u16, dim: u32,
    eps: f32, ldy: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let red: Shared<f32> = static_shared!(rms_red, f32, 128);
    let r = bid_x() as usize;
    let mut ss = 0.0f32;
    let mut i = tid_x();
    while i < dim {
        let v = if in_f32 != 0 { *(x as *const f32).add(r * ldx as usize + i as usize) }
            else { bf(*(x as *const u16).add(r * ldx as usize + i as usize)) };
        ss += v * v;
        i += bdim_x();
    }
    let tot = warp::block_sum(ss, red);
    let inv = math::rsqrt(tot / dim as f32 + eps);
    let mut i = tid_x();
    while i < dim {
        let v = if in_f32 != 0 { *(x as *const f32).add(r * ldx as usize + i as usize) }
            else { bf(*(x as *const u16).add(r * ldx as usize + i as usize)) };
        let mut o = v * inv;
        if !w.is_null() {
            o = bf(*w.add(i as usize)) * o;
        }
        *y.add(r * ldy as usize + i as usize) = tb(o);
        i += bdim_x();
    }
}

/// Attention residual over a stack of block states plus the current state (all RMS-normalized without a weight):
/// out[t][q] = sum_i softmax_i(scale * <query[q], norm(x_i)>) * norm(x_i), f32.
/// x_i = blocks[t * bstride + i * 2304] (bf16, i < nb), then x_nb = hs[t * ldh] (bf16 or f32 by hs_f32).
/// grid = tokens, block = 256, D = 2304 (9 elements a thread). nq <= 4, nb + 1 <= 5. Streaming partial sums (25 accumulators),
/// one fused block reduction, then a second pass mixes the states.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_blockres(out: *mut f32, ldo: u32, blocks: *const u16, bstride: u32, nb: u32,
    hs: *const u8, hs_f32: u32, ldh: u32, query: *const f32, nq: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    const D: u32 = 2304;
    let red: Shared<f32> = static_shared!(br_red, f32, 8 * 32 * 4);
    let t = bid_x() as usize;
    let n = nb + 1;
    let mut vals = [0.0f32; 32]; // [0..5) sum of squares, then q * 5 + i query dots at 5 + q * 5 + i
    let mut j = 0;
    while j < 9 {
        let d = tid_x() as usize + 256 * j;
        let mut qd = [0.0f32; 4];
        kcommon::unroll!(q in 4 => {
            if (q as u32) < nq {
                qd[q] = *query.add(q * D as usize + d);
            }
        });
        kcommon::unroll!(i in 6 => {
            if i < 5 && (i as u32) < n {
                let x = if (i as u32) < nb { bf(*blocks.add(t * bstride as usize + i * D as usize + d)) }
                    else if hs_f32 != 0 { *(hs as *const f32).add(t * ldh as usize + d) }
                    else { bf(*(hs as *const u16).add(t * ldh as usize + d)) };
                vals[i] += x * x;
                kcommon::unroll!(q in 4 => {
                    vals[5 + q * 5 + i] += qd[q] * x;
                });
            }
        });
        j += 1;
    }
    kcommon::unroll!(k in 32 => {
        vals[k] = warp::sum(vals[k]);
    });
    if lane() == 0 {
        kcommon::unroll!(k in 32 => {
            red.set(warp_id() * 32 + k as u32, vals[k]);
        });
    }
    sync_threads();
    let mut tot = [0.0f32; 32];
    kcommon::unroll!(k in 32 => {
        let mut s = 0.0f32;
        let mut w = 0;
        while w < 8 {
            s += red.at(w * 32 + k as u32);
            w += 1;
        }
        tot[k] = s;
    });
    let mut inv = [0.0f32; 5];
    kcommon::unroll!(i in 6 => {
        if i < 5 {
            inv[i] = math::rsqrt(tot[i] / D as f32 + 1e-6);
        }
    });
    let scale = math::rsqrt(D as f32);
    let mut pw = [[0.0f32; 5]; 4];
    kcommon::unroll!(q in 4 => {
        if (q as u32) < nq {
            let mut m = -1.0e30f32;
            kcommon::unroll!(i in 6 => {
                if i < 5 && (i as u32) < n {
                    let v = tot[5 + q * 5 + i] * inv[i] * scale;
                    if v > m {
                        m = v;
                    }
                }
            });
            let mut z = 0.0f32;
            kcommon::unroll!(i in 6 => {
                if i < 5 && (i as u32) < n {
                    let e = ex(tot[5 + q * 5 + i] * inv[i] * scale - m);
                    pw[q][i] = e;
                    z += e;
                }
            });
            kcommon::unroll!(i in 6 => {
                if i < 5 {
                    pw[q][i] = pw[q][i] / z * inv[i];
                }
            });
        }
    });
    let mut j = 0;
    while j < 9 {
        let d = tid_x() as usize + 256 * j;
        let mut o = [0.0f32; 4];
        kcommon::unroll!(i in 6 => {
            if i < 5 && (i as u32) < n {
                let x = if (i as u32) < nb { bf(*blocks.add(t * bstride as usize + i * D as usize + d)) }
                    else if hs_f32 != 0 { *(hs as *const f32).add(t * ldh as usize + d) }
                    else { bf(*(hs as *const u16).add(t * ldh as usize + d)) };
                kcommon::unroll!(q in 4 => {
                    o[q] += pw[q][i] * x;
                });
            }
        });
        kcommon::unroll!(q in 4 => {
            if (q as u32) < nq {
                *out.add(t * ldo as usize + q * D as usize + d) = o[q];
            }
        });
        j += 1;
    }
}

/// out = bf16(attn * sigmoid(gate)) elementwise.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gate_mul(out: *mut u16, attn: *const u16, gate: *const u16, ldg: u32, n: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let i = bid_x() * bdim_x() + tid_x();
    if i < n {
        let g = gate.add((i / 3072) as usize * ldg as usize + (i % 3072) as usize);
        *out.add(i as usize) = tb(bf(*attn.add(i as usize)) * sigm(bf(*g)));
    }
}

/// mode 0: out = f32(a); mode 1: out = f32(hs) + f32(a).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_resid(out: *mut f32, hs: *const u16, a: *const u16, mode: u32, n: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let i = bid_x() * bdim_x() + tid_x();
    if i < n {
        let mut v = bf(*a.add(i as usize));
        if mode == 1 {
            v += bf(*hs.add(i as usize));
        }
        *out.add(i as usize) = v;
    }
}

/// out = bf16(resid + f32(c)).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_final_add(out: *mut u16, resid: *const f32, c: *const u16, n: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let i = bid_x() * bdim_x() + tid_x();
    if i < n {
        *out.add(i as usize) = tb(*resid.add(i as usize) + bf(*c.add(i as usize)));
    }
}

/// Strided row copy of bf16: dst[r * ldd + c] = src[r * lds + c]; grid = rows, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_copy_rows(dst: *mut u16, ldd: u32, src: *const u16, lds: u32, cols: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let r = bid_x() as usize;
    let mut c = tid_x();
    while c < cols {
        *dst.add(r * ldd as usize + c as usize) = *src.add(r * lds as usize + c as usize);
        c += 256;
    }
}
