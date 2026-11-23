//! Fused decode glue for the layer application (bits identical to the kernels they replace; the arithmetic is copied from
//! k-inkm: `ink_headclip`, `ink_sconv`, `ink_sconv_dyn`, `ink_headnorm`, `ink_attn_tc_comb` + `ink_gate_mul`, `ink_router_pick`).
use super::*;
use kcommon::smem::Shared;

#[inline(always)]
fn bf(h: u16) -> f32 {
    bf16::to_f32(h)
}
#[inline(always)]
fn tb(f: f32) -> u16 {
    bf16::from_f32(f)
}
#[inline(always)]
fn rb(f: f32) -> f32 {
    bf16::round(f)
}
#[inline(always)]
fn sigm(x: f32) -> f32 {
    1.0 / (1.0 + math::exp_det(-x))
}

// ---- head clip of q and rel in one launch ----------------------------------------------------------------------------------

/// `ink_headclip` body: one warp a (token, head), dh <= 128.
#[inline(always)]
unsafe fn headclip_body(wi: u32, out: *mut u16, x: *const u16, ld: u32, scale: *const u16, heads: u32, dh: u32, thr: f32, log: u32,
    pos: *const u32, ntok: u32, n_floor: f32, alpha: f32) {
    let pos0 = *pos;
    if wi >= ntok * heads {
        return;
    }
    let t = wi / heads;
    let h = wi % heads;
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

/// The query clip (dh = 128, 24 heads) and the relative-logit clip (dh = 16, 24 heads) of one attention input: blocks [0, hb) do the
/// first, [hb, 2 hb) the second, hb = ceil(T * 24 / 8). Both read the projection output `x` (row stride ld); `xr` is the second slice.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_headclip2(out_q: *mut u16, out_r: *mut u16, x: *const u16, xr: *const u16, ld: u32,
    scale_q: *const u16, scale_r: *const u16, log: u32, pos: *const u32, ntok: u32, hb: u32) {
    pdl_wait();
    pdl_trigger();
    if bid_x() < hb {
        headclip_body(bid_x() * 8 + warp_id(), out_q, x, ld, scale_q, 24, 128, 22.627417, log, pos, ntok, 8192.0, 0.1);
    } else {
        headclip_body((bid_x() - hb) * 8 + warp_id(), out_r, xr, ld, scale_r, 24, 16, 8.0, log, pos, ntok, 8192.0, 0.1);
    }
}

// ---- K / V: short convolution + head norm into the cache --------------------------------------------------------------------

/// Head norm of the conv outputs `tile` (f32 of bf16-rounded values, [nt][128] in smem) of kv head `h` into the cache, as `ink_headnorm`
/// (mode 0: L2 clip with thr, mode 1: rms without weight): warp w takes tokens w, w + 4, ...
#[inline(always)]
unsafe fn head_norm_tile(tile: Shared<f32>, cache: *mut u16, h: u32, mode: u32, thr: f32, pos: *const u32, nt: u32, ring: u32) {
    let pos0 = *pos;
    let l = lane();
    let mut t = warp_id();
    while t < nt {
        let mut v = [0.0f32; 4];
        let mut ss = 0.0f32;
        let mut i = 0;
        while i < 4 {
            v[i] = tile.at(t * 128 + l + 32 * i as u32);
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
        t += 4;
    }
}

/// `ink_sconv` (static taps, /sqrt 5) followed by `ink_headnorm`, one block a kv head (128 channels), rows <= 16.
/// grid = 8, block = 128. x = projection output [T][1024] bf16, w = conv weights [1024][4] bf16, state f32 [3 (+T if spec)][1024].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_kv_conv_norm(cache: *mut u16, x: *const u16, w: *const u16, state: *mut f32, pos: *const u32,
    nt: u32, ring: u32, mode: u32, thr: f32, spec: u32) {
    pdl_wait();
    pdl_trigger();
    let tile: Shared<f32> = static_shared!(kv_tile, f32, 16 * 128 * 4);
    const C: u32 = 1024;
    let h = bid_x();
    let ch = h * 128 + tid_x();
    let w0 = bf(*w.add(ch as usize * 4));
    let w1 = bf(*w.add(ch as usize * 4 + 1));
    let w2 = bf(*w.add(ch as usize * 4 + 2));
    let w3 = bf(*w.add(ch as usize * 4 + 3));
    let mut r0 = *state.add(ch as usize);
    let mut r1 = *state.add((C + ch) as usize);
    let mut r2 = *state.add((2 * C + ch) as usize);
    let mut t = 0;
    while t < nt {
        let xv = bf(*x.add((t * C + ch) as usize));
        if spec != 0 {
            *state.add(((3 + t) * C + ch) as usize) = xv;
        }
        let acc = math::fma(w3, xv, math::fma(w2, r2, math::fma(w1, r1, w0 * r0)));
        tile.set(t * 128 + tid_x(), rb((acc + xv) / 2.236068));
        r0 = r1;
        r1 = r2;
        r2 = xv;
        t += 1;
    }
    if spec == 0 {
        *state.add(ch as usize) = r0;
        *state.add((C + ch) as usize) = r1;
        *state.add((2 * C + ch) as usize) = r2;
    }
    sync_threads();
    head_norm_tile(tile, cache, h, mode, thr, pos, nt, ring);
}

/// `ink_sconv_dyn` (static + dynamic taps, /3) followed by `ink_headnorm`; dynw bf16 [T][32], stat bf16 [4][1024].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_kv_conv_dyn_norm(cache: *mut u16, x: *const u16, stat: *const u16, dynw: *const u16,
    state: *mut f32, pos: *const u32, nt: u32, ring: u32, mode: u32, thr: f32, spec: u32) {
    pdl_wait();
    pdl_trigger();
    let tile: Shared<f32> = static_shared!(kvd_tile, f32, 16 * 128 * 4);
    const C: u32 = 1024;
    let h = bid_x();
    let ch = h * 128 + tid_x();
    let s0 = bf(*stat.add(ch as usize));
    let s1 = bf(*stat.add((C + ch) as usize));
    let s2 = bf(*stat.add((2 * C + ch) as usize));
    let s3 = bf(*stat.add((3 * C + ch) as usize));
    let mut r0 = *state.add(ch as usize);
    let mut r1 = *state.add((C + ch) as usize);
    let mut r2 = *state.add((2 * C + ch) as usize);
    let mut t = 0;
    while t < nt {
        let xv = bf(*x.add((t * C + ch) as usize));
        if spec != 0 {
            *state.add(((3 + t) * C + ch) as usize) = xv;
        }
        let dp = dynw.add(t as usize * 32 + h as usize * 4);
        let acc = math::fma(r0, s3 + bf(*dp.add(3)), math::fma(r1, s2 + bf(*dp.add(2)),
            math::fma(r2, s1 + bf(*dp.add(1)), xv * (s0 + bf(*dp)))));
        tile.set(t * 128 + tid_x(), rb((acc + xv) / 3.0));
        r0 = r1;
        r1 = r2;
        r2 = xv;
        t += 1;
    }
    if spec == 0 {
        *state.add(ch as usize) = r0;
        *state.add((C + ch) as usize) = r1;
        *state.add((2 * C + ch) as usize) = r2;
    }
    sync_threads();
    head_norm_tile(tile, cache, h, mode, thr, pos, nt, ring);
}

// ---- attention combine + output gate -----------------------------------------------------------------------------------------

/// `ink_attn_tc_comb` followed by `ink_gate_mul`: out = bf16(bf16(sum of the partials) * sigmoid(gate)), gate bf16 with row stride ldg
/// (column = head * 128 + d). grid = (24, query tiles), block = 128.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_attn_comb_gate(out: *mut u16, part: *const f32, gate: *const u16, ldg: u32, nsup: u32, ntok: u32,
    qt_off: u32) {
    pdl_wait();
    pdl_trigger();
    let h = bid_x();
    let qi = bid_y();
    let d = tid_x();
    let qbase = (qt_off + qi) * 16;
    let mut r = 0;
    while r < 16 {
        if qbase + r < ntok {
            let mut acc = 0.0f32;
            let mut j = 0;
            while j < nsup {
                acc += *part.add((((qi * 24 + h) * nsup + j) * 2048 + r * 128 + d) as usize);
                j += 1;
            }
            let a = rb(acc);
            let g = *gate.add((qbase + r) as usize * ldg as usize + (h * 128 + d) as usize);
            *out.add((qbase + r) as usize * 3072 + h as usize * 128 + d as usize) = tb(a * sigm(bf(g)));
        }
        r += 1;
    }
}

// ---- attention residual (block states) with a cheaper reduction -------------------------------------------------------------

/// Reduce-scatter of 32 per-lane partial sums over the warp: lane l returns the sum of slot l over all lanes. Each pair of lanes adds the
/// same two operands as the butterfly of `warp::sum` (commutative), so the value equals `warp::sum` of that slot bit for bit, with
/// 31 shuffles instead of 160.
#[inline(always)]
fn scatter_sum(v: [f32; 32]) -> f32 {
    let l = lane();
    let mut a = [0.0f32; 16];
    let up = (l & 16) != 0;
    kcommon::unroll!(j in 16 => {
        let send = if up { v[j] } else { v[16 + j] };
        let keep = if up { v[16 + j] } else { v[j] };
        a[j] = keep + warp::shfl_xor(send, 16);
    });
    let mut b = [0.0f32; 8];
    let up = (l & 8) != 0;
    kcommon::unroll!(j in 8 => {
        let send = if up { a[j] } else { a[8 + j] };
        let keep = if up { a[8 + j] } else { a[j] };
        b[j] = keep + warp::shfl_xor(send, 8);
    });
    let mut c = [0.0f32; 4];
    let up = (l & 4) != 0;
    kcommon::unroll!(j in 4 => {
        let send = if up { b[j] } else { b[4 + j] };
        let keep = if up { b[4 + j] } else { b[j] };
        c[j] = keep + warp::shfl_xor(send, 4);
    });
    let mut d = [0.0f32; 2];
    let up = (l & 2) != 0;
    kcommon::unroll!(j in 2 => {
        let send = if up { c[j] } else { c[2 + j] };
        let keep = if up { c[2 + j] } else { c[j] };
        d[j] = keep + warp::shfl_xor(send, 2);
    });
    let up = (l & 1) != 0;
    let send = if up { d[0] } else { d[1] };
    let keep = if up { d[1] } else { d[0] };
    keep + warp::shfl_xor(send, 1)
}

/// `ink_blockres` (k-inkm/norm.rs) with the 32 block reductions done by `scatter_sum` and a shared [32] total instead of 256 shared reads
/// a thread. Same arithmetic everywhere else. grid = tokens, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_blockres2(out: *mut f32, ldo: u32, blocks: *const u16, bstride: u32, nb: u32,
    hs: *const u8, hs_f32: u32, ldh: u32, query: *const f32, nq: u32) {
    pdl_wait();
    pdl_trigger();
    const D: u32 = 2304;
    let red: Shared<f32> = static_shared!(br2_red, f32, 8 * 32 * 4);
    let tots: Shared<f32> = static_shared!(br2_tot, f32, 32 * 4);
    let t = bid_x() as usize;
    let n = nb + 1;
    // all loads of the thread's 9 channels up front
    let mut xs = [[0.0f32; 5]; 9];
    let mut qs = [[0.0f32; 4]; 9];
    kcommon::unroll!(j in [0, 1, 2, 3, 4, 5, 6, 7, 8] => {
        let d = tid_x() as usize + 256 * j;
        kcommon::unroll!(q in 4 => {
            if (q as u32) < nq {
                qs[j][q] = *query.add(q * D as usize + d);
            }
        });
        kcommon::unroll!(i in [0, 1, 2, 3, 4] => {
            if (i as u32) < n {
                xs[j][i] = if (i as u32) < nb { bf(*blocks.add(t * bstride as usize + i * D as usize + d)) }
                    else if hs_f32 != 0 { *(hs as *const f32).add(t * ldh as usize + d) }
                    else { bf(*(hs as *const u16).add(t * ldh as usize + d)) };
            }
        });
    });
    let mut vals = [0.0f32; 32];
    kcommon::unroll!(j in [0, 1, 2, 3, 4, 5, 6, 7, 8] => {
        kcommon::unroll!(i in [0, 1, 2, 3, 4] => {
            if (i as u32) < n {
                let x = xs[j][i];
                vals[i] += x * x;
                kcommon::unroll!(q in 4 => {
                    vals[5 + q * 5 + i] += qs[j][q] * x;
                });
            }
        });
    });
    let mine = scatter_sum(vals);
    red.set(warp_id() * 32 + lane(), mine);
    sync_threads();
    if warp_id() == 0 {
        let mut s = 0.0f32;
        let mut w = 0;
        while w < 8 {
            s += red.at(w * 32 + lane());
            w += 1;
        }
        tots.set(lane(), s);
    }
    sync_threads();
    let mut tot = [0.0f32; 32];
    kcommon::unroll!(k in 32 => {
        tot[k] = tots.at(k as u32);
    });
    let mut inv = [0.0f32; 5];
    kcommon::unroll!(i in [0, 1, 2, 3, 4] => {
        inv[i] = math::rsqrt(tot[i] / D as f32 + 1e-6);
    });
    let scale = math::rsqrt(D as f32);
    let mut pw = [[0.0f32; 5]; 4];
    kcommon::unroll!(q in 4 => {
        if (q as u32) < nq {
            let mut m = -1.0e30f32;
            kcommon::unroll!(i in [0, 1, 2, 3, 4] => {
                if (i as u32) < n {
                    let v = tot[5 + q * 5 + i] * inv[i] * scale;
                    if v > m {
                        m = v;
                    }
                }
            });
            let mut z = 0.0f32;
            kcommon::unroll!(i in [0, 1, 2, 3, 4] => {
                if (i as u32) < n {
                    let e = math::exp_det(tot[5 + q * 5 + i] * inv[i] * scale - m);
                    pw[q][i] = e;
                    z += e;
                }
            });
            kcommon::unroll!(i in [0, 1, 2, 3, 4] => {
                pw[q][i] = pw[q][i] / z * inv[i];
            });
        }
    });
    kcommon::unroll!(j in [0, 1, 2, 3, 4, 5, 6, 7, 8] => {
        let d = tid_x() as usize + 256 * j;
        let mut o = [0.0f32; 4];
        kcommon::unroll!(i in [0, 1, 2, 3, 4] => {
            if (i as u32) < n {
                kcommon::unroll!(q in 4 => {
                    o[q] += pw[q][i] * xs[j][i];
                });
            }
        });
        kcommon::unroll!(q in 4 => {
            if (q as u32) < nq {
                *out.add(t * ldo as usize + q * D as usize + d) = o[q];
            }
        });
    });
}

// ---- message gate: z = bf16(g . Wd^T) with a short dependent chain -------------------------------------------------------------

/// c[t][64] (bf16) = bf16( a[t][4608] . w[64][4608]^T ) with the arithmetic of k-inkm's `ink_gemm_s` (and so of core's bf16 GEMM):
/// `mma.m16n8k16` bf16, k16 steps in order into one f32 accumulator. One warp per CTA, one n8 column tile and one 16-row tile each,
/// operand fragments loaded straight from global memory 8 steps ahead (the 288-step mma chain is the whole latency, not the
/// pipeline of the tile kernel). grid = (8, ceil(m / 16)), block = 32. K = 4608.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_z(c: *mut u16, a: *const u16, w: *const u16, m: u32, lda: u32, ldc: u32) {
    use kcommon::mma::mma_bf16_16816;
    const K: u32 = 4608;
    let l = lane();
    let gr = l >> 2;
    let t4 = l & 3;
    let n0 = bid_x() * 8;
    let m0 = bid_y() * 16;
    // weights do not depend on the previous grid: prefetch the first lines of this tile's 8 rows, then wait
    let wrow = w.add((n0 + gr) as usize * K as usize);
    pdl_wait();
    pdl_trigger();
    let r0 = m0 + gr;
    let r1 = m0 + gr + 8;
    let a0p = a.add(r0 as usize * lda as usize);
    let a1p = a.add(r1 as usize * lda as usize);
    let mut acc = [0.0f32; 4];
    macro_rules! load_group {
        ($k0:expr) => {{
            let mut af = [[0u32; 4]; 8];
            let mut bf = [[0u32; 2]; 8];
            kcommon::unroll!(s in 8 => {
                let k = $k0 + s as u32 * 16 + 2 * t4;
                af[s][0] = if r0 < m { *(a0p.add(k as usize) as *const u32) } else { 0 };
                af[s][1] = if r1 < m { *(a1p.add(k as usize) as *const u32) } else { 0 };
                af[s][2] = if r0 < m { *(a0p.add(k as usize + 8) as *const u32) } else { 0 };
                af[s][3] = if r1 < m { *(a1p.add(k as usize + 8) as *const u32) } else { 0 };
                bf[s][0] = *(wrow.add(k as usize) as *const u32);
                bf[s][1] = *(wrow.add(k as usize + 8) as *const u32);
            });
            (af, bf)
        }};
    }
    // software pipeline: the next group's loads are issued before the current group's mma chain
    let (mut caf, mut cbf) = load_group!(0u32);
    let mut k0 = 128u32;
    while k0 < K {
        let (naf, nbf) = load_group!(k0);
        kcommon::unroll!(s in 8 => {
            mma_bf16_16816(&mut acc, caf[s], cbf[s]);
        });
        caf = naf;
        cbf = nbf;
        k0 += 128;
    }
    kcommon::unroll!(s in 8 => {
        mma_bf16_16816(&mut acc, caf[s], cbf[s]);
    });
    let col = n0 + 2 * t4;
    if r0 < m {
        *(c.add(r0 as usize * ldc as usize + col as usize) as *mut u32) = bf16::pack2(acc[0], acc[1]);
    }
    if r1 < m {
        *(c.add(r1 as usize * ldc as usize + col as usize) as *mut u32) = bf16::pack2(acc[2], acc[3]);
    }
}
