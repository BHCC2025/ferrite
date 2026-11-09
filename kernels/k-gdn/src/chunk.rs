//! The chunked delta-rule kernels (see the crate docs); head counts from `crate::dims`, so k-gdn-tp2 builds the
//! same source for one rank of a two-way head split.
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

use crate::dims::{CONV_DIM, NKH, NVH};
const DK: u32 = 128;
const DV: u32 = 128;
const K_OFF: u32 = NKH * DK;
const V_OFF: u32 = 2 * NKH * DK;
/// Rows per chunk.
const C: u32 = 32;
/// Row stride (floats) of the q / k tiles: 4 segments of 32 key elements padded to 36, so the 4 parts of a
/// column read 4 distinct bank groups.
const QS: u32 = 144;

// shared-memory layout, f32 offsets
const O_Q: u32 = 0;
const O_K: u32 = O_Q + C * QS;
const O_V: u32 = O_K + C * QS;
const O_A: u32 = O_V + C * DV; // A[r][s] = beta_r exp(L_r - L_s) k_r.k_s, s < r
const O_P: u32 = O_A + C * C; // P[r][s] = exp(L_r - L_s) q_r.k_s, s <= r
const O_B: u32 = O_P + C * C; // beta_r
const O_L: u32 = O_B + C; // ln g_r, then L_r
const O_G: u32 = O_L + C; // gamma_r = exp(L_r); [C] = gamma of the chunk's last row
const O_W: u32 = O_G + C + 4; // exp(L_last - L_s) (0 past the chunk)
const KT: u32 = C + 1; // row stride of the transposed k tile
const O_KT: u32 = O_W + C; // k transposed [128][33] for the Gram products
const O_T: u32 = O_KT + DK * KT; // T = (I + A)^-1, [32][33]
const O_R: u32 = O_T + C * KT; // R = beta (v - gamma K S0), [32][128]
const SMEM_FLOATS: u32 = O_R + C * DV;
/// Dynamic shared memory of `fe_gdn_chunk` in bytes.
pub const GDN_CHUNK_SMEM: u32 = SMEM_FLOATS * 4;

#[inline(always)]
fn shfl_up(v: f32, d: u32) -> f32 {
    f32::from_bits(warp::shfl_up_u32(v.to_bits(), d))
}

/// Padded index of key element e (0..128) inside a q / k row.
#[inline(always)]
const fn pad(e: u32) -> u32 {
    (e >> 5) * 36 + (e & 31)
}

/// Chunked delta-rule recurrence for prompt segments. Arguments as `fe_gdn_rec2` without the history:
/// o: [rows][48*128] bf16 raw output; conv: [rows][10240] post conv+silu; a / b rows (stride ldab) of 48;
/// state: [slot][48][128][128] f32 (S[dk][dv]), read and written back.
/// grid = (48, nseg), block = 512, smem = GDN_CHUNK_SMEM.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gdn_chunk(o: *mut u16, conv: *const u16, ax: *const u16, bx: *const u16, ldab: u32,
    a_log: *const u16, dt_bias: *const u16, state: *mut f32, seg_slot: *const u32, seg_row0: *const u32,
    seg_len: *const u32) {
    let h = bid_x();
    let sg = bid_y() as usize;
    let t = tid_x();
    let j = t >> 2;
    let part = t & 3;
    let l = lane();
    let wid = warp_id();
    let slot = *seg_slot.add(sg) as usize;
    let r0 = *seg_row0.add(sg) as usize;
    let n = *seg_len.add(sg) as usize;
    let sm: Shared<f32> = dyn_base();
    let kh = h / (NVH / NKH);
    let a_neg = -math::exp(bf16::to_f32(*a_log.add(h as usize)));
    let dtb = bf16::to_f32(*dt_bias.add(h as usize));

    let sp = state.add(((slot * NVH as usize + h as usize) * (DK * DV) as usize) + j as usize);
    let mut s = [0.0f32; 32];
    kcommon::unroll!(i in 32 => {
        s[i] = *sp.add(((part * 32 + i as u32) * DV) as usize);
    });

    let mut c0 = 0usize;
    while c0 < n {
        let cl = if n - c0 < C as usize { (n - c0) as u32 } else { C };

        // ---- 1. q / k rows (l2-normalised exactly as fe_gdn_rec2), v rows, beta, ln g ----------------
        kcommon::unroll!(rr in 2 => {
            let r = wid + rr as u32 * 16;
            let valid = r < cl;
            let row = r0 + c0 + r as usize;
            let mut qv = [0.0f32; 4];
            let mut kv = [0.0f32; 4];
            if valid {
                let crow = conv.add(row * CONV_DIM as usize);
                kcommon::unroll!(m in 4 => {
                    qv[m] = bf16::to_f32(*crow.add((kh * DK + m as u32 * 32 + l) as usize));
                    kv[m] = bf16::to_f32(*crow.add((K_OFF + kh * DK + m as u32 * 32 + l) as usize));
                });
            }
            // rec2: one warp sum per 32-element group, then the 4 groups added in order
            let mut tq = 0.0f32;
            let mut tk = 0.0f32;
            kcommon::unroll!(m in 4 => {
                tq += warp::sum(qv[m] * qv[m]);
                tk += warp::sum(kv[m] * kv[m]);
            });
            let iq = math::rsqrt(tq + 1e-6);
            let ik = math::rsqrt(tk + 1e-6);
            kcommon::unroll!(m in 4 => {
                let e = pad(m as u32 * 32 + l);
                sm.set(O_Q + r * QS + e, if valid { qv[m] * iq * 0.088388347648318440550f32 } else { 0.0 });
                let kn = if valid { kv[m] * ik } else { 0.0 };
                sm.set(O_K + r * QS + e, kn);
                sm.set(O_KT + (m as u32 * 32 + l) * KT + r, kn);
            });
            if l == 0 {
                let (beta, lng) = if valid {
                    let bb = bf16::to_f32(*bx.add(row * ldab as usize + h as usize));
                    let aa = bf16::to_f32(*ax.add(row * ldab as usize + h as usize));
                    (bf16::round(math::sigmoid(bb)), a_neg * math::softplus(aa + dtb))
                } else {
                    (0.0, 0.0)
                };
                sm.set(O_B + r, beta);
                sm.set(O_L + r, lng);
            }
        });
        kcommon::unroll!(m in 8 => {
            let idx = t + m as u32 * 512;
            let r = idx >> 7;
            let jj = idx & 127;
            let v = if r < cl {
                bf16::to_f32(*conv.add((r0 + c0 + r as usize) * CONV_DIM as usize + (V_OFF + h * DV + jj) as usize))
            } else {
                0.0
            };
            sm.set(O_V + r * DV + jj, v);
        });
        sync_threads();

        // ---- 2. L_r (inclusive prefix of ln g), gamma_r, exp(L_last - L_s) -------------------------------
        if wid == 0 {
            let mut v = sm.at(O_L + l);
            kcommon::unroll!(d in [1, 2, 4, 8, 16] => {
                let y = shfl_up(v, d as u32);
                if l >= d as u32 {
                    v += y;
                }
            });
            sm.set(O_L + l, v);
            sm.set(O_G + l, math::exp(v));
            let last = warp::shfl_idx(v, cl - 1);
            sm.set(O_W + l, if l < cl { math::exp(last - v) } else { 0.0 });
            if l == 0 {
                sm.set(O_G + C, math::exp(last));
            }
        }
        sync_threads();

        // ---- 3. Gram matrices: A (k.k, strictly lower, x beta_r exp(L_r - L_s)), P (q.k, lower) --------------
        // warp w: rows 2w, 2w+1; lane = column c; q / k rows broadcast, k^T conflict-free
        {
            let ra = 2 * wid;
            let (q0, q1) = (sm.add(O_Q + ra * QS), sm.add(O_Q + (ra + 1) * QS));
            let (k0, k1) = (sm.add(O_K + ra * QS), sm.add(O_K + (ra + 1) * QS));
            let kt = sm.add(O_KT + l);
            let mut g = [0.0f32; 8]; // kk0, kk1, qk0, qk1 x 2 partial chains
            let mut e = 0;
            while e < DK {
                kcommon::unroll!(u in 4 => {
                    let ee = e + u as u32;
                    let pe = pad(ee);
                    let y = kt.at(ee * KT);
                    let h2 = (u & 1) * 4;
                    g[h2] = math::fma(k0.at(pe), y, g[h2]);
                    g[h2 + 1] = math::fma(k1.at(pe), y, g[h2 + 1]);
                    g[h2 + 2] = math::fma(q0.at(pe), y, g[h2 + 2]);
                    g[h2 + 3] = math::fma(q1.at(pe), y, g[h2 + 3]);
                });
                e += 4;
            }
            let c = l;
            kcommon::unroll!(i in 2 => {
                let r = ra + i as u32;
                let kk = g[i] + g[4 + i];
                let qk = g[2 + i] + g[6 + i];
                let ok = r < cl;
                let dec = math::exp(sm.at(O_L + r) - sm.at(O_L + c));
                sm.set(O_A + r * C + c, if ok && c < r { sm.at(O_B + r) * dec * kk } else { 0.0 });
                sm.set(O_P + r * C + c, if ok && c <= r { dec * qk } else { 0.0 });
            });
        }
        sync_threads();

        // ---- 3b. T = (I + A)^-1 (unit lower triangular), row by row in warp 0, lane = column ----------------
        if wid == 0 {
            let c = l;
            let mut r = 0;
            while r < C {
                // T[r][c] = [r == c] - sum_{s<r} A[r][s] T[s][c]
                let ar = sm.add(O_A + r * C);
                let mut acc0 = 0.0f32;
                let mut acc1 = 0.0f32;
                let mut s2 = 0;
                while s2 + 1 < r {
                    acc0 = math::fma(ar.at(s2), sm.at(O_T + s2 * KT + c), acc0);
                    acc1 = math::fma(ar.at(s2 + 1), sm.at(O_T + (s2 + 1) * KT + c), acc1);
                    s2 += 2;
                }
                if s2 < r {
                    acc0 = math::fma(ar.at(s2), sm.at(O_T + s2 * KT + c), acc0);
                }
                sm.set(O_T + r * KT + c, if r == c { 1.0 } else { 0.0 } - (acc0 + acc1));
                sync_warp();
                r += 1;
            }
        }

        // ---- 4. R = beta (v - gamma K S0) for column j (partials over this part's 32 key rows, then the 4 parts
        // added by butterfly)
        let kb = sm.add(O_K + part * 36);
        let qb = sm.add(O_Q + part * 36);
        let mut r = 0;
        while r < cl {
            let kr = kb.add(r * QS);
            let mut a4 = [0.0f32; 4];
            kcommon::unroll!(c4 in 8 => {
                let k4: [f32; 4] = kr.add(c4 as u32 * 4).cast::<[f32; 4]>().ld();
                kcommon::unroll!(x in 4 => {
                    a4[x] = math::fma(k4[x], s[4 * c4 + x], a4[x]);
                });
            });
            let mut acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            acc += warp::shfl_xor(acc, 1);
            acc += warp::shfl_xor(acc, 2);
            if part == 0 {
                sm.set(O_R + r * DV + j, sm.at(O_B + r) * (sm.at(O_V + r * DV + j) - sm.at(O_G + r) * acc));
            }
            r += 1;
        }
        sync_threads();
        // ---- 5. U = T R for column j: the 4 parts split the sum (c = part mod 4), butterfly, over V in place ----
        {
            let mut r = 0;
            while r < cl {
                let tr = sm.add(O_T + r * KT);
                let mut acc = 0.0f32;
                let mut c = part;
                while c <= r {
                    acc = math::fma(tr.at(c), sm.at(O_R + c * DV + j), acc);
                    c += 4;
                }
                acc += warp::shfl_xor(acc, 1);
                acc += warp::shfl_xor(acc, 2);
                if part == 0 {
                    sm.set(O_V + r * DV + j, acc);
                }
                r += 1;
            }
        }
        sync_threads();
        // ---- 6. o_r = gamma_r (S0^T q_r) + sum_{s<=r} P[r][s] u_s: per part its 32 key rows of S0^T q and the
        // s = part (mod 4) terms, then butterfly ----------------------------------------------------------------
        let mut r = 0;
        while r < cl {
            let qr = qb.add(r * QS);
            let mut a4 = [0.0f32; 4];
            kcommon::unroll!(c4 in 8 => {
                let q4: [f32; 4] = qr.add(c4 as u32 * 4).cast::<[f32; 4]>().ld();
                kcommon::unroll!(x in 4 => {
                    a4[x] = math::fma(q4[x], s[4 * c4 + x], a4[x]);
                });
            });
            let pr = sm.add(O_P + r * C);
            let mut pu = 0.0f32;
            let mut c = part;
            while c <= r {
                pu = math::fma(pr.at(c), sm.at(O_V + c * DV + j), pu);
                c += 4;
            }
            let mut acc = sm.at(O_G + r) * ((a4[0] + a4[1]) + (a4[2] + a4[3])) + pu;
            acc += warp::shfl_xor(acc, 1);
            acc += warp::shfl_xor(acc, 2);
            if part == (r & 3) {
                let row = r0 + c0 + r as usize;
                *o.add(row * (NVH * DV) as usize + (h * DV + j) as usize) = bf16::from_f32(acc);
            }
            r += 1;
        }
        // ---- 7. S = gamma_last S0 + sum_s exp(L_last - L_s) k_s u_s^T ---------------------------------------
        let gl = sm.at(O_G + C);
        kcommon::unroll!(i in 32 => {
            s[i] *= gl;
        });
        let mut r = 0;
        while r < cl {
            let wu = sm.at(O_W + r) * sm.at(O_V + r * DV + j);
            let kr = kb.add(r * QS);
            kcommon::unroll!(c4 in 8 => {
                let k4: [f32; 4] = kr.add(c4 as u32 * 4).cast::<[f32; 4]>().ld();
                s[4 * c4] = math::fma(k4[0], wu, s[4 * c4]);
                s[4 * c4 + 1] = math::fma(k4[1], wu, s[4 * c4 + 1]);
                s[4 * c4 + 2] = math::fma(k4[2], wu, s[4 * c4 + 2]);
                s[4 * c4 + 3] = math::fma(k4[3], wu, s[4 * c4 + 3]);
            });
            r += 1;
        }
        sync_threads();
        c0 += C as usize;
    }
    kcommon::unroll!(i in 32 => {
        *sp.add(((part * 32 + i as u32) * DV) as usize) = s[i];
    });
}

// ---- tensor-core (TF32 mma.sync) variant -----------------------------------------------------------------
//
// 8 warps; warp w owns value columns d0 = 16 w .. +16 for the whole chunk loop, its 16 x 128 slice of S^T held
// as m16n8 accumulator fragments (16 dk tiles x 4 = 64 registers a lane). Per chunk, on tensor cores:
//   X1^T = S0^T K^T, X2^T = S0^T Q^T   (A = the state fragments themselves: an accumulator tile is an A
//                                        fragment once the 8 k slots of a tile are read in the order 0,2,4,6 /
//                                        1,3,5,7 -- the B rows are loaded in the same permuted order)
//   O^T  = gamma (.) X2^T + U^T P^T,    S^T = gamma_last S^T + (w (.) U)^T K.
// The Gram matrices, T = (I + A)^-1 and U = T R stay fp32 SIMT (small). Operands are rounded to TF32
// (cvt.rna), accumulation and the carried state are fp32.

const TQ: u32 = 36; // row stride of q^T / k^T ([dk][r])
const US: u32 = 136; // row stride of R / U ([r][dv])
const PS: u32 = 36; // row stride of A / P / T ([r][c])
const T_QT: u32 = 0;
const T_KT: u32 = T_QT + DK * TQ;
const T_R: u32 = T_KT + DK * TQ;
const T_U: u32 = T_R + C * US;
const T_A: u32 = T_U + C * US;
const T_P: u32 = T_A + C * PS;
const T_T: u32 = T_P + C * PS;
const T_B: u32 = T_T + C * PS;
const T_L: u32 = T_B + C;
const T_G: u32 = T_L + C; // gamma_r, [C] = gamma of the last row
const T_W: u32 = T_G + C + 4;
const T_FLOATS: u32 = T_W + C;
/// Dynamic shared memory of `fe_gdn_chunk_tc` in bytes.
pub const GDN_TC_SMEM: u32 = T_FLOATS * 4;

#[inline(always)]
fn tf32(x: f32) -> u32 {
    let r: u32;
    unsafe { core::arch::asm!("cvt.rna.tf32.f32 {r}, {x};", r = out(reg32) r, x = in(reg32) x, options(pure, nomem, nostack)) }
    r
}

/// D += A . B, m16n8k8, TF32 inputs, f32 accumulate.
#[inline(always)]
fn mma_tf32(d: &mut [f32; 4], a: [u32; 4], b: [u32; 2]) {
    unsafe {
        core::arch::asm!("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \
              {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], options(nomem, nostack));
    }
}

/// x = hi + lo as two TF32 operands (3xTF32: hi.hi + hi.lo + lo.hi recovers ~fp32 products). hi is x with the
/// 13 low mantissa bits cleared (one LOP3), lo = x - hi is exact in f32 and the mma reads its top 10 bits.
#[inline(always)]
fn split(x: f32) -> (u32, u32) {
    let hi = x.to_bits() & 0xffff_e000;
    (hi, (x - f32::from_bits(hi)).to_bits())
}

/// Tensor-core chunked delta rule; same arguments as `fe_gdn_chunk`. grid = (48, nseg), block = 256,
/// smem = GDN_TC_SMEM. All mma products use 3xTF32 (fp32-accurate).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gdn_chunk_tc(o: *mut u16, conv: *const u16, ax: *const u16, bx: *const u16,
    ldab: u32, a_log: *const u16, dt_bias: *const u16, state: *mut f32, seg_slot: *const u32, seg_row0: *const u32,
    seg_len: *const u32) {
    gdn_tc::<true>(o, conv, ax, bx, ldab, a_log, dt_bias, state, seg_slot, seg_row0, seg_len);
}
/// Same with single TF32 state products (faster, state rel error ~4e-4).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gdn_chunk_tc1(o: *mut u16, conv: *const u16, ax: *const u16, bx: *const u16,
    ldab: u32, a_log: *const u16, dt_bias: *const u16, state: *mut f32, seg_slot: *const u32, seg_row0: *const u32,
    seg_len: *const u32) {
    gdn_tc::<false>(o, conv, ax, bx, ldab, a_log, dt_bias, state, seg_slot, seg_row0, seg_len);
}

#[inline(always)]
unsafe fn gdn_tc<const X3: bool>(o: *mut u16, conv: *const u16, ax: *const u16, bx: *const u16, ldab: u32,
    a_log: *const u16, dt_bias: *const u16, state: *mut f32, seg_slot: *const u32, seg_row0: *const u32,
    seg_len: *const u32) {
    let h = bid_x();
    let sg = bid_y() as usize;
    let t = tid_x();
    let l = lane();
    let w = warp_id();
    let g = l >> 2;
    let tq = l & 3;
    let slot = *seg_slot.add(sg) as usize;
    let r0 = *seg_row0.add(sg) as usize;
    let n = *seg_len.add(sg) as usize;
    let sm: Shared<f32> = dyn_base();
    let kh = h / (NVH / NKH);
    let a_neg = -math::exp(bf16::to_f32(*a_log.add(h as usize)));
    let dtb = bf16::to_f32(*dt_bias.add(h as usize));
    let d0 = w * 16;

    // st[nt]: S^T[d0 + g (+8)][8 nt + 2 tq (+1)] = S[dk][dv] (state layout [slot][h][dk][dv])
    let sp = state.add((slot * NVH as usize + h as usize) * (DK * DV) as usize);
    let sidx = |nt: u32, q: u32| -> usize {
        ((8 * nt + 2 * tq + (q & 1)) * DV + d0 + g + 8 * (q >> 1)) as usize
    };
    let mut st = [[0.0f32; 4]; 16];
    kcommon::unroll!(nt in 16 => {
        kcommon::unroll!(q in 4 => {
            st[nt][q] = *sp.add(sidx(nt as u32, q as u32));
        });
    });

    let mut c0 = 0usize;
    while c0 < n {
        let cl = if n - c0 < C as usize { (n - c0) as u32 } else { C };

        // ---- 1. q^T / k^T (l2-normalised as fe_gdn_rec2), v -> R, beta, ln g -----------------------------
        // all global loads first (rows past the chunk read the last valid row and are zeroed afterwards), so
        // their latencies overlap
        let last = r0 + c0 + cl as usize - 1;
        let mut qraw = [[0u16; 4]; 4];
        let mut kraw = [[0u16; 4]; 4];
        let mut abraw = [(0u16, 0u16); 4];
        kcommon::unroll!(rr in 4 => {
            let r = w * 4 + rr as u32;
            let row = if r < cl { r0 + c0 + r as usize } else { last };
            let crow = conv.add(row * CONV_DIM as usize);
            kcommon::unroll!(m in 4 => {
                qraw[rr][m] = *crow.add((kh * DK + m as u32 * 32 + l) as usize);
                kraw[rr][m] = *crow.add((K_OFF + kh * DK + m as u32 * 32 + l) as usize);
            });
            abraw[rr] = (*bx.add(row * ldab as usize + h as usize), *ax.add(row * ldab as usize + h as usize));
        });
        let mut vraw = [[0u32; 4]; 2];
        kcommon::unroll!(m in 2 => {
            let idx = t + m as u32 * 256; // 16-byte chunk: row idx / 16, columns (idx % 16) * 8 ..
            let r = idx >> 4;
            let row = if r < cl { r0 + c0 + r as usize } else { last };
            vraw[m] = ldg_nc_v4(conv.add(row * CONV_DIM as usize + (V_OFF + h * DV + (idx & 15) * 8) as usize) as *const u32);
        });
        kcommon::unroll!(rr in 4 => {
            let r = w * 4 + rr as u32;
            let valid = r < cl;
            let mut qv = [0.0f32; 4];
            let mut kv = [0.0f32; 4];
            kcommon::unroll!(m in 4 => {
                qv[m] = if valid { bf16::to_f32(qraw[rr][m]) } else { 0.0 };
                kv[m] = if valid { bf16::to_f32(kraw[rr][m]) } else { 0.0 };
            });
            let mut tqs = 0.0f32;
            let mut tks = 0.0f32;
            kcommon::unroll!(m in 4 => {
                tqs += warp::sum(qv[m] * qv[m]);
                tks += warp::sum(kv[m] * kv[m]);
            });
            let iq = math::rsqrt(tqs + 1e-6);
            let ik = math::rsqrt(tks + 1e-6);
            kcommon::unroll!(m in 4 => {
                let e = m as u32 * 32 + l;
                sm.set(T_QT + e * TQ + r, if valid { qv[m] * iq * 0.088388347648318440550f32 } else { 0.0 });
                sm.set(T_KT + e * TQ + r, if valid { kv[m] * ik } else { 0.0 });
            });
            if l == 0 {
                let (beta, lng) = if valid {
                    let bb = bf16::to_f32(abraw[rr].0);
                    let aa = bf16::to_f32(abraw[rr].1);
                    (bf16::round(math::sigmoid(bb)), a_neg * math::softplus(aa + dtb))
                } else {
                    (0.0, 0.0)
                };
                sm.set(T_B + r, beta);
                sm.set(T_L + r, lng);
            }
        });
        kcommon::unroll!(m in 2 => {
            let idx = t + m as u32 * 256;
            let r = idx >> 4;
            let f = bf16::unpack8(vraw[m]);
            let dst = T_R + r * US + (idx & 15) * 8;
            kcommon::unroll!(x in 8 => {
                sm.set(dst + x as u32, if r < cl { f[x] } else { 0.0 });
            });
        });
        sync_threads();

        // ---- 2. L_r, gamma_r, exp(L_last - L_s) ----------------------------------------------------------------
        if w == 0 {
            let mut v = sm.at(T_L + l);
            kcommon::unroll!(d in [1, 2, 4, 8, 16] => {
                let y = shfl_up(v, d as u32);
                if l >= d as u32 {
                    v += y;
                }
            });
            sm.set(T_L + l, v);
            sm.set(T_G + l, math::exp(v));
            let last = warp::shfl_idx(v, cl - 1);
            sm.set(T_W + l, if l < cl { math::exp(last - v) } else { 0.0 });
            if l == 0 {
                sm.set(T_G + C, math::exp(last));
            }
        }
        sync_threads();

        // ---- 3. Gram matrices (SIMT): warp w rows 4w..4w+4, lane = column -----------------------------------------
        {
            let rb = 4 * w;
            let mut gk = [0.0f32; 4];
            let mut gq = [0.0f32; 4];
            let mut e = 0;
            while e < DK {
                kcommon::unroll!(u in 2 => {
                    let ee = e + u as u32;
                    let y = sm.at(T_KT + ee * TQ + l);
                    let k4: [f32; 4] = sm.add(T_KT + ee * TQ + rb).cast::<[f32; 4]>().ld();
                    let q4: [f32; 4] = sm.add(T_QT + ee * TQ + rb).cast::<[f32; 4]>().ld();
                    kcommon::unroll!(i in 4 => {
                        gk[i] = math::fma(k4[i], y, gk[i]);
                        gq[i] = math::fma(q4[i], y, gq[i]);
                    });
                });
                e += 2;
            }
            kcommon::unroll!(i in 4 => {
                let r = rb + i as u32;
                let c = l;
                let ok = r < cl;
                let dec = math::exp(sm.at(T_L + r) - sm.at(T_L + c));
                sm.set(T_A + r * PS + c, if ok && c < r { sm.at(T_B + r) * dec * gk[i] } else { 0.0 });
                sm.set(T_P + r * PS + c, if ok && c <= r { dec * gq[i] } else { 0.0 });
            });
        }
        sync_threads();

        // ---- 3b. T = (I + A)^-1 in warp 0 (lane = column), overlapping the other warps' step 4 ------------------
        if w == 0 {
            let c = l;
            let mut r = 0;
            while r < C {
                let ar = sm.add(T_A + r * PS);
                let mut acc0 = 0.0f32;
                let mut acc1 = 0.0f32;
                let mut s2 = 0;
                while s2 + 1 < r {
                    acc0 = math::fma(ar.at(s2), sm.at(T_T + s2 * PS + c), acc0);
                    acc1 = math::fma(ar.at(s2 + 1), sm.at(T_T + (s2 + 1) * PS + c), acc1);
                    s2 += 2;
                }
                if s2 < r {
                    acc0 = math::fma(ar.at(s2), sm.at(T_T + s2 * PS + c), acc0);
                }
                sm.set(T_T + r * PS + c, if r == c { 1.0 } else { 0.0 } - (acc0 + acc1));
                r += 1;
            }
        }

        // ---- 4. X1^T = S0^T K^T, X2^T = S0^T Q^T (TF32 mma; A = state fragments, k slots permuted) -----------
        let mut x1 = [[0.0f32; 4]; 4];
        let mut x2 = [[0.0f32; 4]; 4];
        kcommon::unroll!(ks in 16 => {
            let (h0, l0) = split(st[ks][0]);
            let (h1, l1) = split(st[ks][1]);
            let (h2, l2) = split(st[ks][2]);
            let (h3, l3) = split(st[ks][3]);
            let a = [h0, h2, h1, h3];
            let alo = [l0, l2, l1, l3];
            let kr = 8 * ks as u32 + 2 * tq;
            kcommon::unroll!(rt in 4 => {
                let cc = 8 * rt as u32 + g;
                let (k0h, k0l) = split(sm.at(T_KT + kr * TQ + cc));
                let (k1h, k1l) = split(sm.at(T_KT + (kr + 1) * TQ + cc));
                let (q0h, q0l) = split(sm.at(T_QT + kr * TQ + cc));
                let (q1h, q1l) = split(sm.at(T_QT + (kr + 1) * TQ + cc));
                if X3 {
                    mma_tf32(&mut x1[rt], alo, [k0h, k1h]);
                    mma_tf32(&mut x1[rt], a, [k0l, k1l]);
                    mma_tf32(&mut x2[rt], alo, [q0h, q1h]);
                    mma_tf32(&mut x2[rt], a, [q0l, q1l]);
                }
                mma_tf32(&mut x1[rt], a, [k0h, k1h]);
                mma_tf32(&mut x2[rt], a, [q0h, q1h]);
            });
        });
        // R = beta (v - gamma X1) in place over v; O starts as gamma X2
        kcommon::unroll!(rt in 4 => {
            kcommon::unroll!(q in 4 => {
                let r = 8 * rt as u32 + 2 * tq + (q as u32 & 1);
                let dv = d0 + g + 8 * (q as u32 >> 1);
                let vi = T_R + r * US + dv;
                let gm = sm.at(T_G + r);
                sm.set(vi, sm.at(T_B + r) * (sm.at(vi) - gm * x1[rt][q]));
                x2[rt][q] *= gm;
            });
        });
        sync_threads();

        // ---- 5. U = T R for this warp's columns (SIMT) ------------------------------------------------------
        {
            let dv = d0 + (l & 15);
            kcommon::unroll!(i in 16 => {
                let r = (l >> 4) + 2 * i as u32;
                let tr = sm.add(T_T + r * PS);
                let mut a0 = 0.0f32;
                let mut a1 = 0.0f32;
                let mut c = 0;
                while c + 1 <= r {
                    a0 = math::fma(tr.at(c), sm.at(T_R + c * US + dv), a0);
                    a1 = math::fma(tr.at(c + 1), sm.at(T_R + (c + 1) * US + dv), a1);
                    c += 2;
                }
                if c <= r {
                    a0 = math::fma(tr.at(c), sm.at(T_R + c * US + dv), a0);
                }
                sm.set(T_U + r * US + dv, a0 + a1);
            });
        }
        sync_warp();

        // ---- 6. O^T += U^T P^T (TF32 mma), store o ------------------------------------------------------------
        kcommon::unroll!(kk in 4 => {
            let k0 = 8 * kk as u32;
            let ua = T_U + (k0 + tq) * US + d0 + g;
            let ub = T_U + (k0 + tq + 4) * US + d0 + g;
            let (u0h, u0l) = split(sm.at(ua));
            let (u1h, u1l) = split(sm.at(ua + 8));
            let (u2h, u2l) = split(sm.at(ub));
            let (u3h, u3l) = split(sm.at(ub + 8));
            let a = [u0h, u1h, u2h, u3h];
            let alo = [u0l, u1l, u2l, u3l];
            kcommon::unroll!(rt in 4 => {
                let pr = T_P + (8 * rt as u32 + g) * PS + k0 + tq;
                let (p0h, p0l) = split(sm.at(pr));
                let (p1h, p1l) = split(sm.at(pr + 4));
                if X3 {
                    mma_tf32(&mut x2[rt], alo, [p0h, p1h]);
                    mma_tf32(&mut x2[rt], a, [p0l, p1l]);
                }
                mma_tf32(&mut x2[rt], a, [p0h, p1h]);
            });
        });
        kcommon::unroll!(rt in 4 => {
            kcommon::unroll!(q in 4 => {
                let r = 8 * rt as u32 + 2 * tq + (q as u32 & 1);
                let dv = d0 + g + 8 * (q as u32 >> 1);
                if r < cl {
                    *o.add((r0 + c0 + r as usize) * (NVH * DV) as usize + (h * DV + dv) as usize) = bf16::from_f32(x2[rt][q]);
                }
            });
        });

        // ---- 7. S^T = gamma_last S^T + (w (.) U)^T K (TF32 mma) ------------------------------------------------
        let gl = sm.at(T_G + C);
        kcommon::unroll!(nt in 16 => {
            kcommon::unroll!(q in 4 => {
                st[nt][q] *= gl;
            });
        });
        kcommon::unroll!(kk in 4 => {
            let k0 = 8 * kk as u32;
            let (w0, w1) = (sm.at(T_W + k0 + tq), sm.at(T_W + k0 + tq + 4));
            let ua = T_U + (k0 + tq) * US + d0 + g;
            let ub = T_U + (k0 + tq + 4) * US + d0 + g;
            let (u0h, u0l) = split(w0 * sm.at(ua));
            let (u1h, u1l) = split(w0 * sm.at(ua + 8));
            let (u2h, u2l) = split(w1 * sm.at(ub));
            let (u3h, u3l) = split(w1 * sm.at(ub + 8));
            let a = [u0h, u1h, u2h, u3h];
            let alo = [u0l, u1l, u2l, u3l];
            kcommon::unroll!(nt in 16 => {
                let kb = T_KT + (8 * nt as u32 + g) * TQ + k0 + tq;
                let (b0h, b0l) = split(sm.at(kb));
                let (b1h, b1l) = split(sm.at(kb + 4));
                if X3 {
                    mma_tf32(&mut st[nt], alo, [b0h, b1h]);
                    mma_tf32(&mut st[nt], a, [b0l, b1l]);
                }
                mma_tf32(&mut st[nt], a, [b0h, b1h]);
            });
        });
        sync_threads();
        c0 += C as usize;
    }
    kcommon::unroll!(nt in 16 => {
        kcommon::unroll!(q in 4 => {
            *sp.add(sidx(nt as u32, q as u32)) = st[nt][q];
        });
    });
}
