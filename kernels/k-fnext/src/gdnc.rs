//! Chunked gated delta rule for Flash-Next prompt rows (32 rows a step), from kernels/k-gdn (SIMT variant).
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

const NKH: u32 = 16;
const NVH: u32 = 48;
const CONV_DIM: u32 = 2 * NKH * 128 + NVH * 128;
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

/// Chunked delta-rule recurrence for prompt rows (the SIMT kernel of kernels/k-gdn adapted to Flash-Next: f32
/// conv rows, v head h reads q/k head h % 16, a = ssm_a (= -exp(A_log)) and dt_bias f32, alpha | beta rows f32 of
/// stride ldab (beta at + NVH), f32 output o [rows][48*128], one sequence). grid = 48, block = 512,
/// smem = GDN_CHUNK_SMEM.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_gdn_chunk(o: *mut f32, conv: *const f32, ab: *const f32, ldab: u32,
    ssm_a: *const f32, dt_bias: *const f32, state: *mut f32, rows: u32) {
    let h = bid_x();
    let t = tid_x();
    let j = t >> 2;
    let part = t & 3;
    let l = lane();
    let wid = warp_id();
    let slot = 0usize;
    let r0 = 0usize;
    let n = rows as usize;
    let sm: Shared<f32> = dyn_base();
    let kh = h % NKH;
    let a_neg = *ssm_a.add(h as usize);
    let dtb = *dt_bias.add(h as usize);

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
                    qv[m] = *crow.add((kh * DK + m as u32 * 32 + l) as usize);
                    kv[m] = *crow.add((K_OFF + kh * DK + m as u32 * 32 + l) as usize);
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
                    let bb = *ab.add(row * ldab as usize + (NVH + h) as usize);
                    let aa = *ab.add(row * ldab as usize + h as usize);
                    (math::sigmoid(bb), a_neg * math::softplus(aa + dtb))
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
                *conv.add((r0 + c0 + r as usize) * CONV_DIM as usize + (V_OFF + h * DV + jj) as usize)
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
                *o.add(row * (NVH * DV) as usize + (h * DV + j) as usize) = acc;
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

