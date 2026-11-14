//! Prompt-row attention on tensor cores: one block per (row, kv head) runs the 12 q heads of that kv head as the 16
//! rows of an mma tile (12 used) against the row's cells (dense 0 ..= pos, or its QSA selection), 16 cells a step.
//! 4 warps split the cells (each its own online softmax and K / V tiles), then merge. K / V come from the q8 cache
//! (int8 + f32 scale per 32) dequantized to bf16 in shared memory. S = Q K^T and O += P V on mma.sync m16n8k16 bf16.
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

const HD: u32 = 256;
const ROWB: u32 = HD * 2;
const TILE: u32 = 16 * ROWB; // 16 rows of bf16
/// Q tile + 4 warps x (K tile + V tile) + m / l of 4 warps x 16 rows
pub const PATTN_SMEM: u32 = TILE + 4 * 2 * TILE + 2 * 4 * 16 * 4;

#[inline(always)]
fn sw(r: u32, c: u32) -> u32 {
    r * ROWB + ((c ^ (r & 7)) << 4)
}

/// grid = (rows, nkv), block = 128, smem = PATTN_SMEM. Same contract as fx_attn (q [rows][nq][256] f32, q8 caches,
/// gate from the q|gate projection rows, seln / sel QSA selection, positions from prm).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_pattn(out: *mut f32, q: *const f32, kq: *const i8, ks: *const f32, vq: *const i8,
    vs: *const f32, gate: *const f32, ldg: u32, gstride: u32, nq: u32, nkv: u32, prm: *const u32, scale: f32,
    sel: *const u32, seln: *const u32, ldsel: u32) {
    let sm: Shared<u8> = dyn_base();
    let r = bid_x();
    let kh = bid_y();
    let grp = nq / nkv; // 12
    let t = tid_x();
    let w = warp_id();
    let l = lane();
    // ---- Q tile: rows = the group's heads (bf16, pre-scaled), zero padding ----
    {
        let mut i = t;
        while i < 16 * 32 {
            let row = i >> 5;
            let c = i & 31;
            let mut v = [0u32; 4];
            if row < grp {
                let qp = q.add(((r * nq + kh * grp + row) * HD + c * 8) as usize);
                unroll!(e in 4 => { v[e] = bf16::pack2(*qp.add(2 * e) * scale, *qp.add(2 * e + 1) * scale); });
            }
            sm.byte_add(sw(row, c)).cast::<[u32; 4]>().st(v);
            i += 128;
        }
    }
    sync_threads();
    let ns = *seln.add(r as usize);
    let dense = ns == 0;
    let n = if dense { *prm + r + 1 } else { ns };
    let ktile = sm.byte_add(TILE + w * 2 * TILE);
    let vtile = ktile.byte_add(TILE);
    let g = l >> 2;
    let tq = l & 3;
    let mut m = [f32::NEG_INFINITY; 2];
    let mut lsum = [0.0f32; 2];
    let mut o = [[0.0f32; 4]; 32];
    let ntiles = n.div_ceil(16);
    let mut ti = w;
    while ti < ntiles {
        // ---- gather + dequantize 16 cells of K and V: lane -> cell l / 2, half l % 2 (128 dims) ----
        {
            let ci = l >> 1;
            let half = l & 1;
            let idx = ti * 16 + ci;
            let valid = idx < n;
            let cell = if !valid { 0 } else if dense { idx } else { *sel.add((r * ldsel + idx) as usize) };
            let cb = (cell * nkv + kh) as usize;
            unroll!(qd in 4 => {
                // 32 dims = one scale block: dims half*128 + qd*32 .. +32
                let d0 = half * 128 + qd as u32 * 32;
                let (sk, sv) = if valid { (*ks.add(cb * 8 + (d0 / 32) as usize), *vs.add(cb * 8 + (d0 / 32) as usize)) } else { (0.0, 0.0) };
                unroll!(part in 2 => {
                    // 16 dims = 2 chunks of 8
                    let dd = d0 + part as u32 * 16;
                    let (wk, wv) = if valid {
                        (*(kq.add(cb * 256 + dd as usize) as *const [u32; 4]), *(vq.add(cb * 256 + dd as usize) as *const [u32; 4]))
                    } else { ([0u32; 4], [0u32; 4]) };
                    unroll!(h8 in 2 => {
                        let mut ok = [0u32; 4];
                        let mut ov = [0u32; 4];
                        unroll!(e in 2 => {
                            let bk = wk[h8 * 2 + e];
                            let bvv = wv[h8 * 2 + e];
                            let f = |b: u32, s: f32, k: u32| ((b >> (8 * k)) as u8 as i8) as f32 * s;
                            ok[2 * e] = bf16::pack2(f(bk, sk, 0), f(bk, sk, 1));
                            ok[2 * e + 1] = bf16::pack2(f(bk, sk, 2), f(bk, sk, 3));
                            ov[2 * e] = bf16::pack2(f(bvv, sv, 0), f(bvv, sv, 1));
                            ov[2 * e + 1] = bf16::pack2(f(bvv, sv, 2), f(bvv, sv, 3));
                        });
                        let c = (dd + h8 as u32 * 8) / 8;
                        ktile.byte_add(sw(ci, c)).cast::<[u32; 4]>().st(ok);
                        vtile.byte_add(sw(ci, c)).cast::<[u32; 4]>().st(ov);
                    });
                });
            });
        }
        sync_warp();
        // ---- S = Q K^T for the two 8-cell halves ----
        let mut s0 = [0.0f32; 4];
        let mut s1 = [0.0f32; 4];
        unroll!(kst in 16 => {
            let a = ldmatrix_x4(sm.byte_add(sw(l & 15, kst as u32 * 2 + (l >> 4))).addr());
            let rb = l & 7;
            let cb2 = kst as u32 * 2 + ((l >> 3) & 1);
            let b0 = ldmatrix_x2(ktile.byte_add(sw(rb, cb2)).addr());
            let b1 = ldmatrix_x2(ktile.byte_add(sw(8 + rb, cb2)).addr());
            mma_bf16_16816(&mut s0, a, b0);
            mma_bf16_16816(&mut s1, a, b1);
        });
        // mask cells past n
        let c0 = ti * 16 + 2 * tq;
        let mk = |c: u32| c < n;
        if !mk(c0) { s0[0] = f32::NEG_INFINITY; s0[2] = f32::NEG_INFINITY; }
        if !mk(c0 + 1) { s0[1] = f32::NEG_INFINITY; s0[3] = f32::NEG_INFINITY; }
        if !mk(c0 + 8) { s1[0] = f32::NEG_INFINITY; s1[2] = f32::NEG_INFINITY; }
        if !mk(c0 + 9) { s1[1] = f32::NEG_INFINITY; s1[3] = f32::NEG_INFINITY; }
        // ---- online softmax, rows g (index 0) and g + 8 (index 1) ----
        let mut tm = [math::maxf(math::maxf(s0[0], s0[1]), math::maxf(s1[0], s1[1])),
            math::maxf(math::maxf(s0[2], s0[3]), math::maxf(s1[2], s1[3]))];
        unroll!(x in 2 => {
            tm[x] = math::maxf(tm[x], warp::shfl_xor(tm[x], 1));
            tm[x] = math::maxf(tm[x], warp::shfl_xor(tm[x], 2));
        });
        let mut alpha = [1.0f32; 2];
        let mut mn = [0.0f32; 2];
        unroll!(x in 2 => {
            mn[x] = math::maxf(m[x], tm[x]);
            alpha[x] = if mn[x] == f32::NEG_INFINITY { 1.0 } else { math::exp(m[x] - mn[x]) };
        });
        let ex = |v: f32, mm: f32| if v == f32::NEG_INFINITY { 0.0 } else { math::exp(v - mm) };
        let p0 = [ex(s0[0], mn[0]), ex(s0[1], mn[0]), ex(s0[2], mn[1]), ex(s0[3], mn[1])];
        let p1 = [ex(s1[0], mn[0]), ex(s1[1], mn[0]), ex(s1[2], mn[1]), ex(s1[3], mn[1])];
        lsum[0] = lsum[0] * alpha[0] + p0[0] + p0[1] + p1[0] + p1[1];
        lsum[1] = lsum[1] * alpha[1] + p0[2] + p0[3] + p1[2] + p1[3];
        m = mn;
        let pa = [bf16::pack2(p0[0], p0[1]), bf16::pack2(p0[2], p0[3]), bf16::pack2(p1[0], p1[1]), bf16::pack2(p1[2], p1[3])];
        // ---- O = alpha O + P V over the 32 8-dim tiles ----
        unroll!(nt in 32 => {
            o[nt][0] *= alpha[0];
            o[nt][1] *= alpha[0];
            o[nt][2] *= alpha[1];
            o[nt][3] *= alpha[1];
            let b = ldmatrix_x2_trans(vtile.byte_add(sw(l & 15, nt as u32)).addr());
            mma_bf16_16816(&mut o[nt], pa, b);
        });
        sync_warp();
        ti += 4;
    }
    // ---- merge the 4 warps: per warp m, l of its 16 rows, O [16][256] f32 over its K / V tiles ----
    unroll!(x in 2 => {
        lsum[x] += warp::shfl_xor(lsum[x], 1);
        lsum[x] += warp::shfl_xor(lsum[x], 2);
    });
    sync_threads();
    let ml = sm.byte_add(TILE + 8 * TILE).cast::<f32>();
    if tq == 0 {
        ml.set(w * 16 + g, m[0]);
        ml.set(w * 16 + g + 8, m[1]);
        ml.set(64 + w * 16 + g, lsum[0]);
        ml.set(64 + w * 16 + g + 8, lsum[1]);
    }
    let ow = sm.byte_add(TILE + w * 2 * TILE).cast::<f32>(); // [16][256] f32 = 16 KB
    unroll!(nt in 32 => {
        let d = nt as u32 * 8 + 2 * tq;
        ow.set(g * HD + d, o[nt][0]);
        ow.set(g * HD + d + 1, o[nt][1]);
        ow.set((g + 8) * HD + d, o[nt][2]);
        ow.set((g + 8) * HD + d + 1, o[nt][3]);
    });
    sync_threads();
    let mut i = t;
    while i < grp * HD {
        let row = i / HD;
        let d = i % HD;
        let mut mm = f32::NEG_INFINITY;
        unroll!(k in 4 => { mm = math::maxf(mm, ml.at(k as u32 * 16 + row)); });
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        unroll!(k in 4 => {
            let mk = ml.at(k as u32 * 16 + row);
            if mk != f32::NEG_INFINITY {
                let f = math::exp(mk - mm);
                num += sm.byte_add(TILE + k as u32 * 2 * TILE).cast::<f32>().at(row * HD + d) * f;
                den += ml.at(64 + k as u32 * 16 + row) * f;
            }
        });
        let h = kh * grp + row;
        let gv = *gate.add((r * ldg + h * gstride + d) as usize);
        *out.add(((r * nq + h) * HD + d) as usize) = num / den * (1.0 / (1.0 + math::exp(-gv)));
        i += 128;
    }
}
