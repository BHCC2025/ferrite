//! QSA block scores on tensor cores: score[r][b] = scale * sum_h relu(q[r][h] . pooled[b]) for 4 indexer heads of
//! 128, as a TF32 mma.sync GEMM. A block takes 16 rows x 4 heads (= 64 mma rows, head h in m tile h, so a thread
//! holds the 4 heads of its rows and sums them in registers) against 64 pooled keys a step. Same contract as
//! `fx_idx_score` (only b < nv(r) is written; rows with nv <= top are left alone).
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

use crate::fast::{pu, P_POS};

const NH: u32 = 4;
const DK: u32 = 128;
/// Row stride (floats) of the Q and K tiles: 132 makes both fragment loads conflict-free.
const ST: u32 = 132;
const BM: u32 = 16; // rows per block
const BN: u32 = 64; // pooled keys per step
const S_Q: u32 = 0;
const S_K: u32 = S_Q + NH * BM * ST;
const S_FLOATS: u32 = S_K + BN * ST;
/// Dynamic shared memory of `fx_idx_score_tc` in bytes.
pub const IDX_TC_SMEM: u32 = S_FLOATS * 4;

#[inline(always)]
fn tf32(x: f32) -> u32 {
    let r: u32;
    unsafe { core::arch::asm!("cvt.rna.tf32.f32 {r}, {x};", r = out(reg32) r, x = in(reg32) x, options(pure, nomem, nostack)) }
    r
}

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

/// x = hi + lo as two TF32 operands (3xTF32).
#[inline(always)]
fn split(x: f32) -> (u32, u32) {
    let hi = x.to_bits() & 0xffff_e000;
    (hi, (x - f32::from_bits(hi)).to_bits())
}

/// grid = (ceil(rows / 16), splits), block = 128, smem = IDX_TC_SMEM. q [rows][4][128] f32, pooled [blocks][128].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_idx_score_tc(score: *mut f32, q: *const f32, pooled: *const f32, prm: *const u32,
    kp: u32, ld: u32, top: u32, scale: f32, rows: u32) {
    idx_tc::<false>(score, q, pooled, prm, kp, ld, top, scale, rows);
}
/// Same with 3xTF32 products (fp32-accurate scores).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_idx_score_tc3(score: *mut f32, q: *const f32, pooled: *const f32, prm: *const u32,
    kp: u32, ld: u32, top: u32, scale: f32, rows: u32) {
    idx_tc::<true>(score, q, pooled, prm, kp, ld, top, scale, rows);
}

#[inline(always)]
unsafe fn idx_tc<const X3: bool>(score: *mut f32, q: *const f32, pooled: *const f32, prm: *const u32, kp: u32, ld: u32,
    top: u32, scale: f32, rows: u32) {
    let sm: Shared<f32> = dyn_base();
    let r0 = bid_x() * BM;
    let t = tid_x();
    let w = warp_id();
    let l = lane();
    let g = l >> 2;
    let tq = l & 3;
    let pos = pu(prm, P_POS);
    let rlast = (r0 + BM).min(rows) - 1;
    let nvmax = ((pos + rlast + 1) / kp).min(ld);
    if nvmax <= top {
        return;
    }
    // this split's share of the key range, whole steps
    let steps = nvmax.div_ceil(BN);
    let per = steps.div_ceil(gdim_y());
    let b_lo = bid_y() * per * BN;
    let b_hi = ((bid_y() + 1) * per * BN).min(nvmax);
    if b_lo >= b_hi {
        return;
    }
    // ---- Q tile: m row h * 16 + rr = head h of row r0 + rr (zero past the rows) ----
    {
        let mut i = t;
        while i < NH * BM * (DK / 4) {
            let m = i / (DK / 4);
            let c4 = (i % (DK / 4)) * 4;
            let (h, rr) = (m / BM, m % BM);
            let r = r0 + rr;
            let v = if r < rows {
                let p = q.add(((r * NH + h) * DK + c4) as usize);
                [*p, *p.add(1), *p.add(2), *p.add(3)]
            } else {
                [0.0; 4]
            };
            let d = S_Q + m * ST + c4;
            kcommon::unroll!(x in 4 => { sm.set(d + x as u32, v[x]); });
            i += 128;
        }
    }
    // the rows' visible block counts (rows g and g + 8 of the tile)
    let nv_a = (pos + r0 + g + 1) / kp;
    let nv_b = (pos + r0 + g + 8 + 1) / kp;
    let ok_a = r0 + g < rows && nv_a > top;
    let ok_b = r0 + g + 8 < rows && nv_b > top;
    let mut b0 = b_lo;
    while b0 < b_hi {
        sync_threads();
        // ---- K tile: 64 pooled keys (zero past the visible range) ----
        {
            let mut i = t;
            while i < BN * (DK / 4) {
                let n = i / (DK / 4);
                let c4 = (i % (DK / 4)) * 4;
                let b = b0 + n;
                let v = if b < b_hi {
                    let p = pooled.add((b * DK + c4) as usize);
                    [*p, *p.add(1), *p.add(2), *p.add(3)]
                } else {
                    [0.0; 4]
                };
                let d = S_K + n * ST + c4;
                kcommon::unroll!(x in 4 => { sm.set(d + x as u32, v[x]); });
                i += 128;
            }
        }
        sync_threads();
        // ---- warp w: keys w * 16 .. + 16 (two n8 tiles) x the 4 head tiles ----
        let mut acc = [[[0.0f32; 4]; 2]; 4];
        kcommon::unroll!(ks in 16 => {
            let k0 = ks as u32 * 8;
            let mut bh = [[0u32; 2]; 2];
            let mut bl = [[0u32; 2]; 2];
            kcommon::unroll!(nt in 2 => {
                let n = w * 16 + nt as u32 * 8 + g;
                let x0 = sm.at(S_K + n * ST + k0 + tq);
                let x1 = sm.at(S_K + n * ST + k0 + tq + 4);
                if X3 {
                    let (h0, l0) = split(x0);
                    let (h1, l1) = split(x1);
                    bh[nt] = [h0, h1];
                    bl[nt] = [l0, l1];
                } else {
                    bh[nt] = [tf32(x0), tf32(x1)];
                }
            });
            kcommon::unroll!(h in 4 => {
                let mb = S_Q + (h as u32 * BM) * ST + k0;
                let y = [sm.at(mb + g * ST + tq), sm.at(mb + (g + 8) * ST + tq), sm.at(mb + g * ST + tq + 4),
                    sm.at(mb + (g + 8) * ST + tq + 4)];
                if X3 {
                    let (a0, c0) = split(y[0]);
                    let (a1, c1) = split(y[1]);
                    let (a2, c2) = split(y[2]);
                    let (a3, c3) = split(y[3]);
                    let ah = [a0, a1, a2, a3];
                    let al = [c0, c1, c2, c3];
                    kcommon::unroll!(nt in 2 => {
                        mma_tf32(&mut acc[h][nt], al, bh[nt]);
                        mma_tf32(&mut acc[h][nt], ah, bl[nt]);
                        mma_tf32(&mut acc[h][nt], ah, bh[nt]);
                    });
                } else {
                    let ah = [tf32(y[0]), tf32(y[1]), tf32(y[2]), tf32(y[3])];
                    kcommon::unroll!(nt in 2 => {
                        mma_tf32(&mut acc[h][nt], ah, bh[nt]);
                    });
                }
            });
        });
        // ---- relu-sum over heads, store the visible blocks ----
        kcommon::unroll!(nt in 2 => {
            kcommon::unroll!(e in 4 => {
                let mut tot = 0.0f32;
                kcommon::unroll!(h in 4 => { tot += math::maxf(acc[h][nt][e], 0.0); });
                let b = b0 + w * 16 + nt as u32 * 8 + 2 * tq + (e as u32 & 1);
                let hi = e >= 2;
                let (ok, nv, r) = if hi { (ok_b, nv_b, r0 + g + 8) } else { (ok_a, nv_a, r0 + g) };
                if ok && b < nv && b < b_hi {
                    *score.add((r * ld + b) as usize) = tot * scale;
                }
            });
        });
        b0 += BN;
    }
}
