//! Dense prompt GEMM for BF16 weights: `y[i][n] (= or +=) sum_k x[i][k] * W[n][k]`, x bf16 [m][ldx], W bf16 rows
//! `row_bytes` apart, y f32 [m][ldy]. Both tiles by 16-byte cp.async (no weight decode), 4 shared-memory stages,
//! tile 128 x BN x 32, 8 warps. Every output takes the same m16n8k16 bf16 mma steps in the same k order as
//! `ggq_gemm_bf16`, so the result is bit-identical to it.
use kcommon::mma::*;
use kcommon::smem::*;
use kcommon::*;

const BM: u32 = 128;
const BK: u32 = 32;
const LDS: u32 = BK + 8; // bf16 per smem row (80 B: conflict-free ldmatrix)
const NST: u32 = 4;

#[inline(always)]
unsafe fn gemm16<const BN: u32>(y: *mut f32, ldy: u32, x: *const u16, ldx: u32, w: *const u8, row_bytes: u64, m: u32,
    n: u32, k: u32, add: u32) {
    // warps: BN 128 -> 2 x 4 (warp 64 x 32); BN 64 -> 4 x 2 (warp 32 x 32)
    const fn wcols(bn: u32) -> u32 { if bn == 128 { 4 } else { 2 } }
    let wc = wcols(BN);
    let wr = 8 / wc;
    let wmr = BM / wr; // rows per warp: 64 or 32
    let mtiles = wmr / 16;
    let ta = BM * LDS * 2;
    let tb = BN * LDS * 2;
    let stage = ta + tb;
    let r0 = bid_y() * BM;
    let n0 = bid_x() * BN;
    let t = tid_x();
    let sa = dyn_base::<u8>().addr();
    let nk = k / BK;
    let issue = |kt: u32| {
        if kt < nk {
            let so = sa + (kt % NST) * stage;
            let k0 = kt * BK;
            // A: 128 rows x 4 chunks
            unroll!(i in 2 => {
                let c = t + 256 * i as u32;
                let (r, ch) = (c / 4, c % 4);
                let gr = r0 + r;
                let ok = gr < m;
                let src = x.add((if ok { gr } else { 0 }) as usize * ldx as usize + (k0 + ch * 8) as usize) as *const u8;
                cp_async_16_zfill(so + (r * LDS + ch * 8) * 2, src, if ok { 16 } else { 0 });
            });
            // B: BN rows x 4 chunks
            let nb = BN * 4 / 256;
            let mut i = 0;
            while i < nb {
                let c = t + 256 * i;
                let (r, ch) = (c / 4, c % 4);
                let gn = n0 + r;
                let ok = gn < n;
                let src = w.add((if ok { gn } else { 0 }) as usize * row_bytes as usize + ((k0 + ch * 8) * 2) as usize);
                cp_async_16_zfill(so + ta + (r * LDS + ch * 8) * 2, src, if ok { 16 } else { 0 });
                i += 1;
            }
        }
        cp_async_commit();
    };
    unroll!(p in 3 => { issue(p as u32); });
    let wi = warp_id();
    let wm = (wi / wc) * wmr;
    let wn = (wi % wc) * 32;
    let l = lane();
    let mut acc = [[[0f32; 4]; 4]; 4];
    for kt in 0..nk {
        cp_async_wait::<2>();
        sync_threads();
        issue(kt + 3);
        let so = sa + (kt % NST) * stage;
        unroll!(kk in 2 => {
            let mut af = [[0u32; 4]; 4];
            unroll!(mi in 4 => {
                if (mi as u32) < mtiles {
                    af[mi] = ldmatrix_x4(so + ((wm + mi as u32 * 16 + l % 16) * LDS + kk as u32 * 16 + (l / 16) * 8) * 2);
                }
            });
            unroll!(np in 2 => {
                // two n8 tiles: lanes 0-7 rows n 0..8 k lo, 8-15 k hi, 16-23 n 8..16 k lo, 24-31 k hi
                let b = ldmatrix_x4(so + ta + ((wn + np as u32 * 16 + (l / 16) * 8 + l % 8) * LDS + kk as u32 * 16
                    + ((l / 8) % 2) * 8) * 2);
                unroll!(mi in 4 => {
                    if (mi as u32) < mtiles {
                        mma_bf16_16816(&mut acc[mi][2 * np], af[mi], [b[0], b[1]]);
                        mma_bf16_16816(&mut acc[mi][2 * np + 1], af[mi], [b[2], b[3]]);
                    }
                });
            });
        });
    }
    cp_async_wait_all();
    let g = l / 4;
    let tq = l % 4;
    let add = add & 1;
    unroll!(mi in 4 => {
        if (mi as u32) < mtiles {
            unroll!(h in 2 => {
                let gi = r0 + wm + mi as u32 * 16 + g + 8 * h as u32;
                if gi < m {
                    let yp = y.add(gi as usize * ldy as usize);
                    unroll!(ni in 4 => {
                        let col = n0 + wn + ni as u32 * 8 + 2 * tq;
                        if col < n {
                            let o = yp.add(col as usize);
                            let (v0, v1) = (acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1]);
                            if add != 0 {
                                *o += v0;
                                *o.add(1) += v1;
                            } else {
                                *o = v0;
                                *o.add(1) = v1;
                            }
                        }
                    });
                }
            });
        }
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_gemm16_bf16_n128(y: *mut f32, ldy: u32, x: *const u16, ldx: u32, w: *const u8,
    row_bytes: u64, m: u32, n: u32, k: u32, add: u32) {
    gemm16::<128>(y, ldy, x, ldx, w, row_bytes, m, n, k, add)
}
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_gemm16_bf16_n64(y: *mut f32, ldy: u32, x: *const u16, ldx: u32, w: *const u8,
    row_bytes: u64, m: u32, n: u32, k: u32, add: u32) {
    gemm16::<64>(y, ldy, x, ldx, w, row_bytes, m, n, k, add)
}
