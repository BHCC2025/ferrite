//! Prompt-path GEMM: `y[yrow(i)][n] (= or +=) sum_k x[xrow(i)][k] * W[n][k]` with bf16 activations, the weight
//! tile dequantized to bf16 in shared memory (the exact `dec` values, rounded to bf16) and `mma.sync` m16n8k16
//! bf16 -> f32. Tile 64 (rows) x 64 (n) x 32 (k), 4 warps (2 x 2, 32 x 32 each), two shared-memory stages.
//!
//! Segments (MoE): grid.z = segment s with rows `seg_off[s] .. seg_off[s+1]` of the index arrays and weights at
//! `segw[s]` (0 = skip); `xidx` / `yidx` map those rows to activation / output rows (null = identity). Dense:
//! `seg_off` null, one segment of `m` rows, weights at `w`.
use crate::dec::*;
use kcommon::mma::*;
use kcommon::smem::*;
use kcommon::*;

const BM: u32 = 64;
const BN: u32 = 64;
const BK: u32 = 32;
const LDS: u32 = 40; // bf16 elements per smem row (32 + 8 pad: conflict-free ldmatrix)
const TILE: u32 = 64 * LDS * 2; // bytes of one A or B tile

pub trait Dq8 {
    const QK: u32;
    const BYTES: u32;
    unsafe fn dec(b: *const u8, c: u32) -> [f32; 8];
}
macro_rules! dq8 {
    ($t:ident, $f:ident, $qk:expr, $by:expr) => {
        pub struct $t;
        impl Dq8 for $t {
            const QK: u32 = $qk;
            const BYTES: u32 = $by;
            #[inline(always)]
            unsafe fn dec(b: *const u8, c: u32) -> [f32; 8] {
                $f(b, c)
            }
        }
    };
}
dq8!(GQ8_0, dec_q8_0, 32, 34);
dq8!(GQ2_0, dec_q2_0, 64, 18);
dq8!(GIq4Nl, dec_iq4_nl, 32, 18);
dq8!(GIq4Xs, dec_iq4_xs, 256, 136);
dq8!(GQ4K, dec_q4_k, 256, 144);
dq8!(GQ5K, dec_q5_k, 256, 176);
dq8!(GQ6K, dec_q6_k, 256, 210);
dq8!(GIq2S, dec_iq2_s, 256, 82);
dq8!(GIq3Xxs, dec_iq3_xxs, 256, 98);
dq8!(GIq3S, dec_iq3_s, 256, 110);
dq8!(GIq2Xxs, dec_iq2_xxs, 256, 66);
dq8!(GIq1M, dec_iq1_m, 256, 56);
dq8!(GIq2Xs, dec_iq2_xs, 256, 74);
dq8!(GMxfp4R, dec_mxfp4r, 32, 18);
dq8!(GBf16, dec_bf16, 8, 16);
dq8!(GF16, dec_f16, 8, 16);
dq8!(GF32, dec_f32, 8, 32);

#[inline(always)]
unsafe fn load_tiles<D: Dq8>(sa: Shared<u16>, sb: Shared<u16>, k0: u32, xr: &[*const u16; 2], xok: &[bool; 2],
    wrow: &[*const u8; 2]) {
    let t = tid_x();
    // A: 64 rows x 4 chunks of 8 bf16 (16 B), 2 per thread
    for i in 0..2 {
        let idx = t + 128 * i as u32;
        let (r, c) = (idx / 4, idx % 4);
        let dst = sa.add(r * LDS + c * 8).addr();
        let src = xr[i].add((k0 + c * 8) as usize) as *const u8;
        cp_async_16_zfill(dst, src, if xok[i] { 16 } else { 0 });
    }
    // B: 64 weight rows x 4 chunks of 8 values, decoded
    for i in 0..2 {
        let idx = t + 128 * i as u32;
        let (r, c) = (idx / 4, idx % 4);
        let kv = k0 + c * 8;
        let blk = wrow[i].add(((kv / D::QK) * D::BYTES) as usize);
        let y = D::dec(blk, (kv % D::QK) / 8);
        let v = [bf16::pack2(y[0], y[1]), bf16::pack2(y[2], y[3]), bf16::pack2(y[4], y[5]), bf16::pack2(y[6], y[7])];
        sb.add(r * LDS + c * 8).cast::<[u32; 4]>().st(v);
    }
    cp_async_commit();
}

#[inline(always)]
pub unsafe fn gemm<D: Dq8>(y: *mut f32, ldy: u32, x: *const u16, ldx: u32, w: *const u8, row_bytes: u64, m: u32,
    k: u32, xidx: *const u32, yidx: *const u32, segw: *const u64, seg_off: *const u32, add: u32) {
    let (r0, r1, wb) = if seg_off.is_null() {
        (0u32, m, w)
    } else {
        let s = bid_z() as usize;
        let wp = *segw.add(s);
        if wp == 0 {
            return;
        }
        (*seg_off.add(s), *seg_off.add(s + 1), wp as *const u8)
    };
    let mt = r0 + bid_y() * BM;
    if mt >= r1 {
        return;
    }
    let n0 = bid_x() * BN;
    let t = tid_x();
    // per-thread load rows (A: activation rows, B: weight rows)
    let mut xr = [x; 2];
    let mut xok = [false; 2];
    let mut wrow = [wb; 2];
    for i in 0..2 {
        let idx = t + 128 * i as u32;
        let gi = mt + idx / 4;
        xok[i] = gi < r1;
        let xrow = if !xok[i] { 0 } else if xidx.is_null() { gi } else { *xidx.add(gi as usize) };
        xr[i] = x.add(xrow as usize * ldx as usize);
        wrow[i] = wb.add((n0 + idx / 4) as usize * row_bytes as usize);
    }
    let base = smem::dyn_base::<u16>();
    let sa = [base, base.byte_add(TILE)];
    let sb = [base.byte_add(2 * TILE), base.byte_add(3 * TILE)];
    let wm = (warp_id() / 2) * 32;
    let wn = (warp_id() % 2) * 32;
    let l = lane();
    let mut acc = [[[0f32; 4]; 4]; 2];
    let nk = k / BK;
    load_tiles::<D>(sa[0], sb[0], 0, &xr, &xok, &wrow);
    for kt in 0..nk {
        let cur = (kt & 1) as usize;
        if kt + 1 < nk {
            load_tiles::<D>(sa[cur ^ 1], sb[cur ^ 1], (kt + 1) * BK, &xr, &xok, &wrow);
            cp_async_wait::<1>();
        } else {
            cp_async_wait::<0>();
        }
        sync_threads();
        for kk in 0..2u32 {
            let mut af = [[0u32; 4]; 2];
            for mi in 0..2u32 {
                let a = sa[cur].add((wm + mi * 16 + l % 16) * LDS + kk * 16 + (l / 16) * 8);
                af[mi as usize] = ldmatrix_x4(a.addr());
            }
            for ni in 0..4u32 {
                let b = sb[cur].add((wn + ni * 8 + l % 8) * LDS + kk * 16 + ((l / 8) % 2) * 8);
                let bf = ldmatrix_x2(b.addr());
                for mi in 0..2 {
                    mma_bf16_16816(&mut acc[mi][ni as usize], af[mi], bf);
                }
            }
        }
        sync_threads();
    }
    let g = l / 4;
    let tq = l % 4;
    for mi in 0..2u32 {
        for half in 0..2u32 {
            let li = wm + mi * 16 + g + 8 * half;
            let gi = mt + li;
            if gi >= r1 {
                continue;
            }
            let yrow = if yidx.is_null() { gi } else { *yidx.add(gi as usize) };
            let yp = y.add(yrow as usize * ldy as usize);
            for ni in 0..4u32 {
                let col = n0 + wn + ni * 8 + 2 * tq;
                let v0 = acc[mi as usize][ni as usize][2 * half as usize];
                let v1 = acc[mi as usize][ni as usize][2 * half as usize + 1];
                let o = yp.add(col as usize);
                if add != 0 {
                    *o += v0;
                    *o.add(1) += v1;
                } else {
                    *o = v0;
                    *o.add(1) = v1;
                }
            }
        }
    }
}

macro_rules! gemm_kernel {
    ($d:ty, $name:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(y: *mut f32, ldy: u32, x: *const u16, ldx: u32, w: *const u8,
            row_bytes: u64, m: u32, k: u32, xidx: *const u32, yidx: *const u32, segw: *const u64, seg_off: *const u32,
            add: u32) {
            gemm::<$d>(y, ldy, x, ldx, w, row_bytes, m, k, xidx, yidx, segw, seg_off, add)
        }
    };
}
gemm_kernel!(GQ8_0, ggq_gemm_q8_0);
gemm_kernel!(GQ2_0, ggq_gemm_q2_0);
gemm_kernel!(GIq4Nl, ggq_gemm_iq4_nl);
gemm_kernel!(GIq4Xs, ggq_gemm_iq4_xs);
gemm_kernel!(GQ4K, ggq_gemm_q4_k);
gemm_kernel!(GQ5K, ggq_gemm_q5_k);
gemm_kernel!(GQ6K, ggq_gemm_q6_k);
gemm_kernel!(GIq2S, ggq_gemm_iq2_s);
gemm_kernel!(GIq3Xxs, ggq_gemm_iq3_xxs);
gemm_kernel!(GIq3S, ggq_gemm_iq3_s);
gemm_kernel!(GIq2Xxs, ggq_gemm_iq2_xxs);
gemm_kernel!(GIq1M, ggq_gemm_iq1_m);
gemm_kernel!(GIq2Xs, ggq_gemm_iq2_xs);
gemm_kernel!(GMxfp4R, ggq_gemm_mxfp4r);
gemm_kernel!(GBf16, ggq_gemm_bf16);
gemm_kernel!(GF16, ggq_gemm_f16);
gemm_kernel!(GF32, ggq_gemm_f32);

/// f32 -> bf16 rows (activation conversion for the GEMM). Grid: ceil(n / 256), y = row.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_f32_to_bf16(dst: *mut u16, ldd: u32, src: *const f32, lds: u32, n: u32) {
    let i = bid_x() * bdim_x() + tid_x();
    if i < n {
        let r = bid_y() as usize;
        *dst.add(r * ldd as usize + i as usize) = bf16::from_f32(*src.add(r * lds as usize + i as usize));
    }
}

/// Tensor-core peak probe: each warp runs `iters` x 8 independent mma (bf16 m16n8k16 when `int8 == 0`, s8
/// m16n8k32 otherwise) on register data. Bench only.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_mma_peak(out: *mut f32, iters: u32, int8: u32) {
    let a = [tid_x(), tid_x() ^ 3, tid_x() + 7, tid_x() * 3];
    let b = [tid_x() + 1, tid_x() ^ 5];
    let mut acc = [[0f32; 4]; 8];
    let mut iacc = [[0i32; 4]; 8];
    if int8 == 0 {
        for _ in 0..iters {
            unroll!(i in 8 => { mma_bf16_16816(&mut acc[i], a, b); });
        }
    } else {
        for _ in 0..iters {
            unroll!(i in 8 => { mma_s8_16832(&mut iacc[i], a, b); });
        }
    }
    let mut s = 0f32;
    unroll!(i in 8 => { s += acc[i][0] + acc[i][3] + iacc[i][0] as f32 + iacc[i][3] as f32; });
    *out.add((bid_x() * bdim_x() + tid_x()) as usize) = s;
}

/// D = A * B + D, s8 inputs (A 16x32 row, B 32x8 col), s32 accumulators. Fragments as m16n8k16 bf16 with 4-byte
/// k-pairs replaced by 4 int8 k values.
#[inline(always)]
pub fn mma_s8_16832(d: &mut [i32; 4], a: [u32; 4], b: [u32; 2]) {
    unsafe {
        core::arch::asm!("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], options(nomem, nostack));
    }
}
