//! bf16 GEMM on tensor cores: C[M][N] = A[M][K] . W[N][K]^T (+ residual), f32 accumulate, bf16 out.
//!
//! Tile 128x128x32, 8 warps (2 along M x 4 along N, warp tile 64x32), 3-stage cp.async pipeline in
//! 48 KB of dynamic shared memory, XOR-swizzled 16-byte chunks so ldmatrix is conflict-free.
//! Requirements: K % 32 == 0, lda/ldc in elements, rows 16-byte aligned. M and N edges are zero-filled.
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

const BM: u32 = 128;
const BN: u32 = 128;
const BK: u32 = 32;
const STAGES: u32 = 3;
const TILE_BYTES: u32 = BM * BK * 2; // 8 KB (A and W tiles are the same size)
pub const GEMM_SMEM: u32 = STAGES * 2 * TILE_BYTES; // 48 KB

/// Byte offset of 16-byte chunk `c` (0..4) of row `r` inside a [rows][32] bf16 tile, swizzled.
#[inline(always)]
fn swz(r: u32, c: u32) -> u32 {
    r * 64 + ((c ^ ((r >> 1) & 3)) << 4)
}

#[inline(always)]
unsafe fn load_tile(dst: Shared<u8>, src: *const u16, ld: u32, row0: u32, rows: u32, k0: u32) {
    // 128 rows x 4 chunks = 512 chunks over 256 threads.
    let t = tid_x();
    let mut i = t;
    while i < 512 {
        let r = i >> 2;
        let c = i & 3;
        let gr = row0 + r;
        let d = dst.byte_add(swz(r, c)).addr();
        if gr < rows {
            cp_async_16(d, src.add((gr as usize) * ld as usize + (k0 + c * 8) as usize) as *const u8);
        } else {
            cp_async_16_zfill(d, src as *const u8, 0);
        }
        i += 256;
    }
}

/// The block's 128x128 tile of A . W^T: this warp's 64x32 accumulators (4 m16 x 4 n8 fragments).
/// grid = (ceil(M/128), ceil(N/128)): M tiles run fastest, so the blocks resident at once share a few weight
/// tiles (each weight byte comes from DRAM about once).
#[inline(always)]
unsafe fn mainloop(a: *const u16, w: *const u16, m: u32, n: u32, k: u32, lda: u32, m0: u32, n0: u32)
    -> [[[f32; 4]; 4]; 4] {
    let smem: Shared<u8> = dyn_base();
    let wid = warp_id();
    let l = lane();
    let wm = wid >> 2; // 0..2
    let wn = wid & 3; // 0..4
    let ktiles = k / BK;

    let mut acc = [[[0.0f32; 4]; 4]; 4];

    // prologue
    let mut s = 0;
    while s < STAGES - 1 {
        if s < ktiles {
            let st = smem.byte_add(s * 2 * TILE_BYTES);
            load_tile(st, a, lda, m0, m, s * BK);
            load_tile(st.byte_add(TILE_BYTES), w, k, n0, n, s * BK);
        }
        cp_async_commit();
        s += 1;
    }

    let mut kt = 0;
    while kt < ktiles {
        cp_async_wait::<{ STAGES - 2 }>();
        sync_threads();
        // prefetch tile kt + STAGES - 1 into the stage consumed at kt - 1
        let pf = kt + STAGES - 1;
        if pf < ktiles {
            let st = smem.byte_add((pf % STAGES) * 2 * TILE_BYTES);
            load_tile(st, a, lda, m0, m, pf * BK);
            load_tile(st.byte_add(TILE_BYTES), w, k, n0, n, pf * BK);
        }
        cp_async_commit();

        let sa = smem.byte_add((kt % STAGES) * 2 * TILE_BYTES);
        let sw = sa.byte_add(TILE_BYTES);
        kcommon::unroll!(ks in 2 => {
            let kc2 = ks as u32 * 2;
            let mut af = [[0u32; 4]; 4];
            kcommon::unroll!(mt in 4 => {
                let r = wm * 64 + mt as u32 * 16 + (l & 15);
                af[mt] = ldmatrix_x4(sa.byte_add(swz(r, kc2 + (l >> 4))).addr());
            });
            let mut bf = [[0u32; 2]; 4];
            kcommon::unroll!(p in 2 => {
                let r = wn * 32 + (2 * p as u32 + (l >> 4)) * 8 + (l & 7);
                let v = ldmatrix_x4(sw.byte_add(swz(r, kc2 + ((l >> 3) & 1))).addr());
                bf[2 * p] = [v[0], v[1]];
                bf[2 * p + 1] = [v[2], v[3]];
            });
            kcommon::unroll!(mt in 4 => {
                kcommon::unroll!(nt in 4 => {
                    mma_bf16_16816(&mut acc[mt][nt], af[mt], bf[nt]);
                });
            });
        });
        kt += 1;
    }
    cp_async_wait_all();
    acc
}

/// grid = (ceil(M/128), ceil(N/128)), block = 256, smem = GEMM_SMEM.
/// `res` may be null; when set, C = bf16(bf16(acc) + res) (the HF residual add), `res` has stride ldc.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_bf16(c: *mut u16, a: *const u16, w: *const u16, res: *const u16,
    m: u32, n: u32, k: u32, lda: u32, ldc: u32) {
    let m0 = bid_x() * BM;
    let n0 = bid_y() * BN;
    let acc = mainloop(a, w, m, n, k, lda, m0, n0);
    let l = lane();
    let wm = warp_id() >> 2;
    let wn = warp_id() & 3;
    // epilogue
    let g = l >> 2;
    let t4 = l & 3;
    kcommon::unroll!(mt in 4 => {
        kcommon::unroll!(half in 2 => {
            let row = m0 + wm * 64 + mt as u32 * 16 + g + half as u32 * 8;
            if row < m {
                kcommon::unroll!(nt in 4 => {
                    let col = n0 + wn * 32 + nt as u32 * 8 + 2 * t4;
                    if col < n {
                        let (mut x0, mut x1) = (acc[mt][nt][half * 2], acc[mt][nt][half * 2 + 1]);
                        let off = row as usize * ldc as usize + col as usize;
                        if !res.is_null() {
                            let r = *(res.add(off) as *const u32);
                            x0 = bf16::round(x0) + bf16::lo(r);
                            x1 = bf16::round(x1) + bf16::hi(r);
                        }
                        *(c.add(off) as *mut u32) = bf16::pack2(x0, x1);
                    }
                });
            }
        });
    });
}

/// Small-M bf16 GEMV: y[r][n] = x[r][:] . W[n][:] for r < m <= 8, one warp per output column.
/// grid = ceil(N/8), block = 256. K % 256 == 0. `res` as in fe_gemm_bf16 (stride ldc).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemv_bf16(c: *mut u16, a: *const u16, w: *const u16, res: *const u16, m: u32,
    n: u32, k: u32, lda: u32, ldc: u32) {
    let col = bid_x() * 8 + warp_id();
    if col >= n {
        return;
    }
    let l = lane();
    let wr = w.add(col as usize * k as usize) as *const u32;
    let mut acc = [0.0f32; 8];
    let mut i = l;
    while i < k / 8 {
        let wv = bf16::unpack8(ldg_stream_v4(wr.add(i as usize * 4)));
        kcommon::unroll!(r in 8 => {
            if (r as u32) < m {
                let xv = bf16::unpack8(ldg_nc_v4(a.add(r * lda as usize + i as usize * 8) as *const u32));
                let mut d = acc[r];
                kcommon::unroll!(j in 8 => {
                    d = math::fma(wv[j], xv[j], d);
                });
                acc[r] = d;
            }
        });
        i += 32;
    }
    kcommon::unroll!(r in 8 => {
        if (r as u32) < m {
            let v = warp::sum(acc[r]);
            if l == 0 {
                let off = r * ldc as usize + col as usize;
                let mut y = v;
                if !res.is_null() {
                    y = bf16::round(y) + bf16::to_f32(*res.add(off));
                }
                *c.add(off) = bf16::from_f32(y);
            }
        }
    });
}

/// Small-M bf16 GEMV for narrow projections (GDN in_proj_a|b: 48 / 96 columns): y[r][n] = x[r][:] . W[n][:] for
/// r < m <= 8, one block (8 warps) per output column, so the grid fills the GPU where fe_gemv_bf16 (one warp a
/// column) runs 6 blocks. Warp w sums its contiguous K range k/8 * [w, w+1) (lanes strided by 8 elements), the
/// warp's lanes meet in a butterfly, then warp 0 adds the 8 warps in order: a fixed order for every row, so a
/// row's bits do not depend on m. grid = N, block = 256. K % 64 == 0. `res` as in fe_gemv_bf16.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemv_bf16_kw(c: *mut u16, a: *const u16, w: *const u16, res: *const u16, m: u32,
    n: u32, k: u32, lda: u32, ldc: u32) {
    let col = bid_x();
    if col >= n {
        return;
    }
    let red = static_shared!(gemv_kw_red, f32, 256); // [8 warps][8 rows] floats (the size is in bytes)
    let wid = warp_id();
    let l = lane();
    let span = k / 8; // this warp's K range, in elements (a multiple of 8)
    let k0 = wid * span;
    let wr = w.add(col as usize * k as usize + k0 as usize) as *const u32;
    let mut acc = [0.0f32; 8];
    let mut i = l;
    while i < span / 8 {
        let wv = bf16::unpack8(ldg_stream_v4(wr.add(i as usize * 4)));
        kcommon::unroll!(r in 8 => {
            if (r as u32) < m {
                let xv = bf16::unpack8(ldg_nc_v4(a.add(r * lda as usize + (k0 + i * 8) as usize) as *const u32));
                let mut d = acc[r];
                kcommon::unroll!(j in 8 => {
                    d = math::fma(wv[j], xv[j], d);
                });
                acc[r] = d;
            }
        });
        i += 32;
    }
    kcommon::unroll!(r in 8 => {
        if (r as u32) < m {
            let v = warp::sum(acc[r]);
            if l == 0 {
                red.set(wid * 8 + r as u32, v);
            }
        }
    });
    sync_threads();
    if wid == 0 && l < m {
        let mut y = red.at(l);
        let mut q = 1;
        while q < 8 {
            y += red.at(q * 8 + l);
            q += 1;
        }
        let off = l as usize * ldc as usize + col as usize;
        if !res.is_null() {
            y = bf16::round(y) + bf16::to_f32(*res.add(off));
        }
        *c.add(off) = bf16::from_f32(y);
    }
}

/// gate_up GEMM with the SwiGLU fused into the epilogue. W rows are interleaved in blocks of 8
/// ([gate 0..8, up 0..8, gate 8..16, up 8..16, ...], N = 2 * inter), so each warp's n8 fragment pairs hold
/// matching gate and up columns: act[r][c] = bf16(bf16(silu(bf16(gate))) * bf16(up)), act stride `inter`.
/// grid = (ceil(M/128), N/128), block = 256, smem = GEMM_SMEM.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_swiglu(act: *mut u16, a: *const u16, w: *const u16, m: u32, n: u32, k: u32,
    lda: u32) {
    let m0 = bid_x() * BM;
    let n0 = bid_y() * BN;
    let acc = mainloop(a, w, m, n, k, lda, m0, n0);
    let l = lane();
    let wm = warp_id() >> 2;
    let wn = warp_id() & 3;
    let g = l >> 2;
    let t4 = l & 3;
    let inter = n / 2;
    kcommon::unroll!(mt in 4 => {
        kcommon::unroll!(half in 2 => {
            let row = m0 + wm * 64 + mt as u32 * 16 + g + half as u32 * 8;
            if row < m {
                kcommon::unroll!(p in 2 => {
                    let blk = (n0 + wn * 32) / 16 + p as u32;
                    let col = blk * 8 + 2 * t4;
                    let gt = &acc[mt][2 * p];
                    let up = &acc[mt][2 * p + 1];
                    let s0 = bf16::round(math::silu(bf16::round(gt[half * 2])));
                    let s1 = bf16::round(math::silu(bf16::round(gt[half * 2 + 1])));
                    let y0 = s0 * bf16::round(up[half * 2]);
                    let y1 = s1 * bf16::round(up[half * 2 + 1]);
                    *(act.add(row as usize * inter as usize + col as usize) as *mut u32) = bf16::pack2(y0, y1);
                });
            }
        });
    });
}

/// Interleave gate/up weight rows in blocks of 8 for fe_gemm_swiglu: dst row 16b + i = gate row 8b + i (i < 8),
/// up row 8b + i - 8 (i >= 8). grid = 2 * inter, block = 256; k % 8 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_interleave_gu(dst: *mut u16, gate: *const u16, up: *const u16, k: u32) {
    let r = bid_x();
    let b = r / 16;
    let i = r % 16;
    let src = if i < 8 { gate.add(((b * 8 + i) * k) as usize) } else { up.add(((b * 8 + i - 8) * k) as usize) };
    let d = dst.add((r * k) as usize);
    let mut j = tid_x();
    while j < k / 8 {
        stg_v4(d.add(j as usize * 8) as *mut u32, ldg_nc_v4(src.add(j as usize * 8) as *const u32));
        j += bdim_x();
    }
}
