//! Small-M (<= 16 rows) FP8-weight GEMM with the arithmetic of `ink_gemm_s8` (and so of "dequantise to bf16, then
//! core's bf16 GEMM"): effective weight bf16(q * scale[n]) formed in shared memory, x is the mma A operand (m16), weights the
//! n8 side, k16 steps in order into one f32 accumulator, bf16 out. Parametrised over pipeline depth and CTA width so the
//! memory-level parallelism can be raised (the reference kernel keeps ~6 KB in flight per CTA).
use super::*;
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};


#[inline(always)]
fn cvt4(w: u32) -> [f32; 4] {
    e4m3x4(w)
}

macro_rules! s8_gemm {
    ($name:ident, $smem:ident, $st:expr, $wait:expr, $nw:expr, $u:expr) => {
        /// grid = (ceil(N / BN), ceil(M / 16)), BN = 8 * NW, block = 32 * NW, smem = the matching constant. K % (64 * U) == 0.
        pub const $smem: u32 = {
            let bn = 8 * $nw;
            $st * (bn * (64 * $u + 16) + 16 * (128 * $u + 16)) + 2 * bn * (128 * $u + 16)
        };
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(c: *mut u16, a: *const u16, w8: *const u8, scale: *const f32, m: u32, n: u32,
            k: u32, lda: u32, ldc: u32) {
            const ST: u32 = $st;
            const U: u32 = $u;
            const BN: u32 = 8 * $nw;
            const BK: u32 = 64 * U;
            const RAWP: u32 = BK + 16;
            const PITCH: u32 = 2 * BK + 16;
            const RAW_STAGE: u32 = BN * RAWP;
            const A_STAGE: u32 = 16 * PITCH;
            const BF_TILE: u32 = BN * PITCH;
            let smem: Shared<u8> = dyn_base();
            let n0 = bid_x() * BN;
            let m0 = bid_y() * 16;
            let wn = warp_id();
            let l = lane();
            let t = tid_x();
            let ktiles = k / BK;
            let bf_base = smem.byte_add(ST * (RAW_STAGE + A_STAGE));
            let mut acc = [0.0f32; 4];
            let (r8, c8) = (t >> 2, t & 3);
            let sc = if n0 + r8 < n { *scale.add((n0 + r8) as usize) } else { 0.0 };
            macro_rules! stage_w {
                ($s:expr, $kt:expr) => {{
                    let st = smem.byte_add($s * (RAW_STAGE + A_STAGE));
                    let gr = n0 + r8;
                    kcommon::unroll!(u in 4 => {
                        if (u as u32) < U {
                            let d = st.byte_add(r8 * RAWP + u as u32 * 64 + c8 * 16).addr();
                            if gr < n {
                                cp_async_16(d, w8.add(gr as usize * k as usize + ($kt * BK + u as u32 * 64 + c8 * 16) as usize));
                            } else {
                                cp_async_16_zfill(d, w8, 0);
                            }
                        }
                    });
                }};
            }
            macro_rules! stage_a {
                ($s:expr, $kt:expr) => {{
                    let st = smem.byte_add($s * (RAW_STAGE + A_STAGE));
                    // 16 rows x (BK bf16 = 16 U chunks of 16 B): 16 * 8 * U chunks over the block
                    let mut i = t;
                    while i < 16 * 8 * U {
                        let r = i / (8 * U);
                        let cc = i % (8 * U);
                        let d = st.byte_add(RAW_STAGE + r * PITCH + cc * 16).addr();
                        if m0 + r < m {
                            cp_async_16(d, a.add((m0 + r) as usize * lda as usize + ($kt * BK + cc * 8) as usize) as *const u8);
                        } else {
                            cp_async_16_zfill(d, a as *const u8, 0);
                        }
                        i += 32 * $nw;
                    }
                }};
            }
            let mut s = 0;
            while s < ST - 1 {
                if s < ktiles {
                    stage_w!(s, s);
                }
                s += 1;
            }
            cp_async_commit();
            pdl_wait();
            pdl_trigger();
            s = 0;
            while s < ST - 1 {
                if s < ktiles {
                    stage_a!(s, s);
                }
                cp_async_commit();
                s += 1;
            }
            let mut kt = 0;
            while kt < ktiles {
                cp_async_wait::<$wait>();
                sync_threads();
                let pf = kt + ST - 1;
                if pf < ktiles {
                    stage_w!(pf % ST, pf);
                    stage_a!(pf % ST, pf);
                }
                cp_async_commit();
                let st = smem.byte_add((kt % ST) * (RAW_STAGE + A_STAGE));
                let raw = st;
                let asm = st.byte_add(RAW_STAGE);
                let wb = bf_base.byte_add((kt & 1) * BF_TILE);
                kcommon::unroll!(u in 4 => {
                    if (u as u32) < U {
                        let rw: [u32; 4] = <[u32; 4] as kcommon::smem::SmemIo>::lds(raw.byte_add(r8 * RAWP + u as u32 * 64 + c8 * 16).addr());
                        let mut o = [0u32; 8];
                        kcommon::unroll!(i in 4 => {
                            let v = e4m3x4(rw[i]);
                            o[2 * i] = bf16::pack2(v[0] * sc, v[1] * sc);
                            o[2 * i + 1] = bf16::pack2(v[2] * sc, v[3] * sc);
                        });
                        <[u32; 4] as kcommon::smem::SmemIo>::sts(wb.byte_add(r8 * PITCH + u as u32 * 128 + c8 * 32).addr(), [o[0], o[1], o[2], o[3]]);
                        <[u32; 4] as kcommon::smem::SmemIo>::sts(wb.byte_add(r8 * PITCH + u as u32 * 128 + c8 * 32 + 16).addr(), [o[4], o[5], o[6], o[7]]);
                    }
                });
                sync_threads();
                kcommon::unroll!(ks in 16 => {
                    if (ks as u32) < 4 * U {
                        let kc2 = ks as u32 * 2;
                        let af = ldmatrix_x4(asm.byte_add((l & 15) * PITCH + (kc2 + (l >> 4)) * 16).addr());
                        let bfr = ldmatrix_x2(wb.byte_add((wn * 8 + (l & 7)) * PITCH + (kc2 + ((l >> 3) & 1)) * 16).addr());
                        mma_bf16_16816(&mut acc, af, bfr);
                    }
                });
                kt += 1;
            }
            cp_async_wait_all();
            let g = l >> 2;
            let t4 = l & 3;
            kcommon::unroll!(half in 2 => {
                let row = m0 + g + half as u32 * 8;
                let col = n0 + wn * 8 + 2 * t4;
                if row < m && col < n {
                    *(c.add(row as usize * ldc as usize + col as usize) as *mut u32) =
                        bf16::pack2(acc[half * 2], acc[half * 2 + 1]);
                }
            });
        }
    };
}

s8_gemm!(ink_s8_u1_st4_w4, S8_SMEM_U1_ST4_W4, 4, 2, 4, 1);
s8_gemm!(ink_s8_u4_st3_w4, S8_SMEM_U4_ST3_W4, 3, 1, 4, 4);
s8_gemm!(ink_s8_u4_st3_w8, S8_SMEM_U4_ST3_W8, 3, 1, 8, 4);
s8_gemm!(ink_s8_u2_st4_w4, S8_SMEM_U2_ST4_W4, 4, 2, 4, 2);
s8_gemm!(ink_s8_u3_st3_w4, S8_SMEM_U3_ST3_W4, 3, 1, 4, 3);
