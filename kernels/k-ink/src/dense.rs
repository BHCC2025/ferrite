//! FP8 dense projections (P24 format): weights E4M3 with one f32 scale per output row, bf16 activations.
//! The effective weight is the reference emulation's: w = bf16(q * scale[n]). Decode / verify rows (<= 16) run the
//! GEMV below (fixed per-row summation order: a row's bits do not depend on how many rows the call has), prompt rows
//! dequantize the layer to bf16 scratch (`ink_dequant_fp8`) and take core's bf16 tensor-core GEMM.
use super::*;

/// BF16 weights [n][k] -> FP8 codes [n][k] + scale[n] = max(amax_row, 1e-30) / 448 (f32); q = e4m3(w / scale) (RNE,
/// saturating). grid = n, block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_fp8_quant_rows(w8: *mut u8, scale: *mut f32, w: *const u16, n: u32, k: u32) {
    let red = static_shared!(fp8q_red, f32, 128);
    let r = bid_x() as usize;
    if r >= n as usize {
        return;
    }
    let row = w.add(r * k as usize);
    let mut amax = 0.0f32;
    let mut i = tid_x();
    while i < k {
        amax = amax.max(bf16::to_f32(*row.add(i as usize)).abs());
        i += 256;
    }
    amax = warp::block_max(amax, red);
    let s = amax.max(1e-30) / 448.0;
    if tid_x() == 0 {
        *scale.add(r) = s;
    }
    let out = w8.add(r * k as usize);
    i = tid_x();
    while i < k {
        *out.add(i as usize) = to_e4m3(bf16::to_f32(*row.add(i as usize)) / s) as u8;
        i += 256;
    }
}

/// FP8 codes + row scales -> bf16 weights bf16(q * scale[n]). grid = n, block = 256, k % 8 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_dequant_fp8(wb: *mut u16, w8: *const u8, scale: *const f32, n: u32, k: u32) {
    pdl_wait();
    pdl_trigger();
    let r = bid_x() as usize;
    if r >= n as usize {
        return;
    }
    let s = *scale.add(r);
    let src = w8.add(r * k as usize) as *const [u32; 2];
    let dst = wb.add(r * k as usize) as *mut [u32; 4];
    let mut i = tid_x();
    while i < k / 8 {
        let v = *src.add(i as usize);
        let a = e4m3x4(v[0]);
        let b = e4m3x4(v[1]);
        let o0 = bf16::pack2(a[0] * s, a[1] * s);
        let o1 = bf16::pack2(a[2] * s, a[3] * s);
        let o2 = bf16::pack2(b[0] * s, b[1] * s);
        let o3 = bf16::pack2(b[2] * s, b[3] * s);
        *dst.add(i as usize) = [o0, o1, o2, o3];
        i += 256;
    }
}

macro_rules! gemv_fp8 {
    ($name:ident, $R:tt) => {
        /// y[r][n] = bf16( sum_k bf16(q[n][k] * scale[n]) * x[r][k] ) for r < m <= R (with `res`: bf16(bf16(.) + res), res
        /// stride ldy). One warp a weight row; lane l reads 8 code bytes at chunks l, l + 32, ... (3 loads in flight) and
        /// sums sequentially in f32, the warp's lanes meet in a butterfly: the same arithmetic for every row of every R.
        /// grid = ceil(n / 8), block = 256, k % 256 == 0.
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(y: *mut u16, x: *const u16, w8: *const u8, scale: *const f32,
            res: *const u16, m: u32, n: u32, k: u32, lda: u32, ldy: u32) {
            let col = bid_x() * 8 + warp_id();
            if col >= n {
                return;
            }
            let l = lane();
            // prologue (weights do not depend on the previous grid): row into L2, one 128-byte line a lane
            if l * 128 < k {
                let pp = w8.add(col as usize * k as usize + l as usize * 128);
                unsafe { asm!("prefetch.global.L2 [{p}];", p = in(reg64) pp, options(nostack, readonly)) };
            }
            pdl_wait();
            pdl_trigger();
            let s = *scale.add(col as usize);
            let wr = w8.add(col as usize * k as usize) as *const [u32; 2];
            let mut acc = [0.0f32; $R];
            let chunks = k / 8;
            let mut i = l;
            while i + 64 < chunks {
                let mut v = [[0u32; 2]; 3];
                kcommon::unroll!(u in 3 => {
                    v[u] = *wr.add((i + 32 * u as u32) as usize);
                });
                kcommon::unroll!(u in 3 => {
                    let a = e4m3x4(v[u][0]);
                    let b = e4m3x4(v[u][1]);
                    let w = [bf16::round(a[0] * s), bf16::round(a[1] * s), bf16::round(a[2] * s), bf16::round(a[3] * s),
                        bf16::round(b[0] * s), bf16::round(b[1] * s), bf16::round(b[2] * s), bf16::round(b[3] * s)];
                    kcommon::unroll!(r in $R => {
                        if (r as u32) < m {
                            let xv = bf16::unpack8(ldg_nc_v4(x.add(r * lda as usize + (i + 32 * u as u32) as usize * 8) as *const u32));
                            let mut d = acc[r];
                            kcommon::unroll!(j in 8 => {
                                d = math::fma(w[j], xv[j], d);
                            });
                            acc[r] = d;
                        }
                    });
                });
                i += 96;
            }
            while i < chunks {
                let v = *wr.add(i as usize);
                let a = e4m3x4(v[0]);
                let b = e4m3x4(v[1]);
                let w = [bf16::round(a[0] * s), bf16::round(a[1] * s), bf16::round(a[2] * s), bf16::round(a[3] * s),
                    bf16::round(b[0] * s), bf16::round(b[1] * s), bf16::round(b[2] * s), bf16::round(b[3] * s)];
                kcommon::unroll!(r in $R => {
                    if (r as u32) < m {
                        let xv = bf16::unpack8(ldg_nc_v4(x.add(r * lda as usize + i as usize * 8) as *const u32));
                        let mut d = acc[r];
                        kcommon::unroll!(j in 8 => {
                            d = math::fma(w[j], xv[j], d);
                        });
                        acc[r] = d;
                    }
                });
                i += 32;
            }
            kcommon::unroll!(r in $R => {
                if (r as u32) < m {
                    let v = warp::sum(acc[r]);
                    if l == 0 {
                        let off = r * ldy as usize + col as usize;
                        let mut o = v;
                        if !res.is_null() {
                            o = bf16::round(o) + bf16::to_f32(*res.add(off));
                        }
                        *y.add(off) = bf16::from_f32(o);
                    }
                }
            });
        }
    };
}
gemv_fp8!(ink_gemv_fp8_r1, 1);
gemv_fp8!(ink_gemv_fp8_r4, 4);
gemv_fp8!(ink_gemv_fp8_r8, 8);
gemv_fp8!(ink_gemv_fp8_r16, 16);
