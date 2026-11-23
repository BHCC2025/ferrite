//! Decode-size twin of the grouped GEMMs (groups of <= 16 rows): the operands swap roles on the tensor core, the weights
//! are the m16 side (each warp 16 weight rows) and the <= 16 activation rows the n8 side. Same k order and the same
//! per-element products as `moe_gemm`, so each output element should be bit-identical (checked by ink-kcheck). The tile
//! is 128 weight rows x all K like before, but a stage holds only 16 activation rows: 10.5 KB instead of 18, so two
//! CTAs share an SM and the 144 CTAs of one expert layer fit in one wave. smem = SMEM_SW.
use super::*;
use super::moe_gemm::{mma_mxfp8, mma_nvfp4, swz, tile};
use super::quant::{mxfp8_block, quant_block};
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};

const BKB: u32 = 64;
/// warps a CTA = weight rows a CTA / 16 (4: 64 rows, 128 threads, 36 CTAs an expert tile; 8 would be 128 rows)
pub const NW: u32 = 4;
const ROWS: u32 = NW * 16;
const W_CODES: u32 = 0; // ROWS x 64 B
const W_SF: u32 = ROWS * 64; // ROWS x 8 B (MXFP8: x 2 used)
const A_CODES: u32 = W_SF + ROWS * 8; // 16 rows x 64 B
const A_SF: u32 = A_CODES + 1024; // FP4: 16 x 8 = 128 B; MXFP8: 16 x 2 = 32 B
const STAGE: u32 = A_SF + 256;
const STAGES: u32 = 4;
pub const SMEM_SW: u32 = STAGES * STAGE;

#[inline(always)]
unsafe fn ld_w<const FP4: bool>(st: Shared<u8>, w: *const u8, sw: *const u8, n0: u32, ld: u32, kt: u32, ktiles: u32) {
    let t = tid_x();
    // the CTA's ROWS rows are a contiguous slice of the 128-row tile (codes: 64 B a row, scales: 8 B / 2 B a row)
    let base = w.add(((n0 >> 7) * ktiles + kt) as usize * 8192 + (n0 & 127) as usize * 64);
    kcommon::unroll!(j in 2 => {
        let i = t + j as u32 * (NW * 32);
        cp_async_16(st.byte_add(W_CODES + swz(i >> 2, i & 3)).addr(), base.add(i as usize * 16));
    });
    if FP4 {
        if t < ROWS / 2 {
            let o = t * 16;
            let tile = ((n0 >> 7) * ktiles + kt) as usize * 1024 + (n0 & 127) as usize * 8;
            cp_async_16(st.byte_add(W_SF + o).addr(), sw.add(tile + o as usize));
        }
    } else if t < ROWS / 8 {
        let o = t * 16;
        let tile = ((n0 >> 7) * ktiles + kt) as usize * 256 + (n0 & 127) as usize * 2;
        cp_async_16(st.byte_add(W_SF + o).addr(), sw.add(tile + o as usize));
    }
}

/// 16 activation rows (zero-filled past `valid`) and their scales.
#[inline(always)]
unsafe fn ld_a<const FP4: bool>(st: Shared<u8>, a: *const u8, sa: *const u8, m0: u32, valid: u32, ld: u32, kt: u32, ktiles: u32) {
    let t = tid_x();
    if t < 64 {
        let r = t >> 2;
        let c = t & 3;
        let d = st.byte_add(A_CODES + swz(r, c)).addr();
        if r < valid {
            cp_async_16(d, a.add((m0 + r) as usize * ld as usize + (kt * BKB + c * 16) as usize));
        } else {
            cp_async_16_zfill(d, a, 0);
        }
    } else if FP4 {
        if t < 72 {
            let o = (t - 64) * 16;
            let tile = ((m0 >> 7) * ktiles + kt) as usize * 1024;
            cp_async_16(st.byte_add(A_SF + o).addr(), sa.add(tile + o as usize));
        }
    } else if t < 66 {
        let o = (t - 64) * 16;
        let tile = ((m0 >> 7) * ktiles + kt) as usize * 256;
        cp_async_16(st.byte_add(A_SF + o).addr(), sa.add(tile + o as usize));
    }
}

/// acc[j]: n8 token tile j (tokens 8j..8j+8) of this warp's 16 weight rows (row 16 * warp + ..).
#[inline(always)]
unsafe fn mainloop_sw<const FP4: bool>(a: *const u8, sa: *const u8, w: *const u8, sw: *const u8, m0: u32, valid: u32,
    n0: u32, k: u32) -> [[f32; 4]; 2] {
    let smem: Shared<u8> = dyn_base();
    let ld = if FP4 { k >> 1 } else { k };
    let ktiles = ld / BKB;
    let wid = warp_id();
    let l = lane();
    let mut acc = [[0.0f32; 4]; 2];
    let mut s = 0;
    while s < STAGES - 1 {
        if s < ktiles {
            ld_w::<FP4>(smem.byte_add(s * STAGE), w, sw, n0, ld, s, ktiles);
        }
        s += 1;
    }
    cp_async_commit();
    pdl_wait();
    pdl_trigger();
    s = 0;
    while s < STAGES - 1 {
        if s < ktiles {
            ld_a::<FP4>(smem.byte_add(s * STAGE), a, sa, m0, valid, ld, s, ktiles);
        }
        cp_async_commit();
        s += 1;
    }
    let mut kt = 0;
    while kt < ktiles {
        cp_async_wait::<2>();
        sync_threads();
        let pf = kt + STAGES - 1;
        if pf < ktiles {
            let sp = smem.byte_add((pf % STAGES) * STAGE);
            ld_w::<FP4>(sp, w, sw, n0, ld, pf, ktiles);
            ld_a::<FP4>(sp, a, sa, m0, valid, ld, pf, ktiles);
        }
        cp_async_commit();
        let st = smem.byte_add((kt % STAGES) * STAGE);
        kcommon::unroll!(ks in 2 => {
            let kc2 = ks as u32 * 2;
            // A operand = this warp's 16 weight rows
            let af = ldmatrix_x4(st.byte_add(W_CODES + swz(wid * 16 + (l & 15), kc2 + (l >> 4))).addr());
            // B operand = tokens 0..16: two n8 tiles in one x4
            let v = ldmatrix_x4(st.byte_add(A_CODES + swz((l >> 4) * 8 + (l & 7), kc2 + ((l >> 3) & 1))).addr());
            let bf = [[v[0], v[1]], [v[2], v[3]]];
            if FP4 {
                let sfa: u32 = st.byte_add(W_SF + (wid * 16 + (l >> 2) + 8 * (l & 1)) * 8 + ks as u32 * 4).cast::<u32>().ld();
                let sfb0: u32 = st.byte_add(A_SF + (l >> 2) * 8 + ks as u32 * 4).cast::<u32>().ld();
                let sfb1: u32 = st.byte_add(A_SF + (8 + (l >> 2)) * 8 + ks as u32 * 4).cast::<u32>().ld();
                mma_nvfp4(&mut acc[0], af, bf[0], sfa, sfb0);
                mma_nvfp4(&mut acc[1], af, bf[1], sfa, sfb1);
            } else {
                let pa: Shared<u16> = st.byte_add(W_SF + (wid * 16 + (l >> 2) + 8 * (l & 1)) * 2).cast();
                let pb0: Shared<u16> = st.byte_add(A_SF + (l >> 2) * 2).cast();
                let pb1: Shared<u16> = st.byte_add(A_SF + (8 + (l >> 2)) * 2).cast();
                let sh = 8 * ks as u32;
                let sfa = ((pa.ld() as u32) >> sh) & 0xff;
                let sfb0 = ((pb0.ld() as u32) >> sh) & 0xff;
                let sfb1 = ((pb1.ld() as u32) >> sh) & 0xff;
                mma_mxfp8(&mut acc[0], af, bf[0], sfa, sfb0);
                mma_mxfp8(&mut acc[1], af, bf[1], sfa, sfb1);
            }
        });
        kt += 1;
    }
    cp_async_wait_all();
    acc
}

/// Swapped twin of `ink_gemm_gu_nvfp4` (same arguments and output). Groups of <= 16 rows.
/// grid = (2304 / 128, max tiles), block = 256, smem = SMEM_SW.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_gu_nvfp4_sw(h: *mut u16, a: *const u8, sa: *const u8, arow: *const f32,
    w13: *const u8, s13: *const u8, wamax: *const f32, rcp: f32, gam: *const f32, pop: *const u32, tab: *const u32) {
    let Some((e, m0, valid)) = tile(tab) else { return };
    let n0 = bid_x() * ROWS;
    let we = *(w13 as *const *const u8).add(e as usize);
    let se = *(s13 as *const *const u8).add(e as usize);
    let acc = mainloop_sw::<true>(a, sa, we, se, m0, valid, n0, 2304);
    let wa = *wamax.add(e as usize);
    let wid = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    let f = (n0 >> 1) + wid * 8 + g; // feature of this lane: gate row g, up row g + 8 of the warp's 16
    kcommon::unroll!(j in 2 => {
        kcommon::unroll!(ti in 2 => {
            let tok = j as u32 * 8 + 2 * t4 + ti as u32;
            if tok < valid {
                let row = m0 + tok;
                let al = (*arow.add(row as usize) * wa) * rcp;
                let gamma = *gam.add(*pop.add(row as usize) as usize);
                let ga = bf16::round(bf16::round(acc[j][ti]) * al);
                let ua = bf16::round(bf16::round(acc[j][2 + ti]) * al);
                *h.add(row as usize * 1152 + f as usize) = swiglu_gamma(ga, ua, gamma);
            }
        });
    });
}

/// Swapped twin of `ink_gemm_down_nvfp4`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_down_nvfp4_sw(y: *mut u16, a: *const u8, sa: *const u8, arow: *const f32,
    w2: *const u8, s2: *const u8, wamax: *const f32, rcp: f32, tab: *const u32) {
    let Some((e, m0, valid)) = tile(tab) else { return };
    let n0 = bid_x() * ROWS;
    let we = *(w2 as *const *const u8).add(e as usize);
    let se = *(s2 as *const *const u8).add(e as usize);
    let acc = mainloop_sw::<true>(a, sa, we, se, m0, valid, n0, 1152);
    let wa = *wamax.add(e as usize);
    let wid = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    kcommon::unroll!(j in 2 => {
        kcommon::unroll!(ti in 2 => {
            let tok = j as u32 * 8 + 2 * t4 + ti as u32;
            if tok < valid {
                let row = m0 + tok;
                let al = (*arow.add(row as usize) * wa) * rcp;
                let n = n0 + wid * 16 + g;
                *y.add(row as usize * 2304 + n as usize) = bf16::from_f32(bf16::round(acc[j][ti]) * al);
                *y.add(row as usize * 2304 + n as usize + 8) = bf16::from_f32(bf16::round(acc[j][2 + ti]) * al);
            }
        });
    });
}

/// Swapped twin of `ink_gemm_gu_mxfp8`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_gu_mxfp8_sw(h: *mut u16, a: *const u8, sa: *const u8, w13: *const u8,
    s13: *const u8, gam: *const f32, pop: *const u32, tab: *const u32) {
    let Some((e, m0, valid)) = tile(tab) else { return };
    let n0 = bid_x() * ROWS;
    let we = *(w13 as *const *const u8).add(e as usize);
    let se = *(s13 as *const *const u8).add(e as usize);
    let acc = mainloop_sw::<false>(a, sa, we, se, m0, valid, n0, 2304);
    let wid = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    let f = (n0 >> 1) + wid * 8 + g;
    kcommon::unroll!(j in 2 => {
        kcommon::unroll!(ti in 2 => {
            let tok = j as u32 * 8 + 2 * t4 + ti as u32;
            if tok < valid {
                let row = m0 + tok;
                let gamma = *gam.add(*pop.add(row as usize) as usize);
                *h.add(row as usize * 1152 + f as usize) =
                    swiglu_gamma(bf16::round(acc[j][ti]), bf16::round(acc[j][2 + ti]), gamma);
            }
        });
    });
}

/// Swapped twin of `ink_gemm_down_mxfp8`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_down_mxfp8_sw(y: *mut u16, a: *const u8, sa: *const u8, w2: *const u8,
    s2: *const u8, tab: *const u32) {
    let Some((e, m0, valid)) = tile(tab) else { return };
    let n0 = bid_x() * ROWS;
    let we = *(w2 as *const *const u8).add(e as usize);
    let se = *(s2 as *const *const u8).add(e as usize);
    let acc = mainloop_sw::<false>(a, sa, we, se, m0, valid, n0, 1152);
    let wid = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    kcommon::unroll!(j in 2 => {
        kcommon::unroll!(ti in 2 => {
            let tok = j as u32 * 8 + 2 * t4 + ti as u32;
            if tok < valid {
                let row = m0 + tok;
                let n = n0 + wid * 16 + g;
                *y.add(row as usize * 2304 + n as usize) = bf16::from_f32(acc[j][ti]);
                *y.add(row as usize * 2304 + n as usize + 8) = bf16::from_f32(acc[j][2 + ti]);
            }
        });
    });
}

// ---- down GEMMs with the W2-input quantization fused in (decode size) ------------------------------------------------
// The SwiGLU output h (bf16, grouped rows, written by the gate|up GEMM) is quantized by every CTA of the down GEMM into
// shared memory (<= 16 rows x K: tiny) while its weight stages stream in, which removes the quantize launch and the
// global round trip of the quantized operand. Same rule and same bits as `ink_quant_nvfp4` / `ink_quant_mxfp8`.
const WSTAGE: u32 = ROWS * 72; // W codes ROWS x 64 B + W scales ROWS x 8 B

/// One lane's 4 adjacent NVFP4 blocks of row `r`: quantize `h` row into the smem A operand (codes tile kt = b0 / 8, swizzled).
#[inline(always)]
unsafe fn quant_a_fp4(smem: Shared<u8>, a_codes: u32, a_sf: u32, a_row: Shared<f32>, h: *const u16, m0: u32, valid: u32) {
    let wid = warp_id();
    let l = lane();
    kcommon::unroll!(rr in 4 => {
        let r = wid + NW * rr as u32;
        let live = r < valid;
        let mut v = [[0.0f32; 16]; 4];
        let mut bm = [0.0f32; 4];
        let mut amax = 0.0f32;
        if live && l < 18 {
            let p = h.add((m0 + r) as usize * 1152 + l as usize * 64) as *const u32;
            kcommon::unroll!(j in 4 => {
                let f0 = bf16::unpack8(ldg_nc_v4(p.add(j * 8)));
                let f1 = bf16::unpack8(ldg_nc_v4(p.add(j * 8 + 4)));
                kcommon::unroll!(i in 8 => {
                    v[j][i] = f0[i];
                    v[j][8 + i] = f1[i];
                    bm[j] = bm[j].max(f0[i].abs()).max(f1[i].abs());
                });
                amax = amax.max(bm[j]);
            });
        }
        amax = warp::max(amax);
        if l < 18 {
            let b0 = l * 4;
            let kt = b0 >> 3;
            let mut sfp = 0u32;
            kcommon::unroll!(j in 4 => {
                let (s8, q) = if live { quant_block(v[j], amax, bm[j]) } else { (0, [0u32, 0u32]) };
                let bb = (b0 & 7) + j as u32;
                let c = bb >> 1;
                let at = a_codes + kt * 1024 + swz(r, c) + (bb & 1) * 8;
                smem.byte_add(at).cast::<u32>().st(q[0]);
                smem.byte_add(at + 4).cast::<u32>().st(q[1]);
                sfp |= s8 << (8 * j as u32);
            });
            smem.byte_add(a_sf + kt * 128 + r * 8 + (b0 & 7)).cast::<u32>().st(sfp);
        }
        if l == 0 {
            a_row.set(r, if live { amax } else { 0.0 });
        }
    });
}

/// One lane's k-tile (2 adjacent MXFP8 blocks = 64 values) of row `r`.
#[inline(always)]
unsafe fn quant_a_fp8(smem: Shared<u8>, a_codes: u32, a_sf: u32, h: *const u16, m0: u32, valid: u32) {
    let wid = warp_id();
    let l = lane();
    kcommon::unroll!(rr in 4 => {
        let r = wid + NW * rr as u32;
        if l < 18 {
            let live = r < valid;
            let mut e = [0u32; 2];
            let mut w = [[0u32; 8]; 2];
            if live {
                let p = h.add((m0 + r) as usize * 1152 + l as usize * 64) as *const u32;
                kcommon::unroll!(j in 2 => {
                    let mut x = [0.0f32; 32];
                    kcommon::unroll!(q in 4 => {
                        let f = bf16::unpack8(ldg_nc_v4(p.add(j * 16 + q * 4)));
                        kcommon::unroll!(i in 8 => {
                            x[q * 8 + i] = f[i];
                        });
                    });
                    let (ex, ww) = mxfp8_block(x);
                    e[j] = ex;
                    w[j] = ww;
                });
            }
            kcommon::unroll!(j in 2 => {
                kcommon::unroll!(c in 2 => {
                    let at = a_codes + l * 1024 + swz(r, 2 * j as u32 + c as u32);
                    smem.byte_add(at).cast::<[u32; 4]>().st([w[j][c * 4], w[j][c * 4 + 1], w[j][c * 4 + 2], w[j][c * 4 + 3]]);
                });
            });
            smem.byte_add(a_sf + l * 32 + r * 2).cast::<u16>().st((e[0] | e[1] << 8) as u16);
        }
    });
}

#[inline(always)]
unsafe fn mainloop_swq<const FP4: bool>(h: *const u16, w: *const u8, sw: *const u8, m0: u32, valid: u32, n0: u32, k: u32)
    -> ([[f32; 4]; 2], Shared<f32>) {
    let smem: Shared<u8> = dyn_base();
    let ld = if FP4 { k >> 1 } else { k };
    let ktiles = ld / BKB;
    let stg: u32 = if FP4 { 4 } else { 3 };
    let a_codes = stg * WSTAGE;
    let a_sf = a_codes + ktiles * 1024;
    let a_row_off = a_sf + ktiles * if FP4 { 128 } else { 32 };
    let a_row: Shared<f32> = smem.byte_add(a_row_off).cast();
    let wid = warp_id();
    let l = lane();
    let mut acc = [[0.0f32; 4]; 2];
    let mut s = 0;
    while s < stg - 1 {
        if s < ktiles {
            ld_w::<FP4>(smem.byte_add(s * WSTAGE), w, sw, n0, ld, s, ktiles);
        }
        cp_async_commit();
        s += 1;
    }
    pdl_wait();
    pdl_trigger();
    if FP4 {
        quant_a_fp4(smem, a_codes, a_sf, a_row, h, m0, valid);
    } else {
        quant_a_fp8(smem, a_codes, a_sf, h, m0, valid);
    }
    let mut kt = 0;
    while kt < ktiles {
        if FP4 { cp_async_wait::<2>() } else { cp_async_wait::<1>() };
        sync_threads();
        let pf = kt + stg - 1;
        if pf < ktiles {
            ld_w::<FP4>(smem.byte_add((pf % stg) * WSTAGE), w, sw, n0, ld, pf, ktiles);
        }
        cp_async_commit();
        let st = smem.byte_add((kt % stg) * WSTAGE);
        kcommon::unroll!(ks in 2 => {
            let kc2 = ks as u32 * 2;
            let af = ldmatrix_x4(st.byte_add(W_CODES + swz(wid * 16 + (l & 15), kc2 + (l >> 4))).addr());
            let v = ldmatrix_x4(smem.byte_add(a_codes + kt * 1024 + swz((l >> 4) * 8 + (l & 7), kc2 + ((l >> 3) & 1))).addr());
            let bf = [[v[0], v[1]], [v[2], v[3]]];
            if FP4 {
                let sfa: u32 = st.byte_add(W_SF + (wid * 16 + (l >> 2) + 8 * (l & 1)) * 8 + ks as u32 * 4).cast::<u32>().ld();
                let sfb0: u32 = smem.byte_add(a_sf + kt * 128 + (l >> 2) * 8 + ks as u32 * 4).cast::<u32>().ld();
                let sfb1: u32 = smem.byte_add(a_sf + kt * 128 + (8 + (l >> 2)) * 8 + ks as u32 * 4).cast::<u32>().ld();
                mma_nvfp4(&mut acc[0], af, bf[0], sfa, sfb0);
                mma_nvfp4(&mut acc[1], af, bf[1], sfa, sfb1);
            } else {
                let pa: Shared<u16> = st.byte_add(W_SF + (wid * 16 + (l >> 2) + 8 * (l & 1)) * 2).cast();
                let pb0: Shared<u16> = smem.byte_add(a_sf + kt * 32 + (l >> 2) * 2).cast();
                let pb1: Shared<u16> = smem.byte_add(a_sf + kt * 32 + (8 + (l >> 2)) * 2).cast();
                let sh = 8 * ks as u32;
                let sfa = ((pa.ld() as u32) >> sh) & 0xff;
                let sfb0 = ((pb0.ld() as u32) >> sh) & 0xff;
                let sfb1 = ((pb1.ld() as u32) >> sh) & 0xff;
                mma_mxfp8(&mut acc[0], af, bf[0], sfa, sfb0);
                mma_mxfp8(&mut acc[1], af, bf[1], sfa, sfb1);
            }
        });
        kt += 1;
    }
    cp_async_wait_all();
    (acc, a_row)
}

/// smem of the fused-quant down GEMMs: NVFP4 = 4 W stages + A (9 k-tiles); MXFP8 = 3 W stages + A (18 k-tiles).
pub const SMEM_SWQ_FP4: u32 = 4 * WSTAGE + 9 * 1024 + 9 * 128 + 64;
pub const SMEM_SWQ_FP8: u32 = 3 * WSTAGE + 18 * 1024 + 18 * 32 + 64;

/// `ink_gemm_down_nvfp4_sw` with the quantization of `h` (bf16 [pos][1152], the gate|up output) fused in.
/// grid = (2304 / 128, max tiles), block = 256, smem = SMEM_SWQ_FP4. Groups of <= 16 rows.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_down_nvfp4_swq(y: *mut u16, h: *const u16, w2: *const u8, s2: *const u8,
    wamax: *const f32, rcp: f32, tab: *const u32) {
    let Some((e, m0, valid)) = tile(tab) else { return };
    let n0 = bid_x() * ROWS;
    let we = *(w2 as *const *const u8).add(e as usize);
    let se = *(s2 as *const *const u8).add(e as usize);
    let (acc, a_row) = mainloop_swq::<true>(h, we, se, m0, valid, n0, 1152);
    let wa = *wamax.add(e as usize);
    let wid = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    kcommon::unroll!(j in 2 => {
        kcommon::unroll!(ti in 2 => {
            let tok = j as u32 * 8 + 2 * t4 + ti as u32;
            if tok < valid {
                let row = m0 + tok;
                let al = (a_row.at(tok) * wa) * rcp;
                let n = n0 + wid * 16 + g;
                *y.add(row as usize * 2304 + n as usize) = bf16::from_f32(bf16::round(acc[j][ti]) * al);
                *y.add(row as usize * 2304 + n as usize + 8) = bf16::from_f32(bf16::round(acc[j][2 + ti]) * al);
            }
        });
    });
}

/// MXFP8 twin. smem = SMEM_SWQ_FP8.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_down_mxfp8_swq(y: *mut u16, h: *const u16, w2: *const u8, s2: *const u8,
    tab: *const u32) {
    let Some((e, m0, valid)) = tile(tab) else { return };
    let n0 = bid_x() * ROWS;
    let we = *(w2 as *const *const u8).add(e as usize);
    let se = *(s2 as *const *const u8).add(e as usize);
    let (acc, _) = mainloop_swq::<false>(h, we, se, m0, valid, n0, 1152);
    let wid = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    kcommon::unroll!(j in 2 => {
        kcommon::unroll!(ti in 2 => {
            let tok = j as u32 * 8 + 2 * t4 + ti as u32;
            if tok < valid {
                let row = m0 + tok;
                let n = n0 + wid * 16 + g;
                *y.add(row as usize * 2304 + n as usize) = bf16::from_f32(acc[j][ti]);
                *y.add(row as usize * 2304 + n as usize + 8) = bf16::from_f32(acc[j][2 + ti]);
            }
        });
    });
}
