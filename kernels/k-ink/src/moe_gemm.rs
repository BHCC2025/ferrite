//! Grouped NVFP4 x NVFP4 block-scaled tensor-core GEMMs over the experts of a MoE layer (the k-fp4 mainloop, one
//! 128 x 128 tile a CTA, 4-stage cp.async ring, `mma.sync ... mxf4nvf4.block_scale.scale_vec::4X`). A CTA's tile
//! comes from the routing plan's tile table: (expert, first grouped row, valid rows). Rows of the activation
//! operand are in grouped order (128-row aligned groups), scales tiled; weights are one expert's [2304][K/2] codes with
//! tiled scales. The epilogue multiplies each row's accumulator by the outer scale (arow[row] * wamax[expert]) * rcp in f32,
//! then rounds to bf16 (the native rule: both outer scales are applied to the exact FP32 accumulator).
use super::*;
use super::plan::NONE;
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};

const BKB: u32 = 64; // bytes of one row per k-tile (128 fp4)
const CODE_TILE: u32 = 128 * BKB; // 8 KB
const SF_TILE: u32 = 128 * 8; // 1 KB
const STAGES: u32 = 4;
const STAGE_BYTES: u32 = 2 * CODE_TILE + 2 * SF_TILE; // A codes | W codes | A scales | W scales
pub const SMEM: u32 = STAGES * STAGE_BYTES; // 72 KB

/// D += A . B with NVFP4 block scales (e2m1 data, ue4m3 scale per 16 k), m16n8k64.
#[inline(always)]
pub(crate) fn mma_nvfp4(d: &mut [f32; 4], a: [u32; 4], b: [u32; 2], sfa: u32, sfb: u32) {
    unsafe {
        asm!("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 \
              {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}}, \
              {{{sa}}}, {{{z}, {z}}}, {{{sb}}}, {{{z}, {z}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], sa = in(reg32) sfa, sb = in(reg32) sfb, z = in(reg16) 0u16,
            options(nomem, nostack));
    }
}

/// D += A . B with MXFP8 block scales (e4m3 data, ue8m0 scale per 32 k), m16n8k32. `sfa` / `sfb` carry the scale byte in
/// bits 0..8 (same lane convention as `mma_nvfp4`).
#[inline(always)]
pub(crate) fn mma_mxfp8(d: &mut [f32; 4], a: [u32; 4], b: [u32; 2], sfa: u32, sfb: u32) {
    unsafe {
        asm!("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 \
              {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}}, \
              {{{sa}}}, {{{z}, {z}}}, {{{sb}}}, {{{z}, {z}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], sa = in(reg32) sfa, sb = in(reg32) sfb, z = in(reg16) 0u16,
            options(nomem, nostack));
    }
}

/// Byte offset of 16-byte chunk `c` (0..4) of row `r` inside a [rows][64 B] tile, XOR-swizzled.
#[inline(always)]
pub(crate) fn swz(r: u32, c: u32) -> u32 {
    r * 64 + ((c ^ ((r >> 1) & 3)) << 4)
}

/// 128 rows x 64 bytes from `src` (row stride `ld` bytes) at byte column `kb0`; rows >= `rows` zero-filled.
#[inline(always)]
unsafe fn load_codes(dst: Shared<u8>, src: *const u8, ld: u32, row0: u32, rows: u32, kb0: u32) {
    let t = tid_x();
    kcommon::unroll!(j in 2 => {
        let i = t + j as u32 * 256;
        let r = i >> 2;
        let c = i & 3;
        let gr = row0 + r;
        let d = dst.byte_add(swz(r, c)).addr();
        if gr < rows {
            cp_async_16(d, src.add(gr as usize * ld as usize + (kb0 + c * 16) as usize));
        } else {
            cp_async_16_zfill(d, src, 0);
        }
    });
}

/// Weight half of a stage: W codes and W scales (independent of the previous grid).
#[inline(always)]
unsafe fn load_stage_w<const FP4: bool>(st: Shared<u8>, w: *const u8, sw: *const u8, n0: u32, ld: u32, kt: u32, ktiles: u32) {
    // tile-contiguous weight codes: one 8 KB read (see quant.rs `tiled_code_off`)
    {
        let t = tid_x();
        let base = w.add(((n0 >> 7) * ktiles + kt) as usize * 8192);
        kcommon::unroll!(j in 2 => {
            let i = t + j as u32 * 256;
            cp_async_16(st.byte_add(CODE_TILE + swz(i >> 2, i & 3)).addr(), base.add(i as usize * 16));
        });
    }
    let t = tid_x();
    if FP4 {
        if t >= 64 && t < 128 {
            let o = (t & 63) * 16;
            let tile = ((n0 >> 7) * ktiles + kt) as usize * SF_TILE as usize;
            cp_async_16(st.byte_add(2 * CODE_TILE + SF_TILE + o).addr(), sw.add(tile + o as usize));
        }
    } else if t >= 16 && t < 32 {
        // MXFP8: 128 rows x 2 ue8m0 scales (one a 32-k step) = 256 B a tile
        let o = (t & 15) * 16;
        let tile = ((n0 >> 7) * ktiles + kt) as usize * 256;
        cp_async_16(st.byte_add(2 * CODE_TILE + SF_TILE + o).addr(), sw.add(tile + o as usize));
    }
}

/// Activation half of a stage: A codes and A scales (written by the previous grid).
#[inline(always)]
unsafe fn load_stage_a<const FP4: bool>(st: Shared<u8>, a: *const u8, sa: *const u8, m0: u32, m_end: u32, ld: u32, kt: u32,
    ktiles: u32) {
    load_codes(st, a, ld, m0, m_end, kt * BKB);
    let t = tid_x();
    if FP4 {
        if t < 64 {
            let o = t * 16;
            let tile = ((m0 >> 7) * ktiles + kt) as usize * SF_TILE as usize;
            cp_async_16(st.byte_add(2 * CODE_TILE + o).addr(), sa.add(tile + o as usize));
        }
    } else if t < 16 {
        let o = t * 16;
        let tile = ((m0 >> 7) * ktiles + kt) as usize * 256;
        cp_async_16(st.byte_add(2 * CODE_TILE + o).addr(), sa.add(tile + o as usize));
    }
}

/// Mainloop for one tile; returns acc[mt][j] (m16 tile mt of the warp's 64 rows, n8 tile j of its 32 columns).
#[inline(always)]
unsafe fn mainloop<const FP4: bool>(a: *const u8, sa: *const u8, w: *const u8, sw: *const u8, m0: u32, m_end: u32,
    n0: u32, k: u32) -> [[[f32; 4]; 4]; 4] {
    let smem: Shared<u8> = dyn_base();
    let ld = if FP4 { k >> 1 } else { k };
    let ktiles = ld / BKB;
    let wid = warp_id();
    let l = lane();
    let wm = wid >> 2;
    let wn = wid & 3;
    let mut acc = [[[0.0f32; 4]; 4]; 4];
    // prologue: the weights' first stages (no dependency on the previous grid), then wait for it, then the activations'
    let mut s = 0;
    while s < STAGES - 1 {
        if s < ktiles {
            load_stage_w::<FP4>(smem.byte_add(s * STAGE_BYTES), w, sw, n0, ld, s, ktiles);
        }
        s += 1;
    }
    cp_async_commit();
    pdl_wait();
    pdl_trigger();
    s = 0;
    while s < STAGES - 1 {
        if s < ktiles {
            load_stage_a::<FP4>(smem.byte_add(s * STAGE_BYTES), a, sa, m0, m_end, ld, s, ktiles);
        }
        cp_async_commit();
        s += 1;
    }
    let sf0 = 2 * CODE_TILE;
    let sfa_off = sf0 + (wm * 64 + (l >> 2) + 8 * (l & 1)) * 8;
    let sfb_off = sf0 + SF_TILE + (wn * 32 + (l >> 2)) * 8;
    let mut kt = 0;
    while kt < ktiles {
        cp_async_wait::<2>();
        sync_threads();
        let pf = kt + STAGES - 1;
        if pf < ktiles {
            load_stage_w::<FP4>(smem.byte_add((pf % STAGES) * STAGE_BYTES), w, sw, n0, ld, pf, ktiles);
            load_stage_a::<FP4>(smem.byte_add((pf % STAGES) * STAGE_BYTES), a, sa, m0, m_end, ld, pf, ktiles);
        }
        cp_async_commit();
        let st = smem.byte_add((kt % STAGES) * STAGE_BYTES);
        let sb = st.byte_add(CODE_TILE);
        kcommon::unroll!(ks in 2 => {
            let kc2 = ks as u32 * 2;
            let mut af = [[0u32; 4]; 4];
            kcommon::unroll!(mt in 4 => {
                let r = wm * 64 + mt as u32 * 16 + (l & 15);
                af[mt] = ldmatrix_x4(st.byte_add(swz(r, kc2 + (l >> 4))).addr());
            });
            let mut bf = [[0u32; 2]; 4];
            kcommon::unroll!(p in 2 => {
                let r = wn * 32 + (2 * p as u32 + (l >> 4)) * 8 + (l & 7);
                let v = ldmatrix_x4(sb.byte_add(swz(r, kc2 + ((l >> 3) & 1))).addr());
                bf[2 * p] = [v[0], v[1]];
                bf[2 * p + 1] = [v[2], v[3]];
            });
            if FP4 {
                let ssa: Shared<u32> = st.byte_add(sfa_off + ks as u32 * 4).cast();
                let ssb: Shared<u32> = st.byte_add(sfb_off + ks as u32 * 4).cast();
                let mut sfa = [0u32; 4];
                let mut sfb = [0u32; 4];
                kcommon::unroll!(i in 4 => {
                    sfa[i] = ssa.byte_add(i as u32 * 16 * 8).ld();
                    sfb[i] = ssb.byte_add(i as u32 * 8 * 8).ld();
                });
                kcommon::unroll!(mt in 4 => {
                    kcommon::unroll!(j in 4 => {
                        mma_nvfp4(&mut acc[mt][j], af[mt], bf[j], sfa[mt], sfb[j]);
                    });
                });
            } else {
                // 2 ue8m0 scales a row and k-tile: this step's is byte ks
                let ra = wm * 64 + (l >> 2) + 8 * (l & 1);
                let rb = wn * 32 + (l >> 2);
                let mut sfa = [0u32; 4];
                let mut sfb = [0u32; 4];
                kcommon::unroll!(i in 4 => {
                    let pa: Shared<u16> = st.byte_add(sf0 + (ra + i as u32 * 16) * 2).cast();
                    let pb: Shared<u16> = st.byte_add(sf0 + SF_TILE + (rb + i as u32 * 8) * 2).cast();
                    sfa[i] = ((pa.ld() as u32) >> (8 * ks as u32)) & 0xff;
                    sfb[i] = ((pb.ld() as u32) >> (8 * ks as u32)) & 0xff;
                });
                kcommon::unroll!(mt in 4 => {
                    kcommon::unroll!(j in 4 => {
                        mma_mxfp8(&mut acc[mt][j], af[mt], bf[j], sfa[mt], sfb[j]);
                    });
                });
            }
        });
        kt += 1;
    }
    cp_async_wait_all();
    acc
}

/// Tile `t` of the plan: (expert, first grouped row, valid rows), or None past the table. The plan is written two kernels
/// back (plan -> quant -> GEMM), so it is complete and visible before this kernel's `pdl_wait` (see `pdl_wait`).
#[inline(always)]
pub(crate) unsafe fn tile(tab: *const u32) -> Option<(u32, u32, u32)> {
    let t = bid_y();
    if t >= *tab {
        return None;
    }
    let p = tab.add(1 + 3 * t as usize);
    Some((*p, *p.add(1), *p.add(2)))
}

/// Gate|up GEMM for the tile's expert with the softcapped SwiGLU in the epilogue: h[pos][1152] bf16 (grouped rows) =
/// bf16(SwiGLU(bf16(bf16(acc) * alpha)) * gamma) (the sm_120 flow: the accumulator is rounded to bf16 before the outer scales) with alpha = (xamax * wamax[e]) * rcp (rcp = 1 / (1536 * denominator)), gamma =
/// gam[pair of the row] (the native rule: the routing weight is folded in before the W2 input is quantized).
/// A = quantized x (K = 2304), W = the repacked w13 (rows interleaved by 8, `ink_repack_w13_nvfp4`).
/// grid = (2304 / 128, max tiles), block = 256, smem = SMEM.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_gu_nvfp4(h: *mut u16, a: *const u8, sa: *const u8, arow: *const f32,
    w13: *const u8, s13: *const u8, wamax: *const f32, rcp: f32, gam: *const f32, pop: *const u32, tab: *const u32) {
    let Some((e, m0, valid)) = tile(tab) else { return };
    let n0 = bid_x() * 128;
    let we = *(w13 as *const *const u8).add(e as usize);
    let se = *(s13 as *const *const u8).add(e as usize);
    let acc = mainloop::<true>(a, sa, we, se, m0, m0 + valid, n0, 2304);
    let wa = *wamax.add(e as usize);
    let wid = warp_id();
    let l = lane();
    let wm = wid >> 2;
    let wn = wid & 3;
    let g = l >> 2;
    let t4 = l & 3;
    kcommon::unroll!(p in 2 => {
        let fb = (n0 >> 1) + wn * 16 + p as u32 * 8 + 2 * t4; // first of this lane's two features
        kcommon::unroll!(mt in 4 => {
            kcommon::unroll!(half in 2 => {
                let r = wm * 64 + mt as u32 * 16 + g + half as u32 * 8;
                if r < valid {
                    let row = m0 + r;
                    let al = (*arow.add(row as usize) * wa) * rcp;
                    let gamma = *gam.add(*pop.add(row as usize) as usize);
                    let ga = acc[mt][2 * p];
                    let ua = acc[mt][2 * p + 1];
                    let v0 = swiglu_gamma(bf16::round(bf16::round(ga[half * 2]) * al), bf16::round(bf16::round(ua[half * 2]) * al), gamma);
                    let v1 = swiglu_gamma(bf16::round(bf16::round(ga[half * 2 + 1]) * al), bf16::round(bf16::round(ua[half * 2 + 1]) * al), gamma);
                    *(h.add(row as usize * 1152 + fb as usize) as *mut u32) = v0 as u32 | (v1 as u32) << 16;
                }
            });
        });
    });
}

/// Down GEMM: y[pos][2304] bf16 (grouped rows) = A (quantized SwiGLU output, K = 1152) . w2[e]^T.
/// W = w2 codes as stored with tiled scales (`ink_retile_sf`). grid = (2304 / 128, max tiles), block = 256, smem = SMEM.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_down_nvfp4(y: *mut u16, a: *const u8, sa: *const u8, arow: *const f32,
    w2: *const u8, s2: *const u8, wamax: *const f32, rcp: f32, tab: *const u32) {
    let Some((e, m0, valid)) = tile(tab) else { return };
    let n0 = bid_x() * 128;
    let we = *(w2 as *const *const u8).add(e as usize);
    let se = *(s2 as *const *const u8).add(e as usize);
    let acc = mainloop::<true>(a, sa, we, se, m0, m0 + valid, n0, 1152);
    let wa = *wamax.add(e as usize);
    let wid = warp_id();
    let l = lane();
    let wm = wid >> 2;
    let wn = wid & 3;
    let g = l >> 2;
    let t4 = l & 3;
    kcommon::unroll!(j in 4 => {
        let col = n0 + wn * 32 + j as u32 * 8 + 2 * t4;
        kcommon::unroll!(mt in 4 => {
            kcommon::unroll!(half in 2 => {
                let r = wm * 64 + mt as u32 * 16 + g + half as u32 * 8;
                if r < valid {
                    let row = m0 + r;
                    let al = (*arow.add(row as usize) * wa) * rcp;
                    let v = bf16::pack2(bf16::round(acc[mt][j][half * 2]) * al, bf16::round(acc[mt][j][half * 2 + 1]) * al);
                    *(y.add(row as usize * 2304 + col as usize) as *mut u32) = v;
                }
            });
        });
    });
}

/// MXFP8 gate|up GEMM (layers 20-23): h[pos][1152] bf16 = bf16(SwiGLU(bf16(acc)) * gamma), acc exact (e4m3 x e4m3 with
/// ue8m0 block scales on both sides), no outer scales. W = repacked w13 ([2304][2304] values per expert, rows interleaved by
/// 8, tiled scales, 165888 scale bytes an expert). grid = (2304 / 128, max tiles), block = 256, smem = SMEM.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_gu_mxfp8(h: *mut u16, a: *const u8, sa: *const u8, w13: *const u8,
    s13: *const u8, gam: *const f32, pop: *const u32, tab: *const u32) {
    let Some((e, m0, valid)) = tile(tab) else { return };
    let n0 = bid_x() * 128;
    let we = *(w13 as *const *const u8).add(e as usize);
    let se = *(s13 as *const *const u8).add(e as usize);
    let acc = mainloop::<false>(a, sa, we, se, m0, m0 + valid, n0, 2304);
    let wid = warp_id();
    let l = lane();
    let wm = wid >> 2;
    let wn = wid & 3;
    let g = l >> 2;
    let t4 = l & 3;
    kcommon::unroll!(p in 2 => {
        let fb = (n0 >> 1) + wn * 16 + p as u32 * 8 + 2 * t4;
        kcommon::unroll!(mt in 4 => {
            kcommon::unroll!(half in 2 => {
                let r = wm * 64 + mt as u32 * 16 + g + half as u32 * 8;
                if r < valid {
                    let row = m0 + r;
                    let gamma = *gam.add(*pop.add(row as usize) as usize);
                    let ga = acc[mt][2 * p];
                    let ua = acc[mt][2 * p + 1];
                    let v0 = swiglu_gamma(bf16::round(ga[half * 2]), bf16::round(ua[half * 2]), gamma);
                    let v1 = swiglu_gamma(bf16::round(ga[half * 2 + 1]), bf16::round(ua[half * 2 + 1]), gamma);
                    *(h.add(row as usize * 1152 + fb as usize) as *mut u32) = v0 as u32 | (v1 as u32) << 16;
                }
            });
        });
    });
}

/// MXFP8 down GEMM: y[pos][2304] bf16 = bf16(acc). W = w2 values as stored with tiled scales (82944 scale bytes an expert).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_down_mxfp8(y: *mut u16, a: *const u8, sa: *const u8, w2: *const u8,
    s2: *const u8, tab: *const u32) {
    let Some((e, m0, valid)) = tile(tab) else { return };
    let n0 = bid_x() * 128;
    let we = *(w2 as *const *const u8).add(e as usize);
    let se = *(s2 as *const *const u8).add(e as usize);
    let acc = mainloop::<false>(a, sa, we, se, m0, m0 + valid, n0, 1152);
    let wid = warp_id();
    let l = lane();
    let wm = wid >> 2;
    let wn = wid & 3;
    let g = l >> 2;
    let t4 = l & 3;
    kcommon::unroll!(j in 4 => {
        let col = n0 + wn * 32 + j as u32 * 8 + 2 * t4;
        kcommon::unroll!(mt in 4 => {
            kcommon::unroll!(half in 2 => {
                let r = wm * 64 + mt as u32 * 16 + g + half as u32 * 8;
                if r < valid {
                    let row = m0 + r;
                    let v = bf16::pack2(acc[mt][j][half * 2], acc[mt][j][half * 2 + 1]);
                    *(y.add(row as usize * 2304 + col as usize) as *mut u32) = v;
                }
            });
        });
    });
}

#[allow(dead_code)]
const _: u32 = NONE;
