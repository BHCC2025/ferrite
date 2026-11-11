//! NVFP4 / FP8 prompt GEMMs on block-scaled tensor cores (sm_121a `mma.sync`), and the bf16 -> NVFP4 / FP8
//! activation quantizers that feed them.
//!
//! GEMM: C[M][N] = A[M][K] . W[N][K]^T * alpha[n] (+ residual), f32 accumulate, bf16 out. A and W are both
//! K-contiguous ("TN"), so the byte layout of a 64-byte k-tile row is the same as core's bf16 GEMM tile
//! (128 fp4 or 64 fp8 per row) and the ldmatrix addressing, swizzle and fragment layout carry over unchanged:
//! an m16n8k64 e2m1 fragment and an m16n8k32 e4m3 fragment occupy exactly the bytes of an m16n8k16 bf16 one.
//!
//! NVFP4 block scales (e4m3, one per 16 elements along K) live in a tiled layout: for row tile `rt` (128 rows)
//! and k-tile `kt` (128 fp4) a contiguous 1 KB block `[128 rows][8 scales]` at `(rt * K/128 + kt) * 1024`,
//! so a k-tile's scales are one cp.async burst and each lane's mma scale operand (4 scales = one k64 step of
//! one row) is a single aligned 32-bit LDS. `fe_fp4_quant` writes activations in that layout; the host tiles
//! the checkpoint's row-major weight scales the same way at load.
//!
//! mma operand mapping (`kind::mxf4nvf4.block_scale.scale_vec::4X`, ue4m3, byte/thread id 0), g = lane/4:
//! A scales: lane holds row `g + 8 * (lane & 1)` of the m16 tile; B scales: lane holds column `g`.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

extern crate kcommon;

use core::arch::asm;
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

pub const BM: u32 = 128;
pub const BN: u32 = 128;
/// Bytes of one row per k-tile (128 fp4 or 64 fp8 elements).
const BKB: u32 = 64;
const CODE_TILE: u32 = BM * BKB; // 8 KB
/// 128 rows x 8 e4m3 scales.
const SF_TILE: u32 = BM * 8; // 1 KB
pub const FP4_SMEM: u32 = 4 * stage_bytes(true, 1); // 72 KB
pub const FP8_SMEM: u32 = 4 * stage_bytes(false, 1); // 64 KB
pub const FP4_SMEM_N256: u32 = 3 * stage_bytes(true, 2); // 81 KB
pub const FP8_SMEM_N256: u32 = 3 * stage_bytes(false, 2); // 72 KB

// ---- tensor-core wrappers ---------------------------------------------------------------------

/// D += A . B with NVFP4 block scales (e2m1 data, ue4m3 scale per 16 k), m16n8k64.
#[inline(always)]
fn mma_nvfp4(d: &mut [f32; 4], a: [u32; 4], b: [u32; 2], sfa: u32, sfb: u32) {
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

/// D += A . B, e4m3 x e4m3, m16n8k32.
#[inline(always)]
fn mma_fp8(d: &mut [f32; 4], a: [u32; 4], b: [u32; 2]) {
    unsafe {
        asm!("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 \
              {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], options(nomem, nostack));
    }
}

/// D += A . B, e4m3 x e4m3, m16n8k32, through the block-scaled form with every scale 1 (ue8m0 code 127): the
/// same products as `mma_fp8` on the block-scaled datapath.
#[inline(always)]
fn mma_fp8_bs(d: &mut [f32; 4], a: [u32; 4], b: [u32; 2]) {
    unsafe {
        asm!("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 \
              {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}}, \
              {{{sc}}}, {{{z}, {z}}}, {{{sc}}}, {{{z}, {z}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], sc = in(reg32) 0x7f7f_7f7fu32, z = in(reg16) 0u16,
            options(nomem, nostack));
    }
}

/// Same as `mma_fp8` through the sm_89 form (no `kind::`), for the peak probe.
#[inline(always)]
fn mma_fp8_legacy(d: &mut [f32; 4], a: [u32; 4], b: [u32; 2]) {
    unsafe {
        asm!("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \
              {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1], a2 = in(reg32) a[2], a3 = in(reg32) a[3],
            b0 = in(reg32) b[0], b1 = in(reg32) b[1], options(nomem, nostack));
    }
}

// ---- conversions ------------------------------------------------------------------------------

/// f32 -> e4m3 (round to nearest even, saturate to +-448), in the low byte.
#[inline(always)]
fn to_e4m3(x: f32) -> u32 {
    let h: u16;
    unsafe {
        asm!("cvt.rn.satfinite.e4m3x2.f32 {h}, {z}, {x};", h = out(reg16) h, z = in(reg32) 0.0f32, x = in(reg32) x,
            options(pure, nomem, nostack));
    }
    (h & 0xff) as u32
}

/// Four f32 -> four e4m3 bytes (x[0] in the low byte).
#[inline(always)]
fn to_e4m3x4(x0: f32, x1: f32, x2: f32, x3: f32) -> u32 {
    let (lo, hi): (u16, u16);
    unsafe {
        asm!("cvt.rn.satfinite.e4m3x2.f32 {lo}, {x1}, {x0};\n\tcvt.rn.satfinite.e4m3x2.f32 {hi}, {x3}, {x2};",
            lo = out(reg16) lo, hi = out(reg16) hi, x0 = in(reg32) x0, x1 = in(reg32) x1, x2 = in(reg32) x2,
            x3 = in(reg32) x3, options(pure, nomem, nostack));
    }
    lo as u32 | (hi as u32) << 16
}

/// Exact e4m3 -> f32.
#[inline(always)]
fn e4m3_to_f32(b: u32) -> f32 {
    let e = (b >> 3) & 15;
    let m = b & 7;
    let v = if e == 0 { m as f32 * (1.0 / 512.0) } else { f32::from_bits(((e + 120) << 23) | (m << 20)) };
    if b & 0x80 != 0 { -v } else { v }
}

/// Eight f32 -> eight e2m1 nibbles (round to nearest even, saturate to +-6), x[0] in the low nibble.
#[inline(always)]
fn to_e2m1x8(x: [f32; 8]) -> u32 {
    let r: u32;
    unsafe {
        asm!("{{\n\t.reg .b8 q0, q1, q2, q3;\n\t\
              cvt.rn.satfinite.e2m1x2.f32 q0, {x1}, {x0};\n\t\
              cvt.rn.satfinite.e2m1x2.f32 q1, {x3}, {x2};\n\t\
              cvt.rn.satfinite.e2m1x2.f32 q2, {x5}, {x4};\n\t\
              cvt.rn.satfinite.e2m1x2.f32 q3, {x7}, {x6};\n\t\
              mov.b32 {r}, {{q0, q1, q2, q3}};\n\t}}",
            r = out(reg32) r, x0 = in(reg32) x[0], x1 = in(reg32) x[1], x2 = in(reg32) x[2], x3 = in(reg32) x[3],
            x4 = in(reg32) x[4], x5 = in(reg32) x[5], x6 = in(reg32) x[6], x7 = in(reg32) x[7],
            options(pure, nomem, nostack));
    }
    r
}

/// Byte offset of NVFP4 scale (row, 16-block b) in the ferrite tiled layout (`kb` = K/16).
#[inline(always)]
fn sf_tiled(row: u32, b: u32, kb: u32) -> u32 {
    ((row >> 7) * (kb >> 3) + (b >> 3)) * SF_TILE + (row & 127) * 8 + (b & 7)
}

/// Byte offset in cuBLASLt's VEC16_UE4M3 layout (128 x 4 tiles of 512 bytes, row tiles outer).
#[inline(always)]
fn sf_cublaslt(row: u32, b: u32, kb: u32) -> u32 {
    let ct = (kb + 3) / 4;
    ((row >> 7) * ct + (b >> 2)) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4 + (b & 3)
}

// ---- quantizers -------------------------------------------------------------------------------

/// bf16 x[M][ldx] -> NVFP4: codes [M][K/2] (element 2i in the low nibble of byte i) and e4m3 block scales,
/// with global scale `g` (= 1 / input_scale): sf = e4m3(amax16 / 6 * g), q = e2m1(x * g / sf).
/// `layout` 0 = ferrite tiled scales (the GEMM below), 1 = cuBLASLt VEC16 layout. K % 128 == 0.
/// One thread per 16-element block: grid = ceil(M * K/16 / 256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_fp4_quant(codes: *mut u8, sf: *mut u8, x: *const u16, m: u32, k: u32, ldx: u32,
    g: f32, layout: u32) {
    let i = bid_x() * bdim_x() + tid_x();
    let kb = k >> 4;
    if i >= m * kb {
        return;
    }
    let row = i / kb;
    let b = i - row * kb;
    let p = x.add(row as usize * ldx as usize + b as usize * 16) as *const u32;
    let f0 = bf16::unpack8(ldg_nc_v4(p));
    let f1 = bf16::unpack8(ldg_nc_v4(p.add(4)));
    let mut amax = 0.0f32;
    kcommon::unroll!(j in 8 => {
        amax = amax.max(f0[j].abs()).max(f1[j].abs());
    });
    let s8 = to_e4m3(amax * (1.0 / 6.0) * g);
    let sv = e4m3_to_f32(s8);
    let mul = if sv > 0.0 { g / sv } else { 0.0 };
    let mut y0 = [0.0f32; 8];
    let mut y1 = [0.0f32; 8];
    kcommon::unroll!(j in 8 => {
        y0[j] = f0[j] * mul;
        y1[j] = f1[j] * mul;
    });
    let w = codes.add(row as usize * (k as usize >> 1) + b as usize * 8) as *mut u32;
    *w = to_e2m1x8(y0);
    *w.add(1) = to_e2m1x8(y1);
    let off = if layout == 0 { sf_tiled(row, b, kb) } else { sf_cublaslt(row, b, kb) };
    *sf.add(off as usize) = s8 as u8;
}

/// bf16 x[M][ldx] -> e4m3 [M][K]: q = e4m3(x * g), g = 1 / input_scale. K % 8 == 0.
/// One thread per 8 elements: grid = ceil(M * K/8 / 256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_fp8_quant(out: *mut u8, x: *const u16, m: u32, k: u32, ldx: u32, g: f32) {
    let i = bid_x() * bdim_x() + tid_x();
    let k8 = k >> 3;
    if i >= m * k8 {
        return;
    }
    let row = i / k8;
    let c = (i - row * k8) * 8;
    let f = bf16::unpack8(ldg_nc_v4(x.add(row as usize * ldx as usize + c as usize) as *const u32));
    let w = out.add(row as usize * k as usize + c as usize) as *mut u32;
    *w = to_e4m3x4(f[0] * g, f[1] * g, f[2] * g, f[3] * g);
    *w.add(1) = to_e4m3x4(f[4] * g, f[5] * g, f[6] * g, f[7] * g);
}

// ---- GEMM -------------------------------------------------------------------------------------

/// Byte offset of 16-byte chunk `c` (0..4) of row `r` inside a [rows][64 B] tile, XOR-swizzled so the
/// 8 rows one ldmatrix phase reads hit 8 distinct bank groups.
#[inline(always)]
fn swz(r: u32, c: u32) -> u32 {
    r * 64 + ((c ^ ((r >> 1) & 3)) << 4)
}

/// `64 * Q` rows x 64 bytes from `src` (row stride `ld` bytes) at byte column `kb0`; rows >= `rows`
/// zero-filled. 256 threads, Q 16-byte chunks each.
#[inline(always)]
unsafe fn load_codes<const Q: u32>(dst: Shared<u8>, src: *const u8, ld: u32, row0: u32, rows: u32, kb0: u32) {
    let t = tid_x();
    kcommon::unroll!(j in 4 => {
        if (j as u32) < Q {
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
        }
    });
}

/// Stage layout: A codes [128][64] | W codes [128 NH][64] | A scales [128][8] | W scales [128 NH][8].
#[inline(always)]
const fn stage_bytes(fp4: bool, nh: u32) -> u32 {
    CODE_TILE * (1 + nh) + if fp4 { SF_TILE * (1 + nh) } else { 0 }
}

#[inline(always)]
unsafe fn load_stage<const FP4: bool, const NH: u32>(st: Shared<u8>, a: *const u8, sa: *const u8, w: *const u8,
    sw: *const u8, m0: u32, n0: u32, m: u32, n: u32, ld: u32, kt: u32, ktiles: u32) {
    load_codes::<2>(st, a, ld, m0, m, kt * BKB);
    load_codes::<2>(st.byte_add(CODE_TILE), w, ld, n0, n, kt * BKB);
    if NH == 2 {
        load_codes::<2>(st.byte_add(2 * CODE_TILE), w, ld, n0 + 128, n, kt * BKB);
    }
    if FP4 {
        // 64 x 16 B per 128-row scale tile: threads 0..63 A, then NH tiles of W
        let t = tid_x();
        if t < 64 * (1 + NH) {
            let sf0 = CODE_TILE * (1 + NH);
            let q = t >> 6; // 0 = A, 1.. = W row tile q-1
            let (src, rt) = if q == 0 { (sa, m0 >> 7) } else { (sw, (n0 >> 7) + q - 1) };
            let o = (t & 63) * 16;
            let tile = (rt * ktiles + kt) as usize * SF_TILE as usize;
            cp_async_16(st.byte_add(sf0 + q * SF_TILE + o).addr(), src.add(tile + o as usize));
        }
    }
}

#[inline(always)]
fn wait_stages<const STAGES: u32>() {
    if STAGES == 4 {
        cp_async_wait::<2>();
    } else if STAGES == 3 {
        cp_async_wait::<1>();
    } else {
        cp_async_wait::<0>();
    }
}

/// Shared body of the NVFP4 (FP4 = true) and FP8 GEMMs. CTA tile 128 x 128*NH, block = 256 (8 warps,
/// 2 along M x 4 along N, warp tile 64 x 32*NH), STAGES-deep cp.async ring, 1-D grid of
/// ceil(M/128) * N/(128 NH) CTAs in a grouped raster.
#[inline(always)]
unsafe fn gemm<const FP4: bool, const NH: u32, const STAGES: u32, const NOLOAD: bool, const SWIGLU: bool, const BS: bool>(c: *mut u16,
    a: *const u8, sa: *const u8, w: *const u8, sw: *const u8, alpha: *const f32, res: *const u16, m: u32, n: u32, k: u32,
    ldc: u32, group_m: u32, q_codes: *mut u8, q_sf: *mut u8, qg: f32) {
    let smem: Shared<u8> = dyn_base();
    let sb_bytes = stage_bytes(FP4, NH);
    let bn = 128 * NH;
    let ld = if FP4 { k >> 1 } else { k };
    let ktiles = ld / BKB;

    // grouped raster: `group_m` row tiles (row tiles fastest) sweep all column tiles before moving on, so
    // resident CTAs share weight panels in L2 while the group's activation panels stay resident
    let tiles_m = m.div_ceil(BM);
    let tiles_n = n / bn;
    let pid = bid_x();
    let per_group = group_m * tiles_n;
    let first_m = (pid / per_group) * group_m;
    let gsize = if tiles_m - first_m < group_m { tiles_m - first_m } else { group_m };
    let r = pid % per_group;
    let m0 = (first_m + r % gsize) * BM;
    let n0 = (r / gsize) * bn;

    let wid = warp_id();
    let l = lane();
    let wm = wid >> 2;
    let wn = wid & 3;

    // acc[mt][j]: m16 tile mt, n8 tile j (j < 4 NH)
    let mut acc = [[[0.0f32; 4]; 8]; 4];

    let mut s = 0;
    while s < STAGES - 1 {
        if s < ktiles {
            load_stage::<FP4, NH>(smem.byte_add(s * sb_bytes), a, sa, w, sw, m0, n0, m, n, ld, s, ktiles);
        }
        cp_async_commit();
        s += 1;
    }

    // per-lane scale offsets inside a stage's scale tiles ([128][8] for A, [128 NH][8] for W)
    let sf0 = CODE_TILE * (1 + NH);
    let sfa_off = sf0 + (wm * 64 + (l >> 2) + 8 * (l & 1)) * 8;
    let sfb_off = sf0 + SF_TILE + (wn * 32 * NH + (l >> 2)) * 8;

    let mut kt = 0;
    while kt < ktiles {
        wait_stages::<STAGES>();
        sync_threads();
        let pf = kt + STAGES - 1;
        if pf < ktiles && !NOLOAD {
            load_stage::<FP4, NH>(smem.byte_add((pf % STAGES) * sb_bytes), a, sa, w, sw, m0, n0, m, n, ld, pf, ktiles);
        }
        cp_async_commit();

        let st = smem.byte_add((kt % STAGES) * sb_bytes);
        let sb = st.byte_add(CODE_TILE);
        kcommon::unroll!(ks in 2 => {
            let kc2 = ks as u32 * 2;
            let mut af = [[0u32; 4]; 4];
            kcommon::unroll!(mt in 4 => {
                let r = wm * 64 + mt as u32 * 16 + (l & 15);
                af[mt] = ldmatrix_x4(st.byte_add(swz(r, kc2 + (l >> 4))).addr());
            });
            let mut bf = [[0u32; 2]; 8];
            kcommon::unroll!(p in 4 => {
                if (p as u32) < 2 * NH {
                    let r = wn * 32 * NH + (2 * p as u32 + (l >> 4)) * 8 + (l & 7);
                    let v = ldmatrix_x4(sb.byte_add(swz(r, kc2 + ((l >> 3) & 1))).addr());
                    bf[2 * p] = [v[0], v[1]];
                    bf[2 * p + 1] = [v[2], v[3]];
                }
            });
            if FP4 {
                let ssa: Shared<u32> = st.byte_add(sfa_off + ks as u32 * 4).cast();
                let ssb: Shared<u32> = st.byte_add(sfb_off + ks as u32 * 4).cast();
                let mut sfa = [0u32; 4];
                let mut sfb = [0u32; 8];
                kcommon::unroll!(i in 4 => {
                    sfa[i] = ssa.byte_add(i as u32 * 16 * 8).ld();
                });
                kcommon::unroll!(j in 8 => {
                    if (j as u32) < 4 * NH {
                        sfb[j] = ssb.byte_add(j as u32 * 8 * 8).ld();
                    }
                });
                kcommon::unroll!(mt in 4 => {
                    kcommon::unroll!(j in 8 => {
                        if (j as u32) < 4 * NH {
                            mma_nvfp4(&mut acc[mt][j], af[mt], bf[j], sfa[mt], sfb[j]);
                        }
                    });
                });
            } else {
                kcommon::unroll!(mt in 4 => {
                    kcommon::unroll!(j in 8 => {
                        if (j as u32) < 4 * NH {
                            if BS { mma_fp8_bs(&mut acc[mt][j], af[mt], bf[j]) } else { mma_fp8(&mut acc[mt][j], af[mt], bf[j]) }
                        }
                    });
                });
            }
        });
        kt += 1;
    }
    cp_async_wait_all();
    if SWIGLU {
        swiglu_epilogue(acc, alpha, m, n, m0, n0, wm, wn, l, q_codes, q_sf, qg);
        return;
    }
    sync_threads();
    epilogue_staged::<NH>(acc, smem, c, alpha, res, m, m0, n0, wm, wn, l, ldc);
}

/// Row stride (bytes) of the bf16 output staging tile: 256 B per 128 columns + 16 B pad (rows hit
/// distinct banks for the fragment stores).
const fn stage_ld(nh: u32) -> u32 {
    256 * nh + 16
}

/// Epilogue through shared memory: fragments (x alpha, rounded to bf16) go to a [128][128 NH] smem tile, then
/// each thread writes whole 16-byte row chunks (coalesced, residual added in f32 and rounded again like the
/// direct path: y = bf16(bf16(acc * alpha) + res)). The caller has synced (stage memory is free).
#[inline(always)]
unsafe fn epilogue_staged<const NH: u32>(acc: [[[f32; 4]; 8]; 4], smem: Shared<u8>, c: *mut u16, alpha: *const f32,
    res: *const u16, m: u32, m0: u32, n0: u32, wm: u32, wn: u32, l: u32, ldc: u32) {
    let sld = stage_ld(NH);
    let g = l >> 2;
    let t4 = l & 3;
    kcommon::unroll!(j in 8 => {
        if (j as u32) < 4 * NH {
            let cl = wn * 32 * NH + j as u32 * 8 + 2 * t4; // column inside the tile
            let al = *(alpha.add((n0 + cl) as usize) as *const [f32; 2]);
            kcommon::unroll!(mt in 4 => {
                kcommon::unroll!(half in 2 => {
                    let r = wm * 64 + mt as u32 * 16 + g + half as u32 * 8;
                    let v = bf16::pack2(acc[mt][j][half * 2] * al[0], acc[mt][j][half * 2 + 1] * al[1]);
                    smem.byte_add(r * sld + cl * 2).cast::<u32>().st(v);
                });
            });
        }
    });
    sync_threads();
    // 128 rows x (16 NH) chunks of 16 B
    let cpr = 16 * NH;
    let mut i = tid_x();
    while i < 128 * cpr {
        let r = i / cpr;
        let ch = i - r * cpr;
        let row = m0 + r;
        if row < m {
            let v: [u32; 4] = smem.byte_add(r * sld + ch * 16).cast::<[u32; 4]>().ld();
            let off = row as usize * ldc as usize + (n0 + ch * 8) as usize;
            let out = if res.is_null() {
                v
            } else {
                let rv = ldg_nc_v4(res.add(off) as *const u32);
                let mut o = [0u32; 4];
                kcommon::unroll!(q in 4 => {
                    o[q] = bf16::pack2(bf16::lo(v[q]) + bf16::lo(rv[q]), bf16::hi(v[q]) + bf16::hi(rv[q]));
                });
                o
            };
            stg_v4(c.add(off) as *mut u32, out);
        }
        i += 256;
    }
}

/// NVFP4 x NVFP4 -> bf16. `a`/`w`: e2m1 codes [rows][K/2]; `sa`/`sw`: e4m3 scales in the tiled layout
/// (row count padded to 128); `alpha[n]` = act global scale * weight global scale per output column;
/// `res` optional (stride ldc, may equal `c`); `group_m` = raster group height in row tiles.
/// N % 128 == 0, K % 128 == 0. grid = ceil(M/128) * N/128, block = 256, smem = FP4_SMEM.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_nvfp4(c: *mut u16, a: *const u8, sa: *const u8, w: *const u8,
    sw: *const u8, alpha: *const f32, res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32) {
    gemm::<true, 1, 4, false, false, false>(c, a, sa, w, sw, alpha, res, m, n, k, ldc, group_m, core::ptr::null_mut(), core::ptr::null_mut(), 0.0);
}
/// 128 x 256 tiles, 3 stages: N % 256 == 0, grid = ceil(M/128) * N/256, smem = FP4_SMEM_N256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_nvfp4_n256(c: *mut u16, a: *const u8, sa: *const u8, w: *const u8,
    sw: *const u8, alpha: *const f32, res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32) {
    gemm::<true, 2, 3, false, false, false>(c, a, sa, w, sw, alpha, res, m, n, k, ldc, group_m, core::ptr::null_mut(), core::ptr::null_mut(), 0.0);
}

/// FP8 (e4m3) x FP8 -> bf16, per-column `alpha`. N % 128 == 0, K % 64 == 0.
/// grid = ceil(M/128) * N/128, block = 256, smem = FP8_SMEM.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_fp8(c: *mut u16, a: *const u8, w: *const u8, alpha: *const f32,
    res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32) {
    gemm::<false, 1, 4, false, false, false>(c, a, core::ptr::null(), w, core::ptr::null(), alpha, res, m, n, k, ldc, group_m, core::ptr::null_mut(), core::ptr::null_mut(), 0.0);
}
/// 128 x 256 tiles, 3 stages: N % 256 == 0, smem = FP8_SMEM_N256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_fp8_n256(c: *mut u16, a: *const u8, w: *const u8, alpha: *const f32,
    res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32) {
    gemm::<false, 2, 3, false, false, false>(c, a, core::ptr::null(), w, core::ptr::null(), alpha, res, m, n, k, ldc, group_m, core::ptr::null_mut(), core::ptr::null_mut(), 0.0);
}

// ---- mbarrier-pipelined variant -------------------------------------------------------------------

#[inline(always)]
fn mbar_init(addr: u32, count: u32) {
    unsafe { asm!("mbarrier.init.shared::cta.b64 [{a}], {n};", a = in(reg32) addr, n = in(reg32) count, options(nostack)) }
}
#[inline(always)]
fn mbar_arrive(addr: u32) {
    unsafe { asm!("mbarrier.arrive.shared::cta.b64 _, [{a}];", a = in(reg32) addr, options(nostack)) }
}
/// Make the current phase of `addr` wait for this thread's prior cp.async to land, then arrive: the
/// cuda::barrier pattern (`cp.async.mbarrier.arrive` adds a pending arrival that the copies' completion
/// releases, then an ordinary arrive), which compute-sanitizer racecheck tracks; the barrier counts one
/// arrival per thread.
#[inline(always)]
fn mbar_cp_async_arrive(addr: u32) {
    unsafe {
        asm!("cp.async.mbarrier.arrive.shared::cta.b64 [{a}];\n\tmbarrier.arrive.shared::cta.b64 _, [{a}];",
            a = in(reg32) addr, options(nostack))
    }
}
/// Block until the phase with parity `parity` of the mbarrier at `addr` has completed.
#[inline(always)]
fn mbar_wait(addr: u32, parity: u32) {
    loop {
        let done: u32;
        unsafe {
            asm!("{{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [{a}], {ph};\n\tselp.u32 {d}, 1, 0, p;\n\t}}",
                a = in(reg32) addr, ph = in(reg32) parity, d = out(reg32) done, options(nostack));
        }
        if done != 0 {
            break;
        }
    }
}

/// 128 x 256 tile, 8 warps (2 x 4, warp tile 64 x 64) like `gemm::<_, 2, STAGES>`, but stages are handed over
/// with mbarriers instead of a CTA-wide barrier: every thread issues its share of a stage's cp.async and
/// arrives (noinc) on full[s]; each warp arrives on empty[s] when done with it; a stage is refilled after
/// computing (for k-tile kt + STAGES - 1) once every warp released it. Warps drift up to a k-tile apart, so
/// one warp's fragment-load bubble overlaps another's MMAs. block = 256, smem = STAGES * stage + 64.
#[inline(always)]
unsafe fn gemm_mb<const FP4: bool, const STAGES: u32, const SWIGLU: bool>(c: *mut u16, a: *const u8, sa: *const u8,
    w: *const u8, sw: *const u8, alpha: *const f32, res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32,
    q_codes: *mut u8, q_sf: *mut u8, qg: f32) {
    const NH: u32 = 2;
    let smem: Shared<u8> = dyn_base();
    let sb_bytes = stage_bytes(FP4, NH);
    let bn = 128 * NH;
    let ld = if FP4 { k >> 1 } else { k };
    let ktiles = ld / BKB;

    let tiles_m = m.div_ceil(BM);
    let tiles_n = n / bn;
    let pid = bid_x();
    let per_group = group_m * tiles_n;
    let first_m = (pid / per_group) * group_m;
    let gsize = if tiles_m - first_m < group_m { tiles_m - first_m } else { group_m };
    let r = pid % per_group;
    let m0 = (first_m + r % gsize) * BM;
    let n0 = (r / gsize) * bn;

    // full[s] at bars + 8 s (256 async arrivals), empty[s] at bars + 8 (STAGES + s) (8 warp arrivals)
    let bars = smem.byte_add(STAGES * sb_bytes).addr();
    if tid_x() == 0 {
        let mut s = 0;
        while s < STAGES {
            mbar_init(bars + 8 * s, 256);
            mbar_init(bars + 8 * (STAGES + s), 8);
            s += 1;
        }
        asm!("fence.mbarrier_init.release.cluster;", options(nostack));
    }
    sync_threads();

    let mut s = 0;
    while s < STAGES - 1 {
        if s < ktiles {
            load_stage::<FP4, NH>(smem.byte_add(s * sb_bytes), a, sa, w, sw, m0, n0, m, n, ld, s, ktiles);
        }
        mbar_cp_async_arrive(bars + 8 * s);
        s += 1;
    }

    let wid = warp_id();
    let l = lane();
    let wm = wid >> 2;
    let wn = wid & 3;
    let mut acc = [[[0.0f32; 4]; 8]; 4];
    let sf0 = CODE_TILE * (1 + NH);
    let sfa_off = sf0 + (wm * 64 + (l >> 2) + 8 * (l & 1)) * 8;
    let sfb_off = sf0 + SF_TILE + (wn * 32 * NH + (l >> 2)) * 8;

    let mut kt = 0;
    let mut st_i = 0; // kt % STAGES
    let mut use_i = 0; // kt / STAGES
    while kt < ktiles {
        mbar_wait(bars + 8 * st_i, use_i & 1);
        let st = smem.byte_add(st_i * sb_bytes);
        let sb = st.byte_add(CODE_TILE);
        kcommon::unroll!(ks in 2 => {
            let kc2 = ks as u32 * 2;
            let mut af = [[0u32; 4]; 4];
            kcommon::unroll!(mt in 4 => {
                let r = wm * 64 + mt as u32 * 16 + (l & 15);
                af[mt] = ldmatrix_x4(st.byte_add(swz(r, kc2 + (l >> 4))).addr());
            });
            let mut bf = [[0u32; 2]; 8];
            kcommon::unroll!(p in 4 => {
                let r = wn * 64 + (2 * p as u32 + (l >> 4)) * 8 + (l & 7);
                let v = ldmatrix_x4(sb.byte_add(swz(r, kc2 + ((l >> 3) & 1))).addr());
                bf[2 * p] = [v[0], v[1]];
                bf[2 * p + 1] = [v[2], v[3]];
            });
            if FP4 {
                let ssa: Shared<u32> = st.byte_add(sfa_off + ks as u32 * 4).cast();
                let ssb: Shared<u32> = st.byte_add(sfb_off + ks as u32 * 4).cast();
                let mut sfa = [0u32; 4];
                let mut sfb = [0u32; 8];
                kcommon::unroll!(i in 4 => {
                    sfa[i] = ssa.byte_add(i as u32 * 16 * 8).ld();
                });
                kcommon::unroll!(j in 8 => {
                    sfb[j] = ssb.byte_add(j as u32 * 8 * 8).ld();
                });
                kcommon::unroll!(mt in 4 => {
                    kcommon::unroll!(j in 8 => {
                        mma_nvfp4(&mut acc[mt][j], af[mt], bf[j], sfa[mt], sfb[j]);
                    });
                });
            } else {
                kcommon::unroll!(mt in 4 => {
                    kcommon::unroll!(j in 8 => {
                        mma_fp8(&mut acc[mt][j], af[mt], bf[j]);
                    });
                });
            }
        });
        sync_warp();
        if l == 0 {
            mbar_arrive(bars + 8 * (STAGES + st_i));
        }
        // refill the stage of k-tile kt - 1 with k-tile kt + STAGES - 1
        let pf = kt + STAGES - 1;
        if pf < ktiles {
            let ps = if st_i == 0 { STAGES - 1 } else { st_i - 1 };
            let pu = if st_i == 0 { use_i } else { use_i + 1 }; // pf / STAGES
            if pu > 0 {
                mbar_wait(bars + 8 * (STAGES + ps), (pu - 1) & 1);
            }
            load_stage::<FP4, NH>(smem.byte_add(ps * sb_bytes), a, sa, w, sw, m0, n0, m, n, ld, pf, ktiles);
            mbar_cp_async_arrive(bars + 8 * ps);
        }
        kt += 1;
        st_i += 1;
        if st_i == STAGES {
            st_i = 0;
            use_i += 1;
        }
    }
    cp_async_wait_all();
    if SWIGLU {
        swiglu_epilogue(acc, alpha, m, n, m0, n0, wm, wn, l, q_codes, q_sf, qg);
        return;
    }

    sync_threads();
    epilogue_staged::<2>(acc, smem, c, alpha, res, m, m0, n0, wm, wn, l, ldc);
}

// ---- fused SwiGLU + NVFP4 quantization (gate|up -> down's input) -------------------------------------

/// Two f32 -> one byte of two e2m1 nibbles (`lo` in the low nibble).
#[inline(always)]
fn to_e2m1x2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe {
        asm!("{{\n\t.reg .b8 q;\n\tcvt.rn.satfinite.e2m1x2.f32 q, {hi}, {lo};\n\tcvt.u32.u8 {r}, q;\n\t}}",
            r = out(reg32) r, lo = in(reg32) lo, hi = in(reg32) hi, options(pure, nomem, nostack));
    }
    r & 0xff
}

/// act = bf16(bf16(silu(g)) * u) with g, u = bf16 GEMM outputs (core's fe_silu_mul semantics).
#[inline(always)]
fn swiglu(g: f32, u: f32) -> f32 {
    bf16::round(bf16::round(math::silu(g)) * u)
}

/// NVFP4-quantize 16 act values held 4 per lane by the 4 lanes of a quad (lane t4: elements 2t4, 2t4+1,
/// 8+2t4, 8+2t4+1 of the block) and store them (lane t4 == 0) at `codes` (8 bytes) / `sf` (1 byte).
/// Every lane must call it (shuffles); `store` gates the writes.
#[inline(always)]
unsafe fn quant_quad(v: [f32; 4], qg: f32, codes: *mut u8, sf: *mut u8, store: bool) {
    let mut amax = v[0].abs().max(v[1].abs()).max(v[2].abs()).max(v[3].abs());
    amax = amax.max(warp::shfl_xor(amax, 1));
    amax = amax.max(warp::shfl_xor(amax, 2));
    let s8 = to_e4m3(amax * (1.0 / 6.0) * qg);
    let sv = e4m3_to_f32(s8);
    let mul = if sv > 0.0 { qg / sv } else { 0.0 };
    let pk = to_e2m1x2(v[0] * mul, v[1] * mul) | to_e2m1x2(v[2] * mul, v[3] * mul) << 8;
    let base = lane() & !3;
    let p1 = warp::shfl_idx_u32(pk, base + 1);
    let p2 = warp::shfl_idx_u32(pk, base + 2);
    let p3 = warp::shfl_idx_u32(pk, base + 3);
    if store {
        let w0 = (pk & 0xff) | (p1 & 0xff) << 8 | (p2 & 0xff) << 16 | (p3 & 0xff) << 24;
        let w1 = (pk >> 8) | (p1 >> 8) << 8 | (p2 >> 8) << 16 | (p3 >> 8) << 24;
        let d = codes as *mut u32;
        *d = w0;
        *d.add(1) = w1;
        *sf = s8 as u8;
    }
}

/// Epilogue of the fused gate|up GEMM: the weight rows are interleaved in blocks of 8 (gate f..f+7, up
/// f..f+7, ...), so n8 tiles 2p / 2p+1 of a warp are gate / up of the same 8 features and each lane holds
/// both halves of its features. Writes act = SwiGLU, NVFP4-quantized with global scale `qg` (down's
/// 1 / input_scale), as codes [m][N/4] + tiled scales (kb = N/32): exactly what `fe_fp4_quant` would make
/// from the bf16 act of the unfused path.
#[inline(always)]
unsafe fn swiglu_epilogue(acc: [[[f32; 4]; 8]; 4], alpha: *const f32, m: u32, n: u32, m0: u32, n0: u32, wm: u32,
    wn: u32, l: u32, q_codes: *mut u8, q_sf: *mut u8, qg: f32) {
    let g = l >> 2;
    let t4 = l & 3;
    let inter = n >> 1;
    let kb = inter >> 4;
    let feat0 = (n0 >> 1) + wn * 32;
    kcommon::unroll!(q in 2 => {
        let mut al = [[0.0f32; 4]; 2];
        kcommon::unroll!(pp in 2 => {
            let cg = (n0 + wn * 64 + (2 * (2 * q as u32 + pp as u32)) * 8 + 2 * t4) as usize;
            al[pp] = [*alpha.add(cg), *alpha.add(cg + 1), *alpha.add(cg + 8), *alpha.add(cg + 9)];
        });
        let fb = feat0 + q as u32 * 16;
        kcommon::unroll!(mt in 4 => {
            kcommon::unroll!(half in 2 => {
                let row = m0 + wm * 64 + mt as u32 * 16 + g + half as u32 * 8;
                let mut v = [0.0f32; 4];
                kcommon::unroll!(pp in 2 => {
                    let p = 2 * q + pp;
                    let ga = acc[mt][2 * p];
                    let ua = acc[mt][2 * p + 1];
                    v[2 * pp] = swiglu(bf16::round(ga[half * 2] * al[pp][0]), bf16::round(ua[half * 2] * al[pp][2]));
                    v[2 * pp + 1] =
                        swiglu(bf16::round(ga[half * 2 + 1] * al[pp][1]), bf16::round(ua[half * 2 + 1] * al[pp][3]));
                });
                let rr = if row < m { row } else { 0 };
                quant_quad(v, qg, q_codes.add(rr as usize * (inter as usize >> 1) + (fb >> 1) as usize),
                    q_sf.add(sf_tiled(rr, fb >> 4, kb) as usize), row < m && t4 == 0);
            });
        });
    });
}

/// Fused MLP front: act = SwiGLU(x . [gate|up]^T) NVFP4-quantized for down_proj. `w`/`sw`/`alpha` = gate and
/// up interleaved in blocks of 8 rows (N = 2 * inter, N % 256 == 0); `q_codes` [M][inter/2], `q_sf` tiled
/// (rows padded to 128, kb = inter/16); `qg` = 1 / down.input_scale. grid = ceil(M/128) * N/256,
/// block = 256, smem = FP4_SMEM_N256. Main loop = the bar.sync / cp.async.wait_group pipeline of
/// `fe_gemm_nvfp4_n256` (compute-sanitizer racecheck clean).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_nvfp4_swiglu(a: *const u8, sa: *const u8, w: *const u8, sw: *const u8,
    alpha: *const f32, m: u32, n: u32, k: u32, group_m: u32, q_codes: *mut u8, q_sf: *mut u8, qg: f32) {
    gemm::<true, 2, 3, false, true, false>(core::ptr::null_mut(), a, sa, w, sw, alpha, core::ptr::null(), m, n, k, 0, group_m,
        q_codes, q_sf, qg);
}
/// The same on the mbarrier pipeline (`fe_gemm_nvfp4_mb`), smem = FP4_SMEM_MB.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_nvfp4_swiglu_mb(a: *const u8, sa: *const u8, w: *const u8, sw: *const u8,
    alpha: *const f32, m: u32, n: u32, k: u32, group_m: u32, q_codes: *mut u8, q_sf: *mut u8, qg: f32) {
    gemm_mb::<true, 3, true>(core::ptr::null_mut(), a, sa, w, sw, alpha, core::ptr::null(), m, n, k, 0, group_m, q_codes,
        q_sf, qg);
}

/// Unfused reference / fallback of the epilogue above: y = interleaved gate|up bf16 output [M][N] ->
/// act NVFP4 codes + tiled scales. One thread per 16 features (two 8-feature pairs).
/// grid = ceil(M * N/32 / 256), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_fp4_swiglu_quant(q_codes: *mut u8, q_sf: *mut u8, y: *const u16, m: u32, n: u32,
    qg: f32) {
    let i = bid_x() * bdim_x() + tid_x();
    let inter = n >> 1;
    let kb = inter >> 4;
    if i >= m * kb {
        return;
    }
    let row = i / kb;
    let b = i - row * kb;
    // features 16b .. 16b+15: pairs at interleaved columns 32b (gate 0..7, up 0..7) and 32b + 16
    let src = y.add(row as usize * n as usize + b as usize * 32) as *const u32;
    let mut v = [0.0f32; 16];
    kcommon::unroll!(h in 2 => {
        let gv = bf16::unpack8(ldg_nc_v4(src.add(h * 8)));
        let uv = bf16::unpack8(ldg_nc_v4(src.add(h * 8 + 4)));
        kcommon::unroll!(j in 8 => {
            v[h * 8 + j] = swiglu(gv[j], uv[j]);
        });
    });
    let mut amax = 0.0f32;
    kcommon::unroll!(j in 16 => {
        amax = amax.max(v[j].abs());
    });
    let s8 = to_e4m3(amax * (1.0 / 6.0) * qg);
    let sv = e4m3_to_f32(s8);
    let mul = if sv > 0.0 { qg / sv } else { 0.0 };
    let mut y0 = [0.0f32; 8];
    let mut y1 = [0.0f32; 8];
    kcommon::unroll!(j in 8 => {
        y0[j] = v[j] * mul;
        y1[j] = v[8 + j] * mul;
    });
    let d = q_codes.add(row as usize * (inter as usize >> 1) + b as usize * 8) as *mut u32;
    *d = to_e2m1x8(y0);
    *d.add(1) = to_e2m1x8(y1);
    *q_sf.add(sf_tiled(row, b, kb) as usize) = s8 as u8;
}

/// FP8, 128 x 128 tiles, 3 stages (48 KB: two CTAs a SM interleave their barrier bubbles).
/// grid = ceil(M/128) * N/128, block = 256, smem = FP8_SMEM_S3.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_fp8_s3(c: *mut u16, a: *const u8, w: *const u8, alpha: *const f32,
    res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32) {
    gemm::<false, 1, 3, false, false, false>(c, a, core::ptr::null(), w, core::ptr::null(), alpha, res, m, n, k, ldc, group_m,
        core::ptr::null_mut(), core::ptr::null_mut(), 0.0);
}
/// `fe_gemm_fp8_s3` on the block-scaled tensor-core form with unit scales (`mma_fp8_bs`): the same products, twice
/// the tensor-core rate on RTX 50 cards (fp4-peak). Same arguments, grid and smem.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_fp8_s3_bs(c: *mut u16, a: *const u8, w: *const u8, alpha: *const f32,
    res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32) {
    gemm::<false, 1, 3, false, false, true>(c, a, core::ptr::null(), w, core::ptr::null(), alpha, res, m, n, k, ldc, group_m,
        core::ptr::null_mut(), core::ptr::null_mut(), 0.0);
}
/// `fe_gemm_fp8_n256` on the block-scaled form (128 x 256 tiles, one CTA a SM). smem = FP8_SMEM_N256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_fp8_n256_bs(c: *mut u16, a: *const u8, w: *const u8, alpha: *const f32,
    res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32) {
    gemm::<false, 2, 3, false, false, true>(c, a, core::ptr::null(), w, core::ptr::null(), alpha, res, m, n, k, ldc, group_m,
        core::ptr::null_mut(), core::ptr::null_mut(), 0.0);
}
/// NVFP4, 128 x 128 tiles, 2 stages (36 KB: two CTAs a SM). smem = FP4_SMEM_S2.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_nvfp4_s2(c: *mut u16, a: *const u8, sa: *const u8, w: *const u8,
    sw: *const u8, alpha: *const f32, res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32) {
    gemm::<true, 1, 2, false, false, false>(c, a, sa, w, sw, alpha, res, m, n, k, ldc, group_m, core::ptr::null_mut(),
        core::ptr::null_mut(), 0.0);
}

/// mbarrier-pipelined NVFP4 GEMM (128 x 256 tiles): same arguments as `fe_gemm_nvfp4`, N % 256 == 0,
/// grid = ceil(M/128) * N/256, block = 256, smem = FP4_SMEM_MB.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_nvfp4_mb(c: *mut u16, a: *const u8, sa: *const u8, w: *const u8,
    sw: *const u8, alpha: *const f32, res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32) {
    gemm_mb::<true, 3, false>(c, a, sa, w, sw, alpha, res, m, n, k, ldc, group_m, core::ptr::null_mut(),
        core::ptr::null_mut(), 0.0);
}
/// mbarrier-pipelined FP8 GEMM (128 x 256 tiles), block = 256, smem = FP8_SMEM_MB.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_fp8_mb(c: *mut u16, a: *const u8, w: *const u8, alpha: *const f32,
    res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32) {
    gemm_mb::<false, 3, false>(c, a, core::ptr::null(), w, core::ptr::null(), alpha, res, m, n, k, ldc, group_m,
        core::ptr::null_mut(), core::ptr::null_mut(), 0.0);
}
pub const FP4_SMEM_MB: u32 = 3 * stage_bytes(true, 2) + 64;
pub const FP8_SMEM_MB: u32 = 3 * stage_bytes(false, 2) + 64;

/// Diagnostic: `fe_gemm_nvfp4_n256` without main-loop loads (computes on the prologue's stale stages), to
/// separate the inner-loop/barrier cost from the memory pipeline. Output is garbage.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_gemm_nvfp4_n256_noload(c: *mut u16, a: *const u8, sa: *const u8, w: *const u8,
    sw: *const u8, alpha: *const f32, res: *const u16, m: u32, n: u32, k: u32, ldc: u32, group_m: u32) {
    gemm::<true, 2, 3, true, false, false>(c, a, sa, w, sw, alpha, res, m, n, k, ldc, group_m, core::ptr::null_mut(), core::ptr::null_mut(), 0.0);
}

// ---- peak probe -------------------------------------------------------------------------------

/// Tensor-core peak probe: each warp issues `iters` x 16 independent mma (4 x 4 accumulator tiles) from
/// registers, no memory traffic. `fp4` != 0: NVFP4 m16n8k64, else FP8 m16n8k32. Writes one float per
/// thread so nothing is dead code.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_fp4_peak(out: *mut f32, iters: u32, fp4: u32) {
    let l = lane();
    let a = [0x22222222u32 ^ l, 0x11111111, 0x22222222, 0x11111111];
    let b = [0x22222222u32, 0x11111111 ^ l];
    let mut acc = [[[0.0f32; 4]; 4]; 4];
    let mut i = 0;
    if fp4 == 3 {
        while i < iters {
            kcommon::unroll!(mt in 4 => {
                kcommon::unroll!(nt in 4 => {
                    mma_fp8_bs(&mut acc[mt][nt], a, b);
                });
            });
            i += 1;
        }
    } else if fp4 == 2 {
        while i < iters {
            kcommon::unroll!(mt in 4 => {
                kcommon::unroll!(nt in 4 => {
                    mma_fp8_legacy(&mut acc[mt][nt], a, b);
                });
            });
            i += 1;
        }
    } else if fp4 != 0 {
        while i < iters {
            kcommon::unroll!(mt in 4 => {
                kcommon::unroll!(nt in 4 => {
                    mma_nvfp4(&mut acc[mt][nt], a, b, 0x38383838, 0x38383838);
                });
            });
            i += 1;
        }
    } else {
        while i < iters {
            kcommon::unroll!(mt in 4 => {
                kcommon::unroll!(nt in 4 => {
                    mma_fp8(&mut acc[mt][nt], a, b);
                });
            });
            i += 1;
        }
    }
    let mut s = 0.0f32;
    kcommon::unroll!(mt in 4 => {
        kcommon::unroll!(nt in 4 => {
            s += acc[mt][nt][0] + acc[mt][nt][1] + acc[mt][nt][2] + acc[mt][nt][3];
        });
    });
    *out.add((bid_x() * bdim_x() + tid_x()) as usize) = s;
}
