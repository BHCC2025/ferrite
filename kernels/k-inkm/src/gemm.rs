//! Small-M bf16 GEMM (M <= 16 rows) for decode: C[M][N] = A[M][K] . W[N][K]^T, bf16 out. Same mma.m16n8k16 instruction and
//! the same k order as k-core's fe_gemm_bf16, so every element has the same bits as the big-tile kernel gives, but 32-column
//! tiles so a skinny matrix fills the machine. K % 64 == 0.
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

const BN: u32 = 32;
const BK: u32 = 64;
const ST: u32 = 4;
const PITCH: u32 = 144; // 128 B of a 64-k row + 16 B pad: ldmatrix without bank conflicts
const STAGE: u32 = (BN + 16) * PITCH;
pub const SGEMM_SMEM: u32 = ST * STAGE;

#[inline(always)]
unsafe fn load_stage(st: Shared<u8>, a: *const u16, w: *const u16, m: u32, n: u32, k: u32, lda: u32, m0: u32, n0: u32, kt: u32) {
    let t = tid_x();
    let mut i = t;
    while i < BN * 8 {
        let r = i >> 3;
        let c = i & 7;
        let gr = n0 + r;
        let d = st.byte_add(r * PITCH + c * 16).addr();
        if gr < n {
            cp_async_16(d, w.add(gr as usize * k as usize + (kt * BK + c * 8) as usize) as *const u8);
        } else {
            cp_async_16_zfill(d, w as *const u8, 0);
        }
        i += 128;
    }
    let r = t >> 3;
    let c = t & 7;
    let d = st.byte_add(BN * PITCH + r * PITCH + c * 16).addr();
    if m0 + r < m {
        cp_async_16(d, a.add((m0 + r) as usize * lda as usize + (kt * BK + c * 8) as usize) as *const u8);
    } else {
        cp_async_16_zfill(d, a as *const u8, 0);
    }
}

/// grid = (ceil(N / 32), ceil(M / 16)), block = 128, smem = SGEMM_SMEM.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_s(c: *mut u16, a: *const u16, w: *const u16, m: u32, n: u32, k: u32, lda: u32,
    ldc: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let smem: Shared<u8> = dyn_base();
    let n0 = bid_x() * BN;
    let m0 = bid_y() * 16;
    let wn = warp_id();
    let l = lane();
    let ktiles = k / BK;
    let mut acc = [0.0f32; 4];
    let mut s = 0;
    while s < ST - 1 {
        if s < ktiles {
            load_stage(smem.byte_add(s * STAGE), a, w, m, n, k, lda, m0, n0, s);
        }
        cp_async_commit();
        s += 1;
    }
    let mut kt = 0;
    while kt < ktiles {
        cp_async_wait::<{ ST - 2 }>();
        sync_threads();
        let pf = kt + ST - 1;
        if pf < ktiles {
            load_stage(smem.byte_add((pf % ST) * STAGE), a, w, m, n, k, lda, m0, n0, pf);
        }
        cp_async_commit();
        let sw = smem.byte_add((kt % ST) * STAGE);
        let sa = sw.byte_add(BN * PITCH);
        kcommon::unroll!(ks in 4 => {
            let kc2 = ks as u32 * 2;
            let af = ldmatrix_x4(sa.byte_add((l & 15) * PITCH + (kc2 + (l >> 4)) * 16).addr());
            let bfr = ldmatrix_x2(sw.byte_add((wn * 8 + (l & 7)) * PITCH + (kc2 + ((l >> 3) & 1)) * 16).addr());
            mma_bf16_16816(&mut acc, af, bfr);
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
                kcommon::bf16::pack2(acc[half * 2], acc[half * 2 + 1]);
        }
    });
}

// ---- FP8-weight variant ---------------------------------------------------------------------------------------------------
// C = A . W^T with W stored as E4M3 + one f32 scale per output row; the effective weight bf16(q * scale) is formed in shared memory
// and fed to the same mma in the same k order as `ink_gemm_s` / fe_gemm_bf16 on the dequantised matrix: every element has the bits
// of "dequantise to bf16, then bf16 GEMM". Decode rows use it, so a decode row equals the prompt-family row.
const RAWP: u32 = 80; // 64 fp8 + 16 B pad
const RAW_STAGE: u32 = BN * RAWP; // 2560
const A_STAGE: u32 = 16 * PITCH; // 2304
const BF_TILE: u32 = BN * PITCH; // 4608
pub const SGEMM8_SMEM: u32 = ST * (RAW_STAGE + A_STAGE) + 2 * BF_TILE;

#[inline(always)]
fn cvt_e4m3x4(w: u32) -> [f32; 4] {
    let (a0, a1, a2, a3): (f32, f32, f32, f32);
    unsafe {
        core::arch::asm!("{{\n\t.reg .b16 p, q, l, u;\n\t.reg .b32 h0, h1;\n\t\
              mov.b32 {{p, q}}, {w};\n\t\
              cvt.rn.f16x2.e4m3x2 h0, p;\n\tcvt.rn.f16x2.e4m3x2 h1, q;\n\t\
              mov.b32 {{l, u}}, h0;\n\tcvt.f32.f16 {a0}, l;\n\tcvt.f32.f16 {a1}, u;\n\t\
              mov.b32 {{l, u}}, h1;\n\tcvt.f32.f16 {a2}, l;\n\tcvt.f32.f16 {a3}, u;\n\t}}",
            w = in(reg32) w, a0 = out(reg32) a0, a1 = out(reg32) a1, a2 = out(reg32) a2, a3 = out(reg32) a3,
            options(pure, nomem, nostack));
    }
    [a0, a1, a2, a3]
}

#[inline(always)]
unsafe fn load_stage8(raw: Shared<u8>, asm: Shared<u8>, a: *const u16, w8: *const u8, m: u32, n: u32, k: u32, lda: u32, m0: u32,
    n0: u32, kt: u32) {
    let t = tid_x();
    // W: 32 rows x 4 chunks of 16 B = 128 chunks, one a thread
    let r = t >> 2;
    let c = t & 3;
    let gr = n0 + r;
    let d = raw.byte_add(r * RAWP + c * 16).addr();
    if gr < n {
        cp_async_16(d, w8.add(gr as usize * k as usize + (kt * BK + c * 16) as usize));
    } else {
        cp_async_16_zfill(d, w8, 0);
    }
    let r = t >> 3;
    let c = t & 7;
    let d = asm.byte_add(r * PITCH + c * 16).addr();
    if m0 + r < m {
        cp_async_16(d, a.add((m0 + r) as usize * lda as usize + (kt * BK + c * 8) as usize) as *const u8);
    } else {
        cp_async_16_zfill(d, a as *const u8, 0);
    }
}

/// grid = (ceil(N / 32), ceil(M / 16)), block = 128, smem = SGEMM8_SMEM. K % 64 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_s8(c: *mut u16, a: *const u16, w8: *const u8, scale: *const f32, m: u32, n: u32, k: u32,
    lda: u32, ldc: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let smem: Shared<u8> = dyn_base();
    let n0 = bid_x() * BN;
    let m0 = bid_y() * 16;
    let wn = warp_id();
    let l = lane();
    let t = tid_x();
    let ktiles = k / BK;
    let bf_base = smem.byte_add(ST * (RAW_STAGE + A_STAGE));
    let mut acc = [0.0f32; 4];
    // this thread's conversion job: row r8, chunk c8 of the raw tile
    let (r8, c8) = (t >> 2, t & 3);
    let sc = if n0 + r8 < n { *scale.add((n0 + r8) as usize) } else { 0.0 };
    let mut s = 0;
    while s < ST - 1 {
        if s < ktiles {
            let st = smem.byte_add(s * (RAW_STAGE + A_STAGE));
            load_stage8(st, st.byte_add(RAW_STAGE), a, w8, m, n, k, lda, m0, n0, s);
        }
        cp_async_commit();
        s += 1;
    }
    let mut kt = 0;
    while kt < ktiles {
        cp_async_wait::<{ ST - 2 }>();
        sync_threads();
        let pf = kt + ST - 1;
        if pf < ktiles {
            let st = smem.byte_add((pf % ST) * (RAW_STAGE + A_STAGE));
            load_stage8(st, st.byte_add(RAW_STAGE), a, w8, m, n, k, lda, m0, n0, pf);
        }
        cp_async_commit();
        let st = smem.byte_add((kt % ST) * (RAW_STAGE + A_STAGE));
        let raw = st;
        let asm = st.byte_add(RAW_STAGE);
        // convert this thread's 16 fp8 to 16 bf16 (bf16(q * scale)) into the bf16 weight tile
        let rw: [u32; 4] = <[u32; 4] as kcommon::smem::SmemIo>::lds(raw.byte_add(r8 * RAWP + c8 * 16).addr());
        let wb = bf_base.byte_add((kt & 1) * BF_TILE);
        let mut o = [0u32; 8];
        kcommon::unroll!(i in 4 => {
            let v = cvt_e4m3x4(rw[i]);
            o[2 * i] = kcommon::bf16::pack2(v[0] * sc, v[1] * sc);
            o[2 * i + 1] = kcommon::bf16::pack2(v[2] * sc, v[3] * sc);
        });
        <[u32; 4] as kcommon::smem::SmemIo>::sts(wb.byte_add(r8 * PITCH + c8 * 32).addr(), [o[0], o[1], o[2], o[3]]);
        <[u32; 4] as kcommon::smem::SmemIo>::sts(wb.byte_add(r8 * PITCH + c8 * 32 + 16).addr(), [o[4], o[5], o[6], o[7]]);
        sync_threads();
        kcommon::unroll!(ks in 4 => {
            let kc2 = ks as u32 * 2;
            let af = ldmatrix_x4(asm.byte_add((l & 15) * PITCH + (kc2 + (l >> 4)) * 16).addr());
            let bfr = ldmatrix_x2(wb.byte_add((wn * 8 + (l & 7)) * PITCH + (kc2 + ((l >> 3) & 1)) * 16).addr());
            mma_bf16_16816(&mut acc, af, bfr);
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
                kcommon::bf16::pack2(acc[half * 2], acc[half * 2 + 1]);
        }
    });
}

// ---- wide-K variants (decode on bandwidth-bound chips) ---------------------------------------------------------------------
// Same tiles, the same mma.m16n8k16 sequence and the same k order per output element as `ink_gemm_s` / `ink_gemm_s8` (k = 0,
// 16, 32, ... in one accumulator per warp), so every element has the same bits; only the k tile a pipeline stage carries is 256
// wide instead of 64. A 64-wide stage leaves too few bytes in flight on a chip whose memory latency is long (DGX Spark: the
// 1024-column K/V projections ran at ~35 GB/s). K % 256 == 0.
const WBK: u32 = 256;
const WST: u32 = 3;
const WCH: u32 = WBK / 8; // 16 B chunks of a bf16 row
const WCH8: u32 = WBK / 16; // 16 B chunks of an fp8 row
const WJOBS: u32 = BN * WCH8 / 128; // conversion jobs a thread
const WPITCH: u32 = WBK * 2 + 16; // 528: bf16 row of a 256-k tile + 16 B pad
const WRAWP: u32 = WBK + 16; // 272: fp8 row + 16 B pad
const WSTAGE: u32 = (BN + 16) * WPITCH;
pub const SGEMMW_SMEM: u32 = WST * WSTAGE;
const W8_RAW: u32 = BN * WRAWP;
const W8_A: u32 = 16 * WPITCH;
const W8_BF: u32 = BN * WPITCH;
pub const SGEMM8W_SMEM: u32 = WST * (W8_RAW + W8_A) + 2 * W8_BF;

#[inline(always)]
unsafe fn load_a_w(asm: Shared<u8>, a: *const u16, m: u32, lda: u32, m0: u32, kt: u32) {
    let mut i = tid_x();
    while i < 16 * WCH {
        let r = i / WCH;
        let c = i % WCH;
        let d = asm.byte_add(r * WPITCH + c * 16).addr();
        if m0 + r < m {
            cp_async_16(d, a.add((m0 + r) as usize * lda as usize + (kt * WBK + c * 8) as usize) as *const u8);
        } else {
            cp_async_16_zfill(d, a as *const u8, 0);
        }
        i += 128;
    }
}

#[inline(always)]
unsafe fn load_stage_w(st: Shared<u8>, a: *const u16, w: *const u16, m: u32, n: u32, k: u32, lda: u32, m0: u32, n0: u32, kt: u32) {
    let mut i = tid_x();
    while i < BN * WCH {
        let r = i / WCH;
        let c = i % WCH;
        let gr = n0 + r;
        let d = st.byte_add(r * WPITCH + c * 16).addr();
        if gr < n {
            cp_async_16(d, w.add(gr as usize * k as usize + (kt * WBK + c * 8) as usize) as *const u8);
        } else {
            cp_async_16_zfill(d, w as *const u8, 0);
        }
        i += 128;
    }
    load_a_w(st.byte_add(BN * WPITCH), a, m, lda, m0, kt);
}

/// `ink_gemm_s` with 256-wide k stages. grid = (ceil(N / 32), ceil(M / 16)), block = 128, smem = SGEMMW_SMEM. K % 256 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_sw(c: *mut u16, a: *const u16, w: *const u16, m: u32, n: u32, k: u32, lda: u32,
    ldc: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let smem: Shared<u8> = dyn_base();
    let n0 = bid_x() * BN;
    let m0 = bid_y() * 16;
    let wn = warp_id();
    let l = lane();
    let ktiles = k / WBK;
    let mut acc = [0.0f32; 4];
    let mut s = 0;
    while s < WST - 1 {
        if s < ktiles {
            load_stage_w(smem.byte_add(s * WSTAGE), a, w, m, n, k, lda, m0, n0, s);
        }
        cp_async_commit();
        s += 1;
    }
    let mut kt = 0;
    while kt < ktiles {
        cp_async_wait::<{ WST - 2 }>();
        sync_threads();
        let pf = kt + WST - 1;
        if pf < ktiles {
            load_stage_w(smem.byte_add((pf % WST) * WSTAGE), a, w, m, n, k, lda, m0, n0, pf);
        }
        cp_async_commit();
        let sw = smem.byte_add((kt % WST) * WSTAGE);
        let sa = sw.byte_add(BN * WPITCH);
        let mut ks = 0u32;
        while ks < WBK / 16 {
            let kc2 = ks * 2;
            let af = ldmatrix_x4(sa.byte_add((l & 15) * WPITCH + (kc2 + (l >> 4)) * 16).addr());
            let bfr = ldmatrix_x2(sw.byte_add((wn * 8 + (l & 7)) * WPITCH + (kc2 + ((l >> 3) & 1)) * 16).addr());
            mma_bf16_16816(&mut acc, af, bfr);
            ks += 1;
        }
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
                kcommon::bf16::pack2(acc[half * 2], acc[half * 2 + 1]);
        }
    });
}

#[inline(always)]
unsafe fn load_stage8_w(raw: Shared<u8>, asm: Shared<u8>, a: *const u16, w8: *const u8, m: u32, n: u32, k: u32, lda: u32, m0: u32,
    n0: u32, kt: u32) {
    let mut i = tid_x();
    while i < BN * WCH8 {
        let r = i / WCH8;
        let c = i % WCH8;
        let gr = n0 + r;
        let d = raw.byte_add(r * WRAWP + c * 16).addr();
        if gr < n {
            cp_async_16(d, w8.add(gr as usize * k as usize + (kt * WBK + c * 16) as usize));
        } else {
            cp_async_16_zfill(d, w8, 0);
        }
        i += 128;
    }
    load_a_w(asm, a, m, lda, m0, kt);
}

/// `ink_gemm_s8` with 256-wide k stages. grid = (ceil(N / 32), ceil(M / 16)), block = 128, smem = SGEMM8W_SMEM. K % 256 == 0.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_gemm_s8w(c: *mut u16, a: *const u16, w8: *const u8, scale: *const f32, m: u32, n: u32, k: u32,
    lda: u32, ldc: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let smem: Shared<u8> = dyn_base();
    let n0 = bid_x() * BN;
    let m0 = bid_y() * 16;
    let wn = warp_id();
    let l = lane();
    let t = tid_x();
    let ktiles = k / WBK;
    let bf_base = smem.byte_add(WST * (W8_RAW + W8_A));
    let mut acc = [0.0f32; 4];
    // conversion jobs: chunk j = t + 128 * q of the BN x WCH8 raw chunks: row (t / WCH8) + (128 / WCH8) q, chunk t % WCH8
    let c16 = t % WCH8;
    let r0 = t / WCH8;
    let mut sc = [0.0f32; 4];
    kcommon::unroll!(q in 4 => {
        let r = r0 + (128 / WCH8) * q as u32;
        if (q as u32) < WJOBS {
            sc[q] = if n0 + r < n { *scale.add((n0 + r) as usize) } else { 0.0 };
        }
    });
    let mut s = 0;
    while s < WST - 1 {
        if s < ktiles {
            let st = smem.byte_add(s * (W8_RAW + W8_A));
            load_stage8_w(st, st.byte_add(W8_RAW), a, w8, m, n, k, lda, m0, n0, s);
        }
        cp_async_commit();
        s += 1;
    }
    let mut kt = 0;
    while kt < ktiles {
        cp_async_wait::<{ WST - 2 }>();
        sync_threads();
        let pf = kt + WST - 1;
        if pf < ktiles {
            let st = smem.byte_add((pf % WST) * (W8_RAW + W8_A));
            load_stage8_w(st, st.byte_add(W8_RAW), a, w8, m, n, k, lda, m0, n0, pf);
        }
        cp_async_commit();
        let st = smem.byte_add((kt % WST) * (W8_RAW + W8_A));
        let raw = st;
        let asm = st.byte_add(W8_RAW);
        let wb = bf_base.byte_add((kt & 1) * W8_BF);
        kcommon::unroll!(q in 4 => {
          if (q as u32) < WJOBS {
            let r = r0 + (128 / WCH8) * q as u32;
            let rw: [u32; 4] = <[u32; 4] as kcommon::smem::SmemIo>::lds(raw.byte_add(r * WRAWP + c16 * 16).addr());
            let mut o = [0u32; 8];
            kcommon::unroll!(i in 4 => {
                let v = cvt_e4m3x4(rw[i]);
                o[2 * i] = kcommon::bf16::pack2(v[0] * sc[q], v[1] * sc[q]);
                o[2 * i + 1] = kcommon::bf16::pack2(v[2] * sc[q], v[3] * sc[q]);
            });
            <[u32; 4] as kcommon::smem::SmemIo>::sts(wb.byte_add(r * WPITCH + c16 * 32).addr(), [o[0], o[1], o[2], o[3]]);
            <[u32; 4] as kcommon::smem::SmemIo>::sts(wb.byte_add(r * WPITCH + c16 * 32 + 16).addr(), [o[4], o[5], o[6], o[7]]);
          }
        });
        sync_threads();
        let mut ks = 0u32;
        while ks < WBK / 16 {
            let kc2 = ks * 2;
            let af = ldmatrix_x4(asm.byte_add((l & 15) * WPITCH + (kc2 + (l >> 4)) * 16).addr());
            let bfr = ldmatrix_x2(wb.byte_add((wn * 8 + (l & 7)) * WPITCH + (kc2 + ((l >> 3) & 1)) * 16).addr());
            mma_bf16_16816(&mut acc, af, bfr);
            ks += 1;
        }
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
                kcommon::bf16::pack2(acc[half * 2], acc[half * 2 + 1]);
        }
    });
}
