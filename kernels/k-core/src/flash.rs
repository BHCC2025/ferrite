//! Causal flash attention for prompt rows on tensor cores (FA2 style, head dim 256).
//!
//! One block = 4 warps = 64 query rows of one q head of one segment; each warp owns 16 rows. K/V come
//! from the slot's cache in tiles of 32 keys (double-buffered cp.async), Q stays in shared memory.
//! S = Q K^T and O += P V use mma.sync m16n8k16 bf16 with f32 accumulators; P is re-packed from the S
//! accumulators straight into A fragments. The epilogue applies the sigmoid output gate.
//! Shared memory: Q 32 KB + 2 x (K 16 KB + V 16 KB) = 96 KB (opt-in).
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

const HD: u32 = 256;
const NKV: u32 = crate::dims::NKV;
const GRP: u32 = crate::dims::NQ / crate::dims::NKV;
const NQ: u32 = crate::dims::NQ;
const BQ: u32 = 64;
const BKV: u32 = 32;
const ROWB: u32 = HD * 2; // 512 bytes per row
pub const FLASH_SMEM: u32 = BQ * ROWB + 2 * 2 * BKV * ROWB; // 98304

/// Byte offset of 16-byte chunk `c` (0..32) of row `r` in a [rows][256] bf16 tile, XOR-swizzled.
#[inline(always)]
pub(crate) fn sw(r: u32, c: u32) -> u32 {
    r * ROWB + ((c ^ (r & 7)) << 4)
}

/// Load `rows` rows of 256 bf16 (row stride `ld` elements) into a swizzled tile; rows >= `valid` zero-fill.
#[inline(always)]
pub(crate) unsafe fn load_rows(dst: Shared<u8>, src: *const u16, ld: usize, rows: u32, valid: u32) {
    let mut i = tid_x();
    let total = rows * 32;
    while i < total {
        let r = i >> 5;
        let c = i & 31;
        let d = dst.byte_add(sw(r, c)).addr();
        if r < valid {
            cp_async_16(d, src.add(r as usize * ld + (c * 8) as usize) as *const u8);
        } else {
            cp_async_16_zfill(d, src as *const u8, 0);
        }
        i += 128;
    }
}

/// grid = (ceil(n/64), 24), block = 128, smem = FLASH_SMEM.
/// Rows row0 .. row0+n of the batch are consecutive tokens of `slot` at positions pos0 .. pos0+n; their keys
/// (already in the cache) are positions 0 ..= own position. q: [rows][24][256]; qg: projection rows
/// (q|gate per head, stride ld_qg) for the gate; out: [rows][24*256].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_flash_prefill(out: *mut u16, q: *const u16, qg: *const u16, ld_qg: u32,
    kc: *const u16, vc: *const u16, slot_stride: u64, slot: u32, row0: u32, n: u32, pos0: u32, scale_log2: f32) {
    let smem: Shared<u8> = dyn_base();
    let sq = smem;
    let skv = smem.byte_add(BQ * ROWB);
    let qt = bid_x();
    let h = bid_y();
    let kvh = h / GRP;
    let w = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    let q_lo = qt * BQ; // first query (relative to the segment) of this block
    let q_rows = if n - q_lo < BQ { n - q_lo } else { BQ };
    let qbase = q.add(((row0 + q_lo) * NQ + h) as usize * HD as usize);
    load_rows(sq, qbase, (NQ * HD) as usize, BQ, q_rows);
    cp_async_commit();

    let cbase = slot as usize * slot_stride as usize + (kvh * HD) as usize;
    let kbase = kc.add(cbase);
    let vbase = vc.add(cbase);
    let last_pos = pos0 + q_lo + q_rows - 1; // highest key any row of this block needs
    let ntiles = last_pos / BKV + 1;
    let kv_stride = (NKV * HD) as usize;

    let load_kv = |t: u32, buf: u32| {
        let k0 = t * BKV;
        let valid = if last_pos + 1 - k0 < BKV { last_pos + 1 - k0 } else { BKV };
        let st = skv.byte_add(buf * 2 * BKV * ROWB);
        load_rows(st, kbase.add(k0 as usize * kv_stride), kv_stride, BKV, valid);
        load_rows(st.byte_add(BKV * ROWB), vbase.add(k0 as usize * kv_stride), kv_stride, BKV, valid);
    };
    load_kv(0, 0);
    cp_async_commit();

    // this warp's 16 rows: absolute positions
    let r_lo = q_lo + w * 16;
    let p_g = pos0 + r_lo + g; // row g
    let p_g8 = p_g + 8; // row g + 8
    let warp_max_pos = pos0 + r_lo + 15;

    let mut o = [[0.0f32; 4]; 32];
    let mut m0 = f32::NEG_INFINITY;
    let mut m1 = f32::NEG_INFINITY;
    let mut l0 = 0.0f32;
    let mut l1 = 0.0f32;

    let mut t = 0;
    while t < ntiles {
        if t + 1 < ntiles {
            load_kv(t + 1, (t + 1) & 1);
        }
        cp_async_commit();
        cp_async_wait::<1>();
        sync_threads();
        let st = skv.byte_add((t & 1) * 2 * BKV * ROWB);
        let sk = st;
        let sv = st.byte_add(BKV * ROWB);
        let k0 = t * BKV;
        if k0 <= warp_max_pos && w * 16 < q_rows {
            // S = Q K^T : 16 x 32
            let mut s = [[0.0f32; 4]; 4];
            kcommon::unroll!(ks in 16 => {
                let a = ldmatrix_x4(sq.byte_add(sw(w * 16 + (l & 15), ks as u32 * 2 + (l >> 4))).addr());
                kcommon::unroll!(p in 2 => {
                    let r = (2 * p as u32 + (l >> 4)) * 8 + (l & 7);
                    let b = ldmatrix_x4(sk.byte_add(sw(r, ks as u32 * 2 + ((l >> 3) & 1))).addr());
                    mma_bf16_16816(&mut s[2 * p], a, [b[0], b[1]]);
                    mma_bf16_16816(&mut s[2 * p + 1], a, [b[2], b[3]]);
                });
            });
            // scale + causal mask, row max
            let mut mx0 = m0;
            let mut mx1 = m1;
            kcommon::unroll!(nt in 4 => {
                let key = k0 + nt as u32 * 8 + 2 * t4;
                s[nt][0] = if key <= p_g { s[nt][0] * scale_log2 } else { f32::NEG_INFINITY };
                s[nt][1] = if key + 1 <= p_g { s[nt][1] * scale_log2 } else { f32::NEG_INFINITY };
                s[nt][2] = if key <= p_g8 { s[nt][2] * scale_log2 } else { f32::NEG_INFINITY };
                s[nt][3] = if key + 1 <= p_g8 { s[nt][3] * scale_log2 } else { f32::NEG_INFINITY };
                mx0 = mx0.max(s[nt][0]).max(s[nt][1]);
                mx1 = mx1.max(s[nt][2]).max(s[nt][3]);
            });
            mx0 = mx0.max(warp::shfl_xor(mx0, 1));
            mx0 = mx0.max(warp::shfl_xor(mx0, 2));
            mx1 = mx1.max(warp::shfl_xor(mx1, 1));
            mx1 = mx1.max(warp::shfl_xor(mx1, 2));
            // rows past the segment end see only -inf; keep their max finite
            let base0 = if mx0 == f32::NEG_INFINITY { 0.0 } else { mx0 };
            let base1 = if mx1 == f32::NEG_INFINITY { 0.0 } else { mx1 };
            let c0 = math::exp2(m0 - base0);
            let c1 = math::exp2(m1 - base1);
            m0 = mx0;
            m1 = mx1;
            let mut ps0 = 0.0;
            let mut ps1 = 0.0;
            let mut pa = [[0u32; 4]; 2];
            kcommon::unroll!(nt in 4 => {
                let e0 = math::exp2(s[nt][0] - base0);
                let e1 = math::exp2(s[nt][1] - base0);
                let e2 = math::exp2(s[nt][2] - base1);
                let e3 = math::exp2(s[nt][3] - base1);
                // the PV product uses bf16 probabilities; the row sum uses the same rounded values
                let (b0, b1, b2, b3) = (bf16::round(e0), bf16::round(e1), bf16::round(e2), bf16::round(e3));
                ps0 += b0 + b1;
                ps1 += b2 + b3;
                pa[nt / 2][(nt % 2) * 2] = bf16::pack2(b0, b1);
                pa[nt / 2][(nt % 2) * 2 + 1] = bf16::pack2(b2, b3);
            });
            l0 = l0 * c0 + ps0;
            l1 = l1 * c1 + ps1;
            kcommon::unroll!(dt in 32 => {
                o[dt][0] *= c0;
                o[dt][1] *= c0;
                o[dt][2] *= c1;
                o[dt][3] *= c1;
            });
            // O += P V : P 16 x 32 keys (2 k-steps), V tile [32 keys][256]
            kcommon::unroll!(kk in 2 => {
                // A fragment for keys 16kk..16kk+16: tiles 2kk (a0,a1) and 2kk+1 (a2,a3)
                let a = [pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3]];
                kcommon::unroll!(dp in 16 => {
                    let key = kk as u32 * 16 + (l & 7) + ((l >> 3) & 1) * 8;
                    let chunk = dp as u32 * 2 + (l >> 4);
                    let b = ldmatrix_x4_trans(sv.byte_add(sw(key, chunk)).addr());
                    mma_bf16_16816(&mut o[2 * dp], a, [b[0], b[1]]);
                    mma_bf16_16816(&mut o[2 * dp + 1], a, [b[2], b[3]]);
                });
            });
        }
        sync_threads();
        t += 1;
    }
    cp_async_wait_all();

    // quad-reduce the row sums, normalise, gate, store
    l0 += warp::shfl_xor(l0, 1);
    l0 += warp::shfl_xor(l0, 2);
    l1 += warp::shfl_xor(l1, 1);
    l1 += warp::shfl_xor(l1, 2);
    let inv0 = if l0 > 0.0 { 1.0 / l0 } else { 0.0 };
    let inv1 = if l1 > 0.0 { 1.0 / l1 } else { 0.0 };
    let ra = r_lo + g;
    let rb = ra + 8;
    kcommon::unroll!(dt in 32 => {
        let d = dt as u32 * 8 + 2 * t4;
        if ra < n {
            let row = (row0 + ra) as usize;
            let gp = *(qg.add(row * ld_qg as usize + (h * 2 * HD + HD + d) as usize) as *const u32);
            let y0 = bf16::round(o[dt][0] * inv0) * bf16::round(math::sigmoid(bf16::lo(gp)));
            let y1 = bf16::round(o[dt][1] * inv0) * bf16::round(math::sigmoid(bf16::hi(gp)));
            *(out.add(row * (NQ * HD) as usize + (h * HD + d) as usize) as *mut u32) = bf16::pack2(y0, y1);
        }
        if rb < n {
            let row = (row0 + rb) as usize;
            let gp = *(qg.add(row * ld_qg as usize + (h * 2 * HD + HD + d) as usize) as *const u32);
            let y0 = bf16::round(o[dt][2] * inv1) * bf16::round(math::sigmoid(bf16::lo(gp)));
            let y1 = bf16::round(o[dt][3] * inv1) * bf16::round(math::sigmoid(bf16::hi(gp)));
            *(out.add(row * (NQ * HD) as usize + (h * HD + d) as usize) as *mut u32) = bf16::pack2(y0, y1);
        }
    });
}
