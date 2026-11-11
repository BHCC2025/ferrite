//! The quantized K/V cache of small-memory devices: 6-bit or 4-bit records (a 16 GB card holds 128K tokens of
//! either beside the EXL3 weights).
//!
//! Every K and V head vector (256 dims) is rotated by an orthonormal block Hadamard (H32 / sqrt(32) on each
//! group of 32 dims), each group is scaled by its absolute maximum (one bf16 per group, rounded up) and its
//! values are stored as B-bit codes (B = 6 or 4) on the midpoint grid: value = (2 code + 1 - 2^B) / 2^B x scale.
//! This is the cache
//! quantization of exllamav3 (rotation, group absmax scale, midpoint grid; see CREDITS.md) in a record layout of
//! Ferrite's own.
//!
//! The attention kernels never rotate keys back: Q is rotated the same way when it is prepared (a rotation keeps
//! Q.K), the P.V sums run in the rotated basis, and the output row is rotated back once (the rotation is its own
//! inverse). A K/V tile is dequantized into shared memory as it is loaded, so the tensor-core part of the
//! kernels is the one of fe_flash_prefill / fe_attn_vk; a row's arithmetic depends only on its own keys, as there.
//!
//! Record of one head of one token: 32 code words (dims 8 w .. 8 w + 8 in word w, a code's high 4 bits each, dim
//! 8 w in the low nibble); with 6 bits, 32 u16 holding the codes' low 2 bits the same way; then 8 bf16 scales (one
//! per group of 32 dims): 144 bytes at 4 bits, 208 at 6. A token is NKV records; the caches are [slot][ctx] tokens.
//! Every kernel exists once per width (`_q4`, `_q6`).
use crate::attn::{f16r, f16x2, ssw, vk_phys, GRP, HD, NKV, NQ, P_SHIFT, SB, SROWB, VK_STAGES};
use crate::flash::{load_rows, sw};
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

/// Bytes of one head's record at `bits` (4 or 6) a value.
pub const fn qrec(bits: u32) -> u32 {
    if bits == 6 { 208 } else { 144 }
}
const R32: f32 = 0.176_776_7;

const BQ: u32 = 64;
const BKV: u32 = 32;
const ROWB: u32 = HD * 2;
/// fe_flash_prefill_q*'s dynamic shared memory: Q 32 KB + 2 staging buffers x (K + V records) + the working K and V
/// tiles.
pub const fn flash_q_smem(bits: u32) -> u32 {
    BQ * ROWB + 2 * 2 * BKV * qrec(bits) + 2 * BKV * ROWB
}
/// fe_attn_vk_q*'s: Q [GRP][16][256] + VK_STAGES staging buffers x (K + V records of 16 keys) + the working K and V
/// tiles.
pub const fn seg_q_smem(bits: u32) -> u32 {
    GRP * 16 * SROWB + VK_STAGES * 2 * SB * qrec(bits) + 2 * SB * SROWB
}

/// The 8 values of code word `c` of the staged record at `rec`: (2 code + 1 - 2^B) x scale / 2^B.
#[inline(always)]
fn deq8<const B: u32>(rec: Shared<u8>, c: u32) -> [f32; 8] {
    let word = rec.cast::<u32>().at(c);
    let scale = rec.byte_add(qrec(B) - 16).cast::<u16>().at(c >> 2);
    let mut v = [0.0f32; 8];
    if B == 6 {
        let lo = rec.byte_add(128).cast::<u16>().at(c) as u32;
        let s = bf16::to_f32(scale) * (1.0 / 64.0);
        kcommon::unroll!(j in 8 => {
            let code = ((word >> (4 * j as u32)) & 15) << 2 | ((lo >> (2 * j as u32)) & 3);
            v[j] = (2 * code as i32 - 63) as f32 * s;
        });
    } else {
        let s = bf16::to_f32(scale) * (1.0 / 16.0);
        kcommon::unroll!(j in 8 => {
            v[j] = (2 * ((word >> (4 * j as u32)) & 15) as i32 - 15) as f32 * s;
        });
    }
    v
}

/// `deq8` from the record's fields in registers.
#[inline(always)]
fn deq8w<const B: u32>(word: u32, lo: u32, scale: u16) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    if B == 6 {
        let s = bf16::to_f32(scale) * (1.0 / 64.0);
        kcommon::unroll!(j in 8 => {
            let code = ((word >> (4 * j as u32)) & 15) << 2 | ((lo >> (2 * j as u32)) & 3);
            v[j] = (2 * code as i32 - 63) as f32 * s;
        });
    } else {
        let s = bf16::to_f32(scale) * (1.0 / 16.0);
        kcommon::unroll!(j in 8 => {
            v[j] = (2 * ((word >> (4 * j as u32)) & 15) as i32 - 15) as f32 * s;
        });
    }
    v
}

/// fe_flash_prefill_q*p's dynamic shared memory: Q 32 KB + the working K and V tiles + the staged records of one V
/// tile and one K tile.
pub const fn flash_qp_smem(bits: u32) -> u32 {
    BQ * ROWB + 2 * BKV * ROWB + 2 * BKV * qrec(bits)
}

/// H32 / sqrt(32) of a group of 32 dims held by 4 lanes (lane & 3) with 8 values each: dim = (lane & 3) * 8 + j.
#[inline(always)]
fn had32(y: &mut [f32; 8], l: u32) {
    kcommon::unroll!(p in 4 => {
        let (a, b) = (y[2 * p], y[2 * p + 1]);
        y[2 * p] = a + b;
        y[2 * p + 1] = a - b;
    });
    kcommon::unroll!(p in 2 => {
        let (a0, a1, b0, b1) = (y[4 * p], y[4 * p + 1], y[4 * p + 2], y[4 * p + 3]);
        y[4 * p] = a0 + b0;
        y[4 * p + 1] = a1 + b1;
        y[4 * p + 2] = a0 - b0;
        y[4 * p + 3] = a1 - b1;
    });
    kcommon::unroll!(j in 4 => {
        let (a, b) = (y[j], y[j + 4]);
        y[j] = a + b;
        y[j + 4] = a - b;
    });
    kcommon::unroll!(st in 2 => {
        let m = 1u32 << st;
        kcommon::unroll!(j in 8 => {
            let p = warp::shfl_xor(y[j], m);
            y[j] = if l & m == 0 { y[j] + p } else { p - y[j] };
        });
    });
    kcommon::unroll!(j in 8 => {
        y[j] *= R32;
    });
}

/// fe_attn_prep for the quantized cache: q_out gets the normed, roped and rotated Q; K (normed, roped) and V are
/// rotated, quantized and written as records at the row's cache index. grid = rows, block = (NQ + 2 NKV) x 32.
/// kc / vc: [slot][ctx][NKV] records, `slot_stride` in bytes.
#[inline(always)]
unsafe fn prep_q<const B: u32>(q_out: *mut u16, kc: *mut u8, vc: *mut u8, qg: *const u16, ld_qg: u32,
    kx: *const u16, ld_k: u32, vx: *const u16, ld_v: u32, qn_w: *const u16, kn_w: *const u16, rope: *const u16,
    row_slot: *const u32, row_pos: *const u32, row_cpos: *const u32, slot_stride: u64, eps: f32) {
    let qrec = qrec(B);
    let r = bid_x() as usize;
    let w = warp_id();
    let l = lane();
    let slot = *row_slot.add(r) as u64;
    let pos = *row_pos.add(r) as u64;
    let cpos = *row_cpos.add(r) as u64;
    let cache_off = (slot * slot_stride + cpos * (NKV * qrec) as u64) as usize;
    let mut y = [0.0f32; 8];
    if w >= NQ + NKV {
        let h = w - NQ - NKV;
        y = bf16::unpack8(ldg_nc_v4(vx.add(r * ld_v as usize + (h * HD + l * 8) as usize) as *const u32));
    } else {
        let (src, nw) = if w < NQ {
            (qg.add(r * ld_qg as usize + (w * 2 * HD) as usize), qn_w)
        } else {
            (kx.add(r * ld_k as usize + ((w - NQ) * HD) as usize), kn_w)
        };
        let x = bf16::unpack8(ldg_nc_v4(src.add((l * 8) as usize) as *const u32));
        let wv = bf16::unpack8(ldg_nc_v4(nw.add((l * 8) as usize) as *const u32));
        let mut ss = 0.0;
        kcommon::unroll!(j in 8 => {
            ss += x[j] * x[j];
        });
        let inv = math::rsqrt(warp::sum(ss) / HD as f32 + eps);
        kcommon::unroll!(j in 8 => {
            y[j] = bf16::round(x[j] * inv * (1.0 + wv[j]));
        });
        // rotate-half RoPE over dims 0..64 (as fe_attn_prep), rounded to bf16 as its stored K / Q are
        let mut p = [0.0f32; 8];
        kcommon::unroll!(j in 8 => {
            p[j] = warp::shfl_xor(y[j], 4);
        });
        if l < 8 {
            let fi = (l & 3) * 8;
            let cs = rope.add((pos * 64) as usize);
            let c = bf16::unpack8(ldg_nc_v4(cs.add(fi as usize) as *const u32));
            let s = bf16::unpack8(ldg_nc_v4(cs.add((32 + fi) as usize) as *const u32));
            kcommon::unroll!(j in 8 => {
                let a = bf16::round(y[j] * c[j]);
                let b = if l < 4 { bf16::round(-p[j] * s[j]) } else { bf16::round(p[j] * s[j]) };
                y[j] = bf16::round(a + b);
            });
        }
    }
    had32(&mut y, l);
    if w < NQ {
        let dst = q_out.add(r * (NQ * HD) as usize + (w * HD) as usize);
        let o = [bf16::pack2(y[0], y[1]), bf16::pack2(y[2], y[3]), bf16::pack2(y[4], y[5]), bf16::pack2(y[6], y[7])];
        stg_v4(dst.add((l * 8) as usize) as *mut u32, o);
        return;
    }
    // the group's scale: its absolute maximum, as the next bf16 at or above it
    let mut am = 0.0f32;
    kcommon::unroll!(j in 8 => {
        am = am.max(y[j].abs());
    });
    am = am.max(warp::shfl_xor(am, 1));
    am = am.max(warp::shfl_xor(am, 2));
    let mut sb = bf16::from_f32(am);
    if bf16::to_f32(sb) < am {
        sb += 1;
    }
    let s = bf16::to_f32(sb);
    let half = (1u32 << (B - 1)) as f32;
    let top = (1i32 << B) - 1;
    let inv = if s > 0.0 { half / s } else { 0.0 };
    let mut word = 0u32;
    let mut low = 0u32;
    kcommon::unroll!(j in 8 => {
        let qi: i32 = math::fma(y[j], inv, half).to_int_unchecked();
        let qi = (if qi < 0 { 0 } else if qi > top { top } else { qi }) as u32;
        word |= (qi >> (B - 4)) << (4 * j as u32);
        low |= (qi & 3) << (2 * j as u32);
    });
    let (base, h) = if w < NQ + NKV { (kc, w - NQ) } else { (vc, w - NQ - NKV) };
    let dst = base.add(cache_off + (h * qrec) as usize);
    *(dst.add((l * 4) as usize) as *mut u32) = word;
    if B == 6 {
        *(dst.add((128 + l * 2) as usize) as *mut u16) = low as u16;
    }
    if l & 3 == 0 {
        *(dst.add((qrec - 16 + (l >> 2) * 2) as usize) as *mut u16) = sb;
    }
}

/// fe_attn_combine for partials in the rotated basis (the 4-bit cache): the merged row is rotated back (H32 /
/// sqrt(32) over each group of 32 dims = one warp) before rounding and the gate.
/// Merge splits and apply the output gate: out[r][h*256+d] = bf16(bf16(o) * bf16(sigmoid(gate))).
/// grid = (rows, 24), block = 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fe_attn_combine_q(out: *mut u16, part_o: *const f32, part_ml: *const f32,
    qg: *const u16, ld_qg: u32, nsplit: u32, row_base: u32) {
    let lr = bid_x();
    let r = row_base + lr;
    let h = bid_y();
    let d = tid_x();
    let kvh = h / GRP;
    let hh = h % GRP;
    let mut mm = f32::NEG_INFINITY;
    let mut s = 0;
    while s < nsplit {
        let pb = ((lr * NKV + kvh) * nsplit + s) as usize;
        let mw = *part_ml.add(pb * 12 + (hh * 2) as usize);
        mm = if mw > mm { mw } else { mm };
        s += 1;
    }
    let mut o = 0.0;
    let mut ls = 0.0;
    let mut s = 0;
    while s < nsplit {
        let pb = ((lr * NKV + kvh) * nsplit + s) as usize;
        let mw = *part_ml.add(pb * 12 + (hh * 2) as usize);
        if mw != f32::NEG_INFINITY {
            let f = math::exp2(mw - mm);
            o += *part_o.add(pb * 1536 + (hh * 256 + d) as usize) * f;
            ls += *part_ml.add(pb * 12 + (hh * 2 + 1) as usize) * f;
        }
        s += 1;
    }
    let l = lane();
    let mut x = o / ls;
    kcommon::unroll!(st in [0, 1, 2, 3, 4] => {
        let m = 1u32 << st;
        let p = warp::shfl_xor(x, m);
        x = if l & m == 0 { x + p } else { p - x };
    });
    let ov = bf16::round(x * R32);
    let g = bf16::to_f32(*qg.add(r as usize * ld_qg as usize + (h * 2 * HD + HD + d) as usize));
    let sg = bf16::round(math::sigmoid(g));
    *out.add((r * NQ * HD + h * HD + d) as usize) = bf16::from_f32(ov * sg);
}


/// fe_flash_prefill over the 4-bit cache: grid = (ceil(n/64), 24), block = 128, smem = FLASH_Q_SMEM. K/V tiles arrive
/// as records (cp.async into a double-buffered staging area) and are dequantized into one working tile; Q is
/// rotated (fe_attn_prep_q), the output is rotated back before the gate. `slot_stride` in bytes.
/// Rows row0 .. row0+n of the batch are consecutive tokens of `slot` at positions pos0 .. pos0+n; their keys
/// (already in the cache) are positions 0 ..= own position. q: [rows][24][256]; qg: projection rows
/// (q|gate per head, stride ld_qg) for the gate; out: [rows][24*256].
#[inline(always)]
unsafe fn flash_q<const B: u32>(out: *mut u16, q: *const u16, qg: *const u16, ld_qg: u32,
    kc: *const u8, vc: *const u8, slot_stride: u64, slot: u32, row0: u32, n: u32, pos0: u32, scale_log2: f32) {
    let qrec = qrec(B);
    let nch = qrec / 16;
    let qtb = BKV * qrec;
    let qtok = (NKV * qrec) as usize;
    let smem: Shared<u8> = dyn_base();
    let sq = smem;
    let stg = smem.byte_add(BQ * ROWB);
    let swk = stg.byte_add(2 * 2 * qtb);
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

    let cbase = slot as usize * slot_stride as usize + (kvh * qrec) as usize;
    let last_pos = pos0 + q_lo + q_rows - 1; // highest key any row of this block needs
    let ntiles = last_pos / BKV + 1;

    // a tile's records into staging buffer `buf`: [K: 32 records | V: the same], 16-byte chunks
    let load_kv = |t: u32, buf: u32| {
        let k0 = t * BKV;
        let valid = if last_pos + 1 - k0 < BKV { last_pos + 1 - k0 } else { BKV };
        let st = stg.byte_add(buf * 2 * qtb);
        let mut i = tid_x();
        while i < 2 * BKV * nch {
            let which = i / (BKV * nch);
            let rc = i % (BKV * nch);
            let r = rc / nch;
            let c = rc % nch;
            let d = st.byte_add(which * qtb + r * qrec + c * 16).addr();
            let src = if which == 0 { kc } else { vc };
            if r < valid {
                cp_async_16(d, src.add(cbase + (k0 + r) as usize * qtok + (c * 16) as usize));
            } else {
                cp_async_16_zfill(d, src, 0);
            }
            i += 128;
        }
    };
    // staging buffer `buf` -> the working tiles (K then V, [32 keys][256] bf16, swizzled)
    let dequant = |buf: u32| {
        let st = stg.byte_add(buf * 2 * qtb);
        let mut i = tid_x();
        while i < 2 * BKV * 32 {
            let which = i >> 10;
            let r = (i >> 5) & 31;
            let c = i & 31;
            let v = deq8::<B>(st.byte_add(which * qtb + r * qrec), c);
            swk.byte_add(which * BKV * ROWB + sw(r, c)).cast::<[u32; 4]>().st([bf16::pack2(v[0], v[1]),
                bf16::pack2(v[2], v[3]), bf16::pack2(v[4], v[5]), bf16::pack2(v[6], v[7])]);
            i += 128;
        }
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
        dequant(t & 1);
        sync_threads();
        let sk = swk;
        let sv = swk.byte_add(BKV * ROWB);
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
    // normalise, then rotate back (H32 / sqrt(32) over each group of 32 dims: dim = (dt % 4) * 8 + 2 * t4 + j)
    kcommon::unroll!(dt in 32 => {
        let (a0, a1, b0, b1) = (o[dt][0] * inv0, o[dt][1] * inv0, o[dt][2] * inv1, o[dt][3] * inv1);
        o[dt] = [a0 + a1, a0 - a1, b0 + b1, b0 - b1];
    });
    kcommon::unroll!(gq in 8 => {
        kcommon::unroll!(c in 4 => {
            let (x0, x1, x2, x3) = (o[4 * gq][c], o[4 * gq + 1][c], o[4 * gq + 2][c], o[4 * gq + 3][c]);
            let (s0, d0, s1, d1) = (x0 + x1, x0 - x1, x2 + x3, x2 - x3);
            o[4 * gq][c] = s0 + s1;
            o[4 * gq + 1][c] = d0 + d1;
            o[4 * gq + 2][c] = s0 - s1;
            o[4 * gq + 3][c] = d0 - d1;
        });
    });
    kcommon::unroll!(st in 2 => {
        let m = 1u32 << st;
        kcommon::unroll!(dt in 32 => {
            kcommon::unroll!(c in 4 => {
                let p = warp::shfl_xor(o[dt][c], m);
                o[dt][c] = if t4 & m == 0 { o[dt][c] + p } else { p - o[dt][c] };
            });
        });
    });
    let (inv0, inv1) = (R32, R32);
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

/// fe_flash_prefill_q with producer warps: grid = (ceil(n/64), 24), block = 128 + 32 x producer warps, smem =
/// flash_qp_smem. Warps 0..4 are fe_flash_prefill_q's (16 query rows each); the others stage the records of the
/// coming K/V tiles and dequantize them while the first four run the tensor-core part: V(t) during Q K^T of tile t
/// (which does not read the V tile), K(t + 1) during P V of tile t (which no longer reads the K tile). So the
/// dequantization hides behind the tensor cores. The same arithmetic on the same values as fe_flash_prefill_q: the
/// same bits out.
#[inline(always)]
unsafe fn flash_qp<const B: u32>(out: *mut u16, q: *const u16, qg: *const u16, ld_qg: u32,
    kc: *const u8, vc: *const u8, slot_stride: u64, slot: u32, row0: u32, n: u32, pos0: u32, scale_log2: f32) {
    let qrec = qrec(B);
    let qtok = (NKV * qrec) as usize;
    let smem: Shared<u8> = dyn_base();
    let sq = smem;
    // working tiles ([32 keys][256] bf16, swizzled): K, V; then the staged records of a V tile and of a K tile
    let swk = smem.byte_add(BQ * ROWB);
    let swv = swk.byte_add(BKV * ROWB);
    let stg = swv.byte_add(BKV * ROWB);
    let nch = qrec / 16;
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
    let prod = w >= 4;
    // producer threads: the block's beyond the first 128 (the launch picks how many warps)
    let np = bdim_x() - 128;
    if !prod {
        load_rows(sq, qbase, (NQ * HD) as usize, BQ, q_rows);
        cp_async_commit();
    }

    let cbase = slot as usize * slot_stride as usize + (kvh * qrec) as usize;
    let last_pos = pos0 + q_lo + q_rows - 1; // highest key any row of this block needs
    let ntiles = last_pos / BKV + 1;

    // producer warps (threads 128..256). stage: tile t's V (which = 0) or K (1) records by cp.async into staging
    // half `which`; rows past the keys are zero. dequant: staging half `which` into the working tile at `dst`.
    let stage = |t: u32, which: u32| {
        if t >= ntiles {
            return;
        }
        let k0 = t * BKV;
        let valid = if last_pos + 1 - k0 < BKV { last_pos + 1 - k0 } else { BKV };
        let st = stg.byte_add(which * BKV * qrec);
        let src = if which == 0 { vc } else { kc };
        let mut i = tid_x() - 128;
        while i < BKV * nch {
            let r = i / nch;
            let c = i % nch;
            let d = st.byte_add(r * qrec + c * 16).addr();
            if r < valid {
                cp_async_16(d, src.add(cbase + (k0 + r) as usize * qtok + (c * 16) as usize));
            } else {
                cp_async_16_zfill(d, src, 0);
            }
            i += np;
        }
    };
    let dequant = |which: u32, dst: Shared<u8>| {
        let st = stg.byte_add(which * BKV * qrec);
        let mut i = tid_x() - 128;
        while i < BKV * 32 {
            let r = i >> 5;
            let c = i & 31;
            let v = deq8::<B>(st.byte_add(r * qrec), c);
            dst.byte_add(sw(r, c)).cast::<[u32; 4]>().st([bf16::pack2(v[0], v[1]), bf16::pack2(v[2], v[3]),
                bf16::pack2(v[4], v[5]), bf16::pack2(v[6], v[7])]);
            i += np;
        }
    };
    // Every staged half is read only after a block barrier that follows all producers' waits for its copies, and
    // staged again only after a barrier that follows its last read: tile 0's records land here, K(0) is dequantized
    // after the first barrier, V(0) in the first phase.
    if prod {
        stage(0, 0);
        stage(0, 1);
        cp_async_commit();
        cp_async_wait_all();
    }

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

    if !prod {
        cp_async_wait_all();
    }
    sync_threads();
    if prod {
        dequant(1, swk);
    }
    sync_threads();
    let mut t = 0;
    while t < ntiles {
        let sk = swk;
        let sv = swv;
        let k0 = t * BKV;
        let active = !prod && k0 <= warp_max_pos && w * 16 < q_rows;
        // phase 1: S = Q K^T and the softmax step on K(t), while the producers send for K(t + 1)'s records and turn
        // the staged V(t) (landed before the last barrier) into the V tile, which nothing reads in this phase
        if prod {
            stage(t + 1, 1);
            cp_async_commit();
            dequant(0, swv);
            cp_async_wait_all();
        }
        let mut pa = [[0u32; 4]; 2];
        if active {
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
        }
        sync_threads();
        // phase 2: O += P V on V(t), while the producers send for V(t + 1)'s records and overwrite the K tile with
        // the staged K(t + 1)
        if prod {
            stage(t + 1, 0);
            cp_async_commit();
            if t + 1 < ntiles {
                dequant(1, swk);
            }
            cp_async_wait_all();
        }
        if active {
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
    if prod {
        cp_async_wait_all();
        return;
    }

    // quad-reduce the row sums, normalise, gate, store
    l0 += warp::shfl_xor(l0, 1);
    l0 += warp::shfl_xor(l0, 2);
    l1 += warp::shfl_xor(l1, 1);
    l1 += warp::shfl_xor(l1, 2);
    let inv0 = if l0 > 0.0 { 1.0 / l0 } else { 0.0 };
    let inv1 = if l1 > 0.0 { 1.0 / l1 } else { 0.0 };
    // normalise, then rotate back (H32 / sqrt(32) over each group of 32 dims: dim = (dt % 4) * 8 + 2 * t4 + j)
    kcommon::unroll!(dt in 32 => {
        let (a0, a1, b0, b1) = (o[dt][0] * inv0, o[dt][1] * inv0, o[dt][2] * inv1, o[dt][3] * inv1);
        o[dt] = [a0 + a1, a0 - a1, b0 + b1, b0 - b1];
    });
    kcommon::unroll!(gq in 8 => {
        kcommon::unroll!(c in 4 => {
            let (x0, x1, x2, x3) = (o[4 * gq][c], o[4 * gq + 1][c], o[4 * gq + 2][c], o[4 * gq + 3][c]);
            let (s0, d0, s1, d1) = (x0 + x1, x0 - x1, x2 + x3, x2 - x3);
            o[4 * gq][c] = s0 + s1;
            o[4 * gq + 1][c] = d0 + d1;
            o[4 * gq + 2][c] = s0 - s1;
            o[4 * gq + 3][c] = d0 - d1;
        });
    });
    kcommon::unroll!(st in 2 => {
        let m = 1u32 << st;
        kcommon::unroll!(dt in 32 => {
            kcommon::unroll!(c in 4 => {
                let p = warp::shfl_xor(o[dt][c], m);
                o[dt][c] = if t4 & m == 0 { o[dt][c] + p } else { p - o[dt][c] };
            });
        });
    });
    let (inv0, inv1) = (R32, R32);
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


/// fe_attn_vk over the 4-bit cache (records staged by cp.async, dequantized into one working K (bf16) / V (f16)
/// tile; `slot_stride` in bytes; the partials stay in the rotated basis, fe_attn_combine_q rotates back).
/// grid = (segments of the run, NKV, splits), block = 32 x GRP (warp w: q head kvh * GRP + w), dyn smem =
/// SEG_Q_SMEM. Segment seg0 + x has rows seg_row0[..] .. + seg_len[..] (<= 16, one slot); partial row index = row -
/// run_row0. q: [rows][NQ][256] (normed, roped); row_pos: RoPE position (keys = pos + 1).
#[inline(always)]
unsafe fn vk_q<const B: u32>(meta: Shared<u32>, part_o: *mut f32, part_ml: *mut f32, q: *const u16, kc: *const u8,
    vc: *const u8, row_slot: *const u32, row_pos: *const u32, row_tbase: *const u32, row_tmask: *const u32,
    seg_row0: *const u32, seg_len: *const u32, seg0: u32, run_row0: u32, slot_stride: u64, scale_log2: f32) {
    let qrec = qrec(B);
    let nch = qrec / 16;
    let qtok = (NKV * qrec) as usize;
    let smem: Shared<u8> = dyn_base();
    let sq = smem;
    let stg = smem.byte_add(GRP * 16 * SROWB);
    let swk = stg.byte_add(VK_STAGES * 2 * SB * qrec);
    let seg = (seg0 + bid_x()) as usize;
    let row0 = *seg_row0.add(seg);
    let n = *seg_len.add(seg);
    let kvh = bid_y();
    let s = bid_z();
    // splits of every row: the grid's z (the host picks it for the device: one wave of blocks an SM or two)
    let ns = gdim_z();
    let w = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    let nt = bdim_x();
    let tid = tid_x();
    if tid < n {
        let r = (row0 + tid) as usize;
        let nk = *row_pos.add(r) + 1;
        meta.set(tid, nk);
        meta.set(16 + tid, SB * nk.div_ceil(SB * ns));
        meta.set(32 + tid, *row_tbase.add(r));
        meta.set(48 + tid, *row_tmask.add(r));
    }
    sync_threads();
    let nk_of = |rr: u32| meta.at(rr);
    let c_of = |rr: u32| meta.at(16 + rr);
    let tb_of = |rr: u32| meta.at(32 + rr);
    let tm_of = |rr: u32| meta.at(48 + rr);
    let pbase_of = |rr: u32| ((((row0 - run_row0 + rr) * NKV + kvh) * ns) + s) as usize;

    // this split's work: per row keys [s c, min((s+1) c, nk))
    let mut any = false;
    let mut rr = 0;
    while rr < n {
        any |= s * c_of(rr) < nk_of(rr);
        rr += 1;
    }
    if !any {
        let mut i = tid;
        while i < n * GRP {
            let pb = pbase_of(i / GRP);
            *part_ml.add(pb * 12 + ((i % GRP) * 2) as usize) = f32::NEG_INFINITY;
            *part_ml.add(pb * 12 + ((i % GRP) * 2 + 1) as usize) = 0.0;
            i += nt;
        }
        return;
    }
    let slot = *row_slot.add(row0 as usize) as usize;
    // Q: the group's heads of the segment's rows, zero rows past n
    let mut i = tid;
    while i < GRP * 16 * 32 {
        let hh = i >> 9;
        let r = (i >> 5) & 15;
        let c = i & 31;
        let d = sq.byte_add(hh * 16 * SROWB + ssw(r, c)).addr();
        if r < n {
            cp_async_16(d, q.add((((row0 + r) * NQ + kvh * GRP + hh) * HD + c * 8) as usize) as *const u8);
        } else {
            cp_async_16_zfill(d, q as *const u8, 0);
        }
        i += nt;
    }
    cp_async_commit();

    // Work items, in the same order on every thread: chunk groups (rows with the same chunk length, increasing),
    // then tiles t of the group's split in increasing order, then for each tile the shared pass (rows whose keys
    // in it are cache rows t*16..) and one pass for each tree row with ancestors in it. An item is (cg, t, who):
    // who = 16 shared, < 16 that row alone.
    let need_own = |rr: u32, t: u32| {
        let nk = nk_of(rr);
        let end = if t * SB + SB < nk { t * SB + SB } else { nk };
        t * SB < nk && tb_of(rr) < end
    };
    // the next chunk length > cg present among the rows (u32::MAX: none)
    let next_group = |cg: u32| {
        let mut best = u32::MAX;
        let mut rr = 0;
        while rr < n {
            let c = c_of(rr);
            if c > cg && c < best && s * c < nk_of(rr) {
                best = c;
            }
            rr += 1;
        }
        best
    };
    // tile range of group cg in this split: [s cg / 16, ceil(min((s+1) cg, max nk) / 16))
    let tiles_of = |cg: u32| {
        let mut mx = 0;
        let mut rr = 0;
        while rr < n {
            if c_of(rr) == cg && nk_of(rr) > mx {
                mx = nk_of(rr);
            }
            rr += 1;
        }
        let end = if (s + 1) * cg < mx { (s + 1) * cg } else { mx };
        (s * cg / SB, end.div_ceil(SB))
    };
    // the item after (cg, t, who); (u32::MAX, ..) at the end
    let next_item_slow = |cg: u32, t: u32, who: u32| -> (u32, u32, u32) {
        // the next tree row of this tile needing its own pass
        let mut nw = if who == 16 { 0 } else { who + 1 };
        while nw < n && !(c_of(nw) == cg && need_own(nw, t)) {
            nw += 1;
        }
        if nw < n {
            return (cg, t, nw);
        }
        // the next tile's shared pass, else the next group's first tile
        let (_, t1) = tiles_of(cg);
        if t + 1 < t1 {
            return (cg, t + 1, 16);
        }
        let cn = next_group(cg);
        if cn == u32::MAX {
            return (u32::MAX, 0, 0);
        }
        (cn, tiles_of(cn).0, 16)
    };
    let first = {
        let cg = next_group(0);
        let (t0, _) = tiles_of(cg);
        (cg, t0, 16u32)
    };
    // fast path for the first group's leading tiles that no tree row needs on its own (all of a decode row's,
    // a verify segment's committed prefix): its key limit and the end of that tile range
    let cg1 = first.0;
    let (mx1, tmid1) = {
        let (mut mx, mut tb) = (0u32, u32::MAX);
        let mut rr = 0;
        while rr < n {
            if c_of(rr) == cg1 {
                if nk_of(rr) > mx {
                    mx = nk_of(rr);
                }
                if tb_of(rr) < tb {
                    tb = tb_of(rr);
                }
            }
            rr += 1;
        }
        let (_, t1) = tiles_of(cg1);
        (mx, if tb / SB < t1 { tb / SB } else { t1 })
    };
    let next_item = |cg: u32, t: u32, who: u32| -> (u32, u32, u32) {
        if cg == cg1 && who == 16 && t + 1 < tmid1 {
            return (cg, t + 1, 16);
        }
        next_item_slow(cg, t, who)
    };
    let cbase = slot * slot_stride as usize + (kvh * qrec) as usize;
    // K / V tile of an item: virtual keys t*16.. of the group's rows (shared) or of row `who` (its tree tail)
    let load_item = |it: (u32, u32, u32), buf: u32| {
        let (cg, t, who) = it;
        let k0 = t * SB;
        let (lim, tb, tm) = if who < 16 {
            (nk_of(who), tb_of(who), tm_of(who))
        } else if cg == cg1 {
            (mx1, u32::MAX, 0)
        } else {
            let mut mx = 0;
            let mut rr = 0;
            while rr < n {
                if c_of(rr) == cg && nk_of(rr) > mx {
                    mx = nk_of(rr);
                }
                rr += 1;
            }
            (mx, u32::MAX, 0)
        };
        let st = stg.byte_add(buf * 2 * SB * qrec);
        let mut i = tid;
        while i < 2 * SB * nch {
            let which = i / (SB * nch);
            let rc = i % (SB * nch);
            let r = rc / nch;
            let c = rc % nch;
            let d = st.byte_add(which * SB * qrec + r * qrec + c * 16).addr();
            let src = if which == 0 { kc } else { vc };
            let v = k0 + r;
            if v < lim {
                let p = vk_phys(v, tb, tm);
                cp_async_16(d, src.add(cbase + p as usize * qtok + (c * 16) as usize));
            } else {
                cp_async_16_zfill(d, src, 0);
            }
            i += nt;
        }
    };
    // VK_STAGES - 1 items in flight ahead of the one computed
    let mut ring = [first, (u32::MAX, 0, 0), (u32::MAX, 0, 0)];
    load_item(first, 0);
    cp_async_commit();
    let mut k = 1;
    while k < VK_STAGES - 1 {
        let pv = ring[(k - 1) as usize];
        if pv.0 != u32::MAX {
            let nx = next_item(pv.0, pv.1, pv.2);
            ring[k as usize] = nx;
            if nx.0 != u32::MAX {
                load_item(nx, k);
            }
        }
        cp_async_commit();
        k += 1;
    }

    let sqw = sq.byte_add(w * 16 * SROWB);
    // this lane's rows g and g + 8 of the segment
    let rows = [g, g + 8];
    let lim1 = [
        if g < n && c_of(g) == cg1 { nk_of(g) } else { 0 },
        if g + 8 < n && c_of(g + 8) == cg1 { nk_of(g + 8) } else { 0 },
    ];
    let mut o = [[0.0f32; 4]; 32];
    let mut mrow = [f32::NEG_INFINITY; 2];
    let mut lrow = [0.0f32; 2];
    let mut buf = 0u32;
    let mut it = first;
    while it.0 != u32::MAX {
        // the item VK_STAGES - 1 ahead into the buffer last used
        let last = ring[((buf + VK_STAGES - 2) % VK_STAGES) as usize];
        let ahead = if last.0 != u32::MAX { next_item(last.0, last.1, last.2) } else { (u32::MAX, 0, 0) };
        let slot_b = (buf + VK_STAGES - 1) % VK_STAGES;
        ring[slot_b as usize] = ahead;
        if ahead.0 != u32::MAX {
            load_item(ahead, slot_b);
        }
        cp_async_commit();
        cp_async_wait::<{ VK_STAGES - 1 }>();
        sync_threads();
        // the staged records of this item -> the working tiles: K bf16, V f16 (as fe_attn_vk's P V wants it)
        let st = stg.byte_add(buf * 2 * SB * qrec);
        let mut i = tid;
        while i < 2 * SB * 32 {
            let which = i >> 9;
            let r = (i >> 5) & (SB - 1);
            let c = i & 31;
            let v = deq8::<B>(st.byte_add(which * SB * qrec + r * qrec), c);
            let dst = swk.byte_add(which * SB * SROWB + ssw(r, c)).cast::<[u32; 4]>();
            if which == 0 {
                dst.st([bf16::pack2(v[0], v[1]), bf16::pack2(v[2], v[3]), bf16::pack2(v[4], v[5]), bf16::pack2(v[6], v[7])]);
            } else {
                dst.st([f16x2(v[0], v[1]), f16x2(v[2], v[3]), f16x2(v[4], v[5]), f16x2(v[6], v[7])]);
            }
            i += nt;
        }
        sync_threads();
        let sk = swk;
        let sv = swk.byte_add(SB * SROWB);
        let (cg, t, who) = it;
        let k0 = t * SB;
        // which of this lane's rows take part, and their key limits
        let lim = if cg == cg1 && who == 16 && t < tmid1 {
            lim1
        } else {
            let mut lim = [0u32; 2];
            kcommon::unroll!(hf in 2 => {
                let rr = rows[hf];
                let part = rr < n && c_of(rr) == cg && if who < 16 { rr == who } else { !need_own(rr, t) };
                lim[hf] = if part { nk_of(rr) } else { 0 };
            });
            lim
        };
        let mut sc = [[0.0f32; 4]; 2];
        kcommon::unroll!(ks in 16 => {
            let a = ldmatrix_x4(sqw.byte_add(ssw(l & 15, ks as u32 * 2 + (l >> 4))).addr());
            let bb = ldmatrix_x4(sk.byte_add(ssw((l >> 4) * 8 + (l & 7), ks as u32 * 2 + ((l >> 3) & 1))).addr());
            mma_bf16_16816(&mut sc[0], a, [bb[0], bb[1]]);
            mma_bf16_16816(&mut sc[1], a, [bb[2], bb[3]]);
        });
        let mut mx = mrow;
        kcommon::unroll!(tn in 2 => {
            let key = k0 + tn as u32 * 8 + 2 * t4;
            sc[tn][0] = if key < lim[0] { sc[tn][0] * scale_log2 } else { f32::NEG_INFINITY };
            sc[tn][1] = if key + 1 < lim[0] { sc[tn][1] * scale_log2 } else { f32::NEG_INFINITY };
            sc[tn][2] = if key < lim[1] { sc[tn][2] * scale_log2 } else { f32::NEG_INFINITY };
            sc[tn][3] = if key + 1 < lim[1] { sc[tn][3] * scale_log2 } else { f32::NEG_INFINITY };
            mx[0] = mx[0].max(sc[tn][0]).max(sc[tn][1]);
            mx[1] = mx[1].max(sc[tn][2]).max(sc[tn][3]);
        });
        kcommon::unroll!(hf in 2 => {
            mx[hf] = mx[hf].max(warp::shfl_xor(mx[hf], 1));
            mx[hf] = mx[hf].max(warp::shfl_xor(mx[hf], 2));
        });
        let base = [if mx[0] == f32::NEG_INFINITY { 0.0 } else { mx[0] }, if mx[1] == f32::NEG_INFINITY { 0.0 } else { mx[1] }];
        let cr = [math::exp2(mrow[0] - base[0]), math::exp2(mrow[1] - base[1])];
        mrow = mx;
        let mut ps = [0.0f32; 2];
        let mut pa = [0u32; 4];
        kcommon::unroll!(tn in 2 => {
            let e0 = f16r(math::exp2(sc[tn][0] - base[0] + P_SHIFT));
            let e1 = f16r(math::exp2(sc[tn][1] - base[0] + P_SHIFT));
            let e2 = f16r(math::exp2(sc[tn][2] - base[1] + P_SHIFT));
            let e3 = f16r(math::exp2(sc[tn][3] - base[1] + P_SHIFT));
            ps[0] += e0 + e1;
            ps[1] += e2 + e3;
            pa[tn * 2] = f16x2(e0, e1);
            pa[tn * 2 + 1] = f16x2(e2, e3);
        });
        lrow[0] = lrow[0] * cr[0] + ps[0];
        lrow[1] = lrow[1] * cr[1] + ps[1];
        kcommon::unroll!(dt in 32 => {
            o[dt][0] *= cr[0];
            o[dt][1] *= cr[0];
            o[dt][2] *= cr[1];
            o[dt][3] *= cr[1];
        });
        kcommon::unroll!(dp in 16 => {
            let bb = ldmatrix_x4_trans(sv.byte_add(ssw((l & 7) + ((l >> 3) & 1) * 8, dp as u32 * 2 + (l >> 4))).addr());
            mma_f16_16816(&mut o[2 * dp], pa, [bb[0], bb[1]]);
            mma_f16_16816(&mut o[2 * dp + 1], pa, [bb[2], bb[3]]);
        });
        sync_threads();
        buf = (buf + 1) % VK_STAGES;
        it = ring[buf as usize];
    }
    cp_async_wait_all();

    kcommon::unroll!(hf in 2 => {
        lrow[hf] += warp::shfl_xor(lrow[hf], 1);
        lrow[hf] += warp::shfl_xor(lrow[hf], 2);
    });
    kcommon::unroll!(hf in 2 => {
        let rr = rows[hf];
        if rr < n {
            let pb = pbase_of(rr);
            let po = part_o.add(pb * 1536 + (w * HD) as usize);
            kcommon::unroll!(dt in 32 => {
                let d = (dt as u32 * 8 + 2 * t4) as usize;
                *po.add(d) = o[dt][2 * hf];
                *po.add(d + 1) = o[dt][2 * hf + 1];
            });
            if t4 == 0 {
                *part_ml.add(pb * 12 + (w * 2) as usize) = mrow[hf] - P_SHIFT; // -inf stays -inf
                *part_ml.add(pb * 12 + (w * 2 + 1) as usize) = lrow[hf];
            }
        }
    });
}

// ---- the exported kernels, one set per code width ----

macro_rules! kvq_kernels {
    ($b:literal, $prep:ident, $flash:ident, $flashp:ident, $vk:ident, $meta:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $prep(q_out: *mut u16, kc: *mut u8, vc: *mut u8, qg: *const u16, ld_qg: u32,
            kx: *const u16, ld_k: u32, vx: *const u16, ld_v: u32, qn_w: *const u16, kn_w: *const u16, rope: *const u16,
            row_slot: *const u32, row_pos: *const u32, row_cpos: *const u32, slot_stride: u64, eps: f32) {
            prep_q::<$b>(q_out, kc, vc, qg, ld_qg, kx, ld_k, vx, ld_v, qn_w, kn_w, rope, row_slot, row_pos, row_cpos,
                slot_stride, eps)
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $flash(out: *mut u16, q: *const u16, qg: *const u16, ld_qg: u32, kc: *const u8,
            vc: *const u8, slot_stride: u64, slot: u32, row0: u32, n: u32, pos0: u32, scale_log2: f32) {
            flash_q::<$b>(out, q, qg, ld_qg, kc, vc, slot_stride, slot, row0, n, pos0, scale_log2)
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $flashp(out: *mut u16, q: *const u16, qg: *const u16, ld_qg: u32, kc: *const u8,
            vc: *const u8, slot_stride: u64, slot: u32, row0: u32, n: u32, pos0: u32, scale_log2: f32) {
            flash_qp::<$b>(out, q, qg, ld_qg, kc, vc, slot_stride, slot, row0, n, pos0, scale_log2)
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $vk(part_o: *mut f32, part_ml: *mut f32, q: *const u16, kc: *const u8,
            vc: *const u8, row_slot: *const u32, row_pos: *const u32, row_tbase: *const u32, row_tmask: *const u32,
            seg_row0: *const u32, seg_len: *const u32, seg0: u32, run_row0: u32, slot_stride: u64, scale_log2: f32) {
            let meta = static_shared!($meta, u32, 4 * 64);
            vk_q::<$b>(meta, part_o, part_ml, q, kc, vc, row_slot, row_pos, row_tbase, row_tmask, seg_row0, seg_len, seg0,
                run_row0, slot_stride, scale_log2)
        }
    };
}
kvq_kernels!(4, fe_attn_prep_q4, fe_flash_prefill_q4, fe_flash_prefill_q4p, fe_attn_vk_q4, vkq4_meta);
kvq_kernels!(6, fe_attn_prep_q6, fe_flash_prefill_q6, fe_flash_prefill_q6p, fe_attn_vk_q6, vkq6_meta);
