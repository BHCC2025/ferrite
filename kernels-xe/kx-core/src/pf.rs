//! Prompt-window attention over the q8 KV cache with int8 scores: Rust port of arc-engine `kernels/xpu/qwen_attn_pfx.cl`
//! built as qwen_attn_pfxi_q8.spv (-DKVQ=8 -DI8 -DHAD -DNO_HALF2; ARC_PFX=i8, the serving default). In the 256-GRF module
//! with the decode attention.
//! * attn_pfs_prep_i8: queries -> Hadamard-rotated int8 B tiles + one fp32 scale a column (row, head);
//! * kvq_expand_k8: K records of a key range -> Hadamard-rotated int8 (K - mu) + one fp32 scale a (key, kv head);
//! * kvq_expand_vt: V records -> fp16 transposed [kv head][dim][key] (q * scale, one RTE rounding);
//! * attn_prefill_s: 8 rows x 6 heads a work-group (48 columns, 3 N tiles), int8 Q K^T on the matrix engine, online
//!   softmax in base 2 with P x 1024 rounded to fp16, fp16 P V, output gate; ranges of keys chain through a state buffer.
//! Contraction sites of the reference: only `ll = ll * al + sum` (mad) in attn_prefill_s; the module has ContractionOff on
//! the four kernels.
use crate::ext::{fabs, rint, vload16_u16, vload8, vload8_u16, vstore8, vstore8_u16};
use crate::{bf, bfr, sigmoid, to_bf};
use kxcommon::blk2d::Surface;
use kxcommon::sg::Shuffle;
use kxcommon::v::*;
use kxcommon::{attr, bits, blk, cvt, dpas, id, math, sg, slm, unroll};

const HD: usize = 256;
const NH: usize = 24;
const NKV: usize = 4;
const GQ: u32 = 6;
const REC: usize = 272;
const DATA: usize = 256;
const PSC: f32 = 1024.0;
/// qs scale of the reference: (0.0625f * 1.4426950408889634f), folded in f32 like clang.
const QSC: f32 = 0.0625f32 * 1.442_695_040_888_963_4f32;
const STSZ: usize = 1632;

unsafe extern "C" {
    fn _Z5clampfff(x: f32, lo: f32, hi: f32) -> f32;
}

/// OpenCL `clamp(float, float, float)` (OpenCL.std fclamp).
#[inline(always)]
fn fclamp(x: f32, lo: f32, hi: f32) -> f32 {
    unsafe { _Z5clampfff(x, lo, hi) }
}
/// convert_half_rte bits.
#[inline(always)]
fn pcvt(f: f32) -> u16 {
    cvt::f16_bits_rte(f)
}
#[inline(always)]
fn pback(h: u16) -> f32 {
    cvt::f16_bits_to_f32(h)
}
/// int8 code of x * inv: (uint)(int)clamp(rint(x * inv), -127, 127) & 255.
#[inline(always)]
fn q8(x: f32, inv: f32) -> u32 {
    (unsafe { fclamp(rint(x * inv), -127.0, 127.0).to_int_unchecked::<i32>() } as u32) & 255
}

/// Fp32 values (q * block scale) of the 8 dims 8 d8 .. + 7 of a K / V record.
#[inline(always)]
unsafe fn deq8f(rec: *const u8, d8: usize) -> [f32; 8] {
    let sc = cvt::f16_bits_to_f32(*(rec.add(DATA + 2 * (d8 >> 2)) as *const u16));
    let w = kxcommon::mem::global::vload2_u32(0, rec.add(d8 * 8) as *const u32).to_array();
    let mut o = [0f32; 8];
    unroll!(4, 4, |i| {
        o[i] = (((w[0] << (24 - 8 * i)) as i32) >> 24) as f32 * sc;
        o[4 + i] = (((w[1] << (24 - 8 * i)) as i32) >> 24) as f32 * sc;
    });
    o
}
/// The same as fp16 bits (one RTE rounding of q * scale).
#[inline(always)]
unsafe fn deq8(rec: *const u8, d8: usize) -> [u16; 8] {
    let sc = cvt::f16_bits_to_f32(*(rec.add(DATA + 2 * (d8 >> 2)) as *const u16));
    let w = kxcommon::mem::global::vload2_u32(0, rec.add(d8 * 8) as *const u32).to_array();
    let mut o = [0u16; 8];
    unroll!(4, 4, |i| {
        o[i] = pcvt((((w[0] << (24 - 8 * i)) as i32) >> 24) as f32 * sc);
        o[4 + i] = pcvt((((w[1] << (24 - 8 * i)) as i32) >> 24) as f32 * sc);
    });
    o
}

/// Orthonormal Walsh-Hadamard transform of 256 values held by a sub-group, 16 a lane (value i of lane l = dim 16 l + i):
/// in-lane butterflies, then across lanes (qwen_attn_pfx.cl fwht256).
#[inline(always)]
fn fwht256(v: &mut [f32; 16]) {
    unroll!(4, 4, |si| {
        let s = 1usize << si;
        unroll!(16, 16, |i| {
            if i & s == 0 {
                let (a, b) = (v[i], v[i + s]);
                v[i] = a + b;
                v[i + s] = a - b;
            }
        });
    });
    let lane = id::lane();
    unroll!(4, 4, |si| {
        let s = 1u32 << si;
        unroll!(16, 16, |i| {
            let o = v[i].shuffle_xor(s);
            v[i] = if lane & s != 0 { o - v[i] } else { v[i] + o };
        });
    });
    unroll!(16, 16, |i| v[i] *= 0.0625);
}

/// V records of keys k0 .. k0 + n - 1 -> fp16 transposed [kv head][256 dims][rk keys]. Grid ((n + 63) / 64, 128), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kvq_expand_vt(src: *const u8, dst: *mut u16, k0: u32, n: u32, rk: u32) {
    attr::sg_size::<32>(); // the reference compiles at SIMD32 (no required sub-group size)
    let key = id::global_id(0) as u32;
    let hd = id::global_id(1) as u32;
    let (h, d8) = ((hd >> 5) as usize, (hd & 31) as usize);
    if key >= n {
        return;
    }
    let o = deq8(src.add(((key + k0) as usize * NKV + h) * REC), d8);
    let d = dst.add((h * HD + d8 * 8) * rk as usize + key as usize);
    unroll!(8, 8, |i| *d.add(i * rk as usize) = o[i]);
}

/// K records of keys k0 .. k0 + n - 1 -> int8 [key - k0][4][256] of the rotated K - mu, scale ks[key - k0][h] = amax / 127.
/// A sub-group a (key, head), 16 dims a lane. Grid (n), local 64.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kvq_expand_k8(src: *const u8, dst: *mut u8, ks: *mut f32, mu: *const f32, k0: u32) {
    let g = id::global_id(0);
    let lane = g & 15;
    let h = (g >> 4) & 3;
    let key = g >> 6;
    let rec = src.add((key + k0 as usize) * NKV * REC + h * REC);
    let a0 = deq8f(rec, 2 * lane);
    let a1 = deq8f(rec, 2 * lane + 1);
    let m0 = vload8(0, mu.add(h * HD + lane * 16));
    let m1 = vload8(0, mu.add(h * HD + lane * 16 + 8));
    let mut v = [0f32; 16];
    unroll!(8, 8, |i| {
        v[i] = a0[i] - m0[i];
        v[8 + i] = a1[i] - m1[i];
    });
    fwht256(&mut v);
    let mut a = 0.0f32;
    unroll!(8, 8, |i| a = math::fmax(a, math::fmax(fabs(v[i]), fabs(v[8 + i]))));
    let a = sg::reduce_max_f32(a);
    let inv = if a > 0.0 { math::div_ocl(127.0, a) } else { 0.0 };
    let mut w = [0u32; 4];
    unroll!(8, 8, |i| {
        w[i >> 2] |= q8(v[i], inv) << (8 * (i & 3));
        w[2 + (i >> 2)] |= q8(v[8 + i], inv) << (8 * (i & 3));
    });
    kxcommon::mem::global::vstore4_u32(U32x4::from_array(w), 0, dst.add((key * NKV + h) * HD + lane * 16) as *mut u32);
    if lane == 0 {
        *ks.add(key * NKV + h) = math::div_ocl(a, 127.0);
    }
}

/// Queries q [row][24][256] bf16 -> int8 B tiles qb [row tile * 4 + kv head][N tile][32-dim step][dword][lane] and fp32
/// scales qs [row tile * 4 + kv head][N tile][lane] = amax / 127 of the rotated column. A sub-group a column.
/// Grid (ceil(rows / 8) * 24), local 128.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn attn_pfs_prep_i8(q: *const u16, qb: *mut u32, qs: *mut f32, rows: u32) {
    let g = id::global_id(0) as u32;
    let lane = g & 15;
    let col = g >> 4;
    let n = col & 15;
    let j = (col >> 4) % 3;
    let wgi = (col >> 4) / 3;
    let hk = wgi & 3;
    let row = (wgi >> 2) * 8 + (n & 7);
    let head = hk * GQ + 2 * j + (n >> 3);
    let mut v = [0f32; 16];
    if row < rows {
        let x = vload16_u16(0, q.add((row as usize * NH + head as usize) * HD + lane as usize * 16));
        unroll!(16, 16, |i| v[i] = bf(x[i]));
    }
    fwht256(&mut v);
    let mut a = 0.0f32;
    unroll!(16, 16, |i| a = math::fmax(a, fabs(v[i])));
    let a = sg::reduce_max_f32(a);
    let inv = if a > 0.0 { math::div_ocl(127.0, a) } else { 0.0 };
    let o = qb.add((((wgi * 3 + j) * 8 + (lane >> 1)) * 128 + ((lane & 1) * 4) * 16 + n) as usize);
    unroll!(4, 4, |e| {
        let mut w = 0u32;
        unroll!(4, 4, |b| w |= q8(v[4 * e + b], inv) << (8 * b));
        *o.add(e * 16) = w;
    });
    if lane == 0 {
        *qs.add(((wgi * 3 + j) * 16 + n) as usize) = math::div_ocl(a, 127.0);
    }
}

/// 2D load of 16 rows x 32 bytes (8-bit elements, 2 a lane): 16 u16 a lane, row-major (kx K' tiles).
#[inline(always)]
unsafe fn rd_8b_16r32x1c(s: Surface, x: i32, y: i32) -> [u16; 16] {
    let mut d = [0u16; 16];
    kxcommon::ffi::_Z41intel_sub_group_2d_block_read_8b_16r32x1cPU3AS1viiiDv2_iPt(s.base, s.width_bytes, s.height, s.pitch_bytes, I32x2::from_array([x, y]), d.as_mut_ptr());
    d
}
/// 2D load of 32 rows x 16 16-bit columns: 32 u16 a lane (vt V^T tiles).
#[inline(always)]
unsafe fn rd_16b_32r16x1c(s: Surface, x: i32, y: i32) -> [u16; 32] {
    let mut d = [0u16; 32];
    kxcommon::ffi::_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt(s.base, s.width_bytes, s.height, s.pitch_bytes, I32x2::from_array([x, y]), d.as_mut_ptr());
    d
}
#[inline(always)]
fn i16x8(v: &[u16], o: usize) -> I16x8 {
    I16x8::from_array([v[o] as i16, v[o + 1] as i16, v[o + 2] as i16, v[o + 3] as i16, v[o + 4] as i16, v[o + 5] as i16, v[o + 6] as i16, v[o + 7] as i16])
}

/// Causal flash attention of `rows` rows at positions pos0 .. over the keys [kstart, kend) of the expanded range that
/// starts at key koff (K' int8 in kx [key - koff][4][256], scales ks, V' fp16 transposed in vt [4][256][rk]); 8 rows x
/// 6 query heads of one kv head a work-group, 8 sub-groups (16 keys a sub-group a block of 128). flags bit 0: first range
/// (fresh state), bit 1: last range (finish: normalise, gate, write out [row][24][256] bf16); otherwise the online-softmax
/// state is parked in st. Grid (ceil(rows / 8), 4), local 128.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn attn_prefill_s(
    qb: *const u32,
    kx: *const u16,
    vt: *const u16,
    qg: *const u16,
    out: *mut u16,
    st: *mut f32,
    pos0: u32,
    rows: u32,
    koff: u32,
    kstart: u32,
    kend: u32,
    flags: u32,
    rk: u32,
    ks: *const f32,
    qs: *const f32,
) {
    attr::reqd_work_group_size::<128, 1, 1>();
    let mxs = slm::alloc::<f32, { 3 * 8 * 16 }, 0>(); // [N tile][sub-group][column]: tile maxima, later denominators
    let pt = slm::alloc::<u32, { 8 * 3 * 16 * 8 }, 1>(); // P^T: [key tile][N tile][dword of key pairs][column]
    let hk = id::group_id(1);
    let sgi = id::sg_id();
    let lane = id::lane();
    let r0 = id::group_id(0) * 8;
    let kmax = pos0 + (r0 + 8).min(rows);
    if kstart >= kmax {
        return;
    }
    let lim = pos0 + r0 + (lane & 7);
    let wgi = id::group_id(0) * 4 + hk;
    let qt = qb.add(wgi as usize * (3 * 1024));
    let mut qsc = [0f32; 3];
    unroll!(4, 3, |j| qsc[j] = *qs.add((wgi as usize * 3 + j) * 16 + lane as usize) * QSC);
    let kfin = kend.min(kmax);
    let thi = (kfin + 15) / 16;
    let height = (kfin - koff) as i32;
    let skx = Surface { base: kx as *const u8, width_bytes: 1024, height, pitch_bytes: 1024 };
    let svt = Surface { base: vt as *const u8, width_bytes: (rk * 2) as i32, height: (NKV * HD) as i32, pitch_bytes: (rk * 2) as i32 };
    let mut m = [f32::NEG_INFINITY; 3];
    let mut ll = [0f32; 3];
    let mut acc = [[F32x8::splat(0.0); 4]; 3];
    if flags & 1 == 0 {
        let sb = st.add((wgi as usize * 8 + sgi as usize) * STSZ);
        unroll!(4, 3, |j| {
            m[j] = *sb.add(1536 + j * 16 + lane as usize);
            ll[j] = *sb.add(1584 + j * 16 + lane as usize);
            unroll!(4, 4, |g| acc[j][g] = F32x8::from_array(vload8(lane as usize, sb.add((j * 4 + g) * 128))));
        });
    }
    let mut kb = kstart / 16;
    while kb < thi {
        // scores of key tile kb + sg (16 keys) for the 48 columns: c[j][h][i], lane = column, key 8 h + i of the tile
        let tile = kb + sgi;
        let mut ci = [[I32x8::splat(0); 2]; 3];
        unroll!(8, 8, |t| {
            let ka = rd_8b_16r32x1c(skx, (hk as usize * 256 + t * 32) as i32, (tile * 16) as i32 - koff as i32);
            let (a0, a1) = (i16x8(&ka, 0), i16x8(&ka, 8));
            unroll!(4, 3, |j| {
                let b: I32x8 = bits(blk::global::read_u32x8(qt.add((j * 8 + t) * 128)));
                ci[j][0] = dpas::i8_m8(a0, b, ci[j][0]);
                ci[j][1] = dpas::i8_m8(a1, b, ci[j][1]);
            });
        });
        let kr = tile * 16 + lane - koff;
        let ksv = if kr < height as u32 { *ks.add(kr as usize * NKV + hk as usize) } else { 0.0 };
        let mut c = [[[0f32; 8]; 2]; 3];
        unroll!(4, 3, |j| unroll!(2, 2, |h| {
            let cv = ci[j][h].to_array();
            unroll!(8, 8, |i| c[j][h][i] = cv[i] as f32 * (sg::broadcast_f32(ksv, (h * 8 + i) as u32) * qsc[j]));
        }));
        // softmax: masks (diagonal tiles only), tile maxima through local memory
        let tile_ = kb + sgi;
        if tile_ * 16 + 15 <= pos0 + r0 {
            unroll!(4, 3, |j| {
                let mut mx = f32::NEG_INFINITY;
                unroll!(2, 2, |h| unroll!(8, 8, |i| mx = math::fmax(mx, c[j][h][i])));
                *mxs.add((j * 8 + sgi as usize) * 16 + lane as usize) = mx;
            });
        } else {
            unroll!(4, 3, |j| {
                let mut mx = f32::NEG_INFINITY;
                unroll!(2, 2, |h| unroll!(8, 8, |i| {
                    let s = if tile_ * 16 + (h * 8 + i) as u32 <= lim { c[j][h][i] } else { f32::NEG_INFINITY };
                    c[j][h][i] = s;
                    mx = math::fmax(mx, s);
                }));
                *mxs.add((j * 8 + sgi as usize) * 16 + lane as usize) = mx;
            });
        }
        slm::barrier();
        let mut al = [0f32; 3];
        unroll!(4, 3, |j| {
            let mut mn = m[j];
            let tm: [f32; 8] = bits(blk::local::read_u32x8(mxs.add(j * 128) as *const u32));
            unroll!(8, 8, |s| mn = math::fmax(mn, tm[s]));
            al[j] = if m[j] == f32::NEG_INFINITY { 0.0 } else { math::exp2(m[j] - mn) };
            m[j] = mn;
            let mut pk = [0u32; 8];
            let mut sum = 0.0f32;
            unroll!(8, 8, |d| {
                let p0 = pcvt(math::exp2(c[j][d >> 2][2 * (d & 3)] - mn) * PSC);
                let p1 = pcvt(math::exp2(c[j][d >> 2][2 * (d & 3) + 1] - mn) * PSC);
                sum += pback(p0);
                sum += pback(p1);
                pk[d] = p0 as u32 | ((p1 as u32) << 16);
            });
            ll[j] = math::mad(ll[j], al[j], sum);
            blk::local::write_u32x8(pt.add((sgi as usize * 3 + j) * 128), U32x8::from_array(pk));
        });
        slm::barrier();
        unroll!(4, 3, |j| {
            if !sg::all(al[j] == 1.0) {
                unroll!(4, 4, |g| acc[j][g] *= F32x8::splat(al[j]));
            }
        });
        // O^T += V^T P^T over the key tiles kb .. kb + 7 (< thi): sub-group sg owns dims 32 sg .. + 31
        unroll!(8, 8, |uu| {
            if kb + (uu as u32) < thi {
                let mut bp = [I32x8::splat(0); 3];
                unroll!(4, 3, |j| bp[j] = bits(blk::local::read_u32x8(pt.add((uu * 3 + j) * 128))));
                let va = rd_16b_32r16x1c(svt, ((kb + uu as u32) * 16) as i32 - koff as i32, (hk as usize * HD + sgi as usize * 32) as i32);
                unroll!(4, 4, |g| {
                    let a = i16x8(&va, 8 * g);
                    unroll!(4, 3, |j| acc[j][g] = dpas::f16_m8(a, bp[j], acc[j][g]));
                });
            }
        });
        kb += 8;
    }
    if flags & 2 == 0 && kend < kmax {
        // more ranges follow for this work-group: park the state
        let sb = st.add((wgi as usize * 8 + sgi as usize) * STSZ);
        unroll!(4, 3, |j| {
            *sb.add(1536 + j * 16 + lane as usize) = m[j];
            *sb.add(1584 + j * 16 + lane as usize) = ll[j];
            unroll!(4, 4, |g| vstore8(acc[j][g].to_array(), lane as usize, sb.add((j * 4 + g) * 128)));
        });
        return;
    }
    unroll!(4, 3, |j| *mxs.add((j * 8 + sgi as usize) * 16 + lane as usize) = ll[j]); // the maxima are dead
    slm::barrier();
    unroll!(4, 3, |j| {
        let mut l = 0.0f32;
        unroll!(8, 8, |s| l += *mxs.add((j * 8 + s) * 16 + lane as usize));
        let inv = math::div_ocl(1.0, l);
        let row = r0 + (lane & 7);
        let head = hk * GQ + 2 * j as u32 + (lane >> 3);
        if row < rows {
            unroll!(4, 4, |g| {
                let d0 = sgi as usize * 32 + g * 8;
                let gv = vload8_u16(0, qg.add((row as usize * NH + head as usize) * 2 * HD + HD + d0));
                let a = acc[j][g].to_array();
                let mut o = [0u16; 8];
                unroll!(8, 8, |i| o[i] = to_bf(bfr(a[i] * inv) * sigmoid(bf(gv[i]))));
                vstore8_u16(o, 0, out.add((row as usize * NH + head as usize) * HD + d0));
            });
        }
    });
}
