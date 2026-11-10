//! Full-attention decode over the q8 KV cache on the matrix engine: Rust port of arc-engine
//! `kernels/xpu/qwen_attn_dec.cl` (the KVQ=8 build, qwen_attn_dec_q8.spv; FP_CONTRACT OFF there, so no fused site:
//! every product and sum here is two roundings). Built as its own module (cargo feature `dec` -> kx_core_dec.spv) and
//! loaded with `-cl-intel-256-GRF-per-thread`, like the reference.
//!
//! Split-K over the cache: a work-group takes a chunk of LC = 1024 keys of one (kv head, row) (attn_dec_partial) or of
//! one (kv head, tile of 8 (row, head) pairs) (attn_dec_pack*), a sub-group a run of KS = 128 keys taken 32 at a time;
//! the sub-groups merge in a fixed order through local memory, attn_long_merge combines the chunks.
//! * scores: B = 1152 + k in fp16 (exact: mask + xor, `kv16`), lane = key (2D transposed loads of 32 dims); A = q in
//!   fp16; the offset leaves through 1152 * (sum of the fp16 q over the block), then the block scale;
//! * P V: A = p * (V block scale) in fp16, B = 1152 + v (2D VNNI loads of 32 keys x 32 dims, lane = dim), the offset
//!   removed by a second product with B = -1152; fp32 accumulators.
//! kv16 pairs bytes 0, 2 and 1, 3 of a dword, so the K index of a product runs over dims (keys) 0 2 1 3 4 6 5 7 ..: q is
//! stored in that order and p is shuffled to it. A row's bits depend on its key count only.
use crate::bf;
use kxcommon::blk2d::{self, Surface};
use kxcommon::sg::Shuffle;
use kxcommon::v::*;
use kxcommon::{attr, bits, blk, cvt, dpas, id, math, sg, slm, unroll};

const HD: usize = 256;
const NH: usize = 24;
const GQ: usize = 6;
const LC: u32 = 1024;
const KS: u32 = 128;
const REC: u32 = 272;
const DATA: u32 = 256;
const RW: u32 = 4 * REC; // bytes a position (4 kv heads)
const NEG_INF: f32 = f32::NEG_INFINITY;

/// fp16 bits of 1152 + b for the signed bytes 0, 2 (lo) and 1, 3 (hi) of x: 0x6400 | (b ^ 0x80).
#[inline(always)]
fn kvlo(x: u32) -> u32 {
    (x & 0x00FF_00FF) ^ 0x6480_6480
}
#[inline(always)]
fn kvhi(x: u32) -> u32 {
    ((x >> 8) & 0x00FF_00FF) ^ 0x6480_6480
}
#[inline(always)]
fn kv16(x0: u32, x1: u32, x2: u32, x3: u32) -> I32x8 {
    bits(U32x8::from_array([kvlo(x0), kvhi(x0), kvlo(x1), kvhi(x1), kvlo(x2), kvhi(x2), kvlo(x3), kvhi(x3)]))
}
/// K index i of a product -> dim (key) within its 4: 0 2 1 3.
#[inline(always)]
fn perm4(i: u32) -> u32 {
    (i & !3) | ((i & 1) << 1) | ((i >> 1) & 1)
}
/// convert_half (RTE; OpFConvert as the reference) of a bf16 value, as fp16 bits.
#[inline(always)]
fn bf_to_h(v: u16) -> u16 {
    bits::<f16, u16>(bf(v) as f16)
}
/// (float)convert_half(bf16 value).
#[inline(always)]
fn bf_to_hf(v: u16) -> f32 {
    (bf(v) as f16) as f32
}
/// fp16 element e of 8 packed in 4 dwords (as_half8 / vload_half on a uint4), widened.
#[inline(always)]
fn half_of(d: &[u32; 4], e: usize) -> f32 {
    cvt::f16_bits_to_f32((d[e / 2] >> (16 * (e % 2))) as u16)
}

/// 2D VNNI load of 32 rows (keys) x 32 bytes, 2 blocks of 16 columns: 16 dwords a lane. Called through the raw builtin
/// with an indexed local array (kxcommon's Simd-returning wrapper makes LLVM copy the 64 bytes as one i512, which SPIR-V
/// cannot express).
#[inline(always)]
unsafe fn rd_t8_32r16x2c(s: Surface, x: i32, y: i32) -> [u32; 16] {
    let mut d = [0u32; 16];
    kxcommon::ffi::_Z51intel_sub_group_2d_block_read_transform_8b_32r16x2cPU3AS1viiiDv2_iPj(
        s.base,
        s.width_bytes,
        s.height,
        s.pitch_bytes,
        I32x2::from_array([x, y]),
        d.as_mut_ptr(),
    );
    d
}

/// Scores of keys kb + 16 u + lane (u = 0, 1) for the 8 A rows (component = row): per block bl, a fresh 2-step DPAS
/// chain on 1152 + k, minus 1152 * qsum, times the block scale, summed over the blocks in order.
#[inline(always)]
unsafe fn scores(sk: Surface, hk: u32, kb: u32, qa: *const u16, qsum: *const f32) -> [[f32; 8]; 2] {
    let mut cc = [[0f32; 8]; 2];
    unroll!(2, 2, |u| {
        let ksr = blk2d::read_transpose_32b_16r8x1c(sk, ((hk * REC + DATA) / 4) as i32, (kb + 16 * u as u32) as i32).to_array();
        let ks4 = [ksr[0], ksr[1], ksr[2], ksr[3]];
        let mut acc = [0f32; 8];
        unroll!(8, 8, |bl| {
            let kr = blk2d::read_transpose_32b_16r8x1c(sk, ((hk * REC + bl as u32 * 32) / 4) as i32, (kb + 16 * u as u32) as i32).to_array();
            let mut cb = F32x8::splat(0.0);
            unroll!(2, 2, |tt| {
                let a: I16x8 = bits(blk::local::read_u16x8(qa.add((2 * bl + tt) * 128)));
                cb = dpas::f16_m8(a, kv16(kr[4 * tt], kr[4 * tt + 1], kr[4 * tt + 2], kr[4 * tt + 3]), cb);
            });
            let cba = cb.to_array();
            let ksc = half_of(&ks4, bl);
            let qs8 = crate::ext::vload8_local(bl, qsum);
            unroll!(8, 8, |j| acc[j] = acc[j] + (cba[j] - qs8[j]) * ksc);
        });
        cc[u] = acc;
    });
    cc
}

/// The P V products of one 32-key step into o[16] (component = A row, lane = dim): pp[u] are the probabilities of keys
/// kb + 16 u + perm4(lane), svp[u] the V block scales of the same keys (8 fp16 in 4 dwords).
#[inline(always)]
unsafe fn pv(sv: Surface, hk: u32, kb: u32, pp: &[[f32; 8]; 2], svp: &[[u32; 4]; 2], bneg: I32x8, o: &mut [F32x8; 16]) {
    unroll!(8, 8, |bl| {
        let vr = rd_t8_32r16x2c(sv, (hk * REC + bl as u32 * 32) as i32, kb as i32);
        let mut a = [I16x8::splat(0); 2];
        unroll!(2, 2, |u| {
            let sc = half_of(&svp[u], bl);
            let mut h = [0i16; 8];
            unroll!(8, 8, |j| h[j] = bits::<f16, i16>((pp[u][j] * sc) as f16));
            a[u] = I16x8::from_array(h);
        });
        unroll!(2, 2, |tt| {
            unroll!(2, 2, |u| {
                let w = 8 * tt + 4 * u;
                o[2 * bl + tt] = dpas::f16_m8(a[u], kv16(vr[w], vr[w + 1], vr[w + 2], vr[w + 3]), o[2 * bl + tt]);
                o[2 * bl + tt] = dpas::f16_m8(a[u], bneg, o[2 * bl + tt]);
            });
        });
    });
}

/// Grid (4 kv heads * rows, chunks), local 128; row z has len0 + z keys, q [row][24][256] bf16; po [row][maxc][24][256]
/// fp32, pm / pl [row][maxc][24].
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn attn_dec_partial(q: *const u16, kc: *const u8, vc: *const u8, po: *mut f32, pm: *mut f32, pl: *mut f32, len0: u32, scale: f32, maxc: u32) {
    attr::reqd_work_group_size::<128, 1, 1>();
    let qa = slm::alloc::<u16, { 16 * 8 * 16 }, 0>(); // [dim step][head (6 + 2 zero rows)][dim of the step], fp16
    let ms = slm::alloc::<f32, { 8 * GQ }, 1>();
    let ls = slm::alloc::<f32, { 8 * GQ }, 2>();
    let accs = slm::alloc::<f32, { GQ * HD }, 3>();
    let qsum = slm::alloc::<f32, 64, 4>(); // [block][head]: 1152 * the sum of the fp16 q over the block
    let cneg = slm::alloc::<u32, 128, 5>(); // fp16 -1152 pairs (the B operand of the offset product)
    let hk = id::group_id(0) & 3;
    let z = id::group_id(0) >> 2;
    let c = id::group_id(1);
    let lid = id::local_id(0);
    let sgi = id::sg_id();
    let lane = id::lane();
    let len = len0 + z;
    let k0 = c * LC;
    if k0 >= len {
        return;
    }
    let qr = q.add(z as usize * NH * HD + hk as usize * GQ * HD);
    // 16 entries an item, unrolled so the 16 q loads are in flight together (a runtime loop issued them one by one)
    unroll!(16, 16, |it| {
        let i = lid + 128 * it as u32;
        let (t, h, l) = (i >> 7, (i >> 4) & 7, i & 15);
        *qa.add(i as usize) = if (h as usize) < GQ { bf_to_h(*qr.add(h as usize * HD + (t * 16 + perm4(l)) as usize)) } else { 0 };
    });
    *cneg.add(lid as usize) = 0xE480_E480;
    if lid < 64 {
        let (b, h) = (lid >> 3, lid & 7);
        let mut sum = 0.0f32;
        if (h as usize) < GQ {
            let mut i = 0u32;
            while i < 32 {
                sum += bf_to_hf(*qr.add(h as usize * HD + (b * 32 + i) as usize));
                i += 1;
            }
        }
        *qsum.add(lid as usize) = 1152.0 * sum;
    }
    slm::barrier();
    let ks = k0 + sgi * KS;
    let ke = (ks + KS).min(len);
    let pl4 = perm4(lane);
    let bneg: I32x8 = bits(blk::local::read_u32x8(cneg));
    let mut m = [NEG_INF; GQ];
    let mut ll = [0f32; 8];
    let mut o = [F32x8::splat(0.0); 16];
    let sk = Surface { base: kc, width_bytes: RW as i32, height: len as i32, pitch_bytes: RW as i32 };
    let sv = Surface { base: vc, width_bytes: RW as i32, height: len as i32, pitch_bytes: RW as i32 };
    // The key loop runs at least once (a do-while): a sub-group without keys in this chunk (ks >= ke, the end of a row)
    // makes one pass over a block past the row's end, where every key is masked (the 2D loads read zeros outside the
    // surface), the maximum stays -inf and `none` keeps p = 0, alpha = 1: ll, m and o stay exactly 0, -inf, +0 (o gains
    // only exact zero products), the values the reference leaves when it skips the loop. With keys, mn is finite in
    // every step (lane 0's first key is valid), `none` is never true and the arithmetic is the reference's. (A guarded
    // loop gives the o tiles an exit phi that IGC split into scalars: 1.9 KB of spill; the reference spills none.)
    let mut kb = ks;
    loop {
        let cc = scores(sk, hk, kb, qa, qsum);
        let mut p = [[0f32; 8]; 2];
        let mut al = [0f32; GQ];
        let mut moved = false;
        let ok0 = kb + lane < ke;
        let ok1 = kb + 16 + lane < ke;
        unroll!(8, GQ, |h| {
            let s0 = if ok0 { cc[0][h] * scale } else { NEG_INF };
            let s1 = if ok1 { cc[1][h] * scale } else { NEG_INF };
            let mn = math::fmax(m[h], sg::reduce_max_f32(math::fmax(s0, s1)));
            let none = mn == NEG_INF;
            moved |= mn != m[h];
            al[h] = if none || mn == m[h] { 1.0 } else { math::exp(m[h] - mn) };
            p[0][h] = if none { 0.0 } else { math::exp(s0 - mn) };
            p[1][h] = if none { 0.0 } else { math::exp(s1 - mn) };
            ll[h] = ll[h] * al[h] + p[0][h] + p[1][h];
            m[h] = mn;
        });
        if moved {
            let a8 = F32x8::from_array([al[0], al[1], al[2], al[3], al[4], al[5], 1.0, 1.0]);
            unroll!(16, 16, |n| o[n] *= a8);
        }
        let mut pp = [[0f32; 8]; 2];
        let mut svp = [[0u32; 4]; 2];
        unroll!(2, 2, |u| {
            let vsr = blk2d::read_transpose_32b_16r8x1c(sv, ((hk * REC + DATA) / 4) as i32, (kb + 16 * u as u32) as i32).to_array();
            unroll!(4, 4, |j| svp[u][j] = vsr[j].shuffle(pl4));
            unroll!(8, GQ, |h| pp[u][h] = p[u][h].shuffle(pl4));
        });
        pv(sv, hk, kb, &pp, &svp, bneg, &mut o);
        kb += 32;
        if kb >= ke {
            break;
        }
    }
    unroll!(8, GQ, |h| {
        let l = sg::reduce_add_f32(ll[h]);
        if lane == 0 {
            *ms.add(sgi as usize * GQ + h) = m[h];
            *ls.add(sgi as usize * GQ + h) = l;
        }
    });
    slm::barrier();
    let mut f = [0f32; GQ];
    unroll!(8, GQ, |h| {
        let mut mx = NEG_INF;
        let mut s = 0;
        while s < 8 {
            mx = math::fmax(mx, *ms.add(s * GQ + h));
            s += 1;
        }
        f[h] = math::exp(m[h] - mx);
    });
    // the sub-groups add their f o into accs in order 0..7 (r = 0 stores, the others add), a barrier after each
    if sgi == 0 {
        unroll!(16, 16, |n| {
            let on = o[n].to_array();
            unroll!(8, GQ, |h| *accs.add(h * HD + n * 16 + lane as usize) = f[h] * on[h]);
        });
    }
    slm::barrier();
    let mut r = 1u32;
    while r < 8 {
        if sgi == r {
            unroll!(16, 16, |n| {
                let on = o[n].to_array();
                unroll!(8, GQ, |h| {
                    let a = accs.add(h * HD + n * 16 + lane as usize);
                    *a = *a + f[h] * on[h];
                });
            });
        }
        slm::barrier();
        r += 1;
    }
    let cb = z as usize * maxc as usize + c as usize;
    let pr = po.add((cb * NH + hk as usize * GQ) * HD); // the 6 heads of the kv head are consecutive
    let mut i = lid as usize;
    while i < GQ * HD / 4 {
        unroll!(4, 4, |j| *pr.add(4 * i + j) = *accs.add(4 * i + j));
        i += 128;
    }
    if (lid as usize) < GQ {
        let h = lid as usize;
        let mut mx = NEG_INF;
        let mut s = 0;
        while s < 8 {
            mx = math::fmax(mx, *ms.add(s * GQ + h));
            s += 1;
        }
        let mut den = 0.0f32;
        let mut s = 0;
        while s < 8 {
            den = den + *ls.add(s * GQ + h) * math::exp(*ms.add(s * GQ + h) - mx);
            s += 1;
        }
        *pm.add(cb * NH + hk as usize * GQ + h) = mx;
        *pl.add(cb * NH + hk as usize * GQ + h) = den;
    }
}

/// The partials for a window of `rows` rows with the (row, head) pairs of a kv head packed 8 to a tile: pair g = 8 t + i
/// (i the tile row) is row g / 6, head g % 6, with its own key count len0 + g / 6; the rows of a tile share the K / V
/// reads. Each pair's arithmetic is attn_dec_partial's for its row alone, so a window row equals single-row decode
/// bitwise. A work-group runs TGN tiles of one (kv head, chunk) side by side (8 sub-groups each).
/// Grid (4 kv heads * ceil(tiles / TGN), chunks of the longest row), local 128 TGN.
#[inline(always)]
unsafe fn dec_pack<const TGN: u32>(
    q: *const u16,
    kc: *const u8,
    vc: *const u8,
    po: *mut f32,
    pm: *mut f32,
    pl: *mut f32,
    len0: u32,
    scale: f32,
    maxc: u32,
    rows: u32,
    qa: *mut u16,
    ms: *mut f32,
    ls: *mut f32,
    accs: *mut f32,
    qsum: *mut f32,
    cneg: *mut u32,
) {
    let tl = id::sg_id() >> 3; // tile of the group
    let sgi = id::sg_id() & 7; // run (128 keys) of the chunk
    let hk = id::group_id(0) & 3;
    let t = (id::group_id(0) >> 2) * TGN + tl;
    let c = id::group_id(1);
    let lid = id::local_id(0) & 127;
    let lane = id::lane();
    let qa = qa.add(tl as usize * 16 * 8 * 16);
    let ms = ms.add(tl as usize * 64);
    let ls = ls.add(tl as usize * 64);
    let accs = accs.add(tl as usize * 8 * HD);
    let qsum = qsum.add(tl as usize * 64);
    let gq = GQ as u32;
    let g0 = 8 * t;
    let gn = if g0 < gq * rows { 8u32.min(gq * rows - g0) } else { 0 }; // valid tile rows 0 .. gn - 1
    let lenx = if gn != 0 { len0 + (g0 + gn - 1) / gq } else { 0 }; // keys of the longest row of the tile
    let k0 = c * LC;
    let act = k0 < lenx; // a tile past the window or a chunk past its rows: no work, no writes (the barriers are shared)
    if TGN == 1 && !act {
        return;
    }
    unroll!(16, 16, |it| {
        let i = lid + 128 * it as u32;
        let (s, r, l) = (i >> 7, (i >> 4) & 7, i & 15);
        let g = g0 + r;
        *qa.add(i as usize) = if r < gn {
            bf_to_h(*q.add((((g / gq) as usize * NH + (hk * gq + g % gq) as usize) * HD) + (s * 16 + perm4(l)) as usize))
        } else {
            0
        };
    });
    if tl == 0 {
        *cneg.add(lid as usize) = 0xE480_E480;
    }
    if lid < 64 {
        let (b, r) = (lid >> 3, lid & 7);
        let g = g0 + r;
        let mut sum = 0.0f32;
        if r < gn {
            let mut i = 0u32;
            while i < 32 {
                sum += bf_to_hf(*q.add((((g / gq) as usize * NH + (hk * gq + g % gq) as usize) * HD) + (b * 32 + i) as usize));
                i += 1;
            }
        }
        *qsum.add(lid as usize) = 1152.0 * sum;
    }
    slm::barrier();
    let ks = k0 + sgi * KS;
    let kex = if act { (ks + KS).min(lenx) } else { ks };
    // keys of tile row i in this run: a tile holds pairs of at most two rows, the first row's up to tile row isp
    let ra = g0 / gq;
    let isp = (gq * (ra + 1) - g0).min(gn);
    let kea = (ks + KS).min(len0 + ra);
    let keb = (ks + KS).min(len0 + ra + 1);
    let pl4 = perm4(lane);
    let bneg: I32x8 = bits(blk::local::read_u32x8(cneg));
    let mut m = [NEG_INF; 8];
    let mut ll = [0f32; 8];
    let mut o = [F32x8::splat(0.0); 16];
    let sk = Surface { base: kc, width_bytes: RW as i32, height: lenx as i32, pitch_bytes: RW as i32 };
    let sv = Surface { base: vc, width_bytes: RW as i32, height: lenx as i32, pitch_bytes: RW as i32 };
    let mut kb = ks;
    while kb < kex {
        let cc = scores(sk, hk, kb, qa, qsum);
        let mut p = [[0f32; 8]; 2];
        let mut al = [0f32; 8];
        let mut moved = false;
        unroll!(8, 8, |i| {
            let kei = if (i as u32) < isp { kea } else if (i as u32) < gn { keb } else { 0 };
            let s0 = if kb + lane < kei { cc[0][i] * scale } else { NEG_INF };
            let s1 = if kb + 16 + lane < kei { cc[1][i] * scale } else { NEG_INF };
            let mn = math::fmax(m[i], sg::reduce_max_f32(math::fmax(s0, s1)));
            let none = mn == NEG_INF; // no key of this row yet (other rows of the tile are longer)
            moved |= mn != m[i];
            al[i] = if none || mn == m[i] { 1.0 } else { math::exp(m[i] - mn) };
            p[0][i] = if none { 0.0 } else { math::exp(s0 - mn) };
            p[1][i] = if none { 0.0 } else { math::exp(s1 - mn) };
            ll[i] = ll[i] * al[i] + p[0][i] + p[1][i];
            m[i] = mn;
        });
        if moved {
            let a8 = F32x8::from_array(al);
            unroll!(16, 16, |n| o[n] *= a8);
        }
        let mut pp = [[0f32; 8]; 2];
        let mut svp = [[0u32; 4]; 2];
        unroll!(2, 2, |u| {
            let vsr = blk2d::read_transpose_32b_16r8x1c(sv, ((hk * REC + DATA) / 4) as i32, (kb + 16 * u as u32) as i32).to_array();
            unroll!(4, 4, |j| svp[u][j] = vsr[j].shuffle(pl4));
            unroll!(8, 8, |j| pp[u][j] = p[u][j].shuffle(pl4));
        });
        pv(sv, hk, kb, &pp, &svp, bneg, &mut o);
        kb += 32;
    }
    unroll!(8, 8, |i| {
        let l = sg::reduce_add_f32(ll[i]);
        if lane == 0 {
            *ms.add(sgi as usize * 8 + i) = m[i];
            *ls.add(sgi as usize * 8 + i) = l;
        }
    });
    slm::barrier();
    let mut f = [0f32; 8];
    unroll!(8, 8, |i| {
        let mut mx = NEG_INF;
        let mut s = 0;
        while s < 8 {
            mx = math::fmax(mx, *ms.add(s * 8 + i));
            s += 1;
        }
        f[i] = if m[i] == NEG_INF { 0.0 } else { math::exp(m[i] - mx) };
    });
    if sgi == 0 {
        unroll!(16, 16, |n| {
            let on = o[n].to_array();
            unroll!(8, 8, |i| *accs.add(i * HD + n * 16 + lane as usize) = f[i] * on[i]);
        });
    }
    slm::barrier();
    let mut r = 1u32;
    while r < 8 {
        if sgi == r {
            unroll!(16, 16, |n| {
                let on = o[n].to_array();
                unroll!(8, 8, |i| {
                    let a = accs.add(i * HD + n * 16 + lane as usize);
                    *a = *a + f[i] * on[i];
                });
            });
        }
        slm::barrier();
        r += 1;
    }
    let mut e = lid as usize;
    while e < 8 * HD / 4 {
        let i = e / (HD / 4);
        let g = g0 + i as u32;
        if act && (i as u32) < gn {
            let dst = po.add((((g / gq) as usize * maxc as usize + c as usize) * NH + (hk * gq + g % gq) as usize) * HD + 4 * (e % (HD / 4)));
            unroll!(4, 4, |j| *dst.add(j) = *accs.add(4 * e + j));
        }
        e += 128;
    }
    if act && lid < gn {
        let i = lid as usize;
        let g = g0 + lid;
        let mut mx = NEG_INF;
        let mut s = 0;
        while s < 8 {
            mx = math::fmax(mx, *ms.add(s * 8 + i));
            s += 1;
        }
        let mut den = 0.0f32;
        let mut s = 0;
        while s < 8 {
            let msv = *ms.add(s * 8 + i);
            den = den + if msv == NEG_INF { 0.0 } else { *ls.add(s * 8 + i) * math::exp(msv - mx) };
            s += 1;
        }
        let cb = ((g / gq) as usize * maxc as usize + c as usize) * NH + (hk * gq + g % gq) as usize;
        *pm.add(cb) = mx;
        *pl.add(cb) = den;
    }
}

macro_rules! pack_kernel {
    ($name:ident, $tgn:literal) => {
        /// dec_pack with $tgn tiles a work-group. Grid (4 * ceil(tiles / TGN), chunks), local 128 * TGN.
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            q: *const u16,
            kc: *const u8,
            vc: *const u8,
            po: *mut f32,
            pm: *mut f32,
            pl: *mut f32,
            len0: u32,
            scale: f32,
            maxc: u32,
            rows: u32,
        ) {
            attr::reqd_work_group_size::<{ 128 * $tgn }, 1, 1>();
            let qa = slm::alloc::<u16, { $tgn * 16 * 8 * 16 }, 0>();
            let ms = slm::alloc::<f32, { $tgn * 64 }, 1>();
            let ls = slm::alloc::<f32, { $tgn * 64 }, 2>();
            let accs = slm::alloc::<f32, { $tgn * 8 * HD }, 3>();
            let qsum = slm::alloc::<f32, { $tgn * 64 }, 4>();
            let cneg = slm::alloc::<u32, 128, 5>();
            dec_pack::<$tgn>(q, kc, vc, po, pm, pl, len0, scale, maxc, rows, qa, ms, ls, accs, qsum, cneg);
        }
    };
}
pack_kernel!(attn_dec_pack, 1);
pack_kernel!(attn_dec_pack2, 2);
pack_kernel!(attn_dec_pack4, 4);

// ---- tree verify windows ----------------------------------------------------------------------------------------------
// The rows of the window form a draft tree (tree.rs: topological rows, row t at position pos0 + depth[t], its K / V at
// cache slot pos0 + t). Row t must equal attn_dec_partial of its root -> t path alone (len pos0 + depth[t] + 1, those
// positions holding the path): the steps below kbase = pos0 & !31 read the shared cache as dec_pack does; the (at most
// two) steps from kbase read each row's tail window (tree.rs kv_tree_tail: keys kbase .. kbase + 63 of that row's path,
// zeros past it), one pass per row of the tile with the other row's pairs fully masked (no keys: alpha 1, p 0, o and ll
// unchanged bit for bit, as the masked keys of dec_pack).

const TAIL_KEYS: u32 = 64;

/// The tail window of tree row r as a 2D surface ([64 keys][4][272 B]).
#[inline(always)]
unsafe fn tail_surface(base: *const u8, r: u32) -> Surface {
    Surface { base: base.add((r * TAIL_KEYS * RW) as usize), width_bytes: RW as i32, height: TAIL_KEYS as i32, pitch_bytes: RW as i32 }
}

/// One 32-key step for the 8 pairs of a tile: keys kb + 16 u + lane (logical) read at rows y0 + 16 u of the surfaces;
/// pair i sees keys below kei[i]. The arithmetic of dec_pack's loop body.
#[inline(always)]
unsafe fn tree_step(
    sk: Surface,
    sv: Surface,
    hk: u32,
    y0: u32,
    kb: u32,
    kei: &[u32; 8],
    scale: f32,
    lane: u32,
    pl4: u32,
    bneg: I32x8,
    qa: *const u16,
    qsum: *const f32,
    m: &mut [f32; 8],
    ll: &mut [f32; 8],
    o: &mut [F32x8; 16],
) {
    let cc = scores(sk, hk, y0, qa, qsum);
    let mut p = [[0f32; 8]; 2];
    let mut al = [0f32; 8];
    let mut moved = false;
    unroll!(8, 8, |i| {
        let s0 = if kb + lane < kei[i] { cc[0][i] * scale } else { NEG_INF };
        let s1 = if kb + 16 + lane < kei[i] { cc[1][i] * scale } else { NEG_INF };
        let mn = math::fmax(m[i], sg::reduce_max_f32(math::fmax(s0, s1)));
        let none = mn == NEG_INF;
        moved |= mn != m[i];
        al[i] = if none || mn == m[i] { 1.0 } else { math::exp(m[i] - mn) };
        p[0][i] = if none { 0.0 } else { math::exp(s0 - mn) };
        p[1][i] = if none { 0.0 } else { math::exp(s1 - mn) };
        ll[i] = ll[i] * al[i] + p[0][i] + p[1][i];
        m[i] = mn;
    });
    if moved {
        let a8 = F32x8::from_array(al);
        unroll!(16, 16, |n| o[n] *= a8);
    }
    let mut pp = [[0f32; 8]; 2];
    let mut svp = [[0u32; 4]; 2];
    unroll!(2, 2, |u| {
        let vsr = blk2d::read_transpose_32b_16r8x1c(sv, ((hk * REC + DATA) / 4) as i32, (y0 + 16 * u as u32) as i32).to_array();
        unroll!(4, 4, |j| svp[u][j] = vsr[j].shuffle(pl4));
        unroll!(8, 8, |j| pp[u][j] = p[u][j].shuffle(pl4));
    });
    pv(sv, hk, y0, &pp, &svp, bneg, o);
}

/// dec_pack for a tree window: pair g = 8 t + i is row g / 6, head g % 6 with pos0 + depth[row] + 1 keys.
/// tree u32 [rows][4] (tree.rs), tk / tv the tail windows [rows][64][4][272] of kv_tree_tail.
/// Grid (4 kv heads * ceil(tiles / TGN), chunks of the longest row), local 128 TGN.
#[inline(always)]
unsafe fn dec_tree<const TGN: u32>(
    q: *const u16,
    kc: *const u8,
    vc: *const u8,
    po: *mut f32,
    pm: *mut f32,
    pl: *mut f32,
    pos0: u32,
    scale: f32,
    maxc: u32,
    rows: u32,
    tree: *const u32,
    tk: *const u8,
    tv: *const u8,
    qa: *mut u16,
    ms: *mut f32,
    ls: *mut f32,
    accs: *mut f32,
    qsum: *mut f32,
    cneg: *mut u32,
) {
    let tl = id::sg_id() >> 3;
    let sgi = id::sg_id() & 7;
    let hk = id::group_id(0) & 3;
    let t = (id::group_id(0) >> 2) * TGN + tl;
    let c = id::group_id(1);
    let lid = id::local_id(0) & 127;
    let lane = id::lane();
    let qa = qa.add(tl as usize * 16 * 8 * 16);
    let ms = ms.add(tl as usize * 64);
    let ls = ls.add(tl as usize * 64);
    let accs = accs.add(tl as usize * 8 * HD);
    let qsum = qsum.add(tl as usize * 64);
    let gq = GQ as u32;
    let g0 = 8 * t;
    let gn = if g0 < gq * rows { 8u32.min(gq * rows - g0) } else { 0 };
    // the tile's rows: ra (tile rows 0 .. isp - 1) and, when isp < gn, ra + 1
    let ra = g0 / gq;
    let isp = (gq * (ra + 1) - g0).min(gn);
    let two = isp < gn;
    let lena = if gn != 0 { pos0 + *tree.add(4 * ra as usize + 1) + 1 } else { 0 };
    let lenb = if two { pos0 + *tree.add(4 * (ra + 1) as usize + 1) + 1 } else { 0 };
    let lenx = lena.max(lenb);
    let k0 = c * LC;
    let act = k0 < lenx;
    if TGN == 1 && !act {
        return;
    }
    unroll!(16, 16, |it| {
        let i = lid + 128 * it as u32;
        let (s, r, l) = (i >> 7, (i >> 4) & 7, i & 15);
        let g = g0 + r;
        *qa.add(i as usize) = if r < gn {
            bf_to_h(*q.add((((g / gq) as usize * NH + (hk * gq + g % gq) as usize) * HD) + (s * 16 + perm4(l)) as usize))
        } else {
            0
        };
    });
    if tl == 0 {
        *cneg.add(lid as usize) = 0xE480_E480;
    }
    if lid < 64 {
        let (b, r) = (lid >> 3, lid & 7);
        let g = g0 + r;
        let mut sum = 0.0f32;
        if r < gn {
            let mut i = 0u32;
            while i < 32 {
                sum += bf_to_hf(*q.add((((g / gq) as usize * NH + (hk * gq + g % gq) as usize) * HD) + (b * 32 + i) as usize));
                i += 1;
            }
        }
        *qsum.add(lid as usize) = 1152.0 * sum;
    }
    slm::barrier();
    let ks = k0 + sgi * KS;
    let kex = if act { (ks + KS).min(lenx) } else { ks };
    let kea = (ks + KS).min(lena);
    let keb = (ks + KS).min(lenb);
    let mut kei = [0u32; 8];
    let mut kei_a = [0u32; 8];
    let mut kei_b = [0u32; 8];
    unroll!(8, 8, |i| {
        let ii = i as u32;
        kei[i] = if ii < isp { kea } else if ii < gn { keb } else { 0 };
        kei_a[i] = if ii < isp { kea } else { 0 };
        kei_b[i] = if ii >= isp && ii < gn { keb } else { 0 };
    });
    let kbase = pos0 & !31;
    let pl4 = perm4(lane);
    let bneg: I32x8 = bits(blk::local::read_u32x8(cneg));
    let mut m = [NEG_INF; 8];
    let mut ll = [0f32; 8];
    let mut o = [F32x8::splat(0.0); 16];
    let sk = Surface { base: kc, width_bytes: RW as i32, height: lenx as i32, pitch_bytes: RW as i32 };
    let sv = Surface { base: vc, width_bytes: RW as i32, height: lenx as i32, pitch_bytes: RW as i32 };
    // one step body (a single inlined copy): below kbase one pass over the shared cache, from kbase one pass for each row
    // of the tile over the tail window of that row
    let mut kb = ks;
    let mut ps = 0u32;
    while kb < kex {
        let tail_step = kb >= kbase;
        let (sk1, sv1, y0) = if !tail_step {
            (sk, sv, kb)
        } else {
            let r = ra + ps;
            (tail_surface(tk, r), tail_surface(tv, r), kb - kbase)
        };
        let mut k1 = [0u32; 8];
        unroll!(8, 8, |i| k1[i] = if !tail_step { kei[i] } else if ps == 0 { kei_a[i] } else { kei_b[i] });
        tree_step(sk1, sv1, hk, y0, kb, &k1, scale, lane, pl4, bneg, qa, qsum, &mut m, &mut ll, &mut o);
        if tail_step && two && ps == 0 {
            ps = 1;
        } else {
            ps = 0;
            kb += 32;
        }
    }
    unroll!(8, 8, |i| {
        let l = sg::reduce_add_f32(ll[i]);
        if lane == 0 {
            *ms.add(sgi as usize * 8 + i) = m[i];
            *ls.add(sgi as usize * 8 + i) = l;
        }
    });
    slm::barrier();
    let mut f = [0f32; 8];
    unroll!(8, 8, |i| {
        let mut mx = NEG_INF;
        let mut s = 0;
        while s < 8 {
            mx = math::fmax(mx, *ms.add(s * 8 + i));
            s += 1;
        }
        f[i] = if m[i] == NEG_INF { 0.0 } else { math::exp(m[i] - mx) };
    });
    if sgi == 0 {
        unroll!(16, 16, |n| {
            let on = o[n].to_array();
            unroll!(8, 8, |i| *accs.add(i * HD + n * 16 + lane as usize) = f[i] * on[i]);
        });
    }
    slm::barrier();
    let mut r = 1u32;
    while r < 8 {
        if sgi == r {
            unroll!(16, 16, |n| {
                let on = o[n].to_array();
                unroll!(8, 8, |i| {
                    let a = accs.add(i * HD + n * 16 + lane as usize);
                    *a = *a + f[i] * on[i];
                });
            });
        }
        slm::barrier();
        r += 1;
    }
    let mut e = lid as usize;
    while e < 8 * HD / 4 {
        let i = e / (HD / 4);
        let g = g0 + i as u32;
        let rows_keys = if (i as u32) < isp { lena } else { lenb };
        if act && (i as u32) < gn && k0 < rows_keys {
            let dst = po.add((((g / gq) as usize * maxc as usize + c as usize) * NH + (hk * gq + g % gq) as usize) * HD + 4 * (e % (HD / 4)));
            unroll!(4, 4, |j| *dst.add(j) = *accs.add(4 * e + j));
        }
        e += 128;
    }
    if act && lid < gn {
        let i = lid as usize;
        let g = g0 + lid;
        let rows_keys = if lid < isp { lena } else { lenb };
        let mut mx = NEG_INF;
        let mut s = 0;
        while s < 8 {
            mx = math::fmax(mx, *ms.add(s * 8 + i));
            s += 1;
        }
        let mut den = 0.0f32;
        let mut s = 0;
        while s < 8 {
            let msv = *ms.add(s * 8 + i);
            den = den + if msv == NEG_INF { 0.0 } else { *ls.add(s * 8 + i) * math::exp(msv - mx) };
            s += 1;
        }
        if k0 < rows_keys {
            let cb = ((g / gq) as usize * maxc as usize + c as usize) * NH + (hk * gq + g % gq) as usize;
            *pm.add(cb) = mx;
            *pl.add(cb) = den;
        }
    }
}

macro_rules! tree_kernel {
    ($name:ident, $tgn:literal) => {
        /// dec_tree with $tgn tiles a work-group. Grid (4 * ceil(tiles / TGN), chunks of the longest row), local 128 * TGN.
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(
            q: *const u16,
            kc: *const u8,
            vc: *const u8,
            po: *mut f32,
            pm: *mut f32,
            pl: *mut f32,
            pos0: u32,
            scale: f32,
            maxc: u32,
            rows: u32,
            tree: *const u32,
            tk: *const u8,
            tv: *const u8,
        ) {
            attr::reqd_work_group_size::<{ 128 * $tgn }, 1, 1>();
            let qa = slm::alloc::<u16, { $tgn * 16 * 8 * 16 }, 0>();
            let ms = slm::alloc::<f32, { $tgn * 64 }, 1>();
            let ls = slm::alloc::<f32, { $tgn * 64 }, 2>();
            let accs = slm::alloc::<f32, { $tgn * 8 * HD }, 3>();
            let qsum = slm::alloc::<f32, { $tgn * 64 }, 4>();
            let cneg = slm::alloc::<u32, 128, 5>();
            dec_tree::<$tgn>(q, kc, vc, po, pm, pl, pos0, scale, maxc, rows, tree, tk, tv, qa, ms, ls, accs, qsum, cneg);
        }
    };
}
tree_kernel!(attn_dec_tree, 1);
tree_kernel!(attn_dec_tree2, 2);
tree_kernel!(attn_dec_tree4, 4);
