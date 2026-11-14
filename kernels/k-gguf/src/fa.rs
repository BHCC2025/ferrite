//! Prompt attention on tensor cores over the q8 KV cache (kq / vq int8 [cell][nkv][256], ks / vs f32
//! [cell][nkv][8], one scale per 32 values), dense causal or QSA-sparse, output gated by sigmoid(gate).
//!
//! Row r (cell pos + r) attends cell c iff bit c / 4 of its block bitmap is set and c <= pos + r. `ggq_fa_bits`
//! builds the bitmaps from the selection lists of `fx_topk` (any entry order): dense rows (seln == 0) get every
//! block up to their own, QSA rows the blocks of their selected cells (top blocks + the open tail block).
//! `ggq_fa_union` turns each tile of 16 rows into the ascending list of the 8-cell windows any of its rows uses,
//! with a row mask per window. `ggq_fa_tc` then runs flash attention per (tile, kv head, group of 4 q heads): one
//! warp per q head, 16 rows, stages of 2 windows shared by the 4 warps, K / V dequantized to f16 in shared memory,
//! Q . K (m16n8k16) and P . V (m16n8k8, one window per mma) f16 -> f32, online softmax per window in f32 (exp2),
//! windows in ascending order. A row's result depends only on its own cells: deterministic, independent of the
//! selection order, of the rows sharing its tile, and of the prompt-group split.
use core::arch::asm;
use kcommon::mma::*;
use kcommon::smem::*;
use kcommon::*;

const D: u32 = 256;
const QR: u32 = 16; // query rows per tile
pub const FA_HW: u32 = 4; // q heads (warps) per CTA
const KT: u32 = 16; // keys per key tile
const LDH: u32 = D + 8; // halves per f16 smem row (528 B: conflict-free ldmatrix)
const RAW_V: u32 = KT * D;
const RAW_KS: u32 = 2 * KT * D;
const RAW_VS: u32 = RAW_KS + KT * 32;
const RAW: u32 = RAW_VS + KT * 32; // one raw stage: K, V int8 + scales
const H_K: u32 = 2 * RAW;
const H_V: u32 = H_K + KT * LDH * 2;
pub const FA_SMEM: u32 = H_V + KT * LDH * 2;

#[inline(always)]
fn pack_h2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe { asm!("cvt.rn.f16x2.f32 {r}, {h}, {l};", r = out(reg32) r, h = in(reg32) hi, l = in(reg32) lo, options(pure, nomem, nostack)) }
    r
}
/// Sum of the two f16 halves, in f32.
#[inline(always)]
fn h2_sum(v: u32) -> f32 {
    let (a, b): (f32, f32);
    unsafe {
        asm!("{{ .reg .f16 x, y; mov.b32 {{x, y}}, {v}; cvt.f32.f16 {a}, x; cvt.f32.f16 {b}, y; }}", v = in(reg32) v,
            a = out(reg32) a, b = out(reg32) b, options(pure, nomem, nostack))
    }
    a + b
}
#[inline(always)]
unsafe fn lds_v4(addr: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    asm!("ld.shared.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];", a = out(reg32) a, b = out(reg32) b, c = out(reg32) c,
        d = out(reg32) d, p = in(reg32) addr, options(readonly, nostack));
    [a, b, c, d]
}
#[inline(always)]
unsafe fn sts_v4(addr: u32, v: [u32; 4]) {
    asm!("st.shared.v4.u32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg32) addr, a = in(reg32) v[0], b = in(reg32) v[1],
        c = in(reg32) v[2], d = in(reg32) v[3], options(nostack));
}
#[inline(always)]
unsafe fn lds_f32(addr: u32) -> f32 {
    let r: f32;
    asm!("ld.shared.f32 {r}, [{p}];", r = out(reg32) r, p = in(reg32) addr, options(readonly, nostack));
    r
}
/// Signed byte e of w as an exact f32 (full-rate ops: no I2F).
#[inline(always)]
fn sb2f(w: u32, e: u32) -> f32 {
    let v = ((w << (24 - 8 * e)) as i32) >> 24;
    f32::from_bits((v + 0x4b40_0000) as u32) - 12582912.0
}

/// Row block bitmaps: bits [rows][ldb] u32, bit b of row r = row r uses block b (cells 4b..4b+3).
/// grid = rows, block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_fa_bits(bits: *mut u32, sel: *const u32, seln: *const u32, ldsel: u32,
    prm: *const u32, ldb: u32) {
    let r = bid_x();
    let t = tid_x();
    let p = *prm + r;
    let nb = p / 4 + 1;
    let nw = nb.div_ceil(32);
    let row = bits.add((r * ldb) as usize);
    let ns = *seln.add(r as usize);
    let mut w = t;
    while w < nw {
        let v = if ns != 0 { 0 } else if (w + 1) * 32 <= nb { !0u32 } else { (1u32 << (nb - w * 32)) - 1 };
        *row.add(w as usize) = v;
        w += bdim_x();
    }
    if ns == 0 {
        return;
    }
    sync_threads();
    let sr = sel.add((r * ldsel) as usize);
    let mut i = t;
    while i < ns {
        let b = *sr.add(i as usize) / 4;
        asm!("red.global.or.b32 [{a}], {v};", a = in(reg64) row.add((b / 32) as usize), v = in(reg32) 1u32 << (b % 32),
            options(nostack));
        i += bdim_x();
    }
}

/// Per tile of 16 rows: list[tile][2i], [2i + 1] = window w (cells 8w..8w+7, QSA blocks 2w, 2w+1) and its row
/// mask (bit r: row r uses block 2w, bit 16 + r: block 2w + 1), ascending over the windows any row of the tile
/// uses, padded to an even count with mask-0 copies of the last window; cnt[tile] = padded window count.
/// grid = tiles, block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_fa_union(list: *mut u32, cnt: *mut u32, bits: *const u32, ldb: u32, ldl: u32,
    prm: *const u32, rows: u32) {
    let wsum: Shared<u32> = kcommon::static_shared!(fa_wsum, u32, 64);
    let tile = bid_x();
    let t = tid_x();
    let r0 = tile * QR;
    let nr = if rows - r0 < QR { rows - r0 } else { QR };
    let p0 = *prm + r0;
    let nwr = |i: u32| ((p0 + i) / 4 + 1).div_ceil(32);
    let nwmax = nwr(nr - 1);
    let out = list.add((tile * ldl) as usize);
    let mut run = 0u32;
    let mut base = 0u32;
    while base < nwmax {
        let w = base + t;
        let mut ws = [0u32; 16];
        let mut or = 0u32;
        unroll!(i in 16 => {
            if (i as u32) < nr && w < nwr(i as u32) {
                ws[i] = *bits.add(((r0 + i as u32) * ldb + w) as usize);
                or |= ws[i];
            }
        });
        let win = (or | (or >> 1)) & 0x5555_5555; // bit 2j: window 16w + j used
        let c = win.count_ones();
        let mut inc = c;
        let mut d = 1;
        while d < 32 {
            let o = warp::shfl_up_u32(inc, d);
            if lane() >= d {
                inc += o;
            }
            d *= 2;
        }
        if lane() == 31 {
            wsum.set(warp_id(), inc);
        }
        sync_threads();
        let mut before = 0u32;
        let mut tot = 0u32;
        let mut k = 0;
        while k < 8 {
            let v = wsum.at(k);
            if k < warp_id() {
                before += v;
            }
            tot += v;
            k += 1;
        }
        let mut pos = run + before + inc - c;
        let mut m = win;
        while m != 0 {
            let j = m.trailing_zeros(); // even
            m &= m - 1;
            let mut rm = 0u32;
            unroll!(i in 16 => { rm |= ((ws[i] >> j) & 1) << i; rm |= ((ws[i] >> (j + 1)) & 1) << (16 + i); });
            *out.add((2 * pos) as usize) = w * 16 + j / 2;
            *out.add((2 * pos + 1) as usize) = rm;
            pos += 1;
        }
        run += tot;
        sync_threads();
        base += bdim_x();
    }
    sync_threads();
    if t == 0 {
        if run % 2 == 1 {
            *out.add((2 * run) as usize) = *out.add((2 * run - 2) as usize);
            *out.add((2 * run + 1) as usize) = 0;
            run += 1;
        }
        *cnt.add(tile as usize) = run;
    }
}

/// D = A * B + D, m16n8k8 f16 inputs, f32 accumulators (A: a0 = row g k 2t..2t+1, a1 = row g + 8; B: k 2t..2t+1,
/// col g).
#[inline(always)]
fn mma_f16_1688(d: &mut [f32; 4], a: [u32; 2], b: u32) {
    unsafe {
        asm!("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}}}, {{{b0}}}, {{{d0}, {d1}, {d2}, {d3}}};",
            d0 = inout(reg32) d[0], d1 = inout(reg32) d[1], d2 = inout(reg32) d[2], d3 = inout(reg32) d[3],
            a0 = in(reg32) a[0], a1 = in(reg32) a[1], b0 = in(reg32) b, options(nomem, nostack));
    }
}
#[inline(always)]
fn vote_all(p: bool) -> bool {
    let r: u32;
    unsafe {
        asm!("{{ .reg .pred a, b; setp.ne.u32 a, {x}, 0; vote.sync.all.pred b, a, 0xffffffff; selp.u32 {r}, 1, 0, b; }}",
            x = in(reg32) p as u32, r = out(reg32) r, options(nostack));
    }
    r != 0
}

/// Flash attention over the tile lists. grid = (tiles, nq / FA_HW), block 32 * FA_HW, dynamic smem FA_SMEM.
/// out / q f32 [rows][nq][256]; gate at gate[r * ldg + h * gstride + d]; scale = 1 / sqrt(256).
/// Each 8-cell window is one softmax step and one k8 P . V mma per output tile, so a row's arithmetic depends only
/// on its own cells (windows it does not use are exact no-ops): the output bits do not depend on which rows share
/// its tile, the group split, or the selection order.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_fa_tc(out: *mut f32, q: *const f32, kq: *const i8, ks: *const f32,
    vq: *const i8, vs: *const f32, gate: *const f32, ldg: u32, gstride: u32, nq: u32, nkv: u32, prm: *const u32,
    scale: f32, list: *const u32, cnt: *const u32, ldl: u32, rows: u32) {
    let tile = bid_x();
    let r0 = tile * QR;
    let gpk = nq / nkv / FA_HW;
    let kh = bid_y() / gpk;
    let h = kh * (nq / nkv) + (bid_y() % gpk) * FA_HW + warp_id();
    let l = lane();
    let g = l / 4;
    let t = l % 4;
    let tid = tid_x();
    let sb = dyn_base::<u8>().addr();
    let lst = list.add((tile * ldl) as usize);
    let nkt = *cnt.add(tile as usize) / 2; // stages of 2 windows
    let p0 = *prm + r0;

    // async raw copy of stage i (windows 2i, 2i + 1 = 16 cells) into buffer st
    let issue = |i: u32, st: u32| {
        if i < nkt {
            let so = sb + st * RAW;
            unroll!(c in 4 => {
                let idx = tid + c as u32 * 128;
                let kv = idx / 256;
                let rem = idx % 256;
                let key = rem / 16;
                let ch = rem % 16;
                let cell = *lst.add((2 * (2 * i + key / 8)) as usize) * 8 + key % 8;
                let src = if kv == 0 { kq } else { vq };
                cp_async_16(so + kv * RAW_V + key * D + ch * 16, src.add(((cell * nkv + kh) * D + ch * 16) as usize) as *const u8);
            });
            if tid < 64 {
                let kv = tid / 32;
                let key = (tid % 32) / 2;
                let hf = tid % 2;
                let cell = *lst.add((2 * (2 * i + key / 8)) as usize) * 8 + key % 8;
                let src = if kv == 0 { ks } else { vs };
                cp_async_16(so + RAW_KS + kv * KT * 32 + key * 32 + hf * 16,
                    (src.add(((cell * nkv + kh) * 8) as usize) as *const u8).add((hf * 16) as usize));
            }
        }
        cp_async_commit();
    };
    issue(0, 0);
    issue(1, 1);

    let mut qa = [[0u32; 4]; 16];
    let ra = r0 + g;
    let rb = r0 + g + 8;
    unroll!(k in 16 => {
        let d0 = k as u32 * 16 + 2 * t;
        if ra < rows {
            let p = q.add(((ra * nq + h) * D + d0) as usize);
            qa[k][0] = pack_h2(*p, *p.add(1));
            qa[k][2] = pack_h2(*p.add(8), *p.add(9));
        }
        if rb < rows {
            let p = q.add(((rb * nq + h) * D + d0) as usize);
            qa[k][1] = pack_h2(*p, *p.add(1));
            qa[k][3] = pack_h2(*p.add(8), *p.add(9));
        }
    });
    let c2 = scale * math::LOG2E;
    let mut o = [[0.0f32; 4]; 32];
    let mut mx = [f32::NEG_INFINITY; 2];
    let mut ls = [0.0f32; 2];
    let pr = [p0 + g, p0 + g + 8];
    // ldmatrix lane offsets (bytes) within one window (8 keys): K x4 = keys 0..8 x dims +0, +8, +16, +24;
    // V x4.trans = keys 0..8 x dims +0, +8, +16, +24
    let k_off = ((l % 8) * LDH + (l / 8) * 8) * 2;
    let v_off = k_off;

    let mut i = 0u32;
    while i < nkt {
        let st = i % 2;
        cp_async_wait::<1>();
        sync_threads();
        {
            let key = tid / 8;
            let c = tid % 8;
            unroll!(kv in 2 => {
                let ra_ = sb + st * RAW + kv as u32 * RAW_V + key * D + c * 32;
                let w0 = lds_v4(ra_);
                let w1 = lds_v4(ra_ + 16);
                let s = lds_f32(sb + st * RAW + RAW_KS + kv as u32 * KT * 32 + key * 32 + c * 4);
                let ww = [w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3]];
                let mut hv = [0u32; 16];
                unroll!(j in 8 => {
                    hv[2 * j] = pack_h2(sb2f(ww[j], 0) * s, sb2f(ww[j], 1) * s);
                    hv[2 * j + 1] = pack_h2(sb2f(ww[j], 2) * s, sb2f(ww[j], 3) * s);
                });
                let da = sb + H_K + kv as u32 * (KT * LDH * 2) + (key * LDH + c * 32) * 2;
                unroll!(j in 4 => {
                    sts_v4(da + j as u32 * 16, [hv[4 * j], hv[4 * j + 1], hv[4 * j + 2], hv[4 * j + 3]]);
                });
            });
        }
        sync_threads();
        issue(i + 2, st);

        unroll!(wi in 2 => {
            let win = *lst.add((2 * (2 * i + wi as u32)) as usize);
            let msk = *lst.add((2 * (2 * i + wi as u32) + 1) as usize) >> ((t / 2) * 16);
            let kb = sb + H_K + wi as u32 * 8 * LDH * 2 + k_off;
            let vb = sb + H_V + wi as u32 * 8 * LDH * 2 + v_off;
            // S = Q K^T, 16 rows x 8 keys
            let mut sc = [0.0f32; 4];
            unroll!(k2 in 8 => {
                let b = ldmatrix_x4(kb + k2 as u32 * 64);
                mma_f16_16816(&mut sc, qa[2 * k2], [b[0], b[1]]);
                mma_f16_16816(&mut sc, qa[2 * k2 + 1], [b[2], b[3]]);
            });
            let cb = win * 8 + 2 * t;
            let mut alpha = [1.0f32; 2];
            unroll!(hr in 2 => {
                let row = g + 8 * hr as u32;
                let use_ = (msk >> row) & 1 != 0;
                let mut tm = f32::NEG_INFINITY;
                unroll!(c in 2 => {
                    let ok = use_ && cb + c as u32 <= pr[hr];
                    let v = if ok { sc[2 * hr + c] * c2 } else { f32::NEG_INFINITY };
                    sc[2 * hr + c] = v;
                    tm = math::maxf(tm, v);
                });
                tm = math::maxf(tm, warp::shfl_xor(tm, 1));
                tm = math::maxf(tm, warp::shfl_xor(tm, 2));
                let mn = math::maxf(mx[hr], tm);
                if mn == f32::NEG_INFINITY {
                    sc[2 * hr] = 0.0;
                    sc[2 * hr + 1] = 0.0;
                } else {
                    if mn != mx[hr] {
                        alpha[hr] = math::exp2(mx[hr] - mn);
                    }
                    sc[2 * hr] = math::exp2(sc[2 * hr] - mn);
                    sc[2 * hr + 1] = math::exp2(sc[2 * hr + 1] - mn);
                    mx[hr] = mn;
                }
            });
            let pa = [pack_h2(sc[0], sc[1]), pack_h2(sc[2], sc[3])];
            ls[0] = ls[0] * alpha[0] + h2_sum(pa[0]);
            ls[1] = ls[1] * alpha[1] + h2_sum(pa[1]);
            if !vote_all(alpha[0] == 1.0 && alpha[1] == 1.0) {
                unroll!(n in 32 => {
                    o[n][0] *= alpha[0];
                    o[n][1] *= alpha[0];
                    o[n][2] *= alpha[1];
                    o[n][3] *= alpha[1];
                });
            }
            // O += P V (k8 = this window)
            unroll!(dt in 8 => {
                let b = ldmatrix_x4_trans(vb + dt as u32 * 64);
                mma_f16_1688(&mut o[4 * dt], pa, b[0]);
                mma_f16_1688(&mut o[4 * dt + 1], pa, b[1]);
                mma_f16_1688(&mut o[4 * dt + 2], pa, b[2]);
                mma_f16_1688(&mut o[4 * dt + 3], pa, b[3]);
            });
        });
        i += 1;
    }
    cp_async_wait_all();
    unroll!(hr in 2 => {
        let mut lt = ls[hr] + warp::shfl_xor(ls[hr], 1);
        lt = lt + warp::shfl_xor(lt, 2);
        let row = r0 + g + 8 * hr as u32;
        if row < rows && lt > 0.0 {
            let inv = 1.0 / lt;
            let gp = gate.add((row * ldg + h * gstride + 2 * t) as usize);
            let op = out.add(((row * nq + h) * D + 2 * t) as usize);
            unroll!(n in 32 => {
                let g0 = *gp.add(n * 8);
                let g1 = *gp.add(n * 8 + 1);
                *op.add(n * 8) = o[n][2 * hr] * inv * (1.0 / (1.0 + math::exp(-g0)));
                *op.add(n * 8 + 1) = o[n][2 * hr + 1] * inv * (1.0 / (1.0 + math::exp(-g1)));
            });
        }
    });
}

#[inline(always)]
unsafe fn ld8q(q: *const i8, s: f32) -> [f32; 8] {
    let w = *(q as *const [u32; 2]);
    let mut o = [0.0f32; 8];
    unroll!(e in 4 => {
        o[e] = ((w[0] >> (8 * e)) as u8 as i8) as f32 * s;
        o[4 + e] = ((w[1] >> (8 * e)) as u8 as i8) as f32 * s;
    });
    o
}

/// Copy of fe-fnext's SIMT `fx_attn` (k-fnext fast.rs) for comparisons and timing only.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_fa_ref_simt(out: *mut f32, q: *const f32, kq: *const i8, ks: *const f32,
    vq: *const i8, vs: *const f32, gate: *const f32, ldg: u32, gstride: u32, nq: u32, nkv: u32, prm: *const u32,
    scale: f32, sel: *const u32, seln: *const u32, ldsel: u32) {
    let sm: Shared<f32> = kcommon::static_shared!(far_m, f32, 64);
    let sl: Shared<f32> = kcommon::static_shared!(far_l, f32, 64);
    let sacc: Shared<f32> = kcommon::static_shared!(far_acc, f32, 8192);
    let r = bid_x();
    let h = bid_y();
    let kh = h / (nq / nkv);
    let w = warp_id();
    let l = lane();
    let qp = q.add(((r * nq + h) * 256 + l * 8) as usize);
    let mut qv = [0.0f32; 8];
    unroll!(e in 8 => { qv[e] = *qp.add(e); });
    let ns = *seln.add(r as usize);
    let dense = ns == 0;
    let n = if dense { *prm + r + 1 } else { ns };
    let mut m = f32::NEG_INFINITY;
    let mut ls = 0.0f32;
    let mut acc = [0.0f32; 8];
    let mut pi = w;
    while pi < n {
        let p = if dense { pi } else { *sel.add((r * ldsel + pi) as usize) };
        let cb = (p * nkv + kh) as usize;
        let kv = ld8q(kq.add(cb * 256 + (l * 8) as usize), *ks.add(cb * 8 + (l / 4) as usize));
        let mut d = 0.0f32;
        unroll!(e in 8 => { d = math::fma(qv[e], kv[e], d); });
        let sc = warp::sum(d) * scale;
        let mn = math::maxf(m, sc);
        let a = math::exp(m - mn);
        let b = math::exp(sc - mn);
        let vv = ld8q(vq.add(cb * 256 + (l * 8) as usize), *vs.add(cb * 8 + (l / 4) as usize));
        unroll!(e in 8 => { acc[e] = acc[e] * a + b * vv[e]; });
        ls = ls * a + b;
        m = mn;
        pi += 8;
    }
    if l == 0 {
        sm.set(w, m);
        sl.set(w, ls);
    }
    unroll!(e in 8 => { sacc.set(w * 256 + l * 8 + e as u32, acc[e]); });
    sync_threads();
    let t = tid_x();
    let mut mm = f32::NEG_INFINITY;
    unroll!(k in 8 => { mm = math::maxf(mm, sm.at(k as u32)); });
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    unroll!(k in 8 => {
        let mk = sm.at(k as u32);
        if mk != f32::NEG_INFINITY {
            let f = math::exp(mk - mm);
            num += sacc.at(k as u32 * 256 + t) * f;
            den += sl.at(k as u32) * f;
        }
    });
    let gv = *gate.add((r * ldg + h * gstride + t) as usize);
    *out.add(((r * nq + h) * 256 + t) as usize) = num / den * (1.0 / (1.0 + math::exp(-gv)));
}

/// Tensor-pipe rate probe (bench only): mode 0 = f16 m16n8k16, 1 = f16 m16n8k8, 2 = s8 m16n8k32; `n` iterations of
/// 8 independent accumulator chains per warp. out[0] = a checksum.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_fa_mmaprobe(out: *mut f32, mode: u32, n: u32, seed: u32) {
    let a = [seed, seed ^ 1, seed ^ 2, seed ^ 3];
    let b = [seed ^ 5, seed ^ 7];
    let mut acc = [[0.0f32; 4]; 8];
    let mut iacc = [[0i32; 4]; 8];
    let mut i = 0;
    if mode == 0 {
        while i < n {
            unroll!(c in 8 => { mma_f16_16816(&mut acc[c], a, b); });
            i += 1;
        }
    } else if mode == 1 {
        while i < n {
            unroll!(c in 8 => { mma_f16_1688(&mut acc[c], [a[0], a[1]], b[0]); });
            i += 1;
        }
    } else if mode == 2 {
        while i < n {
            unroll!(c in 8 => { crate::mmq::mma_s8_16832(&mut iacc[c], a, b); });
            i += 1;
        }
    } else {
        // MMQ inner step: fresh s8 k32 mma per 32 values + per-element f32 scaling (d8 * wb * i2f(isum)) into acc
        let d8 = f32::from_bits(seed | 0x3f00_0000);
        let wb = f32::from_bits(seed ^ 0x3e00_0000);
        let mut a2 = a;
        while i < n {
            unroll!(c in 8 => {
                let mut is = [0i32; 4];
                if mode >= 5 {
                    let mut lo = [0i32; 4];
                    let mut hi = [0i32; 4];
                    crate::mmq::mma_s8_16816(&mut lo, a2[0], a2[1], b[0]);
                    crate::mmq::mma_s8_16816(&mut hi, a2[2], a2[3], b[1]);
                    unroll!(q in 4 => { is[q] = lo[q] * (seed as i32 & 7) + hi[q] * (seed as i32 & 3); });
                } else {
                    crate::mmq::mma_s8_16832(&mut is, a2, b);
                }
                unroll!(q in 4 => {
                    let v = f32::from_bits((is[q] + 0x4b40_0000) as u32) - 12582912.0;
                    if mode == 3 {
                        acc[c][q] += d8 * wb * v;
                    } else {
                        acc[c][q] = kcommon::math::fma(d8 * wb, v, acc[c][q]);
                    }
                });
            });
            a2[0] = a2[0].wrapping_add(1);
            i += 1;
        }
    }
    let mut s = 0.0f32;
    unroll!(c in 8 => { s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3] + iacc[c][0] as f32; });
    if s == 1.2345 {
        *out = s;
    }
}

// ---- v2: exact-ish numerics. Q . K in int8 (Q as a 16-bit fixed-point hi / lo pair per row and head, K the
// cache's exact int8 codes, per-32 K scales applied in f32), P . V in f16 with the V scales folded into P per
// 32-dim block and P split hi / lo into the two k8 halves of one m16n8k16 (V codes exact in f16).
const RK_LD: u32 = D + 16; // int8 K row stride (272 B: conflict-free ldmatrix)
const R2_V: u32 = KT * RK_LD;
const R2_KS: u32 = R2_V + KT * D;
const R2_VS: u32 = R2_KS + KT * 32;
const R2: u32 = R2_VS + KT * 32;
const H2_V: u32 = 2 * R2;
pub const FA2_SMEM: u32 = H2_V + KT * LDH * 2;
/// Q fixed point: |q16| <= Q16MAX = 127 * 256 + 127 (hi in [-127, 127], lo in [-128, 127]).
const Q16MAX: f32 = 32639.0;
/// P * vscale is split in f16 after scaling by 2^8 (exact), so the lo half stays out of the f16 subnormals.
const PSC: f32 = 256.0;

#[inline(always)]
fn pack_hilo(x0: f32, x1: f32) -> (u32, u32) {
    // (pack(hi(x0), hi(x1)), pack(lo(x0), lo(x1)))
    let (hi, lo): (u32, u32);
    unsafe {
        asm!("{{ .reg .b32 h; .reg .f32 a0, a1; cvt.rn.f16x2.f32 h, {x1}, {x0}; mov.b32 {hi}, h; \
              {{ .reg .f16 p, q; mov.b32 {{p, q}}, h; cvt.f32.f16 a0, p; cvt.f32.f16 a1, q; }} \
              sub.f32 a0, {x0}, a0; sub.f32 a1, {x1}, a1; cvt.rn.f16x2.f32 {lo}, a1, a0; }}",
            x0 = in(reg32) x0, x1 = in(reg32) x1, hi = out(reg32) hi, lo = out(reg32) lo, options(pure, nomem, nostack));
    }
    (hi, lo)
}
/// Two signed bytes (0, 1 or 2, 3 of w) as exact f16 integers.
#[inline(always)]
fn s8x2_h2(w: u32, hi_pair: bool) -> u32 {
    let x = w ^ 0x8080_8080;
    let sel = if hi_pair { 0x4342u32 } else { 0x4140u32 };
    let r: u32;
    unsafe {
        asm!("{{ .reg .b32 t, c; mov.b32 c, 0x64806480; prmt.b32 t, {x}, 0x64646464, {s}; sub.f16x2 {r}, t, c; }}", x = in(reg32) x,
            s = in(reg32) sel, r = out(reg32) r, options(pure, nomem, nostack));
    }
    r
}
#[inline(always)]
fn i2f22(v: i32) -> f32 {
    f32::from_bits((v + 0x4b40_0000) as u32) - 12582912.0
}
#[inline(always)]
fn f2i_rn(x: f32) -> i32 {
    let r: i32;
    unsafe { asm!("cvt.rni.s32.f32 {r}, {x};", r = out(reg32) r, x = in(reg32) x, options(pure, nomem, nostack)) }
    r
}

/// Same contract and invariances as `ggq_fa_tc`; dynamic smem FA2_SMEM.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_fa_tc2(out: *mut f32, q: *const f32, kq: *const i8, ks: *const f32,
    vq: *const i8, vs: *const f32, gate: *const f32, ldg: u32, gstride: u32, nq: u32, nkv: u32, prm: *const u32,
    scale: f32, list: *const u32, cnt: *const u32, ldl: u32, rows: u32) {
    let tile = bid_x();
    let r0 = tile * QR;
    let gpk = nq / nkv / FA_HW;
    let kh = bid_y() / gpk;
    let h = kh * (nq / nkv) + (bid_y() % gpk) * FA_HW + warp_id();
    let l = lane();
    let g = l / 4;
    let t = l % 4;
    let tid = tid_x();
    let sb = dyn_base::<u8>().addr();
    let lst = list.add((tile * ldl) as usize);
    let nkt = *cnt.add(tile as usize) / 2;
    let p0 = *prm + r0;

    let issue = |i: u32, st: u32| {
        if i < nkt {
            let so = sb + st * R2;
            unroll!(c in 4 => {
                let idx = tid + c as u32 * 128;
                let kv = idx / 256;
                let rem = idx % 256;
                let key = rem / 16;
                let ch = rem % 16;
                let cell = *lst.add((2 * (2 * i + key / 8)) as usize) * 8 + key % 8;
                if kv == 0 {
                    cp_async_16(so + key * RK_LD + ch * 16, kq.add(((cell * nkv + kh) * D + ch * 16) as usize) as *const u8);
                } else {
                    cp_async_16(so + R2_V + key * D + ch * 16, vq.add(((cell * nkv + kh) * D + ch * 16) as usize) as *const u8);
                }
            });
            if tid < 64 {
                let kv = tid / 32;
                let key = (tid % 32) / 2;
                let hf = tid % 2;
                let cell = *lst.add((2 * (2 * i + key / 8)) as usize) * 8 + key % 8;
                let src = if kv == 0 { ks } else { vs };
                cp_async_16(so + R2_KS + kv * KT * 32 + key * 32 + hf * 16,
                    (src.add(((cell * nkv + kh) * 8) as usize) as *const u8).add((hf * 16) as usize));
            }
        }
        cp_async_commit();
    };
    issue(0, 0);
    issue(1, 1);

    // Q -> 16-bit fixed point per (row, head): q = dq * (256 * hi + lo)
    let mut qh = [[0u32; 4]; 8];
    let mut ql = [[0u32; 4]; 8];
    let mut dq = [0.0f32; 2];
    {
        let rr = [r0 + g, r0 + g + 8];
        let mut qv = [[[0.0f32; 8]; 8]; 2];
        let mut am = [0.0f32; 2];
        unroll!(hr in 2 => {
            if rr[hr] < rows {
                let p = q.add(((rr[hr] * nq + h) * D + 4 * t) as usize);
                unroll!(b in 8 => {
                    let v0 = *(p.add(b * 32) as *const [f32; 4]);
                    let v1 = *(p.add(b * 32 + 16) as *const [f32; 4]);
                    unroll!(e in 4 => {
                        qv[hr][b][e] = v0[e];
                        qv[hr][b][4 + e] = v1[e];
                        am[hr] = math::maxf(am[hr], math::maxf(v0[e], -v0[e]));
                        am[hr] = math::maxf(am[hr], math::maxf(v1[e], -v1[e]));
                    });
                });
            }
            am[hr] = math::maxf(am[hr], warp::shfl_xor(am[hr], 1));
            am[hr] = math::maxf(am[hr], warp::shfl_xor(am[hr], 2));
            dq[hr] = am[hr] / Q16MAX;
            let inv = if am[hr] > 0.0 { Q16MAX / am[hr] } else { 0.0 };
            unroll!(b in 8 => {
                let mut wh = [0u32; 2];
                let mut wl = [0u32; 2];
                unroll!(e in 8 => {
                    let q16 = f2i_rn(qv[hr][b][e] * inv);
                    let hi = (q16 + 128) >> 8;
                    let lo = q16 - (hi << 8);
                    wh[e / 4] |= ((hi as u32) & 255) << (8 * (e % 4));
                    wl[e / 4] |= ((lo as u32) & 255) << (8 * (e % 4));
                });
                // a0 / a2 = row g (k 4t.., 16 + 4t..), a1 / a3 = row g + 8
                qh[b][hr] = wh[0];
                qh[b][2 + hr] = wh[1];
                ql[b][hr] = wl[0];
                ql[b][2 + hr] = wl[1];
            });
        });
    }
    let c2 = scale * math::LOG2E;
    let sq = [dq[0] * c2, dq[1] * c2];
    let mut o = [[0.0f32; 4]; 32];
    let mut mx = [f32::NEG_INFINITY; 2];
    let mut ls = [0.0f32; 2];
    let pr = [p0 + g, p0 + g + 8];
    let k_off = (l % 8) * RK_LD + (l / 8) * 16;
    let v_off = ((l % 8) * LDH + (l / 8) * 8) * 2;

    let mut i = 0u32;
    while i < nkt {
        let st = i % 2;
        let so = sb + st * R2;
        cp_async_wait::<1>();
        sync_threads();
        {
            // V codes -> exact f16 integers
            let key = tid / 8;
            let c = tid % 8;
            let ra_ = so + R2_V + key * D + c * 32;
            let w0 = lds_v4(ra_);
            let w1 = lds_v4(ra_ + 16);
            let ww = [w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3]];
            let da = sb + H2_V + (key * LDH + c * 32) * 2;
            unroll!(j in 4 => {
                sts_v4(da + j as u32 * 16, [s8x2_h2(ww[2 * j], false), s8x2_h2(ww[2 * j], true),
                    s8x2_h2(ww[2 * j + 1], false), s8x2_h2(ww[2 * j + 1], true)]);
            });
        }
        sync_threads();

        unroll!(wi in 2 => {
            let win = *lst.add((2 * (2 * i + wi as u32)) as usize);
            let msk = *lst.add((2 * (2 * i + wi as u32) + 1) as usize) >> ((t / 2) * 16);
            let kb = so + wi as u32 * 8 * RK_LD + k_off;
            let key0 = wi as u32 * 8 + 2 * t;
            let ksa = so + R2_KS + key0 * 32;
            let vsa = so + R2_VS + key0 * 32;
            let dk = [lds_v4(ksa), lds_v4(ksa + 16), lds_v4(ksa + 32), lds_v4(ksa + 48)];
            let mut sc = [0.0f32; 4];
            unroll!(b2 in 4 => {
                let kf = ldmatrix_x4(kb + b2 as u32 * 64);
                unroll!(bb in 2 => {
                    let b = 2 * b2 + bb;
                    let mut ih = [0i32; 4];
                    let mut il = [0i32; 4];
                    crate::mmq::mma_s8_16832(&mut ih, qh[b], [kf[2 * bb], kf[2 * bb + 1]]);
                    crate::mmq::mma_s8_16832(&mut il, ql[b], [kf[2 * bb], kf[2 * bb + 1]]);
                    // dk: key 2t -> dk[0..2] (blocks 0..8), key 2t + 1 -> dk[2..4]
                    let s0 = f32::from_bits(dk[b / 4][b % 4]);
                    let s1 = f32::from_bits(dk[2 + b / 4][b % 4]);
                    unroll!(e in 4 => {
                        let v = math::fma(i2f22(ih[e]), 256.0, i2f22(il[e]));
                        sc[e] = math::fma(v, if e % 2 == 0 { s0 } else { s1 }, sc[e]);
                    });
                });
            });
            let cb = win * 8 + 2 * t;
            let mut alpha = [1.0f32; 2];
            unroll!(hr in 2 => {
                let row = g + 8 * hr as u32;
                let use_ = (msk >> row) & 1 != 0;
                let mut tm = f32::NEG_INFINITY;
                unroll!(c in 2 => {
                    let ok = use_ && cb + c as u32 <= pr[hr];
                    let v = if ok { sc[2 * hr + c] * sq[hr] } else { f32::NEG_INFINITY };
                    sc[2 * hr + c] = v;
                    tm = math::maxf(tm, v);
                });
                tm = math::maxf(tm, warp::shfl_xor(tm, 1));
                tm = math::maxf(tm, warp::shfl_xor(tm, 2));
                let mn = math::maxf(mx[hr], tm);
                if mn == f32::NEG_INFINITY {
                    sc[2 * hr] = 0.0;
                    sc[2 * hr + 1] = 0.0;
                } else {
                    if mn != mx[hr] {
                        alpha[hr] = math::exp2(mx[hr] - mn);
                    }
                    sc[2 * hr] = math::exp2(sc[2 * hr] - mn);
                    sc[2 * hr + 1] = math::exp2(sc[2 * hr + 1] - mn);
                    mx[hr] = mn;
                }
            });
            ls[0] = ls[0] * alpha[0] + (sc[0] + sc[1]);
            ls[1] = ls[1] * alpha[1] + (sc[2] + sc[3]);
            if !vote_all(alpha[0] == 1.0 && alpha[1] == 1.0) {
                unroll!(n in 32 => {
                    o[n][0] *= alpha[0];
                    o[n][1] *= alpha[0];
                    o[n][2] *= alpha[1];
                    o[n][3] *= alpha[1];
                });
            }
            let dv = [lds_v4(vsa), lds_v4(vsa + 16), lds_v4(vsa + 32), lds_v4(vsa + 48)];
            let vb = sb + H2_V + wi as u32 * 8 * LDH * 2 + v_off;
            unroll!(b in 8 => {
                let s0 = f32::from_bits(dv[b / 4][b % 4]) * PSC;
                let s1 = f32::from_bits(dv[2 + b / 4][b % 4]) * PSC;
                let (h0, l0) = pack_hilo(sc[0] * s0, sc[1] * s1);
                let (h1, l1) = pack_hilo(sc[2] * s0, sc[3] * s1);
                let a = [h0, h1, l0, l1];
                let vf = ldmatrix_x4_trans(vb + b as u32 * 64);
                unroll!(j in 4 => { mma_f16_16816(&mut o[4 * b + j], a, [vf[j], vf[j]]); });
            });
        });
        sync_threads();
        issue(i + 2, st);
        i += 1;
    }
    cp_async_wait_all();
    unroll!(hr in 2 => {
        let mut lt = ls[hr] + warp::shfl_xor(ls[hr], 1);
        lt = lt + warp::shfl_xor(lt, 2);
        let row = r0 + g + 8 * hr as u32;
        if row < rows && lt > 0.0 {
            let inv = (1.0 / lt) * (1.0 / PSC);
            let gp = gate.add((row * ldg + h * gstride + 2 * t) as usize);
            let op = out.add(((row * nq + h) * D + 2 * t) as usize);
            unroll!(n in 32 => {
                let g0 = *gp.add(n * 8);
                let g1 = *gp.add(n * 8 + 1);
                *op.add(n * 8) = o[n][2 * hr] * inv * (1.0 / (1.0 + math::exp(-g0)));
                *op.add(n * 8 + 1) = o[n][2 * hr + 1] * inv * (1.0 / (1.0 + math::exp(-g1)));
            });
        }
    });
}
