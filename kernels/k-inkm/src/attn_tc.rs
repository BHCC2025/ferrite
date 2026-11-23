//! Tensor-core attention for prompt rows (both layer types): one block a (16-query tile, head), 4 warps. Same math as `ink_attn`
//! (scores rounded to bf16 at the reference's points, relative bias, sink, probabilities rounded to bf16 after normalisation) with
//! the products on mma.m16n8k16: Q.K^T with the query tile in registers, the bias from a per-block table, p.V with p taken straight from
//! the score accumulators. Key chunks of 64 at absolute positions; the 4 warps split each chunk in 16-key slices; their statistics and
//! their partial outputs are combined in warp order, so a row's bits depend only on its own key range, never on the other rows of the tile
//! or on the chunking of the prompt.
use crate::*;
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

pub(crate) const PITCH: u32 = 272; // 128 bf16 + 16 B pad
pub(crate) const QS_OFF: u32 = 0;
pub(crate) const KS_OFF: u32 = QS_OFF + 16 * PITCH;
pub(crate) const VS_OFF: u32 = KS_OFF + 64 * PITCH;
pub(crate) const RL_OFF: u32 = VS_OFF + 64 * PITCH;
pub(crate) const RS_OFF: u32 = RL_OFF + 16 * 144 * 2;
pub(crate) const ST_OFF: u32 = RS_OFF + 16 * 16 * 4;
pub const ATC_SMEM: u32 = ST_OFF + 4 * 16 * 2 * 4 + 16 * 4 * 2;

/// 6-bit loader: the 64 keys of a chunk from the packed cache ([pos][8][104] B) of kv head `kvh`, dequantised (still in the H32-rotated basis:
/// Q is rotated instead, and the output is rotated back once) and rounded to bf16 into the smem tile. One (key row, group) job a thread, two jobs each.
#[inline(always)]
pub(crate) unsafe fn load_chunk_q(dst: Shared<u8>, src: *const u8, kvh: u32, kbase: u32, ring: u32) {
    let t = tid_x();
    let mut jid = t;
    while jid < 256 {
        let r = jid >> 2;
        let g = jid & 3;
        let row = (kbase + r) % ring;
        let rec = src.add(((((row >> 6) * 8 + kvh) * 64 + (row & 63)) * 104) as usize);
        let a0 = *(rec.add(24 * g as usize) as *const u64);
        let a1 = *(rec.add(24 * g as usize + 8) as *const u64);
        let a2 = *(rec.add(24 * g as usize + 16) as *const u64);
        let sc = bf(*(rec.add(96 + 2 * g as usize) as *const u16)) * (1.0 / 64.0);
        let mut v = [0.0f32; 32];
        let words = [a0, a1, a2, 0u64];
        kcommon::unroll!(k in 32 => {
            const BP: usize = 6 * k;
            const W: usize = BP >> 6;
            const OFF: u32 = (BP & 63) as u32;
            let mut x = words[W] >> OFF;
            if OFF > 58 {
                x |= words[W + 1] << (64 - OFF);
            }
            v[k] = (2 * ((x & 63) as i32) - 63) as f32 * sc;
        });
        let o = dst.byte_add(r * PITCH + g * 64);
        let mut j = 0;
        while j < 8 {
            let w4 = [bf16::pack2(v[4 * j], v[4 * j + 1]), bf16::pack2(v[4 * j + 2], v[4 * j + 3])];
            <[u32; 2] as kcommon::smem::SmemIo>::sts(o.byte_add(8 * j as u32).addr(), w4);
            j += 1;
        }
        jid += 128;
    }
}

#[inline(always)]
pub(crate) unsafe fn load_chunk(dst: Shared<u8>, src: *const u16, kbase: u32, ring: u32) {
    let t = tid_x();
    let mut i = t;
    while i < 64 * 16 {
        let r = i >> 4;
        let c = i & 15;
        let row = (kbase + r) % ring;
        cp_async_16(dst.byte_add(r * PITCH + c * 16).addr(), src.add(row as usize * 1024 + c as usize * 8) as *const u8);
        i += 128;
    }
}

/// Split-key form: grid = (nsup, 24, q tiles), block 128, smem ATC_SMEM. A super-tile = 2 chunks of 64 keys at absolute positions.
/// phase 0: per super-tile row statistics (max, sum) into `stat` [qtile][24][nsup][16][2]; phase 1: with the statistics of all
/// super-tiles combined in order (plus the sink) the probabilities and the super-tile's partial P.V into `part`
/// [qtile][24][nsup][16][128]; `ink_attn_tc_comb` adds the partials in super-tile order. Every row's bits depend only on its own keys.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_attn_tc(stat: *mut f32, gst: *const f32, part: *mut f32, q: *const u16, kc: *const u16, vc: *const u16,
    rsx: *const u16, proj: *const u16, sink: *const u16, sld: u32, pos: *const u32, window: u32, scaling: f32, ring: u32, ntok: u32,
    phase: u32, qt_off: u32, gqa: u32, qmode: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let smem: Shared<u8> = dyn_base();
    let nsup = gdim_x();
    let j = bid_x();
    let h = bid_y();
    let qi = bid_z();
    let qbase = (qt_off + qi) * 16;
    let gq = gqa != 0;
    // gqa (one decode token): block y = kv head; rows 0..2 = its three query heads, slots of the standard layout (row 0 of head 3g + r)
    let kvh = if gq { h } else { h / 3 };
    let wid = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    let pos0 = *pos;
    let t = tid_x();
    // super-tile range of this block (early exit for inactive ones)
    let (cs_, ce_) = {
        let lo_of = |p: u32| if window > 0 && p + 1 > window { p + 1 - window } else { 0 };
        let last_row = if qbase + 16 <= ntok { 15 } else { ntok - qbase - 1 };
        let tile_lo = lo_of(pos0 + qbase);
        let tile_hi = pos0 + qbase + last_row;
        let sup0 = tile_lo / 128;
        (core::cmp::max((sup0 + j) * 2, tile_lo / 64), core::cmp::min((sup0 + j) * 2 + 1, tile_hi / 64))
    };
    if cs_ > ce_ {
        let idx = (qi * 24 + h) * nsup + j;
        if gq {
            if phase == 0 {
                if t < 6 {
                    let hh = 3 * h + t / 2;
                    *stat.add(((hh * nsup + j) * 32 + (t & 1)) as usize) = if t & 1 == 0 { -1.0e30 } else { 0.0 };
                }
            } else if t < 128 {
                let mut r = 0;
                while r < 3 {
                    *part.add((((3 * h + r) * nsup + j) * 128 + t) as usize) = 0.0;
                    r += 1;
                }
            }
            return;
        }
        if phase == 0 {
            if t < 32 {
                *stat.add((idx * 32 + t) as usize) = if t & 1 == 0 { -1.0e30 } else { 0.0 };
            }
        } else {
            let mut i = t;
            while i < 2048 {
                *part.add((idx * 2048 + i) as usize) = 0.0;
                i += 128;
            }
        }
        return;
    }
    // ---- stage Q tile, relative states ----
    let mut i = t;
    while i < 16 * 16 {
        let r = i >> 4;
        let c = i & 15;
        let row = qbase + r;
        let d = smem.byte_add(QS_OFF + r * PITCH + c * 16).addr();
        if gq && r < 3 {
            cp_async_16(d, q.add((3 * h + r) as usize * 128 + c as usize * 8) as *const u8);
        } else if !gq && row < ntok {
            cp_async_16(d, q.add(row as usize * 3072 + h as usize * 128 + c as usize * 8) as *const u8);
        } else {
            cp_async_16_zfill(d, q as *const u8, 0);
        }
        i += 128;
    }
    cp_async_commit();
    let rs: Shared<f32> = smem.byte_add(RS_OFF).cast();
    let mut i = t;
    while i < 256 {
        let r = i >> 4;
        let row = qbase + r;
        rs.set(i, if gq { if r < 3 { bf(*rsx.add((3 * h + r) as usize * 16 + (i & 15) as usize)) } else { 0.0 } }
            else if row < ntok { bf(*rsx.add(row as usize * 384 + h as usize * 16 + (i & 15) as usize)) } else { 0.0 });
        i += 128;
    }
    cp_async_wait_all();
    sync_threads();
    if qmode != 0 && t < 64 {
        // 6-bit K/V: attention runs in the rotated basis; rotate the query rows (H32 / sqrt(32) per group of 32 dims), rounded to bf16
        let (r, g) = (t >> 2, t & 3);
        let qp = smem.byte_add(QS_OFF + r * PITCH + g * 64);
        let mut v = [0.0f32; 32];
        kcommon::unroll!(k in 16 => {
            let w = <u32 as kcommon::smem::SmemIo>::lds(qp.byte_add(4 * k as u32).addr());
            v[2 * k] = bf16::lo(w);
            v[2 * k + 1] = bf16::hi(w);
        });
        kcommon::unroll!(st in [1, 2, 4, 8, 16] => {
            kcommon::unroll!(i in 32 => {
                if i & st == 0 {
                    let a = v[i];
                    let b = v[i + st];
                    v[i] = a + b;
                    v[i + st] = a - b;
                }
            });
        });
        kcommon::unroll!(k in 16 => {
            <u32 as kcommon::smem::SmemIo>::sts(qp.byte_add(4 * k as u32).addr(), bf16::pack2(v[2 * k] * 0.176_776_7, v[2 * k + 1] * 0.176_776_7));
        });
    }
    sync_threads();
    // ---- bias table RL[r][d] = bf16(sum_j rs[r][j] * proj[j][d]), d < 1024 ----
    let rl: Shared<u16> = smem.byte_add(RL_OFF).cast();
    // only the distances this block's keys can have (older super-tiles of a global layer need none: bias 0 beyond 1023)
    let (dmin, dmax) = {
        let nv = if ntok - qbase < 16 { ntok - qbase } else { 16 };
        let (pmin, pmax) = (pos0 + qbase, pos0 + qbase + nv - 1);
        let (klo, khi) = (cs_ * 64, ce_ * 64 + 63);
        let dmin = if pmin > khi { pmin - khi } else { 0 };
        let dmx = if pmax >= klo { pmax - klo } else { 0 };
        (dmin, if dmx > 1023 { 1023 } else { dmx })
    };
    let mut d = dmin + t;
    while d <= dmax {
        let mut pj = [0.0f32; 16];
        kcommon::unroll!(j in 16 => {
            pj[j] = bf(*proj.add(j * 1024 + d as usize));
        });
        let nvalid = if gq { 3 } else if ntok - qbase < 16 { ntok - qbase } else { 16 };
        let mut r = 0;
        while r < nvalid {
            let mut b = 0.0f32;
            kcommon::unroll!(j in 16 => {
                b = math::fma(rs.at(r * 16 + j as u32), pj[j], b);
            });
            rl.set(r * 144 + (d - dmin), tb(b));
            r += 1;
        }
        d += 128;
    }
    // ---- A fragments of Q (constant over the key chunks) ----
    let mut qf = [[0u32; 4]; 8];
    kcommon::unroll!(ks in 8 => {
        qf[ks] = ldmatrix_x4(smem.byte_add(QS_OFF + (l & 15) * PITCH + (ks as u32 * 2 + (l >> 4)) * 16).addr());
    });
    // rows of this thread: ra = g, rb_ = g + 8
    let (ra, rb_) = (g, g + 8);
    let (pa, pb) = if gq { (pos0, pos0) } else { (pos0 + qbase + ra, pos0 + qbase + rb_) };
    let va = if gq { ra < 3 } else { qbase + ra < ntok };
    let vb = if gq { false } else { qbase + rb_ < ntok };
    let lo_of = |p: u32| if window > 0 && p + 1 > window { p + 1 - window } else { 0 };
    let (lo_a, lo_b) = (lo_of(pa), lo_of(pb));
    // key range of the tile: first row's lo .. last valid row's position
    let last_row = if qbase + 16 <= ntok { 15 } else { ntok - qbase - 1 };
    let tile_lo = lo_of(pos0 + qbase);
    let tile_hi = pos0 + qbase + last_row;
    let c0 = tile_lo / 64;
    let c1 = tile_hi / 64;
    let sup0 = tile_lo / 128;
    let idx = (qi * 24 + h) * nsup + j;
    let (cs, ce) = (cs_, ce_);
    sync_threads(); // the bias table is complete

    let kcb = kc.add(kvh as usize * 128);
    let vcb = vc.add(kvh as usize * 128);
    let ks_s: Shared<u8> = smem.byte_add(KS_OFF);
    let vs_s: Shared<u8> = smem.byte_add(VS_OFF);

    // one chunk's scores for this warp's 16-key slice: s[nt][0..4] = (ra, c), (ra, c+1), (rb, c), (rb, c+1) after rounding, masked -> NEG
    const NEG: f32 = -3.0e38;
    macro_rules! scores {
        ($kb:expr, $s:ident) => {
            let mut $s = [[NEG; 4]; 2];
            {
                let mut acc = [[0.0f32; 4]; 2];
                kcommon::unroll!(ks in 8 => {
                    kcommon::unroll!(nt in 2 => {
                        let bfr = ldmatrix_x2(ks_s.byte_add((wid * 16 + nt as u32 * 8 + (l & 7)) * PITCH + (ks as u32 * 2 + ((l >> 3) & 1)) * 16).addr());
                        mma_bf16_16816(&mut acc[nt], qf[ks], bfr);
                    });
                });
                kcommon::unroll!(nt in 2 => {
                    kcommon::unroll!(e in 4 => {
                        let (row_p, lo, valid, rr) = if e < 2 { (pa, lo_a, va, ra) } else { (pb, lo_b, vb, rb_) };
                        let key = $kb + wid * 16 + nt as u32 * 8 + 2 * t4 + (e as u32 & 1);
                        if valid && key >= lo && key <= row_p {
                            let dist = row_p - key;
                            let mut sv = rb(rb(acc[nt][e]) * scaling);
                            if dist < 1024 {
                                sv = rb(sv + bf(rl.at(rr * 144 + (dist - dmin))));
                            }
                            $s[nt][e] = sv;
                        }
                    });
                });
            }
        };
    }

    let st: Shared<f32> = smem.byte_add(ST_OFF).cast();
    let (gma, gla, gmb, glb);
    if phase == 0 {
    let (mut ma, mut la, mut mb, mut lb) = (-1.0e30f32, 0.0f32, -1.0e30f32, 0.0f32);
    let mut c = cs;
    while c <= ce {
        if qmode != 0 { load_chunk_q(ks_s, kc as *const u8, kvh, c * 64, ring); } else { load_chunk(ks_s, kcb, c * 64, ring); }
        cp_async_commit();
        cp_async_wait_all();
        sync_threads();
        scores!(c * 64, s);
        kcommon::unroll!(nt in 2 => {
            kcommon::unroll!(e in 4 => {
                let sv = s[nt][e];
                if sv > NEG {
                    if e < 2 {
                        if sv > ma { la = la * ex(ma - sv) + 1.0; ma = sv; } else { la += ex(sv - ma); }
                    } else {
                        if sv > mb { lb = lb * ex(mb - sv) + 1.0; mb = sv; } else { lb += ex(sv - mb); }
                    }
                }
            });
        });
        sync_threads();
        c += 1;
    }
    // quad combine (fixed order t4 = 0..3), then warps in order
    let quad = |m: f32, ls: f32| -> (f32, f32) {
        let mut mm = -1.0e30f32;
        let mut ms = [0.0f32; 4];
        let mut lsv = [0.0f32; 4];
        kcommon::unroll!(k in 4 => {
            ms[k] = warp::shfl_idx(m, (l & !3) + k as u32);
            lsv[k] = warp::shfl_idx(ls, (l & !3) + k as u32);
            if ms[k] > mm { mm = ms[k]; }
        });
        let mut tl = 0.0f32;
        kcommon::unroll!(k in 4 => {
            tl += lsv[k] * ex(ms[k] - mm);
        });
        (mm, tl)
    };
    let (qma, qla) = quad(ma, la);
    let (qmb, qlb) = quad(mb, lb);
        if t4 == 0 {
            st.set((wid * 16 + ra) * 2, qma);
            st.set((wid * 16 + ra) * 2 + 1, qla);
            st.set((wid * 16 + rb_) * 2, qmb);
            st.set((wid * 16 + rb_) * 2 + 1, qlb);
        }
        sync_threads();
        if t < 16 {
            let mut m = -1.0e30f32;
            kcommon::unroll!(w in 4 => {
                let mw = st.at((w as u32 * 16 + t) * 2);
                if mw > m { m = mw; }
            });
            let mut ls = 0.0f32;
            kcommon::unroll!(w in 4 => {
                ls += st.at((w as u32 * 16 + t) * 2 + 1) * ex(st.at((w as u32 * 16 + t) * 2) - m);
            });
            if gq {
                if t < 3 {
                    *stat.add(((3 * h + t) * nsup + j) as usize * 32) = m;
                    *stat.add(((3 * h + t) * nsup + j) as usize * 32 + 1) = ls;
                }
            } else {
                *stat.add((idx * 32 + t * 2) as usize) = m;
                *stat.add((idx * 32 + t * 2 + 1) as usize) = ls;
            }
        }
        return;
    } else {
        // global statistics of the rows come from ink_attn_gstat (one block a head slot, not one per super-tile)
        let rd = |r: u32| -> (f32, f32) {
            if gq {
                if r < 3 { (*gst.add(((3 * h + r) * 16 * 2) as usize), *gst.add(((3 * h + r) * 16 * 2) as usize + 1)) } else { (0.0, 1.0) }
            } else {
                let e = (((qi * 24 + h) * 16 + r) * 2) as usize;
                (*gst.add(e), *gst.add(e + 1))
            }
        };
        let (a1, a2) = rd(ra);
        let (b1, b2) = rd(rb_);
        gma = a1; gla = a2; gmb = b1; glb = b2;
    }
    // ---- pass 2: p = bf16(exp(s - m) / l), O += P V ----
    let mut o = [[0.0f32; 4]; 16];
    let mut c = cs;
    while c <= ce {
        if qmode != 0 {
            load_chunk_q(ks_s, kc as *const u8, kvh, c * 64, ring);
            load_chunk_q(vs_s, vc as *const u8, kvh, c * 64, ring);
        } else {
            load_chunk(ks_s, kcb, c * 64, ring);
            load_chunk(vs_s, vcb, c * 64, ring);
        }
        cp_async_commit();
        cp_async_wait_all();
        sync_threads();
        scores!(c * 64, s);
        let mut pf = [0u32; 4];
        {
            let pv = |sv: f32, m: f32, ls: f32| -> f32 { if sv > NEG { rb(ex(sv - m) / ls) } else { 0.0 } };
            pf[0] = bf16::pack2(pv(s[0][0], gma, gla), pv(s[0][1], gma, gla));
            pf[1] = bf16::pack2(pv(s[0][2], gmb, glb), pv(s[0][3], gmb, glb));
            pf[2] = bf16::pack2(pv(s[1][0], gma, gla), pv(s[1][1], gma, gla));
            pf[3] = bf16::pack2(pv(s[1][2], gmb, glb), pv(s[1][3], gmb, glb));
        }
        kcommon::unroll!(dp in 8 => {
            let vb4 = ldmatrix_x4_trans(vs_s.byte_add((wid * 16 + (l & 7) + 8 * ((l >> 3) & 1)) * PITCH + (dp as u32 * 16 + (l >> 4) * 8) * 2).addr());
            mma_bf16_16816(&mut o[2 * dp], pf, [vb4[0], vb4[1]]);
            mma_bf16_16816(&mut o[2 * dp + 1], pf, [vb4[2], vb4[3]]);
        });
        sync_threads();
        c += 1;
    }
    // ---- partial outputs of the 4 warps added in warp order ----
    let ob: Shared<f32> = smem.byte_add(KS_OFF).cast();
    kcommon::unroll!(nt in 16 => {
        let col = nt as u32 * 8 + 2 * t4;
        ob.set(wid * 2048 + ra * 128 + col, o[nt][0]);
        ob.set(wid * 2048 + ra * 128 + col + 1, o[nt][1]);
        ob.set(wid * 2048 + rb_ * 128 + col, o[nt][2]);
        ob.set(wid * 2048 + rb_ * 128 + col + 1, o[nt][3]);
    });
    sync_threads();
    let mut i = t;
    if gq {
        while i < 384 {
            let v = ((ob.at(i) + ob.at(2048 + i)) + ob.at(4096 + i)) + ob.at(6144 + i);
            let (r, dd) = (i >> 7, i & 127);
            *part.add((((3 * h + r) * nsup + j) * 128 + dd) as usize) = v;
            i += 128;
        }
        return;
    }
    while i < 2048 {
        let v = ((ob.at(i) + ob.at(2048 + i)) + ob.at(4096 + i)) + ob.at(6144 + i);
        *part.add((idx * 2048 + i) as usize) = v;
        i += 128;
    }
}

/// grid = (24, q tiles in the batch), block 128: out[row][h*128 + d] = bf16(sum of the nsup partials in order).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_attn_tc_comb(out: *mut u16, part: *const f32, nsup: u32, ntok: u32, qt_off: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let h = bid_x();
    let qi = bid_y();
    let d = tid_x();
    let qbase = (qt_off + qi) * 16;
    let mut r = 0;
    while r < 16 {
        if qbase + r < ntok {
            let mut acc = 0.0f32;
            let mut j = 0;
            while j < nsup {
                acc += *part.add((((qi * 24 + h) * nsup + j) * 2048 + r * 128 + d) as usize);
                j += 1;
            }
            *out.add((qbase + r) as usize * 3072 + h as usize * 128 + d as usize) = tb(acc);
        }
        r += 1;
    }
}

/// Attention combine + output gate with a parallel, fixed-association sum over the super-tile partials: groups of 8 (global layers; one
/// group for sliding layers, whose first super-tile depends on the tile's alignment) are summed sequentially, the groups added in order.
/// out[row][h*128 + d] = bf16(bf16(sum) * sigmoid(gate)). grid = (24, q tiles in the batch), block = 1024 (8 group lanes x 128 channels).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_attn_comb2(out: *mut u16, part: *const f32, gate: *const u16, ldg: u32, nsup: u32, ntok: u32,
    qt_off: u32, window: u32, qmode: u32, pstride: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let gs: Shared<f32> = kcommon::static_shared!(cb_gs, f32, 8 * 128 * 4);
    let h = bid_x();
    let qi = bid_y();
    let d = tid_x() & 127;
    let k = tid_x() >> 7;
    let qbase = (qt_off + qi) * 16;
    let gsz = if window > 0 { 16u32 } else { 8u32 };
    let ng = (nsup + gsz - 1) / gsz;
    let mut r = 0;
    while r < 16 {
        if qbase + r < ntok {
            let mut total = 0.0f32;
            let mut b = 0;
            while b < ng {
                let g = b + k;
                let mut gsum = 0.0f32;
                if g < ng {
                    let mut jj = g * gsz;
                    let e = core::cmp::min(g * gsz + gsz, nsup);
                    while jj < e {
                        gsum += *part.add((((qi * 24 + h) * nsup + jj) * pstride + r * 128 + d) as usize);
                        jj += 1;
                    }
                }
                gs.set(k * 128 + d, gsum);
                sync_threads();
                if k == 0 {
                    let n = core::cmp::min(8, ng - b);
                    let mut kk = 0;
                    while kk < n {
                        total += gs.at(kk * 128 + d);
                        kk += 1;
                    }
                }
                sync_threads();
                b += 8;
            }
            if qmode != 0 {
                // back from the rotated basis (the threads of a warp are the 32 dims of a group)
                let mut x = total;
                let mut m = 1;
                while m < 32 {
                    let p = warp::shfl_xor(x, m);
                    x = if tid_x() & m == 0 { x + p } else { p - x };
                    m <<= 1;
                }
                total = x * 0.176_776_7;
            }
            if k == 0 {
                let a = rb(total);
                let gv = *gate.add((qbase + r) as usize * ldg as usize + (h * 128 + d) as usize);
                *out.add((qbase + r) as usize * 3072 + h as usize * 128 + d as usize) = tb(a * sigm(bf(gv)));
            }
        }
        r += 1;
    }
}

/// Global row statistics for the probability pass: per head slot (and 16-row query tile) the super-tile statistics combined in groups
/// (8 for global layers, one group for sliding layers), the groups in order, then the sink: gst[((qtile*24 + h)*16 + row)*2] = (max, sum).
/// gqa: one row (row 0 of slot h). grid = (24, q tiles), block 128.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_attn_gstat(stat: *const f32, gst: *mut f32, sink: *const u16, sld: u32, nsup: u32, ntok: u32,
    qt_off: u32, window: u32, gqa: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let gm: Shared<f32> = kcommon::static_shared!(gs_gm, f32, 16 * 128 * 8);
    let h = bid_x();
    let qi = bid_y();
    let t = tid_x();
    let qbase = (qt_off + qi) * 16;
    let nrows = if gqa != 0 { 1 } else if ntok - qbase < 16 { ntok - qbase } else { 16 };
    let gsz = if window > 0 { 16u32 } else { 8u32 };
    let ng = (nsup + gsz - 1) / gsz;
    let sbase = ((qi * 24 + h) * nsup) as usize;
    let mut idx2 = t;
    while idx2 < nrows * ng {
        let r = idx2 / ng;
        let g = idx2 % ng;
        let mut m = -1.0e30f32;
        let mut jj = g * gsz;
        while jj < core::cmp::min(g * gsz + gsz, nsup) {
            let mj = *stat.add((sbase + jj as usize) * 32 + (r * 2) as usize);
            if mj > m { m = mj; }
            jj += 1;
        }
        let mut ls = 0.0f32;
        jj = g * gsz;
        while jj < core::cmp::min(g * gsz + gsz, nsup) {
            let e = (sbase + jj as usize) * 32 + (r * 2) as usize;
            ls += *stat.add(e + 1) * ex(*stat.add(e) - m);
            jj += 1;
        }
        gm.set((r * ng + g) * 2, m);
        gm.set((r * ng + g) * 2 + 1, ls);
        idx2 += 128;
    }
    sync_threads();
    if t < nrows {
        let r = t;
        let sk = if gqa != 0 { bf(*sink.add(h as usize)) } else { bf(*sink.add((qbase + r) as usize * sld as usize + h as usize)) };
        let mut m = sk;
        let mut g = 0;
        while g < ng {
            let mj = gm.at((r * ng + g) * 2);
            if mj > m { m = mj; }
            g += 1;
        }
        let mut ls = 0.0f32;
        g = 0;
        while g < ng {
            ls += gm.at((r * ng + g) * 2 + 1) * ex(gm.at((r * ng + g) * 2) - m);
            g += 1;
        }
        ls += ex(sk - m);
        let e = (((qi * 24 + h) * 16 + r) * 2) as usize;
        *gst.add(e) = m;
        *gst.add(e + 1) = ls;
    }
}
