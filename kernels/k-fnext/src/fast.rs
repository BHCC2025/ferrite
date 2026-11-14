//! Decode-path kernels (one CUDA graph per step): positions and per-step tables come from the device copy of the
//! step parameters (`prm`, see crates/fe-fnext/src/fast.rs `Prm`), so a captured graph replays at any position.
//! KV cache is q8 (int8 per value, an f32 scale per 32); QSA selection runs on the device.
use kcommon::smem::Shared;
use kcommon::*;

/// Step parameter words (keep in sync with the host `Prm` layout).
pub const P_POS: usize = 0;
pub const P_ROWS: usize = 1;
pub const P_NB: usize = 2; // QSA blocks completed by this step's rows
pub const P_BLO: usize = 3; // first of them
pub const P_EPOCH: usize = 4; // tensor parallel: step epoch << 8 (all-reduce tags = epoch + reduction index)
pub const P_CS: usize = 16; // [24][32] cos for the rows' positions
pub const P_SN: usize = P_CS + 24 * 32;
pub const P_PCS: usize = P_SN + 24 * 32; // [8][32] cos for the completed blocks' first positions
pub const P_PSN: usize = P_PCS + 8 * 32;

#[inline(always)]
pub(crate) unsafe fn pu(prm: *const u32, i: usize) -> u32 {
    *prm.add(i)
}

/// Attention prep: q heads (qf rows stride ldq, head h at h * qstride) and k heads RMS-normed and roped (tables from
/// prm), q to qo [rows][nq][256]; k and v quantized (32-value blocks, f32 scale) into the caches at cell pos + r:
/// kq / vq [cell][nkv][256] i8, ks / vs [cell][nkv][8] f32. grid = (rows, nq + nkv), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_attn_prep(qo: *mut f32, kq: *mut i8, ks: *mut f32, vq: *mut i8, vs: *mut f32,
    qf: *const f32, ldq: u32, qstride: u32, kx: *const f32, vx: *const f32, ldkv: u32, qnw: *const f32, knw: *const f32,
    prm: *const u32, nq: u32, nkv: u32, nrot: u32, eps: f32, cs: *const f32, sn: *const f32) {
    let red: Shared<f32> = kcommon::static_shared!(xp_red, f32, 128);
    let xs: Shared<f32> = kcommon::static_shared!(xp_x, f32, 1024);
    let r = bid_x();
    let h = bid_y();
    let i = tid_x();
    let isq = h < nq;
    let src = if isq { qf.add((r * ldq + h * qstride + i) as usize) } else { kx.add((r * ldkv + (h - nq) * 256 + i) as usize) };
    let v = *src;
    let tot = warp::block_sum(v * v, red);
    let wv = if isq { *qnw.add(i as usize) } else { *knw.add(i as usize) };
    let xn = v * (1.0 / math::sqrt(tot / 256.0 + eps)) * wv;
    xs.set(i, xn);
    sync_threads();
    let half = nrot / 2;
    let mut out = xn;
    if i < nrot {
        let p = i % half;
        let c = *cs.add((r * 32 + p) as usize);
        let s = *sn.add((r * 32 + p) as usize);
        out = if i < half { xn * c - xs.at(i + half) * s } else { xn * c + xs.at(i - half) * s };
    }
    if isq {
        *qo.add(((r * nq + h) * 256 + i) as usize) = out;
        return;
    }
    let kh = h - nq;
    let cell = (pu(prm, P_POS) + r) as usize;
    let base = (cell * nkv as usize + kh as usize) * 256 + i as usize;
    let sb = (cell * nkv as usize + kh as usize) * 8 + (i / 32) as usize;
    let q8 = |x: f32, q: *mut i8, s: *mut f32| {
        let a = warp::max(if x < 0.0 { -x } else { x });
        let d = a / 127.0;
        let qv = if a == 0.0 { 0.0 } else { x / d };
        let qv = if qv < 0.0 { -math_floor(-qv + 0.5) } else { math_floor(qv + 0.5) };
        *q.add(base) = qv as i32 as i8;
        if lane() == 0 {
            *s.add(sb) = d;
        }
    };
    q8(out, kq, ks);
    q8(*vx.add((r * ldkv + kh * 256 + i) as usize), vq, vs);
}

#[inline(always)]
fn math_floor(f: f32) -> f32 {
    let r: f32;
    unsafe { core::arch::asm!("cvt.rmi.f32.f32 {r}, {f};", r = out(reg32) r, f = in(reg32) f, options(pure, nomem, nostack)) }
    r
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

/// Attention for rows at cells pos + r over the q8 caches: dense over cells 0 ..= pos + r when seln[r] == 0, else
/// over the seln[r] cells sel[r][..] (QSA). Output gated by sigmoid(gate). grid = (rows, nq), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_attn(out: *mut f32, q: *const f32, kq: *const i8, ks: *const f32, vq: *const i8,
    vs: *const f32, gate: *const f32, ldg: u32, gstride: u32, nq: u32, nkv: u32, prm: *const u32, scale: f32,
    sel: *const u32, seln: *const u32, ldsel: u32) {
    let sm: Shared<f32> = kcommon::static_shared!(xa_m, f32, 64);
    let sl: Shared<f32> = kcommon::static_shared!(xa_l, f32, 64);
    let sacc: Shared<f32> = kcommon::static_shared!(xa_acc, f32, 8192);
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
    let n = if dense { pu(prm, P_POS) + r + 1 } else { ns };
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

/// Raw indexer keys of this step's rows into a 16-cell ring (cell c at slot c % 16): ring[(pos + r) % 16] = k[r].
/// grid = rows, block 128.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_idx_kstore(ring: *mut f32, k: *const f32, prm: *const u32, ring_n: u32) {
    let r = bid_x();
    let slot = (pu(prm, P_POS) + r) % ring_n;
    *ring.add((slot * 128 + tid_x()) as usize) = *k.add((r * 128 + tid_x()) as usize);
}

/// Pool the QSA blocks this step completed (prm: nb, b_lo): pooled[b] = rope(rmsnorm(mean of the kp raw keys) * w)
/// at the block's first position. grid = 2, block 128.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_idx_pool(pooled: *mut f32, ring: *const f32, w: *const f32, prm: *const u32, kp: u32,
    nrot: u32, eps: f32, ring_n: u32, pcs: *const f32, psn: *const f32) {
    let red: Shared<f32> = kcommon::static_shared!(xq_red, f32, 128);
    let xs: Shared<f32> = kcommon::static_shared!(xq_x, f32, 512);
    let j = bid_x();
    if j >= pu(prm, P_NB) {
        return;
    }
    let b = pu(prm, P_BLO) + j;
    let i = tid_x();
    let mut acc = 0.0f32;
    let mut k = 0;
    while k < kp {
        acc += *ring.add((((b * kp + k) % ring_n) * 128 + i) as usize);
        k += 1;
    }
    let v = acc * (1.0 / kp as f32);
    let tot = warp::block_sum(v * v, red);
    let xn = v * (1.0 / math::sqrt(tot / 128.0 + eps)) * *w.add(i as usize);
    xs.set(i, xn);
    sync_threads();
    let half = nrot / 2;
    let mut o = xn;
    if i < nrot {
        let p = i % half;
        let c = *pcs.add((j * 32 + p) as usize);
        let s = *psn.add((j * 32 + p) as usize);
        o = if i < half { xn * c - xs.at(i + half) * s } else { xn * c + xs.at(i - half) * s };
    }
    *pooled.add((b * 128 + i) as usize) = o;
}

/// QSA block scores for the visible blocks b < (pos + r + 1) / kp, only for rows that see more than `top` blocks.
/// grid = (rows, G) block 256, a warp per block, grid-stride. score [rows][ld].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_idx_score(score: *mut f32, q: *const f32, pooled: *const f32, nh: u32, prm: *const u32,
    kp: u32, ld: u32, top: u32, scale: f32) {
    let r = bid_x();
    let nv = (pu(prm, P_POS) + r + 1) / kp;
    if nv <= top {
        return;
    }
    let l = lane();
    let mut qv = [[0.0f32; 4]; 4];
    let mut h = 0;
    while h < nh && h < 4 {
        let qp = q.add(((r * nh + h) * 128 + l * 4) as usize);
        qv[h as usize] = [*qp, *qp.add(1), *qp.add(2), *qp.add(3)];
        h += 1;
    }
    let mut b = bid_y() * 8 + warp_id();
    let stride = gdim_y() * 8;
    while b < nv && b < ld {
        let kp4 = pooled.add((b * 128 + l * 4) as usize);
        let kv = [*kp4, *kp4.add(1), *kp4.add(2), *kp4.add(3)];
        let mut tot = 0.0f32;
        let mut h = 0;
        while h < nh {
            let qh = qv[h as usize];
            let d = warp::sum(qh[0] * kv[0] + qh[1] * kv[1] + qh[2] * kv[2] + qh[3] * kv[3]);
            tot += math::maxf(d, 0.0);
            h += 1;
        }
        if l == 0 {
            *score.add((r * ld + b) as usize) = tot * scale;
        }
        b += stride;
    }
}

/// Order-preserving key of a float (larger float -> larger key).
#[inline(always)]
fn fkey(f: f32) -> u32 {
    let b = f.to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}

/// QSA selection per row: rows that see more than `top` blocks get the top `top` blocks by score (ties: lower block
/// first) expanded to their kp cells, plus the open tail ((pos + 1) % kp cells ending at pos); seln[r] = cell count.
/// Rows with nv <= top get seln[r] = 0 (dense). grid = rows, block 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_topk(sel: *mut u32, seln: *mut u32, score: *const f32, prm: *const u32, kp: u32,
    top: u32, ld: u32, ldsel: u32) {
    let hist: Shared<u32> = kcommon::static_shared!(tk_hist, u32, 1024);
    let sh: Shared<u32> = kcommon::static_shared!(tk_sh, u32, 512);
    let r = bid_x();
    let t = tid_x();
    let p = pu(prm, P_POS) + r;
    let nv = (p + 1) / kp;
    if nv <= top {
        if t == 0 {
            *seln.add(r as usize) = 0;
        }
        return;
    }
    let sr = score.add((r * ld) as usize);
    // radix select of the top-th largest key, 8 bits at a time
    let mut prefix = 0u32;
    let mut pmask = 0u32;
    let mut need = top; // how many still to take at or below the current prefix
    let mut shift = 24i32;
    while shift >= 0 {
        if t < 256 {
            hist.set(t, 0);
        }
        sync_threads();
        let mut b = t;
        while b < nv {
            let k = fkey(*sr.add(b as usize));
            if k & pmask == prefix {
                let dgt = (k >> shift as u32) & 255;
                core::arch::asm!("red.shared.add.u32 [{a}], 1;", a = in(reg32) hist.add(dgt).addr(), options(nostack));
            }
            b += bdim_x();
        }
        sync_threads();
        if t == 0 {
            // walk digits from the top until the count reaches `need`
            let mut cum = 0u32;
            let mut d = 255i32;
            while d >= 0 {
                let c = hist.at(d as u32);
                if cum + c >= need {
                    break;
                }
                cum += c;
                d -= 1;
            }
            sh.set(0, d as u32);
            sh.set(1, need - cum);
        }
        sync_threads();
        let d = sh.at(0);
        need = sh.at(1);
        prefix |= d << shift as u32;
        pmask |= 255 << shift as u32;
        sync_threads();
        shift -= 8;
    }
    // prefix = key of the top-th largest; take every key > prefix, then `need` keys == prefix in block order
    let thr = prefix;
    if t == 0 {
        sh.set(2, 0); // count of strictly greater
        sh.set(3, 0); // ties taken
    }
    sync_threads();
    let srow = sel.add((r * ldsel) as usize);
    let mut base = 0u32;
    while base < nv {
        let b = base + t;
        let k = if b < nv { fkey(*sr.add(b as usize)) } else { 0 };
        let gt = b < nv && k > thr;
        let eq = b < nv && k == thr;
        // tie rank in block order: count ties before this lane in the chunk
        let ball = |pred: bool| -> u32 {
            let v: u32;
            core::arch::asm!("{{ .reg .pred p; setp.ne.u32 p, {x}, 0; vote.sync.ballot.b32 {v}, p, 0xffffffff; }}",
                x = in(reg32) pred as u32, v = out(reg32) v, options(nostack));
            v
        };
        let be = ball(eq);
        let lm = (1u32 << lane()) - 1;
        let wid = warp_id();
        if lane() == 0 {
            sh.set(8 + wid, be.count_ones());
        }
        sync_threads();
        let mut before = 0u32;
        let mut wv = 0;
        while wv < wid {
            before += sh.at(8 + wv);
            wv += 1;
        }
        let tie_rank = sh.at(3) + before + (be & lm).count_ones();
        let take = gt || (eq && tie_rank < need);
        // slot = rank of this block among the taken ones in block order (deterministic: no atomic order)
        let bt = ball(take);
        if lane() == 0 {
            sh.set(48 + wid, bt.count_ones());
        }
        sync_threads();
        let mut tbefore = 0u32;
        let mut wv = 0;
        while wv < wid {
            tbefore += sh.at(48 + wv);
            wv += 1;
        }
        if take {
            let slot = sh.at(2) + tbefore + (bt & lm).count_ones();
            let mut k2 = 0;
            while k2 < kp {
                *srow.add((slot * kp + k2) as usize) = b * kp + k2;
                k2 += 1;
            }
        }
        sync_threads();
        if t == 0 {
            let mut tot = 0u32;
            let mut tk = 0u32;
            let mut wv = 0;
            while wv < 32 {
                tot += sh.at(8 + wv);
                tk += sh.at(48 + wv);
                wv += 1;
            }
            sh.set(3, sh.at(3) + tot);
            sh.set(2, sh.at(2) + tk);
        }
        sync_threads();
        base += bdim_x();
    }
    if t == 0 {
        let mut n = top * kp;
        let nt = (p + 1) % kp;
        let mut k = 0;
        while k < nt {
            *srow.add(n as usize) = p - k;
            n += 1;
            k += 1;
        }
        *seln.add(r as usize) = n;
    }
}

/// MoE router: softmax over n_exp logits, top-k by probability (ties: lower id), weights renormalised (sum clamped
/// at 6.1035e-5) and scaled (scale 0: none). ids i32 [rows][k], w f32 [rows][k]. grid = rows, block 32. n_exp <= 512.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_route(ids: *mut i32, wts: *mut f32, logits: *const f32, n_exp: u32, k: u32, scale: f32) {
    let r = bid_x();
    let l = lane();
    let lr = logits.add((r * n_exp) as usize);
    let per = n_exp / 32;
    let mut v = [f32::NEG_INFINITY; 16];
    let mut m = f32::NEG_INFINITY;
    let mut j = 0;
    while j < per {
        v[j as usize] = *lr.add((j * 32 + l) as usize);
        m = math::maxf(m, v[j as usize]);
        j += 1;
    }
    m = warp::max(m);
    let mut s = 0.0f32;
    let mut j = 0;
    while j < per {
        v[j as usize] = math::exp(v[j as usize] - m);
        s += v[j as usize];
        j += 1;
    }
    s = warp::sum(s);
    let mut sel_sum = 0.0f32;
    let mut taken = [0u32; 16];
    let mut kk = 0;
    while kk < k {
        // lane-local best (lowest id on ties), then warp argmax
        let mut bv = f32::NEG_INFINITY;
        let mut bi = u32::MAX;
        let mut j = 0;
        while j < per {
            let id = j * 32 + l;
            if taken[j as usize] == 0 && (v[j as usize] > bv || (v[j as usize] == bv && id < bi)) {
                bv = v[j as usize];
                bi = id;
            }
            j += 1;
        }
        let mut msk = 16;
        while msk > 0 {
            let ov = warp::shfl_xor(bv, msk);
            let oi = warp::shfl_xor_u32(bi, msk);
            if ov > bv || (ov == bv && oi < bi) {
                bv = ov;
                bi = oi;
            }
            msk >>= 1;
        }
        if bi % 32 == l {
            taken[(bi / 32) as usize] = 1;
        }
        let pr = bv / s;
        sel_sum += pr;
        if l == 0 {
            *ids.add((r * k + kk) as usize) = bi as i32;
            *wts.add((r * k + kk) as usize) = pr;
        }
        kk += 1;
    }
    sync_warp();
    if l < k {
        let inv = 1.0 / math::maxf(sel_sum, 6.103515625e-5);
        let p = wts.add((r * k + l) as usize);
        *p = *p * inv * if scale != 0.0 { scale } else { 1.0 };
    }
}

/// Copy the CPU-computed expert rows into y: for pair p whose expert has no slot on the card (tab[ids[p]] == 0),
/// y[p][:] = host[p][:] (host: pinned, mapped). grid = (ceil(n/256), pairs), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_moe_merge(y: *mut f32, host: *const f32, ids: *const i32, tab: *const u64, n: u32) {
    let p = bid_y();
    let e = *ids.add(p as usize);
    if *tab.add(e as usize) != 0 {
        return;
    }
    let i = bid_x() * 256 + tid_x();
    if i < n {
        let hv = core::ptr::read_volatile(host.add((p * n + i) as usize));
        *y.add((p * n + i) as usize) = hv;
    }
}

/// Greedy argmax over f32 logit rows (lowest index on ties); pmax (optional): its softmax probability.
/// grid = rows, block 1024.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_argmax(out: *mut u32, x: *const f32, n: u32, pmax: *mut f32) {
    let sv: Shared<f32> = kcommon::static_shared!(xm_v, f32, 128);
    let si: Shared<u32> = kcommon::static_shared!(xm_i, u32, 128);
    let r = bid_x() as usize;
    let xr = x.add(r * n as usize);
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0u32;
    let mut j = tid_x();
    while j < n {
        let v = *xr.add(j as usize);
        if v > best {
            best = v;
            bi = j;
        }
        j += bdim_x();
    }
    let mut m = 16;
    while m > 0 {
        let ov = warp::shfl_xor(best, m);
        let oi = warp::shfl_xor_u32(bi, m);
        if ov > best || (ov == best && oi < bi) {
            best = ov;
            bi = oi;
        }
        m >>= 1;
    }
    if lane() == 0 {
        sv.set(warp_id(), best);
        si.set(warp_id(), bi);
    }
    sync_threads();
    if warp_id() == 0 {
        let nw = nwarps();
        let l = lane();
        best = if l < nw { sv.at(l) } else { f32::NEG_INFINITY };
        bi = if l < nw { si.at(l) } else { u32::MAX };
        let mut m = 16;
        while m > 0 {
            let ov = warp::shfl_xor(best, m);
            let oi = warp::shfl_xor_u32(bi, m);
            if ov > best || (ov == best && oi < bi) {
                best = ov;
                bi = oi;
            }
            m >>= 1;
        }
        if l == 0 {
            *out.add(r) = bi;
            sv.set(0, best);
        }
    }
    if pmax.is_null() {
        return;
    }
    // softmax probability of the argmax: 1 / sum exp(x - max)
    sync_threads();
    let mx = sv.at(0);
    let mut acc = 0.0f32;
    let mut j = tid_x();
    while j < n {
        acc += math::exp(*xr.add(j as usize) - mx);
        j += bdim_x();
    }
    let tot = warp::block_sum(acc, si.cast::<f32>());
    if tid_x() == 0 {
        *pmax.add(r) = 1.0 / tot;
    }
}

#[inline(always)]
fn h8(wv: [u32; 4]) -> [f32; 8] {
    let mut o = [0.0f32; 8];
    unroll!(i in 4 => {
        o[2 * i] = crate::h2f_pub(wv[i] as u16);
        o[2 * i + 1] = crate::h2f_pub((wv[i] >> 16) as u16);
    });
    o
}

/// bf16 / f16 GEMV for 1..=8 activation rows, `tpr` threads per output row (8..=256, a power of two; 256 / tpr rows
/// per block): y[r][j * ...] (ldy) = (acc ? y : 0) + scale * W[j] . x[r] (ldx). W [n][k], k % 8 == 0.
/// Each thread streams 16-byte weight chunks c = lane-in-row, + tpr, ...; partial sums reduced over the row's
/// threads (shuffles inside a warp, shared memory across warps). grid = ceil(n / (256 / tpr)), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_gemv_h(y: *mut f32, x: *const f32, w: *const u16, k: u32, n: u32, rows: u32,
    scale: f32, bf: u32, acc: u32, ldx: u32, ldy: u32, tpr: u32) {
    let red: Shared<f32> = kcommon::static_shared!(gh_red, f32, 8 * 8 * 4);
    let t = tid_x();
    let gpb = 256 / tpr;
    let jl = t / tpr;
    let li = t % tpr;
    let j = bid_x() * gpb + jl;
    let mut a = [0.0f32; 8];
    if j < n {
        let wr = w.add(j as usize * k as usize);
        let nc = k / 8;
        let mut c = li;
        // two chunks in flight per thread
        while c + tpr < nc {
            let wv0 = ldg_stream_v4(wr.add(c as usize * 8) as *const u32);
            let wv1 = ldg_stream_v4(wr.add((c + tpr) as usize * 8) as *const u32);
            let (wf0, wf1) = if bf != 0 { (bf16::unpack8(wv0), bf16::unpack8(wv1)) } else { (h8(wv0), h8(wv1)) };
            unroll!(r in 8 => {
                if (r as u32) < rows {
                    let xp = x.add(r * ldx as usize + c as usize * 8);
                    let xq = x.add(r * ldx as usize + (c + tpr) as usize * 8);
                    let x0 = ldg_nc_v4(xp as *const u32);
                    let x1 = ldg_nc_v4(xp.add(4) as *const u32);
                    let x2 = ldg_nc_v4(xq as *const u32);
                    let x3 = ldg_nc_v4(xq.add(4) as *const u32);
                    let mut s = 0.0f32;
                    unroll!(e in 4 => { s = math::fma(wf0[e], f32::from_bits(x0[e]), s); });
                    unroll!(e in 4 => { s = math::fma(wf0[4 + e], f32::from_bits(x1[e]), s); });
                    unroll!(e in 4 => { s = math::fma(wf1[e], f32::from_bits(x2[e]), s); });
                    unroll!(e in 4 => { s = math::fma(wf1[4 + e], f32::from_bits(x3[e]), s); });
                    a[r] += s;
                }
            });
            c += 2 * tpr;
        }
        while c < nc {
            let wv = ldg_stream_v4(wr.add(c as usize * 8) as *const u32);
            let wf = if bf != 0 { bf16::unpack8(wv) } else {
                let mut o = [0.0f32; 8];
                unroll!(i in 4 => {
                    o[2 * i] = crate::h2f_pub(wv[i] as u16);
                    o[2 * i + 1] = crate::h2f_pub((wv[i] >> 16) as u16);
                });
                o
            };
            unroll!(r in 8 => {
                if (r as u32) < rows {
                    let xp = x.add(r * ldx as usize + c as usize * 8);
                    let x0 = ldg_nc_v4(xp as *const u32);
                    let x1 = ldg_nc_v4(xp.add(4) as *const u32);
                    let mut s = 0.0f32;
                    unroll!(e in 4 => { s = math::fma(wf[e], f32::from_bits(x0[e]), s); });
                    unroll!(e in 4 => { s = math::fma(wf[4 + e], f32::from_bits(x1[e]), s); });
                    a[r] += s;
                }
            });
            c += tpr;
        }
    }
    // reduce over the row's threads
    let wtp = if tpr < 32 { tpr } else { 32 };
    unroll!(r in 8 => {
        let mut v = a[r];
        let mut m = wtp >> 1;
        while m > 0 {
            v += warp::shfl_xor(v, m);
            m >>= 1;
        }
        a[r] = v;
    });
    if tpr > 32 {
        // one partial per warp: red[warp][r]
        if lane() == 0 {
            unroll!(r in 8 => { red.set(warp_id() * 8 + r as u32, a[r]); });
        }
        sync_threads();
        let wpr = tpr / 32;
        if li == 0 && j < n {
            let w0 = warp_id();
            unroll!(r in 8 => {
                let mut v = 0.0f32;
                let mut q = 0;
                while q < wpr {
                    v += red.at((w0 + q) * 8 + r as u32);
                    q += 1;
                }
                a[r] = v;
            });
        }
    }
    if li == 0 && j < n {
        unroll!(r in 8 => {
            if (r as u32) < rows {
                let p = y.add(r * ldy as usize + j as usize);
                *p = if acc != 0 { *p + scale * a[r] } else { scale * a[r] };
            }
        });
    }
}

/// Batched MoE router for rows x 1 warp each is slow for a single row; this variant spreads the softmax over 8 warps
/// and keeps the selection in warp 0. Same contract as fx_route. grid = rows, block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_route8(ids: *mut i32, wts: *mut f32, logits: *const f32, ld: u32, n_exp: u32, k: u32,
    scale: f32) {
    let ps: Shared<f32> = kcommon::static_shared!(r8_p, f32, 2048);
    let red: Shared<f32> = kcommon::static_shared!(r8_red, f32, 128);
    let r = bid_x();
    let t = tid_x();
    let lr = logits.add((r * ld) as usize);
    let mut v = [f32::NEG_INFINITY; 2];
    let mut m = f32::NEG_INFINITY;
    unroll!(i in 2 => {
        let e = t + i as u32 * 256;
        if e < n_exp {
            v[i] = *lr.add(e as usize);
            m = math::maxf(m, v[i]);
        }
    });
    let m = warp::block_max(m, red);
    let mut s = 0.0f32;
    unroll!(i in 2 => {
        let e = t + i as u32 * 256;
        if e < n_exp {
            let p = math::exp(v[i] - m);
            ps.set(e, p);
            s += p;
        }
    });
    let s = warp::block_sum(s, red);
    if warp_id() != 0 {
        return;
    }
    let l = lane();
    let per = n_exp / 32;
    let mut taken = 0u32; // bit j: element j*32 + l taken
    let mut sel_sum = 0.0f32;
    let mut kk = 0;
    while kk < k {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = u32::MAX;
        let mut jj = 0;
        while jj < per {
            let id = jj * 32 + l;
            let pv = ps.at(id);
            if taken & (1 << jj) == 0 && (pv > bv || (pv == bv && id < bi)) {
                bv = pv;
                bi = id;
            }
            jj += 1;
        }
        let mut msk = 16;
        while msk > 0 {
            let ov = warp::shfl_xor(bv, msk);
            let oi = warp::shfl_xor_u32(bi, msk);
            if ov > bv || (ov == bv && oi < bi) {
                bv = ov;
                bi = oi;
            }
            msk >>= 1;
        }
        if bi % 32 == l {
            taken |= 1 << (bi / 32);
        }
        let pr = bv / s;
        sel_sum += pr;
        if l == 0 {
            *ids.add((r * k + kk) as usize) = bi as i32;
            *wts.add((r * k + kk) as usize) = pr;
        }
        kk += 1;
    }
    sync_warp();
    if l < k {
        let inv = 1.0 / math::maxf(sel_sum, 6.103515625e-5);
        let p = wts.add((r * k + l) as usize);
        *p = *p * inv * if scale != 0.0 { scale } else { 1.0 };
    }
}

/// out[r][i] = moe[r][i] + sh[r][i] * sigmoid(glog[r * ldg]); n = rows * d.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_moe_out(out: *mut f32, moe: *const f32, sh: *const f32, glog: *const f32, ldg: u32,
    d: u32, n: u32) {
    let idx = bid_x() * 256 + tid_x();
    if idx < n {
        let g = 1.0 / (1.0 + math::exp(-*glog.add(((idx / d) * ldg) as usize)));
        *out.add(idx as usize) = *moe.add(idx as usize) + *sh.add(idx as usize) * g;
    }
}

/// x[r][i] = silu(x[r][i]) for i < m, rows of stride ld. grid = (ceil(m/256), rows), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_silu_ld(x: *mut f32, m: u32, ld: u32) {
    let i = bid_x() * 256 + tid_x();
    if i < m {
        let p = x.add((bid_y() * ld + i) as usize);
        let v = *p;
        *p = v / (1.0 + math::exp(-v));
    }
}

/// HC combine with the inject weights at inj[r * ldi + s]: res[r][s][i] += out[r][i] * 2 sigmoid(inj). n = rows*hc*d.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_hc_combine(res: *mut f32, out: *const f32, inj: *const f32, d: u32, hc: u32, ldi: u32,
    n: u32) {
    // let the next (PDL-launched) kernel start now: it prefetches its weights while this one runs / waits, and does
    // griddepcontrol.wait before reading anything written here
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    let idx = bid_x() * 256 + tid_x();
    if idx < n {
        let i = idx % d;
        let rs = idx / d;
        let (r, s) = (rs / hc, rs % hc);
        let w = 2.0 / (1.0 + math::exp(-*inj.add((r * ldi + s) as usize)));
        *res.add(idx as usize) += *out.add((r * d + i) as usize) * w;
    }
}

/// MoE finish + HC combine in one pass: for token t and column i,
/// moe = sum_k w[t][k] * (expert on the card ? ey[p][i] : host[p][i]) (p = t * topk + k, fixed order),
/// out = moe + sh[t][i] * sigmoid(glog[t * ldg]), res[t][s][i] += out * 2 sigmoid(inj[t * ldi + s]).
/// host: the CPU rows (pinned, mapped). grid = (ceil(d / 256), rows), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_moe_fin(res: *mut f32, ey: *const f32, host: *const f32, ids: *const i32,
    dtab: *const u64, wts: *const f32, sh: *const f32, glog: *const f32, ldg: u32, inj: *const f32, ldi: u32, d: u32,
    hc: u32, topk: u32, done: *mut u32) {
    // let the next (PDL-launched) kernel start now: it prefetches its weights while this one runs / waits, and does
    // griddepcontrol.wait before reading anything written here
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    let t = bid_y();
    let i = bid_x() * 256 + tid_x();
    // re-arm the layer's done flag (the CPU raises it again only in the next step) instead of a graph memop node
    if !done.is_null() && i == 0 && t == 0 {
        core::ptr::write_volatile(done, 0);
    }
    if i >= d {
        return;
    }
    // all loads first (the host rows are PCIe round trips: issue them together), then the sum in pair order
    let mut v = [0.0f32; 16];
    kcommon::unroll!(k in 16 => {
        if (k as u32) < topk {
            let p = t * topk + k as u32;
            let e = *ids.add(p as usize);
            v[k] = if *dtab.add(e as usize) != 0 { *ey.add((p * d + i) as usize) } else {
                core::ptr::read_volatile(host.add((p * d + i) as usize))
            };
        }
    });
    let mut moe = 0.0f32;
    kcommon::unroll!(k in 16 => {
        if (k as u32) < topk {
            moe += *wts.add((t * topk + k as u32) as usize) * v[k];
        }
    });
    let g = 1.0 / (1.0 + math::exp(-*glog.add((t * ldg) as usize)));
    let out = moe + *sh.add((t * d + i) as usize) * g;
    let mut s = 0;
    while s < hc {
        let w = 2.0 / (1.0 + math::exp(-*inj.add((t * ldi + s) as usize)));
        *res.add(((t * hc + s) * d + i) as usize) += out * w;
        s += 1;
    }
}

/// x[r][i] *= s for i < m (rows of stride ld). grid = (ceil(m/256), rows), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_scale_ld(x: *mut f32, m: u32, ld: u32, s: f32) {
    let i = bid_x() * 256 + tid_x();
    if i < m {
        let p = x.add((bid_y() * ld + i) as usize);
        *p *= s;
    }
}

/// MTP input rows: cat[r][s][:] = [en[r][:] | hn[r][s][:]] (each d wide). grid = (ceil(2d/256), rows * hc), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_mtp_cat(cat: *mut f32, en: *const f32, hn: *const f32, d: u32, hc: u32) {
    let rs = bid_y();
    let r = rs / hc;
    let i = bid_x() * 256 + tid_x();
    if i >= 2 * d {
        return;
    }
    let v = if i < d { *en.add((r * d + i) as usize) } else { *hn.add((rs * d + i - d) as usize) };
    *cat.add((rs * 2 * d + i) as usize) = v;
}

/// Prompt-group MoE accumulation into the residual: for token t, column i,
/// out = sum over pairs k with tab[ids] != 0 of w[t][k] * ey[p][i] (+ sh[t][i] * sigmoid(glog[t * ldg]) when sh is
/// not null), res[t][s][i] += out * 2 sigmoid(inj[t * ldi + s]). grid = (ceil(d / 256), rows), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_moe_acc(res: *mut f32, ey: *const f32, ids: *const i32, tab: *const u64,
    wts: *const f32, sh: *const f32, glog: *const f32, ldg: u32, inj: *const f32, ldi: u32, d: u32, hc: u32, topk: u32) {
    let t = bid_y();
    let i = bid_x() * 256 + tid_x();
    if i >= d {
        return;
    }
    let mut out = 0.0f32;
    if !tab.is_null() {
        let mut k = 0;
        while k < topk {
            let p = t * topk + k;
            if *tab.add(*ids.add(p as usize) as usize) != 0 {
                out += *wts.add(p as usize) * *ey.add((p * d + i) as usize);
            }
            k += 1;
        }
    }
    if !sh.is_null() {
        out += *sh.add((t * d + i) as usize) * (1.0 / (1.0 + math::exp(-*glog.add((t * ldg) as usize))));
    }
    let mut s = 0;
    while s < hc {
        let w = 2.0 / (1.0 + math::exp(-*inj.add((t * ldi + s) as usize)));
        *res.add(((t * hc + s) * d + i) as usize) += out * w;
        s += 1;
    }
}

/// dst[r][c] = src[r][c] for c < cols (f32 rows of strides ldd / lds). grid = (ceil(cols/256), rows), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_copy2d(dst: *mut f32, ldd: u32, src: *const f32, lds: u32, cols: u32) {
    let c = bid_x() * 256 + tid_x();
    if c < cols {
        *dst.add((bid_y() * ldd + c) as usize) = *src.add((bid_y() * lds + c) as usize);
    }
}

/// Copy the router picks and the FFN input rows to pinned host memory for the CPU expert thread (zero-copy stores):
/// hids[r][k] = ids[r][k], hx[r][:] = x[r][:]. grid = rows, block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_to_host(hids: *mut i32, ids: *const i32, hx: *mut f32, x: *const f32, k: u32, d: u32) {
    let r = bid_x();
    let t = tid_x();
    if t < k {
        core::ptr::write_volatile(hids.add((r * k + t) as usize), *ids.add((r * k + t) as usize));
    }
    let mut i = t * 4;
    while i < d {
        let v = ldg_nc_v4(x.add((r * d + i) as usize) as *const u32);
        core::ptr::write_volatile(hx.add((r * d + i) as usize) as *mut [u32; 4], v);
        i += 1024;
    }
    core::arch::asm!("membar.sys;");
}

/// When none of the pairs misses the card (tab[ids[p]] != 0 for all p), raise the layer's done flag on the device
/// side (the CPU thread does not touch it then). grid = 1, block 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_hit_flag(flag: *mut u32, ids: *const i32, tab: *const u64, np: u32) {
    let l = lane();
    let mut miss = 0u32;
    let mut p = l;
    while p < np {
        if *tab.add(*ids.add(p as usize) as usize) == 0 {
            miss = 1;
        }
        p += 32;
    }
    let any: u32;
    core::arch::asm!("{{ .reg .pred p; setp.ne.u32 p, {x}, 0; vote.sync.any.pred p, p, 0xffffffff; selp.u32 {v}, 1, 0, p; }}",
        x = in(reg32) miss, v = out(reg32) any, options(nostack));
    if any == 0 && l == 0 {
        core::ptr::write_volatile(flag, 1);
        core::arch::asm!("membar.sys;");
    }
}

/// fx_attn for prompt rows with 6 q heads of one kv head per block (the K / V rows are read once for the six).
/// Same contract as fx_attn; grid = (rows, nq / 6), block 256 (8 warps over the cells, lane = 8 dims).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_attn6(out: *mut f32, q: *const f32, kq: *const i8, ks: *const f32, vq: *const i8,
    vs: *const f32, gate: *const f32, ldg: u32, gstride: u32, nq: u32, nkv: u32, prm: *const u32, scale: f32,
    sel: *const u32, seln: *const u32, ldsel: u32) {
    let sm: Shared<f32> = kcommon::static_shared!(x6_m, f32, 256);
    let sl: Shared<f32> = kcommon::static_shared!(x6_l, f32, 256);
    let sacc: Shared<f32> = kcommon::static_shared!(x6_acc, f32, 8192);
    let r = bid_x();
    let h0 = bid_y() * 6;
    let kh = h0 / (nq / nkv);
    let w = warp_id();
    let l = lane();
    let mut qv = [[0.0f32; 8]; 6];
    unroll!(hh in 6 => {
        let qp = q.add(((r * nq + h0 + hh as u32) * 256 + l * 8) as usize);
        unroll!(e in 8 => { qv[hh][e] = *qp.add(e) * scale; });
    });
    let ns = *seln.add(r as usize);
    let dense = ns == 0;
    let n = if dense { pu(prm, P_POS) + r + 1 } else { ns };
    let mut m = [f32::NEG_INFINITY; 6];
    let mut ls = [0.0f32; 6];
    let mut acc = [[0.0f32; 8]; 6];
    let mut pi = w;
    while pi < n {
        let p = if dense { pi } else { *sel.add((r * ldsel + pi) as usize) };
        let cb = (p * nkv + kh) as usize;
        let kv = ld8q(kq.add(cb * 256 + (l * 8) as usize), *ks.add(cb * 8 + (l / 4) as usize));
        let vv = ld8q(vq.add(cb * 256 + (l * 8) as usize), *vs.add(cb * 8 + (l / 4) as usize));
        unroll!(hh in 6 => {
            let mut d = 0.0f32;
            unroll!(e in 8 => { d = math::fma(qv[hh][e], kv[e], d); });
            let sc = warp::sum(d);
            let mn = math::maxf(m[hh], sc);
            let a = math::exp(m[hh] - mn);
            let b = math::exp(sc - mn);
            unroll!(e in 8 => { acc[hh][e] = acc[hh][e] * a + b * vv[e]; });
            ls[hh] = ls[hh] * a + b;
            m[hh] = mn;
        });
        pi += 8;
    }
    let t = tid_x();
    unroll!(hh in 6 => {
        if l == 0 {
            sm.set(hh as u32 * 8 + w, m[hh]);
            sl.set(hh as u32 * 8 + w, ls[hh]);
        }
        unroll!(e in 8 => { sacc.set(w * 256 + l * 8 + e as u32, acc[hh][e]); });
        sync_threads();
        let mut mm = f32::NEG_INFINITY;
        unroll!(k in 8 => { mm = math::maxf(mm, sm.at(hh as u32 * 8 + k as u32)); });
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        unroll!(k in 8 => {
            let mk = sm.at(hh as u32 * 8 + k as u32);
            if mk != f32::NEG_INFINITY {
                let f = math::exp(mk - mm);
                num += sacc.at(k as u32 * 256 + t) * f;
                den += sl.at(hh as u32 * 8 + k as u32) * f;
            }
        });
        let h = h0 + hh as u32;
        let gv = *gate.add((r * ldg + h * gstride + t) as usize);
        *out.add(((r * nq + h) * 256 + t) as usize) = num / den * (1.0 / (1.0 + math::exp(-gv)));
        sync_threads();
    });
}

/// HC prompt norm straight to bf16: for group g (= row * hc + stream) of d values, inv = 1 / sqrt(mean(x^2) + eps),
/// y[g][i] = bf16(x[g][i] * inv * w[(g % ngw) * d + i]) (same order as fn_rmsnorm), rinv[g] = inv. grid = groups,
/// block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_hc_norm_bf16(y: *mut u16, rinv: *mut f32, x: *const f32, w: *const f32, d: u32,
    ngw: u32, eps: f32) {
    let red: Shared<f32> = kcommon::static_shared!(hn_red, f32, 128);
    let g = bid_x() as usize;
    let xr = x.add(g * d as usize);
    let yr = y.add(g * d as usize);
    let mut ss = 0.0f32;
    let mut i = tid_x();
    while i < d {
        let v = *xr.add(i as usize);
        ss += v * v;
        i += bdim_x();
    }
    let tot = warp::block_sum(ss, red);
    let inv = 1.0 / math::sqrt(tot / d as f32 + eps);
    if tid_x() == 0 {
        *rinv.add(g) = inv;
    }
    let wo = (g % ngw.max(1) as usize) * d as usize;
    let mut i = tid_x();
    while i < d {
        *yr.add(i as usize) = kcommon::bf16::from_f32(*xr.add(i as usize) * inv * *w.add(wo + i as usize));
        i += bdim_x();
    }
}

/// HC mix from the residual: mixed[r][i] = mean_s((res[r][s][i] * rinv[r * hc + s] * w[s * d + i]) * sigmoid(gate[r][s][i]))
/// (the normalised input recomputed as fx_hc_norm_bf16 / fn_rmsnorm do). n = rows * d, grid = ceil(n / 256), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_hc_mix_res(mixed: *mut f32, res: *const f32, rinv: *const f32, w: *const f32,
    gate: *const f32, d: u32, hc: u32, n: u32) {
    let idx = bid_x() * 256 + tid_x();
    if idx < n {
        let (r, i) = (idx / d, idx % d);
        let mut acc = 0.0f32;
        let mut s = 0;
        while s < hc {
            let o = ((r * hc + s) * d + i) as usize;
            let xn = *res.add(o) * *rinv.add((r * hc + s) as usize) * *w.add((s * d + i) as usize);
            let gv = *gate.add(o);
            acc += xn * (1.0 / (1.0 + math::exp(-gv)));
            s += 1;
        }
        *mixed.add(idx as usize) = acc / hc as f32;
    }
}

/// Prompt-group MoE combine from the i64 fixed-point expert sum of quant's `mmq_moe_down_acc`: for token t, column
/// i, y = f32(acc[t][i]) * 2^-40 (as `ggq_fx_finish`; acc re-zeroed), out = y + sh[t][i] * sigmoid(glog[t]),
/// res[t][s][i] += out * 2 sigmoid(inj[t * ldi + s]). grid = (ceil(d / 256), rows), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_moe_comb(res: *mut f32, acc: *mut i64, sh: *const f32, glog: *const f32,
    inj: *const f32, ldi: u32, d: u32, hc: u32) {
    let i = bid_x() * 256 + tid_x();
    let t = bid_y();
    if i >= d {
        return;
    }
    // every load first (the pointers may alias as far as the compiler knows: a load after a store waits), then the
    // stores; up to 4 streams
    let ai = (t * d + i) as usize;
    let a = *acc.add(ai);
    let shv = *sh.add(ai);
    let gl = *glog.add(t as usize);
    let mut rv = [0.0f32; 4];
    let mut iv = [0.0f32; 4];
    kcommon::unroll!(s in 4 => {
        if (s as u32) < hc {
            rv[s] = *res.add(((t * hc + s as u32) * d + i) as usize);
            iv[s] = *inj.add((t * ldi + s as u32) as usize);
        }
    });
    // i64 -> f32 through two 32-bit conversions: the high part (a >> 31) is exact below 2^55, one fma rounds the sum
    let hi = (a >> 31) as i32 as f32;
    let lo = (a & 0x7fff_ffff) as i32 as f32;
    let y = math::fma(hi, 2147483648.0, lo) * (1.0 / (1u64 << 40) as f32);
    let out = y + shv * (1.0 / (1.0 + math::exp(-gl)));
    kcommon::unroll!(s in 4 => {
        if (s as u32) < hc {
            let w = 2.0 / (1.0 + math::exp(-iv[s]));
            *res.add(((t * hc + s as u32) * d + i) as usize) = rv[s] + out * w;
        }
    });
    *acc.add(ai) = 0;
}

/// res[r][s][i] += y[r][i] for every stream s (a row term shared by the hc streams). n = rows * d, grid =
/// ceil(n / 256), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_bcast_add(res: *mut f32, y: *const f32, d: u32, hc: u32, n: u32) {
    let idx = bid_x() * 256 + tid_x();
    if idx < n {
        let (r, i) = (idx / d, idx % d);
        let v = *y.add(idx as usize);
        let mut s = 0;
        while s < hc {
            *res.add(((r * hc + s) * d + i) as usize) += v;
            s += 1;
        }
    }
}

/// Prompt GDN causal conv (kernel 4) + SiLU, as fn_gdn_conv but a thread per (channel, 16-row tile): out[t][c] =
/// silu(w0 x[t-3] + w1 x[t-2] + w2 x[t-1] + w3 x[t]), x[-k] from the carried state [3][C] (oldest first), which this
/// kernel only reads (fx_gdn_conv_state shifts it afterwards). grid = (ceil(C/256), ceil(rows/16)), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_gdn_conv_p(out: *mut f32, x: *const f32, ldx: u32, w: *const f32, state: *const f32,
    rows: u32, ch: u32) {
    let c = bid_x() * 256 + tid_x();
    let r0 = bid_y() * 16;
    if c >= ch || r0 >= rows {
        return;
    }
    let (w0, w1, w2, w3) = (*w.add(c as usize * 4), *w.add(c as usize * 4 + 1), *w.add(c as usize * 4 + 2), *w.add(c as usize * 4 + 3));
    // inputs r0 - 3 .. r0 + 16, all loaded before any store
    let mut v = [0.0f32; 19];
    kcommon::unroll!(k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] => {
        let r = r0 as i32 + k as i32 - 3;
        v[k] = if r < 0 {
            *state.add(((3 + r) as u32 * ch + c) as usize)
        } else if (r as u32) < rows {
            *x.add((r as u32 * ldx + c) as usize)
        } else {
            0.0
        };
    });
    kcommon::unroll!(k in 16 => {
        let t = r0 + k as u32;
        if t < rows {
            let y = w0 * v[k] + w1 * v[k + 1] + w2 * v[k + 2] + w3 * v[k + 3];
            *out.add((t * ch + c) as usize) = y / (1.0 + math::exp(-y));
        }
    });
}

/// The conv state after `rows` prompt rows: state[k] = x[rows - 3 + k] (or the old state shifted when rows < 3).
/// grid = ceil(C/256), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_gdn_conv_state(state: *mut f32, x: *const f32, ldx: u32, rows: u32, ch: u32) {
    let c = bid_x() * 256 + tid_x();
    if c >= ch {
        return;
    }
    let mut v = [0.0f32; 3];
    kcommon::unroll!(k in 3 => {
        let r = rows as i32 - 3 + k as i32;
        v[k] = if r >= 0 { *x.add((r as u32 * ldx + c) as usize) } else { *state.add(((3 + r) as u32 * ch + c) as usize) };
    });
    kcommon::unroll!(k in 3 => { *state.add((k as u32 * ch + c) as usize) = v[k]; });
}

/// HC combine fused into the next hc norm (prompt rows): for group g = row * hc + s of d values,
/// x = res + out[row] * 2 sigmoid(inj[row * ldi + s]) (written back to res, as fx_hc_combine), then as fx_hc_norm_bf16:
/// y = bf16(x * inv * w[(g % ngw) * d + i]), rinv[g] = inv (same summation order). grid = groups, block 256, d <= 4096.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_hc_comb_norm(y: *mut u16, rinv: *mut f32, res: *mut f32, out: *const f32,
    inj: *const f32, ldi: u32, w: *const f32, d: u32, ngw: u32, eps: f32) {
    let red: Shared<f32> = kcommon::static_shared!(hcn_red, f32, 128);
    let g = bid_x() as usize;
    let (row, s) = (g / ngw.max(1) as usize, g % ngw.max(1) as usize);
    let xr = res.add(g * d as usize);
    let orow = out.add(row * d as usize);
    let wv = 2.0 / (1.0 + math::exp(-*inj.add(row * ldi as usize + s)));
    let mut xv = [0.0f32; 16];
    let mut ss = 0.0f32;
    kcommon::unroll!(k in 16 => {
        let i = tid_x() + k as u32 * 256;
        if i < d {
            let x = *xr.add(i as usize) + *orow.add(i as usize) * wv;
            xv[k] = x;
            ss += x * x;
        }
    });
    let tot = warp::block_sum(ss, red);
    let inv = 1.0 / math::sqrt(tot / d as f32 + eps);
    if tid_x() == 0 {
        *rinv.add(g) = inv;
    }
    let wo = s * d as usize;
    let yr = y.add(g * d as usize);
    kcommon::unroll!(k in 16 => {
        let i = tid_x() + k as u32 * 256;
        if i < d {
            *xr.add(i as usize) = xv[k];
            *yr.add(i as usize) = kcommon::bf16::from_f32(xv[k] * inv * *w.add(wo + i as usize));
        }
    });
}

/// Two-stage argmax, stage 1: block b of row r takes logits b * n / P .. (b + 1) * n / P and writes
/// part[(r * P + b) * 3 ..] = (its max, its index (lowest on ties), sum exp(x - its max)). grid = (P, rows), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_argmax_part(part: *mut f32, x: *const f32, n: u32) {
    let sv: Shared<f32> = kcommon::static_shared!(am_v, f32, 32);
    let si: Shared<u32> = kcommon::static_shared!(am_i, u32, 32);
    let (b, p, r) = (bid_x(), gdim_x(), bid_y());
    let xr = x.add(r as usize * n as usize);
    let lo = (b as u64 * n as u64 / p as u64) as u32;
    let hi = ((b as u64 + 1) * n as u64 / p as u64) as u32;
    let mut best = f32::NEG_INFINITY;
    let mut bi = u32::MAX;
    let mut j = lo + tid_x();
    while j < hi {
        let v = *xr.add(j as usize);
        if v > best {
            best = v;
            bi = j;
        }
        j += bdim_x();
    }
    let mut m = 16;
    while m > 0 {
        let ov = warp::shfl_xor(best, m);
        let oi = warp::shfl_xor_u32(bi, m);
        if ov > best || (ov == best && oi < bi) {
            best = ov;
            bi = oi;
        }
        m >>= 1;
    }
    if lane() == 0 {
        sv.set(warp_id(), best);
        si.set(warp_id(), bi);
    }
    sync_threads();
    if warp_id() == 0 {
        let nw = nwarps();
        let l = lane();
        best = if l < nw { sv.at(l) } else { f32::NEG_INFINITY };
        bi = if l < nw { si.at(l) } else { u32::MAX };
        let mut m = 16;
        while m > 0 {
            let ov = warp::shfl_xor(best, m);
            let oi = warp::shfl_xor_u32(bi, m);
            if ov > best || (ov == best && oi < bi) {
                best = ov;
                bi = oi;
            }
            m >>= 1;
        }
        if l == 0 {
            sv.set(0, best);
            si.set(0, bi);
        }
    }
    sync_threads();
    let mx = sv.at(0);
    let mut acc = 0.0f32;
    let mut j = lo + tid_x();
    while j < hi {
        acc += math::exp(*xr.add(j as usize) - mx);
        j += bdim_x();
    }
    let red: Shared<f32> = kcommon::static_shared!(am_r, f32, 32);
    let tot = warp::block_sum(acc, red);
    if tid_x() == 0 {
        let o = part.add(((r * p + b) * 3) as usize);
        *o = mx;
        *o.add(1) = f32::from_bits(si.at(0));
        *o.add(2) = tot;
    }
}

/// Two-stage argmax, stage 2: out[r] = the index of the largest part max (lowest index on ties); pmax (optional)
/// = 1 / sum_b s_b exp(m_b - M). grid = rows, block 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_argmax_fin(out: *mut u32, part: *const f32, p: u32, pmax: *mut f32) {
    let r = bid_x();
    let l = lane();
    let mut best = f32::NEG_INFINITY;
    let mut bi = u32::MAX;
    let mut b = l;
    while b < p {
        let o = part.add(((r * p + b) * 3) as usize);
        let (v, i) = (*o, (*o.add(1)).to_bits());
        if v > best || (v == best && i < bi) {
            best = v;
            bi = i;
        }
        b += 32;
    }
    let mut m = 16;
    while m > 0 {
        let ov = warp::shfl_xor(best, m);
        let oi = warp::shfl_xor_u32(bi, m);
        if ov > best || (ov == best && oi < bi) {
            best = ov;
            bi = oi;
        }
        m >>= 1;
    }
    if l == 0 {
        *out.add(r as usize) = bi;
    }
    if pmax.is_null() {
        return;
    }
    let mut acc = 0.0f32;
    let mut b = l;
    while b < p {
        let o = part.add(((r * p + b) * 3) as usize);
        acc += *o.add(2) * math::exp(*o - best);
        b += 32;
    }
    let tot = warp::sum(acc);
    if l == 0 {
        *pmax.add(r as usize) = 1.0 / tot;
    }
}

/// fx_moe_comb from an f32 expert sum (fe-ggq moe_combine / moe_down_ordered output y [rows][d]):
/// out = y + sh * sigmoid(glog[t]), res[t][s][i] += out * 2 sigmoid(inj[t * ldi + s]). grid = (ceil(d/256), rows).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_moe_comb_f(res: *mut f32, y: *const f32, sh: *const f32, glog: *const f32,
    inj: *const f32, ldi: u32, d: u32, hc: u32) {
    let i = bid_x() * 256 + tid_x();
    let t = bid_y();
    if i >= d {
        return;
    }
    let ai = (t * d + i) as usize;
    let yv = *y.add(ai);
    let shv = *sh.add(ai);
    let gl = *glog.add(t as usize);
    let mut rv = [0.0f32; 4];
    let mut iv = [0.0f32; 4];
    kcommon::unroll!(s in 4 => {
        if (s as u32) < hc {
            rv[s] = *res.add(((t * hc + s as u32) * d + i) as usize);
            iv[s] = *inj.add((t * ldi + s as u32) as usize);
        }
    });
    let out = yv + shv * (1.0 / (1.0 + math::exp(-gl)));
    kcommon::unroll!(s in 4 => {
        if (s as u32) < hc {
            let w = 2.0 / (1.0 + math::exp(-iv[s]));
            *res.add(((t * hc + s as u32) * d + i) as usize) = rv[s] + out * w;
        }
    });
}

// ---- tensor parallel over two cards without peer access: all-reduce through pinned host memory ----

#[inline(always)]
unsafe fn ld_vol_u32(p: *const u32) -> u32 {
    let v: u32;
    core::arch::asm!("ld.volatile.global.u32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) p, options(nostack));
    v
}
#[inline(always)]
unsafe fn st_vol_u32(p: *mut u32, v: u32) {
    core::arch::asm!("st.volatile.global.u32 [{p}], {v};", p = in(reg64) p, v = in(reg32) v, options(nostack));
}
#[inline(always)]
unsafe fn ld_vol_v4(p: *const u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    core::arch::asm!("ld.volatile.global.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
        p = in(reg64) p, a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, options(nostack));
    [a, b, c, d]
}
#[inline(always)]
unsafe fn st_vol_v4(p: *mut u32, v: [u32; 4]) {
    core::arch::asm!("st.volatile.global.v4.u32 [{p}], {{{a}, {b}, {c}, {d}}};",
        p = in(reg64) p, a = in(reg32) v[0], b = in(reg32) v[1], c = in(reg32) v[2], d = in(reg32) v[3], options(nostack));
}

/// Two-card all-reduce chunk size (floats per block).
pub const AR_CH: u32 = 1024;

/// hm[i] = x[i] (device -> mapped pinned host, 16-byte stores), n a multiple of 4. grid = ceil(n / 1024), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_ar_put(hm: *mut f32, x: *const f32, n: u32) {
    let i = (bid_x() * 256 + tid_x()) * 4;
    if i < n {
        st_vol_v4(hm.add(i as usize) as *mut u32, ldg_nc_v4(x.add(i as usize) as *const u32));
    }
}

/// y[i] = x[i] + ho[i] (ho: the other card's partial in mapped pinned host memory). The sum of two terms is the same
/// bits in either order, so both cards get identical results. grid = ceil(n / 1024), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_ar_sum(y: *mut f32, x: *const f32, ho: *const f32, n: u32) {
    let i = (bid_x() * 256 + tid_x()) * 4;
    if i < n {
        let o = ld_vol_v4(ho.add(i as usize) as *const u32);
        let a = ldg_nc_v4(x.add(i as usize) as *const u32);
        let mut r = [0u32; 4];
        kcommon::unroll!(k in 4 => { r[k] = (f32::from_bits(a[k]) + f32::from_bits(o[k])).to_bits(); });
        stg_v4(y.add(i as usize) as *mut u32, r);
    }
}

/// The whole two-card all-reduce in one kernel: block b puts its AR_CH floats of x into hm, makes them visible
/// (fence.sys), raises fm[b] = tag, waits for fo[b] == tag (the other card's same chunk), then y = x + ho.
/// grid = ceil(n / AR_CH), block 256. tag: any nonzero value the flags do not hold yet (the step epoch).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_ar2(y: *mut f32, x: *const f32, hm: *mut f32, ho: *const f32, fm: *mut u32,
    fo: *const u32, n: u32, tag: *const u32) {
    let b = bid_x();
    let i = b * AR_CH + tid_x() * 4;
    let a = if i < n { ldg_nc_v4(x.add(i as usize) as *const u32) } else { [0; 4] };
    if i < n {
        st_vol_v4(hm.add(i as usize) as *mut u32, a);
    }
    core::arch::asm!("fence.sc.sys;");
    sync_threads();
    let t = *tag;
    if tid_x() == 0 {
        st_vol_u32(fm.add(b as usize), t);
        while ld_vol_u32(fo.add(b as usize)) != t {}
    }
    sync_threads();
    core::arch::asm!("fence.sc.sys;");
    if i < n {
        let o = ld_vol_v4(ho.add(i as usize) as *const u32);
        let mut r = [0u32; 4];
        kcommon::unroll!(k in 4 => { r[k] = (f32::from_bits(a[k]) + f32::from_bits(o[k])).to_bits(); });
        stg_v4(y.add(i as usize) as *mut u32, r);
    }
}

/// Two-card all-reduce, LL protocol (as NCCL's LL): every 4-byte value travels with a 4-byte tag in one 8-byte word
/// (8-byte stores and loads are single PCIe transactions), so no fence and no separate flag: the reader spins on its
/// own words until both tags of its 16-byte load match. hm / ho: [n] (value, tag) pairs (u32 [2n]) in mapped pinned
/// host memory. tag = *tagp + kofs (step epoch + the reduction's index; unique, so slots need no reset).
/// y[i] = x[i] + other[i] (two terms: the same bits on both cards). n even. grid = ceil(n / 512), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_ar_ll(y: *mut f32, x: *const f32, hm: *mut u32, ho: *const u32, n: u32,
    tagp: *const u32, kofs: u32) {
    let i = (bid_x() * 256 + tid_x()) * 2;
    if i >= n {
        return;
    }
    let t = (*tagp).wrapping_add(kofs);
    let a0 = *x.add(i as usize);
    let a1 = *x.add(i as usize + 1);
    st_vol_v4(hm.add(2 * i as usize), [a0.to_bits(), t, a1.to_bits(), t]);
    let p = ho.add(2 * i as usize);
    let mut v = ld_vol_v4(p);
    while v[1] != t || v[3] != t {
        v = ld_vol_v4(p);
    }
    *y.add(i as usize) = a0 + f32::from_bits(v[0]);
    *y.add(i as usize + 1) = a1 + f32::from_bits(v[2]);
}

/// Flag ping-pong only (latency floor of a cross-card handshake through host memory): thread 0 raises fm = tag + kofs
/// and spins until fo holds it. grid 1, block 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_ar_ping(fm: *mut u32, fo: *const u32, tagp: *const u32, kofs: u32) {
    if tid_x() == 0 {
        let t = (*tagp).wrapping_add(kofs);
        st_vol_u32(fm, t);
        while ld_vol_u32(fo) != t {}
    }
}

/// arbench probes: mode 0 st.volatile.v4 x -> h, 1 st.global.v4 x -> h, 2 ld.volatile.v4 h -> y, 3 ld.global.nc.v4 h -> y,
/// 4 ld.global.cg h -> y (n floats, grid = ceil(n / 1024), block 256).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_ar_probe(y: *mut f32, x: *const f32, h: *mut f32, n: u32, mode: u32) {
    let i = (bid_x() * 256 + tid_x()) * 4;
    if i >= n {
        return;
    }
    let hp = h.add(i as usize) as *mut u32;
    match mode {
        0 => st_vol_v4(hp, ldg_nc_v4(x.add(i as usize) as *const u32)),
        1 => stg_v4(hp, ldg_nc_v4(x.add(i as usize) as *const u32)),
        2 => stg_v4(y.add(i as usize) as *mut u32, ld_vol_v4(hp)),
        3 => stg_v4(y.add(i as usize) as *mut u32, ldg_nc_v4(hp)),
        _ => {
            let (a, b, c, d): (u32, u32, u32, u32);
            core::arch::asm!("ld.global.cg.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
                p = in(reg64) hp, a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, options(nostack));
            stg_v4(y.add(i as usize) as *mut u32, [a, b, c, d]);
        }
    }
}

/// y = x + ho once the other card's flag fo[0] holds `tag` (thread 0 of each block spins on it; the copy that filled ho
/// was stream-ordered before the flag write). grid = ceil(n / 1024), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_ar_sumw(y: *mut f32, x: *const f32, ho: *const f32, n: u32, fo: *const u32, tag: u32) {
    let shared_ok: Shared<u32> = kcommon::static_shared!(arw_ok, u32, 1);
    if tid_x() == 0 {
        while ld_vol_u32(fo) != tag {}
        shared_ok.set(0, 1);
    }
    sync_threads();
    let i = (bid_x() * 256 + tid_x()) * 4;
    if i < n {
        let o = ld_vol_v4(ho.add(i as usize) as *const u32);
        let a = ldg_nc_v4(x.add(i as usize) as *const u32);
        let mut r = [0u32; 4];
        kcommon::unroll!(k in 4 => { r[k] = (f32::from_bits(a[k]) + f32::from_bits(o[k])).to_bits(); });
        stg_v4(y.add(i as usize) as *mut u32, r);
    }
}

/// LL pack on the card: p[2i] = x[i] bits, p[2i + 1] = tag (*tagp + kofs). grid = ceil(n / 256), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_ll_pack(p: *mut u32, x: *const f32, n: u32, tagp: *const u32, kofs: u32) {
    let i = bid_x() * 256 + tid_x();
    if i < n {
        let t = (*tagp).wrapping_add(kofs);
        *(p.add(2 * i as usize) as *mut [u32; 2]) = [(*x.add(i as usize)).to_bits(), t];
    }
}

/// LL receive: y[i] = x[i] + value of ho pair i, spinning until the pair's tag is *tagp + kofs. grid = ceil(n / 512),
/// block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_ll_sum(y: *mut f32, x: *const f32, ho: *const u32, n: u32, tagp: *const u32, kofs: u32) {
    let i = (bid_x() * 256 + tid_x()) * 2;
    if i >= n {
        return;
    }
    let t = (*tagp).wrapping_add(kofs);
    let p = ho.add(2 * i as usize);
    let mut v = ld_vol_v4(p);
    while v[1] != t || v[3] != t {
        v = ld_vol_v4(p);
    }
    *y.add(i as usize) = *x.add(i as usize) + f32::from_bits(v[0]);
    *y.add(i as usize + 1) = *x.add(i as usize + 1) + f32::from_bits(v[2]);
}

// ---- FNEXT_TP=2 decode: the two cards reduce their partial block outputs through pinned host memory ----

#[inline(always)]
unsafe fn ld_vol_v2(p: *const u32) -> [u32; 2] {
    let (a, b): (u32, u32);
    core::arch::asm!("ld.volatile.global.v2.u32 {{{a}, {b}}}, [{p}];", p = in(reg64) p, a = out(reg32) a, b = out(reg32) b,
        options(nostack));
    [a, b]
}

/// Spin on an LL (value, tag) pair until its tag is `tag`; traps after ~2^28 polls (the other card never came: a
/// failed launch must not hang the process silently).
#[inline(always)]
unsafe fn ll_wait(p: *const u32, tag: u32) -> f32 {
    let mut v = ld_vol_v2(p);
    let mut n = 0u32;
    while v[1] != tag {
        n += 1;
        if n > (1 << 28) {
            core::arch::asm!("trap;");
        }
        v = ld_vol_v2(p);
    }
    f32::from_bits(v[0])
}

/// out[t][i] = out[t][i] + other (the other card's partial: LL pairs at ho [rows][d][2] in pinned host memory, tag
/// prm[P_EPOCH] + kofs), then the hyper-connection combine res[t][s][i] += out * 2 sigmoid(inj[t * ldi + s]).
/// Two terms: both cards get the same bits. grid = (ceil(d / 256), rows), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_ar_comb(res: *mut f32, out: *mut f32, ho: *const u32, inj: *const f32, ldi: u32, d: u32,
    hc: u32, prm: *const u32, kofs: u32) {
    // let the next (PDL-launched) kernel start now: it prefetches its weights while this one runs / waits, and does
    // griddepcontrol.wait before reading anything written here
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    let t = bid_y();
    let i = bid_x() * 256 + tid_x();
    if i >= d {
        return;
    }
    let idx = (t * d + i) as usize;
    let tag = (*prm.add(P_EPOCH)).wrapping_add(kofs);
    let o = *out.add(idx) + ll_wait(ho.add(2 * idx), tag);
    *out.add(idx) = o;
    let mut s = 0;
    while s < hc {
        let w = 2.0 / (1.0 + math::exp(-*inj.add((t * ldi + s) as usize)));
        *res.add(((t * hc + s) * d + i) as usize) += o * w;
        s += 1;
    }
}

/// fx_moe_fin for one card of FNEXT_TP=2: the partial FFN output of the experts this card owns (own[e] != 0; on the
/// card when dtab[e] != 0, else the CPU row in host) in pair order, plus this card's half of the shared expert,
/// out[t][i] = moe + sh * sigmoid(glog), also packed as LL pairs (tag prm[P_EPOCH] + kofs) into llp for the copy to the
/// other card. A row with an owned miss first waits until the CPU thread has set `done` to this step's epoch
/// (prm[P_EPOCH]: no reset, no flag kernel). grid = (ceil(d / 256), rows), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_moe_fin_tp(out: *mut f32, llp: *mut u32, ey: *const f32, host: *const f32,
    ids: *const i32, dtab: *const u64, own: *const u32, wts: *const f32, sh: *const f32, glog: *const f32, ldg: u32, d: u32,
    topk: u32, done: *const u32, prm: *const u32, kofs: u32) {
    let t = bid_y();
    let i = bid_x() * 256 + tid_x();
    if tid_x() == 0 {
        let mut miss = false;
        let mut k = 0;
        while k < topk {
            let e = *ids.add((t * topk + k) as usize) as usize;
            miss |= *own.add(e) != 0 && *dtab.add(e) == 0;
            k += 1;
        }
        if miss {
            let ep = *prm.add(P_EPOCH);
            let mut n = 0u32;
            while ld_vol_u32(done) != ep {
                n += 1;
                if n > (1 << 28) {
                    core::arch::asm!("trap;");
                }
            }
            core::arch::asm!("fence.sc.sys;");
        }
    }
    sync_threads();
    if i >= d {
        return;
    }
    let mut v = [0.0f32; 16];
    let mut m = [false; 16];
    kcommon::unroll!(k in 16 => {
        if (k as u32) < topk {
            let p = t * topk + k as u32;
            let e = *ids.add(p as usize);
            m[k] = *own.add(e as usize) != 0;
            if m[k] {
                v[k] = if *dtab.add(e as usize) != 0 { *ey.add((p * d + i) as usize) } else {
                    core::ptr::read_volatile(host.add((p * d + i) as usize))
                };
            }
        }
    });
    let mut moe = 0.0f32;
    kcommon::unroll!(k in 16 => {
        if (k as u32) < topk && m[k] {
            moe += *wts.add((t * topk + k as u32) as usize) * v[k];
        }
    });
    let g = 1.0 / (1.0 + math::exp(-*glog.add((t * ldg) as usize)));
    let o = moe + *sh.add((t * d + i) as usize) * g;
    let idx = (t * d + i) as usize;
    *out.add(idx) = o;
    let tag = (*prm.add(P_EPOCH)).wrapping_add(kofs);
    *(llp.add(2 * idx) as *mut [u32; 2]) = [o.to_bits(), tag];
}

/// fx_hit_flag for one card of FNEXT_TP=2: raise the flag when no pair's expert is owned here (own[e] != 0) and off
/// the card (tab[e] == 0). grid = 1, block 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_hit_flag_tp(flag: *mut u32, ids: *const i32, tab: *const u64, own: *const u32, np: u32) {
    let l = lane();
    let mut miss = 0u32;
    let mut p = l;
    while p < np {
        let e = *ids.add(p as usize) as usize;
        if *own.add(e) != 0 && *tab.add(e) == 0 {
            miss = 1;
        }
        p += 32;
    }
    let any: u32;
    core::arch::asm!("{{ .reg .pred p; setp.ne.u32 p, {x}, 0; vote.sync.any.pred p, p, 0xffffffff; selp.u32 {v}, 1, 0, p; }}",
        x = in(reg32) miss, v = out(reg32) any, options(nostack));
    if any == 0 && l == 0 {
        core::ptr::write_volatile(flag, 1);
        core::arch::asm!("membar.sys;");
    }
}

/// FNEXT_TP=2 vocab-split head: per row r < rows, tv[3r] = tok[r] + off (the index in the whole vocabulary),
/// tv[3r + 1] = the logit there (logits rows of n), tv[3r + 2] = pm[r] (this half's softmax probability of it; 0 when
/// pm is null). grid = 1, block 32.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_argmax_tv(tv: *mut u32, tok: *const u32, logits: *const f32, n: u32, off: u32, rows: u32,
    pm: *const f32) {
    let mut r = tid_x();
    while r < rows {
        let t = *tok.add(r as usize);
        *tv.add(3 * r as usize) = t + off;
        *tv.add(3 * r as usize + 1) = (*logits.add((r * n + t) as usize)).to_bits();
        *tv.add(3 * r as usize + 2) = if pm.is_null() { 0 } else { (*pm.add(r as usize)).to_bits() };
        r += 32;
    }
}

/// y[i] = the values of LL pairs at ho (spinning until each pair's tag is prm[P_EPOCH] + kofs). grid ceil(n/256).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_ll_get(y: *mut f32, ho: *const u32, n: u32, prm: *const u32, kofs: u32) {
    let i = bid_x() * 256 + tid_x();
    if i < n {
        let tag = (*prm.add(P_EPOCH)).wrapping_add(kofs);
        *y.add(i as usize) = ll_wait(ho.add(2 * i as usize), tag);
    }
}

/// FNEXT_TP=2 prompt FFN partial: out[t][i] = y[t][i] + sh[t][i] * sigmoid(glog[t]) (fx_moe_comb_f's out, before the
/// all-reduce and the combine). grid = (ceil(d/256), rows), block 256.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_moe_part(out: *mut f32, y: *const f32, sh: *const f32, glog: *const f32, d: u32) {
    let i = bid_x() * 256 + tid_x();
    let t = bid_y();
    if i >= d {
        return;
    }
    let ai = (t * d + i) as usize;
    let gl = *glog.add(t as usize);
    *out.add(ai) = *y.add(ai) + *sh.add(ai) * (1.0 / (1.0 + math::exp(-gl)));
}

/// FNEXT_TP=2 prompt MoE: wts[p] = 0 for the pairs whose expert another card owns (own[e] == 0). grid ceil(n/256).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn fx_mask_own(wts: *mut f32, ids: *const i32, own: *const u32, n: u32) {
    let p = bid_x() * 256 + tid_x();
    if p < n && *own.add(*ids.add(p as usize) as usize) == 0 {
        *wts.add(p as usize) = 0.0;
    }
}
