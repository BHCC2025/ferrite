//! Sparse MQA attention over one shared 512-wide latent (key = value): the sliding window from the ring plus the
//! selected compressed entries, a per-head sink, then the inverse RoPE of the output.
use kcommon::smem::Shared;
use kcommon::*;
use kcommon::mma;

const D: usize = 512;

/// One (head, row) a block: keys = ring positions max(0, pos - window + 1) ..= pos of the row's slot, then
/// plane entries sel[r][0..nsel] (>= 0) of the slot (when `plane` is non-null). score = q . k * scale (f32),
/// softmax with the sink in the denominator, o = sum p v / den, rounded to bf16; then the last 64 dims are
/// rotated back (inverse RoPE at pos) and rounded again. Output: o[(h / 8) * o_gstride + r * 4096 +
/// (h % 8) * 512 + d] (the wo_a groups' inputs, contiguous per group). With `use_floor`, window positions below
/// floor[slot] are left out (bounded replay: the decoder layers never saw those rows).
/// grid = (heads, rows), block = 128 (4 warps).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_attn(o: *mut u16, o_gstride: u32, q: *const u16, q_ld: u32, ring: *const u16,
    ring_stride: u32, ring_len: u32, window: u32, plane: *const u16, plane_stride: u32, sel: *const i32, nsel: u32,
    meta: *const u32, sink: *const f32, scale: f32, tab: *const f32, floor: *const u32, use_floor: u32) {
    let sm: Shared<f32> = kcommon::static_shared!(at_m, f32, 32);
    let sl: Shared<f32> = kcommon::static_shared!(at_l, f32, 32);
    let sacc: Shared<f32> = kcommon::static_shared!(at_acc, f32, 8192); // [4][512]
    let h = bid_x() as usize;
    let r = bid_y() as usize;
    let w = warp_id() as usize;
    let l = lane() as usize;
    let slot = *meta.add(2 * r) as usize;
    let pos = *meta.add(2 * r + 1);
    // q: lane owns dims l*16 .. l*16+16
    let qp = q.add(r * q_ld as usize + h * D + l * 16) as *const u32;
    let q0 = bf16::unpack8(ldg_nc_v4(qp));
    let q1 = bf16::unpack8(ldg_nc_v4(qp.add(4)));
    let mut qv = [0.0f32; 16];
    let mut i = 0;
    while i < 8 {
        qv[i] = q0[i];
        qv[8 + i] = q1[i];
        i += 1;
    }
    let mut first = if pos + 1 > window { pos + 1 - window } else { 0 };
    if use_floor != 0 {
        first = first.max((*floor.add(slot)).min(pos)); // a replay's decoder layers: no window key below its cut
    }
    let nwin = (pos + 1 - first) as usize;
    let ntot = nwin + if plane.is_null() { 0 } else { nsel as usize };
    let ringb = ring.add(slot * ring_stride as usize);
    let mut m = f32::NEG_INFINITY;
    let mut den = 0.0f32;
    let mut acc = [0.0f32; 16];
    let mut k = w;
    while k < ntot {
        let kp: *const u16 = if k < nwin {
            let p = first + k as u32;
            ringb.add((p % ring_len) as usize * D)
        } else {
            let e = *sel.add(r * nsel as usize + (k - nwin));
            if e < 0 {
                k += 4;
                continue;
            }
            plane.add(slot * plane_stride as usize + e as usize * D)
        };
        let kp = kp.add(l * 16) as *const u32;
        let k0 = bf16::unpack8(ldg_nc_v4(kp));
        let k1 = bf16::unpack8(ldg_nc_v4(kp.add(4)));
        let mut kv = [0.0f32; 16];
        let mut i = 0;
        while i < 8 {
            kv[i] = k0[i];
            kv[8 + i] = k1[i];
            i += 1;
        }
        let mut dot = 0.0f32;
        let mut i = 0;
        while i < 16 {
            dot = math::fma(qv[i], kv[i], dot);
            i += 1;
        }
        let s = warp::sum(dot) * scale;
        let mn = m.max(s);
        let corr = if m == f32::NEG_INFINITY { 0.0 } else { math::exp(m - mn) };
        let p = math::exp(s - mn);
        den = den * corr + p;
        let mut i = 0;
        while i < 16 {
            acc[i] = acc[i] * corr + p * kv[i];
            i += 1;
        }
        m = mn;
        k += 4;
    }
    if l == 0 {
        sm.set(w as u32, m);
        sl.set(w as u32, den);
    }
    let mut i = 0;
    while i < 16 {
        sacc.set((w * D + l * 16 + i) as u32, acc[i]);
        i += 1;
    }
    sync_threads();
    let sk = *sink.add(h);
    let mut mm = sk;
    let mut j = 0;
    while j < 4 {
        mm = mm.max(sm.at(j));
        j += 1;
    }
    let mut cw = [0.0f32; 4];
    let mut tden = math::exp(sk - mm);
    let mut j = 0;
    while j < 4 {
        let mj = sm.at(j as u32);
        cw[j] = if mj == f32::NEG_INFINITY { 0.0 } else { math::exp(mj - mm) };
        tden += sl.at(j as u32) * cw[j];
        j += 1;
    }
    // thread t: dims 4t .. 4t+4
    let t = tid_x() as usize;
    let mut ov = [0.0f32; 4];
    let mut c = 0;
    while c < 4 {
        let d = 4 * t + c;
        let v = sacc.at(d as u32) * cw[0] + sacc.at((D + d) as u32) * cw[1] + sacc.at((2 * D + d) as u32) * cw[2]
            + sacc.at((3 * D + d) as u32) * cw[3];
        ov[c] = bf16::round(v / tden);
        c += 1;
    }
    if 4 * t >= D - 64 {
        let mut c = 0;
        while c < 4 {
            let pi = (4 * t + c - (D - 64)) / 2;
            let tb = tab.add(pos as usize * 64 + 2 * pi);
            let (cs, sn) = (*tb, -*tb.add(1));
            let (a, b) = (ov[c], ov[c + 1]);
            ov[c] = a * cs - b * sn;
            ov[c + 1] = a * sn + b * cs;
            c += 2;
        }
    }
    let dst = o.add((h / 8) * o_gstride as usize + r * 4096 + (h % 8) * D + 4 * t) as *mut u32;
    *dst = bf16::pack2(ov[0], ov[1]);
    *dst.add(1) = bf16::pack2(ov[2], ov[3]);
}

/// ds_attn with 16 warps a block and two keys a warp in flight (the decode / verify form: one (head, row) block has
/// ~640 keys at most, and the single-key loop of ds_attn waits on every key's load). Same contract as ds_attn; the
/// softmax is split over 16 warps (key k to warp k % 16), so the bits differ from ds_attn's.
/// grid = (heads, rows), block = 512.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_attn16(o: *mut u16, o_gstride: u32, q: *const u16, q_ld: u32, ring: *const u16,
    ring_stride: u32, ring_len: u32, window: u32, plane: *const u16, plane_stride: u32, sel: *const i32, nsel: u32,
    meta: *const u32, sink: *const f32, scale: f32, tab: *const f32, floor: *const u32, use_floor: u32) {
    const NW: usize = 16;
    let sm: Shared<f32> = kcommon::static_shared!(a16_m, f32, 128);
    let sl: Shared<f32> = kcommon::static_shared!(a16_l, f32, 128);
    let sacc: Shared<f32> = kcommon::static_shared!(a16_acc, f32, 32768); // [16][512] f32 (bytes)
    let h = bid_x() as usize;
    let r = bid_y() as usize;
    let w = warp_id() as usize;
    let l = lane() as usize;
    let slot = *meta.add(2 * r) as usize;
    let pos = *meta.add(2 * r + 1);
    // q and the keys stay packed (8 bf16-pair words a lane) and unpack at use: fewer registers (two blocks an SM,
    // launch bounds from build.rs), the same values and fma order
    let qp = q.add(r * q_ld as usize + h * D + l * 16) as *const u32;
    let qa = ldg_nc_v4(qp);
    let qb = ldg_nc_v4(qp.add(4));
    let qw = [qa[0], qa[1], qa[2], qa[3], qb[0], qb[1], qb[2], qb[3]];
    let mut first = if pos + 1 > window { pos + 1 - window } else { 0 };
    if use_floor != 0 {
        first = first.max((*floor.add(slot)).min(pos));
    }
    let nwin = (pos + 1 - first) as usize;
    let ntot = nwin + if plane.is_null() { 0 } else { nsel as usize };
    let ringb = ring.add(slot * ring_stride as usize);
    // key k's row (null: a padded selection entry)
    let key = |k: usize| -> *const u16 {
        if k < nwin {
            ringb.add(((first + k as u32) % ring_len) as usize * D)
        } else {
            let e = *sel.add(r * nsel as usize + (k - nwin));
            if e < 0 { core::ptr::null() } else { plane.add(slot * plane_stride as usize + e as usize * D) }
        }
    };
    let load = |kp: *const u16| -> [u32; 8] {
        if kp.is_null() {
            return [0u32; 8];
        }
        let p = kp.add(l * 16) as *const u32;
        let a = ldg_nc_v4(p);
        let b = ldg_nc_v4(p.add(4));
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    };
    let mut m = f32::NEG_INFINITY;
    let mut den = 0.0f32;
    let mut acc = [0.0f32; 16];
    let mut k = w;
    while k < ntot {
        let ka = key(k);
        let kb = if k + NW < ntot { key(k + NW) } else { core::ptr::null() };
        let wa = load(ka);
        let wb = load(kb);
        if !ka.is_null() {
            attn_step(&mut m, &mut den, &mut acc, &qw, &wa, scale);
        }
        if !kb.is_null() {
            attn_step(&mut m, &mut den, &mut acc, &qw, &wb, scale);
        }
        k += 2 * NW;
    }
    attn_combine(sm, sl, sacc, w, l, m, den, &acc, *sink.add(h), tab, pos,
        o.add((h / 8) * o_gstride as usize + r * 4096 + (h % 8) * D));
}


/// One key of ds_attn16's online softmax on packed words (element 2j = lo(w[j]), 2j + 1 = hi(w[j])): the dot in
/// element order, then the running max / denominator / accumulator update (ds_attn's arithmetic).
#[inline(always)]
fn attn_step(m: &mut f32, den: &mut f32, acc: &mut [f32; 16], qw: &[u32; 8], kw: &[u32; 8], scale: f32) {
    let mut dot = 0.0f32;
    let mut j = 0;
    while j < 8 {
        dot = math::fma(bf16::lo(qw[j]), bf16::lo(kw[j]), dot);
        dot = math::fma(bf16::hi(qw[j]), bf16::hi(kw[j]), dot);
        j += 1;
    }
    let s = warp::sum(dot) * scale;
    let mn = m.max(s);
    let corr = if *m == f32::NEG_INFINITY { 0.0 } else { math::exp(*m - mn) };
    let p = math::exp(s - mn);
    *den = *den * corr + p;
    let mut j = 0;
    while j < 8 {
        acc[2 * j] = acc[2 * j] * corr + p * bf16::lo(kw[j]);
        acc[2 * j + 1] = acc[2 * j + 1] * corr + p * bf16::hi(kw[j]);
        j += 1;
    }
    *m = mn;
}

/// The end of ds_attn16 for one (head, row): every warp's state to shared memory, then threads 0..127 merge the 16
/// warps (with the sink), round, rotate the last 64 dims back and write 4 dims each to `dst` (the head's 512).
/// Contains barriers (all threads of the block must call it).
#[inline(always)]
unsafe fn attn_combine(sm: Shared<f32>, sl: Shared<f32>, sacc: Shared<f32>, w: usize, l: usize, m: f32, den: f32,
    acc: &[f32; 16], sk: f32, tab: *const f32, pos: u32, dst: *mut u16) {
    const NW: usize = 16;
    if l == 0 {
        sm.set(w as u32, m);
        sl.set(w as u32, den);
    }
    let mut i = 0;
    while i < 16 {
        sacc.set((w * D + l * 16 + i) as u32, acc[i]);
        i += 1;
    }
    sync_threads();
    let t = tid_x() as usize;
    if t < 128 {
        let mut mm = sk;
        let mut j = 0;
        while j < NW {
            mm = mm.max(sm.at(j as u32));
            j += 1;
        }
        let mut cw = [0.0f32; NW];
        let mut tden = math::exp(sk - mm);
        let mut j = 0;
        while j < NW {
            let mj = sm.at(j as u32);
            cw[j] = if mj == f32::NEG_INFINITY { 0.0 } else { math::exp(mj - mm) };
            tden += sl.at(j as u32) * cw[j];
            j += 1;
        }
        let mut ov = [0.0f32; 4];
        let mut c = 0;
        while c < 4 {
            let d = 4 * t + c;
            let mut v = 0.0f32;
            let mut j = 0;
            while j < NW {
                v += sacc.at((j * D + d) as u32) * cw[j];
                j += 1;
            }
            ov[c] = bf16::round(v / tden);
            c += 1;
        }
        if 4 * t >= D - 64 {
            let mut c = 0;
            while c < 4 {
                let pi = (4 * t + c - (D - 64)) / 2;
                let tb = tab.add(pos as usize * 64 + 2 * pi);
                let (cs, sn) = (*tb, -*tb.add(1));
                let (a, b) = (ov[c], ov[c + 1]);
                ov[c] = a * cs - b * sn;
                ov[c + 1] = a * sn + b * cs;
                c += 2;
            }
        }
        let d = dst.add(4 * t) as *mut u32;
        *d = bf16::pack2(ov[0], ov[1]);
        *d.add(1) = bf16::pack2(ov[2], ov[3]);
    }
    sync_threads();
}

/// ds_attn16 for two heads a block (decode / verify forwards of several rows: half the blocks, each key loaded once
/// for both heads). Each head's keys, order and merge are ds_attn16's, so the bits are the same.
/// grid = (heads / 2, rows), block = 512.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_attn16x2(o: *mut u16, o_gstride: u32, q: *const u16, q_ld: u32, ring: *const u16,
    ring_stride: u32, ring_len: u32, window: u32, plane: *const u16, plane_stride: u32, sel: *const i32, nsel: u32,
    meta: *const u32, sink: *const f32, scale: f32, tab: *const f32, floor: *const u32, use_floor: u32) {
    const NW: usize = 16;
    let sm: Shared<f32> = kcommon::static_shared!(a16_m, f32, 128);
    let sl: Shared<f32> = kcommon::static_shared!(a16_l, f32, 128);
    let sacc: Shared<f32> = kcommon::static_shared!(a16_acc, f32, 32768);
    let h0 = 2 * bid_x() as usize;
    let r = bid_y() as usize;
    let w = warp_id() as usize;
    let l = lane() as usize;
    let slot = *meta.add(2 * r) as usize;
    let pos = *meta.add(2 * r + 1);
    let qword = |h: usize| -> [u32; 8] {
        let qp = q.add(r * q_ld as usize + h * D + l * 16) as *const u32;
        let a = ldg_nc_v4(qp);
        let b = ldg_nc_v4(qp.add(4));
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    };
    let qw0 = qword(h0);
    let qw1 = qword(h0 + 1);
    let mut first = if pos + 1 > window { pos + 1 - window } else { 0 };
    if use_floor != 0 {
        first = first.max((*floor.add(slot)).min(pos));
    }
    let nwin = (pos + 1 - first) as usize;
    let ntot = nwin + if plane.is_null() { 0 } else { nsel as usize };
    let ringb = ring.add(slot * ring_stride as usize);
    let mut m0 = f32::NEG_INFINITY;
    let mut d0 = 0.0f32;
    let mut a0 = [0.0f32; 16];
    let mut m1 = f32::NEG_INFINITY;
    let mut d1 = 0.0f32;
    let mut a1 = [0.0f32; 16];
    let mut k = w;
    while k < ntot {
        let kp: *const u16 = if k < nwin {
            ringb.add(((first + k as u32) % ring_len) as usize * D)
        } else {
            let e = *sel.add(r * nsel as usize + (k - nwin));
            if e < 0 { core::ptr::null() } else { plane.add(slot * plane_stride as usize + e as usize * D) }
        };
        if !kp.is_null() {
            let p = kp.add(l * 16) as *const u32;
            let a = ldg_nc_v4(p);
            let b = ldg_nc_v4(p.add(4));
            let kw = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]];
            attn_step(&mut m0, &mut d0, &mut a0, &qw0, &kw, scale);
            attn_step(&mut m1, &mut d1, &mut a1, &qw1, &kw, scale);
        }
        k += NW;
    }
    let dst = |h: usize| o.add((h / 8) * o_gstride as usize + r * 4096 + (h % 8) * D);
    attn_combine(sm, sl, sacc, w, l, m0, d0, &a0, *sink.add(h0), tab, pos, dst(h0));
    attn_combine(sm, sl, sacc, w, l, m1, d1, &a1, *sink.add(h0 + 1), tab, pos, dst(h0 + 1));
}

/// Key splits of a row in ds_attn_mq (a row's partition depends only on its own key count).
pub const MQ_SPLITS: usize = 8;
const MQ_KT: usize = 16;
/// padded bf16 row of a key tile (1040 bytes: conflict-free fragment loads and ldmatrix rows)
const MQ_KROW: usize = 520;

/// `unroll!(I in [0, 1, ..] { body })`: the body once per listed constant (register arrays stay in registers).
macro_rules! unroll {
    ($i:ident in [$($n:literal),*] $body:block) => { $( { #[allow(non_upper_case_globals)] const $i: usize = $n; $body } )* };
}

/// Online-softmax update of one head row from its 4 scores in this thread (the quad holds the row's 16): returns
/// the probabilities and the rescale of the old state.
#[inline(always)]
fn mq_row(sv: [f32; 4], m: &mut f32, lsum: &mut f32) -> ([f32; 4], f32) {
    let mut mx = sv[0].max(sv[1]).max(sv[2]).max(sv[3]);
    mx = mx.max(warp::shfl_xor(mx, 1));
    mx = mx.max(warp::shfl_xor(mx, 2));
    let mn = m.max(mx);
    let corr = if *m == f32::NEG_INFINITY { 0.0 } else { math::exp(*m - mn) };
    let e = |x: f32| if x == f32::NEG_INFINITY { 0.0 } else { math::exp(x - mn) };
    let p = [e(sv[0]), e(sv[1]), e(sv[2]), e(sv[3])];
    let mut sum = p[0] + p[1] + p[2] + p[3];
    sum += warp::shfl_xor(sum, 1);
    sum += warp::shfl_xor(sum, 2);
    *lsum = *lsum * corr + sum;
    *m = mn;
    (p, corr)
}

/// Tensor-core MQA attention for decode / verify rows: block (s, r) takes key split s of row r (the window keys,
/// then the selected plane entries, cut in MQ_SPLITS equal parts of whole 16-key tiles) for all the rank's 32 heads
/// at once (the heads share the 512-wide latent: each key tile is read once). 16 warps: warp w takes heads
/// 16 (w / 8).. and dims 64 (w % 8)..; S = Q K^T as eight 64-dim partial sums added in a fixed order through shared
/// memory, an online softmax per head, O += P V with P split in two bf16 terms (hi + lo: ~16-bit P). Writes
/// unnormalised partials: part_o[r][s][h][512] f32 and part_ml[r][s][h] = (max, sum); ds_attn_mq_fin merges them.
/// grid = (MQ_SPLITS, rows), block = 512.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_attn_mq(part_o: *mut f32, part_ml: *mut f32, q: *const u16, q_ld: u32,
    ring: *const u16, ring_stride: u32, ring_len: u32, window: u32, plane: *const u16, plane_stride: u32, sel: *const i32,
    nsel: u32, meta: *const u32, scale: f32, floor: *const u32, use_floor: u32) {
    let sk: Shared<u16> = kcommon::static_shared!(mq_k, u16, 16640); // [16][520] bf16
    let svd: Shared<u32> = kcommon::static_shared!(mq_valid, u32, 64); // [16]
    let sp: Shared<f32> = kcommon::static_shared!(mq_s, f32, 16384); // [16 warps][16][16] f32
    let sidx = bid_x() as usize;
    let r = bid_y() as usize;
    let tid = tid_x() as usize;
    let w = warp_id() as usize;
    let l = lane() as usize;
    let (g, t) = (l / 4, l % 4);
    let (mt, dq) = (w >> 3, w & 7);
    let slot = *meta.add(2 * r) as usize;
    let pos = *meta.add(2 * r + 1);
    let mut first = if pos + 1 > window { pos + 1 - window } else { 0 };
    if use_floor != 0 {
        first = first.max((*floor.add(slot)).min(pos));
    }
    let nwin = (pos + 1 - first) as usize;
    let ntot = nwin + if plane.is_null() { 0 } else { nsel as usize };
    let chunk = ntot.div_ceil(MQ_SPLITS).div_ceil(MQ_KT) * MQ_KT;
    let k_lo = (sidx * chunk).min(ntot);
    let k_hi = (k_lo + chunk).min(ntot);
    let ringb = ring.add(slot * ring_stride as usize);
    let key = |k: usize| -> *const u16 {
        if k < nwin {
            ringb.add(((first + k as u32) % ring_len) as usize * D)
        } else {
            let e = *sel.add(r * nsel as usize + (k - nwin));
            if e < 0 { core::ptr::null() } else { plane.add(slot * plane_stride as usize + e as usize * D) }
        }
    };
    // Q fragments of this warp's heads and 64 dims
    let qrow = q.add(r * q_ld as usize);
    let (h0, h1) = (16 * mt + g, 16 * mt + g + 8);
    let mut qa = [[0u32; 4]; 4];
    unroll!(KS in [0, 1, 2, 3] {
        let d0 = 64 * dq + 16 * KS + 2 * t;
        qa[KS] = [*(qrow.add(h0 * D + d0) as *const u32), *(qrow.add(h1 * D + d0) as *const u32),
            *(qrow.add(h0 * D + d0 + 8) as *const u32), *(qrow.add(h1 * D + d0 + 8) as *const u32)];
    });
    let mut m0 = f32::NEG_INFINITY;
    let mut m1 = f32::NEG_INFINITY;
    let mut l0 = 0.0f32;
    let mut l1 = 0.0f32;
    let mut acc = [[0.0f32; 4]; 8];
    let krow_v = ((l & 7) + ((l >> 3) & 1) * 8) * MQ_KROW + 64 * dq + 8 * (l >> 4);
    let mut k0 = k_lo;
    while k0 < k_hi {
        sync_threads();
        // the tile: 16 keys x 64 chunks of 16 bytes, 2 a thread; missing keys are zeros (and masked below)
        unroll!(I in [0, 1] {
            let c = tid + 512 * I;
            let (kr, part) = (c / 64, c % 64);
            let kk = k0 + kr;
            let src = if kk < k_hi { key(kk) } else { core::ptr::null() };
            let v = if src.is_null() { [0u32; 4] } else { ldg_nc_v4(src.add(part * 8) as *const u32) };
            let dst = sk.add((kr * MQ_KROW + part * 8) as u32).cast::<u32>();
            dst.set(0, v[0]);
            dst.set(1, v[1]);
            dst.set(2, v[2]);
            dst.set(3, v[3]);
        });
        if tid < MQ_KT {
            let kk = k0 + tid;
            svd.set(tid as u32, (kk < k_hi && !key(kk).is_null()) as u32);
        }
        sync_threads();
        // S partial over this warp's 64 dims, to shared memory
        let pb = (w * 256) as u32;
        unroll!(NT in [0, 1] {
            let krow = sk.add(((NT * 8 + g) * MQ_KROW) as u32);
            let mut sacc = [0.0f32; 4];
            unroll!(KS in [0, 1, 2, 3] {
                let d0 = (64 * dq + 16 * KS + 2 * t) as u32;
                let b0 = krow.add(d0).cast::<u32>().ld();
                let b1 = krow.add(d0 + 8).cast::<u32>().ld();
                mma::mma_bf16_16816(&mut sacc, qa[KS], [b0, b1]);
            });
            let c = (NT * 8 + 2 * t) as u32;
            sp.set(pb + g as u32 * 16 + c, sacc[0]);
            sp.set(pb + g as u32 * 16 + c + 1, sacc[1]);
            sp.set(pb + (g as u32 + 8) * 16 + c, sacc[2]);
            sp.set(pb + (g as u32 + 8) * 16 + c + 1, sacc[3]);
        });
        sync_threads();
        // the full S of this warp's heads (the eight dim parts in order), scaled and masked
        let full = |row: u32, col: u32| -> f32 {
            let mut v = 0.0f32;
            unroll!(QD in [0, 1, 2, 3, 4, 5, 6, 7] {
                v += sp.at(((mt * 8 + QD) * 256) as u32 + row * 16 + col);
            });
            if svd.at(col) != 0 { v * scale } else { f32::NEG_INFINITY }
        };
        let (gr, c0, c1) = (g as u32, (2 * t) as u32, (8 + 2 * t) as u32);
        let (pa, ca) = mq_row([full(gr, c0), full(gr, c0 + 1), full(gr, c1), full(gr, c1 + 1)], &mut m0, &mut l0);
        let (pb2, cb) = mq_row([full(gr + 8, c0), full(gr + 8, c0 + 1), full(gr + 8, c1), full(gr + 8, c1 + 1)], &mut m1,
            &mut l1);
        unroll!(N in [0, 1, 2, 3, 4, 5, 6, 7] {
            acc[N][0] *= ca;
            acc[N][1] *= ca;
            acc[N][2] *= cb;
            acc[N][3] *= cb;
        });
        // P as two bf16 A fragments (hi + lo): a0 = (row g, keys 2t..), a1 = (row g + 8, ..), a2 / a3 keys 8 + 2t..
        let hi = |a: f32, b: f32| -> u32 { bf16::pack2(a, b) };
        let lo = |a: f32, b: f32| -> u32 { bf16::pack2(a - bf16::round(a), b - bf16::round(b)) };
        let ah = [hi(pa[0], pa[1]), hi(pb2[0], pb2[1]), hi(pa[2], pa[3]), hi(pb2[2], pb2[3])];
        let al = [lo(pa[0], pa[1]), lo(pb2[0], pb2[1]), lo(pa[2], pa[3]), lo(pb2[2], pb2[3])];
        // O += P V over this warp's 64 dims (pairs of 8-dim n-tiles through ldmatrix.trans)
        unroll!(NP in [0, 1, 2, 3] {
            let bb = mma::ldmatrix_x4_trans(sk.add((krow_v + 16 * NP) as u32).addr());
            mma::mma_bf16_16816(&mut acc[2 * NP], ah, [bb[0], bb[1]]);
            mma::mma_bf16_16816(&mut acc[2 * NP], al, [bb[0], bb[1]]);
            mma::mma_bf16_16816(&mut acc[2 * NP + 1], ah, [bb[2], bb[3]]);
            mma::mma_bf16_16816(&mut acc[2 * NP + 1], al, [bb[2], bb[3]]);
        });
        k0 += MQ_KT;
    }
    // partials (an empty split leaves max -inf, sum 0, zeros)
    let ob = ((r * MQ_SPLITS + sidx) * 32) * D;
    unroll!(N in [0, 1, 2, 3, 4, 5, 6, 7] {
        let d = 64 * dq + 8 * N + 2 * t;
        *part_o.add(ob + h0 * D + d) = acc[N][0];
        *part_o.add(ob + h0 * D + d + 1) = acc[N][1];
        *part_o.add(ob + h1 * D + d) = acc[N][2];
        *part_o.add(ob + h1 * D + d + 1) = acc[N][3];
    });
    if dq == 0 && t == 0 {
        let mb = (r * MQ_SPLITS + sidx) * 32 * 2;
        *part_ml.add(mb + h0 * 2) = m0;
        *part_ml.add(mb + h0 * 2 + 1) = l0;
        *part_ml.add(mb + h1 * 2) = m1;
        *part_ml.add(mb + h1 * 2 + 1) = l1;
    }
}

/// ds_attn_mq's merge: per (head, row) the splits in order with the sink, then ds_attn's rounding, inverse RoPE of
/// the last 64 dims and output layout. grid = (heads, rows), block = 128 (4 dims a thread).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ds_attn_mq_fin(o: *mut u16, o_gstride: u32, part_o: *const f32, part_ml: *const f32,
    meta: *const u32, sink: *const f32, tab: *const f32) {
    let h = bid_x() as usize;
    let r = bid_y() as usize;
    let t = tid_x() as usize;
    let pos = *meta.add(2 * r + 1);
    let sk = *sink.add(h);
    let mut mm = sk;
    let mut s = 0;
    while s < MQ_SPLITS {
        mm = mm.max(*part_ml.add(((r * MQ_SPLITS + s) * 32 + h) * 2));
        s += 1;
    }
    let mut cw = [0.0f32; MQ_SPLITS];
    let mut den = math::exp(sk - mm);
    let mut s = 0;
    while s < MQ_SPLITS {
        let ml = part_ml.add(((r * MQ_SPLITS + s) * 32 + h) * 2);
        let ms = *ml;
        cw[s] = if ms == f32::NEG_INFINITY { 0.0 } else { math::exp(ms - mm) };
        den += *ml.add(1) * cw[s];
        s += 1;
    }
    let mut ov = [0.0f32; 4];
    let mut c = 0;
    while c < 4 {
        let d = 4 * t + c;
        let mut v = 0.0f32;
        let mut s = 0;
        while s < MQ_SPLITS {
            v += *part_o.add(((r * MQ_SPLITS + s) * 32 + h) * D + d) * cw[s];
            s += 1;
        }
        ov[c] = bf16::round(v / den);
        c += 1;
    }
    if 4 * t >= D - 64 {
        let mut c = 0;
        while c < 4 {
            let pi = (4 * t + c - (D - 64)) / 2;
            let tb = tab.add(pos as usize * 64 + 2 * pi);
            let (cs, sn) = (*tb, -*tb.add(1));
            let (a, b) = (ov[c], ov[c + 1]);
            ov[c] = a * cs - b * sn;
            ov[c + 1] = a * sn + b * cs;
            c += 2;
        }
    }
    let dst = o.add((h / 8) * o_gstride as usize + r * 4096 + (h % 8) * D + 4 * t) as *mut u32;
    *dst = bf16::pack2(ov[0], ov[1]);
    *dst.add(1) = bf16::pack2(ov[2], ov[3]);
}
