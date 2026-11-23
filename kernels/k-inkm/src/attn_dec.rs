//! Deep-context decode attention for global layers, 1..=5 query rows per call slice (decode and speculative verify rows): grouped-query form, one block a
//! (group of 8 super-tiles = 1024 keys, kv head) handling the 3 query heads of its kv head for every row of the slice, so the K/V of a key are read once
//! whatever the number of rows. The arithmetic of every (row, head) is that of `ink_attn_tc` (scores rounded to bf16 at the same points, the same mma shapes
//! and key slices, statistics and partial outputs combined in the same order), so the bits are identical; what changes is the data flow:
//!   ink_dg_k : K pass. Statistics per super-tile (same layout as ink_attn_tc phase 0 with q tile 0) and the bf16 scores of every key (kept for the V pass).
//!   ink_attn_gstat (existing, gqa = 0): global row statistics.
//!   ink_dg_v : V pass. Probabilities from the stored scores, p.V on the same mma, the 8 super-tile partials of a group added in order (as ink_attn_comb2's
//!              groups of 8) and only the group sum written.
//!   ink_dg_comb: groups added in order, rotation back (6-bit), output gate.
use crate::attn_tc::*;
use crate::*;
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

const NEG: f32 = -3.0e38;


/// Bias table RL[r][d] for the distances this super-tile's keys can have from the rows of the call (as ink_attn_tc), rows r < 3 * nrows.
#[inline(always)]
unsafe fn bias_table(smem: Shared<u8>, proj: *const u16, pos0: u32, ntok: u32, cs_: u32, ce_: u32, m_rows: u32) -> u32 {
    let rs: Shared<f32> = smem.byte_add(DRS_OFF).cast();
    let rl: Shared<u16> = smem.byte_add(DRL_OFF).cast();
    let t = tid_x();
    let (pmin, pmax) = (pos0, pos0 + ntok - 1);
    let (klo, khi) = (cs_ * 64, ce_ * 64 + 63);
    let dmin = if pmin > khi { pmin - khi } else { 0 };
    let dmx = if pmax >= klo { pmax - klo } else { 0 };
    let dmax = if dmx > 1023 { 1023 } else { dmx };
    let mut d = dmin + t;
    while d <= dmax {
        let mut pj = [0.0f32; 16];
        kcommon::unroll!(j in 16 => {
            pj[j] = bf(*proj.add(j * 1024 + d as usize));
        });
        let mut r = 0;
        while r < m_rows {
            let mut b = 0.0f32;
            kcommon::unroll!(j in 16 => {
                b = math::fma(rs.at(r * 16 + j as u32), pj[j], b);
            });
            rl.set(r * 144 + (d - dmin), tb(b));
            r += 1;
        }
        d += 128;
    }
    dmin
}

#[inline(always)]
fn sub_bf16x2(a: u32, b: u32) -> u32 {
    let d: u32;
    unsafe { core::arch::asm!("sub.rn.bf16x2 {d}, {a}, {b};", d = out(reg32) d, a = in(reg32) a, b = in(reg32) b, options(pure, nomem, nostack)) };
    d
}
#[inline(always)]
fn fma_bf16x2(a: u32, b: u32, c: u32) -> u32 {
    let d: u32;
    unsafe { core::arch::asm!("fma.rn.bf16x2 {d}, {a}, {b}, {c};", d = out(reg32) d, a = in(reg32) a, b = in(reg32) b, c = in(reg32) c, options(pure, nomem, nostack)) };
    d
}

// ---- shared memory plan (bytes) ----
pub(crate) const TILE_B: u32 = 64 * PITCH; // 64 keys, 16 per warp
pub(crate) const WTILE: u32 = 16 * PITCH; // one warp's 16 keys
pub(crate) const QREC: u32 = 16 * 104; // 1664 B: the packed records of one warp's 16 keys of a chunk (chunk-major layout: contiguous)
pub(crate) const DQ: u32 = 3; // 6-bit raw stages in flight per warp
pub(crate) const SC_B: u32 = 512; // one warp's stored scores of a chunk (32 lanes x 16 B)
const DQS_OFF: u32 = 0; // 16 x PITCH
const DRL_OFF: u32 = 4352; // 16 x 144 bf16
const DRS_OFF: u32 = 8960; // 16 x 16 f32
const DST_OFF: u32 = 9984; // 4 warps x 16 rows x 2 f32
const K_TILE: u32 = 10752; // after Q (4352) | RL (4608) | RS (1024) | ST (640) | pad
/// Dynamic shared memory of the K pass.
pub const DGK_SMEM: u32 = K_TILE + 2 * TILE_B; // bf16: double-buffered tiles; 6-bit: one tile + 4 warps x DQ raw stages (smaller)
/// Dynamic shared memory of the V pass for `nrows` rows (partial-output exchange 4 warps x 3 nrows x 128 floats).
pub fn dgv_smem(qmode: bool, nrows: u32) -> u32 {
    let tile = if qmode { TILE_B } else { 2 * TILE_B };
    let stg = if qmode { DQ * (QREC + SC_B) } else { 2 * SC_B };
    tile + 4 * stg + 4 * 3 * nrows * 128 * 4
}

/// 6-bit dequantisation of one warp's 16 keys of a chunk from the raw stage into its bf16 tile (same values as `load_chunk_q`): value = (2 q - 63) x scale / 64
/// rounded to bf16; two codes at a time in packed bf16: x = bf16(128 + q) (bits 0x4300 | q), q - 31 = x - 159 exactly, fma(q - 31, 2 sc, -sc) = (2 q - 63) sc
/// with the single rounding of the reference.
#[inline(always)]
pub(crate) unsafe fn dequant_warp(tile: Shared<u8>, stage: Shared<u8>) {
    let l = lane();
    kcommon::unroll!(i in 2 => {
        let jid = l + 32 * i as u32;
        let rr = jid >> 2;
        let g = jid & 3;
        let rec = stage.byte_add(rr * 104 + 24 * g);
        let a0 = <[u32; 2] as kcommon::smem::SmemIo>::lds(rec.addr());
        let a1 = <[u32; 2] as kcommon::smem::SmemIo>::lds(rec.byte_add(8).addr());
        let a2 = <[u32; 2] as kcommon::smem::SmemIo>::lds(rec.byte_add(16).addr());
        let w = [a0[0], a0[1], a1[0], a1[1], a2[0], a2[1]];
        let sb = <u16 as kcommon::smem::SmemIo>::lds(stage.byte_add(rr * 104 + 96 + 2 * g).addr());
        let sc = bf(sb) * (1.0 / 64.0);
        let sc2 = bf16::pack2(2.0 * sc, 2.0 * sc);
        let nsc = bf16::pack2(-sc, -sc);
        let o = tile.byte_add(rr * PITCH + g * 64);
        kcommon::unroll!(p in 8 => {
            const B0: usize = 24 * p;
            const W0: usize = B0 >> 5;
            const O0: u32 = (B0 & 31) as u32;
            let s0 = if O0 + 12 <= 32 { w[W0] >> O0 } else { (w[W0] >> O0) | (w[W0 + 1] << (32 - O0)) };
            const B1: usize = 24 * p + 12;
            const W1: usize = B1 >> 5;
            const O1: u32 = (B1 & 31) as u32;
            let s1 = if O1 + 12 <= 32 { w[W1] >> O1 } else { (w[W1] >> O1) | (w[W1 + 1] << (32 - O1)) };
            let x0 = ((s0 & 0x3F) | ((s0 << 10) & 0x3F0000)) | 0x4300_4300;
            let x1 = ((s1 & 0x3F) | ((s1 << 10) & 0x3F0000)) | 0x4300_4300;
            let v0 = fma_bf16x2(sub_bf16x2(x0, 0x431F_431F), sc2, nsc);
            let v1 = fma_bf16x2(sub_bf16x2(x1, 0x431F_431F), sc2, nsc);
            <[u32; 2] as kcommon::smem::SmemIo>::sts(o.byte_add(8 * p as u32).addr(), [v0, v1]);
        });
    });
}

/// cp.async of one warp's 16 keys of a chunk: 6-bit records (contiguous 1664 B of the chunk-major cache) into `dst`, or bf16 rows into the tile `dst` (pitch PITCH).
#[inline(always)]
pub(crate) unsafe fn warp_fetch(q6: bool, base: *const u8, kvh: u32, c: u32, wid: u32, ring: u32, dst: Shared<u8>) {
    let l = lane();
    if q6 {
        let src = base.add((((c * 8 + kvh) * 6656) + QREC * wid) as usize);
        kcommon::unroll!(k in 4 => {
            let i = l + 32 * k as u32;
            if i < 104 {
                cp_async_16(dst.byte_add(16 * i).addr(), src.add(16 * i as usize));
            }
        });
    } else {
        kcommon::unroll!(k in 8 => {
            let i = l + 32 * k as u32;
            let r = i >> 4;
            let cc = i & 15;
            let row = (c * 64 + 16 * wid + r) % ring;
            cp_async_16(dst.byte_add(r * PITCH + cc * 16).addr(), base.add((row as usize * 1024 + kvh as usize * 128 + cc as usize * 8) * 2));
        });
    }
}

/// K pass. grid = (ceil(nsup / 8), 8 kv heads), block 128 (4 independent warp streams a chunk of 64 keys: each warp fetches, dequantises and scores its own 16 keys, no
/// block barrier inside a super-tile). stat: [head][nsup][16 rows][2] (the ink_attn_tc layout of q tile 0, rows = global row index); scr: stored scores (this
/// slice's region), `nch` 64-key chunks per kv head. The 6-bit cache is chunk-major: [chunk][kv head][64 keys][104 B].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_dg_k(stat: *mut f32, scr: *mut u16, q: *const u16, kc: *const u16, rsx: *const u16, proj: *const u16, pos: *const u32,
    scaling: f32, ring: u32, nsup: u32, nrows: u32, row0: u32, ntok: u32, nch: u32, qmode: u32, window: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let smem: Shared<u8> = dyn_base();
    let q6 = (qmode & 1) != 0;
    let kvh = bid_y();
    let jb = bid_x() * 8;
    let wid = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    let t = tid_x();
    let pos0 = *pos;
    let m_rows = 3 * nrows;
    let tile_hi = pos0 + ntok - 1;
    // sliding layers (window > 0): super-tiles start at the window's first super-tile, one group of 16 holds them all (as ink_attn_comb2)
    let lo_of = |p: u32| if window > 0 && p + 1 > window { p + 1 - window } else { 0 };
    let tile_lo = lo_of(pos0);
    let sup0 = tile_lo / 128;
    let gsz: u32 = if window > 0 { 16 } else { 8 };
    let cbase = 2 * sup0;
    let jn = if jb + gsz < nsup { jb + gsz } else { nsup };
    let inactive = |j: u32| {
        if t < m_rows {
            let (qr, ho) = (t / 3, t % 3);
            let hh = 3 * kvh + ho;
            *stat.add(((hh * nsup + j) * 32 + (row0 + qr) * 2) as usize) = -1.0e30;
            *stat.add(((hh * nsup + j) * 32 + (row0 + qr) * 2) as usize + 1) = 0.0;
        }
    };
    if sup0 + jb > tile_hi / 128 {
        let mut j = jb;
        while j < jn {
            inactive(j);
            j += 1;
        }
        return;
    }
    // ---- stage Q rows (row r = 3 * qrow + head offset), relative-bias sources ----
    let mut i = t;
    while i < 16 * 16 {
        let r = i >> 4;
        let c = i & 15;
        let d = smem.byte_add(DQS_OFF + r * PITCH + c * 16).addr();
        if r < m_rows {
            cp_async_16(d, q.add(((row0 + r / 3) * 3072 + (3 * kvh + r % 3) * 128 + c * 8) as usize) as *const u8);
        } else {
            cp_async_16_zfill(d, q as *const u8, 0);
        }
        i += 128;
    }
    cp_async_commit();
    let rs: Shared<f32> = smem.byte_add(DRS_OFF).cast();
    let mut i = t;
    while i < 256 {
        let r = i >> 4;
        rs.set(i, if r < m_rows { bf(*rsx.add(((row0 + r / 3) * 384 + (3 * kvh + r % 3) * 16 + (i & 15)) as usize)) } else { 0.0 });
        i += 128;
    }
    cp_async_wait_all();
    sync_threads();
    if q6 && t < 64 {
        let (r, gg) = (t >> 2, t & 3);
        let qp = smem.byte_add(DQS_OFF + r * PITCH + gg * 64);
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
    let mut qf = [[0u32; 4]; 8];
    kcommon::unroll!(ks in 8 => {
        qf[ks] = ldmatrix_x4(smem.byte_add(DQS_OFF + (l & 15) * PITCH + (ks as u32 * 2 + (l >> 4)) * 16).addr());
    });
    let (ra, rb_) = (g, g + 8);
    let pa = pos0 + row0 + ra / 3;
    let pb = pos0 + row0 + rb_ / 3;
    let va = ra < m_rows;
    let vb = rb_ < m_rows;
    let rl: Shared<u16> = smem.byte_add(DRL_OFF).cast();
    let st: Shared<f32> = smem.byte_add(DST_OFF).cast();
    let sbase = (((kvh * nch) * 4 + wid) * 32 + l) * 8;
    let base = kc as *const u8;
    let tile0 = smem.byte_add(K_TILE);
    let stg0 = smem.byte_add(K_TILE + TILE_B + wid * DQ * QREC);
    let c0 = if 2 * (sup0 + jb) > tile_lo / 64 { 2 * (sup0 + jb) } else { tile_lo / 64 };
    let c1 = if 2 * (sup0 + jn - 1) + 1 < tile_hi / 64 { 2 * (sup0 + jn - 1) + 1 } else { tile_hi / 64 };
    let (lo_a, lo_b) = (lo_of(pa), lo_of(pb));
    let nd: u32 = if q6 { DQ } else { 2 };
    // prologue: the first `nd` chunks of this warp's stream
    let mut k = 0;
    while k < nd {
        if c0 + k <= c1 {
            let dst = if q6 { stg0.byte_add(k * QREC) } else { tile0.byte_add(k * TILE_B + wid * WTILE) };
            warp_fetch(q6, base, kvh, c0 + k, wid, ring, dst);
        }
        cp_async_commit();
        k += 1;
    }
    // bias table of the first super-tile
    let sup_ce = |j: u32| if 2 * (sup0 + j) + 1 < tile_hi / 64 { 2 * (sup0 + j) + 1 } else { tile_hi / 64 };
    let mut dmin = bias_table(smem, proj, pos0, ntok, c0, sup_ce(jb), m_rows);
    sync_threads();
    let (mut ma, mut la, mut mb, mut lb) = (-1.0e30f32, 0.0f32, -1.0e30f32, 0.0f32);
    let mut c = c0;
    while c <= c1 {
        let slot = (c - c0) % nd;
        if q6 {
            cp_async_wait::<2>();
        } else {
            cp_async_wait::<1>();
        }
        sync_warp();
        let tile = if q6 { tile0.byte_add(wid * WTILE) } else { tile0.byte_add(slot * TILE_B + wid * WTILE) };
        if q6 {
            dequant_warp(tile, stg0.byte_add(slot * QREC));
            sync_warp();
            if c + DQ <= c1 {
                warp_fetch(true, base, kvh, c + DQ, wid, ring, stg0.byte_add(slot * QREC));
            }
            cp_async_commit();
        }
        let mut s = [[NEG; 4]; 2];
        {
            let mut acc = [[0.0f32; 4]; 2];
            kcommon::unroll!(ks in 8 => {
                kcommon::unroll!(nt in 2 => {
                    let bfr = ldmatrix_x2(tile.byte_add((nt as u32 * 8 + (l & 7)) * PITCH + (ks as u32 * 2 + ((l >> 3) & 1)) * 16).addr());
                    mma_bf16_16816(&mut acc[nt], qf[ks], bfr);
                });
            });
            // chunk-uniform facts: every key of the chunk is visible to every row (no mask), no key is within 1024 of any row (no bias)
            let nomask = window == 0 && c * 64 + 63 <= pos0 + row0;
            let nobias = window == 0 && pos0 + row0 > c * 64 + 63 + 1023;
            kcommon::unroll!(nt in 2 => {
                kcommon::unroll!(e in 4 => {
                    let (row_p, valid, rr, lo) = if e < 2 { (pa, va, ra, lo_a) } else { (pb, vb, rb_, lo_b) };
                    let key = c * 64 + wid * 16 + nt as u32 * 8 + 2 * t4 + (e as u32 & 1);
                    if valid && (nomask || (key >= lo && key <= row_p)) {
                        let mut sv = rb(rb(acc[nt][e]) * scaling);
                        if !nobias {
                            let dist = row_p - key;
                            if dist < 1024 {
                                sv = rb(sv + bf(rl.at(rr * 144 + (dist - dmin))));
                            }
                        }
                        s[nt][e] = sv;
                    }
                });
            });
        }
        if !q6 {
            // the tile buffer is free again: fetch the chunk two ahead into it
            sync_warp();
            if c + 2 <= c1 {
                warp_fetch(false, base, kvh, c + 2, wid, ring, tile);
            }
            cp_async_commit();
        }
        // keep the scores for the V pass (0xFFFF = masked)
        if va || vb {
            let o = scr.add((sbase + (c - cbase) * 4 * 32 * 8) as usize) as *mut u32;
            let mut w = [0u32; 4];
            kcommon::unroll!(k in 4 => {
                let (nt, e0) = (k / 2, (k % 2) * 2);
                let a = if s[nt][e0] > NEG { tb(s[nt][e0]) as u32 } else { 0xFFFF };
                let b = if s[nt][e0 + 1] > NEG { tb(s[nt][e0 + 1]) as u32 } else { 0xFFFF };
                w[k] = a | (b << 16);
            });
            *o = w[0];
            *o.add(1) = w[1];
            *o.add(2) = w[2];
            *o.add(3) = w[3];
        }
        kcommon::unroll!(nt in 2 => {
            kcommon::unroll!(e in 4 => {
                let sv = s[nt][e];
                if sv > NEG {
                    // one exp for both cases: ex(min - max) = ex(ma - sv) when sv > ma, ex(sv - ma) otherwise (bit for bit the same arguments)
                    if e < 2 {
                        let up = sv > ma;
                        let ee = ex(if up { ma - sv } else { sv - ma });
                        la = if up { la * ee + 1.0 } else { la + ee };
                        ma = if up { sv } else { ma };
                    } else {
                        let up = sv > mb;
                        let ee = ex(if up { mb - sv } else { sv - mb });
                        lb = if up { lb * ee + 1.0 } else { lb + ee };
                        mb = if up { sv } else { mb };
                    }
                }
            });
        });
        if (c & 1) == 1 || c == c1 {
            // ---- super-tile j = c / 2 done: statistics of the 4 warps combined as in ink_attn_tc ----
            let j = (c >> 1) - sup0;
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
            if t < m_rows {
                let mut m = -1.0e30f32;
                kcommon::unroll!(w in 4 => {
                    let mw = st.at((w as u32 * 16 + t) * 2);
                    if mw > m { m = mw; }
                });
                let mut ls = 0.0f32;
                kcommon::unroll!(w in 4 => {
                    ls += st.at((w as u32 * 16 + t) * 2 + 1) * ex(st.at((w as u32 * 16 + t) * 2) - m);
                });
                let (qr, ho) = (t / 3, t % 3);
                let hh = 3 * kvh + ho;
                *stat.add(((hh * nsup + j) * 32 + (row0 + qr) * 2) as usize) = m;
                *stat.add(((hh * nsup + j) * 32 + (row0 + qr) * 2) as usize + 1) = ls;
            }
            ma = -1.0e30;
            la = 0.0;
            mb = -1.0e30;
            lb = 0.0;
            if c < c1 {
                // next super-tile: its bias table (rl is read by the scores above, st by the combine: all done after this barrier)
                sync_threads();
                dmin = bias_table(smem, proj, pos0, ntok, c + 1, sup_ce(j + 1), m_rows);
                sync_threads();
            }
        }
        c += 1;
    }
    // super-tiles of the group past the last key: statistics (-1e30, 0)
    let mut j = (c1 >> 1) - sup0 + 1;
    while j < jn {
        inactive(j);
        j += 1;
    }
}

/// V pass. grid = (ceil(nsup / 8), 8), block 128, warp streams like the K pass. gst: ink_attn_gstat output (non-gqa layout, q tile 0); gp: group sums
/// [row][head][group][128] f32. Dynamic shared memory `dgv_smem`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_dg_v(gp: *mut f32, scr: *const u16, gst: *const f32, vc: *const u16, pos: *const u32, ring: u32, nsup: u32, ngr: u32,
    nrows: u32, row0: u32, ntok: u32, nch: u32, qmode: u32, window: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let smem: Shared<u8> = dyn_base();
    let q6 = (qmode & 1) != 0;
    let kvh = bid_y();
    let grp = bid_x();
    let gsz: u32 = if window > 0 { 16 } else { 8 };
    let jb = grp * gsz;
    let wid = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    let t = tid_x();
    let pos0 = *pos;
    let m_rows = 3 * nrows;
    let tile_hi = pos0 + ntok - 1;
    let tile_lo = if window > 0 && pos0 + 1 > window { pos0 + 1 - window } else { 0 };
    let sup0 = tile_lo / 128;
    let cbase = 2 * sup0;
    let jn = if jb + gsz < nsup { jb + gsz } else { nsup };
    let mut acc = [0.0f32; 16];
    let zero = f32::from_bits(0);
    let (ra, rb_) = (g, g + 8);
    let va = ra < m_rows;
    let vb = rb_ < m_rows;
    let rd = |r: u32, ok: bool| -> (f32, f32) {
        if ok {
            let e = ((((3 * kvh + r % 3) * 16 + row0 + r / 3) * 2) as usize) as usize;
            (*gst.add(e), *gst.add(e + 1))
        } else {
            (0.0, 1.0)
        }
    };
    let (gma, gla) = rd(ra, va);
    let (gmb, glb) = rd(rb_, vb);
    let sbase = (((kvh * nch) * 4 + wid) * 32 + l) * 8;
    let base = vc as *const u8;
    let nd: u32 = if q6 { DQ } else { 2 };
    let tile_bufs = if q6 { 1 } else { 2 };
    let tile0 = smem;
    let stg_w = if q6 { DQ * (QREC + SC_B) } else { 2 * SC_B };
    let stg0 = smem.byte_add(tile_bufs * TILE_B + wid * stg_w);
    let ob_off = tile_bufs * TILE_B + 4 * stg_w;
    let ob: Shared<f32> = smem.byte_add(ob_off).cast();
    let c0 = if 2 * (sup0 + jb) > tile_lo / 64 { 2 * (sup0 + jb) } else { tile_lo / 64 };
    let c1 = if 2 * (sup0 + jn - 1) + 1 < tile_hi / 64 { 2 * (sup0 + jn - 1) + 1 } else { tile_hi / 64 };
    let last_j = (c1 >> 1) - sup0;
    if sup0 + jb > tile_hi / 128 {
        // all super-tiles of the group are past the last key: zero partials (ink_attn_comb2 adds them)
        kcommon::unroll!(r in 16 => {
            acc[r] += zero;
        });
    } else {
        // per slot: [raw 1664 B (6-bit)] [scores 512 B]
        let sc_at = |slot: u32| -> Shared<u8> { stg0.byte_add(if q6 { slot * (QREC + SC_B) + QREC } else { slot * SC_B }) };
        let fetch = |c: u32, slot: u32| {
            let dst = if q6 { stg0.byte_add(slot * (QREC + SC_B)) } else { tile0.byte_add(slot * TILE_B + wid * WTILE) };
            warp_fetch(q6, base, kvh, c, wid, ring, dst);
            cp_async_16(sc_at(slot).byte_add(16 * l).addr(), scr.add((((kvh * nch + (c - cbase)) * 1024) + (wid * 32 + l) * 8) as usize) as *const u8);
        };
        let mut k = 0;
        while k < nd {
            if c0 + k <= c1 {
                fetch(c0 + k, k);
            }
            cp_async_commit();
            k += 1;
        }
        let mut o = [[0.0f32; 4]; 16];
        let mut c = c0;
        while c <= c1 {
            let slot = (c - c0) % nd;
            if q6 {
                cp_async_wait::<2>();
            } else {
                cp_async_wait::<1>();
            }
            sync_warp();
            let tile = if q6 { tile0.byte_add(wid * WTILE) } else { tile0.byte_add(slot * TILE_B + wid * WTILE) };
            let sw = <[u32; 4] as kcommon::smem::SmemIo>::lds(sc_at(slot).byte_add(16 * l).addr());
            if q6 {
                dequant_warp(tile, stg0.byte_add(slot * (QREC + SC_B)));
        
                sync_warp();
                if c + DQ <= c1 {
                    fetch(c + DQ, slot);
                }
                cp_async_commit();
            }
            let sc = |k: usize, hi: bool| -> f32 {
                let b = if hi { sw[k] >> 16 } else { sw[k] & 0xFFFF };
                if b == 0xFFFF { NEG } else { bf(b as u16) }
            };
            let mut s = [[NEG; 4]; 2];
            if va || vb {
                s[0][0] = sc(0, false);
                s[0][1] = sc(0, true);
                s[0][2] = sc(1, false);
                s[0][3] = sc(1, true);
                s[1][0] = sc(2, false);
                s[1][1] = sc(2, true);
                s[1][2] = sc(3, false);
                s[1][3] = sc(3, true);
            }
            let mut pf = [0u32; 4];
            {
                let pv = |sv: f32, m: f32, ls: f32| -> f32 { if sv > NEG { rb(ex(sv - m) / ls) } else { 0.0 } };
                let pa0 = pv(s[0][0], gma, gla);
                let pa1 = pv(s[0][1], gma, gla);
                let pb0 = pv(s[0][2], gmb, glb);
                let pb1 = pv(s[0][3], gmb, glb);
                let pc0 = pv(s[1][0], gma, gla);
                let pc1 = pv(s[1][1], gma, gla);
                let pd0 = pv(s[1][2], gmb, glb);
                let pd1 = pv(s[1][3], gmb, glb);
                pf[0] = bf16::pack2(pa0, pa1);
                pf[1] = bf16::pack2(pb0, pb1);
                pf[2] = bf16::pack2(pc0, pc1);
                pf[3] = bf16::pack2(pd0, pd1);
            }
            kcommon::unroll!(dp in 8 => {
                let vb4 = ldmatrix_x4_trans(tile.byte_add(((l & 7) + 8 * ((l >> 3) & 1)) * PITCH + (dp as u32 * 16 + (l >> 4) * 8) * 2).addr());
                mma_bf16_16816(&mut o[2 * dp], pf, [vb4[0], vb4[1]]);
                mma_bf16_16816(&mut o[2 * dp + 1], pf, [vb4[2], vb4[3]]);
            });
            if !q6 {
                sync_warp();
                if c + 2 <= c1 {
                    fetch(c + 2, slot);
                }
                cp_async_commit();
            }
            if (c & 1) == 1 || c == c1 {
                // ---- super-tile done: the 4 warps' partials added in warp order, then into the group sum ----
                kcommon::unroll!(nt in 16 => {
                    let col = nt as u32 * 8 + 2 * t4;
                    if va {
                        ob.set(wid * (m_rows * 128) + ra * 128 + col, o[nt][0]);
                        ob.set(wid * (m_rows * 128) + ra * 128 + col + 1, o[nt][1]);
                    }
                    if vb {
                        ob.set(wid * (m_rows * 128) + rb_ * 128 + col, o[nt][2]);
                        ob.set(wid * (m_rows * 128) + rb_ * 128 + col + 1, o[nt][3]);
                    }
                });
                sync_threads();
                kcommon::unroll!(r in 16 => {
                    if (r as u32) < m_rows {
                        let i = r as u32 * 128 + t;
                        let v = ((ob.at(i) + ob.at(m_rows * 128 + i)) + ob.at(2 * m_rows * 128 + i)) + ob.at(3 * m_rows * 128 + i);
                        acc[r] += v;
                    }
                });
                sync_threads();
                o = [[0.0f32; 4]; 16];
            }
            c += 1;
        }
        // super-tiles of the group past the last key: zero partials
        let mut j = last_j + 1;
        while j < jn {
            kcommon::unroll!(r in 16 => {
                acc[r] += zero;
            });
            j += 1;
        }
    }
    kcommon::unroll!(r in 16 => {
        if (r as u32) < m_rows {
            let (qr, ho) = (r as u32 / 3, r as u32 % 3);
            let hh = 3 * kvh + ho;
            *gp.add((((((row0 + qr) * 24 + hh) * ngr + grp) * 128) + t) as usize) = acc[r];
        }
    });
}

/// grid = (24 heads, rows), block 128: out[row][h*128 + d] = bf16(bf16(sum of the group sums in order) * sigmoid(gate)), rotated back for 6-bit K/V.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_dg_comb(out: *mut u16, gp: *const f32, gate: *const u16, ldg: u32, ngr: u32, qmode: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let h = bid_x();
    let r = bid_y();
    let d = tid_x();
    let mut total = 0.0f32;
    let mut g = 0;
    while g < ngr {
        total += *gp.add((((r * 24 + h) * ngr + g) * 128 + d) as usize);
        g += 1;
    }
    if (qmode & 1) != 0 {
        let mut x = total;
        let mut m = 1;
        while m < 32 {
            let p = warp::shfl_xor(x, m);
            x = if tid_x() & m == 0 { x + p } else { p - x };
            m <<= 1;
        }
        total = x * 0.176_776_7;
    }
    let a = rb(total);
    let gv = *gate.add((r * ldg + h * 128 + d) as usize);
    *out.add((r * 3072 + h * 128 + d) as usize) = tb(a * sigm(bf(gv)));
}
