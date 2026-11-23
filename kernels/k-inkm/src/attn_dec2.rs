//! 6..=8 query rows in ONE scan of the K/V: the same kernels as `attn_dec` with two row groups (rows 0..4 and 4..ntok, 3 query heads each, M = 12 and 3 * (ntok - 4)) sharing
//! every fetched and dequantised K / V tile; each group has its own mma row tile, running statistics, stored scores and partial outputs, and per (row, head) the
//! arithmetic is that of `ink_attn_tc` (as in attn_dec). Global layers only (window 0).
use crate::attn_tc::PITCH;
use crate::attn_dec::{dequant_warp, warp_fetch, DQ, QREC, SC_B, TILE_B, WTILE};
use crate::*;
use kcommon::mma::*;
use kcommon::smem::{dyn_base, Shared};
use kcommon::*;

const NEG: f32 = -3.0e38;
// shared memory plan of the K pass (two groups): Q tiles | bias tables | bias sources | statistics | tiles | raw stages
const G_Q: u32 = 0; // 2 x 16 x PITCH
const G_RL: u32 = 8704; // 2 x 16 x 144 bf16
const G_RS: u32 = 17920; // 2 x 16 x 16 f32
const G_ST: u32 = 19968; // 2 x (4 warps x 16 rows x 2 f32), 640 B apart
pub const G_TILE: u32 = 21248;

#[inline(always)]
unsafe fn bias_table2(smem: Shared<u8>, proj: *const u16, pos0: u32, ntok: u32, cs_: u32, ce_: u32, m_rows: [u32; 2]) -> u32 {
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
        kcommon::unroll!(gi in 2 => {
            let rs: Shared<f32> = smem.byte_add(G_RS + gi as u32 * 1024).cast();
            let rl: Shared<u16> = smem.byte_add(G_RL + gi as u32 * 4608).cast();
            let mut r = 0;
            while r < m_rows[gi] {
                let mut b = 0.0f32;
                kcommon::unroll!(j in 16 => {
                    b = math::fma(rs.at(r * 16 + j as u32), pj[j], b);
                });
                rl.set(r * 144 + (d - dmin), tb(b));
                r += 1;
            }
        });
        d += 128;
    }
    dmin
}

/// K pass, two row groups. Same grid / structure as `ink_dg_k`; scr: group 0's scores at `scr`, group 1's at `scr + nch * 8192` (u16 elements).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_dg2_k(stat: *mut f32, scr: *mut u16, q: *const u16, kc: *const u16, rsx: *const u16, proj: *const u16, pos: *const u32,
    scaling: f32, ring: u32, nsup: u32, ntok: u32, nch: u32, qmode: u32) {
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
    let nr = [4u32, ntok - 4];
    let m_rows = [12u32, 3 * (ntok - 4)];
    let row0 = [0u32, 4];
    let tile_hi = pos0 + ntok - 1;
    let jn = if jb + 8 < nsup { jb + 8 } else { nsup };
    let inactive = |j: u32| {
        kcommon::unroll!(gi in 2 => {
            if t < m_rows[gi] {
                let (qr, ho) = (t / 3, t % 3);
                let hh = 3 * kvh + ho;
                *stat.add(((hh * nsup + j) * 32 + (row0[gi] + qr) * 2) as usize) = -1.0e30;
                *stat.add(((hh * nsup + j) * 32 + (row0[gi] + qr) * 2) as usize + 1) = 0.0;
            }
        });
    };
    if jb > tile_hi / 128 {
        let mut j = jb;
        while j < jn {
            inactive(j);
            j += 1;
        }
        return;
    }
    // ---- stage the Q rows and bias sources of both groups ----
    kcommon::unroll!(gi in 2 => {
        let mut i = t;
        while i < 16 * 16 {
            let r = i >> 4;
            let c = i & 15;
            let d = smem.byte_add(G_Q + gi as u32 * 4352 + r * PITCH + c * 16).addr();
            if r < m_rows[gi] {
                cp_async_16(d, q.add(((row0[gi] + r / 3) * 3072 + (3 * kvh + r % 3) * 128 + c * 8) as usize) as *const u8);
            } else {
                cp_async_16_zfill(d, q as *const u8, 0);
            }
            i += 128;
        }
    });
    cp_async_commit();
    kcommon::unroll!(gi in 2 => {
        let rs: Shared<f32> = smem.byte_add(G_RS + gi as u32 * 1024).cast();
        let mut i = t;
        while i < 256 {
            let r = i >> 4;
            rs.set(i, if r < m_rows[gi] { bf(*rsx.add(((row0[gi] + r / 3) * 384 + (3 * kvh + r % 3) * 16 + (i & 15)) as usize)) } else { 0.0 });
            i += 128;
        }
    });
    cp_async_wait_all();
    sync_threads();
    if q6 && t < 128 {
        // rotate the rows of both groups (thread = (group, row, 32-dim group))
        let gi = t >> 6;
        let (r, gg) = ((t >> 2) & 15, t & 3);
        let qp = smem.byte_add(G_Q + gi * 4352 + r * PITCH + gg * 64);
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
    let mut qf = [[[0u32; 4]; 8]; 2];
    kcommon::unroll!(gi in 2 => {
        kcommon::unroll!(ks in 8 => {
            qf[gi][ks] = ldmatrix_x4(smem.byte_add(G_Q + gi as u32 * 4352 + (l & 15) * PITCH + (ks as u32 * 2 + (l >> 4)) * 16).addr());
        });
    });
    let (ra, rb_) = (g, g + 8);
    let pa = [pos0 + row0[0] + ra / 3, pos0 + row0[1] + ra / 3];
    let pb = [pos0 + row0[0] + rb_ / 3, pos0 + row0[1] + rb_ / 3];
    let va = [ra < m_rows[0], ra < m_rows[1]];
    let vb = [rb_ < m_rows[0], rb_ < m_rows[1]];
    let sbase = [(((kvh * nch) * 4 + wid) * 32 + l) * 8, ((((kvh * nch) + nch * 8) * 4 + wid) * 32 + l) * 8];
    let base = kc as *const u8;
    let tile0 = smem.byte_add(G_TILE);
    let stg0 = smem.byte_add(G_TILE + TILE_B + wid * DQ * QREC);
    let c0 = 2 * jb;
    let c1 = if 2 * (jn - 1) + 1 < tile_hi / 64 { 2 * (jn - 1) + 1 } else { tile_hi / 64 };
    let nd: u32 = if q6 { DQ } else { 2 };
    let mut k = 0;
    while k < nd {
        if c0 + k <= c1 {
            let dst = if q6 { stg0.byte_add(k * QREC) } else { tile0.byte_add(k * TILE_B + wid * WTILE) };
            warp_fetch(q6, base, kvh, c0 + k, wid, ring, dst);
        }
        cp_async_commit();
        k += 1;
    }
    let sup_ce = |j: u32| if 2 * j + 1 < tile_hi / 64 { 2 * j + 1 } else { tile_hi / 64 };
    let mut dmin = bias_table2(smem, proj, pos0, ntok, c0, sup_ce(jb), m_rows);
    sync_threads();
    let mut ma = [-1.0e30f32; 2];
    let mut la = [0.0f32; 2];
    let mut mb = [-1.0e30f32; 2];
    let mut lb = [0.0f32; 2];
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
        let nomask = c * 64 + 63 <= pos0;
        let nobias = pos0 > c * 64 + 63 + 1023;
        // the K fragments of this warp's 16 keys, once for both groups
        let mut bfr = [[0u32; 2]; 16];
        kcommon::unroll!(ks in 8 => {
            kcommon::unroll!(nt in 2 => {
                bfr[ks * 2 + nt] = ldmatrix_x2(tile.byte_add((nt as u32 * 8 + (l & 7)) * PITCH + (ks as u32 * 2 + ((l >> 3) & 1)) * 16).addr());
            });
        });
        let mut sg = [[[NEG; 4]; 2]; 2];
        kcommon::unroll!(gi in 2 => {
            let rl: Shared<u16> = smem.byte_add(G_RL + gi as u32 * 4608).cast();
            let mut acc = [[0.0f32; 4]; 2];
            kcommon::unroll!(ks in 8 => {
                kcommon::unroll!(nt in 2 => {
                    mma_bf16_16816(&mut acc[nt], qf[gi][ks], bfr[ks * 2 + nt]);
                });
            });
            kcommon::unroll!(nt in 2 => {
                kcommon::unroll!(e in 4 => {
                    let (row_p, valid, rr) = if e < 2 { (pa[gi], va[gi], ra) } else { (pb[gi], vb[gi], rb_) };
                    let key = c * 64 + wid * 16 + nt as u32 * 8 + 2 * t4 + (e as u32 & 1);
                    if valid && (nomask || key <= row_p) {
                        let mut sv = rb(rb(acc[nt][e]) * scaling);
                        if !nobias {
                            let dist = row_p - key;
                            if dist < 1024 {
                                sv = rb(sv + bf(rl.at(rr * 144 + (dist - dmin))));
                            }
                        }
                        sg[gi][nt][e] = sv;
                    }
                });
            });
        });
        if !q6 {
            sync_warp();
            if c + 2 <= c1 {
                warp_fetch(false, base, kvh, c + 2, wid, ring, tile);
            }
            cp_async_commit();
        }
        kcommon::unroll!(gi in 2 => {
            let s = sg[gi];
            if va[gi] || vb[gi] {
                let o = scr.add((sbase[gi] + c * 4 * 32 * 8) as usize) as *mut u32;
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
                        if e < 2 {
                            let up = sv > ma[gi];
                            let ee = ex(if up { ma[gi] - sv } else { sv - ma[gi] });
                            la[gi] = if up { la[gi] * ee + 1.0 } else { la[gi] + ee };
                            ma[gi] = if up { sv } else { ma[gi] };
                        } else {
                            let up = sv > mb[gi];
                            let ee = ex(if up { mb[gi] - sv } else { sv - mb[gi] });
                            lb[gi] = if up { lb[gi] * ee + 1.0 } else { lb[gi] + ee };
                            mb[gi] = if up { sv } else { mb[gi] };
                        }
                    }
                });
            });
        });
        if (c & 1) == 1 || c == c1 {
            let j = c >> 1;
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
            kcommon::unroll!(gi in 2 => {
                let st: Shared<f32> = smem.byte_add(G_ST + gi as u32 * 640).cast();
                let (qma, qla) = quad(ma[gi], la[gi]);
                let (qmb, qlb) = quad(mb[gi], lb[gi]);
                if t4 == 0 {
                    st.set((wid * 16 + ra) * 2, qma);
                    st.set((wid * 16 + ra) * 2 + 1, qla);
                    st.set((wid * 16 + rb_) * 2, qmb);
                    st.set((wid * 16 + rb_) * 2 + 1, qlb);
                }
            });
            sync_threads();
            kcommon::unroll!(gi in 2 => {
                let st: Shared<f32> = smem.byte_add(G_ST + gi as u32 * 640).cast();
                if t < m_rows[gi] {
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
                    *stat.add(((hh * nsup + j) * 32 + (row0[gi] + qr) * 2) as usize) = m;
                    *stat.add(((hh * nsup + j) * 32 + (row0[gi] + qr) * 2) as usize + 1) = ls;
                }
            });
            ma = [-1.0e30; 2];
            la = [0.0; 2];
            mb = [-1.0e30; 2];
            lb = [0.0; 2];
            if c < c1 {
                sync_threads();
                dmin = bias_table2(smem, proj, pos0, ntok, c + 1, sup_ce(j + 1), m_rows);
                sync_threads();
            }
        }
        c += 1;
    }
    let mut j = (c1 >> 1) + 1;
    while j < jn {
        inactive(j);
        j += 1;
    }
}

/// V pass, two row groups; shared memory: tile + 4 warps' stages (raw + the two groups' scores) + the partial-output exchange of one group (4 x 15 x 128 floats).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ink_dg2_v(gp: *mut f32, scr: *const u16, gst: *const f32, vc: *const u16, pos: *const u32, ring: u32, nsup: u32, ngr: u32,
    ntok: u32, nch: u32, qmode: u32) {
    crate::pdl_wait();
    crate::pdl_trigger();
    let smem: Shared<u8> = dyn_base();
    let q6 = (qmode & 1) != 0;
    let kvh = bid_y();
    let grp = bid_x();
    let jb = grp * 8;
    let wid = warp_id();
    let l = lane();
    let g = l >> 2;
    let t4 = l & 3;
    let t = tid_x();
    let pos0 = *pos;
    let m_rows = [12u32, 3 * (ntok - 4)];
    let row0 = [0u32, 4];
    let tile_hi = pos0 + ntok - 1;
    let jn = if jb + 8 < nsup { jb + 8 } else { nsup };
    let mut acc = [[0.0f32; 16]; 2];
    let zero = f32::from_bits(0);
    let (ra, rb_) = (g, g + 8);
    let va = [ra < m_rows[0], ra < m_rows[1]];
    let vb = [rb_ < m_rows[0], rb_ < m_rows[1]];
    let mut gm = [[0.0f32; 4]; 2];
    kcommon::unroll!(gi in 2 => {
        let rd = |r: u32, ok: bool| -> (f32, f32) {
            if ok {
                let e = ((((3 * kvh + r % 3) * 16 + row0[gi] + r / 3) * 2) as usize) as usize;
                (*gst.add(e), *gst.add(e + 1))
            } else {
                (0.0, 1.0)
            }
        };
        let (a1, a2) = rd(ra, va[gi]);
        let (b1, b2) = rd(rb_, vb[gi]);
        gm[gi] = [a1, a2, b1, b2];
    });
    let sbase = [(((kvh * nch) * 4 + wid) * 32 + l) * 8, ((((kvh * nch) + nch * 8) * 4 + wid) * 32 + l) * 8];
    let base = vc as *const u8;
    let nd: u32 = if q6 { DQ } else { 2 };
    let tile_bufs = if q6 { 1 } else { 2 };
    let tile0 = smem;
    let sc_per = 2 * SC_B;
    let stg_w = if q6 { DQ * (QREC + sc_per) } else { 2 * sc_per };
    let stg0 = smem.byte_add(tile_bufs * TILE_B + wid * stg_w);
    let ob_off = tile_bufs * TILE_B + 4 * stg_w;
    let ob: Shared<f32> = smem.byte_add(ob_off).cast();
    let c0 = 2 * jb;
    let c1 = if 2 * (jn - 1) + 1 < tile_hi / 64 { 2 * (jn - 1) + 1 } else { tile_hi / 64 };
    let last_j = c1 >> 1;
    if jb > tile_hi / 128 {
        kcommon::unroll!(gi in 2 => {
            kcommon::unroll!(r in 16 => {
                acc[gi][r] += zero;
            });
        });
    } else {
        let sc_at = |slot: u32| -> Shared<u8> { stg0.byte_add(if q6 { slot * (QREC + sc_per) + QREC } else { slot * sc_per }) };
        let fetch = |c: u32, slot: u32| {
            let dst = if q6 { stg0.byte_add(slot * (QREC + sc_per)) } else { tile0.byte_add(slot * TILE_B + wid * WTILE) };
            warp_fetch(q6, base, kvh, c, wid, ring, dst);
            cp_async_16(sc_at(slot).byte_add(16 * l).addr(), scr.add((((kvh * nch + c) * 1024) + (wid * 32 + l) * 8) as usize) as *const u8);
            cp_async_16(sc_at(slot).byte_add(SC_B + 16 * l).addr(), scr.add(((((kvh * nch + nch * 8) + c) * 1024) + (wid * 32 + l) * 8) as usize) as *const u8);
        };
        let mut k = 0;
        while k < nd {
            if c0 + k <= c1 {
                fetch(c0 + k, k);
            }
            cp_async_commit();
            k += 1;
        }
        let mut o = [[[0.0f32; 4]; 16]; 2];
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
            let sw = [<[u32; 4] as kcommon::smem::SmemIo>::lds(sc_at(slot).byte_add(16 * l).addr()), <[u32; 4] as kcommon::smem::SmemIo>::lds(sc_at(slot).byte_add(SC_B + 16 * l).addr())];
            if q6 {
                dequant_warp(tile, stg0.byte_add(slot * (QREC + sc_per)));
                sync_warp();
                if c + DQ <= c1 {
                    fetch(c + DQ, slot);
                }
                cp_async_commit();
            }
            kcommon::unroll!(gi in 2 => {
                let sc = |k: usize, hi: bool| -> f32 {
                    let b = if hi { sw[gi][k] >> 16 } else { sw[gi][k] & 0xFFFF };
                    if b == 0xFFFF { NEG } else { bf(b as u16) }
                };
                let mut s = [[NEG; 4]; 2];
                if va[gi] || vb[gi] {
                    s[0][0] = sc(0, false);
                    s[0][1] = sc(0, true);
                    s[0][2] = sc(1, false);
                    s[0][3] = sc(1, true);
                    s[1][0] = sc(2, false);
                    s[1][1] = sc(2, true);
                    s[1][2] = sc(3, false);
                    s[1][3] = sc(3, true);
                }
                let (gma, gla, gmb, glb) = (gm[gi][0], gm[gi][1], gm[gi][2], gm[gi][3]);
                let pv = |sv: f32, m: f32, ls: f32| -> f32 { if sv > NEG { rb(ex(sv - m) / ls) } else { 0.0 } };
                let pf = [
                    bf16::pack2(pv(s[0][0], gma, gla), pv(s[0][1], gma, gla)),
                    bf16::pack2(pv(s[0][2], gmb, glb), pv(s[0][3], gmb, glb)),
                    bf16::pack2(pv(s[1][0], gma, gla), pv(s[1][1], gma, gla)),
                    bf16::pack2(pv(s[1][2], gmb, glb), pv(s[1][3], gmb, glb)),
                ];
                kcommon::unroll!(dp in 8 => {
                    let vb4 = ldmatrix_x4_trans(tile.byte_add(((l & 7) + 8 * ((l >> 3) & 1)) * PITCH + (dp as u32 * 16 + (l >> 4) * 8) * 2).addr());
                    mma_bf16_16816(&mut o[gi][2 * dp], pf, [vb4[0], vb4[1]]);
                    mma_bf16_16816(&mut o[gi][2 * dp + 1], pf, [vb4[2], vb4[3]]);
                });
            });
            if !q6 {
                sync_warp();
                if c + 2 <= c1 {
                    fetch(c + 2, slot);
                }
                cp_async_commit();
            }
            if (c & 1) == 1 || c == c1 {
                kcommon::unroll!(gi in 2 => {
                    // partials of the 4 warps added in warp order, then into the group sum
                    let mr = m_rows[gi];
                    kcommon::unroll!(nt in 16 => {
                        let col = nt as u32 * 8 + 2 * t4;
                        if va[gi] {
                            ob.set(wid * (mr * 128) + ra * 128 + col, o[gi][nt][0]);
                            ob.set(wid * (mr * 128) + ra * 128 + col + 1, o[gi][nt][1]);
                        }
                        if vb[gi] {
                            ob.set(wid * (mr * 128) + rb_ * 128 + col, o[gi][nt][2]);
                            ob.set(wid * (mr * 128) + rb_ * 128 + col + 1, o[gi][nt][3]);
                        }
                    });
                    sync_threads();
                    kcommon::unroll!(r in 16 => {
                        if (r as u32) < mr {
                            let i = r as u32 * 128 + t;
                            let v = ((ob.at(i) + ob.at(mr * 128 + i)) + ob.at(2 * mr * 128 + i)) + ob.at(3 * mr * 128 + i);
                            acc[gi][r] += v;
                        }
                    });
                    sync_threads();
                });
                o = [[[0.0f32; 4]; 16]; 2];
            }
            c += 1;
        }
        let mut j = last_j + 1;
        while j < jn {
            kcommon::unroll!(gi in 2 => {
                kcommon::unroll!(r in 16 => {
                    acc[gi][r] += zero;
                });
            });
            j += 1;
        }
    }
    kcommon::unroll!(gi in 2 => {
        kcommon::unroll!(r in 16 => {
            if (r as u32) < m_rows[gi] {
                let (qr, ho) = (r as u32 / 3, r as u32 % 3);
                let hh = 3 * kvh + ho;
                *gp.add((((((row0[gi] + qr) * 24 + hh) * ngr + grp) * 128) + t) as usize) = acc[gi][r];
            }
        });
    });
}
