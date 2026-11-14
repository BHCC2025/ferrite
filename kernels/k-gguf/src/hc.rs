//! Hyper-connection kernels for qwen4exp decode (fused around the BF16 GEMVs; exact f32 math):
//!
//! - `ggq_hc_down_b<BC>` (norm + down|inject GEMV + scale + SiLU): per activation row r, with res f32 [4][2560]
//!   (row stride `ldres`) and norm weight nw f32 [10240]:
//!   xn[s][i] = res[s][i] * rsqrt(mean_i(res[s][:]^2) + eps) * nw[s*2560 + i];
//!   lo[o] = scale * (W[o] . xn) for o < n (W bf16 [n][10240], n = 324: down 320 + inject 4, or 320 for the head);
//!   lo[o] = silu(lo[o]) for o < n_silu (320). Output lo f32 [rows][ldlo]; xn written to `xn_out` [rows][10240]
//!   (spread over the blocks) when non-null. Any row count in one launch: the rows run in chunks of BC (weights stay
//!   in L2 / L1 between chunks).
//! - `ggq_hc_up_b<BC>` (up GEMV + sigmoid gate + stream mean): gate[s*2560 + j] = U[s*2560 + j] . lo[0..320];
//!   mixed[r][j] = mscale * sum_s xn[r][s][j] * sigmoid(gate[s*2560 + j]) (s = 0..3 in order). U bf16 [10240][320].
//!   A group of 8 lanes owns one j and its 4 stream rows, so the gates never leave the registers.
//!
//! Numerics (the bits do not depend on the row count, the chunking or the block shape): every (output row, activation
//! row) dot product of hc_down is split over 256 roles t (chunks of 8 values c = t, t + 256, ..; sequential FMAs in a
//! fixed element order), a warp butterfly over t % 32, then the 8 warp partials summed in order; the stream norms
//! use the same role split. hc_up: 8 lanes per gate row (chunks c = l8, l8 + 8, ..), a width-8 butterfly.
use crate::bf16::bf;
use kcommon::smem::Shared;
use kcommon::*;

const D: u32 = 2560;
const HD: u32 = 4 * D;
const KU: u32 = 320;

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + math::exp(-x))
}
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + math::exp(-x))
}

#[inline(always)]
unsafe fn pf_l2(p: *const u8) {
    core::arch::asm!("prefetch.global.L2 [{p}];", p = in(reg64) p, options(nostack, readonly));
}

/// Block = RG groups of 256 threads; group g owns the R output rows bid_x * R * RG + g * R ..; each thread of a group
/// is one role t. The stream norms of a chunk's rows are split over the groups (row b by group b % RG).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn hc_down<const BC: usize, const R: usize, const RG: usize>(lo: *mut f32, ldlo: u32, w: *const u16,
    res: *const f32, ldres: u32, nw: *const f32, xn_out: *mut f32, n: u32, n_silu: u32, scale: f32, eps: f32,
    up: *const u16, rows: u32) {
    let rpb = (R * RG) as u32;
    let rblk = bid_x() * rpb;
    if rblk >= n {
        return;
    }
    let tt = tid_x();
    let g = tt / 256;
    let t = tt % 256;
    let wg = t / 32; // warp within the group
    let row0 = rblk + g * R as u32;
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    // this group's weight rows into L2, then the block's slice of the up matrix (`up`, [10240][320] bf16, may be
    // null): the following hc_up then reads L2 and the DRAM stream does not stop at the kernel boundary
    unroll!(r in 16 => {
        if R > r {
            let rr = (row0 + r as u32).min(n - 1);
            let mut c = t * 8;
            while c < HD / 8 {
                pf_l2(w.add(rr as usize * HD as usize + 8 * c as usize) as *const u8);
                c += 256 * 8;
            }
        }
    });
    if !up.is_null() {
        let lines = HD * KU * 2 / 128;
        let (gn, b) = (gdim_x(), bid_x());
        let (l0, l1) = (lines * b / gn, lines * (b + 1) / gn);
        let mut l = l0 + tt;
        while l < l1 {
            pf_l2((up as *const u8).add(l as usize * 128));
            l += bdim_x();
        }
    }
    core::arch::asm!("griddepcontrol.wait;", options(nostack));
    let sm: Shared<f32> = smem::dyn_base::<f32>();
    let part = sm; // [8 warps][8][4] norm partials
    let inv = sm.add(8 * 8 * 4); // [8][4] inverse norms
    let red = sm.add(8 * 8 * 4 + 8 * 4); // [rpb][8][8 warps] dot partials
    let mut b0 = 0u32;
    while b0 < rows {
        let nb = (rows - b0).min(BC as u32);
        // stream norms (row b by group b % RG)
        let mut ss = [[0f32; 4]; BC];
        let mut c = t;
        while c < HD / 8 {
            let s = (8 * c / D) as usize;
            unroll!(b in 8 => {
                if BC > b && (b as u32) < nb && b as u32 % RG as u32 == g {
                    let xp = res.add((b0 as usize + b) * ldres as usize + 8 * c as usize) as *const u32;
                    let x0 = ldg_nc_v4(xp);
                    let x1 = ldg_nc_v4(xp.add(4));
                    let mut a = 0f32;
                    unroll!(q in 4 => {
                        let v0 = f32::from_bits(x0[q]);
                        let v1 = f32::from_bits(x1[q]);
                        a = math::fma(v0, v0, a);
                        a = math::fma(v1, v1, a);
                    });
                    unroll!(q in 4 => {
                        if s == q {
                            ss[b][q] += a;
                        }
                    });
                }
            });
            c += 256;
        }
        unroll!(b in 8 => {
            if BC > b && (b as u32) < nb && b as u32 % RG as u32 == g {
                unroll!(q in 4 => {
                    let v = warp::sum(ss[b][q]);
                    if lane() == 0 {
                        part.add((wg * 8 + b as u32) * 4 + q as u32).st(v);
                    }
                });
            }
        });
        sync_threads();
        if tt < nb * 4 {
            let (b, q) = (tt / 4, tt % 4);
            let mut v = 0f32;
            for wi in 0..8u32 {
                v += part.add((wi * 8 + b) * 4 + q).ld();
            }
            inv.add(b * 4 + q).st(math::rsqrt(v / D as f32 + eps));
        }
        sync_threads();
        let mut rs = [[0f32; 4]; BC];
        unroll!(b in 8 => {
            if BC > b {
                unroll!(q in 4 => { rs[b][q] = inv.add((b * 4 + q) as u32).ld(); });
            }
        });
        // GEMV on xn computed on the fly
        let mut acc = [[0f32; BC]; R];
        let mut c = t;
        while c < HD / 8 {
            let s = (8 * c / D) as usize;
            let mut v = [[0u32; 4]; R];
            unroll!(r in 16 => {
                if R > r {
                    let rr = (row0 + r as u32).min(n - 1);
                    v[r] = ldg_nc_v4(w.add(rr as usize * HD as usize + 8 * c as usize) as *const u32);
                }
            });
            let w0 = ldg_nc_v4(nw.add(8 * c as usize) as *const u32);
            let w1 = ldg_nc_v4(nw.add(8 * c as usize + 4) as *const u32);
            let nwv = [w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3]];
            let wr_xn = !xn_out.is_null() && g == 0 && c % gdim_x() == bid_x();
            unroll!(b in 8 => {
                if BC > b && (b as u32) < nb {
                    let xp = res.add((b0 as usize + b) * ldres as usize + 8 * c as usize) as *const u32;
                    let x0 = ldg_nc_v4(xp);
                    let x1 = ldg_nc_v4(xp.add(4));
                    let xr = [x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]];
                    let mut r_s = rs[b][0];
                    unroll!(q in 4 => { if s == q { r_s = rs[b][q]; } });
                    let mut xn = [0f32; 8];
                    unroll!(e in 8 => { xn[e] = f32::from_bits(xr[e]) * r_s * f32::from_bits(nwv[e]); });
                    if wr_xn {
                        let o = xn_out.add((b0 as usize + b) * HD as usize + 8 * c as usize) as *mut u32;
                        stg_v4(o, [xn[0].to_bits(), xn[1].to_bits(), xn[2].to_bits(), xn[3].to_bits()]);
                        stg_v4(o.add(4), [xn[4].to_bits(), xn[5].to_bits(), xn[6].to_bits(), xn[7].to_bits()]);
                    }
                    unroll!(r in 16 => {
                        if R > r {
                            let mut a = acc[r][b];
                            unroll!(q in 4 => {
                                let (wl, wh) = bf(v[r][q]);
                                a = math::fma(wl, xn[2 * q], a);
                                a = math::fma(wh, xn[2 * q + 1], a);
                            });
                            acc[r][b] = a;
                        }
                    });
                }
            });
            c += 256;
        }
        unroll!(r in 16 => {
            if R > r {
                unroll!(b in 8 => {
                    if BC > b && (b as u32) < nb {
                        let v = warp::sum(acc[r][b]);
                        if lane() == 0 {
                            red.add(((g * R as u32 + r as u32) * 8 + b as u32) * 8 + wg).st(v);
                        }
                    }
                });
            }
        });
        sync_threads();
        if tt < rpb * nb {
            let (r, b) = (tt / nb, tt % nb);
            let row = rblk + r;
            if row < n {
                let mut v = 0f32;
                for i in 0..8u32 {
                    v += red.add((r * 8 + b) * 8 + i).ld();
                }
                v *= scale;
                if row < n_silu {
                    v = silu(v);
                }
                *lo.add((b0 + b) as usize * ldlo as usize + row as usize) = v;
            }
        }
        sync_threads(); // shared memory reused by the next chunk
        b0 += BC as u32;
    }
}

/// Block = bdim_x / 8 values of j, 8 lanes each; the lane group of j computes the 4 gate rows s * 2560 + j (one weight
/// load serves 4 rows x BC activation rows). lo staged in shared memory (16-byte loads); the activation rows run in
/// chunks of BC (the weights again from L1 / L2).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn hc_up<const BC: usize>(mixed: *mut f32, ldm: u32, u: *const u16, lo: *const f32, ldlo: u32,
    xn: *const f32, ldxn: u32, mscale: f32, rows: u32) {
    let t = tid_x();
    let jb = bdim_x() / 8;
    let (jj, l8) = (t / 8, t % 8);
    let j = bid_x() * jb + jj;
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    if l8 < 5 {
        unroll!(s in 4 => {
            pf_l2((u.add(((s as u32 * D + j) * KU) as usize) as *const u8).add(l8 as usize * 128));
        });
    }
    core::arch::asm!("griddepcontrol.wait;", options(nostack));
    // stage: lo rows into shared memory (16-byte loads); a warp's x reads are then 8 distinct 16-byte words
    let xs: Shared<f32> = smem::dyn_base::<f32>();
    let mut i = t;
    while i < rows * (KU / 4) {
        let (b, q) = (i / (KU / 4), i % (KU / 4));
        let v = ld_v4(lo.add((b * ldlo + 4 * q) as usize) as *const u32);
        xs.add(b * KU + 4 * q).cast::<[f32; 4]>().st([f32::from_bits(v[0]), f32::from_bits(v[1]), f32::from_bits(v[2]),
            f32::from_bits(v[3])]);
        i += bdim_x();
    }
    sync_threads();
    let mut b0 = 0u32;
    while b0 < rows {
        let nb = (rows - b0).min(BC as u32);
        let mut acc = [[0f32; BC]; 4];
        let mut c = l8;
        while c < KU / 8 {
            let mut wv = [[0u32; 4]; 4];
            unroll!(s in 4 => {
                wv[s] = ldg_nc_v4(u.add(((s as u32 * D + j) * KU + 8 * c) as usize) as *const u32);
            });
            unroll!(b in 8 => {
                if BC > b && (b as u32) < nb {
                    let xp = xs.add((b0 + b as u32) * KU + 8 * c).cast::<[f32; 4]>();
                    let x0 = xp.ld();
                    let x1 = xp.add(1).ld();
                    let xv = [x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]];
                    unroll!(s in 4 => {
                        let mut a = acc[s][b];
                        unroll!(q in 4 => {
                            let (wl, wh) = bf(wv[s][q]);
                            a = math::fma(wl, xv[2 * q], a);
                            a = math::fma(wh, xv[2 * q + 1], a);
                        });
                        acc[s][b] = a;
                    });
                }
            });
            c += 8;
        }
        // every lane of the group gets each gate; lane l8 finishes activation row b0 + l8
        unroll!(b in 8 => {
            if BC > b && (b as u32) < nb {
                let mut gt = [0f32; 4];
                unroll!(s in 4 => { gt[s] = warp::sum_width::<8>(acc[s][b]); });
                if l8 == b as u32 {
                    let rb = b0 + b as u32;
                    let mut m = 0f32;
                    unroll!(s in 4 => {
                        m += *xn.add((rb * ldxn + s as u32 * D + j) as usize) * sigmoid(gt[s]);
                    });
                    *mixed.add((rb * ldm + j) as usize) = mscale * m;
                }
            }
        });
        b0 += BC as u32;
    }
}

/// Coherent 128-bit global load (lo is written by the previous kernel; L1-cached).
#[inline(always)]
unsafe fn ld_v4(p: *const u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    core::arch::asm!("ld.global.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
        p = in(reg64) p, a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d,
        options(readonly, nostack));
    [a, b, c, d]
}

/// Stream norm + xn for many activation rows (the first step of the role-warp hc_down path): block (s, b) =
/// (bid_x, bid_y), 256 threads. inv = rsqrt(mean(res[b][s][:]^2) + eps) with the hc_down role split (thread t sums
/// the chunks c = t + 256 i of stream s in order, warp butterfly, warps in order), then
/// xn[b][s * 2560 + i] = res * inv * nw (the same bits as hc_down's on-the-fly xn).
#[inline(always)]
pub unsafe fn hc_norm(xn: *mut f32, res: *const f32, ldres: u32, nw: *const f32, eps: f32) {
    let (s, b) = (bid_x(), bid_y());
    let t = tid_x();
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    core::arch::asm!("griddepcontrol.wait;", options(nostack));
    let rp = res.add(b as usize * ldres as usize);
    let mut ss = 0f32;
    let mut c = t;
    while c < HD / 8 {
        if c / (D / 8) == s {
            let xp = rp.add(8 * c as usize) as *const u32;
            let x0 = ldg_nc_v4(xp);
            let x1 = ldg_nc_v4(xp.add(4));
            let mut a = 0f32;
            unroll!(q in 4 => {
                let v0 = f32::from_bits(x0[q]);
                let v1 = f32::from_bits(x1[q]);
                a = math::fma(v0, v0, a);
                a = math::fma(v1, v1, a);
            });
            ss += a;
        }
        c += 256;
    }
    let sm: Shared<f32> = smem::dyn_base::<f32>();
    let v = warp::sum(ss);
    if lane() == 0 {
        sm.add(warp_id()).st(v);
    }
    sync_threads();
    let mut tot = 0f32;
    for wi in 0..8u32 {
        tot += sm.add(wi).ld();
    }
    let inv = math::rsqrt(tot / D as f32 + eps);
    let c = s * (D / 8) + t;
    if t < D / 8 {
        let xp = rp.add(8 * c as usize) as *const u32;
        let x0 = ldg_nc_v4(xp);
        let x1 = ldg_nc_v4(xp.add(4));
        let w0 = ldg_nc_v4(nw.add(8 * c as usize) as *const u32);
        let w1 = ldg_nc_v4(nw.add(8 * c as usize + 4) as *const u32);
        let xr = [x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]];
        let nwv = [w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3]];
        let mut o = [0u32; 8];
        unroll!(e in 8 => { o[e] = (f32::from_bits(xr[e]) * inv * f32::from_bits(nwv[e])).to_bits(); });
        let op = xn.add(b as usize * HD as usize + 8 * c as usize) as *mut u32;
        stg_v4(op, [o[0], o[1], o[2], o[3]]);
        stg_v4(op.add(4), [o[4], o[5], o[6], o[7]]);
    }
    // the second half of a stream (320 chunks > 256 threads)
    let c2 = c + 256;
    if t + 256 < D / 8 {
        let xp = rp.add(8 * c2 as usize) as *const u32;
        let x0 = ldg_nc_v4(xp);
        let x1 = ldg_nc_v4(xp.add(4));
        let w0 = ldg_nc_v4(nw.add(8 * c2 as usize) as *const u32);
        let w1 = ldg_nc_v4(nw.add(8 * c2 as usize + 4) as *const u32);
        let xr = [x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]];
        let nwv = [w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3]];
        let mut o = [0u32; 8];
        unroll!(e in 8 => { o[e] = (f32::from_bits(xr[e]) * inv * f32::from_bits(nwv[e])).to_bits(); });
        let op = xn.add(b as usize * HD as usize + 8 * c2 as usize) as *mut u32;
        stg_v4(op, [o[0], o[1], o[2], o[3]]);
        stg_v4(op.add(4), [o[4], o[5], o[6], o[7]]);
    }
}

/// Rows per role-warp block of hc_dw (9 warps x 4 rows).
pub const DW_ROWS: u32 = 36;

/// Role-warp hc_down for many activation rows (after hc_norm): block = (role warp wr = bid_x % 8, row group
/// rg = bid_x / 8 of 36 rows), 9 warps x 4 rows; lane = role t = 32 wr + lane, so a block reads only 1/8 of each
/// xn row (shared by its 9 warps through L1); the weights are read per chunk of BC rows (L2 after the first). The 8 warp
/// partials of a row go to `part` [row][32][8]; the last of the 8 blocks of a row group sums them in order (fixed:
/// the same bits as hc_down), scales, applies SiLU and writes lo. `cnt` [row groups] counters (zero; reset here).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn hc_dw<const BC: usize>(part: *mut f32, cnt: *mut u32, lo: *mut f32, ldlo: u32, w: *const u16,
    xn: *const f32, n: u32, n_silu: u32, scale: f32, up: *const u16, rows: u32) {
    let wr = bid_x() % 8;
    let rg = bid_x() / 8;
    let tt = tid_x();
    let (wi, ln) = (tt / 32, lane());
    let t = wr * 32 + ln;
    let row0 = rg * DW_ROWS + wi * 4;
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    if ln % 8 == 0 {
        unroll!(r in 4 => {
            let rr = (row0 + r as u32).min(n - 1);
            unroll!(i in [0, 1, 2, 3, 4] => {
                pf_l2(w.add(rr as usize * HD as usize + 8 * (t + 256 * i as u32) as usize) as *const u8);
            });
        });
    }
    if !up.is_null() {
        let lines = HD * KU * 2 / 128;
        let (gn, b) = (gdim_x(), bid_x());
        let (l0, l1) = (lines * b / gn, lines * (b + 1) / gn);
        let mut l = l0 + tt;
        while l < l1 {
            pf_l2((up as *const u8).add(l as usize * 128));
            l += bdim_x();
        }
    }
    core::arch::asm!("griddepcontrol.wait;", options(nostack));
    let mut b0 = 0u32;
    while b0 < rows {
        let nb = (rows - b0).min(BC as u32);
        let mut acc = [[0f32; BC]; 4];
        let mut c = t;
        while c < HD / 8 {
            let mut wv = [[0u32; 4]; 4];
            unroll!(r in 4 => {
                let rr = (row0 + r as u32).min(n - 1);
                wv[r] = ldg_nc_v4(w.add(rr as usize * HD as usize + 8 * c as usize) as *const u32);
            });
            unroll!(b in 8 => {
                if BC > b && (b as u32) < nb {
                    let xp = xn.add((b0 as usize + b) * HD as usize + 8 * c as usize) as *const u32;
                    let x0 = ldg_nc_v4(xp);
                    let x1 = ldg_nc_v4(xp.add(4));
                    let xv = [x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]];
                    unroll!(r in 4 => {
                        let mut a = acc[r][b];
                        unroll!(q in 4 => {
                            let (wl, wh) = bf(wv[r][q]);
                            a = math::fma(wl, f32::from_bits(xv[2 * q]), a);
                            a = math::fma(wh, f32::from_bits(xv[2 * q + 1]), a);
                        });
                        acc[r][b] = a;
                    });
                }
            });
            c += 256;
        }
        unroll!(r in 4 => {
            unroll!(b in 8 => {
                if BC > b && (b as u32) < nb {
                    let v = warp::sum(acc[r][b]);
                    let row = row0 + r as u32;
                    if ln == 0 && row < n {
                        *part.add(((row * 32 + b0 + b as u32) * 8 + wr) as usize) = v;
                    }
                }
            });
        });
        b0 += BC as u32;
    }
    // the last of the row group's 8 blocks reduces
    fence_block();
    core::arch::asm!("fence.acq_rel.gpu;", options(nostack));
    sync_threads();
    let last: Shared<u32> = smem::dyn_base::<u32>();
    if tt == 0 {
        let old = atomic_add_u32(cnt.add(rg as usize), 1);
        last.st(if old == 7 { 1 } else { 0 });
    }
    sync_threads();
    if last.ld() == 1 {
        core::arch::asm!("fence.acq_rel.gpu;", options(nostack));
        let mut idx = tt;
        while idx < DW_ROWS * rows {
            let (r, b) = (idx / rows, idx % rows);
            let row = rg * DW_ROWS + r;
            if row < n {
                let mut v = 0f32;
                for k in 0..8u32 {
                    v += core::ptr::read_volatile(part.add(((row * 32 + b) * 8 + k) as usize));
                }
                v *= scale;
                if row < n_silu {
                    v = silu(v);
                }
                *lo.add((b * ldlo + row) as usize) = v;
            }
            idx += bdim_x();
        }
        if tt == 0 {
            *cnt.add(rg as usize) = 0;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_hc_norm(xn: *mut f32, res: *const f32, ldres: u32, nw: *const f32, eps: f32) {
    hc_norm(xn, res, ldres, nw, eps)
}

macro_rules! hc_dw_kernels {
    ($($name:ident = $b:literal);*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(part: *mut f32, cnt: *mut u32, lo: *mut f32, ldlo: u32, w: *const u16,
            xn: *const f32, n: u32, n_silu: u32, scale: f32, up: *const u16, rows: u32) {
            hc_dw::<$b>(part, cnt, lo, ldlo, w, xn, n, n_silu, scale, up, rows)
        }
    )*};
}
hc_dw_kernels!(ggq_hc_dw_b1 = 1; ggq_hc_dw_b2 = 2; ggq_hc_dw_b3 = 3; ggq_hc_dw_b4 = 4; ggq_hc_dw_b5 = 5; ggq_hc_dw_b6 = 6;
    ggq_hc_dw_b7 = 7; ggq_hc_dw_b8 = 8);

macro_rules! hc_kernels {
    ($($down:ident, $up:ident = $b:literal);*) => {$(
        /// cfg: 0 = 4 rows x 1 group, 1 = 9 x 1, 2 = 8 x 1, 3 = 3 rows x 3 groups (768 threads), 4 = 3 x 1, 5 = 1 x 3
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $down(lo: *mut f32, ldlo: u32, w: *const u16, res: *const f32, ldres: u32,
            nw: *const f32, xn_out: *mut f32, n: u32, n_silu: u32, scale: f32, eps: f32, cfg: u32, up: *const u16,
            rows: u32) {
            match cfg {
                1 => hc_down::<$b, 9, 1>(lo, ldlo, w, res, ldres, nw, xn_out, n, n_silu, scale, eps, up, rows),
                2 => hc_down::<$b, 8, 1>(lo, ldlo, w, res, ldres, nw, xn_out, n, n_silu, scale, eps, up, rows),
                3 => hc_down::<$b, 3, 3>(lo, ldlo, w, res, ldres, nw, xn_out, n, n_silu, scale, eps, up, rows),
                4 => hc_down::<$b, 3, 1>(lo, ldlo, w, res, ldres, nw, xn_out, n, n_silu, scale, eps, up, rows),
                5 => hc_down::<$b, 1, 3>(lo, ldlo, w, res, ldres, nw, xn_out, n, n_silu, scale, eps, up, rows),
                _ => hc_down::<$b, 4, 1>(lo, ldlo, w, res, ldres, nw, xn_out, n, n_silu, scale, eps, up, rows),
            }
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $up(mixed: *mut f32, ldm: u32, u: *const u16, lo: *const f32, ldlo: u32,
            xn: *const f32, ldxn: u32, mscale: f32, rows: u32) {
            hc_up::<$b>(mixed, ldm, u, lo, ldlo, xn, ldxn, mscale, rows)
        }
    )*};
}
hc_kernels!(ggq_hc_down_b1, ggq_hc_up_b1 = 1; ggq_hc_down_b2, ggq_hc_up_b2 = 2; ggq_hc_down_b3, ggq_hc_up_b3 = 3;
    ggq_hc_down_b4, ggq_hc_up_b4 = 4; ggq_hc_down_b5, ggq_hc_up_b5 = 5; ggq_hc_down_b6, ggq_hc_up_b6 = 6;
    ggq_hc_down_b7, ggq_hc_up_b7 = 7; ggq_hc_down_b8, ggq_hc_up_b8 = 8);
