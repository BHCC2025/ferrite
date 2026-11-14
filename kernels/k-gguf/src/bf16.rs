//! BF16-weight decode GEMVs (exact: bf16 weights x f32 activations, f32 accumulate) for 1..8 activation rows:
//! `y[b][n] (= or +=) sum_k W[n][k] * x[b][k]`, W bf16 [N][K] row-major.
//!
//! - `ggq_bf16_wide_b<B>`: long rows (K >= 2048, e.g. hc down [10240 -> 324]): one block of 256 threads per output
//!   row, 16-byte weight loads strided over K, block reduction in shared memory (fixed order: deterministic).
//! - `ggq_bf16_narrow_b<B>`: short rows (K <= 1024, e.g. hc up [320 -> 10240]): 8 lanes per row, 32 rows per block,
//!   the activations staged in shared memory once per block.
use kcommon::smem::Shared;
use kcommon::*;

/// PDL: let the next programmatic launch start, prefetch this thread's weights into L2, then wait for the previous
/// grid (whose output may be our x). No-ops for plain launches.
#[inline(always)]
pub unsafe fn pdl_enter(w: *const u16, row0: u32, nrows: u32, n: u32, k: u32, c0: u32, cstep: u32, nch: u32) {
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    for r in 0..nrows {
        let rr = (row0 + r).min(n - 1);
        let mut c = c0;
        while c < nch {
            let p = w.add(rr as usize * k as usize + 8 * c as usize);
            core::arch::asm!("prefetch.global.L2 [{p}];", p = in(reg64) p, options(nostack, readonly));
            c += cstep * 8; // one prefetch per 128-byte line
        }
    }
    core::arch::asm!("griddepcontrol.wait;", options(nostack));
}

#[inline(always)]
pub fn bf(w: u32) -> (f32, f32) {
    (f32::from_bits(w << 16), f32::from_bits(w & 0xffff_0000))
}

/// Wide rows: one block per R output rows (rows bid_x * R ..), 256 threads strided over K; every activation load is
/// shared by the R rows. Fixed-order block reduction.
#[inline(always)]
pub unsafe fn wide<const B: usize, const R: usize>(y: *mut f32, w: *const u16, x: *const f32, ldx: u32, n: u32, k: u32,
    add: u32) {
    let row0 = bid_x() * R as u32;
    if row0 >= n {
        return;
    }
    let t = tid_x();
    let nt = bdim_x();
    pdl_enter(w, row0, R as u32, n, k, t * 8, nt, k / 8);
    let mut acc = [[0f32; B]; R];
    let nch = k / 8;
    let mut c = t;
    while c < nch {
        let mut v = [[0u32; 4]; R];
        unroll!(r in 16 => {
            if R > r {
                let rr = (row0 + r as u32).min(n - 1);
                v[r] = ldg_stream_v4(w.add(rr as usize * k as usize + 8 * c as usize) as *const u32);
            }
        });
        unroll!(b in 8 => {
            if B > b {
                let xp = x.add(b * ldx as usize + 8 * c as usize) as *const u32;
                let x0 = ldg_nc_v4(xp);
                let x1 = ldg_nc_v4(xp.add(4));
                let xv = [x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]];
                unroll!(r in 16 => {
                    if R > r {
                        let mut a = acc[r][b];
                        unroll!(q in 4 => {
                            let (wl, wh) = bf(v[r][q]);
                            a = math::fma(wl, f32::from_bits(xv[2 * q]), a);
                            a = math::fma(wh, f32::from_bits(xv[2 * q + 1]), a);
                        });
                        acc[r][b] = a;
                    }
                });
            }
        });
        c += nt;
    }
    let red: Shared<f32> = smem::dyn_base::<f32>(); // R(<=16) * 8 rows * 8 warps f32 (launch smem 4 KB)
    unroll!(r in 16 => {
        if R > r {
            unroll!(b in 8 => {
                if B > b {
                    let s = warp::sum(acc[r][b]);
                    if lane() == 0 {
                        red.add(((r * 8 + b) as u32) * 8 + warp_id()).st(s);
                    }
                }
            });
        }
    });
    sync_threads();
    let nw = nt / 32;
    if t < (R * B) as u32 {
        let (r, b) = (t / B as u32, t % B as u32);
        let row = row0 + r;
        if row < n {
            let mut s = 0f32;
            for i in 0..nw {
                s += red.add((r * 8 + b) * 8 + i).ld();
            }
            let o = y.add(b as usize * n as usize + row as usize);
            if add != 0 { *o += s } else { *o = s }
        }
    }
}

/// Short rows: block = 256 threads = 8 warps x 4 rows (8 lanes each); x staged in dynamic smem (B x K f32).
#[inline(always)]
pub unsafe fn narrow<const B: usize>(y: *mut f32, w: *const u16, x: *const f32, ldx: u32, n: u32, k: u32, add: u32) {
    let xs: Shared<f32> = smem::dyn_base::<f32>(); // B * K f32 (launch smem)
    let t = tid_x();
    pdl_enter(w, bid_x() * 32 + t / 8, 1, n, k, (t % 8) * 8, 1, k / 8);
    let mut i = t;
    while i < B as u32 * k {
        let b = i / k;
        xs.add(i).st(*x.add((b * ldx + i % k) as usize));
        i += bdim_x();
    }
    sync_threads();
    let row = bid_x() * 32 + t / 8;
    let l8 = t % 8;
    let nch = k / 8;
    let mut acc = [0f32; B];
    if row < n {
        let wr = w.add(row as usize * k as usize);
        let mut c = l8;
        while c < nch {
            let v = ldg_stream_v4(wr.add(8 * c as usize) as *const u32);
            unroll!(b in 8 => {
                if B > b {
                    let xp = xs.add(b as u32 * k + 8 * c).cast::<[f32; 4]>();
                    let x0 = xp.ld();
                    let x1 = xp.add(1).ld();
                    let mut a = acc[b];
                    let (w0, w1) = bf(v[0]);
                    let (w2, w3) = bf(v[1]);
                    let (w4, w5) = bf(v[2]);
                    let (w6, w7) = bf(v[3]);
                    a = math::fma(w0, x0[0], a);
                    a = math::fma(w1, x0[1], a);
                    a = math::fma(w2, x0[2], a);
                    a = math::fma(w3, x0[3], a);
                    a = math::fma(w4, x1[0], a);
                    a = math::fma(w5, x1[1], a);
                    a = math::fma(w6, x1[2], a);
                    a = math::fma(w7, x1[3], a);
                    acc[b] = a;
                }
            });
            c += 8;
        }
    }
    unroll!(b in 8 => {
        if B > b {
            let s = warp::sum_width::<8>(acc[b]);
            if row < n && l8 == 0 {
                let o = y.add(b * n as usize + row as usize);
                if add != 0 { *o += s } else { *o = s }
            }
        }
    });
}

/// Split-K for long rows and few outputs (hc down [10240 -> 324]): block (x = row group of 32 rows, y = K slice of
/// `ksl` values): the x slice is read once into shared memory, 8 warps x 4 rows, 8 lanes per row; partial sums go to
/// `part[slice][b][row]`; the last block of a row group (counter) adds the slices in fixed order (deterministic) and
/// resets the counter (graph-replay safe).
#[inline(always)]
pub unsafe fn splitk<const B: usize>(y: *mut f32, w: *const u16, x: *const f32, ldx: u32, n: u32, k: u32, add: u32,
    ksl: u32, part: *mut f32, cnt: *mut u32) {
    let xs: Shared<f32> = smem::dyn_base::<f32>(); // B * ksl f32
    let t = tid_x();
    let rg = bid_x();
    let ks = bid_y();
    let nks = gdim_y();
    let k0 = ks * ksl;
    let row = rg * 32 + t / 8;
    let l8 = t % 8;
    pdl_enter(w.add(k0 as usize), row, 1, n, k, l8 * 8, 1, ksl / 8);
    let mut i = t;
    while i < B as u32 * ksl {
        let b = i / ksl;
        xs.add(i).st(*x.add((b * ldx + k0 + i % ksl) as usize));
        i += bdim_x();
    }
    sync_threads();
    let mut acc = [0f32; B];
    if row < n {
        let wr = w.add(row as usize * k as usize + k0 as usize);
        let mut c = l8;
        while c < ksl / 8 {
            let v = ldg_stream_v4(wr.add(8 * c as usize) as *const u32);
            unroll!(b in 8 => {
                if B > b {
                    let xp = xs.add(b as u32 * ksl + 8 * c).cast::<[f32; 4]>();
                    let x0 = xp.ld();
                    let x1 = xp.add(1).ld();
                    let xv = [x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]];
                    let mut a = acc[b];
                    unroll!(q in 4 => {
                        let (wl, wh) = bf(v[q]);
                        a = math::fma(wl, xv[2 * q], a);
                        a = math::fma(wh, xv[2 * q + 1], a);
                    });
                    acc[b] = a;
                }
            });
            c += 8;
        }
    }
    unroll!(b in 8 => {
        if B > b {
            let s = warp::sum_width::<8>(acc[b]);
            if row < n && l8 == 0 {
                *part.add(((ks as usize * B + b) * n as usize) + row as usize) = s;
            }
        }
    });
    // last block of this row group reduces
    fence_block();
    core::arch::asm!("fence.acq_rel.gpu;", options(nostack));
    sync_threads();
    let last: Shared<u32> = smem::dyn_base::<u32>();
    if t == 0 {
        let old = atomic_add_u32(cnt.add(rg as usize), 1);
        last.st(if old == nks - 1 { 1 } else { 0 });
    }
    sync_threads();
    if last.ld() == 1 {
        core::arch::asm!("fence.acq_rel.gpu;", options(nostack));
        let i = t;
        if i < 32 * B as u32 {
            let (b, r) = (i / 32, i % 32);
            let rr = rg * 32 + r;
            if rr < n {
                let mut sum = 0f32;
                for s2 in 0..nks {
                    sum += core::ptr::read_volatile(part.add(((s2 as usize * B + b as usize) * n as usize) + rr as usize));
                }
                let o = y.add(b as usize * n as usize + rr as usize);
                if add != 0 { *o += sum } else { *o = sum }
            }
        }
        if t == 0 {
            *cnt.add(rg as usize) = 0;
        }
    }
}

macro_rules! splitk_kernels {
    ($($name:ident = $b:literal);*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(y: *mut f32, w: *const u16, x: *const f32, ldx: u32, n: u32, k: u32,
            add: u32, ksl: u32, part: *mut f32, cnt: *mut u32) {
            splitk::<$b>(y, w, x, ldx, n, k, add, ksl, part, cnt)
        }
    )*};
}
splitk_kernels!(ggq_bf16_splitk_b1 = 1; ggq_bf16_splitk_b2 = 2; ggq_bf16_splitk_b3 = 3; ggq_bf16_splitk_b4 = 4;
    ggq_bf16_splitk_b5 = 5; ggq_bf16_splitk_b6 = 6; ggq_bf16_splitk_b7 = 7; ggq_bf16_splitk_b8 = 8);

macro_rules! bf16_kernels {
    ($($wide:ident, $wider:ident, $narrow:ident = $b:literal);*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $wide(y: *mut f32, w: *const u16, x: *const f32, ldx: u32, n: u32, k: u32,
            add: u32, rpb: u32) {
            match rpb {
                4 => wide::<$b, 4>(y, w, x, ldx, n, k, add),
                2 => wide::<$b, 2>(y, w, x, ldx, n, k, add),
                _ => wide::<$b, 1>(y, w, x, ldx, n, k, add),
            }
        }
        /// 8 or 9 rows per block (separate entry: its register count does not cap the small-R occupancy)
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $wider(y: *mut f32, w: *const u16, x: *const f32, ldx: u32, n: u32, k: u32,
            add: u32, rpb: u32) {
            match rpb {
                9 => wide::<$b, 9>(y, w, x, ldx, n, k, add),
                _ => wide::<$b, 8>(y, w, x, ldx, n, k, add),
            }
        }
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $narrow(y: *mut f32, w: *const u16, x: *const f32, ldx: u32, n: u32, k: u32,
            add: u32, _rpb: u32) {
            narrow::<$b>(y, w, x, ldx, n, k, add)
        }
    )*};
}
bf16_kernels!(ggq_bf16_wide_b1, ggq_bf16_wider_b1, ggq_bf16_narrow_b1 = 1; ggq_bf16_wide_b2, ggq_bf16_wider_b2, ggq_bf16_narrow_b2 = 2;
    ggq_bf16_wide_b3, ggq_bf16_wider_b3, ggq_bf16_narrow_b3 = 3; ggq_bf16_wide_b4, ggq_bf16_wider_b4, ggq_bf16_narrow_b4 = 4;
    ggq_bf16_wide_b5, ggq_bf16_wider_b5, ggq_bf16_narrow_b5 = 5; ggq_bf16_wide_b6, ggq_bf16_wider_b6, ggq_bf16_narrow_b6 = 6;
    ggq_bf16_wide_b7, ggq_bf16_wider_b7, ggq_bf16_narrow_b7 = 7; ggq_bf16_wide_b8, ggq_bf16_wider_b8, ggq_bf16_narrow_b8 = 8);
