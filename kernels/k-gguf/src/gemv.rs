//! Decode GEMV on q8_1 activations (dp4a): `y[b][n] (=|+=) sum_k W[n][k] x[b][k]` for B = 1..8 activation rows.
//!
//! Work split: `lpr` lanes per weight row (power of two, 1..32), so a warp covers `32 / lpr` rows; lane `i` of a
//! row takes 32-value sub-blocks `i, i + lpr, ...` (`nsb = K / 32` of them), decodes each once (vq) and dots it
//! with every activation row. Activations come from L1/L2 (`ld.global.nc`), reused by every warp of the SM.
//!
//! Activation layout (written by `ggq_q8_1_*`): `aq` int8 [B][K], `ad` f32 [B][K/32] (scale), `as_` f32 [B][K/32]
//! (= f16-rounded sum of the block's float inputs, as ggml's ds.y, for the `m` terms of Q4_K/Q5_K).
use crate::vq::*;
use kcommon::*;

/// One weight-row dot for B activation rows, accumulated into `acc`.
#[inline(always)]
pub unsafe fn row_dot<V: Vq, const B: usize>(row: *const u8, j0: u32, step: u32, nsb: u32, aq: *const i8,
    ad: *const f32, as_: *const f32, k: u32, acc: &mut [f32; B]) {
    let ar: [u32; B] = core::array::from_fn(|b| b as u32);
    row_dot_idx::<V, B>(row, j0, step, nsb, aq, ad, as_, k, &ar, acc)
}

/// [`row_dot`] with arbitrary activation rows: activation `b` is row `ar[b]` of the q8 buffers.
#[inline(always)]
pub unsafe fn row_dot_idx<V: Vq, const B: usize>(row: *const u8, j0: u32, step: u32, nsb: u32, aq: *const i8,
    ad: *const f32, as_: *const f32, k: u32, ar: &[u32; B], acc: &mut [f32; B]) {
    // two sub-blocks decoded before either is used: twice the weight loads in flight per lane
    let mut j = j0;
    while j + step < nsb {
        let s0 = V::sb(row, j);
        let s1 = V::sb(row, j + step);
        dot_sb::<V, B>(&s0, j, nsb, aq, ad, as_, k, ar, acc);
        dot_sb::<V, B>(&s1, j + step, nsb, aq, ad, as_, k, ar, acc);
        j += 2 * step;
    }
    if j < nsb {
        let s0 = V::sb(row, j);
        dot_sb::<V, B>(&s0, j, nsb, aq, ad, as_, k, ar, acc);
    }
}

#[inline(always)]
unsafe fn dot_sb<V: Vq, const B: usize>(s: &Sb, j: u32, nsb: u32, aq: *const i8, ad: *const f32, as_: *const f32,
    k: u32, ar: &[u32; B], acc: &mut [f32; B]) {
    for b in 0..B {
        let r = ar[b] as usize;
        let a = ldg_nc_v4(aq.add(r * k as usize + 32 * j as usize) as *const u32);
        let c = ldg_nc_v4(aq.add(r * k as usize + 32 * j as usize + 16) as *const u32);
        let d8 = *ad.add(r * nsb as usize + j as usize);
        let mut v = (d8 * s.base) * sb_dot::<V>(s, &a, &c);
        if V::HAS_M {
            v -= s.m * *as_.add(r * nsb as usize + j as usize);
        }
        acc[b] += v;
    }
}

#[inline(always)]
pub fn reduce_lanes(mut v: f32, lpr: u32) -> f32 {
    let mut m = lpr >> 1;
    while m > 0 {
        v += warp::shfl_xor(v, m);
        m >>= 1;
    }
    v
}

/// Dense GEMV body. Grid: x = ceil(N / rows_per_block); block = 32 * warps.
#[inline(always)]
pub unsafe fn gemv<V: Vq, const B: usize>(y: *mut f32, w: *const u8, row_bytes: u64, aq: *const i8, ad: *const f32,
    as_: *const f32, k: u32, n: u32, lpr: u32, add: u32) {
    let rpw = 32 / lpr;
    let row = (bid_x() * nwarps() + warp_id()) * rpw + lane() / lpr;
    let nsb = k / 32;
    let mut acc = [0f32; B];
    if row < n {
        row_dot::<V, B>(w.add(row as usize * row_bytes as usize), lane() % lpr, lpr, nsb, aq, ad, as_, k, &mut acc);
    }
    for b in 0..B {
        let v = reduce_lanes(acc[b], lpr);
        if row < n && lane() % lpr == 0 {
            let o = y.add(b * n as usize + row as usize);
            if add != 0 { *o += v } else { *o = v }
        }
    }
}

macro_rules! gemv_kernels {
    ($v:ty, $($name:ident = $b:literal),*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(y: *mut f32, w: *const u8, row_bytes: u64, aq: *const i8,
            ad: *const f32, as_: *const f32, k: u32, n: u32, lpr: u32, add: u32) {
            gemv::<$v, $b>(y, w, row_bytes, aq, ad, as_, k, n, lpr, add)
        }
    )*};
}
macro_rules! gemv_all_b {
    ($v:ty, $p:ident) => {
        paste_b!($v, $p);
    };
}
// names: ggq_gemv_<ty>_b<B>
macro_rules! paste_b {
    ($v:ty, q8_0) => { gemv_kernels!($v, ggq_gemv_q8_0_b1 = 1, ggq_gemv_q8_0_b2 = 2, ggq_gemv_q8_0_b3 = 3, ggq_gemv_q8_0_b4 = 4, ggq_gemv_q8_0_b5 = 5, ggq_gemv_q8_0_b6 = 6, ggq_gemv_q8_0_b7 = 7, ggq_gemv_q8_0_b8 = 8); };
    ($v:ty, q2_0) => { gemv_kernels!($v, ggq_gemv_q2_0_b1 = 1, ggq_gemv_q2_0_b2 = 2, ggq_gemv_q2_0_b3 = 3, ggq_gemv_q2_0_b4 = 4, ggq_gemv_q2_0_b5 = 5, ggq_gemv_q2_0_b6 = 6, ggq_gemv_q2_0_b7 = 7, ggq_gemv_q2_0_b8 = 8); };
    ($v:ty, iq4_nl) => { gemv_kernels!($v, ggq_gemv_iq4_nl_b1 = 1, ggq_gemv_iq4_nl_b2 = 2, ggq_gemv_iq4_nl_b3 = 3, ggq_gemv_iq4_nl_b4 = 4, ggq_gemv_iq4_nl_b5 = 5, ggq_gemv_iq4_nl_b6 = 6, ggq_gemv_iq4_nl_b7 = 7, ggq_gemv_iq4_nl_b8 = 8); };
    ($v:ty, iq4_xs) => { gemv_kernels!($v, ggq_gemv_iq4_xs_b1 = 1, ggq_gemv_iq4_xs_b2 = 2, ggq_gemv_iq4_xs_b3 = 3, ggq_gemv_iq4_xs_b4 = 4, ggq_gemv_iq4_xs_b5 = 5, ggq_gemv_iq4_xs_b6 = 6, ggq_gemv_iq4_xs_b7 = 7, ggq_gemv_iq4_xs_b8 = 8); };
    ($v:ty, q4_k) => { gemv_kernels!($v, ggq_gemv_q4_k_b1 = 1, ggq_gemv_q4_k_b2 = 2, ggq_gemv_q4_k_b3 = 3, ggq_gemv_q4_k_b4 = 4, ggq_gemv_q4_k_b5 = 5, ggq_gemv_q4_k_b6 = 6, ggq_gemv_q4_k_b7 = 7, ggq_gemv_q4_k_b8 = 8); };
    ($v:ty, q5_k) => { gemv_kernels!($v, ggq_gemv_q5_k_b1 = 1, ggq_gemv_q5_k_b2 = 2, ggq_gemv_q5_k_b3 = 3, ggq_gemv_q5_k_b4 = 4, ggq_gemv_q5_k_b5 = 5, ggq_gemv_q5_k_b6 = 6, ggq_gemv_q5_k_b7 = 7, ggq_gemv_q5_k_b8 = 8); };
    ($v:ty, q6_k) => { gemv_kernels!($v, ggq_gemv_q6_k_b1 = 1, ggq_gemv_q6_k_b2 = 2, ggq_gemv_q6_k_b3 = 3, ggq_gemv_q6_k_b4 = 4, ggq_gemv_q6_k_b5 = 5, ggq_gemv_q6_k_b6 = 6, ggq_gemv_q6_k_b7 = 7, ggq_gemv_q6_k_b8 = 8); };
    ($v:ty, iq2_s) => { gemv_kernels!($v, ggq_gemv_iq2_s_b1 = 1, ggq_gemv_iq2_s_b2 = 2, ggq_gemv_iq2_s_b3 = 3, ggq_gemv_iq2_s_b4 = 4, ggq_gemv_iq2_s_b5 = 5, ggq_gemv_iq2_s_b6 = 6, ggq_gemv_iq2_s_b7 = 7, ggq_gemv_iq2_s_b8 = 8); };
    ($v:ty, iq3_xxs) => { gemv_kernels!($v, ggq_gemv_iq3_xxs_b1 = 1, ggq_gemv_iq3_xxs_b2 = 2, ggq_gemv_iq3_xxs_b3 = 3, ggq_gemv_iq3_xxs_b4 = 4, ggq_gemv_iq3_xxs_b5 = 5, ggq_gemv_iq3_xxs_b6 = 6, ggq_gemv_iq3_xxs_b7 = 7, ggq_gemv_iq3_xxs_b8 = 8); };
    ($v:ty, iq3_s) => { gemv_kernels!($v, ggq_gemv_iq3_s_b1 = 1, ggq_gemv_iq3_s_b2 = 2, ggq_gemv_iq3_s_b3 = 3, ggq_gemv_iq3_s_b4 = 4, ggq_gemv_iq3_s_b5 = 5, ggq_gemv_iq3_s_b6 = 6, ggq_gemv_iq3_s_b7 = 7, ggq_gemv_iq3_s_b8 = 8); };
}
gemv_all_b!(Q8_0, q8_0);
gemv_all_b!(Q2_0, q2_0);
gemv_all_b!(Iq4Nl, iq4_nl);
gemv_all_b!(Iq4Xs, iq4_xs);
gemv_all_b!(Q4K, q4_k);
gemv_all_b!(Q5K, q5_k);
gemv_all_b!(Q6K, q6_k);
gemv_all_b!(Iq2S, iq2_s);
gemv_all_b!(Iq3Xxs, iq3_xxs);
gemv_all_b!(Iq3S, iq3_s);


/// Prefetch the next sub-block in the row loops (superblock types only: it slows the small-block down types).
pub const PREFETCH: bool = true;

#[inline(always)]
pub fn prefetch_l1(p: *const u8) {
    unsafe { core::arch::asm!("prefetch.global.L1 [{p}];", p = in(reg64) p, options(nostack, readonly)) }
}

/// R weight rows per lane group sharing every activation load (B >= 3: activation traffic per weight byte / R).
#[inline(always)]
pub unsafe fn row_dot_r<V: Vq, const B: usize, const R: usize>(rows: &[*const u8; R], j0: u32, step: u32, nsb: u32,
    aq: *const i8, ad: *const f32, as_: *const f32, k: u32, ar: &[u32; B], acc: &mut [[f32; B]; R]) {
    debug_assert!(R <= 4);
    let mut j = j0;
    while j < nsb {
        // prefetch the next sub-block's quant block of every row into L1 (more loads in flight per lane)
        if PREFETCH && V::QK >= 256 && j + step < nsb {
            let jn = j + step;
            let off = ((jn * 32 / V::QK) * V::BYTES) as usize;
            unroll!(r in 4 => {
                if R > r {
                    prefetch_l1(rows[r].add(off));
                    if V::BYTES > 64 {
                        prefetch_l1(rows[r].add(off + 64));
                    }
                    if V::BYTES > 128 {
                        prefetch_l1(rows[r].add(off + 128));
                    }
                }
            });
        }
        // explicit per-row copies (constant indices only: keeps everything in registers)
        let s0 = V::sb(rows[0], j);
        let s1 = if R > 1 { V::sb(rows[1], j) } else { V::sb(rows[0], j) };
        let s2 = if R > 2 { V::sb(rows[2], j) } else { V::sb(rows[0], j) };
        let s3 = if R > 3 { V::sb(rows[3], j) } else { V::sb(rows[0], j) };
        // explicit activation rows too (B loop fully expanded with constant indices)
        macro_rules! act {
            ($b:literal) => {
                if B > $b {
                    let b = $b;
                    let rr = *ar.get_unchecked(b) as usize;
                    let a = ldg_nc_v4(aq.add(rr * k as usize + 32 * j as usize) as *const u32);
                    let c = ldg_nc_v4(aq.add(rr * k as usize + 32 * j as usize + 16) as *const u32);
                    let d8 = *ad.add(rr * nsb as usize + j as usize);
                    let s8 = if V::HAS_M { *as_.add(rr * nsb as usize + j as usize) } else { 0.0 };
                    macro_rules! one {
                        ($w:ident, $r:literal) => {
                            if R > $r {
                                let w = &$w;
                                let mut v = (d8 * w.base) * sb_dot::<V>(w, &a, &c);
                                if V::HAS_M {
                                    v -= w.m * s8;
                                }
                                *acc.get_unchecked_mut($r).get_unchecked_mut(b) += v;
                            }
                        };
                    }
                    one!(s0, 0);
                    one!(s1, 1);
                    one!(s2, 2);
                    one!(s3, 3);
                }
            };
        }
        act!(0);
        act!(1);
        act!(2);
        act!(3);
        act!(4);
        act!(5);
        act!(6);
        act!(7);
        act!(8);
        act!(9);
        act!(10);
        act!(11);
        act!(12);
        act!(13);
        act!(14);
        act!(15);
        act!(16);
        act!(17);
        act!(18);
        act!(19);
        act!(20);
        act!(21);
        act!(22);
        act!(23);
        j += step;
    }
}

/// Dense GEMV with R rows per lane group: a warp covers `R * 32 / lpr` rows.
#[inline(always)]
pub unsafe fn gemv_r<V: Vq, const B: usize, const R: usize>(y: *mut f32, w: *const u8, row_bytes: u64, aq: *const i8,
    ad: *const f32, as_: *const f32, k: u32, n: u32, lpr: u32, add: u32) {
    let rpw = 32 / lpr;
    let row0 = ((bid_x() * nwarps() + warp_id()) * rpw + lane() / lpr) * R as u32;
    let nsb = k / 32;
    let mut acc = [[0f32; B]; R];
    let rows: [*const u8; R] = core::array::from_fn(|r| w.add((row0 + r as u32).min(n - 1) as usize * row_bytes as usize));
    let ar: [u32; B] = core::array::from_fn(|b| b as u32);
    if row0 < n {
        row_dot_r::<V, B, R>(&rows, lane() % lpr, lpr, nsb, aq, ad, as_, k, &ar, &mut acc);
    }
    macro_rules! out {
        ($r:literal, $b:literal) => {
            if R > $r && B > $b {
                let v = reduce_lanes(*acc.get_unchecked($r).get_unchecked($b), lpr);
                let row = row0 + $r;
                if row < n && lane() % lpr == 0 {
                    let o = y.add($b * n as usize + row as usize);
                    if add != 0 { *o += v } else { *o = v }
                }
            }
        };
    }
    macro_rules! outr {
        ($r:literal) => {
            out!($r, 0); out!($r, 1); out!($r, 2); out!($r, 3); out!($r, 4); out!($r, 5); out!($r, 6); out!($r, 7);
            out!($r, 8); out!($r, 9); out!($r, 10); out!($r, 11); out!($r, 12); out!($r, 13); out!($r, 14); out!($r, 15);
            out!($r, 16); out!($r, 17); out!($r, 18); out!($r, 19); out!($r, 20); out!($r, 21); out!($r, 22); out!($r, 23);
        };
    }
    outr!(0);
    outr!(1);
    outr!(2);
    outr!(3);
}

macro_rules! gemv_r_kernel {
    ($v:ty, $name:ident, $b:literal, $r:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(y: *mut f32, w: *const u8, row_bytes: u64, aq: *const i8,
            ad: *const f32, as_: *const f32, k: u32, n: u32, lpr: u32, add: u32) {
            gemv_r::<$v, $b, $r>(y, w, row_bytes, aq, ad, as_, k, n, lpr, add)
        }
    };
}
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b1_r2, 1, 2);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b1_r4, 1, 4);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b2_r2, 2, 2);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b2_r4, 2, 4);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b3_r2, 3, 2);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b3_r4, 3, 4);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b4_r2, 4, 2);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b4_r4, 4, 4);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b5_r2, 5, 2);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b5_r4, 5, 4);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b6_r2, 6, 2);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b6_r4, 6, 4);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b7_r2, 7, 2);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b7_r4, 7, 4);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b8_r2, 8, 2);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b8_r4, 8, 4);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b1_r2, 1, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b1_r4, 1, 4);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b2_r2, 2, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b2_r4, 2, 4);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b3_r2, 3, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b3_r4, 3, 4);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b4_r2, 4, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b4_r4, 4, 4);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b5_r2, 5, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b5_r4, 5, 4);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b6_r2, 6, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b6_r4, 6, 4);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b7_r2, 7, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b7_r4, 7, 4);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b8_r2, 8, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b8_r4, 8, 4);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b1_r2, 1, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b1_r4, 1, 4);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b2_r2, 2, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b2_r4, 2, 4);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b3_r2, 3, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b3_r4, 3, 4);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b4_r2, 4, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b4_r4, 4, 4);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b5_r2, 5, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b5_r4, 5, 4);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b6_r2, 6, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b6_r4, 6, 4);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b7_r2, 7, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b7_r4, 7, 4);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b8_r2, 8, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b8_r4, 8, 4);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b1_r2, 1, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b1_r4, 1, 4);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b2_r2, 2, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b2_r4, 2, 4);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b3_r2, 3, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b3_r4, 3, 4);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b4_r2, 4, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b4_r4, 4, 4);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b5_r2, 5, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b5_r4, 5, 4);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b6_r2, 6, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b6_r4, 6, 4);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b7_r2, 7, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b7_r4, 7, 4);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b8_r2, 8, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b8_r4, 8, 4);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b1_r2, 1, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b1_r4, 1, 4);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b2_r2, 2, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b2_r4, 2, 4);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b3_r2, 3, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b3_r4, 3, 4);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b4_r2, 4, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b4_r4, 4, 4);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b5_r2, 5, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b5_r4, 5, 4);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b6_r2, 6, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b6_r4, 6, 4);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b7_r2, 7, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b7_r4, 7, 4);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b8_r2, 8, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b8_r4, 8, 4);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b1_r2, 1, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b1_r4, 1, 4);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b2_r2, 2, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b2_r4, 2, 4);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b3_r2, 3, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b3_r4, 3, 4);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b4_r2, 4, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b4_r4, 4, 4);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b5_r2, 5, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b5_r4, 5, 4);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b6_r2, 6, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b6_r4, 6, 4);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b7_r2, 7, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b7_r4, 7, 4);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b8_r2, 8, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b8_r4, 8, 4);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b1_r2, 1, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b1_r4, 1, 4);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b2_r2, 2, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b2_r4, 2, 4);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b3_r2, 3, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b3_r4, 3, 4);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b4_r2, 4, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b4_r4, 4, 4);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b5_r2, 5, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b5_r4, 5, 4);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b6_r2, 6, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b6_r4, 6, 4);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b7_r2, 7, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b7_r4, 7, 4);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b8_r2, 8, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b8_r4, 8, 4);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b1_r2, 1, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b1_r4, 1, 4);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b2_r2, 2, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b2_r4, 2, 4);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b3_r2, 3, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b3_r4, 3, 4);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b4_r2, 4, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b4_r4, 4, 4);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b5_r2, 5, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b5_r4, 5, 4);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b6_r2, 6, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b6_r4, 6, 4);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b7_r2, 7, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b7_r4, 7, 4);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b8_r2, 8, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b8_r4, 8, 4);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b1_r2, 1, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b1_r4, 1, 4);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b2_r2, 2, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b2_r4, 2, 4);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b3_r2, 3, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b3_r4, 3, 4);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b4_r2, 4, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b4_r4, 4, 4);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b5_r2, 5, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b5_r4, 5, 4);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b6_r2, 6, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b6_r4, 6, 4);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b7_r2, 7, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b7_r4, 7, 4);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b8_r2, 8, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b8_r4, 8, 4);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b1_r2, 1, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b1_r4, 1, 4);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b2_r2, 2, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b2_r4, 2, 4);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b3_r2, 3, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b3_r4, 3, 4);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b4_r2, 4, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b4_r4, 4, 4);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b5_r2, 5, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b5_r4, 5, 4);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b6_r2, 6, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b6_r4, 6, 4);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b7_r2, 7, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b7_r4, 7, 4);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b8_r2, 8, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b8_r4, 8, 4);

/// Round half away from zero (C `roundf`), for |v| < 2^23.
#[inline(always)]
fn roundf(v: f32) -> f32 {
    let a = v.abs();
    let mut r = (a + 0.5) as i32 as f32;
    if a < 0.5 {
        r = 0.0;
    }
    if v < 0.0 { -r } else { r }
}

/// q8_1 activation quantization (ggml-cuda `quantize_q8_1`): one warp per 32-value block.
/// Grid: x = ceil(K / 32 / 4) blocks of 128 threads, y = activation row. Input f32 (`bf16 = 0`) or bf16 (`bf16 = 1`),
/// row stride `ld` elements.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn ggq_q8_1(x: *const u8, ld: u32, bf16in: u32, aq: *mut i8, ad: *mut f32,
    as_: *mut f32, k: u32, sum_mode: u32, rows: u32) {
    let nsb = k / 32;
    let blk = bid_x() * 4 + warp_id();
    if blk >= nsb {
        return;
    }
    // grid.y may be smaller than rows (65535 cap): stride over rows
    let mut r = bid_y();
    while r < rows {
        let ru = r as usize;
        let i = (blk * 32 + lane()) as usize;
        let v = if bf16in != 0 {
            bf16::to_f32(*(x as *const u16).add(ru * ld as usize + i))
        } else {
            *(x as *const f32).add(ru * ld as usize + i)
        };
        let amax = warp::max(v.abs());
        let d = amax / 127.0;
        let q = if amax == 0.0 { 0i32 } else { roundf(v / d) as i32 };
        *aq.add(ru * k as usize + i) = q as i8;
        // 0: f16(sum x) (ggml ds.y, default), 1: f32 sum x, 2: d * sum(q) (consistent with the quantized dot)
        let sx = match sum_mode {
            1 => warp::sum(v),
            2 => d * warp::sum(q as f32),
            _ => f16_round(warp::sum(v)),
        };
        if lane() == 0 {
            *ad.add(ru * nsb as usize + blk as usize) = d;
            *as_.add(ru * nsb as usize + blk as usize) = sx;
        }
        r += gdim_y();
    }
}

#[inline(always)]
fn f16_round(x: f32) -> f32 {
    let h: u16;
    unsafe { core::arch::asm!("cvt.rn.f16.f32 {h}, {x};", h = out(reg16) h, x = in(reg32) x, options(pure, nomem, nostack)) }
    crate::dec::f16_to_f32(h)
}

// wide decode GEMVs (one weight row per lane group): 12 / 16 / 20 / 24 activation rows in one pass, the same per-row
 // float order as B <= 8 (lane split by lpr only)
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b12_w, 12, 1);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b16_w, 16, 1);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b20_w, 20, 1);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b24_w, 24, 1);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b12_w, 12, 1);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b16_w, 16, 1);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b20_w, 20, 1);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b24_w, 24, 1);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b12_w, 12, 1);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b16_w, 16, 1);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b20_w, 20, 1);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b24_w, 24, 1);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b12_w, 12, 1);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b16_w, 16, 1);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b20_w, 20, 1);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b24_w, 24, 1);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b12_w, 12, 1);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b16_w, 16, 1);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b20_w, 20, 1);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b24_w, 24, 1);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b12_w, 12, 1);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b16_w, 16, 1);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b20_w, 20, 1);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b24_w, 24, 1);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b12_w, 12, 1);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b16_w, 16, 1);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b20_w, 20, 1);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b24_w, 24, 1);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b12_w, 12, 1);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b16_w, 16, 1);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b20_w, 20, 1);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b24_w, 24, 1);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b12_w, 12, 1);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b16_w, 16, 1);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b20_w, 20, 1);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b24_w, 24, 1);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b12_w, 12, 1);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b16_w, 16, 1);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b20_w, 20, 1);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b24_w, 24, 1);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b12_w2, 12, 2);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b16_w2, 16, 2);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b20_w2, 20, 2);
gemv_r_kernel!(Q8_0, ggq_gemv_q8_0_b24_w2, 24, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b12_w2, 12, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b16_w2, 16, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b20_w2, 20, 2);
gemv_r_kernel!(Q2_0, ggq_gemv_q2_0_b24_w2, 24, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b12_w2, 12, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b16_w2, 16, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b20_w2, 20, 2);
gemv_r_kernel!(Iq4Nl, ggq_gemv_iq4_nl_b24_w2, 24, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b12_w2, 12, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b16_w2, 16, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b20_w2, 20, 2);
gemv_r_kernel!(Iq4Xs, ggq_gemv_iq4_xs_b24_w2, 24, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b12_w2, 12, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b16_w2, 16, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b20_w2, 20, 2);
gemv_r_kernel!(Q4K, ggq_gemv_q4_k_b24_w2, 24, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b12_w2, 12, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b16_w2, 16, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b20_w2, 20, 2);
gemv_r_kernel!(Q5K, ggq_gemv_q5_k_b24_w2, 24, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b12_w2, 12, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b16_w2, 16, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b20_w2, 20, 2);
gemv_r_kernel!(Q6K, ggq_gemv_q6_k_b24_w2, 24, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b12_w2, 12, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b16_w2, 16, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b20_w2, 20, 2);
gemv_r_kernel!(Iq2S, ggq_gemv_iq2_s_b24_w2, 24, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b12_w2, 12, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b16_w2, 16, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b20_w2, 20, 2);
gemv_r_kernel!(Iq3Xxs, ggq_gemv_iq3_xxs_b24_w2, 24, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b12_w2, 12, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b16_w2, 16, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b20_w2, 20, 2);
gemv_r_kernel!(Iq3S, ggq_gemv_iq3_s_b24_w2, 24, 2);
