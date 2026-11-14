#![allow(unsafe_op_in_unsafe_fn)]
//! CPU kernels (Rust, AVX2 + FMA, runtime-detected; scalar fallback) for MoE experts that are not on the card:
//! gate/up GEMV + SwiGLU + down GEMV on 1..8 token rows, threaded over a persistent spin pool.
//!
//! ```ignore
//! let mut moe = fe_ggq::cpu::CpuMoe::new(threads, 2560, 640);
//! let jobs = [MoeJob { gate, up, down, ty_gu, ty_down, rows: vec![0, 2] }, ...];   // slices of the mmap
//! moe.run(&jobs, &x /* [n_tok][2560] f32 */);
//! let y = moe.out(j);   // [jobs[j].rows.len()][2560] f32, un-weighted expert outputs
//! ```
pub mod avx2;
pub mod pool;
pub mod tile;

use crate::refq;
use anyhow::{bail, Result};
use avx2::Q8Rows;
use fe_gguf::GgmlType;
use std::sync::atomic::{AtomicUsize, Ordering};

pub use pool::Pool;

pub fn has_avx2() -> bool {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(avx2::init_gather);
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c") && is_x86_feature_detected!("bmi2")
}

/// Smallest row count that takes the AVX-VNNI kernels (FE_GGQ_CPU_VNNI_MINB, default 2: at B1 the maddubs form wins on both core types).
fn vnni_min_b() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FE_GGQ_CPU_VNNI_MINB").ok().and_then(|v| v.parse().ok()).unwrap_or(2))
}

/// Smallest row count for the AVX-VNNI offset-form kernels (FE_GGQ_CPU_VNNI_OFF_MINB; default off: no faster than maddubs + madd on the 285K).
fn vnni_off_min_b() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FE_GGQ_CPU_VNNI_OFF_MINB").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX))
}

/// Types with an AVX2 kernel.
pub fn supported(ty: GgmlType) -> bool {
    crate::gpu::GEMV_TYPES.contains(&ty)
}

macro_rules! dispatch_gpu_vnni {
    ($c:ty, $b:expr, $w:expr, $rb:expr, $rows:expr, $a:expr, $y:expr, $ld:expr, $lpr:expr) => {
        match $b {
            1 => avx2::gemv_rows_gpu_vnni::<$c, 1>($w, $rb, $rows, $a, $y, $ld, $lpr),
            2 => avx2::gemv_rows_gpu_vnni::<$c, 2>($w, $rb, $rows, $a, $y, $ld, $lpr),
            3 => avx2::gemv_rows_gpu_vnni::<$c, 3>($w, $rb, $rows, $a, $y, $ld, $lpr),
            4 => avx2::gemv_rows_gpu_vnni::<$c, 4>($w, $rb, $rows, $a, $y, $ld, $lpr),
            5 => avx2::gemv_rows_gpu_vnni::<$c, 5>($w, $rb, $rows, $a, $y, $ld, $lpr),
            6 => avx2::gemv_rows_gpu_vnni::<$c, 6>($w, $rb, $rows, $a, $y, $ld, $lpr),
            7 => avx2::gemv_rows_gpu_vnni::<$c, 7>($w, $rb, $rows, $a, $y, $ld, $lpr),
            _ => avx2::gemv_rows_gpu_vnni::<$c, 8>($w, $rb, $rows, $a, $y, $ld, $lpr),
        }
    };
}

macro_rules! dispatch_gpu {
    ($c:ty, $b:expr, $w:expr, $rb:expr, $rows:expr, $a:expr, $y:expr, $ld:expr, $lpr:expr) => {
        match $b {
            1 => avx2::gemv_rows_gpu::<$c, 1>($w, $rb, $rows, $a, $y, $ld, $lpr),
            2 => avx2::gemv_rows_gpu::<$c, 2>($w, $rb, $rows, $a, $y, $ld, $lpr),
            3 => avx2::gemv_rows_gpu::<$c, 3>($w, $rb, $rows, $a, $y, $ld, $lpr),
            4 => avx2::gemv_rows_gpu::<$c, 4>($w, $rb, $rows, $a, $y, $ld, $lpr),
            5 => avx2::gemv_rows_gpu::<$c, 5>($w, $rb, $rows, $a, $y, $ld, $lpr),
            6 => avx2::gemv_rows_gpu::<$c, 6>($w, $rb, $rows, $a, $y, $ld, $lpr),
            7 => avx2::gemv_rows_gpu::<$c, 7>($w, $rb, $rows, $a, $y, $ld, $lpr),
            _ => avx2::gemv_rows_gpu::<$c, 8>($w, $rb, $rows, $a, $y, $ld, $lpr),
        }
    };
}

/// [`gemv_rows`] with the GPU decode GEMV's float order for `lpr` lanes per row: bit-identical to `Ggq::gemv` /
/// `moe_gu_q8` / `moe_down` run with the same `lpr` (AVX2 only; K % 128 == 0; lpr in 4, 8, 16, 32).
///
/// # Safety
/// As [`gemv_rows`].
pub unsafe fn gemv_rows_gpu(ty: GgmlType, w: *const u8, row_bytes: usize, rows: std::ops::Range<usize>, a: &Q8Rows,
    y: *mut f32, ld: usize, lpr: usize) {
    use GgmlType::*;
    debug_assert!(a.rows >= 1 && a.k % 128 == 0 && lpr >= 4 && lpr <= 32);
    if a.rows > 8 {
        // chunks of <= 8 rows (a row's order does not depend on the others)
        let mut r0 = 0;
        let mut c = Q8Rows::default();
        while r0 < a.rows {
            let rb = (a.rows - r0).min(8);
            c.shape(rb, a.k);
            for i in 0..rb {
                c.copy_row(i, a, r0 + i);
            }
            gemv_rows_gpu(ty, w, row_bytes, rows.clone(), &c, y.add(r0 * ld), ld, lpr);
            r0 += rb;
        }
        return;
    }
    let b = a.rows;
    match ty {
        Q8_0 if avx2::vnni() && vnni_min_b() <= b => dispatch_gpu_vnni!(avx2::Q8_0, b, w, row_bytes, rows, a, y, ld, lpr),
        Q8_0 => dispatch_gpu!(avx2::Q8_0, b, w, row_bytes, rows, a, y, ld, lpr),
        Q2_0 if avx2::vnni() && vnni_min_b() <= b => dispatch_gpu_vnni!(avx2::Q2_0, b, w, row_bytes, rows, a, y, ld, lpr),
        Q2_0 => dispatch_gpu!(avx2::Q2_0, b, w, row_bytes, rows, a, y, ld, lpr),
        IQ4NL if avx2::vnni() && vnni_min_b() <= b => dispatch_gpu_vnni!(avx2::Iq4Nl, b, w, row_bytes, rows, a, y, ld, lpr),
        IQ4NL => dispatch_gpu!(avx2::Iq4Nl, b, w, row_bytes, rows, a, y, ld, lpr),
        IQ4XS => dispatch_gpu!(avx2::Iq4Xs, b, w, row_bytes, rows, a, y, ld, lpr),
        Q4K => dispatch_gpu!(avx2::Q4K, b, w, row_bytes, rows, a, y, ld, lpr),
        Q5K => dispatch_gpu!(avx2::Q5K, b, w, row_bytes, rows, a, y, ld, lpr),
        Q6K => dispatch_gpu!(avx2::Q6K, b, w, row_bytes, rows, a, y, ld, lpr),
        IQ2S if avx2::gather() => dispatch_gpu!(avx2::Iq2S<true>, b, w, row_bytes, rows, a, y, ld, lpr),
        IQ2S => dispatch_gpu!(avx2::Iq2S<false>, b, w, row_bytes, rows, a, y, ld, lpr),
        IQ3XXS if avx2::gather() => dispatch_gpu!(avx2::Iq3Xxs<true>, b, w, row_bytes, rows, a, y, ld, lpr),
        IQ3XXS => dispatch_gpu!(avx2::Iq3Xxs<false>, b, w, row_bytes, rows, a, y, ld, lpr),
        IQ3S if avx2::gather() => dispatch_gpu!(avx2::Iq3S<true>, b, w, row_bytes, rows, a, y, ld, lpr),
        IQ3S => dispatch_gpu!(avx2::Iq3S<false>, b, w, row_bytes, rows, a, y, ld, lpr),
        _ => unreachable!(),
    }
}

/// exp(x) with exactly the GPU's operations (kcommon `math::exp_det`): mul / add / fma only.
#[inline]
pub fn exp_det(x: f32) -> f32 {
    let t = x * std::f32::consts::LOG2_E;
    let t = if t > 128.0 { 128.0 } else if t < -150.0 { -150.0 } else { t };
    let n = (t + 12582912.0) - 12582912.0;
    let f = t - n;
    let mut p = 1.535336188319500e-4f32;
    p = p.mul_add(f, 1.339887440266574e-3);
    p = p.mul_add(f, 9.618437357674640e-3);
    p = p.mul_add(f, 5.550332471162809e-2);
    p = p.mul_add(f, 2.402264791363012e-1);
    p = p.mul_add(f, 6.931472028550421e-1);
    let r = f.mul_add(p, 1.0);
    let ni = n as i32;
    let n1 = ni / 2;
    let n2 = ni - n1;
    r * f32::from_bits(((n1 + 127) as u32) << 23) * f32::from_bits(((n2 + 127) as u32) << 23)
}
/// x / (1 + exp_det(-x)): the GPU's SiLU.
#[inline]
pub fn silu_det(x: f32) -> f32 {
    x / (1.0 + exp_det(-x))
}

macro_rules! dispatch_vnni {
    ($c:ty, $b:expr, $w:expr, $rb:expr, $rows:expr, $a:expr, $y:expr, $ld:expr) => {
        match $b {
            1 => avx2::gemv_rows_vnni::<$c, 1>($w, $rb, $rows, $a, $y, $ld),
            2 => avx2::gemv_rows_vnni::<$c, 2>($w, $rb, $rows, $a, $y, $ld),
            3 => avx2::gemv_rows_vnni::<$c, 3>($w, $rb, $rows, $a, $y, $ld),
            4 => avx2::gemv_rows_vnni::<$c, 4>($w, $rb, $rows, $a, $y, $ld),
            5 => avx2::gemv_rows_vnni::<$c, 5>($w, $rb, $rows, $a, $y, $ld),
            6 => avx2::gemv_rows_vnni::<$c, 6>($w, $rb, $rows, $a, $y, $ld),
            7 => avx2::gemv_rows_vnni::<$c, 7>($w, $rb, $rows, $a, $y, $ld),
            _ => avx2::gemv_rows_vnni::<$c, 8>($w, $rb, $rows, $a, $y, $ld),
        }
    };
}

macro_rules! dispatch_vnni_off {
    ($c:ty, $b:expr, $w:expr, $rb:expr, $rows:expr, $a:expr, $y:expr, $ld:expr) => {
        match $b {
            1 => avx2::gemv_rows_vnni_off::<$c, 1>($w, $rb, $rows, $a, $y, $ld),
            2 => avx2::gemv_rows_vnni_off::<$c, 2>($w, $rb, $rows, $a, $y, $ld),
            3 => avx2::gemv_rows_vnni_off::<$c, 3>($w, $rb, $rows, $a, $y, $ld),
            4 => avx2::gemv_rows_vnni_off::<$c, 4>($w, $rb, $rows, $a, $y, $ld),
            5 => avx2::gemv_rows_vnni_off::<$c, 5>($w, $rb, $rows, $a, $y, $ld),
            6 => avx2::gemv_rows_vnni_off::<$c, 6>($w, $rb, $rows, $a, $y, $ld),
            7 => avx2::gemv_rows_vnni_off::<$c, 7>($w, $rb, $rows, $a, $y, $ld),
            _ => avx2::gemv_rows_vnni_off::<$c, 8>($w, $rb, $rows, $a, $y, $ld),
        }
    };
}

macro_rules! dispatch_b {
    ($c:ty, $b:expr, $w:expr, $rb:expr, $rows:expr, $a:expr, $y:expr, $ld:expr) => {
        match $b {
            1 => avx2::gemv_rows::<$c, 1>($w, $rb, $rows, $a, $y, $ld),
            2 => avx2::gemv_rows::<$c, 2>($w, $rb, $rows, $a, $y, $ld),
            3 => avx2::gemv_rows::<$c, 3>($w, $rb, $rows, $a, $y, $ld),
            4 => avx2::gemv_rows::<$c, 4>($w, $rb, $rows, $a, $y, $ld),
            5 => avx2::gemv_rows::<$c, 5>($w, $rb, $rows, $a, $y, $ld),
            6 => avx2::gemv_rows::<$c, 6>($w, $rb, $rows, $a, $y, $ld),
            7 => avx2::gemv_rows::<$c, 7>($w, $rb, $rows, $a, $y, $ld),
            _ => avx2::gemv_rows::<$c, 8>($w, $rb, $rows, $a, $y, $ld),
        }
    };
}

/// `y[b * ld + r] = W[r] . a[b]` for `r` in `rows` (W rows `row_bytes` apart, `a.k` columns, `a.rows` <= 8).
///
/// # Safety
/// `w` must hold every row in `rows`; `y` must have room for `a.rows` rows of stride `ld` covering `rows`.
pub unsafe fn gemv_rows(ty: GgmlType, w: *const u8, row_bytes: usize, rows: std::ops::Range<usize>, a: &Q8Rows,
    y: *mut f32, ld: usize) {
    use GgmlType::*;
    debug_assert!(a.rows >= 1 && a.rows <= 8);
    if !has_avx2() {
        return gemv_rows_scalar(ty, w, row_bytes, rows, a, y, ld);
    }
    let b = a.rows;
    match ty {
        Q8_0 if a.k % 128 == 0 && avx2::vnni() && vnni_min_b() <= b => dispatch_vnni!(avx2::Q8_0, b, w, row_bytes, rows, a, y, ld),
        Q8_0 => dispatch_b!(avx2::Q8_0, b, w, row_bytes, rows, a, y, ld),
        Q2_0 if a.k % 128 == 0 && avx2::vnni() && vnni_min_b() <= b => dispatch_vnni!(avx2::Q2_0, b, w, row_bytes, rows, a, y, ld),
        Q2_0 => dispatch_b!(avx2::Q2_0, b, w, row_bytes, rows, a, y, ld),
        IQ4NL if a.k % 128 == 0 && avx2::vnni() && vnni_min_b() <= b => dispatch_vnni!(avx2::Iq4Nl, b, w, row_bytes, rows, a, y, ld),
        IQ4NL => dispatch_b!(avx2::Iq4Nl, b, w, row_bytes, rows, a, y, ld),
        IQ4XS => dispatch_b!(avx2::Iq4Xs, b, w, row_bytes, rows, a, y, ld),
        Q4K => dispatch_b!(avx2::Q4K, b, w, row_bytes, rows, a, y, ld),
        Q5K => dispatch_b!(avx2::Q5K, b, w, row_bytes, rows, a, y, ld),
        Q6K => dispatch_b!(avx2::Q6K, b, w, row_bytes, rows, a, y, ld),
        IQ2S if avx2::gather() => dispatch_b!(avx2::Iq2S<true>, b, w, row_bytes, rows, a, y, ld),
        IQ2S => dispatch_b!(avx2::Iq2S<false>, b, w, row_bytes, rows, a, y, ld),
        IQ3XXS if a.k % 128 == 0 && avx2::vnni() && vnni_off_min_b() <= b && avx2::gather() =>
            dispatch_vnni_off!(avx2::Iq3Xxs<true>, b, w, row_bytes, rows, a, y, ld),
        IQ3XXS if a.k % 128 == 0 && avx2::vnni() && vnni_off_min_b() <= b => dispatch_vnni_off!(avx2::Iq3Xxs<false>, b, w, row_bytes, rows, a, y, ld),
        IQ3XXS if avx2::gather() => dispatch_b!(avx2::Iq3Xxs<true>, b, w, row_bytes, rows, a, y, ld),
        IQ3XXS => dispatch_b!(avx2::Iq3Xxs<false>, b, w, row_bytes, rows, a, y, ld),
        IQ3S if a.k % 128 == 0 && avx2::vnni() && vnni_off_min_b() <= b && avx2::gather() =>
            dispatch_vnni_off!(avx2::Iq3S<true>, b, w, row_bytes, rows, a, y, ld),
        IQ3S if a.k % 128 == 0 && avx2::vnni() && vnni_off_min_b() <= b => dispatch_vnni_off!(avx2::Iq3S<false>, b, w, row_bytes, rows, a, y, ld),
        IQ3S if avx2::gather() => dispatch_b!(avx2::Iq3S<true>, b, w, row_bytes, rows, a, y, ld),
        IQ3S => dispatch_b!(avx2::Iq3S<false>, b, w, row_bytes, rows, a, y, ld),
        _ => gemv_rows_scalar(ty, w, row_bytes, rows, a, y, ld),
    }
}

/// Scalar fallback (no AVX2): dequantize the row (refq) and dot with the q8 activations.
///
/// # Safety
/// As [`gemv_rows`].
pub unsafe fn gemv_rows_scalar(ty: GgmlType, w: *const u8, row_bytes: usize, rows: std::ops::Range<usize>,
    a: &Q8Rows, y: *mut f32, ld: usize) {
    let (k, nb) = (a.k, a.k / 32);
    for r in rows {
        let row = std::slice::from_raw_parts(w.add(r * row_bytes), row_bytes);
        for b in 0..a.rows {
            *y.add(b * ld + r) = refq::dot_q8(ty, row, &a.q[b * k..(b + 1) * k], &a.d[b * nb..(b + 1) * nb],
                &a.s[b * nb..(b + 1) * nb]) as f32;
        }
    }
}

/// One expert of one MoE layer on the CPU, for the token rows `rows` (indices into the `x` given to `run`).
pub struct MoeJob<'a> {
    /// `[n_embd -> n_ff]` gate and up, `[n_ff -> n_embd]` down: one expert's slices (e.g. `Gguf::slice(t, e)`).
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub down: &'a [u8],
    pub ty_gu: GgmlType,
    pub ty_down: GgmlType,
    pub rows: Vec<usize>,
}

/// Batched CPU expert FFN: gate/up + SwiGLU + down for many (expert, token rows) jobs in two pool passes.
pub struct CpuMoe {
    pub pool: Pool,
    pub n_embd: usize,
    pub n_ff: usize,
    xq: Vec<Q8Rows>,
    hq: Vec<Q8Rows>,
    h: Vec<Vec<f32>>,
    out: Vec<Vec<f32>>,
    /// rows of gate/up per work unit: 16 (two units share a q8 block of h: the second to finish quantizes it) or a
    /// multiple of 32
    pub gu_chunk: usize,
    /// rows of down per work unit
    pub down_chunk: usize,
    tokq: Q8Rows,
    tmpq: Q8Rows,
    tok_seen: Vec<bool>,
    /// compute in the GPU kernels' float order (default; FE_GGQ_CPU_GPU_ORDER=0 = the faster AVX2 order)
    pub gpu_order: bool,
    /// lanes per row of the GPU kernels this must match: moe_gu_q8 (`gpu::moe_gu_lpr`) and moe_down
    /// (`gpu::default_lpr(n_ff)`)
    pub gu_lpr: usize,
    pub down_lpr: usize,
}

struct SyncPtr<T>(*mut T);
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}

impl CpuMoe {
    /// `threads` in total (the calling thread is one of them), unpinned. Prefer [`CpuMoe::pinned`].
    pub fn new(threads: usize, n_embd: usize, n_ff: usize) -> CpuMoe {
        Self::with_pool(Pool::new(threads, std::time::Duration::from_millis(5)), n_embd, n_ff)
    }
    /// Workers pinned one per CPU in `cpus` (e.g. `pool::parse_cpus("1-7")`), plus the calling thread. Workers spin
    /// for `spin` after each layer (keep it above the time between two layers' calls).
    pub fn pinned(cpus: &[usize], spin: std::time::Duration, n_embd: usize, n_ff: usize) -> CpuMoe {
        Self::with_pool(Pool::pinned(cpus, spin), n_embd, n_ff)
    }
    pub fn with_pool(pool: Pool, n_embd: usize, n_ff: usize) -> CpuMoe {
        CpuMoe {
            pool,
            n_embd,
            n_ff,
            xq: vec![],
            hq: vec![],
            h: vec![],
            out: vec![],
            gu_chunk: 16,
            down_chunk: 64,
            tokq: Q8Rows::default(),
            tmpq: Q8Rows::default(),
            tok_seen: vec![],
            gpu_order: std::env::var("FE_GGQ_CPU_GPU_ORDER").map_or(true, |v| v != "0"),
            gu_lpr: crate::gpu::moe_gu_lpr() as usize,
            down_lpr: crate::gpu::default_lpr(n_ff) as usize,
        }
    }

    /// q8 of h (codes, scales) of job `j`'s rows from the last `run` (`[rows][n_ff]`, `[rows][n_ff / 32]`).
    pub fn h_q8(&self, j: usize) -> (&[i8], &[f32]) {
        let r = self.hq[j].rows;
        (&self.hq[j].q[..r * self.n_ff], &self.hq[j].d[..r * self.n_ff / 32])
    }

    /// Expert output of job `j` from the last `run`: `[jobs[j].rows.len()][n_embd]`.
    pub fn out(&self, j: usize) -> &[f32] {
        &self.out[j]
    }

    /// Run every job. `x`: `[n_tok][n_embd]` f32 (the expert input, i.e. after the FFN norm). Jobs of 9..=32 rows
    /// run as <= 8-row chunks of the per-row kernels (faster than the tiled path there; same bits), larger ones tiled.
    pub fn run(&mut self, jobs: &[MoeJob], x: &[f32]) -> Result<()> {
        let big = jobs.iter().map(|j| j.rows.len()).max().unwrap_or(0);
        if big > 8 && big <= 32 && self.gpu_order {
            let mut chunks: Vec<MoeJob> = vec![];
            let mut owner: Vec<(usize, usize)> = vec![]; // (job, first row)
            for (j, job) in jobs.iter().enumerate() {
                for (ci, c) in job.rows.chunks(8).enumerate() {
                    chunks.push(MoeJob { gate: job.gate, up: job.up, down: job.down, ty_gu: job.ty_gu, ty_down: job.ty_down,
                        rows: c.to_vec() });
                    owner.push((j, ci * 8));
                }
            }
            self.run_le8(&chunks, x)?;
            let ne = self.n_embd;
            let mut outs: Vec<Vec<f32>> = jobs.iter().map(|j| vec![0f32; j.rows.len() * ne]).collect();
            for (ci, &(j, r0)) in owner.iter().enumerate() {
                let n = chunks[ci].rows.len();
                outs[j][r0 * ne..(r0 + n) * ne].copy_from_slice(&self.out[ci][..n * ne]);
            }
            for (j, o) in outs.into_iter().enumerate() {
                self.out[j] = o;
            }
            return Ok(());
        }
        self.run_le8(jobs, x)
    }

    fn run_le8(&mut self, jobs: &[MoeJob], x: &[f32]) -> Result<()> {
        let t_start = std::time::Instant::now();
        let (ne, nf) = (self.n_embd, self.n_ff);
        if nf % 32 != 0 || !(self.gu_chunk == 16 || self.gu_chunk % 32 == 0) || nf % self.gu_chunk != 0 {
            bail!("CpuMoe: n_ff {nf} / gu_chunk {}", self.gu_chunk);
        }
        let nj = jobs.len();
        while self.xq.len() < nj {
            self.xq.push(Q8Rows::default());
            self.hq.push(Q8Rows::default());
            self.h.push(vec![]);
            self.out.push(vec![]);
        }
        // quantize each token row once (jobs share rows: MTP verify, batched decode)
        let ntok = x.len() / ne;
        for (j, job) in jobs.iter().enumerate() {
            if job.rows.is_empty() {
                bail!("CpuMoe: job {j} has no rows");
            }
            if job.rows.len() > 8 {
                // many rows (prompt): the tiled path decodes each weight row once and reuses it across rows
                return self.run_tiled(jobs, x);
            }
            if job.rows.iter().any(|&r| r >= ntok) {
                bail!("CpuMoe: job {j}: row out of range");
            }
        }
        self.tokq.shape(ntok, ne);
        self.tok_seen.clear();
        self.tok_seen.resize(ntok, false);
        for job in jobs {
            for &r in &job.rows {
                if !self.tok_seen[r] {
                    self.tok_seen[r] = true;
                    let mut one = std::mem::take(&mut self.tmpq);
                    one.set(&x[r * ne..(r + 1) * ne], 1, ne);
                    self.tokq.copy_row(r, &one, 0);
                    self.tmpq = one;
                }
            }
        }
        for (j, job) in jobs.iter().enumerate() {
            let b = job.rows.len();
            if !supported(job.ty_gu) || !supported(job.ty_down) {
                bail!("CpuMoe: types {} / {}", job.ty_gu.name(), job.ty_down.name());
            }
            if job.gate.len() < job.ty_gu.row_bytes(ne) * nf || job.down.len() < job.ty_down.row_bytes(nf) * ne {
                bail!("CpuMoe: job {j}: weight slices too short");
            }
            self.xq[j].shape(b, ne);
            for (i, &r) in job.rows.iter().enumerate() {
                let (dst, src) = (&mut self.xq[j], &self.tokq);
                dst.copy_row(i, src, r);
            }
            self.h[j].resize(b * nf, 0.0);
            self.out[j].resize(b * ne, 0.0);
            let hq = &mut self.hq[j];
            hq.k = nf;
            hq.rows = b;
            hq.q.resize(b * nf, 0);
            hq.d.resize(b * nf / 32, 0.0);
            hq.s.resize(b * nf / 32, 0.0);
            hq.qu.resize(b * nf, 0);
        }
        // One pass, work-claiming: units 0..nj*gu_units are gate/up (+ SwiGLU + q8 of h) chunks of gu_chunk rows,
        // the rest are down chunks of down_chunk rows. Units are claimed in that order, so by the time a down unit of
        // job j is taken every gate/up unit of j is claimed; it waits (spinning) only for those still running.
        let gu_units = nf / self.gu_chunk;
        let dc = self.down_chunk;
        let d_units = ne.div_ceil(dc);
        let total = nj * (gu_units + d_units);
        let next = AtomicUsize::new(0);
        let gu_done: Vec<AtomicUsize> = (0..nj).map(|_| AtomicUsize::new(0)).collect();
        // per (job, 32-row h block): halves finished (gu_chunk = 16)
        let half_done: Vec<AtomicUsize> = (0..nj * nf / 32).map(|_| AtomicUsize::new(0)).collect();
        let hp: Vec<SyncPtr<f32>> = self.h.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
        let hqp: Vec<(SyncPtr<i8>, SyncPtr<f32>, SyncPtr<f32>, SyncPtr<u8>)> = self.hq.iter_mut()
            .map(|v| (SyncPtr(v.q.as_mut_ptr()), SyncPtr(v.d.as_mut_ptr()), SyncPtr(v.s.as_mut_ptr()), SyncPtr(v.qu.as_mut_ptr())))
            .collect();
        let op: Vec<SyncPtr<f32>> = self.out.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
        let hq_ro: Vec<SyncPtr<Q8Rows>> = self.hq.iter_mut().map(|v| SyncPtr(v as *mut Q8Rows)).collect();
        let xq = &self.xq;
        let gc = self.gu_chunk;
        let prof = PROF.load(Ordering::Relaxed);
        // the GPU decode kernels' float order (bit-identical to moe_gu_q8 / moe_down) unless FE_GGQ_CPU_GPU_ORDER=0
        let gpu_order = has_avx2() && ne % 128 == 0 && nf % 128 == 0 && self.gpu_order;
        let (lgu, ldn) = (self.gu_lpr, self.down_lpr);
        let t_call = std::time::Instant::now();
        if prof {
            PROF_NS[6].fetch_add(t_start.elapsed().as_nanos() as usize, Ordering::Relaxed);
        }
        self.pool.run_dyn(|_, _| {
            let mut first = true;
            let mut g = [0f32; 8 * 64];
            let mut u = [0f32; 8 * 64];
            let mut tl = std::time::Instant::now();
            let mut acc = [0u64; 3];
            loop {
                let t = next.fetch_add(1, Ordering::Relaxed);
                if t >= total {
                    break;
                }
                if prof {
                    tl = std::time::Instant::now();
                    if first {
                        first = false;
                        PROF_NS[3].fetch_add(t_call.elapsed().as_nanos() as usize, Ordering::Relaxed);
                        PROF_NS[5].fetch_add(1, Ordering::Relaxed);
                    }
                }
                if t < nj * gu_units {
                    let (j, c) = (t / gu_units, t % gu_units);
                    let job = &jobs[j];
                    let a = &xq[j];
                    let b = a.rows;
                    let rb = job.ty_gu.row_bytes(ne);
                    let r0 = c * gc;
                    unsafe {
                        let gp = g.as_mut_ptr().wrapping_sub(r0);
                        let up = u.as_mut_ptr().wrapping_sub(r0);
                        if gpu_order {
                            gemv_rows_gpu(job.ty_gu, job.gate.as_ptr(), rb, r0..r0 + gc, a, gp, gc, lgu);
                            gemv_rows_gpu(job.ty_gu, job.up.as_ptr(), rb, r0..r0 + gc, a, up, gc, lgu);
                        } else {
                            gemv_rows(job.ty_gu, job.gate.as_ptr(), rb, r0..r0 + gc, a, gp, gc);
                            gemv_rows(job.ty_gu, job.up.as_ptr(), rb, r0..r0 + gc, a, up, gc);
                        }
                        let h = hp[j].0;
                        let (hqq, hqd, hqs, hqu) = (hqp[j].0 .0, hqp[j].1 .0, hqp[j].2 .0, hqp[j].3 .0);
                        for bb in 0..b {
                            for i in 0..gc {
                                let gv = g[bb * gc + i];
                                *h.add(bb * nf + r0 + i) = if gpu_order { silu_det(gv) * u[bb * gc + i] }
                                    else { gv / (1.0 + (-gv).exp()) * u[bb * gc + i] };
                            }
                            if gc >= 32 {
                                for blk in 0..gc / 32 {
                                    let base = bb * nf + r0 + blk * 32;
                                    q8_h(h.add(base), hqq.add(base), hqu.add(base), hqd.add(base / 32), hqs.add(base / 32));
                                }
                            }
                        }
                        if gc < 32 {
                            // the second half to finish quantizes the 32-row block for every row
                            let blk = r0 / 32;
                            if half_done[j * (nf / 32) + blk].fetch_add(1, Ordering::AcqRel) == 1 {
                                for bb in 0..b {
                                    let base = bb * nf + blk * 32;
                                    q8_h(h.add(base), hqq.add(base), hqu.add(base), hqd.add(base / 32), hqs.add(base / 32));
                                }
                            }
                        }
                    }
                    gu_done[j].fetch_add(1, Ordering::Release);
                    if prof {
                        acc[0] += tl.elapsed().as_nanos() as u64;
                    }
                } else {
                    let t = t - nj * gu_units;
                    let (j, c) = (t / d_units, t % d_units);
                    while gu_done[j].load(Ordering::Acquire) < gu_units {
                        std::hint::spin_loop();
                    }
                    if prof {
                        acc[1] += tl.elapsed().as_nanos() as u64;
                        tl = std::time::Instant::now();
                    }
                    let job = &jobs[j];
                    let r0 = c * dc;
                    let r1 = (r0 + dc).min(ne);
                    unsafe {
                        if gpu_order {
                            gemv_rows_gpu(job.ty_down, job.down.as_ptr(), job.ty_down.row_bytes(nf), r0..r1, &*hq_ro[j].0, op[j].0, ne, ldn);
                        } else {
                            gemv_rows(job.ty_down, job.down.as_ptr(), job.ty_down.row_bytes(nf), r0..r1, &*hq_ro[j].0, op[j].0, ne);
                        }
                    }
                    if prof {
                        acc[2] += tl.elapsed().as_nanos() as u64;
                    }
                }
            }
            if prof {
                if !first {
                    PROF_NS[4].fetch_add(t_call.elapsed().as_nanos() as usize, Ordering::Relaxed);
                }
                for i in 0..3 {
                    PROF_NS[i].fetch_add(acc[i] as usize, Ordering::Relaxed);
                }
            }
        });
        if prof {
            PROF_NS[7].fetch_add(t_call.elapsed().as_nanos() as usize, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Many-row path (any row count per job, meant for 8..256+): each expert's weights are decoded once into int8
    /// and reused by all its rows (`tile`). Two pool passes: gate/up + SwiGLU + q8(h) in units of 32 h columns,
    /// then down in units of 64 output rows. Same numerics as `run`.
    pub fn run_tiled(&mut self, jobs: &[MoeJob], x: &[f32]) -> Result<()> {
        let (ne, nf) = (self.n_embd, self.n_ff);
        let nj = jobs.len();
        while self.xq.len() < nj {
            self.xq.push(Q8Rows::default());
            self.hq.push(Q8Rows::default());
            self.h.push(vec![]);
            self.out.push(vec![]);
        }
        let mut xr = vec![];
        for (j, job) in jobs.iter().enumerate() {
            let b = job.rows.len();
            if !supported(job.ty_gu) || !supported(job.ty_down) {
                bail!("CpuMoe: types {} / {}", job.ty_gu.name(), job.ty_down.name());
            }
            xr.clear();
            for &r in &job.rows {
                xr.extend_from_slice(&x[r * ne..(r + 1) * ne]);
            }
            self.xq[j].set(&xr, b, ne);
            self.out[j].resize(b * ne, 0.0);
            let hq = &mut self.hq[j];
            hq.k = nf;
            hq.rows = b;
            hq.q.resize(b * nf, 0);
            hq.qu.resize(b * nf, 0);
            hq.d.resize(b * nf / 32, 0.0);
            hq.s.resize(b * nf / 32, 0.0);
        }
        // phase 1: units (job, 32 h columns)
        let gu_units = nf / 32;
        let gpu_order = has_avx2() && ne % 128 == 0 && nf % 128 == 0 && self.gpu_order;
        let (lgu, ldn) = (self.gu_lpr, self.down_lpr);
        let next = AtomicUsize::new(0);
        let hqp: Vec<[usize; 4]> = self.hq.iter_mut()
            .map(|v| [v.q.as_mut_ptr() as usize, v.qu.as_mut_ptr() as usize, v.d.as_mut_ptr() as usize, v.s.as_mut_ptr() as usize])
            .collect();
        let xq = &self.xq;
        self.pool.run_dyn(|_, _| {
            TILE_BUF.with(|tb| {
                let mut tb = tb.borrow_mut();
                let tb = &mut *tb;
                loop {
                    let t = next.fetch_add(1, Ordering::Relaxed);
                    if t >= nj * gu_units {
                        break;
                    }
                    let (j, c) = (t / gu_units, t % gu_units);
                    let job = &jobs[j];
                    let a = &xq[j];
                    let m = a.rows;
                    let rb = job.ty_gu.row_bytes(ne);
                    let off = tile::offset_ok(job.ty_gu);
                    let corr = off || matches!(job.ty_gu, GgmlType::Q4K | GgmlType::Q5K);
                    unsafe {
                        // rows 0..32 = gate, 32..64 = up: one micro call gives g and u
                        decode_any(job.ty_gu, job.gate.as_ptr(), rb, c * 32, 32, ne, off, 0, &mut tb.0);
                        decode_any(job.ty_gu, job.up.as_ptr(), rb, c * 32, 32, ne, off, 32, &mut tb.0);
                        let [pq, pqu, pd, ps] = hqp[j];
                        let (pq, pqu, pd, ps) = (pq as *mut i8, pqu as *mut u8, pd as *mut f32, ps as *mut f32);
                        let mut h = [0f32; 32 * 4];
                        let mut b0 = 0;
                        while b0 < m {
                            let na = (m - b0).min(4);
                            let ar: [usize; 4] = core::array::from_fn(|i| b0 + i.min(na - 1));
                            for r in 0..32 {
                                // gate row r and up row r are the two weight rows of the micro kernel
                                if gpu_order {
                                    let gu = tile::micro_gpu_any(&tb.0, [r, 32 + r], a, ar, na, off, lgu);
                                    for i in 0..na {
                                        h[i * 32 + r] = silu_det(gu[i][0]) * gu[i][1];
                                    }
                                } else {
                                    let gu = tile::micro_any(&tb.0, [r, 32 + r], a, ar, na, off, corr);
                                    for i in 0..na {
                                        let gv = gu[i][0];
                                        h[i * 32 + r] = gv / (1.0 + (-gv).exp()) * gu[i][1];
                                    }
                                }
                            }
                            for i in 0..na {
                                let row = b0 + i;
                                let (q, d, sm) = refq::q8_block(&h[i * 32..i * 32 + 32]);
                                let base = row * nf + c * 32;
                                for z in 0..32 {
                                    *pq.add(base + z) = q[z];
                                    *pqu.add(base + z) = (q[z] as u8) ^ 0x80;
                                }
                                *pd.add(base / 32) = d;
                                *ps.add(base / 32) = sm;
                            }
                            b0 += 4;
                        }
                    }
                }
            });
        });
        // phase 2: units (job, 64 output rows)
        let dch = 64;
        let d_units = ne / dch;
        let next = AtomicUsize::new(0);
        let op: Vec<SyncPtr<f32>> = self.out.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
        let hq = &self.hq;
        self.pool.run_dyn(|_, _| {
            TILE_BUF.with(|tb| {
                let mut tb = tb.borrow_mut();
                let tb = &mut *tb;
                loop {
                    let t = next.fetch_add(1, Ordering::Relaxed);
                    if t >= nj * d_units {
                        break;
                    }
                    let (j, c) = (t / d_units, t % d_units);
                    let job = &jobs[j];
                    let a = &hq[j];
                    let m = a.rows;
                    // short rows (K = 640): the sign form (no per-pair correction epilogue)
                    let off = false;
                    let corr = matches!(job.ty_down, GgmlType::Q4K | GgmlType::Q5K);
                    unsafe {
                        decode_any(job.ty_down, job.down.as_ptr(), job.ty_down.row_bytes(nf), c * dch, dch, nf, off, 0, &mut tb.0);
                        let y = op[j].0;
                        let mut b0 = 0;
                        while b0 < m {
                            let na = (m - b0).min(4);
                            let ar: [usize; 4] = core::array::from_fn(|i| b0 + i.min(na - 1));
                            for r in (0..dch).step_by(2) {
                                let o = if gpu_order { tile::micro_gpu_any(&tb.0, [r, r + 1], a, ar, na, off, ldn) }
                                    else { tile::micro_any(&tb.0, [r, r + 1], a, ar, na, off, corr) };
                                for i in 0..na {
                                    *y.add((b0 + i) * ne + c * dch + r) = o[i][0];
                                    *y.add((b0 + i) * ne + c * dch + r + 1) = o[i][1];
                                }
                            }
                            b0 += 4;
                        }
                    }
                }
            });
        });
        Ok(())
    }
}

impl CpuMoe {
    /// Prompt-path expert FFN on the CPU, bit-identical per pair to the GPU prompt kernels `Ggq::mmq_moe_gu` (h as
    /// q8) + `Ggq::mmq_moe` (down, by pair): the MMQ's float order (one accumulator per output over K), its SiLU and
    /// h quantization (two 16-value halves summed in order). `xq`: the token rows exactly as the GPU quantized them
    /// (`Q8Rows::from_parts` of the group's `Q8Act`); job rows index `xq`. Any row count; tiled (each expert's
    /// weights decoded once per call). Output `out(j)`: [rows][n_embd].
    pub fn run_mmq_order(&mut self, jobs: &[MoeJob], xq: &Q8Rows) -> Result<()> {
        let (ne, nf) = (self.n_embd, self.n_ff);
        if !has_avx2() || ne % 128 != 0 || nf % 128 != 0 || xq.k != ne {
            bail!("run_mmq_order: AVX2 and n_embd, n_ff % 128 == 0 (k {} vs n_embd {ne})", xq.k);
        }
        let nj = jobs.len();
        while self.xq.len() < nj {
            self.xq.push(Q8Rows::default());
            self.hq.push(Q8Rows::default());
            self.h.push(vec![]);
            self.out.push(vec![]);
        }
        for (j, job) in jobs.iter().enumerate() {
            if !supported(job.ty_gu) || !supported(job.ty_down) {
                bail!("CpuMoe: types {} / {}", job.ty_gu.name(), job.ty_down.name());
            }
            let b = job.rows.len();
            self.xq[j].shape(b, ne);
            for (i, &r) in job.rows.iter().enumerate() {
                let dst = &mut self.xq[j];
                dst.copy_row(i, xq, r);
            }
            self.out[j].resize(b * ne, 0.0);
            self.hq[j].shape(b, nf);
        }
        let gu_units = nf / 32;
        let next = AtomicUsize::new(0);
        let hqp: Vec<[usize; 4]> = self.hq.iter_mut()
            .map(|v| [v.q.as_mut_ptr() as usize, v.qu.as_mut_ptr() as usize, v.d.as_mut_ptr() as usize, v.s.as_mut_ptr() as usize])
            .collect();
        let xqs = &self.xq;
        self.pool.run_dyn(|_, _| {
            TILE_BUF.with(|tb| {
                let mut tb = tb.borrow_mut();
                let tb = &mut *tb;
                loop {
                    let t = next.fetch_add(1, Ordering::Relaxed);
                    if t >= nj * gu_units {
                        break;
                    }
                    let (j, c) = (t / gu_units, t % gu_units);
                    let job = &jobs[j];
                    let a = &xqs[j];
                    let m = a.rows;
                    let rb = job.ty_gu.row_bytes(ne);
                    let off = tile::offset_ok(job.ty_gu);
                    unsafe {
                        decode_any(job.ty_gu, job.gate.as_ptr(), rb, c * 32, 32, ne, off, 0, &mut tb.0);
                        decode_any(job.ty_gu, job.up.as_ptr(), rb, c * 32, 32, ne, off, 32, &mut tb.0);
                        let [pq, pqu, pd, ps] = hqp[j];
                        let (pq, pqu, pd, ps) = (pq as *mut i8, pqu as *mut u8, pd as *mut f32, ps as *mut f32);
                        let mut h = [0f32; 32 * 4];
                        let mut b0 = 0;
                        while b0 < m {
                            let na = (m - b0).min(4);
                            let ar: [usize; 4] = core::array::from_fn(|i| b0 + i.min(na - 1));
                            for r in 0..32 {
                                let gu = if off { tile::micro_mmq::<true>(&tb.0, [r, 32 + r], a, ar, na) }
                                    else { tile::micro_mmq::<false>(&tb.0, [r, 32 + r], a, ar, na) };
                                for i in 0..na {
                                    h[i * 32 + r] = silu_det(gu[i][0]) * gu[i][1];
                                }
                            }
                            for i in 0..na {
                                let row = b0 + i;
                                let base = row * nf + c * 32;
                                q8_mmq_h(&h[i * 32..i * 32 + 32], pq.add(base), pqu.add(base), pd.add(base / 32), ps.add(base / 32));
                            }
                            b0 += 4;
                        }
                    }
                }
            });
        });
        let dch = 64;
        let d_units = ne / dch;
        let next = AtomicUsize::new(0);
        let op: Vec<SyncPtr<f32>> = self.out.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
        let hq = &self.hq;
        self.pool.run_dyn(|_, _| {
            TILE_BUF.with(|tb| {
                let mut tb = tb.borrow_mut();
                let tb = &mut *tb;
                loop {
                    let t = next.fetch_add(1, Ordering::Relaxed);
                    if t >= nj * d_units {
                        break;
                    }
                    let (j, c) = (t / d_units, t % d_units);
                    let job = &jobs[j];
                    let a = &hq[j];
                    let m = a.rows;
                    unsafe {
                        decode_any(job.ty_down, job.down.as_ptr(), job.ty_down.row_bytes(nf), c * dch, dch, nf, false, 0, &mut tb.0);
                        let y = op[j].0;
                        let mut b0 = 0;
                        while b0 < m {
                            let na = (m - b0).min(4);
                            let ar: [usize; 4] = core::array::from_fn(|i| b0 + i.min(na - 1));
                            for r in (0..dch).step_by(2) {
                                let o = tile::micro_mmq::<false>(&tb.0, [r, r + 1], a, ar, na);
                                for i in 0..na {
                                    *y.add((b0 + i) * ne + c * dch + r) = o[i][0];
                                    *y.add((b0 + i) * ne + c * dch + r + 1) = o[i][1];
                                }
                            }
                            b0 += 4;
                        }
                    }
                }
            });
        });
        Ok(())
    }
}

/// q8 of 32 h values as the prompt MMQ's gate/up epilogue does it: d = amax / 127, q = round-half-away(h / d),
/// s = f16(sum of h[0..16] in order + sum of h[16..32] in order).
#[inline]
unsafe fn q8_mmq_h(h: &[f32], q: *mut i8, qu: *mut u8, d: *mut f32, s: *mut f32) {
    let amax = h.iter().fold(0f32, |m, v| m.max(v.abs()));
    let dd = amax / 127.0;
    let (mut s0, mut s1) = (0f32, 0f32);
    for i in 0..16 {
        s0 += h[i];
        s1 += h[16 + i];
    }
    for i in 0..32 {
        let v = if amax == 0.0 { 0i32 } else {
            // roundf of k-gguf mmq.rs: (|x| + 0.5) truncated, 0 below 0.5, sign restored
            let x = h[i] / dd;
            let a = x.abs();
            let mut r = (a + 0.5) as i32;
            if a < 0.5 {
                r = 0;
            }
            if x < 0.0 { -r } else { r }
        };
        *q.add(i) = v as i8;
        *qu.add(i) = (v as i8 as u8) ^ 0x80;
    }
    *d = dd;
    *s = refq::f16_to_f32(refq::f32_to_f16(s0 + s1));
}

/// Bench-only profile of `run`: summed thread time in gate/up units, waiting for gate/up before down, down units.
pub static PROF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static PROF_NS: [AtomicUsize; 8] = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)];

/// q8 of 32 h values into (q, qu, d, s): AVX2 (bit-identical to `refq::q8_block`) or the scalar reference.
#[inline]
unsafe fn q8_h(h: *const f32, q: *mut i8, qu: *mut u8, d: *mut f32, s: *mut f32) {
    if has_avx2() {
        let (dd, ss) = avx2::q8_block_fast(h, q, qu);
        *d = dd;
        *s = ss;
    } else {
        let (qq, dd, ss) = refq::q8_block(std::slice::from_raw_parts(h, 32));
        for z in 0..32 {
            *q.add(z) = qq[z];
            *qu.add(z) = (qq[z] as u8) ^ 0x80;
        }
        *d = dd;
        *s = ss;
    }
}

thread_local! {
    static TILE_BUF: std::cell::RefCell<(tile::DecRows, tile::DecRows)> = std::cell::RefCell::new(Default::default());
}

/// `tile::decode_rows` for a runtime type.
#[allow(clippy::too_many_arguments)]
unsafe fn decode_any(ty: GgmlType, w: *const u8, rb: usize, r0: usize, n: usize, k: usize, off: bool, at: usize,
    d: &mut tile::DecRows) {
    use GgmlType::*;
    match ty {
        Q8_0 => tile::decode_rows::<avx2::Q8_0>(w, rb, r0, n, k, off, at, d),
        Q2_0 => tile::decode_rows::<avx2::Q2_0>(w, rb, r0, n, k, off, at, d),
        IQ4NL => tile::decode_rows::<avx2::Iq4Nl>(w, rb, r0, n, k, off, at, d),
        IQ4XS => tile::decode_rows::<avx2::Iq4Xs>(w, rb, r0, n, k, off, at, d),
        Q4K => tile::decode_rows::<avx2::Q4K>(w, rb, r0, n, k, off, at, d),
        Q5K => tile::decode_rows::<avx2::Q5K>(w, rb, r0, n, k, off, at, d),
        Q6K => tile::decode_rows::<avx2::Q6K>(w, rb, r0, n, k, off, at, d),
        IQ2S => tile::decode_rows::<avx2::Iq2S>(w, rb, r0, n, k, off, at, d),
        IQ3XXS => tile::decode_rows::<avx2::Iq3Xxs>(w, rb, r0, n, k, off, at, d),
        IQ3S => tile::decode_rows::<avx2::Iq3S>(w, rb, r0, n, k, off, at, d),
        _ => unreachable!(),
    }
}

/// Bench helper: one expert's gate + up (decoded 32 rows at a time) and down (64 rows at a time) through the
/// decode-once + micro-kernel path, single thread, results discarded (no SwiGLU / requantization).
///
/// # Safety
/// Slices must hold a whole expert ([640][2560] gate / up, [2560][640] down).
pub unsafe fn small_expert_bench(tgu: GgmlType, td: GgmlType, gw: &[u8], uw: &[u8], dw: &[u8], xq: &avx2::Q8Rows,
    hq: &avx2::Q8Rows, y: &mut [f32]) {
    let (ne, nf) = (2560usize, 640usize);
    TILE_BUF.with(|tb| {
        let mut tb = tb.borrow_mut();
        let tb = &mut *tb;
        let m = xq.rows;
        let rb = tgu.row_bytes(ne);
        let off = tile::offset_ok(tgu);
        let corr = off || matches!(tgu, GgmlType::Q4K | GgmlType::Q5K);
        let ar: [usize; 4] = core::array::from_fn(|i| i.min(m - 1));
        for c in 0..nf / 32 {
            decode_any(tgu, gw.as_ptr(), rb, c * 32, 32, ne, off, 0, &mut tb.0);
            decode_any(tgu, uw.as_ptr(), rb, c * 32, 32, ne, off, 32, &mut tb.0);
            for r in 0..32 {
                let gu = tile::micro_any(&tb.0, [r, 32 + r], xq, ar, m.min(4), off, corr);
                y[c * 32 + r] = gu[0][0] + gu[m.min(4) - 1][1];
            }
        }
        let corr = matches!(td, GgmlType::Q4K | GgmlType::Q5K);
        let ar: [usize; 4] = core::array::from_fn(|i| i.min(hq.rows - 1));
        for c in 0..ne / 64 {
            decode_any(td, dw.as_ptr(), td.row_bytes(nf), c * 64, 64, nf, false, 0, &mut tb.0);
            for r in (0..64).step_by(2) {
                let o = tile::micro_any(&tb.0, [r, r + 1], hq, ar, hq.rows.min(4), false, corr);
                y[c * 64 + r] = o[0][0] + o[0][1];
            }
        }
    });
}
