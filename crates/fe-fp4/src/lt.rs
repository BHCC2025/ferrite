//! cuBLASLt NVFP4 / FP8 matmul through FFI: the measured baseline for the Rust GEMMs and the fallback
//! path (NON-RUST.md). Row-major C[M][N] = A[M][K] . W[N][K]^T maps onto cuBLASLt's column-major
//! D[N][M] = op(W)^T... i.e. cuBLASLt "A" = W (K x N, ld K, TRANSA = T), "B" = activations (K x M, ld K,
//! TRANSB = N), D = N x M with ld = ldc. NVFP4 scale tensors use cuBLASLt's VEC16_UE4M3 128x4 tiled layout
//! (weights: rows = N; activations: rows = M), see `format::cublaslt_scales` / `fe_fp4_quant` layout 1.
//! Gotchas: the FP4 heuristic needs A/B scale pointers set before planning (INVALID_VALUE otherwise);
//! the library is linked, never dlclose'd.
use anyhow::{bail, Result};
use fe_cuda::{DevPtr, DeviceBuffer, Stream};
use std::ffi::c_void;

type P = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct Algo {
    data: [u64; 8],
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Heur {
    algo: Algo,
    workspace: usize,
    state: i32,
    waves: f32,
    reserved: [i32; 4],
}

#[link(name = "cublasLt")]
unsafe extern "C" {
    fn cublasLtCreate(h: *mut P) -> i32;
    fn cublasLtMatmulDescCreate(d: *mut P, compute: i32, scale: i32) -> i32;
    fn cublasLtMatmulDescDestroy(d: P) -> i32;
    fn cublasLtMatmulDescSetAttribute(d: P, attr: i32, buf: *const c_void, size: usize) -> i32;
    fn cublasLtMatrixLayoutCreate(l: *mut P, ty: i32, rows: u64, cols: u64, ld: i64) -> i32;
    fn cublasLtMatrixLayoutDestroy(l: P) -> i32;
    fn cublasLtMatmulPreferenceCreate(p: *mut P) -> i32;
    fn cublasLtMatmulPreferenceDestroy(p: P) -> i32;
    fn cublasLtMatmulPreferenceSetAttribute(p: P, attr: i32, buf: *const c_void, size: usize) -> i32;
    fn cublasLtMatmulAlgoGetHeuristic(h: P, d: P, a: P, b: P, c: P, dd: P, pref: P, req: i32, res: *mut Heur,
        n: *mut i32) -> i32;
    fn cublasLtMatmul(h: P, d: P, alpha: *const c_void, a: *const c_void, ad: P, b: *const c_void, bd: P,
        beta: *const c_void, c: *const c_void, cd: P, dd: *mut c_void, ddd: P, algo: *const Algo, ws: *mut c_void,
        ws_size: usize, stream: P) -> i32;
}

const COMPUTE_32F: i32 = 68;
const R_32F: i32 = 0;
const R_16BF: i32 = 14;
const R_8F_E4M3: i32 = 28;
const R_4F_E2M1: i32 = 33;
const DESC_TRANSA: i32 = 3;
const DESC_TRANSB: i32 = 4;
const DESC_A_SCALE_POINTER: i32 = 17;
const DESC_B_SCALE_POINTER: i32 = 18;
const DESC_A_SCALE_MODE: i32 = 31;
const DESC_B_SCALE_MODE: i32 = 32;
const SCALE_VEC16_UE4M3: i32 = 1;
const PREF_MAX_WORKSPACE: i32 = 1;
const OP_N: i32 = 0;
const OP_T: i32 = 1;

fn ck(r: i32, what: &str) -> Result<()> {
    if r != 0 {
        bail!("cuBLASLt {what}: status {r}");
    }
    Ok(())
}

pub struct Lt {
    h: P,
    ws: DeviceBuffer<u8>,
}
unsafe impl Send for Lt {}
unsafe impl Sync for Lt {}

/// One planned matmul shape (descriptor, layouts, algo).
pub struct Plan {
    pub fp4: bool,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    desc: P,
    la: P,
    lb: P,
    ld: P,
    algo: Algo,
    ws_size: usize,
}
unsafe impl Send for Plan {}
unsafe impl Sync for Plan {}
impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            cublasLtMatmulDescDestroy(self.desc);
            cublasLtMatrixLayoutDestroy(self.la);
            cublasLtMatrixLayoutDestroy(self.lb);
            cublasLtMatrixLayoutDestroy(self.ld);
        }
    }
}

const WS_BYTES: usize = 64 << 20;

impl Lt {
    pub fn new() -> Result<Lt> {
        let mut h = std::ptr::null_mut();
        ck(unsafe { cublasLtCreate(&mut h) }, "create")?;
        Ok(Lt { h, ws: DeviceBuffer::alloc(WS_BYTES)? })
    }

    unsafe fn set(d: P, attr: i32, v: &impl Copy) -> Result<()> {
        let r = unsafe { cublasLtMatmulDescSetAttribute(d, attr, v as *const _ as *const c_void, std::mem::size_of_val(v)) };
        ck(r, "set attribute")
    }

    /// Plan C[m][n] (row stride ldc) = A[m][k] . W[n][k]^T. `sa`/`sw` = any valid scale buffers (FP4 only;
    /// the real ones are passed to `run`).
    pub fn plan(&self, fp4: bool, m: usize, n: usize, k: usize, ldc: usize, sa: DevPtr, sw: DevPtr) -> Result<Plan> {
        unsafe {
            let mut desc = std::ptr::null_mut();
            ck(cublasLtMatmulDescCreate(&mut desc, COMPUTE_32F, R_32F), "desc")?;
            Self::set(desc, DESC_TRANSA, &OP_T)?;
            Self::set(desc, DESC_TRANSB, &OP_N)?;
            let ty = if fp4 { R_4F_E2M1 } else { R_8F_E4M3 };
            if fp4 {
                Self::set(desc, DESC_A_SCALE_MODE, &SCALE_VEC16_UE4M3)?;
                Self::set(desc, DESC_B_SCALE_MODE, &SCALE_VEC16_UE4M3)?;
                Self::set(desc, DESC_A_SCALE_POINTER, &sw.0)?;
                Self::set(desc, DESC_B_SCALE_POINTER, &sa.0)?;
            }
            let (mut la, mut lb, mut ld) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            ck(cublasLtMatrixLayoutCreate(&mut la, ty, k as u64, n as u64, k as i64), "layout A")?;
            ck(cublasLtMatrixLayoutCreate(&mut lb, ty, k as u64, m as u64, k as i64), "layout B")?;
            ck(cublasLtMatrixLayoutCreate(&mut ld, R_16BF, n as u64, m as u64, ldc as i64), "layout D")?;
            let mut pref = std::ptr::null_mut();
            ck(cublasLtMatmulPreferenceCreate(&mut pref), "pref")?;
            let wsb = WS_BYTES as u64;
            ck(cublasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WORKSPACE, &wsb as *const u64 as *const c_void, 8),
                "pref ws")?;
            let mut res = [Heur { algo: Algo { data: [0; 8] }, workspace: 0, state: 0, waves: 0.0, reserved: [0; 4] }; 1];
            let mut cnt = 0;
            let r = cublasLtMatmulAlgoGetHeuristic(self.h, desc, la, lb, ld, ld, pref, 1, res.as_mut_ptr(), &mut cnt);
            cublasLtMatmulPreferenceDestroy(pref);
            ck(r, "heuristic")?;
            if cnt < 1 {
                bail!("cuBLASLt: no algo for {} m{m} n{n} k{k}", if fp4 { "nvfp4" } else { "fp8" });
            }
            Ok(Plan { fp4, m, n, k, desc, la, lb, ld, algo: res[0].algo, ws_size: res[0].workspace })
        }
    }

    /// Run a plan: c = alpha * (a . w^T). For FP4, `sa`/`sw` are cuBLASLt-layout scales.
    pub unsafe fn run(&self, p: &Plan, s: &Stream, c: DevPtr, a: DevPtr, sa: DevPtr, w: DevPtr, sw: DevPtr,
        alpha: f32) -> Result<()> {
        if p.fp4 {
            unsafe {
                Self::set(p.desc, DESC_A_SCALE_POINTER, &sw.0)?;
                Self::set(p.desc, DESC_B_SCALE_POINTER, &sa.0)?;
            }
        }
        let beta = 0.0f32;
        let r = unsafe { cublasLtMatmul(self.h, p.desc, &alpha as *const f32 as *const c_void, w.0 as *const c_void, p.la,
            a.0 as *const c_void, p.lb, &beta as *const f32 as *const c_void, c.0 as *const c_void, p.ld,
            c.0 as *mut c_void, p.ld, &p.algo, self.ws.ptr().0 as *mut c_void, p.ws_size, s.raw() as P) };
        ck(r, "matmul")
    }
}
