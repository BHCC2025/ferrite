//! GPU side: the k-exl3 module, device tensors, launch wrappers (over fe-cuda).

pub mod pdl;

use fe_cuda::{DevPtr, DeviceBuffer, Function, Module, Stream};

use crate::cpu::Layout;
use crate::format::Exl3Tensor;
use crate::st::Checkpoint;
use crate::{Error, Result};

/// The k-exl3 cubin for sm_121a.
pub static K_EXL3_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_exl3.cubin"));

/// Device pointers of one EXL3 tensor: trellis u32 words, suh fp16 [k], svh fp16 [n].
#[derive(Clone, Copy, Debug)]
pub struct Exl3View {
    pub k: u32,
    pub n: u32,
    pub bits: u32,
    pub trellis: DevPtr,
    pub suh: DevPtr,
    pub svh: DevPtr,
    /// trellis in the strips layout [n/128][k/16][8][tw] (decode GEMV only) instead of the stored one
    pub strips: bool,
}

impl Exl3View {
    /// (words between k steps, words between 128-column blocks) of the trellis layout.
    pub fn strides(&self) -> (u32, u32) {
        let tw = 8 * self.bits;
        if self.strips { (8 * tw, self.k / 16 * 8 * tw) } else { (self.n / 16 * tw, 8 * tw) }
    }
}

/// An EXL3 tensor uploaded to device memory (owns its buffers).
pub struct DeviceExl3 {
    pub name: String,
    pub trellis: DeviceBuffer<u32>,
    pub suh: DeviceBuffer<u16>,
    pub svh: DeviceBuffer<u16>,
    pub view: Exl3View,
}

impl DeviceExl3 {
    /// Upload in the stored layout (what reconstruct reads).
    pub fn upload(t: &Exl3Tensor) -> Result<Self> {
        Self::upload_as(t, false)
    }

    /// Upload for the decode GEMV: strips layout.
    pub fn upload_strips(t: &Exl3Tensor) -> Result<Self> {
        Self::upload_as(t, true)
    }

    pub fn upload_as(t: &Exl3Tensor, strips: bool) -> Result<Self> {
        let trellis = if strips { DeviceBuffer::from_slice(&t.strips())? } else { DeviceBuffer::from_slice(&t.trellis)? };
        let suh = DeviceBuffer::from_slice(&t.suh)?;
        let svh = DeviceBuffer::from_slice(&t.svh)?;
        let view = Exl3View {
            k: t.k as u32,
            n: t.n as u32,
            bits: t.bits,
            trellis: trellis.ptr(),
            suh: suh.ptr(),
            svh: svh.ptr(),
            strips,
        };
        Ok(Self { name: t.name.clone(), trellis, suh, svh, view })
    }

    pub fn load(ck: &Checkpoint, base: &str) -> Result<Self> {
        Self::upload(&Exl3Tensor::load(ck, base)?)
    }
}

/// The k-exl3 module loaded into the current context.
pub struct Kernels {
    pub module: Module,
    pub recon_hat_f32: Function,
    pub recon_hat_f16: Function,
    pub had_rows: Function,
    pub had_cols_bf16: Function,
    pub rot_in: Function,
}

impl Kernels {
    /// Load the embedded cubin (or the file named by FE_EXL3_CUBIN, for kernel experiments).
    pub fn load() -> Result<Self> {
        let module = match std::env::var("FE_EXL3_CUBIN") {
            Ok(path) => Module::load(&std::fs::read(&path)?)?,
            Err(_) => Module::load(K_EXL3_CUBIN)?,
        };
        Ok(Self {
            recon_hat_f32: module.function("exl3_recon_hat_f32")?,
            recon_hat_f16: module.function("exl3_recon_hat_f16")?,
            had_rows: module.function("exl3_had_rows")?,
            had_cols_bf16: module.function("exl3_had_cols_bf16")?,
            rot_in: module.function("exl3_rot_in")?,
            module,
        })
    }

    /// f32 scratch bytes for a single full-width reconstruct pass; any smaller scratch of at least
    /// `k * 128 * 4` bytes works too (columns are done in slices).
    pub fn recon_scratch_bytes(w: &Exl3View) -> usize {
        w.k as usize * w.n as usize * 4
    }

    /// Original-basis W as bf16 into `out` (k * n elements; [k][n] for Layout::KN, [n][k] HF layout for NK).
    /// `scratch`: f32 workspace of `scratch_bytes` (>= k * 128 * 4). Asynchronous on `s`.
    pub fn reconstruct_bf16(
        &self,
        s: &Stream,
        w: &Exl3View,
        out: DevPtr,
        layout: Layout,
        scratch: DevPtr,
        scratch_bytes: usize,
    ) -> Result<()> {
        let (k, n) = (w.k as usize, w.n as usize);
        if k % 128 != 0 || n % 128 != 0 || w.strips {
            return Err(Error::msg(format!("reconstruct_bf16: {k}x{n} not 128-aligned or strips layout")));
        }
        let max_ns = (scratch_bytes / (k * 4)) / 128 * 128;
        if max_ns == 0 {
            return Err(Error::msg(format!("reconstruct_bf16: scratch {scratch_bytes} < {}", k * 128 * 4)));
        }
        let (ld, nk) = match layout {
            Layout::KN => (n as u32, 0u32),
            Layout::NK => (k as u32, 1u32),
        };
        let mut n0 = 0;
        while n0 < n {
            let ns = max_ns.min(n - n0);
            let (ns32, n032, k32) = (ns as u32, n0 as u32, k as u32);
            unsafe {
                s.launch(
                    &self.recon_hat_f32,
                    (ns32 / 128, k32 / 16),
                    256,
                    0,
                    (scratch, w.trellis, w.n / 16, n032 / 16, ns32, w.bits),
                )?;
                s.launch(&self.had_rows, (ns32 / 128, k32), 32, 0, (scratch, ns32, w.svh, n032))?;
                s.launch(
                    &self.had_cols_bf16,
                    (ns32 / 32, k32 / 128),
                    128,
                    0,
                    (scratch, out, ns32, w.suh, n032, ld, nk),
                )?;
            }
            n0 += ns;
        }
        Ok(())
    }

    /// Rotated-basis W_hat as fp16 [k][n] (exllamav3's `reconstruct`). Asynchronous on `s`.
    pub fn reconstruct_hat_f16(&self, s: &Stream, w: &Exl3View, out: DevPtr) -> Result<()> {
        unsafe {
            s.launch(&self.recon_hat_f16, (w.n / 128, w.k / 16), 256, 0, (out, w.trellis, w.n / 16, 0u32, w.n, w.bits))?;
        }
        Ok(())
    }

    /// bf16 weights of an in-memory tensor (e.g. a tensor-parallel slice): uploads it in the stored layout,
    /// reconstructs into a new buffer (k * n elements in `layout`), frees the trellis. Synchronizes `s`.
    pub fn tensor_bf16(&self, s: &Stream, t: &Exl3Tensor, layout: Layout, scratch: &DeviceBuffer<f32>) -> Result<DeviceBuffer<u16>> {
        let d = DeviceExl3::upload(t)?;
        let out = DeviceBuffer::<u16>::alloc(t.k * t.n)?;
        self.reconstruct_bf16(s, &d.view, out.ptr(), layout, scratch.ptr(), scratch.bytes())?;
        s.sync()?;
        Ok(out)
    }

    /// Model-load convenience: read `base` from the checkpoint, reconstruct it into a new bf16 device buffer
    /// (k * n elements in `layout`), drop the trellis. Synchronizes `s`. Returns (weight, k, n).
    pub fn load_bf16(
        &self,
        s: &Stream,
        ck: &Checkpoint,
        base: &str,
        layout: Layout,
        scratch: &DeviceBuffer<f32>,
    ) -> Result<(DeviceBuffer<u16>, usize, usize)> {
        let d = DeviceExl3::load(ck, base)?;
        let (k, n) = (d.view.k as usize, d.view.n as usize);
        let out = DeviceBuffer::<u16>::alloc(k * n)?;
        self.reconstruct_bf16(s, &d.view, out.ptr(), layout, scratch.ptr(), scratch.bytes())?;
        s.sync()?;
        Ok((out, k, n))
    }
}
