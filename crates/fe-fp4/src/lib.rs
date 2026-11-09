//! NVFP4 / FP8 prompt (prefill) linears for the ModelOpt Qwen3.8-27B checkpoint (MLP NVFP4 group 16 with e4m3
//! block scales + fp32 `weight_scale_2`; attention / GDN projections FP8 per-tensor).
//!
//! [`PromptLinear::forward`] quantizes bf16 activations (static per-tensor `input_scale`, like vLLM/TRT-LLM)
//! into a [`Workspace`] and runs the Rust block-scaled tensor-core GEMM from `kernels/k-fp4`. Several
//! checkpoint tensors with the same input can be fused along N (gate|up, q|k|v): each output column has its
//! own `alpha` (input scale x weight global scale), the activation scale is the parts' max `input_scale`.
//! `format` is the CPU reference (bit-exact quantizers, dequant, reference GEMM), `lt` the cuBLASLt baseline.
pub mod format;
pub mod lt;
pub mod st;

use anyhow::{bail, ensure, Context as _, Result};
use fe_cuda::{DevPtr, DeviceBuffer, Function, Module, Stream};
pub use st::Checkpoint;

pub static K_FP4_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_fp4.cubin"));

/// Must match kernels/k-fp4.
pub const FP4_SMEM: u32 = 4 * (2 * 8192 + 2 * 1024);
pub const FP8_SMEM: u32 = 4 * (2 * 8192);
pub const FP4_SMEM_N256: u32 = 3 * (3 * 8192 + 3 * 1024);
pub const FP8_SMEM_N256: u32 = 3 * (3 * 8192);
pub const FP4_SMEM_MB: u32 = FP4_SMEM_N256 + 64;
pub const FP8_SMEM_S3: u32 = 3 * (2 * 8192);
pub const FP4_SMEM_S2: u32 = 2 * (2 * 8192 + 2 * 1024);
pub const FP8_SMEM_MB: u32 = FP8_SMEM_N256 + 64;
pub const TILE: usize = 128;

pub struct Kernels {
    /// raster group height in row tiles (default 8, 0 = all row tiles; env FE_FP4_GROUP)
    pub group_m: usize,
    /// GEMM variant for N % 256 == 0 (env FE_FP4_KERNEL): 0 = n128, 1 = n256 (default: bar.sync pipeline,
    /// racecheck clean), 2 = mb (mbarrier pipeline, +5-10% at M >= 2048, flagged by racecheck), 3 = auto
    /// (n256 below 1536 rows, mb above)
    pub variant: u32,
    /// FP8 GEMM variant (env FE_FP8_KERNEL): same numbering as `variant`, 4 = 128 x 128 tiles / 3 stages / two
    /// CTAs a SM (default: +8-11% over n256, bar.sync pipeline); unset falls back to `variant` if FE_FP4_KERNEL is set
    pub variant8: u32,
    pub module: Module,
    pub quant4: Function,
    pub quant8: Function,
    pub gemm4: Function,
    pub gemm8: Function,
    pub gemm4w: Function,
    pub gemm8w: Function,
    pub gemm4mb: Function,
    pub gemm8mb: Function,
    pub gemm8s3: Function,
    pub gemm4s2: Function,
    pub swiglu4: Function,
    pub swiglu4mb: Function,
    pub swiglu_quant: Function,
}
impl Kernels {
    pub fn load() -> Result<Kernels> {
        let module = Module::load(K_FP4_CUBIN)?;
        let k = Kernels {
            group_m: std::env::var("FE_FP4_GROUP").ok().and_then(|v| v.parse().ok()).unwrap_or(8),
            quant4: module.function("fe_fp4_quant")?,
            quant8: module.function("fe_fp8_quant")?,
            gemm4: module.function("fe_gemm_nvfp4")?,
            gemm8: module.function("fe_gemm_fp8")?,
            gemm4w: module.function("fe_gemm_nvfp4_n256")?,
            gemm8w: module.function("fe_gemm_fp8_n256")?,
            gemm4mb: module.function("fe_gemm_nvfp4_mb")?,
            gemm8mb: module.function("fe_gemm_fp8_mb")?,
            gemm8s3: module.function("fe_gemm_fp8_s3")?,
            gemm4s2: module.function("fe_gemm_nvfp4_s2")?,
            swiglu4: module.function("fe_gemm_nvfp4_swiglu")?,
            swiglu4mb: module.function("fe_gemm_nvfp4_swiglu_mb")?,
            swiglu_quant: module.function("fe_fp4_swiglu_quant")?,
            variant: match std::env::var("FE_FP4_KERNEL").ok().as_deref() {
                Some("n128") | Some("0") => 0,
                Some("n256") | Some("1") => 1,
                Some("mb") | Some("2") => 2,
                Some("auto") | Some("3") => 3,
                Some("s") | Some("4") => 4,
                _ => 1,
            },
            variant8: match std::env::var("FE_FP8_KERNEL").ok().as_deref() {
                Some("n128") | Some("0") => 0,
                Some("n256") | Some("1") => 1,
                Some("mb") | Some("2") => 2,
                Some("auto") | Some("3") => 3,
                Some("s3") | Some("s") | Some("4") => 4,
                _ => 4,
            },
            module,
        };
        k.gemm4.set_max_dynamic_smem(FP4_SMEM)?;
        k.gemm8.set_max_dynamic_smem(FP8_SMEM)?;
        k.gemm4w.set_max_dynamic_smem(FP4_SMEM_N256)?;
        k.gemm8w.set_max_dynamic_smem(FP8_SMEM_N256)?;
        k.gemm4mb.set_max_dynamic_smem(FP4_SMEM_MB)?;
        k.gemm8mb.set_max_dynamic_smem(FP8_SMEM_MB)?;
        k.swiglu4.set_max_dynamic_smem(FP4_SMEM_N256)?;
        k.gemm8s3.set_max_dynamic_smem(FP8_SMEM_S3)?;
        k.gemm4s2.set_max_dynamic_smem(FP4_SMEM_S2)?;
        k.swiglu4mb.set_max_dynamic_smem(FP4_SMEM_MB)?;
        Ok(k)
    }

    /// bf16 x[m][lda] -> NVFP4 codes [m][k/2] + scales (`lt_layout`: cuBLASLt VEC16 layout instead of tiled).
    pub fn quant_fp4(&self, s: &Stream, codes: DevPtr, sf: DevPtr, x: DevPtr, m: usize, k: usize, lda: usize, g: f32,
        lt_layout: bool) -> Result<()> {
        let n = (m * k / 16) as u32;
        unsafe {
            s.launch(&self.quant4, n.div_ceil(256), 256, 0, (codes, sf, x, m as u32, k as u32, lda as u32, g,
                lt_layout as u32))?;
        }
        Ok(())
    }
    /// bf16 x[m][lda] -> e4m3 [m][k].
    pub fn quant_fp8(&self, s: &Stream, out: DevPtr, x: DevPtr, m: usize, k: usize, lda: usize, g: f32) -> Result<()> {
        let n = (m * k / 8) as u32;
        unsafe { s.launch(&self.quant8, n.div_ceil(256), 256, 0, (out, x, m as u32, k as u32, lda as u32, g))? };
        Ok(())
    }
    /// Kernel variant for an m x n GEMM under setting `v`.
    fn pick(v: u32, m: usize, n: usize) -> u32 {
        if v == 4 {
            4
        } else if n % 256 != 0 {
            0
        } else if v == 3 {
            if m < 1536 { 1 } else { 2 }
        } else {
            v
        }
    }

    fn group(&self, m: usize) -> u32 {
        let t = m.div_ceil(TILE);
        (if self.group_m == 0 { t } else { self.group_m.min(t) }) as u32
    }

    /// y[m][ldc] = (a . w^T) * alpha[col] (+ res). Inputs already quantized (tiled scales for NVFP4).
    pub fn gemm_fp4(&self, s: &Stream, y: DevPtr, ldc: usize, a: DevPtr, sa: DevPtr, w: DevPtr, sw: DevPtr,
        alpha: DevPtr, res: Option<DevPtr>, m: usize, n: usize, k: usize) -> Result<()> {
        ensure!(n % TILE == 0 && k % 128 == 0, "nvfp4 gemm: n {n} k {k}");
        ensure!(ldc % 8 == 0 && y.0 % 16 == 0 && res.is_none_or(|r| r.0 % 16 == 0), "gemm: y/res must be 16-byte aligned rows");
        let v = Self::pick(self.variant, m, n);
        let (f, bn, smem, threads) = match v {
            0 => (&self.gemm4, TILE, FP4_SMEM, 256),
            4 => (&self.gemm4s2, TILE, FP4_SMEM_S2, 256),
            1 => (&self.gemm4w, 256, FP4_SMEM_N256, 256),
            _ => (&self.gemm4mb, 256, FP4_SMEM_MB, 256),
        };
        let grid = (m.div_ceil(TILE) * (n / bn)) as u32;
        unsafe {
            s.launch(f, grid, threads, smem, (y, a, sa, w, sw, alpha, res.unwrap_or(DevPtr::NULL), m as u32,
                n as u32, k as u32, ldc as u32, self.group(m)))?;
        }
        Ok(())
    }
    /// Fused MLP front over gate|up weights interleaved by 8 rows (n = 2 * inter): act = SwiGLU(a . w^T) as
    /// NVFP4 codes [m][inter/2] + tiled scales in `act` with global scale `qg` (down's `g`).
    pub fn gemm_fp4_swiglu(&self, s: &Stream, act: &Workspace, a: DevPtr, sa: DevPtr, w: DevPtr, sw: DevPtr,
        alpha: DevPtr, m: usize, n: usize, k: usize, qg: f32) -> Result<()> {
        ensure!(n % 256 == 0 && k % 128 == 0 && m <= act.max_rows && n / 2 <= act.max_k, "swiglu gemm: m {m} n {n} k {k}");
        let grid = (m.div_ceil(TILE) * (n / 256)) as u32;
        let (f, smem) = if self.variant == 2 { (&self.swiglu4mb, FP4_SMEM_MB) } else { (&self.swiglu4, FP4_SMEM_N256) };
        unsafe {
            s.launch(f, grid, 256, smem, (a, sa, w, sw, alpha, m as u32, n as u32, k as u32,
                self.group(m), act.codes.ptr(), act.sf.ptr(), qg))?;
        }
        Ok(())
    }
    /// Unfused form of the same epilogue: interleaved gate|up bf16 y [m][n] -> NVFP4 act in `act`.
    pub fn swiglu_quant(&self, s: &Stream, act: &Workspace, y: DevPtr, m: usize, n: usize, qg: f32) -> Result<()> {
        let t = (m * n / 32) as u32;
        unsafe { s.launch(&self.swiglu_quant, t.div_ceil(256), 256, 0, (act.codes.ptr(), act.sf.ptr(), y, m as u32, n as u32, qg))? };
        Ok(())
    }
    pub fn gemm_fp8(&self, s: &Stream, y: DevPtr, ldc: usize, a: DevPtr, w: DevPtr, alpha: DevPtr,
        res: Option<DevPtr>, m: usize, n: usize, k: usize) -> Result<()> {
        ensure!(n % TILE == 0 && k % 64 == 0, "fp8 gemm: n {n} k {k}");
        ensure!(ldc % 8 == 0 && y.0 % 16 == 0 && res.is_none_or(|r| r.0 % 16 == 0), "gemm: y/res must be 16-byte aligned rows");
        let v = Self::pick(self.variant8, m, n);
        let (f, bn, smem, threads) = match v {
            0 => (&self.gemm8, TILE, FP8_SMEM, 256),
            4 => (&self.gemm8s3, TILE, FP8_SMEM_S3, 256),
            1 => (&self.gemm8w, 256, FP8_SMEM_N256, 256),
            _ => (&self.gemm8mb, 256, FP8_SMEM_MB, 256),
        };
        let grid = (m.div_ceil(TILE) * (n / bn)) as u32;
        unsafe {
            s.launch(f, grid, threads, smem, (y, a, w, alpha, res.unwrap_or(DevPtr::NULL), m as u32,
                n as u32, k as u32, ldc as u32, self.group(m)))?;
        }
        Ok(())
    }
}

/// Quantized-activation scratch shared by all prompt linears: codes [rows][k] bytes (enough for FP8; NVFP4
/// uses half) + NVFP4 scales [rows padded to 128][k/16].
pub struct Workspace {
    pub max_rows: usize,
    pub max_k: usize,
    pub codes: DeviceBuffer<u8>,
    pub sf: DeviceBuffer<u8>,
}
impl Workspace {
    pub fn new(max_rows: usize, max_k: usize) -> Result<Workspace> {
        let rp = max_rows.div_ceil(TILE) * TILE;
        Ok(Workspace { max_rows, max_k, codes: DeviceBuffer::alloc(rp * max_k)?, sf: DeviceBuffer::alloc(rp * max_k / 16)? })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Nvfp4,
    Fp8,
}

/// One logical checkpoint tensor inside a fused [`PromptLinear`].
#[derive(Clone, Debug)]
pub struct Part {
    pub name: String,
    pub n0: usize,
    pub n: usize,
    /// input_scale (activation) x weight global scale (`weight_scale_2` / FP8 `weight_scale`)
    pub alpha: f32,
    pub input_scale: f32,
}

/// The part of a checkpoint tensor a tensor-parallel rank keeps: output rows (N) in `rows` (concatenated in
/// order, each a multiple of [`TILE`]) and input columns `k.0..k.1` (multiples of 128).
#[derive(Clone, Debug)]
pub struct Slice {
    pub rows: Vec<(usize, usize)>,
    pub k: (usize, usize),
}

/// Rows `keep` (ranges, concatenated) and bytes `c0..c1` of each of a row-major [rows][row_len] byte matrix.
fn select(src: &[u8], row_len: usize, keep: &[(usize, usize)], c0: usize, c1: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(keep.iter().map(|r| r.1 - r.0).sum::<usize>() * (c1 - c0));
    for &(a, b) in keep {
        for r in a..b {
            out.extend_from_slice(&src[r * row_len + c0..r * row_len + c1]);
        }
    }
    out
}

/// y[M][N] = x[M][K] . W[N][K]^T for a ModelOpt NVFP4 or FP8 weight (possibly several fused along N).
pub struct PromptLinear {
    pub kind: Kind,
    pub n: usize,
    pub k: usize,
    pub name: String,
    pub parts: Vec<Part>,
    /// e2m1 codes [n][k/2] or e4m3 [n][k]
    pub w: DeviceBuffer<u8>,
    /// NVFP4 block scales, ferrite tiled layout
    pub sw: Option<DeviceBuffer<u8>>,
    /// NVFP4 block scales in cuBLASLt's layout (only with `load_opts(.., true)`)
    pub sw_lt: Option<DeviceBuffer<u8>>,
    /// per output column: act scale x weight global scale
    pub alpha: DeviceBuffer<f32>,
    /// activation quantization multiplier = 1 / max(input_scale)
    pub g: f32,
    /// gate|up interleaved in blocks of 8 rows for [`PromptLinear::forward_swiglu`] (see `load_swiglu`)
    pub swiglu: bool,
}

impl PromptLinear {
    /// Load and fuse `parts` (layer names without `.weight`, e.g. `model.language_model.layers.0.mlp.gate_proj`).
    pub fn load(ck: &Checkpoint, parts: &[&str]) -> Result<PromptLinear> {
        Self::load_opts(ck, parts, false)
    }

    /// `with_lt`: also keep cuBLASLt-layout scales so [`PromptLinear::forward_lt`] works.
    pub fn load_opts(ck: &Checkpoint, parts: &[&str], with_lt: bool) -> Result<PromptLinear> {
        Self::load_sliced(ck, parts, with_lt, None)
    }

    /// [`PromptLinear::load_opts`] of one [`Slice`] of every part (tensor parallelism; `None`: whole tensors).
    pub fn load_sliced(ck: &Checkpoint, parts: &[&str], with_lt: bool, slices: Option<&[Slice]>) -> Result<PromptLinear> {
        ensure!(!parts.is_empty(), "no parts");
        ensure!(slices.is_none_or(|v| v.len() == parts.len()), "{}: {} parts, slices for {}", parts[0], parts.len(),
            slices.map_or(0, |v| v.len()));
        let algo = ck.algo(parts[0]).with_context(|| format!("{}: not NVFP4/FP8", parts[0]))?;
        let kind = match algo.as_str() {
            "NVFP4" => Kind::Nvfp4,
            "FP8" => Kind::Fp8,
            a => bail!("{}: unsupported quant algo {a}", parts[0]),
        };
        let mut k = 0;
        let mut n = 0;
        let mut ps = Vec::new();
        for (i, p) in parts.iter().enumerate() {
            ensure!(ck.algo(p).as_deref() == Some(algo.as_str()), "{p}: mixed quant algos in one fused linear");
            let t = ck.get(&format!("{p}.weight"))?;
            let (fpn, fpk) = (t.shape[0], if kind == Kind::Nvfp4 { t.shape[1] * 2 } else { t.shape[1] });
            let (pn, pk) = match slices.map(|v| &v[i]) {
                Some(sl) => {
                    ensure!(sl.rows.iter().all(|r| r.0 < r.1 && r.1 <= fpn && r.0 % TILE == 0 && r.1 % TILE == 0)
                        && sl.k.0 < sl.k.1 && sl.k.1 <= fpk && sl.k.0 % 128 == 0 && sl.k.1 % 128 == 0,
                        "{p}: slice {sl:?} of {fpn}x{fpk}");
                    (sl.rows.iter().map(|r| r.1 - r.0).sum(), sl.k.1 - sl.k.0)
                }
                None => (fpn, fpk),
            };
            if i == 0 {
                k = pk;
            }
            ensure!(pk == k, "{p}: k {pk} != {k}");
            ensure!(pn % TILE == 0 && n % TILE == 0, "{p}: n {pn} not a multiple of {TILE}");
            let wscale = if kind == Kind::Nvfp4 { ck.scalar(&format!("{p}.weight_scale_2"))? } else {
                ck.scalar(&format!("{p}.weight_scale"))?
            };
            let input_scale = ck.scalar(&format!("{p}.input_scale"))?;
            ps.push(Part { name: p.to_string(), n0: n, n: pn, alpha: wscale, input_scale });
            n += pn;
        }
        ensure!(k % 128 == 0, "{}: k {k} not a multiple of 128", parts[0]);
        let in_scale = ps.iter().map(|p| p.input_scale).fold(0.0f32, f32::max);
        ensure!(in_scale > 0.0, "{}: input_scale 0", parts[0]);
        for p in ps.iter_mut() {
            p.alpha *= in_scale;
        }
        let row_bytes = if kind == Kind::Nvfp4 { k / 2 } else { k };
        let w = DeviceBuffer::<u8>::alloc(n * row_bytes)?;
        let mut sf_all = Vec::new();
        // codes bytes / scale bytes a k column (NVFP4: 2 codes a byte, a scale a 16-group)
        let (cb, sb) = if kind == Kind::Nvfp4 { (2, 16) } else { (1, 16) };
        for (i, p) in ps.iter().enumerate() {
            let mut codes = ck.read(&format!("{}.weight", p.name))?;
            let mut sf = if kind == Kind::Nvfp4 { ck.read(&format!("{}.weight_scale", p.name))? } else { Vec::new() };
            if let Some(sl) = slices.map(|v| &v[i]) {
                let fpn = ck.get(&format!("{}.weight", p.name))?.shape[0];
                ensure!(codes.len() % fpn == 0, "{}: weight bytes", p.name);
                let rl = codes.len() / fpn;
                codes = select(&codes, rl, &sl.rows, sl.k.0 / cb, sl.k.1 / cb);
                if kind == Kind::Nvfp4 {
                    let sl_len = sf.len() / fpn;
                    sf = select(&sf, sl_len, &sl.rows, sl.k.0 / sb, sl.k.1 / sb);
                }
            }
            ensure!(codes.len() == p.n * row_bytes, "{}: weight bytes", p.name);
            fe_cuda::memcpy_htod(w.at(p.n0 * row_bytes), &codes)?;
            if kind == Kind::Nvfp4 {
                ensure!(sf.len() == p.n * k / 16, "{}: weight_scale bytes", p.name);
                sf_all.extend_from_slice(&sf);
            }
        }
        let (sw, sw_lt) = if kind == Kind::Nvfp4 {
            let t = format::tile_scales(&sf_all, n, k / 16);
            let lt = if with_lt { Some(DeviceBuffer::from_slice(&format::cublaslt_scales(&sf_all, n, k / 16))?) } else {
                None
            };
            (Some(DeviceBuffer::from_slice(&t)?), lt)
        } else {
            (None, None)
        };
        let mut alpha = vec![0.0f32; n];
        for p in &ps {
            alpha[p.n0..p.n0 + p.n].fill(p.alpha);
        }
        Ok(PromptLinear {
            kind,
            n,
            k,
            name: parts.join("+"),
            parts: ps,
            w,
            sw,
            sw_lt,
            alpha: DeviceBuffer::from_slice(&alpha)?,
            g: 1.0 / in_scale,
            swiglu: false,
        })
    }

    /// gate and up (NVFP4, same K) fused with their rows interleaved in blocks of 8 (gate f..f+7, up f..f+7,
    /// ...) for the fused SwiGLU GEMM ([`PromptLinear::forward_swiglu`]). Its plain `forward` output is that
    /// interleaved [m][2 inter] layout.
    pub fn load_swiglu(ck: &Checkpoint, gate: &str, up: &str) -> Result<PromptLinear> {
        Self::load_swiglu_sliced(ck, gate, up, None)
    }

    /// [`PromptLinear::load_swiglu`] of one [`Slice`] each of gate and up (the same columns of both).
    pub fn load_swiglu_sliced(ck: &Checkpoint, gate: &str, up: &str, slices: Option<&[Slice]>) -> Result<PromptLinear> {
        let mut l = Self::load_sliced(ck, &[gate, up], false, slices)?;
        ensure!(l.kind == Kind::Nvfp4 && l.parts.len() == 2 && l.parts[0].n == l.parts[1].n, "{}: not an NVFP4 gate|up pair", l.name);
        ensure!(l.n % 256 == 0, "{}: n {}", l.name, l.n);
        let (n, k, inter) = (l.n, l.k, l.n / 2);
        let perm = |r: usize| -> usize {
            // interleaved row r -> source row of the gate|up concatenation
            let (blk, w) = (r / 16, r % 16);
            if w < 8 { blk * 8 + w } else { inter + blk * 8 + w - 8 }
        };
        let rb = k / 2;
        let codes = l.w.to_vec()?;
        let mut ic = vec![0u8; codes.len()];
        for r in 0..n {
            let src = perm(r);
            ic[r * rb..(r + 1) * rb].copy_from_slice(&codes[src * rb..(src + 1) * rb]);
        }
        l.w.copy_from(&ic)?;
        let kb = k / 16;
        let st = format::untile_scales(&l.sw.as_ref().unwrap().to_vec()?, n, kb);
        let mut is = vec![0u8; st.len()];
        for r in 0..n {
            let src = perm(r);
            is[r * kb..(r + 1) * kb].copy_from_slice(&st[src * kb..(src + 1) * kb]);
        }
        l.sw.as_mut().unwrap().copy_from(&format::tile_scales(&is, n, kb))?;
        let al = l.alpha.to_vec()?;
        let ia: Vec<f32> = (0..n).map(|r| al[perm(r)]).collect();
        l.alpha.copy_from(&ia)?;
        l.sw_lt = None;
        l.swiglu = true;
        l.name = format!("{}(swiglu)", l.name);
        Ok(l)
    }

    /// Fused MLP front for prompt rows: act = SwiGLU(x . [gate|up]^T), written NVFP4-quantized with `down_g`
    /// (= `down.g`) into `act`; then `down.gemm(kern, s, act, y, ldy, m, res)` finishes the MLP. Bit-identical to
    /// forward + silu_mul (core's fe_silu_mul semantics) + down's own quantization. `ws` holds x quantized.
    pub fn forward_swiglu(&self, kern: &Kernels, s: &Stream, ws: &Workspace, act: &Workspace, x: DevPtr, lda: usize,
        m: usize, down_g: f32) -> Result<()> {
        ensure!(self.swiglu, "{}: not loaded with load_swiglu", self.name);
        if m == 0 {
            return Ok(());
        }
        self.quantize(kern, s, ws, x, lda, m)?;
        kern.gemm_fp4_swiglu(s, act, ws.codes.ptr(), ws.sf.ptr(), self.w.ptr(), self.sw.as_ref().unwrap().ptr(),
            self.alpha.ptr(), m, self.n, self.k, down_g)
    }

    /// y (row stride ldy) = x (bf16, row stride lda) . W^T for `m` rows; with `res` (stride ldy, may equal y):
    /// y = bf16(bf16(x.W^T) + res). Uses `ws` for the quantized activations (m <= ws.max_rows, k <= ws.max_k).
    pub fn forward(&self, kern: &Kernels, s: &Stream, ws: &Workspace, y: DevPtr, ldy: usize, x: DevPtr, lda: usize,
        m: usize, res: Option<DevPtr>) -> Result<()> {
        self.quantize(kern, s, ws, x, lda, m)?;
        self.gemm(kern, s, ws, y, ldy, m, res)
    }

    /// Quantize x (bf16 [m][lda]) into `ws` with this linear's activation scale. Linears with the same
    /// `kind`, `k` and `g` can then share it: one `quantize`, several [`PromptLinear::gemm`].
    pub fn quantize(&self, kern: &Kernels, s: &Stream, ws: &Workspace, x: DevPtr, lda: usize, m: usize) -> Result<()> {
        ensure!(m <= ws.max_rows && self.k <= ws.max_k, "{}: workspace {}x{} < {m}x{}", self.name, ws.max_rows,
            ws.max_k, self.k);
        if m == 0 {
            return Ok(());
        }
        match self.kind {
            Kind::Nvfp4 => kern.quant_fp4(s, ws.codes.ptr(), ws.sf.ptr(), x, m, self.k, lda, self.g, false),
            Kind::Fp8 => kern.quant_fp8(s, ws.codes.ptr(), x, m, self.k, lda, self.g),
        }
    }

    /// The GEMM over activations already quantized into `ws` (by `quantize` of a compatible linear).
    pub fn gemm(&self, kern: &Kernels, s: &Stream, ws: &Workspace, y: DevPtr, ldy: usize, m: usize,
        res: Option<DevPtr>) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        match self.kind {
            Kind::Nvfp4 => kern.gemm_fp4(s, y, ldy, ws.codes.ptr(), ws.sf.ptr(), self.w.ptr(),
                self.sw.as_ref().unwrap().ptr(), self.alpha.ptr(), res, m, self.n, self.k),
            Kind::Fp8 => kern.gemm_fp8(s, y, ldy, ws.codes.ptr(), self.w.ptr(), self.alpha.ptr(), res, m, self.n, self.k),
        }
    }

    /// Same contract through cuBLASLt (one matmul per fused part; no residual: `res` must be None).
    /// Needs `load_opts(.., true)` for NVFP4. `plans` caches one plan per (part, m).
    pub fn forward_lt(&self, kern: &Kernels, lt: &lt::Lt, plans: &mut Vec<((usize, usize), lt::Plan)>, s: &Stream,
        ws: &Workspace, y: DevPtr, ldy: usize, x: DevPtr, lda: usize, m: usize) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        let fp4 = self.kind == Kind::Nvfp4;
        if fp4 {
            kern.quant_fp4(s, ws.codes.ptr(), ws.sf.ptr(), x, m, self.k, lda, self.g, true)?;
        } else {
            kern.quant_fp8(s, ws.codes.ptr(), x, m, self.k, lda, self.g)?;
        }
        let row_bytes = if fp4 { self.k / 2 } else { self.k };
        let ct = (self.k / 16).div_ceil(4);
        for (i, p) in self.parts.iter().enumerate() {
            let sw = if fp4 {
                self.sw_lt.as_ref().context("forward_lt: loaded without cuBLASLt scales")?.ptr().byte_add(p.n0 / 128 * ct * 512)
            } else {
                DevPtr::NULL
            };
            let idx = match plans.iter().position(|(key, _)| *key == (i, m)) {
                Some(j) => j,
                None => {
                    plans.push(((i, m), lt.plan(fp4, m, p.n, self.k, ldy, ws.sf.ptr(), sw)?));
                    plans.len() - 1
                }
            };
            unsafe {
                lt.run(&plans[idx].1, s, y.add::<u16>(p.n0), ws.codes.ptr(), ws.sf.ptr(), self.w.ptr().byte_add(p.n0 * row_bytes),
                    sw, p.alpha)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod slice_tests {
    #[test]
    fn select_rows_and_cols() {
        let src: Vec<u8> = (0..40).collect(); // 5 rows x 8 bytes
        assert_eq!(super::select(&src, 8, &[(0, 1), (3, 5)], 2, 6), [2, 3, 4, 5, 26, 27, 28, 29, 34, 35, 36, 37]);
        assert_eq!(super::select(&src, 8, &[(0, 5)], 0, 8), src);
    }
}
