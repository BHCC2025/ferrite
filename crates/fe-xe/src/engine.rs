//! The device side of the forward: weights on the B70, per-sequence state, window scratch, and the 1..16-row window forward in
//! arc-engine's launch order (`qwen_win.zig Win.forward`, XQ weights, q8 KV, ARC_DEC=1). Kernels come from [`Kx`] (Rust kernels, or
//! the reference SPIR-V in tests), called with arc-engine's names, arguments and grids.
//!
//! A window of m <= 16 rows is bit-identical per row to arc-engine's one-token forward (row invariance), so this path alone is a
//! complete, exact forward: plain decode (m = 1), verify windows, and prompts in 16-row windows. Prompt windows of up to `rcap` rows
//! (4096 when serving) take arc-engine's prompt path: the XQ prefill GEMM (XQ_PF=f16: k-major fp16 prepass, per-call fp16 weight decode,
//! work-group-tiled DPAS GEMM), the Hadamard-int8 prompt attention (ARC_PFX=i8) and the row-sequential GDN kernels over all rows.
use crate::config::Config;
use crate::kx::Kx;
use crate::rope;
use crate::state::KvMode;
use crate::weights::{HostTensor, Planner};
use anyhow::{bail, ensure, Context as _, Result};
use fe_xq::pack::{variant_for, VARIANTS};
use fe_xq::Checkpoint;
use fe_ze::{DevPtr, DeviceBuffer, HostBuffer, KernelArg, Local, Queue, Ze};
use std::collections::HashMap;
use std::sync::Arc;

pub const HIDDEN: u32 = 5120;
pub const INTER: u32 = 17408;
pub const HEADS: u32 = 24;
pub const KV_HEADS: u32 = 4;
pub const HEAD_DIM: u32 = 256;
pub const Q_DIM: u32 = HEADS * HEAD_DIM;
pub const KV_DIM: u32 = KV_HEADS * HEAD_DIM;
pub const GDN_QKV: u32 = 10240;
pub const GDN_V: u32 = 6144;
pub const GDN_HEADS: u32 = 48;
pub const MAX_ROWS: u32 = 16;
pub const MAX_CHUNKS: u32 = 128;
pub const ARGMAX_PARTS: u32 = 64;
pub const KV_CHUNK: u32 = 1024;
/// head_rows value of a prompt window that keeps the normalised hidden rows for [`Engine::head_batch`] (arc-engine keep_hidden)
pub const KEEP_HIDDEN: u32 = 0xffff;
/// keys a prompt-attention range (expanded K' / V' scratch: 16384 x 4 x 256 x 2 bytes each)
pub const PF_RANGE: u32 = 16384;
const PF_STATE_BYTES: usize = 8 * 1632 * 4;

/// The device, its in-order queue and a pinned staging buffer for uploads.
pub struct Dev {
    pub ze: Arc<Ze>,
    pub q: Queue,
    stage: std::cell::RefCell<HostBuffer<u8>>,
    pub bytes: std::cell::Cell<usize>,
}

const STAGE: usize = 64 << 20;

impl Dev {
    pub fn new() -> Result<Dev> {
        let ze = Ze::new().map_err(|e| anyhow::anyhow!("{e}"))?;
        let q = Queue::new(&ze).map_err(|e| anyhow::anyhow!("{e}"))?;
        let stage = HostBuffer::<u8>::alloc(&ze, STAGE).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Dev { ze, q, stage: stage.into(), bytes: 0.into() })
    }

    pub fn alloc<T: Copy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        let b = DeviceBuffer::<T>::alloc(&self.ze, len.max(1)).map_err(|e| anyhow::anyhow!("alloc {len}: {e}"))?;
        self.bytes.set(self.bytes.get() + b.bytes());
        Ok(b)
    }

    pub fn zeros<T: Copy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        let b = self.alloc::<T>(len)?;
        self.q.fill(b.ptr(), &[0u8], b.bytes()).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(b)
    }

    /// Host -> device through the pinned staging buffer in 64 MB pieces (waits).
    pub fn upload<T: Copy>(&self, dst: DevPtr, src: &[T]) -> Result<()> {
        // SAFETY: T is plain data (u16 / u32 / f32 / u8 here)
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src)) };
        let mut st = self.stage.borrow_mut();
        let mut off = 0;
        while off < bytes.len() {
            let n = (bytes.len() - off).min(STAGE);
            st.as_mut_slice()[..n].copy_from_slice(&bytes[off..off + n]);
            unsafe { self.q.copy_raw(dst.byte_add(off).0 as *mut _, st.ptr().0 as *const _, n) }.map_err(|e| anyhow::anyhow!("{e}"))?;
            self.q.sync().map_err(|e| anyhow::anyhow!("{e}"))?;
            off += n;
        }
        Ok(())
    }

    pub fn from_host<T: Copy>(&self, src: &[T]) -> Result<DeviceBuffer<T>> {
        let b = self.alloc::<T>(src.len())?;
        self.upload(b.ptr(), src)?;
        Ok(b)
    }

    pub fn download<T: Copy>(&self, dst: &mut [T], src: DevPtr) -> Result<()> {
        self.q.download(dst, src).map_err(|e| anyhow::anyhow!("{e}"))
    }

    pub fn sync(&self) -> Result<()> {
        self.q.sync().map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// An XQ projection on the device (packed codes + scales; su shared by projections with the same rotation vector).
pub struct XqW {
    pub w: DeviceBuffer<u32>,
    pub su: Option<Arc<DeviceBuffer<u16>>>,
    pub bits: u32,
    pub n: u32,
    pub k: u32,
    pub split: u32,
}

pub struct GdnW {
    pub norm: DeviceBuffer<u16>,
    pub qkv: XqW,
    pub z: XqW,
    pub b: XqW,
    pub a: XqW,
    pub out: XqW,
    pub conv: DeviceBuffer<u16>,
    pub a_log: DeviceBuffer<f32>,
    pub dt_bias: DeviceBuffer<f32>,
    pub gnorm: DeviceBuffer<u16>,
}

pub struct AttnW {
    pub norm: DeviceBuffer<u16>,
    pub q: XqW,
    pub k: XqW,
    pub v: XqW,
    pub o: XqW,
    pub qn: DeviceBuffer<u16>,
    pub kn: DeviceBuffer<u16>,
}

pub struct MlpW {
    pub norm: DeviceBuffer<u16>,
    pub gate: XqW,
    pub up: XqW,
    pub down: XqW,
}

pub enum Mixer {
    Gdn(GdnW),
    Attn(AttnW),
}

pub struct LayerW {
    pub mixer: Mixer,
    pub mlp: MlpW,
}

pub struct Weights {
    pub emb: DeviceBuffer<u16>,
    pub final_norm: DeviceBuffer<u16>,
    pub head: XqW,
    pub layers: Vec<LayerW>,
}

/// Uploads prepared tensors, sharing su vectors with equal contents.
pub struct Uploader<'a> {
    dev: &'a Dev,
    su: HashMap<Vec<u16>, Arc<DeviceBuffer<u16>>>,
}

impl<'a> Uploader<'a> {
    pub fn new(dev: &'a Dev) -> Self {
        Uploader { dev, su: HashMap::new() }
    }
    pub fn xq(&mut self, t: HostTensor) -> Result<XqW> {
        let HostTensor::Xq { packed, su, bits, n, k, split, .. } = t else { bail!("not an XQ tensor") };
        let w = self.dev.from_host(&packed)?;
        let su = match su {
            Some(v) => Some(match self.su.get(&v) {
                Some(b) => b.clone(),
                None => {
                    let b = Arc::new(self.dev.from_host(&v)?);
                    self.su.insert(v, b.clone());
                    b
                }
            }),
            None => None,
        };
        Ok(XqW { w, su, bits, n: n as u32, k: k as u32, split })
    }
    pub fn u16s(&mut self, t: HostTensor) -> Result<DeviceBuffer<u16>> {
        let HostTensor::Bf16(v) = t else { bail!("not a bf16 tensor") };
        self.dev.from_host(&v)
    }
    pub fn f32s(&mut self, t: HostTensor) -> Result<DeviceBuffer<f32>> {
        let HostTensor::F32(v) = t else { bail!("not an f32 tensor") };
        self.dev.from_host(&v)
    }
}

fn take(v: &mut Vec<(&'static str, HostTensor)>, role: &str) -> Result<HostTensor> {
    let i = v.iter().position(|(r, _)| *r == role).with_context(|| format!("no {role}"))?;
    Ok(v.swap_remove(i).1)
}

impl Weights {
    /// Prepares (packs) every tensor on the host and uploads it, layer by layer.
    pub fn load(dev: &Dev, ck: &Checkpoint, cfg: &Config, verbose: bool) -> Result<Weights> {
        let p = Planner::new(ck, cfg);
        let mut ld = Uploader::new(dev);
        let t0 = std::time::Instant::now();
        let mut top = p.top()?;
        let emb = ld.u16s(take(&mut top, "embed")?)?;
        let final_norm = ld.u16s(take(&mut top, "final_norm")?)?;
        let head = ld.xq(take(&mut top, "lm_head")?)?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let mut v = p.layer(i)?;
            let mixer = if cfg.full_attn[i] {
                Mixer::Attn(AttnW {
                    norm: ld.u16s(take(&mut v, "input_norm")?)?,
                    q: ld.xq(take(&mut v, "q")?)?,
                    k: ld.xq(take(&mut v, "k")?)?,
                    v: ld.xq(take(&mut v, "v")?)?,
                    o: ld.xq(take(&mut v, "o")?)?,
                    qn: ld.u16s(take(&mut v, "q_norm")?)?,
                    kn: ld.u16s(take(&mut v, "k_norm")?)?,
                })
            } else {
                Mixer::Gdn(GdnW {
                    norm: ld.u16s(take(&mut v, "input_norm")?)?,
                    qkv: ld.xq(take(&mut v, "qkv")?)?,
                    z: ld.xq(take(&mut v, "z")?)?,
                    b: ld.xq(take(&mut v, "b")?)?,
                    a: ld.xq(take(&mut v, "a")?)?,
                    out: ld.xq(take(&mut v, "out")?)?,
                    conv: ld.u16s(take(&mut v, "conv")?)?,
                    a_log: ld.f32s(take(&mut v, "a_log")?)?,
                    dt_bias: ld.f32s(take(&mut v, "dt_bias")?)?,
                    gnorm: ld.u16s(take(&mut v, "gnorm")?)?,
                })
            };
            let mlp = MlpW {
                norm: ld.u16s(take(&mut v, "post_norm")?)?,
                gate: ld.xq(take(&mut v, "gate")?)?,
                up: ld.xq(take(&mut v, "up")?)?,
                down: ld.xq(take(&mut v, "down")?)?,
            };
            layers.push(LayerW { mixer, mlp });
            if verbose {
                eprint!("\rloaded layer {}/{}, {:.2} GB on the device, {:.0} s", i + 1, cfg.n_layers, dev.bytes.get() as f64 / 1e9, t0.elapsed().as_secs_f64());
            }
        }
        if verbose {
            eprintln!(" ({} distinct su vectors)", ld.su.len());
        }
        Ok(Weights { emb, final_norm, head, layers })
    }
}

/// Per-GDN-layer state, including the window extras (pending state, replay inputs).
pub struct GdnState {
    pub cstate: DeviceBuffer<u16>,
    pub sstate: DeviceBuffer<f32>,
    pub sstate2: DeviceBuffer<f32>,
    pub wqkv: DeviceBuffer<u16>,
    pub wb: DeviceBuffer<u16>,
    pub wa: DeviceBuffer<u16>,
}

pub struct AttnState {
    pub kc: DeviceBuffer<u8>,
    pub vc: DeviceBuffer<u8>,
}

/// A set of attention KV caches not in use (see [`Engine::swap_kv`]): one (K, V) q8 cache per full-attention layer, the MTP head's,
/// and the context they hold.
pub struct KvPool {
    pub ctx: u32,
    pub kv: Vec<(DeviceBuffer<u8>, DeviceBuffer<u8>)>,
    /// the MTP head's caches (None when the engine runs without the MTP head, e.g. DFlash2 drafting)
    pub mtp: Option<(DeviceBuffer<u8>, DeviceBuffer<u8>)>,
}

pub enum LayerState {
    Gdn(GdnState),
    Attn(AttnState),
}

/// Window scratch for up to `rcap` rows (arc-engine Win buffers); the head, decode-attention and argmax buffers for 16 rows.
pub struct Scratch {
    pub tok: DeviceBuffer<u32>,
    pub x: DeviceBuffer<u16>,
    pub xn: DeviceBuffer<u16>,
    pub delta: DeviceBuffer<u16>,
    pub z: DeviceBuffer<u16>,
    pub qn: DeviceBuffer<f32>,
    pub kn: DeviceBuffer<f32>,
    pub y: DeviceBuffer<u16>,
    pub vconv: DeviceBuffer<u16>,
    pub gbeta: DeviceBuffer<f32>,
    pub ggate: DeviceBuffer<f32>,
    pub yg: DeviceBuffer<u16>,
    pub qg: DeviceBuffer<u16>,
    pub kraw: DeviceBuffer<u16>,
    pub ktmp: DeviceBuffer<u16>,
    pub vtmp: DeviceBuffer<u16>,
    pub q: DeviceBuffer<u16>,
    pub att: DeviceBuffer<u16>,
    /// gate, and the SwiGLU output in place (`act`)
    pub gate: DeviceBuffer<u16>,
    pub up: DeviceBuffer<u16>,
    pub logits: DeviceBuffer<f32>,
    pub po: DeviceBuffer<f32>,
    pub pm: DeviceBuffer<f32>,
    pub pl: DeviceBuffer<f32>,
    pub rope: DeviceBuffer<f32>,
    pub am_scratch: DeviceBuffer<u64>,
    pub am_out: DeviceBuffer<i32>,
    /// XQ prepass: rotated rows fp16 [2][K/16][8][16], group sums fp32 [2][K/32][8]
    pub xt: DeviceBuffer<u16>,
    pub xs: DeviceBuffer<f32>,
}

impl Scratch {
    fn new(dev: &Dev, vocab: usize, rcap: u32) -> Result<Scratch> {
        let n = rcap as usize;
        let n16 = MAX_ROWS as usize;
        let h = HIDDEN as usize;
        Ok(Scratch {
            tok: dev.zeros(n)?,
            x: dev.zeros(n * h)?,
            xn: dev.zeros(n * h)?,
            delta: dev.zeros(n * h)?,
            z: dev.zeros(n * GDN_V as usize)?,
            qn: dev.zeros(n * 2048)?,
            kn: dev.zeros(n * 2048)?,
            y: dev.zeros(n * GDN_V as usize)?,
            vconv: dev.zeros(n * GDN_V as usize)?,
            gbeta: dev.zeros(n * GDN_HEADS as usize)?,
            ggate: dev.zeros(n * GDN_HEADS as usize)?,
            yg: dev.zeros(n * GDN_V as usize)?,
            qg: dev.zeros(n * 2 * Q_DIM as usize)?,
            kraw: dev.zeros(n * KV_DIM as usize)?,
            ktmp: dev.zeros(n * KV_DIM as usize)?,
            vtmp: dev.zeros(n * KV_DIM as usize)?,
            q: dev.zeros(n * Q_DIM as usize)?,
            att: dev.zeros(n * Q_DIM as usize)?,
            gate: dev.zeros(n * INTER as usize)?,
            up: dev.zeros(n * INTER as usize)?,
            logits: dev.zeros(n16 * vocab)?,
            po: dev.zeros(n16 * MAX_CHUNKS as usize * HEADS as usize * HEAD_DIM as usize)?,
            pm: dev.zeros(n16 * MAX_CHUNKS as usize * HEADS as usize)?,
            pl: dev.zeros(n16 * MAX_CHUNKS as usize * HEADS as usize)?,
            rope: dev.zeros(n * 64)?,
            am_scratch: dev.zeros(n16 * ARGMAX_PARTS as usize)?,
            am_out: dev.zeros(n16)?,
            xt: dev.zeros(16 * INTER as usize)?,
            xs: dev.zeros(16 * (INTER as usize / 32))?,
        })
    }
}

/// The prompt GEMM's arithmetic (FE_XE_PF): [`Engine::pf_gemm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PfMode {
    F16,
    I8g2,
    I8c2,
    /// diagnostics: per-token int8 activations with the exact grouped weights (i8g2 GEMM)
    I8tw,
    /// diagnostics: per-group int8 activations with per-channel requantized weights (i8g2 GEMM)
    I8gc,
    /// diagnostics: per-(token, slice of FE_XE_PF_SLICE 128-blocks) int8 activations with the exact grouped weights (i8g2 GEMM)
    I8ts,
    /// K-sliced int8: per-(channel, slice) int8 weights, per-(token, slice) int8 activations (slices of FE_XE_PF_SLICE 128-blocks,
    /// default 8), exact int32 a slice, fp32 across slices (xq_pfgemm_i8s: the i8c2 tile, fp32 accumulators in SLM)
    I8s,
    /// K-sliced int8 with a rescaled int32 accumulator (xq_pfgemm_i8r / i8r3, FE_XE_PF_RU = 4 | 3 token blocks a sub-group):
    /// per-(channel, slice) weights and per-(token, slice) activations, the accumulator's units changed at each slice start;
    /// FE_XE_PF_CODES=1 with FE_XE_PF_SLICE=1: the exact XQ codes with their group scales (i8g's numerics)
    I8r,
}

impl PfMode {
    pub fn from_env(var: &str) -> Result<PfMode> {
        Ok(match std::env::var(var).as_deref() {
            // default since the X3 gate (2026-11-09): i8c2, the int8 prompt GEMM (prefill only; decode is unchanged)
            Err(_) | Ok("") | Ok("i8c2") => PfMode::I8c2,
            Ok("f16") | Ok("fp16") => PfMode::F16,
            Ok("i8g2") => PfMode::I8g2,
            Ok("i8tw") => PfMode::I8tw,
            Ok("i8gc") => PfMode::I8gc,
            Ok("i8ts") => PfMode::I8ts,
            Ok("i8s") => PfMode::I8s,
            Ok("i8r") => PfMode::I8r,
            Ok(v) => bail!("{var}={v}: f16 (fp16) | i8g2 | i8c2 | i8s | i8r (diagnostics: i8tw | i8gc | i8ts)"),
        })
    }
}

/// Prompt-window buffers (rcap > 16): the shared GDN replay inputs of a big window (bigx), the XQ prefill GEMM buffers and the
/// prompt-attention scratch (qwen_attn_long.zig Pfs, int8 generation).
pub struct Prompt {
    pub rcap: u32,
    pub wqkv: DeviceBuffer<u16>,
    pub wb: DeviceBuffer<u16>,
    pub wa: DeviceBuffer<u16>,
    /// decoded fp16 weights [N][K] of the projection being multiplied (largest 17408 x 5120)
    pub wq: DeviceBuffer<u16>,
    /// k-major fp16 activations [K][Rp]
    pub act: DeviceBuffer<u16>,
    /// int8-mode activation scales (unused by the f16 GEMM, passed to the prepass)
    pub sa: DeviceBuffer<f32>,
    pub qb: DeviceBuffer<u16>,
    pub kx: DeviceBuffer<u8>,
    pub vx: DeviceBuffer<u8>,
    pub ks: DeviceBuffer<f32>,
    pub qs: DeviceBuffer<f32>,
    pub mu: DeviceBuffer<f32>,
    /// online-softmax state between key ranges (windows past PF_RANGE keys)
    pub st: DeviceBuffer<u8>,
}

impl Prompt {
    fn new(dev: &Dev, rcap: u32) -> Result<Prompt> {
        let n = rcap as usize;
        let rp = (rcap as usize).div_ceil(64) * 64;
        let tiles = (rcap as usize).div_ceil(8);
        let range = PF_RANGE as usize;
        Ok(Prompt {
            rcap,
            wqkv: dev.zeros(n * GDN_QKV as usize)?,
            wb: dev.zeros(n * GDN_HEADS as usize)?,
            wa: dev.zeros(n * GDN_HEADS as usize)?,
            wq: dev.zeros(INTER as usize * HIDDEN as usize)?,
            act: dev.zeros(rp * INTER as usize)?,
            sa: dev.zeros(rp * (INTER as usize / 128))?,
            qb: dev.zeros(tiles * 8 * 24 * 256)?,
            kx: dev.zeros(range * 4 * 256 * 2)?,
            vx: dev.zeros(range * 4 * 256 * 2)?,
            ks: dev.zeros(range * 4)?,
            qs: dev.zeros(tiles * 4 * 3 * 16)?,
            mu: dev.zeros(1024)?,
            st: dev.zeros(tiles * 4 * PF_STATE_BYTES)?,
        })
    }
}

/// DFlash2 feature taps (crates/fe-xdf): after layer `layers[j]` of a window, rows row0.. of the layer's output bf16(x + delta)
/// go to buf[(r - row0) * ld + j * 5120 ..] (kernel `df_tap` of kx_dflash, which the drafter registers with the engine's
/// kernel registry). At most `cap` rows a window.
#[derive(Clone, Debug)]
pub struct Taps {
    pub layers: Vec<usize>,
    pub buf: DevPtr,
    pub ld: u32,
    pub row0: u32,
    pub cap: u32,
}

/// One sequence on the B70: weights, state, scratch, position.
pub struct Engine {
    pub dev: Dev,
    pub kx: Kx,
    pub cfg: Config,
    pub w: Weights,
    pub st: Vec<LayerState>,
    pub sc: Scratch,
    pub pf: Option<Prompt>,
    pub ctx: u32,
    pub pos: u32,
    pub eps: f32,
    inv_freq: [f32; rope::HALF],
    /// a block output in `delta` not yet added to `x` (the next norm folds the add in)
    pending: bool,
    /// what the XQ prepass scratch holds: (x, su, k, m); cleared by every other launch and upload (arc-engine Scratch.key/seq)
    xq_key: Option<(u64, u64, u32, u32)>,
    /// what the prefill GEMM's k-major activation buffer holds: (x, su, k, rows)
    pf_key: Option<(u64, u64, u32, u32)>,
    /// rows / start position / in place of the last window, and its head rows
    pub rows: u32,
    pub pos0: u32,
    pub inplace: bool,
    pub head_rows: u32,
    /// round the head's fp32 logits to bf16 before argmax / readback (arc-engine does; the KL harnesses do not)
    pub bf16_logits: bool,
    /// 1-row windows take the one-token path's kernels (xq_add_rms_prep, xq_prep_swiglu; in place also the one-token GDN core):
    /// same bits, fewer launches.
    /// FE_XE_FUSE1=0 turns it off.
    pub fuse1: bool,
    /// Windows of <= 16 rows run the GDN chain before the delta rule (q / k conv prep, v conv, gates) as one kx-core
    /// gdn_pre_r / gdn_pre_tr launch instead of three (same bits). FE_XE_GDNPRE=0 turns it off; off when the source lacks
    /// the kernel (the reference SPIR-V).
    pub gdnpre: bool,
    /// The gates (beta, g) of a GDN forward window are already computed by the b / a projection's epilogue (the
    /// fused b / a GEMV): gdn_pre then launches without its gates work-group, so they are computed exactly once. Commit
    /// replays (no projection) always compute them.
    pub gates_by_proj: bool,
    /// Tiles a work-group of the window decode attention (attn_dec_pack{,2,4} / attn_dec_tree{,2,4}): DEC_TG=1|2|4 forces
    /// it, else [`dec_tgn`] on the device's Xe-core count; picks cached by (tiles, chunks, steps of the last chunk).
    dec_tg: Option<u32>,
    dec_cores: u32,
    dec_pick: HashMap<(u32, u32, u32), u32>,
    /// DFlash2 feature taps of the next windows (None: off)
    pub taps: Option<Taps>,
    /// the last window was a verify tree (forward_tree): its table; device buffers of the tree windows (created on first use)
    tree: Option<TreeTable>,
    tbuf: Option<TreeBufs>,
    /// 2..16-row windows: the norm + prepass in one launch (xq_add_rms_prep_r) and swiglu + down's prepass (xq_prep_swiglu over m
    /// rows); same bits as add_rmsnorm_r + xq_prep and swiglu + xq_prep. FE_XE_FUSEM=0 turns it off.
    pub fusem: bool,
    /// 9..16-row XQ matvecs (the 8x2 variant) as xq_mv_128_8x2p_1: one sub-group per (K split, row group), bit-identical to
    /// xq_mv_128_8x2_1 with twice the threads, for int5 tensors with K split <= 4. FE_XE_MVP=1 turns it on.
    pub mvp: bool,
    /// X3: the prompt GEMM's arithmetic, FE_XE_PF = i8c2 (default) | f16 (fp16 activations and pre-scaled fp16 weights, bit-exact
    /// with arc-engine: the oracle checks and the lossless gate against arc-engine run with FE_XE_PF=f16) | i8g2 (the exact grouped int8 numerics of arc-engine XQ_PF=i8: int8 codes with their fp32 group scales, int8
    /// Hadamard-rotated activations with one scale a token and 128-group, int32 a group then fp32) | i8c2 (group-free int8:
    /// weights requantized to one int8 scale a channel, activations to one a token, one exact int32 sum over k; 2x the fp16 GEMM)
    pub pf_mode: PfMode,
    /// the prompt GEMM's arithmetic for down_proj (k = INTER, the SwiGLU output as input): FE_XE_PF_DOWN, default FE_XE_PF
    pub pf_down: PfMode,
    /// per-tensor fallback: weight shapes (n, k) whose prompt GEMM stays fp16 whatever the mode (FE_XE_PF_F16="NxK,NxK")
    pub pf_f16: Vec<(u32, u32)>,
}

/// A draft-tree verify window's table (kx-core tree.rs; built as fe-kxcore's TreeTable): u32 [rows][4] = par, depth, ancestor
/// mask, delta-rule state slots (save | restore << 8).
#[derive(Clone, Debug)]
pub struct TreeTable {
    pub words: Vec<u32>,
    pub par: Vec<i32>,
    pub depth: Vec<u32>,
    pub slots: u32,
}

impl TreeTable {
    /// From the parent of every row: par[0] = -1 (the root continues the committed sequence), 0 <= par[t] < t; <= 16 rows.
    pub fn new(par: &[i32]) -> Result<TreeTable> {
        let m = par.len();
        ensure!(m > 0 && m <= 16 && par[0] == -1 && par.iter().enumerate().skip(1).all(|(t, &p)| p >= 0 && (p as usize) < t),
            "tree: rows must be 1..16 in topological order with par[0] = -1: {par:?}");
        let mut depth = vec![0u32; m];
        let mut anc = vec![1u32; m];
        for t in 1..m {
            let p = par[t] as usize;
            depth[t] = depth[p] + 1;
            anc[t] = anc[p] | (1 << t);
        }
        let mut last_child = vec![-1i64; m];
        let mut needs_save = vec![false; m];
        for t in 1..m {
            let p = par[t] as usize;
            if t != p + 1 {
                needs_save[p] = true;
                last_child[p] = last_child[p].max(t as i64);
            }
        }
        let mut slot_of = vec![0xffu32; m];
        let mut free: Vec<u32> = Vec::new();
        let mut next = 0u32;
        let mut words = vec![0u32; 4 * m];
        for t in 0..m {
            let mut restore = 0xffu32;
            if t > 0 && par[t] as usize != t - 1 {
                let p = par[t] as usize;
                restore = slot_of[p];
                ensure!(restore != 0xff, "tree: no saved state for row {p}");
                if last_child[p] == t as i64 {
                    free.push(restore);
                }
            }
            let mut save = 0xffu32;
            if needs_save[t] {
                save = free.pop().unwrap_or_else(|| {
                    next += 1;
                    next - 1
                });
                slot_of[t] = save;
            }
            words[4 * t] = par[t] as u32;
            words[4 * t + 1] = depth[t];
            words[4 * t + 2] = anc[t];
            words[4 * t + 3] = save | (restore << 8);
        }
        Ok(TreeTable { words, par: par.to_vec(), depth, slots: next.max(1) })
    }
}

/// Device buffers of tree windows: the table, delta-rule state slots, per-row K / V tail windows, the commit's path and the
/// gathered replay inputs.
pub struct TreeBufs {
    table: DeviceBuffer<u32>,
    sscr: DeviceBuffer<f32>,
    slots: u32,
    tk: DeviceBuffer<u8>,
    tv: DeviceBuffer<u8>,
    path: DeviceBuffer<u32>,
    gqkv: DeviceBuffer<u16>,
    gb: DeviceBuffer<u16>,
    ga: DeviceBuffer<u16>,
}

/// keys of a tree row's K / V tail window (kx-core tree.rs TAIL_KEYS)
const TREE_TAIL_KEYS: u32 = 64;
/// delta-rule state scratch slots allocated for tree windows (3 MB each)
const TREE_SLOTS: u32 = 8;

fn ptr_overlap(a0: u64, a1: u64, b0: u64, b1: u64) -> bool {
    a0 < b1 && b0 < a1
}

impl Engine {
    /// `rcap`: the largest window (16 = verify windows only; up to 4096 for prompt windows).
    pub fn new(dev: Dev, kx: Kx, cfg: Config, w: Weights, ctx: u32, rcap: u32) -> Result<Engine> {
        ensure!((MAX_ROWS..=4096).contains(&rcap), "rcap {rcap}: 16..4096");
        ensure!(ctx as usize <= crate::state::MAX_CTX, "ctx {ctx} > {}", crate::state::MAX_CTX);
        let rec = KvMode::Q8.rec_bytes();
        let mut st = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            st.push(if cfg.full_attn[i] {
                LayerState::Attn(AttnState { kc: dev.zeros(ctx as usize * 4 * rec)?, vc: dev.zeros(ctx as usize * 4 * rec)? })
            } else {
                let s = (GDN_HEADS * 128 * 128) as usize;
                LayerState::Gdn(GdnState {
                    cstate: dev.zeros(3 * GDN_QKV as usize)?,
                    sstate: dev.zeros(s)?,
                    sstate2: dev.zeros(s)?,
                    wqkv: dev.zeros(MAX_ROWS as usize * GDN_QKV as usize)?,
                    wb: dev.zeros(MAX_ROWS as usize * GDN_HEADS as usize)?,
                    wa: dev.zeros(MAX_ROWS as usize * GDN_HEADS as usize)?,
                })
            });
        }
        let sc = Scratch::new(&dev, cfg.vocab, rcap)?;
        let pf = if rcap > MAX_ROWS { Some(Prompt::new(&dev, rcap)?) } else { None };
        dev.sync()?;
        let inv_freq = rope::inv_freq(cfg.rope_theta);
        let eps = cfg.eps;
        let dec_cores = (dev.ze.info.slices * dev.ze.info.subslices_per_slice).max(1);
        let gdnpre = std::env::var("FE_XE_GDNPRE").map_or(true, |v| v != "0") && kx.provides("rows", "gdn_pre_r") && kx.provides("rows", "gdn_pre_tr");
        Ok(Engine {
            dev, kx, cfg, w, st, sc, pf, ctx, pos: 0, eps, inv_freq, pending: false, xq_key: None, pf_key: None, rows: 0, pos0: 0, inplace: true,
            head_rows: 0, bf16_logits: false, fuse1: std::env::var("FE_XE_FUSE1").map_or(true, |v| v != "0"), gdnpre, gates_by_proj: false,
            taps: None, tree: None, tbuf: None, dec_tg: std::env::var("DEC_TG").ok().and_then(|v| v.parse().ok()).filter(|g| [1, 2, 4].contains(g)),
            dec_cores, dec_pick: HashMap::new(),
            fusem: std::env::var("FE_XE_FUSEM").map_or(true, |v| v != "0"),
            mvp: std::env::var("FE_XE_MVP").map_or(false, |v| v == "1"),
            pf_mode: PfMode::from_env("FE_XE_PF")?,
            pf_down: PfMode::from_env(if std::env::var("FE_XE_PF_DOWN").is_ok() { "FE_XE_PF_DOWN" } else { "FE_XE_PF" })?,
            pf_f16: std::env::var("FE_XE_PF_F16")
                .unwrap_or_default()
                .split(',')
                .filter(|v| !v.is_empty())
                .map(|v| v.split_once('x').and_then(|(a, b)| Some((a.trim().parse().ok()?, b.trim().parse().ok()?))).context("FE_XE_PF_F16: NxK,NxK"))
                .collect::<Result<_>>()?,
        })
    }

    /// Back to an empty context: position 0, GDN conv and delta-rule states zeroed (the KV cache is read only up to pos).
    pub fn reset(&mut self) -> Result<()> {
        for s in &self.st {
            if let LayerState::Gdn(g) = s {
                self.dev.q.fill(g.cstate.ptr(), &[0u8], g.cstate.bytes()).map_err(|e| anyhow::anyhow!("{e}"))?;
                self.dev.q.fill(g.sstate.ptr(), &[0u8], g.sstate.bytes()).map_err(|e| anyhow::anyhow!("{e}"))?;
            }
        }
        self.pos = 0;
        self.pending = false;
        self.xq_key = None;
        self.pf_key = None;
        self.dev.sync()
    }

    /// The largest window this engine runs.
    pub fn rcap(&self) -> u32 {
        self.pf.as_ref().map_or(MAX_ROWS, |p| p.rcap)
    }

    /// A non-XQ launch for code outside the engine (the MTP head): invalidates the prepass scratch.
    pub fn launch(&mut self, tag: &str, name: &str, local: [u32; 3], groups: [u32; 3], args: &[&dyn KernelArg]) -> Result<()> {
        self.run(tag, name, local, groups, args)
    }

    /// XQ projection of m <= 16 rows (the decode kernel; prepass reuse as in the forward).
    pub fn xq_rows(&mut self, t: &XqW, x: DevPtr, m: u32, y: DevPtr, f32out: bool) -> Result<()> {
        self.xq(t, x, m, y, 0, f32out)
    }

    /// XQ projection of any row count (decode kernel / prefill GEMM).
    pub fn proj_rows(&mut self, t: &XqW, x: DevPtr, m: u32, y: DevPtr, f32out: bool) -> Result<()> {
        self.proj(t, x, m, y, f32out)
    }

    /// Forget what the prepass buffers hold (after an upload outside the engine).
    pub fn invalidate(&mut self) {
        self.xq_key = None;
        self.pf_key = None;
    }

    pub fn set_pending(&mut self, p: bool) {
        self.pending = p;
    }

    /// One row of 5120: x = bf16(x + d) when `add`, y = rmsnorm(x) * w, and the prepass of y for projection t (xq_add_rms_prep: the
    /// one-token path's fused norm; the next xq_rows of t's input group on y reuses it).
    pub fn add_rms_prep(&mut self, t: &XqW, x: DevPtr, d: DevPtr, w: DevPtr, y: DevPtr, add: bool) -> Result<()> {
        ensure!(t.k == HIDDEN, "add_rms_prep: k {}", t.k);
        let (xt, xs, eps) = (self.sc.xt.ptr(), self.sc.xs.ptr(), self.eps);
        let su = t.su.as_ref().map(|b| b.ptr()).unwrap_or(xs);
        let rot = u32::from(t.su.is_some());
        self.run("xq4", "xq_add_rms_prep", [512, 1, 1], [1, 1, 1], &[&x, &d, &w, &y, &eps, &u32::from(add), &su, &rot, &xt, &xs])?;
        let sk = t.su.as_ref().map(|b| b.ptr().0).unwrap_or(0);
        self.xq_key = Some((y.0, sk, t.k, 1));
        Ok(())
    }

    /// m (2..16) rows of 5120: [`Engine::add_rms_prep`] for a verify / decode window (xq_add_rms_prep_r, one work-group a row, the
    /// prepass in the m-row variant's layout).
    pub fn add_rms_prep_r(&mut self, t: &XqW, x: DevPtr, d: DevPtr, w: DevPtr, y: DevPtr, add: bool, m: u32) -> Result<()> {
        ensure!(t.k == HIDDEN && (2..=MAX_ROWS).contains(&m), "add_rms_prep_r: k {} m {m}", t.k);
        let (mw, nr) = VARIANTS[variant_for(m)?];
        let (xt, xs, eps) = (self.sc.xt.ptr(), self.sc.xs.ptr(), self.eps);
        let su = t.su.as_ref().map(|b| b.ptr()).unwrap_or(xs);
        let rot = u32::from(t.su.is_some());
        self.run("xq4", "xq_add_rms_prep_r", [512, 1, 1], [mw * nr, 1, 1], &[&x, &d, &w, &y, &eps, &u32::from(add), &su, &rot, &xt, &xs, &m, &mw])?;
        let sk = t.su.as_ref().map(|b| b.ptr().0).unwrap_or(0);
        self.xq_key = Some((y.0, sk, t.k, m));
        Ok(())
    }

    /// GDN in_proj_b + in_proj_a of m (2..16) rows of x and gdn_gates_r in one launch (xq_mv_ba_{MW}x{NR}): wb / wa (bf16, the
    /// replay inputs) and the gates beta / g, same bits as the two XQ matvecs + gdn_gates_r. b and a share the prepass.
    #[allow(clippy::too_many_arguments)]
    fn xq_ba(&mut self, b: &XqW, a: &XqW, x: DevPtr, m: u32, wb: DevPtr, wa: DevPtr, a_log: DevPtr, dt_bias: DevPtr) -> Result<()> {
        let (mw, nr) = VARIANTS[variant_for(m)?];
        let su = b.su.as_ref().map(|v| v.ptr()).unwrap_or(DevPtr::NULL);
        let key = (x.0, su.0, b.k, m);
        let (xt, xs) = (self.sc.xt.ptr(), self.sc.xs.ptr());
        if self.xq_key != Some(key) {
            let rot = u32::from(b.su.is_some());
            let su_arg = if b.su.is_some() { su } else { xs };
            self.kx.run(&self.dev.q, "xq4", "xq_prep", [16, 1, 1], [b.k / 128, mw * nr, 1], &[&x, &1u32, &b.k, &su_arg, &rot, &xt, &xs, &b.k, &m, &mw])?;
        }
        self.pf_key = None;
        let sp = b.split;
        let red = (2 * sp * mw * nr * 16 * 4) as usize;
        let (gbeta, ggate) = (self.sc.gbeta.ptr(), self.sc.ggate.ptr());
        self.kx.run(&self.dev.q, &format!("xq{}", b.bits), &format!("xq_mv_ba_{mw}x{nr}"), [2 * sp * 16, 1, 1], [b.n / 16, 1, 1],
            &[&xt, &xs, &b.w.ptr(), &a.w.ptr(), &wb, &wa, &b.k, &m, &Local(red), &a_log, &dt_bias, &gbeta, &ggate])?;
        self.xq_key = Some(key);
        Ok(())
    }

    /// act = bf16(silu(g) * u) for m (2..16) rows and the prepass of act for projection t (xq_prep_swiglu over m rows).
    pub fn prep_swiglu_r(&mut self, t: &XqW, g: DevPtr, u: DevPtr, act: DevPtr, m: u32) -> Result<()> {
        ensure!((2..=MAX_ROWS).contains(&m), "prep_swiglu_r: m {m}");
        let (mw, nr) = VARIANTS[variant_for(m)?];
        let (xt, xs) = (self.sc.xt.ptr(), self.sc.xs.ptr());
        let su = t.su.as_ref().map(|b| b.ptr()).unwrap_or(xs);
        let rot = u32::from(t.su.is_some());
        self.run("xq4", "xq_prep_swiglu", [16, 1, 1], [t.k / 128, mw * nr, 1], &[&g, &u, &act, &su, &rot, &xt, &xs, &t.k, &m, &mw])?;
        let sk = t.su.as_ref().map(|b| b.ptr().0).unwrap_or(0);
        self.xq_key = Some((act.0, sk, t.k, m));
        Ok(())
    }

    /// act = bf16(silu(g) * u) for one row and the prepass of act for projection t (xq_prep_swiglu).
    pub fn prep_swiglu(&mut self, t: &XqW, g: DevPtr, u: DevPtr, act: DevPtr) -> Result<()> {
        let (xt, xs) = (self.sc.xt.ptr(), self.sc.xs.ptr());
        let su = t.su.as_ref().map(|b| b.ptr()).unwrap_or(xs);
        let rot = u32::from(t.su.is_some());
        self.run("xq4", "xq_prep_swiglu", [16, 1, 1], [t.k / 128, 1, 1], &[&g, &u, &act, &su, &rot, &xt, &xs, &t.k, &1u32, &1u32])?;
        let sk = t.su.as_ref().map(|b| b.ptr().0).unwrap_or(0);
        self.xq_key = Some((act.0, sk, t.k, 1));
        Ok(())
    }

    /// A non-XQ launch (invalidates the XQ prepass scratch).
    fn run(&mut self, tag: &str, name: &str, local: [u32; 3], groups: [u32; 3], args: &[&dyn KernelArg]) -> Result<()> {
        self.xq_key = None;
        self.pf_key = None;
        self.kx.run(&self.dev.q, tag, name, local, groups, args)
    }

    /// XQ projection of m <= 16 rows of x (bf16, row stride t.k) into y (row stride t.n, at y_off elements), bf16 or fp32
    /// (`Layer.forwardRows`: the prepass is skipped when the scratch holds this x / su / k / m and only XQ matvecs ran since).
    fn xq(&mut self, t: &XqW, x: DevPtr, m: u32, y: DevPtr, y_off: u32, f32out: bool) -> Result<()> {
        ensure!((1..=MAX_ROWS).contains(&m), "xq rows {m}");
        let (mw, nr) = VARIANTS[variant_for(m)?];
        let su = t.su.as_ref().map(|b| b.ptr()).unwrap_or(DevPtr::NULL);
        let key = (x.0, su.0, t.k, m);
        let (xt, xs) = (self.sc.xt.ptr(), self.sc.xs.ptr());
        if self.xq_key != Some(key) {
            let rot = u32::from(t.su.is_some());
            let su_arg = if t.su.is_some() { su } else { xs };
            self.kx.run(&self.dev.q, "xq4", "xq_prep", [16, 1, 1], [t.k / 128, mw * nr, 1],
                &[&x, &1u32, &t.k, &su_arg, &rot, &xt, &xs, &t.k, &m, &mw])?;
        }
        self.pf_key = None;
        let s = t.split;
        // where it pays (B70, xq-check m16, split sweep 1..8): K split <= 4 and int5, or down (k = INTER) at any width; int4
        // gate / up / qkv and the high-split shapes lose (more decode than latency to hide)
        let par = self.mvp && nr == 2 && s <= 4 && (t.bits == 5 || t.k == INTER);
        let red = if s > 1 || par { (s * mw * nr * 16 * 4) as usize } else { 4 };
        let name = if par { "xq_mv_128_8x2p_1".to_string() } else { format!("xq_mv_128_{mw}x{nr}_1") };
        let tag = format!("xq{}", t.bits);
        self.kx.run(&self.dev.q, &tag, &name, [if par { 2 * s * 16 } else { s * 16 }, 1, 1], [t.n / 16, 1, 1],
            &[&xt, &xs, &t.w.ptr(), &y, &t.k, &t.n, &m, &y_off, &t.n, &u32::from(f32out), &Local(red)])?;
        // a matvec that writes over its own input invalidates the prepass for the next call
        let esz = if f32out { 4u64 } else { 2 };
        let y0 = y.0 + y_off as u64 * esz;
        let y1 = y0 + ((m as u64 - 1) * t.n as u64 + t.n as u64) * esz;
        self.xq_key = if ptr_overlap(y0, y1, x.0, x.0 + m as u64 * t.k as u64 * 2) { None } else { Some(key) };
        Ok(())
    }

    /// m rows of any count: the decode kernel up to 16 rows, the prefill GEMM from 32 rows (Ops.prefillRows), 16-row passes between.
    fn proj(&mut self, t: &XqW, x: DevPtr, m: u32, y: DevPtr, f32out: bool) -> Result<()> {
        if m <= MAX_ROWS {
            return self.xq(t, x, m, y, 0, f32out);
        }
        if m >= 32 && self.pf.is_some() && t.n % 16 == 0 {
            return self.pf_gemm(t, x, m, y, f32out);
        }
        let esz = if f32out { 4 } else { 2 };
        let mut d = 0;
        while d < m {
            let n = MAX_ROWS.min(m - d);
            self.xq(t, x.byte_add((d * t.k * 2) as usize), n, y.byte_add((d * t.n * esz) as usize), 0, f32out)?;
            d += n;
        }
        Ok(())
    }

    /// The XQ prompt GEMM, XQ_PF=f16 (xq.zig Prefill.run): k-major fp16 prepass (skipped when the buffer holds this x / su / k / R
    /// and nothing else ran since), the layer's weights decoded to pre-scaled fp16, then the 4 x 4 work-group-tiled DPAS GEMM.
    /// [`PfMode::I8g2`]: the same steps in int8 (prepass with int8 output and per (token, 128-group) scales in `sa`; weights decoded
    /// to int8 codes in `wq` with their fp32 group scales after them; the 64-k-step grouped int8 GEMM).
    fn pf_gemm(&mut self, t: &XqW, x: DevPtr, r: u32, y: DevPtr, f32out: bool) -> Result<()> {
        let pf = self.pf.as_ref().unwrap();
        ensure!(r <= pf.rcap && (t.k / 128) % 8 == 0, "prefill GEMM: {r} rows, k {}", t.k);
        let (wq, act, sa) = (pf.wq.ptr(), pf.act.ptr(), pf.sa.ptr());
        let rp = r.div_ceil(64) * 64;
        let su = t.su.as_ref().map(|b| b.ptr()).unwrap_or(DevPtr::NULL);
        let mode = if self.pf_f16.contains(&(t.n, t.k)) { PfMode::F16 } else if t.k == INTER { self.pf_down } else { self.pf_mode };
        // the prepass output depends on the mode: part of the reuse key
        let key = (x.0, su.0, t.k, r | (mode as u32) << 24);
        let i8 = mode != PfMode::F16;
        // i8c2 operands in wq past the int8 weights [N][K] (N K <= INTER HIDDEN, the fp16 decode's half): X8 [K][Rp], then sn / invn
        // [N] and st / tinv [Rp] fp32
        let rcap_p = pf.rcap.div_ceil(64) as usize * 64;
        let (x8_off, s_off) = (INTER as usize * HIDDEN as usize, INTER as usize * HIDDEN as usize + INTER as usize * rcap_p);
        let (x8, sn, invn) = (wq.byte_add(x8_off), wq.byte_add(s_off), wq.byte_add(s_off + 4 * INTER as usize));
        let (st, tinv) = (wq.byte_add(s_off + 8 * INTER as usize), wq.byte_add(s_off + 8 * INTER as usize + 4 * rcap_p));
        // diagnostics' fp32 group scales [K/128][N] (<= INTER HIDDEN / 32 bytes), past everything else
        let sw_d = wq.byte_add(s_off + 8 * INTER as usize + 8 * rcap_p);
        // i8ts: per-(token, block) scales and multipliers [Rp][K/128] after the diagnostics' group scales
        let ts_off = s_off + 8 * INTER as usize + 8 * rcap_p + INTER as usize * HIDDEN as usize / 32;
        let (sa_s, tinv_s) = (wq.byte_add(ts_off), wq.byte_add(ts_off + rcap_p * (INTER as usize / 128) * 4));
        // i8s: per-(channel, block) multipliers [K/128][N] after those (its scales use sw_d)
        let invn_s = wq.byte_add(ts_off + 8 * rcap_p * (INTER as usize / 128));
        let sb: u32 = std::env::var("FE_XE_PF_SLICE").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        // i8r: [nsl][N] sn / invn / rn and [Rp][nsl] st / tinv / rt, over the diagnostics' region (from sw_d)
        let r0 = s_off + 8 * INTER as usize + 8 * rcap_p;
        let (szw, szt) = (4 * INTER as usize * HIDDEN as usize / 128, 4 * rcap_p * (INTER as usize / 128));
        let (sn_r, invn_r, rn_r) = (wq.byte_add(r0), wq.byte_add(r0 + szw), wq.byte_add(r0 + 2 * szw));
        let (st_r, tinv_r, rt_r) = (wq.byte_add(r0 + 3 * szw), wq.byte_add(r0 + 3 * szw + szt), wq.byte_add(r0 + 3 * szw + 2 * szt));
        if self.pf_key != Some(key) {
            let rot = u32::from(t.su.is_some());
            // the int8 prepass writes the scales into sa: a layer without rotation passes a never-read su; the dummy must not be sa
            let su_arg = if t.su.is_some() { su } else if i8 { wq } else { sa };
            if mode == PfMode::I8r {
                ensure!(r0 + 3 * szw + 3 * szt <= pf.wq.bytes() && t.n * t.k <= INTER * HIDDEN, "i8r operands do not fit wq");
                let nsl = (t.k / 128).div_ceil(sb);
                self.kx.run(&self.dev.q, "xqpfd4", "xq_pfprep_k16m", [512, 1, 1], [t.k / 128, rp / 32, 1],
                    &[&x, &1u32, &t.k, &su_arg, &rot, &act, &sa, &t.k, &r, &rp])?;
                self.kx.run(&self.dev.q, "xqpfd4", "xq_pftok_r", [16, 1, 1], [rp / 16, 1, 1], &[&sa, &st_r, &tinv_r, &rt_r, &(t.k / 128), &sb])?;
                self.kx.run(&self.dev.q, "xqpfd4", "xq_pfq8r", [16, 1, 1], [rp.div_ceil(128), t.k / 32, 1], &[&act, &tinv_r, &x8, &rp, &nsl, &sb])?;
            } else if matches!(mode, PfMode::I8ts | PfMode::I8s) {
                ensure!(ts_off + 8 * rcap_p * (INTER as usize / 128) + INTER as usize * HIDDEN as usize / 32 <= pf.wq.bytes(), "i8ts / i8s operands do not fit wq");
                self.kx.run(&self.dev.q, "xqpfd4", "xq_pfprep_k16m", [512, 1, 1], [t.k / 128, rp / 32, 1],
                    &[&x, &1u32, &t.k, &su_arg, &rot, &act, &sa, &t.k, &r, &rp])?;
                self.kx.run(&self.dev.q, "xqpfd4", "xq_pftok_s", [16, 1, 1], [rp / 16, 1, 1], &[&sa, &sa_s, &tinv_s, &(t.k / 128), &sb])?;
                self.kx.run(&self.dev.q, "xqpfd4", "xq_pfq8s", [16, 1, 1], [rp.div_ceil(128), t.k / 32, 1], &[&act, &tinv_s, &x8, &rp, &(t.k / 128)])?;
            } else if matches!(mode, PfMode::I8c2 | PfMode::I8tw) {
                ensure!(s_off + 8 * INTER as usize + 8 * rcap_p + INTER as usize * HIDDEN as usize / 32 <= pf.wq.bytes() && t.n * t.k <= INTER * HIDDEN, "i8c2 operands do not fit wq");
                self.kx.run(&self.dev.q, "xqpfd4", "xq_pfprep_k16m", [512, 1, 1], [t.k / 128, rp / 32, 1],
                    &[&x, &1u32, &t.k, &su_arg, &rot, &act, &sa, &t.k, &r, &rp])?;
                self.kx.run(&self.dev.q, "xqpfd4", "xq_pftok", [16, 1, 1], [rp / 16, 1, 1], &[&sa, &st, &tinv, &(t.k / 128)])?;
                self.kx.run(&self.dev.q, "xqpfd4", "xq_pfq8t", [16, 1, 1], [rp.div_ceil(128), t.k / 32, 1], &[&act, &tinv, &x8, &rp])?;
                if mode == PfMode::I8tw {
                    self.kx.run(&self.dev.q, "xqpfd4", "xq_pf_fill_tok", [16, 1, 1], [rp / 16, 1, 1], &[&sa, &st, &(t.k / 128)])?;
                }
            } else {
                self.kx.run(&self.dev.q, "xqpfd4", "xq_pfprep_k16", [512, 1, 1], [t.k / 128, rp / 32, 1],
                    &[&x, &1u32, &t.k, &su_arg, &rot, &act, &sa, &t.k, &r, &rp, &u32::from(i8)])?;
            }
        }
        self.xq_key = None;
        // Prompt GEMM tiling by window rows: 256-row x 128-channel work-groups up to 3583 rows, 256 x 256 from 3584 (the
        // cliff at 3585..3711, where the last row tile is 16..128 rows: f16 down 2.1x, i8c2 down 1.8x, gate_up 1.5x
        // slower; the 256-channel tile halves the work-groups that re-read the weight tile and lifts pfbench 3600 rows
        // from 1027 to 1249 tok/s on f16 and from 1576 to ~1900 on i8c2, bit-identical outputs).
        let wide = r >= 3584;
        match mode {
            PfMode::F16 => {
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfdec16x_128", [16, 8, 1], [t.n / 16, t.k / 128 / 8, 1], &[&t.w.ptr(), &wq, &t.k, &t.n])?;
                let (kname, local, ncols) = if wide { ("xq_pfgemm_f16_c8t4", 512, 256) } else { ("xq_pfgemm_f16_c4t4", 256, 128) };
                self.kx.run(&self.dev.q, "xq_pfg", kname, [local, 1, 1], [r.div_ceil(256), t.n.div_ceil(ncols), 1],
                    &[&wq, &act, &y, &r, &rp, &t.k, &t.n, &t.n, &0u32, &u32::from(f32out)])?;
            }
            PfMode::I8tw => {
                let sw = sw_d;
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfdec_128", [16, 1, 1], [t.n / 16, t.k / 128, 1],
                    &[&t.w.ptr(), &wq, &sw, &t.k, &t.n, &0u32])?;
                self.kx.run(&self.dev.q, "xq_pfg", "xq_pfgemm_i8g2_c4t4", [256, 1, 1], [r.div_ceil(128), t.n.div_ceil(128), 1],
                    &[&wq, &sw, &x8, &sa, &y, &r, &rp, &t.k, &t.n, &t.n, &0u32, &u32::from(f32out)])?;
            }
            PfMode::I8gc => {
                let sw = sw_d;
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfscale_c", [16, 1, 1], [t.n / 16, 1, 1], &[&t.w.ptr(), &sn, &invn, &t.k])?;
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfdec_c128", [16, 1, 1], [t.n / 16, t.k / 128, 1], &[&t.w.ptr(), &wq, &invn, &t.k])?;
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pf_fill_chan", [16, 1, 1], [t.n / 16, t.k / 128, 1], &[&sw, &sn, &t.n])?;
                self.kx.run(&self.dev.q, "xq_pfg", "xq_pfgemm_i8g2_c4t4", [256, 1, 1], [r.div_ceil(128), t.n.div_ceil(128), 1],
                    &[&wq, &sw, &act, &sa, &y, &r, &rp, &t.k, &t.n, &t.n, &0u32, &u32::from(f32out)])?;
            }
            PfMode::I8ts => {
                let sw = sw_d;
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfdec_128", [16, 1, 1], [t.n / 16, t.k / 128, 1],
                    &[&t.w.ptr(), &wq, &sw, &t.k, &t.n, &0u32])?;
                self.kx.run(&self.dev.q, "xq_pfg", "xq_pfgemm_i8g2_c4t4", [256, 1, 1], [r.div_ceil(128), t.n.div_ceil(128), 1],
                    &[&wq, &sw, &x8, &sa_s, &y, &r, &rp, &t.k, &t.n, &t.n, &0u32, &u32::from(f32out)])?;
            }
            PfMode::I8s => {
                let nsl = (t.k / 128).div_ceil(sb);
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfscale_s", [16, 1, 1], [t.n / 16, nsl, 1], &[&t.w.ptr(), &sw_d, &invn_s, &t.k, &t.n, &sb])?;
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfdec16x_s128", [16, 8, 1], [t.n / 16, t.k / 128 / 8, 1], &[&t.w.ptr(), &wq, &invn_s, &t.k, &t.n])?;
                self.kx.run(&self.dev.q, "xq_pfg", "xq_pfgemm_i8s_c4t4", [256, 1, 1], [r.div_ceil(256), t.n.div_ceil(128), 1],
                    &[&wq, &sw_d, &x8, &sa_s, &y, &r, &rp, &t.k, &t.n, &t.n, &0u32, &u32::from(f32out), &sb])?;
            }
            PfMode::I8r => {
                let codes: u32 = std::env::var("FE_XE_PF_CODES").map_or(0, |v| u32::from(v == "1"));
                let ru: u32 = std::env::var("FE_XE_PF_RU").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfscale_r", [16, 1, 1], [t.n / 16, 1, 1],
                    &[&t.w.ptr(), &sn_r, &invn_r, &rn_r, &t.k, &t.n, &sb, &codes])?;
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfdec16x_r128", [16, 8, 1], [t.n / 16, t.k / 128 / 8, 1],
                    &[&t.w.ptr(), &wq, &invn_r, &t.k, &t.n, &sb])?;
                let kname = if ru == 3 { "xq_pfgemm_i8r3_c4t4" } else { "xq_pfgemm_i8r_c4t4" };
                self.kx.run(&self.dev.q, "xq_pfg", kname, [256, 1, 1], [r.div_ceil(64 * ru), t.n.div_ceil(128), 1],
                    &[&wq, &sn_r, &rn_r, &x8, &st_r, &rt_r, &y, &r, &rp, &t.k, &t.n, &t.n, &0u32, &u32::from(f32out), &sb])?;
            }
            PfMode::I8g2 => {
                let sw = wq.byte_add(t.n as usize * t.k as usize); // [K/128][N] fp32 after the [N][K] int8 codes
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfdec_128", [16, 1, 1], [t.n / 16, t.k / 128, 1],
                    &[&t.w.ptr(), &wq, &sw, &t.k, &t.n, &0u32])?;
                self.kx.run(&self.dev.q, "xq_pfg", "xq_pfgemm_i8g2_c4t4", [256, 1, 1], [r.div_ceil(128), t.n.div_ceil(128), 1],
                    &[&wq, &sw, &act, &sa, &y, &r, &rp, &t.k, &t.n, &t.n, &0u32, &u32::from(f32out)])?;
            }
            PfMode::I8c2 => {
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfscale_c", [16, 1, 1], [t.n / 16, 1, 1], &[&t.w.ptr(), &sn, &invn, &t.k])?;
                self.kx.run(&self.dev.q, &format!("xqpfd{}", t.bits), "xq_pfdec16x_c128", [16, 8, 1], [t.n / 16, t.k / 128 / 8, 1], &[&t.w.ptr(), &wq, &invn, &t.k, &t.n])?;
                let (kname, local, ncols) = if wide { ("xq_pfgemm_i8c2_c8t4", 512, 256) } else { ("xq_pfgemm_i8c2_c4t4", 256, 128) };
                self.kx.run(&self.dev.q, "xq_pfg", kname, [local, 1, 1], [r.div_ceil(256), t.n.div_ceil(ncols), 1],
                    &[&wq, &sn, &x8, &st, &y, &r, &rp, &t.k, &t.n, &t.n, &0u32, &u32::from(f32out)])?;
            }
        }
        let esz = if f32out { 4u64 } else { 2 };
        let (y0, y1) = (y.0, y.0 + r as u64 * t.n as u64 * esz);
        self.pf_key = if ptr_overlap(y0, y1, x.0, x.0 + r as u64 * t.k as u64 * 2) { None } else { Some(key) };
        Ok(())
    }

    /// norm_in, and for one row with `fuse1` the one-token path's xq_add_rms_prep: the norm plus the prepass of `first` (the group's
    /// first projection; the next xq of the group reuses it).
    fn norm_in1(&mut self, w: DevPtr, m: u32, first: &XqW) -> Result<()> {
        if first.k == HIDDEN && self.fusem && (2..=MAX_ROWS).contains(&m) {
            let (x, xn, d) = (self.sc.x.ptr(), self.sc.xn.ptr(), self.sc.delta.ptr());
            let add = self.pending;
            self.pending = false;
            return self.add_rms_prep_r(first, x, d, w, xn, add, m);
        }
        if m != 1 || !self.fuse1 || first.k != HIDDEN {
            return self.norm_in(w, m);
        }
        let (x, xn, d) = (self.sc.x.ptr(), self.sc.xn.ptr(), self.sc.delta.ptr());
        let add = self.pending;
        self.pending = false;
        self.add_rms_prep(first, x, d, w, xn, add)
    }

    fn norm_in(&mut self, w: DevPtr, m: u32) -> Result<()> {
        let (x, xn, d, eps) = (self.sc.x.ptr(), self.sc.xn.ptr(), self.sc.delta.ptr(), self.eps);
        if !self.pending {
            return self.run("rows", "rmsnorm_r", [512, 1, 1], [m, 1, 1], &[&x, &w, &xn, &HIDDEN, &eps]);
        }
        self.pending = false;
        self.run("rows", "add_rmsnorm_r", [512, 1, 1], [m, 1, 1], &[&x, &d, &w, &xn, &HIDDEN, &eps])
    }

    fn upload_rows(&mut self, tokens: &[u32], pos0: u32) -> Result<()> {
        let m = tokens.len();
        self.dev.q.upload(self.sc.tok.ptr(), tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut tab = vec![0f32; m * 64];
        for r in 0..m {
            rope::row(&self.inv_freq, pos0 + r as u32, &mut tab[r * 64..(r + 1) * 64]);
        }
        self.dev.q.upload(self.sc.rope.ptr(), &tab).map_err(|e| anyhow::anyhow!("{e}"))?;
        self.xq_key = None;
        self.pf_key = None;
        Ok(())
    }

    fn gdn(&mut self, li: usize, m: u32, inplace: bool) -> Result<()> {
        let Mixer::Gdn(g) = &self.w.layers[li].mixer else { unreachable!() };
        let LayerState::Gdn(gs) = &self.st[li] else { unreachable!() };
        // (raw pointers: the weights / state outlive the calls; `self` is borrowed mutably by the launch helpers)
        let g: *const GdnW = g;
        let gs: *const GdnState = gs;
        let (g, gs) = unsafe { (&*g, &*gs) };
        // replay inputs: per layer for windows <= 16 rows, the shared big-window buffers above
        let (wqkv, wb, wa) = if m > MAX_ROWS {
            let pf = self.pf.as_ref().context("window > 16 rows without prompt buffers")?;
            (pf.wqkv.ptr(), pf.wb.ptr(), pf.wa.ptr())
        } else {
            (gs.wqkv.ptr(), gs.wb.ptr(), gs.wa.ptr())
        };
        let s = &self.sc;
        let (xn, z, y, yg, delta) = (s.xn.ptr(), s.z.ptr(), s.y.ptr(), s.yg.ptr(), s.delta.ptr());
        self.norm_in1(g.norm.ptr(), m, &g.qkv)?;
        self.proj(&g.qkv, xn, m, wqkv, false)?;
        self.proj(&g.z, xn, m, z, false)?;
        let same_su = match (&g.b.su, &g.a.su) {
            (Some(p), Some(q)) => p.ptr().0 == q.ptr().0,
            (None, None) => true,
            _ => false,
        };
        // b / a + gates in one launch on the row path (the one-token GDN core reads wb / wa itself)
        let fuse_ba = self.fusem && (2..=MAX_ROWS).contains(&m) && g.b.n == 48 && g.a.n == 48 && g.b.bits == g.a.bits && g.b.k == g.a.k
            && g.b.split == g.a.split && same_su && 2 * g.b.split * 16 <= 1024;
        if fuse_ba {
            self.xq_ba(&g.b, &g.a, xn, m, wb, wa, g.a_log.ptr(), g.dt_bias.ptr())?;
        } else {
            self.proj(&g.b, xn, m, wb, false)?;
            self.proj(&g.a, xn, m, wa, false)?;
        }
        let eps = self.eps;
        if m == 1 && inplace && self.fuse1 {
            // the one-token path (Ops.gdn): conv + q/k prep shifting the q/k conv state, the delta rule with the v conv by lane 0
            // (state and v conv state in place), the gated norm; the row kernels' bits at m = 1
            let (qn, kn, conv) = (self.sc.qn.ptr(), self.sc.kn.ptr(), g.conv.ptr());
            let (cst, sst) = (gs.cstate.ptr(), gs.sstate.ptr());
            self.run("gdn", "gdn_conv_prep", [16, 1, 1], [16, 1, 1], &[&wqkv, &cst, &conv, &qn, &kn, &GDN_QKV])?;
            self.run("gdn", "gdn_step", [64, 1, 1], [GDN_HEADS, 32, 1],
                &[&qn, &kn, &wqkv, &cst, &conv, &wb, &wa, &g.a_log.ptr(), &g.dt_bias.ptr(), &sst, &y, &0u32, &GDN_QKV])?;
            self.run("gdn", "gdn_gate_norm", [16, 1, 1], [GDN_HEADS, 1, 1], &[&y, &z, &g.gnorm.ptr(), &yg, &eps])?;
        } else {
            let s_out = if inplace { gs.sstate.ptr() } else { gs.sstate2.ptr() };
            self.gdn_core(g, gs.cstate.ptr(), gs.sstate.ptr(), (wqkv, wb, wa), m, inplace, s_out, fuse_ba || self.gates_by_proj)?;
            self.run("rows", "gdn_gate_norm_r", [16, 1, 1], [GDN_HEADS, m, 1], &[&y, &z, &g.gnorm.ptr(), &yg, &eps])?;
        }
        self.proj(&g.out, yg, m, delta, false)?;
        self.pending = true;
        Ok(())
    }

    /// conv + q/k prep and the delta rule over m rows from the state s_in into s_out (then the conv-state shift when in place), on the
    /// window's saved projection outputs (wqkv, wb, wa).
    #[allow(clippy::too_many_arguments)]
    fn gdn_core(&mut self, g: &GdnW, cst: DevPtr, s_in: DevPtr, (wqkv, wb, wa): (DevPtr, DevPtr, DevPtr), m: u32, inplace: bool, s_out: DevPtr, gates_done: bool) -> Result<()> {
        let s = &self.sc;
        let (qn, kn, vconv, gbeta, ggate, y) = (s.qn.ptr(), s.kn.ptr(), s.vconv.ptr(), s.gbeta.ptr(), s.ggate.ptr(), s.y.ptr());
        let conv = g.conv.ptr();
        if self.gdnpre && m <= MAX_ROWS {
            // q / k prep + v conv (+ gates) in one launch (kx-core gdnpre.rs; prompt windows keep the chain: faster there)
            let gx = if gates_done { 100 } else { 101 };
            self.run("rows", "gdn_pre_r", [64, 1, 1], [gx, m, 1],
                &[&wqkv, &cst, &conv, &qn, &kn, &vconv, &GDN_QKV, &wb, &wa, &g.a_log.ptr(), &g.dt_bias.ptr(), &gbeta, &ggate])?;
        } else {
            self.run("rows", "gdn_conv_prep_r", [16, 1, 1], [16, m, 1], &[&wqkv, &cst, &conv, &qn, &kn, &GDN_QKV])?;
            self.run("rows", "gdn_vconv_r", [64, 1, 1], [m, 96, 1], &[&wqkv, &cst, &conv, &vconv, &GDN_QKV])?;
            if !gates_done {
                self.run("rows", "gdn_gates_r", [48, 1, 1], [m, 1, 1], &[&wb, &wa, &g.a_log.ptr(), &g.dt_bias.ptr(), &gbeta, &ggate])?;
            }
        }
        self.run("rows", "gdn_step_r2x4", [64, 1, 1], [GDN_HEADS, 32 / 4, 1], &[&qn, &kn, &vconv, &gbeta, &ggate, &s_in, &s_out, &y, &0u32, &m])?;
        if inplace {
            self.run("rows", "gdn_conv_commit", [64, 1, 1], [GDN_QKV.div_ceil(64), 1, 1], &[&wqkv, &cst, &GDN_QKV, &m, &0u32, &GDN_QKV])?;
        }
        Ok(())
    }

    /// Tiles a work-group for a window decode attention over `keys` keys (the longest row): DEC_TG, else [`dec_tgn`] (cached).
    fn pick_tg(&mut self, tiles: u32, keys: u32) -> u32 {
        if let Some(g) = self.dec_tg {
            return g;
        }
        let nch = keys.div_ceil(KV_CHUNK);
        let key = (tiles, nch, dec_last_steps(keys));
        let cores = self.dec_cores;
        *self.dec_pick.entry(key).or_insert_with(|| dec_tgn(tiles, keys, cores))
    }

    fn attention(&mut self, li: usize, m: u32, pos0: u32) -> Result<()> {
        ensure!(pos0 + m <= self.ctx, "context full ({} + {m} > {})", pos0, self.ctx);
        let Mixer::Attn(a) = &self.w.layers[li].mixer else { unreachable!() };
        let LayerState::Attn(st) = &self.st[li] else { unreachable!() };
        let a: *const AttnW = a;
        let st: *const AttnState = st;
        let (a, st) = unsafe { (&*a, &*st) };
        let s = &self.sc;
        let (xn, qg, kraw, vtmp, ktmp, rope, q, att, delta) =
            (s.xn.ptr(), s.qg.ptr(), s.kraw.ptr(), s.vtmp.ptr(), s.ktmp.ptr(), s.rope.ptr(), s.q.ptr(), s.att.ptr(), s.delta.ptr());
        let (po, pm, pl) = (s.po.ptr(), s.pm.ptr(), s.pl.ptr());
        self.norm_in1(a.norm.ptr(), m, &a.q)?;
        self.proj(&a.q, xn, m, qg, false)?;
        self.proj(&a.k, xn, m, kraw, false)?;
        self.proj(&a.v, xn, m, vtmp, false)?;
        let eps = self.eps;
        self.run("rows", "attn_prep_r", [256, 1, 1], [HEADS + KV_HEADS, m, 1],
            &[&qg, &kraw, &a.qn.ptr(), &a.kn.ptr(), &rope, &q, &ktmp, &0u64, &eps])?;
        self.run("kvq", "kvq_quant8", [32, 1, 1], [m, 1, 1], &[&ktmp, &st.kc.ptr(), &pos0])?;
        self.run("kvq", "kvq_quant8", [32, 1, 1], [m, 1, 1], &[&vtmp, &st.vc.ptr(), &pos0])?;
        let (kc, vc) = (st.kc.ptr(), st.vc.ptr());
        if m > MAX_ROWS {
            // prompt window: causal attention on the matrix engine, int8 Hadamard scores (qwen_attn_pfx.cl -DI8 -DHAD; Pfs.run)
            let pf = self.pf.as_ref().context("window > 16 rows without prompt buffers")?;
            let (qb, kx, vx, ks, qs, mu, pst) = (pf.qb.ptr(), pf.kx.ptr(), pf.vx.ptr(), pf.ks.ptr(), pf.qs.ptr(), pf.mu.ptr(), pf.st.ptr());
            let tiles = m.div_ceil(8);
            self.run("pfxi_q8", "attn_pfs_prep_i8", [128, 1, 1], [tiles * 24, 1, 1], &[&q, &qb, &qs, &m])?;
            let kall = pos0 + m;
            let mut k0 = 0u32;
            while k0 < kall {
                let k1 = (k0 + PF_RANGE).min(kall);
                let nk = k1 - k0;
                self.run("pfxi_q8", "kvq_expand_k8", [64, 1, 1], [nk, 1, 1], &[&kc, &kx, &ks, &mu, &k0])?;
                self.run("pfxi_q8", "kvq_expand_vt", [64, 1, 1], [nk.div_ceil(64), 128, 1], &[&vc, &vx, &k0, &nk, &PF_RANGE])?;
                let flags = u32::from(k0 == 0) | (u32::from(k1 == kall) << 1);
                self.run("pfxi_q8", "attn_prefill_s", [128, 1, 1], [tiles, 4, 1],
                    &[&qb, &kx, &vx, &qg, &att, &pst, &pos0, &m, &k0, &k0, &k1, &flags, &PF_RANGE, &ks, &qs])?;
                k0 = k1;
            }
            self.proj(&a.o, att, m, delta, false)?;
            self.pending = true;
            return Ok(());
        }
        // decode attention of the m rows (row r over pos0 + 1 + r keys): ARC_DEC=1 kernels on the q8 cache (qwen_attn_long.zig Long.run)
        let len0 = pos0 + 1;
        let nch = (len0 + m - 1).div_ceil(KV_CHUNK);
        let scale = 0.0625f32;
        if m == 1 {
            self.run("attn_dec_q8", "attn_dec_partial", [128, 1, 1], [4, nch, 1], &[&q, &kc, &vc, &po, &pm, &pl, &len0, &scale, &MAX_CHUNKS])?;
        } else {
            let tiles = (6 * m).div_ceil(8);
            let tg = self.pick_tg(tiles, len0 + m - 1);
            let (name, local) = match tg {
                4 => ("attn_dec_pack4", 512),
                2 => ("attn_dec_pack2", 256),
                _ => ("attn_dec_pack", 128),
            };
            self.run("attn_dec_q8", name, [local, 1, 1], [4 * tiles.div_ceil(tg), nch, 1],
                &[&q, &kc, &vc, &po, &pm, &pl, &len0, &scale, &MAX_CHUNKS, &m])?;
        }
        self.run("attn_long_q8", "attn_long_merge", [256, 1, 1], [HEADS, m, 1], &[&po, &pm, &pl, &qg, &att, &len0, &MAX_CHUNKS])?;
        self.proj(&a.o, att, m, delta, false)?;
        self.pending = true;
        Ok(())
    }

    fn mlp(&mut self, li: usize, m: u32) -> Result<()> {
        let p: *const MlpW = &self.w.layers[li].mlp;
        let p = unsafe { &*p };
        let s = &self.sc;
        let (xn, gate, up, delta) = (s.xn.ptr(), s.gate.ptr(), s.up.ptr(), s.delta.ptr());
        self.norm_in1(p.norm.ptr(), m, &p.gate)?;
        self.proj(&p.gate, xn, m, gate, false)?;
        self.proj(&p.up, xn, m, up, false)?;
        let n = m * INTER;
        if m == 1 && self.fuse1 {
            self.prep_swiglu(&p.down, gate, up, gate)?; // act in place of gate, and down's prepass
        } else if self.fusem && (2..=MAX_ROWS).contains(&m) {
            self.prep_swiglu_r(&p.down, gate, up, gate, m)?;
        } else {
            self.run("basic", "swiglu", [64, 1, 1], [n / 64, 1, 1], &[&gate, &up, &gate, &n])?;
        }
        self.proj(&p.down, gate, m, delta, false)?;
        self.pending = true;
        Ok(())
    }

    /// The DFlash2 feature tap after layer li (see [`Taps`]): bf16(x + delta) of rows row0..m.
    fn tap(&mut self, li: usize, m: u32) -> Result<()> {
        let Some(t) = self.taps.as_ref() else { return Ok(()) };
        let Some(j) = t.layers.iter().position(|&l| l == li) else { return Ok(()) };
        if m <= t.row0 {
            return Ok(());
        }
        let n = m - t.row0;
        ensure!(n <= t.cap, "taps: {n} rows (cap {})", t.cap);
        let (buf, row0, ld, j) = (t.buf, t.row0, t.ld, j as u32);
        let (x, d) = (self.sc.x.ptr(), self.sc.delta.ptr());
        self.run("dflash", "df_tap", [256, 1, 1], [n, HIDDEN / 256, 1], &[&x, &d, &buf, &row0, &j, &ld])
    }

    /// `tokens` (1..16) at positions pos..pos+m through all layers; the last `head_rows` rows get logits (fp32 [rows][vocab] in
    /// sc.logits, argmax per row in sc.am_out). `inplace`: the GDN states advance by all rows now (and the position); else the window
    /// stays pending until [`Engine::commit`]. The caller syncs (fetch_* do).
    /// Windows over 16 rows (prompt windows, up to rcap) must be in place, with head_rows 0, 1 or [`KEEP_HIDDEN`].
    pub fn forward_rows(&mut self, tokens: &[u32], head_rows: u32, inplace: bool) -> Result<()> {
        let m = tokens.len() as u32;
        let rcap = self.pf.as_ref().map_or(MAX_ROWS, |p| p.rcap);
        ensure!(m >= 1 && m <= rcap && (head_rows <= m || head_rows == KEEP_HIDDEN), "window of {m} rows (cap {rcap}), head {head_rows}");
        ensure!(m <= MAX_ROWS || (inplace && (head_rows <= 1 || head_rows == KEEP_HIDDEN)), "prompt window of {m} rows: in place, head 0 / 1 / keep");
        let pos0 = self.pos;
        ensure!(pos0 + m <= self.ctx, "context full");
        self.tree = None;
        self.upload_rows(tokens, pos0)?;
        let (emb, tok, x) = (self.w.emb.ptr(), self.sc.tok.ptr(), self.sc.x.ptr());
        self.run("basic", "embed_bf16", [64, 1, 1], [m, 1, 1], &[&emb, &tok, &x, &HIDDEN])?;
        self.pending = false;
        for li in 0..self.cfg.n_layers {
            if self.cfg.full_attn[li] {
                self.attention(li, m, pos0)?;
            } else {
                self.gdn(li, m, inplace)?;
            }
            self.mlp(li, m)?;
            self.tap(li, m)?;
        }
        self.rows = m;
        self.pos0 = pos0;
        self.inplace = inplace;
        self.head_rows = head_rows;
        if inplace {
            self.pos += m;
        }
        if head_rows == 0 {
            return Ok(());
        }
        let fnorm = self.w.final_norm.ptr();
        if head_rows == 1 && m == 1 {
            let head: *const XqW = &self.w.head;
            self.norm_in1(fnorm, m, unsafe { &*head })?;
        } else {
            self.norm_in(fnorm, m)?;
        }
        if head_rows == KEEP_HIDDEN {
            self.head_rows = 0;
            return Ok(());
        }
        self.head_batch(m - head_rows, head_rows)
    }

    /// lm_head over n <= 16 normalised hidden rows from row0 of the last window: fp32 logits [n][vocab], argmax per row.
    pub fn head_batch(&mut self, row0: u32, n: u32) -> Result<()> {
        ensure!((1..=MAX_ROWS).contains(&n), "head batch of {n} rows");
        let v = self.cfg.vocab as u32;
        let head: *const XqW = &self.w.head;
        let x = self.sc.xn.ptr().add::<u16>((row0 * HIDDEN) as usize);
        let (logits, am_s, am_o) = (self.sc.logits.ptr(), self.sc.am_scratch.ptr(), self.sc.am_out.ptr());
        self.xq(unsafe { &*head }, x, n, logits, 0, true)?;
        if self.bf16_logits {
            let tot = n * v;
            self.run("basic", "round_bf16_f32", [64, 1, 1], [tot.div_ceil(64), 1, 1], &[&logits, &tot])?;
        }
        let per = v.div_ceil(ARGMAX_PARTS);
        self.run("basic", "argmax_partial", [256, 1, 1], [ARGMAX_PARTS, n, 1], &[&logits, &am_s, &v, &per])?;
        self.run("basic", "argmax_final", [256, 1, 1], [n, 1, 1], &[&am_s, &am_o, &ARGMAX_PARTS])?;
        self.head_rows = n;
        Ok(())
    }

    fn tree_bufs(&mut self) -> Result<()> {
        if self.tbuf.is_some() {
            return Ok(());
        }
        let d = &self.dev;
        let rec = KvMode::Q8.rec_bytes();
        let n16 = MAX_ROWS as usize;
        self.tbuf = Some(TreeBufs {
            table: d.zeros(4 * n16)?,
            sscr: d.zeros(TREE_SLOTS as usize * (GDN_HEADS * 128 * 128) as usize)?,
            slots: TREE_SLOTS,
            tk: d.zeros(n16 * TREE_TAIL_KEYS as usize * 4 * rec)?,
            tv: d.zeros(n16 * TREE_TAIL_KEYS as usize * 4 * rec)?,
            path: d.zeros(n16)?,
            gqkv: d.zeros(n16 * GDN_QKV as usize)?,
            gb: d.zeros(n16 * GDN_HEADS as usize)?,
            ga: d.zeros(n16 * GDN_HEADS as usize)?,
        });
        Ok(())
    }

    /// A draft-tree verify window (never in place): row t = tokens[t] at position pos + depth[t], its K / V at cache slot pos + t;
    /// `par` = the parent row of every row (par[0] = -1: row 0 continues the committed sequence; 0 <= par[t] < t). Every row's
    /// logits (head on all rows) are bit-identical to decoding its root path alone (the tree kernels: conv over each row's
    /// predecessor chain, the delta rule along the branches with saved branch states, per-row K / V tail windows in the decode
    /// attention). Keep a root path with [`Engine::commit_tree`].
    pub fn forward_tree(&mut self, tokens: &[u32], par: &[i32]) -> Result<()> {
        let m = tokens.len() as u32;
        ensure!((1..=MAX_ROWS).contains(&m) && par.len() == tokens.len(), "tree window of {m} rows");
        let tt = TreeTable::new(par)?;
        self.tree_bufs()?;
        ensure!(tt.slots <= self.tbuf.as_ref().unwrap().slots, "tree needs {} state slots", tt.slots);
        let pos0 = self.pos;
        let maxd = *tt.depth.iter().max().unwrap();
        ensure!(pos0 + m <= self.ctx, "context full");
        // tokens, the RoPE rows of each row's own position (pos0 + depth), the table
        self.dev.q.upload(self.sc.tok.ptr(), tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut tab = vec![0f32; m as usize * 64];
        for r in 0..m as usize {
            rope::row(&self.inv_freq, pos0 + tt.depth[r], &mut tab[r * 64..(r + 1) * 64]);
        }
        self.dev.q.upload(self.sc.rope.ptr(), &tab).map_err(|e| anyhow::anyhow!("{e}"))?;
        let tb = self.tbuf.as_ref().unwrap();
        self.dev.q.upload(tb.table.ptr(), &tt.words).map_err(|e| anyhow::anyhow!("{e}"))?;
        let (table, sscr, tk, tv) = (tb.table.ptr(), tb.sscr.ptr(), tb.tk.ptr(), tb.tv.ptr());
        self.xq_key = None;
        self.pf_key = None;
        let (emb, tok, x) = (self.w.emb.ptr(), self.sc.tok.ptr(), self.sc.x.ptr());
        self.run("basic", "embed_bf16", [64, 1, 1], [m, 1, 1], &[&emb, &tok, &x, &HIDDEN])?;
        self.pending = false;
        for li in 0..self.cfg.n_layers {
            if self.cfg.full_attn[li] {
                self.attention_tree(li, m, pos0, maxd, table, tk, tv)?;
            } else {
                self.gdn_tree(li, m, table, sscr)?;
            }
            self.mlp(li, m)?;
            self.tap(li, m)?;
        }
        self.rows = m;
        self.pos0 = pos0;
        self.inplace = false;
        let fnorm = self.w.final_norm.ptr();
        self.norm_in(fnorm, m)?;
        self.head_batch(0, m)?;
        self.tree = Some(tt);
        Ok(())
    }

    fn gdn_tree(&mut self, li: usize, m: u32, table: DevPtr, sscr: DevPtr) -> Result<()> {
        let Mixer::Gdn(g) = &self.w.layers[li].mixer else { unreachable!() };
        let LayerState::Gdn(gs) = &self.st[li] else { unreachable!() };
        let g: *const GdnW = g;
        let gs: *const GdnState = gs;
        let (g, gs) = unsafe { (&*g, &*gs) };
        let (wqkv, wb, wa) = (gs.wqkv.ptr(), gs.wb.ptr(), gs.wa.ptr());
        let s = &self.sc;
        let (xn, z, y, yg, delta) = (s.xn.ptr(), s.z.ptr(), s.y.ptr(), s.yg.ptr(), s.delta.ptr());
        let (qn, kn, vconv, gbeta, ggate) = (s.qn.ptr(), s.kn.ptr(), s.vconv.ptr(), s.gbeta.ptr(), s.ggate.ptr());
        self.norm_in(g.norm.ptr(), m)?;
        self.proj(&g.qkv, xn, m, wqkv, false)?;
        self.proj(&g.z, xn, m, z, false)?;
        self.proj(&g.b, xn, m, wb, false)?;
        self.proj(&g.a, xn, m, wa, false)?;
        let (eps, conv, cst, sst) = (self.eps, g.conv.ptr(), gs.cstate.ptr(), gs.sstate.ptr());
        if self.gdnpre {
            let gx = if self.gates_by_proj { 100 } else { 101 };
            self.run("rows", "gdn_pre_tr", [64, 1, 1], [gx, m, 1],
                &[&wqkv, &cst, &conv, &qn, &kn, &vconv, &GDN_QKV, &wb, &wa, &g.a_log.ptr(), &g.dt_bias.ptr(), &gbeta, &ggate, &table])?;
        } else {
            self.run("rows", "gdn_conv_prep_tr", [16, 1, 1], [16, m, 1], &[&wqkv, &cst, &conv, &qn, &kn, &GDN_QKV, &table])?;
            self.run("rows", "gdn_vconv_tr", [64, 1, 1], [m, 96, 1], &[&wqkv, &cst, &conv, &vconv, &GDN_QKV, &table])?;
            if !self.gates_by_proj {
                self.run("rows", "gdn_gates_r", [48, 1, 1], [m, 1, 1], &[&wb, &wa, &g.a_log.ptr(), &g.dt_bias.ptr(), &gbeta, &ggate])?;
            }
        }
        self.run("rows", "gdn_step_tr", [64, 1, 1], [GDN_HEADS, 8, 1], &[&qn, &kn, &vconv, &gbeta, &ggate, &sst, &sscr, &y, &0u32, &m, &table])?;
        self.run("rows", "gdn_gate_norm_r", [16, 1, 1], [GDN_HEADS, m, 1], &[&y, &z, &g.gnorm.ptr(), &yg, &eps])?;
        self.proj(&g.out, yg, m, delta, false)?;
        self.pending = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_tree(&mut self, li: usize, m: u32, pos0: u32, maxd: u32, table: DevPtr, tk: DevPtr, tv: DevPtr) -> Result<()> {
        let Mixer::Attn(a) = &self.w.layers[li].mixer else { unreachable!() };
        let LayerState::Attn(st) = &self.st[li] else { unreachable!() };
        let a: *const AttnW = a;
        let st: *const AttnState = st;
        let (a, st) = unsafe { (&*a, &*st) };
        let s = &self.sc;
        let (xn, qg, kraw, vtmp, ktmp, rope, q, att, delta) =
            (s.xn.ptr(), s.qg.ptr(), s.kraw.ptr(), s.vtmp.ptr(), s.ktmp.ptr(), s.rope.ptr(), s.q.ptr(), s.att.ptr(), s.delta.ptr());
        let (po, pm, pl) = (s.po.ptr(), s.pm.ptr(), s.pl.ptr());
        self.norm_in(a.norm.ptr(), m)?;
        self.proj(&a.q, xn, m, qg, false)?;
        self.proj(&a.k, xn, m, kraw, false)?;
        self.proj(&a.v, xn, m, vtmp, false)?;
        let eps = self.eps;
        self.run("rows", "attn_prep_r", [256, 1, 1], [HEADS + KV_HEADS, m, 1], &[&qg, &kraw, &a.qn.ptr(), &a.kn.ptr(), &rope, &q, &ktmp, &0u64, &eps])?;
        self.run("kvq", "kvq_quant8", [32, 1, 1], [m, 1, 1], &[&ktmp, &st.kc.ptr(), &pos0])?;
        self.run("kvq", "kvq_quant8", [32, 1, 1], [m, 1, 1], &[&vtmp, &st.vc.ptr(), &pos0])?;
        let (kc, vc) = (st.kc.ptr(), st.vc.ptr());
        let kbase = pos0 & !31;
        self.run("rows", "kv_tree_tail", [272, 1, 1], [TREE_TAIL_KEYS, m, 1], &[&kc, &vc, &tk, &tv, &table, &pos0, &kbase])?;
        let nch = (pos0 + maxd + 1).div_ceil(KV_CHUNK);
        let tiles = (6 * m).div_ceil(8);
        let tg = self.pick_tg(tiles, pos0 + maxd + 1);
        let (name, local) = match tg {
            4 => ("attn_dec_tree4", 512),
            2 => ("attn_dec_tree2", 256),
            _ => ("attn_dec_tree", 128),
        };
        let scale = 0.0625f32;
        self.run("attn_dec_q8", name, [local, 1, 1], [4 * tiles.div_ceil(tg), nch, 1],
            &[&q, &kc, &vc, &po, &pm, &pl, &pos0, &scale, &MAX_CHUNKS, &m, &table, &tk, &tv])?;
        self.run("attn_long_q8", "attn_long_merge_tr", [256, 1, 1], [HEADS, m, 1], &[&po, &pm, &pl, &qg, &att, &pos0, &MAX_CHUNKS, &table])?;
        self.proj(&a.o, att, m, delta, false)?;
        self.pending = true;
        Ok(())
    }

    /// After a [`Engine::forward_tree`] window: keep the root path `path` (window rows, path[0] = 0, each the parent's child):
    /// the path's K / V move to cache positions pos .. pos + n (kv_tree_commit), and every GDN layer replays the path's rows from
    /// their saved inputs gathered in path order (rows_gather_u16), as a chain commit of n rows would.
    pub fn commit_tree(&mut self, path: &[usize]) -> Result<()> {
        let tt = self.tree.take().context("commit_tree after a window that was not a tree")?;
        let n = path.len() as u32;
        ensure!(!self.inplace && n >= 1 && n <= self.rows && path[0] == 0
            && path.windows(2).all(|w| tt.par[w[1]] == w[0] as i32), "commit_tree: path {path:?} of {:?}", tt.par);
        let p32: Vec<u32> = path.iter().map(|&r| r as u32).collect();
        let tb = self.tbuf.as_ref().unwrap();
        self.dev.q.upload(tb.path.ptr(), &p32).map_err(|e| anyhow::anyhow!("{e}"))?;
        let (pp, gq, gb, ga) = (tb.path.ptr(), tb.gqkv.ptr(), tb.gb.ptr(), tb.ga.ptr());
        let pos0 = self.pos0;
        for li in 0..self.cfg.n_layers {
            match (&self.w.layers[li].mixer, &self.st[li]) {
                (Mixer::Attn(_), LayerState::Attn(st)) => {
                    let (kc, vc) = (st.kc.ptr(), st.vc.ptr());
                    self.run("rows", "kv_tree_commit", [256, 1, 1], [1, 1, 1], &[&kc, &vc, &pp, &n, &pos0])?;
                }
                (Mixer::Gdn(g), LayerState::Gdn(gs)) => {
                    let g: *const GdnW = g;
                    let gs: *const GdnState = gs;
                    let (g, gs) = unsafe { (&*g, &*gs) };
                    let (sq, sb, sa) = (gs.wqkv.ptr(), gs.wb.ptr(), gs.wa.ptr());
                    let (eq, eh) = (GDN_QKV, GDN_HEADS);
                    self.run("rows", "rows_gather_u16", [256, 1, 1], [n, eq.div_ceil(256), 1], &[&sq, &gq, &pp, &eq])?;
                    self.run("rows", "rows_gather_u16", [256, 1, 1], [n, eh.div_ceil(256), 1], &[&sb, &gb, &pp, &eh])?;
                    self.run("rows", "rows_gather_u16", [256, 1, 1], [n, eh.div_ceil(256), 1], &[&sa, &ga, &pp, &eh])?;
                    self.gdn_core(g, gs.cstate.ptr(), gs.sstate.ptr(), (gq, gb, ga), n, true, gs.sstate.ptr(), false)?;
                }
                _ => unreachable!(),
            }
        }
        self.pos += n;
        self.inplace = true;
        Ok(())
    }

    /// After a non-inplace window: keep its first c rows. All rows: the pending GDN state becomes current; fewer: the kept rows are
    /// replayed in place from the saved inputs. The KV cache needs nothing.
    pub fn commit(&mut self, c: u32) -> Result<()> {
        ensure!(!self.inplace && c <= self.rows, "commit {c} of {} rows (inplace {})", self.rows, self.inplace);
        ensure!(self.tree.is_none(), "commit after a tree window: use commit_tree");
        if c == 0 {
            return Ok(());
        }
        for li in 0..self.cfg.n_layers {
            if self.cfg.full_attn[li] {
                continue;
            }
            let Mixer::Gdn(g) = &self.w.layers[li].mixer else { unreachable!() };
            let g: *const GdnW = g;
            let g = unsafe { &*g };
            if c == self.rows {
                let LayerState::Gdn(gs) = &mut self.st[li] else { unreachable!() };
                std::mem::swap(&mut gs.sstate, &mut gs.sstate2);
                let (wqkv, cst) = (gs.wqkv.ptr(), gs.cstate.ptr());
                self.run("rows", "gdn_conv_commit", [64, 1, 1], [GDN_QKV.div_ceil(64), 1, 1], &[&wqkv, &cst, &GDN_QKV, &c, &0u32, &GDN_QKV])?;
            } else {
                let LayerState::Gdn(gs) = &self.st[li] else { unreachable!() };
                let gs: *const GdnState = gs;
                let gs = unsafe { &*gs };
                self.gdn_core(g, gs.cstate.ptr(), gs.sstate.ptr(), (gs.wqkv.ptr(), gs.wb.ptr(), gs.wa.ptr()), c, true, gs.sstate.ptr(), false)?;
            }
        }
        self.pos += c;
        self.inplace = true;
        Ok(())
    }

    /// A second set of KV caches (the 16 attention layers and, with `mtp`, the MTP head's) for `ctx` positions: arc-engine's side pool.
    pub fn alloc_pool(&self, ctx: u32, mtp: bool) -> Result<KvPool> {
        let n = ctx as usize * 4 * KvMode::Q8.rec_bytes();
        let mut kv = Vec::new();
        for s in &self.st {
            if let LayerState::Attn(_) = s {
                kv.push((self.dev.zeros::<u8>(n)?, self.dev.zeros::<u8>(n)?));
            }
        }
        let mtp = if mtp { Some((self.dev.zeros::<u8>(n)?, self.dev.zeros::<u8>(n)?)) } else { None };
        Ok(KvPool { ctx, kv, mtp })
    }

    /// Points the attention layers and the MTP head at `p`'s caches and gives `p` the current ones, the context limit with them.
    /// The GDN state and the position stay: the caller restores the sequence of the pool it switched to (checkpoints).
    pub fn swap_kv(&mut self, mt: Option<&mut crate::mtp::Mtp>, p: &mut KvPool) {
        let mut j = 0;
        for s in &mut self.st {
            if let LayerState::Attn(a) = s {
                std::mem::swap(&mut a.kc, &mut p.kv[j].0);
                std::mem::swap(&mut a.vc, &mut p.kv[j].1);
                j += 1;
            }
        }
        if let (Some(mt), Some(pm)) = (mt, p.mtp.as_mut()) {
            std::mem::swap(&mut mt.kc, &mut pm.0);
            std::mem::swap(&mut mt.vc, &mut pm.1);
        }
        std::mem::swap(&mut self.ctx, &mut p.ctx);
        self.invalidate();
    }

    /// Copies positions 0..n of `src`'s caches into the current ones (a side sequence promoted to the main pool). Queued.
    pub fn copy_kv_from(&self, mt: Option<&crate::mtp::Mtp>, src: &KvPool, n: usize) -> Result<()> {
        ensure!(n <= src.ctx as usize && n <= self.ctx as usize, "copy of {n} positions: pools of {} / {}", src.ctx, self.ctx);
        let bytes = n * 4 * KvMode::Q8.rec_bytes();
        let q = &self.dev.q;
        let mut j = 0;
        for s in &self.st {
            if let LayerState::Attn(a) = s {
                q.copy_dtod(a.kc.ptr(), src.kv[j].0.ptr(), bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
                q.copy_dtod(a.vc.ptr(), src.kv[j].1.ptr(), bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
                j += 1;
            }
        }
        if let (Some(mt), Some(sm)) = (mt, src.mtp.as_ref()) {
            q.copy_dtod(mt.kc.ptr(), sm.0.ptr(), bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
            q.copy_dtod(mt.vc.ptr(), sm.1.ptr(), bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Ok(())
    }

    /// Bytes of one GDN-state checkpoint (every GDN layer's conv + delta-rule state).
    pub fn ckpt_bytes(&self) -> usize {
        self.st.iter().map(|s| match s {
            LayerState::Gdn(g) => g.cstate.bytes() + g.sstate.bytes(),
            LayerState::Attn(_) => 0,
        }).sum()
    }

    /// Copies every GDN layer's state into `slot` (device to device, queued): the prefix cache's resume point at the current position.
    pub fn snapshot(&self, slot: &DeviceBuffer<u8>) -> Result<()> {
        let mut off = 0;
        for s in &self.st {
            if let LayerState::Gdn(g) = s {
                for (src, n) in [(g.cstate.ptr(), g.cstate.bytes()), (g.sstate.ptr(), g.sstate.bytes())] {
                    self.dev.q.copy_dtod(slot.ptr().byte_add(off), src, n).map_err(|e| anyhow::anyhow!("{e}"))?;
                    off += n;
                }
            }
        }
        Ok(())
    }

    /// The inverse of [`Engine::snapshot`], then the position `pos` (the KV cache is valid below it).
    pub fn restore(&mut self, slot: &DeviceBuffer<u8>, pos: u32) -> Result<()> {
        let mut off = 0;
        for s in &self.st {
            if let LayerState::Gdn(g) = s {
                for (dst, n) in [(g.cstate.ptr(), g.cstate.bytes()), (g.sstate.ptr(), g.sstate.bytes())] {
                    self.dev.q.copy_dtod(dst, slot.ptr().byte_add(off), n).map_err(|e| anyhow::anyhow!("{e}"))?;
                    off += n;
                }
            }
        }
        self.pos = pos;
        self.pending = false;
        self.invalidate();
        Ok(())
    }

    /// fp32 logits of the head rows of the last window, [head_rows][vocab] (syncs).
    pub fn fetch_logits(&self, out: &mut [f32]) -> Result<()> {
        let n = self.head_rows as usize * self.cfg.vocab;
        self.dev.download(&mut out[..n], self.sc.logits.ptr())
    }

    /// Greedy token of each head row (syncs).
    pub fn argmax(&self, out: &mut [i32]) -> Result<()> {
        self.dev.download(&mut out[..self.head_rows as usize], self.sc.am_out.ptr())
    }
}

/// The kernel source: FE_XE_KERNELS=rust (default: the Rust kernels embedded at build time), ref (arc-engine's reference SPIR-V) or
/// mixed (Rust where it exists, else the reference); ref and mixed are test only and need the `ref-spv` feature. Then the per-kernel /
/// per-module Rust overrides of FE_XE_RUST_SPV / FE_XE_RUST_ONLY.
pub fn kernels(dev: &Dev) -> Result<Kx> {
    let mut kx = match std::env::var("FE_XE_KERNELS").as_deref().unwrap_or("rust") {
        #[cfg(feature = "ref-spv")]
        "ref" => {
            eprintln!("fe-xe: kernels = arc-engine reference SPIR-V (test only)");
            Kx::reference(&dev.ze, None)
        }
        #[cfg(feature = "ref-spv")]
        "mixed" => {
            eprintln!("fe-xe: kernels = Rust where built, else arc-engine reference SPIR-V (test only)");
            Kx::mixed(&dev.ze, None)
        }
        "rust" => Kx::rust(&dev.ze),
        k => bail!("FE_XE_KERNELS={k}: unknown (or built without the ref-spv feature)"),
    };
    let taken = kx.overrides_from_env()?;
    if !taken.is_empty() {
        eprintln!("fe-xe: Rust kernels in use: {}", taken.join(" "));
    }
    Ok(kx)
}

/// Loads an XQ checkpoint onto the B70 and builds an engine with a `ctx`-token KV cache and windows of up to `rcap` rows.
pub fn open(model: &std::path::Path, ctx: u32, rcap: u32) -> Result<Engine> {
    let cfg = Config::load(&Config::locate(model))?;
    cfg.check_supported()?;
    let ck = Checkpoint::open(model)?;
    let dev = Dev::new()?;
    eprintln!("fe-xe: {} ({} MHz), loading {model:?}", dev.ze.info.name, dev.ze.info.core_clock_mhz);
    let w = Weights::load(&dev, &ck, &cfg, true)?;
    let kx = kernels(&dev)?;
    let e = Engine::new(dev, kx, cfg, w, ctx, rcap.max(MAX_ROWS))?;
    eprintln!("fe-xe: ready, ctx {ctx}, windows up to {} rows, {:.2} GB on the device", rcap.max(MAX_ROWS), e.dev.bytes.get() as f64 / 1e9);
    Ok(e)
}

/// Steps (32 keys) of the longest sub-group run (128 keys) in the last chunk of `keys` keys.
fn dec_last_steps(keys: u32) -> u32 {
    let last = keys - (keys.div_ceil(KV_CHUNK) - 1) * KV_CHUNK;
    last.min(128).div_ceil(32)
}

/// Makespan (arbitrary units) of a window decode attention with `tgn` tiles a work-group: the grid (4 kv heads x
/// ceil(tiles / tgn), chunks of KV_CHUNK keys) dispatched in order (chunk-major) onto `cores` Xe cores, each holding
/// 4 / tgn work-groups (the 256-GRF module: 4 threads an XVE, 8 XVEs a core = 32 threads; a work-group is 8 tgn
/// sub-group threads). A work-group takes 28 + 12 x (steps of its longest 128-key run): 76 for a full chunk, 40 for
/// a chunk of <= 32 keys (measured on the B70: 75-78 / ~40 us). Padding tiles (tiles not a multiple of tgn) cost
/// their threads but no time.
pub fn dec_makespan(tiles: u32, keys: u32, tgn: u32, cores: u32) -> u64 {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let nch = keys.div_ceil(KV_CHUNK);
    let per_chunk = 4 * tiles.div_ceil(tgn);
    let slots = (cores * 4 / tgn).max(1) as usize;
    let mut free: BinaryHeap<Reverse<u64>> = (0..slots).map(|_| Reverse(0)).collect();
    let (mut start, mut end) = (0u64, 0u64);
    for c in 0..nch {
        let kc = (keys - c * KV_CHUNK).min(KV_CHUNK);
        let d = 28 + 12 * kc.min(128).div_ceil(32) as u64;
        for _ in 0..per_chunk {
            let Reverse(t) = free.pop().unwrap();
            start = start.max(t);
            end = end.max(start + d);
            free.push(Reverse(start + d));
        }
    }
    end
}

/// Tiles a work-group (1, 2 or 4) for a window decode attention of `tiles` tiles (8 (row, head) pairs each) over `keys`
/// keys (the longest row) on `cores` Xe cores. Every choice gives the same bits (a tile's arithmetic does not depend on
/// the tiles beside it), so this is speed only: the shortest [`dec_makespan`], and among the forms within 3 % of it
/// 2 tiles (two tiles share their K / V reads in the L1, and half-core work-groups fill the tail of the grid), then 1,
/// then 4. The old fixed pick (4 from 4 tiles) put the 8-row window's 40 full-core work-groups at 4K keys on 32 cores
/// in two waves: 115 us vs 78 us for 2 tiles a group.
pub fn dec_tgn(tiles: u32, keys: u32, cores: u32) -> u32 {
    if tiles < 2 {
        return 1;
    }
    let cand: &[u32] = if tiles >= 4 { &[2, 1, 4] } else { &[2, 1] };
    let t: Vec<(u32, u64)> = cand.iter().map(|&g| (g, dec_makespan(tiles, keys, g, cores))).collect();
    let best = t.iter().map(|x| x.1).min().unwrap();
    t.iter().find(|x| x.1 * 100 <= best * 103).unwrap().0
}

#[cfg(test)]
mod dec_tg_tests {
    use super::*;

    /// The picks of the B70 timing sweep (us for pack / pack2 / pack4): the form picked must be the
    /// fastest or within ~4 % of it.
    #[test]
    fn picks_follow_the_b70_sweep() {
        let tiles = |m: u32| (6 * m).div_ceil(8);
        // m = 8: 2048 68.5 / 69.4 / 75.2; 4096 78.3 / 78.7 / 115.3; 4608 86.3 / 79.1 / 149.9; 5120 108.3 / 108.0 / 150.7;
        // 6144 156.3 / 147.9 / 150.9; 8192 170.9 / 152.1 / 173.8; 16384 284.8 / 267.3 / 322.7
        for len0 in [2048u32, 3072, 4096, 4608, 5120, 6144, 8192, 16384] {
            assert_eq!(dec_tgn(tiles(8), len0 + 7, 32), 2, "m 8 len0 {len0}");
        }
        // m = 4 (3 tiles): 8192 78.1 / 103.3 / 110.2 (pack: one wave; pack2's padding tile: two)
        assert_eq!(dec_tgn(tiles(4), 8192 + 3, 32), 1);
        // m = 16: 2048 82.2 / 81.0 / 115.9; 4096 162.7 / 152.3 / 152.3; 8192 273.9 / 261.8 / 265.1
        for len0 in [2048u32, 4096, 8192] {
            assert_eq!(dec_tgn(tiles(16), len0 + 15, 32), 2, "m 16 len0 {len0}");
        }
        assert_eq!(dec_tgn(1, 5000, 32), 1);
    }

    /// `cargo test -p fe-xe --lib dec_tg_table -- --ignored --nocapture`: the picks over rows x context.
    #[test]
    #[ignore]
    fn dec_tg_table() {
        for m in 2..=16u32 {
            let t = (6 * m).div_ceil(8);
            let row: Vec<String> = [512u32, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768, 65536, 98304, 131000]
                .iter().map(|&l| format!("{}", dec_tgn(t, l + m - 1, 32))).collect();
            println!("m {m:>2} tiles {t:>2}: {}", row.join(" "));
        }
    }

    #[test]
    fn makespan_reproduces_the_wave_cliff() {
        // 8 rows at 4096 keys: 4 tiles a group = 32 full + 8 tail work-groups of a full core: two waves
        let (p4, p2) = (dec_makespan(6, 4103, 4, 32), dec_makespan(6, 4103, 2, 32));
        assert_eq!(p2, 76);
        assert_eq!(p4, 76 + 40);
    }
}
