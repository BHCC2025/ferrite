//! Qwen3.5 forward pass on k-core kernels: weights, per-slot KV cache + GDN state, batched forward.
//!
//! A forward call takes segments (slot, tokens). All rows of all segments go through the linear layers
//! together; attention and GDN use each row's slot and position. Prefill = one long segment, decode =
//! many length-1 segments (one per slot), and they can be mixed.
//!
//! Linears: bf16 weights run the tensor-core GEMM (prompt rows); EXL3 trellis weights, when loaded, run
//! fe-exl3's GEMV for small row counts (decode).
use crate::config::Config;
use crate::safetensors::SafeTensors;
use anyhow::{ensure, Context as _, Result};
use fe_cuda::{memcpy_htod, DevPtr, DeviceBuffer, Function, Module, Stream};
use std::path::Path;

pub struct Kernels {
    pub embed: Function,
    pub rmsnorm: Function,
    pub gemm: Function,
    pub gemm_swiglu: Function,
    pub interleave_gu: Function,
    pub gemv: Function,
    /// fe_gemv_bf16_kw: one block a column, for narrow plain linears (GDN in_proj_a|b)
    pub gemv_kw: Function,
    pub silu_mul: Function,
    pub gather_rows: Function,
    pub copy2d: Function,
    pub rows_gather: Function,
    pub argmax: Function,
    pub add: Function,
    pub attn_prep: Function,
    pub attn_split: Function,
    pub attn_combine: Function,
    pub flash: Function,
    pub gdn_conv: Function,
    pub gdn_rec: Function,
    pub gdn_rec2: Function,
    pub gdn_norm: Function,
    /// fe_gdn_rec2 over every GDN layer in one launch (Engine::commit's replay)
    pub gdn_replay: Function,
    /// verify segments (2..16 rows of a slot): the shared cache part on tensor cores + the masked segment part
    pub attn_vk: Function,
    /// the quantized K/V cache forms (k-core kvq.rs): prep, flash, vk and combine, at 4 and at 6 bits
    pub kvq: [[Function; 4]; 2],
}
impl Kernels {
    /// One rank of a two-way head split: the head-dependent kernels from kernels/k-core-tp2's module.
    pub fn with_tp2(mut self, m2: &Module) -> Result<Kernels> {
        self.attn_prep = m2.function("fe_attn_prep")?;
        self.attn_split = m2.function("fe_attn_split")?;
        self.attn_combine = m2.function("fe_attn_combine")?;
        self.flash = {
            let f = m2.function("fe_flash_prefill")?;
            f.set_max_dynamic_smem(FLASH_SMEM)?;
            f
        };
        self.gdn_conv = m2.function("fe_gdn_conv")?;
        self.gdn_rec = m2.function("fe_gdn_rec")?;
        self.gdn_rec2 = m2.function("fe_gdn_rec2")?;
        self.gdn_norm = m2.function("fe_gdn_norm")?;
        self.gdn_replay = m2.function("fe_gdn_replay")?;
        self.attn_vk = {
            let f = m2.function("fe_attn_vk")?;
            f.set_max_dynamic_smem(SEG_SMEM)?;
            f
        };
        Ok(self)
    }

    pub fn load(m: &Module) -> Result<Kernels> {
        Ok(Kernels {
            embed: m.function("fe_embed")?,
            rmsnorm: m.function("fe_rmsnorm")?,
            gemm: m.function("fe_gemm_bf16")?,
            gemm_swiglu: m.function("fe_gemm_swiglu")?,
            interleave_gu: m.function("fe_interleave_gu")?,
            gemv: m.function("fe_gemv_bf16")?,
            gemv_kw: m.function("fe_gemv_bf16_kw")?,
            silu_mul: m.function("fe_silu_mul")?,
            gather_rows: m.function("fe_gather_rows")?,
            copy2d: m.function("fe_copy2d")?,
            rows_gather: m.function("fe_rows_gather")?,
            argmax: m.function("fe_argmax")?,
            add: m.function("fe_add_bf16")?,
            attn_prep: m.function("fe_attn_prep")?,
            attn_split: m.function("fe_attn_split")?,
            attn_combine: m.function("fe_attn_combine")?,
            flash: {
                let f = m.function("fe_flash_prefill")?;
                f.set_max_dynamic_smem(FLASH_SMEM)?;
                f
            },
            gdn_conv: m.function("fe_gdn_conv")?,
            gdn_rec: m.function("fe_gdn_rec")?,
            gdn_rec2: m.function("fe_gdn_rec2")?,
            gdn_norm: m.function("fe_gdn_norm")?,
            gdn_replay: m.function("fe_gdn_replay")?,
            attn_vk: {
                let f = m.function("fe_attn_vk")?;
                f.set_max_dynamic_smem(SEG_SMEM)?;
                f
            },
            kvq: {
                // the flash kernel with producer warps (the same bits, the dequantization behind the tensor cores) unless
                // FE_FLASH_PROD=0
                let prod = flash_prod();
                let set = |bits: usize| -> Result<[Function; 4]> {
                    let (fl, vk) = (m.function(&format!("fe_flash_prefill_q{bits}{}", if prod { "p" } else { "" }))?,
                        m.function(&format!("fe_attn_vk_q{bits}"))?);
                    fl.set_max_dynamic_smem(flash_q_smem(bits))?;
                    vk.set_max_dynamic_smem(seg_q_smem(bits))?;
                    Ok([m.function(&format!("fe_attn_prep_q{bits}"))?, fl, vk, m.function("fe_attn_combine_q")?])
                };
                [set(4)?, set(6)?]
            },
        })
    }
}

const GEMM_SMEM: u32 = 3 * 2 * 128 * 32 * 2;
const FLASH_SMEM: u32 = 64 * 512 + 2 * 2 * 32 * 512;
/// k-core attn.rs SEG_SMEM: Q [6 heads][16][256] + 3 x (K + V) [16][256], bf16.
const SEG_SMEM: u32 = 6 * 16 * 512 + 3 * 2 * 16 * 512;
/// k-core kvq.rs (the quantized K/V cache): bytes of one head's record at 4 or 6 bits a value, and the kernels'
/// dynamic shared memory.
const fn kvq_rec(bits: usize) -> usize {
    if bits == 6 { 208 } else { 144 }
}
fn flash_q_smem(bits: usize) -> u32 {
    if flash_prod() {
        (64 * 512 + 2 * 32 * 512 + 2 * 32 * kvq_rec(bits)) as u32
    } else {
        (64 * 512 + 2 * 2 * 32 * kvq_rec(bits) + 2 * 32 * 512) as u32
    }
}
/// FE_FLASH_PROD (default 1): fe_flash_prefill_q*p, the quantized flash kernel with four producer warps (block 256).
fn flash_prod() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FE_FLASH_PROD").map(|v| v != "0").unwrap_or(true))
}
const fn seg_q_smem(bits: usize) -> u32 {
    (6 * 16 * 512 + 3 * 2 * 16 * kvq_rec(bits) + 2 * 16 * 512) as u32
}

/// FE_KVQ = 6 | 4 | 0 (default: 6 on small-memory devices, crate::lowmem; else 0 = a bf16 cache): the K/V cache as
/// 6-bit or 4-bit records (rotated, group scaled; k-core kvq.rs): 832 or 576 bytes a token and layer for K and V
/// each instead of 2048. KL against BF16 at 6 bits is the bf16 cache's; 4 bits cost 4-8%. One GPU only.
fn kvq_bits() -> Result<usize> {
    match std::env::var("FE_KVQ").ok().as_deref() {
        Some("0") => Ok(0),
        Some("4") => Ok(4),
        Some("6") | Some("1") => Ok(6),
        Some(v) => anyhow::bail!("FE_KVQ={v}: 0, 4 or 6"),
        None => Ok(if crate::lowmem::on() { 6 } else { 0 }),
    }
}

/// k-core attn.rs VK_SPLITS: fe_attn_vk's splits a row (grid z, the combine's nsplit).
const VK_SPLITS: u32 = 48;

/// FE_ATTN_SEG (default 1): every run of short segments (decode rows, chain and tree verifies) takes fe_attn_vk:
/// one tensor-core kernel for decode and verify rows, each row partitioned by its own key count, K/V tiles shared
/// across a segment's rows where their keys coincide, so a verify row is bitwise its 1-row decode. 0: fe_attn_split.
fn seg_attn() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FE_ATTN_SEG").map(|v| v != "0").unwrap_or(true))
}
/// Segments at least this long use the tensor-core flash kernel; shorter ones the split-K path.
const FLASH_MIN: usize = 17; // segments of up to 16 rows (decode, 16-row verifies) take the split kernel
/// rows of one fe_attn_vk segment at most (k-core attn.rs: its per-row metadata and Q tile hold 16)
const VK_ROWS: u32 = 16;
/// Row counts up to this use the EXL3 GEMV when a linear has a trellis form.
pub const X3_ROWS: usize = 16;
/// Decode / verify rows per EXL3 GEMV call: one call reads the weights once (32 rows cost a TP2 rank 34.8 ms over
/// all linears, two 16-row calls 56.9); 33-48 rows run as one 48-row pass (g6), 49-64 as 48 + 16 (or 32 + 32 with
/// `FE_EXL3_G6=0`). A row's bits do not depend on the call's row count (exl3-rowcheck 1-64, TP1 and both TP2
/// ranks' slices).
pub const X3_PASS: usize = 64;

/// EXL3 decode machinery shared by all linears (one workspace: calls on one stream run in order).
pub struct X3 {
    pub kern: fe_exl3::gpu::Kernels,
    pub ws: fe_exl3::linear::Workspace,
}

/// Per-call context for linears.
#[derive(Clone, Copy)]
pub struct LinCtx<'a> {
    pub k: &'a Kernels,
    pub x3: Option<&'a X3>,
    /// NVFP4/FP8 prompt GEMMs (fe-fp4) and their activation workspace
    pub fp4: Option<(&'a fe_fp4::Kernels, &'a fe_fp4::Workspace)>,
    pub s: &'a Stream,
    /// scratch for decode rows of fused prompt projections: [DEC_MAX][max projection width]
    pub tmp: DevPtr,
    /// the first `dec` rows of the batch are decode / verify rows: they always run the EXL3 GEMV (when the
    /// linear has a trellis form), whatever the batch size; prompt rows use the prompt weights
    pub dec: usize,
}

/// Most decode/verify rows a batch carries (routing scratch size).
pub const DEC_MAX: usize = 64;
/// Tokens of a prompt-cache KV block (Engine::cache_pool / kv_out / kv_in).
pub const KV_BLOCK: usize = 2048;
/// buffers one commit_tree gathers (K and V per full-attention layer, two per GDN layer, the feature taps)
const GATHER_MAX: usize = 256;

/// A tree row's last three ancestors for fe_gdn_conv: p1 | p2 << 8 | p3 << 16 (segment rows, 0xff above the root).
pub fn ancestors(par: &[i32], row: usize) -> u32 {
    let mut a = 0xff_ffffu32;
    let mut r = par[row];
    for k in 0..3 {
        if r < 0 {
            break;
        }
        a = (a & !(0xff << (8 * k))) | (r as u32) << (8 * k);
        r = par[r as usize];
    }
    a
}
fn ancestors_chain(row: usize) -> u32 {
    (0..3).fold(0xff_ffffu32, |a, k| if row > k { (a & !(0xff << (8 * k))) | ((row - 1 - k) as u32) << (8 * k) } else { a })
}

/// saved node states fe_gdn_rec2 holds for a tree (k-core gdn.rs TREE_SLOTS)
const TREE_SLOTS: usize = 8;

/// The GDN recurrence's step order for a verify tree (fe_gdn_rec2): depth first from the root; right after a
/// node its leaf children (output only, from the node's state, which stays), then its inner children; a node
/// with several inner children has its state saved (a stack slot per nesting level) before the first descends
/// and restored before each other. Per step: row | leaf << 5 | (save slot + 1) << 6 | (restore slot + 1) << 10.
pub fn tree_steps(par: &[i32]) -> Result<Vec<u32>> {
    let n = par.len();
    let mut kids = vec![Vec::new(); n];
    for (r, &p) in par.iter().enumerate().skip(1) {
        kids[p as usize].push(r);
    }
    fn visit(v: usize, kids: &[Vec<usize>], steps: &mut Vec<u32>, depth: usize) -> Result<()> {
        steps.extend(kids[v].iter().filter(|&&c| kids[c].is_empty()).map(|&c| c as u32 | 1 << 5));
        let inner: Vec<usize> = kids[v].iter().copied().filter(|&c| !kids[c].is_empty()).collect();
        let several = inner.len() > 1;
        ensure!(!several || depth < TREE_SLOTS, "tree: branching nested deeper than {}",
            TREE_SLOTS);
        for (i, &c) in inner.iter().enumerate() {
            let slot = (depth as u32 + 1) << if i == 0 { 6 } else { 10 };
            steps.push(c as u32 | if several { slot } else { 0 });
            visit(c, kids, steps, depth + several as usize)?;
        }
        Ok(())
    }
    let mut steps = vec![if kids[0].is_empty() { 1 << 5 } else { 0 }];
    visit(0, &kids, &mut steps, 0)?;
    debug_assert_eq!(steps.len(), n);
    Ok(steps)
}
/// Widest single linear inside a fused prompt projection (q_proj with its gate: 12288).
const MAX_PROJ: usize = 17408;

/// y[M][N] = x[M][K] . W[N][K]^T. `w` = bf16 weights (GEMM), `q` = EXL3 trellis (GEMV for <= X3_ROWS rows).
pub struct Linear {
    pub n: usize,
    pub k: usize,
    pub w: Option<DeviceBuffer<u16>>,
    pub q: Option<fe_exl3::linear::Linear>,
    /// NVFP4/FP8 prompt form (hybrid: prompt rows from the ModelOpt checkpoint)
    pub p: Option<fe_fp4::PromptLinear>,
    pub name: String,
    /// tensor parallel: this rank holds input rows of the weight, its output is a partial to reduce
    pub row_par: bool,
}
impl Linear {
    /// y (row stride ldy) = x (row stride lda) . W^T; with `res` (stride ldy): y = bf16(bf16(x.W) + res),
    /// `res` may equal `y`.
    pub fn forward(&self, c: &LinCtx, y: DevPtr, ldy: usize, x: DevPtr, lda: usize, m: usize, res: Option<DevPtr>)
        -> Result<()> {
        if let (Some(q), Some(x3)) = (&self.q, c.x3) {
            // decode rows (all rows of a small batch): the trellis GEMV, 16 rows a call
            let d = if m <= X3_ROWS { m } else { c.dec.min(m) };
            if d > 0 && lda == self.k && ldy == self.n {
                use fe_exl3::linear::Dtype;
                if let Some(r) = res {
                    ensure!(r == y, "{}: residual must be in place", self.name);
                }
                // residual: fused in the GEMV epilogue, y = bf16(f32(y) + x.W)
                let yd = if res.is_some() { Dtype::Bf16Add } else { Dtype::Bf16 };
                let mut r = 0;
                while r < d {
                    let n = (d - r).min(X3_PASS);
                    q.forward(&x3.kern, c.s, x.add::<u16>(r * lda), Dtype::Bf16, y.add::<u16>(r * ldy), yd, n as u32, &x3.ws)?;
                    r += n;
                }
                if d == m {
                    return Ok(());
                }
                // the prompt rows after them
                let c2 = LinCtx { dec: 0, ..*c };
                return self.forward(&c2, y.add::<u16>(d * ldy), ldy, x.add::<u16>(d * lda), lda, m - d,
                    res.map(|p| p.add::<u16>(d * ldy)));
            }
        }
        if let (Some(p), Some((fk, fw))) = (&self.p, c.fp4) {
            return p.forward(fk, c.s, fw, y, ldy, x, lda, m, res);
        }
        if self.w.is_none() && self.q.is_some() && m > X3_ROWS {
            // no prompt weights: every row through the trellis GEMV
            let c2 = LinCtx { dec: m, ..*c };
            return self.forward(&c2, y, ldy, x, lda, m, res);
        }
        let w = self.w.as_ref().with_context(|| format!("{}: no bf16 weights for {m} rows", self.name))?;
        // decode / verify rows (all of a <= 16-row call, else the leading c.dec): the GEMV in passes of up to 8 rows,
        // so a row's bits do not depend on how many rows the call has (through the GEMM they would differ from
        // 1-row decode); prompt rows after them take the GEMM
        let d = if m <= X3_ROWS { m } else { c.dec.min(m) };
        if d > 0 && self.k % 256 == 0 {
            // narrow linears: one block a column fills the GPU, where one warp a column ran 6-12 blocks. Any plain bf16
            // linear of <= 512 columns takes it (today only the fused GDN in_proj_a|b: 96 columns, 48 a rank); the
            // k % 64 test is implied by the k % 256 above. Chosen by the shape alone, so decode and verify rows match.
            let (gemv, grid) = if self.n <= 512 && self.k % 64 == 0 {
                (&c.k.gemv_kw, self.n as u32)
            } else {
                (&c.k.gemv, (self.n as u32).div_ceil(8))
            };
            let mut r = 0;
            while r < d {
                let mr = (d - r).min(8);
                unsafe {
                    c.s.launch(gemv, grid, 256, 0, (y.add::<u16>(r * ldy),
                        x.add::<u16>(r * lda), w.ptr(), res.map_or(DevPtr::NULL, |p| p.add::<u16>(r * ldy)), mr as u32,
                        self.n as u32, self.k as u32, lda as u32, ldy as u32))?;
                }
                r += mr;
            }
            if d == m {
                return Ok(());
            }
        }
        let d = if self.k % 256 == 0 { d } else { 0 };
        let (y, x, res, m) = (y.add::<u16>(d * ldy), x.add::<u16>(d * lda), res.map(|p| p.add::<u16>(d * ldy)), m - d);
        let grid = ((m as u32).div_ceil(128), (self.n as u32).div_ceil(128));
        unsafe {
            c.s.launch(&c.k.gemm, grid, 256, GEMM_SMEM, (y, x, w.ptr(), res.unwrap_or(DevPtr::NULL), m as u32,
                self.n as u32, self.k as u32, lda as u32, ldy as u32))?;
        }
        Ok(())
    }
}

/// Several logical outputs of one input (e.g. q|k|v). Each linear writes its own [rows][n] block of
/// `out`; a linear can carry several logical parts side by side (fused weights).
pub struct Proj {
    pub lins: Vec<(Linear, Vec<usize>)>,
    /// prompt rows: one fused NVFP4/FP8 GEMM covering the first `.1` linears
    pub prompt: Option<(fe_fp4::PromptLinear, usize)>,
}
impl Proj {
    pub fn total_n(&self) -> usize {
        self.lins.iter().map(|l| l.0.n).sum()
    }
    /// Returns (pointer, row stride) of every logical part, in order.
    pub fn forward(&self, c: &LinCtx, out: DevPtr, x: DevPtr, lda: usize, m: usize) -> Result<Vec<(DevPtr, usize)>> {
        let mut parts = Vec::new();
        let mut off = 0;
        let mut first = 0;
        if let (Some((p, cover)), Some((fk, fw))) = (&self.prompt, c.fp4) {
            if m > X3_ROWS {
                let d = c.dec.min(m);
                if d < m {
                    p.forward(fk, c.s, fw, out.add::<u16>(d * p.n), p.n, x.add::<u16>(d * lda), lda, m - d, None)?;
                }
                let mut col = 0;
                for (l, widths) in &self.lins[..*cover] {
                    if d > 0 {
                        // decode rows: the linear's own GEMV into scratch, then into the fused layout
                        ensure!(d <= DEC_MAX && l.n <= MAX_PROJ, "{}: {d} decode rows", l.name);
                        let c2 = LinCtx { dec: d, ..*c };
                        l.forward(&c2, c.tmp, l.n, x, lda, d, None)?;
                        unsafe {
                            c.s.launch(&c.k.copy2d, (d as u32, (l.n as u32 / 2).div_ceil(256)), 256, 0,
                                (out.add::<u16>(col), p.n as u32, c.tmp, l.n as u32, l.n as u32))?;
                        }
                    }
                    for &wdt in widths {
                        parts.push((out.add::<u16>(col), p.n));
                        col += wdt;
                    }
                }
                off = m * p.n;
                first = *cover;
            }
        }
        for (l, widths) in &self.lins[first..] {
            let y = out.add::<u16>(off);
            l.forward(c, y, l.n, x, lda, m, None)?;
            let mut col = 0;
            for &wdt in widths {
                parts.push((y.add::<u16>(col), l.n));
                col += wdt;
            }
            off += m * l.n;
        }
        Ok(parts)
    }
}

pub enum Mixer {
    /// Gated DeltaNet. proj parts: qkv (10240), z (6144), a (48), b (48).
    Gdn { proj: Proj, conv_w: DeviceBuffer<u16>, a_log: DeviceBuffer<u16>, dt_bias: DeviceBuffer<u16>,
        norm_w: DeviceBuffer<u16>, out: Linear, idx: usize },
    /// Gated full attention. qkv parts: q|gate (12288, interleaved per head), k (1024), v (1024).
    Attn { qkv: Proj, q_norm: DeviceBuffer<u16>, k_norm: DeviceBuffer<u16>, o: Linear, idx: usize },
}

pub struct Layer {
    pub in_norm: DeviceBuffer<u16>,
    pub post_norm: DeviceBuffer<u16>,
    pub mixer: Mixer,
    /// parts: gate (inter), up (inter)
    pub gate_up: Proj,
    /// prompt-row form of gate_up: bf16 rows interleaved in blocks of 8 for the fused SwiGLU GEMM
    pub swiglu: Option<DeviceBuffer<u16>>,
    /// hybrid prompt rows: NVFP4 gate|up interleaved by 8 for fe-fp4's fused GEMM + SwiGLU + NVFP4 quant of
    /// down's input (`forward_swiglu`, then `down.p.gemm`)
    pub swiglu_fp4: Option<fe_fp4::PromptLinear>,
    pub down: Linear,
}

/// The MTP head (one full-attention layer over [norm(embed(next token)) | norm(final hidden)]).
pub struct MtpW {
    pub pre_e: DeviceBuffer<u16>,
    pub pre_h: DeviceBuffer<u16>,
    /// [5120][10240]: input = [embedding half | hidden half]
    pub fc: Linear,
    pub layer: Layer,
    pub norm: DeviceBuffer<u16>,
    /// draft head: the first `draft_vocab` rows of lm_head (EXL3 strips are 128-column blocks, so a prefix of
    /// the vocabulary is a contiguous view); the target always verifies against the full vocabulary
    pub head: Option<Linear>,
}

/// The token embedding table: on the device, or (small-memory devices, `FE_EMBED_HOST`) in host memory, from
/// where each forward's rows are gathered and uploaded (the same bf16 bits either way).
pub enum Embed {
    Dev(DeviceBuffer<u16>),
    Host(Vec<u8>),
}

pub struct Weights {
    pub embed: Embed,
    pub layers: Vec<Layer>,
    pub norm: DeviceBuffer<u16>,
    pub lm_head: Linear,
    pub mtp: Option<MtpW>,
    pub x3: Option<X3>,
    pub fp4: Option<fe_fp4::Kernels>,
    /// the NVFP4/FP8 prompt weights live in host memory and visit the device a layer at a time
    pub pstream: Option<PromptStream>,
    /// the device's SM count (launch geometry)
    pub sm_count: u32,
}

/// Prompt weights streamed from pinned host memory (small-memory devices; `FE_PROMPT_STREAM`, ring size
/// `FE_PROMPT_RING`, default 2 layers): every layer's NVFP4/FP8 codes and scales are one pinned blob, copied on
/// its own stream into slot `layer % ring` of a device ring just before the layer's prompt GEMMs run, while the
/// layers before it compute. A slot is rewritten only after the layer that used it has finished.
pub struct PromptStream {
    blobs: Vec<fe_cuda::HostBuffer<u8>>,
    ring: DeviceBuffer<u8>,
    slot: usize,
    n_ring: usize,
    copy: Stream,
    up: Vec<fe_cuda::Event>,
    done: Vec<fe_cuda::Event>,
    dry: bool,
}
impl PromptStream {
    /// Bytes in host memory, and the device ring's.
    pub fn bytes(&self) -> (usize, usize) {
        (self.blobs.iter().map(|b| b.len()).sum(), self.ring.len())
    }
    /// Enqueue layer `li`'s upload; stream `s` then waits for it before whatever is launched next.
    fn fetch(&self, li: usize, s: &Stream) -> Result<()> {
        if li >= self.n_ring {
            self.copy.wait(&self.done[li - self.n_ring])?;
        }
        // FE_PROMPT_STREAM_DRY=1 (timing only: wrong results) skips the copies, to see the compute alone
        if !self.dry {
            unsafe {
                self.copy.memcpy_htod(self.ring.ptr().byte_add((li % self.n_ring) * self.slot), self.blobs[li].as_slice())?;
            }
        }
        self.copy.record(&self.up[li])?;
        s.wait(&self.up[li])?;
        Ok(())
    }
    /// Layer `li`'s launches are all enqueued on `s`: its slot may be reused once they finish.
    fn release(&self, li: usize, s: &Stream) -> Result<()> {
        s.record(&self.done[li])?;
        Ok(())
    }
}

/// Load options.
#[derive(Clone, Debug)]
pub struct LoadOpts {
    /// keep bf16 weights (reconstructed from EXL3 if needed) for prompt rows
    pub bf16: bool,
    /// use the EXL3 trellis GEMV for decode rows (EXL3 checkpoints only)
    pub exl3_decode: bool,
    /// load the MTP head when the checkpoint has one
    pub mtp: bool,
    /// hybrid: prompt rows through NVFP4/FP8 weights of this ModelOpt checkpoint (drops the bf16 copies)
    pub prompt_fp4: Option<std::path::PathBuf>,
    /// tensor parallel (rank, world): load this rank's slices (crate::tp)
    pub tp: Option<(usize, usize)>,
    /// keep the embedding table in host memory (Embed::Host)
    pub embed_host: bool,
    /// stream the NVFP4/FP8 prompt weights from host memory (PromptStream) instead of keeping them on the device
    pub prompt_stream: bool,
    /// bf16 copies of the MTP head's weights for its prompt rows (false: they take the trellis GEMV)
    pub mtp_bf16: bool,
}
impl Default for LoadOpts {
    fn default() -> Self {
        LoadOpts { bf16: true, exl3_decode: true, mtp: true, prompt_fp4: None, tp: None, embed_host: false, prompt_stream: false,
            mtp_bf16: true }
    }
}
impl LoadOpts {
    /// FE_BF16=0 drops the bf16 prompt weights (decode-only runs; the default on small-memory devices), FE_EXL3_DECODE=0 forces the bf16 GEMM path,
    /// FE_MTP=0 skips the MTP head, FE_PROMPT=fp4 selects the NVFP4/FP8 prompt GEMMs (checkpoint: FERRITE_NVFP4_DIR).
    pub fn from_env() -> LoadOpts {
        let on = |k: &str| std::env::var(k).map(|v| v != "0").unwrap_or(true);
        let fp4 = std::env::var("FE_PROMPT").map(|v| v == "fp4").unwrap_or(false);
        let path = fe_site::nvfp4_dir();
        let keep_bf16 = std::env::var("FE_BF16").map(|v| v == "1").unwrap_or(false);
        // small-memory devices have no room for bf16 copies: prompt rows then take the trellis GEMV (FE_BF16=1 keeps them)
        let bf16 = std::env::var("FE_BF16").map(|v| v != "0").unwrap_or(!crate::lowmem::on());
        LoadOpts { bf16: (bf16 && !fp4) || keep_bf16, exl3_decode: on("FE_EXL3_DECODE"), mtp: on("FE_MTP"),
            prompt_fp4: fp4.then_some(path), tp: crate::tp::from_env(),
            embed_host: crate::lowmem::flag("FE_EMBED_HOST"), prompt_stream: fp4 && crate::lowmem::flag("FE_PROMPT_STREAM"),
            mtp_bf16: std::env::var("FE_MTP_BF16").map(|v| v != "0").unwrap_or(!crate::lowmem::on()) }
    }
}

struct Source {
    st: SafeTensors,
    exl3: Option<(fe_exl3::st::Checkpoint, fe_exl3::gpu::Kernels, Stream, DeviceBuffer<f32>)>,
    o: LoadOpts,
    blocks: u32,
    tp: Option<(usize, usize)>,
}

impl Source {
    /// A plain tensor as bf16, this rank's rows of it under TP (crate::tp::plain_rows).
    fn plain_bf16(&mut self, name: &str) -> Result<DeviceBuffer<u16>> {
        match self.tp {
            Some((rank, world)) => {
                let rows = self.st.get(name)?.shape[0];
                let v = crate::tp::rows_bf16(&self.st, name, &crate::tp::plain_rows(name, rows, rank, world))?;
                Ok(DeviceBuffer::from_slice(&v)?)
            }
            None => self.st.upload_bf16(name),
        }
    }

    /// Rows (output features) of a plain linear part this rank keeps.
    fn part_rows(&self, name: &str) -> Result<usize> {
        let rows = self.st.get(name)?.shape[0];
        Ok(match self.tp {
            Some((rank, world)) => crate::tp::plain_rows(name, rows, rank, world).iter().map(|r| r.1 - r.0).sum(),
            None => rows,
        })
    }

    /// A plain (BF16 or F16) tensor uploaded as bf16 into `dst`.
    fn plain_to(&mut self, name: &str, dst: DevPtr) -> Result<()> {
        match self.st.get(name)?.dtype.as_str() {
            "BF16" => {
                self.st.upload_to(name, dst)?;
            }
            "F16" => {
                let raw = self.st.read_host(name)?;
                let v: Vec<u16> = raw.chunks_exact(2).map(|c| bf16_bits(f16_to_f32(u16::from_le_bytes([c[0], c[1]])))).collect();
                memcpy_htod(dst, &v)?;
            }
            d => anyhow::bail!("{name}: unsupported dtype {d}"),
        }
        Ok(())
    }

    fn is_exl3(&self, base: &str) -> bool {
        self.exl3.as_ref().is_some_and(|e| e.0.has(&format!("{base}.trellis")))
    }

    /// One quantized or plain linear `base` with input width k.
    fn single(&mut self, base: &str, k: usize) -> Result<Linear> {
        if self.is_exl3(base) {
            let (ck, kern, s, scratch) = self.exl3.as_ref().unwrap();
            if let Some((rank, world)) = self.tp {
                use fe_exl3::tp::Split;
                let t = fe_exl3::Exl3Tensor::load(ck, base)?;
                let split = fe_exl3::tp::qwen35_split(base, t.k, t.n, rank, world);
                let row_par = matches!(split, Split::Rows(..));
                let t = match &split {
                    Split::Cols(r) => t.slice_cols(r)?,
                    Split::Rows(a, b) => t.slice_rows(*a, *b)?,
                    Split::Replicate => t,
                };
                ensure!(t.k == k, "{base}: rank slice k {} != {k}", t.k);
                let q = if self.o.exl3_decode {
                    Some(fe_exl3::linear::Linear::new(kern, fe_exl3::gpu::DeviceExl3::upload_strips(&t)?, self.blocks)?)
                } else {
                    None
                };
                let w = if self.o.bf16 || q.is_none() {
                    Some(kern.tensor_bf16(s, &t, fe_exl3::cpu::Layout::NK, scratch)?)
                } else {
                    None
                };
                return Ok(Linear { n: t.n, k, w, q, p: None, name: base.to_string(), row_par });
            }
            let q = if self.o.exl3_decode { Some(fe_exl3::linear::Linear::load(kern, ck, base, self.blocks)?) } else { None };
            let (w, n) = if self.o.bf16 || q.is_none() {
                let (w, kk, n) = kern.load_bf16(s, ck, base, fe_exl3::cpu::Layout::NK, scratch)?;
                ensure!(kk == k, "{base}: k {kk} != {k}");
                (Some(w), n)
            } else {
                (None, q.as_ref().unwrap().n as usize)
            };
            return Ok(Linear { n, k, w, q, p: None, name: base.to_string(), row_par: false });
        }
        self.fused(&[base.to_string()], k)
    }

    /// Plain (non-trellis) linears stacked along the output dim into one bf16 weight.
    fn fused(&mut self, parts: &[String], k: usize) -> Result<Linear> {
        let mut n_total = 0;
        for p in parts {
            let t = self.st.get(&format!("{p}.weight"))?;
            ensure!(t.shape.len() == 2 && t.shape[1] == k, "{p}: shape {:?}, want [n][{k}]", t.shape);
            n_total += self.part_rows(&format!("{p}.weight"))?;
        }
        let w = DeviceBuffer::<u16>::alloc(n_total * k)?;
        let mut row = 0;
        for p in parts {
            let name = format!("{p}.weight");
            let n = self.part_rows(&name)?;
            if let Some((rank, world)) = self.tp {
                let rows = self.st.get(&name)?.shape[0];
                let v = crate::tp::rows_bf16(&self.st, &name, &crate::tp::plain_rows(&name, rows, rank, world))?;
                ensure!(v.len() == n * k, "{name}: rank slice");
                fe_cuda::memcpy_htod(w.at(row * k), &v)?;
            } else {
                self.plain_to(&name, w.at(row * k))?;
            }
            row += n;
        }
        Ok(Linear { n: n_total, k, w: Some(w), q: None, p: None, name: parts.join("+"), row_par: false })
    }

    /// A projection with logical parts `parts`: plain checkpoints fuse everything into one GEMM, EXL3 keeps one
    /// linear per quantized tensor and fuses runs of plain ones (e.g. GDN in_proj_a + in_proj_b).
    fn proj(&mut self, parts: &[String], k: usize) -> Result<Proj> {
        let mut lins = Vec::new();
        let mut run: Vec<String> = Vec::new();
        let flush = |src: &mut Source, run: &mut Vec<String>, lins: &mut Vec<(Linear, Vec<usize>)>| -> Result<()> {
            if !run.is_empty() {
                let mut widths = Vec::new();
                for p in run.iter() {
                    widths.push(src.part_rows(&format!("{p}.weight"))?);
                }
                lins.push((src.fused(run, k)?, widths));
                run.clear();
            }
            Ok(())
        };
        for p in parts {
            if self.is_exl3(p) {
                flush(self, &mut run, &mut lins)?;
                let l = self.single(p, k)?;
                let n = l.n;
                lins.push((l, vec![n]));
            } else {
                run.push(p.clone());
            }
        }
        flush(self, &mut run, &mut lins)?;
        Ok(Proj { lins, prompt: None })
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

impl Weights {
    /// Load a checkpoint: plain BF16 HF layout (`model.layers.N...` or `model.language_model.layers.N...`),
    /// or EXL3 (trellis GEMV for decode, plus bf16 reconstructed on the GPU for prompt rows).
    pub fn load(cfg: &Config, dir: &Path, o: &LoadOpts, sm_count: u32) -> Result<Weights> {
        let st = SafeTensors::open_dir(dir)?;
        let is_exl3 = st.tensors.keys().any(|k| k.ends_with(".trellis"));
        let exl3 = if is_exl3 {
            let ck = fe_exl3::st::Checkpoint::open(dir)?;
            let kern = fe_exl3::gpu::Kernels::load()?;
            Some((ck, kern, Stream::new()?, DeviceBuffer::<f32>::alloc(64 << 20)?))
        } else {
            None
        };
        eprintln!("ferrite: {} checkpoint (bf16 prompt weights: {}, exl3 decode: {})",
            if is_exl3 { "EXL3" } else { "BF16" }, o.bf16, is_exl3 && o.exl3_decode);
        let mut src = Source { st, exl3, o: o.clone(), blocks: fe_exl3::linear::default_blocks(sm_count), tp: o.tp };
        let pfx = if src.st.has("model.language_model.embed_tokens.weight") { "model.language_model" } else { "model" };
        let h = cfg.hidden;
        let t0 = std::time::Instant::now();
        let embed = if o.embed_host {
            let name = format!("{pfx}.embed_tokens.weight");
            ensure!(src.st.get(&name)?.dtype == "BF16", "{name}: host embedding needs a BF16 table");
            eprintln!("ferrite: embedding table in host memory");
            Embed::Host(src.st.read_host(&name)?)
        } else {
            Embed::Dev(src.st.upload_bf16(&format!("{pfx}.embed_tokens.weight"))?)
        };
        let mut layers: Vec<Layer> = Vec::new();
        let (mut gi, mut ai) = (0, 0);
        for l in 0..cfg.n_layers {
            let p = format!("{pfx}.layers.{l}");
            let names = |v: &[&str]| -> Vec<String> { v.iter().map(|n| format!("{p}.{n}")).collect() };
            let mixer = if cfg.full_attn[l] {
                let m = Mixer::Attn {
                    qkv: src.proj(&names(&["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]), h)?,
                    q_norm: src.st.upload_bf16(&format!("{p}.self_attn.q_norm.weight"))?,
                    k_norm: src.st.upload_bf16(&format!("{p}.self_attn.k_norm.weight"))?,
                    o: src.single(&format!("{p}.self_attn.o_proj"), cfg.n_q * cfg.head_dim)?,
                    idx: ai,
                };
                ai += 1;
                m
            } else {
                let m = Mixer::Gdn {
                    proj: src.proj(&names(&["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
                        "linear_attn.in_proj_b"]), h)?,
                    conv_w: src.plain_bf16(&format!("{p}.linear_attn.conv1d.weight"))?,
                    a_log: src.plain_bf16(&format!("{p}.linear_attn.A_log"))?,
                    dt_bias: src.plain_bf16(&format!("{p}.linear_attn.dt_bias"))?,
                    norm_w: src.st.upload_bf16(&format!("{p}.linear_attn.norm.weight"))?,
                    out: src.single(&format!("{p}.linear_attn.out_proj"), cfg.lin_v_heads * cfg.lin_dv)?,
                    idx: gi,
                };
                gi += 1;
                m
            };
            layers.push(Layer {
                in_norm: src.st.upload_bf16(&format!("{p}.input_layernorm.weight"))?,
                post_norm: src.st.upload_bf16(&format!("{p}.post_attention_layernorm.weight"))?,
                mixer,
                gate_up: src.proj(&names(&["mlp.gate_proj", "mlp.up_proj"]), h)?,
                swiglu: None,
                swiglu_fp4: None,
                down: src.single(&format!("{p}.mlp.down_proj"), cfg.inter)?,
            });
            if l % 16 == 15 {
                eprintln!("  loaded {} layers ({:.1}s)", l + 1, t0.elapsed().as_secs_f32());
            }
        }
        crate::lowmem::trace("the layers");
        let norm = src.st.upload_bf16(&format!("{pfx}.norm.weight"))?;
        let mut lm_head = src.single("lm_head", h)?;
        crate::lowmem::trace("lm_head");
        // the ModelOpt checkpoint keeps MTP in bf16 (excluded from quantization): its prompt rows use bf16 copies
        // (small-memory devices: none, the head's prompt rows take the trellis GEMV)
        src.o.bf16 = o.mtp_bf16;
        let mtp = if o.mtp && src.st.has("mtp.norm.weight") {
            let p = "mtp.layers.0";
            let names = |v: &[&str]| -> Vec<String> { v.iter().map(|n| format!("{p}.{n}")).collect() };
            let layer = Layer {
                in_norm: src.st.upload_bf16(&format!("{p}.input_layernorm.weight"))?,
                post_norm: src.st.upload_bf16(&format!("{p}.post_attention_layernorm.weight"))?,
                mixer: Mixer::Attn {
                    qkv: src.proj(&names(&["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]), h)?,
                    q_norm: src.st.upload_bf16(&format!("{p}.self_attn.q_norm.weight"))?,
                    k_norm: src.st.upload_bf16(&format!("{p}.self_attn.k_norm.weight"))?,
                    o: src.single(&format!("{p}.self_attn.o_proj"), cfg.n_q * cfg.head_dim)?,
                    idx: ai,
                },
                gate_up: src.proj(&names(&["mlp.gate_proj", "mlp.up_proj"]), h)?,
                swiglu: None,
                swiglu_fp4: None,
                down: src.single(&format!("{p}.mlp.down_proj"), cfg.inter)?,
            };
            eprintln!("ferrite: MTP head loaded");
            Some(MtpW {
                pre_e: src.st.upload_bf16("mtp.pre_fc_norm_embedding.weight")?,
                pre_h: src.st.upload_bf16("mtp.pre_fc_norm_hidden.weight")?,
                fc: src.single("mtp.fc", 2 * h)?,
                layer,
                norm: src.st.upload_bf16("mtp.norm.weight")?,
                head: None,
            })
        } else {
            None
        };
        // TP: the MTP draft head is a replicated slice of lm_head's first ids (lm_head itself is split by
        // vocabulary), loaded the same way on both ranks so both draft the same tokens
        let tp_draft = match (&mtp, o.tp, src.exl3.as_ref()) {
            (Some(_), Some(_), Some((ck, kern, _, _))) if o.exl3_decode => {
                let dv = std::env::var("FE_DRAFT_VOCAB").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| v > 0 && v % 128 == 0)
                    .unwrap_or(65536);
                let q = fe_exl3::linear::Linear::load_cols(kern, ck, "lm_head", &[(0, dv)], src.blocks)?;
                eprintln!("ferrite: MTP draft head (TP, replicated) over the first {dv} vocabulary ids");
                Some(Linear { n: dv, k: h, w: None, q: Some(q), p: None, name: "draft_head".into(), row_par: false })
            }
            _ => None,
        };
        crate::lowmem::trace("the MTP head");
        let x3 = match src.exl3.take() {
            Some((_, kern, _, _)) if o.exl3_decode => {
                let mut all = x3_linears(&layers, &lm_head);
                if let Some(m) = &mtp {
                    all.extend(x3_linears(std::slice::from_ref(&m.layer), &m.fc));
                }
                if let Some(d) = &tp_draft {
                    all.extend(d.q.as_ref());
                }
                let ws = fe_exl3::linear::Workspace::new(&all, X3_PASS as u32)?;
                Some(X3 { kern, ws })
            }
            _ => None,
        };
        let mut mtp = mtp;
        if let (Some(m), Some(d)) = (mtp.as_mut(), tp_draft) {
            m.head = Some(d);
        } else if let (Some(m), Some(q), Some(x3)) = (mtp.as_mut(), lm_head.q.as_ref(), x3.as_ref()) {
            let dv = std::env::var("FE_DRAFT_VOCAB").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(65536);
            if dv > 0 && dv % 128 == 0 && dv < q.n && q.view.strips {
                let mut view = q.view;
                view.n = dv;
                let dq = fe_exl3::linear::Linear::from_view(&x3.kern, "draft_head", view, q.plan)?;
                m.head = Some(Linear { n: dv as usize, k: h, w: None, q: Some(dq), p: None, name: "draft_head".into(),
                    row_par: false });
                eprintln!("ferrite: MTP draft head over the first {dv} vocabulary ids");
            }
        }
        crate::lowmem::trace("the GEMV workspace (load scratch freed)");
        let mut pstream = None;
        let fp4 = match &o.prompt_fp4 {
            Some(dir) => {
                let ck = fe_fp4::Checkpoint::open(dir)?;
                let mut fk = fe_fp4::Kernels::load()?;
                // FP8 GEMMs on the block-scaled tensor-core form (the same bits, fp4-check): on by default on compute
                // capability 12.0 (RTX 50), where it runs at twice the plain form's rate; FE_FP8_BS overrides
                if std::env::var("FE_FP8_BS").is_err() {
                    fk.fp8_bs = crate::lowmem::compute_capability() == 120;
                }
                let t = std::time::Instant::now();
                ensure!(!(o.prompt_stream && o.tp.is_some()), "FE_PROMPT_STREAM: one GPU only");
                pstream = attach_fp4(&ck, pfx, cfg, &mut layers, &mut lm_head, o.tp, o.prompt_stream)?;
                eprintln!("ferrite: NVFP4/FP8 prompt weights from {dir:?} ({:.1}s)", t.elapsed().as_secs_f32());
                if let Some(ps) = &pstream {
                    let (hb, rb) = ps.bytes();
                    eprintln!("ferrite: prompt weights streamed from host memory ({:.1} GB pinned, a {}-layer ring of {:.2} GB on the device)",
                        hb as f64 / 1e9, ps.n_ring, rb as f64 / 1e9);
                }
                Some(fk)
            }
            None => None,
        };
        crate::lowmem::trace("the prompt weights");
        Ok(Weights { embed, layers, norm, lm_head, mtp, x3, fp4, pstream, sm_count })
    }
}

/// Hybrid prompt path: NVFP4/FP8 linears from the ModelOpt checkpoint for every layer (fused per projection).
/// `tp`: (rank, world): each rank loads its slice of every weight (crate::tp::fp4_slice). `stream`: the weights stay
/// in host memory (the returned PromptStream) instead of on the device.
fn attach_fp4(ck: &fe_fp4::Checkpoint, pfx: &str, cfg: &Config, layers: &mut [Layer], lm_head: &mut Linear,
    tp: Option<(usize, usize)>, stream: bool) -> Result<Option<PromptStream>> {
    use fe_fp4::{HostLinear, PromptLinear};
    // FE_FP4_ONLY=qkv,o,gdn,out,mlp,down,lm limits which projections use it (debugging; needs FE_BF16=1); under TP
    // lm_head stays on the trellis head (logits rows always take it), and a streamed set has no head either
    let only = std::env::var("FE_FP4_ONLY").unwrap_or_else(|_| "qkv,o,gdn,out,mlp,down,lm".into());
    let on = |k: &str| only.split(',').any(|x| x == k) && !((tp.is_some() || stream) && k == "lm");
    let slices = |names: &[&str]| -> Result<Option<Vec<fe_fp4::Slice>>> {
        tp.map(|(r, w)| names.iter().map(|n| crate::tp::fp4_slice(ck, n, r, w)).collect()).transpose()
    };
    // a layer's streamed linears in creation order (mixer projection, mixer output, gate|up, down): codes, scales
    let pending: std::cell::RefCell<Vec<(Vec<u8>, Vec<u8>)>> = Default::default();
    let fin = |h: HostLinear| -> Result<PromptLinear> {
        if stream {
            let (lin, codes, sf) = h.place()?;
            pending.borrow_mut().push((codes, sf));
            Ok(lin)
        } else {
            h.upload()
        }
    };
    let load = |names: &[&str]| -> Result<PromptLinear> { fin(HostLinear::load_sliced(ck, names, false, slices(names)?.as_deref())?) };
    let align = |x: usize| x.div_ceil(256) * 256;
    let mut blobs = Vec::new();
    let mut offs: Vec<Vec<(usize, usize)>> = Vec::new();
    for (l, layer) in layers.iter_mut().enumerate() {
        let p = format!("{pfx}.layers.{l}");
        let nm = |n: &str| format!("{p}.{n}");
        match &mut layer.mixer {
            Mixer::Attn { qkv, o, .. } => {
                let parts = [nm("self_attn.q_proj"), nm("self_attn.k_proj"), nm("self_attn.v_proj")];
                let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
                ensure!(qkv.lins.len() == 3, "layer {l}: q/k/v linears");
                if on("qkv") {
                    qkv.prompt = Some((load(&refs)?, 3));
                }
                if on("o") {
                    o.p = Some(load(&[nm("self_attn.o_proj").as_str()])?);
                }
            }
            Mixer::Gdn { proj, out, .. } => {
                let parts = [nm("linear_attn.in_proj_qkv"), nm("linear_attn.in_proj_z")];
                let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
                ensure!(proj.lins.len() == 3, "layer {l}: GDN linears (qkv, z, a|b)");
                if on("gdn") {
                    proj.prompt = Some((load(&refs)?, 2));
                }
                if on("out") {
                    out.p = Some(load(&[nm("linear_attn.out_proj").as_str()])?);
                }
            }
        }
        let parts = [nm("mlp.gate_proj"), nm("mlp.up_proj")];
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        if on("mlp") {
            if std::env::var("FE_FP4_SWIGLU").map(|v| v != "0").unwrap_or(true) {
                layer.swiglu_fp4 = Some(fin(HostLinear::load_sliced(ck, &refs, false, slices(&refs)?.as_deref())?.into_swiglu()?)?);
            } else {
                layer.gate_up.prompt = Some((load(&refs)?, 2));
            }
        }
        if on("down") {
            layer.down.p = Some(load(&[nm("mlp.down_proj").as_str()])?);
        }
        if stream {
            // the layer's blob: every linear's codes then scales, each piece on a 256-byte boundary
            let pend = std::mem::take(&mut *pending.borrow_mut());
            let (mut total, mut lo) = (0usize, Vec::new());
            for (c, sf) in &pend {
                let ow = total;
                let os = ow + align(c.len());
                total = os + align(sf.len());
                lo.push((ow, os));
            }
            let mut blob = fe_cuda::HostBuffer::<u8>::alloc(total)?;
            let dst = blob.as_mut_slice();
            for ((c, sf), &(ow, os)) in pend.iter().zip(&lo) {
                dst[ow..ow + c.len()].copy_from_slice(c);
                dst[os..os + sf.len()].copy_from_slice(sf);
            }
            blobs.push(blob);
            offs.push(lo);
        }
    }
    let _ = cfg;
    if on("lm") {
        lm_head.p = Some(load(&["lm_head"])?);
    }
    if !stream {
        return Ok(None);
    }
    let n_ring = std::env::var("FE_PROMPT_RING").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(2).clamp(1, layers.len());
    let slot = blobs.iter().map(|b| b.len()).max().unwrap_or(0).div_ceil(4096) * 4096;
    let ring = DeviceBuffer::<u8>::alloc(n_ring * slot)?;
    for (l, layer) in layers.iter_mut().enumerate() {
        let base = ring.ptr().byte_add((l % n_ring) * slot);
        let mut it = offs[l].iter();
        let mut set = |lin: Option<&mut PromptLinear>| {
            if let Some(lin) = lin {
                let &(ow, os) = it.next().expect("streamed linear without a blob piece");
                lin.set_views(base.byte_add(ow), base.byte_add(os));
            }
        };
        match &mut layer.mixer {
            Mixer::Attn { qkv, o, .. } => {
                set(qkv.prompt.as_mut().map(|p| &mut p.0));
                set(o.p.as_mut());
            }
            Mixer::Gdn { proj, out, .. } => {
                set(proj.prompt.as_mut().map(|p| &mut p.0));
                set(out.p.as_mut());
            }
        }
        set(layer.swiglu_fp4.as_mut());
        set(layer.gate_up.prompt.as_mut().map(|p| &mut p.0));
        set(layer.down.p.as_mut());
        ensure!(it.next().is_none(), "layer {l}: blob pieces left over");
    }
    let mk = |n: usize| -> Result<Vec<fe_cuda::Event>> { (0..n).map(|_| Ok(fe_cuda::Event::new_no_timing()?)).collect() };
    Ok(Some(PromptStream { up: mk(blobs.len())?, done: mk(blobs.len())?, blobs, ring, slot, n_ring, copy: Stream::new()?,
        dry: std::env::var("FE_PROMPT_STREAM_DRY").is_ok_and(|v| v == "1") }))
}

/// For layers whose gate/up are separate EXL3 linears with bf16 prompt copies: build the interleaved bf16
/// weight of the fused SwiGLU GEMM and drop the two copies (decode keeps using the trellis GEMV).
fn fuse_swiglu(w: &mut Weights, k: &Kernels, s: &Stream, hidden: usize) -> Result<()> {
    let mut layers: Vec<&mut Layer> = w.layers.iter_mut().collect();
    if let Some(m) = w.mtp.as_mut() {
        layers.push(&mut m.layer);
    }
    let mut n = 0;
    for layer in layers {
        let lins = &mut layer.gate_up.lins;
        if lins.len() != 2 || lins.iter().any(|l| l.0.q.is_none() || l.0.w.is_none()) {
            continue;
        }
        let inter = lins[0].0.n;
        let buf = DeviceBuffer::<u16>::alloc(2 * inter * hidden)?;
        unsafe {
            s.launch(&k.interleave_gu, (2 * inter) as u32, 256, 0,
                (buf.ptr(), lins[0].0.w.as_ref().unwrap().ptr(), lins[1].0.w.as_ref().unwrap().ptr(), hidden as u32))?;
        }
        s.sync()?;
        lins[0].0.w = None;
        lins[1].0.w = None;
        layer.swiglu = Some(buf);
        n += 1;
    }
    if n > 0 {
        eprintln!("ferrite: fused SwiGLU prompt GEMM for {n} layers");
    }
    Ok(())
}

/// Every EXL3 GEMV linear of the model (to size the shared workspace).
fn x3_linears<'a>(layers: &'a [Layer], lm_head: &'a Linear) -> Vec<&'a fe_exl3::linear::Linear> {
    let mut lins: Vec<&'a Linear> = Vec::new();
    for layer in layers {
        match &layer.mixer {
            Mixer::Gdn { proj, out, .. } => {
                lins.extend(proj.lins.iter().map(|l| &l.0));
                lins.push(out);
            }
            Mixer::Attn { qkv, o, .. } => {
                lins.extend(qkv.lins.iter().map(|l| &l.0));
                lins.push(o);
            }
        }
        lins.extend(layer.gate_up.lins.iter().map(|l| &l.0));
        lins.push(&layer.down);
    }
    lins.push(lm_head);
    lins.iter().filter_map(|l| l.q.as_ref()).collect()
}

struct Prepared {
    rows: usize,
    nseg: usize,
    o_ids: usize,
    o_rslot: usize,
    o_rpos: usize,
    o_sslot: usize,
    o_srow0: usize,
    o_slen: usize,
    o_lr: usize,
    /// MTP: gather indices of the hidden inputs
    o_src: usize,
    n_logits: usize,
    max_pos: usize,
    /// (slot, first row, rows, first position) per segment
    segs: Vec<(u32, u32, u32, u32)>,
    /// leading decode / verify rows (GDN recurrence / replay, attention numerics as decode)
    dec: usize,
    /// leading rows the linears take as decode rows (EXL3 GEMV): the decode rows and the short prompt segments
    /// right after them (each <= X3_ROWS: a short prompt goes through the GEMV alone or batched, so its numerics
    /// do not depend on what else is in the step; the scheduler puts short prompt segments first)
    lin_dec: usize,
    /// verify trees: per row the cache index (o_cpos), the tree's first cache index (o_tbase) and the
    /// ancestor-or-self mask over the segment's rows (o_tmask); per row its last three ancestors in the segment
    /// (o_par, `ancestors`; chain rows: the previous three), the GDN recurrence's step order (o_tstep, tree_steps)
    /// and per segment a tree flag (o_stree). Chain rows: cpos = rpos, tbase > cpos.
    o_cpos: usize,
    o_tbase: usize,
    o_tmask: usize,
    o_par: usize,
    o_tstep: usize,
    o_stree: usize,
    /// per segment: a verify tree (never the flash path)
    seg_tree: Vec<bool>,
    /// attention's segments: `segs` with every verify tree and decode / verify chain of more than 16 rows cut into
    /// pieces of 16 (fe_attn_vk takes a segment of up to 16 rows; a row's keys and arithmetic do not depend on its
    /// piece, so a long verify's rows stay bitwise their 1-row decodes, which the flash kernel's would not), whether
    /// a piece takes fe_attn_vk, and their first rows / lengths in the metadata
    asegs: Vec<(u32, u32, u32, u32)>,
    aseg_vk: Vec<bool>,
    o_asrow0: usize,
    o_aslen: usize,
}

/// Device scratch for up to `max_rows` rows per forward.
struct Scratch {
    max_rows: usize,
    x: DeviceBuffer<u16>,
    xn: DeviceBuffer<u16>,
    big: DeviceBuffer<u16>,
    conv: DeviceBuffer<u16>,
    mix: DeviceBuffer<u16>,
    act: DeviceBuffer<u16>,
    q: DeviceBuffer<u16>,
    tmp: DeviceBuffer<u16>,
    part_o: DeviceBuffer<f32>,
    part_ml: DeviceBuffer<f32>,
    meta: DeviceBuffer<u32>,
    /// pinned staging for `meta` (uploaded on the engine stream)
    meta_host: fe_cuda::HostBuffer<u32>,
    sel: DeviceBuffer<u16>,
    /// final-normed hidden of every row of the last target forward (the MTP head's input)
    taps: DeviceBuffer<u16>,
    /// MTP head outputs (normed) of the last MTP forward (chained drafts read them)
    mtp_out: DeviceBuffer<u16>,
}
const PART_ROWS: usize = 4096; // rows x splits budget for attention partials

/// Attention splits (the grid's): decode / verify rows (short segments, up to DEC_MAX of them) split their keys by
/// their own count (k-core attn.rs row_splits: 16..=64 parts of ~512 keys, a power of two), so a row sums as 1-row
/// decode does whatever the batch; the grid needs the longest row's count. Prompt batches of short segments use
/// ~256 keys a split within the partials budget.
fn attn_splits(rows: usize, max_pos: usize) -> usize {
    const _: () = assert!(DEC_MAX * 64 <= PART_ROWS);
    if rows <= DEC_MAX {
        max_pos.div_ceil(512).next_power_of_two().clamp(16, 64)
    } else {
        max_pos.div_ceil(256).clamp(1, (PART_ROWS / rows).max(1))
    }
}

/// Where an MTP row's hidden input comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MtpSrc {
    /// row i of the last target forward (its final-normed hidden)
    Tap(usize),
    /// row i of the last MTP forward (its normed output): chained drafts
    Out(usize),
}

/// One MTP row: the head at `pos` of `slot` reads `src` and the embedding of `token` (the token after `pos`).
#[derive(Clone, Copy, Debug)]
pub struct MtpRow {
    pub slot: usize,
    pub pos: usize,
    pub token: u32,
    pub src: MtpSrc,
}

/// device memory below which the captured graphs are dropped before another is captured
const GRAPH_MIN_FREE: usize = 192 << 20;

#[derive(Clone, PartialEq, Eq, Hash)]
struct GraphKey {
    seg_lens: Vec<u32>,
    n_logits: usize,
    hist: bool,
    mtp: Option<bool>, // Some(src is Tap)
    /// attention split count (depends on the longest context in the batch)
    nsplit: usize,
    /// leading decode / verify rows (EXL3 routing, GDN replay segments)
    dec: usize,
    /// which segments are verify trees (attention path, GDN tree kernels)
    trees: Vec<bool>,
}

/// GDN replay (FE_GDN_REPLAY, default on for spec engines; under TP sized and run per rank): the verify segments of a `hist` forward run the
/// recurrence read-only (fe_gdn_rec2's `ro` flags) and save their inputs here; `Engine::commit` replays the kept
/// rows onto the state with the same kernel (bit-identical to the per-row history copy, which writes and copies
/// a full fp32 state per verify row: ~37 MB per GDN layer and round vs ~9 MB).
struct Replay {
    /// per GDN layer: conv output rows [segment][hist_cap][conv_dim] and a|b rows [segment][hist_cap][2 * v heads],
    /// by segment index (a captured graph replays its copies for whatever slots the segments belong to)
    conv: Vec<DeviceBuffer<u16>>,
    ab: Vec<DeviceBuffer<u16>>,
    /// per-segment read-only flags of the current forward (pinned staging, uploaded on the engine stream)
    ro: DeviceBuffer<u32>,
    ro_host: fe_cuda::HostBuffer<u32>,
    /// (slot, first saved row, rows) of a commit's replay
    seg: DeviceBuffer<u32>,
    /// per GDN layer: [conv, a|b, a_log, dt_bias, state] pointers (fe_gdn_replay)
    tab: DeviceBuffer<u64>,
    seg_host: fe_cuda::HostBuffer<u32>,
    /// per slot: the segment index whose inputs commit must replay (the last forward left the state at the
    /// verify start)
    pending: Vec<Option<usize>>,
}

pub struct Engine {
    pub cfg: Config,
    pub k: Kernels,
    pub w: Weights,
    pub stream: Stream,
    pub n_slots: usize,
    pub max_ctx: usize,
    /// tokens already in each slot's cache
    pub slot_len: Vec<usize>,
    kv: Vec<(DeviceBuffer<u16>, DeviceBuffer<u16>)>,
    /// bits a value of the cache's records (kvq_bits; 0: a bf16 cache)
    kvq: usize,
    /// fe_attn_vk_q's splits a row (FE_VK_SPLITS; default SMs / 4 kv heads, within 16..=64)
    vkq_splits: u32,
    /// FE_FWD_PROF=1: (last mark, per-phase milliseconds) of prompt forwards
    fwd_prof: Option<std::cell::RefCell<(std::time::Instant, Vec<(&'static str, f64)>)>>,
    conv_state: Vec<DeviceBuffer<u16>>,
    rec_state: Vec<DeviceBuffer<f32>>,
    /// per-row state history of the last verify forward (rows 0..hist_cap), per GDN layer; the recurrent one only
    /// without GDN replay (151 MB a row and slot: replay keeps the verify read-only and re-runs the kept rows)
    conv_hist: Vec<DeviceBuffer<u16>>,
    rec_hist: Vec<DeviceBuffer<f32>>,
    /// speculative depth (a verify has up to hist_rows + 1 rows)
    pub hist_rows: usize,
    /// verify rows whose conv history (and replay inputs) are kept: hist_rows + 1 with replay, else hist_rows (the
    /// last row of a chain verify never needs a rollback point)
    hist_cap: usize,
    rope: DeviceBuffer<u16>,
    sc: Scratch,
    /// logits of the last forward: [n_logit_rows][vocab] bf16
    pub logits: DeviceBuffer<u16>,
    max_logit_rows: usize,
    argmax_out: DeviceBuffer<u32>,
    fp4_ws: Option<fe_fp4::Workspace>,
    /// commit_tree: the kept path and the (base, row elements) table of the buffers it gathers (pinned staging +
    /// device copies)
    tree_path_host: fe_cuda::HostBuffer<u32>,
    tree_path: DeviceBuffer<u32>,
    gather_host: fe_cuda::HostBuffer<u64>,
    gather_tab: DeviceBuffer<u64>,
    /// down_proj's NVFP4 input written by the fused SwiGLU GEMM
    fp4_act: Option<fe_fp4::Workspace>,
    /// chunked (TF32 tensor-core) delta rule for GDN prompt rows (FE_GDN_CHUNK=0 keeps fe_gdn_rec2)
    gdn: Option<fe_gdn::Gdn>,
    replay: Option<Replay>,
    /// drafter feature taps: target layer outputs copied per forward into [rows][layers.len() * hidden] bf16
    /// (set_feature_taps; DFlash2 reads layers 5, 19, 33, 47, 61)
    feat: Option<(Vec<usize>, DeviceBuffer<u16>)>,
    /// debug taps (set_mid_taps): every layer's residual after its mixer, [rows][n_layers * hidden] bf16
    mid: Option<DeviceBuffer<u16>>,
    /// prompt cache: the park pool (park_pool / park_save / park_restore), same size on every rank
    park: Option<ParkMem>,
    /// prompt cache block layout on rank 0 (cache_pool): (state blocks, bytes a state block, KV blocks)
    park_layout: Option<(usize, usize, usize)>,
    /// captured graphs for batches made only of short segments (FE_GRAPHS=0 disables)
    graphs: std::collections::HashMap<GraphKey, fe_cuda::Graph>,
    pub use_graphs: bool,
    /// tensor parallel state (crate::tp), attached after construction
    pub tp: Option<crate::tp::Tp>,
    /// Embed::Host: pinned staging of a forward's embedding rows, their device copy [max_rows][hidden] and the
    /// identity row index fe_embed gathers them with
    emb: Option<(fe_cuda::HostBuffer<u16>, DeviceBuffer<u16>, DeviceBuffer<u32>)>,
}

/// The park pool's memory: on the device, or (small-memory devices, `FE_PARK_HOST`) pinned host memory, with the
/// same block layout; entries then move over the bus on the engine stream instead of device to device.
enum ParkMem {
    Dev(DeviceBuffer<u8>),
    Host(fe_cuda::HostBuffer<u8>),
}
impl ParkMem {
    fn len(&self) -> usize {
        match self {
            ParkMem::Dev(b) => b.len(),
            ParkMem::Host(b) => b.len(),
        }
    }
}

pub(crate) fn f16_bits_to_bf16(h: u16) -> u16 {
    bf16_bits(f16_to_f32(h))
}

pub(crate) fn f32_to_bf16_bits(f: f32) -> u16 {
    bf16_bits(f)
}

fn bf16_bits(f: f32) -> u16 {
    let b = f.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

impl Engine {
    /// `hist_rows`: speculative depth (rows of GDN state history kept for rollback; 0 = no speculation).
    pub fn new(cfg: Config, k: Kernels, w: Weights, n_slots: usize, max_ctx: usize, max_rows: usize) -> Result<Engine> {
        Self::with_spec(cfg, k, w, n_slots, max_ctx, max_rows, 0)
    }

    pub fn with_spec(cfg: Config, k: Kernels, w: Weights, n_slots: usize, max_ctx: usize, max_rows: usize,
        hist_rows: usize) -> Result<Engine> {
        cfg.check_supported()?;
        let stream = Stream::new()?;
        let mut w = w;
        if std::env::var("FE_SWIGLU").map(|v| v != "0").unwrap_or(true) {
            fuse_swiglu(&mut w, &k, &stream, cfg.hidden)?;
        }
        let kvq = kvq_bits()?;
        let vkq_splits = std::env::var("FE_VK_SPLITS").ok().and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(w.sm_count / cfg.n_kv as u32).clamp(16, 64);
        ensure!(kvq == 0 || (cfg.n_kv == 4 && cfg.n_q == 24), "FE_KVQ: the quantized K/V cache is built for one GPU (no TP split)");
        if kvq > 0 {
            eprintln!("ferrite: {kvq}-bit K/V cache ({} bytes a token)", (cfg.n_full() + w.mtp.is_some() as usize) * 2 * cfg.n_kv * kvq_rec(kvq));
        }
        let kv_elems = n_slots * max_ctx * if kvq > 0 { cfg.n_kv * kvq_rec(kvq) / 2 } else { cfg.n_kv * cfg.head_dim };
        let mut kv = Vec::new();
        for _ in 0..cfg.n_full() + w.mtp.is_some() as usize {
            kv.push((DeviceBuffer::<u16>::zeroed(kv_elems)?, DeviceBuffer::<u16>::zeroed(kv_elems)?));
        }
        crate::lowmem::trace("the K/V cache");
        let n_gdn = cfg.n_layers - cfg.n_full();
        let cd = cfg.conv_dim();
        let rs = cfg.lin_v_heads * cfg.lin_dk * cfg.lin_dv;
        let mut conv_state = Vec::new();
        let mut rec_state = Vec::new();
        let mut conv_hist = Vec::new();
        let mut rec_hist = Vec::new();
        let use_replay = hist_rows > 0 && n_slots * (hist_rows + 1) <= max_rows
            && std::env::var("FE_GDN_REPLAY").map(|v| v != "0").unwrap_or(true);
        let hist_cap = if use_replay { hist_rows + 1 } else { hist_rows };
        for _ in 0..n_gdn {
            conv_state.push(DeviceBuffer::<u16>::zeroed(n_slots * 3 * cd)?);
            rec_state.push(DeviceBuffer::<f32>::zeroed(n_slots * rs)?);
            if hist_rows > 0 {
                conv_hist.push(DeviceBuffer::<u16>::alloc(n_slots * hist_cap * 3 * cd)?);
                if !use_replay {
                    rec_hist.push(DeviceBuffer::<f32>::alloc(n_slots * hist_rows * rs)?);
                }
            }
        }
        crate::lowmem::trace("the GDN state and history");
        // RoPE table [pos][cos 32 | sin 32] in bf16, computed like HF: f32 inv_freq, f32 angle, bf16 cos/sin.
        let half = cfg.rotary_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / (cfg.rope_theta as f32).powf((2 * i) as f32 / cfg.rotary_dim as f32))
            .collect();
        let mut tab = vec![0u16; n_slots * max_ctx * 64];
        for p in 0..n_slots * max_ctx {
            for i in 0..half {
                let ang = (p as f32 * inv[i]) as f64;
                tab[p * 64 + i] = bf16_bits(ang.cos() as f32);
                tab[p * 64 + 32 + i] = bf16_bits(ang.sin() as f32);
            }
        }
        let rope = DeviceBuffer::from_slice(&tab)?;
        let h = cfg.hidden;
        let big_cols = (2 * cfg.inter).max(cfg.conv_dim() + cfg.lin_v_heads * cfg.lin_dv + 2 * cfg.lin_v_heads)
            .max(cfg.n_q * cfg.head_dim * 2 + 2 * cfg.n_kv * cfg.head_dim);
        let max_logit_rows = max_rows.min(64.max(n_slots));
        let sc = Scratch {
            max_rows,
            x: DeviceBuffer::alloc(max_rows * h)?,
            xn: DeviceBuffer::alloc(max_rows * h)?,
            big: DeviceBuffer::alloc(max_rows * big_cols)?,
            conv: DeviceBuffer::alloc(max_rows * cd)?,
            mix: DeviceBuffer::alloc(max_rows * (cfg.lin_v_heads * cfg.lin_dv).max(cfg.n_q * cfg.head_dim))?,
            act: DeviceBuffer::alloc(max_rows * cfg.inter)?,
            q: DeviceBuffer::alloc(max_rows * cfg.n_q * cfg.head_dim)?,
            tmp: DeviceBuffer::alloc(DEC_MAX * MAX_PROJ)?,
            part_o: DeviceBuffer::alloc(PART_ROWS * cfg.n_kv * 6 * 256)?,
            part_ml: DeviceBuffer::alloc(PART_ROWS * cfg.n_kv * 12)?,
            meta: DeviceBuffer::alloc(15 * max_rows + 64)?,
            meta_host: fe_cuda::HostBuffer::alloc(15 * max_rows + 64)?,
            sel: DeviceBuffer::alloc(max_logit_rows * h)?,
            taps: DeviceBuffer::alloc(max_rows * h)?,
            mtp_out: DeviceBuffer::alloc(max_rows * h)?,
        };
        crate::lowmem::trace("the forward scratch");
        let logits = DeviceBuffer::alloc(max_logit_rows * cfg.vocab)?;
        let argmax_out = DeviceBuffer::alloc(max_logit_rows)?;
        let fp4_ws = if w.fp4.is_some() { Some(fe_fp4::Workspace::new(max_rows, cfg.inter.max(2 * cfg.hidden))?) } else { None };
        let gdn = if std::env::var("FE_GDN_CHUNK").map(|v| v != "0").unwrap_or(true) { Some(fe_gdn::Gdn::load()?) } else { None };
        let fp4_act = if w.layers.iter().any(|l| l.swiglu_fp4.is_some()) {
            Some(fe_fp4::Workspace::new(max_rows, cfg.inter)?)
        } else {
            None
        };
        let replay = if use_replay {
            let (mut conv, mut ab) = (Vec::new(), Vec::new());
            for _ in 0..n_gdn {
                conv.push(DeviceBuffer::<u16>::alloc(n_slots * hist_cap * cd)?);
                ab.push(DeviceBuffer::<u16>::alloc(n_slots * hist_cap * 2 * cfg.lin_v_heads)?);
            }
            let nseg = n_slots.max(64);
            let mut tab = vec![0u64; 5 * n_gdn];
            for layer in &w.layers {
                if let Mixer::Gdn { a_log, dt_bias, idx, .. } = &layer.mixer {
                    tab[5 * idx..5 * idx + 5].copy_from_slice(&[conv[*idx].ptr().0, ab[*idx].ptr().0, a_log.ptr().0,
                        dt_bias.ptr().0, rec_state[*idx].ptr().0]);
                }
            }
            Some(Replay { conv, ab, tab: DeviceBuffer::from_slice(&tab)?, ro: DeviceBuffer::zeroed(nseg)?,
                ro_host: fe_cuda::HostBuffer::alloc(nseg)?,
                seg: DeviceBuffer::zeroed(3)?, seg_host: fe_cuda::HostBuffer::alloc(3)?, pending: vec![None; n_slots] })
        } else {
            None
        };
        crate::lowmem::trace("logits, prompt workspaces, replay buffers");
        let emb = match &w.embed {
            Embed::Host(_) => Some((fe_cuda::HostBuffer::alloc(max_rows * h)?, DeviceBuffer::alloc(max_rows * h)?,
                DeviceBuffer::from_slice(&(0..max_rows as u32).collect::<Vec<_>>())?)),
            Embed::Dev(_) => None,
        };
        Ok(Engine { emb, slot_len: vec![0; n_slots], cfg, k, w, stream, n_slots, max_ctx, kv, kvq, vkq_splits,
            fwd_prof: std::env::var("FE_FWD_PROF").is_ok_and(|v| v == "1").then(|| std::cell::RefCell::new((std::time::Instant::now(), Vec::new()))),
            conv_state, rec_state, conv_hist,
            rec_hist, hist_rows, hist_cap, rope, sc, logits, max_logit_rows, argmax_out, fp4_ws,
            tree_path_host: fe_cuda::HostBuffer::alloc(32)?, tree_path: DeviceBuffer::alloc(32)?,
            gather_host: fe_cuda::HostBuffer::alloc(2 * GATHER_MAX)?, gather_tab: DeviceBuffer::alloc(2 * GATHER_MAX)?, fp4_act, gdn, replay, feat: None, mid: None, park: None, park_layout: None, graphs: Default::default(),
            use_graphs: std::env::var("FE_GRAPHS").map(|v| v != "0").unwrap_or(true), tp: None })
    }

    /// Tokens the K/V cache holds from `slot` to its end. The slots' caches are one buffer, a slot after another, each
    /// `max_ctx` tokens: a sequence may run past its own slot into the slots after it when the scheduler keeps those
    /// empty (its shared context pool), so this is the bound the engine checks.
    pub fn cap(&self, slot: usize) -> usize {
        (self.n_slots - slot.min(self.n_slots)) * self.max_ctx
    }

    /// Tokens of all slots together: what slot 0 holds when it has the pool to itself.
    pub fn pool_ctx(&self) -> usize {
        self.n_slots * self.max_ctx
    }

    /// u16 elements one token takes in a K (or V) cache buffer: its bf16 values, or its 4-bit records.
    fn kv_tok(&self) -> usize {
        if self.kvq > 0 { self.cfg.n_kv * kvq_rec(self.kvq) / 2 } else { self.cfg.n_kv * self.cfg.head_dim }
    }

    /// (max rows a forward, max logit rows): the sizes a crate::tp::Tp is built for.
    pub fn tp_sizes(&self) -> (usize, usize) {
        (self.sc.max_rows, self.max_logit_rows)
    }

    /// Attach the tensor-parallel fabric (weights must have been loaded with LoadOpts::tp, kernels with_tp2).
    pub fn attach_tp(&mut self, tp: crate::tp::Tp) -> Result<()> {
        self.graphs.clear();
        // the chunked GDN prompt kernel for the rank's 8 key / 24 value heads (k-gdn-tp2)
        if self.gdn.is_some() {
            self.gdn = Some(fe_gdn::Gdn::load_tp2()?);
        }
        self.tp = Some(tp);
        Ok(())
    }

    /// Rank 0 with a following rank 1: send the call before running it (crate::follow).
    fn mirror(&mut self, call: impl FnOnce() -> crate::follow::Call) -> Result<()> {
        if let Some(tp) = self.tp.as_mut() {
            if tp.mirror {
                tp.fab.link.send(&call().encode())?;
            }
        }
        Ok(())
    }

    /// Rank 0: send `call` to a following rank 1 (calls outside the Engine, e.g. a TP-split drafter's).
    pub fn mirror_call(&mut self, call: crate::follow::Call) -> Result<()> {
        self.mirror(|| call)
    }

    /// Rank 1 of a TP pair driven by rank 0's scheduler / server: replay its engine calls until it stops.
    /// Returns the number of calls replayed.
    pub fn follow(&mut self) -> Result<u64> {
        self.follow_with(None)
    }

    /// `follow`, with rank 1's half of a TP-split block drafter (its feature taps on): Call::Draft / Call::Absorb
    /// go to it.
    pub fn follow_with(&mut self, mut df: Option<&mut Box<dyn crate::sched::Drafter>>) -> Result<u64> {
        use crate::follow::Call;
        if let Some(d) = df.as_deref() {
            self.set_feature_taps(&d.layers())?;
        }
        let mut n = 0u64;
        // FE_FOLLOW_PROF=1: per call kind, the time waiting for the call (idle in recv) and running it, every 500 calls
        let prof = std::env::var("FE_FOLLOW_PROF").is_ok_and(|v| v == "1");
        let mut pt = [[0f64; 3]; 16]; // [kind] = (calls, idle ms, run ms)
        let mut t_idle = std::time::Instant::now();
        // FE_FOLLOW_SPIN_MS (default 200, 0 = blocking reads): wait for the next call spinning on the link this long,
        // then block. A blocking read pays this core's wake-up from deep idle, ~0.3-0.6 ms on GB10 after a few ms of
        // quiet (a verify forward), and rank 0 then waits that long at the next call's first reduction.
        // FE_FOLLOW_PIN: the CPU taking the link's receive interrupt sleeps too; 1 pins this thread there, 2 keeps it
        // awake with a spinning thread, 3 (default) pins on a big core (cpu_capacity >= 900) and keeps a little one
        // awake, 0 neither (re-checked every 1024 calls). TP2 seeded 1c: blocking 270.1, keeper 271.3, pinned 274.4.
        let spin = std::time::Duration::from_micros((std::env::var("FE_FOLLOW_SPIN_MS").ok()
            .and_then(|v| v.parse::<f64>().ok()).unwrap_or(200.0).max(0.0) * 1e3) as u64);
        let pin: u8 = if spin.is_zero() { 0 } else {
            std::env::var("FE_FOLLOW_PIN").ok().and_then(|v| v.parse().ok()).unwrap_or(3)
        };
        struct Keeper(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Keeper {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let mut keeper: Option<Keeper> = None;
        let mut irq_cpu: Option<usize> = None;
        let (clock, busy) = (std::time::Instant::now(), std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
        loop {
            let b = match self.tp.as_mut() {
                Some(tp) if spin.is_zero() => tp.fab.link.recv()?,
                Some(tp) => tp.fab.link.recv_spin(spin)?,
                None => anyhow::bail!("follow: no TP fabric attached"),
            };
            if keeper.is_some() {
                busy.store(clock.elapsed().as_millis() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            if pin > 0 && (n == 4 || n % 1024 == 1023) {
                let cpu = self.tp.as_ref().and_then(|tp| tp.fab.link.incoming_cpu());
                if let Some(c) = cpu.filter(|&c| irq_cpu != Some(c)) {
                    irq_cpu = Some(c);
                    let big = fe_fabric::boot::cpu_capacity(c).is_none_or(|k| k >= 900);
                    drop(keeper.take());
                    let how = if pin == 1 || (pin == 3 && big) {
                        if fe_fabric::boot::pin_thread(c) { "follower pinned there" } else { "pinning refused" }
                    } else {
                        if pin == 3 {
                            fe_fabric::boot::unpin_thread();
                        }
                        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                        fe_fabric::boot::keep_awake(c, stop.clone(), busy.clone(), clock, spin);
                        keeper = Some(Keeper(stop));
                        "kept awake"
                    };
                    eprintln!("follow: link interrupts on cpu {c} ({how})");
                }
            }
            let call = Call::decode(&b)?;
            let (t_run, kind) = (std::time::Instant::now(), call.kind());
            match call {
                Call::Forward { segs, logit_rows, hist, dec, parents } => {
                    let refs: Vec<(usize, &[u32])> = segs.iter().map(|(s, t)| (*s, t.as_slice())).collect();
                    let prefs: Vec<Option<&[i32]>> = parents.iter().map(|p| p.as_deref()).collect();
                    self.forward_tree(&refs, &prefs, &logit_rows, hist, dec)?;
                }
                Call::Commit { slot, start, n, keep } => self.commit(slot, start, n, keep)?,
                Call::CommitTree { slot, start, n, path, taps } => self.commit_tree(slot, start, n, &path, taps)?,
                Call::Reset { slot } => self.reset_slot(slot)?,
                Call::Mtp { rows, draw } => {
                    self.mtp_forward(&rows, &draw)?;
                }
                Call::Draft { reqs } => match df.as_deref_mut() {
                    Some(d) => d.follow_draft(self, &reqs)?,
                    None => anyhow::bail!("follow: a drafter call without a drafter (FE_DRAFT on rank 1?)"),
                },
                Call::Absorb { items } => match df.as_deref_mut() {
                    Some(d) => {
                        let feats = self.feature_taps().ok_or_else(|| anyhow::anyhow!("follow: no feature taps"))?;
                        let fw = d.layers().len() * self.cfg.hidden;
                        let items: Vec<(usize, DevPtr, usize, usize)> =
                            items.iter().map(|&(slot, r0, rows, pos)| (slot, feats.add::<u16>(r0 * fw), rows, pos)).collect();
                        d.absorb_many(self, &items)?;
                    }
                    None => anyhow::bail!("follow: a drafter call without a drafter"),
                },
                Call::Stop => return Ok(n),
                Call::ParkPool { bytes } => self.park_pool(bytes)?,
                Call::ParkSave { slot, a, b, gdn, off } => self.park_save(slot, a..b, gdn, off)?,
                Call::ParkRestore { slot, a, b, gdn, off, len } => self.park_restore(slot, a..b, gdn, off, len)?,
                Call::CopySlot { src, dst, a, b, gdn, len } => self.copy_slot(src, dst, a..b, gdn, len)?,
                // the drafter's part of a snapshot (Drafter::save_state / restore_state at off +
                // park_gdn_bytes()) goes right after these two, as for Draft / Absorb
                Call::CacheSave { slot, pos, off } => {
                    self.cache_save_at(slot, pos, off)?;
                    if let Some(d) = df.as_deref() {
                        d.save_state(self, slot, pos, self.park_at(off + self.park_gdn_bytes(), d.state_bytes())?)?;
                    }
                }
                Call::CacheRestore { slot, pos, off } => {
                    self.cache_restore_at(slot, pos, off)?;
                    if let Some(d) = df.as_deref_mut() {
                        { let n = d.state_bytes(); d.restore_state(self, slot, pos, self.park_at(off + self.park_gdn_bytes(), n)?)?; }
                    }
                }
                Call::KvOut { slot, rows, offs } => self.kv_blocks_copy(slot, rows, &offs, true)?,
                Call::KvIn { slot, rows, offs } => self.kv_in_at(slot, rows, &offs)?,
            }
            n += 1;
            if prof {
                let now = std::time::Instant::now();
                let p = &mut pt[kind];
                p[0] += 1.0;
                p[1] += 1e3 * (t_run - t_idle).as_secs_f64();
                p[2] += 1e3 * (now - t_run).as_secs_f64();
                t_idle = now;
                if n % 500 == 0 {
                    let names = ["forward", "commit", "commit_tree", "reset", "mtp", "draft", "absorb", "stop", "park_pool", "park_save",
                        "park_restore", "copy_slot", "cache_save", "cache_restore", "kv_out", "kv_in"];
                    let line: Vec<String> = pt.iter().zip(names).filter(|(p, _)| p[0] > 0.0)
                        .map(|(p, nm)| format!("{nm} {} idle {:.3} run {:.3}", p[0], p[1] / p[0], p[2] / p[0])).collect();
                    eprintln!("follow: {n} calls, ms a call: {}", line.join(" | "));
                    pt = [[0f64; 3]; 16];
                }
            }
        }
    }

    /// x[r0..r0 + rows] += input . W^T for one of a layer's residual linears; under TP the rank's partial goes
    /// through the fabric reduction (which adds the residual). The first `lc.dec` rows are decode / verify rows:
    /// under TP they take the f32 GEMV partial + RoCE reduction in 16-row passes, as in a pure decode batch,
    /// so their bits do not depend on what else is in the batch.
    fn resid_linear(&self, lc: &LinCtx, lin: &Linear, input: DevPtr, k_in: usize, rows: usize, r0: usize) -> Result<()> {
        let h = self.cfg.hidden;
        let x = self.sc.x.ptr().add::<u16>(r0 * h);
        match &self.tp {
            Some(tp) if lin.row_par => {
                let mut r = 0;
                if let (Some(q), Some(x3)) = (&lin.q, lc.x3) {
                    use fe_exl3::linear::Dtype;
                    let d = if rows <= X3_ROWS { rows } else { lc.dec.min(rows) };
                    // bf16 partials: X3_PASS rows a pass (the RoCE slot holds them, tp.rs); f32 partials
                    // ([X3_ROWS][hidden]) keep X3_ROWS rows a pass
                    let pass = if tp.part_bf16_dec { X3_PASS } else { X3_ROWS };
                    while r < d {
                        let n = (d - r).min(pass);
                        let xr = x.add::<u16>(r * h);
                        if tp.part_bf16_dec {
                            q.forward(&x3.kern, lc.s, input.add::<u16>(r * k_in), Dtype::Bf16, tp.part_bf16.ptr(),
                                Dtype::Bf16, n as u32, &x3.ws)?;
                            tp.fab.reduce_bf16_to_bf16(lc.s, tp.part_bf16.ptr(), xr, Some(xr), n * h)?;
                        } else {
                            q.forward(&x3.kern, lc.s, input.add::<u16>(r * k_in), Dtype::Bf16, tp.part.ptr(), Dtype::F32,
                                n as u32, &x3.ws)?;
                            tp.fab.reduce_to_bf16(lc.s, tp.part.ptr(), xr, Some(xr), n * h)?;
                        }
                        r += n;
                    }
                }
                if r == rows {
                    return Ok(());
                }
                let c2 = LinCtx { dec: 0, ..*lc };
                lin.forward(&c2, tp.part_bf16.ptr(), h, input.add::<u16>(r * k_in), k_in, rows - r, None)?;
                tp.fab.reduce_add_bf16(lc.s, tp.part_bf16.ptr(), x.add::<u16>(r * h), (rows - r) * h)
            }
            _ => lin.forward(lc, x, h, input, k_in, rows, Some(x)),
        }
    }

    /// Copy the outputs of target `layers` (in this order) for every row of each later forward into a
    /// [max_rows][layers.len() * hidden] bf16 buffer (`feature_taps`). Drops captured graphs.
    pub fn set_feature_taps(&mut self, layers: &[usize]) -> Result<()> {
        let buf = DeviceBuffer::<u16>::alloc(self.sc.max_rows * layers.len() * self.cfg.hidden)?;
        self.feat = Some((layers.to_vec(), buf));
        self.graphs.clear();
        Ok(())
    }

    /// Debug: also copy every layer's residual after its mixer (before post_norm) per forward (mid_taps).
    pub fn set_mid_taps(&mut self) -> Result<()> {
        self.mid = Some(DeviceBuffer::<u16>::alloc(self.sc.max_rows * self.cfg.n_layers * self.cfg.hidden)?);
        self.graphs.clear();
        Ok(())
    }

    /// The mid-tap buffer of the last forward (row r at r * n_layers * hidden).
    pub fn mid_taps(&self) -> Option<DevPtr> {
        self.mid.as_ref().map(|b| b.ptr())
    }

    /// The feature-tap buffer of the last forward (row r at r * layers.len() * hidden).
    pub fn feature_taps(&self) -> Option<DevPtr> {
        self.feat.as_ref().map(|f| f.1.ptr())
    }

    /// Target lm_head (EXL3 GEMV) over `rows` final-normed hidden rows at `x` (stride hidden) into `logits`, rows
    /// packed with stride lm_head.n (returned). Under TP this is the rank's own vocabulary slice only, with no
    /// gather: on rank 0 that is the first half of the ids (a drafter's prefix head).
    pub fn lm_head_rows(&self, x: DevPtr, rows: usize) -> Result<usize> {
        ensure!(rows <= self.max_logit_rows, "lm_head_rows: {rows} rows (max {})", self.max_logit_rows);
        let lc = LinCtx { k: &self.k, x3: self.w.x3.as_ref(), fp4: self.w.fp4.as_ref().zip(self.fp4_ws.as_ref()),
            s: &self.stream, tmp: self.sc.tmp.ptr(), dec: rows };
        let n = self.w.lm_head.n;
        self.w.lm_head.forward(&lc, self.logits.ptr(), n, x, self.cfg.hidden, rows, None)?;
        Ok(n)
    }

    pub fn has_mtp(&self) -> bool {
        self.w.mtp.is_some()
    }

    /// (rank, world) when tensor-parallel.
    pub fn tp_rank(&self) -> Option<(usize, usize)> {
        self.tp.as_ref().map(|t| (t.rank, t.world))
    }

    // ---- prompt-cache primitives -----------------------------------------------------------------------------
    // A slot's resumable state is its KV tokens (every full-attention layer, + the MTP layer's when loaded) and its
    // recurrent state (every GDN layer's conv window and f32 delta-rule state); the GDN state exists only at the
    // slot's current length (it cannot be rewound), so a resume point needs a saved copy. Copies are plain
    // device memcpys on the engine stream: a restored state is bitwise the state that was saved. Every call is
    // mirrored to a TP follower, whose pool has the same layout (per-rank sizes are equal).
    // Park entry layout at `off`: for each KV layer K[a..b] then V[a..b], then (gdn) for each GDN layer its conv
    // window then its recurrent state.

    /// Bytes one parked KV token takes on this rank.
    pub fn park_kv_bytes(&self) -> usize {
        self.kv.len() * 2 * self.kv_tok() * 2
    }

    /// Bytes of a slot's recurrent state (all GDN layers) on this rank.
    pub fn park_gdn_bytes(&self) -> usize {
        let (cd, rs) = (self.cfg.conv_dim(), self.cfg.lin_v_heads * self.cfg.lin_dk * self.cfg.lin_dv);
        self.conv_state.len() * (3 * cd * 2 + rs * 4)
    }

    /// Bytes of a park entry with `kv_tokens` KV tokens (+ the recurrent state when `gdn`).
    pub fn park_bytes(&self, kv_tokens: usize, gdn: bool) -> usize {
        kv_tokens * self.park_kv_bytes() + if gdn { self.park_gdn_bytes() } else { 0 }
    }

    /// Bytes of the park pool (0 = none).
    pub fn park_capacity(&self) -> usize {
        self.park.as_ref().map_or(0, |p| p.len())
    }

    /// (Re)allocate the park pool with `bytes` on every rank (0 frees it). Earlier entries are lost.
    pub fn park_pool(&mut self, bytes: usize) -> Result<()> {
        self.mirror(|| crate::follow::Call::ParkPool { bytes })?;
        self.park = None;
        if bytes > 0 {
            self.park = Some(if crate::lowmem::flag("FE_PARK_HOST") && self.tp.is_none() {
                ParkMem::Host(fe_cuda::HostBuffer::alloc(bytes)?)
            } else {
                ParkMem::Dev(DeviceBuffer::<u8>::alloc(bytes)?)
            });
        }
        Ok(())
    }

    /// (slot-side address, pool-side offset, bytes) of every piece of a park entry, in layout order.
    fn park_pieces(&self, slot: usize, kv: &std::ops::Range<usize>, gdn: bool) -> Result<Vec<(DevPtr, usize, usize)>> {
        ensure!(slot < self.n_slots && kv.start <= kv.end && kv.end <= self.cap(slot), "park: slot {slot} tokens {kv:?}");
        let tok = self.kv_tok();
        let n = (kv.end - kv.start) * tok * 2;
        let mut v = Vec::new();
        let mut off = 0;
        if n > 0 {
            for (k, vv) in &self.kv {
                for b in [k, vv] {
                    v.push((b.at((slot * self.max_ctx + kv.start) * tok), off, n));
                    off += n;
                }
            }
        }
        if gdn {
            let (cd, rs) = (self.cfg.conv_dim(), self.cfg.lin_v_heads * self.cfg.lin_dk * self.cfg.lin_dv);
            for (c, r) in self.conv_state.iter().zip(&self.rec_state) {
                v.push((c.at(slot * 3 * cd), off, 3 * cd * 2));
                off += 3 * cd * 2;
                v.push((r.at(slot * rs), off, rs * 4));
                off += rs * 4;
            }
        }
        Ok(v)
    }

    /// Device address of pool bytes [off, off + bytes) (a pool on the device only: a drafter's part of a snapshot).
    fn park_at(&self, off: usize, bytes: usize) -> Result<DevPtr> {
        let p = self.park.as_ref().ok_or_else(|| anyhow::anyhow!("park: no pool (park_pool)"))?;
        ensure!(off + bytes <= p.len(), "park: entry [{off}, {}) outside the {}-byte pool", off + bytes, p.len());
        match p {
            ParkMem::Dev(b) => Ok(b.at(off)),
            ParkMem::Host(_) => anyhow::bail!("park: the pool is in host memory (FE_PARK_HOST): no device address"),
        }
    }

    /// Enqueue a copy of `n` bytes between pool offset `off` and device memory at `dev` (`out`: device -> pool).
    fn park_xfer(&self, off: usize, dev: DevPtr, n: usize, out: bool) -> Result<()> {
        let p = self.park.as_ref().ok_or_else(|| anyhow::anyhow!("park: no pool (park_pool)"))?;
        ensure!(off + n <= p.len(), "park: entry [{off}, {}) outside the {}-byte pool", off + n, p.len());
        match p {
            ParkMem::Dev(b) => {
                if out {
                    self.stream.memcpy_dtod(b.at(off), dev, n)?;
                } else {
                    self.stream.memcpy_dtod(dev, b.at(off), n)?;
                }
            }
            ParkMem::Host(h) => {
                // SAFETY: the pool is only touched by these copies, on the engine thread, and every caller
                // synchronises the stream before returning
                let hs = unsafe { std::slice::from_raw_parts_mut((h.as_ptr() as *mut u8).add(off), n) };
                unsafe {
                    if out {
                        self.stream.memcpy_dtoh(hs, dev)?;
                    } else {
                        self.stream.memcpy_htod(dev, hs)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn settled(&self, slot: usize) -> Result<()> {
        ensure!(self.replay.as_ref().is_none_or(|rp| rp.pending[slot].is_none()),
            "park: slot {slot} has an uncommitted verify");
        Ok(())
    }

    // Block layout for the scheduler's pool (docs/prompt-cache.md, park pool): [state blocks][KV blocks of KV_BLOCK tokens].
    // Rank 0 keeps the layout and sends byte offsets, so a follower needs only the pool size.

    /// Allocate the prompt-cache pool: `state_blks` snapshot blocks of park_gdn_bytes() + `state_extra` bytes
    /// (the drafter's part, at park_gdn_bytes() inside the block) and `kv_blks` KV blocks of KV_BLOCK tokens.
    pub fn cache_pool(&mut self, state_blks: usize, state_extra: usize, kv_blks: usize) -> Result<usize> {
        let sb = (self.park_gdn_bytes() + state_extra).next_multiple_of(4096);
        let total = state_blks * sb + kv_blks * KV_BLOCK * self.park_kv_bytes();
        self.park_pool(total)?;
        self.park_layout = Some((state_blks, sb, kv_blks));
        Ok(total)
    }

    /// Byte offset of state block `blk` (cache_pool layout).
    pub fn state_block_off(&self, blk: usize) -> Result<usize> {
        let (ns, sb, _) = self.park_layout.ok_or_else(|| anyhow::anyhow!("cache: no cache_pool"))?;
        ensure!(blk < ns, "cache: state block {blk} of {ns}");
        Ok(blk * sb)
    }

    /// Byte offset of KV block `blk` (cache_pool layout).
    pub fn kv_block_off(&self, blk: usize) -> Result<usize> {
        let (ns, sb, nk) = self.park_layout.ok_or_else(|| anyhow::anyhow!("cache: no cache_pool"))?;
        ensure!(blk < nk, "cache: KV block {blk} of {nk}");
        Ok(ns * sb + blk * KV_BLOCK * self.park_kv_bytes())
    }

    /// Device address of pool byte `off` (the drafter writes its part of a snapshot there).
    pub fn park_ptr(&self, off: usize) -> Result<DevPtr> {
        self.park_at(off, 0)
    }

    /// Snapshot: slot `slot`'s recurrent state, which must be at length `pos`, into state block `blk`. Returns the
    /// byte offset where the drafter's part of the snapshot goes (block + park_gdn_bytes()).
    pub fn cache_save(&mut self, slot: usize, pos: usize, blk: usize) -> Result<usize> {
        let off = self.state_block_off(blk)?;
        self.mirror(|| crate::follow::Call::CacheSave { slot, pos, off })?;
        self.cache_save_at(slot, pos, off)?;
        Ok(off + self.park_gdn_bytes())
    }

    fn cache_save_at(&mut self, slot: usize, pos: usize, off: usize) -> Result<()> {
        self.settled(slot)?;
        ensure!(self.slot_len[slot] == pos, "cache_save: slot {slot} is at {}, not {pos}", self.slot_len[slot]);
        self.park_copy(slot, &(pos..pos), true, off, true)
    }

    /// Resume: slot `slot`'s recurrent state from state block `blk`, its length set to `pos` (the snapshot's).
    /// KV rows [0, pos) must already be the slot's (its own history, or kv_in). Returns the drafter part's offset.
    pub fn cache_restore(&mut self, slot: usize, pos: usize, blk: usize) -> Result<usize> {
        let off = self.state_block_off(blk)?;
        self.mirror(|| crate::follow::Call::CacheRestore { slot, pos, off })?;
        self.cache_restore_at(slot, pos, off)?;
        Ok(off + self.park_gdn_bytes())
    }

    fn cache_restore_at(&mut self, slot: usize, pos: usize, off: usize) -> Result<()> {
        ensure!(pos <= self.pool_ctx(), "cache_restore: position {pos}");
        if let Some(rp) = self.replay.as_mut() {
            rp.pending[slot] = None;
        }
        self.park_copy(slot, &(pos..pos), true, off, false)?;
        self.slot_len[slot] = pos;
        Ok(())
    }

    /// KV rows [0, rows) of `slot` out to KV blocks `blks` (KV_BLOCK rows each, in order). Blocks have a fixed
    /// stride whatever the row count (layer i's K at i * 2 * KV_BLOCK rows, its V KV_BLOCK rows later, row r at
    /// r % KV_BLOCK), so kv_in can take back any prefix of what was written.
    pub fn kv_out(&mut self, slot: usize, rows: usize, blks: &[usize]) -> Result<()> {
        ensure!(blks.len() * KV_BLOCK >= rows && rows <= self.slot_len[slot], "kv_out: {rows} rows (slot {slot} at {}) in {} blocks",
            self.slot_len[slot], blks.len());
        let offs = blks.iter().take(rows.div_ceil(KV_BLOCK)).map(|&b| self.kv_block_off(b)).collect::<Result<Vec<_>>>()?;
        self.mirror(|| crate::follow::Call::KvOut { slot, rows, offs: offs.clone() })?;
        self.settled(slot)?;
        self.kv_blocks_copy(slot, rows, &offs, true)
    }

    /// KV rows [0, rows) of `slot` back from KV blocks `blks` written by kv_out with at least `rows` rows (any
    /// prefix: e.g. a parked conversation restored at an earlier snapshot); the slot's length becomes `rows`
    /// (cache_restore then sets the snapshot's).
    pub fn kv_in(&mut self, slot: usize, rows: usize, blks: &[usize]) -> Result<()> {
        ensure!(blks.len() * KV_BLOCK >= rows, "kv_in: {rows} rows in {} blocks", blks.len());
        let offs = blks.iter().take(rows.div_ceil(KV_BLOCK)).map(|&b| self.kv_block_off(b)).collect::<Result<Vec<_>>>()?;
        self.mirror(|| crate::follow::Call::KvIn { slot, rows, offs: offs.clone() })?;
        self.kv_in_at(slot, rows, &offs)
    }

    fn kv_in_at(&mut self, slot: usize, rows: usize, offs: &[usize]) -> Result<()> {
        ensure!(rows <= self.pool_ctx(), "kv_in: {rows} rows");
        if let Some(rp) = self.replay.as_mut() {
            rp.pending[slot] = None;
        }
        self.kv_blocks_copy(slot, rows, offs, false)?;
        self.slot_len[slot] = rows;
        Ok(())
    }

    /// The copies of kv_out / kv_in (`out`: slot -> blocks), fixed-stride block layout.
    fn kv_blocks_copy(&self, slot: usize, rows: usize, offs: &[usize], out: bool) -> Result<()> {
        ensure!(slot < self.n_slots && offs.len() >= rows.div_ceil(KV_BLOCK) && rows <= self.cap(slot),
            "kv blocks: slot {slot} rows {rows} blocks {}", offs.len());
        let tok = self.kv_tok();
        let tokb = tok * 2;
        for (j, &off) in offs.iter().enumerate().take(rows.div_ceil(KV_BLOCK)) {
            let (a, b) = (j * KV_BLOCK, rows.min((j + 1) * KV_BLOCK));
            let n = (b - a) * tokb;
            for (i, (k, v)) in self.kv.iter().enumerate() {
                for (h, buf) in [k, v].into_iter().enumerate() {
                    let sp = buf.at((slot * self.max_ctx + a) * tok);
                    self.park_xfer(off + (2 * i + h) * KV_BLOCK * tokb, sp, n, out)?;
                }
            }
        }
        self.stream.sync()?;
        Ok(())
    }

    /// The copies of a park entry, without the mirror (`out`: slot -> pool, else pool -> slot).
    fn park_copy(&self, slot: usize, kv: &std::ops::Range<usize>, gdn: bool, off: usize, out: bool) -> Result<()> {
        for (sp, o, n) in self.park_pieces(slot, kv, gdn)? {
            self.park_xfer(off + o, sp, n, out)?;
        }
        self.stream.sync()?;
        Ok(())
    }

    /// Save slot `slot`'s KV tokens `kv` (and, with `gdn`, its recurrent state, which is the state at the slot's
    /// current length) into the park pool at byte `off` (park_bytes(kv.len(), gdn) bytes).
    pub fn park_save(&mut self, slot: usize, kv: std::ops::Range<usize>, gdn: bool, off: usize) -> Result<()> {
        self.mirror(|| crate::follow::Call::ParkSave { slot, a: kv.start, b: kv.end, gdn, off })?;
        self.settled(slot)?;
        ensure!(kv.end <= self.slot_len[slot], "park_save: tokens {kv:?} past slot {slot}'s length {}", self.slot_len[slot]);
        for (sp, o, n) in self.park_pieces(slot, &kv, gdn)? {
            self.park_xfer(off + o, sp, n, true)?;
        }
        self.stream.sync()?;
        Ok(())
    }

    /// Restore a park entry saved with the same `kv` / `gdn` into slot `slot` and set its length to `len`. With
    /// `gdn` the slot's recurrent state becomes the saved one, so `len` must be the length it was saved at; KV
    /// tokens outside `kv` are the slot's own (e.g. a snapshot restore keeps the slot's KV [0, len) and passes an
    /// empty `kv`).
    pub fn park_restore(&mut self, slot: usize, kv: std::ops::Range<usize>, gdn: bool, off: usize, len: usize) -> Result<()> {
        self.mirror(|| crate::follow::Call::ParkRestore { slot, a: kv.start, b: kv.end, gdn, off, len })?;
        ensure!(kv.end <= len && len <= self.cap(slot), "park_restore: tokens {kv:?}, length {len}");
        if let Some(rp) = self.replay.as_mut() {
            rp.pending[slot] = None;
        }
        for (sp, o, n) in self.park_pieces(slot, &kv, gdn)? {
            self.park_xfer(off + o, sp, n, false)?;
        }
        self.stream.sync()?;
        self.slot_len[slot] = len;
        Ok(())
    }

    /// Copy slot `src`'s KV tokens `kv` (and, with `gdn`, its recurrent state) into slot `dst` and set `dst`'s
    /// length to `len` (a shared prefix computed once: `kv` = 0..len, gdn, with `src` at length `len`).
    pub fn copy_slot(&mut self, src: usize, dst: usize, kv: std::ops::Range<usize>, gdn: bool, len: usize) -> Result<()> {
        self.mirror(|| crate::follow::Call::CopySlot { src, dst, a: kv.start, b: kv.end, gdn, len })?;
        self.settled(src)?;
        ensure!(src != dst && kv.end <= len && kv.end <= self.slot_len[src] && len <= self.cap(dst),
            "copy_slot: {src} -> {dst} tokens {kv:?} length {len}");
        ensure!(!gdn || len == self.slot_len[src], "copy_slot: the recurrent state is {src}'s at {}, not {len}",
            self.slot_len[src]);
        if let Some(rp) = self.replay.as_mut() {
            rp.pending[dst] = None;
        }
        let (s, d) = (self.park_pieces(src, &kv, gdn)?, self.park_pieces(dst, &kv, gdn)?);
        for ((sp, _, n), (dp, _, _)) in s.into_iter().zip(d) {
            self.stream.memcpy_dtod(dp, sp, n)?;
        }
        self.stream.sync()?;
        self.slot_len[dst] = len;
        Ok(())
    }

    /// Forget a slot's context (zero its recurrent + conv state; KV entries are simply overwritten).
    pub fn reset_slot(&mut self, slot: usize) -> Result<()> {
        self.mirror(|| crate::follow::Call::Reset { slot })?;
        if let Some(rp) = self.replay.as_mut() {
            rp.pending[slot] = None;
        }
        let cd = self.cfg.conv_dim();
        let rs = self.cfg.lin_v_heads * self.cfg.lin_dk * self.cfg.lin_dv;
        for c in &self.conv_state {
            self.stream.memset_d8(c.at(slot * 3 * cd), 0, 3 * cd * 2)?;
        }
        for r in &self.rec_state {
            self.stream.memset_d8(r.at(slot * rs), 0, rs * 4)?;
        }
        self.stream.sync()?;
        self.slot_len[slot] = 0;
        Ok(())
    }

    /// Run the model over `segs` (slot, tokens). Logits are produced for the rows listed in `logit_rows`
    /// (indices into the concatenated rows) into `self.logits` in that order. Synchronous.
    pub fn forward(&mut self, segs: &[(usize, &[u32])], logit_rows: &[usize]) -> Result<()> {
        let dec = segs.iter().take_while(|s| s.1.len() == 1).count();
        self.forward_ex(segs, logit_rows, false, dec)
    }

    /// `hist`: also keep what cuts a speculative verify back ([`Engine::commit`]): the conv window after each of its
    /// rows and its GDN recurrence inputs (replay, the verify itself leaves the state alone; without replay the
    /// state after each of the first `hist_rows` rows). `dec`: the leading rows that are decode /
    /// verify rows (they run the EXL3 GEMV even in a large batch; the rest are prompt rows).
    pub fn forward_ex(&mut self, segs: &[(usize, &[u32])], logit_rows: &[usize], hist: bool, dec: usize) -> Result<()> {
        self.forward_tree(segs, &[], logit_rows, hist, dec)
    }

    /// forward_ex with verify trees: `parents[i]` (Some) makes segment i a tree (see prepare_tree). Tree segments
    /// must be decode/verify rows (inside `dec`) and are cut back with [`Engine::commit_tree`].
    pub fn forward_tree(&mut self, segs: &[(usize, &[u32])], parents: &[Option<&[i32]>], logit_rows: &[usize],
        hist: bool, dec: usize) -> Result<()> {
        self.mirror(|| crate::follow::Call::Forward {
            segs: segs.iter().map(|(s, t)| (*s, t.to_vec())).collect(),
            logit_rows: logit_rows.to_vec(),
            hist,
            dec,
            parents: parents.iter().map(|p| p.map(|v| v.to_vec())).collect(),
        })?;
        ensure!(!hist || self.hist_rows > 0, "forward: no state history allocated");
        let pos: Vec<usize> = segs.iter().map(|s| self.slot_len[s.0]).collect();
        let mut p = self.prepare_tree(segs, &pos, logit_rows, None, parents, dec)?;
        ensure!(dec <= DEC_MAX || dec == p.rows, "forward: {dec} decode rows (max {DEC_MAX})");
        p.dec = dec.min(p.rows);
        p.lin_dec = p.dec;
        for sg in p.segs.iter().filter(|sg| sg.1 as usize >= p.dec) {
            if sg.1 as usize != p.lin_dec || sg.2 as usize > X3_ROWS || p.lin_dec + sg.2 as usize > DEC_MAX {
                break;
            }
            p.lin_dec += sg.2 as usize;
        }
        // a tree's GDN layers run read-only and commit_tree replays the kept path from the saved inputs
        for (sg, _) in p.segs.iter().zip(&p.seg_tree).filter(|(_, t)| **t) {
            ensure!(hist && self.replay_seg(&p, sg), "forward: a verify tree needs GDN replay (hist, <= {} decode rows)",
                self.hist_cap);
        }
        // with replay there is no recurrent history: every multi-row verify segment must be a replay segment
        if hist && self.replay.is_some() {
            for sg in p.segs.iter().filter(|sg| sg.2 > 1 && (sg.1 + sg.2) as usize <= p.dec) {
                ensure!(self.replay_seg(&p, sg), "forward: a {}-row verify (GDN replay keeps {} rows)", sg.2, self.hist_cap);
            }
        }
        if hist && self.replay.is_some() {
            let flags: Vec<u32> = p.segs.iter().map(|sg| self.replay_seg(&p, sg) as u32).collect();
            let rp = self.replay.as_mut().unwrap();
            ensure!(flags.len() <= rp.ro_host.len(), "forward: {} segments", flags.len());
            for (i, (sg, &f)) in p.segs.iter().zip(&flags).enumerate() {
                rp.pending[sg.0 as usize] = (f != 0).then_some(i);
            }
            rp.ro_host.as_mut_slice()[..flags.len()].copy_from_slice(&flags);
            unsafe { self.stream.memcpy_htod(rp.ro.ptr(), &rp.ro_host.as_slice()[..flags.len()])? };
        }
        let key = GraphKey { seg_lens: p.segs.iter().map(|s| s.2).collect(), n_logits: p.n_logits, hist, mtp: None,
            nsplit: attn_splits(p.rows, p.max_pos), dec: p.dec, trees: p.seg_tree.clone() };
        self.run(&p, key, |e, p| e.enqueue(p, hist))?;
        for &(slot, toks) in segs {
            self.slot_len[slot] += toks.len();
        }
        Ok(())
    }

    /// After a verify forward of `n` rows for `slot` that started at length `start`: keep the first `keep` rows
    /// (1..=n). The GDN and conv states are rolled back to the state after row keep-1.
    pub fn commit(&mut self, slot: usize, start: usize, n: usize, keep: usize) -> Result<()> {
        self.mirror(|| crate::follow::Call::Commit { slot, start, n, keep })?;
        ensure!(keep >= 1 && keep <= n, "commit: keep {keep} of {n}");
        self.commit_rows(slot, start, n, keep, keep - 1)
    }

    /// Keep `keep` rows of the last verify (already consecutive): conv state (and without replay the recurrent
    /// state) from the history of row `last`, or the replay of the first `keep` saved rows.
    fn commit_rows(&mut self, slot: usize, start: usize, n: usize, keep: usize, last: usize) -> Result<()> {
        let replay = self.replay.as_mut().and_then(|rp| rp.pending[slot].take());
        if keep < n {
            ensure!(last < self.hist_cap, "commit: no history for row {last}");
            let cd = self.cfg.conv_dim();
            let rs = self.cfg.lin_v_heads * self.cfg.lin_dk * self.cfg.lin_dv;
            let t = slot * self.hist_cap + last;
            for (i, c) in self.conv_state.iter().enumerate() {
                self.stream.memcpy_dtod(c.at(slot * 3 * cd), self.conv_hist[i].at(t * 3 * cd), 3 * cd * 2)?;
            }
            if replay.is_none() {
                for (i, r) in self.rec_state.iter().enumerate() {
                    self.stream.memcpy_dtod(r.at(slot * rs), self.rec_hist[i].at(t * rs), rs * 4)?;
                }
            }
        }
        if let Some(si) = replay {
            // the verify left the GDN state at its start: run the kept rows' recurrence from the saved inputs
            let nvh = self.cfg.lin_v_heads;
            let rp = self.replay.as_mut().unwrap();
            rp.seg_host.as_mut_slice().copy_from_slice(&[slot as u32, (si * self.hist_cap) as u32, keep as u32]);
            let rp = self.replay.as_ref().unwrap();
            unsafe { self.stream.memcpy_htod(rp.seg.ptr(), rp.seg_host.as_slice())? };
            unsafe {
                self.stream.launch(&self.k.gdn_replay, (4 * nvh as u32, 1u32, rp.conv.len() as u32), 512, 0,
                    (rp.tab.ptr(), (2 * nvh) as u32, rp.seg.ptr(), rp.seg.at(1), rp.seg.at(2)))?;
            }
        }
        if keep < n || replay.is_some() {
            self.stream.sync()?;
        }
        self.slot_len[slot] = start + keep;
        Ok(())
    }

    /// After a tree verify of `n` rows for `slot` that started at length `start`: keep the root path `path` (tree row
    /// indices, path[0] = 0, strictly increasing). One launch moves the kept rows to consecutive rows: K/V at the
    /// verify start, every GDN layer's saved replay inputs and, with `taps` (the segment's first row in the last
    /// forward), the drafter's feature taps. Then the commit is the chain's: replay of path.len() rows, conv state
    /// from the history of the path's last row (the GDN tree kernels write each row's along its own root path).
    pub fn commit_tree(&mut self, slot: usize, start: usize, n: usize, path: &[usize], taps: Option<usize>) -> Result<()> {
        self.mirror(|| crate::follow::Call::CommitTree { slot, start, n, path: path.to_vec(), taps })?;
        ensure!(!path.is_empty() && path[0] == 0 && path.windows(2).all(|w| w[0] < w[1]) && *path.last().unwrap() < n
            && path.len() <= self.tree_path_host.len(), "commit_tree: bad path {path:?} of {n}");
        let last = *path.last().unwrap();
        if last + 1 == path.len() {
            return self.commit_rows(slot, start, n, path.len(), last);
        }
        let si = self.replay.as_ref().and_then(|rp| rp.pending[slot]);
        let Some(si) = si else { anyhow::bail!("commit_tree: slot {slot} has no tree verify to replay") };
        let mut tab: Vec<(DevPtr, usize)> = Vec::with_capacity(GATHER_MAX);
        let row = self.kv_tok();
        for (kc, vc) in &self.kv {
            for c in [kc, vc] {
                tab.push((c.at((slot * self.max_ctx + start) * row), row));
            }
        }
        let rp = self.replay.as_ref().unwrap();
        let (cd, ab) = (self.cfg.conv_dim(), 2 * self.cfg.lin_v_heads);
        for (c, a) in rp.conv.iter().zip(&rp.ab) {
            tab.push((c.at(si * self.hist_cap * cd), cd));
            tab.push((a.at(si * self.hist_cap * ab), ab));
        }
        if let (Some(r0), Some((layers, f))) = (taps, &self.feat) {
            let fw = layers.len() * self.cfg.hidden;
            tab.push((f.at(r0 * fw), fw));
        }
        ensure!(tab.len() <= GATHER_MAX && tab.iter().all(|(p, r)| p.0 % 16 == 0 && r % 8 == 0),
            "commit_tree: {} gather buffers (rows of 8 elements, 16-byte aligned)", tab.len());
        for (j, &p) in path.iter().enumerate() {
            self.tree_path_host.as_mut_slice()[j] = p as u32;
        }
        for (j, &(p, r)) in tab.iter().enumerate() {
            self.gather_host.as_mut_slice()[2 * j..2 * j + 2].copy_from_slice(&[p.0, r as u64]);
        }
        unsafe {
            self.stream.memcpy_htod(self.tree_path.ptr(), &self.tree_path_host.as_slice()[..path.len()])?;
            self.stream.memcpy_htod(self.gather_tab.ptr(), &self.gather_host.as_slice()[..2 * tab.len()])?;
            self.stream.launch(&self.k.rows_gather, tab.len() as u32, 256, 0,
                (self.gather_tab.ptr(), self.tree_path.ptr(), path.len() as u32))?;
        }
        self.commit_rows(slot, start, n, path.len(), last)
    }

    /// GDN replay mode: is this segment (slot, first row, rows, position) of a `hist` forward a verify that runs
    /// read-only (leading decode rows, more than one row, within the saved rows)?
    fn replay_seg(&self, p: &Prepared, sg: &(u32, u32, u32, u32)) -> bool {
        self.replay.is_some() && sg.2 > 1 && sg.2 as usize <= self.hist_cap && (sg.1 + sg.2) as usize <= p.dec
    }

    /// The MTP head over `rows`; returns its greedy draft for each row index in `draw`. Rows of one slot at
    /// consecutive positions should be adjacent (they form one attention segment). All rows read the same
    /// kind of source. Writes the head's KV cache at each row's position.
    pub fn mtp_forward(&mut self, rows: &[MtpRow], draw: &[usize]) -> Result<Vec<u32>> {
        self.mirror(|| crate::follow::Call::Mtp { rows: rows.to_vec(), draw: draw.to_vec() })?;
        ensure!(self.w.mtp.is_some(), "no MTP head loaded");
        ensure!(!rows.is_empty(), "mtp: no rows");
        let tap = matches!(rows[0].src, MtpSrc::Tap(_));
        let mut segs: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut pos0: Vec<usize> = Vec::new();
        let mut srcs = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            ensure!(matches!(r.src, MtpSrc::Tap(_)) == tap, "mtp: mixed sources");
            srcs.push(match r.src { MtpSrc::Tap(j) | MtpSrc::Out(j) => j as u32 });
            ensure!(r.pos < self.cap(r.slot), "mtp: position {} past context", r.pos);
            let cont = i > 0 && rows[i - 1].slot == r.slot && rows[i - 1].pos + 1 == r.pos;
            if cont {
                segs.last_mut().unwrap().1.push(r.token);
            } else {
                segs.push((r.slot, vec![r.token]));
                pos0.push(r.pos);
            }
        }
        let seg_refs: Vec<(usize, &[u32])> = segs.iter().map(|(s, t)| (*s, t.as_slice())).collect();
        let p = self.prepare(&seg_refs, &pos0, draw, Some(&srcs))?;
        let key = GraphKey { seg_lens: p.segs.iter().map(|s| s.2).collect(), n_logits: p.n_logits, hist: false,
            mtp: Some(tap), nsplit: attn_splits(p.rows, p.max_pos), dec: p.dec, trees: Vec::new() };
        self.run(&p, key, |e, p| e.enqueue_mtp(p, tap))?;
        let mut v = vec![0u32; draw.len()];
        fe_cuda::memcpy_dtoh(&mut v, self.argmax_out.ptr())?;
        Ok(v)
    }

    /// Replay a captured graph for batches of short segments, else enqueue directly; then synchronise.
    fn run(&mut self, p: &Prepared, key: GraphKey, f: impl Fn(&Engine, &Prepared) -> Result<()>) -> Result<()> {
        let short = p.segs.iter().all(|s| (s.2 as usize) < FLASH_MIN) && p.rows <= 16;
        if self.use_graphs && short {
            if !self.graphs.contains_key(&key) {
                // captured graphs hold device memory: with several sequences the batch shapes multiply, so drop them
                // all when the device runs low, and run this batch directly if a capture still does not fit
                if !self.graphs.is_empty() && fe_cuda::mem_info().is_ok_and(|(free, _)| free < GRAPH_MIN_FREE) {
                    self.graphs.clear();
                }
                self.stream.begin_capture()?;
                let r = f(self, p);
                let g = self.stream.end_capture();
                r?;
                match g {
                    Ok(g) => { self.graphs.insert(key.clone(), g); }
                    Err(_) => {
                        self.graphs.clear();
                        f(self, p)?;
                        self.stream.sync()?;
                        return Ok(());
                    }
                }
            }
            self.graphs[&key].launch(&self.stream)?;
        } else {
            f(self, p)?;
        }
        self.stream.sync()?;
        Ok(())
    }

    /// Validate the batch and upload its metadata (ids, row slot/pos, segments, logit rows, MTP sources).
    fn prepare(&mut self, segs: &[(usize, &[u32])], pos0: &[usize], logit_rows: &[usize], srcs: Option<&[u32]>)
        -> Result<Prepared> {
        self.prepare_tree(segs, pos0, logit_rows, srcs, &[], 0)
    }

    /// `parents[i]` (when given and Some) makes segment i a verify tree: parents[i][r] = parent row of row r within
    /// the segment (-1 for the root, row 0), every parent before its child. Rows get RoPE position start + depth,
    /// cache index start + r, and attend to the cache before the tree plus their ancestors (and themselves).
    /// `dec`: the leading decode / verify rows (their segments' attention takes fe_attn_vk at any length).
    fn prepare_tree(&mut self, segs: &[(usize, &[u32])], pos0: &[usize], logit_rows: &[usize], srcs: Option<&[u32]>,
        parents: &[Option<&[i32]>], dec: usize) -> Result<Prepared> {
        let rows: usize = segs.iter().map(|s| s.1.len()).sum();
        ensure!(rows > 0 && rows <= self.sc.max_rows, "forward: {rows} rows (max {})", self.sc.max_rows);
        ensure!(logit_rows.len() <= self.max_logit_rows, "too many logit rows");
        let mut ids = Vec::with_capacity(rows);
        let mut rslot = Vec::with_capacity(rows);
        let mut rpos = Vec::with_capacity(rows);
        let (mut sslot, mut srow0, mut slen) = (Vec::new(), Vec::new(), Vec::new());
        let mut max_pos = 0;
        let mut seginfo = Vec::with_capacity(segs.len());
        let (mut cpos, mut tbase, mut tmask, mut tpar) = (Vec::with_capacity(rows), Vec::with_capacity(rows),
            Vec::with_capacity(rows), Vec::with_capacity(rows));
        let mut tstep = Vec::with_capacity(rows);
        let mut seg_tree = Vec::with_capacity(segs.len());
        for (si, &(slot, toks)) in segs.iter().enumerate() {
            ensure!(slot < self.n_slots, "slot {slot}");
            let p0 = pos0[si];
            ensure!(p0 + toks.len() <= self.cap(slot), "slot {slot}: context {} > {}", p0 + toks.len(), self.cap(slot));
            seginfo.push((slot as u32, ids.len() as u32, toks.len() as u32, p0 as u32));
            sslot.push(slot as u32);
            srow0.push(ids.len() as u32);
            slen.push(toks.len() as u32);
            let tree = parents.get(si).copied().flatten();
            seg_tree.push(tree.is_some());
            if let Some(par) = tree {
                ensure!(par.len() == toks.len() && toks.len() <= 32, "tree: {} parents for {} rows (<= 32)", par.len(),
                    toks.len());
                ensure!(par[0] == -1, "tree: row 0 must be the root");
                ensure!(par.iter().enumerate().skip(1).all(|(i, &p)| p >= 0 && (p as usize) < i),
                    "tree: parents must precede their rows");
                tstep.extend(tree_steps(par)?);
            } else {
                tstep.extend(0..toks.len() as u32);
            }
            let mut depth = vec![0usize; toks.len()];
            let mut mask = vec![0u32; toks.len()];
            for (i, &t) in toks.iter().enumerate() {
                ids.push(t);
                rslot.push(slot as u32);
                match tree {
                    Some(par) => {
                        let pa = par[i];
                        ensure!(i == 0 || (pa >= 0 && (pa as usize) < i), "tree: row {i} parent {pa}");
                        if i > 0 {
                            depth[i] = depth[pa as usize] + 1;
                            mask[i] = mask[pa as usize];
                        }
                        mask[i] |= 1 << i;
                        rpos.push((p0 + depth[i]) as u32);
                        cpos.push((p0 + i) as u32);
                        tbase.push(p0 as u32);
                        tmask.push(mask[i]);
                        tpar.push(ancestors(par, i));
                    }
                    None => {
                        rpos.push((p0 + i) as u32);
                        cpos.push((p0 + i) as u32);
                        tbase.push(u32::MAX);
                        tmask.push(0);
                        tpar.push(ancestors_chain(i));
                    }
                }
            }
            max_pos = max_pos.max(p0 + toks.len());
        }
        let nseg = segs.len();
        let mut meta = Vec::with_capacity(4 * rows + 3 * nseg + logit_rows.len());
        meta.extend_from_slice(&ids);
        let o_rslot = meta.len();
        meta.extend_from_slice(&rslot);
        let o_rpos = meta.len();
        meta.extend_from_slice(&rpos);
        let o_sslot = meta.len();
        meta.extend_from_slice(&sslot);
        let o_srow0 = meta.len();
        meta.extend_from_slice(&srow0);
        let o_slen = meta.len();
        meta.extend_from_slice(&slen);
        let o_lr = meta.len();
        meta.extend(logit_rows.iter().map(|&r| r as u32));
        let o_cpos = meta.len();
        meta.extend_from_slice(&cpos);
        let o_tbase = meta.len();
        meta.extend_from_slice(&tbase);
        let o_tmask = meta.len();
        meta.extend_from_slice(&tmask);
        let o_par = meta.len();
        meta.extend_from_slice(&tpar);
        let o_tstep = meta.len();
        meta.extend_from_slice(&tstep);
        let o_stree = meta.len();
        meta.extend(seg_tree.iter().map(|&t| t as u32));
        let (mut asegs, mut aseg_vk) = (Vec::with_capacity(nseg), Vec::with_capacity(nseg));
        for (&(slot, r0, n, p0), &t) in seginfo.iter().zip(&seg_tree) {
            let vk = t || (r0 + n) as usize <= dec;
            let piece = if vk { VK_ROWS } else { n.max(1) };
            let mut k = 0;
            while k < n {
                asegs.push((slot, r0 + k, piece.min(n - k), p0 + k));
                aseg_vk.push(vk);
                k += piece;
            }
        }
        let (o_asrow0, o_aslen) = if asegs.len() == nseg { (o_srow0, o_slen) } else {
            let o = meta.len();
            meta.extend(asegs.iter().map(|a| a.1));
            meta.extend(asegs.iter().map(|a| a.2));
            (o, o + asegs.len())
        };
        let o_src = meta.len();
        if let Some(sr) = srcs {
            meta.extend_from_slice(sr);
        }
        ensure!(meta.len() <= self.sc.meta.len(), "metadata overflow");
        // Upload on the engine stream from pinned memory. A synchronous cuMemcpyHtoD from pageable memory returns
        // once the bytes are staged, before the DMA lands, and does not order against our non-blocking stream:
        // the kernels could read the previous step's ids/positions (the hybrid run's random bad steps).
        self.sc.meta_host.as_mut_slice()[..meta.len()].copy_from_slice(&meta);
        unsafe { self.stream.memcpy_htod(self.sc.meta.ptr(), &self.sc.meta_host.as_slice()[..meta.len()])? };
        if let (Embed::Host(tab), Some((host, dev, _))) = (&self.w.embed, self.emb.as_mut()) {
            // host embedding: this forward's rows, gathered into pinned memory and uploaded on the engine stream
            let hb = self.cfg.hidden * 2;
            let dst = unsafe { std::slice::from_raw_parts_mut(host.as_mut_slice().as_mut_ptr() as *mut u8, rows * hb) };
            for (r, &id) in meta[..rows].iter().enumerate() {
                let o = id as usize * hb;
                ensure!(o + hb <= tab.len(), "token id {id} outside the embedding table");
                dst[r * hb..(r + 1) * hb].copy_from_slice(&tab[o..o + hb]);
            }
            unsafe { self.stream.memcpy_htod(dev.ptr(), &host.as_slice()[..rows * self.cfg.hidden])? };
        }
        Ok(Prepared { rows, nseg, o_ids: 0, o_rslot, o_rpos, o_sslot, o_srow0, o_slen, o_lr, o_src,
            n_logits: logit_rows.len(), max_pos, segs: seginfo, dec: 0, lin_dec: 0, o_cpos, o_tbase, o_tmask, o_par,
            o_tstep, o_stree, seg_tree, asegs, aseg_vk, o_asrow0, o_aslen })
    }

    /// FE_FWD_PROF=1: synchronise and add the time since the last mark to `label` (a forward's phases; printed by
    /// fwd_prof_report). Off: nothing.
    fn mark(&self, label: &'static str) {
        if let Some(p) = &self.fwd_prof {
            let _ = self.stream.sync();
            let mut p = p.borrow_mut();
            let now = std::time::Instant::now();
            let dt = now.duration_since(p.0).as_secs_f64() * 1e3;
            p.0 = now;
            match p.1.iter_mut().find(|e| e.0 == label) {
                Some(e) => e.1 += dt,
                None => p.1.push((label, dt)),
            }
        }
    }

    /// Print and reset the FE_FWD_PROF totals.
    pub fn fwd_prof_report(&self, what: &str) {
        if let Some(p) = &self.fwd_prof {
            let mut p = p.borrow_mut();
            let tot: f64 = p.1.iter().map(|e| e.1).sum();
            eprintln!("fwd prof ({what}): {:.1} ms: {}", tot, p.1.iter().map(|e| format!("{} {:.1}", e.0, e.1)).collect::<Vec<_>>().join(" | "));
            p.1.clear();
            p.0 = std::time::Instant::now();
        }
    }

    /// (table, row ids) for fe_embed: the device table with the batch's token ids, or the uploaded rows of a host
    /// table with the identity index.
    fn embed_src(&self, pr: &Prepared) -> (DevPtr, DevPtr) {
        match (&self.w.embed, &self.emb) {
            (Embed::Dev(t), _) => (t.ptr(), self.sc.meta.at(pr.o_ids)),
            (Embed::Host(_), Some((_, dev, ident))) => (dev.ptr(), ident.ptr()),
            (Embed::Host(_), None) => unreachable!("host embedding without staging"),
        }
    }

    /// The embedding table on the device (None when it lives in host memory).
    pub fn embed_dev(&self) -> Option<DevPtr> {
        match &self.w.embed {
            Embed::Dev(t) => Some(t.ptr()),
            Embed::Host(_) => None,
        }
    }

    /// Enqueue the target forward (no host syncs: capturable).
    fn enqueue(&self, pr: &Prepared, hist: bool) -> Result<()> {
        let cfg = &self.cfg;
        let s = &self.stream;
        let k = &self.k;
        let lc = LinCtx { k, x3: self.w.x3.as_ref(), fp4: self.w.fp4.as_ref().zip(self.fp4_ws.as_ref()), s,
            tmp: self.sc.tmp.ptr(), dec: pr.lin_dec };
        let h = cfg.hidden as u32;
        let (x, taps) = (self.sc.x.ptr(), self.sc.taps.ptr());
        unsafe {
            let (tab, ids) = self.embed_src(pr);
            s.launch(&k.embed, pr.rows as u32, 256, 0, (x, tab, ids, h))?;
        }
        // streamed prompt weights: only forwards with prompt rows past the decode rows read them
        let ps = self.w.pstream.as_ref().filter(|_| pr.rows > X3_ROWS && pr.lin_dec < pr.rows);
        self.mark("other");
        for (li, layer) in self.w.layers.iter().enumerate() {
            if let Some(ps) = ps {
                ps.fetch(li, s)?;
            }
            self.enqueue_layer(layer, pr, &lc, hist)?;
            if let Some(ps) = ps {
                ps.release(li, s)?;
            }
            if let Some((ls, buf)) = &self.feat {
                if let Some(j) = ls.iter().position(|&t| t == li) {
                    let ld = (ls.len() * cfg.hidden) as u32;
                    unsafe {
                        s.launch(&k.copy2d, (pr.rows as u32, h.div_ceil(512)), 256, 0,
                            (buf.ptr().add::<u16>(j * cfg.hidden), ld, x, h, h))?;
                    }
                }
            }
        }
        // final norm over every row (the MTP head reads these), then lm_head on the logit rows
        unsafe { s.launch(&k.rmsnorm, pr.rows as u32, 256, 0, (taps, x, self.w.norm.ptr(), h, cfg.eps, h, h))? };
        if pr.n_logits > 0 {
            let sel = self.sc.sel.ptr();
            unsafe { s.launch(&k.gather_rows, pr.n_logits as u32, 256, 0, (sel, taps, self.sc.meta.at(pr.o_lr), h))? };
            // logits rows are decode / verify / last-prompt rows: always the trellis head
            let lh = LinCtx { dec: pr.n_logits, ..lc };
            match &self.tp {
                Some(tp) => {
                    let half = self.w.lm_head.n;
                    self.w.lm_head.forward(&lh, tp.logits_half.ptr(), half, sel, h as usize, pr.n_logits, None)?;
                    tp.fab.gather_halves(s, tp.logits_half.ptr(), self.logits.ptr(), tp.gather.ptr(), pr.n_logits, half)?;
                }
                None => self.w.lm_head.forward(&lh, self.logits.ptr(), cfg.vocab, sel, h as usize, pr.n_logits, None)?,
            }
        }
        Ok(())
    }

    /// Enqueue the MTP head: x = fc([norm_e(embed(ids)) | norm_h(src rows)]), the layer, mtp.norm -> mtp_out,
    /// lm_head on the draw rows.
    fn enqueue_mtp(&self, pr: &Prepared, tap: bool) -> Result<()> {
        let cfg = &self.cfg;
        let m = self.w.mtp.as_ref().unwrap();
        let s = &self.stream;
        let k = &self.k;
        let lc = LinCtx { k, x3: self.w.x3.as_ref(), fp4: self.w.fp4.as_ref().zip(self.fp4_ws.as_ref()), s,
            tmp: self.sc.tmp.ptr(), dec: pr.dec };
        let h = cfg.hidden as u32;
        let r32 = pr.rows as u32;
        let (x, xn, cat, out) = (self.sc.x.ptr(), self.sc.xn.ptr(), self.sc.big.ptr(), self.sc.mtp_out.ptr());
        let src = if tap { self.sc.taps.ptr() } else { out };
        unsafe {
            let (tab, ids) = self.embed_src(pr);
            s.launch(&k.embed, r32, 256, 0, (x, tab, ids, h))?;
            s.launch(&k.gather_rows, r32, 256, 0, (xn, src, self.sc.meta.at(pr.o_src), h))?;
            s.launch(&k.rmsnorm, r32, 256, 0, (cat, x, m.pre_e.ptr(), h, cfg.eps, h, 2 * h))?;
            s.launch(&k.rmsnorm, r32, 256, 0, (cat.add::<u16>(h as usize), xn, m.pre_h.ptr(), h, cfg.eps, h, 2 * h))?;
        }
        m.fc.forward(&lc, x, h as usize, cat, 2 * h as usize, pr.rows, None)?;
        self.enqueue_layer(&m.layer, pr, &lc, false)?;
        unsafe { s.launch(&k.rmsnorm, r32, 256, 0, (out, x, m.norm.ptr(), h, cfg.eps, h, h))? };
        if pr.n_logits > 0 {
            let sel = self.sc.sel.ptr();
            unsafe { s.launch(&k.gather_rows, pr.n_logits as u32, 256, 0, (sel, out, self.sc.meta.at(pr.o_lr), h))? };
            let head = m.head.as_ref().unwrap_or(&self.w.lm_head);
            head.forward(&lc, self.logits.ptr(), head.n, sel, h as usize, pr.n_logits, None)?;
            unsafe {
                s.launch(&k.argmax, pr.n_logits as u32, 1024, 0, (self.argmax_out.ptr(), self.logits.ptr(), head.n as u32))?;
            }
        }
        Ok(())
    }

    /// One decoder layer on the residual stream `x` (pre-norm mixer + residual, pre-norm SwiGLU + residual).
    fn enqueue_layer(&self, layer: &Layer, pr: &Prepared, lc: &LinCtx, hist: bool) -> Result<()> {
        let cfg = &self.cfg;
        let rows = pr.rows;
        let nseg = pr.nseg;
        let mp = |o: usize| self.sc.meta.at(o);
        let s = lc.s;
        let k = lc.k;
        let h = cfg.hidden;
        let (x, xn, big) = (self.sc.x.ptr(), self.sc.xn.ptr(), self.sc.big.ptr());
        let eps = cfg.eps;
        let r32 = rows as u32;
        // the cache's slot stride as its kernels want it: bf16 elements, or bytes of 4-bit records
        let slot_stride = if self.kvq > 0 { (self.max_ctx * cfg.n_kv * kvq_rec(self.kvq)) as u64 } else { (self.max_ctx * cfg.n_kv * cfg.head_dim) as u64 };
        // splits of a decode / verify row: fe_attn_vk has VK_SPLITS built in; the quantized forms take the grid's,
        // one block an SM for each segment's four K/V heads (a block of these kernels fills an SM's shared memory)
        let vk_splits = if self.kvq > 0 { self.vkq_splits } else { VK_SPLITS };
        let (k_prep, k_flash, k_vk, k_comb, flash_smem, seg_smem) = if self.kvq > 0 {
            let q = &k.kvq[(self.kvq == 6) as usize];
            (&q[0], &q[1], &q[2], &q[3], flash_q_smem(self.kvq), seg_q_smem(self.kvq))
        } else {
            (&k.attn_prep, &k.flash, &k.attn_vk, &k.attn_combine, FLASH_SMEM, SEG_SMEM)
        };
        let attn_smem = 4 * (6 * 256 + 12) * 4;
        let flash_threads = if self.kvq > 0 && flash_prod() { 256u32 } else { 128 };
        let h32 = h as u32;
        unsafe { s.launch(&k.rmsnorm, r32, 256, 0, (xn, x, layer.in_norm.ptr(), h32, eps, h32, h32))? };
        self.mark("norm");
        match &layer.mixer {
            Mixer::Gdn { proj, conv_w, a_log, dt_bias, norm_w, out, idx } => {
                let p = proj.forward(lc, big, xn, h, rows)?;
                self.mark("gdn proj");
                let ((qkv, ldqkv), (z, ldz), (a, lda), (b, ldb)) = (p[0], p[1], p[2], p[3]);
                ensure!(lda == ldb, "GDN a/b strides differ");
                let cd = cfg.conv_dim() as u32;
                let conv = self.sc.conv.ptr();
                let cst = self.conv_state[*idx].ptr();
                let rst = self.rec_state[*idx].ptr();
                // conv history for hist_cap rows; the recurrent one only without replay (hist_cap = hist_rows then)
                let (ch, rh, hr) = if hist {
                    (self.conv_hist[*idx].ptr(), self.rec_hist.get(*idx).map_or(DevPtr::NULL, |r| r.ptr()),
                     self.hist_cap as u32)
                } else {
                    (DevPtr::NULL, DevPtr::NULL, 0)
                };
                let vd = cfg.lin_v_heads * cfg.lin_dv;
                let ro = match &self.replay {
                    Some(rp) if hist => rp.ro.ptr(),
                    _ => DevPtr::NULL,
                };
                unsafe {
                    s.launch(&k.gdn_conv, (cd / 256, nseg as u32), 256, 0,
                        (conv, qkv, ldqkv as u32, conv_w.ptr(), cst, mp(pr.o_sslot), mp(pr.o_srow0), mp(pr.o_slen), ch, hr,
                         mp(pr.o_par), mp(pr.o_stree)))?;
                    // replay mode: save the verify segments' recurrence inputs for Engine::commit
                    if let (true, Some(rp)) = (hist, &self.replay) {
                        let (nvh, cdu) = (cfg.lin_v_heads, cd as usize);
                        for (si, sg) in pr.segs.iter().enumerate().filter(|(_, sg)| self.replay_seg(pr, sg)) {
                            let (r, row0, len) = (si * self.hist_cap, sg.1 as usize, sg.2 as usize);
                            s.memcpy_dtod(rp.conv[*idx].at(r * cdu), conv.add::<u16>(row0 * cdu), len * cdu * 2)?;
                            for (j, src) in [a, b].into_iter().enumerate() {
                                s.launch(&k.copy2d, (len as u32, 1u32), 256, 0, (rp.ab[*idx].at(r * 2 * nvh + j * nvh),
                                    (2 * nvh) as u32, src.add::<u16>(row0 * lda), lda as u32, nvh as u32))?;
                            }
                        }
                    }
                    // raw recurrence output into q's scratch, then the gated norm into mix. Chosen per segment, so
                    // a segment's numerics never depend on the rest of the batch: decode / verify segments (starting
                    // inside the leading dec rows) take the recurrence (state history / replay, trees), prompt segments
                    // the chunked tensor-core kernel when it is loaded (prompts are never rolled back)
                    let o = self.sc.q.ptr();
                    let nd = if self.gdn.is_some() { pr.segs.iter().take_while(|sg| (sg.1 as usize) < pr.dec).count() } else { nseg };
                    if nd > 0 {
                        s.launch(&k.gdn_rec2, (4 * cfg.lin_v_heads as u32, nd as u32), 512, 0,
                            (o, conv, a, b, lda as u32, a_log.ptr(), dt_bias.ptr(), rst, mp(pr.o_sslot), mp(pr.o_srow0),
                             mp(pr.o_slen), rh, hr, ro, mp(pr.o_tstep), mp(pr.o_stree)))?;
                    }
                    if let (Some(gk), true) = (&self.gdn, nd < nseg) {
                        gk.run(s, o, conv, a, b, lda, a_log.ptr(), dt_bias.ptr(), rst, mp(pr.o_sslot).add::<u32>(nd),
                            mp(pr.o_srow0).add::<u32>(nd), mp(pr.o_slen).add::<u32>(nd), nseg - nd)?;
                    }
                    s.launch(&k.gdn_norm, (r32, 2u32), 16 * cfg.lin_v_heads as u32, 0,
                        (self.sc.mix.ptr(), o, z, ldz as u32, norm_w.ptr(), eps))?;
                }
                self.mark("gdn core");
                self.resid_linear(lc, out, self.sc.mix.ptr(), vd, rows, 0)?;
                self.mark("gdn out");
            }
            Mixer::Attn { qkv, q_norm, k_norm, o, idx } => {
                let p = qkv.forward(lc, big, xn, h, rows)?;
                self.mark("attn proj");
                let ((qg, ldqg), (kx, ldk), (vx, ldv)) = (p[0], p[1], p[2]);
                let (kc, vc) = (&self.kv[*idx].0, &self.kv[*idx].1);
                let q = self.sc.q.ptr();
                unsafe {
                    s.launch(k_prep, r32, ((cfg.n_q + 2 * cfg.n_kv) * 32) as u32, 0, (q, kc.ptr(), vc.ptr(), qg, ldqg as u32, kx, ldk as u32, vx,
                        ldv as u32, q_norm.ptr(), k_norm.ptr(), self.rope.ptr(), mp(pr.o_rslot), mp(pr.o_rpos), mp(pr.o_cpos), slot_stride,
                        eps))?;
                }
                let scale_log2 = (1.0 / (cfg.head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
                // long segments: flash kernel; runs of short segments (decode rows): split-K + combine
                let mut i = 0;
                while i < pr.asegs.len() {
                    let (slot, r0, n, p0) = pr.asegs[i];
                    if n as usize >= FLASH_MIN && !pr.aseg_vk[i] {
                        unsafe {
                            s.launch(k_flash, (n.div_ceil(64), cfg.n_q as u32), flash_threads, flash_smem,
                                (self.sc.mix.ptr(), q, qg, ldqg as u32, kc.ptr(), vc.ptr(), slot_stride, slot, r0, n, p0,
                                 scale_log2))?;
                        }
                        i += 1;
                        continue;
                    }
                    let mut j = i;
                    let mut cnt = 0;
                    while j < pr.asegs.len() && ((pr.asegs[j].2 as usize) < FLASH_MIN || pr.aseg_vk[j]) {
                        cnt += pr.asegs[j].2;
                        j += 1;
                    }
                    ensure!(seg_attn() || self.kvq == 0, "FE_KVQ needs FE_ATTN_SEG (the split-K kernel reads a bf16 cache)");
                    if seg_attn() {
                        // the run in pieces of <= PART_ROWS / splits rows (the partials budget)
                        let mut a = i;
                        while a < j {
                            let (pr0, mut pc, mut b) = (pr.asegs[a].1, 0u32, a);
                            while b < j && (pc + pr.asegs[b].2) as usize * vk_splits as usize <= PART_ROWS {
                                pc += pr.asegs[b].2;
                                b += 1;
                            }
                            unsafe {
                                s.launch(k_vk, ((b - a) as u32, cfg.n_kv as u32, vk_splits), 32 * (cfg.n_q / cfg.n_kv) as u32,
                                    seg_smem, (self.sc.part_o.ptr(), self.sc.part_ml.ptr(), q, kc.ptr(), vc.ptr(), mp(pr.o_rslot),
                                     mp(pr.o_rpos), mp(pr.o_tbase), mp(pr.o_tmask), mp(pr.o_asrow0), mp(pr.o_aslen), a as u32, pr0,
                                     slot_stride, scale_log2))?;
                                s.launch(k_comb, (pc, cfg.n_q as u32), 256, 0,
                                    (self.sc.mix.ptr(), self.sc.part_o.ptr(), self.sc.part_ml.ptr(), qg, ldqg as u32, vk_splits,
                                     pr0))?;
                            }
                            a = b;
                        }
                        i = j;
                        continue;
                    }
                    let nsplit = attn_splits(cnt as usize, pr.max_pos);
                    unsafe {
                        s.launch(&k.attn_split, (cnt, cfg.n_kv as u32, nsplit as u32), 128, attn_smem as u32,
                            (self.sc.part_o.ptr(), self.sc.part_ml.ptr(), q, kc.ptr(), vc.ptr(), mp(pr.o_rslot),
                             mp(pr.o_rpos), mp(pr.o_tbase), mp(pr.o_tmask), slot_stride, scale_log2, r0))?;
                        s.launch(&k.attn_combine, (cnt, cfg.n_q as u32), 256, 0,
                            (self.sc.mix.ptr(), self.sc.part_o.ptr(), self.sc.part_ml.ptr(), qg, ldqg as u32,
                             nsplit as u32, r0))?;
                    }
                    i = j;
                }
                self.mark("attn core");
                self.resid_linear(lc, o, self.sc.mix.ptr(), cfg.n_q * cfg.head_dim, rows, 0)?;
                self.mark("attn out");
            }
        }
        if let Some(buf) = &self.mid {
            let li = (layer as *const Layer as usize).wrapping_sub(self.w.layers.as_ptr() as usize)
                / std::mem::size_of::<Layer>();
            if li < self.w.layers.len() {
                let ld = (self.w.layers.len() * h) as u32;
                unsafe {
                    s.launch(&k.copy2d, (r32, h32.div_ceil(512)), 256, 0, (buf.ptr().add::<u16>(li * h), ld, x, h32, h32))?
                };
            }
        }
        unsafe { s.launch(&k.rmsnorm, r32, 256, 0, (xn, x, layer.post_norm.ptr(), h32, eps, h32, h32))? };
        self.mark("norm");
        let inter = cfg.inter as u32;
        // prompt rows [d, rows) may take a fused SwiGLU path; decode rows [0, d) always the per-linear one
        let d = if rows > X3_ROWS { lc.dec.min(rows) } else { rows };
        let pr_rows = rows - d;
        let mut fused = false;
        if pr_rows > X3_ROWS {
            let (xn_p, x_p) = (xn.add::<u16>(d * h), x.add::<u16>(d * h));
            if let (Some(gu), Some(dp), Some((fk, fw)), Some(act)) = (&layer.swiglu_fp4, &layer.down.p, lc.fp4, self.fp4_act.as_ref()) {
                // gate|up GEMM + SwiGLU + NVFP4 quant in one kernel, then down on the quantized act (under TP: the
                // rank's down partial, reduced into the residual)
                gu.forward_swiglu(fk, s, fw, act, xn_p, h, pr_rows, dp.g)?;
                match &self.tp {
                    Some(tp) => {
                        dp.gemm(fk, s, act, tp.part_bf16.ptr(), h, pr_rows, None)?;
                        tp.fab.reduce_add_bf16(s, tp.part_bf16.ptr(), x_p, pr_rows * h)?;
                    }
                    None => dp.gemm(fk, s, act, x_p, h, pr_rows, Some(x_p))?,
                }
                fused = true;
            } else if let Some(wi) = &layer.swiglu {
                let act_p = self.sc.act.ptr().add::<u16>(d * cfg.inter);
                unsafe {
                    s.launch(&k.gemm_swiglu, ((pr_rows as u32).div_ceil(128), (2 * inter).div_ceil(128)), 256, GEMM_SMEM,
                        (act_p, xn_p, wi.ptr(), pr_rows as u32, 2 * inter, h32, h32))?;
                }
                let c2 = LinCtx { dec: 0, ..*lc };
                self.resid_linear(&c2, &layer.down, act_p, cfg.inter, pr_rows, d)?;
                fused = true;
            }
        }
        let n_rest = if fused { d } else { rows };
        if n_rest > 0 {
            let c2 = LinCtx { dec: d, ..*lc };
            let p = layer.gate_up.forward(&c2, big, xn, h, n_rest)?;
            let ((g, ldg), (u, ldu)) = (p[0], p[1]);
            ensure!(ldg == ldu, "gate/up strides differ");
            unsafe {
                s.launch(&k.silu_mul, ((inter / 2).div_ceil(256), n_rest as u32), 256, 0,
                    (self.sc.act.ptr(), g, u, ldg as u32, inter))?;
            }
            self.resid_linear(&c2, &layer.down, self.sc.act.ptr(), cfg.inter, n_rest, 0)?;
        }
        self.mark("mlp");
        Ok(())
    }

    /// Greedy next token for each logit row of the last forward.
    pub fn argmax(&self, n: usize) -> Result<Vec<u32>> {
        unsafe {
            self.stream.launch(&self.k.argmax, n as u32, 1024, 0,
                (self.argmax_out.ptr(), self.logits.ptr(), self.cfg.vocab as u32))?;
        }
        self.stream.sync()?;
        let mut v = vec![0u32; n];
        fe_cuda::memcpy_dtoh(&mut v, self.argmax_out.ptr())?;
        Ok(v)
    }

    /// Copy logits rows [0, n) to host as bf16 bits.
    pub fn logits_host(&self, n: usize, dst: &mut [u16]) -> Result<()> {
        let v = self.cfg.vocab;
        assert!(dst.len() >= n * v);
        fe_cuda::memcpy_dtoh(&mut dst[..n * v], self.logits.ptr())?;
        Ok(())
    }
}
