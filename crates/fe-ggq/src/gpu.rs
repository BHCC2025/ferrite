//! GPU side: the k-gguf module and typed launch wrappers.
use anyhow::{bail, Result};
use fe_cuda::{DevPtr, DeviceBuffer, Function, Module, Stream};
use fe_gguf::GgmlType;

pub static K_GGUF_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_gguf.cubin"));

/// Output element type of a dequant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    F32,
    Bf16,
}

/// Kernel-name stem of a type (`ggq_deq_<stem>_f32` etc.).
pub fn stem(ty: GgmlType) -> Option<&'static str> {
    use GgmlType::*;
    Some(match ty {
        Q8_0 => "q8_0",
        Q2_0 => "q2_0",
        IQ4NL => "iq4_nl",
        IQ4XS => "iq4_xs",
        Q4K => "q4_k",
        Q5K => "q5_k",
        Q6K => "q6_k",
        IQ2S => "iq2_s",
        IQ3XXS => "iq3_xxs",
        IQ3S => "iq3_s",
        IQ2XXS => "iq2_xxs",
        IQ1M => "iq1_m",
        IQ2XS => "iq2_xs",
        RepMxfp4 => "mxfp4r",
        F32 => "f32",
        F16 => "f16",
        BF16 => "bf16",
        _ => return None,
    })
}

/// Types with a dp4a GEMV (`ggq_gemv_<stem>_b<B>`).
pub const GEMV_TYPES: [GgmlType; 14] = [GgmlType::Q8_0, GgmlType::Q2_0, GgmlType::IQ4NL, GgmlType::IQ4XS, GgmlType::Q4K,
    GgmlType::Q5K, GgmlType::Q6K, GgmlType::IQ2S, GgmlType::IQ3XXS, GgmlType::IQ3S, GgmlType::IQ2XXS,
    GgmlType::IQ1M, GgmlType::IQ2XS,
    GgmlType::RepMxfp4];

/// q8_1-quantized activation rows on the device: int8 `q` [rows][k], f32 scale `d` [rows][k/32], f32 `s` [rows][k/32]
/// (= the f16-rounded sum of the block's float inputs, ggml's ds.y, for the Q4_K/Q5_K min terms). Allocate once for
/// the largest k and row count; reuse.
pub struct Q8Act {
    pub q: DeviceBuffer<i8>,
    pub d: DeviceBuffer<f32>,
    pub s: DeviceBuffer<f32>,
    pub max_rows: usize,
    pub max_k: usize,
    /// Rows and k of the current contents (set by [`Ggq::quantize_q8_1`]).
    pub rows: usize,
    pub k: usize,
    /// How the min-term sums `s` are formed (Q4_K/Q5_K only): [`SumMode`].
    pub sum_mode: SumMode,
}
impl Q8Act {
    pub fn alloc(max_rows: usize, max_k: usize) -> Result<Q8Act> {
        Ok(Q8Act {
            q: DeviceBuffer::alloc(max_rows * max_k)?,
            d: DeviceBuffer::alloc(max_rows * max_k / 32)?,
            s: DeviceBuffer::alloc(max_rows * max_k / 32)?,
            max_rows,
            max_k,
            rows: 0,
            k: 0,
            sum_mode: SumMode::F16X,
        })
    }
}

/// Min-term activation sums of a q8 block (only Q4_K / Q5_K weights use them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SumMode {
    /// f16-rounded sum of the float inputs: ggml's q8_1 `ds.y` (llama.cpp-identical numerics). Default.
    F16X = 0,
    /// f32 sum of the float inputs.
    F32X = 1,
    /// d * sum(q): consistent with the quantized dot.
    DSumQ = 2,
}

/// Default lanes per weight row for a GEMV over `k` columns: as wide as possible while every lane keeps >= 4
/// sub-blocks of 32 values (k = 640 -> 4, 2560 -> 16, 6144 -> 32).
pub fn default_lpr(k: usize) -> u32 {
    let nsb = k / 32;
    let mut l = 32;
    while l > 1 && nsb / l < 4 {
        l /= 2;
    }
    l as u32
}
/// Default (lanes per row, rows per lane group) of the decode GEMV for a type, K and activation row count (from
/// ggq-check bench on the RTX 5060 Ti).
pub fn default_cfg(ty: GgmlType, k: usize, rows: usize) -> (u32, u32) {
    use GgmlType::*;
    let lpr = default_lpr(k);
    // bench (ggq-check bench, whole tensors > L2): 4-bit+ types are memory-bound up to B5 with one row per lane
    // group; the 2/3-bit IQ types and Q2_0 need the activation loads shared by 2 rows from B3 on, and 4 at B >= 6
    let rpl = match ty {
        Q8_0 => 1,
        Q4K | Q5K | Q6K | IQ4NL | IQ4XS | RepMxfp4 => {
            if rows >= 6 { 4 } else if rows >= 3 { 2 } else { 1 }
        }
        IQ2S => {
            if rows >= 2 { 2 } else { 1 }
        }
        // IQ2_XXS: memory-bound at B1-2; IQ1_M (costlier decode) gains 11% at B1 from 4 rows per lane group
        IQ2XS => {
            if rows >= 2 { 2 } else { 1 }
        }
        IQ2XXS => {
            if rows >= 2 { 2 } else { 1 }
        }
        IQ1M => {
            if rows == 1 || rows >= 6 { 4 } else { 2 }
        }
        _ => {
            if rows >= 6 { 4 } else if rows >= 2 { 2 } else { 1 }
        }
    };
    // env override for tuning on other cards (the bits do not depend on the rows per lane group)
    static O: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    let rpl = O.get_or_init(|| std::env::var("FE_GGQ_RPL").ok().and_then(|v| v.parse().ok())).filter(|_| matches!(ty, Q4K | Q5K | Q6K | Q8_0)).map_or(rpl, |v| v.min(rows as u32).max(1).next_power_of_two().min(4));
    (lpr, rpl)
}
/// Split-K path for long rows with few outputs (env FE_GGQ_BF16_SPLITK=1; off by default: slower than the
/// one-block-per-row kernel on the 5060 Ti, kept for other cards).
pub fn bf16_splitk() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("FE_GGQ_BF16_SPLITK").map(|v| v == "1").unwrap_or(false))
}
/// Output rows per block of the wide BF16 GEMV (env FE_GGQ_BF16_RPB = 1, 2, 4 overrides).
/// B1-2: 4 rows per block (short n); from B3 every x load is shared by up to 9 rows (n = 324 -> 36 blocks = one per SM of the 5060 Ti; 320 -> 40):
/// x re-reads from L2 shrink 9x.
pub fn bf16_rpb(rows: usize, n: usize) -> u32 {
    static W: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    W.get_or_init(|| std::env::var("FE_GGQ_BF16_RPB").ok().and_then(|v| v.parse().ok())).unwrap_or(if rows < 3 {
        if n <= 1024 && n / 4 >= 32 { 4 } else { 1 }
    } else if n % 9 == 0 && n / 9 >= 32 && n <= 1024 {
        9
    } else if n / 8 >= 32 && n <= 1024 {
        8
    } else if n / 2 >= 32 {
        2
    } else {
        1
    })
}
/// Rows per lane group of moe_down (env FE_GGQ_DOWN_RPL: 1, 2, 4; default 4).
pub fn moe_down_rpl() -> u32 {
    static W: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("FE_GGQ_DOWN_RPL").ok().and_then(|v| v.parse().ok()).unwrap_or(4))
}
/// Lanes per h row (= warps per 32-row block) of moe_gu (env FE_GGQ_GU_LPR, default 8).
pub fn moe_gu_lpr() -> u32 {
    static W: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("FE_GGQ_GU_LPR").ok().and_then(|v| v.parse().ok()).unwrap_or(8))
}
/// Wide one-pass GEMV kernels for > 8 rows (env FE_GGQ_GEMV_WIDE=0: chunks of 8 instead; same bits either way).
pub fn gemv_wide() -> bool {
    static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("FE_GGQ_GEMV_WIDE").map_or(true, |v| v != "0"))
}
/// Warps per block of the dense GEMV (env FE_GGQ_WARPS, default [`GEMV_WARPS`]).
pub fn gemv_warps() -> u32 {
    static W: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("FE_GGQ_WARPS").ok().and_then(|v| v.parse().ok()).unwrap_or(GEMV_WARPS))
}
/// Warps per block of the GEMV kernels.
pub const GEMV_WARPS: u32 = 4;

/// All k-gguf kernels, looked up once. Cheap to share (`&Ggq`) across threads/streams.
pub struct Ggq {
    pub module: Module,
    /// DeepSeek's SwiGLU clamp L (bits of an f32): h = silu(min(g, L)) * clamp(u, -L, L) in moe_gu and the fused mmq gate/up; +inf = off.
    swiglu_lim: std::sync::atomic::AtomicU32,
    deq: Vec<(GgmlType, Function, Function)>,
    /// (type, kernels for B = 1..=8, rows per lane group 1 / 2 / 4)
    gemv: Vec<(GgmlType, Vec<[Function; 3]>)>,
    /// Grouped Q5_K GEMV kernels [B-1][rpl index].
    gemvg: Vec<[Function; 3]>,
    /// (type, wide one-pass kernels for B = 12, 16, 20, 24)
    gemvw: Vec<(GgmlType, [Function; 4], [Function; 4])>,
    q8: Function,
    /// (type, gate/up kernel, down kernel)
    moe: Vec<(GgmlType, Function, Function)>,
    combine: Function,
    misses: Function,
    slotk: Function,
    cold: [Function; 3],
    gemm: Vec<(GgmlType, Function)>,
    f2bf: Function,
    mmq: Vec<(GgmlType, Function)>,
    sort: [Function; 3],
    sort_small: Function,
    /// (type, [64-row, 32-row tile kernels]) of the grouped prompt GEMM
    mmq_s: Vec<(GgmlType, [Function; 2])>,
    /// BF16 GEMV: (wide, narrow, split-K) for B = 1..=8
    bf16: Vec<(Function, Function, Function, Function)>,
    /// split-K scratch: partial sums + per-row-group counters (zeroed once; kernels reset them)
    bf16_ws: std::sync::Mutex<Option<(DeviceBuffer<f32>, DeviceBuffer<u32>)>>,
    /// hc fused (down, up) for B = 1..=8
    hc: Vec<(Function, Function)>,
    /// many-row hc_down: stream norm + xn, role-warp down for chunks of 1..=8 rows, scratch (partials, counters)
    hcn: Function,
    hcdw: Vec<Function>,
    hc_ws: (DeviceBuffer<f32>, DeviceBuffer<u32>),
    fx_finish: Function,
    fx_add: Function,
    /// prompt attention: (bits, union, tensor-core, SIMT reference)
    fa: [Function; 7],
    /// dense BF16-weight GEMM (cp.async, 4 stages): BN 128, BN 64
    gemm16: [Function; 2],
    /// (table offset, strided combine) of the ordered prompt MoE down
    ord: [Function; 2],
}


/// Tensor-core prompt attention: shared-memory bytes and q heads per CTA (keep in sync with k-gguf fa.rs).
pub const FA_SMEM: u32 = 2 * (2 * 16 * 256 + 2 * 16 * 32) + 2 * 16 * 264 * 2;
pub const FA_HW: usize = 4;
pub const FA2_SMEM: u32 = 2 * (16 * 272 + 16 * 256 + 2 * 16 * 32) + 16 * 264 * 2;

/// Scratch of [`Ggq::attn_prompt`] for up to `rows` prompt rows at context up to `ctx` cells.
pub struct AttnWs {
    pub rows: usize,
    pub ctx: usize,
    ldb: usize,
    ldl: usize,
    bits: DeviceBuffer<u32>,
    list: DeviceBuffer<u32>,
    cnt: DeviceBuffer<u32>,
}
impl AttnWs {
    pub fn new(rows: usize, ctx: usize) -> Result<AttnWs> {
        if ctx > 4 * 65536 {
            bail!("AttnWs: ctx {ctx} > 262144 (16-bit block ids)");
        }
        let nb = ctx / 4 + 1;
        let ldb = nb.div_ceil(32);
        // a tile's union: <= 16 rows x 513 blocks (QSA: 512 + the open block; dense rows see <= 513 blocks) windows
        // of 2 blocks, + pad; 2 words per window
        let ldl = 2 * ((16 * 513).min(nb.div_ceil(2)) + 2);
        let tiles = rows.div_ceil(16);
        Ok(AttnWs { rows, ctx, ldb, ldl, bits: DeviceBuffer::zeroed(rows * ldb)?, list: DeviceBuffer::zeroed(tiles * ldl)?,
            cnt: DeviceBuffer::zeroed(tiles)? })
    }
    /// Tile lists of the last call ([window, row mask] pairs), the padded window count per tile, the list stride.
    pub fn lists(&self) -> Result<(Vec<u32>, Vec<u32>, usize)> {
        Ok((self.list.to_vec()?, self.cnt.to_vec()?, self.ldl))
    }
}

/// Workspace of [`Ggq::moe_sort`] for up to `max_pairs` pairs and `nexp` experts (<= 512).
pub struct SortWs {
    pub nexp: usize,
    pub max_pairs: usize,
    pub counts: DeviceBuffer<u32>,
    pub chunk_off: DeviceBuffer<u32>,
    /// u32 [nexp + 1]: sorted rows of expert e = seg_off[e] .. seg_off[e + 1]
    pub seg_off: DeviceBuffer<u32>,
    /// u32 [2 * max_tiles]: (expert, first sorted row) of each MMQ_BM-row tile; `ntiles[0]` valid
    pub tiles: DeviceBuffer<u32>,
    /// u32 [3]: the 128-row tile count, and (after [`Ggq::moe_sort_small`]) the 64- and 32-row tile counts
    pub ntiles: DeviceBuffer<u32>,
    /// u32 [4 * nexp]: after `moe_sort_small` the (expert, first row) lists of the 64-row tiles then the 32-row tiles
    pub tiles_s: DeviceBuffer<u32>,
    /// set by `moe_sort_small`, cleared by `moe_sort`: the grouped GEMMs then also launch the small-tile kernels
    pub small: std::sync::atomic::AtomicBool,
    /// u32 [max_pairs]: token (activation row) and pair (output row) of each sorted row
    pub xidx: DeviceBuffer<u32>,
    pub yidx: DeviceBuffer<u32>,
}
impl SortWs {
    pub fn alloc(max_pairs: usize, nexp: usize) -> Result<SortWs> {
        let nchunk = max_pairs.div_ceil(4096).max(1);
        Ok(SortWs {
            nexp,
            max_pairs,
            counts: DeviceBuffer::alloc(nchunk * nexp)?,
            chunk_off: DeviceBuffer::alloc(nchunk * nexp)?,
            seg_off: DeviceBuffer::alloc(nexp + 1)?,
            tiles: DeviceBuffer::alloc(2 * Self::max_tiles(max_pairs, nexp))?,
            ntiles: DeviceBuffer::alloc(3)?,
            tiles_s: DeviceBuffer::alloc(4 * nexp)?,
            small: std::sync::atomic::AtomicBool::new(false),
            xidx: DeviceBuffer::alloc(max_pairs)?,
            yidx: DeviceBuffer::alloc(max_pairs)?,
        })
    }
    /// Upper bound of the 64-row tile count (sizes the GEMM grid).
    pub fn max_tiles(np: usize, nexp: usize) -> usize {
        np.div_ceil(MMQ_BM) + nexp
    }
}
/// Int8 GEMM dynamic shared memory (2 stages).
/// Bench-only ablation bits for the int8 GEMM (env FE_MMQ_DBG, shifted into the kernel's `add` argument).
pub fn mmq_dbg() -> u32 {
    static W: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("FE_MMQ_DBG").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) << 8
}
/// Rows per int8-GEMM tile (kernel BM), threads per block = 2 * MMQ_BM.
pub const MMQ_BM: usize = 128;
/// Threads per int8-GEMM block: 2 * BM consumers + 128 producers.
pub const MMQ_THREADS: u32 = 2 * MMQ_BM as u32 + 128;
/// Weight rows per int8-GEMM tile (kernel BN): n % MMQ_BN == 0.
pub const MMQ_BN: usize = 64;
pub const MMQ_NSTAGE: u32 = 2;
/// Dynamic shared memory of the BM-row tile kernels (BM = 128, 64, 32).
pub const fn mmq_smem_bm(bm: u32) -> u32 {
    MMQ_NSTAGE * ((bm + MMQ_BN as u32) * 144 + 2 * bm * 16 + 4 * MMQ_BN as u32 * 16)
}
pub const MMQ_SMEM: u32 = MMQ_NSTAGE * ((MMQ_BM + MMQ_BN) as u32 * 144 + 2 * MMQ_BM as u32 * 16 + 4 * MMQ_BN as u32 * 16);

/// Types with a prompt GEMM (`ggq_gemm_<stem>`): the GEMV types plus BF16, F16, F32 weights.
pub const GEMM_TYPES: [GgmlType; 17] = [GgmlType::Q8_0, GgmlType::Q2_0, GgmlType::IQ4NL, GgmlType::IQ4XS, GgmlType::Q4K,
    GgmlType::Q5K, GgmlType::Q6K, GgmlType::IQ2S, GgmlType::IQ3XXS, GgmlType::IQ3S, GgmlType::IQ2XXS, GgmlType::IQ1M,
    GgmlType::IQ2XS, GgmlType::RepMxfp4, GgmlType::BF16, GgmlType::F16, GgmlType::F32];
const GEMM_SMEM: u32 = 4 * 64 * 40 * 2;

impl Ggq {
    /// Set the SwiGLU clamp used by [`Ggq::moe_gu_q8`] / [`Ggq::moe_gu`] / [`Ggq::mmq_moe_gu`] (default +inf = no clamp, the old bits).
    /// `CpuMoe::swiglu_limit` must be set to the same value for CPU == GPU bits.
    pub fn set_swiglu_limit(&self, limit: f32) {
        self.swiglu_lim.store(limit.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
    pub fn swiglu_limit(&self) -> f32 {
        f32::from_bits(self.swiglu_lim.load(std::sync::atomic::Ordering::Relaxed))
    }
    /// Load the module into the current context.
    pub fn load() -> Result<Ggq> {
        let module = Module::load(K_GGUF_CUBIN)?;
        let mut deq = vec![];
        use GgmlType::*;
        for ty in [Q8_0, Q2_0, IQ4NL, IQ4XS, Q4K, Q5K, Q6K, IQ2S, IQ3XXS, IQ3S, IQ2XXS, IQ1M, IQ2XS, RepMxfp4, F32, F16, BF16] {
            let s = stem(ty).unwrap();
            deq.push((ty, module.function(&format!("ggq_deq_{s}_f32"))?, module.function(&format!("ggq_deq_{s}_bf16"))?));
        }
        let mut gemv = vec![];
        for ty in GEMV_TYPES {
            let s = stem(ty).unwrap();
            let mut fs = vec![];
            for b in 1..=8 {
                fs.push([module.function(&format!("ggq_gemv_{s}_b{b}"))?, module.function(&format!("ggq_gemv_{s}_b{b}_r2"))?,
                    module.function(&format!("ggq_gemv_{s}_b{b}_r4"))?]);
            }
            gemv.push((ty, fs));
        }
        let mut gemvg = vec![];
        for b in 1..=8 {
            gemvg.push([module.function(&format!("ggq_gemv_q5_k_g_b{b}_r1"))?, module.function(&format!("ggq_gemv_q5_k_g_b{b}_r2"))?,
                module.function(&format!("ggq_gemv_q5_k_g_b{b}_r4"))?]);
        }
        let mut gemvw = vec![];
        for ty in GEMV_TYPES {
            let st = stem(ty).unwrap();
            let f = |b: u32| module.function(&format!("ggq_gemv_{st}_b{b}_w"));
            let f2 = |b: u32| module.function(&format!("ggq_gemv_{st}_b{b}_w2"));
            gemvw.push((ty, [f(12)?, f(16)?, f(20)?, f(24)?], [f2(12)?, f2(16)?, f2(20)?, f2(24)?]));
        }
        let q8 = module.function("ggq_q8_1")?;
        let mut moe = vec![];
        for ty in GEMV_TYPES {
            let s = stem(ty).unwrap();
            moe.push((ty, module.function(&format!("ggq_moe_gu_{s}"))?, module.function(&format!("ggq_moe_down_{s}"))?));
        }
        let combine = module.function("ggq_moe_combine")?;
        let misses = module.function("ggq_moe_misses")?;
        let slotk = module.function("ggq_slot_owner")?;
        let cold = [module.function("ggq_cold_claim")?, module.function("ggq_cold_copy")?, module.function("ggq_cold_restore")?];
        let mut gemm = vec![];
        for ty in GEMM_TYPES {
            gemm.push((ty, module.function(&format!("ggq_gemm_{}", stem(ty).unwrap()))?));
        }
        let f2bf = module.function("ggq_f32_to_bf16")?;
        let mut mmq = vec![];
        for ty in GEMV_TYPES {
            let f = module.function(&format!("ggq_mmq_{}", stem(ty).unwrap()))?;
            f.set_max_dynamic_smem(MMQ_SMEM)?;
            mmq.push((ty, f));
        }
        let mut mmq_s = vec![];
        for ty in GEMV_TYPES {
            let st = stem(ty).unwrap();
            let (f64_, f32_) = (module.function(&format!("ggq_mmq_{st}_bm64"))?, module.function(&format!("ggq_mmq_{st}_bm32"))?);
            f64_.set_max_dynamic_smem(mmq_smem_bm(64))?;
            f32_.set_max_dynamic_smem(mmq_smem_bm(32))?;
            // more than one small-tile CTA per SM needs the shared array carved out of L1
            f64_.set_smem_carveout(100)?;
            f32_.set_smem_carveout(100)?;
            mmq_s.push((ty, [f64_, f32_]));
        }
        let sort_small = module.function("ggq_moe_sort_scan_small")?;
        let sort = [module.function("ggq_moe_sort_count")?, module.function("ggq_moe_sort_scan")?,
            module.function("ggq_moe_sort_scatter")?];
        let mut bf16 = vec![];
        for b in 1..=8 {
            bf16.push((module.function(&format!("ggq_bf16_wide_b{b}"))?, module.function(&format!("ggq_bf16_narrow_b{b}"))?,
                module.function(&format!("ggq_bf16_splitk_b{b}"))?, module.function(&format!("ggq_bf16_wider_b{b}"))?));
        }
        let mut hc = vec![];
        for b in 1..=8 {
            hc.push((module.function(&format!("ggq_hc_down_b{b}"))?, module.function(&format!("ggq_hc_up_b{b}"))?));
        }
        let fx_finish = module.function("ggq_fx_finish")?;
        let fx_add = module.function("ggq_fx_add")?;
        let hcn = module.function("ggq_hc_norm")?;
        let mut hcdw = vec![];
        for b in 1..=8 {
            hcdw.push(module.function(&format!("ggq_hc_dw_b{b}"))?);
        }
        let hc_ws = (DeviceBuffer::zeroed(HC_DW_MAXN.div_ceil(36) * 36 * 32 * 8)?, DeviceBuffer::zeroed(64)?);

        let ord = [module.function("ggq_tab_offset")?, module.function("ggq_moe_combine_ld")?];
        let gemm16 = [module.function("ggq_gemm16_bf16_n128")?, module.function("ggq_gemm16_bf16_n64")?];
        gemm16[0].set_max_dynamic_smem(4 * 2 * 128 * 80)?;
        gemm16[1].set_max_dynamic_smem(4 * (128 + 64) * 80)?;
        let fa = [module.function("ggq_fa_bits")?, module.function("ggq_fa_union")?, module.function("ggq_fa_tc")?,
            module.function("ggq_fa_ref_simt")?, module.function("ggq_fa_ref_pattn")?,
            module.function("ggq_fa_mmaprobe")?, module.function("ggq_fa_tc2")?];
        fa[4].set_max_dynamic_smem(512 * 16 + 4 * 2 * 512 * 16 + 2 * 4 * 16 * 4)?;
        Ok(Ggq { swiglu_lim: std::sync::atomic::AtomicU32::new(f32::INFINITY.to_bits()), module, deq, gemv, q8, moe, combine, misses, slotk, cold, gemm, f2bf, mmq, sort, sort_small, mmq_s, bf16, bf16_ws: std::sync::Mutex::new(None), hc,
            hcn, hcdw, hc_ws, fx_finish, fx_add, fa, gemm16, ord, gemvw, gemvg })
    }

    /// Dequantize `n_values` values (a whole number of blocks; for F32/F16/BF16 a multiple of 8) of type `ty` at
    /// `src` into `dst` (f32 or bf16, contiguous). Works on any block-aligned range: a tensor, rows, an expert slice.
    pub fn dequant(&self, s: &Stream, ty: GgmlType, src: DevPtr, n_values: usize, dst: DevPtr, out: Out) -> Result<()> {
        let (bs, _) = ty.block();
        let unit = bs.max(8);
        if n_values % unit != 0 {
            bail!("dequant {}: {n_values} values is not a multiple of {unit}", ty.name());
        }
        let Some((_, f32k, bf16k)) = self.deq.iter().find(|(t, _, _)| *t == ty) else {
            bail!("dequant: type {} not supported", ty.name());
        };
        let chunks = (n_values / 8) as u64;
        let grid = chunks.div_ceil(256) as u32;
        let f = if out == Out::F32 { f32k } else { bf16k };
        unsafe { s.launch(f, grid, 256, 0, (dst, src, chunks))? };
        Ok(())
    }

    /// Quantize `rows` activation rows of `k` values (f32, or bf16 when `bf16`; row stride `ld` elements) into `a`.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_q8_1(&self, s: &Stream, x: DevPtr, ld: usize, bf16: bool, rows: usize, k: usize, a: &mut Q8Act)
        -> Result<()> {
        if k % 32 != 0 || rows > a.max_rows || k > a.max_k {
            bail!("quantize_q8_1: rows {rows} k {k} (max {} x {}, k % 32 == 0)", a.max_rows, a.max_k);
        }
        let grid = ((k / 32).div_ceil(4) as u32, rows.min(65535) as u32);
        unsafe {
            s.launch(&self.q8, grid, 128, 0, (x, ld as u32, bf16 as u32, a.q.ptr(), a.d.ptr(), a.s.ptr(), k as u32,
                a.sum_mode as u32, rows as u32))?
        };
        a.rows = rows;
        a.k = k;
        Ok(())
    }

    /// Decode GEMV: `y[b][n] (= or +=) sum_k W[n][k] * x[b][k]` for the `a.rows` (1..=8) activation rows in `a`
    /// (quantized with [`Ggq::quantize_q8_1`], `a.k` = K). `w` points at row 0 of `n` rows of type `ty`, consecutive
    /// rows `row_bytes` apart (`ty.row_bytes(K)` for a plain tensor). `y` is f32 [rows][n]. `lpr` = lanes per row
    /// (None: [`default_lpr`]).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv(&self, s: &Stream, ty: GgmlType, w: DevPtr, row_bytes: usize, n: usize, a: &Q8Act, y: DevPtr, add: bool,
        lpr: Option<u32>) -> Result<()> {
        let (dl, _) = default_cfg(ty, a.k, a.rows);
        let lpr = lpr.unwrap_or(dl);
        if a.rows > 8 {
            // more rows than the B <= 8 kernels take: wide one-pass kernels (12 / 16 / 20 / 24 rows) and a <= 8-row
            // rest; each row's float order depends only on K and lpr, so every row has the bits it gets at B = 1
            let mut r0 = 0;
            while r0 < a.rows {
                let left = a.rows - r0;
                let wb = [24usize, 20, 16, 12].into_iter().find(|&b| b <= left && gemv_wide());
                if let Some(b) = wb {
                    self.gemv_wide_raw(s, ty, w, row_bytes, n, a, r0, b, y.add::<f32>(r0 * n), add, lpr)?;
                    r0 += b;
                } else {
                    let rb = left.min(8);
                    let (_, dr) = default_cfg(ty, a.k, rb);
                    self.gemv_raw(s, ty, w, row_bytes, n, a, r0, rb, y.add::<f32>(r0 * n), add, lpr, dr)?;
                    r0 += rb;
                }
            }
            return Ok(());
        }
        let (_, dr) = default_cfg(ty, a.k, a.rows);
        self.gemv_cfg(s, ty, w, row_bytes, n, a, y, add, lpr, dr)
    }

    /// Q5_K GEMV over `groups` independent matrices in one launch: group g multiplies the weights at `w + g * wg` (n rows each) by the activation
    /// rows `g * rows_per_group ..` of `a` and writes f32 `y + g * n + b * yld` (rows b < rows_per_group). `a` holds groups * rows_per_group rows
    /// of K. Same per-element float order as `gemv` on each group (rows <= 8 per group).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_groups_q5k(&self, s: &Stream, w: DevPtr, row_bytes: usize, wg: usize, n: usize, groups: usize, a: &Q8Act, rows_per_group: usize,
        y: DevPtr, yld: usize) -> Result<()> {
        if rows_per_group == 0 || rows_per_group > 8 || a.rows != groups * rows_per_group {
            bail!("gemv_groups_q5k: {} act rows for {groups} groups x {rows_per_group}", a.rows);
        }
        let (lpr, rpl) = default_cfg(GgmlType::Q5K, a.k, rows_per_group);
        let ri = match rpl { 1 => 0, 2 => 1, _ => 2 };
        let warps = gemv_warps();
        let gx = (n as u32).div_ceil(warps * (32 / lpr) * rpl);
        unsafe {
            s.launch(&self.gemvg[rows_per_group - 1][ri], (gx, groups as u32), 32 * warps, 0, (y, w, row_bytes as u64, a.q.ptr(), a.d.ptr(), a.s.ptr(),
                a.k as u32, n as u32, lpr, 0u32, wg as u64, rows_per_group as u32, n as u32, yld as u32))?
        };
        Ok(())
    }

    /// Cold-expert staging around one layer's MoE kernels (see `ggq_cold_claim`): `stage_begin` claims and copies, `stage_end` restores the tables.
    #[allow(clippy::too_many_arguments)]
    pub fn cold_begin(&self, s: &Stream, ids: DevPtr, np: usize, tabs: [DevPtr; 3], lo: u64, hi: u64, counter: DevPtr, slot_e: DevPtr, slot_orig: DevPtr,
        stage: DevPtr, slot_bytes: usize, nslots: usize, sz: [usize; 3]) -> Result<()> {
        s.memset_d8(counter, 0, 4)?;
        unsafe {
            s.launch(&self.cold[0], np as u32, 32, 0, (ids, np as u32, tabs[0], tabs[1], tabs[2], lo, hi, counter, slot_e, slot_orig, stage, slot_bytes as u64,
                nslots as u32, sz[0] as u64, sz[1] as u64))?;
            s.launch(&self.cold[1], (nslots as u32, 16u32), 256, 0, (counter, slot_orig, stage, slot_bytes as u64, nslots as u32, sz[0] as u64, sz[1] as u64,
                sz[2] as u64, 16u32))?;
        }
        Ok(())
    }

    pub fn cold_end(&self, s: &Stream, tabs: [DevPtr; 3], counter: DevPtr, slot_e: DevPtr, slot_orig: DevPtr, nslots: usize) -> Result<()> {
        unsafe { s.launch(&self.cold[2], nslots as u32, 32, 0, (counter, slot_e, slot_orig, tabs[0], tabs[1], tabs[2], nslots as u32))? };
        Ok(())
    }

    /// A wide one-pass GEMV (rows = 12, 16, 20 or 24) on activation rows r0 .. r0 + rows of `a`.
    #[allow(clippy::too_many_arguments)]
    fn gemv_wide_raw(&self, s: &Stream, ty: GgmlType, w: DevPtr, row_bytes: usize, n: usize, a: &Q8Act, r0: usize, rows: usize,
        y: DevPtr, add: bool, lpr: u32) -> Result<()> {
        let Some((_, fs, fs2)) = self.gemvw.iter().find(|(t, _, _)| *t == ty) else {
            bail!("gemv: type {} not supported", ty.name());
        };
        let i = match rows { 12 => 0, 16 => 1, 20 => 2, 24 => 3, _ => bail!("gemv wide: rows {rows}") };
        // two weight rows per lane group share every activation load (FE_GGQ_WIDE_R=1: one); same bits either way
        let r2 = std::env::var("FE_GGQ_WIDE_R").map_or(true, |v| v != "1");
        let (f, rpl) = if r2 { (&fs2[i], 2) } else { (&fs[i], 1) };
        let (k, nb) = (a.k, a.k / 32);
        let warps = gemv_warps();
        let grid = (n as u32).div_ceil(warps * (32 / lpr) * rpl);
        unsafe {
            s.launch(f, grid, 32 * warps, 0, (y, w, row_bytes as u64, a.q.ptr().add::<i8>(r0 * k), a.d.ptr().add::<f32>(r0 * nb),
                a.s.ptr().add::<f32>(r0 * nb), k as u32, n as u32, lpr, add as u32))?
        };
        Ok(())
    }

    /// [`Ggq::gemv_cfg`] on activation rows r0 .. r0 + rows (<= 8) of `a`; `y` points at the first output row.
    #[allow(clippy::too_many_arguments)]
    fn gemv_raw(&self, s: &Stream, ty: GgmlType, w: DevPtr, row_bytes: usize, n: usize, a: &Q8Act, r0: usize, rows: usize,
        y: DevPtr, add: bool, lpr: u32, rpl: u32) -> Result<()> {
        let Some((_, fs)) = self.gemv.iter().find(|(t, _)| *t == ty) else {
            bail!("gemv: type {} not supported", ty.name());
        };
        let ri = match rpl {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => bail!("gemv: rpl {rpl} (1, 2, 4)"),
        };
        let (k, nb) = (a.k, a.k / 32);
        let warps = gemv_warps();
        let grid = (n as u32).div_ceil(warps * (32 / lpr) * rpl);
        unsafe {
            s.launch(&fs[rows - 1][ri], grid, 32 * warps, 0, (y, w, row_bytes as u64, a.q.ptr().add::<i8>(r0 * k),
                a.d.ptr().add::<f32>(r0 * nb), a.s.ptr().add::<f32>(r0 * nb), k as u32, n as u32, lpr, add as u32))?
        };
        Ok(())
    }

    /// [`Ggq::gemv`] with an explicit config: `lpr` lanes per row group, `rpl` weight rows per lane group (1, 2, 4;
    /// every activation load is shared by `rpl` rows).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_cfg(&self, s: &Stream, ty: GgmlType, w: DevPtr, row_bytes: usize, n: usize, a: &Q8Act, y: DevPtr,
        add: bool, lpr: u32, rpl: u32) -> Result<()> {
        let Some((_, fs)) = self.gemv.iter().find(|(t, _)| *t == ty) else {
            bail!("gemv: type {} not supported", ty.name());
        };
        if a.rows == 0 {
            bail!("gemv: no activation rows");
        }
        if a.rows > 8 {
            let mut r0 = 0;
            while r0 < a.rows {
                let rb = (a.rows - r0).min(8);
                self.gemv_raw(s, ty, w, row_bytes, n, a, r0, rb, y.add::<f32>(r0 * n), add, lpr, rpl)?;
                r0 += rb;
            }
            return Ok(());
        }
        if a.k % ty.block().0 != 0 {
            bail!("gemv {}: k {} not a multiple of the block", ty.name(), a.k);
        }
        let ri = match rpl {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => bail!("gemv: rpl {rpl} (1, 2, 4)"),
        };
        let warps = gemv_warps();
        let rows_per_block = warps * (32 / lpr) * rpl;
        let grid = (n as u32).div_ceil(rows_per_block);
        unsafe {
            s.launch(&fs[a.rows - 1][ri], grid, 32 * warps, 0, (y, w, row_bytes as u64, a.q.ptr(), a.d.ptr(),
                a.s.ptr(), a.k as u32, n as u32, lpr, add as u32))?
        };
        Ok(())
    }

    /// MoE gate/up + SwiGLU for `n_tok` (1..=8) tokens x `topk` experts, fused with the q8 quantization of h:
    /// `h[p][r] = silu(gate_e[r] . x[t]) * (up_e[r] . x[t])` (p = t * topk + k, e = ids[p]) goes straight into `hq`
    /// (`Q8Act` with >= n_tok * topk rows of n_ff; its rows/k are set) = the activations of [`Ggq::moe_down`].
    /// `x` = q8 rows of the token inputs (`x.rows` >= n_tok, `x.k` = n_embd). `gtab` / `utab`: u64 device tables
    /// [n_expert] of each expert's slice address (0 = not on the card: those pairs are skipped, their rows left
    /// untouched). Pairs sharing an expert are computed together (weights read once). `h_f32`: optional f32 copy of
    /// h [n_tok * topk][n_ff]. `row_bytes` = `ty.row_bytes(n_embd)`. n_ff % 32 == 0. `hq.sum_mode` applies.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gu_q8(&self, s: &Stream, ty: GgmlType, gtab: DevPtr, utab: DevPtr, row_bytes: usize, ids: DevPtr,
        n_tok: usize, topk: usize, x: &Q8Act, n_ff: usize, hq: &mut Q8Act, h_f32: Option<DevPtr>, lpr: Option<u32>)
        -> Result<()> {
        let np = n_tok * topk;
        if np > hq.max_rows || n_ff > hq.max_k {
            bail!("moe_gu_q8: hq holds {} x {}, need {np} x {n_ff}", hq.max_rows, hq.max_k);
        }
        self.moe_gu_raw(s, ty, gtab, utab, row_bytes, ids, n_tok, topk, x, n_ff,
            (hq.q.ptr(), hq.d.ptr(), hq.s.ptr()), h_f32.unwrap_or(DevPtr(0)), lpr, hq.sum_mode)?;
        hq.rows = np;
        hq.k = n_ff;
        Ok(())
    }

    /// Unfused form: only the f32 `h` [n_tok * topk][n_ff] (quantize it yourself for moe_down).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gu(&self, s: &Stream, ty: GgmlType, gtab: DevPtr, utab: DevPtr, row_bytes: usize, ids: DevPtr,
        n_tok: usize, topk: usize, x: &Q8Act, n_ff: usize, h: DevPtr, lpr: Option<u32>) -> Result<()> {
        self.moe_gu_raw(s, ty, gtab, utab, row_bytes, ids, n_tok, topk, x, n_ff, (DevPtr(0), DevPtr(0), DevPtr(0)), h,
            lpr, SumMode::F16X)
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_gu_raw(&self, s: &Stream, ty: GgmlType, gtab: DevPtr, utab: DevPtr, row_bytes: usize, ids: DevPtr,
        n_tok: usize, topk: usize, x: &Q8Act, n_ff: usize, hq: (DevPtr, DevPtr, DevPtr), hf: DevPtr, lpr: Option<u32>,
        mode: SumMode) -> Result<()> {
        let Some((_, f, _)) = self.moe.iter().find(|(t, _, _)| *t == ty) else {
            bail!("moe_gu: type {} not supported", ty.name());
        };
        if n_tok == 0 || x.rows < n_tok || n_ff % 32 != 0 {
            bail!("moe_gu: n_tok {n_tok} (x has {} rows), n_ff {n_ff} (% 32)", x.rows);
        }
        let lpr = lpr.unwrap_or(moe_gu_lpr());
        let (k, nb, hb) = (x.k, x.k / 32, n_ff / 32);
        // chunks of <= 8 tokens (the kernel groups at most 8 tokens per expert); per-pair bits do not depend on it
        let mut t0 = 0;
        while t0 < n_tok {
            let nt = (n_tok - t0).min(8);
            let (p0, np) = (t0 * topk, (nt * topk) as u32);
            let hfp = if hf.0 == 0 { hf } else { hf.add::<f32>(p0 * n_ff) };
            // block = lpr warps = 32 rows
            unsafe {
                s.launch(f, ((n_ff / 32) as u32, np), 32 * lpr, 0, (hq.0.add::<i8>(p0 * n_ff), hq.1.add::<f32>(p0 * hb),
                    hq.2.add::<f32>(p0 * hb), hfp, gtab, utab, row_bytes as u64, ids.add::<i32>(p0), np, topk as u32,
                    x.q.ptr().add::<i8>(t0 * k), x.d.ptr().add::<f32>(t0 * nb), x.s.ptr().add::<f32>(t0 * nb), k as u32,
                    n_ff as u32, lpr, mode as u32, self.swiglu_limit()))?
            };
            t0 += nt;
        }
        Ok(())
    }

    /// Which pairs need the CPU: `count` (device u32), `pairs` (u32 [n_tok * topk], the first `count` valid, in pair
    /// order) and `slot` (i32 [n_tok * topk]: the pair's index in `pairs`, -1 = on the card), from the gate table
    /// (`tab[e] == 0` = not on the card) and `ids`. Read `count` + `pairs` back (one D2H), run those experts with
    /// `CpuMoe`, upload their outputs as rows `0..count` of a CPU buffer, and pass it with `slot` to
    /// [`Ggq::moe_combine_cpu`].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_misses(&self, s: &Stream, tab: DevPtr, ids: DevPtr, n_tok: usize, topk: usize, count: DevPtr,
        pairs: DevPtr, slot: DevPtr) -> Result<()> {
        unsafe { s.launch(&self.misses, 1, 32, 0, (count, pairs, slot, tab, ids, (n_tok * topk) as u32))? };
        Ok(())
    }

    /// MoE down: `y[p][r] = down_e[r] . h[p]` (f32 [n_tok * topk][n_out]); `h` = q8 rows of the SwiGLU outputs, one
    /// per pair (`h.rows` = n_tok * topk, `h.k` = n_ff). `dtab` as in [`Ggq::moe_gu`] (0 = skipped: write those `y`
    /// rows yourself, e.g. from [`crate::cpu::CpuMoe`]). `row_bytes` = `ty.row_bytes(n_ff)`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down(&self, s: &Stream, ty: GgmlType, dtab: DevPtr, row_bytes: usize, ids: DevPtr, n_tok: usize,
        topk: usize, h: &Q8Act, n_out: usize, y: DevPtr, lpr: Option<u32>) -> Result<()> {
        let Some((_, _, f)) = self.moe.iter().find(|(t, _, _)| *t == ty) else {
            bail!("moe_down: type {} not supported", ty.name());
        };
        let lpr = lpr.unwrap_or_else(|| std::env::var("FE_GGQ_DOWN_LPR").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| default_lpr(h.k)));
        let rpl = moe_down_rpl();
        let rpb = GEMV_WARPS * (32 / lpr) * rpl;
        let (k, nb) = (h.k, h.k / 32);
        let mut t0 = 0;
        while t0 < n_tok {
            let nt = (n_tok - t0).min(8);
            let (p0, np) = (t0 * topk, (nt * topk) as u32);
            unsafe {
                s.launch(f, ((n_out as u32).div_ceil(rpb), np), 32 * GEMV_WARPS, 0, (y.add::<f32>(p0 * n_out), dtab,
                    row_bytes as u64, ids.add::<i32>(p0), np, topk as u32, h.q.ptr().add::<i8>(p0 * k), h.d.ptr().add::<f32>(p0 * nb),
                    h.s.ptr().add::<f32>(p0 * nb), k as u32, n_out as u32, lpr, rpl))?
            };
            t0 += nt;
        }
        Ok(())
    }

    /// `out[t][i] (= or +=) sum_k w[t * topk + k] * y[t * topk + k][i]` (fixed order, deterministic).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine(&self, s: &Stream, out: DevPtr, y: DevPtr, w: DevPtr, n_tok: usize, topk: usize, n: usize,
        add: bool) -> Result<()> {
        self.moe_combine_cpu(s, out, y, w, n_tok, topk, n, add, DevPtr(0), DevPtr(0))
    }

    /// [`Ggq::moe_combine`] where pair p with `slot[p] >= 0` takes its row from `ycpu[slot[p]]` (f32 [count][n]).
    #[allow(clippy::too_many_arguments)]
    /// Pair routing on the card: slot[q] = -1 (this card's expert) or the row of the expert card `owner[ids[q]]` in the results buffer.
    pub fn slot_owner(&self, s: &Stream, slot: DevPtr, owner: DevPtr, ids: DevPtr, np: usize, stride: usize) -> Result<()> {
        unsafe { s.launch(&self.slotk, 1, 32, 0, (slot, owner, ids, np as u32, stride as u32))? };
        Ok(())
    }

    pub fn moe_combine_cpu(&self, s: &Stream, out: DevPtr, y: DevPtr, w: DevPtr, n_tok: usize, topk: usize, n: usize,
        add: bool, ycpu: DevPtr, slot: DevPtr) -> Result<()> {
        unsafe {
            s.launch(&self.combine, ((n as u32).div_ceil(256), n_tok as u32), 256, 0, (out, y, w, topk as u32, n as u32,
                add as u32, ycpu, slot))?
        };
        Ok(())
    }

    /// Prompt GEMM: `y[i][n] (= or +=) sum_k x[i][k] * W[n][k]`, x bf16 [m][ldx], y f32 [m][ldy], W `n` rows of type
    /// `ty` (`row_bytes` apart). n % 64 == 0, k % 32 == 0 (and a multiple of the block), any m.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(&self, s: &Stream, ty: GgmlType, y: DevPtr, ldy: usize, x: DevPtr, ldx: usize, w: DevPtr,
        row_bytes: usize, m: usize, n: usize, k: usize, add: bool) -> Result<()> {
        self.gemm_seg(s, ty, y, ldy, x, ldx, w, row_bytes, m, n, k, DevPtr(0), DevPtr(0), DevPtr(0), DevPtr(0), 1, add)
    }

    /// Segmented (grouped-expert) prompt GEMM. Segment `s` (0..nseg) covers rows `seg_off[s] .. seg_off[s+1]`
    /// (device u32 [nseg + 1]) of the index arrays `xidx` (activation row of each) and `yidx` (output row of each),
    /// device u32, with weights at `segw[s]` (device u64; 0 = skip). `m` = an upper bound of the rows of any
    /// segment (sizes the grid; blocks past a segment's end exit). Null `xidx`/`yidx` = identity. A null `seg_off`
    /// means one dense segment of `m` rows with weights `w`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_seg(&self, s: &Stream, ty: GgmlType, y: DevPtr, ldy: usize, x: DevPtr, ldx: usize, w: DevPtr,
        row_bytes: usize, m: usize, n: usize, k: usize, xidx: DevPtr, yidx: DevPtr, segw: DevPtr, seg_off: DevPtr,
        nseg: usize, add: bool) -> Result<()> {
        let Some((_, f)) = self.gemm.iter().find(|(t, _)| *t == ty) else {
            bail!("gemm: type {} not supported", ty.name());
        };
        if n % 64 != 0 || k % 32 != 0 || k % ty.block().0.max(8) != 0 {
            bail!("gemm {}: n {n} k {k}", ty.name());
        }
        if m == 0 {
            return Ok(());
        }
        // dense BF16 weights: the cp.async kernel (bit-identical); FE_GEMM16=0 for the generic one
        if ty == GgmlType::BF16 && seg_off.0 == 0 && xidx.0 == 0 && yidx.0 == 0 && nseg == 1 && ldx % 8 == 0 && row_bytes % 16 == 0
            && x.0 % 16 == 0 && w.0 % 16 == 0 && std::env::var("FE_GEMM16").map_or(true, |v| v != "0") {
            let (f, bn, smem) = if n <= 1024 { (&self.gemm16[1], 64, 4 * (128 + 64) * 80) } else { (&self.gemm16[0], 128, 4 * 2 * 128 * 80) };
            unsafe {
                s.launch(f, (n.div_ceil(bn) as u32, m.div_ceil(128) as u32), 256, smem as u32, (y, ldy as u32, x, ldx as u32, w,
                    row_bytes as u64, m as u32, n as u32, k as u32, add as u32))?
            };
            return Ok(());
        }
        let grid = ((n / 64) as u32, m.div_ceil(64) as u32, nseg as u32);
        unsafe {
            s.launch(f, grid, 128, GEMM_SMEM, (y, ldy as u32, x, ldx as u32, w, row_bytes as u64, m as u32, k as u32,
                xidx, yidx, segw, seg_off, add as u32))?
        };
        Ok(())
    }

    /// f32 -> bf16 rows: dst [rows][ldd] from src [rows][lds], n columns.
    #[allow(clippy::too_many_arguments)]
    pub fn f32_to_bf16(&self, s: &Stream, dst: DevPtr, ldd: usize, src: DevPtr, lds: usize, rows: usize, n: usize)
        -> Result<()> {
        unsafe { s.launch(&self.f2bf, ((n as u32).div_ceil(256), rows as u32), 256, 0, (dst, ldd as u32, src, lds as u32, n as u32))? };
        Ok(())
    }

    /// Sort `n_tok * topk` pairs (`ids` i32 [n_tok][topk], experts < ws.nexp) by expert, stably, into `ws`
    /// (seg_off, tile list, xidx = token, yidx = pair). Deterministic; 3 launches, no host sync.
    pub fn moe_sort(&self, s: &Stream, ids: DevPtr, n_tok: usize, topk: usize, ws: &SortWs) -> Result<()> {
        let np = n_tok * topk;
        if np > ws.max_pairs || ws.nexp > 512 || np == 0 {
            bail!("moe_sort: {np} pairs (max {}), {} experts (max 512)", ws.max_pairs, ws.nexp);
        }
        let nchunk = np.div_ceil(4096) as u32;
        let (np32, ne) = (np as u32, ws.nexp as u32);
        ws.small.store(false, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            s.launch(&self.sort[0], nchunk, 256, 0, (ws.counts.ptr(), ids, np32, ne))?;
            s.launch(&self.sort[1], 1, 512, 0, (ws.counts.ptr(), nchunk, ne, ws.chunk_off.ptr(), ws.seg_off.ptr(),
                ws.tiles.ptr(), ws.ntiles.ptr()))?;
            s.launch(&self.sort[2], nchunk, 32, 0, (ids, np32, ne, topk as u32, ws.chunk_off.ptr(), ws.xidx.ptr(),
                ws.yidx.ptr()))?;
        }
        Ok(())
    }

    /// [`Ggq::moe_sort`] with small tiles for the tail of every expert: the grouped GEMMs after it (`mmq_moe`, `mmq_moe_gu`,
    /// `mmq_moe_down_acc`, `moe_down_ordered`) run an expert's full 128-row tiles as before and its remaining rows (1..127)
    /// in one 32-, 64- or 128-row tile, up to 3 launches (empty experts get no tile). The bits do not depend on the tile
    /// rows. `moe_sort` switches back to the plain tiles.
    pub fn moe_sort_small(&self, s: &Stream, ids: DevPtr, n_tok: usize, topk: usize, ws: &SortWs) -> Result<()> {
        let np = n_tok * topk;
        if np > ws.max_pairs || ws.nexp > 512 || np == 0 {
            bail!("moe_sort_small: {np} pairs (max {}), {} experts (max 512)", ws.max_pairs, ws.nexp);
        }
        let nchunk = np.div_ceil(4096) as u32;
        let (np32, ne) = (np as u32, ws.nexp as u32);
        unsafe {
            s.launch(&self.sort[0], nchunk, 256, 0, (ws.counts.ptr(), ids, np32, ne))?;
            s.launch(&self.sort_small, 1, 512, 0, (ws.counts.ptr(), nchunk, ne, ws.chunk_off.ptr(), ws.seg_off.ptr(),
                ws.tiles.ptr(), ws.ntiles.ptr(), ws.tiles_s.ptr()))?;
            s.launch(&self.sort[2], nchunk, 32, 0, (ids, np32, ne, topk as u32, ws.chunk_off.ptr(), ws.xidx.ptr(),
                ws.yidx.ptr()))?;
        }
        ws.small.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// The grouped GEMM launches for the tile lists of `ws`: the 128-row kernel `f128` over `ws.tiles`, plus (after
    /// `moe_sort_small`) the 64- and 32-row kernels over their lists. `mk(tiles, ntiles)` builds the kernel arguments.
    #[allow(clippy::too_many_arguments)]
    fn mmq_grouped<A: fe_cuda::KernelArgs>(&self, s: &Stream, ty: GgmlType, f128: &Function, grid_x: u32, ws: &SortWs,
        n_pairs: usize, mk: impl Fn(DevPtr, DevPtr) -> A) -> Result<()> {
        let grid_y = SortWs::max_tiles(n_pairs, ws.nexp) as u32;
        unsafe { s.launch(f128, (grid_x, grid_y), MMQ_THREADS, MMQ_SMEM, mk(ws.tiles.ptr(), ws.ntiles.ptr()))? };
        if ws.small.load(std::sync::atomic::Ordering::Relaxed) {
            let Some((_, fs)) = self.mmq_s.iter().find(|(t, _)| *t == ty) else {
                bail!("mmq: small tiles for {} not supported", ty.name());
            };
            for (i, bm) in [64u32, 32].into_iter().enumerate() {
                unsafe {
                    s.launch(&fs[i], (grid_x, ws.nexp as u32), 2 * bm + 128, mmq_smem_bm(bm),
                        mk(ws.tiles_s.ptr().add::<u32>(i * 2 * ws.nexp), ws.ntiles.ptr().add::<u32>(i + 1)))?
                };
            }
        }
        Ok(())
    }

    /// Int8 prompt GEMM, dense: `y[i][n] (= or +=) W[n] . x[i]` for the `x.rows` q8 rows in `x` (K = x.k), W `n`
    /// rows of `ty` `row_bytes` apart. n % 64 == 0, K % 128 == 0. y f32 [rows][ldy].
    #[allow(clippy::too_many_arguments)]
    pub fn mmq(&self, s: &Stream, ty: GgmlType, y: DevPtr, ldy: usize, x: &Q8Act, w: DevPtr, row_bytes: usize,
        n: usize, add: bool) -> Result<()> {
        let f = self.mmq_fn(ty, n, x.k)?;
        if x.rows == 0 {
            return Ok(());
        }
        let z = DevPtr(0);
        unsafe {
            s.launch(f, ((n / MMQ_BN) as u32, x.rows.div_ceil(MMQ_BM) as u32), MMQ_THREADS, MMQ_SMEM, (y, ldy as u32, x.q.ptr(), x.d.ptr(),
                x.s.ptr(), w, row_bytes as u64, x.rows as u32, x.k as u32, z, z, z, z, z, z, add as u32 | mmq_dbg(), z, z, z, self.swiglu_limit()))?
        };
        Ok(())
    }

    /// Prompt MoE down + combine, deterministic without atomics and without a full pair buffer: after
    /// [`Ggq::moe_sort`] of all n_tok * topk pairs into `ws`, the down GEMM runs over `ncol`-column blocks of the
    /// output (`ybuf` f32 [n_tok * topk][ncol], `tabw` a device u64 [nexp] scratch), each block combined at once in
    /// fixed slot order per token: out[t][c] (= or +=) sum_k wts[t][k] * down_e(t,k)(h[t * topk + k])[c]. Each output
    /// depends only on its own pairs: identical bits for any `ncol`, any prompt split, and the same combine
    /// arithmetic as decode (`ggq_moe_combine`). `out` f32 [n_tok][n_out]. n_out % ncol == 0, ncol % 64 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down_ordered(&self, s: &Stream, ty: GgmlType, out: DevPtr, hq: &Q8Act, dtab: DevPtr, row_bytes: usize,
        n_out: usize, wts: DevPtr, n_tok: usize, topk: usize, ws: &SortWs, ncol: usize, ybuf: DevPtr, tabw: DevPtr,
        add: bool) -> Result<()> {
        let f = self.mmq_fn(ty, ncol, hq.k)?;
        if n_out % ncol != 0 {
            bail!("moe_down_ordered: n_out {n_out} % ncol {ncol}");
        }
        let np = n_tok * topk;
        for c0 in (0..n_out).step_by(ncol) {
            unsafe {
                s.launch(&self.ord[0], (ws.nexp as u32).div_ceil(256), 256, 0, (tabw, dtab, ws.nexp as u32,
                    (c0 * row_bytes) as u64))?;
            }
            self.mmq_grouped(s, ty, f, (ncol / MMQ_BN) as u32, ws, np, |tl, nt| (ybuf, ncol as u32, hq.q.ptr(), hq.d.ptr(),
                hq.s.ptr(), DevPtr(0), row_bytes as u64, 0u32, hq.k as u32, ws.yidx.ptr(), ws.yidx.ptr(), tabw,
                ws.seg_off.ptr(), tl, nt, mmq_dbg(), DevPtr(0), DevPtr(0), DevPtr(0), self.swiglu_limit()))?;
            unsafe {
                s.launch(&self.ord[1], ((ncol as u32).div_ceil(256), n_tok as u32), 256, 0, (out.add::<f32>(c0), n_out as u32,
                    ybuf, ncol as u32, wts, topk as u32, ncol as u32, add as u32))?;
            }
        }
        Ok(())
    }

    /// Int8 prompt GEMM over experts after [`Ggq::moe_sort`]: sorted row r (expert e) computes
    /// `y[ws.yidx[r]] (= or +=) W_e . x[ws.xidx[r]]` with `W_e` at `tab[e]` (u64 device table, 0 = skip: those
    /// rows untouched). `x`: q8 rows indexed by `xidx` (tokens, or pairs for the down GEMM with `xidx` = pair:
    /// pass `by_pair`); `n_pairs` = n_tok * topk of the sort.
    #[allow(clippy::too_many_arguments)]
    pub fn mmq_moe(&self, s: &Stream, ty: GgmlType, y: DevPtr, ldy: usize, x: &Q8Act, tab: DevPtr, row_bytes: usize,
        n: usize, ws: &SortWs, n_pairs: usize, by_pair: bool, add: bool) -> Result<()> {
        let f = self.mmq_fn(ty, n, x.k)?;
        let xi = if by_pair { ws.yidx.ptr() } else { ws.xidx.ptr() };
        self.mmq_grouped(s, ty, f, (n / MMQ_BN) as u32, ws, n_pairs, |tl, nt| (y, ldy as u32, x.q.ptr(), x.d.ptr(), x.s.ptr(),
            DevPtr(0), row_bytes as u64, 0u32, x.k as u32, xi, ws.yidx.ptr(), tab, ws.seg_off.ptr(), tl,
            nt, add as u32 | mmq_dbg(), DevPtr(0), DevPtr(0), DevPtr(0), self.swiglu_limit()))
    }

    /// Prompt MoE down fused with the weighted combine, deterministic: for sorted row r (pair p = yidx[r], token
    /// t = p / topk), `acc[t][i] += fx(wts[p] * (D_e[i] . h[p]))` where fx = round(v * 2^40) as i64 and the adds are
    /// integer atomics, so the result has the same bits in any order (and across expert-group passes). `acc` i64
    /// [tokens][lda], zero before the first pass (`fx_finish` re-zeroes it). Per-term range |v| < 2^22; terms out of
    /// range are dropped and counted in `status` (u32, device; check it is 0). Skipped experts (table 0) add nothing.
    /// Then [`Ggq::fx_finish`] converts to f32.
    #[allow(clippy::too_many_arguments)]
    pub fn mmq_moe_down_acc(&self, s: &Stream, ty: GgmlType, acc: DevPtr, lda: usize, h: &Q8Act, tab: DevPtr,
        row_bytes: usize, n: usize, ws: &SortWs, n_pairs: usize, topk: usize, wts: DevPtr, status: DevPtr) -> Result<()> {
        let f = self.mmq_fn(ty, n, h.k)?;
        self.mmq_grouped(s, ty, f, (n / MMQ_BN) as u32, ws, n_pairs, |tl, nt| (acc, lda as u32, h.q.ptr(), h.d.ptr(),
            h.s.ptr(), status, row_bytes as u64, topk as u32, h.k as u32, ws.yidx.ptr(), ws.yidx.ptr(), tab,
            ws.seg_off.ptr(), tl, nt, mmq_dbg(), DevPtr(0), DevPtr(0), wts, self.swiglu_limit()))
    }

    /// `out[i] (= or +=) acc[i] * 2^-40` for `n_total` values (i64 fixed point from `mmq_moe_down_acc`), and zero
    /// `acc` for the next layer.
    /// Prompt attention (drop-in for fe-fnext's `fx_attn` on prompt rows), tensor cores: rows at cells
    /// prm[0] + r over the q8 caches (kq / vq int8 [cell][nkv][256], ks / vs f32 [cell][nkv][8]); q / out f32
    /// [rows][nq][256]; gate at gate[r * ldg + h * gstride + d]; QSA selection as `fx_topk` writes it (sel
    /// [rows][ldsel] cells in any order, seln[r] cells, 0 = dense causal). Keys in ascending cell order:
    /// deterministic and independent of the selection order. Head dim 256, QSA block 4 cells, nq / nkv % 4 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prompt(&self, s: &Stream, ws: &AttnWs, out: DevPtr, q: DevPtr, kq: DevPtr, ks: DevPtr, vq: DevPtr,
        vs: DevPtr, gate: DevPtr, ldg: usize, gstride: usize, nq: usize, nkv: usize, prm: DevPtr, scale: f32, sel: DevPtr,
        seln: DevPtr, ldsel: usize, rows: usize) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if rows > ws.rows || nq % nkv != 0 || (nq / nkv) % FA_HW != 0 {
            bail!("attn_prompt: rows {rows} (ws {}), nq {nq}, nkv {nkv}", ws.rows);
        }
        let tiles = rows.div_ceil(16) as u32;
        unsafe {
            s.launch(&self.fa[0], rows as u32, 256, 0, (ws.bits.ptr(), sel, seln, ldsel as u32, prm, ws.ldb as u32))?;
            s.launch(&self.fa[1], tiles, 256, 0, (ws.list.ptr(), ws.cnt.ptr(), ws.bits.ptr(), ws.ldb as u32, ws.ldl as u32,
                prm, rows as u32))?;
            let v1 = std::env::var("FE_FA_V1").is_ok_and(|v| v == "1");
            let (f, smem) = if v1 { (&self.fa[2], FA_SMEM) } else { (&self.fa[6], FA2_SMEM) };
            s.launch(f, (tiles, (nq / FA_HW) as u32), 32 * FA_HW as u32, smem, (out, q, kq, ks, vq, vs, gate,
                ldg as u32, gstride as u32, nq as u32, nkv as u32, prm, scale, ws.list.ptr(), ws.cnt.ptr(), ws.ldl as u32,
                rows as u32))?;
        }
        Ok(())
    }
    /// The SIMT reference (a copy of fe-fnext's `fx_attn`), same arguments; for tests and timing.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_ref_simt(&self, s: &Stream, out: DevPtr, q: DevPtr, kq: DevPtr, ks: DevPtr, vq: DevPtr, vs: DevPtr,
        gate: DevPtr, ldg: usize, gstride: usize, nq: usize, nkv: usize, prm: DevPtr, scale: f32, sel: DevPtr, seln: DevPtr,
        ldsel: usize, rows: usize) -> Result<()> {
        unsafe {
            s.launch(&self.fa[3], (rows as u32, nq as u32), 256, 0, (out, q, kq, ks, vq, vs, gate, ldg as u32, gstride as u32,
                nq as u32, nkv as u32, prm, scale, sel, seln, ldsel as u32))?;
        }
        Ok(())
    }

    /// Tensor-pipe rate probe (bench): mode 0 f16 k16, 1 f16 k8, 2 s8 k32; grid x 128 threads x n x 8 mma.
    pub fn mma_probe(&self, s: &Stream, out: DevPtr, mode: u32, n: u32, grid: u32) -> Result<()> {
        unsafe { s.launch(&self.fa[5], grid, 128, 0, (out, mode, n, 3u32))? };
        Ok(())
    }
    /// fe-fnext's `fx_pattn` (copy, wt-core 6d91439), same arguments as [`Ggq::attn_ref_simt`]; for comparisons.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_ref_pattn(&self, s: &Stream, out: DevPtr, q: DevPtr, kq: DevPtr, ks: DevPtr, vq: DevPtr, vs: DevPtr,
        gate: DevPtr, ldg: usize, gstride: usize, nq: usize, nkv: usize, prm: DevPtr, scale: f32, sel: DevPtr, seln: DevPtr,
        ldsel: usize, rows: usize) -> Result<()> {
        unsafe {
            s.launch(&self.fa[4], (rows as u32, nkv as u32), 128, 512 * 16 + 4 * 2 * 512 * 16 + 2 * 4 * 16 * 4, (out, q, kq, ks, vq,
                vs, gate, ldg as u32, gstride as u32, nq as u32, nkv as u32, prm, scale, sel, seln, ldsel as u32))?;
        }
        Ok(())
    }

    /// dst (i64 fixed-point) += src, src zeroed: merges the partial sums of two cards.
    pub fn fx_add(&self, s: &Stream, dst: DevPtr, src: DevPtr, n_total: usize) -> Result<()> {
        unsafe { s.launch(&self.fx_add, (n_total as u32).div_ceil(256), 256, 0, (dst, src, n_total as u32))? };
        Ok(())
    }

    pub fn fx_finish(&self, s: &Stream, out: DevPtr, acc: DevPtr, n_total: usize, add: bool) -> Result<()> {
        unsafe { s.launch(&self.fx_finish, (n_total as u32).div_ceil(256), 256, 0, (out, acc, n_total as u32, add as u32))? };
        Ok(())
    }

    /// Fused prompt gate/up over experts after [`Ggq::moe_sort`]: for sorted row r (expert e, token xidx[r],
    /// pair yidx[r]) `h = silu(G_e . x) * (U_e . x)` (n_ff values), written straight as q8 rows of `hq` indexed by
    /// pair (`hq.rows` = n_pairs, `hq.k` = n_ff), the input of `mmq_moe(.., by_pair = true)` for down. `gtab` /
    /// `utab`: u64 tables (0 = skip). n_ff % 32 == 0, K % 128 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn mmq_moe_gu(&self, s: &Stream, ty: GgmlType, x: &Q8Act, gtab: DevPtr, utab: DevPtr, row_bytes: usize,
        n_ff: usize, ws: &SortWs, n_pairs: usize, hq: &mut Q8Act) -> Result<()> {
        let f = self.mmq_fn(ty, 64, x.k)?;
        if n_ff % 32 != 0 || n_pairs > hq.max_rows || n_ff > hq.max_k {
            bail!("mmq_moe_gu: n_ff {n_ff}, hq {} x {}", hq.max_rows, hq.max_k);
        }
        // GU mode: y = hd, ldy = n_ff, w = hs
        let (hd, hs, hqq) = (hq.d.ptr(), hq.s.ptr(), hq.q.ptr());
        self.mmq_grouped(s, ty, f, (n_ff / 32) as u32, ws, n_pairs, |tl, nt| (hd, n_ff as u32, x.q.ptr(), x.d.ptr(),
            x.s.ptr(), hs, row_bytes as u64, 0u32, x.k as u32, ws.xidx.ptr(), ws.yidx.ptr(), gtab, ws.seg_off.ptr(),
            tl, nt, mmq_dbg(), utab, hqq, DevPtr(0), self.swiglu_limit()))?;
        hq.rows = n_pairs;
        hq.k = n_ff;
        Ok(())
    }

    fn mmq_fn(&self, ty: GgmlType, n: usize, k: usize) -> Result<&Function> {
        let Some((_, f)) = self.mmq.iter().find(|(t, _)| *t == ty) else {
            bail!("mmq: type {} not supported", ty.name());
        };
        if n % MMQ_BN != 0 || k % 128 != 0 {
            bail!("mmq {}: n {n} % {MMQ_BN}, k {k} % 128", ty.name());
        }
        Ok(f)
    }

    /// BF16-weight decode GEMV, exact (bf16 x f32, f32 accumulate): `y[b][i] (= or +=) sum_k W[i][k] * x[b][k]` for
    /// `rows` (1..=8) f32 activation rows `x` (row stride `ldx`), W bf16 [n][k] row-major, y f32 [rows][n].
    /// k % 8 == 0, x 16-byte aligned rows. Long rows (k >= 2048) use one block per output row; short rows 8 lanes
    /// per row with x in shared memory. Launched programmatically (PDL, `FE_GGQ_PDL=0` off): the kernel prefetches
    /// its weights into L2 before waiting for the previous grid, so a chain of these overlaps launch and ramp.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_bf16(&self, s: &Stream, w: DevPtr, n: usize, k: usize, x: DevPtr, ldx: usize, rows: usize, y: DevPtr,
        add: bool) -> Result<()> {
        if rows == 0 || k % 8 != 0 {
            bail!("gemv_bf16: rows {rows}, k {k} (% 8)");
        }
        if rows > 8 {
            let mut r0 = 0;
            while r0 < rows {
                let rb = (rows - r0).min(8);
                self.gemv_bf16(s, w, n, k, x.add::<f32>(r0 * ldx), ldx, rb, y.add::<f32>(r0 * n), add)?;
                r0 += rb;
            }
            return Ok(());
        }
        let (wide0, narrow, splitk, wider) = &self.bf16[rows - 1];
        let rpb = bf16_rpb(rows, n);
        let wide = if rpb >= 8 { wider } else { wide0 };
        let args = (y, w, x, ldx as u32, n as u32, k as u32, add as u32, rpb);
        if k >= 2048 && n <= 2048 && k % 1024 == 0 && bf16_splitk() {
            // split-K: K slices of 1024, row groups of 32; scratch sized for n <= 2048, k <= 32768, 8 rows
            let ksl = 1024u32;
            let nks = (k as u32) / ksl;
            let mut g = self.bf16_ws.lock().unwrap();
            if g.is_none() {
                *g = Some((DeviceBuffer::zeroed(32 * 8 * 2048)?, DeviceBuffer::zeroed(64)?));
            }
            let (part, cnt) = g.as_ref().unwrap();
            if nks > 32 {
                bail!("gemv_bf16: k {k} too long for the split-K scratch");
            }
            let smem = (rows as u32 * ksl * 4).max(16);
            if smem > 48 * 1024 {
                splitk.set_max_dynamic_smem(smem)?;
            }
            unsafe {
                crate::pdl::launch(s, splitk, ((n as u32).div_ceil(32), nks), 256, smem, (y, w, x, ldx as u32, n as u32,
                    k as u32, add as u32, ksl, part.ptr(), cnt.ptr()))?
            };
        } else if k >= 2048 {
            unsafe { crate::pdl::launch(s, wide, ((n as u32).div_ceil(rpb), 1), 256, 4096, args)? };
        } else {
            let smem = (rows * k * 4) as u32;
            if smem > 48 * 1024 {
                narrow.set_max_dynamic_smem(smem)?;
            }
            unsafe { crate::pdl::launch(s, narrow, ((n as u32).div_ceil(32), 1), 256, smem, args)? };
        }
        Ok(())
    }

    /// Fused hyper-connection mix, first half (qwen4exp): for `rows` (1..=8) residual rows `res` f32 [rows][4 x 2560]
    /// (stride `ldres`): xn = per-stream RMSNorm(res) * nw (eps), `lo[r][o] = scale * (W[o] . xn)` for o < n (W bf16
    /// [n][10240]: down 320 + inject 4 = 324, or 320 for the head), then SiLU on o < n_silu (320). lo f32 [rows][ldlo].
    /// xn f32 [rows][10240] goes to `xn_out` (needed by [`Ggq::hc_up`]). Exact f32 math, PDL launch.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_down(&self, s: &Stream, lo: DevPtr, ldlo: usize, w: DevPtr, n: usize, res: DevPtr, ldres: usize, nw: DevPtr,
        eps: f32, scale: f32, n_silu: usize, xn_out: DevPtr, rows: usize) -> Result<()> {
        self.hc_down_pf(s, lo, ldlo, w, n, res, ldres, nw, eps, scale, n_silu, xn_out, rows, DevPtr::NULL)
    }

    /// [`Ggq::hc_down`] that also prefetches the following hc_up's weights `up` ([10240][320] bf16) into L2 (spread over
    /// its blocks, before its own reads), so the DRAM stream runs through the kernel boundary. Same bits.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_down_pf(&self, s: &Stream, lo: DevPtr, ldlo: usize, w: DevPtr, n: usize, res: DevPtr, ldres: usize, nw: DevPtr,
        eps: f32, scale: f32, n_silu: usize, xn_out: DevPtr, rows: usize, up: DevPtr) -> Result<()> {
        if rows == 0 {
            bail!("hc_down: rows {rows}");
        }
        if rows > HC_MAX_ROWS {
            let mut r0 = 0;
            while r0 < rows {
                let rb = (rows - r0).min(HC_MAX_ROWS);
                let xo = if xn_out.0 == 0 { xn_out } else { xn_out.add::<f32>(r0 * 10240) };
                let pf = if r0 == 0 { up } else { DevPtr::NULL };
                self.hc_down_pf(s, lo.add::<f32>(r0 * ldlo), ldlo, w, n, res.add::<f32>(r0 * ldres), ldres, nw, eps, scale, n_silu, xo, rb, pf)?;
                r0 += rb;
            }
            return Ok(());
        }
        if rows >= hc_dw_min() && xn_out.0 != 0 && n <= HC_DW_MAXN {
            // many rows: norm + xn once, then the role-warp GEMV (1/8 of each xn row per block); same bits
            let bc = hc_dw_bc(rows);
            unsafe {
                crate::pdl::launch(s, &self.hcn, (4, rows as u32), 256, 64, (xn_out, res, ldres as u32, nw, eps))?;
                crate::pdl::launch(s, &self.hcdw[bc - 1], (8 * (n as u32).div_ceil(36), 1), 288, 16, (self.hc_ws.0.ptr(),
                    self.hc_ws.1.ptr(), lo, ldlo as u32, w, xn_out, n as u32, n_silu as u32, scale, up, rows as u32))?;
            }
            return Ok(());
        }
        let bc = hc_bc(rows);
        let cfg = hc_down_cfg(rows);
        let (r, rg) = HC_DOWN_CFGS[cfg as usize];
        unsafe {
            crate::pdl::launch(s, &self.hc[bc - 1].0, ((n as u32).div_ceil(r * rg), 1), 256 * rg, 8192, (lo, ldlo as u32, w,
                res, ldres as u32, nw, xn_out, n as u32, n_silu as u32, scale, eps, cfg, up, rows as u32))?
        };
        Ok(())
    }

    /// Fused hyper-connection mix, second half: gate = U . lo[r][0..320] (U bf16 [10240][320]),
    /// `mixed[r][j] = mscale * sum_s xn[r][s*2560 + j] * sigmoid(gate[s*2560 + j])` (s = 0..4 in order), mixed f32
    /// [rows][ldm >= 2560]. Exact f32 math, PDL launch.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_up(&self, s: &Stream, mixed: DevPtr, ldm: usize, u: DevPtr, lo: DevPtr, ldlo: usize, xn: DevPtr, ldxn: usize,
        mscale: f32, rows: usize) -> Result<()> {
        if rows == 0 {
            bail!("hc_up: rows {rows}");
        }
        if rows > HC_MAX_ROWS {
            let mut r0 = 0;
            while r0 < rows {
                let rb = (rows - r0).min(HC_MAX_ROWS);
                self.hc_up(s, mixed.add::<f32>(r0 * ldm), ldm, u, lo.add::<f32>(r0 * ldlo), ldlo, xn.add::<f32>(r0 * ldxn), ldxn, mscale, rb)?;
                r0 += rb;
            }
            return Ok(());
        }
        let bc = hc_bc(rows);
        let jb = hc_up_jb(rows);
        unsafe {
            crate::pdl::launch(s, &self.hc[bc - 1].1, ((2560 / jb) as u32, 1), 8 * jb as u32, (rows * 320 * 4) as u32, (mixed, ldm as u32, u,
                lo, ldlo as u32, xn, ldxn as u32, mscale, rows as u32))?
        };
        Ok(())
    }
}

/// hc_down / hc_up rows per launch (more run as several launches; the bits do not depend on the split).
const HC_MAX_ROWS: usize = 32;
/// Largest hc_down output count for the role-warp path (its partial scratch).
const HC_DW_MAXN: usize = 324;

/// Rows from which hc_down takes the norm + role-warp path (FE_GGQ_HC_DW overrides; 0 = never).
fn hc_dw_min() -> usize {
    match env_usize("FE_GGQ_HC_DW") {
        Some(0) => usize::MAX,
        Some(v) => v,
        None => 2,
    }
}

/// Activation rows per chunk of the role-warp path: up to 8, else equal chunks of <= 8 (FE_GGQ_HC_DBC overrides).
fn hc_dw_bc(rows: usize) -> usize {
    if let Some(v) = env_usize("FE_GGQ_HC_DBC") {
        return v.clamp(1, 8).min(rows);
    }
    if rows <= 8 { rows } else { rows.div_ceil(rows.div_ceil(8)) }
}

/// hc_down block configs (rows per 256-thread group, groups per block), indexed by the kernel's cfg argument.
const HC_DOWN_CFGS: [(u32, u32); 6] = [(4, 1), (9, 1), (8, 1), (3, 3), (3, 1), (1, 3)];

fn env_usize(k: &str) -> Option<usize> {
    std::env::var(k).ok().and_then(|v| v.parse().ok())
}

/// Activation rows per in-kernel chunk: all of them up to 8, else equal chunks of <= 8 (FE_GGQ_HC_BC overrides).
fn hc_bc(rows: usize) -> usize {
    if let Some(v) = env_usize("FE_GGQ_HC_BC") {
        return v.clamp(1, 8).min(rows);
    }
    if rows <= 8 { rows } else { rows.div_ceil(rows.div_ceil(8)) }
}

/// hc_down block config by row count (FE_GGQ_HC_CFG overrides, see HC_DOWN_CFGS).
fn hc_down_cfg(rows: usize) -> u32 {
    if let Some(v) = env_usize("FE_GGQ_HC_CFG") {
        return (v as u32).min(5);
    }
    let _ = rows;
    1
}

/// hc_up values of j per block (8 lanes each): 32 up to 5 rows, 16 above (FE_GGQ_HC_UPJ overrides, a divisor of 2560
/// up to 128).
fn hc_up_jb(rows: usize) -> usize {
    env_usize("FE_GGQ_HC_UPJ").filter(|v| 2560 % v == 0 && *v <= 128).unwrap_or(if rows <= 5 { 32 } else { 16 })
}
