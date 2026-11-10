//! The DFlash2 drafter on the B70 (the CUDA original: crates/fe-dflash).
//!
//! Context: for every target position the drafter needs, the target's outputs of layers 5, 19, 33, 47, 61 (fe-xe
//! [`Taps`], kernel df_tap) are concatenated ([25600] bf16), projected by `fc`, RMS-normed (`hidden_norm`) and turned into
//! each draft layer's K / V (k_proj + k_norm + RoPE, v_proj) at the position: [`DFlash::absorb`]. Only the last `window`
//! (2048) positions before a draft are ever read (sliding-window attention), so the draft K / V caches are rings of 4096.
//! Draft: a block [anchor, mask x (b - 1)] at positions pos.. runs the 5 draft layers (Qwen3 GQA 32 / 8, head 128,
//! non-causal inside the block, every sublayer wrapped in a two-tap grouped dynamic conv), the final norm, the target's
//! lm_head over the first `span` vocabulary ids on rows 1.., top-16 per row, and the candidate selector's greedy walk from
//! the anchor: [`DFlash::draft`] returns b - 1 proposals. The target verifies them (greedy accept-while-match), so the
//! drafter's numerics never change the output.
//!
//! Linears: XQ tensors (crate::quant, RTN) on kx-xq's GEMV through the engine's registry, with this drafter's own prepass
//! scratch (fc has 25600 inputs, more than the engine's). The draft model's RMSNorm is plain `w * x` (Qwen3), which is
//! what kx-core's rmsnorm_r computes (fe-xe uploads the target's norms as 1 + w).
use crate::quant::{self, Packed};
use anyhow::{ensure, Context as _, Result};
use fe_xe::engine::{Engine, Taps};
use fe_xe::kx::{Kx, Source};
use fe_xq::pack::{split_for, variant_for, VARIANTS};
use fe_xq::st::SafeTensors;
use fe_ze::{DevPtr, DeviceBuffer, Local};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub static KX_DFLASH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_dflash.spv"));
/// IGC flags of the module (the matrix-engine draft attention keeps its accumulators in registers with 256 GRF)
pub const KX_FLAGS: &str = "-cl-intel-256-GRF-per-thread";

pub const H: usize = 5120;
pub const NQ: usize = 32;
pub const NKV: usize = 8;
pub const HD: usize = 128;
pub const INTER: usize = 17408;
pub const CONV: usize = 1280;
pub const RANK: usize = 256;
pub const TOPK: usize = 16;
/// rows of one draft block or absorb call (the XQ decode GEMV's 16)
pub const MAXR: usize = 16;
/// draft K / V ring positions per layer (>= window + block)
pub const RING: usize = 4096;
const TOPK_PARTS: u32 = 16;
/// row width of the absorb's fused k | v of all 5 layers
const KVA: usize = 5 * 2 * NKV * HD;

/// A drafter linear on the device (XQ, rotated with quant::su_for(k)).
pub struct Lin {
    pub w: DeviceBuffer<u32>,
    pub bits: u32,
    pub n: u32,
    pub k: u32,
    pub split: u32,
}

struct Layer {
    in_norm: DeviceBuffer<u16>,
    post_norm: DeviceBuffer<u16>,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    q_norm: DeviceBuffer<u16>,
    k_norm: DeviceBuffer<u16>,
    attn_base: DeviceBuffer<u16>,
    attn_proj: Lin,
    mlp_base: DeviceBuffer<u16>,
    mlp_proj: Lin,
    gate: Lin,
    up: Lin,
    down: Lin,
    kc: DeviceBuffer<u16>,
    vc: DeviceBuffer<u16>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DraftTimes {
    pub drafts: u64,
    pub absorbs: u64,
    pub absorb_rows: u64,
}

pub struct DFlash {
    kx: Kx,
    su: HashMap<u32, DeviceBuffer<u16>>,
    layers: Vec<Layer>,
    fc: Lin,
    /// every layer's k | v for absorbs (5 x 2048 rows: the absorb's K / V of all layers in one GEMV)
    kv_all: Lin,
    hidden_norm: DeviceBuffer<u16>,
    norm: DeviceBuffer<u16>,
    hproj: Lin,
    /// FE_DF_HEAD=q4: a 4-bit XQ (RTN) copy of the target's BF16 lm_head rows [0, span) as the draft head (0.34 GB read a draft
    /// instead of the 6-bit GPTQ head's 0.50 GB); None = the target's own head
    head4: Option<Lin>,
    pred_cb: DeviceBuffer<u16>,
    succ_cb: DeviceBuffer<u16>,
    /// host copies of the selector codebooks (verify trees; FE_DF_TREE=0 skips them)
    pub cb_host: Option<(Vec<u16>, Vec<u16>)>,
    // scratch
    xt: DeviceBuffer<u16>,
    xs: DeviceBuffer<f32>,
    res: DeviceBuffer<u16>,
    hn: DeviceBuffer<u16>,
    hc: DeviceBuffer<u16>,
    coef: DeviceBuffer<u16>,
    qb: DeviceBuffer<u16>,
    kb: DeviceBuffer<u16>,
    vb: DeviceBuffer<u16>,
    kva: DeviceBuffer<u16>,
    at: DeviceBuffer<u16>,
    g: DeviceBuffer<u16>,
    u: DeviceBuffer<u16>,
    part: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    tpv: DeviceBuffer<f32>,
    tpi: DeviceBuffer<u32>,
    pub tv: DeviceBuffer<f32>,
    pub ti: DeviceBuffer<u32>,
    pub gate_sel: DeviceBuffer<u16>,
    out: DeviceBuffer<u32>,
    ids: DeviceBuffer<u32>,
    rope: DeviceBuffer<f32>,
    /// pinned staging: [0] the draft's ids + RoPE rows, [1] an absorb's RoPE rows (queued copies; a staging region is rewritten
    /// only after a sync that follows its last copy: every draft ends with one, and the verify window's upload syncs)
    stage: [fe_ze::HostBuffer<f32>; 2],
    /// the target's feature taps of the current window (fe-xe writes them, see [`DFlash::taps`]): [tap_cap][5 x 5120]
    pub taps_buf: DeviceBuffer<u16>,
    pub tap_cap: usize,
    inv: Vec<f32>,
    pub block: usize,
    pub layer_ids: Vec<usize>,
    pub mask_id: u32,
    pub window: u32,
    pub eps: f32,
    /// the draft head scores the first `span` vocabulary ids (FE_DF_VOCAB, default 131072; the target still verifies over all)
    pub span: usize,
    nsplit: u32,
    /// what the prepass scratch holds: (x, su, k, m)
    prep: Option<(u64, u64, u32, u32)>,
    pub times: DraftTimes,
    pub bytes: usize,
}

fn bf16s(st: &SafeTensors, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
    st.u16s(name, "BF16", shape).with_context(|| format!("drafter tensor {name}"))
}

/// The drafter linears in load order: (name, n, k).
fn linears(nl: usize) -> Vec<(String, usize, usize)> {
    let mut v = vec![("fc.weight".to_string(), H, 5 * H), ("candidate_selector.hidden_projection.weight".to_string(), RANK, H)];
    for i in 0..nl {
        let p = |s: &str| format!("layers.{i}.{s}");
        v.extend([
            (p("self_attn.q_proj.weight"), NQ * HD, H),
            (p("self_attn.k_proj.weight"), NKV * HD, H),
            (p("self_attn.v_proj.weight"), NKV * HD, H),
            (p("self_attn.o_proj.weight"), H, NQ * HD),
            (p("attention_conv.kernel_projection.weight"), CONV, H),
            (p("mlp_conv.kernel_projection.weight"), CONV, H),
            (p("mlp.gate_proj.weight"), INTER, H),
            (p("mlp.up_proj.weight"), INTER, H),
            (p("mlp.down_proj.weight"), H, INTER),
        ]);
    }
    v
}

/// Code width of a drafter linear: FE_DF_BITS (default 4) for all, FE_DF_BITS_SMALL (default 8) for the selector projection.
fn bits_for(name: &str) -> u32 {
    let env = |k: &str, d: u32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    if name.starts_with("candidate_selector") { env("FE_DF_BITS_SMALL", 8) } else if name == "fc.weight" { env("FE_DF_BITS_FC", env("FE_DF_BITS", 4)) } else { env("FE_DF_BITS", 4) }
}

/// The packed linears of the drafter in `dir`: from the cache file (FE_DF_CACHE, default
/// `ferrite-xq<bits>.bin` in the drafter folder) or quantized now and cached.
pub fn packed_linears(dir: &Path, nl: usize) -> Result<Vec<Packed>> {
    let list = linears(nl);
    let tag: String = list.iter().map(|(n, _, _)| bits_for(n)).fold(String::new(), |mut s, b| {
        if !s.ends_with(&b.to_string()) {
            s += &b.to_string();
        }
        s
    });
    let cache = std::env::var("FE_DF_CACHE").map(PathBuf::from).unwrap_or_else(|_| {
        dir.join(format!("ferrite-xq{tag}.bin"))
    });
    if cache.exists() {
        let ts = quant::load(&cache)?;
        let ok = ts.len() == list.len() && ts.iter().zip(&list).all(|(t, (n, r, c))| t.name == *n && t.n == *r && t.k == *c && t.bits == bits_for(n));
        if ok {
            eprintln!("fe-xdf: drafter XQ weights from {cache:?}");
            return Ok(ts);
        }
        eprintln!("fe-xdf: {cache:?} does not match this drafter / bits, quantizing again");
    }
    let st = SafeTensors::open_dir(dir)?;
    let t0 = std::time::Instant::now();
    let mut out = Vec::with_capacity(list.len());
    for (name, n, k) in &list {
        let w = bf16s(&st, name, &[*n, *k])?;
        out.push(quant::pack_tensor(name, &w, *n, *k, bits_for(name))?);
    }
    eprintln!("fe-xdf: quantized {} drafter linears in {:.1} s", out.len(), t0.elapsed().as_secs_f64());
    if let Some(p) = cache.parent() {
        std::fs::create_dir_all(p).ok();
    }
    match quant::save(&cache, &out) {
        Ok(()) => eprintln!("fe-xdf: cached as {cache:?}"),
        Err(e) => eprintln!("fe-xdf: could not cache {cache:?}: {e}"),
    }
    Ok(out)
}

/// The 4-bit RTN XQ copy of the BF16 lm_head's first `span` rows (FE_DF_HEAD_SRC: the Qwen3.8-27B BF16 checkpoint folder),
/// cached in that folder.
pub fn head_q4(span: usize) -> Result<Packed> {
    let src = std::env::var("FE_DF_HEAD_SRC").map_err(|_| anyhow::anyhow!("FE_DF_HEAD_SRC is not set: the Qwen3.8-27B BF16 checkpoint folder (FE_DF_HEAD=q4 reads its lm_head)"))?;
    let cache = Path::new(&src).join(format!("ferrite-lm_head-q4-{span}.bin"));
    if cache.exists() {
        if let Ok(mut v) = quant::load(&cache) {
            if v.len() == 1 && v[0].n == span && v[0].k == H {
                return Ok(v.remove(0));
            }
        }
    }
    let st = SafeTensors::open_dir(Path::new(&src))?;
    let b = st.bytes("lm_head.weight")?;
    ensure!(b.len() >= span * H * 2, "lm_head too small");
    let w: Vec<u16> = b[..span * H * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let t0 = std::time::Instant::now();
    let p = quant::pack_tensor("lm_head", &w, span, H, 4)?;
    eprintln!("fe-xdf: 4-bit draft head ({span} rows) quantized in {:.1} s", t0.elapsed().as_secs_f64());
    if let Some(d) = cache.parent() {
        std::fs::create_dir_all(d).ok();
    }
    if let Err(e) = quant::save(&cache, std::slice::from_ref(&p)) {
        eprintln!("fe-xdf: could not cache {cache:?}: {e}");
    }
    Ok(p)
}

fn ze<T>(r: std::result::Result<T, fe_ze::ZeError>) -> Result<T> {
    r.map_err(|e| anyhow::anyhow!("{e}"))
}

impl DFlash {
    /// Loads the drafter from `dir` (z-lab Qwen3.8-27B-DFlash2 layout) onto `e`'s device, registers df_tap with the engine's
    /// kernel registry. Block: FE_DF_BLOCK (2..=16) or the checkpoint's block_size.
    pub fn load(e: &mut Engine, dir: &Path) -> Result<DFlash> {
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let dc = &cfg["dflash_config"];
        let block = std::env::var("FE_DF_BLOCK").ok().and_then(|v| v.parse().ok()).unwrap_or(dc["block_size"].as_u64().unwrap_or(8) as usize);
        ensure!((2..=MAXR).contains(&block), "block {block} (2..=16)");
        let layer_ids: Vec<usize> = dc["target_layer_ids"].as_array().context("target_layer_ids")?.iter().map(|v| v.as_u64().unwrap() as usize).collect();
        ensure!(layer_ids.len() == 5, "{} target layers (want 5)", layer_ids.len());
        let mask_id = dc["mask_token_id"].as_u64().context("mask_token_id")? as u32;
        let window = cfg["sliding_window"].as_u64().unwrap_or(2048) as u32;
        ensure!(window as usize + MAXR <= RING, "sliding window {window} too large for the ring");
        let theta = cfg["rope_parameters"]["rope_theta"].as_f64().unwrap_or(1e7);
        let eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let nl = cfg["num_hidden_layers"].as_u64().unwrap_or(5) as usize;
        ensure!(cfg["hidden_size"].as_u64() == Some(H as u64) && cfg["head_dim"].as_u64() == Some(HD as u64)
            && dc["conv_kernel_size"].as_u64() == Some(2) && dc["conv_group_size"].as_u64() == Some(16)
            && dc["selector_rank"].as_u64() == Some(RANK as u64) && dc["selector_top_k"].as_u64() == Some(TOPK as u64)
            && cfg["is_causal"].as_bool() == Some(false), "unsupported DFlash2 config");

        let b0 = e.dev.bytes.get();
        let mut packed: HashMap<String, Packed> = packed_linears(dir, nl)?.into_iter().map(|p| (p.name.clone(), p)).collect();
        // the absorb's k | v of every layer as one linear: XQ packs 16-row blocks independently, so stacking tensors along n (same
        // k and bits) is concatenating their packed streams. (Fusing the draft's q | k | v and gate | up was measured slower:
        // draft 4.10 vs 3.83 ms; the absorb gains 0.29 -> 0.22 ms.)
        let cat = |parts: &[&Packed], name: &str| -> Result<Packed> {
            ensure!(parts.iter().all(|p| p.k == parts[0].k && p.bits == parts[0].bits), "fusing {name}: k / bits differ");
            Ok(Packed { name: name.to_string(), bits: parts[0].bits, n: parts.iter().map(|p| p.n).sum(), k: parts[0].k,
                data: parts.iter().flat_map(|p| p.data.iter().copied()).collect() })
        };
        let mut fused = Vec::new();
        let mut kv_parts: Vec<Packed> = Vec::new();
        for i in 0..nl {
            let nm = |s: &str| format!("layers.{i}.{s}");
            let get = |s: &str| packed.get(&nm(s)).with_context(|| format!("no packed {}", nm(s)));
            kv_parts.push(cat(&[get("self_attn.k_proj.weight")?, get("self_attn.v_proj.weight")?], &nm("kv"))?);
        }
        fused.push(cat(&kv_parts.iter().collect::<Vec<_>>(), "kv_all")?);
        for f in fused {
            packed.insert(f.name.clone(), f);
        }
        let st = SafeTensors::open_dir(dir)?;
        let dev = &e.dev;
        let mut lins: HashMap<String, Lin> = HashMap::new();
        for (_, p) in packed {
            let w = dev.from_host(&p.data)?;
            let split = split_for(p.n, p.k, 128, None);
            lins.insert(p.name.clone(), Lin { w, bits: p.bits, n: p.n as u32, k: p.k as u32, split });
        }
        let mut su = HashMap::new();
        for k in [H, NQ * HD, INTER, 5 * H] {
            su.insert(k as u32, dev.from_host(&quant::su_for(k))?);
        }
        let up = |name: &str, shape: &[usize]| -> Result<DeviceBuffer<u16>> { dev.from_host(&bf16s(&st, name, shape)?) };
        let mut take = |name: String| -> Result<Lin> { lins.remove(&name).with_context(|| format!("no packed {name}")) };
        let mut layers = Vec::with_capacity(nl);
        for i in 0..nl {
            let p = |s: &str| format!("layers.{i}.{s}");
            layers.push(Layer {
                in_norm: up(&p("input_layernorm.weight"), &[H])?,
                post_norm: up(&p("post_attention_layernorm.weight"), &[H])?,
                q: take(p("self_attn.q_proj.weight"))?,
                k: take(p("self_attn.k_proj.weight"))?,
                v: take(p("self_attn.v_proj.weight"))?,
                o: take(p("self_attn.o_proj.weight"))?,
                q_norm: up(&p("self_attn.q_norm.weight"), &[HD])?,
                k_norm: up(&p("self_attn.k_norm.weight"), &[HD])?,
                attn_base: up(&p("attention_conv.base_kernel"), &[2, 2, H])?,
                attn_proj: take(p("attention_conv.kernel_projection.weight"))?,
                mlp_base: up(&p("mlp_conv.base_kernel"), &[2, 2, H])?,
                mlp_proj: take(p("mlp_conv.kernel_projection.weight"))?,
                gate: take(p("mlp.gate_proj.weight"))?,
                up: take(p("mlp.up_proj.weight"))?,
                down: take(p("mlp.down_proj.weight"))?,
                kc: dev.zeros(RING * NKV * HD)?,
                vc: dev.zeros(RING * NKV * HD)?,
            });
        }
        let fc = take("fc.weight".into())?;
        let kv_all = take("kv_all".into())?;
        let hproj = take("candidate_selector.hidden_projection.weight".into())?;
        let pred_cb = up("candidate_selector.predecessor_codebook", &[248320, RANK])?;
        let succ_cb = up("candidate_selector.successor_codebook", &[248320, RANK])?;
        let cb_host = if std::env::var("FE_DF_TREE").map_or(true, |v| v != "0") {
            Some((bf16s(&st, "candidate_selector.predecessor_codebook", &[248320, RANK])?,
                bf16s(&st, "candidate_selector.successor_codebook", &[248320, RANK])?))
        } else {
            None
        };
        let span = std::env::var("FE_DF_VOCAB").ok().and_then(|v| v.parse().ok()).unwrap_or(131072usize);
        let span = if span == 0 || span >= e.cfg.vocab || span % (16 * TOPK_PARTS as usize) != 0 { e.cfg.vocab } else { span };
        let head4 = if std::env::var("FE_DF_HEAD").is_ok_and(|v| v == "q4") {
            let p = head_q4(span)?;
            let split = split_for(p.n, p.k, 128, None);
            Some(Lin { w: dev.from_host(&p.data)?, bits: p.bits, n: p.n as u32, k: p.k as u32, split })
        } else {
            None
        };
        let nsplit: u32 = std::env::var("FE_DF_ATTN_SPLIT").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        let tap_cap = window as usize + 2 * MAXR;
        let inv: Vec<f32> = (0..HD / 2).map(|i| 1.0f32 / (theta as f32).powf((2 * i) as f32 / HD as f32)).collect();
        let r = MAXR;
        let d = DFlash {
            kx: Kx::new(&e.dev.ze, Source::Rust([("dflash".to_string(), (KX_DFLASH, KX_FLAGS))].into_iter().collect())),
            su,
            layers,
            fc,
            kv_all,
            hidden_norm: up("hidden_norm.weight", &[H])?,
            norm: up("norm.weight", &[H])?,
            hproj,
            head4,
            pred_cb,
            succ_cb,
            cb_host,
            xt: dev.zeros(r * 5 * H)?,
            xs: dev.zeros(r * 5 * H / 32)?,
            res: dev.zeros(r * H)?,
            hn: dev.zeros(r * H)?,
            hc: dev.zeros(r * H)?,
            coef: dev.zeros(r * CONV)?,
            qb: dev.zeros(r * NQ * HD)?,
            kb: dev.zeros(r * NKV * HD)?,
            vb: dev.zeros(r * NKV * HD)?,
            kva: dev.zeros(r * KVA)?,
            at: dev.zeros(r * NQ * HD)?,
            g: dev.zeros(r * INTER)?,
            u: dev.zeros(r * INTER)?,
            part: dev.zeros(nsplit as usize * r * NQ * 130)?,
            logits: dev.zeros(r * span)?,
            tpv: dev.zeros(r * TOPK_PARTS as usize * TOPK)?,
            tpi: dev.zeros(r * TOPK_PARTS as usize * TOPK)?,
            tv: dev.zeros(r * TOPK)?,
            ti: dev.zeros(r * TOPK)?,
            gate_sel: dev.zeros(r * RANK)?,
            out: dev.zeros(r)?,
            ids: dev.zeros(r)?,
            rope: dev.zeros((tap_cap + r) * HD)?,
            stage: [ze(fe_ze::HostBuffer::alloc(&e.dev.ze, r * (HD + 1)))?, ze(fe_ze::HostBuffer::alloc(&e.dev.ze, tap_cap * HD))?],
            taps_buf: dev.zeros(tap_cap * 5 * H)?,
            tap_cap,
            inv,
            block,
            layer_ids,
            mask_id,
            window,
            eps,
            span,
            nsplit,
            prep: None,
            times: DraftTimes::default(),
            bytes: 0,
        };
        let mut d = d;
        d.bytes = e.dev.bytes.get() - b0;
        e.kx.add_override(KX_DFLASH, "", Some(&["df_tap".to_string()]))?;
        e.dev.sync()?;
        eprintln!("fe-xdf: DFlash2 drafter {dir:?}: block {block}, window {window}, head span {span}, {:.2} GB on the device", d.bytes as f64 / 1e9);
        Ok(d)
    }

    /// The engine taps for a window whose rows row0.. need features (row0 = 0: all).
    pub fn taps(&self, row0: u32) -> Taps {
        Taps { layers: self.layer_ids.clone(), buf: self.taps_buf.ptr(), ld: (5 * H) as u32, row0, cap: self.tap_cap as u32 }
    }

    fn run(&mut self, e: &Engine, name: &str, local: [u32; 3], groups: [u32; 3], args: &[&dyn fe_ze::KernelArg]) -> Result<()> {
        self.prep = None;
        self.kx.run(&e.dev.q, "dflash", name, local, groups, args)
    }

    /// A kx-core / kx-xq launch through the engine's registry (not a prepass-preserving one).
    fn erun(&mut self, e: &Engine, tag: &str, name: &str, local: [u32; 3], groups: [u32; 3], args: &[&dyn fe_ze::KernelArg]) -> Result<()> {
        self.prep = None;
        e.kx.run(&e.dev.q, tag, name, local, groups, args)
    }

    /// y [m][ldy] = x [m][k] (row stride xstride, bf16) . W^T for m <= 16 rows on the XQ decode GEMV (prepass skipped when the
    /// scratch already holds this x / rotation / k / m). `n` <= w's rows (a prefix: the draft head's span).
    #[allow(clippy::too_many_arguments)]
    fn xq_raw(&mut self, e: &Engine, w: DevPtr, bits: u32, k: u32, n: u32, split: u32, su: Option<DevPtr>, x: DevPtr, xstride: u32, m: u32,
        y: DevPtr, ldy: u32, f32out: bool) -> Result<()> {
        let (mw, nr) = VARIANTS[variant_for(m)?];
        let (xt, xs) = (self.xt.ptr(), self.xs.ptr());
        let key = (x.0, su.map_or(0, |p| p.0), k, m);
        if self.prep != Some(key) {
            let rot = u32::from(su.is_some());
            let su_arg = su.unwrap_or(xs);
            e.kx.run(&e.dev.q, "xq4", "xq_prep", [16, 1, 1], [k / 128, mw * nr, 1], &[&x, &1u32, &xstride, &su_arg, &rot, &xt, &xs, &k, &m, &mw])?;
        }
        let red = if split > 1 { (split * mw * nr * 16 * 4) as usize } else { 4 };
        e.kx.run(&e.dev.q, &format!("xq{bits}"), &format!("xq_mv_128_{mw}x{nr}_1"), [split * 16, 1, 1], [n / 16, 1, 1],
            &[&xt, &xs, &w, &y, &k, &n, &m, &0u32, &ldy, &u32::from(f32out), &Local(red)])?;
        let esz = if f32out { 4u64 } else { 2 };
        let (y0, y1) = (y.0, y.0 + ((m as u64 - 1) * ldy as u64 + n as u64) * esz);
        let (x0, x1) = (x.0, x.0 + ((m as u64 - 1) * xstride as u64 + k as u64) * 2);
        self.prep = if y0 < x1 && x0 < y1 { None } else { Some(key) };
        Ok(())
    }

    fn xq(&mut self, e: &Engine, t: &Lin, x: DevPtr, m: u32, y: DevPtr) -> Result<()> {
        let su = self.su[&t.k].ptr();
        self.xq_raw(e, t.w.ptr(), t.bits, t.k, t.n, t.split, Some(su), x, t.k, m, y, t.n, false)
    }

    fn rmsnorm(&mut self, e: &Engine, x: DevPtr, w: DevPtr, y: DevPtr, m: u32) -> Result<()> {
        let eps = self.eps;
        self.erun(e, "rows", "rmsnorm_r", [512, 1, 1], [m, 1, 1], &[&x, &w, &y, &(H as u32), &eps])
    }

    /// cos[64] | sin[64] (f32, rounded to bf16 like the HF table) of positions pos0 .. pos0 + n into tab.
    fn rope_rows(inv: &[f32], pos0: usize, n: usize, tab: &mut [f32]) {
        let r16 = |v: f64| fe_xq::half::bf16_to_f32(fe_xq::half::f32_to_bf16(v as f32));
        for r in 0..n {
            for i in 0..HD / 2 {
                let a = ((pos0 + r) as f32 * inv[i]) as f64;
                tab[r * HD + i] = r16(a.cos());
                tab[r * HD + HD / 2 + i] = r16(a.sin());
            }
        }
    }

    /// The RoPE rows of positions pos0 .. pos0 + n for an absorb, to `rope` rows MAXR.. (queued copy from staging 1).
    fn upload_rope(&mut self, e: &Engine, pos0: usize, n: usize) -> Result<()> {
        ensure!(n <= self.tap_cap, "rope rows {n}");
        let st = &mut self.stage[1];
        Self::rope_rows(&self.inv, pos0, n, &mut st.as_mut_slice()[..n * HD]);
        let dst = self.rope.ptr().add::<f32>(MAXR * HD);
        ze(unsafe { e.dev.q.copy_raw(dst.0 as *mut _, st.as_slice().as_ptr() as *const _, n * HD * 4) })?;
        self.prep = None;
        Ok(())
    }

    /// Adds target positions pos0 .. pos0 + rows to the draft context from their feature taps `feats` ([rows][25600], rows
    /// <= 16). Queued (the rope upload syncs first).
    pub fn absorb(&mut self, e: &Engine, feats: DevPtr, rows: usize, pos0: usize) -> Result<()> {
        ensure!((1..=MAXR).contains(&rows), "absorb of {rows} rows");
        self.upload_rope(e, pos0, rows)?;
        self.absorb_at(e, feats, rows, pos0, 0)
    }

    /// absorb with the RoPE rows already uploaded at row `rr` of the table.
    fn absorb_at(&mut self, e: &Engine, feats: DevPtr, rows: usize, pos0: usize, rr: usize) -> Result<()> {
        let m = rows as u32;
        let (res, hn, kva, rope) = (self.res.ptr(), self.hn.ptr(), self.kva.ptr(), self.rope.ptr().add::<f32>((MAXR + rr) * HD));
        let fc: *const Lin = &self.fc;
        let su = self.su[&(5 * H as u32)].ptr();
        let fc = unsafe { &*fc };
        self.xq_raw(e, fc.w.ptr(), fc.bits, fc.k, fc.n, fc.split, Some(su), feats, (5 * H) as u32, m, res, H as u32, false)?;
        let hnw = self.hidden_norm.ptr();
        self.rmsnorm(e, res, hnw, hn, m)?;
        let (eps, mask, p0) = (self.eps, (RING - 1) as u32, pos0 as u32);
        let kv_all: *const Lin = &self.kv_all;
        self.xq(e, unsafe { &*kv_all }, hn, m, kva)?;
        for li in 0..self.layers.len() {
            let l: *const Layer = &self.layers[li];
            let l = unsafe { &*l };
            let kb = kva.add::<u16>(li * 2 * NKV * HD);
            let vb = kb.add::<u16>(NKV * HD);
            self.run(e, "df_norm_rope", [16, 1, 1], [m, NKV as u32, 1], &[&kb, &(KVA as u32), &l.k_norm.ptr(), &eps, &rope, &p0,
                &l.kc.ptr(), &l.vc.ptr(), &vb, &(KVA as u32), &mask])?;
        }
        self.times.absorbs += 1;
        self.times.absorb_rows += rows as u64;
        Ok(())
    }

    /// `absorb` of any number of rows (16 at a time), features `ld` apart.
    pub fn absorb_rows(&mut self, e: &Engine, feats: DevPtr, rows: usize, pos0: usize) -> Result<()> {
        ensure!(rows <= self.tap_cap, "absorb of {rows} rows (cap {})", self.tap_cap);
        self.upload_rope(e, pos0, rows)?;
        let mut r = 0;
        while r < rows {
            let n = (rows - r).min(MAXR);
            self.absorb_at(e, feats.add::<u16>(r * 5 * H), n, pos0 + r, r)?;
            r += n;
        }
        Ok(())
    }

    /// The draft pass of block [anchor, mask x (b - 1)] at positions pos..pos + b (every position < pos absorbed): leaves rows
    /// 1..b's top-16 ids / values in ti / tv and the selector projections in gate_sel (rows 0..b - 1 there = block rows 1..b).
    fn draft_rows(&mut self, e: &Engine, anchor: u32, pos: usize) -> Result<()> {
        let b = self.block;
        let rows = b as u32;
        {
            let st = self.stage[0].as_mut_slice();
            Self::rope_rows(&self.inv, pos, b, &mut st[..b * HD]);
            let idf = &mut st[MAXR * HD..MAXR * HD + b];
            idf.fill(f32::from_bits(self.mask_id));
            idf[0] = f32::from_bits(anchor);
            let p = self.stage[0].as_slice().as_ptr();
            ze(unsafe { e.dev.q.copy_raw(self.rope.ptr().0 as *mut _, p as *const _, b * HD * 4) })?;
            ze(unsafe { e.dev.q.copy_raw(self.ids.ptr().0 as *mut _, p.add(MAXR * HD) as *const _, b * 4) })?;
            self.prep = None;
        }
        let (res, hn, hc, coef, qb, kb, vb, at, g, u, part, rope) = (self.res.ptr(), self.hn.ptr(), self.hc.ptr(), self.coef.ptr(), self.qb.ptr(),
            self.kb.ptr(), self.vb.ptr(), self.at.ptr(), self.g.ptr(), self.u.ptr(), self.part.ptr(), self.rope.ptr());
        let (emb, idp) = (e.w.emb.ptr(), self.ids.ptr());
        self.erun(e, "basic", "embed_bf16", [64, 1, 1], [rows, 1, 1], &[&emb, &idp, &res, &(H as u32)])?;
        let (eps, mask, p0, win, nsplit) = (self.eps, (RING - 1) as u32, pos as u32, self.window, self.nsplit);
        let scale = 1.0f32 / (HD as f32).sqrt();
        let null = DevPtr::NULL;
        let (ldc, side1) = (CONV as u32, 640u32);
        for li in 0..self.layers.len() {
            let l: *const Layer = &self.layers[li];
            let l = unsafe { &*l };
            // attention sublayer: res += conv1(o(attn(conv0(norm(res)))))
            self.rmsnorm(e, res, l.in_norm.ptr(), hn, rows)?;
            self.xq(e, &l.attn_proj, hn, rows, coef)?;
            let base0 = l.attn_base.ptr();
            let base1 = base0.add::<u16>(2 * H);
            self.run(e, "df_dynconv", [256, 1, 1], [rows, (H / 256) as u32, 1], &[&hc, &hn, &coef, &ldc, &0u32, &base0, &null, &rows])?;
            self.xq(e, &l.q, hc, rows, qb)?;
            self.xq(e, &l.k, hc, rows, kb)?;
            self.xq(e, &l.v, hc, rows, vb)?;
            self.run(e, "df_norm_rope", [16, 1, 1], [rows, NQ as u32, 1], &[&qb, &((NQ * HD) as u32), &l.q_norm.ptr(), &eps, &rope, &p0,
                &null, &null, &null, &0u32, &mask])?;
            self.run(e, "df_norm_rope", [16, 1, 1], [rows, NKV as u32, 1], &[&kb, &((NKV * HD) as u32), &l.k_norm.ptr(), &eps, &rope, &p0,
                &l.kc.ptr(), &l.vc.ptr(), &vb, &((NKV * HD) as u32), &mask])?;
            let sgs = (4 * rows).div_ceil(8);
            self.run(e, "df_attn_dpas", [16 * sgs, 1, 1], [NKV as u32, nsplit, 1], &[&part, &qb, &l.kc.ptr(), &l.vc.ptr(), &rows, &p0,
                &win, &scale, &mask, &nsplit, &((NQ * HD) as u32)])?;
            self.run(e, "df_attn_comb", [16, 1, 1], [NQ as u32, rows, 1], &[&at, &part, &rows, &nsplit])?;
            self.xq(e, &l.o, at, rows, hn)?;
            self.run(e, "df_dynconv", [256, 1, 1], [rows, (H / 256) as u32, 1], &[&null, &hn, &coef, &ldc, &side1, &base1, &res, &rows])?;
            // MLP sublayer: res += conv1(down(swiglu(gate, up)(conv0(norm(res)))))
            self.rmsnorm(e, res, l.post_norm.ptr(), hn, rows)?;
            self.xq(e, &l.mlp_proj, hn, rows, coef)?;
            let mb0 = l.mlp_base.ptr();
            let mb1 = mb0.add::<u16>(2 * H);
            self.run(e, "df_dynconv", [256, 1, 1], [rows, (H / 256) as u32, 1], &[&hc, &hn, &coef, &ldc, &0u32, &mb0, &null, &rows])?;
            self.xq(e, &l.gate, hc, rows, g)?;
            self.xq(e, &l.up, hc, rows, u)?;
            let n = rows * INTER as u32;
            self.erun(e, "basic", "swiglu", [64, 1, 1], [n / 64, 1, 1], &[&g, &u, &g, &n])?;
            self.xq(e, &l.down, g, rows, hn)?;
            self.run(e, "df_dynconv", [256, 1, 1], [rows, (H / 256) as u32, 1], &[&null, &hn, &coef, &ldc, &side1, &mb1, &res, &rows])?;
        }
        // final norm; block rows 1.. through the target head (first `span` ids), the selector projection, top-16
        let nw = self.norm.ptr();
        self.rmsnorm(e, res, nw, hn, rows)?;
        let m = rows - 1;
        let x1 = hn.add::<u16>(H);
        let span = self.span as u32;
        let logits = self.logits.ptr();
        if let Some(h4) = self.head4.as_ref() {
            let (w, bits, k, split) = (h4.w.ptr(), h4.bits, h4.k, h4.split);
            let su = self.su[&k].ptr();
            self.xq_raw(e, w, bits, k, span, split, Some(su), x1, H as u32, m, logits, span, true)?;
        } else {
            let hd = &e.w.head;
            self.xq_raw(e, hd.w.ptr(), hd.bits, hd.k, span, hd.split, hd.su.as_ref().map(|s| s.ptr()), x1, H as u32, m, logits, span, true)?;
        }
        let hproj: *const Lin = &self.hproj;
        let gs = self.gate_sel.ptr();
        self.xq(e, unsafe { &*hproj }, x1, m, gs)?;
        let (tpv, tpi, tv, ti) = (self.tpv.ptr(), self.tpi.ptr(), self.tv.ptr(), self.ti.ptr());
        let per = span / TOPK_PARTS;
        self.run(e, "df_topk_part", [256, 1, 1], [m, TOPK_PARTS, 1], &[&tpv, &tpi, &logits, &span, &span, &per])?;
        self.run(e, "df_topk_merge", [256, 1, 1], [m, 1, 1], &[&tv, &ti, &tpv, &tpi, &TOPK_PARTS])?;
        Ok(())
    }

    /// b - 1 chain proposals after `anchor` at position `pos` (the selector's walk over each row's top-16). Synchronous.
    pub fn draft(&mut self, e: &Engine, anchor: u32, pos: usize) -> Result<Vec<u32>> {
        self.draft_rows(e, anchor, pos)?;
        let m = (self.block - 1) as u32;
        let (out, tv, ti, gs, a_cb, b_cb) = (self.out.ptr(), self.tv.ptr(), self.ti.ptr(), self.gate_sel.ptr(), self.pred_cb.ptr(), self.succ_cb.ptr());
        self.run(e, "df_select", [256, 1, 1], [1, 1, 1], &[&out, &tv, &ti, &gs, &a_cb, &b_cb, &anchor, &m])?;
        let mut o = vec![0u32; m as usize];
        ze(e.dev.q.download(&mut o, out))?;
        self.times.drafts += 1;
        Ok(o)
    }

    /// FE_XE_PROFILE: device time per kx-dflash kernel (the drafter's XQ / kx-core launches are in the engine's report).
    pub fn profile_report(&self) -> Result<Option<String>> {
        self.kx.profile_report()
    }

    pub fn profile_reset(&self) -> Result<()> {
        self.kx.profile_reset()
    }

    /// A best-first verify tree of up to `budget` nodes after `anchor` at `pos` (tree::grow_tree over rows 1..b's top-16 and
    /// the selector's edges): (tokens, parents) with parent -1 = the anchor. Needs the host codebooks (FE_DF_TREE=1). Syncs.
    pub fn draft_tree(&mut self, e: &Engine, anchor: u32, pos: usize, budget: usize) -> Result<(Vec<u32>, Vec<i32>)> {
        self.draft_rows(e, anchor, pos)?;
        let (ti, tv) = self.last_topk(e)?;
        let gate = self.last_gate(e)?;
        self.times.drafts += 1;
        let cb = self.cb_host.as_ref().context("draft_tree without the host codebooks (FE_DF_TREE=1)")?;
        Ok(crate::tree::grow_tree(&tv, &ti, &gate, cb, anchor, self.block - 1, budget))
    }

    /// Moves feature-tap rows rows[i] -> i (rows ascending, rows[i] >= i: safe in order). Queued.
    pub fn gather_taps(&self, e: &Engine, rows: &[usize]) -> Result<()> {
        let w = 5 * H;
        for (i, &r) in rows.iter().enumerate() {
            if r != i {
                ze(e.dev.q.copy_dtod(self.taps_buf.ptr().add::<u16>(i * w), self.taps_buf.ptr().add::<u16>(r * w), w * 2))?;
            }
        }
        Ok(())
    }

    /// The last draft pass's top-16 (ids, values) of block rows 1.. ([b - 1][16] each). Syncs.
    pub fn last_topk(&self, e: &Engine) -> Result<(Vec<u32>, Vec<f32>)> {
        let n = (self.block - 1) * TOPK;
        let mut ti = vec![0u32; n];
        let mut tv = vec![0f32; n];
        ze(e.dev.q.download(&mut ti, self.ti.ptr()))?;
        ze(e.dev.q.download(&mut tv, self.tv.ptr()))?;
        Ok((ti, tv))
    }

    /// Bytes of a draft-context snapshot ([`DFlash::save_window`]): `window` positions x layers x K and V x 1024 bf16 (42 MB).
    pub fn window_bytes(&self) -> usize {
        self.window as usize * self.layers.len() * 2 * NKV * HD * 2
    }

    /// (ring index, snapshot row, positions) of each contiguous run of the positions [pos - w, pos), w = min(window, pos).
    fn window_runs(&self, pos: usize) -> Vec<(usize, usize, usize)> {
        let w = (self.window as usize).min(pos);
        let (mut p, mut out) = (pos - w, Vec::new());
        while p < pos {
            let at = p & (RING - 1);
            let n = (pos - p).min(RING - at);
            out.push((at, p - (pos - w), n));
            p += n;
        }
        out
    }

    /// Saves the draft context a draft at `pos` reads (positions [pos - window, pos)) to `dst` (window_bytes). Queued.
    pub fn save_window(&self, e: &Engine, pos: usize, dst: DevPtr) -> Result<()> {
        self.window_copy(e, pos, dst, true)
    }

    /// Restores a save_window snapshot taken at `pos`. Queued.
    pub fn restore_window(&self, e: &Engine, pos: usize, src: DevPtr) -> Result<()> {
        self.window_copy(e, pos, src, false)
    }

    fn window_copy(&self, e: &Engine, pos: usize, buf: DevPtr, save: bool) -> Result<()> {
        let row = NKV * HD;
        let win = self.window as usize;
        for (li, l) in self.layers.iter().enumerate() {
            for (ci, c) in [&l.kc, &l.vc].into_iter().enumerate() {
                let base = buf.add::<u16>((li * 2 + ci) * win * row);
                for (at, off, n) in self.window_runs(pos) {
                    let (ring, snap) = (c.at(at * row), base.add::<u16>(off * row));
                    let (d, s) = if save { (snap, ring) } else { (ring, snap) };
                    ze(e.dev.q.copy_dtod(d, s, n * row * 2))?;
                }
            }
        }
        Ok(())
    }

    /// The selector projections of block rows 1.. of the last draft pass ([b - 1][256] bf16). Syncs.
    pub fn last_gate(&self, e: &Engine) -> Result<Vec<u16>> {
        let mut g = vec![0u16; (self.block - 1) * RANK];
        ze(e.dev.q.download(&mut g, self.gate_sel.ptr()))?;
        Ok(g)
    }

    /// fp32 draft logits of block row r (1..b) over the head span. Syncs (debugging).
    pub fn row_logits(&self, e: &Engine, r: usize) -> Result<Vec<f32>> {
        let mut v = vec![0f32; self.span];
        ze(e.dev.q.download(&mut v, self.logits.ptr().add::<f32>((r - 1) * self.span)))?;
        Ok(v)
    }
}
