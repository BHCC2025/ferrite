//! DFlash2 drafter (z-lab Qwen3.8-27B-DFlash2) on the ferrite engine.
//!
//! Context: for every token the target has processed, its outputs of layers 5, 19, 33, 47, 61 (Engine feature
//! taps) are concatenated, projected by `fc` (25600 -> 5120), RMS-normed (`hidden_norm`) and turned into each
//! draft layer's K/V (k_proj + k_norm + RoPE, v_proj) at the token's position: `absorb`.
//! Draft: a block [anchor, mask x 7] at positions pos..pos+7 runs the 5 draft layers (Qwen3 GQA 32/8, head 128,
//! sliding window 2048, non-causal inside the block; every attention / MLP sublayer wrapped in a two-tap grouped
//! dynamic conv), the final norm, the target lm_head on rows 1..7, top-16 per row and the candidate selector's
//! greedy walk from the anchor: `draft` returns 7 proposals. Verification and acceptance are the caller's
//! (greedy accept-while-match keeps the output identical to plain decoding).
use anyhow::{ensure, Context as _, Result};
use fe_cuda::{memcpy_dtoh, DevPtr, DeviceBuffer, Function, Module};
use fe_model::model::Engine;
use std::path::Path;

pub static K_DFLASH_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_dflash.cubin"));

const H: usize = 5120;
const NQ: usize = 32;
const NKV: usize = 8;
const HD: usize = 128;
const INTER: usize = 17408;
const CONV: usize = 1280; // kernel_projection out: 2 sides x 2 taps x 320 groups
const RANK: usize = 256;
const TOPK: usize = 16;
/// rows one absorb_many batches (decode-size absorbs of several streams; more go stream by stream)
const ABSORB_MANY: usize = 64;
/// k-core fe_gemm_bf16 dynamic shared memory (3 stages x 2 tiles x 128 x 32 bf16)
const GEMM_SMEM: u32 = 3 * 2 * 128 * 32 * 2;

pub struct Kd {
    pub module: Module,
    dynconv: Function,
    norm_rope: Function,
    attn: Function,
    attn_part: Function,
    attn_comb: Function,
    attn_mma: Function,
    topk: Function,
    topk_pack: Function,
    topk_merge: Function,
    select: Function,
    rmsnorm: Function,
    gemv4: Function,
    gemv4t: Function,
    gemv4t4: Function,
    gemv4t16: Function,
    gemv4t16c: Function,
    /// FE_DF_GEMV_TILES=4: 64 columns a block where n % 64 == 0 (default 2: measured the same)
    tiles4: bool,
}

/// A drafter linear [n][k]: bf16 (prefill-size absorbs, or everything with FE_DF_W4=0) and/or affine 4-bit,
/// group 64 (draft and per-round absorb rows), quantized at load like TensorFold's drafter.
struct Lin {
    n: usize,
    k: usize,
    bf: Option<DeviceBuffer<u16>>,
    q4: Option<(DeviceBuffer<u32>, DeviceBuffer<u16>, DeviceBuffer<u16>)>,
    /// the same values repacked for the tensor-core GEMV (fe_df_gemv4t): weights, (scale, bias) words
    t4: Option<(DeviceBuffer<u32>, DeviceBuffer<u32>)>,
}

/// MLX-style affine 4-bit, group 64: scale bf16((max - min) / 15), bias bf16(min), q = round((w - bias) / scale).
fn quant4(w: &[u16], n: usize, k: usize) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    use fe_fp4::format::{bf16_to_f32, f32_to_bf16};
    let kg = k / 64;
    let mut q = vec![0u32; n * k / 8];
    let mut sc = vec![0u16; n * kg];
    let mut bi = vec![0u16; n * kg];
    let threads = 20;
    let rows_per = n.div_ceil(threads);
    std::thread::scope(|scp| {
        let mut qs = q.chunks_mut(rows_per * k / 8);
        let mut ss = sc.chunks_mut(rows_per * kg);
        let mut bs = bi.chunks_mut(rows_per * kg);
        for t in 0..threads {
            let (Some(qc), Some(sc_), Some(bc)) = (qs.next(), ss.next(), bs.next()) else { break };
            scp.spawn(move || {
                let r0 = t * rows_per;
                for rr in 0..qc.len() / (k / 8) {
                    let row = &w[(r0 + rr) * k..(r0 + rr + 1) * k];
                    for g in 0..kg {
                        let blk = &row[g * 64..(g + 1) * 64];
                        let (mut mn, mut mx) = (f32::INFINITY, -f32::INFINITY);
                        for &h in blk {
                            let v = bf16_to_f32(h);
                            mn = mn.min(v);
                            mx = mx.max(v);
                        }
                        let sb = f32_to_bf16((mx - mn) / 15.0);
                        let bb = f32_to_bf16(mn);
                        let (sv, bv) = (bf16_to_f32(sb), bf16_to_f32(bb));
                        sc_[rr * kg + g] = sb;
                        bc[rr * kg + g] = bb;
                        for (i, &h) in blk.iter().enumerate() {
                            let qv = if sv > 0.0 { ((bf16_to_f32(h) - bv) / sv).round().clamp(0.0, 15.0) as u32 } else { 0 };
                            qc[rr * (k / 8) + g * 8 + i / 8] |= qv << (4 * (i % 8));
                        }
                    }
                }
            });
        }
    });
    (q, sc, bi)
}

/// Repack quant4's output for fe_df_gemv4t (layout in the kernel's doc): wp [n/16][k/128][2][32][4],
/// sp [n/16][k/128][32]. n % 32 == 0, k % 128 == 0.
fn pack4t(q: &[u32], sc: &[u16], bi: &[u16], n: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
    let (nch, kw, kg) = (k / 128, k / 8, k / 64);
    let nib = |c: usize, kk: usize| (q[c * kw + kk / 8] >> (4 * (kk % 8))) & 15;
    let mut wp = vec![0u32; n * k / 8];
    let mut sp = vec![0u32; n / 16 * nch * 32];
    for tile in 0..n / 16 {
        for ch in 0..nch {
            let blk = tile * nch + ch;
            for l in 0..32 {
                let (g, t) = (l / 4, l % 4);
                let (cg, c8) = (tile * 16 + g, tile * 16 + g + 8);
                for st in 0..8 {
                    let k0 = ch * 128 + 32 * t + 4 * st;
                    wp[(blk * 2 + st / 4) * 128 + l * 4 + st % 4] = nib(cg, k0) | nib(cg, k0 + 1) << 4
                        | nib(c8, k0) << 8 | nib(c8, k0 + 1) << 12 | nib(cg, k0 + 2) << 16 | nib(cg, k0 + 3) << 20
                        | nib(c8, k0 + 2) << 24 | nib(c8, k0 + 3) << 28;
                }
                let (col, grp) = (tile * 16 + g + 8 * (t & 1), ch * 2 + t / 2);
                sp[blk * 32 + l] = sc[col * kg + grp] as u32 | (bi[col * kg + grp] as u32) << 16;
            }
        }
    }
    (wp, sp)
}

fn lin(ck: &fe_fp4::Checkpoint, name: &str, n: usize, k: usize, keep_bf16: bool, w4: bool) -> Result<Lin> {
    lin_part(ck, name, n, k, 0..n, 0..k, keep_bf16, w4)
}

/// `lin` over the rows (output features) `rs` and columns (inputs) `cs` of the [n][k] checkpoint tensor: a TP rank's
/// column-parallel (rows) or row-parallel (columns) slice.
#[allow(clippy::too_many_arguments)]
fn lin_part(ck: &fe_fp4::Checkpoint, name: &str, n0: usize, k0: usize, rs: std::ops::Range<usize>,
    cs: std::ops::Range<usize>, keep_bf16: bool, w4: bool) -> Result<Lin> {
    let t = ck.get(name)?;
    ensure!(t.dtype == "BF16" && t.shape == [n0, k0], "{name}: {} {:?}, want bf16 [{n0}, {k0}]", t.dtype, t.shape);
    let b = ck.read(name)?;
    let full: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let (n, k) = (rs.len(), cs.len());
    let v: Vec<u16> = if n == n0 && k == k0 { full } else {
        rs.flat_map(|r| full[r * k0 + cs.start..r * k0 + cs.end].to_vec()).collect()
    };
    // FE_DF_GEMV=simt keeps the warp-per-column layout (fe_df_gemv4) instead of the tensor-core one
    let simt = std::env::var("FE_DF_GEMV").is_ok_and(|v| v == "simt") || n % 32 != 0 || k % 128 != 0;
    let (mut q4, mut t4) = (None, None);
    if w4 {
        let (q, sc, bi) = quant4(&v, n, k);
        if simt {
            q4 = Some((DeviceBuffer::from_slice(&q)?, DeviceBuffer::from_slice(&sc)?, DeviceBuffer::from_slice(&bi)?));
        } else {
            let (wp, sp) = pack4t(&q, &sc, &bi, n, k);
            t4 = Some((DeviceBuffer::from_slice(&wp)?, DeviceBuffer::from_slice(&sp)?));
        }
    }
    let bf = if keep_bf16 || !w4 { Some(DeviceBuffer::from_slice(&v)?) } else { None };
    Ok(Lin { n, k, bf, q4, t4 })
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

/// FE_DF_PROF2=1: CUDA events between the phases of every batched draft (draft_rows) plus the host side of
/// draft_trees / draft_many, summed by stream count and printed every 64 drafts (rank 0; rank 1 records and drops).
struct DraftProf {
    ev: Vec<fe_cuda::Event>,
    lab: Vec<&'static str>,
    /// (streams, phase) -> ms summed over the drafts since the last print
    sum: std::collections::BTreeMap<(usize, &'static str), f64>,
    n: std::collections::BTreeMap<usize, usize>,
    /// the last absorb_many's start / end events and rows (read after the next draft's sync)
    ab: [fe_cuda::Event; 2],
    ab_rows: usize,
}

impl DraftProf {
    fn new() -> Result<DraftProf> {
        Ok(DraftProf { ev: (0..96).map(|_| fe_cuda::Event::new()).collect::<Result<_, _>>()?, lab: Vec::new(),
            sum: Default::default(), n: Default::default(), ab: [fe_cuda::Event::new()?, fe_cuda::Event::new()?], ab_rows: 0 })
    }
}

pub struct DFlash {
    kd: Kd,
    layers: Vec<Layer>,
    fc: Lin,
    hidden_norm: DeviceBuffer<u16>,
    norm: DeviceBuffer<u16>,
    hproj: Lin,
    pred_cb: DeviceBuffer<u16>,
    succ_cb: DeviceBuffer<u16>,
    /// host copies of the selector codebooks for draft_tree's edges (254 MB; loaded with FE_TREE=1)
    cb_host: Option<(Vec<u16>, Vec<u16>)>,
    rope: DeviceBuffer<f32>,
    // scratch: absorb up to max_rows rows, draft `block` rows
    res: DeviceBuffer<u16>,
    hn: DeviceBuffer<u16>,
    hc: DeviceBuffer<u16>,
    coef: DeviceBuffer<u16>,
    qb: DeviceBuffer<u16>,
    kb: DeviceBuffer<u16>,
    vb: DeviceBuffer<u16>,
    at: DeviceBuffer<u16>,
    g: DeviceBuffer<u16>,
    u: DeviceBuffer<u16>,
    act: DeviceBuffer<u16>,
    ids: DeviceBuffer<u32>,
    /// pinned staging for `ids` (uploaded on the engine stream: a pageable cuMemcpyHtoD returns before the DMA
    /// lands and does not order with the non-blocking stream)
    ids_host: fe_cuda::HostBuffer<u32>,
    /// absorb_many: the streams' kept taps rows gathered contiguous, and per-row positions / cache offsets
    tin: DeviceBuffer<u16>,
    ab_meta: DeviceBuffer<u32>,
    ab_host: fe_cuda::HostBuffer<u32>,
    tv: DeviceBuffer<f32>,
    ti: DeviceBuffer<u32>,
    gate_sel: DeviceBuffer<u16>,
    out: DeviceBuffer<u32>,
    /// split-K draft attention (FE_DF_ATTN_SPLIT parts, default 8; 1 = one pass per query row) and its partials
    attn_split: u32,
    /// FE_DF_ATTN_MMA (default 1): draft attention on tensor cores (fe_df_attn_mma, a kv head's 4 q heads share each
    /// K / V tile; splits for about FE_DF_ATTN_BLOCKS blocks (default 32; 192 = the earlier sizing),
    /// FE_DF_ATTN_MMA_SPLIT overrides); 0 = fe_df_attn_part
    attn_mma: bool,
    attn_mma_split: u32,
    attn_blocks: u32,
    /// FE_DF_PART (default bf16): TP draft reductions of bf16 partials (half the bytes over the fabric; out zeroed and
    /// passed as the residual, so RoCE and the NCCL fallback give the same bits); f32 = f32 partials (TP2 a893b17:
    /// bf16 saves ~0.3 / 0.4 ms a draft at 1 / 4 streams, acceptance the same within noise)
    part_bf16: bool,
    attn_part: DeviceBuffer<f32>,
    /// draft logits through the NVFP4 lm_head (fe-fp4; the default when it is loaded, FE_DF_HEAD=exl3 uses
    /// the EXL3 one over the full vocabulary)
    head_ws: Option<fe_fp4::Workspace>,
    /// the NVFP4 draft head scores only the first `span` vocabulary ids (FE_DF_VOCAB, default 131072, 0 = all;
    /// like TensorFold's and core's MTP draft_vocab; 65536 cost acceptance on code: 4.68 vs 5.14 / round);
    /// the target still verifies over the full vocabulary
    span: usize,
    pub block: usize,
    pub layer_ids: Vec<usize>,
    pub mask_id: u32,
    pub window: u32,
    pub max_ctx: usize,
    pub max_rows: usize,
    /// draft K/V cache per slot: a ring of `ring` positions (>= window + block, or all of max_ctx): 335 MB for
    /// 4 slots instead of 10.7 GB at 128K
    ring: usize,
    eps: f32,
    /// TP (rank, world) and this rank's q / kv heads; `part` the f32 partial of a row-parallel linear, `pk` / `pkr`
    /// the packed top-16 lists and their all_gather
    tp: Option<(usize, usize)>,
    /// TP: this rank's quarter of the draft head, NVFP4 lm_head rows [rank q, (rank + 1) q) with q = FE_DF_VOCAB / 2
    /// (the two ranks score the first 128K ids in half the time); None: each rank's EXL3 vocabulary half
    tp_head: Option<(fe_fp4::PromptLinear, usize)>,
    nq: usize,
    nkv: usize,
    part: DeviceBuffer<f32>,
    pk: DeviceBuffer<u32>,
    pkr: DeviceBuffer<u32>,
    prof2: std::cell::RefCell<Option<DraftProf>>,
}

fn read_bf16(ck: &fe_fp4::Checkpoint, name: &str, numel: usize) -> Result<Vec<u16>> {
    let t = ck.get(name)?;
    ensure!(t.dtype == "BF16" && t.shape.iter().product::<usize>() == numel, "{name}: {} {:?}, want {numel} bf16",
        t.dtype, t.shape);
    let b = ck.read(name)?;
    Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

fn up_bf16(ck: &fe_fp4::Checkpoint, name: &str, numel: usize) -> Result<DeviceBuffer<u16>> {
    Ok(DeviceBuffer::from_slice(&read_bf16(ck, name, numel)?)?)
}

/// Best-first verify tree (TensorFold's cuda_draft_tree.zig grow, greedy): each row's top-16 candidates after a
/// predecessor scored as log-softmax((logit + 0.6 edge) / 1.5), edge = pred_cb[pred] . gate[row] . succ_cb[cand];
/// the anchor and every taken node open their 4 best children on the next row; the open node of least path cost
/// (- sum of log-chances; ties by parent, token, row) is taken next, up to `most` nodes. Returns (tokens, parents),
/// parent -1 = the anchor.
fn grow_tree(tv: &[f32], ti: &[u32], gate: &[u16], cb: &(Vec<u16>, Vec<u16>), anchor: u32, rows: usize, most: usize)
    -> (Vec<u32>, Vec<i32>) {
    const BRANCH: usize = 4;
    const EDGE: f64 = 0.6;
    const SCALE: f64 = 1.5;
    let bf = |h: u16| f32::from_bits((h as u32) << 16) as f64;
    let chances = |r: usize, pred: u32| -> [f64; TOPK] {
        let g = &gate[r * RANK..(r + 1) * RANK];
        let a = &cb.0[pred as usize * RANK..(pred as usize + 1) * RANK];
        let mut out = [0f64; TOPK];
        let mut most = f64::NEG_INFINITY;
        for (j, o) in out.iter_mut().enumerate() {
            let c = ti[r * TOPK + j] as usize;
            let b = &cb.1[c * RANK..(c + 1) * RANK];
            let edge: f64 = (0..RANK).map(|k| bf(a[k]) * bf(g[k]) * bf(b[k])).sum();
            *o = (tv[r * TOPK + j] as f64 + EDGE * edge) / SCALE;
            most = most.max(*o);
        }
        let lt = most + out.iter().map(|v| (v - most).exp()).sum::<f64>().ln();
        for o in out.iter_mut() {
            *o -= lt;
        }
        out
    };
    #[derive(Clone, Copy)]
    struct Open {
        cost: f64,
        parent: i32,
        token: u32,
        row: usize,
    }
    let (mut tokens, mut parents) = (Vec::new(), Vec::new());
    let mut open: Vec<Open> = Vec::new();
    let (mut pred, mut parent, mut row, mut path) = (anchor, -1i32, 0usize, 0f64);
    loop {
        if row < rows {
            let lc = chances(row, pred);
            let mut order: Vec<usize> = (0..TOPK).collect();
            order.sort_by(|&x, &y| lc[y].partial_cmp(&lc[x]).unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(&y)));
            for &j in order.iter().take(BRANCH) {
                if lc[j].is_finite() {
                    open.push(Open { cost: path - lc[j], parent, token: ti[row * TOPK + j], row });
                }
            }
        }
        if tokens.len() == most || open.is_empty() {
            break;
        }
        let mut best = 0;
        for (i, o) in open.iter().enumerate().skip(1) {
            let b = open[best];
            if o.cost < b.cost || (o.cost == b.cost && (o.parent, o.token, o.row) < (b.parent, b.token, b.row)) {
                best = i;
            }
        }
        let node = open.swap_remove(best);
        tokens.push(node.token);
        parents.push(node.parent);
        pred = node.token;
        parent = (tokens.len() - 1) as i32;
        row = node.row + 1;
        path = node.cost;
    }
    (tokens, parents)
}

/// The drafted block: FE_DF_BLOCK (2..=16; TensorFold and exllamav3 run 16-row blocks past the 8 the checkpoint
/// was trained with), else the checkpoint's dflash_config.block_size.
pub fn block_size(cfg: &serde_json::Value) -> usize {
    std::env::var("FE_DF_BLOCK").ok().and_then(|v| v.parse().ok())
        .unwrap_or_else(|| cfg["dflash_config"]["block_size"].as_u64().unwrap_or(8) as usize)
}

impl DFlash {
    /// Load the drafter from `dir` with a draft cache of `max_ctx` positions; `max_rows` = largest `absorb`.
    pub fn load(dir: &Path, max_ctx: usize, max_rows: usize) -> Result<DFlash> {
        Self::load_slots(dir, max_ctx, max_rows, 1)
    }

    /// As `load`, with one draft K/V cache per engine slot (the scheduler's streams).
    pub fn load_slots(dir: &Path, max_ctx: usize, max_rows: usize, slots: usize) -> Result<DFlash> {
        Self::load_tp(dir, max_ctx, max_rows, slots, None)
    }

    /// `load_slots` for one rank of a TP pair (`tp` = (rank, world), world 2): the rank's half of the heads and MLP
    /// features of every layer and of the draft K/V cache; fc, the convolutions' projections, norms and the
    /// selector stay whole. Both ranks run every draft (rank 1 through Drafter::follow_draft).
    pub fn load_tp(dir: &Path, max_ctx: usize, max_rows: usize, slots: usize, tp: Option<(usize, usize)>)
        -> Result<DFlash> {
        let (rank, world) = tp.unwrap_or((0, 1));
        ensure!(world == 1 || world == 2, "DFlash2 TP: world {world}");
        let (nq, nkv, ni) = (NQ / world, NKV / world, INTER / world);
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let dc = &cfg["dflash_config"];
        let block = block_size(&cfg);
        let layer_ids: Vec<usize> = dc["target_layer_ids"].as_array().context("target_layer_ids")?.iter()
            .map(|v| v.as_u64().unwrap() as usize).collect();
        let mask_id = dc["mask_token_id"].as_u64().context("mask_token_id")? as u32;
        let win = cfg["sliding_window"].as_u64().unwrap_or(2048) as usize;
        // FE_DF_RING=0: one linear cache of max_ctx positions per slot (testing)
        let ring = if max_ctx <= 4096 || std::env::var("FE_DF_RING").is_ok_and(|v| v == "0") {
            max_ctx.next_power_of_two()
        } else {
            4096
        };
        // one absorb call writes its rows into one slot's ring, so a call never takes more rows than the ring holds
        // (longer absorbs are split, absorb_rows); max_rows may be the engine's (a forward of several prompts' chunks)
        let max_rows = if max_ctx <= ring { max_rows } else { max_rows.min(ring) };
        ensure!(max_ctx <= ring || ring >= win + block, "draft ring {ring} too small");
        let window = cfg["sliding_window"].as_u64().unwrap_or(2048) as u32;
        let theta = cfg["rope_parameters"]["rope_theta"].as_f64().unwrap_or(1e7);
        let eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let nl = cfg["num_hidden_layers"].as_u64().unwrap_or(5) as usize;
        ensure!(cfg["hidden_size"].as_u64() == Some(H as u64) && cfg["head_dim"].as_u64() == Some(HD as u64)
            && dc["conv_kernel_size"].as_u64() == Some(2) && dc["conv_group_size"].as_u64() == Some(16)
            && dc["selector_rank"].as_u64() == Some(RANK as u64) && dc["selector_top_k"].as_u64() == Some(TOPK as u64)
            && cfg["is_causal"].as_bool() == Some(false), "unsupported DFlash2 config");
        ensure!((2..=16).contains(&block), "block {block} (2..=16)");

        let module = Module::load(K_DFLASH_CUBIN)?;
        let kd = Kd {
            dynconv: module.function("fe_df_dynconv")?,
            norm_rope: module.function("fe_df_norm_rope")?,
            attn: module.function("fe_df_attn")?,
            attn_part: module.function("fe_df_attn_part")?,
            attn_comb: module.function("fe_df_attn_comb")?,
            attn_mma: module.function("fe_df_attn_mma")?,
            topk: module.function("fe_df_topk16")?,
            topk_pack: module.function("fe_df_topk_pack")?,
            topk_merge: module.function("fe_df_topk_merge")?,
            select: module.function("fe_df_select")?,
            rmsnorm: module.function("fe_df_rmsnorm")?,
            gemv4: module.function("fe_df_gemv4")?,
            gemv4t: module.function("fe_df_gemv4t")?,
            gemv4t4: module.function("fe_df_gemv4t4")?,
            gemv4t16: module.function("fe_df_gemv4t16")?,
            gemv4t16c: module.function("fe_df_gemv4t16c")?,
            tiles4: std::env::var("FE_DF_GEMV_TILES").is_ok_and(|v| v == "4"),
            module,
        };
        let ck = fe_fp4::Checkpoint::open(dir)?;
        let w4 = std::env::var("FE_DF_W4").map(|v| v != "0").unwrap_or(true) || world > 1;
        let mut layers = Vec::new();
        for i in 0..nl {
            let p = |n: &str| format!("layers.{i}.{n}");
            layers.push(Layer {
                in_norm: up_bf16(&ck, &p("input_layernorm.weight"), H)?,
                post_norm: up_bf16(&ck, &p("post_attention_layernorm.weight"), H)?,
                // TP: this rank's q / k / v heads and gate / up features (column-parallel), o / down inputs
                // (row-parallel: f32 partials summed by the fabric)
                q: lin_part(&ck, &p("self_attn.q_proj.weight"), NQ * HD, H, rank * nq * HD..(rank + 1) * nq * HD, 0..H,
                    false, w4)?,
                k: lin_part(&ck, &p("self_attn.k_proj.weight"), NKV * HD, H, rank * nkv * HD..(rank + 1) * nkv * HD, 0..H,
                    true, w4)?,
                v: lin_part(&ck, &p("self_attn.v_proj.weight"), NKV * HD, H, rank * nkv * HD..(rank + 1) * nkv * HD, 0..H,
                    true, w4)?,
                o: lin_part(&ck, &p("self_attn.o_proj.weight"), H, NQ * HD, 0..H, rank * nq * HD..(rank + 1) * nq * HD,
                    false, w4)?,
                q_norm: up_bf16(&ck, &p("self_attn.q_norm.weight"), HD)?,
                k_norm: up_bf16(&ck, &p("self_attn.k_norm.weight"), HD)?,
                attn_base: up_bf16(&ck, &p("attention_conv.base_kernel"), 4 * H)?,
                attn_proj: lin(&ck, &p("attention_conv.kernel_projection.weight"), CONV, H, false, w4)?,
                mlp_base: up_bf16(&ck, &p("mlp_conv.base_kernel"), 4 * H)?,
                mlp_proj: lin(&ck, &p("mlp_conv.kernel_projection.weight"), CONV, H, false, w4)?,
                gate: lin_part(&ck, &p("mlp.gate_proj.weight"), INTER, H, rank * ni..(rank + 1) * ni, 0..H, false, w4)?,
                up: lin_part(&ck, &p("mlp.up_proj.weight"), INTER, H, rank * ni..(rank + 1) * ni, 0..H, false, w4)?,
                down: lin_part(&ck, &p("mlp.down_proj.weight"), H, INTER, 0..H, rank * ni..(rank + 1) * ni, false, w4)?,
                kc: DeviceBuffer::zeroed(slots * ring * nkv * HD)?,
                vc: DeviceBuffer::zeroed(slots * ring * nkv * HD)?,
            });
        }
        let nt = layer_ids.len();
        // RoPE table like HF: f32 inv_freq, f32 angle, cos / sin rounded to bf16
        let half = HD / 2;
        let inv: Vec<f32> = (0..half).map(|i| 1.0f32 / (theta as f32).powf((2 * i) as f32 / HD as f32)).collect();
        let mut tab = vec![0.0f32; max_ctx * HD];
        for p in 0..max_ctx {
            for i in 0..half {
                let a = (p as f32 * inv[i]) as f64;
                tab[p * HD + i] = fe_fp4::format::bf16_to_f32(fe_fp4::format::f32_to_bf16(a.cos() as f32));
                tab[p * HD + half + i] = fe_fp4::format::bf16_to_f32(fe_fp4::format::f32_to_bf16(a.sin() as f32));
            }
        }
        // draft scratch for one block per slot (draft_many batches the decoding streams)
        let bs = slots.max(1) * block;
        let r = max_rows.max(bs);
        Ok(DFlash {
            kd,
            layers,
            fc: lin(&ck, "fc.weight", H, nt * H, true, w4)?,
            hidden_norm: up_bf16(&ck, "hidden_norm.weight", H)?,
            norm: up_bf16(&ck, "norm.weight", H)?,
            hproj: lin(&ck, "candidate_selector.hidden_projection.weight", RANK, H, false, w4)?,
            pred_cb: up_bf16(&ck, "candidate_selector.predecessor_codebook", 248320 * RANK)?,
            succ_cb: up_bf16(&ck, "candidate_selector.successor_codebook", 248320 * RANK)?,
            // only when the scheduler verifies trees (FE_TREE=1, or FE_DF_TREE=1); FE_DF_TREE=0 forces chains
            cb_host: if std::env::var("FE_DF_TREE").map(|v| v == "0")
                .unwrap_or_else(|_| !std::env::var("FE_TREE").is_ok_and(|v| v == "1")) { None } else {
                Some((read_bf16(&ck, "candidate_selector.predecessor_codebook", 248320 * RANK)?,
                    read_bf16(&ck, "candidate_selector.successor_codebook", 248320 * RANK)?))
            },
            rope: DeviceBuffer::from_slice(&tab)?,
            res: DeviceBuffer::alloc(r * H)?,
            hn: DeviceBuffer::alloc(r * H)?,
            hc: DeviceBuffer::alloc(bs * H)?,
            coef: DeviceBuffer::alloc(bs * CONV)?,
            qb: DeviceBuffer::alloc(bs * NQ * HD)?,
            kb: DeviceBuffer::alloc(r * NKV * HD)?,
            vb: DeviceBuffer::alloc(r * NKV * HD)?,
            at: DeviceBuffer::alloc(bs * NQ * HD)?,
            g: DeviceBuffer::alloc(bs * INTER)?,
            u: DeviceBuffer::alloc(bs * INTER)?,
            act: DeviceBuffer::alloc(bs * INTER)?,
            ids: DeviceBuffer::alloc(bs + 2 * slots.max(1))?,
            ids_host: fe_cuda::HostBuffer::alloc(bs + 2 * slots.max(1))?,
            tin: DeviceBuffer::alloc(ABSORB_MANY * nt * H)?,
            ab_meta: DeviceBuffer::alloc(2 * ABSORB_MANY)?,
            ab_host: fe_cuda::HostBuffer::alloc(2 * ABSORB_MANY)?,
            tv: DeviceBuffer::alloc(bs * TOPK)?,
            ti: DeviceBuffer::alloc(bs * TOPK)?,
            gate_sel: DeviceBuffer::alloc(bs * RANK)?,
            out: DeviceBuffer::alloc(bs)?,
            attn_split: std::env::var("FE_DF_ATTN_SPLIT").ok().and_then(|v| v.parse().ok()).unwrap_or(8u32).max(1),
            attn_mma: std::env::var("FE_DF_ATTN_MMA").map_or(true, |v| v != "0"),
            part_bf16: std::env::var("FE_DF_PART").map_or(true, |v| v != "f32"),
            attn_mma_split: std::env::var("FE_DF_ATTN_MMA_SPLIT").ok().and_then(|v| v.parse().ok()).unwrap_or(0u32),
            attn_blocks: std::env::var("FE_DF_ATTN_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(32u32).max(1),
            attn_part: DeviceBuffer::alloc((64 * block).max(16 * bs) * NQ * 130)?,
            head_ws: if std::env::var("FE_DF_HEAD").is_ok_and(|v| v == "exl3") { None } else { Some(fe_fp4::Workspace::new(bs, H)?) },
            span: std::env::var("FE_DF_VOCAB").ok().and_then(|v| v.parse().ok()).unwrap_or(131072),
            block,
            layer_ids,
            mask_id,
            window,
            max_ctx,
            max_rows: r,
            ring,
            eps,
            tp: tp.filter(|t| t.1 > 1),
            tp_head: match (world > 1, fe_model::model::LoadOpts::from_env().prompt_fp4) {
                (true, Some(dir)) => {
                    let span = std::env::var("FE_DF_VOCAB").ok().and_then(|v| v.parse().ok()).unwrap_or(131072usize);
                    let q = span / world;
                    ensure!(q % 256 == 0, "TP draft head: {q} rows a rank");
                    let ck4 = fe_fp4::Checkpoint::open(&dir)?;
                    let sl = fe_fp4::Slice { rows: vec![(rank * q, (rank + 1) * q)], k: (0, H) };
                    Some((fe_fp4::PromptLinear::load_sliced(&ck4, &["lm_head"], false, Some(&[sl]))?, q))
                }
                _ => None,
            },
            nq,
            nkv,
            part: DeviceBuffer::alloc(if world > 1 { bs * H } else { 1 })?,
            pk: DeviceBuffer::alloc(bs * 32)?,
            pkr: DeviceBuffer::alloc(2 * bs * 32)?,
            prof2: std::cell::RefCell::new(if std::env::var("FE_DF_PROF2").is_ok_and(|v| v == "1") {
                Some(DraftProf::new()?)
            } else {
                None
            }),
        })
    }

    /// FE_DF_PROF2: an event on the engine stream closing phase `lab` (the first mark of a draft opens it).
    fn pmark(&self, e: &Engine, lab: &'static str) {
        if let Some(p) = self.prof2.borrow_mut().as_mut() {
            if lab == "start" {
                p.lab.clear();
            }
            // the first reduction of a draft also waits for the other rank to arrive
            let lab = if lab == "reduce" && !p.lab.iter().any(|l| l.starts_with("reduce")) { "reduce1st" } else { lab };
            let i = p.lab.len();
            if i < p.ev.len() && e.stream.record(&p.ev[i]).is_ok() {
                p.lab.push(lab);
            }
        }
    }

    /// FE_DF_PROF2: after the draft's sync, add its phases (GPU, between consecutive marks) and the host times.
    fn pcollect(&self, ns: usize, host: &[(&'static str, f64)]) {
        let mut g = self.prof2.borrow_mut();
        let Some(p) = g.as_mut() else { return };
        for i in 1..p.lab.len() {
            let ms = p.ev[i].elapsed_since(&p.ev[i - 1]).unwrap_or(0.0) as f64;
            *p.sum.entry((ns, p.lab[i])).or_default() += ms;
        }
        if p.lab.len() > 1 {
            let ms = p.ev[p.lab.len() - 1].elapsed_since(&p.ev[0]).unwrap_or(0.0) as f64;
            *p.sum.entry((ns, "gpu")).or_default() += ms;
        }
        for &(l, ms) in host {
            *p.sum.entry((ns, l)).or_default() += ms;
        }
        if p.ab_rows > 0 && p.lab.len() > 1 {
            *p.sum.entry((ns, "absorb")).or_default() += p.ab[1].elapsed_since(&p.ab[0]).unwrap_or(0.0) as f64;
            *p.sum.entry((ns, "gap_ab_draft")).or_default() += p.ev[0].elapsed_since(&p.ab[1]).unwrap_or(0.0) as f64;
            *p.sum.entry((ns, "absorb_rows")).or_default() += p.ab_rows as f64;
            p.ab_rows = 0;
        }
        let n = p.n.entry(ns).or_default();
        *n += 1;
        if *n == 64 {
            let mut line = format!("dflash prof2: {ns} streams, ms a draft over {n} drafts:");
            let keys: Vec<_> = p.sum.keys().filter(|k| k.0 == ns).copied().collect();
            for k in keys {
                line += &format!(" {} {:.3}", k.1, p.sum.remove(&k).unwrap() / *n as f64);
            }
            eprintln!("{line}");
            *n = 0;
        }
    }

    /// y [m][n] (stride ldy) = x [m][k] (stride lda) . W[n][k]^T (+ res): the W4 GEMV for m <= 16 when the linear
    /// is quantized, else k-core's bf16 GEMV (m <= 8) / GEMM.
    fn linear(&self, e: &Engine, y: DevPtr, ldy: usize, x: DevPtr, lda: usize, w: &Lin, m: usize, res: Option<DevPtr>)
        -> Result<()> {
        self.linear_mode(e, y, ldy, x, lda, w, m, res, false)
    }

    /// out (bf16 [m][5120]) = x . W^T for a row-parallel linear: whole, or under TP this rank's f32 partial summed
    /// with the other rank's through the fabric (RoCE up to its slot, else NCCL; both ranks get the same bits).
    fn linear_reduce(&self, e: &Engine, out: DevPtr, x: DevPtr, lda: usize, w: &Lin, m: usize, lab: &'static str)
        -> Result<()> {
        match (&self.tp, e.tp.as_ref()) {
            (Some(_), Some(tp)) => {
                ensure!(w.t4.is_some(), "TP drafter: row-parallel linear without the tensor-core W4 form");
                if self.part_bf16 {
                    self.linear_mode(e, self.part.ptr(), H, x, lda, w, m, None, false)?;
                    e.stream.memset_d32(out, 0, m * H / 2)?;
                    self.pmark(e, lab);
                    tp.fab.reduce_bf16_to_bf16(&e.stream, self.part.ptr(), out, Some(out), m * H)?;
                } else {
                    self.linear_mode(e, self.part.ptr(), H, x, lda, w, m, None, true)?;
                    self.pmark(e, lab);
                    tp.fab.reduce_to_bf16(&e.stream, self.part.ptr(), out, None, m * H)?;
                }
                self.pmark(e, "reduce");
                Ok(())
            }
            _ => {
                self.linear(e, out, H, x, lda, w, m, None)?;
                self.pmark(e, lab);
                Ok(())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_mode(&self, e: &Engine, y: DevPtr, ldy: usize, x: DevPtr, lda: usize, w: &Lin, m: usize, res: Option<DevPtr>,
        f32_out: bool) -> Result<()> {
        let s = &e.stream;
        let r = res.unwrap_or(DevPtr::NULL);
        let (n, k) = (w.n, w.k);
        let yf = f32_out as u32;
        ensure!(!f32_out || w.t4.is_some() && m <= 64, "f32 linear output needs the tensor-core W4 form");
        if let (Some((wp, sp)), true) = (&w.t4, m > 16 && m <= 64) {
            // batched streams: 16-row chunks of a column block side by side (its weights shared through L2)
            let chunks = m.div_ceil(16) as u32;
            unsafe {
                s.launch(&self.kd.gemv4t16c, (n / 32) as u32 * chunks, 256, 0, (y, x, wp.ptr(), sp.ptr(), r, m as u32,
                    n as u32, k as u32, lda as u32, ldy as u32, chunks, yf))?;
            }
            return Ok(());
        }
        if let (Some((wp, sp)), true) = (&w.t4, m <= 16) {
            unsafe {
                let (f, cols) = if m > 8 {
                    (&self.kd.gemv4t16, 32)
                } else if self.kd.tiles4 && n % 64 == 0 {
                    (&self.kd.gemv4t4, 64)
                } else {
                    (&self.kd.gemv4t, 32)
                };
                s.launch(f, (n / cols) as u32, 256, 0, (y, x, wp.ptr(), sp.ptr(), r, m as u32, n as u32,
                    k as u32, lda as u32, ldy as u32, yf))?;
            }
            return Ok(());
        }
        if let (Some((q, sc, bi)), true) = (&w.q4, m <= 8) {
            unsafe {
                s.launch(&self.kd.gemv4, (n as u32).div_ceil(8), 256, 0, (y, x, q.ptr(), sc.ptr(), bi.ptr(), r, m as u32,
                    n as u32, k as u32, lda as u32, ldy as u32))?;
            }
            return Ok(());
        }
        let w = w.bf.as_ref().context("drafter linear without bf16 weights for a large row count")?;
        unsafe {
            if m <= 8 && k % 256 == 0 {
                s.launch(&e.k.gemv, (n as u32).div_ceil(8), 256, 0, (y, x, w.ptr(), r, m as u32, n as u32, k as u32,
                    lda as u32, ldy as u32))?;
            } else {
                s.launch(&e.k.gemm, ((m as u32).div_ceil(128), (n as u32).div_ceil(128)), 256, GEMM_SMEM,
                    (y, x, w.ptr(), r, m as u32, n as u32, k as u32, lda as u32, ldy as u32))?;
            }
        }
        Ok(())
    }

    /// Qwen3 RMSNorm (plain weight; k-core's fe_rmsnorm is Qwen3.5's 1 + w).
    fn rmsnorm(&self, e: &Engine, y: DevPtr, x: DevPtr, w: &DeviceBuffer<u16>, rows: usize) -> Result<()> {
        unsafe { e.stream.launch(&self.kd.rmsnorm, rows as u32, 256, 0, (y, x, w.ptr(), self.eps))? };
        Ok(())
    }

    /// Add `rows` target positions pos0.. to the draft context: `feats` = the Engine's feature taps of those rows
    /// ([rows][5 * 5120] bf16, contiguous).
    pub fn absorb(&mut self, e: &Engine, feats: DevPtr, rows: usize, pos0: usize) -> Result<()> {
        self.absorb_slot(e, 0, feats, rows, pos0)
    }

    /// `absorb_slot` in pieces of at most max_rows rows (prompt chunks can be longer than one absorb; the rows are
    /// independent, so the pieces fill the same cache as one call would).
    pub fn absorb_rows(&mut self, e: &Engine, slot: usize, feats: DevPtr, rows: usize, pos0: usize) -> Result<()> {
        let mut r = 0;
        while r < rows {
            let n = (rows - r).min(self.max_rows);
            self.absorb_slot(e, slot, feats.add::<u16>(r * self.layer_ids.len() * H), n, pos0 + r)?;
            r += n;
        }
        Ok(())
    }

    /// `absorb` into slot `slot`'s draft context.
    pub fn absorb_slot(&mut self, e: &Engine, slot: usize, feats: DevPtr, rows: usize, pos0: usize) -> Result<()> {
        ensure!(rows <= self.max_rows && pos0 + rows <= self.max_ctx, "absorb: {rows} rows at {pos0}");
        let (so, mask) = (slot * self.ring * self.nkv * HD, (self.ring - 1) as u32);
        let s = &e.stream;
        let nt = self.layer_ids.len();
        self.linear(e, self.res.ptr(), H, feats, nt * H, &self.fc, rows, None)?;
        self.rmsnorm(e, self.hn.ptr(), self.res.ptr(), &self.hidden_norm, rows)?;
        for l in &self.layers {
            self.linear(e, self.kb.ptr(), self.nkv * HD, self.hn.ptr(), H, &l.k, rows, None)?;
            self.linear(e, self.vb.ptr(), self.nkv * HD, self.hn.ptr(), H, &l.v, rows, None)?;
            unsafe {
                s.launch(&self.kd.norm_rope, (rows as u32, self.nkv as u32), 32, 0, (self.kb.ptr(), (self.nkv * HD) as u32,
                    l.k_norm.ptr(), self.eps, self.rope.ptr(), pos0 as u32, l.kc.at(so), l.vc.at(so), (self.nkv * HD) as u32,
                    self.vb.ptr(), (self.nkv * HD) as u32, mask, 1u32, DevPtr::NULL, DevPtr::NULL))?;
            }
        }
        Ok(())
    }

    /// Several streams' kept rows (slot, their feature taps, rows, first position) in one pass: the taps gathered
    /// into one block of rows, one fc / k / v GEMV each, per-row positions and draft caches in norm_rope.
    pub fn absorb_many(&mut self, e: &Engine, items: &[(usize, DevPtr, usize, usize)]) -> Result<()> {
        let total: usize = items.iter().map(|it| it.2).sum();
        if let Some(p) = self.prof2.borrow_mut().as_mut() {
            let _ = e.stream.record(&p.ab[0]);
            p.ab_rows = total;
        }
        let r = self.absorb_many_inner(e, items, total);
        if let Some(p) = self.prof2.borrow_mut().as_mut() {
            let _ = e.stream.record(&p.ab[1]);
        }
        r
    }

    fn absorb_many_inner(&mut self, e: &Engine, items: &[(usize, DevPtr, usize, usize)], total: usize) -> Result<()> {
        if items.len() < 2 || total > ABSORB_MANY {
            for &(slot, feats, rows, pos0) in items {
                self.absorb_rows(e, slot, feats, rows, pos0)?;
            }
            return Ok(());
        }
        let s = &e.stream;
        let nt = self.layer_ids.len();
        let mask = (self.ring - 1) as u32;
        let (ring, max_ctx, nkv) = (self.ring, self.max_ctx, self.nkv);
        let meta = self.ab_host.as_mut_slice();
        let mut r = 0;
        for &(slot, feats, rows, pos0) in items {
            ensure!(pos0 + rows <= max_ctx, "absorb: {rows} rows at {pos0}");
            s.memcpy_dtod(self.tin.at(r * nt * H), feats, rows * nt * H * 2)?;
            for i in 0..rows {
                meta[r + i] = (pos0 + i) as u32;
                meta[ABSORB_MANY + r + i] = (slot * ring * nkv * HD) as u32;
            }
            r += rows;
        }
        // the previous absorb_many's upload finished before this step's syncs
        unsafe { s.memcpy_htod(self.ab_meta.ptr(), self.ab_host.as_slice())? };
        let (posv, cso) = (self.ab_meta.ptr(), self.ab_meta.at(ABSORB_MANY));
        self.linear(e, self.res.ptr(), H, self.tin.ptr(), nt * H, &self.fc, total, None)?;
        self.rmsnorm(e, self.hn.ptr(), self.res.ptr(), &self.hidden_norm, total)?;
        for l in &self.layers {
            self.linear(e, self.kb.ptr(), self.nkv * HD, self.hn.ptr(), H, &l.k, total, None)?;
            self.linear(e, self.vb.ptr(), self.nkv * HD, self.hn.ptr(), H, &l.v, total, None)?;
            unsafe {
                s.launch(&self.kd.norm_rope, (total as u32, self.nkv as u32), 32, 0, (self.kb.ptr(), (self.nkv * HD) as u32,
                    l.k_norm.ptr(), self.eps, self.rope.ptr(), 0u32, l.kc.ptr(), l.vc.ptr(), (self.nkv * HD) as u32,
                    self.vb.ptr(), (self.nkv * HD) as u32, mask, 1u32, posv, cso))?;
            }
        }
        Ok(())
    }

    /// Draft `block - 1` tokens after `anchor` (at position `pos` = context length). Uses (and overwrites)
    /// e.logits. Synchronous.
    pub fn draft(&mut self, e: &Engine, anchor: u32, pos: usize) -> Result<Vec<u32>> {
        self.draft_slot(e, 0, anchor, pos)
    }

    /// `draft` from slot `slot`'s draft context.
    pub fn draft_slot(&mut self, e: &Engine, slot: usize, anchor: u32, pos: usize) -> Result<Vec<u32>> {
        Ok(self.draft_many(e, &[(slot, anchor, pos)])?.remove(0))
    }

    /// Chain drafts for several streams (slot, anchor, pos) from one batched pass of the draft model (one block
    /// per stream; the weights are read once for all of them). Synchronous.
    pub fn draft_many(&mut self, e: &Engine, reqs: &[(usize, u32, usize)]) -> Result<Vec<Vec<u32>>> {
        let b = self.block;
        self.draft_rows(e, reqs)?;
        let s = &e.stream;
        for (st, &(_, anchor, _)) in reqs.iter().enumerate() {
            let r1 = st * b + 1; // the stream's candidate rows: its block's rows 1..
            unsafe {
                s.launch(&self.kd.select, 1u32, 512, 0, (self.out.at(st * b), self.tv.at(r1 * TOPK), self.ti.at(r1 * TOPK),
                    self.gate_sel.at(r1 * RANK), self.pred_cb.ptr(), self.succ_cb.ptr(), anchor, (b - 1) as u32))?;
            }
        }
        s.sync()?;
        let mut out = vec![0u32; reqs.len() * b];
        memcpy_dtoh(&mut out, self.out.ptr())?;
        Ok(out.chunks(b).map(|c| c[..b - 1].to_vec()).collect())
    }

    /// A best-first verify tree of up to `budget` nodes from the block's candidates (`grow_tree`); None (chains)
    /// unless the codebooks were loaded (FE_TREE=1 / FE_DF_TREE=1). Synchronous.
    pub fn draft_tree_slot(&mut self, e: &Engine, slot: usize, anchor: u32, pos: usize, budget: usize)
        -> Result<Option<(Vec<u32>, Vec<i32>)>> {
        Ok(self.draft_trees(e, &[(slot, anchor, pos, budget)])?.map(|mut v| v.remove(0)))
    }

    /// Trees for several streams (slot, anchor, pos, budget) from one batched pass of the draft model.
    pub fn draft_trees(&mut self, e: &Engine, reqs: &[(usize, u32, usize, usize)])
        -> Result<Option<Vec<(Vec<u32>, Vec<i32>)>>> {
        if self.cb_host.is_none() {
            return Ok(None);
        }
        let b = self.block;
        let rq: Vec<(usize, u32, usize)> = reqs.iter().map(|&(sl, a, p, _)| (sl, a, p)).collect();
        let t0 = std::time::Instant::now();
        self.draft_rows(e, &rq)?;
        let t1 = std::time::Instant::now();
        e.stream.sync()?;
        let t2 = std::time::Instant::now();
        let rows = reqs.len() * b;
        let mut ti = vec![0u32; rows * TOPK];
        let mut tv = vec![0f32; rows * TOPK];
        let mut gs = vec![0u16; rows * RANK];
        memcpy_dtoh(&mut ti, self.ti.ptr())?;
        memcpy_dtoh(&mut tv, self.tv.ptr())?;
        memcpy_dtoh(&mut gs, self.gate_sel.ptr())?;
        let t3 = std::time::Instant::now();
        // FE_DF_DUMP=<file>: append each tree pass's top-16 ids and values (rank 0) to compare drafts across builds
        if let Ok(path) = std::env::var("FE_DF_DUMP") {
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = f.write_all(&ti.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
                let _ = f.write_all(&tv.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect::<Vec<u8>>());
            }
        }
        let cb = self.cb_host.as_ref().unwrap();
        let trees = reqs.iter().enumerate().map(|(st, &(_, anchor, _, budget))| {
            let r1 = st * b + 1;
            grow_tree(&tv[r1 * TOPK..(st + 1) * b * TOPK], &ti[r1 * TOPK..(st + 1) * b * TOPK],
                &gs[r1 * RANK..(st + 1) * b * RANK], cb, anchor, b - 1, budget)
        }).collect();
        let t4 = std::time::Instant::now();
        let ms = |a: std::time::Instant, b: std::time::Instant| 1e3 * (b - a).as_secs_f64();
        self.pcollect(reqs.len(), &[("h_launch", ms(t0, t1)), ("h_wait", ms(t1, t2)), ("h_dtoh", ms(t2, t3)),
            ("h_grow", ms(t3, t4)), ("h_total", ms(t0, t4))]);
        Ok(Some(trees))
    }

    fn draft_rows(&mut self, e: &Engine, reqs: &[(usize, u32, usize)]) -> Result<()> {
        let mask = (self.ring - 1) as u32;
        let b = self.block;
        let ns = reqs.len();
        let rows = ns * b;
        ensure!(ns >= 1 && rows <= self.out.len(), "draft: {ns} streams (scratch for {})", self.out.len() / b);
        let s = &e.stream;
        // staging: block ids [ns][b] | positions [ns] | draft-cache offsets [ns]
        let (n_slots, max_ctx, ring, nkv) = (self.n_slots(), self.max_ctx, self.ring, self.nkv);
        let ids = self.ids_host.as_mut_slice();
        ids[..rows].fill(self.mask_id);
        for (st, &(slot, anchor, pos)) in reqs.iter().enumerate() {
            ensure!(slot < n_slots && pos + b <= max_ctx, "draft: position {pos} of slot {slot}");
            ids[st * b] = anchor;
            ids[rows + st] = pos as u32;
            ids[rows + ns + st] = (slot * ring * nkv * HD) as u32;
        }
        let (posv, cso) = (self.ids.at(rows), self.ids.at(rows + ns));
        // the previous draft ended with a stream sync, so the staging buffer is free
        self.pmark(e, "start");
        unsafe { s.memcpy_htod(self.ids.ptr(), &self.ids_host.as_slice()[..rows + 2 * ns])? };
        let (res, hn, hc, coef) = (self.res.ptr(), self.hn.ptr(), self.hc.ptr(), self.coef.ptr());
        unsafe {
            s.launch(&e.k.embed, rows as u32, 256, 0, (res, e.w.embed.ptr(), self.ids.ptr(), H as u32))?;
        }
        self.pmark(e, "embed");
        let conv = |x: DevPtr, out: DevPtr, base: DevPtr, side: usize, r: DevPtr| -> Result<()> {
            unsafe {
                s.launch(&self.kd.dynconv, (rows as u32, (H / 256) as u32), 256, 0, (out, x, coef, CONV as u32,
                    (side * 640) as u32, base.add::<u16>(side * 2 * H), r, b as u32))?;
            }
            Ok(())
        };
        for l in &self.layers {
            // attention sublayer
            self.rmsnorm(e, hn, res, &l.in_norm, rows)?;
            self.linear(e, coef, CONV, hn, H, &l.attn_proj, rows, None)?;
            conv(hn, hc, l.attn_base.ptr(), 0, DevPtr::NULL)?;
            self.linear(e, self.qb.ptr(), self.nq * HD, hc, H, &l.q, rows, None)?;
            self.linear(e, self.kb.ptr(), self.nkv * HD, hc, H, &l.k, rows, None)?;
            self.linear(e, self.vb.ptr(), self.nkv * HD, hc, H, &l.v, rows, None)?;
            self.pmark(e, "a_lin");
            unsafe {
                s.launch(&self.kd.norm_rope, (rows as u32, self.nq as u32), 32, 0, (self.qb.ptr(), (self.nq * HD) as u32,
                    l.q_norm.ptr(), self.eps, self.rope.ptr(), 0u32, DevPtr::NULL, DevPtr::NULL, 0u32,
                    DevPtr::NULL, 0u32, 0u32, b as u32, posv, cso))?;
                s.launch(&self.kd.norm_rope, (rows as u32, self.nkv as u32), 32, 0, (self.kb.ptr(), (self.nkv * HD) as u32,
                    l.k_norm.ptr(), self.eps, self.rope.ptr(), 0u32, l.kc.ptr(), l.vc.ptr(), (self.nkv * HD) as u32,
                    self.vb.ptr(), (self.nkv * HD) as u32, mask, b as u32, posv, cso))?;
                let scale = 1.0f32 / (HD as f32).sqrt();
                // split-K over each stream's window (grid z = stream); the partials fit 16 splits of every block
                let sp = if self.attn_mma {
                    // ~32 blocks: fewer splits cost less in partials and the merge (TP1 1 stream 4 splits 0.16 ms vs 24
                    // splits 0.35 a draft; 4 streams 2 vs 6: 0.36 vs 0.48)
                    let want = if self.attn_mma_split > 0 { self.attn_mma_split } else { (self.attn_blocks / (self.nkv * ns) as u32).max(2) };
                    want.clamp(1, if ns > 1 { 16 } else { 64 })
                } else {
                    self.attn_split.clamp(1, if ns > 1 { 16 } else { 64 })
                };
                if self.attn_mma {
                    s.launch(&self.kd.attn_mma, ((self.nq / 4) as u32, sp, ns as u32), 128u32, 0, (self.attn_part.ptr(),
                        self.qb.ptr(), l.kc.ptr(), l.vc.ptr(), b as u32, 0u32, self.window, scale * std::f32::consts::LOG2_E,
                        mask, sp, posv, cso, self.nq as u32, (self.nkv * HD) as u32))?;
                } else {
                    s.launch(&self.kd.attn_part, (self.nq as u32, sp, ns as u32), (32 * b) as u32, 0, (self.attn_part.ptr(),
                        self.qb.ptr(), l.kc.ptr(), l.vc.ptr(), b as u32, 0u32, self.window, scale, mask, sp, posv, cso,
                        self.nq as u32, (self.nkv * HD) as u32))?;
                }
                s.launch(&self.kd.attn_comb, (self.nq as u32, b as u32, ns as u32), 32, 0, (self.at.ptr(), self.attn_part.ptr(),
                    b as u32, sp, self.nq as u32))?;
            }
            self.pmark(e, "attn");
            self.linear_reduce(e, hn, self.at.ptr(), self.nq * HD, &l.o, rows, "o_lin")?;
            conv(hn, DevPtr::NULL, l.attn_base.ptr(), 1, res)?;
            self.pmark(e, "conv");
            // MLP sublayer
            self.rmsnorm(e, hn, res, &l.post_norm, rows)?;
            self.linear(e, coef, CONV, hn, H, &l.mlp_proj, rows, None)?;
            conv(hn, hc, l.mlp_base.ptr(), 0, DevPtr::NULL)?;
            self.pmark(e, "m_lin");
            let ni = l.gate.n; // this rank's MLP features (all of them without TP)
            self.linear(e, self.g.ptr(), ni, hc, H, &l.gate, rows, None)?;
            self.linear(e, self.u.ptr(), ni, hc, H, &l.up, rows, None)?;
            unsafe {
                s.launch(&e.k.silu_mul, ((ni as u32 / 2).div_ceil(256), rows as u32), 256, 0, (self.act.ptr(),
                    self.g.ptr(), self.u.ptr(), ni as u32, ni as u32))?;
            }
            self.pmark(e, "gate_up");
            self.linear_reduce(e, hn, self.act.ptr(), ni, &l.down, rows, "down")?;
            conv(hn, DevPtr::NULL, l.mlp_base.ptr(), 1, res)?;
            self.pmark(e, "conv");
        }
        // final norm; every row through the target head (each stream's rows 1.. are its candidates), top-16
        self.rmsnorm(e, hn, res, &self.norm, rows)?;
        self.pmark(e, "fnorm");
        let nd = rows;
        let fin = hn;
        let vocab = e.cfg.vocab;
        let (ld, span) = match (&self.tp_head, &self.head_ws, e.w.fp4.as_ref()) {
            (Some((h, q)), Some(ws), Some(fk)) => {
                h.quantize(fk, s, ws, fin, H, nd)?;
                self.pmark(e, "hquant");
                fk.gemm_fp4(s, e.logits.ptr(), *q, ws.codes.ptr(), ws.sf.ptr(), h.w.ptr(),
                    h.sw.as_ref().context("lm_head scales")?.ptr(), h.alpha.ptr(), None, nd, *q, H)?;
                (*q, *q)
            }
            _ => match (&self.head_ws, e.w.lm_head.p.as_ref(), e.w.fp4.as_ref()) {
            (Some(ws), Some(p), Some(fk)) => {
                // a prefix of the rows is a prefix of the codes, the tiled scales and alpha
                let span = if self.span == 0 || self.span >= p.n || self.span % 256 != 0 { p.n } else { self.span };
                p.quantize(fk, s, ws, fin, H, nd)?;
                match p.kind {
                    fe_fp4::Kind::Nvfp4 => fk.gemm_fp4(s, e.logits.ptr(), vocab, ws.codes.ptr(), ws.sf.ptr(), p.w.ptr(),
                        p.sw.as_ref().context("lm_head scales")?.ptr(), p.alpha.ptr(), None, nd, span, H)?,
                    fe_fp4::Kind::Fp8 => fk.gemm_fp8(s, e.logits.ptr(), vocab, ws.codes.ptr(), p.w.ptr(), p.alpha.ptr(),
                        None, nd, span, H)?,
                }
                (vocab, span)
            }
            _ => {
                // the EXL3 head, rows packed by its width; under TP each rank scores its vocabulary half
                let n = e.lm_head_rows(fin, nd)?;
                (n, n)
            }
            },
        };
        self.pmark(e, "head");
        let rank = self.tp.map_or(0, |t| t.0);
        if rank == 0 {
            // the selector projection (rank 0 walks the candidates)
            self.linear(e, self.gate_sel.ptr(), RANK, fin, H, &self.hproj, nd, None)?;
            self.pmark(e, "hproj");
        }
        unsafe {
            s.launch(&self.kd.topk, nd as u32, 256, 0, (self.tv.ptr(), self.ti.ptr(), e.logits.ptr(), ld as u32,
                span as u32))?;
        }
        self.pmark(e, "topk");
        if let (Some(_), Some(tp)) = (self.tp, e.tp.as_ref()) {
            // TP: each rank scored its vocabulary half; the two top-16 lists (ids made global) meet on both ranks
            // and merge into rank 0's tv / ti
            unsafe {
                // ids made global: the rank's first id (its head quarter or its vocabulary half)
                let first = self.tp_head.as_ref().map_or(e.w.lm_head.n, |h| h.1) * rank;
                s.launch(&self.kd.topk_pack, nd as u32, 32, 0, (self.pk.ptr(), self.tv.ptr(), self.ti.ptr(), first as u32))?;
            }
            tp.fab.all_gather(s, self.pk.ptr(), self.pkr.ptr(), nd * 32, fe_fabric::DataType::F32)?;
            if rank == 0 {
                unsafe {
                    s.launch(&self.kd.topk_merge, nd as u32, 32, 0, (self.tv.ptr(), self.ti.ptr(), self.pkr.ptr(),
                        nd as u32))?;
                }
            }
            self.pmark(e, "gather");
        }
        Ok(())
    }

    /// TP rank 1's side of a draft (Drafter::follow_draft): the same pass as rank 0's, meeting it in every
    /// reduction and the top-16 exchange; the drafts themselves come out on rank 0.
    pub fn follow_rows(&mut self, e: &Engine, reqs: &[(usize, u32, usize)]) -> Result<()> {
        self.draft_rows(e, reqs)?;
        e.stream.sync()?;
        Ok(())
    }

    fn n_slots(&self) -> usize {
        self.layers[0].kc.len() / (self.ring * self.nkv * HD)
    }

    /// Bytes of one slot's drafting state at a snapshot (`save_window`): the sliding window of draft K / V a draft
    /// reads, `window` positions x layers x K and V x this rank's kv heads (TP2: 21 MB a rank, TP1 42 MB).
    pub fn window_bytes(&self) -> usize {
        self.window as usize * self.layers.len() * 2 * self.nkv * HD * 2
    }

    /// The draft-cache copies of a snapshot at `pos`: (position's index in the slot's ring, row in the snapshot,
    /// positions) for each contiguous run of the positions [pos - w, pos) a draft at `pos` reads (w = min(window,
    /// pos)); a run splits where the ring wraps.
    fn window_runs(&self, pos: usize) -> Vec<(usize, usize, usize)> {
        let w = (self.window as usize).min(pos);
        let (mut p, mut out) = (pos - w, Vec::new());
        while p < pos {
            let at = p & (self.ring - 1);
            let n = (pos - p).min(self.ring - at);
            out.push((at, p - (pos - w), n));
            p += n;
        }
        out
    }

    /// Saves slot `slot`'s drafting state for a resume at `pos` (every position < pos absorbed) to `dst`
    /// (window_bytes): per layer K then V, `window` rows each, in position order. Asynchronous on e.stream. Only the
    /// draft context is saved: copy rows rebuild their index from each request's tokens, so they need nothing.
    pub fn save_window(&self, e: &Engine, slot: usize, pos: usize, dst: DevPtr) -> Result<()> {
        self.window_copy(e, slot, pos, dst, true)
    }

    /// Restores a `save_window` snapshot taken at `pos` into slot `slot` (any slot): the draft context a draft at pos
    /// reads is then bitwise the saved one; the ring's other positions are written by absorb / draft before any read.
    pub fn restore_window(&self, e: &Engine, slot: usize, pos: usize, src: DevPtr) -> Result<()> {
        self.window_copy(e, slot, pos, src, false)
    }

    fn window_copy(&self, e: &Engine, slot: usize, pos: usize, buf: DevPtr, save: bool) -> Result<()> {
        ensure!(slot < self.n_slots() && pos <= self.max_ctx, "draft snapshot: position {pos} of slot {slot}");
        let row = self.nkv * HD;
        let (win, so) = (self.window as usize, slot * self.ring * row);
        for (li, l) in self.layers.iter().enumerate() {
            for (ci, c) in [&l.kc, &l.vc].into_iter().enumerate() {
                let base = buf.add::<u16>((li * 2 + ci) * win * row);
                for &(at, off, n) in &self.window_runs(pos) {
                    let (ring, snap) = (c.at(so + at * row), base.add::<u16>(off * row));
                    let (d, s) = if save { (snap, ring) } else { (ring, snap) };
                    e.stream.memcpy_dtod(d, s, n * row * 2)?;
                }
            }
        }
        Ok(())
    }

    /// The last draft pass's top-16 ids and values of every block row (snapshot tests compare them bitwise).
    pub fn last_topk(&self, streams: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        let rows = streams * self.block;
        let mut ti = vec![0u32; rows * TOPK];
        let mut tv = vec![0f32; rows * TOPK];
        memcpy_dtoh(&mut ti, self.ti.ptr())?;
        memcpy_dtoh(&mut tv, self.tv.ptr())?;
        Ok((ti, tv))
    }
}

/// Greedy generation of `max_new` tokens after the raw text `prompt` (single stream), drafting with DFlash2
/// unless `plain`. Prints the text, then timing / acceptance and the token ids to stderr. Env: FE_DRAFT_DIR,
/// FE_PREFILL_CHUNK, FE_DF_TRACE, FE_DF_DEBUG, FE_DF_W4, FE_DF_GEMV, FE_DF_HEAD, FE_DF_VOCAB, FE_DF_PROF (see README).
/// Drafter snapshot test (TP1, `ferrite <model> df-snaptest [n]`): slot 1 first holds another prompt's draft
/// context; slot 0 prefills n tokens of a code prompt; slot 0's draft window at n is saved and restored into slot 1,
/// and each slot then drafts alone at n. The top-16 ids and values of every block row must be bitwise equal (and
/// differ from slot 1's draft before the restore). n > 4096 + 2048 (default 6000) makes the window wrap the ring.
pub fn snaptest(model: &Path, n: usize) -> Result<()> {
    use fe_cuda::Context;
    use fe_model::config::Config;
    use fe_model::model::{Kernels, LoadOpts, Weights};
    let draft_dir = fe_site::draft_dir();
    let chunk: usize = std::env::var("FE_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let tok_path = {
        let p = model.join("tokenizer.json");
        if p.exists() { p } else { Config::locate(model).with_file_name("tokenizer.json") }
    };
    let tok = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids_of = |unit: &str| -> Result<Vec<u32>> {
        let text = unit.repeat(n / 8 + 1);
        let ids = tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
        ensure!(ids.len() >= n, "test text too short");
        Ok(ids[..n].to_vec())
    };
    let a = ids_of("def parse(line):\n    key, _, value = line.partition('=')\n    return key.strip(), value.strip()\n")?;
    let b = ids_of("The harbor town woke slowly in winter, and the baker opened his shutters to the frost. ")?;
    let max_ctx = n.max(6144) + 64;
    let ctx = Context::new(0)?;
    let cfg = Config::load(&Config::locate(model))?;
    let k = Kernels::load(&Module::load(fe_model::K_CORE_CUBIN)?)?;
    let w = Weights::load(&cfg, model, &LoadOpts::from_env(), ctx.sm_count()? as u32)?;
    let dcfg: serde_json::Value = serde_json::from_slice(&std::fs::read(Path::new(&draft_dir).join("config.json"))?)?;
    let block = block_size(&dcfg);
    let mut e = Engine::with_spec(cfg, k, w, 2, max_ctx, chunk.max(16), block)?;
    let mut d = DFlash::load_slots(Path::new(&draft_dir), max_ctx, chunk, 2)?;
    e.set_feature_taps(&d.layer_ids)?;
    let prefill = |e: &mut Engine, d: &mut DFlash, slot: usize, ids: &[u32]| -> Result<u32> {
        let (mut done, mut first) = (0, 0);
        while done < ids.len() {
            let end = (done + chunk).min(ids.len());
            let lr: Vec<usize> = if end == ids.len() { vec![end - done - 1] } else { vec![] };
            e.forward(&[(slot, &ids[done..end])], &lr)?;
            d.absorb_rows(e, slot, e.feature_taps().unwrap(), end - done, done)?;
            if end == ids.len() {
                first = e.argmax(1)?[0];
            }
            done = end;
        }
        Ok(first)
    };
    prefill(&mut e, &mut d, 1, &b)?;
    let first = prefill(&mut e, &mut d, 0, &a)?;
    let draft = |e: &Engine, d: &mut DFlash, slot: usize| -> Result<(Vec<u32>, Vec<f32>)> {
        d.draft_rows(e, &[(slot, first, n)])?;
        e.stream.sync()?;
        d.last_topk(1)
    };
    let before = draft(&e, &mut d, 1)?;
    let snap = DeviceBuffer::<u16>::zeroed(d.window_bytes() / 2)?;
    d.save_window(&e, 0, n, snap.ptr())?;
    d.restore_window(&e, 1, n, snap.ptr())?;
    let x = draft(&e, &mut d, 0)?;
    let y = draft(&e, &mut d, 1)?;
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
    let same = x.0 == y.0 && bits(&x.1) == bits(&y.1);
    let control = x.0 != before.0 || bits(&x.1) != bits(&before.1);
    let runs = d.window_runs(n).len();
    eprintln!("df-snaptest: n {n}, window {} positions in {runs} ring run(s), snapshot {:.1} MB; restored slot == source: {}; \
        slot before the restore differed: {}", (d.window as usize).min(n), d.window_bytes() as f64 / 1e6,
        if same { "BITWISE EQUAL" } else { "DIFFER" }, control);
    ensure!(same && control, "df-snaptest FAILED");
    println!("df-snaptest PASS");
    Ok(())
}

pub fn generate(model: &Path, prompt: &str, max_new: usize, plain: bool) -> Result<()> {
    use fe_cuda::Context;
    use fe_model::config::Config;
    use fe_model::model::{Kernels, LoadOpts, Weights};
    use std::time::Instant;
    let model = model.to_path_buf();
    let draft_dir = fe_site::draft_dir();
    let chunk: usize = std::env::var("FE_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);

    let tok_path = {
        let p = model.join("tokenizer.json");
        if p.exists() { p } else { Config::locate(&model).with_file_name("tokenizer.json") }
    };
    let tok = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = tok.encode(prompt, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let n = ids.len();
    let max_ctx = n + max_new + 32;

    let ctx = Context::new(0)?;
    let cfg = Config::load(&Config::locate(&model))?;
    let k = Kernels::load(&Module::load(fe_model::K_CORE_CUBIN)?)?;
    let t0 = Instant::now();
    let w = Weights::load(&cfg, &model, &LoadOpts::from_env(), ctx.sm_count()? as u32)?;
    // GDN history / replay rows = the verify rows (the drafted block)
    let dcfg: serde_json::Value = serde_json::from_slice(&std::fs::read(Path::new(&draft_dir).join("config.json"))?)?;
    let block = block_size(&dcfg);
    let mut e = Engine::with_spec(cfg, k, w, 1, max_ctx, chunk.max(16), block)?;
    let mut df = if plain { None } else {
        let d = DFlash::load(Path::new(&draft_dir), max_ctx, chunk)?;
        e.set_feature_taps(&d.layer_ids)?;
        Some(d)
    };
    let (free, _) = ctx.mem_info()?;
    eprintln!("loaded in {:.1}s, free {:.1} GB; prompt {n} tokens", t0.elapsed().as_secs_f32(), free as f64 / 1e9);

    // prefill (absorbing every chunk's taps into the draft context)
    let t = Instant::now();
    let mut done = 0;
    let mut first = 0;
    while done < n {
        let end = (done + chunk).min(n);
        let lr: Vec<usize> = if end == n { vec![end - done - 1] } else { vec![] };
        e.forward(&[(0, &ids[done..end])], &lr)?;
        if let Some(d) = df.as_mut() {
            d.absorb(&e, e.feature_taps().unwrap(), end - done, done)?;
        }
        if end == n {
            first = e.argmax(1)?[0];
        }
        done = end;
    }
    e.stream.sync()?;
    let pf = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let eos = e.cfg.eos;
    let mut out = vec![first];
    let (mut rounds, mut accepted) = (0usize, 0usize);
    let mut pending = first;
    // FE_DF_PROF=1: sync around each phase and print the per-round split (draft / verify / absorb)
    let prof = std::env::var("FE_DF_PROF").is_ok();
    let mut tp = [0f64; 3];
    // FE_DF_VERIFY=8 with FE_DF_BLOCK=16: verify only the first 8 rows while the accepted-drafts average is below
    // FE_DF_VERIFY_AT (default 3; prose accepts ~1.6 and a 16-row verify costs ~6 ms more), the whole block above
    // (code: 5.2 -> 7.3 accepted). 0 = always the whole block.
    let short: usize = std::env::var("FE_DF_VERIFY").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let at: f32 = std::env::var("FE_DF_VERIFY_AT").ok().and_then(|v| v.parse().ok()).unwrap_or(3.0);
    let mut ema = 7.0f32;
    while out.len() < max_new && pending != eos {
        let pos = e.slot_len[0];
        match df.as_mut() {
            None => {
                e.forward(&[(0, &[pending])], &[0])?;
                pending = e.argmax(1)?[0];
                out.push(pending);
            }
            Some(d) => {
                let t1 = Instant::now();
                let mut drafts = d.draft(&e, pending, pos)?;
                if short > 0 && ema < at {
                    drafts.truncate(short - 1);
                }
                let t2 = Instant::now();
                let mut rows = vec![pending];
                rows.extend_from_slice(&drafts);
                let lr: Vec<usize> = (0..rows.len()).collect();
                e.forward_ex(&[(0, &rows)], &lr, true, rows.len())?;
                let y = e.argmax(rows.len())?;
                let t3 = Instant::now();
                let mut a = 0;
                while a < drafts.len() && drafts[a] == y[a] {
                    a += 1;
                }
                e.commit(0, pos, rows.len(), a + 1)?;
                d.absorb(&e, e.feature_taps().unwrap(), a + 1, pos)?;
                ema = 0.7 * ema + 0.3 * a as f32;
                if prof {
                    e.stream.sync()?;
                    tp[0] += (t2 - t1).as_secs_f64();
                    tp[1] += (t3 - t2).as_secs_f64();
                    tp[2] += t3.elapsed().as_secs_f64();
                }
                if rounds < 4 && std::env::var("FE_DF_DEBUG").is_ok() {
                    let dec = |v: &[u32]| tok.decode(v, false).unwrap_or_default();
                    eprintln!("round {rounds} pos {pos} anchor {:?}\n  drafts {:?} {:?}\n  target {:?} {:?}", dec(&[pending]),
                        drafts, dec(&drafts), &y, dec(&y));
                    if let Some(d) = df.as_ref() {
                        let _ = d;
                    }
                }
                if std::env::var("FE_DF_TRACE").is_ok() {
                    eprintln!("trace round {rounds} pos {pos} keep {}", a + 1);
                }
                rounds += 1;
                accepted += a;
                for &tk in &drafts[..a] {
                    out.push(tk);
                }
                out.push(y[a]);
                pending = y[a];
            }
        }
    }
    e.stream.sync()?;
    let dt = t.elapsed().as_secs_f64();
    if let Some(p) = out.iter().position(|&t| t == eos) {
        out.truncate(p + 1);
    }
    let text = tok.decode(&out, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{text}");
    eprintln!("prefill {n} tok {pf:.2}s ({:.0} tok/s); decode {} tok in {dt:.2}s ({:.2} tok/s){}", n as f64 / pf,
        out.len(), out.len() as f64 / dt,
        if rounds > 0 { format!(", {rounds} rounds, {:.2} accepted drafts/round", accepted as f64 / rounds as f64) }
        else { String::new() });
    if prof && rounds > 0 {
        let r = rounds as f64 / 1e3;
        eprintln!("per round: draft {:.2} ms, verify+argmax {:.2} ms, commit+absorb {:.2} ms", tp[0] / r, tp[1] / r, tp[2] / r);
    }
    // the token ids, for comparing runs
    eprintln!("ids {:?}", out);
    Ok(())
}

impl fe_model::sched::Drafter for DFlash {
    fn layers(&self) -> Vec<usize> {
        self.layer_ids.clone()
    }
    fn max_drafts(&self) -> usize {
        self.block - 1
    }
    fn absorb(&mut self, e: &Engine, slot: usize, feats: DevPtr, rows: usize, pos0: usize) -> Result<()> {
        self.absorb_rows(e, slot, feats, rows, pos0)
    }
    fn state_bytes(&self) -> usize {
        self.window_bytes()
    }
    fn save_state(&self, e: &Engine, slot: usize, pos: usize, dst: DevPtr) -> Result<()> {
        self.save_window(e, slot, pos, dst)
    }
    fn restore_state(&mut self, e: &Engine, slot: usize, pos: usize, src: DevPtr) -> Result<()> {
        self.restore_window(e, slot, pos, src)
    }
    fn draft(&mut self, e: &Engine, slot: usize, anchor: u32, pos: usize) -> Result<Vec<u32>> {
        self.draft_slot(e, slot, anchor, pos)
    }
    fn draft_tree(&mut self, e: &Engine, slot: usize, anchor: u32, pos: usize, budget: usize)
        -> Result<Option<(Vec<u32>, Vec<i32>)>> {
        self.draft_tree_slot(e, slot, anchor, pos, budget)
    }
    fn draft_many(&mut self, e: &Engine, reqs: &[(usize, u32, usize)]) -> Result<Vec<Vec<u32>>> {
        DFlash::draft_many(self, e, reqs)
    }
    fn absorb_many(&mut self, e: &Engine, items: &[(usize, DevPtr, usize, usize)]) -> Result<()> {
        DFlash::absorb_many(self, e, items)
    }
    fn tp_split(&self) -> bool {
        self.tp.is_some()
    }
    fn follow_draft(&mut self, e: &Engine, reqs: &[(usize, u32, usize, usize)]) -> Result<()> {
        self.follow_rows(e, &reqs.iter().map(|&(sl, a, p, _)| (sl, a, p)).collect::<Vec<_>>())
    }
    fn draft_tree_many(&mut self, e: &Engine, reqs: &[(usize, u32, usize, usize)])
        -> Result<Option<Vec<(Vec<u32>, Vec<i32>)>>> {
        self.draft_trees(e, reqs)
    }
}

/// A scheduler drafter factory for DFlash2 (FE_DRAFT_DIR, default the z-lab checkpoint): one draft cache per slot.
/// Streams one DFlash2 draft pass serves at most (4 x 16-row blocks = the 4-bit drafter linears' 64 rows).
pub const MAX_DRAFT_STREAMS: usize = 4;

pub fn factory() -> fe_model::sched::DrafterFactory {
    std::sync::Arc::new(|e: &Engine, o: &fe_model::sched::SchedOptions| {
        // TP: both ranks hold half of the drafter (FE_DF_TP=0: rank 0 drafts alone with the whole model)
        let tp = e.tp_rank().filter(|_| std::env::var("FE_DF_TP").map(|v| v != "0").unwrap_or(true));
        if let Some((rank, world)) = e.tp_rank() {
            ensure!(tp.is_some() || rank == 0, "DFlash2 without the TP split runs on rank 0 only (rank {rank} of {world})");
            ensure!(e.w.lm_head.n % 256 == 0, "TP draft head: vocabulary slice {} not a multiple of 256", e.w.lm_head.n);
        }
        // one draft pass covers at most MAX_DRAFT_STREAMS streams: the 4-bit drafter linears take 64 rows (4 blocks of
        // 16) and the bf16 copies a larger pass needs are not loaded (TP ranks never, TP1 unless FE_DF_W4=0). With the
        // MTP head too (DRAFT=auto) only FE_DF_STREAMS (2) streams draft with DFlash2 at a time.
        let df_streams = if e.has_mtp() {
            std::env::var("FE_DF_STREAMS").ok().and_then(|v| v.parse().ok()).unwrap_or(2usize).min(o.n_slots)
        } else {
            o.n_slots
        };
        let bf16 = tp.is_none() && std::env::var("FE_DF_W4").is_ok_and(|v| v == "0");
        ensure!(df_streams <= MAX_DRAFT_STREAMS || bf16,
            "--slots {} is not supported with DFlash2 drafting (DRAFT=dflash): one draft pass serves at most {} streams \
            in this build. Use --slots {} (SLOTS={}), or DRAFT=auto / DRAFT=mtp for more slots (MTP drafting above 2 \
            streams; not tested above 4 slots)", o.n_slots, MAX_DRAFT_STREAMS, MAX_DRAFT_STREAMS, MAX_DRAFT_STREAMS);
        let dir = fe_site::draft_dir();
        let d = DFlash::load_tp(&dir, e.max_ctx, o.prefill_chunk.max(64), o.n_slots, tp)?;
        Ok(Box::new(d) as Box<dyn fe_model::sched::Drafter>)
    })
}
