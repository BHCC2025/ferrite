//! Fast decode: always-used weights kept quantized on the card (k-gguf GEMV, q8_1 activations), bf16 weights
//! through the plain bf16 GEMV, routed experts from a static on-card set (k-gguf grouped-expert kernels) with the
//! misses computed on the CPU (fe-ggq CpuMoe) while the card runs the shared expert and the resident experts.
//!
//! One CUDA graph per step (1..=8 rows): the step parameters (position, rope tables, embeddings, PLE rows) are
//! copied from pinned memory at the start of the graph; per MoE layer the graph hands the router's picks and the
//! FFN input to a CPU thread through pinned memory (stream write-value), and waits (stream wait-value) for its rows
//! before combining. No host sync inside a step.
use crate::config::Config;
use crate::engine::{g256, tensor, vecf, Kern, Lin};
use anyhow::{anyhow, bail, Result};
use fe_cuda::{DevPtr, DeviceBuffer, Event, Function, Graph, HostBuffer, Module, Stream, VmBuffer};
use fe_ggq::cpu::{CpuMoe, MoeJob};
use fe_ggq::{GgmlType, Ggq, gpu::Q8Act};
use fe_gguf::Gguf;
use rayon::prelude::*;
use std::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

pub mod hostmem;
mod prefill;
mod lanes;
mod tp;
pub use lanes::{DraftTuner, Parked};

/// A prompt stopped early because its request was cancelled ([`Fast::cancel`]).
#[derive(Debug, Clone, Copy)]
pub struct Cancelled;
impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("prompt cancelled")
    }
}
impl std::error::Error for Cancelled {}

/// Dynamic shared memory of fx_pattn (kernels/k-fnext/src/pattn.rs PATTN_SMEM).
const PATTN_SMEM: u32 = 512 * 16 + 4 * 2 * 512 * 16 + 2 * 4 * 16 * 4;

/// Dynamic shared memory of fx_gdn_chunk (kernels/k-fnext/src/gdnc.rs SMEM_FLOATS * 4).
const GDN_CHUNK_SMEM: u32 = 24868 * 4;
/// Dynamic shared memory of fx_gdn_chunk_tc (kernels/k-fnext/src/gdntc.rs GDN_TC_SMEM).
pub(crate) const GDN_TC_SMEM: u32 = 21508 * 4;
/// Dynamic shared memory of fx_idx_score_tc (kernels/k-fnext/src/idxtc.rs IDX_TC_SMEM).
pub(crate) const IDX_TC_SMEM: u32 = (4 * 16 * 132 + 64 * 132) * 4;

/// Step parameter word offsets (kernels/k-fnext/src/fast.rs).
const P_POS: usize = 0;
const P_ROWS: usize = 1;
const P_NB: usize = 2;
const P_BLO: usize = 3;
/// FNEXT_TP=2 step epoch << 8 (kernels/k-fnext/src/fast.rs P_EPOCH)
const P_EPOCH: usize = 4;
const P_CS: usize = 16;
const P_SN: usize = P_CS + 24 * 32;
const P_PCS: usize = P_SN + 24 * 32;
const P_PSN: usize = P_PCS + 8 * 32;
const P_WORDS: usize = P_PSN + 8 * 32;
pub const MAX_ROWS: usize = 24;
/// Lanes a batched step can hold (one step-parameter block each).
pub const MAX_LANES: usize = 8;
/// Step graphs of one kind set (per lane and device): plain rows 1..=MAX_ROWS, verify rows, commits.
pub(crate) const N_GRAPHS: usize = 3 * MAX_ROWS + 1;
/// Index offset of the step graphs captured WITH the logits host copy. The copy is baked into a graph when it is
/// captured, so `want_logits` is part of every step-graph key: a graph captured for a greedy request must never serve
/// a sampled one (it would read the previous request's logits), nor the other way round (a wasted copy).
pub(crate) const G_LOGITS: usize = N_GRAPHS;

/// A weight kept in its GGUF quant type on the card.
pub struct QW {
    pub buf: DeviceBuffer<u8>,
    pub ty: GgmlType,
    pub k: usize,
    pub n: usize,
}
impl QW {
    fn load(g: &Gguf, name: &str) -> Result<QW> {
        let t = tensor(g, name)?;
        if !fe_ggq::gpu::GEMV_TYPES.contains(&t.ty) {
            bail!("{name}: {} has no GEMV", t.ty.name());
        }
        Ok(QW { buf: DeviceBuffer::from_slice(g.data(t))?, ty: t.ty, k: t.ne0(), n: t.n_rows() })
    }
    fn run(&self, q: &Ggq, s: &Stream, a: &Q8Act, y: DevPtr, add: bool) -> Result<()> {
        q.gemv(s, self.ty, self.buf.ptr(), self.ty.row_bytes(self.k), self.n, a, y, add, None)
    }
    /// Output rows row0 .. row0 + n only (into y[0 .. n]).
    fn run_rows(&self, q: &Ggq, s: &Stream, a: &Q8Act, y: DevPtr, row0: usize, n: usize) -> Result<()> {
        let rb = self.ty.row_bytes(self.k);
        q.gemv(s, self.ty, self.buf.ptr().byte_add(row0 * rb), rb, n, a, y, false, None)
    }
}

struct FKern {
    _m: Module,
    /// this device's k-gguf kernels for hrun / hc_mix (leaked: they live as long as the process)
    gq: &'static Ggq,
    attn_prep: Function,
    attn: Function,
    idx_kstore: Function,
    idx_pool: Function,
    idx_score: Function,
    topk: Function,
    route: Function,
    moe_merge: Function,
    argmax: Function,
    gemv_h: Function,
    route8: Function,
    moe_out: Function,
    silu_ld: Function,
    hc_combine: Function,
    moe_fin: Function,
    scale_ld: Function,
    mtp_cat: Function,
    moe_acc: Function,
    copy2d: Function,
    to_host: Function,
    hit_flag: Function,
    gdn_chunk: Function,
    gdn_tc: Function,
    gdn_tc1: Function,
    idx_tc: Function,
    idx_tc3: Function,
    hc_norm_bf16: Function,
    hc_comb_norm: Function,
    hc_mix_res: Function,
    moe_comb: Function,
    moe_comb_f: Function,
    bcast_add: Function,
    argmax_part: Function,
    argmax_fin: Function,
    conv_p: Function,
    conv_state: Function,
    attn6: Function,
    pattn: Function,
    ll_pack: Function,
    ar_comb: Function,
    moe_fin_tp: Function,
    hit_flag_tp: Function,
    argmax_tv: Function,
    moe_part: Function,
    ll_get: Function,
    mask_own: Function,
    ar_sum: Function,
    gdn_tc_h: Function,
}

/// FNEXT_SUMMODE: how the q8 activation sums for the K-quant min terms are formed (f16x default, dsumq, f32x).
pub(crate) fn sum_mode() -> fe_ggq::SumMode {
    match std::env::var("FNEXT_SUMMODE").as_deref() {
        Ok("dsumq") => fe_ggq::SumMode::DSumQ,
        Ok("f32x") => fe_ggq::SumMode::F32X,
        _ => fe_ggq::SumMode::F16X,
    }
}

/// Threads per output row for the bf16 GEMV: about 5 16-byte chunks per thread, at least 72 blocks.
fn tpr_for(k: usize, n: usize) -> u32 {
    let chunks = k / 8;
    let mut t = 8usize;
    while t < 256 && chunks > 5 * t {
        t *= 2;
    }
    while t < 256 && n.div_ceil(256 / t) < 72 {
        t *= 2;
    }
    t as u32
}

impl FKern {
    fn load() -> Result<FKern> {
        let m = Module::load(crate::K_FNEXT_CUBIN)?;
        let f = |n: &str| m.function(n);
        Ok(FKern {
            attn_prep: f("fx_attn_prep")?,
            attn: f("fx_attn")?,
            idx_kstore: f("fx_idx_kstore")?,
            idx_pool: f("fx_idx_pool")?,
            idx_score: f("fx_idx_score")?,
            topk: f("fx_topk")?,
            route: f("fx_route")?,
            moe_merge: f("fx_moe_merge")?,
            argmax: f("fx_argmax")?,
            gemv_h: f("fx_gemv_h")?,
            route8: f("fx_route8")?,
            moe_out: f("fx_moe_out")?,
            silu_ld: f("fx_silu_ld")?,
            hc_combine: f("fx_hc_combine")?,
            moe_fin: f("fx_moe_fin")?,
            scale_ld: f("fx_scale_ld")?,
            mtp_cat: f("fx_mtp_cat")?,
            moe_acc: f("fx_moe_acc")?,
            copy2d: f("fx_copy2d")?,
            to_host: f("fx_to_host")?,
            hit_flag: f("fx_hit_flag")?,
            attn6: f("fx_attn6")?,
            ll_pack: f("fx_ll_pack")?,
            ar_comb: f("fx_ar_comb")?,
            moe_fin_tp: f("fx_moe_fin_tp")?,
            hit_flag_tp: f("fx_hit_flag_tp")?,
            argmax_tv: f("fx_argmax_tv")?,
            moe_part: f("fx_moe_part")?,
            ll_get: f("fx_ll_get")?,
            mask_own: f("fx_mask_own")?,
            ar_sum: f("fx_ar_sum")?,
            gdn_tc_h: {
                let k = f("fx_gdn_chunk_tc_h")?;
                k.set_max_dynamic_smem(GDN_TC_SMEM)?;
                k
            },
            pattn: {
                let k = f("fx_pattn")?;
                k.set_max_dynamic_smem(PATTN_SMEM)?;
                k
            },
            gdn_chunk: {
                let k = f("fx_gdn_chunk")?;
                k.set_max_dynamic_smem(GDN_CHUNK_SMEM)?;
                k
            },
            gdn_tc: {
                let k = f("fx_gdn_chunk_tc")?;
                k.set_max_dynamic_smem(GDN_TC_SMEM)?;
                k
            },
            gdn_tc1: {
                let k = f("fx_gdn_chunk_tc1")?;
                k.set_max_dynamic_smem(GDN_TC_SMEM)?;
                k
            },
            hc_norm_bf16: f("fx_hc_norm_bf16")?,
            hc_comb_norm: f("fx_hc_comb_norm")?,
            hc_mix_res: f("fx_hc_mix_res")?,
            moe_comb: f("fx_moe_comb")?,
            moe_comb_f: f("fx_moe_comb_f")?,
            bcast_add: f("fx_bcast_add")?,
            argmax_part: f("fx_argmax_part")?,
            argmax_fin: f("fx_argmax_fin")?,
            conv_p: f("fx_gdn_conv_p")?,
            conv_state: f("fx_gdn_conv_state")?,
            idx_tc: {
                let k = f("fx_idx_score_tc")?;
                k.set_max_dynamic_smem(IDX_TC_SMEM)?;
                k
            },
            idx_tc3: {
                let k = f("fx_idx_score_tc3")?;
                k.set_max_dynamic_smem(IDX_TC_SMEM)?;
                k
            },
            gq: Box::leak(Box::new(Ggq::load()?)),
            _m: m,
        })
    }
}

/// bf16 GEMV y[r] (row stride ldy) = (acc ? y : 0) + scale * W x[r] (row stride ldx).
unsafe fn hrun(fk: &FKern, s: &Stream, l: &Lin, x: DevPtr, ldx: usize, y: DevPtr, ldy: usize, rows: usize, scale: f32,
    acc: bool) -> Result<()> {
    // bf16 weights with a dense output: quant's gemv_bf16 (PDL, split-K); the scale is folded by the callers
    if l.bf && ldy == l.n && scale == 1.0 && std::env::var("FNEXT_OWN_BF16").is_err() {
        return fk.gq.gemv_bf16(s, l.w.ptr(), l.n, l.k, x, ldx, rows, y, acc);
    }
    let tpr = tpr_for(l.k, l.n);
    let grid = l.n.div_ceil(256 / tpr as usize) as u32;
    // fx_gemv_h takes up to 8 rows: chunks (rows are independent, so the bits do not depend on the split)
    let mut r0 = 0;
    while r0 < rows {
        let nr = (rows - r0).min(8);
        s.launch(&fk.gemv_h, grid, 256, 0, (y.add::<f32>(r0 * ldy), x.add::<f32>(r0 * ldx), l.w.ptr(), l.k as u32,
            l.n as u32, nr as u32, scale, l.bf as u32, acc as u32, ldx as u32, ldy as u32, tpr))?;
        r0 += nr;
    }
    Ok(())
}

/// Hyper-connection mixer weights: norm, [down | inject] stacked (one GEMV, both scaled by 1/hc), up.
struct Hc {
    norm: DeviceBuffer<f32>,
    dinj: Lin,
    up: Lin,
    n_inj: usize,
}
impl Hc {
    fn load(g: &Gguf, pre: &str, norm: &str, down: &str, up: &str, inject: Option<&str>) -> Result<Hc> {
        let dinj = match inject {
            Some(i) => Lin::stacked(g, &format!("{pre}{down}"), &format!("{pre}{i}"))?,
            None => Lin::from_tensor(g, &format!("{pre}{down}"))?,
        };
        let n_inj = if inject.is_some() { tensor(g, &format!("{pre}{}", inject.unwrap()))?.n_rows() } else { 0 };
        Ok(Hc { norm: vecf(g, &format!("{pre}{norm}"))?, dinj, up: Lin::from_tensor(g, &format!("{pre}{up}"))?, n_inj })
    }
}

struct Gdn {
    qkv: QW,
    z: QW,
    ab: Lin,
    conv_w: DeviceBuffer<f32>,
    dt: DeviceBuffer<f32>,
    a: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    out: QW,
    conv_state: DeviceBuffer<f32>,
    ssm: DeviceBuffer<f32>,
    /// this step's projections / conv outputs / alpha|beta rows (kept per layer for the speculative commit)
    qkv_s: DeviceBuffer<f32>,
    conv_s: DeviceBuffer<f32>,
    ab_s: DeviceBuffer<f32>,
}

struct Attn {
    wq: QW,
    wk: QW,
    wv: QW,
    wo: QW,
    qn: DeviceBuffer<f32>,
    kn: DeviceBuffer<f32>,
    /// q8 K / V (int8 [cell][nkv][256]) and their scales ([cell][nkv][8] f32): address space for the whole context,
    /// memory mapped as the sequence grows (Fast::ensure_ctx)
    kq: VmBuffer,
    ks: VmBuffer,
    vq: VmBuffer,
    vs: VmBuffer,
    iq: Lin,
    ik: Lin,
    iqn: DeviceBuffer<f32>,
    ikn: DeviceBuffer<f32>,
    ring: DeviceBuffer<f32>,
    pooled: VmBuffer,
}

enum Mixer {
    Gdn(Gdn),
    Attn(Attn),
}

struct Ffn {
    /// router logits (n_expert rows) + the shared expert's gate logit (1 row)
    gate_inp: Lin,
    shg: QW,
    shu: QW,
    shd: QW,
    ty_gu: GgmlType,
    ty_d: GgmlType,
    gtab: DeviceBuffer<u64>,
    utab: DeviceBuffer<u64>,
    dtab: DeviceBuffer<u64>,
    /// FNEXT_TP=2: 1 for the experts this card owns (unused otherwise)
    own: DeviceBuffer<u32>,
    /// FNEXT_TP=2: this card's shared-expert down columns padded with zero blocks to a multiple of 128 (the prompt
    /// int8 GEMM's K granularity); the padding adds exact zeros
    shd_pad: Option<QW>,
}

struct Ple {
    key: Lin,
    value: Lin,
    nk: DeviceBuffer<f32>,
    nq: DeviceBuffer<f32>,
    nc: DeviceBuffer<f32>,
    conv_w: DeviceBuffer<f32>,
    hist: DeviceBuffer<f32>,
}

struct Layer {
    hca: Hc,
    hcf: Hc,
    mix: Mixer,
    ffn: Ffn,
    ple: Option<Ple>,
}

/// Pinned memory shared by the graph and the CPU expert thread.
struct Shared {
    flags: HostBuffer<u32>, // [0..64) ready[l], [64..128) done[l]
    ids: HostBuffer<i32>,   // [MAX_ROWS * topk]
    wts: HostBuffer<f32>,
    x: HostBuffer<f32>,     // [MAX_ROWS][n_embd]
    y: HostBuffer<f32>,     // [MAX_ROWS * topk][n_embd]
}
unsafe impl Sync for Shared {}

struct Work {
    prm: DeviceBuffer<u32>,
    res: DeviceBuffer<f32>,
    xn: DeviceBuffer<f32>,
    lo: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    inj: DeviceBuffer<f32>,
    mixed: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    z: DeviceBuffer<f32>,
    ab: DeviceBuffer<f32>,
    conv: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    qf: DeviceBuffer<f32>,
    kx: DeviceBuffer<f32>,
    vx: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    ikx: DeviceBuffer<f32>,
    iqv: DeviceBuffer<f32>,
    score: DeviceBuffer<f32>,
    sel: DeviceBuffer<u32>,
    seln: DeviceBuffer<u32>,
    rl: DeviceBuffer<f32>,
    ids: DeviceBuffer<i32>,
    wts: DeviceBuffer<f32>,
    sg: DeviceBuffer<f32>,
    su: DeviceBuffer<f32>,
    sh: DeviceBuffer<f32>,
    glog: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    ey: DeviceBuffer<f32>,
    moe: DeviceBuffer<f32>,
    emb: DeviceBuffer<f32>,
    pemb: DeviceBuffer<f32>,
    pkey: DeviceBuffer<f32>,
    pval: DeviceBuffer<f32>,
    pkn: DeviceBuffer<f32>,
    pqn: DeviceBuffer<f32>,
    pgated: DeviceBuffer<f32>,
    pcn: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    tok: DeviceBuffer<u32>,
    /// two-stage argmax parts [rows][ARGMAX_PARTS][3]
    amx: DeviceBuffer<f32>,
    /// trunk residual (hc-wide) after the last layer, per row of the step (the MTP block reads it)
    pub hsave: DeviceBuffer<f32>,
    /// the same rows of a multi-lane step (every lane's rows; lanes keep their own hsave for their MTP chain)
    hbat: DeviceBuffer<f32>,
    /// greedy-pick probability per row of a multi-lane MTP pass
    pmx: DeviceBuffer<f32>,
    /// rope tables of the rows / of the QSA blocks they complete ([n][32] cos, sin): the step parameters for decode
    cs: DevPtr,
    sn: DevPtr,
    pcs: DevPtr,
    psn: DevPtr,
    xa: Q8Act,
    ea: Q8Act,
    sa: Q8Act,
    ha: Q8Act,
    /// FNEXT_TP=2: this card's partial block output as LL (value, tag) pairs, copied to the host slot
    llp: DeviceBuffer<u32>,
    /// FNEXT_TP=2: (token, logit bits) per row of this card's half of the vocabulary
    tv: DeviceBuffer<u32>,
}

/// Per-step timing / routing statistics.
#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub steps: u64,
    pub rows: u64,
    pub pairs: u64,
    pub misses: u64,
    /// CPU expert time (µs) summed over layers
    pub cpu_us: u64,
    /// misses an LRU of FNEXT_SIM_DYN extra slots per layer would have caught
    pub sim_hits: u64,
    /// host time per step parts (µs, summed): prepare (embeddings, PLE rows), graph (launch to done), refill
    pub prep_us: u64,
    pub graph_us: u64,
    pub refill_us: u64,
    /// misses per (layer, owner card): index l * 2 + card (statistics only)
    pub miss_lc: Vec<u64>,
}

pub struct Fast {
    pub c: Config,
    pub g: Arc<Gguf>,
    kn: Kern,
    fk: FKern,
    q: Ggq,
    pub s: Stream,
    layers: Vec<Layer>,
    head: Hc,
    output: QW,
    w: Work,
    pub ctx: usize,
    pub tokens: Vec<u32>,
    graphs: Vec<Option<Graph>>,
    prm_h: HostBuffer<u32>,
    emb_h: HostBuffer<f32>,
    ple_h: HostBuffer<f32>,
    tok_h: HostBuffer<u32>,
    logit_h: HostBuffer<f32>,
    /// per-row greedy probabilities of a multi-lane MTP pass
    pmx_h: HostBuffer<f32>,
    sh: Arc<Shared>,
    /// host mirror of the expert placement: resident[l][e] (0 no, 1 static, 2 dynamic slot)
    pub resident: Arc<Vec<Vec<AtomicU8>>>,
    /// last step that used expert (l, e) (written by the CPU thread)
    last_use: Arc<Vec<Vec<AtomicU64>>>,
    /// misses of the last step (written by the CPU thread)
    missed: Arc<std::sync::Mutex<Vec<(usize, usize)>>>,
    step_no: Arc<AtomicU64>,
    dynp: Option<DynCache>,
    _arena: Vec<DeviceBuffer<u8>>,
    pub expert_bytes_on_card: usize,
    tabs_h: Vec<[Vec<u64>; 3]>,
    pub n_resident: usize,
    cpu: Option<std::thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    cur_rows: Arc<AtomicU32>,
    pub counts: Arc<Vec<Vec<std::sync::atomic::AtomicU64>>>,
    pub stats: Arc<std::sync::Mutex<Stats>>,
    done_ev: Event,
    /// copy the last row's logits to the host each step (sampling); greedy only needs the token
    pub want_logits: bool,
    /// rows of the last trunk step whose greedy pick came from a row with a non-finite maximum or probability (bit r =
    /// logit row r). Only FNEXT_TP=2 brings those two numbers to the host; without it the mask stays 0.
    pub bad_rows: u32,
    /// the running request's cancel flag (set by the server when its client goes away): a prompt stops between
    /// groups and between a group's layers with [`Cancelled`]; the lane's state is then incomplete (restore a
    /// snapshot or reset it)
    pub cancel: Option<Arc<AtomicBool>>,
    /// experts uploaded into dynamic slots so far
    pub uploads: u64,
    mtp: Option<Mtp>,
    pw: Option<prefill::PWork>,
    /// prompt workspace: (chunk rows, streamed expert group, group tokens); allocated per prompt from LRU blocks
    pcfg: (usize, usize, usize),
    hm: hostmem::HostMem,
    /// context cells with KV memory mapped
    pub ctx_mapped: usize,
    /// side stream + fork / join events of the per-layer CPU hand-off
    s2: Stream,
    ev_fork: Vec<Event>,
    ev_join: Vec<Event>,
    mtp_graphs: Vec<Option<Graph>>,
    /// MTP drafting statistics: (drafted, accepted)
    pub spec_stats: (u64, u64),
    /// probability of the last MTP draft
    pub draft_p: f32,
    /// moving average of the fraction of drafts accepted
    pub acc_ema: f32,
    /// speculative step time split (ms, summed): chained drafts, verify, commit, catch-up; count
    pub spec_ms: [f64; 5],
    /// MTP pass time split (ms, summed): host prep, graph; count
    pub mtp_ms: [f64; 3],
    /// FNEXT_MTP_COUNT: routing counts of the MTP block's experts over the MTP passes (statistics)
    pub mtp_counts: Vec<u64>,
    /// prompt-cache snapshots of the current sequence (ascending positions; the server keeps the first + last two)
    pub snaps: Vec<Snap>,
    /// CPU expert pool + pinned pair rows for the slotless experts of short prompts
    pcpu: Option<fe_ggq::cpu::CpuMoe>,
    ycpu: Option<HostBuffer<f32>>,
    /// parked lanes (None at the current lane's index) and the current lane
    lanes: Vec<Option<lanes::LaneState>>,
    cur_lane: usize,
    /// the step's (lane, first row, rows) segments for enqueue (empty: the current lane alone)
    plan: Vec<(usize, usize, usize)>,
    /// batched multi-lane step graphs by lane set
    lane_graphs: lanes::GraphCache,
    /// most cells the current lane may map (the lane budget; the whole pool when it runs alone)
    pub ctx_cap: usize,
    /// PLE table rows of the current step
    ple_idx: Vec<u32>,
    /// raised by the host once ple_h holds the step's PLE rows (the graph waits for it before the PLE layer)
    ple_flag: HostBuffer<u32>,
    /// card bytes each prompt-workspace shape took (chunk, ctx, group, group tokens)
    pw_need: std::collections::HashMap<(usize, usize, usize, usize), usize>,
    /// stop drafting when the MTP's probability of its draft falls below this
    pub p_min: f32,
    /// FNEXT_DUMP with FNEXT_NO_GRAPH (debug): per (layer, stage) the bits hash of every row of the step
    pub dump: Vec<(usize, usize, Vec<u64>)>,
    /// blocks of the active device (FNEXT_GPUS=2 splits the layers; one device: 0..n_layer)
    lo: usize,
    hi: usize,
    /// pinned residual rows handed from device 0's step graph to device 1's
    xfer: HostBuffer<f32>,
    /// the parked device set of a layer split (device 0 while device 1 is active)
    other: Option<Box<DevSet>>,
    /// FNEXT_TP=2: tensor parallel (both cards run every layer on their half; `c` is the active card's half config)
    pub tp: bool,
    /// the active card under FNEXT_TP=2 (1 between calls: head, MTP, logits)
    tp_rank: usize,
    /// the whole model's config (== c without FNEXT_TP)
    pub cfull: Config,
    /// expert owner card per (layer, expert) under FNEXT_TP=2 (all 0 otherwise)
    own_h: Arc<Vec<Vec<u8>>>,
    /// all-reduce slots in pinned host memory: [card][2 alternating][MAX_ROWS * n_embd] LL (value, tag) pairs
    ar_h: HostBuffer<u32>,
    /// step epoch (all-reduce tags are epoch << 8 + reduction index)
    epoch: u32,
    /// FNEXT_TP=2: card 0's graph launcher thread
    launcher: Option<tp::Launcher>,
    /// FNEXT_TP=2 prompts: pinned staging of the all-reduce [card][2 slots][chunk rows * n_embd]
    pa_h: Option<HostBuffer<f32>>,
    /// FNEXT_TP=2: the head mixer + first half of the output rows on card 0 (swapped in with card 0; None on card 1,
    /// which uses `head` / the second half of `output`)
    head_tp: Option<(Hc, QW)>,
    /// FNEXT_TP=2: (token, logit bits, half-softmax probability bits) per row from each card's half of the vocabulary
    /// [card][MAX_ROWS][3]
    tv_h: HostBuffer<u32>,
    /// FNEXT_TP=2: card 0's MTP head graph (its half of the vocabulary for the MTP block's head row; swapped in with
    /// card 0) and the LL slot that row travels in
    mtp_head_g: Vec<Option<Graph>>,
    mtp_h: HostBuffer<u32>,
    /// last field: dropped after every buffer and stream above (fields drop in declaration order)
    cu: Arc<fe_cuda::Context>,
}

/// FNEXT_GPUS=2 default split: blocks 0..26 on device 0 (it also has no head / MTP block, so it gets more layers).
const SPLIT_DEFAULT: usize = 26;

/// What each device of a layer split has for itself. Fast holds the active device's set in its own fields (the last
/// device between calls: head, MTP block, logits); `Fast::swap_dev` exchanges it with the parked one.
struct DevSet {
    kn: Kern,
    fk: FKern,
    q: Ggq,
    s: Stream,
    s2: Stream,
    w: Work,
    graphs: Vec<Option<Graph>>,
    lane_graphs: lanes::GraphCache,
    dynp: Option<DynCache>,
    arena: Vec<DeviceBuffer<u8>>,
    done_ev: Event,
    pw: Option<prefill::PWork>,
    pw_need: std::collections::HashMap<(usize, usize, usize, usize), usize>,
    lo: usize,
    hi: usize,
    /// FNEXT_TP=2: the card's own layers (half weights, its experts, its state), config, tables, hand-off events
    layers: Vec<Layer>,
    c: Config,
    tabs_h: Vec<[Vec<u64>; 3]>,
    ev_fork: Vec<Event>,
    ev_join: Vec<Event>,
    tp_rank: usize,
    head_tp: Option<(Hc, QW)>,
    mtp_head_g: Vec<Option<Graph>>,
    /// last field: dropped after every buffer above
    cu: Arc<fe_cuda::Context>,
}

/// The decode step buffers of one device.
fn alloc_work(c: &Config, ctx: usize) -> Result<Work> {
    let r = MAX_ROWS;
    let a = |n: usize| DeviceBuffer::<f32>::zeroed(n);
    let hd = c.hc_dim();
    let topk = c.n_expert_used;
    let kvd = c.n_head_kv * c.head_dim;
    let mut w = Work {
        prm: DeviceBuffer::zeroed(MAX_LANES * P_WORDS)?,
        res: a(r * hd)?,
        xn: a(r * hd)?,
        lo: a(r * (c.hc_lr + c.hc))?,
        gate: a(r * hd)?,
        inj: a(r * c.hc)?,
        mixed: a(r * c.n_embd)?,
        out: a(r * c.n_embd)?,
        qkv: a(r * c.conv_dim())?,
        z: a(r * c.ssm_inner)?,
        ab: a(r * 2 * c.ssm_vh)?,
        conv: a(r * c.conv_dim())?,
        o: a(r * c.ssm_inner)?,
        y: a(r * c.ssm_inner.max(c.n_head * c.head_dim))?,
        qf: a(r * 2 * c.n_head * c.head_dim)?,
        kx: a(r * kvd)?,
        vx: a(r * kvd)?,
        q: a(r * c.n_head * c.head_dim)?,
        att: a(r * c.n_head * c.head_dim)?,
        ikx: a(r * c.idx_dim)?,
        iqv: a(r * c.idx_heads * c.idx_dim)?,
        score: a(r * (ctx / c.kpool + 1))?,
        sel: DeviceBuffer::zeroed(r * (c.idx_top_k + c.kpool - 1))?,
        seln: DeviceBuffer::zeroed(r)?,
        rl: a(r * (c.n_expert + 1))?,
        ids: DeviceBuffer::zeroed(r * topk)?,
        wts: a(r * topk)?,
        sg: a(r * c.n_ff_shexp)?,
        su: a(r * c.n_ff_shexp)?,
        sh: a(r * c.n_embd)?,
        glog: a(r)?,
        h: a(r * topk * c.n_ff_exp)?,
        ey: a(r * topk * c.n_embd)?,
        moe: a(r * c.n_embd)?,
        emb: a(r * c.n_embd)?,
        pemb: a(r * c.n_embd)?,
        pkey: a(r * hd)?,
        pval: a(r * c.n_embd)?,
        pkn: a(r * hd)?,
        pqn: a(r * hd)?,
        pgated: a(r * hd)?,
        pcn: a(r * hd)?,
        logits: a(r * c.n_vocab)?,
        tok: DeviceBuffer::zeroed(r)?,
        amx: DeviceBuffer::zeroed(r * ARGMAX_PARTS * 3)?,
        hsave: a(r * hd)?,
        hbat: a(r * hd)?,
        pmx: a(r)?,
        cs: DevPtr::NULL,
        sn: DevPtr::NULL,
        pcs: DevPtr::NULL,
        psn: DevPtr::NULL,
        xa: Q8Act::alloc(r, 12288)?,
        ea: Q8Act::alloc(r, c.n_embd)?,
        sa: Q8Act::alloc(r, c.n_ff_shexp)?,
        ha: Q8Act::alloc(r * topk, c.n_ff_exp)?,
        llp: DeviceBuffer::zeroed(r * c.n_embd * 2)?,
        tv: DeviceBuffer::zeroed(3 * r)?,
    };
    let sm = sum_mode();
    for q8 in [&mut w.xa, &mut w.ea, &mut w.sa, &mut w.ha] {
        q8.sum_mode = sm;
    }
    let pp = w.prm.ptr();
    w.cs = pp.add::<u32>(P_CS);
    w.sn = pp.add::<u32>(P_SN);
    w.pcs = pp.add::<u32>(P_PCS);
    w.psn = pp.add::<u32>(P_PSN);
    Ok(w)
}

fn quant_mixed(q: &Ggq, s: &Stream, x: DevPtr, rows: usize, k: usize, a: &mut Q8Act) -> Result<()> {
    q.quantize_q8_1(s, x, k, false, rows, k, a)
}

impl Fast {
    /// `counts`: routing counts [layer][expert] choosing the static set (None: no experts on the card);
    /// `expert_mib`: cap on the expert arena (None: all free memory minus a reserve).
    pub fn load(path: &std::path::Path, ctx: usize, counts: Option<&Vec<Vec<u64>>>, expert_mib: Option<usize>,
        cpu_threads: usize) -> Result<Fast> {
        // the launching thread on CPU 0, the CPU expert thread on 1, the uploader on 2, the miss pool on FNEXT_CPU_POOL
        crate::cores::init_rayon();
        if std::env::var("FNEXT_NO_PIN").is_err() {
            fe_ggq::cpu::pool::pin_current(crate::cores::plan().main);
        }
        // FNEXT_GPUS=2: layer split over two cards in one process. Blocks 0..FNEXT_SPLIT run on device 0, the rest plus
        // the head and the MTP block on device 1; each card keeps the LRU expert slots of its own layers.
        let ngpu: usize = std::env::var("FNEXT_GPUS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        if !(1..=2).contains(&ngpu) {
            bail!("FNEXT_GPUS={ngpu}: 1 or 2");
        }
        // FNEXT_TP=2: tensor parallel over two cards instead of the layer split (tp.rs)
        let tp = std::env::var("FNEXT_TP").is_ok_and(|v| v == "2");
        let ngpu = if tp { 2 } else { ngpu };
        if ngpu > 1 && std::env::var("CUDA_MODULE_LOADING").is_err() {
            // lazy kernel loading inside a graph instantiation waits for the device, which a layer split can have
            // blocked (first verify graph of an MTP run hung there): load every kernel when its module loads
            // SAFETY: before the driver initialises; the engine's load runs before its other threads exist
            unsafe { std::env::set_var("CUDA_MODULE_LOADING", "EAGER") };
        }
        let cus: Vec<Arc<fe_cuda::Context>> = (0..ngpu).map(|i| fe_cuda::Context::new(i as i32)).collect::<fe_cuda::Result<_>>()?;
        let g = Arc::new(crate::open_model(path)?);
        let c = Config::from_gguf(&g)?;
        let split = if ngpu == 1 || tp {
            c.n_layer
        } else {
            std::env::var("FNEXT_SPLIT").ok().and_then(|v| v.parse().ok()).unwrap_or(SPLIT_DEFAULT).clamp(2, c.n_layer - 1)
        };
        let last = ngpu - 1;
        let dev_of = |l: usize| if l < split { 0 } else { last };
        let range = |dv: usize| -> (usize, usize) {
            if tp { (0, c.n_layer) } else if dv == last { (if ngpu == 1 { 0 } else { split }, c.n_layer) } else { (0, split) }
        };
        // the per-card config (half the heads under FNEXT_TP=2) and the expert owners
        let ch = if tp { tp::half_config(&c)? } else { c.clone() };
        let own = if tp { tp::owners(&c, counts) } else { vec![vec![0u8; c.n_expert]; c.n_layer] };
        if tp {
            eprintln!("fnext-fast: tensor parallel: every block on device 0 ({}) and device 1 ({}), half the heads each, experts \
                split by owner; head + MTP on device 1", cus[0].name(), cus[1].name());
        }
        if ngpu > 1 && !tp {
            eprintln!("fnext-fast: layer split: blocks 0..{split} on device 0 ({}), {split}..{} + head + MTP on device 1 ({})",
                cus[0].name(), c.n_layer, cus[1].name());
        }
        // per device: the model kernels, the k-gguf kernels, the compute + side streams, the step-done event
        let mut kits = vec![];
        for cu in &cus {
            cu.bind()?;
            kits.push((Kern::load()?, FKern::load()?, Ggq::load()?, Stream::new()?, Stream::new()?, Event::new()?));
        }
        let t0 = std::time::Instant::now();
        let mut layers = vec![];
        let kvd = c.n_head_kv * c.head_dim;
        let mut layers0: Vec<Layer> = vec![];
        if tp {
            for dv in 0..2 {
                cus[dv].bind()?;
                let ord = cus[dv].ordinal();
                let mut v = vec![];
                for il in 0..c.n_layer {
                    v.push(tp::load_layer(&g, &c, &ch, il, dv, ctx, ord)?);
                }
                if dv == 0 { layers0 = v } else { layers = v }
            }
        }
        for il in 0..if tp { 0 } else { c.n_layer } {
            let cu = &cus[dev_of(il)];
            cu.bind()?;
            let ord = cu.ordinal();
            let p = format!("blk.{il}.");
            let hca = Hc::load(&g, &p, "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", Some("hc_attn_inject.weight"))?;
            let hcf = Hc::load(&g, &p, "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", Some("hc_ffn_inject.weight"))?;
            let mix = if c.is_attn(il) {
                Mixer::Attn(Attn {
                    wq: QW::load(&g, &format!("{p}attn_q.weight"))?,
                    wk: QW::load(&g, &format!("{p}attn_k.weight"))?,
                    wv: QW::load(&g, &format!("{p}attn_v.weight"))?,
                    wo: QW::load(&g, &format!("{p}attn_output.weight"))?,
                    qn: vecf(&g, &format!("{p}attn_q_norm.weight"))?,
                    kn: vecf(&g, &format!("{p}attn_k_norm.weight"))?,
                    kq: VmBuffer::reserve(ctx * kvd, ord)?,
                    ks: VmBuffer::reserve(ctx * kvd / 32 * 4, ord)?,
                    vq: VmBuffer::reserve(ctx * kvd, ord)?,
                    vs: VmBuffer::reserve(ctx * kvd / 32 * 4, ord)?,
                    iq: Lin::from_tensor(&g, &format!("{p}indexer.q_proj.weight"))?,
                    ik: Lin::from_tensor(&g, &format!("{p}indexer.k_proj.weight"))?,
                    iqn: vecf(&g, &format!("{p}indexer.q_norm.weight"))?,
                    ikn: vecf(&g, &format!("{p}indexer.k_norm.weight"))?,
                    ring: DeviceBuffer::zeroed(ring_cells() * c.idx_dim)?,
                    pooled: VmBuffer::reserve((ctx / c.kpool + 1) * c.idx_dim * 4, ord)?,
                })
            } else {
                Mixer::Gdn(Gdn {
                    qkv: QW::load(&g, &format!("{p}attn_qkv.weight"))?,
                    z: QW::load(&g, &format!("{p}attn_gate.weight"))?,
                    ab: Lin::stacked(&g, &format!("{p}ssm_alpha.weight"), &format!("{p}ssm_beta.weight"))?,
                    conv_w: vecf(&g, &format!("{p}ssm_conv1d.weight"))?,
                    dt: vecf(&g, &format!("{p}ssm_dt.bias"))?,
                    a: vecf(&g, &format!("{p}ssm_a"))?,
                    norm: vecf(&g, &format!("{p}ssm_norm.weight"))?,
                    out: QW::load(&g, &format!("{p}ssm_out.weight"))?,
                    conv_state: DeviceBuffer::zeroed((c.ssm_conv - 1) * c.conv_dim())?,
                    ssm: DeviceBuffer::zeroed(c.ssm_vh * c.ssm_state * c.ssm_state)?,
                    qkv_s: DeviceBuffer::zeroed(MAX_ROWS * c.conv_dim())?,
                    conv_s: DeviceBuffer::zeroed(MAX_ROWS * c.conv_dim())?,
                    ab_s: DeviceBuffer::zeroed(MAX_ROWS * 2 * c.ssm_vh)?,
                })
            };
            let tg = tensor(&g, &format!("{p}ffn_gate_exps.weight"))?;
            let td = tensor(&g, &format!("{p}ffn_down_exps.weight"))?;
            let ffn = Ffn {
                gate_inp: Lin::stacked(&g, &format!("{p}ffn_gate_inp.weight"), &format!("{p}ffn_gate_inp_shexp.weight"))?,
                shg: QW::load(&g, &format!("{p}ffn_gate_shexp.weight"))?,
                shu: QW::load(&g, &format!("{p}ffn_up_shexp.weight"))?,
                shd: QW::load(&g, &format!("{p}ffn_down_shexp.weight"))?,
                ty_gu: tg.ty,
                ty_d: td.ty,
                gtab: DeviceBuffer::zeroed(c.n_expert)?,
                utab: DeviceBuffer::zeroed(c.n_expert)?,
                dtab: DeviceBuffer::zeroed(c.n_expert)?,
                own: DeviceBuffer::zeroed(1)?,
                shd_pad: None,
            };
            let ple = if il == c.ple_layer {
                Some(Ple {
                    key: Lin::from_tensor(&g, &format!("{p}ple_key.weight"))?,
                    value: Lin::from_tensor(&g, &format!("{p}ple_value.weight"))?,
                    nk: vecf(&g, &format!("{p}ple_norm_key.weight"))?,
                    nq: vecf(&g, &format!("{p}ple_norm_query.weight"))?,
                    nc: vecf(&g, &format!("{p}ple_norm_conv.weight"))?,
                    conv_w: vecf(&g, &format!("{p}ple_conv1d.weight"))?,
                    hist: DeviceBuffer::zeroed((c.ple_conv - 1) * c.ple_ngram * c.hc_dim())?,
                })
            } else {
                None
            };
            layers.push(Layer { hca, hcf, mix, ffn, ple });
        }
        if split <= c.ple_layer && !tp {
            bail!("FNEXT_SPLIT={split}: the PLE layer {} must be on device 0", c.ple_layer);
        }
        let mut head0 = None;
        if tp {
            // card 0: the head mixer and the first half of the output rows (card 1 keeps all of them: the MTP head)
            cus[0].bind()?;
            let h = Hc::load(&g, "", "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight", None)?;
            let t = tensor(&g, "output.weight")?;
            let half = c.n_vocab / 2;
            let o = QW { buf: DeviceBuffer::from_slice(&g.data(t)[..half * t.row_bytes()])?, ty: t.ty, k: t.ne0(), n: half };
            head0 = Some((h, o));
        }
        cus[last].bind()?;
        let head = Hc::load(&g, "", "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight", None)?;
        let output = if tp {
            // card 1: the second half of the output rows (trunk and MTP heads are vocab-split)
            let t = tensor(&g, "output.weight")?;
            let half = c.n_vocab / 2;
            QW { buf: DeviceBuffer::from_slice(&g.data(t)[half * t.row_bytes()..])?, ty: t.ty, k: t.ne0(), n: c.n_vocab - half }
        } else {
            QW::load(&g, "output.weight")?
        };
        let r = MAX_ROWS;
        let hd = c.hc_dim();
        let mut works = vec![];
        for cu in &cus {
            cu.bind()?;
            works.push(alloc_work(&c, ctx)?);
        }
        // the first 8192 cells of KV are mapped up front (more on demand)
        let kvd = ch.n_head_kv * ch.head_dim;
        for (il, l) in layers.iter_mut().enumerate().map(|(i, l)| (i, l, 1usize)).chain(layers0.iter_mut().enumerate().map(|(i, l)| (i, l, 0usize)))
            .map(|(i, l, dv)| ((i, dv), l)) {
            if let Mixer::Attn(a) = &mut l.mix {
                if tp { cus[il.1].bind()? } else { cus[dev_of(il.0)].bind()? };
                let c0 = 8192.min(ctx);
                a.kq.grow(c0 * kvd)?;
                a.vq.grow(c0 * kvd)?;
                a.ks.grow(c0 * kvd / 32 * 4)?;
                a.vs.grow(c0 * kvd / 32 * 4)?;
                a.pooled.grow((c0 / c.kpool + 1) * c.idx_dim * 4)?;
            }
        }
        cus[last].bind()?;
        let kvd = c.n_head_kv * c.head_dim;
        let mtp = if std::env::var("FNEXT_MTP").map_or(false, |v| v != "0") && c.n_layer_all > c.n_layer {
            let mut m = Mtp::load(&g, &c, ctx, cus[last].ordinal())?;
            let c0 = 8192.min(ctx);
            m.at.kq.grow(c0 * kvd)?;
            m.at.vq.grow(c0 * kvd)?;
            m.at.ks.grow(c0 * kvd / 32 * 4)?;
            m.at.vs.grow(c0 * kvd / 32 * 4)?;
            m.at.pooled.grow((c0 / c.kpool + 1) * c.idx_dim * 4)?;
            Some(m)
        } else {
            None
        };
        let chunk: usize = std::env::var("FNEXT_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
        let group: usize = std::env::var("FNEXT_GROUP").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
        let gtok: usize = std::env::var("FNEXT_GROUP_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(16384);
        let pcfg = (chunk, group, gtok);
        eprintln!("fnext-fast: dense weights + caches on the card in {:.1} s{}", t0.elapsed().as_secs_f32(),
            if mtp.is_some() { " (with the MTP block, all its experts)" } else { "" });
        // ---- prompt workspace sizes, probed while the cards are still free ----
        let mut probes = vec![];
        for dv in 0..ngpu {
            cus[dv].bind()?;
            let t = std::time::Instant::now();
            probes.push(probe_pw(&c, &g, pcfg, ctx, tp)?);
            if std::env::var("FNEXT_VERBOSE").is_ok() {
                eprintln!("fnext-fast: device {dv}: {} prompt workspace shapes probed in {:.0} ms", probes[dv].len(),
                    t.elapsed().as_secs_f64() * 1e3);
            }
        }
        // ---- expert placement per device: a static set (FNEXT_DYN_FRAC < 1) and LRU slots for its layers ----
        // ---- lanes (FNEXT_SLOTS, default 1): per-sequence state on each layer's device, before the expert budget ----
        let n_lanes: usize = std::env::var("FNEXT_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).clamp(1, 8);

        let lanes = lanes::alloc_lanes(&c, &ch, ctx, &layers, &layers0, mtp.as_ref(), r * hd, n_lanes,
            &|l| if tp { cus[1].clone() } else { cus[dev_of(l)].clone() }, &cus[last], &cus[0])?;
        let mut resident = vec![vec![0u8; c.n_expert]; c.n_layer];
        let mut arenas: Vec<Vec<DeviceBuffer<u8>>> = vec![];
        let mut dyns: Vec<Option<DynCache>> = vec![];
        let mut on_card = 0usize;
        let mut n_res = 0usize;
        for dv in 0..ngpu {
            let cu = &cus[dv];
            cu.bind()?;
            let (lo, hi) = range(dv);
            let mut dyn_budget = 0usize;
            let mut arena = vec![];
            if let Some(cnt) = counts {
                let (free, _) = fe_cuda::mem_info()?;
                let reserve = std::env::var("FNEXT_RESERVE_MIB").ok().and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(if tp { 400 } else { 800 }) << 20;
                let mut budget = free.saturating_sub(reserve);
                if let Some(mib) = expert_mib {
                    budget = budget.min(mib << 20);
                }
                // FNEXT_DYN_FRAC of the budget goes to LRU slots refilled from the misses (default all of it)
                let frac: f64 = std::env::var("FNEXT_DYN_FRAC").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
                dyn_budget = (budget as f64 * frac) as usize;
                budget -= dyn_budget;
                // rank (layer, expert) by count per byte
                let mut cand = vec![];
                for l in lo..hi {
                    let eb = expert_bytes(&g, l)?;
                    for e in 0..c.n_expert {
                        if cnt[l][e] > 0 && (!tp || own[l][e] as usize == dv) {
                            cand.push((cnt[l][e] as f64 / eb as f64, l, e, eb));
                        }
                    }
                }
                cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let chunk = 256usize << 20;
                let mut cur: Option<(DeviceBuffer<u8>, usize)> = None;
                let mut tabs: Vec<[Vec<u64>; 3]> = (0..c.n_layer).map(|_| [vec![0u64; c.n_expert], vec![0u64; c.n_expert], vec![0u64; c.n_expert]]).collect();
                let (mut dev_bytes, mut dev_n) = (0usize, 0usize);
                for (_, l, e, eb) in cand {
                    if dev_bytes + eb > budget {
                        continue;
                    }
                    if cur.as_ref().map_or(true, |(b, used)| used + eb > b.len()) {
                        if let Some((b, _)) = cur.take() {
                            arena.push(b);
                        }
                        let want = chunk.min(budget - dev_bytes).max(eb);
                        cur = Some((DeviceBuffer::alloc(want)?, 0));
                    }
                    let (b, used) = cur.as_mut().unwrap();
                    let p = format!("blk.{l}.");
                    let mut off = *used;
                    for (i, nm) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].iter().enumerate() {
                        let t = tensor(&g, &format!("{p}{nm}.weight"))?;
                        let src = g.slice(t, e);
                        fe_cuda::memcpy_htod(b.ptr().byte_add(off), src)?;
                        tabs[l][i][e] = b.ptr().byte_add(off).0;
                        off += src.len().div_ceil(256) * 256;
                    }
                    *used = off;
                    dev_bytes += eb;
                    dev_n += 1;
                    resident[l][e] = 1;
                }
                if let Some((b, _)) = cur.take() {
                    arena.push(b);
                }
                for l in lo..hi {
                    let ly = if tp && dv == 0 { &mut layers0[l] } else { &mut layers[l] };
                    ly.ffn.gtab.copy_from(&tabs[l][0])?;
                    ly.ffn.utab.copy_from(&tabs[l][1])?;
                    ly.ffn.dtab.copy_from(&tabs[l][2])?;
                }
                on_card += dev_bytes;
                n_res += dev_n;
                if dev_n > 0 {
                    eprintln!("fnext-fast: device {dv}: {} experts on the card ({:.0} MiB, {:.1}% of all) in {:.1} s", dev_n,
                        dev_bytes as f64 / 1048576.0, 100.0 * dev_n as f64 / (c.n_layer * c.n_expert) as f64, t0.elapsed().as_secs_f32());
                }
            }
            let cap = if tp { c.n_expert / 2 } else { c.n_expert };
            let dynp = if dyn_budget > 0 { Some(DynCache::new(&g, &c, dyn_budget, cu, lo, hi, cap)?) } else { None };
            let n_dyn = dynp.as_ref().map_or(0, |d| d.per_layer);
            if n_dyn > 0 {
                eprintln!("fnext-fast: device {dv}: + {} LRU slots per layer for blocks {lo}..{hi} ({:.0} MiB)", n_dyn,
                    dyn_budget as f64 / 1048576.0);
            }
            // pre-fill the LRU slots with the most used experts that are not static (counts), so it starts warm
            let mut dynp = dynp;
            if let (Some(dc), Some(cnt)) = (dynp.as_mut(), counts) {
                let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
                for l in lo..hi {
                    let mut order: Vec<usize> = (0..c.n_expert)
                        .filter(|&e| resident[l][e] == 0 && cnt[l][e] > 0 && (!tp || own[l][e] as usize == dv)).collect();
                    order.sort_by(|&a, &b| cnt[l][b].cmp(&cnt[l][a]));
                    let mut tabs = [vec![0u64; c.n_expert], vec![0u64; c.n_expert], vec![0u64; c.n_expert]];
                    let ly = if tp && dv == 0 { &mut layers0[l] } else { &mut layers[l] };
                    ly.ffn.gtab.copy_to(&mut tabs[0])?;
                    ly.ffn.utab.copy_to(&mut tabs[1])?;
                    ly.ffn.dtab.copy_to(&mut tabs[2])?;
                    for (sl, &e) in order.iter().take(dc.per_layer).enumerate() {
                        let base = dc.slot_ptr(l, sl);
                        for i in 0..3 {
                            let t = tensor(&g, &format!("blk.{l}.{}.weight", names[i]))?;
                            let src = g.slice(t, e);
                            fe_cuda::memcpy_htod(base.byte_add(dc.offs[l][i]), src)?;
                            tabs[i][e] = base.0 + dc.offs[l][i] as u64;
                        }
                        dc.slots[l][sl] = Some(e);
                        resident[l][e] = 2;
                    }
                    let ly = if tp && dv == 0 { &mut layers0[l] } else { &mut layers[l] };
                    ly.ffn.gtab.copy_from(&tabs[0])?;
                    ly.ffn.utab.copy_from(&tabs[1])?;
                    ly.ffn.dtab.copy_from(&tabs[2])?;
                }
                eprintln!("fnext-fast: device {dv}: LRU slots pre-filled from the counts ({:.1} s)", t0.elapsed().as_secs_f32());
            }
            if tp {
                for l in 0..c.n_layer {
                    let o: Vec<u32> = (0..c.n_expert).map(|e| (own[l][e] as usize == dv) as u32).collect();
                    let ly = if dv == 0 { &mut layers0[l] } else { &mut layers[l] };
                    ly.ffn.own.copy_from(&o)?;
                }
            }
            arenas.push(arena);
            dyns.push(dynp);
        }
        let mut tabs_h: Vec<[Vec<u64>; 3]> = vec![];
        let mut tabs_h0: Vec<[Vec<u64>; 3]> = vec![];
        for (dv, lys) in [(1usize, &layers), (0, &layers0)] {
            for (l, ly) in lys.iter().enumerate() {
                if tp { cus[dv].bind()? } else { cus[dev_of(l)].bind()? };
                let mut t = [vec![0u64; c.n_expert], vec![0u64; c.n_expert], vec![0u64; c.n_expert]];
                ly.ffn.gtab.copy_to(&mut t[0])?;
                ly.ffn.utab.copy_to(&mut t[1])?;
                ly.ffn.dtab.copy_to(&mut t[2])?;
                if dv == 1 { tabs_h.push(t) } else { tabs_h0.push(t) }
            }
        }
        let topk = c.n_expert_used;
        let sh = Arc::new(Shared {
            flags: HostBuffer::alloc(192)?,
            ids: HostBuffer::alloc(c.n_layer * r * topk)?,
            wts: HostBuffer::alloc(r * topk)?,
            x: HostBuffer::alloc(c.n_layer * r * c.n_embd)?,
            y: HostBuffer::alloc(c.n_layer * r * topk * c.n_embd)?,
        });
        unsafe { std::ptr::write_bytes(sh.flags.as_ptr() as *mut u32, 0, 192) };
        // the per-layer fork / join events of the CPU hand-off live on the layer's device (FNEXT_TP=2: one set per card)
        let (mut ev_fork, mut ev_join, mut ev_fork0, mut ev_join0) = (vec![], vec![], vec![], vec![]);
        for l in 0..c.n_layer {
            if tp { cus[1].bind()? } else { cus[dev_of(l)].bind()? };
            ev_fork.push(Event::new_no_timing()?);
            ev_join.push(Event::new_no_timing()?);
            if tp {
                cus[0].bind()?;
                ev_fork0.push(Event::new_no_timing()?);
                ev_join0.push(Event::new_no_timing()?);
            }
        }
        cus[last].bind()?;
        let counts_a: Arc<Vec<Vec<std::sync::atomic::AtomicU64>>> =
            Arc::new((0..c.n_layer).map(|_| (0..c.n_expert).map(|_| std::sync::atomic::AtomicU64::new(0)).collect()).collect());
        // the last device's set is active between calls; device 0's is parked in `other` (swap_dev)
        let other = if ngpu == 2 {
            let (kn, fk, q, s, s2, done_ev) = kits.remove(0);
            Some(Box::new(DevSet { kn, fk, q, s, s2, w: works.remove(0), graphs: (0..2 * N_GRAPHS).map(|_| None).collect(),
                lane_graphs: Default::default(), dynp: dyns.remove(0), arena: arenas.remove(0), done_ev, pw: None, pw_need: Default::default(), lo: 0, hi: split,
                layers: std::mem::take(&mut layers0), c: ch.clone(), tabs_h: std::mem::take(&mut tabs_h0), ev_fork: std::mem::take(&mut ev_fork0),
                ev_join: std::mem::take(&mut ev_join0), tp_rank: 0, head_tp: head0.take(), mtp_head_g: (0..=MAX_ROWS).map(|_| None).collect(), cu: cus[0].clone() }))
        } else {
            None
        };
        let (kn, fk, q, s, s2, done_ev) = kits.remove(0);
        let (w, dynp, arena, cu) = (works.remove(0), dyns.remove(0), arenas.remove(0), cus[last].clone());
        let (lo, hi) = range(last);
        let mut fast = Fast {
            c: c.clone(), g: g.clone(), kn, fk, q, s, layers, head, output, w, ctx, tokens: vec![], graphs: (0..2 * N_GRAPHS).map(|_| None).collect(),
            prm_h: HostBuffer::alloc(MAX_LANES * P_WORDS)?, emb_h: HostBuffer::alloc(r * c.n_embd)?, ple_h: HostBuffer::alloc(r * c.n_embd)?,
            tok_h: HostBuffer::alloc(r)?, logit_h: HostBuffer::alloc(MAX_LANES * c.n_vocab)?, pmx_h: HostBuffer::alloc(r)?, sh: sh.clone(),
            resident: Arc::new(resident.into_iter().map(|l| l.into_iter().map(AtomicU8::new).collect()).collect()),
            last_use: Arc::new((0..c.n_layer).map(|_| (0..c.n_expert).map(|_| AtomicU64::new(0)).collect()).collect()),
            missed: Arc::new(std::sync::Mutex::new(vec![])), step_no: Arc::new(AtomicU64::new(1)), dynp, cu,
            tabs_h, _arena: arena, expert_bytes_on_card: on_card, n_resident: n_res, cpu: None,
            stop: Arc::new(AtomicBool::new(false)), cur_rows: Arc::new(AtomicU32::new(1)), counts: counts_a,
            stats: Arc::new(std::sync::Mutex::new(Stats::default())), done_ev, want_logits: true, bad_rows: 0, cancel: None, uploads: 0, mtp, pw: None, pcfg, hm: hostmem::HostMem::new(&g), ctx_mapped: 8192.min(ctx), s2,
            ev_fork, ev_join, mtp_graphs: (0..=MAX_ROWS).map(|_| None).collect(),
            spec_stats: (0, 0), draft_p: 1.0, acc_ema: 0.9, spec_ms: [0.0; 5], mtp_ms: [0.0; 3], mtp_counts: vec![], ple_idx: vec![], snaps: vec![], lanes, cur_lane: 0, plan: vec![], lane_graphs: Default::default(), ctx_cap: ctx, pcpu: None, ycpu: None, ple_flag: HostBuffer::alloc(1)?, pw_need: Default::default(),
            p_min: std::env::var("FNEXT_PMIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.7), dump: vec![],
            lo, hi, xfer: HostBuffer::alloc(r * hd)?, other,
            tp, tp_rank: tp as usize, cfull: c.clone(), own_h: Arc::new(own), ar_h: HostBuffer::alloc(2 * 2 * MAX_ROWS * c.n_embd * 2)?,
            epoch: 0, launcher: None, head_tp: None, pa_h: None, tv_h: HostBuffer::alloc(2 * MAX_ROWS * 3)?,
            mtp_head_g: (0..=MAX_ROWS).map(|_| None).collect(), mtp_h: HostBuffer::alloc(2 * MAX_ROWS * c.n_embd)?,
        };
        if tp {
            // FNEXT_TP_LAUNCH_CPU (default 34: an SMT sibling of the service cores outside the CPU pool)
            let cpu = std::env::var("FNEXT_TP_LAUNCH_CPU").ok().and_then(|v| v.parse().ok()).unwrap_or(34);
            fast.launcher = Some(tp::Launcher::start(cus[0].clone(), cpu)?);
        }
        fast.c = ch;
        fast.pw_need = probes.pop().unwrap_or_default();
        if let (Some(o), Some(p0)) = (fast.other.as_mut(), probes.pop()) {
            o.pw_need = p0;
        }
        // pinned pools sized by the host memory the process may use (MemTotal, or the cgroup limit if lower): a big box
        // prewarms every lane's snapshots, a 32 GB box only one and no prompt stages (FNEXT_STAGE_PREWARM=0/1 and
        // FNEXT_SNAP_PREWARM=n override)
        let host_gb = host_mem_bytes() >> 30;
        let stage_on = match std::env::var("FNEXT_STAGE_PREWARM").as_deref() { Ok("0") => false, Ok(_) => true, Err(_) => host_gb >= 48 };
        if fast.pcfg.0 > MAX_ROWS && stage_on {
            prefill::prewarm_stages(&fast.cfull, &fast.g, fast.pcfg.1, 2 * ngpu)?;
        }
        let snap_n: usize = std::env::var("FNEXT_SNAP_PREWARM").ok().and_then(|v| v.parse().ok()).unwrap_or(
            if host_gb >= 48 { 3 * n_lanes + 2 } else if host_gb >= 40 { 4 } else { 1 });
        let mut sizes = vec![];
        fast.each_dev(|f| {
            sizes.push(f.snap_regions().iter().map(|r| r.1).sum::<usize>());
            Ok(())
        })?;
        prewarm_snaps(&sizes, snap_n, cus[0].clone(), host_gb >= 40)?;
        if std::env::var("FNEXT_VERBOSE").is_ok() {
            eprintln!("fnext-fast: host {host_gb} GiB: snapshot pool {snap_n} x {:?} MiB, prompt stages {}",
                sizes.iter().map(|s| s >> 20).collect::<Vec<_>>(), if stage_on { "prewarmed" } else { "on demand" });
        }
        fast.start_cpu(cpu_threads)?;
        fast.tidy_page_cache();
        Ok(fast)
    }

    /// FNEXT_GPUS=2: exchange the active device set with the parked one and make its context current.
    fn swap_dev(&mut self) -> Result<()> {
        let o = self.other.as_mut().ok_or_else(|| anyhow!("swap_dev with one device"))?;
        std::mem::swap(&mut self.kn, &mut o.kn);
        std::mem::swap(&mut self.fk, &mut o.fk);
        std::mem::swap(&mut self.q, &mut o.q);
        std::mem::swap(&mut self.s, &mut o.s);
        std::mem::swap(&mut self.s2, &mut o.s2);
        std::mem::swap(&mut self.w, &mut o.w);
        std::mem::swap(&mut self.graphs, &mut o.graphs);
        std::mem::swap(&mut self.lane_graphs, &mut o.lane_graphs);
        std::mem::swap(&mut self.dynp, &mut o.dynp);
        std::mem::swap(&mut self._arena, &mut o.arena);
        std::mem::swap(&mut self.done_ev, &mut o.done_ev);
        std::mem::swap(&mut self.pw, &mut o.pw);
        std::mem::swap(&mut self.pw_need, &mut o.pw_need);
        std::mem::swap(&mut self.lo, &mut o.lo);
        std::mem::swap(&mut self.hi, &mut o.hi);
        if self.tp {
            std::mem::swap(&mut self.layers, &mut o.layers);
            std::mem::swap(&mut self.c, &mut o.c);
            std::mem::swap(&mut self.tabs_h, &mut o.tabs_h);
            std::mem::swap(&mut self.ev_fork, &mut o.ev_fork);
            std::mem::swap(&mut self.ev_join, &mut o.ev_join);
            std::mem::swap(&mut self.tp_rank, &mut o.tp_rank);
            std::mem::swap(&mut self.head_tp, &mut o.head_tp);
            std::mem::swap(&mut self.mtp_head_g, &mut o.mtp_head_g);
        }
        std::mem::swap(&mut self.cu, &mut o.cu);
        self.cu.bind()?;
        Ok(())
    }

    /// Run `f` on every device: the parked one first (if any), then the active one, which stays active.
    fn each_dev(&mut self, mut f: impl FnMut(&mut Fast) -> Result<()>) -> Result<()> {
        if self.other.is_some() {
            self.swap_dev()?;
            let r = f(self);
            self.swap_dev()?;
            r?;
        }
        f(self)
    }

    /// The active device runs the last blocks (head, MTP block, logits).
    fn is_last(&self) -> bool {
        if self.tp { self.tp_rank == 1 } else { self.hi == self.c.n_layer }
    }

    /// Number of devices (FNEXT_GPUS).
    pub fn n_gpus(&self) -> usize {
        1 + self.other.is_some() as usize
    }

    /// LRU slots per layer of each device (first device first).
    pub fn lru_slots(&self) -> Vec<usize> {
        let mut v = vec![];
        if let Some(o) = &self.other {
            v.push(o.dynp.as_ref().map_or(0, |d| d.per_layer));
        }
        v.push(self.dynp.as_ref().map_or(0, |d| d.per_layer));
        v
    }

    /// The CPU expert thread: per MoE layer, wait for the graph's flag, compute the experts that are not on the card
    /// for this step's rows, write their rows into the pinned y, raise the done flag.
    fn start_cpu(&mut self, threads: usize) -> Result<()> {
        let sh = self.sh.clone();
        let g = self.g.clone();
        let c = self.c.clone();
        let resident = self.resident.clone();
        let last_use = self.last_use.clone();
        let missed = self.missed.clone();
        let step_no = self.step_no.clone();
        let stop = self.stop.clone();
        let rows_a = self.cur_rows.clone();
        let counts = self.counts.clone();
        let stats = self.stats.clone();
        let no_miss = std::env::var("FNEXT_NO_MISS").is_ok();
        let tp = self.tp;
        let own = self.own_h.clone();
        // FNEXT_TP=2: the done flags carry the step epoch (prm word P_EPOCH, set by the host before the launch)
        let prm_addr = self.prm_h.as_ptr() as usize;
        let h = std::thread::Builder::new().name("fnext-cpu-moe".into()).spawn(move || {
            // FNEXT_CPU_POOL (default 8-23, the E-cores): one pinned worker per CPU; this thread (CPU 1) joins each call.
            let pool_list = crate::cores::plan().pool.clone();
            let _ = threads;
            fe_ggq::cpu::pool::pin_current(crate::cores::plan().cpu);
            let mut moe = CpuMoe::pinned(&pool_list, std::time::Duration::from_millis(20),
                c.n_embd, c.n_ff_exp);
            let topk = c.n_expert_used;
            let flags = sh.flags.as_ptr() as *mut u32;
            let mut l = 0usize;
            // FNEXT_ROUTE_LOG=path: every layer's picks, one line per layer call (statistics only)
            let mut rlog = std::env::var("FNEXT_ROUTE_LOG").ok().and_then(|p| std::fs::File::create(p).ok())
                .map(std::io::BufWriter::new);
            let lag_us: u64 = std::env::var("FNEXT_CPU_LAG_US").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let mut lag_rng = 0x9e3779b97f4a7c15u64;
            // FNEXT_SIM_DYN=n: simulate an LRU of n extra slots per layer on top of the static set (statistics only)
            let sim_n: usize = std::env::var("FNEXT_SIM_DYN").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let mut sim: Vec<Vec<(usize, u64)>> = vec![vec![]; c.n_layer];
            let mut sim_clock = 0u64;
            let tens: Vec<_> = (0..c.n_layer).map(|l| {
                let f = |n: &str| g.tensor(&format!("blk.{l}.{n}.weight")).unwrap().clone();
                (f("ffn_gate_exps"), f("ffn_up_exps"), f("ffn_down_exps"))
            }).collect();
            loop {
                // wait for ready[l]
                let rf = unsafe { flags.add(l) };
                let mut spins = 0u64;
                let mut rv;
                while {
                    rv = unsafe { std::ptr::read_volatile(rf) };
                    rv == 0
                } {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    spins += 1;
                    if spins > 4_000_000 {
                        std::thread::yield_now();
                    } else {
                        std::hint::spin_loop();
                    }
                }
                unsafe { std::ptr::write_volatile(rf, 0) };
                fence(Ordering::SeqCst);
                if lag_us > 0 {
                    // test hook: a CPU thread that falls behind the card (FNEXT_CPU_LAG_US)
                    lag_rng = lag_rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    std::thread::sleep(std::time::Duration::from_micros((lag_rng >> 33) % lag_us));
                }
                let t0 = std::time::Instant::now();
                // per-layer hand-off slots, and the row count travels in the ready flag: a late look at a layer the card
                // already finished (all hits) can never read another layer's or step's picks
                let rows = (rv as usize).min(MAX_ROWS);
                let _ = &rows_a;
                let ids = unsafe { std::slice::from_raw_parts(sh.ids.as_ptr().add(l * MAX_ROWS * topk), rows * topk) };
                let x = unsafe { std::slice::from_raw_parts(sh.x.as_ptr().add(l * MAX_ROWS * c.n_embd), rows * c.n_embd) };
                let mut groups: Vec<(usize, Vec<usize>, Vec<usize>)> = vec![]; // (expert, rows, pairs)
                let mut nmiss = 0;
                // FNEXT_TP=2: which cards have a miss among the experts they own (each waits on its own done flag)
                let mut miss_on = [false; 2];
                let mut miss_lc = [0u64; 2];
                if let Some(f) = rlog.as_mut() {
                    use std::io::Write;
                    let v: Vec<String> = ids.iter().map(|e| e.to_string()).collect();
                    let _ = writeln!(f, "{l} {}", v.join(" "));
                }
                for (p, &e) in ids.iter().enumerate() {
                    let e = e as usize;
                    counts[l][e].fetch_add(1, Ordering::Relaxed);
                    last_use[l][e].store(step_no.load(Ordering::Relaxed), Ordering::Relaxed);
                    if resident[l][e].load(Ordering::Relaxed) != 0 {
                        continue;
                    }
                    nmiss += 1;
                    miss_on[own[l][e] as usize] = true;
                    miss_lc[own[l][e] as usize] += 1;
                    missed.lock().unwrap().push((l, e));
                    if sim_n > 0 {
                        sim_clock += 1;
                        let lr = &mut sim[l];
                        if let Some(x) = lr.iter_mut().find(|x| x.0 == e) {
                            x.1 = sim_clock;
                            stats.lock().unwrap().sim_hits += 1;
                        } else if lr.len() < sim_n {
                            lr.push((e, sim_clock));
                        } else {
                            let i = (0..lr.len()).min_by_key(|&i| lr[i].1).unwrap();
                            lr[i] = (e, sim_clock);
                        }
                    }
                    match groups.iter_mut().find(|gq| gq.0 == e) {
                        Some(gq) => {
                            gq.1.push(p / topk);
                            gq.2.push(p);
                        }
                        None => groups.push((e, vec![p / topk], vec![p])),
                    }
                }
                if !groups.is_empty() && !no_miss {
                    let (tg, tu, td) = &tens[l];
                    let jobs: Vec<MoeJob> = groups.iter().map(|(e, rws, _)| MoeJob {
                        gate: g.slice(tg, *e), up: g.slice(tu, *e), down: g.slice(td, *e), ty_gu: tg.ty, ty_down: td.ty,
                        rows: rws.clone(),
                    }).collect();
                    moe.run(&jobs, x).expect("cpu moe");
                    let yb = unsafe { (sh.y.as_ptr() as *mut f32).add(l * MAX_ROWS * topk * c.n_embd) };
                    for (j, (_, _, pairs)) in groups.iter().enumerate() {
                        let out = moe.out(j);
                        for (i, &p) in pairs.iter().enumerate() {
                            unsafe {
                                std::ptr::copy_nonoverlapping(out.as_ptr().add(i * c.n_embd), yb.add(p * c.n_embd), c.n_embd)
                            };
                        }
                    }
                }
                fence(Ordering::SeqCst);
                if tp {
                    let ep = unsafe { std::ptr::read_volatile((prm_addr as *const u32).add(P_EPOCH)) };
                    for r in 0..2 {
                        if miss_on[r] {
                            unsafe { std::ptr::write_volatile(flags.add(64 + 64 * r + l), ep) };
                        }
                    }
                } else if nmiss > 0 {
                    unsafe { std::ptr::write_volatile(flags.add(64 + l), 1) };
                }
                {
                    let mut st = stats.lock().unwrap();
                    st.pairs += ids.len() as u64;
                    st.misses += nmiss;
                    st.cpu_us += t0.elapsed().as_micros() as u64;
                    if st.miss_lc.len() < c.n_layer * 2 {
                        st.miss_lc.resize(c.n_layer * 2, 0);
                    }
                    st.miss_lc[l * 2] += miss_lc[0];
                    st.miss_lc[l * 2 + 1] += miss_lc[1];
                }
                l = (l + 1) % c.n_layer;
            }
        })?;
        self.cpu = Some(h);
        Ok(())
    }

    fn rope_rows(&self, pos: &[usize], cs: &mut [f32], sn: &mut [f32]) {
        let half = self.c.n_rot / 2;
        let ts = self.c.rope_base.powf(-2.0 / self.c.n_rot as f32);
        for (r, &p) in pos.iter().enumerate() {
            let mut th = p as f32;
            for k in 0..half {
                cs[r * 32 + k] = th.cos();
                sn[r * 32 + k] = th.sin();
                th *= ts;
            }
        }
    }

    /// Fill the pinned step parameters, embeddings and PLE rows for `toks` at the next positions.
    /// Host side of a step before its graph: positions, embeddings, and the PLE rows requested from the page cache
    /// (their gather is ple_gather, which overlaps layer 0 when the graph waits for it).
    fn prepare(&mut self, toks: &[u32]) -> Result<()> {
        self.prepare_at(toks, 0, 0)
    }

    /// prepare for rows row0.. of a step and step-parameter block `blk`.
    fn prepare_at(&mut self, toks: &[u32], row0: usize, blk: usize) -> Result<()> {
        let c = self.c.clone();
        let rows = toks.len();
        let pos0 = self.tokens.len();
        if pos0 + rows > self.ctx {
            bail!("context full ({} cells)", self.ctx);
        }
        self.ensure_ctx(pos0 + rows + MAX_ROWS)?;
        self.fill_prm_at(pos0, rows, blk);
        self.fill_emb_at(toks, row0);
        self.ple_idx = crate::engine::ple_rows(&c, &self.tokens, pos0, toks);
        // request the pages of every row first so the random reads go out together (page faults one after another
        // cost a disk latency each)
        let pt = self.g.tensor("per_layer_token_embd.weight").unwrap();
        let (prb, pdata) = (pt.row_bytes(), self.g.data(pt));
        let page = self.hm.page();
        for &rw in &self.ple_idx {
            hostmem::HostMem::willneed_range(pdata, rw as usize * prb, prb, page);
        }
        Ok(())
    }

    /// WILLNEED on the PLE rows of `toks` at the next positions (pages read ahead of the step that needs them).
    fn ple_willneed(&self, toks: &[u32]) {
        let idx = crate::engine::ple_rows(&self.c, &self.tokens, self.tokens.len(), toks);
        let pt = self.g.tensor("per_layer_token_embd.weight").unwrap();
        let (prb, pdata) = (pt.row_bytes(), self.g.data(pt));
        let page = self.hm.page();
        for &rw in &idx {
            hostmem::HostMem::willneed_range(pdata, rw as usize * prb, prb, page);
        }
    }

    /// The PLE rows of the step into the pinned host buffer the graph copies from.
    fn ple_gather(&mut self) {
        self.ple_gather_at(0);
    }

    fn ple_gather_at(&mut self, row0: usize) {
        let (d, heads) = (self.c.n_embd, self.c.ple_heads());
        ple_gather_rows(&self.g, &self.hm, &self.ple_idx, self.ple_h.as_mut_slice(), d, heads, row0);
    }

    fn fill_emb(&mut self, toks: &[u32]) {
        self.fill_emb_at(toks, 0);
    }

    fn fill_emb_at(&mut self, toks: &[u32], row0: usize) {
        let d = self.c.n_embd;
        let te = self.g.tensor("token_embd.weight").unwrap();
        let rb = te.row_bytes();
        let data = self.g.data(te);
        let emb = self.emb_h.as_mut_slice();
        for (r, &t) in toks.iter().enumerate() {
            fe_ggq::refq::dequant(te.ty, &data[t as usize * rb..(t as usize + 1) * rb], &mut emb[(row0 + r) * d..(row0 + r + 1) * d]);
        }
    }

    /// Step parameters for `rows` rows at positions pos0..: rope tables, QSA blocks completed by them.
    fn fill_prm(&mut self, pos0: usize, rows: usize) {
        self.fill_prm_at(pos0, rows, 0);
    }

    /// Step parameters into block `blk` of the host buffer (one block per lane of a batched step).
    fn fill_prm_at(&mut self, pos0: usize, rows: usize, blk: usize) {
        let c = self.c.clone();
        let prm = &mut self.prm_h.as_mut_slice()[blk * P_WORDS..(blk + 1) * P_WORDS];
        prm[P_POS] = pos0 as u32;
        prm[P_ROWS] = rows as u32;
        let kp = c.kpool;
        let b_lo = pos0 / kp;
        let b_hi = (pos0 + rows) / kp;
        prm[P_NB] = (b_hi - b_lo) as u32;
        prm[P_BLO] = b_lo as u32;
        let mut cs = vec![0f32; MAX_ROWS * 32];
        let mut sn = vec![0f32; MAX_ROWS * 32];
        self.rope_rows(&(0..rows).map(|r| pos0 + r).collect::<Vec<_>>(), &mut cs, &mut sn);
        let mut pcs = vec![0f32; 8 * 32];
        let mut psn = vec![0f32; 8 * 32];
        self.rope_rows(&(b_lo..b_hi).map(|b| b * kp).collect::<Vec<_>>(), &mut pcs, &mut psn);
        let prm = &mut self.prm_h.as_mut_slice()[blk * P_WORDS..(blk + 1) * P_WORDS];
        for i in 0..MAX_ROWS * 32 {
            prm[P_CS + i] = cs[i].to_bits();
            prm[P_SN + i] = sn[i].to_bits();
        }
        for i in 0..8 * 32 {
            prm[P_PCS + i] = pcs[i].to_bits();
            prm[P_PSN + i] = psn[i].to_bits();
        }
    }

    /// Enqueue one step of `rows` rows (everything reads positions from w.prm).
    unsafe fn enqueue(&mut self, rows: usize, verify: bool) -> Result<()> {
        let ro = verify as u32;
        let c = self.c.clone();
        let (kn, fk, q, s) = (&self.kn, &self.fk, &self.q, &self.s);
        let d = c.n_embd;
        let hd = c.hc_dim();
        let topk = c.n_expert_used;
        let (lo, hi) = (self.lo, self.hi);
        // FNEXT_TP=2: the block outputs are all-reduced with the other card through the LL slots of ar_h (alternating)
        let (tp, rank) = (self.tp, self.tp_rank);
        let mut kar = 0u32;
        let slot = MAX_ROWS * c.n_embd * 2;
        let ar_base = self.ar_h.as_ptr() as *mut u32;
        let ar_mine = |k: u32| -> &mut [u32] {
            std::slice::from_raw_parts_mut(ar_base.add((rank * 2 + k as usize % 2) * slot), rows * c.n_embd * 2)
        };
        let ar_other = |k: u32| -> DevPtr { DevPtr(ar_base.add(((1 - rank) * 2 + k as usize % 2) * slot) as u64) };
        // segments: one per lane of the step (rows contiguous per lane, one step-parameter block each)
        let plan: Vec<(usize, usize, usize)> = if self.plan.is_empty() { vec![(self.cur_lane, 0, rows)] } else { self.plan.clone() };
        let ptrs: Vec<Vec<lanes::SeqPtrs>> = plan.iter().map(|&(ln, _, _)| self.seq_ptrs(ln)).collect();
        let segp = |i: usize| -> DevPtr { self.w.prm.ptr().add::<u32>(i * P_WORDS) };
        let segs: Vec<(usize, usize, DevPtr)> = plan.iter().enumerate().map(|(i, &(_, r0, n))| (r0, n, segp(i))).collect();
        let w = &mut self.w;
        s.memcpy_htod(w.prm.ptr(), &self.prm_h.as_slice()[..plan.len() * P_WORDS])?;
        let late = ple_late_tp(tp);
        let n = rows * d;
        if lo == 0 {
            s.memcpy_htod(w.emb.ptr(), &self.emb_h.as_slice()[..rows * d])?;
            if !late {
                s.memcpy_htod(w.pemb.ptr(), &self.ple_h.as_slice()[..rows * d])?;
            }
            s.launch(&kn.hc_init, g256(n), 256, 0, (w.res.ptr(), w.emb.ptr(), d as u32, c.hc as u32, n as u32))?;
        } else {
            // layer split: the residual streams after block lo - 1, from the other device's half of the step
            s.memcpy_htod(w.res.ptr(), &self.xfer.as_slice()[..rows * hd])?;
        }
        let prm = w.prm.ptr();
        let flags = self.sh.flags.dev();
        let no_exp = std::env::var("FNEXT_NO_EXPERTS").is_ok();
        let dbg = std::env::var("FNEXT_DUMP").is_ok() && std::env::var("FNEXT_NO_GRAPH").is_ok();
        for (l, ly) in self.layers.iter().enumerate().take(hi).skip(lo) {
            if let Some(p) = &ly.ple {
                if late {
                    // the host gathers the rows while layer 0 runs
                    s.wait_value32(self.ple_flag.dev(), 1, true)?;
                    s.memcpy_htod(w.pemb.ptr(), &self.ple_h.as_slice()[..rows * d])?;
                }
                hrun(fk, s, &p.key, w.pemb.ptr(), d, w.pkey.ptr(), hd, rows, 1.0, false)?;
                hrun(fk, s, &p.value, w.pemb.ptr(), d, w.pval.ptr(), d, rows, 1.0, false)?;
                kn.rmsnorm_dec(s, w.pkn.ptr(), w.pkey.ptr(), p.nk.ptr(), d, c.hc, rows * c.hc, c.eps)?;
                kn.rmsnorm_dec(s, w.pqn.ptr(), w.res.ptr(), p.nq.ptr(), d, c.hc, rows * c.hc, c.eps)?;
                s.launch(&kn.ple_gate, (rows * c.hc) as u32, 256, 0,
                    (w.pgated.ptr(), w.pkn.ptr(), w.pqn.ptr(), w.pval.ptr(), d as u32, c.hc as u32))?;
                kn.rmsnorm_dec(s, w.pcn.ptr(), w.pgated.ptr(), p.nc.ptr(), d, c.hc, rows * c.hc, c.eps)?;
                if dbg {
                    dump_rows(s, &mut self.dump, l, 20, w.pemb.ptr(), rows, d)?;
                    dump_rows(s, &mut self.dump, l, 21, w.pkey.ptr(), rows, hd)?;
                    dump_rows(s, &mut self.dump, l, 22, w.pval.ptr(), rows, d)?;
                    dump_rows(s, &mut self.dump, l, 23, w.pgated.ptr(), rows, hd)?;
                    dump_rows(s, &mut self.dump, l, 24, w.pcn.ptr(), rows, hd)?;
                    dump_rows(s, &mut self.dump, l, 25, w.res.ptr(), rows, hd)?;
                }
                for (si, &(r0, n, _)) in segs.iter().enumerate() {
                    s.launch(&kn.ple_conv, g256(hd), 256, 0, (w.res.ptr().add::<f32>(r0 * hd), w.pgated.ptr().add::<f32>(r0 * hd),
                        w.pcn.ptr().add::<f32>(r0 * hd), p.conv_w.ptr(), ptrs[si][l].hist, n as u32, hd as u32,
                        c.ple_ngram as u32, DevPtr::NULL, ro))?;
                }
            }
            if dbg {
                dump_rows(s, &mut self.dump, l, 10, w.res.ptr(), rows, hd)?;
            }
            hc_mix(kn, fk, s, w, &ly.hca, &c, rows)?;
            if dbg {
                dump_rows(s, &mut self.dump, l, 11, w.mixed.ptr(), rows, d)?;
            }
            match &ly.mix {
                Mixer::Gdn(gd) => {
                    let cd = c.conv_dim();
                    quant_mixed(q, s, w.mixed.ptr(), rows, d, &mut w.xa)?;
                    gd.qkv.run(q, s, &w.xa, gd.qkv_s.ptr(), false)?;
                    gd.z.run(q, s, &w.xa, w.z.ptr(), false)?;
                    hrun(fk, s, &gd.ab, w.mixed.ptr(), d, gd.ab_s.ptr(), 2 * c.ssm_vh, rows, 1.0, false)?;
                    if dbg {
                        dump_rows(s, &mut self.dump, l, 12, gd.qkv_s.ptr(), rows, cd)?;
                        dump_rows(s, &mut self.dump, l, 13, w.z.ptr(), rows, c.ssm_inner)?;
                        dump_rows(s, &mut self.dump, l, 14, gd.ab_s.ptr(), rows, 2 * c.ssm_vh)?;
                    }
                    for (si, &(r0, n, _)) in segs.iter().enumerate() {
                        let sp = &ptrs[si][l];
                        s.launch(&kn.gdn_conv, g256(cd), 256, 0, (gd.conv_s.ptr().add::<f32>(r0 * cd),
                            gd.qkv_s.ptr().add::<f32>(r0 * cd), cd as u32, gd.conv_w.ptr(), sp.conv, n as u32, cd as u32, ro))?;
                        s.launch(&kn.gdn_rec, (c.ssm_vh as u32, 4), 128, 0, (w.o.ptr().add::<f32>(r0 * c.ssm_inner),
                            gd.conv_s.ptr().add::<f32>(r0 * cd), gd.ab_s.ptr().add::<f32>(r0 * 2 * c.ssm_vh), (2 * c.ssm_vh) as u32,
                            gd.dt.ptr(), gd.a.ptr(), sp.ssm, n as u32, c.ssm_kh as u32, c.ssm_vh as u32, c.eps, ro))?;
                    }
                    if dbg {
                        dump_rows(s, &mut self.dump, l, 15, gd.conv_s.ptr(), rows, cd)?;
                        dump_rows(s, &mut self.dump, l, 16, w.o.ptr(), rows, c.ssm_inner)?;
                    }
                    s.launch(&kn.gdn_gnorm, (rows as u32, c.ssm_vh.div_ceil(8) as u32), 256, 0, (w.y.ptr(), w.o.ptr(),
                        w.z.ptr(), c.ssm_inner as u32, gd.norm.ptr(), c.ssm_vh as u32, c.eps))?;
                    quant_mixed(q, s, w.y.ptr(), rows, c.ssm_inner, &mut w.xa)?;
                    gd.out.run(q, s, &w.xa, w.out.ptr(), false)?;
                }
                Mixer::Attn(at) => {
                    let sg: Vec<(usize, usize, DevPtr, lanes::KvPtr)> =
                        segs.iter().enumerate().map(|(si, &(r0, n, pp))| (r0, n, pp, ptrs[si][l].kv)).collect();
                    attn_block_segs(kn, fk, q, s, w, at, &c, rows, self.ctx, &sg)?;
                }
            }
            if dbg {
                dump_rows(s, &mut self.dump, l, 0, w.out.ptr(), rows, d)?;
            }
            if tp {
                // this card's partial: LL pairs to the host slot, then the sum with the other card's + the combine
                s.launch(&fk.ll_pack, g256(rows * d), 256, 0, (w.llp.ptr(), w.out.ptr(), (rows * d) as u32,
                    prm.add::<u32>(P_EPOCH), kar))?;
                s.memcpy_dtoh(ar_mine(kar), w.llp.ptr())?;
                s.launch(&fk.ar_comb, (g256(d), rows as u32), 256, 0, (w.res.ptr(), w.out.ptr(), ar_other(kar),
                    w.lo.ptr().add::<f32>(c.hc_lr), (c.hc_lr + c.hc) as u32, d as u32, c.hc as u32, prm, kar))?;
                kar += 1;
            } else {
                hc_combine(fk, s, w, &c, rows)?;
            }
            if dbg {
                dump_rows(s, &mut self.dump, l, 1, w.res.ptr(), rows, hd)?;
            }
            hc_mix(kn, fk, s, w, &ly.hcf, &c, rows)?;
            // ---- MoE ----
            let f = &ly.ffn;
            let ldr = c.n_expert + 1;
            hrun(fk, s, &f.gate_inp, w.mixed.ptr(), d, w.rl.ptr(), ldr, rows, 1.0, false)?;
            s.launch(&fk.route8, rows as u32, 256, 0, (w.ids.ptr(), w.wts.ptr(), w.rl.ptr(), ldr as u32, c.n_expert as u32,
                topk as u32, c.expert_weights_scale))?;
            if !no_exp && rank == 0 {
                // hand the picks and the FFN input to the CPU thread (FNEXT_TP=2: card 0 for both cards)
                // on a side stream, so the copies' latency overlaps the shared expert and the resident experts
                let s2 = &self.s2;
                s.record(&self.ev_fork[l])?;
                s2.wait(&self.ev_fork[l])?;
                s2.memcpy_dtoh(std::slice::from_raw_parts_mut((self.sh.ids.as_ptr() as *mut i32).add(l * MAX_ROWS * topk),
                    rows * topk), w.ids.ptr())?;
                s2.memcpy_dtoh(std::slice::from_raw_parts_mut((self.sh.x.as_ptr() as *mut f32).add(l * MAX_ROWS * d), rows * d),
                    w.mixed.ptr())?;
                s2.write_value32(flags.add::<u32>(l), rows as u32)?;
                s2.record(&self.ev_join[l])?;
            }
            quant_mixed(q, s, w.mixed.ptr(), rows, d, &mut w.ea)?;
            f.shg.run(q, s, &w.ea, w.sg.ptr(), false)?;
            f.shu.run(q, s, &w.ea, w.su.ptr(), false)?;
            let nf = rows * c.n_ff_shexp;
            s.launch(&kn.swiglu, g256(nf), 256, 0, (w.sg.ptr(), w.sg.ptr(), w.su.ptr(), nf as u32))?;
            quant_mixed(q, s, w.sg.ptr(), rows, c.n_ff_shexp, &mut w.sa)?;
            f.shd.run(q, s, &w.sa, w.sh.ptr(), false)?;
            if no_exp {
                s.memset_d8(w.moe.ptr(), 0, rows * d * 4)?;
            } else {
                let rb_gu = f.ty_gu.row_bytes(d);
                let rb_d = f.ty_d.row_bytes(c.n_ff_exp);
                q.moe_gu_q8(s, f.ty_gu, f.gtab.ptr(), f.utab.ptr(), rb_gu, w.ids.ptr(), rows, topk, &w.ea, c.n_ff_exp,
                    &mut w.ha, None, None)?;
                q.moe_down(s, f.ty_d, f.dtab.ptr(), rb_d, w.ids.ptr(), rows, topk, &w.ha, d, w.ey.ptr(), None)?;
                if rank == 0 {
                    s.wait(&self.ev_join[l])?;
                }
                let hy = self.sh.y.dev().add::<f32>(l * MAX_ROWS * topk * d);
                if tp {
                    // the experts this card owns (the CPU rows of its misses), its half of the shared expert, the sum
                    // with the other card's partial, the combine
                    let fl = flags.add::<u32>(64 + 64 * rank + l);
                    s.launch(&fk.moe_fin_tp, (g256(d), rows as u32), 256, 0, (w.out.ptr(), w.llp.ptr(), w.ey.ptr(), hy,
                        w.ids.ptr(), f.dtab.ptr(), f.own.ptr(), w.wts.ptr(), w.sh.ptr(), w.rl.ptr().add::<f32>(c.n_expert),
                        ldr as u32, d as u32, topk as u32, fl, prm, kar))?;
                    s.memcpy_dtoh(ar_mine(kar), w.llp.ptr())?;
                    s.launch(&fk.ar_comb, (g256(d), rows as u32), 256, 0, (w.res.ptr(), w.out.ptr(), ar_other(kar),
                        w.lo.ptr().add::<f32>(c.hc_lr), (c.hc_lr + c.hc) as u32, d as u32, c.hc as u32, prm, kar))?;
                    kar += 1;
                } else {
                // a layer without misses raises its own done flag; otherwise the CPU thread does
                s.launch(&fk.hit_flag, 1, 32, 0, (flags.add::<u32>(64 + l), w.ids.ptr(), f.dtab.ptr(), (rows * topk) as u32))?;
                s.wait_value32(flags.add::<u32>(64 + l), 1, true)?;
                s.launch(&fk.moe_fin, (g256(d), rows as u32), 256, 0, (w.res.ptr(), w.ey.ptr(), hy, w.ids.ptr(),
                    f.dtab.ptr(), w.wts.ptr(), w.sh.ptr(), w.rl.ptr().add::<f32>(c.n_expert), ldr as u32,
                    w.lo.ptr().add::<f32>(c.hc_lr), (c.hc_lr + c.hc) as u32, d as u32, c.hc as u32, topk as u32,
                    flags.add::<u32>(64 + l)))?;
                }
            }
            if no_exp {
                s.launch(&fk.moe_out, g256(n), 256, 0, (w.out.ptr(), w.moe.ptr(), w.sh.ptr(), w.rl.ptr().add::<f32>(c.n_expert),
                    ldr as u32, d as u32, n as u32))?;
                hc_combine(fk, s, w, &c, rows)?;
            }
            if dbg {
                dump_rows(s, &mut self.dump, l, 2, w.res.ptr(), rows, hd)?;
            }
        }
        if tp {
            // FNEXT_TP=2: each card its half of the vocabulary (card 1 also keeps the trunk residual for the MTP block);
            // the host takes the larger logit (card 0 on ties: the lower index)
            let half = c.n_vocab / 2;
            let (hcm, ow, row0, nrows) = match &self.head_tp {
                Some((h, o)) => (h, o, 0, half),
                None => (&self.head, &self.output, half, c.n_vocab - half),
            };
            hc_mix(kn, fk, s, w, hcm, &c, rows)?;
            if rank == 1 {
                // the trunk residual rows for the MTP block (a multi-lane step: all lanes' rows, in hbat)
                let hs = if plan.len() > 1 { w.hbat.ptr() } else { w.hsave.ptr() };
                s.memcpy_dtod(hs, w.res.ptr(), rows * hd * 4)?;
            }
            let (r0, nl) = if verify || plan.len() > 1 { (0, rows) } else { (rows - 1, 1) };
            q.quantize_q8_1(s, w.mixed.ptr().add::<f32>(r0 * d), d, false, nl, d, &mut w.xa)?;
            ow.run(q, s, &w.xa, w.logits.ptr(), false)?;
            argmax_rows(fk, s, w, w.logits.ptr(), w.tok.ptr(), nl, nrows, DevPtr::NULL)?;
            s.launch(&fk.argmax_tv, 1, 32, 0, (w.tv.ptr(), w.tok.ptr(), w.logits.ptr(), nrows as u32, row0 as u32, nl as u32,
                DevPtr::NULL))?;
            let tvh = std::slice::from_raw_parts_mut((self.tv_h.as_ptr() as *mut u32).add(rank * MAX_ROWS * 3), nl * 3);
            s.memcpy_dtoh(tvh, w.tv.ptr())?;
            if self.want_logits && nl <= MAX_LANES {
                if plan.len() > 1 {
                    // every lane's row: its half into the lane's logits row
                    for i in 0..nl {
                        let lh = std::slice::from_raw_parts_mut((self.logit_h.as_ptr() as *mut f32).add(i * c.n_vocab + row0), nrows);
                        s.memcpy_dtoh(lh, w.logits.ptr().add::<f32>(i * nrows))?;
                    }
                } else {
                    let lh = std::slice::from_raw_parts_mut((self.logit_h.as_ptr() as *mut f32).add(row0), nrows);
                    s.memcpy_dtoh(lh, w.logits.ptr().add::<f32>((nl - 1) * nrows))?;
                }
            }
            return Ok(());
        }
        if hi < c.n_layer {
            // layer split: hand the residual streams to the device of the next blocks
            s.memcpy_dtoh(&mut self.xfer.as_mut_slice()[..rows * hd], w.res.ptr())?;
            return Ok(());
        }
        hc_mix(kn, fk, s, w, &self.head, &c, rows)?;
        let hs = if plan.len() > 1 { w.hbat.ptr() } else { w.hsave.ptr() };
        s.memcpy_dtod(hs, w.res.ptr(), rows * hd * 4)?;
        // logits and argmax: the last row (or every row of a verify)
        let (r0, nl) = if verify || plan.len() > 1 { (0, rows) } else { (rows - 1, 1) };
        q.quantize_q8_1(s, w.mixed.ptr().add::<f32>(r0 * d), d, false, nl, d, &mut w.xa)?;
        self.output.run(q, s, &w.xa, w.logits.ptr(), false)?;
        argmax_rows(fk, s, w, w.logits.ptr(), w.tok.ptr(), nl, c.n_vocab, DevPtr::NULL)?;
        s.memcpy_dtoh(&mut self.tok_h.as_mut_slice()[..nl], w.tok.ptr())?;
        if self.want_logits && nl <= MAX_LANES {
            if plan.len() > 1 {
                // every lane's row (sampled lanes pick from theirs)
                s.memcpy_dtoh(&mut self.logit_h.as_mut_slice()[..nl * c.n_vocab], w.logits.ptr())?;
            } else {
                s.memcpy_dtoh(&mut self.logit_h.as_mut_slice()[..c.n_vocab], w.logits.ptr().add::<f32>((nl - 1) * c.n_vocab))?;
            }
        }
        Ok(())
    }

    /// Run `toks` (1..=8 rows) at the next positions; returns the greedy next token after the last row.
    pub fn step(&mut self, toks: &[u32]) -> Result<u32> {
        self.run_step(toks, false)?;
        Ok(self.tok_h.as_slice()[0])
    }

    /// Speculative verify: run `toks` (row 0 = the last sampled token, then the drafts) WITHOUT advancing the
    /// recurrent state; returns the greedy token after every row. Follow with [`Fast::commit`].
    pub fn verify(&mut self, toks: &[u32]) -> Result<Vec<u32>> {
        self.run_step(toks, true)?;
        // positions are claimed by commit
        self.tokens.truncate(self.tokens.len() - toks.len());
        Ok(self.tok_h.as_slice()[..toks.len()].to_vec())
    }

    /// Keep the first `n` rows of the last verify: advance the GDN conv / state and the PLE history over them.
    pub fn commit(&mut self, toks: &[u32], n: usize) -> Result<()> {
        let c = self.c.clone();
        let idx = 2 * MAX_ROWS + n;
        let two = self.other.is_some();
        self.each_dev(|f| {
            if f.graphs[idx].is_none() {
                let g = f.capture(|m| unsafe { m.enqueue_commit(n) })?;
                f.graphs[idx] = Some(g);
            }
            f.graphs[idx].as_ref().unwrap().launch(&f.s)?;
            if two && !f.is_last() {
                f.s.sync()?;
            }
            Ok(())
        })?;
        if let Some(m) = &self.mtp {
            let hd = c.hc_dim();
            self.s.memcpy_dtod(m.hprev.ptr(), self.w.hsave.ptr().add::<f32>((n - 1) * hd), hd * 4)?;
        }
        self.s.sync()?;
        self.tokens.extend_from_slice(&toks[..n]);
        let _ = c;
        Ok(())
    }

    unsafe fn enqueue_commit(&mut self, n: usize) -> Result<()> {
        let c = self.c.clone();
        let (kn, s, w) = (&self.kn, &self.s, &self.w);
        let cd = c.conv_dim();
        for ly in &self.layers[self.lo..self.hi] {
            if let Some(p) = &ly.ple {
                let hd = c.hc_dim();
                s.launch(&kn.conv_shift, g256(hd), 256, 0, (p.hist.ptr(), w.pcn.ptr(), hd as u32, n as u32, hd as u32,
                    ((c.ple_conv - 1) * c.ple_ngram) as u32))?;
            }
            if let Mixer::Gdn(gd) = &ly.mix {
                s.launch(&kn.conv_shift, g256(cd), 256, 0, (gd.conv_state.ptr(), gd.qkv_s.ptr(), cd as u32, n as u32,
                    cd as u32, (c.ssm_conv - 1) as u32))?;
                s.launch(&kn.gdn_rec, (c.ssm_vh as u32, 4), 128, 0, (DevPtr::NULL, gd.conv_s.ptr(), gd.ab_s.ptr(),
                    (2 * c.ssm_vh) as u32, gd.dt.ptr(), gd.a.ptr(), gd.ssm.ptr(), n as u32, c.ssm_kh as u32, c.ssm_vh as u32,
                    c.eps, 0u32))?;
            }
        }
        Ok(())
    }

    fn run_step(&mut self, toks: &[u32], verify: bool) -> Result<()> {
        let rows = toks.len();
        if rows == 0 || rows > MAX_ROWS {
            bail!("step: {rows} rows (1..={MAX_ROWS})");
        }
        let t0 = std::time::Instant::now();
        self.prepare(toks)?;
        // FNEXT_TP=2 gathers the PLE rows before the launches (measured faster than beside the two graph launches;
        // FNEXT_PLE_LATE=1 there = on a helper thread during them)
        let late = self.ple_late_now();
        let pf = self.ple_flag.as_ptr() as *mut u32;
        if late {
            unsafe { std::ptr::write_volatile(pf, 0) };
            fence(Ordering::SeqCst);
        } else {
            self.ple_gather();
        }
        let t1 = std::time::Instant::now();
        self.cur_rows.store(rows as u32, Ordering::Relaxed);
        let gi = self.step_graph_index(rows, verify);
        if self.tp {
            // a fresh epoch: the all-reduce tags of this step differ from every earlier one
            self.epoch = self.epoch.wrapping_add(1);
            self.prm_h.as_mut_slice()[P_EPOCH] = self.epoch << 8;
        }
        if self.other.is_some() {
            // layer split: device 0's half of the step first, device 1's graph waits for it (cross-device event).
            // Both graphs are captured before anything launches: instantiation can wait for the device, and a launched
            // device-0 half waits for the PLE rows the host only writes after the launches.
            if std::env::var("FNEXT_NO_GRAPH").is_err() {
                self.each_dev(|m| {
                    if m.graphs[gi].is_none() {
                        let g = m.capture(|m| unsafe { m.enqueue(rows, verify) })?;
                        m.graphs[gi] = Some(g);
                    }
                    Ok(())
                })?;
            }
            let o = self.other.as_ref().unwrap();
            if let (Some(la), Some(gph)) = (&self.launcher, &o.graphs[gi]) {
                // FNEXT_TP=2: card 0's launch on the launcher thread, card 1's below at the same time
                unsafe { la.post(gph, &o.s, &o.done_ev) };
            } else {
                self.swap_dev()?;
                let r = match &self.graphs[gi] {
                    Some(gph) => gph.launch(&self.s).map_err(anyhow::Error::from),
                    None => unsafe { self.enqueue(rows, verify) },
                };
                let r = r.and_then(|_| self.s.record(&self.done_ev).map_err(anyhow::Error::from));
                self.swap_dev()?;
                r?;
            }
            if !self.tp {
                self.s.wait(&self.other.as_ref().unwrap().done_ev)?;
            }
            if late && !self.tp {
                // device 0 runs layer 0 now: gather the PLE rows before device 1's graph launch (its host cost would
                // otherwise delay the PLE layer on device 0; device 1's half never waits for the flag)
                self.ple_gather();
                fence(Ordering::SeqCst);
                unsafe { std::ptr::write_volatile(pf, 1) };
            }
        }
        if self.graphs[gi].is_none() && std::env::var("FNEXT_NO_GRAPH").is_err() {
            // capture runs nothing: keep the flag up so a direct (uncaptured) run cannot stall
            let g = self.capture(|m| unsafe { m.enqueue(rows, verify) })?;
            self.graphs[gi] = Some(g);
        }
        if self.tp && late && self.graphs[gi].is_some() {
            // FNEXT_TP=2: the PLE rows on a helper thread while this thread launches card 1's graph (card 0's is
            // launching on the launcher thread): both cards wait for them at the PLE layer
            let (d, heads) = (self.c.n_embd, self.c.ple_heads());
            let pfu = pf as usize;
            let (g, hm, idx, ple) = (&self.g, &self.hm, &self.ple_idx, self.ple_h.as_mut_slice());
            let (gph, st) = (self.graphs[gi].as_ref().unwrap(), &self.s);
            std::thread::scope(|sc| -> Result<()> {
                let j = sc.spawn(move || {
                    ple_gather_rows(g, hm, idx, ple, d, heads, 0);
                    fence(Ordering::SeqCst);
                    unsafe { std::ptr::write_volatile(pfu as *mut u32, 1) };
                });
                let r = gph.launch(st);
                j.join().map_err(|_| anyhow!("PLE gather thread panicked"))?;
                r.map_err(anyhow::Error::from)
            })?;
        } else {
            match &self.graphs[gi] {
                Some(gph) => gph.launch(&self.s)?,
                None => unsafe { self.enqueue(rows, verify)? },
            }
        }
        if late && (self.other.is_none() || self.tp) && !(self.tp && self.graphs[gi].is_some()) {
            // the PLE rows while the card(s) run layer 0
            self.ple_gather();
            fence(Ordering::SeqCst);
            unsafe { std::ptr::write_volatile(pf, 1) };
        }
        self.s.record(&self.done_ev)?;
        if let Some(la) = &self.launcher {
            la.wait()?;
        }
        self.done_ev.sync()?;
        if self.tp {
            self.other.as_ref().unwrap().done_ev.sync()?;
            // the greedy token of each row: the larger of the two halves' best logits (card 0 on ties)
            let nl = if verify { rows } else { 1 };
            self.bad_rows = 0;
            for r in 0..nl {
                let (t, _) = self.tp_pick(r);
                self.tok_h.as_mut_slice()[r] = t;
                if !self.tp_row_finite(r) {
                    self.bad_rows |= 1 << r;
                }
            }
        }
        self.tokens.extend_from_slice(toks);
        let t2 = std::time::Instant::now();
        if !verify {
            if let Some(m) = &self.mtp {
                // the MTP block reads the trunk residual of the last position
                let hd = self.c.hc_dim();
                self.s.memcpy_dtod(m.hprev.ptr(), self.w.hsave.ptr().add::<f32>((rows - 1) * hd), hd * 4)?;
            }
        }
        self.step_no.fetch_add(1, Ordering::Relaxed);
        self.refill()?;
        {
            let mut st = self.stats.lock().unwrap();
            st.prep_us += (t1 - t0).as_micros() as u64;
            st.graph_us += (t2 - t1).as_micros() as u64;
            st.refill_us += t2.elapsed().as_micros() as u64;
        }
        let mut st = self.stats.lock().unwrap();
        st.steps += 1;
        st.rows += rows as u64;
        Ok(())
    }

    /// Does a step graph over `rows` rows copy logits to the host? (`multi`: a batched step over several lanes.) One
    /// logit row per lane, or the last row of a single-lane step; passes with more logit rows than MAX_LANES never copy.
    /// Must stay the condition `enqueue` uses for its copies (`want_logits && nl <= MAX_LANES`).
    pub(crate) fn logits_in_graph(&self, rows: usize, verify: bool, multi: bool) -> bool {
        let nl = if verify || multi { rows } else { 1 };
        self.want_logits && nl <= MAX_LANES
    }

    /// Slot of the single-lane step graph for `rows` rows in `graphs`.
    fn step_graph_index(&self, rows: usize, verify: bool) -> usize {
        rows + if verify { MAX_ROWS } else { 0 } + if self.logits_in_graph(rows, verify, false) { G_LOGITS } else { 0 }
    }

    /// Logits of the last row of the last step (when `want_logits`).
    pub fn logits(&self) -> &[f32] {
        &self.logit_h.as_slice()[..self.c.n_vocab]
    }

    /// Feed a prompt in steps of up to 8 rows; returns the greedy token after it.
    pub fn prefill(&mut self, toks: &[u32]) -> Result<u32> {
        if !self.long_prompt(toks.len()) {
            return self.prefill_small(toks);
        }
        let t0 = std::time::Instant::now();
        self.ensure_pw(toks.len())?;
        let t1 = t0.elapsed().as_secs_f64();
        let gmax = self.pw.as_ref().unwrap().gmax;
        let mut t = 0;
        let mut r = Ok(());
        for ch in toks.chunks(gmax) {
            if self.cancelled() {
                r = Err(Cancelled.into());
                break;
            }
            match if ch.len() > MAX_ROWS { self.prefill_group(ch) } else { self.prefill_small(ch) } {
                Ok(v) => t = v,
                Err(e) => {
                    r = Err(e);
                    break;
                }
            }
        }
        let t2 = t0.elapsed().as_secs_f64();
        self.release_pw()?;
        if std::env::var("FNEXT_PF_TIME").is_ok() {
            eprintln!("prefill {} tokens: workspace {:.0} ms, groups {:.0} ms, release {:.0} ms", toks.len(), t1 * 1e3,
                (t2 - t1) * 1e3, (t0.elapsed().as_secs_f64() - t2) * 1e3);
        }
        r.map(|_| t)
    }

    /// The running request was cancelled.
    pub(crate) fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.load(Ordering::Acquire))
    }

    /// Wait for the work queued on every device (a prompt group stopped between layers), then [`Cancelled`].
    pub(crate) fn stop_cancelled(&self) -> anyhow::Error {
        let r = (|| -> Result<()> {
            if let Some(o) = &self.other {
                o.cu.bind()?;
                o.cu.synchronize()?;
            }
            self.cu.bind()?;
            self.cu.synchronize()?;
            Ok(())
        })();
        match r {
            Ok(()) => Cancelled.into(),
            Err(e) => e.context("waiting for the GPUs after a cancelled prompt"),
        }
    }

    /// Prompts this long take the GEMM path (FNEXT_PREFILL_MIN, default 64 tokens); shorter ones run as decode steps.
    fn long_prompt(&self, n: usize) -> bool {
        let min: usize = std::env::var("FNEXT_PREFILL_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        self.pcfg.0 > MAX_ROWS && n >= min
    }

    /// Allocate the prompt workspace (on every device), releasing LRU blocks for it as needed.
    fn ensure_pw(&mut self, n: usize) -> Result<()> {
        self.each_dev(|f| f.ensure_pw_dev(n))
    }

    fn ensure_pw_dev(&mut self, n: usize) -> Result<()> {
        if self.pw.is_some() {
            return Ok(());
        }
        let (chunk, pctx, group, gtok) = pw_shape(self.pcfg, self.ctx, self.c.n_expert, self.tokens.len(), n);
        let t0 = std::time::Instant::now();
        let mut released = 0;
        let free0 = fe_cuda::mem_info()?.0;
        // a workspace of a known shape: release what it took last time in one go (an overshoot would cost the prompt
        // the experts in those slots); unknown shapes release 256 MiB per failed attempt
        // no LRU upload may still be in flight while the prompt workspace takes card memory
        self.drain_uploads()?;
        let key = (chunk, pctx, group, gtok);
        // an unknown shape: estimate from a known one with the same chunk and group (the group-sized buffers scale
        // with gtok, the indexer scores with pctx): every failed attempt costs ~250 ms of allocations and frees
        let est = self.pw_need.get(&key).copied().or_else(|| {
            // the known shapes of this chunk and group (load probes them at pctx 8192): the nearest gtok at or above,
            // else the largest scaled up, + the per-row indexer scores / attention bits that grow with pctx
            let mut ks: Vec<(usize, usize, usize)> = self.pw_need.iter().filter(|(k, _)| k.0 == chunk && k.2 == group)
                .map(|(k, &v)| (k.3, k.1, v)).collect();
            ks.sort();
            let (g0, p0, v0) = ks.iter().copied().find(|e| e.0 >= gtok).or_else(|| ks.last().copied())?;
            let v = if g0 >= gtok { v0 } else { v0 * gtok / g0.max(1) };
            let per_ctx = chunk * 4 * (1 + 1) / self.c.kpool.max(1);
            Some(v + pctx.saturating_sub(p0) * per_ctx + (32 << 20))
        });
        if let Some(need) = est {
            let (free, _) = fe_cuda::mem_info()?;
            if free < need + (64 << 20) {
                released += self.shrink_lru(need + (64 << 20))?;
            }
        }
        let mut step = 256usize << 20;
        let dbg_t = std::env::var("FNEXT_PF_TIME").is_ok();
        if dbg_t {
            eprintln!("ensure_pw: drain + known-shape release {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        loop {
            let free_before = fe_cuda::mem_info()?.0;
            let ta = std::time::Instant::now();
            let res = prefill::PWork::alloc(&self.cfull, &self.g, chunk, pctx, group, gtok, self.tp);
            if dbg_t {
                eprintln!("ensure_pw: alloc attempt {} in {:.0} ms", if res.is_ok() { "ok" } else { "failed" }, ta.elapsed().as_secs_f64() * 1e3);
            }
            match res {
                Ok(p) => {
                    self.pw = Some(p);
                    let took = free_before.saturating_sub(fe_cuda::mem_info()?.0);
                    self.pw_need.insert(key, took);
                    if std::env::var("FNEXT_VERBOSE").is_ok() {
                        eprintln!("fnext-fast: prompt workspace ready (chunk {chunk}, group {gtok}: {released} LRU blocks released, \
                            free {} -> {} MiB, {:.0} ms)", free0 >> 20, fe_cuda::mem_info()?.0 >> 20,
                            t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    return Ok(());
                }
                Err(e) => {
                    let (free, _) = fe_cuda::mem_info()?;
                    if std::env::var("FNEXT_VERBOSE").is_ok() {
                        eprintln!("fnext-fast: workspace alloc failed at {} MiB free: {e:#}", free >> 20);
                    }
                    let ts = std::time::Instant::now();
                    let n = self.shrink_lru(free + step)?;
                    step *= 2;
                    if dbg_t {
                        eprintln!("ensure_pw: shrink {n} blocks in {:.0} ms", ts.elapsed().as_secs_f64() * 1e3);
                    }
                    if n == 0 {
                        return Err(e.context("no memory for the prompt workspace"));
                    }
                    released += n;
                }
            }
        }
    }

    fn release_pw(&mut self) -> Result<()> {
        self.each_dev(|f| f.release_pw_dev())
    }

    fn release_pw_dev(&mut self) -> Result<()> {
        self.s.sync()?;
        let t0 = std::time::Instant::now();
        self.pw = None;
        let t1 = t0.elapsed().as_secs_f64();
        let r = self.grow_lru();
        if std::env::var("FNEXT_VERBOSE").is_ok() {
            eprintln!("fnext-fast: prompt workspace freed in {:.0} ms, LRU blocks back in {:.0} ms", t1 * 1e3,
                (t0.elapsed().as_secs_f64() - t1) * 1e3);
        }
        r
    }

    fn prefill_small(&mut self, toks: &[u32]) -> Result<u32> {
        let mut t = 0;
        for ch in toks.chunks(MAX_ROWS) {
            t = self.step(ch)?;
        }
        Ok(t)
    }

    /// Forget the sequence (clear recurrent state; KV cells get overwritten).
    pub fn reset(&mut self) -> Result<()> {
        self.each_dev(|f| f.reset_dev())?;
        self.draft_p = 1.0;
        self.acc_ema = 0.9;
        self.snaps.clear();
        self.tokens.clear();
        Ok(())
    }

    fn reset_dev(&mut self) -> Result<()> {
        self.s.sync()?;
        for l in &mut self.layers[self.lo..self.hi] {
            match &mut l.mix {
                Mixer::Gdn(gd) => {
                    self.s.memset_d8(gd.conv_state.ptr(), 0, gd.conv_state.bytes())?;
                    self.s.memset_d8(gd.ssm.ptr(), 0, gd.ssm.bytes())?;
                }
                Mixer::Attn(at) => {
                    self.s.memset_d8(at.ring.ptr(), 0, at.ring.bytes())?;
                }
            }
            if let Some(p) = &l.ple {
                self.s.memset_d8(p.hist.ptr(), 0, p.hist.bytes())?;
            }
        }
        if let (Some(m), true) = (&self.mtp, self.is_last()) {
            self.s.memset_d8(m.at.ring.ptr(), 0, m.at.ring.bytes())?;
            self.s.memset_d8(m.hprev.ptr(), 0, m.hprev.bytes())?;
        }
        self.s.sync()?;
        Ok(())
    }

    /// Between steps: install finished uploads into the tables, and start uploads of this step's misses into the
    /// least recently used dynamic slots (at most FNEXT_UP_MAX in flight).
    fn refill(&mut self) -> Result<()> {
        let mut miss = std::mem::take(&mut *self.missed.lock().unwrap());
        miss.sort();
        miss.dedup();
        self.each_dev(|f| f.refill_dev(&miss))
    }

    /// refill for the active device: its finished uploads, and the misses of its blocks.
    fn refill_dev(&mut self, miss: &[(usize, usize)]) -> Result<()> {
        let (lo, hi) = (self.lo, self.hi);
        let Some(dc) = self.dynp.as_mut() else { return Ok(()) };
        let mut dirty = vec![false; self.c.n_layer];
        while let Ok((l, slot, e)) = dc.done_rx.try_recv() {
            dc.inflight -= 1;
            let base = dc.slot_ptr(l, slot);
            let off = &dc.offs[l];
            self.tabs_h[l][0][e] = base.0 + off[0] as u64;
            self.tabs_h[l][1][e] = base.0 + off[1] as u64;
            self.tabs_h[l][2][e] = base.0 + off[2] as u64;
            self.resident[l][e].store(2, Ordering::Relaxed);
            dc.slots[l][slot] = Some(e);
            dc.pending[l][slot] = false;
            dirty[l] = true;
            self.uploads += 1;
        }
        // most recently missed first is arbitrary; keep layer order
        let (tp, rank, own) = (self.tp, self.tp_rank, self.own_h.clone());
        for &(l, e) in miss.iter().filter(|m| if tp { own[m.0][m.1] as usize == rank } else { (lo..hi).contains(&m.0) }) {
            if dc.inflight >= dc.max_inflight {
                break;
            }
            if self.resident[l][e].load(Ordering::Relaxed) != 0 || dc.queued[l].contains(&e) {
                continue;
            }
            // victim: a free slot, else the least recently used non-pending slot
            let mut best: Option<(usize, u64)> = None;
            for sl in 0..dc.per_layer {
                if dc.pending[l][sl] {
                    continue;
                }
                let lu = match dc.slots[l][sl] {
                    None => 0,
                    Some(v) => self.last_use[l][v].load(Ordering::Relaxed),
                };
                if best.map_or(true, |b| lu < b.1) {
                    best = Some((sl, lu));
                }
            }
            let Some((sl, lu)) = best else { continue };
            // never evict something used in the current step window
            if lu + 1 >= self.step_no.load(Ordering::Relaxed) && dc.slots[l][sl].is_some() {
                continue;
            }
            if let Some(v) = dc.slots[l][sl].take() {
                self.resident[l][v].store(0, Ordering::Relaxed);
                self.tabs_h[l][0][v] = 0;
                self.tabs_h[l][1][v] = 0;
                self.tabs_h[l][2][v] = 0;
                dirty[l] = true;
            }
            dc.pending[l][sl] = true;
            dc.inflight += 1;
            dc.queued[l].push(e);
            dc.job_tx.send((l, sl, e, dc.slot_ptr(l, sl))).ok();
        }
        for l in lo..hi {
            dc.queued[l].retain(|&e| self.resident[l][e].load(Ordering::Relaxed) == 0);
            if dirty[l] {
                // stream-ordered (no device-wide sync: the uploader keeps running)
                let f = &self.layers[l].ffn;
                unsafe {
                    self.s.memcpy_htod(f.gtab.ptr(), &self.tabs_h[l][0])?;
                    self.s.memcpy_htod(f.utab.ptr(), &self.tabs_h[l][1])?;
                    self.s.memcpy_htod(f.dtab.ptr(), &self.tabs_h[l][2])?;
                }
            }
        }
        Ok(())
    }

    /// Drop every tensor that lives on the card (all but the routed trunk experts, the token embedding and the PLE
    /// table) from the page cache, and read the trunk experts ahead.
    fn tidy_page_cache(&self) {
        let hm = &self.hm;
        if !hm.on {
            return;
        }
        let nl = self.c.n_layer;
        for t in &self.g.tensors {
            let n = &t.name;
            let trunk_expert = n.contains("_exps.") && n.split('.').nth(1).and_then(|l| l.parse::<usize>().ok()).is_some_and(|l| l < nl);
            if trunk_expert {
                hm.willneed(&self.g, t);
            } else if n != "token_embd.weight" && n != "per_layer_token_embd.weight" {
                hm.drop_range(&self.g, t, 0, t.bytes as usize);
            }
        }
    }

    pub fn counts_snapshot(&self) -> Vec<Vec<u64>> {
        self.counts.iter().map(|l| l.iter().map(|v| v.load(Ordering::Relaxed)).collect()).collect()
    }
}

impl Drop for Fast {
    fn drop(&mut self) {
        // orderly teardown: no thread may still be inside the driver when the context goes
        let _ = self.drain_uploads();
        let _ = self.s.sync();
        if self.other.is_some() && self.swap_dev().is_ok() {
            let _ = self.drain_uploads();
            let _ = self.s.sync();
            let _ = self.swap_dev();
        }
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.cpu.take() {
            let _ = h.join();
        }
        let parked = self.other.as_mut().and_then(|o| o.dynp.take());
        for dc in [self.dynp.take(), parked].into_iter().flatten() {
            let mut dc = dc;
            let h = dc.up_h.take();
            drop(dc); // closes the job channel: the uploader's recv ends
            if let Some(h) = h {
                let _ = h.join();
            }
        }
    }
}

unsafe fn hc_mix(kn: &Kern, fk: &FKern, s: &Stream, w: &Work, hc: &Hc, c: &Config, rows: usize) -> Result<()> {
    let ld = c.hc_lr + hc.n_inj;
    // quant's fused pair: norm + [down | inject] GEMV + 1/hc + SiLU, then up GEMV + sigmoid gate + stream mean
    // (any row count: hc_down / hc_up run > 8 rows as 8-row chunks with the same per-row bits, so a 24-row prompt
    // step or a batched verify gets the bits of single-row steps)
    if hc.dinj.bf && hc.up.bf && hc_fused() {
        let q = fk.gq;
        let sc = 1.0 / c.hc as f32;
        let pf = if hc_pf() { hc.up.w.ptr() } else { DevPtr::NULL };
        q.hc_down_pf(s, w.lo.ptr(), ld, hc.dinj.w.ptr(), ld, w.res.ptr(), c.hc_dim(), hc.norm.ptr(), c.eps, sc, c.hc_lr,
            w.xn.ptr(), rows, pf)?;
        q.hc_up(s, w.mixed.ptr(), c.n_embd, hc.up.w.ptr(), w.lo.ptr(), ld, w.xn.ptr(), c.hc_dim(), sc, rows)?;
        return Ok(());
    }
    kn.rmsnorm_dec(s, w.xn.ptr(), w.res.ptr(), hc.norm.ptr(), c.n_embd, c.hc, rows * c.hc, c.eps)?;
    hrun(fk, s, &hc.dinj, w.xn.ptr(), c.hc_dim(), w.lo.ptr(), ld, rows, 1.0, false)?;
    s.launch(&fk.scale_ld, (g256(ld), rows as u32), 256, 0, (w.lo.ptr(), ld as u32, ld as u32, 1.0 / c.hc as f32))?;
    s.launch(&fk.silu_ld, (g256(c.hc_lr), rows as u32), 256, 0, (w.lo.ptr(), c.hc_lr as u32, ld as u32))?;
    hrun(fk, s, &hc.up, w.lo.ptr(), ld, w.gate.ptr(), c.hc_dim(), rows, 1.0, false)?;
    let n = rows * c.n_embd;
    s.launch(&kn.hc_mix, g256(n), 256, 0, (w.mixed.ptr(), w.xn.ptr(), w.gate.ptr(), c.n_embd as u32, c.hc as u32, n as u32))?;
    Ok(())
}

/// res += out * 2 sigmoid(inject) with the inject logits behind the low-rank rows of w.lo.
unsafe fn hc_combine(fk: &FKern, s: &Stream, w: &Work, c: &Config, rows: usize) -> Result<()> {
    let n = rows * c.hc_dim();
    let ld = c.hc_lr + c.hc;
    s.launch(&fk.hc_combine, g256(n), 256, 0,
        (w.res.ptr(), w.out.ptr(), w.lo.ptr().add::<f32>(c.hc_lr), c.n_embd as u32, c.hc as u32, ld as u32, n as u32))?;
    Ok(())
}

/// Full-attention (QSA) block on w.mixed -> w.out, positions from w.prm.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn attn_block(kn: &Kern, fk: &FKern, q: &Ggq, s: &Stream, w: &mut Work, at: &Attn, c: &Config, rows: usize,
    ctx: usize) -> Result<()> {
    let kv = lanes::KvPtr::of(at, c);
    let prm = w.prm.ptr();
    attn_block_segs(kn, fk, q, s, w, at, c, rows, ctx, &[(0, rows, prm, kv)])
}

/// The attention block over the step's rows: projections for all rows, then per segment (one lane's rows, its step
/// parameters, its KV / pooled keys / ring) the indexer, QSA selection, KV store and attention.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn attn_block_segs(kn: &Kern, fk: &FKern, q: &Ggq, s: &Stream, w: &mut Work, at: &Attn, c: &Config,
    rows: usize, ctx: usize, segs: &[(usize, usize, DevPtr, lanes::KvPtr)]) -> Result<()> {
    let d = c.n_embd;
    let qd = c.n_head * c.head_dim;
    let kvd = c.n_head_kv * c.head_dim;
    let dd = c.idx_dim;
    quant_mixed(q, s, w.mixed.ptr(), rows, d, &mut w.xa)?;
    at.wq.run(q, s, &w.xa, w.qf.ptr(), false)?;
    at.wk.run(q, s, &w.xa, w.kx.ptr(), false)?;
    at.wv.run(q, s, &w.xa, w.vx.ptr(), false)?;
    hrun(fk, s, &at.ik, w.mixed.ptr(), d, w.ikx.ptr(), dd, rows, 1.0, false)?;
    hrun(fk, s, &at.iq, w.mixed.ptr(), d, w.iqv.ptr(), dd * c.idx_heads, rows, 1.0, false)?;
    kn.rmsnorm_dec(s, w.iqv.ptr(), w.iqv.ptr(), at.iqn.ptr(), dd, 1, rows * c.idx_heads, c.eps)?;
    let top = (c.idx_top_k / c.kpool) as u32;
    let ldsc = (ctx / c.kpool + 1) as u32;
    let ldsel = (c.idx_top_k + c.kpool - 1) as u32;
    for &(r0, n, prm, kv) in segs {
        let (cs, sn, pcs, psn) = (prm.add::<u32>(P_CS), prm.add::<u32>(P_SN), prm.add::<u32>(P_PCS), prm.add::<u32>(P_PSN));
        let (ikx, iqv) = (w.ikx.ptr().add::<f32>(r0 * dd), w.iqv.ptr().add::<f32>(r0 * dd * c.idx_heads));
        let (score, sel, seln) = (w.score.ptr().add::<f32>(r0 * ldsc as usize), w.sel.ptr().add::<u32>(r0 * ldsel as usize),
            w.seln.ptr().add::<u32>(r0));
        let (qf, kx, vx) = (w.qf.ptr().add::<f32>(r0 * 2 * qd), w.kx.ptr().add::<f32>(r0 * kvd), w.vx.ptr().add::<f32>(r0 * kvd));
        let (qq, att) = (w.q.ptr().add::<f32>(r0 * qd), w.att.ptr().add::<f32>(r0 * qd));
        s.launch(&fk.idx_kstore, n as u32, dd as u32, 0, (kv.ring, ikx, prm, kv.ring_n))?;
        s.launch(&fk.idx_pool, (n / c.kpool + 1) as u32, dd as u32, 0, (kv.pooled, kv.ring, at.ikn.ptr(), prm,
            c.kpool as u32, c.n_rot as u32, c.eps, kv.ring_n, pcs, psn))?;
        s.launch(&kn.rope_rows, (n as u32, c.idx_heads as u32), dd as u32, 0, (iqv, cs, sn, c.idx_heads as u32, c.n_rot as u32))?;
        s.launch(&fk.idx_score, (n as u32, 72), 256, 0, (score, iqv, kv.pooled, c.idx_heads as u32, prm, c.kpool as u32,
            ldsc, top, 1.0 / (dd as f32).sqrt()))?;
        s.launch(&fk.topk, n as u32, 1024, 0, (sel, seln, score, prm, c.kpool as u32, top, ldsc, ldsel))?;
        s.launch(&fk.attn_prep, (n as u32, (c.n_head + c.n_head_kv) as u32), 256, 0,
            (qq, kv.kq, kv.ks, kv.vq, kv.vs, qf, (2 * qd) as u32, (2 * c.head_dim) as u32, kx, vx, kvd as u32, at.qn.ptr(),
                at.kn.ptr(), prm, c.n_head as u32, c.n_head_kv as u32, c.n_rot as u32, c.eps, cs, sn))?;
        s.launch(&fk.attn, (n as u32, c.n_head as u32), 256, 0, (att, qq, kv.kq, kv.ks, kv.vq, kv.vs,
            qf.add::<f32>(c.head_dim), (2 * qd) as u32, (2 * c.head_dim) as u32, c.n_head as u32, c.n_head_kv as u32, prm,
            1.0 / (c.head_dim as f32).sqrt(), sel, seln, ldsel))?;
    }
    quant_mixed(q, s, w.att.ptr(), rows, qd, &mut w.xa)?;
    at.wo.run(q, s, &w.xa, w.out.ptr(), false)?;
    Ok(())
}

/// Cells of the raw indexer-key ring: a power of two above a prompt chunk plus the open block.
/// Largest prompt chunk ensure_pw picks on its own (long prompts).
const PROMPT_CHUNK_MAX: usize = 4096;

/// Indexer key ring cells: the largest prompt chunk plus the pooling tail.
fn ring_cells() -> usize {
    let chunk: usize = std::env::var("FNEXT_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    (chunk.max(PROMPT_CHUNK_MAX).max(MAX_ROWS) + 8).next_power_of_two()
}

/// Bytes of one expert (gate + up + down) of layer l.
pub fn expert_bytes(g: &Gguf, l: usize) -> Result<usize> {
    let mut b = 0;
    for nm in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
        b += tensor(g, &format!("blk.{l}.{nm}.weight"))?.slice_bytes().div_ceil(256) * 256;
    }
    Ok(b)
}

pub fn load_counts(path: &str) -> Result<Vec<Vec<u64>>> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    v.as_array().ok_or_else(|| anyhow!("counts json"))?.iter()
        .map(|l| l.as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0)).collect()).ok_or_else(|| anyhow!("counts json")))
        .collect()
}

/// LRU expert slots per layer, refilled by an uploader thread (pinned staging, its own stream).
struct DynCache {
    per_layer: usize,
    /// per layer, blocks of blk[l] slots (a block can be released for the prompt workspace and allocated again)
    bufs: Vec<Vec<Option<DeviceBuffer<u8>>>>,
    slot_bytes: Vec<usize>,
    /// byte offsets of gate / up / down inside a slot, per layer
    offs: Vec<[usize; 3]>,
    /// slots per allocation block, per layer
    blk: Vec<usize>,
    slots: Vec<Vec<Option<usize>>>,
    pending: Vec<Vec<bool>>,
    queued: Vec<Vec<usize>>,
    inflight: usize,
    max_inflight: usize,
    job_tx: std::sync::mpsc::Sender<(usize, usize, usize, DevPtr)>,
    /// the uploader thread (joined on drop, after the sender closes)
    up_h: Option<std::thread::JoinHandle<()>>,
    done_rx: std::sync::mpsc::Receiver<(usize, usize, usize)>,
}

impl DynCache {
    /// Slots for blocks lo..hi (the device's layers; the others get no buffers) on the current context.
    fn new(g: &Arc<Gguf>, c: &Config, budget: usize, cu: &Arc<fe_cuda::Context>, lo: usize, hi: usize, cap: usize)
        -> Result<DynCache> {
        let mut slot_bytes = vec![];
        let mut offs = vec![];
        for l in 0..c.n_layer {
            let mut o = [0usize; 3];
            let mut b = 0;
            for (i, nm) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].iter().enumerate() {
                o[i] = b;
                b += tensor(g, &format!("blk.{l}.{nm}.weight"))?.slice_bytes().div_ceil(256) * 256;
            }
            offs.push(o);
            slot_bytes.push(b);
        }
        // slots per block per layer: card allocations come in 2 MiB pages, so a block of 8 ~1.9 MB slots wasted ~1.3
        // MiB (~720 blocks a card: ~0.9 GB). Pick the block size (4..=16 slots) with the least waste per slot.
        let blk: Vec<usize> = slot_bytes.iter().map(|&b| {
            (4..=16usize).min_by_key(|&n| ((n * b).next_multiple_of(ALLOC_G) - n * b) * 1024 / n * 64 + (16 - n)).unwrap()
        }).collect();
        let layer_bytes = |l: usize, per: usize| -> usize {
            let n = blk[l];
            (0..per.div_ceil(n)).map(|k| ((per - k * n).min(n) * slot_bytes[l]).next_multiple_of(ALLOC_G)).sum()
        };
        // never more slots than experts (two cards can hold most of a layer's experts)
        let mut per = 0usize;
        while per < cap && (lo..hi).map(|l| layer_bytes(l, per + 1)).sum::<usize>() <= budget {
            per += 1;
        }
        let mut bufs = vec![];
        for l in 0..c.n_layer {
            let n = blk[l];
            let mut bl = vec![];
            for k in 0..if (lo..hi).contains(&l) { per.div_ceil(n) } else { 0 } {
                let m = (per - k * n).min(n);
                bl.push(Some(DeviceBuffer::<u8>::alloc(slot_bytes[l] * m)?));
            }
            bufs.push(bl);
        }
        let (job_tx, job_rx) = std::sync::mpsc::channel::<(usize, usize, usize, DevPtr)>();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let g2 = g.clone();
        let cu2 = cu.clone();
        let offs2 = offs.clone();
        let maxb = *slot_bytes.iter().max().unwrap();
        let up_h = std::thread::Builder::new().name("fnext-upload".into()).spawn(move || {
            // two pinned staging buffers: fill one from the mmap while the other's upload runs
            if std::env::var("FNEXT_NO_PIN").is_err() {
                fe_ggq::cpu::pool::pin_current(crate::cores::plan().up);
            }
            cu2.bind().unwrap();
            let st = Stream::new().unwrap();
            let mut stage: Vec<HostBuffer<u8>> = (0..2).map(|_| HostBuffer::alloc(maxb).unwrap()).collect();
            // poll the copies with sleeps: a spinning sync here holds up the launching thread's driver calls
            let ev: Vec<Event> = (0..2).map(|_| Event::new_blocking().unwrap()).collect();
            let wait = |e: &Event| {
                while !e.is_done().unwrap_or(true) {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            };
            let mut busy: [Option<(usize, usize, usize)>; 2] = [None, None];
            let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
            let mut k = 0usize;
            loop {
                // finish the buffer we are about to reuse
                if let Some(job) = busy[k].take() {
                    wait(&ev[k]);
                    if done_tx.send(job).is_err() {
                        return;
                    }
                }
                let job = if busy[1 - k].is_some() {
                    match job_rx.try_recv() {
                        Ok(j) => j,
                        Err(std::sync::mpsc::TryRecvError::Empty) => {
                            // nothing new: complete the other one, then block
                            let o = 1 - k;
                            wait(&ev[o]);
                            if done_tx.send(busy[o].take().unwrap()).is_err() {
                                return;
                            }
                            match job_rx.recv() { Ok(j) => j, Err(_) => return }
                        }
                        Err(_) => return,
                    }
                } else {
                    match job_rx.recv() { Ok(j) => j, Err(_) => return }
                };
                let (l, slot, e, base) = job;
                let sb = stage[k].as_mut_slice();
                let mut n = 0;
                for i in 0..3 {
                    let t = g2.tensor(&format!("blk.{l}.{}.weight", names[i])).unwrap();
                    let src = g2.slice(t, e);
                    sb[offs2[l][i]..offs2[l][i] + src.len()].copy_from_slice(src);
                    n = offs2[l][i] + src.len();
                }
                unsafe { st.memcpy_htod(base, &stage[k].as_slice()[..n]).unwrap() };
                st.record(&ev[k]).unwrap();
                busy[k] = Some((l, slot, e));
                k = 1 - k;
            }
        })?;
        let max_inflight = std::env::var("FNEXT_UP_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        Ok(DynCache {
            per_layer: per, bufs, slot_bytes, offs, blk,
            slots: vec![vec![None; per]; c.n_layer], pending: vec![vec![false; per]; c.n_layer], queued: vec![vec![]; c.n_layer],
            inflight: 0, max_inflight, job_tx, done_rx, up_h: Some(up_h),
        })
    }
    fn slot_ptr(&self, l: usize, slot: usize) -> DevPtr {
        let n = self.blk[l];
        self.bufs[l][slot / n].as_ref().expect("slot block released").ptr().byte_add((slot % n) * self.slot_bytes[l])
    }
}

/// Card allocation granularity (large cuMemAlloc sizes round up to 2 MiB pages).
const ALLOC_G: usize = 2 << 20;

impl Fast {
    /// Map KV / pooled-key memory for at least `cells` positions (steps of 8192), releasing LRU blocks when the card
    /// is short. Returns the bytes newly mapped.
    pub fn ensure_ctx(&mut self, cells: usize) -> Result<usize> {
        if cells > self.ctx_cap {
            bail!("context of {cells} cells over this lane's budget ({})", self.ctx_cap);
        }
        let cells = cells.min(self.ctx).next_multiple_of(8192).min(self.ctx);
        if cells <= self.ctx_mapped {
            return Ok(0);
        }
        let mut grown = 0;
        self.each_dev(|f| {
            grown += f.ensure_ctx_dev(cells)?;
            Ok(())
        })?;
        self.ctx_mapped = cells;
        Ok(grown)
    }

    /// ensure_ctx for the active device's attention layers (and the MTP block on the last device).
    fn ensure_ctx_dev(&mut self, cells: usize) -> Result<usize> {
        let c = self.c.clone();
        let kvd = c.n_head_kv * c.head_dim;
        let last = self.is_last();
        let (lo, hi) = (self.lo, self.hi);
        let n_attn = self.layers[lo..hi].iter().filter(|l| matches!(l.mix, Mixer::Attn(_))).count()
            + (self.mtp.is_some() && last) as usize * (1 + self.tp as usize);
        let per_cell = 2 * kvd + 2 * kvd / 32 * 4 + c.idx_dim * 4 / c.kpool;
        let need = (cells - self.ctx_mapped) * per_cell * n_attn + (n_attn * 5 * (2 << 20));
        let (free, _) = fe_cuda::mem_info()?;
        let margin = 256usize << 20;
        if free < need + margin {
            self.shrink_lru(need + margin)?;
        }
        let mut grown = 0;
        let kvd_full = self.cfull.n_head_kv * self.cfull.head_dim;
        let mut grow = |a: &mut Attn, kvd: usize| -> Result<()> {
            grown += a.kq.grow(cells * kvd)? + a.vq.grow(cells * kvd)?;
            grown += a.ks.grow(cells * kvd / 32 * 4)? + a.vs.grow(cells * kvd / 32 * 4)?;
            grown += a.pooled.grow((cells / c.kpool + 1) * c.idx_dim * 4)?;
            Ok(())
        };
        for l in &mut self.layers[lo..hi] {
            if let Mixer::Attn(a) = &mut l.mix {
                grow(a, kvd)?;
            }
        }
        if let (Some(m), true) = (&mut self.mtp, last) {
            grow(&mut m.at, kvd_full)?;
        }
        Ok(grown)
    }

    /// Capture a graph; when instantiation runs out of card memory, release LRU blocks and try again.
    fn capture(&mut self, f: impl Fn(&mut Fast) -> Result<()>) -> Result<Graph> {
        for attempt in 0..4 {
            self.s.begin_capture()?;
            let r = f(self);
            let g = self.s.end_capture();
            r?;
            match g {
                Ok(g) => return Ok(g),
                Err(e) if attempt < 3 && e.code == 2 => {
                    let (free, _) = fe_cuda::mem_info()?;
                    self.shrink_lru(free + (384 << 20))?;
                }
                Err(e) => return Err(e.into()),
            }
        }
        unreachable!()
    }

    /// Wait for every upload in flight and install it.
    fn drain_uploads(&mut self) -> Result<()> {
        // every layer an install lands in gets its device tables rewritten: the CPU thread reads `resident`, the
        // card the tables, and they must agree (a resident-on-host / missing-on-card expert hangs the step)
        let mut dirty: Vec<usize> = vec![];
        loop {
            let busy = self.dynp.as_ref().map_or(0, |d| d.inflight);
            if busy == 0 {
                for l in dirty {
                    self.flush_tables(l)?;
                }
                return Ok(());
            }
            let dc = self.dynp.as_mut().unwrap();
            let (l, slot, e) = dc.done_rx.recv()?;
            if !dirty.contains(&l) {
                dirty.push(l);
            }
            dc.inflight -= 1;
            let base = dc.slot_ptr(l, slot);
            let off = dc.offs[l];
            self.tabs_h[l][0][e] = base.0 + off[0] as u64;
            self.tabs_h[l][1][e] = base.0 + off[1] as u64;
            self.tabs_h[l][2][e] = base.0 + off[2] as u64;
            self.resident[l][e].store(2, Ordering::Relaxed);
            dc.slots[l][slot] = Some(e);
            dc.pending[l][slot] = false;
            dc.queued[l].retain(|&x| x != e);
        }
    }

    fn flush_tables(&mut self, l: usize) -> Result<()> {
        let f = &mut self.layers[l].ffn;
        f.gtab.copy_from(&self.tabs_h[l][0])?;
        f.utab.copy_from(&self.tabs_h[l][1])?;
        f.dtab.copy_from(&self.tabs_h[l][2])?;
        Ok(())
    }

    /// Release LRU blocks (the last block of each layer, round robin) until `bytes` are free on the card.
    fn shrink_lru(&mut self, bytes: usize) -> Result<usize> {
        self.drain_uploads()?;
        let nl = self.c.n_layer;
        let mut released = 0;
        loop {
            let (free, _) = fe_cuda::mem_info()?;
            if free >= bytes {
                return Ok(released);
            }
            // release blocks until their (2 MiB-rounded) sizes cover the shortfall, then rewrite each touched layer's
            // tables once and free the buffers (one table upload + one mem_info per batch, not per block)
            let mut short = bytes - free;
            let mut dirty: Vec<usize> = vec![];
            let mut freed: Vec<DeviceBuffer<u8>> = vec![];
            while short > 0 {
                let Some(dc) = self.dynp.as_mut() else { break };
                // the layer with the most live blocks
                let l = (0..nl).max_by_key(|&l| dc.bufs[l].iter().filter(|b| b.is_some()).count()).unwrap();
                let Some(k) = dc.bufs[l].iter().rposition(|b| b.is_some()) else { break };
                let n = dc.blk[l];
                let m = ((k + 1) * n).min(dc.per_layer) - k * n;
                for sl in k * n..k * n + m {
                    if let Some(e) = dc.slots[l][sl].take() {
                        self.resident[l][e].store(0, Ordering::Relaxed);
                        self.tabs_h[l][0][e] = 0;
                        self.tabs_h[l][1][e] = 0;
                        self.tabs_h[l][2][e] = 0;
                    }
                    dc.pending[l][sl] = true;
                }
                let sz = (m * dc.slot_bytes[l]).next_multiple_of(ALLOC_G);
                if let Some(b) = dc.bufs[l][k].take() {
                    freed.push(b);
                }
                if !dirty.contains(&l) {
                    dirty.push(l);
                }
                released += 1;
                short = short.saturating_sub(sz);
            }
            if dirty.is_empty() {
                return Ok(released);
            }
            for &l in &dirty {
                self.flush_tables(l)?;
            }
            drop(freed);
        }
    }

    /// Allocate the released LRU blocks again (empty slots: the LRU refills them).
    fn grow_lru(&mut self) -> Result<()> {
        let Some(dc) = self.dynp.as_mut() else { return Ok(()) };
        for l in 0..dc.bufs.len() {
            for k in 0..dc.bufs[l].len() {
                if dc.bufs[l][k].is_none() {
                    let bn = dc.blk[l];
                    let n = (dc.per_layer - k * bn).min(bn);
                    match DeviceBuffer::<u8>::alloc(dc.slot_bytes[l] * n) {
                        Ok(b) => {
                            dc.bufs[l][k] = Some(b);
                            for sl in k * bn..k * bn + n {
                                dc.pending[l][sl] = false;
                            }
                        }
                        Err(_) => return Ok(()),
                    }
                }
            }
        }
        Ok(())
    }
}

/// Speculative decoding state: the last sampled token (not yet run through the trunk) and its drafts.
pub struct Spec {
    pub cur: u32,
    pub drafts: Vec<u32>,
}

/// Hyper-connection mixer with quantized weights (the MTP block stores them as Q8_0).
struct QHc {
    norm: DeviceBuffer<f32>,
    down: QW,
    inject: Option<QW>,
    up: QW,
}
impl QHc {
    fn load(g: &Gguf, pre: &str, norm: &str, down: &str, up: &str, inject: Option<&str>) -> Result<QHc> {
        Ok(QHc {
            norm: vecf(g, &format!("{pre}{norm}"))?,
            down: QW::load(g, &format!("{pre}{down}"))?,
            inject: inject.map(|i| QW::load(g, &format!("{pre}{i}"))).transpose()?,
            up: QW::load(g, &format!("{pre}{up}"))?,
        })
    }
}

/// The MTP block (blk.48): [enorm(e) | hnorm(h)_s] -> eh_proj per stream, one QSA attention layer + MoE (all its
/// experts on the card), its own head mixer, the shared output head.
pub(crate) struct Mtp {
    /// hc_attn_up as a 16-bit Lin for the prompt pass (K = 320 is not an int8-GEMM shape)
    hca_up: Lin,
    hnorm: DeviceBuffer<f32>,
    enorm: DeviceBuffer<f32>,
    eh: QW,
    hca: QHc,
    hcf: QHc,
    head: QHc,
    at: Attn,
    gate_inp: Lin,
    n_exp: usize,
    shg: QW,
    shu: QW,
    shd: QW,
    ty_gu: GgmlType,
    ty_d: GgmlType,
    gtab: DeviceBuffer<u64>,
    utab: DeviceBuffer<u64>,
    dtab: DeviceBuffer<u64>,
    _arena: Vec<DeviceBuffer<u8>>,
    hin: DeviceBuffer<f32>,
    hprev: DeviceBuffer<f32>,
    en: DeviceBuffer<f32>,
    cat: DeviceBuffer<f32>,
    injb: DeviceBuffer<f32>,
    rl: DeviceBuffer<f32>,
    ids: DeviceBuffer<i32>,
    wts: DeviceBuffer<f32>,
    ea8: Q8Act,
}

impl Mtp {
    fn load(g: &Gguf, c: &Config, ctx: usize, ord: i32) -> Result<Mtp> {
        let l = c.n_layer;
        let p = format!("blk.{l}.");
        let kvd = c.n_head_kv * c.head_dim;
        let at = Attn {
            wq: QW::load(g, &format!("{p}attn_q.weight"))?,
            wk: QW::load(g, &format!("{p}attn_k.weight"))?,
            wv: QW::load(g, &format!("{p}attn_v.weight"))?,
            wo: QW::load(g, &format!("{p}attn_output.weight"))?,
            qn: vecf(g, &format!("{p}attn_q_norm.weight"))?,
            kn: vecf(g, &format!("{p}attn_k_norm.weight"))?,
            kq: VmBuffer::reserve(ctx * kvd, ord)?,
            ks: VmBuffer::reserve(ctx * kvd / 32 * 4, ord)?,
            vq: VmBuffer::reserve(ctx * kvd, ord)?,
            vs: VmBuffer::reserve(ctx * kvd / 32 * 4, ord)?,
            iq: Lin::from_tensor(g, &format!("{p}indexer.q_proj.weight"))?,
            ik: Lin::from_tensor(g, &format!("{p}indexer.k_proj.weight"))?,
            iqn: vecf(g, &format!("{p}indexer.q_norm.weight"))?,
            ikn: vecf(g, &format!("{p}indexer.k_norm.weight"))?,
            ring: DeviceBuffer::zeroed(ring_cells() * c.idx_dim)?,
            pooled: VmBuffer::reserve((ctx / c.kpool + 1) * c.idx_dim * 4, ord)?,
        };
        let tg = tensor(g, &format!("{p}ffn_gate_exps.weight"))?.clone();
        let tu = tensor(g, &format!("{p}ffn_up_exps.weight"))?.clone();
        let td = tensor(g, &format!("{p}ffn_down_exps.weight"))?.clone();
        let n_exp = tg.dims[2] as usize;
        // all MTP experts on the card
        let mut arena = vec![];
        let mut tabs = [vec![0u64; n_exp], vec![0u64; n_exp], vec![0u64; n_exp]];
        for (i, t) in [&tg, &tu, &td].iter().enumerate() {
            let buf = DeviceBuffer::from_slice(g.data(t))?;
            for e in 0..n_exp {
                tabs[i][e] = buf.ptr().byte_add(e * t.slice_bytes()).0;
            }
            arena.push(buf);
        }
        let a = |n: usize| DeviceBuffer::<f32>::zeroed(n);
        let r = MAX_ROWS;
        Ok(Mtp {
            hnorm: vecf(g, &format!("{p}nextn.hnorm.weight"))?,
            enorm: vecf(g, &format!("{p}nextn.enorm.weight"))?,
            eh: QW::load(g, &format!("{p}nextn.eh_proj.weight"))?,
            hca_up: Lin::from_tensor(g, &format!("{p}hc_attn_up.weight"))?,
            hca: QHc::load(g, &p, "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", Some("hc_attn_inject.weight"))?,
            hcf: QHc::load(g, &p, "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", Some("hc_ffn_inject.weight"))?,
            head: QHc::load(g, &p, "nextn.hc_head_norm.weight", "nextn.hc_head_down.weight", "nextn.hc_head_up.weight", None)?,
            at,
            gate_inp: Lin::stacked(g, &format!("{p}ffn_gate_inp.weight"), &format!("{p}ffn_gate_inp_shexp.weight"))?,
            n_exp,
            shg: QW::load(g, &format!("{p}ffn_gate_shexp.weight"))?,
            shu: QW::load(g, &format!("{p}ffn_up_shexp.weight"))?,
            shd: QW::load(g, &format!("{p}ffn_down_shexp.weight"))?,
            ty_gu: tg.ty,
            ty_d: td.ty,
            gtab: DeviceBuffer::from_slice(&tabs[0])?,
            utab: DeviceBuffer::from_slice(&tabs[1])?,
            dtab: DeviceBuffer::from_slice(&tabs[2])?,
            _arena: arena,
            hin: a(r * c.hc_dim())?,
            hprev: a(c.hc_dim())?,
            en: a(r * c.n_embd)?,
            cat: a(r * c.hc * 2 * c.n_embd)?,
            injb: a(r * c.hc)?,
            rl: a(r * (n_exp + 1))?,
            ids: DeviceBuffer::zeroed(r * c.n_expert_used)?,
            wts: a(r * c.n_expert_used)?,
            ea8: Q8Act::alloc(8, 2 * c.n_embd)?,
        })
    }
}

/// HC mixer with quantized weights: w.res -> w.xn, w.mixed; inject logits (scaled) into `injb` [rows][hc].
unsafe fn qhc_mix(kn: &Kern, fk: &FKern, q: &Ggq, s: &Stream, w: &mut Work, hc: &QHc, injb: DevPtr, c: &Config, rows: usize)
    -> Result<()> {
    let sc = 1.0 / c.hc as f32;
    kn.rmsnorm_dec(s, w.xn.ptr(), w.res.ptr(), hc.norm.ptr(), c.n_embd, c.hc, rows * c.hc, c.eps)?;
    q.quantize_q8_1(s, w.xn.ptr(), c.hc_dim(), false, rows, c.hc_dim(), &mut w.xa)?;
    hc.down.run(q, s, &w.xa, w.lo.ptr(), false)?;
    s.launch(&fk.scale_ld, (g256(c.hc_lr), rows as u32), 256, 0, (w.lo.ptr(), c.hc_lr as u32, c.hc_lr as u32, sc))?;
    if let Some(inj) = &hc.inject {
        inj.run(q, s, &w.xa, injb, false)?;
        s.launch(&fk.scale_ld, (1u32, rows as u32), 256, 0, (injb, c.hc as u32, c.hc as u32, sc))?;
    }
    s.launch(&fk.silu_ld, (g256(c.hc_lr), rows as u32), 256, 0, (w.lo.ptr(), c.hc_lr as u32, c.hc_lr as u32))?;
    q.quantize_q8_1(s, w.lo.ptr(), c.hc_lr, false, rows, c.hc_lr, &mut w.sa)?;
    hc.up.run(q, s, &w.sa, w.gate.ptr(), false)?;
    let n = rows * c.n_embd;
    s.launch(&kn.hc_mix, g256(n), 256, 0, (w.mixed.ptr(), w.xn.ptr(), w.gate.ptr(), c.n_embd as u32, c.hc as u32, n as u32))?;
    Ok(())
}

impl Fast {
    /// The device state a prompt-cache snapshot holds besides the positional KV / pooled keys: GDN conv + recurrent
    /// state, the indexer key rings (the open pooling block), the PLE conv history, the MTP ring and its h row.
    fn snap_regions(&self) -> Vec<(DevPtr, usize)> {
        let mut v = vec![];
        for l in &self.layers[self.lo..self.hi] {
            match &l.mix {
                Mixer::Gdn(gd) => {
                    v.push((gd.conv_state.ptr(), gd.conv_state.bytes()));
                    v.push((gd.ssm.ptr(), gd.ssm.bytes()));
                }
                Mixer::Attn(at) => v.push((at.ring.ptr(), at.ring.bytes())),
            }
            if let Some(p) = &l.ple {
                v.push((p.hist.ptr(), p.hist.bytes()));
            }
        }
        if let (Some(m), true) = (&self.mtp, self.is_last()) {
            v.push((m.at.ring.ptr(), m.at.ring.bytes()));
            v.push((m.hprev.ptr(), m.hprev.bytes()));
        }
        v
    }

    /// Save the state at the current position (pinned host memory, ~170 MB; one buffer per device).
    pub fn snap_save(&mut self) -> Result<Snap> {
        let mut bufs = vec![];
        self.each_dev(|f| {
            let regs = f.snap_regions();
            let tot: usize = regs.iter().map(|r| r.1).sum();
            let tq = std::time::Instant::now();
            let mut buf = snap_take(tot)?;
            let t_take = tq.elapsed().as_secs_f64();
            let mut o = 0;
            f.s.sync()?;
            let t_sync = tq.elapsed().as_secs_f64();
            for (p, n) in regs {
                unsafe { f.s.memcpy_dtoh(&mut buf.as_mut_slice()[o..o + n], p)? };
                o += n;
            }
            f.s.sync()?;
            if std::env::var("FNEXT_VERBOSE").is_ok() {
                eprintln!("snap_save card {}: {} MB, take {:.0} ms, sync {:.0} ms, copy {:.0} ms", f.tp_rank, tot >> 20, t_take * 1e3,
                    (t_sync - t_take) * 1e3, (tq.elapsed().as_secs_f64() - t_sync) * 1e3);
            }
            bufs.push(buf);
            Ok(())
        })?;
        Ok(Snap { pos: self.tokens.len(), bufs })
    }

    /// Resume at a snapshot of this sequence (self.tokens[..s.pos] must be the tokens it was taken at).
    pub fn snap_restore(&mut self, sn: &Snap) -> Result<()> {
        if sn.pos > self.tokens.len() {
            bail!("snapshot at {} past the sequence ({})", sn.pos, self.tokens.len());
        }
        let mut k = 0;
        self.each_dev(|f| {
            let regs = f.snap_regions();
            let buf = &sn.bufs[k];
            let mut o = 0;
            f.s.sync()?;
            for (p, n) in regs {
                unsafe { f.s.memcpy_htod(p, &buf.as_slice()[o..o + n])? };
                o += n;
            }
            f.s.sync()?;
            k += 1;
            Ok(())
        })?;
        self.tokens.truncate(sn.pos);
        self.draft_p = 1.0;
        Ok(())
    }

    /// Prefill `toks` (positions after self.tokens) split at the absolute positions `cuts` (prompt-cache snapshot
    /// points: a fresh request and a resumed one then compute every segment the same way, so their bits match);
    /// `on_cut` runs after each cut that is reached (also one at the very end). Returns the next token and, with
    /// `mtp`, the first draft.
    pub fn prefill_cut(&mut self, toks: &[u32], cuts: &[usize], mtp: bool,
        mut on_cut: impl FnMut(&mut Fast) -> Result<()>) -> Result<(u32, Option<u32>)> {
        PREFILL_BUSY.store(true, Ordering::Relaxed);
        let r = self.prefill_cut_inner(toks, cuts, mtp, &mut on_cut);
        PREFILL_BUSY.store(false, Ordering::Relaxed);
        r
    }

    fn prefill_cut_inner(&mut self, toks: &[u32], cuts: &[usize], mtp: bool,
        on_cut: &mut impl FnMut(&mut Fast) -> Result<()>) -> Result<(u32, Option<u32>)> {
        let base = self.tokens.len();
        let end = base + toks.len();
        let mut a = 0usize;
        let mut out = (0u32, None);
        let mut run = |m: &mut Fast, seg: &[u32]| -> Result<(u32, Option<u32>)> {
            if mtp { m.prefill_mtp(seg).map(|(t, d)| (t, Some(d))) } else { m.prefill(seg).map(|t| (t, None)) }
        };
        for &c in cuts.iter().filter(|&&c| c > base && c <= end) {
            let seg = &toks[a..c - base];
            if seg.is_empty() {
                continue;
            }
            out = run(self, seg)?;
            on_cut(self)?;
            a = c - base;
        }
        if a < toks.len() {
            out = run(self, &toks[a..])?;
        }
        Ok(out)
    }

    /// Confidence of the last MTP draft (softmax probability of its argmax).
    pub fn draft_p(&self) -> f32 {
        self.draft_p
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Time `n` back-to-back replays of the 1-row MTP graph (ms each), and of the 1-row trunk graph.
    pub fn time_graphs(&mut self, n: usize) -> Result<(f64, f64)> {
        let pos = self.tokens.len();
        self.mtp_run(&[1000], pos)?;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            self.mtp_graphs[1].as_ref().unwrap().launch(&self.s)?;
        }
        self.s.sync()?;
        let a = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..n {
            if let Some(g) = self.graphs[9].as_ref() {
                g.launch(&self.s)?;
            }
        }
        self.s.sync()?;
        let b = t1.elapsed().as_secs_f64() * 1e3 / n as f64;
        Ok((a, b))
    }

    /// Enqueue an MTP pass over `rows` rows (positions and embeddings in the pinned step buffers, h in mtp.hin):
    /// leaves the block's output residual in w.res (hc-wide) and the greedy draft after the last row in tok_h[0].
    unsafe fn enqueue_mtp(&mut self, rows: usize) -> Result<()> {
        let c = self.cfull.clone();
        let (kn, fk, q, s) = (&self.kn, &self.fk, &self.q, &self.s);
        let w = &mut self.w;
        let m = self.mtp.as_mut().unwrap();
        let d = c.n_embd;
        let hd = c.hc_dim();
        let topk = c.n_expert_used;
        s.memcpy_htod(w.prm.ptr(), self.prm_h.as_slice())?;
        s.memcpy_htod(w.emb.ptr(), &self.emb_h.as_slice()[..rows * d])?;
        // [enorm(e) | hnorm(h)_s] -> eh_proj, one activation row per (row, stream)
        kn.rmsnorm_dec(s, w.xn.ptr(), m.hin.ptr(), m.hnorm.ptr(), d, c.hc, rows * c.hc, c.eps)?;
        kn.rmsnorm_dec(s, m.en.ptr(), w.emb.ptr(), m.enorm.ptr(), d, 1, rows, c.eps)?;
        s.launch(&fk.mtp_cat, (g256(2 * d), (rows * c.hc) as u32), 256, 0, (m.cat.ptr(), m.en.ptr(), w.xn.ptr(), d as u32,
            c.hc as u32))?;
        let na = rows * c.hc;
        let mut r0 = 0;
        while r0 < na {
            let nr = (na - r0).min(8);
            q.quantize_q8_1(s, m.cat.ptr().add::<f32>(r0 * 2 * d), 2 * d, false, nr, 2 * d, &mut m.ea8)?;
            m.eh.run(q, s, &m.ea8, w.res.ptr().add::<f32>(r0 * d), false)?;
            r0 += nr;
        }
        // attention block
        let skip = std::env::var("FNEXT_MTP_SKIP").unwrap_or_default();
        qhc_mix(kn, fk, q, s, w, &m.hca, m.injb.ptr(), &c, rows)?;
        if !skip.contains("attn") {
            attn_block(kn, fk, q, s, w, &m.at, &c, rows, self.ctx)?;
        }
        let n = rows * hd;
        s.launch(&fk.hc_combine, g256(n), 256, 0, (w.res.ptr(), w.out.ptr(), m.injb.ptr(), d as u32, c.hc as u32,
            c.hc as u32, n as u32))?;
        // MoE (every expert on the card)
        qhc_mix(kn, fk, q, s, w, &m.hcf, m.injb.ptr(), &c, rows)?;
        let ldr = m.n_exp + 1;
        hrun(fk, s, &m.gate_inp, w.mixed.ptr(), d, m.rl.ptr(), ldr, rows, 1.0, false)?;
        s.launch(&fk.route8, rows as u32, 256, 0, (m.ids.ptr(), m.wts.ptr(), m.rl.ptr(), ldr as u32, m.n_exp as u32,
            topk as u32, c.expert_weights_scale))?;
        quant_mixed(q, s, w.mixed.ptr(), rows, d, &mut w.ea)?;
        m.shg.run(q, s, &w.ea, w.sg.ptr(), false)?;
        m.shu.run(q, s, &w.ea, w.su.ptr(), false)?;
        let nf = rows * c.n_ff_shexp;
        s.launch(&kn.swiglu, g256(nf), 256, 0, (w.sg.ptr(), w.sg.ptr(), w.su.ptr(), nf as u32))?;
        quant_mixed(q, s, w.sg.ptr(), rows, c.n_ff_shexp, &mut w.sa)?;
        m.shd.run(q, s, &w.sa, w.sh.ptr(), false)?;
        if !skip.contains("moe") {
            q.moe_gu_q8(s, m.ty_gu, m.gtab.ptr(), m.utab.ptr(), m.ty_gu.row_bytes(d), m.ids.ptr(), rows, topk, &w.ea,
                c.n_ff_exp, &mut w.ha, None, None)?;
            q.moe_down(s, m.ty_d, m.dtab.ptr(), m.ty_d.row_bytes(c.n_ff_exp), m.ids.ptr(), rows, topk, &w.ha, d, w.ey.ptr(), None)?;
        }
        s.launch(&fk.moe_fin, (g256(d), rows as u32), 256, 0, (w.res.ptr(), w.ey.ptr(), self.sh.y.dev(), m.ids.ptr(),
            m.dtab.ptr(), m.wts.ptr(), w.sh.ptr(), m.rl.ptr().add::<f32>(m.n_exp), ldr as u32, m.injb.ptr(), c.hc as u32,
            d as u32, c.hc as u32, topk as u32, DevPtr::NULL))?;
        // the next draft reads this residual as its h
        s.memcpy_dtod(w.hsave.ptr(), w.res.ptr(), rows * hd * 4)?;
        qhc_mix(kn, fk, q, s, w, &m.head, DevPtr::NULL, &c, rows)?;
        if self.tp {
            // vocab-split MTP head: the head row to card 0 (LL pairs, kofs 255), this card's half here
            let half = c.n_vocab / 2;
            let last = w.mixed.ptr().add::<f32>((rows - 1) * d);
            s.launch(&fk.ll_pack, g256(d), 256, 0, (w.llp.ptr(), last, d as u32, w.prm.ptr().add::<u32>(P_EPOCH), 255u32))?;
            s.memcpy_dtoh(std::slice::from_raw_parts_mut(self.mtp_h.as_ptr() as *mut u32, 2 * d), w.llp.ptr())?;
            q.quantize_q8_1(s, last, d, false, 1, d, &mut w.xa)?;
            self.output.run(q, s, &w.xa, w.logits.ptr(), false)?;
            argmax_rows(fk, s, w, w.logits.ptr(), w.tok.ptr(), 1, c.n_vocab - half, w.pmx.ptr())?;
            s.launch(&fk.argmax_tv, 1, 32, 0, (w.tv.ptr(), w.tok.ptr(), w.logits.ptr(), (c.n_vocab - half) as u32, half as u32,
                1u32, w.pmx.ptr()))?;
            s.memcpy_dtoh(std::slice::from_raw_parts_mut((self.tv_h.as_ptr() as *mut u32).add(MAX_ROWS * 3), 3), w.tv.ptr())?;
            return Ok(());
        }
        q.quantize_q8_1(s, w.mixed.ptr().add::<f32>((rows - 1) * d), d, false, 1, d, &mut w.xa)?;
        if !skip.contains("head") {
            self.output.run(q, s, &w.xa, w.logits.ptr(), false)?;
        }
        argmax_rows(fk, s, w, w.logits.ptr(), w.tok.ptr(), 1, c.n_vocab, w.tok.ptr().add::<u32>(1))?;
        s.memcpy_dtoh(&mut self.tok_h.as_mut_slice()[..2], w.tok.ptr())?;
        Ok(())
    }

    /// FNEXT_TP=2, card 0: its half of the MTP head for the `rows` head rows card 1 sends (LL slot mtp_h).
    unsafe fn enqueue_mtp_head_tp(&mut self, rows: usize) -> Result<()> {
        let c = self.cfull.clone();
        let d = c.n_embd;
        let half = c.n_vocab / 2;
        let (fk, q, s) = (&self.fk, &self.q, &self.s);
        let w = &mut self.w;
        let (_, ow) = self.head_tp.as_ref().ok_or_else(|| anyhow!("MTP head on the wrong card"))?;
        s.memcpy_htod(w.prm.ptr(), &self.prm_h.as_slice()[..P_WORDS])?;
        s.launch(&fk.ll_get, g256(rows * d), 256, 0, (w.mixed.ptr(), self.mtp_h.dev(), (rows * d) as u32, w.prm.ptr(), 255u32))?;
        q.quantize_q8_1(s, w.mixed.ptr(), d, false, rows, d, &mut w.xa)?;
        ow.run(q, s, &w.xa, w.logits.ptr(), false)?;
        argmax_rows(fk, s, w, w.logits.ptr(), w.tok.ptr(), rows, half, w.pmx.ptr())?;
        s.launch(&fk.argmax_tv, 1, 32, 0, (w.tv.ptr(), w.tok.ptr(), w.logits.ptr(), half as u32, 0u32, rows as u32, w.pmx.ptr()))?;
        s.memcpy_dtoh(std::slice::from_raw_parts_mut(self.tv_h.as_ptr() as *mut u32, 3 * rows), w.tv.ptr())?;
        Ok(())
    }

    /// FNEXT_TP=2: card 0's MTP head graph for `rows` head rows (captured once).
    fn ensure_mtp_head_tp(&mut self, rows: usize) -> Result<()> {
        if self.other.as_ref().unwrap().mtp_head_g[rows].is_none() {
            self.swap_dev()?;
            let g = self.capture(|m| unsafe { m.enqueue_mtp_head_tp(rows) });
            if let Ok(g) = g {
                self.mtp_head_g[rows] = Some(g);
            }
            self.swap_dev()?;
            if self.other.as_ref().unwrap().mtp_head_g[rows].is_none() {
                bail!("FNEXT_TP=2: card 0 MTP head graph capture failed");
            }
        }
        Ok(())
    }

    /// FNEXT_TP=2: row r's greedy token over both vocabulary halves (the larger logit; card 0 on ties: the lower
    /// index) and its softmax probability over the whole vocabulary from the halves' (max, probability) pairs.
    fn tp_pick(&self, r: usize) -> (u32, f32) {
        let tv = self.tv_h.as_slice();
        let (a, b) = (&tv[3 * r..3 * r + 3], &tv[MAX_ROWS * 3 + 3 * r..MAX_ROWS * 3 + 3 * r + 3]);
        let (ma, mb) = (f32::from_bits(a[1]), f32::from_bits(b[1]));
        let (pa, pb) = (f32::from_bits(a[2]), f32::from_bits(b[2]));
        let m = ma.max(mb);
        // each half's sum of exp(logit - its max) is 1 / its probability
        let tot = (ma - m).exp() / pa + (mb - m).exp() / pb;
        (if mb > ma { b[0] } else { a[0] }, 1.0 / tot)
    }

    /// FNEXT_TP=2: are row r's best logit and its probability finite on both vocabulary halves? (A NaN anywhere in a
    /// half makes that half's probability a NaN; an infinity shows in its maximum.)
    fn tp_row_finite(&self, r: usize) -> bool {
        let tv = self.tv_h.as_slice();
        [&tv[3 * r..3 * r + 3], &tv[MAX_ROWS * 3 + 3 * r..MAX_ROWS * 3 + 3 * r + 3]].iter()
            .all(|h| f32::from_bits(h[1]).is_finite() && f32::from_bits(h[2]).is_finite())
    }

    /// A new draft for `tok` as the next token of the current lane (the token at position tokens.len(), not yet run
    /// through the trunk), replacing the draft the MTP block made for another token there: the block's row of that
    /// position is run again with `tok` and the trunk residual of the position before it. For a first token the
    /// server replaced, and for a lane that returns to drafting after steps through [`Fast::prefill_mtp`].
    pub fn mtp_draft_for(&mut self, tok: u32) -> Result<u32> {
        let hd = self.c.hc_dim();
        let pos = self.tokens.len();
        let m = self.mtp.as_ref().ok_or_else(|| anyhow!("mtp_draft_for without the MTP block"))?;
        self.s.memcpy_dtod(m.hin.ptr(), m.hprev.ptr(), hd * 4)?;
        self.mtp_run(&[tok], pos)
    }

    /// One MTP pass: tokens at positions pos0.. with h rows already in mtp.hin; returns the draft after the last row.
    fn mtp_run(&mut self, toks: &[u32], pos0: usize) -> Result<u32> {
        if self.tp {
            return self.mtp_run_tp(toks, pos0);
        }
        let rows = toks.len();
        let t0 = std::time::Instant::now();
        self.ensure_ctx(pos0 + rows + MAX_ROWS)?;
        self.fill_prm(pos0, rows);
        self.fill_emb(toks);
        self.mtp_ms[0] += t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        if self.mtp_graphs[rows].is_none() {
            let g = self.capture(|m| unsafe { m.enqueue_mtp(rows) })?;
            self.mtp_graphs[rows] = Some(g);
        }
        self.mtp_graphs[rows].as_ref().unwrap().launch(&self.s)?;
        if std::env::var("FNEXT_MTP_TWICE").is_ok() {
            self.s.sync()?;
            self.mtp_ms[0] += t1.elapsed().as_secs_f64() * 1e3;
            self.mtp_graphs[rows].as_ref().unwrap().launch(&self.s)?;
        }
        self.s.sync()?;
        self.mtp_ms[1] += t1.elapsed().as_secs_f64() * 1e3;
        self.mtp_ms[2] += 1.0;
        if std::env::var("FNEXT_MTP_COUNT").is_ok() {
            let m = self.mtp.as_ref().unwrap();
            let mut ids = vec![0i32; rows * self.c.n_expert_used];
            fe_cuda::memcpy_dtoh(&mut ids, m.ids.ptr())?;
            if self.mtp_counts.is_empty() {
                self.mtp_counts = vec![0; m.n_exp];
            }
            for &e in &ids {
                self.mtp_counts[e as usize] += 1;
            }
        }
        self.draft_p = f32::from_bits(self.tok_h.as_slice()[1]);
        Ok(self.tok_h.as_slice()[0])
    }

    /// mtp_run under FNEXT_TP=2: the MTP block on card 1, its head row split with card 0 (both launched together).
    fn mtp_run_tp(&mut self, toks: &[u32], pos0: usize) -> Result<u32> {
        let rows = toks.len();
        let t0 = std::time::Instant::now();
        self.ensure_ctx(pos0 + rows + MAX_ROWS)?;
        self.fill_prm(pos0, rows);
        self.fill_emb(toks);
        self.epoch = self.epoch.wrapping_add(1);
        self.prm_h.as_mut_slice()[P_EPOCH] = self.epoch << 8;
        self.mtp_ms[0] += t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        if self.mtp_graphs[rows].is_none() {
            let g = self.capture(|m| unsafe { m.enqueue_mtp(rows) })?;
            self.mtp_graphs[rows] = Some(g);
        }
        self.ensure_mtp_head_tp(1)?;
        let o = self.other.as_ref().unwrap();
        unsafe { self.launcher.as_ref().unwrap().post(o.mtp_head_g[1].as_ref().unwrap(), &o.s, &o.done_ev) };
        let r = self.mtp_graphs[rows].as_ref().unwrap().launch(&self.s);
        self.launcher.as_ref().unwrap().wait()?;
        r?;
        self.s.sync()?;
        self.other.as_ref().unwrap().done_ev.sync()?;
        self.mtp_ms[1] += t1.elapsed().as_secs_f64() * 1e3;
        self.mtp_ms[2] += 1.0;
        let (t, p) = self.tp_pick(0);
        self.tok_h.as_mut_slice()[0] = t;
        self.tok_h.as_mut_slice()[1] = p.to_bits();
        self.draft_p = p;
        Ok(t)
    }

    /// Prompt with the MTP block kept in step: each trunk step's rows also run through MTP (position q reads token q
    /// and the trunk residual of q - 1; position 0 a zero residual). Returns (greedy next token, first draft).
    pub fn prefill_mtp(&mut self, toks: &[u32]) -> Result<(u32, u32)> {
        let hd = self.c.hc_dim();
        if self.long_prompt(toks.len()) {
            self.ensure_pw(toks.len())?;
            let r = self.prefill_mtp_long(toks);
            self.release_pw()?;
            return r;
        }
        let mut t = 0;
        for ch in toks.chunks(MAX_ROWS) {
            let pos0 = self.tokens.len();
            let r = ch.len();
            {
                let m = self.mtp.as_ref().unwrap();
                if pos0 == 0 {
                    self.s.memset_d8(m.hprev.ptr(), 0, hd * 4)?;
                }
                // hin = [h of pos0 - 1, then this step's rows 0 .. r-1]
                self.s.memcpy_dtod(m.hin.ptr(), m.hprev.ptr(), hd * 4)?;
            }
            t = self.step(ch)?; // also sets hprev to the last row
            let m = self.mtp.as_ref().unwrap();
            if r > 1 {
                self.s.memcpy_dtod(m.hin.ptr().add::<f32>(hd), self.w.hsave.ptr(), (r - 1) * hd * 4)?;
            }
            self.mtp_run(ch, pos0)?;
        }
        let pos = self.tokens.len();
        let m = self.mtp.as_ref().unwrap();
        self.s.memcpy_dtod(m.hin.ptr(), m.hprev.ptr(), hd * 4)?;
        let d1 = self.mtp_run(&[t], pos)?;
        Ok((t, d1))
    }

    fn prefill_mtp_long(&mut self, toks: &[u32]) -> Result<(u32, u32)> {
        let hd = self.c.hc_dim();
        let gmax = self.pw.as_ref().unwrap().gmax;
        let mut t = 0;
        for grp in toks.chunks(gmax) {
            if self.cancelled() {
                return Err(Cancelled.into());
            }
            let base = self.tokens.len();
            t = self.prefill_group(grp)?;
            if std::env::var("FNEXT_MTP_PROMPT").as_deref() != Ok("rows") {
                // the MTP block's caches for the group's rows in prompt form (row q: token q, trunk residual of q - 1)
                if base == 0 {
                    let m = self.mtp.as_ref().unwrap();
                    self.s.memset_d8(m.hprev.ptr(), 0, hd * 4)?;
                }
                unsafe { self.mtp_group_kv(grp, base)? };
                let m = self.mtp.as_ref().unwrap();
                let hall = self.pw.as_ref().unwrap().hall.ptr();
                self.s.memcpy_dtod(m.hprev.ptr(), hall.add::<f32>((grp.len() - 1) * hd), hd * 4)?;
                continue;
            }
            // the MTP block over the group's rows, 8 at a time: row q reads token q and the trunk residual of q - 1
            let mut a = 0;
            while a < grp.len() {
                let n = (grp.len() - a).min(MAX_ROWS);
                let m = self.mtp.as_ref().unwrap();
                let hall = self.pw.as_ref().unwrap().hall.ptr();
                if base + a == 0 {
                    self.s.memset_d8(m.hprev.ptr(), 0, hd * 4)?;
                }
                self.s.memcpy_dtod(m.hin.ptr(), m.hprev.ptr(), hd * 4)?;
                if n > 1 {
                    self.s.memcpy_dtod(m.hin.ptr().add::<f32>(hd), hall.add::<f32>(a * hd), (n - 1) * hd * 4)?;
                }
                self.s.memcpy_dtod(m.hprev.ptr(), hall.add::<f32>((a + n - 1) * hd), hd * 4)?;
                self.mtp_run(&grp[a..a + n], base + a)?;
                a += n;
            }
        }
        let pos = self.tokens.len();
        let m = self.mtp.as_ref().unwrap();
        self.s.memcpy_dtod(m.hin.ptr(), m.hprev.ptr(), hd * 4)?;
        let d1 = self.mtp_run(&[t], pos)?;
        Ok((t, d1))
    }

    /// One speculative step: draft more (chained MTP while confident), verify, commit, catch the MTP block up.
    /// Returns the tokens it produced (the kept drafts, then the trunk's next token); `sp` moves on.
    pub fn spec_step(&mut self, sp: &mut Spec, n_draft: usize) -> Result<Vec<u32>> {
        self.spec_step_ctl(sp, n_draft, None, &mut |_, _| {})
    }

    /// [`Fast::spec_step`] under the caller's control. `force`: the engine writes this token itself as the next one
    /// (the step then verifies the last token alone and drops the model's choice; the MTP block absorbs the forced
    /// token like any kept row, so the next step drafts again). `keep` sees the tokens the step is about to produce
    /// (and whether the logit rows behind them were finite) and may shorten them, to one at least: the tokens it cuts
    /// are given back, the verify keeps fewer rows and the step ends as if the drafts after them had missed.
    pub fn spec_step_ctl(&mut self, sp: &mut Spec, n_draft: usize, force: Option<u32>,
        keep: &mut dyn FnMut(&mut Vec<u32>, bool)) -> Result<Vec<u32>> {
        let hd = self.c.hc_dim();
        let t0 = std::time::Instant::now();
        // adaptive length: drafts while the recent acceptance rate keeps them likely to pay (FNEXT_ADAPT=0 off)
        let n_draft = if std::env::var("FNEXT_ADAPT").is_ok_and(|v| v == "1") {
            let r = self.acc_ema;
            let n = if r > 0.85 { n_draft } else if r > 0.7 { n_draft.min(3) } else if r > 0.5 { n_draft.min(2) } else { 1 };
            n.max(1)
        } else {
            n_draft
        };
        let mut confident = force.is_none() && self.draft_p >= self.p_min;
        if !confident {
            sp.drafts.clear();
        }
        while sp.drafts.len() < n_draft && confident {
            let pos = self.tokens.len() + sp.drafts.len();
            let m = self.mtp.as_ref().unwrap();
            self.s.memcpy_dtod(m.hin.ptr(), self.w.hsave.ptr(), hd * 4)?;
            let last = *sp.drafts.last().unwrap_or(&sp.cur);
            let nd = self.mtp_run(&[last], pos)?;
            confident = self.draft_p >= self.p_min;
            if confident {
                sp.drafts.push(nd);
                // the verify step will read the PLE rows of [cur, drafts..]: request their pages now
                if std::env::var("FNEXT_PLE_SPEC").map_or(true, |v| v != "0") {
                    let mut rows = vec![sp.cur];
                    rows.extend_from_slice(&sp.drafts);
                    self.ple_willneed(&rows);
                }
            }
        }
        let mut rows = vec![sp.cur];
        rows.extend_from_slice(&sp.drafts);
        let t1 = std::time::Instant::now();
        let pred = self.verify(&rows)?;
        let t2 = std::time::Instant::now();
        let mut k = 0;
        while k < sp.drafts.len() && sp.drafts[k] == pred[k] {
            k += 1;
        }
        self.spec_stats.0 += sp.drafts.len() as u64;
        self.spec_stats.1 += k as u64;
        if !sp.drafts.is_empty() {
            self.acc_ema = 0.8 * self.acc_ema + 0.2 * (k as f32 / sp.drafts.len() as f32);
        }
        let mut out: Vec<u32> = sp.drafts[..k].to_vec();
        out.push(force.unwrap_or(pred[k]));
        let finite = self.bad_rows & ((1u32 << (k + 1)) - 1) == 0;
        let full = out.len();
        keep(&mut out, finite);
        if out.is_empty() || out.len() > full {
            bail!("spec_step: keep left {} of {full} tokens", out.len());
        }
        // tokens given back: the verify keeps fewer rows (out[k] is then the accepted draft of that row)
        let k = out.len() - 1;
        self.commit(&rows, k + 1)?;
        let t3 = std::time::Instant::now();
        let next = out[k];
        // MTP catch-up over the kept positions: tokens [d1..dk, next], h = trunk rows 0..k of the verify
        let pos0 = self.tokens.len() - k;
        let m = self.mtp.as_ref().unwrap();
        self.s.memcpy_dtod(m.hin.ptr(), self.w.hsave.ptr(), (k + 1) * hd * 4)?;
        let d1 = self.mtp_run(&out, pos0)?;
        if k > 0 {
            // the chained drafts read the MTP residual of the last catch-up row
            self.s.memcpy_dtod(self.w.hsave.ptr(), self.w.hsave.ptr().add::<f32>(k * hd), hd * 4)?;
        }
        sp.cur = next;
        sp.drafts = vec![d1];
        let t4 = std::time::Instant::now();
        let st = &mut self.spec_ms;
        st[0] += (t1 - t0).as_secs_f64() * 1e3;
        st[1] += (t2 - t1).as_secs_f64() * 1e3;
        st[2] += (t3 - t2).as_secs_f64() * 1e3;
        st[3] += (t4 - t3).as_secs_f64() * 1e3;
        st[4] += 1.0;
        Ok(out)
    }

    /// Greedy generation with MTP drafts: `n_draft` drafts per verify. Returns the generated tokens (`first` included).
    pub fn generate_spec(&mut self, first: u32, draft1: u32, n: usize, n_draft: usize, stop: &[u32]) -> Result<Vec<u32>> {
        let mut out = vec![first];
        let mut sp = Spec { cur: first, drafts: vec![draft1] };
        while out.len() < n {
            let got = self.spec_step(&mut sp, n_draft)?;
            let hit = got.iter().any(|t| stop.contains(t));
            out.extend(got);
            if hit {
                break;
            }
        }
        out.truncate(n);
        Ok(out)
    }
}

/// The PLE table rows `idx` (heads per row) dequantized into rows row0.. of `ple` (then dropped from the page cache).
fn ple_gather_rows(g: &Gguf, hm: &hostmem::HostMem, idx: &[u32], ple: &mut [f32], d: usize, heads: usize, row0: usize) {
    let rows = idx.len() / heads.max(1);
    let pt = g.tensor("per_layer_token_embd.weight").unwrap();
    let (prb, pdata, dd) = (pt.row_bytes(), g.data(pt), pt.ne0());
    // decode-sized gathers stay on the calling thread: waking rayon workers costs more than the work, and they land
    // on CPUs where the launch threads and the spinning CPU expert pool run
    let out = &mut ple[row0 * d..(row0 + rows) * d];
    if idx.len() <= 512 {
        out.chunks_mut(dd).zip(idx.iter()).for_each(|(o, &rw)| {
            fe_ggq::refq::dequant(pt.ty, &pdata[rw as usize * prb..(rw as usize + 1) * prb], o);
        });
    } else {
        out.par_chunks_mut(dd).zip(idx.par_iter()).for_each(|(o, &rw)| {
            fe_ggq::refq::dequant(pt.ty, &pdata[rw as usize * prb..(rw as usize + 1) * prb], o);
        });
    }
    if std::env::var("FNEXT_PLE_KEEP").is_err() {
        for &rw in idx {
            hm.drop_range(g, pt, rw as usize * prb, prb);
        }
    }
}

/// Whether a step gathers its PLE rows while its graph runs (the graph waits for the flag) or before it launches.
fn ple_late_tp(tp: bool) -> bool {
    if tp { std::env::var("FNEXT_PLE_LATE").is_ok_and(|v| v == "1") } else { ple_late() }
}

impl Fast {
    fn ple_late_now(&self) -> bool {
        ple_late_tp(self.tp)
    }
}

/// FNEXT_PLE_LATE=0: gather the PLE rows before launching a step (instead of while layer 0 runs).
fn ple_late() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FNEXT_PLE_LATE").map_or(true, |v| v != "0"))
}

/// Blocks per logit row of the two-stage argmax.
const ARGMAX_PARTS: usize = 128;

/// Greedy token of each of `rows` logit rows (lowest index on ties) into out[r]; pmax (optional): its softmax
/// probability. Two stages (a 1024-thread block per 248K-wide row was ~75 us); FNEXT_ARGMAX1=1 = the one-block kernel.
unsafe fn argmax_rows(fk: &FKern, s: &Stream, w: &Work, logits: DevPtr, out: DevPtr, rows: usize, n: usize, pmax: DevPtr)
    -> Result<()> {
    if std::env::var("FNEXT_ARGMAX1").is_ok_and(|v| v == "1") {
        s.launch(&fk.argmax, rows as u32, 1024, 0, (out, logits, n as u32, pmax))?;
    } else {
        s.launch(&fk.argmax_part, (ARGMAX_PARTS as u32, rows as u32), 256, 0, (w.amx.ptr(), logits, n as u32))?;
        s.launch(&fk.argmax_fin, rows as u32, 32, 0, (out, w.amx.ptr(), ARGMAX_PARTS as u32, pmax))?;
    }
    Ok(())
}

/// FNEXT_HC_PF=0: hc_down does not prefetch hc_up's weights (A/B switch; same bits either way).
fn hc_pf() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FNEXT_HC_PF").map_or(true, |v| v != "0"))
}

/// FNEXT_HC_FUSED=0: decode hc mixes as norm + two GEMVs + elementwise kernels instead of quant's fused pair.
fn hc_fused() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FNEXT_HC_FUSED").map_or(true, |v| v != "0"))
}

/// A prompt-cache snapshot: the non-positional state after `pos` tokens (Fast::snap_save / snap_restore).
pub struct Snap {
    pub pos: usize,
    bufs: Vec<HostBuffer<u8>>,
}

impl Snap {
    pub fn bytes(&self) -> usize {
        self.bufs.iter().map(|b| b.len()).sum()
    }
}

/// Pinned snapshot buffers of dropped snapshots, reused: a fresh cuMemHostAlloc of a ~100 MB snapshot cost ~85 ms per
/// card at every prompt-cache cut (and its cuMemFreeHost again on eviction), most of a short request's TTFT. Lanes keep
/// up to three snapshots each, so the pool is prewarmed for all of them (memory permitting) and, when a take leaves no
/// spare of that size, a background thread allocates the next one off the request path.
static SNAP_POOL: std::sync::Mutex<Vec<HostBuffer<u8>>> = std::sync::Mutex::new(Vec::new());
/// set while a prompt is prefilled (the snapshot allocator waits for it)
static PREFILL_BUSY: AtomicBool = AtomicBool::new(false);
static SNAP_CAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(16);
static SNAP_REFILL: std::sync::Mutex<Option<std::sync::mpsc::Sender<usize>>> = std::sync::Mutex::new(None);

/// (buffers, bytes) of free snapshot buffers in the pool (memory reporting).
pub fn snap_pool_bytes() -> (usize, usize) {
    let p = SNAP_POOL.lock().unwrap();
    (p.len(), p.iter().map(|b| b.len()).sum())
}

fn snap_take(n: usize) -> Result<HostBuffer<u8>> {
    let mut p = SNAP_POOL.lock().unwrap();
    if let Some(i) = p.iter().position(|b| b.len() == n) {
        let b = p.swap_remove(i);
        drop(p);
        // the allocator tops the spares up between prompts
        if let Some(tx) = SNAP_REFILL.lock().unwrap().as_ref() {
            let _ = tx.send(n);
        }
        return Ok(b);
    }
    drop(p);
    if let Some(tx) = SNAP_REFILL.lock().unwrap().as_ref() {
        let _ = tx.send(n);
    }
    Ok(HostBuffer::<u8>::alloc(n)?)
}

impl Drop for Snap {
    fn drop(&mut self) {
        let mut p = SNAP_POOL.lock().unwrap();
        p.extend(self.bufs.drain(..));
        let cap = SNAP_CAP.load(Ordering::Relaxed);
        let n = p.len();
        if n > cap {
            p.drain(..n - cap);
        }
    }
}

/// Host memory this process may use: MemTotal, or its cgroup's memory.max if lower.
fn host_mem_bytes() -> usize {
    let total = std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| s.lines().find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok())).map_or(usize::MAX, |k| k << 10);
    let cg = std::fs::read_to_string("/proc/self/cgroup").ok().and_then(|s| s.lines().next().map(|l| l.rsplit(':').next().unwrap_or("").to_string()))
        .and_then(|p| std::fs::read_to_string(format!("/sys/fs/cgroup{p}/memory.max")).ok())
        .and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(usize::MAX);
    total.min(cg)
}

/// Pre-fill the snapshot pool (`per` snapshots per card) and, with `refill`, start the background allocator.
pub(crate) fn prewarm_snaps(sizes: &[usize], per: usize, cu: Arc<fe_cuda::Context>, refill: bool) -> Result<()> {
    let mut v = vec![];
    for &n in sizes {
        for _ in 0..per {
            v.push(HostBuffer::<u8>::alloc(n)?);
        }
    }
    SNAP_CAP.store((sizes.len() * (per + 6)).max(8), Ordering::Relaxed);
    SNAP_POOL.lock().unwrap().extend(v);
    if refill && std::env::var("FNEXT_SNAP_REFILL").as_deref() != Ok("0") {
        let (tx, rx) = std::sync::mpsc::channel::<usize>();
        *SNAP_REFILL.lock().unwrap() = Some(tx);
        std::thread::Builder::new().name("fnext-snap-alloc".into()).spawn(move || {
            crate::cores::unpin_current();
            if cu.bind().is_err() {
                return;
            }
            // keep SPARE free buffers of every size the snapshots use (parked conversations hold theirs), allocated
            // between prompts: a cuMemHostAlloc beside a prefill or snapshot copy holds it up (the driver serializes)
            const SPARE: usize = 4;
            let mut want: Vec<usize> = vec![];
            while let Ok(n) = rx.recv() {
                if !want.contains(&n) {
                    want.push(n);
                }
                loop {
                    let Some(&n) = want.iter().find(|&&n| SNAP_POOL.lock().unwrap().iter().filter(|b| b.len() == n).count() < SPARE)
                    else { break };
                    while PREFILL_BUSY.load(Ordering::Relaxed) {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    match HostBuffer::<u8>::alloc(n) {
                        Ok(b) => SNAP_POOL.lock().unwrap().push(b),
                        Err(_) => break,
                    }
                }
            }
        })?;
    }
    Ok(())
}

/// Debug (FNEXT_DUMP): sync, then the bits hash of each of `rows` rows of `n` f32 at `p`.
unsafe fn dump_rows(s: &Stream, out: &mut Vec<(usize, usize, Vec<u64>)>, l: usize, stage: usize, p: DevPtr, rows: usize,
    n: usize) -> Result<()> {
    s.sync()?;
    let mut v = vec![0f32; rows * n];
    fe_cuda::memcpy_dtoh(&mut v, p)?;
    let hs = v.chunks(n).map(|r| {
        let mut h = 0xcbf29ce484222325u64;
        for x in r {
            h ^= x.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }).collect();
    out.push((l, stage, hs));
    Ok(())
}

/// The prompt workspace shape of an n-token prompt at position `pos`: (chunk, pctx, group, gtok).
fn pw_shape(pcfg: (usize, usize, usize), ctx: usize, n_expert: usize, pos: usize, n: usize) -> (usize, usize, usize, usize) {
    // sized to the prompt: a short one fits in the free reserve without releasing LRU blocks
    let (chunk, group, gtok) = pcfg;
    // bigger chunks for long prompts: the expert GEMMs see more rows per expert (FNEXT_CHUNK pins it)
    let chunk = if std::env::var("FNEXT_CHUNK").is_ok() { chunk } else if n >= 8192 { PROMPT_CHUNK_MAX } else if n >= 2048 { 2048 } else { chunk };
    // the indexer key ring holds a chunk plus the pooling tail
    let chunk = chunk.min(n.next_power_of_two().max(64)).min(ring_cells() - 8);
    // indexer scores only up to where this prompt ends
    let pctx = (pos + n + MAX_ROWS).next_multiple_of(8192).min(ctx);
    let gtok = gtok.min(n.next_multiple_of(chunk));
    // long prompts prefetch whole layers (arena = every expert of a layer); short ones stream groups
    let group = if n >= 2048 && std::env::var("FNEXT_NO_PREFETCH").is_err() { n_expert } else if n < 1024 { group.min(16) } else { group };
    (chunk, pctx, group, gtok)
}

/// Load-time probes of the prompt workspace sizes (card bytes per shape class, at position 0), taken while the card is
/// still free: the first prompt of a new size then releases the right number of LRU blocks in one go instead of
/// failing allocations (~15-250 ms each) and releasing 256 MiB steps. FNEXT_PW_PROBE=0: off.
fn probe_pw(c: &Config, g: &Gguf, pcfg: (usize, usize, usize), ctx: usize, tp: bool)
    -> Result<std::collections::HashMap<(usize, usize, usize, usize), usize>> {
    let mut m = std::collections::HashMap::new();
    if std::env::var("FNEXT_PW_PROBE").as_deref() == Ok("0") || pcfg.0 <= MAX_ROWS {
        return Ok(m);
    }
    prefill::set_probe(true);
    for n in [64usize, 128, 256, 512, 1000, 1024, 2047, 2048, 4096, 8191, 8192, 12288, 16384] {
        let key = pw_shape(pcfg, ctx, c.n_expert, 0, n);
        if m.contains_key(&key) {
            continue;
        }
        let f0 = fe_cuda::mem_info()?.0;
        let pw = prefill::PWork::alloc(c, g, key.0, key.1, key.2, key.3, tp);
        let took = f0.saturating_sub(fe_cuda::mem_info()?.0);
        if let Ok(pw) = pw {
            m.insert(key, took);
            drop(pw);
        }
    }
    prefill::set_probe(false);
    Ok(m)
}
