//! M3: the one-card DeepSeek-V4.1-Flash Coder decode engine. One row at a time (prompt tokens are fed one by one):
//! dense weights stay in their GGUF quantisation on the card (Q5_K / Q8_0 through fe-ggq's q8_1 GEMVs, BF16 through
//! gemv_bf16), the k-ds kernels do hc mixes / norms / rope / MQA attention / compressor / indexer / router, the routed
//! experts run on the CPU lane (fe-ggq CpuMoe, all experts host-side), the Engram rows come from pread.
//! Semantics follow the ds-model forward (and the f32 oracle in refmodel.rs); activations between k-ds kernels are bf16.
use crate::config::{DscConfig, LayerRole};
use crate::engram::Engram;
use crate::refmodel::rope_table;
use crate::{K_DS_CUBIN, K_ENGRAM_CUBIN};
use anyhow::{anyhow, bail, ensure, Context as _, Result};
use fe_cuda::{memcpy_dtoh, memcpy_htod, Context, DevPtr, DeviceBuffer, Function, HostBuffer, Module, Stream};
use fe_ggq::cpu::{CpuMoe, MoeJob};
use fe_ggq::gpu::{Ggq, Q8Act};
use fe_ggq::refq;
use fe_gguf::{GgmlType, Gguf};
use std::time::{Duration, Instant};

/// VRAM kept free on card `k` (0 = main, 1.. = expert cards) when the resident experts are sized, in GiB: DSC_RESERVE_GBS=<main>,<card 1>,<card 2>
/// (may be negative on the drafter's card: its fixed 8.8 GiB allowance is more than the drafter takes), else DSC_RESERVE_GB for every card, default 0.5.
pub fn reserve_gb(k: usize) -> f64 {
    if let Some(v) = std::env::var("DSC_RESERVE_GBS").ok().and_then(|v| v.split(',').nth(k).and_then(|p| p.trim().parse::<f64>().ok())) {
        return v;
    }
    std::env::var("DSC_RESERVE_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.5)
}

/// diagnostic: minor faults taken inside `CpuMoe::run` (DSC_LANEFAULTS=1)
pub static LANE_FAULTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub mod draft;
pub mod pipe;
pub mod spec;

/// Rows a step may carry (verify rows / lanes): every row-count form of the kernels below keeps each row's bits.
pub const RMAX: usize = 32;

/// Rows of a prompt-feed pass (DSC_PREFILL_ROWS, 1..=RMAX, default 8: the row counts gated bit-identical).
pub fn prefill_rows() -> usize {
    std::env::var("DSC_PREFILL_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize).clamp(1, RMAX)
}

/// A dense tensor on the card in its GGUF form: `n` rows of `k` values, `rb` bytes a row.
pub struct Wt {
    pub ty: GgmlType,
    pub n: usize,
    pub k: usize,
    pub rb: usize,
    pub buf: DeviceBuffer<u8>,
}

impl Wt {
    fn load(g: &Gguf, name: &str) -> Result<Wt> {
        let t = g.tensor(name).ok_or_else(|| anyhow!("tensor {name} missing"))?;
        let n = t.dims.get(1).copied().unwrap_or(1) as usize;
        Ok(Wt { ty: t.ty, n, k: t.ne0(), rb: t.row_bytes(), buf: DeviceBuffer::from_slice(g.data(t)).with_context(|| name.to_string())? })
    }
    fn ptr(&self) -> DevPtr {
        self.buf.ptr()
    }
}

/// An F32 tensor of the file as bf16 on the card (the k-ds norms take bf16 weights; the file's values came from bf16).
fn norm_bf16(g: &Gguf, name: &str) -> Result<DeviceBuffer<u16>> {
    let t = g.tensor(name).ok_or_else(|| anyhow!("tensor {name} missing"))?;
    ensure!(t.ty == GgmlType::F32, "{name}: {} not F32", t.ty.name());
    let v: Vec<u16> = g.data(t).chunks_exact(4).map(|b| refq::f32_to_bf16(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))).collect();
    Ok(DeviceBuffer::from_slice(&v)?)
}

fn raw(g: &Gguf, name: &str) -> Result<DeviceBuffer<u8>> {
    let t = g.tensor(name).ok_or_else(|| anyhow!("tensor {name} missing"))?;
    Ok(DeviceBuffer::from_slice(g.data(t))?)
}

struct Hc {
    fnw: DeviceBuffer<u8>,
    scale: DeviceBuffer<u8>,
    base: DeviceBuffer<u8>,
}

struct Comp {
    wkv: Wt,
    wgate: Option<Wt>,
    norm: DeviceBuffer<u16>,
}

struct Idx {
    wq_b: Wt,
    proj: Wt,
    wk: Option<Wt>,
    k_norm: Option<DeviceBuffer<u16>>,
}

struct Eng {
    kv: Wt,
    qk: DeviceBuffer<u8>,
}

struct Layer {
    role: LayerRole,
    attn_norm: DeviceBuffer<u16>,
    q_a: Wt,
    q_a_norm: DeviceBuffer<u16>,
    q_b: Wt,
    kv: Wt,
    kv_norm: DeviceBuffer<u16>,
    sinks: DeviceBuffer<u8>,
    wo_a: Wt,
    wo_b: Wt,
    hc_attn: Hc,
    hc_ffn: Hc,
    ffn_norm: DeviceBuffer<u16>,
    gate_w: Wt,
    gate_b: DeviceBuffer<u8>,
    sh_gate: Wt,
    sh_up: Wt,
    sh_down: Wt,
    comp: Option<Comp>,
    idx: Option<Idx>,
    eng: Option<Eng>,
}

/// GPU-resident experts of one layer: device address tables (0 = not on the card) into the arena.
struct LayerRes {
    gtab: DeviceBuffer<u64>,
    utab: DeviceBuffer<u64>,
    dtab: DeviceBuffer<u64>,
    present: Vec<bool>,
    ty_gu: GgmlType,
    ty_d: GgmlType,
    rb_gu: usize,
    rb_d: usize,
}

/// The expert-only card: its own context, stream and kernels; holds resident experts and runs their pairs.
pub struct StageBuf {
    buf: DeviceBuffer<u8>,
    counter: DeviceBuffer<u32>,
    slot_e: DeviceBuffer<u32>,
    orig: DeviceBuffer<u64>,
    nslots: usize,
    slot_bytes: usize,
}

struct XCard {
    stage: Option<StageBuf>,
    ctx: std::sync::Arc<Context>,
    s: Stream,
    ggq: Ggq,
    act: Q8Act,
    hq: Q8Act,
    x: DeviceBuffer<u16>,
    ids: DeviceBuffer<u32>,
    y6: DeviceBuffer<f32>,
    arena: Option<DeviceBuffer<u8>>,
    res: Vec<Option<LayerRes>>,
    budget: f64,
    /// recorded on `s` after this card's results were copied into the main card's `ycpu` region
    ev: fe_cuda::Event,
    /// diagnostics (DSC_CARDTIME): timing events around this card's work of a layer
    ev_a: fe_cuda::Event,
    ev_b: fe_cuda::Event,
    /// pinned staging for the per-layer async uploads (x rows, expert ids)
    x_h: HostBuffer<u16>,
    ids_h: HostBuffer<u32>,
    pf: Option<prefill::PfCard>,
}

/// Host copy of the experts the cards do not hold: anonymous transparent-huge-page memory (the page cache's scattered 4 KiB
/// pages defeat the hardware prefetcher), NUMA-interleaved over both nodes.
struct HostRegion {
    ptr: *mut u8,
    len: usize,
    /// per (layer * n_expert + e): byte offsets of gate / up / down, usize::MAX = not copied
    off: Vec<[usize; 3]>,
    /// plane-repacked experts (DSC_REPACK): per layer the repacked (gate/up, down) types and the bytes of gate / up / down of one expert; empty = the GGUF's own
    rty: Vec<(GgmlType, GgmlType)>,
    rsz: Vec<[usize; 3]>,
    /// cuMemHostRegister'ed (DSC_PF_REG): the big-prompt path uploads straight from it
    registered: bool,
}
unsafe impl Send for HostRegion {}

struct Kern {
    _m: Module,
    _me: Module,
    hc_mix: Function,
    copy_rows: Function,
    tap: Function,
    hc_part_pn: Function,
    hc_fin: Function,
    hc_pre_norm: Function,
    hc_post: Function,
    rmsnorm: Function,
    rope: Function,
    kv_norm_window: Function,
    comp_stash: Function,
    comp_pair: Function,
    comp_store: Function,
    ikey_store: Function,
    idx_query: Function,
    idx_score: Function,
    idx_score_tc: Function,
    block_max: Function,
    topk: Function,
    sel_to_mask: Function,
    attn16: Function,
    attn_mq: Function,
    attn_mq_fin: Function,
    router_pick: Function,
    swiglu: Function,
    b2f: Function,
    round_scale: Function,
    egate: Function,
}

impl Kern {
    fn load() -> Result<Kern> {
        let m = match std::env::var("FE_DS_CUBIN") {
            Ok(p) => Module::load(&std::fs::read(&p)?)?,
            Err(_) => Module::load(K_DS_CUBIN)?,
        };
        let me = Module::load(K_ENGRAM_CUBIN)?;
        let f = |n: &str| m.function(n);
        Ok(Kern {
            hc_mix: f("ds_hc_mix")?,
            copy_rows: f("ds_copy_rows")?,
            tap: f("ds_tap")?,
            hc_part_pn: f("ds_hc_mix_part_pn")?,
            hc_fin: f("ds_hc_mix_fin")?,
            hc_pre_norm: f("ds_hc_pre_norm")?,
            hc_post: f("ds_hc_post")?,
            rmsnorm: f("ds_rmsnorm")?,
            rope: f("ds_rope")?,
            kv_norm_window: f("ds_kv_norm_window")?,
            comp_stash: f("ds_comp_stash")?,
            comp_pair: f("ds_comp_pair")?,
            comp_store: f("ds_comp_store")?,
            ikey_store: f("ds_ikey_store")?,
            idx_query: f("ds_idx_query")?,
            idx_score: f("ds_idx_score")?,
            idx_score_tc: f("ds_idx_score_tc")?,
            block_max: f("ds_block_max")?,
            topk: f("ds_topk")?,
            sel_to_mask: f("ds_sel_to_mask")?,
            attn16: f("ds_attn16")?,
            attn_mq: f("ds_attn_mq")?,
            attn_mq_fin: f("ds_attn_mq_fin")?,
            router_pick: f("ds_router_pick")?,
            swiglu: f("ds_swiglu")?,
            b2f: f("ds_bf16_to_f32")?,
            round_scale: f("ds_round_scale_f32")?,
            egate: me.function("engram_gate")?,
            _m: m,
            _me: me,
        })
    }
}

/// Where the time of a token goes (milliseconds, summed over layers).
#[derive(Default, Clone, Debug)]
pub struct Split {
    pub hit: u32,
    /// per-layer phases summed over the token (ms): attention path to ids on the host, enqueue of the card work, CpuMoe::run, tail (card-0 wait, combine, post)
    pub ph: [f64; 10],
    pub hit0: u32,
    /// distinct experts the CPU lane ran (summed over layers)
    pub distinct: u32,
    pub miss: u32,
    pub gpu_ms: f64,
    pub cpu_ms: f64,
    pub engram_ms: f64,
    pub total_ms: f64,
}

/// One layer between `layer_pre` and `layer_post`.
pub(crate) struct Pend<'j> {
    l: usize,
    pos: usize,
    nr: usize,
    np: usize,
    xh: Vec<f32>,
    /// per pair: 0 = not on an expert card, k + 1 = resident on expert card k
    onx: Vec<u8>,
    on1: Vec<bool>,
    n0: usize,
    pair_job: Vec<(usize, usize)>,
    jobs: Vec<MoeJob<'j>>,
    limit: f32,
    t_layer: Instant,
}

pub struct Engine<'a> {
    pub g: &'a Gguf,
    pub c: &'a DscConfig,
    _ctx: std::sync::Arc<Context>,
    s: Stream,
    ggq: Ggq,
    k: Kern,
    layers: Vec<Layer>,
    out_norm: DeviceBuffer<u16>,
    out_w: Wt,
    eng: Engram,
    cpu: CpuMoe,
    act: Q8Act,
    hq: Q8Act,
    y6: DeviceBuffer<f32>,
    ycpu: DeviceBuffer<f32>,
    slot: DeviceBuffer<i32>,
    ug: DeviceBuffer<u16>,
    /// DSpark taps: the stream mean of the INPUT of layers n-3 .. n-1, bf16 [row][3 d] (written by segment A)
    pub taps: DeviceBuffer<u16>,
    /// rows of the current step (1..=RMAX)
    pub nr: usize,
    /// rows the scratch buffers hold (RMAX, or DSC_PF_ROWS for the big-prompt path)
    rmax: usize,
    /// smallest remaining prompt the big path takes (DSC_PF_MIN)
    pub pf_min: usize,
    pf_mmq: bool,
    pf_mq: bool,
    pf_fullbp: bool,
    pmq_o: DeviceBuffer<f32>,
    pmq_ml: DeviceBuffer<f32>,
    /// rows of one big chunk (DSC_PF_CHUNK >= rmax): attention runs in blocks of rmax rows, the MoE over all rows
    pf_n: usize,
    tap_off: usize,
    no_graph: bool,
    grp_wo: bool,
    depth_bucket: bool,
    stage_on: bool,
    graph_max: usize,
    graph_clears: usize,
    graph_tick: u64,
    graph_use: std::collections::HashMap<(u8, usize, usize, u64), u64>,
    big: Option<prefill::BigStore>,
    pf_idxtc: bool,
    /// big-prompt (GPU layer-major, streamed experts) workspace of this card
    pf: Option<prefill::PfCard>,
    pf_run: Option<prefill::PfRun>,
    arena: Option<DeviceBuffer<u8>>,
    xcs: Vec<XCard>,
    cardtime: bool,
    zc_ev: Vec<Vec<(fe_cuda::Event, fe_cuda::Event)>>,
    zc_used: Vec<Vec<bool>>,
    /// zero-host layer chain: every expert (resident or read from pinned host memory by its card) has an owner card; routing on the cards
    zc: bool,
    /// zero-host mailboxes in pinned host memory: x / ids published by the main card, the expert cards' result rows (read / written by kernels over PCIe)
    cs: Vec<Stream>,
    ev_b: Vec<fe_cuda::Event>,
    ev_c: Vec<fe_cuda::Event>,
    mb_x: Option<HostBuffer<u16>>,
    mb_ids: Option<HostBuffer<u32>>,
    mb_y: Option<HostBuffer<f32>>,
    zc_buf: Option<HostBuffer<u8>>,
    /// on-demand residency of host-backed experts: a slot ring on every expert card, promotion from the per-token ids log
    stage: Option<StageBuf>,
    /// pinned slots for the big path's per-block metadata / Engram vectors (async uploads: no context sync while the cards work)
    /// the main card's resident tables for the big path with a share of the experts dropped (DSC_PF_MAIN_KEEP): the dropped ones are streamed by the expert cards
    pf_hold: bool,
    res_pf: Vec<Option<LayerRes>>,
    xres_pf: Vec<Vec<Option<LayerRes>>>,
    pf_pin: Option<HostBuffer<u8>>,
    pf_meta_h: Option<HostBuffer<u32>>,
    pf_eng_h: Option<HostBuffer<f32>>,
    zc_sz: Vec<[usize; 3]>,
    zc_lo: u64,
    zc_hi: u64,
    zc_off: Vec<[usize; 3]>,
    zc_tot: Vec<usize>,
    owner_h: Vec<u8>,
    zring: Vec<DeviceBuffer<u8>>,
    ring_slot_bytes: usize,
    ring_expert: Vec<Vec<usize>>,
    ring_stamp: Vec<Vec<u64>>,
    ring_of: Vec<u32>,
    ring_clock: u64,
    ring_promo: usize,
    promo_stage: Option<HostBuffer<u64>>,
    ids_log: Option<HostBuffer<u32>>,
    pub zc_hist: Vec<(u32, u16)>,
    /// measurement: picks of every (layer, expert) over the small passes (DSC_HEAT_LOG=<file.cnt> or `heat_on`), the per-pass log (DSC_PASS_LOG)
    pub heat_cnt: Vec<u64>,
    heat_path: Option<String>,
    heat_acc: bool,
    heat_passes: usize,
    pass_log: Option<std::io::BufWriter<std::fs::File>>,
    zc_sync: bool,
    owner: Option<DeviceBuffer<u8>>,
    ev_main: Option<fe_cuda::Event>,
    /// pinned staging: slot table and the CPU lane's result rows (uploaded on the stream, no context sync)
    slot_h: HostBuffer<i32>,
    yc_h: HostBuffer<f32>,
    host: Option<HostRegion>,
    res: Vec<Option<LayerRes>>,
    pub res_on: bool,
    pub resident_experts: usize,
    tab_plain: DeviceBuffer<f32>,
    tab_comp: DeviceBuffer<f32>,
    pub max_ctx: usize,
    ring_len: usize,
    // caches (one slot)
    ring: Vec<DeviceBuffer<u16>>,
    tail: Vec<Option<DeviceBuffer<f32>>>,
    plane: Vec<Option<DeviceBuffer<u16>>>,
    ikeys: Vec<Option<DeviceBuffer<u16>>>,
    entries: Vec<usize>,
    /// lanes (sequence slots): every cache is [slots][...] per layer; the row metadata carries (slot, position)
    n_slots: usize,
    pub cur_slot: usize,
    /// token histories of the lanes that are not active (the active lane's is `ids_seen`)
    lane_ids: Vec<Vec<u32>>,
    // scratch
    h: DeviceBuffer<u16>,
    h2: DeviceBuffer<u16>,
    pre0: DeviceBuffer<f32>,
    mix_a: DeviceBuffer<f32>,
    mixp: DeviceBuffer<f32>,
    mix_f: DeviceBuffer<f32>,
    x: DeviceBuffer<u16>,
    xf: DeviceBuffer<f32>,
    t32: DeviceBuffer<f32>,
    qr_raw: DeviceBuffer<u16>,
    qr: DeviceBuffer<u16>,
    qrf: DeviceBuffer<f32>,
    q: DeviceBuffer<u16>,
    kv_raw: DeviceBuffer<u16>,
    kv: DeviceBuffer<u16>,
    kvc: DeviceBuffer<f32>,
    sc: DeviceBuffer<f32>,
    latpre: DeviceBuffer<u16>,
    lat: DeviceBuffer<u16>,
    latf: DeviceBuffer<f32>,
    ik_raw: DeviceBuffer<u16>,
    ik: DeviceBuffer<u16>,
    iqb: DeviceBuffer<u16>,
    iqf: DeviceBuffer<f32>,
    iw: DeviceBuffer<f32>,
    score: DeviceBuffer<f32>,
    sel: DeviceBuffer<i32>,
    bmax: DeviceBuffer<f32>,
    cand_sel: DeviceBuffer<i32>,
    cand_mask: DeviceBuffer<u8>,
    o: DeviceBuffer<u16>,
    u: DeviceBuffer<f32>,
    ub: DeviceBuffer<u16>,
    part: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    wts: DeviceBuffer<f32>,
    rscores: DeviceBuffer<f32>,
    rlogits: DeviceBuffer<f32>,
    meta: DeviceBuffer<u32>,
    floor: DeviceBuffer<u32>,
    ekv: DeviceBuffer<f32>,
    evecs: Vec<DeviceBuffer<f32>>,
    graphs: std::collections::HashMap<(u8, usize, usize, u64), (fe_cuda::Graph, usize)>,
    use_graphs: bool,
    just_captured: bool,
    hc_simple: bool,
    bucket_pos: bool,
    tokens_done: usize,
    sh_g: DeviceBuffer<f32>,
    sh_u: DeviceBuffer<f32>,
    sh_gb: DeviceBuffer<u16>,
    sh_ub: DeviceBuffer<u16>,
    sh_h: DeviceBuffer<u16>,
    sh_o: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    /// the tokens fed so far (position = index)
    pub ids_seen: Vec<u32>,
    /// per-layer residual of the last row (bf16 -> f32 [4 d]) after each layer, when `tap_layers` is on
    pub tap_layers: bool,
    /// routing trace (DSC trace mode): (layer, row, ids, weights) of every MoE call
    pub trace: Option<Vec<(u8, u32, [u16; 6], [f32; 6])>>,
    pub layer_h: Vec<Vec<f32>>,
    pub split: Split,
    /// FE: skip the routed experts' clamp request (kern not merged yet)
    pub swiglu_limit: bool,
    /// micro-batch pipeline (DSC_PIPE=n: passes of >= n rows split in two, CPU lane overlapped with the other half's GPU work); 0 = off
    pub pipe_rows: usize,
    sc2: Option<pipe::Scratch>,
    xsc2: Option<pipe::XScratch>,
}

/// Quantize one bf16 (or f32) row of `k` values into the shared q8_1 activation (a macro: disjoint field borrows).
macro_rules! q8 {
    ($e:expr, $x:expr, $bf16:expr, $k:expr) => {{
        let k = $k;
        $e.ggq.quantize_q8_1(&$e.s, $x, k, $bf16, $e.nr, k, &mut $e.act)?
    }};
}

pub mod prefill;

/// Elements of the 4 streams one split-K mixes block sums.
const MIX_SLICE: usize = 512;

fn g1(n: usize) -> u32 {
    n.div_ceil(256) as u32
}

impl<'a> Engine<'a> {
    pub fn new(g: &'a Gguf, c: &'a DscConfig, max_ctx: usize) -> Result<Engine<'a>> {
        let ctx = Context::new(0)?;
        let s = if std::env::var("DSC_S_PRIO").map_or(true, |v| v != "0") { Stream::new_high()? } else { Stream::new()? };
        let ggq = Ggq::load()?;
        let k = Kern::load()?;
        let t0 = Instant::now();
        let (d, hd) = (c.n_embd, c.head_dim);
        let mut layers = Vec::new();
        for l in 0..c.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}.weight");
            let role = c.role(l);
            let hc = |a: &str| -> Result<Hc> {
                Ok(Hc { fnw: raw(g, &p(&format!("hc_{a}_fn")))?, scale: raw(g, &p(&format!("hc_{a}_scale")))?, base: raw(g, &p(&format!("hc_{a}_base")))? })
            };
            let comp = if role.kv_source {
                Some(Comp {
                    wkv: Wt::load(g, &p("attn_compressor_kv"))?,
                    wgate: if role.ratio == 2 { Some(Wt::load(g, &p("attn_compressor_gate"))?) } else { None },
                    norm: norm_bf16(g, &p("attn_compressor_norm"))?,
                })
            } else {
                None
            };
            let idx = if role.index_source {
                let has_k = g.tensor(&p("indexer.attn_k")).is_some();
                Some(Idx {
                    wq_b: Wt::load(g, &p("indexer.attn_q_b"))?,
                    proj: Wt::load(g, &p("indexer.proj"))?,
                    wk: if has_k { Some(Wt::load(g, &p("indexer.attn_k"))?) } else { None },
                    k_norm: if has_k { Some(norm_bf16(g, &p("indexer.k_norm"))?) } else { None },
                })
            } else if role.kv_source && g.tensor(&p("indexer.attn_k")).is_some() {
                Some(Idx {
                    wq_b: Wt::load(g, &p("indexer.attn_q_b")).or_else(|_| Wt::load(g, &p("attn_q_a")))?,
                    proj: Wt::load(g, &p("attn_q_a"))?,
                    wk: Some(Wt::load(g, &p("indexer.attn_k"))?),
                    k_norm: Some(norm_bf16(g, &p("indexer.k_norm"))?),
                })
            } else {
                None
            };
            let eng = if c.engram_layers.contains(&l) {
                Some(Eng { kv: Wt::load(g, &p("engram_kv"))?, qk: raw(g, &p("engram_gate"))? })
            } else {
                None
            };
            layers.push(Layer {
                role,
                attn_norm: norm_bf16(g, &p("attn_norm"))?,
                q_a: Wt::load(g, &p("attn_q_a"))?,
                q_a_norm: norm_bf16(g, &p("attn_q_a_norm"))?,
                q_b: Wt::load(g, &p("attn_q_b"))?,
                kv: Wt::load(g, &p("attn_kv"))?,
                kv_norm: norm_bf16(g, &p("attn_kv_a_norm"))?,
                sinks: raw(g, &p("attn_sinks"))?,
                wo_a: Wt::load(g, &p("attn_output_a"))?,
                wo_b: Wt::load(g, &p("attn_output_b"))?,
                hc_attn: hc("attn")?,
                hc_ffn: hc("ffn")?,
                ffn_norm: norm_bf16(g, &p("ffn_norm"))?,
                gate_w: Wt::load(g, &p("ffn_gate_inp"))?,
                gate_b: raw(g, &format!("blk.{l}.exp_probs_b.bias"))?,
                sh_gate: Wt::load(g, &p("ffn_gate_shexp"))?,
                sh_up: Wt::load(g, &p("ffn_up_shexp"))?,
                sh_down: Wt::load(g, &p("ffn_down_shexp"))?,
                comp,
                idx,
                eng,
            });
            let tg = g.tensor(&p("ffn_gate_exps")).unwrap();
            let tu = g.tensor(&p("ffn_up_exps")).unwrap();
            ensure!(tg.ty == tu.ty, "layer {l}: gate {} != up {}", tg.ty.name(), tu.ty.name());
        }
        eprintln!("dense weights on the card in {:.1}s", t0.elapsed().as_secs_f32());
        let ring_min: usize = std::env::var("DSC_RING").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
        let ring_len = (c.window + 8).next_power_of_two().max(ring_min);
        let (mut ring, mut tail, mut plane, mut ikeys, mut entries) = (vec![], vec![], vec![], vec![], vec![]);
        // DSC_LANES sequence slots (default 1): the kernels index every cache by the slot in the row metadata
        let n_slots: usize = std::env::var("DSC_LANES").ok().and_then(|v| v.parse().ok()).unwrap_or(1).clamp(1, 8);
        for l in 0..c.n_layer {
            let role = &layers[l].role;
            ring.push(DeviceBuffer::zeroed(n_slots * ring_len * hd)?);
            let ne = if role.kv_source { max_ctx / role.ratio as usize } else { 0 };
            entries.push(ne);
            tail.push(if role.kv_source { Some(DeviceBuffer::zeroed(n_slots * ring_len * 2 * hd)?) } else { None });
            plane.push(if role.kv_source { Some(DeviceBuffer::zeroed(n_slots * ne * hd)?) } else { None });
            ikeys.push(if role.kv_source { Some(DeviceBuffer::zeroed(n_slots * ne * c.idx_dim)?) } else { None });
        }
        let nb = max_ctx.div_ceil(c.cand_block.max(1));
        let ih = c.idx_heads * c.idx_dim;
        let tab_plain = DeviceBuffer::from_slice(&rope_table(c, c.rope_base as f64, 0, max_ctx))?;
        let tab_comp = DeviceBuffer::from_slice(&rope_table(c, c.rope_compress_base as f64, c.yarn_orig_ctx, max_ctx))?;
        let cpus = std::env::var("DSC_CPUS").ok().or_else(|| std::env::var("DSC_REPACK").ok().filter(|v| v == "1").map(|_| "1-31".to_string()));
        let cpu = match cpus {
            Some(s) => {
                let mut m = CpuMoe::pinned(&fe_ggq::cpu::pool::parse_cpus(&s), Duration::from_millis(50), d, c.n_ff_exp);
                m.set_caller_cpu(0);
                m
            }
            None => CpuMoe::new(std::env::var("DSC_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(32), d, c.n_ff_exp),
        };
        // rows the big-prompt path may take in one pass (DSC_PF_ROWS, 0 = off): bounded by the window ring
        let pf_rows = std::env::var("DSC_PF_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let pf_rows = pf_rows.min(ring_len.saturating_sub(c.window + 16));
        let rmax = if pf_rows > RMAX { pf_rows } else { RMAX };
        let pf_n = std::env::var("DSC_PF_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0).max(rmax);
        let e = Engine {
            g, c, _ctx: ctx, s, ggq, k, out_norm: norm_bf16(g, "output_norm.weight")?, out_w: Wt::load(g, "output.weight")?,
            eng: Engram::new(g, c)?, cpu,
            act: Q8Act::alloc(rmax.max(8 * 8), 8192)?,
            hq: Q8Act::alloc(RMAX * c.n_expert_used, c.n_ff_exp)?,
            ug: DeviceBuffer::zeroed(rmax * c.o_lora)?,
            taps: DeviceBuffer::zeroed(pf_n * 3 * c.n_embd)?,
            nr: 1,
            rmax,
            pf_run: None,
            pf_idxtc: std::env::var("DSC_PF_IDXTC").is_ok_and(|v| v == "1"),
            pf_n,
            tap_off: 0,
            no_graph: false,
            stage_on: std::env::var("DSC_STAGE").map_or(true, |v| v != "0"),
            depth_bucket: std::env::var("DSC_DEPTH_BUCKET").map_or(false, |v| v == "1"),
            grp_wo: std::env::var("DSC_GRPWO").map_or(true, |v| v != "0"),
            graph_clears: 0,
            graph_tick: 0,
            graph_use: Default::default(),
            graph_max: std::env::var("DSC_GRAPH_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(1000),
            big: None,
            pf_fullbp: std::env::var("DSC_PF_FULLBP").is_ok_and(|v| v == "1"),
            pf_mq: c.n_head == 64 && std::env::var("DSC_PF_MQ").map_or(true, |v| v != "0") && rmax > RMAX,
            pmq_o: DeviceBuffer::zeroed(if rmax > RMAX { rmax * 8 * 32 * 512 } else { 1 })?,
            pmq_ml: DeviceBuffer::zeroed(if rmax > RMAX { rmax * 8 * 32 * 2 } else { 1 })?,
            pf_mmq: std::env::var("DSC_PF_MMQ").map_or(true, |v| v != "0"),
            pf_min: std::env::var("DSC_PF_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(192),
            pf: None,
            y6: DeviceBuffer::zeroed(RMAX * (c.n_expert_used * d))?,
            ycpu: DeviceBuffer::zeroed(RMAX * (c.n_expert_used * d))?,
            slot: DeviceBuffer::zeroed(RMAX * (c.n_expert_used))?,
            arena: None,
            xcs: vec![],
            cardtime: std::env::var_os("DSC_CARDTIME").is_some(),
            zc: false,
            zc_ev: vec![],
            zc_used: vec![],
            cs: vec![],
            ev_b: vec![],
            ev_c: vec![],
            mb_x: None,
            mb_ids: None,
            mb_y: None,
            zc_buf: None,
            stage: None,
            pf_hold: false,
            res_pf: vec![],
            xres_pf: vec![],
            pf_pin: None,
            pf_meta_h: None,
            pf_eng_h: None,
            zc_sz: vec![],
            zc_lo: 0,
            zc_hi: 0,
            zc_off: vec![],
            zc_tot: vec![],
            owner_h: vec![],
            zring: vec![],
            ring_slot_bytes: 0,
            ring_expert: vec![],
            ring_stamp: vec![],
            ring_of: vec![],
            ring_clock: 0,
            ring_promo: 0,
            promo_stage: None,
            ids_log: None,
            zc_hist: vec![],
            heat_cnt: vec![],
            heat_path: None,
            heat_acc: false,
            heat_passes: 0,
            pass_log: None,
            zc_sync: std::env::var_os("DSC_ZC_SYNC").is_some(),
            owner: None,
            ev_main: None,
            slot_h: HostBuffer::alloc(RMAX * c.n_expert_used)?,
            yc_h: HostBuffer::alloc(RMAX * c.n_expert_used * c.n_embd)?,
            host: None,
            res: vec![],
            res_on: false,
            resident_experts: 0,
            tab_plain, tab_comp, max_ctx, ring_len, ring, tail, plane, ikeys, entries, layers,
            h: DeviceBuffer::zeroed(rmax * (4 * d))?,
            h2: DeviceBuffer::zeroed(rmax * (4 * d))?,
            pre0: DeviceBuffer::from_slice(&(0..rmax).flat_map(|_| [1.0f32, 0.0, 0.0, 0.0]).collect::<Vec<f32>>())?,
            mix_a: DeviceBuffer::zeroed(rmax * (24))?,
            mixp: DeviceBuffer::zeroed(rmax * (c.hc * d / MIX_SLICE * 25))?,
            mix_f: DeviceBuffer::zeroed(rmax * (24))?,
            x: DeviceBuffer::zeroed(rmax * (d))?,
            xf: DeviceBuffer::zeroed(rmax * (d))?,
            t32: DeviceBuffer::zeroed(rmax * (c.q_dim().max(8192)))?,
            qr_raw: DeviceBuffer::zeroed(rmax * (c.q_lora))?,
            qr: DeviceBuffer::zeroed(rmax * (c.q_lora))?,
            qrf: DeviceBuffer::zeroed(rmax * (c.q_lora))?,
            q: DeviceBuffer::zeroed(rmax * (c.q_dim()))?,
            kv_raw: DeviceBuffer::zeroed(rmax * (hd))?,
            kv: DeviceBuffer::zeroed(rmax * (hd))?,
            kvc: DeviceBuffer::zeroed(rmax * (hd))?,
            sc: DeviceBuffer::zeroed(rmax * (hd))?,
            latpre: DeviceBuffer::zeroed(rmax * (hd))?,
            lat: DeviceBuffer::zeroed(rmax * (hd))?,
            latf: DeviceBuffer::zeroed(rmax * (hd))?,
            ik_raw: DeviceBuffer::zeroed(rmax * (c.idx_dim))?,
            ik: DeviceBuffer::zeroed(rmax * (c.idx_dim))?,
            iqb: DeviceBuffer::zeroed(rmax * (ih))?,
            iqf: DeviceBuffer::zeroed(rmax * (ih))?,
            iw: DeviceBuffer::zeroed(rmax * (c.idx_heads))?,
            score: DeviceBuffer::zeroed(rmax * (max_ctx))?,
            sel: DeviceBuffer::zeroed(rmax * (c.idx_top_k))?,
            bmax: DeviceBuffer::zeroed(rmax * (nb))?,
            cand_sel: DeviceBuffer::zeroed(rmax * (c.cand_blocks.max(1)))?,
            cand_mask: DeviceBuffer::zeroed(rmax * (nb))?,
            o: DeviceBuffer::zeroed(rmax * (c.q_dim()))?,
            u: DeviceBuffer::zeroed(rmax * (c.o_groups * c.o_lora))?,
            ub: DeviceBuffer::zeroed(rmax * (c.o_groups * c.o_lora))?,
            part: DeviceBuffer::zeroed(rmax * (d))?,
            ids: DeviceBuffer::zeroed(rmax * (c.n_expert_used))?,
            wts: DeviceBuffer::zeroed(rmax * (c.n_expert_used))?,
            rscores: DeviceBuffer::zeroed(rmax * (c.n_expert))?,
            rlogits: DeviceBuffer::zeroed(rmax * (c.n_expert))?,
            meta: DeviceBuffer::zeroed(rmax * (2))?,
            floor: DeviceBuffer::zeroed(1)?,
            ekv: DeviceBuffer::zeroed(rmax * (c.engram_width().max((c.hc + 1) * d)))?,
            evecs: (0..c.engram_layers.len()).map(|_| DeviceBuffer::zeroed(rmax * c.engram_width())).collect::<Result<Vec<_>, _>>()?,
            graphs: Default::default(),
            use_graphs: std::env::var("DSC_GRAPH").map_or(true, |v| v != "0"),
            just_captured: false,
            bucket_pos: std::env::var("DSC_BUCKET").map_or(false, |v| v == "1"),
            hc_simple: std::env::var("DSC_HC").map_or(false, |v| v == "simple"),
            tokens_done: 0,
            sh_g: DeviceBuffer::zeroed(rmax * (c.n_ff_exp))?,
            sh_u: DeviceBuffer::zeroed(rmax * (c.n_ff_exp))?,
            sh_gb: DeviceBuffer::zeroed(rmax * (c.n_ff_exp))?,
            sh_ub: DeviceBuffer::zeroed(rmax * (c.n_ff_exp))?,
            sh_h: DeviceBuffer::zeroed(rmax * (c.n_ff_exp))?,
            sh_o: DeviceBuffer::zeroed(rmax * (d))?,
            logits: DeviceBuffer::zeroed(RMAX * (c.n_vocab))?,
            ids_seen: vec![],
            n_slots,
            cur_slot: 0,
            lane_ids: vec![vec![]; n_slots],
            tap_layers: false,
            trace: None,
            layer_h: vec![],
            split: Split::default(),
            swiglu_limit: true,
            pipe_rows: 0,
            sc2: None,
            xsc2: None,
        };
        let mut e = e;
        if e.rmax > RMAX {
            e.pf = Some(prefill::PfCard::alloc(c, g, e.pf_n)?);
            e.big = Some(prefill::BigStore::alloc(c, e.pf_n, e.max_ctx)?);
        }
        {
            let nst: usize = std::env::var("DSC_STAGE_SLOTS_MAIN").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
            if nst > 0 {
                let mut sb = 0usize;
                for l in 0..c.n_layer {
                    let t: usize = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].iter().map(|n| g.tensor(&format!("blk.{l}.{n}.weight")).unwrap().slice_bytes().div_ceil(256) * 256).sum();
                    sb = sb.max(t);
                }
                e.stage = Some(StageBuf { buf: DeviceBuffer::zeroed(nst * sb)?, counter: DeviceBuffer::zeroed(1)?, slot_e: DeviceBuffer::zeroed(nst)?, orig: DeviceBuffer::zeroed(3 * nst)?, nslots: nst, slot_bytes: sb });
            }
        }
        match std::env::var("DSC_ARENA_GB").as_deref() {
            Ok("off") => {
                e.res = (0..c.n_layer).map(|_| None).collect();
                e.res_on = true;
            }
            Ok(v) => e.load_residents(v.parse::<f64>()? * (1u64 << 30) as f64)?,
            Err(_) => {
                let (free, _) = fe_cuda::mem_info()?;
                // reserve for the context caches / graphs / scratch that grow after the load (DSC_RESERVE_GB, default 0.5)
                let res = reserve_gb(0);
                e.load_residents((free as f64 - res * (1u64 << 30) as f64).max(0.0))?
            }
        }
        if e.rmax > RMAX {
            e.build_pf_pin()?;
        }
        if let Some(n) = std::env::var("DSC_PIPE").ok().and_then(|v| v.parse::<usize>().ok()) {
            e.enable_pipe(n)?;
        }
        Ok(e)
    }

    /// Put the hottest experts (picks per byte, from the routing traces in DSC_HEAT, default trace-a.bin) on the cards in
    /// byte-sized arenas of at most `budget` bytes (card 1 = this engine's card) and `budget0` (the expert-only card,
    /// DSC_XCARD=<ordinal>); the rest stay on the CPU lane. Static placement (no eviction yet). Experts are dealt to the
    /// card with the smaller used fraction.
    pub fn load_residents(&mut self, budget: f64) -> Result<()> {
        let (c, g) = (self.c, self.g);
        let ne = c.n_expert;
        let heat = std::env::var("DSC_HEAT").map_err(|_| anyhow::anyhow!("DSC_HEAT is not set: the placement heat file (the launcher tools/serve-dsc.sh sets it to crates/fe-dsc/heat/coder.cnt)"))?;
        let mut cnt = vec![0f64; c.n_layer * ne];
        // DSC_HEAT=a.bin:b.bin (raw pick counts added) or a.bin@0.7:b.bin@0.3 (each trace normalised to its share of the heat: a mix of
        // corpora of different sizes by weight, not by token count)
        for p in heat.split(':').filter(|p| !p.is_empty()) {
            let (path, share) = match p.rsplit_once('@') {
                Some((a, w)) => (a, w.parse::<f64>().ok()),
                None => (p, None),
            };
            let b = std::fs::read(path).with_context(|| path.to_string())?;
            let mut part = vec![0f64; c.n_layer * ne];
            if path.ends_with(".cnt") {
                // pick counts (u64 LE per (layer, expert)) as DSC_HEAT_LOG / dsc-run p2 write them
                for (i, v) in b.chunks_exact(8).enumerate().take(part.len()) {
                    part[i] = u64::from_le_bytes(v.try_into().unwrap()) as f64;
                }
            } else {
            for r in b.chunks_exact(45) {
                let l = r[4] as usize;
                for k in 0..6 {
                    part[l * ne + u16::from_le_bytes([r[9 + 2 * k], r[10 + 2 * k]]) as usize] += 1.0;
                }
            }
            }
            let scale = share.map_or(1.0, |w| w * 1.0e6 / part.iter().sum::<f64>().max(1.0));
            for (a, v) in cnt.iter_mut().zip(&part) {
                *a += v * scale;
            }
        }
        let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
        let ts: Vec<Vec<&fe_gguf::TensorInfo>> = (0..c.n_layer)
            .map(|l| names.iter().map(|n| g.tensor(&format!("blk.{l}.{n}.weight")).unwrap()).collect()).collect();
        let al = |b: usize| b.div_ceil(256) * 256;
        let eb: Vec<usize> = ts.iter().map(|t| t.iter().map(|x| al(x.slice_bytes())).sum()).collect();
        let mut order: Vec<usize> = (0..c.n_layer * ne).collect();
        order.sort_by(|&a, &b| ((cnt[b] + 0.01) / eb[b / ne] as f64).partial_cmp(&((cnt[a] + 0.01) / eb[a / ne] as f64)).unwrap());
        // the expert-only cards: DSC_XCARDS=1,2 (or the old DSC_XCARD=1); budgets DSC_ARENAX_GB=a,b (DSC_ARENA0_GB for the first), default
        // free memory minus 1 GiB, minus 9 GiB on the card that hosts the drafter (DSC_DRAFT_CARD)
        let xlist: Vec<i32> = std::env::var("DSC_XCARDS").or_else(|_| std::env::var("DSC_XCARD")).ok()
            .map(|v| v.split(',').filter(|p| !p.trim().is_empty()).map(|p| p.trim().parse::<i32>()).collect::<std::result::Result<Vec<_>, _>>()).transpose()?.unwrap_or_default();
        let xbud: Vec<f64> = std::env::var("DSC_ARENAX_GB").ok().map(|v| v.split(',').filter_map(|p| p.trim().parse::<f64>().ok()).collect()).unwrap_or_default();
        let draft_card: i32 = std::env::var("DSC_DRAFT_CARD").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let ring_gb: f64 = if !xlist.is_empty() && std::env::var("DSC_ZC").map_or(true, |v| v != "0") { std::env::var("DSC_RING_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0) } else { 0.0 };
        for (k, &ord) in xlist.iter().enumerate() {
            let ctx = Context::new(ord)?;
            let s = Stream::new()?;
            let ggq = Ggq::load()?;
            // the big-prompt workspace first, so the free memory below already excludes it
            let pf_opt = if self.rmax > RMAX { Some(prefill::PfCard::alloc(c, g, self.pf_n)?) } else { None };
            let nst: usize = std::env::var("DSC_STAGE_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
            let stage_opt = if nst > 0 {
                let sb = eb.iter().copied().max().unwrap();
                Some(StageBuf { buf: DeviceBuffer::zeroed(nst * sb)?, counter: DeviceBuffer::zeroed(1)?, slot_e: DeviceBuffer::zeroed(nst)?, orig: DeviceBuffer::zeroed(3 * nst)?, nslots: nst, slot_bytes: sb })
            } else {
                None
            };
            let (free, _) = fe_cuda::mem_info()?;
            let x0 = DeviceBuffer::zeroed(RMAX * c.n_embd)?;
            let ids0 = DeviceBuffer::zeroed(RMAX * c.n_expert_used)?;
            let y60 = DeviceBuffer::zeroed(RMAX * c.n_expert_used * c.n_embd)?;
            let act = Q8Act::alloc(RMAX, 8192)?;
            let hq = Q8Act::alloc(RMAX * c.n_expert_used, c.n_ff_exp)?;
            let gib = (1u64 << 30) as f64;
            let b0 = xbud.get(k).copied().or_else(|| if k == 0 { std::env::var("DSC_ARENA0_GB").ok().and_then(|v| v.parse::<f64>().ok()) } else { None })
                .map(|v| v * gib)
                .unwrap_or_else(|| free as f64 - ring_gb * gib - reserve_gb(k + 1) * gib - if ord == draft_card { 8.8 * gib } else { 0.0 });
            self.xcs.push(XCard { stage: stage_opt, ctx, s, ggq, act, hq, x: x0, ids: ids0, y6: y60, arena: None, res: vec![], budget: b0.max(0.0), ev: fe_cuda::Event::new_no_timing()?, ev_a: fe_cuda::Event::new()?, ev_b: fe_cuda::Event::new()?, x_h: HostBuffer::alloc(RMAX * c.n_embd)?, ids_h: HostBuffer::alloc(RMAX * c.n_expert_used)?, pf: pf_opt });
            self._ctx.bind()?;
        }
        let nx = self.xcs.len();
        if nx > 0 {
            for xc in &self.xcs {
                // NOTE: cuCtxEnablePeerAccess makes cross-device copies silently deliver zeros on this box; the plain peer copy works
                let _ = xc;
            }
            self._ctx.bind()?;
            // ycpu: one row region per expert card (indexed by pair) + the CPU lane's rows after them
            self.ycpu = DeviceBuffer::zeroed((nx + 1) * RMAX * c.n_expert_used * c.n_embd)?;
        }
        // cards: 0 = this engine's card (`budget`), 1.. = the expert cards; each expert goes to the card with the smaller used fraction
        let bud: Vec<f64> = std::iter::once(budget).chain(self.xcs.iter().map(|x| x.budget)).collect();
        let mut used = vec![0usize; nx + 1];
        let mut chs: Vec<Vec<(usize, usize)>> = vec![vec![]; nx + 1];
        // DSC_PLACE=frac (old): the card with the smaller used fraction; default: the card with the least heat per speed that still has room, so every
        // card reads about the same bytes a token (the card with the most room no longer gets most of the hot experts). DSC_PLACE_SPEED=<main>,<c1>,<c2> weights.
        let by_frac = std::env::var("DSC_PLACE").map_or(false, |v| v == "frac");
        let pspeed: Vec<f64> = std::env::var("DSC_PLACE_SPEED").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<f64>>())
            .filter(|w| w.len() == nx + 1).unwrap_or_else(|| vec![1.0; nx + 1]);
        let cold_mode = std::env::var("DSC_PLACE").map_or(false, |v| v == "cold") && nx > 0;
        let mut hload = vec![0f64; nx + 1];
        let order: Vec<usize> = if std::env::var("DSC_PLACE").map_or(false, |v| v == "cold") && nx > 0 {
            // the resident set = the hottest experts that fit the three budgets; the main card (all the dense work) takes the COLDEST of it,
            // the hottest go to the expert cards (dealt by heat per speed below)
            let total: f64 = bud.iter().sum();
            let mut acc = 0.0;
            let mut res: Vec<usize> = vec![];
            for &i in &order {
                let sz = eb[i / ne] as f64;
                if acc + sz > total { break; }
                acc += sz;
                res.push(i);
            }
            let mut main_set: Vec<usize> = vec![];
            let mut u0 = 0.0;
            for &i in res.iter().rev() {
                let sz = eb[i / ne] as f64;
                if u0 + sz <= bud[0] {
                    u0 += sz;
                    main_set.push(i);
                    chs[0].push((i, used[0]));
                    used[0] += sz as usize;
                    hload[0] += cnt[i] + 0.01;
                }
            }
            let in_main: std::collections::HashSet<usize> = main_set.into_iter().collect();
            res.into_iter().filter(|i| !in_main.contains(i)).collect()
        } else {
            order
        };
        for i in order {
            let sz = eb[i / ne] as f64;
            let mut best: Option<(usize, f64)> = None;
            for k in 0..=nx {
                if k == 0 && cold_mode {
                    continue;
                }
                if used[k] as f64 + sz <= bud[k] {
                    let f = if by_frac { used[k] as f64 / bud[k].max(1.0) } else { hload[k] / pspeed[k] };
                    if best.map_or(true, |(_, bf)| f < bf) {
                        best = Some((k, f));
                    }
                }
            }
            if let Some((k, _)) = best {
                chs[k].push((i, used[k]));
                used[k] += sz as usize;
                hload[k] += cnt[i] + 0.01;
            }
        }
        let ch1 = chs[0].clone();
        let u1 = used[0];
        let ch0: Vec<(usize, usize)> = chs[1..].iter().flatten().copied().collect();
        self.res_on = true;
        self.resident_experts = chs.iter().map(|v| v.len()).sum();
        let tot = cnt.iter().sum::<f64>().max(1.0);
        let heat_of = |ch: &[(usize, usize)]| 100.0 * ch.iter().map(|&(i, _)| cnt[i]).sum::<f64>() / tot;
        for (k, ch) in chs.iter().enumerate() {
            eprintln!("resident experts: card {} {} ({:.2} GiB, {:.1}% of the heat; budget {:.2} GiB)", k, ch.len(), used[k] as f64 / (1u64 << 30) as f64, heat_of(ch), bud[k] / (1u64 << 30) as f64);
        }
        eprintln!("resident experts total {} of {} (heat from {})", self.resident_experts, c.n_layer * ne, heat);
        // zero-copy host experts: every expert that is not resident goes to pinned host memory and is read over PCIe by the card that owns it
        // (its table entries point at host addresses); every expert then has an owner card, routing happens on the cards, the CPU lane is not used
        let zc_on = nx > 0 && std::env::var("DSC_ZC").map_or(true, |v| v != "0");
        let mut zc_extra: Vec<Vec<(usize, [u64; 3])>> = vec![vec![]; nx + 1];
        let mut owner = vec![0u8; c.n_layer * ne];
        for (k, ch) in chs.iter().enumerate() {
            for &(i, _) in ch {
                owner[i] = k as u8;
            }
        }
        if zc_on {
            let t0 = Instant::now();
            let mut on = vec![false; c.n_layer * ne];
            for ch in &chs {
                for &(i, _) in ch {
                    on[i] = true;
                }
            }
            let mut off = vec![[0usize; 3]; c.n_layer * ne];
            let mut len = 0usize;
            for i in 0..c.n_layer * ne {
                if !on[i] {
                    for j in 0..3 {
                        off[i][j] = len;
                        len += al(ts[i / ne][j].slice_bytes());
                    }
                }
            }
            let mut zc_off_v = vec![[usize::MAX; 3]; c.n_layer * ne];
            let mut zc_tot_v = vec![0usize; c.n_layer * ne];
            let mut hb = HostBuffer::<u8>::alloc(len.max(256))?;
            let heat_v = cnt.clone();
            let mut cnt = vec![0usize; nx + 1];
            // readers of host-backed experts: DSC_ZC_W=<main>,<card 1>,<card 2>... relative weights (default 1,1,1: the main card reads too); the experts are dealt hottest first
            // to the reader with the least heat per weight, so the cards share the cold reads (not just the expert count)
            let zw: Vec<f64> = std::env::var("DSC_ZC_W").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<f64>>())
                .filter(|w| w.len() == nx + 1).unwrap_or_else(|| (0..=nx).map(|r| if r == 2 { 0.5 } else { 1.0 }).collect());
            let mut load = vec![0f64; nx + 1];
            let mut order_cold: Vec<usize> = (0..c.n_layer * ne).filter(|&i| !on[i]).collect();
            order_cold.sort_by(|&a, &b| heat_v[b].partial_cmp(&heat_v[a]).unwrap());
            for i in order_cold {
                let (l, e) = (i / ne, i % ne);
                let mut ptrs = [0u64; 3];
                for j in 0..3 {
                    let sl = g.slice(ts[l][j], e);
                    unsafe { std::ptr::copy_nonoverlapping(sl.as_ptr(), hb.as_mut_ptr().add(off[i][j]), sl.len()) };
                    ptrs[j] = hb.dev().0 + off[i][j] as u64;
                }
                let r = (0..=nx).filter(|&r| zw[r] > 0.0).min_by(|&a, &b| (load[a] / zw[a]).partial_cmp(&(load[b] / zw[b])).unwrap()).unwrap();
                load[r] += heat_v[i] + 0.5;
                cnt[r] += 1;
                owner[i] = r as u8;
                zc_extra[r].push((i, ptrs));
                zc_off_v[i] = off[i];
                zc_tot_v[i] = al(ts[l][0].slice_bytes()) + al(ts[l][1].slice_bytes()) + al(ts[l][2].slice_bytes());
            }
            eprintln!("zero-copy host experts: {} experts, {:.2} GiB pinned in {:.1}s, per card {:?}", cnt.iter().sum::<usize>(), len as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f32(), cnt);
            self.zc_buf = Some(hb);
            self.zc_sz = (0..c.n_layer).map(|l| [al(ts[l][0].slice_bytes()), al(ts[l][1].slice_bytes()), al(ts[l][2].slice_bytes())]).collect();
            self.zc_lo = self.zc_buf.as_ref().unwrap().dev().0;
            self.zc_hi = self.zc_lo + len as u64;
            self.zc_off = zc_off_v;
            self.zc_tot = zc_tot_v;
            self.owner_h = owner.clone();
            self.zc = true;
            // true peer-to-peer (needs the IOMMU in passthrough: with the default translated domain this made copies deliver zeros, which the
            // self-test below would catch). DSC_PEER=0 keeps the staged path.
            if std::env::var("DSC_PEER").map_or(true, |v| v != "0") {
                let mut cs: Vec<&Context> = vec![&*self._ctx];
                cs.extend(self.xcs.iter().map(|x| &*x.ctx));
                for a in 0..cs.len() {
                    for b in 0..cs.len() {
                        if a != b {
                            let _ = cs[a].enable_peer(cs[b]);
                        }
                    }
                }
            }
            self._ctx.bind()?;
            self.owner = Some(DeviceBuffer::from_slice(&owner)?);
            self.ev_main = Some(fe_cuda::Event::new_no_timing()?);
            if self.cardtime {
                for k in 0..nx {
                    self.xcs[k].ctx.bind()?;
                    self.zc_ev.push((0..c.n_layer).map(|_| Ok((fe_cuda::Event::new()?, fe_cuda::Event::new()?))).collect::<Result<Vec<_>>>()?);
                    self.zc_used.push(vec![false; c.n_layer]);
                }
                self._ctx.bind()?;
            }
            for _ in 0..nx {
                self.cs.push(Stream::new()?);
                self.ev_b.push(fe_cuda::Event::new_no_timing()?);
                self.ev_c.push(fe_cuda::Event::new_no_timing()?);
            }
            self.p2p_selftest()?;
        }
        let upload = |chosen: &[(usize, usize)], used: usize, extra: &[(usize, [u64; 3])]| -> Result<(Option<DeviceBuffer<u8>>, Vec<Option<LayerRes>>)> {
            let mut res: Vec<Option<LayerRes>> = (0..c.n_layer).map(|_| None).collect();
            if chosen.is_empty() && extra.is_empty() {
                return Ok((None, res));
            }
            let arena = DeviceBuffer::<u8>::zeroed(used.max(256))?;
            let mut tabs = vec![[vec![0u64; ne], vec![0u64; ne], vec![0u64; ne]]; c.n_layer];
            let mut present = vec![vec![false; ne]; c.n_layer];
            for &(i, off) in chosen {
                let (l, e) = (i / ne, i % ne);
                let mut o = off;
                for (j, t) in ts[l].iter().enumerate() {
                    let sl = g.slice(t, e);
                    memcpy_htod(arena.ptr().byte_add(o), sl)?;
                    tabs[l][j][e] = arena.ptr().byte_add(o).0;
                    o += al(sl.len());
                }
                present[l][e] = true;
            }
            for &(i, ptrs) in extra {
                let (l, e) = (i / ne, i % ne);
                for j in 0..3 {
                    tabs[l][j][e] = ptrs[j];
                }
                present[l][e] = true;
            }
            for l in 0..c.n_layer {
                if present[l].iter().any(|&p| p) {
                    res[l] = Some(LayerRes {
                        gtab: DeviceBuffer::from_slice(&tabs[l][0])?, utab: DeviceBuffer::from_slice(&tabs[l][1])?,
                        dtab: DeviceBuffer::from_slice(&tabs[l][2])?, present: present[l].clone(), ty_gu: ts[l][0].ty,
                        ty_d: ts[l][2].ty, rb_gu: ts[l][0].row_bytes(), rb_d: ts[l][2].row_bytes(),
                    });
                }
            }
            Ok((Some(arena), res))
        };
        // Pre-fault the mapped GGUF pages of the experts the CPU lane serves (MADV_POPULATE_READ, Linux >= 5.14): first-touch page faults
        // in the workers cost 12-14 GB/s cold against 38-40 populated (dsc-kern's cold bench).
        if std::env::var("DSC_POPULATE").map_or(true, |v| v != "0") && std::env::var("DSC_REPACK").map_or(true, |v| v != "1") && !zc_on {
            let t0 = Instant::now();
            let mut on = vec![false; c.n_layer * ne];
            for &(i, _) in ch1.iter().chain(ch0.iter()) {
                on[i] = true;
            }
            let mut bytes = 0usize;
            for i in 0..c.n_layer * ne {
                if on[i] {
                    continue;
                }
                let (l, e) = (i / ne, i % ne);
                for t in &ts[l] {
                    let sl = g.slice(t, e);
                    let start = (sl.as_ptr() as usize) & !4095;
                    let end = (sl.as_ptr() as usize + sl.len() + 4095) & !4095;
                    unsafe { libc::madvise(start as *mut libc::c_void, end - start, 22 /* MADV_POPULATE_READ */) };
                    bytes += end - start;
                }
            }
            eprintln!("pre-faulted {:.1} GiB of CPU-lane experts in {:.1}s", bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f32());
        }
        let (a1, r1) = upload(&ch1, u1, &zc_extra[0])?;
        self.arena = a1;
        self.res = r1;
        if std::env::var("DSC_REPACK").map_or(false, |v| v == "1") {
            let mut on = vec![false; c.n_layer * ne];
            for &(i, _) in ch1.iter().chain(ch0.iter()) {
                on[i] = true;
            }
            let rt = |t: GgmlType| crate::rep_type(t).ok_or_else(|| anyhow!("DSC_REPACK: no repack for {}", t.name()));
            let mut rty = vec![];
            let mut rsz = vec![];
            for l in 0..c.n_layer {
                let (a, b) = (rt(ts[l][0].ty)?, rt(ts[l][2].ty)?);
                rty.push((a, b));
                rsz.push([0, 1, 2].map(|j| ts[l][j].dims[1] as usize * crate::rep_row_bytes(if j == 2 { b } else { a }, ts[l][j].ne0())));
            }
            let mut off = vec![[usize::MAX; 3]; c.n_layer * ne];
            let mut len = 0usize;
            for i in 0..c.n_layer * ne {
                if !on[i] {
                    let l = i / ne;
                    for j in 0..3 {
                        off[i][j] = len;
                        len += al(rsz[l][j]);
                    }
                }
            }
            let len = len.div_ceil(2 << 20) * (2 << 20);
            // free-memory guard: the repacked copy must fit (the page cache of the GGUF is reclaimable and counts as available)
            if let Ok(mi) = std::fs::read_to_string("/proc/meminfo") {
                let avail_kb: usize = mi.lines().find(|l| l.starts_with("MemAvailable")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok()).unwrap_or(usize::MAX / 2048);
                ensure!(len / 1024 + (6 << 20) < avail_kb, "DSC_REPACK needs {:.1} GiB, MemAvailable is {:.1} GiB", len as f64 / (1u64 << 30) as f64, avail_kb as f64 / (1u64 << 20) as f64);
            }
            // drop the GGUF shards from the page cache first: the node-bound huge pages would otherwise stall in direct reclaim
            let mp = crate::model_path();
            for f in fe_gguf::shard_paths(&mp) {
                if let Ok(fh) = std::fs::File::open(&f) {
                    use std::os::fd::AsRawFd;
                    unsafe { libc::posix_fadvise(fh.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
                }
            }
            let t0 = Instant::now();
            let ptr = unsafe {
                let raw = libc::mmap(std::ptr::null_mut(), len + (2 << 20), libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0);
                anyhow::ensure!(raw != libc::MAP_FAILED, "mmap {len} bytes failed");
                let p = ((raw as usize + (2 << 20) - 1) & !((2 << 20) - 1)) as *mut u8;
                libc::madvise(p as *mut libc::c_void, len, libc::MADV_HUGEPAGE);
                // 2 MiB page k lives on node k % 2 (as far as each node has the memory for it, else the other node): the CPU lane claims each
                // work unit on the node that holds its weights
                let free_gib = |n: usize| -> f64 {
                    let t = std::fs::read_to_string(format!("/sys/devices/system/node/node{n}/meminfo")).unwrap_or_default();
                    let g = |k: &str| t.lines().find(|l| l.contains(k)).and_then(|l| l.split_whitespace().rev().nth(1)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
                    (g("MemFree:") + 0.8 * g("FilePages:")) / (1u64 << 20) as f64
                };
                let headroom = 6.0;
                let mut quota = [((free_gib(0) - headroom).max(0.0) * 512.0) as usize, ((free_gib(1) - headroom).max(0.0) * 512.0) as usize];
                ensure!(quota[0] + quota[1] >= len >> 21, "DSC_REPACK: node memory free {:.1} + {:.1} GiB (minus {headroom} GiB headroom each) < {:.1} GiB needed", free_gib(0), free_gib(1), len as f64 / (1u64 << 30) as f64);
                let mut map = vec![0u8; len >> 21];
                for k in 0..len >> 21 {
                    let mut n = k & 1;
                    if quota[n] == 0 {
                        n ^= 1;
                    }
                    quota[n] -= 1;
                    map[k] = n as u8;
                    let mask: u64 = 1 << n;
                    libc::syscall(libc::SYS_mbind, p.add(k << 21) as *mut libc::c_void, 2usize << 20, 1usize /* MPOL_PREFERRED: spill to the other node instead of swapping */, &mask as *const u64, 64usize, 0usize);
                }
                fe_ggq::cpu::set_node_map(map);
                p
            };
            fe_ggq::cpu::NODE_REGION.0.store(ptr as usize, std::sync::atomic::Ordering::Relaxed);
            fe_ggq::cpu::NODE_REGION.1.store(len, std::sync::atomic::Ordering::Relaxed);
            // pre-fault every 2 MiB page now, while memory is mostly free (touching it later, with the GGUF streaming through the page cache, falls back to 4 KiB pages)
            {
                use rayon::prelude::*;
                let pp = ptr as usize;
                let tp = rayon::ThreadPoolBuilder::new().num_threads(32).start_handler(|_| unsafe {
                    let mut set: libc::cpu_set_t = std::mem::zeroed();
                    for cpu in 0..64 {
                        libc::CPU_SET(cpu, &mut set);
                    }
                    libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
                }).build()?;
                tp.install(|| (0..len >> 21).into_par_iter().for_each(|k| unsafe { std::ptr::write_volatile((pp + (k << 21)) as *mut u8, 0) }));
            }
            {
                // every page must now be resident: a bound page that cannot be allocated is swapped out later and faults on every read
                let npg = len / 4096;
                let mut vec = vec![0u8; npg.min(1 << 24)];
                let (mut nonres, mut done) = (0usize, 0usize);
                while done < npg {
                    let n = (npg - done).min(vec.len());
                    if unsafe { libc::mincore(ptr.add(done * 4096) as *mut libc::c_void, n * 4096, vec.as_mut_ptr()) } == 0 {
                        nonres += vec[..n].iter().filter(|&&b| b & 1 == 0).count();
                    }
                    done += n;
                }
                // THP pre-fault touches one byte per 2 MiB; non-resident 4 KiB pages are expected only where THP failed: they are filled by the repack below
                eprintln!("pre-touch: {:.1} GiB of the {:.1} GiB region not yet resident (4 KiB fallbacks, filled by the repack)", nonres as f64 * 4096.0 / (1u64 << 30) as f64, len as f64 / (1u64 << 30) as f64);
            }
            struct SP(*mut u8);
            unsafe impl Send for SP {}
            unsafe impl Sync for SP {}
            let base = SP(ptr);
            let todo: Vec<usize> = (0..c.n_layer * ne).filter(|&i| !on[i]).collect();
            use rayon::prelude::*;
            let nthreads = std::env::var("DSC_REPACK_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(32usize);
            // the engine thread is pinned (CpuMoe::set_caller_cpu) and rayon threads inherit that mask: give the repack threads every CPU back
            let pool = rayon::ThreadPoolBuilder::new().num_threads(nthreads).start_handler(|_| unsafe {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                for cpu in 0..64 {
                    libc::CPU_SET(cpu, &mut set);
                }
                libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
            }).build()?;
            let bad = std::sync::atomic::AtomicUsize::new(0);
            // layer by layer: read the layer's three expert tensors into the page cache with big sequential faults (MADV_POPULATE_READ, one
            // thread each; random expert-sized faults from a cold file are IO bound), repack the layer's CPU experts in parallel, drop the pages
            let shard_files: Vec<std::fs::File> = fe_gguf::shard_paths(&mp).iter().filter_map(|f| std::fs::File::open(f).ok()).collect();
            for l in 0..c.n_layer {
                let rng = |t: &fe_gguf::TensorInfo| {
                    let d = g.data(t);
                    let s0 = (d.as_ptr() as usize) & !4095;
                    (s0, (d.as_ptr() as usize + d.len() - s0).div_ceil(4096) * 4096)
                };
                std::thread::scope(|sc| {
                    for j in 0..3 {
                        let (s0, ln) = rng(ts[l][j]);
                        sc.spawn(move || unsafe { libc::madvise(s0 as *mut libc::c_void, ln, 22 /* MADV_POPULATE_READ */) });
                    }
                });
                let es: Vec<usize> = todo.iter().copied().filter(|&i| i / ne == l).collect();
                pool.install(|| {
                    es.par_iter().for_each(|&i| {
                        let e = i % ne;
                        let b = &base;
                        for j in 0..3 {
                            let t = ts[l][j];
                            let src = g.slice(t, e);
                            let dst = unsafe { std::slice::from_raw_parts_mut(b.0.add(off[i][j]), rsz[l][j]) };
                            let (k, rows) = (t.ne0(), t.dims[1] as usize);
                            let rtj = if j == 2 { rty[l].1 } else { rty[l].0 };
                            let (orb, rrb) = (t.ty.row_bytes(k), crate::rep_row_bytes(rtj, k));
                            for r in 0..rows {
                                let ok = unsafe { crate::repack_row(rtj, t.ty, &src[r * orb..(r + 1) * orb], k, &mut dst[r * rrb..(r + 1) * rrb]) };
                                if !ok {
                                    bad.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                        }
                    });
                });
                for j in 0..3 {
                    let (s0, ln) = rng(ts[l][j]);
                    unsafe { libc::madvise(s0 as *mut libc::c_void, ln, libc::MADV_DONTNEED) };
                }
                // and out of the page cache: with 80 GiB of anonymous pages held the kernel would otherwise swap them out under the cache's pressure
                for fh in &shard_files {
                    use std::os::fd::AsRawFd;
                    unsafe { libc::posix_fadvise(fh.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
                }
            }
            ensure!(bad.load(std::sync::atomic::Ordering::Relaxed) == 0, "DSC_REPACK: weights outside the three magnitude levels");
            let kb: u64 = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default().lines().find(|l| l.starts_with("AnonHugePages")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok()).unwrap_or(0);
            eprintln!("repacked {} experts to planes: {:.1} GiB in {:.1}s (2 MiB pages alternate nodes; AnonHugePages {:.1} GiB)", todo.len(), len as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f32(), kb as f64 / (1u64 << 20) as f64);
            {
                // diagnostic: 4 KiB pages of the region that are not resident (they would fault on first read)
                let npg = len / 4096;
                let mut vec = vec![0u8; npg.min(1 << 24)];
                let mut nonres = 0usize;
                let mut done = 0usize;
                while done < npg {
                    let n = (npg - done).min(vec.len());
                    let r = unsafe { libc::mincore(ptr.add(done * 4096) as *mut libc::c_void, n * 4096, vec.as_mut_ptr()) };
                    if r == 0 {
                        nonres += vec[..n].iter().filter(|&&b| b & 1 == 0).count();
                    }
                    done += n;
                }
                eprintln!("mincore: {} of {} 4 KiB pages of the repacked region not resident", nonres, npg);
                // where: per 4 GiB bucket of the region
                let mut per = vec![0usize; (len >> 32) + 1];
                let mut d2 = 0usize;
                while d2 < npg {
                    let n = (npg - d2).min(vec.len());
                    if unsafe { libc::mincore(ptr.add(d2 * 4096) as *mut libc::c_void, n * 4096, vec.as_mut_ptr()) } == 0 {
                        for (i, &b) in vec[..n].iter().enumerate() {
                            if b & 1 == 0 {
                                per[((d2 + i) * 4096) >> 32] += 1;
                            }
                        }
                    }
                    d2 += n;
                }
                eprintln!("mincore: non-resident 4 KiB pages per 4 GiB of the region: {:?}", per);
            }
            {
                // the node each 2 MiB page really landed on (a full node spills its share to the other one)
                let mut map = vec![0u8; len >> 21];
                let mut moved = 0usize;
                for k in 0..len >> 21 {
                    let mut node: libc::c_int = -1;
                    let r = unsafe { libc::syscall(libc::SYS_get_mempolicy, &mut node as *mut libc::c_int, std::ptr::null::<u64>(), 0usize, ptr.add(k << 21) as *mut libc::c_void, 3usize /* MPOL_F_NODE | MPOL_F_ADDR */) };
                    map[k] = if r == 0 && node >= 0 { node as u8 & 1 } else { (k & 1) as u8 };
                    if map[k] as usize != (k & 1) {
                        moved += 1;
                    }
                }
                eprintln!("node map: {} of {} 2 MiB pages are not on their preferred node", moved, len >> 21);
                fe_ggq::cpu::set_node_map(map);
            }
            if let Ok(nm) = std::fs::read_to_string("/proc/self/numa_maps") {
                let key = format!("{:x}", ptr as usize);
                // the region is one mapping (split by the per-2-MiB policies): sum the nodes of every line inside [ptr, ptr + len)
                let (mut n0, mut n1) = (0u64, 0u64);
                for l in nm.lines() {
                    let a = l.split_whitespace().next().and_then(|v| usize::from_str_radix(v, 16).ok()).unwrap_or(0);
                    if a >= ptr as usize && a < ptr as usize + len {
                        for w in l.split_whitespace() {
                            if let Some(v) = w.strip_prefix("N0=") { n0 += v.parse::<u64>().unwrap_or(0); }
                            if let Some(v) = w.strip_prefix("N1=") { n1 += v.parse::<u64>().unwrap_or(0); }
                        }
                    }
                }
                let _ = key;
                eprintln!("numa_maps of the repacked region: node0 {:.1} GiB, node1 {:.1} GiB (pages as counted by the kernel)", n0 as f64 * 4096.0 / (1u64 << 30) as f64, n1 as f64 * 4096.0 / (1u64 << 30) as f64);
            }
            self.host = Some(HostRegion { ptr, len, off, rty, rsz, registered: false });
        }
        if std::env::var("DSC_HOSTCOPY").map_or(false, |v| v == "1") {
            let mut on = vec![false; c.n_layer * ne];
            for &(i, _) in ch1.iter().chain(ch0.iter()) {
                on[i] = true;
            }
            let mut off = vec![[usize::MAX; 3]; c.n_layer * ne];
            let mut len = 0usize;
            for i in 0..c.n_layer * ne {
                if !on[i] {
                    let l = i / ne;
                    for j in 0..3 {
                        off[i][j] = len;
                        len += al(ts[l][j].slice_bytes());
                    }
                }
            }
            let len = len.div_ceil(2 << 20) * (2 << 20);
            let t0 = Instant::now();
            let ptr = unsafe {
                let nm: u64 = 0b11;
                libc::syscall(libc::SYS_set_mempolicy, 3 /* MPOL_INTERLEAVE */, &nm as *const u64, 64u64);
                let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0);
                anyhow::ensure!(p != libc::MAP_FAILED, "mmap {len} bytes failed");
                libc::madvise(p, len, libc::MADV_HUGEPAGE);
                p as *mut u8
            };
            for i in 0..c.n_layer * ne {
                if !on[i] {
                    let (l, e) = (i / ne, i % ne);
                    for j in 0..3 {
                        let sl = g.slice(ts[l][j], e);
                        unsafe { std::ptr::copy_nonoverlapping(sl.as_ptr(), ptr.add(off[i][j]), sl.len()) };
                    }
                }
            }
            unsafe { libc::syscall(libc::SYS_set_mempolicy, 0, std::ptr::null::<u64>(), 0u64) };
            eprintln!("host copy of {} experts: {:.1} GiB in {:.1}s (anonymous, THP, interleaved)", on.iter().filter(|&&b| !b).count(),
                len as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f32());
            let mut registered = false;
            if std::env::var("DSC_PF_REG").is_ok_and(|v| v == "1") {
                let t1 = Instant::now();
                let r = unsafe { fe_cuda::ffi::cuMemHostRegister_v2(ptr as *mut std::ffi::c_void, len, 0x01) };
                eprintln!("cuMemHostRegister of {:.1} GiB: {} in {:.1}s", len as f64 / (1u64 << 30) as f64, r, t1.elapsed().as_secs_f32());
                registered = r == 0;
            }
            self.host = Some(HostRegion { ptr, len, off, rty: vec![], rsz: vec![], registered });
        }
        for (k, xc) in self.xcs.iter_mut().enumerate() {
            xc.ctx.bind()?;
            let (a0, r0) = upload(&chs[k + 1], used[k + 1], &zc_extra[k + 1])?;
            xc.arena = a0;
            xc.res = r0;
            self._ctx.bind()?;
        }
        if self.zc && ring_gb > 0.0 {
            let sb = self.zc_tot.iter().copied().max().unwrap_or(0);
            let nslots = ((ring_gb * (1u64 << 30) as f64) as usize / sb.max(1)).max(1);
            self.ring_slot_bytes = sb;
            for xc in self.xcs.iter() {
                xc.ctx.bind()?;
                self.zring.push(DeviceBuffer::<u8>::zeroed(nslots * sb)?);
                self.ring_expert.push(vec![usize::MAX; nslots]);
                self.ring_stamp.push(vec![0u64; nslots]);
            }
            self._ctx.bind()?;
            self.ring_of = vec![0u32; c.n_layer * ne];
            self.ring_promo = std::env::var("DSC_RING_PROMO").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
            self.promo_stage = Some(HostBuffer::alloc(6 * self.ring_promo.max(1))?);
            eprintln!("slot ring: {} slots of {:.1} MiB per expert card ({:.2} GiB each)", nslots, sb as f64 / 1048576.0, (nslots * sb) as f64 / (1u64 << 30) as f64);
        }
        self.ids_log = Some(HostBuffer::alloc(c.n_layer * RMAX * c.n_expert_used)?);
        self.heat_cnt = vec![0u64; c.n_layer * ne];
        if let Ok(p) = std::env::var("DSC_HEAT_LOG") {
            self.heat_path = Some(p);
            self.heat_acc = true;
        }
        if let Ok(p) = std::env::var("DSC_PASS_LOG") {
            self.pass_log = Some(std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(p)?));
        }
        Ok(())
    }

    /// The engine card context (for a side module that binds its own and must rebind this one).
    pub fn ctx(&self) -> std::sync::Arc<Context> {
        self._ctx.clone()
    }

    /// Roll lane `i` back to `len` tokens.
    pub fn truncate_lane(&mut self, i: usize, len: usize) {
        if i == self.cur_slot { self.ids_seen.truncate(len) } else { self.lane_ids[i].truncate(len) }
    }

    /// The DSpark taps of the last pass's first `rows` rows, to the host.
    pub fn taps_to_host(&self, rows: usize, out: &mut [u16]) -> Result<()> {
        memcpy_dtoh(&mut out[..rows * 3 * self.c.n_embd], self.taps.ptr())?;
        Ok(())
    }

    pub fn n_lanes(&self) -> usize {
        self.n_slots
    }

    /// Make lane `i` the sequence `step_rows` / `truncate` / `ids_seen` work on (only the token history moves; the caches are indexed by slot).
    pub fn activate(&mut self, i: usize) {
        if i == self.cur_slot {
            return;
        }
        std::mem::swap(&mut self.ids_seen, &mut self.lane_ids[self.cur_slot]);
        std::mem::swap(&mut self.ids_seen, &mut self.lane_ids[i]);
        self.cur_slot = i;
    }

    /// Tokens seen by lane `i`.
    pub fn lane_len(&self, i: usize) -> usize {
        if i == self.cur_slot { self.ids_seen.len() } else { self.lane_ids[i].len() }
    }

    pub fn lane_tokens(&self, i: usize) -> &[u32] {
        if i == self.cur_slot { &self.ids_seen } else { &self.lane_ids[i] }
    }

    /// Lane `dst` becomes a copy of lane `src` up to `len` positions (the shared prefix of two requests): every cache is positional, so the
    /// whole slot is copied (device to device) and the history cut to `len`; exact while `src` has not moved more than `max_rewind` past `len`.
    pub fn fork_lane(&mut self, src: usize, dst: usize, len: usize) -> Result<()> {
        ensure!(src != dst && src < self.n_slots && dst < self.n_slots, "fork_lane {src} -> {dst}");
        let have = self.lane_len(src);
        ensure!(len <= have && have - len <= self.max_rewind(), "fork_lane: source at {have} tokens, share point {len}");
        let toks = self.lane_tokens(src)[..len].to_vec();
        let (hd, idx) = (self.c.head_dim, self.c.idx_dim);
        let rl = self.ring_len;
        for l in 0..self.c.n_layer {
            let ne = self.entries[l];
            let cp = |b: &DeviceBuffer<u16>, per: usize| -> Result<()> {
                Ok(self.s.memcpy_dtod(b.ptr().add::<u16>(dst * per), b.ptr().add::<u16>(src * per), per * 2)?)
            };
            cp(&self.ring[l], rl * hd)?;
            if let Some(t) = &self.tail[l] {
                self.s.memcpy_dtod(t.ptr().add::<f32>(dst * rl * 2 * hd), t.ptr().add::<f32>(src * rl * 2 * hd), rl * 2 * hd * 4)?;
            }
            if let Some(b) = &self.plane[l] {
                cp(b, ne * hd)?;
            }
            if let Some(b) = &self.ikeys[l] {
                cp(b, ne * idx)?;
            }
        }
        self.s.sync()?;
        if dst == self.cur_slot {
            self.ids_seen = toks;
        } else {
            self.lane_ids[dst] = toks;
        }
        Ok(())
    }

    /// One pass over the rows of several lanes (`parts`: lane, its next consecutive tokens): the dense and expert work is batched over all rows,
    /// every row attends to its own lane's caches. Returns the logits of every row (in order) when asked. Rows total <= RMAX.
    pub fn step_multi(&mut self, parts: &[(usize, &[u32])], want_logits: bool) -> Result<Option<Vec<Vec<f32>>>> {
        let c = self.c;
        let d = c.n_embd;
        let nr: usize = parts.iter().map(|p| p.1.len()).sum();
        ensure!(nr >= 1 && nr <= RMAX, "step_multi: {nr} rows (1..={RMAX})");
        let t_all = Instant::now();
        self.split = Split::default();
        // positions and histories
        let mut rows: Vec<(usize, usize, u32)> = vec![]; // (lane, position, token)
        for &(lane, toks) in parts {
            ensure!(lane < self.n_slots, "lane {lane}");
            let h = if lane == self.cur_slot { &mut self.ids_seen } else { &mut self.lane_ids[lane] };
            let p0 = h.len();
            ensure!(p0 + toks.len() <= self.max_ctx, "lane {lane}: position {} >= max_ctx {}", p0 + toks.len(), self.max_ctx);
            for (i, &t) in toks.iter().enumerate() {
                rows.push((lane, p0 + i, t));
            }
            h.extend_from_slice(toks);
        }
        self.nr = nr;
        let emb = self.g.tensor("token_embd.weight").ok_or_else(|| anyhow!("token_embd"))?;
        let rb = emb.row_bytes();
        let mut h4: Vec<u16> = Vec::with_capacity(nr * 4 * d);
        let mut meta: Vec<u32> = Vec::with_capacity(nr * 2);
        let mut row = vec![0f32; d];
        for &(lane, pos, token) in &rows {
            refq::dequant(emb.ty, &self.g.data(emb)[token as usize * rb..(token as usize + 1) * rb], &mut row);
            let rowb: Vec<u16> = row.iter().map(|&v| refq::f32_to_bf16(v)).collect();
            for _ in 0..4 {
                h4.extend_from_slice(&rowb);
            }
            meta.extend_from_slice(&[lane as u32, pos as u32]);
        }
        memcpy_htod(self.h.ptr(), &h4)?;
        memcpy_htod(self.meta.ptr(), &meta)?;
        let w = c.engram_width();
        let nl = c.engram_layers.len();
        // every row's Engram reads of every Engram layer at once (parallel reads), one upload a layer
        let mut all = vec![0f32; nl * nr * w];
        {
            let reqs: Vec<(&[u32], usize)> = rows.iter().map(|&(lane, pos, _)| (if lane == self.cur_slot { &self.ids_seen[..] } else { &self.lane_ids[lane][..] }, pos)).collect();
            self.eng.vectors(&reqs, &mut all)?;
        }
        for li in 0..nl {
            memcpy_htod(self.evecs[li].ptr(), &all[li * nr * w..(li + 1) * nr * w])?;
        }
        let pos_max = rows.iter().map(|r| r.1).max().unwrap();
        let lr = (|| -> Result<()> {
            for l in 0..c.n_layer {
                if self.zc && !self.tap_layers {
                    self.layer_zc(l, pos_max + 1 - nr)?;
                } else {
                    self.layer(l, pos_max + 1 - nr)?;
                }
            }
            Ok(())
        })();
        lr?;
        let out = self.finish_rows(want_logits)?;
        if self.zc {
            self.zc_promote(nr)?;
        }
        self.tokens_done += 1;
        self.split.total_ms = t_all.elapsed().as_secs_f64() * 1e3;
        Ok(out)
    }

    /// Zero the caches IN PLACE (the buffers keep their addresses: the cached CUDA graphs hold them).
    pub fn reset(&mut self) -> Result<()> {
        self.ids_seen.clear();
        if self.n_slots > 1 {
            // lanes: the other lanes' caches must stay (positional caches: nothing stale is ever read)
            return Ok(());
        }
        // caches are positional: stale entries past the new length are never read, but clear for bit-identical reruns
        for b in &self.ring {
            self.s.memset_d8(b.ptr(), 0, b.bytes())?;
        }
        for b in self.tail.iter().flatten() {
            self.s.memset_d8(b.ptr(), 0, b.bytes())?;
        }
        for b in self.plane.iter().flatten() {
            self.s.memset_d8(b.ptr(), 0, b.bytes())?;
        }
        for b in self.ikeys.iter().flatten() {
            self.s.memset_d8(b.ptr(), 0, b.bytes())?;
        }
        self.s.sync()?;
        Ok(())
    }

    /// Test only: zero some of the positional caches without touching the sequence (bit 0 window ring, 1 compressor tail, 2 compressed
    /// planes, 3 indexer keys).
    pub fn clear_caches(&mut self, mask: u32) -> Result<()> {
        if mask & 1 != 0 {
            for b in &self.ring {
                self.s.memset_d8(b.ptr(), 0, b.bytes())?;
            }
        }
        if mask & 2 != 0 {
            for b in self.tail.iter().flatten() {
                self.s.memset_d8(b.ptr(), 0, b.bytes())?;
            }
        }
        if mask & 4 != 0 {
            for b in self.plane.iter().flatten() {
                self.s.memset_d8(b.ptr(), 0, b.bytes())?;
            }
        }
        if mask & 8 != 0 {
            for b in self.ikeys.iter().flatten() {
                self.s.memset_d8(b.ptr(), 0, b.bytes())?;
            }
        }
        self.s.sync()?;
        Ok(())
    }

    // ------------------------------------------------------------ small helpers

    /// y (f32 [w.n]) = w . (the quantized activation).
    fn gq(&self, w: &Wt, y: DevPtr) -> Result<()> {
        ensure!(w.k == self.act.k, "gemv k {} != activation k {}", w.k, self.act.k);
        if self.act.rows > RMAX && self.pf_mmq && w.n % 64 == 0 && w.k % 128 == 0 {
            // many rows: the int8 GEMM reads the weights once (the wide GEMVs re-read them every 24 rows)
            if self.ggq.mmq(&self.s, w.ty, y, w.n, &self.act, w.ptr(), w.rb, w.n, false).is_ok() {
                return Ok(());
            }
        }
        self.ggq.gemv(&self.s, w.ty, w.ptr(), w.rb, w.n, &self.act, y, false, None)
    }

    /// y (f32 [w.n]) = w . xf (f32 row), BF16 weights.
    fn gb(&self, w: &Wt, xf: DevPtr, xb: DevPtr, y: DevPtr) -> Result<()> {
        ensure!(w.ty == GgmlType::BF16, "gb: {} weights", w.ty.name());
        if self.nr > RMAX && self.pf_mmq && w.n % 64 == 0 && self.ggq.gemm(&self.s, GgmlType::BF16, y, w.n, xb, w.k, w.ptr(), w.rb, self.nr, w.n, w.k, false).is_ok() {
            return Ok(());
        }
        self.ggq.gemv_bf16(&self.s, w.ptr(), w.n, w.k, xf, w.k, self.nr, y, false)
    }

    fn f2b(&self, dst: DevPtr, src: DevPtr, n: usize) -> Result<()> {
        self.ggq.f32_to_bf16(&self.s, dst, n, src, n, self.nr, n)
    }

    fn b2f(&self, dst: DevPtr, src: DevPtr, n: usize) -> Result<()> {
        unsafe { self.s.launch(&self.k.b2f, g1(n * self.nr), 256, 0, (dst, src, (n * self.nr) as u32))? };
        Ok(())
    }

    /// Debug tap (DSC_DBG_DIR + DSC_DBG_LAYER): write `n` values of a device buffer as f32.
    fn dbg(&self, l: usize, name: &str, p: DevPtr, n: usize, bf16: bool) -> Result<()> {
        let (Ok(dir), Ok(dl)) = (std::env::var("DSC_DBG_DIR"), std::env::var("DSC_DBG_LAYER")) else { return Ok(()) };
        if dl.parse::<usize>().ok() != Some(l) {
            return Ok(());
        }
        self.s.sync()?;
        let v: Vec<f32> = if bf16 {
            let mut b = vec![0u16; n];
            memcpy_dtoh(&mut b, p)?;
            b.iter().map(|&x| refq::bf16_to_f32(x)).collect()
        } else {
            let mut b = vec![0f32; n];
            memcpy_dtoh(&mut b, p)?;
            b
        };
        std::fs::write(format!("{dir}/eng_L{l}_{name}.f32"), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        Ok(())
    }

    fn rmsnorm(&self, y: DevPtr, x: DevPtr, w: DevPtr, dim: usize) -> Result<()> {
        let bs = if dim >= 1024 { 256 } else { 128 };
        unsafe { self.s.launch(&self.k.rmsnorm, self.nr as u32, bs, 0, (y, x, 0u32, w, dim as u32, self.c.eps, dim as u32, dim as u32))? };
        Ok(())
    }

    fn hc_mix(&self, hc: &Hc, out: DevPtr) -> Result<()> {
        let c = self.c;
        unsafe {
            self.s.launch(&self.k.hc_mix, self.nr as u32, 768, 0, (out, self.h.ptr(), hc.fnw.ptr(), hc.scale.ptr(), hc.base.ptr(),
                (c.hc * c.n_embd) as u32, c.eps, c.hc_eps, c.hc_iters as u32))?
        };
        Ok(())
    }

    /// hc mixes into `out` and the pre-norm of the same streams with the earlier mixes `pre` (split-K over 40 blocks, the pre-norm as
    /// one more block of the partials launch; the same bits as hc_mix then pre_norm).
    fn hc_mix_pn(&self, hc: &Hc, out: DevPtr, pre: DevPtr, pre_ld: u32, w: DevPtr) -> Result<()> {
        let c = self.c;
        let dim4 = c.hc * c.n_embd;
        let nks = dim4 / MIX_SLICE;
        if self.hc_simple {
            self.hc_mix(hc, out)?;
            return self.pre_norm(pre, pre_ld, w);
        }
        unsafe {
            self.s.launch(&self.k.hc_part_pn, (nks as u32 + 1, self.nr as u32), 768, 0, (self.mixp.ptr(), self.h.ptr(), hc.fnw.ptr(), dim4 as u32,
                MIX_SLICE as u32, self.x.ptr(), pre, pre_ld, w, c.n_embd as u32, c.eps))?;
            self.s.launch(&self.k.hc_fin, self.nr as u32, 32, 0, (out, self.mixp.ptr(), nks as u32, hc.scale.ptr(), hc.base.ptr(), dim4 as u32,
                c.eps, c.hc_eps, c.hc_iters as u32))?;
        }
        Ok(())
    }

    fn pre_norm(&self, pre: DevPtr, pre_ld: u32, w: DevPtr) -> Result<()> {
        unsafe { self.s.launch(&self.k.hc_pre_norm, self.nr as u32, 256, 0, (self.x.ptr(), self.h.ptr(), pre, pre_ld, w, self.c.n_embd as u32, self.c.eps))? };
        Ok(())
    }

    fn post(&mut self, mix: DevPtr) -> Result<()> {
        let d = self.c.n_embd;
        unsafe { self.s.launch(&self.k.hc_post, (g1(d), self.nr as u32), 256, 0, (self.h2.ptr(), self.part.ptr(), self.h.ptr(), mix, d as u32))? };
        std::mem::swap(&mut self.h, &mut self.h2);
        Ok(())
    }

    // ------------------------------------------------------------ the forward

    /// Feed one token at the next position; returns the logits (f32 [vocab]) when `want_logits`.
    pub fn step(&mut self, token: u32, want_logits: bool) -> Result<Option<Vec<f32>>> {
        Ok(self.step_rows(&[token], want_logits)?.map(|mut v| v.remove(0)))
    }

    /// How many positions the sequence may be rolled back (`truncate`) and stay exact: the window ring and the compressor tail are
    /// rings of `ring_len` positions, so rows older than ring_len - (window | block) are gone once the sequence has moved on.
    pub fn max_rewind(&self) -> usize {
        self.ring_len.saturating_sub(self.c.window.max(128) + 8)
    }

    /// Chunked prompt feed: `tokens` through `step_rows` in RMAX-row passes (the multi-row path is row-invariant: same bits as one
    /// `step` per token); the logits of the LAST token are returned (`want_logits`) from a single-row final pass.
    pub fn prefill_chunked(&mut self, tokens: &[u32], want_logits: bool) -> Result<Option<Vec<f32>>> {
        ensure!(!tokens.is_empty(), "prefill_chunked: empty");
        let np = tokens.len() - 1;
        let mut i = 0;
        while i < np {
            let c = self.chunk_for(np - i);
            self.step_rows(&tokens[i..i + c], false)?;
            i += c;
        }
        self.step(tokens[np], want_logits)
    }

    /// Rows of the next prompt pass when `left` prompt rows remain: a big chunk (DSC_PF_ROWS) when at least DSC_PF_MIN remain.
    pub fn chunk_for(&self, left: usize) -> usize {
        if self.pf.is_some() && self.pf_n > RMAX && left >= self.pf_min { left.min(self.pf_n) } else { left.min(prefill_rows()) }
    }

    /// Rows one prompt pass may take: DSC_PF_ROWS when the big path is on, else RMAX.
    pub fn pf_chunk(&self) -> usize {
        if self.pf.is_some() { self.pf_n } else { RMAX }
    }

    /// Roll the sequence back to `len` tokens (the verify pass's rejected rows): every cache is positional (window ring, compressor
    /// tail, compressed planes, indexer keys: entries past the length are never read and are rewritten by the next rows).
    pub fn truncate(&mut self, len: usize) {
        self.ids_seen.truncate(len);
    }

    /// One pass over `tokens.len()` (1..=RMAX) consecutive positions of the one sequence (the verify rows of a drafter, or a
    /// prompt chunk): every kernel takes the rows together and keeps each row's bits (a row alone == the row inside the batch).
    /// Returns the logits of every row (f32 [vocab] each) when `want_logits`. The DSpark taps of the rows are in `self.taps`.
    pub fn step_rows(&mut self, tokens: &[u32], want_logits: bool) -> Result<Option<Vec<Vec<f32>>>> {
        let c = self.c;
        let d = c.n_embd;
        let nr = tokens.len();
        ensure!(nr >= 1 && nr <= self.pf_n, "step_rows: {nr} rows (1..={})", self.pf_n);
        ensure!(nr <= RMAX || !want_logits, "step_rows: logits only for <= {RMAX} rows");
        let pos0 = self.ids_seen.len();
        ensure!(pos0 + nr <= self.max_ctx, "position {} >= max_ctx {}", pos0 + nr, self.max_ctx);
        if self.pipe_rows > 0 && nr >= self.pipe_rows && !self.tap_layers && self.trace.is_none() && self.sc2.is_some() {
            return self.step_rows_pipe(tokens, want_logits);
        }
        self.ids_seen.extend_from_slice(tokens);
        let t_all = Instant::now();
        self.split = Split::default();
        if nr > RMAX {
            let r = self.step_big(tokens, pos0);
            self.tokens_done += 1;
            self.split.total_ms = t_all.elapsed().as_secs_f64() * 1e3;
            return r.map(|_| None);
        }
        self.begin_rows(tokens, pos0)?;
        if self.tap_layers {
            self.layer_h.clear();
        }
        if nr > RMAX {
            self.pf_begin(nr)?;
        }
        let lr = (|| -> Result<()> {
            for l in 0..c.n_layer {
                if nr > RMAX {
                    unreachable!("big chunks take step_big");
                } else {
                    if self.zc && !self.tap_layers {
                        self.layer_zc(l, pos0)?;
                    } else {
                        self.layer(l, pos0)?;
                    }
                }
                if let (Ok(dir), Ok(nl)) = (std::env::var("DSC_PF_DUMP"), std::env::var("DSC_PF_DUMPL")) {
                    if l < nl.parse::<usize>().unwrap_or(0) {
                        self.s.sync()?;
                        let mut v = vec![0u16; nr * 4 * d];
                        memcpy_dtoh(&mut v, self.h.ptr())?;
                        use std::io::Write;
                        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/h_L{l}.bin"))?;
                        f.write_all(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                    }
                }
                if self.tap_layers {
                    self.s.sync()?;
                    let mut v = vec![0u16; 4 * d];
                    memcpy_dtoh(&mut v, self.h.ptr().add::<u16>((nr - 1) * 4 * d))?;
                    self.layer_h.push(v.iter().map(|&b| refq::bf16_to_f32(b)).collect());
                }
            }
            Ok(())
        })();
        if nr > RMAX {
            self.pf_end();
        }
        self.split.ph[0] = t_all.elapsed().as_secs_f64() * 1e3;

        lr?;
        let out = self.finish_rows(want_logits)?;
        if self.zc && nr <= RMAX {
            self.zc_promote(nr)?;
        }
        if nr > RMAX && std::env::var("DSC_PF_VERBOSE").is_ok() {
            let st = self.pf_status()?;
            let (ba, bb) = self.pf_take_stats();
            eprintln!("pf chunk {nr} rows: {:.2}s, streamed A {:.2} GB, B {:.2} GB, fx overflow {st}; attention+router {:.2}s, moe {:.2}s, compute waited {:.2}s (both cards)",
                t_all.elapsed().as_secs_f64(), ba as f64 / 1e9, bb as f64 / 1e9, self.split.ph[0] / 1e3, self.split.ph[2] / 1e3, prefill::WAIT_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
        }
        self.tokens_done += 1;
        self.split.total_ms = t_all.elapsed().as_secs_f64() * 1e3;
        self.split.ph[5] = self.split.total_ms - self.split.ph[0];
        if self.zc && nr <= RMAX {
            self.pass_note(nr, pos0, want_logits);
        }
        if self.cardtime && self.zc {
            for k in 0..self.zc_ev.len() {
                let mut t = 0f64;
                for l in 0..self.zc_ev[k].len() {
                    if self.zc_used[k][l] {
                        self.zc_ev[k][l].1.sync()?;
                        t += self.zc_ev[k][l].1.elapsed_since(&self.zc_ev[k][l].0)? as f64;
                        self.zc_used[k][l] = false;
                    }
                }
                self.split.ph[if k == 0 { 4 } else { 9 }] = t;
            }
        }
        Ok(out)
    }

    /// Embedding rows + positions + Engram vectors of `tokens` (positions pos0 ..; `ids_seen` already holds them) into the scratch.
    fn begin_rows(&mut self, tokens: &[u32], pos0: usize) -> Result<()> {
        let c = self.c;
        let d = c.n_embd;
        let nr = tokens.len();
        self.nr = nr;
        // embedding: 4 identical bf16 streams a row (the row is dequantised on the host)
        let emb = self.g.tensor("token_embd.weight").ok_or_else(|| anyhow!("token_embd"))?;
        let rb = emb.row_bytes();
        let mut h4: Vec<u16> = Vec::with_capacity(nr * 4 * d);
        let mut meta: Vec<u32> = Vec::with_capacity(nr * 2);
        let mut row = vec![0f32; d];
        for (i, &token) in tokens.iter().enumerate() {
            refq::dequant(emb.ty, &self.g.data(emb)[token as usize * rb..(token as usize + 1) * rb], &mut row);
            let rowb: Vec<u16> = row.iter().map(|&v| refq::f32_to_bf16(v)).collect();
            for _ in 0..4 {
                h4.extend_from_slice(&rowb);
            }
            meta.extend_from_slice(&[self.cur_slot as u32, (pos0 + i) as u32]);
        }
        memcpy_htod(self.h.ptr(), &h4)?;
        memcpy_htod(self.meta.ptr(), &meta)?;
        let t_e = Instant::now();
        let w = c.engram_width();
        let nl = c.engram_layers.len();
        // every row's Engram reads of every Engram layer at once (parallel reads), one upload a layer
        let mut all = vec![0f32; nl * nr * w];
        {
            let reqs: Vec<(&[u32], usize)> = (0..nr).map(|i| (&self.ids_seen[..], pos0 + i)).collect();
            self.eng.vectors(&reqs, &mut all)?;
        }
        for li in 0..nl {
            memcpy_htod(self.evecs[li].ptr(), &all[li * nr * w..(li + 1) * nr * w])?;
        }
        self.split.engram_ms += t_e.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// The head (logits of every row when `want_logits`) after the last layer; syncs the stream.
    fn finish_rows(&mut self, want_logits: bool) -> Result<Option<Vec<Vec<f32>>>> {
        let c = self.c;
        let (d, nr) = (c.n_embd, self.nr);
        Ok(if want_logits {
            // head: collapse with the last mix's pre, norm, output
            self.pre_norm(self.mix_f.ptr(), 24, self.out_norm.ptr())?;
            let x = self.x.ptr();
            q8!(self, x, true, d);
            self.gq(&self.out_w, self.logits.ptr())?;
            self.s.sync()?;
            let mut lg = vec![0f32; nr * c.n_vocab];
            memcpy_dtoh(&mut lg, self.logits.ptr())?;
            Some(lg.chunks(c.n_vocab).map(|v| v.to_vec()).collect())
        } else {
            self.s.sync()?;
            None
        })
    }

    /// Segment A of layer l (GPU only, graph-capturable): Engram, attention, its hc post, the FFN mixes and the router pick.
    /// `bp` = the position bucket (the indexer scores / top-k cover bucket / ratio entries; the row's own length masks the rest).
    fn seg_a(&mut self, l: usize, bp: usize) -> Result<usize> {
        let mut swaps = 0;
        let c = self.c;
        let (d, hd) = (c.n_embd, c.head_dim);
        let role = self.layers[l].role.clone();
        let tab = if role.ratio > 0 { self.tab_comp.ptr() } else { self.tab_plain.ptr() };
        let meta = self.meta.ptr();
        if l + 3 >= c.n_layer {
            unsafe { self.s.launch(&self.k.tap, (g1(d), self.nr as u32), 256, 0, (self.taps.ptr().add::<u16>(self.tap_off), (3 * d) as u32, ((l + 3 - c.n_layer) * d) as u32, self.h.ptr(), d as u32, DevPtr(0)))? };
        }
        if let Some(li) = c.engram_layers.iter().position(|&x| x == l) {
            let e = self.layers[l].eng.as_ref().unwrap();
            let w = c.engram_width();
            self.ggq.quantize_q8_1(&self.s, self.evecs[li].ptr(), w, false, self.nr, w, &mut self.act)?;
            ensure!(e.kv.k == self.act.k, "engram_kv k {} != {}", e.kv.k, self.act.k);
            self.ggq.gemv(&self.s, e.kv.ty, e.kv.ptr(), e.kv.rb, e.kv.n, &self.act, self.ekv.ptr(), false, None)?;
            unsafe {
                self.s.launch(&self.k.egate, (self.nr as u32, c.hc as u32), 256, 0, (self.h2.ptr(), self.h.ptr(), self.ekv.ptr(), DevPtr(0),
                    e.qk.ptr(), self.nr as u32, c.hc as u32, d as u32, 1u32, c.eps))?
            };
            std::mem::swap(&mut self.h, &mut self.h2);
            swaps += 1;
        }
        // ---- attention
        let (pre, pre_ld) = if l == 0 { (self.pre0.ptr(), 4u32) } else { (self.mix_f.ptr(), 24u32) };
        self.hc_mix_pn(&self.layers[l].hc_attn, self.mix_a.ptr(), pre, pre_ld, self.layers[l].attn_norm.ptr())?;
        let x = self.x.ptr();
        self.dbg(l, "attn_x", self.x.ptr(), d, true)?;
        // x (bf16) -> q8 once for wq_a, wkv; f32 copy for the BF16 compressor linears
        q8!(self, x, true, d);
        let ly = &self.layers[l];
        self.gq(&ly.q_a, self.t32.ptr())?;
        self.f2b(self.qr_raw.ptr(), self.t32.ptr(), c.q_lora)?;
        self.gq(&ly.kv, self.t32.ptr())?;
        self.f2b(self.kv_raw.ptr(), self.t32.ptr(), hd)?;
        self.rmsnorm(self.qr.ptr(), self.qr_raw.ptr(), ly.q_a_norm.ptr(), c.q_lora)?;
        // wq_b from qr (bf16)
        q8!(self, self.qr.ptr(), true, c.q_lora);
        self.gq(&ly.q_b, self.t32.ptr())?;
        self.f2b(self.q.ptr(), self.t32.ptr(), c.q_dim())?;
        self.dbg(l, "qr", self.qr.ptr(), c.q_lora, true)?;
        unsafe {
            self.s.launch(&self.k.rope, (self.nr as u32, c.n_head as u32), 32, 0, (self.q.ptr(), c.q_dim() as u32, hd as u32,
                (hd - c.n_rot) as u32, meta, tab, 0u32))?;
            self.s.launch(&self.k.kv_norm_window, self.nr as u32, 32, 0, (self.kv.ptr(), self.kv_raw.ptr(), ly.kv_norm.ptr(), c.eps,
                self.ring[l].ptr(), (self.ring_len * hd) as u32, self.ring_len as u32, meta, tab))?;
        }
        self.dbg(l, "q", self.q.ptr(), c.q_dim(), true)?;
        // compressor + indexer keys
        if role.kv_source {
            let comp = ly.comp.as_ref().unwrap();
            self.b2f(self.xf.ptr(), x, d)?;
            self.gb(&comp.wkv, self.xf.ptr(), x, self.kvc.ptr())?;
            if let Some(gw) = &comp.wgate {
                self.gb(gw, self.xf.ptr(), x, self.sc.ptr())?;
            }
            let tail = self.tail[l].as_ref().unwrap().ptr();
            let ts = (self.ring_len * 2 * hd) as u32;
            unsafe {
                self.s.launch(&self.k.comp_stash, self.nr as u32, 256, 0, (tail, ts, self.ring_len as u32, self.kvc.ptr(), self.sc.ptr(), meta,
                    role.ratio, hd as u32))?;
                self.s.launch(&self.k.comp_pair, self.nr as u32, 256, 0, (self.latpre.ptr(), tail, ts, self.ring_len as u32, meta, role.ratio,
                    hd as u32))?;
            }
            self.rmsnorm(self.lat.ptr(), self.latpre.ptr(), comp.norm.ptr(), hd)?;
            let ne = self.entries[l];
            unsafe {
                self.s.launch(&self.k.comp_store, self.nr as u32, 32, 0, (self.plane[l].as_ref().unwrap().ptr(), (ne * hd) as u32,
                    self.lat.ptr(), meta, role.ratio, tab, hd as u32))?;
            }
            if let Some(ix) = &ly.idx {
                if let (Some(wk), Some(kn)) = (&ix.wk, &ix.k_norm) {
                    self.b2f(self.latf.ptr(), self.lat.ptr(), hd)?;
                    self.gb(wk, self.latf.ptr(), self.lat.ptr(), self.t32.ptr())?;
                    self.f2b(self.ik_raw.ptr(), self.t32.ptr(), c.idx_dim)?;
                    self.rmsnorm(self.ik.ptr(), self.ik_raw.ptr(), kn.ptr(), c.idx_dim)?;
                    unsafe {
                        self.s.launch(&self.k.ikey_store, self.nr as u32, 32, 0, (self.ikeys[l].as_ref().unwrap().ptr(), (ne * c.idx_dim) as u32,
                            self.ik.ptr(), meta, role.ratio, tab))?;
                    }
                }
            }
        }
        // selection (index sources)
        let ratio = role.ratio as usize;
        if role.index_source {
            let ix = ly.idx.as_ref().unwrap();
            let src = role.kv_from.unwrap();
            let ne = self.entries[src];
            self.b2f(self.qrf.ptr(), self.qr.ptr(), c.q_lora)?;
            self.gb(&ix.wq_b, self.qrf.ptr(), self.qr.ptr(), self.t32.ptr())?;
            self.f2b(self.iqb.ptr(), self.t32.ptr(), c.idx_heads * c.idx_dim)?;
            unsafe {
                self.s.launch(&self.k.idx_query, (self.nr as u32, c.idx_heads as u32), 32, 0, (self.iqf.ptr(), self.iqb.ptr(), meta, tab,
                    c.idx_heads as u32))?;
            }
            // head weights: bf16(x @ proj^T) * idx_dim^-0.5 * heads^-0.5 (rounded to bf16, as the reference)
            self.gb(&ix.proj, self.xf.ptr(), x, self.iw.ptr())?;
            let sc = (c.idx_dim as f32).powf(-0.5) * (c.idx_heads as f32).powf(-0.5);
            unsafe { self.s.launch(&self.k.round_scale, g1(self.nr * c.idx_heads), 256, 0, (self.iw.ptr(), sc, (self.nr * c.idx_heads) as u32))? };
            let ld = self.max_ctx as u32;
            let nmax = (bp / ratio).min(self.max_ctx / ratio.max(1));
            let ikeys = self.ikeys[src].as_ref().unwrap().ptr();
            let mask_ld = self.max_ctx.div_ceil(c.cand_block.max(1)) as u32;
            let mask = if role.cand_masked { self.cand_mask.ptr() } else { DevPtr(0) };
            if self.pf_idxtc && self.nr > RMAX {
                // tensor-core scores (bf16 q on the FP4 grid is exact); the bf16 copy goes to iqb, free after idx_query
                self.f2b(self.iqb.ptr(), self.iqf.ptr(), c.idx_heads * c.idx_dim)?;
                unsafe {
                    self.s.launch(&self.k.idx_score_tc, (nmax.div_ceil(128).max(1) as u32, self.nr as u32), 256, 0, (self.score.ptr(), ld,
                        self.iqb.ptr(), self.iw.ptr(), ikeys, (ne * c.idx_dim) as u32, ne as u32, meta, role.ratio, mask, mask_ld))?;
                }
            } else {
            unsafe {
                self.s.launch(&self.k.idx_score, (nmax.div_ceil(128).max(1) as u32, self.nr as u32), 128, 0, (self.score.ptr(), ld,
                    self.iqf.ptr(), self.iw.ptr(), ikeys, (ne * c.idx_dim) as u32, meta, role.ratio, mask, mask_ld))?;
            }
            }
            if role.cand_source {
                let nb = nmax.div_ceil(c.cand_block);
                unsafe {
                    self.s.launch(&self.k.block_max, self.nr as u32, 256, 0, (self.bmax.ptr(), mask_ld, self.score.ptr(), ld, meta, role.ratio,
                        c.cand_block as u32))?;
                    self.s.launch(&self.k.topk, self.nr as u32, 1024, 0, (self.cand_sel.ptr(), self.bmax.ptr(), mask_ld, nb as u32,
                        c.cand_blocks as u32))?;
                    self.s.launch(&self.k.sel_to_mask, self.nr as u32, 256, 0, (self.cand_mask.ptr(), mask_ld, self.cand_sel.ptr(),
                        c.cand_blocks as u32))?;
                }
            }
            unsafe {
                self.s.launch(&self.k.topk, self.nr as u32, 1024, 0, (self.sel.ptr(), self.score.ptr(), ld, nmax as u32, c.idx_top_k as u32))?;
            }
        }
        // attention core
        let (plane, ps) = match role.kv_from {
            Some(src) if ratio > 0 => (self.plane[src].as_ref().unwrap().ptr(), (self.entries[src] * hd) as u32),
            _ => (DevPtr(0), 0),
        };
        if self.pf_mq && self.nr > RMAX {
            // tensor-core MQA attention, 32 heads a launch (each key tile read once for all of them)
            let gs = (self.nr * 8 * hd) as u32;
            for half in 0..2usize {
                unsafe {
                    self.s.launch(&self.k.attn_mq, (8u32, self.nr as u32), 512, 0, (self.pmq_o.ptr(), self.pmq_ml.ptr(),
                        self.q.ptr().add::<u16>(half * 32 * hd), c.q_dim() as u32, self.ring[l].ptr(), (self.ring_len * hd) as u32,
                        self.ring_len as u32, c.window as u32, plane, ps, self.sel.ptr(), c.idx_top_k as u32, meta, (hd as f32).powf(-0.5),
                        self.floor.ptr(), 0u32))?;
                    self.s.launch(&self.k.attn_mq_fin, (32u32, self.nr as u32), 128, 0, (self.o.ptr().add::<u16>(half * 4 * gs as usize), gs,
                        self.pmq_o.ptr(), self.pmq_ml.ptr(), meta, ly.sinks.ptr().byte_add(half * 32 * 4), tab))?;
                }
            }
        } else {
        unsafe {
            self.s.launch(&self.k.attn16, (c.n_head as u32, self.nr as u32), 512, 0, (self.o.ptr(), (self.nr * 8 * hd) as u32, self.q.ptr(),
                c.q_dim() as u32, self.ring[l].ptr(), (self.ring_len * hd) as u32, self.ring_len as u32, c.window as u32, plane, ps,
                self.sel.ptr(), c.idx_top_k as u32, meta, ly.sinks.ptr(), (hd as f32).powf(-0.5), tab, self.floor.ptr(), 0u32))?;
        }
        }
        self.dbg(l, "attn_o", self.o.ptr(), c.q_dim(), true)?;
        // wo_a: 8 groups of 4096 -> 1024, then wo_b
        let gin = c.q_dim() / c.o_groups;
        if self.grp_wo && self.nr <= 8 && c.o_groups * self.nr <= self.act.max_rows && ly.wo_a.ty == GgmlType::Q5K && ly.wo_a.k == gin {
            // all groups in one launch: o is [group][row][gin] = groups * nr rows of gin; the f32 result lands as [row][group * o_lora]
            self.ggq.quantize_q8_1(&self.s, self.o.ptr(), gin, true, c.o_groups * self.nr, gin, &mut self.act)?;
            self.ggq.gemv_groups_q5k(&self.s, ly.wo_a.ptr(), ly.wo_a.rb, c.o_lora * ly.wo_a.rb, c.o_lora, c.o_groups, &self.act, self.nr, self.t32.ptr(), c.o_groups * c.o_lora)?;
            self.f2b(self.ub.ptr(), self.t32.ptr(), c.o_groups * c.o_lora)?;
        } else {
            for gi in 0..c.o_groups {
                q8!(self, self.o.ptr().add::<u16>(gi * self.nr * gin), true, gin);
                let wg = ly.wo_a.ptr().byte_add(gi * c.o_lora * ly.wo_a.rb);
                if !(self.nr > RMAX && self.pf_mmq && self.ggq.mmq(&self.s, ly.wo_a.ty, self.t32.ptr(), c.o_lora, &self.act, wg, ly.wo_a.rb, c.o_lora, false).is_ok()) {
                    self.ggq.gemv(&self.s, ly.wo_a.ty, wg, ly.wo_a.rb, c.o_lora, &self.act, self.t32.ptr(), false, None)?;
                }
                self.f2b(self.ug.ptr(), self.t32.ptr(), c.o_lora)?;
                unsafe {
                    self.s.launch(&self.k.copy_rows, (self.nr as u32, g1(c.o_lora)), 256, 0, (self.ub.ptr().add::<u16>(gi * c.o_lora),
                        (c.o_groups * c.o_lora) as u32, self.ug.ptr(), c.o_lora as u32, c.o_lora as u32))?;
                }
            }
        }
        q8!(self, self.ub.ptr(), true, c.o_groups * c.o_lora);
        self.gq(&ly.wo_b, self.part.ptr())?;
        self.dbg(l, "attn_out", self.part.ptr(), d, false)?;
        self.post(self.mix_a.ptr())?;
        swaps += 1;
        // ---- MoE
        let ly = &self.layers[l];
        self.hc_mix_pn(&ly.hc_ffn, self.mix_f.ptr(), self.mix_a.ptr(), 24, ly.ffn_norm.ptr())?;
        let x = self.x.ptr();
        self.dbg(l, "ffn_x", x, d, true)?;
        let nx = c.n_expert_used;
        self.b2f(self.xf.ptr(), x, d)?;
        self.gb(&ly.gate_w, self.xf.ptr(), x, self.rlogits.ptr())?;
        unsafe {
            self.s.launch(&self.k.router_pick, self.nr as u32, 128, 0, (self.ids.ptr(), self.wts.ptr(), self.rscores.ptr(), self.rlogits.ptr(),
                ly.gate_b.ptr(), 1u32, c.n_expert as u32, nx as u32, c.weights_scale))?;
        }
        Ok(swaps)
    }

    /// Segment B (GPU only): the shared expert and this card's resident routed experts (ids from the router on the card).
    fn seg_b(&mut self, l: usize) -> Result<usize> {
        let c = self.c;
        let (d, nx) = (c.n_embd, c.n_expert_used);
        let ly = &self.layers[l];
        let x = self.x.ptr();
        q8!(self, x, true, d);
        self.gq(&ly.sh_gate, self.sh_g.ptr())?;
        self.gq(&ly.sh_up, self.sh_u.ptr())?;
        if let Some(r) = self.res[l].as_ref() {
            self.ggq.set_swiglu_limit(c.clamp_exp[l]);
            let tabs = [r.gtab.ptr(), r.utab.ptr(), r.dtab.ptr()];
            let staged = if self.zc_hi > 0 && self.stage_on { self.stage.as_ref() } else { None };
            if let Some(sg) = staged {
                self.ggq.cold_begin(&self.s, self.ids.ptr(), self.nr * nx, tabs, self.zc_lo, self.zc_hi, sg.counter.ptr(), sg.slot_e.ptr(), sg.orig.ptr(), sg.buf.ptr(), sg.slot_bytes, sg.nslots, self.zc_sz[l])?;
            }
            self.ggq.moe_gu_q8(&self.s, r.ty_gu, r.gtab.ptr(), r.utab.ptr(), r.rb_gu, self.ids.ptr(), self.nr, nx, &self.act, c.n_ff_exp,
                &mut self.hq, None, None)?;
            self.ggq.moe_down(&self.s, r.ty_d, r.dtab.ptr(), r.rb_d, self.ids.ptr(), self.nr, nx, &self.hq, d, self.y6.ptr(), None)?;
            if let Some(sg) = staged {
                self.ggq.cold_end(&self.s, tabs, sg.counter.ptr(), sg.slot_e.ptr(), sg.orig.ptr(), sg.nslots)?;
            }
        }
        self.f2b(self.sh_gb.ptr(), self.sh_g.ptr(), c.n_ff_exp)?;
        self.f2b(self.sh_ub.ptr(), self.sh_u.ptr(), c.n_ff_exp)?;
        unsafe {
            self.s.launch(&self.k.swiglu, (g1(c.n_ff_exp), self.nr as u32), 256, 0, (self.sh_h.ptr(), self.sh_gb.ptr(), self.sh_ub.ptr(),
                DevPtr(0), c.n_ff_exp as u32, c.clamp_shexp[l]))?;
        }
        q8!(self, self.sh_h.ptr(), true, c.n_ff_exp);
        self.gq(&ly.sh_down, self.part.ptr())?;
        Ok(0)
    }

    /// Segment C (GPU only): the six routed pairs (this card's rows, the other card's and the CPU's from `ycpu` / `slot`)
    /// added to the shared expert in slot order, then the FFN hc post.
    fn seg_c(&mut self, _l: usize) -> Result<usize> {
        let (d, nx) = (self.c.n_embd, self.c.n_expert_used);
        self.ggq.moe_combine_cpu(&self.s, self.part.ptr(), self.y6.ptr(), self.wts.ptr(), self.nr, nx, d, true, self.mb_y.as_ref().map_or(self.ycpu.ptr(), |m| m.dev()), self.slot.ptr())?;
        self.post(self.mix_f.ptr())?;
        Ok(1)
    }

    /// Run segment `kind` (0 = A, 1 = B, 2 = C) of layer l: as a cached CUDA graph (captured on first use; the host-side h / h2
    /// pointer swaps of the segment are replayed on a cache hit) or directly.
    fn seg(&mut self, kind: u8, l: usize, pos: usize) -> Result<()> {
        // the indexer / top-k range is a property of the context size, NOT of the rows' positions: a position bucket made a row's
        // selection scan depend on the other rows of its pass (rows straddling a bucket boundary scanned a longer range). DSC_BUCKET=1 restores it.
        let bp = if self.bucket_pos || ((self.nr > RMAX || self.no_graph) && !self.pf_fullbp) {
            ((pos + 1).div_ceil(1024) * 1024).min(self.max_ctx.div_ceil(1024) * 1024)
        } else if self.depth_bucket {
            // graphs are keyed by the depth range (powers of two from 4K): the indexer scans the lane's depth, not the configured context
            ((pos + 1).next_power_of_two().max(4096)).min(self.max_ctx.div_ceil(1024) * 1024)
        } else {
            self.max_ctx.div_ceil(1024) * 1024
        };
        if !self.use_graphs || self.tokens_done < 2 || self.nr > RMAX || self.no_graph {
            return self.seg_run(kind, l, bp).map(|_| ());
        }
        let idx = kind == 0 && self.layers[l].role.index_source;
        let key = (kind, l, (if idx { bp } else { 0 }) * 16 + self.nr, self.h.ptr().0);
        if !self.graphs.contains_key(&key) {
            // every (segment, layer, rows, parity) is its own instantiated graph and they cost device memory: a long-running server that
            // sees many row counts used to run out (cuGraphInstantiate OOM after ~1.4K graphs); past the cap, or on an OOM, drop them all
            // (they are re-captured on use)
            if self.graphs.len() >= self.graph_max {
                // least recently used out
                if let Some(k) = self.graph_use.iter().min_by_key(|(_, t)| **t).map(|(k, _)| *k) {
                    self.graphs.remove(&k);
                    self.graph_use.remove(&k);
                }
                self.graph_clears += 1;
            }
            let mut tries = 0;
            let (g, sw) = loop {
                self.s.sync()?;
                self.s.begin_capture()?;
                let r = self.seg_run(kind, l, bp);
                let g = self.s.end_capture();
                let sw = r?;
                match g {
                    Ok(g) => break (g, sw),
                    Err(e) if tries == 0 && !self.graphs.is_empty() && format!("{e:#}").contains("OUT_OF_MEMORY") => {
                        eprintln!("dsc: graph memory exhausted at {} graphs: dropping them", self.graphs.len());
                        self.graphs.clear(); self.graph_use.clear();
                        // the capture already ran the swaps of this segment's host side: undo them for the retry
                        for _ in 0..sw {
                            std::mem::swap(&mut self.h, &mut self.h2);
                        }
                        tries += 1;
                    }
                    Err(e) => return Err(e.into()),
                }
            };
            self.graphs.insert(key, (g, sw));
            self.just_captured = true;
        }
        self.graph_tick += 1;
        self.graph_use.insert(key, self.graph_tick);
        let (g, sw) = &self.graphs[&key];
        g.launch(&self.s)?;
        // a capture ran the swaps already (and does not execute); a hit replays them
        if self.just_captured {
            self.just_captured = false;
        } else {
            for _ in 0..*sw {
                std::mem::swap(&mut self.h, &mut self.h2);
            }
        }
        Ok(())
    }

    fn seg_run(&mut self, kind: u8, l: usize, bp: usize) -> Result<usize> {
        match kind {
            0 => self.seg_a(l, bp),
            1 => self.seg_b(l),
            _ => self.seg_c(l),
        }
    }

    /// Measurement only: the (layer, expert, owner) of every pick of the last pass (`nr` rows) that went to a host-backed expert
    /// (owner as the placement stores it: the card that reads / stages it). Empty without the zero-copy path.
    pub fn cold_picks(&self, nr: usize) -> Vec<(u16, u16, u8)> {
        let c = self.c;
        let (ne, nx) = (c.n_expert, c.n_expert_used);
        let stride = RMAX * nx;
        let mut out = vec![];
        let Some(il) = self.ids_log.as_ref() else { return out };
        if self.zc_off.is_empty() {
            return out;
        }
        for l in 0..c.n_layer {
            for q in 0..nr * nx {
                let e = il.as_slice()[l * stride + q] as usize;
                let i = l * ne + e;
                if self.zc_off[i][0] != usize::MAX {
                    out.push((l as u16, e as u16, self.owner_h.get(i).copied().unwrap_or(255)));
                }
            }
        }
        out
    }

    /// Measurement: count the picks of the pass into the heat counters and write the per-pass log line
    /// `<pos0> <rows> <logits 0/1> <ms> <picks of host-backed experts by owner x3> <distinct host-backed experts by owner x3> <distinct resident experts>`.
    fn pass_note(&mut self, nr: usize, pos0: usize, want: bool) {
        if !self.heat_acc && self.pass_log.is_none() {
            return;
        }
        let c = self.c;
        let (ne, nx) = (c.n_expert, c.n_expert_used);
        let stride = RMAX * nx;
        let Some(il) = self.ids_log.as_ref() else { return };
        let il = il.as_slice();
        let (mut picks, mut dist, mut dres) = ([0u32; 3], [0u32; 3], 0u32);
        let mut seen: Vec<usize> = Vec::with_capacity(nr * nx);
        for l in 0..c.n_layer {
            seen.clear();
            for q in 0..nr * nx {
                let e = il[l * stride + q] as usize;
                let i = l * ne + e;
                if self.heat_acc {
                    self.heat_cnt[i] += 1;
                }
                let cold = self.zc_off[i][0] != usize::MAX;
                let o = (self.owner_h[i] as usize).min(2);
                if cold {
                    picks[o] += 1;
                }
                if !seen.contains(&e) {
                    seen.push(e);
                    if cold { dist[o] += 1 } else { dres += 1 }
                }
            }
        }
        if let Some(f) = self.pass_log.as_mut() {
            use std::io::Write;
            let _ = writeln!(f, "{pos0} {nr} {} {:.3} {} {} {} {} {} {} {dres}", want as u8, self.split.total_ms, picks[0], picks[1], picks[2], dist[0], dist[1], dist[2]);
        }
        self.heat_passes += 1;
        if self.heat_passes % 512 == 0 {
            self.heat_flush();
        }
    }

    /// Write the heat counters (u64 LE per (layer, expert)) to DSC_HEAT_LOG and flush the pass log.
    pub fn heat_flush(&mut self) {
        if let Some(f) = self.pass_log.as_mut() {
            use std::io::Write;
            let _ = f.flush();
        }
        if let Some(p) = self.heat_path.clone() {
            let _ = self.heat_write(&p);
        }
    }

    pub fn heat_write(&self, p: &str) -> Result<()> {
        let b: Vec<u8> = self.heat_cnt.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tmp = format!("{p}.tmp");
        std::fs::write(&tmp, b)?;
        std::fs::rename(&tmp, p)?;
        Ok(())
    }

    /// Start (or restart from zero) counting picks without a file.
    pub fn heat_on(&mut self) {
        self.heat_acc = true;
        self.heat_cnt.iter_mut().for_each(|v| *v = 0);
    }

    /// Measurement only: per layer the host-backed experts (count, bytes of one expert: gate, up, down as stored) and their owners.
    pub fn host_backed(&self) -> Vec<(usize, [usize; 3], Vec<(u16, u8)>)> {
        let c = self.c;
        let ne = c.n_expert;
        (0..c.n_layer).map(|l| {
            let v: Vec<(u16, u8)> = (0..ne).filter(|&e| !self.zc_off.is_empty() && self.zc_off[l * ne + e][0] != usize::MAX).map(|e| (e as u16, self.owner_h.get(l * ne + e).copied().unwrap_or(255))).collect();
            (v.len(), self.zc_sz.get(l).copied().unwrap_or([0; 3]), v)
        }).collect()
    }

    /// After a pass: count the pairs that went to host-backed experts not in a card's slot ring (cold pairs) and promote the experts they used into
    /// the rings (async uploads on the cards' streams, LRU eviction restoring the evicted expert's table entries to its host address). The next
    /// pass finds them on the card.
    fn zc_promote(&mut self, nr: usize) -> Result<()> {
        let c = self.c;
        let (ne, nx) = (c.n_expert, c.n_expert_used);
        let stride = RMAX * nx;
        let np = nr * nx;
        let mut cold = 0u32;
        let mut uses: std::collections::HashMap<usize, u32> = Default::default();
        self.ring_clock += 1;
        for l in 0..c.n_layer {
            for q in 0..np {
                let e = self.ids_log.as_ref().unwrap().as_slice()[l * stride + q] as usize;
                let i = l * ne + e;
                if self.zc_off[i][0] == usize::MAX {
                    continue;
                }
                if !self.ring_of.is_empty() && self.ring_of[i] != 0 {
                    let k = self.owner_h[i] as usize - 1;
                    self.ring_stamp[k][self.ring_of[i] as usize - 1] = self.ring_clock;
                } else {
                    cold += 1;
                    *uses.entry(i).or_insert(0) += 1;
                }
            }
        }
        self.zc_hist.push((cold, nr as u16));
        if self.zring.is_empty() || uses.is_empty() {
            return Ok(());
        }
        let mut cand: Vec<(usize, u32)> = uses.into_iter().collect();
        cand.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        cand.truncate(self.ring_promo);
        let sb = self.ring_slot_bytes;
        let mut si = 0usize;
        for (i, _) in cand {
            let k = self.owner_h[i] as usize - 1;
            let (l, e) = (i / ne, i % ne);
            // a free slot, else the least recently used
            let slot = {
                let st = &self.ring_stamp[k];
                let ex = &self.ring_expert[k];
                (0..ex.len()).min_by_key(|&s| if ex[s] == usize::MAX { 0 } else { st[s] + 1 }).unwrap()
            };
            let old = self.ring_expert[k][slot];
            let hb = self.zc_buf.as_ref().unwrap();
            let stage = self.promo_stage.as_mut().unwrap();
            let xc = &self.xcs[k];
            xc.ctx.bind()?;
            if old != usize::MAX {
                let (lo, eo) = (old / ne, old % ne);
                let ro = xc.res[lo].as_ref().unwrap();
                for (j, tab) in [&ro.gtab, &ro.utab, &ro.dtab].into_iter().enumerate() {
                    stage.as_mut_slice()[si] = hb.dev().0 + self.zc_off[old][j] as u64;
                    unsafe { xc.s.memcpy_htod(tab.ptr().add::<u64>(eo), &stage.as_slice()[si..si + 1])? };
                    si += 1;
                }
                self.ring_of[old] = 0;
            }
            let base = self.zring[k].ptr().byte_add(slot * sb);
            let off0 = self.zc_off[i][0];
            let tot = self.zc_tot[i];
            unsafe { xc.s.memcpy_htod(base, &hb.as_slice()[off0..off0 + tot])? };
            let r = xc.res[l].as_ref().unwrap();
            for (j, tab) in [&r.gtab, &r.utab, &r.dtab].into_iter().enumerate() {
                stage.as_mut_slice()[si] = base.0 + (self.zc_off[i][j] - off0) as u64;
                unsafe { xc.s.memcpy_htod(tab.ptr().add::<u64>(e), &stage.as_slice()[si..si + 1])? };
                si += 1;
            }
            self.ring_expert[k][slot] = i;
            self.ring_stamp[k][slot] = self.ring_clock;
            self.ring_of[i] = slot as u32 + 1;
        }
        self._ctx.bind()?;
        Ok(())
    }

    /// One layer with no host in the loop: the router ids and x go to the expert cards by peer copies on the main stream, the cards compute the
    /// pairs they own (resident or read from pinned host memory), their results come back by peer copies behind events, the pair routing is a
    /// kernel on the card. Nothing here blocks the host.
    fn layer_zc(&mut self, l: usize, pos0: usize) -> Result<()> {
        let c = self.c;
        let (d, nx, nr) = (c.n_embd, c.n_expert_used, self.nr);
        let np = nr * nx;
        let pos = pos0 + nr - 1;
        let stride = RMAX * nx;
        self.seg(0, l, pos)?;
        if self.zc_sync {
            // experiment: what a host in the loop costs (the router ids and x of every layer brought to the host before the experts run)
            self.s.sync()?;
            let mut ids = vec![0u32; np];
            memcpy_dtoh(&mut ids, self.ids.ptr())?;
            let mut xb = vec![0u16; nr * d];
            memcpy_dtoh(&mut xb, self.x.ptr())?;
            std::hint::black_box((&ids, &xb));
        }
        if let Ok(f) = std::env::var("DSC_ROUTES") {
            // debug (eager mode only, DSC_GRAPH=0): the router's picks and weights of every row, the format ORACLE_FORCE_ROUTES reads
            use std::io::Write;
            self.s.sync()?;
            let mut ids = vec![0u32; np];
            let mut wts = vec![0f32; np];
            memcpy_dtoh(&mut ids, self.ids.ptr())?;
            memcpy_dtoh(&mut wts, self.wts.ptr())?;
            if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(f) {
                for r in 0..nr {
                    let _ = writeln!(fh, "{} {} {:?} {:?}", pos0 + r, l, &ids[r * nx..(r + 1) * nx], &wts[r * nx..(r + 1) * nx]);
                }
            }
        }
        if let Some(il) = self.ids_log.as_mut() {
            unsafe { self.s.memcpy_dtoh(&mut il.as_mut_slice()[l * stride..l * stride + np], self.ids.ptr())? };
        }
        // per card a copy stream on the main context: x and the ids go out behind the main stream's event, the results come back behind the
        // card's event, so the copies of different cards overlap each other and the main card's segment B
        let evm = self.ev_main.as_ref().unwrap();
        self.s.record(evm)?;
        for k in 0..self.xcs.len() {
            if self.xcs[k].res[l].is_none() {
                continue;
            }
            let xc = &mut self.xcs[k];
            let cs = &self.cs[k];
            cs.wait(evm)?;
            xc.ctx.copy_peer(xc.x.ptr(), &self._ctx, self.x.ptr(), nr * d * 2, cs)?;
            xc.ctx.copy_peer(xc.ids.ptr(), &self._ctx, self.ids.ptr(), np * 4, cs)?;
            cs.record(&self.ev_b[k])?;
            let r = xc.res[l].as_ref().unwrap();
            xc.ctx.bind()?;
            xc.s.wait(&self.ev_b[k])?;
            if self.cardtime {
                xc.s.record(&self.zc_ev[k][l].0)?;
                self.zc_used[k][l] = true;
            }
            xc.ggq.quantize_q8_1(&xc.s, xc.x.ptr(), d, true, nr, d, &mut xc.act)?;
            xc.ggq.set_swiglu_limit(c.clamp_exp[l]);
            let tabs = [r.gtab.ptr(), r.utab.ptr(), r.dtab.ptr()];
            let staged = if let Some(sg) = &xc.stage { if self.zc_hi > 0 && self.stage_on { Some(sg) } else { None } } else { None };
            if let Some(sg) = staged {
                xc.ggq.cold_begin(&xc.s, xc.ids.ptr(), np, tabs, self.zc_lo, self.zc_hi, sg.counter.ptr(), sg.slot_e.ptr(), sg.orig.ptr(), sg.buf.ptr(), sg.slot_bytes, sg.nslots, self.zc_sz[l])?;
            }
            xc.ggq.moe_gu_q8(&xc.s, r.ty_gu, r.gtab.ptr(), r.utab.ptr(), r.rb_gu, xc.ids.ptr(), nr, nx, &xc.act, c.n_ff_exp, &mut xc.hq, None, None)?;
            xc.ggq.moe_down(&xc.s, r.ty_d, r.dtab.ptr(), r.rb_d, xc.ids.ptr(), nr, nx, &xc.hq, d, xc.y6.ptr(), None)?;
            if let Some(sg) = staged {
                xc.ggq.cold_end(&xc.s, tabs, sg.counter.ptr(), sg.slot_e.ptr(), sg.orig.ptr(), sg.nslots)?;
            }
            xc.s.record(&xc.ev)?;
            if self.cardtime {
                xc.s.record(&self.zc_ev[k][l].1)?;
            }
            self._ctx.bind()?;
            cs.wait(&xc.ev)?;
            let dst = self.ycpu.ptr().add::<f32>(k * stride * d);
            self._ctx.copy_peer(dst, &xc.ctx, xc.y6.ptr(), np * d * 4, cs)?;
            cs.record(&self.ev_c[k])?;
        }
        self._ctx.bind()?;
        let owner = self.owner.as_ref().unwrap().ptr().byte_add(l * c.n_expert);
        self.ggq.slot_owner(&self.s, self.slot.ptr(), owner, self.ids.ptr(), np, stride)?;
        self.seg(1, l, pos)?;
        for k in 0..self.xcs.len() {
            if self.xcs[k].res[l].is_some() {
                self.s.wait(&self.ev_c[k])?;
            }
        }
        self.seg(2, l, pos)?;
        Ok(())
    }

    /// Startup self-test of the cross-card copies: a known pattern across every ordered pair of cards must arrive intact (cuCtxEnablePeerAccess
    /// is known to make copies silently deliver zeros on this box; a driver change must not corrupt outputs unnoticed).
    fn p2p_selftest(&self) -> Result<()> {
        let ctxs: Vec<&Context> = std::iter::once(&*self._ctx).chain(self.xcs.iter().map(|x| &*x.ctx)).collect();
        let pat: Vec<u32> = (0..16384u32).map(|i| i.wrapping_mul(2654435761).wrapping_add(12345)).collect();
        for a in 0..ctxs.len() {
            for b in 0..ctxs.len() {
                if a == b {
                    continue;
                }
                ctxs[a].bind()?;
                let src = DeviceBuffer::<u32>::zeroed(16384)?;
                memcpy_htod(src.ptr(), &pat)?;
                ctxs[b].bind()?;
                let dst = DeviceBuffer::<u32>::zeroed(16384)?;
                let s = Stream::new()?;
                ctxs[b].copy_peer(dst.ptr(), ctxs[a], src.ptr(), 16384 * 4, &s)?;
                s.sync()?;
                let mut out = vec![0u32; 16384];
                memcpy_dtoh(&mut out, dst.ptr())?;
                let ok = out == pat;
                drop(s);
                drop(dst);
                ctxs[a].bind()?;
                drop(src);
                ensure!(ok, "cross-card copy {a} -> {b} returned wrong data (peer access broken?): refusing to run, results would be corrupt");
            }
        }
        self._ctx.bind()?;
        eprintln!("cross-card copy self-test passed for {} cards", ctxs.len());
        Ok(())
    }

    /// DSC_PF_DUMP debugging: append the rows' h (after attention) and router ids of layer l to files.
    fn dump_mid(&self, l: usize) -> Result<()> {
        let (Ok(dir), Ok(nl)) = (std::env::var("DSC_PF_DUMP"), std::env::var("DSC_PF_DUMPL")) else { return Ok(()) };
        if l >= nl.parse::<usize>().unwrap_or(0) {
            return Ok(());
        }
        let (nr, d) = (self.nr, self.c.n_embd);
        let mut v = vec![0u16; nr * 4 * d];
        memcpy_dtoh(&mut v, self.h.ptr())?;
        let mut ids = vec![0u32; nr * self.c.n_expert_used];
        memcpy_dtoh(&mut ids, self.ids.ptr())?;
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/a_L{l}.bin"))?;
        f.write_all(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/i_L{l}.bin"))?;
        f.write_all(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
        Ok(())
    }

    fn layer(&mut self, l: usize, pos0: usize) -> Result<()> {
        let p = self.layer_pre(l, pos0)?;
        let t0 = Instant::now();
        self.cpu.swiglu_limit = p.limit;
        if !p.jobs.is_empty() {
            let lf = std::env::var_os("DSC_LANEFAULTS").map(|_| minflt_self());
            self.cpu.run(&p.jobs, &p.xh)?;
            if let Some(f0) = lf {
                LANE_FAULTS.fetch_add(minflt_self() - f0, std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(n) = std::env::var("DSC_REPEATCPU").ok().and_then(|v| v.parse::<usize>().ok()) {
                let t1 = Instant::now();
                for _ in 0..n {
                    self.cpu.run(&p.jobs, &p.xh)?;
                }
                self.split.ph[4] += t1.elapsed().as_secs_f64() * 1e3 / n as f64;
            }
        }
        let outs: Vec<Vec<f32>> = (0..p.jobs.len()).map(|j| self.cpu.out(j).to_vec()).collect();
        let t_cpu = t0.elapsed();
        self.split.ph[2] += t_cpu.as_secs_f64() * 1e3;
        self.layer_post(p, &outs, t_cpu)
    }

    /// First half of a layer: segment A, the router ids to the host, segment B and the second card's resident pairs enqueued, and the
    /// CPU lane's jobs built (their results come from `CpuMoe::run` on `p.jobs` / `p.xh`, run by the caller, possibly on another thread).
    fn layer_pre(&mut self, l: usize, pos0: usize) -> Result<Pend<'a>> {
        let c = self.c;
        let g: &'a Gguf = self.g;
        let (d, nx, nr) = (c.n_embd, c.n_expert_used, self.nr);
        let np = nr * nx;
        let pos = pos0 + nr - 1;
        let t_layer = Instant::now();
        self.seg(0, l, pos)?;
        let t_sy = Instant::now();
        self.s.sync()?;
        self.split.ph[5] += t_sy.elapsed().as_secs_f64() * 1e3;

        self.dump_mid(l)?;
        let x = self.x.ptr();
        let mut ids = vec![0u32; np];
        let mut wts = vec![0f32; np];
        memcpy_dtoh(&mut ids, self.ids.ptr())?;
        memcpy_dtoh(&mut wts, self.wts.ptr())?;
        let mut xb = vec![0u16; nr * d];
        memcpy_dtoh(&mut xb, x)?;
        let xh: Vec<f32> = xb.iter().map(|&b| refq::bf16_to_f32(b)).collect();
        if let Some(t) = self.trace.as_mut() {
            for r in 0..nr {
                let mut i6 = [0u16; 6];
                let mut w6 = [0f32; 6];
                for k in 0..6.min(nx) {
                    i6[k] = ids[r * nx + k] as u16;
                    w6[k] = wts[r * nx + k];
                }
                t.push((l as u8, (pos0 + r) as u32, i6, w6));
            }
        }
        let t_p1 = Instant::now();
        self.split.ph[0] += (t_p1 - t_layer).as_secs_f64() * 1e3;
        let lres = self.res[l].as_ref();
        let lres = self.res[l].as_ref();
        let on1: Vec<bool> = (0..np).map(|p| lres.is_some_and(|r| r.present[ids[p] as usize])).collect();
        // the first expert card that holds the pair's expert (0 = none)
        let onx: Vec<u8> = (0..np).map(|p| if on1[p] { 0 } else {
            self.xcs.iter().position(|x| x.res[l].as_ref().is_some_and(|r| r.present[ids[p] as usize])).map_or(0, |k| k as u8 + 1)
        }).collect();
        let miss: Vec<usize> = (0..np).filter(|&p| !on1[p] && onx[p] == 0).collect();
        for k in 0..self.xcs.len() {
            if !onx.iter().any(|&v| v as usize == k + 1) {
                continue;
            }
            let xc = &mut self.xcs[k];
            let r = xc.res[l].as_ref().unwrap();
            xc.ctx.bind()?;
            // x straight from the main card (P2P), the ids from the host
            xc.x_h.as_mut_slice()[..nr * d].copy_from_slice(&xb);
            xc.ids_h.as_mut_slice()[..np].copy_from_slice(&ids);
            unsafe {
                xc.s.memcpy_htod(xc.x.ptr(), &xc.x_h.as_slice()[..nr * d])?;
                xc.s.memcpy_htod(xc.ids.ptr(), &xc.ids_h.as_slice()[..np])?;
            }
            if self.cardtime {
                xc.s.record(&xc.ev_a)?;
            }
            xc.ggq.quantize_q8_1(&xc.s, xc.x.ptr(), d, true, nr, d, &mut xc.act)?;
            xc.ggq.set_swiglu_limit(c.clamp_exp[l]);
            xc.ggq.moe_gu_q8(&xc.s, r.ty_gu, r.gtab.ptr(), r.utab.ptr(), r.rb_gu, xc.ids.ptr(), nr, nx, &xc.act, c.n_ff_exp, &mut xc.hq, None, None)?;
            xc.ggq.moe_down(&xc.s, r.ty_d, r.dtab.ptr(), r.rb_d, xc.ids.ptr(), nr, nx, &xc.hq, d, xc.y6.ptr(), None)?;
            // the main stream copies the results into its ycpu region k (layer_post), after waiting for this event
            if std::env::var_os("DSC_PRESYNC").is_some() {
                xc.s.sync()?;
            }
            xc.s.record(&xc.ev)?;
            if self.cardtime {
                xc.s.record(&xc.ev_b)?;
            }
            self._ctx.bind()?;
        }
        // segment B (shared expert + this card's residents) only after the expert cards have read x (it may reuse the buffer)
        self.seg(1, l, pos)?;
        let n0 = onx.iter().filter(|&&b| b != 0).count();
        self.split.hit += (np - miss.len() - n0) as u32;
        self.split.hit0 += n0 as u32;
        self.split.miss += miss.len() as u32;
        self.split.ph[1] += t_p1.elapsed().as_secs_f64() * 1e3;
        let (tg, tu, td) = (
            g.tensor(&format!("blk.{l}.ffn_gate_exps.weight")).unwrap(),
            g.tensor(&format!("blk.{l}.ffn_up_exps.weight")).unwrap(),
            g.tensor(&format!("blk.{l}.ffn_down_exps.weight")).unwrap(),
        );
        // the missed pairs grouped by expert: one job per distinct expert with the rows that picked it
        let mut eorder: Vec<u32> = vec![];
        let mut erows: Vec<Vec<usize>> = vec![];
        let mut pair_job = vec![(usize::MAX, 0usize); np];
        for &p in &miss {
            let e = ids[p];
            let ji = match eorder.iter().position(|&q| q == e) {
                Some(j) => j,
                None => {
                    eorder.push(e);
                    erows.push(vec![]);
                    eorder.len() - 1
                }
            };
            pair_job[p] = (ji, erows[ji].len());
            erows[ji].push(p / nx);
        }
        self.split.distinct += eorder.len() as u32;
        let hostr = self.host.as_ref();
        let jobs: Vec<MoeJob<'a>> = eorder.iter().zip(&erows).map(|(&e, rows)| {
            let e = e as usize;
            if let Some(h) = hostr {
                let o = h.off[l * c.n_expert + e];
                if !h.rty.is_empty() {
                    let z = h.rsz[l];
                    let sl = |j: usize| unsafe { std::slice::from_raw_parts(h.ptr.add(o[j]), z[j]) };
                    return MoeJob { gate: sl(0), up: sl(1), down: sl(2), ty_gu: h.rty[l].0, ty_down: h.rty[l].1, rows: rows.clone() };
                }
                let sl = |j: usize, t: &fe_gguf::TensorInfo| unsafe { std::slice::from_raw_parts(h.ptr.add(o[j]), t.slice_bytes()) };
                MoeJob { gate: sl(0, tg), up: sl(1, tu), down: sl(2, td), ty_gu: tg.ty, ty_down: td.ty, rows: rows.clone() }
            } else {
                MoeJob { gate: g.slice(tg, e), up: g.slice(tu, e), down: g.slice(td, e), ty_gu: tg.ty, ty_down: td.ty, rows: rows.clone() }
            }
        }).collect();
        let limit = if self.swiglu_limit { c.clamp_exp[l] } else { f32::INFINITY };
        Ok(Pend { l, pos, nr, np, xh, onx, on1, n0, pair_job, jobs, limit, t_layer })
    }

    /// Second half of a layer: the second card's pairs and the CPU lane's results (`outs[job]`, rows of the job) gathered into the slot
    /// table, uploaded, and segment C (combine + hc post) enqueued.
    fn layer_post(&mut self, p: Pend<'a>, outs: &[Vec<f32>], t_cpu: Duration) -> Result<()> {
        let d = self.c.n_embd;
        let (np, n0) = (p.np, p.n0);
        let t_tail = Instant::now();
        let mut slot = vec![-1i32; np];
        let ncpu = (0..np).filter(|&q| p.onx[q] == 0 && !p.on1[q]).count();
        let mut yc = vec![0f32; ncpu * d];
        // expert-card results are already in this card's `ycpu` regions (P2P copies enqueued in layer_pre): the main stream just waits for
        // their events; the host only builds the slot table and uploads the CPU lane's rows
        let ncards = self.xcs.len();
        let stride = RMAX * self.c.n_expert_used;
        let mut ycpu_rows = 0usize;
        for q in 0..np {
            if p.onx[q] != 0 {
                slot[q] = ((p.onx[q] as usize - 1) * stride + q) as i32;
            } else if !p.on1[q] {
                let (ji, ri) = p.pair_job[q];
                slot[q] = (ncards * stride + ycpu_rows) as i32;
                yc[ycpu_rows * d..(ycpu_rows + 1) * d].copy_from_slice(&outs[ji][ri * d..(ri + 1) * d]);
                ycpu_rows += 1;
            }
        }
        for k in 0..ncards {
            if p.onx.iter().any(|&v| v as usize == k + 1) {
                if self.cardtime {
                    self.xcs[k].ev_b.sync()?;
                    let ms = self.xcs[k].ev_b.elapsed_since(&self.xcs[k].ev_a)? as f64;
                    self.split.ph[if k == 0 { 4 } else { 9 }] += ms;
                }
                self.s.wait(&self.xcs[k].ev)?;
                let dst = self.ycpu.ptr().add::<f32>(k * stride * d);
                self._ctx.copy_peer(dst, &self.xcs[k].ctx, self.xcs[k].y6.ptr(), np * d * 4, &self.s)?;
                if std::env::var_os("DSC_P2PCHECK").is_some() {
                    self.s.sync()?;
                    let mut b = vec![0f32; np * d];
                    memcpy_dtoh(&mut b, dst)?;
                    let xc = &self.xcs[k];
                    xc.ctx.bind()?;
                    let mut a = vec![0f32; np * d];
                    memcpy_dtoh(&mut a, xc.y6.ptr())?;
                    self._ctx.bind()?;
                    let bad = a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                    eprintln!("p2pcheck(post) layer {} card {k}: {bad} of {} differ; first src {:?} dst {:?}", p.l, np * d, &a[..3], &b[..3]);
                }
            }
        }
        self.slot_h.as_mut_slice()[..np].copy_from_slice(&slot);
        unsafe { self.s.memcpy_htod(self.slot.ptr(), &self.slot_h.as_slice()[..np])? };
        if ycpu_rows > 0 {
            self.yc_h.as_mut_slice()[..ycpu_rows * d].copy_from_slice(&yc[..ycpu_rows * d]);
            unsafe { self.s.memcpy_htod(self.ycpu.ptr().add::<f32>(ncards * stride * d), &self.yc_h.as_slice()[..ycpu_rows * d])? };
        }
        self.split.ph[7] += t_tail.elapsed().as_secs_f64() * 1e3 - self.split.ph[6] * 0.0;
        let t_s2 = Instant::now();
        self.seg(2, p.l, p.pos)?;
        self.split.ph[8] += t_s2.elapsed().as_secs_f64() * 1e3;
        self.split.ph[3] += t_tail.elapsed().as_secs_f64() * 1e3;
        self.split.cpu_ms += t_cpu.as_secs_f64() * 1e3;
        self.split.gpu_ms += (p.t_layer.elapsed().saturating_sub(t_cpu)).as_secs_f64() * 1e3;
        let _ = p.nr;
        Ok(())
    }
}

impl Drop for Engine<'_> {
    /// The CUDA graphs go before the streams and contexts they were captured on.
    fn drop(&mut self) {
        let _ = self.s.sync();
        self.graphs.clear(); self.graph_use.clear();
        // the expert-only card goes with its own context bound
        for xc in std::mem::take(&mut self.xcs) {
            let _ = xc.ctx.bind();
            let _ = xc.s.sync();
            std::mem::forget(xc); // freed with the process (destroying a second context's objects at exit segfaults)
        }
        let _ = self._ctx.bind();
    }
}

pub fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
}

#[allow(dead_code)]
fn _unused() -> Result<()> {
    bail!("unused")
}

fn minflt_self() -> u64 {
    // process-wide minor faults (all threads: the CPU lane's workers included)
    let t = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    t.rsplit(')').next().unwrap_or("").split_whitespace().nth(7).and_then(|v| v.parse().ok()).unwrap_or(0)
}
