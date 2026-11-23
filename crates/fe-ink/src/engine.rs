//! The Inkling-2-Mini forward: straightforward, correctness-first (bf16 GEMMs, simple expert GEMVs). Prefill and decode
//! run the same code on a chunk of T tokens (T = 1 for decode).
use crate::ckpt::Ckpt;
use crate::weights::{self, Experts, Weights};
use crate::*;
use anyhow::{ensure, Result};
use fe_cuda::{Context, DevPtr, DeviceBuffer, Function, Module, Stream};
use std::path::Path;
use std::sync::Arc;

type Buf = DeviceBuffer<u8>;

pub struct Kern {
    pub gemm: Function,
    pub rms: Function,
    pub blockres: Function,
    pub gate_mul: Function,
    pub resid: Function,
    pub final_add: Function,
    pub copy_rows: Function,
    pub headclip: Function,
    pub sconv: Function,
    pub sconv_dyn: Function,
    pub argmax_rows: Function,
    pub conv_commit: Function,
    pub headnorm: Function,
    pub kvq_pack: Function,
    pub attn_tc: Function,
    pub dg_k: Function,
    pub dg_v: Function,
    pub dg_comb: Function,
    pub dg2_k: Function,
    pub dg2_v: Function,
    pub attn_comb2: Function,
    pub attn_gstat: Function,
    pub gate_z: Function,
    pub gate_out: Function,
    pub gate_g: Function,
    pub gate_mix: Function,
    pub router_logits: Function,
    pub router_pick: Function,
    pub gemm_s: Function,
    pub gemm_s8: Function,
    pub dequant: Function,
}

const GEMM_SMEM: u32 = 3 * 2 * 128 * 32 * 2;
const SGEMM_SMEM: u32 = 4 * 48 * 144;
const SGEMM8_SMEM: u32 = 4 * (32 * 80 + 16 * 144) + 2 * 32 * 144;
/// k-inkm attn_tc.rs ATC_SMEM
const ATC_SMEM: u32 = 16 * 272 + 2 * 64 * 272 + 16 * 144 * 2 + 16 * 16 * 4 + 4 * 16 * 2 * 4 + 16 * 4 * 2;

/// A K/V cache: a plain buffer (sliding rings, eager) or a reserved virtual range whose memory is mapped as the context grows (global
/// applications with INK_KV_LAZY=1: the address never changes, so the captured graphs stay valid).
enum KvBuf {
    Plain(Buf),
    Lazy(fe_cuda::VmBuffer),
}
impl KvBuf {
    fn ptr(&self) -> DevPtr {
        match self {
            KvBuf::Plain(b) => b.ptr(),
            KvBuf::Lazy(v) => v.ptr(),
        }
    }
    fn grow(&mut self, bytes: usize) -> Result<usize> {
        match self {
            KvBuf::Plain(_) => Ok(0),
            KvBuf::Lazy(v) => Ok(v.grow(bytes)?),
        }
    }
    fn reserved(&self) -> usize {
        match self {
            KvBuf::Plain(b) => b.bytes(),
            KvBuf::Lazy(v) => v.reserved(),
        }
    }
    fn mapped(&self) -> usize {
        match self {
            KvBuf::Plain(b) => b.bytes(),
            KvBuf::Lazy(v) => v.mapped(),
        }
    }
    fn shrink(&mut self, bytes: usize) -> Result<usize> {
        match self {
            KvBuf::Plain(_) => Ok(0),
            KvBuf::Lazy(v) => Ok(v.shrink(bytes)?),
        }
    }
}

struct App {
    k: KvBuf,
    v: KvBuf,
    conv_k: Buf,
    conv_v: Buf,
    conv_a: Buf,
    conv_m: Buf,
}

struct Scratch {
    hs: [Buf; 2],
    pre_hs: Buf,
    prev12: Buf,
    prevfin: Buf,
    blocks: Buf,
    ai: Buf,
    msg: Buf,
    gh: Buf,
    gh2: [Buf; 2],
    gg2: [Buf; 2],
    gz2: [Buf; 2],
    x: [Buf; 3],
    qgr: Buf,
    klin: Buf,
    vlin: Buf,
    slin: Buf,
    dynk: Buf,
    dynv: Buf,
    qp: Buf,
    rsx: Buf,
    kcv: Buf,
    vcv: Buf,
    att: Buf,
    gat: Buf,
    o: Buf,
    a2: Buf,
    resid: Buf,
    mi: Buf,
    mg: Buf,
    xm: Buf,
    picks: Buf,
    wts: Buf,
    rlog: Buf,
    forced: Buf,
    moe: Buf,
    mc: Buf,
    gz: Buf,
    gzf: Buf,
    gg: Buf,
    fin: Buf,
    xf: Buf,
    logits: Buf,
}

pub struct Tap {
    pub hs: Vec<u16>,
    pub picks: Vec<i32>,
    pub wts: Vec<f32>,
    pub rlog: Vec<f32>,
}

#[derive(Default)]
pub struct Taps {
    pub embed: Vec<u16>,
    pub app: Vec<Option<Tap>>,
    pub fin: Vec<u16>,
}

pub struct Ink {
    pub s: Stream,
    k: Kern,
    pub w: Weights,
    ck: Ckpt,
    apps: Vec<App>,
    sc: Scratch,
    pub cap: usize,
    pub tmax: usize,
    pub len: usize,
    pub prof_on: bool,
    kk: fe_inkk::Kern,
    /// prompt tokens: dense FP8 linears dequantize to bf16 and take the tensor-core GEMM whatever the row count (chunk-size
    /// invariant); decode tokens (<= 16 rows) take the FP8 GEMV
    pub prompt: bool,
    /// speculative chunk: conv inputs are kept after the 3 state rows so `rollback` can keep any prefix of the chunk
    pub spec: bool,
    argmax_only: bool,
    pub last_argmax: Vec<u32>,
    am_out: Buf,
    pending: usize,
    /// fill `dtaps` with the draft head's inputs (taps 0..2 = hidden entering applications 5, 23, 41; 3 = final norm output)
    pub draft_taps: bool,
    pub dtaps: Buf,
    dq: Buf,
    ast: Buf,
    apt: KvBuf,
    att_mb: usize,
    apt_reserved: usize,
    /// grouped-query deep decode attention for global layers (1..=8 rows)
    dg: bool,
    dg_slide: bool,
    dg2: bool,
    /// INK_FIT_MB=n: device memory (bytes) kept free by moving experts to the host before K/V growth, promoted back on reset (needs slot layers)
    fit_floor: Option<usize>,
    capturing: bool,
    posbuf: Buf,
    /// captured 1-row forwards keyed by (draft_taps, spec): the flags are baked into the captured kernels
    graphs: std::collections::HashMap<(bool, bool), fe_cuda::Graph>,
    pub use_graph: bool,
    /// global K/V in the 6-bit record form (INK_KVQ6=1): staging rows + packer
    pub kvq6: bool,
    kstg: Buf,
    vstg: Buf,
    posbuf0: Buf,
    aux: [Stream; 2],
    ev_fork: fe_cuda::Event,
    ev_join: [fe_cuda::Event; 2],
    pf_scr: fe_inkk::PrefillScratch,
    prof: std::cell::RefCell<std::collections::BTreeMap<&'static str, f64>>,
    evs: std::cell::RefCell<Vec<(&'static str, fe_cuda::Event)>>,
    last: std::cell::Cell<Option<std::time::Instant>>,
    /// G16 expert slots (INK_HOST_EXPERTS=N): staging / adaptation runtime
    slots: Option<SlotRt>,
    /// last: the context outlives every buffer, stream and graph above
    pub ctx: Arc<Context>,
}

/// Runtime of the host-resident experts: prompt staging through spare device slots (two sets, layer-ahead), adaptation of the residency
/// between graph launches from the device use counters (one adapt after the prompt, then every `adapt_every` decode tokens).
struct SlotRt {
    spare: fe_inkk::slots::SparePool,
    nmax: usize,
    copy_s: Stream,
    ev_st: [fe_cuda::Event; 2],
    ev_dn: [fe_cuda::Event; 2],
    ev_un: Vec<fe_cuda::Event>,
    adapt_every: usize,
    /// adapts since the prompt ended: the first three are synchronous (large batches while the residency converges)
    warm_adapts: usize,
    /// host milliseconds spent in the residency hook (cumulative)
    hook_ms: f64,
    since: usize,
    prompt_seen: bool,
    log: bool,
    host_use: u64,
    total_use: u64,
    swaps: usize,
    adapts: usize,
    /// the adapt after the prompt reports the prompt's own counters: kept apart from the decode totals
    prompt_host_use: u64,
    prompt_total_use: u64,
    /// application whose host experts are already staged (one application ahead)
    staged: std::cell::Cell<Option<usize>>,
    /// heat prior [24][128] for re-ranking when the resident target changes
    prior: Vec<Vec<f64>>,
    /// current number of host experts
    host_target: usize,
}

/// Layer of cache slot `app` (inverse of `app_index`).
fn li_of(app: usize) -> usize {
    if app < LOOP0 {
        app
    } else if app < LOOP0 + PASSES * (LOOP1 - LOOP0) {
        LOOP0 + (app - LOOP0) % (LOOP1 - LOOP0)
    } else {
        app - (LOOP1 - LOOP0) * (PASSES - 1)
    }
}
/// Rows of the K/V cache of a layer: the sliding layers keep a ring of window + chunk rows, the global ones every position.
fn ring_of(li: usize, cap: usize, tmax: usize) -> usize {
    if is_full(li) { cap } else { (1024 + tmax).min(cap) }
}

/// Sliding-window rows a prompt-cache snapshot keeps (the window of the sliding layers).
const SNAP_WIN: usize = 1024;
/// Bytes of the three conv tails of one application (K, V, attention and MLP convolutions, f32).
const CONV_SNAP: usize = 3 * (1024 + 1024 + D + D) * 4;

/// Positions [q0, q0 + w) of a ring of `r` rows (row = position % r) as (ring row, host row, rows) segments, host row = position - q0.
fn ring_segments(q0: usize, w: usize, r: usize) -> Vec<(usize, usize, usize)> {
    let mut v = vec![];
    let mut done = 0;
    while done < w {
        let row = (q0 + done) % r;
        let n = (w - done).min(r - row);
        v.push((row, done, n));
        done += n;
    }
    v
}

fn at_ptr(p: DevPtr, off: usize) -> DevPtr {
    p.byte_add(off)
}
fn mk_kv(li: usize, cap: usize, tmax: usize, kvq6: bool, lazy: bool, dev: i32) -> Result<KvBuf> {
    if is_full(li) {
        let per_tok = if kvq6 { 832 } else { 2048 };
        if lazy {
            return Ok(KvBuf::Lazy(fe_cuda::VmBuffer::reserve(cap * per_tok, dev)?));
        }
        return Ok(KvBuf::Plain(alloc(cap * per_tok)?));
    }
    Ok(KvBuf::Plain(alloc(ring_of(li, cap, tmax) * 1024 * 2)?))
}

fn alloc(bytes: usize) -> Result<Buf> {
    Ok(Buf::alloc(bytes.max(16))?)
}
fn at(b: &Buf, off: usize) -> DevPtr {
    b.ptr().byte_add(off)
}
const NUL: DevPtr = DevPtr(0);

impl Ink {
    pub fn load(model_dir: &Path, dev: i32, cap: usize, tmax: usize) -> Result<Ink> {
        // the 6-bit cache is chunk-major ([chunk of 64 keys][kv head][64][104 B]): whole chunks
        let cap = cap.next_multiple_of(64);
        let ctx = Context::new(dev)?;
        ctx.bind()?;
        let s = Stream::new()?;
        let memlog = crate::env_flag("INK_MEM_LOG");
        let mem_mark = |ctx: &Context, tag: &str, last: &std::cell::Cell<usize>| {
            if memlog {
                if let Ok((free, total)) = ctx.mem_info() {
                    let used = total - free;
                    eprintln!("ink mem: {tag:34} used {:8.1} MiB  (+{:.1} MiB)", used as f64 / 1048576.0, (used as f64 - last.get() as f64) / 1048576.0);
                    last.set(used);
                }
            }
        };
        let last = std::cell::Cell::new(0usize);
        mem_mark(&ctx, "context + driver", &last);
        let core = Module::load(fe_model::K_CORE_CUBIN)?;
        let m = Module::load(K_INKM_CUBIN)?;
        let kk = fe_inkk::Kern::load()?;
        let k = Kern {
            gemm: core.function("fe_gemm_bf16")?,
            rms: m.function("ink_rms")?,
            blockres: m.function("ink_blockres")?,
            gate_mul: m.function("ink_gate_mul")?,
            resid: m.function("ink_resid")?,
            final_add: m.function("ink_final_add")?,
            copy_rows: m.function("ink_copy_rows")?,
            headclip: m.function("ink_headclip")?,
            sconv: m.function("ink_sconv")?,
            sconv_dyn: m.function("ink_sconv_dyn")?,
            argmax_rows: Module::load(K_DRAFT_CUBIN)?.function("dr_argmax_rows")?,
            conv_commit: m.function("ink_conv_commit")?,
            headnorm: m.function("ink_headnorm")?,
            kvq_pack: m.function("ink_kvq_pack")?,
            attn_tc: {
                let f = m.function("ink_attn_tc")?;
                f.set_max_dynamic_smem(ATC_SMEM)?;
                f
            },
            dg_k: {
                let f = m.function("ink_dg_k")?;
                f.set_max_dynamic_smem(100_000)?;
                f
            },
            dg_v: {
                let f = m.function("ink_dg_v")?;
                f.set_max_dynamic_smem(100_000)?;
                f
            },
            dg_comb: m.function("ink_dg_comb")?,
            dg2_k: {
                let f = m.function("ink_dg2_k")?;
                f.set_max_dynamic_smem(100_000)?;
                f
            },
            dg2_v: {
                let f = m.function("ink_dg2_v")?;
                f.set_max_dynamic_smem(100_000)?;
                f
            },
            attn_comb2: m.function("ink_attn_comb2")?,
            attn_gstat: m.function("ink_attn_gstat")?,
            gate_z: m.function("ink_gate_z")?,
            gate_out: m.function("ink_gate_out")?,
            gate_g: m.function("ink_gate_g")?,
            gate_mix: m.function("ink_gate_mix")?,
            router_logits: m.function("ink_router_logits")?,
            router_pick: m.function("ink_router_pick")?,
            gemm_s: m.function("ink_gemm_s")?,
            gemm_s8: m.function("ink_gemm_s8")?,
            dequant: kk.module.function("ink_dequant_fp8")?,
        };
        let mut ck = Ckpt::open(&model_dir.join("model-00001-of-00001.safetensors"))?;
        let fp8 = true;
        mem_mark(&ctx, "kernels (modules)", &last);
        let w = weights::load(&mut ck, &kk, &s, fp8)?;
        mem_mark(&ctx, "weights (experts slots + dense + rest)", &last);
        if memlog {
            let ex: usize = w.layers.iter().map(|l| if let Experts::Slot { sl, .. } = &*l.experts { sl.lock().unwrap().dev_bytes() } else { 0 }).sum();
            eprintln!("ink mem:   of which expert slot blocks {:.1} MiB", ex as f64 / 1048576.0);
        }
        // tmax == 0: size the prompt chunk from the free memory left after the weights (measured on a 16 GB RTX 5060 Ti: ~1.95 MiB per chunk
        // row for the sliding rings, conv state, layer and MoE scratch and the attention scratch; K/V of the full layers 32 KiB per position
        // of cap; ~1030 MiB of rings / staging pool / misc that do not depend on the chunk; 200 MiB margin, plus INK_RESERVE_MB, e.g. 800 for the draft head). Rounded down to 64 rows, 64..=1024.
        let kvq6 = crate::env_flag("INK_KVQ6");
        let kv_lazy = crate::env_flag("INK_KV_LAZY");
        let tmax = if tmax == 0 {
            let (free, _) = ctx.mem_info()?;
            let reserve = (200usize + std::env::var("INK_RESERVE_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0)) << 20;
            // lazily mapped K/V (INK_KV_LAZY) is not part of the load-time footprint: INK_FIT_MB makes room for it when it is mapped
            let kv_fixed = if kv_lazy { 0 } else { cap * if kvq6 { 13312 } else { 32 * 1024 } };
            let fixed = kv_fixed + (1030usize << 20) + reserve;
            let rows = free.saturating_sub(fixed) as f64 / (1.95 * 1048576.0);
            let t = ((rows as usize) / 64 * 64).clamp(64, 1024);
            eprintln!("ink: chunk sized from free memory: {:.2} GiB free after the weights, cap {cap} -> tmax {t}", free as f64 / 1073741824.0);
            t
        } else {
            tmax
        };
        let mut apps = Vec::new();
        for _ in 0..NAPP {
            apps.push(App {
                k: mk_kv(li_of(apps.len()), cap, tmax, kvq6, kv_lazy, dev)?,
                v: mk_kv(li_of(apps.len()), cap, tmax, kvq6, kv_lazy, dev)?,
                conv_k: Buf::from_slice(&vec![0u8; (3 + tmax) * 1024 * 4])?, conv_v: Buf::from_slice(&vec![0u8; (3 + tmax) * 1024 * 4])?,
                conv_a: Buf::from_slice(&vec![0u8; (3 + tmax) * D * 4])?, conv_m: Buf::from_slice(&vec![0u8; (3 + tmax) * D * 4])?,
            });
        }
        mem_mark(&ctx, &format!("K/V + rings + conv (cap {cap})"), &last);
        let t = tmax;
        let sc = Scratch {
            hs: [alloc(t * D * 2)?, alloc(t * D * 2)?], pre_hs: alloc(t * D * 2)?, prev12: alloc(t * D * 2)?,
            prevfin: alloc(t * D * 2)?, blocks: alloc(t * 4 * D * 2)?, ai: alloc(t * 3 * D * 4)?, msg: alloc(t * 4 * D * 4)?,
            gh: alloc(t * D * 4)?, gh2: [alloc(t * D * 4)?, alloc(t * D * 4)?], gg2: [alloc(t * 4608 * 2)?, alloc(t * 4608 * 2)?], gz2: [alloc(t * 64 * 2)?, alloc(t * 64 * 2)?], x: [alloc(t * D * 2)?, alloc(t * D * 2)?, alloc(t * D * 2)?],
            qgr: alloc(t * 6528 * 2)?, klin: alloc(t * 1024 * 2)?, vlin: alloc(t * 1024 * 2)?,
            slin: alloc(t * 24 * 2)?, dynk: alloc(t * 32 * 2)?, dynv: alloc(t * 32 * 2)?,
            qp: alloc(t * 3072 * 2)?, rsx: alloc(t * 384 * 2)?, kcv: alloc(t * 1024 * 2)?, vcv: alloc(t * 1024 * 2)?,
            att: alloc(t * 3072 * 2)?, gat: alloc(t * 3072 * 2)?, o: alloc(t * D * 2)?, a2: alloc(t * D * 2)?,
            resid: alloc(t * D * 4)?, mi: alloc(t * D * 4)?, mg: alloc(t * D * 4)?, xm: alloc(t * D * 2)?,
            picks: alloc(t * 8 * 4)?, wts: alloc(t * 8 * 4)?, rlog: alloc(t * 128 * 4)?, forced: alloc(t * 8 * 4)?,
            moe: alloc(t * D * 2)?, mc: alloc(t * D * 2)?,
            gz: alloc(t * 64 * 2)?, gzf: alloc(t * 64 * 4)?, gg: alloc(t * 4608 * 2)?,
            fin: alloc(t * D * 4)?, xf: alloc(t * D * 2)?, logits: alloc(if std::env::var_os("INK_HOST_EXPERTS").is_some() { t.min(16) } else { t } * VOCAB * 2)?,
        };
        mem_mark(&ctx, &format!("layer scratch (tmax {tmax})"), &last);
        let ntl_max = (cap.div_ceil(128) + 1).max(10);
        mem_mark(&ctx, "decode MoE scratch", &last);
        // split-attention partials: 256 MiB bound them, but one 16-query tile of the whole context must fit
        let att_mb: usize = 256;
        let att_bytes = (att_mb << 20).max(24 * ntl_max.max(10) * 8192).min((tmax / 16 + 1) * 24 * ntl_max.max(10) * 8192);
        let pf_scr = fe_inkk::PrefillScratch::new(tmax)?;
        mem_mark(&ctx, &format!("prefill MoE scratch (tmax {tmax})"), &last);
        let me = Ink { ctx, s, k, w, ck, apps, sc, cap, tmax, len: 0, prof_on: false, kk, prompt: true, spec: false, argmax_only: false, last_argmax: vec![], am_out: alloc(64 * 4)?, pending: 0, draft_taps: false, dtaps: alloc(tmax * 4 * D * 2)?,
            dq: alloc(6528 * 2304 * 2)?, ast: alloc(16 << 20)?, apt: if kv_lazy { KvBuf::Lazy(fe_cuda::VmBuffer::reserve(att_bytes, dev)?) } else { KvBuf::Plain(alloc(att_bytes)?) }, fit_floor: std::env::var("INK_FIT_MB").ok().and_then(|v| v.parse::<usize>().ok()).map(|m| m << 20), dg: true, dg_slide: false, dg2: true, att_mb, apt_reserved: att_bytes, capturing: false, posbuf: alloc(16)?, graphs: std::collections::HashMap::new(), kvq6, kstg: alloc(if kvq6 { tmax * 2048 } else { 16 })?, vstg: alloc(if kvq6 { tmax * 2048 } else { 16 })?, posbuf0: Buf::from_slice(&[0u8; 16])?, use_graph: true, pf_scr,
            aux: [Stream::new()?, Stream::new()?], ev_fork: fe_cuda::Event::new_no_timing()?,
            ev_join: [fe_cuda::Event::new_no_timing()?, fe_cuda::Event::new_no_timing()?],
            prof: Default::default(), evs: Default::default(), last: Default::default(), slots: None };
        let mut me = me;
        let nmax = me.w.layers.iter().filter_map(|l| if let Experts::Slot { sl, .. } = &*l.experts { Some(sl.lock().unwrap().n_host()) } else { None }).max();
        if let Some(nmax) = nmax {
            me.slots = Some(SlotRt {
                spare: fe_inkk::slots::SparePool::new(2 * nmax.min(4))?, nmax: nmax.min(20), copy_s: Stream::new()?,
                ev_st: [fe_cuda::Event::new_no_timing()?, fe_cuda::Event::new_no_timing()?],
                ev_dn: [fe_cuda::Event::new_no_timing()?, fe_cuda::Event::new_no_timing()?],
                ev_un: (0..NL).map(|_| fe_cuda::Event::new_no_timing()).collect::<Result<_, _>>()?,
                warm_adapts: 0, hook_ms: 0.0, adapt_every: 64, since: 0, prompt_seen: false,
                log: crate::env_flag("INK_SLOT_LOG"), host_use: 0, total_use: 0, swaps: 0, adapts: 0, prompt_host_use: 0, prompt_total_use: 0, staged: std::cell::Cell::new(None),
                prior: crate::heat_share()?,
                host_target: std::env::var("INK_HOST_EXPERTS").ok().and_then(|v| v.parse().ok()).unwrap_or(0) });
        }
        mem_mark(&me.ctx, "attention scratch, staging pool, misc", &last);
        let (free, total) = me.ctx.mem_info()?;
        eprintln!("ink: loaded, device memory free {:.2} GiB of {:.2}", free as f64 / 1073741824.0, total as f64 / 1073741824.0);
        Ok(me)
    }

    /// Slot residency schedule (the stream is idle at forward() entry): one adapt after the prompt, then every `adapt_every` decode tokens.
    fn slot_hook(&mut self, t: usize) -> Result<()> {
        let t_hook = std::time::Instant::now();
        let r = self.slot_hook_inner(t);
        if let Some(rt) = self.slots.as_mut() {
            rt.hook_ms += t_hook.elapsed().as_secs_f64() * 1e3;
        }
        r
    }

    fn slot_hook_inner(&mut self, t: usize) -> Result<()> {
        if self.slots.is_some() && self.prompt {
            self.slot_finish()?;
        }
        let Some(rt) = self.slots.as_mut() else { return Ok(()) };
        if self.prompt {
            rt.prompt_seen = true;
            return Ok(());
        }
        let first = rt.prompt_seen;
        if first {
            rt.warm_adapts = 0;
        }
        if first || rt.since >= rt.adapt_every {
            rt.prompt_seen = false;
            rt.since = 0;
            let (mut swaps, mut hu, mut tu) = (0usize, 0u64, 0u64);
            for l in self.w.layers.iter() {
                if let Experts::Slot { sl, .. } = &*l.experts {
                    let st = if first || rt.warm_adapts < 3 {
                        // the one big adapt after the prompt (up to 16 swaps a layer) is done at once: spread over the next forwards it would leave the host reads high for ~50 tokens
                        sl.lock().unwrap().adapt(&self.s, if first { 16 } else { 8 })?
                    } else {
                        sl.lock().unwrap().adapt_plan(&self.s, if first { 16 } else { 8 })?
                    };
                    swaps += st.swaps;
                    hu += st.host_use;
                    tu += st.total_use;
                }
            }
            rt.swaps += swaps;
            if first {
                rt.prompt_host_use += hu;
                rt.prompt_total_use += tu;
            } else {
                rt.host_use += hu;
                rt.total_use += tu;
            }
            rt.adapts += 1;
            rt.warm_adapts += 1;
            if rt.log {
                let fb: usize = self.w.layers.iter().filter_map(|l| if let Experts::Slot { sl, .. } = &*l.experts { Some(sl.lock().unwrap().fallbacks) } else { None }).sum();
                eprintln!("ink slots: [hook {:.0} ms so far, {} sync fallbacks] adapt #{}{}: {} swaps, host-expert reads since the last adapt {} of {} expert uses ({:.2}%)", rt.hook_ms, fb, rt.adapts,
                    if first { " (after the prompt)" } else { "" }, swaps, hu, tu, 100.0 * hu as f64 / tu.max(1) as f64);
            }
        }
        rt.since += t;
        {
            // pipelined adapt: one transition of each layer's swap per forward, the PCIe copies run on the copy stream beside the graph
            for l in self.w.layers.iter() {
                if let Experts::Slot { sl, .. } = &*l.experts {
                    let mut g = sl.lock().unwrap();
                    if g.adapt_busy() {
                        g.adapt_step(&self.s, &rt.copy_s)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Finish every queued or in-flight pipelined swap (before prompt staging or a change of the resident count).
    fn slot_finish(&mut self) -> Result<()> {
        let Some(rt) = self.slots.as_ref() else { return Ok(()) };
        for l in self.w.layers.iter() {
            if let Experts::Slot { sl, .. } = &*l.experts {
                let mut g = sl.lock().unwrap();
                if g.adapt_busy() {
                    g.adapt_finish(&self.s, &rt.copy_s)?;
                }
            }
        }
        Ok(())
    }

    /// Decode-phase (host-expert reads, expert uses) so far, including the counters not yet folded by an adapt; None without host experts.
    pub fn slot_decode_reads(&self) -> Result<Option<(u64, u64)>> {
        let Some(rt) = self.slots.as_ref() else { return Ok(None) };
        let (mut h, mut t) = (rt.host_use, rt.total_use);
        if !rt.prompt_seen {
            for l in self.w.layers.iter() {
                if let Experts::Slot { sl, .. } = &*l.experts {
                    let (a, b) = sl.lock().unwrap().peek_use(&self.s)?;
                    h += a;
                    t += b;
                }
            }
        }
        Ok(Some((h, t)))
    }

    /// Change the number of host-resident experts (all layers together, coldest by counters + heat prior): device memory is released
    /// (or taken) in blocks of 4 slots. The stream must be idle (call between forwards). Bits do not change.
    pub fn set_host_experts(&mut self, n: usize) -> Result<fe_inkk::slots::TargetStats> {
        self.slot_finish()?;
        let rt = self.slots.as_mut().ok_or_else(|| anyhow::anyhow!("no slot layers (INK_HOST_EXPERTS=N at load)"))?;
        let mut guards: Vec<std::sync::MutexGuard<fe_inkk::slots::SlotLayer>> = vec![];
        for l in self.w.layers.iter() {
            if let Experts::Slot { sl, .. } = &*l.experts {
                guards.push(sl.lock().unwrap());
            }
        }
        let mut refs: Vec<&mut fe_inkk::slots::SlotLayer> = guards.iter_mut().map(|g| &mut **g).collect();
        let st = fe_inkk::slots::set_host_target(&mut refs, &self.s, n, &rt.prior, 64)?;
        rt.host_target = n;
        Ok(st)
    }

    /// Keep `min_free` bytes of device memory free by moving the host-expert count (stream idle): when free memory is below `min_free`
    /// more experts go to the host; when it exceeds `min_free + slack` experts are promoted back until the free memory is `min_free`
    /// again. Returns the new host count if it changed. Use before / after K/V or scratch growth.
    pub fn fit_host_experts(&mut self, min_free: usize, slack: usize) -> Result<Option<usize>> {
        let slot = 4_492_800usize; // bytes of one expert slot (NVFP4: w13 + w2 codes and scales)
        let start = self.slots.as_ref().map(|r| r.host_target).unwrap_or(0);
        for _ in 0..4 {
            let (free, _) = self.ctx.mem_info()?;
            let cur = self.slots.as_ref().map(|r| r.host_target).unwrap_or(0);
            // aim 5% above the minimum: slots are held in blocks and the driver rounds allocations
            let aim = min_free + min_free / 20;
            let want = if free < min_free {
                cur + (aim - free).div_ceil(slot)
            } else if free > min_free + slack {
                cur.saturating_sub((free - aim) * 3 / 4 / slot)
            } else {
                break;
            };
            let want = want.min(24 * 64);
            if want == cur {
                break;
            }
            self.set_host_experts(want)?;
        }
        let now = self.slots.as_ref().map(|r| r.host_target).unwrap_or(0);
        Ok(if now != start { Some(now) } else { None })
    }

    /// Current host-expert target (0 without slot layers).
    pub fn host_experts(&self) -> usize {
        self.slots.as_ref().map(|r| r.host_target).unwrap_or(0)
    }

    /// Device bytes held by the shared slot pool.
    pub fn slot_pool_bytes(&self) -> usize {
        self.w.layers.iter().find_map(|l| if let Experts::Slot { sl, .. } = &*l.experts { Some(sl.lock().unwrap().pool_bytes()) } else { None }).unwrap_or(0)
    }

    /// Device bytes held by the expert slot blocks (all layers).
    pub fn slot_dev_bytes(&self) -> usize {
        self.w.layers.iter().map(|l| if let Experts::Slot { sl, .. } = &*l.experts { sl.lock().unwrap().dev_bytes() } else { 0 }).sum()
    }

    /// (adapts, swaps, host-expert reads, expert uses) so far; None without host experts.
    pub fn slot_stats(&self) -> Option<(usize, usize, u64, u64)> {
        self.slots.as_ref().map(|r| (r.adapts, r.swaps, r.host_use, r.total_use))
    }

    /// Time since the previous tick (device synchronised) added to `name`; a no-op unless profiling.
    fn tick(&self, name: &'static str) {
        if !self.prof_on {
            return;
        }
        if let Ok(e) = fe_cuda::Event::new() {
            let _ = self.s.record(&e);
            self.evs.borrow_mut().push((name, e));
        }
    }
    /// Fold the recorded events into the per-stage account (GPU time between consecutive events).
    fn prof_fold(&self) {
        let _ = self.s.sync();
        let mut evs = self.evs.borrow_mut();
        let mut p = self.prof.borrow_mut();
        for w in evs.windows(2) {
            if let Ok(ms) = w[1].1.elapsed_since(&w[0].1) {
                *p.entry(w[1].0).or_insert(0.0) += ms as f64 * 1e-3;
            }
        }
        evs.clear();
    }
    pub fn prof_report(&self, forwards: usize) -> String {
        let p = self.prof.borrow();
        let tot: f64 = p.values().sum();
        let mut v: Vec<_> = p.iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let mut o = format!("per-forward account over {forwards} forwards, total {:.2} ms each:\n", tot * 1e3 / forwards as f64);
        for (k, t) in v {
            o += &format!("  {k:<14} {:8.3} ms  {:5.1}%\n", t * 1e3 / forwards as f64, 100.0 * t / tot);
        }
        o
    }
    pub fn prof_clear(&mut self) {
        self.prof.borrow_mut().clear();
    }

    pub fn reset(&mut self) -> Result<()> {
        for a in &self.apps {
            for (b, n) in [(&a.conv_k, 3 * 1024), (&a.conv_v, 3 * 1024), (&a.conv_a, 3 * D), (&a.conv_m, 3 * D)] {
                self.s.memset_d8(b.ptr(), 0, n * 4)?;
            }
        }
        self.len = 0;
        self.pending = 0;
        for a in self.apps.iter_mut() {
            a.k.shrink(0)?;
            a.v.shrink(0)?;
        }
        // the K/V memory is back: promote experts again up to the floor (INK_FIT_MB)
        if let (Some(floor), true) = (self.fit_floor, self.slots.is_some()) {
            self.fit_host_experts(floor, floor)?;
        }
        Ok(())
    }

    /// Map K/V memory for the global applications up to `ctx` tokens, in steps of 2048 tokens (INK_KV_LAZY=1; eager buffers: no-op).
    pub fn ensure_kv(&mut self, ctx: usize) -> Result<()> {
        let per_tok = if self.kvq6 { 832 } else { 2048 };
        let want = ((ctx + 2047) / 2048 * 2048).min(self.cap) * per_tok;
        // INK_FIT_MB=n: keep n MiB of device memory free by moving experts to the host BEFORE the mapping below needs the memory (stream idle)
        if let (Some(floor), true) = (self.fit_floor, self.slots.is_some()) {
            let gran = 2usize << 20;
            let mut need = 0usize;
            for (i, a) in self.apps.iter().enumerate() {
                if is_full(li_of(i)) {
                    let w = want.min(a.k.reserved()).next_multiple_of(gran);
                    need += w.saturating_sub(a.k.mapped()) + w.saturating_sub(a.v.mapped());
                }
            }
            let nsup = (ctx.div_ceil(128) + 1).max(10);
            let tile = (24 * nsup * 8192).max(24 * (self.cap.div_ceil(128) + 1) * 512);
            let apt_want = tile.max(self.att_mb << 20).min(self.apt_reserved);
            need += apt_want.saturating_sub(self.apt.mapped());
            if need > 0 {
                // the draft head grows after the target's forward: 4.5 KiB a position for this step plus one 18 MiB growth step
                let nfull = self.apps.iter().enumerate().filter(|(i, _)| is_full(li_of(*i))).count().max(1);
                need += (need / (nfull * 2 * per_tok)).min(4096) * 4608 + (18 << 20);
            }
            if need > 0 {
                // evict as needed; never promote here (slack = everything)
                if let Some(n) = self.fit_host_experts(need + floor, usize::MAX / 4)? {
                    if self.slots.as_ref().map_or(false, |r| r.log) {
                        eprintln!("ink slots: K/V mapping of {:.0} MiB for {ctx} tokens: host experts -> {n}", need as f64 / 1048576.0);
                    }
                }
            }
        }
        for a in self.apps.iter_mut() {
            a.k.grow(want)?;
            a.v.grow(want)?;
        }
        let nsup = (ctx.div_ceil(128) + 1).max(10);
        let tile = (24 * nsup * 8192).max(24 * (self.cap.div_ceil(128) + 1) * 512);
        self.apt.grow(tile.max(self.att_mb << 20).min(self.apt_reserved))?;
        Ok(())
    }

    /// Bytes of K/V memory currently committed for the global applications.
    pub fn kv_mapped_bytes(&self) -> usize {
        self.apps.iter().enumerate().filter(|(i, _)| is_full(li_of(*i))).map(|(_, a)| a.k.mapped() + a.v.mapped()).sum()
    }

    /// Commit the pending speculative rows: the conv states keep rows [keep, keep + 3) as their 3 previous inputs.
    fn commit(&mut self, keep: usize) -> Result<()> {
        for a in &self.apps {
            for (b, c) in [(&a.conv_k, 1024usize), (&a.conv_v, 1024), (&a.conv_a, D), (&a.conv_m, D)] {
                unsafe { self.s.launch(&self.k.conv_commit, (c as u32 + 255) / 256, 256u32, 0, (b.ptr(), c as u32, keep as u32))? };
            }
        }
        Ok(())
    }

    /// One plain decode row (the CUDA-graph path, dtaps row 0 filled when `draft_taps` is on) leaving its logits on the device; returns the greedy token.
    pub fn decode_argmax(&mut self, tok: u32) -> Result<u32> {
        self.argmax_only = true;
        let r = self.forward(&[tok], 1, None, None);
        self.argmax_only = false;
        r?;
        Ok(self.last_argmax[0])
    }

    /// Device pointer of the bf16 logits [rows][VOCAB] of the last forward.
    pub fn logits_ptr(&self) -> DevPtr {
        self.sc.logits.ptr()
    }

    /// Speculative forward returning the greedy token of every row (largest bf16 logit, lowest index on ties, on the device) instead of the logits.
    pub fn spec_argmax(&mut self, ids: &[u32]) -> Result<Vec<u32>> {
        self.spec = true;
        self.argmax_only = true;
        let r = self.forward(ids, ids.len(), None, None);
        self.spec = false;
        self.argmax_only = false;
        r?;
        Ok(self.last_argmax.clone())
    }

    /// After a forward with `spec` set: keep the first `keep` of its rows (len and the conv tails as of row keep - 1; K/V rows past len are
    /// never read: global caches are indexed by position, sliding rings hold window + chunk rows).
    pub fn rollback(&mut self, keep: usize) -> Result<()> {
        ensure!(keep <= self.pending);
        let p = self.pending;
        self.commit(keep)?;
        self.len = self.len - p + keep;
        self.pending = 0;
        Ok(())
    }

    /// Bytes of one prompt-cache snapshot of the engine state (see [`Ink::snapshot`]): the sliding layers' K/V window and the conv tails.
    pub fn snap_bytes(&self) -> usize {
        let sliding = (0..NAPP).filter(|&i| !is_full(li_of(i))).count();
        sliding * 2 * SNAP_WIN * 2048 + NAPP * CONV_SNAP
    }

    /// Copy the state at position `mark` that later rows overwrite into `dst` (pinned host memory, [`Ink::snap_bytes`] bytes): the K/V of
    /// positions [mark - 1024, mark) of the 40 sliding-layer applications (their rings are reused modulo 1024 + chunk rows) and the three conv
    /// tails of all 48 applications. The global applications' K/V is indexed by position and only appended to: its rows below `mark` stay
    /// valid until a forward writes those positions again, so it is not copied. Two cases: no rows pending and mark == len, or the last
    /// forward was a speculative chunk (its conv inputs kept after the 3 state rows) and mark lies in that chunk [len - pending, len].
    pub fn snapshot(&self, dst: &mut [u8], mark: usize) -> Result<()> {
        let c0 = self.len - self.pending;
        ensure!(mark >= c0 && mark <= self.len && (self.pending > 0 || mark == self.len), "snapshot at {mark}: state at {} with {} pending rows", self.len, self.pending);
        ensure!(dst.len() >= self.snap_bytes(), "snapshot buffer too small");
        self.s.sync()?;
        let row0 = mark - c0;
        let w = mark.min(SNAP_WIN);
        let mut off = 0;
        for (i, a) in self.apps.iter().enumerate() {
            if is_full(li_of(i)) {
                continue;
            }
            let r = ring_of(li_of(i), self.cap, self.tmax);
            for b in [&a.k, &a.v] {
                for (dev_row, host_row, rows) in ring_segments(mark - w, w, r) {
                    unsafe { self.s.memcpy_dtoh(&mut dst[off + host_row * 2048..off + (host_row + rows) * 2048], b.ptr().byte_add(dev_row * 2048))? };
                }
                off += SNAP_WIN * 2048;
            }
        }
        for a in &self.apps {
            for (b, c) in [(&a.conv_k, 1024usize), (&a.conv_v, 1024), (&a.conv_a, D), (&a.conv_m, D)] {
                let n = 3 * c * 4;
                unsafe { self.s.memcpy_dtoh(&mut dst[off..off + n], b.ptr().byte_add(row0 * c * 4))? };
                off += n;
            }
        }
        self.s.sync()?;
        Ok(())
    }

    /// Put back a snapshot taken at `mark` (see [`Ink::snapshot`]): the engine continues at position `mark`. The global K/V rows below
    /// `mark` must not have been written since the snapshot was taken (the prompt cache keeps that invariant).
    pub fn restore(&mut self, src: &[u8], mark: usize) -> Result<()> {
        ensure!(src.len() >= self.snap_bytes(), "snapshot buffer too small");
        ensure!(mark < self.cap, "snapshot at {mark} beyond the cache {}", self.cap);
        self.ctx.bind()?;
        self.s.sync()?;
        let plant = std::env::var("INK_PCACHE_PLANT").unwrap_or_default();
        let w = mark.min(SNAP_WIN);
        let mut off = 0;
        for (i, a) in self.apps.iter().enumerate() {
            if is_full(li_of(i)) {
                continue;
            }
            let r = ring_of(li_of(i), self.cap, self.tmax);
            for b in [&a.k, &a.v] {
                for (dev_row, host_row, rows) in ring_segments(mark - w, w, r) {
                    if plant == "ring" && i == 7 {
                        continue;
                    }
                    unsafe { self.s.memcpy_htod(b.ptr().byte_add(dev_row * 2048), &src[off + host_row * 2048..off + (host_row + rows) * 2048])? };
                }
                off += SNAP_WIN * 2048;
            }
        }
        for (i, a) in self.apps.iter().enumerate() {
            for (b, c) in [(&a.conv_k, 1024usize), (&a.conv_v, 1024), (&a.conv_a, D), (&a.conv_m, D)] {
                let n = 3 * c * 4;
                if !(plant == "conv" && i == 20 && c == D) {
                    unsafe { self.s.memcpy_htod(b.ptr(), &src[off..off + n])? };
                }
                off += n;
            }
        }
        self.s.sync()?;
        self.len = mark;
        self.pending = 0;
        self.spec = false;
        Ok(())
    }

    /// Rows of the last forward still pending (a speculative chunk not yet committed).
    pub fn pending_rows(&self) -> usize {
        self.pending
    }

    /// Test hook: ink_headclip on one row (`heads` x `dh` bf16) at position `pos` (log scaling on/off).
    pub fn test_headclip(&mut self, x: &[u16], scale: &[u16], heads: usize, dh: usize, thr: f32, log: bool, pos: u32) -> Result<Vec<u16>> {
        let xb = Buf::alloc(x.len() * 2)?;
        let sb = Buf::alloc(scale.len() * 2)?;
        let ob = Buf::alloc(x.len() * 2)?;
        fe_cuda::memcpy_htod(xb.ptr(), x)?;
        fe_cuda::memcpy_htod(sb.ptr(), scale)?;
        fe_cuda::memcpy_htod(self.posbuf.ptr(), &[pos])?;
        unsafe {
            self.s.launch(&self.k.headclip, ((heads as u32 + 7) / 8, 1u32), 256u32, 0, (ob.ptr(), xb.ptr(), (heads * dh) as u32, sb.ptr(), heads as u32,
                dh as u32, thr, log as u32, self.posbuf.ptr(), 1u32, 8192.0f32, 0.1f32))?;
        }
        self.s.sync()?;
        let mut out = vec![0u16; x.len()];
        fe_cuda::memcpy_dtoh(&mut out, ob.ptr())?;
        Ok(out)
    }

    /// Embedding rows of `ids` (host gather) through embed_norm into `dst` (bf16 [ids.len()][D]).
    pub fn embed_into(&mut self, ids: &[u32], dst: DevPtr) -> Result<()> {
        let t = ids.len();
        let mut raw = vec![0u8; t * D * 2];
        for (i, &id) in ids.iter().enumerate() {
            let r = self.ck.read_range("model.llm.embed.weight", id as usize * D * 2, D * 2)?;
            raw[i * D * 2..(i + 1) * D * 2].copy_from_slice(&r);
        }
        unsafe { self.s.memcpy_htod(self.sc.x[0].ptr(), &raw)? };
        self.rms(dst, self.sc.x[0].ptr(), false, D, self.w.embed_norm.ptr(), t)
    }

    /// Unembed `rows` (<= 16 per call, looped) bf16 rows of `x` into bf16 logits (softcap not applied).
    pub fn unembed(&self, y: DevPtr, x: DevPtr, rows: usize) -> Result<()> {
        let l = &self.w.unembed;
        let mut r0 = 0;
        while r0 < rows {
            let g = (rows - r0).min(16);
            self.kk.fp8_forward(&self.s, l.f8.as_ref().unwrap(), &self.k.gemm, NUL, at_ptr(y, r0 * VOCAB * 2), VOCAB, at_ptr(x, r0 * D * 2), D, g, None)?;
            r0 += g;
        }
        Ok(())
    }

    /// C[m][n] = A[m][k] . W[n][k]^T (bf16). Rows <= 16 take the 32-column small-M kernel (same mma, same k order, so the
    /// same bits as the 128x128 kernel), more rows the tensor-core tile kernel.
    pub fn gemm(&self, st: &Stream, c: DevPtr, a: DevPtr, w: DevPtr, m: usize, n: usize, k: usize, lda: usize, ldc: usize) -> Result<()> {
        unsafe {
            if (m <= 16 || n <= 128) && k % 64 == 0 {
                // same mma and k order as the tile kernel: bits do not depend on which of the two runs
                fe_inkk::pdl::launch(st, &self.k.gemm_s, ((n as u32 + 31) / 32, (m as u32 + 15) / 16), 128u32, SGEMM_SMEM,
                    (c, a, w, m as u32, n as u32, k as u32, lda as u32, ldc as u32))?;
            } else {
                st.launch(&self.k.gemm, ((m as u32 + 127) / 128, (n as u32 + 127) / 128), 256u32, GEMM_SMEM,
                    (c, a, w, NUL, m as u32, n as u32, k as u32, lda as u32, ldc as u32))?;
            }
        }
        Ok(())
    }

    /// y[m][n] = x[m][k] . W^T with the layer's linear (bf16 or FP8).
    pub fn lin(&self, st: &Stream, y: DevPtr, x: DevPtr, l: &weights::Lin, m: usize, lda: usize, ldy: usize) -> Result<()> {
        if let Some(b) = &l.bf {
            return self.gemm(st, y, x, b.ptr(), m, l.n, l.k, lda, ldy);
        }
        let f = l.f8.as_ref().unwrap();
        if m <= 16 && l.k % 64 == 0 {
            // FP8 weights through the same mma and k order as dequantise-then-GEMM: bits of the prompt family, half the bytes
            unsafe {
                fe_inkk::pdl::launch(st, &self.k.gemm_s8, ((l.n as u32 + 31) / 32, 1u32), 128u32, SGEMM8_SMEM,
                    (y, x, f.w8.ptr(), f.scale.ptr(), m as u32, l.n as u32, l.k as u32, lda as u32, ldy as u32))?;
            }
        } else {
            ensure!(l.n * l.k <= 6528 * 2304, "dense scratch too small");
            unsafe {
                fe_inkk::pdl::launch(st, &self.k.dequant, l.n as u32, 256u32, 0, (self.dq.ptr(), f.w8.ptr(), f.scale.ptr(), l.n as u32, l.k as u32))?;
            }
            self.gemm(st, y, x, self.dq.ptr(), m, l.n, l.k, lda, ldy)?;
        }
        Ok(())
    }

    fn rms(&self, y: DevPtr, x: DevPtr, in_f32: bool, ldx: usize, w: DevPtr, rows: usize) -> Result<()> {
        unsafe {
            fe_inkk::pdl::launch(&self.s, &self.k.rms, rows as u32, 256u32, 0,
                (y, x, in_f32 as u32, ldx as u32, w, D as u32, 1e-6f32, D as u32))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn blockres(&self, out: DevPtr, ldo: usize, blocks: DevPtr, bstride: usize, nb: usize, hs: DevPtr, hs_f32: bool,
        ldh: usize, query: DevPtr, nq: usize, t: usize) -> Result<()> {
        self.kk.blockres(&self.s, out, ldo, blocks, bstride, nb, hs, hs_f32, ldh, query, nq, t)
    }

    /// Decode-path message gate + the layer norm of its result on stream `st`, scratch set `k` (0 = main, 1..3 = aux).
    #[allow(clippy::too_many_arguments)]
    fn gate_decode(&self, st: &Stream, k: usize, out: DevPtr, h: DevPtr, ldh: usize, m: DevPtr, ldm: usize, g: &weights::Gate,
        t: usize, ln_out: DevPtr, ln_w: DevPtr) -> Result<()> {
        let (gg, gz, gh) = if k == 0 { (self.sc.gg.ptr(), self.sc.gz.ptr(), self.sc.gh.ptr()) }
            else { (self.sc.gg2[k - 1].ptr(), self.sc.gz2[k - 1].ptr(), self.sc.gh2[k - 1].ptr()) };
        let _ = out;
        unsafe {
            fe_inkk::pdl::launch(st, &self.k.gate_g, t as u32, 256u32, 0, (gg, h, ldh as u32, m, ldm as u32))?;
        }
        self.gemm(st, gz, gg, g.down.ptr(), t, 64, 4608, 4608, 64)?;
        unsafe {
            fe_inkk::pdl::launch(st, &self.k.gate_mix, (((t + 15) / 16) as u32, 72u32), 128u32, 0,
                (gh, D as u32, h, ldh as u32, m, ldm as u32, gz, g.forget.ptr(), g.gain.ptr(), t as u32))?;
            fe_inkk::pdl::launch(st, &self.k.rms, t as u32, 256u32, 0, (ln_out, gh, 1u32, D as u32, ln_w, D as u32, 1e-6f32, D as u32))?;
        }
        Ok(())
    }

    fn msg_gate(&self, out: DevPtr, h: DevPtr, ldh: usize, m: DevPtr, ldm: usize, g: &weights::Gate, t: usize) -> Result<()> {
        if false {
            // (old decode-only per-token kernels, kept for reference)
            unsafe {
                fe_inkk::pdl::launch(&self.s, &self.k.gate_z, (t as u32, 16u32), 128u32, 0, (self.sc.gzf.ptr(), h, ldh as u32, m, ldm as u32, g.down.ptr()))?;
                fe_inkk::pdl::launch(&self.s, &self.k.gate_out, (t as u32, 9u32), 256u32, 0,
                    (out, D as u32, h, ldh as u32, m, ldm as u32, self.sc.gzf.ptr(), g.forget.ptr(), g.gain.ptr()))?;
            }
            return Ok(());
        }
        self.tick("g_pre");
        unsafe {
            fe_inkk::pdl::launch(&self.s, &self.k.gate_g, t as u32, 256u32, 0, (self.sc.gg.ptr(), h, ldh as u32, m, ldm as u32))?;
        }
        self.tick("g_g");
        self.gemm(&self.s, self.sc.gz.ptr(), self.sc.gg.ptr(), g.down.ptr(), t, 64, 4608, 4608, 64)?;
        self.tick("g_gemm");
        unsafe {
            fe_inkk::pdl::launch(&self.s, &self.k.gate_mix, (((t + 15) / 16) as u32, 72u32), 128u32, 0,
                (out, D as u32, h, ldh as u32, m, ldm as u32, self.sc.gz.ptr(), g.forget.ptr(), g.gain.ptr(), t as u32))?;
        }
        self.tick("g_out");
        Ok(())
    }

    fn sconv(&self, st: &Stream, out: DevPtr, x: DevPtr, w: DevPtr, state: DevPtr, t: usize, c: usize) -> Result<()> {
        unsafe {
            fe_inkk::pdl::launch(st, &self.k.sconv, (c as u32 + 255) / 256, 256u32, 0,
                (out, c as u32, x, c as u32, w, state, t as u32, c as u32, self.spec as u32))?;
        }
        Ok(())
    }

    /// One layer application. Reads `hs_in`, writes `hs_out`. `msgs`: messages of the previous pass [T][4][D] f32.
    #[allow(clippy::too_many_arguments)]
    fn apply(&mut self, li: usize, pass: usize, nb: &mut usize, hs_in: DevPtr, hs_out: DevPtr, msgs: bool, t: usize,
        pos0: usize, forced: Option<&[i32]>, tap: Option<&mut Tap>) -> Result<()> {
        let lw = &self.w.layers[li];
        let app = &self.apps[app_index(li, pass)];
        let sc = &self.sc;
        let full = lw.full;
        let boundary = li % 6 == 0;
        let ring = ring_of(li, self.cap, self.tmax) as u32;
        let q6 = self.kvq6 && full;
        // 6-bit K/V: the writers fill a bf16 staging row per chunk token (row t), the packer then writes the record
        let (kw, vw, wp, wr) = if q6 { (self.kstg.ptr(), self.vstg.ptr(), self.posbuf0.ptr(), self.tmax as u32) } else { (app.k.ptr(), app.v.ptr(), self.posbuf.ptr(), ring) };
        let nq = if full { 3 } else { 1 };
        let s = &self.s;
        let dec = !self.prompt && t <= 16;
        // G16 prompt chunks (> 32 rows): stage this layer's host experts into spare slots on the copy stream while its attention runs
        let stage_par: Option<usize> = if self.prompt && t > 32 && self.slots.is_some() && matches!(&*lw.experts, Experts::Slot { .. }) {
            let rt = self.slots.as_ref().unwrap();
            let ai = app_index(li, pass);
            let stage_one = |l: usize, a: usize| -> Result<()> {
                if let Experts::Slot { sl, .. } = &*self.w.layers[l].experts {
                    let par = a % 2;
                    rt.copy_s.wait(&rt.ev_dn[par])?;
                    rt.copy_s.wait(&rt.ev_un[l])?;
                    sl.lock().unwrap().stage(&rt.copy_s, &rt.spare, par * rt.nmax, rt.nmax)?;
                    rt.copy_s.record(&rt.ev_st[par])?;
                }
                Ok(())
            };
            if rt.staged.get() != Some(ai) {
                stage_one(li, ai)?;
            }
            // one application ahead (the copy of the next application's host experts overlaps this application's compute)
            let next = if li == 17 && pass < 2 { Some((6, pass + 1)) } else if li == 23 { None } else { Some((li + 1, pass)) };
            match next {
                Some((l2, p2)) if matches!(&*self.w.layers[l2].experts, Experts::Slot { .. }) => {
                    let a2 = app_index(l2, p2);
                    stage_one(l2, a2)?;
                    rt.staged.set(Some(a2));
                }
                _ => rt.staged.set(None),
            }
            Some(ai % 2)
        } else {
            None
        };
        // 1. attention input
        if li > 0 {
            self.blockres(sc.ai.ptr(), nq * D, sc.blocks.ptr(), 4 * D, *nb, hs_in, false, D, lw.attn_q.as_ref().unwrap().ptr(),
                nq, t)?;
        }
        // inputs of q / k / v after the layer norms
        let (xq, xk, xv) = (sc.x[0].ptr(), sc.x[1].ptr(), sc.x[2].ptr());
        self.tick("attn_in_res");
        if msgs {
            let gates = lw.gates.as_ref().unwrap();
            if !self.prompt && t <= 16 {
                // decode: the three gates are independent, run them on three streams (joined before the projections)
                s.record(&self.ev_fork)?;
                for slot in 0..3 {
                    let st = if slot == 0 { &self.s } else { &self.aux[slot - 1] };
                    if slot > 0 {
                        st.wait(&self.ev_fork)?;
                    }
                    let (h, ldh) = if full { (at(&sc.ai, slot * D * 4), 3 * D) } else { (sc.ai.ptr(), D) };
                    let w = if full { lw.ln[slot].ptr() } else { lw.ln[0].ptr() };
                    self.gate_decode(st, slot, NUL, h, ldh, at(&sc.msg, slot * D * 4), 4 * D, &gates[slot], t, sc.x[slot].ptr(), w)?;
                }
                for k in 0..2 {
                    self.aux[k].record(&self.ev_join[k])?;
                    s.wait(&self.ev_join[k])?;
                }
            } else {
            for slot in 0..3 {
                let (h, ldh) = if full { (at(&sc.ai, slot * D * 4), 3 * D) } else { (sc.ai.ptr(), D) };
                self.msg_gate(sc.gh.ptr(), h, ldh, at(&sc.msg, slot * D * 4), 4 * D, &gates[slot], t)?;
                let w = if full { lw.ln[slot].ptr() } else { lw.ln[0].ptr() };
                self.rms(sc.x[slot].ptr(), sc.gh.ptr(), true, D, w, t)?;
            }
            }
            self.tick("gates_qkv");
        } else if li == 0 {
            self.rms(xq, hs_in, false, D, lw.ln[0].ptr(), t)?;
        } else if !full {
            self.rms(xq, sc.ai.ptr(), true, D, lw.ln[0].ptr(), t)?;
        } else {
            for slot in 0..3 {
                self.rms(sc.x[slot].ptr(), at(&sc.ai, slot * D * 4), true, 3 * D, lw.ln[slot].ptr(), t)?;
            }
        }
        self.tick("in_norm");
        let (xk, xv) = if full || msgs { (xk, xv) } else { (xq, xq) };
        // 2. attention
        let fork = !self.prompt && t <= 16;
        let hb = (t as u32 * 24 + 7) / 8;
        let nbk = (t as u32 * 8 + 7) / 8;
        // k and v chains: projection, (dynamic weights), short convolution, head norm into the cache. Independent of the query chain and of
        // each other: in decode they run on the two aux streams.
        let kchain = |st: &Stream| -> Result<()> {
            self.lin(st, sc.klin.ptr(), xk, &lw.wk, t, D, 1024)?;
            if dec {
                // decode: the attention-sink projection rides on this stream (off the query chain), conv + head norm in one launch
                self.gemm(st, sc.slin.ptr(), xq, lw.sink.ptr(), t, 24, D, D, 24)?;
                if full {
                    self.gemm(st, sc.dynk.ptr(), xk, lw.kdyn.as_ref().unwrap().ptr(), t, 32, D, D, 32)?;
                    self.kk.kv_conv_dyn_norm(st, kw, sc.klin.ptr(), lw.kconv.ptr(), sc.dynk.ptr(), app.conv_k.ptr(), wp, t, wr, 0, 22.627417f32, self.spec)?;
                } else {
                    self.kk.kv_conv_norm(st, kw, sc.klin.ptr(), lw.kconv.ptr(), app.conv_k.ptr(), wp, t, wr, 0, 22.627417f32, self.spec)?;
                }
                return Ok(());
            }
            if full {
                self.gemm(st, sc.dynk.ptr(), xk, lw.kdyn.as_ref().unwrap().ptr(), t, 32, D, D, 32)?;
                unsafe {
                    fe_inkk::pdl::launch(st, &self.k.sconv_dyn, 4u32, 256u32, 0, (sc.kcv.ptr(), 1024u32, sc.klin.ptr(), 1024u32, lw.kconv.ptr(),
                        sc.dynk.ptr(), app.conv_k.ptr(), t as u32, 1024u32, self.spec as u32))?;
                }
            } else {
                self.sconv(st, sc.kcv.ptr(), sc.klin.ptr(), lw.kconv.ptr(), app.conv_k.ptr(), t, 1024)?;
            }
            unsafe {
                fe_inkk::pdl::launch(st, &self.k.headnorm, nbk, 256u32, 0, (kw, sc.kcv.ptr(), 0u32, 22.627417f32, wp, t as u32, wr))?;
            }
            Ok(())
        };
        let vchain = |st: &Stream| -> Result<()> {
            self.lin(st, sc.vlin.ptr(), xv, &lw.wv, t, D, 1024)?;
            if dec {
                if full {
                    self.gemm(st, sc.dynv.ptr(), xv, lw.vdyn.as_ref().unwrap().ptr(), t, 32, D, D, 32)?;
                    self.kk.kv_conv_dyn_norm(st, vw, sc.vlin.ptr(), lw.vconv.ptr(), sc.dynv.ptr(), app.conv_v.ptr(), wp, t, wr, 1, 0.0f32, self.spec)?;
                } else {
                    self.kk.kv_conv_norm(st, vw, sc.vlin.ptr(), lw.vconv.ptr(), app.conv_v.ptr(), wp, t, wr, 1, 0.0f32, self.spec)?;
                }
                return Ok(());
            }
            if full {
                self.gemm(st, sc.dynv.ptr(), xv, lw.vdyn.as_ref().unwrap().ptr(), t, 32, D, D, 32)?;
                unsafe {
                    fe_inkk::pdl::launch(st, &self.k.sconv_dyn, 4u32, 256u32, 0, (sc.vcv.ptr(), 1024u32, sc.vlin.ptr(), 1024u32, lw.vconv.ptr(),
                        sc.dynv.ptr(), app.conv_v.ptr(), t as u32, 1024u32, self.spec as u32))?;
                }
            } else {
                self.sconv(st, sc.vcv.ptr(), sc.vlin.ptr(), lw.vconv.ptr(), app.conv_v.ptr(), t, 1024)?;
            }
            unsafe {
                fe_inkk::pdl::launch(st, &self.k.headnorm, nbk, 256u32, 0, (vw, sc.vcv.ptr(), 1u32, 0.0f32, wp, t as u32, wr))?;
            }
            Ok(())
        };
        let qchain = |st: &Stream| -> Result<()> {
            self.lin(st, sc.qgr.ptr(), xq, &lw.wqgr, t, D, 6528)?;
            if dec {
                return self.kk.headclip2(st, sc.qp.ptr(), sc.rsx.ptr(), sc.qgr.ptr(), at(&sc.qgr, 6144 * 2), 6528, lw.q_scale.ptr(), lw.rel_scale.ptr(),
                    full, self.posbuf.ptr(), t);
            }
            self.gemm(st, sc.slin.ptr(), xq, lw.sink.ptr(), t, 24, D, D, 24)?;
            unsafe {
                fe_inkk::pdl::launch(st, &self.k.headclip, hb, 256u32, 0, (sc.qp.ptr(), sc.qgr.ptr(), 6528u32, lw.q_scale.ptr(), 24u32, 128u32,
                    22.627417f32, full as u32, self.posbuf.ptr(), t as u32, 8192.0f32, 0.1f32))?;
                fe_inkk::pdl::launch(st, &self.k.headclip, hb, 256u32, 0, (sc.rsx.ptr(), at(&sc.qgr, 6144 * 2), 6528u32, lw.rel_scale.ptr(), 24u32, 16u32,
                    8.0f32, full as u32, self.posbuf.ptr(), t as u32, 8192.0f32, 0.1f32))?;
            }
            Ok(())
        };
        if fork {
            s.record(&self.ev_fork)?;
            self.aux[0].wait(&self.ev_fork)?;
            self.aux[1].wait(&self.ev_fork)?;
            kchain(&self.aux[0])?;
            vchain(&self.aux[1])?;
            qchain(s)?;
            for k in 0..2 {
                self.aux[k].record(&self.ev_join[k])?;
                s.wait(&self.ev_join[k])?;
            }
        } else {
            qchain(s)?;
            kchain(s)?;
            vchain(s)?;
        }
        self.tick("clip_conv");
        unsafe {
            if q6 {
                let nb = ((t * 8 + 7) / 8) as u32;
                s.launch(&self.k.kvq_pack, nb, 256u32, 0, (app.k.ptr(), self.kstg.ptr(), self.posbuf.ptr(), t as u32))?;
                s.launch(&self.k.kvq_pack, nb, 256u32, 0, (app.v.ptr(), self.vstg.ptr(), self.posbuf.ptr(), t as u32))?;
            }
            // tensor-core attention in three launches over super-tiles of 512 keys (same bits for a row whatever the chunking)
            let nsup = if !full { 10u32 } else if self.capturing { self.cap.div_ceil(128) as u32 } else { ((pos0 + t - 1) / 128 + 1) as u32 };
            let ntiles = (t + 15) / 16;
            let qtb = ntiles.min((self.apt.mapped() / (24 * nsup as usize * 8192)).max(1));
            let window = if full { 0u32 } else { 1024u32 };
            let mut qt0 = 0;
            // 1..=8 rows (decode, verify): grouped-query scan (K read once for all rows and the three heads of a kv head, scores kept for the V pass); sliding layers
            // use the same kernels over their window (one group of 16 super-tiles, ring cache)
            let gsz = if full { 8usize } else { 16 };
            let ngr = (nsup as usize + gsz - 1) / gsz;
            let nch = if full { self.cap / 64 + 1 } else { 24 };
            let gp_bytes = (8 * 24 * (self.cap.div_ceil(128) + 1).div_ceil(8) * 512).next_multiple_of(4096);
            let dg_need = gp_bytes + 2 * nch * 16384;
            // (the sliding layers keep ink_attn_tc: with one block a kv head the grouped scan is slower than its 80 blocks)
            if self.dg && (full || self.dg_slide) && t <= 8 && qt0 == 0 && dg_need <= self.apt.mapped() {
                let gp = self.apt.ptr();
                let scr0 = self.apt.ptr().byte_add(gp_bytes);
                let gst = self.ast.ptr().byte_add(8 << 20);
                // 6..=8 rows of a global layer: two row groups in ONE scan (else two slices of 4 rows, two scans)
                let one_scan = full && t >= 6 && self.dg2;
                let slices: &[(u32, u32)] = if one_scan { &[(0, 0)] } else if t <= 5 { &[(0, t as u32)] } else { &[(0, 4), (4, t as u32 - 4)] };
                for (si, &(row0, nr)) in slices.iter().enumerate() {
                    let scr = scr0.byte_add(si * nch * 16384);
                    if one_scan {
                        fe_inkk::pdl::launch(s, &self.k.dg2_k, (ngr as u32, 8u32), 128u32, if q6 { 58624u32 } else { 56064u32 }, (self.ast.ptr(), scr, sc.qp.ptr(), app.k.ptr(), sc.rsx.ptr(), lw.proj.ptr(),
                            self.posbuf.ptr(), 0.088388346f32, ring, nsup, t as u32, nch as u32, q6 as u32))?;
                    } else {
                        fe_inkk::pdl::launch(s, &self.k.dg_k, (ngr as u32, 8u32), 128u32, if q6 { 48128u32 } else { 45568u32 }, (self.ast.ptr(), scr, sc.qp.ptr(), app.k.ptr(), sc.rsx.ptr(), lw.proj.ptr(),
                            self.posbuf.ptr(), 0.088388346f32, ring, nsup, nr, row0, t as u32, nch as u32, q6 as u32, window))?;
                    }
                }
                self.tick(if full { "G_k_stats" } else { "S_k_stats" });
                fe_inkk::pdl::launch(s, &self.k.attn_gstat, (24u32, 1u32), 128u32, 0, (self.ast.ptr(), gst, sc.slin.ptr(), 24u32, nsup, t as u32, 0u32, window, 0u32))?;
                self.tick(if full { "G_gstat" } else { "S_gstat" });
                for (si, &(row0, nr)) in slices.iter().enumerate() {
                    let scr = scr0.byte_add(si * nch * 16384);
                    if one_scan {
                        fe_inkk::pdl::launch(s, &self.k.dg2_v, (ngr as u32, 8u32), 128u32, (if q6 { 17408u32 + 4 * 8064 } else { 34816u32 + 4 * 2048 }) + 24576,
                            (gp, scr, gst, app.v.ptr(), self.posbuf.ptr(), ring, nsup, ngr as u32, t as u32, nch as u32, q6 as u32))?;
                    } else {
                        fe_inkk::pdl::launch(s, &self.k.dg_v, (ngr as u32, 8u32), 128u32, (if q6 { 17408u32 + 4 * 6528 } else { 34816u32 + 4096 }) + 4 * 3 * nr * 128 * 4,
                            (gp, scr, gst, app.v.ptr(), self.posbuf.ptr(), ring, nsup, ngr as u32, nr, row0, t as u32, nch as u32, q6 as u32, window))?;
                    }
                }
                self.tick(if full { "G_kv_pass" } else { "S_kv_pass" });
                fe_inkk::pdl::launch(s, &self.k.dg_comb, (24u32, t as u32), 128u32, 0, (sc.gat.ptr(), gp, at(&sc.qgr, 3072 * 2), 6528u32, ngr as u32, q6 as u32))?;
                self.tick(if full { "G_comb" } else { "S_comb" });
                qt0 = ntiles;
            }
            while qt0 < ntiles {
                let qc = (ntiles - qt0).min(qtb) as u32;
                let gqa = (t == 1) as u32;
                let gst = self.ast.ptr().byte_add(8 << 20);
                let (n0, ng, n1, nc) = if full { ("G_k_stats", "G_gstat", "G_kv_pass", "G_comb") } else { ("S_k_stats", "S_gstat", "S_kv_pass", "S_comb") };
                for phase in 0..2u32 {
                    if phase == 1 {
                        fe_inkk::pdl::launch(s, &self.k.attn_gstat, (24u32, if gqa != 0 { 1u32 } else { qc }), 128u32, 0, (self.ast.ptr(), gst,
                            sc.slin.ptr(), 24u32, nsup, t as u32, qt0 as u32, window, gqa))?;
                        self.tick(ng);
                    }
                    fe_inkk::pdl::launch(s, &self.k.attn_tc, (nsup, if gqa != 0 { 8u32 } else { 24u32 }, if gqa != 0 { 1u32 } else { qc }), 128u32, ATC_SMEM, (self.ast.ptr(), gst, self.apt.ptr(), sc.qp.ptr(), app.k.ptr(),
                        app.v.ptr(), sc.rsx.ptr(), lw.proj.ptr(), sc.slin.ptr(), 24u32, self.posbuf.ptr(), window, 0.088388346f32, ring,
                        t as u32, phase, qt0 as u32, gqa, q6 as u32))?;
                    self.tick(if phase == 0 { n0 } else { n1 });
                }
                fe_inkk::pdl::launch(s, &self.k.attn_comb2, (24u32, qc), 1024u32, 0, (sc.gat.ptr(), self.apt.ptr(), at(&sc.qgr, 3072 * 2), 6528u32, nsup, t as u32,
                    qt0 as u32, window, q6 as u32, if gqa != 0 { 128u32 } else { 2048u32 }))?;
                self.tick(nc);
                qt0 += qc as usize;
            }
            if false {
                fe_inkk::pdl::launch(s, &self.k.gate_mul, ((t * 3072) as u32 + 255) / 256, 256u32, 0, (sc.gat.ptr(), sc.att.ptr(), at(&sc.qgr, 3072 * 2), 6528u32,
                    (t * 3072) as u32))?;
            }
        }
        self.tick(if full { "attn_global" } else { "attn_slide" });
        self.lin(s, sc.o.ptr(), sc.gat.ptr(), &lw.wo, t, 3072, D)?;
        self.tick("gemm_wo");
        let n = (t * D) as u32;
        if dec {
            // decode: attention conv + residual (+ block-state copy at a boundary) in one launch
            let cdst = if boundary { at(&sc.blocks, *nb * D * 2) } else { NUL };
            self.kk.sconv_resid(s, sc.resid.ptr(), sc.o.ptr(), lw.attn_conv.ptr(), app.conv_a.ptr(), hs_in, !boundary, cdst, 4 * D, t, self.spec)?;
            if boundary {
                *nb += 1;
            }
        } else {
        self.sconv(s, sc.a2.ptr(), sc.o.ptr(), lw.attn_conv.ptr(), app.conv_a.ptr(), t, D)?;
        self.tick("conv_a");
        // 3. residual and block states
        unsafe {
            fe_inkk::pdl::launch(s, &self.k.resid, (n + 255) / 256, 256u32, 0, (sc.resid.ptr(), hs_in, sc.a2.ptr(), (!boundary) as u32, n))?;
        }
        if boundary {
            unsafe {
                fe_inkk::pdl::launch(s, &self.k.copy_rows, t as u32, 256u32, 0, (at(&sc.blocks, *nb * D * 2), (4 * D) as u32, hs_in, D as u32, D as u32))?;
            }
            *nb += 1;
        }
        }
        self.tick("resid");
        // 4. mlp
        self.blockres(sc.mi.ptr(), D, sc.blocks.ptr(), 4 * D, *nb, sc.resid.ptr(), true, D, lw.mlp_q.ptr(), 1, t)?;
        let src = if msgs {
            self.tick("mlp_in_res");
            self.msg_gate(sc.mg.ptr(), sc.mi.ptr(), D, at(&sc.msg, 3 * D * 4), 4 * D, &lw.gates.as_ref().unwrap()[3], t)?;
            self.tick("gate_mlp");
            sc.mg.ptr()
        } else {
            sc.mi.ptr()
        };
        if !dec {
            self.rms(sc.xm.ptr(), src, true, D, lw.post_ln.ptr(), t)?;
        }
        self.tick("mlp_in");
        let fp = match forced {
            Some(f) => {
                ensure!(f.len() == t * 8);
                unsafe { s.memcpy_htod(sc.forced.ptr(), f)? };
                sc.forced.ptr()
            }
            None => NUL,
        };
        unsafe {
            if dec {
                // decode: norm + router logits in one multi-block launch (same bits as rms + router_logits)
                self.kk.router_front(s, sc.xm.ptr(), sc.rlog.ptr(), src, lw.post_ln.ptr(), lw.router.ptr(), t)?;
            } else {
                fe_inkk::pdl::launch(s, &self.k.router_logits, (t as u32, 4u32), 256u32, 0, (sc.rlog.ptr(), sc.xm.ptr(), lw.router.ptr()))?;
            }
            fe_inkk::pdl::launch(s, &self.k.router_pick, t as u32, 32u32, 0, (sc.picks.ptr(), sc.wts.ptr(), sc.rlog.ptr(),
                lw.router_bias.ptr(), fp, 8.0f32))?;
            self.tick("router");
            let p = (t * 8) as u32;
            // one numeric definition for every row count: the native-rule grouped kernel (chunk-invariant, works from 1 row)
            let mut guard = None;
            let pf = match &*lw.experts {
                Experts::Fp4 { pf, .. } => pf,
                Experts::Mx { pf, .. } => pf,
                Experts::Slot { sl, .. } => {
                    guard = Some(sl.lock().unwrap());
                    &guard.as_ref().unwrap().pl
                }
            };
            // prompt chunks: the layer's host experts were staged into spare device slots on the copy stream (see below)
            let staged = guard.is_some() && stage_par.is_some();
            if let (true, Some(par)) = (staged, stage_par) {
                s.wait(&self.slots.as_ref().unwrap().ev_st[par])?;
            }
            let _ = p;
            if dec {
                // decode: slot sum + MLP short conv + residual add in the MoE's last launch
                let tail = fe_inkk::MoeTail { resid: sc.resid.ptr(), out: hs_out, conv_w: lw.mlp_conv.ptr(), state: app.conv_m.ptr(), spec: self.spec };
                self.kk.moe_prefill_tail(s, pf, &self.pf_scr, sc.xm.ptr(), D, sc.picks.ptr(), sc.wts.ptr(), t, &tail)?;
            } else {
                self.kk.moe_prefill(s, pf, &self.pf_scr, sc.xm.ptr(), D, sc.picks.ptr(), sc.wts.ptr(), t, sc.moe.ptr(), D)?;
            }
        }
        if let (Some(par), Some(rt)) = (stage_par, self.slots.as_ref()) {
            // the MoE of this application has been enqueued: point the tables back at the home locations, release the spare set
            if let Experts::Slot { sl, .. } = &*lw.experts {
                sl.lock().unwrap().unstage(s)?;
                s.record(&rt.ev_un[li])?;
                s.record(&rt.ev_dn[par])?;
            }
        }
        self.tick("comb");
        if !dec {
            self.sconv(s, sc.mc.ptr(), sc.moe.ptr(), lw.mlp_conv.ptr(), app.conv_m.ptr(), t, D)?;
            unsafe {
                fe_inkk::pdl::launch(s, &self.k.final_add, (n + 255) / 256, 256u32, 0, (hs_out, sc.resid.ptr(), sc.mc.ptr(), n))?;
            }
        }
        self.tick("conv_m+add");
        if let Some(tp) = tap {
            s.sync()?;
            tp.hs = vec![0u16; t * D];
            tp.picks = vec![0i32; t * 8];
            tp.wts = vec![0f32; t * 8];
            tp.rlog = vec![0f32; t * 128];
            fe_cuda::memcpy_dtoh(&mut tp.hs, hs_out)?;
            fe_cuda::memcpy_dtoh(&mut tp.picks, sc.picks.ptr())?;
            fe_cuda::memcpy_dtoh(&mut tp.wts, sc.wts.ptr())?;
            fe_cuda::memcpy_dtoh(&mut tp.rlog, sc.rlog.ptr())?;
        }
        Ok(())
    }

    /// The device work of one forward (everything after the embedding rows are uploaded): graph-capturable for t = 1.
    fn body(&mut self, t: usize, nl: usize, forced: Option<&[Vec<i32>]>, mut taps: Option<&mut Taps>) -> Result<()> {
        let pos0 = self.len;
        let mut cur = 0usize;
        self.last.set(Some(std::time::Instant::now()));
        self.rms(self.sc.hs[0].ptr(), self.sc.x[0].ptr(), false, D, self.w.embed_norm.ptr(), t)?;
        if let Some(tp) = taps.as_deref_mut() {
            self.s.sync()?;
            tp.embed = vec![0u16; t * D];
            fe_cuda::memcpy_dtoh(&mut tp.embed, self.sc.hs[0].ptr())?;
            tp.app = (0..NAPP).map(|_| None).collect();
        }
        self.tick("embed");
        let mut nb = 0usize;
        let msg_stride = 4 * D * 4;
        let _ = msg_stride;
        macro_rules! run {
            ($li:expr, $pass:expr, $msgs:expr) => {{
                let (hin, hout) = (self.sc.hs[cur].ptr(), self.sc.hs[1 - cur].ptr());
                let ai = app_index($li, $pass);
                let mut tp = Tap { hs: vec![], picks: vec![], wts: vec![], rlog: vec![] };
                let want = taps.is_some();
                if self.draft_taps && (ai == 5 || ai == 23 || ai == 41) {
                    let slot = if ai == 5 { 0 } else if ai == 23 { 1 } else { 2 };
                    unsafe { self.s.launch(&self.k.copy_rows, t as u32, 256u32, 0, (at(&self.dtaps, slot * D * 2), (4 * D) as u32, hin, D as u32, D as u32))? };
                }
                let f = forced.map(|f| f[ai].as_slice());
                self.apply($li, $pass, &mut nb, hin, hout, $msgs, t, pos0, f, if want { Some(&mut tp) } else { None })?;
                if let Some(tt) = taps.as_deref_mut() {
                    tt.app[ai] = Some(tp);
                }
                cur = 1 - cur;
            }};
        }
        for li in 0..LOOP0 {
            run!(li, 0, false);
        }
        let n = t * D * 2;
        self.s.memcpy_dtod(self.sc.pre_hs.ptr(), self.sc.hs[cur].ptr(), n)?;
        let pre_nb = nb;
        for pass in 0..PASSES {
            if pass > 0 {
                self.s.memcpy_dtod(self.sc.hs[cur].ptr(), self.sc.pre_hs.ptr(), n)?;
                nb = pre_nb;
            }
            for li in LOOP0..LOOP1 {
                if pass > 0 {
                    // messages: stacked [h@12 of the previous pass, its final hidden] queried by the loop layer's 4 queries
                    self.blockres(self.sc.msg.ptr(), 4 * D, self.sc.prev12.ptr(), D, 1, self.sc.prevfin.ptr(), false, D,
                        at(&self.w.msg_q, (li - LOOP0) * 4 * D * 4), 4, t)?;
                    self.tick("msg_blockres");
                }
                run!(li, pass, pass > 0);
            }
            // slot 2 of the block states = the state entering layer 12; the final hidden of the pass
            unsafe {
                fe_inkk::pdl::launch(&self.s, &self.k.copy_rows, t as u32, 256u32, 0,
                    (self.sc.prev12.ptr(), D as u32, at(&self.sc.blocks, 2 * D * 2), (4 * D) as u32, D as u32))?;
            }
            self.s.memcpy_dtod(self.sc.prevfin.ptr(), self.sc.hs[cur].ptr(), n)?;
        }
        for li in LOOP1..NL {
            run!(li, 0, false);
        }
        self.tick("loop_misc");
        // final attention residual, norm, unembed
        self.blockres(self.sc.fin.ptr(), D, self.sc.blocks.ptr(), 4 * D, nb, self.sc.hs[cur].ptr(), false, D,
            self.w.out_q.ptr(), 1, t)?;
        self.rms(self.sc.xf.ptr(), self.sc.fin.ptr(), true, D, self.w.final_norm.ptr(), t)?;
        if self.draft_taps {
            unsafe { self.s.launch(&self.k.copy_rows, t as u32, 256u32, 0, (at(&self.dtaps, 3 * D * 2), (4 * D) as u32, self.sc.xf.ptr(), D as u32, D as u32))? };
        }
        let a = at(&self.sc.xf, (t - nl) * D * 2);
        let mut r0 = 0;
        while r0 < nl {
            let g = (nl - r0).min(16);
            let l = &self.w.unembed;
            if let Some(b) = &l.bf {
                self.gemm(&self.s, at(&self.sc.logits, r0 * VOCAB * 2), at(&self.sc.xf, (t - nl + r0) * D * 2), b.ptr(), g, VOCAB, D, D, VOCAB)?;
            } else {
                self.kk.fp8_forward(&self.s, l.f8.as_ref().unwrap(), &self.k.gemm, NUL, at(&self.sc.logits, r0 * VOCAB * 2), VOCAB,
                    at(&self.sc.xf, (t - nl + r0) * D * 2), D, g, None)?;
            }
            r0 += g;
        }
        let _ = a;
        Ok(())
    }

    /// Run `ids` (positions len..len+T) through the model; returns the bf16 logits (softcap not applied) of the last
    /// `nl` rows, row-major [nl][VOCAB]. `forced[app]`: expert picks to use instead of the router's, [T*8] ascending.
    pub fn forward(&mut self, ids: &[u32], nl: usize, forced: Option<&[Vec<i32>]>, mut taps: Option<&mut Taps>) -> Result<Vec<u16>> {
        let t = ids.len();
        ensure!(t >= 1 && t <= self.tmax && nl <= t);
        ensure!(nl <= 16 || self.slots.is_none(), "INK_HOST_EXPERTS: logits are kept for at most 16 rows");
        if self.pending > 0 {
            let p = self.pending;
            self.commit(p)?;
            self.pending = 0;
        }
        let pos0 = self.len;
        ensure!(pos0 + t <= self.cap, "context {} + {} exceeds the cache {}", pos0, t, self.cap);
        self.ctx.bind()?;
        self.slot_hook(t)?;
        self.ensure_kv(self.len + t)?;
        // embedding rows (host gather) and norm
        let mut raw = vec![0u8; t * D * 2];
        for (i, &id) in ids.iter().enumerate() {
            let r = self.ck.read_range("model.llm.embed.weight", id as usize * D * 2, D * 2)?;
            raw[i * D * 2..(i + 1) * D * 2].copy_from_slice(&r);
        }
        self.s.sync()?;
        unsafe {
            self.s.memcpy_htod(self.sc.x[0].ptr(), &raw)?;
            self.s.memcpy_htod(self.posbuf.ptr(), &[pos0 as u32])?;
        }
        self.tick("start");
        let graphable = self.use_graph && t == 1 && nl == 1 && !self.prompt && taps.is_none() && forced.is_none() && !self.prof_on;
        if graphable {
            let key = (self.draft_taps, self.spec);
            if !self.graphs.contains_key(&key) {
                self.s.sync()?;
                self.s.begin_capture()?;
                self.capturing = true;
                let r = self.body(t, nl, forced, None);
                self.capturing = false;
                r?;
                let g = self.s.end_capture()?;
                self.graphs.insert(key, g);
            }
            self.graphs.get(&key).unwrap().launch(&self.s)?;
        } else {
            self.body(t, nl, forced, taps.as_deref_mut())?;
        }
        self.tick("final+unembed");
        self.s.sync()?;
        if let Some(tp) = taps.as_deref_mut() {
            tp.fin = vec![0u16; t * D];
            fe_cuda::memcpy_dtoh(&mut tp.fin, self.sc.xf.ptr())?;
        }
        let mut out = vec![0u16; if self.argmax_only { 0 } else { nl * VOCAB }];
        if self.argmax_only {
            unsafe { self.s.launch(&self.k.argmax_rows, nl as u32, 1024u32, 0, (self.am_out.ptr(), self.sc.logits.ptr(), VOCAB as u32))? };
            let mut ids = vec![0u32; nl];
            self.s.sync()?;
            fe_cuda::memcpy_dtoh(&mut ids, self.am_out.ptr())?;
            self.last_argmax = ids;
        } else {
            fe_cuda::memcpy_dtoh(&mut out, self.sc.logits.ptr())?;
        }
        self.tick("logits_dtoh");
        if self.prof_on {
            self.prof_fold();
        }
        self.len += t;
        if self.spec {
            self.pending = t;
        }
        Ok(out)
    }
}
