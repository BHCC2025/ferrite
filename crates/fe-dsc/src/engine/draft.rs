//! DSpark drafter on the GPU (unsloth `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, arch `dflash`), its own module with its own
//! context (any card; default the expert-only card) so it never touches the engine's state. Recipe = the f32 CPU reference
//! drafter (always the `swap` Markov orientation: head = markov_w2, embed = markov_w1):
//!   absorb(taps)  : taps bf16 [row][3 x 5120] -> fc (Q8_0) -> enc norm -> per block wkv + kv norm + rope + FP8 -> ring[pos % 256]
//!   draft(tok, t) : rows [tok, noise x 4] at positions t .. t+4 through 3 blocks (window attention over the 128 newest absorbed
//!                   positions + the 5 block keys, F32 router, shared expert + 3 of 128 MXFP4 experts), DSpark norm, the TARGET
//!                   head (Q5_K) and the Markov loop (+ the confidence head).
//! Arithmetic is the engine's GPU one (q8_1 activations, bf16 states), so agreement with the f32 CPU oracle is near-tie, and
//! the GPU result itself is deterministic (row-invariant kernels, fixed reduction orders).
use super::{g1, norm_bf16, raw, Hc, Wt, MIX_SLICE};
use crate::config::DscConfig;
use crate::refmodel::rope_table;
use crate::{K_DSPARK_CUBIN, K_DS_CUBIN};
use anyhow::{anyhow, ensure, Result};
use fe_cuda::{memcpy_dtoh, memcpy_htod, Context, DevPtr, DeviceBuffer, Function, Module, Stream};
use fe_ggq::gpu::{Ggq, Q8Act};
use fe_ggq::refq;
use fe_gguf::{GgmlType, Gguf};
use std::sync::Arc;
use std::time::Instant;

const D: usize = 5120;
const HD: usize = 512;
const NH: usize = 64;
const BLOCK: usize = 5;
const WINDOW: usize = 128;
const RING: usize = 256;
const NOISE: u32 = 128799;
const NB: usize = 3;
const NE: usize = 128;
const TOPK: usize = 3;
const NFF: usize = 2304;
const QDIM: usize = NH * HD;
const OGROUPS: usize = 8;
const OLORA: usize = 1024;
/// rows an absorb call carries
const AROWS: usize = 8;
const MARKOV_PARTS: u32 = 64;
const OUT_STRIDE: usize = 8;

struct Blk {
    hc_attn: Hc,
    hc_ffn: Hc,
    attn_norm: DeviceBuffer<u16>,
    ffn_norm: DeviceBuffer<u16>,
    q_a_norm: DeviceBuffer<u16>,
    kv_norm: DeviceBuffer<u16>,
    q_a: Wt,
    q_b: Wt,
    kv: Wt,
    wo_a: Wt,
    wo_b: Wt,
    sinks: DeviceBuffer<u8>,
    gate_w: DeviceBuffer<u8>,
    gate_b: DeviceBuffer<u8>,
    sh_gate: Wt,
    sh_up: Wt,
    sh_down: Wt,
    _arena: DeviceBuffer<u8>,
    gtab: DeviceBuffer<u64>,
    utab: DeviceBuffer<u64>,
    dtab: DeviceBuffer<u64>,
    rb_gu: usize,
    rb_d: usize,
    clamp_exp: f32,
    clamp_sh: f32,
}

struct Kd {
    _m: Module,
    _md: Module,
    rmsnorm: Function,
    hc_part_pn: Function,
    hc_fin: Function,
    hc_pre_norm: Function,
    hc_post: Function,
    rope: Function,
    kv_norm_window: Function,
    router_pick: Function,
    swiglu: Function,
    b2f: Function,
    copy_rows: Function,
    attn: Function,
    collapse: Function,
    gate32: Function,
    mk_bias: Function,
    mk_score: Function,
    mk_pick: Function,
    mk_begin: Function,
    conf: Function,
}

/// What a draft call returns: the drafted ids (anchor excluded), the confidence-head logits (sigmoid = survival of the draft given
/// the earlier ones), and the wall time of the call.
#[derive(Clone, Debug, Default)]
pub struct DraftOut {
    pub ids: Vec<u32>,
    pub conf: Vec<f32>,
    pub ms: f32,
}

/// The DSpark drafter. `absorb` the taps of the accepted rows, `draft` from the last accepted token.
pub struct Drafter<'a> {
    ctx: Arc<Context>,
    back: Option<Arc<Context>>,
    s: Stream,
    ggq: Ggq,
    k: Kd,
    tg: &'a Gguf,
    eps: f32,
    hc_eps: f32,
    hc_iters: u32,
    max_ctx: usize,
    tab: DeviceBuffer<f32>,
    fc: Wt,
    enc_norm: DeviceBuffer<u16>,
    blk: Vec<Blk>,
    onorm: DeviceBuffer<u16>,
    head: Wt,
    w1: DeviceBuffer<u8>,
    w2: DeviceBuffer<u8>,
    conf_w: DeviceBuffer<u8>,
    nv: usize,
    noise_row: Vec<u16>,
    rings: Vec<DeviceBuffer<u16>>,
    /// positions 0 .. n_abs hold their ring rows
    pub n_abs: usize,
    /// absorbed counts of the lanes (the active lane's lives in `n_abs`); every lane is a slot of the ring buffers
    lane_nabs: Vec<usize>,
    cur_lane: usize,
    /// lanes the rings and the scratch were sized for (DSC_LANES at load)
    nslots: usize,
    // scratch
    h: [DeviceBuffer<u16>; 2],
    cur: usize,
    x: DeviceBuffer<u16>,
    xf: DeviceBuffer<f32>,
    t32: DeviceBuffer<f32>,
    qr_raw: DeviceBuffer<u16>,
    qr: DeviceBuffer<u16>,
    q: DeviceBuffer<u16>,
    kv_raw: DeviceBuffer<u16>,
    kv: DeviceBuffer<u16>,
    o: DeviceBuffer<u16>,
    ug: DeviceBuffer<u16>,
    ub: DeviceBuffer<u16>,
    part: DeviceBuffer<f32>,
    mix_a: DeviceBuffer<f32>,
    mix_f: DeviceBuffer<f32>,
    mixp: DeviceBuffer<f32>,
    pre0: DeviceBuffer<f32>,
    rlogits: DeviceBuffer<f32>,
    rscores: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    wts: DeviceBuffer<f32>,
    sh_g: DeviceBuffer<f32>,
    sh_u: DeviceBuffer<f32>,
    sh_gb: DeviceBuffer<u16>,
    sh_ub: DeviceBuffer<u16>,
    sh_h: DeviceBuffer<u16>,
    y6: DeviceBuffer<f32>,
    act: Q8Act,
    hq: Q8Act,
    xn: DeviceBuffer<u16>,
    xc: DeviceBuffer<u16>,
    logits: DeviceBuffer<f32>,
    taps_dev: DeviceBuffer<u16>,
    meta2: DeviceBuffer<u32>,
    meta4: DeviceBuffer<u32>,
    scratch_ring: DeviceBuffer<u16>,
    // markov
    stage: DeviceBuffer<u16>,
    pairs: DeviceBuffer<u32>,
    out: DeviceBuffer<u32>,
    tok: DeviceBuffer<u32>,
    slot: DeviceBuffer<i32>,
    slot_of: DeviceBuffer<i32>,
    conf_out: DeviceBuffer<f32>,
}

impl<'a> Drafter<'a> {
    /// Load the drafter from `dg` (the dspark GGUF) and the target's head / embedding from `tg`, onto CUDA ordinal `card`
    /// (a new context there; the caller rebinds its own afterwards, or `set_back` makes every call do it).
    pub fn new(dg: &Gguf, tg: &'a Gguf, c: &DscConfig, max_ctx: usize, card: i32) -> Result<Drafter<'a>> {
        let t0 = Instant::now();
        let ctx = Context::new(card)?;
        let s = Stream::new()?;
        let ggq = Ggq::load()?;
        let m = match std::env::var("FE_DS_CUBIN") {
            Ok(p) => Module::load(&std::fs::read(&p)?)?,
            Err(_) => Module::load(K_DS_CUBIN)?,
        };
        let md = Module::load(K_DSPARK_CUBIN)?;
        let k = Kd {
            rmsnorm: m.function("ds_rmsnorm")?,
            hc_part_pn: m.function("ds_hc_mix_part_pn")?,
            hc_fin: m.function("ds_hc_mix_fin")?,
            hc_pre_norm: m.function("ds_hc_pre_norm")?,
            hc_post: m.function("ds_hc_post")?,
            rope: m.function("ds_rope")?,
            kv_norm_window: m.function("ds_kv_norm_window")?,
            router_pick: m.function("ds_router_pick")?,
            swiglu: m.function("ds_swiglu")?,
            b2f: m.function("ds_bf16_to_f32")?,
            copy_rows: m.function("ds_copy_rows")?,
            attn: md.function("dspark_attn")?,
            collapse: md.function("dspark_collapse")?,
            gate32: md.function("dspark_gate32")?,
            mk_bias: md.function("markov_bias")?,
            mk_score: md.function("markov_score")?,
            mk_pick: md.function("markov_pick")?,
            mk_begin: md.function("markov_begin")?,
            conf: md.function("conf_head")?,
            _m: m,
            _md: md,
        };
        let clamp = dg.get("dflash.swiglu_clamp_exp").and_then(|v| v.as_f32_vec()).unwrap_or(vec![10.0; NB]);
        let clamp_sh = dg.get("dflash.swiglu_clamp_shexp").and_then(|v| v.as_f32_vec()).unwrap_or(vec![10.0; NB]);
        let tab = DeviceBuffer::from_slice(&rope_table(c, 10000.0, 0, max_ctx + 16))?;
        let fc = Wt::load(dg, "fc.weight")?;
        ensure!(fc.k == 3 * D && fc.n == D, "fc.weight {} x {}", fc.n, fc.k);
        let mut blk: Vec<Blk> = vec![];
        for j in 0..NB {
            let p = |s: &str| format!("blk.{j}.{s}.weight");
            let hc = |a: &str| -> Result<Hc> {
                Ok(Hc { fnw: raw(dg, &p(&format!("hc_{a}_fn")))?, scale: raw(dg, &p(&format!("hc_{a}_scale")))?, base: raw(dg, &p(&format!("hc_{a}_base")))? })
            };
            // routed experts: MXFP4 -> 18-byte aligned blocks, all 128 resident
            let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
            let ts: Vec<&fe_gguf::TensorInfo> = names.iter().map(|n| dg.tensor(&p(n)).ok_or_else(|| anyhow!("{n} missing"))).collect::<Result<_>>()?;
            for t in &ts {
                ensure!(t.ty == GgmlType::MXFP4, "expert tensor {}: {} not MXFP4", j, t.ty.name());
            }
            let al = |b: usize| b.div_ceil(256) * 256;
            let rb: Vec<usize> = ts.iter().map(|t| t.ne0() / 32 * 18).collect();
            let sz: Vec<usize> = ts.iter().zip(&rb).map(|(t, r)| al(t.dims[1] as usize * r)).collect();
            let per: usize = sz.iter().sum();
            let arena = DeviceBuffer::<u8>::zeroed(per * NE)?;
            let mut tabs = [vec![0u64; NE], vec![0u64; NE], vec![0u64; NE]];
            for e in 0..NE {
                let mut o = e * per;
                for (i, t) in ts.iter().enumerate() {
                    let rep = refq::repack_mxfp4(dg.slice(t, e));
                    ensure!(rep.len() == t.dims[1] as usize * rb[i], "repacked expert size");
                    memcpy_htod(arena.ptr().byte_add(o), &rep)?;
                    tabs[i][e] = arena.ptr().byte_add(o).0;
                    o += sz[i];
                }
            }
            let bias_name = format!("blk.{j}.exp_probs_b.bias");
            let gate_b = if dg.tensor(&bias_name).is_some() { raw(dg, &bias_name)? } else { raw(dg, &p("exp_probs_b"))? };
            let b = Blk {
                hc_attn: hc("attn")?,
                hc_ffn: hc("ffn")?,
                attn_norm: norm_bf16(dg, &p("attn_norm"))?,
                ffn_norm: norm_bf16(dg, &p("ffn_norm"))?,
                q_a_norm: norm_bf16(dg, &p("attn_q_a_norm"))?,
                kv_norm: norm_bf16(dg, &p("attn_kv_a_norm"))?,
                q_a: Wt::load(dg, &p("attn_q_a"))?,
                q_b: Wt::load(dg, &p("attn_q_b"))?,
                kv: Wt::load(dg, &p("attn_kv"))?,
                wo_a: Wt::load(dg, &p("attn_output_a"))?,
                wo_b: Wt::load(dg, &p("attn_output_b"))?,
                sinks: raw(dg, &p("attn_sinks"))?,
                gate_w: raw(dg, &p("ffn_gate_inp"))?,
                gate_b,
                sh_gate: Wt::load(dg, &p("ffn_gate_shexp"))?,
                sh_up: Wt::load(dg, &p("ffn_up_shexp"))?,
                sh_down: Wt::load(dg, &p("ffn_down_shexp"))?,
                gtab: DeviceBuffer::from_slice(&tabs[0])?,
                utab: DeviceBuffer::from_slice(&tabs[1])?,
                dtab: DeviceBuffer::from_slice(&tabs[2])?,
                _arena: arena,
                rb_gu: rb[0],
                rb_d: rb[2],
                clamp_exp: clamp[j],
                clamp_sh: clamp_sh[j],
            };
            ensure!(b.q_a.k == D && b.q_b.k == b.q_a.n && b.q_b.n == QDIM, "block {j} dense dims");
            ensure!(b.wo_a.n == OGROUPS * OLORA && b.wo_a.k == QDIM / OGROUPS && b.wo_b.k == OGROUPS * OLORA, "block {j} wo dims");
            blk.push(b);
            eprintln!("drafter: block {j} loaded ({:.1}s)", t0.elapsed().as_secs_f32());
        }
        let head = Wt::load(tg, "output.weight")?;
        let nv = head.n;
        ensure!(head.k == D, "output head k {}", head.k);
        let w1 = raw(dg, "markov_w1.weight")?;
        let w2 = raw(dg, "markov_w2.weight")?;
        ensure!(dg.tensor("markov_w1.weight").unwrap().ty == GgmlType::BF16 && dg.tensor("markov_w2.weight").unwrap().n_elements() as usize == nv * 256, "markov tensors");
        let conf_w = raw(dg, "conf_proj.weight")?;
        ensure!(dg.tensor("conf_proj.weight").unwrap().ty == GgmlType::F32 && dg.tensor("conf_proj.weight").unwrap().n_elements() as usize == D + 256, "conf_proj");
        // every lane is a ring slot; the scratch holds a draft of all lanes at once (5 rows a lane)
        let nslots: usize = std::env::var("DSC_LANES").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
        let ar = AROWS.max(nslots * BLOCK);
        let mut d = Drafter {
            back: None,
            tg,
            eps: c.eps,
            hc_eps: c.hc_eps,
            hc_iters: c.hc_iters as u32,
            max_ctx,
            tab,
            fc,
            enc_norm: norm_bf16(dg, "enc.output_norm.weight")?,
            onorm: norm_bf16(dg, "output_norm.weight")?,
            head,
            w1,
            w2,
            conf_w,
            nv,
            noise_row: vec![],
            rings: (0..NB).map(|_| DeviceBuffer::<u16>::zeroed(nslots * RING * HD)).collect::<std::result::Result<_, _>>()?,
            n_abs: 0,
            lane_nabs: vec![0; nslots],
            cur_lane: 0,
            nslots,
            h: [DeviceBuffer::zeroed(ar * 4 * D)?, DeviceBuffer::zeroed(ar * 4 * D)?],
            cur: 0,
            x: DeviceBuffer::zeroed(ar * D)?,
            xf: DeviceBuffer::zeroed(ar * D)?,
            t32: DeviceBuffer::zeroed(ar * QDIM)?,
            qr_raw: DeviceBuffer::zeroed(ar * 1280)?,
            qr: DeviceBuffer::zeroed(ar * 1280)?,
            q: DeviceBuffer::zeroed(ar * QDIM)?,
            kv_raw: DeviceBuffer::zeroed(ar * HD)?,
            kv: DeviceBuffer::zeroed(ar * HD)?,
            o: DeviceBuffer::zeroed(ar * QDIM)?,
            ug: DeviceBuffer::zeroed(ar * OLORA)?,
            ub: DeviceBuffer::zeroed(ar * OGROUPS * OLORA)?,
            part: DeviceBuffer::zeroed(ar * D)?,
            mix_a: DeviceBuffer::zeroed(ar * 24)?,
            mix_f: DeviceBuffer::zeroed(ar * 24)?,
            mixp: DeviceBuffer::zeroed(ar * (4 * D / MIX_SLICE * 25))?,
            pre0: DeviceBuffer::from_slice(&(0..ar).flat_map(|_| [1.0f32, 0.0, 0.0, 0.0]).collect::<Vec<f32>>())?,
            rlogits: DeviceBuffer::zeroed(ar * NE)?,
            rscores: DeviceBuffer::zeroed(ar * NE)?,
            ids: DeviceBuffer::zeroed(ar * TOPK)?,
            wts: DeviceBuffer::zeroed(ar * TOPK)?,
            sh_g: DeviceBuffer::zeroed(ar * NFF)?,
            sh_u: DeviceBuffer::zeroed(ar * NFF)?,
            sh_gb: DeviceBuffer::zeroed(ar * NFF)?,
            sh_ub: DeviceBuffer::zeroed(ar * NFF)?,
            sh_h: DeviceBuffer::zeroed(ar * NFF)?,
            y6: DeviceBuffer::zeroed(ar * TOPK * D)?,
            act: Q8Act::alloc(ar, 3 * D)?,
            hq: Q8Act::alloc(ar * TOPK, NFF)?,
            xn: DeviceBuffer::zeroed(ar * D)?,
            xc: DeviceBuffer::zeroed(ar * D)?,
            logits: DeviceBuffer::zeroed(ar * nv)?,
            taps_dev: DeviceBuffer::zeroed(ar * 3 * D)?,
            meta2: DeviceBuffer::zeroed(ar * 2)?,
            meta4: DeviceBuffer::zeroed(ar * 4)?,
            scratch_ring: DeviceBuffer::zeroed(ar * RING * HD)?,
            stage: DeviceBuffer::zeroed(nslots * nv)?,
            pairs: DeviceBuffer::zeroed(MARKOV_PARTS as usize * nslots * 2)?,
            out: DeviceBuffer::zeroed(nslots * OUT_STRIDE)?,
            tok: DeviceBuffer::zeroed(nslots)?,
            slot: DeviceBuffer::from_slice(&vec![-1i32; nslots])?,
            slot_of: DeviceBuffer::from_slice(&vec![-1i32; nv])?,
            conf_out: DeviceBuffer::zeroed(nslots * BLOCK)?,
            ctx,
            s,
            ggq,
            k,
            blk,
        };
        d.noise_row = d.embed_row(NOISE)?;
        d.s.sync()?;
        eprintln!("drafter: loaded in {:.1}s", t0.elapsed().as_secs_f32());
        Ok(d)
    }

    /// Make every public call re-bind `ctx` (the caller's context) before it returns.
    pub fn set_back(&mut self, ctx: Arc<Context>) {
        self.back = Some(ctx);
    }

    pub fn reset(&mut self) {
        self.n_abs = 0;
    }

    /// `n` lanes in total (each has its own ring slot and absorbed count; the drafter's weights and scratch are shared).
    pub fn alloc_lanes(&mut self, n: usize) -> Result<()> {
        ensure!(n <= self.nslots, "drafter: {n} lanes but it was loaded for {} (DSC_LANES)", self.nslots);
        Ok(())
    }

    /// Make lane `i` the one `absorb` / `draft` work on.
    pub fn activate(&mut self, i: usize) {
        if i == self.cur_lane {
            return;
        }
        self.lane_nabs[self.cur_lane] = self.n_abs;
        self.n_abs = self.lane_nabs[i];
        self.cur_lane = i;
    }

    /// The target's embedding row of `tok` as bf16 x 4 streams [4][D].
    fn embed_row(&self, tok: u32) -> Result<Vec<u16>> {
        let et = self.tg.tensor("token_embd.weight").ok_or_else(|| anyhow!("token_embd"))?;
        let rb = et.row_bytes();
        let t = tok as usize;
        ensure!(t < self.nv, "token {t} out of vocab");
        let mut row = vec![0f32; D];
        refq::dequant(et.ty, &self.tg.data(et)[t * rb..(t + 1) * rb], &mut row);
        let b: Vec<u16> = row.iter().map(|&v| refq::f32_to_bf16(v)).collect();
        let mut o = Vec::with_capacity(4 * D);
        for _ in 0..4 {
            o.extend_from_slice(&b);
        }
        Ok(o)
    }

    // ---------------------------------------------------------------- helpers (nr rows)
    fn hp(&self) -> DevPtr {
        self.h[self.cur].ptr()
    }

    fn gq(&self, w: &Wt, y: DevPtr) -> Result<()> {
        ensure!(w.k == self.act.k, "gemv k {} != activation k {}", w.k, self.act.k);
        self.ggq.gemv(&self.s, w.ty, w.ptr(), w.rb, w.n, &self.act, y, false, None)
    }

    fn q8(&mut self, x: DevPtr, bf16: bool, k: usize, nr: usize) -> Result<()> {
        self.ggq.quantize_q8_1(&self.s, x, k, bf16, nr, k, &mut self.act)
    }

    fn f2b(&self, dst: DevPtr, src: DevPtr, n: usize, nr: usize) -> Result<()> {
        self.ggq.f32_to_bf16(&self.s, dst, n, src, n, nr, n)
    }

    fn rmsnorm(&self, y: DevPtr, x: DevPtr, w: DevPtr, dim: usize, nr: usize) -> Result<()> {
        let bs = if dim >= 1024 { 256 } else { 128 };
        unsafe { self.s.launch(&self.k.rmsnorm, nr as u32, bs, 0, (y, x, 0u32, w, dim as u32, self.eps, dim as u32, dim as u32))? };
        Ok(())
    }

    /// hc mixes of `h` into `out` and the pre-norm of the same streams with the earlier mixes `pre`.
    fn hc_mix_pn(&self, hc: &Hc, out: DevPtr, pre: DevPtr, pre_ld: u32, w: DevPtr, nr: usize) -> Result<()> {
        let dim4 = 4 * D;
        let nks = dim4 / MIX_SLICE;
        unsafe {
            self.s.launch(&self.k.hc_part_pn, (nks as u32 + 1, nr as u32), 768, 0, (self.mixp.ptr(), self.hp(), hc.fnw.ptr(), dim4 as u32,
                MIX_SLICE as u32, self.x.ptr(), pre, pre_ld, w, D as u32, self.eps))?;
            self.s.launch(&self.k.hc_fin, nr as u32, 32, 0, (out, self.mixp.ptr(), nks as u32, hc.scale.ptr(), hc.base.ptr(), dim4 as u32,
                self.eps, self.hc_eps, self.hc_iters))?;
        }
        Ok(())
    }

    /// h = hc post of the sublayer output `part` (f32 [nr][D]) with the mixes `mix`; swaps the stream buffers.
    fn post(&mut self, mix: DevPtr, nr: usize) -> Result<()> {
        unsafe { self.s.launch(&self.k.hc_post, (g1(D), nr as u32), 256, 0, (self.h[1 - self.cur].ptr(), self.part.ptr(), self.hp(), mix, D as u32))? };
        self.cur = 1 - self.cur;
        Ok(())
    }

    fn bind_in(&self) -> Result<()> {
        self.ctx.bind()?;
        Ok(())
    }

    fn bind_out(&self) -> Result<()> {
        if let Some(b) = &self.back {
            b.bind()?;
        }
        Ok(())
    }

    // ---------------------------------------------------------------- absorb

    /// Absorb the taps (bf16 [n][3 x 5120], DEVICE memory of this drafter's card) of the rows at positions start .. start + n.
    pub fn absorb_dev(&mut self, taps: DevPtr, n: usize, start: usize) -> Result<()> {
        ensure!(n >= 1 && n <= AROWS, "absorb: {n} rows (1..={AROWS})");
        ensure!(start + n <= self.max_ctx, "absorb: position {} beyond max_ctx {}", start + n, self.max_ctx);
        let slot = self.cur_lane as u32;
        let meta: Vec<u32> = (0..n).flat_map(|i| [slot, (start + i) as u32]).collect();
        memcpy_htod(self.meta2.ptr(), &meta)?;
        self.q8(taps, true, 3 * D, n)?;
        self.gq(&self.fc, self.t32.ptr())?;
        self.f2b(self.xn.ptr(), self.t32.ptr(), D, n)?;
        self.rmsnorm(self.x.ptr(), self.xn.ptr(), self.enc_norm.ptr(), D, n)?;
        for j in 0..NB {
            self.q8(self.x.ptr(), true, D, n)?;
            self.gq(&self.blk[j].kv, self.t32.ptr())?;
            self.f2b(self.kv_raw.ptr(), self.t32.ptr(), HD, n)?;
            unsafe {
                self.s.launch(&self.k.kv_norm_window, n as u32, 32, 0, (self.kv.ptr(), self.kv_raw.ptr(), self.blk[j].kv_norm.ptr(), self.eps,
                    self.rings[j].ptr(), (RING * HD) as u32, RING as u32, self.meta2.ptr(), self.tab.ptr()))?;
            }
        }
        self.n_abs = start + n;
        Ok(())
    }

    /// [`Drafter::absorb_dev`] for taps in host memory (bf16 [n][3 x 5120]); this is the entry point across cards.
    pub fn absorb(&mut self, taps: &[u16], n: usize, start: usize) -> Result<()> {
        ensure!(taps.len() >= n * 3 * D, "absorb: {} taps for {n} rows", taps.len());
        self.bind_in()?;
        memcpy_htod(self.taps_dev.ptr(), &taps[..n * 3 * D])?;
        let r = self.absorb_dev(self.taps_dev.ptr(), n, start);
        self.bind_out()?;
        r
    }

    // ---------------------------------------------------------------- one block over the 5 rows

    fn block(&mut self, j: usize, first: bool, nr: usize) -> Result<()> {
        let (hd, qd) = (HD, QDIM);
        // ---- attention
        let (pre, pre_ld) = if first { (self.pre0.ptr(), 4u32) } else { (self.mix_f.ptr(), 24u32) };
        self.hc_mix_pn(&self.blk[j].hc_attn, self.mix_a.ptr(), pre, pre_ld, self.blk[j].attn_norm.ptr(), nr)?;
        self.q8(self.x.ptr(), true, D, nr)?;
        self.gq(&self.blk[j].q_a, self.t32.ptr())?;
        self.f2b(self.qr_raw.ptr(), self.t32.ptr(), 1280, nr)?;
        self.gq(&self.blk[j].kv, self.t32.ptr())?;
        self.f2b(self.kv_raw.ptr(), self.t32.ptr(), hd, nr)?;
        self.rmsnorm(self.qr.ptr(), self.qr_raw.ptr(), self.blk[j].q_a_norm.ptr(), 1280, nr)?;
        self.q8(self.qr.ptr(), true, 1280, nr)?;
        self.gq(&self.blk[j].q_b, self.t32.ptr())?;
        self.f2b(self.q.ptr(), self.t32.ptr(), qd, nr)?;
        unsafe {
            self.s.launch(&self.k.rope, (nr as u32, NH as u32), 32, 0, (self.q.ptr(), qd as u32, hd as u32, (hd - 64) as u32, self.meta2.ptr(),
                self.tab.ptr(), 0u32))?;
            // the block's own keys (slot = row: scratch ring), normed + roped + FP8 grid, in place in kv
            self.s.launch(&self.k.kv_norm_window, nr as u32, 32, 0, (self.kv.ptr(), self.kv_raw.ptr(), self.blk[j].kv_norm.ptr(), self.eps,
                self.scratch_ring.ptr(), (RING * HD) as u32, RING as u32, self.meta2.ptr(), self.tab.ptr()))?;
            self.s.launch(&self.k.attn, (NH as u32, nr as u32), 128, 0, (self.o.ptr(), (nr * 8 * hd) as u32, self.q.ptr(), qd as u32,
                self.rings[j].ptr(), (RING * HD) as u32, RING as u32, WINDOW as u32, self.kv.ptr(), BLOCK as u32, self.meta4.ptr(),
                self.blk[j].sinks.ptr(), (hd as f32).powf(-0.5), self.tab.ptr()))?;
        }
        // wo_a: 8 groups of 4096 -> 1024, then wo_b
        let gin = qd / OGROUPS;
        for gi in 0..OGROUPS {
            self.q8(self.o.ptr().add::<u16>(gi * nr * gin), true, gin, nr)?;
            let w = &self.blk[j].wo_a;
            self.ggq.gemv(&self.s, w.ty, w.ptr().byte_add(gi * OLORA * w.rb), w.rb, OLORA, &self.act, self.t32.ptr(), false, None)?;
            self.f2b(self.ug.ptr(), self.t32.ptr(), OLORA, nr)?;
            unsafe {
                self.s.launch(&self.k.copy_rows, (nr as u32, g1(OLORA)), 256, 0, (self.ub.ptr().add::<u16>(gi * OLORA),
                    (OGROUPS * OLORA) as u32, self.ug.ptr(), OLORA as u32, OLORA as u32))?;
            }
        }
        self.q8(self.ub.ptr(), true, OGROUPS * OLORA, nr)?;
        self.gq(&self.blk[j].wo_b, self.part.ptr())?;
        self.post(self.mix_a.ptr(), nr)?;
        // ---- MoE
        self.hc_mix_pn(&self.blk[j].hc_ffn, self.mix_f.ptr(), self.mix_a.ptr(), 24, self.blk[j].ffn_norm.ptr(), nr)?;
        unsafe {
            self.s.launch(&self.k.b2f, g1(D * nr), 256, 0, (self.xf.ptr(), self.x.ptr(), (D * nr) as u32))?;
            self.s.launch(&self.k.gate32, ((NE as u32).div_ceil(8), nr as u32), 256, 0, (self.rlogits.ptr(), self.xf.ptr(), D as u32,
                self.blk[j].gate_w.ptr(), D as u32, NE as u32))?;
            self.s.launch(&self.k.router_pick, nr as u32, 128, 0, (self.ids.ptr(), self.wts.ptr(), self.rscores.ptr(), self.rlogits.ptr(),
                self.blk[j].gate_b.ptr(), 1u32, NE as u32, TOPK as u32, 1.5f32))?;
        }
        self.q8(self.x.ptr(), true, D, nr)?;
        self.gq(&self.blk[j].sh_gate, self.sh_g.ptr())?;
        self.gq(&self.blk[j].sh_up, self.sh_u.ptr())?;
        self.ggq.set_swiglu_limit(self.blk[j].clamp_exp);
        self.ggq.moe_gu_q8(&self.s, GgmlType::RepMxfp4, self.blk[j].gtab.ptr(), self.blk[j].utab.ptr(), self.blk[j].rb_gu, self.ids.ptr(), nr, TOPK,
            &self.act, NFF, &mut self.hq, None, None)?;
        self.ggq.moe_down(&self.s, GgmlType::RepMxfp4, self.blk[j].dtab.ptr(), self.blk[j].rb_d, self.ids.ptr(), nr, TOPK, &self.hq, D, self.y6.ptr(), None)?;
        self.f2b(self.sh_gb.ptr(), self.sh_g.ptr(), NFF, nr)?;
        self.f2b(self.sh_ub.ptr(), self.sh_u.ptr(), NFF, nr)?;
        unsafe {
            self.s.launch(&self.k.swiglu, (g1(NFF), nr as u32), 256, 0, (self.sh_h.ptr(), self.sh_gb.ptr(), self.sh_ub.ptr(), DevPtr(0),
                NFF as u32, self.blk[j].clamp_sh))?;
        }
        self.q8(self.sh_h.ptr(), true, NFF, nr)?;
        self.gq(&self.blk[j].sh_down, self.part.ptr())?;
        self.ggq.moe_combine(&self.s, self.part.ptr(), self.y6.ptr(), self.wts.ptr(), nr, TOPK, D, true)?;
        self.post(self.mix_f.ptr(), nr)?;
        Ok(())
    }

    // ---------------------------------------------------------------- draft

    /// Draft `k` (1..=5) tokens after `last` (the token at position `pos`, which must equal the number of absorbed positions) on the
    /// active lane.
    pub fn draft(&mut self, last: u32, pos: usize, k: usize) -> Result<DraftOut> {
        let l = self.cur_lane;
        Ok(self.draft_multi(&[(l, last, pos)], k)?.remove(0))
    }

    /// One drafter pass for several lanes: `reqs` = (lane, its last token, its position), distinct lanes. The rows of all lanes ride in
    /// the same kernels (5 rows a lane; every row attends to its own lane's ring slot and its own lane's block keys), so the weights are
    /// read once. Each lane's result is what `draft` gives for it alone (row-invariant kernels).
    pub fn draft_multi(&mut self, reqs: &[(usize, u32, usize)], k: usize) -> Result<Vec<DraftOut>> {
        ensure!((1..=BLOCK).contains(&k), "draft: k {k}");
        ensure!(!reqs.is_empty() && reqs.len() <= self.nslots, "draft: {} lanes (1..={})", reqs.len(), self.nslots);
        self.lane_nabs[self.cur_lane] = self.n_abs;
        for (i, &(l, _, pos)) in reqs.iter().enumerate() {
            ensure!(l < self.nslots && !reqs[..i].iter().any(|r| r.0 == l), "draft: lane {l}");
            ensure!(pos == self.lane_nabs[l], "draft at position {pos} but {} positions are absorbed (lane {l})", self.lane_nabs[l]);
            ensure!(pos + BLOCK <= self.max_ctx, "draft: position {} beyond max_ctx {}", pos + BLOCK, self.max_ctx);
        }
        self.bind_in()?;
        let r = self.draft_inner(reqs, k);
        self.bind_out()?;
        r
    }

    fn draft_inner(&mut self, reqs: &[(usize, u32, usize)], k: usize) -> Result<Vec<DraftOut>> {
        let t0 = Instant::now();
        let ns = reqs.len();
        let nr = ns * BLOCK;
        // rows [last, noise x 4] a lane, 4 streams each; meta2 = (scratch slot, position), meta4 = (ring slot, position, newest absorbed + 1, first block key)
        let mut h0: Vec<u16> = Vec::with_capacity(nr * 4 * D);
        let mut meta2: Vec<u32> = Vec::with_capacity(nr * 2);
        let mut meta4: Vec<u32> = Vec::with_capacity(nr * 4);
        let mut anchors = vec![0u32; ns * OUT_STRIDE];
        for (s, &(lane, last, pos)) in reqs.iter().enumerate() {
            h0.extend_from_slice(&self.embed_row(last)?);
            for _ in 1..BLOCK {
                h0.extend_from_slice(&self.noise_row);
            }
            for i in 0..BLOCK {
                meta2.extend_from_slice(&[(s * BLOCK + i) as u32, (pos + i) as u32]);
                meta4.extend_from_slice(&[lane as u32, (pos + i) as u32, pos as u32, (s * BLOCK) as u32]);
            }
            anchors[s * OUT_STRIDE] = last;
        }
        self.cur = 0;
        memcpy_htod(self.h[0].ptr(), &h0)?;
        memcpy_htod(self.meta2.ptr(), &meta2)?;
        memcpy_htod(self.meta4.ptr(), &meta4)?;
        for j in 0..NB {
            self.block(j, j == 0, nr)?;
        }
        // DSpark norm (collapse with the last mixes) -> the target head
        unsafe {
            self.s.launch(&self.k.hc_pre_norm, nr as u32, 256, 0, (self.xn.ptr(), self.hp(), self.mix_f.ptr(), 24u32, self.onorm.ptr(), D as u32, self.eps))?;
            self.s.launch(&self.k.collapse, nr as u32, 256, 0, (self.xc.ptr(), self.hp(), self.mix_f.ptr(), 24u32, D as u32))?;
        }
        self.q8(self.xn.ptr(), true, D, nr)?;
        self.gq(&self.head, self.logits.ptr())?;
        // Markov loop (a row a lane): step i scores the lane's logits row i + the bias row of its previous token
        memcpy_htod(self.out.ptr(), &anchors)?;
        let nsu = ns as u32;
        unsafe {
            self.s.launch(&self.k.mk_begin, 1u32, 32, 0, (self.out.ptr(), self.tok.ptr(), self.slot.ptr(), self.slot_of.ptr(), nsu, OUT_STRIDE as u32))?;
            for i in 0..k {
                self.s.launch(&self.k.mk_bias, (1024u32, nsu), 256, 0, (self.stage.ptr(), self.tok.ptr(), self.slot.ptr(), self.w1.ptr(), self.w2.ptr(), self.nv as u32))?;
                self.s.launch(&self.k.mk_score, (nsu, MARKOV_PARTS), 256, 0, (self.pairs.ptr(), self.logits.ptr().add::<f32>(i * self.nv), (BLOCK * self.nv) as u32,
                    self.stage.ptr(), self.stage.ptr(), self.slot.ptr(), self.nv as u32, 0u32))?;
                self.s.launch(&self.k.mk_pick, 1u32, 32, 0, (self.out.ptr(), self.tok.ptr(), self.slot.ptr(), self.pairs.ptr(), self.slot_of.ptr(),
                    MARKOV_PARTS, nsu, OUT_STRIDE as u32, i as u32))?;
            }
            self.s.launch(&self.k.conf, (ns * k) as u32, 256, 0, (self.conf_out.ptr(), self.xc.ptr(), self.w1.ptr(), self.out.ptr(), self.conf_w.ptr(),
                D as u32, k as u32, BLOCK as u32, OUT_STRIDE as u32))?;
        }
        let mut out = vec![0u32; ns * OUT_STRIDE];
        let mut conf = vec![0f32; ns * k];
        self.s.sync()?;
        memcpy_dtoh(&mut out, self.out.ptr())?;
        memcpy_dtoh(&mut conf, self.conf_out.ptr())?;
        let ms = t0.elapsed().as_secs_f32() * 1e3;
        Ok((0..ns).map(|s| DraftOut { ids: out[s * OUT_STRIDE + 1..=s * OUT_STRIDE + k].to_vec(), conf: conf[s * k..(s + 1) * k].to_vec(), ms }).collect())
    }
}
