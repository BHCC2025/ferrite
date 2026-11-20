//! Big-prompt path (DSC_PF_ROWS): a prompt chunk of up to DSC_PF_ROWS rows goes through the layers together. The dense part is
//! the existing kernels with many rows (the wide GEMVs are row-invariant); the routed experts run as sorted int8 GEMMs
//! (fe-ggq mmq) on the cards: experts resident on a card are used in place, every other expert is streamed from host memory
//! into a ring of device slots by one stager thread per card that runs AHEAD of the compute across layers (the expert set of a
//! layer is static: every expert not resident on a card; a chunk of hundreds of rows uses nearly all of them), so the uploads
//! overlap attention and the GEMMs of the layers before. The two cards each take their resident experts and half of the
//! streamed ones; the weighted pair sums are order-independent i64 fixed point (mmq_moe_down_acc), so the result does not
//! depend on the split.

use super::*;
use fe_cuda::{Event, HostBuffer};
use fe_ggq::gpu::SortWs;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

pub static OV_NS: AtomicUsize = AtomicUsize::new(0);
pub static CARD_NS: [AtomicUsize; 3] = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)];
pub static JW_NS: AtomicUsize = AtomicUsize::new(0);
/// ns the compute side waited for the stager (all cards) since the last stats call
pub static WAIT_NS: AtomicUsize = AtomicUsize::new(0);

/// What the stager and the compute side of a card both touch (events and device slots).
struct Shared {
    arena: Vec<DeviceBuffer<u8>>,
    gtabs: Vec<DeviceBuffer<u64>>,
    up_done: Vec<Event>,
    use_done: Vec<Event>,
}
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// Per-row state between the parts of a big chunk's layer (all rows of the chunk).
pub struct BigStore {
    h: DeviceBuffer<u16>,
    mf: DeviceBuffer<f32>,
    x: DeviceBuffer<u16>,
    ids: DeviceBuffer<u32>,
    wts: DeviceBuffer<f32>,
    /// per-row selection and candidate mask: set by an index-source layer, read by later layers
    sel: DeviceBuffer<i32>,
    cmask: DeviceBuffer<u8>,
}
impl BigStore {
    pub fn alloc(c: &DscConfig, n: usize, max_ctx: usize) -> Result<BigStore> {
        let (d, nx) = (c.n_embd, c.n_expert_used);
        Ok(BigStore { h: DeviceBuffer::zeroed(n * 4 * d)?, mf: DeviceBuffer::zeroed(n * 24)?, x: DeviceBuffer::zeroed(n * d)?,
            ids: DeviceBuffer::zeroed(n * nx)?, wts: DeviceBuffer::zeroed(n * nx)?,
            sel: DeviceBuffer::zeroed(n * c.idx_top_k)?, cmask: DeviceBuffer::zeroed(n * max_ctx.div_ceil(c.cand_block.max(1)))? })
    }
}

/// Per-card workspace of the big-prompt MoE.
pub struct PfCard {
    cs: Stream,
    cps: Stream,
    ea: Q8Act,
    hq: Q8Act,
    acc: DeviceBuffer<i64>,
    accb: DeviceBuffer<i64>,
    status: DeviceBuffer<u32>,
    ws: SortWs,
    x: DeviceBuffer<u16>,
    ids: DeviceBuffer<u32>,
    wts: DeviceBuffer<f32>,
    sh: Arc<Shared>,
    stage: [HostBuffer<u8>; 2],
    stage_done: [Event; 2],
    /// experts per streamed group
    group: usize,
    /// streamed bytes since the last `pf_take_stats`
    bytes: Arc<AtomicUsize>,
}

/// Whether the main card leaves its resident expert `e` of layer `l` to the expert cards in the big path (DSC_PF_MAIN_KEEP = share it keeps).
pub(super) fn pf_drop(l: usize, e: usize, keep: f64) -> bool {
    keep < 1.0 && ((((e as u64).wrapping_mul(2654435761).wrapping_add(l as u64 * 40503) >> 7) % 100) as f64) >= keep * 100.0
}

fn al(b: usize) -> usize {
    b.div_ceil(256) * 256
}

impl PfCard {
    /// Allocate on the CURRENT context: workspace for chunks of `n` rows.
    pub fn alloc(c: &DscConfig, g: &Gguf, n: usize) -> Result<PfCard> {
        let (d, ff, nx, ne) = (c.n_embd, c.n_ff_exp, c.n_expert_used, c.n_expert);
        let group: usize = std::env::var("DSC_PF_GROUP").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let nslots: usize = std::env::var("DSC_PF_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        let mut eb = 0;
        for l in 0..c.n_layer {
            let e: usize = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].iter()
                .map(|nm| al(g.tensor(&format!("blk.{l}.{nm}.weight")).unwrap().slice_bytes())).sum();
            eb = eb.max(e);
        }
        let ev = || Event::new_no_timing();
        let mut arena = vec![];
        let mut gtabs = vec![];
        let mut up_done = vec![];
        let mut use_done = vec![];
        for _ in 0..nslots {
            arena.push(DeviceBuffer::zeroed(group * eb)?);
            gtabs.push(DeviceBuffer::zeroed(3 * ne)?);
            up_done.push(ev()?);
            use_done.push(ev()?);
        }
        Ok(PfCard {
            cs: Stream::new()?,
            cps: Stream::new()?,
            ea: Q8Act::alloc(n, d)?,
            hq: Q8Act::alloc(n * nx, ff)?,
            acc: DeviceBuffer::zeroed(n * d)?,
            accb: DeviceBuffer::zeroed(n * d)?,
            status: DeviceBuffer::zeroed(1)?,
            ws: SortWs::alloc(n * nx, ne)?,
            x: DeviceBuffer::zeroed(n * d)?,
            ids: DeviceBuffer::zeroed(n * nx)?,
            wts: DeviceBuffer::zeroed(n * nx)?,
            sh: Arc::new(Shared { arena, gtabs, up_done, use_done }),
            stage: [HostBuffer::alloc(group * eb)?, HostBuffer::alloc(group * eb)?],
            stage_done: [ev()?, ev()?],
            group,
            bytes: Arc::new(AtomicUsize::new(0)),
        })
    }
}

/// The stager thread of one card for one chunk: uploads (layer, group) items in order into free slots.
struct StageJob {
    ctx: *const Context,
    g: *const Gguf,
    host: *const HostRegion,
    c: *const DscConfig,
    cps: *const Stream,
    sh: Arc<Shared>,
    stage: *mut HostBuffer<u8>,
    stage_done: *const Event,
    items: Vec<(usize, Vec<usize>)>,
    free_rx: Receiver<usize>,
    ready_tx: Sender<usize>,
    bytes: Arc<AtomicUsize>,
}
unsafe impl Send for StageJob {}

fn stager(j: StageJob) -> Result<()> {
    unsafe {
        let (ctx, g, c, cps) = (&*j.ctx, &*j.g, &*j.c, &*j.cps);
        let host = if j.host.is_null() { None } else { Some(&*j.host) };
        ctx.bind()?;
        let ne = c.n_expert;
        let mut nst = 0usize;
        for (l, grp) in &j.items {
            let l = *l;
            let Ok(slot) = j.free_rx.recv() else { return Ok(()) };
            j.sh.use_done[slot].sync()?;
            let ts = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].map(|nm| g.tensor(&format!("blk.{l}.{nm}.weight")).unwrap());
            let sb: Vec<usize> = ts.iter().map(|t| al(t.slice_bytes())).collect();
            let eb = sb.iter().sum::<usize>();
            let arena = j.sh.arena[slot].ptr();
            // registered host copy: one DMA per expert straight from it (gate | up | down are contiguous there, laid out as the arena)
            let direct = host.is_some_and(|h| h.registered && grp.iter().all(|&e| h.off[l * ne + e][0] != usize::MAX));
            if direct {
                let h = host.unwrap();
                for (jx, &e) in grp.iter().enumerate() {
                    let o = h.off[l * ne + e][0];
                    cps.memcpy_htod(arena.byte_add(jx * eb), std::slice::from_raw_parts(h.ptr.add(o), eb))?;
                }
            } else {
                let st = nst % 2;
                nst += 1;
                (*j.stage_done.add(st)).sync()?;
                let stage = &mut *j.stage.add(st);
                let srcs: Vec<[(usize, usize); 3]> = grp.iter().map(|&e| {
                    let mut r = [(0usize, 0usize); 3];
                    for p in 0..3 {
                        r[p] = match host.map(|h| h.off[l * ne + e][p]) {
                            Some(o) if o != usize::MAX => (host.unwrap().ptr as usize + o, ts[p].slice_bytes()),
                            _ => {
                                let s = g.slice(ts[p], e);
                                (s.as_ptr() as usize, s.len())
                            }
                        };
                    }
                    r
                }).collect();
                let stg = stage.as_mut_slice();
                stg[..grp.len() * eb].par_chunks_mut(eb).zip(srcs.par_iter()).for_each(|(dst, sr)| {
                    let mut o = 0;
                    for p in 0..3 {
                        let (a, n) = sr[p];
                        dst[o..o + n].copy_from_slice(std::slice::from_raw_parts(a as *const u8, n));
                        o += sb[p];
                    }
                });
                cps.memcpy_htod(arena, &stage.as_slice()[..grp.len() * eb])?;
                cps.record(&*j.stage_done.add(st))?;
            }
            let mut tabs = vec![0u64; 3 * ne];
            for (jx, &e) in grp.iter().enumerate() {
                let base = arena.byte_add(jx * eb).0;
                tabs[e] = base;
                tabs[ne + e] = base + sb[0] as u64;
                tabs[2 * ne + e] = base + (sb[0] + sb[1]) as u64;
            }
            cps.memcpy_htod(j.sh.gtabs[slot].ptr(), &tabs)?;
            cps.record(&j.sh.up_done[slot])?;
            j.bytes.fetch_add(grp.len() * eb, Ordering::Relaxed);
            if j.ready_tx.send(slot).is_err() {
                return Ok(());
            }
        }
    }
    Ok(())
}

/// One card's side of a running big chunk.
struct CardRun {
    ready_rx: Receiver<usize>,
    free_tx: Sender<usize>,
    /// per layer: the experts this card streams (static for the chunk)
    streamed: Vec<Vec<usize>>,
    held: Vec<usize>,
    handle: Option<std::thread::JoinHandle<Result<()>>>,
}

pub struct PfRun {
    a: CardRun,
    b: Option<CardRun>,
    c: Option<CardRun>,
}

/// One card's share of a layer's routed experts.
struct CardJob<'e> {
    ctx: &'e Context,
    ggq: &'e Ggq,
    pf: *mut PfCard,
    run: *mut CardRun,
    g: &'e Gguf,
    c: &'e DscConfig,
    l: usize,
    nr: usize,
    x: DevPtr,
    ids: DevPtr,
    wts: DevPtr,
    res: Option<&'e LayerRes>,
    /// element offset of this job's rows in the card's i64 accumulator
    acc_off: usize,
    /// 0: free each staged slot after use; 1: keep it (first half of a pipelined layer); 2: run the kept slots again and free them (second half)
    hold: u8,
}
unsafe impl Send for CardJob<'_> {}

/// gate/up GEMMs + SwiGLU + down with the weighted fixed-point sum for the pairs whose expert is in the tables.
#[allow(clippy::too_many_arguments)]
unsafe fn exps(j: &CardJob, pf: &mut PfCard, ty_gu: GgmlType, ty_d: GgmlType, rb_gu: usize, rb_d: usize, gt: DevPtr, ut: DevPtr,
    dt: DevPtr) -> Result<()> {
    let c = j.c;
    let (d, ff, nx) = (c.n_embd, c.n_ff_exp, c.n_expert_used);
    let np = j.nr * nx;
    j.ggq.mmq_moe_gu(&pf.cs, ty_gu, &pf.ea, gt, ut, rb_gu, ff, &pf.ws, np, &mut pf.hq)?;
    j.ggq.mmq_moe_down_acc(&pf.cs, ty_d, pf.acc.ptr().add::<i64>(j.acc_off), d, &pf.hq, dt, rb_d, d, &pf.ws, np, nx, j.wts, pf.status.ptr())?;
    Ok(())
}

unsafe fn run_card(j: CardJob) -> Result<()> {
    j.ctx.bind()?;
    let pf = &mut *j.pf;
    let run = &mut *j.run;
    let c = j.c;
    let (d, nx, ne) = (c.n_embd, c.n_expert_used, c.n_expert);
    let l = j.l;
    let ts = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].map(|nm| j.g.tensor(&format!("blk.{l}.{nm}.weight")).unwrap());
    let (ty_gu, ty_d) = (ts[0].ty, ts[2].ty);
    let (rb_gu, rb_d) = (ts[0].row_bytes(), ts[2].row_bytes());
    j.ggq.set_swiglu_limit(c.clamp_exp[l]);
    j.ggq.quantize_q8_1(&pf.cs, j.x, d, true, j.nr, d, &mut pf.ea)?;
    j.ggq.moe_sort_small(&pf.cs, j.ids, j.nr, nx, &pf.ws)?;
    if let Some(r) = j.res {
        exps(&j, pf, ty_gu, ty_d, rb_gu, rb_d, r.gtab.ptr(), r.utab.ptr(), r.dtab.ptr())?;
    }
    let ng = run.streamed[l].len().div_ceil(pf.group);
    if j.hold == 2 {
        let held = std::mem::take(&mut run.held);
        for slot in held {
            let gt = pf.sh.gtabs[slot].ptr();
            exps(&j, pf, ty_gu, ty_d, rb_gu, rb_d, gt, gt.add::<u64>(ne), gt.add::<u64>(2 * ne))?;
            pf.cs.record(&pf.sh.use_done[slot])?;
            let _ = run.free_tx.send(slot);
        }
    } else {
        for _ in 0..ng {
            let tw = Instant::now();
            let slot = run.ready_rx.recv().map_err(|_| anyhow!("stager ended early"))?;
            WAIT_NS.fetch_add(tw.elapsed().as_nanos() as usize, Ordering::Relaxed);
            pf.cs.wait(&pf.sh.up_done[slot])?;
            let gt = pf.sh.gtabs[slot].ptr();
            exps(&j, pf, ty_gu, ty_d, rb_gu, rb_d, gt, gt.add::<u64>(ne), gt.add::<u64>(2 * ne))?;
            if j.hold == 1 {
                run.held.push(slot);
            } else {
                pf.cs.record(&pf.sh.use_done[slot])?;
                let _ = run.free_tx.send(slot);
            }
        }
    }
    pf.cs.sync()?;
    Ok(())
}

impl<'a> Engine<'a> {
    /// Shared expert into `part` (the routed experts are added by the caller).
    fn seg_shared(&mut self, l: usize) -> Result<()> {
        let c = self.c;
        let d = c.n_embd;
        let ly = &self.layers[l];
        let x = self.x.ptr();
        q8!(self, x, true, d);
        self.gq(&ly.sh_gate, self.sh_g.ptr())?;
        self.gq(&ly.sh_up, self.sh_u.ptr())?;
        self.f2b(self.sh_gb.ptr(), self.sh_g.ptr(), c.n_ff_exp)?;
        self.f2b(self.sh_ub.ptr(), self.sh_u.ptr(), c.n_ff_exp)?;
        unsafe {
            self.s.launch(&self.k.swiglu, (g1(c.n_ff_exp), self.nr as u32), 256, 0, (self.sh_h.ptr(), self.sh_gb.ptr(), self.sh_ub.ptr(),
                DevPtr(0), c.n_ff_exp as u32, c.clamp_shexp[l]))?;
        }
        q8!(self, self.sh_h.ptr(), true, c.n_ff_exp);
        self.gq(&ly.sh_down, self.part.ptr())?;
        Ok(())
    }

    /// Start the stager threads of a big chunk: which expert each card streams (static: every expert not resident on a card; the
    /// second card takes about half), then the uploads run ahead of the layers.
    pub(super) fn pf_begin(&mut self, n_rows: usize) -> Result<()> {
        let c = self.c;
        if self.res_pf.is_empty() {
            // the big path never computes a host-backed (cold) expert in place over PCIe: its table entries are dropped and the expert is staged
            // by DMA like any other streamed one (the main card also drops a share of its residents, DSC_PF_MAIN_KEEP)
            let keep: f64 = std::env::var("DSC_PF_MAIN_KEEP").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
            let ne = c.n_expert;
            let mask = |r: &LayerRes, l: usize, main: bool, zc_off: &Vec<[usize; 3]>| -> Result<LayerRes> {
                let (mut g, mut u, mut dt) = (r.gtab.to_vec()?, r.utab.to_vec()?, r.dtab.to_vec()?);
                let mut present = r.present.clone();
                for e in 0..present.len() {
                    let cold = zc_off[l * ne + e][0] != usize::MAX;
                    if present[e] && (cold || (main && pf_drop(l, e, keep))) {
                        present[e] = false;
                        g[e] = 0;
                        u[e] = 0;
                        dt[e] = 0;
                    }
                }
                Ok(LayerRes { gtab: DeviceBuffer::from_slice(&g)?, utab: DeviceBuffer::from_slice(&u)?, dtab: DeviceBuffer::from_slice(&dt)?, present, ty_gu: r.ty_gu, ty_d: r.ty_d, rb_gu: r.rb_gu, rb_d: r.rb_d })
            };
            let mut v = vec![];
            for l in 0..c.n_layer {
                v.push(match self.res[l].as_ref() { Some(r) => Some(mask(r, l, true, &self.zc_off)?), None => None });
            }
            let mut xv = vec![];
            for k in 0..self.xcs.len() {
                self.xcs[k].ctx.bind()?;
                let mut vk = vec![];
                for l in 0..c.n_layer {
                    vk.push(match self.xcs[k].res[l].as_ref() { Some(r) => Some(mask(r, l, false, &self.zc_off)?), None => None });
                }
                xv.push(vk);
            }
            self._ctx.bind()?;
            self.res_pf = v;
            self.xres_pf = xv;
        }
        if self.pf_meta_h.is_none() {
            self.pf_meta_h = Some(HostBuffer::alloc(16 * self.rmax * 2)?);
            self.pf_eng_h = Some(HostBuffer::alloc(16 * self.rmax * c.engram_width().max(1))?);
        }
        let ne = c.n_expert;
        let two = self.xcs.first().is_some_and(|x| x.pf.is_some()) && std::env::var("DSC_PF_ONE").is_err();
        let three = two && self.xcs.get(1).is_some_and(|x| x.pf.is_some()) && std::env::var("DSC_PF_THREE").map_or(true, |v| v != "0");
        let ncard = 1 + two as usize + three as usize;
        let mut lists: Vec<Vec<Vec<usize>>> = vec![vec![]; ncard];
        for l in 0..c.n_layer {
            let mut sl: Vec<Vec<usize>> = vec![vec![]; ncard];
            let main_noexp = self.pipe_split(self.pf_n) && std::env::var("DSC_PF_MAIN_EXP").map_or(false, |v| v == "0");
            let mut rs: Vec<Option<&LayerRes>> = vec![if main_noexp { None } else { self.res_pf[l].as_ref().or(self.res[l].as_ref()) }];
            if two {
                rs.push(self.xres_pf[0][l].as_ref());
            }
            if three {
                rs.push(self.xres_pf[1][l].as_ref());
            }
            let mut w = vec![0f64; ncard];
            if self.pipe_split(self.pf_n) {
                // the main card also does the attention: it takes fewer streamed experts
                w[0] += std::env::var("DSC_PF_MAIN_PEN").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            }
            for e in 0..ne {
                for k in 0..ncard {
                    if rs[k].is_some_and(|r| r.present[e]) {
                        w[k] += 0.3;
                    }
                }
            }
            for e in 0..ne {
                if (0..ncard).any(|k| rs[k].is_some_and(|r| r.present[e])) {
                    continue;
                }
                let k = (0..ncard).min_by(|&x, &y| w[x].partial_cmp(&w[y]).unwrap()).unwrap();
                sl[k].push(e);
                w[k] += 1.0;
            }
            for k in 0..ncard {
                lists[k].push(std::mem::take(&mut sl[k]));
            }
        }
        let host: *const HostRegion = self.host.as_ref().map_or(std::ptr::null(), |h| h as *const HostRegion);
        let g: *const Gguf = self.g;
        let cfg: *const DscConfig = self.c;
        let slots_n = self.pf.as_ref().map_or(0, |p| p.sh.arena.len());
        let ng_max = lists.iter().map(|ll| ll.iter().map(|v| v.len().div_ceil(self.pf.as_ref().unwrap().group)).max().unwrap_or(0)).max().unwrap_or(0);
        self.pf_hold = self.pipe_split(n_rows) && ng_max + 1 <= slots_n && std::env::var("DSC_PF_HOLD").map_or(true, |v| v != "0");
        let reps: usize = if self.pipe_split(n_rows) && !self.pf_hold { 2 } else { 1 };
        let mk = |ctx: *const Context, pf: &mut PfCard, streamed: Vec<Vec<usize>>| -> CardRun {
            let (free_tx, free_rx) = channel();
            let (ready_tx, ready_rx) = channel();
            for s in 0..pf.sh.arena.len() {
                free_tx.send(s).unwrap();
            }
            let grp = pf.group;
            let items: Vec<(usize, Vec<usize>)> = streamed.iter().enumerate()
                .flat_map(|(l, v)| (0..reps).flat_map(move |_| v.chunks(grp).map(move |ch| (l, ch.to_vec())))).collect();
            let job = StageJob { ctx, g, host, c: cfg, cps: &pf.cps as *const Stream, sh: pf.sh.clone(),
                stage: pf.stage.as_mut_ptr(), stage_done: pf.stage_done.as_ptr(), items, free_rx, ready_tx, bytes: pf.bytes.clone() };
            CardRun { ready_rx, free_tx, streamed, held: vec![], handle: Some(std::thread::spawn(move || stager(job))) }
        };
        let mut it = lists.into_iter();
        let ctx_a: *const Context = &*self._ctx;
        let pf_a = self.pf.as_mut().ok_or_else(|| anyhow!("big path without workspace"))?;
        let a = mk(ctx_a, pf_a, it.next().unwrap());
        let b = if two {
            let xc = self.xcs.first_mut().unwrap();
            let ctx_b: *const Context = &*xc.ctx;
            Some(mk(ctx_b, xc.pf.as_mut().unwrap(), it.next().unwrap()))
        } else {
            None
        };
        let cr = if three {
            let xc = &mut self.xcs[1];
            let ctx_c: *const Context = &*xc.ctx;
            Some(mk(ctx_c, xc.pf.as_mut().unwrap(), it.next().unwrap()))
        } else {
            None
        };
        self.pf_run = Some(PfRun { a, b, c: cr });
        Ok(())
    }

    /// End a big chunk: stop and join the stagers.
    pub(super) fn pf_end(&mut self) {
        if let Some(mut r) = self.pf_run.take() {
            for cr in [Some(&mut r.a), r.b.as_mut(), r.c.as_mut()].into_iter().flatten() {
                // unblock a stager waiting for a free slot, then join it
                let (tx, _rx) = channel();
                cr.free_tx = tx;
                if let Some(h) = cr.handle.take() {
                    let _ = h.join();
                }
            }
        }
        let _ = self._ctx.bind();
    }

    /// Embedding rows of a block into `h` / `meta` (no Engram).
    fn embed_block(&mut self, tokens: &[u32], pos0: usize) -> Result<()> {
        let d = self.c.n_embd;
        let nr = tokens.len();
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
        Ok(())
    }

    /// A big chunk of `tokens` (positions pos0 ..; `ids_seen` holds them): layer by layer, the attention / router part in blocks of
    /// `rmax` rows (each block against the same caches, in order), the routed experts over ALL rows at once (every expert is
    /// uploaded once per chunk), then the shared expert + hc post per block. The per-row state between the parts lives in `big`.
    pub(super) fn step_big(&mut self, tokens: &[u32], pos0: usize) -> Result<()> {
        let c = self.c;
        let (d, n, m) = (c.n_embd, tokens.len(), self.rmax);
        let t_all = Instant::now();
        let st = self.big.as_ref().ok_or_else(|| anyhow!("big path without store"))?;
        let (bh, bx) = (st.h.ptr(), st.x.ptr());
        // embeddings of every block into the store
        for r0 in (0..n).step_by(m) {
            let nr = m.min(n - r0);
            self.nr = nr;
            self.embed_block(&tokens[r0..r0 + nr], pos0 + r0)?;
            self.s.memcpy_dtod(bh.add::<u16>(r0 * 4 * d), self.h.ptr(), nr * 4 * d * 2)?;
        }
        let _ = bx;
        self.no_graph = true;
        self.pf_begin(n)?;
        let r = (|| -> Result<()> {
            if self.pipe_split(n) {
                self.run_pipelined(pos0, n)?;
            } else {
                for l in 0..c.n_layer {
                    self.layer_tiled(l, pos0, n)?;
                }
            }
            Ok(())
        })();
        self.pf_end();
        self.no_graph = false;
        self.tap_off = 0;
        r?;
        self.s.sync()?;
        if std::env::var("DSC_PF_VERBOSE").is_ok() {
            let stt = self.pf_status()?;
            let (ba, bb) = self.pf_take_stats();
            eprintln!("pf chunk {n} rows: {:.2}s, streamed A {:.2} GB, B {:.2} GB, fx overflow {stt}; attention+router {:.2}s, moe {:.2}s, post {:.2}s, compute waited {:.2}s (both cards), engram host {:.2}s, overlap work {:.2}s, join wait {:.2}s; expert time main {:.2}s card1 {:.2}s card2 {:.2}s",
                t_all.elapsed().as_secs_f64(), ba as f64 / 1e9, bb as f64 / 1e9, self.split.ph[0] / 1e3, self.split.ph[2] / 1e3,
                self.split.ph[3] / 1e3, WAIT_NS.swap(0, Ordering::Relaxed) as f64 / 1e9, self.split.engram_ms / 1e3, OV_NS.swap(0, Ordering::Relaxed) as f64 / 1e9, JW_NS.swap(0, Ordering::Relaxed) as f64 / 1e9, CARD_NS[0].swap(0, Ordering::Relaxed) as f64 / 1e9, CARD_NS[1].swap(0, Ordering::Relaxed) as f64 / 1e9, CARD_NS[2].swap(0, Ordering::Relaxed) as f64 / 1e9);
        }
        Ok(())
    }

    /// Pinned host copies of the experts the big path streams (the cold ones and the main card's dropped residents), so the stagers DMA straight
    /// from them at link speed instead of copying out of the GGUF mapping with the CPU.
    pub(super) fn build_pf_pin(&mut self) -> Result<()> {
        if !self.zc || self.zc_buf.is_none() {
            return Ok(());
        }
        let c = self.c;
        let (nl, ne) = (c.n_layer, c.n_expert);
        let keep: f64 = std::env::var("DSC_PF_MAIN_KEEP").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
        let t0 = Instant::now();
        if keep >= 1.0 {
            // the cold experts only: the zero-copy region has the arena layout already
            let zb = self.zc_buf.as_mut().unwrap();
            self.host = Some(HostRegion { ptr: zb.as_mut_ptr(), len: zb.len(), off: self.zc_off.clone(), rty: vec![], rsz: vec![], registered: true });
            return Ok(());
        }
        let ts: Vec<Vec<&fe_gguf::TensorInfo>> = (0..nl).map(|l| ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].iter().map(|n| self.g.tensor(&format!("blk.{l}.{n}.weight")).unwrap()).collect()).collect();
        let mut off = vec![[usize::MAX; 3]; nl * ne];
        let mut len = 0usize;
        for l in 0..nl {
            for e in 0..ne {
                let i = l * ne + e;
                let cold = self.zc_off[i][0] != usize::MAX;
                let dropped = self.res[l].as_ref().is_some_and(|r| r.present[e]) && pf_drop(l, e, keep);
                if cold || dropped {
                    for p in 0..3 {
                        off[i][p] = len;
                        len += al(ts[l][p].slice_bytes());
                    }
                }
            }
        }
        // the bulk copy goes off the strict node-0 binding: prefer node 1 (DMA rate measured equal, and node 0 keeps its room for the latency-critical regions)
        let (mut old_mode, mut old_mask) = (0 as libc::c_int, [0u64; 4]);
        let have_old = unsafe { libc::syscall(libc::SYS_get_mempolicy, &mut old_mode as *mut libc::c_int, old_mask.as_mut_ptr(), 256usize, 0usize, 0usize) } == 0;
        if std::env::var("DSC_PF_PIN_NODE").map_or(true, |v| v != "0") {
            let nm: u64 = 0b10;
            unsafe { libc::syscall(libc::SYS_set_mempolicy, 1 /* MPOL_PREFERRED */, &nm as *const u64, 64u64) };
        }
        let hb = HostBuffer::<u8>::alloc(len.max(256));
        if have_old {
            unsafe { libc::syscall(libc::SYS_set_mempolicy, old_mode as libc::c_long, old_mask.as_ptr(), 256u64) };
        }
        let mut hb = hb?;
        for l in 0..nl {
            for e in 0..ne {
                let i = l * ne + e;
                if off[i][0] == usize::MAX {
                    continue;
                }
                for p in 0..3 {
                    let sl = self.g.slice(ts[l][p], e);
                    unsafe { std::ptr::copy_nonoverlapping(sl.as_ptr(), hb.as_mut_ptr().add(off[i][p]), sl.len()) };
                }
            }
        }
        eprintln!("big-path pinned experts: {:.2} GiB in {:.1}s", len as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f32());
        self.host = Some(HostRegion { ptr: hb.as_mut_ptr(), len, off, rty: vec![], rsz: vec![], registered: true });
        self.pf_pin = Some(hb);
        Ok(())
    }

    /// Whether a chunk of `n` rows is cut in two halves that overlap attention (main card) and experts (cards): needs two row blocks at least.
    pub(super) fn pipe_split(&self, n: usize) -> bool {
        std::env::var("DSC_PF_PIPE").map_or(true, |v| v != "0") && n >= 2 * self.rmax
    }

    fn layer_tiled(&mut self, l: usize, pos0: usize, n: usize) -> Result<()> {
        self.lt_attn(l, pos0, 0, n)?;
        self.lt_experts(l, 0, n, 0, &mut |_| Ok(()))?;
        self.lt_post(l, 0, n)
    }

    /// Two-stage pipeline over two row halves: the experts of one half run on the cards while the main card does the attention of the other.
    fn run_pipelined(&mut self, pos0: usize, n: usize) -> Result<()> {
        let m = self.rmax;
        let split = ((n / 2).div_ceil(m) * m).min(n);
        let nl = self.c.n_layer;
        let (a, b) = ((0usize, split), (split, n));
        self.lt_attn(0, pos0, a.0, a.1)?;
        for l in 0..nl {
            self.lt_experts(l, a.0, a.1, 1, &mut |e| {
                if l > 0 {
                    e.lt_post(l - 1, b.0, b.1)?;
                }
                e.lt_attn(l, pos0, b.0, b.1)
            })?;
            self.lt_experts(l, b.0, b.1, 2, &mut |e| {
                e.lt_post(l, a.0, a.1)?;
                if l + 1 < nl {
                    e.lt_attn(l + 1, pos0, a.0, a.1)?;
                }
                Ok(())
            })?;
        }
        self.lt_post(nl - 1, b.0, b.1)
    }

    fn lt_attn(&mut self, l: usize, pos0: usize, lo: usize, hi: usize) -> Result<()> {
        let c = self.c;
        let (d, nx, m) = (c.n_embd, c.n_expert_used, self.rmax);
        let t0 = Instant::now();
        let (bh, bmf, bx, bids, bwts) = {
            let s = self.big.as_ref().unwrap();
            (s.h.ptr(), s.mf.ptr(), s.x.ptr(), s.ids.ptr(), s.wts.ptr())
        };
        let (bsel, bcm) = { let s = self.big.as_ref().unwrap(); (s.sel.ptr(), s.cmask.ptr()) };
        let (tk, nbm) = (c.idx_top_k, self.max_ctx.div_ceil(c.cand_block.max(1)));
        // 1. attention + router per block
        for r0 in (lo..hi).step_by(m) {
            let nr = m.min(hi - r0);
            self.nr = nr;
            self.s.memcpy_dtod(self.h.ptr(), bh.add::<u16>(r0 * 4 * d), nr * 4 * d * 2)?;
            if l > 0 {
                self.s.memcpy_dtod(self.mix_f.ptr(), bmf.add::<f32>(r0 * 24), nr * 24 * 4)?;
            }
            let slot_ = self.cur_slot as u32;
            let bi = ((r0 - lo) / m) % 16;
            {
                let mh = self.pf_meta_h.as_mut().unwrap();
                let sl = &mut mh.as_mut_slice()[bi * m * 2..(bi + 1) * m * 2];
                for i in 0..nr {
                    sl[2 * i] = slot_;
                    sl[2 * i + 1] = (pos0 + r0 + i) as u32;
                }
                unsafe { self.s.memcpy_htod(self.meta.ptr(), &mh.as_slice()[bi * m * 2..bi * m * 2 + nr * 2])? };
            }
            if let Some(li) = c.engram_layers.iter().position(|&x| x == l) {
                let w = c.engram_width();
                let te = Instant::now();
                // random 144-byte preads from a 110 GB file: cold rows are one NVMe read each, so issue many at once (queue depth) instead of one by one
                let mut all: Vec<f32> = vec![0f32; nr * w];
                {
                    let (eng, ids) = (&self.eng, &self.ids_seen);
                    let base = pos0 + r0;
                    let nt = 48usize.min(nr.max(1));
                    let per = nr.div_ceil(nt);
                    let res: Vec<Result<()>> = std::thread::scope(|sc| {
                        let hs: Vec<_> = all.chunks_mut(per * w).enumerate().map(|(ti, ch)| sc.spawn(move || -> Result<()> {
                            for (k, o) in ch.chunks_mut(w).enumerate() {
                                o.copy_from_slice(&eng.vector(ids, base + ti * per + k, li)?);
                            }
                            Ok(())
                        })).collect();
                        hs.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    for r in res {
                        r?;
                    }
                }
                let eh = self.pf_eng_h.as_mut().unwrap();
                let sl = &mut eh.as_mut_slice()[bi * m * w..bi * m * w + nr * w];
                sl.copy_from_slice(&all);
                unsafe { self.s.memcpy_htod(self.evecs[li].ptr(), &eh.as_slice()[bi * m * w..bi * m * w + nr * w])? };
                self.split.engram_ms += te.elapsed().as_secs_f64() * 1e3;
            }
            self.s.memcpy_dtod(self.sel.ptr(), bsel.add::<i32>(r0 * tk), nr * tk * 4)?;
            self.s.memcpy_dtod(self.cand_mask.ptr(), bcm.add::<u8>(r0 * nbm), nr * nbm)?;
            self.tap_off = r0 * 3 * d;
            self.seg(0, l, pos0 + r0 + nr - 1)?;
            self.s.memcpy_dtod(bh.add::<u16>(r0 * 4 * d), self.h.ptr(), nr * 4 * d * 2)?;
            self.s.memcpy_dtod(bmf.add::<f32>(r0 * 24), self.mix_f.ptr(), nr * 24 * 4)?;
            self.s.memcpy_dtod(bx.add::<u16>(r0 * d), self.x.ptr(), nr * d * 2)?;
            self.s.memcpy_dtod(bids.add::<u32>(r0 * nx), self.ids.ptr(), nr * nx * 4)?;
            self.s.memcpy_dtod(bwts.add::<f32>(r0 * nx), self.wts.ptr(), nr * nx * 4)?;
            self.s.memcpy_dtod(bsel.add::<i32>(r0 * tk), self.sel.ptr(), nr * tk * 4)?;
            self.s.memcpy_dtod(bcm.add::<u8>(r0 * nbm), self.cand_mask.ptr(), nr * nbm)?;
        }
        self.s.sync()?;
        self.split.ph[0] += t0.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// The routed experts of rows lo..hi of layer `l` on the cards; `overlap` runs on this thread (the main card's attention of the other half) while
    /// they work. The fixed-point results are merged into the main card's accumulator rows lo..hi.
    fn lt_experts(&mut self, l: usize, lo: usize, hi: usize, half: u8, overlap: &mut dyn FnMut(&mut Self) -> Result<()>) -> Result<()> {
        let hold = if self.pf_hold { half } else { 0 };
        let c = self.c;
        let (d, nx) = (c.n_embd, c.n_expert_used);
        let n = hi - lo;
        let np = n * nx;
        self.s.sync()?;
        let t1 = Instant::now();
        let (bx, bids, bwts) = {
            let s = self.big.as_ref().unwrap();
            (s.x.ptr().add::<u16>(lo * d), s.ids.ptr().add::<u32>(lo * nx), s.wts.ptr().add::<f32>(lo * nx))
        };
        let two = self.pf_run.as_ref().is_some_and(|r| r.b.is_some());
        let three = self.pf_run.as_ref().is_some_and(|r| r.c.is_some());
        let acc_main = self.pf.as_ref().unwrap().acc.ptr();
        let accb = self.pf.as_ref().unwrap().accb.ptr();
        let selfp: *mut Self = self;
        let r = {
            let Engine { ggq, g, pf, xcs, res, res_pf, xres_pf, _ctx, pf_run, .. } = unsafe { &mut *selfp };
            let run = pf_run.as_mut().ok_or_else(|| anyhow!("big layer without a running chunk"))?;
            let pf_a = pf.as_mut().ok_or_else(|| anyhow!("big path without workspace"))?;
            let job_a = CardJob { ctx: &**_ctx, ggq, pf: pf_a as *mut PfCard, run: &mut run.a as *mut CardRun, g: *g, c, l, nr: n, x: bx,
                ids: bids, wts: bwts, res: if std::env::var("DSC_PF_MAIN_EXP").map_or(false, |v| v == "0") { None } else { res_pf[l].as_ref().or(res[l].as_ref()) }, acc_off: lo * d, hold };
            let mut jobs = vec![];
            let res_b_on = std::env::var("DSC_PF_DBG").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0) & 1 == 0;
            let (run_b, run_c) = (run.b.as_mut().map(|r| r as *mut CardRun), run.c.as_mut().map(|r| r as *mut CardRun));
            for (k, rr) in [(0usize, run_b), (1usize, run_c)] {
                if (k == 0 && !two) || (k == 1 && !three) {
                    continue;
                }
                let xc: &mut super::XCard = unsafe { &mut *xcs.as_mut_ptr().add(k) };
                let pfb = xc.pf.as_mut().unwrap();
                xc.ctx.copy_peer(pfb.x.ptr(), _ctx, bx, n * d * 2, &pfb.cs)?;
                xc.ctx.copy_peer(pfb.ids.ptr(), _ctx, bids, np * 4, &pfb.cs)?;
                xc.ctx.copy_peer(pfb.wts.ptr(), _ctx, bwts, np * 4, &pfb.cs)?;
                jobs.push(CardJob { ctx: &*xc.ctx, ggq: &xc.ggq, pf: pfb as *mut PfCard, run: rr.unwrap(), g: *g, c,
                    l, nr: n, x: pfb.x.ptr(), ids: pfb.ids.ptr(), wts: pfb.wts.ptr(), res: xres_pf[k][l].as_ref().filter(|_| res_b_on), acc_off: 0, hold });
            }
            std::thread::scope(|sc| -> Result<()> {
                let mut hs: Vec<_> = jobs.into_iter().enumerate().map(|(ji, jb)| sc.spawn(move || { let t = Instant::now(); let r = unsafe { run_card(jb) }; CARD_NS[ji + 1].fetch_add(t.elapsed().as_nanos() as usize, Ordering::Relaxed); r })).collect();
                hs.push(sc.spawn(move || { let t = Instant::now(); let r = unsafe { run_card(job_a) }; CARD_NS[0].fetch_add(t.elapsed().as_nanos() as usize, Ordering::Relaxed); r }));
                let t_ov = Instant::now();
                let ro = overlap(unsafe { &mut *selfp });
                OV_NS.fetch_add(t_ov.elapsed().as_nanos() as usize, Ordering::Relaxed);
                let t_jw = Instant::now();
                let mut out = ro;
                for h in hs {
                    let r = h.join().unwrap();
                    out = out.and(r);
                }
                JW_NS.fetch_add(t_jw.elapsed().as_nanos() as usize, Ordering::Relaxed);
                out
            })
        };
        self._ctx.bind()?;
        r?;
        for k in 0..2usize {
            if (k == 0 && !two) || (k == 1 && !three) {
                continue;
            }
            let xc = &self.xcs[k];
            let pfb = xc.pf.as_ref().unwrap();
            self._ctx.copy_peer(accb, &xc.ctx, pfb.acc.ptr(), n * d * 8, &self.s)?;
            self.s.sync()?;
            pfb.cs.memset_d8(pfb.acc.ptr(), 0, n * d * 8)?;
            xc.ctx.bind()?;
            pfb.cs.sync()?;
            self._ctx.bind()?;
            self.ggq.fx_add(&self.s, acc_main.add::<i64>(lo * d), accb, n * d)?;
        }
        self.split.ph[2] += t1.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    fn lt_post(&mut self, l: usize, lo: usize, hi: usize) -> Result<()> {
        let c = self.c;
        let (d, m) = (c.n_embd, self.rmax);
        let (bh, bmf, bx) = {
            let s = self.big.as_ref().unwrap();
            (s.h.ptr(), s.mf.ptr(), s.x.ptr())
        };
        let acc = self.pf.as_ref().unwrap().acc.ptr();
        let t2 = Instant::now();
        for r0 in (lo..hi).step_by(m) {
            let nr = m.min(hi - r0);
            self.nr = nr;
            self.s.memcpy_dtod(self.x.ptr(), bx.add::<u16>(r0 * d), nr * d * 2)?;
            self.s.memcpy_dtod(self.h.ptr(), bh.add::<u16>(r0 * 4 * d), nr * 4 * d * 2)?;
            self.s.memcpy_dtod(self.mix_f.ptr(), bmf.add::<f32>(r0 * 24), nr * 24 * 4)?;
            self.seg_shared(l)?;
            self.ggq.fx_finish(&self.s, self.part.ptr(), acc.add::<i64>(r0 * d), nr * d, true)?;
            let mf = self.mix_f.ptr();
            self.post(mf)?;
            self.s.memcpy_dtod(bh.add::<u16>(r0 * 4 * d), self.h.ptr(), nr * 4 * d * 2)?;
        }
        self.s.sync()?;
        self.split.ph[3] += t2.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// Streamed bytes of the big path since the last call: (card A, card B).
    pub fn pf_take_stats(&mut self) -> (usize, usize) {
        let a = self.pf.as_ref().map_or(0, |p| p.bytes.swap(0, Ordering::Relaxed));
        let b = self.xcs.first().and_then(|x| x.pf.as_ref()).map_or(0, |p| p.bytes.swap(0, Ordering::Relaxed))
            + self.xcs.get(1).and_then(|x| x.pf.as_ref()).map_or(0, |p| p.bytes.swap(0, Ordering::Relaxed));
        (a, b)
    }

    /// Fixed-point range overflows (dropped terms) counted so far on the cards (must be 0).
    pub fn pf_status(&self) -> Result<u32> {
        let mut v = [0u32; 1];
        let mut tot = 0;
        if let Some(p) = &self.pf {
            memcpy_dtoh(&mut v, p.status.ptr())?;
            tot += v[0];
        }
        for x in self.xcs.iter() {
            if let Some(p) = &x.pf {
                x.ctx.bind()?;
                memcpy_dtoh(&mut v, p.status.ptr())?;
                self._ctx.bind()?;
                tot += v[0];
            }
        }
        Ok(tot)
    }
}
