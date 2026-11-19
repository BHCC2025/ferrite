//! Prompt chunks (up to FNEXT_CHUNK rows at once, default 1024): int8 GEMMs (k-gguf mmq) for the quantized
//! weights, bf16 GEMMs for the bf16 ones, the decode kernels with `rows` = chunk for the rest, and the MoE as sorted
//! grouped GEMMs: experts with a decode slot run from it, the others are streamed to the card in groups (two
//! buffers, a copy stream) and run group by group. No CUDA graph (launch overhead is noise at these sizes).
use super::*;
use fe_ggq::gpu::SortWs;
use std::sync::Arc;

pub(crate) struct PWork {
    /// context the indexer score rows are sized for (prompt end, not the session maximum)
    pub(crate) ctx: usize,
    /// 1 / rms of each (row, stream) of the last hc norm
    rinv: DeviceBuffer<f32>,
    /// scratch of quant's prompt attention (tile window lists)
    aws: fe_ggq::gpu::AttnWs,
    /// whole-group MoE (prefetch workspaces): i64 expert sums [gmax][n_embd], q8 h of every pair, one sort over the
    /// group, the merged slot | arena tables of the layer, the shared-expert gate logit per token, the range status
    wg: bool,
    acc: DeviceBuffer<i64>,
    hq: Q8Act,
    ws_g: SortWs,
    mtab: DeviceBuffer<u64>,
    glog_n: DeviceBuffer<f32>,
    status: DeviceBuffer<u32>,
    /// expert sums of the group [gmax][n_embd] f32 (moe_combine / moe_down_ordered), the per-pair rows behind them
    /// (short prompts: [pairs][n_embd]; one-pass long prompts: [pairs][MOE_NCOL]) and moe_down_ordered's table scratch
    moey: DeviceBuffer<f32>,
    ybuf: DeviceBuffer<f32>,
    tabw: DeviceBuffer<u64>,
    /// per pair: its row among the CPU-computed ones (-1: on the card)
    slot: DeviceBuffer<i32>,
    pub cmax: usize,
    res: DeviceBuffer<f32>,
    xn: DeviceBuffer<f32>,
    lo: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
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
    rl: DeviceBuffer<f32>,
    sg: DeviceBuffer<f32>,
    su: DeviceBuffer<f32>,
    sh: DeviceBuffer<f32>,
    hg: DeviceBuffer<f32>,
    hu: DeviceBuffer<f32>,
    hh: DeviceBuffer<f32>,
    ey: DeviceBuffer<f32>,
    emb: DeviceBuffer<f32>,
    pemb: DeviceBuffer<f32>,
    pval: DeviceBuffer<f32>,
    bfx: DeviceBuffer<u16>,
    xa: Q8Act,
    ea: Q8Act,
    sa: Q8Act,
    ha: Q8Act,
    score: DeviceBuffer<f32>,
    sel: DeviceBuffer<u32>,
    seln: DeviceBuffer<u32>,
    ids: DeviceBuffer<i32>,
    wts: DeviceBuffer<f32>,
    cs: DeviceBuffer<f32>,
    sn: DeviceBuffer<f32>,
    pcs: DeviceBuffer<f32>,
    psn: DeviceBuffer<f32>,
    prm: DeviceBuffer<u32>,
    ws: SortWs,
    ones: DeviceBuffer<u64>,
    /// per streamed group: device tables [3][n_expert]
    gtabs: [DeviceBuffer<u64>; 2],
    arena: [DeviceBuffer<u8>; 2],
    pub group: usize,
    cps: Stream,
    up_done: [Event; 2],
    use_done: [Event; 2],
    stage: Vec<HostBuffer<u8>>,
    /// prompt group (layer-major): residual of every row, the FFN inputs (q8), picks, weights, inject logits
    pub gmax: usize,
    pub hall: DeviceBuffer<f32>,
    ea_n: Q8Act,
    ids_n: DeviceBuffer<i32>,
    wts_n: DeviceBuffer<f32>,
    inj_n: DeviceBuffer<f32>,
    cs_n: DeviceBuffer<f32>,
    sn_n: DeviceBuffer<f32>,
    pcs_n: DeviceBuffer<f32>,
    psn_n: DeviceBuffer<f32>,
    prm_n: DeviceBuffer<u32>,
    ws_n: Vec<SortWs>,
    /// FNEXT_TP=2: chunk-sized buffers of the two-card all-reduce of the prompt rows: [0..2) this card's partial
    /// (then the sum) per slot, [2..4) the other card's partial copied in (empty otherwise)
    tpo: Vec<DeviceBuffer<f32>>,
    /// FNEXT_TP=2: the mixer-side hc inject logits of every row of the group (all chunks' mixers run before their
    /// combines)
    inja_n: DeviceBuffer<f32>,
}

fn slot_max(c: &Config, g: &Gguf) -> Result<usize> {
    let mut m = 0;
    for l in 0..c.n_layer {
        m = m.max(expert_bytes(g, l)?);
    }
    Ok(m)
}

impl PWork {
    fn with_sum_mode(mut self) -> PWork {
        let sm = super::sum_mode();
        for q8 in [&mut self.xa, &mut self.ea, &mut self.sa, &mut self.ha, &mut self.ea_n] {
            q8.sum_mode = sm;
        }
        self
    }

    pub fn alloc(c: &Config, g: &Gguf, cmax: usize, ctx: usize, group: usize, gmax: usize, tp: bool) -> Result<PWork> {
        let gmax = gmax.max(cmax).next_multiple_of(cmax);
        let nch = gmax / cmax;
        let a = |n: usize| DeviceBuffer::<f32>::zeroed(n);
        let r = cmax;
        let hd = c.hc_dim();
        let topk = c.n_expert_used;
        let np = r * topk;
        let sm = slot_max(c, g)?;
        let ld_lo = (c.hc_lr + c.hc).next_multiple_of(64);
        // experts of the whole group per layer into an i64 accumulator (order-free: the same bits whichever experts
        // are resident); FNEXT_MOE=chunk = the old per-chunk f32 combine
        let wg = std::env::var("FNEXT_MOE").as_deref() != Ok("chunk");
        // per-pair buffers of the chunked MoE path (a whole-group workspace keeps ey as PLE scratch only)
        let npo = if wg { 1 } else { np };
        let stage = if PROBE.load(std::sync::atomic::Ordering::Relaxed) {
            (0..2).map(|_| HostBuffer::alloc(1).map_err(Into::into)).collect::<Result<Vec<_>>>()?
        } else {
            (0..2).map(|_| stage_take(if group >= c.n_expert { 1 } else { group * sm })).collect::<Result<Vec<_>>>()?
        };
        Ok((PWork {
            wg,
            acc: DeviceBuffer::zeroed(1)?,
            moey: a(if wg { gmax * c.n_embd } else { 1 })?,
            ybuf: a(if !wg { 1 } else if group >= c.n_expert { gmax * topk * MOE_NCOL } else { gmax * topk * c.n_embd })?,
            tabw: DeviceBuffer::zeroed(c.n_expert)?,
            slot: DeviceBuffer::zeroed(if wg { gmax * topk } else { 1 })?,
            hq: Q8Act::alloc(if wg { gmax * topk } else { 1 }, c.n_ff_exp)?,
            ws_g: SortWs::alloc(if wg { gmax * topk } else { topk }, c.n_expert)?,
            mtab: DeviceBuffer::zeroed(3 * c.n_expert)?,
            glog_n: a(gmax)?,
            status: DeviceBuffer::zeroed(1)?,
            cmax,
            ctx,
            rinv: a(r * c.hc)?,
            aws: fe_ggq::gpu::AttnWs::new(r, ctx)?,
            res: a(r * hd)?,
            xn: a(r * hd)?,
            lo: a(r * ld_lo)?,
            gate: a(r * hd)?,
            mixed: a(r * c.n_embd)?,
            out: a(r * c.n_embd)?,
            qkv: a(r * c.conv_dim())?,
            z: a(r * c.ssm_inner)?,
            ab: a(r * (2 * c.ssm_vh).next_multiple_of(64))?,
            conv: a(r * c.conv_dim())?,
            o: a(r * c.ssm_inner)?,
            y: a(r * c.ssm_inner)?,
            qf: a(r * 2 * c.n_head * c.head_dim)?,
            kx: a(r * c.n_head_kv * c.head_dim)?,
            vx: a(r * c.n_head_kv * c.head_dim)?,
            q: a(r * c.n_head * c.head_dim)?,
            att: a(r * c.n_head * c.head_dim)?,
            ikx: a(r * c.idx_dim.next_multiple_of(64))?,
            iqv: a(r * c.idx_heads * c.idx_dim)?,
            rl: a(r * (c.n_expert + 1).next_multiple_of(64))?,
            sg: a(r * c.n_ff_shexp)?,
            su: a(r * c.n_ff_shexp)?,
            sh: a(r * c.n_embd)?,
            hg: a(npo * c.n_ff_exp)?,
            hu: a(npo * c.n_ff_exp)?,
            hh: a(npo * c.n_ff_exp)?,
            ey: a((npo * c.n_embd).max(r * hd))?,
            emb: a(r * c.n_embd)?,
            pemb: a(r * c.n_embd)?,
            pval: a(r * c.n_embd)?,
            bfx: DeviceBuffer::zeroed(r * 12288)?,
            xa: Q8Act::alloc(r, 12288)?,
            ea: Q8Act::alloc(r, c.n_embd)?,
            sa: Q8Act::alloc(r, c.n_ff_shexp)?,
            ha: Q8Act::alloc(npo, c.n_ff_exp)?,
            score: a(r * (ctx / c.kpool + 1))?,
            sel: DeviceBuffer::zeroed(r * (c.idx_top_k + c.kpool - 1))?,
            seln: DeviceBuffer::zeroed(r)?,
            ids: DeviceBuffer::zeroed(np)?,
            wts: a(np)?,
            cs: a(r * 32)?,
            sn: a(r * 32)?,
            pcs: a((r / c.kpool + 2) * 32)?,
            psn: a((r / c.kpool + 2) * 32)?,
            prm: DeviceBuffer::zeroed(P_WORDS)?,
            ws: SortWs::alloc(np, c.n_expert)?,
            ones: DeviceBuffer::from_slice(&vec![1u64; c.n_expert])?,
            gtabs: [DeviceBuffer::zeroed(3 * c.n_expert)?, DeviceBuffer::zeroed(3 * c.n_expert)?],
            arena: [DeviceBuffer::alloc(group * sm)?, DeviceBuffer::alloc(group * sm)?],
            group,
            cps: Stream::new()?,
            up_done: [Event::new_no_timing()?, Event::new_no_timing()?],
            use_done: [Event::new_no_timing()?, Event::new_no_timing()?],
            // the whole-layer prefetch stages on its own thread: no group staging needed then
            stage,
            gmax,
            hall: a(gmax * hd)?,
            ea_n: Q8Act::alloc(gmax, c.n_embd)?,
            ids_n: DeviceBuffer::zeroed(gmax * topk)?,
            wts_n: a(gmax * topk)?,
            inj_n: a(gmax * c.hc)?,
            cs_n: a(gmax * 32)?,
            sn_n: a(gmax * 32)?,
            pcs_n: a(nch * (r / c.kpool + 2) * 32)?,
            psn_n: a(nch * (r / c.kpool + 2) * 32)?,
            prm_n: DeviceBuffer::zeroed(nch * P_WORDS)?,
            ws_n: (0..nch).map(|_| SortWs::alloc(np, c.n_expert)).collect::<Result<Vec<_>>>()?,
            tpo: if tp { (0..4).map(|_| a(r * c.n_embd)).collect::<fe_cuda::Result<Vec<_>>>()? } else { vec![] },
            inja_n: a(if tp { gmax * c.hc } else { 1 })?,
        }
        ).with_sum_mode())
    }
}

impl Fast {
    /// Quantized weight x f32 rows: q8 the rows, int8 GEMM.
    unsafe fn pq(&self, qw: &QW, x: DevPtr, ldx: usize, rows: usize, y: DevPtr, ldy: usize, a: &mut Q8Act) -> Result<()> {
        self.q.quantize_q8_1(&self.s, x, ldx, false, rows, qw.k, a)?;
        self.q.mmq(&self.s, qw.ty, y, ldy, a, qw.buf.ptr(), qw.ty.row_bytes(qw.k), qw.n, false)
    }

    /// bf16 (or f16) weight x f32 rows: bf16 the rows, bf16 GEMM over the 64-padded output rows.
    unsafe fn pb(&self, l: &Lin, x: DevPtr, ldx: usize, rows: usize, y: DevPtr, ldy: usize, bfx: DevPtr) -> Result<()> {
        self.q.f32_to_bf16(&self.s, bfx, l.k, x, ldx, rows, l.k)?;
        let ty = if l.bf { GgmlType::BF16 } else { GgmlType::F16 };
        self.q.gemm(&self.s, ty, y, ldy, bfx, l.k, l.w.ptr(), l.k * 2, rows, l.n.next_multiple_of(64), l.k, false)
    }

    /// Run up to `gmax` prompt tokens at the next positions, layer by layer over chunks of `cmax` rows: each layer's
    /// experts without a decode slot cross the bus once for the whole group. Returns the greedy token after the
    /// last row.
    pub fn prefill_group(&mut self, toks: &[u32]) -> Result<u32> {
        if self.tp {
            return self.prefill_group_tp(toks);
        }
        let n = self.tokens.len() + toks.len();
        self.ensure_ctx(n + MAX_ROWS)?;
        let mut pw = self.pw.take().ok_or_else(|| anyhow!("no prompt workspace (FNEXT_CHUNK=0)"))?;
        // layer split: device 0's workspace too (the group runs blocks 0..split there, then the rest here)
        let mut pw0 = match self.other.as_mut() {
            Some(o) => match o.pw.take() {
                Some(p) => Some(p),
                None => {
                    self.pw = Some(pw);
                    bail!("no prompt workspace on device 0");
                }
            },
            None => None,
        };
        let r = unsafe { self.group_inner(&mut pw, pw0.as_mut(), toks) };
        // an error on device 0's half leaves its set active: back to the last device
        if self.other.is_some() && !self.is_last() {
            self.swap_dev()?;
        }
        self.pw = Some(pw);
        if let Some(p) = pw0 {
            self.other.as_mut().unwrap().pw = Some(p);
        }
        r
    }

    /// Step parameters and rope tables of a group's chunks into `pw` (synchronous copies on the current device).
    unsafe fn group_params(&self, pw: &PWork, pos0: usize, n_all: usize, chunks: &[(usize, usize)]) -> Result<()> {
        let c = &self.c;
        let kp = c.kpool;
        let cm = pw.cmax;
        let half = c.n_rot / 2;
        let ts = c.rope_base.powf(-2.0 / c.n_rot as f32);
        let tab = |pos: &[usize]| -> (Vec<f32>, Vec<f32>) {
            let (mut cs, mut sn) = (vec![0f32; pos.len() * 32], vec![0f32; pos.len() * 32]);
            for (i, &p) in pos.iter().enumerate() {
                let mut th = p as f32;
                for k in 0..half {
                    cs[i * 32 + k] = th.cos();
                    sn[i * 32 + k] = th.sin();
                    th *= ts;
                }
            }
            (cs, sn)
        };
        let (cs, sn) = tab(&(pos0..pos0 + n_all).collect::<Vec<_>>());
        fe_cuda::memcpy_htod(pw.cs_n.ptr(), &cs)?;
        fe_cuda::memcpy_htod(pw.sn_n.ptr(), &sn)?;
        let pstride = (cm / kp + 2) * 32;
        for (j, &(a, rows)) in chunks.iter().enumerate() {
            let p = pos0 + a;
            let (b_lo, b_hi) = (p / kp, (p + rows) / kp);
            let (pcs, psn) = tab(&(b_lo..b_hi).map(|b| b * kp).collect::<Vec<_>>());
            if !pcs.is_empty() {
                fe_cuda::memcpy_htod(pw.pcs_n.ptr().add::<f32>(j * pstride), &pcs)?;
                fe_cuda::memcpy_htod(pw.psn_n.ptr().add::<f32>(j * pstride), &psn)?;
            }
            let mut prm = vec![0u32; P_WORDS];
            prm[P_POS] = p as u32;
            prm[P_ROWS] = rows as u32;
            prm[P_NB] = (b_hi - b_lo) as u32;
            prm[P_BLO] = b_lo as u32;
            fe_cuda::memcpy_htod(pw.prm_n.ptr().add::<u32>(j * P_WORDS), &prm)?;
        }
        Ok(())
    }

    /// `pw_last`: the active (last) device's workspace; `pw_first`: device 0's under a layer split.
    unsafe fn group_inner(&mut self, pw_last: &mut PWork, pw_first: Option<&mut PWork>, toks: &[u32]) -> Result<u32> {
        let c = self.c.clone();
        let n_all = toks.len();
        let two = pw_first.is_some();
        let (pw, mut pw_next): (&mut PWork, Option<&mut PWork>) = match pw_first {
            Some(p) => (p, Some(pw_last)),
            None => (pw_last, None),
        };
        let mut pw = pw;
        let segs: Vec<(usize, usize)> = match &self.other {
            Some(o) => vec![(0, o.hi), (o.hi, c.n_layer)],
            None => vec![(0, c.n_layer)],
        };
        if two {
            // device 0 runs the first blocks
            self.swap_dev()?;
        }
        if n_all == 0 || n_all > pw.gmax {
            bail!("prefill_group: {n_all} rows (max {})", pw.gmax);
        }
        let pos0 = self.tokens.len();
        if pos0 + n_all > self.ctx {
            bail!("context full");
        }
        if pos0 + n_all > pw.ctx {
            bail!("prompt workspace sized for {} positions, need {}", pw.ctx, pos0 + n_all);
        }
        let d = c.n_embd;
        let hd = c.hc_dim();
        let topk = c.n_expert_used;
        let kp = c.kpool;
        let cm = pw.cmax;
        let chunks: Vec<(usize, usize)> = (0..n_all).step_by(cm).map(|a| (a, (n_all - a).min(cm))).collect();
        let s = &self.s;
        s.sync()?;
        let mut prof = Prof::new();
        // ---- per-chunk step parameters and rope tables ----
        self.group_params(pw, pos0, n_all, &chunks)?;
        let pstride = (cm / kp + 2) * 32;
        // ---- residual streams from the token embeddings ----
        let te = self.g.tensor("token_embd.weight").unwrap();
        let (trb, tdata) = (te.row_bytes(), self.g.data(te));
        let kn = &self.kn;
        for &(a, rows) in &chunks {
            let mut emb = vec![0f32; rows * d];
            emb.par_chunks_mut(d).zip(toks[a..a + rows].par_iter()).for_each(|(o, &t)| {
                fe_ggq::refq::dequant(te.ty, &tdata[t as usize * trb..(t as usize + 1) * trb], o);
            });
            fe_cuda::memcpy_htod(pw.emb.ptr(), &emb)?;
            let n = rows * d;
            s.launch(&kn.hc_init, g256(n), 256, 0, (pw.hall.ptr().add::<f32>(a * hd), pw.emb.ptr(), d as u32, c.hc as u32,
                n as u32))?;
        }
        let ld_lo = (c.hc_lr + c.hc).next_multiple_of(64);
        let ld_ab = (2 * c.ssm_vh).next_multiple_of(64);
        let ld_rl = (c.n_expert + 1).next_multiple_of(64);
        prof.mark(s, 8);
        // PLE rows of the whole group, gathered on a host thread while layer 0 runs (random 4 KB pages of a 27.5 GB
        // table: every page is requested ahead with WILLNEED so the reads go out in parallel)
        let idx_all = crate::engine::ple_rows(&c, &self.tokens, pos0, toks);
        let ple_job = {
            let g = self.g.clone();
            let idx = idx_all.clone();
            let page = self.hm.page();
            std::thread::spawn(move || {
                let tq = std::time::Instant::now();
                let pt = g.tensor("per_layer_token_embd.weight").unwrap();
                let ple = hostmem::ple_gather(&g, pt, &idx, page);
                if std::env::var("FNEXT_PF_TIME").is_ok() {
                    eprintln!("PLE gather: {} rows in {:.0} ms", idx.len(), tq.elapsed().as_secs_f64() * 1e3);
                }
                ple
            })
        };
        let mut ple_job = Some(ple_job);
        // long groups: whole-layer expert prefetch one layer ahead (arena = every expert of a layer)
        let ne = c.n_expert;
        let slot_tabs = |fast: &Fast, l: usize, base: u64| -> (Vec<usize>, Vec<u64>) {
            let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
            let sb: Vec<usize> = names.iter().map(|nm| fast.g.tensor(&format!("blk.{l}.{nm}.weight")).unwrap().slice_bytes().div_ceil(256) * 256).collect();
            let eb: usize = sb.iter().sum();
            let ex: Vec<usize> = (0..ne).filter(|&e| fast.tabs_h[l][2][e] == 0).collect();
            let mut tabs = vec![0u64; 3 * ne];
            for (j, &e) in ex.iter().enumerate() {
                let b = base + (j * eb) as u64;
                tabs[e] = b;
                tabs[ne + e] = b + sb[0] as u64;
                tabs[2 * ne + e] = b + (sb[0] + sb[1]) as u64;
            }
            (ex, tabs)
        };
        let mut pf_tabs: [Vec<u64>; 2] = [vec![], vec![]];
        // prompt GDN kernel: tensor cores (3xTF32) by default; FNEXT_GDN=simt | tc1 (single TF32)
        // prompt indexer scores: tensor cores (TF32) by default; FNEXT_IDX=simt | tc3 (3xTF32)
        let idx_kind = match std::env::var("FNEXT_IDX").as_deref() { Ok("simt") => 0, Ok("tc3") => 2, _ => 1 };
        let gdn_kind = match std::env::var("FNEXT_GDN").as_deref() { Ok("simt") => 0, Ok("tc1") => 2, _ => 1 };
        let mut ple_all: Vec<f32> = vec![];
        for (si, &(lo, hi)) in segs.iter().enumerate() {
        if si > 0 {
            // layer split: the group's residual rows to the last device, its step parameters there, then blocks lo..hi
            self.s.sync()?;
            let nb = pw_next.take().ok_or_else(|| anyhow!("layer split without a second workspace"))?;
            let tx = std::time::Instant::now();
            self.swap_dev()?;
            let src = self.other.as_ref().unwrap().cu.clone();
            self.cu.copy_peer(nb.hall.ptr(), &src, pw.hall.ptr(), n_all * hd * 4, &self.s)?;
            pw = nb;
            self.group_params(pw, pos0, n_all, &chunks)?;
            if std::env::var("FNEXT_PF_TIME").is_ok() {
                self.s.sync()?;
                eprintln!("layer split hand-off: {n_all} rows ({:.0} MB) in {:.1} ms", (n_all * hd * 4) as f64 / 1e6,
                    tx.elapsed().as_secs_f64() * 1e3);
            }
        }
        let (s, kn, fk) = (&self.s, &self.kn, &self.fk);
        let bfx = pw.bfx.ptr();
        // the workspace decides (ensure_pw sizes it for prefetch on long prompts; a short tail group of a long prompt
        // has no streaming stage then)
        let pf_on = pw.group >= ne;
        let pf = if pf_on { Some(Prefetch::start(self.g.clone(), self.cu.clone())) } else { None };
        let pf_ev: [Arc<Event>; 2] = [Arc::new(Event::new_no_timing()?), Arc::new(Event::new_no_timing()?)];
        let use_ev: [Arc<Event>; 2] = [Arc::new(Event::new_no_timing()?), Arc::new(Event::new_no_timing()?)];
        if let Some(p) = &pf {
            let k0 = lo % 2;
            let (ex, tabs) = slot_tabs(self, lo, pw.arena[k0].ptr().0);
            pf_tabs[k0] = tabs;
            p.tx.as_ref().unwrap().send(PfJob { layer: lo, experts: ex, base: pw.arena[k0].ptr().0, wait: use_ev[k0].clone(),
                done: pf_ev[k0].clone() }).ok();
        }
        for l in lo..hi {
            if self.cancelled() {
                // the client went away: stop between layers (the lane's state is incomplete; the caller restores it)
                drop(pf);
                return Err(self.stop_cancelled());
            }
            let ly = &self.layers[l];
            if let Some(p) = &pf {
                if l + 1 < hi {
                    let k = (l + 1) % 2;
                    let (ex, tabs) = slot_tabs(self, l + 1, pw.arena[k].ptr().0);
                    pf_tabs[k] = tabs;
                    p.tx.as_ref().unwrap().send(PfJob { layer: l + 1, experts: ex, base: pw.arena[k].ptr().0,
                        wait: use_ev[k].clone(), done: pf_ev[k].clone() }).ok();
                }
            }
            for (j, &(a, rows)) in chunks.iter().enumerate() {
                let res = pw.hall.ptr().add::<f32>(a * hd);
                let prmp = pw.prm_n.ptr().add::<u32>(j * P_WORDS);
                let (csp, snp) = (pw.cs_n.ptr().add::<f32>(a * 32), pw.sn_n.ptr().add::<f32>(a * 32));
                let (pcsp, psnp) = (pw.pcs_n.ptr().add::<f32>(j * pstride), pw.psn_n.ptr().add::<f32>(j * pstride));
                if let Some(p) = &ly.ple {
                    prof.mark(s, 12);
                    if let Some(h) = ple_job.take() {
                        ple_all = h.join().map_err(|_| anyhow!("PLE gather thread panicked"))?;
                        let pt = self.g.tensor("per_layer_token_embd.weight").unwrap();
                        if hostmem::ple_mmap() {
                            for &rw in &idx_all {
                                self.hm.drop_range(&self.g, pt, rw as usize * pt.row_bytes(), pt.row_bytes());
                            }
                        }
                    }
                    let ple = &ple_all[a * d..(a + rows) * d];
                    prof.mark(s, 13);
                    // stream-ordered (pageable source: staged before the call returns); a context-wide copy would
                    // also wait for the expert prefetch stream
                    s.memcpy_htod(pw.pemb.ptr(), ple)?;
                    self.pb(&p.key, pw.pemb.ptr(), d, rows, pw.qkv.ptr(), hd, bfx)?;
                    self.pb(&p.value, pw.pemb.ptr(), d, rows, pw.pval.ptr(), d, bfx)?;
                    kn.rmsnorm(s, pw.conv.ptr(), pw.qkv.ptr(), p.nk.ptr(), d, c.hc, rows * c.hc, c.eps)?;
                    kn.rmsnorm(s, pw.gate.ptr(), res, p.nq.ptr(), d, c.hc, rows * c.hc, c.eps)?;
                    s.launch(&kn.ple_gate, (rows * c.hc) as u32, 256, 0,
                        (pw.xn.ptr(), pw.conv.ptr(), pw.gate.ptr(), pw.pval.ptr(), d as u32, c.hc as u32))?;
                    kn.rmsnorm(s, pw.ey.ptr(), pw.xn.ptr(), p.nc.ptr(), d, c.hc, rows * c.hc, c.eps)?;
                    prof.mark(s, 0);
                    s.launch(&kn.ple_conv, g256(hd), 256, 0, (res, pw.xn.ptr(), pw.ey.ptr(), p.conv_w.ptr(),
                        p.hist.ptr(), rows as u32, hd as u32, c.ple_ngram as u32, DevPtr::NULL, 0u32))?;
                    prof.mark(s, 9);
                }
                prof.mark(s, 0);
                self.p_hc_mix(pw, res, &ly.hca, rows, &mut prof)?;
                prof.mark(s, 1);
                match &ly.mix {
                    Mixer::Gdn(gd) => {
                        let cd = c.conv_dim();
                        self.pq(&gd.qkv, pw.mixed.ptr(), d, rows, pw.qkv.ptr(), cd, &mut pw.xa)?;
                        self.pq(&gd.z, pw.mixed.ptr(), d, rows, pw.z.ptr(), c.ssm_inner, &mut pw.xa)?;
                        prof.mark(s, 14);
                        self.pb(&gd.ab, pw.mixed.ptr(), d, rows, pw.ab.ptr(), ld_ab, bfx)?;
                        prof.mark(s, 15);
                        if std::env::var("FNEXT_CONV_SEQ").is_ok() {
                            s.launch(&kn.gdn_conv, g256(cd), 256, 0, (pw.conv.ptr(), pw.qkv.ptr(), cd as u32, gd.conv_w.ptr(),
                                gd.conv_state.ptr(), rows as u32, cd as u32, 0u32))?;
                        } else {
                            s.launch(&fk.conv_p, (g256(cd), rows.div_ceil(16) as u32), 256, 0, (pw.conv.ptr(), pw.qkv.ptr(),
                                cd as u32, gd.conv_w.ptr(), gd.conv_state.ptr(), rows as u32, cd as u32))?;
                            s.launch(&fk.conv_state, g256(cd), 256, 0, (gd.conv_state.ptr(), pw.qkv.ptr(), cd as u32,
                                rows as u32, cd as u32))?;
                        }
                        prof.mark(s, 16);
                        if std::env::var("FNEXT_GDN_ROWS").is_ok() {
                            s.launch(&kn.gdn_rec, c.ssm_vh as u32, 512, 0, (pw.o.ptr(), pw.conv.ptr(), pw.ab.ptr(), ld_ab as u32,
                                gd.dt.ptr(), gd.a.ptr(), gd.ssm.ptr(), rows as u32, c.ssm_kh as u32, c.ssm_vh as u32, c.eps, 0u32))?;
                        } else {
                            let args = (pw.o.ptr(), pw.conv.ptr(), pw.ab.ptr(), ld_ab as u32, gd.a.ptr(), gd.dt.ptr(),
                                gd.ssm.ptr(), rows as u32);
                            match gdn_kind {
                                1 => s.launch(&fk.gdn_tc, c.ssm_vh as u32, 256, super::GDN_TC_SMEM, args)?,
                                2 => s.launch(&fk.gdn_tc1, c.ssm_vh as u32, 256, super::GDN_TC_SMEM, args)?,
                                _ => s.launch(&fk.gdn_chunk, c.ssm_vh as u32, 512, GDN_CHUNK_SMEM, args)?,
                            }
                        }
                        prof.mark(s, 17);
                        s.launch(&kn.gdn_gnorm, (rows as u32, c.ssm_vh.div_ceil(8) as u32), 256, 0, (pw.y.ptr(), pw.o.ptr(),
                            pw.z.ptr(), c.ssm_inner as u32, gd.norm.ptr(), c.ssm_vh as u32, c.eps))?;
                        prof.mark(s, 18);
                        self.pq(&gd.out, pw.y.ptr(), c.ssm_inner, rows, pw.out.ptr(), d, &mut pw.xa)?;
                        prof.mark(s, 10);
                    }
                    Mixer::Attn(at) => {
                        let qd = c.n_head * c.head_dim;
                        let kvd = c.n_head_kv * c.head_dim;
                        let dd = c.idx_dim;
                        self.pq(&at.wq, pw.mixed.ptr(), d, rows, pw.qf.ptr(), 2 * qd, &mut pw.xa)?;
                        self.pq(&at.wk, pw.mixed.ptr(), d, rows, pw.kx.ptr(), kvd, &mut pw.xa)?;
                        self.pq(&at.wv, pw.mixed.ptr(), d, rows, pw.vx.ptr(), kvd, &mut pw.xa)?;
                        self.pb(&at.ik, pw.mixed.ptr(), d, rows, pw.ikx.ptr(), dd, bfx)?;
                        prof.mark(s, 19);
                        let rn = (at.ring.len() / dd) as u32;
                        s.launch(&fk.idx_kstore, rows as u32, dd as u32, 0, (at.ring.ptr(), pw.ikx.ptr(), prmp, rn))?;
                        s.launch(&fk.idx_pool, (rows / kp + 1) as u32, dd as u32, 0, (at.pooled.ptr(), at.ring.ptr(),
                            at.ikn.ptr(), prmp, kp as u32, c.n_rot as u32, c.eps, rn, pcsp, psnp))?;
                        self.pb(&at.iq, pw.mixed.ptr(), d, rows, pw.iqv.ptr(), dd * c.idx_heads, bfx)?;
                        kn.rmsnorm(s, pw.iqv.ptr(), pw.iqv.ptr(), at.iqn.ptr(), dd, 1, rows * c.idx_heads, c.eps)?;
                        s.launch(&kn.rope_rows, (rows as u32, c.idx_heads as u32), dd as u32, 0,
                            (pw.iqv.ptr(), csp, snp, c.idx_heads as u32, c.n_rot as u32))?;
                        let top = (c.idx_top_k / kp) as u32;
                        let ldsc = (pw.ctx / kp + 1) as u32;
                        let ldsel = (c.idx_top_k + kp - 1) as u32;
                        if idx_kind == 0 || c.idx_heads != 4 {
                            s.launch(&fk.idx_score, (rows as u32, 72), 256, 0, (pw.score.ptr(), pw.iqv.ptr(), at.pooled.ptr(),
                            c.idx_heads as u32, prmp, kp as u32, ldsc, top, 1.0 / (dd as f32).sqrt()))?;
                        } else {
                            let tiles = rows.div_ceil(16);
                            let splits = (144 / tiles).max(1) as u32;
                            let f = if idx_kind == 2 { &fk.idx_tc3 } else { &fk.idx_tc };
                            s.launch(f, (tiles as u32, splits), 128, super::IDX_TC_SMEM, (pw.score.ptr(), pw.iqv.ptr(),
                                at.pooled.ptr(), prmp, kp as u32, ldsc, top, 1.0 / (dd as f32).sqrt(), rows as u32))?;
                        }
                        s.launch(&fk.topk, rows as u32, 1024, 0, (pw.sel.ptr(), pw.seln.ptr(), pw.score.ptr(), prmp,
                            kp as u32, top, ldsc, ldsel))?;
                        prof.mark(s, 20);
                        s.launch(&fk.attn_prep, (rows as u32, (c.n_head + c.n_head_kv) as u32), 256, 0,
                            (pw.q.ptr(), at.kq.ptr(), at.ks.ptr(), at.vq.ptr(), at.vs.ptr(), pw.qf.ptr(), (2 * qd) as u32,
                                (2 * c.head_dim) as u32, pw.kx.ptr(), pw.vx.ptr(), kvd as u32, at.qn.ptr(), at.kn.ptr(), prmp,
                                c.n_head as u32, c.n_head_kv as u32, c.n_rot as u32, c.eps, csp, snp))?;
                        // prompt attention: quant's tensor-core kernel (16-row tiles over the union of their windows)
                        // by default; FNEXT_PATTN=1 fx_pattn, 0 the SIMT fx_attn
                        let pk = std::env::var("FNEXT_PATTN").unwrap_or_default();
                        if pk.is_empty() && (c.n_head / c.n_head_kv) % fe_ggq::gpu::FA_HW == 0 {
                            self.q.attn_prompt(s, &pw.aws, pw.att.ptr(), pw.q.ptr(), at.kq.ptr(), at.ks.ptr(), at.vq.ptr(),
                                at.vs.ptr(), pw.qf.ptr().add::<f32>(c.head_dim), 2 * qd, 2 * c.head_dim, c.n_head,
                                c.n_head_kv, prmp, 1.0 / (c.head_dim as f32).sqrt(), pw.sel.ptr(), pw.seln.ptr(),
                                ldsel as usize, rows)?;
                        } else {
                            let tc = pk != "0" && c.n_head / c.n_head_kv <= 16;
                            let (af, ny, nt, smem) = if tc { (&fk.pattn, c.n_head_kv, 128, PATTN_SMEM) }
                                else if c.n_head % 6 == 0 && std::env::var("FNEXT_ATTN6").is_ok() { (&fk.attn6, c.n_head / 6, 256, 0) }
                                else { (&fk.attn, c.n_head, 256, 0) };
                            s.launch(af, (rows as u32, ny as u32), nt, smem, (pw.att.ptr(), pw.q.ptr(), at.kq.ptr(),
                                at.ks.ptr(), at.vq.ptr(), at.vs.ptr(), pw.qf.ptr().add::<f32>(c.head_dim), (2 * qd) as u32,
                                (2 * c.head_dim) as u32, c.n_head as u32, c.n_head_kv as u32, prmp,
                                1.0 / (c.head_dim as f32).sqrt(), pw.sel.ptr(), pw.seln.ptr(), ldsel))?;
                        }
                        prof.mark(s, 21);
                        self.pq(&at.wo, pw.att.ptr(), qd, rows, pw.out.ptr(), d, &mut pw.xa)?;
                        prof.mark(s, 22);
                    }
                }
                prof.mark(s, 2);
                let nh = rows * hd;
                let comb = (pw.out.ptr(), pw.lo.ptr().add::<f32>(c.hc_lr), ld_lo);
                if hc_fuse() {
                    // the combine runs inside the ffn hc norm
                    self.p_hc_mix_c(pw, res, &ly.hcf, rows, &mut prof, Some(comb))?;
                } else {
                    s.launch(&fk.hc_combine, g256(nh), 256, 0, (res, comb.0, comb.1, d as u32, c.hc as u32, ld_lo as u32,
                        nh as u32))?;
                    self.p_hc_mix(pw, res, &ly.hcf, rows, &mut prof)?;
                }
                // ---- MoE: router, shared expert, the experts with a decode slot; the rest after all chunks ----
                prof.mark(s, 3);
                let f = &ly.ffn;
                let ids = pw.ids_n.ptr().add::<i32>(a * topk);
                let wts = pw.wts_n.ptr().add::<f32>(a * topk);
                let inj = pw.inj_n.ptr().add::<f32>(a * c.hc);
                self.pb(&f.gate_inp, pw.mixed.ptr(), d, rows, pw.rl.ptr(), ld_rl, bfx)?;
                s.launch(&fk.route8, rows as u32, 256, 0, (ids, wts, pw.rl.ptr(), ld_rl as u32, c.n_expert as u32,
                    topk as u32, c.expert_weights_scale))?;
                s.launch(&fk.copy2d, (1u32, rows as u32), 256, 0, (inj, c.hc as u32, pw.lo.ptr().add::<f32>(c.hc_lr),
                    ld_lo as u32, c.hc as u32))?;
                if prof.fine { prof.mark(s, 30); }
                let mut ea = q8_view(&pw.ea_n, a, rows);
                self.q.quantize_q8_1(s, pw.mixed.ptr(), d, false, rows, d, &mut ea)?;
                if prof.fine { prof.mark(s, 31); }
                if pw.wg {
                    // experts, shared expert and combine run for the whole group after the chunks
                    s.launch(&fk.copy2d, (1u32, rows as u32), 256, 0, (pw.glog_n.ptr().add::<f32>(a), 1u32,
                        pw.rl.ptr().add::<f32>(c.n_expert), ld_rl as u32, 1u32))?;
                    prof.mark(s, 4);
                    continue;
                }
                self.q.mmq(s, f.shg.ty, pw.sg.ptr(), c.n_ff_shexp, &ea, f.shg.buf.ptr(), f.shg.ty.row_bytes(d), f.shg.n, false)?;
                self.q.mmq(s, f.shu.ty, pw.su.ptr(), c.n_ff_shexp, &ea, f.shu.buf.ptr(), f.shu.ty.row_bytes(d), f.shu.n, false)?;
                let nf = rows * c.n_ff_shexp;
                s.launch(&kn.swiglu, g256(nf), 256, 0, (pw.sg.ptr(), pw.sg.ptr(), pw.su.ptr(), nf as u32))?;
                self.pq(&f.shd, pw.sg.ptr(), c.n_ff_shexp, rows, pw.sh.ptr(), d, &mut pw.sa)?;
                self.q.moe_sort(s, ids, rows, topk, &pw.ws_n[j])?;
                prof.mark(s, 4);
                let np = rows * topk;
                self.p_experts(pw, &ea, j, f, f.gtab.ptr(), f.utab.ptr(), f.dtab.ptr(), np)?;
                s.launch(&fk.moe_acc, (g256(d), rows as u32), 256, 0, (res, pw.ey.ptr(), ids, f.dtab.ptr(), wts,
                    pw.sh.ptr(), pw.rl.ptr().add::<f32>(c.n_expert), ld_rl as u32, inj, c.hc as u32, d as u32,
                    c.hc as u32, topk as u32))?;
            }
            // the experts without a slot used anywhere in the group, streamed once
            prof.mark(s, 5);
            let f = &ly.ffn;
            if pw.wg && pf.is_none() {
                // short prompt: picks and q8 rows to the host, then the resident experts on the card while the CPU takes
                // the slotless experts with few rows (run_mmq_order: the same bits as the card's prompt MMQ); slotless
                // experts with many rows are streamed. Every pair row lands in ybuf / the CPU rows, summed in slot order.
                let np_all = n_all * topk;
                let ff = c.n_ff_exp;
                let tm0 = std::time::Instant::now();
                let mut ids = vec![0i32; np_all];
                s.memcpy_dtoh(&mut ids, pw.ids_n.ptr())?;
                let cpu_rows: usize = std::env::var("FNEXT_CPU_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
                let mut hq8 = Hq8::empty();
                if cpu_rows > 0 {
                    hq8 = Hq8::take(n_all * d, n_all * d / 32);
                    s.memcpy_dtoh(&mut hq8.0, pw.ea_n.q.ptr())?;
                    s.memcpy_dtoh(&mut hq8.1, pw.ea_n.d.ptr())?;
                    s.memcpy_dtoh(&mut hq8.2, pw.ea_n.s.ptr())?;
                }
                s.sync()?;
                self.q.moe_sort(s, pw.ids_n.ptr(), n_all, topk, &pw.ws_g)?;
                let ea_all = q8_view(&pw.ea_n, 0, n_all);
                self.q.mmq_moe_gu(s, f.ty_gu, &ea_all, f.gtab.ptr(), f.utab.ptr(), f.ty_gu.row_bytes(d), ff, &pw.ws_g,
                    np_all, &mut pw.hq)?;
                self.q.mmq_moe(s, f.ty_d, pw.ybuf.ptr(), d, &pw.hq, f.dtab.ptr(), f.ty_d.row_bytes(ff), d, &pw.ws_g, np_all,
                    true, false)?;
                // slotless experts: (expert, pairs)
                let mut by_e: Vec<Vec<usize>> = vec![vec![]; ne];
                for (p, &e) in ids.iter().enumerate() {
                    if self.tabs_h[l][2][e as usize] == 0 {
                        by_e[e as usize].push(p);
                    }
                }
                let cpu_e: Vec<usize> = (0..ne).filter(|&e| !by_e[e].is_empty() && by_e[e].len() <= cpu_rows).collect();
                let stream_e: Vec<usize> = (0..ne).filter(|&e| by_e[e].len() > cpu_rows).collect();
                let mut slot = vec![-1i32; np_all];
                let mut ncpu = 0usize;
                if !cpu_e.is_empty() {
                    let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
                    let ts: Vec<_> = names.iter().map(|nm| self.g.tensor(&format!("blk.{l}.{nm}.weight")).unwrap().clone()).collect();
                    let g = self.g.clone();
                    let jobs: Vec<fe_ggq::cpu::MoeJob> = cpu_e.iter().map(|&e| fe_ggq::cpu::MoeJob {
                        gate: g.slice(&ts[0], e), up: g.slice(&ts[1], e), down: g.slice(&ts[2], e), ty_gu: ts[0].ty,
                        ty_down: ts[2].ty, rows: by_e[e].iter().map(|p| p / topk).collect(),
                    }).collect();
                    let xq = fe_ggq::cpu::avx2::Q8Rows::from_parts(d, n_all, &hq8.0, &hq8.1, &hq8.2);
                    let cm = prompt_cpu(&mut self.pcpu, d, ff);
                    let tc0 = std::time::Instant::now();
                    cm.run_mmq_order(&jobs, &xq)?;
                    if std::env::var("FNEXT_PF_TIME").is_ok() {
                        eprintln!("layer {l}: {} cpu experts ({} pairs), {} streamed; readback+setup {:.2} ms, cpu {:.2} ms",
                            cpu_e.len(), cpu_e.iter().map(|&e| by_e[e].len()).sum::<usize>(), stream_e.len(),
                            (tc0 - tm0).as_secs_f64() * 1e3, tc0.elapsed().as_secs_f64() * 1e3);
                    }
                    let hy = ycpu_host(&mut self.ycpu, np_all * d)?;
                    let cm = self.pcpu.as_ref().unwrap();
                    for (j, &e) in cpu_e.iter().enumerate() {
                        let out = cm.out(j);
                        for (i, &p) in by_e[e].iter().enumerate() {
                            hy[ncpu * d..(ncpu + 1) * d].copy_from_slice(&out[i * d..(i + 1) * d]);
                            slot[p] = ncpu as i32;
                            ncpu += 1;
                        }
                    }
                }
                let groups: Vec<&[usize]> = stream_e.chunks(pw.group).collect();
                self.p_stream(pw, l, f, &groups, &chunks)?;
                // every pair row is in ybuf or the CPU rows now: fixed slot order sum
                if ncpu > 0 {
                    s.memcpy_htod(pw.slot.ptr(), &slot)?;
                    let yd = self.ycpu.as_ref().unwrap().dev();
                    self.q.moe_combine_cpu(s, pw.moey.ptr(), pw.ybuf.ptr(), pw.wts_n.ptr(), n_all, topk, d, false, yd,
                        pw.slot.ptr())?;
                } else {
                    self.q.moe_combine(s, pw.moey.ptr(), pw.ybuf.ptr(), pw.wts_n.ptr(), n_all, topk, d, false)?;
                }
                prof.mark(s, 6);
                for &(a, rows) in chunks.iter() {
                    let ea = q8_view(&pw.ea_n, a, rows);
                    self.q.mmq(s, f.shg.ty, pw.sg.ptr(), c.n_ff_shexp, &ea, f.shg.buf.ptr(), f.shg.ty.row_bytes(d), f.shg.n,
                        false)?;
                    self.q.mmq(s, f.shu.ty, pw.su.ptr(), c.n_ff_shexp, &ea, f.shu.buf.ptr(), f.shu.ty.row_bytes(d), f.shu.n,
                        false)?;
                    let nf = rows * c.n_ff_shexp;
                    s.launch(&kn.swiglu, g256(nf), 256, 0, (pw.sg.ptr(), pw.sg.ptr(), pw.su.ptr(), nf as u32))?;
                    self.pq(&f.shd, pw.sg.ptr(), c.n_ff_shexp, rows, pw.sh.ptr(), d, &mut pw.sa)?;
                    s.launch(&fk.moe_comb_f, (g256(d), rows as u32), 256, 0, (pw.hall.ptr().add::<f32>(a * hd),
                        pw.moey.ptr().add::<f32>(a * d), pw.sh.ptr(), pw.glog_n.ptr().add::<f32>(a),
                        pw.inj_n.ptr().add::<f32>(a * c.hc), c.hc as u32, d as u32, c.hc as u32))?;
                }
                prof.mark(s, 4);
                continue;
            }
            if pw.wg {
                let p = pf.as_ref().unwrap();
                let k = l % 2;
                let got = p.rx.recv().map_err(|_| anyhow!("prefetch thread ended"))?;
                if got != l {
                    return Err(anyhow!("prefetch order: got layer {got} at layer {l}"));
                }
                // one table for the layer: the decode slot of an expert that has one, else its arena copy
                let mut mt = std::mem::take(&mut pf_tabs[k]);
                for kind in 0..3 {
                    for e in 0..ne {
                        let v = self.tabs_h[l][kind][e];
                        if v != 0 {
                            mt[kind * ne + e] = v;
                        }
                    }
                }
                s.memcpy_htod(pw.mtab.ptr(), &mt)?;
                s.wait(&pf_ev[k])?;
                let np_all = n_all * topk;
                let ff = c.n_ff_exp;
                self.q.moe_sort(s, pw.ids_n.ptr(), n_all, topk, &pw.ws_g)?;
                let ea_all = q8_view(&pw.ea_n, 0, n_all);
                let mt0 = pw.mtab.ptr();
                self.q.mmq_moe_gu(s, f.ty_gu, &ea_all, mt0, mt0.add::<u64>(ne), f.ty_gu.row_bytes(d), ff, &pw.ws_g,
                    np_all, &mut pw.hq)?;
                self.q.moe_down_ordered(s, f.ty_d, pw.moey.ptr(), &pw.hq, mt0.add::<u64>(2 * ne), f.ty_d.row_bytes(ff), d,
                    pw.wts_n.ptr(), n_all, topk, &pw.ws_g, MOE_NCOL, pw.ybuf.ptr(), pw.tabw.ptr(), false)?;
                s.record(&use_ev[k])?;
                prof.mark(s, 6);
                for &(a, rows) in chunks.iter() {
                    let ea = q8_view(&pw.ea_n, a, rows);
                    self.q.mmq(s, f.shg.ty, pw.sg.ptr(), c.n_ff_shexp, &ea, f.shg.buf.ptr(), f.shg.ty.row_bytes(d), f.shg.n,
                        false)?;
                    self.q.mmq(s, f.shu.ty, pw.su.ptr(), c.n_ff_shexp, &ea, f.shu.buf.ptr(), f.shu.ty.row_bytes(d), f.shu.n,
                        false)?;
                    let nf = rows * c.n_ff_shexp;
                    s.launch(&kn.swiglu, g256(nf), 256, 0, (pw.sg.ptr(), pw.sg.ptr(), pw.su.ptr(), nf as u32))?;
                    if prof.fine { prof.mark(s, 32); }
                    self.pq(&f.shd, pw.sg.ptr(), c.n_ff_shexp, rows, pw.sh.ptr(), d, &mut pw.sa)?;
                    if prof.fine { prof.mark(s, 33); }
                    s.launch(&fk.moe_comb_f, (g256(d), rows as u32), 256, 0, (pw.hall.ptr().add::<f32>(a * hd),
                        pw.moey.ptr().add::<f32>(a * d), pw.sh.ptr(), pw.glog_n.ptr().add::<f32>(a),
                        pw.inj_n.ptr().add::<f32>(a * c.hc), c.hc as u32, d as u32, c.hc as u32))?;
                    if prof.fine { prof.mark(s, 34); }
                }
                prof.mark(s, 4);
                continue;
            }
            let mut ids = vec![0i32; n_all * topk];
            // stream-ordered: a context-wide sync would also wait for the next layer's prefetch copies
            s.memcpy_dtoh(&mut ids, pw.ids_n.ptr())?;
            s.sync()?;
            let mut used = vec![false; c.n_expert];
            for &e in &ids {
                used[e as usize] = true;
            }
            let missing: Vec<usize> = (0..c.n_expert).filter(|&e| used[e] && self.tabs_h[l][2][e] == 0).collect();
            if let Some(p) = &pf {
                // this layer's slotless experts were copied while the card ran the previous layer
                let k = l % 2;
                let got = p.rx.recv().map_err(|_| anyhow!("prefetch thread ended"))?;
                if got != l {
                    return Err(anyhow!("prefetch order: got layer {got} at layer {l}"));
                }
                debug_assert_eq!(got, l);
                s.memcpy_htod(pw.gtabs[k].ptr(), &pf_tabs[k])?;
                s.wait(&pf_ev[k])?;
                let gt = pw.gtabs[k].ptr();
                let topk = c.n_expert_used;
                if !missing.is_empty() {
                    for (j, &(a, rows)) in chunks.iter().enumerate() {
                        let ea = q8_view(&pw.ea_n, a, rows);
                        self.p_experts(pw, &ea, j, f, gt, gt.add::<u64>(ne), gt.add::<u64>(2 * ne), rows * topk)?;
                        s.launch(&fk.moe_acc, (g256(c.n_embd), rows as u32), 256, 0, (pw.hall.ptr().add::<f32>(a * hd),
                            pw.ey.ptr(), pw.ids_n.ptr().add::<i32>(a * topk), gt.add::<u64>(2 * ne),
                            pw.wts_n.ptr().add::<f32>(a * topk), DevPtr::NULL, DevPtr::NULL, 0u32,
                            pw.inj_n.ptr().add::<f32>(a * c.hc), c.hc as u32, c.n_embd as u32, c.hc as u32, topk as u32))?;
                    }
                }
                s.record(&use_ev[k])?;
            } else {
                let groups: Vec<&[usize]> = missing.chunks(pw.group).collect();
                self.p_stream(pw, l, f, &groups, &chunks)?;
            }
            prof.mark(s, 6);
        }
        if pw.wg {
            let mut st = [0u32];
            s.memcpy_dtoh(&mut st, pw.status.ptr())?;
            s.sync()?;
            if st[0] != 0 {
                eprintln!("fnext-fast: prompt MoE: {} expert terms out of the fixed-point range (dropped)", st[0]);
                s.memset_d8(pw.status.ptr(), 0, 4)?;
            }
        }
        }
        let (s, kn, fk) = (&self.s, &self.kn, &self.fk);
        prof.mark(s, 7);
        // head on the last row (decode path)
        let last = n_all - 1;
        s.memcpy_dtod(self.w.res.ptr(), pw.hall.ptr().add::<f32>(last * hd), hd * 4)?;
        hc_mix(kn, fk, s, &self.w, &self.head, &c, 1)?;
        let w = &mut self.w;
        self.q.quantize_q8_1(s, w.mixed.ptr(), d, false, 1, d, &mut w.xa)?;
        self.output.run(&self.q, s, &w.xa, w.logits.ptr(), false)?;
        super::argmax_rows(fk, s, w, w.logits.ptr(), w.tok.ptr(), 1, c.n_vocab, DevPtr::NULL)?;
        s.sync()?;
        let mut t = [0u32];
        fe_cuda::memcpy_dtoh(&mut t, w.tok.ptr())?;
        if self.want_logits {
            fe_cuda::memcpy_dtoh(&mut self.logit_h.as_mut_slice()[..c.n_vocab], w.logits.ptr())?;
        }
        prof.report();
        self.tokens.extend_from_slice(toks);
        Ok(t[0])
    }

    unsafe fn p_hc_mix(&self, pw: &mut PWork, res: DevPtr, hc: &Hc, rows: usize, prof: &mut Prof) -> Result<()> {
        self.p_hc_mix_c(pw, res, hc, rows, prof, None)
    }

    /// p_hc_mix, with the combine of the previous block (res += out * 2 sigmoid(inj)) fused into its norm when given
    /// as (out, inj, inj stride).
    unsafe fn p_hc_mix_c(&self, pw: &mut PWork, res: DevPtr, hc: &Hc, rows: usize, prof: &mut Prof,
        comb: Option<(DevPtr, DevPtr, usize)>) -> Result<()> {
        let c = &self.c;
        let (kn, fk, s) = (&self.kn, &self.fk, &self.s);
        let ld_lo = (c.hc_lr + c.hc).next_multiple_of(64);
        let fine = prof.fine;
        if fine { prof.mark(s, 29); }
        // fused: the norm writes the bf16 GEMM input and the per-stream 1/rms; the mix recomputes x_norm from res
        let fuse = hc_fuse();
        if fuse {
            if let Some((out, inj, ldi)) = comb {
                s.launch(&fk.hc_comb_norm, (rows * c.hc) as u32, 256, 0, (pw.bfx.ptr(), pw.rinv.ptr(), res, out, inj,
                    ldi as u32, hc.norm.ptr(), c.n_embd as u32, c.hc as u32, c.eps))?;
            } else {
                s.launch(&fk.hc_norm_bf16, (rows * c.hc) as u32, 256, 0, (pw.bfx.ptr(), pw.rinv.ptr(), res, hc.norm.ptr(),
                    c.n_embd as u32, c.hc as u32, c.eps))?;
            }
            if fine { prof.mark(s, 23); }
            let l = &hc.dinj;
            let ty = if l.bf { GgmlType::BF16 } else { GgmlType::F16 };
            self.q.gemm(s, ty, pw.lo.ptr(), ld_lo, pw.bfx.ptr(), l.k, l.w.ptr(), l.k * 2, rows, l.n.next_multiple_of(64), l.k,
                false)?;
        } else {
            kn.rmsnorm(s, pw.xn.ptr(), res, hc.norm.ptr(), c.n_embd, c.hc, rows * c.hc, c.eps)?;
            if fine { prof.mark(s, 23); }
            self.pb(&hc.dinj, pw.xn.ptr(), c.hc_dim(), rows, pw.lo.ptr(), ld_lo, pw.bfx.ptr())?;
        }
        if fine { prof.mark(s, 24); }
        let m = c.hc_lr + hc.n_inj;
        s.launch(&fk.scale_ld, (g256(m), rows as u32), 256, 0, (pw.lo.ptr(), m as u32, ld_lo as u32, 1.0 / c.hc as f32))?;
        s.launch(&fk.silu_ld, (g256(c.hc_lr), rows as u32), 256, 0, (pw.lo.ptr(), c.hc_lr as u32, ld_lo as u32))?;
        if fine { prof.mark(s, 25); }
        self.pb(&hc.up, pw.lo.ptr(), ld_lo, rows, pw.gate.ptr(), c.hc_dim(), pw.bfx.ptr())?;
        if fine { prof.mark(s, 26); }
        let n = rows * c.n_embd;
        if fuse {
            s.launch(&fk.hc_mix_res, g256(n), 256, 0, (pw.mixed.ptr(), res, pw.rinv.ptr(), hc.norm.ptr(), pw.gate.ptr(),
                c.n_embd as u32, c.hc as u32, n as u32))?;
        } else {
            s.launch(&kn.hc_mix, g256(n), 256, 0, (pw.mixed.ptr(), pw.xn.ptr(), pw.gate.ptr(), c.n_embd as u32, c.hc as u32,
                n as u32))?;
        }
        if fine { prof.mark(s, 27); }
        Ok(())
    }

    /// Expert GEMMs for the pairs whose expert has an address in the tables (the other pairs are left untouched).
    #[allow(clippy::too_many_arguments)]
    unsafe fn p_experts(&self, pw: &mut PWork, ea: &Q8Act, j: usize, f: &Ffn, gt: DevPtr, ut: DevPtr, dt: DevPtr, np: usize)
        -> Result<()> {
        let c = &self.c;
        let s = &self.s;
        let d = c.n_embd;
        let ff = c.n_ff_exp;
        let q = &self.q;
        q.mmq_moe(s, f.ty_gu, pw.hg.ptr(), ff, ea, gt, f.ty_gu.row_bytes(d), ff, &pw.ws_n[j], np, false, false)?;
        q.mmq_moe(s, f.ty_gu, pw.hu.ptr(), ff, ea, ut, f.ty_gu.row_bytes(d), ff, &pw.ws_n[j], np, false, false)?;
        // SwiGLU of every pair into hh (hg / hu of earlier groups are kept, so their rows come out the same again)
        let nf = np * ff;
        s.launch(&self.kn.swiglu, g256(nf), 256, 0, (pw.hh.ptr(), pw.hg.ptr(), pw.hu.ptr(), nf as u32))?;
        q.quantize_q8_1(s, pw.hh.ptr(), ff, false, np, ff, &mut pw.ha)?;
        q.mmq_moe(s, f.ty_d, pw.ey.ptr(), d, &pw.ha, dt, f.ty_d.row_bytes(ff), d, &pw.ws_n[j], np, true, false)?;
        Ok(())
    }

    /// Stream the experts of each group to the card (copy stream, two buffers) and run them.
    unsafe fn p_stream(&self, pw: &mut PWork, l: usize, f: &Ffn, groups: &[&[usize]], chunks: &[(usize, usize)])
        -> Result<()> {
        if groups.is_empty() {
            return Ok(());
        }
        let ne = self.c.n_expert;
        let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
        let ts: Vec<_> = names.iter().map(|nm| self.g.tensor(&format!("blk.{l}.{nm}.weight")).unwrap()).collect();
        let sb: Vec<usize> = ts.iter().map(|t| t.slice_bytes().div_ceil(256) * 256).collect();
        let eb = sb.iter().sum::<usize>();
        let g = &self.g;
        let stage_one = |pw: &mut PWork, gi: usize| -> Result<()> {
            let k = gi % 2;
            // the group that used this buffer two steps ago must have run
            pw.use_done[k].sync()?;
            let grp = groups[gi];
            let stg = pw.stage[k].as_mut_slice();
            stg[..grp.len() * eb].par_chunks_mut(eb).zip(grp.par_iter()).for_each(|(dst, &e)| {
                let mut o = 0;
                for (i, t) in ts.iter().enumerate() {
                    let src = g.slice(t, e);
                    dst[o..o + src.len()].copy_from_slice(src);
                    o += sb[i];
                }
            });
            pw.cps.memcpy_htod(pw.arena[k].ptr(), &pw.stage[k].as_slice()[..grp.len() * eb])?;
            let mut tabs = vec![0u64; 3 * ne];
            for (j, &e) in grp.iter().enumerate() {
                let base = pw.arena[k].ptr().byte_add(j * eb).0;
                tabs[e] = base;
                tabs[ne + e] = base + sb[0] as u64;
                tabs[2 * ne + e] = base + (sb[0] + sb[1]) as u64;
            }
            // synchronous for the pageable table: tiny
            pw.cps.memcpy_htod(pw.gtabs[k].ptr(), &tabs)?;
            pw.cps.record(&pw.up_done[k])?;
            Ok(())
        };
        stage_one(pw, 0)?;
        for gi in 0..groups.len() {
            let k = gi % 2;
            self.s.wait(&pw.up_done[k])?;
            let gt = pw.gtabs[k].ptr();
            let c = &self.c;
            let topk = c.n_expert_used;
            let hd = c.hc_dim();
            if pw.wg {
                let n_all = chunks.last().map_or(0, |&(a, r)| a + r);
                let (d, ff) = (c.n_embd, c.n_ff_exp);
                let ea_all = q8_view(&pw.ea_n, 0, n_all);
                self.q.mmq_moe_gu(&self.s, f.ty_gu, &ea_all, gt, gt.add::<u64>(ne), f.ty_gu.row_bytes(d), ff, &pw.ws_g,
                    n_all * topk, &mut pw.hq)?;
                self.q.mmq_moe(&self.s, f.ty_d, pw.ybuf.ptr(), d, &pw.hq, gt.add::<u64>(2 * ne), f.ty_d.row_bytes(ff), d,
                    &pw.ws_g, n_all * topk, true, false)?;
                let _ = hd;
            }
            for (j, &(a, rows)) in chunks.iter().enumerate() {
                if pw.wg {
                    break;
                }
                let ea = q8_view(&pw.ea_n, a, rows);
                self.p_experts(pw, &ea, j, f, gt, gt.add::<u64>(ne), gt.add::<u64>(2 * ne), rows * topk)?;
                self.s.launch(&self.fk.moe_acc, (g256(c.n_embd), rows as u32), 256, 0, (pw.hall.ptr().add::<f32>(a * hd),
                    pw.ey.ptr(), pw.ids_n.ptr().add::<i32>(a * topk), gt.add::<u64>(2 * ne), pw.wts_n.ptr().add::<f32>(a * topk),
                    DevPtr::NULL, DevPtr::NULL, 0u32, pw.inj_n.ptr().add::<f32>(a * c.hc), c.hc as u32, c.n_embd as u32,
                    c.hc as u32, topk as u32))?;
            }
            self.s.record(&pw.use_done[k])?;
            if gi + 1 < groups.len() {
                // staging the next group (host copy + upload) overlaps this group's GEMMs
                stage_one(pw, gi + 1)?;
            }
        }
        Ok(())
    }
}

/// A view of rows a .. a + rows of a q8 activation buffer (no ownership).
unsafe fn q8_view(src: &Q8Act, a: usize, rows: usize) -> Q8Act {
    let k = src.max_k;
    Q8Act {
        q: DeviceBuffer::view(src.q.ptr().add::<i8>(a * k), rows * k),
        d: DeviceBuffer::view(src.d.ptr().add::<f32>(a * k / 32), rows * k / 32),
        s: DeviceBuffer::view(src.s.ptr().add::<f32>(a * k / 32), rows * k / 32),
        max_rows: rows,
        max_k: k,
        rows,
        k,
        sum_mode: src.sum_mode,
    }
}

/// FNEXT_PROFILE=1: wall time per prompt phase (synchronising at every mark).
struct Prof {
    on: bool,
    t: std::time::Instant,
    acc: [f64; 36],
    /// FNEXT_PROFILE=2: also the parts of the hc mixes
    fine: bool,
}
impl Prof {
    fn new() -> Prof {
        Prof { on: std::env::var("FNEXT_PROFILE").is_ok(), t: std::time::Instant::now(), acc: [0.0; 36],
            fine: std::env::var("FNEXT_PROFILE").as_deref() == Ok("2") }
    }
    /// Time since the last mark goes to phase `k`.
    fn mark(&mut self, s: &Stream, k: usize) {
        if !self.on {
            return;
        }
        let _ = s.sync();
        let now = std::time::Instant::now();
        self.acc[k] += (now - self.t).as_secs_f64();
        self.t = now;
    }
    fn report(&self) {
        if !self.on {
            return;
        }
        let names = ["PLE + slot experts", "hc mix (mixer)", "GDN/attention", "combine + hc mix (ffn)", "router+shexp+sort",
            "slot experts (last chunk)", "ids readback + streamed experts", "head", "host prep+embed", "PLE conv", "GDN",
            "attention", "slot experts (prev chunks)", "PLE host gather", "gdn qkv+z", "gdn ab", "gdn conv", "gdn chunk",
            "gdn gnorm", "attn proj qkv+ik", "attn idx q+score+topk", "attn prep+core", "attn wo", "hc norm", "hc down (bf16+gemm)", "hc small",
            "hc up (bf16+gemm)", "hc mix kernel", "-", "hc combine (ffn side)", "router gemm+route",
            "q8 of x", "shexp gate/up", "shexp down", "moe comb"];
        let tot: f64 = self.acc.iter().sum();
        let mut v: Vec<String> = (0..35).map(|i| format!("{} {:.2}s", names[i], self.acc[i])).collect();
        v.push(format!("total {tot:.2}s"));
        eprintln!("prefill phases: {}", v.join(" | "));
    }
}

/// One layer's slotless experts copied to an arena on the prefetch thread.
struct PfJob {
    layer: usize,
    experts: Vec<usize>,
    base: u64,
    /// the arena's previous user must have finished on the card
    wait: Arc<Event>,
    /// recorded on the copy stream after the last copy
    done: Arc<Event>,
}

/// Prefetch thread for long prompts: while the card runs layer l, layer l + 1's experts without a decode slot are
/// copied into the other arena (pinned staging, its own stream), so the bus works under the dense compute.
pub(crate) struct Prefetch {
    tx: Option<std::sync::mpsc::Sender<PfJob>>,
    rx: std::sync::mpsc::Receiver<usize>,
    h: Option<std::thread::JoinHandle<()>>,
}

impl Prefetch {
    fn start(g: Arc<Gguf>, cu: Arc<fe_cuda::Context>) -> Prefetch {
        let (tx, jrx) = std::sync::mpsc::channel::<PfJob>();
        let (dtx, rx) = std::sync::mpsc::channel::<usize>();
        let h = std::thread::spawn(move || {
            if std::env::var("FNEXT_NO_PIN").is_err() {
                fe_ggq::cpu::pool::pin_current(crate::cores::plan().up);
            }
            cu.bind().unwrap();
            let st = Stream::new().unwrap();
            const STAGE: usize = 48 << 20;
            let mut stage: Vec<HostBuffer<u8>> = (0..2).map(|_| stage_take(STAGE).unwrap()).collect();
            let ev: Vec<Event> = (0..2).map(|_| Event::new_blocking().unwrap()).collect();
            let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
            let poll = |e: &Event| while !e.is_done().unwrap_or(true) {
                std::thread::sleep(std::time::Duration::from_micros(50));
            };
            // staging state carries across jobs: the last copies of one layer may still read a stage buffer
            let mut k = 0;
            let mut used = [false; 2];
            let dbg = std::env::var("FNEXT_PF_TIME").is_ok();
            while let Ok(job) = jrx.recv() {
                let t0 = std::time::Instant::now();
                poll(&job.wait);
                let (mut t_stage, mut t_gather) = (0f64, 0f64);
                let t_arena = t0.elapsed().as_secs_f64();
                let ts: Vec<_> = names.iter().map(|nm| g.tensor(&format!("blk.{}.{nm}.weight", job.layer)).unwrap().clone()).collect();
                let sb: Vec<usize> = ts.iter().map(|t| t.slice_bytes().div_ceil(256) * 256).collect();
                let eb: usize = sb.iter().sum();
                let per = (STAGE / eb).max(1);
                for (bi, batch) in job.experts.chunks(per).enumerate() {
                    let ta = std::time::Instant::now();
                    if used[k] {
                        poll(&ev[k]);
                    }
                    let tb = std::time::Instant::now();
                    t_stage += (tb - ta).as_secs_f64();
                    let sbuf = stage[k].as_mut_slice();
                    sbuf[..batch.len() * eb].par_chunks_mut(eb).zip(batch.par_iter()).for_each(|(dst, &e)| {
                        let mut o = 0;
                        for (i, t) in ts.iter().enumerate() {
                            let src = g.slice(t, e);
                            dst[o..o + src.len()].copy_from_slice(src);
                            o += sb[i];
                        }
                    });
                    t_gather += tb.elapsed().as_secs_f64();
                    let dst = DevPtr(job.base).byte_add(bi * per * eb);
                    unsafe { st.memcpy_htod(dst, &stage[k].as_slice()[..batch.len() * eb]).unwrap() };
                    st.record(&ev[k]).unwrap();
                    used[k] = true;
                    k = 1 - k;
                }
                st.record(&job.done).unwrap();
                if dbg {
                    eprintln!("pf layer {}: wait arena {:.0} ms, wait stage {:.0} ms, gather {:.0} ms ({} experts), total {:.0} ms",
                        job.layer, t_arena * 1e3, t_stage * 1e3, t_gather * 1e3, job.experts.len(), t0.elapsed().as_secs_f64() * 1e3);
                }
                if dtx.send(job.layer).is_err() {
                    break;
                }
            }
            // the copies out of the stages must be done before they go back to the pool
            for (i, e) in ev.iter().enumerate() {
                if used[i] {
                    poll(e);
                }
            }
            stage_put(std::mem::take(&mut stage));
        });
        Prefetch { tx: Some(tx), rx, h: Some(h) }
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(h) = self.h.take() {
            let _ = h.join();
        }
    }
}

/// A Q8Act view of `src`'s buffers as `rows` x `k` (rows * k must fit in src).
unsafe fn q8_shape(src: &Q8Act, rows: usize, k: usize) -> Q8Act {
    assert!(rows * k <= src.max_rows * src.max_k && k % 32 == 0);
    Q8Act {
        q: DeviceBuffer::view(src.q.ptr(), rows * k),
        d: DeviceBuffer::view(src.d.ptr(), rows * k / 32),
        s: DeviceBuffer::view(src.s.ptr(), rows * k / 32),
        max_rows: rows,
        max_k: k,
        rows,
        k,
        sum_mode: src.sum_mode,
    }
}

impl Fast {
    /// The MTP block's KV / indexer caches for a prompt group whose trunk just ran (`pw` still holds its rows, rope
    /// tables and step parameters): row q reads token q and the trunk residual of q - 1 (the MTP `hprev` for the
    /// first row). Only the cache writes of its attention run: the block's output, MoE and head are not needed for
    /// prompt rows (the first draft comes from a 1-row pass afterwards).
    pub(crate) unsafe fn mtp_group_kv(&mut self, toks: &[u32], pos0: usize) -> Result<()> {
        let mut pw = self.pw.take().ok_or_else(|| anyhow!("no prompt workspace"))?;
        let r = self.mtp_group_inner(&mut pw, toks, pos0);
        self.pw = Some(pw);
        r
    }

    unsafe fn mtp_group_inner(&mut self, pw: &mut PWork, toks: &[u32], pos0: usize) -> Result<()> {
        let c = self.cfull.clone();
        let n_all = toks.len();
        let _ = pos0;
        let d = c.n_embd;
        let hd = c.hc_dim();
        let kp = c.kpool;
        let cm = pw.cmax;
        let chunks: Vec<(usize, usize)> = (0..n_all).step_by(cm).map(|a| (a, (n_all - a).min(cm))).collect();
        let pstride = (cm / kp + 2) * 32;
        let ld_lo = (c.hc_lr + c.hc).next_multiple_of(64);
        let (kn, fk, s) = (&self.kn, &self.fk, &self.s);
        let m = self.mtp.as_ref().unwrap();
        let at = &m.at;
        let te = self.g.tensor("token_embd.weight").unwrap();
        let (trb, tdata) = (te.row_bytes(), self.g.data(te));
        let qd = c.n_head * c.head_dim;
        let kvd = c.n_head_kv * c.head_dim;
        let dd = c.idx_dim;
        let bfx = pw.bfx.ptr();
        for (j, &(a, rows)) in chunks.iter().enumerate() {
            let prmp = pw.prm_n.ptr().add::<u32>(j * P_WORDS);
            let (csp, snp) = (pw.cs_n.ptr().add::<f32>(a * 32), pw.sn_n.ptr().add::<f32>(a * 32));
            let (pcsp, psnp) = (pw.pcs_n.ptr().add::<f32>(j * pstride), pw.psn_n.ptr().add::<f32>(j * pstride));
            // token embeddings (pageable source: staged before the call returns)
            let mut emb = vec![0f32; rows * d];
            emb.par_chunks_mut(d).zip(toks[a..a + rows].par_iter()).for_each(|(o, &t)| {
                fe_ggq::refq::dequant(te.ty, &tdata[t as usize * trb..(t as usize + 1) * trb], o);
            });
            s.memcpy_htod(pw.emb.ptr(), &emb)?;
            // h rows: the trunk residual of the previous position
            let hin = if a == 0 {
                s.memcpy_dtod(pw.gate.ptr(), m.hprev.ptr(), hd * 4)?;
                if rows > 1 {
                    s.memcpy_dtod(pw.gate.ptr().add::<f32>(hd), pw.hall.ptr(), (rows - 1) * hd * 4)?;
                }
                pw.gate.ptr()
            } else {
                pw.hall.ptr().add::<f32>((a - 1) * hd)
            };
            kn.rmsnorm(s, pw.xn.ptr(), hin, m.hnorm.ptr(), d, c.hc, rows * c.hc, c.eps)?;
            kn.rmsnorm(s, pw.out.ptr(), pw.emb.ptr(), m.enorm.ptr(), d, 1, rows, c.eps)?;
            // eh_proj over [enorm(e) | hnorm(h)_s] = W_e . en (shared by the streams) + W_h . hn_s
            let rbe = m.eh.ty.row_bytes(2 * d);
            self.q.quantize_q8_1(s, pw.out.ptr(), d, false, rows, d, &mut pw.ea)?;
            self.q.mmq(s, m.eh.ty, pw.mixed.ptr(), d, &pw.ea, m.eh.buf.ptr(), rbe, d, false)?;
            let mut xh = q8_shape(&pw.xa, rows * c.hc, d);
            self.q.quantize_q8_1(s, pw.xn.ptr(), d, false, rows * c.hc, d, &mut xh)?;
            self.q.mmq(s, m.eh.ty, pw.res.ptr(), d, &xh, m.eh.buf.ptr().byte_add(m.eh.ty.row_bytes(d)), rbe, d, false)?;
            let n = rows * d;
            s.launch(&fk.bcast_add, g256(n), 256, 0, (pw.res.ptr(), pw.mixed.ptr(), d as u32, c.hc as u32, n as u32))?;
            // mixer hc mix (quantized hc weights; up through the 16-bit copy)
            let hc = &m.hca;
            kn.rmsnorm(s, pw.xn.ptr(), pw.res.ptr(), hc.norm.ptr(), d, c.hc, rows * c.hc, c.eps)?;
            self.q.quantize_q8_1(s, pw.xn.ptr(), hd, false, rows, hd, &mut pw.xa)?;
            self.q.mmq(s, hc.down.ty, pw.lo.ptr(), ld_lo, &pw.xa, hc.down.buf.ptr(), hc.down.ty.row_bytes(hd), c.hc_lr, false)?;
            s.launch(&fk.scale_ld, (g256(c.hc_lr), rows as u32), 256, 0, (pw.lo.ptr(), c.hc_lr as u32, ld_lo as u32,
                1.0 / c.hc as f32))?;
            s.launch(&fk.silu_ld, (g256(c.hc_lr), rows as u32), 256, 0, (pw.lo.ptr(), c.hc_lr as u32, ld_lo as u32))?;
            self.pb(&m.hca_up, pw.lo.ptr(), ld_lo, rows, pw.gate.ptr(), hd, bfx)?;
            s.launch(&kn.hc_mix, g256(n), 256, 0, (pw.mixed.ptr(), pw.xn.ptr(), pw.gate.ptr(), d as u32, c.hc as u32,
                n as u32))?;
            // attention: only the K / V cache and the indexer keys
            self.pq(&at.wq, pw.mixed.ptr(), d, rows, pw.qf.ptr(), 2 * qd, &mut pw.xa)?;
            self.pq(&at.wk, pw.mixed.ptr(), d, rows, pw.kx.ptr(), kvd, &mut pw.xa)?;
            self.pq(&at.wv, pw.mixed.ptr(), d, rows, pw.vx.ptr(), kvd, &mut pw.xa)?;
            self.pb(&at.ik, pw.mixed.ptr(), d, rows, pw.ikx.ptr(), dd, bfx)?;
            let rn = (at.ring.len() / dd) as u32;
            s.launch(&fk.idx_kstore, rows as u32, dd as u32, 0, (at.ring.ptr(), pw.ikx.ptr(), prmp, rn))?;
            s.launch(&fk.idx_pool, (rows / kp + 1) as u32, dd as u32, 0, (at.pooled.ptr(), at.ring.ptr(), at.ikn.ptr(), prmp,
                kp as u32, c.n_rot as u32, c.eps, rn, pcsp, psnp))?;
            s.launch(&fk.attn_prep, (rows as u32, (c.n_head + c.n_head_kv) as u32), 256, 0,
                (pw.q.ptr(), at.kq.ptr(), at.ks.ptr(), at.vq.ptr(), at.vs.ptr(), pw.qf.ptr(), (2 * qd) as u32,
                    (2 * c.head_dim) as u32, pw.kx.ptr(), pw.vx.ptr(), kvd as u32, at.qn.ptr(), at.kn.ptr(), prmp,
                    c.n_head as u32, c.n_head_kv as u32, c.n_rot as u32, c.eps, csp, snp))?;
        }
        Ok(())
    }
}

/// FNEXT_HC_FUSE=0: the unfused prompt hc path (separate norm, f32 x_norm, separate combine).
fn hc_fuse() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FNEXT_HC_FUSE").map_or(true, |v| v != "0"))
}

/// Column block of moe_down_ordered (pair rows of this width live at once).
const MOE_NCOL: usize = 320;

/// The CPU expert pool for prompt rows (created on first use, on the core plan's pool CPUs).
fn prompt_cpu(pc: &mut Option<fe_ggq::cpu::CpuMoe>, d: usize, ff: usize) -> &mut fe_ggq::cpu::CpuMoe {
    if pc.is_none() {
        let cores = crate::cores::plan().pool.clone();
        *pc = Some(fe_ggq::cpu::CpuMoe::pinned(&cores, std::time::Duration::from_millis(2), d, ff));
    }
    pc.as_mut().unwrap()
}

/// Pinned host rows for CPU-computed pair outputs (read by moe_combine_cpu through the mapping), grown on demand.
fn ycpu_host(y: &mut Option<HostBuffer<f32>>, n: usize) -> Result<&mut [f32]> {
    if y.as_ref().is_none_or(|b| b.len() < n) {
        *y = Some(HostBuffer::alloc(n)?);
    }
    Ok(&mut y.as_mut().unwrap().as_mut_slice()[..n])
}

/// FNEXT_TP=2 prompt all-reduce events of one card (x 2 slots): partial ready, copied out (DtoH done), the other
/// card's partial copied in (HtoD done: its host slot may be reused).
struct TpEv {
    ready: Vec<Event>,
    dh: Vec<Event>,
    /// per piece of a slot's copy-out (FNEXT_AR_PIECES): the other card copies a piece in as soon as it is out
    dhp: Vec<Vec<Event>>,
    cons: Vec<Event>,
    /// the sum read the copied-in partial (its buffer may be refilled)
    sum: Vec<Event>,
}

impl TpEv {
    fn new(slots: usize) -> Result<TpEv> {
        let mk = || -> Result<Vec<Event>> { (0..slots).map(|_| Event::new_no_timing().map_err(Into::into)).collect() };
        let dhp = (0..slots).map(|_| (0..ar_pieces()).map(|_| Event::new_no_timing().map_err(Into::into)).collect::<Result<Vec<_>>>())
            .collect::<Result<Vec<_>>>()?;
        Ok(TpEv { ready: mk()?, dh: mk()?, dhp, cons: mk()?, sum: mk()? })
    }
}

impl Fast {
    /// Make card `r` the active one (FNEXT_TP=2).
    fn tp_on(&mut self, r: usize) -> Result<()> {
        if self.tp_rank != r {
            self.swap_dev()?;
        }
        Ok(())
    }

    /// FNEXT_TP=2 prompt group: both cards run every layer on their half (the decode split), layer-major over
    /// chunks as group_inner; each chunk's mixer and FFN outputs are all-reduced through pinned host memory (copy
    /// engine DtoH on the side stream, the other card's partial read zero-copy by fx_ar_sum), chunk j's copies
    /// overlapping chunk j+1's work. Experts: each card its owned ones (the other pairs weigh 0 in its sums).
    pub(crate) fn prefill_group_tp(&mut self, toks: &[u32]) -> Result<u32> {
        let n = self.tokens.len() + toks.len();
        self.ensure_ctx(n + MAX_ROWS)?;
        self.tp_on(1)?;
        let mut pw1 = self.pw.take().ok_or_else(|| anyhow!("no prompt workspace (FNEXT_CHUNK=0)"))?;
        let Some(mut pw0) = self.other.as_mut().unwrap().pw.take() else {
            self.pw = Some(pw1);
            bail!("no prompt workspace on card 0");
        };
        let r = unsafe { self.group_tp(&mut pw0, &mut pw1, toks) };
        let r2 = self.tp_on(1);
        self.pw = Some(pw1);
        self.other.as_mut().unwrap().pw = Some(pw0);
        r2?;
        r
    }

    unsafe fn group_tp(&mut self, pw0: &mut PWork, pw1: &mut PWork, toks: &[u32]) -> Result<u32> {
        let c = self.c.clone();
        let n_all = toks.len();
        let pws: [&mut PWork; 2] = [pw0, pw1];
        let (cm, gmax, pctx) = (pws[1].cmax, pws[1].gmax, pws[1].ctx);
        if n_all == 0 || n_all > gmax || pws[0].cmax != cm || pws[0].gmax != gmax {
            bail!("prefill_group_tp: {n_all} rows (max {gmax})");
        }
        let pos0 = self.tokens.len();
        if pos0 + n_all > self.ctx || pos0 + n_all > pctx {
            bail!("context full / prompt workspace too small");
        }
        let d = c.n_embd;
        let hd = c.hc_dim();
        let topk = c.n_expert_used;
        let kp = c.kpool;
        let ne = c.n_expert;
        let chunks: Vec<(usize, usize)> = (0..n_all).step_by(cm).map(|a| (a, (n_all - a).min(cm))).collect();
        let nch = chunks.len();
        let pstride = (cm / kp + 2) * 32;
        let ld_lo = (c.hc_lr + c.hc).next_multiple_of(64);
        let ld_ab = (2 * c.ssm_vh).next_multiple_of(64);
        let ld_rl = (c.n_expert + 1).next_multiple_of(64);
        let mut prof = Prof::new();
        // pinned staging [card][slot][cm * d]
        let nstage = 4 + 2 * nch;
        if self.pa_h.as_ref().is_none_or(|b| b.len() < nstage * cm * d) {
            self.pa_h = Some(HostBuffer::alloc(nstage * cm * d)?);
        }
        let pa = self.pa_h.as_ref().unwrap().as_ptr() as usize;
        let host = |r: usize, k: usize| -> usize { pa + ((r * 2 + k) * cm * d) * 4 };
        let hostf = |r: usize, j: usize| -> usize { pa + ((4 + r * nch + j) * cm * d) * 4 };
        let own = self.own_h.clone();
        // events and per-card setup: step parameters, rope tables, the residual streams from the embeddings
        let mut evs: Vec<TpEv> = vec![];
        // the FFN-side all-reduce: one slot per chunk; chunk j's sum + combine run just before the next layer's mixer
        // of chunk j (cross-layer overlap: its copies hide under the other chunks' mixers); partials in place in moey,
        // the received ones in ybuf (free between the MoE passes) once this card's MoE is done with it
        let mut evf: Vec<TpEv> = vec![];
        let mut moe_done: Vec<Event> = vec![];
        let xl = std::env::var("FNEXT_TP_XL").as_deref() != Ok("0");
        let gg = self.g.clone();
        let te = gg.tensor("token_embd.weight").unwrap();
        let (trb, tdata) = (te.row_bytes(), gg.data(te));
        for r in 0..2 {
            self.tp_on(r)?;
            evs.push(TpEv::new(2)?);
            evf.push(TpEv::new(nch)?);
            moe_done.push(Event::new_no_timing()?);
            self.s.sync()?;
            self.group_params(pws[r], pos0, n_all, &chunks)?;
            for &(a, rows) in &chunks {
                let mut emb = vec![0f32; rows * d];
                emb.par_chunks_mut(d).zip(toks[a..a + rows].par_iter()).for_each(|(o, &t)| {
                    fe_ggq::refq::dequant(te.ty, &tdata[t as usize * trb..(t as usize + 1) * trb], o);
                });
                fe_cuda::memcpy_htod(pws[r].emb.ptr(), &emb)?;
                let n = rows * d;
                self.s.launch(&self.kn.hc_init, g256(n), 256, 0, (pws[r].hall.ptr().add::<f32>(a * hd), pws[r].emb.ptr(),
                    d as u32, c.hc as u32, n as u32))?;
            }
        }
        // PLE rows of the whole group on a host thread
        let idx_all = crate::engine::ple_rows(&c, &self.tokens, pos0, toks);
        let ple_job = {
            let g = self.g.clone();
            let idx = idx_all.clone();
            let page = self.hm.page();
            std::thread::spawn(move || {
                let tq = std::time::Instant::now();
                let pt = g.tensor("per_layer_token_embd.weight").unwrap();
                let ple = hostmem::ple_gather(&g, pt, &idx, page);
                if std::env::var("FNEXT_PF_TIME").is_ok() {
                    eprintln!("PLE gather: {} rows in {:.0} ms", idx.len(), tq.elapsed().as_secs_f64() * 1e3);
                }
                ple
            })
        };
        let mut ple_job = Some(ple_job);
        let mut ple_all: Vec<f32> = vec![];
        let idx_kind = match std::env::var("FNEXT_IDX").as_deref() { Ok("simt") => 0, Ok("tc3") => 2, _ => 1 };
        let gdn_kind = match std::env::var("FNEXT_GDN").as_deref() { Ok("simt") => 0, Ok("tc1") => 2, _ => 1 };
        // long groups: each card prefetches its own experts without a decode slot, one layer ahead
        let pf_on = pws[1].group >= ne;
        let mut pfs: Vec<Option<Prefetch>> = vec![];
        let pf_ev: Vec<[Arc<Event>; 2]> = (0..2).map(|r| -> Result<[Arc<Event>; 2]> {
            self.tp_on(r)?;
            Ok([Arc::new(Event::new_no_timing()?), Arc::new(Event::new_no_timing()?)])
        }).collect::<Result<_>>()?;
        let use_ev: Vec<[Arc<Event>; 2]> = (0..2).map(|r| -> Result<[Arc<Event>; 2]> {
            self.tp_on(r)?;
            Ok([Arc::new(Event::new_no_timing()?), Arc::new(Event::new_no_timing()?)])
        }).collect::<Result<_>>()?;
        let mut pf_tabs: Vec<[Vec<u64>; 2]> = vec![[vec![], vec![]], [vec![], vec![]]];
        // this card's experts of layer l without a decode slot, laid out in an arena at `base`: (experts, tables)
        let slot_tabs = |fast: &Fast, r: usize, l: usize, base: u64| -> (Vec<usize>, Vec<u64>) {
            let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
            let sb: Vec<usize> = names.iter().map(|nm| fast.g.tensor(&format!("blk.{l}.{nm}.weight")).unwrap().slice_bytes().div_ceil(256) * 256).collect();
            let eb: usize = sb.iter().sum();
            let ex: Vec<usize> = (0..ne).filter(|&e| own[l][e] as usize == r && fast.tabs_h[l][2][e] == 0).collect();
            let mut tabs = vec![0u64; 3 * ne];
            for (j, &e) in ex.iter().enumerate() {
                let b = base + (j * eb) as u64;
                tabs[e] = b;
                tabs[ne + e] = b + sb[0] as u64;
                tabs[2 * ne + e] = b + (sb[0] + sb[1]) as u64;
            }
            (ex, tabs)
        };
        for r in 0..2 {
            self.tp_on(r)?;
            let pf = if pf_on { Some(Prefetch::start(self.g.clone(), self.cu.clone())) } else { None };
            if let Some(p) = &pf {
                let (ex, tabs) = slot_tabs(self, r, 0, pws[r].arena[0].ptr().0);
                pf_tabs[r][0] = tabs;
                p.tx.as_ref().unwrap().send(PfJob { layer: 0, experts: ex, base: pws[r].arena[0].ptr().0,
                    wait: use_ev[r][0].clone(), done: pf_ev[r][0].clone() }).ok();
            }
            pfs.push(pf);
        }
        let bfx = |pw: &PWork| pw.bfx.ptr();
        for l in 0..c.n_layer {
            if self.cancelled() {
                // the client went away: stop between layers on both cards (the caller restores the lane)
                drop(pfs);
                return Err(self.stop_cancelled());
            }
            // ---- next layer's prefetch ----
            for r in 0..2 {
                if let Some(p) = &pfs[r] {
                    if l + 1 < c.n_layer {
                        self.tp_on(r)?;
                        let k = (l + 1) % 2;
                        let (ex, tabs) = slot_tabs(self, r, l + 1, pws[r].arena[k].ptr().0);
                        pf_tabs[r][k] = tabs;
                        p.tx.as_ref().unwrap().send(PfJob { layer: l + 1, experts: ex, base: pws[r].arena[k].ptr().0,
                            wait: use_ev[r][k].clone(), done: pf_ev[r][k].clone() }).ok();
                    }
                }
            }
            // ---- mixers, chunk by chunk on both cards; chunk j's all-reduce overlaps chunk j + 1 ----
            for j in 0..=nch {
                if j < nch {
                    let (a, rows) = chunks[j];
                    let k = j % 2;
                    for r in 0..2 {
                        self.tp_on(r)?;
                        let pw = &mut *pws[r];
                        let ly = &self.layers[l];
                        let (s, kn, fk) = (&self.s, &self.kn, &self.fk);
                        let res = pw.hall.ptr().add::<f32>(a * hd);
                        let prmp = pw.prm_n.ptr().add::<u32>(j * P_WORDS);
                        let (csp, snp) = (pw.cs_n.ptr().add::<f32>(a * 32), pw.sn_n.ptr().add::<f32>(a * 32));
                        let (pcsp, psnp) = (pw.pcs_n.ptr().add::<f32>(j * pstride), pw.psn_n.ptr().add::<f32>(j * pstride));
                        if xl && l > 0 {
                            // layer l - 1's FFN all-reduce of chunk j, its combine into the residual rows
                            let fo = pw.moey.ptr().add::<f32>(a * d);
                            let rv = pw.ybuf.ptr().add::<f32>(k * cm * d);
                            self.tp_sum_rv(pw, &evf, r, j, fo, rows * d, hostf(1 - r, j), rv, Some(&moe_done[r]))?;
                            let nh = rows * hd;
                            s.launch(&fk.hc_combine, g256(nh), 256, 0, (res, fo, pw.inj_n.ptr().add::<f32>(a * c.hc), d as u32,
                                c.hc as u32, c.hc as u32, nh as u32))?;
                        }
                        // the slot's previous copy-out must be done before its buffer is written again
                        s.wait(&evs[r].dh[k])?;
                        let outp = pw.tpo[k].ptr();
                        if let Some(p) = &ly.ple {
                            if let Some(h) = ple_job.take() {
                                ple_all = h.join().map_err(|_| anyhow!("PLE gather thread panicked"))?;
                                let pt = self.g.tensor("per_layer_token_embd.weight").unwrap();
                                if hostmem::ple_mmap() {
                                    for &rw in &idx_all {
                                        self.hm.drop_range(&self.g, pt, rw as usize * pt.row_bytes(), pt.row_bytes());
                                    }
                                }
                            }
                            let ple = &ple_all[a * d..(a + rows) * d];
                            s.memcpy_htod(pw.pemb.ptr(), ple)?;
                            self.pb(&p.key, pw.pemb.ptr(), d, rows, pw.qkv.ptr(), hd, bfx(pw))?;
                            self.pb(&p.value, pw.pemb.ptr(), d, rows, pw.pval.ptr(), d, bfx(pw))?;
                            kn.rmsnorm(s, pw.conv.ptr(), pw.qkv.ptr(), p.nk.ptr(), d, c.hc, rows * c.hc, c.eps)?;
                            kn.rmsnorm(s, pw.gate.ptr(), res, p.nq.ptr(), d, c.hc, rows * c.hc, c.eps)?;
                            s.launch(&kn.ple_gate, (rows * c.hc) as u32, 256, 0,
                                (pw.xn.ptr(), pw.conv.ptr(), pw.gate.ptr(), pw.pval.ptr(), d as u32, c.hc as u32))?;
                            kn.rmsnorm(s, pw.ey.ptr(), pw.xn.ptr(), p.nc.ptr(), d, c.hc, rows * c.hc, c.eps)?;
                            s.launch(&kn.ple_conv, g256(hd), 256, 0, (res, pw.xn.ptr(), pw.ey.ptr(), p.conv_w.ptr(),
                                p.hist.ptr(), rows as u32, hd as u32, c.ple_ngram as u32, DevPtr::NULL, 0u32))?;
                        }
                        self.p_hc_mix(pw, res, &ly.hca, rows, &mut prof)?;
                        // its inject logits wait for the combine after the all-reduce
                        s.launch(&fk.copy2d, (1u32, rows as u32), 256, 0, (pw.inja_n.ptr().add::<f32>(a * c.hc), c.hc as u32,
                            pw.lo.ptr().add::<f32>(c.hc_lr), ld_lo as u32, c.hc as u32))?;
                        match &ly.mix {
                            Mixer::Gdn(gd) => {
                                let cd = c.conv_dim();
                                self.pq(&gd.qkv, pw.mixed.ptr(), d, rows, pw.qkv.ptr(), cd, &mut pw.xa)?;
                                self.pq(&gd.z, pw.mixed.ptr(), d, rows, pw.z.ptr(), c.ssm_inner, &mut pw.xa)?;
                                self.pb(&gd.ab, pw.mixed.ptr(), d, rows, pw.ab.ptr(), ld_ab, bfx(pw))?;
                                s.launch(&fk.conv_p, (g256(cd), rows.div_ceil(16) as u32), 256, 0, (pw.conv.ptr(), pw.qkv.ptr(),
                                    cd as u32, gd.conv_w.ptr(), gd.conv_state.ptr(), rows as u32, cd as u32))?;
                                s.launch(&fk.conv_state, g256(cd), 256, 0, (gd.conv_state.ptr(), pw.qkv.ptr(), cd as u32,
                                    rows as u32, cd as u32))?;
                                let args = (pw.o.ptr(), pw.conv.ptr(), pw.ab.ptr(), ld_ab as u32, gd.a.ptr(), gd.dt.ptr(),
                                    gd.ssm.ptr(), rows as u32);
                                // the prompt GDN kernels are built for a head count: the half-head instance here
                                let _ = gdn_kind;
                                s.launch(&fk.gdn_tc_h, c.ssm_vh as u32, 256, super::GDN_TC_SMEM, args)?;
                                s.launch(&kn.gdn_gnorm, (rows as u32, c.ssm_vh.div_ceil(8) as u32), 256, 0, (pw.y.ptr(), pw.o.ptr(),
                                    pw.z.ptr(), c.ssm_inner as u32, gd.norm.ptr(), c.ssm_vh as u32, c.eps))?;
                                self.pq(&gd.out, pw.y.ptr(), c.ssm_inner, rows, outp, d, &mut pw.xa)?;
                            }
                            Mixer::Attn(at) => {
                                let qd = c.n_head * c.head_dim;
                                let kvd = c.n_head_kv * c.head_dim;
                                let dd = c.idx_dim;
                                self.pq(&at.wq, pw.mixed.ptr(), d, rows, pw.qf.ptr(), 2 * qd, &mut pw.xa)?;
                                self.pq(&at.wk, pw.mixed.ptr(), d, rows, pw.kx.ptr(), kvd, &mut pw.xa)?;
                                self.pq(&at.wv, pw.mixed.ptr(), d, rows, pw.vx.ptr(), kvd, &mut pw.xa)?;
                                self.pb(&at.ik, pw.mixed.ptr(), d, rows, pw.ikx.ptr(), dd, bfx(pw))?;
                                let rn = (at.ring.len() / dd) as u32;
                                s.launch(&fk.idx_kstore, rows as u32, dd as u32, 0, (at.ring.ptr(), pw.ikx.ptr(), prmp, rn))?;
                                s.launch(&fk.idx_pool, (rows / kp + 1) as u32, dd as u32, 0, (at.pooled.ptr(), at.ring.ptr(),
                                    at.ikn.ptr(), prmp, kp as u32, c.n_rot as u32, c.eps, rn, pcsp, psnp))?;
                                self.pb(&at.iq, pw.mixed.ptr(), d, rows, pw.iqv.ptr(), dd * c.idx_heads, bfx(pw))?;
                                kn.rmsnorm(s, pw.iqv.ptr(), pw.iqv.ptr(), at.iqn.ptr(), dd, 1, rows * c.idx_heads, c.eps)?;
                                s.launch(&kn.rope_rows, (rows as u32, c.idx_heads as u32), dd as u32, 0,
                                    (pw.iqv.ptr(), csp, snp, c.idx_heads as u32, c.n_rot as u32))?;
                                let top = (c.idx_top_k / kp) as u32;
                                let ldsc = (pw.ctx / kp + 1) as u32;
                                let ldsel = (c.idx_top_k + kp - 1) as u32;
                                if idx_kind == 0 || c.idx_heads != 4 {
                                    s.launch(&fk.idx_score, (rows as u32, 72), 256, 0, (pw.score.ptr(), pw.iqv.ptr(), at.pooled.ptr(),
                                        c.idx_heads as u32, prmp, kp as u32, ldsc, top, 1.0 / (dd as f32).sqrt()))?;
                                } else {
                                    let tiles = rows.div_ceil(16);
                                    let splits = (144 / tiles).max(1) as u32;
                                    let f = if idx_kind == 2 { &fk.idx_tc3 } else { &fk.idx_tc };
                                    s.launch(f, (tiles as u32, splits), 128, super::IDX_TC_SMEM, (pw.score.ptr(), pw.iqv.ptr(),
                                        at.pooled.ptr(), prmp, kp as u32, ldsc, top, 1.0 / (dd as f32).sqrt(), rows as u32))?;
                                }
                                s.launch(&fk.topk, rows as u32, 1024, 0, (pw.sel.ptr(), pw.seln.ptr(), pw.score.ptr(), prmp,
                                    kp as u32, top, ldsc, ldsel))?;
                                s.launch(&fk.attn_prep, (rows as u32, (c.n_head + c.n_head_kv) as u32), 256, 0,
                                    (pw.q.ptr(), at.kq.ptr(), at.ks.ptr(), at.vq.ptr(), at.vs.ptr(), pw.qf.ptr(), (2 * qd) as u32,
                                        (2 * c.head_dim) as u32, pw.kx.ptr(), pw.vx.ptr(), kvd as u32, at.qn.ptr(), at.kn.ptr(), prmp,
                                        c.n_head as u32, c.n_head_kv as u32, c.n_rot as u32, c.eps, csp, snp))?;
                                self.q.attn_prompt(s, &pw.aws, pw.att.ptr(), pw.q.ptr(), at.kq.ptr(), at.ks.ptr(), at.vq.ptr(),
                                    at.vs.ptr(), pw.qf.ptr().add::<f32>(c.head_dim), 2 * qd, 2 * c.head_dim, c.n_head,
                                    c.n_head_kv, prmp, 1.0 / (c.head_dim as f32).sqrt(), pw.sel.ptr(), pw.seln.ptr(),
                                    ldsel as usize, rows)?;
                                self.pq(&at.wo, pw.att.ptr(), qd, rows, outp, d, &mut pw.xa)?;
                            }
                        }
                        self.tp_put(&evs, r, k, outp, rows * d, host(r, k))?;
                    }
                }
                if j >= 1 {
                    // chunk j - 1: the sum of both cards' mixer outputs, then the FFN side (combine + hc norm + mix,
                    // router, this card's pair weights, FFN input q8)
                    let i = j - 1;
                    let (a, rows) = chunks[i];
                    let k = i % 2;
                    for r in 0..2 {
                        self.tp_on(r)?;
                        let pw = &mut *pws[r];
                        let outp = pw.tpo[k].ptr();
                        self.tp_sum(pw, &evs, r, k, outp, rows * d, host(1 - r, k))?;
                        let ly = &self.layers[l];
                        let (s, fk) = (&self.s, &self.fk);
                        let res = pw.hall.ptr().add::<f32>(a * hd);
                        let comb = (outp, pw.inja_n.ptr().add::<f32>(a * c.hc), c.hc);
                        if hc_fuse() {
                            self.p_hc_mix_c(pw, res, &ly.hcf, rows, &mut prof, Some(comb))?;
                        } else {
                            s.launch(&fk.hc_combine, g256(rows * hd), 256, 0, (res, comb.0, comb.1, d as u32, c.hc as u32,
                                comb.2 as u32, (rows * hd) as u32))?;
                            self.p_hc_mix(pw, res, &ly.hcf, rows, &mut prof)?;
                        }
                        let f = &ly.ffn;
                        let ids = pw.ids_n.ptr().add::<i32>(a * topk);
                        let wts = pw.wts_n.ptr().add::<f32>(a * topk);
                        let inj = pw.inj_n.ptr().add::<f32>(a * c.hc);
                        self.pb(&f.gate_inp, pw.mixed.ptr(), d, rows, pw.rl.ptr(), ld_rl, bfx(pw))?;
                        s.launch(&fk.route8, rows as u32, 256, 0, (ids, wts, pw.rl.ptr(), ld_rl as u32, c.n_expert as u32,
                            topk as u32, c.expert_weights_scale))?;
                        s.launch(&fk.mask_own, g256(rows * topk), 256, 0, (wts, ids, f.own.ptr(), (rows * topk) as u32))?;
                        s.launch(&fk.copy2d, (1u32, rows as u32), 256, 0, (inj, c.hc as u32, pw.lo.ptr().add::<f32>(c.hc_lr),
                            ld_lo as u32, c.hc as u32))?;
                        let mut ea = q8_view(&pw.ea_n, a, rows);
                        self.q.quantize_q8_1(s, pw.mixed.ptr(), d, false, rows, d, &mut ea)?;
                        s.launch(&fk.copy2d, (1u32, rows as u32), 256, 0, (pw.glog_n.ptr().add::<f32>(a), 1u32,
                            pw.rl.ptr().add::<f32>(c.n_expert), ld_rl as u32, 1u32))?;
                    }
                }
            }
            // ---- MoE over the whole group: each card the experts it owns ----
            let np_all = n_all * topk;
            let ff = c.n_ff_exp;
            if !pf_on {
                // short prompt: one look at the picks (the same on both cards); the CPU takes the slotless experts
                // with few rows (run_mmq_order: the card's prompt MMQ bits), the cards stream the others
                self.tp_on(0)?;
                let mut ids = vec![0i32; np_all];
                self.s.memcpy_dtoh(&mut ids, pws[0].ids_n.ptr())?;
                let cpu_rows: usize = std::env::var("FNEXT_CPU_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
                let mut hq8 = Hq8::empty();
                if cpu_rows > 0 {
                    hq8 = Hq8::take(n_all * d, n_all * d / 32);
                    self.s.memcpy_dtoh(&mut hq8.0, pws[0].ea_n.q.ptr())?;
                    self.s.memcpy_dtoh(&mut hq8.1, pws[0].ea_n.d.ptr())?;
                    self.s.memcpy_dtoh(&mut hq8.2, pws[0].ea_n.s.ptr())?;
                }
                // the resident experts' GEMMs go out on both cards before the host waits for the picks: the CPU's
                // slotless experts (below) then overlap them (slotless experts have null table entries: skipped)
                let early = std::env::var("FNEXT_SHORT_EARLY").as_deref() != Ok("0");
                if early {
                    // card 0's copies of the picks: their own event, before its GEMMs are queued
                    self.s.record(&evs[0].ready[0])?;
                    for r in [1usize, 0] {
                        self.tp_on(r)?;
                        let pw = &mut *pws[r];
                        let f = &self.layers[l].ffn;
                        let s = &self.s;
                        self.q.moe_sort(s, pw.ids_n.ptr(), n_all, topk, &pw.ws_g)?;
                        let ea_all = q8_view(&pw.ea_n, 0, n_all);
                        self.q.mmq_moe_gu(s, f.ty_gu, &ea_all, f.gtab.ptr(), f.utab.ptr(), f.ty_gu.row_bytes(d), ff, &pw.ws_g,
                            np_all, &mut pw.hq)?;
                        self.q.mmq_moe(s, f.ty_d, pw.ybuf.ptr(), d, &pw.hq, f.dtab.ptr(), f.ty_d.row_bytes(ff), d, &pw.ws_g, np_all,
                            true, false)?;
                    }
                    evs[0].ready[0].sync()?;
                } else {
                    self.s.sync()?;
                }
                // per owner card: (expert, pairs) of the experts without a slot on it
                let mut by_e: Vec<Vec<usize>> = vec![vec![]; ne];
                for (p, &e) in ids.iter().enumerate() {
                    by_e[e as usize].push(p);
                }
                let tabs_of = |fast: &Fast, r: usize| -> Vec<u64> {
                    if fast.tp_rank == r { fast.tabs_h[l][2].clone() } else { fast.other.as_ref().unwrap().tabs_h[l][2].clone() }
                };
                let dt = [tabs_of(self, 0), tabs_of(self, 1)];
                let slotless = |e: usize| -> bool { let r = own[l][e] as usize; dt[r][e] == 0 };
                let cpu_e: Vec<usize> = (0..ne).filter(|&e| !by_e[e].is_empty() && by_e[e].len() <= cpu_rows && slotless(e)).collect();
                let mut slot = [vec![-1i32; np_all], vec![-1i32; np_all]];
                let mut ncpu = 0usize;
                if !cpu_e.is_empty() {
                    let names = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
                    let ts: Vec<_> = names.iter().map(|nm| self.g.tensor(&format!("blk.{l}.{nm}.weight")).unwrap().clone()).collect();
                    let g = self.g.clone();
                    let jobs: Vec<fe_ggq::cpu::MoeJob> = cpu_e.iter().map(|&e| fe_ggq::cpu::MoeJob {
                        gate: g.slice(&ts[0], e), up: g.slice(&ts[1], e), down: g.slice(&ts[2], e), ty_gu: ts[0].ty,
                        ty_down: ts[2].ty, rows: by_e[e].iter().map(|p| p / topk).collect(),
                    }).collect();
                    let xq = fe_ggq::cpu::avx2::Q8Rows::from_parts(d, n_all, &hq8.0, &hq8.1, &hq8.2);
                    let cmo = prompt_cpu(&mut self.pcpu, d, ff);
                    cmo.run_mmq_order(&jobs, &xq)?;
                    let hy = ycpu_host(&mut self.ycpu, np_all * d)?;
                    let cmo = self.pcpu.as_ref().unwrap();
                    for (jj, &e) in cpu_e.iter().enumerate() {
                        let out = cmo.out(jj);
                        let r = own[l][e] as usize;
                        for (ii, &p) in by_e[e].iter().enumerate() {
                            hy[ncpu * d..(ncpu + 1) * d].copy_from_slice(&out[ii * d..(ii + 1) * d]);
                            slot[r][p] = ncpu as i32;
                            ncpu += 1;
                        }
                    }
                }
                for r in 0..2 {
                    self.tp_on(r)?;
                    let pw = &mut *pws[r];
                    let f = &self.layers[l].ffn;
                    let s = &self.s;
                    if !early {
                        self.q.moe_sort(s, pw.ids_n.ptr(), n_all, topk, &pw.ws_g)?;
                        let ea_all = q8_view(&pw.ea_n, 0, n_all);
                        self.q.mmq_moe_gu(s, f.ty_gu, &ea_all, f.gtab.ptr(), f.utab.ptr(), f.ty_gu.row_bytes(d), ff, &pw.ws_g,
                            np_all, &mut pw.hq)?;
                        self.q.mmq_moe(s, f.ty_d, pw.ybuf.ptr(), d, &pw.hq, f.dtab.ptr(), f.ty_d.row_bytes(ff), d, &pw.ws_g, np_all,
                            true, false)?;
                    }
                    let stream_e: Vec<usize> = (0..ne).filter(|&e| own[l][e] as usize == r && by_e[e].len() > cpu_rows
                        && dt[r][e] == 0).collect();
                    let groups: Vec<&[usize]> = stream_e.chunks(pw.group).collect();
                    self.p_stream(pw, l, f, &groups, &chunks)?;
                    if ncpu > 0 {
                        self.s.memcpy_htod(pw.slot.ptr(), &slot[r])?;
                        let yd = self.ycpu.as_ref().unwrap().dev();
                        self.q.moe_combine_cpu(&self.s, pw.moey.ptr(), pw.ybuf.ptr(), pw.wts_n.ptr(), n_all, topk, d, false, yd,
                            pw.slot.ptr())?;
                    } else {
                        self.q.moe_combine(&self.s, pw.moey.ptr(), pw.ybuf.ptr(), pw.wts_n.ptr(), n_all, topk, d, false)?;
                    }
                }
            } else {
                for r in 0..2 {
                    self.tp_on(r)?;
                    let pw = &mut *pws[r];
                    let p = pfs[r].as_ref().unwrap();
                    let k = l % 2;
                    let got = p.rx.recv().map_err(|_| anyhow!("prefetch thread ended"))?;
                    if got != l {
                        bail!("prefetch order: got layer {got} at layer {l}");
                    }
                    // one table: this card's decode slot of an owned expert, else its arena copy (others: 0 = skip)
                    let mut mt = std::mem::take(&mut pf_tabs[r][k]);
                    for kind in 0..3 {
                        for e in 0..ne {
                            let v = self.tabs_h[l][kind][e];
                            if v != 0 {
                                mt[kind * ne + e] = v;
                            }
                        }
                    }
                    let f = &self.layers[l].ffn;
                    let s = &self.s;
                    s.memcpy_htod(pw.mtab.ptr(), &mt)?;
                    s.wait(&pf_ev[r][k])?;
                    self.q.moe_sort(s, pw.ids_n.ptr(), n_all, topk, &pw.ws_g)?;
                    let ea_all = q8_view(&pw.ea_n, 0, n_all);
                    let mt0 = pw.mtab.ptr();
                    self.q.mmq_moe_gu(s, f.ty_gu, &ea_all, mt0, mt0.add::<u64>(ne), f.ty_gu.row_bytes(d), ff, &pw.ws_g,
                        np_all, &mut pw.hq)?;
                    self.q.moe_down_ordered(s, f.ty_d, pw.moey.ptr(), &pw.hq, mt0.add::<u64>(2 * ne), f.ty_d.row_bytes(ff), d,
                        pw.wts_n.ptr(), n_all, topk, &pw.ws_g, MOE_NCOL, pw.ybuf.ptr(), pw.tabw.ptr(), false)?;
                    s.record(&use_ev[r][k])?;
                }
            }
            // ---- shared expert (each card its half), the FFN all-reduce, the combine ----
            for r in 0..2 {
                self.tp_on(r)?;
                self.s.record(&moe_done[r])?;
            }
            for j in 0..=nch {
                if j < nch {
                    let (a, rows) = chunks[j];
                    let k = j % 2;
                    for r in 0..2 {
                        self.tp_on(r)?;
                        let pw = &mut *pws[r];
                        let f = &self.layers[l].ffn;
                        let (s, kn, fk) = (&self.s, &self.kn, &self.fk);
                        let outp = if xl { pw.moey.ptr().add::<f32>(a * d) } else {
                            s.wait(&evs[r].dh[k])?;
                            pw.tpo[k].ptr()
                        };
                        let ea = q8_view(&pw.ea_n, a, rows);
                        // the half shared expert with its h rows padded to the down weight's zero-padded K (the pad
                        // columns of sg / su are never written: zero, and SwiGLU keeps them zero)
                        let shd = f.shd_pad.as_ref().unwrap();
                        let ldf = shd.k;
                        self.q.mmq(s, f.shg.ty, pw.sg.ptr(), ldf, &ea, f.shg.buf.ptr(), f.shg.ty.row_bytes(d), f.shg.n,
                            false)?;
                        self.q.mmq(s, f.shu.ty, pw.su.ptr(), ldf, &ea, f.shu.buf.ptr(), f.shu.ty.row_bytes(d), f.shu.n,
                            false)?;
                        let nf = rows * ldf;
                        s.launch(&kn.swiglu, g256(nf), 256, 0, (pw.sg.ptr(), pw.sg.ptr(), pw.su.ptr(), nf as u32))?;
                        let mut sa = q8_shape(&pw.xa, rows, ldf);
                        self.pq(shd, pw.sg.ptr(), ldf, rows, pw.sh.ptr(), d, &mut sa)?;
                        s.launch(&fk.moe_part, (g256(d), rows as u32), 256, 0, (outp, pw.moey.ptr().add::<f32>(a * d),
                            pw.sh.ptr(), pw.glog_n.ptr().add::<f32>(a), d as u32))?;
                        if xl {
                            self.tp_put(&evf, r, j, outp, rows * d, hostf(r, j))?;
                        } else {
                            self.tp_put(&evs, r, k, outp, rows * d, host(r, k))?;
                        }
                    }
                }
                if j >= 1 && !xl {
                    let i = j - 1;
                    let (a, rows) = chunks[i];
                    let k = i % 2;
                    for r in 0..2 {
                        self.tp_on(r)?;
                        let pw = &mut *pws[r];
                        let outp = pw.tpo[k].ptr();
                        self.tp_sum(pw, &evs, r, k, outp, rows * d, host(1 - r, k))?;
                        let nh = rows * hd;
                        self.s.launch(&self.fk.hc_combine, g256(nh), 256, 0, (pw.hall.ptr().add::<f32>(a * hd), outp,
                            pw.inj_n.ptr().add::<f32>(a * c.hc), d as u32, c.hc as u32, c.hc as u32, nh as u32))?;
                    }
                }
            }
        }
        if xl {
            // the last layer's FFN all-reduce
            for (j, &(a, rows)) in chunks.iter().enumerate() {
                let k = j % 2;
                for r in 0..2 {
                    self.tp_on(r)?;
                    let pw = &mut *pws[r];
                    let fo = pw.moey.ptr().add::<f32>(a * d);
                    let rv = pw.ybuf.ptr().add::<f32>(k * cm * d);
                    self.tp_sum_rv(pw, &evf, r, j, fo, rows * d, hostf(1 - r, j), rv, Some(&moe_done[r]))?;
                    let nh = rows * hd;
                    self.s.launch(&self.fk.hc_combine, g256(nh), 256, 0, (pw.hall.ptr().add::<f32>(a * hd), fo,
                        pw.inj_n.ptr().add::<f32>(a * c.hc), d as u32, c.hc as u32, c.hc as u32, nh as u32))?;
                }
            }
        }
        drop(pfs);
        // head on the last row, vocab-split: each card its half, the host picks
        let last = n_all - 1;
        let half = c.n_vocab / 2;
        for r in 0..2 {
            self.tp_on(r)?;
            let pw = &*pws[r];
            let (s, kn, fk) = (&self.s, &self.kn, &self.fk);
            s.memcpy_dtod(self.w.res.ptr(), pw.hall.ptr().add::<f32>(last * hd), hd * 4)?;
            let (hcm, ow) = match &self.head_tp { Some((h, o)) => (h, o), None => (&self.head, &self.output) };
            let (row0, nrows) = if r == 0 { (0, half) } else { (half, c.n_vocab - half) };
            hc_mix(kn, fk, s, &self.w, hcm, &c, 1)?;
            let w = &mut self.w;
            self.q.quantize_q8_1(s, w.mixed.ptr(), d, false, 1, d, &mut w.xa)?;
            ow.run(&self.q, s, &w.xa, w.logits.ptr(), false)?;
            super::argmax_rows(fk, s, w, w.logits.ptr(), w.tok.ptr(), 1, nrows, DevPtr::NULL)?;
            s.launch(&fk.argmax_tv, 1, 32, 0, (w.tv.ptr(), w.tok.ptr(), w.logits.ptr(), nrows as u32, row0 as u32, 1u32,
                DevPtr::NULL))?;
            s.memcpy_dtoh(std::slice::from_raw_parts_mut((self.tv_h.as_ptr() as *mut u32).add(r * MAX_ROWS * 3), 3), w.tv.ptr())?;
            if self.want_logits {
                s.memcpy_dtoh(std::slice::from_raw_parts_mut((self.logit_h.as_ptr() as *mut f32).add(row0), nrows), w.logits.ptr())?;
            }
        }
        for r in 0..2 {
            self.tp_on(r)?;
            self.s2.sync()?;
            self.s.sync()?;
        }
        self.tp_on(1)?;
        let (t, _) = self.tp_pick(0);
        self.bad_rows = if self.tp_row_finite(0) { 0 } else { 1 };
        self.tokens.extend_from_slice(toks);
        Ok(t)
    }

    /// FNEXT_TP=2 prompt all-reduce, send side (card r, slot k): `n` floats at `x` (on the card) to its host slot
    /// `hm`, on the side stream after the compute stream's work so far, once the other card has read the slot's
    /// previous contents.
    unsafe fn tp_put(&self, evs: &[TpEv], r: usize, k: usize, x: DevPtr, n: usize, hm: usize) -> Result<()> {
        self.s.record(&evs[r].ready[k])?;
        self.s2.wait(&evs[r].ready[k])?;
        self.s2.wait(&evs[1 - r].cons[k])?;
        for (p, (a, b)) in pieces(n).into_iter().enumerate() {
            self.s2.memcpy_dtoh(std::slice::from_raw_parts_mut((hm as *mut f32).add(a), b - a), x.add::<f32>(a))?;
            self.s2.record(&evs[r].dhp[k][p])?;
        }
        self.s2.record(&evs[r].dh[k])?;
        Ok(())
    }

    /// tp_sum with the receive buffer `rv` (its own reuse slot `rk` in the set's sum events is the slot k itself here:
    /// the caller alternates rv between two buffers by chunk parity) and an extra event the copy-in waits for (the
    /// buffer's previous user).
    unsafe fn tp_sum_rv(&self, pw: &PWork, evs: &[TpEv], r: usize, k: usize, x: DevPtr, n: usize, ho: usize, rv: DevPtr,
        after: Option<&Event>) -> Result<()> {
        let pw_cps = &pw.cps;
        if let Some(e) = after {
            pw_cps.wait(e)?;
        }
        // the previous use of this receive buffer (chunk k - 2 of the same set) must have been summed
        if k >= 2 {
            pw_cps.wait(&evs[r].sum[k - 2])?;
        }
        pw_cps.wait(&evs[r].sum[k])?;
        for (p, (a, b)) in pieces(n).into_iter().enumerate() {
            pw_cps.wait(&evs[1 - r].dhp[k][p])?;
            pw_cps.memcpy_htod(rv.add::<f32>(a), std::slice::from_raw_parts((ho as *const f32).add(a), b - a))?;
        }
        pw_cps.record(&evs[r].cons[k])?;
        self.s.wait(&evs[r].dh[k])?;
        self.s.wait(&evs[r].cons[k])?;
        self.s.launch(&self.fk.ar_sum, (n as u32).div_ceil(1024), 256, 0, (x, x, rv, n as u32))?;
        self.s.record(&evs[r].sum[k])?;
        Ok(())
    }

    /// Receive side (card r, slot k): the other card's partial from its host slot `ho` into `rv` (copy engine, the
    /// prompt copy stream), then x += rv on the compute stream once this card's own copy out of x is done too.
    unsafe fn tp_sum(&self, pw: &PWork, evs: &[TpEv], r: usize, k: usize, x: DevPtr, n: usize, ho: usize) -> Result<()> {
        let rv = pw.tpo[2 + k].ptr();
        pw.cps.wait(&evs[r].sum[k])?;
        for (p, (a, b)) in pieces(n).into_iter().enumerate() {
            pw.cps.wait(&evs[1 - r].dhp[k][p])?;
            pw.cps.memcpy_htod(rv.add::<f32>(a), std::slice::from_raw_parts((ho as *const f32).add(a), b - a))?;
        }
        pw.cps.record(&evs[r].cons[k])?;
        self.s.wait(&evs[r].dh[k])?;
        self.s.wait(&evs[r].cons[k])?;
        self.s.launch(&self.fk.ar_sum, (n as u32).div_ceil(1024), 256, 0, (x, x, rv, n as u32))?;
        self.s.record(&evs[r].sum[k])?;
        Ok(())
    }
}

/// FNEXT_TP=2 prompt all-reduce: pieces per copy (FNEXT_AR_PIECES, default 8), so the receiving copy of a piece
/// overlaps the sending copy of the next (a 42 MB partial: DtoH then HtoD back to back was ~7 ms a reduction).
fn ar_pieces() -> usize {
    std::env::var("FNEXT_AR_PIECES").ok().and_then(|v| v.parse().ok()).unwrap_or(8).clamp(1, 32)
}

/// [a, b) float ranges of the pieces of n floats (multiples of 64 floats).
fn pieces(n: usize) -> Vec<(usize, usize)> {
    let np = ar_pieces();
    let step = n.div_ceil(np).next_multiple_of(64).max(64);
    (0..n).step_by(step).map(|a| (a, (a + step).min(n))).collect()
}


/// Pinned staging buffers of the prompt workspaces, kept across prompts: cuMemHostAlloc of the two 32 MB stages of a
/// short prompt took ~65 ms per card (and ~20 ms to free), most of a short prompt's setup.
static STAGE_POOL: std::sync::Mutex<Vec<HostBuffer<u8>>> = std::sync::Mutex::new(Vec::new());

/// A pinned buffer of at least `n` bytes: the smallest pooled one that fits, else a new one.
fn stage_take(n: usize) -> Result<HostBuffer<u8>> {
    let mut p = STAGE_POOL.lock().unwrap();
    if let Some(i) = (0..p.len()).filter(|&i| p[i].len() >= n).min_by_key(|&i| p[i].len()) {
        return Ok(p.swap_remove(i));
    }
    drop(p);
    Ok(HostBuffer::alloc(n)?)
}

/// Back to the pool (the largest eight are kept: two cards x two stages of the short-prompt workspace and of the
/// long-prompt prefetch thread).
fn stage_put(bufs: Vec<HostBuffer<u8>>) {
    let mut p = STAGE_POOL.lock().unwrap();
    p.extend(bufs.into_iter().filter(|b| b.len() > 1));
    p.sort_by_key(|b| std::cmp::Reverse(b.len()));
    p.truncate(8);
}

impl Drop for PWork {
    fn drop(&mut self) {
        stage_put(std::mem::take(&mut self.stage));
    }
}

/// Fill the stage pool at load (two cards x two stages of a short prompt's group), so the first prompt of a shape does
/// not pay ~65 ms of cuMemHostAlloc per stage (FNEXT_STAGE_PREWARM=0: off).
pub(crate) fn prewarm_stages(c: &Config, g: &Gguf, group: usize, n: usize) -> Result<()> {
    if std::env::var("FNEXT_STAGE_PREWARM").as_deref() == Ok("0") || group == 0 {
        return Ok(());
    }
    let sm = slot_max(c, g)?;
    let bufs = (0..n).map(|_| stage_take(group * sm)).collect::<Result<Vec<_>>>()?;
    let mut p = STAGE_POOL.lock().unwrap();
    p.extend(bufs);
    Ok(())
}

/// Load-time workspace probes (device sizes only): the pinned stages are not taken from the pool.
static PROBE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub(crate) fn set_probe(on: bool) {
    PROBE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Host copies of a short prompt's q8 activation rows (for the experts the CPU computes), one set reused across
/// layers and prompts: a fresh ~10 MB allocation in each of the 48 layers is a new mapping under the fixed malloc
/// thresholds (fe-server), and its page faults cost ~10% of a 2K prompt.
static HQ8_POOL: std::sync::Mutex<Option<(Vec<i8>, Vec<f32>, Vec<f32>)>> = std::sync::Mutex::new(None);

struct Hq8 {
    v: (Vec<i8>, Vec<f32>, Vec<f32>),
}

impl Hq8 {
    fn empty() -> Hq8 {
        Hq8 { v: (vec![], vec![], vec![]) }
    }
    /// Buffers of nq quants and ns scales (contents undefined: the caller overwrites them).
    fn take(nq: usize, ns: usize) -> Hq8 {
        let mut v = HQ8_POOL.lock().unwrap().take().unwrap_or_default();
        v.0.resize(nq, 0);
        v.1.resize(ns, 0.0);
        v.2.resize(ns, 0.0);
        Hq8 { v }
    }
}

impl Drop for Hq8 {
    fn drop(&mut self) {
        if self.v.0.capacity() > 0 {
            let mut p = HQ8_POOL.lock().unwrap();
            if p.as_ref().is_none_or(|q| q.0.capacity() <= self.v.0.capacity()) {
                *p = Some(std::mem::take(&mut self.v));
            }
        }
    }
}

impl std::ops::Deref for Hq8 {
    type Target = (Vec<i8>, Vec<f32>, Vec<f32>);
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl std::ops::DerefMut for Hq8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}
