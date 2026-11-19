//! Lanes: concurrent sequences sharing the weights, the expert LRU and the prompt workspace. The layers always hold
//! the current lane's per-sequence state (KV / pooled keys / indexer ring, GDN conv + recurrent state, PLE history,
//! the MTP block's KV / ring / h row, the step graphs that captured those pointers); the other lanes' state waits in
//! `Fast::lanes` and `select_lane` swaps it in. Single-lane code paths are unchanged.
use super::*;

/// Kinds of batched graphs (the first word of a `lane_graphs` key).
const K_STEP: usize = 0;
const K_VERIFY: usize = 1;
const K_MTP: usize = 2;
const K_COMMIT: usize = 3;
/// Batched step graphs kept per device (FNEXT_LANE_GRAPHS, default 160): past it the least recently used are
/// destroyed down to three quarters and recaptured when used again. Every captured graph holds host memory inside
/// the CUDA driver (~9.7 MiB a graph measured on an RTX 5080 with 4 lanes), and the commit graphs alone have one shape per lane,
/// row offset and length, so an uncapped cache grew the process by gigabytes over hours of mixed traffic.
fn max_lane_graphs() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FNEXT_LANE_GRAPHS").ok().and_then(|v| v.parse().ok()).unwrap_or(160).max(16))
}

/// The batched step graphs of one device by shape key, with a use stamp for LRU eviction.
#[derive(Default)]
pub(crate) struct GraphCache {
    map: std::collections::HashMap<Vec<usize>, (Graph, std::cell::Cell<u64>)>,
    tick: std::cell::Cell<u64>,
}

// SAFETY: like Graph (Send + Sync); the stamps are only touched by the engine thread that owns the Fast
unsafe impl Sync for GraphCache {}

impl GraphCache {
    pub(crate) fn contains_key(&self, k: &[usize]) -> bool {
        self.map.contains_key(k)
    }
    pub(crate) fn insert(&mut self, k: Vec<usize>, g: Graph) {
        let t = self.tick.get() + 1;
        self.tick.set(t);
        self.map.insert(k, (g, std::cell::Cell::new(t)));
    }
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }
    pub(crate) fn clear(&mut self) {
        self.map.clear();
    }
    /// Destroy the least recently used graphs until `keep` are left.
    pub(crate) fn evict_to(&mut self, keep: usize) {
        if self.map.len() <= keep {
            return;
        }
        let mut stamps: Vec<(u64, Vec<usize>)> = self.map.iter().map(|(k, v)| (v.1.get(), k.clone())).collect();
        stamps.sort_unstable();
        let n = self.map.len() - keep;
        for (_, k) in stamps.into_iter().take(n) {
            self.map.remove(&k);
        }
    }
    fn get(&self, k: &[usize]) -> &Graph {
        let e = self.map.get(k).expect("lane graph captured before use");
        let t = self.tick.get() + 1;
        self.tick.set(t);
        e.1.set(t);
        &e.0
    }
}

impl std::ops::Index<&[usize]> for GraphCache {
    type Output = Graph;
    fn index(&self, k: &[usize]) -> &Graph {
        self.get(k)
    }
}

impl std::ops::Index<&Vec<usize>> for GraphCache {
    type Output = Graph;
    fn index(&self, k: &Vec<usize>) -> &Graph {
        self.get(k)
    }
}

/// A lane's KV / pooled keys / indexer ring of one attention layer, as raw pointers for a batched step.
#[derive(Clone, Copy)]
pub(crate) struct KvPtr {
    pub(crate) kq: DevPtr,
    pub(crate) ks: DevPtr,
    pub(crate) vq: DevPtr,
    pub(crate) vs: DevPtr,
    pub(crate) pooled: DevPtr,
    pub(crate) ring: DevPtr,
    pub(crate) ring_n: u32,
}

impl KvPtr {
    pub(crate) fn of(at: &Attn, c: &Config) -> KvPtr {
        KvPtr { kq: at.kq.ptr(), ks: at.ks.ptr(), vq: at.vq.ptr(), vs: at.vs.ptr(), pooled: at.pooled.ptr(), ring: at.ring.ptr(),
            ring_n: (at.ring.len() / c.idx_dim) as u32 }
    }
}

/// A lane's per-sequence pointers for one layer (unused fields NULL).
#[derive(Clone, Copy)]
pub(crate) struct SeqPtrs {
    pub(crate) conv: DevPtr,
    pub(crate) ssm: DevPtr,
    pub(crate) kv: KvPtr,
    pub(crate) hist: DevPtr,
}

/// One lane's per-sequence device state of a layer while it is not the current lane.
pub(crate) enum LaneLayer {
    Gdn { conv_state: DeviceBuffer<f32>, ssm: DeviceBuffer<f32> },
    Attn { kq: VmBuffer, ks: VmBuffer, vq: VmBuffer, vs: VmBuffer, pooled: VmBuffer, ring: DeviceBuffer<f32> },
}

/// A parked lane (not the current one): its device state and host bookkeeping.
pub(crate) struct LaneState {
    pub(crate) layers: Vec<LaneLayer>,
    pub(crate) ple_hist: Vec<Option<DeviceBuffer<f32>>>,
    /// FNEXT_TP=2: the same for card 0's layers (`layers` / `ple_hist` are card 1's); empty otherwise
    pub(crate) layers0: Vec<LaneLayer>,
    pub(crate) ple_hist0: Vec<Option<DeviceBuffer<f32>>>,
    pub(crate) mtp: Option<(VmBuffer, VmBuffer, VmBuffer, VmBuffer, VmBuffer, DeviceBuffer<f32>, DeviceBuffer<f32>)>,
    pub(crate) hsave: DeviceBuffer<f32>,
    pub(crate) tokens: Vec<u32>,
    pub(crate) snaps: Vec<Snap>,
    pub(crate) ctx_mapped: usize,
    pub(crate) draft_p: f32,
    pub(crate) acc_ema: f32,
    pub(crate) graphs: Vec<Option<Graph>>,
    /// the step graphs of the parked device set (FNEXT_GPUS=2: device 0's segment captured this lane's pointers)
    pub(crate) graphs0: Vec<Option<Graph>>,
    pub(crate) mtp_graphs: Vec<Option<Graph>>,
}

fn kv_set(c: &Config, ctx: usize, ord: i32) -> Result<(VmBuffer, VmBuffer, VmBuffer, VmBuffer, VmBuffer)> {
    let kvd = c.n_head_kv * c.head_dim;
    let mut kq = VmBuffer::reserve(ctx * kvd, ord)?;
    let mut ks = VmBuffer::reserve(ctx * kvd / 32 * 4, ord)?;
    let mut vq = VmBuffer::reserve(ctx * kvd, ord)?;
    let mut vs = VmBuffer::reserve(ctx * kvd / 32 * 4, ord)?;
    let mut pooled = VmBuffer::reserve((ctx / c.kpool + 1) * c.idx_dim * 4, ord)?;
    let c0 = 8192.min(ctx);
    kq.grow(c0 * kvd)?;
    vq.grow(c0 * kvd)?;
    ks.grow(c0 * kvd / 32 * 4)?;
    vs.grow(c0 * kvd / 32 * 4)?;
    pooled.grow((c0 / c.kpool + 1) * c.idx_dim * 4)?;
    Ok((kq, ks, vq, vs, pooled))
}

/// The per-sequence state of lanes 1 .. n (lane 0 is what load built into the layers); each lane reserves address
/// space for the whole context and maps its first 8192 cells (the rest on demand). Every layer's lane state lives on
/// the device that runs the layer (`cu_of`); the MTP block's and the trunk residual copy on the last device.
#[allow(clippy::too_many_arguments)]
pub(crate) fn alloc_lanes(c: &Config, ch: &Config, ctx: usize, layers: &[Layer], layers0: &[Layer], mtp: Option<&Mtp>,
    hsave_len: usize, n: usize, cu_of: &dyn Fn(usize) -> Arc<fe_cuda::Context>, cu_last: &Arc<fe_cuda::Context>,
    cu0: &Arc<fe_cuda::Context>) -> Result<Vec<Option<LaneState>>> {
    let mut out: Vec<Option<LaneState>> = vec![None];
    // one lane's state of a set of layers (`ch`: the config of their KV width)
    let state = |layers: &[Layer], cu_of: &dyn Fn(usize) -> Arc<fe_cuda::Context>|
        -> Result<(Vec<LaneLayer>, Vec<Option<DeviceBuffer<f32>>>)> {
        let mut ls = vec![];
        let mut ple_hist = vec![];
        for (l, ly) in layers.iter().enumerate() {
            let cu = cu_of(l);
            cu.bind()?;
            let ord = cu.ordinal();
            ls.push(match &ly.mix {
                Mixer::Gdn(gd) => LaneLayer::Gdn {
                    conv_state: DeviceBuffer::zeroed(gd.conv_state.len())?,
                    ssm: DeviceBuffer::zeroed(gd.ssm.len())?,
                },
                Mixer::Attn(at) => {
                    let (kq, ks, vq, vs, pooled) = kv_set(ch, ctx, ord)?;
                    LaneLayer::Attn { kq, ks, vq, vs, pooled, ring: DeviceBuffer::zeroed(at.ring.len())? }
                }
            });
            ple_hist.push(match &ly.ple {
                Some(p) => Some(DeviceBuffer::zeroed(p.hist.len())?),
                None => None,
            });
        }
        Ok((ls, ple_hist))
    };
    for _ in 1..n {
        let (ls, ple_hist) = state(layers, cu_of)?;
        let (ls0, ple_hist0) = state(layers0, &|_| cu0.clone())?;
        cu_last.bind()?;
        let m = match mtp {
            Some(m) => {
                let (kq, ks, vq, vs, pooled) = kv_set(c, ctx, cu_last.ordinal())?;
                Some((kq, ks, vq, vs, pooled, DeviceBuffer::zeroed(m.at.ring.len())?, DeviceBuffer::zeroed(m.hprev.len())?))
            }
            None => None,
        };
        out.push(Some(LaneState {
            layers: ls, ple_hist, layers0: ls0, ple_hist0, mtp: m, hsave: DeviceBuffer::zeroed(hsave_len)?, tokens: vec![], snaps: vec![],
            ctx_mapped: 8192.min(ctx), draft_p: 1.0, acc_ema: 0.9,
            graphs: (0..2 * N_GRAPHS).map(|_| None).collect(), graphs0: (0..2 * N_GRAPHS).map(|_| None).collect(),
            mtp_graphs: (0..=MAX_ROWS).map(|_| None).collect(),
        }));
    }
    Ok(out)
}

impl Fast {
    /// Lane `ln`'s per-sequence pointers for every layer (from the layers when it is current, else its parked state).
    pub(crate) fn seq_ptrs(&self, ln: usize) -> Vec<SeqPtrs> {
        let null = KvPtr { kq: DevPtr::NULL, ks: DevPtr::NULL, vq: DevPtr::NULL, vs: DevPtr::NULL, pooled: DevPtr::NULL,
            ring: DevPtr::NULL, ring_n: 0 };
        let c = &self.c;
        self.layers.iter().enumerate().map(|(l, ly)| {
            let mut sp = SeqPtrs { conv: DevPtr::NULL, ssm: DevPtr::NULL, kv: null, hist: DevPtr::NULL };
            if ln == self.cur_lane {
                match &ly.mix {
                    Mixer::Gdn(gd) => {
                        sp.conv = gd.conv_state.ptr();
                        sp.ssm = gd.ssm.ptr();
                    }
                    Mixer::Attn(at) => sp.kv = KvPtr::of(at, c),
                }
                if let Some(p) = &ly.ple {
                    sp.hist = p.hist.ptr();
                }
            } else {
                let st = self.lanes[ln].as_ref().expect("parked lane");
                let (sl, ph) = if self.tp && self.tp_rank == 0 { (&st.layers0, &st.ple_hist0) } else { (&st.layers, &st.ple_hist) };
                match &sl[l] {
                    LaneLayer::Gdn { conv_state, ssm } => {
                        sp.conv = conv_state.ptr();
                        sp.ssm = ssm.ptr();
                    }
                    LaneLayer::Attn { kq, ks, vq, vs, pooled, ring } => {
                        sp.kv = KvPtr { kq: kq.ptr(), ks: ks.ptr(), vq: vq.ptr(), vs: vs.ptr(), pooled: pooled.ptr(),
                            ring: ring.ptr(), ring_n: (ring.len() / c.idx_dim) as u32 };
                    }
                }
                if let Some(h) = &ph[l] {
                    sp.hist = h.ptr();
                }
            }
            sp
        }).collect()
    }

    /// One decode step for several lanes at once, one row each: (lane, token) pairs over distinct lanes (rows in
    /// this order). Returns each lane's greedy next token. One graph per lane set; per-sequence kernels run per lane,
    /// everything row-wise over all rows, so each lane gets the bits it would get alone.
    pub fn step_lanes(&mut self, items: &[(usize, u32)]) -> Result<Vec<u32>> {
        let n = items.len();
        if n == 0 {
            return Ok(vec![]);
        }
        if n == 1 {
            self.select_lane(items[0].0)?;
            return Ok(vec![self.step(&[items[0].1])?]);
        }
        if n > MAX_LANES.min(MAX_ROWS) {
            bail!("step_lanes: {n} lanes");
        }
        let t0 = std::time::Instant::now();
        for (i, &(ln, t)) in items.iter().enumerate() {
            self.select_lane(ln)?;
            self.prepare_at(&[t], i, i)?;
            self.ple_gather_at(i);
        }
        let t1 = std::time::Instant::now();
        // the PLE rows are already in place
        unsafe { std::ptr::write_volatile(self.ple_flag.as_ptr() as *mut u32, 1) };
        std::sync::atomic::fence(Ordering::SeqCst);
        self.plan = items.iter().enumerate().map(|(i, &(ln, _))| (ln, i, 1)).collect();
        self.cur_rows.store(n as u32, Ordering::Relaxed);
        let key: Vec<usize> = [K_STEP, 1].into_iter().chain(items.iter().map(|x| x.0)).collect();
        let r = self.launch_lanes(&key, n, false);
        self.plan.clear();
        r?;
        for &(ln, t) in items {
            if ln == self.cur_lane {
                self.tokens.push(t);
            } else {
                self.lanes[ln].as_mut().unwrap().tokens.push(t);
            }
        }
        let t2 = std::time::Instant::now();
        self.step_no.fetch_add(1, Ordering::Relaxed);
        if self.dynp.is_some() {
            self.refill()?;
        }
        {
            let mut st = self.stats.lock().unwrap();
            st.prep_us += (t1 - t0).as_micros() as u64;
            st.graph_us += (t2 - t1).as_micros() as u64;
            st.refill_us += t2.elapsed().as_micros() as u64;
            st.steps += 1;
            st.rows += n as u64;
        }
        Ok(self.tok_h.as_slice()[..n].to_vec())
    }

    /// Capture (once per lane set and device) and run the batched step graph(s) of `self.plan`, wait for the result.
    /// Layer split: device 0's segment first, device 1's waits for it on the cross-device event (as run_step).
    fn launch_lanes(&mut self, key: &[usize], n: usize, verify: bool) -> Result<()> {
        // the logits host copy is part of the captured graph: part of the key (see G_LOGITS)
        let key: Vec<usize> = key.iter().copied().chain([self.logits_in_graph(n, verify, true) as usize]).collect();
        let key = &key[..];
        self.bound_lane_graphs()?;
        // every graph is captured before anything launches (instantiation can wait for the device)
        self.each_dev(|m| {
            if !m.lane_graphs.contains_key(key) {
                let g = m.capture(|m| unsafe { m.enqueue(n, verify) })?;
                m.lane_graphs.insert(key.to_vec(), g);
            }
            Ok(())
        })?;
        if self.tp {
            // both cards at once (card 0 on the launcher thread), a fresh epoch for the all-reduce tags
            self.epoch = self.epoch.wrapping_add(1);
            self.prm_h.as_mut_slice()[P_EPOCH] = self.epoch << 8;
            let o = self.other.as_ref().unwrap();
            unsafe { self.launcher.as_ref().unwrap().post(&o.lane_graphs[key], &o.s, &o.done_ev) };
            let r = self.lane_graphs[key].launch(&self.s);
            self.launcher.as_ref().unwrap().wait()?;
            r?;
            self.s.record(&self.done_ev)?;
            self.done_ev.sync()?;
            self.other.as_ref().unwrap().done_ev.sync()?;
            // every row's greedy token over both vocabulary halves
            self.bad_rows = 0;
            for r in 0..n.min(MAX_ROWS) {
                let (t, _) = self.tp_pick(r);
                self.tok_h.as_mut_slice()[r] = t;
                if !self.tp_row_finite(r) {
                    self.bad_rows |= 1 << r;
                }
            }
            return Ok(());
        }
        if self.other.is_some() {
            self.swap_dev()?;
            let r = self.lane_graphs[key].launch(&self.s).map_err(anyhow::Error::from)
                .and_then(|_| self.s.record(&self.done_ev).map_err(anyhow::Error::from));
            self.swap_dev()?;
            r?;
            self.s.wait(&self.other.as_ref().unwrap().done_ev)?;
        }
        self.lane_graphs[key].launch(&self.s)?;
        self.s.record(&self.done_ev)?;
        self.done_ev.sync()?;
        Ok(())
    }

    /// The context of the device that runs block `l` (the MTP block: `l == n_layer`, the last device).
    pub(crate) fn cu_of_layer(&self, l: usize) -> Arc<fe_cuda::Context> {
        match &self.other {
            Some(o) if l >= o.lo && l < o.hi => o.cu.clone(),
            _ => self.cu.clone(),
        }
    }

    /// Wait for every device's streams.
    pub(crate) fn sync_all(&mut self) -> Result<()> {
        self.each_dev(|f| {
            f.s.sync()?;
            f.s2.sync()?;
            Ok(())
        })
    }

    /// Logits of row `r` of the last step's logits buffer on the card (a verify computes every row's).
    pub fn logits_row(&mut self, r: usize) -> Result<Vec<f32>> {
        let v = self.c.n_vocab;
        let mut out = vec![0f32; v];
        if self.tp {
            // FNEXT_TP=2: each card holds its half of the vocabulary, rows of that half's width
            let half = v / 2;
            let mut part: [(usize, usize, Vec<f32>); 2] = [(0, half, vec![]), (half, v - half, vec![])];
            self.sync_all()?;
            for _ in 0..2 {
                let rk = self.tp_rank;
                let (_, n, buf) = &mut part[rk];
                buf.resize(*n, 0.0);
                fe_cuda::memcpy_dtoh(buf, self.w.logits.ptr().add::<f32>(r * *n))?;
                self.swap_dev()?;
            }
            for (o, n, buf) in &part {
                out[*o..*o + *n].copy_from_slice(&buf[..*n]);
            }
            return Ok(out);
        }
        self.s.sync()?;
        fe_cuda::memcpy_dtoh(&mut out, self.w.logits.ptr().add::<f32>(r * v))?;
        Ok(out)
    }

    /// Logits of row `i` of the last batched step (with want_logits).
    pub fn lane_logits(&self, i: usize) -> &[f32] {
        let v = self.c.n_vocab;
        &self.logit_h.as_slice()[i * v..(i + 1) * v]
    }

    /// Give lane `ln`'s KV back to the pool: unmap past the first 8192 cells and forget its sequence (snapshots too).
    /// The lane must not be current.
    pub fn trim_lane(&mut self, ln: usize) -> Result<()> {
        if ln == self.cur_lane {
            bail!("trim_lane: lane {ln} is current");
        }
        let c = self.c.clone();
        let kvd = c.n_head_kv * c.head_dim;
        let c0 = 8192.min(self.ctx);
        self.sync_all()?;
        let cus: Vec<Arc<fe_cuda::Context>> = (0..=c.n_layer).map(|l| self.cu_of_layer(l)).collect();
        let tp = self.tp;
        let (cu1, cu0) = (self.cu.clone(), self.other.as_ref().map(|o| o.cu.clone()));
        let kvd_m = self.cfull.n_head_kv * self.cfull.head_dim;
        let st = self.lanes[ln].as_mut().unwrap();
        let shrink = |kq: &mut VmBuffer, ks: &mut VmBuffer, vq: &mut VmBuffer, vs: &mut VmBuffer, pooled: &mut VmBuffer, kvd: usize| -> Result<()> {
            kq.shrink(c0 * kvd)?;
            vq.shrink(c0 * kvd)?;
            ks.shrink(c0 * kvd / 32 * 4)?;
            vs.shrink(c0 * kvd / 32 * 4)?;
            pooled.shrink((c0 / c.kpool + 1) * c.idx_dim * 4)?;
            Ok(())
        };
        for (l, ly) in st.layers.iter_mut().enumerate() {
            if let LaneLayer::Attn { kq, ks, vq, vs, pooled, .. } = ly {
                if tp { cu1.bind()? } else { cus[l].bind()? };
                shrink(kq, ks, vq, vs, pooled, kvd)?;
            }
        }
        for ly in st.layers0.iter_mut() {
            if let LaneLayer::Attn { kq, ks, vq, vs, pooled, .. } = ly {
                cu0.as_ref().unwrap().bind()?;
                shrink(kq, ks, vq, vs, pooled, kvd)?;
            }
        }
        if tp { cu1.bind()? } else { cus[c.n_layer].bind()? };
        if let Some((kq, ks, vq, vs, pooled, _, _)) = &mut st.mtp {
            shrink(kq, ks, vq, vs, pooled, kvd_m)?;
        }
        st.ctx_mapped = c0;
        st.tokens.clear();
        st.snaps.clear();
        Ok(())
    }

    /// Cells lane `ln` has mapped.
    pub fn lane_mapped(&self, ln: usize) -> usize {
        if ln == self.cur_lane { self.ctx_mapped } else { self.lanes[ln].as_ref().unwrap().ctx_mapped }
    }

    /// Lane `ln`'s tokens.
    pub fn lane_tokens(&self, ln: usize) -> &[u32] {
        if ln == self.cur_lane { &self.tokens } else { &self.lanes[ln].as_ref().unwrap().tokens }
    }

    /// Park the current lane's conversation in host memory: its KV / pooled keys up to its last snapshot, the
    /// snapshots and the tokens; the lane is reset. None when it has no snapshot to resume from.
    pub fn park(&mut self) -> Result<Option<Parked>> {
        let Some(n) = self.snaps.iter().map(|s| s.pos).max() else { return Ok(None) };
        if self.tp {
            // FNEXT_TP=2: each card's KV halves (card 0 first), then the MTP block's (card 1), on their own streams
            self.sync_all()?;
            let mut kv: Vec<u8> = vec![];
            self.each_dev(|f| {
                let parts = f.kv_parts_dev(n);
                let base = kv.len();
                kv.resize(base + parts.iter().map(|p| p.1).sum::<usize>(), 0);
                let mut o = base;
                for (p, b) in parts {
                    unsafe { f.s.memcpy_dtoh(&mut kv[o..o + b], p)? };
                    o += b;
                }
                f.s.sync()?;
                Ok(())
            })?;
            let snaps = std::mem::take(&mut self.snaps);
            let tokens = self.tokens[..n].to_vec();
            self.reset()?;
            return Ok(Some(Parked { tokens, snaps, kv, cells: n }));
        }
        let c = self.c.clone();
        let kvd = c.n_head_kv * c.head_dim;
        let parts = |n: usize| -> [usize; 5] { [n * kvd, n * kvd / 32 * 4, n * kvd, n * kvd / 32 * 4, (n / c.kpool + 1) * c.idx_dim * 4] };
        let mut bufs: Vec<DevPtr> = vec![];
        for ly in &self.layers {
            if let Mixer::Attn(at) = &ly.mix {
                bufs.extend([at.kq.ptr(), at.ks.ptr(), at.vq.ptr(), at.vs.ptr(), at.pooled.ptr()]);
            }
        }
        if let Some(m) = &self.mtp {
            bufs.extend([m.at.kq.ptr(), m.at.ks.ptr(), m.at.vq.ptr(), m.at.vs.ptr(), m.at.pooled.ptr()]);
        }
        let sz = parts(n);
        let per: usize = sz.iter().sum();
        let mut kv = vec![0u8; per * bufs.len() / 5];
        self.sync_all()?;
        // buffers in block order (MTP last); each device copies its own on its stream
        let devs: Vec<bool> = self.kv_layer_devs();
        let mut o = 0;
        let mut offs = vec![];
        for i in 0..bufs.len() {
            offs.push(o);
            o += sz[i % 5];
        }
        let kvp = kv.as_mut_ptr() as usize;
        self.each_dev(|f| {
            let last = f.is_last();
            for (i, p) in bufs.iter().enumerate() {
                if devs[i / 5] == last {
                    let dst = unsafe { std::slice::from_raw_parts_mut((kvp as *mut u8).add(offs[i]), sz[i % 5]) };
                    unsafe { f.s.memcpy_dtoh(dst, *p)? };
                }
            }
            f.s.sync()?;
            Ok(())
        })?;
        let snaps = std::mem::take(&mut self.snaps);
        let tokens = self.tokens[..n].to_vec();
        self.reset()?;
        Ok(Some(Parked { tokens, snaps, kv, cells: n }))
    }

    /// Bring a parked conversation back into the current lane (reset first): KV rows copied back, its snapshots and
    /// tokens restored; the next request resumes at one of its snapshots as usual.
    pub fn unpark(&mut self, p: Parked) -> Result<()> {
        self.reset()?;
        self.ensure_ctx(p.cells + MAX_ROWS)?;
        if self.tp {
            let n = p.cells;
            let mut o = 0usize;
            let kv = &p.kv;
            self.each_dev(|f| {
                for (d, b) in f.kv_parts_dev(n) {
                    unsafe { f.s.memcpy_htod(d, &kv[o..o + b])? };
                    o += b;
                }
                f.s.sync()?;
                Ok(())
            })?;
            self.tokens = p.tokens;
            self.snaps = p.snaps;
            return Ok(());
        }
        let c = self.c.clone();
        let kvd = c.n_head_kv * c.head_dim;
        let n = p.cells;
        let sz = [n * kvd, n * kvd / 32 * 4, n * kvd, n * kvd / 32 * 4, (n / c.kpool + 1) * c.idx_dim * 4];
        let mut bufs: Vec<DevPtr> = vec![];
        for ly in &self.layers {
            if let Mixer::Attn(at) = &ly.mix {
                bufs.extend([at.kq.ptr(), at.ks.ptr(), at.vq.ptr(), at.vs.ptr(), at.pooled.ptr()]);
            }
        }
        if let Some(m) = &self.mtp {
            bufs.extend([m.at.kq.ptr(), m.at.ks.ptr(), m.at.vq.ptr(), m.at.vs.ptr(), m.at.pooled.ptr()]);
        }
        let devs: Vec<bool> = self.kv_layer_devs();
        let mut o = 0;
        let mut offs = vec![];
        for i in 0..bufs.len() {
            offs.push(o);
            o += sz[i % 5];
        }
        let kv = &p.kv;
        self.each_dev(|f| {
            let last = f.is_last();
            for (i, b) in bufs.iter().enumerate() {
                if devs[i / 5] == last {
                    unsafe { f.s.memcpy_htod(*b, &kv[offs[i]..offs[i] + sz[i % 5]])? };
                }
            }
            f.s.sync()?;
            Ok(())
        })?;
        self.tokens = p.tokens;
        self.snaps = p.snaps;
        Ok(())
    }

    /// FNEXT_TP=2 park / unpark: the active card's KV pieces of the first `n` cells (its attention blocks' halves in
    /// order, then the MTP block's on card 1), as (device pointer, bytes).
    fn kv_parts_dev(&self, n: usize) -> Vec<(DevPtr, usize)> {
        let parts = |c: &Config| -> [usize; 5] {
            let kvd = c.n_head_kv * c.head_dim;
            [n * kvd, n * kvd / 32 * 4, n * kvd, n * kvd / 32 * 4, (n / c.kpool + 1) * c.idx_dim * 4]
        };
        let mut v = vec![];
        let sz = parts(&self.c);
        for ly in &self.layers {
            if let Mixer::Attn(at) = &ly.mix {
                for (p, b) in [at.kq.ptr(), at.ks.ptr(), at.vq.ptr(), at.vs.ptr(), at.pooled.ptr()].into_iter().zip(sz) {
                    v.push((p, b));
                }
            }
        }
        if let (Some(m), true) = (&self.mtp, self.is_last()) {
            let sz = parts(&self.cfull);
            for (p, b) in [m.at.kq.ptr(), m.at.ks.ptr(), m.at.vq.ptr(), m.at.vs.ptr(), m.at.pooled.ptr()].into_iter().zip(sz) {
                v.push((p, b));
            }
        }
        v
    }

    /// For each KV set park / unpark copies (attention blocks in order, then the MTP block): true when it lives on
    /// the last device.
    fn kv_layer_devs(&self) -> Vec<bool> {
        let last_lo = self.other.as_ref().map_or(0, |o| o.hi);
        let mut v: Vec<bool> = self.layers.iter().enumerate().filter(|(_, ly)| matches!(ly.mix, Mixer::Attn(_)))
            .map(|(l, _)| l >= last_lo).collect();
        if self.mtp.is_some() {
            v.push(true);
        }
        v
    }

    /// Lanes this engine holds.
    pub fn n_lanes(&self) -> usize {
        self.lanes.len()
    }

    /// Captured batched-step graphs held, on every device (bounded by MAX_LANE_GRAPHS a device).
    pub fn n_lane_graphs(&self) -> usize {
        self.lane_graphs.len() + self.other.as_ref().map_or(0, |o| o.lane_graphs.len())
    }

    /// Host bytes of lane `i`'s prompt-cache snapshots.
    pub fn lane_snap_bytes(&self, i: usize) -> usize {
        let v = if i == self.cur_lane { &self.snaps } else { match self.lanes.get(i) { Some(Some(l)) => &l.snaps, _ => return 0 } };
        v.iter().map(|s| s.bytes()).sum()
    }

    /// The lane the single-sequence calls (prefill, step, spec_step, snapshots) act on.
    pub fn lane(&self) -> usize {
        self.cur_lane
    }

    /// Make lane `i` current (swaps the per-sequence state in; the stream is synced first).
    pub fn select_lane(&mut self, i: usize) -> Result<()> {
        if i == self.cur_lane {
            return Ok(());
        }
        if i >= self.lanes.len() {
            bail!("lane {i} of {}", self.lanes.len());
        }
        self.sync_all()?;
        let mut st = self.lanes[i].take().ok_or_else(|| anyhow!("lane {i} has no parked state"))?;
        self.swap_with(&mut st);
        let old = self.cur_lane;
        self.lanes[old] = Some(st);
        self.cur_lane = i;
        Ok(())
    }

    /// Exchange the current per-sequence state with `st`.
    fn swap_with(&mut self, st: &mut LaneState) {
        use std::mem::swap;
        swap_layer_state(&mut self.layers, &mut st.layers, &mut st.ple_hist);
        if self.tp {
            // card 0's layers (parked in `other` between calls)
            swap_layer_state(&mut self.other.as_mut().unwrap().layers, &mut st.layers0, &mut st.ple_hist0);
        }
        if let (Some(m), Some((kq, ks, vq, vs, pooled, ring, hprev))) = (&mut self.mtp, &mut st.mtp) {
            swap(&mut m.at.kq, kq);
            swap(&mut m.at.ks, ks);
            swap(&mut m.at.vq, vq);
            swap(&mut m.at.vs, vs);
            swap(&mut m.at.pooled, pooled);
            swap(&mut m.at.ring, ring);
            swap(&mut m.hprev, hprev);
        }
        swap(&mut self.w.hsave, &mut st.hsave);
        swap(&mut self.tokens, &mut st.tokens);
        swap(&mut self.snaps, &mut st.snaps);
        swap(&mut self.ctx_mapped, &mut st.ctx_mapped);
        swap(&mut self.draft_p, &mut st.draft_p);
        swap(&mut self.acc_ema, &mut st.acc_ema);
        swap(&mut self.graphs, &mut st.graphs);
        if let Some(o) = self.other.as_mut() {
            // between calls the last device is active: the parked set is device 0's
            swap(&mut o.graphs, &mut st.graphs0);
        }
        swap(&mut self.mtp_graphs, &mut st.mtp_graphs);
    }
}

/// Exchange the per-sequence buffers of `layers` with a lane's.
fn swap_layer_state(layers: &mut [Layer], sl: &mut [LaneLayer], ph: &mut [Option<DeviceBuffer<f32>>]) {
    use std::mem::swap;
    for (ly, (sl, ph)) in layers.iter_mut().zip(sl.iter_mut().zip(ph.iter_mut())) {
        match (&mut ly.mix, sl) {
            (Mixer::Gdn(gd), LaneLayer::Gdn { conv_state, ssm }) => {
                swap(&mut gd.conv_state, conv_state);
                swap(&mut gd.ssm, ssm);
            }
            (Mixer::Attn(at), LaneLayer::Attn { kq, ks, vq, vs, pooled, ring }) => {
                swap(&mut at.kq, kq);
                swap(&mut at.ks, ks);
                swap(&mut at.vq, vq);
                swap(&mut at.vs, vs);
                swap(&mut at.pooled, pooled);
                swap(&mut at.ring, ring);
            }
            _ => unreachable!("lane layer kinds follow the model"),
        }
        if let (Some(p), Some(h)) = (&mut ly.ple, ph) {
            swap(&mut p.hist, h);
        }
    }
}

/// A parked conversation (host memory): tokens, snapshots and the KV rows up to its last snapshot.
pub struct Parked {
    pub tokens: Vec<u32>,
    pub snaps: Vec<Snap>,
    kv: Vec<u8>,
    cells: usize,
}

impl Parked {
    /// Host bytes of its KV rows (ordinary heap memory; the snapshots are pinned buffers).
    pub fn kv_bytes(&self) -> usize {
        self.kv.len()
    }

    /// Host bytes it holds (KV rows + snapshots).
    pub fn bytes(&self) -> usize {
        self.kv.len() + self.snaps.len() * self.snaps.first().map_or(0, |s| s.bytes())
    }
}

/// Speculative decoding over several lanes at once (greedy lanes with the MTP block): chained drafts as batched MTP
/// passes (one row per lane), one batched verify (every lane padded to the same row count, so the graph set stays
/// small), per-lane commits, one batched MTP catch-up. The trunk rows are the bits each lane gets alone (row-wise
/// kernels are batch-invariant, the per-sequence ones run per lane), so every lane's tokens equal its plain greedy
/// stream; the drafts only decide how many of them a step yields.
impl Fast {
    /// Capture the batched graphs of these lane sets ahead of use (server start): plain steps, verifies and MTP
    /// passes of 1 ..= rmax rows per lane (fewer when the rows would pass MAX_ROWS). Nothing runs; the graphs only hold
    /// the lanes' state pointers, which never move.
    pub fn warm_lanes(&mut self, sets: &[Vec<usize>], rmax: usize) -> Result<usize> {
        let mut made = 0;
        for set in sets {
            let n = set.len();
            if n < 2 || n > MAX_LANES || set.iter().any(|&l| l >= self.lanes.len()) {
                continue;
            }
            let rr_max = rmax.min(MAX_ROWS / n).max(1);
            let mut jobs: Vec<(usize, usize)> = vec![(K_STEP, 1)];
            if self.has_mtp() {
                jobs.extend((1..=rr_max).map(|r| (K_VERIFY, r)));
            }
            let want = self.want_logits;
            for (kind, rr) in jobs {
                let rows = n * rr;
                let verify = kind == K_VERIFY;
                // plain steps: the greedy graph and the one that copies logits (sampled lanes); verifies are greedy only
                // (spec_lanes), keys as launch_lanes builds them
                for lg in [false, true] {
                    if lg && verify {
                        continue;
                    }
                    self.want_logits = lg;
                    self.plan = set.iter().enumerate().map(|(i, &ln)| (ln, i * rr, rr)).collect();
                    let key: Vec<usize> = [kind, rr].into_iter().chain(set.iter().copied())
                        .chain([self.logits_in_graph(rows, verify, true) as usize]).collect();
                    let r = self.each_dev(|m| {
                        if !m.lane_graphs.contains_key(&key) {
                            let g = m.capture(|m| unsafe { m.enqueue(rows, verify) })?;
                            m.lane_graphs.insert(key.clone(), g);
                            made += 1;
                        }
                        Ok(())
                    });
                    self.plan.clear();
                    self.want_logits = want;
                    r?;
                }
            }
            if self.has_mtp() {
                for k in 1..=rr_max {
                    self.plan = set.iter().enumerate().map(|(i, &ln)| (ln, i * k, k)).collect();
                    let key: Vec<usize> = [K_MTP, k].into_iter().chain(set.iter().copied()).collect();
                    let r = (|| -> Result<()> {
                        if !self.lane_graphs.contains_key(&key) {
                            let g = self.capture(|m| unsafe { m.enqueue_mtp_lanes(n * k) })?;
                            self.lane_graphs.insert(key.clone(), g);
                            made += 1;
                        }
                        Ok(())
                    })();
                    self.plan.clear();
                    r?;
                }
            }
        }
        Ok(made)
    }

    /// Keep at most max_lane_graphs() batched graphs a device: past it the least recently used go (after the
    /// devices are idle), down to three quarters, so eviction runs rarely; they are recaptured when used again.
    fn bound_lane_graphs(&mut self) -> Result<()> {
        let cap = max_lane_graphs();
        if self.lane_graphs.len() >= cap || self.other.as_ref().is_some_and(|o| o.lane_graphs.len() >= cap) {
            self.sync_all()?;
            self.each_dev(|f| {
                f.lane_graphs.evict_to(cap * 3 / 4);
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Lane `ln`'s MTP KV set, its hprev row and its hsave rows (device pointers; stable across lane swaps).
    fn lane_mtp_ptrs(&self, ln: usize) -> (KvPtr, DevPtr, DevPtr) {
        let c = &self.c;
        if ln == self.cur_lane {
            let m = self.mtp.as_ref().expect("MTP block");
            (KvPtr::of(&m.at, c), m.hprev.ptr(), self.w.hsave.ptr())
        } else {
            let st = self.lanes[ln].as_ref().expect("parked lane");
            let (kq, ks, vq, vs, pooled, ring, hprev) = st.mtp.as_ref().expect("lane MTP state");
            (KvPtr { kq: kq.ptr(), ks: ks.ptr(), vq: vq.ptr(), vs: vs.ptr(), pooled: pooled.ptr(), ring: ring.ptr(),
                ring_n: (ring.len() / c.idx_dim) as u32 }, hprev.ptr(), st.hsave.ptr())
        }
    }

    fn lane_draft_p(&mut self, ln: usize) -> &mut f32 {
        if ln == self.cur_lane { &mut self.draft_p } else { &mut self.lanes[ln].as_mut().unwrap().draft_p }
    }

    fn lane_acc(&mut self, ln: usize) -> &mut f32 {
        if ln == self.cur_lane { &mut self.acc_ema } else { &mut self.lanes[ln].as_mut().unwrap().acc_ema }
    }

    fn lane_tokens_mut(&mut self, ln: usize) -> &mut Vec<u32> {
        if ln == self.cur_lane { &mut self.tokens } else { &mut self.lanes[ln].as_mut().unwrap().tokens }
    }

    /// One MTP pass over several lanes, `k` rows each: segment i = (lane, tokens, first position); the h rows are
    /// already in mtp.hin (segment i at rows i*k ..). Returns every row's draft and its probability; the block's
    /// output residual rows are left in w.hbat.
    fn mtp_lanes(&mut self, segs: &[(usize, Vec<u32>, usize)], k: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        let n = segs.len();
        let rows = n * k;
        for (i, (ln, toks, pos)) in segs.iter().enumerate() {
            debug_assert_eq!(toks.len(), k);
            self.select_lane(*ln)?;
            self.ensure_ctx(pos + k + MAX_ROWS)?;
            self.fill_prm_at(*pos, k, i);
            self.fill_emb_at(toks, i * k);
        }
        self.plan = segs.iter().enumerate().map(|(i, (ln, _, _))| (*ln, i * k, k)).collect();
        let key: Vec<usize> = [K_MTP, k].into_iter().chain(segs.iter().map(|x| x.0)).collect();
        let r = (|| -> Result<()> {
            self.bound_lane_graphs()?;
            if !self.lane_graphs.contains_key(&key) {
                let g = self.capture(|m| unsafe { m.enqueue_mtp_lanes(rows) })?;
                self.lane_graphs.insert(key.clone(), g);
            }
            if self.tp {
                // vocab-split head: card 0's half of every row on the launcher thread
                self.epoch = self.epoch.wrapping_add(1);
                self.prm_h.as_mut_slice()[P_EPOCH] = self.epoch << 8;
                self.ensure_mtp_head_tp(rows)?;
                let o = self.other.as_ref().unwrap();
                unsafe { self.launcher.as_ref().unwrap().post(o.mtp_head_g[rows].as_ref().unwrap(), &o.s, &o.done_ev) };
                let r = self.lane_graphs[&key].launch(&self.s);
                self.launcher.as_ref().unwrap().wait()?;
                r?;
                self.s.sync()?;
                self.other.as_ref().unwrap().done_ev.sync()?;
                for i in 0..rows {
                    let (t, p) = self.tp_pick(i);
                    self.tok_h.as_mut_slice()[i] = t;
                    self.pmx_h.as_mut_slice()[i] = p;
                }
                return Ok(());
            }
            self.lane_graphs[&key].launch(&self.s)?;
            self.s.sync()?;
            Ok(())
        })();
        self.plan.clear();
        r?;
        self.mtp_ms[2] += 1.0;
        Ok((self.tok_h.as_slice()[..rows].to_vec(), self.pmx_h.as_slice()[..rows].to_vec()))
    }

    /// enqueue_mtp over the segments of `self.plan` (per-lane MTP KV / ring / step parameters for the attention,
    /// everything else row-wise); logits, draft and probability for every row; residual rows into w.hbat.
    unsafe fn enqueue_mtp_lanes(&mut self, rows: usize) -> Result<()> {
        let c = self.cfull.clone();
        let plan = self.plan.clone();
        let kvs: Vec<KvPtr> = plan.iter().map(|&(ln, _, _)| self.lane_mtp_ptrs(ln).0).collect();
        let (kn, fk, q, s) = (&self.kn, &self.fk, &self.q, &self.s);
        let w = &mut self.w;
        let m = self.mtp.as_mut().unwrap();
        let d = c.n_embd;
        let hd = c.hc_dim();
        let topk = c.n_expert_used;
        s.memcpy_htod(w.prm.ptr(), &self.prm_h.as_slice()[..plan.len() * P_WORDS])?;
        s.memcpy_htod(w.emb.ptr(), &self.emb_h.as_slice()[..rows * d])?;
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
        qhc_mix(kn, fk, q, s, w, &m.hca, m.injb.ptr(), &c, rows)?;
        let sg: Vec<(usize, usize, DevPtr, KvPtr)> = plan.iter().enumerate()
            .map(|(i, &(_, r0, n))| (r0, n, w.prm.ptr().add::<u32>(i * P_WORDS), kvs[i])).collect();
        attn_block_segs(kn, fk, q, s, w, &m.at, &c, rows, self.ctx, &sg)?;
        let n = rows * hd;
        s.launch(&fk.hc_combine, g256(n), 256, 0, (w.res.ptr(), w.out.ptr(), m.injb.ptr(), d as u32, c.hc as u32,
            c.hc as u32, n as u32))?;
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
        q.moe_gu_q8(s, m.ty_gu, m.gtab.ptr(), m.utab.ptr(), m.ty_gu.row_bytes(d), m.ids.ptr(), rows, topk, &w.ea,
            c.n_ff_exp, &mut w.ha, None, None)?;
        q.moe_down(s, m.ty_d, m.dtab.ptr(), m.ty_d.row_bytes(c.n_ff_exp), m.ids.ptr(), rows, topk, &w.ha, d, w.ey.ptr(), None)?;
        s.launch(&fk.moe_fin, (g256(d), rows as u32), 256, 0, (w.res.ptr(), w.ey.ptr(), self.sh.y.dev(), m.ids.ptr(),
            m.dtab.ptr(), m.wts.ptr(), w.sh.ptr(), m.rl.ptr().add::<f32>(m.n_exp), ldr as u32, m.injb.ptr(), c.hc as u32,
            d as u32, c.hc as u32, topk as u32, DevPtr::NULL))?;
        s.memcpy_dtod(w.hbat.ptr(), w.res.ptr(), rows * hd * 4)?;
        qhc_mix(kn, fk, q, s, w, &m.head, DevPtr::NULL, &c, rows)?;
        if self.tp {
            // vocab-split head: the head rows to card 0 (LL pairs, kofs 255), this card's half here
            let half = c.n_vocab / 2;
            s.launch(&fk.ll_pack, g256(rows * d), 256, 0, (w.llp.ptr(), w.mixed.ptr(), (rows * d) as u32,
                w.prm.ptr().add::<u32>(P_EPOCH), 255u32))?;
            s.memcpy_dtoh(std::slice::from_raw_parts_mut(self.mtp_h.as_ptr() as *mut u32, 2 * rows * d), w.llp.ptr())?;
            q.quantize_q8_1(s, w.mixed.ptr(), d, false, rows, d, &mut w.xa)?;
            self.output.run(q, s, &w.xa, w.logits.ptr(), false)?;
            argmax_rows(fk, s, w, w.logits.ptr(), w.tok.ptr(), rows, c.n_vocab - half, w.pmx.ptr())?;
            s.launch(&fk.argmax_tv, 1, 32, 0, (w.tv.ptr(), w.tok.ptr(), w.logits.ptr(), (c.n_vocab - half) as u32, half as u32,
                rows as u32, w.pmx.ptr()))?;
            s.memcpy_dtoh(std::slice::from_raw_parts_mut((self.tv_h.as_ptr() as *mut u32).add(MAX_ROWS * 3), 3 * rows), w.tv.ptr())?;
            return Ok(());
        }
        q.quantize_q8_1(s, w.mixed.ptr(), d, false, rows, d, &mut w.xa)?;
        self.output.run(q, s, &w.xa, w.logits.ptr(), false)?;
        argmax_rows(fk, s, w, w.logits.ptr(), w.tok.ptr(), rows, c.n_vocab, w.pmx.ptr())?;
        s.memcpy_dtoh(&mut self.tok_h.as_mut_slice()[..rows], w.tok.ptr())?;
        s.memcpy_dtoh(&mut self.pmx_h.as_mut_slice()[..rows], w.pmx.ptr())?;
        Ok(())
    }

    /// Commit kernels of one lane after a batched verify: its rows r0 .. r0 + n advance its GDN conv / state and
    /// PLE history (this device's blocks).
    unsafe fn enqueue_commit_seg(&mut self, ln: usize, r0: usize, n: usize) -> Result<()> {
        let c = self.c.clone();
        let ptrs = self.seq_ptrs(ln);
        let (kn, s, w) = (&self.kn, &self.s, &self.w);
        let cd = c.conv_dim();
        let hd = c.hc_dim();
        for l in self.lo..self.hi {
            let (ly, sp) = (&self.layers[l], &ptrs[l]);
            if ly.ple.is_some() {
                s.launch(&kn.conv_shift, g256(hd), 256, 0, (sp.hist, w.pcn.ptr().add::<f32>(r0 * hd), hd as u32, n as u32,
                    hd as u32, ((c.ple_conv - 1) * c.ple_ngram) as u32))?;
            }
            if let Mixer::Gdn(gd) = &ly.mix {
                s.launch(&kn.conv_shift, g256(cd), 256, 0, (sp.conv, gd.qkv_s.ptr().add::<f32>(r0 * cd), cd as u32, n as u32,
                    cd as u32, (c.ssm_conv - 1) as u32))?;
                s.launch(&kn.gdn_rec, (c.ssm_vh as u32, 4), 128, 0, (DevPtr::NULL, gd.conv_s.ptr().add::<f32>(r0 * cd),
                    gd.ab_s.ptr().add::<f32>(r0 * 2 * c.ssm_vh), (2 * c.ssm_vh) as u32, gd.dt.ptr(), gd.a.ptr(), sp.ssm,
                    n as u32, c.ssm_kh as u32, c.ssm_vh as u32, c.eps, 0u32))?;
            }
        }
        Ok(())
    }

    /// One speculative step for several greedy lanes with the MTP block: `items` = (lane, its Spec). Returns each
    /// lane's new tokens (kept drafts, then the trunk's next token). One lane: `spec_step`.
    pub fn spec_lanes(&mut self, items: &mut [(usize, &mut Spec)], n_draft: usize) -> Result<Vec<Vec<u32>>> {
        self.spec_lanes_ctl(items, n_draft, &[], &mut |_, _, _| {})
    }

    /// [`Fast::spec_lanes`] under the caller's control, as [`Fast::spec_step_ctl`]: `force[i]` (missing = None) is a
    /// token the engine writes itself as item i's next one (that lane verifies its last token alone in this step),
    /// `keep(i, tokens, finite)` may shorten what item i is about to produce (to one token at least); what it cuts is
    /// given back (the lane keeps fewer verify rows).
    pub fn spec_lanes_ctl(&mut self, items: &mut [(usize, &mut Spec)], n_draft: usize, force: &[Option<u32>],
        keep: &mut dyn FnMut(usize, &mut Vec<u32>, bool)) -> Result<Vec<Vec<u32>>> {
        let n = items.len();
        if n == 0 {
            return Ok(vec![]);
        }
        if !self.has_mtp() {
            bail!("spec_lanes without the MTP block");
        }
        let forced = |i: usize| force.get(i).copied().flatten();
        if n == 1 {
            self.select_lane(items[0].0)?;
            return Ok(vec![self.spec_step_ctl(items[0].1, n_draft, forced(0), &mut |o, f| keep(0, o, f))?]);
        }
        if n > MAX_LANES.min(MAX_ROWS) {
            bail!("spec_lanes: {n} lanes");
        }
        let hd = self.c.hc_dim();
        let t0 = std::time::Instant::now();
        // FNEXT_LANE_DRAFTS=d2,d3,d4,..: drafts per lane when 2, 3, 4, .. lanes draft together (default n_draft);
        // at most MAX_ROWS verify rows: each lane gets 1 + D rows
        let n_draft = lane_drafts(n).unwrap_or(n_draft).min(MAX_ROWS / n - 1);
        let want = self.want_logits;
        self.want_logits = false;
        let lanes: Vec<usize> = items.iter().map(|x| x.0).collect();
        // ---- chained drafts: one MTP row per lane still drafting ----
        let mut conf: Vec<bool> = vec![false; n];
        for i in 0..n {
            let ln = lanes[i];
            conf[i] = forced(i).is_none() && *self.lane_draft_p(ln) >= self.p_min;
            if !conf[i] {
                items[i].1.drafts.clear();
            }
            items[i].1.drafts.truncate(n_draft);
        }
        loop {
            let act: Vec<usize> = (0..n).filter(|&i| conf[i] && items[i].1.drafts.len() < n_draft).collect();
            if act.is_empty() {
                break;
            }
            let hin = self.mtp.as_ref().unwrap().hin.ptr();
            let mut segs = vec![];
            for (j, &i) in act.iter().enumerate() {
                let ln = lanes[i];
                let (_, _, hs) = self.lane_mtp_ptrs(ln);
                self.s.memcpy_dtod(hin.add::<f32>(j * hd), hs, hd * 4)?;
                let sp = &items[i].1;
                let pos = self.lane_tokens(ln).len() + sp.drafts.len();
                segs.push((ln, vec![*sp.drafts.last().unwrap_or(&sp.cur)], pos));
            }
            let (tk, pr) = self.mtp_lanes(&segs, 1)?;
            let hb = self.w.hbat.ptr();
            for (j, &i) in act.iter().enumerate() {
                let ln = lanes[i];
                let (_, _, hs) = self.lane_mtp_ptrs(ln);
                self.s.memcpy_dtod(hs, hb.add::<f32>(j * hd), hd * 4)?;
                *self.lane_draft_p(ln) = pr[j];
                conf[i] = pr[j] >= self.p_min;
                if conf[i] {
                    items[i].1.drafts.push(tk[j]);
                }
            }
        }
        let t1 = std::time::Instant::now();
        // ---- one batched verify: each lane [cur, drafts.., padding] (R rows) ----
        let dmax = items.iter().map(|x| x.1.drafts.len()).max().unwrap();
        let rr = dmax + 1;
        let rows = n * rr;
        let mut vrows: Vec<Vec<u32>> = vec![];
        for i in 0..n {
            let sp = &items[i].1;
            let mut v = vec![sp.cur];
            v.extend_from_slice(&sp.drafts);
            let pad = *v.last().unwrap();
            v.resize(rr, pad);
            vrows.push(v);
        }
        for i in 0..n {
            self.select_lane(lanes[i])?;
            self.prepare_at(&vrows[i], i * rr, i)?;
            self.ple_gather_at(i * rr);
        }
        unsafe { std::ptr::write_volatile(self.ple_flag.as_ptr() as *mut u32, 1) };
        std::sync::atomic::fence(Ordering::SeqCst);
        self.plan = (0..n).map(|i| (lanes[i], i * rr, rr)).collect();
        self.cur_rows.store(rows as u32, Ordering::Relaxed);
        let key: Vec<usize> = [K_VERIFY, rr].into_iter().chain(lanes.iter().copied()).collect();
        let r = self.launch_lanes(&key, rows, true);
        self.plan.clear();
        if let Err(e) = r {
            self.want_logits = want;
            return Err(e);
        }
        let pred: Vec<u32> = self.tok_h.as_slice()[..rows].to_vec();
        let bad = self.bad_rows;
        self.step_no.fetch_add(1, Ordering::Relaxed);
        let t2 = std::time::Instant::now();
        // ---- commits: lane i keeps k_i + 1 rows ----
        let mut ks = vec![0usize; n];
        let mut outs: Vec<Vec<u32>> = vec![];
        for i in 0..n {
            let sp = &items[i].1;
            let p = &pred[i * rr..(i + 1) * rr];
            let mut k = 0;
            while k < sp.drafts.len() && sp.drafts[k] == p[k] {
                k += 1;
            }
            ks[i] = k;
            self.spec_stats.0 += sp.drafts.len() as u64;
            self.spec_stats.1 += k as u64;
            if !sp.drafts.is_empty() {
                let a = self.lane_acc(lanes[i]);
                *a = 0.8 * *a + 0.2 * (k as f32 / sp.drafts.len() as f32);
            }
            // what the lane is about to produce: the kept drafts, then the trunk's next token (or the forced one)
            let sp = &items[i].1;
            let mut out: Vec<u32> = sp.drafts[..k].to_vec();
            out.push(forced(i).unwrap_or(p[k]));
            let finite = (bad >> (i * rr)) & ((1u32 << (k + 1)) - 1) == 0;
            let full = out.len();
            keep(i, &mut out, finite);
            if out.is_empty() || out.len() > full {
                self.want_logits = want;
                bail!("spec_lanes: keep left {} of {full} tokens", out.len());
            }
            // tokens given back: the lane keeps fewer rows
            ks[i] = out.len() - 1;
            outs.push(out);
        }
        self.bound_lane_graphs()?;
        let two = self.other.is_some();
        let segs: Vec<(usize, usize, usize)> = (0..n).map(|i| (lanes[i], i * rr, ks[i] + 1)).collect();
        self.each_dev(|f| {
            for &(ln, r0, kk) in &segs {
                let key = vec![K_COMMIT, ln, r0, kk];
                if !f.lane_graphs.contains_key(&key) {
                    let g = f.capture(|m| unsafe { m.enqueue_commit_seg(ln, r0, kk) })?;
                    f.lane_graphs.insert(key.clone(), g);
                }
                f.lane_graphs[&key].launch(&f.s)?;
            }
            if two && !f.is_last() {
                f.s.sync()?;
            }
            Ok(())
        })?;
        let hb = self.w.hbat.ptr();
        for i in 0..n {
            let (_, hp, _) = self.lane_mtp_ptrs(lanes[i]);
            self.s.memcpy_dtod(hp, hb.add::<f32>((i * rr + ks[i]) * hd), hd * 4)?;
            let keep = vrows[i][..ks[i] + 1].to_vec();
            self.lane_tokens_mut(lanes[i]).extend_from_slice(&keep);
        }
        let t3 = std::time::Instant::now();
        // ---- MTP catch-up over the kept positions, K rows per lane (padded) ----
        let kk = ks.iter().max().unwrap() + 1;
        let hin = self.mtp.as_ref().unwrap().hin.ptr();
        let mut segs = vec![];
        for i in 0..n {
            let k = ks[i];
            let next = outs[i][k];
            let mut out: Vec<u32> = outs[i].clone();
            out.resize(kk, next);
            self.s.memcpy_dtod(hin.add::<f32>(i * kk * hd), hb.add::<f32>(i * rr * hd), kk * hd * 4)?;
            let pos0 = self.lane_tokens(lanes[i]).len() - k;
            segs.push((lanes[i], out, pos0));
        }
        let (tk, pr) = self.mtp_lanes(&segs, kk)?;
        let hb = self.w.hbat.ptr();
        for i in 0..n {
            let j = i * kk + ks[i];
            let (_, _, hs) = self.lane_mtp_ptrs(lanes[i]);
            self.s.memcpy_dtod(hs, hb.add::<f32>(j * hd), hd * 4)?;
            *self.lane_draft_p(lanes[i]) = pr[j];
            let sp = &mut items[i].1;
            sp.cur = *outs[i].last().unwrap();
            sp.drafts = vec![tk[j]];
        }
        self.s.sync()?;
        self.want_logits = want;
        if self.dynp.is_some() || self.other.is_some() {
            self.refill()?;
        }
        let t4 = std::time::Instant::now();
        let st = &mut self.spec_ms;
        st[0] += (t1 - t0).as_secs_f64() * 1e3;
        st[1] += (t2 - t1).as_secs_f64() * 1e3;
        st[2] += (t3 - t2).as_secs_f64() * 1e3;
        st[3] += (t4 - t3).as_secs_f64() * 1e3;
        st[4] += 1.0;
        {
            let mut st = self.stats.lock().unwrap();
            st.graph_us += (t2 - t1).as_micros() as u64;
            st.steps += 1;
            st.rows += rows as u64;
        }
        Ok(outs)
    }
}

/// FNEXT_LANE_DRAFTS: drafts per lane for 2, 3, 4, .. lanes (comma list; the last entry covers more lanes).
fn lane_drafts(n: usize) -> Option<usize> {
    static V: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    let v = V.get_or_init(|| std::env::var("FNEXT_LANE_DRAFTS").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default());
    if v.is_empty() || n < 2 { None } else { Some(v[(n - 2).min(v.len() - 1)]) }
}

/// Draft count per speculative step over several lanes, learned online: for each lane count an EMA of tokens per
/// second for every draft count 0 ..= max (0 = a plain step plus the MTP catch-up, which keeps the MTP block in step),
/// the best one used, a neighbour tried every `explore` steps. One lane is not tuned (`spec_step` with the max).
pub struct DraftTuner {
    max: usize,
    explore: u64,
    /// [lanes][drafts] = (tokens per second EMA, samples)
    tp: Vec<Vec<(f64, u32)>>,
    step: u64,
    /// the (lanes, drafts) of the step being timed
    cur: Option<(usize, usize)>,
}

impl DraftTuner {
    pub fn new(max: usize) -> DraftTuner {
        let explore = std::env::var("FNEXT_TUNE_EXPLORE").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
        DraftTuner { max, explore, tp: (0..=MAX_LANES).map(|_| vec![(0.0, 0); max + 1]).collect(), step: 0, cur: None }
    }

    /// Drafts for a step over `n` lanes.
    pub fn pick(&mut self, n: usize) -> usize {
        self.step += 1;
        let d = if n < 2 {
            self.max
        } else {
            let t = &self.tp[n.min(MAX_LANES)];
            // every count measured twice first (largest first: the usual best for code), then the best, with a
            // neighbour now and then
            if let Some(d) = (0..=self.max).rev().find(|&d| t[d].1 < 2) {
                d
            } else {
                let best = (0..=self.max).max_by(|&a, &b| t[a].0.partial_cmp(&t[b].0).unwrap()).unwrap();
                if self.explore > 0 && self.step % self.explore == 0 {
                    let up = (self.step / self.explore) % 2 == 0;
                    if up && best < self.max { best + 1 } else if best > 0 { best - 1 } else { best + 1 }
                } else {
                    best
                }
            }
        };
        self.cur = Some((n, d));
        d
    }

    /// The step picked last made `tokens` in `secs`.
    pub fn record(&mut self, tokens: usize, secs: f64) {
        if let Some((n, d)) = self.cur.take() {
            if n >= 2 && secs > 0.0 {
                let e = &mut self.tp[n.min(MAX_LANES)][d];
                let v = tokens as f64 / secs;
                e.0 = if e.1 == 0 { v } else { 0.8 * e.0 + 0.2 * v };
                e.1 += 1;
            }
        }
    }

    /// (lanes, best drafts, its tokens per second) for every lane count measured so far.
    pub fn summary(&self) -> Vec<(usize, usize, f64)> {
        (2..=MAX_LANES).filter_map(|n| {
            let t = &self.tp[n];
            if t.iter().all(|e| e.1 == 0) {
                return None;
            }
            let best = (0..=self.max).max_by(|&a, &b| t[a].0.partial_cmp(&t[b].0).unwrap()).unwrap();
            Some((n, best, t[best].0))
        }).collect()
    }
}
