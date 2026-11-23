//! G16 expert slots: experts of a layer live either in device slots (resident) or in pinned host memory; the tensor-core GEMMs read
//! every expert through per-layer pointer tables, and a table entry may point at pinned host memory (the kernel reads the expert over
//! PCIe: bits identical, ~0.335 ms an expert at 13.4 GB/s). Residency is therefore a pure performance matter. It changes between graph
//! launches: [`SlotLayer::adapt`] swaps hot host experts with cold resident ones at a fixed resident count,
//! [`set_host_target`] changes the resident count itself (device memory is held in one pool of 30 MiB chunks shared by all layers, freed when empty,
//! so the K/V cache can grow into it); prompt chunks stage a layer's hottest host experts through spare device slots
//! ([`SlotLayer::stage`] / [`SlotLayer::unstage`]).
use crate::*;
use fe_cuda::HostBuffer;

/// Device slots per allocation chunk: 7 x 4,492,800 B = 29.9927 MiB, which the driver rounds to 30 MiB (2 MiB granularity), so a chunk wastes
/// 7.7 KB; a single slot would be rounded to 6 MiB (40% waste) and blocks of 4 to 18 MiB (5%, and 0-3 slots per layer unused).
pub const PER_CHUNK: usize = 7;

struct PoolSt {
    chunks: Vec<Option<DeviceBuffer<u8>>>,
    /// (layer id, expert) of the resident in each slot
    owner: Vec<Option<(usize, usize)>>,
    layers: usize,
    inflight: usize,
}

/// ONE pool of expert-sized device slots for the whole model (every layer's resident experts and the spare slots of the pipelined adapt live
/// here, so no device memory is held per layer that is not an expert in use). Chunks are allocated on demand and freed when empty;
/// [`SlotPool::compact`] packs residents so that parked experts return their bytes.
pub struct SlotPool {
    pub slot_bytes: usize,
    pub max_inflight: usize,
    st: std::sync::Mutex<PoolSt>,
}

impl SlotPool {
    pub fn new() -> std::sync::Arc<SlotPool> {
        let slot_bytes: usize = PrefillLayer::expert_bytes(ExpertFmt::Nvfp4).iter().sum();
        let max_inflight = 8;
        std::sync::Arc::new(SlotPool { slot_bytes, max_inflight, st: std::sync::Mutex::new(PoolSt { chunks: vec![], owner: vec![], layers: 0, inflight: 0 }) })
    }
    fn register(&self) -> usize {
        let mut g = self.st.lock().unwrap();
        g.layers += 1;
        g.layers - 1
    }
    fn ptr(&self, slot: usize) -> DevPtr {
        let g = self.st.lock().unwrap();
        g.chunks[slot / PER_CHUNK].as_ref().expect("slot chunk").at((slot % PER_CHUNK) * self.slot_bytes)
    }
    /// A free slot (lowest index first, so the early chunks fill up); with `grow` a new chunk is allocated when none is free.
    fn alloc(&self, owner: (usize, usize), grow: bool) -> Result<Option<usize>> {
        let mut g = self.st.lock().unwrap();
        if let Some(i) = (0..g.chunks.len() * PER_CHUNK).find(|&i| g.chunks[i / PER_CHUNK].is_some() && g.owner[i].is_none()) {
            g.owner[i] = Some(owner);
            return Ok(Some(i));
        }
        if !grow {
            return Ok(None);
        }
        let c = match g.chunks.iter().position(|b| b.is_none()) {
            Some(c) => c,
            None => {
                g.chunks.push(None);
                g.owner.extend([None; PER_CHUNK]);
                g.chunks.len() - 1
            }
        };
        g.chunks[c] = Some(DeviceBuffer::<u8>::alloc(PER_CHUNK * self.slot_bytes)?);
        g.owner[c * PER_CHUNK] = Some(owner);
        Ok(Some(c * PER_CHUNK))
    }
    fn set_owner(&self, slot: usize, o: Option<(usize, usize)>) {
        self.st.lock().unwrap().owner[slot] = o;
    }
    /// Device bytes held by the pool (live chunks).
    pub fn bytes(&self) -> usize {
        self.st.lock().unwrap().chunks.iter().filter(|c| c.is_some()).count() * PER_CHUNK * self.slot_bytes
    }
    pub fn free_slots(&self) -> usize {
        let g = self.st.lock().unwrap();
        (0..g.chunks.len() * PER_CHUNK).filter(|&i| g.chunks[i / PER_CHUNK].is_some() && g.owner[i].is_none()).count()
    }
    /// Make sure at least `n` slots are free (the spare slots of the pipelined adapt): allocates chunks as needed.
    pub fn reserve_free(&self, n: usize) -> Result<()> {
        while self.free_slots() < n {
            let mut g = self.st.lock().unwrap();
            let c = match g.chunks.iter().position(|b| b.is_none()) {
                Some(c) => c,
                None => {
                    g.chunks.push(None);
                    g.owner.extend([None; PER_CHUNK]);
                    g.chunks.len() - 1
                }
            };
            g.chunks[c] = Some(DeviceBuffer::<u8>::alloc(PER_CHUNK * self.slot_bytes)?);
        }
        Ok(())
    }
    /// Free chunks that hold no resident and are not needed to keep `keep_free` free slots. Returns chunks freed.
    pub fn trim(&self, keep_free: usize) -> usize {
        let mut g = self.st.lock().unwrap();
        let mut freed = 0;
        for c in 0..g.chunks.len() {
            if g.chunks[c].is_none() || (0..PER_CHUNK).any(|k| g.owner[c * PER_CHUNK + k].is_some()) {
                continue;
            }
            let free_total = (0..g.chunks.len() * PER_CHUNK).filter(|&i| g.chunks[i / PER_CHUNK].is_some() && g.owner[i].is_none()).count();
            if free_total - PER_CHUNK < keep_free {
                break;
            }
            g.chunks[c] = None;
            freed += 1;
        }
        freed
    }
    fn try_start(&self) -> bool {
        let mut g = self.st.lock().unwrap();
        if g.inflight < self.max_inflight {
            g.inflight += 1;
            true
        } else {
            false
        }
    }
    pub fn inflight(&self) -> usize {
        self.st.lock().unwrap().inflight
    }
    /// Free slots the pool keeps for the pipelined adapt (none by default: the leftovers of the 7-slot chunks are the spares, a new
    /// chunk is only taken when none is free and no other swap is in flight, and only with 300 MiB of free memory).
    pub fn reserve(&self) -> usize {
        0
    }
    fn done(&self) {
        let mut g = self.st.lock().unwrap();
        g.inflight = g.inflight.saturating_sub(1);
    }

    /// Pack the residents of `layers` into as few chunks as possible (stream idle) and free the chunks that empty, keeping `keep_free` free
    /// slots. Returns chunks freed; `moved` counts the slots copied.
    pub fn compact(&self, layers: &mut [&mut SlotLayer], s: &Stream, keep_free: usize, moved: &mut usize) -> Result<usize> {
        s.sync()?;
        let idx: std::collections::HashMap<usize, usize> = layers.iter().enumerate().map(|(i, l)| (l.id, i)).collect();
        let mut freed = 0;
        let mut dirty = vec![false; layers.len()];
        loop {
            let moves: Vec<(usize, usize, (usize, usize))>;
            let b;
            {
                let mut g = self.st.lock().unwrap();
                let live: Vec<usize> = (0..g.chunks.len()).filter(|&c| g.chunks[c].is_some()).collect();
                let used = |g: &PoolSt, c: usize| (0..PER_CHUNK).filter(|&k| g.owner[c * PER_CHUNK + k].is_some()).count();
                let Some(&bb) = live.iter().min_by_key(|&&c| used(&g, c)) else { break };
                b = bb;
                let u = used(&g, b);
                let free_other: usize = live.iter().filter(|&&c| c != b).map(|&c| PER_CHUNK - used(&g, c)).sum();
                if u == 0 {
                    let free_total = free_other + PER_CHUNK;
                    if free_total - PER_CHUNK < keep_free {
                        break;
                    }
                    g.chunks[b] = None;
                    freed += 1;
                    continue;
                }
                if free_other < u + keep_free {
                    break;
                }
                // destinations: free slots of the fullest other chunks first
                let mut order: Vec<usize> = live.iter().cloned().filter(|&c| c != b && used(&g, c) < PER_CHUNK).collect();
                order.sort_by_key(|&c| std::cmp::Reverse(used(&g, c)));
                let mut mv = vec![];
                'outer: for k in 0..PER_CHUNK {
                    let src = b * PER_CHUNK + k;
                    let Some(o) = g.owner[src] else { continue };
                    for &c in &order {
                        if let Some(dst) = (0..PER_CHUNK).map(|k2| c * PER_CHUNK + k2).find(|&i| g.owner[i].is_none()) {
                            g.owner[dst] = Some(o);
                            g.owner[src] = None;
                            mv.push((src, dst, o));
                            continue 'outer;
                        }
                    }
                    anyhow::bail!("compact: no free destination");
                }
                moves = mv;
            }
            for (src, dst, (l, e)) in moves {
                fe_cuda::memcpy_dtod(self.ptr(dst), self.ptr(src), self.slot_bytes)?;
                let li = idx[&l];
                layers[li].loc[e] = Loc::Dev(dst);
                dirty[li] = true;
                *moved += 1;
            }
            self.st.lock().unwrap().chunks[b] = None;
            freed += 1;
        }
        for (i, l) in layers.iter_mut().enumerate() {
            if dirty[i] {
                l.rebuild_home();
                unsafe { l.push(s, false)? };
            }
        }
        s.sync()?;
        Ok(freed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loc {
    Dev(usize),
    Host(usize),
}

#[derive(Debug, Default, Clone)]
pub struct AdaptStats {
    /// experts swapped host <-> device
    pub swaps: usize,
    /// uses of host-resident / all experts by the GEMMs since the last adapt (exact deltas)
    pub host_use: u64,
    pub total_use: u64,
}

#[derive(Debug, Default, Clone)]
pub struct TargetStats {
    pub evicted: usize,
    pub promoted: usize,
    pub blocks_freed: usize,
    pub blocks_allocated: usize,
    pub moved: usize,
    pub ms: f64,
}

/// Spare device slots shared by all layers for prompt staging.
pub struct SparePool {
    pub blob: DeviceBuffer<u8>,
    pub n: usize,
    pub slot_bytes: usize,
}

impl SparePool {
    pub fn new(n: usize) -> Result<SparePool> {
        let slot_bytes: usize = PrefillLayer::expert_bytes(ExpertFmt::Nvfp4).iter().sum();
        Ok(SparePool { blob: DeviceBuffer::alloc((n * slot_bytes).max(16))?, n, slot_bytes })
    }
}

pub struct SlotLayer {
    pub pl: PrefillLayer,
    sizes: [usize; 4],
    slot_bytes: usize,
    pool: std::sync::Arc<SlotPool>,
    id: usize,
    host: Vec<Option<HostBuffer<u8>>>,
    free_host: Vec<usize>,
    loc: Vec<Loc>,
    /// the tables as the kernels see them while unstaged / staged (pinned, never modified while a copy of them may be in flight)
    tab_home: HostBuffer<u64>,
    tab_staged: HostBuffer<u64>,
    pub total_swaps: usize,
    /// counter values written back by the last adapt (so deltas since then are exact)
    base: Vec<u32>,
    /// pipelined adapt: swaps still to do (hot host expert, cold resident victim) and the one in flight
    plan: Vec<(usize, usize)>,
    pend: Option<Pend>,
    /// swaps the pipelined adapt had to do synchronously (no free device slot to prepare in)
    pub fallbacks: usize,
    ev: [fe_cuda::Event; 2],
}

/// One swap in flight on the copy stream. Stage 0: the hot expert is being copied into a spare device slot (the compute stream does not
/// read the spare). Stage 1: the tables point at the spare; the victim is being copied into the hot expert's old host slot while it is
/// still read from its device slot. When that copy is done the tables flip the victim to the host slot and its device slot is free.
#[derive(Clone, Copy, Debug)]
struct Pend {
    hot: usize,
    vic: usize,
    host_slot: usize,
    spare: usize,
    vic_slot: usize,
    stage: u8,
}

fn pointers(sizes: &[usize; 4], base: u64) -> [u64; 4] {
    [base, base + sizes[0] as u64, base + (sizes[0] + sizes[1]) as u64, base + (sizes[0] + sizes[1] + sizes[2]) as u64]
}

impl Kern {
    /// Turn a fully device-resident NVFP4 layer (from `repack_nvfp4`) into a slot layer: experts with `resident[e]` are copied into
    /// device slots, the others into pinned host slots; the full-size buffers are freed.
    pub fn make_slot_layer(&self, s: &Stream, pl: PrefillLayer, resident: &[bool]) -> Result<SlotLayer> {
        self.make_slot_layer_in(s, pl, resident, &SlotPool::new())
    }

    /// As [`make_slot_layer`](Self::make_slot_layer), with the resident experts in the shared `pool` (all layers of a model share one).
    pub fn make_slot_layer_in(&self, s: &Stream, mut pl: PrefillLayer, resident: &[bool], pool: &std::sync::Arc<SlotPool>) -> Result<SlotLayer> {
        ensure!(pl.fmt == ExpertFmt::Nvfp4 && resident.len() == N_EXPERTS, "make_slot_layer: NVFP4 layer, 128 flags");
        let sizes = PrefillLayer::expert_bytes(pl.fmt);
        let slot_bytes: usize = sizes.iter().sum();
        let mut sl = SlotLayer {
            sizes,
            slot_bytes,
            pool: pool.clone(),
            id: pool.register(),
            host: vec![],
            free_host: vec![],
            loc: vec![Loc::Dev(0); N_EXPERTS],
            tab_home: HostBuffer::<u64>::alloc(4 * N_EXPERTS)?,
            tab_staged: HostBuffer::<u64>::alloc(4 * N_EXPERTS)?,
            total_swaps: 0,
            base: vec![0; N_EXPERTS],
            plan: vec![],
            pend: None,
            fallbacks: 0,
            ev: [fe_cuda::Event::new_no_timing()?, fe_cuda::Event::new_no_timing()?],
            // placeholder: the real layer is moved in below
            pl: PrefillLayer::shell(&pl),
        };
        let ml = crate::env_flag("INK_MEM_LOG");
        let used = || fe_cuda::mem_info().map(|(f, t)| (t - f) as f64 / 1048576.0).unwrap_or(0.0);
        let u0 = used();
        for e in 0..N_EXPERTS {
            let src = [pl.keep[0].at(e * sizes[0]), pl.keep[1].at(e * sizes[1]), pl.keep[2].at(e * sizes[2]), pl.keep[3].at(e * sizes[3])];
            if resident[e] {
                let d = sl.alloc_dev_slot(e)?;
                let dp = sl.dev_ptr(d);
                let mut off = 0usize;
                for i in 0..4 {
                    fe_cuda::memcpy_dtod(dp.byte_add(off), src[i], sizes[i])?;
                    off += sizes[i];
                }
                sl.loc[e] = Loc::Dev(d);
            } else {
                let h = sl.alloc_host_slot()?;
                let mut off = 0usize;
                for i in 0..4 {
                    unsafe {
                        s.memcpy_dtoh(std::slice::from_raw_parts_mut(sl.host[h].as_mut().unwrap().as_mut_ptr().add(off), sizes[i]), src[i])?;
                    }
                    off += sizes[i];
                }
                sl.loc[e] = Loc::Host(h);
            }
        }
        s.sync()?;
        let u1 = used();
        pl.keep.clear(); // frees the full-size buffers
        sl.pl = pl;
        sl.rebuild_home();
        unsafe { sl.push(s, false)? };
        s.sync()?;
        if ml {
            eprintln!("memlog: make_slot_layer: used {u0:.1} MiB at entry (full layer resident), {u1:.1} after copies, {:.1} after freeing the full layer; pool {:.1} MiB", used(), sl.pool.bytes() as f64 / 1048576.0);
        }
        Ok(sl)
    }
}

impl SlotLayer {
    pub fn n_host(&self) -> usize {
        self.loc.iter().filter(|l| matches!(l, Loc::Host(_))).count()
    }
    pub fn n_dev(&self) -> usize {
        N_EXPERTS - self.n_host()
    }
    pub fn is_host(&self, e: usize) -> bool {
        matches!(self.loc[e], Loc::Host(_))
    }
    pub fn host_experts(&self) -> Vec<usize> {
        (0..N_EXPERTS).filter(|&e| self.is_host(e)).collect()
    }
    /// Device bytes held by the shared pool (all layers).
    pub fn pool_bytes(&self) -> usize {
        self.pool.bytes()
    }
    /// Device bytes of this layer's resident experts (the pool holds them; [`SlotPool::bytes`] is what the whole pool holds).
    pub fn dev_bytes(&self) -> usize {
        self.n_dev() * self.slot_bytes
    }
    fn dev_ptr(&self, slot: usize) -> DevPtr {
        self.pool.ptr(slot)
    }
    fn alloc_dev_slot(&mut self, e: usize) -> Result<usize> {
        Ok(self.pool.alloc((self.id, e), true)?.expect("grow"))
    }
    fn alloc_host_slot(&mut self) -> Result<usize> {
        if let Some(h) = self.free_host.pop() {
            return Ok(h);
        }
        self.host.push(Some(HostBuffer::<u8>::alloc(self.slot_bytes)?));
        Ok(self.host.len() - 1)
    }
    fn ptrs_of(&self, loc: Loc) -> [u64; 4] {
        match loc {
            Loc::Dev(i) => pointers(&self.sizes, self.dev_ptr(i).0),
            Loc::Host(j) => pointers(&self.sizes, self.host[j].as_ref().unwrap().dev().0),
        }
    }
    fn rebuild_home(&mut self) {
        for e in 0..N_EXPERTS {
            let p = self.ptrs_of(self.loc[e]);
            let t = self.tab_home.as_mut_slice();
            for k in 0..4 {
                t[k * N_EXPERTS + e] = p[k];
            }
        }
    }
    /// Enqueue the 4 table copies (pinned host mirror -> device tables) on `s`.
    unsafe fn push(&self, s: &Stream, staged: bool) -> Result<()> {
        let t = if staged { &self.tab_staged } else { &self.tab_home };
        let dst = [&self.pl.tw13, &self.pl.ts13, &self.pl.tw2, &self.pl.ts2];
        for k in 0..4 {
            unsafe { s.memcpy_htod(dst[k].ptr(), &t.as_slice()[k * N_EXPERTS..(k + 1) * N_EXPERTS])? };
        }
        Ok(())
    }

    /// Use counts since the last adapt (host copy; syncs `s`).
    pub fn use_counts(&self, s: &Stream) -> Result<Vec<u32>> {
        s.sync()?;
        Ok(self.pl.use_cnt.to_vec()?)
    }

    /// (host-expert uses, all expert uses) since the last adapt, without changing anything (syncs `s`).
    pub fn peek_use(&self, s: &Stream) -> Result<(u64, u64)> {
        s.sync()?;
        let cnt = self.pl.use_cnt.to_vec()?;
        let d = |e: usize| (cnt[e] - self.base[e].min(cnt[e])) as u64;
        Ok(((0..N_EXPERTS).filter(|&e| self.is_host(e)).map(d).sum(), (0..N_EXPERTS).map(d).sum()))
    }

    fn swap_in(&mut self, s: &Stream, h: usize, d: usize) -> Result<()> {
        let (Loc::Host(hs_), Loc::Dev(ds)) = (self.loc[h], self.loc[d]) else { return Ok(()) };
        let t = self.alloc_host_slot()?;
        let dp = self.dev_ptr(ds);
        unsafe {
            // victim -> host slot t, hot expert -> the victim's device slot
            s.memcpy_dtoh(std::slice::from_raw_parts_mut(self.host[t].as_mut().unwrap().as_mut_ptr(), self.slot_bytes), dp)?;
            s.memcpy_htod(dp, std::slice::from_raw_parts(self.host[hs_].as_ref().unwrap().as_ptr(), self.slot_bytes))?;
        }
        s.sync()?;
        self.free_host.push(hs_);
        self.loc[h] = Loc::Dev(ds);
        self.pool.set_owner(ds, Some((self.id, h)));
        self.loc[d] = Loc::Host(t);
        Ok(())
    }

    /// Between graph launches (the stream must be idle): swap up to `max_swaps` hot host experts with the coldest resident ones when the
    /// host expert was used clearly more since the last call (count > 2 x the victim's + 1), then halve the counters (recency).
    pub fn adapt(&mut self, s: &Stream, max_swaps: usize) -> Result<AdaptStats> {
        s.sync()?;
        let mut cnt = self.pl.use_cnt.to_vec()?;
        let mut st = AdaptStats::default();
        st.total_use = (0..N_EXPERTS).map(|e| (cnt[e] - self.base[e].min(cnt[e])) as u64).sum();
        st.host_use = (0..N_EXPERTS).filter(|&e| self.is_host(e)).map(|e| (cnt[e] - self.base[e].min(cnt[e])) as u64).sum();
        let mut hosts: Vec<usize> = (0..N_EXPERTS).filter(|&e| self.is_host(e)).collect();
        let mut devs: Vec<usize> = (0..N_EXPERTS).filter(|&e| !self.is_host(e)).collect();
        hosts.sort_by_key(|&e| std::cmp::Reverse(cnt[e]));
        devs.sort_by_key(|&e| cnt[e]);
        for (&h, &d) in hosts.iter().zip(devs.iter()) {
            if st.swaps >= max_swaps || cnt[h] < 2 || cnt[h] <= 2 * cnt[d] + 1 {
                break;
            }
            self.swap_in(s, h, d)?;
            st.swaps += 1;
        }
        self.total_swaps += st.swaps;
        for c in cnt.iter_mut() {
            *c /= 2;
        }
        self.pl.use_cnt.copy_from(&cnt)?;
        self.base = cnt.clone();
        if st.swaps > 0 {
            self.rebuild_home();
        }
        unsafe { self.push(s, false)? };
        s.sync()?;
        Ok(st)
    }

    fn set_entry(&mut self, e: usize) {
        let p = self.ptrs_of(self.loc[e]);
        let t = self.tab_home.as_mut_slice();
        for k in 0..4 {
            t[k * N_EXPERTS + e] = p[k];
        }
    }
    /// Pipelined adapt, part 1 (stream idle): like [`adapt`](Self::adapt) it reads and halves the counters and picks the swaps, but only
    /// queues them; [`adapt_step`](Self::adapt_step) then carries them out in the background, one transition per call.
    pub fn adapt_plan(&mut self, s: &Stream, max_swaps: usize) -> Result<AdaptStats> {
        s.sync()?;
        let mut cnt = self.pl.use_cnt.to_vec()?;
        let mut st = AdaptStats::default();
        st.total_use = (0..N_EXPERTS).map(|e| (cnt[e] - self.base[e].min(cnt[e])) as u64).sum();
        st.host_use = (0..N_EXPERTS).filter(|&e| self.is_host(e)).map(|e| (cnt[e] - self.base[e].min(cnt[e])) as u64).sum();
        let busy = |sl: &SlotLayer, e: usize| sl.pend.map_or(false, |p| p.hot == e || p.vic == e);
        let mut hosts: Vec<usize> = (0..N_EXPERTS).filter(|&e| self.is_host(e) && !busy(self, e)).collect();
        let mut devs: Vec<usize> = (0..N_EXPERTS).filter(|&e| !self.is_host(e) && !busy(self, e)).collect();
        hosts.sort_by_key(|&e| std::cmp::Reverse(cnt[e]));
        devs.sort_by_key(|&e| cnt[e]);
        self.plan.clear();
        for (&h, &d) in hosts.iter().zip(devs.iter()) {
            if self.plan.len() >= max_swaps || cnt[h] < 2 || cnt[h] <= 2 * cnt[d] + 1 {
                break;
            }
            self.plan.push((h, d));
        }
        st.swaps = self.plan.len();
        self.plan.reverse(); // pop() takes the hottest first
        self.total_swaps += st.swaps;
        for c in cnt.iter_mut() {
            *c /= 2;
        }
        self.pl.use_cnt.copy_from(&cnt)?;
        self.base = cnt.clone();
        Ok(st)
    }

    /// True while swaps are queued or in flight.
    pub fn adapt_busy(&self) -> bool {
        self.pend.is_some() || !self.plan.is_empty()
    }

    /// Pipelined adapt, part 2: call at the start of every forward (compute stream `s` idle, copy stream `cs`). Moves the layer's swap along
    /// by one transition; the PCIe copies run on `cs` while the next graph runs on `s`. The compute stream never reads a slot that a copy is
    /// writing, and never waits for a copy that is not finished, so the bits do not depend on the timing.
    pub fn adapt_step(&mut self, s: &Stream, cs: &Stream) -> Result<()> {
        if let Some(mut p) = self.pend {
            // the copy of the previous call must be done before the tables change
            s.wait(&self.ev[0])?;
            s.sync()?;
            if p.stage == 0 {
                // flip A: the hot expert reads from the spare slot; its old host slot receives the victim next
                self.loc[p.hot] = Loc::Dev(p.spare);
                self.set_entry(p.hot);
                unsafe { self.push(s, false)? };
                s.record(&self.ev[1])?;
                cs.wait(&self.ev[1])?;
                unsafe {
                    cs.memcpy_dtoh(std::slice::from_raw_parts_mut(self.host[p.host_slot].as_mut().unwrap().as_mut_ptr(), self.slot_bytes), self.dev_ptr(p.vic_slot))?;
                }
                cs.record(&self.ev[0])?;
                p.stage = 1;
                self.pend = Some(p);
                return Ok(());
            }
            // flip C: the victim reads from the host slot, its device slot is free again
            self.loc[p.vic] = Loc::Host(p.host_slot);
            self.pool.set_owner(p.vic_slot, None);
            self.pool.done();
            self.set_entry(p.vic);
            unsafe { self.push(s, false)? };
            s.record(&self.ev[1])?;
            cs.wait(&self.ev[1])?;
            self.pend = None;
        }
        while let Some((hot, vic)) = self.plan.pop() {
            let (Loc::Host(hs), Loc::Dev(vs)) = (self.loc[hot], self.loc[vic]) else { continue };
            if !self.pool.try_start() {
                // the shared spare slots are all in use by other layers' swaps: try again on the next forward
                self.plan.push((hot, vic));
                break;
            }
            let owner = (self.id, hot);
            let room = || fe_cuda::mem_info().map(|(f, _)| f > (300usize << 20)).unwrap_or(false);
            // a free slot of the shared pool (the rounding leftovers and the reserve); a new chunk only when nothing else is in flight
            let spare = match self.pool.alloc(owner, false)? {
                Some(x) => Some(x),
                None if self.pool.inflight() > 1 => {
                    // the other layers' swaps hold the free slots: wait for them
                    self.pool.done();
                    self.plan.push((hot, vic));
                    break;
                }
                None if room() => self.pool.alloc(owner, true).ok().flatten(),
                None => None,
            };
            let Some(sp) = spare else {
                // no free slot and no memory for a new chunk: do this one swap synchronously, in place
                self.pool.done();
                s.sync()?;
                self.fallbacks += 1;
                self.swap_in(s, hot, vic)?;
                self.rebuild_home();
                unsafe { self.push(s, false)? };
                s.sync()?;
                continue;
            };
            unsafe {
                cs.memcpy_htod(self.dev_ptr(sp), std::slice::from_raw_parts(self.host[hs].as_ref().unwrap().as_ptr(), self.slot_bytes))?;
            }
            cs.record(&self.ev[0])?;
            self.pend = Some(Pend { hot, vic, host_slot: hs, spare: sp, vic_slot: vs, stage: 0 });
            break;
        }
        Ok(())
    }

    /// Finish everything queued or in flight (both streams are synchronised). Call before anything else that changes the residency.
    pub fn adapt_finish(&mut self, s: &Stream, cs: &Stream) -> Result<()> {
        self.plan.clear();
        while self.pend.is_some() {
            cs.sync()?;
            self.adapt_step(s, cs)?;
        }
        cs.sync()?;
        s.sync()?;
        Ok(())
    }

    /// Evict the resident experts with `want_host[e]` to pinned host slots (stream idle). The tables are updated by [`finish_tables`](Self::finish_tables).
    fn evict_to(&mut self, s: &Stream, want_host: &[bool], st: &mut TargetStats) -> Result<()> {
        s.sync()?;
        let mut freed = vec![];
        for e in 0..N_EXPERTS {
            if want_host[e] && !self.is_host(e) {
                let Loc::Dev(ds) = self.loc[e] else { continue };
                let h = self.alloc_host_slot()?;
                unsafe {
                    s.memcpy_dtoh(std::slice::from_raw_parts_mut(self.host[h].as_mut().unwrap().as_mut_ptr(), self.slot_bytes), self.dev_ptr(ds))?;
                }
                self.loc[e] = Loc::Host(h);
                freed.push(ds);
                st.evicted += 1;
            }
        }
        s.sync()?;
        for ds in freed {
            self.pool.set_owner(ds, None);
        }
        Ok(())
    }

    /// Promote the host experts with `!want_host[e]` into pool slots (allocating chunks as needed; stream idle).
    fn promote_to(&mut self, s: &Stream, want_host: &[bool], st: &mut TargetStats) -> Result<()> {
        let mut hs = vec![];
        for e in 0..N_EXPERTS {
            if !want_host[e] && self.is_host(e) {
                let Loc::Host(h) = self.loc[e] else { continue };
                let d = self.alloc_dev_slot(e)?;
                unsafe {
                    s.memcpy_htod(self.dev_ptr(d), std::slice::from_raw_parts(self.host[h].as_ref().unwrap().as_ptr(), self.slot_bytes))?;
                }
                self.loc[e] = Loc::Dev(d);
                hs.push(h);
                st.promoted += 1;
            }
        }
        s.sync()?;
        self.free_host.extend(hs);
        Ok(())
    }

    fn finish_tables(&mut self, s: &Stream) -> Result<()> {
        self.rebuild_home();
        unsafe { self.push(s, false)? };
        s.sync()?;
        Ok(())
    }

    /// Make exactly the experts with `want_host[e]` host-resident (stream idle): evict, compact the pool, promote. Single-layer convenience
    /// (tests); the engine uses [`set_host_target`] over all layers.
    pub fn set_hosts(&mut self, s: &Stream, want_host: &[bool]) -> Result<TargetStats> {
        let t0 = std::time::Instant::now();
        let mut st = TargetStats::default();
        self.evict_to(s, want_host, &mut st)?;
        let pool = self.pool.clone();
        st.blocks_freed += pool.compact(&mut [&mut *self], s, 0, &mut st.moved)?;
        let before = pool.bytes();
        self.promote_to(s, want_host, &mut st)?;
        st.blocks_allocated = pool.bytes().saturating_sub(before) / (PER_CHUNK * self.slot_bytes);
        self.finish_tables(s)?;
        st.ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(st)
    }

    /// Prompt staging: copy up to `max_n` of the layer's hottest host experts (by the last adapt's counters) into spare device slots
    /// `first ..` of `pool` and point the tables there; the rest stay read in place. Enqueued on `s` (the copy stream). Record an event
    /// on `s` and make the compute stream wait for it. The previous use of this layer's staging must have completed.
    pub fn stage(&mut self, s: &Stream, pool: &SparePool, first: usize, max_n: usize) -> Result<()> {
        let mut hosts = self.host_experts();
        hosts.sort_by_key(|&e| std::cmp::Reverse(self.base[e]));
        hosts.truncate(max_n.min(pool.n.saturating_sub(first)));
        let mut staged = self.tab_home.as_slice().to_vec();
        for (i, &e) in hosts.iter().enumerate() {
            let Loc::Host(j) = self.loc[e] else { continue };
            let dst = pool.blob.at((first + i) * pool.slot_bytes);
            unsafe { s.memcpy_htod(dst, std::slice::from_raw_parts(self.host[j].as_ref().unwrap().as_ptr(), self.slot_bytes))? };
            let p = pointers(&self.sizes, dst.0);
            for k in 0..4 {
                staged[k * N_EXPERTS + e] = p[k];
            }
        }
        self.tab_staged.as_mut_slice().copy_from_slice(&staged);
        unsafe { self.push(s, true)? };
        Ok(())
    }

    /// Point the tables back at the home locations (enqueue on the compute stream after the layer's MoE).
    pub fn unstage(&self, s: &Stream) -> Result<()> {
        unsafe { self.push(s, false) }
    }
}

/// Change the number of host-resident experts across ALL layers to `n_host` (stream idle): the coldest `n_host` experts by a blend of
/// the layers' current use counters and the heat prior `prior` ([24][128] pick shares) become host experts (each layer keeps at least
/// `min_resident` on the device); evicts / promotes accordingly and frees (or allocates) device blocks. Returns the summed stats.
pub fn set_host_target(layers: &mut [&mut SlotLayer], s: &Stream, n_host: usize, prior: &[Vec<f64>], min_resident: usize) -> Result<TargetStats> {
    s.sync()?;
    let t0 = std::time::Instant::now();
    let mut score: Vec<(f64, usize, usize)> = vec![];
    for (l, sl) in layers.iter().enumerate() {
        let cnt = sl.pl.use_cnt.to_vec()?;
        let tot: f64 = cnt.iter().map(|&c| c as f64).sum();
        for e in 0..N_EXPERTS {
            let cs = if tot > 0.0 { cnt[e] as f64 / tot } else { 0.0 };
            score.push((cs + 0.25 * prior[l][e], l, e));
        }
    }
    score.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut want = vec![vec![false; N_EXPERTS]; layers.len()];
    let mut cnt = vec![0usize; layers.len()];
    let mut left = n_host;
    for &(_, l, e) in &score {
        if left == 0 {
            break;
        }
        if cnt[l] < N_EXPERTS - min_resident {
            want[l][e] = true;
            cnt[l] += 1;
            left -= 1;
        }
    }
    let mut tot = TargetStats::default();
    // evict everywhere first, pack the pool and free chunks, then promote (allocates)
    for (l, sl) in layers.iter_mut().enumerate() {
        let ev: Vec<bool> = (0..N_EXPERTS).map(|e| want[l][e] || sl.is_host(e)).collect();
        sl.evict_to(s, &ev, &mut tot)?;
    }
    let pool = layers[0].pool.clone();
    let keep = if n_host > 0 { pool.reserve() } else { 0 };
    tot.blocks_freed += pool.compact(layers, s, keep, &mut tot.moved)?;
    let before = pool.bytes();
    for (l, sl) in layers.iter_mut().enumerate() {
        sl.promote_to(s, &want[l], &mut tot)?;
    }
    if keep > 0 {
        let _ = pool.reserve_free(keep); // the shared spare slots of the pipelined adapt (best effort)
    }
    for sl in layers.iter_mut() {
        sl.finish_tables(s)?;
    }
    tot.blocks_allocated = pool.bytes().saturating_sub(before) / (PER_CHUNK * layers[0].slot_bytes);
    if crate::env_flag("INK_SLOT_LOG") {
        let res: usize = layers.iter().map(|l| l.n_dev()).sum();
        eprintln!("ink slots: pool holds {:.2} GiB for {res} resident experts ({:.2} GiB needed)", pool.bytes() as f64 / 1073741824.0, (res * layers[0].slot_bytes) as f64 / 1073741824.0);
    }
    tot.ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok(tot)
}

/// Resident set per layer from a heat ranking: `order[l]` lists the experts of layer `l` from coldest to hottest; the `host_per_layer`
/// coldest become host experts.
pub fn resident_from_order(order: &[usize], host_per_layer: usize) -> Vec<bool> {
    let mut r = vec![true; N_EXPERTS];
    for &e in order.iter().take(host_per_layer) {
        r[e] = false;
    }
    r
}

/// Per-layer pick shares `[24][128]`: JSON `{"share": [[...]]}`, from a file.
pub fn load_heat_share(path: &str) -> Result<Vec<Vec<f64>>> {
    parse_heat_share(&std::fs::read(path)?)
}

/// The same from JSON bytes (the table the crate ships: crates/fe-ink/heat/heat_share.json).
pub fn parse_heat_share(json: &[u8]) -> Result<Vec<Vec<f64>>> {
    let v: serde_json::Value = serde_json::from_slice(json)?;
    let o = v["share"].as_array().context("heat share: no `share` array")?;
    Ok(o.iter().map(|l| l.as_array().unwrap().iter().map(|e| e.as_f64().unwrap()).collect()).collect())
}

/// Resident flags per layer when the `n_host` coldest experts ANYWHERE (by share) live on the host (uneven per layer): `[layer][expert]`,
/// true = device resident. A layer keeps at least 64 experts on the device.
pub fn resident_global(share: &[Vec<f64>], n_host: usize) -> Vec<Vec<bool>> {
    let mut all: Vec<(f64, usize, usize)> = share.iter().enumerate().flat_map(|(l, v)| v.iter().enumerate().map(move |(e, &s)| (s, l, e))).collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut r = vec![vec![true; N_EXPERTS]; share.len()];
    let mut cnt = vec![0usize; share.len()];
    let mut left = n_host;
    for (_, l, e) in all {
        if left == 0 {
            break;
        }
        if cnt[l] < N_EXPERTS - 64 {
            r[l][e] = false;
            cnt[l] += 1;
            left -= 1;
        }
    }
    r
}
