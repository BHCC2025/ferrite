//! Micro-batch pipeline: a multi-row pass split in two halves A / B. The CPU expert lane runs on a worker thread, so while it computes the
//! routed experts of half A (layer l) the GPUs run half B's layer l attention / dense work, and vice versa: the pass costs about
//! max(GPU, CPU) instead of their sum. Rows are independent in every kernel (the row-invariance gate), B's layer l only needs the caches A
//! wrote in layer l (same stream, enqueued before), so the bits equal the single pass.
use super::*;
use fe_cuda::memcpy_dtod;
use std::sync::mpsc;

/// The per-row scratch of the engine (everything a pass of `nr` rows keeps between layers); the second half's copy lives parked in
/// `Engine::sc2` and is swapped with the engine's own fields around that half's stages.
pub struct Scratch {
    pub(super) act: Q8Act,
    pub(super) hq: Q8Act,
    pub(super) ug: DeviceBuffer<u16>,
    pub(super) taps: DeviceBuffer<u16>,
    pub(super) y6: DeviceBuffer<f32>,
    pub(super) ycpu: DeviceBuffer<f32>,
    pub(super) slot: DeviceBuffer<i32>,
    pub(super) h: DeviceBuffer<u16>,
    pub(super) h2: DeviceBuffer<u16>,
    pub(super) pre0: DeviceBuffer<f32>,
    pub(super) mix_a: DeviceBuffer<f32>,
    pub(super) mixp: DeviceBuffer<f32>,
    pub(super) mix_f: DeviceBuffer<f32>,
    pub(super) x: DeviceBuffer<u16>,
    pub(super) xf: DeviceBuffer<f32>,
    pub(super) t32: DeviceBuffer<f32>,
    pub(super) qr_raw: DeviceBuffer<u16>,
    pub(super) qr: DeviceBuffer<u16>,
    pub(super) qrf: DeviceBuffer<f32>,
    pub(super) q: DeviceBuffer<u16>,
    pub(super) kv_raw: DeviceBuffer<u16>,
    pub(super) kv: DeviceBuffer<u16>,
    pub(super) kvc: DeviceBuffer<f32>,
    pub(super) sc: DeviceBuffer<f32>,
    pub(super) latpre: DeviceBuffer<u16>,
    pub(super) lat: DeviceBuffer<u16>,
    pub(super) latf: DeviceBuffer<f32>,
    pub(super) ik_raw: DeviceBuffer<u16>,
    pub(super) ik: DeviceBuffer<u16>,
    pub(super) iqb: DeviceBuffer<u16>,
    pub(super) iqf: DeviceBuffer<f32>,
    pub(super) iw: DeviceBuffer<f32>,
    pub(super) score: DeviceBuffer<f32>,
    pub(super) sel: DeviceBuffer<i32>,
    pub(super) bmax: DeviceBuffer<f32>,
    pub(super) cand_sel: DeviceBuffer<i32>,
    pub(super) cand_mask: DeviceBuffer<u8>,
    pub(super) o: DeviceBuffer<u16>,
    pub(super) u: DeviceBuffer<f32>,
    pub(super) ub: DeviceBuffer<u16>,
    pub(super) part: DeviceBuffer<f32>,
    pub(super) ids: DeviceBuffer<u32>,
    pub(super) wts: DeviceBuffer<f32>,
    pub(super) rscores: DeviceBuffer<f32>,
    pub(super) rlogits: DeviceBuffer<f32>,
    pub(super) meta: DeviceBuffer<u32>,
    pub(super) ekv: DeviceBuffer<f32>,
    pub(super) evecs: Vec<DeviceBuffer<f32>>,
    pub(super) sh_g: DeviceBuffer<f32>,
    pub(super) sh_u: DeviceBuffer<f32>,
    pub(super) sh_gb: DeviceBuffer<u16>,
    pub(super) sh_ub: DeviceBuffer<u16>,
    pub(super) sh_h: DeviceBuffer<u16>,
    pub(super) sh_o: DeviceBuffer<f32>,
    pub(super) logits: DeviceBuffer<f32>,
    pub(super) nr: usize,
}

/// The second card's per-row buffers.
pub struct XScratch {
    x: DeviceBuffer<u16>,
    ids: DeviceBuffer<u32>,
    y6: DeviceBuffer<f32>,
    act: Q8Act,
    hq: Q8Act,
}

struct SendPtr(*mut CpuMoe);
unsafe impl Send for SendPtr {}

impl Scratch {
    fn alloc(c: &DscConfig, max_ctx: usize) -> Result<Scratch> {
        let d = c.n_embd;
        let hd = c.head_dim;
        let nb = max_ctx.div_ceil(c.cand_block.max(1));
        let ih = c.idx_heads * c.idx_dim;
        Ok(Scratch {
            act: Q8Act::alloc(RMAX, 8192)?,
            hq: Q8Act::alloc(RMAX * c.n_expert_used, c.n_ff_exp)?,
            ug: DeviceBuffer::zeroed(RMAX * c.o_lora)?,
            taps: DeviceBuffer::zeroed(RMAX * 3 * c.n_embd)?,
            y6: DeviceBuffer::zeroed(RMAX * (c.n_expert_used * d))?,
            ycpu: DeviceBuffer::zeroed(RMAX * (c.n_expert_used * d))?,
            slot: DeviceBuffer::zeroed(RMAX * (c.n_expert_used))?,
            h: DeviceBuffer::zeroed(RMAX * (4 * d))?,
            h2: DeviceBuffer::zeroed(RMAX * (4 * d))?,
            pre0: DeviceBuffer::from_slice(&(0..RMAX).flat_map(|_| [1.0f32, 0.0, 0.0, 0.0]).collect::<Vec<f32>>())?,
            mix_a: DeviceBuffer::zeroed(RMAX * (24))?,
            mixp: DeviceBuffer::zeroed(RMAX * (c.hc * d / MIX_SLICE * 25))?,
            mix_f: DeviceBuffer::zeroed(RMAX * (24))?,
            x: DeviceBuffer::zeroed(RMAX * (d))?,
            xf: DeviceBuffer::zeroed(RMAX * (d))?,
            t32: DeviceBuffer::zeroed(RMAX * (c.q_dim().max(8192)))?,
            qr_raw: DeviceBuffer::zeroed(RMAX * (c.q_lora))?,
            qr: DeviceBuffer::zeroed(RMAX * (c.q_lora))?,
            qrf: DeviceBuffer::zeroed(RMAX * (c.q_lora))?,
            q: DeviceBuffer::zeroed(RMAX * (c.q_dim()))?,
            kv_raw: DeviceBuffer::zeroed(RMAX * (hd))?,
            kv: DeviceBuffer::zeroed(RMAX * (hd))?,
            kvc: DeviceBuffer::zeroed(RMAX * (hd))?,
            sc: DeviceBuffer::zeroed(RMAX * (hd))?,
            latpre: DeviceBuffer::zeroed(RMAX * (hd))?,
            lat: DeviceBuffer::zeroed(RMAX * (hd))?,
            latf: DeviceBuffer::zeroed(RMAX * (hd))?,
            ik_raw: DeviceBuffer::zeroed(RMAX * (c.idx_dim))?,
            ik: DeviceBuffer::zeroed(RMAX * (c.idx_dim))?,
            iqb: DeviceBuffer::zeroed(RMAX * (ih))?,
            iqf: DeviceBuffer::zeroed(RMAX * (ih))?,
            iw: DeviceBuffer::zeroed(RMAX * (c.idx_heads))?,
            score: DeviceBuffer::zeroed(RMAX * (max_ctx))?,
            sel: DeviceBuffer::zeroed(RMAX * (c.idx_top_k))?,
            bmax: DeviceBuffer::zeroed(RMAX * (nb))?,
            cand_sel: DeviceBuffer::zeroed(RMAX * (c.cand_blocks.max(1)))?,
            cand_mask: DeviceBuffer::zeroed(RMAX * (nb))?,
            o: DeviceBuffer::zeroed(RMAX * (c.q_dim()))?,
            u: DeviceBuffer::zeroed(RMAX * (c.o_groups * c.o_lora))?,
            ub: DeviceBuffer::zeroed(RMAX * (c.o_groups * c.o_lora))?,
            part: DeviceBuffer::zeroed(RMAX * (d))?,
            ids: DeviceBuffer::zeroed(RMAX * (c.n_expert_used))?,
            wts: DeviceBuffer::zeroed(RMAX * (c.n_expert_used))?,
            rscores: DeviceBuffer::zeroed(RMAX * (c.n_expert))?,
            rlogits: DeviceBuffer::zeroed(RMAX * (c.n_expert))?,
            meta: DeviceBuffer::zeroed(RMAX * (2))?,
            ekv: DeviceBuffer::zeroed(RMAX * (c.engram_width().max((c.hc + 1) * d)))?,
            evecs: (0..c.engram_layers.len()).map(|_| DeviceBuffer::zeroed(RMAX * c.engram_width())).collect::<Result<Vec<_>, _>>()?,
            sh_g: DeviceBuffer::zeroed(RMAX * (c.n_ff_exp))?,
            sh_u: DeviceBuffer::zeroed(RMAX * (c.n_ff_exp))?,
            sh_gb: DeviceBuffer::zeroed(RMAX * (c.n_ff_exp))?,
            sh_ub: DeviceBuffer::zeroed(RMAX * (c.n_ff_exp))?,
            sh_h: DeviceBuffer::zeroed(RMAX * (c.n_ff_exp))?,
            sh_o: DeviceBuffer::zeroed(RMAX * (d))?,
            logits: DeviceBuffer::zeroed(RMAX * (c.n_vocab))?,
            nr: 1,
        })
    }
}

impl<'a> Engine<'a> {
    /// Allocate the second scratch set and turn the pipeline on for passes of >= `min_rows` rows.
    pub fn enable_pipe(&mut self, min_rows: usize) -> Result<()> {
        let c = self.c;
        self.sc2 = Some(Scratch::alloc(c, self.max_ctx)?);
        if let Some(xc) = self.xcs.first() {
            xc.ctx.bind()?;
            self.xsc2 = Some(XScratch {
                x: DeviceBuffer::zeroed(RMAX * c.n_embd)?,
                ids: DeviceBuffer::zeroed(RMAX * c.n_expert_used)?,
                y6: DeviceBuffer::zeroed(RMAX * c.n_expert_used * c.n_embd)?,
                act: Q8Act::alloc(RMAX, 8192)?,
                hq: Q8Act::alloc(RMAX * c.n_expert_used, c.n_ff_exp)?,
            });
            self._ctx.bind()?;
        }
        self.pipe_rows = min_rows.max(2);
        Ok(())
    }

    /// Swap the engine's scratch with the parked one (the second card's too).
    fn swap_mb(&mut self) {
        let mut sc = self.sc2.take().unwrap();
        std::mem::swap(&mut self.act, &mut sc.act);
        std::mem::swap(&mut self.hq, &mut sc.hq);
        std::mem::swap(&mut self.ug, &mut sc.ug);
        std::mem::swap(&mut self.taps, &mut sc.taps);
        std::mem::swap(&mut self.y6, &mut sc.y6);
        std::mem::swap(&mut self.ycpu, &mut sc.ycpu);
        std::mem::swap(&mut self.slot, &mut sc.slot);
        std::mem::swap(&mut self.h, &mut sc.h);
        std::mem::swap(&mut self.h2, &mut sc.h2);
        std::mem::swap(&mut self.pre0, &mut sc.pre0);
        std::mem::swap(&mut self.mix_a, &mut sc.mix_a);
        std::mem::swap(&mut self.mixp, &mut sc.mixp);
        std::mem::swap(&mut self.mix_f, &mut sc.mix_f);
        std::mem::swap(&mut self.x, &mut sc.x);
        std::mem::swap(&mut self.xf, &mut sc.xf);
        std::mem::swap(&mut self.t32, &mut sc.t32);
        std::mem::swap(&mut self.qr_raw, &mut sc.qr_raw);
        std::mem::swap(&mut self.qr, &mut sc.qr);
        std::mem::swap(&mut self.qrf, &mut sc.qrf);
        std::mem::swap(&mut self.q, &mut sc.q);
        std::mem::swap(&mut self.kv_raw, &mut sc.kv_raw);
        std::mem::swap(&mut self.kv, &mut sc.kv);
        std::mem::swap(&mut self.kvc, &mut sc.kvc);
        std::mem::swap(&mut self.sc, &mut sc.sc);
        std::mem::swap(&mut self.latpre, &mut sc.latpre);
        std::mem::swap(&mut self.lat, &mut sc.lat);
        std::mem::swap(&mut self.latf, &mut sc.latf);
        std::mem::swap(&mut self.ik_raw, &mut sc.ik_raw);
        std::mem::swap(&mut self.ik, &mut sc.ik);
        std::mem::swap(&mut self.iqb, &mut sc.iqb);
        std::mem::swap(&mut self.iqf, &mut sc.iqf);
        std::mem::swap(&mut self.iw, &mut sc.iw);
        std::mem::swap(&mut self.score, &mut sc.score);
        std::mem::swap(&mut self.sel, &mut sc.sel);
        std::mem::swap(&mut self.bmax, &mut sc.bmax);
        std::mem::swap(&mut self.cand_sel, &mut sc.cand_sel);
        std::mem::swap(&mut self.cand_mask, &mut sc.cand_mask);
        std::mem::swap(&mut self.o, &mut sc.o);
        std::mem::swap(&mut self.u, &mut sc.u);
        std::mem::swap(&mut self.ub, &mut sc.ub);
        std::mem::swap(&mut self.part, &mut sc.part);
        std::mem::swap(&mut self.ids, &mut sc.ids);
        std::mem::swap(&mut self.wts, &mut sc.wts);
        std::mem::swap(&mut self.rscores, &mut sc.rscores);
        std::mem::swap(&mut self.rlogits, &mut sc.rlogits);
        std::mem::swap(&mut self.meta, &mut sc.meta);
        std::mem::swap(&mut self.ekv, &mut sc.ekv);
        std::mem::swap(&mut self.evecs, &mut sc.evecs);
        std::mem::swap(&mut self.sh_g, &mut sc.sh_g);
        std::mem::swap(&mut self.sh_u, &mut sc.sh_u);
        std::mem::swap(&mut self.sh_gb, &mut sc.sh_gb);
        std::mem::swap(&mut self.sh_ub, &mut sc.sh_ub);
        std::mem::swap(&mut self.sh_h, &mut sc.sh_h);
        std::mem::swap(&mut self.sh_o, &mut sc.sh_o);
        std::mem::swap(&mut self.logits, &mut sc.logits);
        std::mem::swap(&mut self.nr, &mut sc.nr);
        self.sc2 = Some(sc);
        if let (Some(x), Some(xc)) = (self.xsc2.as_mut(), self.xcs.first_mut()) {
            std::mem::swap(&mut xc.x, &mut x.x);
            std::mem::swap(&mut xc.ids, &mut x.ids);
            std::mem::swap(&mut xc.y6, &mut x.y6);
            std::mem::swap(&mut xc.act, &mut x.act);
            std::mem::swap(&mut xc.hq, &mut x.hq);
        }
    }

    pub(super) fn step_rows_pipe(&mut self, tokens: &[u32], want_logits: bool) -> Result<Option<Vec<Vec<f32>>>> {
        let c = self.c;
        let nr = tokens.len();
        let na = nr.div_ceil(2);
        let pos0 = self.ids_seen.len();
        self.ids_seen.extend_from_slice(tokens);
        let t_all = Instant::now();
        self.split = Split::default();
        let (ta, tb) = tokens.split_at(na);
        self.begin_rows(ta, pos0)?;
        self.swap_mb();
        let r = self.begin_rows(tb, pos0 + na);
        self.swap_mb();
        r?;
        let nl = c.n_layer;
        let cpu_ptr = SendPtr(&mut self.cpu as *mut CpuMoe);
        let res: Result<()> = std::thread::scope(|scope| {
            let (jtx, jrx) = mpsc::channel::<(Vec<MoeJob<'a>>, Vec<f32>, f32)>();
            let (rtx, rrx) = mpsc::channel::<Result<(Vec<Vec<f32>>, Duration)>>();
            scope.spawn(move || {
                let cp = cpu_ptr;
                let cpu = unsafe { &mut *cp.0 };
                for (jobs, xh, limit) in jrx {
                    let t = Instant::now();
                    cpu.swiglu_limit = limit;
                    let r = if jobs.is_empty() {
                        Ok(vec![])
                    } else {
                        cpu.run(&jobs, &xh).map(|_| (0..jobs.len()).map(|j| cpu.out(j).to_vec()).collect::<Vec<_>>())
                    };
                    if rtx.send(r.map(|o| (o, t.elapsed()))).is_err() {
                        break;
                    }
                }
            });
            let send = |p: &mut Pend<'a>| {
                let _ = jtx.send((std::mem::take(&mut p.jobs), std::mem::take(&mut p.xh), p.limit));
            };
            let mut pa = self.layer_pre(0, pos0)?;
            send(&mut pa);
            self.swap_mb();
            let rb = self.layer_pre(0, pos0 + na);
            self.swap_mb();
            let mut pb = rb?;
            send(&mut pb);
            let (mut pa, mut pb) = (Some(pa), Some(pb));
            let (mut w_cpu, mut t_pre, mut t_post, mut t_run) = (0f64, 0f64, 0f64, 0f64);
            let mut tm = Instant::now();
            for l in 0..nl {
                tm = Instant::now();
                let (outs, dur) = rrx.recv().map_err(|_| anyhow!("cpu worker gone"))??;
                w_cpu += tm.elapsed().as_secs_f64() * 1e3;
                t_run += dur.as_secs_f64() * 1e3;
                tm = Instant::now();
                self.layer_post(pa.take().unwrap(), &outs, dur)?;
                t_post += tm.elapsed().as_secs_f64() * 1e3;
                tm = Instant::now();
                if l + 1 < nl {
                    let mut p = self.layer_pre(l + 1, pos0)?;
                    send(&mut p);
                    pa = Some(p);
                }
                t_pre += tm.elapsed().as_secs_f64() * 1e3;
                tm = Instant::now();
                let (outs, dur) = rrx.recv().map_err(|_| anyhow!("cpu worker gone"))??;
                w_cpu += tm.elapsed().as_secs_f64() * 1e3;
                t_run += dur.as_secs_f64() * 1e3;
                self.swap_mb();
                let r = (|| -> Result<()> {
                    let tq = Instant::now();
                    self.layer_post(pb.take().unwrap(), &outs, dur)?;
                    t_post += tq.elapsed().as_secs_f64() * 1e3;
                    let tq = Instant::now();
                    if l + 1 < nl {
                        let mut p = self.layer_pre(l + 1, pos0 + na)?;
                        send(&mut p);
                        pb = Some(p);
                    }
                    t_pre += tq.elapsed().as_secs_f64() * 1e3;
                    Ok(())
                })();
                self.swap_mb();
                r?;
            }
            drop(jtx);
            let _ = tm;
            self.split.ph = [w_cpu, t_pre, t_post, t_run, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            Ok(())
        });
        res?;
        let out_a = self.finish_rows(want_logits)?;
        self.swap_mb();
        let out_b = self.finish_rows(want_logits);
        self.swap_mb();
        let out_b = out_b?;
        // the second half's taps after the first half's rows
        let d3 = 3 * c.n_embd;
        memcpy_dtod(self.taps.ptr().add::<u16>(na * d3), self.sc2.as_ref().unwrap().taps.ptr(), (nr - na) * d3 * 2)?;
        self.tokens_done += 1;
        self.split.total_ms = t_all.elapsed().as_secs_f64() * 1e3;
        Ok(match (out_a, out_b) {
            (Some(mut a), Some(b)) => {
                a.extend(b);
                Some(a)
            }
            _ => None,
        })
    }
}
