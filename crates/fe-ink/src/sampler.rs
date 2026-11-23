//! Device-side sampling for the Inkling engine: temperature / top-k / min-p / top-p / penalties / banned tokens computed on the GPU from the bf16 logits
//! (the server's Sampler semantics), plus the lossless accept / reject of drafted tokens. Used by the draft loop (`spec.rs`) and by plain decode.
use crate::*;
use anyhow::{ensure, Result};
use fe_cuda::{Context, DevPtr, DeviceBuffer, Function, Module, Stream};
use std::collections::HashMap;
use std::sync::Arc;

type Buf = DeviceBuffer<u8>;
/// capacity of the penalty history table (distinct tokens seen)
pub const FTAB_CAP: usize = 65536;

/// A request's sampling configuration (the server's Sampler semantics: penalties -> temperature -> top-k -> min-p -> top-p).
#[derive(Clone, Copy, Debug)]
pub struct SampleCfg {
    /// < 1e-5 = greedy
    pub temperature: f32,
    /// 0 = off
    pub top_k: u32,
    /// 1.0 = off
    pub top_p: f32,
    /// 0.0 = off
    pub min_p: f32,
    /// repetition penalty (HF style) over prompt + output, 1.0 = off
    pub rep: f32,
    pub pres: f32,
    pub freq: f32,
}

impl SampleCfg {
    pub fn greedy() -> Self {
        SampleCfg { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, rep: 1.0, pres: 0.0, freq: 0.0 }
    }
    pub fn temperature(t: f32) -> Self {
        SampleCfg { temperature: t, ..Self::greedy() }
    }
    pub fn is_greedy(&self) -> bool {
        self.temperature < 1e-5
    }
    pub fn has_penalties(&self) -> bool {
        self.rep != 1.0 || self.pres != 0.0 || self.freq != 0.0
    }
    /// plain greedy: the device argmax path, bit-identical to plain decode
    pub fn plain_greedy(&self) -> bool {
        self.is_greedy() && !self.has_penalties()
    }
    /// anything beyond temperature: needs the filtered distribution kernel
    pub fn filtered(&self) -> bool {
        self.top_k > 0 || self.top_p < 1.0 || self.min_p > 0.0 || self.has_penalties() || self.is_greedy()
    }
    pub fn words(&self) -> [u32; 8] {
        let greedy = self.is_greedy();
        [(if greedy { 1.0 } else { 1.0 / self.temperature }).to_bits(), greedy as u32, self.top_k, self.top_p.to_bits(), self.min_p.to_bits(), self.rep.to_bits(),
            self.pres.to_bits(), self.freq.to_bits()]
    }
}

/// Penalty history of one request: token -> (output count, seen in prompt or output).
#[derive(Default)]
pub struct PenHist {
    pub map: HashMap<u32, (u32, bool)>,
    on: bool,
}

impl PenHist {
    pub fn begin(&mut self, cfg: &SampleCfg, prompt: &[u32]) {
        self.map.clear();
        self.on = cfg.has_penalties();
        if self.on {
            for &t in prompt {
                self.map.entry(t).or_insert((0, false)).1 = true;
            }
        }
    }
    pub fn push(&mut self, t: u32) {
        if self.on {
            let e = self.map.entry(t).or_insert((0, false));
            e.0 += 1;
            e.1 = true;
        }
    }
    pub fn table(&self) -> Vec<[u32; 3]> {
        if !self.on {
            return vec![];
        }
        self.map.iter().map(|(&t, &(c, s))| [t, c, s as u32]).collect()
    }
}

pub struct DevSampler {
    pub ctx: Arc<Context>,
    f_stats: Function,
    f_sample: Function,
    f_fprep: Function,
    f_fsample: Function,
    sdraft: Buf,
    spd: Buf,
    smx: Buf,
    szs: Buf,
    stok: Buf,
    fs: Buf,
    fcfg: Buf,
    ftab: Buf,
    fext: Buf,
    fnext: Buf,
    foutf: Buf,
    foutu: Buf,
    ftok: Buf,
    flg: Buf,
    fban: Buf,
}

fn alloc(b: usize) -> Result<Buf> {
    Ok(Buf::alloc(b.max(16))?)
}

impl DevSampler {
    pub fn new(ctx: Arc<Context>) -> Result<DevSampler> {
        ctx.bind()?;
        let d = Module::load(K_DRAFT_CUBIN)?;
        Ok(DevSampler {
            f_stats: d.function("dr_rows_stats")?,
            f_sample: d.function("dr_sample_row")?,
            f_fprep: d.function("dr_filter_prep")?,
            f_fsample: d.function("dr_sample_filtered")?,
            sdraft: alloc(8 * 4)?,
            spd: alloc(8 * 4)?,
            smx: alloc(8 * 4)?,
            szs: alloc(8 * 4)?,
            stok: alloc(16)?,
            fs: alloc(8 * VOCAB * 4)?,
            fcfg: alloc(8 * 4)?,
            ftab: alloc(FTAB_CAP * 12)?,
            fext: alloc(8 * 7 * 16)?,
            fnext: alloc(8 * 4)?,
            foutf: alloc(8 * 32)?,
            foutu: alloc(8 * 16)?,
            ftok: alloc(16)?,
            flg: alloc(VOCAB * 2)?,
            fban: alloc(4096 * 4)?,
            ctx,
        })
    }

    pub fn sample_rows(&mut self, st: &Stream, logits: DevPtr, drafts: &[u32], inv_t: f32, mut u: impl FnMut() -> f64) -> Result<(usize, u32)> {
        let d = drafts.len();
        let mut dd = [0xffff_ffffu32; 8];
        dd[..d].copy_from_slice(drafts);
        unsafe {
            st.memcpy_htod(self.sdraft.ptr(), &dd)?;
            st.launch(&self.f_stats, (d + 1) as u32, 1024u32, 0, (self.spd.ptr(), self.smx.ptr(), self.szs.ptr(), logits, VOCAB as u32, inv_t, self.sdraft.ptr()))?;
        }
        st.sync()?;
        let mut pd = [0f32; 8];
        fe_cuda::memcpy_dtoh(&mut pd, self.spd.ptr())?;
        let mut k = 0;
        while k < d && u() < pd[k] as f64 {
            k += 1;
        }
        let skip = if k < d { drafts[k] } else { 0xffff_ffffu32 };
        unsafe {
            st.launch(&self.f_sample, 1u32, 1024u32, 0, (self.stok.ptr(), logits, k as u32, VOCAB as u32, inv_t, self.smx.ptr(), self.szs.ptr(), skip, u() as f32))?;
        }
        st.sync()?;
        let mut t = [0u32; 4];
        fe_cuda::memcpy_dtoh(&mut t, self.stok.ptr())?;
        Ok((k, t[0]))
    }

    /// Filtered speculative sampling over `drafts.len() + 1` verified rows with bf16 logits on the device at `logits` (or, when `host_row` is Some, a single
    /// host row uploaded first): the final distribution of every row is the request's (penalties with the drafted prefix as history, temperature, top-k, min-p,
    /// top-p; greedy when `cfg.greedy`), computed on the device. Draft j is accepted with probability p_j(draft_j), a rejection draws from p_j without it, after d
    /// accepted drafts a token is drawn from row d. `tab`: base history [(token, output count, seen)]. Returns (accepted, token).
    pub fn sample_rows_filtered(&mut self, st: &Stream, logits: DevPtr, drafts: &[u32], cfg: &[u32; 8], tab: &[[u32; 3]], banned: &[u32], mut u: impl FnMut() -> f64) -> Result<(usize, u32)> {
        let d = drafts.len();
        ensure!(tab.len() <= FTAB_CAP, "penalty history too large");
        let mut dd = [0xffff_ffffu32; 8];
        dd[..d].copy_from_slice(drafts);
        // per-row extras: unique tokens of the drafted prefix with their multiplicity and whether the base history has them
        let base: std::collections::HashMap<u32, (u32, u32)> = if cfg[5] != 1.0f32.to_bits() || f32::from_bits(cfg[6]) != 0.0 || f32::from_bits(cfg[7]) != 0.0 {
            tab.iter().map(|e| (e[0], (e[1], e[2]))).collect()
        } else {
            Default::default()
        };
        let mut ext = [0u32; 8 * 7 * 4];
        let mut next = [0u32; 8];
        for j in 1..=d {
            let mut uniq: Vec<(u32, u32)> = vec![];
            for &t in &drafts[..j] {
                if let Some(x) = uniq.iter_mut().find(|x| x.0 == t) { x.1 += 1 } else { uniq.push((t, 1)) }
            }
            next[j] = uniq.len() as u32;
            for (i, (t, c)) in uniq.iter().enumerate() {
                let (bc, bs) = base.get(t).copied().unwrap_or((0, 0));
                let o = (j * 7 + i) * 4;
                ext[o..o + 4].copy_from_slice(&[*t, *c, bs, bc]);
            }
        }
        let flat: Vec<u32> = tab.iter().flat_map(|e| e.iter().copied()).collect();
        unsafe {
            st.memcpy_htod(self.sdraft.ptr(), &dd)?;
            st.memcpy_htod(self.fcfg.ptr(), cfg)?;
            if !flat.is_empty() {
                st.memcpy_htod(self.ftab.ptr(), &flat)?;
            }
            st.memcpy_htod(self.fext.ptr(), &ext)?;
            st.memcpy_htod(self.fnext.ptr(), &next)?;
            if !banned.is_empty() {
                st.memcpy_htod(self.fban.ptr(), banned)?;
            }
            st.launch(&self.f_fprep, (d + 1) as u32, 1024u32, 0, (self.fs.ptr(), logits, VOCAB as u32, self.fcfg.ptr(), self.ftab.ptr(), tab.len() as u32,
                self.fext.ptr(), self.fnext.ptr(), self.sdraft.ptr(), self.foutf.ptr(), self.foutu.ptr(), self.fban.ptr(), banned.len() as u32))?;
        }
        st.sync()?;
        let mut of = [0f32; 64];
        fe_cuda::memcpy_dtoh(&mut of, self.foutf.ptr())?;
        let mut k = 0;
        while k < d && u() < of[k * 8 + 6] as f64 {
            k += 1;
        }
        let skip = if k < d { drafts[k] } else { 0xffff_ffffu32 };
        unsafe {
            st.launch(&self.f_fsample, 1u32, 1024u32, 0, (self.ftok.ptr(), self.fs.ptr(), k as u32, VOCAB as u32, self.foutf.ptr(), self.foutu.ptr(), skip,
                u() as f32, cfg[1]))?;
        }
        st.sync()?;
        let mut t = [0u32; 4];
        fe_cuda::memcpy_dtoh(&mut t, self.ftok.ptr())?;
        Ok((k, t[0]))
    }

    /// Upload one host row of bf16 logits to scratch (first token after the prompt) and return its device pointer.
    pub fn upload_row(&mut self, st: &Stream, row: &[u16]) -> Result<DevPtr> {
        unsafe { st.memcpy_htod(self.flg.ptr(), row)? };
        Ok(self.flg.ptr())
    }

    /// One token from the bf16 logits row at `logits` (device): the request's distribution with `banned` tokens removed, history `hist`. Plain (non-speculative) decode.
    pub fn sample_one(&mut self, st: &Stream, logits: DevPtr, cfg: &SampleCfg, hist: &PenHist, banned: &[u32], u: impl FnMut() -> f64) -> Result<u32> {
        let words = cfg.words();
        let tab = hist.table();
        Ok(self.sample_rows_filtered(st, logits, &[], &words, &tab, banned, u)?.1)
    }
}
