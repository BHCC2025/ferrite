//! The MTP head as a drafter (arc-engine `qwen_mtp.zig`, `qwen_mtp_rows.zig`): one full-attention block behind fc over
//! [rmsnorm(embed(u)) | rmsnorm(h)], its own q8 KV cache, and the ranked int8 draft head (MTP_DHEAD). A step at position p takes the
//! next token u and the target's final-normed hidden row h of row p; the MTP cache entry of row p is written at p (made with token
//! p + 1). Steps run with the one-token kernels (fused XQ norms / SwiGLU, single-row decode attention); the batched absorb writes the
//! cache entries of many kept rows at once (fc, input norm, k / v, k-norm + RoPE, append) with the window kernels.
//! Drafts only steer speculation: the target verifies every token, so the output is the plain greedy output.
use crate::engine::{AttnW, Dev, Engine, MlpW, XqW, ARGMAX_PARTS, GDN_QKV, HEADS, HIDDEN, KV_DIM, KV_HEADS, MAX_CHUNKS, MAX_ROWS};
use crate::state::KvMode;
use crate::weights::{HostTensor, Planner};
use anyhow::{bail, ensure, Context as _, Result};
use fe_ze::{DevPtr, DeviceBuffer, KernelArg};
use std::fs::File;
use std::os::unix::fs::FileExt;

/// The ranked int8 draft head (`tools/make_dhead.py`, file "DHQ8" v1): the first `n_base` rows (most frequent tokens) on the device,
/// plus the rows of a request's prompt tokens outside them (up to `extra`). Drafts are argmaxed over the active rows and mapped back.
pub struct RvHead {
    file: File,
    hdr: u64,
    cols: usize,
    n_base: usize,
    cap_rows: usize,
    pub n: usize,
    rank: Vec<u32>,
    ids_host: Vec<u32>,
    scale_host: Vec<f32>,
    q: DeviceBuffer<i8>,
    scale: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    /// the rows past the base ones in host memory (FE_XE_DHEAD_RAM, default on: a prompt's rare tokens then cost no file reads;
    /// on the HDD a cold row is a seek)
    tail_host: Option<Vec<u8>>,
    /// the last prompt's added rows and the time they took (ms)
    pub last_rows: usize,
    pub last_ms: f64,
}

impl RvHead {
    pub fn load(dev: &Dev, path: &str, n_base: usize, extra: usize) -> Result<RvHead> {
        let file = File::open(path).with_context(|| format!("draft head {path}"))?;
        let mut h = [0u8; 20];
        file.read_exact_at(&mut h, 0)?;
        let u = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap()) as usize;
        ensure!(&h[..4] == b"DHQ8" && u(4) == 1, "{path}: not a DHQ8 v1 draft head");
        let (rows, cols) = (u(8), u(12));
        ensure!(cols == HIDDEN as usize && n_base > 0 && n_base <= rows, "{path}: rows {rows} cols {cols}");
        let mut b = vec![0u8; rows * 4];
        file.read_exact_at(&mut b, 20)?;
        let ids_host: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        file.read_exact_at(&mut b, 20 + rows as u64 * 4)?;
        let scale_host: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut rank = vec![u32::MAX; rows.max(ids_host.iter().copied().max().unwrap_or(0) as usize + 1)];
        for (i, &t) in ids_host.iter().enumerate() {
            rank[t as usize] = i as u32;
        }
        let cap_rows = rows.min(n_base + extra);
        let hdr = 20 + rows as u64 * 8;
        let q = dev.alloc::<i8>(cap_rows * cols)?;
        let mut piece = vec![0i8; 64 << 20];
        let total = n_base * cols;
        let mut off = 0;
        while off < total {
            let len = piece.len().min(total - off);
            let bytes = unsafe { std::slice::from_raw_parts_mut(piece.as_mut_ptr() as *mut u8, len) };
            file.read_exact_at(bytes, hdr + off as u64)?;
            dev.upload(q.ptr().byte_add(off), &piece[..len])?;
            off += len;
        }
        let scale = dev.alloc::<f32>(cap_rows)?;
        dev.upload(scale.ptr(), &scale_host[..n_base])?;
        let ids = dev.alloc::<u32>(cap_rows)?;
        dev.upload(ids.ptr(), &ids_host[..n_base])?;
        eprintln!("draft head {path}: {n_base} ranked rows of {rows} on the device ({:.0} MB), room for {} prompt tokens", total as f64 / 1e6, cap_rows - n_base);
        let tail_host = if std::env::var("FE_XE_DHEAD_RAM").map_or(true, |v| v != "0") && rows > n_base {
            let mut t = vec![0u8; (rows - n_base) * cols];
            file.read_exact_at(&mut t, hdr + (n_base * cols) as u64)?;
            eprintln!("draft head: the other {} rows in host memory ({:.0} MB)", rows - n_base, t.len() as f64 / 1e6);
            Some(t)
        } else {
            None
        };
        Ok(RvHead { file, hdr, cols, n_base, cap_rows, n: n_base, rank, ids_host, scale_host, q, scale, ids, tail_host, last_rows: 0, last_ms: 0.0 })
    }

    /// Back to the base rows, then the rows of `tokens` outside them (a new request's prompt).
    pub fn prompt(&mut self, dev: &Dev, tokens: &[u32]) -> Result<()> {
        let t0 = std::time::Instant::now();
        self.n = self.n_base;
        self.last_rows = 0;
        let mut rows: Vec<u32> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &t in tokens {
            let r = *self.rank.get(t as usize).unwrap_or(&u32::MAX);
            if r == u32::MAX || (r as usize) < self.n_base {
                continue;
            }
            if self.n_base + rows.len() >= self.cap_rows {
                break;
            }
            if seen.insert(t) {
                rows.push(r);
            }
        }
        if rows.is_empty() {
            return Ok(());
        }
        let k = self.cols;
        let mut qh = vec![0i8; rows.len() * k];
        let mut sh = Vec::with_capacity(rows.len());
        let mut ih = Vec::with_capacity(rows.len());
        for (i, &fr) in rows.iter().enumerate() {
            let bytes = unsafe { std::slice::from_raw_parts_mut(qh[i * k..].as_mut_ptr() as *mut u8, k) };
            match &self.tail_host {
                Some(t) => {
                    let o = (fr as usize - self.n_base) * k;
                    bytes.copy_from_slice(&t[o..o + k]);
                }
                None => self.file.read_exact_at(bytes, self.hdr + fr as u64 * k as u64)?,
            }
            sh.push(self.scale_host[fr as usize]);
            ih.push(self.ids_host[fr as usize]);
        }
        let n0 = self.n_base;
        dev.upload(self.q.ptr().byte_add(n0 * k), &qh)?;
        dev.upload(self.scale.at(n0), &sh)?;
        dev.upload(self.ids.at(n0), &ih)?;
        self.n = n0 + rows.len();
        self.last_rows = rows.len();
        self.last_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }
}

pub struct Mtp {
    pub fc: XqW,
    pub attn: AttnW,
    pub mlp: MlpW,
    pub norm: DeviceBuffer<u16>,
    pub n_emb: DeviceBuffer<u16>,
    pub n_hid: DeviceBuffer<u16>,
    pub kc: DeviceBuffer<u8>,
    pub vc: DeviceBuffer<u8>,
    /// [rows][embedding | hidden] normed, bf16 (batched absorb; row 0 for steps)
    pub cat: DeviceBuffer<u16>,
    pub tok: DeviceBuffer<u32>,
    /// RoPE cos / sin of every position [ctx][64] fp32: a step needs no host upload
    pub rope_tab: DeviceBuffer<f32>,
    pub tokdev: DeviceBuffer<u32>,
    pub drafts: DeviceBuffer<i32>,
    pub dhead: Option<RvHead>,
    rcap: u32,
}

fn take(v: &mut Vec<(&'static str, HostTensor)>, role: &str) -> Result<HostTensor> {
    let i = v.iter().position(|(r, _)| *r == role).with_context(|| format!("no MTP {role}"))?;
    Ok(v.swap_remove(i).1)
}

impl Mtp {
    /// MTP weights from the checkpoint, a q8 cache of the engine's ctx, the int8 draft head from `dhead` (MTP_DHEAD) when given.
    pub fn load(e: &Engine, ck: &fe_xq::Checkpoint, dhead: Option<&str>) -> Result<Mtp> {
        let dev = &e.dev;
        let p = Planner::new(ck, &e.cfg);
        let mut v = p.mtp()?;
        let mut ld = crate::engine::Uploader::new(dev);
        let ctx = e.ctx as usize;
        let rec = KvMode::Q8.rec_bytes();
        let rcap = e.rcap();
        let mut rope = vec![0f32; ctx * 64];
        let inv = crate::rope::inv_freq(e.cfg.rope_theta);
        for pos in 0..ctx {
            crate::rope::row(&inv, pos as u32, &mut rope[pos * 64..(pos + 1) * 64]);
        }
        let m = Mtp {
            fc: ld.xq(take(&mut v, "fc")?)?,
            attn: AttnW {
                norm: ld.u16s(take(&mut v, "input_norm")?)?,
                q: ld.xq(take(&mut v, "q")?)?,
                k: ld.xq(take(&mut v, "k")?)?,
                v: ld.xq(take(&mut v, "v")?)?,
                o: ld.xq(take(&mut v, "o")?)?,
                qn: ld.u16s(take(&mut v, "q_norm")?)?,
                kn: ld.u16s(take(&mut v, "k_norm")?)?,
            },
            mlp: MlpW {
                norm: ld.u16s(take(&mut v, "post_norm")?)?,
                gate: ld.xq(take(&mut v, "gate")?)?,
                up: ld.xq(take(&mut v, "up")?)?,
                down: ld.xq(take(&mut v, "down")?)?,
            },
            norm: ld.u16s(take(&mut v, "norm")?)?,
            n_emb: ld.u16s(take(&mut v, "pre_fc_norm_embedding")?)?,
            n_hid: ld.u16s(take(&mut v, "pre_fc_norm_hidden")?)?,
            kc: dev.zeros(ctx * 4 * rec)?,
            vc: dev.zeros(ctx * 4 * rec)?,
            cat: dev.zeros(rcap as usize * 2 * HIDDEN as usize)?,
            tok: dev.zeros(rcap as usize)?,
            rope_tab: dev.from_host(&rope)?,
            tokdev: dev.zeros(16)?,
            drafts: dev.zeros(16)?,
            dhead: match dhead {
                Some(path) => Some(RvHead::load(dev, path, 65536, 8192)?),
                None => None,
            },
            rcap,
        };
        dev.sync()?;
        Ok(m)
    }

    /// (arc-engine's one-row kernels rmsnorm512 / add_rmsnorm / attn_prep are the row kernels' arithmetic at m = 1, operation for
    /// operation and in the same summation order: the head runs the row kernels with one row, so it needs no extra kernel ports.)
    /// One MTP position: input token at device address `tok` (u32), the target's normed hidden row `h` (bf16 [5120]). With `out`, the
    /// draft (i32) is written there and the normed output stays in the scratch's xn row 0 (the h of a chained step).
    pub fn step(&mut self, e: &mut Engine, tok: DevPtr, h: DevPtr, pos: u32, out: Option<DevPtr>) -> Result<()> {
        ensure!(pos < e.ctx, "MTP position {pos} past the cache");
        let eps = e.eps;
        let (x, xn, delta) = (e.sc.x.ptr(), e.sc.xn.ptr(), e.sc.delta.ptr());
        let emb = e.w.emb.ptr();
        e.launch("basic", "embed_bf16", [64, 1, 1], [1, 1, 1], &[&emb, &tok, &x, &HIDDEN])?;
        let cat = self.cat.ptr();
        let cat_h = cat.add::<u16>(HIDDEN as usize);
        e.launch("rows", "rmsnorm_r", [512, 1, 1], [1, 1, 1], &[&x, &self.n_emb.ptr(), &cat, &HIDDEN, &eps])?;
        e.launch("rows", "rmsnorm_r", [512, 1, 1], [1, 1, 1], &[&h, &self.n_hid.ptr(), &cat_h, &HIDDEN, &eps])?;
        e.xq_rows(&self.fc, cat, 1, x, false)?;
        // the attention block, one row, on the MTP cache (Ops.attention with the MTP's rope table row)
        let rope = self.rope_tab.ptr().add::<f32>(pos as usize * 64);
        let s = &e.sc;
        let (qg, kraw, vtmp, ktmp, q, att, po, pm, pl) =
            (s.qg.ptr(), s.kraw.ptr(), s.vtmp.ptr(), s.ktmp.ptr(), s.q.ptr(), s.att.ptr(), s.po.ptr(), s.pm.ptr(), s.pl.ptr());
        e.add_rms_prep(&self.attn.q, x, delta, self.attn.norm.ptr(), xn, false)?;
        e.xq_rows(&self.attn.q, xn, 1, qg, false)?;
        e.xq_rows(&self.attn.k, xn, 1, kraw, false)?;
        e.xq_rows(&self.attn.v, xn, 1, vtmp, false)?;
        e.launch("rows", "attn_prep_r", [256, 1, 1], [HEADS + KV_HEADS, 1, 1],
            &[&qg, &kraw, &self.attn.qn.ptr(), &self.attn.kn.ptr(), &rope, &q, &ktmp, &0u64, &eps])?;
        let (kc, vc) = (self.kc.ptr(), self.vc.ptr());
        e.launch("kvq", "kvq_quant8", [32, 1, 1], [1, 1, 1], &[&ktmp, &kc, &pos])?;
        e.launch("kvq", "kvq_quant8", [32, 1, 1], [1, 1, 1], &[&vtmp, &vc, &pos])?;
        let len = pos + 1;
        let nch = len.div_ceil(1024);
        let scale = 0.0625f32;
        e.launch("attn_dec_q8", "attn_dec_partial", [128, 1, 1], [4, nch, 1], &[&q, &kc, &vc, &po, &pm, &pl, &len, &scale, &MAX_CHUNKS])?;
        e.launch("attn_long_q8", "attn_long_merge", [256, 1, 1], [HEADS, 1, 1], &[&po, &pm, &pl, &qg, &att, &len, &MAX_CHUNKS])?;
        e.xq_rows(&self.attn.o, att, 1, delta, false)?;
        // the MLP (Ops.mlp, XQ fused path)
        let (gate, up) = (e.sc.gate.ptr(), e.sc.up.ptr());
        e.add_rms_prep(&self.mlp.gate, x, delta, self.mlp.norm.ptr(), xn, true)?;
        e.xq_rows(&self.mlp.gate, xn, 1, gate, false)?;
        e.xq_rows(&self.mlp.up, xn, 1, up, false)?;
        e.prep_swiglu(&self.mlp.down, gate, up, gate)?;
        e.xq_rows(&self.mlp.down, gate, 1, delta, false)?;
        if let Some(out) = out {
            let dh = self.dhead.as_ref().context("MTP drafts need the int8 draft head (MTP_DHEAD)")?;
            // mtp.norm with the pending MLP add folded in (RvHead.head: Ops.normIn -> add_rmsnorm)
            e.launch("rows", "add_rmsnorm_r", [512, 1, 1], [1, 1, 1], &[&x, &delta, &self.norm.ptr(), &xn, &HIDDEN, &eps])?;
            let n = dh.n as u32;
            let cols = dh.cols as u32;
            let (logits, am_s) = (e.sc.logits.ptr(), e.sc.am_scratch.ptr());
            e.launch("spec_head", "dhead_q8b", [64, 1, 1], [n.div_ceil(32), 1, 1], &[&dh.q.ptr(), &dh.scale.ptr(), &xn, &logits, &n, &cols])?;
            let per = n.div_ceil(ARGMAX_PARTS);
            e.launch("basic", "argmax_partial", [256, 1, 1], [ARGMAX_PARTS, 1, 1], &[&logits, &am_s, &n, &per])?;
            e.launch("spec_head", "dhead_final", [256, 1, 1], [1, 1, 1], &[&am_s, &out, &ARGMAX_PARTS, &dh.ids.ptr()])?;
        }
        Ok(())
    }

    /// Chains drafts[from..k] on the previous draft (device side), then reads all k back (syncs).
    pub fn chain(&mut self, e: &mut Engine, from: usize, k: usize, pos_of_first: u32, out: &mut [u32]) -> Result<()> {
        for j in from..k {
            let tok = self.drafts.ptr().add::<i32>(j - 1);
            let xn = e.sc.xn.ptr();
            let slot = self.drafts.ptr().add::<i32>(j);
            self.step(e, tok, xn, pos_of_first + (j - from) as u32, Some(slot))?;
        }
        let mut d = [0i32; 16];
        e.dev.download(&mut d[..k], self.drafts.ptr())?;
        for i in 0..k {
            out[i] = d[i] as u32;
        }
        Ok(())
    }

    /// The MTP cache entries of positions pos0 .. pos0 + next.len() (next[i]: the token that follows row i) in one pass. The target's
    /// window over exactly these rows just ran: its xn rows are the normed hidden rows and its rope table covers pos0.. .
    /// Clobbers the window's x / xn / kraw / q scratch.
    pub fn absorb_rows(&mut self, e: &mut Engine, next: &[u32], pos0: u32) -> Result<()> {
        let n = next.len() as u32;
        if n == 0 {
            return Ok(());
        }
        ensure!(n <= self.rcap, "absorb of {n} rows (cap {})", self.rcap);
        let eps = e.eps;
        e.dev.q.upload(self.tok.ptr(), next).map_err(|err| anyhow::anyhow!("{err}"))?;
        e.invalidate();
        let (x, xn) = (e.sc.x.ptr(), e.sc.xn.ptr());
        let emb = e.w.emb.ptr();
        let tok = self.tok.ptr();
        e.launch("basic", "embed_bf16", [64, 1, 1], [n, 1, 1], &[&emb, &tok, &x, &HIDDEN])?;
        let row = 2 * HIDDEN as usize;
        for i in 0..n as usize {
            let (xi, xni) = (x.add::<u16>(i * HIDDEN as usize), xn.add::<u16>(i * HIDDEN as usize));
            let (ce, ch) = (self.cat.ptr().add::<u16>(i * row), self.cat.ptr().add::<u16>(i * row + HIDDEN as usize));
            e.launch("rows", "rmsnorm_r", [512, 1, 1], [1, 1, 1], &[&xi, &self.n_emb.ptr(), &ce, &HIDDEN, &eps])?;
            e.launch("rows", "rmsnorm_r", [512, 1, 1], [1, 1, 1], &[&xni, &self.n_hid.ptr(), &ch, &HIDDEN, &eps])?;
        }
        e.proj_rows(&self.fc, self.cat.ptr(), n, x, false)?;
        e.launch("rows", "rmsnorm_r", [512, 1, 1], [n, 1, 1], &[&x, &self.attn.norm.ptr(), &xn, &HIDDEN, &eps])?;
        let s = &e.sc;
        let (qg, kraw, vtmp, ktmp, q, rope) = (s.qg.ptr(), s.kraw.ptr(), s.vtmp.ptr(), s.ktmp.ptr(), s.q.ptr(), s.rope.ptr());
        e.proj_rows(&self.attn.k, xn, n, kraw, false)?;
        e.proj_rows(&self.attn.v, xn, n, vtmp, false)?;
        // the query heads read stale rows and write the q scratch; only the K heads matter here
        e.launch("rows", "attn_prep_r", [256, 1, 1], [HEADS + KV_HEADS, n, 1],
            &[&qg, &kraw, &self.attn.qn.ptr(), &self.attn.kn.ptr(), &rope, &q, &ktmp, &0u64, &eps])?;
        let (kc, vc) = (self.kc.ptr(), self.vc.ptr());
        e.launch("kvq", "kvq_quant8", [32, 1, 1], [n, 1, 1], &[&ktmp, &kc, &pos0])?;
        e.launch("kvq", "kvq_quant8", [32, 1, 1], [n, 1, 1], &[&vtmp, &vc, &pos0])?;
        e.set_pending(false);
        let _ = (KV_DIM, GDN_QKV, MAX_ROWS);
        Ok(())
    }
}

/// Unused placeholder so the module compiles without a draft head when none is configured.
pub fn no_dhead() -> Result<()> {
    bail!("no draft head")
}

#[allow(dead_code)]
fn _args(_: &[&dyn KernelArg]) {}
