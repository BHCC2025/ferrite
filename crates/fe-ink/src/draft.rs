//! The model's own draft head (mtp.safetensors): 3 dense decoder layers over an 8-position block (anchor + 7 mask positions) that
//! attend to the draft K/V cache built from the target's taps, then a Markov head that drafts 7 tokens one after the other.
//! Rounding points follow the publisher's reference implementation (bf16 at every op it rounds).
use crate::ckpt::Ckpt;
use crate::*;
use anyhow::{ensure, Result};
use fe_cuda::{Context, DevPtr, DeviceBuffer, Function, Module, Stream};
use std::path::Path;
use std::sync::Arc;

type Buf = DeviceBuffer<u8>;

/// A buffer that is either plain or backed by a reserved address range whose memory is mapped in steps (no copy and no transient
/// double allocation when it grows: the full draft layer's K/V and RoPE tables grow with the context).
enum GBuf {
    P(Buf),
    V(fe_cuda::VmBuffer),
}
impl GBuf {
    fn ptr(&self) -> DevPtr {
        match self {
            GBuf::P(b) => b.ptr(),
            GBuf::V(v) => v.ptr(),
        }
    }
    fn grow(&mut self, bytes: usize) -> Result<()> {
        if let GBuf::V(v) = self {
            v.grow(bytes)?;
        }
        Ok(())
    }
    fn shrink(&mut self, bytes: usize) -> Result<()> {
        if let GBuf::V(v) = self {
            v.shrink(bytes)?;
        }
        Ok(())
    }
}
fn vm(reserve: usize, init: usize, dev: i32) -> Result<GBuf> {
    let mut v = fe_cuda::VmBuffer::reserve(reserve, dev)?;
    v.grow(init)?;
    Ok(GBuf::V(v))
}
use crate::weights::Lin;
pub const BLOCK: usize = 8;
pub const NDRAFT: usize = 7;
/// capacity of the penalty history table (distinct tokens seen)

/// rows of the K/V ring of the sliding draft layers (window 1024 needs 1024)
const RING: usize = 1024;
/// Bytes of a draft-context snapshot (two sliding layers, K and V rings of RING rows x 1024 bf16).
pub const DRAFT_SNAP_BYTES: usize = 2 * 2 * RING * 1024 * 2;
/// the growing K/V cache of the full draft layer and the RoPE tables grow in steps of this many positions
const STEP_DEFAULT: usize = 4096;
/// growth step (positions)
fn step() -> usize {
    STEP_DEFAULT
}
const CCH: usize = 256; // rows of tap context projected at a time

struct DLayer {
    in_norm: Buf,
    post_norm: Buf,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    qn: Buf,
    kn: Buf,
    kcn: Buf,
    kctx: Lin,
    vctx: Lin,
    dnorm: Buf,
    gate: Lin,
    up: Lin,
    down: Lin,
    kcache: GBuf,
    vcache: GBuf,
    window: u32,
    /// rows of the K/V ring (sliding layers) or 0 = position-indexed growing cache
    ring: usize,
}

/// A lane's draft context while another lane is the current one (see `Draft::select_lane`).
struct DraftLane {
    kc: Vec<GBuf>,
    vc: Vec<GBuf>,
    ctx_len: usize,
    kv_rows: usize,
}

pub struct Draft {
    pub ctx: Arc<Context>,
    rms: Function,
    f_mix: Function,
    f_hnr: Function,
    f_attn: Function,
    f_attn2: Function,
    f_attn2c: Function,
    /// split-K partials of the full-layer attention: [32 splits][8 kv heads][24 rows][132] f32
    apart: Buf,
    /// the split-K full-layer attention
    attn2: bool,
    f_add: Function,
    f_silu: Function,
    f_rin: Function,
    f_gemv: Function,
    f_gates: Function,
    f_bias: Function,
    f_argmax: Function,
    pub samp: crate::sampler::DevSampler,
    layers: Vec<DLayer>,
    alpha: [[f32; 4]; 4],
    mask_embed: Buf,
    norm_w: Buf,
    enorm: Buf,
    fw: Buf,
    fw8: Option<fe_inkk::Fp8Linear>,
    w18: Option<fe_inkk::Fp8Linear>,
    f_rin8: Function,
    f_bias8: Function,
    fb: Buf,
    w1: Buf,
    rnn_w: Buf,
    rnn_b: Buf,
    mnorm: Buf,
    cos: GBuf,
    sin: GBuf,
    // scratch
    x0: Buf,
    erow: Buf,
    res: Buf,
    hn: Buf,
    qlin: Buf,
    qh: Buf,
    klin: Buf,
    kb: Buf,
    vb: Buf,
    ao: Buf,
    aout: Buf,
    g: Buf,
    u: Buf,
    act: Buf,
    mlp: Buf,
    y: Buf,
    base: Buf,
    inp: Buf,
    proj: Buf,
    state: Buf,
    latent: Buf,
    fr: Buf,
    tokbuf: Buf,
    pv: Buf,
    pi: Buf,
    cmix: Buf,
    ctmp: Buf,
    ck: Ckpt,
    pub cap: usize,
    /// positions the full layer's K/V cache and the RoPE tables currently hold (grown in steps)
    kv_rows: usize,
    /// number of context positions in the draft K/V cache (positions 0..ctx_len)
    pub ctx_len: usize,
    /// block attention mask: false = bidirectional (reference default), true = causal inside the block
    pub causal: bool,
    /// lanes: the layers' caches, `ctx_len` and `kv_rows` belong to lane `cur`; the others are parked (None at `cur`)
    cur: usize,
    parked: Vec<Option<DraftLane>>,
    /// positions the shared RoPE tables hold
    tab_rows: usize,
}

/// NeoX RoPE tables (f32 bytes) for positions lo..hi: cos then sin, [pos][64].
fn rope_tables(lo: usize, hi: usize) -> (Vec<u8>, Vec<u8>) {
    let inv: Vec<f32> = (0..64).map(|i| (1.0f64 / 1e6f64.powf(2.0 * i as f64 / 128.0)) as f32).collect();
    let (mut c, mut s) = (Vec::with_capacity((hi - lo) * 256), Vec::with_capacity((hi - lo) * 256));
    for p in lo..hi {
        for i in 0..64 {
            let a = (p as f32) * inv[i];
            c.extend_from_slice(&a.cos().to_le_bytes());
            s.extend_from_slice(&a.sin().to_le_bytes());
        }
    }
    (c, s)
}

fn alloc(b: usize) -> Result<Buf> {
    Ok(Buf::alloc(b.max(16))?)
}

impl Draft {
    /// `model_dir`: the release directory (mtp.safetensors, the main checkpoint for embed / unembed); `fold`: fold.safetensors (lm_head @ hidden_out,
    /// computed in float64 by `crate::fold`); `cap`: maximum positions.
    /// `fp8`: 0 = bf16 head, 1 = layer matrices FP8 (E4M3, per-output-row scale: the engine's dense format), 2 = also markov_w1 and the fold table.
    pub fn load(ctx: Arc<Context>, s: &Stream, model_dir: &Path, fold: &Path, cap: usize, fp8: u8) -> Result<Draft> {
        ctx.bind()?;
        let m = Module::load(K_INKM_CUBIN)?;
        let d = Module::load(K_DRAFT_CUBIN)?;
        let mut mt = Ckpt::open(&model_dir.join("mtp.safetensors"))?;
        let mut ck = Ckpt::open(&model_dir.join("model-00001-of-00001.safetensors"))?;
        let mut fo = Ckpt::open(fold)?;
        let up = |c: &mut Ckpt, n: &str| -> Result<Buf> { Ok(c.upload(n, None)?) };
        let kk = fe_inkk::Kern::load()?;
        // a dense matrix [n][k]: bf16 or FP8 (quantised on the device from the bf16 upload, which is then freed)
        let lin = |c: &mut Ckpt, name: &str| -> Result<Lin> {
            let (n, k) = { let i = c.info(name)?; (i.shape[0], i.shape[1]) };
            let b = c.upload(name, None)?;
            if fp8 >= 1 {
                let f = kk.fp8_from_bf16(s, b.ptr(), n, k)?;
                s.sync()?;
                Ok(Lin { n, k, bf: None, f8: Some(f) })
            } else {
                Ok(Lin { n, k, bf: Some(b), f8: None })
            }
        };
        let init_rows = cap.min(step());
        let mut layers = Vec::new();
        for i in 0..3 {
            let p = format!("layers.{i}.");
            let a = format!("{p}self_attn.");
            let window = if i == 0 || i == 2 { 1024 } else { 0 };
            layers.push(DLayer {
                in_norm: up(&mut mt, &format!("{p}input_layernorm.weight"))?,
                post_norm: up(&mut mt, &format!("{p}post_attention_layernorm.weight"))?,
                q: lin(&mut mt, &format!("{a}q_proj.weight"))?,
                k: lin(&mut mt, &format!("{a}k_proj.weight"))?,
                v: lin(&mut mt, &format!("{a}v_proj.weight"))?,
                o: lin(&mut mt, &format!("{a}o_proj.weight"))?,
                qn: up(&mut mt, &format!("{a}q_norm.weight"))?,
                kn: up(&mut mt, &format!("{a}k_norm.weight"))?,
                kcn: up(&mut mt, &format!("{a}k_norm_ctx.weight"))?,
                kctx: lin(&mut mt, &format!("{a}k_proj_ctx.weight"))?,
                vctx: lin(&mut mt, &format!("{a}v_proj_ctx.weight"))?,
                dnorm: up(&mut mt, &format!("{p}dflare_norm.weight"))?,
                gate: lin(&mut mt, &format!("{p}mlp.gate_proj.weight"))?,
                up: lin(&mut mt, &format!("{p}mlp.up_proj.weight"))?,
                down: lin(&mut mt, &format!("{p}mlp.down_proj.weight"))?,
                kcache: if window > 0 { GBuf::P(alloc(RING * 1024 * 2)?) } else { vm((cap + 32) * 1024 * 2, init_rows * 1024 * 2, ctx.device_id())? },
                vcache: if window > 0 { GBuf::P(alloc(RING * 1024 * 2)?) } else { vm((cap + 32) * 1024 * 2, init_rows * 1024 * 2, ctx.device_id())? },
                window,
                ring: if window > 0 { RING } else { 0 },
            });
        }
        // alpha = bf16(softmax(fuse_weight rows))
        let fwt = mt.read_f32("fuse_weight")?;
        let mut alpha = [[0.0f32; 4]; 4];
        for r in 0..4 {
            let row = &fwt[r * 4..r * 4 + 4];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = row.iter().map(|x| (x - mx).exp()).collect();
            let sm: f32 = e.iter().sum();
            for t in 0..4 {
                let v = e[t] / sm;
                alpha[r][t] = f32::from_bits(((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) << 16);
            }
        }
        let mut cosv = vec![0f32; init_rows * 64];
        let mut sinv = vec![0f32; init_rows * 64];
        let inv: Vec<f32> = (0..64).map(|i| (1.0f64 / 1e6f64.powf(2.0 * i as f64 / 128.0)) as f32).collect();
        for p in 0..init_rows {
            for i in 0..64 {
                let a = (p as f32) * inv[i];
                cosv[p * 64 + i] = a.cos();
                sinv[p * 64 + i] = a.sin();
            }
        }
        let nblk = (VOCAB + 7) / 8;
        let dr = Draft {
            rms: m.function("ink_rms")?,
            f_mix: d.function("dr_mix")?,
            f_hnr: d.function("dr_hnr")?,
            f_attn: d.function("dr_attn")?,
            f_attn2: {
                let f = d.function("dr_attn_split")?;
                f.set_max_dynamic_smem(50176)?;
                f
            },
            f_attn2c: d.function("dr_attn_comb")?,
            apart: alloc(32 * 8 * 24 * 132 * 4)?,
            attn2: true,
            f_add: d.function("dr_add")?,
            f_silu: d.function("dr_silu_mul")?,
            f_rin: d.function("dr_rnn_in")?,
            f_gemv: d.function("dr_gemv_bias")?,
            f_gates: d.function("dr_rnn_gates")?,
            f_bias: d.function("dr_bias_step")?,
            f_argmax: d.function("dr_argmax")?,
            samp: crate::sampler::DevSampler::new(ctx.clone())?,
            layers,
            alpha,
            mask_embed: up(&mut mt, "mask_embed")?,
            norm_w: up(&mut mt, "norm.weight")?,
            enorm: ck.upload("model.llm.embed_norm.weight", None)?,
            fw: if fp8 >= 2 { Buf::alloc(16)? } else { up(&mut fo, "fold_weight")? },
            fw8: if fp8 >= 2 { let b = fo.upload("fold_weight", None)?; let f = kk.fp8_from_bf16(s, b.ptr(), VOCAB, 256)?; s.sync()?; Some(f) } else { None },
            fb: up(&mut fo, "fold_bias")?,
            w1: if fp8 >= 2 { Buf::alloc(16)? } else { mt.upload("markov_head.markov_w1.weight", None)? },
            w18: if fp8 >= 2 { let (n, k) = { let i = mt.info("markov_head.markov_w1.weight")?; (i.shape[0], i.shape[1]) }; let b = mt.upload("markov_head.markov_w1.weight", None)?; let f = kk.fp8_from_bf16(s, b.ptr(), n, k)?; s.sync()?; Some(f) } else { None },
            f_rin8: d.function("dr_rnn_in8")?,
            f_bias8: d.function("dr_bias_step8")?,
            rnn_w: up(&mut mt, "markov_head.rnn_proj.weight")?,
            rnn_b: up(&mut mt, "markov_head.rnn_proj.bias")?,
            mnorm: up(&mut mt, "markov_head.dflare_norm.weight")?,
            cos: { let g = vm((cap + 32) * 64 * 4, init_rows * 64 * 4, ctx.device_id())?; unsafe { s.memcpy_htod(g.ptr(), &cosv.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())? }; g },
            sin: { let g = vm((cap + 32) * 64 * 4, init_rows * 64 * 4, ctx.device_id())?; unsafe { s.memcpy_htod(g.ptr(), &sinv.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())? }; g },
            x0: alloc(BLOCK * D * 2)?,
            erow: alloc(D * 2)?,
            res: alloc(BLOCK * D * 2)?,
            hn: alloc(BLOCK * D * 2)?,
            qlin: alloc(BLOCK * 3072 * 2)?,
            qh: alloc(BLOCK * 3072 * 2)?,
            klin: alloc(BLOCK * 1024 * 2)?,
            kb: alloc(BLOCK * 1024 * 2)?,
            vb: alloc(BLOCK * 1024 * 2)?,
            ao: alloc(BLOCK * 3072 * 2)?,
            aout: alloc(BLOCK * D * 2)?,
            g: alloc(BLOCK * 9216 * 2)?,
            u: alloc(BLOCK * 9216 * 2)?,
            act: alloc(BLOCK * 9216 * 2)?,
            mlp: alloc(BLOCK * D * 2)?,
            y: alloc(BLOCK * D * 2)?,
            base: alloc(NDRAFT * VOCAB * 2)?,
            inp: alloc(5120 * 2)?,
            proj: alloc(768 * 4)?,
            state: alloc(256 * 2)?,
            latent: alloc(256 * 2)?,
            fr: alloc(D * 2)?,
            tokbuf: alloc(8 * 4)?,
            pv: alloc(nblk * 4)?,
            pi: alloc(nblk * 4)?,
            cmix: alloc(CCH * D * 2)?,
            ctmp: alloc(CCH * 1024 * 2)?,
            ck,
            cap,
            kv_rows: init_rows,
            ctx_len: 0,
            cur: 0,
            parked: vec![None],
            tab_rows: init_rows,
            causal: false,
            ctx,
        };
        // block rows 1..7 = mask_embed (constant)
        for r in 1..BLOCK {
            s.memcpy_dtod(dr.x0.ptr().byte_add(r * D * 2), dr.mask_embed.ptr(), D * 2)?;
        }
        s.sync()?;
        Ok(dr)
    }

    /// Give the head `n` lanes in all (each its own K/V caches: two rings + the growing full layer).
    pub fn add_lanes(&mut self, n: usize) -> Result<()> {
        let dev = self.ctx.device_id();
        let init_rows = self.cap.min(step());
        while self.parked.len() < n {
            let (mut kc, mut vc) = (vec![], vec![]);
            for l in &self.layers {
                if l.ring > 0 {
                    kc.push(GBuf::P(alloc(RING * 1024 * 2)?));
                    vc.push(GBuf::P(alloc(RING * 1024 * 2)?));
                } else {
                    kc.push(vm((self.cap + 32) * 1024 * 2, init_rows * 1024 * 2, dev)?);
                    vc.push(vm((self.cap + 32) * 1024 * 2, init_rows * 1024 * 2, dev)?);
                }
            }
            self.parked.push(Some(DraftLane { kc, vc, ctx_len: 0, kv_rows: init_rows }));
        }
        Ok(())
    }

    /// The full draft layer's K and V caches with the bytes of positions [0, m).
    pub fn full_prefix(&self, m: usize) -> Vec<(DevPtr, usize)> {
        let mut v = vec![];
        for l in self.layers.iter().filter(|l| l.ring == 0) {
            v.push((l.kcache.ptr(), m * 1024 * 2));
            v.push((l.vcache.ptr(), m * 1024 * 2));
        }
        v
    }

    /// Take the physical memory of the current lane's full draft layer out (see `Ink::kv_detach`); returns it with the positions it holds.
    /// The lane gets fresh memory for one growth step.
    pub fn full_detach(&mut self, st: &Stream) -> Result<(Vec<fe_cuda::VmHandles>, usize)> {
        st.sync()?;
        let mut v = vec![];
        for l in self.layers.iter_mut().filter(|l| l.ring == 0) {
            for b in [&mut l.kcache, &mut l.vcache] {
                match b {
                    GBuf::V(vb) => v.push(vb.detach()?),
                    GBuf::P(_) => anyhow::bail!("full_detach: the full layer cache is not a mapped range"),
                }
            }
        }
        let rows = self.kv_rows;
        self.kv_rows = 0;
        self.ctx_len = 0;
        self.ensure_capacity(st, 1)?;
        Ok((v, rows))
    }

    /// Put memory taken by `full_detach` back (the lane's own is released first); `rows` = the positions it holds.
    pub fn full_attach(&mut self, st: &Stream, h: Vec<fe_cuda::VmHandles>, rows: usize) -> Result<()> {
        st.sync()?;
        let mut it = h.into_iter();
        for l in self.layers.iter_mut().filter(|l| l.ring == 0) {
            for b in [&mut l.kcache, &mut l.vcache] {
                match b {
                    GBuf::V(vb) => {
                        vb.shrink(0)?;
                        vb.attach(it.next().ok_or_else(|| anyhow::anyhow!("full_attach: too few buffers"))?)?;
                    }
                    GBuf::P(_) => anyhow::bail!("full_attach: the full layer cache is not a mapped range"),
                }
            }
        }
        self.kv_rows = rows;
        Ok(())
    }

    /// Lanes the head has caches for.
    pub fn n_lanes(&self) -> usize {
        self.parked.len()
    }

    /// Make lane `i`'s draft context the current one (a swap of the cache objects, no device work).
    pub fn select_lane(&mut self, i: usize) -> Result<()> {
        ensure!(i < self.parked.len(), "draft lane {i} of {}", self.parked.len());
        if i == self.cur {
            return Ok(());
        }
        let mut st = self.parked[i].take().unwrap();
        for (j, l) in self.layers.iter_mut().enumerate() {
            std::mem::swap(&mut l.kcache, &mut st.kc[j]);
            std::mem::swap(&mut l.vcache, &mut st.vc[j]);
        }
        std::mem::swap(&mut self.ctx_len, &mut st.ctx_len);
        std::mem::swap(&mut self.kv_rows, &mut st.kv_rows);
        self.parked[self.cur] = Some(st);
        self.cur = i;
        Ok(())
    }

    /// Device bytes held by the draft's growing buffers (full-layer K/V + RoPE tables) at the current size.
    pub fn grown_bytes(&self) -> usize {
        self.kv_rows * (2 * 1024 * 2 + 64 * 4 * 2)
    }

    /// Bytes of one prompt-cache snapshot of the draft context ([`DRAFT_SNAP_BYTES`]).
    pub fn snap_bytes(&self) -> usize {
        self.layers.iter().filter(|l| l.ring > 0).count() * 2 * RING * 1024 * 2
    }

    /// Copy the K/V rings of the sliding draft layers (positions [ctx_len - 1024, ctx_len)) into `dst` (pinned host); returns ctx_len. The full
    /// draft layer's cache is position-indexed and only appended to, like the target's global K/V.
    pub fn snapshot(&self, st: &Stream, dst: &mut [u8]) -> Result<usize> {
        ensure!(dst.len() >= self.snap_bytes(), "draft snapshot buffer too small");
        st.sync()?;
        let n = RING * 1024 * 2;
        let mut off = 0;
        for l in self.layers.iter().filter(|l| l.ring > 0) {
            for b in [&l.kcache, &l.vcache] {
                unsafe { st.memcpy_dtoh(&mut dst[off..off + n], b.ptr())? };
                off += n;
            }
        }
        st.sync()?;
        Ok(self.ctx_len)
    }

    /// Put back a snapshot of the draft context taken at `ctx_len` positions (see [`Draft::snapshot`]).
    pub fn restore(&mut self, st: &Stream, src: &[u8], ctx_len: usize) -> Result<()> {
        ensure!(src.len() >= self.snap_bytes(), "draft snapshot buffer too small");
        ensure!(ctx_len <= self.kv_rows, "draft snapshot at {ctx_len} beyond the grown cache {}", self.kv_rows);
        st.sync()?;
        let n = RING * 1024 * 2;
        let mut off = 0;
        for l in self.layers.iter().filter(|l| l.ring > 0) {
            for b in [&l.kcache, &l.vcache] {
                unsafe { st.memcpy_htod(b.ptr(), &src[off..off + n])? };
                off += n;
            }
        }
        st.sync()?;
        self.ctx_len = ctx_len;
        Ok(())
    }

    /// Gate control only (INK_PCACHE_PLANT=draft): set the context length without restoring the rings.
    pub fn restore_len_only(&mut self, ctx_len: usize) {
        self.ctx_len = ctx_len;
    }

    pub fn reset(&mut self, st: &Stream) -> Result<()> {
        self.ctx_len = 0;
        // give the grown buffers back (a new request starts small; `ensure_capacity` regrows with its context)
        let init_rows = self.cap.min(step());
        if self.kv_rows > init_rows {
            st.sync()?;
            for l in self.layers.iter_mut().filter(|l| l.ring == 0) {
                l.kcache.shrink(init_rows * 1024 * 2)?;
                l.vcache.shrink(init_rows * 1024 * 2)?;
            }
            // the RoPE tables are shared by the lanes: they shrink only with one lane
            if self.parked.len() == 1 {
                self.cos.shrink(init_rows * 64 * 4)?;
                self.sin.shrink(init_rows * 64 * 4)?;
                self.tab_rows = init_rows;
            }
            self.kv_rows = init_rows;
        }
        Ok(())
    }

    /// Make the full layer's K/V cache and the RoPE tables hold `need` positions, growing in steps of 4096 positions (16 MiB of K/V + 2 MiB of tables each).
    /// For the 16 GB layout call `Ink::set_host_experts(n)` (frees expert blocks) BEFORE a growth that would not fit.
    pub fn ensure_capacity(&mut self, st: &Stream, need: usize) -> Result<()> {
        if need <= self.kv_rows {
            return Ok(());
        }
        let max_rows = self.cap + 32;
        ensure!(need <= max_rows, "draft cache: {} positions exceed cap {}", need, self.cap);
        let stp = step();
        let new_rows = ((need + stp - 1) / stp * stp).min(max_rows);
        st.sync()?;
        for l in self.layers.iter_mut().filter(|l| l.ring == 0) {
            l.kcache.grow(new_rows * 1024 * 2)?;
            l.vcache.grow(new_rows * 1024 * 2)?;
        }
        if new_rows > self.tab_rows {
            let (c, sn) = rope_tables(self.tab_rows, new_rows);
            self.cos.grow(new_rows * 64 * 4)?;
            self.sin.grow(new_rows * 64 * 4)?;
            unsafe {
                st.memcpy_htod(self.cos.ptr().byte_add(self.tab_rows * 64 * 4), &c)?;
                st.memcpy_htod(self.sin.ptr().byte_add(self.tab_rows * 64 * 4), &sn)?;
            }
            self.tab_rows = new_rows;
        }
        st.sync()?;
        self.kv_rows = new_rows;
        Ok(())
    }

    /// C[m][n] = A[m][k] . W[n][k]^T (bf16), the engine's rule: <= 16 rows the small-M kernel, else the tile kernel.
    fn rmsn(&self, st: &Stream, y: DevPtr, x: DevPtr, w: DevPtr, rows: usize) -> Result<()> {
        unsafe { st.launch(&self.rms, rows as u32, 256u32, 0, (y, x, 0u32, D as u32, w, D as u32, 1e-6f32, D as u32))? };
        Ok(())
    }

    /// Append the draft context K/V of `c` new positions pos0..pos0+c. `taps`: device bf16 [c][4][2304] (taps of those positions).
    pub fn extend(&mut self, ink: &crate::engine::Ink, taps: DevPtr, c: usize, pos0: usize) -> Result<()> {
        let st = &ink.s;
        ensure!(pos0 + c <= self.cap, "draft cache: position {} exceeds cap {}", pos0 + c, self.cap);
        ensure!(pos0 == self.ctx_len, "draft context must be extended in order ({} != {})", pos0, self.ctx_len);
        self.ensure_capacity(st, pos0 + c + BLOCK)?;
        let mut off = 0;
        while off < c {
            // chunks do not cross the ring boundary of the sliding layers (their V rows are written by one GEMM)
            let r = (c - off).min(CCH).min(RING - (pos0 + off) % RING);
            for (i, l) in self.layers.iter().enumerate() {
                let al = self.alpha[i];
                unsafe {
                    st.launch(&self.f_mix, r as u32, 256u32, 0, (self.cmix.ptr(), taps.byte_add(off * 4 * D * 2), l.dnorm.ptr(), al[0], al[1], al[2], al[3],
                        (4 * D) as u32))?;
                }
                ink.lin(st, self.ctmp.ptr(), self.cmix.ptr(), &l.kctx, r, D, 1024)?;
                unsafe {
                    st.launch(&self.f_hnr, ((r * 8 + 7) / 8) as u32, 256u32, 0, (l.kcache.ptr(), self.ctmp.ptr(), l.kcn.ptr(), self.cos.ptr(), self.sin.ptr(),
                        8u32, r as u32, 1024u32, 1024u32, (pos0 + off) as u32, (pos0 + off) as u32, l.ring as u32))?;
                }
                ink.lin(st, l.vcache.ptr().byte_add(if l.ring > 0 { (pos0 + off) % l.ring } else { pos0 + off } * 1024 * 2), self.cmix.ptr(), &l.vctx, r, D, 1024)?;
            }
            off += r;
        }
        self.ctx_len = pos0 + c;
        Ok(())
    }

    /// Draft 7 tokens. The anchor (position n - 1, not yet processed by the target) with n - 1 = `ctx_len` positions of context in the cache;
    /// `last_taps`: device bf16 [1][4][2304], the taps of position n - 2 (the last processed token), used for the Markov head's f_rnn.
    pub fn draft(&mut self, ink: &crate::engine::Ink, anchor: u32, last_taps: DevPtr) -> Result<[u32; NDRAFT]> {
        let st = &ink.s;
        let n = self.ctx_len + 1;
        let qpos0 = (n - 1) as u32;
        ensure!(n - 1 + BLOCK <= self.cap + 32, "draft cache too small for the block");
        self.ensure_capacity(&ink.s, n + BLOCK)?;
        // block input: row 0 = embed_norm(embed[anchor]), rows 1..7 = mask_embed (preloaded in x0)
        let row = self.ck.read_range("model.llm.embed.weight", anchor as usize * D * 2, D * 2)?;
        unsafe { st.memcpy_htod(self.erow.ptr(), &row)? };
        self.rmsn(st, self.x0.ptr(), self.erow.ptr(), self.enorm.ptr(), 1)?;
        st.memcpy_dtod(self.res.ptr(), self.x0.ptr(), BLOCK * D * 2)?;
        let nl = self.layers.len();
        for i in 0..nl {
            let l = &self.layers[i];
            if i > 0 {
                // residual += previous mlp output
                unsafe { st.launch(&self.f_add, 36u32, 256u32, 0, (self.res.ptr(), self.mlp.ptr(), self.res.ptr(), (BLOCK * D) as u32))? };
            }
            self.rmsn(st, self.hn.ptr(), self.res.ptr(), l.in_norm.ptr(), BLOCK)?;
            ink.lin(st, self.qlin.ptr(), self.hn.ptr(), &l.q, BLOCK, D, 3072)?;
            ink.lin(st, self.klin.ptr(), self.hn.ptr(), &l.k, BLOCK, D, 1024)?;
            ink.lin(st, self.vb.ptr(), self.hn.ptr(), &l.v, BLOCK, D, 1024)?;
            unsafe {
                st.launch(&self.f_hnr, 24u32, 256u32, 0, (self.qh.ptr(), self.qlin.ptr(), l.qn.ptr(), self.cos.ptr(), self.sin.ptr(), 24u32, BLOCK as u32,
                    3072u32, 3072u32, 0u32, qpos0, 0u32))?;
                st.launch(&self.f_hnr, 8u32, 256u32, 0, (self.kb.ptr(), self.klin.ptr(), l.kn.ptr(), self.cos.ptr(), self.sin.ptr(), 8u32, BLOCK as u32,
                    1024u32, 1024u32, 0u32, qpos0, 0u32))?;
                if self.attn2 && l.window == 0 && l.ring == 0 && self.ctx_len >= 2048 {
                    st.launch(&self.f_attn2, (32u32, 8u32), 256u32, 50176, (self.apart.ptr(), self.qh.ptr(), l.kcache.ptr(), l.vcache.ptr(), self.kb.ptr(), self.vb.ptr(),
                        self.ctx_len as u32, self.causal as u32, 32u32))?;
                    st.launch(&self.f_attn2c, (24u32, BLOCK as u32), 128u32, 0, (self.ao.ptr(), self.apart.ptr(), 32u32))?;
                } else {
                    st.launch(&self.f_attn, (24u32, BLOCK as u32), 256u32, 0, (self.ao.ptr(), self.qh.ptr(), l.kcache.ptr(), l.vcache.ptr(), self.kb.ptr(),
                        self.vb.ptr(), self.ctx_len as u32, qpos0, l.window, self.causal as u32, l.ring as u32))?;
                }
            }
            ink.lin(st, self.aout.ptr(), self.ao.ptr(), &l.o, BLOCK, 3072, D)?;
            unsafe { st.launch(&self.f_add, 36u32, 256u32, 0, (self.res.ptr(), self.aout.ptr(), self.res.ptr(), (BLOCK * D) as u32))? };
            self.rmsn(st, self.hn.ptr(), self.res.ptr(), l.post_norm.ptr(), BLOCK)?;
            ink.lin(st, self.g.ptr(), self.hn.ptr(), &l.gate, BLOCK, D, 9216)?;
            ink.lin(st, self.u.ptr(), self.hn.ptr(), &l.up, BLOCK, D, 9216)?;
            unsafe { st.launch(&self.f_silu, 144u32, 256u32, 0, (self.act.ptr(), self.g.ptr(), self.u.ptr(), (BLOCK * 9216) as u32))? };
            ink.lin(st, self.mlp.ptr(), self.act.ptr(), &l.down, BLOCK, 9216, D)?;
        }
        unsafe { st.launch(&self.f_add, 36u32, 256u32, 0, (self.res.ptr(), self.mlp.ptr(), self.res.ptr(), (BLOCK * D) as u32))? };
        self.rmsn(st, self.y.ptr(), self.res.ptr(), self.norm_w.ptr(), BLOCK)?;
        // base logits of rows 1..7 through the (bf16) lm_head
        ink.unembed(self.base.ptr(), self.y.ptr().byte_add(D * 2), NDRAFT)?;
        // Markov head
        let al = self.alpha[3];
        unsafe {
            st.launch(&self.f_mix, 1u32, 256u32, 0, (self.fr.ptr(), last_taps, self.mnorm.ptr(), al[0], al[1], al[2], al[3], (4 * D) as u32))?;
            st.memset_d8(self.state.ptr(), 0, 256 * 2)?;
            st.memcpy_htod(self.tokbuf.ptr(), &[anchor])?;
        }
        let nblk = (VOCAB + 7) / 8;
        for j in 0..NDRAFT {
            unsafe {
                if let Some(w1) = &self.w18 {
                    st.launch(&self.f_rin8, 1u32, 256u32, 0, (self.inp.ptr(), self.state.ptr(), w1.w8.ptr(), w1.scale.ptr(), self.tokbuf.ptr().byte_add(4 * j),
                        self.y.ptr().byte_add((j + 1) * D * 2), self.fr.ptr()))?;
                } else {
                    st.launch(&self.f_rin, 1u32, 256u32, 0, (self.inp.ptr(), self.state.ptr(), self.w1.ptr(), self.tokbuf.ptr().byte_add(4 * j),
                        self.y.ptr().byte_add((j + 1) * D * 2), self.fr.ptr()))?;
                }
                st.launch(&self.f_gemv, 768u32, 256u32, 0, (self.proj.ptr(), self.inp.ptr(), self.rnn_w.ptr(), self.rnn_b.ptr(), 5120u32))?;
                st.launch(&self.f_gates, 1u32, 256u32, 0, (self.state.ptr(), self.latent.ptr(), self.proj.ptr()))?;
                if let Some(fw) = &self.fw8 {
                    st.launch(&self.f_bias8, nblk as u32, 256u32, 0, (self.pv.ptr(), self.pi.ptr(), self.latent.ptr(), fw.w8.ptr(), fw.scale.ptr(), self.fb.ptr(),
                        self.base.ptr().byte_add(j * VOCAB * 2), VOCAB as u32))?;
                } else {
                    st.launch(&self.f_bias, nblk as u32, 256u32, 0, (self.pv.ptr(), self.pi.ptr(), self.latent.ptr(), self.fw.ptr(), self.fb.ptr(),
                        self.base.ptr().byte_add(j * VOCAB * 2), VOCAB as u32))?;
                }
                st.launch(&self.f_argmax, 1u32, 1024u32, 0, (self.tokbuf.ptr(), j as u32, self.pv.ptr(), self.pi.ptr(), nblk as u32))?;
            }
        }
        st.sync()?;
        let mut out = [0u32; 8];
        fe_cuda::memcpy_dtoh(&mut out, self.tokbuf.ptr())?;
        let mut r = [0u32; NDRAFT];
        r.copy_from_slice(&out[1..8]);
        Ok(r)
    }

    /// Temperature-only lossless speculative sampling (see `DevSampler::sample_rows`).
    pub fn sample_rows(&mut self, st: &Stream, logits: DevPtr, drafts: &[u32], inv_t: f32, u: impl FnMut() -> f64) -> Result<(usize, u32)> {
        self.samp.sample_rows(st, logits, drafts, inv_t, u)
    }

    pub fn sample_rows_filtered(&mut self, st: &Stream, logits: DevPtr, drafts: &[u32], cfg: &[u32; 8], tab: &[[u32; 3]], u: impl FnMut() -> f64) -> Result<(usize, u32)> {
        self.samp.sample_rows_filtered(st, logits, drafts, cfg, tab, &[], u)
    }

    pub fn upload_row(&mut self, st: &Stream, row: &[u16]) -> Result<DevPtr> {
        self.samp.upload_row(st, row)
    }
}
