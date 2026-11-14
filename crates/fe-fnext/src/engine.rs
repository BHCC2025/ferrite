//! The Flash-Next forward on one card, f32 activations. Correctness path: always-used weights dequantized to f16
//! (bf16 tensors kept as bf16) at load and run through a plain GEMV; experts on the CPU from the mmap (plain
//! f32 dot products over dequantized rows). One sequence; any number of rows per forward (a prompt chunk or one
//! decode token). Named intermediate tensors can be captured for the check against the llama.cpp reference.
use crate::config::Config;
use anyhow::{anyhow, bail, Result};
use fe_cuda::{DevPtr, DeviceBuffer, Function, Module, Stream};
use fe_ggq::refq;
use fe_gguf::{GgmlType, Gguf, TensorInfo};
use rayon::prelude::*;
use std::collections::HashMap;

pub struct Kern {
    pub(crate) _m: Module,
    pub(crate) gemv: Function,
    pub(crate) rmsnorm: Function,
    pub(crate) silu: Function,
    pub(crate) swiglu: Function,
    pub(crate) hc_init: Function,
    pub(crate) hc_mix: Function,
    pub(crate) hc_combine: Function,
    pub(crate) ple_gate: Function,
    pub(crate) ple_conv: Function,
    pub(crate) gdn_conv: Function,
    pub(crate) gdn_rec: Function,
    pub(crate) gdn_gnorm: Function,
    pub(crate) attn_prep: Function,
    pub(crate) attn: Function,
    pub(crate) moe_out: Function,
    pub(crate) rope_rows: Function,
    pub(crate) idx_pool: Function,
    pub(crate) idx_score: Function,
    pub(crate) emu_round: Function,
    pub(crate) conv_shift: Function,
    /// FNEXT_EMU: 0 off; 1 = round GEMV inputs of quantized weights through q8_1 like the reference's MMQ/MMVQ;
    /// 2 = also bf16 inputs for bf16 weights (checks only)
    pub emu: u32,
    pub(crate) emu_buf: DeviceBuffer<f32>,
}

pub(crate) fn g256(n: usize) -> u32 {
    n.div_ceil(256) as u32
}

impl Kern {
    pub fn load() -> Result<Kern> {
        let m = Module::load(crate::K_FNEXT_CUBIN)?;
        let f = |n: &str| m.function(n);
        Ok(Kern {
            gemv: f("fn_gemv_h")?,
            rmsnorm: f("fn_rmsnorm")?,
            silu: f("fn_silu")?,
            swiglu: f("fn_swiglu")?,
            hc_init: f("fn_hc_init")?,
            hc_mix: f("fn_hc_mix")?,
            hc_combine: f("fn_hc_combine")?,
            ple_gate: f("fn_ple_gate")?,
            ple_conv: f("fn_ple_conv")?,
            gdn_conv: f("fn_gdn_conv")?,
            gdn_rec: f("fn_gdn_rec")?,
            gdn_gnorm: f("fn_gdn_gnorm")?,
            attn_prep: f("fn_attn_prep")?,
            attn: f("fn_attn")?,
            moe_out: f("fn_moe_out")?,
            rope_rows: f("fn_rope_rows")?,
            idx_pool: f("fn_idx_pool")?,
            idx_score: f("fn_idx_score")?,
            emu_round: f("fn_emu_round")?,
            conv_shift: f("fn_conv_shift")?,
            emu: std::env::var("FNEXT_EMU").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            emu_buf: DeviceBuffer::zeroed(if std::env::var("FNEXT_EMU").is_ok() { 1024 * 12288 } else { 32 })?,
            _m: m,
        })
    }
    /// y = x * rsqrt(mean(x^2)+eps) * w over `groups` groups of `d` (weight period ngw groups).
    pub(crate) unsafe fn rmsnorm(&self, s: &Stream, y: DevPtr, x: DevPtr, w: DevPtr, d: usize, ngw: usize, groups: usize, eps: f32) -> Result<()> {
        // few groups (decode rows): wider blocks, the norm is latency-bound there
        let bs = if groups <= 32 && std::env::var("FNEXT_NORM_NARROW").is_err() { 1024 } else { 256 };
        s.launch(&self.rmsnorm, groups as u32, bs, 0, (y, x, w, d as u32, ngw as u32, eps))?;
        Ok(())
    }
    /// The decode-step norm: one block size for every group count, so a row's bits never depend on how many rows
    /// share the step (rmsnorm switches to narrow blocks above 32 groups, i.e. above 8 hc rows).
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn rmsnorm_dec(&self, s: &Stream, y: DevPtr, x: DevPtr, w: DevPtr, d: usize, ngw: usize, groups: usize,
        eps: f32) -> Result<()> {
        let bs = if std::env::var("FNEXT_NORM_NARROW").is_err() { 1024 } else { 256 };
        s.launch(&self.rmsnorm, groups as u32, bs, 0, (y, x, w, d as u32, ngw as u32, eps))?;
        Ok(())
    }
    pub(crate) unsafe fn silu(&self, s: &Stream, x: DevPtr, n: usize) -> Result<()> {
        s.launch(&self.silu, g256(n), 256, 0, (x, n as u32))?;
        Ok(())
    }
}

/// A linear layer y = W x with W [n][k] held on the card as f16 (or the file's bf16).
pub struct Lin {
    pub w: DeviceBuffer<u16>,
    pub k: usize,
    pub n: usize,
    pub bf: bool,
    /// dequantized from a quant type (the reference quantizes the activations of those to q8_1)
    pub quant: bool,
}

impl Lin {
    /// From raw rows (n rows of k) of a GGUF type.
    pub fn from_bytes(ty: GgmlType, bytes: &[u8], k: usize, n: usize) -> Result<Lin> {
        let rb = ty.row_bytes(k);
        if bytes.len() < rb * n {
            bail!("Lin: {} bytes for {n} rows of {k} {:?}", bytes.len(), ty);
        }
        let bf = ty == GgmlType::BF16;
        // rows padded to a multiple of 64 with zeros (the prompt GEMM tiles 64 output rows)
        let mut h = vec![0u16; k * n.next_multiple_of(64)];
        if bf || ty == GgmlType::F16 {
            for (i, v) in h[..k * n].iter_mut().enumerate() {
                *v = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
            }
        } else {
            h[..k * n].par_chunks_mut(k).enumerate().for_each(|(r, out)| {
                let mut f = vec![0f32; k];
                refq::dequant(ty, &bytes[r * rb..(r + 1) * rb], &mut f);
                for (o, v) in out.iter_mut().zip(&f) {
                    *o = refq::f32_to_f16(*v);
                }
            });
        }
        let quant = !bf && ty != GgmlType::F16 && ty != GgmlType::F32;
        Ok(Lin { w: DeviceBuffer::from_slice(&h)?, k, n, bf, quant })
    }
    pub fn from_tensor(g: &Gguf, name: &str) -> Result<Lin> {
        let t = tensor(g, name)?;
        let k = t.ne0();
        let n = t.n_rows();
        Self::from_bytes(t.ty, g.data(t), k, n)
    }
    /// Two tensors with the same input stacked as one (rows of a, then rows of b).
    pub fn stacked(g: &Gguf, a: &str, b: &str) -> Result<Lin> {
        let (ta, tb) = (tensor(g, a)?, tensor(g, b)?);
        if ta.ty != tb.ty || ta.ne0() != tb.ne0() {
            bail!("cannot stack {a} and {b}");
        }
        let mut bytes = g.data(ta).to_vec();
        bytes.extend_from_slice(g.data(tb));
        Self::from_bytes(ta.ty, &bytes, ta.ne0(), ta.n_rows() + tb.n_rows())
    }
    /// y[rows][n] = (acc ? y : 0) + scale * W x[rows][k].
    pub unsafe fn run(&self, kn: &Kern, s: &Stream, x: DevPtr, y: DevPtr, rows: usize, scale: f32, acc: bool) -> Result<()> {
        let mode = if self.quant && kn.emu >= 1 { 1 } else if self.bf && kn.emu >= 2 { 2 } else { 0 };
        let x = if mode > 0 && rows * self.k <= kn.emu_buf.len() && self.k % 32 == 0 {
            let n = rows * self.k;
            s.launch(&kn.emu_round, g256(n), 256, 0, (kn.emu_buf.ptr(), x, n as u32, mode as u32))?;
            kn.emu_buf.ptr()
        } else {
            x
        };
        s.launch(&kn.gemv, (self.n.div_ceil(8) as u32, rows.div_ceil(4) as u32), 256, 0,
            (y, x, self.w.ptr(), self.k as u32, self.n as u32, rows as u32, scale, self.bf as u32, acc as u32))?;
        Ok(())
    }
}

pub(crate) fn tensor<'a>(g: &'a Gguf, name: &str) -> Result<&'a TensorInfo> {
    g.tensor(name).ok_or_else(|| anyhow!("tensor {name} not in the GGUF"))
}

/// A small tensor (norm weights, biases, conv kernels) as f32 on the card.
pub(crate) fn vecf(g: &Gguf, name: &str) -> Result<DeviceBuffer<f32>> {
    let t = tensor(g, name)?;
    let mut f = vec![0f32; t.n_elements() as usize];
    refq::dequant(t.ty, g.data(t), &mut f);
    Ok(DeviceBuffer::from_slice(&f)?)
}

pub(crate) struct Hc {
    pub(crate) norm: DeviceBuffer<f32>,
    pub(crate) down: Lin,
    pub(crate) up: Lin,
    pub(crate) inject: Option<Lin>,
}
impl Hc {
    pub(crate) fn load(g: &Gguf, pre: &str, norm: &str, down: &str, up: &str, inject: Option<&str>) -> Result<Hc> {
        Ok(Hc {
            norm: vecf(g, &format!("{pre}{norm}"))?,
            down: Lin::from_tensor(g, &format!("{pre}{down}"))?,
            up: Lin::from_tensor(g, &format!("{pre}{up}"))?,
            inject: inject.map(|i| Lin::from_tensor(g, &format!("{pre}{i}"))).transpose()?,
        })
    }
}

struct Gdn {
    qkv: Lin,
    z: Lin,
    ab: Lin,
    conv_w: DeviceBuffer<f32>,
    dt: DeviceBuffer<f32>,
    a: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    out: Lin,
    conv_state: DeviceBuffer<f32>,
    ssm: DeviceBuffer<f32>,
}

struct Attn {
    wq: Lin,
    wk: Lin,
    wv: Lin,
    wo: Lin,
    qn: DeviceBuffer<f32>,
    kn: DeviceBuffer<f32>,
    kc: DeviceBuffer<f32>,
    vc: DeviceBuffer<f32>,
    // QSA indexer: projections, norms, raw keys per cell, pooled (normed, roped) keys per block of kpool cells
    iq: Lin,
    ik: Lin,
    iqn: DeviceBuffer<f32>,
    ikn: DeviceBuffer<f32>,
    raw: DeviceBuffer<f32>,
    pooled: DeviceBuffer<f32>,
}

enum Mixer {
    Gdn(Gdn),
    Attn(Attn),
}

struct Ffn {
    gate_inp: Lin,
    sh_gate: Lin,
    sh_up: Lin,
    sh_down: Lin,
    sh_ginp: Lin,
}

struct Ple {
    key: Lin,
    value: Lin,
    nk: DeviceBuffer<f32>,
    nq: DeviceBuffer<f32>,
    nc: DeviceBuffer<f32>,
    conv_w: DeviceBuffer<f32>,
    hist: DeviceBuffer<f32>,
}

struct Layer {
    hca: Hc,
    hcf: Hc,
    mix: Mixer,
    ffn: Ffn,
    ple: Option<Ple>,
}

/// Captured intermediate tensors: (name, layer, rep) -> rows x n floats (or i32 bits for index tensors).
#[derive(Default)]
pub struct Dumps {
    pub on: bool,
    pub map: HashMap<(String, i32, u32), Vec<f32>>,
    reps: HashMap<(String, i32), u32>,
}
impl Dumps {
    fn put(&mut self, name: &str, il: i32, v: Vec<f32>) {
        let rep = self.reps.entry((name.to_string(), il)).or_insert(0);
        self.map.insert((name.to_string(), il, *rep), v);
        *rep += 1;
    }
}

struct Work {
    res: DeviceBuffer<f32>,
    xn: DeviceBuffer<f32>,
    lo: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    inj: DeviceBuffer<f32>,
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
    cs: DeviceBuffer<f32>,
    sn: DeviceBuffer<f32>,
    rl: DeviceBuffer<f32>,
    sg: DeviceBuffer<f32>,
    su: DeviceBuffer<f32>,
    sh: DeviceBuffer<f32>,
    glog: DeviceBuffer<f32>,
    moe: DeviceBuffer<f32>,
    emb: DeviceBuffer<f32>,
    pkey: DeviceBuffer<f32>,
    pval: DeviceBuffer<f32>,
    pkn: DeviceBuffer<f32>,
    pqn: DeviceBuffer<f32>,
    pgated: DeviceBuffer<f32>,
    pcn: DeviceBuffer<f32>,
    pco: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    iqv: DeviceBuffer<f32>,
    score: DeviceBuffer<f32>,
    ptmp: DeviceBuffer<f32>,
    pcs: DeviceBuffer<f32>,
    psn: DeviceBuffer<f32>,
    sel: DeviceBuffer<u32>,
    seln: DeviceBuffer<u32>,
}

pub struct Model {
    pub cu: std::sync::Arc<fe_cuda::Context>,
    pub c: Config,
    pub g: Gguf,
    kn: Kern,
    pub s: Stream,
    layers: Vec<Layer>,
    head: Hc,
    output: Lin,
    w: Work,
    pub rows_max: usize,
    pub logit_rows_max: usize,
    pub ctx: usize,
    /// tokens seen so far (positions 0..pos)
    pub tokens: Vec<u32>,
    pub dumps: Dumps,
}

impl Model {
    pub fn load(path: &std::path::Path, ctx: usize, rows_max: usize, logit_rows_max: usize) -> Result<Model> {
        let cu = fe_cuda::Context::new(0)?;
        eprintln!("fnext: {} ({} SMs)", cu.name(), cu.sm_count()?);
        let g = crate::open_model(path)?;
        let c = Config::from_gguf(&g)?;
        let kn = Kern::load()?;
        let s = Stream::new()?;
        let t0 = std::time::Instant::now();
        let mut layers = vec![];
        for il in 0..c.n_layer {
            let p = format!("blk.{il}.");
            let hca = Hc::load(&g, &p, "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", Some("hc_attn_inject.weight"))?;
            let hcf = Hc::load(&g, &p, "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", Some("hc_ffn_inject.weight"))?;
            let mix = if c.is_attn(il) {
                let kvd = c.n_head_kv * c.head_dim;
                Mixer::Attn(Attn {
                    wq: Lin::from_tensor(&g, &format!("{p}attn_q.weight"))?,
                    wk: Lin::from_tensor(&g, &format!("{p}attn_k.weight"))?,
                    wv: Lin::from_tensor(&g, &format!("{p}attn_v.weight"))?,
                    wo: Lin::from_tensor(&g, &format!("{p}attn_output.weight"))?,
                    qn: vecf(&g, &format!("{p}attn_q_norm.weight"))?,
                    kn: vecf(&g, &format!("{p}attn_k_norm.weight"))?,
                    kc: DeviceBuffer::zeroed(ctx * kvd)?,
                    vc: DeviceBuffer::zeroed(ctx * kvd)?,
                    iq: Lin::from_tensor(&g, &format!("{p}indexer.q_proj.weight"))?,
                    ik: Lin::from_tensor(&g, &format!("{p}indexer.k_proj.weight"))?,
                    iqn: vecf(&g, &format!("{p}indexer.q_norm.weight"))?,
                    ikn: vecf(&g, &format!("{p}indexer.k_norm.weight"))?,
                    raw: DeviceBuffer::zeroed(ctx * c.idx_dim)?,
                    pooled: DeviceBuffer::zeroed((ctx / c.kpool + 1) * c.idx_dim)?,
                })
            } else {
                Mixer::Gdn(Gdn {
                    qkv: Lin::from_tensor(&g, &format!("{p}attn_qkv.weight"))?,
                    z: Lin::from_tensor(&g, &format!("{p}attn_gate.weight"))?,
                    ab: Lin::stacked(&g, &format!("{p}ssm_alpha.weight"), &format!("{p}ssm_beta.weight"))?,
                    conv_w: vecf(&g, &format!("{p}ssm_conv1d.weight"))?,
                    dt: vecf(&g, &format!("{p}ssm_dt.bias"))?,
                    a: vecf(&g, &format!("{p}ssm_a"))?,
                    norm: vecf(&g, &format!("{p}ssm_norm.weight"))?,
                    out: Lin::from_tensor(&g, &format!("{p}ssm_out.weight"))?,
                    conv_state: DeviceBuffer::zeroed((c.ssm_conv - 1) * c.conv_dim())?,
                    ssm: DeviceBuffer::zeroed(c.ssm_vh * c.ssm_state * c.ssm_state)?,
                })
            };
            let ffn = Ffn {
                gate_inp: Lin::from_tensor(&g, &format!("{p}ffn_gate_inp.weight"))?,
                sh_gate: Lin::from_tensor(&g, &format!("{p}ffn_gate_shexp.weight"))?,
                sh_up: Lin::from_tensor(&g, &format!("{p}ffn_up_shexp.weight"))?,
                sh_down: Lin::from_tensor(&g, &format!("{p}ffn_down_shexp.weight"))?,
                sh_ginp: Lin::from_tensor(&g, &format!("{p}ffn_gate_inp_shexp.weight"))?,
            };
            let ple = if il == c.ple_layer {
                Some(Ple {
                    key: Lin::from_tensor(&g, &format!("{p}ple_key.weight"))?,
                    value: Lin::from_tensor(&g, &format!("{p}ple_value.weight"))?,
                    nk: vecf(&g, &format!("{p}ple_norm_key.weight"))?,
                    nq: vecf(&g, &format!("{p}ple_norm_query.weight"))?,
                    nc: vecf(&g, &format!("{p}ple_norm_conv.weight"))?,
                    conv_w: vecf(&g, &format!("{p}ple_conv1d.weight"))?,
                    hist: DeviceBuffer::zeroed((c.ple_conv - 1) * c.ple_ngram * c.hc_dim())?,
                })
            } else {
                None
            };
            layers.push(Layer { hca, hcf, mix, ffn, ple });
            if il % 8 == 7 {
                eprintln!("fnext: loaded layers 0..={il} ({:.1} s)", t0.elapsed().as_secs_f32());
            }
        }
        let head = Hc::load(&g, "", "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight", None)?;
        let output = Lin::from_tensor(&g, "output.weight")?;
        eprintln!("fnext: weights on the card in {:.1} s", t0.elapsed().as_secs_f32());
        let r = rows_max;
        let a = |n: usize| DeviceBuffer::<f32>::zeroed(n);
        let hd = c.hc_dim();
        let w = Work {
            res: a(r * hd)?,
            xn: a(r * hd)?,
            lo: a(r * c.hc_lr)?,
            gate: a(r * hd)?,
            inj: a(r * c.hc)?,
            mixed: a(r * c.n_embd)?,
            out: a(r * c.n_embd)?,
            qkv: a(r * c.conv_dim())?,
            z: a(r * c.ssm_inner)?,
            ab: a(r * 2 * c.ssm_vh)?,
            conv: a(r * c.conv_dim())?,
            o: a(r * c.ssm_inner)?,
            y: a(r * c.ssm_inner.max(c.n_head * c.head_dim))?,
            qf: a(r * 2 * c.n_head * c.head_dim)?,
            kx: a(r * c.n_head_kv * c.head_dim)?,
            vx: a(r * c.n_head_kv * c.head_dim)?,
            q: a(r * c.n_head * c.head_dim)?,
            att: a(r * c.n_head * c.head_dim)?,
            cs: a(r * c.n_rot / 2)?,
            sn: a(r * c.n_rot / 2)?,
            rl: a(r * c.n_expert)?,
            sg: a(r * c.n_ff_shexp)?,
            su: a(r * c.n_ff_shexp)?,
            sh: a(r * c.n_embd)?,
            glog: a(r)?,
            moe: a(r * c.n_embd)?,
            emb: a(r * c.n_embd)?,
            pkey: a(r * hd)?,
            pval: a(r * c.n_embd)?,
            pkn: a(r * hd)?,
            pqn: a(r * hd)?,
            pgated: a(r * hd)?,
            pcn: a(r * hd)?,
            pco: a(r * hd)?,
            logits: a(logit_rows_max * c.n_vocab)?,
            iqv: a(r * c.idx_heads * c.idx_dim)?,
            score: a(r * (ctx / c.kpool + 1))?,
            ptmp: a((r / c.kpool + 2) * c.idx_dim)?,
            pcs: a((r / c.kpool + 2) * c.n_rot / 2)?,
            psn: a((r / c.kpool + 2) * c.n_rot / 2)?,
            sel: DeviceBuffer::zeroed(r * (c.idx_top_k + c.kpool - 1))?,
            seln: DeviceBuffer::zeroed(r)?,
        };
        Ok(Model { cu, c, g, kn, s, layers, head, output, w, rows_max, logit_rows_max, ctx, tokens: vec![], dumps: Dumps::default() })
    }

    /// Forget the sequence (states, caches stay allocated; KV cells are simply overwritten).
    pub fn reset(&mut self) -> Result<()> {
        self.s.sync()?;
        for l in &mut self.layers {
            if let Mixer::Gdn(gd) = &mut l.mix {
                self.s.memset_d8(gd.conv_state.ptr(), 0, gd.conv_state.bytes())?;
                self.s.memset_d8(gd.ssm.ptr(), 0, gd.ssm.bytes())?;
            }
            if let Some(p) = &l.ple {
                self.s.memset_d8(p.hist.ptr(), 0, p.hist.bytes())?;
            }
        }
        self.s.sync()?;
        self.tokens.clear();
        self.dumps.reps.clear();
        self.dumps.map.clear();
        Ok(())
    }

    fn dump(&mut self, name: &str, il: i32, p: DevPtr, n: usize) -> Result<()> {
        if !self.dumps.on {
            return Ok(());
        }
        let mut v = vec![0f32; n];
        self.s.sync()?;
        fe_cuda::memcpy_dtoh(&mut v, p)?;
        self.dumps.put(name, il, v);
        Ok(())
    }

    /// Host row of the token embedding (IQ4_XS etc. dequantized from the mmap).
    fn embed_rows(&self, toks: &[u32]) -> Vec<f32> {
        let t = self.g.tensor("token_embd.weight").unwrap();
        let rb = t.row_bytes();
        let d = self.c.n_embd;
        let data = self.g.data(t);
        let mut out = vec![0f32; toks.len() * d];
        out.par_chunks_mut(d).zip(toks.par_iter()).for_each(|(o, &tk)| {
            refq::dequant(t.ty, &data[tk as usize * rb..(tk as usize + 1) * rb], o);
        });
        out
    }

    /// PLE table rows for the tokens at positions pos0.. (see [`ple_rows`]).
    pub fn ple_rows(&self, pos0: usize, toks: &[u32]) -> Vec<u32> {
        ple_rows(&self.c, &self.tokens, pos0, toks)
    }

    fn ple_embed(&self, rows: &[u32]) -> Vec<f32> {
        let t = self.g.tensor("per_layer_token_embd.weight").unwrap();
        let rb = t.row_bytes();
        let dd = t.ne0();
        let data = self.g.data(t);
        let mut out = vec![0f32; rows.len() * dd];
        out.par_chunks_mut(dd).zip(rows.par_iter()).for_each(|(o, &r)| {
            refq::dequant(t.ty, &data[r as usize * rb..(r as usize + 1) * rb], o);
        });
        out
    }

    unsafe fn hc_mix(&mut self, li: Option<usize>, ffn: bool, rows: usize, dump_il: i32) -> Result<()> {
        let c = &self.c;
        let hc = match li {
            Some(l) => if ffn { &self.layers[l].hcf } else { &self.layers[l].hca },
            None => &self.head,
        };
        let (kn, s, w) = (&self.kn, &self.s, &self.w);
        kn.rmsnorm(s, w.xn.ptr(), w.res.ptr(), hc.norm.ptr(), c.n_embd, c.hc, rows * c.hc, c.eps)?;
        hc.down.run(kn, s, w.xn.ptr(), w.lo.ptr(), rows, 1.0 / c.hc as f32, false)?;
        kn.silu(s, w.lo.ptr(), rows * c.hc_lr)?;
        hc.up.run(kn, s, w.lo.ptr(), w.gate.ptr(), rows, 1.0, false)?;
        let n = rows * c.n_embd;
        s.launch(&kn.hc_mix, g256(n), 256, 0, (w.mixed.ptr(), w.xn.ptr(), w.gate.ptr(), c.n_embd as u32, c.hc as u32, n as u32))?;
        let has_inj = hc.inject.is_some();
        if let Some(inj) = &hc.inject {
            inj.run(kn, s, w.xn.ptr(), w.inj.ptr(), rows, 1.0 / c.hc as f32, false)?;
        }
        let (hd, ne, nh) = (c.hc_dim(), c.n_embd, c.hc);
        let (pxn, pg, pm, pi) = (w.xn.ptr(), w.gate.ptr(), w.mixed.ptr(), w.inj.ptr());
        self.dump("hc_norm", dump_il, pxn, rows * hd)?;
        self.dump("hc_gate", dump_il, pg, rows * hd)?;
        self.dump("hc_mixed", dump_il, pm, rows * ne)?;
        if has_inj {
            self.dump("hc_inject", dump_il, pi, rows * nh)?;
        }
        Ok(())
    }

    unsafe fn hc_combine(&mut self, rows: usize, name: &str, il: i32) -> Result<()> {
        let c = &self.c;
        let n = rows * c.hc_dim();
        self.s.launch(&self.kn.hc_combine, g256(n), 256, 0,
            (self.w.res.ptr(), self.w.out.ptr(), self.w.inj.ptr(), c.n_embd as u32, c.hc as u32, c.hc as u32, n as u32))?;
        let p = self.w.res.ptr();
        self.dump(name, il, p, n)
    }

    unsafe fn ple(&mut self, l: usize, toks: &[u32], pos0: usize) -> Result<()> {
        let rows = toks.len();
        let c = self.c.clone();
        let idx = self.ple_rows(pos0, toks);
        let emb = self.ple_embed(&idx);
        if self.dumps.on {
            self.dumps.put("ple_rows", -1, idx.iter().map(|&v| f32::from_bits(v)).collect());
            self.dumps.put("ple_embd", -1, emb.clone());
        }
        let (kn, s, w) = (&self.kn, &self.s, &self.w);
        let p = self.layers[l].ple.as_ref().unwrap();
        s.sync()?;
        fe_cuda::memcpy_htod(w.emb.ptr(), &emb)?;
        let hd = c.hc_dim();
        p.key.run(kn, s, w.emb.ptr(), w.pkey.ptr(), rows, 1.0, false)?;
        p.value.run(kn, s, w.emb.ptr(), w.pval.ptr(), rows, 1.0, false)?;
        kn.rmsnorm(s, w.pkn.ptr(), w.pkey.ptr(), p.nk.ptr(), c.n_embd, c.hc, rows * c.hc, c.eps)?;
        kn.rmsnorm(s, w.pqn.ptr(), w.res.ptr(), p.nq.ptr(), c.n_embd, c.hc, rows * c.hc, c.eps)?;
        s.launch(&kn.ple_gate, (rows * c.hc) as u32, 256, 0,
            (w.pgated.ptr(), w.pkn.ptr(), w.pqn.ptr(), w.pval.ptr(), c.n_embd as u32, c.hc as u32))?;
        kn.rmsnorm(s, w.pcn.ptr(), w.pgated.ptr(), p.nc.ptr(), c.n_embd, c.hc, rows * c.hc, c.eps)?;
        s.launch(&kn.ple_conv, g256(hd), 256, 0, (w.res.ptr(), w.pgated.ptr(), w.pcn.ptr(), p.conv_w.ptr(), p.hist.ptr(),
            rows as u32, hd as u32, c.ple_ngram as u32, w.pco.ptr(), 0u32))?;
        if self.dumps.on {
            // ple_gate per stream: recompute from the gated value (gate = gated / value) is lossy; dump gated + conv
            let (pg, pc) = (self.w.pgated.ptr(), self.w.pco.ptr());
            self.dump("ple_gated_value", l as i32, pg, rows * hd)?;
            self.dump("ple_conv_out", l as i32, pc, rows * hd)?;
        }
        Ok(())
    }

    unsafe fn gdn(&mut self, l: usize, rows: usize) -> Result<()> {
        let c = self.c.clone();
        let (kn, s, w) = (&self.kn, &self.s, &self.w);
        let Mixer::Gdn(gd) = &self.layers[l].mix else { unreachable!() };
        let cd = c.conv_dim();
        gd.qkv.run(kn, s, w.mixed.ptr(), w.qkv.ptr(), rows, 1.0, false)?;
        gd.z.run(kn, s, w.mixed.ptr(), w.z.ptr(), rows, 1.0, false)?;
        gd.ab.run(kn, s, w.mixed.ptr(), w.ab.ptr(), rows, 1.0, false)?;
        s.launch(&kn.gdn_conv, g256(cd), 256, 0, (w.conv.ptr(), w.qkv.ptr(), cd as u32, gd.conv_w.ptr(), gd.conv_state.ptr(),
            rows as u32, cd as u32, 0u32))?;
        s.launch(&kn.gdn_rec, c.ssm_vh as u32, 512, 0, (w.o.ptr(), w.conv.ptr(), w.ab.ptr(), (2 * c.ssm_vh) as u32, gd.dt.ptr(),
            gd.a.ptr(), gd.ssm.ptr(), rows as u32, c.ssm_kh as u32, c.ssm_vh as u32, c.eps, 0u32))?;
        s.launch(&kn.gdn_gnorm, (rows as u32, c.ssm_vh.div_ceil(8) as u32), 256, 0, (w.y.ptr(), w.o.ptr(), w.z.ptr(),
            c.ssm_inner as u32, gd.norm.ptr(), c.ssm_vh as u32, c.eps))?;
        gd.out.run(kn, s, w.y.ptr(), w.out.ptr(), rows, 1.0, false)?;
        if self.dumps.on {
            let il = l as i32;
            let (q, z, cv, o, y, out) = (w.qkv.ptr(), w.z.ptr(), w.conv.ptr(), w.o.ptr(), w.y.ptr(), w.out.ptr());
            self.dump("linear_attn_qkv_mixed", il, q, rows * cd)?;
            self.dump("z", il, z, rows * c.ssm_inner)?;
            self.dump("conv_output_silu", il, cv, rows * cd)?;
            self.dump("attn_output", il, o, rows * c.ssm_inner)?;
            self.dump("final_output", il, y, rows * c.ssm_inner)?;
            self.dump("linear_attn_out", il, out, rows * c.n_embd)?;
        }
        Ok(())
    }

    fn rope_tables(&self, pos0: usize, rows: usize) -> (Vec<f32>, Vec<f32>) {
        self.rope_tables_at(&(0..rows).map(|r| pos0 + r).collect::<Vec<_>>())
    }

    /// cos / sin of the rotary pairs at each position (theta stepped in f32 like ggml's rope cache).
    fn rope_tables_at(&self, pos: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let half = self.c.n_rot / 2;
        let ts = self.c.rope_base.powf(-2.0 / self.c.n_rot as f32);
        let rows = pos.len();
        let (mut cs, mut sn) = (vec![0f32; rows * half], vec![0f32; rows * half]);
        for r in 0..rows {
            let mut th = pos[r] as f32;
            for k in 0..half {
                cs[r * half + k] = th.cos();
                sn[r * half + k] = th.sin();
                th *= ts;
            }
        }
        (cs, sn)
    }

    unsafe fn attn(&mut self, l: usize, rows: usize, pos0: usize) -> Result<()> {
        let c = self.c.clone();
        if pos0 + rows > self.ctx {
            bail!("context full ({} cells)", self.ctx);
        }
        let (cs, sn) = self.rope_tables(pos0, rows);
        let (kn, s, w) = (&self.kn, &self.s, &self.w);
        let Mixer::Attn(at) = &self.layers[l].mix else { unreachable!() };
        let qd = c.n_head * c.head_dim;
        let kvd = c.n_head_kv * c.head_dim;
        at.wq.run(kn, s, w.mixed.ptr(), w.qf.ptr(), rows, 1.0, false)?;
        at.wk.run(kn, s, w.mixed.ptr(), w.kx.ptr(), rows, 1.0, false)?;
        at.wv.run(kn, s, w.mixed.ptr(), w.vx.ptr(), rows, 1.0, false)?;
        s.sync()?;
        fe_cuda::memcpy_htod(w.cs.ptr(), &cs)?;
        fe_cuda::memcpy_htod(w.sn.ptr(), &sn)?;
        s.launch(&kn.attn_prep, (rows as u32, (c.n_head + c.n_head_kv) as u32), c.head_dim as u32, 0,
            (w.q.ptr(), at.kc.ptr(), at.vc.ptr(), w.qf.ptr(), (2 * qd) as u32, (2 * c.head_dim) as u32, w.kx.ptr(), w.vx.ptr(),
                kvd as u32, at.qn.ptr(), at.kn.ptr(), w.cs.ptr(), w.sn.ptr(), c.n_head as u32, c.n_head_kv as u32,
                c.n_rot as u32, pos0 as u32, c.eps))?;
        let sel = self.qsa_select(l, rows, pos0)?;
        let (kn, s, w) = (&self.kn, &self.s, &self.w);
        let Mixer::Attn(at) = &self.layers[l].mix else { unreachable!() };
        let (selp, selnp) = if sel { (w.sel.ptr(), w.seln.ptr()) } else { (DevPtr::NULL, DevPtr::NULL) };
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        s.launch(&kn.attn, (rows as u32, c.n_head as u32), 256, 0, (w.att.ptr(), w.q.ptr(), at.kc.ptr(), at.vc.ptr(),
            w.qf.ptr().add::<f32>(c.head_dim), (2 * qd) as u32, (2 * c.head_dim) as u32, c.n_head as u32,
            c.n_head_kv as u32, pos0 as u32, scale, selp, selnp, (c.idx_top_k + c.kpool - 1) as u32))?;
        at.wo.run(kn, s, w.att.ptr(), w.out.ptr(), rows, 1.0, false)?;
        let _ = sel;
        if self.dumps.on {
            let il = l as i32;
            let (qf, kx, vx, q, att, out) = (w.qf.ptr(), w.kx.ptr(), w.vx.ptr(), w.q.ptr(), w.att.ptr(), w.out.ptr());
            self.dump("Qcur_full", il, qf, rows * 2 * qd)?;
            self.dump("Kcur", il, kx, rows * kvd)?;
            self.dump("Vcur", il, vx, rows * kvd)?;
            self.dump("Qcur", il, q, rows * qd)?;
            self.dump("attn_gated", il, att, rows * qd)?;
            self.dump("attn_output", il, out, rows * c.n_embd)?;
        }
        Ok(())
    }

    /// QSA (layer l, rows at pos0..): store the raw indexer keys, pool the blocks this chunk completes, score the
    /// visible blocks and, when a row sees more than top_k / kpool of them, select the top blocks + the open tail
    /// into w.sel / w.seln. Returns whether a selection was made (else the attention is dense, which is the same).
    unsafe fn qsa_select(&mut self, l: usize, rows: usize, pos0: usize) -> Result<bool> {
        let c = self.c.clone();
        let (kp, dd) = (c.kpool, c.idx_dim);
        let b_lo = pos0 / kp;
        let b_hi = (pos0 + rows) / kp;
        let bpos: Vec<usize> = (b_lo..b_hi).map(|b| b * kp).collect();
        let (pcs, psn) = self.rope_tables_at(&bpos);
        let (qcs, qsn) = self.rope_tables(pos0, rows);
        let (kn, s, w) = (&self.kn, &self.s, &self.w);
        let Mixer::Attn(at) = &self.layers[l].mix else { unreachable!() };
        at.ik.run(kn, s, w.mixed.ptr(), at.raw.ptr().add::<f32>(pos0 * dd), rows, 1.0, false)?;
        let nb = b_hi - b_lo;
        if nb > 0 {
            s.sync()?;
            fe_cuda::memcpy_htod(w.pcs.ptr(), &pcs)?;
            fe_cuda::memcpy_htod(w.psn.ptr(), &psn)?;
            s.launch(&kn.idx_pool, nb as u32, dd as u32, 0, (w.ptmp.ptr(), at.raw.ptr(), b_lo as u32, kp as u32))?;
            let dst = at.pooled.ptr().add::<f32>(b_lo * dd);
            kn.rmsnorm(s, dst, w.ptmp.ptr(), at.ikn.ptr(), dd, 1, nb, c.eps)?;
            s.launch(&kn.rope_rows, (nb as u32, 1), dd as u32, 0, (dst, w.pcs.ptr(), w.psn.ptr(), 1u32, c.n_rot as u32))?;
        }
        let nv_max = (pos0 + rows) / kp;
        let top = c.idx_top_k / kp;
        if self.dumps.on {
            let (rp, n) = (at.raw.ptr().add::<f32>(pos0 * dd), rows * dd);
            self.dump("indexer_k_raw", l as i32, rp, n)?;
        }
        if nv_max <= top {
            return Ok(false);
        }
        let (kn, s, w) = (&self.kn, &self.s, &self.w);
        let Mixer::Attn(at) = &self.layers[l].mix else { unreachable!() };
        at.iq.run(kn, s, w.mixed.ptr(), w.iqv.ptr(), rows, 1.0, false)?;
        kn.rmsnorm(s, w.iqv.ptr(), w.iqv.ptr(), at.iqn.ptr(), dd, 1, rows * c.idx_heads, c.eps)?;
        s.sync()?;
        fe_cuda::memcpy_htod(w.cs.ptr(), &qcs)?;
        fe_cuda::memcpy_htod(w.sn.ptr(), &qsn)?;
        s.launch(&kn.rope_rows, (rows as u32, c.idx_heads as u32), dd as u32, 0,
            (w.iqv.ptr(), w.cs.ptr(), w.sn.ptr(), c.idx_heads as u32, c.n_rot as u32))?;
        s.launch(&kn.idx_score, (rows as u32, nv_max.div_ceil(8) as u32), 256, 0, (w.score.ptr(), w.iqv.ptr(),
            at.pooled.ptr(), c.idx_heads as u32, pos0 as u32, kp as u32, nv_max as u32, 1.0 / (dd as f32).sqrt()))?;
        s.sync()?;
        let mut sc = vec![0f32; rows * nv_max];
        fe_cuda::memcpy_dtoh(&mut sc, w.score.ptr())?;
        let ldsel = c.idx_top_k + kp - 1;
        let mut sel = vec![0u32; rows * ldsel];
        let mut seln = vec![0u32; rows];
        let mut tops = vec![0f32; rows * top];
        for r in 0..rows {
            let p = pos0 + r;
            let nv = (p + 1) / kp;
            let srow = &mut sel[r * ldsel..(r + 1) * ldsel];
            let mut n = 0;
            if nv <= top {
                for cell in 0..=p {
                    srow[n] = cell as u32;
                    n += 1;
                }
            } else {
                let row = &sc[r * nv_max..r * nv_max + nv];
                let mut ord: Vec<u32> = (0..nv as u32).collect();
                ord.select_nth_unstable_by(top - 1, |&a, &b| row[b as usize].partial_cmp(&row[a as usize]).unwrap().then(a.cmp(&b)));
                let mut chosen = ord[..top].to_vec();
                chosen.sort();
                for (j, &b) in chosen.iter().enumerate() {
                    tops[r * top + j] = f32::from_bits(b);
                    for k in 0..kp {
                        srow[n] = (b as usize * kp + k) as u32;
                        n += 1;
                    }
                }
                for k in 0..(p + 1) % kp {
                    srow[n] = (p - k) as u32;
                    n += 1;
                }
            }
            seln[r] = n as u32;
        }
        fe_cuda::memcpy_htod(self.w.sel.ptr(), &sel)?;
        fe_cuda::memcpy_htod(self.w.seln.ptr(), &seln)?;
        if self.dumps.on {
            self.dumps.put("indexer_top_k", l as i32, tops);
        }
        Ok(true)
    }

    /// Router + shared expert on the card, routed experts on the CPU; result in w.out.
    unsafe fn ffn(&mut self, l: usize, rows: usize) -> Result<()> {
        let c = self.c.clone();
        let il = l as i32;
        {
            let (kn, s, w) = (&self.kn, &self.s, &self.w);
            let f = &self.layers[l].ffn;
            f.gate_inp.run(kn, s, w.mixed.ptr(), w.rl.ptr(), rows, 1.0, false)?;
            f.sh_gate.run(kn, s, w.mixed.ptr(), w.sg.ptr(), rows, 1.0, false)?;
            f.sh_up.run(kn, s, w.mixed.ptr(), w.su.ptr(), rows, 1.0, false)?;
            let n = rows * c.n_ff_shexp;
            s.launch(&kn.swiglu, g256(n), 256, 0, (w.sg.ptr(), w.sg.ptr(), w.su.ptr(), n as u32))?;
            f.sh_down.run(kn, s, w.sg.ptr(), w.sh.ptr(), rows, 1.0, false)?;
            f.sh_ginp.run(kn, s, w.mixed.ptr(), w.glog.ptr(), rows, 1.0, false)?;
        }
        self.s.sync()?;
        let mut x = vec![0f32; rows * c.n_embd];
        let mut logits = vec![0f32; rows * c.n_expert];
        fe_cuda::memcpy_dtoh(&mut x, self.w.mixed.ptr())?;
        fe_cuda::memcpy_dtoh(&mut logits, self.w.rl.ptr())?;
        let (ids, wts, probs) = route(&c, &logits, rows);
        let y = self.experts_cpu(l, &x, &ids, &wts, rows)?;
        fe_cuda::memcpy_htod(self.w.moe.ptr(), &y)?;
        let n = rows * c.n_embd;
        self.s.launch(&self.kn.moe_out, g256(n), 256, 0,
            (self.w.out.ptr(), self.w.moe.ptr(), self.w.sh.ptr(), self.w.glog.ptr(), c.n_embd as u32, n as u32))?;
        if self.dumps.on {
            self.dumps.put("ffn_moe_logits", il, logits);
            self.dumps.put("ffn_moe_probs", il, probs);
            self.dumps.put("ffn_moe_topk", il, ids.iter().map(|&v| f32::from_bits(v as u32)).collect());
            self.dumps.put("ffn_moe_weights_norm", il, wts.clone());
            self.dumps.put("ffn_moe_out", il, y);
            let (sh, gl, out) = (self.w.sh.ptr(), self.w.glog.ptr(), self.w.out.ptr());
            self.dump("ffn_shexp", il, sh, n)?;
            self.dump("shared_expert_gate", il, gl, rows)?;
            self.dump("ffn_out", il, out, n)?;
        }
        Ok(())
    }

    /// Routed experts of layer l on the CPU: each used expert is dequantized once and applied to its rows.
    fn experts_cpu(&self, l: usize, x: &[f32], ids: &[usize], wts: &[f32], rows: usize) -> Result<Vec<f32>> {
        let c = &self.c;
        let k = c.n_expert_used;
        let (d, ff) = (c.n_embd, c.n_ff_exp);
        let tg = tensor(&self.g, &format!("blk.{l}.ffn_gate_exps.weight"))?;
        let tu = tensor(&self.g, &format!("blk.{l}.ffn_up_exps.weight"))?;
        let td = tensor(&self.g, &format!("blk.{l}.ffn_down_exps.weight"))?;
        let mut by_exp: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for r in 0..rows {
            for j in 0..k {
                by_exp.entry(ids[r * k + j]).or_default().push((r, j));
            }
        }
        let jobs: Vec<(usize, Vec<(usize, usize)>)> = by_exp.into_iter().collect();
        let results: Vec<Vec<(usize, usize, Vec<f32>)>> = jobs.par_iter().map(|(e, uses)| {
            let mut wg = vec![0f32; ff * d];
            let mut wu = vec![0f32; ff * d];
            let mut wd = vec![0f32; d * ff];
            refq::dequant(tg.ty, self.g.slice(tg, *e), &mut wg);
            refq::dequant(tu.ty, self.g.slice(tu, *e), &mut wu);
            refq::dequant(td.ty, self.g.slice(td, *e), &mut wd);
            let mut out = vec![];
            let emu = self.kn.emu >= 1;
            for &(r, j) in uses {
                let xq;
                let xr = if emu { xq = q8_round(&x[r * d..(r + 1) * d]); &xq[..] } else { &x[r * d..(r + 1) * d] };
                let mut act = vec![0f32; ff];
                for i in 0..ff {
                    let gv = dot(&wg[i * d..(i + 1) * d], xr);
                    let uv = dot(&wu[i * d..(i + 1) * d], xr);
                    act[i] = gv / (1.0 + (-gv).exp()) * uv;
                }
                let mut yv = vec![0f32; d];
                if emu {
                    act = q8_round(&act);
                }
                for i in 0..d {
                    yv[i] = dot(&wd[i * ff..(i + 1) * ff], &act);
                }
                out.push((r, j, yv));
            }
            out
        }).collect();
        let mut per: Vec<Option<Vec<f32>>> = vec![None; rows * k];
        for v in results {
            for (r, j, yv) in v {
                per[r * k + j] = Some(yv);
            }
        }
        let mut y = vec![0f32; rows * d];
        for r in 0..rows {
            let yr = &mut y[r * d..(r + 1) * d];
            for j in 0..k {
                let wv = wts[r * k + j];
                let e = per[r * k + j].as_ref().unwrap();
                for i in 0..d {
                    yr[i] += wv * e[i];
                }
            }
        }
        Ok(y)
    }

    /// Run `toks` (the next positions of the sequence) through the model; returns the logits of the last
    /// `n_logits` rows (row-major [n_logits][vocab]).
    pub fn forward(&mut self, toks: &[u32], n_logits: usize) -> Result<Vec<f32>> {
        let rows = toks.len();
        if rows == 0 || rows > self.rows_max || n_logits > rows || n_logits > self.logit_rows_max {
            bail!("forward: {rows} rows ({n_logits} logits) outside the workspace");
        }
        let c = self.c.clone();
        let pos0 = self.tokens.len();
        let emb = self.embed_rows(toks);
        unsafe {
            self.s.sync()?;
            fe_cuda::memcpy_htod(self.w.emb.ptr(), &emb)?;
            if self.dumps.on {
                self.dumps.put("model.input_embed", -1, emb.clone());
            }
            let n = rows * c.n_embd;
            self.s.launch(&self.kn.hc_init, g256(n), 256, 0, (self.w.res.ptr(), self.w.emb.ptr(), c.n_embd as u32, c.hc as u32, n as u32))?;
            for l in 0..c.n_layer {
                let il = l as i32;
                if self.layers[l].ple.is_some() {
                    self.ple(l, toks, pos0)?;
                }
                self.hc_mix(Some(l), false, rows, il)?;
                if c.is_attn(l) {
                    self.attn(l, rows, pos0)?;
                } else {
                    self.gdn(l, rows)?;
                }
                self.hc_combine(rows, "hc_combine", il)?;
                self.hc_mix(Some(l), true, rows, il)?;
                self.ffn(l, rows)?;
                self.hc_combine(rows, "l_last", il)?;
            }
            let p = self.w.res.ptr();
            self.dump("h_nextn", -1, p, rows * c.hc_dim())?;
            self.hc_mix(None, false, rows, -1)?;
            let r0 = rows - n_logits;
            let pm = self.w.mixed.ptr();
            self.dump("result_norm", -1, pm, rows * c.n_embd)?;
            self.output.run(&self.kn, &self.s, self.w.mixed.ptr().add::<f32>(r0 * c.n_embd), self.w.logits.ptr(), n_logits, 1.0, false)?;
            self.s.sync()?;
            let mut lg = vec![0f32; n_logits * c.n_vocab];
            fe_cuda::memcpy_dtoh(&mut lg, self.w.logits.ptr())?;
            if self.dumps.on {
                self.dumps.put("result_output", -1, lg.clone());
            }
            self.tokens.extend_from_slice(toks);
            Ok(lg)
        }
    }
}

/// q8_1 round trip per 32 values (d = amax / 127 stored as f16, q = roundf(x / d)).
pub(crate) fn q8_round(x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for (xb, yb) in x.chunks(32).zip(y.chunks_mut(32)) {
        let a = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = a / 127.0;
        let dh = refq::f16_to_f32(refq::f32_to_f16(d));
        for (o, v) in yb.iter_mut().zip(xb) {
            *o = if a == 0.0 { 0.0 } else { (v / d).round() * dh };
        }
    }
    y
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n8 = a.len() / 8 * 8;
    let mut i = 0;
    while i < n8 {
        for j in 0..8 {
            acc[j] += a[i + j] * b[i + j];
        }
        i += 8;
    }
    let mut s = acc.iter().sum::<f32>();
    while i < a.len() {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// Softmax router, top-k by probability, weights renormalised (llm_graph_context::build_moe_ffn, gating SOFTMAX,
/// norm_w): returns (expert ids [rows][k], weights [rows][k], probs [rows][n_expert]).
pub fn route(c: &Config, logits: &[f32], rows: usize) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let (ne, k) = (c.n_expert, c.n_expert_used);
    let mut ids = vec![0usize; rows * k];
    let mut wts = vec![0f32; rows * k];
    let mut probs = vec![0f32; rows * ne];
    for r in 0..rows {
        let lr = &logits[r * ne..(r + 1) * ne];
        let m = lr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let pr = &mut probs[r * ne..(r + 1) * ne];
        let mut sum = 0f32;
        for (p, &v) in pr.iter_mut().zip(lr) {
            *p = (v - m).exp();
            sum += *p;
        }
        for p in pr.iter_mut() {
            *p /= sum;
        }
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_by(|&a, &b| pr[b].partial_cmp(&pr[a]).unwrap().then(a.cmp(&b)));
        let mut ws = 0f32;
        for j in 0..k {
            ids[r * k + j] = order[j];
            wts[r * k + j] = pr[order[j]];
            ws += pr[order[j]];
        }
        let ws = ws.max(6.103515625e-5);
        for j in 0..k {
            wts[r * k + j] /= ws;
            if c.expert_weights_scale != 0.0 && c.expert_weights_scale != 1.0 {
                wts[r * k + j] *= c.expert_weights_scale;
            }
        }
    }
    (ids, wts, probs)
}

/// PLE table rows for the tokens at positions pos0.. after the history `hist` (the hash of
/// llm_graph_input_qwen4exp_ple::set_input).
pub fn ple_rows(c: &Config, hist: &[u32], pos0: usize, toks: &[u32]) -> Vec<u32> {
    let nh = c.ple_heads();
    let mut idx = vec![0u32; toks.len() * nh];
    let all = |p: usize| -> u32 { if p < pos0 { hist[p] } else { toks[p - pos0] } };
    for i in 0..toks.len() {
        let p = pos0 + i;
        let mut ctxt = vec![0u64; c.ple_ngram];
        ctxt[0] = all(p) as u64;
        let mut cut = false;
        for s in 1..c.ple_ngram {
            let t: i64 = if cut || p < s { -1 } else { all(p - s) as i64 };
            cut = cut || t < 0 || t == c.ple_eos as i64;
            ctxt[s] = if cut { c.ple_eos as u64 } else { t as u64 };
        }
        for n in 2..=c.ple_ngram {
            let mut mixed = ctxt[0].wrapping_mul(c.ple_mult[0]);
            for j in 1..n {
                mixed ^= ctxt[j].wrapping_mul(c.ple_mult[j]);
            }
            let base = (n - 2) * c.ple_heads_per_ngram;
            for gg in 0..c.ple_heads_per_ngram {
                let h = base + gg;
                idx[i * nh + h] = (mixed % c.ple_vocab[h] + c.ple_off[h]) as u32;
            }
        }
    }
    idx
}

