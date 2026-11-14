//! FNEXT_TP=2: tensor parallel over two cards without peer access. Both cards run every layer: each holds half of
//! the attention heads (and that half of the KV cache), half of the GDN heads (and their conv / recurrent state),
//! half of the shared expert, and the routed experts it owns (expert parallel: the full expert, so the CPU miss path
//! keeps its bits); the hyper-connection mixers, the router, the indexer and the PLE layer are replicated. Each block
//! ends in a two-card all-reduce of its 2560-wide output through pinned host memory (LL value|tag pairs copied by the
//! copy engine, read by the consuming kernel), so both cards keep bit-identical residual streams. The head and the
//! MTP block run on card 1.
use super::*;

/// The per-card config: half the attention / GDN heads, half the shared expert.
pub(crate) fn half_config(c: &Config) -> Result<Config> {
    if c.n_head % 2 != 0 || c.n_head_kv % 2 != 0 || c.ssm_kh % 2 != 0 || c.ssm_vh % c.ssm_kh != 0 || c.n_ff_shexp % 64 != 0 {
        bail!("FNEXT_TP=2: heads / shared expert do not split in two");
    }
    let mut h = c.clone();
    h.n_head /= 2;
    h.n_head_kv /= 2;
    h.ssm_kh /= 2;
    h.ssm_vh /= 2;
    h.ssm_inner = h.ssm_vh * h.ssm_state;
    h.n_ff_shexp /= 2;
    Ok(h)
}

/// GDN v heads of card r in local order: v head h reads k head h % kh, card r keeps k heads r*kh/2 .. (r+1)*kh/2 and
/// the v heads that read them (local v head j then reads local k head j % (kh/2), as the kernels assume).
pub(crate) fn gdn_vheads(c: &Config, r: usize) -> Vec<usize> {
    let kh = c.ssm_kh;
    (0..c.ssm_vh).filter(|h| (h % kh) / (kh / 2) == r).collect()
}

/// Bytes of the rows in `rows` (ranges of row indices, concatenated in order) of a 2-D tensor.
fn row_bytes_of(g: &Gguf, name: &str, rows: &[std::ops::Range<usize>]) -> Result<(GgmlType, usize, Vec<u8>, usize)> {
    let t = tensor(g, name)?;
    let rb = t.row_bytes();
    let d = g.data(t);
    let mut out = vec![];
    let mut n = 0;
    for r in rows {
        if r.end > t.n_rows() {
            bail!("{name}: rows {r:?} of {}", t.n_rows());
        }
        out.extend_from_slice(&d[r.start * rb..r.end * rb]);
        n += r.len();
    }
    Ok((t.ty, t.ne0(), out, n))
}

/// Every row of a 2-D tensor cut to the input columns in `cols` (ranges, concatenated in order; each range must start
/// and end on a quant block): the same values, a narrower K.
fn col_bytes_of(g: &Gguf, name: &str, cols: &[std::ops::Range<usize>]) -> Result<(GgmlType, usize, Vec<u8>, usize)> {
    let t = tensor(g, name)?;
    let (bs, _) = t.ty.block();
    for c in cols {
        if c.start % bs != 0 || c.end % bs != 0 || c.end > t.ne0() {
            bail!("{name}: {} columns {c:?} not on {bs}-blocks", t.ty.name());
        }
    }
    let rb = t.row_bytes();
    let d = g.data(t);
    let k: usize = cols.iter().map(|c| c.len()).sum();
    let mut out = Vec::with_capacity(t.n_rows() * t.ty.row_bytes(k));
    for r in 0..t.n_rows() {
        let row = &d[r * rb..(r + 1) * rb];
        for c in cols {
            out.extend_from_slice(&row[t.ty.row_bytes(c.start)..t.ty.row_bytes(c.end)]);
        }
    }
    Ok((t.ty, k, out, t.n_rows()))
}

fn qw_from((ty, k, b, n): (GgmlType, usize, Vec<u8>, usize), name: &str) -> Result<QW> {
    if !fe_ggq::gpu::GEMV_TYPES.contains(&ty) {
        bail!("{name}: {} has no GEMV", ty.name());
    }
    Ok(QW { buf: DeviceBuffer::from_slice(&b)?, ty, k, n })
}

/// The rows of a column-cut weight padded with zero quant blocks to K a multiple of `m` (a zero block decodes to 0 in
/// every type with a leading scale: all its bytes 0).
fn pad_k((ty, k, b, n): (GgmlType, usize, Vec<u8>, usize), m: usize, name: &str) -> Result<QW> {
    let kp = k.next_multiple_of(m);
    let (rb, rbp) = (ty.row_bytes(k), ty.row_bytes(kp));
    if !matches!(ty, GgmlType::IQ4NL | GgmlType::Q8_0 | GgmlType::Q4_0) {
        bail!("{name}: {} padding not supported", ty.name());
    }
    let mut out = vec![0u8; n * rbp];
    for r in 0..n {
        out[r * rbp..r * rbp + rb].copy_from_slice(&b[r * rb..(r + 1) * rb]);
    }
    qw_from((ty, kp, out, n), name)
}

fn lin_from((ty, k, b, n): (GgmlType, usize, Vec<u8>, usize)) -> Result<Lin> {
    Lin::from_bytes(ty, &b, k, n)
}

/// f32 vector entries in `ix` order (ranges).
fn vec_of(g: &Gguf, name: &str, ix: &[std::ops::Range<usize>]) -> Result<DeviceBuffer<f32>> {
    let t = tensor(g, name)?;
    let d = g.data(t);
    let mut v = vec![];
    for r in ix {
        for i in r.clone() {
            v.push(f32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()));
        }
    }
    Ok(DeviceBuffer::from_slice(&v)?)
}

/// Ranges of `n`-wide blocks at the given block indices (merged when adjacent).
fn blocks(ix: &[usize], n: usize, base: usize) -> Vec<std::ops::Range<usize>> {
    let mut v: Vec<std::ops::Range<usize>> = vec![];
    for &i in ix {
        let r = base + i * n..base + (i + 1) * n;
        match v.last_mut() {
            Some(l) if l.end == r.start => l.end = r.end,
            _ => v.push(r),
        }
    }
    v
}

/// Layer `il` for card r of a two-card tensor-parallel split (`ch`: that card's half config, `c`: the full one).
pub(crate) fn load_layer(g: &Gguf, c: &Config, ch: &Config, il: usize, r: usize, ctx: usize, ord: i32) -> Result<Layer> {
    let p = format!("blk.{il}.");
    let kvd = ch.n_head_kv * ch.head_dim;
    let hca = Hc::load(g, &p, "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", Some("hc_attn_inject.weight"))?;
    let hcf = Hc::load(g, &p, "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", Some("hc_ffn_inject.weight"))?;
    let mix = if c.is_attn(il) {
        // q heads r*nh/2.. (q | gate rows of each head), kv head r (the q heads of a kv head are contiguous)
        let hq = ch.n_head;
        let qrows = 2 * c.head_dim;
        let qh: Vec<usize> = (r * hq..(r + 1) * hq).collect();
        let kh: Vec<usize> = (r * ch.n_head_kv..(r + 1) * ch.n_head_kv).collect();
        let n = |s: &str| format!("{p}{s}.weight");
        Mixer::Attn(Attn {
            wq: qw_from(row_bytes_of(g, &n("attn_q"), &blocks(&qh, qrows, 0))?, &n("attn_q"))?,
            wk: qw_from(row_bytes_of(g, &n("attn_k"), &blocks(&kh, c.head_dim, 0))?, &n("attn_k"))?,
            wv: qw_from(row_bytes_of(g, &n("attn_v"), &blocks(&kh, c.head_dim, 0))?, &n("attn_v"))?,
            wo: qw_from(col_bytes_of(g, &n("attn_output"), &blocks(&qh, c.head_dim, 0))?, &n("attn_output"))?,
            qn: vecf(g, &n("attn_q_norm"))?,
            kn: vecf(g, &n("attn_k_norm"))?,
            kq: VmBuffer::reserve(ctx * kvd, ord)?,
            ks: VmBuffer::reserve(ctx * kvd / 32 * 4, ord)?,
            vq: VmBuffer::reserve(ctx * kvd, ord)?,
            vs: VmBuffer::reserve(ctx * kvd / 32 * 4, ord)?,
            iq: Lin::from_tensor(g, &n("indexer.q_proj"))?,
            ik: Lin::from_tensor(g, &n("indexer.k_proj"))?,
            iqn: vecf(g, &n("indexer.q_norm"))?,
            ikn: vecf(g, &n("indexer.k_norm"))?,
            ring: DeviceBuffer::zeroed(ring_cells() * c.idx_dim)?,
            pooled: VmBuffer::reserve((ctx / c.kpool + 1) * c.idx_dim * 4, ord)?,
        })
    } else {
        let st = c.ssm_state;
        let half = c.ssm_kh / 2;
        let kh: Vec<usize> = (r * half..(r + 1) * half).collect();
        let vh = gdn_vheads(c, r);
        // qkv channels: [q of the k heads | k of the k heads | v of the v heads]
        let mut ch_ix = blocks(&kh, st, 0);
        ch_ix.extend(blocks(&kh, st, c.ssm_kh * st));
        ch_ix.extend(blocks(&vh, st, 2 * c.ssm_kh * st));
        let n = |s: &str| format!("{p}{s}");
        // alpha | beta rows of the v heads
        let (ta, _, mut ab, na) = row_bytes_of(g, &n("ssm_alpha.weight"), &blocks(&vh, 1, 0))?;
        let (tb, kb, bb, nb) = row_bytes_of(g, &n("ssm_beta.weight"), &blocks(&vh, 1, 0))?;
        if ta != tb {
            bail!("{p}: alpha / beta types differ");
        }
        ab.extend_from_slice(&bb);
        // conv1d weight [channel][taps]
        let ct = tensor(g, &n("ssm_conv1d.weight"))?;
        let taps = ct.ne0();
        let conv_ix: Vec<std::ops::Range<usize>> = ch_ix.iter().map(|r| r.start * taps..r.end * taps).collect();
        Mixer::Gdn(Gdn {
            qkv: qw_from(row_bytes_of(g, &n("attn_qkv.weight"), &ch_ix)?, &n("attn_qkv.weight"))?,
            z: qw_from(row_bytes_of(g, &n("attn_gate.weight"), &blocks(&vh, st, 0))?, &n("attn_gate.weight"))?,
            ab: lin_from((ta, kb, ab, na + nb))?,
            conv_w: vec_of(g, &n("ssm_conv1d.weight"), &conv_ix)?,
            dt: vec_of(g, &n("ssm_dt.bias"), &blocks(&vh, 1, 0))?,
            a: vec_of(g, &n("ssm_a"), &blocks(&vh, 1, 0))?,
            norm: vecf(g, &n("ssm_norm.weight"))?,
            out: qw_from(col_bytes_of(g, &n("ssm_out.weight"), &blocks(&vh, st, 0))?, &n("ssm_out.weight"))?,
            conv_state: DeviceBuffer::zeroed((c.ssm_conv - 1) * ch.conv_dim())?,
            ssm: DeviceBuffer::zeroed(ch.ssm_vh * st * st)?,
            qkv_s: DeviceBuffer::zeroed(MAX_ROWS * ch.conv_dim())?,
            conv_s: DeviceBuffer::zeroed(MAX_ROWS * ch.conv_dim())?,
            ab_s: DeviceBuffer::zeroed(MAX_ROWS * 2 * ch.ssm_vh)?,
        })
    };
    let tg = tensor(g, &format!("{p}ffn_gate_exps.weight"))?;
    let td = tensor(g, &format!("{p}ffn_down_exps.weight"))?;
    let sf = ch.n_ff_shexp;
    let sh_rows = [r * sf..(r + 1) * sf];
    let ffn = Ffn {
        gate_inp: Lin::stacked(g, &format!("{p}ffn_gate_inp.weight"), &format!("{p}ffn_gate_inp_shexp.weight"))?,
        shg: qw_from(row_bytes_of(g, &format!("{p}ffn_gate_shexp.weight"), &sh_rows)?, "ffn_gate_shexp")?,
        shu: qw_from(row_bytes_of(g, &format!("{p}ffn_up_shexp.weight"), &sh_rows)?, "ffn_up_shexp")?,
        shd: qw_from(col_bytes_of(g, &format!("{p}ffn_down_shexp.weight"), &sh_rows)?, "ffn_down_shexp")?,
        ty_gu: tg.ty,
        ty_d: td.ty,
        gtab: DeviceBuffer::zeroed(c.n_expert)?,
        utab: DeviceBuffer::zeroed(c.n_expert)?,
        dtab: DeviceBuffer::zeroed(c.n_expert)?,
        own: DeviceBuffer::zeroed(c.n_expert)?,
        shd_pad: Some(pad_k(col_bytes_of(g, &format!("{p}ffn_down_shexp.weight"), &sh_rows)?, 128, "ffn_down_shexp")?),
    };
    let ple = if il == c.ple_layer {
        Some(Ple {
            key: Lin::from_tensor(g, &format!("{p}ple_key.weight"))?,
            value: Lin::from_tensor(g, &format!("{p}ple_value.weight"))?,
            nk: vecf(g, &format!("{p}ple_norm_key.weight"))?,
            nq: vecf(g, &format!("{p}ple_norm_query.weight"))?,
            nc: vecf(g, &format!("{p}ple_norm_conv.weight"))?,
            conv_w: vecf(g, &format!("{p}ple_conv1d.weight"))?,
            hist: DeviceBuffer::zeroed((c.ple_conv - 1) * c.ple_ngram * c.hc_dim())?,
        })
    } else {
        None
    };
    Ok(Layer { hca, hcf, mix, ffn, ple })
}

/// Expert owners per layer: experts ranked by routing count, dealt to the cards in a snake (0, 1, 1, 0, 0, 1, ..) so
/// both cards expect about the same expert work per token.
pub(crate) fn owners(c: &Config, counts: Option<&Vec<Vec<u64>>>) -> Vec<Vec<u8>> {
    (0..c.n_layer).map(|l| {
        let mut ord: Vec<usize> = (0..c.n_expert).collect();
        if let Some(cn) = counts {
            ord.sort_by(|&a, &b| cn[l][b].cmp(&cn[l][a]).then(a.cmp(&b)));
        }
        let mut own = vec![0u8; c.n_expert];
        for (i, &e) in ord.iter().enumerate() {
            own[e] = if i % 4 == 1 || i % 4 == 2 { 1 } else { 0 };
        }
        own
    }).collect()
}

/// Card 0's step-graph launches on their own thread while the main thread launches card 1's: a ~1400-node graph
/// launch costs ~1.5 ms of host time, so launched one after the other card 1 would start that much later and card 0
/// would wait for it at the first all-reduce of every step.
pub(crate) struct Launcher {
    st: Arc<AtomicU32>,
    job: Arc<[AtomicU64; 3]>,
    stop: Arc<AtomicBool>,
    h: Option<std::thread::JoinHandle<()>>,
}

const L_IDLE: u32 = 0;
const L_POSTED: u32 = 1;
const L_DONE: u32 = 2;
const L_FAILED: u32 = 3;

impl Launcher {
    /// A launcher thread with `cu` current, pinned to `cpu`.
    pub(crate) fn start(cu: Arc<fe_cuda::Context>, cpu: usize) -> Result<Launcher> {
        let st = Arc::new(AtomicU32::new(L_IDLE));
        let job: Arc<[AtomicU64; 3]> = Arc::new([AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)]);
        let stop = Arc::new(AtomicBool::new(false));
        let (st2, job2, stop2) = (st.clone(), job.clone(), stop.clone());
        let h = std::thread::Builder::new().name("fnext-tp-launch".into()).spawn(move || {
            if std::env::var("FNEXT_NO_PIN").is_err() {
                fe_ggq::cpu::pool::pin_current(cpu);
            }
            if cu.bind().is_err() {
                return;
            }
            let mut spins = 0u64;
            loop {
                if st2.load(Ordering::Acquire) != L_POSTED {
                    if stop2.load(Ordering::Relaxed) {
                        return;
                    }
                    spins += 1;
                    if spins > (1 << 22) {
                        std::thread::sleep(std::time::Duration::from_micros(20));
                    } else {
                        std::hint::spin_loop();
                    }
                    continue;
                }
                spins = 0;
                // SAFETY: the poster keeps the graph, stream and event alive and untouched until wait() returns
                let r = unsafe {
                    let g = &*(job2[0].load(Ordering::Relaxed) as *const Graph);
                    let s = &*(job2[1].load(Ordering::Relaxed) as *const Stream);
                    let e = &*(job2[2].load(Ordering::Relaxed) as *const Event);
                    g.launch(s).and_then(|_| s.record(e))
                };
                st2.store(if r.is_ok() { L_DONE } else { L_FAILED }, Ordering::Release);
            }
        })?;
        Ok(Launcher { st, job, stop, h: Some(h) })
    }

    /// Launch `g` on `s` and record `e` after it, on the launcher thread.
    /// # Safety
    /// `g`, `s` and `e` must stay alive and unmoved until [`Launcher::wait`] returns.
    pub(crate) unsafe fn post(&self, g: &Graph, s: &Stream, e: &Event) {
        self.job[0].store(g as *const Graph as u64, Ordering::Relaxed);
        self.job[1].store(s as *const Stream as u64, Ordering::Relaxed);
        self.job[2].store(e as *const Event as u64, Ordering::Relaxed);
        self.st.store(L_POSTED, Ordering::Release);
    }

    /// Wait until the posted launch has been issued.
    pub(crate) fn wait(&self) -> Result<()> {
        loop {
            match self.st.load(Ordering::Acquire) {
                L_DONE => {
                    self.st.store(L_IDLE, Ordering::Relaxed);
                    return Ok(());
                }
                L_FAILED => {
                    self.st.store(L_IDLE, Ordering::Relaxed);
                    bail!("FNEXT_TP=2: card 0 graph launch failed");
                }
                L_IDLE => return Ok(()),
                _ => std::hint::spin_loop(),
            }
        }
    }
}

impl Drop for Launcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.h.take() {
            let _ = h.join();
        }
    }
}
