//! Image input for Inkling-2-Mini: decode (PNG / JPEG), the reference preprocessing, and the vision encoder
//! (ViT backbone of 23 layers, width 1024, 16 heads, axial 2-D RoPE, then the 2-layer projector to the text width).
//!
//! One image token is a 40 x 40 pixel cell of the image (no resize): `rows = ceil(height / 40)`, `cols = ceil(width / 40)`,
//! `rows * cols` tokens. The encoder sees every cell as 2 x 2 native patches of 20 x 20 pixels plus one class token.
//!
//! Numerics: weights as shipped (bf16), activations f32 (kernels/k-inkv). Device memory is only held during `encode`:
//! the weights live in pinned host memory and are streamed one layer at a time through a small device buffer.
use crate::ckpt::Ckpt;
use anyhow::{bail, ensure, Result};
use fe_cuda::{DevPtr, DeviceBuffer, Function, HostBuffer, Module, Stream};

/// Pixels a side of one image token.
pub const CELL: usize = 40;
/// Pixels a side of one native ViT patch (2 x 2 patches a token).
pub const NATIVE: usize = 20;
pub const VD: usize = 1024;
pub const VHEADS: usize = 16;
pub const VLAYERS: usize = 23;
pub const VFF: usize = 4096;
pub const POS_GRID: usize = 32;
/// The placeholder token of the chat template (`<|unused_200054|>`): one per image token, replaced by an embedding row.
pub const IMAGE_TOKEN: u32 = 200054;
const PK: usize = 3 * NATIVE * NATIVE; // 1200 inputs of the patch embedding
const LN_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10000.0;
const NUL: DevPtr = DevPtr(0);

// flags of inkv_gemm (kernels/k-inkv)
const W_BF16: u32 = 1;
const W_TR: u32 = 2;
const ACT_GELU: u32 = 4;
const ACCUM: u32 = 8;
const GEMM_SMEM: u32 = 2 * 16 * 64 * 4;

/// A decoded image, 8-bit RGB, row-major.
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Decode a PNG or JPEG file. An image with transparency is drawn on the reference's checkerboard (8-pixel squares of 204 / 153).
pub fn decode(bytes: &[u8]) -> Result<Rgb> {
    let fmt = image::guess_format(bytes).map_err(|e| anyhow::anyhow!("image: unknown format ({e})"))?;
    ensure!(matches!(fmt, image::ImageFormat::Png | image::ImageFormat::Jpeg), "image: only PNG and JPEG are accepted (got {fmt:?})");
    let img = image::load_from_memory_with_format(bytes, fmt).map_err(|e| anyhow::anyhow!("image: decode failed ({e})"))?;
    let (width, height) = (img.width() as usize, img.height() as usize);
    ensure!(width > 0 && height > 0, "image: empty");
    let data = if img.color().has_alpha() {
        let rgba = img.to_rgba8().into_raw();
        let mut out = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let bg: u32 = if (y / 8 + x / 8) % 2 == 0 { 204 } else { 153 };
                let p = &rgba[(y * width + x) * 4..][..4];
                let a = p[3] as u32;
                for c in 0..3 {
                    out[(y * width + x) * 3 + c] = ((p[c] as u32 * a + bg * (255 - a) + 127) / 255) as u8;
                }
            }
        }
        out
    } else {
        img.to_rgb8().into_raw()
    };
    Ok(Rgb { width, height, data })
}

/// Width and height of a PNG or JPEG file, from its header (nothing is decoded).
pub fn dimensions(bytes: &[u8]) -> Result<(usize, usize)> {
    let fmt = image::guess_format(bytes).map_err(|e| anyhow::anyhow!("image: unknown format ({e})"))?;
    ensure!(matches!(fmt, image::ImageFormat::Png | image::ImageFormat::Jpeg), "image: only PNG and JPEG are accepted (got {fmt:?})");
    let (w, h) = image::ImageReader::with_format(std::io::Cursor::new(bytes), fmt).into_dimensions().map_err(|e| anyhow::anyhow!("image: bad header ({e})"))?;
    ensure!(w > 0 && h > 0, "image: empty");
    Ok((w as usize, h as usize))
}

/// The processor's output: `rows * cols` cells of [40][40][3] f32 in raster order, values (p - 127.5) / 127.5; the padding on the
/// right and at the bottom is the value -1 on the 0-255 scale.
pub struct Patches {
    pub rows: usize,
    pub cols: usize,
    pub pixels: Vec<f32>,
}
impl Patches {
    pub fn tokens(&self) -> usize {
        self.rows * self.cols
    }
}

pub fn preprocess(img: &Rgb) -> Patches {
    let (rows, cols) = (img.height.div_ceil(CELL), img.width.div_ceil(CELL));
    let mut pixels = vec![0f32; rows * cols * CELL * CELL * 3];
    let pad = (-1.0f32 - 127.5) / 127.5;
    for tr in 0..rows {
        for tc in 0..cols {
            let cell = &mut pixels[(tr * cols + tc) * CELL * CELL * 3..][..CELL * CELL * 3];
            for y in 0..CELL {
                let sy = tr * CELL + y;
                for x in 0..CELL {
                    let sx = tc * CELL + x;
                    for c in 0..3 {
                        cell[(y * CELL + x) * 3 + c] = if sy < img.height && sx < img.width {
                            (img.data[(sy * img.width + sx) * 3 + c] as f32 - 127.5) / 127.5
                        } else {
                            pad
                        };
                    }
                }
            }
        }
    }
    Patches { rows, cols, pixels }
}

/// Byte ranges of one layer's tensors inside its blob (all bf16).
#[derive(Clone, Copy, Default)]
struct LayerOff {
    ln1w: usize, ln1b: usize, qw: usize, qb: usize, kw: usize, kb: usize, vw: usize, vb: usize, ow: usize, ob: usize, ls1: usize,
    ln2w: usize, ln2b: usize, f1w: usize, f1b: usize, f2w: usize, f2b: usize, ls2: usize,
}

struct Kern {
    gemm: Function,
    ln: Function,
    softmax: Function,
    rope: Function,
}

pub struct Vision {
    /// every vision weight (bf16), pinned: [embed blob][layer blobs][proj_in][proj_out]
    host: HostBuffer<u8>,
    /// (offset, bytes) in `host`
    embed: (usize, usize),
    layers: Vec<(usize, usize, LayerOff)>,
    proj_in: (usize, usize),
    proj_out: (usize, usize),
    /// embed blob: patch weight, then ln_pre weight, bias
    lnpre_w: usize,
    lnpre_b: usize,
    /// learned position table [32 * 32][1024] and class + class-position row, f32
    pos_table: Vec<f32>,
    cls: Vec<f32>,
    k: Kern,
    text_dim: usize,
    proj_hidden: usize,
    /// largest blob, the size of the device staging buffer
    stage_bytes: usize,
    /// bytes of scores a query chunk may use (INK_VISION_SCORE_MB, default 64)
    score_bytes: usize,
}

/// Rows of the MLP processed at once (bounds the [rows][4096] f32 scratch to 16 MiB).
const MLP_ROWS: usize = 1024;

/// What `encode` returns.
pub struct Encoded {
    pub rows: usize,
    pub cols: usize,
    /// [rows * cols][text width] f32
    pub embeds: Vec<f32>,
    /// the backbone's last hidden state [1 + 4 * tokens][1024] (class token first), only when asked for
    pub hidden: Option<Vec<f32>>,
}
impl Encoded {
    /// The rows the text model takes: bf16 bits, round to nearest even.
    pub fn bf16(&self) -> Vec<u16> {
        self.embeds.iter().map(|&f| crate::cpuref::bf16_bits(f)).collect()
    }
}

impl Vision {
    /// Read the vision weights (`model.visual.*`) into pinned host memory. Needs a current CUDA context; takes no device memory
    /// beyond the kernel module.
    pub fn load(ck: &Ckpt) -> Result<Vision> {
        let pre = "model.visual.";
        let need = |name: &str, shape: &[usize]| -> Result<usize> {
            let i = ck.info(&format!("{pre}{name}"))?;
            ensure!(i.dtype == "BF16" && i.shape == shape, "{pre}{name}: {} {:?}, expected BF16 {shape:?}", i.dtype, i.shape);
            Ok(i.bytes)
        };
        let proj_hidden = ck.info(&format!("{pre}proj_in.weight"))?.shape[0];
        let text_dim = ck.info(&format!("{pre}proj_out.weight"))?.shape[0];
        ensure!(text_dim == crate::D, "vision projector output {text_dim} != text width {}", crate::D);
        ensure!(ck.info(&format!("{pre}backbone.layers.{}.ln_1.weight", VLAYERS - 1)).is_ok()
            && ck.info(&format!("{pre}backbone.layers.{VLAYERS}.ln_1.weight")).is_err(), "vision backbone: expected {VLAYERS} layers");
        // plan
        let mut names: Vec<(String, usize)> = vec![];
        let add = |names: &mut Vec<(String, usize)>, n: String, shape: &[usize]| -> Result<usize> {
            let off: usize = names.iter().map(|x| x.1).sum();
            let b = need(&n, shape)?;
            names.push((n, b));
            Ok(off)
        };
        let e0 = add(&mut names, "backbone.embeddings.patch_embedding.weight".into(), &[VD, PK])?;
        let lnpre_w = add(&mut names, "backbone.ln_pre.weight".into(), &[VD])? - e0;
        let lnpre_b = add(&mut names, "backbone.ln_pre.bias".into(), &[VD])? - e0;
        let mut marks = vec![e0];
        let mut layers = vec![];
        for li in 0..VLAYERS {
            let p = format!("backbone.layers.{li}.");
            let l0: usize = names.iter().map(|x| x.1).sum();
            marks.push(l0);
            let mut o = LayerOff::default();
            macro_rules! t { ($f:ident, $n:expr, $s:expr) => { o.$f = add(&mut names, format!("{p}{}", $n), $s)? - l0; }; }
            t!(ln1w, "ln_1.weight", &[VD]);
            t!(ln1b, "ln_1.bias", &[VD]);
            t!(qw, "self_attn.q_proj.weight", &[VD, VD]);
            t!(qb, "self_attn.q_proj.bias", &[VD]);
            t!(kw, "self_attn.k_proj.weight", &[VD, VD]);
            t!(kb, "self_attn.k_proj.bias", &[VD]);
            t!(vw, "self_attn.v_proj.weight", &[VD, VD]);
            t!(vb, "self_attn.v_proj.bias", &[VD]);
            t!(ow, "self_attn.out_proj.weight", &[VD, VD]);
            t!(ob, "self_attn.out_proj.bias", &[VD]);
            t!(ls1, "ls_1.gamma", &[VD]);
            t!(ln2w, "ln_2.weight", &[VD]);
            t!(ln2b, "ln_2.bias", &[VD]);
            t!(f1w, "mlp.c_fc.weight", &[VFF, VD]);
            t!(f1b, "mlp.c_fc.bias", &[VFF]);
            t!(f2w, "mlp.c_proj.weight", &[VD, VFF]);
            t!(f2b, "mlp.c_proj.bias", &[VD]);
            t!(ls2, "ls_2.gamma", &[VD]);
            layers.push((l0, 0usize, o));
        }
        let pi = add(&mut names, "proj_in.weight".into(), &[proj_hidden, 4 * VD])?;
        let po = add(&mut names, "proj_out.weight".into(), &[text_dim, proj_hidden])?;
        let total: usize = names.iter().map(|x| x.1).sum();
        marks.push(pi);
        for (i, l) in layers.iter_mut().enumerate() {
            l.1 = marks[i + 2] - l.0;
        }
        let embed = (e0, marks[1] - e0);
        let (proj_in, proj_out) = ((pi, po - pi), (po, total - po));
        let mut host = HostBuffer::<u8>::alloc(total)?;
        let mut off = 0;
        for (n, b) in &names {
            let raw = ck.read(&format!("{pre}{n}"))?;
            host.as_mut_slice()[off..off + b].copy_from_slice(&raw);
            off += b;
        }
        let pos_table = ck.read_f32(&format!("{pre}backbone.embeddings.position_embedding"))?;
        ensure!(pos_table.len() == POS_GRID * POS_GRID * VD, "vision position table: {} values", pos_table.len());
        let ce = ck.read_f32(&format!("{pre}backbone.embeddings.class_embedding"))?;
        let cp = ck.read_f32(&format!("{pre}backbone.embeddings.cls_position_embedding"))?;
        let cls: Vec<f32> = ce.iter().zip(&cp).map(|(a, b)| a + b).collect();
        let m = Module::load(crate::K_INKV_CUBIN)?;
        let k = Kern { gemm: m.function("inkv_gemm")?, ln: m.function("inkv_ln")?, softmax: m.function("inkv_softmax")?,
            rope: m.function("inkv_rope")? };
        let stage_bytes = layers.iter().map(|l| l.1).chain([embed.1, proj_in.1, proj_out.1]).max().unwrap();
        let score_mb: usize = std::env::var("INK_VISION_SCORE_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        Ok(Vision { host, embed, layers, proj_in, proj_out, lnpre_w, lnpre_b, pos_table, cls, k, text_dim, proj_hidden, stage_bytes,
            score_bytes: score_mb.max(1) << 20 })
    }

    /// Pinned host bytes held by the weights.
    pub fn host_bytes(&self) -> usize {
        self.host.len()
    }

    fn score_rows(&self, n: usize) -> usize {
        (self.score_bytes / (n * 4)).clamp(64, n.max(64)).min(n)
    }

    /// Device bytes `encode` allocates for an image of `tokens` image tokens (freed when it returns).
    pub fn work_bytes(&self, tokens: usize) -> usize {
        let n = 4 * tokens + 1;
        5 * n * VD * 4 + n.min(MLP_ROWS) * VFF * 4 + self.score_rows(n) * n * 4 + self.stage_bytes + n * 64 * 4
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, s: &Stream, c: DevPtr, a: DevPtr, w: DevPtr, bias: DevPtr, ls: DevPtr, m: usize, n: usize, k: usize, lda: usize,
        ldw: usize, ldc: usize, flags: u32) -> Result<()> {
        // the kernel computes element indexes in 32 bits
        let w_last = if flags & W_TR != 0 { (k - 1) * ldw + n } else { (n - 1) * ldw + k };
        ensure!(m >= 1 && n >= 1 && k >= 1 && ((m - 1) * lda + k).max(w_last).max((m - 1) * ldc + n) <= u32::MAX as usize, "vision: a GEMM operand has more than 2^32 elements");
        unsafe {
            s.launch(&self.k.gemm, ((n as u32).div_ceil(64), (m as u32).div_ceil(64)), 256u32, GEMM_SMEM,
                (c, a, w, bias, ls, m as u32, n as u32, k as u32, lda as u32, ldw as u32, ldc as u32, flags))?;
        }
        Ok(())
    }
    fn ln(&self, s: &Stream, y: DevPtr, x: DevPtr, w: DevPtr, b: DevPtr, rows: usize) -> Result<()> {
        unsafe { s.launch(&self.k.ln, rows as u32, 256u32, 0, (y, x, w, b, VD as u32, VD as u32, VD as u32, LN_EPS))? };
        Ok(())
    }
    fn stage(&self, s: &Stream, dst: &DeviceBuffer<u8>, blob: (usize, usize)) -> Result<()> {
        // pinned source, same stream as the kernels: the copy runs after the kernels that still read the previous blob
        unsafe { s.memcpy_htod(dst.ptr(), &self.host.as_slice()[blob.0..blob.0 + blob.1])? };
        Ok(())
    }

    /// Encode one image: `rows * cols` embedding rows of the text width. `want_hidden` also returns the backbone's last hidden state.
    pub fn encode(&self, s: &Stream, p: &Patches, want_hidden: bool) -> Result<Encoded> {
        let (rows, cols) = (p.rows, p.cols);
        let t = rows * cols;
        ensure!(t > 0 && p.pixels.len() == t * CELL * CELL * 3, "vision: bad patch buffer");
        let np = 4 * t;
        let n = np + 1;
        ensure!(n < (1 << 22), "vision: image too large ({t} tokens)");
        // ---- host side: patch matrix, position embeddings, rotary table
        // native patch i = token-major, then the 2 x 2 patches of the token in raster order
        let mut pm = vec![0f32; np * PK];
        let mut h0 = vec![0f32; n * VD];
        let mut cs = vec![0f32; n * 64];
        h0[..VD].copy_from_slice(&self.cls);
        let inv: Vec<f32> = (0..16).map(|j| 1.0 / ROPE_THETA.powf((2 * j) as f32 / 32.0)).collect();
        for j in 0..32 {
            cs[j] = 1.0; // class token at (0, 0): cos 1, sin 0
        }
        let (gh, gw) = (rows * 2, cols * 2);
        let taps = |i: usize, size: usize| -> (usize, usize, f32, f32) {
            let src = (i as f32 + 0.5) * POS_GRID as f32 / size as f32 - 0.5;
            let fl = src.floor();
            let w1 = src - fl;
            let c = |v: f32| (v as i64).clamp(0, POS_GRID as i64 - 1) as usize;
            (c(fl), c(fl + 1.0), 1.0 - w1, w1)
        };
        for tok in 0..t {
            let (tr, tc) = (tok / cols, tok % cols);
            let cell = &p.pixels[tok * CELL * CELL * 3..][..CELL * CELL * 3];
            for sub in 0..4 {
                let (mr, mc) = (sub / 2, sub % 2);
                let i = tok * 4 + sub;
                let dst = &mut pm[i * PK..][..PK];
                for y in 0..NATIVE {
                    let srow = &cell[((mr * NATIVE + y) * CELL + mc * NATIVE) * 3..][..NATIVE * 3];
                    dst[y * NATIVE * 3..][..NATIVE * 3].copy_from_slice(srow);
                }
                let (pr, pc) = (tr * 2 + mr, tc * 2 + mc);
                let (a0, a1, wa0, wa1) = taps(pr, gh);
                let (b0, b1, wb0, wb1) = taps(pc, gw);
                let out = &mut h0[(1 + i) * VD..][..VD];
                for (ti, wt) in [(a0 * POS_GRID + b0, wa0 * wb0), (a0 * POS_GRID + b1, wa0 * wb1), (a1 * POS_GRID + b0, wa1 * wb0),
                    (a1 * POS_GRID + b1, wa1 * wb1)]
                {
                    let row = &self.pos_table[ti * VD..][..VD];
                    for d in 0..VD {
                        out[d] += row[d] * wt;
                    }
                }
                // rotary angles: positions are 1-based (row, col); pairs 0..16 turn by the column, 16..32 by the row
                let c = &mut cs[(1 + i) * 64..][..64];
                for j in 0..16 {
                    let (aw, ah) = ((pc + 1) as f32 * inv[j], (pr + 1) as f32 * inv[j]);
                    c[j] = aw.cos();
                    c[16 + j] = ah.cos();
                    c[32 + j] = aw.sin();
                    c[48 + j] = ah.sin();
                }
            }
        }
        // ---- device buffers (freed on return)
        let x = n * VD * 4; // bytes of one [n][1024] f32 plane
        let act = DeviceBuffer::<u8>::alloc(5 * x)?;
        let (h, a, q, kx, v) = (act.ptr(), act.ptr().byte_add(x), act.ptr().byte_add(2 * x), act.ptr().byte_add(3 * x), act.ptr().byte_add(4 * x));
        let mrows = n.min(MLP_ROWS);
        let f1 = DeviceBuffer::<u8>::alloc(mrows * VFF * 4)?;
        let rc = self.score_rows(n);
        let sc = DeviceBuffer::<u8>::alloc(rc * n * 4)?;
        let stage = DeviceBuffer::<u8>::alloc(self.stage_bytes)?;
        let csd = DeviceBuffer::<u8>::alloc(n * 64 * 4)?;
        let w = |off: usize| stage.ptr().byte_add(off);
        unsafe {
            s.memcpy_htod(h, &h0)?;
            s.memcpy_htod(a, &pm)?; // the patch matrix borrows the a..v planes (np * 1200 <= 4 * n * 1024)
            s.memcpy_htod(csd.ptr(), &cs)?;
        }
        // ---- embeddings: h[1 + i] = pos[i] + patch[i] . We^T; pre-LayerNorm on every row
        self.stage(s, &stage, self.embed)?;
        self.gemm(s, h.byte_add(VD * 4), a, w(0), NUL, NUL, np, VD, PK, PK, PK, VD, W_BF16 | ACCUM)?;
        self.ln(s, h, h, w(self.lnpre_w), w(self.lnpre_b), n)?;
        s.sync()?; // h0 / pm / cs are host vectors: the copies must be done before they are dropped (they live to the end, but keep it simple)
        // ---- layers
        for (l0, lb, o) in &self.layers {
            self.stage(s, &stage, (*l0, *lb))?;
            self.ln(s, a, h, w(o.ln1w), w(o.ln1b), n)?;
            self.gemm(s, q, a, w(o.qw), w(o.qb), NUL, n, VD, VD, VD, VD, VD, W_BF16)?;
            self.gemm(s, kx, a, w(o.kw), w(o.kb), NUL, n, VD, VD, VD, VD, VD, W_BF16)?;
            self.gemm(s, v, a, w(o.vw), w(o.vb), NUL, n, VD, VD, VD, VD, VD, W_BF16)?;
            unsafe { s.launch(&self.k.rope, n as u32, 256u32, 0, (q, kx, csd.ptr(), VHEADS as u32, VD as u32, 0.125f32))? };
            // attention, a head and a chunk of query rows at a time: scores, softmax, p . V into the `a` plane
            for hd in 0..VHEADS {
                let mut r0 = 0;
                while r0 < n {
                    let m = (n - r0).min(rc);
                    self.gemm(s, sc.ptr(), q.byte_add((r0 * VD + hd * 64) * 4), kx.byte_add(hd * 64 * 4), NUL, NUL, m, n, 64, VD, VD, n, 0)?;
                    unsafe { s.launch(&self.k.softmax, m as u32, 256u32, 0, (sc.ptr(), n as u32, n as u32))? };
                    self.gemm(s, a.byte_add((r0 * VD + hd * 64) * 4), sc.ptr(), v.byte_add(hd * 64 * 4), NUL, NUL, m, 64, n, n, VD, VD, W_TR)?;
                    r0 += m;
                }
            }
            self.gemm(s, h, a, w(o.ow), w(o.ob), w(o.ls1), n, VD, VD, VD, VD, VD, W_BF16 | ACCUM)?;
            self.ln(s, a, h, w(o.ln2w), w(o.ln2b), n)?;
            let mut r0 = 0;
            while r0 < n {
                let m = (n - r0).min(mrows);
                self.gemm(s, f1.ptr(), a.byte_add(r0 * VD * 4), w(o.f1w), w(o.f1b), NUL, m, VFF, VD, VD, VD, VFF, W_BF16 | ACT_GELU)?;
                self.gemm(s, h.byte_add(r0 * VD * 4), f1.ptr(), w(o.f2w), w(o.f2b), w(o.ls2), m, VD, VFF, VFF, VFF, VD, W_BF16 | ACCUM)?;
                r0 += m;
            }
        }
        // ---- projector: the 4 patch rows of a token side by side -> proj_in, GELU -> proj_out
        ensure!(self.proj_hidden * 4 <= 4 * VD * 4 && self.proj_hidden <= 4 * VD, "vision projector width");
        self.stage(s, &stage, self.proj_in)?;
        self.gemm(s, a, h.byte_add(VD * 4), w(0), NUL, NUL, t, self.proj_hidden, 4 * VD, 4 * VD, 4 * VD, self.proj_hidden, W_BF16 | ACT_GELU)?;
        self.stage(s, &stage, self.proj_out)?;
        self.gemm(s, q, a, w(0), NUL, NUL, t, self.text_dim, self.proj_hidden, self.proj_hidden, self.proj_hidden, self.text_dim, W_BF16)?;
        s.sync()?;
        let mut embeds = vec![0f32; t * self.text_dim];
        fe_cuda::memcpy_dtoh(&mut embeds, q)?;
        let hidden = if want_hidden {
            let mut hv = vec![0f32; n * VD];
            fe_cuda::memcpy_dtoh(&mut hv, h)?;
            Some(hv)
        } else {
            None
        };
        if embeds.iter().any(|x| !x.is_finite()) {
            bail!("vision: non-finite embedding");
        }
        Ok(Encoded { rows, cols, embeds, hidden })
    }
}
