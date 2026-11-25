//! Image input for a request: `data:` URLs, the limits, the identity of a picture in the prompt cache and the rows the prompt takes.
//!
//! The engine takes pictures only with a request. A picture is decoded once (PNG / JPEG, `vision::decode`), hashed (SHA-256 of its
//! size and RGB pixels) and given a number; the same pixels get the same number for the life of the process. In the prompt every image
//! token of a picture carries the id `IMG_ID_BASE | number` instead of the template's placeholder id (all placeholders share one id, so
//! ids alone could not tell two pictures apart). The prompt cache, the lanes and the park pool compare those ids like any token: another
//! picture never matches a cached prefix, the same picture does. `Ink::forward` replaces such a row by the picture's embedding row.
//!
//! The model has no resize rule (one image token per 40 x 40 pixels, whatever the size). A picture above the per-picture budget
//! (`INK_IMAGE_TOKENS_EACH`) is shrunk here, once, on its decoded 8-bit RGB pixels, with its aspect ratio kept, until it fits
//! ([`fit_size`], Lanczos3); the shrunk pixels are the picture from then on (hash, number, rows). A picture within the
//! budget is not touched.
//!
//! The vision encoder runs only for pictures whose rows are prefilled: a later turn that resumes behind a picture needs no rows at all.
//! Rows of recent pictures are kept in host memory (`INK_IMAGE_CACHE_MB`), so a prompt that is prefilled again (cache off, another lane,
//! an edit in front of the picture) does not encode again either.
use crate::engine::{ImageSpan, Ink};
use crate::vision::{self, Rgb, Vision, CELL};
use crate::D;
use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Prompt ids from here up stand for image rows: `IMG_ID_BASE | picture number`.
pub const IMG_ID_BASE: u32 = 0x8000_0000;

pub fn is_image_id(id: u32) -> bool {
    id >= IMG_ID_BASE
}

/// Limits of a request (settings). A request beyond them is refused with a clear message, except for `max_tokens_each`: a picture
/// above it is shrunk to fit.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// pictures a request (INK_IMAGE_MAX, default 16)
    pub max_images: usize,
    /// image tokens a request, all pictures together (INK_IMAGE_TOKENS, default 16384)
    pub max_tokens: usize,
    /// image tokens of one picture (INK_IMAGE_TOKENS_EACH, default 2304 = 2560 x 1440); a larger picture is shrunk to fit
    pub max_tokens_each: usize,
    /// bytes of one picture file (INK_IMAGE_MB, default 20)
    pub max_bytes: usize,
}

impl Limits {
    pub fn from_env() -> Limits {
        let e = |n: &str, d: usize| std::env::var(n).ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(d);
        Limits { max_images: e("INK_IMAGE_MAX", 16), max_tokens: e("INK_IMAGE_TOKENS", 16384), max_tokens_each: e("INK_IMAGE_TOKENS_EACH", 2304),
            max_bytes: e("INK_IMAGE_MB", 20) << 20 }
    }
}

/// A decoded picture of a request.
pub struct Picture {
    /// its number: the same pixels get the same number (see [`IMG_ID_BASE`])
    pub key: u32,
    pub width: usize,
    pub height: usize,
    /// 8-bit RGB, row-major
    pub rgb: Vec<u8>,
}

impl std::fmt::Debug for Picture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Picture {{ key: {}, {} x {} }}", self.key, self.width, self.height)
    }
}

impl Picture {
    pub fn rows(&self) -> usize {
        self.height.div_ceil(CELL)
    }
    pub fn cols(&self) -> usize {
        self.width.div_ceil(CELL)
    }
    /// Image tokens: one per 40 x 40 pixel cell.
    pub fn tokens(&self) -> usize {
        self.rows() * self.cols()
    }
    /// The id its rows carry in the prompt.
    pub fn id(&self) -> u32 {
        IMG_ID_BASE | self.key
    }
}

/// The bytes of a `data:image/png;base64,...` or `data:image/jpeg;base64,...` URL. Anything else is an error: the server never fetches a
/// picture from the network or from a file.
pub fn data_url_bytes(url: &str, max_bytes: usize) -> Result<Vec<u8>, String> {
    let head: String = url.chars().take(8).collect::<String>().to_ascii_lowercase();
    if head.starts_with("http://") || head.starts_with("https://") {
        return Err("image_url: only data: URLs are accepted (data:image/png;base64,... or data:image/jpeg;base64,...); this server does not fetch pictures from the network".into());
    }
    if !head.starts_with("data:") {
        return Err("image_url: only data: URLs are accepted (data:image/png;base64,... or data:image/jpeg;base64,...)".into());
    }
    let Some(comma) = url.find(',') else {
        return Err("image_url: a data: URL without a ',' before the data".into());
    };
    let meta = url[5..comma].to_ascii_lowercase();
    let mut parts = meta.split(';').map(str::trim);
    let mime = parts.next().unwrap_or("");
    if !matches!(mime, "image/png" | "image/jpeg" | "image/jpg") {
        return Err(format!("image_url: media type '{mime}' is not accepted (image/png and image/jpeg are)"));
    }
    if !parts.any(|p| p == "base64") {
        return Err("image_url: the data: URL must be base64 (data:image/png;base64,...)".into());
    }
    let data = &url[comma + 1..];
    if data.len() / 4 * 3 > max_bytes + 3 {
        return Err(format!("image_url: the picture is larger than {} MiB (INK_IMAGE_MB)", max_bytes >> 20));
    }
    base64_decode(data).ok_or_else(|| "image_url: the base64 data is not valid".to_string())
}

/// Standard base64 (also the URL-safe letters); white space is skipped, padding is optional.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    let mut pad = false;
    for &c in s.as_bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => {
                pad = true;
                continue;
            }
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        };
        if pad {
            return None;
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    // left-over bits must be the zero padding of the last group
    if bits >= 6 || acc != 0 {
        return None;
    }
    Some(out)
}

/// SHA-256 (FIPS 180-4).
pub fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let block = |h: &mut [u32; 8], b: &[u8]| {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = *h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    };
    let mut buf = [0u8; 64];
    let mut fill = 0usize;
    let mut total = 0u64;
    for p in parts {
        total += p.len() as u64;
        let mut p = *p;
        if fill > 0 {
            let c = (64 - fill).min(p.len());
            buf[fill..fill + c].copy_from_slice(&p[..c]);
            fill += c;
            p = &p[c..];
            if fill == 64 {
                block(&mut h, &buf);
                fill = 0;
            }
        }
        while p.len() >= 64 {
            block(&mut h, &p[..64]);
            p = &p[64..];
        }
        buf[..p.len()].copy_from_slice(p);
        fill += p.len();
    }
    buf[fill] = 0x80;
    for b in buf[fill + 1..].iter_mut() {
        *b = 0;
    }
    if fill + 1 > 56 {
        block(&mut h, &buf);
        buf = [0u8; 64];
    }
    buf[56..].copy_from_slice(&(total * 8).to_be_bytes());
    block(&mut h, &buf);
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

/// Picture numbers by pixel hash, for the life of the process (the oldest are forgotten beyond 65536 pictures: such a picture gets a
/// new number when it comes again, which only costs a cache miss).
struct Registry {
    map: HashMap<[u8; 32], u32>,
    order: VecDeque<[u8; 32]>,
    next: u32,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

fn number_of(hash: [u8; 32]) -> Result<u32, String> {
    let mut g = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let r = g.get_or_insert_with(|| Registry { map: HashMap::new(), order: VecDeque::new(), next: 1 });
    if let Some(&n) = r.map.get(&hash) {
        return Ok(n);
    }
    if r.next >= IMG_ID_BASE - 1 {
        return Err("image input: picture numbers are used up (restart the server)".into());
    }
    let n = r.next;
    r.next += 1;
    r.map.insert(hash, n);
    r.order.push_back(hash);
    if r.order.len() > 65536 {
        if let Some(old) = r.order.pop_front() {
            r.map.remove(&old);
        }
    }
    Ok(n)
}

/// A picture file whose header gives a side above this is refused before it is decoded: a small file must not decode to a giant picture.
pub const MAX_SIDE: usize = 16384;
/// ... and so is one of more pixels than this (3 bytes a pixel in host memory while it is decoded and shrunk).
pub const MAX_PIXELS: usize = 100_000_000;

fn check_size(w: usize, h: usize) -> Result<(), String> {
    if w > MAX_SIDE || h > MAX_SIDE || w * h > MAX_PIXELS {
        return Err(format!("image: a picture of {w} x {h} pixels is too large to take in (at most {MAX_SIDE} pixels a side and {} million pixels). Send a smaller picture.",
            MAX_PIXELS / 1_000_000));
    }
    Ok(())
}

/// The size a picture of `w` x `h` pixels is given so that it is at most `budget` image tokens (one per 40 x 40 pixel cell, rounded
/// up): `(w, h)` itself when it fits, never larger. Otherwise both sides are scaled by s = sqrt(budget * 1600 / (w * h)) and rounded
/// down; if the rounding up to whole cells still gives too many tokens, s steps down to the next scale that drops a column or a row,
/// until it fits. 3840 x 2160 with a budget of 2304 is 2560 x 1440.
pub fn fit_size(w: usize, h: usize, budget: usize) -> (usize, usize) {
    let budget = budget.max(1);
    let tokens = |w: usize, h: usize| w.div_ceil(CELL) * h.div_ceil(CELL);
    if w == 0 || h == 0 || tokens(w, h) <= budget {
        return (w, h);
    }
    let (wf, hf) = (w as f64, h as f64);
    let mut s = (budget as f64 * (CELL * CELL) as f64 / (wf * hf)).sqrt().min(1.0);
    loop {
        // 1e-6: a product that is a whole number in exact arithmetic must not lose a pixel to the rounding of s
        let nw = ((wf * s + 1e-6).floor() as usize).clamp(1, w);
        let nh = ((hf * s + 1e-6).floor() as usize).clamp(1, h);
        if tokens(nw, nh) <= budget {
            return (nw, nh);
        }
        let (cols, rows) = (nw.div_ceil(CELL), nh.div_ceil(CELL));
        let sa = if cols > 1 { ((cols - 1) * CELL) as f64 / wf } else { 0.0 };
        let sb = if rows > 1 { ((rows - 1) * CELL) as f64 / hf } else { 0.0 };
        // one column or one row less each time; cols = rows = 1 is one token and has returned above
        s = sa.max(sb);
    }
}

/// Shrink 8-bit RGB pixels to `nw` x `nh`. Lanczos3: the filter is widened by the scale factor, so every source pixel is weighed in
/// (no aliasing), and unlike a plain area average it keeps thin strokes sharp at factors that are not whole numbers (1.5 for a
/// 3840 x 2160 screenshot), which is what small text needs.
fn shrink(rgb: Rgb, nw: usize, nh: usize) -> Result<Rgb, String> {
    let img = image::RgbImage::from_raw(rgb.width as u32, rgb.height as u32, rgb.data).ok_or("image: pixel buffer of the wrong size")?;
    let out = image::imageops::resize(&img, nw as u32, nh as u32, image::imageops::FilterType::Lanczos3);
    Ok(Rgb { width: nw, height: nh, data: out.into_raw() })
}

/// Decode one picture file of a request and give it its number. The size is read from the file header first, so a picture of an
/// absurd size is refused before it is decoded. A picture above `lim.max_tokens_each` image tokens is shrunk to fit ([`fit_size`]);
/// one within it keeps its pixels as decoded.
pub fn intake(bytes: &[u8], lim: &Limits) -> Result<Picture, String> {
    if bytes.len() > lim.max_bytes {
        return Err(format!("image: the picture file is {:.1} MiB; the limit is {} MiB (INK_IMAGE_MB)", bytes.len() as f64 / 1048576.0, lim.max_bytes >> 20));
    }
    let (w, h) = vision::dimensions(bytes).map_err(|e| format!("{e:#}"))?;
    check_size(w, h)?;
    let mut rgb = vision::decode(bytes).map_err(|e| format!("{e:#}"))?;
    check_size(rgb.width, rgb.height)?;
    let (w, h) = (rgb.width, rgb.height);
    let (nw, nh) = fit_size(w, h, lim.max_tokens_each);
    if (nw, nh) != (w, h) {
        let t = std::time::Instant::now();
        rgb = shrink(rgb, nw, nh)?;
        eprintln!("ink vision: picture of {w} x {h} pixels ({} image tokens) shrunk to {nw} x {nh} ({} image tokens; INK_IMAGE_TOKENS_EACH is {}), {:.0} ms",
            w.div_ceil(CELL) * h.div_ceil(CELL), nw.div_ceil(CELL) * nh.div_ceil(CELL), lim.max_tokens_each, t.elapsed().as_secs_f64() * 1e3);
    }
    let Rgb { width, height, data } = rgb;
    let hash = sha256(&[&(width as u32).to_le_bytes(), &(height as u32).to_le_bytes(), &data]);
    Ok(Picture { key: number_of(hash)?, width, height, rgb: data })
}

/// A picture of a request at its place in the prompt: its rows are positions [pos, pos + rows * cols).
pub struct Placed<'a> {
    pub pos: usize,
    pub key: u32,
    pub width: usize,
    pub height: usize,
    pub rgb: &'a [u8],
}

/// What [`Images::prepare`] did for a request.
#[derive(Default, Clone, Debug)]
pub struct Report {
    /// pictures wholly inside the resumed prefix: no rows needed, no encode
    pub in_prompt_cache: usize,
    /// pictures whose rows came from the row cache: no encode
    pub from_row_cache: usize,
    pub encoded: usize,
    pub encoded_tokens: usize,
    pub encode_ms: f64,
    /// experts moved to host memory to make room (16 GB layout), and the time of both moves
    pub experts_moved: usize,
    pub fit_ms: f64,
}

/// The engine thread's image state: the vision encoder (loaded with the first picture; its weights stay in pinned host memory) and
/// the row cache.
pub struct Images {
    model_dir: PathBuf,
    vis: Option<Vision>,
    cache: HashMap<u32, (Arc<Vec<u16>>, u64)>,
    bytes: usize,
    budget: usize,
    tick: u64,
}

impl Images {
    /// INK_IMAGE_CACHE_MB: host memory for the rows of recent pictures (default 256; 0 = keep none).
    pub fn new(model_dir: &std::path::Path) -> Images {
        let mb = std::env::var("INK_IMAGE_CACHE_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(256);
        Images { model_dir: model_dir.to_path_buf(), vis: None, cache: HashMap::new(), bytes: 0, budget: mb << 20, tick: 0 }
    }

    /// Host bytes of the cached rows.
    pub fn cache_bytes(&self) -> usize {
        self.bytes
    }

    /// The rows the prefill of a request needs when it starts at position `from` (the prompt cache's resume point): nothing for a
    /// picture that ends at or before `from`, cached rows, or a run of the encoder. The encoder's device buffers exist only inside
    /// this call; on a card with expert slots a few experts make room for them and come back before the call returns.
    pub fn prepare(&mut self, ink: &mut Ink, pics: &[Placed], from: usize) -> Result<(Vec<ImageSpan>, Report)> {
        let mut rep = Report::default();
        let mut spans = Vec::new();
        let mut moved = false;
        let mut done: Vec<(u32, Arc<Vec<u16>>)> = Vec::new();
        let h0 = ink.host_experts();
        for p in pics {
            let (rows, cols) = (p.height.div_ceil(CELL), p.width.div_ceil(CELL));
            let tokens = rows * cols;
            if p.pos + tokens <= from {
                rep.in_prompt_cache += 1;
                continue;
            }
            self.tick += 1;
            if let Some(e) = self.cache.get_mut(&p.key) {
                e.1 = self.tick;
                spans.push(ImageSpan { pos: p.pos, rows: e.0.clone() });
                rep.from_row_cache += 1;
                continue;
            }
            // the same picture twice in this request, with the row cache off
            if let Some(r) = done.iter().find(|d| d.0 == p.key).map(|d| d.1.clone()) {
                spans.push(ImageSpan { pos: p.pos, rows: r });
                rep.from_row_cache += 1;
                continue;
            }
            ink.ctx.bind()?;
            if self.vis.is_none() {
                let t = std::time::Instant::now();
                let ck = crate::ckpt::Ckpt::open(&self.model_dir.join("model-00001-of-00001.safetensors"))?;
                let v = Vision::load(&ck).context("vision encoder weights")?;
                eprintln!("ink vision: encoder weights loaded, {:.0} MiB in pinned host memory ({:.2} s)", v.host_bytes() as f64 / 1048576.0, t.elapsed().as_secs_f64());
                self.vis = Some(v);
            }
            let vis = self.vis.as_ref().unwrap();
            let pt = vision::preprocess(&Rgb { width: p.width, height: p.height, data: p.rgb.to_vec() });
            anyhow::ensure!(pt.tokens() == tokens, "image: {} tokens after preprocessing, {tokens} placed", pt.tokens());
            if let (Some(floor), true) = (ink.fit_floor(), ink.slot_dev_bytes() > 0) {
                let t = std::time::Instant::now();
                ink.s.sync()?;
                ink.fit_host_experts(vis.work_bytes(tokens) + floor, usize::MAX / 4)?;
                rep.fit_ms += t.elapsed().as_secs_f64() * 1e3;
                moved = true;
            }
            let t = std::time::Instant::now();
            let enc = vis.encode(&ink.s, &pt, false).with_context(|| format!("vision encoder ({} x {} pixels, {tokens} tokens, {:.0} MiB of device memory while it runs)",
                p.width, p.height, vis.work_bytes(tokens) as f64 / 1048576.0))?;
            rep.encode_ms += t.elapsed().as_secs_f64() * 1e3;
            rep.encoded += 1;
            rep.encoded_tokens += tokens;
            let rows = Arc::new(enc.bf16());
            anyhow::ensure!(rows.len() == tokens * D, "vision: {} values for {tokens} tokens", rows.len());
            self.put(p.key, rows.clone());
            done.push((p.key, rows.clone()));
            spans.push(ImageSpan { pos: p.pos, rows });
        }
        if moved {
            rep.experts_moved = ink.host_experts().saturating_sub(h0);
            // the encoder's buffers are free again: the experts come back up to the floor
            if let Some(floor) = ink.fit_floor() {
                let t = std::time::Instant::now();
                ink.fit_host_experts(floor, floor)?;
                rep.fit_ms += t.elapsed().as_secs_f64() * 1e3;
            }
        }
        Ok((spans, rep))
    }

    fn put(&mut self, key: u32, rows: Arc<Vec<u16>>) {
        let b = rows.len() * 2;
        if b > self.budget {
            return;
        }
        while self.bytes + b > self.budget {
            let Some((&old, _)) = self.cache.iter().min_by_key(|(_, v)| v.1) else { break };
            if let Some((r, _)) = self.cache.remove(&old) {
                self.bytes -= r.len() * 2;
            }
        }
        self.bytes += b;
        self.cache.insert(key, (rows, self.tick));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn sha256_matches_the_standard_vectors() {
        assert_eq!(hex(&sha256(&[b""])), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256(&[b"abc"])), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(hex(&sha256(&[b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"])), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        // the same bytes in pieces
        assert_eq!(sha256(&[b"abcdbcdecdefdefgefghfghigh", b"", b"ijhijkijkljklmklmnlmnomnopnopq"]), sha256(&[b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"]));
        let m = vec![b'a'; 1_000_000];
        assert_eq!(hex(&sha256(&[&m[..999], &m[999..]])), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    #[test]
    fn base64_and_data_urls() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVs\nbG8").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8h").unwrap(), b"hello!");
        assert!(base64_decode("aGV$bG8=").is_none());
        assert!(base64_decode("aGVsbG8=x").is_none());
        assert_eq!(data_url_bytes("data:image/png;base64,aGVsbG8=", 1 << 20).unwrap(), b"hello");
        assert_eq!(data_url_bytes("DATA:image/JPEG;charset=x;base64,aGVsbG8=", 1 << 20).unwrap(), b"hello");
        assert!(data_url_bytes("https://example.com/a.png", 1 << 20).unwrap_err().contains("does not fetch"));
        assert!(data_url_bytes("file:///tmp/a.png", 1 << 20).unwrap_err().contains("only data: URLs"));
        assert!(data_url_bytes("data:image/gif;base64,aGVsbG8=", 1 << 20).unwrap_err().contains("image/gif"));
        assert!(data_url_bytes("data:image/png,hello", 1 << 20).unwrap_err().contains("base64"));
        assert!(data_url_bytes(&format!("data:image/png;base64,{}", "A".repeat(4000)), 1000).unwrap_err().contains("INK_IMAGE_MB"));
    }

    fn png(w: u32, h: u32, shade: u8) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x % 251) as u8, (y % 241) as u8, shade]));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    #[test]
    fn intake_counts_tokens_and_tells_pictures_apart() {
        let lim = Limits { max_images: 16, max_tokens: 16384, max_tokens_each: 300, max_bytes: 1 << 20 };
        let a = intake(&png(800, 600, 1), &lim).unwrap();
        assert_eq!((a.width, a.height, a.rows(), a.cols(), a.tokens()), (800, 600, 15, 20, 300));
        assert_eq!(a.rgb.len(), 800 * 600 * 3);
        // the same pixels again: the same number; other pixels of the same size: another number
        assert_eq!(intake(&png(800, 600, 1), &lim).unwrap().key, a.key);
        let b = intake(&png(800, 600, 2), &lim).unwrap();
        assert_ne!(b.key, a.key);
        assert!(is_image_id(a.id()) && a.id() != b.id());
        assert!(intake(b"GIF89a not a picture", &lim).is_err());
        assert!(intake(&png(64, 64, 1), &Limits { max_bytes: 10, ..lim }).unwrap_err().contains("INK_IMAGE_MB"));
    }

    #[test]
    fn fit_size_keeps_the_budget_and_the_aspect() {
        // within the budget: untouched
        assert_eq!(fit_size(2560, 1440, 2304), (2560, 1440));
        assert_eq!(fit_size(1920, 1080, 2304), (1920, 1080));
        assert_eq!(fit_size(800, 600, 300), (800, 600));
        assert_eq!(fit_size(92160, 40, 2304), (92160, 40));
        assert_eq!(fit_size(3840, 2160, usize::MAX), (3840, 2160));
        // above it
        assert_eq!(fit_size(3840, 2160, 2304), (2560, 1440));
        assert_eq!(fit_size(2160, 3840, 2304), (1440, 2560));
        assert_eq!(fit_size(7680, 4320, 2304), (2560, 1440));
        assert_eq!(fit_size(3840, 2160, 5184), (3840, 2160));
        assert_eq!(fit_size(7680, 4320, 5184), (3840, 2160));
        // the area rule alone gives 7010 x 525 = 176 x 14 = 2464 tokens: it steps down to 13 rows
        assert_eq!(fit_size(8000, 600, 2304), (6933, 520));
        assert_eq!(fit_size(801, 600, 300), (800, 599));
        assert_eq!(fit_size(16384, 2, 100), (4000, 1));
        assert_eq!(fit_size(5000, 5000, 0), (40, 40));
        let tokens = |w: usize, h: usize| w.div_ceil(CELL) * h.div_ceil(CELL);
        let mut x = 12345u64;
        let mut next = |m: u64| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 33) % m) as usize + 1
        };
        for _ in 0..20000 {
            let (w, h, budget) = (next(16384), next(16384), next(6000));
            let (nw, nh) = fit_size(w, h, budget);
            assert!(nw >= 1 && nh >= 1 && nw <= w && nh <= h, "{w} x {h} @ {budget}: {nw} x {nh}");
            assert!(tokens(nw, nh) <= budget, "{w} x {h} @ {budget}: {nw} x {nh}");
            if tokens(w, h) <= budget {
                assert_eq!((nw, nh), (w, h));
            } else {
                // one scale for both sides (each rounded down, at least 1 pixel)
                let (sw, sh) = (nw as f64 / w as f64, nh as f64 / h as f64);
                assert!((sw - sh).abs() <= 1.0 / w as f64 + 1.0 / h as f64 || nw == 1 || nh == 1, "{w} x {h} @ {budget}: {nw} x {nh}");
                // and it is not smaller than needed: the step before it (one more column or row of cells) was above the budget
                assert!((nw.div_ceil(CELL) + 1) * (nh.div_ceil(CELL) + 1) > budget, "{w} x {h} @ {budget}: {nw} x {nh}");
            }
        }
    }

    #[test]
    fn a_large_picture_is_shrunk_and_a_small_one_is_untouched() {
        let lim = Limits { max_images: 16, max_tokens: 16384, max_tokens_each: 300, max_bytes: 64 << 20 };
        // within the budget: the pixels are the decoder's, bit for bit
        let file = png(800, 600, 3);
        let a = intake(&file, &lim).unwrap();
        assert_eq!(a.rgb, crate::vision::decode(&file).unwrap().data);
        // 801 pixels wide is one more column (315 tokens): shrunk to fit, not refused
        let b = intake(&png(801, 600, 3), &lim).unwrap();
        assert_eq!((b.width, b.height, b.tokens()), (800, 599, 300));
        assert_eq!(b.rgb.len(), 800 * 599 * 3);
        assert_ne!(b.key, a.key);
        // the same large picture again: the same shrunk pixels, the same number
        let c = intake(&png(801, 600, 3), &lim).unwrap();
        assert_eq!((c.key, &c.rgb), (b.key, &b.rgb));
        // 3840 x 2160 at the default budget is 2560 x 1440
        let lim = Limits { max_tokens_each: 2304, ..lim };
        let d = intake(&png(3840, 2160, 3), &lim).unwrap();
        assert_eq!((d.width, d.height, d.rows(), d.cols(), d.tokens()), (2560, 1440, 36, 64, 2304));
        // the shrunk picture is the same picture: a smooth ramp stays the ramp (x % 251, y % 241 in the source)
        let px = |x: usize, y: usize| &d.rgb[(y * 2560 + x) * 3..][..3];
        assert!((px(60, 40)[0] as i32 - 90).abs() <= 2 && (px(60, 40)[1] as i32 - 60).abs() <= 2 && px(60, 40)[2] == 3, "{:?}", px(60, 40));
        // a raised budget takes it whole
        let e = intake(&png(3840, 2160, 3), &Limits { max_tokens_each: 5184, ..lim }).unwrap();
        assert_eq!((e.width, e.height, e.tokens()), (3840, 2160, 5184));
    }

    #[test]
    fn an_absurd_size_is_refused_from_the_header() {
        let lim = Limits { max_images: 16, max_tokens: 16384, max_tokens_each: 2304, max_bytes: 1 << 20 };
        let e = intake(&png(16385, 1, 0), &lim).unwrap_err();
        assert!(e.contains("16385 x 1") && e.contains("too large"), "{e}");
        assert!(intake(&png(1, 16385, 0), &lim).is_err());
        assert!(intake(&png(16384, 1, 0), &lim).is_ok());
        assert!(check_size(16384, 6103).is_ok());
        assert!(check_size(10001, 10000).is_err());
    }

    #[test]
    fn the_same_pixels_get_the_same_number() {
        let a = number_of(sha256(&[b"picture a"])).unwrap();
        let b = number_of(sha256(&[b"picture b"])).unwrap();
        assert_ne!(a, b);
        assert_eq!(number_of(sha256(&[b"picture a"])).unwrap(), a);
        assert!(is_image_id(IMG_ID_BASE | a) && !is_image_id(crate::vision::IMAGE_TOKEN));
    }
}
