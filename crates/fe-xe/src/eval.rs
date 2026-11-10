//! The engine's evaluation outputs, independent of the device: Ferrite's `logits` contract and arc-engine's fpkl row records.
//!
//! - `logits`: `<bin> <model> logits <ids.txt> <out.bin> <skip> <rows>` -> bf16 logits `[nseq][rows][vocab]` (positions skip..skip+rows-1 of
//!   every sequence; the row at position t is the distribution after ids[0..=t], predicting ids[t + 1]). Scored by
//!   `tools/kl.sh`'s scorer.
//! - `fpkl`: `<bin> <model> fpkl <IN.bin> <OUTDIR> [--rows A:B]` with arc-engine's inputs (`FPKLIN01`) -> `OUTDIR/row_<n>.bin`, the exact record
//!   of arc-engine's `xpu-qwen_fpkl-test` (`rowStats`): `[lse, label logit, argmax (u32 bits), nonfinite count (u32 bits), logits at the
//!   reference's K ids]` per position after a 4-word header `[row, len, K, window]`. Scored by a script outside this repository
//!   (all positions of the five board sets, paired with arc-engine's own rows).
use anyhow::{bail, ensure, Context as _, Result};
use std::io::Write;
use std::path::Path;

/// Token sequences of a `logits` ids file: one sequence a line, ids separated by commas (or spaces).
pub fn read_ids(path: &Path) -> Result<Vec<Vec<u32>>> {
    let txt = std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?;
    let mut out = Vec::new();
    for (i, line) in txt.lines().enumerate() {
        let v: Result<Vec<u32>> = line
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|t| !t.is_empty())
            .map(|t| t.parse::<u32>().with_context(|| format!("{path:?} line {}: {t:?}", i + 1)))
            .collect();
        let v = v?;
        if !v.is_empty() {
            out.push(v);
        }
    }
    Ok(out)
}

/// fp32 -> bf16 bits, round to nearest even (the device's `to_bf`).
pub fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Appends logits rows (fp32 from the head) as the contract's bf16 rows.
pub struct LogitsWriter {
    f: std::io::BufWriter<std::fs::File>,
    buf: Vec<u8>,
    pub rows: usize,
}

impl LogitsWriter {
    pub fn create(path: &Path) -> Result<Self> {
        let f = std::fs::File::create(path).with_context(|| format!("creating {path:?}"))?;
        Ok(Self { f: std::io::BufWriter::with_capacity(8 << 20, f), buf: Vec::new(), rows: 0 })
    }
    pub fn row(&mut self, logits: &[f32]) -> Result<()> {
        self.buf.clear();
        self.buf.extend(logits.iter().flat_map(|&v| to_bf16(v).to_le_bytes()));
        self.f.write_all(&self.buf)?;
        self.rows += 1;
        Ok(())
    }
    pub fn finish(mut self) -> Result<()> {
        self.f.flush()?;
        Ok(())
    }
}

/// arc-engine's fpkl input set (`FPKLIN01`).
pub struct FpklIn {
    pub n: usize,
    pub l: usize,
    pub k: usize,
    pub vocab: usize,
    /// scored positions per row
    pub lens: Vec<u32>,
    /// [n][l]
    pub ids: Vec<i32>,
    /// [n][l - 1][k]
    pub ref_idx: Vec<i32>,
}

impl FpklIn {
    pub fn read(path: &Path) -> Result<FpklIn> {
        let b = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
        ensure!(b.len() >= 24 && &b[..8] == b"FPKLIN01", "{path:?}: not an FPKLIN01 file");
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
        let (n, l, k, vocab) = (u(8), u(12), u(16), u(20));
        let need = 24 + 4 * n + 4 * n * l + 4 * n * (l - 1) * k;
        ensure!(b.len() == need, "{path:?}: {} bytes, want {need}", b.len());
        let words = |o: usize, cnt: usize| -> Vec<u32> { (0..cnt).map(|i| u(o + 4 * i) as u32).collect() };
        let lens = words(24, n);
        let ids = words(24 + 4 * n, n * l).into_iter().map(|v| v as i32).collect();
        let ref_idx = words(24 + 4 * n + 4 * n * l, n * (l - 1) * k).into_iter().map(|v| v as i32).collect();
        Ok(FpklIn { n, l, k, vocab, lens, ids, ref_idx })
    }

    pub fn row_ids(&self, n: usize) -> &[i32] {
        &self.ids[n * self.l..(n + 1) * self.l]
    }

    pub fn ref_ids(&self, n: usize, t: usize) -> &[i32] {
        let o = (n * (self.l - 1) + t) * self.k;
        &self.ref_idx[o..o + self.k]
    }
}

/// One fpkl record from a logits row (`rowStats` of qwen_fpkl_test.zig): max over finite values, sum of exp(v - max) with fp32 exp
/// accumulated in fp64 (16 lanes, then the lanes added, then the tail), lse = max + ln(sum).
pub fn fpkl_record(lg: &[f32], ref_idx: &[i32], label: usize, rec: &mut [f32]) -> u32 {
    let mut mx = lg[0];
    let mut imax = 0usize;
    let mut bad = 0u32;
    for (k, &v) in lg.iter().enumerate() {
        if !v.is_finite() {
            bad += 1;
            continue;
        }
        if v > mx {
            mx = v;
            imax = k;
        }
    }
    let mut acc = [0f64; 16];
    let chunks = lg.len() / 16;
    for c in 0..chunks {
        for (j, a) in acc.iter_mut().enumerate() {
            *a += (lg[c * 16 + j] - mx).exp() as f64;
        }
    }
    let mut se: f64 = acc.iter().sum();
    for &v in &lg[chunks * 16..] {
        se += ((v - mx) as f64).exp();
    }
    rec[0] = (mx as f64 + se.ln()) as f32;
    rec[1] = lg[label];
    rec[2] = f32::from_bits(imax as u32);
    rec[3] = f32::from_bits(bad);
    for (j, &id) in ref_idx.iter().enumerate() {
        rec[4 + j] = lg[id as usize];
    }
    bad
}

/// Writes `OUTDIR/row_<n>.bin` atomically (tmp + rename): header [row, len, K, window] then `recs`.
pub fn write_fpkl_row(dir: &Path, n: usize, len: usize, k: usize, window: u32, recs: &[f32]) -> Result<()> {
    if recs.len() != len * (4 + k) {
        bail!("fpkl row {n}: {} floats for {len} records of {}", recs.len(), 4 + k);
    }
    std::fs::create_dir_all(dir)?;
    let mut bytes = Vec::with_capacity(16 + recs.len() * 4);
    for w in [n as u32, len as u32, k as u32, window] {
        bytes.extend(w.to_le_bytes());
    }
    for v in recs {
        bytes.extend(v.to_le_bytes());
    }
    let p = dir.join(format!("row_{n}.bin"));
    let tmp = dir.join(format!("row_{n}.bin.tmp"));
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &p)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_matches_direct_lse() {
        let lg: Vec<f32> = (0..1000).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.1).collect();
        let idx: Vec<i32> = (0..8).map(|i| i * 3).collect();
        let mut rec = vec![0f32; 12];
        assert_eq!(fpkl_record(&lg, &idx, 5, &mut rec), 0);
        let mx = lg.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let lse = mx + lg.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
        assert!((rec[0] as f64 - lse).abs() < 1e-5);
        assert_eq!(rec[1], lg[5]);
        assert_eq!(rec[4 + 2], lg[6]);
    }

    #[test]
    fn bf16_rounding() {
        assert_eq!(to_bf16(1.0), 0x3f80);
        assert_eq!(to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80); // tie to even
        assert_eq!(to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    }
}
