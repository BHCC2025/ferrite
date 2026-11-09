//! The EXL3 tensor format (exllamav3 1.6, mul1 codebook, integer bit widths). Format and decode after exllamav3
//! (MIT, Copyright (c) 2025 Turboderp; see NOTICE).
//!
//! A linear `y = x @ W` (W is [in = k, out = n], the transpose of the HF `weight`) is stored as
//! `W = diag(suh) . H_k . W_hat . H_n . diag(svh)` where H is block-diagonal of 128x128 Sylvester Hadamards
//! scaled by 1/sqrt(128), suh / svh are fp16 vectors (signs times scales), and W_hat is a trellis-coded fp16
//! matrix: `trellis` int16 [k/16, n/16, 16 * bits], one 16x16 tile per (k/16, n/16) cell, 256 values of a tile
//! in tensor-core fragment order (see `value_row` / `value_col`).

use crate::st::Checkpoint;
use crate::{Error, Result};

/// The mul1 codebook's multiplier (`.mul1` scalar in the checkpoint).
pub const MUL1: u32 = 0x83DC_D12D;

#[derive(Clone)]
pub struct Exl3Tensor {
    pub name: String,
    /// in_features (rows of W, multiple of 128 here).
    pub k: usize,
    /// out_features (columns of W).
    pub n: usize,
    pub bits: u32,
    /// Trellis as little-endian u32 words: tile (kt, nt) is words `[(kt * n/16 + nt) * 8 * bits ..][.. 8 * bits]`.
    pub trellis: Vec<u32>,
    /// fp16 bits [k]
    pub suh: Vec<u16>,
    /// fp16 bits [n]
    pub svh: Vec<u16>,
}

impl Exl3Tensor {
    /// Load `base` (e.g. `model.language_model.layers.0.mlp.up_proj`); bits come from the trellis shape.
    pub fn load(ck: &Checkpoint, base: &str) -> Result<Self> {
        let tn = format!("{base}.trellis");
        let info = ck.info(&tn)?.clone();
        if info.shape.len() != 3 || info.shape[2] % 16 != 0 {
            return Err(Error::msg(format!("{tn}: shape {:?} (half-integer bit widths unsupported)", info.shape)));
        }
        let (kt, nt, w) = (info.shape[0], info.shape[1], info.shape[2]);
        let bits = (w / 16) as u32;
        if !(1..=8).contains(&bits) {
            return Err(Error::msg(format!("{tn}: {bits} bits")));
        }
        let mul = ck.read_u32(&format!("{base}.mul1"))?;
        if mul.first() != Some(&MUL1) {
            return Err(Error::msg(format!("{base}: codebook is not mul1 ({mul:?})")));
        }
        let trellis = ck.read_u32(&tn)?;
        let suh = ck.read_u16(&format!("{base}.suh"))?;
        let svh = ck.read_u16(&format!("{base}.svh"))?;
        let (k, n) = (kt * 16, nt * 16);
        if suh.len() != k || svh.len() != n {
            return Err(Error::msg(format!("{base}: suh {} svh {} for {k}x{n}", suh.len(), svh.len())));
        }
        Ok(Self { name: base.to_string(), k, n, bits, trellis, suh, svh })
    }

    /// u32 words per 16x16 tile.
    pub fn tile_words(&self) -> usize {
        8 * self.bits as usize
    }

    pub fn tile(&self, kt: usize, nt: usize) -> &[u32] {
        let tw = self.tile_words();
        let i = (kt * (self.n / 16) + nt) * tw;
        &self.trellis[i..i + tw]
    }

    /// The trellis in the "strips" layout [n/128][k/16][8][tw]: each 128-column block's whole K range is one
    /// contiguous stream (decode reads long sequential runs instead of 256*bits-byte pieces n/16 tiles apart).
    pub fn strips(&self) -> Vec<u32> {
        let tw = self.tile_words();
        let (kt, nt) = (self.k / 16, self.n / 16);
        let mut out = vec![0u32; self.trellis.len()];
        for b in 0..nt / 8 {
            for t in 0..kt {
                let dst = (b * kt + t) * 8 * tw;
                let src = (t * nt + b * 8) * tw;
                out[dst..dst + 8 * tw].copy_from_slice(&self.trellis[src..src + 8 * tw]);
            }
        }
        out
    }

    /// Tensor-parallel column slice: the concatenation of output-column ranges `[n0, n1)` (each 128-aligned:
    /// the output Hadamard works on 128-column blocks, so a slice is an exact EXL3 tensor of its own).
    /// `y_slice = x @ W[:, ranges]`.
    pub fn slice_cols(&self, ranges: &[(usize, usize)]) -> Result<Self> {
        let tw = self.tile_words();
        let nt = self.n / 16;
        let mut cols = 0;
        for &(a, b) in ranges {
            if a % 128 != 0 || b % 128 != 0 || a >= b || b > self.n {
                return Err(Error::msg(format!("{}: column range {a}..{b} not 128-aligned in 0..{}", self.name, self.n)));
            }
            cols += b - a;
        }
        let mut trellis = Vec::with_capacity(self.k / 16 * cols / 16 * tw);
        for kt in 0..self.k / 16 {
            for &(a, b) in ranges {
                trellis.extend_from_slice(&self.trellis[(kt * nt + a / 16) * tw..(kt * nt + b / 16) * tw]);
            }
        }
        let mut svh = Vec::with_capacity(cols);
        for &(a, b) in ranges {
            svh.extend_from_slice(&self.svh[a..b]);
        }
        Ok(Self { name: format!("{}[cols]", self.name), k: self.k, n: cols, bits: self.bits, trellis, suh: self.suh.clone(), svh })
    }

    /// Tensor-parallel row (input) slice `[k0, k1)`, 128-aligned: `y_partial = x[:, k0..k1] @ W[k0..k1, :]`, and
    /// the partials of all ranks sum to the full output (the output rotation is linear, so each partial is
    /// already in the original basis).
    pub fn slice_rows(&self, k0: usize, k1: usize) -> Result<Self> {
        if k0 % 128 != 0 || k1 % 128 != 0 || k0 >= k1 || k1 > self.k {
            return Err(Error::msg(format!("{}: row range {k0}..{k1} not 128-aligned in 0..{}", self.name, self.k)));
        }
        let row_words = self.n / 16 * self.tile_words();
        let trellis = self.trellis[k0 / 16 * row_words..k1 / 16 * row_words].to_vec();
        Ok(Self {
            name: format!("{}[rows {k0}..{k1}]", self.name),
            k: k1 - k0,
            n: self.n,
            bits: self.bits,
            trellis,
            suh: self.suh[k0..k1].to_vec(),
            svh: self.svh.clone(),
        })
    }

    /// Trellis bytes (the decode working set).
    pub fn trellis_bytes(&self) -> usize {
        self.trellis.len() * 4
    }
}

/// Every EXL3 linear base name in a checkpoint (tensors with a `.trellis`), sorted.
pub fn exl3_bases(ck: &Checkpoint) -> Vec<String> {
    let mut v: Vec<String> =
        ck.tensors.keys().filter_map(|k| k.strip_suffix(".trellis").map(|s| s.to_string())).collect();
    v.sort();
    v
}

/// Row (k within the tile) of value `j` (0..8) decoded by lane `lane` (0..32): the mma B-fragment layout.
#[inline(always)]
pub const fn value_row(lane: usize, j: usize) -> usize {
    2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1)
}

/// Column (n within the tile) of value `j` decoded by lane `lane`.
#[inline(always)]
pub const fn value_col(lane: usize, j: usize) -> usize {
    (lane >> 2) + 8 * (j >> 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(k: usize, n: usize, bits: u32) -> Exl3Tensor {
        let tw = 8 * bits as usize;
        let trellis = (0..k / 16 * n / 16 * tw).map(|i| (i as u32).wrapping_mul(2654435761)).collect();
        let suh = (0..k).map(|i| 0x3c00 | (i as u16 & 0x3ff)).collect();
        let svh = (0..n).map(|i| 0xbc00 | (i as u16 & 0x3ff)).collect();
        Exl3Tensor { name: "t".into(), k, n, bits, trellis, suh, svh }
    }

    #[test]
    fn slices_keep_tiles() {
        let t = synthetic(256, 512, 3);
        let c = t.slice_cols(&[(128, 256), (384, 512)]).unwrap();
        assert_eq!((c.k, c.n), (256, 256));
        for kt in 0..16 {
            assert_eq!(c.tile(kt, 0), t.tile(kt, 8));
            assert_eq!(c.tile(kt, 8), t.tile(kt, 24));
        }
        assert_eq!(&c.svh[..128], &t.svh[128..256]);
        let r = t.slice_rows(128, 256).unwrap();
        assert_eq!((r.k, r.n), (128, 512));
        assert_eq!(r.tile(0, 5), t.tile(8, 5));
        assert_eq!(&r.suh[..], &t.suh[128..256]);
        assert!(t.slice_cols(&[(0, 100)]).is_err());
        assert!(t.slice_rows(64, 128).is_err());
    }

    #[test]
    fn strips_layout() {
        let t = synthetic(256, 256, 2);
        let s = t.strips();
        let tw = t.tile_words();
        // strips [n/128][k/16][8][tw]: block b, k tile kt, tile j = stored tile (kt, 8b + j)
        for b in 0..2 {
            for kt in 0..16 {
                for j in 0..8 {
                    let at = ((b * 16 + kt) * 8 + j) * tw;
                    assert_eq!(&s[at..at + tw], t.tile(kt, 8 * b + j));
                }
            }
        }
    }
}
