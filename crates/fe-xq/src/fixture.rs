//! Reader for arc-engine's XQ oracle fixtures (`tools/make_xq_fixtures.py`, numpy): random XQ v0 tensors with the numpy emulation of
//! the rotation prepass (bit-exact target), the float64 product (tolerance target) and the device layout packed by an independent
//! implementation of the layout (byte-exact target).
//!
//! File: int32 header `[magic 0x31515858, bits, group, rot, N, K, M]`, then q int8 [N*K], s fp16 [N*K/g], su fp16 [K], x bf16 [M*K],
//! xt fp16 [M*K] (rotated rows, row-major), y float64 [M*N], packed uint32 [N/16 * K/g * (16 D + 8)].
use crate::pack::packed_dwords;
use anyhow::{ensure, Context as _, Result};
use std::path::Path;

pub struct XqFixture {
    pub bits: u32,
    pub g: u32,
    pub rot: bool,
    pub n: usize,
    pub k: usize,
    pub m: usize,
    pub q: Vec<i8>,
    pub s: Vec<u16>,
    pub su: Vec<u16>,
    pub x: Vec<u16>,
    pub xt: Vec<u16>,
    pub y: Vec<f64>,
    pub packed: Vec<u32>,
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

impl XqFixture {
    pub fn read(path: &Path) -> Result<XqFixture> {
        let raw = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
        ensure!(raw.len() >= 28, "{path:?}: short");
        let h: Vec<i32> = raw[..28].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        ensure!(h[0] == 0x3151_5858, "{path:?}: bad magic");
        let (bits, g, rot) = (h[1] as u32, h[2] as u32, h[3] != 0);
        let (n, k, m) = (h[4] as usize, h[5] as usize, h[6] as usize);
        let sizes = [n * k, n * (k / g as usize) * 2, k * 2, m * k * 2, m * k * 2, m * n * 8, packed_dwords(bits, g, n, k) * 4];
        ensure!(28 + sizes.iter().sum::<usize>() <= raw.len(), "{path:?}: truncated");
        let mut parts = Vec::new();
        let mut off = 28;
        for b in sizes {
            parts.push(&raw[off..off + b]);
            off += b;
        }
        let q = parts[0].iter().map(|&b| b as i8).collect();
        let s = u16s(parts[1]);
        let su = u16s(parts[2]);
        let x = u16s(parts[3]);
        let xt = u16s(parts[4]);
        let y = parts[5].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        let packed = parts[6].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(XqFixture { bits, g, rot, n, k, m, q, s, su, x, xt, y, packed })
    }
}

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.o + n <= self.b.len(), "dump truncated at byte {} (+{n})", self.o);
        self.o += n;
        Ok(&self.b[self.o - n..self.o])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// The XQ decode-matvec oracle on a real checkpoint tensor: `xqmv/<name>.bin` of arc-engine's `xpu-xq_oracle-test` (ferrite-xe oracle tool):
/// `"XQOR0001"`, u32 bits, group, rot, n, k, split, nm, u32 m[nm], u64 seed, x bf16 [16][k], then per m: xt fp16 [NR][K/16][MW][16],
/// xs fp32 [NR][K/32][MW], y fp32 [m][n], y bf16 [m][n].
pub struct XqMvDump {
    pub bits: u32,
    pub group: u32,
    pub rot: bool,
    pub n: usize,
    pub k: usize,
    pub split: u32,
    pub seed: u64,
    /// bf16 [16][k]
    pub x: Vec<u16>,
    pub cases: Vec<XqMvCase>,
}

pub struct XqMvCase {
    pub m: u32,
    /// fp16 [NR][K/16][MW][16] (device layout)
    pub xt: Vec<u16>,
    /// fp32 [NR][K/32][MW]
    pub xs: Vec<f32>,
    /// [m][n]
    pub y32: Vec<f32>,
    pub y16: Vec<u16>,
}

impl XqMvDump {
    pub fn read(path: &Path) -> Result<XqMvDump> {
        let b = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
        let mut r = Rd { b: &b, o: 0 };
        ensure!(r.take(8)? == b"XQOR0001", "{path:?}: not an xq oracle dump");
        let (bits, group, rot, n, k, split, nm) = (r.u32()?, r.u32()?, r.u32()? != 0, r.u32()? as usize, r.u32()? as usize, r.u32()?, r.u32()?);
        let ms: Vec<u32> = (0..nm).map(|_| r.u32()).collect::<Result<_>>()?;
        let seed = r.u64()?;
        let x = u16s(r.take(16 * k * 2)?);
        let mut cases = Vec::new();
        for m in ms {
            let (mw, nr) = crate::pack::VARIANTS[crate::pack::variant_for(m)?];
            let (mw, nr) = (mw as usize, nr as usize);
            let xt = u16s(r.take(nr * k * mw * 2)?);
            let xs = f32s(r.take(nr * (k / 32) * mw * 4)?);
            let y32 = f32s(r.take(m as usize * n * 4)?);
            let y16 = u16s(r.take(m as usize * n * 2)?);
            cases.push(XqMvCase { m, xt, xs, y32, y16 });
        }
        ensure!(r.o == b.len(), "{path:?}: {} trailing bytes", b.len() - r.o);
        Ok(XqMvDump { bits, group, rot, n, k, split, seed, x, cases })
    }
}
