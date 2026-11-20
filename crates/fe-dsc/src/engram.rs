//! Engram host side: n-gram row ids (DeepSeek's engram.py rule, as the earlier fe-engram port) with every parameter
//! read from the GGUF metadata, and the table reader (plain pread of 144-byte IQ4_NL rows from the Engram GGUF).
use crate::config::DscConfig;
use anyhow::{ensure, Context, Result};
use fe_gguf::Gguf;
use std::os::unix::fs::FileExt;

pub struct Engram {
    pub layers: Vec<usize>,
    pub cols: usize,
    pub heads: usize,
    pub ngram: usize,
    pub dim: usize,
    mult: Vec<[i64; 4]>,
    primes: Vec<Vec<i64>>,
    /// row offset of each column inside its layer's table
    offs: Vec<Vec<i64>>,
    map: Vec<u32>,
    pad: i64,
    /// per layer: file and absolute byte offset of the table
    tables: Vec<(std::fs::File, u64)>,
}

pub const ROW_BYTES: usize = 144;

impl Engram {
    pub fn new(g: &Gguf, c: &DscConfig) -> Result<Engram> {
        ensure!(c.ngram == 4, "engram multipliers: ngram {} unsupported", c.ngram);
        let cols = c.engram_cols();
        let mut mult = vec![];
        let (mut primes, mut offs, mut tables) = (vec![], vec![], vec![]);
        for (li, &l) in c.engram_layers.iter().enumerate() {
            let m = &c.ple_mult[li * 4..li * 4 + 4];
            mult.push([m[0] as i64, m[1] as i64, m[2] as i64, m[3] as i64]);
            let p: Vec<i64> = c.head_vocab[li * cols..(li + 1) * cols].iter().map(|&x| x as i64).collect();
            let base = c.head_off[li * cols] as i64;
            offs.push(c.head_off[li * cols..(li + 1) * cols].iter().map(|&o| o as i64 - base).collect::<Vec<_>>());
            ensure!(c.head_off[li * cols] == if li == 0 { 0 } else { c.head_off[li * cols - 1] + c.head_vocab[li * cols - 1] }, "head offsets not contiguous");
            primes.push(p);
            let t = g.tensor(&crate::tensors::engram_table_name(l)).with_context(|| format!("engram table of layer {l}"))?;
            let f = std::fs::File::open(&g.shards[t.shard].path)?;
            tables.push((f, t.offset));
        }
        let pad = c.token_map[c.engram_pad as usize] as i64;
        Ok(Engram { layers: c.engram_layers.clone(), cols, heads: c.heads_per_ngram, ngram: c.ngram, dim: c.ple_dim,
            mult, primes, offs, map: c.token_map.clone(), pad, tables })
    }

    fn comp(&self, id: i64) -> i64 {
        if id < 0 { -1 } else { self.map.get(id as usize).map_or(-1, |&m| m as i64) }
    }

    /// Row ids (local to layer index `li`'s table) of position `pos` of `ids`.
    pub fn hashes(&self, ids: &[u32], pos: usize, li: usize) -> Vec<i64> {
        let mut toks = [0i64; 4];
        let mut blocked = false;
        for (shift, t) in toks.iter_mut().enumerate().take(self.ngram) {
            let src = if pos >= shift { self.comp(ids[pos - shift] as i64) } else { -1 };
            blocked = blocked || pos < shift || src < 0;
            *t = if blocked { self.pad } else { src };
        }
        let mut out = vec![0i64; self.cols];
        let mut rolling = toks[0].wrapping_mul(self.mult[li][0]);
        for i in 1..self.ngram {
            rolling ^= toks[i].wrapping_mul(self.mult[li][i]);
            for k in 0..self.heads {
                let c = (i - 1) * self.heads + k;
                out[c] = rolling.rem_euclid(self.primes[li][c]) + self.offs[li][c];
            }
        }
        out
    }

    /// The Engram vector (cols x dim f32, concatenated) of position `pos` for layer index `li`.
    pub fn vector(&self, ids: &[u32], pos: usize, li: usize) -> Result<Vec<f32>> {
        let rows = self.hashes(ids, pos, li);
        let (f, base) = &self.tables[li];
        let mut out = vec![0f32; self.cols * self.dim];
        let mut buf = [0u8; ROW_BYTES];
        for (c, &r) in rows.iter().enumerate() {
            f.read_exact_at(&mut buf, base + r as u64 * ROW_BYTES as u64)?;
            fe_ggq::refq::dequant(fe_gguf::GgmlType::IQ4NL, &buf, &mut out[c * self.dim..(c + 1) * self.dim]);
        }
        Ok(out)
    }
}
