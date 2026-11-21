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
    /// reader threads for `vectors` (DSC_ENGRAM_THREADS, default 128): they block in pread almost all the time, so the pool size is
    /// the queue depth the NVMe sees, not a CPU count
    pool: rayon::ThreadPool,
}

pub const ROW_BYTES: usize = 144;

/// One 144-byte row at `off` if it is in the page cache: preadv2 with RWF_NOWAIT returns at once, with the row or without it
/// (EAGAIN, or a short read when the row spans a page that is not cached); false means read it the ordinary way.
fn read_cached(f: &std::fs::File, buf: &mut [u8; ROW_BYTES], off: u64) -> bool {
    use std::os::fd::AsRawFd;
    let iov = libc::iovec { iov_base: buf.as_mut_ptr() as *mut libc::c_void, iov_len: ROW_BYTES };
    let n = unsafe { libc::preadv2(f.as_raw_fd(), &iov, 1, off as libc::off_t, libc::RWF_NOWAIT) };
    n == ROW_BYTES as isize
}

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
        let nthreads = std::env::var("DSC_ENGRAM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(128usize).max(1);
        // the engine thread may be pinned (CpuMoe::set_caller_cpu) and pool threads inherit its mask: give the readers every online CPU
        let ncpu = (unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) }.max(1) as usize).min(libc::CPU_SETSIZE as usize);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nthreads).thread_name(|i| format!("engram-{i}")).start_handler(move |_| unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for cpu in 0..ncpu {
                libc::CPU_SET(cpu, &mut set);
            }
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }).build()?;
        Ok(Engram { layers: c.engram_layers.clone(), cols, heads: c.heads_per_ngram, ngram: c.ngram, dim: c.ple_dim,
            mult, primes, offs, map: c.token_map.clone(), pad, tables, pool })
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

    /// The Engram vectors of several rows for every Engram layer at once: `reqs` is (history, position) a row, `out` gets them
    /// layer-major, [layer][row][cols x dim] f32, the same values as `vector` row by row and layer by layer. The rows' 144-byte
    /// reads are independent random reads of a 110 GB file (a cold one is one NVMe read, ~70 us; 24 a row and Engram layer), so
    /// they are all issued together on the reader pool instead of one after another.
    pub fn vectors(&self, reqs: &[(&[u32], usize)], out: &mut [f32]) -> Result<()> {
        use rayon::prelude::*;
        let nl = self.tables.len();
        ensure!(out.len() == nl * reqs.len() * self.cols * self.dim, "engram vectors: out {} != {} layers x {} rows x {}", out.len(), nl, reqs.len(), self.cols * self.dim);
        let mut at: Vec<(usize, u64)> = Vec::with_capacity(nl * reqs.len() * self.cols);
        for li in 0..nl {
            for &(ids, pos) in reqs {
                for r in self.hashes(ids, pos, li) {
                    at.push((li, self.tables[li].1 + r as u64 * ROW_BYTES as u64));
                }
            }
        }
        // rows already in the page cache are read here, without waiting (preadv2 RWF_NOWAIT: a full row or nothing); only the
        // others go to the reader pool, so a warm pass costs no thread wake-ups (each one cost more than the whole cached read)
        let tables = &self.tables;
        let mut misses: Vec<(&mut [f32], usize, u64)> = vec![];
        let mut buf = [0u8; ROW_BYTES];
        for (o, &(li, off)) in out.chunks_mut(self.dim).zip(at.iter()) {
            if read_cached(&tables[li].0, &mut buf, off) {
                fe_ggq::refq::dequant(fe_gguf::GgmlType::IQ4NL, &buf, o);
            } else {
                misses.push((o, li, off));
            }
        }
        if misses.is_empty() {
            return Ok(());
        }
        self.pool.install(|| {
            misses.par_iter_mut().with_max_len(1).try_for_each(|(o, li, off)| -> Result<()> {
                let mut buf = [0u8; ROW_BYTES];
                tables[*li].0.read_exact_at(&mut buf, *off)?;
                fe_ggq::refq::dequant(fe_gguf::GgmlType::IQ4NL, &buf, o);
                Ok(())
            })
        })
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
