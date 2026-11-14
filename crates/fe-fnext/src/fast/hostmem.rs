//! Host page-cache hygiene for running under a memory cap (the 32 GB box): the GGUF is only mmapped, so every byte
//! read stays in the page cache, charged to the process's cgroup. Weights that live on the card are dropped
//! from the cache after upload; the experts (read by CPU misses and uploads) are read ahead at start; PLE table
//! rows (16 random pages a token from a 27.5 GB table) are dropped after use so they never crowd out experts.
use fe_gguf::{Gguf, TensorInfo};
use std::os::fd::AsRawFd;

pub(crate) struct HostMem {
    files: Vec<std::fs::File>,
    page: usize,
    pub on: bool,
}

impl HostMem {
    pub fn new(g: &Gguf) -> HostMem {
        let files = g.shards.iter().filter_map(|s| std::fs::File::open(&s.path).ok()).collect();
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let on = std::env::var("FNEXT_PAGECACHE").map_or(true, |v| v != "0");
        HostMem { files, page, on }
    }

    /// Drop a byte range of a tensor from this process's mapping and from the page cache.
    pub fn drop_range(&self, g: &Gguf, t: &TensorInfo, off: usize, len: usize) {
        if !self.on || len == 0 {
            return;
        }
        let data = g.data(t);
        let a = data.as_ptr() as usize + off;
        let a0 = a & !(self.page - 1);
        let a1 = (a + len).next_multiple_of(self.page);
        // only whole pages inside the tensor: a page shared with a neighbour may still be in use
        let (lo, hi) = (data.as_ptr() as usize, data.as_ptr() as usize + data.len());
        let (a0, a1) = (if a0 < lo { a0 + self.page } else { a0 }, if a1 > hi { a1 - self.page } else { a1 });
        if a1 <= a0 {
            return;
        }
        unsafe {
            libc::madvise(a0 as *mut libc::c_void, a1 - a0, libc::MADV_DONTNEED);
            if let Some(f) = self.files.get(t.shard) {
                let foff = t.offset as usize + (a0 - lo);
                libc::posix_fadvise(f.as_raw_fd(), foff as i64, (a1 - a0) as i64, libc::POSIX_FADV_DONTNEED);
            }
        }
    }

    pub fn drop_tensor(&self, g: &Gguf, name: &str) {
        if let Some(t) = g.tensor(name) {
            self.drop_range(g, t, 0, t.bytes as usize);
        }
    }

    /// Ask the kernel to read a byte range of a tensor ahead (async).
    pub fn willneed_range(data: &[u8], off: usize, len: usize, page: usize) {
        let a0 = (data.as_ptr() as usize + off) & !(page - 1);
        let a1 = (data.as_ptr() as usize + off + len).next_multiple_of(page);
        unsafe {
            libc::madvise(a0 as *mut libc::c_void, a1 - a0, libc::MADV_WILLNEED);
        }
    }

    pub fn page(&self) -> usize {
        self.page
    }

    /// Ask the kernel to read a tensor ahead (async).
    pub fn willneed(&self, g: &Gguf, t: &TensorInfo) {
        if !self.on {
            return;
        }
        let data = g.data(t);
        let a0 = data.as_ptr() as usize & !(self.page - 1);
        let a1 = (data.as_ptr() as usize + data.len()).next_multiple_of(self.page);
        unsafe {
            libc::madvise(a0 as *mut libc::c_void, a1 - a0, libc::MADV_WILLNEED);
        }
    }
}

/// PLE rows `idx` of the table `pt` (row-major [rows][ne0], any GGUF row type), dequantized to f32, read with O_DIRECT
/// preads from many threads (prompt groups: ~170K distinct random rows per 16K tokens; the page-fault path keeps far
/// fewer reads in flight than the NVMe takes). Nothing enters the page cache. Falls back to the mmap when the shard
/// cannot be opened with O_DIRECT. FNEXT_PLE_IO=mmap = the old path, FNEXT_PLE_IO_THREADS (default 96).
pub fn ple_gather(g: &Gguf, pt: &TensorInfo, idx: &[u32], page: usize) -> Vec<f32> {
    use rayon::prelude::*;
    use std::os::unix::fs::{FileExt, OpenOptionsExt};
    let (prb, pdd) = (pt.row_bytes(), pt.ne0());
    let t0 = std::time::Instant::now();
    let mut uniq = idx.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    let mut rows = vec![0f32; uniq.len() * pdd];
    let t_sort = t0.elapsed().as_secs_f64();
    let direct = !ple_mmap();
    let buffered = std::env::var("FNEXT_PLE_IO").as_deref() == Ok("buffered");
    let f = if direct {
        let mut o = std::fs::OpenOptions::new();
        o.read(true);
        if !buffered {
            o.custom_flags(libc::O_DIRECT);
        }
        let f = o.open(&g.shards[pt.shard].path).ok();
        if let (Some(f), true) = (&f, buffered) {
            use std::os::fd::AsRawFd;
            unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
        }
        f
    } else {
        None
    };
    match f {
        Some(f) => {
            const AL: usize = 4096;
            let nt: usize = std::env::var("FNEXT_PLE_IO_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(96).max(1);
            let next = std::sync::atomic::AtomicUsize::new(0);
            let failed = std::sync::atomic::AtomicBool::new(false);
            let out = rows.as_mut_ptr() as usize;
            let span = (prb + AL - 1) / AL * AL + AL;
            std::thread::scope(|s| {
                for _ in 0..nt {
                    s.spawn(|| {
                        crate::cores::unpin_current();
                        // one descriptor per thread
                        let mut o = std::fs::OpenOptions::new();
                        o.read(true);
                        if !buffered {
                            o.custom_flags(libc::O_DIRECT);
                        }
                        let fo = o.open(&g.shards[pt.shard].path).ok();
                        let f = fo.as_ref().unwrap_or(&f);
                        let mut raw = vec![0u8; span + AL];
                        let a = (AL - raw.as_ptr() as usize % AL) % AL;
                        loop {
                            let i0 = next.fetch_add(32, std::sync::atomic::Ordering::Relaxed);
                            if i0 >= uniq.len() {
                                break;
                            }
                            for i in i0..(i0 + 32).min(uniq.len()) {
                                let off = pt.offset as usize + uniq[i] as usize * prb;
                                let (p0, len) = if buffered { (off, prb) } else { (off / AL * AL, (off + prb).div_ceil(AL) * AL - off / AL * AL) };
                                let buf = &mut raw[a..a + len];
                                let mut got = 0;
                                while got < len {
                                    match f.read_at(&mut buf[got..], (p0 + got) as u64) {
                                        Ok(0) | Err(_) => break,
                                        Ok(n) => got += n,
                                    }
                                }
                                if got < off - p0 + prb {
                                    failed.store(true, std::sync::atomic::Ordering::Relaxed);
                                    return;
                                }
                                let o = unsafe { std::slice::from_raw_parts_mut((out as *mut f32).add(i * pdd), pdd) };
                                fe_ggq::refq::dequant(pt.ty, &buf[off - p0..off - p0 + prb], o);
                            }
                        }
                    });
                }
            });
            if failed.load(std::sync::atomic::Ordering::Relaxed) {
                eprintln!("fnext-fast: PLE O_DIRECT read failed, mmap path");
                return ple_gather_mmap(g, pt, idx, page);
            }
        }
        None => {
            if direct {
                eprintln!("fnext-fast: PLE shard O_DIRECT open failed, mmap path");
            }
            return ple_gather_mmap(g, pt, idx, page);
        }
    }
    let t_read = t0.elapsed().as_secs_f64();
    let mut ple = vec![0f32; idx.len() * pdd];
    let _t = PleT(t0, t_sort, t_read);
    ple.par_chunks_mut(pdd).zip(idx.par_iter()).for_each(|(o, rw)| {
        let i = uniq.binary_search(rw).unwrap();
        o.copy_from_slice(&rows[i * pdd..(i + 1) * pdd]);
    });
    ple
}

/// The page-fault path: WILLNEED every distinct row, then dequantize through the mapping.
pub fn ple_gather_mmap(g: &Gguf, pt: &TensorInfo, idx: &[u32], page: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let (prb, pdata, pdd) = (pt.row_bytes(), g.data(pt), pt.ne0());
    let mut uniq = idx.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    uniq.par_chunks(2048).for_each(|ch| {
        for &rw in ch {
            HostMem::willneed_range(pdata, rw as usize * prb, prb, page);
        }
    });
    let mut ple = vec![0f32; idx.len() * pdd];
    ple.par_chunks_mut(pdd).zip(idx.par_iter()).for_each(|(o, &rw)| {
        fe_ggq::refq::dequant(pt.ty, &pdata[rw as usize * prb..(rw as usize + 1) * prb], o);
    });
    ple
}

/// FNEXT_PLE_IO=mmap: prompt PLE rows through the mapping (WILLNEED + page faults, dropped after use).
pub fn ple_mmap() -> bool {
    std::env::var("FNEXT_PLE_IO").as_deref() == Ok("mmap")
}

struct PleT(std::time::Instant, f64, f64);
impl Drop for PleT {
    fn drop(&mut self) {
        if std::env::var("FNEXT_PF_TIME").is_ok() {
            eprintln!("ple_gather: sort {:.0} ms, reads {:.0} ms, scatter {:.0} ms", self.1 * 1e3, (self.2 - self.1) * 1e3,
                (self.0.elapsed().as_secs_f64() - self.2) * 1e3);
        }
    }
}
