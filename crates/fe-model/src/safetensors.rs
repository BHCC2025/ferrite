//! Safetensors index over a directory of shards, and uploads straight to device memory.
//!
//! Uploads read with pread into a pinned staging buffer and drop the file's page cache behind them
//! (POSIX_FADV_DONTNEED): on the GB10 the page cache and the GPU allocations share the same 121 GB,
//! so a 50 GB checkpoint must not sit in both at once.
use anyhow::{bail, ensure, Context as _, Result};
use fe_cuda::{memcpy_htod, DevPtr, DeviceBuffer, HostBuffer};
use std::collections::HashMap;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

unsafe extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}
const POSIX_FADV_DONTNEED: i32 = 4;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub file: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// absolute byte offset in the file
    pub offset: u64,
    pub bytes: usize,
}

pub struct SafeTensors {
    pub files: Vec<(PathBuf, File)>,
    pub tensors: HashMap<String, TensorInfo>,
    staging: HostBuffer<u8>,
}

const STAGE: usize = 64 << 20;

impl SafeTensors {
    pub fn open_dir(dir: &Path) -> Result<SafeTensors> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {dir:?}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        ensure!(!paths.is_empty(), "no .safetensors in {dir:?}");
        let mut files = Vec::new();
        let mut tensors = HashMap::new();
        for (fi, p) in paths.into_iter().enumerate() {
            let f = File::open(&p).with_context(|| format!("open {p:?}"))?;
            let mut n8 = [0u8; 8];
            f.read_exact_at(&mut n8, 0)?;
            let n = u64::from_le_bytes(n8) as usize;
            let mut hb = vec![0u8; n];
            f.read_exact_at(&mut hb, 8)?;
            let h: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&hb)?;
            for (k, v) in h {
                if k == "__metadata__" {
                    continue;
                }
                let dtype = v["dtype"].as_str().unwrap_or("").to_string();
                let shape: Vec<usize> = v["shape"].as_array().context("shape")?.iter()
                    .map(|x| x.as_u64().unwrap_or(0) as usize).collect();
                let off = v["data_offsets"].as_array().context("data_offsets")?;
                let (a, b) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
                tensors.insert(k, TensorInfo { file: fi, dtype, shape, offset: 8 + n as u64 + a, bytes: (b - a) as usize });
            }
            files.push((p, f));
        }
        Ok(SafeTensors { files, tensors, staging: HostBuffer::alloc(STAGE)? })
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("tensor {name} not found"))
    }
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Copy a tensor's raw bytes to `dst` on the device.
    pub fn upload_to(&mut self, name: &str, dst: DevPtr) -> Result<usize> {
        let t = self.get(name)?.clone();
        let f = &self.files[t.file].1;
        let mut done = 0;
        while done < t.bytes {
            let n = (t.bytes - done).min(STAGE);
            let buf = &mut self.staging.as_mut_slice()[..n];
            f.read_exact_at(buf, t.offset + done as u64).with_context(|| format!("reading {name}"))?;
            memcpy_htod(dst.byte_add(done), &buf[..n])?;
            done += n;
        }
        unsafe { posix_fadvise(f.as_raw_fd(), t.offset as i64, t.bytes as i64, POSIX_FADV_DONTNEED) };
        Ok(t.bytes)
    }

    /// Read a tensor to host memory.
    pub fn read_host(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.get(name)?;
        let mut v = vec![0u8; t.bytes];
        self.files[t.file].1.read_exact_at(&mut v, t.offset)?;
        Ok(v)
    }

    /// Upload bf16 tensors concatenated along dim 0 (e.g. [q|k|v] projection rows) into one buffer.
    pub fn upload_cat_bf16(&mut self, names: &[String]) -> Result<DeviceBuffer<u16>> {
        let mut total = 0;
        for n in names {
            let t = self.get(n)?;
            if t.dtype != "BF16" {
                bail!("{n}: dtype {} (want BF16)", t.dtype);
            }
            total += t.bytes;
        }
        let buf = DeviceBuffer::<u16>::alloc(total / 2)?;
        let mut off = 0;
        for n in names {
            off += self.upload_to(n, buf.ptr().byte_add(off))?;
        }
        Ok(buf)
    }
    pub fn upload_bf16(&mut self, name: &str) -> Result<DeviceBuffer<u16>> {
        self.upload_cat_bf16(&[name.to_string()])
    }
}
