//! Safetensors index over a directory of shards, host side: every shard is memory-mapped once and tensors are borrowed
//! slices of the maps (the XQ code tensors are read straight from the page cache into the packer, no copy).
use anyhow::{ensure, Context as _, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub file: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// byte offset in the file (header included)
    pub offset: usize,
    pub bytes: usize,
}

pub struct SafeTensors {
    pub dir: PathBuf,
    pub files: Vec<(PathBuf, Mmap)>,
    pub tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    /// Index every `*.safetensors` in `dir` (sorted by name).
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
            // SAFETY: the checkpoint is read-only input; nothing truncates it while the map lives.
            let map = unsafe { Mmap::map(&f) }.with_context(|| format!("mmap {p:?}"))?;
            ensure!(map.len() >= 8, "{p:?}: too short");
            let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
            ensure!(8 + n <= map.len(), "{p:?}: bad header length");
            let h: serde_json::Map<String, serde_json::Value> =
                serde_json::from_slice(&map[8..8 + n]).with_context(|| format!("{p:?}: header"))?;
            for (k, v) in h {
                if k == "__metadata__" {
                    continue;
                }
                let dtype = v["dtype"].as_str().unwrap_or("").to_string();
                let shape: Vec<usize> =
                    v["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
                let off = v["data_offsets"].as_array().context("data_offsets")?;
                let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
                ensure!(a <= b && 8 + n + b <= map.len(), "{p:?}: {k}: data out of range");
                tensors.insert(k, TensorInfo { file: fi, dtype, shape, offset: 8 + n + a, bytes: b - a });
            }
            files.push((p, map));
        }
        Ok(SafeTensors { dir: dir.to_path_buf(), files, tensors })
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("tensor {name} not found in {:?}", self.dir))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// The raw bytes of a tensor (borrowed from the map).
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.get(name)?;
        Ok(&self.files[t.file].1[t.offset..t.offset + t.bytes])
    }

    /// Bytes of a tensor with a dtype and shape check.
    pub fn typed(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<&[u8]> {
        let t = self.get(name)?;
        ensure!(t.dtype == dtype, "{name}: dtype {} (want {dtype})", t.dtype);
        ensure!(t.shape == shape, "{name}: shape {:?} (want {shape:?})", t.shape);
        self.bytes(name)
    }

    /// int8 view (alignment 1: no copy).
    pub fn i8s(&self, name: &str, shape: &[usize]) -> Result<&[i8]> {
        let b = self.typed(name, "I8", shape)?;
        // SAFETY: i8 and u8 have the same size and alignment.
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const i8, b.len()) })
    }

    /// 16-bit elements (fp16 / bf16 bits) copied out little-endian (the data offset need not be 2-aligned).
    pub fn u16s(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<Vec<u16>> {
        let b = self.typed(name, dtype, shape)?;
        Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
}
