//! Minimal sharded safetensors reader: the index, each shard's header, tensor bytes by positioned reads.
//! (fe-model has the engine's loader; this one keeps fe-exl3 testable on its own.)

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub file: usize,
    /// Absolute byte offset in the shard file.
    pub offset: u64,
    pub len: u64,
}

pub struct Checkpoint {
    pub dir: PathBuf,
    files: Vec<File>,
    pub tensors: HashMap<String, TensorInfo>,
}

impl Checkpoint {
    /// Open `dir` holding `model.safetensors.index.json` (or a single `model.safetensors`).
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let idx = dir.join("model.safetensors.index.json");
        let mut shard_names: Vec<String> = Vec::new();
        if idx.exists() {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx)?)?;
            let map = v["weight_map"].as_object().ok_or_else(|| Error::msg("index without weight_map"))?;
            for f in map.values() {
                let f = f.as_str().ok_or_else(|| Error::msg("weight_map value"))?.to_string();
                if !shard_names.contains(&f) {
                    shard_names.push(f);
                }
            }
            shard_names.sort();
        } else {
            shard_names.push("model.safetensors".into());
        }
        let mut files = Vec::new();
        let mut tensors = HashMap::new();
        for (fi, name) in shard_names.iter().enumerate() {
            let f = File::open(dir.join(name))?;
            let mut n = [0u8; 8];
            f.read_exact_at(&mut n, 0)?;
            let hlen = u64::from_le_bytes(n);
            let mut h = vec![0u8; hlen as usize];
            f.read_exact_at(&mut h, 8)?;
            let v: serde_json::Value = serde_json::from_slice(&h)?;
            for (k, t) in v.as_object().ok_or_else(|| Error::msg("header"))? {
                if k == "__metadata__" {
                    continue;
                }
                let dtype = t["dtype"].as_str().unwrap_or("").to_string();
                let shape = t["shape"]
                    .as_array()
                    .ok_or_else(|| Error::msg("shape"))?
                    .iter()
                    .map(|d| d.as_u64().unwrap_or(0) as usize)
                    .collect();
                let off = t["data_offsets"].as_array().ok_or_else(|| Error::msg("data_offsets"))?;
                let (a, b) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
                tensors.insert(k.clone(), TensorInfo { dtype, shape, file: fi, offset: 8 + hlen + a, len: b - a });
            }
            files.push(f);
        }
        Ok(Self { dir, files, tensors })
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| Error::msg(format!("tensor {name} not in checkpoint")))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn read_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.info(name)?;
        let mut v = vec![0u8; t.len as usize];
        self.files[t.file].read_exact_at(&mut v, t.offset)?;
        Ok(v)
    }

    /// A 16-bit tensor (f16 / bf16 / i16 / u16) as little-endian u16 words.
    pub fn read_u16(&self, name: &str) -> Result<Vec<u16>> {
        let b = self.read_bytes(name)?;
        if b.len() % 2 != 0 {
            return Err(Error::msg(format!("{name}: odd byte length")));
        }
        Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    pub fn read_u32(&self, name: &str) -> Result<Vec<u32>> {
        let b = self.read_bytes(name)?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}
