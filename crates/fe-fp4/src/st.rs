//! Minimal safetensors reader over a checkpoint directory (pread, page cache dropped behind each read:
//! on the GB10 the page cache and GPU allocations share the same 121 GB).
use anyhow::{bail, Context as _, Result};
use std::collections::HashMap;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

unsafe extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub file: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub bytes: usize,
}

pub struct Checkpoint {
    pub dir: PathBuf,
    files: Vec<File>,
    pub tensors: HashMap<String, Tensor>,
    /// hf_quant_config.json `quantization.quantized_layers` (layer -> {"quant_algo": ..})
    pub quant: serde_json::Map<String, serde_json::Value>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Checkpoint> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {dir:?}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        let mut files = Vec::new();
        let mut tensors = HashMap::new();
        for (fi, p) in paths.iter().enumerate() {
            let f = File::open(p).with_context(|| format!("open {p:?}"))?;
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
                let off = &v["data_offsets"];
                let (a, b) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
                tensors.insert(k, Tensor {
                    file: fi,
                    dtype: v["dtype"].as_str().unwrap_or("").to_string(),
                    shape: v["shape"].as_array().map(|s| s.iter().map(|x| x.as_u64().unwrap() as usize).collect())
                        .unwrap_or_default(),
                    offset: 8 + n as u64 + a,
                    bytes: (b - a) as usize,
                });
            }
            files.push(f);
        }
        let quant = std::fs::read(dir.join("hf_quant_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| v["quantization"]["quantized_layers"].as_object().cloned())
            .unwrap_or_default();
        Ok(Checkpoint { dir: dir.to_path_buf(), files, tensors, quant })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name).with_context(|| format!("tensor {name} not in {:?}", self.dir))
    }
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Raw bytes of a tensor.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.get(name)?;
        let f = &self.files[t.file];
        let mut v = vec![0u8; t.bytes];
        f.read_exact_at(&mut v, t.offset).with_context(|| format!("reading {name}"))?;
        unsafe { posix_fadvise(f.as_raw_fd(), t.offset as i64, t.bytes as i64, 4) };
        Ok(v)
    }

    /// A scalar F32 tensor (ModelOpt `weight_scale_2`, `input_scale`, FP8 `weight_scale`).
    pub fn scalar(&self, name: &str) -> Result<f32> {
        let t = self.get(name)?;
        if t.dtype != "F32" || t.bytes != 4 {
            bail!("{name}: expected scalar F32, got {} {:?}", t.dtype, t.shape);
        }
        let b = self.read(name)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// ModelOpt quant algo of a layer ("NVFP4", "FP8") from hf_quant_config.json, or from the tensor dtypes.
    pub fn algo(&self, layer: &str) -> Option<String> {
        if let Some(a) = self.quant.get(layer).and_then(|v| v["quant_algo"].as_str()) {
            return Some(a.to_string());
        }
        match self.tensors.get(&format!("{layer}.weight"))?.dtype.as_str() {
            "U8" if self.has(&format!("{layer}.weight_scale_2")) => Some("NVFP4".into()),
            "F8_E4M3" => Some("FP8".into()),
            _ => None,
        }
    }
}
