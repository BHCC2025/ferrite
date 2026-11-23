//! Direct reader of the release safetensors: header JSON, pread into pinned staging, upload.
use anyhow::{bail, Context, Result};
use fe_cuda::{memcpy_htod, DevPtr, DeviceBuffer, HostBuffer};
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Info {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub bytes: usize,
}

pub struct Ckpt {
    f: File,
    pub t: HashMap<String, Info>,
    stage: HostBuffer<u8>,
}
const STAGE: usize = 64 << 20;

impl Ckpt {
    pub fn open(path: &Path) -> Result<Ckpt> {
        let f = File::open(path).with_context(|| format!("open {path:?}"))?;
        let mut n8 = [0u8; 8];
        f.read_exact_at(&mut n8, 0)?;
        let n = u64::from_le_bytes(n8) as usize;
        let mut hb = vec![0u8; n];
        f.read_exact_at(&mut hb, 8)?;
        let h: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&hb)?;
        let mut t = HashMap::new();
        for (k, v) in h {
            if k == "__metadata__" {
                continue;
            }
            let shape = v["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let o = v["data_offsets"].as_array().context("offsets")?;
            let (a, b) = (o[0].as_u64().unwrap(), o[1].as_u64().unwrap());
            t.insert(k, Info { dtype: v["dtype"].as_str().unwrap().to_string(), shape, offset: 8 + n as u64 + a,
                bytes: (b - a) as usize });
        }
        Ok(Ckpt { f, t, stage: HostBuffer::alloc(STAGE)? })
    }
    pub fn info(&self, name: &str) -> Result<&Info> {
        self.t.get(name).with_context(|| format!("tensor {name} not found"))
    }
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let i = self.info(name)?;
        let mut v = vec![0u8; i.bytes];
        self.f.read_exact_at(&mut v, i.offset)?;
        Ok(v)
    }
    pub fn read_range(&self, name: &str, byte_off: usize, len: usize) -> Result<Vec<u8>> {
        let i = self.info(name)?;
        if byte_off + len > i.bytes {
            bail!("{name}: range past the end");
        }
        let mut v = vec![0u8; len];
        self.f.read_exact_at(&mut v, i.offset + byte_off as u64)?;
        Ok(v)
    }
    /// f32 tensor from F32 / BF16 storage.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let i = self.info(name)?;
        let raw = self.read(name)?;
        Ok(match i.dtype.as_str() {
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "BF16" => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            d => bail!("{name}: dtype {d}"),
        })
    }
    /// Upload the first `bytes` bytes (all when None) of a tensor into a new device buffer.
    pub fn upload(&mut self, name: &str, bytes: Option<usize>) -> Result<DeviceBuffer<u8>> {
        let i = self.info(name)?.clone();
        let total = bytes.unwrap_or(i.bytes);
        let buf = DeviceBuffer::<u8>::alloc(total)?;
        let mut done = 0;
        while done < total {
            let n = (total - done).min(STAGE);
            let b = &mut self.stage.as_mut_slice()[..n];
            self.f.read_exact_at(b, i.offset + done as u64)?;
            memcpy_htod(buf.ptr().byte_add(done), &b[..n])?;
            done += n;
        }
        Ok(buf)
    }
    /// Upload a tensor (all of it) to `dst` on the device; returns the byte count.
    pub fn upload_to(&mut self, name: &str, dst: DevPtr) -> Result<usize> {
        let i = self.info(name)?.clone();
        let mut done = 0;
        while done < i.bytes {
            let n = (i.bytes - done).min(STAGE);
            let b = &mut self.stage.as_mut_slice()[..n];
            self.f.read_exact_at(b, i.offset + done as u64)?;
            memcpy_htod(dst.byte_add(done), &b[..n])?;
            done += n;
        }
        Ok(i.bytes)
    }
    pub fn upload_f32(&self, v: &[f32]) -> Result<DeviceBuffer<u8>> {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Ok(DeviceBuffer::<u8>::from_slice(&b)?)
    }
}

pub fn dp(b: &DeviceBuffer<u8>) -> DevPtr {
    b.ptr()
}
