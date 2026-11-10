//! Readers for the arc-engine oracle dumps (reference logits and per-kernel launch dumps written by arc-engine; not in this repository).
//!
//! - [`KernelDump`]: one kernel launch from a real forward (`ARC_KDUMP`, `*.kd`): the launch shape, every argument, and each device buffer
//!   argument's bytes before the launch and after it (when the launch changed them). A Rust kernel is checked by uploading the `pre`
//!   buffers, launching with the same scalars / shape, and comparing the changed buffers with `post` bit for bit.
//! - [`XqMvDump`] (fe-xq): the XQ decode matvec on a real checkpoint tensor (`xqmv/<name>.bin`).
use anyhow::{bail, ensure, Context as _, Result};
use std::path::Path;

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

#[derive(Debug, Clone)]
pub enum KArg {
    Unset,
    /// raw bytes of a scalar argument (u32 / f32 / u64 ...)
    Scalar(Vec<u8>),
    /// local (shared) memory of this many bytes
    Local(u64),
    /// a device buffer: the pointer value (KDMP0002; arguments with equal / overlapping pointers alias), its offset in its allocation,
    /// the allocation size, the dumped bytes from the pointer on (at most ARC_KDUMP_MAX), and the bytes after the launch when it changed them
    Buffer { ptr: Option<u64>, offset: u64, alloc: u64, pre: Vec<u8>, post: Option<Vec<u8>> },
}

impl KArg {
    pub fn u32(&self) -> Option<u32> {
        match self {
            KArg::Scalar(b) if b.len() >= 4 => Some(u32::from_le_bytes(b[..4].try_into().unwrap())),
            _ => None,
        }
    }
    pub fn f32(&self) -> Option<f32> {
        self.u32().map(f32::from_bits)
    }
}

#[derive(Debug, Clone)]
pub struct KernelDump {
    pub name: String,
    /// work-group size
    pub local: [u32; 3],
    /// work-group counts
    pub groups: [u32; 3],
    pub args: Vec<KArg>,
}

impl KernelDump {
    pub fn read(path: &Path) -> Result<KernelDump> {
        let b = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
        let mut r = Rd { b: &b, o: 0 };
        let magic = r.take(8)?;
        let v2 = magic == b"KDMP0002";
        ensure!(v2 || magic == b"KDMP0001", "{path:?}: not a kernel dump");
        let nl = r.u32()? as usize;
        let name = String::from_utf8_lossy(r.take(nl)?).to_string();
        let local = [r.u32()?, r.u32()?, r.u32()?];
        let groups = [r.u32()?, r.u32()?, r.u32()?];
        let na = r.u32()? as usize;
        let mut args = Vec::with_capacity(na);
        for _ in 0..na {
            let kind = r.u32()?;
            let size = r.u64()?;
            args.push(match kind {
                0 => KArg::Unset,
                1 => KArg::Scalar(r.take(size.min(8) as usize)?.to_vec()),
                2 => KArg::Local(size),
                3 => {
                    let ptr = if v2 { Some(r.u64()?) } else { None };
                    let offset = r.u64()?;
                    let alloc = r.u64()?;
                    let n = r.u64()? as usize;
                    let changed = r.take(1)?[0] != 0;
                    let pre = r.take(n)?.to_vec();
                    let post = if changed { Some(r.take(n)?.to_vec()) } else { None };
                    KArg::Buffer { ptr, offset, alloc, pre, post }
                }
                k => bail!("{path:?}: argument kind {k}"),
            });
        }
        Ok(KernelDump { name, local, groups, args })
    }

    /// Indices of the buffer arguments the launch wrote.
    pub fn outputs(&self) -> Vec<usize> {
        self.args.iter().enumerate().filter(|(_, a)| matches!(a, KArg::Buffer { post: Some(_), .. })).map(|(i, _)| i).collect()
    }
}

pub use fe_xq::fixture::{XqMvCase, XqMvDump};
