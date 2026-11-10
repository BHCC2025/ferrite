//! Replay of arc-engine ARC_KDUMP launch dumps (`*.kd`, KDMP0001 / KDMP0002; format in the oracle README and
//! fe_xe::oracle::KernelDump, read here again so fe-kxcore does not depend on fe-xe): upload every buffer argument's
//! `pre` bytes, launch the same-named kernel of a [`KxCore`] with the dumped scalars and shape, download, and compare
//! with `post` (or `pre` for buffers the reference left unchanged) bit for bit. Arguments whose dumped ranges overlap
//! (KDMP0002 pointer values) share one device allocation, so in-place launches replay as they ran.
use crate::KxCore;
use anyhow::{bail, ensure, Context, Result};
use fe_ze::{DevPtr, DeviceBuffer, KernelArg, Local, Queue};
use std::ffi::c_void;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum KArg {
    Unset,
    Scalar(Vec<u8>),
    Local(u64),
    Buffer { ptr: Option<u64>, offset: u64, alloc: u64, pre: Vec<u8>, post: Option<Vec<u8>> },
}

#[derive(Debug, Clone)]
pub struct KernelDump {
    pub name: String,
    pub local: [u32; 3],
    pub groups: [u32; 3],
    pub args: Vec<KArg>,
}

impl KernelDump {
    pub fn read(path: &Path) -> Result<KernelDump> {
        let b = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
        let mut o = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            ensure!(o + n <= b.len(), "{path:?}: truncated at {o}");
            o += n;
            Ok(&b[o - n..o])
        };
        let magic = take(8)?.to_vec();
        let v2 = magic == b"KDMP0002";
        ensure!(v2 || magic == b"KDMP0001", "{path:?}: not a kernel dump");
        let u32_ = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());
        let u64_ = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap());
        let nl = u32_(take(4)?) as usize;
        let name = String::from_utf8_lossy(take(nl)?).to_string();
        let mut l3 = [0u32; 6];
        for v in l3.iter_mut() {
            *v = u32_(take(4)?);
        }
        let na = u32_(take(4)?) as usize;
        let mut args = Vec::with_capacity(na);
        for _ in 0..na {
            let kind = u32_(take(4)?);
            let size = u64_(take(8)?);
            args.push(match kind {
                0 => KArg::Unset,
                1 => KArg::Scalar(take(size.min(8) as usize)?.to_vec()),
                2 => KArg::Local(size),
                3 => {
                    let ptr = if v2 { Some(u64_(take(8)?)) } else { None };
                    let offset = u64_(take(8)?);
                    let alloc = u64_(take(8)?);
                    let n = u64_(take(8)?) as usize;
                    let changed = take(1)?[0] != 0;
                    let pre = take(n)?.to_vec();
                    let post = if changed { Some(take(n)?.to_vec()) } else { None };
                    KArg::Buffer { ptr, offset, alloc, pre, post }
                }
                k => bail!("{path:?}: argument kind {k}"),
            });
        }
        Ok(KernelDump { name, local: [l3[0], l3[1], l3[2]], groups: [l3[3], l3[4], l3[5]], args })
    }
}

struct Raw(Vec<u8>);
impl KernelArg for Raw {
    fn raw(&self) -> (usize, *const c_void) {
        (self.0.len(), self.0.as_ptr() as *const c_void)
    }
}

/// Result of one replay: per written buffer (argument index, mismatching 4-byte words, words compared), and the bytes
/// every buffer argument held after the launch. `truncated`: some buffer was dumped only in part (ARC_KDUMP_MAX); its
/// tail is zeros in the replay, so a launch that reads there cannot reproduce `post`: compare two replays instead.
pub struct Replay {
    pub outputs: Vec<(usize, usize, usize)>,
    pub unchanged_bad: usize,
    pub truncated: bool,
    pub after: Vec<Option<Vec<u8>>>,
}

/// Replay `d` on `kx` (its kernel of the dump's name, with the dump's work-group size).
pub fn replay(kx: &KxCore, q: &Queue, d: &KernelDump) -> Result<Replay> {
    // alias groups: [start, end) in the original address space (KDMP0002), else one group a buffer
    let mut spans: Vec<(usize, u64, u64)> = Vec::new(); // (arg, start, end of its allocation)
    let mut truncated = false;
    for (i, a) in d.args.iter().enumerate() {
        if let KArg::Buffer { ptr, offset, alloc, pre, .. } = a {
            let s = ptr.unwrap_or(((i as u64) + 1) << 48);
            let full = alloc.saturating_sub(*offset).max(pre.len() as u64);
            truncated |= full > pre.len() as u64;
            spans.push((i, s, s + full));
        }
    }
    let mut groups: Vec<(u64, u64, Vec<usize>)> = Vec::new();
    let mut sorted = spans.clone();
    sorted.sort_by_key(|x| x.1);
    for (i, s, e) in sorted {
        match groups.last_mut() {
            Some(g) if s < g.1 => {
                g.1 = g.1.max(e);
                g.2.push(i);
            }
            _ => groups.push((s, e, vec![i])),
        }
    }
    let mut bufs = Vec::new();
    let mut base_of = vec![DevPtr::NULL; d.args.len()];
    for (s, e, members) in &groups {
        let b = DeviceBuffer::<u8>::zeroed(q, (e - s) as usize)?;
        for &i in members {
            let (_, st, _) = spans.iter().find(|x| x.0 == i).unwrap();
            if let KArg::Buffer { pre, .. } = &d.args[i] {
                q.upload(b.ptr().byte_add((st - s) as usize), pre)?;
            }
            base_of[i] = b.ptr().byte_add((st - s) as usize);
        }
        bufs.push(b);
    }
    let k = kx.kernel_for(&d.name, d.local)?;
    let mut raws: Vec<Box<dyn KernelArg>> = Vec::new();
    for (i, a) in d.args.iter().enumerate() {
        raws.push(match a {
            KArg::Unset => bail!("{}: argument {i} unset", d.name),
            KArg::Scalar(b) => Box::new(Raw(b.clone())),
            KArg::Local(n) => Box::new(Local(*n as usize)),
            KArg::Buffer { .. } => Box::new(base_of[i]),
        });
    }
    let refs: Vec<&dyn KernelArg> = raws.iter().map(|b| b.as_ref()).collect();
    k.args(&refs)?;
    q.launch(&k, d.groups)?;
    q.sync()?;
    let mut out = Replay { outputs: Vec::new(), unchanged_bad: 0, truncated, after: vec![None; d.args.len()] };
    for (i, a) in d.args.iter().enumerate() {
        if let KArg::Buffer { pre, post, .. } = a {
            let mut got = vec![0u8; pre.len()];
            q.download(&mut got, base_of[i])?;
            let want = post.as_ref().unwrap_or(pre);
            let bad = got.chunks(4).zip(want.chunks(4)).filter(|(x, y)| x != y).count();
            if post.is_some() {
                out.outputs.push((i, bad, got.len() / 4));
            } else {
                out.unchanged_bad += bad;
            }
            out.after[i] = Some(got);
        }
    }
    Ok(out)
}
