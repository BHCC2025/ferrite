//! The XQ prompt-window kernels in Rust (kx-xq `xq_pfdec16x_128` + `xq_pfprep_k16`, kx-xqpf `xq_pfgemm_f16_cCtT`) against
//! arc-engine's OpenCL (`xqpfd{bits}.spv`, `xq_pfg.spv` 256 GRF), bit for bit and in time:
//! * `--kd DIR`: replay arc-engine launch dumps of a real prompt window (ARC_KDUMP, the oracle's kdump/prompt) on the Rust
//!   kernels: every written buffer must equal the dump's `post` bytes;
//! * real checkpoint tensors (one per shape and width with G = 128 and K/128 % 8 == 0) and R-row windows of bf16 activations:
//!   decode W16 [N][K], prep (fp16 and int8: xt [K][Rp], sa) and the GEMM (fp32 and bf16 out) on the same inputs,
//!   Rust == OpenCL; `--time` adds per-launch times (OpenCL and Rust alternate rep by rep, median).
//! * `--i8` (X3 groundwork): the int8 path too: xq_pfdec_128 to int8 codes + group scales (and its f16out form), the exact
//!   XQ_PF=i8 GEMMs (untiled xq_pfgemm_i8{,w}_128 vs xqpf{bits}.spv, tiled xq_pfgemm_i8g_128_c4t4 vs xq_pfg.spv) and the
//!   group-free xq_pfgemm_i8c_c4t4 (on random per-channel / per-token scales), bitwise; `--time` reports TOP/s next to fp16.
//! usage: xqpf-check [--kd DIR] [--model DIR] [--bits 4,5] [--rows 2048,4096,1500] [--shapes gate_up,down] [--tile c4t4] [--time] [--i8]
use anyhow::{bail, Context, Result};
use fe_kx_tests::*;
use fe_kxcore::kd::{KArg, KernelDump};
use fe_xq::{Checkpoint, Rotate};
use fe_ze::*;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const GRF: &str = "-cl-intel-256-GRF-per-thread";

struct Raw(Vec<u8>);
impl KernelArg for Raw {
    fn raw(&self) -> (usize, *const c_void) {
        (self.0.len(), self.0.as_ptr() as *const c_void)
    }
}

fn label(n: usize, k: usize) -> String {
    match (n, k) {
        (17408, 5120) => "gate_up".into(),
        (5120, 17408) => "down".into(),
        (10240, 5120) => "qkv".into(),
        (12288, 5120) => "q".into(),
        (6144, 5120) => "z".into(),
        (5120, 6144) => "out_o".into(),
        (1024, 5120) => "kv".into(),
        (48, 5120) => "ab".into(),
        (248320, 5120) => "lm_head".into(),
        (5120, 10240) => "mtp_fc".into(),
        _ => format!("{n}x{k}"),
    }
}

/// Code width of a packed XQ tensor from its size (pfdec dumps do not record the module).
fn bits_of(packed_bytes: u64, n: u64, k: u64) -> Option<u32> {
    [3u32, 4, 5, 6, 8].into_iter().find(|&b| (n / 16) * (k / 128) * (16 * (128 * b as u64 / 32) + 8) * 4 == packed_bytes)
}

/// Replay one dump on a kernel: upload `pre`, launch with the dumped scalars / shape, compare every written buffer with `post`.
fn replay(q: &Queue, k: &Kernel, d: &KernelDump) -> Result<(usize, usize, bool)> {
    let mut spans = Vec::new();
    let mut truncated = false;
    for (i, a) in d.args.iter().enumerate() {
        if let KArg::Buffer { ptr, offset, alloc, pre, .. } = a {
            let s = ptr.unwrap_or(((i as u64) + 1) << 48);
            let full = alloc.saturating_sub(*offset).max(pre.len() as u64);
            truncated |= full > pre.len() as u64;
            spans.push((i, s, s + full));
        }
    }
    let mut sorted = spans.clone();
    sorted.sort_by_key(|x| x.1);
    let mut groups: Vec<(u64, u64, Vec<usize>)> = Vec::new();
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
    let mut base = vec![DevPtr::NULL; d.args.len()];
    for (s, e, members) in &groups {
        let b = DeviceBuffer::<u8>::zeroed(q, (e - s) as usize)?;
        for &i in members {
            let st = spans.iter().find(|x| x.0 == i).unwrap().1;
            if let KArg::Buffer { pre, .. } = &d.args[i] {
                q.upload(b.ptr().byte_add((st - s) as usize), pre)?;
            }
            base[i] = b.ptr().byte_add((st - s) as usize);
        }
        bufs.push(b);
    }
    let mut raws: Vec<Box<dyn KernelArg>> = Vec::new();
    for (i, a) in d.args.iter().enumerate() {
        raws.push(match a {
            KArg::Unset => bail!("{}: argument {i} unset", d.name),
            KArg::Scalar(b) => Box::new(Raw(b.clone())),
            KArg::Local(n) => Box::new(Local(*n as usize)),
            KArg::Buffer { .. } => Box::new(base[i]),
        });
    }
    let refs: Vec<&dyn KernelArg> = raws.iter().map(|b| b.as_ref()).collect();
    k.args(&refs)?;
    q.launch(k, d.groups)?;
    q.sync()?;
    let (mut bad, mut total) = (0, 0);
    for (i, a) in d.args.iter().enumerate() {
        if let KArg::Buffer { pre, post: Some(post), .. } = a {
            let mut got = vec![0u8; pre.len()];
            q.download(&mut got, base[i])?;
            bad += got.chunks(4).zip(post.chunks(4)).filter(|(x, y)| x != y).count();
            total += got.len() / 4;
        }
    }
    Ok((bad, total, truncated))
}

fn scalar_u32(d: &KernelDump, i: usize) -> u64 {
    match &d.args[i] {
        KArg::Scalar(b) => u32::from_le_bytes(b[..4].try_into().unwrap()) as u64,
        _ => 0,
    }
}

struct Mods {
    ze: Arc<Ze>,
    xq: BTreeMap<u32, (Module, Module)>,
    pfg: (Module, Module),
}
impl Mods {
    fn xq(&mut self, bits: u32) -> Result<&(Module, Module)> {
        if !self.xq.contains_key(&bits) {
            let r = Module::from_spirv(&self.ze, kx_xq(bits), "")?;
            let o = arc_module(&self.ze, &format!("xqpfd{bits}.spv"), "")?;
            self.xq.insert(bits, (r, o));
        }
        Ok(&self.xq[&bits])
    }
}

fn time_pair(q: &Queue, ks: [&Kernel; 2], groups: [u32; 3], reps: usize, calls: usize) -> Result<[f64; 2]> {
    let mut s = [Vec::new(), Vec::new()];
    for rep in 0..=reps {
        for i in 0..2 {
            q.sync()?;
            let t0 = now_ns();
            for _ in 0..calls {
                q.launch(ks[i], groups)?;
            }
            q.sync()?;
            if rep > 0 {
                s[i].push((now_ns() - t0) as f64 / calls as f64 / 1e3);
            }
        }
    }
    Ok([median(&s[0]), median(&s[1])])
}

/// As [`time_pair`] with a grid each (kernels of different tilings).
fn time_pair2(q: &Queue, ks: [(&Kernel, [u32; 3]); 2], reps: usize, calls: usize) -> Result<[f64; 2]> {
    let mut s = [Vec::new(), Vec::new()];
    for rep in 0..=reps {
        for i in 0..2 {
            q.sync()?;
            let t0 = now_ns();
            for _ in 0..calls {
                q.launch(ks[i].0, ks[i].1)?;
            }
            q.sync()?;
            if rep > 0 {
                s[i].push((now_ns() - t0) as f64 / calls as f64 / 1e3);
            }
        }
    }
    Ok([median(&s[0]), median(&s[1])])
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut opt = |name: &str| -> Option<String> {
        let i = args.iter().position(|a| a == name)?;
        args.remove(i);
        if i < args.len() && !args[i].starts_with("--") { Some(args.remove(i)) } else { Some(String::new()) }
    };
    let kd = opt("--kd").map(PathBuf::from);
    let model = PathBuf::from(opt("--model").or_else(|| std::env::var("FE_XQ_MODEL").ok()).expect("--model DIR (or FE_XQ_MODEL): an XQ checkpoint folder"));
    let list = |s: Option<String>, d: &str| -> Vec<u32> { s.unwrap_or(d.into()).split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect() };
    let bits_want = list(opt("--bits"), "3,4,5,6,8");
    let rows = list(opt("--rows"), "2048,4096,1500");
    let shapes_want: Vec<String> = opt("--shapes").map(|s| s.split(',').map(String::from).collect()).unwrap_or_default();
    let tile = opt("--tile").unwrap_or("c4t4".into());
    let do_time = opt("--time").is_some();
    let do_i8 = opt("--i8").is_some();
    let no_check = opt("--nocheck").is_some();
    // X3 kernels with no OpenCL twin, checked against the OpenCL kernel of the same numerics: i8c2_cCtT vs i8c_c4t4
    // --i8c: the group-free int8 path end to end (per-channel weight requant, per-token activation scale, i8c2 GEMM): exact host
    // checks of every rounding step and the GEMM, the error of i8c2 and of the exact grouped i8g2 against the fp16 GEMM
    let do_i8c = opt("--i8c").is_some();
    // --sb S: the K-sliced int8 path (xq_pfgemm_i8s) with slices of S 128-blocks, checked and timed inside --i8c
    let sb: u32 = opt("--sb").map(|v| v.parse()).transpose()?.unwrap_or(0);
    // --rb S [--rcodes]: the rescaled-int32 sliced path (xq_pfgemm_i8r / i8r3), slices of S blocks; --rcodes: weight scale =
    // the group scale (S = 1: the exact XQ codes, i8g's numerics)
    let rb: u32 = opt("--rb").map(|v| v.parse()).transpose()?.unwrap_or(0);
    let rcodes = u32::from(opt("--rcodes").is_some());
    let x3: Vec<String> = opt("--x3").map(|s| s.split(',').filter(|x| !x.is_empty()).map(String::from).collect()).unwrap_or_default();
    let (tc, tt): (u32, u32) = {
        let c = tile.find('c').unwrap();
        let t = tile.find('t').unwrap();
        (tile[c + 1..t].parse()?, tile[t + 1..].parse()?)
    };
    let gname = format!("xq_pfgemm_f16_{tile}");

    let ze = Ze::new()?;
    let q = Queue::new(&ze)?;
    // KX_XQPF_SPV=path: a kx_xqpf.spv built elsewhere (kbuild-xe kx-xqpf DIR) instead of the embedded one
    let pfg_spv = match std::env::var("KX_XQPF_SPV") {
        Ok(p) => std::fs::read(&p).with_context(|| format!("KX_XQPF_SPV={p}"))?,
        Err(_) => KX_XQPF.to_vec(),
    };
    let pfg = (Module::from_spirv(&ze, &pfg_spv, GRF)?, arc_module(&ze, "xq_pfg.spv", GRF)?);
    let mut mods = Mods { ze: ze.clone(), xq: BTreeMap::new(), pfg };
    let mut fails = 0;

    if let Some(dir) = &kd {
        let mut files: Vec<_> = std::fs::read_dir(dir)?.flatten().map(|e| e.path()).filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            n.ends_with(".kd") && (n.contains("xq_pfgemm_f16") || n.contains("xq_pfprep_k16") || n.contains("xq_pfdec16x_128"))
        }).collect();
        files.sort();
        for f in files {
            let d = KernelDump::read(&f)?;
            let (rk, ok_) = if d.name.starts_with("xq_pfgemm_f16") {
                (mods.pfg.0.kernel(&d.name, d.local)?, mods.pfg.1.kernel(&d.name, d.local)?)
            } else {
                let bits = if d.name == "xq_pfdec16x_128" {
                    let (n, k) = (scalar_u32(&d, 3), scalar_u32(&d, 2));
                    let pb = match &d.args[0] {
                        KArg::Buffer { pre, offset, alloc, .. } => (alloc - offset).min(pre.len() as u64).max(pre.len() as u64),
                        _ => 0,
                    };
                    bits_of(pb, n, k).with_context(|| format!("{f:?}: width of {pb} packed bytes for {n} x {k}"))?
                } else {
                    4
                };
                let m = mods.xq(bits)?;
                (m.0.kernel(&d.name, d.local)?, m.1.kernel(&d.name, d.local)?)
            };
            let (bad, total, trunc) = replay(&q, &rk, &d)?;
            let (obad, _, _) = replay(&q, &ok_, &d)?;
            println!(
                "kd {}: {} groups {:?} local {:?}: rust {} of {} words differ from the dump, opencl replay {}{}",
                f.file_name().unwrap().to_string_lossy(),
                d.name,
                d.groups,
                d.local,
                bad,
                total,
                obad,
                if trunc { " (truncated dump)" } else { "" }
            );
            if bad > 0 && !(trunc && obad > 0) {
                fails += 1;
            }
        }
    }

    if !bits_want.is_empty() && !(kd.is_some() && shapes_want.is_empty() && !do_time) {
        let ckpt = Checkpoint::open(&model).with_context(|| format!("opening {model:?}"))?;
        let mut reps: BTreeMap<(usize, usize, u32, bool), String> = BTreeMap::new();
        for (name, m) in &ckpt.cfg.tensors {
            if m.group == 128 && (m.k / 128) % 8 == 0 && m.n % 16 == 0 && m.n < 200_000 {
                reps.entry((m.n, m.k, m.bits, m.rotate == Rotate::Had128In)).or_insert(name.clone());
            }
        }
        let mut rng = Rng(0x5eed_1234_abcd_0001);
        for ((n, k, bits, rot), name) in reps {
            if !bits_want.contains(&bits) || (!shapes_want.is_empty() && !shapes_want.contains(&label(n, k))) {
                continue;
            }
            let t = ckpt.xq(&name)?;
            let w = DeviceBuffer::from_slice(&q, &t.pack()?)?;
            let su = t.su.as_ref().map(|s| DeviceBuffer::from_slice(&q, s)).transpose()?;
            println!("\n== {} {n}x{k} int{bits} {}: {name}", label(n, k), if rot { "rot" } else { "norot" });
            let (rmod, omod) = { let m = mods.xq(bits)?; (m.0.clone(), m.1.clone()) };
            // int8 decode: Rust kx_xq{bits} (256-GRF build, as the reference) vs xqpf{bits}.spv
            let i8mods = if do_i8 {
                Some((Module::from_spirv(&ze, kx_xq(bits), GRF)?, arc_module(&ze, &format!("xqpf{bits}.spv"), GRF)?))
            } else {
                None
            };
            let ng = k / 128;
            let (w8o, w8r) = (DeviceBuffer::<u32>::zeroed(&q, if do_i8 { n * k / 2 } else { 1 })?, DeviceBuffer::<u32>::zeroed(&q, if do_i8 { n * k / 2 } else { 1 })?);
            let (swo, swr) = (DeviceBuffer::<f32>::zeroed(&q, if do_i8 { ng * n } else { 1 })?, DeviceBuffer::<f32>::zeroed(&q, if do_i8 { ng * n } else { 1 })?);
            let mut i8line = Vec::new();
            if let Some((rm, om)) = &i8mods {
                for f16out in [1u32, 0] {
                    let mk = |m: &Module, wq: &DeviceBuffer<u32>, sw: &DeviceBuffer<f32>| -> Result<Kernel> {
                        let kk = m.kernel("xq_pfdec_128", [16, 1, 1])?;
                        kk.args(&[&w.ptr(), &wq.ptr(), &sw.ptr(), &(k as u32), &(n as u32), &f16out])?;
                        Ok(kk)
                    };
                    let (ko, kr) = (mk(om, &w8o, &swo)?, mk(rm, &w8r, &swr)?);
                    let gd = [(n / 16) as u32, ng as u32, 1];
                    q.launch(&ko, gd)?;
                    q.launch(&kr, gd)?;
                    let words = if f16out == 1 { n * k / 2 } else { n * k / 4 };
                    let nd = w8o.to_vec(&q)?[..words].iter().zip(&w8r.to_vec(&q)?[..words]).filter(|(a, b)| a != b).count()
                        + swo.to_vec(&q)?.iter().zip(swr.to_vec(&q)?.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    i8line.push(format!("dec_128 {} {}", if f16out == 1 { "f16" } else { "i8" }, if nd == 0 { "ok".into() } else { format!("FAIL {nd}") }));
                    fails += (nd > 0) as u32;
                    if do_time {
                        let us = time_pair(&q, [&ko, &kr], gd, 9, 10)?;
                        i8line.push(format!("{:.1}/{:.1} us ({:.3})", us[0], us[1], us[1] / us[0]));
                    }
                }
                println!("  int8: {}", i8line.join(", "));
            }
            // decode
            let w16o = DeviceBuffer::<u16>::zeroed(&q, n * k)?;
            let w16r = DeviceBuffer::<u16>::zeroed(&q, n * k)?;
            let dg = [(n / 16) as u32, (k / 128 / 8) as u32, 1];
            let dko = omod.kernel("xq_pfdec16x_128", [16, 8, 1])?;
            dko.args(&[&w.ptr(), &w16o.ptr(), &(k as u32), &(n as u32)])?;
            let dkr = rmod.kernel("xq_pfdec16x_128", [16, 8, 1])?;
            dkr.args(&[&w.ptr(), &w16r.ptr(), &(k as u32), &(n as u32)])?;
            q.launch(&dko, dg)?;
            q.launch(&dkr, dg)?;
            let mut line = Vec::new();
            if !no_check {
                let (a, b) = (w16o.to_vec(&q)?, w16r.to_vec(&q)?);
                let nd = a.iter().zip(&b).filter(|(x, y)| x != y).count();
                line.push(format!("dec {}", if nd == 0 { "ok".into() } else { format!("FAIL {nd}") }));
                fails += (nd > 0) as u32;
            }
            if do_time {
                let us = time_pair(&q, [&dko, &dkr], dg, 9, 10)?;
                line.push(format!("dec {:.1}/{:.1} us ({:.3})", us[0], us[1], us[1] / us[0]));
            }
            for &r in &rows {
                let r = r as usize;
                let rp = r.div_ceil(64) * 64;
                let x: Vec<u16> = (0..r * k).map(|i| f32_to_bf16((rng.f32() * 2.0 - 1.0) * if i % 89 == 0 { 12.0 } else { 1.5 })).collect();
                let dx = DeviceBuffer::from_slice(&q, &x)?;
                let (ao, ar) = (DeviceBuffer::<u16>::zeroed(&q, k * rp)?, DeviceBuffer::<u16>::zeroed(&q, k * rp)?);
                let (sao, sar) = (DeviceBuffer::<f32>::zeroed(&q, rp * k / 128)?, DeviceBuffer::<f32>::zeroed(&q, rp * k / 128)?);
                let sup = su.as_ref().map(|b| b.ptr()).unwrap_or(sao.ptr());
                let pg = [(k / 128) as u32, (rp / 32) as u32, 1];
                let mut prep_k = |m: &Module, a: &DeviceBuffer<u16>, sa: &DeviceBuffer<f32>, i8: u32| -> Result<Kernel> {
                    let kk = m.kernel("xq_pfprep_k16", [512, 1, 1])?;
                    kk.args(&[&dx.ptr(), &1u32, &(k as u32), &sup, &(rot as u32), &a.ptr(), &sa.ptr(), &(k as u32), &(r as u32), &(rp as u32), &i8])?;
                    Ok(kk)
                };
                let mut cells = Vec::new();
                // int8 prep (bitwise only)
                if !no_check {
                    let (po, pr) = (prep_k(&omod, &ao, &sao, 1)?, prep_k(&rmod, &ar, &sar, 1)?);
                    q.launch(&po, pg)?;
                    q.launch(&pr, pg)?;
                    let nd = ao.to_vec(&q)?.iter().zip(ar.to_vec(&q)?.iter()).filter(|(a, b)| a != b).count()
                        + sao.to_vec(&q)?.iter().zip(sar.to_vec(&q)?.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    cells.push(format!("prep-i8 {}", if nd == 0 { "ok".into() } else { format!("FAIL {nd}") }));
                    fails += (nd > 0) as u32;
                }
                // fp16 prep
                let (po, pr) = (prep_k(&omod, &ao, &sao, 0)?, prep_k(&rmod, &ar, &sar, 0)?);
                q.launch(&po, pg)?;
                q.launch(&pr, pg)?;
                if !no_check {
                    let nd = ao.to_vec(&q)?.iter().zip(ar.to_vec(&q)?.iter()).filter(|(a, b)| a != b).count();
                    cells.push(format!("prep {}", if nd == 0 { "ok".into() } else { format!("FAIL {nd}") }));
                    fails += (nd > 0) as u32;
                }
                if do_time {
                    let us = time_pair(&q, [&po, &pr], pg, 9, 10)?;
                    cells.push(format!("prep {:.1}/{:.1} us ({:.3})", us[0], us[1], us[1] / us[0]));
                }
                // GEMM on the OpenCL decode + prep outputs
                let (yo, yr) = (DeviceBuffer::<f32>::zeroed(&q, r * n)?, DeviceBuffer::<f32>::zeroed(&q, r * n)?);
                let gg = [r.div_ceil(64 * tt as usize) as u32, n.div_ceil(32 * tc as usize) as u32, 1];
                let gk = |m: &Module, y: &DeviceBuffer<f32>, f32out: u32| -> Result<Kernel> {
                    let kk = m.kernel(&gname, [16 * tc * tt, 1, 1])?;
                    kk.args(&[&w16o.ptr(), &ao.ptr(), &y.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &f32out])?;
                    Ok(kk)
                };
                for f32out in [1u32, 0] {
                    let (go, gr) = (gk(&mods.pfg.1, &yo, f32out)?, gk(&mods.pfg.0, &yr, f32out)?);
                    q.launch(&go, gg)?;
                    q.launch(&gr, gg)?;
                    if !no_check {
                        let (a, b) = (yo.to_vec(&q)?, yr.to_vec(&q)?);
                        let words = if f32out == 1 { r * n } else { r * n / 2 };
                        let nd = a[..words].iter().zip(&b[..words]).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                        cells.push(format!("gemm-{} {}", if f32out == 1 { "f32" } else { "bf16" }, if nd == 0 { "ok".into() } else { format!("FAIL {nd}") }));
                        fails += (nd > 0) as u32;
                    }
                    if do_time && f32out == 0 {
                        let us = time_pair(&q, [&go, &gr], gg, 9, 5)?;
                        let tf = |u: f64| 2.0 * r as f64 * n as f64 * k as f64 / u / 1e6;
                        cells.push(format!("gemm {:.0}/{:.0} us {:.0}/{:.0} TFLOP/s ({:.3})", us[0], us[1], tf(us[0]), tf(us[1]), us[1] / us[0]));
                    }
                }
                if let Some((_, om)) = &i8mods {
                    // int8 activations (prep i8 into separate buffers) and the int8 GEMMs on the reference decode / prep
                    let (x8, sa8) = (DeviceBuffer::<u8>::zeroed(&q, k * rp)?, DeviceBuffer::<f32>::zeroed(&q, rp * k / 128)?);
                    let pk = omod.kernel("xq_pfprep_k16", [512, 1, 1])?;
                    pk.args(&[&dx.ptr(), &1u32, &(k as u32), &sup, &(rot as u32), &x8.ptr(), &sa8.ptr(), &(k as u32), &(r as u32), &(rp as u32), &1u32])?;
                    q.launch(&pk, pg)?;
                    // random per-channel / per-token scales for the group-free kernel
                    let sn: Vec<f32> = (0..n).map(|_| rng.f32() * 0.01 + 0.001).collect();
                    let stv: Vec<f32> = (0..rp).map(|_| rng.f32() * 0.05 + 0.001).collect();
                    let (dsn, dst) = (DeviceBuffer::from_slice(&q, &sn)?, DeviceBuffer::from_slice(&q, &stv)?);
                    let flop = 2.0 * r as f64 * n as f64 * k as f64;
                    // (rust name, rust module, reference module, reference name, rust local, rust grid, ref local, ref grid, group-free)
                    type Case<'a> = (String, &'a Module, &'a Module, String, [u32; 3], [u32; 3], [u32; 3], [u32; 3], bool);
                    macro_rules! same {
                        ($nm:expr, $rm:expr, $om:expr, $l:expr, $g:expr, $ng:expr) => {
                            (String::from($nm), $rm, $om, String::from($nm), $l, $g, $l, $g, $ng)
                        };
                    }
                    let i8c_l = [256, 1, 1];
                    let i8c_g = [r.div_ceil(256) as u32, n.div_ceil(128) as u32, 1];
                    let i8g_g = [r.div_ceil(128) as u32, n.div_ceil(128) as u32, 1];
                    let mut cases: Vec<Case<'_>> = Vec::new();
                    if !x3.is_empty() && no_check && do_time {
                        // only the X3 variants
                    } else {
                        cases.push(same!("xq_pfgemm_i8_128", &mods.pfg.0, om, [16, 1, 1], [r.div_ceil(32) as u32, (n / 32) as u32, 1], false));
                        cases.push(same!("xq_pfgemm_i8w_128", &mods.pfg.0, om, [16, 1, 1], [r.div_ceil(48) as u32, (n / 32) as u32, 1], false));
                        cases.push(same!("xq_pfgemm_i8g_128_c4t4", &mods.pfg.0, &mods.pfg.1, [256, 1, 1], [r.div_ceil(128) as u32, n.div_ceil(128) as u32, 1], false));
                        cases.push(same!("xq_pfgemm_i8c_c4t4", &mods.pfg.0, &mods.pfg.1, i8c_l, i8c_g, true));
                    }
                    for v in &x3 {
                        let c = v.rfind("_c").context("--x3 name_cCtT")?;
                        let t = v.rfind('t').context("--x3 name_cCtT")?;
                        let (vc, vt): (u32, u32) = (v[c + 2..t].parse()?, v[t + 1..].parse()?);
                        let fam = &v[..c];
                        let (tok, ng, refn) = match fam {
                            "i8c2" => (64u32, true, "xq_pfgemm_i8c_c4t4"),
                            "i8g2" => (32u32, false, "xq_pfgemm_i8g_128_c4t4"),
                            _ => anyhow::bail!("--x3: unknown family {fam}"),
                        };
                        cases.push((
                            format!("xq_pfgemm_{v}"),
                            &mods.pfg.0,
                            &mods.pfg.1,
                            refn.into(),
                            [16 * vc * vt, 1, 1],
                            [r.div_ceil((tok * vt) as usize) as u32, n.div_ceil(32 * vc as usize) as u32, 1],
                            i8c_l,
                            if ng { i8c_g } else { i8g_g },
                            ng,
                        ));
                    }
                    for (name, rm, om2, refname, rl, rg, ol, og, nogroup) in cases {
                        if n % 32 != 0 && !name.contains("_c") {
                            continue; // the untiled int8 GEMMs need N % 32 == 0
                        }
                        for f32out in [1u32, 0] {
                            let mk = |m: &Module, kn: &str, local: [u32; 3], y: &DeviceBuffer<f32>| -> Result<Kernel> {
                                let kk = m.kernel(kn, local)?;
                                let (ws, xs_) = if nogroup { (dsn.ptr(), dst.ptr()) } else { (swo.ptr(), sa8.ptr()) };
                                kk.args(&[&w8o.ptr(), &ws, &x8.ptr(), &xs_, &y.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &f32out])?;
                                Ok(kk)
                            };
                            let (ko, kr) = (mk(om2, &refname, ol, &yo)?, mk(rm, &name, rl, &yr)?);
                            let short = name.trim_start_matches("xq_pfgemm_");
                            if !no_check || !x3.is_empty() {
                                q.fill(yo.ptr(), &[0u8], yo.bytes())?;
                                q.fill(yr.ptr(), &[0xffu8], yr.bytes())?;
                                q.launch(&ko, og)?;
                                q.launch(&kr, rg)?;
                                let words = if f32out == 1 { r * n } else { r * n / 2 };
                                let nd = yo.to_vec(&q)?[..words].iter().zip(&yr.to_vec(&q)?[..words]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                                cells.push(format!("{short}-{} {}", if f32out == 1 { "f32" } else { "bf16" }, if nd == 0 { "ok".into() } else { format!("FAIL {nd}") }));
                                fails += (nd > 0) as u32;
                            }
                            if do_time && f32out == 0 {
                                let us = time_pair2(&q, [(&ko, og), (&kr, rg)], 9, 5)?;
                                let vs = if refname == name { String::new() } else { format!(" vs {}", refname.trim_start_matches("xq_pfgemm_")) };
                                cells.push(format!("{short} {:.0}/{:.0} TOP/s ({:.3}){vs}", flop / us[0] / 1e6, flop / us[1] / 1e6, us[1] / us[0]));
                            }
                        }
                    }
                    if do_i8c && i8mods.is_some() {
                        let rx = &i8mods.as_ref().unwrap().0; // Rust kx_xq{bits}
                        let kb = k / 128;
                        // weights: per-channel scale + requantized decode, checked exactly against the (bitwise-checked) int8 codes w8r / swr
                        let (dsn, dinv) = (DeviceBuffer::<f32>::zeroed(&q, n)?, DeviceBuffer::<f32>::zeroed(&q, n)?);
                        let w8c = DeviceBuffer::<u32>::zeroed(&q, n * k / 4)?;
                        let ks = rx.kernel("xq_pfscale_c", [16, 1, 1])?;
                        ks.args(&[&w.ptr(), &dsn.ptr(), &dinv.ptr(), &(k as u32)])?;
                        let kdc = rx.kernel("xq_pfdec_c128", [16, 1, 1])?;
                        kdc.args(&[&w.ptr(), &w8c.ptr(), &dinv.ptr(), &(k as u32)])?;
                        q.launch(&ks, [(n / 16) as u32, 1, 1])?;
                        q.launch(&kdc, [(n / 16) as u32, kb as u32, 1])?;
                        let (snv, invv) = (dsn.to_vec(&q)?, dinv.to_vec(&q)?);
                        let (codes, swv, w8v) = (w8r.to_vec(&q)?, swr.to_vec(&q)?, w8c.to_vec(&q)?);
                        let byte = |v: &[u32], i: usize| (v[i / 4] >> (8 * (i % 4))) as u8 as i8;
                        let sat = |x: f32| x.round_ties_even().clamp(-128.0, 127.0) as i8;
                        let (mut bad_w, mut bad_s) = (0usize, 0usize);
                        for nn in 0..n {
                            let m = (0..kb).map(|g| swv[g * n + nn].abs()).fold(0f32, f32::max) * (1u32 << (bits - 1)) as f32;
                            if m > 0.0 && ((invv[nn] as f64 * m as f64 / 127.0 - 1.0).abs() > 4e-7 || (snv[nn] as f64 * 127.0 / m as f64 - 1.0).abs() > 4e-7) {
                                bad_s += 1;
                            }
                            for kk in 0..k {
                                let qv = byte(&codes, nn * k + kk) as f32;
                                if byte(&w8v, nn * k + kk) != sat(qv * swv[(kk / 128) * n + nn] * invv[nn]) {
                                    bad_w += 1;
                                }
                            }
                        }
                        cells.push(format!("i8c-w {}", if bad_w + bad_s == 0 { "ok".into() } else { format!("FAIL w {bad_w} s {bad_s}") }));
                        // the 16x-structured decode: == xq_pfdec_c128 bit for bit
                        let w8x = DeviceBuffer::<u32>::zeroed(&q, n * k / 4)?;
                        let kdx = rx.kernel("xq_pfdec16x_c128", [16, 8, 1])?;
                        kdx.args(&[&w.ptr(), &w8x.ptr(), &dinv.ptr(), &(k as u32), &(n as u32)])?;
                        let gdx = [(n / 16) as u32, (kb / 8) as u32, 1];
                        q.launch(&kdx, gdx)?;
                        let ndx = w8x.to_vec(&q)?.iter().zip(&w8v).filter(|(a, b)| a != b).count();
                        cells.push(format!("dec16x_c {}", if ndx == 0 { "ok".into() } else { format!("FAIL {ndx}") }));
                        fails += (ndx > 0) as u32;
                        fails += (bad_w + bad_s > 0) as u32;
                        // activations: fp16 prep + block maxima (fp16 output == the reference fp16 prep), per-token scale, int8
                        let (xm, sam) = (DeviceBuffer::<u16>::zeroed(&q, k * rp)?, DeviceBuffer::<f32>::zeroed(&q, rp * kb)?);
                        let (dst2, dtinv) = (DeviceBuffer::<f32>::zeroed(&q, rp)?, DeviceBuffer::<f32>::zeroed(&q, rp)?);
                        let x8c = DeviceBuffer::<u8>::zeroed(&q, k * rp)?;
                        let kp = rx.kernel("xq_pfprep_k16m", [512, 1, 1])?;
                        kp.args(&[&dx.ptr(), &1u32, &(k as u32), &sup, &(rot as u32), &xm.ptr(), &sam.ptr(), &(k as u32), &(r as u32), &(rp as u32)])?;
                        let kt = rx.kernel("xq_pftok", [16, 1, 1])?;
                        kt.args(&[&sam.ptr(), &dst2.ptr(), &dtinv.ptr(), &(kb as u32)])?;
                        let kq = rx.kernel("xq_pfq8t", [16, 1, 1])?;
                        kq.args(&[&xm.ptr(), &dtinv.ptr(), &x8c.ptr(), &(rp as u32)])?;
                        let (gp, gt, gq) = (pg, [(rp / 16) as u32, 1, 1], [rp.div_ceil(128) as u32, (k / 32) as u32, 1]);
                        q.launch(&kp, gp)?;
                        q.launch(&kt, gt)?;
                        q.launch(&kq, gq)?;
                        // reference fp16 prep (OpenCL, i8 = 0) into its own buffer
                        let a16 = DeviceBuffer::<u16>::zeroed(&q, k * rp)?;
                        let pk16 = omod.kernel("xq_pfprep_k16", [512, 1, 1])?;
                        pk16.args(&[&dx.ptr(), &1u32, &(k as u32), &sup, &(rot as u32), &a16.ptr(), &sao.ptr(), &(k as u32), &(r as u32), &(rp as u32), &0u32])?;
                        q.launch(&pk16, pg)?;
                        let (xmv, a16v, samv, tinvv, x8v) = (xm.to_vec(&q)?, a16.to_vec(&q)?, sam.to_vec(&q)?, dtinv.to_vec(&q)?, x8c.to_vec(&q)?);
                        let nd16 = xmv.iter().zip(&a16v).filter(|(a, b)| a != b).count();
                        let (mut bad_m, mut bad_x) = (0usize, 0usize);
                        for t in 0..rp {
                            for g in 0..kb {
                                let m16 = (0..128).map(|j| f16_to_f32(xmv[(g * 128 + j) * rp + t]).abs()).fold(0f32, f32::max);
                                if (samv[t * kb + g] - m16).abs() > m16 * 1e-3 {
                                    bad_m += 1;
                                }
                            }
                            for kk in 0..k {
                                let e = sat(f16_to_f32(xmv[kk * rp + t]) * tinvv[t]);
                                if x8v[kk * rp + t] as i8 != e {
                                    bad_x += 1;
                                }
                            }
                        }
                        cells.push(format!("i8c-x {}", if nd16 + bad_m + bad_x == 0 { "ok".into() } else { format!("FAIL f16 {nd16} max {bad_m} x8 {bad_x}") }));
                        fails += (nd16 + bad_m + bad_x > 0) as u32;
                        // GEMMs: i8c2 (Rust) and the exact grouped i8g2 (Rust, on the OpenCL int8 decode + prep x8 / sa8), fp16 reference
                        let (yc, yg, yf) = (DeviceBuffer::<f32>::zeroed(&q, r * n)?, DeviceBuffer::<f32>::zeroed(&q, r * n)?, DeviceBuffer::<f32>::zeroed(&q, r * n)?);
                        let gc = mods.pfg.0.kernel("xq_pfgemm_i8c2_c4t4", [256, 1, 1])?;
                        gc.args(&[&w8c.ptr(), &dsn.ptr(), &x8c.ptr(), &dst2.ptr(), &yc.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &1u32])?;
                        let gg2 = mods.pfg.0.kernel("xq_pfgemm_i8g2_c4t4", [256, 1, 1])?;
                        gg2.args(&[&w8o.ptr(), &swo.ptr(), &x8.ptr(), &sa8.ptr(), &yg.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &1u32])?;
                        let gf = mods.pfg.1.kernel("xq_pfgemm_f16_c4t4", [256, 1, 1])?;
                        gf.args(&[&w16o.ptr(), &a16.ptr(), &yf.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &1u32])?;
                        let (ggc, ggg) = ([r.div_ceil(256) as u32, n.div_ceil(128) as u32, 1], [r.div_ceil(128) as u32, n.div_ceil(128) as u32, 1]);
                        q.launch(&gc, ggc)?;
                        q.launch(&gg2, ggg)?;
                        q.launch(&gf, ggc)?;
                        let (ycv, ygv, yfv) = (yc.to_vec(&q)?, yg.to_vec(&q)?, yf.to_vec(&q)?);
                        // exact host check of i8c2 on a sample: y = (float(sum) * sn) * st
                        let st2v = dst2.to_vec(&q)?;
                        let mut bad_y = 0usize;
                        for t in (0..r).step_by((r / 16).max(1)) {
                            for nn in (0..n).step_by((n / 64).max(1)) {
                                let mut acc = 0i32;
                                for kk in 0..k {
                                    acc += byte(&w8v, nn * k + kk) as i32 * x8v[kk * rp + t] as i8 as i32;
                                }
                                let e = acc as f32 * snv[nn] * st2v[t];
                                if e.to_bits() != ycv[t * n + nn].to_bits() {
                                    bad_y += 1;
                                }
                            }
                        }
                        cells.push(format!("i8c2-y {}", if bad_y == 0 { "ok".into() } else { format!("FAIL {bad_y}") }));
                        fails += (bad_y > 0) as u32;
                        // K-sliced: per-(channel, slice) weights + per-(token, slice) activations, exact host check of a sample
                        let mut ys_cell = String::new();
                        let ys = DeviceBuffer::<f32>::zeroed(&q, if sb > 0 { r * n } else { 1 })?;
                        let mut gs_k: Option<(Kernel, Kernel, Kernel, Kernel, Kernel)> = None;
                        if sb > 0 {
                            let nsl = (kb as u32).div_ceil(sb);
                            let (dsns, dinvs) = (DeviceBuffer::<f32>::zeroed(&q, kb * n)?, DeviceBuffer::<f32>::zeroed(&q, kb * n)?);
                            let w8s = DeviceBuffer::<u32>::zeroed(&q, n * k / 4)?;
                            let (sas, tis) = (DeviceBuffer::<f32>::zeroed(&q, rp * kb)?, DeviceBuffer::<f32>::zeroed(&q, rp * kb)?);
                            let x8s = DeviceBuffer::<u8>::zeroed(&q, k * rp)?;
                            let k1 = rx.kernel("xq_pfscale_s", [16, 1, 1])?;
                            k1.args(&[&w.ptr(), &dsns.ptr(), &dinvs.ptr(), &(k as u32), &(n as u32), &sb])?;
                            let k2 = rx.kernel("xq_pfdec16x_s128", [16, 8, 1])?;
                            k2.args(&[&w.ptr(), &w8s.ptr(), &dinvs.ptr(), &(k as u32), &(n as u32)])?;
                            let k3 = rx.kernel("xq_pftok_s", [16, 1, 1])?;
                            k3.args(&[&sam.ptr(), &sas.ptr(), &tis.ptr(), &(kb as u32), &sb])?;
                            let k4 = rx.kernel("xq_pfq8s", [16, 1, 1])?;
                            k4.args(&[&xm.ptr(), &tis.ptr(), &x8s.ptr(), &(rp as u32), &(kb as u32)])?;
                            let k5 = mods.pfg.0.kernel("xq_pfgemm_i8s_c4t4", [256, 1, 1])?;
                            k5.args(&[&w8s.ptr(), &dsns.ptr(), &x8s.ptr(), &sas.ptr(), &ys.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &1u32, &sb])?;
                            q.launch(&k1, [(n / 16) as u32, nsl, 1])?;
                            q.launch(&k2, gdx)?;
                            q.launch(&k3, gt)?;
                            q.launch(&k4, gq)?;
                            q.launch(&k5, ggc)?;
                            let (w8sv, snsv, x8sv, sasv, ysv) = (w8s.to_vec(&q)?, dsns.to_vec(&q)?, x8s.to_vec(&q)?, sas.to_vec(&q)?, ys.to_vec(&q)?);
                            let mut bad = 0usize;
                            for t in (0..r).step_by((r / 16).max(1)) {
                                for nn in (0..n).step_by((n / 64).max(1)) {
                                    let mut acc = 0f32;
                                    let mut g0 = 0usize;
                                    while g0 < kb {
                                        let g1 = (g0 + sb as usize).min(kb);
                                        let mut c = 0i32;
                                        for kk in g0 * 128..g1 * 128 {
                                            c += byte(&w8sv, nn * k + kk) as i32 * x8sv[kk * rp + t] as i8 as i32;
                                        }
                                        acc = (c as f32).mul_add(snsv[g0 * n + nn] * sasv[t * kb + g0], acc);
                                        g0 = g1;
                                    }
                                    if acc.to_bits() != ysv[t * n + nn].to_bits() {
                                        bad += 1;
                                    }
                                }
                            }
                            ys_cell = format!("i8s{sb}-y {}", if bad == 0 { "ok".into() } else { format!("FAIL {bad}") });
                            fails += (bad > 0) as u32;
                            gs_k = Some((k1, k2, k3, k4, k5));
                        }
                        if !ys_cell.is_empty() {
                            cells.push(ys_cell);
                        }
                        // rescaled-int32 sliced GEMMs: exact host check (int32 units changed at each slice start, fma-free)
                        let mut yr_v: Vec<(String, Vec<f32>)> = Vec::new();
                        let mut gr_k: Vec<(String, Kernel, [u32; 3])> = Vec::new();
                        if rb > 0 {
                            let nsl = (kb as u32).div_ceil(rb) as usize;
                            let (dsr, dir, drn) = (DeviceBuffer::<f32>::zeroed(&q, nsl * n)?, DeviceBuffer::<f32>::zeroed(&q, nsl * n)?, DeviceBuffer::<f32>::zeroed(&q, nsl * n)?);
                            let w8r_ = DeviceBuffer::<u32>::zeroed(&q, n * k / 4)?;
                            let (dstr, dtir, drt) = (DeviceBuffer::<f32>::zeroed(&q, rp * nsl)?, DeviceBuffer::<f32>::zeroed(&q, rp * nsl)?, DeviceBuffer::<f32>::zeroed(&q, rp * nsl)?);
                            let x8r_ = DeviceBuffer::<u8>::zeroed(&q, k * rp)?;
                            let k1 = rx.kernel("xq_pfscale_r", [16, 1, 1])?;
                            k1.args(&[&w.ptr(), &dsr.ptr(), &dir.ptr(), &drn.ptr(), &(k as u32), &(n as u32), &rb, &rcodes])?;
                            let k2 = rx.kernel("xq_pfdec16x_r128", [16, 8, 1])?;
                            k2.args(&[&w.ptr(), &w8r_.ptr(), &dir.ptr(), &(k as u32), &(n as u32), &rb])?;
                            let k3 = rx.kernel("xq_pftok_r", [16, 1, 1])?;
                            k3.args(&[&sam.ptr(), &dstr.ptr(), &dtir.ptr(), &drt.ptr(), &(kb as u32), &rb])?;
                            let k4 = rx.kernel("xq_pfq8r", [16, 1, 1])?;
                            k4.args(&[&xm.ptr(), &dtir.ptr(), &x8r_.ptr(), &(rp as u32), &(nsl as u32), &rb])?;
                            q.launch(&k1, [(n / 16) as u32, 1, 1])?;
                            q.launch(&k2, gdx)?;
                            q.launch(&k3, gt)?;
                            q.launch(&k4, gq)?;
                            let (w8v_, snv_, rnv_, x8v_, stv_, rtv_) =
                                (w8r_.to_vec(&q)?, dsr.to_vec(&q)?, drn.to_vec(&q)?, x8r_.to_vec(&q)?, dstr.to_vec(&q)?, drt.to_vec(&q)?);
                            for (kname, u) in [("xq_pfgemm_i8r_c4t4", 4u32), ("xq_pfgemm_i8r3_c4t4", 3u32)] {
                                let yy = DeviceBuffer::<f32>::zeroed(&q, r * n)?;
                                let kk = mods.pfg.0.kernel(kname, [256, 1, 1])?;
                                kk.args(&[&w8r_.ptr(), &dsr.ptr(), &drn.ptr(), &x8r_.ptr(), &dstr.ptr(), &drt.ptr(), &yy.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &1u32, &rb])?;
                                let grid = [r.div_ceil(64 * u as usize) as u32, n.div_ceil(128) as u32, 1];
                                q.launch(&kk, grid)?;
                                let yv = yy.to_vec(&q)?;
                                let mut bad = 0usize;
                                for t in (0..r).step_by((r / 16).max(1)) {
                                    for nn in (0..n).step_by((n / 64).max(1)) {
                                        let mut ia = 0i32;
                                        for j in 0..nsl {
                                            let rr = rnv_[j * n + nn] * rtv_[t * nsl + j];
                                            ia = (ia as f32 * rr).round_ties_even().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
                                            let (g0, g1) = (j * rb as usize, ((j + 1) * rb as usize).min(kb));
                                            for kk2 in g0 * 128..g1 * 128 {
                                                ia = ia.wrapping_add(byte(&w8v_, nn * k + kk2) as i32 * x8v_[kk2 * rp + t] as i8 as i32);
                                            }
                                        }
                                        let e = ia as f32 * snv_[(nsl - 1) * n + nn] * stv_[t * nsl + nsl - 1];
                                        if e.to_bits() != yv[t * n + nn].to_bits() {
                                            bad += 1;
                                        }
                                    }
                                }
                                let short = kname.trim_start_matches("xq_pfgemm_").trim_end_matches("_c4t4").to_string();
                                cells.push(format!("{short}{}{rb}-y {}", if rcodes != 0 { "c" } else { "" }, if bad == 0 { "ok".into() } else { format!("FAIL {bad}") }));
                                fails += (bad > 0) as u32;
                                kk.args(&[&w8r_.ptr(), &dsr.ptr(), &drn.ptr(), &x8r_.ptr(), &dstr.ptr(), &drt.ptr(), &yy.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &0u32, &rb])?;
                                yr_v.push((short.clone(), yv));
                                gr_k.push((short, kk, grid));
                            }
                        }
                        let rel = |a: &[f32]| {
                            let (mut e, mut s) = (0f64, 0f64);
                            for (x, y) in a.iter().zip(&yfv) {
                                e += ((x - y) as f64).powi(2);
                                s += (*y as f64).powi(2);
                            }
                            (e / s).sqrt() * 100.0
                        };
                        cells.push(format!("err vs fp16: i8c2 {:.3}% i8g2 {:.3}%{}", rel(&ycv), rel(&ygv),
                            if sb > 0 { format!(" i8s{sb} {:.3}%", rel(&ys.to_vec(&q)?)) } else { String::new() }));
                        for (nm, yv) in &yr_v {
                            cells.push(format!("err {nm}{}{rb} {:.3}%", if rcodes != 0 { "c" } else { "" }, rel(yv)));
                        }
                        if do_time {
                            gc.args(&[&w8c.ptr(), &dsn.ptr(), &x8c.ptr(), &dst2.ptr(), &yc.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &0u32])?;
                            for (nm, kk, grid) in &gr_k {
                                let us = time_pair2(&q, [(&gc, ggc), (kk, *grid)], 9, 5)?;
                                let fl = 2.0 * r as f64 * n as f64 * k as f64;
                                cells.push(format!("gemm i8c2 {:.0} -> {nm}{rb} {:.0} TOP/s", fl / us[0] / 1e6, fl / us[1] / 1e6));
                            }
                        }
                        if let (Some((_, _, _, _, k5)), true) = (&gs_k, do_time) {
                            gc.args(&[&w8c.ptr(), &dsn.ptr(), &x8c.ptr(), &dst2.ptr(), &yc.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &0u32])?;
                            let us = time_pair2(&q, [(&gc, ggc), (k5, ggc)], 9, 5)?;
                            let fl = 2.0 * r as f64 * n as f64 * k as f64;
                            cells.push(format!("gemm i8c2 {:.0} -> i8s{sb} {:.0} TOP/s", fl / us[0] / 1e6, fl / us[1] / 1e6));
                        }
                        if do_time {
                            // the whole projection: weight path (scale + decode + GEMM) and activation path, Rust int8 vs Rust fp16
                            let kdf = rmod.kernel("xq_pfdec16x_128", [16, 8, 1])?;
                            kdf.args(&[&w.ptr(), &w16r.ptr(), &(k as u32), &(n as u32)])?;
                            let gfr = mods.pfg.0.kernel("xq_pfgemm_f16_c4t4", [256, 1, 1])?;
                            gfr.args(&[&w16r.ptr(), &a16.ptr(), &yf.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &0u32])?;
                            let pf16 = rmod.kernel("xq_pfprep_k16", [512, 1, 1])?;
                            pf16.args(&[&dx.ptr(), &1u32, &(k as u32), &sup, &(rot as u32), &a16.ptr(), &sar.ptr(), &(k as u32), &(r as u32), &(rp as u32), &0u32])?;
                            gc.args(&[&w8c.ptr(), &dsn.ptr(), &x8c.ptr(), &dst2.ptr(), &yc.ptr(), &(r as u32), &(rp as u32), &(k as u32), &(n as u32), &(n as u32), &0u32, &0u32])?;
                            let us_dc = time_pair2(&q, [(&kdc, [(n / 16) as u32, kb as u32, 1]), (&kdx, gdx)], 9, 10)?;
                            let us_df = time_pair(&q, [&kdf, &kdf], dg, 4, 10)?;
                            cells.push(format!("dec fp16 {:.0} us, i8 c128 {:.0} -> 16x_c {:.0} us", us_df[0], us_dc[0], us_dc[1]));
                            let mut tt_ = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                            for rep in 0..=9 {
                                let seqs: [Vec<(&Kernel, [u32; 3])>; 4] = [
                                    vec![(&kdf, dg), (&gfr, ggc)],
                                    vec![(&ks, [(n / 16) as u32, 1, 1]), (&kdx, gdx), (&gc, ggc)],
                                    vec![(&pf16, pg)],
                                    vec![(&kp, gp), (&kt, gt), (&kq, gq)],
                                ];
                                for (i, sq) in seqs.iter().enumerate() {
                                    q.sync()?;
                                    let t0 = now_ns();
                                    for _ in 0..3 {
                                        for (kk, g) in sq {
                                            q.launch(kk, *g)?;
                                        }
                                    }
                                    q.sync()?;
                                    if rep > 0 {
                                        tt_[i].push((now_ns() - t0) as f64 / 3.0 / 1e3);
                                    }
                                }
                            }
                            let md: Vec<f64> = tt_.iter().map(|v| median(v)).collect();
                            cells.push(format!(
                                "proj fp16 {:.0} us -> i8c2 {:.0} us ({:.2}x), act fp16 {:.0} -> i8 {:.0} us",
                                md[0], md[1], md[0] / md[1], md[2], md[3]
                            ));
                        }
                    }
                }
                line.push(format!("R{r}: {}", cells.join(", ")));
            }
            println!("  {}", line.join("\n  "));
        }
    }
    if fails > 0 {
        println!("FAILURES: {fails}");
        std::process::exit(1);
    }
    println!("ALL BITWISE CHECKS PASS");
    let _ = Path::new("");
    Ok(())
}
