//! X1: the Rust XQ decode GEMV (kernels-xe/kx-xq) against arc-engine's OpenCL `xq{3,4,5,6,8}.spv` on real checkpoint
//! tensors, with identical inputs and launch geometry (xq.zig `Layer.mv` / `prepRows`):
//! * prepass: Rust `xq_prep` == OpenCL `xq_prep` (xt fp16 + xs fp32, bitwise) for every row count;
//! * matvec: Rust `xq_mv_*` == OpenCL `xq_mv_*` (fp32 out and bf16 out, bitwise) on the same prepared rows;
//! * row invariance: every row of an m-row window == the same row run alone (Rust prep + Rust mv), bitwise;
//! * timing (--time): per call, weights cycled over >= 256 MB of copies, median of 5 runs of >= 30 calls, GB/s of packed
//!   bytes; Rust / OpenCL ratio (target >= 0.95, gate 0.85).
//! usage: xq-check [--model DIR] [--bits 4,5] [--rows 1,2,..] [--shapes gate_up,down,..] [--time] [--trows 1,2,4,8,16] [--nb 1|2]
use anyhow::{bail, Context, Result};
use fe_kx_tests::*;
use fe_xq::pack::{split_for, variant_for, VARIANTS};
use fe_xq::{Checkpoint, Rotate};
use fe_ze::*;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

struct Mods {
    rust: Module,
    ocl: Module,
}

/// Shape label of a Qwen3.8 projection.
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

struct Case {
    bits: u32,
    g: u32,
    n: usize,
    k: usize,
    rot: bool,
    w: DeviceBuffer<u32>,
    su: Option<DeviceBuffer<u16>>,
    split: u32,
    name: String,
}

struct Scratch {
    xt: DeviceBuffer<u16>,
    xs: DeviceBuffer<f32>,
}

fn scratch(q: &Queue, k: usize) -> Result<Scratch> {
    Ok(Scratch { xt: DeviceBuffer::zeroed(q, 16 * k)?, xs: DeviceBuffer::zeroed(q, 16 * k / 32)? })
}

/// xq_prep of rows [r0, r0 + m) of x (bf16 [16][k]) into sc (xq.zig prepRows).
fn prep(q: &Queue, m: &Module, c: &Case, x: &DeviceBuffer<u16>, r0: usize, rows: u32, sc: &Scratch) -> Result<()> {
    let vi = variant_for(rows)?;
    let (mw, nr) = VARIANTS[vi];
    let k = m.kernel("xq_prep", [16, 1, 1])?;
    let su = c.su.as_ref().map(|s| s.ptr()).unwrap_or(sc.xs.ptr());
    k.args(&[&x.at(r0 * c.k), &1u32, &(c.k as u32), &su, &(c.rot as u32), &sc.xt.ptr(), &sc.xs.ptr(), &(c.k as u32), &rows, &mw])?;
    q.launch(&k, [(c.k / 128) as u32, mw * nr, 1])?;
    Ok(())
}

/// The decode matvec kernel for `rows` rows with its group size and arguments set (xq.zig Layer.mv).
fn mv_kernel(m: &Module, c: &Case, w: DevPtr, sc: &Scratch, rows: u32, y: DevPtr, f32out: bool, nb: u32) -> Result<(Kernel, u32)> {
    mv_kernel_r(false, m, c, w, sc, rows, y, f32out, nb)
}

/// As [`mv_kernel`]; `rust` with XQ_P=1: the Rust module's row-group-parallel xq_mv_128_8x2p_1 for the 8x2 variant (bit-identical
/// to xq_mv_128_8x2_1, two sub-groups a K split).
#[allow(clippy::too_many_arguments)]
fn mv_kernel_r(rust: bool, m: &Module, c: &Case, w: DevPtr, sc: &Scratch, rows: u32, y: DevPtr, f32out: bool, nb: u32) -> Result<(Kernel, u32)> {
    let vi = variant_for(rows)?;
    let (mw, nr) = VARIANTS[vi];
    let nb = if nb == 2 && c.n % 32 == 0 { 2 } else { 1 };
    let s = c.split;
    let par = rust && nr == 2 && nb == 1 && c.g == 128 && std::env::var("XQ_P").map(|v| v == "1").unwrap_or(false);
    let k = if par {
        m.kernel("xq_mv_128_8x2p_1", [2 * s * 16, 1, 1])?
    } else {
        m.kernel(&format!("xq_mv_{}_{}x{}_{}", c.g, mw, nr, nb), [s * 16, 1, 1])?
    };
    let mt = (mw * nr) as usize;
    let red = if s > 1 || par { s as usize * nb as usize * mt * 16 * 4 } else { 4 };
    k.args(&[
        &sc.xt.ptr(),
        &sc.xs.ptr(),
        &w,
        &y,
        &(c.k as u32),
        &(c.n as u32),
        &rows,
        &0u32,
        &(c.n as u32),
        &(f32out as u32),
        &Local(red),
    ])?;
    Ok((k, c.n as u32 / (16 * nb)))
}

fn first_diff<T: PartialEq + Copy + std::fmt::Debug>(a: &[T], b: &[T]) -> Option<(usize, T, T)> {
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| (i, a[i], b[i]))
}

fn f32_bits_eq(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut opt = |name: &str| -> Option<String> {
        let i = args.iter().position(|a| a == name)?;
        args.remove(i);
        if i < args.len() && !args[i].starts_with("--") { Some(args.remove(i)) } else { Some(String::new()) }
    };
    let model = PathBuf::from(opt("--model").or_else(|| std::env::var("FE_XQ_MODEL").ok()).expect("--model DIR (or FE_XQ_MODEL): an XQ checkpoint folder"));
    let list = |s: Option<String>, d: &str| -> Vec<u32> { s.unwrap_or(d.into()).split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect() };
    let bits_want = list(opt("--bits"), "3,4,5,6,8");
    let rows = list(opt("--rows"), "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16");
    let trows = list(opt("--trows"), "1,2,4,8,16");
    let shapes_want: Vec<String> = opt("--shapes").map(|s| s.split(',').map(String::from).collect()).unwrap_or_default();
    let do_time = opt("--time").is_some();
    let no_check = opt("--nocheck").is_some();
    let nb_req = opt("--nb").map(|s| s.parse().unwrap()).unwrap_or(1u32);
    let ckpt = Checkpoint::open(&model).with_context(|| format!("opening {model:?}"))?;

    let ze = Ze::new()?;
    let q = Queue::new(&ze)?;
    let mut mods: BTreeMap<u32, Mods> = BTreeMap::new();
    let mut rng = Rng(0xdead_beef_1234_5678);
    let mut fails = 0;
    let mut timing = Vec::new();

    // one representative tensor per (n, k, bits, group, rotation)
    let mut reps: BTreeMap<(usize, usize, u32, u32, bool), String> = BTreeMap::new();
    for (name, m) in &ckpt.cfg.tensors {
        reps.entry((m.n, m.k, m.bits, m.group, m.rotate == Rotate::Had128In)).or_insert(name.clone());
    }
    for ((n, k, bits, g, rot), name) in reps {
        if !bits_want.contains(&bits) || (!shapes_want.is_empty() && !shapes_want.contains(&label(n, k))) {
            continue;
        }
        if !mods.contains_key(&bits) {
            let t0 = now_ns();
            let rust = Module::from_spirv(&ze, kx_xq(bits), "")?;
            let t1 = now_ns();
            let ocl = arc_module(&ze, &format!("xq{bits}.spv"), "")?;
            println!("module xq{bits}: rust build {:.1} s, ocl build {:.1} s", (t1 - t0) as f64 / 1e9, (now_ns() - t1) as f64 / 1e9);
            mods.insert(bits, Mods { rust, ocl });
        }
        let md = &mods[&bits];
        let t = ckpt.xq(&name)?;
        let packed = t.pack()?;
        let c = Case {
            bits,
            g,
            n,
            k,
            rot,
            w: DeviceBuffer::from_slice(&q, &packed)?,
            su: t.su.as_ref().map(|s| DeviceBuffer::from_slice(&q, s)).transpose()?,
            // XQ_SPLIT=S: that K split for every shape (capped at K / G); else split_for with XQ_TARGET
            split: match std::env::var("XQ_SPLIT").ok().and_then(|v| v.parse::<u32>().ok()) {
                Some(sp) => sp.min((k / g as usize) as u32).max(1),
                None => split_for(n, k, g, std::env::var("XQ_TARGET").ok().and_then(|v| v.parse().ok())),
            },
            name: name.clone(),
        };
        let bytes = packed.len() * 4;
        drop(packed);
        println!("\n== {} {}x{} int{} g{} {} S={} ({:.1} MB): {}", label(n, k), n, k, bits, g, if rot { "rot" } else { "norot" }, c.split, bytes as f64 / 1e6, c.name);

        // activations: bf16 rows, roughly unit scale with a few outliers
        let x: Vec<u16> = (0..16 * k)
            .map(|i| {
                let v = (rng.f32() * 2.0 - 1.0) * if i % 97 == 0 { 8.0 } else { 1.0 };
                f32_to_bf16(v)
            })
            .collect();
        let dx = DeviceBuffer::from_slice(&q, &x)?;
        let (sa, sr, s1) = (scratch(&q, k)?, scratch(&q, k)?, scratch(&q, k)?);
        let ya = DeviceBuffer::<f32>::zeroed(&q, 16 * n)?;
        let yr = DeviceBuffer::<f32>::zeroed(&q, 16 * n)?;
        let y1 = DeviceBuffer::<f32>::zeroed(&q, 16 * n)?;
        if !no_check {
            let mut line = Vec::new();
            for &m in &rows {
                let mut ok = true;
                let mut why = String::new();
                // prepass: both from the same x
                prep(&q, &md.ocl, &c, &dx, 0, m, &sa)?;
                prep(&q, &md.rust, &c, &dx, 0, m, &sr)?;
                let (xta, xtr) = (sa.xt.to_vec(&q)?, sr.xt.to_vec(&q)?);
                let (xsa, xsr) = (sa.xs.to_vec(&q)?, sr.xs.to_vec(&q)?);
                if let Some((i, a, b)) = first_diff(&xta, &xtr) {
                    ok = false;
                    why += &format!(" xt[{i}] {a:04x}/{b:04x}");
                }
                if f32_bits_eq(&xsa, &xsr) > 0 {
                    ok = false;
                    why += &format!(" xs {} differ", f32_bits_eq(&xsa, &xsr));
                }
                // matvec on the OpenCL-prepared rows, fp32 out
                q.fill(ya.ptr(), &[0u8], ya.bytes())?;
                q.fill(yr.ptr(), &[0xffu8], yr.bytes())?;
                let (ka, ga) = mv_kernel(&md.ocl, &c, c.w.ptr(), &sa, m, ya.ptr(), true, nb_req)?;
                q.launch(&ka, [ga, 1, 1])?;
                let (kr, gr) = mv_kernel_r(true, &md.rust, &c, c.w.ptr(), &sa, m, yr.ptr(), true, nb_req)?;
                q.launch(&kr, [gr, 1, 1])?;
                let (va, vr) = (ya.to_vec(&q)?, yr.to_vec(&q)?);
                let nd = f32_bits_eq(&va[..m as usize * n], &vr[..m as usize * n]);
                if nd > 0 {
                    ok = false;
                    let i = (0..m as usize * n).find(|&i| va[i].to_bits() != vr[i].to_bits()).unwrap();
                    why += &format!(" y {nd} differ (first [{}][{}] {:e} vs {:e})", i / n, i % n, va[i], vr[i]);
                }
                // bf16 output (one call)
                if m == 1 {
                    let yb_a = DeviceBuffer::<u16>::zeroed(&q, n)?;
                    let yb_r = DeviceBuffer::<u16>::zeroed(&q, n)?;
                    let (ka, ga) = mv_kernel(&md.ocl, &c, c.w.ptr(), &sa, 1, yb_a.ptr(), false, nb_req)?;
                    q.launch(&ka, [ga, 1, 1])?;
                    let (kr, gr) = mv_kernel_r(true, &md.rust, &c, c.w.ptr(), &sa, 1, yb_r.ptr(), false, nb_req)?;
                    q.launch(&kr, [gr, 1, 1])?;
                    if yb_a.to_vec(&q)? != yb_r.to_vec(&q)? {
                        ok = false;
                        why += " bf16-out differs";
                    }
                }
                // row invariance of the Rust path: row r of the m-row window == row r alone (Rust prep + Rust mv)
                if m > 1 {
                    let (kr, gr) = mv_kernel_r(true, &md.rust, &c, c.w.ptr(), &sr, m, yr.ptr(), true, nb_req)?;
                    q.launch(&kr, [gr, 1, 1])?;
                    let vw = yr.to_vec(&q)?;
                    let mut bad_rows = 0;
                    for r in 0..m as usize {
                        prep(&q, &md.rust, &c, &dx, r, 1, &s1)?;
                        let (k1, g1) = mv_kernel_r(true, &md.rust, &c, c.w.ptr(), &s1, 1, y1.ptr(), true, nb_req)?;
                        q.launch(&k1, [g1, 1, 1])?;
                        let v1 = y1.to_vec(&q)?;
                        if f32_bits_eq(&v1[..n], &vw[r * n..(r + 1) * n]) > 0 {
                            bad_rows += 1;
                        }
                    }
                    if bad_rows > 0 {
                        ok = false;
                        why += &format!(" {bad_rows} rows not invariant");
                    }
                }
                line.push(format!("m{m}:{}", if ok { "ok".to_string() } else { format!("FAIL{why}") }));
                if !ok {
                    fails += 1;
                }
            }
            println!("  bitwise vs OpenCL (prep, mv f32/bf16) + row invariance: {}", line.join(" "));
            let p = mv_kernel_r(true, &md.rust, &c, c.w.ptr(), &sa, 1, ya.ptr(), true, 1)?.0.props()?;
            let po = mv_kernel(&md.ocl, &c, c.w.ptr(), &sa, 1, ya.ptr(), true, 1)?.0.props()?;
            println!("  m=1 kernel: rust spill {} B private {} B | ocl spill {} B private {} B", p.spill_bytes, p.private_bytes, po.spill_bytes, po.private_bytes);
        }

        if do_time {
            // weights cycled over >= 256 MB: copies of the packed tensor
            let copies = (256usize << 20).div_ceil(bytes).max(1);
            let mut ws = vec![];
            for _ in 1..copies {
                let b = DeviceBuffer::<u32>::alloc(&ze, bytes / 4)?;
                q.copy_dtod(b.ptr(), c.w.ptr(), bytes)?;
                ws.push(b);
            }
            q.sync()?;
            let wptrs: Vec<DevPtr> = std::iter::once(c.w.ptr()).chain(ws.iter().map(|b| b.ptr())).collect();
            let calls = (30usize).max(copies * 2).div_ceil(wptrs.len()) * wptrs.len();
            let mut cells = Vec::new();
            for &m in &trows {
                prep(&q, &md.ocl, &c, &dx, 0, m, &sa)?;
                // OpenCL and Rust alternate rep by rep (ABAB...), so clock / thermal drift hits both alike; median of
                // XQ_TIME_REPS (default 11) runs of `calls` launches each, after one warm-up run of each
                let ks: Vec<Vec<(Kernel, u32)>> = [(false, &md.ocl), (true, &md.rust)]
                    .iter()
                    .map(|(ru, mo)| wptrs.iter().map(|&w| mv_kernel_r(*ru, mo, &c, w, &sa, m, ya.ptr(), false, nb_req)).collect::<Result<_>>())
                    .collect::<Result<_>>()?;
                let reps: usize = std::env::var("XQ_TIME_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(11);
                let mut samples = [Vec::new(), Vec::new()];
                for rep in 0..=reps {
                    for i in 0..2 {
                        q.sync()?;
                        let t0 = now_ns();
                        for j in 0..calls {
                            let (k, g) = &ks[i][j % ks[i].len()];
                            q.launch(k, [*g, 1, 1])?;
                        }
                        q.sync()?;
                        if rep > 0 {
                            samples[i].push((now_ns() - t0) as f64 / calls as f64 / 1e3);
                        }
                    }
                }
                let us = [median(&samples[0]), median(&samples[1])];
                let gbs = |u: f64| bytes as f64 / u / 1e3;
                cells.push(format!("m{m} ocl {:.1}us {:.0} | rust {:.1}us {:.0} GB/s ({:.3}x)", us[0], gbs(us[0]), us[1], gbs(us[1]), us[0] / us[1]));
                timing.push((label(n, k), bits, m, us[0], us[1], bytes));
            }
            println!("  time: {}", cells.join("\n        "));
        }
        let _ = Arc::strong_count(&ze);
    }
    if do_time {
        println!("\nsummary (rust time / ocl time per call; >1.0 = rust slower):");
        let worst = timing.iter().map(|t| t.4 / t.3).fold(0f64, f64::max);
        for (l, b, m, uo, ur, by) in &timing {
            println!("  {l:<8} int{b} m{m:<2} ocl {uo:>8.1} us ({:>4.0} GB/s)  rust {ur:>8.1} us ({:>4.0} GB/s)  {:.3}", *by as f64 / uo / 1e3, *by as f64 / ur / 1e3, ur / uo);
        }
        println!("worst rust/ocl time ratio: {worst:.3}");
    }
    if fails > 0 {
        println!("FAILURES: {fails}");
        std::process::exit(1);
    }
    if !no_check {
        println!("ALL BITWISE CHECKS PASS");
    }
    if bits_want.is_empty() {
        bail!("no widths");
    }
    Ok(())
}
