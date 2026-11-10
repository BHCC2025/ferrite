//! X0 ceilings, Rust kernels (kx-bench) and arc-engine's OpenCL kernels (the same probes, baselines only) side by side
//! in one process with one harness: DRAM streaming read (GB/s), load primitives from a cache-resident region (TB/s),
//! matrix engine (TFLOP/s / TOP/s). Host wall time of launch + sync, as the arc-engine benches measure.
//! usage: kx-bench [bw|lb|me|dpas ...]   (default: all; run through b70run)
use fe_kx_tests::*;
use fe_ze::*;

fn time(q: &Queue, k: &Kernel, groups: u32, reps: usize) -> Result<Vec<f64>> {
    let mut v = Vec::new();
    for _ in 0..reps {
        let t0 = now_ns();
        q.launch(k, [groups, 1, 1])?;
        q.sync()?;
        v.push((now_ns() - t0) as f64);
    }
    Ok(v)
}
fn best(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::MAX, f64::min)
}

fn main() -> Result<()> {
    let want: Vec<String> = std::env::args().skip(1).collect();
    let on = |s: &str| want.is_empty() || want.iter().any(|w| w == s);
    let ze = Ze::new()?;
    let q = Queue::new(&ze)?;
    let grf = "-cl-intel-256-GRF-per-thread";
    let rust = Module::from_spirv(&ze, KX_BENCH, grf)?;
    let rust_plain = Module::from_spirv(&ze, KX_BENCH, "")?;
    let mut rng = Rng(0x1234_5678_9abc_def1);

    if on("bw") {
        // ggml_quant.cl bw_probe: 512 MB of random data, sgs sub-groups each streaming 512 MB / sgs (module without flags,
        // as xpu-ggml-test --probe builds it)
        let bytes: usize = 512 << 20;
        let host: Vec<u32> = (0..bytes / 4).map(|_| rng.next_u32()).collect();
        let w = DeviceBuffer::from_slice(&q, &host)?;
        drop(host);
        let out = DeviceBuffer::<u32>::zeroed(&q, 16)?;
        let ocl = arc_module(&ze, "ggml_quant.spv", "")?;
        println!("DRAM streaming read, 512 MB random (GB/s, best of 5):");
        println!("  {:<8} {:>10} {:>10} {:>12}", "sgs", "ocl probe", "rust probe", "rust blk8");
        for sgs in [1024u32, 4096, 16384] {
            let units = (bytes / 16 / sgs as usize) as u32;
            let ko = ocl.kernel("bw_probe", [64, 1, 1])?;
            ko.args(&[&w.ptr(), &out.ptr(), &units])?;
            let kr = rust_plain.kernel("bw_probe", [64, 1, 1])?;
            kr.args(&[&w.ptr(), &out.ptr(), &units])?;
            let kb = rust_plain.kernel("bw_blk", [64, 1, 1])?;
            kb.args(&[&w.ptr(), &out.ptr(), &(units * 4)])?;
            let mut r = [0f64; 3];
            for (i, k) in [&ko, &kr, &kb].iter().enumerate() {
                time(&q, k, sgs / 4, 1)?;
                r[i] = bytes as f64 / best(&time(&q, k, sgs / 4, 5)?);
            }
            println!("  {:<8} {:>10.0} {:>10.0} {:>12.0}", sgs, r[0], r[1], r[2]);
        }
    }

    if on("lb") {
        // qwen_lbench.cl: 4 MB cache-resident region, 4096 iterations, TB/s of delivered bytes (256-GRF build)
        let host: Vec<u8> = (0..4usize << 20).map(|i| (i.wrapping_mul(31)) as u8).collect();
        let dp = DeviceBuffer::from_slice(&q, &host)?;
        let out = DeviceBuffer::<u32>::zeroed(&q, 1 << 22)?;
        let ocl = arc_module(&ze, "qwen_lbench.spv", grf)?;
        let cases: [(&str, u32, u32, bool); 6] = [
            ("lb_blk8", 512, 8191, false),
            ("lb_vl8", 256, 16383, false),
            ("lb_vl16", 1024, 4095, false),
            ("lb_2d_8r16x2c", 512, 8191, true),
            ("lb_2d_32r16x2c", 2048, 2047, true),
            ("lb_2d_tr16x2c", 1024, 4095, true),
        ];
        let iters = 4096u32;
        println!("load primitives, 4 MB region (TB/s, best of 4):");
        for wgs in [1024u32, 4096, 16384] {
            println!("  {wgs} sub-groups:   {:>8} {:>8}", "ocl", "rust");
            for (name, bytes, mask, two_d) in cases {
                let mut r = [0f64; 2];
                for (i, m) in [&ocl, &rust].iter().enumerate() {
                    let k = m.kernel(name, [16, 1, 1])?;
                    k.args(&[&dp.ptr(), &out.ptr(), &iters, &mask])?;
                    if two_d {
                        k.arg(4, &4096i32)?.arg(5, &1024i32)?.arg(6, &4096i32)?;
                    }
                    let b = best(&time(&q, &k, wgs, 4)?);
                    r[i] = wgs as f64 * iters as f64 * bytes as f64 / b / 1e3;
                }
                println!("    {:<16} {:>8.2} {:>8.2}", name, r[0], r[1]);
            }
        }
    }

    if on("me") {
        // qwen_mebench.cl: random fp16/bf16 operands in registers, 16 DPAS an iteration, median of 5 after a warm-up
        let host: Vec<u32> = (0..4096 * 256)
            .map(|_| {
                let h = |r: &mut Rng| 0x2000 + (r.next_u32() % 0x1c00) | if r.next_u32() & 1 == 1 { 0x8000 } else { 0 };
                h(&mut rng) | (h(&mut rng) << 16)
            })
            .collect();
        let src = DeviceBuffer::from_slice(&q, &host)?;
        let out = DeviceBuffer::<f32>::zeroed(&q, 1 << 22)?;
        let ocl = arc_module(&ze, "qwen_mebench.spv", grf)?;
        let iters = 2048u32;
        println!("matrix engine, random operands (T/s, median of 5):");
        println!("  {:<8} {:>6} {:>8} {:>8}", "kernel", "wgs", "ocl", "rust");
        for name in ["me_f16", "me_bf16", "me_i8"] {
            let per = if name == "me_i8" { 8192.0 } else { 4096.0 };
            for wgs in [256u32, 1024, 4096] {
                let mut r = [0f64; 2];
                for (i, m) in [&ocl, &rust].iter().enumerate() {
                    let k = m.kernel(name, [64, 1, 1])?;
                    k.args(&[&src.ptr(), &out.ptr(), &iters])?;
                    let t = time(&q, &k, wgs, 6)?;
                    let flop = wgs as f64 * 4.0 * iters as f64 * 16.0 * per;
                    r[i] = flop / median(&t[1..]) / 1e3;
                }
                println!("  {:<8} {:>6} {:>8.1} {:>8.1}", name, wgs, r[0], r[1]);
            }
        }
    }

    if on("dpas") {
        // qwen_dpasbench.cl: dpas_loop (operands in registers) and dpas_tile (32 x 64 micro-tile from a small buffer)
        let n_a = 64 * 4 * 16 * 8;
        let host: Vec<u16> = (0..64 * 4 * 128 * 2).map(|i| 0x3c00 + (i & 0x3f) as u16).collect();
        let mut ha = host[..n_a].to_vec();
        ha.resize(n_a + 2048, 0);
        let da = DeviceBuffer::from_slice(&q, &ha)?;
        let db = DeviceBuffer::from_slice(&q, &host)?;
        let out = DeviceBuffer::<f32>::zeroed(&q, 1 << 22)?;
        let ocl = arc_module(&ze, "qwen_dpasbench.spv", grf)?;
        let iters = 4096u32;
        println!("DPAS (TFLOP/s, best of 4):");
        println!("  {:<10} {:>6} {:>8} {:>8}", "kernel", "wgs", "ocl", "rust");
        for wgs in [64u32, 256, 1024, 4096] {
            for which in ["dpas_loop", "dpas_tile"] {
                let mut r = [0f64; 2];
                for (i, m) in [&ocl, &rust].iter().enumerate() {
                    let k = m.kernel(which, [64, 1, 1])?;
                    if which == "dpas_loop" {
                        k.args(&[&out.ptr(), &iters])?;
                    } else {
                        k.args(&[&da.ptr(), &db.ptr(), &out.ptr(), &(iters / 4)])?;
                    }
                    let b = best(&time(&q, &k, wgs, 4)?);
                    let (per_it, n_it) = if which == "dpas_loop" { (8.0, iters as f64) } else { (16.0, (iters / 4) as f64) };
                    r[i] = wgs as f64 * 4.0 * n_it * per_it * 4096.0 / b / 1e3;
                }
                println!("  {:<10} {:>6} {:>8.1} {:>8.1}", which, wgs, r[0], r[1]);
            }
        }
    }
    // spill / register report of the Rust kernels
    for name in ["bw_probe", "bw_blk", "lb_blk8", "lb_2d_32r16x2c", "me_f16", "me_i8", "dpas_loop", "dpas_tile"] {
        let p = rust.kernel(name, [16, 1, 1])?.props()?;
        if p.spill_bytes > 0 || p.private_bytes > 0 {
            println!("note: rust {name}: spill {} B, private {} B", p.spill_bytes, p.private_bytes);
        }
    }
    Ok(())
}
