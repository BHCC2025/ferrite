//! Prompt-GEMM bench at the 27B's fused shapes: Rust k-fp4 GEMM vs cuBLASLt on the same quantized inputs.
//!
//! `fp4-bench [model_dir] [iters=20] [m list, default 512,2048,4096,8192]` (model_dir default FERRITE_NVFP4_DIR;
//! run under the GPU lock, tools/site.sh). Prints a markdown table: GEMM-only TFLOPS for both, plus the activation
//! quantizer time (shared by both paths).
use std::path::PathBuf;
use anyhow::Result;
use fe_cuda::{Context, DeviceBuffer, Event, Stream};
use fe_fp4::format::f32_to_bf16;
use fe_fp4::{lt::Lt, Checkpoint, Kernels, Kind, PromptLinear, Workspace};

/// Mean milliseconds of `f` over `iters` back-to-back launches (after 3 warm-up calls).
fn time(s: &Stream, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f32> {
    for _ in 0..3 {
        f()?;
    }
    let (a, b) = (Event::new()?, Event::new()?);
    s.record(&a)?;
    for _ in 0..iters {
        f()?;
    }
    s.record(&b)?;
    b.sync()?;
    Ok(b.elapsed_since(&a)? / iters as f32)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(PathBuf::from).unwrap_or_else(fe_site::nvfp4_dir);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let ms: Vec<usize> = args.get(3).map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or(vec![512, 2048, 4096, 8192]);
    let only: Option<String> = std::env::var("FE_BENCH_ONLY").ok();
    let _ctx = Context::new(0)?;
    let kern = Kernels::load()?;
    let lt = Lt::new()?;
    let s = Stream::new()?;
    let ck = Checkpoint::open(&dir)?;
    let l = |i: usize, t: &str| format!("model.language_model.layers.{i}.{t}");
    let sets: Vec<(&str, Vec<String>)> = vec![
        ("mlp gate|up", vec![l(3, "mlp.gate_proj"), l(3, "mlp.up_proj")]),
        ("mlp down", vec![l(3, "mlp.down_proj")]),
        ("gdn qkv|z", vec![l(0, "linear_attn.in_proj_qkv"), l(0, "linear_attn.in_proj_z")]),
        ("gdn out", vec![l(0, "linear_attn.out_proj")]),
        ("attn q|k|v", vec![l(3, "self_attn.q_proj"), l(3, "self_attn.k_proj"), l(3, "self_attn.v_proj")]),
        ("attn o", vec![l(3, "self_attn.o_proj")]),
    ];
    let mmax = *ms.iter().max().unwrap();
    let kmax = 17408;
    let ws = Workspace::new(mmax, kmax)?;
    let ws_lt = Workspace::new(mmax, kmax)?;
    let mut seed = 0x2545F4914F6CDD1Du64;
    let x: Vec<u16> = (0..mmax * kmax)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            f32_to_bf16(((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0)
        })
        .collect();
    let xd = DeviceBuffer::from_slice(&x)?;
    let y = DeviceBuffer::<u16>::alloc(mmax * 34816)?;
    println!("| linear | kind | N x K | M | Rust TFLOPS | cuBLASLt TFLOPS | Rust/Lt | quant us |");
    println!("|---|---|---|---|---|---|---|---|");
    for (label, parts) in &sets {
        if only.as_deref().is_some_and(|o| !label.contains(o)) {
            continue;
        }
        let p: Vec<&str> = parts.iter().map(|s| s.as_str()).collect();
        let lin = PromptLinear::load_opts(&ck, &p, true)?;
        let (n, k) = (lin.n, lin.k);
        let fp4 = lin.kind == Kind::Nvfp4;
        for &m in &ms {
            // quantize once for each path (tiled scales for Rust, cuBLASLt layout for Lt)
            let quant = |w: &Workspace, ltl: bool| -> Result<()> {
                if fp4 {
                    kern.quant_fp4(&s, w.codes.ptr(), w.sf.ptr(), xd.ptr(), m, k, k, lin.g, ltl)
                } else {
                    kern.quant_fp8(&s, w.codes.ptr(), xd.ptr(), m, k, k, lin.g)
                }
            };
            quant(&ws, false)?;
            quant(&ws_lt, true)?;
            let tq = time(&s, iters, || quant(&ws, false))?;
            let rust = || -> Result<()> {
                if fp4 {
                    kern.gemm_fp4(&s, y.ptr(), n, ws.codes.ptr(), ws.sf.ptr(), lin.w.ptr(), lin.sw.as_ref().unwrap().ptr(),
                        lin.alpha.ptr(), None, m, n, k)
                } else {
                    kern.gemm_fp8(&s, y.ptr(), n, ws.codes.ptr(), lin.w.ptr(), lin.alpha.ptr(), None, m, n, k)
                }
            };
            let tr = time(&s, iters, rust)?;
            let sw = lin.sw_lt.as_ref().map(|b| b.ptr()).unwrap_or(fe_cuda::DevPtr::NULL);
            let plan = lt.plan(fp4, m, n, k, n, ws_lt.sf.ptr(), sw)?;
            let run_lt = || -> Result<()> {
                unsafe { lt.run(&plan, &s, y.ptr(), ws_lt.codes.ptr(), ws_lt.sf.ptr(), lin.w.ptr(), sw, lin.parts[0].alpha) }
            };
            let tl = time(&s, iters, run_lt)?;
            let fl = 2.0 * m as f64 * n as f64 * k as f64;
            let (tfr, tfl) = (fl / (tr as f64 * 1e9), fl / (tl as f64 * 1e9));
            println!("| {label} | {} | {n} x {k} | {m} | {tfr:.0} | {tfl:.0} | {:.2} | {:.0} |",
                if fp4 { "NVFP4" } else { "FP8" }, tfr / tfl, tq * 1000.0);
        }
    }
    // fused MLP front: gate|up GEMM + SwiGLU + NVFP4 quant of down's input, vs GEMM (bf16 out) + swiglu_quant
    if only.as_deref().is_none_or(|o| o.contains("swiglu")) {
        let gu = PromptLinear::load_swiglu(&ck, &l(3, "mlp.gate_proj"), &l(3, "mlp.up_proj"))?;
        let down = PromptLinear::load(&ck, &[l(3, "mlp.down_proj").as_str()])?;
        let (n, k) = (gu.n, gu.k);
        let act = Workspace::new(mmax, n / 2)?;
        println!();
        println!("| fused MLP front (gate|up 34816 x 5120 -> NVFP4 act) | M | fused ms | GEMM + swiglu_quant ms | fused TFLOPS | speedup |");
        println!("|---|---|---|---|---|---|");
        for &m in &ms {
            kern.quant_fp4(&s, ws.codes.ptr(), ws.sf.ptr(), xd.ptr(), m, k, k, gu.g, false)?;
            let sw = gu.sw.as_ref().unwrap().ptr();
            let tf = time(&s, iters, || kern.gemm_fp4_swiglu(&s, &act, ws.codes.ptr(), ws.sf.ptr(), gu.w.ptr(), sw, gu.alpha.ptr(), m, n, k, down.g))?;
            let tu = time(&s, iters, || {
                kern.gemm_fp4(&s, y.ptr(), n, ws.codes.ptr(), ws.sf.ptr(), gu.w.ptr(), sw, gu.alpha.ptr(), None, m, n, k)?;
                kern.swiglu_quant(&s, &act, y.ptr(), m, n, down.g)
            })?;
            let fl = 2.0 * m as f64 * n as f64 * k as f64;
            println!("| | {m} | {tf:.3} | {tu:.3} | {:.0} | {:.2}x |", fl / (tf as f64 * 1e9), tu / tf);
        }
    }
    Ok(())
}
