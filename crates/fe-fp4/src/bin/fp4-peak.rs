//! Tensor-core peak of the block-scaled NVFP4 and FP8 mma.sync forms on this GPU (no memory traffic).
use anyhow::Result;
use fe_cuda::{Context, DeviceBuffer, Event, Stream};

fn main() -> Result<()> {
    let ctx = Context::new(0)?;
    let sms = ctx.sm_count()? as u32;
    let kern = fe_fp4::Kernels::load()?;
    let f = kern.module.function("fe_fp4_peak")?;
    let s = Stream::new()?;
    let iters = 4096u32;
    for fp4 in [1u32, 0, 2, 3] {
        for (ctas, threads) in [(sms, 256u32), (2 * sms, 256), (sms, 512), (4 * sms, 256)] {
            let out = DeviceBuffer::<f32>::alloc((ctas * threads) as usize)?;
            let run = || unsafe { s.launch(&f, ctas, threads, 0, (out.ptr(), iters, fp4)) };
            run()?;
            s.sync()?;
            let (a, b) = (Event::new()?, Event::new()?);
            s.record(&a)?;
            for _ in 0..5 {
                run()?;
            }
            s.record(&b)?;
            b.sync()?;
            let ms = b.elapsed_since(&a)? / 5.0;
            let k = if fp4 == 1 { 64.0 } else { 32.0 };
            let label = ["FP8 ", "NVFP4", "FP8 (sm_89 form)", "FP8 (block-scaled form, scales 1)"][fp4 as usize];
            let flops = (ctas * threads / 32) as f64 * iters as f64 * 16.0 * 2.0 * 16.0 * 8.0 * k;
            println!("{label} ctas {ctas} x {threads} thr: {:.0} TFLOPS",
                flops / (ms as f64 * 1e9));
        }
    }
    // inner-loop ceiling: the wide NVFP4 GEMM with no main-loop loads (gate|up shape, M = 4096)
    let (m, n, k) = (4096usize, 34816usize, 5120usize);
    let g = kern.module.function("fe_gemm_nvfp4_n256_noload")?;
    g.set_max_dynamic_smem(fe_fp4::FP4_SMEM_N256)?;
    let a = DeviceBuffer::<u8>::zeroed(m * k / 2)?;
    let sa = DeviceBuffer::<u8>::zeroed(m * k / 16)?;
    let w = DeviceBuffer::<u8>::zeroed(n * k / 2)?;
    let sw = DeviceBuffer::<u8>::zeroed(n * k / 16)?;
    let al = DeviceBuffer::<f32>::zeroed(n)?;
    let y = DeviceBuffer::<u16>::alloc(m * n)?;
    let grid = (m / 128 * (n / 256)) as u32;
    let run = || unsafe {
        s.launch(&g, grid, 256, fe_fp4::FP4_SMEM_N256, (y.ptr(), a.ptr(), sa.ptr(), w.ptr(), sw.ptr(), al.ptr(),
            fe_cuda::DevPtr::NULL, m as u32, n as u32, k as u32, n as u32, 8u32))
    };
    run()?;
    s.sync()?;
    let (e0, e1) = (Event::new()?, Event::new()?);
    s.record(&e0)?;
    for _ in 0..5 {
        run()?;
    }
    s.record(&e1)?;
    e1.sync()?;
    let ms = e1.elapsed_since(&e0)? / 5.0;
    println!("NVFP4 n256 GEMM without main-loop loads ({m}x{n}x{k}): {:.0} TFLOPS",
        2.0 * (m * n * k) as f64 / (ms as f64 * 1e9));
    Ok(())
}
