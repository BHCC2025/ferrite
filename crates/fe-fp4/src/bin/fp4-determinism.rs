//! Run every GEMM variant (n128 / n256 / mb, NVFP4 + FP8) and the fused SwiGLU kernel `reps` times on the same
//! inputs (real layer-3 weights, M rows) and require bit-identical outputs every time.
//! `fp4-determinism [model_dir] [reps=50] [m=2048]`; env FE_DET_VARIANTS (default 0,1,2), FE_DET_LINS (indices
//! into gate|up, down, q|k|v, o; default all), FE_DET_SWIGLU=0 skips the fused kernel (small runs for racecheck).
use std::path::PathBuf;
use anyhow::{ensure, Result};
use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_fp4::format::f32_to_bf16;
use fe_fp4::{Checkpoint, Kernels, PromptLinear, Workspace};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(PathBuf::from).unwrap_or_else(fe_site::nvfp4_dir);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let m: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let _ctx = Context::new(0)?;
    let mut kern = Kernels::load()?;
    let s = Stream::new()?;
    let ck = Checkpoint::open(&dir)?;
    let l = |t: &str| format!("model.language_model.layers.3.{t}");
    let lins = [
        PromptLinear::load(&ck, &[l("mlp.gate_proj").as_str(), l("mlp.up_proj").as_str()])?,
        PromptLinear::load(&ck, &[l("mlp.down_proj").as_str()])?,
        PromptLinear::load(&ck, &[l("self_attn.q_proj").as_str(), l("self_attn.k_proj").as_str(), l("self_attn.v_proj").as_str()])?,
        PromptLinear::load(&ck, &[l("self_attn.o_proj").as_str()])?,
    ];
    let gu = PromptLinear::load_swiglu(&ck, &l("mlp.gate_proj"), &l("mlp.up_proj"))?;
    let kmax = 17408;
    let mut seed = 0x9E3779B97F4A7C15u64;
    let x: Vec<u16> = (0..m * kmax)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            f32_to_bf16(((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0)
        })
        .collect();
    let xd = DeviceBuffer::from_slice(&x)?;
    let ws = Workspace::new(m, kmax)?;
    let act = Workspace::new(m, kmax)?;
    let y = DeviceBuffer::<u16>::zeroed(m * 34816)?;
    let list = |k: &str, d: &str| -> Vec<usize> {
        std::env::var(k).unwrap_or_else(|_| d.into()).split(',').filter_map(|x| x.parse().ok()).collect()
    };
    let mut bad = 0;
    for variant in list("FE_DET_VARIANTS", "0,1,2,4") {
        kern.variant = variant as u32;
        kern.variant8 = variant as u32;
        for li in list("FE_DET_LINS", "0,1,2,3") {
            let lin = &lins[li];
            let mut first: Option<Vec<u16>> = None;
            let mut diffs = 0;
            for _ in 0..reps {
                lin.forward(&kern, &s, &ws, y.ptr(), lin.n, xd.ptr(), lin.k, m, None)?;
                s.sync()?;
                let mut v = vec![0u16; m * lin.n];
                fe_cuda::memcpy_dtoh(&mut v, y.ptr())?;
                match &first {
                    None => first = Some(v),
                    Some(f) => diffs += (f != &v) as usize,
                }
            }
            println!("variant {variant} {}: {diffs} / {} runs differ", lin.name, reps - 1);
            bad += diffs;
        }
    }
    let mut first: Option<Vec<u8>> = None;
    let mut diffs = 0;
    let sw_reps = if std::env::var("FE_DET_SWIGLU").is_ok_and(|v| v == "0") { 0 } else { reps };
    for _ in 0..sw_reps {
        gu.forward_swiglu(&kern, &s, &ws, &act, xd.ptr(), gu.k, m, lins[1].g)?;
        s.sync()?;
        let mut v = vec![0u8; m * gu.n / 4 + m.div_ceil(128) * 128 * gu.n / 32];
        fe_cuda::memcpy_dtoh(&mut v[..m * gu.n / 4], act.codes.ptr())?;
        fe_cuda::memcpy_dtoh(&mut v[m * gu.n / 4..], act.sf.ptr())?;
        match &first {
            None => first = Some(v),
            Some(f) => diffs += (f != &v) as usize,
        }
    }
    println!("swiglu: {diffs} / {} runs differ", sw_reps.saturating_sub(1));
    bad += diffs;
    ensure!(bad == 0, "non-deterministic output");
    println!("all deterministic");
    Ok(())
}
