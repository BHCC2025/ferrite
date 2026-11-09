//! Determinism loop-check of the decode GEMV: for one tensor per (kind, bits), every row count in the list,
//! `iters` forwards (rot_in + GEMV, tuned plan, PDL as configured) on the same input must give bit-identical
//! f32 outputs; consecutive calls alternate with another linear's call so PDL overlap and workspace reuse are
//! exercised. Also used under compute-sanitizer (small `iters`).
//! usage: exl3-detcheck <model dir> [name filter] [iters] [rows,rows,...]
use std::collections::BTreeMap;

use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_exl3::gpu::Kernels;
use fe_exl3::half::f32_to_bf16;
use fe_exl3::linear::{default_blocks, Dtype, Linear, Workspace};
use fe_exl3::st::Checkpoint;
use fe_exl3::exl3_bases;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ck = Checkpoint::open(&args[1]).unwrap();
    let filter = args.get(2).cloned().unwrap_or_default();
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(200);
    let rows_list: Vec<u32> =
        args.get(4).map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1, 5, 16, 33]);
    let max_rows = *rows_list.iter().max().unwrap();
    let ctx = Context::new(0).unwrap();
    let blocks = default_blocks(ctx.sm_count().unwrap() as u32);
    let kern = Kernels::load().unwrap();
    let s = Stream::new().unwrap();
    let mut picked: BTreeMap<(String, u32), String> = BTreeMap::new();
    for b in exl3_bases(&ck) {
        if b.starts_with("model.visual") || !b.contains(&filter) {
            continue;
        }
        let bits = ck.info(&format!("{b}.trellis")).unwrap().shape[2] as u32 / 16;
        let kind: String = b.split('.').filter(|p| p.parse::<u32>().is_err()).collect::<Vec<_>>().join(".");
        picked.entry((kind, bits)).or_insert(b);
    }
    // a second linear whose calls interleave with the checked one (PDL neighbours, shared workspace)
    let other = Linear::load(&kern, &ck, "model.language_model.layers.0.linear_attn.in_proj_z", blocks).unwrap();
    let mut bad_total = 0usize;
    for (_, base) in picked {
        let lin = Linear::load(&kern, &ck, &base, blocks).unwrap();
        let ws = Workspace::new(&[&lin, &other], max_rows).unwrap();
        let (k, n) = (lin.k as usize, lin.n as usize);
        let xb: Vec<u16> = (0..max_rows as usize * k.max(other.k as usize))
            .map(|i| f32_to_bf16((((i * 2654435761usize) >> 7) % 4001) as f32 / 1000.0 - 2.0))
            .collect();
        let x = DeviceBuffer::from_slice(&xb).unwrap();
        // a pageable upload is not ordered with the non-blocking stream: let it land first
        ctx.synchronize().unwrap();
        let y = DeviceBuffer::<f32>::alloc(max_rows as usize * n).unwrap();
        let yo = DeviceBuffer::<f32>::alloc(max_rows as usize * other.n as usize).unwrap();
        let mut line = format!("{:<44} {}b {:>5}x{:<6} sk{} wk{} v{} |", base.replace("model.language_model.", ""), lin.bits,
            k, n, lin.plan.sk, lin.plan.wk, lin.plan.ver);
        for &rows in &rows_list {
            let r = rows as usize * n;
            lin.forward(&kern, &s, x.ptr(), Dtype::Bf16, y.ptr(), Dtype::F32, rows, &ws).unwrap();
            s.sync().unwrap();
            let want: Vec<f32> = y.to_vec().unwrap()[..r].to_vec();
            let mut bad = 0usize;
            for i in 0..iters {
                other.forward(&kern, &s, x.ptr(), Dtype::Bf16, yo.ptr(), Dtype::F32, rows, &ws).unwrap();
                lin.forward(&kern, &s, x.ptr(), Dtype::Bf16, y.ptr(), Dtype::F32, rows, &ws).unwrap();
                if i % 10 == 9 || i + 1 == iters {
                    s.sync().unwrap();
                    let got: Vec<f32> = y.to_vec().unwrap();
                    bad += got[..r].iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                }
            }
            bad_total += bad;
            line += &format!(" m{rows} {}", if bad == 0 { "same".to_string() } else { format!("DIFF {bad}") });
        }
        println!("{} {line}", if line.contains("DIFF") { "FAIL" } else { "PASS" });
    }
    println!("{} ({} PDL)", if bad_total == 0 { "all bit-identical" } else { "NONDETERMINISM FOUND" },
        if fe_exl3::gpu::pdl::enabled() { "with" } else { "without" });
    if bad_total > 0 {
        std::process::exit(1);
    }
}
