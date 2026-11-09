//! Tensor-parallel slices vs the whole tensor on one GPU: column slices (two ranks' halves, and the GDN
//! q/k/v section split) must reproduce the full output's columns, row slices' partials must sum to it
//! (f32 outputs, fp32-accumulation tolerance: the slices run their own plans).
//! usage: exl3-tpcheck <model dir> [layer index]

use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_exl3::gpu::{DeviceExl3, Kernels};
use fe_exl3::half::f32_to_bf16;
use fe_exl3::linear::{default_blocks, Dtype, Linear, Workspace};
use fe_exl3::st::Checkpoint;
use fe_exl3::Exl3Tensor;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ck = Checkpoint::open(&args[1]).unwrap();
    let layer: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
    let ctx = Context::new(0).unwrap();
    let blocks = default_blocks(ctx.sm_count().unwrap() as u32);
    let kern = Kernels::load().unwrap();
    let s = Stream::new().unwrap();
    let rows = 8usize;
    let p = |l: usize, m: &str| format!("model.language_model.layers.{l}.{m}");
    // (tensor, split): "cols" = halves of n, "gdn" = GDN in_proj_qkv q/k/v sections, "rows" = halves of k
    let cases = [
        (p(layer, "self_attn.q_proj"), "cols"),
        (p(layer, "self_attn.k_proj"), "cols"),
        (p(layer, "self_attn.o_proj"), "rows"),
        (p(layer, "mlp.gate_proj"), "cols"),
        (p(layer, "mlp.down_proj"), "rows"),
        (p(layer - 1, "linear_attn.in_proj_qkv"), "gdn"),
        (p(layer - 1, "linear_attn.in_proj_z"), "cols"),
        (p(layer - 1, "linear_attn.out_proj"), "rows"),
        ("lm_head".to_string(), "cols"),
    ];
    let mut fails = 0;
    for (base, split) in cases {
        let t = Exl3Tensor::load(&ck, &base).unwrap();
        let (k, n) = (t.k, t.n);
        let full = Linear::new(&kern, DeviceExl3::upload_strips(&t).unwrap(), blocks).unwrap();
        let xb: Vec<u16> = (0..rows * k).map(|i| f32_to_bf16(((i * 7919 % 2003) as f32 / 1001.0) - 1.0)).collect();
        let x = DeviceBuffer::from_slice(&xb).unwrap();
        let yf = DeviceBuffer::<f32>::alloc(rows * n).unwrap();
        let mut parts: Vec<(Linear, Vec<(usize, usize)>, Option<(usize, usize)>)> = Vec::new();
        match split {
            "cols" => {
                for r in 0..2 {
                    let rg = vec![(r * n / 2, (r + 1) * n / 2)];
                    parts.push((Linear::new(&kern, DeviceExl3::upload_strips(&t.slice_cols(&rg).unwrap()).unwrap(), blocks).unwrap(), rg, None));
                }
            }
            "gdn" => {
                // q (16 heads x 128), k (16 x 128), v (48 x 128): rank r takes half of each section
                let (q, kk) = (2048, 2048);
                for r in 0..2 {
                    let rg = vec![
                        (r * q / 2, (r + 1) * q / 2),
                        (q + r * kk / 2, q + (r + 1) * kk / 2),
                        (q + kk + r * (n - q - kk) / 2, q + kk + (r + 1) * (n - q - kk) / 2),
                    ];
                    parts.push((Linear::new(&kern, DeviceExl3::upload_strips(&t.slice_cols(&rg).unwrap()).unwrap(), blocks).unwrap(), rg, None));
                }
            }
            _ => {
                for r in 0..2 {
                    let kr = (r * k / 2, (r + 1) * k / 2);
                    parts.push((Linear::new(&kern, DeviceExl3::upload_strips(&t.slice_rows(kr.0, kr.1).unwrap()).unwrap(), blocks).unwrap(), vec![], Some(kr)));
                }
            }
        }
        let mut lins: Vec<&Linear> = vec![&full];
        lins.extend(parts.iter().map(|p| &p.0));
        let ws = Workspace::new(&lins, rows as u32).unwrap();
        full.forward(&kern, &s, x.ptr(), Dtype::Bf16, yf.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
        s.sync().unwrap();
        let yfull: Vec<f32> = yf.to_vec().unwrap();
        let mut recon = vec![0f32; rows * n];
        for (lin, rg, kr) in &parts {
            let yp = DeviceBuffer::<f32>::alloc(rows * lin.n as usize).unwrap();
            if let Some((k0, k1)) = kr {
                // this rank's input columns, contiguous
                let xs: Vec<u16> = (0..rows).flat_map(|r| xb[r * k + k0..r * k + k1].to_vec()).collect();
                let xd = DeviceBuffer::from_slice(&xs).unwrap();
                lin.forward(&kern, &s, xd.ptr(), Dtype::Bf16, yp.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
                s.sync().unwrap();
                let v: Vec<f32> = yp.to_vec().unwrap();
                for (a, b) in recon.iter_mut().zip(&v) {
                    *a += *b;
                }
            } else {
                lin.forward(&kern, &s, x.ptr(), Dtype::Bf16, yp.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
                s.sync().unwrap();
                let v: Vec<f32> = yp.to_vec().unwrap();
                let ns = lin.n as usize;
                for r in 0..rows {
                    let mut c = 0;
                    for &(a, b) in rg {
                        recon[r * n + a..r * n + b].copy_from_slice(&v[r * ns + c..r * ns + c + (b - a)]);
                        c += b - a;
                    }
                }
            }
        }
        let rms = (yfull.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / yfull.len() as f64).sqrt();
        let maxd = yfull.iter().zip(&recon).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0f64, f64::max) / rms;
        let ok = maxd < 2e-5;
        if !ok {
            fails += 1;
        }
        println!(
            "{} {:<46} {split:>4} {k}x{n} -> {} | max|full - tp2| / rms {:.1e}",
            if ok { "PASS" } else { "FAIL" },
            base.replace("model.language_model.", ""),
            parts.iter().map(|p| format!("{}x{} sk{} wk{} v{}", p.0.k, p.0.n, p.0.plan.sk, p.0.plan.wk, p.0.plan.ver)).collect::<Vec<_>>().join(", "),
            maxd
        );
    }
    if fails > 0 {
        println!("{fails} FAILED");
        std::process::exit(1);
    }
    println!("all PASS");
}
