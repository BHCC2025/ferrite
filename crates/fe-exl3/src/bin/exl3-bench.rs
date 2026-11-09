//! Decode GEMV microbench: every distinct (kind, bits) shape of the checkpoint, rows 1 and 16.
//! Times rot_in + GEMV (forward) and the GEMV alone over many back-to-back calls with events; GB/s is
//! trellis bytes / GEMV time. Calls cycle over enough device copies of the weight (>= 96 MB in all) that
//! nothing is served from L2, as in a real forward. Run under the GPU lock (`flock $FERRITE_GPU_LOCK`, tools/site.sh).
//! usage: exl3-bench <model dir> [name filter] [iters] [sk,wk plan override]
//! env: FE_BENCH_ROWS2 (second row count, 16), FE_BENCH_TP2=1 (rank 0's TP2 slices), FE_BENCH_COPIES

use std::collections::BTreeMap;

use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_exl3::{exl3_bases, Exl3Tensor};
use fe_exl3::gpu::{DeviceExl3, Kernels};
use fe_exl3::linear::{default_blocks, Dtype, Linear, Plan, Workspace};
use fe_exl3::st::Checkpoint;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = &args[1];
    let filter = args.get(2).cloned().unwrap_or_default();
    let iters: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(200);
    let forced = args.get(4).map(|s| Plan::parse(s).expect("plan sk,wk[,ver[,occ]]"));
    let ctx = Context::new(0).unwrap();
    let sms = ctx.sm_count().unwrap() as u32;
    let blocks = default_blocks(sms);
    let kern = Kernels::load().unwrap();
    let s = Stream::new().unwrap();
    let ck = Checkpoint::open(model).unwrap();

    let mut picked: BTreeMap<(String, u32), (String, usize)> = BTreeMap::new();
    for b in exl3_bases(&ck) {
        if b.starts_with("model.visual") || !b.contains(&filter) {
            continue;
        }
        let bits = ck.info(&format!("{b}.trellis")).unwrap().shape[2] as u32 / 16;
        let kind: String = b.split('.').filter(|p| p.parse::<u32>().is_err()).collect::<Vec<_>>().join(".");
        picked.entry((kind, bits)).or_insert((b.clone(), 0)).1 += 1;
    }
    println!("{} SMs, program target {blocks}, {iters} calls each", sms);
    println!(
        "| shape | bits | k x n | count | plan | MB | m=1 gemv us | m=1 GB/s | m=1 fwd us | m=16 gemv us | m=16 GB/s | m=16 fwd us |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut total = [0f64; 4];
    for ((kind, bits), (base, count)) in picked {
        let mut t = Exl3Tensor::load(&ck, &base).unwrap();
        // FE_BENCH_TP2=1: rank 0's slice of a two-Spark split (the shapes a TP2 rank runs, tuned TP2 plans)
        if std::env::var("FE_BENCH_TP2").is_ok() {
            use fe_exl3::tp::{qwen35_split, Split};
            t = match qwen35_split(&base, t.k, t.n, 0, 2) {
                Split::Cols(r) => t.slice_cols(&r).unwrap(),
                Split::Rows(a, b) => t.slice_rows(a, b).unwrap(),
                Split::Replicate => t,
            };
        }
        let copies = std::env::var("FE_BENCH_COPIES").ok().and_then(|v| v.parse().ok()).unwrap_or((96usize << 20).div_ceil(t.trellis_bytes()).clamp(1, 64));
        let mut lins = Vec::new();
        for _ in 0..copies {
            let d = DeviceExl3::upload_as(&t, std::env::var("FE_EXL3_STORED").is_err()).unwrap();
            let l = match forced {
                Some(p) => Linear::with_plan(&kern, d, p),
                None => Linear::new(&kern, d, blocks),
            };
            match l {
                Ok(l) => lins.push(l),
                Err(_) => break,
            }
        }
        if lins.is_empty() {
            println!("| {kind} | {bits} | plan does not fit |");
            continue;
        }
        let lin = &lins[0];
        let (k, n) = (lin.k as usize, lin.n as usize);
        // the second row count: 16 unless FE_BENCH_ROWS2 says otherwise (e.g. 32, 64 for batched verify)
        let r2: u32 = std::env::var("FE_BENCH_ROWS2").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        let ws = Workspace::new(&[lin], r2.max(16)).unwrap();
        let x = DeviceBuffer::<u16>::from_slice(&vec![0x3f80u16; r2.max(16) as usize * k]).unwrap(); // bf16 1.0
        let y = DeviceBuffer::<u16>::alloc(r2.max(16) as usize * n).unwrap();
        let mut cols = Vec::new();
        for rows in [1u32, r2] {
            for _ in 0..20 {
                lin.forward(&kern, &s, x.ptr(), Dtype::Bf16, y.ptr(), Dtype::Bf16, rows, &ws).unwrap();
            }
            let g = s
                .time(|s| {
                    for i in 0..iters as usize {
                        lins[i % copies].gemv_rotated(s, ws.xh.ptr(), y.ptr(), Dtype::Bf16, rows, &ws).unwrap();
                    }
                    Ok(())
                })
                .unwrap() as f64
                * 1e3
                / iters as f64;
            let f = s
                .time(|s| {
                    for i in 0..iters as usize {
                        lins[i % copies].forward(&kern, s, x.ptr(), Dtype::Bf16, y.ptr(), Dtype::Bf16, rows, &ws).unwrap();
                    }
                    Ok(())
                })
                .unwrap() as f64
                * 1e3
                / iters as f64;
            cols.push((g, lin.weight_bytes() as f64 / (g * 1e-6) / 1e9, f));
        }
        total[0] += cols[0].0 * count as f64;
        total[1] += cols[0].2 * count as f64;
        total[2] += cols[1].0 * count as f64;
        total[3] += cols[1].2 * count as f64;
        println!(
            "| {} | {bits} | {k} x {n} | {count} | sk{} wk{} | {:.2} | {:.1} | {:.0} | {:.1} | {:.1} | {:.0} | {:.1} |",
            kind.replace("model.language_model.", ""),
            lin.plan.sk,
            lin.plan.wk,
            lin.weight_bytes() as f64 / 1e6,
            cols[0].0,
            cols[0].1,
            cols[0].2,
            cols[1].0,
            cols[1].1,
            cols[1].2
        );
    }
    println!(
        "sum over the model's linears (count-weighted): m=1 gemv {:.2} ms, fwd {:.2} ms; m=16 gemv {:.2} ms, fwd {:.2} ms",
        total[0] / 1e3,
        total[1] / 1e3,
        total[2] / 1e3,
        total[3] / 1e3
    );
}
