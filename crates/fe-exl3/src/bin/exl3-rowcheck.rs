//! Batch invariance of the decode GEMV: for one tensor per (kind, bits), row r's f32 output must not depend on how
//! many rows the call has (the 1..8-, 9..16- and 17+-row kernels): every row count in the list is compared, row
//! by row, with the 1-row call of the same input row, `iters` times (tuned plans, PDL as configured).
//! The 1-row references read their input row straight from the device copy of x: an upload per row through a
//! synchronous pageable copy is not ordered with the non-blocking stream and can feed a kernel the previous row.
//! usage: exl3-rowcheck <model dir> [name filter] [rows,rows,... | 1-64] [iters]
//! env: FE_ROWCHECK_TP2=1 checks a TP2 rank's slices (FE_ROWCHECK_RANK, default 0) with the tuned TP2 plans
use std::collections::BTreeMap;

use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_exl3::{exl3_bases, Exl3Tensor};
use fe_exl3::gpu::{DeviceExl3, Kernels};
use fe_exl3::half::f32_to_bf16;
use fe_exl3::linear::{default_blocks, Dtype, Linear, Workspace};
use fe_exl3::st::Checkpoint;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ck = Checkpoint::open(&args[1]).unwrap();
    let filter = args.get(2).cloned().unwrap_or_default();
    let rows_list: Vec<u32> = match args.get(3).map(String::as_str) {
        Some(r) if r.contains('-') => {
            let (a, b) = r.split_once('-').unwrap();
            (a.parse().unwrap()..=b.parse().unwrap()).collect()
        }
        Some(r) => r.split(',').map(|x| x.parse().unwrap()).collect(),
        None => vec![2, 8, 9, 15, 16, 17],
    };
    let iters: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1);
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
    let mut bad_total = 0usize;
    for (_, base) in picked {
        // FE_ROWCHECK_TP2=1: a TP2 rank's slice (FE_ROWCHECK_RANK, default 0), built as the engine builds it, with
        // the tuned TP2 plans
        let lin = if std::env::var("FE_ROWCHECK_TP2").is_ok_and(|v| v == "1") {
            use fe_exl3::tp::{qwen35_split, Split};
            let rank: usize = std::env::var("FE_ROWCHECK_RANK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let t = Exl3Tensor::load(&ck, &base).unwrap();
            let t = match qwen35_split(&base, t.k, t.n, rank, 2) {
                Split::Cols(r) => t.slice_cols(&r).unwrap(),
                Split::Rows(a, b) => t.slice_rows(a, b).unwrap(),
                Split::Replicate => t,
            };
            Linear::new(&kern, DeviceExl3::upload_strips(&t).unwrap(), blocks).unwrap()
        } else {
            Linear::load(&kern, &ck, &base, blocks).unwrap()
        };
        let ws = Workspace::new(&[&lin], max_rows).unwrap();
        let (k, n) = (lin.k as usize, lin.n as usize);
        let xb: Vec<u16> = (0..max_rows as usize * k)
            .map(|i| f32_to_bf16((((i * 2654435761usize) >> 7) % 4001) as f32 / 1000.0 - 2.0))
            .collect();
        let x = DeviceBuffer::from_slice(&xb).unwrap();
        ctx.synchronize().unwrap();
        let y = DeviceBuffer::<f32>::alloc(max_rows as usize * n).unwrap();
        // reference: each row alone, its input row read in place
        let mut want = vec![0f32; max_rows as usize * n];
        for r in 0..max_rows as usize {
            lin.forward(&kern, &s, x.ptr().add::<u16>(r * k), Dtype::Bf16, y.ptr(), Dtype::F32, 1, &ws).unwrap();
            s.sync().unwrap();
            fe_cuda::memcpy_dtoh(&mut want[r * n..(r + 1) * n], y.ptr()).unwrap();
        }
        let mut fails: BTreeMap<u32, (usize, Vec<usize>, f64)> = BTreeMap::new();
        for _ in 0..iters {
            for &rows in &rows_list {
                let r = rows as usize * n;
                lin.forward(&kern, &s, x.ptr(), Dtype::Bf16, y.ptr(), Dtype::F32, rows, &ws).unwrap();
                s.sync().unwrap();
                let mut got = vec![0f32; r];
                fe_cuda::memcpy_dtoh(&mut got, y.ptr()).unwrap();
                let bad = got.iter().zip(&want[..r]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                if bad > 0 {
                    let rows_bad: Vec<usize> = (0..rows as usize)
                        .filter(|&i| (0..n).any(|j| got[i * n + j].to_bits() != want[i * n + j].to_bits())).collect();
                    let rel = got.iter().zip(&want[..r]).map(|(a, b)| ((a - b).abs() / b.abs().max(1e-3)) as f64)
                        .fold(0.0f64, f64::max);
                    let e = fails.entry(rows).or_insert((0, rows_bad.clone(), 0.0));
                    e.0 += 1;
                    e.2 = e.2.max(rel);
                }
                bad_total += bad;
            }
        }
        let mut line = format!("{:<44} {}b {:>5}x{:<6} sk{} wk{} v{} | {} row counts x {iters}:", base.replace("model.language_model.", ""),
            lin.bits, k, n, lin.plan.sk, lin.plan.wk, lin.plan.ver, rows_list.len());
        if fails.is_empty() {
            line += " all rows identical to their 1-row calls";
        }
        for (m, (cnt, rows_bad, rel)) in &fails {
            line += &format!(" m{m} DIFF in {cnt} calls (first rows {rows_bad:?}, max rel {rel:.2e})");
        }
        println!("{} {line}", if fails.is_empty() { "PASS" } else { "FAIL" });
    }
    println!("{} ({} PDL)", if bad_total == 0 { "every row identical to its 1-row call" } else { "ROW-COUNT DEPENDENCE FOUND" },
        if fe_exl3::gpu::pdl::enabled() { "with" } else { "without" });
    if bad_total > 0 {
        std::process::exit(1);
    }
}
