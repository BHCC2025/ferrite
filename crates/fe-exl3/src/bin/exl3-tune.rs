//! Per-shape plan search for the decode GEMV: every (sk, wk) that fits, every kernel variant, timed at
//! rows 1 and 16 over L2-cold weight copies. (sk, wk) fix the summation order, so they are chosen once per
//! shape (row-count invariant bits); the kernel variant (v1 / v2 + occupancy) does not change bits.
//! Prints the table and the Rust `TUNED` entries for crates/fe-exl3/src/plans.rs.
//! FE_TUNE_TP2=1 tunes the TP2 slices instead (o_proj / out_proj / down_proj / mtp.fc: halves of k, the
//! rest: halves of n), the shapes each rank of a two-Spark split runs.
//! usage: exl3-tune <model dir> [name filter] [iters] [weight of the 16-row time, default 0.25]

use std::collections::BTreeMap;

use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_exl3::gpu::{DeviceExl3, Kernels};
use fe_exl3::linear::{default_blocks, plan as default_plan, Dtype, Linear, Plan, Workspace};
use fe_exl3::st::Checkpoint;
use fe_exl3::{exl3_bases, Exl3Tensor};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = &args[1];
    let filter = args.get(2).cloned().unwrap_or_default();
    let iters: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(100);
    let w16: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(0.25);
    let ctx = Context::new(0).unwrap();
    let blocks = default_blocks(ctx.sm_count().unwrap() as u32);
    let kern = Kernels::load().unwrap();
    let s = Stream::new().unwrap();
    let ck = Checkpoint::open(model).unwrap();

    // distinct (k, n, bits) shapes, with a representative tensor and a use count
    let tp2 = std::env::var("FE_TUNE_TP2").is_ok();
    let row_split = |b: &str| ["o_proj", "out_proj", "down_proj", "mtp.fc"].iter().any(|p| b.contains(p));
    let mut shapes: BTreeMap<(u32, u32, u32), (String, usize)> = BTreeMap::new();
    for b in exl3_bases(&ck) {
        if b.starts_with("model.visual") || !b.contains(&filter) {
            continue;
        }
        let sh = &ck.info(&format!("{b}.trellis")).unwrap().shape;
        let (mut k, mut n) = (sh[0] as u32 * 16, sh[1] as u32 * 16);
        if tp2 {
            if row_split(&b) { k /= 2 } else { n /= 2 }
        }
        let key = (k, n, sh[2] as u32 / 16);
        shapes.entry(key).or_insert((b.clone(), 0)).1 += 1;
    }
    let mut entries = Vec::new();
    let mut tot = [0f64; 4];
    println!("| k x n | bits | uses | default plan | us m1 / m16 | best plan | us m1 / m16 | speedup m1 |");
    println!("|---|---|---|---|---|---|---|---|");
    for ((k, n, bits), (base, uses)) in shapes {
        let mut t = Exl3Tensor::load(&ck, &base).unwrap();
        if tp2 {
            t = if row_split(&base) { t.slice_rows(0, t.k / 2).unwrap() } else { t.slice_cols(&[(0, t.n / 2)]).unwrap() };
        }
        let copies = (96usize << 20).div_ceil(t.trellis_bytes()).clamp(1, 64);
        let devs: Vec<DeviceExl3> = (0..copies).map(|_| DeviceExl3::upload_as(&t, std::env::var("FE_EXL3_STORED").is_err()).unwrap()).collect();
        let x = DeviceBuffer::<u16>::from_slice(&vec![0x3f80u16; 16 * k as usize]).unwrap();
        let y = DeviceBuffer::<u16>::alloc(16 * n as usize).unwrap();
        let mut cands = Vec::new();
        for wk in [4u32, 8] {
            for sk in [1u32, 2, 4, 5, 8, 10, 16, 20] {
                if (k / 16) % (sk * wk) != 0 || (k / 16) / (sk * wk) < 4 {
                    continue;
                }
                for occ in if wk == 8 { vec![0, 2] } else { vec![0, 3, 4] } {
                    cands.push(Plan { sk, wk, ver: 1, occ });
                }
                for occ in if wk == 8 { vec![0, 2] } else { vec![0, 4, 5] } {
                    cands.push(Plan { sk, wk, ver: 8, occ });
                }
                for occ in if wk == 8 { vec![0, 2] } else { vec![0, 4] } {
                    cands.push(Plan { sk, wk, ver: 9, occ });
                }
            }
        }
        let time = |p: Plan| -> [f64; 2] {
            // one Linear per weight copy, all with plan p (they share the device buffers by view)
            let lins: Vec<Linear> = devs.iter().map(|d| Linear::from_view(&kern, &d.name, d.view, p).unwrap()).collect();
            let ws = Workspace::new(&[&lins[0]], 16).unwrap();
            let mut r = [0f64; 2];
            for (ri, rows) in [1u32, 16].into_iter().enumerate() {
                lins[0].rotate_input(&kern, &s, x.ptr(), Dtype::Bf16, rows, &ws).unwrap();
                for l in lins.iter().take(4) {
                    l.gemv_rotated(&s, ws.xh.ptr(), y.ptr(), Dtype::Bf16, rows, &ws).unwrap();
                }
                let ms = s
                    .time(|s| {
                        for i in 0..iters as usize {
                            lins[i % copies].gemv_rotated(s, ws.xh.ptr(), y.ptr(), Dtype::Bf16, rows, &ws).unwrap();
                        }
                        Ok(())
                    })
                    .unwrap() as f64;
                r[ri] = ms * 1e3 / iters as f64;
            }
            r
        };
        let dp = default_plan(k, n, blocks, 8);
        let dt = time(dp);
        // best (sk, wk) by m1 + w16 * m16 over its best variant; then the best variant at each row count
        let mut res: Vec<(Plan, [f64; 2])> = cands.iter().map(|&p| (p, time(p))).collect();
        // the top 8 again, 4 interleaved rounds, median (single runs vary by +-5%)
        res.sort_by(|a, b| (a.1[0] + w16 * a.1[1]).partial_cmp(&(b.1[0] + w16 * b.1[1])).unwrap());
        res.truncate(8);
        // the TensorFold rule's plan always competes in the final
        if !res.iter().any(|e| e.0 == dp) {
            res.push((dp, dt));
        }
        let mut samples: Vec<Vec<[f64; 2]>> = res.iter().map(|e| vec![e.1]).collect();
        for _ in 0..4 {
            for (i, e) in res.iter().enumerate() {
                samples[i].push(time(e.0));
            }
        }
        for (i, e) in res.iter_mut().enumerate() {
            let med = |c: usize| {
                let mut v: Vec<f64> = samples[i].iter().map(|x| x[c]).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            e.1 = [med(0), med(1)];
        }
        res.sort_by(|a, b| (a.1[0] + w16 * a.1[1]).partial_cmp(&(b.1[0] + w16 * b.1[1])).unwrap());
        let dt = res.iter().find(|e| e.0 == dp).map(|e| e.1).unwrap_or(dt);
        let (bp, bt) = res[0];
        tot[0] += dt[0] * uses as f64;
        tot[1] += dt[1] * uses as f64;
        tot[2] += bt[0] * uses as f64;
        tot[3] += bt[1] * uses as f64;
        println!(
            "| {k} x {n} | {bits} | {uses} | sk{} wk{} v{} o{} | {:.1} / {:.1} | sk{} wk{} v{} o{} | {:.1} / {:.1} | {:.3} |",
            dp.sk, dp.wk, dp.ver, dp.occ, dt[0], dt[1], bp.sk, bp.wk, bp.ver, bp.occ, bt[0], bt[1], dt[0] / bt[0]
        );
        entries.push(format!(
            "    (({k}, {n}, {bits}), Plan {{ sk: {}, wk: {}, ver: {}, occ: {} }}), // {:.1} / {:.1} us",
            bp.sk, bp.wk, bp.ver, bp.occ, bt[0], bt[1]
        ));
    }
    println!(
        "count-weighted sum: default {:.2} / {:.2} ms, tuned {:.2} / {:.2} ms (m1 / m16)",
        tot[0] / 1e3,
        tot[1] / 1e3,
        tot[2] / 1e3,
        tot[3] / 1e3
    );
    println!("\npub const TUNED: &[((u32, u32, u32), Plan)] = &[");
    for e in entries {
        println!("{e}");
    }
    println!("];");
}
