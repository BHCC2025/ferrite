//! Split-K stress of the decode GEMV, engine-style: many linears (tuned plans, split-K kept with
//! FE_EXL3_SPLIT=1) share one workspace, and a long pseudo-random sequence of forwards (random linear, random
//! row count) runs back to back on one stream, each into its own output slot. Every output is compared bit for
//! bit with that (linear, rows) pair's first result. Rare races in the split-K handoff show up as DIFF lines.
//! usage: exl3-stress <model dir> [layers (16)] [batches (1000)] [calls a batch (256)]
use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_exl3::exl3_bases;
use fe_exl3::gpu::Kernels;
use fe_exl3::half::f32_to_bf16;
use fe_exl3::linear::{default_blocks, Dtype, Linear, Workspace};
use fe_exl3::st::Checkpoint;

const ROWS: [u32; 6] = [1, 2, 4, 8, 12, 16];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ck = Checkpoint::open(&args[1]).unwrap();
    let layers: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(16);
    let batches: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let calls: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(256);
    let ctx = Context::new(0).unwrap();
    let blocks = default_blocks(ctx.sm_count().unwrap() as u32);
    let kern = Kernels::load().unwrap();
    let s = Stream::new().unwrap();
    let mut lins = Vec::new();
    for b in exl3_bases(&ck) {
        let layer = b.split('.').find_map(|p| p.parse::<usize>().ok());
        let keep = !b.starts_with("model.visual")
            && !b.ends_with("lm_head")
            && (b.contains("mtp.") || layer.is_some_and(|l| l < layers));
        if keep {
            lins.push(Linear::load(&kern, &ck, &b, blocks).unwrap());
        }
    }
    let nsplit = lins.iter().filter(|l| l.plan.sk > 1).count();
    eprintln!("{} linears ({nsplit} split-K), PDL {}", lins.len(), fe_exl3::gpu::pdl::enabled());
    let refs: Vec<&Linear> = lins.iter().collect();
    let ws = Workspace::new(&refs, 16).unwrap();
    let max_k = lins.iter().map(|l| l.k as usize).max().unwrap();
    let max_n = lins.iter().map(|l| l.n as usize).max().unwrap();
    let xb: Vec<u16> = (0..16 * max_k)
        .map(|i| f32_to_bf16((((i * 2654435761usize) >> 7) % 4001) as f32 / 1000.0 - 2.0))
        .collect();
    let x = DeviceBuffer::from_slice(&xb).unwrap();
    // a pageable upload is not ordered with the non-blocking stream: let it land first
    ctx.synchronize().unwrap();
    let slot = 16 * max_n;
    let out = DeviceBuffer::<f32>::alloc(calls * slot).unwrap();
    // reference: each (linear, rows) once
    let mut want: Vec<Vec<Vec<f32>>> = Vec::new();
    for l in &lins {
        let mut per = Vec::new();
        for &m in &ROWS {
            l.forward(&kern, &s, x.ptr(), Dtype::Bf16, out.ptr(), Dtype::F32, m, &ws).unwrap();
            s.sync().unwrap();
            let mut v = vec![0f32; m as usize * l.n as usize];
            fe_cuda::memcpy_dtoh(&mut v, out.ptr()).unwrap();
            per.push(v);
        }
        want.push(per);
    }
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let (mut total, mut bad_calls) = (0usize, 0usize);
    let t = std::time::Instant::now();
    for bi in 0..batches {
        let picks: Vec<(usize, usize)> =
            (0..calls).map(|_| ((next() % lins.len() as u64) as usize, (next() % ROWS.len() as u64) as usize)).collect();
        for (c, &(li, ri)) in picks.iter().enumerate() {
            lins[li].forward(&kern, &s, x.ptr(), Dtype::Bf16, out.ptr().add::<f32>(c * slot), Dtype::F32, ROWS[ri], &ws)
                .unwrap();
        }
        s.sync().unwrap();
        for (c, &(li, ri)) in picks.iter().enumerate() {
            let w = &want[li][ri];
            let mut v = vec![0f32; w.len()];
            fe_cuda::memcpy_dtoh(&mut v, out.ptr().add::<f32>(c * slot)).unwrap();
            let diff: Vec<usize> = (0..w.len()).filter(|&i| v[i].to_bits() != w[i].to_bits()).collect();
            total += 1;
            if !diff.is_empty() {
                bad_calls += 1;
                let l = &lins[li];
                let i = diff[0];
                println!("DIFF batch {bi} call {c}: {} sk{} wk{} v{} m{} | {} of {} values, first at row {} col {} ({} vs {})",
                    l.name, l.plan.sk, l.plan.wk, l.plan.ver, ROWS[ri], diff.len(), w.len(), i / l.n as usize,
                    i % l.n as usize, v[i], w[i]);
            }
        }
        if bi % 100 == 99 {
            eprintln!("{} calls, {bad_calls} bad, {:.0} s", total, t.elapsed().as_secs_f32());
        }
    }
    println!("{total} calls, {bad_calls} with differing bits ({})", if bad_calls == 0 { "deterministic" } else { "RACE" });
    if bad_calls > 0 {
        std::process::exit(1);
    }
}
