//! TP2 reduction check + latency (RoCE one-shot vs NCCL): run `fabric-reduce 0 <ip:port>` on the first node
//! and `fabric-reduce 1 <ip:port>` on the second. out = bf16(resid + p0 + p1) checked exactly against the CPU, on
//! both paths, then µs a call over `iters` back-to-back calls, and a CUDA-graph replay of 64 reductions; then the
//! bf16-partial RoCE reduction (reduce_bf16_to_bf16, half the bytes) on the same values rounded to bf16.
use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_fabric::Fabric;

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rank: u32 = args[1].parse().unwrap();
    let addr = &args[2];
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(2000);
    let _ctx = Context::new(0).unwrap();
    let s = Stream::new().unwrap();
    let mut fab = Fabric::connect(rank, addr).unwrap();
    let roce = fab.enable_roce(1 << 20).unwrap();
    if let Ok(v) = std::env::var("FE_SPIN") {
        fab.spin_limit = v.parse().unwrap();
    }
    if rank == 0 {
        println!("RoCE {} (NCCL {})", if roce { fab.roce.as_ref().unwrap().hca.clone() } else { "off".into() }, fab.nccl_version());
        println!("| f32 elems | bytes a rank | RoCE us | NCCL+cvt us | graph of 64 RoCE us/call | bf16 partials RoCE us |\n|---|---|---|---|---|---|");
    }
    for n in [1024usize, 5120, 10240, 40960, 81920, 163840, 262144] {
        let part: Vec<f32> = (0..n).map(|i| ((i * 7 + rank as usize * 13) % 1001) as f32 / 101.0 - 3.0).collect();
        let peer: Vec<f32> = (0..n).map(|i| ((i * 7 + (1 - rank) as usize * 13) % 1001) as f32 / 101.0 - 3.0).collect();
        let resid: Vec<u16> = (0..n).map(|i| bf16((i % 17) as f32 * 0.25)).collect();
        let want: Vec<u16> = (0..n)
            .map(|i| {
                let (a, b) = if rank == 0 { (part[i], peer[i]) } else { (peer[i], part[i]) };
                bf16(f32::from_bits((resid[i] as u32) << 16) + (a + b))
            })
            .collect();
        let mut times = [0f64; 3];
        for path in 0..2 {
            // path 0: RoCE, path 1: NCCL
            fab.use_roce = path == 0 && roce;
            let p = DeviceBuffer::from_slice(&part).unwrap();
            let out = DeviceBuffer::<u16>::zeroed(n).unwrap();
            let r = DeviceBuffer::from_slice(&resid).unwrap();
            fab.reduce_to_bf16(&s, p.ptr(), out.ptr(), Some(r.ptr()), n).unwrap();
            s.sync().unwrap();
            fab.check().unwrap();
            let got: Vec<u16> = out.to_vec().unwrap();
            let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(bad, 0, "path {path} n {n}: {bad} wrong");
            // timing (the partial is re-uploaded for NCCL, which reduces in place)
            fab.barrier().unwrap();
            let t = s
                .time(|s| {
                    for _ in 0..iters {
                        fab.reduce_to_bf16(s, p.ptr(), out.ptr(), Some(r.ptr()), n).unwrap();
                    }
                    Ok(())
                })
                .unwrap() as f64
                * 1e3
                / iters as f64;
            s.sync().unwrap();
            fab.check().unwrap();
            fab.barrier().unwrap();
            times[path] = t;
            if path == 0 && roce {
                s.begin_capture().unwrap();
                for _ in 0..64 {
                    fab.reduce_to_bf16(&s, p.ptr(), out.ptr(), Some(r.ptr()), n).unwrap();
                }
                let g = s.end_capture().unwrap();
                g.launch(&s).unwrap();
                s.sync().unwrap();
                fab.barrier().unwrap();
                let tg = s.time(|s| { for _ in 0..20 { g.launch(s)?; } Ok(()) }).unwrap() as f64 * 1e3 / (20.0 * 64.0);
                s.sync().unwrap();
                fab.check().unwrap();
                fab.barrier().unwrap();
                times[2] = tg;
            }
        }
        let mut t16 = 0f64;
        if roce && n % 8 == 0 {
            fab.use_roce = true;
            let p16: Vec<u16> = part.iter().map(|&v| bf16(v)).collect();
            let peer16: Vec<u16> = peer.iter().map(|&v| bf16(v)).collect();
            let f = |h: u16| f32::from_bits((h as u32) << 16);
            let want16: Vec<u16> = (0..n).map(|i| {
                let (a, b) = if rank == 0 { (p16[i], peer16[i]) } else { (peer16[i], p16[i]) };
                bf16((f(a) + f(b)) + f(resid[i]))
            }).collect();
            let p = DeviceBuffer::from_slice(&p16).unwrap();
            let out = DeviceBuffer::<u16>::zeroed(n).unwrap();
            let r = DeviceBuffer::from_slice(&resid).unwrap();
            fab.reduce_bf16_to_bf16(&s, p.ptr(), out.ptr(), Some(r.ptr()), n).unwrap();
            s.sync().unwrap();
            fab.check().unwrap();
            let got: Vec<u16> = out.to_vec().unwrap();
            let bad = got.iter().zip(&want16).filter(|(a, b)| a != b).count();
            assert_eq!(bad, 0, "bf16 partials n {n}: {bad} wrong");
            fab.barrier().unwrap();
            t16 = s.time(|s| {
                for _ in 0..iters {
                    fab.reduce_bf16_to_bf16(s, p.ptr(), out.ptr(), Some(r.ptr()), n).unwrap();
                }
                Ok(())
            }).unwrap() as f64 * 1e3 / iters as f64;
            s.sync().unwrap();
            fab.check().unwrap();
            fab.barrier().unwrap();
        }
        if rank == 0 {
            println!("| {n} | {} | {:.1} | {:.1} | {:.1} | {:.1} |", n * 4, times[0], times[1], times[2], t16);
        }
    }
}
