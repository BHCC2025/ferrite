//! Two-rank collective check + latency: run `fabric-bench 0 <ip:port>` on the first node and `fabric-bench 1
//! <ip:port>` on the second (same address, rank 0's fabric IP). All-reduce (f32 sum, in place) and all-gather at decode-sized
//! messages: results checked exactly, then µs a call over `iters` back-to-back calls (CUDA events).
use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_fabric::{DataType, Fabric};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rank: u32 = args[1].parse().unwrap();
    let addr = &args[2];
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let _ctx = Context::new(0).unwrap();
    let s = Stream::new().unwrap();
    let mut fab = Fabric::connect(rank, addr).unwrap();
    if rank == 0 {
        println!("NCCL {} connected", fab.nccl_version());
        println!("| bytes a rank | all-reduce us | all-gather us |\n|---|---|---|");
    }
    for bytes in [4096usize, 8192, 16384, 20480, 40960, 65536, 163840, 327680, 1 << 20] {
        let n = bytes / 4;
        let host: Vec<f32> = (0..n).map(|i| (rank as f32 + 1.0) * (i % 1000) as f32).collect();
        let buf = DeviceBuffer::from_slice(&host).unwrap();
        fab.all_reduce_sum(&s, buf.ptr(), buf.ptr(), n, DataType::F32).unwrap();
        s.sync().unwrap();
        let got: Vec<f32> = buf.to_vec().unwrap();
        let bad = got.iter().enumerate().filter(|(i, v)| **v != 3.0 * (*i % 1000) as f32).count();
        assert_eq!(bad, 0, "all-reduce wrong at {bytes} bytes");
        let g = DeviceBuffer::<f32>::alloc(2 * n).unwrap();
        let src = DeviceBuffer::from_slice(&host).unwrap();
        fab.all_gather(&s, src.ptr(), g.ptr(), n, DataType::F32).unwrap();
        s.sync().unwrap();
        let gv: Vec<f32> = g.to_vec().unwrap();
        for r in 0..2 {
            for i in (0..n).step_by(97) {
                assert_eq!(gv[r * n + i], (r as f32 + 1.0) * (i % 1000) as f32, "all-gather wrong");
            }
        }
        fab.barrier().unwrap();
        let ar = s
            .time(|s| {
                for _ in 0..iters {
                    fab.all_reduce_sum(s, buf.ptr(), buf.ptr(), n, DataType::F32).unwrap();
                }
                Ok(())
            })
            .unwrap() as f64
            * 1e3
            / iters as f64;
        fab.barrier().unwrap();
        let ag = s
            .time(|s| {
                for _ in 0..iters {
                    fab.all_gather(s, src.ptr(), g.ptr(), n, DataType::F32).unwrap();
                }
                Ok(())
            })
            .unwrap() as f64
            * 1e3
            / iters as f64;
        fab.barrier().unwrap();
        if rank == 0 {
            println!("| {bytes} | {ar:.1} | {ag:.1} |");
        }
    }
}
