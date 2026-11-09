//! Device streaming-read roof (exl3_probe_v4 / exl3_probe_u32) over a 1 GB buffer, a few grid shapes.
use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_exl3::gpu::Kernels;

fn main() {
    let _ctx = Context::new(0).unwrap();
    let kern = Kernels::load().unwrap();
    let s = Stream::new().unwrap();
    let words = 256usize << 20; // 1 GB
    let buf = DeviceBuffer::<u32>::zeroed(words).unwrap();
    let out = DeviceBuffer::<u32>::zeroed(1).unwrap();
    let sms = 48.0;
    for name in ["imad", "idp", "lop", "iadd", "shf", "hfma2", "ffma", "prmt"] {
        let f = kern.module.function(&format!("exl3_probe_{name}")).unwrap();
        let (blocks, threads, iters) = (48u32 * 8, 256u32, 4096u32);
        let run = |s: &Stream| unsafe { s.launch(&f, blocks, threads, 0, (out.ptr(), iters, 7u32)) };
        run(&s).unwrap();
        let ms = s.time(|s| run(s)).unwrap() as f64;
        let ops = blocks as f64 * threads as f64 * iters as f64 * 8.0;
        println!("{name:>6}: {:.1} lane-ops/clk/SM at 2.9 GHz ({:.0} Gop/s)", ops / (ms * 1e-3) / sms / 2.9e9, ops / (ms * 1e-3) / 1e9);
    }
    for name in ["exl3_probe_v4", "exl3_probe_u32"] {
        let f = kern.module.function(name).unwrap();
        for (blocks, threads) in [(48u32, 1024u32), (96, 512), (192, 256), (384, 256), (768, 256), (1536, 128)] {
            let run = |s: &Stream| unsafe { s.launch(&f, blocks, threads, 0, (buf.ptr(), words as u32, out.ptr())) };
            run(&s).unwrap();
            let ms = s
                .time(|s| {
                    for _ in 0..5 {
                        run(s)?;
                    }
                    Ok(())
                })
                .unwrap();
            println!("{name} {blocks}x{threads}: {:.0} GB/s", 5.0 * words as f64 * 4.0 / (ms as f64 * 1e-3) / 1e9);
        }
    }
}
