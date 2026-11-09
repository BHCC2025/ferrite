//! A TP2 MLP block across the two Sparks against the same block on one GPU: gate / up column halves, down
//! row halves, then `reduce_to_bf16` (+ residual). Checks the TP2 result against the single-GPU one (fp32
//! summation order is the only difference) and that both ranks hold identical bits; then times one MLP block
//! a token both ways (CUDA graphs, the activation computed once on the host and reused).
//! usage: tp2-mlp <rank> <ip:port> <model dir> [layer] [rows]
use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_exl3::gpu::Kernels;
use fe_exl3::half::{bf16_to_f32, f32_to_bf16};
use fe_exl3::linear::{default_blocks, Dtype, Linear, Workspace};
use fe_exl3::st::Checkpoint;
use fe_fabric::Fabric;

fn silu_mul(g: &[f32], u: &[f32]) -> Vec<u16> {
    g.iter().zip(u).map(|(g, u)| f32_to_bf16(g / (1.0 + (-g).exp()) * u)).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rank: usize = args[1].parse().unwrap();
    let ck = Checkpoint::open(&args[3]).unwrap();
    let layer: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(10);
    let rows: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(1);
    let ctx = Context::new(0).unwrap();
    let blocks = default_blocks(ctx.sm_count().unwrap() as u32);
    let kern = Kernels::load().unwrap();
    let s = Stream::new().unwrap();
    let mut fab = Fabric::connect(rank as u32, &args[2]).unwrap();
    let roce = fab.enable_roce(1 << 20).unwrap();
    let base = |m: &str| format!("model.language_model.layers.{layer}.mlp.{m}");
    let full: Vec<Linear> = ["gate_proj", "up_proj", "down_proj"].iter().map(|m| Linear::load(&kern, &ck, &base(m), blocks).unwrap()).collect();
    let tp: Vec<Linear> = ["gate_proj", "up_proj", "down_proj"].iter().map(|m| Linear::load_tp(&kern, &ck, &base(m), rank, 2, blocks).unwrap().0).collect();
    let mut all: Vec<&Linear> = full.iter().collect();
    all.extend(tp.iter());
    let ws = Workspace::new(&all, rows as u32).unwrap();
    let (h, inter) = (5120usize, 17408usize);
    let half = inter / 2;
    let xb: Vec<u16> = (0..rows * h).map(|i| f32_to_bf16(((i * 7919 % 2003) as f32 / 1001.0) - 1.0)).collect();
    let rb: Vec<u16> = (0..rows * h).map(|i| f32_to_bf16(((i * 31 % 101) as f32 / 50.0) - 1.0)).collect();
    let x = DeviceBuffer::from_slice(&xb).unwrap();
    let f32buf = |n: usize| DeviceBuffer::<f32>::zeroed(n).unwrap();

    // single GPU
    let (g, u) = (f32buf(rows * inter), f32buf(rows * inter));
    full[0].forward(&kern, &s, x.ptr(), Dtype::Bf16, g.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
    full[1].forward(&kern, &s, x.ptr(), Dtype::Bf16, u.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
    s.sync().unwrap();
    let act = DeviceBuffer::from_slice(&silu_mul(&g.to_vec().unwrap(), &u.to_vec().unwrap())).unwrap();
    let out_ref = DeviceBuffer::from_slice(&rb).unwrap();
    full[2].forward(&kern, &s, act.ptr(), Dtype::Bf16, out_ref.ptr(), Dtype::Bf16Add, rows as u32, &ws).unwrap();
    s.sync().unwrap();
    let want: Vec<u16> = out_ref.to_vec().unwrap();

    // TP2: this rank's gate / up columns, its rows of down, the reduction
    let (gr, ur) = (f32buf(rows * half), f32buf(rows * half));
    tp[0].forward(&kern, &s, x.ptr(), Dtype::Bf16, gr.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
    tp[1].forward(&kern, &s, x.ptr(), Dtype::Bf16, ur.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
    s.sync().unwrap();
    let act_r = DeviceBuffer::from_slice(&silu_mul(&gr.to_vec().unwrap(), &ur.to_vec().unwrap())).unwrap();
    let part = f32buf(rows * h);
    let out = DeviceBuffer::from_slice(&rb).unwrap();
    tp[2].forward(&kern, &s, act_r.ptr(), Dtype::Bf16, part.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
    fab.reduce_to_bf16(&s, part.ptr(), out.ptr(), Some(out.ptr()), rows * h).unwrap();
    s.sync().unwrap();
    fab.check().unwrap();
    let got: Vec<u16> = out.to_vec().unwrap();
    let mut hist = [0usize; 4];
    let mut maxd = 0f32;
    for (a, b) in got.iter().zip(&want) {
        let d = (*a as i32 - *b as i32).unsigned_abs() as usize;
        hist[d.min(3)] += 1;
        maxd = maxd.max((bf16_to_f32(*a) - bf16_to_f32(*b)).abs());
    }
    let mine: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
    let theirs = fab.link.exchange(&mine).unwrap();
    let same = mine == theirs;
    // timing: one MLP block a token, graphs (activation reused)
    let t_full = {
        s.begin_capture().unwrap();
        full[0].forward(&kern, &s, x.ptr(), Dtype::Bf16, g.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
        full[1].forward(&kern, &s, x.ptr(), Dtype::Bf16, u.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
        full[2].forward(&kern, &s, act.ptr(), Dtype::Bf16, out_ref.ptr(), Dtype::Bf16Add, rows as u32, &ws).unwrap();
        let gr = s.end_capture().unwrap();
        gr.launch(&s).unwrap();
        s.time(|s| { for _ in 0..100 { gr.launch(s)?; } Ok(()) }).unwrap() * 10.0
    };
    fab.barrier().unwrap();
    let t_tp = {
        s.begin_capture().unwrap();
        tp[0].forward(&kern, &s, x.ptr(), Dtype::Bf16, gr.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
        tp[1].forward(&kern, &s, x.ptr(), Dtype::Bf16, ur.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
        tp[2].forward(&kern, &s, act_r.ptr(), Dtype::Bf16, part.ptr(), Dtype::F32, rows as u32, &ws).unwrap();
        fab.reduce_to_bf16(&s, part.ptr(), out.ptr(), Some(out.ptr()), rows * h).unwrap();
        let g2 = s.end_capture().unwrap();
        g2.launch(&s).unwrap();
        s.sync().unwrap();
        fab.barrier().unwrap();
        let t = s.time(|s| { for _ in 0..100 { g2.launch(s)?; } Ok(()) }).unwrap() * 10.0;
        s.sync().unwrap();
        fab.check().unwrap();
        t
    };
    fab.barrier().unwrap();
    println!(
        "rank {rank} layer {layer} rows {rows} (RoCE {roce}): TP2 vs 1-GPU bf16 ulp0 {} ulp1 {} ulp2 {} ulp3+ {} max|d| {maxd:.2e}; ranks identical: {same}; MLP block: 1 GPU {t_full:.1} us, TP2 {t_tp:.1} us",
        hist[0], hist[1], hist[2], hist[3]
    );
    // differences are fp32 summation order only: at most one bf16 rounding step (7.8e-3 at |y| ~ 1-2)
    if !same || maxd > 1.6e-2 {
        std::process::exit(1);
    }
}
