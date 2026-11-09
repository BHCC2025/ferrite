//! CUDA-graph capture of a decode-like chain of Linear forwards (PDL launches inside the capture):
//! the replayed graph must give bit-identical outputs to eager launches, and its time is reported
//! next to the eager time. usage: exl3-graphcheck <model dir> [layer]
use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_exl3::gpu::Kernels;
use fe_exl3::half::f32_to_bf16;
use fe_exl3::linear::{default_blocks, Dtype, Linear, Workspace};
use fe_exl3::st::Checkpoint;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ck = Checkpoint::open(&args[1]).unwrap();
    let l: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
    let ctx = Context::new(0).unwrap();
    let blocks = default_blocks(ctx.sm_count().unwrap() as u32);
    let kern = Kernels::load().unwrap();
    let s = Stream::new().unwrap();
    let p = |m: &str| format!("model.language_model.layers.{l}.{m}");
    // an attention layer's projections then its MLP: q, k, v, o (+resid), gate, up, down (+resid)
    let names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];
    let lins: Vec<Linear> = names.iter().map(|n| Linear::load(&kern, &ck, &p(n), blocks).unwrap()).collect();
    let refs: Vec<&Linear> = lins.iter().collect();
    for rows in [1u32, 8] {
        let ws = Workspace::new(&refs, rows).unwrap();
        let maxk = lins.iter().map(|l| l.k).max().unwrap() as usize;
        let xb: Vec<u16> = (0..rows as usize * maxk).map(|i| f32_to_bf16(((i * 7919 % 2003) as f32 / 1001.0) - 1.0)).collect();
        let x = DeviceBuffer::from_slice(&xb).unwrap();
        let outs: Vec<DeviceBuffer<u16>> = lins.iter().map(|l| DeviceBuffer::<u16>::zeroed(rows as usize * l.n as usize).unwrap()).collect();
        let resid = DeviceBuffer::<u16>::zeroed(rows as usize * 5120).unwrap();
        let chain = |s: &Stream| -> fe_exl3::Result<()> {
            for (i, l) in lins.iter().enumerate() {
                let add = names[i].ends_with("o_proj") || names[i].ends_with("down_proj");
                let (y, dt) = if add { (resid.ptr(), Dtype::Bf16Add) } else { (outs[i].ptr(), Dtype::Bf16) };
                l.forward(&kern, s, x.ptr(), Dtype::Bf16, y, dt, rows, &ws)?;
            }
            Ok(())
        };
        // eager
        s.memset_d8(resid.ptr(), 0, rows as usize * 5120 * 2).unwrap();
        chain(&s).unwrap();
        s.sync().unwrap();
        let eager: Vec<Vec<u16>> = outs.iter().map(|o| o.to_vec().unwrap()).collect();
        let eager_r: Vec<u16> = resid.to_vec().unwrap();
        // graph
        s.begin_capture().unwrap();
        chain(&s).unwrap();
        let g = s.end_capture().unwrap();
        for o in &outs {
            s.memset_d8(o.ptr(), 0, o.bytes()).unwrap();
        }
        s.memset_d8(resid.ptr(), 0, rows as usize * 5120 * 2).unwrap();
        g.launch(&s).unwrap();
        s.sync().unwrap();
        let mut diff = 0usize;
        for (o, e) in outs.iter().zip(&eager) {
            diff += o.to_vec().unwrap().iter().zip(e).filter(|(a, b)| a != b).count();
        }
        diff += resid.to_vec().unwrap().iter().zip(&eager_r).filter(|(a, b)| a != b).count();
        let n = 50;
        let te = s.time(|s| { for _ in 0..n { chain(s).map_err(|_| fe_cuda::CuError { code: 1, call: "chain" })?; } Ok(()) }).unwrap() / n as f32;
        let tg = s.time(|s| { for _ in 0..n { g.launch(s)?; } Ok(()) }).unwrap() / n as f32;
        println!(
            "rows {rows}: graph vs eager outputs {} | 7 linears: eager {:.1} us, graph {:.1} us (PDL {})",
            if diff == 0 { "identical".to_string() } else { format!("DIFFER in {diff}") },
            te * 1e3,
            tg * 1e3,
            fe_exl3::gpu::pdl::enabled()
        );
        if diff != 0 {
            std::process::exit(1);
        }
    }
}
