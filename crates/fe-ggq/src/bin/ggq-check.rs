//! ggq-check <model.gguf> <deq|gemv|bench>: GPU kernels vs the host reference on real tensors of every type in the
//! file, and timings.
//! - deq: dequant f32 must be bit-exact; bf16 must equal the RNE rounding of the reference.
//! - gemv: decode GEMV (B = 1..8) vs f64 host dot of the dequantized weights with the q8_1-dequantized activations
//!   (should agree to f32 rounding), and vs the unquantized activations (the q8_1 activation error, as ggml).
//! - bench: GEMV GB/s on whole real tensors (expert tensors as one tall matrix), B = 1, 2, 4, 8.
use anyhow::Result;
use fe_cuda::{Context, DevPtr, DeviceBuffer, Stream};
use fe_ggq::{cpu, gpu, refq, Ggq, GgmlType, Out, Q8Act, SortWs};
use fe_gguf::{Gguf, TensorInfo};
use std::collections::BTreeMap;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(&a[1])?;
    let what = a.get(2).map(|s| s.as_str()).unwrap_or("deq");
    let _ctx = Context::new(0)?;
    let k = Ggq::load()?;
    let s = Stream::new()?;
    let mut first: BTreeMap<u32, &TensorInfo> = BTreeMap::new();
    let mut big: BTreeMap<u32, &TensorInfo> = BTreeMap::new();
    for t in &g.tensors {
        first.entry(t.ty as u32).or_insert(t);
        if t.name != "per_layer_token_embd.weight" {
            let e = big.entry(t.ty as u32).or_insert(t);
            if t.bytes > e.bytes {
                *e = t;
            }
        }
    }
    match what {
        "deq" => deq(&g, &k, &s, &first, &big)?,
        "gemv" => gemv_check(&g, &k, &s, &first)?,
        "bench" => bench(&g, &k, &s, a.get(3).map(|s| s.as_str()))?,
        "moe" => moe(&g, &k, &s)?,
        "gemm" => gemm_check(&g, &k, &s)?,
        "mmq" => mmq_check(&g, &k, &s)?,
        "bf16" => bf16_check(&g, &k, &s)?,
        "hc" => hc_check(&g, &k, &s)?,
        "hcb" => hc_bench(&g, &k, &s)?,
        "hcdump" => hc_dump(&g, &k, &s, &a[3..])?,
        "binv" => binv(&g, &k, &s, &first)?,
        "fa" => fa_check(&k, &s, &a[3..])?,
        "mmaprobe" => {
            let o = DeviceBuffer::<f32>::zeroed(1)?;
            let (e0, e1) = (fe_cuda::Event::new()?, fe_cuda::Event::new()?);
            for mode in 0..6u32 {
                let n = 20000u32;
                let grid = 36 * 8;
                k.mma_probe(&s, o.ptr(), mode, 100, grid)?;
                s.record(&e0)?;
                k.mma_probe(&s, o.ptr(), mode, n, grid)?;
                s.record(&e1)?;
                s.sync()?;
                let ms = e1.elapsed_since(&e0)? as f64;
                let ops = [4096.0, 2048.0, 8192.0, 8192.0, 8192.0, 8192.0][mode as usize] * 2.0 * 8.0 * n as f64 * (grid * 4) as f64 / 2.0;
                println!("mode {mode} ({}): {:.1} T(FL)OP/s", ["f16 m16n8k16", "f16 m16n8k8", "s8 m16n8k32", "s8 k32 + f32 epilogue (mul mul add)", "s8 k32 + f32 epilogue (mul fma)", "2 x s8 k16 + imad + epilogue (Q6_K form)"][mode as usize], ops / ms / 1e9);
            }
        }
        "guprof" => {
            // moe_gu_q8 of one layer, B1, 10 distinct experts (for a profiler); layer a[3] (default 17)
            let layer: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(17);
            let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
            let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
            let nexp = gt.dims[2] as usize;
            let (dg, du) = (DeviceBuffer::from_slice(g.data(gt))?, DeviceBuffer::from_slice(g.data(ut))?);
            let tg: Vec<u64> = (0..nexp).map(|e| dg.ptr().0 + (e * gt.slice_bytes()) as u64).collect();
            let tu: Vec<u64> = (0..nexp).map(|e| du.ptr().0 + (e * ut.slice_bytes()) as u64).collect();
            let (dtg, dtu) = (DeviceBuffer::from_slice(&tg)?, DeviceBuffer::from_slice(&tu)?);
            let nt: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(1);
            let ids: Vec<i32> = (0..nt * 10).map(|p| ((p * 37) % nexp) as i32).collect();
            let dids = DeviceBuffer::from_slice(&ids)?;
            let x: Vec<f32> = (0..nt * 2560).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
            let dx = DeviceBuffer::from_slice(&x)?;
            let mut xa = Q8Act::alloc(8, 2560)?;
            let mut ha = Q8Act::alloc(80, 640)?;
            k.quantize_q8_1(&s, dx.ptr(), 2560, false, nt, 2560, &mut xa)?;
            for _ in 0..3 {
                k.moe_gu_q8(&s, gt.ty, dtg.ptr(), dtu.ptr(), gt.row_bytes(), dids.ptr(), nt, 10, &xa, 640, &mut ha, None, None)?;
            }
            s.sync()?;
        }
        "bench24" => {
            // decode GEMV at B = 1, 8, 12, 16, 20, 24 (default config, as the engine calls it): weight GB/s per call
            let names = ["blk.1.ffn_gate_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.17.ffn_gate_exps.weight",
                "blk.48.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight", "blk.1.ffn_down_exps.weight", "output.weight",
                "blk.0.attn_qkv.weight", "blk.1.attn_qkv.weight", "blk.2.attn_gate.weight", "blk.47.ffn_down_shexp.weight"];
            let mut rng = Lcg(7);
            let mut seen = vec![];
            for name in names {
                let Some(t) = g.tensor(name) else { continue };
                if !gpu::GEMV_TYPES.contains(&t.ty) || seen.contains(&t.ty) {
                    continue;
                }
                seen.push(t.ty);
                let (kk, n) = (t.ne0(), t.n_rows());
                let dw = DeviceBuffer::from_slice(g.data(t))?;
                let mut act = Q8Act::alloc(24, kk)?;
                let x: Vec<f32> = (0..24 * kk).map(|_| rng.f()).collect();
                let dx = DeviceBuffer::from_slice(&x)?;
                let dy = DeviceBuffer::<f32>::alloc(24 * n)?;
                let mut line = format!("{:8} {:>28} [{kk:5} x {n:6}] GB/s:", t.ty.name(), name);
                for b in [1usize, 8, 12, 16, 20, 24] {
                    k.quantize_q8_1(&s, dx.ptr(), kk, false, b, kk, &mut act)?;
                    k.gemv(&s, t.ty, dw.ptr(), t.row_bytes(), n, &act, dy.ptr(), false, None)?;
                    let mut best = f32::MAX;
                    for _ in 0..5 {
                        let (e0, e1) = (fe_cuda::Event::new()?, fe_cuda::Event::new()?);
                        s.record(&e0)?;
                        k.gemv(&s, t.ty, dw.ptr(), t.row_bytes(), n, &act, dy.ptr(), false, None)?;
                        s.record(&e1)?;
                        s.sync()?;
                        best = best.min(e1.elapsed_since(&e0)?);
                    }
                    line += &format!("  B{b} {:.0}", t.bytes as f64 / best as f64 / 1e6);
                }
                println!("{line}");
            }
            // BF16 hc shapes (gemv_bf16, fused hc_down + hc_up) and one MoE layer (moe_gu_q8 + moe_down, 10 experts per
            // token from a pool of 64 so tokens share experts) per call, ms
            let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f32> {
                f()?;
                let mut best = f32::MAX;
                for _ in 0..5 {
                    let (e0, e1) = (fe_cuda::Event::new()?, fe_cuda::Event::new()?);
                    s.record(&e0)?;
                    f()?;
                    s.record(&e1)?;
                    s.sync()?;
                    best = best.min(e1.elapsed_since(&e0)?);
                }
                Ok(best)
            };
            let (dt, it, ut, nt) = (g.tensor("blk.0.hc_attn_down.weight").unwrap(), g.tensor("blk.0.hc_attn_inject.weight").unwrap(),
                g.tensor("blk.0.hc_attn_up.weight").unwrap(), g.tensor("blk.0.hc_attn_norm.weight").unwrap());
            let mut cat = g.data(dt).to_vec();
            cat.extend_from_slice(g.data(it));
            let nw: Vec<f32> = g.data(nt).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let (dcat, du, dnw) = (DeviceBuffer::from_slice(&cat)?, DeviceBuffer::from_slice(g.data(ut))?, DeviceBuffer::from_slice(&nw)?);
            let hd = 10240usize;
            let dres = DeviceBuffer::from_slice(&(0..24 * hd).map(|_| rng.f()).collect::<Vec<f32>>())?;
            let (dlo, dxn, dmix) = (DeviceBuffer::<f32>::alloc(24 * 324)?, DeviceBuffer::<f32>::alloc(24 * hd)?, DeviceBuffer::<f32>::alloc(24 * 2560)?);
            let mut line = "hc (gemv_bf16 down 10240->324 | fused hc_down + hc_up) ms:".to_string();
            for b in [1usize, 8, 12, 16, 20, 24] {
                let t1 = time(&mut || k.gemv_bf16(&s, dcat.ptr(), 324, hd, dres.ptr(), hd, b, dlo.ptr(), false))?;
                let t2 = time(&mut || { k.hc_down(&s, dlo.ptr(), 324, dcat.ptr(), 324, dres.ptr(), hd, dnw.ptr(), 1e-6, 0.25, 320, dxn.ptr(), b)?;
                    k.hc_up(&s, dmix.ptr(), 2560, du.ptr(), dlo.ptr(), 324, dxn.ptr(), hd, 0.25, b) })?;
                line += &format!("  B{b} {t1:.3} | {t2:.3}");
            }
            println!("{line}");
            let (ne, nf, topk) = (2560usize, 640usize, 10usize);
            for layer in [0usize, 48] {
                let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
                let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
                let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
                let nexp = gt.dims[2] as usize;
                let (dg, du, dd) = (DeviceBuffer::from_slice(g.data(gt))?, DeviceBuffer::from_slice(g.data(ut))?, DeviceBuffer::from_slice(g.data(dt))?);
                let tab = |b: &DeviceBuffer<u8>, sb: usize| -> Vec<u64> { (0..nexp).map(|e| b.ptr().0 + (e * sb) as u64).collect() };
                let (tg, tu, td) = (DeviceBuffer::from_slice(&tab(&dg, gt.slice_bytes()))?, DeviceBuffer::from_slice(&tab(&du, ut.slice_bytes()))?,
                    DeviceBuffer::from_slice(&tab(&dd, dt.slice_bytes()))?);
                let ids: Vec<i32> = (0..24 * topk).map(|p| (((p / topk) * 7 + (p % topk) * 13) % 64 * (nexp / 64)) as i32).collect();
                let dids = DeviceBuffer::from_slice(&ids)?;
                let dx = DeviceBuffer::from_slice(&(0..24 * ne).map(|_| rng.f()).collect::<Vec<f32>>())?;
                let mut xa = Q8Act::alloc(24, ne)?;
                let mut ha = Q8Act::alloc(24 * topk, nf)?;
                let dy = DeviceBuffer::<f32>::alloc(24 * topk * ne)?;
                let mut line = format!("MoE layer {layer} {}/{} (moe_gu_q8 + moe_down) ms:", gt.ty.name(), dt.ty.name());
                for b in [1usize, 8, 12, 16, 20, 24] {
                    k.quantize_q8_1(&s, dx.ptr(), ne, false, b, ne, &mut xa)?;
                    let t = time(&mut || { k.moe_gu_q8(&s, gt.ty, tg.ptr(), tu.ptr(), gt.row_bytes(), dids.ptr(), b, topk, &xa, nf, &mut ha, None, None)?;
                        k.moe_down(&s, dt.ty, td.ptr(), dt.row_bytes(), dids.ptr(), b, topk, &ha, ne, dy.ptr(), None) })?;
                    line += &format!("  B{b} {t:.3}");
                }
                println!("{line}");
            }
        }
        "g16" => {
            // BF16 GEMM: hc down (k 10240 -> n 384) and hc up (k 320 -> n 10240) at m = a[3] (default 4096), the cp.async
            // kernel vs the generic one, bit for bit
            let m: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(4096);
            let mut rng = Lcg(5);
            for (kk, n, nreal) in [(10240usize, 384usize, 324usize), (320, 10240, 10240), (2560, 512, 512)] {
                let x: Vec<u16> = (0..m * kk).map(|_| refq::f32_to_bf16(rng.f())).collect();
                let wv: Vec<u16> = (0..n * kk).map(|i| if i / kk < nreal { refq::f32_to_bf16(rng.f() * 0.05) } else { 0 }).collect();
                let (dx, dw) = (DeviceBuffer::from_slice(&x)?, DeviceBuffer::from_slice(&wv)?);
                let (y1, y2) = (DeviceBuffer::<f32>::alloc(m * n)?, DeviceBuffer::<f32>::alloc(m * n)?);
                let run = |y: DevPtr, fast: bool| -> Result<f32> {
                    unsafe { std::env::set_var("FE_GEMM16", if fast { "1" } else { "0" }) };
                    k.gemm(&s, GgmlType::BF16, y, n, dx.ptr(), kk, dw.ptr(), kk * 2, m, n, kk, false)?;
                    let mut best = f32::MAX;
                    for _ in 0..3 {
                        let (e0, e1) = (fe_cuda::Event::new()?, fe_cuda::Event::new()?);
                        s.record(&e0)?;
                        k.gemm(&s, GgmlType::BF16, y, n, dx.ptr(), kk, dw.ptr(), kk * 2, m, n, kk, false)?;
                        s.record(&e1)?;
                        s.sync()?;
                        best = best.min(e1.elapsed_since(&e0)?);
                    }
                    Ok(best)
                };
                let t1 = run(y1.ptr(), false)?;
                let t2 = run(y2.ptr(), true)?;
                let (a1, a2) = (y1.to_vec()?, y2.to_vec()?);
                let nd = a1.iter().zip(&a2).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                let fl = 2.0 * m as f64 * n as f64 * kk as f64;
                println!("bf16 gemm m {m} k {kk} n {n}: generic {t1:.3} ms {:.1} TFLOP/s | cp.async {t2:.3} ms {:.1} TFLOP/s, {nd} outputs differ",
                    fl / t1 as f64 / 1e9, fl / t2 as f64 / 1e9);
            }
        }
        "mmqd" => {
            // dense int8 GEMM timing: mmqd <m> [layers...]: attn_qkv / attn_gate / ssm_out of the given layers
            let m: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(16384);
            let layers: Vec<usize> = if a.len() > 4 { a[4..].iter().filter_map(|v| v.parse().ok()).collect() } else { vec![0, 1, 2] };
            for l in layers {
                for nm in ["attn_qkv", "attn_gate", "ssm_out"] {
                    let Some(t) = g.tensor(&format!("blk.{l}.{nm}.weight")) else { continue };
                    let (kk, n) = (t.ne0(), t.n_rows());
                    let x: Vec<f32> = (0..m * kk).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
                    let dx = DeviceBuffer::from_slice(&x)?;
                    let mut xa = Q8Act::alloc(m, kk)?;
                    k.quantize_q8_1(&s, dx.ptr(), kk, false, m, kk, &mut xa)?;
                    drop(dx);
                    let dw = DeviceBuffer::from_slice(g.data(t))?;
                    let dy = DeviceBuffer::<f32>::alloc(m * n)?;
                    k.mmq(&s, t.ty, dy.ptr(), n, &xa, dw.ptr(), t.row_bytes(), n, false)?;
                    let mut best = f32::MAX;
                    for _ in 0..3 {
                        let (e0, e1) = (fe_cuda::Event::new()?, fe_cuda::Event::new()?);
                        s.record(&e0)?;
                        k.mmq(&s, t.ty, dy.ptr(), n, &xa, dw.ptr(), t.row_bytes(), n, false)?;
                        s.record(&e1)?;
                        s.sync()?;
                        let ms = e1.elapsed_since(&e0)?;
                        best = best.min(ms);
                    }
                    println!("blk.{l}.{nm} {} [{kk} -> {n}] m {m}: {best:.2} ms, {:.1} TOPS", t.ty.name(),
                        2.0 * m as f64 * n as f64 * kk as f64 / best as f64 / 1e9);
                }
            }
        }
        "mmqprof" => {
            // one dense int8 GEMM (for a profiler): tensor a[3] (default blk.0.attn_gate.weight), m = 2048
            let name = a.get(3).map(|s| s.as_str()).unwrap_or("blk.0.attn_gate.weight");
            let t = g.tensor(name).unwrap();
            let (kk, n, m) = (t.ne0(), t.n_rows(), 2048usize);
            let x: Vec<f32> = (0..m * kk).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
            let dx = DeviceBuffer::from_slice(&x)?;
            let mut xa = Q8Act::alloc(m, kk)?;
            k.quantize_q8_1(&s, dx.ptr(), kk, false, m, kk, &mut xa)?;
            let dw = DeviceBuffer::from_slice(g.data(t))?;
            let dy = DeviceBuffer::<f32>::alloc(m * n)?;
            for _ in 0..2 {
                k.mmq(&s, t.ty, dy.ptr(), n, &xa, dw.ptr(), t.row_bytes(), n, false)?;
            }
            s.sync()?;
        }
        "peak" => {
            let f = k.module.function("ggq_mma_peak")?;
            let out = DeviceBuffer::<f32>::alloc(36 * 64 * 256)?;
            for (int8, name, flop) in [(0u32, "bf16 m16n8k16", 2.0 * 16.0 * 8.0 * 16.0), (1, "s8 m16n8k32", 2.0 * 16.0 * 8.0 * 32.0)] {
                let iters = 4096u32;
                let grid = 36 * 16;
                unsafe { s.launch(&f, grid, 256, 0, (out.ptr(), iters, int8))? };
                let ms = s.time(|s| unsafe { s.launch(&f, grid, 256, 0, (out.ptr(), iters, int8)) })?;
                let tf = grid as f64 * 8.0 * iters as f64 * 8.0 * flop / ms as f64 / 1e9;
                println!("{name}: {tf:.1} T(FL)OP/s");
            }
        }
        _ => anyhow::bail!("unknown mode {what}"),
    }
    Ok(())
}

fn deq(g: &Gguf, k: &Ggq, s: &Stream, first: &BTreeMap<u32, &TensorInfo>, big: &BTreeMap<u32, &TensorInfo>)
    -> Result<()> {
    for t in first.values() {
        if gpu::stem(t.ty).is_none() {
            continue;
        }
        let rows = t.n_rows().min(256);
        let n = rows * t.ne0();
        let bytes = &g.data(t)[..t.ty.row_bytes(t.ne0()) * rows];
        let mut r = vec![0f32; n];
        refq::dequant(t.ty, bytes, &mut r);
        let src = DeviceBuffer::from_slice(bytes)?;
        let df = DeviceBuffer::<f32>::alloc(n)?;
        let db = DeviceBuffer::<u16>::alloc(n)?;
        k.dequant(s, t.ty, src.ptr(), n, df.ptr(), Out::F32)?;
        k.dequant(s, t.ty, src.ptr(), n, db.ptr(), Out::Bf16)?;
        s.sync()?;
        let hf = df.to_vec()?;
        let hb = db.to_vec()?;
        let bad_f = r.iter().zip(&hf).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let bad_b = r.iter().zip(&hb).filter(|(x, y)| refq::f32_to_bf16(**x) != **y).count();
        let tb = big[&(t.ty as u32)];
        let (bs, by) = tb.ty.block();
        let unit = bs.max(8);
        let nv = ((tb.n_elements() as usize).min(1 << 28) / unit) * unit;
        let inb = nv / bs * by;
        let bsrc = DeviceBuffer::from_slice(&g.data(tb)[..inb])?;
        let bdst = DeviceBuffer::<u16>::alloc(nv)?;
        k.dequant(s, tb.ty, bsrc.ptr(), nv, bdst.ptr(), Out::Bf16)?;
        let ms = s.time(|s| {
            for _ in 0..5 {
                k.dequant(s, tb.ty, bsrc.ptr(), nv, bdst.ptr(), Out::Bf16).unwrap();
            }
            Ok(())
        })? / 5.0;
        println!("{:8} {:7} values: f32 bit-exact {} ({bad_f} diffs), bf16 RNE-exact {} ({bad_b} diffs) | {:>34}: {:.2} ms, {:.0} GB/s in + {:.0} GB/s out",
            t.ty.name(), n, bad_f == 0, bad_b == 0, tb.name, ms, inb as f64 / ms as f64 / 1e6, nv as f64 * 2.0 / ms as f64 / 1e6);
    }
    Ok(())
}

fn gemv_check(g: &Gguf, k: &Ggq, s: &Stream, first: &BTreeMap<u32, &TensorInfo>) -> Result<()> {
    let mut rng = Lcg(12345);
    for t in first.values() {
        if !gpu::GEMV_TYPES.contains(&t.ty) {
            continue;
        }
        let kk = t.ne0();
        let n = t.n_rows().min(1024);
        let rb = t.row_bytes();
        let wbytes = &g.data(t)[..rb * n];
        let mut wf = vec![0f32; n * kk];
        refq::dequant(t.ty, wbytes, &mut wf);
        let dw = DeviceBuffer::from_slice(wbytes)?;
        let mut act = Q8Act::alloc(8, kk)?;
        let dy = DeviceBuffer::<f32>::alloc(8 * n)?;
        let mut worst_q = 0f64;
        let mut worst_x = 0f64;
        let mut worst_x32 = 0f64;
        for b in 1..=8usize {
            let x: Vec<f32> = (0..b * kk).map(|_| rng.f()).collect();
            let dx = DeviceBuffer::from_slice(&x)?;
            k.quantize_q8_1(s, dx.ptr(), kk, false, b, kk, &mut act)?;
            // the f32-sum option, vs the unquantized-activation reference only
            act.sum_mode = fe_ggq::SumMode::DSumQ;
            k.quantize_q8_1(s, dx.ptr(), kk, false, b, kk, &mut act)?;
            k.gemv(s, t.ty, dw.ptr(), rb, n, &act, dy.ptr(), false, None)?;
            s.sync()?;
            let y32 = dy.to_vec()?;
            act.sum_mode = fe_ggq::SumMode::F16X;
            k.quantize_q8_1(s, dx.ptr(), kk, false, b, kk, &mut act)?;
            k.gemv(s, t.ty, dw.ptr(), rb, n, &act, dy.ptr(), false, None)?;
            s.sync()?;
            let y = dy.to_vec()?;
            // every (lpr, rpl) config must give the same bits as each other up to summation order
            for rpl in [1u32, 2, 4] {
                for lpr in [4u32, 8, 16, 32] {
                    if kk / 32 < lpr as usize {
                        continue;
                    }
                    k.gemv_cfg(s, t.ty, dw.ptr(), rb, n, &act, dy.ptr(), false, lpr, rpl)?;
                    s.sync()?;
                    let y2 = dy.to_vec()?;
                    let md = y.iter().zip(&y2).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    let mx = y.iter().map(|a| a.abs()).fold(0f32, f32::max);
                    if md > 1e-4 * mx {
                        println!("  CONFIG MISMATCH {} b{b} lpr {lpr} rpl {rpl}: {md:e}", t.ty.name());
                    }
                }
            }
            for r in 0..b {
                let (q, d, sx) = refq::quantize_q8_1(&x[r * kk..(r + 1) * kk]);
                let xq: Vec<f64> = (0..kk).map(|i| q[i] as f64 * d[i / 32] as f64).collect();
                let mut num_q = 0f64;
                let mut num_x = 0f64;
                let mut num_x32 = 0f64;
                let mut den = 0f64;
                for row in 0..n {
                    let wr = &wf[row * kk..(row + 1) * kk];
                    let yq = refq::dot_q8(t.ty, &wbytes[row * rb..(row + 1) * rb], &q, &d, &sx);
                    let _ = &xq;
                    let yx: f64 = wr.iter().zip(&x[r * kk..]).map(|(w, x)| *w as f64 * *x as f64).sum();
                    let yg = y[r * n + row] as f64;
                    num_q = num_q.max((yg - yq).abs());
                    num_x = num_x.max((yg - yx).abs());
                    num_x32 = num_x32.max((y32[r * n + row] as f64 - yx).abs());
                    den += yx * yx;
                }
                let rms = (den / n as f64).sqrt();
                worst_q = worst_q.max(num_q / rms);
                worst_x = worst_x.max(num_x / rms);
                worst_x32 = worst_x32.max(num_x32 / rms);
            }
        }
        println!("{:8} {:>34} [{kk} x {n}] B=1..8: max |err| / rms(y): vs q8-activation ref {worst_q:.2e}, vs f32-activation ref {worst_x:.2e} (with d*sum(q) sums {worst_x32:.2e})",
            t.ty.name(), t.name);
    }
    Ok(())
}

fn bench(g: &Gguf, k: &Ggq, s: &Stream, filter: Option<&str>) -> Result<()> {
    let names = ["blk.1.ffn_gate_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.17.ffn_gate_exps.weight",
        "blk.48.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight", "blk.1.ffn_down_exps.weight", "output.weight",
        "blk.0.attn_qkv.weight", "blk.0.ssm_out.weight", "blk.1.attn_qkv.weight", "blk.47.ffn_down_shexp.weight",
        "token_embd.weight", "blk.0.attn_gate.weight"];
    let mut rng = Lcg(7);
    for name in names {
        let Some(t) = g.tensor(name) else { continue };
        if !gpu::GEMV_TYPES.contains(&t.ty) {
            continue;
        }
        if let Some(f) = filter {
            if !t.ty.name().eq_ignore_ascii_case(f) {
                continue;
            }
        }
        let kk = t.ne0();
        let n = t.n_rows();
        let dw = DeviceBuffer::from_slice(g.data(t))?;
        let mut act = Q8Act::alloc(8, kk)?;
        let x: Vec<f32> = (0..8 * kk).map(|_| rng.f()).collect();
        let dx = DeviceBuffer::from_slice(&x)?;
        let dy = DeviceBuffer::<f32>::alloc(8 * n)?;
        let mut line = format!("{:8} {:>28} [{kk:5} x {n:6}] {:5.0} MB GB/s default [best cfg]:", t.ty.name(), name, t.bytes as f64 / 1e6);
        for b in [1usize, 2, 3, 4, 5, 8] {
            k.quantize_q8_1(s, dx.ptr(), kk, false, b, kk, &mut act)?;
            let mut best = (f32::MAX, 0u32, 0u32);
            for rpl in [1u32, 2, 4] {
                for lpr in [4u32, 8, 16, 32] {
                    if kk / 32 < lpr as usize {
                        continue;
                    }
                    k.gemv_cfg(s, t.ty, dw.ptr(), t.row_bytes(), n, &act, dy.ptr(), false, lpr, rpl)?;
                    let ms = s.time(|s| {
                        for _ in 0..10 {
                            k.gemv_cfg(s, t.ty, dw.ptr(), t.row_bytes(), n, &act, dy.ptr(), false, lpr, rpl).unwrap();
                        }
                        Ok(())
                    })? / 10.0;
                    if ms < best.0 {
                        best = (ms, lpr, rpl);
                    }
                }
            }
            let msd = s.time(|s| {
                for _ in 0..10 {
                    k.gemv(s, t.ty, dw.ptr(), t.row_bytes(), n, &act, dy.ptr(), false, None).unwrap();
                }
                Ok(())
            })? / 10.0;
            line += &format!("  B{b} {:4.0} [{:4.0}]", t.bytes as f64 / msd as f64 / 1e6, t.bytes as f64 / best.0 as f64 / 1e6);
        }
        println!("{line}");
        let _ = GgmlType::F32;
    }
    Ok(())
}

/// Grouped-expert MoE (gate/up + SwiGLU, q8 of h, down, combine) vs a host reference with the same activation
/// quantization; some experts are "not on the card" (table entry 0) and come from CpuMoe instead. Then timings.
fn moe(g: &Gguf, k: &Ggq, s: &Stream) -> Result<()> {
    let (ne, nf, topk) = (2560usize, 640usize, 10usize);
    let mut rng = Lcg(31);
    let mut cpu_moe = cpu::CpuMoe::new(8, ne, nf);
    for layer in [1usize, 0, 17, 48] {
        let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
        let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
        let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
        let nexp = gt.dims[2] as usize;
        let (dg, du, dd) = (DeviceBuffer::from_slice(g.data(gt))?, DeviceBuffer::from_slice(g.data(ut))?,
            DeviceBuffer::from_slice(g.data(dt))?);
        let missing = |e: usize| e % 7 == 3;
        let tab = |b: &DeviceBuffer<u8>, sb: usize, all: bool| -> Vec<u64> {
            (0..nexp).map(|e| if !all && missing(e) { 0 } else { b.ptr().0 + (e * sb) as u64 }).collect()
        };
        let mut xa = Q8Act::alloc(8, ne)?;
        let mut ha = Q8Act::alloc(8 * topk, nf)?;

        let dy = DeviceBuffer::<f32>::alloc(8 * topk * ne)?;
        let dout = DeviceBuffer::<f32>::alloc(8 * ne)?;
        let dcnt = DeviceBuffer::<u32>::alloc(1)?;
        let dpairs = DeviceBuffer::<u32>::alloc(8 * topk)?;
        let dslot = DeviceBuffer::<i32>::alloc(8 * topk)?;
        let dycpu = DeviceBuffer::<f32>::alloc(8 * topk * ne)?;
        // correctness, n_tok = 1..8, experts drawn from a pool of 24 so tokens share some
        let (tg, tu, td) = (DeviceBuffer::from_slice(&tab(&dg, gt.slice_bytes(), false))?,
            DeviceBuffer::from_slice(&tab(&du, ut.slice_bytes(), false))?,
            DeviceBuffer::from_slice(&tab(&dd, dt.slice_bytes(), false))?);
        let mut worst_y = 0f64;
        let mut worst_o = 0f64;
        let mut shared = 0usize;
        let mut nmiss = 0usize;
        for nt in 1..=8usize {
            let x: Vec<f32> = (0..nt * ne).map(|_| rng.f()).collect();
            let mut ids = vec![0i32; nt * topk];
            let mut w = vec![0f32; nt * topk];
            for t in 0..nt {
                let mut chosen: Vec<i32> = vec![];
                while chosen.len() < topk {
                    let e = ((rng.f() + 1.0) * 12.0) as i32 % 24 * (nexp as i32 / 24) + layer as i32 % 3;
                    if !chosen.contains(&e) {
                        chosen.push(e);
                    }
                }
                ids[t * topk..(t + 1) * topk].copy_from_slice(&chosen);
                let ws: Vec<f32> = (0..topk).map(|_| rng.f().abs() + 0.05).collect();
                let sum: f32 = ws.iter().sum();
                for kk in 0..topk {
                    w[t * topk + kk] = ws[kk] / sum;
                }
            }
            let dids = DeviceBuffer::from_slice(&ids)?;
            let dw = DeviceBuffer::from_slice(&w)?;
            let dx = DeviceBuffer::from_slice(&x)?;
            k.quantize_q8_1(s, dx.ptr(), ne, false, nt, ne, &mut xa)?;
            k.moe_gu_q8(s, gt.ty, tg.ptr(), tu.ptr(), gt.row_bytes(), dids.ptr(), nt, topk, &xa, nf, &mut ha, None, None)?;
            k.moe_misses(s, tg.ptr(), dids.ptr(), nt, topk, dcnt.ptr(), dpairs.ptr(), dslot.ptr())?;
            k.moe_down(s, dt.ty, td.ptr(), dt.row_bytes(), dids.ptr(), nt, topk, &ha, ne, dy.ptr(), None)?;
            s.sync()?;
            // the host part: which pairs need the CPU (one D2H), CpuMoe, one H2D of their rows
            let cnt = dcnt.to_vec()?[0] as usize;
            let pairs = dpairs.to_vec()?;
            let expect: Vec<u32> = (0..nt * topk).filter(|&p| missing(ids[p] as usize)).map(|p| p as u32).collect();
            assert_eq!(&pairs[..cnt], &expect[..], "miss list");
            nmiss += cnt;
            let mut jobs: Vec<cpu::MoeJob> = vec![];
            let mut job_slots: Vec<Vec<usize>> = vec![];
            for (sl, &p) in pairs[..cnt].iter().enumerate() {
                let e = ids[p as usize] as usize;
                if let Some(j) = jobs.iter().position(|j| std::ptr::eq(j.gate.as_ptr(), g.slice(gt, e).as_ptr())) {
                    jobs[j].rows.push(p as usize / topk);
                    job_slots[j].push(sl);
                } else {
                    jobs.push(cpu::MoeJob { gate: g.slice(gt, e), up: g.slice(ut, e), down: g.slice(dt, e), ty_gu: gt.ty,
                        ty_down: dt.ty, rows: vec![p as usize / topk] });
                    job_slots.push(vec![sl]);
                }
            }
            let mut ycpu = vec![0f32; cnt.max(1) * ne];
            if cnt > 0 {
                cpu_moe.run(&jobs, &x)?;
                for (j, sls) in job_slots.iter().enumerate() {
                    for (bi, &sl) in sls.iter().enumerate() {
                        ycpu[sl * ne..(sl + 1) * ne].copy_from_slice(&cpu_moe.out(j)[bi * ne..(bi + 1) * ne]);
                    }
                }
                fe_cuda::memcpy_htod(dycpu.ptr(), &ycpu[..cnt * ne])?;
            }
            k.moe_combine_cpu(s, dout.ptr(), dy.ptr(), dw.ptr(), nt, topk, ne, false, dycpu.ptr(), dslot.ptr())?;
            s.sync()?;
            let mut y = dy.to_vec()?;
            for (sl, &p) in pairs[..cnt].iter().enumerate() {
                y[p as usize * ne..(p as usize + 1) * ne].copy_from_slice(&ycpu[sl * ne..(sl + 1) * ne]);
            }
            let out = dout.to_vec()?;
            // host reference per pair
            let mut yref = vec![0f64; nt * topk * ne];
            for p in 0..nt * topk {
                let e = ids[p] as usize;
                if ids.iter().take(p).any(|&q| q as usize == e) {
                    shared += 1;
                }
                let t = p / topk;
                let mut gw = vec![0f32; nf * ne];
                let mut uw = vec![0f32; nf * ne];
                let mut dwf = vec![0f32; ne * nf];
                refq::dequant(gt.ty, g.slice(gt, e), &mut gw);
                refq::dequant(ut.ty, g.slice(ut, e), &mut uw);
                refq::dequant(dt.ty, g.slice(dt, e), &mut dwf);
                let xq = cpu::avx2::Q8Rows::quantize(&x[t * ne..(t + 1) * ne], 1, ne);
                let xqf: Vec<f64> = (0..ne).map(|c| xq.q[c] as f64 * xq.d[c / 32] as f64).collect();
                let hq: Vec<f32> = (0..nf).map(|i| {
                    let rbg = gt.row_bytes();
                    let gv = refq::dot_q8(gt.ty, &g.slice(gt, e)[i * rbg..(i + 1) * rbg], &xq.q, &xq.d, &xq.s);
                    let uv = refq::dot_q8(ut.ty, &g.slice(ut, e)[i * rbg..(i + 1) * rbg], &xq.q, &xq.d, &xq.s);
                    let _ = (&xqf, &gw, &uw);
                    (gv / (1.0 + (-gv).exp()) * uv) as f32
                }).collect();
                let hqq = cpu::avx2::Q8Rows::quantize(&hq, 1, nf);
                let mut den = 0f64;
                let mut mx = 0f64;
                for o in 0..ne {
                    let v: f64 = (0..nf).map(|i| dwf[o * nf + i] as f64 * hqq.q[i] as f64 * hqq.d[i / 32] as f64).sum();
                    yref[p * ne + o] = v;
                    den += v * v;
                    mx = mx.max((y[p * ne + o] as f64 - v).abs());
                }
                worst_y = worst_y.max(mx / (den / ne as f64).sqrt());
            }
            for t in 0..nt {
                let r: Vec<f64> = (0..ne).map(|o| (0..topk).map(|kk| w[t * topk + kk] as f64 * yref[(t * topk + kk) * ne + o]).sum()).collect();
                let rms = (r.iter().map(|v| v * v).sum::<f64>() / ne as f64).sqrt();
                let mx = (0..ne).map(|o| (out[t * ne + o] as f64 - r[o]).abs()).fold(0f64, f64::max);
                worst_o = worst_o.max(mx / rms);
            }
        }
        println!("layer {layer:2} {}/{} ({nexp} experts): n_tok 1..8, {shared} shared pairs, {nmiss} CPU pairs: max|err|/rms per pair {worst_y:.2e}, combined {worst_o:.2e}",
            gt.ty.name(), dt.ty.name());
        // timing: all resident, distinct random experts per token (worst case), then the GB/s of unique bytes
        let (tg, tu, td) = (DeviceBuffer::from_slice(&tab(&dg, gt.slice_bytes(), true))?,
            DeviceBuffer::from_slice(&tab(&du, ut.slice_bytes(), true))?,
            DeviceBuffer::from_slice(&tab(&dd, dt.slice_bytes(), true))?);
        let per_e = gt.slice_bytes() + ut.slice_bytes() + dt.slice_bytes();
        let mut line = format!("   time (all on card, distinct experts; 4 launches: q8 x, gu+q8 h, down, combine):");
        for nt in [1usize, 2, 4, 8] {
            // 24 different expert sets, rotated, so the 32 MB L2 does not serve repeats
            let sets: Vec<Vec<i32>> = (0..24).map(|r| (0..nt * topk).map(|p| ((p * 37 + r * 101 + layer * 11) % nexp) as i32).collect()).collect();
            let ids = sets[0].clone();
            let w = vec![0.1f32; nt * topk];
            let dsets: Vec<DeviceBuffer<i32>> = sets.iter().map(|v| DeviceBuffer::from_slice(v)).collect::<std::result::Result<_, _>>()?;
            let dw = DeviceBuffer::from_slice(&w)?;
            let x: Vec<f32> = (0..nt * ne).map(|_| rng.f()).collect();
            let dx = DeviceBuffer::from_slice(&x)?;
            let mut uniq = ids.clone();
            uniq.sort();
            uniq.dedup();
            let run = |s: &Stream, xa: &mut Q8Act, ha: &mut Q8Act, r: usize| -> Result<()> {
                let dids = &dsets[r % dsets.len()];
                k.quantize_q8_1(s, dx.ptr(), ne, false, nt, ne, xa)?;
                k.moe_gu_q8(s, gt.ty, tg.ptr(), tu.ptr(), gt.row_bytes(), dids.ptr(), nt, topk, xa, nf, ha, None, None)?;
                k.moe_down(s, dt.ty, td.ptr(), dt.row_bytes(), dids.ptr(), nt, topk, ha, ne, dy.ptr(), None)?;
                k.moe_combine(s, dout.ptr(), dy.ptr(), dw.ptr(), nt, topk, ne, false)?;
                Ok(())
            };
            run(s, &mut xa, &mut ha, 0)?;
            let reps = 24;
            let ms = s.time(|s| {
                for r in 0..reps {
                    run(s, &mut xa, &mut ha, r).unwrap();
                }
                Ok(())
            })? / reps as f32;
            // the same 4 launches captured as one CUDA graph per expert set
            let mut graphs = vec![];
            for r in 0..sets.len() {
                s.begin_capture()?;
                run(s, &mut xa, &mut ha, r)?;
                graphs.push(s.end_capture()?);
            }
            graphs[0].launch(s)?;
            let msg = s.time(|s| {
                for r in 0..reps {
                    graphs[r % graphs.len()].launch(s)?;
                }
                Ok(())
            })? / reps as f32;
            // per-kernel split (graph-free, same rotation)
            let mut split = [0f32; 4];
            for (i, sp) in split.iter_mut().enumerate() {
                *sp = s.time(|s| {
                    for r in 0..reps {
                        let dids = &dsets[r % dsets.len()];
                        match i {
                            0 => k.quantize_q8_1(s, dx.ptr(), ne, false, nt, ne, &mut xa).unwrap(),
                            1 => k.moe_gu_q8(s, gt.ty, tg.ptr(), tu.ptr(), gt.row_bytes(), dids.ptr(), nt, topk, &xa, nf, &mut ha, None, None).unwrap(),
                            2 => k.moe_down(s, dt.ty, td.ptr(), dt.row_bytes(), dids.ptr(), nt, topk, &ha, ne, dy.ptr(), None).unwrap(),
                            _ => k.moe_combine(s, dout.ptr(), dy.ptr(), dw.ptr(), nt, topk, ne, false).unwrap(),
                        }
                    }
                    Ok(())
                })? / reps as f32 * 1e3;
            }
            line += &format!(" [{:.0}/{:.0}/{:.0}/{:.0}]", split[0], split[1], split[2], split[3]);
            let mut tot = 0usize;
            for v in &sets {
                let mut u = v.clone();
                u.sort();
                u.dedup();
                tot += u.len();
            }
            let _ = uniq;
            line += &format!("  B{nt} {:.0} us / graph {:.0} us ({:.0} GB/s)", ms * 1e3, msg * 1e3,
                (tot as f64 / sets.len() as f64) * per_e as f64 / msg as f64 / 1e6);
        }
        println!("{line}");
    }
    Ok(())
}

/// Prompt GEMM vs f64 host (dequantized weights x bf16-rounded activations; the GEMM rounds the weights to bf16, so
/// the expected error is ~bf16 precision), dense and segmented, then TFLOP/s and GB/s on real tensors.
fn gemm_check(g: &Gguf, k: &Ggq, s: &Stream) -> Result<()> {
    let mut rng = Lcg(77);
    let mut seen = vec![];
    for t in &g.tensors {
        if !gpu::GEMM_TYPES.contains(&t.ty) || seen.contains(&t.ty) || t.dims.len() < 2 || t.dims[1] % 64 != 0
            || t.ne0() % 32 != 0 {
            continue;
        }
        seen.push(t.ty);
        let kk = t.ne0();
        let n = t.n_rows().min(256);
        let m = 77usize;
        let rb = t.row_bytes();
        let wb = &g.data(t)[..rb * n];
        let mut wf = vec![0f32; n * kk];
        refq::dequant(t.ty, wb, &mut wf);
        let x: Vec<f32> = (0..m * kk).map(|_| refq::bf16_to_f32(refq::f32_to_bf16(rng.f()))).collect();
        let xb: Vec<u16> = x.iter().map(|v| refq::f32_to_bf16(*v)).collect();
        let (dw, dx) = (DeviceBuffer::from_slice(wb)?, DeviceBuffer::from_slice(&xb)?);
        let dy = DeviceBuffer::<f32>::alloc(m * n)?;
        k.gemm(s, t.ty, dy.ptr(), n, dx.ptr(), kk, dw.ptr(), rb, m, n, kk, false)?;
        s.sync()?;
        let y = dy.to_vec()?;
        let mut worst = 0f64;
        for r in 0..m {
            let mut den = 0f64;
            let mut mx = 0f64;
            for c in 0..n {
                let v: f64 = (0..kk).map(|i| wf[c * kk + i] as f64 * x[r * kk + i] as f64).sum();
                den += v * v;
                mx = mx.max((y[r * n + c] as f64 - v).abs());
            }
            worst = worst.max(mx / (den / n as f64).sqrt());
        }
        println!("gemm {:8} {:>32} [m {m} x n {n} x k {kk}]: max|err|/rms {worst:.2e}", t.ty.name(), t.name);
    }
    // segmented: 3 experts of layer 1 gate, token lists with gathers, one skipped segment
    let gt = g.tensor("blk.1.ffn_gate_exps.weight").unwrap();
    let dg = DeviceBuffer::from_slice(g.data(gt))?;
    let ntok = 50usize;
    let x: Vec<f32> = (0..ntok * 2560).map(|_| refq::bf16_to_f32(refq::f32_to_bf16(rng.f()))).collect();
    let xb: Vec<u16> = x.iter().map(|v| refq::f32_to_bf16(*v)).collect();
    let dx = DeviceBuffer::from_slice(&xb)?;
    let experts = [5usize, 100, 7, 255];
    let lists: Vec<Vec<u32>> = vec![(0..30).map(|i| (i * 3 % ntok) as u32).collect(), (0..70).map(|i| (i % ntok) as u32).collect(),
        (0..5).map(|i| (i * 7 % ntok) as u32).collect(), (0..9).map(|i| i as u32).collect()];
    let mut xidx = vec![];
    let mut off = vec![0u32];
    for l in &lists {
        xidx.extend_from_slice(l);
        off.push(xidx.len() as u32);
    }
    let yidx: Vec<u32> = (0..xidx.len() as u32).collect();
    let segw: Vec<u64> = experts.iter().enumerate().map(|(i, &e)| if i == 2 { 0 } else { dg.ptr().0 + (e * gt.slice_bytes()) as u64 }).collect();
    let (dxi, dyi, dsw, doff) = (DeviceBuffer::from_slice(&xidx)?, DeviceBuffer::from_slice(&yidx)?, DeviceBuffer::from_slice(&segw)?,
        DeviceBuffer::from_slice(&off)?);
    let dy = DeviceBuffer::<f32>::zeroed(xidx.len() * 640)?;
    k.gemm_seg(s, gt.ty, dy.ptr(), 640, dx.ptr(), 2560, DevPtr(0), gt.row_bytes(), 70, 640, 2560, dxi.ptr(), dyi.ptr(),
        dsw.ptr(), doff.ptr(), experts.len(), false)?;
    s.sync()?;
    let y = dy.to_vec()?;
    let mut worst = 0f64;
    let mut skipped_ok = true;
    for (si, &e) in experts.iter().enumerate() {
        let mut wf = vec![0f32; 640 * 2560];
        refq::dequant(gt.ty, g.slice(gt, e), &mut wf);
        for i in off[si] as usize..off[si + 1] as usize {
            let xr = &x[xidx[i] as usize * 2560..];
            if si == 2 {
                skipped_ok &= y[i * 640..(i + 1) * 640].iter().all(|v| *v == 0.0);
                continue;
            }
            let mut den = 0f64;
            let mut mx = 0f64;
            for c in 0..640 {
                let v: f64 = (0..2560).map(|q| wf[c * 2560 + q] as f64 * xr[q] as f64).sum();
                den += v * v;
                mx = mx.max((y[i * 640 + c] as f64 - v).abs());
            }
            worst = worst.max(mx / (den / 640.0).sqrt());
        }
    }
    println!("gemm_seg IQ2_S 4 segments (30/70/5 skipped/9 rows, gathered): max|err|/rms {worst:.2e}, skipped segment untouched {skipped_ok}");
    // speed: dense, m = 2048
    let m = 2048usize;
    let xbig: Vec<u16> = (0..m * 6144).map(|_| refq::f32_to_bf16(rng.f())).collect();
    let dx = DeviceBuffer::from_slice(&xbig)?;
    for name in ["blk.0.attn_qkv.weight", "blk.0.ssm_out.weight", "blk.1.attn_qkv.weight", "blk.0.attn_gate.weight",
        "blk.1.ple_key.weight"] {
        let Some(t) = g.tensor(name) else { continue };
        let (kk, n) = (t.ne0(), t.n_rows());
        let dw = DeviceBuffer::from_slice(g.data(t))?;
        let dy = DeviceBuffer::<f32>::alloc(m * n)?;
        k.gemm(s, t.ty, dy.ptr(), n, dx.ptr(), kk, dw.ptr(), t.row_bytes(), m, n, kk, false)?;
        let ms = s.time(|s| { for _ in 0..5 { k.gemm(s, t.ty, dy.ptr(), n, dx.ptr(), kk, dw.ptr(), t.row_bytes(), m, n, kk, false).unwrap(); } Ok(()) })? / 5.0;
        println!("gemm {:8} {name:26} m {m} [{kk} x {n}]: {:.3} ms, {:.1} TFLOP/s", t.ty.name(), ms, 2.0 * (m * n * kk) as f64 / ms as f64 / 1e9);
    }
    // expert gate, 2048 tokens x top10 spread over 256 experts (80 rows each)
    for name in ["blk.1.ffn_gate_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight", "blk.1.ffn_down_exps.weight"] {
        let t = g.tensor(name).unwrap();
        let (kk, n, ne) = (t.ne0(), t.dims[1] as usize, t.dims[2] as usize);
        let dw = DeviceBuffer::from_slice(g.data(t))?;
        let per = 80usize;
        let xidx: Vec<u32> = (0..ne * per).map(|i| ((i * 13) % m) as u32).collect();
        let off: Vec<u32> = (0..=ne).map(|e| (e * per) as u32).collect();
        let yidx: Vec<u32> = (0..(ne * per) as u32).collect();
        let segw: Vec<u64> = (0..ne).map(|e| dw.ptr().0 + (e * t.slice_bytes()) as u64).collect();
        let (dxi, dyi, dsw, doff) = (DeviceBuffer::from_slice(&xidx)?, DeviceBuffer::from_slice(&yidx)?, DeviceBuffer::from_slice(&segw)?,
            DeviceBuffer::from_slice(&off)?);
        let dy = DeviceBuffer::<f32>::alloc(ne * per * n)?;
        let run = |s: &Stream| k.gemm_seg(s, t.ty, dy.ptr(), n, dx.ptr(), kk, DevPtr(0), t.row_bytes(), per, n, kk, dxi.ptr(),
            dyi.ptr(), dsw.ptr(), doff.ptr(), ne, false);
        run(s)?;
        let ms = s.time(|s| { for _ in 0..5 { run(s).unwrap(); } Ok(()) })? / 5.0;
        println!("gemm_seg {:8} {name:26} {ne} experts x {per} rows [{kk} x {n}]: {:.3} ms, {:.1} TFLOP/s, {:.0} GB/s weights",
            t.ty.name(), ms, 2.0 * (ne * per * n * kk) as f64 / ms as f64 / 1e9, t.bytes as f64 / ms as f64 / 1e6);
    }
    Ok(())
}

/// Int8 prompt GEMM (dense and expert-sorted) vs the exact f64 model (refq::dot_q8 on the same q8 activations),
/// the GPU sort's invariants, then timings.
fn mmq_check(g: &Gguf, k: &Ggq, s: &Stream) -> Result<()> {
    let mut rng = Lcg(5150);
    let mut seen = vec![];
    for t in &g.tensors {
        if !gpu::GEMV_TYPES.contains(&t.ty) || seen.contains(&t.ty) || t.ne0() % 128 != 0 || t.n_rows() % 64 != 0 {
            continue;
        }
        seen.push(t.ty);
        let kk = t.ne0();
        let n = t.n_rows().min(256);
        let rb = t.row_bytes();
        let wb = &g.data(t)[..rb * n];
        let m = 77usize;
        let x: Vec<f32> = (0..m * kk).map(|_| rng.f()).collect();
        let dx = DeviceBuffer::from_slice(&x)?;
        let mut xa = Q8Act::alloc(m, kk)?;
        k.quantize_q8_1(s, dx.ptr(), kk, false, m, kk, &mut xa)?;
        let dw = DeviceBuffer::from_slice(wb)?;
        let dy = DeviceBuffer::<f32>::alloc(m * n)?;
        k.mmq(s, t.ty, dy.ptr(), n, &xa, dw.ptr(), rb, n, false)?;
        s.sync()?;
        let y = dy.to_vec()?;
        let mut worst = 0f64;
        for r in 0..m {
            let (q, d, sx) = refq::quantize_q8_1(&x[r * kk..(r + 1) * kk]);
            let mut den = 0f64;
            let mut mx = 0f64;
            for c in 0..n {
                let v = refq::dot_q8(t.ty, &wb[c * rb..(c + 1) * rb], &q, &d, &sx);
                den += v * v;
                mx = mx.max((y[r * n + c] as f64 - v).abs());
            }
            worst = worst.max(mx / (den / n as f64).sqrt());
        }
        println!("mmq {:8} {:>30} [m {m} n {n} k {kk}]: max|err|/rms vs exact model {worst:.2e}", t.ty.name(), t.name);
    }
    // expert-sorted: layer 1 gate (IQ2_S) and down (Q2_0), 300 tokens, top-10 of 256, some experts off the card
    let (ne, nf, topk) = (2560usize, 640usize, 10usize);
    for layer in [1usize, 17, 48] {
        let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
        let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
        let nexp = gt.dims[2] as usize;
        let ntok = 300usize;
        let mut ids = vec![0i32; ntok * topk];
        for t in 0..ntok {
            let mut ch: Vec<i32> = vec![];
            while ch.len() < topk {
                let e = (((rng.f() + 1.0) * 0.5) * nexp as f32) as i32 % nexp as i32;
                if !ch.contains(&e) {
                    ch.push(e);
                }
            }
            ids[t * topk..(t + 1) * topk].copy_from_slice(&ch);
        }
        let np = ntok * topk;
        let dids = DeviceBuffer::from_slice(&ids)?;
        let ws = SortWs::alloc(np, nexp)?;
        k.moe_sort(s, dids.ptr(), ntok, topk, &ws)?;
        s.sync()?;
        let (seg, xi, yi) = (ws.seg_off.to_vec()?, ws.xidx.to_vec()?, ws.yidx.to_vec()?);
        let mut sort_ok = seg[nexp] as usize == np;
        for e in 0..nexp {
            let rows = &yi[seg[e] as usize..seg[e + 1] as usize];
            sort_ok &= rows.windows(2).all(|w| w[0] < w[1]);
            sort_ok &= rows.iter().all(|&p| ids[p as usize] as usize == e && xi[(seg[e] as usize)..].len() > 0);
        }
        sort_ok &= (0..np).all(|r| xi[r] == yi[r] / topk as u32);
        let dg = DeviceBuffer::from_slice(g.data(gt))?;
        let dd = DeviceBuffer::from_slice(g.data(dt))?;
        let missing = |e: usize| e % 9 == 4;
        let tg: Vec<u64> = (0..nexp).map(|e| if missing(e) { 0 } else { dg.ptr().0 + (e * gt.slice_bytes()) as u64 }).collect();
        let td: Vec<u64> = (0..nexp).map(|e| if missing(e) { 0 } else { dd.ptr().0 + (e * dt.slice_bytes()) as u64 }).collect();
        let (dtg, dtd) = (DeviceBuffer::from_slice(&tg)?, DeviceBuffer::from_slice(&td)?);
        let x: Vec<f32> = (0..ntok * ne).map(|_| rng.f()).collect();
        let dx = DeviceBuffer::from_slice(&x)?;
        let mut xa = Q8Act::alloc(ntok, ne)?;
        k.quantize_q8_1(s, dx.ptr(), ne, false, ntok, ne, &mut xa)?;
        let dy = DeviceBuffer::<f32>::zeroed(np * nf)?;
        k.mmq_moe(s, gt.ty, dy.ptr(), nf, &xa, dtg.ptr(), gt.row_bytes(), nf, &ws, np, false, false)?;
        // down on per-pair activations (random h rows, one per pair)
        let h: Vec<f32> = (0..np * nf).map(|_| rng.f()).collect();
        let dh = DeviceBuffer::from_slice(&h)?;
        let mut ha = Q8Act::alloc(np, nf)?;
        k.quantize_q8_1(s, dh.ptr(), nf, false, np, nf, &mut ha)?;
        let dz = DeviceBuffer::<f32>::zeroed(np * ne)?;
        k.mmq_moe(s, dt.ty, dz.ptr(), ne, &ha, dtd.ptr(), dt.row_bytes(), ne, &ws, np, true, false)?;
        s.sync()?;
        let (y, z) = (dy.to_vec()?, dz.to_vec()?);
        let (mut wg, mut wd) = (0f64, 0f64);
        let mut untouched = true;
        for p in (0..np).step_by(37) {
            let e = ids[p] as usize;
            if missing(e) {
                untouched &= y[p * nf..(p + 1) * nf].iter().all(|v| *v == 0.0) && z[p * ne..(p + 1) * ne].iter().all(|v| *v == 0.0);
                continue;
            }
            let t = p / topk;
            let (q, d, sx) = refq::quantize_q8_1(&x[t * ne..(t + 1) * ne]);
            let gsl = g.slice(gt, e);
            let rb = gt.row_bytes();
            let (mut den, mut mx) = (0f64, 0f64);
            for c in 0..nf {
                let v = refq::dot_q8(gt.ty, &gsl[c * rb..(c + 1) * rb], &q, &d, &sx);
                den += v * v;
                mx = mx.max((y[p * nf + c] as f64 - v).abs());
            }
            wg = wg.max(mx / (den / nf as f64).sqrt());
            let (q, d, sx) = refq::quantize_q8_1(&h[p * nf..(p + 1) * nf]);
            let dsl = g.slice(dt, e);
            let rb = dt.row_bytes();
            let (mut den, mut mx) = (0f64, 0f64);
            for c in 0..ne {
                let v = refq::dot_q8(dt.ty, &dsl[c * rb..(c + 1) * rb], &q, &d, &sx);
                den += v * v;
                mx = mx.max((z[p * ne + c] as f64 - v).abs());
            }
            wd = wd.max(mx / (den / ne as f64).sqrt());
        }
        // fused gate/up/SwiGLU/q8 vs the separate gate and up GEMMs + host SwiGLU
        let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
        let du = DeviceBuffer::from_slice(g.data(ut))?;
        let tu: Vec<u64> = (0..nexp).map(|e| if missing(e) { 0 } else { du.ptr().0 + (e * ut.slice_bytes()) as u64 }).collect();
        let dtu = DeviceBuffer::from_slice(&tu)?;
        let dyu = DeviceBuffer::<f32>::zeroed(np * nf)?;
        k.mmq_moe(s, ut.ty, dyu.ptr(), nf, &xa, dtu.ptr(), ut.row_bytes(), nf, &ws, np, false, false)?;
        let mut hq = Q8Act::alloc(np, nf)?;
        k.mmq_moe_gu(s, gt.ty, &xa, dtg.ptr(), dtu.ptr(), gt.row_bytes(), nf, &ws, np, &mut hq)?;
        s.sync()?;
        let (yu, hqq, hqd) = (dyu.to_vec()?, hq.q.to_vec()?, hq.d.to_vec()?);
        let mut werr = 0f64;
        for p in 0..np {
            if missing(ids[p] as usize) {
                continue;
            }
            for blk in 0..nf / 32 {
                let hv: Vec<f32> = (0..32).map(|i| { let c = blk * 32 + i; let gv = y[p * nf + c]; gv / (1.0 + (-gv).exp()) * yu[p * nf + c] }).collect();
                let amax = hv.iter().fold(0f32, |a, v| a.max(v.abs()));
                for i in 0..32 {
                    let got = hqq[p * nf + blk * 32 + i] as f32 * hqd[p * nf / 32 + blk];
                    werr = werr.max(((got - hv[i]).abs() / amax.max(1e-30)) as f64);
                }
            }
        }
        println!("mmq_moe_gu layer {layer}: fused h (q8) vs separate GEMMs + SwiGLU: max |err| / block amax {werr:.2e} (q8 step = {:.2e})", 1.0 / 127.0);
        // down fused with the weighted combine (atomics into token rows) vs host combine of the per-pair outputs
        let wts: Vec<f32> = (0..np).map(|_| rng.f().abs()).collect();
        let dwts = DeviceBuffer::from_slice(&wts)?;
        let dout = DeviceBuffer::<f32>::zeroed(ntok * ne)?;
        let dacc = DeviceBuffer::<i64>::zeroed(ntok * ne)?;
        let dst = DeviceBuffer::<u32>::zeroed(1)?;
        k.mmq_moe_down_acc(s, dt.ty, dacc.ptr(), ne, &ha, dtd.ptr(), dt.row_bytes(), ne, &ws, np, topk, dwts.ptr(), dst.ptr())?;
        k.fx_finish(s, dout.ptr(), dacc.ptr(), ntok * ne, false)?;
        s.sync()?;
        let outv = dout.to_vec()?;
        // run again: must be bit-identical
        k.mmq_moe_down_acc(s, dt.ty, dacc.ptr(), ne, &ha, dtd.ptr(), dt.row_bytes(), ne, &ws, np, topk, dwts.ptr(), dst.ptr())?;
        k.fx_finish(s, dout.ptr(), dacc.ptr(), ntok * ne, false)?;
        s.sync()?;
        let same = dout.to_vec()?.iter().zip(&outv).all(|(a, b)| a.to_bits() == b.to_bits());
        println!("mmq_moe_down_acc layer {layer}: rerun bit-identical {same}, status {}", dst.to_vec()?[0]);
        let mut eacc = 0f64;
        for t in 0..ntok {
            let r: Vec<f64> = (0..ne).map(|i| (0..topk).map(|kk| { let p = t * topk + kk; wts[p] as f64 * z[p * ne + i] as f64 }).sum()).collect();
            let rms = (r.iter().map(|v| v * v).sum::<f64>() / ne as f64).sqrt().max(1e-30);
            for i in 0..ne {
                eacc = eacc.max((outv[t * ne + i] as f64 - r[i]).abs() / rms);
            }
        }
        println!("mmq_moe_down_acc layer {layer}: vs per-pair down + host combine {eacc:.1e}");
        println!("mmq_moe layer {layer} {}/{}: sort ok {sort_ok}; gate max|err|/rms {wg:.2e}, down {wd:.2e}; skipped experts untouched {untouched}",
            gt.ty.name(), dt.ty.name());
    }
    // timings: dense m = 2048, and a prompt group of 16384 tokens through one layer's experts
    let m = 2048usize;
    let x: Vec<f32> = (0..m * 6144).map(|_| rng.f()).collect();
    let dx = DeviceBuffer::from_slice(&x)?;
    for name in ["blk.0.attn_qkv.weight", "blk.0.ssm_out.weight", "blk.1.attn_qkv.weight", "blk.0.attn_gate.weight",
        "output.weight"] {
        let Some(t) = g.tensor(name) else { continue };
        let (kk, n) = (t.ne0(), t.n_rows());
        let mut xa = Q8Act::alloc(m, kk)?;
        k.quantize_q8_1(s, dx.ptr(), kk, false, m, kk, &mut xa)?;
        let dw = DeviceBuffer::from_slice(g.data(t))?;
        let dy = DeviceBuffer::<f32>::alloc(m * n)?;
        k.mmq(s, t.ty, dy.ptr(), n, &xa, dw.ptr(), t.row_bytes(), n, false)?;
        let ms = s.time(|s| { for _ in 0..3 { k.mmq(s, t.ty, dy.ptr(), n, &xa, dw.ptr(), t.row_bytes(), n, false).unwrap(); } Ok(()) })? / 3.0;
        println!("mmq {:8} {name:24} m {m} [{kk} x {n}]: {:.3} ms, {:.1} TOP/s", t.ty.name(), ms, 2.0 * (m * n * kk) as f64 / ms as f64 / 1e9);
    }
    let ntok = 16384usize;
    for layer in [1usize, 0, 17, 48] {
        let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
        let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
        let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
        let nexp = gt.dims[2] as usize;
        let ids: Vec<i32> = (0..ntok * topk).map(|p| ((p / topk * 7919 + (p % topk) * (nexp / topk) + p / topk % 13) % nexp) as i32).collect();
        let np = ntok * topk;
        let dids = DeviceBuffer::from_slice(&ids)?;
        let ws = SortWs::alloc(np, nexp)?;
        let (dg, du, dd) = (DeviceBuffer::from_slice(g.data(gt))?, DeviceBuffer::from_slice(g.data(ut))?, DeviceBuffer::from_slice(g.data(dt))?);
        let tab = |b: &DeviceBuffer<u8>, sb: usize| -> Vec<u64> { (0..nexp).map(|e| b.ptr().0 + (e * sb) as u64).collect() };
        let (tg, tu, td) = (DeviceBuffer::from_slice(&tab(&dg, gt.slice_bytes()))?, DeviceBuffer::from_slice(&tab(&du, ut.slice_bytes()))?,
            DeviceBuffer::from_slice(&tab(&dd, dt.slice_bytes()))?);
        let xs: Vec<f32> = (0..ntok * ne).map(|_| rng.f()).collect();
        let dxs = DeviceBuffer::from_slice(&xs)?;
        let mut xa = Q8Act::alloc(ntok, ne)?;
        k.quantize_q8_1(s, dxs.ptr(), ne, false, ntok, ne, &mut xa)?;
        let mut ha = Q8Act::alloc(np, nf)?;
        let hs: Vec<f32> = (0..np * nf).map(|_| rng.f()).collect();
        let dhs = DeviceBuffer::from_slice(&hs)?;
        k.quantize_q8_1(s, dhs.ptr(), nf, false, np, nf, &mut ha)?;
        let dyg = DeviceBuffer::<f32>::alloc(np * nf)?;
        let dyu = DeviceBuffer::<f32>::alloc(np * nf)?;
        let dyd = DeviceBuffer::<f32>::alloc(np * ne)?;
        let tsort = s.time(|s| { k.moe_sort(s, dids.ptr(), ntok, topk, &ws).unwrap(); Ok(()) })?;
        let mut hq2 = Q8Act::alloc(np, nf)?;
        let tfused = s.time(|s| { k.mmq_moe_gu(s, gt.ty, &xa, tg.ptr(), tu.ptr(), gt.row_bytes(), nf, &ws, np, &mut hq2).unwrap(); Ok(()) })?;
        let tg_ms = s.time(|s| { k.mmq_moe(s, gt.ty, dyg.ptr(), nf, &xa, tg.ptr(), gt.row_bytes(), nf, &ws, np, false, false).unwrap(); Ok(()) })?;
        let tu_ms = s.time(|s| { k.mmq_moe(s, ut.ty, dyu.ptr(), nf, &xa, tu.ptr(), ut.row_bytes(), nf, &ws, np, false, false).unwrap(); Ok(()) })?;
        let td_ms = s.time(|s| { k.mmq_moe(s, dt.ty, dyd.ptr(), ne, &ha, td.ptr(), dt.row_bytes(), ne, &ws, np, true, false).unwrap(); Ok(()) })?;
        let wts = DeviceBuffer::from_slice(&vec![0.1f32; np])?;
        let dacc = DeviceBuffer::<i64>::zeroed(ntok * ne)?;
        let dout = DeviceBuffer::<f32>::zeroed(ntok * ne)?;
        let dst = DeviceBuffer::<u32>::zeroed(1)?;
        let tacc = s.time(|s| {
            k.mmq_moe_down_acc(s, dt.ty, dacc.ptr(), ne, &ha, td.ptr(), dt.row_bytes(), ne, &ws, np, topk, wts.ptr(), dst.ptr()).unwrap();
            k.fx_finish(s, dout.ptr(), dacc.ptr(), ntok * ne, false).unwrap();
            Ok(())
        })?;
        // option 2: plain down into [pairs][2560] + the fixed-order combine (moe_combine) per token
        let dcomb = DeviceBuffer::<f32>::alloc(ntok * ne)?;
        let tcomb = s.time(|s| { k.moe_combine(s, dcomb.ptr(), dyd.ptr(), wts.ptr(), ntok, topk, ne, false).unwrap(); Ok(()) })?;
        println!("   option 2: plain down {td_ms:.2} ms + ordered combine {tcomb:.2} ms = {:.2} ms (pair buffer {:.2} GB)", td_ms + tcomb,
            (np * ne * 4) as f64 / 1e9);
        println!("   down fused with combine (i64 fixed-point atomics into {ntok} token rows + finish): {tacc:.2} ms (vs {td_ms:.2} ms writing [pairs][2560])");
        // column-blocked ordered down + combine: bit-identical to option 2 for any block width
        let want = dcomb.to_vec()?;
        let tabw = DeviceBuffer::<u64>::alloc(nexp)?;
        for ncol in [640usize, 1280, 320] {
            let ybuf = DeviceBuffer::<f32>::alloc(np * ncol)?;
            let dord = DeviceBuffer::<f32>::alloc(ntok * ne)?;
            k.moe_down_ordered(s, dt.ty, dord.ptr(), &ha, td.ptr(), dt.row_bytes(), ne, wts.ptr(), ntok, topk, &ws, ncol, ybuf.ptr(), tabw.ptr(), false)?;
            let t = s.time(|s| { k.moe_down_ordered(s, dt.ty, dord.ptr(), &ha, td.ptr(), dt.row_bytes(), ne, wts.ptr(), ntok, topk, &ws, ncol, ybuf.ptr(), tabw.ptr(), false).unwrap(); Ok(()) })?;
            s.sync()?;
            let got = dord.to_vec()?;
            let nd = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            println!("   ordered down + combine in {ncol}-column blocks: {t:.2} ms, pair buffer {:.0} MB, {nd} outputs differ from option 2",
                (np * ncol * 4) as f64 / 1e6);
        }
        let ops = 2.0 * (np * nf * ne) as f64;
        println!("prompt group {ntok} tok, layer {layer} {}/{}: sort {:.2} ms, gate {:.2} ms ({:.0} TOP/s), up {:.2}, down {:.2} ms ({:.0} TOP/s); experts total {:.1} ms/layer; FUSED gate+up+SwiGLU+q8 {:.2} ms ({:.0} TOP/s) -> total {:.1} ms/layer",
            gt.ty.name(), dt.ty.name(), tsort, tg_ms, ops / tg_ms as f64 / 1e9, tu_ms, td_ms, ops / td_ms as f64 / 1e9, tsort + tg_ms + tu_ms + td_ms, tfused, 2.0 * ops / tfused as f64 / 1e9, tsort + tfused + td_ms);
    }
    Ok(())
}

/// BF16 GEMVs: exactness vs an f64 host dot (expect f32-accumulation rounding only) and GB/s on the hc / indexer
/// tensors, B = 1..8; also the per-token total for 48 layers x (attn + ffn) x (down + inject + up).
fn bf16_check(g: &Gguf, k: &Ggq, s: &Stream) -> Result<()> {
    let mut rng = Lcg(42);
    let names = ["blk.0.hc_attn_down.weight", "blk.0.hc_attn_up.weight", "blk.0.hc_attn_inject.weight",
        "blk.3.indexer.q_proj.weight", "blk.3.indexer.k_proj.weight", "output_hc_down.weight", "blk.1.ple_key.weight",
        "blk.0.ssm_alpha.weight"];
    for name in names {
        let Some(t) = g.tensor(name) else { continue };
        let (kk, n) = (t.ne0(), t.n_rows());
        let wb = g.data(t);
        let w16: Vec<u16> = wb.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let dw = DeviceBuffer::from_slice(wb)?;
        let x: Vec<f32> = (0..8 * kk).map(|_| rng.f()).collect();
        let dx = DeviceBuffer::from_slice(&x)?;
        let dy = DeviceBuffer::<f32>::alloc(8 * n)?;
        let mut worst = 0f64;
        for b in [1usize, 3, 8] {
            k.gemv_bf16(s, dw.ptr(), n, kk, dx.ptr(), kk, b, dy.ptr(), false)?;
            s.sync()?;
            let y = dy.to_vec()?;
            for r in 0..b {
                for i in (0..n).step_by((n / 64).max(1)) {
                    let v: f64 = (0..kk).map(|c| refq::bf16_to_f32(w16[i * kk + c]) as f64 * x[r * kk + c] as f64).sum();
                    let sc: f64 = (0..kk).map(|c| (refq::bf16_to_f32(w16[i * kk + c]) as f64 * x[r * kk + c] as f64).abs()).sum();
                    worst = worst.max((y[r * n + i] as f64 - v).abs() / sc.max(1e-30));
                }
            }
        }
        let mut line = format!("bf16 {name:28} [{kk:5} -> {n:5}] {:5.2} MB: max|err|/sum|terms| {worst:.1e} |", t.bytes as f64 / 1e6);
        for b in [1usize, 2, 4, 5, 8] {
            k.gemv_bf16(s, dw.ptr(), n, kk, dx.ptr(), kk, b, dy.ptr(), false)?;
            // rotate through 8 copies so L2 (32 MB) does not serve repeats of small tensors
            let ms = s.time(|s| { for _ in 0..20 { k.gemv_bf16(s, dw.ptr(), n, kk, dx.ptr(), kk, b, dy.ptr(), false).unwrap(); } Ok(()) })? / 20.0;
            line += &format!(" B{b} {:4.0} GB/s", t.bytes as f64 / ms as f64 / 1e6);
        }
        println!("{line}");
    }
    // cold-L2 per-token estimate: 48 layers x 2 sub-blocks x (down + inject + up), each from a different buffer
    let (dt, ut, it) = (g.tensor("blk.0.hc_attn_down.weight").unwrap(), g.tensor("blk.0.hc_attn_up.weight").unwrap(),
        g.tensor("blk.0.hc_attn_inject.weight").unwrap());
    let nl = 96usize;
    let mut bufs = vec![];
    for _ in 0..nl {
        bufs.push((DeviceBuffer::from_slice(g.data(dt))?, DeviceBuffer::from_slice(g.data(it))?, DeviceBuffer::from_slice(g.data(ut))?));
    }
    let x: Vec<f32> = (0..8 * 10240).map(|_| rng.f()).collect();
    let dx = DeviceBuffer::from_slice(&x)?;
    let dh = DeviceBuffer::<f32>::alloc(8 * 10240)?;
    let dy = DeviceBuffer::<f32>::alloc(8 * 10240)?;
    // the fused form: inject rows concatenated after down (n = 324), as fn-core will store it
    let mut cat = g.data(dt).to_vec();
    cat.extend_from_slice(g.data(it));
    let cats: Vec<DeviceBuffer<u8>> = (0..nl).map(|_| DeviceBuffer::from_slice(&cat)).collect::<std::result::Result<_, _>>()?;
    for b in [1usize, 2, 4, 5, 8] {
        let ms = s.time(|s| {
            for (l, (_, _, u)) in bufs.iter().enumerate() {
                k.gemv_bf16(s, cats[l].ptr(), 324, 10240, dx.ptr(), 10240, b, dh.ptr(), false).unwrap();
                k.gemv_bf16(s, u.ptr(), 10240, 320, dh.ptr(), 324, b, dy.ptr(), false).unwrap();
            }
            Ok(())
        })?;
        let bytes = nl * (dt.bytes + it.bytes + ut.bytes) as usize;
        println!("hc per token, down|inject concat (n=324): B{b} {:.2} ms, {:.0} GB/s", ms, bytes as f64 / ms as f64 / 1e6);
    }
    for b in [1usize, 4, 5] {
        let ms = s.time(|s| {
            for (d, i, u) in &bufs {
                k.gemv_bf16(s, d.ptr(), 320, 10240, dx.ptr(), 10240, b, dh.ptr(), false).unwrap();
                k.gemv_bf16(s, i.ptr(), 4, 10240, dx.ptr(), 10240, b, dy.ptr(), false).unwrap();
                k.gemv_bf16(s, u.ptr(), 10240, 320, dh.ptr(), 320, b, dy.ptr(), false).unwrap();
            }
            Ok(())
        })?;
        let bytes = nl * (dt.bytes + it.bytes + ut.bytes) as usize;
        println!("hc per token (96 sub-blocks, cold): B{b} {:.2} ms, {:.0} GB/s", ms, bytes as f64 / ms as f64 / 1e6);
    }
    Ok(())
}

/// Raw outputs of hc_down (lo, xn) + hc_up (mixed) for B = 1..=24 on fixed random residual rows, written to the file
/// `ggq-check <gguf> hcdump <out>`: compare two builds' files byte for byte (cmp) to prove identical bits.
fn hc_dump(g: &Gguf, k: &Ggq, s: &Stream, args: &[String]) -> Result<()> {
    let out = args.first().cloned().unwrap_or("/tmp/hcdump.bin".into());
    let mut rng = Lcg(77);
    let (dt, it, ut, nt) = (g.tensor("blk.5.hc_ffn_down.weight").unwrap(), g.tensor("blk.5.hc_ffn_inject.weight").unwrap(),
        g.tensor("blk.5.hc_ffn_up.weight").unwrap(), g.tensor("blk.5.hc_ffn_norm.weight").unwrap());
    let mut cat = g.data(dt).to_vec();
    cat.extend_from_slice(g.data(it));
    let nw: Vec<f32> = g.data(nt).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let (dcat, du, dnw) = (DeviceBuffer::from_slice(&cat)?, DeviceBuffer::from_slice(g.data(ut))?, DeviceBuffer::from_slice(&nw)?);
    let hd = 10240usize;
    let mut bytes: Vec<u8> = vec![];
    for b in 1..=24usize {
        let res: Vec<f32> = (0..b * hd).map(|_| rng.f() * 3.0).collect();
        let dres = DeviceBuffer::from_slice(&res)?;
        let dlo = DeviceBuffer::<f32>::alloc(b * 324)?;
        let dxn = DeviceBuffer::<f32>::alloc(b * hd)?;
        let dmix = DeviceBuffer::<f32>::alloc(b * 2560)?;
        k.hc_down_pf(s, dlo.ptr(), 324, dcat.ptr(), 324, dres.ptr(), hd, dnw.ptr(), 1e-6, 0.25, 320, dxn.ptr(), b, du.ptr())?;
        k.hc_up(s, dmix.ptr(), 2560, du.ptr(), dlo.ptr(), 324, dxn.ptr(), hd, 0.25, b)?;
        s.sync()?;
        for v in dlo.to_vec()?.iter().chain(dxn.to_vec()?.iter()).chain(dmix.to_vec()?.iter()) {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    std::fs::write(&out, &bytes)?;
    println!("hcdump: {} bytes to {out}", bytes.len());
    Ok(())
}

/// hc chain timing split: 96 x hc_down (with the up prefetch) alone, hc_up alone, the pair; B from FE_HCB_B (default
/// 1,2,3,4,5,8,12,16,24); cold weights (96 distinct copies).
fn hc_bench(g: &Gguf, k: &Ggq, s: &Stream) -> Result<()> {
    let mut rng = Lcg(9);
    let (dt, it, ut, nt) = (g.tensor("blk.0.hc_attn_down.weight").unwrap(), g.tensor("blk.0.hc_attn_inject.weight").unwrap(),
        g.tensor("blk.0.hc_attn_up.weight").unwrap(), g.tensor("blk.0.hc_attn_norm.weight").unwrap());
    let mut cat = g.data(dt).to_vec();
    cat.extend_from_slice(g.data(it));
    let nw: Vec<f32> = g.data(nt).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let dnw = DeviceBuffer::from_slice(&nw)?;
    let hd = 10240usize;
    let nl = 96usize;
    let mut bufs = vec![];
    for _ in 0..nl {
        bufs.push((DeviceBuffer::from_slice(&cat)?, DeviceBuffer::from_slice(g.data(ut))?));
    }
    let res: Vec<f32> = (0..24 * hd).map(|_| rng.f()).collect();
    let dres = DeviceBuffer::from_slice(&res)?;
    let dlo = DeviceBuffer::<f32>::alloc(24 * 324)?;
    let dxn = DeviceBuffer::<f32>::alloc(24 * hd)?;
    let dmix = DeviceBuffer::<f32>::alloc(24 * 2560)?;
    let bs: Vec<usize> = std::env::var("FE_HCB_B").unwrap_or("1,2,3,4,5,8,12,16,24".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let pf = std::env::var("FE_HCB_PF").map_or(true, |v| v != "0");
    for b in bs {
        let mut t = [0f32; 3];
        for (i, m) in [1, 2, 3].iter().enumerate() {
            let mut best = f32::MAX;
            for _ in 0..3 {
                let ms = s.time(|s| {
                    for (c, u) in &bufs {
                        if m & 1 != 0 {
                            let up = if pf && *m == 3 { u.ptr() } else { DevPtr::NULL };
                            k.hc_down_pf(s, dlo.ptr(), 324, c.ptr(), 324, dres.ptr(), hd, dnw.ptr(), 1e-6, 0.25, 320, dxn.ptr(), b, up).unwrap();
                        }
                        if m & 2 != 0 {
                            k.hc_up(s, dmix.ptr(), 2560, u.ptr(), dlo.ptr(), 324, dxn.ptr(), hd, 0.25, b).unwrap();
                        }
                    }
                    Ok(())
                })?;
                best = best.min(ms);
            }
            t[i] = best * 1000.0 / nl as f32;
        }
        let gb = |n: usize, us: f32| (n * 10240 * 2) as f32 / us / 1e3;
        println!("B{b:2}: down {:.1} us ({:.0} GB/s) | up {:.1} us ({:.0}) | pair {:.1} us ({:.0} GB/s)", t[0], gb(324, t[0]), t[1],
            gb(320, t[1]), t[2], gb(644, t[2]));
    }
    Ok(())
}

/// Fused hc mix (hc_down + hc_up) vs an f64 host model of fn-core's formulas, then the per-token chain time.
fn hc_check(g: &Gguf, k: &Ggq, s: &Stream) -> Result<()> {
    let mut rng = Lcg(9);
    let (dt, it, ut, nt) = (g.tensor("blk.0.hc_attn_down.weight").unwrap(), g.tensor("blk.0.hc_attn_inject.weight").unwrap(),
        g.tensor("blk.0.hc_attn_up.weight").unwrap(), g.tensor("blk.0.hc_attn_norm.weight").unwrap());
    let mut cat = g.data(dt).to_vec();
    cat.extend_from_slice(g.data(it));
    let w16: Vec<f32> = cat.chunks_exact(2).map(|c| refq::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
    let u16v: Vec<f32> = g.data(ut).chunks_exact(2).map(|c| refq::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
    let nw: Vec<f32> = g.data(nt).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let (dcat, du, dnw) = (DeviceBuffer::from_slice(&cat)?, DeviceBuffer::from_slice(g.data(ut))?, DeviceBuffer::from_slice(&nw)?);
    let hd = 10240usize;
    let (eps, scale, mscale) = (1e-6f32, 0.25f32, 0.25f32);
    for b in [1usize, 4, 5] {
        let res: Vec<f32> = (0..b * hd).map(|_| rng.f() * 3.0).collect();
        let dres = DeviceBuffer::from_slice(&res)?;
        let dlo = DeviceBuffer::<f32>::alloc(b * 324)?;
        let dxn = DeviceBuffer::<f32>::alloc(b * hd)?;
        let dmix = DeviceBuffer::<f32>::alloc(b * 2560)?;
        k.hc_down(s, dlo.ptr(), 324, dcat.ptr(), 324, dres.ptr(), hd, dnw.ptr(), eps, scale, 320, dxn.ptr(), b)?;
        k.hc_up(s, dmix.ptr(), 2560, du.ptr(), dlo.ptr(), 324, dxn.ptr(), hd, mscale, b)?;
        s.sync()?;
        let (lo, mix) = (dlo.to_vec()?, dmix.to_vec()?);
        let (mut elo, mut emix) = (0f64, 0f64);
        for r in 0..b {
            let xr = &res[r * hd..(r + 1) * hd];
            let mut xn = vec![0f64; hd];
            for st in 0..4 {
                let ms: f64 = xr[st * 2560..(st + 1) * 2560].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 2560.0;
                let rs = 1.0 / (ms + eps as f64).sqrt();
                for i in 0..2560 {
                    xn[st * 2560 + i] = xr[st * 2560 + i] as f64 * rs * nw[st * 2560 + i] as f64;
                }
            }
            let mut lor = vec![0f64; 324];
            for o in 0..324 {
                let v: f64 = (0..hd).map(|c| w16[o * hd + c] as f64 * xn[c]).sum::<f64>() * scale as f64;
                lor[o] = if o < 320 { v / (1.0 + (-v).exp()) } else { v };
                elo = elo.max((lo[r * 324 + o] as f64 - lor[o]).abs() / (1.0 + lor[o].abs()));
            }
            for j in 0..2560 {
                let mut m = 0f64;
                for st in 0..4 {
                    let row = st * 2560 + j;
                    let gte: f64 = (0..320).map(|c| u16v[row * 320 + c] as f64 * lo[r * 324 + c] as f64).sum();
                    m += xn[row] / (1.0 + (-gte).exp());
                }
                m *= mscale as f64;
                emix = emix.max((mix[r * 2560 + j] as f64 - m).abs() / (1.0 + m.abs()));
            }
        }
        println!("hc fused B{b}: lo max err {elo:.1e}, mixed max err {emix:.1e} (relative to 1 + |value|)");
    }
    // per-token chain: 96 x (hc_down n=324 + hc_up), cold weights
    let nl = 96usize;
    let mut bufs = vec![];
    for _ in 0..nl {
        bufs.push((DeviceBuffer::from_slice(&cat)?, DeviceBuffer::from_slice(g.data(ut))?));
    }
    let res: Vec<f32> = (0..24 * hd).map(|_| rng.f()).collect();
    let dres = DeviceBuffer::from_slice(&res)?;
    let dlo = DeviceBuffer::<f32>::alloc(24 * 324)?;
    let dxn = DeviceBuffer::<f32>::alloc(24 * hd)?;
    let dmix = DeviceBuffer::<f32>::alloc(24 * 2560)?;
    for b in [1usize, 2, 4, 5, 8] {
        let ms = s.time(|s| {
            for (c, u) in &bufs {
                k.hc_down(s, dlo.ptr(), 324, c.ptr(), 324, dres.ptr(), hd, dnw.ptr(), eps, scale, 320, dxn.ptr(), b).unwrap();
                k.hc_up(s, dmix.ptr(), 2560, u.ptr(), dlo.ptr(), 324, dxn.ptr(), hd, mscale, b).unwrap();
            }
            Ok(())
        })?;
        println!("hc fused per token (96 x down|inject + up incl. norm/SiLU/gate/mean): B{b} {ms:.2} ms");
    }
    // the same chain with hc_down prefetching hc_up's weights (hc_down_pf), the engine's call
    for b in [1usize, 2, 4, 5, 8, 16, 24] {
        let ms = s.time(|s| {
            for (c, u) in &bufs {
                k.hc_down_pf(s, dlo.ptr(), 324, c.ptr(), 324, dres.ptr(), hd, dnw.ptr(), eps, scale, 320, dxn.ptr(), b, u.ptr()).unwrap();
                k.hc_up(s, dmix.ptr(), 2560, u.ptr(), dlo.ptr(), 324, dxn.ptr(), hd, mscale, b).unwrap();
            }
            Ok(())
        })?;
        let gbs = (nl * (324 + 320) * 10240 * 2) as f64 / ms as f64 / 1e6;
        println!("hc fused + up prefetch per token: B{b} {ms:.3} ms ({:.1} us/pair, {gbs:.0} GB/s)", ms * 1000.0 / nl as f32);
    }
    Ok(())
}

fn hash2(a: u64, b: u64) -> f32 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn q8_rows(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; x.len()];
    let mut d = vec![0f32; x.len() / 32];
    for (b, c) in x.chunks_exact(32).enumerate() {
        let a = c.iter().fold(0f32, |m, v| m.max(v.abs()));
        let dd = a / 127.0;
        d[b] = dd;
        for i in 0..32 {
            let v = if a == 0.0 { 0.0 } else { c[i] / dd };
            q[b * 32 + i] = v.round() as i32 as i8;
        }
    }
    (q, d)
}

/// fa <ctx> <rows> [sigma] [ref rows]: tensor-core prompt attention vs the SIMT kernel and an f64 host reference.
/// Rows are the last `rows` cells of a `ctx`-cell context; QSA selections from a synthetic score (fixed per-block
/// base + recency + per-row noise of size sigma: 0 = every row picks the same blocks).
fn fa_check(k: &Ggq, s: &Stream, a: &[String]) -> Result<()> {
    let ctx: usize = a.first().and_then(|v| v.parse().ok()).unwrap_or(16384);
    let rows: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(8192).min(ctx);
    let sigma: f32 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(0.3);
    let nref: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(24);
    let (nq, nkv, dh, kp, top) = (24usize, 2usize, 256usize, 4usize, 512usize);
    let pos0 = ctx - rows;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut rng = Lcg(77);
    // K / V for every cell, quantized like fx_attn_prep
    let kf: Vec<f32> = (0..ctx * nkv * dh).map(|_| rng.f()).collect();
    let vf: Vec<f32> = (0..ctx * nkv * dh).map(|_| rng.f() * 2.0).collect();
    let (kqv, ksv) = q8_rows(&kf);
    let (vqv, vsv) = q8_rows(&vf);
    drop((kf, vf));
    let qv: Vec<f32> = (0..rows * nq * dh).map(|_| rng.f() * 3.0).collect();
    let gv: Vec<f32> = (0..rows * nq * dh).map(|_| rng.f() * 2.0).collect();
    // selections
    let ldsel = top * kp + kp - 1;
    let nbmax = ctx / kp + 1;
    let base: Vec<f32> = (0..nbmax).map(|b| hash2(1, b as u64) + hash2(2, b as u64) + hash2(3, b as u64)).collect();
    let mut sel = vec![0u32; rows * ldsel];
    let mut seln = vec![0u32; rows];
    {
        let nth = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let chunk = rows.div_ceil(nth);
        std::thread::scope(|sc| {
            for (ci, (sl, sn)) in sel.chunks_mut(chunk * ldsel).zip(seln.chunks_mut(chunk)).enumerate() {
                let base = &base;
                sc.spawn(move || {
                    let mut idx: Vec<u32> = vec![];
                    for (j, n) in sn.iter_mut().enumerate() {
                        let r = ci * chunk + j;
                        let p = pos0 + r;
                        let nv = (p + 1) / kp;
                        if nv <= top {
                            *n = 0;
                            continue;
                        }
                        let sc_: Vec<f32> = (0..nv).map(|b| base[b] + if b + 32 >= nv { 3.0 } else { 0.0 }
                            + sigma * hash2(r as u64 + 1000, b as u64)).collect();
                        idx.clear();
                        idx.extend(0..nv as u32);
                        idx.select_nth_unstable_by(top - 1, |x, y| sc_[*y as usize].partial_cmp(&sc_[*x as usize]).unwrap().then(x.cmp(y)));
                        let row = &mut sl[j * ldsel..(j + 1) * ldsel];
                        let mut m = 0;
                        // shuffled block order (fx_topk's order is arbitrary)
                        for i in 0..top {
                            let b = idx[(i * 7919 + r) % top];
                            for c in 0..kp {
                                row[m] = b * kp as u32 + c as u32;
                                m += 1;
                            }
                        }
                        for c in 0..(p + 1) % kp {
                            row[m] = (p - c) as u32;
                            m += 1;
                        }
                        *n = m as u32;
                    }
                });
            }
        });
    }
    let (dkq, dks, dvq, dvs) = (DeviceBuffer::from_slice(&kqv)?, DeviceBuffer::from_slice(&ksv)?,
        DeviceBuffer::from_slice(&vqv)?, DeviceBuffer::from_slice(&vsv)?);
    let (dq, dg) = (DeviceBuffer::from_slice(&qv)?, DeviceBuffer::from_slice(&gv)?);
    let (dsel, dseln) = (DeviceBuffer::from_slice(&sel)?, DeviceBuffer::from_slice(&seln)?);
    let dprm = DeviceBuffer::from_slice(&[pos0 as u32])?;
    let dout = DeviceBuffer::<f32>::zeroed(rows * nq * dh)?;
    let dref = DeviceBuffer::<f32>::zeroed(rows * nq * dh)?;
    let ws = gpu::AttnWs::new(rows, ctx)?;
    let ldg = nq * dh;
    let run_tc = |sel: DevPtr| k.attn_prompt(s, &ws, dout.ptr(), dq.ptr(), dkq.ptr(), dks.ptr(), dvq.ptr(), dvs.ptr(),
        dg.ptr(), ldg, dh, nq, nkv, dprm.ptr(), scale, sel, dseln.ptr(), ldsel, rows);
    let run_simt = || k.attn_ref_simt(s, dref.ptr(), dq.ptr(), dkq.ptr(), dks.ptr(), dvq.ptr(), dvs.ptr(), dg.ptr(), ldg, dh,
        nq, nkv, dprm.ptr(), scale, dsel.ptr(), dseln.ptr(), ldsel, rows);
    let run_pattn = || k.attn_ref_pattn(s, dref.ptr(), dq.ptr(), dkq.ptr(), dks.ptr(), dvq.ptr(), dvs.ptr(), dg.ptr(), ldg, dh,
        nq, nkv, dprm.ptr(), scale, dsel.ptr(), dseln.ptr(), ldsel, rows);
    run_pattn()?;
    s.sync()?;
    let opa = dref.to_vec()?;
    run_tc(dsel.ptr())?;
    run_simt()?;
    s.sync()?;
    let o1 = dout.to_vec()?;
    let orf = dref.to_vec()?;
    // determinism: rerun, and rerun with every row's selection in another order
    run_tc(dsel.ptr())?;
    s.sync()?;
    let same1 = dout.to_vec()?.iter().zip(&o1).all(|(x, y)| x.to_bits() == y.to_bits());
    let mut sel2 = sel.clone();
    for r in 0..rows {
        let n = seln[r] as usize;
        sel2[r * ldsel..r * ldsel + n].reverse();
    }
    let dsel2 = DeviceBuffer::from_slice(&sel2)?;
    run_tc(dsel2.ptr())?;
    s.sync()?;
    let same2 = dout.to_vec()?.iter().zip(&o1).all(|(x, y)| x.to_bits() == y.to_bits());
    // group split: rows [off, rows) as their own group (other tile mates, other pos0) must give the same bits
    let off = 5usize.min(rows - 1);
    let dprm2 = DeviceBuffer::from_slice(&[(pos0 + off) as u32])?;
    let dout2 = DeviceBuffer::<f32>::zeroed(rows * nq * dh)?;
    k.attn_prompt(s, &ws, dout2.ptr(), dq.ptr().add::<f32>(off * nq * dh), dkq.ptr(), dks.ptr(), dvq.ptr(), dvs.ptr(),
        dg.ptr().add::<f32>(off * ldg), ldg, dh, nq, nkv, dprm2.ptr(), scale, dsel.ptr().add::<u32>(off * ldsel),
        dseln.ptr().add::<u32>(off), ldsel, rows - off)?;
    s.sync()?;
    let same3 = dout2.to_vec()?[..(rows - off) * nq * dh].iter().zip(&o1[off * nq * dh..]).all(|(x, y)| x.to_bits() == y.to_bits());
    println!("  group split at +{off} (different tile mates): bit-identical {same3}");
    drop(dout2);
    // union statistics
    let (lst, cnt, _ldl) = ws.lists()?;
    let _ = lst;
    let tiles = rows.div_ceil(16);
    let tot: u64 = cnt.iter().take(tiles).map(|&c| c as u64).sum();
    let mut need = 0u64; // blocks actually used summed over rows, / 16 -> per-row-tile equivalent
    for r in 0..rows {
        let p = pos0 + r;
        need += if seln[r] == 0 { (p / kp + 1) as u64 } else { (top + 1) as u64 };
    }
    // f64 reference on sampled rows
    let mut emax_tc = 0f64;
    let mut emax_si = 0f64;
    let mut esum_tc = 0f64;
    let mut esum_si = 0f64;
    let mut ne = 0usize;
    let mut emax_pa = 0f64;
    let mut esum_pa = 0f64;
    for si in 0..nref {
        let r = if nref == 1 { rows - 1 } else { si * (rows - 1) / (nref - 1) };
        let p = pos0 + r;
        let cells: Vec<usize> = if seln[r] == 0 { (0..=p).collect() } else { sel[r * ldsel..r * ldsel + seln[r] as usize].iter().map(|&c| c as usize).collect() };
        for h in 0..nq {
            let kh = h / (nq / nkv);
            let qr0 = &qv[(r * nq + h) * dh..(r * nq + h + 1) * dh];
            // FA_REFQ16=1: reference with the 16-bit fixed-point Q of ggq_fa_tc2 (isolates the Q rounding)
            let qq: Vec<f32> = if std::env::var("FA_REFQ16").is_ok() {
                let am = qr0.iter().fold(0f32, |m, v| m.max(v.abs()));
                let dq = am / 32639.0;
                qr0.iter().map(|v| if am > 0.0 { (v * (32639.0 / am)).round() * dq } else { 0.0 }).collect()
            } else { qr0.to_vec() };
            let qr = &qq[..];
            let sc: Vec<f64> = cells.iter().map(|&c| {
                let b = (c * nkv + kh) * dh;
                (0..dh).map(|i| qr[i] as f64 * (kqv[b + i] as f64 * ksv[(b + i) / 32] as f64)).sum::<f64>() * scale as f64
            }).collect();
            let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = sc.iter().map(|v| (v - m).exp()).collect();
            let den: f64 = w.iter().sum();
            let mut o = vec![0f64; dh];
            for (j, &c) in cells.iter().enumerate() {
                let b = (c * nkv + kh) * dh;
                for i in 0..dh {
                    o[i] += w[j] * vqv[b + i] as f64 * vsv[(b + i) / 32] as f64;
                }
            }
            let rms = (o.iter().map(|v| (v / den).powi(2)).sum::<f64>() / dh as f64).sqrt();
            for i in 0..dh {
                let gt = gv[r * ldg + h * dh + i] as f64;
                let rv = o[i] / den / (1.0 + (-gt).exp());
                let sg = 1.0 / (1.0 + (-gt).exp());
                let e1 = (o1[(r * nq + h) * dh + i] as f64 - rv).abs() / (rms * sg);
                let e2 = (orf[(r * nq + h) * dh + i] as f64 - rv).abs() / (rms * sg);
                emax_tc = emax_tc.max(e1);
                emax_si = emax_si.max(e2);
                let e3 = (opa[(r * nq + h) * dh + i] as f64 - rv).abs() / (rms * sg);
                emax_pa = emax_pa.max(e3);
                esum_pa += e3;
                esum_tc += e1;
                esum_si += e2;
                ne += 1;
            }
        }
    }
    println!("fa ctx {ctx} rows {rows} sigma {sigma}: bit-identical rerun {same1}, reordered selection {same2}");
    println!("  union: {:.0} 8-cell windows per 16-row tile; a row uses {:.0} blocks = {:.0} window-equivalents ({:.0}% of the work is used)",
        tot as f64 / tiles as f64, need as f64 / rows as f64, need as f64 / rows as f64 / 2.0,
        100.0 * (need as f64 / rows as f64 / 2.0) / (tot as f64 / tiles as f64));
    println!("  error vs f64 (|d| / row rms): tc max {emax_tc:.2e} mean {:.2e} | simt max {emax_si:.2e} mean {:.2e} ({nref} rows x {nq} heads)",
        esum_tc / ne as f64, esum_si / ne as f64);
    println!("  fx_pattn (bf16) error: max {emax_pa:.2e} mean {:.2e}", esum_pa / ne as f64);
    // timing
    let (e0, e1, e2, e3) = (fe_cuda::Event::new()?, fe_cuda::Event::new()?, fe_cuda::Event::new()?, fe_cuda::Event::new()?);
    for it in 0..3 {
        s.record(&e0)?;
        run_tc(dsel.ptr())?;
        s.record(&e1)?;
        run_simt()?;
        s.record(&e2)?;
        run_pattn()?;
        s.record(&e3)?;
        s.sync()?;
        println!("  run {it}: tc {:.2} ms, simt {:.2} ms, fx_pattn {:.2} ms", e1.elapsed_since(&e0)?, e2.elapsed_since(&e1)?,
            e3.elapsed_since(&e2)?);
    }
    Ok(())
}

/// Rows (of `n` values) of `got` whose bits differ from `want`.
fn rows_differ(got: &[f32], want: &[f32], n: usize) -> usize {
    got.chunks(n).zip(want.chunks(n)).filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.to_bits() != y.to_bits())).count()
}

/// binv: batch invariance of every decode path. Each row computed in a batch of B = 2..8 must have the bits it gets
/// computed alone (B = 1). Prints one line per path with the differing rows per B; "OK" when none.
/// Batch sizes checked by binv (decode rows per step: 4 lanes x 5 MTP rows = 20, + margin).
const BMAX: usize = 24;

fn binv(g: &Gguf, k: &Ggq, s: &Stream, first: &BTreeMap<u32, &TensorInfo>) -> Result<()> {
    let mut rng = Lcg(4242);
    let mut total_bad = 0usize;
    let mut tb = 0usize;
    let mut report = |name: &str, bad: &[usize]| {
        let t: usize = bad.iter().sum();
        tb += t;
        if t == 0 {
            println!("{name:44} OK");
        } else {
            println!("{name:44} DIFFER: rows per B=1..{BMAX} {bad:?}");
        }
    };
    // q8 activation quantization
    {
        let kk = 2560;
        let x: Vec<f32> = (0..BMAX * kk).map(|_| rng.f() * 2.0).collect();
        let dx = DeviceBuffer::from_slice(&x)?;
        let mut a = Q8Act::alloc(BMAX, kk)?;
        let mut a1 = Q8Act::alloc(1, kk)?;
        let mut bad = vec![0usize; BMAX];
        let mut single = vec![];
        for r in 0..BMAX {
            k.quantize_q8_1(s, dx.ptr().add::<f32>(r * kk), kk, false, 1, kk, &mut a1)?;
            s.sync()?;
            single.push((a1.q.to_vec()?[..kk].to_vec(), a1.d.to_vec()?[..kk / 32].to_vec(), a1.s.to_vec()?[..kk / 32].to_vec()));
        }
        for b in 1..=BMAX {
            k.quantize_q8_1(s, dx.ptr(), kk, false, b, kk, &mut a)?;
            s.sync()?;
            let (q, d, sm) = (a.q.to_vec()?, a.d.to_vec()?, a.s.to_vec()?);
            for r in 0..b {
                let (sq, sd, ss) = &single[r];
                let same = q[r * kk..(r + 1) * kk] == sq[..] && d[r * kk / 32..(r + 1) * kk / 32].iter().zip(sd).all(|(x, y)| x.to_bits() == y.to_bits())
                    && sm[r * kk / 32..(r + 1) * kk / 32].iter().zip(ss).all(|(x, y)| x.to_bits() == y.to_bits());
                if !same {
                    bad[b - 1] += 1;
                }
            }
        }
        report("quantize_q8_1", &bad);
        let mut cb = vec![0usize; BMAX];
        for r in 0..BMAX {
            let c = cpu::avx2::Q8Rows::quantize(&x[r * kk..(r + 1) * kk], 1, kk);
            let (sq, sd, ss) = &single[r];
            if c.q != *sq || c.d.iter().zip(sd).any(|(x, y)| x.to_bits() != y.to_bits()) || c.s.iter().zip(ss).any(|(x, y)| x.to_bits() != y.to_bits()) {
                cb[0] += 1;
            }
        }
        report("  CPU Q8Rows == GPU quantize_q8_1 (8 rows)", &cb);
    }
    // GEMV per type (default config, as the engine calls it)
    for (_, t) in first.iter() {
        if !gpu::GEMV_TYPES.contains(&t.ty) || t.dims.len() < 2 || t.dims[1] < 64 {
            continue;
        }
        let kk = t.ne0();
        let n = (t.dims[1] as usize).min(2048);
        let dw = DeviceBuffer::from_slice(&g.data(t)[..n * t.row_bytes()])?;
        let x: Vec<f32> = (0..BMAX * kk).map(|_| rng.f()).collect();
        let dx = DeviceBuffer::from_slice(&x)?;
        let mut a = Q8Act::alloc(BMAX, kk)?;
        let dy = DeviceBuffer::<f32>::alloc(BMAX * n)?;
        let mut single = vec![];
        for r in 0..BMAX {
            k.quantize_q8_1(s, dx.ptr().add::<f32>(r * kk), kk, false, 1, kk, &mut a)?;
            k.gemv(s, t.ty, dw.ptr(), t.row_bytes(), n, &a, dy.ptr(), false, None)?;
            s.sync()?;
            single.extend_from_slice(&dy.to_vec()?[..n]);
        }
        let mut bad = vec![0usize; BMAX];
        let mut cbad = vec![0usize; BMAX];
        let lpr = gpu::default_lpr(kk) as usize;
        for b in 1..=BMAX {
            k.quantize_q8_1(s, dx.ptr(), kk, false, b, kk, &mut a)?;
            k.gemv(s, t.ty, dw.ptr(), t.row_bytes(), n, &a, dy.ptr(), false, None)?;
            s.sync()?;
            let gy = dy.to_vec()?[..b * n].to_vec();
            bad[b - 1] = rows_differ(&gy, &single[..b * n], n);
            if kk % 128 == 0 && lpr >= 4 {
                let q = cpu::avx2::Q8Rows::quantize(&x[..b * kk], b, kk);
                let mut cy = vec![0f32; b * n];
                unsafe { cpu::gemv_rows_gpu(t.ty, g.data(t).as_ptr(), t.row_bytes(), 0..n, &q, cy.as_mut_ptr(), n, lpr) };
                cbad[b - 1] = rows_differ(&cy, &gy, n);
            }
        }
        report(&format!("gemv {} [{kk} -> {n}]", t.ty.name()), &bad);
        report(&format!("  CPU gemv_rows_gpu == GPU gemv {} (lpr {lpr})", t.ty.name()), &cbad);
    }
    if g.tensor("blk.0.hc_attn_down.weight").is_none() {
        println!("hc / moe sections need Flash-Next tensors: skipped");
        return Ok(());
    }
    // BF16 GEMV + fused hc kernels (layer 0 hc weights)
    {
        let (dt, it, ut, nt) = (g.tensor("blk.0.hc_attn_down.weight").unwrap(), g.tensor("blk.0.hc_attn_inject.weight").unwrap(),
            g.tensor("blk.0.hc_attn_up.weight").unwrap(), g.tensor("blk.0.hc_attn_norm.weight").unwrap());
        let mut cat = g.data(dt).to_vec();
        cat.extend_from_slice(g.data(it));
        let nw: Vec<f32> = g.data(nt).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let (dcat, du, dnw) = (DeviceBuffer::from_slice(&cat)?, DeviceBuffer::from_slice(g.data(ut))?, DeviceBuffer::from_slice(&nw)?);
        let hd = 10240usize;
        let res: Vec<f32> = (0..BMAX * hd).map(|_| rng.f() * 3.0).collect();
        let dres = DeviceBuffer::from_slice(&res)?;
        let (dlo, dxn, dmix, dy) = (DeviceBuffer::<f32>::alloc(BMAX * 324)?, DeviceBuffer::<f32>::alloc(BMAX * hd)?,
            DeviceBuffer::<f32>::alloc(BMAX * 2560)?, DeviceBuffer::<f32>::alloc(BMAX * 324)?);
        // gemv_bf16 [10240 -> 324] and [320 -> 10240]
        let run_bf = |b: usize, r0: usize| -> Result<(Vec<f32>, Vec<f32>)> {
            k.gemv_bf16(s, dcat.ptr(), 324, hd, dres.ptr().add::<f32>(r0 * hd), hd, b, dy.ptr(), false)?;
            k.gemv_bf16(s, du.ptr(), hd, 320, dres.ptr().add::<f32>(r0 * hd), hd, b, dxn.ptr(), false)?;
            s.sync()?;
            Ok((dy.to_vec()?[..b * 324].to_vec(), dxn.to_vec()?[..b * hd].to_vec()))
        };
        let mut s1 = (vec![], vec![]);
        for r in 0..BMAX {
            let (a, c) = run_bf(1, r)?;
            s1.0.extend(a);
            s1.1.extend(c);
        }
        let (mut bad1, mut bad2) = (vec![0usize; BMAX], vec![0usize; BMAX]);
        for b in 1..=BMAX {
            let (a, c) = run_bf(b, 0)?;
            bad1[b - 1] = rows_differ(&a, &s1.0[..b * 324], 324);
            bad2[b - 1] = rows_differ(&c, &s1.1[..b * hd], hd);
        }
        report("gemv_bf16 [10240 -> 324]", &bad1);
        report("gemv_bf16 [320 -> 10240]", &bad2);
        let run_hc = |b: usize, r0: usize| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
            k.hc_down(s, dlo.ptr(), 324, dcat.ptr(), 324, dres.ptr().add::<f32>(r0 * hd), hd, dnw.ptr(), 1e-6, 0.25, 320, dxn.ptr(), b)?;
            k.hc_up(s, dmix.ptr(), 2560, du.ptr(), dlo.ptr(), 324, dxn.ptr(), hd, 0.25, b)?;
            s.sync()?;
            Ok((dlo.to_vec()?[..b * 324].to_vec(), dxn.to_vec()?[..b * hd].to_vec(), dmix.to_vec()?[..b * 2560].to_vec()))
        };
        let mut h1 = (vec![], vec![], vec![]);
        for r in 0..BMAX {
            let (a, b_, c) = run_hc(1, r)?;
            h1.0.extend(a);
            h1.1.extend(b_);
            h1.2.extend(c);
        }
        let (mut b1, mut b2, mut b3) = (vec![0usize; BMAX], vec![0usize; BMAX], vec![0usize; BMAX]);
        for b in 1..=BMAX {
            let (a, x, c) = run_hc(b, 0)?;
            b1[b - 1] = rows_differ(&a, &h1.0[..b * 324], 324);
            b2[b - 1] = rows_differ(&x, &h1.1[..b * hd], hd);
            b3[b - 1] = rows_differ(&c, &h1.2[..b * 2560], 2560);
        }
        report("hc_down lo", &b1);
        report("hc_down xn", &b2);
        report("hc_up mixed", &b3);
    }
    // MoE: gate/up (+q8 of h), down, combine with CPU rows; tokens share experts; CpuMoe with VNNI on and off
    let (ne, nf, topk) = (2560usize, 640usize, 10usize);
    for layer in [0usize, 1, 17, 48] {
        let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
        let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
        let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
        let nexp = gt.dims[2] as usize;
        let (dg, du, dd) = (DeviceBuffer::from_slice(g.data(gt))?, DeviceBuffer::from_slice(g.data(ut))?,
            DeviceBuffer::from_slice(g.data(dt))?);
        let missing = |e: usize| e % 7 == 3;
        let tab = |b: &DeviceBuffer<u8>, sb: usize| -> Vec<u64> {
            (0..nexp).map(|e| if missing(e) { 0 } else { b.ptr().0 + (e * sb) as u64 }).collect()
        };
        let (tg, tu, td) = (DeviceBuffer::from_slice(&tab(&dg, gt.slice_bytes()))?, DeviceBuffer::from_slice(&tab(&du, ut.slice_bytes()))?,
            DeviceBuffer::from_slice(&tab(&dd, dt.slice_bytes()))?);
        let x: Vec<f32> = (0..BMAX * ne).map(|_| rng.f()).collect();
        // experts from a pool of 16 so tokens share many
        let mut ids = vec![0i32; BMAX * topk];
        let mut w = vec![0f32; BMAX * topk];
        for t in 0..BMAX {
            let mut chosen: Vec<i32> = vec![];
            while chosen.len() < topk {
                let e = (((rng.f() + 1.0) * 8.0) as i32 % 16) * (nexp as i32 / 16) + (layer as i32 % 3);
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }
            ids[t * topk..(t + 1) * topk].copy_from_slice(&chosen);
            for kk in 0..topk {
                w[t * topk + kk] = rng.f().abs() / topk as f32 + 0.01;
            }
        }
        let mut xa = Q8Act::alloc(BMAX, ne)?;
        let mut ha = Q8Act::alloc(BMAX * topk, nf)?;
        let dy = DeviceBuffer::<f32>::alloc(BMAX * topk * ne)?;
        let dout = DeviceBuffer::<f32>::alloc(BMAX * ne)?;
        let (dcnt, dpairs, dslot) = (DeviceBuffer::<u32>::alloc(1)?, DeviceBuffer::<u32>::alloc(BMAX * topk)?, DeviceBuffer::<i32>::alloc(BMAX * topk)?);
        let dycpu = DeviceBuffer::<f32>::alloc(BMAX * topk * ne)?;
        let mut cm = cpu::CpuMoe::new(8, ne, nf);
        // one step: tokens t0 .. t0 + nt -> (h q8 rows as f32 bits of q|d, y of GPU pairs, out, cpu rows)
        let mut step = |t0: usize, nt: usize| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
            let dids = DeviceBuffer::from_slice(&ids[t0 * topk..(t0 + nt) * topk])?;
            let dw = DeviceBuffer::from_slice(&w[t0 * topk..(t0 + nt) * topk])?;
            let dx = DeviceBuffer::from_slice(&x[t0 * ne..(t0 + nt) * ne])?;
            k.quantize_q8_1(s, dx.ptr(), ne, false, nt, ne, &mut xa)?;
            k.moe_gu_q8(s, gt.ty, tg.ptr(), tu.ptr(), gt.row_bytes(), dids.ptr(), nt, topk, &xa, nf, &mut ha, None, None)?;
            k.moe_misses(s, tg.ptr(), dids.ptr(), nt, topk, dcnt.ptr(), dpairs.ptr(), dslot.ptr())?;
            k.moe_down(s, dt.ty, td.ptr(), dt.row_bytes(), dids.ptr(), nt, topk, &ha, ne, dy.ptr(), None)?;
            s.sync()?;
            let cnt = dcnt.to_vec()?[0] as usize;
            let pairs = dpairs.to_vec()?;
            let xs = &x[t0 * ne..(t0 + nt) * ne];
            let mut jobs: Vec<cpu::MoeJob> = vec![];
            let mut job_slots: Vec<Vec<usize>> = vec![];
            for (sl, &p) in pairs[..cnt].iter().enumerate() {
                let e = ids[t0 * topk + p as usize] as usize;
                if let Some(j) = jobs.iter().position(|j| std::ptr::eq(j.gate.as_ptr(), g.slice(gt, e).as_ptr())) {
                    jobs[j].rows.push(p as usize / topk);
                    job_slots[j].push(sl);
                } else {
                    jobs.push(cpu::MoeJob { gate: g.slice(gt, e), up: g.slice(ut, e), down: g.slice(dt, e), ty_gu: gt.ty,
                        ty_down: dt.ty, rows: vec![p as usize / topk] });
                    job_slots.push(vec![sl]);
                }
            }
            let mut ycpu = vec![0f32; cnt.max(1) * ne];
            if cnt > 0 {
                cm.run(&jobs, xs)?;
                for (j, sls) in job_slots.iter().enumerate() {
                    for (bi, &sl) in sls.iter().enumerate() {
                        ycpu[sl * ne..(sl + 1) * ne].copy_from_slice(&cm.out(j)[bi * ne..(bi + 1) * ne]);
                    }
                }
                fe_cuda::memcpy_htod(dycpu.ptr(), &ycpu[..cnt * ne])?;
            }
            k.moe_combine_cpu(s, dout.ptr(), dy.ptr(), dw.ptr(), nt, topk, ne, false, dycpu.ptr(), dslot.ptr())?;
            s.sync()?;
            // h as bits: q (as f32 values) then d
            let (hq, hd) = (ha.q.to_vec()?, ha.d.to_vec()?);
            let mut h = vec![];
            for p in 0..nt * topk {
                if missing(ids[t0 * topk + p] as usize) {
                    // not computed on the card (the CPU makes its own h)
                    h.extend(std::iter::repeat(0f32).take(nf + nf / 32));
                    continue;
                }
                h.extend(hq[p * nf..(p + 1) * nf].iter().map(|v| *v as f32));
                h.extend_from_slice(&hd[p * nf / 32..(p + 1) * nf / 32]);
            }
            let mut y = dy.to_vec()?[..nt * topk * ne].to_vec();
            for p in 0..nt * topk {
                if missing(ids[t0 * topk + p] as usize) {
                    y[p * ne..(p + 1) * ne].fill(0.0);
                }
            }
            for (sl, &p) in pairs[..cnt].iter().enumerate() {
                y[p as usize * ne..(p as usize + 1) * ne].copy_from_slice(&ycpu[sl * ne..(sl + 1) * ne]);
            }
            Ok((h, y, dout.to_vec()?[..nt * ne].to_vec()))
        };
        for vnni in [true, false] {
            cpu::avx2::set_vnni(vnni);
            let hl = nf + nf / 32;
            let mut one = (vec![], vec![], vec![]);
            for t in 0..BMAX {
                let (h, y, o) = step(t, 1)?;
                one.0.extend(h);
                one.1.extend(y);
                one.2.extend(o);
            }
            let (mut bh, mut by, mut bo) = (vec![0usize; BMAX], vec![0usize; BMAX], vec![0usize; BMAX]);
            for b in 1..=BMAX {
                let (h, y, o) = step(0, b)?;
                bh[b - 1] = rows_differ(&h, &one.0[..b * topk * hl], hl);
                by[b - 1] = rows_differ(&y, &one.1[..b * topk * ne], ne);
                bo[b - 1] = rows_differ(&o, &one.2[..b * ne], ne);
            }
            let tag = format!("L{layer} {}/{} vnni {}", gt.ty.name(), dt.ty.name(), if vnni { "on" } else { "off" });
            report(&format!("moe_gu_q8 h pairs {tag}"), &bh);
            report(&format!("moe_down + CpuMoe y pairs {tag}"), &by);
            report(&format!("moe_combine_cpu out {tag}"), &bo);
        }
        cpu::avx2::set_vnni(true);
    }
    total_bad += tb;
    // GPU vs CPU for the same (expert, token): a pair is computed on the card when its expert is resident and on the
    // CPU when it is not, and residency differs between runs (MTP on / off), so the two must agree bit for bit
    for layer in [0usize, 1, 17, 48] {
        let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
        let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
        let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
        let nexp = gt.dims[2] as usize;
        let (dg, du, dd) = (DeviceBuffer::from_slice(g.data(gt))?, DeviceBuffer::from_slice(g.data(ut))?,
            DeviceBuffer::from_slice(g.data(dt))?);
        let tab = |b: &DeviceBuffer<u8>, sb: usize| -> Vec<u64> { (0..nexp).map(|e| b.ptr().0 + (e * sb) as u64).collect() };
        let (tg, tu, td) = (DeviceBuffer::from_slice(&tab(&dg, gt.slice_bytes()))?, DeviceBuffer::from_slice(&tab(&du, ut.slice_bytes()))?,
            DeviceBuffer::from_slice(&tab(&dd, dt.slice_bytes()))?);
        let x: Vec<f32> = (0..ne).map(|_| rng.f()).collect();
        let ids: Vec<i32> = (0..topk as i32).map(|i| i * 13 + 5).collect();
        let mut xa = Q8Act::alloc(1, ne)?;
        let mut ha = Q8Act::alloc(topk, nf)?;
        let dy = DeviceBuffer::<f32>::alloc(topk * ne)?;
        let dids = DeviceBuffer::from_slice(&ids)?;
        let dx = DeviceBuffer::from_slice(&x)?;
        k.quantize_q8_1(s, dx.ptr(), ne, false, 1, ne, &mut xa)?;
        k.moe_gu_q8(s, gt.ty, tg.ptr(), tu.ptr(), gt.row_bytes(), dids.ptr(), 1, topk, &xa, nf, &mut ha, None, None)?;
        k.moe_down(s, dt.ty, td.ptr(), dt.row_bytes(), dids.ptr(), 1, topk, &ha, ne, dy.ptr(), None)?;
        s.sync()?;
        let gy = dy.to_vec()?;
        let (hq_g, hd_g) = (ha.q.to_vec()?, ha.d.to_vec()?);
        let mut cm = cpu::CpuMoe::new(8, ne, nf);
        let jobs: Vec<cpu::MoeJob> = ids.iter().map(|&e| cpu::MoeJob { gate: g.slice(gt, e as usize), up: g.slice(ut, e as usize),
            down: g.slice(dt, e as usize), ty_gu: gt.ty, ty_down: dt.ty, rows: vec![0] }).collect();
        cm.run(&jobs, &x)?;
        let mut ybad = 0;
        let mut hbad = 0;
        let mut maxrel = 0f64;
        for p in 0..topk {
            let c = cm.out(p);
            if c.iter().zip(&gy[p * ne..(p + 1) * ne]).any(|(a, b)| a.to_bits() != b.to_bits()) {
                ybad += 1;
            }
            for i in 0..ne {
                maxrel = maxrel.max(((c[i] - gy[p * ne + i]) as f64).abs() / (1e-6 + (gy[p * ne + i] as f64).abs()));
            }
            let (cq, cd) = cm.h_q8(p);
            if cq != &hq_g[p * nf..(p + 1) * nf] || cd.iter().zip(&hd_g[p * nf / 32..(p + 1) * nf / 32]).any(|(a, b)| a.to_bits() != b.to_bits()) {
                hbad += 1;
            }
        }
        // the many-row CPU path (run_tiled, prompt-sized jobs) on the same pairs must give the same bits too
        cm.run_tiled(&jobs, &x)?;
        let mut tbad = 0;
        for p in 0..topk {
            if cm.out(p).iter().zip(&gy[p * ne..(p + 1) * ne]).any(|(a, b)| a.to_bits() != b.to_bits()) {
                tbad += 1;
            }
        }
        // and with 13 rows per job (several 4-row micro groups) vs each row alone through run()
        let x13: Vec<f32> = (0..13 * ne).map(|_| rng.f()).collect();
        let jobs13: Vec<cpu::MoeJob> = ids[..3].iter().map(|&e| cpu::MoeJob { gate: g.slice(gt, e as usize), up: g.slice(ut, e as usize),
            down: g.slice(dt, e as usize), ty_gu: gt.ty, ty_down: dt.ty, rows: (0..13).collect() }).collect();
        cm.run_tiled(&jobs13, &x13)?;
        let t13: Vec<Vec<f32>> = (0..3).map(|j| cm.out(j).to_vec()).collect();
        let mut rbad = 0;
        for r in 0..13 {
            let j1: Vec<cpu::MoeJob> = ids[..3].iter().map(|&e| cpu::MoeJob { gate: g.slice(gt, e as usize), up: g.slice(ut, e as usize),
                down: g.slice(dt, e as usize), ty_gu: gt.ty, ty_down: dt.ty, rows: vec![r] }).collect();
            cm.run(&j1, &x13)?;
            for j in 0..3 {
                if cm.out(j).iter().zip(&t13[j][r * ne..(r + 1) * ne]).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    rbad += 1;
                }
            }
        }
        let tn = format!("CPU tiled vs GPU / per-row L{layer} {}/{}", gt.ty.name(), dt.ty.name());
        if tbad + rbad == 0 {
            println!("{tn:44} OK");
        } else {
            println!("{tn:44} DIFFER: vs GPU {tbad} of {topk} pairs, 13-row jobs vs per-row {rbad} of 39");
        }
        total_bad += tbad + rbad;
        let name = format!("GPU vs CPU expert L{layer} {}/{}", gt.ty.name(), dt.ty.name());
        if ybad + hbad == 0 {
            println!("{name:44} OK");
        } else {
            println!("{name:44} DIFFER: h {hbad} / y {ybad} of {topk} pairs (y max rel {maxrel:.1e})");
        }
        total_bad += ybad + hbad;
    }
    // prompt path: CPU run_mmq_order vs the GPU prompt kernels (moe_sort + mmq_moe_gu + mmq_moe by pair), per pair
    for layer in [0usize, 1, 17, 48] {
        let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
        let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
        let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
        let nexp = gt.dims[2] as usize;
        let (dg, du, dd) = (DeviceBuffer::from_slice(g.data(gt))?, DeviceBuffer::from_slice(g.data(ut))?,
            DeviceBuffer::from_slice(g.data(dt))?);
        let tab = |b: &DeviceBuffer<u8>, sb: usize| -> Vec<u64> { (0..nexp).map(|e| b.ptr().0 + (e * sb) as u64).collect() };
        let (tg, tu, td) = (DeviceBuffer::from_slice(&tab(&dg, gt.slice_bytes()))?, DeviceBuffer::from_slice(&tab(&du, ut.slice_bytes()))?,
            DeviceBuffer::from_slice(&tab(&dd, dt.slice_bytes()))?);
        let nt = 13usize;
        let np = nt * topk;
        let x: Vec<f32> = (0..nt * ne).map(|_| rng.f()).collect();
        let mut ids = vec![0i32; np];
        for t in 0..nt {
            let mut ch: Vec<i32> = vec![];
            while ch.len() < topk {
                let e = (((rng.f() + 1.0) * 8.0) as i32 % 16) * (nexp as i32 / 16) + 1;
                if !ch.contains(&e) {
                    ch.push(e);
                }
            }
            ids[t * topk..(t + 1) * topk].copy_from_slice(&ch);
        }
        let dids = DeviceBuffer::from_slice(&ids)?;
        let dx = DeviceBuffer::from_slice(&x)?;
        let mut xa = Q8Act::alloc(nt, ne)?;
        let mut hq = Q8Act::alloc(np, nf)?;
        let ws = SortWs::alloc(np, nexp)?;
        let dy = DeviceBuffer::<f32>::alloc(np * ne)?;
        k.quantize_q8_1(s, dx.ptr(), ne, false, nt, ne, &mut xa)?;
        k.moe_sort(s, dids.ptr(), nt, topk, &ws)?;
        k.mmq_moe_gu(s, gt.ty, &xa, tg.ptr(), tu.ptr(), gt.row_bytes(), nf, &ws, np, &mut hq)?;
        k.mmq_moe(s, dt.ty, dy.ptr(), ne, &hq, td.ptr(), dt.row_bytes(), ne, &ws, np, true, false)?;
        s.sync()?;
        let (gy, ghq, ghd) = (dy.to_vec()?, hq.q.to_vec()?, hq.d.to_vec()?);
        let xq = cpu::avx2::Q8Rows::from_parts(ne, nt, &xa.q.to_vec()?, &xa.d.to_vec()?, &xa.s.to_vec()?);
        let mut experts: Vec<i32> = ids.clone();
        experts.sort();
        experts.dedup();
        let jobs: Vec<cpu::MoeJob> = experts.iter().map(|&e| cpu::MoeJob { gate: g.slice(gt, e as usize), up: g.slice(ut, e as usize),
            down: g.slice(dt, e as usize), ty_gu: gt.ty, ty_down: dt.ty,
            rows: (0..nt).filter(|t| ids[t * topk..(t + 1) * topk].contains(&e)).collect() }).collect();
        let mut cm = cpu::CpuMoe::new(8, ne, nf);
        cm.run_mmq_order(&jobs, &xq)?;
        let (mut ybad, mut hbad) = (0usize, 0usize);
        for (j, &e) in experts.iter().enumerate() {
            let (cq, cd) = cm.h_q8(j);
            for (i, &t) in jobs[j].rows.iter().enumerate() {
                let p = t * topk + ids[t * topk..(t + 1) * topk].iter().position(|&v| v == e).unwrap();
                if cm.out(j)[i * ne..(i + 1) * ne].iter().zip(&gy[p * ne..(p + 1) * ne]).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    ybad += 1;
                }
                if cq[i * nf..(i + 1) * nf] != ghq[p * nf..(p + 1) * nf]
                    || cd[i * nf / 32..(i + 1) * nf / 32].iter().zip(&ghd[p * nf / 32..(p + 1) * nf / 32]).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    hbad += 1;
                }
            }
        }
        let name = format!("CPU run_mmq_order vs GPU prompt MMQ L{layer} {}/{}", gt.ty.name(), dt.ty.name());
        if ybad + hbad == 0 {
            println!("{name:52} OK ({np} pairs, {} experts)", experts.len());
        } else {
            println!("{name:52} DIFFER: h {hbad} / y {ybad} of {np} pairs");
        }
        total_bad += ybad + hbad;
    }
    println!("batch invariance: {total_bad} differing rows in total");
    Ok(())
}
