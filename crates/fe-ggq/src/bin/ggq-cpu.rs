//! ggq-cpu <model.gguf> <check|bench|moe> [threads...]
//! - check: AVX2 GEMV vs the f64 host reference (same q8 activations) for every type, B = 1..8; the scalar
//!   fallback too; and one full CpuMoe job vs a host reference (dequantized weights, f64).
//! - bench: weights/s of the expert GEMVs on a working set far larger than L3 (64 experts copied to anonymous
//!   memory), per thread count and B; plus a plain multi-threaded read for the DRAM ceiling.
//! - moe: CpuMoe::run time for 10 experts x B rows of one layer (the per-layer miss path).
use anyhow::Result;
use fe_ggq::cpu::{self, avx2::Q8Rows, CpuMoe, MoeJob, Pool};
use fe_ggq::{refq, GgmlType};
use fe_gguf::Gguf;
use std::time::Instant;

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
    let threads: Vec<usize> = a.iter().skip(3).filter_map(|s| s.parse().ok()).collect();
    println!("avx2+fma: {}", cpu::has_avx2());
    match a[2].as_str() {
        "check" => check(&g),
        "bench" => bench(&g, &threads),
        "moe" => moe(&g, &threads),
        "lat" => lat(&g, &a[3..]),
        "tiled" => tiled(&g, &a[3..]),
        "small" => small(&g, &a[3..]),
        "mmqorder" => {
            // run_mmq_order (prompt MMQ bits) vs run_tiled (decode GEMV bits): time per expert for M rows, 10 cold experts
            let cpus = fe_ggq::cpu::pool::parse_cpus(a.get(3).map(|s| s.as_str()).unwrap_or("8-23"));
            fe_ggq::cpu::pool::pin_current(0);
            let mut m = CpuMoe::pinned(&cpus, std::time::Duration::from_millis(20), 2560, 640);
            let mut rng = Lcg(9);
            for layer in [0usize, 17, 48] {
                let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
                let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
                let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
                let ne = 120;
                let copy = |t: &fe_gguf::TensorInfo| -> Vec<Vec<u8>> { (0..ne).map(|e| g.slice(t, e).to_vec()).collect() };
                let (gc, uc, dc) = (copy(gt), copy(ut), copy(dt));
                let mut line = format!("layer {layer:2} {}/{}:", gt.ty.name(), dt.ty.name());
                for mm in [1usize, 4, 8, 32] {
                    let x: Vec<f32> = (0..mm * 2560).map(|_| rng.f()).collect();
                    let xq = Q8Rows::quantize(&x, mm, 2560);
                    let (mut t1, mut t2) = (vec![], vec![]);
                    for it in 0..40 {
                        let jobs: Vec<MoeJob> = (0..10).map(|i| {
                            let e = (it * 13 + i * 7) % ne;
                            MoeJob { gate: &gc[e], up: &uc[e], down: &dc[e], ty_gu: gt.ty, ty_down: dt.ty, rows: (0..mm).collect() }
                        }).collect();
                        let t0 = Instant::now();
                        m.run_mmq_order(&jobs, &xq)?;
                        t1.push(t0.elapsed().as_secs_f64() * 1e6 / 10.0);
                        let t0 = Instant::now();
                        m.run_tiled(&jobs, &x)?;
                        t2.push(t0.elapsed().as_secs_f64() * 1e6 / 10.0);
                    }
                    t1.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    t2.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    line += &format!("  M{mm} {:.0} us/exp (tiled {:.0})", t1[20], t2[20]);
                }
                println!("{line}");
            }
            Ok(())
        }
        "vnni" => {
            // VNNI kernels vs the plain AVX2 kernels, bit for bit: every supported type, B = 1..8, real tensors;
            // plus a CpuMoe job (B = 1..5 rows) per layer type pair
            let mut rng = Lcg(21);
            let mut seen = vec![];
            let mut bad = 0usize;
            println!("vnni available: {}", { cpu::avx2::set_vnni(true); cpu::avx2::vnni() });
            for t in &g.tensors {
                if !cpu::supported(t.ty) || seen.contains(&t.ty) || t.dims.len() < 2 {
                    continue;
                }
                seen.push(t.ty);
                let (kk, rows, rb) = (t.ne0(), (t.dims[1] as usize).min(256), t.row_bytes());
                let w = &g.data(t)[..rows * rb];
                for b in 1..=8usize {
                    let x: Vec<f32> = (0..b * kk).map(|_| rng.f() * 3.0).collect();
                    let q = Q8Rows::quantize(&x, b, kk);
                    let mut y1 = vec![0f32; b * rows];
                    let mut y2 = vec![0f32; b * rows];
                    cpu::avx2::set_vnni(true);
                    unsafe { cpu::gemv_rows(t.ty, w.as_ptr(), rb, 0..rows, &q, y1.as_mut_ptr(), rows) };
                    cpu::avx2::set_vnni(false);
                    unsafe { cpu::gemv_rows(t.ty, w.as_ptr(), rb, 0..rows, &q, y2.as_mut_ptr(), rows) };
                    let nd = y1.iter().zip(&y2).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    if nd > 0 {
                        println!("{} B{b}: {nd} of {} outputs differ", t.ty.name(), y1.len());
                    }
                    bad += nd;
                }
                println!("{:8} B=1..8 checked", t.ty.name());
            }
            let mut moe = CpuMoe::new(4, 2560, 640);
            for layer in [0usize, 1, 17, 48] {
                let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
                let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
                let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
                let x: Vec<f32> = (0..5 * 2560).map(|_| rng.f()).collect();
                let jobs: Vec<MoeJob> = (1..=5usize).map(|b| MoeJob { gate: g.slice(gt, b * 3), up: g.slice(ut, b * 3),
                    down: g.slice(dt, b * 3), ty_gu: gt.ty, ty_down: dt.ty, rows: (0..b).collect() }).collect();
                cpu::avx2::set_vnni(true);
                moe.run(&jobs, &x)?;
                let o1: Vec<Vec<f32>> = (0..5).map(|j| moe.out(j).to_vec()).collect();
                cpu::avx2::set_vnni(false);
                moe.run(&jobs, &x)?;
                let nd: usize = (0..5).map(|j| moe.out(j).iter().zip(&o1[j]).filter(|(a, b)| a.to_bits() != b.to_bits()).count()).sum();
                println!("CpuMoe layer {layer} ({}/{}) B1..5: {nd} outputs differ between VNNI and AVX2", gt.ty.name(), dt.ty.name());
                bad += nd;
            }
            cpu::avx2::set_vnni(true);
            println!("VNNI vs AVX2: {bad} differing outputs in total");
            Ok(())
        }
        "q8" => {
            // AVX2 q8 block quantizer vs refq::q8_block, bit for bit (random scales, ties, zeros, subnormals, big)
            let mut rng = Lcg(11);
            let mut bad = 0usize;
            let n = 2_000_000usize;
            for it in 0..n {
                let sc = [1e-30f32, 1e-6, 0.01, 1.0, 7.3, 300.0, 6e4, 1e30][it % 8];
                let mut x: Vec<f32> = (0..32).map(|_| rng.f() * sc).collect();
                match it % 5 {
                    0 => {
                        // exact .5 ties after scaling: amax = 127 -> d = 1
                        x[0] = 127.0;
                        for j in 1..32 { x[j] = ((rng.f() * 120.0).round() + 0.5) * if j % 2 == 0 { 1.0 } else { -1.0 }; }
                    }
                    1 => { for v in x.iter_mut().skip(3) { *v = 0.0; } }
                    2 => { x = vec![0.0; 32]; }
                    _ => {}
                }
                let (rq, rd, rs) = refq::q8_block(&x);
                let mut q = [0i8; 32];
                let mut qu = [0u8; 32];
                let (d, sm) = unsafe { cpu::avx2::q8_block_fast(x.as_ptr(), q.as_mut_ptr(), qu.as_mut_ptr()) };
                let ok = q == rq && d.to_bits() == rd.to_bits() && sm.to_bits() == rs.to_bits()
                    && qu.iter().zip(&q).all(|(u, v)| *u == (*v as u8) ^ 0x80);
                if !ok {
                    if bad < 5 {
                        println!("mismatch it {it}: d {d} vs {rd}, s {sm} vs {rs}, q {:?} vs {:?}", &q[..8], &rq[..8]);
                    }
                    bad += 1;
                }
            }
            println!("q8_block_fast vs refq: {bad} of {n} blocks differ");
            let x: Vec<f32> = (0..32 * 4096).map(|_| rng.f()).collect();
            let t0 = Instant::now();
            let mut q = Q8Rows::default();
            for _ in 0..100 { q.set(&x, 4096, 32); }
            println!("Q8Rows::set: {:.1} ns per 32-block", t0.elapsed().as_secs_f64() / 100.0 / 4096.0 * 1e9);
            Ok(())
        }
        "tiled1" => {
            // single thread, layer 0, 1 expert x 128 rows, repeated (for perf stat)
            let gt = g.tensor("blk.0.ffn_gate_exps.weight").unwrap();
            let ut = g.tensor("blk.0.ffn_up_exps.weight").unwrap();
            let dt = g.tensor("blk.0.ffn_down_exps.weight").unwrap();
            let mut m = CpuMoe::new(1, 2560, 640);
            let mut rng = Lcg(1);
            let x: Vec<f32> = (0..128 * 2560).map(|_| rng.f()).collect();
            let jobs = vec![MoeJob { gate: g.slice(gt, 3), up: g.slice(ut, 3), down: g.slice(dt, 3), ty_gu: gt.ty, ty_down: dt.ty, rows: (0..128).collect() }];
            let t0 = Instant::now();
            for _ in 0..10 { m.run_tiled(&jobs, &x)?; }
            println!("1 thread: {:.0} us per expert x 128 rows", t0.elapsed().as_secs_f64() / 10.0 * 1e6);
            Ok(())
        }
        "kern" => {
            // single-thread kernel loop on one expert slice (cache-resident): kern <tensor> <B> <seconds>
            let t = g.tensor(&a[3]).unwrap();
            let b: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(1);
            let secs: f64 = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(3.0);
            let w = g.slice(t, 0).to_vec();
            let (kk, rows, rb) = (t.ne0(), t.dims[1] as usize, t.row_bytes());
            let mut rng = Lcg(1);
            let x: Vec<f32> = (0..b * kk).map(|_| rng.f()).collect();
            let q = Q8Rows::quantize(&x, b, kk);
            let mut y = vec![0f32; b * rows];
            let t0 = Instant::now();
            let mut n = 0usize;
            while t0.elapsed().as_secs_f64() < secs {
                unsafe { cpu::gemv_rows(t.ty, w.as_ptr(), rb, 0..rows, &q, y.as_mut_ptr(), rows) };
                n += 1;
            }
            let el = t0.elapsed().as_secs_f64();
            println!("{} B{b}: {:.2} GB/s ({n} passes, y0 {})", t.ty.name(), (n * w.len()) as f64 / el / 1e9, y[0]);
            Ok(())
        }
        "pool" => {
            fe_ggq::cpu::pool::pin_current(0);
            for cl in &a[3..] {
                let cs = fe_ggq::cpu::pool::parse_cpus(cl);
                let p = Pool::pinned(&cs, std::time::Duration::from_millis(20));
                let cnt = std::sync::atomic::AtomicUsize::new(0);
                let mut ts = vec![];
                for _ in 0..2000 {
                    let t0 = Instant::now();
                    p.run_dyn(|_, _| { cnt.fetch_add(1, std::sync::atomic::Ordering::Relaxed); });
                    ts.push(t0.elapsed().as_secs_f64() * 1e6);
                    let tw = Instant::now();
                    while tw.elapsed().as_micros() < 100 {}
                }
                let mut ts2 = vec![];
                for _ in 0..2000 {
                    let t0 = Instant::now();
                    p.run(|_, _| { cnt.fetch_add(1, std::sync::atomic::Ordering::Relaxed); });
                    ts2.push(t0.elapsed().as_secs_f64() * 1e6);
                    let tw = Instant::now();
                    while tw.elapsed().as_micros() < 100 {}
                }
                ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ts2.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!("pool {cl}: run_dyn empty median {:.1} us p90 {:.1}; run (all threads) median {:.1} us p90 {:.1}; avg participants {:.1}",
                    ts[1000], ts[1800], ts2[1000], ts2[1800], cnt.load(std::sync::atomic::Ordering::Relaxed) as f64 / 4000.0);
            }
            Ok(())
        }
        m => anyhow::bail!("mode {m}"),
    }
}

fn ref_dot(ty: GgmlType, row: &[u8], a: &Q8Rows, b: usize) -> f64 {
    let (k, nb) = (a.k, a.k / 32);
    refq::dot_q8(ty, row, &a.q[b * k..(b + 1) * k], &a.d[b * nb..(b + 1) * nb], &a.s[b * nb..(b + 1) * nb])
}

fn check(g: &Gguf) -> Result<()> {
    let mut rng = Lcg(99);
    let mut seen = vec![];
    for t in &g.tensors {
        if !cpu::supported(t.ty) || seen.contains(&t.ty) {
            continue;
        }
        seen.push(t.ty);
        let k = t.ne0();
        let n = t.n_rows().min(256);
        let rb = t.row_bytes();
        let w = &g.data(t)[..rb * n];
        let mut wf = vec![0f32; n * k];
        refq::dequant(t.ty, w, &mut wf);
        let (mut e_avx, mut e_sc) = (0f64, 0f64);
        for b in 1..=8 {
            let x: Vec<f32> = (0..b * k).map(|_| rng.f()).collect();
            let q = Q8Rows::quantize(&x, b, k);
            let mut y = vec![0f32; b * n];
            let mut ys = vec![0f32; b * n];
            unsafe {
                cpu::gemv_rows(t.ty, w.as_ptr(), rb, 0..n, &q, y.as_mut_ptr(), n);
                cpu::gemv_rows_scalar(t.ty, w.as_ptr(), rb, 0..n, &q, ys.as_mut_ptr(), n);
            }
            for bb in 0..b {
                let mut den = 0f64;
                let (mut m1, mut m2) = (0f64, 0f64);
                for r in 0..n {
                    let rv = ref_dot(t.ty, &w[r * rb..(r + 1) * rb], &q, bb);
                    let _ = &wf;
                    den += rv * rv;
                    m1 = m1.max((y[bb * n + r] as f64 - rv).abs());
                    m2 = m2.max((ys[bb * n + r] as f64 - rv).abs());
                }
                let rms = (den / n as f64).sqrt();
                e_avx = e_avx.max(m1 / rms);
                e_sc = e_sc.max(m2 / rms);
            }
        }
        println!("{:8} {:>32} [{k} x {n}] B=1..8 max|err|/rms: avx2 {e_avx:.2e}, scalar {e_sc:.2e}", t.ty.name(), t.name);
    }
    // full expert FFN vs host reference, a few layers with different types
    let mut moe = CpuMoe::new(4, 2560, 640);
    for layer in [0usize, 1, 17, 48] {
        let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
        let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
        let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
        let ntok = 5;
        let x: Vec<f32> = (0..ntok * 2560).map(|_| rng.f()).collect();
        let jobs: Vec<MoeJob> = [(3usize, vec![0usize, 2, 4]), (77, vec![1]), (200, vec![0, 1, 2, 3, 4])]
            .into_iter()
            .map(|(e, rows)| MoeJob { gate: g.slice(gt, e), up: g.slice(ut, e), down: g.slice(dt, e), ty_gu: gt.ty,
                ty_down: dt.ty, rows })
            .collect();
        moe.run(&jobs, &x)?;
        let mut worst = 0f64;
        let mut worst_q = 0f64;
        for (j, job) in jobs.iter().enumerate() {
            let mut gw = vec![0f32; 640 * 2560];
            let mut uw = vec![0f32; 640 * 2560];
            let mut dw = vec![0f32; 2560 * 640];
            refq::dequant(gt.ty, job.gate, &mut gw);
            refq::dequant(ut.ty, job.up, &mut uw);
            refq::dequant(dt.ty, job.down, &mut dw);
            for (bi, &r) in job.rows.iter().enumerate() {
                let xr = &x[r * 2560..(r + 1) * 2560];
                // emulated-quantization reference: q8(x) -> f32 gate/up -> SwiGLU -> q8(h) -> down
                let xq = Q8Rows::quantize(xr, 1, 2560);
                let xqf: Vec<f64> = (0..2560).map(|c| xq.q[c] as f64 * xq.d[c / 32] as f64).collect();
                let hq: Vec<f32> = (0..640).map(|i| {
                    let rbg = gt.row_bytes();
                    let gv = ref_dot(gt.ty, &job.gate[i * rbg..(i + 1) * rbg], &xq, 0);
                    let uv = ref_dot(ut.ty, &job.up[i * rbg..(i + 1) * rbg], &xq, 0);
                    let _ = &xqf;
                    (gv / (1.0 + (-gv).exp()) * uv) as f32
                }).collect();
                let hqq = Q8Rows::quantize(&hq, 1, 640);
                let yq: Vec<f64> = (0..2560).map(|o| (0..640).map(|i| dw[o * 640 + i] as f64 * hqq.q[i] as f64 * hqq.d[i / 32] as f64).sum()).collect();
                let hh: Vec<f64> = (0..640).map(|i| {
                    let gv: f64 = (0..2560).map(|c| gw[i * 2560 + c] as f64 * xr[c] as f64).sum();
                    let uv: f64 = (0..2560).map(|c| uw[i * 2560 + c] as f64 * xr[c] as f64).sum();
                    gv / (1.0 + (-gv).exp()) * uv
                }).collect();
                let yr: Vec<f64> = (0..2560).map(|o| (0..640).map(|i| dw[o * 640 + i] as f64 * hh[i]).sum()).collect();
                let rms = (yr.iter().map(|v| v * v).sum::<f64>() / 2560.0).sqrt();
                let got = &moe.out(j)[bi * 2560..(bi + 1) * 2560];
                let e = got.iter().zip(&yr).map(|(a, b)| (*a as f64 - b).abs()).fold(0f64, f64::max) / rms;
                worst = worst.max(e);
                let e2 = got.iter().zip(&yq).map(|(a, b)| (*a as f64 - b).abs()).fold(0f64, f64::max) / rms;
                worst_q = worst_q.max(e2);
            }
        }
        println!("CpuMoe layer {layer} ({} / {}): max|err|/rms vs f64 unquantized-activation reference {worst:.2e}; vs emulated-q8 reference {worst_q:.2e}",
            gt.ty.name(), dt.ty.name());
    }
    Ok(())
}

fn bench(g: &Gguf, threads: &[usize]) -> Result<()> {
    let threads = if threads.is_empty() { vec![1, 2, 4, 8, 12, 16, 24] } else { threads.to_vec() };
    // DRAM ceiling: plain read of 1 GiB
    let big = vec![1u64; 1 << 27];
    for &t in &threads {
        let pool = Pool::new(t, std::time::Duration::from_millis(5));
        let chunk = big.len() / t;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = Instant::now();
            let sums: Vec<std::sync::atomic::AtomicU64> = (0..t).map(|_| Default::default()).collect();
            pool.run(|i, _| {
                let s: u64 = big[i * chunk..(i + 1) * chunk].iter().fold(0u64, |a, &v| a.wrapping_add(v));
                sums[i].store(s, std::sync::atomic::Ordering::Relaxed);
            });
            best = best.min(t0.elapsed().as_secs_f64());
        }
        println!("DRAM read, {t:2} threads: {:.1} GB/s", (big.len() * 8) as f64 / best / 1e9);
    }
    let cases = [("blk.1.ffn_gate_exps.weight", 2560), ("blk.0.ffn_gate_exps.weight", 2560),
        ("blk.17.ffn_gate_exps.weight", 2560), ("blk.48.ffn_gate_exps.weight", 2560),
        ("blk.0.ffn_down_exps.weight", 640), ("blk.1.ffn_down_exps.weight", 640)];
    let mut rng = Lcg(5);
    let x: Vec<f32> = (0..8 * 2560).map(|_| rng.f()).collect();
    for (name, k) in cases {
        let Some(t) = g.tensor(name) else { continue };
        let ne = std::env::var("FE_BENCH_EXPERTS").ok().and_then(|v| v.parse().ok()).unwrap_or(t.dims[2] as usize); // default all experts: 120-470 MB, well past the 36 MB L3
        let sb = t.slice_bytes();
        let w: Vec<u8> = g.data(t)[..sb * ne].to_vec(); // anonymous copy: >= L3
        let rows = t.dims[1] as usize * ne;
        let rb = t.row_bytes();
        let mut line = format!("{:8} {name:28} {:5.0} MB:", t.ty.name(), w.len() as f64 / 1e6);
        for &th in &threads {
            let pool = Pool::new(th, std::time::Duration::from_millis(5));
            line += &format!("\n   T{th:2}");
            for b in [1usize, 2, 4, 8] {
                let q = Q8Rows::quantize(&x, b, k);
                let mut y = vec![0f32; b * rows];
                let yp = y.as_mut_ptr() as usize;
                let next = std::sync::atomic::AtomicUsize::new(0);
                let chunk = 64;
                let mut best = f64::MAX;
                for _ in 0..3 {
                    next.store(0, std::sync::atomic::Ordering::Relaxed);
                    let t0 = Instant::now();
                    pool.run(|_, _| loop {
                        let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if c * chunk >= rows {
                            break;
                        }
                        let r1 = ((c + 1) * chunk).min(rows);
                        unsafe { cpu::gemv_rows(t.ty, w.as_ptr(), rb, c * chunk..r1, &q, yp as *mut f32, rows) };
                    });
                    best = best.min(t0.elapsed().as_secs_f64());
                }
                line += &format!("  B{b} {:5.1} GB/s", w.len() as f64 / best / 1e9);
            }
        }
        println!("{line}");
    }
    Ok(())
}

fn moe(g: &Gguf, threads: &[usize]) -> Result<()> {
    let threads = if threads.is_empty() { vec![8, 12, 16, 24] } else { threads.to_vec() };
    let mut rng = Lcg(11);
    for layer in [1usize, 0, 17] {
        let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
        let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
        let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
        // copies of 40 experts' slices (anonymous memory, like page-cache-resident mmap pages)
        let ne = 40;
        let copy = |t: &fe_gguf::TensorInfo| -> Vec<Vec<u8>> { (0..ne).map(|e| g.slice(t, e).to_vec()).collect() };
        let (gc, uc, dc) = (copy(gt), copy(ut), copy(dt));
        let per_expert = gc[0].len() + uc[0].len() + dc[0].len();
        for &th in &threads {
            let mut m = CpuMoe::new(th, 2560, 640);
            let mut line = format!("layer {layer:2} {}/{} T{th:2}:", gt.ty.name(), dt.ty.name());
            for b in [1usize, 2, 4, 8] {
                let x: Vec<f32> = (0..b * 2560).map(|_| rng.f()).collect();
                let mut best = f64::MAX;
                for it in 0..4 {
                    // 10 experts per call, a different set each time (cold in L2/L3)
                    let jobs: Vec<MoeJob> = (0..10).map(|i| {
                        let e = (it * 10 + i) % ne;
                        MoeJob { gate: &gc[e], up: &uc[e], down: &dc[e], ty_gu: gt.ty, ty_down: dt.ty, rows: (0..b).collect() }
                    }).collect();
                    let t0 = Instant::now();
                    m.run(&jobs, &x)?;
                    best = best.min(t0.elapsed().as_secs_f64());
                }
                line += &format!("  B{b} {:.2} ms ({:.1} GB/s)", best * 1e3, 10.0 * per_expert as f64 / best / 1e9);
            }
            println!("{line}");
        }
    }
    let _ = GgmlType::F32;
    Ok(())
}

/// lat <cpus...>: per-layer latency of CpuMoe for E = 1..4 experts x B = 1, 2, 4 rows, cold experts (a pool of 120
/// experts per tensor, 2.2 MB each, > L3), workers pinned to each CPU list given (e.g. "1-7" "1-7,8-23"); the
/// caller is pinned to the first CPU minus one if FE_LAT_PIN_CALLER is set to a CPU. Median and p90 over 200 calls.
fn lat(g: &Gguf, cpus: &[String]) -> Result<()> {
    if let Ok(c) = std::env::var("FE_LAT_PIN_CALLER") {
        if let Ok(c) = c.parse::<usize>() {
            fe_ggq::cpu::pool::pin_current(c);
        }
    }
    let gap_us: u64 = std::env::var("FE_LAT_GAP_US").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let mut rng = Lcg(3);
    for layer in [0usize, 1, 17] {
        let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
        let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
        let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
        let ne = 120;
        let copy = |t: &fe_gguf::TensorInfo| -> Vec<Vec<u8>> { (0..ne).map(|e| g.slice(t, e).to_vec()).collect() };
        let (gc, uc, dc) = (copy(gt), copy(ut), copy(dt));
        let per_e = gc[0].len() + uc[0].len() + dc[0].len();
        for cl in cpus {
            let cs = fe_ggq::cpu::pool::parse_cpus(cl);
            let mut m = CpuMoe::pinned(&cs, std::time::Duration::from_millis(20), 2560, 640);
            if let Some(v) = std::env::var("FE_GU_CHUNK").ok().and_then(|v| v.parse().ok()) {
                m.gu_chunk = v;
            }
            if let Some(v) = std::env::var("FE_DOWN_CHUNK").ok().and_then(|v| v.parse().ok()) {
                m.down_chunk = v;
            }
            let prof = std::env::var("FE_CPU_PROF").is_ok();
            cpu::PROF.store(prof, std::sync::atomic::Ordering::Relaxed);
            let bl: Vec<usize> = std::env::var("FE_LAT_B").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or(vec![1, 2, 4]);
            for b in bl {
                let x: Vec<f32> = (0..b * 2560).map(|_| rng.f()).collect();
                let mut line = format!("layer {layer:2} {}/{} cpus {cl:10} B{b}:", gt.ty.name(), dt.ty.name());
                for e in [1usize, 2, 4] {
                    let mut ts = vec![];
                    let mut k = 0usize;
                    for p in &cpu::PROF_NS {
                        p.store(0, std::sync::atomic::Ordering::Relaxed);
                    }
                    for _ in 0..200 {
                        let jobs: Vec<MoeJob> = (0..e).map(|i| {
                            let ex = if std::env::var("FE_LAT_WARM").is_ok() { i } else { (k + i * 7) % ne };
                            MoeJob { gate: &gc[ex], up: &uc[ex], down: &dc[ex], ty_gu: gt.ty, ty_down: dt.ty, rows: (0..b).collect() }
                        }).collect();
                        k += 13;
                        let t0 = Instant::now();
                        m.run(&jobs, &x)?;
                        ts.push(t0.elapsed().as_secs_f64() * 1e6);
                        let tw = Instant::now();
                        while tw.elapsed().as_micros() < gap_us as u128 {
                            std::hint::spin_loop();
                        }
                    }
                    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let med = ts[ts.len() / 2];
                    line += &format!("  E{e} {:5.0} us ({:3.0}/exp, {:3.0} GB/s, p90 {:4.0})", med, med / e as f64,
                        (e * per_e) as f64 / med / 1e3, ts[ts.len() * 9 / 10]);
                    if prof {
                        let tot: f64 = ts.iter().sum::<f64>();
                        let p: Vec<f64> = cpu::PROF_NS.iter().map(|v| v.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3).collect();
                        let np = p[5] * 1e3;
                        line += &format!(" [thread-us per wall-us: gu {:.1} wait {:.1} down {:.1} of {}; {:.1} threads/call, first claim {:.1} us, last exit {:.1} us; prep {:.1} us, pool {:.1} us]",
                            p[0] / tot, p[1] / tot, p[2] / tot, cs.len() + 1, np / ts.len() as f64, p[3] / np, p[4] / np,
                            p[6] / ts.len() as f64, p[7] / ts.len() as f64);
                    }
                }
                println!("{line}");
            }
        }
    }
    Ok(())
}

/// tiled [cpus]: the many-row CpuMoe path. Correctness vs the emulated-q8 f64 reference (20 rows, all 4 layer
/// type pairs), then time per expert for M = 8, 32, 128 rows (10 experts per call, cold), rows/s, weight GB/s
/// equivalent, and the crossover against streaming the expert over PCIe (6.5 GB/s x4 and 16 GB/s).
fn tiled(g: &Gguf, args: &[String]) -> Result<()> {
    let cpus = fe_ggq::cpu::pool::parse_cpus(args.first().map(|s| s.as_str()).unwrap_or("2-7,8-23"));
    fe_ggq::cpu::pool::pin_current(0);
    let mut m = CpuMoe::pinned(&cpus, std::time::Duration::from_millis(20), 2560, 640);
    let mut rng = Lcg(77);
    for layer in [0usize, 1, 17, 48] {
        let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
        let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
        let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
        // correctness: one expert, 21 rows
        let nr = 21;
        let x: Vec<f32> = (0..nr * 2560).map(|_| rng.f()).collect();
        let e = 5;
        let jobs = vec![MoeJob { gate: g.slice(gt, e), up: g.slice(ut, e), down: g.slice(dt, e), ty_gu: gt.ty, ty_down: dt.ty,
            rows: (0..nr).collect() }];
        m.run(&jobs, &x)?;
        let got = m.out(0).to_vec();
        let (rbg, rbd) = (gt.row_bytes(), dt.row_bytes());
        let mut worst = 0f64;
        for r in [0usize, 7, 20] {
            let xq = Q8Rows::quantize(&x[r * 2560..(r + 1) * 2560], 1, 2560);
            let h: Vec<f32> = (0..640).map(|i| {
                let gv = ref_dot(gt.ty, &g.slice(gt, e)[i * rbg..(i + 1) * rbg], &xq, 0);
                let uv = ref_dot(ut.ty, &g.slice(ut, e)[i * rbg..(i + 1) * rbg], &xq, 0);
                (gv / (1.0 + (-gv).exp()) * uv) as f32
            }).collect();
            let hq = Q8Rows::quantize(&h, 1, 640);
            let yr: Vec<f64> = (0..2560).map(|o| ref_dot(dt.ty, &g.slice(dt, e)[o * rbd..(o + 1) * rbd], &hq, 0)).collect();
            let rms = (yr.iter().map(|v| v * v).sum::<f64>() / 2560.0).sqrt();
            let mx = (0..2560).map(|o| (got[r * 2560 + o] as f64 - yr[o]).abs()).fold(0f64, f64::max);
            worst = worst.max(mx / rms);
        }
        let per_e = gt.slice_bytes() + ut.slice_bytes() + dt.slice_bytes();
        let mut line = format!("layer {layer:2} {}/{} ({:.2} MB/expert): tiled vs emulated-q8 ref {worst:.1e} |", gt.ty.name(),
            dt.ty.name(), per_e as f64 / 1e6);
        // timing: 10 experts x M rows per call, 4 different expert sets (cold)
        let ne = 40;
        let copy = |t: &fe_gguf::TensorInfo| -> Vec<Vec<u8>> { (0..ne).map(|e| g.slice(t, e).to_vec()).collect() };
        let (gc, uc, dc) = (copy(gt), copy(ut), copy(dt));
        for mm in [8usize, 32, 128] {
            let ntok = 256;
            let x: Vec<f32> = (0..ntok * 2560).map(|_| rng.f()).collect();
            let mut best = f64::MAX;
            for it in 0..4 {
                let jobs: Vec<MoeJob> = (0..10).map(|i| {
                    let e = (it * 10 + i) % ne;
                    MoeJob { gate: &gc[e], up: &uc[e], down: &dc[e], ty_gu: gt.ty, ty_down: dt.ty, rows: (0..mm).map(|r| (r * 7 + i) % ntok).collect() }
                }).collect();
                let t0 = Instant::now();
                m.run_tiled(&jobs, &x)?;
                best = best.min(t0.elapsed().as_secs_f64());
            }
            let per = best / 10.0 * 1e6; // us per expert
            let x4 = per_e as f64 / 6.5e3; // us to stream it over Gen4 x4 (6.5 GB/s)
            let fast = per_e as f64 / 16.0e3;
            line += &format!("  M{mm}: {per:.0} us/exp ({:.0} rows/ms, {:.1} GB/s eq; x4 {x4:.0} us, x16-ish {fast:.0} us)",
                mm as f64 / per * 1e3, per_e as f64 / per / 1e3);
        }
        println!("{line}");
    }
    Ok(())
}

/// small <cpu> [layers]: single-thread cost of one expert (gate + up + down, warm in cache) at B = 1..5 rows,
/// GEMV kernels (decode per row group, B-row dot) vs decode-once + 4x2 micro-kernel, pinned to `cpu`.
fn small(g: &Gguf, args: &[String]) -> Result<()> {
    let cpu: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(8);
    fe_ggq::cpu::pool::pin_current(cpu);
    let layers: Vec<usize> = if args.len() > 1 { args[1..].iter().filter_map(|v| v.parse().ok()).collect() } else { vec![0, 1, 17] };
    let mut rng = Lcg(5);
    for layer in layers {
        let gt = g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight")).unwrap();
        let ut = g.tensor(&format!("blk.{layer}.ffn_up_exps.weight")).unwrap();
        let dt = g.tensor(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap();
        let (gw, uw, dw) = (g.slice(gt, 2).to_vec(), g.slice(ut, 2).to_vec(), g.slice(dt, 2).to_vec());
        let (rbg, rbd) = (gt.row_bytes(), dt.row_bytes());
        let mut line = format!("layer {layer:2} {}/{} cpu {cpu}:", gt.ty.name(), dt.ty.name());
        for b in 1..=5usize {
            let x: Vec<f32> = (0..b * 2560).map(|_| rng.f()).collect();
            let h: Vec<f32> = (0..b * 640).map(|_| rng.f()).collect();
            let xq = Q8Rows::quantize(&x, b, 2560);
            let hq = Q8Rows::quantize(&h, b, 640);
            let mut y = vec![0f32; b * 2560];
            let reps = 200;
            // GEMV path
            let t0 = Instant::now();
            for _ in 0..reps {
                unsafe {
                    cpu::gemv_rows(gt.ty, gw.as_ptr(), rbg, 0..640, &xq, y.as_mut_ptr(), 640);
                    cpu::gemv_rows(ut.ty, uw.as_ptr(), rbg, 0..640, &xq, y.as_mut_ptr(), 640);
                    cpu::gemv_rows(dt.ty, dw.as_ptr(), rbd, 0..2560, &hq, y.as_mut_ptr(), 2560);
                }
            }
            let tg = t0.elapsed().as_secs_f64() / reps as f64 * 1e6;
            // decode-once + micro
            let t0 = Instant::now();
            for _ in 0..reps {
                unsafe { cpu::small_expert_bench(gt.ty, dt.ty, &gw, &uw, &dw, &xq, &hq, &mut y) };
            }
            let tm = t0.elapsed().as_secs_f64() / reps as f64 * 1e6;
            line += &format!("  B{b} gemv {tg:4.0} us / micro {tm:4.0} us");
        }
        println!("{line}");
    }
    Ok(())
}
