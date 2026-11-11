//! M0 checks: launch the k-core proof kernels and compare against host math.
use anyhow::{ensure, Result};
use fe_cuda::{DeviceBuffer, Module, Stream};

fn bf16_bits(f: f32) -> u16 {
    // round-to-nearest-even
    let b = f.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn bf16_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

pub fn run(m: &Module, s: &Stream) -> Result<()> {
    if std::env::var("FE_SELFTEST_ONLY").is_ok_and(|v| v == "kvq") {
        return kvq_selftest(m, s);
    }
    // FE_KVQ_BENCH_ONLY=<keys>: only the attention kernel bench (kvq_bench)
    if let Some(keys) = std::env::var("FE_KVQ_BENCH_ONLY").ok().and_then(|v| v.parse::<usize>().ok()) {
        return kvq_bench(m, s, keys.max(4096));
    }
    // shared memory (static + dynamic) + block reduction
    {
        let f = m.function("fe_t_smem")?;
        let blocks = 4u32;
        let n = blocks * 256;
        let inp: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.25 - 3.0).collect();
        let din = DeviceBuffer::from_slice(&inp)?;
        let dout = DeviceBuffer::<f32>::zeroed((n + blocks) as usize)?;
        unsafe { s.launch(&f, blocks, 256, 1024 + 128, (dout.ptr(), din.ptr(), n))? };
        s.sync()?;
        let out = dout.to_vec()?;
        for b in 0..blocks as usize {
            for t in 0..256 {
                let want = 3.0 * inp[b * 256 + 255 - t];
                ensure!(out[b * 256 + t] == want, "smem: block {b} t {t}: {} != {want}", out[b * 256 + t]);
            }
            let sum: f32 = inp[b * 256..(b + 1) * 256].iter().sum();
            ensure!((out[n as usize + b] - sum).abs() < 1e-3, "block_sum {b}: {} != {sum}", out[n as usize + b]);
        }
        println!("selftest smem: static+dynamic shared, ld/st.shared, block_sum OK (regs {}, static smem {} B)",
            f.num_regs()?, f.static_smem()?);
    }
    // cp.async + ldmatrix + mma.sync bf16 m16n8k16
    {
        let f = m.function("fe_t_mma")?;
        let a: Vec<u16> = (0..256).map(|i| bf16_bits(((i * 7) % 13) as f32 * 0.125 - 0.5)).collect();
        let w: Vec<u16> = (0..128).map(|i| bf16_bits(((i * 5) % 11) as f32 * 0.25 - 1.0)).collect();
        let (da, dw) = (DeviceBuffer::from_slice(&a)?, DeviceBuffer::from_slice(&w)?);
        let dc = DeviceBuffer::<f32>::zeroed(128)?;
        unsafe { s.launch(&f, 1, 32, 0, (dc.ptr(), da.ptr(), dw.ptr()))? };
        s.sync()?;
        let c = dc.to_vec()?;
        for r in 0..16 {
            for n in 0..8 {
                let want: f32 = (0..16).map(|k| bf16_f32(a[r * 16 + k]) * bf16_f32(w[n * 16 + k])).sum();
                ensure!((c[r * 8 + n] - want).abs() < 1e-4, "mma: C[{r}][{n}] = {} want {want}", c[r * 8 + n]);
            }
        }
        println!("selftest mma: cp.async -> ldmatrix.x4/x2 -> mma.sync.m16n8k16 bf16 OK");
    }
    // fast math
    {
        let f = m.function("fe_t_math")?;
        let n = 1000u32;
        let inp: Vec<f32> = (0..n).map(|i| i as f32 * 0.02 - 10.0).collect();
        let din = DeviceBuffer::from_slice(&inp)?;
        let dout = DeviceBuffer::<f32>::zeroed(4 * n as usize)?;
        unsafe { s.launch(&f, n.div_ceil(256), 256, 0, (dout.ptr(), din.ptr(), n))? };
        s.sync()?;
        let o = dout.to_vec()?;
        let mut worst = [0f32; 4];
        for (i, &x) in inp.iter().enumerate() {
            let want = [x.exp(), x / (1.0 + (-x).exp()), 1.0 / (x.abs() + 1.0).sqrt(), bf16_f32(bf16_bits(x))];
            for j in 0..4 {
                let rel = (o[i * 4 + j] - want[j]).abs() / want[j].abs().max(1e-6);
                worst[j] = worst[j].max(rel);
            }
        }
        ensure!(worst[0] < 1e-5 && worst[1] < 1e-5 && worst[2] < 1e-5 && worst[3] == 0.0, "math rel err {worst:?}");
        println!("selftest math: exp/silu/rsqrt/bf16 OK (max rel err {worst:?})");

        // CUDA graph: capture the same launch, replay it, and time 100 replays.
        s.memset_d32(dout.ptr(), 0, 4 * n as usize)?;
        s.begin_capture()?;
        unsafe { s.launch(&f, n.div_ceil(256), 256, 0, (dout.ptr(), din.ptr(), n))? };
        let g = s.end_capture()?;
        g.launch(s)?;
        s.sync()?;
        ensure!(dout.to_vec()? == o, "graph replay output differs");
        let ms = s.time(|s| {
            for _ in 0..100 {
                g.launch(s)?;
            }
            Ok(())
        })?;
        println!("selftest graph: capture/instantiate/replay OK ({:.2} us per replay)", ms * 10.0);
    }
    // bf16 GEMM with M/N edges and a residual
    {
        let f = m.function("fe_gemm_bf16")?;
        for &(mm, nn, kk) in &[(37usize, 200usize, 96usize), (300, 384, 512), (1, 130, 64)] {
            let mut seed = 12345u32;
            let mut rnd = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
            };
            let a: Vec<u16> = (0..mm * kk).map(|_| bf16_bits(rnd())).collect();
            let w: Vec<u16> = (0..nn * kk).map(|_| bf16_bits(rnd())).collect();
            let r: Vec<u16> = (0..mm * nn).map(|_| bf16_bits(rnd())).collect();
            let (da, dw, dr) = (DeviceBuffer::from_slice(&a)?, DeviceBuffer::from_slice(&w)?, DeviceBuffer::from_slice(&r)?);
            let dc = DeviceBuffer::<u16>::zeroed(mm * nn)?;
            let grid = ((mm as u32).div_ceil(128), (nn as u32).div_ceil(128));
            unsafe {
                s.launch(&f, grid, 256, 49152, (dc.ptr(), da.ptr(), dw.ptr(), dr.ptr(), mm as u32, nn as u32, kk as u32,
                    kk as u32, nn as u32))?
            };
            s.sync()?;
            let c = dc.to_vec()?;
            let mut worst = 0f32;
            for i in 0..mm {
                for j in 0..nn {
                    let dot: f32 = (0..kk).map(|q| bf16_f32(a[i * kk + q]) * bf16_f32(w[j * kk + q])).sum();
                    let want = bf16_f32(bf16_bits(dot)) + bf16_f32(r[i * nn + j]);
                    worst = worst.max((bf16_f32(c[i * nn + j]) - want).abs() / (want.abs() + 0.05));
                }
            }
            ensure!(worst < 0.02, "gemm {mm}x{nn}x{kk}: worst rel err {worst}");
            println!("selftest gemm {mm}x{nn}x{kk} (+res): OK (worst rel {worst:.4})");
        }
    }
    // flash prefill attention vs the split-K path on the same cache
    {
        let fl = m.function("fe_flash_prefill")?;
        fl.set_max_dynamic_smem(98304)?;
        let sp = m.function("fe_attn_split")?;
        let cb = m.function("fe_attn_combine")?;
        let (ctx, n, pos0) = (300usize, 100usize, 200usize);
        let mut seed = 99u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
        };
        let q: Vec<u16> = (0..n * 24 * 256).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let kc: Vec<u16> = (0..ctx * 1024).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let vc: Vec<u16> = (0..ctx * 1024).map(|_| bf16_bits(rnd())).collect();
        let qg: Vec<u16> = (0..n * 12288).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let (dq, dk, dv, dg) = (DeviceBuffer::from_slice(&q)?, DeviceBuffer::from_slice(&kc)?, DeviceBuffer::from_slice(&vc)?,
            DeviceBuffer::from_slice(&qg)?);
        let o1 = DeviceBuffer::<u16>::zeroed(n * 6144)?;
        let o2 = DeviceBuffer::<u16>::zeroed(n * 6144)?;
        let scale = (1.0f32 / 16.0) * std::f32::consts::LOG2_E;
        let stride = (ctx * 1024) as u64;
        unsafe {
            s.launch(&fl, ((n as u32).div_ceil(64), 24), 128, 98304, (o1.ptr(), dq.ptr(), dg.ptr(), 12288u32, dk.ptr(),
                dv.ptr(), stride, 0u32, 0u32, n as u32, pos0 as u32, scale))?;
        }
        let rslot = DeviceBuffer::from_slice(&vec![0u32; n])?;
        let rpos = DeviceBuffer::from_slice(&(0..n).map(|i| (pos0 + i) as u32).collect::<Vec<_>>())?;
        let ns = 2u32;
        let po = DeviceBuffer::<f32>::zeroed(n * 4 * ns as usize * 1536)?;
        let pm = DeviceBuffer::<f32>::zeroed(n * 4 * ns as usize * 12)?;
        unsafe {
            let nobase = DeviceBuffer::from_slice(&vec![u32::MAX; n])?;
            let nomask = DeviceBuffer::from_slice(&vec![0u32; n])?;
            s.launch(&sp, (n as u32, 4, ns), 128, 4 * (1536 + 12) * 4, (po.ptr(), pm.ptr(), dq.ptr(), dk.ptr(), dv.ptr(),
                rslot.ptr(), rpos.ptr(), nobase.ptr(), nomask.ptr(), stride, scale, 0u32))?;
            s.launch(&cb, (n as u32, 24), 256, 0, (o2.ptr(), po.ptr(), pm.ptr(), dg.ptr(), 12288u32, ns, 0u32))?;
        }
        s.sync()?;
        let (a, b) = (o1.to_vec()?, o2.to_vec()?);
        let mut worst = 0f32;
        let mut wi = 0;
        for i in 0..a.len() {
            let (x, y) = (bf16_f32(a[i]), bf16_f32(b[i]));
            let e = (x - y).abs() / (y.abs() + 0.02);
            if e > worst {
                worst = e;
                wi = i;
            }
        }
        let mut bad_rows = std::collections::BTreeMap::new();
        for i in 0..a.len() {
            let (x, y) = (bf16_f32(a[i]), bf16_f32(b[i]));
            if (x - y).abs() / (y.abs() + 0.02) > 0.05 {
                *bad_rows.entry(i / 6144).or_insert(0) += 1;
            }
        }
        if !bad_rows.is_empty() {
            println!("flash bad rows (row: count of 6144): {:?}", bad_rows);
            let r = *bad_rows.keys().next().unwrap();
            println!("row {r} head0 flash {:?}", (0..8).map(|d| bf16_f32(a[r * 6144 + d])).collect::<Vec<_>>());
            println!("row {r} head0 split {:?}", (0..8).map(|d| bf16_f32(b[r * 6144 + d])).collect::<Vec<_>>());
        }
        println!("selftest flash vs split: worst rel {worst:.4} at row {} head {} dim {} ({} vs {})", wi / 6144,
            (wi % 6144) / 256, wi % 256, bf16_f32(a[wi]), bf16_f32(b[wi]));
        ensure!(worst < 0.05, "flash attention mismatch");
    }
    kvq_selftest(m, s)?;
    // FE_KVQ_BENCH=<keys>: time the attention kernels over a cache of that many keys, bf16 against the quantized forms:
    // a 2048-row prompt chunk at the end of it (flash), a 1-row decode and a 16-row verify (vk + combine)
    if let Some(keys) = std::env::var("FE_KVQ_BENCH").ok().and_then(|v| v.parse::<usize>().ok()) {
        kvq_bench(m, s, keys.max(4096))?;
    }
    // split-column GDN recurrence + separate gated norm vs the fused kernel
    {
        let f1 = m.function("fe_gdn_rec")?;
        let f2 = m.function("fe_gdn_rec2")?;
        let fnm = m.function("fe_gdn_norm")?;
        let rows = 37usize;
        let mut seed = 4242u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
        };
        let conv: Vec<u16> = (0..rows * 10240).map(|_| bf16_bits(rnd() * 2.0)).collect();
        let z: Vec<u16> = (0..rows * 6144).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let ab: Vec<u16> = (0..rows * 96).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let alog: Vec<u16> = (0..48).map(|_| bf16_bits(rnd() * 4.0 - 2.0)).collect();
        let dtb: Vec<u16> = (0..48).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let nw: Vec<u16> = (0..128).map(|_| bf16_bits(1.0 + rnd() * 0.2)).collect();
        let st0: Vec<f32> = (0..48 * 128 * 128).map(|_| rnd() * 0.1).collect();
        let (dc, dz, dab, dal, ddt, dnw) = (DeviceBuffer::from_slice(&conv)?, DeviceBuffer::from_slice(&z)?,
            DeviceBuffer::from_slice(&ab)?, DeviceBuffer::from_slice(&alog)?, DeviceBuffer::from_slice(&dtb)?,
            DeviceBuffer::from_slice(&nw)?);
        let (s1, s2) = (DeviceBuffer::from_slice(&st0)?, DeviceBuffer::from_slice(&st0)?);
        let (y1, y2, o) = (DeviceBuffer::<u16>::zeroed(rows * 6144)?, DeviceBuffer::<u16>::zeroed(rows * 6144)?,
            DeviceBuffer::<u16>::zeroed(rows * 6144)?);
        let seg = DeviceBuffer::from_slice(&[0u32, 0, rows as u32])?;
        let null = fe_cuda::DevPtr::NULL;
        unsafe {
            s.launch(&f1, (48u32, 1u32), 512, 0, (y1.ptr(), dc.ptr(), dz.ptr(), 6144u32, dab.ptr(), dab.at(48), 96u32,
                dal.ptr(), ddt.ptr(), dnw.ptr(), s1.ptr(), seg.at(0), seg.at(1), seg.at(2), 1e-6f32, null, 0u32))?;
            s.launch(&f2, (192u32, 1u32), 512, 0, (o.ptr(), dc.ptr(), dab.ptr(), dab.at(48), 96u32, dal.ptr(), ddt.ptr(),
                s2.ptr(), seg.at(0), seg.at(1), seg.at(2), null, 0u32, null, null, null))?;
            s.launch(&fnm, (rows as u32, 2u32), 24 * 32, 0, (y2.ptr(), o.ptr(), dz.ptr(), 6144u32, dnw.ptr(), 1e-6f32))?;
        }
        s.sync()?;
        let (a, b) = (y1.to_vec()?, y2.to_vec()?);
        let mut wy = 0f32;
        for i in 0..a.len() {
            wy = wy.max((bf16_f32(a[i]) - bf16_f32(b[i])).abs() / (bf16_f32(a[i]).abs() + 0.05));
        }
        let (sa, sb) = (s1.to_vec()?, s2.to_vec()?);
        let mut ws = 0f32;
        for i in 0..sa.len() {
            ws = ws.max((sa[i] - sb[i]).abs() / (sa[i].abs() + 1e-2));
        }
        println!("selftest gdn rec2+norm vs fused: worst rel y {wy:.4}, state {ws:.2e}");
        ensure!(wy < 0.05 && ws < 1e-3, "gdn split kernel mismatch");
    }
    // fused SwiGLU GEMM (interleaved gate/up) vs GEMM + silu_mul
    {
        let (fg, fs, fi, fm) = (m.function("fe_gemm_bf16")?, m.function("fe_gemm_swiglu")?, m.function("fe_interleave_gu")?,
            m.function("fe_silu_mul")?);
        let (mm, inter, kk) = (200usize, 192usize, 256usize);
        let mut seed = 777u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
        };
        let a: Vec<u16> = (0..mm * kk).map(|_| bf16_bits(rnd())).collect();
        let w: Vec<u16> = (0..2 * inter * kk).map(|_| bf16_bits(rnd() * 0.5)).collect(); // [gate; up]
        let (da, dw) = (DeviceBuffer::from_slice(&a)?, DeviceBuffer::from_slice(&w)?);
        let wi = DeviceBuffer::<u16>::zeroed(2 * inter * kk)?;
        let gu = DeviceBuffer::<u16>::zeroed(mm * 2 * inter)?;
        let (act1, act2) = (DeviceBuffer::<u16>::zeroed(mm * inter)?, DeviceBuffer::<u16>::zeroed(mm * inter)?);
        let null = fe_cuda::DevPtr::NULL;
        unsafe {
            s.launch(&fi, (2 * inter) as u32, 256, 0, (wi.ptr(), dw.ptr(), dw.at(inter * kk), kk as u32))?;
            s.launch(&fs, ((mm as u32).div_ceil(128), (2 * inter as u32).div_ceil(128)), 256, 49152,
                (act1.ptr(), da.ptr(), wi.ptr(), mm as u32, 2 * inter as u32, kk as u32, kk as u32))?;
            s.launch(&fg, ((mm as u32).div_ceil(128), (2 * inter as u32).div_ceil(128)), 256, 49152,
                (gu.ptr(), da.ptr(), dw.ptr(), null, mm as u32, 2 * inter as u32, kk as u32, kk as u32, 2 * inter as u32))?;
            s.launch(&fm, ((inter as u32 / 2).div_ceil(256), mm as u32), 256, 0,
                (act2.ptr(), gu.ptr(), gu.at(inter), 2 * inter as u32, inter as u32))?;
        }
        s.sync()?;
        let (x, y) = (act1.to_vec()?, act2.to_vec()?);
        let mut worst = 0f32;
        for i in 0..x.len() {
            worst = worst.max((bf16_f32(x[i]) - bf16_f32(y[i])).abs() / (bf16_f32(y[i]).abs() + 0.02));
        }
        println!("selftest swiglu GEMM vs GEMM + silu_mul: worst rel {worst:.4}");
        ensure!(worst < 0.02, "fused swiglu mismatch");
    }
    // verify-tree attention: each tree row sees the cache before the tree plus its ancestors (CPU reference)
    {
        let sp = m.function("fe_attn_split")?;
        let cb = m.function("fe_attn_combine")?;
        let (base, nt) = (100usize, 16usize);
        let ctx = base + nt;
        let mut seed = 2024u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
        };
        // best-first-like tree: parent of row i in [max(0, i-4), i)
        let mut par = vec![-1i32; nt];
        for i in 1..nt {
            let lo = i.saturating_sub(4);
            par[i] = (lo + ((rnd() + 0.5) * 1000.0) as usize % (i - lo)) as i32;
        }
        let (mut mask, mut depth) = (vec![0u32; nt], vec![0usize; nt]);
        for i in 0..nt {
            mask[i] = if i == 0 { 1 } else { mask[par[i] as usize] | (1 << i) };
            depth[i] = if i == 0 { 0 } else { depth[par[i] as usize] + 1 };
        }
        let q: Vec<u16> = (0..nt * 24 * 256).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let kc: Vec<u16> = (0..ctx * 1024).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let vc: Vec<u16> = (0..ctx * 1024).map(|_| bf16_bits(rnd())).collect();
        // gates at +30: sigmoid rounds to 1.0 in bf16, so the output is the plain attention
        let qg: Vec<u16> = (0..nt * 12288).map(|i| if (i % 512) >= 256 { bf16_bits(30.0) } else { 0 }).collect();
        let (dq, dk, dv, dg) = (DeviceBuffer::from_slice(&q)?, DeviceBuffer::from_slice(&kc)?, DeviceBuffer::from_slice(&vc)?,
            DeviceBuffer::from_slice(&qg)?);
        let rslot = DeviceBuffer::from_slice(&vec![0u32; nt])?;
        let rpos = DeviceBuffer::from_slice(&(0..nt).map(|i| (base + depth[i]) as u32).collect::<Vec<_>>())?;
        let rtbase = DeviceBuffer::from_slice(&vec![base as u32; nt])?;
        let rtmask = DeviceBuffer::from_slice(&mask)?;
        // more splits than any row's own count: the extra ones stay empty
        let ns = 32u32;
        let po = DeviceBuffer::<f32>::zeroed(nt * 4 * ns as usize * 1536)?;
        let pm = DeviceBuffer::<f32>::zeroed(nt * 4 * ns as usize * 12)?;
        let out = DeviceBuffer::<u16>::zeroed(nt * 6144)?;
        let scale = (1.0f32 / 16.0) * std::f32::consts::LOG2_E;
        unsafe {
            s.launch(&sp, (nt as u32, 4, ns), 128, 4 * (1536 + 12) * 4, (po.ptr(), pm.ptr(), dq.ptr(), dk.ptr(), dv.ptr(),
                rslot.ptr(), rpos.ptr(), rtbase.ptr(), rtmask.ptr(), (ctx * 1024) as u64, scale, 0u32))?;
            s.launch(&cb, (nt as u32, 24), 256, 0, (out.ptr(), po.ptr(), pm.ptr(), dg.ptr(), 12288u32, ns, 0u32))?;
        }
        s.sync()?;
        let o = out.to_vec()?;
        // bitwise against 1-row decode: the row's path gathered to consecutive positions, a 16-split grid
        let mut diff_rows = 0;
        for i in 0..nt {
            let path: Vec<usize> = (0..nt).filter(|j| mask[i] >> j & 1 == 1).collect();
            let (mut k1, mut v1) = (kc[..base * 1024].to_vec(), vc[..base * 1024].to_vec());
            for &j in &path {
                k1.extend_from_slice(&kc[(base + j) * 1024..(base + j + 1) * 1024]);
                v1.extend_from_slice(&vc[(base + j) * 1024..(base + j + 1) * 1024]);
            }
            let n1 = base + path.len();
            let (dk1, dv1) = (DeviceBuffer::from_slice(&k1)?, DeviceBuffer::from_slice(&v1)?);
            let dq1 = DeviceBuffer::from_slice(&q[i * 6144..(i + 1) * 6144])?;
            let dg1 = DeviceBuffer::from_slice(&qg[i * 12288..(i + 1) * 12288])?;
            let (p1, nb, nm) = (DeviceBuffer::from_slice(&[(n1 - 1) as u32])?, DeviceBuffer::from_slice(&[u32::MAX])?,
                DeviceBuffer::from_slice(&[0u32])?);
            let out1 = DeviceBuffer::<u16>::zeroed(6144)?;
            unsafe {
                s.launch(&sp, (1u32, 4, 16u32), 128, 4 * (1536 + 12) * 4, (po.ptr(), pm.ptr(), dq1.ptr(), dk1.ptr(),
                    dv1.ptr(), rslot.ptr(), p1.ptr(), nb.ptr(), nm.ptr(), (n1 * 1024) as u64, scale, 0u32))?;
                s.launch(&cb, (1u32, 24), 256, 0, (out1.ptr(), po.ptr(), pm.ptr(), dg1.ptr(), 12288u32, 16u32, 0u32))?;
            }
            s.sync()?;
            diff_rows += (out1.to_vec()?[..] != o[i * 6144..(i + 1) * 6144]) as usize;
        }
        let mut worst = 0f32;
        for i in 0..nt {
            let keys: Vec<usize> = (0..base).chain((0..nt).filter(|j| mask[i] >> j & 1 == 1).map(|j| base + j)).collect();
            for h in 0..24 {
                let kvh = h / 6;
                let sc: Vec<f32> = keys.iter().map(|&k| (0..256).map(|d| bf16_f32(q[(i * 24 + h) * 256 + d])
                    * bf16_f32(kc[k * 1024 + kvh * 256 + d])).sum::<f32>() / 16.0).collect();
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f32> = sc.iter().map(|v| (v - mx).exp()).collect();
                let z: f32 = w.iter().sum();
                for d in (0..256).step_by(17) {
                    let want: f32 = keys.iter().zip(&w).map(|(&k, &wk)| wk * bf16_f32(vc[k * 1024 + kvh * 256 + d])).sum::<f32>() / z;
                    let got = bf16_f32(o[i * 6144 + h * 256 + d]);
                    worst = worst.max((got - want).abs() / (want.abs() + 0.02));
                }
            }
        }
        println!("selftest tree attention (16 rows, ancestors + 100 cached keys): worst rel {worst:.4} vs CPU, \
            {diff_rows} rows differ from 1-row decode over the gathered path");
        ensure!(worst < 0.05 && diff_rows == 0, "tree attention mismatch");
    }
    // fe_attn_vk: 16 tree rows (segment 0, slot 0) must be BITWISE equal to the same rows run as 1-row decodes over
    // the gathered path (segments 1..16, slots 1..16), and as accurate as fe_attn_split vs an f64 reference.
    // 4090 cached keys: the rows straddle the 4096-key chunk boundary (two chunk groups in one segment).
    // FE_ATTN_BENCH=<keys>: also time 16-row trees and 1-row decode, fe_attn_vk vs fe_attn_split, at that context.
    {
        let (sp, cb, vk) = (m.function("fe_attn_split")?, m.function("fe_attn_combine")?, m.function("fe_attn_vk")?);
        vk.set_max_dynamic_smem(6 * 16 * 512 + 3 * 2 * 16 * 512)?;
        let bench: Option<usize> = std::env::var("FE_ATTN_BENCH").ok().and_then(|v| v.parse().ok());
        let (base, nt) = (bench.unwrap_or(4090), 16usize);
        let ctx = base + nt;
        let nslot = 1 + nt;
        let mut seed = 99u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
        };
        let mut par = vec![-1i32; nt];
        for i in 1..nt {
            let lo = i.saturating_sub(4);
            par[i] = (lo + ((rnd() + 0.5) * 1000.0) as usize % (i - lo)) as i32;
        }
        let mut mask = vec![0u32; nt];
        let mut path: Vec<Vec<usize>> = vec![Vec::new(); nt];
        for i in 0..nt {
            mask[i] = if i == 0 { 1 } else { mask[par[i] as usize] | (1 << i) };
            path[i] = (0..nt).filter(|j| mask[i] >> j & 1 == 1).collect(); // ancestors in depth order (tree order)
        }
        let qt: Vec<u16> = (0..nt * 24 * 256).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let mut q = qt.clone();
        q.extend_from_slice(&qt); // decode rows 16..32 = tree rows 0..16
        // slot 0: prefix + the tree rows at base + i; slot 1 + i: prefix + row i's ancestors at base + depth
        let pre_k: Vec<u16> = (0..base * 1024).map(|i| bf16_bits(rnd() * if (i / 1024) % 97 == 3 { 6.0 } else { 2.0 })).collect();
        let pre_v: Vec<u16> = (0..base * 1024).map(|_| bf16_bits(rnd())).collect();
        let tk: Vec<u16> = (0..nt * 1024).map(|_| bf16_bits(rnd() * 2.0)).collect();
        let tv: Vec<u16> = (0..nt * 1024).map(|_| bf16_bits(rnd())).collect();
        let mut kc = vec![0u16; nslot * ctx * 1024];
        let mut vc = vec![0u16; nslot * ctx * 1024];
        for sl in 0..nslot {
            kc[sl * ctx * 1024..sl * ctx * 1024 + base * 1024].copy_from_slice(&pre_k);
            vc[sl * ctx * 1024..sl * ctx * 1024 + base * 1024].copy_from_slice(&pre_v);
        }
        for i in 0..nt {
            let d0 = (base + i) * 1024;
            kc[d0..d0 + 1024].copy_from_slice(&tk[i * 1024..(i + 1) * 1024]);
            vc[d0..d0 + 1024].copy_from_slice(&tv[i * 1024..(i + 1) * 1024]);
            for (d, &a) in path[i].iter().enumerate() {
                let o = ((1 + i) * ctx + base + d) * 1024;
                kc[o..o + 1024].copy_from_slice(&tk[a * 1024..(a + 1) * 1024]);
                vc[o..o + 1024].copy_from_slice(&tv[a * 1024..(a + 1) * 1024]);
            }
        }
        let qg: Vec<u16> = (0..2 * nt * 12288).map(|i| if (i % 512) >= 256 { bf16_bits(30.0) } else { 0 }).collect();
        let (dq, dk, dv, dg) = (DeviceBuffer::from_slice(&q)?, DeviceBuffer::from_slice(&kc)?, DeviceBuffer::from_slice(&vc)?,
            DeviceBuffer::from_slice(&qg)?);
        let depth: Vec<u32> = (0..nt).map(|i| path[i].len() as u32 - 1).collect();
        let rslot = DeviceBuffer::from_slice(&(0..2 * nt).map(|r| if r < nt { 0 } else { (r - nt + 1) as u32 }).collect::<Vec<_>>())?;
        let rpos = DeviceBuffer::from_slice(&(0..2 * nt).map(|r| base as u32 + depth[r % nt]).collect::<Vec<_>>())?;
        let rtbase = DeviceBuffer::from_slice(&(0..2 * nt).map(|r| if r < nt { base as u32 } else { u32::MAX }).collect::<Vec<_>>())?;
        let rtmask = DeviceBuffer::from_slice(&(0..2 * nt).map(|r| if r < nt { mask[r] } else { 0 }).collect::<Vec<_>>())?;
        let srow0 = DeviceBuffer::from_slice(&std::iter::once(0u32).chain((0..nt as u32).map(|i| nt as u32 + i)).collect::<Vec<_>>())?;
        let slen = DeviceBuffer::from_slice(&std::iter::once(nt as u32).chain(std::iter::repeat(1u32).take(nt)).collect::<Vec<_>>())?;
        let po = DeviceBuffer::<f32>::zeroed(2 * nt * 4 * 64 * 1536)?;
        let pm = DeviceBuffer::<f32>::zeroed(2 * nt * 4 * 64 * 12)?;
        let (out_vk, out_sp) = (DeviceBuffer::<u16>::zeroed(2 * nt * 6144)?, DeviceBuffer::<u16>::zeroed(nt * 6144)?);
        let scale = (1.0f32 / 16.0) * std::f32::consts::LOG2_E;
        let stride = (ctx * 1024) as u64;
        let smem_vk = 6 * 16 * 512 + 3 * 2 * 16 * 512;
        let run_vk = |s: &Stream, nseg: u32, rows: u32| -> Result<()> {
            unsafe {
                s.launch(&vk, (nseg, 4, 48), 192, smem_vk, (po.ptr(), pm.ptr(), dq.ptr(), dk.ptr(), dv.ptr(), rslot.ptr(),
                    rpos.ptr(), rtbase.ptr(), rtmask.ptr(), srow0.ptr(), slen.ptr(), 0u32, 0u32, stride, scale))?;
                s.launch(&cb, (rows, 24), 256, 0, (out_vk.ptr(), po.ptr(), pm.ptr(), dg.ptr(), 12288u32, 48u32, 0u32))?;
            }
            Ok(())
        };
        let ns_sp = 64u32;
        let run_sp = |s: &Stream, rows: u32| -> Result<()> {
            unsafe {
                s.launch(&sp, (rows, 4, ns_sp), 128, 4 * (1536 + 12) * 4, (po.ptr(), pm.ptr(), dq.ptr(), dk.ptr(), dv.ptr(),
                    rslot.ptr(), rpos.ptr(), rtbase.ptr(), rtmask.ptr(), stride, scale, 0u32))?;
                s.launch(&cb, (rows, 24), 256, 0, (out_sp.ptr(), po.ptr(), pm.ptr(), dg.ptr(), 12288u32, ns_sp, 0u32))?;
            }
            Ok(())
        };
        run_vk(s, 1 + nt as u32, 2 * nt as u32)?;
        run_sp(s, nt as u32)?;
        s.sync()?;
        let (ov, os) = (out_vk.to_vec()?, out_sp.to_vec()?);
        let tree_vs_decode = (0..nt * 6144).filter(|&i| ov[i] != ov[nt * 6144 + i]).count();
        let (mut w_vk, mut w_sp, mut n_cmp) = (0f64, 0f64, 0usize);
        for i in 0..nt {
            for h in 0..24 {
                let kvh = h / 6;
                let keys: Vec<(&[u16], &[u16])> = (0..base).map(|k| (&pre_k[k * 1024..], &pre_v[k * 1024..]))
                    .chain(path[i].iter().map(|&a| (&tk[a * 1024..], &tv[a * 1024..]))).collect();
                let sc: Vec<f64> = keys.iter().map(|(k, _)| (0..256).map(|d| bf16_f32(qt[(i * 24 + h) * 256 + d]) as f64
                    * bf16_f32(k[kvh * 256 + d]) as f64).sum::<f64>() / 16.0).collect();
                let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = sc.iter().map(|v| (v - mx).exp()).collect();
                let z: f64 = w.iter().sum();
                for d in (0..256).step_by(5) {
                    let want: f64 = keys.iter().zip(&w).map(|((_, v), wk)| wk * bf16_f32(v[kvh * 256 + d]) as f64).sum::<f64>() / z;
                    let den = want.abs() + 0.01;
                    w_vk = w_vk.max((bf16_f32(ov[i * 6144 + h * 256 + d]) as f64 - want).abs() / den);
                    w_sp = w_sp.max((bf16_f32(os[i * 6144 + h * 256 + d]) as f64 - want).abs() / den);
                    n_cmp += 1;
                }
            }
        }
        println!("selftest fe_attn_vk (16 tree rows, {base} cached keys): tree rows vs their 1-row decodes: {tree_vs_decode} of \
            {} outputs differ; vs f64 worst rel vk {w_vk:.4}, split-K {w_sp:.4} ({n_cmp} values)", nt * 6144);
        ensure!(tree_vs_decode == 0, "fe_attn_vk: a tree row differs from its 1-row decode");
        ensure!(w_vk < 0.05 && w_vk <= 2.0 * w_sp.max(0.004), "fe_attn_vk accuracy");
        if bench.is_some() {
            let tm = |f: &dyn Fn(&Stream) -> Result<()>| -> Result<f32> {
                let (e0, e1) = (fe_cuda::Event::new()?, fe_cuda::Event::new()?);
                f(s)?;
                s.record(&e0)?;
                for _ in 0..20 {
                    f(s)?;
                }
                s.record(&e1)?;
                e1.sync()?;
                Ok(e1.elapsed_since(&e0)? / 20.0)
            };
            // 16-row tree: segment 0 only; 1-row decode: segment 1 only (its own launch)
            let tv16 = tm(&|s| run_vk(s, 1, nt as u32))?;
            let ts16 = tm(&|s| run_sp(s, nt as u32))?;
            let tv1 = tm(&|s| unsafe {
                s.launch(&vk, (1u32, 4, 48), 192, smem_vk, (po.ptr(), pm.ptr(), dq.ptr(), dk.ptr(), dv.ptr(), rslot.ptr(),
                    rpos.ptr(), rtbase.ptr(), rtmask.ptr(), srow0.ptr(), slen.ptr(), 1u32, nt as u32, stride, scale))?;
                s.launch(&cb, (1u32, 24), 256, 0, (out_vk.ptr(), po.ptr(), pm.ptr(), dg.ptr(), 12288u32, 48u32, nt as u32))?;
                Ok(())
            })?;
            let ts1 = tm(&|s| unsafe {
                s.launch(&sp, (1u32, 4, ns_sp), 128, 4 * (1536 + 12) * 4, (po.ptr(), pm.ptr(), dq.ptr(), dk.ptr(), dv.ptr(),
                    rslot.ptr(), rpos.ptr(), rtbase.ptr(), rtmask.ptr(), stride, scale, nt as u32))?;
                s.launch(&cb, (1u32, 24), 256, 0, (out_sp.ptr(), po.ptr(), pm.ptr(), dg.ptr(), 12288u32, ns_sp, nt as u32))?;
                Ok(())
            })?;
            println!("attention bench at {base} keys, 4 kv heads: 16-row tree vk {tv16:.3} ms vs split-K {ts16:.3} ms; \
                1-row decode vk {tv1:.3} ms vs split-K {ts1:.3} ms");
        }
    }
    // verify-tree GDN: conv taps and recurrence along each row's own root path == the chain kernels run over that path
    {
        let (fc, fr) = (m.function("fe_gdn_conv")?, m.function("fe_gdn_rec2")?);
        let (n, cd, hd) = (16usize, 10240usize, 6144usize);
        let mut seed = 99u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
        };
        let mut par = vec![u32::MAX; n];
        for i in 1..n {
            let lo = i.saturating_sub(5);
            par[i] = (lo + ((rnd() + 0.5) * 1000.0) as usize % (i - lo)) as u32;
        }
        let x: Vec<u16> = (0..n * cd).map(|_| bf16_bits(rnd() * 2.0)).collect();
        let w: Vec<u16> = (0..cd * 4).map(|_| bf16_bits(rnd())).collect();
        let cst: Vec<u16> = (0..3 * cd).map(|_| bf16_bits(rnd() * 2.0)).collect();
        let ab: Vec<u16> = (0..n * 96).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let alog: Vec<u16> = (0..48).map(|_| bf16_bits(rnd() * 4.0 - 2.0)).collect();
        let dtb: Vec<u16> = (0..48).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let rst: Vec<f32> = (0..48 * 128 * 128).map(|_| rnd() * 0.1).collect();
        let (dw, dal, ddt) = (DeviceBuffer::from_slice(&w)?, DeviceBuffer::from_slice(&alog)?, DeviceBuffer::from_slice(&dtb)?);
        let null = fe_cuda::DevPtr::NULL;
        // one run of conv + recurrence over `rows` rows (x / a|b rows given) with the meta of one segment
        let run = |xr: &[u16], abr: &[u16], parr: &[u32], tree: bool| -> Result<(Vec<u16>, Vec<u16>, Vec<u16>)> {
            let rows = parr.len();
            let pi: Vec<i32> = parr.iter().map(|&p| p as i32).collect();
            let dsteps = DeviceBuffer::from_slice(&crate::model::tree_steps(&pi)?)?;
            let (dx, dab) = (DeviceBuffer::from_slice(xr)?, DeviceBuffer::from_slice(abr)?);
            let (dcs, drs) = (DeviceBuffer::from_slice(&cst)?, DeviceBuffer::from_slice(&rst)?);
            let meta = DeviceBuffer::from_slice(&[0u32, 0, rows as u32, tree as u32, 1])?;
            let anc: Vec<u32> = (0..rows).map(|r| crate::model::ancestors(&pi, r)).collect();
            let dpar = DeviceBuffer::from_slice(&anc)?;
            let (out, hist, o) = (DeviceBuffer::<u16>::zeroed(rows * cd)?, DeviceBuffer::<u16>::zeroed(rows * 3 * cd)?,
                DeviceBuffer::<u16>::zeroed(rows * hd)?);
            unsafe {
                s.launch(&fc, ((cd / 256) as u32, 1u32), 256, 0, (out.ptr(), dx.ptr(), cd as u32, dw.ptr(), dcs.ptr(),
                    meta.at(0), meta.at(1), meta.at(2), hist.ptr(), rows as u32, dpar.ptr(), meta.at(3)))?;
                s.launch(&fr, (192u32, 1u32), 512, 0, (o.ptr(), out.ptr(), dab.ptr(), dab.at(48), 96u32, dal.ptr(),
                    ddt.ptr(), drs.ptr(), meta.at(0), meta.at(1), meta.at(2), null, 0u32,
                    if tree { meta.at(4) } else { null }, dsteps.ptr(), meta.at(3)))?;
            }
            s.sync()?;
            Ok((out.to_vec()?, hist.to_vec()?, o.to_vec()?))
        };
        let (tc, th, to) = run(&x, &ab, &par, true)?;
        let mut bad = 0usize;
        for r in 0..n {
            let mut path = vec![r];
            while par[*path.last().unwrap()] != u32::MAX {
                path.push(par[*path.last().unwrap()] as usize);
            }
            path.reverse();
            let xr: Vec<u16> = path.iter().flat_map(|&q| x[q * cd..(q + 1) * cd].iter().copied()).collect();
            let abr: Vec<u16> = path.iter().flat_map(|&q| ab[q * 96..(q + 1) * 96].iter().copied()).collect();
            let cp: Vec<u32> = (0..path.len()).map(|i| if i == 0 { u32::MAX } else { i as u32 - 1 }).collect();
            let (cc, ch, co) = run(&xr, &abr, &cp, false)?;
            let l = path.len() - 1;
            bad += (cc[l * cd..(l + 1) * cd] != tc[r * cd..(r + 1) * cd]) as usize
                + (ch[l * 3 * cd..(l + 1) * 3 * cd] != th[r * 3 * cd..(r + 1) * 3 * cd]) as usize
                + (co[l * hd..(l + 1) * hd] != to[r * hd..(r + 1) * hd]) as usize;
        }
        let st = crate::model::tree_steps(&par.iter().map(|&p| p as i32).collect::<Vec<_>>())?;
        let saves = st.iter().filter(|&&x| (x >> 6) & 15 != 0).count();
        let leaves = st.iter().filter(|&&x| (x >> 5) & 1 != 0).count();
        println!("selftest tree GDN (16 rows, {leaves} leaves, {saves} saved states: conv, history, recurrence vs the chain \
            over each path): {bad} differing");
        ensure!(bad == 0, "tree GDN differs from the chain kernels");
    }
    Ok(())
}
fn kvq_bench(m: &Module, s: &Stream, keys: usize) -> Result<()> {
    let n = 2048usize;
    let mut seed = 77u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed
    };
    let q: Vec<u16> = (0..n * 6144).map(|_| bf16_bits(((rnd() >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * 2.0)).collect();
    let (dq, dqg) = (DeviceBuffer::from_slice(&q)?, DeviceBuffer::<u16>::zeroed(n * 12288)?);
    let out = DeviceBuffer::<u16>::zeroed(n * 6144)?;
    let scale = (1.0f32 / 16.0) * std::f32::consts::LOG2_E;
    let rslot = DeviceBuffer::from_slice(&vec![0u32; 16])?;
    let (nobase, nomask) = (DeviceBuffer::from_slice(&vec![u32::MAX; 16])?, DeviceBuffer::from_slice(&vec![0u32; 16])?);
    // the quantized forms take their split count from the grid (FE_KVQ_BENCH_SPLITS); fe_attn_vk has 48 built in
    let qsplits: u32 = std::env::var("FE_KVQ_BENCH_SPLITS").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
    let po = DeviceBuffer::<f32>::zeroed(16 * 4 * qsplits.max(48) as usize * 1536)?;
    let pm = DeviceBuffer::<f32>::zeroed(16 * 4 * qsplits.max(48) as usize * 12)?;
    println!("kvq bench: {keys} keys; ms for a 2048-row prompt chunk (flash) | 1-row decode | 16-row verify (vk + combine), one layer");
    for bits in [0usize, 4, 6] {
        let rec = match bits { 0 => 512, 6 => 208, _ => 144 };
        // random cache contents (records: random codes, scales of 1.0)
        let mut c: Vec<u32> = (0..keys * 4 * rec / 4).map(|_| rnd()).collect();
        if bits == 0 {
            for w in c.iter_mut() {
                *w = (*w & 0x00ff_00ff) | 0x3f00_3f00;
            }
        } else {
            for t in 0..keys * 4 {
                for g in 0..4 {
                    c[(t * rec + rec - 16) / 4 + g] = 0x3f80_3f80;
                }
            }
        }
        let (dk, dv) = (DeviceBuffer::from_slice(&c)?, DeviceBuffer::from_slice(&c)?);
        let stride = if bits == 0 { (keys * 1024) as u64 } else { (keys * 4 * rec) as u64 };
        let sfx = if bits == 0 { String::new() } else { format!("_q{bits}") };
        let fl = m.function(&format!("fe_flash_prefill{sfx}"))?;
        let vk = m.function(&format!("fe_attn_vk{sfx}"))?;
        let cb = m.function(if bits == 0 { "fe_attn_combine" } else { "fe_attn_combine_q" })?;
        let (fl_smem, vk_smem) = if bits == 0 { (98304u32, 98304u32) } else {
            ((64 * 512 + 2 * 2 * 32 * rec + 2 * 32 * 512) as u32, (6 * 16 * 512 + 3 * 2 * 16 * rec + 2 * 16 * 512) as u32)
        };
        fl.set_max_dynamic_smem(fl_smem)?;
        vk.set_max_dynamic_smem(vk_smem)?;
        let pos0 = (keys - n) as u32;
        let splits = if bits == 0 { 48 } else { qsplits };
        let mut t = [0f32; 3];
        for it in 0..4 {
            let ms = s.time(|s| unsafe {
                s.launch(&fl, ((n as u32).div_ceil(64), 24), 128, fl_smem, (out.ptr(), dq.ptr(), dqg.ptr(), 12288u32, dk.ptr(),
                    dv.ptr(), stride, 0u32, 0u32, n as u32, pos0, scale))?;
                Ok(())
            })?;
            if it > 0 {
                t[0] += ms / 3.0;
            }
        }
        for (ti, rows) in [(1usize, 1u32), (2, 16)] {
            let rpos = DeviceBuffer::from_slice(&(0..16u32).map(|i| keys as u32 - rows + i.min(rows - 1)).collect::<Vec<_>>())?;
            let (srow0, slen) = (DeviceBuffer::from_slice(&[0u32])?, DeviceBuffer::from_slice(&[rows])?);
            for it in 0..6 {
                let ms = s.time(|s| unsafe {
                    s.launch(&vk, (1u32, 4, splits), 192, vk_smem, (po.ptr(), pm.ptr(), dq.ptr(), dk.ptr(), dv.ptr(), rslot.ptr(),
                        rpos.ptr(), nobase.ptr(), nomask.ptr(), srow0.ptr(), slen.ptr(), 0u32, 0u32, stride, scale))?;
                    s.launch(&cb, (rows, 24), 256, 0, (out.ptr(), po.ptr(), pm.ptr(), dqg.ptr(), 12288u32, splits, 0u32))?;
                    Ok(())
                })?;
                if it > 0 {
                    t[ti] += ms / 5.0;
                }
            }
        }
        let mut tp = 0f32;
        let pthreads: u32 = 128 + 32 * std::env::var("FE_KVQ_BENCH_PROD").ok().and_then(|v| v.parse().ok()).unwrap_or(4u32);
        if bits != 0 {
            let flp = m.function(&format!("fe_flash_prefill_q{bits}p"))?;
            let flp_smem = (64 * 512 + 2 * 32 * 512 + 2 * 32 * rec) as u32;
        flp.set_max_dynamic_smem(flp_smem)?;
            for it in 0..4 {
                let ms = s.time(|s| unsafe {
                    s.launch(&flp, ((n as u32).div_ceil(64), 24), pthreads, flp_smem, (out.ptr(), dq.ptr(), dqg.ptr(), 12288u32, dk.ptr(),
                        dv.ptr(), stride, 0u32, 0u32, n as u32, pos0, scale))?;
                    Ok(())
                })?;
                if it > 0 {
                    tp += ms / 3.0;
                }
            }
        }
        if bits != 0 {
            let flp = m.function(&format!("fe_flash_prefill_q{bits}p"))?;
            let flp_smem = (64 * 512 + 2 * 32 * 512 + 2 * 32 * rec) as u32;
            let o2 = DeviceBuffer::<u16>::zeroed(n * 6144)?;
            unsafe {
                s.launch(&fl, ((n as u32).div_ceil(64), 24), 128, fl_smem, (out.ptr(), dq.ptr(), dqg.ptr(), 12288u32, dk.ptr(),
                    dv.ptr(), stride, 0u32, 0u32, n as u32, pos0, scale))?;
            }
            s.sync()?;
            let want = out.to_vec()?;
            let mut nd = 0usize;
            for _ in 0..10 {
                unsafe {
                    s.launch(&flp, ((n as u32).div_ceil(64), 24), pthreads, flp_smem, (o2.ptr(), dq.ptr(), dqg.ptr(), 12288u32,
                        dk.ptr(), dv.ptr(), stride, 0u32, 0u32, n as u32, pos0, scale))?;
                }
                s.sync()?;
                nd += want.iter().zip(&o2.to_vec()?).filter(|(x, y)| x != y).count();
            }
            println!("kvq bench {bits}-bit: producer-warp flash vs plain over 10 runs: {nd} outputs differ");
            ensure!(nd == 0, "kvq bench: the producer-warp flash kernel differs");
        }
        println!("kvq bench {}: flash {:8.2} (producer warps {:8.2}) | decode {:6.3} | verify16 {:6.3}",
            if bits == 0 { "bf16 ".into() } else { format!("{bits}-bit") }, t[0], tp, t[1], t[2]);
    }
    Ok(())
}

/// The quantized K/V cache kernels (k-core kvq.rs); `FE_SELFTEST_ONLY=kvq` runs only this (e.g. under
/// compute-sanitizer's racecheck).
fn kvq_selftest(m: &Module, s: &Stream) -> Result<()> {
    // the quantized K/V cache (k-core kvq.rs) at both widths: fe_attn_prep_q's V records against the host quantizer
    // (exact), then fe_flash_prefill_q against fe_attn_vk_q + fe_attn_combine_q over the same records (the two differ in
    // how they round a dequantized V: bf16 against f16)
    for bits in [4usize, 6] {
        let rec = if bits == 6 { 208usize } else { 144 };
        let (ctx, n) = (128usize, 100usize);
        let prep = m.function(&format!("fe_attn_prep_q{bits}"))?;
        let fl = m.function(&format!("fe_flash_prefill_q{bits}"))?;
        let vk = m.function(&format!("fe_attn_vk_q{bits}"))?;
        let cb = m.function("fe_attn_combine_q")?;
        let (fl_smem, vk_smem) = ((64 * 512 + 2 * 2 * 32 * rec + 2 * 32 * 512) as u32, (6 * 16 * 512 + 3 * 2 * 16 * rec + 2 * 16 * 512) as u32);
        fl.set_max_dynamic_smem(fl_smem)?;
        vk.set_max_dynamic_smem(vk_smem)?;
        let mut seed = 4711u32 + bits as u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5
        };
        let qg: Vec<u16> = (0..n * 12288).map(|_| bf16_bits(rnd() * 4.0)).collect();
        let kx: Vec<u16> = (0..n * 1024).map(|_| bf16_bits(rnd() * 4.0)).collect();
        // V with a few large values, so group scales differ
        let vx: Vec<u16> = (0..n * 1024).map(|i| bf16_bits(rnd() * if i % 37 == 0 { 3.0 } else { 1.0 })).collect();
        let nw: Vec<u16> = (0..256).map(|_| bf16_bits(rnd() * 0.2)).collect();
        let rope: Vec<u16> = (0..ctx * 64).map(|i| {
            let ang = (i / 64) as f32 * 0.37 / (1.0 + (i % 32) as f32);
            bf16_bits(if i % 64 < 32 { ang.cos() } else { ang.sin() })
        }).collect();
        let (dqg, dkx, dvx, dnw, drope) = (DeviceBuffer::from_slice(&qg)?, DeviceBuffer::from_slice(&kx)?,
            DeviceBuffer::from_slice(&vx)?, DeviceBuffer::from_slice(&nw)?, DeviceBuffer::from_slice(&rope)?);
        let dq = DeviceBuffer::<u16>::zeroed(n * 6144)?;
        let (dk, dv) = (DeviceBuffer::<u8>::zeroed(ctx * 4 * rec)?, DeviceBuffer::<u8>::zeroed(ctx * 4 * rec)?);
        let rslot = DeviceBuffer::from_slice(&vec![0u32; n])?;
        let rpos = DeviceBuffer::from_slice(&(0..n as u32).collect::<Vec<_>>())?;
        let stride = (ctx * 4 * rec) as u64;
        unsafe {
            s.launch(&prep, n as u32, 1024, 0, (dq.ptr(), dk.ptr(), dv.ptr(), dqg.ptr(), 12288u32, dkx.ptr(), 1024u32,
                dvx.ptr(), 1024u32, dnw.ptr(), dnw.ptr(), drope.ptr(), rslot.ptr(), rpos.ptr(), rpos.ptr(), stride, 1e-6f32))?;
        }
        s.sync().map_err(|e| anyhow::anyhow!("prep_q: {e}"))?;
        // host quantizer of V (kvq.rs had32 + prep_q, the same operation order)
        let got = dv.to_vec()?;
        let mut bad = 0usize;
        let (half, top) = ((1u32 << (bits - 1)) as f32, (1i32 << bits) - 1);
        for r in 0..n {
            for hg in 0..32 {
                let (h, g) = (hg / 8, hg % 8);
                let mut y = [[0f32; 8]; 4];
                for (l, yl) in y.iter_mut().enumerate() {
                    for (j, v) in yl.iter_mut().enumerate() {
                        *v = bf16_f32(vx[r * 1024 + h * 256 + g * 32 + l * 8 + j]);
                    }
                    for st in [1usize, 2, 4] {
                        let t = *yl;
                        for j in 0..8 {
                            yl[j] = if j & st == 0 { t[j] + t[j + st] } else { t[j - st] - t[j] };
                        }
                    }
                }
                for m2 in [1usize, 2] {
                    let t = y;
                    for l in 0..4 {
                        for j in 0..8 {
                            y[l][j] = if l & m2 == 0 { t[l][j] + t[l ^ m2][j] } else { t[l ^ m2][j] - t[l][j] };
                        }
                    }
                }
                let mut am = 0f32;
                for yl in y.iter_mut() {
                    for v in yl.iter_mut() {
                        *v *= 0.176_776_7;
                        am = am.max(v.abs());
                    }
                }
                let mut sb = bf16_bits(am);
                if bf16_f32(sb) < am {
                    sb += 1;
                }
                let sc = bf16_f32(sb);
                let inv = if sc > 0.0 { half / sc } else { 0.0 };
                let base = (r * 4 + h) * rec;
                let rd16 = |o: usize| u16::from_le_bytes([got[o], got[o + 1]]);
                bad += (rd16(base + rec - 16 + g * 2) != sb) as usize;
                for l in 0..4 {
                    let (mut word, mut low) = (0u32, 0u32);
                    for j in 0..8 {
                        let qi = (y[l][j].mul_add(inv, half) as i32).clamp(0, top) as u32;
                        word |= (qi >> (bits - 4)) << (4 * j);
                        low |= (qi & 3) << (2 * j);
                    }
                    let wo = base + (g * 4 + l) * 4;
                    bad += (u32::from_le_bytes([got[wo], got[wo + 1], got[wo + 2], got[wo + 3]]) != word) as usize;
                    if bits == 6 {
                        bad += (rd16(base + 128 + (g * 4 + l) * 2) as u32 != low) as usize;
                    }
                }
            }
        }
        ensure!(bad == 0, "kvq {bits}-bit: {bad} V record fields differ from the host quantizer");
        let o1 = DeviceBuffer::<u16>::zeroed(n * 6144)?;
        let o2 = DeviceBuffer::<u16>::zeroed(n * 6144)?;
        let scale = (1.0f32 / 16.0) * std::f32::consts::LOG2_E;
        let nseg = n.div_ceil(16);
        let srow0 = DeviceBuffer::from_slice(&(0..nseg).map(|i| (i * 16) as u32).collect::<Vec<_>>())?;
        let slen = DeviceBuffer::from_slice(&(0..nseg).map(|i| (n - i * 16).min(16) as u32).collect::<Vec<_>>())?;
        let (nobase, nomask) = (DeviceBuffer::from_slice(&vec![u32::MAX; n])?, DeviceBuffer::from_slice(&vec![0u32; n])?);
        let po = DeviceBuffer::<f32>::zeroed(n * 4 * 48 * 1536)?;
        let pm = DeviceBuffer::<f32>::zeroed(n * 4 * 48 * 12)?;
        unsafe {
            s.launch(&fl, ((n as u32).div_ceil(64), 24), 128, fl_smem, (o1.ptr(), dq.ptr(), dqg.ptr(), 12288u32, dk.ptr(),
                dv.ptr(), stride, 0u32, 0u32, n as u32, 0u32, scale))?;
            s.sync().map_err(|e| anyhow::anyhow!("flash_q: {e}"))?;
            s.launch(&vk, (nseg as u32, 4, 48), 192, vk_smem, (po.ptr(), pm.ptr(), dq.ptr(), dk.ptr(), dv.ptr(), rslot.ptr(),
                rpos.ptr(), nobase.ptr(), nomask.ptr(), srow0.ptr(), slen.ptr(), 0u32, 0u32, stride, scale))?;
            s.sync().map_err(|e| anyhow::anyhow!("vk_q: {e}"))?;
            s.launch(&cb, (n as u32, 24), 256, 0, (o2.ptr(), po.ptr(), pm.ptr(), dqg.ptr(), 12288u32, 48u32, 0u32))?;
        }
        // the producer-warp form of the flash kernel: the same bits
        let flp = m.function(&format!("fe_flash_prefill_q{bits}p"))?;
        let flp_smem = (64 * 512 + 2 * 32 * 512 + 2 * 32 * rec) as u32;
        flp.set_max_dynamic_smem(flp_smem)?;
        let o3 = DeviceBuffer::<u16>::zeroed(n * 6144)?;
        let (a, b) = (o1.to_vec()?, o2.to_vec()?);
        // 25 runs: its producer and tensor-core warps hand tiles over under barriers, and a missing one would show
        // as an occasional difference
        for rep in 0..25 {
            unsafe {
                s.launch(&flp, ((n as u32).div_ceil(64), 24), 256, flp_smem, (o3.ptr(), dq.ptr(), dqg.ptr(), 12288u32, dk.ptr(),
                    dv.ptr(), stride, 0u32, 0u32, n as u32, 0u32, scale))?;
            }
            s.sync()?;
            let nd = a.iter().zip(&o3.to_vec()?).filter(|(x, y)| x != y).count();
            ensure!(nd == 0, "kvq {bits}-bit: fe_flash_prefill_q{bits}p differs from fe_flash_prefill_q{bits} in {nd} outputs (run {rep})");
        }
        let mut worst = 0f32;
        let mut wi = 0;
        let mut rows_bad = std::collections::BTreeMap::new();
        for i in 0..a.len() {
            let (x, y) = (bf16_f32(a[i]), bf16_f32(b[i]));
            // flash_q rounds a dequantized V to bf16, vk_q to f16: up to ~0.004 apart at these magnitudes
            let e = (x - y).abs() / (y.abs() + 0.1);
            if e > worst {
                worst = e;
                wi = i;
            }
            if e > 0.03 {
                *rows_bad.entry(i / 6144).or_insert(0usize) += 1;
            }
        }
        if std::env::var("FE_SELFTEST_V").is_ok() {
            println!("kvq {bits}: worst at row {} head {} dim {}: flash {} vk {}; rows over 0.03: {:?}", wi / 6144, (wi % 6144) / 256,
                wi % 256, bf16_f32(a[wi]), bf16_f32(b[wi]), rows_bad);
        }
        // row 0 has one key (itself): its output is its own V through the cache, times the gate
        let mut se = 0f64;
        let mut sv = 0f64;
        for h in 0..24 {
            for d in 0..256 {
                let v = bf16_f32(vx[(h / 6) * 256 + d]);
                let g = bf16_f32(qg[h * 512 + 256 + d]);
                let want = v * bf16_f32(bf16_bits(1.0 / (1.0 + (-g).exp())));
                let got_ = bf16_f32(b[h * 256 + d]);
                se += ((got_ - want) as f64).powi(2);
                sv += (want as f64).powi(2);
            }
        }
        let rel = (se / sv).sqrt();
        println!("selftest kvq {bits}-bit: V records = host quantizer; flash_q vs vk_q + combine_q worst rel {worst:.4}; a V row through the cache: rms error {rel:.4} of its rms");
        ensure!(rel < if bits == 6 { 0.03 } else { 0.12 }, "kvq {bits}-bit: a row's own V comes back {rel:.4} off");
        ensure!(worst < 0.05 && b.iter().any(|&x| x & 0x7fff != 0), "kvq {bits}-bit: flash_q and vk_q disagree");
    }
    Ok(())
}
