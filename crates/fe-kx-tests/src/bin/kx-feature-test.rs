//! X0 feature proofs: every kernel of kx-test (Rust -> SPIR-V) against a CPU reference. Exit 1 on any mismatch.
//! usage: kx-feature-test  (run through b70run)
use fe_kx_tests::*;
use fe_ze::*;

struct T {
    fails: u32,
}
impl T {
    fn check(&mut self, name: &str, ok: bool, detail: String) {
        println!("{:<44} {}  {detail}", name, if ok { "PASS" } else { "FAIL" });
        if !ok {
            self.fails += 1;
        }
    }
}

fn main() -> Result<()> {
    let ze = Ze::new()?;
    let q = Queue::new(&ze)?;
    let i = &ze.info;
    println!(
        "{}: {} slices x {} subslices x {} EUs, {} threads/EU, {} MHz, {:.1} GiB, sub-groups {:?}, SLM {} KiB, timer {} Hz",
        i.name,
        i.slices,
        i.subslices_per_slice,
        i.eus_per_subslice,
        i.threads_per_eu,
        i.core_clock_mhz,
        i.mem_total as f64 / (1u64 << 30) as f64,
        i.sg_sizes,
        i.max_slm / 1024,
        i.timer_hz
    );
    let m = Module::from_spirv(&ze, KX_TEST, "")?;
    if !m.log.is_empty() {
        println!("build log: {}", m.log);
    }
    println!("kernels: {:?}", m.kernel_names()?);
    let mut t = T { fails: 0 };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    // ---- ids ----
    {
        let k = m.kernel("t_ids", [64, 1, 1])?;
        let out = DeviceBuffer::<u32>::zeroed(&q, 4 * 64 * 8)?;
        k.args(&[&out.ptr()])?;
        q.launch(&k, [4, 1, 1])?;
        let o = out.to_vec(&q)?;
        let mut bad = 0;
        for g in 0..256usize {
            let l = g % 64;
            let want = [g as u32, l as u32, (g / 64) as u32, (l % 16) as u32, (l / 16) as u32, 4, 16, 64];
            if o[g * 8..g * 8 + 8] != want {
                bad += 1;
            }
        }
        let p = k.props()?;
        t.check("ids (global/local/group/lane/sg/nsg/sgsize)", bad == 0, format!("{bad} bad items; required sg {}", p.required_sg_size));
    }

    // ---- sub-group ops ----
    {
        let k = m.kernel("t_sg", [64, 1, 1])?;
        let n = 64usize;
        let x: Vec<f32> = (0..n).map(|_| rng.range(-50, 50) as f32).collect();
        let u: Vec<u32> = (0..n).map(|i| if i % 13 == 0 { 7 } else { rng.next_u32() % 1000 }).collect();
        let dx = DeviceBuffer::from_slice(&q, &x)?;
        let du = DeviceBuffer::from_slice(&q, &u)?;
        let out = DeviceBuffer::<f32>::zeroed(&q, n * 10)?;
        let outu = DeviceBuffer::<u32>::zeroed(&q, n * 5)?;
        k.args(&[&dx.ptr(), &du.ptr(), &out.ptr(), &outu.ptr()])?;
        q.launch(&k, [1, 1, 1])?;
        let o = out.to_vec(&q)?;
        let ou = outu.to_vec(&q)?;
        let mut bad = Vec::new();
        for g in 0..n {
            let (b, l) = (g / 16 * 16, g % 16);
            let sgx = &x[b..b + 16];
            let sgu = &u[b..b + 16];
            let sum: f32 = sgx.iter().sum();
            let incl: f32 = sgx[..=l].iter().sum();
            let excl: f32 = sgx[..l].iter().sum();
            let down = if l + 3 < 16 { sgx[l + 3] } else { sgx[l + 3 - 16] + 100.0 };
            let up = if l >= 3 { sgx[l - 3] } else { sgx[16 + l - 3] - 100.0 };
            let want = [
                sgx[(l * 3) & 15],
                sgx[l ^ 5],
                sgx[7],
                sum,
                sgx.iter().cloned().fold(f32::MIN, f32::max),
                sgx.iter().cloned().fold(f32::MAX, f32::min),
                incl,
                excl,
                down,
                up,
            ];
            for (j, w) in want.iter().enumerate() {
                if o[g * 10 + j] != *w {
                    bad.push(format!("f{j}@{g}: {} vs {w}", o[g * 10 + j]));
                }
            }
            let any7 = sgu.contains(&7) as u32;
            let wu = [sgu[15 - l], sgu[l ^ 3], sgu.iter().sum(), sgu[..=l].iter().sum(), any7 + 2];
            for (j, w) in wu.iter().enumerate() {
                if ou[g * 5 + j] != *w {
                    bad.push(format!("u{j}@{g}: {} vs {w}", ou[g * 5 + j]));
                }
            }
        }
        t.check(
            "sub-group shuffle/xor/down/up/bcast/reduce/scan/any/all",
            bad.is_empty(),
            format!("{} bad {}", bad.len(), bad.iter().take(4).cloned().collect::<Vec<_>>().join(", ")),
        );
    }

    // ---- SLM (static + dynamic), barrier, split barrier ----
    {
        let k = m.kernel("t_slm", [256, 1, 1])?;
        let n = 4 * 256;
        let x: Vec<f32> = (0..n).map(|_| rng.f32() * 8.0 - 4.0).collect();
        let dx = DeviceBuffer::from_slice(&q, &x)?;
        let out = DeviceBuffer::<f32>::zeroed(&q, n)?;
        k.args(&[&dx.ptr(), &out.ptr(), &Local(256 * 4)])?;
        q.launch(&k, [4, 1, 1])?;
        let o = out.to_vec(&q)?;
        let mut bad = 0;
        for g in 0..n {
            let (b, l) = (g / 256 * 256, g % 256);
            let mm = l ^ 1;
            let want = x[b + 255 - mm] + x[b + ((mm + 1) & 255)] * 2.0;
            if o[g].to_bits() != want.to_bits() {
                bad += 1;
            }
        }
        let p = k.props()?;
        t.check(
            "SLM static + dynamic (__local arg), barrier, split barrier",
            bad == 0,
            format!("{bad} bad; reqd group {:?}, static SLM {} B", p.required_group, p.slm_bytes),
        );
    }

    // ---- block reads / writes ----
    {
        let k = m.kernel("t_blk", [16, 1, 1])?;
        let ng = 8usize;
        let src: Vec<u32> = (0..ng * 128).map(|_| rng.next_u32()).collect();
        let src16: Vec<u16> = (0..ng * 128).map(|_| rng.next_u32() as u16).collect();
        let ds = DeviceBuffer::from_slice(&q, &src)?;
        let ds16 = DeviceBuffer::from_slice(&q, &src16)?;
        let copy = DeviceBuffer::<u32>::zeroed(&q, ng * 128)?;
        let chk = DeviceBuffer::<u32>::zeroed(&q, ng * 16 * 7)?;
        let copy16 = DeviceBuffer::<u16>::zeroed(&q, ng * 128)?;
        let chk16 = DeviceBuffer::<u16>::zeroed(&q, ng * 16)?;
        let lcopy = DeviceBuffer::<u32>::zeroed(&q, ng * 128)?;
        k.args(&[&ds.ptr(), &ds16.ptr(), &copy.ptr(), &chk.ptr(), &copy16.ptr(), &chk16.ptr(), &lcopy.ptr()])?;
        q.launch(&k, [ng as u32, 1, 1])?;
        let (c, h, c16, h16, lc) = (copy.to_vec(&q)?, chk.to_vec(&q)?, copy16.to_vec(&q)?, chk16.to_vec(&q)?, lcopy.to_vec(&q)?);
        let mut bad = 0;
        for gi in 0..ng {
            for l in 0..16 {
                let r = |i: usize| src[gi * 128 + i * 16 + l];
                let want = [r(0), r(0), r(1), r(0), r(1), r(2), r(3)];
                if h[(gi * 16 + l) * 7..(gi * 16 + l) * 7 + 7] != want {
                    bad += 1;
                }
                if h16[gi * 16 + l] != src16[gi * 128 + l] {
                    bad += 1;
                }
            }
        }
        let ok = bad == 0 && c == src && c16 == src16 && lc == src;
        t.check(
            "sub-group block read/write x1/2/4/8 (u32,u16), global + SLM",
            ok,
            format!("lane bad {bad}, copy {} copy16 {} slm {}", c == src, c16 == src16, lc == src),
        );
    }

    // ---- DPAS ----
    {
        let k = m.kernel("t_dpas", [16, 1, 1])?;
        let am: Vec<i32> = (0..8 * 16).map(|_| rng.range(-3, 4)).collect();
        let bm: Vec<i32> = (0..16 * 16).map(|_| rng.range(-3, 4)).collect();
        let a16: Vec<u16> = am.iter().map(|&v| f32_to_f16(v as f32)).collect();
        let abf: Vec<u16> = am.iter().map(|&v| ((v as f32).to_bits() >> 16) as u16).collect();
        let pack = |f: &dyn Fn(i32) -> u16| -> Vec<u32> {
            let mut b = vec![0u32; 8 * 16];
            for d in 0..8 {
                for nn in 0..16 {
                    b[d * 16 + nn] = f(bm[2 * d * 16 + nn]) as u32 | (f(bm[(2 * d + 1) * 16 + nn]) as u32) << 16;
                }
            }
            b
        };
        let b16 = pack(&|v| f32_to_f16(v as f32));
        let bbf = pack(&|v| ((v as f32).to_bits() >> 16) as u16);
        let a8m: Vec<i32> = (0..8 * 32).map(|_| rng.range(-8, 8)).collect();
        let b8m: Vec<i32> = (0..32 * 16).map(|_| rng.range(-8, 8)).collect();
        let a8: Vec<u16> = (0..8 * 16).map(|i| (a8m[(i / 16) * 32 + 2 * (i % 16)] as u8 as u16) | ((a8m[(i / 16) * 32 + 2 * (i % 16) + 1] as u8 as u16) << 8)).collect();
        let mut b8 = vec![0u32; 8 * 16];
        for d in 0..8 {
            for nn in 0..16 {
                for j in 0..4 {
                    b8[d * 16 + nn] |= (b8m[(4 * d + j) * 16 + nn] as u8 as u32) << (8 * j);
                }
            }
        }
        let bufs = (
            DeviceBuffer::from_slice(&q, &a16)?,
            DeviceBuffer::from_slice(&q, &b16)?,
            DeviceBuffer::from_slice(&q, &abf)?,
            DeviceBuffer::from_slice(&q, &bbf)?,
            DeviceBuffer::<f32>::zeroed(&q, 30 * 16)?,
            DeviceBuffer::from_slice(&q, &a8)?,
            DeviceBuffer::from_slice(&q, &b8)?,
            DeviceBuffer::<i32>::zeroed(&q, 9 * 16)?,
        );
        k.args(&[&bufs.0.ptr(), &bufs.1.ptr(), &bufs.2.ptr(), &bufs.3.ptr(), &bufs.4.ptr(), &bufs.5.ptr(), &bufs.6.ptr(), &bufs.7.ptr()])?;
        q.launch(&k, [1, 1, 1])?;
        let c = bufs.4.to_vec(&q)?;
        let c8 = bufs.7.to_vec(&q)?;
        let cref = |r: usize, nn: usize| (0..16).map(|kk| am[r * 16 + kk] * bm[kk * 16 + nn]).sum::<i32>() as f32;
        let mut bad = Vec::new();
        for (base, name) in [(0usize, "f16"), (15, "bf16")] {
            for (rows, off, form) in [(8usize, 0usize, "m8"), (4, 8, "m4"), (2, 12, "m2"), (1, 14, "m1")] {
                for r in 0..rows {
                    for nn in 0..16 {
                        if c[(base + off + r) * 16 + nn] != cref(r, nn) {
                            bad.push(format!("{name}_{form}[{r}][{nn}]"));
                        }
                    }
                }
            }
        }
        for r in 0..9 {
            let rr = if r == 8 { 0 } else { r };
            for nn in 0..16 {
                let w: i32 = (0..32).map(|kk| a8m[rr * 32 + kk] * b8m[kk * 16 + nn]).sum();
                if c8[r * 16 + nn] != w {
                    bad.push(format!("i8[{r}][{nn}]"));
                }
            }
        }
        t.check("DPAS fp16/bf16 (M=8,4,2,1) + int8 (M=8,1)", bad.is_empty(), format!("{} bad {:?}", bad.len(), bad.iter().take(4).collect::<Vec<_>>()));
    }

    // ---- 2D block IO ----
    {
        let k = m.kernel("t_blk2d", [16, 1, 1])?;
        let src: Vec<u16> = (0..64 * 64).map(|i| i as u16).collect();
        let ds = DeviceBuffer::from_slice(&q, &src)?;
        let mut all_ok = true;
        let mut notes = Vec::new();
        for (x0, y0) in [(16i32, 8i32), (0, 0), (48, 40), (56, 60)] {
            let out = DeviceBuffer::<u16>::zeroed(&q, 32 * 16)?;
            let out32 = DeviceBuffer::<u32>::zeroed(&q, 8 * 16)?;
            let dst = DeviceBuffer::<u16>::zeroed(&q, 64 * 64)?;
            k.args(&[&ds.ptr(), &out.ptr(), &out32.ptr(), &dst.ptr(), &x0, &y0])?;
            q.launch(&k, [1, 1, 1])?;
            let (o, o32, d) = (out.to_vec(&q)?, out32.to_vec(&q)?, dst.to_vec(&q)?);
            let at = |r: i32, c: i32| -> u16 { if (0..64).contains(&r) && (0..64).contains(&c) { src[(r * 64 + c) as usize] } else { 0 } };
            let mut bad = 0;
            for l in 0..16i32 {
                for r in 0..16i32 {
                    if o[(r * 16 + l) as usize] != at(y0 + r, x0 + l) {
                        bad += 1;
                    }
                }
                for r in 0..8i32 {
                    if o[((16 + r) * 16 + l) as usize] != at(y0 + r, x0 + l) || o[((24 + r) * 16 + l) as usize] != at(y0 + r, x0 + 16 + l) {
                        bad += 1;
                    }
                }
                for dd in 0..8i32 {
                    let w = at(y0 + 2 * dd, x0 + l) as u32 | (at(y0 + 2 * dd + 1, x0 + l) as u32) << 16;
                    if o32[(dd * 16 + l) as usize] != w {
                        bad += 1;
                    }
                }
            }
            for r in 0..64i32 {
                for c in 0..64i32 {
                    let inb = r >= y0 && r < y0 + 8 && c >= x0 && c < x0 + 16;
                    let want = if inb { at(r, c) } else { 0 };
                    if d[(r * 64 + c) as usize] != want {
                        bad += 1;
                    }
                }
            }
            notes.push(format!("({x0},{y0}):{bad}"));
            all_ok &= bad == 0;
        }
        t.check("2D block read 16r16x1c/8r16x2c/transform, write, prefetch", all_ok, format!("bad per origin {}", notes.join(" ")));
    }

    // ---- conversions ----
    {
        let k = m.kernel("t_cvt", [16, 1, 1])?;
        let mut x: Vec<f32> = vec![
            0.0, -0.0, 1.0, -1.0, 65504.0, 65520.0, 65519.99, 1e-8, 5.96e-8, 2.98e-8, 6.1e-5, 1.0 + 1.0 / 2048.0, 1.0 + 3.0 / 2048.0,
            f32::INFINITY, f32::NEG_INFINITY, 3.0e38, 1.17e-38, 0.1, 0.2, 0.333, -7.77,
        ];
        while x.len() < 1024 {
            let e = rng.range(-30, 30);
            x.push((rng.f32() * 2.0 - 1.0) * 2f32.powi(e));
        }
        let n = x.len();
        let dx = DeviceBuffer::from_slice(&q, &x)?;
        let (h, b, bh) = (DeviceBuffer::<u16>::zeroed(&q, n)?, DeviceBuffer::<u16>::zeroed(&q, n)?, DeviceBuffer::<u16>::zeroed(&q, n)?);
        let back = DeviceBuffer::<f32>::zeroed(&q, 2 * n)?;
        k.args(&[&dx.ptr(), &h.ptr(), &b.ptr(), &bh.ptr(), &back.ptr()])?;
        q.launch(&k, [(n / 16) as u32, 1, 1])?;
        let (hv, bv, bhv, bk) = (h.to_vec(&q)?, b.to_vec(&q)?, bh.to_vec(&q)?, back.to_vec(&q)?);
        let mut bad = Vec::new();
        for i in 0..n {
            if hv[i] != f32_to_f16(x[i]) {
                bad.push(format!("f16({:e}) {:04x} vs {:04x}", x[i], hv[i], f32_to_f16(x[i])));
            }
            if bv[i] != f32_to_bf16(x[i]) {
                bad.push(format!("bf16({:e})", x[i]));
            }
            if bhv[i] != f32_to_bf16(x[i]) {
                bad.push(format!("bf16hw({:e}) {:04x} vs {:04x}", x[i], bhv[i], f32_to_bf16(x[i])));
            }
            if bk[2 * i].to_bits() != f16_to_f32(hv[i]).to_bits() || bk[2 * i + 1].to_bits() != f32::from_bits((bv[i] as u32) << 16).to_bits() {
                bad.push(format!("back({:e})", x[i]));
            }
        }
        t.check("fp32->fp16 RTE, fp16->fp32, bf16 RNE (int + hw)", bad.is_empty(), format!("{} bad {:?}", bad.len(), bad.iter().take(3).collect::<Vec<_>>()));
    }

    // ---- rounding control: no contraction, explicit fma / mad, IEEE vs OpenCL division ----
    {
        let k = m.kernel("t_fp", [64, 1, 1])?;
        let n = 1 << 16;
        let a: Vec<f32> = (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * 3.0).collect();
        let b: Vec<f32> = (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * 3.0 + 0.01).collect();
        let c: Vec<f32> = (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * 3.0).collect();
        let (da, db, dc) = (DeviceBuffer::from_slice(&q, &a)?, DeviceBuffer::from_slice(&q, &b)?, DeviceBuffer::from_slice(&q, &c)?);
        let out = DeviceBuffer::<f32>::zeroed(&q, n * 8)?;
        k.args(&[&da.ptr(), &db.ptr(), &dc.ptr(), &out.ptr()])?;
        // timed through a kernel-timestamp event (fe-ze EventPool), compared with host wall time of launch + sync
        let pool = EventPool::new(&ze, 1)?;
        let ev = pool.event(0)?;
        let t0 = now_ns();
        q.launch_ev(&k, [(n / 64) as u32, 1, 1], Some(&ev))?;
        q.sync()?;
        let host_us = (now_ns() - t0) as f64 / 1e3;
        let dev_us = ev.kernel_ns()? / 1e3;
        t.check("kernel timestamp event", dev_us > 0.0 && dev_us <= host_us, format!("device {dev_us:.1} us, host launch+sync {host_us:.1} us"));
        let o = out.to_vec(&q)?;
        let (mut unf_bad, mut fma_bad, mut mad_eq_fma, mut fused_differs, mut div_bad, mut two_bad) = (0, 0, 0, 0, 0, 0);
        let (mut ocl_max, mut exp_max, mut rsq_max) = (0u64, 0u64, 0u64);
        for i in 0..n {
            let unf = a[i] * b[i] + c[i];
            let fused = a[i].mul_add(b[i], c[i]);
            if unf.to_bits() != fused.to_bits() {
                fused_differs += 1;
            }
            unf_bad += (o[i * 8].to_bits() != unf.to_bits()) as u32;
            fma_bad += (o[i * 8 + 1].to_bits() != fused.to_bits()) as u32;
            mad_eq_fma += (o[i * 8 + 2].to_bits() == fused.to_bits()) as u32;
            div_bad += (o[i * 8 + 3].to_bits() != (a[i] / b[i]).to_bits()) as u32;
            ocl_max = ocl_max.max(ulps(o[i * 8 + 4], a[i] / b[i]));
            exp_max = exp_max.max(ulps(o[i * 8 + 5], (a[i] as f64).exp() as f32));
            rsq_max = rsq_max.max(ulps(o[i * 8 + 6], (1.0 / (b[i].abs() as f64).sqrt()) as f32));
            two_bad += (o[i * 8 + 7].to_bits() != (a[i] * b[i] + c[i] * a[i]).to_bits()) as u32;
        }
        t.check(
            "a*b+c unfused (ContractionOff), a*b+c*a unfused",
            unf_bad == 0 && two_bad == 0 && fused_differs > 0,
            format!("{unf_bad}+{two_bad} differ from two roundings; inputs where fusing changes the bits: {fused_differs}"),
        );
        t.check("fma fused (== correctly rounded fma)", fma_bad == 0, format!("{fma_bad} differ"));
        let div_eq_ocl = (0..n).filter(|&i| o[i * 8 + 3].to_bits() == o[i * 8 + 4].to_bits()).count();
        let div_max = (0..n).map(|i| ulps(o[i * 8 + 3], a[i] / b[i])).max().unwrap();
        println!(
            "  info: mad == fma on {mad_eq_fma}/{n}; Rust `/` (plain OpFDiv): {div_bad} not IEEE, max {div_max} ulp, == OpenCL `/` on {div_eq_ocl}/{n}; OpenCL div max {ocl_max} ulp; exp max {exp_max} ulp; rsqrt max {rsq_max} ulp"
        );
        // IGC compiles an undecorated OpFDiv at OpenCL precision (<= 2.5 ulp), like OpenCL C's `/`, unless the module is
        // built with -cl-fp32-correctly-rounded-divide-sqrt (checked below)
        t.check("Rust a / b == OpenCL a / b (IGC default precision)", div_eq_ocl == n, format!("{} differ", n - div_eq_ocl));
        t.check("OpenCL division within 3 ulp", ocl_max <= 3, format!("{ocl_max} ulp"));
    }

    // -cl-fp32-correctly-rounded-divide-sqrt: Rust `/` becomes the IEEE division
    {
        let mcr = Module::from_spirv(&ze, KX_TEST, "-cl-fp32-correctly-rounded-divide-sqrt")?;
        let k = mcr.kernel("t_fp", [64, 1, 1])?;
        let n = 1 << 16;
        let a: Vec<f32> = (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * 3.0).collect();
        let b: Vec<f32> = (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * 3.0 + 0.01).collect();
        let c: Vec<f32> = vec![0.0; n];
        let (da, db, dc) = (DeviceBuffer::from_slice(&q, &a)?, DeviceBuffer::from_slice(&q, &b)?, DeviceBuffer::from_slice(&q, &c)?);
        let out = DeviceBuffer::<f32>::zeroed(&q, n * 8)?;
        k.args(&[&da.ptr(), &db.ptr(), &dc.ptr(), &out.ptr()])?;
        q.launch(&k, [(n / 64) as u32, 1, 1])?;
        let o = out.to_vec(&q)?;
        let bad = (0..n).filter(|&i| o[i * 8 + 3].to_bits() != (a[i] / b[i]).to_bits()).count();
        let ocl_max = (0..n).map(|i| ulps(o[i * 8 + 4], a[i] / b[i])).max().unwrap();
        t.check("-cl-fp32-correctly-rounded-divide-sqrt: Rust a / b == IEEE", bad == 0, format!("{bad} differ; OpenCL `/` (2.5-ulp decorated) max {ocl_max} ulp"));
    }

    println!("{}", if t.fails == 0 { "ALL PASS" } else { "FAILURES" });
    if t.fails > 0 {
        std::process::exit(1);
    }
    Ok(())
}
