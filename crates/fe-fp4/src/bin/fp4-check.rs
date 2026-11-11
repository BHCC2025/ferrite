//! Correctness of the k-fp4 quantizers + GEMMs against the CPU reference on real checkpoint weights.
//!
//! `fp4-check <model_dir> [layer=3] [m=200]`: for the layer's NVFP4 MLP (gate|up fused, down) and FP8
//! attention projections, checks (1) GPU activation quantization bit-exact vs the CPU quantizer, (2) the
//! GPU GEMM vs an f64 CPU GEMM over the *same* quantized operands (sampled rows, all columns), (3) the
//! residual epilogue, (4) cuBLASLt on the same inputs, and reports the quantization error vs bf16 math.
use std::path::PathBuf;
use anyhow::{ensure, Result};
use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_fp4::format::*;
use fe_fp4::{lt::Lt, Checkpoint, Kernels, Kind, PromptLinear, Workspace};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
}

fn check(kern: &Kernels, lt: &Lt, s: &Stream, ck: &Checkpoint, parts: &[&str], m: usize) -> Result<bool> {
    let t0 = std::time::Instant::now();
    let lin = PromptLinear::load_opts(ck, parts, true)?;
    let (n, k) = (lin.n, lin.k);
    let fp4 = lin.kind == Kind::Nvfp4;
    let amax_cal = if fp4 { 448.0 * 6.0 } else { 448.0 } / lin.g;
    println!("== {} [{:?}] n {n} k {k} m {m}: g {:.4} (calibrated amax {:.2}), alpha {:?} (load {:.1}s)", lin.name,
        lin.kind, lin.g, amax_cal, lin.parts.iter().map(|p| p.alpha).collect::<Vec<_>>(), t0.elapsed().as_secs_f32());

    // activations: gaussian with a few outliers, amax around the calibrated range
    let mut rng = Rng(0x9e3779b97f4a7c15 ^ (n * 31 + k) as u64);
    let sigma = amax_cal / 10.0;
    let x: Vec<u16> = (0..m * k)
        .map(|i| {
            let v = rng.normal() * sigma * if i % 997 == 0 { 6.0 } else { 1.0 };
            f32_to_bf16(v)
        })
        .collect();
    let xd = DeviceBuffer::from_slice(&x)?;
    let ws = Workspace::new(m, k)?;
    let y = DeviceBuffer::<u16>::zeroed(m * n)?;
    lin.forward(kern, s, &ws, y.ptr(), n, xd.ptr(), k, m, None)?;
    s.sync()?;
    let yg = y.to_vec()?;

    // (0) FP8: the block-scaled tensor-core form (FE_FP8_BS) must give the plain form's bits
    if !fp4 {
        let y2 = DeviceBuffer::<u16>::zeroed(m * n)?;
        let f = if kern.fp8_bs { &kern.gemm8s3 } else { &kern.gemm8s3_bs };
        unsafe {
            s.launch(f, (m.div_ceil(128) * (n / 128)) as u32, 256, fe_fp4::FP8_SMEM_S3, (y2.ptr(), ws.codes.ptr(), lin.w.ptr(),
                lin.alpha.ptr(), fe_cuda::DevPtr::NULL, m as u32, n as u32, k as u32, n as u32, 8u32))?;
        }
        s.sync()?;
        let y2 = y2.to_vec()?;
        let diff = yg.iter().zip(&y2).filter(|(a, b)| a != b).count();
        println!("  FP8 plain vs block-scaled form: {diff} of {} outputs differ", yg.len());
    }

    // (1) quantizer bit-exactness
    let row_bytes = if fp4 { k / 2 } else { k };
    let mut codes = vec![0u8; m * row_bytes];
    fe_cuda::memcpy_dtoh(&mut codes, ws.codes.ptr())?;
    let sfa = if fp4 {
        let rp = m.div_ceil(128) * 128;
        let mut t = vec![0u8; rp * k / 16];
        fe_cuda::memcpy_dtoh(&mut t, ws.sf.ptr())?;
        untile_scales(&t, m, k / 16)
    } else {
        Vec::new()
    };
    let (mut bad_c, mut bad_s) = (0usize, 0usize);
    for r in 0..m {
        let xr = &x[r * k..(r + 1) * k];
        if fp4 {
            let (c, sf) = quant_fp4_row(xr, lin.g);
            bad_c += c.iter().zip(&codes[r * row_bytes..(r + 1) * row_bytes]).filter(|(a, b)| a != b).count();
            bad_s += sf.iter().zip(&sfa[r * k / 16..(r + 1) * k / 16]).filter(|(a, b)| a != b).count();
        } else {
            let c = quant_fp8_row(xr, lin.g);
            bad_c += c.iter().zip(&codes[r * row_bytes..(r + 1) * row_bytes]).filter(|(a, b)| a != b).count();
        }
    }
    println!("  quant: {bad_c} / {} code bytes differ, {bad_s} scale bytes differ", m * row_bytes);

    // (2) GEMM vs CPU over the same quantized operands
    let mut rows: Vec<usize> = vec![0, 1, 17, 127, 128, m / 2, m - 1];
    rows.retain(|&r| r < m);
    rows.dedup();
    let mut wcodes = vec![0u8; n * row_bytes];
    fe_cuda::memcpy_dtoh(&mut wcodes, lin.w.ptr())?;
    let wsf = if fp4 {
        let mut t = vec![0u8; n * k / 16];
        fe_cuda::memcpy_dtoh(&mut t, lin.sw.as_ref().unwrap().ptr())?;
        untile_scales(&t, n, k / 16)
    } else {
        Vec::new()
    };
    let mut wf = vec![0.0f32; n * k];
    for c in 0..n {
        let dst = &mut wf[c * k..(c + 1) * k];
        if fp4 {
            dequant_fp4_row(&wcodes[c * row_bytes..(c + 1) * row_bytes], &wsf[c * k / 16..(c + 1) * k / 16], 1.0, dst);
        } else {
            dequant_fp8_row(&wcodes[c * row_bytes..(c + 1) * row_bytes], 1.0, dst);
        }
    }
    let aq: Vec<Vec<f32>> = rows
        .iter()
        .map(|&r| {
            let mut v = vec![0.0f32; k];
            if fp4 {
                dequant_fp4_row(&codes[r * row_bytes..(r + 1) * row_bytes], &sfa[r * k / 16..(r + 1) * k / 16], 1.0, &mut v);
            } else {
                dequant_fp8_row(&codes[r * row_bytes..(r + 1) * row_bytes], 1.0, &mut v);
            }
            v
        })
        .collect();
    let refq = ref_gemm(&aq, &wf, n, k);
    let mut alpha = vec![0.0f32; n];
    for p in &lin.parts {
        alpha[p.n0..p.n0 + p.n].fill(p.alpha);
    }
    let mut worst = 0.0f32;
    for (i, &r) in rows.iter().enumerate() {
        let scale = refq[i].iter().zip(&alpha).fold(0.0f32, |a, (v, al)| a.max((v * al).abs())).max(1e-20);
        for c in 0..n {
            let e = (bf16_to_f32(yg[r * n + c]) - refq[i][c] * alpha[c]).abs() / scale;
            worst = worst.max(e);
        }
    }
    let gemm_ok = worst < 8e-3;
    println!("  gemm vs CPU (same quantized inputs, rows {rows:?}): max err / row max = {worst:.2e} {}",
        if gemm_ok { "OK" } else { "FAIL" });

    // (3) quantization error vs bf16 x . dequant(W) with the true global scales
    let xf: Vec<Vec<f32>> = rows.iter().map(|&r| x[r * k..(r + 1) * k].iter().map(|&h| bf16_to_f32(h)).collect()).collect();
    let refb = ref_gemm(&xf, &wf, n, k);
    let (mut se, mut ss) = (0.0f64, 0.0f64);
    for (i, &r) in rows.iter().enumerate() {
        for c in 0..n {
            let wglob = alpha[c] * lin.g; // weight global scale alone
            let t = refb[i][c] * wglob;
            se += ((bf16_to_f32(yg[r * n + c]) - t) as f64).powi(2);
            ss += (t as f64).powi(2);
        }
    }
    println!("  quantization error vs bf16 activations: rel rms {:.4}", (se / ss).sqrt());

    // (4) residual epilogue: y2 = bf16(bf16(x.W) + res), in place
    let res: Vec<u16> = (0..m * n).map(|_| f32_to_bf16(rng.normal())).collect();
    let y2 = DeviceBuffer::from_slice(&res)?;
    lin.forward(kern, s, &ws, y2.ptr(), n, xd.ptr(), k, m, Some(y2.ptr()))?;
    s.sync()?;
    let y2h = y2.to_vec()?;
    let mut bad_res = 0;
    for i in 0..m * n {
        let want = f32_to_bf16(bf16_to_f32(yg[i]) + bf16_to_f32(res[i]));
        if y2h[i] != want {
            bad_res += 1;
        }
    }
    println!("  residual epilogue: {bad_res} / {} differ", m * n);

    // (4b) canary: strided output (ldy = n + 72) inside a larger buffer, every element outside rows < m,
    // cols < n must stay untouched (residual path too)
    let ldy = n + 72;
    let canary = 0xABCDu16;
    let yc = DeviceBuffer::from_slice(&vec![canary; (m + 3) * ldy])?;
    lin.forward(kern, s, &ws, yc.ptr(), ldy, xd.ptr(), k, m, None)?;
    lin.forward(kern, s, &ws, yc.ptr(), ldy, xd.ptr(), k, m, Some(yc.ptr()))?;
    s.sync()?;
    let ych = yc.to_vec()?;
    let mut clobbered = 0;
    for (i, &v) in ych.iter().enumerate() {
        let (r, c) = (i / ldy, i % ldy);
        if (r >= m || c >= n) && v != canary {
            clobbered += 1;
        }
    }
    println!("  canary outside the [m][n] block (ldy {ldy}): {clobbered} elements written");

    // (5) cuBLASLt on the same activations
    let ylt = DeviceBuffer::<u16>::zeroed(m * n)?;
    let mut plans = Vec::new();
    lin.forward_lt(kern, lt, &mut plans, s, &ws, ylt.ptr(), n, xd.ptr(), k, m)?;
    s.sync()?;
    let yl = ylt.to_vec()?;
    let (mut d2, mut n2) = (0.0f64, 0.0f64);
    for i in 0..m * n {
        let (a, b) = (bf16_to_f32(yg[i]) as f64, bf16_to_f32(yl[i]) as f64);
        d2 += (a - b).powi(2);
        n2 += b * b;
    }
    let lt_rel = (d2 / n2).sqrt();
    println!("  vs cuBLASLt: rel rms diff {lt_rel:.2e}");
    Ok(gemm_ok && bad_res == 0 && bad_s == 0 && bad_c * 1000 < m * row_bytes && lt_rel < 1e-2 && clobbered == 0)
}

/// Fused gate|up + SwiGLU + NVFP4 quant vs the unfused GPU path (interleaved GEMM -> bf16 -> swiglu_quant):
/// bit-exact codes and scales; plus the dequantized act vs a CPU SwiGLU of the bf16 GEMM output.
fn check_swiglu(kern: &Kernels, s: &Stream, ck: &Checkpoint, layer: usize, m: usize) -> Result<bool> {
    let p = format!("model.language_model.layers.{layer}.mlp");
    let gu = PromptLinear::load_swiglu(ck, &format!("{p}.gate_proj"), &format!("{p}.up_proj"))?;
    let down = PromptLinear::load(ck, &[format!("{p}.down_proj").as_str()])?;
    let (n, k, inter) = (gu.n, gu.k, gu.n / 2);
    println!("== {} n {n} k {k} m {m} -> act for {} (g {:.3})", gu.name, down.name, down.g);
    let mut rng = Rng(0x51ed270b27f1b1f5);
    let sigma = 448.0 * 6.0 / gu.g / 10.0;
    let x: Vec<u16> = (0..m * k).map(|_| f32_to_bf16(rng.normal() * sigma)).collect();
    let xd = DeviceBuffer::from_slice(&x)?;
    let ws = Workspace::new(m, k)?;
    let (act_a, act_b) = (Workspace::new(m + 130, inter)?, Workspace::new(m + 130, inter)?);
    let cb_len = act_a.codes.len();
    fe_cuda::memcpy_htod(act_a.codes.ptr(), &vec![0x5Au8; cb_len])?;
    gu.forward_swiglu(kern, s, &ws, &act_a, xd.ptr(), k, m, down.g)?;
    let y = DeviceBuffer::<u16>::zeroed(m * n)?;
    gu.forward(kern, s, &ws, y.ptr(), n, xd.ptr(), k, m, None)?;
    kern.swiglu_quant(s, &act_b, y.ptr(), m, n, down.g)?;
    s.sync()?;
    let rp = m.div_ceil(128) * 128;
    let (mut ca, mut cb) = (vec![0u8; m * inter / 2], vec![0u8; m * inter / 2]);
    fe_cuda::memcpy_dtoh(&mut ca, act_a.codes.ptr())?;
    fe_cuda::memcpy_dtoh(&mut cb, act_b.codes.ptr())?;
    let (mut sa, mut sb) = (vec![0u8; rp * inter / 16], vec![0u8; rp * inter / 16]);
    fe_cuda::memcpy_dtoh(&mut sa, act_a.sf.ptr())?;
    fe_cuda::memcpy_dtoh(&mut sb, act_b.sf.ptr())?;
    let (sa, sb) = (untile_scales(&sa, m, inter / 16), untile_scales(&sb, m, inter / 16));
    let mut tail = vec![0u8; cb_len - m * inter / 2];
    fe_cuda::memcpy_dtoh(&mut tail, act_a.codes.ptr().byte_add(m * inter / 2))?;
    let act_clobbered = tail.iter().filter(|&&b| b != 0x5A).count();
    println!("  act codes beyond row {m}: {act_clobbered} bytes written");
    let dc = ca.iter().zip(&cb).filter(|(a, b)| a != b).count();
    let ds = sa.iter().zip(&sb).filter(|(a, b)| a != b).count();
    println!("  fused vs unfused: {dc} / {} code bytes, {ds} / {} scale bytes differ", ca.len(), sa.len());
    // CPU: act from the interleaved bf16 y (exact exp), dequantized fused act vs it
    let yh = y.to_vec()?;
    let (mut se, mut ss) = (0.0f64, 0.0f64);
    let mut deq = vec![0.0f32; inter];
    for r in [0, m / 2, m - 1] {
        dequant_fp4_row(&ca[r * inter / 2..(r + 1) * inter / 2], &sa[r * inter / 16..(r + 1) * inter / 16], 1.0 / down.g, &mut deq);
        for f in 0..inter {
            let col = (f / 8) * 16 + f % 8;
            let (g, u) = (bf16_to_f32(yh[r * n + col]), bf16_to_f32(yh[r * n + col + 8]));
            let sl = bf16_to_f32(f32_to_bf16(g / (1.0 + (-g).exp())));
            let a = bf16_to_f32(f32_to_bf16(sl * u));
            se += ((deq[f] - a) as f64).powi(2);
            ss += (a as f64).powi(2);
        }
    }
    let rel = (se / ss).sqrt();
    println!("  dequantized act vs CPU SwiGLU of the bf16 GEMM output: rel rms {rel:.4} (NVFP4 rounding ~0.1)");
    Ok(dc == 0 && ds == 0 && rel < 0.2 && act_clobbered == 0)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(PathBuf::from).unwrap_or_else(fe_site::nvfp4_dir);
    let layer: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let m: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    let _ctx = Context::new(0)?;
    let kern = Kernels::load()?;
    let lt = Lt::new()?;
    let s = Stream::new()?;
    let ck = Checkpoint::open(&dir)?;
    let p = format!("model.language_model.layers.{layer}");
    let attn = if ck.has(&format!("{p}.self_attn.q_proj.weight")) {
        vec![
            vec![format!("{p}.self_attn.q_proj"), format!("{p}.self_attn.k_proj"), format!("{p}.self_attn.v_proj")],
            vec![format!("{p}.self_attn.o_proj")],
        ]
    } else {
        vec![
            vec![format!("{p}.linear_attn.in_proj_qkv"), format!("{p}.linear_attn.in_proj_z")],
            vec![format!("{p}.linear_attn.out_proj")],
        ]
    };
    let mut sets = vec![vec![format!("{p}.mlp.gate_proj"), format!("{p}.mlp.up_proj")], vec![format!("{p}.mlp.down_proj")]];
    sets.extend(attn);
    let mut ok = true;
    for set in &sets {
        let parts: Vec<&str> = set.iter().map(|s| s.as_str()).collect();
        ok &= check(&kern, &lt, &s, &ck, &parts, m)?;
    }
    ok &= check_swiglu(&kern, &s, &ck, layer, m)?;
    println!("{}", if ok { "ALL OK" } else { "FAILURES" });
    ensure!(ok, "fp4-check failed");
    Ok(())
}
