//! Decode GEMV (rot_in + trellis GEMV) vs the f64 CPU reference on the decoded W_hat (bit-exact vs
//! exllamav3, see exl3-refcheck), one tensor per kind x bit width, rows 1..16.
//! rot_in is checked against f64 -> fp16 (a rare 1-ulp tie flip allowed); the GEMV reference then runs on the
//! GPU's own fp16 rotated input, so what is left is fp32 accumulation order.
//! usage: exl3-gemvcheck <model dir> [name filter] [plan override sk,wk]

use std::collections::BTreeMap;

use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_exl3::cpu::{gemv_ref, rotate_input_ref, w_hat_f16};
use fe_exl3::gpu::{DeviceExl3, Kernels};
use fe_exl3::half::{bf16_to_f32, f32_to_bf16};
use fe_exl3::linear::{default_blocks, Dtype, Linear, Plan, Workspace};
use fe_exl3::st::Checkpoint;
use fe_exl3::{exl3_bases, Exl3Tensor};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Roughly normal, with a few large outliers like real activations.
    fn act(&mut self) -> f64 {
        let g = (0..4).map(|_| self.next()).sum::<f64>() - 2.0;
        if self.next() < 0.002 { g * 30.0 } else { g }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = &args[1];
    let filter = args.get(2).cloned().unwrap_or_default();
    let forced = args.get(3).map(|s| Plan::parse(s).expect("plan sk,wk[,ver[,occ]]"));
    let ctx = Context::new(0).unwrap();
    let blocks = default_blocks(ctx.sm_count().unwrap() as u32);
    let kern = Kernels::load().unwrap();
    let s = Stream::new().unwrap();
    let ck = Checkpoint::open(model).unwrap();

    // one tensor per (kind, bits)
    let mut picked: BTreeMap<(String, u32), String> = BTreeMap::new();
    for b in exl3_bases(&ck) {
        if b.starts_with("model.visual") || !b.contains(&filter) {
            continue;
        }
        let bits = ck.info(&format!("{b}.trellis")).unwrap().shape[2] as u32 / 16;
        let kind: String = b.split('.').filter(|p| p.parse::<u32>().is_err()).collect::<Vec<_>>().join(".");
        picked.entry((kind, bits)).or_insert(b);
    }
    let rows_list = [1u32, 2, 5, 8, 9, 16, 17, 33, 64];
    let mr = 64usize;
    let mut fails = 0;
    for ((_, _), base) in picked {
        let t = Exl3Tensor::load(&ck, &base).unwrap();
        let (k, n) = (t.k, t.n);
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (k * 31 + n) as u64);
        let xb: Vec<u16> = (0..mr * k).map(|_| f32_to_bf16(rng.act() as f32)).collect();
        let x64: Vec<f64> = xb.iter().map(|&h| bf16_to_f32(h) as f64).collect();
        let t0 = std::time::Instant::now();
        let nc = n.min(8192); // checked columns (lm_head: the first 8192)
        let hat = w_hat_f16(&t, 0, nc);
        let xh_ref = rotate_input_ref(&t, &x64, mr);
        let cpu_s = t0.elapsed().as_secs_f64();
        let d = DeviceExl3::upload_as(&t, std::env::var("FE_EXL3_STORED").is_err()).unwrap();
        let lin = match forced {
            Some(p) => Linear::with_plan(&kern, d, p),
            None => Linear::new(&kern, d, blocks),
        }
        .unwrap();
        let ws = Workspace::new(&[&lin], mr as u32).unwrap();
        let xd = DeviceBuffer::from_slice(&xb).unwrap();
        let yd = DeviceBuffer::<f32>::alloc(mr * n).unwrap();
        let yb = DeviceBuffer::<u16>::alloc(mr * n).unwrap();
        let mut line = format!("{:<42} {}b {:>5}x{:<6} plan sk{} wk{} |", base.replace("model.language_model.", ""), t.bits, k, n, lin.plan.sk, lin.plan.wk);
        let mut ok = true;
        // rot_in alone vs the f64 reference
        lin.rotate_input(&kern, &s, xd.ptr(), Dtype::Bf16, mr as u32, &ws).unwrap();
        s.sync().unwrap();
        let xh_gpu: Vec<u16> = ws.xh.to_vec().unwrap()[..mr * k].to_vec();
        let xh_bad = xh_gpu.iter().zip(&xh_ref).filter(|(a, b)| fe_exl3::half::f16_to_f64(**a) != fe_exl3::half::f16_to_f64(**b)).count();
        let xh_far = xh_gpu.iter().zip(&xh_ref).filter(|(a, b)| (**a as i32 - **b as i32).abs() > 1).count();
        if xh_far > 0 {
            ok = false;
        }
        line += &format!(" rot_in 1ulp {xh_bad} >1ulp {xh_far} |");
        let yref = gemv_ref(&t, &hat, nc, &xh_gpu, mr);
        for &rows in &rows_list {
            s.memset_d32(yd.ptr(), 0x7fc0_0000, mr * n).unwrap(); // NaN: unwritten outputs show up
            lin.forward(&kern, &s, xd.ptr(), Dtype::Bf16, yd.ptr(), Dtype::F32, rows, &ws).unwrap();
            lin.forward(&kern, &s, xd.ptr(), Dtype::Bf16, yb.ptr(), Dtype::Bf16, rows, &ws).unwrap();
            s.sync().unwrap();
            let y: Vec<f32> = yd.to_vec().unwrap();
            let ybv: Vec<u16> = yb.to_vec().unwrap();
            let r = rows as usize * nc;
            let rms = (yref[..r].iter().map(|v| v * v).sum::<f64>() / r as f64).sqrt();
            let (mut maxd, mut sq, mut bad_bf) = (0f64, 0f64, 0usize);
            for row in 0..rows as usize {
                for c in 0..nc {
                    let (i, ri) = (row * n + c, row * nc + c);
                    let dd = (y[i] as f64 - yref[ri]).abs();
                    maxd = if dd.is_nan() { f64::INFINITY } else { maxd.max(dd) };
                    sq += dd * dd;
                    // bf16 output: within one bf16 rounding of the exact value (plus fp32 noise)
                    let e = (bf16_to_f32(ybv[i]) as f64 - yref[ri]).abs();
                    if !(e <= yref[ri].abs() * (1.0 / 256.0) + 1e-5 * rms) {
                        bad_bf += 1;
                    }
                }
            }
            // rows past `rows` must be untouched (still NaN)
            if rows < mr as u32 && !y[rows as usize * n].is_nan() {
                bad_bf += 1_000_000;
            }
            let rel_max = maxd / rms;
            let rel_rms = (sq / r as f64).sqrt() / rms;
            if !(rel_max < 1e-4) || bad_bf > 0 {
                ok = false;
            }
            line += &format!(" m{rows} {rel_max:.1e}/{rel_rms:.1e}{}", if bad_bf > 0 { format!(" bf16bad {bad_bf}") } else { String::new() });
        }
        // row invariance: the first 5 rows' f32 bits at m = 5, 8, 16, 64 are identical
        {
            let mut first: Option<Vec<f32>> = None;
            let mut diff = 0;
            for rows in [5u32, 8, 16, 64] {
                lin.forward(&kern, &s, xd.ptr(), Dtype::Bf16, yd.ptr(), Dtype::F32, rows, &ws).unwrap();
                s.sync().unwrap();
                let y: Vec<f32> = yd.to_vec().unwrap()[..5 * n].to_vec();
                match &first {
                    None => first = Some(y),
                    Some(f) => diff += f.iter().zip(&y).filter(|(a, b)| a.to_bits() != b.to_bits()).count(),
                }
            }
            if diff > 0 {
                ok = false;
            }
            line += &format!(" | rowinv {}", if diff == 0 { "ok".to_string() } else { format!("DIFF {diff}") });
        }
        // fused residual add: y (bf16, prefilled with 1.5) += x @ W, 5 rows
        {
            let rows = 5usize;
            let pre = vec![0x3fc0u16; mr * n];
            let ya = DeviceBuffer::from_slice(&pre).unwrap();
            lin.forward(&kern, &s, xd.ptr(), Dtype::Bf16, ya.ptr(), Dtype::Bf16Add, rows as u32, &ws).unwrap();
            s.sync().unwrap();
            let yav: Vec<u16> = ya.to_vec().unwrap();
            let rms = (yref[..rows * nc].iter().map(|v| v * v).sum::<f64>() / (rows * nc) as f64).sqrt();
            let mut bad = 0;
            for row in 0..rows {
                for c in 0..nc {
                    let want = 1.5 + yref[row * nc + c];
                    let e = (bf16_to_f32(yav[row * n + c]) as f64 - want).abs();
                    if !(e <= want.abs() / 256.0 + 1e-5 * rms) {
                        bad += 1;
                    }
                }
            }
            if yav[rows * n] != 0x3fc0 {
                bad += 1_000_000;
            }
            if bad > 0 {
                ok = false;
            }
            line += &format!(" | add {}", if bad == 0 { "ok".to_string() } else { format!("BAD {bad}") });
        }
        if !ok {
            fails += 1;
        }
        println!("{} {line}  (cpu ref {cpu_s:.1}s)", if ok { "PASS" } else { "FAIL" });
    }
    println!("columns: rows m: max|y - ref| / rms(ref) / rms error / rms(ref); bf16 output checked to 1 rounding");
    if fails > 0 {
        println!("{fails} FAILED");
        std::process::exit(1);
    }
    println!("all PASS");
}
