//! GPU reconstruct vs the exllamav3 dumps and the CPU reference:
//!   W_hat fp16 (rotated basis)      bit-exact vs exllamav3 `reconstruct`,
//!   W bf16 [k][n] (original basis)  vs torch f64 -> bf16: ulp histogram (f32 vs f64 math, <= 1 ulp expected),
//!   W bf16 [n][k]                   must be the exact transpose of [k][n].
//! usage: exl3-gpucheck <model dir> <ref dir> [name filter]

use fe_exl3::cpu::Layout;
use fe_cuda::{Context, DeviceBuffer, Stream};
use fe_exl3::gpu::{DeviceExl3, Kernels};
use fe_exl3::half::{bf16_to_f32, bf16_ulp_diff};
use fe_exl3::st::Checkpoint;
use fe_exl3::Exl3Tensor;

fn read_u16(path: &str) -> Option<Vec<u16>> {
    let b = std::fs::read(path).ok()?;
    Some(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (model, refdir) = (&args[1], &args[2]);
    let filter = args.get(3).cloned().unwrap_or_default();
    let _ctx = Context::new(0).unwrap();
    let kern = Kernels::load().unwrap();
    let st = Stream::new().unwrap();
    let ck = Checkpoint::open(model).expect("checkpoint");
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{refdir}/manifest.json")).unwrap()).unwrap();
    let scratch = DeviceBuffer::<f32>::alloc(64 << 20).unwrap();
    let mut fails = 0;
    for e in man.as_array().unwrap() {
        let tensor = e["tensor"].as_str().unwrap();
        let name = e["name"].as_str().unwrap();
        if tensor.starts_with("model.visual") || !name.contains(&filter) {
            continue;
        }
        let nslice = e["nslice"].as_u64().unwrap() as usize;
        let Some(ref_hat) = read_u16(&format!("{refdir}/{name}.w_hat_f16.bin")) else { continue };
        let ref_w = read_u16(&format!("{refdir}/{name}.w_bf16.bin")).unwrap();
        let t = Exl3Tensor::load(&ck, tensor).expect("load");
        let d = DeviceExl3::upload(&t).unwrap();
        let (k, n) = (t.k, t.n);

        let hat = DeviceBuffer::<u16>::alloc(k * n).unwrap();
        kern.reconstruct_hat_f16(&st, &d.view, hat.ptr()).unwrap();
        st.sync().unwrap();
        let hat_v: Vec<u16> = hat.to_vec().unwrap();
        drop(hat);
        let mut hat_bad = 0;
        for r in 0..k {
            for c in 0..nslice {
                if hat_v[r * n + c] != ref_hat[r * nslice + c] {
                    hat_bad += 1;
                }
            }
        }
        drop(hat_v);

        let out = DeviceBuffer::<u16>::alloc(k * n).unwrap();
        let ms = st
            .time(|s| {
                kern.reconstruct_bf16(s, &d.view, out.ptr(), Layout::KN, scratch.ptr(), scratch.bytes()).unwrap();
                Ok(())
            })
            .unwrap();
        let w: Vec<u16> = out.to_vec().unwrap();
        // ulp histogram; a >1 ulp difference is acceptable only as f32 cancellation noise near zero, i.e. an
        // absolute error below 1e-4 of the tensor's rms (counted as "~0")
        let rms = (ref_w.iter().map(|&b| (bf16_to_f32(b) as f64).powi(2)).sum::<f64>() / ref_w.len() as f64).sqrt();
        let mut hist = [0usize; 4];
        let mut near_zero = 0;
        let mut max_rel = 0f64;
        for r in 0..k {
            for c in 0..nslice {
                let (a, b) = (w[r * n + c], ref_w[r * nslice + c]);
                let dd = bf16_ulp_diff(a, b) as usize;
                let err = (bf16_to_f32(a) as f64 - bf16_to_f32(b) as f64).abs() / rms;
                max_rel = max_rel.max(err);
                if dd > 1 && err < 1e-4 {
                    near_zero += 1;
                    continue;
                }
                hist[dd.min(3)] += 1;
            }
        }
        // [n][k] must be the exact transpose
        kern.reconstruct_bf16(&st, &d.view, out.ptr(), Layout::NK, scratch.ptr(), scratch.bytes()).unwrap();
        st.sync().unwrap();
        let wt: Vec<u16> = out.to_vec().unwrap();
        let mut tbad = 0usize;
        for r in 0..k {
            for c in 0..n {
                if wt[c * k + r] != w[r * n + c] {
                    tbad += 1;
                }
            }
        }
        let ok = hat_bad == 0 && hist[2] + hist[3] == 0 && tbad == 0;
        if !ok {
            fails += 1;
        }
        let gbs = (t.trellis_bytes() + k * n * 2) as f64 / (ms as f64 * 1e-3) / 1e9;
        println!(
            "{} {:<40} {}b {:>5}x{:<6} w_hat mism {:>2} | W bf16 ulp0 {} ulp1 {} ulp2 {} ulp3+ {} (~0: {}) max|d|/rms {:.1e} | NK transpose mism {} | {:.2} ms ({:.0} GB/s out+in)",
            if ok { "PASS" } else { "FAIL" },
            name,
            t.bits,
            k,
            nslice,
            hat_bad,
            hist[0],
            hist[1],
            hist[2],
            hist[3],
            near_zero,
            max_rel,
            tbad,
            ms,
            gbs
        );
    }
    if fails > 0 {
        println!("{fails} FAILED");
        std::process::exit(1);
    }
    println!("all PASS");
}
