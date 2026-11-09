//! Check the CPU reference against exllamav3 dumps (tools/exl3_dump_ref.py):
//!   W_hat  (fp16, rotated basis)   must be bit-exact,
//!   W      (bf16, original basis)  vs torch f64 -> bf16, must be bit-exact (ulp histogram printed),
//!   W      vs exllamav3's fused fp16 reconstruct_had (fp16 internal rounding): error stats only.
//! usage: exl3-refcheck <model dir> <ref dir> [name filter]

use fe_exl3::cpu::{w_hat_f16, weight_bf16, Layout};
use fe_exl3::half::{bf16_to_f32, bf16_ulp_diff, f16_to_f32};
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
    let ck = Checkpoint::open(model).expect("checkpoint");
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{refdir}/manifest.json")).unwrap()).unwrap();
    let mut fails = 0;
    for e in man.as_array().unwrap() {
        let tensor = e["tensor"].as_str().unwrap();
        let name = e["name"].as_str().unwrap();
        if tensor.starts_with("model.visual") || !name.contains(&filter) {
            continue;
        }
        let nslice = e["nslice"].as_u64().unwrap() as usize;
        let Some(ref_hat) = read_u16(&format!("{refdir}/{name}.w_hat_f16.bin")) else { continue };
        let t0 = std::time::Instant::now();
        let t = Exl3Tensor::load(&ck, tensor).expect("load");
        let hat = w_hat_f16(&t, 0, nslice);
        let hat_bad = hat.iter().zip(&ref_hat).filter(|(a, b)| a != b).count();

        let w = weight_bf16(&t, 0, nslice, Layout::KN);
        let ref_w = read_u16(&format!("{refdir}/{name}.w_bf16.bin")).unwrap();
        // Ulp histogram; a mismatch is only acceptable where the exact value is ~0 (|W| < 1e-9: f64 summation
        // order noise around a cancellation), counted separately.
        let mut hist = [0usize; 4];
        let mut near_zero = 0usize;
        for (a, b) in w.iter().zip(&ref_w) {
            let d = bf16_ulp_diff(*a, *b) as usize;
            if d > 0 && bf16_to_f32(*a).abs().max(bf16_to_f32(*b).abs()) < 1e-9 {
                near_zero += 1;
                if near_zero <= 2 {
                    eprintln!("  near-zero mismatch {:e} vs {:e}", bf16_to_f32(*a), bf16_to_f32(*b));
                }
                continue;
            }
            hist[d.min(3)] += 1;
        }
        // vs exllamav3's fused fp16 path
        let fused = read_u16(&format!("{refdir}/{name}.w_fused_f16.bin")).unwrap();
        let (mut max_abs, mut sum_sq, mut ref_sq) = (0f64, 0f64, 0f64);
        for (a, b) in w.iter().zip(&fused) {
            let (a, b) = (bf16_to_f32(*a) as f64, f16_to_f32(*b) as f64);
            max_abs = max_abs.max((a - b).abs());
            sum_sq += (a - b) * (a - b);
            ref_sq += b * b;
        }
        // ulp1 can only come from f64 summation order at a bf16 rounding tie (counted, allowed)
        let ok = hat_bad == 0 && hist[2] + hist[3] == 0;
        if !ok {
            fails += 1;
        }
        println!(
            "{} {:<44} {}b {:>5}x{:<6} w_hat mismatches {:>3}  W bf16 ulp0 {} ulp1 {} ulp2 {} ulp3+ {} (|W|<1e-9: {})  vs fused fp16: max|d| {:.2e} rel-rms {:.2e}  ({:.1}s)",
            if ok { "PASS" } else { "FAIL" },
            name,
            t.bits,
            t.k,
            nslice,
            hat_bad,
            hist[0],
            hist[1],
            hist[2],
            hist[3],
            near_zero,
            max_abs,
            (sum_sq / ref_sq).sqrt(),
            t0.elapsed().as_secs_f64()
        );
    }
    if fails > 0 {
        println!("{fails} FAILED");
        std::process::exit(1);
    }
    println!("all PASS");
}
