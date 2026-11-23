//! The draft head's "fold" tables: `lm_head @ markov_head.hidden_out.weight` and `lm_head @ markov_head.hidden_out.bias`, computed in
//! float64 and stored as bf16 (the unembedding folded through the head's output projection, so the Markov head scores the whole
//! vocabulary with a [vocab, 256] GEMV). They are derived from two tensors of the release, so nothing is downloaded for them: the
//! first start computes them (a minute or so of CPU time on a few cores) and keeps them next to the model as `fold.safetensors`
//! (or in `INK_FOLD`, when set); later starts read that file. Only draft quality depends on them, never the generated text.
use crate::ckpt::Ckpt;
use crate::{D, VOCAB};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const R: usize = 256;

/// Where the fold file lives: `INK_FOLD`, else `<model_dir>/fold.safetensors`.
pub fn path(model_dir: &Path) -> PathBuf {
    match std::env::var("INK_FOLD") {
        Ok(p) => PathBuf::from(p),
        Err(_) => model_dir.join("fold.safetensors"),
    }
}

/// The fold file, computed first when it is not there. When the folder cannot be written the file goes to the temporary folder.
pub fn ensure(model_dir: &Path) -> Result<PathBuf> {
    let p = path(model_dir);
    if p.exists() {
        return Ok(p);
    }
    eprintln!("ink: computing the draft head's fold tables (once; kept as {})", p.display());
    let (w, b) = compute(model_dir)?;
    match write(&p, &w, &b) {
        Ok(()) => Ok(p),
        Err(e) => {
            let t = std::env::temp_dir().join("ferrite-ink-fold.safetensors");
            eprintln!("ink: cannot write {} ({e:#}); using {}", p.display(), t.display());
            write(&t, &w, &b)?;
            Ok(t)
        }
    }
}

fn bf16_to_f64(h: u16) -> f64 {
    f32::from_bits((h as u32) << 16) as f64
}

/// float64 -> bf16, round to nearest even straight from the double (as torch's `.to(bfloat16)` does).
fn f64_to_bf16(x: f64) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 63) as u16) << 15;
    let exp = ((bits >> 52) & 0x7ff) as i32 - 1023;
    if x == 0.0 || exp < -126 || exp > 127 {
        let f = (x as f32).to_bits();
        let r = f.wrapping_add(0x7fff + ((f >> 16) & 1)) >> 16;
        return r as u16;
    }
    let mant = bits & ((1u64 << 52) - 1);
    let keep = (mant >> 45) as u32;
    let rem = mant & ((1u64 << 45) - 1);
    let half = 1u64 << 44;
    let up = rem > half || (rem == half && keep & 1 == 1);
    let v = (((exp + 127) as u32) << 7) + keep + up as u32;
    sign | v as u16
}

/// (fold_weight [VOCAB][256] bf16, fold_bias [VOCAB] bf16)
pub fn compute(model_dir: &Path) -> Result<(Vec<u16>, Vec<u16>)> {
    let main = Ckpt::open(&model_dir.join("model-00001-of-00001.safetensors")).context("main checkpoint")?;
    let mtp = Ckpt::open(&model_dir.join("mtp.safetensors")).context("mtp.safetensors")?;
    let raw = main.read_range("model.llm.unembed.weight", 0, VOCAB * D * 2)?;
    let lm: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    drop(raw);
    let ho: Vec<f64> = mtp.read("markov_head.hidden_out.weight")?.chunks_exact(2).map(|c| bf16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect();
    let hb: Vec<f64> = mtp.read("markov_head.hidden_out.bias")?.chunks_exact(2).map(|c| bf16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect();
    anyhow::ensure!(ho.len() == D * R && hb.len() == D, "markov_head.hidden_out has an unexpected shape");
    let mut fw = vec![0u16; VOCAB * R];
    let mut fb = vec![0u16; VOCAB];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(32);
    const BLK: usize = 16;
    let rows_per = VOCAB.div_ceil(threads).div_ceil(BLK) * BLK;
    std::thread::scope(|sc| {
        for ((lmc, fwc), fbc) in lm.chunks(rows_per * D).zip(fw.chunks_mut(rows_per * R)).zip(fb.chunks_mut(rows_per)) {
            let (ho, hb) = (&ho, &hb);
            sc.spawn(move || {
                let nrow = lmc.len() / D;
                let mut acc = vec![0f64; BLK * R];
                let mut accb = [0f64; BLK];
                let mut r0 = 0;
                while r0 < nrow {
                    let nb = (nrow - r0).min(BLK);
                    acc[..nb * R].fill(0.0);
                    accb[..nb].fill(0.0);
                    for d in 0..D {
                        let hrow = &ho[d * R..(d + 1) * R];
                        for k in 0..nb {
                            let a = bf16_to_f64(lmc[(r0 + k) * D + d]);
                            accb[k] += a * hb[d];
                            let ar = &mut acc[k * R..(k + 1) * R];
                            for j in 0..R {
                                ar[j] += a * hrow[j];
                            }
                        }
                    }
                    for k in 0..nb {
                        for j in 0..R {
                            fwc[(r0 + k) * R + j] = f64_to_bf16(acc[k * R + j]);
                        }
                        fbc[r0 + k] = f64_to_bf16(accb[k]);
                    }
                    r0 += nb;
                }
            });
        }
    });
    Ok((fw, fb))
}

fn write(path: &Path, fw: &[u16], fb: &[u16]) -> Result<()> {
    let nw = fw.len() * 2;
    let nb = fb.len() * 2;
    let mut hdr = format!(
        "{{\"__metadata__\":{{\"note\":\"lm_head @ markov_head.hidden_out.weight / .bias in float64, stored bf16\"}},\"fold_weight\":{{\"dtype\":\"BF16\",\"shape\":[{VOCAB},{R}],\"data_offsets\":[0,{nw}]}},\"fold_bias\":{{\"dtype\":\"BF16\",\"shape\":[{VOCAB}],\"data_offsets\":[{nw},{}]}}}}",
        nw + nb
    );
    while (hdr.len() + 8) % 8 != 0 || hdr.len() % 8 != 0 {
        hdr.push(' ');
    }
    let tmp = path.with_extension("tmp");
    let mut out = Vec::with_capacity(8 + hdr.len() + nw + nb);
    out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    out.extend_from_slice(hdr.as_bytes());
    for v in fw.iter().chain(fb) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&tmp, out)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding_from_double() {
        // exact bf16 values stay, ties go to even, values just above / below a tie round away / toward it
        assert_eq!(f64_to_bf16(1.0), 0x3f80);
        assert_eq!(f64_to_bf16(-2.0), 0xc000);
        assert_eq!(f64_to_bf16(1.0 + 2f64.powi(-8)), 0x3f80); // tie between 1.0 and 1 + 2^-7: even is 1.0
        assert_eq!(f64_to_bf16(1.0 + 3.0 * 2f64.powi(-8)), 0x3f82); // tie between 0x3f81 and 0x3f82: even is 0x3f82
        assert_eq!(f64_to_bf16(1.0 + 2f64.powi(-8) + 2f64.powi(-30)), 0x3f81);
        assert_eq!(f64_to_bf16(1.0 + 2f64.powi(-8) - 2f64.powi(-30)), 0x3f80);
        assert_eq!(f64_to_bf16(0.0), 0);
        // the carry into the exponent
        assert_eq!(f64_to_bf16(1.999999999), 0x4000);
    }

    #[test]
    fn bf16_rounding_agrees_with_f32_path_on_floats() {
        let mut x = 0x1234_5678_9abc_def1u64;
        for _ in 0..10000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let f = f32::from_bits((x >> 32) as u32 & 0x7f7f_ffff);
            if !f.is_finite() || f == 0.0 {
                continue;
            }
            let via_f32 = {
                let b = f.to_bits();
                (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
            };
            assert_eq!(f64_to_bf16(f as f64), via_f32, "{f}");
        }
    }
}
