//! The drafter linears as XQ tensors, round to nearest: per 128-block of inputs the row is rotated like the target's XQ
//! weights (w~ = FWHT128(su * w) / sqrt(128), su a fixed +-1 vector per input width, so every projection of one input shares
//! the activation prepass), then symmetric b-bit codes with one fp16 scale per 128 group (per group the scale of least
//! squared error among a few clip ratios of max|w~|). Packed with fe_xq::pack, so kx-xq's GEMV runs them as is.
//! (The Sparks drafter is MLX-style affine 4-bit g64 RTN; the target's XQ is GPTQ. RTN is enough for a drafter: it only
//! proposes tokens.)
use anyhow::{ensure, Context as _, Result};
use fe_xq::half::{bf16_to_f32, f16_to_f32, f64_to_f16};
use std::io::{Read, Write};
use std::path::Path;

/// fp16 +-1 signs for inputs of width k (xorshift64 seeded by k; deterministic, never stored).
pub fn su_for(k: usize) -> Vec<u16> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ (k as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    (0..k)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if (x >> 32) & 1 == 0 { 0x3c00 } else { 0xbc00 }
        })
        .collect()
}

/// Unnormalised Walsh-Hadamard transform of 128 values (natural order: H[i][j] = (-1)^popcount(i & j), as fwht128_xq).
fn fwht(v: &mut [f64; 128]) {
    let mut h = 1;
    while h < 128 {
        let mut i = 0;
        while i < 128 {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// RTN of a bf16 [n][k] tensor to XQ codes (int8 [n][k]) and fp16 scales [n][k / 128] for the rotation `su`.
pub fn quantize(w: &[u16], n: usize, k: usize, bits: u32, su: &[u16]) -> Result<(Vec<i8>, Vec<u16>)> {
    ensure!(k % 128 == 0 && n % 16 == 0 && w.len() == n * k && su.len() == k, "quantize [{n}][{k}]");
    let ng = k / 128;
    let mut q = vec![0i8; n * k];
    let mut s = vec![0u16; n * ng];
    let lo = -(1i32 << (bits - 1)) as f64;
    let hi = ((1i32 << (bits - 1)) - 1) as f64;
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let rows_per = n.div_ceil(threads);
    let inv = (1.0f64 / 128.0).sqrt();
    std::thread::scope(|scp| {
        for (t, (qc, sc)) in q.chunks_mut(rows_per * k).zip(s.chunks_mut(rows_per * ng)).enumerate() {
            scp.spawn(move || {
                let r0 = t * rows_per;
                for rr in 0..qc.len() / k {
                    let row = &w[(r0 + rr) * k..(r0 + rr + 1) * k];
                    for g in 0..ng {
                        let mut v = [0f64; 128];
                        for j in 0..128 {
                            v[j] = bf16_to_f32(row[g * 128 + j]) as f64 * f16_to_f32(su[g * 128 + j]) as f64;
                        }
                        fwht(&mut v);
                        for x in v.iter_mut() {
                            *x *= inv;
                        }
                        let amax = v.iter().fold(0f64, |a, &x| a.max(x.abs()));
                        let (mut best_e, mut best_s, mut best_q) = (f64::INFINITY, 0u16, [0i8; 128]);
                        if amax > 0.0 {
                            for c in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7] {
                                let s16 = f64_to_f16(amax * c / hi);
                                let sf = f16_to_f32(s16) as f64;
                                if sf <= 0.0 || !sf.is_finite() {
                                    continue;
                                }
                                let mut e = 0.0;
                                let mut qq = [0i8; 128];
                                for j in 0..128 {
                                    let qi = (v[j] / sf).round().clamp(lo, hi);
                                    qq[j] = qi as i8;
                                    let d = v[j] - qi * sf;
                                    e += d * d;
                                }
                                if e < best_e {
                                    best_e = e;
                                    best_s = s16;
                                    best_q = qq;
                                }
                            }
                        }
                        sc[rr * ng + g] = best_s;
                        qc[rr * k + g * 128..rr * k + g * 128 + 128].copy_from_slice(&best_q);
                    }
                }
            });
        }
    });
    Ok((q, s))
}

/// One packed drafter linear on the host: the B70 layout (fe_xq::pack) and its shape.
pub struct Packed {
    pub name: String,
    pub bits: u32,
    pub n: usize,
    pub k: usize,
    pub data: Vec<u32>,
}

/// Quantizes and packs a bf16 [n][k] tensor at `bits`.
pub fn pack_tensor(name: &str, w: &[u16], n: usize, k: usize, bits: u32) -> Result<Packed> {
    let su = su_for(k);
    let (q, s) = quantize(w, n, k, bits, &su)?;
    let data = fe_xq::pack::pack(bits, 128, n, k, &q, &s).with_context(|| format!("packing {name}"))?;
    Ok(Packed { name: name.to_string(), bits, n, k, data })
}

const MAGIC: &[u8; 8] = b"XDFXQ01\n";

/// The packed tensors to a cache file (MAGIC, count, then per tensor: name length + name, bits, n, k, dwords, data).
pub fn save(path: &Path, ts: &[Packed]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp).with_context(|| format!("creating {tmp:?}"))?);
    f.write_all(MAGIC)?;
    f.write_all(&(ts.len() as u32).to_le_bytes())?;
    for t in ts {
        f.write_all(&(t.name.len() as u32).to_le_bytes())?;
        f.write_all(t.name.as_bytes())?;
        for v in [t.bits, t.n as u32, t.k as u32] {
            f.write_all(&v.to_le_bytes())?;
        }
        f.write_all(&(t.data.len() as u64).to_le_bytes())?;
        let bytes = unsafe { std::slice::from_raw_parts(t.data.as_ptr() as *const u8, t.data.len() * 4) };
        f.write_all(bytes)?;
    }
    f.flush()?;
    drop(f);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn load(path: &Path) -> Result<Vec<Packed>> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path).with_context(|| format!("opening {path:?}"))?);
    let mut m = [0u8; 8];
    f.read_exact(&mut m)?;
    ensure!(&m == MAGIC, "{path:?}: not a drafter XQ cache");
    let mut u4 = [0u8; 4];
    let mut rd = |f: &mut std::io::BufReader<std::fs::File>| -> Result<u32> {
        f.read_exact(&mut u4)?;
        Ok(u32::from_le_bytes(u4))
    };
    let count = rd(&mut f)?;
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let nl = rd(&mut f)? as usize;
        let mut name = vec![0u8; nl];
        f.read_exact(&mut name)?;
        let (bits, n, k) = (rd(&mut f)?, rd(&mut f)? as usize, rd(&mut f)? as usize);
        let mut u8b = [0u8; 8];
        f.read_exact(&mut u8b)?;
        let dw = u64::from_le_bytes(u8b) as usize;
        ensure!(dw == fe_xq::pack::packed_dwords(bits, 128, n, k), "{path:?}: bad tensor size");
        let mut data = vec![0u32; dw];
        let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, dw * 4) };
        f.read_exact(bytes)?;
        out.push(Packed { name: String::from_utf8(name)?, bits, n, k, data });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quantized tensor applied to the rotated input reproduces x . W^T within the 4-bit RTN error, through the CPU
    /// reference of the XQ format (prep + float64 GEMV over the packed codes).
    #[test]
    fn xq_rtn_roundtrip() {
        let (n, k) = (32usize, 256usize);
        let mut x = 12345u64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        };
        let w: Vec<u16> = (0..n * k).map(|_| fe_xq::half::f32_to_bf16(rnd())).collect();
        let a: Vec<u16> = (0..k).map(|_| fe_xq::half::f32_to_bf16(rnd() * 4.0)).collect();
        let su = su_for(k);
        for bits in [4u32, 8] {
            let (q, s) = quantize(&w, n, k, bits, &su).unwrap();
            let p = fe_xq::cpu::prep(fe_xq::cpu::Act::Bf16(&a), k, 1, k, Some(&su));
            let y = fe_xq::cpu::gemv_f64(128, n, k, &q, &s, &p.xt, 1);
            let mut err = 0f64;
            let mut mag = 0f64;
            for r in 0..n {
                let want: f64 = (0..k).map(|j| bf16_to_f32(w[r * k + j]) as f64 * bf16_to_f32(a[j]) as f64).sum();
                err += (y[r] - want).powi(2);
                mag += want * want;
            }
            let rel = (err / mag).sqrt();
            assert!(rel < if bits == 4 { 0.12 } else { 0.01 }, "bits {bits}: relative error {rel}");
        }
    }
}
