//! The B70 device layout of an XQ tensor (arc-engine `xq.zig` `pack` / `kernels/xpu/xq.cl`), byte for byte.
//!
//! One contiguous stream a 16-row block. Chunk (row block rb, group gi) starts at dword `(rb * NG + gi) * (16 D + 8)`,
//! NG = k / g groups, D = g * bits / 32 dwords a lane: `[D][16 lanes]` code dwords (lane = row `rb * 16 + lane`), then the
//! 16 rows' fp16 scales as 8 dwords (row 2i in the low half of dword i, row 2i + 1 in the high half).
//!
//! A code is stored unsigned, `u = q + 2^(b-1)`. Element kk of a group: step `t = kk / 16`, pair `d = (kk % 16) / 2`, `e = kk % 2`.
//! - b = 4, 5, 6, 8: low nibble in dword `2t + d/4` at bit `4 (d % 4) + 16 e`; the high plane after the g/8 nibble dwords:
//!   - 5: 1 bit, dword `g/8 + t/2`, bit `16 e + 8 (t % 2) + d`
//!   - 6: 2 bits, dword `g/8 + t`, bits `16 e + 2 d`
//!   - 8: 4 bits, dword `g/8 + 2t + d/4`, bits `16 e + 4 (d % 4)`
//! - b = 3: no nibble plane: a 2-bit plane (dword `t`, bits `16 e + 2 d`), then a 1-bit plane (dword `g/16 + t/2`, bit `16 e + 8 (t % 2) + d`).
//!
//! The decode kernel builds the DPAS B dword (fp16 pair `1024 + u[2d]`, `1024 + u[2d + 1]` of step t) with one shift + mask a plane and an
//! or with 0x64006400 (`bdw` in xq.cl); [`b_dword`] is that function on the host.
use anyhow::{bail, ensure, Result};

/// Dwords a lane holds per group.
pub fn lane_dwords(bits: u32, g: u32) -> usize {
    (g * bits / 32) as usize
}

/// Dwords of one (16-row block, group) chunk: 16 lanes of codes + 16 fp16 scales.
pub fn chunk_dwords(bits: u32, g: u32) -> usize {
    16 * lane_dwords(bits, g) + 8
}

/// Dwords of a packed [n][k] tensor.
pub fn packed_dwords(bits: u32, g: u32, n: usize, k: usize) -> usize {
    (n / 16) * (k / g as usize) * chunk_dwords(bits, g)
}

pub fn check_shape(bits: u32, g: u32, n: usize, k: usize) -> Result<()> {
    ensure!(matches!(bits, 3 | 4 | 5 | 6 | 8), "bits {bits} (want 3, 4, 5, 6 or 8)");
    ensure!(matches!(g, 32 | 64 | 128), "group {g} (want 32, 64 or 128)");
    ensure!(n % 16 == 0 && n > 0, "n {n} not a positive multiple of 16");
    ensure!(k % g as usize == 0 && k > 0, "k {k} not a positive multiple of the group {g}");
    Ok(())
}

/// Codes q (int8 [n][k], -2^(b-1) <= q < 2^(b-1)) and scales s (fp16 bits [n][k/g]) to the device layout.
pub fn pack(bits: u32, g: u32, n: usize, k: usize, q: &[i8], s: &[u16]) -> Result<Vec<u32>> {
    let mut out = vec![0u32; packed_dwords(bits, g, n, k)];
    pack_into(bits, g, n, k, q, s, &mut out)?;
    Ok(out)
}

/// [`pack`] into a caller buffer (`packed_dwords` long), row blocks spread over the available cores.
pub fn pack_into(bits: u32, g: u32, n: usize, k: usize, q: &[i8], s: &[u16], out: &mut [u32]) -> Result<()> {
    check_shape(bits, g, n, k)?;
    let ng = k / g as usize;
    ensure!(q.len() >= n * k && s.len() >= n * ng, "pack: q {} / s {} too short for [{n}][{k}]", q.len(), s.len());
    ensure!(out.len() >= packed_dwords(bits, g, n, k), "pack: out too short");
    let lo = -(1i32 << (bits - 1));
    let hi = (1i32 << (bits - 1)) - 1;
    if let Some(i) = q[..n * k].iter().position(|&v| (v as i32) < lo || (v as i32) > hi) {
        bail!("pack: code {} at [{}][{}] outside the {bits}-bit range", q[i], i / k, i % k);
    }
    let blk = ng * chunk_dwords(bits, g); // dwords of one 16-row block
    let nb = n / 16;
    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1).min(16).min(nb).max(1);
    let per = nb.div_ceil(threads);
    std::thread::scope(|sc| {
        for (ti, part) in out[..nb * blk].chunks_mut(per * blk).enumerate() {
            sc.spawn(move || {
                for (j, rb_out) in part.chunks_mut(blk).enumerate() {
                    pack_block(bits, g, k, q, s, ti * per + j, rb_out);
                }
            });
        }
    });
    Ok(())
}

/// One 16-row block (all its groups) into `o` (ng * chunk dwords), the loop of `xq.zig packBlocks`.
fn pack_block(bits: u32, g: u32, k: usize, q: &[i8], s: &[u16], rb: usize, o: &mut [u32]) {
    let g = g as usize;
    let ng = k / g;
    let d = lane_dwords(bits, g as u32);
    let ch = chunk_dwords(bits, g as u32);
    let off = 1i32 << (bits - 1);
    for gi in 0..ng {
        let c = &mut o[gi * ch..(gi + 1) * ch];
        c.fill(0);
        for lane in 0..16 {
            let row = rb * 16 + lane;
            let src = &q[row * k + gi * g..row * k + gi * g + g];
            for (kk, &v) in src.iter().enumerate() {
                let u = (v as i32 + off) as u32;
                let t = kk / 16;
                let dd = (kk % 16) / 2;
                let e = kk % 2;
                if bits == 3 {
                    c[t * 16 + lane] |= (u & 3) << (16 * e + 2 * dd);
                    c[(g / 16 + t / 2) * 16 + lane] |= ((u >> 2) & 1) << (16 * e + 8 * (t % 2) + dd);
                    continue;
                }
                c[(2 * t + dd / 4) * 16 + lane] |= (u & 15) << (4 * (dd % 4) + 16 * e);
                match bits {
                    5 => c[(g / 8 + t / 2) * 16 + lane] |= ((u >> 4) & 1) << (16 * e + 8 * (t % 2) + dd),
                    6 => c[(g / 8 + t) * 16 + lane] |= ((u >> 4) & 3) << (16 * e + 2 * dd),
                    8 => c[(g / 8 + 2 * t + dd / 4) * 16 + lane] |= ((u >> 4) & 15) << (16 * e + 4 * (dd % 4)),
                    _ => {}
                }
            }
            c[16 * d + lane / 2] |= (s[row * ng + gi] as u32) << (16 * (lane % 2));
        }
    }
}

fn shl(v: u32, from: u32, to: u32) -> u32 {
    if from >= to { v >> (from - to) } else { v << (to - from) }
}

/// The DPAS B dword `d` (0..8) of step `t` (0..g/16) of one lane's group words `w` (D dwords): fp16 bits of `1024 + u[16t + 2d]`
/// (low half) and `1024 + u[16t + 2d + 1]` (high half). Same expression as `bdw` in xq.cl.
pub fn b_dword(bits: u32, g: u32, w: &[u32], t: usize, d: usize) -> u32 {
    let g = g as usize;
    let (h, j) = (d >> 2, (d & 3) as u32);
    if bits == 3 {
        return ((w[t] >> (2 * d)) & 0x0003_0003) | (shl(w[g / 16 + (t >> 1)], ((t & 1) * 8 + d) as u32, 2) & 0x0004_0004) | 0x6400_6400;
    }
    let mut v = (w[2 * t + h] >> (4 * j)) & 0x000F_000F;
    match bits {
        5 => v |= shl(w[g / 8 + (t >> 1)], ((t & 1) * 8 + d) as u32, 4) & 0x0010_0010,
        6 => v |= shl(w[g / 8 + t], 2 * d as u32, 4) & 0x0030_0030,
        8 => v |= shl(w[g / 8 + 2 * t + h], 4 * j, 4) & 0x00F0_00F0,
        _ => {}
    }
    v | 0x6400_6400
}

/// The words of lane `lane` in chunk `c` (D dwords, gathered from the [D][16] layout).
pub fn lane_words(bits: u32, g: u32, c: &[u32], lane: usize) -> Vec<u32> {
    (0..lane_dwords(bits, g)).map(|i| c[i * 16 + lane]).collect()
}

/// Inverse of [`pack`] through the kernel's own decode ([`b_dword`]): codes int8 [n][k] and scales fp16 bits [n][k/g].
pub fn unpack(bits: u32, g: u32, n: usize, k: usize, p: &[u32]) -> Result<(Vec<i8>, Vec<u16>)> {
    check_shape(bits, g, n, k)?;
    ensure!(p.len() >= packed_dwords(bits, g, n, k), "unpack: packed buffer too short");
    let gu = g as usize;
    let ng = k / gu;
    let ch = chunk_dwords(bits, g);
    let dl = lane_dwords(bits, g);
    let off = 1i32 << (bits - 1);
    let mut q = vec![0i8; n * k];
    let mut s = vec![0u16; n * ng];
    for rb in 0..n / 16 {
        for gi in 0..ng {
            let c = &p[(rb * ng + gi) * ch..(rb * ng + gi + 1) * ch];
            for lane in 0..16 {
                let row = rb * 16 + lane;
                let w = lane_words(bits, g, c, lane);
                for t in 0..gu / 16 {
                    for d in 0..8 {
                        let b = b_dword(bits, g, &w, t, d);
                        for e in 0..2 {
                            let u = ((b >> (16 * e)) & 0xffff) as i32 - 0x6400;
                            q[row * k + gi * gu + 16 * t + 2 * d + e] = (u - off) as i8;
                        }
                    }
                }
                s[row * ng + gi] = (c[16 * dl + lane / 2] >> (16 * (lane % 2))) as u16;
            }
        }
    }
    Ok((q, s))
}

/// The row-count variant of a decode call (rows a DPAS `MW`, row groups `NR`) for m = 1, 2, 3..4, 5..8, 9..16 (`xq.zig variantFor`).
/// The kernel name is `xq_mv_{g}_{MW}x{NR}_{NB}`; rows m..MW*NR of the scratch are zero.
pub const VARIANTS: [(u32, u32); 5] = [(1, 1), (2, 1), (4, 1), (8, 1), (8, 2)];

pub fn variant_for(m: u32) -> Result<usize> {
    ensure!((1..=16).contains(&m), "rows {m} (decode kernel: 1..16)");
    Ok(match m {
        1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        _ => 4,
    })
}

/// K splits of the decode matvec (sub-groups of a work-group, partials added in split order): a function of the shape only, so every
/// row count shares one summation order (`xq.zig splitFor`; `target` = env XQ_TARGET there, None = the shipped default).
pub fn split_for(n: usize, k: usize, g: u32, target: Option<u32>) -> u32 {
    let nk = n as u64 * k as u64;
    let target = target.unwrap_or(if nk > 16 << 20 && nk < 40 << 20 { 1900 } else { 1280 }) as f64;
    let blocks = (n / 16) as f64;
    let want = (target / blocks).round().max(1.0) as u32;
    want.min(32).min((k / g as usize) as u32)
}

/// The group range of split `s` of `S` over NG groups (`g0 = s NG / S`, `g1 = (s + 1) NG / S`).
pub fn split_range(ng: u32, s: u32, splits: u32) -> (u32, u32) {
    (s * ng / splits, (s + 1) * ng / splits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn pack_unpack_roundtrip_all_widths() {
        let mut seed = 7u64;
        for bits in [3u32, 4, 5, 6, 8] {
            for g in [32u32, 64, 128] {
                let (n, k) = (48usize, 512usize);
                let span = 1u64 << bits;
                let q: Vec<i8> = (0..n * k).map(|_| (lcg(&mut seed) % span) as i32 - (1i32 << (bits - 1))).map(|v| v as i8).collect();
                let s: Vec<u16> = (0..n * k / g as usize).map(|_| (lcg(&mut seed) & 0x7bff) as u16).collect();
                let p = pack(bits, g, n, k, &q, &s).unwrap();
                assert_eq!(p.len(), packed_dwords(bits, g, n, k));
                let (q2, s2) = unpack(bits, g, n, k, &p).unwrap();
                assert_eq!(q, q2, "codes bits {bits} g {g}");
                assert_eq!(s, s2, "scales bits {bits} g {g}");
            }
        }
    }

    #[test]
    fn rejects_out_of_range() {
        let q = vec![8i8; 16 * 128];
        let s = vec![0u16; 16];
        assert!(pack(4, 128, 16, 128, &q, &s).is_err());
        let q = vec![-8i8; 16 * 128];
        assert!(pack(4, 128, 16, 128, &q, &s).is_ok());
    }

    #[test]
    fn split_on_qwen_shapes() {
        // target 1280 sub-groups, 1900 for 16M..40M weights (xq.zig splitFor), at most 32 and one split a group
        assert_eq!(split_for(17408, 5120, 128, None), 1); // gate / up: 1088 blocks
        assert_eq!(split_for(5120, 17408, 128, None), 4); // down: 320 blocks
        assert_eq!(split_for(10240, 5120, 128, None), 2); // GDN qkv: 640 blocks
        assert_eq!(split_for(6144, 5120, 128, None), 5); // GDN z: 31.5M weights -> 1900 / 384
        assert_eq!(split_for(5120, 6144, 128, None), 6); // GDN out / attn o: 1900 / 320
        assert_eq!(split_for(12288, 5120, 128, None), 2); // attn q (+ gate): 768 blocks
        assert_eq!(split_for(1024, 5120, 128, None), 20); // attn k / v
        assert_eq!(split_for(48, 5120, 128, None), 32); // GDN b / a: capped at 32
        assert_eq!(split_for(248320, 5120, 128, None), 1); // lm_head
        assert_eq!(split_for(5120, 10240, 128, None), 4); // mtp.fc
    }
}
