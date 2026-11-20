//! Load-time repack of IQ1_M rows for the CPU expert lane (lookup-free decode).
//!
//! IQ1_M weights are 8 g + delta with g in {-1, 0, 1} from an 11-bit codebook (a gather / insert chain per 8 values on
//! the CPU). Here the same int8 weights are stored as base-3 digits: 10 digits per 16-bit lane (3^10 < 2^16, the same
//! 1.6 bits per digit as 5 per byte), decoded with `vpmullw` / `vpmulhuw` (digit k = ((Q * 3^k mod 2^16) * 3) >> 16),
//! 16 lanes = 160 values = 5 sub-blocks of 32 per 32-byte unit. The delta bits (4 per sub-block), the scale pair (6
//! bits) and a per-256 f32 base `d / 8` are kept as they are, so a decoded sub-block is bit-identical to the original.
//!
//! Row layout (`iq1m_row_bytes(k)`): header u32 meta_off, u32 base_off | units: nunit x 32 B (Q lanes u16) |
//! meta: nunit x u64 (5 x 10 bits per unit: scale pair | delta bits << 6) | base: nblk x f32 | pad to 8.
//! Sub-block `s` (0..5) of a unit uses digits 2s (values 0..8, 16..24) and 2s + 1 (values 8..16, 24..32), so
//! `packus(digits(2s), digits(2s + 1))` is the 32 values in order.
use crate::refq::f16_to_f32;
use crate::tables::IQ1S_GRID;

pub const POW3: [u16; 10] = [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683];

pub fn iq1m_row_bytes(k: usize) -> usize {
    let nsb = k / 32;
    (8 + nsb.div_ceil(5) * 40 + k / 256 * 4 + 7) & !7
}

/// (meta_off, base_off) of a row of `k` values.
pub fn iq1m_offsets(k: usize) -> (usize, usize) {
    let nunit = (k / 32).div_ceil(5);
    (8 + nunit * 32, 8 + nunit * 40)
}

fn rd16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

/// Repack one IQ1_M row (`k / 256` blocks of 56 B) into `dst` (`iq1m_row_bytes(k)` bytes).
pub fn repack_iq1m_row(src: &[u8], k: usize, dst: &mut [u8]) {
    assert!(k % 256 == 0 && dst.len() == iq1m_row_bytes(k) && src.len() == k / 256 * 56);
    let (nsb, nblk) = (k / 32, k / 256);
    let nunit = nsb.div_ceil(5);
    let (meta_off, base_off) = iq1m_offsets(k);
    dst.fill(0);
    dst[0..4].copy_from_slice(&(meta_off as u32).to_le_bytes());
    dst[4..8].copy_from_slice(&(base_off as u32).to_le_bytes());
    let mut dig = vec![[1u8; 32]; nunit * 5]; // g + 1 (padding sub-blocks: g = 0)
    let mut m10 = vec![0u16; nunit * 5];
    for blk in 0..nblk {
        let b = &src[blk * 56..(blk + 1) * 56];
        let sc = [rd16(b, 48), rd16(b, 50), rd16(b, 52), rd16(b, 54)];
        let du = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let base = f16_to_f32(du) * 0.125;
        dst[base_off + 4 * blk..base_off + 4 * blk + 4].copy_from_slice(&base.to_le_bytes());
        for ib in 0..8 {
            let sbj = blk * 8 + ib;
            let pair = (sc[ib / 2] >> (6 * (ib % 2))) & 63;
            let qh = rd16(b, 32 + 2 * ib) as usize;
            let mut dl = 0u16;
            for l in 0..4 {
                let hi = match l {
                    0 => (qh << 8) & 0x700,
                    1 => (qh << 4) & 0x700,
                    2 => qh & 0x700,
                    _ => (qh >> 4) & 0x700,
                };
                let idx = b[4 * ib + l] as usize | hi;
                let g = IQ1S_GRID[idx].to_le_bytes();
                for j in 0..8 {
                    dig[sbj][8 * l + j] = (g[j] as i8 + 1) as u8;
                }
                dl |= (((qh >> (4 * l + 3)) & 1) as u16) << l;
            }
            m10[sbj] = pair | (dl << 6);
        }
    }
    for u in 0..nunit {
        let mut meta = 0u64;
        for s in 0..5 {
            meta |= (m10[5 * u + s] as u64) << (10 * s);
        }
        dst[meta_off + 8 * u..meta_off + 8 * u + 8].copy_from_slice(&meta.to_le_bytes());
        for lane in 0..16 {
            let mut v = 0u32;
            for kk in 0..10 {
                let (s, is_b) = (kk / 2, kk % 2 == 1);
                let j = match (is_b, lane < 8) {
                    (false, true) => lane,
                    (true, true) => lane + 8,
                    (false, false) => lane + 8,
                    (true, false) => lane + 16,
                };
                v = v * 3 + dig[5 * u + s][j] as u32;
            }
            // Q = ceil(v * 65536 / 3^10)
            let q = ((v as u64 * 65536).div_ceil(59049)) as u16;
            dst[8 + 32 * u + 2 * lane..8 + 32 * u + 2 * lane + 2].copy_from_slice(&q.to_le_bytes());
        }
    }
}

/// Repack `rows` rows of IQ1_M (`row_bytes = k / 256 * 56`) into a new buffer.
pub fn repack_iq1m(src: &[u8], rows: usize, k: usize) -> Vec<u8> {
    let (rb, ob) = (k / 256 * 56, iq1m_row_bytes(k));
    let mut out = vec![0u8; rows * ob];
    for r in 0..rows {
        repack_iq1m_row(&src[r * rb..(r + 1) * rb], k, &mut out[r * ob..(r + 1) * ob]);
    }
    out
}

#[cfg(test)]
mod tests {
    /// The digit extraction identity for every 10-digit value.
    #[test]
    fn digits_roundtrip() {
        for v in 0u32..59049 {
            let q = ((v as u64 * 65536).div_ceil(59049)) as u16;
            let mut got = 0u32;
            for k in 0..10 {
                let x = q.wrapping_mul(super::POW3[k]);
                got = got * 3 + ((x as u32 * 3) >> 16);
            }
            assert_eq!(got, v);
        }
    }
}

/// Row bytes of the plane repack (`GgmlType::RepR3*`): 108 B per 256 values.
pub fn rep3_row_bytes(k: usize) -> usize {
    k / 256 * 108
}

/// Repack one IQ2_XXS / IQ2_XS / IQ1_M row (`ty` = the ORIGINAL type) into the plane layout of `avx2::Rep3`: per 256 values
/// [64 B magnitude codes | 32 B sign bits | 8 scale bytes | f32]. The magnitudes / signs come from the original decoder (so they are exactly
/// what the dot sees); the scale bytes and f32 are the raw ones. Returns false when a magnitude is outside the type's three levels.
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn repack_rep3_row(ty: fe_gguf::GgmlType, src: &[u8], k: usize, dst: &mut [u8]) -> bool {
    use crate::cpu::avx2::{Cq, Iq1M, Iq2Xs, Iq2Xxs};
    use fe_gguf::GgmlType::*;
    assert!(k % 256 == 0 && dst.len() == rep3_row_bytes(k));
    let lut: [u8; 3] = if ty == IQ1M { [1, 7, 9] } else { [8, 25, 43] };
    let mut inv = [0xffu8; 256];
    for (c, &v) in lut.iter().enumerate() {
        inv[v as usize] = c as u8;
    }
    let bsz = match ty { IQ2XXS => 66, IQ2XS => 74, _ => 56 };
    assert!(src.len() == k / 256 * bsz);
    let row = src.as_ptr();
    let mut ok = true;
    for blk in 0..k / 256 {
        let b = &src[blk * bsz..(blk + 1) * bsz];
        let o = &mut dst[blk * 108..(blk + 1) * 108];
        o.fill(0);
        let (d, base): (f32, f32);
        match ty {
            IQ2XXS => { d = f16_to_f32(rd16(b, 0)); base = d; }
            IQ2XS => { d = f16_to_f32(rd16(b, 0)); base = d * 0.125; }
            _ => {
                let sc = [rd16(b, 48), rd16(b, 50), rd16(b, 52), rd16(b, 54)];
                let du = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
                d = f16_to_f32(du);
                base = d * 0.125;
            }
        }
        let _ = d;
        o[104..108].copy_from_slice(&base.to_le_bytes());
        for ib in 0..8 {
            let j = blk * 8 + ib;
            let dq = match ty {
                IQ2XXS => Iq2Xxs::<false>::dq(row, j),
                IQ2XS => Iq2Xs::<false>::dq(row, j),
                _ => Iq1M::<false>::dq(row, j),
            };
            let mut wa = [0u8; 32];
            let mut sg = [0u8; 32];
            std::arch::x86_64::_mm256_storeu_si256(wa.as_mut_ptr() as *mut _, dq.wa);
            std::arch::x86_64::_mm256_storeu_si256(sg.as_mut_ptr() as *mut _, dq.sg);
            for i in 0..32 {
                let mut code = inv[wa[i] as usize];
                if code == 0xff {
                    ok = false;
                    code = 0;
                }
                // value i lives in byte (i % 8) of the 8-byte plane at bit 2 * (i / 8)
                o[8 * ib + i % 8] |= code << (2 * (i / 8));
                if (sg[i] as i8) < 0 {
                    o[64 + 4 * ib + i / 8] |= 1 << (i % 8);
                }
            }
            o[96 + ib] = match ty {
                IQ2XXS => (rd32(b, 2 + 8 * ib + 4) >> 28) as u8,
                IQ2XS => b[66 + ib],
                _ => ((rd16(b, 48 + 2 * (ib / 2)) >> (6 * (ib % 2))) & 63) as u8,
            };
        }
    }
    ok
}

fn rd32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

pub fn hxxs_row_bytes(k: usize) -> usize {
    k / 256 * 76
}

/// IQ2_XXS row -> the hybrid layout of `avx2::RepH`: [32 B grid indices | 32 B resolved sign bits | 8 scale bytes | f32 d] per 256 values.
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub unsafe fn repack_hxxs_row(src: &[u8], k: usize, dst: &mut [u8]) {
    use crate::cpu::avx2::{Cq, Iq2Xxs};
    assert!(k % 256 == 0 && dst.len() == hxxs_row_bytes(k) && src.len() == k / 256 * 66);
    for blk in 0..k / 256 {
        let b = &src[blk * 66..(blk + 1) * 66];
        let o = &mut dst[blk * 76..(blk + 1) * 76];
        o.fill(0);
        o[72..76].copy_from_slice(&f16_to_f32(rd16(b, 0)).to_bits().to_le_bytes());
        for ib in 0..8 {
            o[4 * ib..4 * ib + 4].copy_from_slice(&b[2 + 8 * ib..2 + 8 * ib + 4]);
            let dq = Iq2Xxs::<false>::dq(src.as_ptr(), blk * 8 + ib);
            let mut sg = [0u8; 32];
            std::arch::x86_64::_mm256_storeu_si256(sg.as_mut_ptr() as *mut _, dq.sg);
            for i in 0..32 {
                if (sg[i] as i8) < 0 {
                    o[32 + 4 * ib + i / 8] |= 1 << (i % 8);
                }
            }
            o[64 + ib] = (rd32(b, 2 + 8 * ib + 4) >> 28) as u8;
        }
    }
}
