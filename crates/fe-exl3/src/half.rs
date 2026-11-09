//! fp16 / bf16 bit-level helpers (no `f16` type on stable, no external crate).

/// Exact fp16 -> f64.
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    match e {
        0 => sign * m * 2f64.powi(-24),
        31 => {
            if m == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    f16_to_f64(h) as f32
}

/// f64 -> fp16, round to nearest even, in one rounding (no f32 detour).
pub fn f64_to_f16(x: f64) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let ax = x.abs();
    if ax.is_nan() {
        return 0x7e00;
    }
    if ax == 0.0 {
        return sign;
    }
    let e = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let mant = (bits & ((1u64 << 52) - 1)) | (1u64 << 52); // 53 bits, f64 subnormals are far below fp16's range
    if e < -14 {
        // fp16 subnormal: units of 2^-24
        let shift = (28 - e) as u32; // value = mant * 2^(e - 52) = q * 2^-24
        if shift >= 64 {
            return sign;
        }
        let q = mant >> shift;
        let rem = mant & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        let q = if rem > half || (rem == half && q & 1 == 1) { q + 1 } else { q };
        return sign | q as u16; // q == 1024 is the smallest normal, encoded the same way
    }
    let mut q = mant >> 42; // 11 bits with the implicit one
    let rem = mant & ((1u64 << 42) - 1);
    let half = 1u64 << 41;
    let mut e = e;
    if rem > half || (rem == half && q & 1 == 1) {
        q += 1;
        if q == 2048 {
            q = 1024;
            e += 1;
        }
    }
    if e + 15 >= 31 {
        return sign | 0x7c00;
    }
    sign | (((e + 15) as u16) << 10) | (q as u16 & 0x3ff)
}

/// fp16 fused multiply-add with a single rounding, as `fma.rn.f16` does.
pub fn f16_fma(a: u16, b: u16, c: u16) -> u16 {
    // The product of two fp16 values is exact in f64 (22 significant bits) and so is the sum with c for the
    // magnitudes involved here (all within 2^±30 of each other).
    f64_to_f16(f16_to_f64(a) * f16_to_f64(b) + f16_to_f64(c))
}

/// f32 -> bf16 bits, round to nearest even (NaN kept quiet).
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// f64 -> bf16 the way torch converts a float64 tensor: f64 -> f32 (RNE) -> bf16 (RNE).
pub fn f64_to_bf16_torch(x: f64) -> u16 {
    f32_to_bf16(x as f32)
}

/// Distance in units of last place between two finite bf16 values of possibly different sign.
pub fn bf16_ulp_diff(a: u16, b: u16) -> u32 {
    fn ord(h: u16) -> i32 {
        if h & 0x8000 != 0 { -((h & 0x7fff) as i32) } else { h as i32 }
    }
    (ord(a) - ord(b)).unsigned_abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_all() {
        for h in 0u16..=0xffff {
            let e = (h >> 10) & 0x1f;
            if e == 31 {
                continue;
            }
            assert_eq!(f64_to_f16(f16_to_f64(h)), if h == 0x8000 { 0x8000 } else { h }, "{h:#x}");
        }
    }

    #[test]
    fn f16_ties_to_even() {
        // 1 + 2^-11 lies halfway between 1.0 and the next fp16 value: rounds to 1.0 (even)
        assert_eq!(f64_to_f16(1.0 + 2f64.powi(-11)), 0x3c00);
        assert_eq!(f64_to_f16(1.0 + 3.0 * 2f64.powi(-11)), 0x3c02);
        // subnormal ties: 2^-25 rounds to 0, 3 * 2^-25 to 2 * 2^-24
        assert_eq!(f64_to_f16(2f64.powi(-25)), 0);
        assert_eq!(f64_to_f16(3.0 * 2f64.powi(-25)), 2);
    }
}
