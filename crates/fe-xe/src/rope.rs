//! The RoPE cos / sin table, computed on the host exactly as arc-engine does (`Win.setRope`, `Ops.setPosition`, `qwen_mtp.ropeTable`):
//! `inv = f32(theta^(-i / 32))` (f64 pow), `ang = f32(pos) * inv` (f32), then f32 cos / sin. Zig's `@cos` / `@sin` on f32 are its own
//! musl-derived routines, not glibc's: Rust's `f32::cos` (glibc `cosf`) differs in the last bit for 1.1 % of the 128K x 32 table, the
//! `libm` crate (the same musl port) matches all 8,388,608 values (checked 2026-11-09 against a Zig dump of the full table).
//! Row layout per position: `[cos 0..32 | sin 0..32]` fp32 (64 floats), the kernels' `rope` buffer.
pub const HALF: usize = 32;

pub fn inv_freq(theta: f64) -> [f32; HALF] {
    let mut v = [0f32; HALF];
    for (i, x) in v.iter_mut().enumerate() {
        *x = theta.powf(-(i as f64) / HALF as f64) as f32;
    }
    v
}

/// The 64 floats of position `pos`.
pub fn row(inv: &[f32; HALF], pos: u32, out: &mut [f32]) {
    for i in 0..HALF {
        let ang = pos as f32 * inv[i];
        out[i] = libm::cosf(ang);
        out[HALF + i] = libm::sinf(ang);
    }
}

/// Rows for positions pos0 .. pos0 + n, [n][64].
pub fn table(theta: f64, pos0: u32, n: usize) -> Vec<f32> {
    let inv = inv_freq(theta);
    let mut t = vec![0f32; n * 2 * HALF];
    for r in 0..n {
        row(&inv, pos0 + r as u32, &mut t[r * 2 * HALF..(r + 1) * 2 * HALF]);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_arc_engine_bits() {
        // (position, index in the 64-float row, bits from arc-engine's zig table); the first five differ under glibc cosf / sinf
        let want = [(1u32, 35usize, 0x3e6023d9u32), (8, 4, 0x3ef7407a), (9, 34, 0xbe13ef29), (12, 40, 0x3e58dc3f), (13, 40, 0x3e6a9f25),
                    (131071, 0, 0xbf51675e), (131071, 63, 0x3cb1ab5a), (70000, 17, 0x3f3074b7)];
        let inv = inv_freq(1e7);
        let mut r = [0f32; 64];
        for (p, j, bits) in want {
            row(&inv, p, &mut r);
            assert_eq!(r[j].to_bits(), bits, "pos {p} idx {j}");
        }
    }
}
