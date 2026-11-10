//! Host side of the X0 / X1 proofs for the Rust Xe2 kernels: the SPIR-V modules (built from kernels-xe by build.rs),
//! small numeric helpers, and the arc-engine oracle modules (OpenCL-built `.spv`, test baselines only; NON-RUST.md).
use fe_ze::{Module, Ze};
use std::sync::Arc;

pub static KX_TEST: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_test.spv"));
pub static KX_BENCH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_bench.spv"));
/// Prompt GEMM module (bits-independent; build with -cl-intel-256-GRF-per-thread).
pub static KX_XQPF: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xqpf.spv"));
pub static KX_XQ3: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xq3.spv"));
pub static KX_XQ4: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xq4.spv"));
pub static KX_XQ5: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xq5.spv"));
pub static KX_XQ6: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xq6.spv"));
pub static KX_XQ8: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xq8.spv"));

/// The Rust XQ module of a code width (KX_SPV_DIR=dir: read `dir/kx_xq{bits}.spv` instead of the built-in one, to time
/// alternative builds with the same binary).
pub fn kx_xq(bits: u32) -> &'static [u8] {
    if let Ok(d) = std::env::var("KX_SPV_DIR") {
        let p = format!("{d}/kx_xq{bits}.spv");
        let v = std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
        return Box::leak(v.into_boxed_slice());
    }
    match bits {
        3 => KX_XQ3,
        4 => KX_XQ4,
        5 => KX_XQ5,
        6 => KX_XQ6,
        8 => KX_XQ8,
        _ => panic!("no XQ module for {bits} bits"),
    }
}

/// Directory of arc-engine's committed OpenCL `.spv` files (oracles / baselines only). ARC_SPV_DIR overrides.
pub fn arc_spv_dir() -> String {
    std::env::var("ARC_SPV_DIR").expect("ARC_SPV_DIR is not set: the folder of the reference OpenCL .spv modules (test oracles, not in this repository)")
}

/// Load an arc-engine oracle module (`xq4.spv`, `qwen_mebench.spv`, ...).
pub fn arc_module(ze: &Arc<Ze>, name: &str, flags: &str) -> fe_ze::Result<Module> {
    let p = format!("{}/{name}", arc_spv_dir());
    let b = std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
    Module::from_spirv(ze, &b, flags)
}

/// fp32 -> fp16 bits, round to nearest even (NaN -> quiet NaN).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 | (man >> 13) as u16 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

/// fp16 bits -> fp32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// fp32 -> bf16 bits by integer RNE (arc-engine `to_bf`).
pub fn f32_to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

/// Units in the last place between two finite fp32 values.
pub fn ulps(a: f32, b: f32) -> u64 {
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 { i64::from(i32::MIN) - i64::from(i) } else { i64::from(i) }
    };
    (key(a) - key(b)).unsigned_abs()
}

/// Deterministic xorshift for test data.
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 16) as u32
    }
    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_u32() % (hi - lo) as u32) as i32
    }
    pub fn f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Median of a sample (sorted copy).
pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}
