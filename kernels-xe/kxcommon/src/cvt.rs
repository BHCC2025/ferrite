//! fp16 / bf16 conversions. fp32 -> fp16 is `convert_half_rte` (OpFConvert + FPRoundingMode RTE, as xq.cl); fp16 ->
//! fp32 is exact. bf16 helpers are the bit operations the arc-engine kernels use (`bf` / `to_bf`: RNE, no NaN special
//! case) plus the hardware forms of cl_intel_bfloat16_conversions.
use crate::ffi::*;

/// fp32 -> fp16, round to nearest even (`convert_half_rte`).
#[inline(always)]
pub fn f16_rte(x: f32) -> f16 {
    unsafe { _Z16convert_half_rtef(x) }
}
/// fp32 -> fp16 bits, round to nearest even.
#[inline(always)]
pub fn f16_bits_rte(x: f32) -> u16 {
    crate::bits(f16_rte(x))
}
/// fp32 -> fp16, round toward zero.
#[inline(always)]
pub fn f16_rtz(x: f32) -> f16 {
    unsafe { _Z16convert_half_rtzf(x) }
}
/// fp16 bits -> fp32 (exact).
#[inline(always)]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let v: f16 = crate::bits(h);
    v as f32
}
/// bf16 bits -> fp32 (exact; `bf` in the arc-engine kernels).
#[inline(always)]
pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}
/// fp32 -> bf16 bits, round to nearest even by integer add (`to_bf` in the arc-engine kernels, bit-identical to it).
#[inline(always)]
pub fn f32_to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
}
/// fp32 -> bf16 bits by the hardware conversion (`intel_convert_bfloat16_as_ushort`, RNE with NaN handling).
#[inline(always)]
pub fn f32_to_bf16_hw(f: f32) -> u16 {
    unsafe { _Z32intel_convert_bfloat16_as_ushortf(f) }
}
/// bf16 bits -> fp32 by the hardware conversion (`intel_convert_as_bfloat16_float`).
#[inline(always)]
pub fn bf16_to_f32_hw(v: u16) -> f32 {
    unsafe { _Z31intel_convert_as_bfloat16_floatt(v) }
}
/// fp32 -> int8, round to nearest even, no saturation (`convert_char_rte`: out-of-range is undefined, as in OpenCL).
#[inline(always)]
pub fn i8_rte(x: f32) -> i8 {
    unsafe { _Z16convert_char_rtef(x) }
}
/// fp32 -> int8, round to nearest even, saturating (`convert_char_sat_rte`).
#[inline(always)]
pub fn i8_sat_rte(x: f32) -> i8 {
    unsafe { _Z20convert_char_sat_rtef(x) }
}
/// fp32 -> int32, round to nearest even (`convert_int_rte`).
#[inline(always)]
pub fn i32_rte(x: f32) -> i32 {
    unsafe { _Z15convert_int_rtef(x) }
}

/// fp32 -> int32, round to nearest even, saturating (`convert_int_sat_rte`; NaN -> 0).
#[inline(always)]
pub fn i32_sat_rte(x: f32) -> i32 {
    unsafe { _Z19convert_int_sat_rtef(x) }
}

/// float8 -> int8 vector of i32, round to nearest even, saturating (`convert_int8_sat_rte`): whole-vector, so a DPAS accumulator
/// it rewrites stays one register block (per-element conversions make IGC split the accumulator into scalars and spill).
#[inline(always)]
pub fn i32x8_sat_rte(x: crate::v::F32x8) -> crate::v::I32x8 {
    unsafe { _Z20convert_int8_sat_rteDv8_f(x) }
}
