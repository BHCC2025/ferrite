//! DPAS: the Xe2 matrix engine (`dpas.8x8` at sub-group size 16), through the cl_intel_subgroup_matrix_multiply_accumulate
//! builtins IGC lowers to `dpas` (the same calls ocloc's SPIR-V makes).
//!
//! Shapes (sub-group 16): C[M x 16] += A[M x K] * B[K x 16], M = 1, 2, 4, 8 rows.
//! * A: row r of A is held by the sub-group as one 16-bit value per lane (lane = k pair index; fp16/bf16: lane l holds
//!   A[r][l] for K = 16 ... see the extension spec: for 16-bit types one `short` a lane = one element; for int8 one `short`
//!   a lane = two int8 elements, K = 32).
//! * B: 8 dwords per lane (VNNI-packed K x 16 tile: dword d of lane n = B[2d][n], B[2d+1][n] for 16-bit types; four
//!   int8 values for int8).
//! * C: M floats (or ints) per lane, lane = column, component = row.
//! The forms with M < 8 keep the unused accumulator rows out of the chain (xq.cl `dpas_m`).
use crate::ffi::*;
use crate::v::*;

#[inline(always)]
pub fn f16_m1(a: i16, b: I32x8, c: f32) -> f32 {
    unsafe { _Z38intel_sub_group_f16_f16_matrix_mad_k16sDv8_if(a, b, c) }
}
#[inline(always)]
pub fn f16_m2(a: I16x2, b: I32x8, c: F32x2) -> F32x2 {
    unsafe { _Z38intel_sub_group_f16_f16_matrix_mad_k16Dv2_sDv8_iDv2_f(a, b, c) }
}
#[inline(always)]
pub fn f16_m4(a: I16x4, b: I32x8, c: F32x4) -> F32x4 {
    unsafe { _Z38intel_sub_group_f16_f16_matrix_mad_k16Dv4_sDv8_iDv4_f(a, b, c) }
}
#[inline(always)]
pub fn f16_m8(a: I16x8, b: I32x8, c: F32x8) -> F32x8 {
    unsafe { _Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(a, b, c) }
}
#[inline(always)]
pub fn bf16_m1(a: i16, b: I32x8, c: f32) -> f32 {
    unsafe { _Z40intel_sub_group_bf16_bf16_matrix_mad_k16sDv8_if(a, b, c) }
}
#[inline(always)]
pub fn bf16_m2(a: I16x2, b: I32x8, c: F32x2) -> F32x2 {
    unsafe { _Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv2_sDv8_iDv2_f(a, b, c) }
}
#[inline(always)]
pub fn bf16_m4(a: I16x4, b: I32x8, c: F32x4) -> F32x4 {
    unsafe { _Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv4_sDv8_iDv4_f(a, b, c) }
}
#[inline(always)]
pub fn bf16_m8(a: I16x8, b: I32x8, c: F32x8) -> F32x8 {
    unsafe { _Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iDv8_f(a, b, c) }
}
/// int8 x int8 -> int32, K = 32 (a: two int8 per lane per row, b: 8 dwords = 32 int8 per lane in VNNI order).
#[inline(always)]
pub fn i8_m1(a: i16, b: I32x8, c: i32) -> i32 {
    unsafe { _Z36intel_sub_group_i8_i8_matrix_mad_k32sDv8_ii(a, b, c) }
}
#[inline(always)]
pub fn i8_m2(a: I16x2, b: I32x8, c: I32x2) -> I32x2 {
    unsafe { _Z36intel_sub_group_i8_i8_matrix_mad_k32Dv2_sDv8_iDv2_i(a, b, c) }
}
#[inline(always)]
pub fn i8_m4(a: I16x4, b: I32x8, c: I32x4) -> I32x4 {
    unsafe { _Z36intel_sub_group_i8_i8_matrix_mad_k32Dv4_sDv8_iDv4_i(a, b, c) }
}
#[inline(always)]
pub fn i8_m8(a: I16x8, b: I32x8, c: I32x8) -> I32x8 {
    unsafe { _Z36intel_sub_group_i8_i8_matrix_mad_k32Dv8_sDv8_iS0_(a, b, c) }
}
