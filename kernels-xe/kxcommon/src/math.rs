//! Math with the rounding spelled out.
//!
//! * Plain Rust `a * b + c` is two roundings, always (no contraction: Rust semantics + ContractionOff on every kernel).
//! * [`fma`] is one rounding (`llvm.fma` -> OpenCL.std fma -> Xe `mad`, fused).
//! * [`mad`] is OpenCL `mad` (OpenCL.std mad; IGC emits the fused `mad`): use it where an OpenCL kernel being matched
//!   used `mad()` or where clang contracted `a * b + c` (FP_CONTRACT ON, no ContractionOff in its SPIR-V).
//! * Division: Rust `/` becomes a plain OpFDiv, which IGC compiles at OpenCL precision (<= 2.5 ulp; measured: bit-identical
//!   to OpenCL C's `/` on 64K random pairs, max 2 ulp from IEEE). [`div_ocl`] spells the same thing explicitly (OpFDiv +
//!   FPMaxErrorDecorationINTEL 2.5, what clang + llvm-spirv emit for OpenCL C). For IEEE division build the module with
//!   `-cl-fp32-correctly-rounded-divide-sqrt` (then both are correctly rounded).
//! * `exp`, `rsqrt`, `sqrt`, ...: the OpenCL builtins at OpenCL precision (OpenCL.std), `native_*` the hardware forms.
use crate::ffi::*;

#[inline(always)]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    core::intrinsics::fma(a, b, c)
}
#[inline(always)]
pub fn mad(a: f32, b: f32, c: f32) -> f32 {
    unsafe { _Z3madfff(a, b, c) }
}
/// OpenCL C `a / b` (2.5 ulp).
#[inline(always)]
pub fn div_ocl(a: f32, b: f32) -> f32 {
    unsafe { __xe_fdiv_ocl(a, b) }
}
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    unsafe { _Z3expf(x) }
}
#[inline(always)]
pub fn exp2(x: f32) -> f32 {
    unsafe { _Z4exp2f(x) }
}
#[inline(always)]
pub fn log(x: f32) -> f32 {
    unsafe { _Z3logf(x) }
}
#[inline(always)]
pub fn log2(x: f32) -> f32 {
    unsafe { _Z4log2f(x) }
}
#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    unsafe { _Z5rsqrtf(x) }
}
/// OpenCL `sqrt` (3 ulp unless the module is built with -cl-fp32-correctly-rounded-divide-sqrt).
#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    unsafe { _Z4sqrtf(x) }
}
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    unsafe { _Z4tanhf(x) }
}
/// OpenCL `fmax` (NaN-ignoring max).
#[inline(always)]
pub fn fmax(a: f32, b: f32) -> f32 {
    unsafe { _Z4fmaxff(a, b) }
}
#[inline(always)]
pub fn fmin(a: f32, b: f32) -> f32 {
    unsafe { _Z4fminff(a, b) }
}
#[inline(always)]
pub fn fabs(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}
#[inline(always)]
pub fn native_exp(x: f32) -> f32 {
    unsafe { _Z10native_expf(x) }
}
#[inline(always)]
pub fn native_exp2(x: f32) -> f32 {
    unsafe { _Z11native_exp2f(x) }
}
#[inline(always)]
pub fn native_log2(x: f32) -> f32 {
    unsafe { _Z11native_log2f(x) }
}
#[inline(always)]
pub fn native_recip(x: f32) -> f32 {
    unsafe { _Z12native_recipf(x) }
}
#[inline(always)]
pub fn native_rsqrt(x: f32) -> f32 {
    unsafe { _Z12native_rsqrtf(x) }
}
#[inline(always)]
pub fn native_sqrt(x: f32) -> f32 {
    unsafe { _Z11native_sqrtf(x) }
}
#[inline(always)]
pub fn native_divide(a: f32, b: f32) -> f32 {
    unsafe { _Z13native_divideff(a, b) }
}

/// OpenCL `dot` (OpenCL.std / OpDot: IGC's own summation order, which a ported kernel must keep).
#[inline(always)]
pub fn dot2(a: crate::v::F32x2, b: crate::v::F32x2) -> f32 {
    unsafe { _Z3dotDv2_fS_(a, b) }
}
#[inline(always)]
pub fn dot4(a: crate::v::F32x4, b: crate::v::F32x4) -> f32 {
    unsafe { _Z3dotDv4_fS_(a, b) }
}
#[inline(always)]
pub fn log1p(x: f32) -> f32 {
    unsafe { _Z5log1pf(x) }
}
/// Round to nearest even integer (OpenCL `rint`).
#[inline(always)]
pub fn rint(x: f32) -> f32 {
    unsafe { _Z4rintf(x) }
}
#[inline(always)]
pub fn trunc(x: f32) -> f32 {
    unsafe { _Z5truncf(x) }
}
#[inline(always)]
pub fn floor(x: f32) -> f32 {
    unsafe { _Z5floorf(x) }
}
#[inline(always)]
pub fn ceil(x: f32) -> f32 {
    unsafe { _Z4ceilf(x) }
}

/// OpenCL `bitselect(a, b, c)` on u32: (a & !c) | (b & c), bit for bit. An opaque call, so LLVM keeps the select form and IGC
/// can emit one 3-input bfn (an and / or chain written in Rust gets canonicalized into two ands and an or).
#[inline(always)]
pub fn bitselect(a: u32, b: u32, c: u32) -> u32 {
    unsafe { crate::ffi::_Z9bitselectjjj(a, b, c) }
}
