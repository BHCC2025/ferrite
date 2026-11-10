//! Thin array-valued wrappers over the kxcommon builtins the ports use (dot4, log1p, rint, reduce_max(ulong), the OpenCL
//! vloadN / vstoreN of kxcommon::mem), plus the two builtins kxcommon has no wrapper for in the form the reference SPIR-V
//! uses: `fabs` as the OpenCL.std ext-inst (kxcommon clears the sign bit: same bits) and nothing else. Every call lowers
//! to the SPIR-V instruction of the reference `.spv` (dot -> OpDot, sub_group_reduce_max(ulong) -> OpGroupUMax, vloadN ->
//! OpenCL.std vloadn, the rest OpenCL.std ext-insts).
use kxcommon::mem;
use kxcommon::v::{F32x4, F32x8, U16x8};

unsafe extern "C" {
    fn _Z4fabsf(x: f32) -> f32;
}

/// OpenCL `dot(float4, float4)` (OpDot; IGC fixed lowering order).
#[inline(always)]
pub fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    kxcommon::math::dot4(F32x4::from_array(a), F32x4::from_array(b))
}
/// OpenCL `log1p`.
#[inline(always)]
pub fn log1p(x: f32) -> f32 {
    kxcommon::math::log1p(x)
}
/// OpenCL `rint` (round to nearest even).
#[inline(always)]
pub fn rint(x: f32) -> f32 {
    kxcommon::math::rint(x)
}
/// OpenCL `fabs` (the ext-inst, as the reference; same bits as clearing the sign).
#[inline(always)]
pub fn fabs(x: f32) -> f32 {
    unsafe { _Z4fabsf(x) }
}
/// `sub_group_reduce_max(ulong)` (OpGroupUMax).
#[inline(always)]
pub fn reduce_max_u64(v: u64) -> u64 {
    kxcommon::sg::reduce_max_u64(v)
}
/// OpenCL `vload8(offset, p)` on global floats: one 8-dword message a work-item.
#[inline(always)]
pub unsafe fn vload8(offset: usize, p: *const f32) -> [f32; 8] {
    unsafe { mem::global::vload8_f32(offset, p) }.to_array()
}
/// OpenCL `vstore8(v, offset, p)` on global floats.
#[inline(always)]
pub unsafe fn vstore8(v: [f32; 8], offset: usize, p: *mut f32) {
    unsafe { mem::global::vstore8_f32(F32x8::from_array(v), offset, p) }
}
/// OpenCL `vload8(offset, p)` on global ushorts.
#[inline(always)]
pub unsafe fn vload8_u16(offset: usize, p: *const u16) -> [u16; 8] {
    unsafe { mem::global::vload8_u16(offset, p) }.to_array()
}
/// OpenCL `vload16(offset, p)` on global ushorts.
#[inline(always)]
pub unsafe fn vload16_u16(offset: usize, p: *const u16) -> [u16; 16] {
    unsafe { mem::global::vload16_u16(offset, p) }.to_array()
}
/// OpenCL `vstore8(v, offset, p)` on global ushorts.
#[inline(always)]
pub unsafe fn vstore8_u16(v: [u16; 8], offset: usize, p: *mut u16) {
    unsafe { mem::global::vstore8_u16(U16x8::from_array(v), offset, p) }
}
/// OpenCL `vload8(offset, p)` on local (SLM) floats (IGC: one `load.slm.d32x8t` for a uniform address).
#[inline(always)]
pub unsafe fn vload8_local(offset: usize, p: *const f32) -> [f32; 8] {
    unsafe { mem::local::vload8_f32(offset, p) }.to_array()
}
