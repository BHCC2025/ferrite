//! Shared local memory (SLM, OpenCL `__local`) and barriers.
//!
//! [`alloc`] gives a work-group-shared array: fe-kbuild-xe replaces the marker call with an `addrspace(3)` global
//! (a SPIR-V Workgroup variable, sized statically, allocated by IGC per kernel). `bytes` and `id` must be constants;
//! two regions of one kernel need different ids. Loads and stores through the pointer are SLM accesses after
//! `infer-address-spaces`. A kernel may instead take a `*mut T` parameter set with a size and no data on the host
//! (dynamic SLM, `zeKernelSetArgumentValue(i, bytes, null)`): see [`crate::attr::local_arg`].
use crate::ffi::*;

/// `CLK_LOCAL_MEM_FENCE`.
pub const LOCAL: u32 = 1;
/// `CLK_GLOBAL_MEM_FENCE`.
pub const GLOBAL: u32 = 2;

/// Static SLM region of `N` elements of `T` (id `ID`), shared by the work-group.
#[inline(always)]
pub fn alloc<T, const N: usize, const ID: u32>() -> *mut T {
    unsafe { __xe_slm((N * core::mem::size_of::<T>()) as u32, ID) as *mut T }
}

/// `barrier(CLK_LOCAL_MEM_FENCE)`: every work-item of the group arrives; SLM writes before are visible after.
#[inline(always)]
pub fn barrier() {
    unsafe { _Z7barrierj(LOCAL) }
}
/// `barrier(CLK_GLOBAL_MEM_FENCE)`.
#[inline(always)]
pub fn barrier_global() {
    unsafe { _Z7barrierj(GLOBAL) }
}
/// `barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)`.
#[inline(always)]
pub fn barrier_all() {
    unsafe { _Z7barrierj(LOCAL | GLOBAL) }
}
/// Split barrier, first half (`intel_work_group_barrier_arrive`, cl_intel_split_work_group_barrier).
#[inline(always)]
pub fn barrier_arrive(flags: u32) {
    unsafe { _Z31intel_work_group_barrier_arrivej(flags) }
}
/// Split barrier, second half (`intel_work_group_barrier_wait`).
#[inline(always)]
pub fn barrier_wait(flags: u32) {
    unsafe { _Z29intel_work_group_barrier_waitj(flags) }
}
/// `sub_group_barrier(CLK_LOCAL_MEM_FENCE)`.
#[inline(always)]
pub fn sg_barrier() {
    unsafe { _Z17sub_group_barrierj(LOCAL) }
}
