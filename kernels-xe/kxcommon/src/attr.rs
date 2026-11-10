//! Kernel attributes. Call them first in the kernel body; fe-kbuild-xe turns them into kernel metadata
//! (`reqd_work_group_size` -> ExecutionMode LocalSize, `intel_reqd_sub_group_size` -> ExecutionMode SubgroupSize) and
//! removes the calls. Every kernel gets sub-group size 16 unless it asks for another.

/// `__attribute__((reqd_work_group_size(X, Y, Z)))`.
#[inline(always)]
pub fn reqd_work_group_size<const X: u32, const Y: u32, const Z: u32>() {
    unsafe { crate::ffi::__xe_attr_reqd_wg_size(X, Y, Z) }
}
/// `__attribute__((intel_reqd_sub_group_size(N)))` (16 or 32 on Xe2; 16 is the default).
#[inline(always)]
pub fn sg_size<const N: u32>() {
    unsafe { crate::ffi::__xe_attr_sg_size(N) }
}

/// Marks a kernel pointer parameter as `__local` (dynamic SLM: the host sets the argument with a size and no data).
/// Call it first thing on the raw parameter and use only the returned pointer: fe-kbuild-xe retypes the parameter to
/// a Workgroup pointer and replaces the call with the parameter itself.
#[inline(always)]
pub fn local_arg<T>(p: *mut T) -> *mut T {
    unsafe { crate::ffi::__xe_local_arg(p as *mut u8) as *mut T }
}
