//! Work-item, work-group and sub-group ids (OpenCL `get_*` builtins; IGC turns them into BuiltIn variables / payload
//! registers). Dimension `d` must be a constant 0, 1 or 2.
use crate::ffi::*;

/// Lane within the sub-group (`get_sub_group_local_id`).
#[inline(always)]
pub fn lane() -> u32 {
    unsafe { _Z22get_sub_group_local_idv() }
}
/// Sub-group index within the work-group (`get_sub_group_id`).
#[inline(always)]
pub fn sg_id() -> u32 {
    unsafe { _Z16get_sub_group_idv() }
}
/// Sub-groups in this work-group (`get_num_sub_groups`).
#[inline(always)]
pub fn num_sg() -> u32 {
    unsafe { _Z18get_num_sub_groupsv() }
}
/// Sub-group size (`get_sub_group_size`).
#[inline(always)]
pub fn sg_size() -> u32 {
    unsafe { _Z18get_sub_group_sizev() }
}
#[inline(always)]
pub fn max_sg_size() -> u32 {
    unsafe { _Z22get_max_sub_group_sizev() }
}
#[inline(always)]
pub fn group_id(d: u32) -> u32 {
    unsafe { _Z12get_group_idj(d) as u32 }
}
#[inline(always)]
pub fn local_id(d: u32) -> u32 {
    unsafe { _Z12get_local_idj(d) as u32 }
}
#[inline(always)]
pub fn global_id(d: u32) -> usize {
    unsafe { _Z13get_global_idj(d) as usize }
}
#[inline(always)]
pub fn local_size(d: u32) -> u32 {
    unsafe { _Z14get_local_sizej(d) as u32 }
}
#[inline(always)]
pub fn num_groups(d: u32) -> u32 {
    unsafe { _Z14get_num_groupsj(d) as u32 }
}
#[inline(always)]
pub fn global_size(d: u32) -> usize {
    unsafe { _Z15get_global_sizej(d) as usize }
}
