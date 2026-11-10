//! Sub-group shuffles, broadcasts, reductions and scans (cl_intel_subgroups + OpenCL 2.0 sub-groups). Shuffles become
//! OpSubgroupShuffle*INTEL, collectives OpGroup* (IGC: register regioning / `mov` across lanes, no memory).
use crate::ffi::*;

/// Values that can move between lanes.
pub trait Shuffle: Copy {
    /// Value of lane `src` (`intel_sub_group_shuffle`).
    fn shuffle(self, src: u32) -> Self;
    /// Value of lane `lane ^ m` (`intel_sub_group_shuffle_xor`).
    fn shuffle_xor(self, m: u32) -> Self;
}
impl Shuffle for f32 {
    #[inline(always)]
    fn shuffle(self, src: u32) -> f32 {
        unsafe { _Z23intel_sub_group_shufflefj(self, src) }
    }
    #[inline(always)]
    fn shuffle_xor(self, m: u32) -> f32 {
        unsafe { _Z27intel_sub_group_shuffle_xorfj(self, m) }
    }
}
impl Shuffle for u32 {
    #[inline(always)]
    fn shuffle(self, src: u32) -> u32 {
        unsafe { _Z23intel_sub_group_shufflejj(self, src) }
    }
    #[inline(always)]
    fn shuffle_xor(self, m: u32) -> u32 {
        unsafe { _Z27intel_sub_group_shuffle_xorjj(self, m) }
    }
}
impl Shuffle for i32 {
    #[inline(always)]
    fn shuffle(self, src: u32) -> i32 {
        unsafe { _Z23intel_sub_group_shuffleij(self, src) }
    }
    #[inline(always)]
    fn shuffle_xor(self, m: u32) -> i32 {
        unsafe { _Z27intel_sub_group_shuffle_xorij(self, m) }
    }
}
impl Shuffle for u16 {
    #[inline(always)]
    fn shuffle(self, src: u32) -> u16 {
        unsafe { _Z23intel_sub_group_shuffletj(self, src) }
    }
    #[inline(always)]
    fn shuffle_xor(self, m: u32) -> u16 {
        // no ushort xor form in cl_intel_subgroups_short: through u32
        unsafe { _Z27intel_sub_group_shuffle_xorjj(self as u32, m) as u16 }
    }
}
impl Shuffle for f16 {
    #[inline(always)]
    fn shuffle(self, src: u32) -> f16 {
        unsafe { _Z23intel_sub_group_shuffleDhj(self, src) }
    }
    #[inline(always)]
    fn shuffle_xor(self, m: u32) -> f16 {
        let b: u16 = crate::bits(self);
        crate::bits(b.shuffle_xor(m))
    }
}

/// `intel_sub_group_shuffle_down(cur, next, d)`: lane l gets lane l + d of the 2 x sub-group window (cur ++ next).
#[inline(always)]
pub fn shuffle_down_f32(cur: f32, next: f32, d: u32) -> f32 {
    unsafe { _Z28intel_sub_group_shuffle_downffj(cur, next, d) }
}
/// `intel_sub_group_shuffle_up(prev, cur, d)`.
#[inline(always)]
pub fn shuffle_up_f32(prev: f32, cur: f32, d: u32) -> f32 {
    unsafe { _Z26intel_sub_group_shuffle_upffj(prev, cur, d) }
}

#[inline(always)]
pub fn broadcast_f32(v: f32, lane: u32) -> f32 {
    unsafe { _Z19sub_group_broadcastfj(v, lane) }
}
#[inline(always)]
pub fn broadcast_u32(v: u32, lane: u32) -> u32 {
    unsafe { _Z19sub_group_broadcastjj(v, lane) }
}

/// `sub_group_reduce_add` (IGC's fixed reduction tree; the same order as the OpenCL kernels that use it).
#[inline(always)]
pub fn reduce_add_f32(v: f32) -> f32 {
    unsafe { _Z20sub_group_reduce_addf(v) }
}
#[inline(always)]
pub fn reduce_add_u32(v: u32) -> u32 {
    unsafe { _Z20sub_group_reduce_addj(v) }
}
#[inline(always)]
pub fn reduce_add_i32(v: i32) -> i32 {
    unsafe { _Z20sub_group_reduce_addi(v) }
}
#[inline(always)]
pub fn reduce_max_f32(v: f32) -> f32 {
    unsafe { _Z20sub_group_reduce_maxf(v) }
}
#[inline(always)]
pub fn reduce_max_u32(v: u32) -> u32 {
    unsafe { _Z20sub_group_reduce_maxj(v) }
}
#[inline(always)]
pub fn reduce_min_f32(v: f32) -> f32 {
    unsafe { _Z20sub_group_reduce_minf(v) }
}
#[inline(always)]
pub fn reduce_min_u32(v: u32) -> u32 {
    unsafe { _Z20sub_group_reduce_minj(v) }
}
#[inline(always)]
pub fn scan_incl_add_f32(v: f32) -> f32 {
    unsafe { _Z28sub_group_scan_inclusive_addf(v) }
}
#[inline(always)]
pub fn scan_incl_add_u32(v: u32) -> u32 {
    unsafe { _Z28sub_group_scan_inclusive_addj(v) }
}
#[inline(always)]
pub fn scan_excl_add_f32(v: f32) -> f32 {
    unsafe { _Z28sub_group_scan_exclusive_addf(v) }
}
#[inline(always)]
pub fn any(p: bool) -> bool {
    unsafe { _Z13sub_group_anyi(p as i32) != 0 }
}
#[inline(always)]
pub fn all(p: bool) -> bool {
    unsafe { _Z13sub_group_alli(p as i32) != 0 }
}

/// Butterfly sum with shuffle_xor in a fixed order (lane-independent result): every lane gets the same total.
#[inline(always)]
pub fn butterfly_sum16(mut v: f32) -> f32 {
    v += v.shuffle_xor(8);
    v += v.shuffle_xor(4);
    v += v.shuffle_xor(2);
    v += v.shuffle_xor(1);
    v
}

#[inline(always)]
pub fn reduce_max_u64(v: u64) -> u64 {
    unsafe { _Z20sub_group_reduce_maxm(v) }
}
#[inline(always)]
pub fn reduce_min_u64(v: u64) -> u64 {
    unsafe { _Z20sub_group_reduce_minm(v) }
}
#[inline(always)]
pub fn reduce_add_u64(v: u64) -> u64 {
    unsafe { _Z20sub_group_reduce_addm(v) }
}
