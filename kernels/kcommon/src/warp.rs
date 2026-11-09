//! Warp shuffles and reductions (full-warp mask).
use core::arch::asm;

#[inline(always)]
pub fn shfl_xor_u32(v: u32, mask: u32) -> u32 {
    let r: u32;
    unsafe {
        asm!("shfl.sync.bfly.b32 {r}, {v}, {m}, 0x1f, 0xffffffff;", r = out(reg32) r, v = in(reg32) v,
            m = in(reg32) mask, options(nomem, nostack, preserves_flags));
    }
    r
}
#[inline(always)]
pub fn shfl_down_u32(v: u32, delta: u32) -> u32 {
    let r: u32;
    unsafe {
        asm!("shfl.sync.down.b32 {r}, {v}, {d}, 0x1f, 0xffffffff;", r = out(reg32) r, v = in(reg32) v,
            d = in(reg32) delta, options(nomem, nostack, preserves_flags));
    }
    r
}
#[inline(always)]
pub fn shfl_up_u32(v: u32, delta: u32) -> u32 {
    let r: u32;
    unsafe {
        asm!("shfl.sync.up.b32 {r}, {v}, {d}, 0x0, 0xffffffff;", r = out(reg32) r, v = in(reg32) v,
            d = in(reg32) delta, options(nomem, nostack, preserves_flags));
    }
    r
}
/// Broadcast `v` from `src_lane`.
#[inline(always)]
pub fn shfl_idx_u32(v: u32, src_lane: u32) -> u32 {
    let r: u32;
    unsafe {
        asm!("shfl.sync.idx.b32 {r}, {v}, {s}, 0x1f, 0xffffffff;", r = out(reg32) r, v = in(reg32) v,
            s = in(reg32) src_lane, options(nomem, nostack, preserves_flags));
    }
    r
}
#[inline(always)]
pub fn shfl_xor(v: f32, mask: u32) -> f32 {
    f32::from_bits(shfl_xor_u32(v.to_bits(), mask))
}
#[inline(always)]
pub fn shfl_down(v: f32, delta: u32) -> f32 {
    f32::from_bits(shfl_down_u32(v.to_bits(), delta))
}
#[inline(always)]
pub fn shfl_idx(v: f32, src_lane: u32) -> f32 {
    f32::from_bits(shfl_idx_u32(v.to_bits(), src_lane))
}

/// Butterfly sum: every lane gets the total.
#[inline(always)]
pub fn sum(mut v: f32) -> f32 {
    v += shfl_xor(v, 16);
    v += shfl_xor(v, 8);
    v += shfl_xor(v, 4);
    v += shfl_xor(v, 2);
    v += shfl_xor(v, 1);
    v
}
/// Butterfly max: every lane gets the max.
#[inline(always)]
pub fn max(mut v: f32) -> f32 {
    v = v.max(shfl_xor(v, 16));
    v = v.max(shfl_xor(v, 8));
    v = v.max(shfl_xor(v, 4));
    v = v.max(shfl_xor(v, 2));
    v = v.max(shfl_xor(v, 1));
    v
}
/// Sum over groups of `W` adjacent lanes (W = 2, 4, 8, 16); every lane of a group gets the group total.
#[inline(always)]
pub fn sum_width<const W: u32>(mut v: f32) -> f32 {
    let mut m = W >> 1;
    while m > 0 {
        v += shfl_xor(v, m);
        m >>= 1;
    }
    v
}

/// Block-wide sum through a caller-provided shared scratch of >= 32 f32 (`scratch` is a shared address).
/// All threads of the block must call it; every thread gets the total. Contains two barriers.
#[inline(always)]
pub fn block_sum(v: f32, scratch: crate::smem::Shared<f32>) -> f32 {
    let w = sum(v);
    let lane = crate::lane();
    let wid = crate::warp_id();
    if lane == 0 {
        scratch.add(wid).st(w);
    }
    crate::sync_threads();
    let n = crate::nwarps();
    let mut t = if lane < n { scratch.add(lane).ld() } else { 0.0 };
    t = sum(t);
    crate::sync_threads();
    t
}
/// Block-wide max, same contract as [`block_sum`].
#[inline(always)]
pub fn block_max(v: f32, scratch: crate::smem::Shared<f32>) -> f32 {
    let w = max(v);
    let lane = crate::lane();
    let wid = crate::warp_id();
    if lane == 0 {
        scratch.add(wid).st(w);
    }
    crate::sync_threads();
    let n = crate::nwarps();
    let mut t = if lane < n { scratch.add(lane).ld() } else { f32::NEG_INFINITY };
    t = max(t);
    crate::sync_threads();
    t
}
