//! Shared device-side helpers for every ferrite kernel crate (nvptx64-nvidia-cuda, sm_121a).
//!
//! Everything here is `#[inline(always)]`: thread/block ids, barriers, warp shuffles and reductions,
//! shared memory (dynamic + static) with real `ld.shared`/`st.shared`, bf16 conversions, fast math,
//! and tensor-core wrappers (`mma.sync` bf16 m16n8k16, `ldmatrix`, `cp.async`). See kernels/README.md.
#![no_std]
#![feature(stdarch_nvptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

pub mod bf16;
pub mod math;
pub mod mma;
pub mod smem;
pub mod warp;

pub use core::arch::nvptx;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { core::arch::asm!("trap;", options(noreturn)) }
}

// ---- ids --------------------------------------------------------------------------------------

#[inline(always)]
pub fn tid_x() -> u32 {
    unsafe { nvptx::_thread_idx_x() as u32 }
}
#[inline(always)]
pub fn tid_y() -> u32 {
    unsafe { nvptx::_thread_idx_y() as u32 }
}
#[inline(always)]
pub fn bid_x() -> u32 {
    unsafe { nvptx::_block_idx_x() as u32 }
}
#[inline(always)]
pub fn bid_y() -> u32 {
    unsafe { nvptx::_block_idx_y() as u32 }
}
#[inline(always)]
pub fn bid_z() -> u32 {
    unsafe { nvptx::_block_idx_z() as u32 }
}
#[inline(always)]
pub fn bdim_x() -> u32 {
    unsafe { nvptx::_block_dim_x() as u32 }
}
#[inline(always)]
pub fn bdim_y() -> u32 {
    unsafe { nvptx::_block_dim_y() as u32 }
}
#[inline(always)]
pub fn gdim_x() -> u32 {
    unsafe { nvptx::_grid_dim_x() as u32 }
}
#[inline(always)]
pub fn gdim_z() -> u32 {
    unsafe { nvptx::_grid_dim_z() as u32 }
}
#[inline(always)]
pub fn gdim_y() -> u32 {
    unsafe { nvptx::_grid_dim_y() as u32 }
}
/// Lane within the warp (assumes blockDim.x is a multiple of 32 for 1-D blocks).
#[inline(always)]
pub fn lane() -> u32 {
    tid_x() & 31
}
#[inline(always)]
pub fn warp_id() -> u32 {
    tid_x() >> 5
}
#[inline(always)]
pub fn nwarps() -> u32 {
    bdim_x() >> 5
}

// ---- barriers ---------------------------------------------------------------------------------

/// `bar.sync 0` (__syncthreads).
#[inline(always)]
pub fn sync_threads() {
    unsafe { nvptx::_syncthreads() }
}
/// `bar.warp.sync 0xffffffff` (__syncwarp).
#[inline(always)]
pub fn sync_warp() {
    unsafe { core::arch::asm!("bar.warp.sync 0xffffffff;") }
}
/// Named barrier `bar.sync id, nthreads` for a subset of warps.
#[inline(always)]
pub fn bar_sync(id: u32, nthreads: u32) {
    unsafe { core::arch::asm!("bar.sync {i}, {n};", i = in(reg32) id, n = in(reg32) nthreads) }
}
#[inline(always)]
pub fn fence_block() {
    unsafe { core::arch::asm!("fence.cta;") }
}

// ---- global memory helpers --------------------------------------------------------------------

/// 128-bit global load through the read-only/non-coherent path (`ld.global.nc.v4.u32`).
#[inline(always)]
pub unsafe fn ldg_nc_v4(p: *const u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    core::arch::asm!("ld.global.nc.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
        p = in(reg64) p, a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d,
        options(readonly, nostack, pure));
    [a, b, c, d]
}
/// Streaming 128-bit global load that bypasses L1 allocation (`ld.global.nc.L1::no_allocate`).
#[inline(always)]
pub unsafe fn ldg_stream_v4(p: *const u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    core::arch::asm!("ld.global.nc.L1::no_allocate.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];",
        p = in(reg64) p, a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d,
        options(readonly, nostack, pure));
    [a, b, c, d]
}
#[inline(always)]
pub unsafe fn stg_v4(p: *mut u32, v: [u32; 4]) {
    core::arch::asm!("st.global.v4.u32 [{p}], {{{a}, {b}, {c}, {d}}};",
        p = in(reg64) p, a = in(reg32) v[0], b = in(reg32) v[1], c = in(reg32) v[2], d = in(reg32) v[3],
        options(nostack));
}
#[inline(always)]
pub unsafe fn atomic_add_f32(p: *mut f32, v: f32) {
    core::arch::asm!("red.global.add.f32 [{p}], {v};", p = in(reg64) p, v = in(reg32) v, options(nostack));
}
#[inline(always)]
pub unsafe fn atomic_add_u32(p: *mut u32, v: u32) -> u32 {
    let old: u32;
    core::arch::asm!("atom.global.add.u32 {o}, [{p}], {v};", o = out(reg32) old, p = in(reg64) p, v = in(reg32) v,
        options(nostack));
    old
}

// ---- unrolling --------------------------------------------------------------------------------

/// Fully unroll a loop with a compile-time trip count: `unroll!(i in 8 => { acc[i] += x[i]; })`.
/// `i` is a `usize` constant in each copy, so register arrays indexed by it stay in registers
/// (LLVM does not reliably unroll `while` loops containing asm, and a dynamically indexed array goes
/// to local memory). Supported counts: 1, 2, 3, 4, 6, 8, 16, 32.
#[macro_export]
macro_rules! unroll {
    ($i:ident in [$($n:literal),*] => $body:block) => { $( { #[allow(non_upper_case_globals)] const $i: usize = $n; $body } )* };
    ($i:ident in 1 => $body:block) => { $crate::unroll!($i in [0] => $body) };
    ($i:ident in 2 => $body:block) => { $crate::unroll!($i in [0, 1] => $body) };
    ($i:ident in 3 => $body:block) => { $crate::unroll!($i in [0, 1, 2] => $body) };
    ($i:ident in 4 => $body:block) => { $crate::unroll!($i in [0, 1, 2, 3] => $body) };
    ($i:ident in 6 => $body:block) => { $crate::unroll!($i in [0, 1, 2, 3, 4, 5] => $body) };
    ($i:ident in 8 => $body:block) => { $crate::unroll!($i in [0, 1, 2, 3, 4, 5, 6, 7] => $body) };
    ($i:ident in 16 => $body:block) => {
        $crate::unroll!($i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] => $body)
    };
    ($i:ident in 32 => $body:block) => {
        $crate::unroll!($i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31] => $body)
    };
}
