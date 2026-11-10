//! Device-side helpers for the Rust kernels of the Intel Arc Pro B70 (Xe2), compiled to SPIR-V by crates/fe-kbuild-xe.
//!
//! Every function is `#[inline(always)]`. The hardware features are reached through the OpenCL builtins IGC resolves
//! by name, called exactly as clang + llvm-spirv call them for OpenCL C (same mangled names, same SPIR-V forms as the
//! arc-engine `.spv` files): work-item / group / sub-group ids ([`id`]), sub-group shuffles, broadcasts, reductions
//! and scans ([`sg`]), local memory and (split) barriers ([`slm`]), sub-group block reads/writes ([`blk`]), DPAS on the
//! matrix engine ([`dpas`]), 2D block load / store / prefetch ([`blk2d`]), fp16 / bf16 conversions ([`cvt`]), and math
//! with explicit fused / unfused / OpenCL-precision forms ([`math`]), and OpenCL vloadN / vstoreN ([`mem`]).
//!
//! Floating point: Rust never contracts `a * b + c`, and fe-kbuild-xe compiles every kernel with ContractionOff, so a
//! product and a sum stay two roundings unless the kernel says [`math::fma`] or [`math::mad`]. Rust `/` is the IEEE
//! division; [`math::div_ocl`] is OpenCL C's default 2.5-ulp division (what `a / b` means in the arc-engine kernels).
//!
//! Kernels are `#[unsafe(no_mangle)] pub unsafe extern "gpu-kernel" fn name(..)` with pointer and scalar parameters
//! only; pointer parameters arrive as global (`__global`) pointers. The sub-group size is 16 unless the kernel calls
//! [`attr::sg_size`]; [`attr::reqd_work_group_size`] fixes the work-group size (both must be the kernel's first calls).
#![no_std]
#![feature(simd_ffi, portable_simd, f16, core_intrinsics)]
#![allow(internal_features, improper_ctypes, clippy::missing_safety_doc)]

pub mod attr;
pub mod blk;
pub mod blk2d;
pub mod cvt;
pub mod dpas;
pub mod ffi;
pub mod id;
pub mod math;
pub mod mem;
pub mod sg;
pub mod slm;

pub use core::simd::Simd;

/// Vector types (LLVM / SPIR-V vectors; what the builtins take and return).
pub mod v {
    use core::simd::Simd;
    pub type F32x2 = Simd<f32, 2>;
    pub type F32x4 = Simd<f32, 4>;
    pub type F32x8 = Simd<f32, 8>;
    pub type I32x2 = Simd<i32, 2>;
    pub type I32x4 = Simd<i32, 4>;
    pub type I32x8 = Simd<i32, 8>;
    pub type U32x2 = Simd<u32, 2>;
    pub type U32x4 = Simd<u32, 4>;
    pub type U32x8 = Simd<u32, 8>;
    pub type U32x16 = Simd<u32, 16>;
    pub type U32x32 = Simd<u32, 32>;
    pub type I16x2 = Simd<i16, 2>;
    pub type I16x4 = Simd<i16, 4>;
    pub type I16x8 = Simd<i16, 8>;
    pub type U16x2 = Simd<u16, 2>;
    pub type U16x4 = Simd<u16, 4>;
    pub type U16x8 = Simd<u16, 8>;
    pub type U16x16 = Simd<u16, 16>;
    pub type U16x32 = Simd<u16, 32>;
    pub type U16x64 = Simd<u16, 64>;
    pub type U8x2 = Simd<u8, 2>;
    pub type U8x4 = Simd<u8, 4>;
    pub type U8x8 = Simd<u8, 8>;
    pub type U8x16 = Simd<u8, 16>;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // llvm.trap is dropped by the retarget pass: a panic path is unreachable code (release kernels have no overflow or
    // debug checks; a failed bounds check is undefined behaviour, as in OpenCL C).
    core::intrinsics::abort_immediate()
}

/// Reinterpret the bits of one Copy value as another of the same size (vector <-> vector of another lane type): one
/// LLVM `bitcast`. (A byte-wise copy here makes LLVM build i128 / i256 values, which SPIR-V cannot express.)
#[inline(always)]
pub fn bits<A: Copy, B: Copy>(a: A) -> B {
    const { assert!(core::mem::size_of::<A>() == core::mem::size_of::<B>()) };
    unsafe { core::intrinsics::transmute_unchecked::<A, B>(a) }
}

/// `unroll!(MAX, n, |i| body)`: `body` for i = 0 .. n - 1 as straight-line code (OpenCL's `#pragma unroll`). `MAX` is a
/// literal bound (1, 2, 4, 8, 16, 32 or 64), `n <= MAX`. The body is pasted MAX times with `i` bound to a literal
/// (no closure: nothing for LLVM to inline or decline to inline), each copy guarded by `i < n`; with `n` a constant after
/// monomorphization the guards fold, every index is a constant and arrays indexed by `i` stay in registers. A generic
/// helper with a fixed 64 copies made nested unrolls explode (LLVM inlines all 64 before folding: minutes per kernel).
#[macro_export]
macro_rules! unroll {
    (@steps $n:ident, $i:ident, $body:expr, [$($k:literal)*]) => {{ $( if $k < $n { let $i: usize = $k; $body; } )* }};
    (1, $n:expr, |$i:ident| $body:expr) => {{ let n: usize = $n; $crate::unroll!(@steps n, $i, $body, [0]) }};
    (2, $n:expr, |$i:ident| $body:expr) => {{ let n: usize = $n; $crate::unroll!(@steps n, $i, $body, [0 1]) }};
    (4, $n:expr, |$i:ident| $body:expr) => {{ let n: usize = $n; $crate::unroll!(@steps n, $i, $body, [0 1 2 3]) }};
    (8, $n:expr, |$i:ident| $body:expr) => {{ let n: usize = $n; $crate::unroll!(@steps n, $i, $body, [0 1 2 3 4 5 6 7]) }};
    (16, $n:expr, |$i:ident| $body:expr) => {{
        let n: usize = $n;
        $crate::unroll!(@steps n, $i, $body, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15])
    }};
    (32, $n:expr, |$i:ident| $body:expr) => {{
        let n: usize = $n;
        $crate::unroll!(@steps n, $i, $body, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31])
    }};
    (64, $n:expr, |$i:ident| $body:expr) => {{
        let n: usize = $n;
        $crate::unroll!(@steps n, $i, $body, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
            32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63])
    }};
}
