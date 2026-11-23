//! Inkling-2-Mini model kernels: norms, short convolutions, attention with relative bias and sinks, attention residuals,
//! message gates and the router. Reference-grade arithmetic (the order of every sum is fixed); the tuned expert / dense kernels live in k-ink.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

extern crate kcommon;

pub mod attn_tc;
pub mod attn_dec;
pub mod attn_dec2;
pub mod conv;
pub mod gemm;
pub mod gate;
pub mod norm;

use kcommon::math;

/// bf16 bits -> f32.
#[inline(always)]
pub fn bf(h: u16) -> f32 {
    kcommon::bf16::to_f32(h)
}
/// f32 -> bf16 bits (RNE).
#[inline(always)]
pub fn tb(f: f32) -> u16 {
    kcommon::bf16::from_f32(f)
}
/// Round through bf16.
#[inline(always)]
pub fn rb(f: f32) -> f32 {
    kcommon::bf16::round(f)
}
/// exp without MUFU approximation error (rel 2e-7).
#[inline(always)]
pub fn ex(x: f32) -> f32 {
    math::exp_det(x)
}
#[inline(always)]
pub fn sigm(x: f32) -> f32 {
    1.0 / (1.0 + ex(-x))
}
/// tanh to ~1e-7 absolute.
#[inline(always)]
pub fn th(x: f32) -> f32 {
    let x = if x > 20.0 { 20.0 } else if x < -20.0 { -20.0 } else { x };
    1.0 - 2.0 / (ex(2.0 * x) + 1.0)
}

/// Programmatic dependent launch: wait for the previous grid, then let the next one start (no-ops under plain launches).
#[inline(always)]
pub fn pdl_wait() {
    unsafe { core::arch::asm!("griddepcontrol.wait;", options(nostack)) }
}
#[inline(always)]
pub fn pdl_trigger() {
    unsafe { core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack)) }
}
