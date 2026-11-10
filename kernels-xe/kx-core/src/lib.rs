//! The non-GEMV kernels of the Qwen3.8-27B forward on the Arc Pro B70: Rust ports of arc-engine's OpenCL
//! (`kernels/xpu/qwen_rows.cl`, `qwen_basic.cl`, `qwen_kvq.cl`, `qwen_attn_long.cl`, `qwen_attn_dec.cl`), operation for
//! operation, so every output is bit-identical to the reference `.spv` on the same inputs. Kernel names, arguments and
//! launch geometry are the reference's (crates/fe-kxcore has the launch wrappers).
//!
//! Floating point follows the reference SPIR-V exactly (`spirv-dis` of the ocloc build; fe-xe DESIGN.md section 2):
//! * where clang contracted `a * b + c` (an `OpExtInst mad` in the reference) the port writes [`math::mad`]; where the
//!   source wrote `fma()` it writes [`math::fma`]; every other product and sum is two roundings (Rust never contracts and
//!   the module has ContractionOff, as the reference kernels have);
//! * `/` in the OpenCL source is the 2.5-ulp OpenCL division ([`math::div_ocl`]);
//! * exp / log1p / rsqrt / sqrt / rint / fmax / fabs / dot / sub-group reductions are the same OpenCL builtins (same
//!   SPIR-V instructions), so IGC picks the same implementations and reduction trees.
//!
//! Two modules: the default build (kx_core.spv) has everything except the matrix-engine attention kernels, which are the
//! `dec` feature build (kx_core_dec.spv, loaded with `-cl-intel-256-GRF-per-thread` like arc-engine's qwen_attn_dec_q8.spv
//! and qwen_attn_pfxi_q8.spv): the decode attention (dec.rs) and the prompt-window attention (pf.rs).
#![no_std]
#![feature(abi_gpu_kernel, portable_simd, f16, simd_ffi, stmt_expr_attributes)]
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments, improper_ctypes)]

pub mod ext;

#[cfg(not(feature = "dec"))]
pub mod basic;
#[cfg(not(feature = "dec"))]
pub mod kvq;
#[cfg(not(feature = "dec"))]
pub mod long;
#[cfg(not(feature = "dec"))]
pub mod one;
#[cfg(not(feature = "dec"))]
pub mod spec;
#[cfg(not(feature = "dec"))]
pub mod tree;
#[cfg(not(feature = "dec"))]
pub mod gdnpre;
#[cfg(not(feature = "dec"))]
pub mod rows;

#[cfg(feature = "dec")]
pub mod dec;
#[cfg(feature = "dec")]
pub mod pf;

use kxcommon::{cvt, math};

/// bf16 bits -> fp32 (`bf` of the OpenCL kernels).
#[inline(always)]
pub fn bf(v: u16) -> f32 {
    cvt::bf16_to_f32(v)
}
/// fp32 -> bf16 bits, RNE by integer add (`to_bf`).
#[inline(always)]
pub fn to_bf(f: f32) -> u16 {
    cvt::f32_to_bf16(f)
}
/// fp32 rounded to bf16 and widened back (`bfr`).
#[inline(always)]
pub fn bfr(f: f32) -> f32 {
    bf(to_bf(f))
}
/// `x / (1.0f + exp(-x))`: OpFNegate, exp, OpFAdd, 2.5-ulp OpFDiv.
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    math::div_ocl(x, 1.0 + math::exp(-x))
}
/// `1.0f / (1.0f + exp(-x))` (the sigmoid of the gates).
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    math::div_ocl(1.0, 1.0 + math::exp(-x))
}
