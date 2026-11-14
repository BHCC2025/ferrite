//! k-gguf: ggml quant kernels in Rust (nvptx). Built to one PTX module (`k_gguf`) by crates/fe-kbuild; loaded by
//! crates/fe-ggq.
//!
//! - `dec`: per-type block decoders (8 values of a block, as f32) shared by every kernel.
//! - `deq`: dequant to f32 / bf16 (`ggq_deq_<ty>_{f32,bf16}`).
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc, clippy::needless_range_loop)]

extern crate kcommon;

pub mod tables;
pub mod dec;
pub mod deq;
pub mod vq;
pub mod gemv;
pub mod moe;
pub mod gemm;
pub mod mmq;
pub mod sort;
pub mod bf16;
pub mod hc;
pub mod fa;
pub mod fa_ref;
pub mod gemm16;
