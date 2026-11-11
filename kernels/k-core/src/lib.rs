//! Core model kernels: norms, rope, attention, GDN, sampling, embeddings, bf16 GEMV/GEMM.
//! Built to one PTX module (`k_core`) by crates/fe-kbuild; loaded by fe-model.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

extern crate kcommon;

pub mod attn;
pub mod dims;
pub mod flash;
pub mod gdn;
pub mod gemm;
pub mod kvq;
pub mod misc;
pub mod selftest;
