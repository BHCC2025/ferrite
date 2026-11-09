//! k-exl3: EXL3 trellis kernels in Rust for sm_121a (GB10). Built to one PTX module (`k_exl3`) by
//! crates/fe-kbuild; loaded by fe-exl3.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

extern crate kcommon;

pub mod gemv;
pub mod probe;
pub mod recon;
pub mod util;
