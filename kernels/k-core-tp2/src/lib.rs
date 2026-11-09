//! One TP2 rank of the head-dependent k-core kernels: k-core's attn.rs, flash.rs and gdn.rs compiled with
//! halved head counts (dims.rs here). Loaded by fe-model next to k_core when a Fabric is attached.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

extern crate kcommon;

#[path = "../../k-core/src/attn.rs"]
pub mod attn;
pub mod dims;
#[path = "../../k-core/src/flash.rs"]
pub mod flash;
#[path = "../../k-core/src/gdn.rs"]
pub mod gdn;
