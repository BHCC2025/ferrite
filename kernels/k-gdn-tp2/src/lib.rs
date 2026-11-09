//! One TP2 rank of k-gdn: its chunk.rs compiled with halved head counts (dims.rs here). Loaded by fe-model
//! (`fe_gdn::Gdn::load_tp2`) when a fabric is attached.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

extern crate kcommon;

#[path = "../../k-gdn/src/chunk.rs"]
pub mod chunk;
pub mod dims;
