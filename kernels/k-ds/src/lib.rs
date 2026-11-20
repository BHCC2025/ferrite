//! DeepSeek-V4.1 model kernels (fe-ds): hyper-connections (mixes + Sinkhorn, collapse, post), norms, RoPE from
//! tables, the FP8 / FP4 cache formats, window + sparse attention, the compressor, the indexer (scores, exact
//! top-k), the router and the per-expert MoE fallback glue. Correctness-first forms (D0/D1); the math follows the
//! R1 oracle (tools/ds/ref/dsv41_ref.py) op by op, including where it rounds to bf16.
//!
//! Row metadata: `meta[r] = (slot, pos)` u32 pairs. Caches are per layer `[slots][...]` with a slot stride.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(internal_features, clippy::missing_safety_doc)]

extern crate kcommon;

pub mod attn;
pub mod hc;
pub mod index;
pub mod moe;
pub mod norm;
pub mod q;
pub mod util;
