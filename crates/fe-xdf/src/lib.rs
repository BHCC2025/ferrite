//! fe-xdf: the DFlash2 block drafter (z-lab Qwen3.8-27B-DFlash2) on the Arc Pro B70, for fe-xe.
//!
//! - [`quant`]: the drafter linears as XQ tensors (Hadamard-128 RTN), so they run on the target's XQ kernels (kx-xq);
//! - [`draft`]: device weights, the draft K/V ring, `absorb` (target feature taps -> draft context) and `draft` (one block
//!   pass: 5 layers, the target lm_head over the first `span` ids, top-16, the candidate selector's walk);
//! - [`spec`]: speculative generation with DFlash2 drafts verified by fe-xe windows (greedy output == plain greedy).
//!
//! Kernels: kernels-xe/kx-dflash (dynamic conv, norm + RoPE + ring write, block attention, top-16, selector, feature tap);
//! everything else is kx-core / kx-xq through the engine's registry.
pub mod draft;
pub mod spec;
pub mod quant;
pub mod tree;
