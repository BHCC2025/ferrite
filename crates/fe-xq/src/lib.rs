//! XQ: the weight format of arc-engine (unpublished Zig + OpenCL C engine for the B70; `src/xpu/xq.zig`, `kernels/xpu/xq.cl`), in Rust.
//!
//! A projection `y = x W^T` (W `[n][k]`) is stored as
//! - `<name>.xq_q`  int8 `[n][k]`: symmetric integer codes, `-2^(b-1) <= q <= 2^(b-1) - 1`, b in {3, 4, 5, 6, 8} (per tensor),
//! - `<name>.xq_s`  fp16 `[n][k / g]`: one scale per output row and group of g = 128 inputs,
//! - `<name>.xq_su` fp16 `[k]`: the sign / scale vector of the input-side Hadamard-128 rotation (absent when `rotate` is "none"),
//!
//! and `xq_config.json` gives bits / group / rotate / shape per tensor. The effective weight is `W[n][k] = s[n][k / g] * q[n][k]`
//! applied to the rotated input `x~ = fp16(FWHT128(su * x) / sqrt(128))` (128-blocks along k).
//!
//! - [`st`]: memory-mapped safetensors index of a checkpoint directory.
//! - [`format`]: `xq_config.json` + typed access to a checkpoint's XQ / bf16 tensors.
//! - [`pack`]: the B70 device layout (16-row blocks, code planes shaped for DPAS B fragments), byte-identical to `xq.zig pack`.
//! - [`cpu`]: CPU reference: the rotation prepass bit for bit (`xq_prep`), dequant, float64 GEMV, and an fp32 model of the decode matvec.
//! - [`fixture`]: reader for arc-engine's `tools/make_xq_fixtures.py` oracle files.
#[path = "../../fe-exl3/src/half.rs"]
pub mod half;

pub mod cpu;
pub mod fixture;
pub mod format;
pub mod pack;
pub mod st;

pub use format::{Checkpoint, Rotate, XqConfig, XqMeta, XqTensor};
