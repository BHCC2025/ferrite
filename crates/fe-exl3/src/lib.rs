//! fe-exl3: EXL3 (exllamav3 trellis) tensors for ferrite — format, CPU reference, GPU launch wrappers.

pub mod cpu;
pub mod format;
pub mod gpu;
pub mod half;
pub mod linear;
pub mod plans;
pub mod st;
pub mod tp;

pub use format::{exl3_bases, Exl3Tensor};

pub use anyhow::{Error, Result};
