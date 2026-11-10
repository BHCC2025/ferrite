//! fe-xe: Qwen3.8-27B on the Intel Arc Pro B70 (Xe2), Rust host + Rust SPIR-V kernels, XQ weights. See DESIGN.md.
//!
//! Status: skeleton. Host-side pieces that need no device are implemented and tested (weights preparation, evaluation outputs, oracle
//! readers, state sizing); the forward itself lands kernel by kernel as kernels-xe provides them (fe-ze runs them).
#[path = "../../fe-model/src/config.rs"]
pub mod config;

pub mod engine;
pub mod eval;
pub mod forward;
pub mod kernels;
pub mod kx;
pub mod mtp;
pub mod oracle;
pub mod rope;
pub mod spec;
pub mod specgen;
pub mod state;
pub mod weights;

// for engine hosts (fe-server): the checkpoint reader and device buffers without their own dependencies
pub use fe_xq;
pub use fe_ze;
