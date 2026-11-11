//! Qwen3.5-family model: config, safetensors weights, forward pass, KV/GDN state, sampling.
pub mod config;
pub mod copy;
pub mod follow;
pub mod lowmem;
pub mod model;
pub mod pcache;
pub mod tp;
pub mod safetensors;
pub mod sched;
pub mod selftest;

/// k-core kernels (kernels/k-core), assembled for sm_121a by build.rs.
pub static K_CORE_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_core.cubin"));
/// The head-dependent k-core kernels built for one rank of a two-way split (kernels/k-core-tp2).
pub static K_CORE_TP2_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_core_tp2.cubin"));
