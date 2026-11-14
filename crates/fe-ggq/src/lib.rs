//! ggml quant formats for Ferrite (GGUF models): host reference dequantization, GPU kernels (k-gguf: dequant,
//! decode GEMV, grouped-expert GEMV, prompt GEMM) and AVX2 CPU expert kernels.
pub mod cpu;
pub mod gpu;
pub mod pdl;
pub mod refq;
pub mod tables {
    #![allow(dead_code)]
    include!("../../../kernels/k-gguf/src/tables.rs");
}
pub use fe_gguf::GgmlType;
pub use gpu::{AttnWs, Ggq, Out, Q8Act, SortWs, SumMode};
