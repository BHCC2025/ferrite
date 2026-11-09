//! Head counts the attention / GDN kernels are built for. kernels/k-core-tp2 compiles the same sources with
//! its own dims.rs (one rank of a two-way head split).
pub const NQ: u32 = 24;
pub const NKV: u32 = 4;
pub const NKH: u32 = 16;
pub const NVH: u32 = 48;
