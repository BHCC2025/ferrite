//! One rank of two: 8 key / 24 value GDN heads of 128.
pub const CONV_DIM: u32 = 5120; // q 1024 | k 1024 | v 3072
pub const NKH: u32 = 8;
pub const NVH: u32 = 24;
