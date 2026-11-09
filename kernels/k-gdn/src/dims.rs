//! The whole model: 16 key / 48 value GDN heads of 128.
pub const CONV_DIM: u32 = 10240; // q 2048 | k 2048 | v 6144
pub const NKH: u32 = 16;
pub const NVH: u32 = 48;
