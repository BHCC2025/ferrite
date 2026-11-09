//! One rank of two: 12 q / 2 kv attention heads, 8 key / 24 value GDN heads.
pub const NQ: u32 = 12;
pub const NKV: u32 = 2;
pub const NKH: u32 = 8;
pub const NVH: u32 = 24;
