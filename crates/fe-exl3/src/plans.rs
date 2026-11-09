//! Tuned decode-GEMV plans for GB10 (48 SMs), from `exl3-tune` on the Qwen3.8-27B EXL3 3.0 mix
//! (L2-cold weights, PDL launches, L1-allocating trellis loads, rows 1 + 0.25 x rows 16, median of 5 runs for the 8 best; 2026-11-03).
//! Keyed by (k, n, bits); other shapes use `linear::plan`.
//! Comments: measured us at rows 1 / 16. The v1 winners were moved to v9 (gemv8 + two-step prefetch, equal at
//! rows <= 16 within noise) so that rows 17..64 use the 32-row-pass kernel instead of 16-row passes.
use crate::linear::Plan;

pub const TUNED: &[((u32, u32, u32), Plan)] = &[
    ((5120, 1024, 3), Plan { sk: 10, wk: 8, ver: 8, occ: 2 }), // 13.0 / 15.6 us
    ((5120, 1024, 4), Plan { sk: 5, wk: 8, ver: 9, occ: 0 }), // 15.2 / 16.7 us
    ((5120, 6144, 2), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 39.2 / 42.6 us
    ((5120, 6144, 3), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 54.8 / 57.2 us
    ((5120, 10240, 2), Plan { sk: 1, wk: 8, ver: 8, occ: 2 }), // 63.9 / 69.2 us
    ((5120, 10240, 3), Plan { sk: 1, wk: 8, ver: 8, occ: 2 }), // 91.2 / 94.7 us
    ((5120, 12288, 3), Plan { sk: 1, wk: 8, ver: 8, occ: 2 }), // 104.6 / 109.5 us
    ((5120, 12288, 4), Plan { sk: 1, wk: 8, ver: 8, occ: 2 }), // 131.9 / 150.4 us
    ((5120, 17408, 2), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 101.9 / 117.1 us
    ((5120, 17408, 3), Plan { sk: 1, wk: 4, ver: 9, occ: 0 }), // 154.1 / 160.9 us
    ((5120, 17408, 4), Plan { sk: 1, wk: 4, ver: 9, occ: 0 }), // 194.0 / 202.5 us
    ((5120, 248320, 5), Plan { sk: 1, wk: 8, ver: 9, occ: 0 }), // 3397.9 / 3474.8 us
    ((6144, 5120, 3), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 55.2 / 57.5 us
    ((6144, 5120, 4), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 68.5 / 72.1 us
    ((10240, 5120, 4), Plan { sk: 10, wk: 4, ver: 9, occ: 0 }), // 112.1 / 126.7 us
    ((17408, 5120, 2), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 103.1 / 114.9 us
    ((17408, 5120, 3), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 151.1 / 162.7 us
    ((17408, 5120, 4), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 188.1 / 210.4 us
    // TP2 slices (exl3-tune with FE_TUNE_TP2=1): halves of n, or of k for o_proj / out_proj / down_proj
    ((3072, 5120, 3), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 29.1 / 31.4 us
    ((3072, 5120, 4), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 34.6 / 37.5 us
    ((5120, 512, 3), Plan { sk: 10, wk: 8, ver: 8, occ: 2 }), // 7.5 / 9.7 us
    ((5120, 512, 4), Plan { sk: 10, wk: 8, ver: 8, occ: 2 }), // 9.0 / 11.4 us
    ((5120, 3072, 2), Plan { sk: 4, wk: 8, ver: 8, occ: 2 }), // 21.9 / 25.5 us
    ((5120, 3072, 3), Plan { sk: 4, wk: 8, ver: 8, occ: 2 }), // 29.2 / 31.6 us
    ((5120, 5120, 2), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 34.7 / 37.7 us
    ((5120, 5120, 3), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 45.9 / 48.2 us
    ((5120, 5120, 4), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 58.5 / 62.2 us
    ((5120, 6144, 4), Plan { sk: 4, wk: 8, ver: 8, occ: 2 }), // 66.7 / 73.4 us
    ((5120, 8704, 2), Plan { sk: 5, wk: 4, ver: 8, occ: 4 }), // 54.5 / 67.8 us
    ((5120, 8704, 3), Plan { sk: 5, wk: 4, ver: 8, occ: 0 }), // 76.1 / 83.0 us
    ((5120, 8704, 4), Plan { sk: 5, wk: 4, ver: 8, occ: 0 }), // 95.4 / 109.4 us
    ((5120, 124160, 5), Plan { sk: 1, wk: 8, ver: 9, occ: 0 }), // 1701.5 / 1762.6 us
    ((8704, 5120, 2), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 54.8 / 59.0 us
    ((8704, 5120, 3), Plan { sk: 2, wk: 8, ver: 8, occ: 2 }), // 76.7 / 80.7 us
    ((8704, 5120, 4), Plan { sk: 4, wk: 8, ver: 8, occ: 2 }), // 93.4 / 103.8 us
];

pub fn tuned(k: u32, n: u32, bits: u32) -> Option<Plan> {
    TUNED.iter().find(|(key, _)| *key == (k, n, bits)).map(|(_, p)| *p)
}
