//! Chunked gated delta rule for GDN prompt rows: the same contract as k-core's `fe_gdn_rec2` (raw per-head
//! output o, fp32 state carried in `state`), but 32 rows at a time instead of one.
//!
//! Per head the recurrence (fe_gdn_rec2) is, row by row: S *= g; d = beta (v - S^T k); S += k d^T; o = S^T q,
//! with q, k l2-normalised (q also / sqrt(128)). Within a chunk of rows 0..C, with L_r = sum_{s<=r} ln g_s,
//! gamma_r = exp(L_r) and S0 the state entering the chunk, the per-row deltas u_r satisfy
//!   u_r + beta_r sum_{s<r} exp(L_r - L_s) (k_r . k_s) u_s = beta_r (v_r - gamma_r S0^T k_r)      (1)
//! (a unit lower-triangular solve, column by column of the value dim), and
//!   o_r  = gamma_r S0^T q_r + sum_{s<=r} exp(L_r - L_s) (q_r . k_s) u_s
//!   S_C  = gamma_{C-1} S0 + sum_s exp(L_{C-1} - L_s) k_s u_s^T.
//! Value columns are independent, so thread (j = t / 4, part = t % 4) keeps rows part*32.. of state column j
//! in registers (as fe_gdn_rec); T = (I + A)^-1 is formed once per chunk (one warp) and U = T R is a parallel
//! product; q / k / v rows, the Gram matrices and the decays of a chunk sit in shared memory. Everything is fp32 (no tensor
//! cores): the result differs from the row recurrence only by summation order.
#![no_std]
#![feature(abi_ptx, asm_experimental_arch)]
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

extern crate kcommon;

pub mod chunk;
pub mod dims;
