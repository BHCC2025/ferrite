//! The forward of Qwen3.8-27B on the B70 (skeleton: signatures and the launch sequence; bodies land with the kernels).
//!
//! One token (decode, `Model::forward` in arc-engine), residual stream bf16 `x[5120]`, block outputs added lazily (`pending`: the next
//! norm folds `x = bf16(x + delta)` into its launch):
//!
//! ```text
//! embed_bf16(tok) -> x
//! per layer i:
//!   GDN (i % 4 != 3):  xq_add_rms_prep(x, delta, input_norm) -> xn + prepass of in_proj_qkv's su
//!                      xq_mv qkv -> qkv[10240], xq_mv z -> z[6144], xq_mv b -> b[48], xq_mv a -> a[48]   (one prepass shared: same x, same su)
//!                      gdn_conv_prep(qkv, cstate, conv) -> qn, kn ; gdn_step(qn, kn, qkv, cstate, conv, b, a, A_log, dt_bias, sstate) -> y[6144]
//!                      gdn_gate_norm(y, z, gnorm) -> yg ; xq_mv out(yg) -> delta ; pending
//!   attention:         xq_add_rms_prep(x, delta, input_norm) -> xn ; xq_mv q -> qg[12288] (q | gate), k -> kraw[1024], v -> vtmp[1024]
//!                      attn_prep(qg, kraw, q_norm, k_norm, rope cos/sin) -> q[6144], ktmp ; kvq append(ktmp, vtmp) at pos
//!                      decode attention over pos + 1 keys (q8 cache; ARC_DEC kernel) gated by sigmoid(gate) -> att[6144]
//!                      xq_mv o(att) -> delta ; pending
//!   MLP:               xq_add_rms_prep(x, delta, post_norm) -> xn ; xq_mv gate, up -> [17408] each
//!                      xq_prep_swiglu(gate, up) -> act (bf16(silu(g) u)) + down's prepass ; xq_mv down -> delta ; pending
//! head:                xq_add_rms_prep(x, delta, final_norm) ; xq_mv lm_head -> logits fp32 [248320] ; (round_bf16_f32) ; argmax_partial/final
//! ```
//!
//! Windows of 1..16 rows run the same launches with m rows (XQ variants by m; GDN advances row by row inside the kernel; causal attention
//! inside the window), bitwise equal per row to the one-token path (row invariance: required for lossless speculation). Prompt windows
//! (> 16 rows, 4096 by default) use the DPAS prompt GEMM, prompt attention and the chunked GDN kernels. Details: DESIGN.md.
use anyhow::{bail, Result};

/// What a forward call computes logits for.
#[derive(Clone, Copy, Debug)]
pub enum Head {
    /// no logits (prompt rows whose output is not needed)
    None,
    /// the last `n` rows
    Last(u32),
    /// keep the normalised hidden rows for head batches (scoring)
    KeepHidden,
}

/// The engine of one sequence (single-stream like arc-engine; the side pool is a second instance of the per-sequence state).
pub struct Model {
    pub pos: u32,
}

impl Model {
    /// One token through all layers at `pos` (and the head when asked); advances the position.
    pub fn forward(&mut self, _token: u32, _logits: bool) -> Result<()> {
        bail!("fe-xe forward: not implemented yet (waits for the X1/X2 kernels)")
    }

    /// 1..16 rows as one window at the current position. `inplace`: commit all rows; else `commit(c)` keeps the first c.
    pub fn forward_rows(&mut self, _tokens: &[u32], _head: Head, _inplace: bool) -> Result<()> {
        bail!("fe-xe forward_rows: not implemented yet")
    }

    /// Prompt window (> 16 rows): prefill GEMMs, prompt attention, chunked GDN; in place.
    pub fn prefill(&mut self, _tokens: &[u32], _head: Head) -> Result<()> {
        bail!("fe-xe prefill: not implemented yet")
    }

    /// Keep the first `c` rows of the last non-inplace window (GDN state swap / replay, KV trim by position).
    pub fn commit(&mut self, _c: u32) -> Result<()> {
        bail!("fe-xe commit: not implemented yet")
    }

    /// fp32 logits of the head rows of the last call, [rows][vocab].
    pub fn fetch_logits(&mut self, _out: &mut [f32]) -> Result<()> {
        bail!("fe-xe fetch_logits: not implemented yet")
    }
}
