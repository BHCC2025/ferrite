//! Per-sequence device state of the B70 forward and its sizes (arc-engine layouts, so oracle dumps compare directly).
//!
//! - KV cache, per full-attention layer, K and V separately: `[pos][kv head][record]`.
//!   - q8 (serving default): a record is 8 blocks of 32 dims, each 32 int8 + one fp16 scale = 272 bytes a head and position
//!     (`qwen_attn_long.zig recBytes`, quantized on append by `qwen_kvq.cl`);
//!   - q4: 8 x (16 bytes + fp16) = 144 bytes; bf16: 512 bytes.
//! - GDN, per linear-attention layer: conv state bf16 `[3][10240]` (the last 3 rows of the conv input), delta-rule state fp32
//!   `[48 heads][128][128]` (3.1 MB; 151 MB for the 48 layers).
//! - Prefix cache (serving): `ckpt_slots` device copies of every GDN state (154 MB a slot), taken on a position grid / at the prompt tail /
//!   at the end of a generation; the KV cache needs none (it is trimmed by position).
//! - MTP block: its own KV cache (one attention layer, same record format as the target's quantized mode).
use crate::config::Config;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvMode {
    Bf16,
    Q8,
    Q4,
}

impl KvMode {
    /// Bytes of one kv head of one position in a K or V cache.
    pub fn rec_bytes(self) -> usize {
        match self {
            KvMode::Bf16 => 512,
            KvMode::Q8 => 8 * (32 + 2),
            KvMode::Q4 => 8 * (16 + 2),
        }
    }
    pub fn parse(s: &str) -> Option<KvMode> {
        match s {
            "bf16" => Some(KvMode::Bf16),
            "q8" => Some(KvMode::Q8),
            "q4" => Some(KvMode::Q4),
            _ => None,
        }
    }
}

/// Positions a long-context decode chunk covers and the most chunks (arc-engine qwen_attn_long: 1024 x 128 = 131072 keys).
pub const ATTN_CHUNK: usize = 1024;
pub const MAX_CTX: usize = 131072;

#[derive(Clone, Debug)]
pub struct Sizes {
    /// one K (or V) cache of one attention layer
    pub kv_layer: usize,
    /// all K + V caches of the target model
    pub kv_total: usize,
    /// conv + delta-rule state of one GDN layer
    pub gdn_layer: usize,
    /// all GDN layers (= one prefix-cache checkpoint slot)
    pub gdn_total: usize,
    /// the MTP block's K + V caches
    pub mtp_kv: usize,
}

pub fn sizes(cfg: &Config, ctx: usize, kv: KvMode, mtp_kv: KvMode) -> Sizes {
    let kv_layer = ctx * cfg.n_kv * kv.rec_bytes();
    let n_attn = cfg.n_full();
    let n_gdn = cfg.n_layers - n_attn;
    let conv = (cfg.conv_k - 1) * cfg.conv_dim() * 2;
    let delta = cfg.lin_v_heads * cfg.lin_dk * cfg.lin_dv * 4;
    Sizes {
        kv_layer,
        kv_total: 2 * n_attn * kv_layer,
        gdn_layer: conv + delta,
        gdn_total: n_gdn * (conv + delta),
        mtp_kv: 2 * ctx * cfg.n_kv * mtp_kv.rec_bytes(),
    }
}
