//! Placement arithmetic for 2 x 16 GB cards + host RAM (M1: numbers only, nothing is allocated).
//! TP2: attention heads / shared expert / head vocabulary split, everything small replicated, routed experts
//! expert-parallel (expert e is owned by card e % 2, whole, so a CPU miss keeps its bits). KV is replicated (MQA).
use crate::config::DscConfig;
use fe_gguf::Gguf;

pub const MIB: f64 = 1048576.0;

#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub split_bytes: usize,      // tensors cut in two between the cards (counted once)
    pub repl_bytes: usize,       // tensors both cards hold in full
    pub embd_bytes: usize,       // token_embd (host rows, never on a card)
    pub engram_embd_bytes: usize, // Engram tables (mmapped from NVMe)
    pub engram_kv_bytes: usize,
    pub expert_bytes_layer: Vec<usize>, // one expert, gate+up+down, per layer
    pub experts_total: usize,
    pub per_card_dense: usize,
    pub kv_per_token: usize, // per lane, per card (replicated)
    pub window_per_lane: usize,
    pub work_bytes: usize,
    pub reserve: usize,
    pub card_expert_budget: usize,
    pub slots_per_card: usize,
    pub slot_bytes: usize,
    pub host_experts: usize, // bytes of expert weights the host holds (all of them, mmap)
}

fn is_split(n: &str) -> bool {
    ["attn_q_b.weight", "attn_output_a.weight", "attn_output_b.weight", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
        "output.weight", "engram_kv"].iter().any(|s| n.ends_with(s) || n.contains(s))
        && !n.contains("indexer.")
}

impl Plan {
    /// `card`: device bytes; `ctx`: context cells per lane; `lanes`; `chunk`: prompt rows per forward.
    pub fn new(c: &DscConfig, g: &Gguf, card: usize, ctx: usize, lanes: usize, chunk: usize) -> Plan {
        let mut p = Plan::default();
        for t in &g.tensors {
            let n = &t.name;
            let b = t.bytes as usize;
            if n.contains("engram_embd") {
                p.engram_embd_bytes += b;
            } else if n.contains("_exps.") {
                p.experts_total += b;
            } else if n == "token_embd.weight" {
                p.embd_bytes += b;
            } else if is_split(n) {
                p.split_bytes += b;
                if n.contains("engram_kv") {
                    p.engram_kv_bytes += b;
                }
            } else {
                p.repl_bytes += b;
            }
        }
        for l in 0..c.n_layer {
            let mut b = 0;
            for k in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                if let Some(t) = g.tensor(&format!("blk.{l}.{k}.weight")) {
                    b += t.slice_bytes();
                }
            }
            p.expert_bytes_layer.push(b);
        }
        p.host_experts = p.experts_total;
        p.per_card_dense = p.split_bytes / 2 + p.repl_bytes;
        // DeepSeek formats (PLAN-DS 1.3): compressed main plane 288 B an entry (ratio 1: layer 20; ratio 2: 3 layers at
        // half the entries), indexer keys 68 B an entry in the same layers, replicated on both cards.
        let ratio2 = c.kv_sources.iter().filter(|&&l| c.compress[l] == 2).count();
        let ratio1 = c.kv_sources.iter().filter(|&&l| c.compress[l] == 1).count();
        p.kv_per_token = ratio1 * (288 + 68) + ratio2 * (144 + 34);
        p.window_per_lane = c.n_layer * (c.window + 16) * c.head_dim * 2;
        let _ = ctx;
        // per prompt row (f32): 3 residual copies, q, latent kv, attention out, expert activations, logits excluded
        let per_row = 4 * (3 * c.hc_dim() + c.q_dim() + 2 * c.n_embd + c.n_expert_used * c.n_ff_exp * 3 + c.n_expert);
        p.work_bytes = chunk * per_row + c.n_vocab * 4 * 2 * lanes + (128 << 20);
        p.reserve = 600 << 20;
        let used = p.per_card_dense + lanes * (ctx * p.kv_per_token + p.window_per_lane) + p.work_bytes + p.reserve;
        p.card_expert_budget = card.saturating_sub(used);
        p.slot_bytes = *p.expert_bytes_layer.iter().max().unwrap_or(&1);
        p.slots_per_card = p.card_expert_budget / p.slot_bytes;
        p
    }

    pub fn print(&self, c: &DscConfig, card: usize, ctx: usize, lanes: usize) {
        let m = |b: usize| b as f64 / MIB;
        let g = |b: usize| b as f64 / 1e9;
        println!("placement (2 cards of {:.0} MiB, {lanes} lanes x {ctx} cells):", m(card));
        println!("  dense, cut in two        {:>9.1} MiB total  -> {:.1} MiB a card", m(self.split_bytes), m(self.split_bytes / 2));
        println!("  dense, replicated        {:>9.1} MiB a card", m(self.repl_bytes));
        println!("  dense per card           {:>9.1} MiB", m(self.per_card_dense));
        println!("  KV planes + indexer keys {:>9.1} MiB a card ({} B a token a lane)", m(lanes * ctx * self.kv_per_token), self.kv_per_token);
        println!("  window rings             {:>9.1} MiB a card ({:.1} MiB a lane)", m(lanes * self.window_per_lane), m(self.window_per_lane));
        println!("  workspace + reserve      {:>9.1} MiB", m(self.work_bytes + self.reserve));
        println!("  -> expert budget a card  {:>9.1} MiB = {} slots of {:.2} MiB", m(self.card_expert_budget), self.slots_per_card, m(self.slot_bytes));
        let total_experts = c.n_layer * c.n_expert;
        println!("     both cards hold {} of {} experts ({:.1}%)", 2 * self.slots_per_card, total_experts,
            100.0 * (2 * self.slots_per_card) as f64 / total_experts as f64);
        println!("  host: experts {:.1} GB (all, mmapped; NUMA-split by e % 2), token_embd {:.2} GB, engram tables {:.1} GB (NVMe, 4 KiB rows on demand), engram_kv {:.2} GB",
            g(self.host_experts), g(self.embd_bytes), g(self.engram_embd_bytes), g(self.engram_kv_bytes));
    }
}
