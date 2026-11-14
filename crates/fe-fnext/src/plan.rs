//! Where every byte lives on a 16 GB card: always-used weights, the KV / indexer caches, recurrent state,
//! workspace, then experts (a static hot set + dynamic cache slots) in what is left. Host side holds only the mmap.
use crate::config::Config;
use fe_gguf::Gguf;
use anyhow::Result;

pub const MIB: f64 = 1048576.0;

#[derive(Clone, Debug)]
pub struct Plan {
    pub ctx: usize,
    /// card bytes (total) and the reserve kept free (context, cubins, fragmentation)
    pub card: usize,
    pub reserve: usize,
    /// (tensor name, bytes) of the weights read by every token (as stored in the GGUF)
    pub always: Vec<(String, usize)>,
    pub always_bytes: usize,
    pub mtp_bytes: usize,
    pub kv_bytes: usize,
    pub idx_bytes: usize,
    pub state_bytes: usize,
    pub work_bytes: usize,
    /// bytes of one expert (gate + up + down) per layer (trunk + MTP)
    pub expert_bytes: Vec<usize>,
    pub experts_total: usize,
    /// bytes left for experts after everything above
    pub expert_budget: usize,
    /// static hot-set share of the expert budget
    pub static_bytes: usize,
    pub dynamic_slot_bytes: usize,
    pub dynamic_slots: usize,
}

fn is_expert(n: &str) -> bool {
    n.contains("_exps.")
}

impl Plan {
    /// `card`: device bytes; `ctx`: context cells; `chunk`: prompt rows per forward (workspace sizing);
    /// `static_frac`: share of the expert budget given to the static hot set (rest: dynamic slots);
    /// `with_mtp`: keep the MTP block's non-expert weights resident.
    pub fn new(c: &Config, g: &Gguf, card: usize, ctx: usize, chunk: usize, static_frac: f64, with_mtp: bool) -> Result<Plan> {
        let mut always = vec![];
        let mut mtp_bytes = 0;
        for t in &g.tensors {
            let n = &t.name;
            if is_expert(n) || n == "token_embd.weight" || n == "per_layer_token_embd.weight" {
                continue; // experts: below; embeddings: host rows
            }
            let mtp = n.starts_with(&format!("blk.{}.", c.n_layer));
            if mtp {
                mtp_bytes += t.bytes as usize;
                continue;
            }
            always.push((n.clone(), t.bytes as usize));
        }
        let always_bytes: usize = always.iter().map(|a| a.1).sum();
        let n_attn = c.n_attn_layers() + with_mtp as usize;
        let n_gdn = c.n_layer - c.n_attn_layers();
        // q8_0 K and V: 34 bytes per 32 values
        let kv_bytes = n_attn * ctx * c.n_head_kv * c.head_dim * 2 * 34 / 32;
        // pooled indexer keys (f32, one per block of kpool cells) + raw keys of the open tail block
        let idx_bytes = n_attn * (ctx / c.kpool.max(1) + 1) * c.idx_dim * 4 + n_attn * c.kpool * c.idx_dim * 4;
        let state_bytes = n_gdn * (c.ssm_vh * c.ssm_state * c.ssm_state * 4 + (c.ssm_conv - 1) * c.conv_dim() * 4)
            + (c.ple_conv - 1) * c.ple_ngram * c.hc_dim() * 4;
        // per prompt row: residual streams, mixer temporaries, GDN projections, attention q, expert activations (f32)
        let per_row = 4 * (3 * c.hc_dim() + c.conv_dim() + c.ssm_inner + 2 * c.n_head * c.head_dim + 8 * c.n_embd
            + c.n_expert_used * c.n_ff_exp * 3 + c.n_expert);
        // logits of one row (f32) and attention scores of a chunk over the context
        let work_bytes = chunk * per_row + c.n_vocab * 4 * 2 + chunk.min(256) * c.n_head * ctx.min(4096) * 4 + (64 << 20);
        let mut expert_bytes = vec![];
        for il in 0..c.n_layer_all {
            let mut b = 0;
            for p in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                let n = format!("blk.{il}.{p}.weight");
                b += g.tensor(&n).ok_or_else(|| anyhow::anyhow!("no {n}"))?.slice_bytes();
            }
            expert_bytes.push(b);
        }
        let experts_total: usize = (0..c.n_layer).map(|il| expert_bytes[il] * c.n_expert).sum();
        let reserve = 400 << 20;
        let used = always_bytes + if with_mtp { mtp_bytes } else { 0 } + kv_bytes + idx_bytes + state_bytes + work_bytes + reserve;
        let expert_budget = card.saturating_sub(used);
        let static_bytes = (expert_budget as f64 * static_frac) as usize;
        let dynamic_slot_bytes = *expert_bytes[..c.n_layer].iter().max().unwrap();
        let dynamic_slots = (expert_budget - static_bytes) / dynamic_slot_bytes;
        Ok(Plan { ctx, card, reserve, always, always_bytes, mtp_bytes, kv_bytes, idx_bytes, state_bytes, work_bytes,
            expert_bytes, experts_total, expert_budget, static_bytes, dynamic_slot_bytes, dynamic_slots })
    }

    pub fn print(&self, c: &Config) {
        let m = |b: usize| b as f64 / MIB;
        println!("VRAM plan: card {:.0} MiB, context {} cells", m(self.card), self.ctx);
        println!("  always-used weights   {:>9.1} MiB  ({} tensors, GGUF types kept)", m(self.always_bytes), self.always.len());
        println!("  MTP block (non-expert){:>9.1} MiB", m(self.mtp_bytes));
        println!("  KV cache (q8_0)       {:>9.1} MiB  ({} attention layers)", m(self.kv_bytes), c.n_attn_layers());
        println!("  indexer pooled keys   {:>9.1} MiB", m(self.idx_bytes));
        println!("  GDN / conv / PLE state{:>9.1} MiB", m(self.state_bytes));
        println!("  workspace             {:>9.1} MiB", m(self.work_bytes));
        println!("  reserve               {:>9.1} MiB", m(self.reserve));
        println!("  -> experts            {:>9.1} MiB of {:.1} MiB total ({:.1}% resident)", m(self.expert_budget),
            m(self.experts_total), 100.0 * self.expert_budget as f64 / self.experts_total as f64);
        let avg = self.experts_total / (c.n_layer * c.n_expert);
        println!("     static hot set     {:>9.1} MiB  (~{} experts at the {:.2} MiB average)", m(self.static_bytes),
            self.static_bytes / avg, m(avg));
        println!("     dynamic cache      {:>9.1} MiB  ({} slots of {:.2} MiB)",
            m(self.dynamic_slots * self.dynamic_slot_bytes), self.dynamic_slots, m(self.dynamic_slot_bytes));
        let mut groups: std::collections::BTreeMap<String, usize> = Default::default();
        for (n, b) in &self.always {
            let key = n.split('.').skip(if n.starts_with("blk.") { 2 } else { 0 }).collect::<Vec<_>>().join(".");
            *groups.entry(key).or_default() += b;
        }
        println!("  always-used by tensor:");
        let mut v: Vec<_> = groups.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        for (k, b) in v.iter().take(14) {
            println!("     {:<28} {:>8.1} MiB", k, m(*b));
        }
        println!("  host: the GGUF mmap only (token rows and PLE rows read on demand; resident experts never touched)");
    }
}
