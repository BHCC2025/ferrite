//! Qwen3.8-Flash-Next (llama.cpp arch `qwen4exp`) hyper-parameters, read from the GGUF header.
use fe_gguf::Gguf;
use anyhow::{anyhow, bail, Result};

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize,     // trunk blocks (48)
    pub n_layer_all: usize, // + MTP (49)
    pub n_embd: usize,      // 2560
    pub n_vocab: usize,
    pub eps: f32,
    pub hc: usize,      // hyper-connection streams (4)
    pub hc_lr: usize,   // low rank (320)
    // attention
    pub n_head: usize,    // 24
    pub n_head_kv: usize, // 2
    pub head_dim: usize,  // 256
    pub n_rot: usize,     // 64
    pub rope_base: f32,
    pub rope_sections: [u32; 4],
    // QSA indexer
    pub idx_heads: usize, // 4
    pub idx_dim: usize,   // 128
    pub idx_top_k: usize, // 2048
    pub kpool: usize,     // 4
    pub compress: Vec<u32>,
    // GDN
    pub ssm_conv: usize, // 4
    pub ssm_inner: usize,
    pub ssm_state: usize, // 128 (head dim)
    pub ssm_kh: usize,    // 16
    pub ssm_vh: usize,    // 48
    pub full_attn_interval: usize,
    // MoE
    pub n_expert: usize, // 256
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub expert_weights_scale: f32,
    // PLE
    pub ple_layer: usize,
    pub ple_ngram: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_conv: usize,
    pub ple_eos: u32,
    pub ple_image_tok: u32,
    pub ple_dim: usize, // 160
    pub ple_mult: Vec<u64>,
    pub ple_off: Vec<u64>,
    pub ple_vocab: Vec<u64>,
    pub n_ctx_train: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Config> {
        let a = g.arch();
        if a != "qwen4exp" {
            bail!("expected a qwen4exp GGUF, got architecture `{a}`");
        }
        let k = |s: &str| format!("qwen4exp.{s}");
        let u = |s: &str| g.get_i64(&k(s)).map(|v| v as usize).ok_or_else(|| anyhow!("gguf key {} missing", k(s)));
        let f = |s: &str| g.get_f64(&k(s)).ok_or_else(|| anyhow!("gguf key {} missing", k(s)));
        let arr = |s: &str| g.get(&k(s)).and_then(|v| v.as_i64_vec()).map(|v| v.into_iter().map(|x| x as u64).collect::<Vec<u64>>())
            .ok_or_else(|| anyhow!("gguf key {} missing", k(s)));
        let n_layer_all = u("block_count")?;
        let nextn = u("nextn_predict_layers").unwrap_or(0);
        let secs = arr("rope.dimension_sections")?;
        let compress: Vec<u32> = arr("attention.compress_ratios")?.iter().map(|&v| v as u32).collect();
        let kpool = compress.iter().copied().find(|&r| r > 0).unwrap_or(0) as usize;
        let ple_layers = arr("ple.layers")?;
        if ple_layers.len() != 1 {
            bail!("one PLE layer supported, got {:?}", ple_layers);
        }
        let n_vocab = g.tensor("token_embd.weight").ok_or_else(|| anyhow!("no token_embd"))?.dims[1] as usize;
        let c = Config {
            n_layer: n_layer_all - nextn,
            n_layer_all,
            n_embd: u("embedding_length")?,
            n_vocab,
            eps: f("attention.layer_norm_rms_epsilon")? as f32,
            hc: u("hyper_connection.count")?,
            hc_lr: u("hyper_connection.low_rank")?,
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            rope_base: f("rope.freq_base")? as f32,
            rope_sections: [secs[0] as u32, secs[1] as u32, secs[2] as u32, secs[3] as u32],
            idx_heads: u("attention.indexer.head_count")?,
            idx_dim: u("attention.indexer.key_length")?,
            idx_top_k: u("attention.indexer.top_k")?,
            kpool,
            compress,
            ssm_conv: u("ssm.conv_kernel")?,
            ssm_inner: u("ssm.inner_size")?,
            ssm_state: u("ssm.state_size")?,
            ssm_kh: u("ssm.group_count")?,
            ssm_vh: u("ssm.time_step_rank")?,
            full_attn_interval: u("full_attention_interval").unwrap_or(4),
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            n_ff_shexp: u("expert_shared_feed_forward_length")?,
            expert_weights_scale: f("expert_weights_scale").unwrap_or(0.0) as f32,
            ple_layer: ple_layers[0] as usize,
            ple_ngram: u("ple.ngram_size")?,
            ple_heads_per_ngram: u("ple.heads_per_ngram")?,
            ple_conv: u("ple.conv_kernel")?,
            ple_eos: u("ple.eos_token_id")? as u32,
            ple_image_tok: u("ple.image_token_id").unwrap_or(0) as u32,
            ple_dim: u("embedding_length_per_layer_input")?,
            ple_mult: arr("ple.layer_multipliers")?,
            ple_off: arr("ple.head_offsets")?,
            ple_vocab: arr("ple.head_vocab_sizes")?,
            n_ctx_train: u("context_length")?,
        };
        if c.ssm_vh * c.ssm_state != c.ssm_inner || c.hc * c.n_embd != g.tensor("blk.0.hc_attn_norm.weight").map_or(0, |t| t.n_elements() as usize) {
            bail!("inconsistent GDN / HC sizes");
        }
        if c.ple_heads() * c.ple_dim != c.n_embd {
            bail!("PLE heads x dim must equal n_embd");
        }
        Ok(c)
    }
    /// Full attention (QSA) layer? Every `full_attn_interval`-th trunk layer, and the MTP block.
    pub fn is_attn(&self, il: usize) -> bool {
        il >= self.n_layer || (il + 1) % self.full_attn_interval == 0
    }
    pub fn ple_heads(&self) -> usize {
        (self.ple_ngram - 1) * self.ple_heads_per_ngram
    }
    pub fn hc_dim(&self) -> usize {
        self.hc * self.n_embd
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.ssm_kh * self.ssm_state + self.ssm_vh * self.ssm_state
    }
    pub fn n_attn_layers(&self) -> usize {
        (0..self.n_layer).filter(|&i| self.is_attn(i)).count()
    }
}
