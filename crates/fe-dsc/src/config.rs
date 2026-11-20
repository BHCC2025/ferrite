//! DeepSeek-V4.1-Flash (llama.cpp-style arch `deepseek41`) hyper-parameters, read from the GGUF header, and the
//! per-layer roles (window only / compressor + indexer source / reuse / reindex) derived from them.
use anyhow::{anyhow, bail, Result};
use fe_gguf::Gguf;

/// What a layer's attention reads beyond its sliding window (DeepSeek's Full / Reuse / Reindex).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerRole {
    /// `compress_ratios[l]`: 0 = window only, 2 = a compressed plane of one entry per 2 tokens, 1 = per token.
    pub ratio: u32,
    /// builds its own compressed plane (and, with `attn_k`/`k_norm`, indexer keys)
    pub kv_source: bool,
    /// owns an indexer query path (`indexer.attn_q_b` + `indexer.proj`)
    pub index_source: bool,
    /// the kv-source layer whose plane this layer reads (ratio > 0)
    pub kv_from: Option<usize>,
    /// the index-source layer whose top-k selection this layer uses (ratio > 0)
    pub sel_from: Option<usize>,
    /// the candidate source itself (builds the 2048-block pool)
    pub cand_source: bool,
    /// searches only inside the candidate pool (index source after the candidate source)
    pub cand_masked: bool,
    /// runs Engram (the n-gram memory) before its attention mixes
    pub engram: bool,
}

#[derive(Clone, Debug)]
pub struct DscConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_ctx_train: usize,
    pub eps: f32,
    // attention (MQA latent: one 512-wide KV head, key = value)
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub q_lora: usize,
    pub o_lora: usize,
    pub o_groups: usize,
    pub window: usize,
    // rope: plain on ratio-0 layers, YaRN (and the compress theta) on compressed layers
    pub rope_base: f32,
    pub rope_compress_base: f32,
    pub yarn_factor: f32,
    pub yarn_orig_ctx: usize,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    // indexer
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_top_k: usize,
    pub cand_layer: usize,
    pub cand_blocks: usize,
    pub cand_block: usize,
    pub kv_sources: Vec<usize>,
    pub index_sources: Vec<usize>,
    pub compress: Vec<u32>,
    // hyper-connections
    pub hc: usize,
    pub hc_iters: usize,
    pub hc_eps: f32,
    // MoE
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_expert_shared: usize,
    pub n_ff_exp: usize,
    pub gating_func: i64,
    pub weights_scale: f32,
    pub weights_norm: bool,
    pub hash_layers: usize,
    pub clamp_exp: Vec<f32>,
    pub clamp_shexp: Vec<f32>,
    // Engram / PLE
    pub engram_layers: Vec<usize>,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub ple_dim: usize,
    pub ple_mult: Vec<u64>,
    pub head_off: Vec<u64>,
    pub head_vocab: Vec<u64>,
    pub engram_pad: u32,
    pub token_map: Vec<u32>,
}

impl DscConfig {
    pub fn from_gguf(g: &Gguf) -> Result<DscConfig> {
        let a = g.arch();
        if a != "deepseek41" {
            bail!("expected a deepseek41 GGUF, got architecture `{a}`");
        }
        let k = |s: &str| format!("deepseek41.{s}");
        let u = |s: &str| g.get_i64(&k(s)).map(|v| v as usize).ok_or_else(|| anyhow!("gguf key {} missing", k(s)));
        let f = |s: &str| g.get_f64(&k(s)).ok_or_else(|| anyhow!("gguf key {} missing", k(s)));
        let ints = |s: &str| g.get(&k(s)).and_then(|v| v.as_i64_vec()).ok_or_else(|| anyhow!("gguf key {} missing", k(s)));
        let floats = |s: &str| g.get(&k(s)).and_then(|v| v.as_f32_vec()).ok_or_else(|| anyhow!("gguf key {} missing", k(s)));
        let uv = |s: &str| ints(s).map(|v| v.into_iter().map(|x| x as u64).collect::<Vec<u64>>());
        let us = |s: &str| ints(s).map(|v| v.into_iter().map(|x| x as usize).collect::<Vec<usize>>());
        let n_layer = u("block_count")?;
        let n_vocab = g.tensor("token_embd.weight").map(|t| t.dims[1] as usize)
            .or_else(|| g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).map(|a| a.len()))
            .ok_or_else(|| anyhow!("no vocabulary size"))?;
        let token_map: Vec<u32> = ints("engram.token_map")?.into_iter().map(|x| x as u32).collect();
        let c = DscConfig {
            n_layer,
            n_embd: u("embedding_length")?,
            n_vocab,
            n_ctx_train: u("context_length")?,
            eps: f("attention.layer_norm_rms_epsilon")? as f32,
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            q_lora: u("attention.q_lora_rank")?,
            o_lora: u("attention.output_lora_rank")?,
            o_groups: u("attention.output_group_count")?,
            window: u("attention.sliding_window")?,
            rope_base: f("rope.freq_base")? as f32,
            rope_compress_base: f("attention.compress_rope_freq_base")? as f32,
            yarn_factor: f("rope.scaling.factor")? as f32,
            yarn_orig_ctx: u("rope.scaling.original_context_length")?,
            yarn_beta_fast: f("rope.scaling.yarn_beta_fast")? as f32,
            yarn_beta_slow: f("rope.scaling.yarn_beta_slow")? as f32,
            idx_heads: u("attention.indexer.head_count")?,
            idx_dim: u("attention.indexer.key_length")?,
            idx_top_k: u("attention.indexer.top_k")?,
            cand_layer: u("attention.indexer.candidate_source_layer")?,
            cand_blocks: u("attention.indexer.candidate_top_k_blocks")?,
            cand_block: u("attention.indexer.candidate_block_size")?,
            kv_sources: us("attention.kv_source_layers")?,
            index_sources: us("attention.indexer.source_layers")?,
            compress: ints("attention.compress_ratios")?.into_iter().map(|x| x as u32).collect(),
            hc: u("hyper_connection.count")?,
            hc_iters: u("hyper_connection.sinkhorn_iterations")?,
            hc_eps: f("hyper_connection.epsilon")? as f32,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_expert_shared: u("expert_shared_count")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            gating_func: g.get_i64(&k("expert_gating_func")).ok_or_else(|| anyhow!("expert_gating_func missing"))?,
            weights_scale: f("expert_weights_scale")? as f32,
            weights_norm: matches!(g.get(&k("expert_weights_norm")), Some(fe_gguf::Value::Bool(true))),
            hash_layers: u("hash_layer_count")?,
            clamp_exp: floats("swiglu_clamp_exp")?,
            clamp_shexp: floats("swiglu_clamp_shexp")?,
            engram_layers: us("ple.layers")?,
            ngram: u("ple.ngram_size")?,
            heads_per_ngram: u("ple.heads_per_ngram")?,
            ple_dim: u("embedding_length_per_layer_input")?,
            ple_mult: uv("ple.layer_multipliers")?,
            head_off: uv("ple.head_offsets")?,
            head_vocab: uv("ple.head_vocab_sizes")?,
            engram_pad: u("engram.pad_token_id")? as u32,
            token_map,
        };
        c.validate()?;
        Ok(c)
    }

    fn validate(&self) -> Result<()> {
        let c = self;
        if c.compress.len() < c.n_layer {
            bail!("compress_ratios has {} entries for {} layers", c.compress.len(), c.n_layer);
        }
        if c.clamp_exp.len() != c.n_layer || c.clamp_shexp.len() != c.n_layer {
            bail!("swiglu clamps: {} / {} entries for {} layers", c.clamp_exp.len(), c.clamp_shexp.len(), c.n_layer);
        }
        if c.n_head % c.o_groups != 0 || c.n_head_kv != 1 {
            bail!("attention: {} heads, {} output groups, {} kv heads", c.n_head, c.o_groups, c.n_head_kv);
        }
        if c.head_dim < c.n_rot {
            bail!("rope dims {} exceed head dim {}", c.n_rot, c.head_dim);
        }
        let cols = (c.ngram - 1) * c.heads_per_ngram;
        if c.head_off.len() != c.engram_layers.len() * cols || c.head_vocab.len() != c.head_off.len() {
            bail!("PLE: {} head offsets / {} vocab sizes for {} layers x {cols} columns", c.head_off.len(),
                c.head_vocab.len(), c.engram_layers.len());
        }
        if c.token_map.len() != c.n_vocab {
            bail!("engram.token_map has {} entries for a vocabulary of {}", c.token_map.len(), c.n_vocab);
        }
        if c.gating_func != 4 {
            bail!("expert_gating_func {} (only 4, sqrt-softplus, is implemented)", c.gating_func);
        }
        for &l in c.kv_sources.iter().chain(&c.index_sources).chain(&c.engram_layers) {
            if l >= c.n_layer {
                bail!("layer id {l} out of range");
            }
        }
        Ok(())
    }

    /// Engram columns (n-gram orders 2..=ngram, `heads_per_ngram` hashes each) of one layer.
    pub fn engram_cols(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }
    /// Width of one Engram lookup (all columns concatenated), the input rows of `engram_kv`.
    pub fn engram_width(&self) -> usize {
        self.engram_cols() * self.ple_dim
    }
    pub fn hc_dim(&self) -> usize {
        self.hc * self.n_embd
    }
    /// rows of `hc_*_fn`: pre (hc) + post (hc) + comb (hc x hc)
    pub fn hc_mix(&self) -> usize {
        2 * self.hc + self.hc * self.hc
    }
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    pub fn role(&self, l: usize) -> LayerRole {
        let ratio = self.compress[l];
        let engram = self.engram_layers.contains(&l);
        if ratio == 0 {
            return LayerRole { ratio, kv_source: false, index_source: false, kv_from: None, sel_from: None,
                cand_source: false, cand_masked: false, engram };
        }
        let index_source = self.index_sources.contains(&l);
        LayerRole {
            ratio,
            kv_source: self.kv_sources.contains(&l),
            index_source,
            kv_from: self.kv_sources.iter().copied().filter(|&s| s <= l).max(),
            sel_from: self.index_sources.iter().copied().filter(|&s| s <= l).max(),
            cand_source: self.cand_layer == l,
            cand_masked: index_source && self.cand_layer < l,
            engram,
        }
    }
}
