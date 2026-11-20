//! The tensor map of a deepseek41 GGUF: every tensor name the forward needs, with its allowed types and exact shape,
//! checked against the files. Names are llama.cpp's; the comment on each is the DeepSeek reference (HF) name that
//! the 2-Spark port (crates/fe-ds, branch ds-model) read from EXL3 shards.
use crate::config::DscConfig;
use fe_gguf::{GgmlType as T, Gguf};

/// One expected tensor.
pub struct Spec {
    pub name: String,
    pub types: &'static [T],
    /// ggml order (`dims[0]` contiguous)
    pub dims: Vec<u64>,
    pub doc: &'static str,
}

const WQ: &[T] = &[T::Q8_0, T::Q5K, T::Q4K, T::Q6K, T::BF16, T::F16];
const FP: &[T] = &[T::F32];
const BF: &[T] = &[T::BF16, T::F16, T::F32];
const EX: &[T] = &[T::IQ1M, T::IQ2XXS, T::IQ2XS, T::IQ2S, T::IQ3XXS, T::Q2K];

fn s(name: String, types: &'static [T], dims: &[usize], doc: &'static str) -> Spec {
    Spec { name, types, dims: dims.iter().map(|&d| d as u64).collect(), doc }
}

/// Tensors outside the blocks.
pub fn global_specs(c: &DscConfig) -> Vec<Spec> {
    vec![
        s("token_embd.weight".into(), WQ, &[c.n_embd, c.n_vocab], "embed.weight (host rows)"),
        s("output.weight".into(), WQ, &[c.n_embd, c.n_vocab], "head.weight (vocabulary-split under TP2)"),
        s("output_norm.weight".into(), FP, &[c.n_embd], "norm.weight"),
    ]
}

/// Tensors of block `l` (required ones; roles decide which optional groups exist).
pub fn layer_specs(c: &DscConfig, l: usize) -> Vec<Spec> {
    let r = c.role(l);
    let p = |n: &str| format!("blk.{l}.{n}.weight");
    let pb = |n: &str| format!("blk.{l}.{n}.bias");
    let d = c.n_embd;
    let mut v = vec![
        s(p("attn_norm"), FP, &[d], "layers.L.attn_norm"),
        s(p("attn_q_a"), WQ, &[d, c.q_lora], "attn.wq_a"),
        s(p("attn_q_a_norm"), FP, &[c.q_lora], "attn.q_norm"),
        s(p("attn_q_b"), WQ, &[c.q_lora, c.q_dim()], "attn.wq_b (64 heads x 512; TP2 splits the columns)"),
        s(p("attn_kv"), WQ, &[d, c.head_dim], "attn.wkv (the shared 512-wide latent)"),
        s(p("attn_kv_a_norm"), FP, &[c.head_dim], "attn.kv_norm"),
        s(p("attn_sinks"), FP, &[c.n_head], "attn.attn_sink"),
        s(p("attn_output_a"), WQ, &[c.q_dim() / c.o_groups, c.o_groups * c.o_lora], "attn.wo_a (8 groups x 4096 -> 1024)"),
        s(p("attn_output_b"), WQ, &[c.o_groups * c.o_lora, d], "attn.wo_b"),
        s(p("ffn_norm"), FP, &[d], "layers.L.ffn_norm"),
        s(p("ffn_gate_inp"), BF, &[d, c.n_expert], "ffn.gate.weight (router)"),
        s(pb("exp_probs_b"), FP, &[c.n_expert], "ffn.gate.bias (selection bias)"),
        s(pb("exp_probs_b_vl"), FP, &[c.n_expert], "ffn.gate.bias_vl (image tokens only; unused for text)"),
        s(p("ffn_gate_exps"), EX, &[d, c.n_ff_exp, c.n_expert], "ffn.experts.E.w1"),
        s(p("ffn_up_exps"), EX, &[d, c.n_ff_exp, c.n_expert], "ffn.experts.E.w3"),
        s(p("ffn_down_exps"), EX, &[c.n_ff_exp, d, c.n_expert], "ffn.experts.E.w2"),
        s(p("ffn_gate_shexp"), WQ, &[d, c.n_ff_exp], "ffn.shared_experts.w1"),
        s(p("ffn_up_shexp"), WQ, &[d, c.n_ff_exp], "ffn.shared_experts.w3"),
        s(p("ffn_down_shexp"), WQ, &[c.n_ff_exp, d], "ffn.shared_experts.w2"),
        s(p("hc_attn_fn"), FP, &[c.hc_dim(), c.hc_mix()], "hc_attn_fn"),
        s(p("hc_attn_base"), FP, &[c.hc_mix()], "hc_attn_base"),
        s(p("hc_attn_scale"), FP, &[3], "hc_attn_scale"),
        s(p("hc_ffn_fn"), FP, &[c.hc_dim(), c.hc_mix()], "hc_ffn_fn"),
        s(p("hc_ffn_base"), FP, &[c.hc_mix()], "hc_ffn_base"),
        s(p("hc_ffn_scale"), FP, &[3], "hc_ffn_scale"),
    ];
    if r.kv_source {
        // ratio-2 planes weight the token pair by a learned gate; the ratio-1 plane (layer 20) has no gate
        v.push(s(p("attn_compressor_kv"), BF, &[d, c.head_dim], "attn.compressor.wkv"));
        v.push(s(p("attn_compressor_norm"), FP, &[c.head_dim], "attn.compressor.norm"));
        if r.ratio == 2 {
            v.push(s(p("attn_compressor_gate"), BF, &[d, c.head_dim], "attn.compressor.wgate (+ ape)"));
        }
        v.push(s(p("indexer.attn_k"), BF, &[c.head_dim, c.idx_dim], "indexer.wk (512 -> 128, read from the plane)"));
        v.push(s(p("indexer.k_norm"), FP, &[c.idx_dim], "indexer.k_norm"));
    }
    if r.index_source {
        v.push(s(p("indexer.attn_q_b"), BF, &[c.q_lora, c.idx_heads * c.idx_dim], "indexer.wq_b"));
        v.push(s(p("indexer.proj"), BF, &[d, c.idx_heads], "indexer.weights_proj"));
    }
    if r.engram {
        v.push(s(p("engram_kv"), WQ, &[c.engram_width(), engram_kv_rows(c)], "engram.wkv: hc key rows + one value row"));
        v.push(s(p("engram_gate"), FP, &[d, c.hc], "engram q/k weights folded (q_weight * k_weight, per stream)"));
    }
    v
}

/// Rows of `engram_kv`: one key row of `n_embd` per hyper-connection stream plus one value row: 5 x 5120 = 25,600.
pub fn engram_kv_rows(c: &DscConfig) -> usize {
    (c.hc + 1) * c.n_embd
}

/// Engram table tensor of layer `l` in the Engram GGUF: `blk.<l>.engram_embd.weight`, rows of `ple_dim` IQ4_NL.
pub fn engram_table_name(l: usize) -> String {
    format!("blk.{l}.engram_embd.weight")
}

/// Rows of layer `l`'s table (the sum of its head bucket sizes) and the row offset of its first head.
pub fn engram_rows(c: &DscConfig, li: usize) -> (u64, u64) {
    let cols = c.engram_cols();
    let heads = &c.head_vocab[li * cols..(li + 1) * cols];
    (heads.iter().sum(), c.head_off[li * cols])
}

/// Everything wrong with a loaded model against the spec: missing tensors (unless the loader marked them
/// `missing` because the download is in progress), wrong type or shape, tensors nobody expects.
pub struct Report {
    pub checked: usize,
    pub errors: Vec<String>,
    pub pending: Vec<String>,
    pub unexpected: Vec<String>,
}

pub fn check(g: &Gguf, c: &DscConfig) -> Report {
    let mut r = Report { checked: 0, errors: vec![], pending: vec![], unexpected: vec![] };
    let mut seen = std::collections::HashSet::new();
    let mut specs = global_specs(c);
    for l in 0..c.n_layer {
        specs.extend(layer_specs(c, l));
    }
    for sp in &specs {
        seen.insert(sp.name.clone());
        match g.tensor(&sp.name) {
            None if g.missing.iter().any(|m| m == &sp.name) || g.missing.iter().any(|m| m.starts_with("shard ")) => {
                r.pending.push(sp.name.clone())
            }
            None => r.errors.push(format!("{}: missing", sp.name)),
            Some(t) => {
                r.checked += 1;
                if !sp.types.contains(&t.ty) {
                    r.errors.push(format!("{}: type {} not one of {:?}", sp.name, t.ty.name(), sp.types.iter().map(|x| x.name()).collect::<Vec<_>>()));
                }
                if t.dims != sp.dims {
                    r.errors.push(format!("{}: dims {:?}, expected {:?}", sp.name, t.dims, sp.dims));
                }
            }
        }
    }
    for t in &g.tensors {
        if !seen.contains(&t.name) && !t.name.contains("engram_embd") {
            r.unexpected.push(format!("{} {} {:?}", t.name, t.ty.name(), t.dims));
        }
    }
    r
}
