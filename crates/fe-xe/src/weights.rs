//! The weights of Qwen3.8-27B as the B70 forward holds them, prepared on the host (arc-engine `qwen_load.zig` conventions).
//!
//! | tensor | device form |
//! |---|---|
//! | every projection (q/k/v/o, GDN qkv/z/b/a/out, MLP gate/up/down, lm_head, MTP fc + block) | XQ packed (fe-xq `pack`), su fp16 [k] shared by projections with identical su |
//! | embed_tokens [248320][5120] | bf16 as stored (2.54 GB on the device) |
//! | input / post-attention / final / q / k norms, MTP norms | bf16(1 + w): the checkpoint keeps the zero-centred form (`qwen_load.normWs`) |
//! | GDN gated norm `linear_attn.norm` [128] | bf16 as stored (no shift) |
//! | conv1d [10240][1][4] | bf16 as stored |
//! | A_log, dt_bias [48] | fp32 (widened from bf16) |
//!
//! Loading is two steps so it can be tested without a device: [`Planner`] reads + packs on the host ([`HostTensor`]), the device side
//! (fe-ze) uploads each one into its own buffer. The packed XQ bytes are byte-identical to arc-engine's (fe-xq tests).
use crate::config::Config;
use anyhow::{ensure, Context as _, Result};
use fe_xq::half::{bf16_to_f32, f32_to_bf16};
use fe_xq::{Checkpoint, Rotate};

/// Checkpoint name prefix of the language model in XQ checkpoints.
pub const BASE: &str = "model.language_model";

/// One tensor ready for upload.
pub enum HostTensor {
    /// XQ: packed codes + scales (dwords), su (fp16 bits, rotated tensors only), shape and width
    Xq { packed: Vec<u32>, su: Option<Vec<u16>>, bits: u32, group: u32, n: usize, k: usize, split: u32 },
    /// bf16 bits
    Bf16(Vec<u16>),
    /// fp32
    F32(Vec<f32>),
}

impl HostTensor {
    pub fn device_bytes(&self) -> usize {
        match self {
            HostTensor::Xq { packed, su, .. } => packed.len() * 4 + su.as_ref().map_or(0, |s| s.len() * 2),
            HostTensor::Bf16(v) => v.len() * 2,
            HostTensor::F32(v) => v.len() * 4,
        }
    }
}

/// bf16(1 + w) with RNE, as arc-engine's `normWs` (f32 add, then the RNE rounding of the bits).
pub fn shift_norm(w: &[u16]) -> Vec<u16> {
    w.iter().map(|&e| f32_to_bf16(bf16_to_f32(e) + 1.0)).collect()
}

pub struct Planner<'a> {
    pub ck: &'a Checkpoint,
    pub cfg: &'a Config,
}

/// FE_XE_SPLIT=table: per-shape K splits for 16-row verify windows (B70 split sweep, xq-check m1 / m8 / m16; per window 16-row
/// 52.9 -> 42.6 ms, 8-row 36.3 -> 36.8 ms with FE_XE_MVP=1): gate / up 6, down 3, GDN qkv 3, attn q 6; the other shapes keep arc.
pub const SPLIT_TABLE: &str = "17408x5120=6,5120x17408=3,10240x5120=3,12288x5120=6";

/// The XQ decode matvec's K split of an (n, k) tensor: a function of the shape only, so every row count (1-row decode, 8 / 16-row
/// verify windows) sums in one order and verify == decode stays bitwise. FE_XE_SPLIT = arc (default: arc-engine's split_for, the
/// oracle-compatible one) | table ([`SPLIT_TABLE`]) | <target> (split_for with that sub-group target) | "NxK=S,NxK=S" (those shapes'
/// splits, the rest arc).
pub fn split_choice(n: usize, k: usize, g: u32) -> Result<u32> {
    let arc = fe_xq::pack::split_for(n, k, g, None);
    let mut v = std::env::var("FE_XE_SPLIT").unwrap_or_default();
    if v.is_empty() || v == "arc" {
        return Ok(arc);
    }
    if v == "table" {
        v = SPLIT_TABLE.to_string();
    }
    if let Ok(t) = v.parse::<u32>() {
        return Ok(fe_xq::pack::split_for(n, k, g, Some(t)));
    }
    for e in v.split(',').filter(|e| !e.is_empty()) {
        let (shape, sp) = e.split_once('=').context("FE_XE_SPLIT: NxK=S,...")?;
        let (a, b) = shape.split_once('x').context("FE_XE_SPLIT: NxK=S,...")?;
        if a.trim().parse::<usize>()? == n && b.trim().parse::<usize>()? == k {
            let sp: u32 = sp.trim().parse()?;
            ensure!(sp >= 1 && 2 * sp * 16 <= 1024, "FE_XE_SPLIT {e}: split 1..32");
            return Ok(sp.min((k / g as usize) as u32));
        }
    }
    Ok(arc)
}

impl<'a> Planner<'a> {
    pub fn new(ck: &'a Checkpoint, cfg: &'a Config) -> Self {
        Self { ck, cfg }
    }

    pub fn xq(&self, name: &str) -> Result<HostTensor> {
        let t = self.ck.xq(name)?;
        let packed = t.pack()?;
        let m = &t.meta;
        ensure!(m.group == 128, "{name}: group {} (the B70 kernels are built for g128)", m.group);
        let split = split_choice(m.n, m.k, m.group)?;
        let su = if m.rotate == Rotate::Had128In { t.su.clone() } else { None };
        Ok(HostTensor::Xq { packed, su, bits: m.bits, group: m.group, n: m.n, k: m.k, split })
    }

    pub fn norm(&self, name: &str, n: usize) -> Result<HostTensor> {
        Ok(HostTensor::Bf16(shift_norm(&self.ck.bf16(name, &[n])?)))
    }

    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<HostTensor> {
        Ok(HostTensor::Bf16(self.ck.bf16(name, shape)?))
    }

    pub fn f32(&self, name: &str, n: usize) -> Result<HostTensor> {
        Ok(HostTensor::F32(self.ck.bf16(name, &[n])?.into_iter().map(bf16_to_f32).collect()))
    }

    /// Every tensor of layer `i`, in a fixed order: (role, tensor).
    pub fn layer(&self, i: usize) -> Result<Vec<(&'static str, HostTensor)>> {
        let c = self.cfg;
        let p = format!("{BASE}.layers.{i}");
        let mut v = vec![("input_norm", self.norm(&format!("{p}.input_layernorm.weight"), c.hidden)?)];
        if c.full_attn[i] {
            let a = format!("{p}.self_attn");
            v.push(("q", self.xq(&format!("{a}.q_proj"))?));
            v.push(("k", self.xq(&format!("{a}.k_proj"))?));
            v.push(("v", self.xq(&format!("{a}.v_proj"))?));
            v.push(("o", self.xq(&format!("{a}.o_proj"))?));
            v.push(("q_norm", self.norm(&format!("{a}.q_norm.weight"), c.head_dim)?));
            v.push(("k_norm", self.norm(&format!("{a}.k_norm.weight"), c.head_dim)?));
        } else {
            let g = format!("{p}.linear_attn");
            v.push(("qkv", self.xq(&format!("{g}.in_proj_qkv"))?));
            v.push(("z", self.xq(&format!("{g}.in_proj_z"))?));
            v.push(("b", self.xq(&format!("{g}.in_proj_b"))?));
            v.push(("a", self.xq(&format!("{g}.in_proj_a"))?));
            v.push(("out", self.xq(&format!("{g}.out_proj"))?));
            v.push(("conv", self.bf16(&format!("{g}.conv1d.weight"), &[c.conv_dim(), 1, c.conv_k])?));
            v.push(("a_log", self.f32(&format!("{g}.A_log"), c.lin_v_heads)?));
            v.push(("dt_bias", self.f32(&format!("{g}.dt_bias"), c.lin_v_heads)?));
            v.push(("gnorm", self.bf16(&format!("{g}.norm.weight"), &[c.lin_dv])?));
        }
        v.push(("post_norm", self.norm(&format!("{p}.post_attention_layernorm.weight"), c.hidden)?));
        v.push(("gate", self.xq(&format!("{p}.mlp.gate_proj"))?));
        v.push(("up", self.xq(&format!("{p}.mlp.up_proj"))?));
        v.push(("down", self.xq(&format!("{p}.mlp.down_proj"))?));
        Ok(v)
    }

    /// Embedding, final norm, lm_head.
    pub fn top(&self) -> Result<Vec<(&'static str, HostTensor)>> {
        let c = self.cfg;
        Ok(vec![
            ("embed", self.bf16(&format!("{BASE}.embed_tokens.weight"), &[c.vocab, c.hidden])?),
            ("final_norm", self.norm(&format!("{BASE}.norm.weight"), c.hidden)?),
            ("lm_head", self.xq("lm_head")?),
        ])
    }

    /// The MTP head (one full-attention block behind fc over [norm(embedding) | norm(hidden)]).
    pub fn mtp(&self) -> Result<Vec<(&'static str, HostTensor)>> {
        let c = self.cfg;
        let a = "mtp.layers.0.self_attn";
        Ok(vec![
            ("pre_fc_norm_embedding", self.norm("mtp.pre_fc_norm_embedding.weight", c.hidden)?),
            ("pre_fc_norm_hidden", self.norm("mtp.pre_fc_norm_hidden.weight", c.hidden)?),
            ("fc", self.xq("mtp.fc")?),
            ("input_norm", self.norm("mtp.layers.0.input_layernorm.weight", c.hidden)?),
            ("q", self.xq(&format!("{a}.q_proj"))?),
            ("k", self.xq(&format!("{a}.k_proj"))?),
            ("v", self.xq(&format!("{a}.v_proj"))?),
            ("o", self.xq(&format!("{a}.o_proj"))?),
            ("q_norm", self.norm(&format!("{a}.q_norm.weight"), c.head_dim)?),
            ("k_norm", self.norm(&format!("{a}.k_norm.weight"), c.head_dim)?),
            ("post_norm", self.norm("mtp.layers.0.post_attention_layernorm.weight", c.hidden)?),
            ("gate", self.xq("mtp.layers.0.mlp.gate_proj")?),
            ("up", self.xq("mtp.layers.0.mlp.up_proj")?),
            ("down", self.xq("mtp.layers.0.mlp.down_proj")?),
            ("norm", self.norm("mtp.norm.weight", c.hidden)?),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_shift_rounds_like_zig() {
        // 0.0 -> 1.0; a tiny w rounds to 1.0 in bf16; -1.0 -> 0
        assert_eq!(shift_norm(&[0x0000, 0x3a00, 0xbf80]), vec![0x3f80, 0x3f80, 0x0000]);
    }
}
