//! Qwen3.5 text config (the `text_config` block of config.json).
use anyhow::{Context as _, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub inter: usize,
    pub n_layers: usize,
    /// true = full attention, false = Gated DeltaNet
    pub full_attn: Vec<bool>,
    pub head_dim: usize,
    pub n_q: usize,
    pub n_kv: usize,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_dk: usize,
    pub lin_dv: usize,
    pub conv_k: usize,
    pub rope_theta: f64,
    pub rotary_dim: usize,
    pub eps: f32,
    pub vocab: usize,
    pub eos: u32,
    /// tensor-parallel ranks the head counts / MLP width are divided over (1 = whole model)
    pub world: usize,
}

impl Config {
    /// Find config.json: in the model dir, else in $FE_CONFIG_DIR, else in the EXL3 checkpoint (FERRITE_EXL3_DIR).
    pub fn locate(model_dir: &Path) -> PathBuf {
        let p = model_dir.join("config.json");
        if p.exists() {
            return p;
        }
        if let Ok(d) = std::env::var("FE_CONFIG_DIR") {
            return Path::new(&d).join("config.json");
        }
        fe_site::exl3_dir().join("config.json")
    }

    pub fn load(path: &Path) -> Result<Config> {
        let v: Value = serde_json::from_slice(&std::fs::read(path).with_context(|| format!("reading {path:?}"))?)?;
        let t = v.get("text_config").unwrap_or(&v);
        let u = |k: &str| -> Result<usize> {
            t.get(k).and_then(Value::as_u64).map(|x| x as usize).with_context(|| format!("config: missing {k}"))
        };
        let rp = t.get("rope_parameters").cloned().unwrap_or(Value::Null);
        let rope_theta = rp.get("rope_theta").or(t.get("rope_theta")).and_then(Value::as_f64).unwrap_or(1e7);
        let prf = rp.get("partial_rotary_factor").or(t.get("partial_rotary_factor")).and_then(Value::as_f64)
            .unwrap_or(0.25);
        let head_dim = u("head_dim")?;
        let lt = t.get("layer_types").and_then(Value::as_array).context("config: layer_types")?;
        let full_attn: Vec<bool> = lt.iter().map(|x| x.as_str() == Some("full_attention")).collect();
        let eos = t.get("eos_token_id").and_then(Value::as_u64).unwrap_or(248044) as u32;
        Ok(Config {
            hidden: u("hidden_size")?,
            inter: u("intermediate_size")?,
            n_layers: u("num_hidden_layers")?,
            full_attn,
            head_dim,
            n_q: u("num_attention_heads")?,
            n_kv: u("num_key_value_heads")?,
            lin_k_heads: u("linear_num_key_heads")?,
            lin_v_heads: u("linear_num_value_heads")?,
            lin_dk: u("linear_key_head_dim")?,
            lin_dv: u("linear_value_head_dim")?,
            conv_k: u("linear_conv_kernel_dim")?,
            rope_theta,
            rotary_dim: (head_dim as f64 * prf) as usize,
            eps: t.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6) as f32,
            vocab: u("vocab_size")?,
            eos,
            world: 1,
        })
    }

    /// One rank of a `world`-way tensor-parallel split: whole attention / GDN heads and MLP columns a rank
    /// (hidden, vocab and the layer list stay whole).
    pub fn shard(&self, world: usize) -> Config {
        let mut c = self.clone();
        c.n_q /= world;
        c.n_kv /= world;
        c.lin_k_heads /= world;
        c.lin_v_heads /= world;
        c.inter /= world;
        c.world = world;
        c
    }

    /// The kernels are specialised for the Qwen3.8-27B shapes (whole, or one rank of two); refuse anything else.
    pub fn check_supported(&self) -> Result<()> {
        let w = self.world;
        let ok = (w == 1 || w == 2) && self.hidden == 5120 && self.head_dim == 256 && self.n_q * w == 24 && self.n_kv * w == 4
            && self.lin_k_heads * w == 16 && self.lin_v_heads * w == 48 && self.lin_dk == 128 && self.lin_dv == 128
            && self.conv_k == 4 && self.rotary_dim == 64 && self.inter % 32 == 0;
        anyhow::ensure!(ok, "unsupported model shape: {self:?}");
        Ok(())
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.lin_k_heads * self.lin_dk + self.lin_v_heads * self.lin_dv
    }
    pub fn n_full(&self) -> usize {
        self.full_attn.iter().filter(|&&f| f).count()
    }
}
