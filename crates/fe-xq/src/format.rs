//! `xq_config.json` and typed access to an XQ checkpoint directory (XQ v0, as written by arc-engine's quant tools).
//!
//! Checkpoint layout (gptq-mix5.0 / mix6.3 / mix3.9p): `xq-0000N.safetensors` shards + `xq_config.json` + config / tokenizer files.
//! XQ projections: `<name>.xq_q` I8 [n, k], `<name>.xq_s` F16 [n, k / g], `<name>.xq_su` F16 [k] (only when rotated).
//! Everything else is plain BF16 (embedding [vocab, hidden], norms, conv1d [conv_dim, 1, 4], A_log, dt_bias).
use crate::pack;
use crate::st::SafeTensors;
use anyhow::{bail, ensure, Context as _, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotate {
    /// x~ = fp16(x)
    None,
    /// x~ = fp16(FWHT128(su * x) / sqrt(128)) over consecutive 128-blocks of k
    Had128In,
}

#[derive(Debug, Clone)]
pub struct XqMeta {
    pub bits: u32,
    pub group: u32,
    pub rotate: Rotate,
    /// output rows
    pub n: usize,
    /// inputs
    pub k: usize,
    /// "gptq" | "rtn" (informational)
    pub method: String,
}

impl XqMeta {
    /// Device bytes of the packed tensor (codes + scales; su is separate, 2 k bytes).
    pub fn packed_bytes(&self) -> usize {
        pack::packed_dwords(self.bits, self.group, self.n, self.k) * 4
    }
}

#[derive(Debug, Clone)]
pub struct XqConfig {
    pub version: u64,
    /// tensor base name (without .xq_q) -> meta, sorted
    pub tensors: BTreeMap<String, XqMeta>,
    pub raw: Value,
}

impl XqConfig {
    pub fn load(path: &Path) -> Result<XqConfig> {
        let raw: Value = serde_json::from_slice(&std::fs::read(path).with_context(|| format!("reading {path:?}"))?)
            .with_context(|| format!("parsing {path:?}"))?;
        ensure!(raw.get("format").and_then(Value::as_str) == Some("xq"), "{path:?}: format is not \"xq\"");
        let version = raw.get("version").and_then(Value::as_u64).unwrap_or(0);
        ensure!(version == 0, "{path:?}: XQ version {version} (this reader knows v0)");
        let mut tensors = BTreeMap::new();
        for (name, v) in raw.get("tensors").and_then(Value::as_object).context("xq_config: tensors")? {
            let u = |k: &str| v.get(k).and_then(Value::as_u64).with_context(|| format!("xq_config {name}: {k}"));
            let bits = u("bits")? as u32;
            let group = u("group")? as u32;
            let rotate = match v.get("rotate").and_then(Value::as_str).unwrap_or("none") {
                "had128_in" => Rotate::Had128In,
                "none" => Rotate::None,
                r => bail!("xq_config {name}: rotate {r:?}"),
            };
            let shape = v.get("shape").and_then(Value::as_array).with_context(|| format!("xq_config {name}: shape"))?;
            ensure!(shape.len() == 2, "xq_config {name}: shape {shape:?}");
            let (n, k) = (shape[0].as_u64().unwrap_or(0) as usize, shape[1].as_u64().unwrap_or(0) as usize);
            pack::check_shape(bits, group, n, k).with_context(|| format!("xq_config {name}"))?;
            ensure!(rotate == Rotate::None || k % 128 == 0, "xq_config {name}: rotated with k {k}");
            let method = v.get("method").and_then(Value::as_str).unwrap_or("").to_string();
            tensors.insert(name.clone(), XqMeta { bits, group, rotate, n, k, method });
        }
        Ok(XqConfig { version, tensors, raw })
    }

    pub fn get(&self, name: &str) -> Result<&XqMeta> {
        self.tensors.get(name).with_context(|| format!("{name} is not an XQ tensor"))
    }

    /// Histogram (bits -> (tensors, codes)).
    pub fn bits_histogram(&self) -> BTreeMap<u32, (usize, usize)> {
        let mut h = BTreeMap::new();
        for m in self.tensors.values() {
            let e = h.entry(m.bits).or_insert((0, 0));
            e.0 += 1;
            e.1 += m.n * m.k;
        }
        h
    }
}

/// One XQ projection of a checkpoint: codes borrowed from the map, scales / su copied (they are small).
pub struct XqTensor<'a> {
    pub name: String,
    pub meta: XqMeta,
    /// int8 [n][k]
    pub q: &'a [i8],
    /// fp16 bits [n][k / g]
    pub s: Vec<u16>,
    /// fp16 bits [k] (Had128In only)
    pub su: Option<Vec<u16>>,
}

impl XqTensor<'_> {
    /// The device layout of codes + scales (`xq.zig pack`).
    pub fn pack(&self) -> Result<Vec<u32>> {
        pack::pack(self.meta.bits, self.meta.group, self.meta.n, self.meta.k, self.q, &self.s)
    }

    /// Scan every code against the width's range (the packer checks too; this reports where).
    pub fn check_codes(&self) -> Result<()> {
        let lo = -(1i32 << (self.meta.bits - 1));
        let hi = (1i32 << (self.meta.bits - 1)) - 1;
        if let Some(i) = self.q.iter().position(|&v| (v as i32) < lo || (v as i32) > hi) {
            bail!("{}: code {} at [{}][{}] outside {}-bit range", self.name, self.q[i], i / self.meta.k, i % self.meta.k, self.meta.bits);
        }
        Ok(())
    }
}

/// An XQ checkpoint directory.
pub struct Checkpoint {
    pub dir: PathBuf,
    pub cfg: XqConfig,
    pub st: SafeTensors,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Checkpoint> {
        let cfg = XqConfig::load(&dir.join("xq_config.json"))?;
        let st = SafeTensors::open_dir(dir)?;
        Ok(Checkpoint { dir: dir.to_path_buf(), cfg, st })
    }

    /// XQ projection by base name, e.g. `model.language_model.layers.0.mlp.gate_proj` or `lm_head`.
    pub fn xq(&self, name: &str) -> Result<XqTensor<'_>> {
        let meta = self.cfg.get(name)?.clone();
        let (n, k, g) = (meta.n, meta.k, meta.group as usize);
        let q = self.st.i8s(&format!("{name}.xq_q"), &[n, k])?;
        let s = self.st.u16s(&format!("{name}.xq_s"), "F16", &[n, k / g])?;
        let su_name = format!("{name}.xq_su");
        let su = match meta.rotate {
            Rotate::Had128In => Some(self.st.u16s(&su_name, "F16", &[k])?),
            Rotate::None => {
                ensure!(!self.st.has(&su_name), "{name}: rotate none but {su_name} present");
                None
            }
        };
        Ok(XqTensor { name: name.to_string(), meta, q, s, su })
    }

    /// A plain BF16 tensor's bits.
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        self.st.u16s(name, "BF16", shape)
    }

    pub fn has(&self, name: &str) -> bool {
        self.st.has(name)
    }

    /// Every config entry has its tensors with the right dtypes / shapes; every `.xq_q` in the shards has a config entry.
    pub fn validate(&self, scan_codes: bool) -> Result<()> {
        for name in self.cfg.tensors.keys() {
            let t = self.xq(name)?;
            if scan_codes {
                t.check_codes()?;
            }
        }
        for n in self.st.names() {
            if let Some(base) = n.strip_suffix(".xq_q") {
                ensure!(self.cfg.tensors.contains_key(base), "{n}: no xq_config entry");
            }
        }
        Ok(())
    }

    /// Total device bytes of all packed XQ tensors (+ su vectors).
    pub fn packed_bytes(&self) -> usize {
        self.cfg.tensors.values().map(|m| m.packed_bytes() + if m.rotate == Rotate::Had128In { 2 * m.k } else { 0 }).sum()
    }
}
