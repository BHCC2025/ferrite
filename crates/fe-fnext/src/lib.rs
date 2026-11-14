//! Qwen3.8-Flash-Next (llama.cpp `qwen4exp`): 48 blocks of hyper-connected GDN / QSA attention + 256-expert MoE,
//! a PLE n-gram table on layer 1 and an MTP block, read from GGUF, for one 16 GB card with the experts mostly
//! on the host. Kernels: kernels/k-fnext.
#![allow(unsafe_op_in_unsafe_fn)]

pub mod config;
pub mod engine;
pub mod cores;
pub mod fast;
pub mod plan;

pub static K_FNEXT_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_fnext.cubin"));

/// Open the model: the trunk GGUF (any shard of a split; the ISTA-DASLab IQ1_M files as published) plus the MTP draft
/// head when there is one. The draft is a separate single-file GGUF (`blk.<n_layer>.*`, `nextn_predict_layers` >= 1):
/// FNEXT_DRAFT=<path>, or found next to the model (the only `*.gguf` in its folder that is not one of its shards and
/// has `mtp` in its name); FNEXT_DRAFT=none skips it (no MTP). The draft's tensors become one more shard and the
/// trunk's block count / nextn / compress ratios are extended in memory, the same view a model with the MTP block
/// inside it gives. A trunk that already carries an MTP block is used as it is.
pub fn open_model(path: &std::path::Path) -> anyhow::Result<fe_gguf::Gguf> {
    use anyhow::{bail, Context};
    use fe_gguf::Value;
    let mut g = fe_gguf::Gguf::open(path)?;
    let key = |s: &str| format!("qwen4exp.{s}");
    let nextn = g.get_i64(&key("nextn_predict_layers")).unwrap_or(0);
    let draft = match std::env::var("FNEXT_DRAFT") {
        Ok(v) if v.is_empty() || v == "none" || v == "0" => None,
        Ok(v) => Some(std::path::PathBuf::from(v)),
        Err(_) => find_draft(path),
    };
    let Some(draft) = draft else { return Ok(g) };
    if nextn > 0 {
        eprintln!("fnext: {} already has an MTP block, draft {} not used", path.display(), draft.display());
        return Ok(g);
    }
    let n_layer = g.get_i64(&key("block_count")).context("block_count missing")?;
    let kv = g.attach(&draft).with_context(|| format!("draft {}", draft.display()))?;
    let dget = |k: &str| kv.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
    let di = |k: &str| dget(&key(k)).and_then(|v| v.as_i64());
    if dget("general.architecture").as_ref().and_then(|v| v.as_str()) != Some("qwen4exp") {
        bail!("draft {}: not a qwen4exp GGUF", draft.display());
    }
    let d_nextn = di("nextn_predict_layers").unwrap_or(0);
    if d_nextn < 1 || di("block_count") != Some(n_layer + d_nextn) {
        bail!("draft {}: expected block_count {} + nextn_predict_layers >= 1, got {:?} / {d_nextn}", draft.display(),
            n_layer, di("block_count"));
    }
    for k in ["embedding_length", "hyper_connection.count", "hyper_connection.low_rank", "attention.head_count",
        "attention.head_count_kv", "attention.key_length", "attention.indexer.head_count"] {
        if di(k) != g.get_i64(&key(k)) {
            bail!("draft {}: {} = {:?}, the model has {:?}", draft.display(), key(k), di(k), g.get_i64(&key(k)));
        }
    }
    if g.tensor(&format!("blk.{n_layer}.nextn.eh_proj.weight")).is_none() {
        bail!("draft {}: no blk.{n_layer}.nextn.eh_proj.weight", draft.display());
    }
    let ratios = |v: Option<Value>| v.and_then(|v| v.as_i64_vec()).unwrap_or_default();
    let mut cr = ratios(g.get(&key("attention.compress_ratios")).cloned());
    let dcr = ratios(dget(&key("attention.compress_ratios")));
    if cr.len() != n_layer as usize || dcr.len() != (n_layer + d_nextn) as usize {
        bail!("draft {}: compress_ratios length {} / {}", draft.display(), cr.len(), dcr.len());
    }
    cr.extend_from_slice(&dcr[n_layer as usize..]);
    g.set(&key("block_count"), Value::U32((n_layer + d_nextn) as u32));
    g.set(&key("nextn_predict_layers"), Value::U32(d_nextn as u32));
    g.set(&key("attention.compress_ratios"), Value::Array(cr.into_iter().map(|v| Value::I32(v as i32)).collect()));
    eprintln!("fnext: MTP draft {}", draft.display());
    Ok(g)
}

/// The draft next to `model`: the one `*.gguf` in its folder that is not a shard of it and has `mtp` in its name.
fn find_draft(model: &std::path::Path) -> Option<std::path::PathBuf> {
    let shards = fe_gguf::shard_paths(model);
    let dir = model.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
    let mut v: Vec<std::path::PathBuf> = std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            n.ends_with(".gguf") && n.contains("mtp") && !shards.iter().any(|s| s.file_name() == p.file_name())
        })
        .collect();
    v.sort();
    if v.len() > 1 {
        eprintln!("fnext: several MTP files next to the model, using {} (FNEXT_DRAFT picks one)", v[0].display());
    }
    v.into_iter().next()
}

/// The model file (first split): FNEXT_MODEL, else the layout of scripts/fetch-flashnext.sh under FERRITE_MODELS
/// (default /var/tmp/models).
pub fn default_model() -> std::path::PathBuf {
    std::env::var("FNEXT_MODEL").map(Into::into).unwrap_or_else(|_| {
        let root = std::env::var("FERRITE_MODELS").unwrap_or_else(|_| "/var/tmp/models".into());
        format!("{root}/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf").into()
    })
}
