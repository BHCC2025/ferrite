//! DeepSeek-V4.1-Flash Coder (GGUF arch `deepseek41`, expert-pruned, IQ1_M / IQ2_XXS experts) on Ferrite.
//! M1: config, tensor map, loader, inventory and placement plan. No GPU memory is touched by this crate yet.
pub mod config;
pub mod engine;
pub mod engram;
pub mod plan;
pub mod refmodel;
pub mod tok;
pub mod tensors;

use anyhow::{Context, Result};
use fe_gguf::Gguf;
use std::path::{Path, PathBuf};

pub use config::{DscConfig, LayerRole};

/// Default Engram file next to the model shards: `<dir>/../DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (the Coder folder
/// keeps the six-shard model in `IQ2_XXS/` and the Engram file beside it), or `DSC_ENGRAM`.
/// The folder scripts/fetch-deepseek.sh fills: `$FERRITE_MODELS/DeepSeek-V4.1-Flash-Coder` (FERRITE_MODELS default /var/tmp/models),
/// or DSC_DIR.
pub fn model_dir() -> PathBuf {
    if let Ok(d) = std::env::var("DSC_DIR") {
        return PathBuf::from(d);
    }
    let base = std::env::var("FERRITE_MODELS").unwrap_or_else(|_| "/var/tmp/models".into());
    Path::new(&base).join("DeepSeek-V4.1-Flash-Coder")
}

/// First shard of the Coder GGUF: DSC_MODEL, else the one in [`model_dir`].
pub fn model_path() -> PathBuf {
    match std::env::var("DSC_MODEL") {
        Ok(p) => PathBuf::from(p),
        Err(_) => model_dir().join("DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf"),
    }
}

/// The DSpark drafter GGUF: DSC_DRAFT_MODEL, else the one in [`model_dir`].
pub fn draft_path() -> PathBuf {
    match std::env::var("DSC_DRAFT_MODEL") {
        Ok(p) => PathBuf::from(p),
        Err(_) => model_dir().join("dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"),
    }
}

pub fn engram_path(model: &Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("DSC_ENGRAM") {
        return Some(PathBuf::from(p));
    }
    let dir = model.parent()?;
    for d in [dir, dir.parent()?] {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                if n.contains("Engram") && n.ends_with(".gguf") {
                    return Some(e.path());
                }
            }
        }
    }
    None
}

/// Open the model (any shard path of the split) and attach the Engram GGUF when there is one. The split is named
/// `-0000N-of-00006` but the sixth file is the Engram GGUF, published under its own name, so the split always
/// opens leniently and the strict mode here requires exactly that one shard (and no tensor) to be absent.
/// `partial` tolerates shards / tensors still downloading (inventory and tests only).
pub fn open_model(model: &Path, engram: Option<&Path>, partial: bool) -> Result<Gguf> {
    let mut g = Gguf::open_partial(model).with_context(|| format!("model {}", model.display()))?;
    if let Some(e) = engram {
        if e.exists() {
            let kv = if partial { g.attach_partial(e) } else { g.attach(e) }
                .with_context(|| format!("engram {}", e.display()))?;
            let arch = kv.iter().find(|(k, _)| k == "general.architecture").and_then(|(_, v)| v.as_str().map(String::from));
            anyhow::ensure!(arch.as_deref() == Some("deepseek41-engram"), "{}: architecture {:?}, expected deepseek41-engram", e.display(), arch);
        }
    }
    if !partial {
        let paths = fe_gguf::shard_paths(model);
        let sixth = paths.last().map(|p| format!("shard {}", p.display()));
        let rest: Vec<&String> = g.missing.iter().filter(|m| Some(*m) != sixth.as_ref()).collect();
        anyhow::ensure!(rest.is_empty(), "model incomplete: {:?}", &rest[..rest.len().min(5)]);
        anyhow::ensure!(engram.is_some_and(|e| e.exists()), "the Engram GGUF (sixth file of the split) was not found");
        g.missing.clear();
    } else {
        // a download in progress: the sixth shard is expected to be absent, not "pending"
        let paths = fe_gguf::shard_paths(model);
        if let Some(last) = paths.last() {
            if !last.exists() && engram.is_some_and(|e| e.exists()) {
                g.missing.retain(|m| m != &format!("shard {}", last.display()));
            }
        }
    }
    Ok(g)
}

pub static K_DS_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_ds.cubin"));
pub static K_ENGRAM_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_engram.cubin"));
pub static K_DSPARK_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_dspark.cubin"));

/// Row bytes of the plane repack (GgmlType::RepR3*) of a row of `k` values.
pub fn cpu_rep_row_bytes(k: usize) -> usize {
    fe_ggq::cpu::repack::rep3_row_bytes(k)
}

/// The CPU-lane repack type of an original expert type (DSC_REPACK_FMT=planes forces planes everywhere; default: the hybrid for IQ2_XXS).
pub fn rep_type(t: fe_gguf::GgmlType) -> Option<fe_gguf::GgmlType> {
    use fe_gguf::GgmlType::*;
    let planes = std::env::var("DSC_REPACK_FMT").map_or(false, |v| v == "planes");
    match t {
        IQ2XXS if !planes => Some(RepHXxs),
        IQ2XXS => Some(RepR3Xxs),
        IQ2XS => Some(RepR3Xs),
        IQ1M => Some(RepR3Iq1M),
        _ => None,
    }
}

pub fn rep_row_bytes(rt: fe_gguf::GgmlType, k: usize) -> usize {
    if rt == fe_gguf::GgmlType::RepHXxs { fe_ggq::cpu::repack::hxxs_row_bytes(k) } else { fe_ggq::cpu::repack::rep3_row_bytes(k) }
}

/// Repack one row of original type `ot` into `rt`.
///
/// # Safety
/// Needs AVX2 (the engine's CPU lane already requires it).
pub unsafe fn repack_row(rt: fe_gguf::GgmlType, ot: fe_gguf::GgmlType, src: &[u8], k: usize, dst: &mut [u8]) -> bool {
    if rt == fe_gguf::GgmlType::RepHXxs {
        fe_ggq::cpu::repack::repack_hxxs_row(src, k, dst);
        true
    } else {
        fe_ggq::cpu::repack::repack_rep3_row(ot, src, k, dst)
    }
}
