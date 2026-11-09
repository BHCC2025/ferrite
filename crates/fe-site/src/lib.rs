//! Site settings: where the model folders are and how the two TP ranks reach each other. Each comes from a
//! FERRITE_* environment variable (an older FE_* name is still read after it) and otherwise has the default below
//! (for the fabric, the DGX Spark's ConnectX-7 device names; rank 0's address has none). The tools/ scripts read the same variables (tools/site.sh).
//!
//! | variable | default |
//! |---|---|
//! | FERRITE_MODELS | /var/tmp/models (the folder holding the three below) |
//! | FERRITE_EXL3_DIR | $FERRITE_MODELS/Qwen3.8-27B-EXL3-3.0bpw-mix |
//! | FERRITE_NVFP4_DIR (FE_PROMPT_WEIGHTS) | $FERRITE_MODELS/nvidia-Qwen3.8-27B-NVFP4 |
//! | FERRITE_DRAFT_DIR (FE_DRAFT_DIR) | $FERRITE_MODELS/z-lab-Qwen3.8-27B-DFlash2 |
//! | FERRITE_TP_ADDR (FE_TP_ADDR) | $FERRITE_TP_HOST:29600; no default: TP2 needs one of them (tools/tp2.sh sets it) |
//! | FERRITE_ROCE_HCA (FE_ROCE_HCA) | rocep1s0f0,roceP2p1s0f0 |
//! | FERRITE_ROCE_GID (FE_ROCE_GID) | 5 |
//! | FERRITE_NCCL_LIB (FE_NCCL_LIB) | libnccl.so.2 on the loader path |
use std::path::PathBuf;

/// The first of `names` that is set (and not empty).
pub fn var(names: &[&str]) -> Option<String> {
    names.iter().find_map(|n| std::env::var(n).ok().filter(|v| !v.is_empty()))
}

/// FERRITE_MODELS: the folder holding the standard model folders.
pub fn models_root() -> PathBuf {
    var(&["FERRITE_MODELS"]).unwrap_or_else(|| "/var/tmp/models".into()).into()
}

/// FERRITE_EXL3_DIR: the EXL3 checkpoint (decode weights, config.json, tokenizer).
pub fn exl3_dir() -> PathBuf {
    var(&["FERRITE_EXL3_DIR"]).map_or_else(|| models_root().join("Qwen3.8-27B-EXL3-3.0bpw-mix"), PathBuf::from)
}

/// FERRITE_NVFP4_DIR (FE_PROMPT_WEIGHTS): the NVIDIA NVFP4 / FP8 checkpoint (prompt rows with FE_PROMPT=fp4).
pub fn nvfp4_dir() -> PathBuf {
    var(&["FERRITE_NVFP4_DIR", "FE_PROMPT_WEIGHTS"])
        .map_or_else(|| models_root().join("nvidia-Qwen3.8-27B-NVFP4"), PathBuf::from)
}

/// FERRITE_DRAFT_DIR (FE_DRAFT_DIR): the DFlash2 drafter checkpoint.
pub fn draft_dir() -> PathBuf {
    var(&["FERRITE_DRAFT_DIR", "FE_DRAFT_DIR"]).map_or_else(|| models_root().join("z-lab-Qwen3.8-27B-DFlash2"), PathBuf::from)
}

/// FERRITE_TP_ADDR (FE_TP_ADDR): the TP2 rendezvous, rank 0's fabric address (FERRITE_TP_HOST) and port.
/// There is no default address: TP2 without either variable stops here with a message.
pub fn tp_addr() -> String {
    var(&["FERRITE_TP_ADDR", "FE_TP_ADDR"])
        .or_else(|| var(&["FERRITE_TP_HOST"]).map(|h| format!("{h}:29600")))
        .unwrap_or_else(|| {
            panic!(
                "TP2 needs rank 0's fabric address: set FERRITE_TP_ADDR=<host>:<port> or FERRITE_TP_HOST=<host> on both \
                 ranks (tools/tp2.sh does; tools/fabric-detect.sh suggests the host)"
            )
        })
}

/// FERRITE_ROCE_HCA (FE_ROCE_HCA): one or two RDMA devices (comma-separated) for the TP2 RoCE exchange.
pub fn roce_hca() -> String {
    var(&["FERRITE_ROCE_HCA", "FE_ROCE_HCA"]).unwrap_or_else(|| "rocep1s0f0,roceP2p1s0f0".into())
}

/// FERRITE_ROCE_GID (FE_ROCE_GID): the RoCE v2 GID index of the fabric address on those devices.
pub fn roce_gid() -> i32 {
    var(&["FERRITE_ROCE_GID", "FE_ROCE_GID"]).and_then(|v| v.parse().ok()).unwrap_or(5)
}

/// NCCL libraries to try in order: FERRITE_NCCL_LIB (FE_NCCL_LIB), then libnccl.so.2 on the loader path.
pub fn nccl_libs() -> Vec<String> {
    let mut v: Vec<String> = var(&["FERRITE_NCCL_LIB", "FE_NCCL_LIB"]).into_iter().collect();
    v.push("libnccl.so.2".into());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    // one test: the environment is process-wide
    #[test]
    fn defaults_and_overrides() {
        for k in ["FERRITE_MODELS", "FERRITE_EXL3_DIR", "FERRITE_NVFP4_DIR", "FE_PROMPT_WEIGHTS", "FERRITE_DRAFT_DIR",
            "FE_DRAFT_DIR", "FERRITE_TP_ADDR", "FE_TP_ADDR", "FERRITE_TP_HOST", "FERRITE_ROCE_HCA", "FE_ROCE_HCA",
            "FERRITE_ROCE_GID", "FE_ROCE_GID", "FERRITE_NCCL_LIB", "FE_NCCL_LIB"] {
            unsafe { std::env::remove_var(k) };
        }
        assert_eq!(exl3_dir(), PathBuf::from("/var/tmp/models/Qwen3.8-27B-EXL3-3.0bpw-mix"));
        assert_eq!(nvfp4_dir(), PathBuf::from("/var/tmp/models/nvidia-Qwen3.8-27B-NVFP4"));
        assert_eq!(draft_dir(), PathBuf::from("/var/tmp/models/z-lab-Qwen3.8-27B-DFlash2"));
        assert!(std::panic::catch_unwind(tp_addr).is_err(), "no default rendezvous address");
        assert_eq!((roce_hca().as_str(), roce_gid()), ("rocep1s0f0,roceP2p1s0f0", 5));
        unsafe {
            std::env::set_var("FERRITE_MODELS", "/m");
            std::env::set_var("FE_DRAFT_DIR", "/old/draft");
            std::env::set_var("FERRITE_TP_HOST", "10.0.0.7");
            std::env::set_var("FE_ROCE_GID", "3");
            std::env::set_var("FERRITE_ROCE_GID", "");
        }
        assert_eq!(exl3_dir(), PathBuf::from("/m/Qwen3.8-27B-EXL3-3.0bpw-mix"));
        assert_eq!(nvfp4_dir(), PathBuf::from("/m/nvidia-Qwen3.8-27B-NVFP4"));
        assert_eq!(draft_dir(), PathBuf::from("/old/draft"));
        assert_eq!(tp_addr(), "10.0.0.7:29600");
        assert_eq!(roce_gid(), 3);
        unsafe { std::env::set_var("FERRITE_DRAFT_DIR", "/new/draft") };
        assert_eq!(draft_dir(), PathBuf::from("/new/draft"));
    }
}
