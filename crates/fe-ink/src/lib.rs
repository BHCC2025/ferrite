//! Inkling-2-Mini on Ferrite: loader (safetensors read directly), the engine (looped layers with attention
//! residuals, sliding / global attention with relative bias and sinks, NVFP4 / MXFP8 experts) and CLI helpers.
pub mod ckpt;
pub mod cpuref;
pub mod draft;
pub mod sampler;
pub mod spec;
pub mod engine;
pub mod pcache;
pub mod fold;
pub mod weights;

/// Per-layer pick shares of the experts (code and prose averaged), the order in which experts leave the card on the 16 GB layout.
pub static HEAT_SHARE: &str = include_str!("../heat/heat_share.json");

/// Folder with the model files (config.json, tokenizer.json, the safetensors, mtp.safetensors): `INK_DIR`, else
/// `$FERRITE_MODELS/Inkling-2-Mini` (`FERRITE_MODELS` default /var/tmp/models), the folder scripts/fetch-inkling.sh fills.
pub fn model_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("INK_DIR") {
        return d.into();
    }
    let base = std::env::var("FERRITE_MODELS").unwrap_or_else(|_| "/var/tmp/models".into());
    std::path::Path::new(&base).join("Inkling-2-Mini")
}

/// The shipped heat table, or the JSON file named by `INK_HEAT_SHARE`.
pub fn heat_share() -> anyhow::Result<Vec<Vec<f64>>> {
    match std::env::var("INK_HEAT_SHARE") {
        Ok(p) => fe_inkk::slots::load_heat_share(&p),
        Err(_) => fe_inkk::slots::parse_heat_share(HEAT_SHARE.as_bytes()),
    }
}

pub static K_DRAFT_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_draft.cubin"));
pub static K_INKM_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_inkm.cubin"));

pub const D: usize = 2304;
pub const NL: usize = 24;
pub const VOCAB: usize = 200058;
pub const LOOP0: usize = 6;
pub const LOOP1: usize = 18;
pub const PASSES: usize = 3;
pub const NAPP: usize = NL + (PASSES - 1) * (LOOP1 - LOOP0); // 48
pub const FULL_LAYERS: [usize; 4] = [5, 11, 17, 23];

pub fn is_full(li: usize) -> bool {
    FULL_LAYERS.contains(&li)
}
/// Cache / state slot of layer `li` in loop pass `pass` (0..3).
pub fn app_index(li: usize, pass: usize) -> usize {
    if li < LOOP0 {
        li
    } else if li < LOOP1 {
        li + (LOOP1 - LOOP0) * pass
    } else {
        li + (LOOP1 - LOOP0) * (PASSES - 1)
    }
}

/// Boolean INK_* switch: set and not "0" / "false" / "off" / "no" / empty (so INK_KVQ6=0 is OFF).
pub fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "off" | "no"),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn model_dir_follows_the_environment() {
        // SAFETY: tests of this crate that touch the environment run in this one function
        unsafe {
            std::env::set_var("INK_DIR", "/models/ink");
            assert_eq!(super::model_dir(), std::path::PathBuf::from("/models/ink"));
            std::env::remove_var("INK_DIR");
            std::env::set_var("FERRITE_MODELS", "/m");
            assert_eq!(super::model_dir(), std::path::PathBuf::from("/m/Inkling-2-Mini"));
            std::env::set_var("INK_TEST_FLAG", "0");
            assert!(!super::env_flag("INK_TEST_FLAG"));
            std::env::set_var("INK_TEST_FLAG", "off");
            assert!(!super::env_flag("INK_TEST_FLAG"));
            std::env::set_var("INK_TEST_FLAG", "1");
            assert!(super::env_flag("INK_TEST_FLAG"));
            std::env::set_var("INK_TEST_FLAG", "");
            assert!(!super::env_flag("INK_TEST_FLAG"));
            std::env::remove_var("INK_TEST_FLAG");
            assert!(!super::env_flag("INK_TEST_FLAG"));
        }
    }

    #[test]
    fn shipped_heat_table_has_24_layers_of_128_experts() {
        let h = super::heat_share().unwrap();
        assert_eq!(h.len(), 24);
        assert!(h.iter().all(|l| l.len() == 128));
    }
}
