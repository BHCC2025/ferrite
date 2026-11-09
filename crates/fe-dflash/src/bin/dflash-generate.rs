//! Greedy generation with the DFlash2 drafter on the ferrite engine (single stream).
//!
//! `dflash-generate <target-model-dir> "<prompt>" [max_new] [--plain]`; env FERRITE_DRAFT_DIR (the drafter
//! checkpoint, fe_site::draft_dir), FE_PREFILL_CHUNK (2048), FE_CHAT=1 wraps the prompt in the chat
//! template's user turn with thinking off. Each round drafts a block [anchor, 7 proposals], verifies the 8 rows
//! in one target forward, keeps the matching prefix plus the target's next token (so the output equals plain
//! greedy decoding) and feeds the kept rows' target taps back into the draft context. --plain decodes without
//! the drafter for comparison.
use anyhow::{bail, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: dflash-generate <model-dir> \"<prompt>\" [max_new] [--plain]");
    }
    let model = PathBuf::from(&args[1]);
    let mut prompt = args[2].clone();
    let max_new: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let plain = args.iter().any(|a| a == "--plain");
    if std::env::var("FE_CHAT").is_ok_and(|v| v == "1") {
        prompt = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
    fe_dflash::generate(&model, &prompt, max_new, plain)
}
