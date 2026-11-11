//! Small-memory devices (a discrete card instead of the DGX Spark's unified memory): `FE_LOWMEM=1|0`, default on
//! when the device has less than 40 GB. It turns on the memory-saving forms, each with its own override:
//! `FE_EMBED_HOST` (the embedding table stays in host memory), `FE_KVQ` (the K/V cache as 6-bit or 4-bit records),
//! `FE_PROMPT_STREAM` (NVFP4/FP8 prompt weights streamed from host memory), `FE_MTP_BF16=0` (no bf16 copies of the MTP
//! head: its prompt rows take the trellis GEMV), `FE_PARK_HOST` (the prompt cache's park pool in pinned host memory),
//! `FE_CACHE_MTP` (the prompt cache with the MTP head as the drafter), `FE_CACHE_LEAD_MIN` (long prompts are saved
//! at the last whole chunk before each turn boundary, so the next turn resumes).
use std::sync::atomic::{AtomicU8, Ordering};

static LOW: AtomicU8 = AtomicU8::new(2);
static CC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The device's compute capability as major * 10 + minor (0 before `init`): 120 = RTX 50, 121 = GB10.
pub fn compute_capability() -> u32 {
    CC.load(Ordering::Relaxed)
}

/// Called once with the device's total memory and compute capability, before the weights load.
pub fn init(device_total_bytes: usize, cc: (i32, i32)) {
    CC.store((cc.0 * 10 + cc.1) as u32, Ordering::Relaxed);
    let low = match std::env::var("FE_LOWMEM") {
        Ok(v) => v != "0",
        Err(_) => device_total_bytes < 40 << 30,
    };
    LOW.store(low as u8, Ordering::Relaxed);
}

/// Whether the small-memory forms are on (false before `init`).
pub fn on() -> bool {
    LOW.load(Ordering::Relaxed) == 1
}

/// A small-memory option: its own variable when set (`0` = off), else `on()`.
pub fn flag(var: &str) -> bool {
    match std::env::var(var) {
        Ok(v) => v != "0",
        Err(_) => on(),
    }
}

/// `FE_MEM_TRACE=1`: print the device's free memory with a label (where the load's memory goes).
pub fn trace(label: &str) {
    if std::env::var("FE_MEM_TRACE").is_ok_and(|v| v == "1") {
        if let Ok((f, t)) = fe_cuda::mem_info() {
            eprintln!("mem: {:7.3} GB free of {:.2} after {label}", f as f64 / 1e9, t as f64 / 1e9);
        }
    }
}
