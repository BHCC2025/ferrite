//! xq-tool: inspect / validate an XQ checkpoint, write the packed device bytes of a tensor.
//!   xq-tool info DIR                      tensors, bits histogram, packed size
//!   xq-tool validate DIR [--scan]         config vs shards (dtypes, shapes); --scan: every code in range
//!   xq-tool pack DIR NAME OUT.bin         packed dwords of one projection (compare with the zig xq_packdump oracle)
use anyhow::{bail, Result};
use fe_xq::Checkpoint;
use std::path::Path;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let cmd = a.get(1).map(String::as_str).unwrap_or("");
    match cmd {
        "info" if a.len() >= 3 => {
            let ck = Checkpoint::open(Path::new(&a[2]))?;
            println!("{}: {} XQ tensors, {} safetensors entries", a[2], ck.cfg.tensors.len(), ck.st.tensors.len());
            for (b, (n, codes)) in ck.cfg.bits_histogram() {
                println!("  {b}-bit: {n} tensors, {:.3} G codes", codes as f64 / 1e9);
            }
            println!("  packed device bytes (XQ + su): {:.3} GB", ck.packed_bytes() as f64 / 1e9);
            for (name, m) in &ck.cfg.tensors {
                if !name.contains(".layers.") || name.starts_with("mtp.") {
                    println!("  {name}: {} bits g{} {:?} [{}x{}] {}", m.bits, m.group, m.rotate, m.n, m.k, m.method);
                }
            }
        }
        "validate" if a.len() >= 3 => {
            let ck = Checkpoint::open(Path::new(&a[2]))?;
            let scan = a.iter().any(|x| x == "--scan");
            ck.validate(scan)?;
            println!("ok: {} XQ tensors{}", ck.cfg.tensors.len(), if scan { ", all codes in range" } else { "" });
        }
        "pack" if a.len() >= 5 => {
            let ck = Checkpoint::open(Path::new(&a[2]))?;
            let t = ck.xq(&a[3])?;
            let p = t.pack()?;
            let bytes: Vec<u8> = p.iter().flat_map(|w| w.to_le_bytes()).collect();
            std::fs::write(&a[4], &bytes)?;
            println!("{}: {} bits g{} [{}x{}] -> {} bytes", a[3], t.meta.bits, t.meta.group, t.meta.n, t.meta.k, bytes.len());
        }
        _ => bail!("usage: xq-tool info DIR | validate DIR [--scan] | pack DIR NAME OUT.bin"),
    }
    Ok(())
}
