//! gguf-ls <file.gguf> [--kv] [--all]: metadata and a tensor inventory (types, shapes, bytes; blk.N grouped).
use std::collections::BTreeMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: gguf-ls <file.gguf> [--kv] [--all]");
    let t0 = std::time::Instant::now();
    let g = fe_gguf::Gguf::open(path)?;
    eprintln!("opened {} shard(s), {} tensors, {} kv in {:.1} ms", g.shards.len(), g.tensors.len(), g.kv.len(),
        t0.elapsed().as_secs_f64() * 1e3);
    if args.iter().any(|a| a == "--kv") {
        for (k, v) in &g.kv {
            println!("{k} = {}", v.short());
        }
    }
    if args.iter().any(|a| a == "--all") {
        for t in &g.tensors {
            println!("{:8} {:>20} {:>12} s{} {}", t.ty.name(), format!("{:?}", t.dims), t.bytes, t.shard, t.name);
        }
        return Ok(());
    }
    // group blk.<n>. names
    let mut groups: BTreeMap<(String, &str, String), (usize, u64, Vec<usize>)> = BTreeMap::new();
    let mut by_type: BTreeMap<&str, u64> = BTreeMap::new();
    for t in &g.tensors {
        let (key, layer) = if let Some(rest) = t.name.strip_prefix("blk.") {
            let dot = rest.find('.').unwrap();
            (format!("blk.N.{}", &rest[dot + 1..]), rest[..dot].parse::<usize>().ok())
        } else {
            (t.name.clone(), None)
        };
        let e = groups.entry((key, t.ty.name(), format!("{:?}", t.dims))).or_default();
        e.0 += 1;
        e.1 += t.bytes;
        if let Some(l) = layer {
            e.2.push(l);
        }
        *by_type.entry(t.ty.name()).or_default() += t.bytes;
    }
    for ((k, ty, dims), (n, bytes, _)) in &groups {
        println!("{n:3} {ty:8} {dims:22} {:9.1} MiB  {k}", *bytes as f64 / 1048576.0);
    }
    println!("--- bytes by type");
    for (ty, b) in &by_type {
        println!("{ty:8} {:9.2} GiB", *b as f64 / (1u64 << 30) as f64);
    }
    Ok(())
}
