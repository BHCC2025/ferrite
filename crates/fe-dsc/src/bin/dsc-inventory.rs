//! Print the deepseek41 model inventory (metadata, layer roles, tensor groups, spec check, placement plan) without
//! allocating any GPU memory: dsc-inventory [model.gguf (first shard)] [--partial] [--layers] [--list <prefix>]
//! Model path: argument, DSC_MODEL, or the Coder folder.
use anyhow::{bail, Result};
use fe_dsc::{config::DscConfig, plan::Plan, tensors};
use std::collections::BTreeMap;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let partial = a.iter().any(|x| x == "--partial");
    let layers = a.iter().any(|x| x == "--layers");
    let list = a.iter().position(|x| x == "--list").and_then(|i| a.get(i + 1)).cloned();
    let path = a.iter().find(|x| !x.starts_with("--") && Some(*x) != list.as_ref()).cloned()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(fe_dsc::model_path);
    let eng = fe_dsc::engram_path(&path);
    let g = fe_dsc::open_model(&path, eng.as_deref(), partial)?;
    if let Some(pre) = list {
        for t in g.tensors.iter().filter(|t| t.name.starts_with(&pre)) {
            println!("{} {} {:?} {} B shard {}", t.name, t.ty.name(), t.dims, t.bytes, t.shard);
        }
        return Ok(());
    }
    let c = DscConfig::from_gguf(&g)?;
    println!("model {} ({} shards, {} tensors, {:.2} GB){}", path.display(), g.shards.len(), g.tensors.len(),
        g.total_tensor_bytes() as f64 / 1e9, eng.as_ref().map(|e| format!(", engram {}", e.display())).unwrap_or_default());
    if !g.missing.is_empty() {
        println!("  NOT YET PRESENT (download in progress): {} items", g.missing.len());
    }
    println!("config: layers {} hidden {} vocab {} heads {} x {} (kv heads {}, rope {} dims) q_lora {} o_lora {} x {} groups window {}",
        c.n_layer, c.n_embd, c.n_vocab, c.n_head, c.head_dim, c.n_head_kv, c.n_rot, c.q_lora, c.o_lora, c.o_groups, c.window);
    println!("  rope base {} compress base {} yarn x{} orig {} beta {}/{}  eps {:e}  ctx {}", c.rope_base, c.rope_compress_base,
        c.yarn_factor, c.yarn_orig_ctx, c.yarn_beta_fast, c.yarn_beta_slow, c.eps, c.n_ctx_train);
    println!("  indexer {} heads x {} top-{}  candidate layer {} ({} blocks x {})  kv sources {:?}  index sources {:?}", c.idx_heads, c.idx_dim,
        c.idx_top_k, c.cand_layer, c.cand_blocks, c.cand_block, c.kv_sources, c.index_sources);
    println!("  hc {} streams, sinkhorn {} iters eps {:e}  experts {} top-{} + {} shared, ffn {}, gating {}, scale {} norm {} hash layers {}",
        c.hc, c.hc_iters, c.hc_eps, c.n_expert, c.n_expert_used, c.n_expert_shared, c.n_ff_exp, c.gating_func, c.weights_scale,
        c.weights_norm, c.hash_layers);
    let uniq = |v: &[f32]| { let mut u: Vec<f32> = v.to_vec(); u.dedup(); u };
    println!("  swiglu clamps routed {:?} shared {:?}", uniq(&c.clamp_exp), uniq(&c.clamp_shexp));
    println!("  engram layers {:?}: {}-gram, {} heads/order = {} columns x {} = {} wide; head offsets {}..{}; pad {}; token map {} -> {} classes",
        c.engram_layers, c.ngram, c.heads_per_ngram, c.engram_cols(), c.ple_dim, c.engram_width(), c.head_off[0],
        c.head_off[c.head_off.len() - 1], c.engram_pad, c.token_map.len(), c.token_map.iter().max().unwrap() + 1);
    if layers {
        for l in 0..c.n_layer {
            let r = c.role(l);
            println!("  layer {l:>2}: ratio {} kv_src {} idx_src {} kv_from {:?} sel_from {:?} cand_src {} cand_masked {} engram {}",
                r.ratio, r.kv_source as u8, r.index_source as u8, r.kv_from, r.sel_from, r.cand_source as u8, r.cand_masked as u8, r.engram as u8);
        }
    }
    // tensor groups by (suffix, type)
    let mut groups: BTreeMap<(String, &str), (usize, u64)> = BTreeMap::new();
    for t in &g.tensors {
        let kind = if t.name.starts_with("blk.") { t.name.splitn(3, '.').nth(2).unwrap_or("").to_string() } else { t.name.clone() };
        let e = groups.entry((kind, t.ty.name())).or_default();
        e.0 += 1;
        e.1 += t.bytes;
    }
    println!("tensor groups (kind, type, count, MiB):");
    for ((k, ty), (n, b)) in &groups {
        println!("  {k:<34} {ty:<8} x{n:<3} {:>10.1}", *b as f64 / 1048576.0);
    }
    let rep = tensors::check(&g, &c);
    println!("spec check: {} tensors matched, {} errors, {} pending, {} unexpected", rep.checked, rep.errors.len(), rep.pending.len(), rep.unexpected.len());
    for e in rep.errors.iter().take(20) {
        println!("  ERROR {e}");
    }
    for e in rep.unexpected.iter().take(20) {
        println!("  unexpected {e}");
    }
    let (ctx, lanes) = (131072usize, 4usize);
    let p = Plan::new(&c, &g, 16311 << 20, ctx, lanes, 2048);
    p.print(&c, 16311 << 20, ctx, lanes);
    if !rep.errors.is_empty() {
        bail!("{} spec errors", rep.errors.len());
    }
    Ok(())
}
