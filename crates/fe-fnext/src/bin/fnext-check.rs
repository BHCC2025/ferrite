//! Correctness gate against llama.cpp reference dumps of named tensors (directory: FNEXT_REF, default `ref`):
//!   fnext-check dump <ref dir: a|b|c>   every dumped tensor: relative error per tensor, worst layer per name
//!   fnext-check greedy <a|b>            teacher-forced over the reference's 64 greedy tokens: argmax agreement,
//!                                       top-20 KL, free-running greedy match length
use anyhow::{anyhow, Context, Result};
use fe_fnext::engine::Model;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Reference dumps (named tensors, prompts, greedy streams): FNEXT_REF, default `ref`.
static REF: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| std::env::var("FNEXT_REF").unwrap_or_else(|_| "ref".into()));

fn prompt_tokens(name: &str) -> Result<Vec<u32>> {
    let f = match name {
        "a" => "a_short",
        "b" => "b_code600",
        "c" => "c_long5000",
        _ => name,
    };
    let v: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/prompts/{f}.tokens.json", &*REF))?)?;
    Ok(v.as_array().ok_or_else(|| anyhow!("tokens json"))?.iter().map(|x| x.as_u64().unwrap() as u32).collect())
}

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("{}", p.display()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(String::as_str).unwrap_or("dump");
    let which = a.get(2).cloned().unwrap_or_else(|| "a".into());
    let ctx = 8192;
    let toks = prompt_tokens(&which)?;
    let rows_max = std::env::var("FNEXT_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(640);
    let mut m = Model::load(&fe_fnext::default_model(), ctx, rows_max, 64)?;
    match mode {
        "dump" => dump_check(&mut m, &which, &toks),
        "greedy" => greedy_check(&mut m, &which, &toks),
        "truth" => truth(&mut m, &which, &toks),
        _ => Err(anyhow!("usage: fnext-check dump|greedy <a|b|c>")),
    }
}

fn dump_check(m: &mut Model, which: &str, toks: &[u32]) -> Result<()> {
    let dir = PathBuf::from(&*REF).join(which);
    let idx: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("index.json"))?)?;
    let meta = &idx["meta"];
    let ub = meta["n_ubatch"].as_u64().unwrap() as usize;
    let last_only = meta["dump_last_ubatch"].as_bool().unwrap_or(false);
    let n = toks.len();
    let n_logit = (meta["logit_rows"].as_u64().unwrap_or(0) as usize).max(0);
    // run the prompt in the reference's ubatches; dump the (last) ubatch
    let t0 = std::time::Instant::now();
    let mut p = 0;
    while p < n {
        let e = (p + ub).min(n);
        let last = e == n;
        m.dumps.on = !last_only || last;
        let nl = if last { if n_logit == 0 { (e - p).min(64) } else { n_logit } } else { 1 };
        m.forward(&toks[p..e], nl)?;
        p = e;
    }
    eprintln!("forward {} tokens in {:.1} s", n, t0.elapsed().as_secs_f32());
    let ub_pos0 = if last_only { (n - 1) / ub * ub } else { 0 };
    let mut worst: BTreeMap<String, (f64, i64, usize)> = BTreeMap::new();
    let mut missing: BTreeMap<String, usize> = BTreeMap::new();
    println!("{:<28} {:>5} {:>3} {:>11} {:>11} {:>9}", "tensor", "layer", "rep", "rel_err", "max_abs", "ref_rms");
    for t in idx["tensors"].as_array().unwrap() {
        let name = t["name"].as_str().unwrap();
        let layer = t["layer"].as_i64().unwrap();
        let rep = t["rep"].as_u64().unwrap() as u32;
        let ne: Vec<usize> = t["ne"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        // the reference scores QSA in chunks of 256 rows (LLAMA_QSA_CHUNK): rep k = rows 256k.. of the ubatch
        let qsa_chunked = name == "indexer_top_k";
        let my_rep = if qsa_chunked { 0 } else { rep };
        let Some(mine) = m.dumps.map.get(&(name.to_string(), layer as i32, my_rep)) else {
            *missing.entry(name.to_string()).or_default() += 1;
            continue;
        };
        let refv = read_f32(&dir.join(t["file"].as_str().unwrap()))?;
        let tax = t["token_axis"].as_i64().unwrap_or(-1);
        let row0 = t["row0_pos"].as_i64().unwrap_or(0);
        // per-row width and row offset into my dump (rows of the dumped ubatch)
        let (width, nrows) = if qsa_chunked {
            (ne[0], ne[1])
        } else if tax >= 0 {
            let w: usize = ne[..tax as usize].iter().product();
            (w, ne[tax as usize])
        } else {
            (refv.len(), 1)
        };
        let mine_rows = mine.len() / width.max(1);
        // my logits dump holds only the last rows: compare the matching tail
        let off_rows = if qsa_chunked { rep as usize * 256 } else if tax < 0 { 0 } else if name == "result_output" { mine_rows.saturating_sub(nrows) } else {
            (row0 as usize).saturating_sub(ub_pos0)
        };
        let (mut mine_s, refs): (Vec<f32>, &[f32]) = if tax >= 0 || qsa_chunked {
            if mine_rows < off_rows + nrows || mine.len() % width != 0 {
                println!("{:<28} {:>5} {:>3} shape mismatch: ref {:?}, mine {} values", name, layer, rep, ne, mine.len());
                continue;
            }
            (mine[off_rows * width..(off_rows + nrows) * width].to_vec(), &refv)
        } else {
            if mine.len() != refv.len() {
                println!("{:<28} {:>5} {:>3} shape mismatch: ref {:?}, mine {} values", name, layer, rep, ne, mine.len());
                continue;
            }
            (mine.clone(), &refv)
        };
        if t["dtype"] == "i32" {
            // index tensors: set agreement per row (top-k is unordered in ggml)
            let ri: Vec<i32> = refs.iter().map(|v| v.to_bits() as i32).collect();
            let mi: Vec<i32> = mine_s.iter().map(|v| v.to_bits() as i32).collect();
            let w = width.max(1);
            let mut agree = 0;
            for (rr, mr) in ri.chunks(w).zip(mi.chunks(w)) {
                let mut a = rr.to_vec();
                let mut b = mr.to_vec();
                a.sort();
                b.sort();
                agree += a.iter().filter(|x| b.binary_search(x).is_ok()).count();
            }
            let frac = agree as f64 / ri.len() as f64;
            println!("{:<28} {:>5} {:>3} {:>10.4}% match", name, layer, rep, 100.0 * frac);
            let e = worst.entry(name.to_string()).or_insert((0.0, layer, 0));
            if 1.0 - frac > e.0 { *e = (1.0 - frac, layer, e.2); }
            e.2 += 1;
            continue;
        }
        let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
        for (x, y) in mine_s.iter_mut().zip(refs) {
            let d = (*x - *y) as f64;
            num += d * d;
            den += (*y as f64) * (*y as f64);
            mx = mx.max(d.abs());
        }
        let rel = (num / den.max(1e-30)).sqrt();
        let rms = (den / refs.len() as f64).sqrt();
        println!("{:<28} {:>5} {:>3} {:>11.3e} {:>11.3e} {:>9.3e}", name, layer, rep, rel, mx, rms);
        let e = worst.entry(name.to_string()).or_insert((0.0, layer, 0));
        if rel > e.0 {
            *e = (rel, layer, e.2);
        }
        e.2 += 1;
    }
    println!("\nworst per tensor name (relative L2 error; index tensors: 1 - set match):");
    for (k, (e, l, c)) in &worst {
        println!("  {:<28} {:>10.3e}  layer {:>3}  ({} compared)", k, e, l, c);
    }
    if !missing.is_empty() {
        println!("not produced by fnext (skipped): {:?}", missing);
    }
    Ok(())
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
    let ls = s.ln() + m;
    v.iter().map(|&x| x as f64 - ls).collect()
}

fn greedy_check(m: &mut Model, which: &str, toks: &[u32]) -> Result<()> {
    let g: Value = serde_json::from_str(&String::from_utf8_lossy(&std::fs::read(format!("{}/greedy_{which}.json", &*REF))?))?;
    let steps = g["steps"].as_array().unwrap();
    let ref_toks: Vec<u32> = steps.iter().map(|s| s["token"].as_u64().unwrap() as u32).collect();
    let rows_max = m.rows_max;
    // prompt
    let mut p = 0;
    let mut logits = vec![];
    while p < toks.len() {
        let e = (p + rows_max).min(toks.len());
        logits = m.forward(&toks[p..e], 1)?;
        p = e;
    }
    let mut agree = 0;
    let mut kl_sum = 0f64;
    let mut kl_max = 0f64;
    let mut free_run = None;
    let mut maxdiff = 0f64;
    for (i, s) in steps.iter().enumerate() {
        let am = logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (j, &v)| if v > b.1 { (j, v) } else { b }).0 as u32;
        if am == ref_toks[i] {
            agree += 1;
        } else if free_run.is_none() {
            free_run = Some(i);
        }
        // KL over the reference's top-20 (both renormalised over those ids)
        let top = s["top"].as_array().unwrap();
        let ids: Vec<usize> = top.iter().map(|t| t["id"].as_u64().unwrap() as usize).collect();
        let rl: Vec<f32> = top.iter().map(|t| t["logit"].as_f64().unwrap() as f32).collect();
        let ml: Vec<f32> = ids.iter().map(|&j| logits[j]).collect();
        for (a, b) in rl.iter().zip(&ml) {
            maxdiff = maxdiff.max((a - b).abs() as f64);
        }
        let (lp, lq) = (log_softmax(&rl), log_softmax(&ml));
        let kl: f64 = lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum();
        kl_sum += kl;
        kl_max = kl_max.max(kl);
        println!("step {:>2}: ref {:>6} mine {:>6} {}  top20-KL {:.2e}  logit[ref] {:.3} vs {:.3}", i, ref_toks[i], am,
            if am == ref_toks[i] { "ok  " } else { "DIFF" }, kl, rl[0], logits[ref_toks[i] as usize]);
        if i + 1 < steps.len() {
            logits = m.forward(&ref_toks[i..i + 1], 1)?;
        }
    }
    let n = steps.len();
    println!("\n{which}: teacher-forced argmax agreement {agree}/{n}; top-20 KL mean {:.3e} max {:.3e}; max |logit diff| \
        over top-20 {:.3}; greedy stream identical for the first {} tokens", kl_sum / n as f64, kl_max, maxdiff,
        free_run.unwrap_or(n));
    Ok(())
}

/// Teacher-forced over the REF greedy stream with the f32 path; writes the logits of every step to
/// <FNEXT_REF>/truth_<which>.bin (steps x vocab f32): the reference for comparing quantized paths.
fn truth(m: &mut Model, which: &str, toks: &[u32]) -> Result<()> {
    let g: Value = serde_json::from_str(&String::from_utf8_lossy(&std::fs::read(format!("{}/greedy_{which}.json", &*REF))?))?;
    let rt: Vec<u32> = g["steps"].as_array().unwrap().iter().map(|s| s["token"].as_u64().unwrap() as u32).collect();
    let mut p = 0;
    let mut logits = vec![];
    while p < toks.len() {
        let e = (p + m.rows_max).min(toks.len());
        logits = m.forward(&toks[p..e], 1)?;
        p = e;
    }
    let mut out: Vec<u8> = vec![];
    for i in 0..rt.len() {
        for v in &logits {
            out.extend_from_slice(&v.to_le_bytes());
        }
        if i + 1 < rt.len() {
            logits = m.forward(&rt[i..i + 1], 1)?;
        }
    }
    std::fs::write(format!("{}/truth_{which}.bin", &*REF), out)?;
    println!("wrote truth_{which}.bin ({} steps)", rt.len());
    Ok(())
}
