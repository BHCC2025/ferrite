//! The CPU oracle driver (CPU only, no GPU; the f32 reference the engine is checked against):
//!   dsc-ref ppl <text file> [ntok]            mean NLL / perplexity of the text (BOS first)
//!   dsc-ref gen <prompt text> [n]             greedy completion
//!   dsc-ref logits <text file> [ntok] <out>   last-row top-20 and the whole last-row logits (f32 LE) to <out>
//!   dsc-ref trace <list file> <out.bin> [ntok] [start] [end]  routing trace of each file in the list (one path a line); only
//!                                             lines start..end (default 0..all) are run, seq numbers stay the line index
//!   dsc-ref kl <text file> [ntok] [out prefix]  f32 oracle vs q8_1-activation oracle on the same tokens: KL, top-1, PPL (the
//!                                             noise floor of the engine numerics); the prefix saves both logit sets (f32 LE rows)
//!   dsc-ref copy <text file> <npass> [reps]   copy/needle gate: BOS + passage of npass tokens repeated (default twice); PPL of
//!                                             the later copies, in total and past the 128-token window (gate < 1.5)
//! Env: ORACLE_Q8=1 q8_1 activations (engine numerics) in every command; ORACLE_HC=own|first|ffnprev|headmean|head0|combT
//! hc-rule ablations; ORACLE_NOQUANT=1 no FP8/FP4 KV grids.
//!   dsc-ref analyze <trace.bin> [<trace.bin> ...]   resident-set hit rate vs slots, balance
//! Model path: DSC_MODEL (first shard), default the folder of scripts/fetch-deepseek.sh.
use anyhow::{bail, Context, Result};
use fe_dsc::{config::DscConfig, refmodel::{logprob, Oracle}, tok};

fn open() -> Result<(fe_gguf::Gguf, DscConfig)> {
    let path = fe_dsc::model_path();
    let eng = fe_dsc::engram_path(&path);
    let g = fe_dsc::open_model(&path, eng.as_deref(), false)?;
    let c = DscConfig::from_gguf(&g)?;
    Ok((g, c))
}

fn text_ids(g: &fe_gguf::Gguf, c: &DscConfig, text: &str, ntok: usize) -> Result<Vec<u32>> {
    let t = tok::build(g, c)?;
    let mut ids = tok::encode(&t, text)?;
    if ids.first() != Some(&0) {
        ids.insert(0, 0); // BOS unless the text (a rendered chat template) already starts with it
    }
    ids.truncate(ntok);
    Ok(ids)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("ppl") => {
            let (g, c) = open()?;
            let text = std::fs::read_to_string(&a[1])?;
            let ntok = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
            let ids = text_ids(&g, &c, &text, ntok)?;
            let mut o = Oracle::new(&g, &c, ids.len() + 8)?;
            let rows: Vec<usize> = (0..ids.len() - 1).collect();
            let t0 = std::time::Instant::now();
            let lg = o.forward(&ids, &rows)?;
            let nll: f64 = rows.iter().map(|&i| -logprob(&lg[i], ids[i + 1] as usize)).sum::<f64>() / rows.len() as f64;
            println!("tokens {} mean NLL {:.4} perplexity {:.3} ({:.1}s)", ids.len(), nll, nll.exp(), t0.elapsed().as_secs_f32());
        }
        Some("gen") => {
            let (g, c) = open()?;
            let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
            let t = tok::build(&g, &c)?;
            let mut ids = text_ids(&g, &c, &a[1], 4096)?;
            let mut o = Oracle::new(&g, &c, ids.len() + n + 8)?;
            let mut out = vec![];
            let mut feed = ids.clone();
            for _ in 0..n {
                let lg = o.forward(&feed, &[feed.len() - 1])?;
                let next = lg[0].iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0 as u32;
                out.push(next);
                ids.push(next);
                feed = vec![next];
                eprintln!("tok {next}");
            }
            println!("{}{}", a[1], t.decode(&out, false).map_err(|e| anyhow::anyhow!("{e}"))?);
        }
        Some("logits") => {
            let (g, c) = open()?;
            let text = std::fs::read_to_string(&a[1])?;
            let ntok = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
            let ids = text_ids(&g, &c, &text, ntok)?;
            let mut o = Oracle::new(&g, &c, ids.len() + 8)?;
            let lg = o.forward(&ids, &[ids.len() - 1])?;
            let mut top: Vec<(usize, f32)> = lg[0].iter().cloned().enumerate().collect();
            top.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
            for (i, v) in &top[..20] {
                println!("{i} {v:.4}");
            }
            if let Some(out) = a.get(3) {
                std::fs::write(out, lg[0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            }
        }
        Some("trace") => {
            let (g, c) = open()?;
            let list = std::fs::read_to_string(&a[1])?;
            let ntok = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
            let start: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let end: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
            let mut w: Vec<u8> = vec![];
            for (seq, path) in list.lines().filter(|l| !l.trim().is_empty()).enumerate() {
                if seq < start || seq >= end {
                    continue;
                }
                let text = std::fs::read_to_string(path).with_context(|| path.to_string())?;
                let ids = text_ids(&g, &c, &text, ntok)?;
                let mut o = Oracle::new(&g, &c, ids.len() + 8)?;
                let t0 = std::time::Instant::now();
                o.forward(&ids, &[])?;
                for r in &o.trace {
                    w.extend((seq as u32).to_le_bytes());
                    w.push(r.layer);
                    w.extend(r.row.to_le_bytes());
                    for k in 0..6 {
                        w.extend(r.ids[k].to_le_bytes());
                    }
                    for k in 0..6 {
                        w.extend(r.w[k].to_le_bytes());
                    }
                }
                std::fs::write(&a[2], &w)?; // checkpoint after every prompt
                eprintln!("trace seq {seq} {path}: {} tokens, {:.1}s", ids.len(), t0.elapsed().as_secs_f32());
            }
        }
        Some("kl") => {
            let (g, c) = open()?;
            let text = std::fs::read_to_string(&a[1])?;
            let ntok = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
            let ids = text_ids(&g, &c, &text, ntok)?;
            let rows: Vec<usize> = (0..ids.len() - 1).collect();
            let mut runs = vec![];
            for q8 in [false, true] {
                let mut o = Oracle::new(&g, &c, ids.len() + 8)?;
                o.q8 = q8;
                let t0 = std::time::Instant::now();
                runs.push(o.forward(&ids, &rows)?);
                eprintln!("kl: q8={q8} done in {:.1}s", t0.elapsed().as_secs_f32());
            }
            if let Some(pre) = a.get(3) {
                for (nm, lg) in [("f32", &runs[0]), ("q8", &runs[1])] {
                    std::fs::write(format!("{pre}.{nm}.bin"), lg.iter().flat_map(|r| r.iter().flat_map(|v| v.to_le_bytes())).collect::<Vec<u8>>())?;
                }
            }
            let lsm = |l: &[f32]| -> Vec<f64> {
                let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let z: f64 = l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
                l.iter().map(|&v| v as f64 - m - z).collect()
            };
            let (mut kls, mut agree, mut nll_f, mut nll_q) = (vec![], 0usize, 0f64, 0f64);
            for &i in &rows {
                let (lf, lq) = (lsm(&runs[0][i]), lsm(&runs[1][i]));
                kls.push(lf.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum::<f64>());
                let am = |l: &[f32]| l.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
                agree += (am(&runs[0][i]) == am(&runs[1][i])) as usize;
                nll_f -= lf[ids[i + 1] as usize];
                nll_q -= lq[ids[i + 1] as usize];
            }
            let n = rows.len() as f64;
            let mut sorted = kls.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("tokens {} rows {}: q8 vs f32 oracle  mean KL(f32||q8) {:.5}  median {:.5}  p99 {:.5}  max {:.5}  top-1 agreement {:.4}",
                ids.len(), rows.len(), kls.iter().sum::<f64>() / n, sorted[sorted.len() / 2], sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)], sorted[sorted.len() - 1], agree as f64 / n);
            println!("PPL f32 {:.3}  q8 {:.3}  (mean NLL {:.4} / {:.4})", (nll_f / n).exp(), (nll_q / n).exp(), nll_f / n, nll_q / n);
        }
        Some("copy") => {
            let (g, c) = open()?;
            let text = std::fs::read_to_string(&a[1])?;
            let np: usize = a[2].parse()?;
            let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
            let base = text_ids(&g, &c, &text, np + 1)?;
            if base.len() < np + 1 {
                bail!("text has only {} tokens, need {}", base.len() - 1, np);
            }
            let mut ids = base.clone();
            for _ in 1..reps {
                ids.extend_from_slice(&base[1..]);
            }
            let rows: Vec<usize> = (0..ids.len() - 1).collect();
            let mut o = Oracle::new(&g, &c, ids.len() + 8)?;
            let t0 = std::time::Instant::now();
            let lg = o.forward(&ids, &rows)?;
            let nll = |p: usize| -logprob(&lg[p - 1], ids[p] as usize); // NLL of the token at position p
            let ppl = |lo: usize, hi: usize| ((lo..hi).map(nll).sum::<f64>() / (hi - lo) as f64).exp();
            println!("copy test: passage {np} tokens x{reps} (total {} tokens, window {}, q8={}) {:.0}s", ids.len(), c.window, o.q8, t0.elapsed().as_secs_f32());
            println!("  copy 1 PPL {:.3}", ppl(1, 1 + np));
            let mut worst = 0f64;
            for k in 1..reps {
                let s0 = 1 + k * np;
                let all = ppl(s0, s0 + np);
                let past = ppl(s0 + c.window.min(np - 1), s0 + np);
                worst = worst.max(past);
                println!("  copy {} PPL {:.3} total, {:.3} past the window (copy offset >= {})", k + 1, all, past, c.window);
            }
            println!("  GATE (past-window PPL < 1.5): {}", if worst < 1.5 { "PASS" } else { "FAIL" });
        }
        Some("analyze") => analyze(&a[1..])?,
        _ => bail!("usage: dsc-ref ppl|gen|logits|trace|analyze (see the file header)"),
    }
    Ok(())
}

const REC: usize = 4 + 1 + 4 + 12 + 24;

struct Rec {
    seq: u32,
    layer: usize,
    ids: [usize; 6],
    w: [f32; 6],
}

fn read_trace(path: &str) -> Result<Vec<Rec>> {
    let b = std::fs::read(path)?;
    Ok(b.chunks_exact(REC).map(|r| {
        let mut ids = [0usize; 6];
        let mut w = [0f32; 6];
        for k in 0..6 {
            ids[k] = u16::from_le_bytes([r[9 + 2 * k], r[10 + 2 * k]]) as usize;
            w[k] = f32::from_le_bytes(r[21 + 4 * k..25 + 4 * k].try_into().unwrap());
        }
        Rec { seq: u32::from_le_bytes(r[0..4].try_into().unwrap()), layer: r[4] as usize, ids, w }
    }).collect())
}

/// Resident-set hit rate. The resident set is chosen from the TRAIN picks only and scored on the TEST picks. Train/test: files
/// given as `test:<file>` are test only; every other file is split by sequence parity (even = train, odd = test is NOT used
/// when a `test:` file exists: then all of the plain files are train). Two slot models, both for a budget of S "largest
/// expert" slots (S x the largest expert = 9.76 MiB, i.e. S = 2,240 is the 21.4 GiB of the two cards' expert arenas):
///   fixed slots: exactly S experts, picked by count (layer sizes ignored);
///   byte arena : the experts with the most picks per byte until S x largest bytes are spent (more, smaller experts fit).
/// Card balance: per layer the resident experts are dealt (by train heat) to the lighter of the two cards; NUMA balance: the
/// non-resident experts likewise to the lighter of the two nodes; both are scored on the test picks (the share of the test
/// picks of the resident / missing experts each side serves). `worst imb` = the largest per-layer |a - b| / (a + b).
fn analyze(paths: &[String]) -> Result<()> {
    let (g, c) = open()?;
    let mut train = vec![];
    let mut test = vec![];
    let have_test = paths.iter().any(|p| p.starts_with("test:"));
    for p in paths {
        if let Some(f) = p.strip_prefix("test:") {
            test.extend(read_trace(f)?);
        } else {
            for r in read_trace(p)? {
                if have_test || r.seq % 2 == 0 { train.push(r) } else { test.push(r) }
            }
        }
    }
    let ebytes: Vec<usize> = (0..c.n_layer).map(|l| ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].iter().map(|k| g.tensor(&format!("blk.{l}.{k}.weight")).unwrap().slice_bytes()).sum()).collect();
    let ne = c.n_expert;
    let tally = |recs: &[Rec]| -> (Vec<f64>, f64) {
        let mut cnt = vec![0f64; c.n_layer * ne];
        let mut n = 0f64;
        for r in recs {
            for k in 0..6 {
                cnt[r.layer * ne + r.ids[k]] += 1.0;
                n += 1.0;
            }
        }
        (cnt, n)
    };
    let (cnt_tr, ntr) = tally(&train);
    let (cnt_te, nte) = tally(&test);
    let maxb = *ebytes.iter().max().unwrap();
    let mib = |b: usize| b as f64 / 1048576.0;
    println!("train: {} rows ({} picks), test: {} rows ({} picks); {} layers x {} experts; expert bytes {:.2}..{:.2} MiB (largest {:.2})",
        train.len() / c.n_layer.max(1), ntr, test.len() / c.n_layer.max(1), nte, c.n_layer, ne, mib(*ebytes.iter().min().unwrap()), mib(maxb), mib(maxb));
    let share = {
        let mut v: Vec<f64> = cnt_tr.clone();
        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
        v[..v.len() / 10].iter().sum::<f64>() / ntr
    };
    println!("top 10% of (layer, expert) pairs carry {:.1}% of train picks", 100.0 * share);
    let mut by_count: Vec<usize> = (0..c.n_layer * ne).collect();
    by_count.sort_by(|&a, &b| cnt_tr[b].partial_cmp(&cnt_tr[a]).unwrap().then(a.cmp(&b)));
    let mut by_byte: Vec<usize> = (0..c.n_layer * ne).collect();
    by_byte.sort_by(|&a, &b| (cnt_tr[b] / ebytes[b / ne] as f64).partial_cmp(&(cnt_tr[a] / ebytes[a / ne] as f64)).unwrap().then(a.cmp(&b)));
    println!("S = slots of the largest expert (S x {:.2} MiB); n = experts the byte arena holds", mib(maxb));
    println!("     S    GiB | fixed: hit(train) hit(test) | byte arena:    n  hit(train) hit(test) worst-layer | cards c0/c1 (worst imb) | nodes n0/n1 of misses (worst imb)");
    for slots in [500usize, 1000, 1500, 2000, 2240, 2600, 3000, 3500, 4000] {
        let mut resf = vec![false; c.n_layer * ne];
        for &i in by_count.iter().take(slots) {
            resf[i] = true;
        }
        let budget = slots * maxb;
        let mut used = 0usize;
        let mut resb = vec![false; c.n_layer * ne];
        let mut nb = 0usize;
        for &i in &by_byte {
            let b = ebytes[i / ne];
            if used + b > budget { continue; }
            used += b;
            resb[i] = true;
            nb += 1;
        }
        let hit = |res: &[bool], cnt: &[f64], n: f64| (0..cnt.len()).filter(|&i| res[i]).map(|i| cnt[i]).sum::<f64>() / n;
        let worst = (0..c.n_layer).map(|l| {
            let tot: f64 = (0..ne).map(|e| cnt_te[l * ne + e]).sum();
            (0..ne).filter(|&e| resb[l * ne + e]).map(|e| cnt_te[l * ne + e]).sum::<f64>() / tot.max(1.0)
        }).fold(1f64, f64::min);
        // balance of a set: deal the members of each layer by descending train heat to the lighter side (by train heat), score on test
        let split = |member: &dyn Fn(usize) -> bool| -> (f64, f64, f64) {
            let (mut s0, mut s1, mut wi) = (0f64, 0f64, 0f64);
            for l in 0..c.n_layer {
                let mut es: Vec<usize> = (0..ne).filter(|&e| member(l * ne + e)).collect();
                es.sort_by(|&a, &b| cnt_tr[l * ne + b].partial_cmp(&cnt_tr[l * ne + a]).unwrap());
                let (mut h0, mut h1, mut a0, mut a1) = (0f64, 0f64, 0f64, 0f64);
                for e in es {
                    if h0 <= h1 { h0 += cnt_tr[l * ne + e]; a0 += cnt_te[l * ne + e] } else { h1 += cnt_tr[l * ne + e]; a1 += cnt_te[l * ne + e] }
                }
                s0 += a0;
                s1 += a1;
                if a0 + a1 > 0.0 { wi = wi.max((a0 - a1).abs() / (a0 + a1)); }
            }
            (s0 / nte, s1 / nte, wi)
        };
        let (c0, c1, cw) = split(&|i| resb[i]);
        let (n0, n1, nw) = split(&|i| !resb[i]);
        println!("{slots:>6} {:>6.1} | {:>17.3} {:>9.3} | {:>19} {:>9.3} {:>9.3} {:>9.3} | {:.3}/{:.3} ({:.2}) | {:.3}/{:.3} ({:.2})",
            mib(budget) / 1024.0, hit(&resf, &cnt_tr, ntr), hit(&resf, &cnt_te, nte), nb, hit(&resb, &cnt_tr, ntr), hit(&resb, &cnt_te, nte), worst, c0, c1, cw, n0, n1, nw);
    }
    Ok(())
}
