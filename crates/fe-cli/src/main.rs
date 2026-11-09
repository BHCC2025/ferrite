//! ferrite command line.
//!
//! ```text
//! ferrite selftest
//! ferrite <model-dir> generate "<prompt>" [max_new]       raw text prompt (no chat template), greedy
//! ferrite <model-dir> logits <ids.txt> <out.bin> <skip> <rows>
//! ferrite <model-dir> bench [len,len,...] [decode_tokens] [streams]
//! ferrite <model-dir> sched-test [n_requests]
//! ferrite <model-dir> tree-test "<prompt>" [max_new] [nodes]   oracle verify trees vs plain greedy
//! ```
//! `logits` follows the KL harness contract: per sequence (one comma-separated line of ids), positions
//! 0..skip go through the prompt path, then positions skip..skip+rows-1 are decoded one token at a time;
//! the bf16 logits of those decode rows are written as [nseq][rows][vocab]. Sequences run FE_BATCH at a
//! time (default 20) as parallel slots. Config/tokenizer come from the model dir or $FE_CONFIG_DIR.
use anyhow::{bail, ensure, Context as _, Result};
use fe_cuda::{Context, Module, Stream};
use fe_model::config::Config;
use fe_model::model::Engine;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn load_engine(model: &Path, n_slots: usize, max_ctx: usize, max_rows: usize) -> Result<(std::sync::Arc<Context>, Engine)> {
    load_engine_spec(model, n_slots, max_ctx, max_rows, 0)
}

fn load_engine_spec(model: &Path, n_slots: usize, max_ctx: usize, max_rows: usize, depth: usize)
    -> Result<(std::sync::Arc<Context>, Engine)> {
    // whole model, or one TP rank (FE_TP_WORLD=2 FE_TP_RANK=0|1 [FE_TP_ADDR]; both ranks run this command)
    fe_model::tp::load_engine(model, n_slots, max_ctx, max_rows, depth, false)
}

/// TP rank 1 behind rank 0's scheduler / server: load (sizes from rank 0) and replay its engine calls.
fn cmd_follow(model: &Path) -> Result<()> {
    let (_ctx, mut e) = fe_model::tp::load_engine(model, 1, 1, 1, 0, true)?;
    // FE_DRAFT=dflash (with the TP split, FE_DF_TP != 0): this rank's half of the DFlash2 drafter
    let mut df = if std::env::var("FE_DRAFT").is_ok_and(|v| v == "dflash")
        && std::env::var("FE_DF_TP").map(|v| v != "0").unwrap_or(true) {
        let o = fe_model::sched::SchedOptions { n_slots: e.n_slots, prefill_chunk: e.tp_sizes().0, ..Default::default() };
        Some(fe_dflash::factory()(&e, &o)?)
    } else {
        None
    };
    let n = e.follow_with(df.as_mut())?;
    eprintln!("ferrite follow: rank 0 stopped after {n} calls");
    Ok(())
}

fn tokenizer_path(model: &Path) -> PathBuf {
    let p = model.join("tokenizer.json");
    if p.exists() {
        return p;
    }
    Config::locate(model).with_file_name("tokenizer.json")
}

/// Greedy generation with MTP speculative decoding (single stream): absorb the prompt into the head, then
/// rounds of `depth` chained drafts verified by one target forward of depth + 1 rows.
fn cmd_generate_spec(model: &Path, prompt: &str, max_new: usize, depth: usize) -> Result<()> {
    use fe_model::model::{MtpRow, MtpSrc};
    let tok = tokenizers::Tokenizer::from_file(tokenizer_path(model)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = tok.encode(prompt, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let chunk = env_usize("FE_PREFILL_CHUNK", 2048);
    let (_ctx, mut e) = load_engine_spec(model, 1, ids.len() + max_new + depth + 8, chunk, depth)?;
    anyhow::ensure!(e.has_mtp(), "checkpoint has no MTP head");
    let n = ids.len();
    let t0 = Instant::now();
    // prefill, absorbing each chunk into the head (follow token = the next prompt token)
    let mut done = 0;
    let mut first = 0;
    while done < n {
        let end = (done + chunk).min(n);
        let last = end == n;
        let lr: Vec<usize> = if last { vec![end - done - 1] } else { vec![] };
        e.forward(&[(0, &ids[done..end])], &lr)?;
        if last {
            first = e.argmax(1)?[0];
        }
        let rows: Vec<MtpRow> = (done..end).map(|i| MtpRow { slot: 0, pos: i, token: if i + 1 < n { ids[i + 1] } else { first },
            src: MtpSrc::Tap(i - done) }).collect();
        let draw: Vec<usize> = if last { vec![rows.len() - 1] } else { vec![] };
        let d = e.mtp_forward(&rows, &draw)?;
        if last {
            done = end;
            let pf = t0.elapsed().as_secs_f64();
            eprintln!("prompt {n} tokens, prefill + absorb {pf:.2}s");
            let t = Instant::now();
            let mut out = vec![first];
            let mut pending = first;
            let mut drafts = vec![d[0]];
            // the head's output row of the newest absorbed position (the first chained draft reads it)
            let mut last_out = rows.len() - 1;
            let (mut rounds, mut accepted) = (0usize, 0usize);
            print!("{prompt}");
            let mut printed = String::new();
            while out.len() < max_new && pending != e.cfg.eos {
                // chain the rest of the drafts
                let p = e.slot_len[0];
                while drafts.len() < depth {
                    let k = drafts.len();
                    let src = if k == 1 { MtpSrc::Out(last_out) } else { MtpSrc::Out(0) };
                    let nd = e.mtp_forward(&[MtpRow { slot: 0, pos: p + k - 1, token: drafts[k - 1], src }], &[0])?;
                    drafts.push(nd[0]);
                }
                // verify [pending, d1..dD]
                let mut rows = vec![pending];
                rows.extend_from_slice(&drafts);
                let lr: Vec<usize> = (0..rows.len()).collect();
                e.forward_ex(&[(0, &rows)], &lr, true, rows.len())?;
                let y = e.argmax(rows.len())?;
                let mut a = 0;
                while a < depth && drafts[a] == y[a] {
                    a += 1;
                }
                e.commit(0, p, rows.len(), a + 1)?;
                rounds += 1;
                accepted += a;
                let mut follow: Vec<u32> = drafts[..a].to_vec();
                follow.push(y[a]);
                for &t in &follow {
                    out.push(t);
                }
                pending = y[a];
                // absorb the kept rows into the head, draft the next round's first token
                let rows: Vec<MtpRow> = (0..=a).map(|i| MtpRow { slot: 0, pos: p + i, token: follow[i], src: MtpSrc::Tap(i) })
                    .collect();
                let d = e.mtp_forward(&rows, &[a])?;
                drafts = vec![d[0]];
                last_out = a;
                let text = tok.decode(&out, false).map_err(|e| anyhow::anyhow!("{e}"))?;
                print!("{}", &text[printed.len().min(text.len())..]);
                std::io::stdout().flush()?;
                printed = text;
            }
            println!();
            let dt = t.elapsed().as_secs_f64();
            eprintln!("decode {} tok in {:.2}s ({:.2} tok/s), {rounds} rounds, {:.2} accepted drafts/round (depth {depth})",
                out.len(), dt, out.len() as f64 / dt, accepted as f64 / rounds.max(1) as f64);
            return Ok(());
        }
        done = end;
    }
    Ok(())
}


fn cmd_generate(model: &Path, prompt: &str, max_new: usize) -> Result<()> {
    if std::env::var("FE_DRAFT").is_ok_and(|v| v == "dflash") {
        // DFlash2 block drafter (crates/fe-dflash); FE_DRAFT_DIR selects the checkpoint
        return fe_dflash::generate(model, prompt, max_new, false);
    }
    let depth = env_usize("FE_SPEC", 0);
    if depth > 0 {
        return cmd_generate_spec(model, prompt, max_new, depth);
    }
    let tok = tokenizers::Tokenizer::from_file(tokenizer_path(model)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = tok.encode(prompt, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let chunk = env_usize("FE_PREFILL_CHUNK", 2048);
    let (_ctx, mut e) = load_engine(model, 1, ids.len() + max_new + 8, chunk)?;
    eprintln!("prompt: {} tokens", ids.len());
    let t = Instant::now();
    let n = ids.len();
    let mut done = 0;
    while done < n {
        let end = (done + chunk).min(n);
        let want: &[usize] = if end == n { &[0] } else { &[] };
        let rows = end - done;
        let lr: Vec<usize> = want.iter().map(|_| rows - 1).collect();
        e.forward(&[(0, &ids[done..end])], &lr)?;
        done = end;
    }
    let pf = t.elapsed().as_secs_f64();
    let mut next = e.argmax(1)?[0];
    let mut out = vec![next];
    print!("{prompt}");
    std::io::stdout().flush()?;
    let t = Instant::now();
    let mut printed = String::new();
    for _ in 1..max_new {
        if next == e.cfg.eos {
            break;
        }
        let text = tok.decode(&out, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        print!("{}", &text[printed.len().min(text.len())..]);
        std::io::stdout().flush()?;
        printed = text;
        e.forward(&[(0, &[next])], &[0])?;
        next = e.argmax(1)?[0];
        out.push(next);
    }
    let text = tok.decode(&out, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", &text[printed.len().min(text.len())..]);
    let dt = t.elapsed().as_secs_f64();
    eprintln!("\nprefill {n} tok in {:.2}s ({:.0} tok/s); decode {} tok in {:.2}s ({:.2} tok/s)", pf, n as f64 / pf,
        out.len() - 1, dt, (out.len() - 1) as f64 / dt);
    Ok(())
}

/// Gate for verify trees (attention mask, RoPE by depth, commit_tree's gathers, the GDN tree path): plain greedy
/// decoding in slot 1 gives the reference; slot 0 runs rounds that verify a tree of `nodes` drafted rows holding
/// the reference continuation (a random length) among random distractor branches, walk it and commit the kept path.
/// Every token must match the reference, except near-ties: where the tree's target logits for its pick and the
/// reference token are within one bf16 step (multi-row verify vs one-row decode rounding), the tree's pick is kept
/// and the reference is redone from there. FE_TREE_LAYOUT=prefix puts the true path in rows 1..k (a chain for the
/// GDN layers); the default interleaves it with the distractors. FE_TREE_DIFF=1 also keeps the reference's logits
/// and reports how far the kept rows' logits are from them (max |difference| per round: the interleaved layout
/// should match the prefix one, any mask or position bug shows up far above it).
fn cmd_tree_test(model: &Path, prompt: &str, max_new: usize, nodes: usize) -> Result<()> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_path(model)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = tok.encode(prompt, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    let chunk = env_usize("FE_PREFILL_CHUNK", 2048);
    let hist = env_usize("FE_SPEC", nodes + 1).max(nodes + 1);
    let (_ctx, mut e) = load_engine_spec(model, 2, ids.len() + max_new + nodes + 8, chunk, hist)?;
    // chain: the prefix layout verified as a plain chain (no tree flag; the baseline for the round timings)
    let layout = std::env::var("FE_TREE_LAYOUT").unwrap_or_default();
    let chain = layout == "chain";
    let prefix = chain || layout == "prefix";
    let (mut t_ver, mut t_com) = (0.0f64, 0.0f64);
    // prompt (+ tokens so far) into `slot`; returns the next greedy token
    let prefill = |e: &mut Engine, slot: usize, toks: &[u32]| -> Result<u32> {
        e.reset_slot(slot)?;
        let mut done = 0;
        while done < toks.len() {
            let end = (done + chunk).min(toks.len());
            let lr: Vec<usize> = if end == toks.len() { vec![end - done - 1] } else { vec![] };
            e.forward(&[(slot, &toks[done..end])], &lr)?;
            done = end;
        }
        Ok(e.argmax(1)?[0])
    };
    // the plain greedy continuation of `toks` in slot 1 until the output holds max_new tokens
    let mut t_plain = 0.0;
    let mut n_plain = 0usize;
    let diff = std::env::var("FE_TREE_DIFF").is_ok_and(|v| v == "1");
    let v = e.cfg.vocab;
    let mut row = vec![0u16; v];
    // (tokens, their logits rows when diff)
    let mut reference = |e: &mut Engine, out: &[u32]| -> Result<(Vec<u32>, Vec<u16>)> {
        let mut all = ids.clone();
        all.extend_from_slice(out);
        let mut r = vec![prefill(e, 1, &all)?];
        let mut lg = Vec::new();
        if diff {
            e.logits_host(1, &mut row)?;
            lg.extend_from_slice(&row);
        }
        let t = Instant::now();
        while out.len() + r.len() < max_new {
            e.forward(&[(1, &[*r.last().unwrap()])], &[0])?;
            r.push(e.argmax(1)?[0]);
            if diff {
                e.logits_host(1, &mut row)?;
                lg.extend_from_slice(&row);
            }
        }
        t_plain += t.elapsed().as_secs_f64();
        n_plain += r.len() - 1;
        Ok((r, lg))
    };
    let first = prefill(&mut e, 0, &ids)?;
    let mut out = vec![first];
    let (mut refs, mut rlog) = reference(&mut e, &out)?; // the reference continuation of out (and its logits)
    let mut dmax: Vec<f32> = Vec::new();
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = |m: usize| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as usize % m.max(1)
    };
    let (mut rounds, mut kept, mut trees) = (0usize, 0usize, 0usize);
    let mut ties: Vec<(usize, f32, f32)> = Vec::new();
    let mut bad = None;
    let mut host = vec![0u16; (nodes + 1) * v];
    let t = Instant::now();
    while out.len() < max_new {
        let p = e.slot_len[0];
        let truth = refs.clone();
        let want = rnd(nodes + 1).min(truth.len());
        let (mut toks, mut par) = (vec![*out.last().unwrap()], vec![-1i32]);
        let (mut tip, mut k) = (0usize, 0usize);
        while toks.len() < nodes + 1 {
            let left = nodes + 1 - toks.len();
            if k < want && (prefix || want - k >= left || rnd(2) == 0) {
                toks.push(truth[k]);
                par.push(tip as i32);
                tip = toks.len() - 1;
                k += 1;
            } else {
                par.push(rnd(toks.len()) as i32);
                toks.push(1000 + rnd(100_000) as u32);
            }
        }
        let n = toks.len();
        let lr: Vec<usize> = (0..n).collect();
        let tv = Instant::now();
        let parents = [(!chain).then_some(par.as_slice())];
        e.forward_tree(&[(0, &toks)], &parents, &lr, true, n)?;
        let am = e.argmax(n)?;
        t_ver += tv.elapsed().as_secs_f64();
        let (path, bonus) = if chain {
            // the distractors follow the true path, so the chain walk sees the same tree
            let a = (0..n - 1).take_while(|&j| toks[j + 1] == am[j]).count();
            ((0..=a).collect::<Vec<_>>(), am[a])
        } else {
            fe_model::sched::tree_walk(&toks, &par, &am)
        };
        let tc = Instant::now();
        if chain {
            e.commit(0, p, n, path.len())?;
        } else {
            e.commit_tree(0, p, n, &path, None)?;
        }
        t_com += tc.elapsed().as_secs_f64();
        rounds += 1;
        trees += path.iter().enumerate().any(|(j, &r)| r != j) as usize;
        kept += path.len() - 1;
        let mut got: Vec<u32> = path[1..].iter().map(|&r| toks[r]).collect();
        got.push(bonus);
        let m = got.iter().zip(&truth).take_while(|(a, b)| a == b).count();
        if diff {
            // rows path[0..=m] predicted reference tokens 0..=m
            e.logits_host(n, &mut host)?;
            let mut d = 0f32;
            for (j, &r) in path.iter().enumerate().take((m + 1).min(truth.len())) {
                for c in 0..v {
                    let (a, b) = (host[r * v + c], rlog[j * v + c]);
                    d = d.max((f32::from_bits((a as u32) << 16) - f32::from_bits((b as u32) << 16)).abs());
                }
            }
            dmax.push(d);
        }
        if m < got.len().min(truth.len()) {
            // the walk left the reference at depth m (row path[m]): a near-tie, or a bug
            e.logits_host(n, &mut host)?;
            let r = path[m];
            let lg = |t: u32| f32::from_bits((host[r * v + t as usize] as u32) << 16);
            let (lt, lr) = (lg(got[m]), lg(truth[m]));
            let step = f32::from_bits((lt.abs().max(lr.abs()).to_bits() & 0x7f80_0000) - (7 << 23));
            if lt - lr <= step {
                ties.push((out.len() + m, lt, lr));
                out.extend_from_slice(&got[..=m]);
                (refs, rlog) = reference(&mut e, &out)?;
                continue;
            }
            bad = Some((out.len() + m, rounds, want, path.len() - 1, got[m], lt, truth[m], lr));
            out.extend_from_slice(&got[..m]);
            break;
        }
        let k = got.len().min(refs.len());
        refs.drain(..k);
        if diff {
            rlog.drain(..k * v);
        }
        out.extend_from_slice(&got);
    }
    let dt = t.elapsed().as_secs_f64();
    out.truncate(max_new);
    eprintln!("tree-test: {} tokens, {rounds} rounds of {} rows ({trees} kept a non-chain path), {:.2} kept/round; plain \
        {:.1} tok/s, trees {:.1} tok/s (with reference redos)", out.len(), nodes + 1, kept as f64 / rounds.max(1) as f64,
        n_plain as f64 / t_plain.max(1e-9), out.len() as f64 / dt);
    if diff && !dmax.is_empty() {
        let mut d = dmax.clone();
        d.sort_by(|a, b| a.total_cmp(b));
        eprintln!("tree-test: kept rows vs reference logits, max |diff| per round: median {:.4}, p90 {:.4}, max {:.4}",
            d[d.len() / 2], d[d.len() * 9 / 10], d[d.len() - 1]);
    }
    eprintln!("tree-test: per round verify forward + argmax {:.2} ms, commit {:.2} ms", 1e3 * t_ver / rounds.max(1) as f64,
        1e3 * t_com / rounds.max(1) as f64);
    for (at, lt, lr) in &ties {
        eprintln!("tree-test: near-tie at token {at}: tree pick logit {lt:.4}, reference logit {lr:.4}");
    }
    match bad {
        None => println!("tree-test: IDENTICAL to plain greedy ({} tokens, {} near-ties within one bf16 step)", out.len(),
            ties.len()),
        Some((at, r, want, got, gt, lt, rt, lr)) => {
            println!("tree-test: MISMATCH at token {at} (round {r}: oracle path {want}, kept {got}): tree {gt} logit {lt:.4} \
                vs reference {rt} logit {lr:.4}; output so far:");
            println!("{}", tok.decode(&out, false).map_err(|e| anyhow::anyhow!("{e}"))?);
            bail!("tree decode differs from plain greedy");
        }
    }
    Ok(())
}

fn hash16(v: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for c in v.chunks_exact(4) {
        let w = c[0] as u64 | (c[1] as u64) << 16 | (c[2] as u64) << 32 | (c[3] as u64) << 48;
        h = (h ^ w).wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

/// Debug: `logits` under a server-like step schedule (FE_LOG_SCHED, all sequences as parallel slots, budget
/// FE_LOG_BUDGET rows a forward, default 2048), teacher-forced, to find step-composition dependence:
///   stagger: each forward = one decode row of every slot whose prompt is done, then prompt rows in slot order up
///            to the budget (the scheduler with FE_BATCH_ROWS = budget: prompts cut across forwards and sharing
///            forwards with decode rows);
///   split:   the same prompt packing, but decode rows only after every prompt is in (chunk splits only);
///   mix:     one whole prompt a forward, with the decode rows of the slots before it (mixed forwards only).
/// FE_LOG_ALIGN=1 cuts prompts with the scheduler's rule (sched::prompt_cut: 32-row aligned cuts).
/// With identical sequences every slot must give the same bits; the report compares each slot to slot 0.
/// FE_LOG_DUMP=<prefix>: also tap every layer's residual (after the mixer and after the MLP) for every row and
/// report the first (position, layer, point) where a slot differs from slot 0, prompt rows and decode rows.
fn cmd_logits_sched(model: &Path, seqs: &[Vec<u32>], out_path: &Path, skip: usize, rows: usize, sched: &str)
    -> Result<()> {
    let n = seqs.len();
    let budget = env_usize("FE_LOG_BUDGET", 2048);
    let (_ctx, mut e) = load_engine(model, n, skip + rows, budget.max(n))?;
    let align = env_usize("FE_LOG_ALIGN", 0) == 1;
    let (v, h, nl) = (e.cfg.vocab, e.cfg.hidden, e.cfg.n_layers);
    let dump = std::env::var("FE_LOG_DUMP").ok();
    if dump.is_some() {
        e.set_feature_taps(&(0..nl).collect::<Vec<_>>())?;
        e.set_mid_taps()?;
    }
    let mut logits = vec![0u16; n * rows * v];
    let mut host = vec![0u16; n * v];
    let (mut fed, mut done) = (vec![0usize; n], vec![0usize; n]);
    // hashes [slot][pos][point 0 = mid, 1 = end][layer], pos over the whole sequence (prompt then decode rows)
    let tot = skip + rows;
    let mut hs = vec![0u64; n * tot * 2 * nl];
    let mut tap_end: Vec<u16> = Vec::new();
    let mut tap_mid: Vec<u16> = Vec::new();
    let mut nf = 0;
    // FE_LOG_CACHE=P,W: a prompt-cache history first: every slot prefills [0, P) alone, then W decode rows (slots
    // batched); the rest of the prompt [P + W, skip) then resumes at that cached length (cut grid from there)
    let mut base = 0;
    if let Ok(c) = std::env::var("FE_LOG_CACHE") {
        let v: Vec<usize> = c.split(',').map(|x| x.trim().parse()).collect::<std::result::Result<_, _>>()?;
        ensure!(v.len() == 2 && v[0] + v[1] < skip, "FE_LOG_CACHE=P,W with P + W < skip");
        for sl in 0..n {
            e.forward(&[(sl, &seqs[sl][..v[0]])], &[])?;
        }
        for k in 0..v[1] {
            let segs: Vec<(usize, &[u32])> = (0..n).map(|sl| (sl, &seqs[sl][v[0] + k..v[0] + k + 1])).collect();
            e.forward(&segs, &[])?;
        }
        base = v[0] + v[1];
        fed.iter_mut().for_each(|f| *f = base);
        eprintln!("cache phase: prefill {} + {} decode rows a slot, prompt resumes at {base} ({} mod 32)", v[0], v[1],
            base % 32);
    }
    while done.iter().any(|&d| d < rows) {
        let all_fed = fed.iter().all(|&f| f == skip);
        let mut segs: Vec<(usize, &[u32])> = Vec::new();
        let mut own: Vec<(usize, usize)> = Vec::new(); // (slot, first position)
        if sched != "split" || all_fed {
            for sl in 0..n {
                if fed[sl] == skip && done[sl] < rows {
                    let p = skip + done[sl];
                    segs.push((sl, &seqs[sl][p..p + 1]));
                    own.push((sl, p));
                }
            }
        }
        let ndec = segs.len();
        let mut used = ndec;
        if sched == "mix" {
            if let Some(sl) = (0..n).find(|&sl| fed[sl] < skip) {
                segs.push((sl, &seqs[sl][fed[sl]..skip]));
                own.push((sl, fed[sl]));
                used += skip - fed[sl];
            }
        } else {
            for sl in 0..n {
                if fed[sl] < skip && used < budget {
                    // FE_LOG_ALIGN=1: the scheduler's cut rule (fe_model::sched::prompt_cut)
                    let c = if align { fe_model::sched::prompt_cut(skip, base, fed[sl], budget - used) }
                        else { (skip - fed[sl]).min(budget - used) };
                    if c == 0 {
                        continue;
                    }
                    ensure!(c > 1, "a 1-row prompt chunk would count as a decode row");
                    segs.push((sl, &seqs[sl][fed[sl]..fed[sl] + c]));
                    own.push((sl, fed[sl]));
                    used += c;
                }
            }
        }
        let lr: Vec<usize> = (0..ndec).collect();
        e.forward(&segs, &lr)?;
        if ndec > 0 {
            e.logits_host(ndec, &mut host)?;
            for i in 0..ndec {
                let (sl, p) = own[i];
                let r = p - skip;
                logits[(sl * rows + r) * v..(sl * rows + r + 1) * v].copy_from_slice(&host[i * v..(i + 1) * v]);
            }
        }
        let comp: Vec<String> = segs.iter().map(|(sl, t)| format!("{sl}:{}", t.len())).collect();
        eprintln!("forward {nf}: {} rows, {ndec} decode, segs [{}]", used, comp.join(" "));
        if dump.is_some() {
            let w = used * nl * h;
            tap_end.resize(w, 0);
            tap_mid.resize(w, 0);
            fe_cuda::memcpy_dtoh(&mut tap_end[..w], e.feature_taps().unwrap())?;
            fe_cuda::memcpy_dtoh(&mut tap_mid[..w], e.mid_taps().unwrap())?;
            let mut row = 0;
            for (i, (_, t)) in segs.iter().enumerate() {
                let (sl, p0) = own[i];
                for j in 0..t.len() {
                    for l in 0..nl {
                        let o = ((row + j) * nl + l) * h;
                        let b = ((sl * tot + p0 + j) * 2) * nl + l;
                        hs[b] = hash16(&tap_mid[o..o + h]);
                        hs[b + nl] = hash16(&tap_end[o..o + h]);
                    }
                }
                row += t.len();
            }
        }
        for (i, (_, t)) in segs.iter().enumerate() {
            let sl = own[i].0;
            if i < ndec { done[sl] += 1 } else { fed[sl] += t.len() }
        }
        nf += 1;
    }
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(logits.as_ptr() as *const u8, logits.len() * 2) };
    std::fs::write(out_path, bytes)?;
    for sl in 1..n {
        let d: Vec<usize> = (0..rows).filter(|&r| logits[(sl * rows + r) * v..(sl * rows + r + 1) * v]
            != logits[r * v..(r + 1) * v]).collect();
        eprintln!("sched {sched}: slot {sl} vs slot 0: {} of {rows} logits rows differ{}", d.len(),
            d.first().map_or(String::new(), |r| format!(" (first row {r})")));
    }
    if let Some(pfx) = dump {
        let hb: &[u8] = unsafe { std::slice::from_raw_parts(hs.as_ptr() as *const u8, hs.len() * 8) };
        std::fs::write(format!("{pfx}.hash"), hb)?;
        for sl in 1..n {
            let mut first = None;
            'pos: for p in 0..tot {
                for l in 0..nl {
                    for pt in 0..2 {
                        let b = |s: usize| ((s * tot + p) * 2 + pt) * nl + l;
                        if hs[b(sl)] != hs[b(0)] {
                            first = Some((p, l, pt));
                            break 'pos;
                        }
                    }
                }
            }
            let nd = (0..tot).filter(|&p| (0..2 * nl).any(|k| hs[(sl * tot + p) * 2 * nl + k] != hs[p * 2 * nl + k])).count();
            match first {
                Some((p, l, pt)) => eprintln!("sched {sched} dump: slot {sl} vs 0: first difference at pos {p} ({}) layer {l} \
                    ({}) after the {}; {nd} of {tot} positions differ somewhere",
                    if p < skip { "prompt" } else { "decode" }, if e.cfg.full_attn[l] { "attn" } else { "gdn" },
                    if pt == 0 { "mixer" } else { "mlp" }),
                None => eprintln!("sched {sched} dump: slot {sl} vs 0: every position and layer identical"),
            }
        }
    }
    Ok(())
}

fn cmd_logits(model: &Path, ids_path: &Path, out_path: &Path, skip: usize, rows: usize) -> Result<()> {
    let text = std::fs::read_to_string(ids_path).with_context(|| format!("reading {ids_path:?}"))?;
    let seqs: Vec<Vec<u32>> = text.lines().filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|x| x.trim().parse::<u32>()).collect::<std::result::Result<Vec<_>, _>>())
        .collect::<std::result::Result<_, _>>()?;
    for (i, s) in seqs.iter().enumerate() {
        if s.len() < skip + rows {
            bail!("sequence {i} has {} ids < skip + rows = {}", s.len(), skip + rows);
        }
    }
    if let Ok(sched) = std::env::var("FE_LOG_SCHED") {
        return cmd_logits_sched(model, &seqs, out_path, skip, rows, &sched);
    }
    let batch = env_usize("FE_BATCH", 20).min(seqs.len());
    let chunk = env_usize("FE_PREFILL_CHUNK", 2048);
    let (_ctx, mut e) = load_engine(model, batch, skip + rows, chunk.max(batch))?;
    let v = e.cfg.vocab;
    let f = std::fs::File::create(out_path)?;
    f.set_len((seqs.len() * rows * v * 2) as u64)?;
    let mut host = vec![0u16; batch * v];
    let t_all = Instant::now();
    let (mut t_pf, mut t_dec, mut n_pf, mut n_dec) = (0.0, 0.0, 0usize, 0usize);
    for b0 in (0..seqs.len()).step_by(batch) {
        let group: Vec<usize> = (b0..(b0 + batch).min(seqs.len())).collect();
        for (slot, &si) in group.iter().enumerate() {
            e.reset_slot(slot)?;
            let t = Instant::now();
            for c in seqs[si][..skip].chunks(chunk) {
                e.forward(&[(slot, c)], &[])?;
            }
            t_pf += t.elapsed().as_secs_f64();
            n_pf += skip;
        }
        let t = Instant::now();
        for r in 0..rows {
            let segs: Vec<(usize, &[u32])> =
                group.iter().enumerate().map(|(slot, &si)| (slot, &seqs[si][skip + r..skip + r + 1])).collect();
            let lr: Vec<usize> = (0..group.len()).collect();
            e.forward(&segs, &lr)?;
            e.logits_host(group.len(), &mut host)?;
            for (slot, &si) in group.iter().enumerate() {
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(host[slot * v..(slot + 1) * v].as_ptr() as *const u8, v * 2)
                };
                f.write_all_at(bytes, ((si * rows + r) * v * 2) as u64)?;
            }
        }
        t_dec += t.elapsed().as_secs_f64();
        n_dec += rows * group.len();
        eprintln!("ferrite logits: seqs {}..{} done ({:.0}s)", b0, b0 + group.len(), t_all.elapsed().as_secs_f32());
    }
    if fe_cuda::canary::on() {
        let bad = fe_cuda::canary::tails()?;
        eprintln!("ferrite canary: {} damaged buffer tails", bad.len());
        for (base, bytes, first, n) in bad.iter().take(20) {
            eprintln!("  buffer 0x{base:x} ({bytes} B): tail written from +{first}, {n} bytes");
        }
    }
    eprintln!("ferrite logits: prefill {n_pf} tok {:.1}s ({:.0} tok/s), decode {n_dec} rows {:.1}s ({:.1} rows/s, batch {batch})",
        t_pf, n_pf as f64 / t_pf, t_dec, n_dec as f64 / t_dec);
    Ok(())
}

/// Prefill / decode throughput: for each prompt length, prefill once (chunked) then decode `dec` tokens greedily;
/// then `slots` parallel decode streams.
fn cmd_bench(model: &Path, lens: &[usize], dec: usize, slots: usize) -> Result<()> {
    let chunk = env_usize("FE_PREFILL_CHUNK", 2048);
    let max_len = lens.iter().copied().max().unwrap_or(2048);
    let ctx_len = env_usize("FE_MAX_CTX", max_len + dec + 8).max(max_len + dec + 8);
    let (_ctx, mut e) = load_engine(model, slots.max(1), ctx_len, chunk.max(slots))?;
    // deterministic pseudo-text ids (common-token range)
    let mut seed = 7u64;
    let ids: Vec<u32> = (0..max_len).map(|_| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (1000 + (seed >> 33) % 30000) as u32
    }).collect();
    for &len in lens {
        e.reset_slot(0)?;
        let t = Instant::now();
        let mut done = 0;
        while done < len {
            let end = (done + chunk).min(len);
            let lr: Vec<usize> = if end == len { vec![end - done - 1] } else { vec![] };
            e.forward(&[(0, &ids[done..end])], &lr)?;
            done = end;
        }
        let pf = t.elapsed().as_secs_f64();
        let mut next = e.argmax(1)?[0];
        // one warm-up step (graph capture), then timed decode
        e.forward(&[(0, &[next])], &[0])?;
        next = e.argmax(1)?[0];
        let t = Instant::now();
        for _ in 0..dec {
            e.forward(&[(0, &[next])], &[0])?;
            next = e.argmax(1)?[0];
        }
        let dt = t.elapsed().as_secs_f64();
        println!("bench len {len:6}: prefill {:7.0} tok/s ({pf:.2}s)   decode {:6.2} tok/s ({:.1} ms/tok)", len as f64 / pf,
            dec as f64 / dt, dt * 1e3 / dec as f64);
    }
    if slots > 1 {
        for s in 0..slots {
            e.reset_slot(s)?;
            e.forward(&[(s, &ids[..512])], &[511])?;
        }
        let mut next: Vec<u32> = vec![1000; slots];
        let all: Vec<usize> = (0..slots).collect();
        let segs = |n: &Vec<u32>| -> Vec<(usize, Vec<u32>)> { (0..slots).map(|s| (s, vec![n[s]])).collect() };
        let sg = segs(&next);
        let r: Vec<(usize, &[u32])> = sg.iter().map(|(s, t)| (*s, t.as_slice())).collect();
        e.forward(&r, &all)?;
        next = e.argmax(slots)?;
        let t = Instant::now();
        for _ in 0..dec {
            let sg = segs(&next);
            let r: Vec<(usize, &[u32])> = sg.iter().map(|(s, t)| (*s, t.as_slice())).collect();
            e.forward(&r, &all)?;
            next = e.argmax(slots)?;
        }
        let dt = t.elapsed().as_secs_f64();
        println!("bench {slots} streams: {:.2} tok/s aggregate, {:.2} per stream ({:.1} ms/step)",
            (slots * dec) as f64 / dt, dec as f64 / dt, dt * 1e3 / dec as f64);
    }
    Ok(())
}

/// Scheduler smoke test: several concurrent greedy requests through fe_model::sched.
fn cmd_sched_test(model: &Path, n: usize) -> Result<()> {
    use fe_model::sched::{Event, Request, Sampling, SchedOptions, Scheduler};
    let tok = tokenizers::Tokenizer::from_file(tokenizer_path(model)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let drafter = (std::env::var("FE_DRAFT").as_deref() == Ok("dflash")).then(fe_dflash::factory);
    let sch = Scheduler::start(model, SchedOptions { n_slots: 4, max_ctx: 4096, prefill_chunk: 2048, drafter,
        ..Default::default() })?;
    if let Ok(follow) = std::env::var("FE_SCHED_FOLLOW") {
        return sched_cache_test(&sch, &tok, n, &follow);
    }
    let prompts = ["The three primary colors are", "def fibonacci(n):", "In 1969, the first humans", "A haiku about rust:",
        "The capital of Japan is", "SELECT name FROM users WHERE"];
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Event)>();
    let t = Instant::now();
    // FE_SCHED_PROMPT: every request uses this text instead
    let own = std::env::var("FE_SCHED_PROMPT").ok();
    let prompts: Vec<&str> = match &own {
        Some(p) => vec![p.as_str()],
        None => prompts.to_vec(),
    };
    for i in 0..n {
        let ids = tok.encode(prompts[i % prompts.len()], false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
        let tx = tx.clone();
        sch.submit(Request { id: format!("req{i}"), prompt_ids: ids, snapshots: Vec::new(), sampling: Sampling { max_tokens: env_usize("FE_MAX_TOKENS", 48),
            stop_token_ids: vec![248044, 248046], ..Default::default() }, cancel: Default::default() },
            Box::new(move |ev| tx.send((i, ev)).is_ok()));
    }
    drop(tx);
    let mut outs = vec![Vec::new(); n];
    let mut total = 0;
    for (i, ev) in rx {
        match ev {
            Event::Tokens(t) => {
                total += t.len();
                outs[i].extend(t);
            }
            Event::Finished { reason, cached_prompt_tokens } => {
                let text = tok.decode(&outs[i], false).map_err(|e| anyhow::anyhow!("{e}"))?;
                println!("[req{i}] {:?} cached {cached_prompt_tokens}: {}{}", reason, prompts[i % prompts.len()], text);
            }
        }
    }
    let dt = t.elapsed().as_secs_f64();
    println!("sched-test: {n} requests, {total} tokens in {dt:.2}s ({:.1} tok/s aggregate)", total as f64 / dt);
    Ok(())
}

/// Prompt-cache identity (FE_SCHED_FOLLOW=<text>, with FE_SCHED_PROMPT): n requests of FE_SCHED_PROMPT warm n slots
/// (FE_WARM_TOKENS, default 8, greedy), then n follow-ups = prompt + request 0's output + the FE_SCHED_FOLLOW text
/// (ids concatenated) hit those slots' cached histories (cached length = prompt + output - 1: any position).
/// Every follow-up must give the same text, whatever the arrival schedule (FE_BATCH_ROWS).
fn sched_cache_test(sch: &fe_model::sched::Scheduler, tok: &tokenizers::Tokenizer, n: usize, follow: &str) -> Result<()> {
    use fe_model::sched::{Event, Request, Sampling};
    let enc = |t: &str| -> Result<Vec<u32>> { Ok(tok.encode(t, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec()) };
    let p = enc(&std::env::var("FE_SCHED_PROMPT").context("FE_SCHED_FOLLOW needs FE_SCHED_PROMPT")?)?;
    let run = |ids: &[Vec<u32>], max: usize| -> Result<Vec<(Vec<u32>, String, usize)>> {
        let (tx, rx) = std::sync::mpsc::channel::<(usize, Event)>();
        for (i, v) in ids.iter().enumerate() {
            let tx = tx.clone();
            sch.submit(Request { id: format!("req{i}"), prompt_ids: v.clone(), snapshots: Vec::new(), sampling: Sampling { max_tokens: max,
                stop_token_ids: vec![248044, 248046], ..Default::default() }, cancel: Default::default() },
                Box::new(move |ev| tx.send((i, ev)).is_ok()));
        }
        drop(tx);
        let mut r = vec![(Vec::new(), String::new(), 0usize); ids.len()];
        for (i, ev) in rx {
            match ev {
                Event::Tokens(t) => r[i].0.extend(t),
                Event::Finished { reason, cached_prompt_tokens } => {
                    r[i].1 = format!("{reason:?}");
                    r[i].2 = cached_prompt_tokens;
                }
            }
        }
        Ok(r)
    };
    let warm = run(&vec![p.clone(); n], env_usize("FE_WARM_TOKENS", 8))?;
    let same = warm.iter().all(|w| w.0 == warm[0].0);
    let mut f = p.clone();
    f.extend_from_slice(&warm[0].0);
    f.extend(enc(follow)?);
    println!("sched-cache: prompt {} ids, warm-up {} tokens ({}), follow-up {} ids", p.len(), warm[0].0.len(),
        if same { "all slots identical" } else { "warm-ups DIFFER" }, f.len());
    let t = Instant::now();
    let r = run(&vec![f; n], env_usize("FE_MAX_TOKENS", 48))?;
    let mut total = 0;
    for (i, (out, reason, cached)) in r.iter().enumerate() {
        total += out.len();
        let text = tok.decode(out, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("[req{i}] {reason} cached {cached}: FOLLOW-UP {text}");
    }
    let cached: Vec<usize> = r.iter().map(|x| x.2).collect();
    println!("sched-cache: cached prefix per follow-up {cached:?} (mod 32: {:?})", cached.iter().map(|c| c % 32).collect::<Vec<_>>());
    let dt = t.elapsed().as_secs_f64();
    println!("sched-test: {n} requests, {total} tokens in {dt:.2}s ({:.1} tok/s aggregate)", total as f64 / dt);
    Ok(())
}

/// Prompt-cache primitive test (park_save / park_restore / copy_slot), bitwise. One code sequence (FE_PARK_IDS,
/// default the first KL code sequence), 4 slots, every prompt cut at the resume point P = 700 (as the cache's
/// cut rule does) and on the 32-row grid:
///   slot 0: reference, [0, 512) [512, 700) [700, 1000)
///   slot 1: [0, 512) [512, 700), snapshot (recurrent state only) at 700, then a 200-token other "reply",
///           restore the snapshot (KV [0, 700) is still the slot's), then [700, 1000)   (turn-boundary resume)
///   slot 3: copy_slot(1 -> 3, KV 0..700 + state) right at 700, then [700, 1000)          (shared prefix)
///   slot 2: [0, 512) [512, 700) [700, 1000), park KV 0..1000 + state, reset, 400 other tokens, restore (parking);
///           FE_PARK_BLOCKS=1: kv_out 1000 rows, snapshot at 700, clobber, kv_in 700 rows + restore 700, [700, 1000)
/// With FE_PARK_BLOCKS=1 slots 2 and 3 use the scheduler's block API instead: slot 2 kv_out(1000 rows) + cache_save
/// at 1000, reset, clobber, kv_in + cache_restore; slot 3 gets slot 1's KV [0, 700) via kv_out / kv_in and its
/// snapshot via cache_save(slot 1) / cache_restore(slot 3).
/// then 16 teacher-forced decode rows on all four slots in each forward; every row's logits must equal slot 0's.
/// TP2: run as rank 0 with a follower (tools/tp2.sh FOLLOW=<model>), so every park call goes through the mirror.
fn cmd_park_test(model: &Path) -> Result<()> {
    let path = std::env::var("FE_PARK_IDS").unwrap_or_else(|_| {
        format!("{}/kl-ids-code.txt", std::env::var("FERRITE_KL_DIR").unwrap_or_else(|_| "/var/tmp/kl27b".into()))
    });
    let line = std::fs::read_to_string(&path)?.lines().next().context("no ids")?.to_string();
    let ids: Vec<u32> = line.split(',').map(|x| x.trim().parse::<u32>()).collect::<std::result::Result<_, _>>()?;
    ensure!(ids.len() >= 1916, "need >= 1916 ids");
    let (p, end, rows) = (700usize, 1000usize, 16usize);
    let other = &ids[1500..1900];
    let (_ctx, mut e) = load_engine(model, 4, 2048, 1024)?;
    let v = e.cfg.vocab;
    let blocks = std::env::var("FE_PARK_BLOCKS").is_ok_and(|v| v == "1");
    let entry_snap = 0usize;
    let entry_park = e.park_bytes(0, true);
    let pool = if blocks { e.cache_pool(3, 0, 2)? } else {
        let pool = entry_park + e.park_bytes(end, true);
        e.park_pool(pool)?;
        pool
    };
    eprintln!("park-test: pool {:.1} MB (KV {:.1} KiB a token, recurrent state {:.1} MB a slot on this rank)",
        pool as f64 / 1e6, e.park_kv_bytes() as f64 / 1024.0, e.park_gdn_bytes() as f64 / 1e6);
    let pre = |e: &mut Engine, s: usize, a: usize, b: usize| e.forward(&[(s, &ids[a..b])], &[]);
    for s in 0..4 {
        e.reset_slot(s)?;
    }
    // slot 0: reference
    pre(&mut e, 0, 0, 512)?; pre(&mut e, 0, 512, p)?; pre(&mut e, 0, p, end)?;
    // slot 1 + slot 3: snapshot at P, shared-prefix copy at P
    pre(&mut e, 1, 0, 512)?; pre(&mut e, 1, 512, p)?;
    if blocks {
        e.cache_save(1, p, 0)?;
        e.kv_out(1, p, &[0])?;
        e.reset_slot(3)?;
        e.forward(&[(3, &other[..300])], &[])?;
        e.kv_in(3, p, &[0])?;
        e.cache_restore(3, p, 0)?;
    } else {
        e.park_save(1, p..p, true, entry_snap)?;
        e.copy_slot(1, 3, 0..p, true, p)?;
    }
    e.forward(&[(1, &other[..200])], &[])?;
    if blocks { e.cache_restore(1, p, 0)?; } else { e.park_restore(1, p..p, true, entry_snap, p)?; }
    pre(&mut e, 1, p, end)?;
    pre(&mut e, 3, p, end)?;
    // slot 2: park the whole state, clobber the slot, restore
    pre(&mut e, 2, 0, 512)?; pre(&mut e, 2, 512, p)?;
    if blocks {
        // parked with 1000 rows, restored at the earlier snapshot (700): kv_in takes a prefix of the block
        e.cache_save(2, p, 2)?;
    }
    pre(&mut e, 2, p, end)?;
    if blocks {
        e.kv_out(2, end, &[1])?;
        e.reset_slot(2)?;
        e.forward(&[(2, other)], &[])?;
        e.kv_in(2, p, &[1])?;
        e.cache_restore(2, p, 2)?;
        pre(&mut e, 2, p, end)?;
    } else {
        e.park_save(2, 0..end, true, entry_park)?;
        e.reset_slot(2)?;
        e.forward(&[(2, other)], &[])?;
        e.park_restore(2, 0..end, true, entry_park, end)?;
    }
    ensure!(e.slot_len.iter().all(|&l| l == end), "slot lengths {:?}", e.slot_len);
    let mut host = vec![0u16; 4 * v];
    let mut bad = [0usize; 4];
    let names = ["reference", "snapshot resume", "parked + restored", "shared-prefix copy"];
    for r in 0..rows {
        let t = &ids[end + r..end + r + 1];
        e.forward(&[(0, t), (1, t), (2, t), (3, t)], &[0, 1, 2, 3])?;
        e.logits_host(4, &mut host)?;
        for s in 1..4 {
            if host[s * v..(s + 1) * v] != host[..v] {
                bad[s] += 1;
            }
        }
    }
    for s in 1..4 {
        println!("park-test: slot {s} ({}): {} of {rows} decode rows differ from the reference", names[s], bad[s]);
    }
    println!("park-test: {}", if bad.iter().all(|&b| b == 0) { "PASS (bitwise)" } else { "FAIL" });
    Ok(())
}

/// Tensor-core peak and bf16 GEMM throughput at prefill shapes.
fn cmd_gemm_bench() -> Result<()> {
    use fe_cuda::DeviceBuffer;
    let _ctx = Context::new(0)?;
    let m = Module::load(fe_model::K_CORE_CUBIN)?;
    let s = Stream::new()?;
    let peak = m.function("fe_t_mma_peak")?;
    let out = DeviceBuffer::<f32>::zeroed(1)?;
    let iters = 4096u32;
    let blocks = 48 * 8;
    let threads = 256;
    let ms = s.time(|s| unsafe { s.launch(&peak, blocks, threads, 0, (out.ptr(), iters)) })?;
    let ms = s.time(|s| unsafe { s.launch(&peak, blocks, threads, 0, (out.ptr(), iters)) })?.min(ms);
    let flops = blocks as f64 * (threads / 32) as f64 * iters as f64 * 8.0 * 16.0 * 8.0 * 16.0 * 2.0;
    println!("mma.sync bf16 peak: {:.1} TFLOPS", flops / ms as f64 / 1e9);
    let g = m.function("fe_gemm_bf16")?;
    for &(mm, nn, kk) in &[(2048usize, 34816usize, 5120usize), (2048, 5120, 17408), (2048, 16480, 5120), (2048, 5120, 6144)] {
        let a = DeviceBuffer::<u16>::zeroed(mm * kk)?;
        let w = DeviceBuffer::<u16>::zeroed(nn * kk)?;
        let c = DeviceBuffer::<u16>::zeroed(mm * nn)?;
        let grid = ((mm as u32).div_ceil(128), (nn as u32).div_ceil(128));
        let run = |s: &Stream| unsafe {
            s.launch(&g, grid, 256, 49152, (c.ptr(), a.ptr(), w.ptr(), fe_cuda::DevPtr::NULL, mm as u32, nn as u32,
                kk as u32, kk as u32, nn as u32))
        };
        run(&s)?;
        let ms = s.time(|s| {
            for _ in 0..5 {
                run(s)?;
            }
            Ok(())
        })? / 5.0;
        println!("gemm {mm}x{nn}x{kk}: {:.3} ms, {:.1} TFLOPS", ms, 2.0 * (mm * nn * kk) as f64 / ms as f64 / 1e9);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("selftest") {
        let ctx = Context::new(0)?;
        let (maj, min) = ctx.compute_capability()?;
        println!("device: {} sm_{maj}{min}, {} SMs, smem optin {} B", ctx.name(), ctx.sm_count()?, ctx.max_smem_optin()?);
        let m = Module::load(fe_model::K_CORE_CUBIN)?;
        let s = Stream::new()?;
        fe_model::selftest::run(&m, &s)?;
        println!("selftest: all OK");
        return Ok(());
    }
    if args.get(1).map(String::as_str) == Some("gemm-bench") {
        return cmd_gemm_bench();
    }
    if args.len() < 3 {
        bail!("usage: ferrite selftest | ferrite <model> generate <prompt> [n] | ferrite <model> logits <ids> <out> <skip> <rows>");
    }
    let model = Path::new(&args[1]);
    match args[2].as_str() {
        "follow" => cmd_follow(model),
        "generate" => cmd_generate(model, args.get(3).context("prompt")?, args.get(4).and_then(|x| x.parse().ok()).unwrap_or(64)),
        "logits" => {
            let a = |i: usize| args.get(i).with_context(|| "logits <ids.txt> <out.bin> <skip> <rows>");
            cmd_logits(model, Path::new(a(3)?), Path::new(a(4)?), a(5)?.parse()?, a(6)?.parse()?)
        }
        "bench" => {
            let lens: Vec<usize> = args.get(3).map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or_else(|| vec![2048, 8192, 32768]);
            let dec = args.get(4).and_then(|x| x.parse().ok()).unwrap_or(64);
            let slots = args.get(5).and_then(|x| x.parse().ok()).unwrap_or(4);
            cmd_bench(model, &lens, dec, slots)
        }
        "df-snaptest" => fe_dflash::snaptest(model, args.get(3).and_then(|x| x.parse().ok()).unwrap_or(6000)),
        "park-test" => cmd_park_test(model),
        "sched-test" => cmd_sched_test(model, args.get(3).and_then(|x| x.parse().ok()).unwrap_or(6)),
        "tree-test" => cmd_tree_test(model, args.get(3).context("prompt")?, args.get(4).and_then(|x| x.parse().ok()).unwrap_or(128),
            args.get(5).and_then(|x| x.parse().ok()).unwrap_or(7)),
        c => bail!("unknown command {c}"),
    }
}
