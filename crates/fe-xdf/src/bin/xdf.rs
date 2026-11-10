//! xdf: DFlash2 drafting on the B70.
//!   xdf quant [draft dir]                                   quantize + cache the drafter linears (no GPU)
//!   xdf <model> gen <ids.txt> <n> [--check] [--k K] [--window W] [--seq I]
//!                                                           greedy generation after sequence I (default 0) of ids.txt with
//!                                                           DFlash2 drafts; --check: plain greedy too, ids must match
//! Env: FE_DRAFT_DIR (the Qwen3.8-27B-DFlash2 folder), FE_DF_BLOCK, FE_DF_BITS, FE_DF_VOCAB, FE_DF_PROF.
use anyhow::{bail, ensure, Result};
use fe_xdf::draft::{packed_linears, DFlash};
use fe_xdf::spec::{DfCfg, DfSpec};
use std::path::{Path, PathBuf};

fn flag<'a>(a: &'a [String], name: &str) -> Option<&'a str> {
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).map(String::as_str)
}

fn draft_dir(a: Option<&str>) -> PathBuf {
    a.map(PathBuf::from)
        .or_else(|| std::env::var("FE_DRAFT_DIR").ok().map(PathBuf::from))
        .expect("the drafter folder: an argument or FE_DRAFT_DIR")
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len() >= 2, "usage: xdf quant [dir] | xdf <model> gen <ids.txt> <n> [--check] [--k K] [--window W] [--seq I]");
    if a[1] == "kprops" {
        // register / spill report of the kx-dflash kernels (IGC compile only, no launch)
        let ze = fe_ze::Ze::new().map_err(|e| anyhow::anyhow!("{e}"))?;
        let flags = a.get(2).map(String::as_str).unwrap_or("");
        let m = fe_ze::Module::from_spirv(&ze, fe_xdf::draft::KX_DFLASH, flags).map_err(|e| anyhow::anyhow!("{e}"))?;
        for (name, local) in [("df_dynconv", 256u32), ("df_norm_rope", 16), ("df_attn_dpas", 64), ("df_swiglu2", 256),
            ("df_attn_comb", 16), ("df_topk_part", 256), ("df_topk_merge", 256), ("df_select", 256), ("df_tap", 256)] {
            let k = m.kernel(name, [local, 1, 1]).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            let p = k.props().map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("{name:<16} local {local:>4}: {p:?}");
        }
        return Ok(());
    }
    if a[1] == "treesim" {
        // xdf treesim <dump.rounds>...: tree acceptance (7 nodes) over a grid of growth constants, CPU only
        let dir = draft_dir(None);
        let st = fe_xq::st::SafeTensors::open_dir(&dir)?;
        let cb = (st.u16s("candidate_selector.predecessor_codebook", "BF16", &[248320, 256])?,
            st.u16s("candidate_selector.successor_codebook", "BF16", &[248320, 256])?);
        let dumps: Vec<_> = a[2..].iter().map(|f| (f.clone(), fe_xdf::tree::load_dump(Path::new(f)).unwrap())).collect();
        for branch in [2usize, 3, 4, 6] {
            for edge in [0.0f64, 0.3, 0.6, 1.0] {
                for scale in [1.0f64, 1.5, 2.5] {
                    let tp = fe_xdf::tree::TreeParams { branch, edge, scale };
                    let mut line = format!("branch {branch} edge {edge:.1} scale {scale:.1}:");
                    for (f, (rs, seq)) in &dumps {
                        let r = fe_xdf::tree::simulate_p(rs, seq, &cb, &[7], tp);
                        line += &format!(" {} {:.3}/{:.3}", Path::new(f).file_stem().unwrap().to_string_lossy(), r[0].1, r[0].2);
                    }
                    println!("{line}");
                }
            }
        }
        return Ok(());
    }
    if a[1] == "quant" {
        let dir = draft_dir(a.get(2).map(String::as_str));
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let nl = cfg["num_hidden_layers"].as_u64().unwrap_or(5) as usize;
        let ts = packed_linears(&dir, nl)?;
        let bytes: usize = ts.iter().map(|t| t.data.len() * 4).sum();
        println!("{} linears, {:.2} GB packed", ts.len(), bytes as f64 / 1e9);
        if std::env::var("FE_DF_HEAD").is_ok_and(|v| v == "q4") {
            let span = std::env::var("FE_DF_VOCAB").ok().and_then(|v| v.parse().ok()).unwrap_or(131072usize);
            let h = fe_xdf::draft::head_q4(span)?;
            println!("draft head q4: {} x {}, {:.2} GB", h.n, h.k, h.data.len() as f64 * 4.0 / 1e9);
        }
        return Ok(());
    }
    if a.len() >= 4 && a[2] == "dprof" {
        return dprof(&a);
    }
    if a.len() >= 5 && a[2] == "bench" {
        return bench(&a);
    }
    ensure!(a.len() >= 5 && a[2] == "gen", "usage: xdf <model> gen <ids.txt> <n> [--check] [--k K] [--window W] [--seq I]");
    let model = Path::new(&a[1]);
    let seqs = fe_xe::eval::read_ids(Path::new(&a[3]))?;
    let n: usize = a[4].parse()?;
    let si: usize = flag(&a, "--seq").map(str::parse).transpose()?.unwrap_or(0);
    let prompt = seqs.get(si).ok_or_else(|| anyhow::anyhow!("no sequence {si}"))?.clone();
    let win: u32 = flag(&a, "--window").map(str::parse).transpose()?.unwrap_or(4096);
    let ctx = (prompt.len() + n + 64).next_multiple_of(1024) as u32;
    let mut e = fe_xe::engine::open(model, ctx, win.max(16))?;
    let mut d = DFlash::load(&mut e, &draft_dir(None))?;
    eprintln!("fe-xdf: {:.2} GB on the device with the drafter", e.dev.bytes.get() as f64 / 1e9);
    let mut cfg = DfCfg { rows: win as usize, ..Default::default() };
    if let Some(k) = flag(&a, "--k") {
        cfg.k = k.parse()?;
    }
    ensure!(prompt.len() >= 2, "prompt too short");
    e.reset()?;
    let mut sp = DfSpec::new(cfg.clone());
    let first = sp.prefill(&mut e, &mut d, &prompt, 0, &[], &mut |_, _| Ok(()), None)?;
    e.kx.profile_reset()?;
    let mut out = vec![first];
    while sp.st.tokens < n {
        let (kept, send, hit) = sp.step(&mut e, &mut d, n - sp.st.tokens, &[], None)?;
        out.extend_from_slice(&kept[..send]);
        if hit {
            break;
        }
    }
    e.dev.sync()?;
    let st = sp.st.clone();
    println!("dflash ids: {}", out.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
    eprintln!("prompt {} in {:.2} s ({:.0} tok/s incl. absorb); {} tokens in {:.2} s = {:.1} tok/s; windows {} plain {} drafted {} accepted {} ({:.0} %, {:.2} a window)",
        st.prompt, st.prefill_s, st.prompt as f64 / st.prefill_s, st.tokens, st.gen_s, (st.tokens - 1) as f64 / st.gen_s,
        st.windows, st.plain, st.drafted, st.accepted, 100.0 * st.accepted as f64 / st.drafted.max(1) as f64,
        st.accepted as f64 / st.windows.max(1) as f64);
    let tm = &sp.tm;
    if cfg.prof {
        let w = st.windows.max(1) as f64 / 1e3;
        eprintln!("per window: draft {:.2} ms, verify {:.2} ms, commit {:.2} ms, absorb {:.2} ms", tm.draft / w, tm.verify / w, tm.commit / w, tm.absorb / w);
    }
    eprintln!("accepted-drafts histogram: {:?}", &tm.hist[..d.block]);
    if let (Some(rs), Some(cb)) = (sp.rounds.as_ref(), d.cb_host.as_ref()) {
        let mut seq = prompt.clone();
        seq.extend_from_slice(&out);
        let budgets: Vec<usize> = std::env::var("FE_DF_TREESIM").ok().filter(|v| !v.is_empty() && v != "1")
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or(vec![3, 5, 7, 11, 15, 23, 31]);
        for (b, c, t) in fe_xdf::tree::simulate(rs, &seq, cb, &budgets) {
            eprintln!("treesim: budget {b:>2}: chain {c:.2} / tree {t:.2} accepted a round ({} rounds)", rs.len());
        }
    }
    if a.iter().any(|x| x == "--check") {
        let mut plain = vec![first];
        let mut am = [0i32; 16];
        e.reset()?;
        // plain greedy: the same prompt windows, then 1-row steps
        let r = cfg.rows;
        let mut t = 0;
        while t < prompt.len() {
            let end = prompt.len().min((t / r + 1) * r);
            let last = end == prompt.len();
            e.forward_rows(&prompt[t..end], if last { 1 } else { fe_xe::engine::KEEP_HIDDEN }, true)?;
            t = end;
        }
        e.argmax(&mut am)?;
        if am[0] as u32 != first {
            bail!("plain first token {} != {first}", am[0]);
        }
        let t1 = std::time::Instant::now();
        while plain.len() < out.len() {
            e.forward_rows(&[*plain.last().unwrap()], 1, true)?;
            e.argmax(&mut am)?;
            plain.push(am[0] as u32);
        }
        eprintln!("plain: {} tokens in {:.2} s = {:.1} tok/s", plain.len() - 1, t1.elapsed().as_secs_f64(), (plain.len() - 1) as f64 / t1.elapsed().as_secs_f64());
        if plain == out {
            println!("LOSSLESS: DFlash2 ids == plain greedy ids ({} tokens)", out.len());
        } else {
            let i = plain.iter().zip(&out).position(|(x, y)| x != y).unwrap_or(plain.len().min(out.len()));
            println!("DIFFERS at token {i}: plain {:?} dflash {:?}", &plain[i..(i + 5).min(plain.len())], &out[i..(i + 5).min(out.len())]);
        }
    }
    eprintln!("fe-xe: {}", e.kx.origin_line());
    Ok(())
}

/// Plain greedy ids: the prompt in windows of `r` rows, then 1-row steps, `n` tokens (the first from the prompt).
fn plain(e: &mut fe_xe::engine::Engine, prompt: &[u32], n: usize, r: usize) -> Result<(Vec<u32>, f64)> {
    let mut am = [0i32; 16];
    e.reset()?;
    let mut t = 0;
    while t < prompt.len() {
        let end = prompt.len().min((t / r + 1) * r);
        let last = end == prompt.len();
        e.forward_rows(&prompt[t..end], if last { 1 } else { fe_xe::engine::KEEP_HIDDEN }, true)?;
        t = end;
    }
    e.argmax(&mut am)?;
    let mut out = vec![am[0] as u32];
    let t1 = std::time::Instant::now();
    while out.len() < n {
        e.forward_rows(&[*out.last().unwrap()], 1, true)?;
        e.argmax(&mut am)?;
        out.push(am[0] as u32);
    }
    Ok((out, (n - 1) as f64 / t1.elapsed().as_secs_f64()))
}

/// xdf <model> bench <n> <ids.txt>... [--configs B:K,B:K..] [--check] [--reps R]: every prompt (first sequence of each file)
/// with every (block, verify drafts) config, one summary line each; --check also runs plain greedy once per prompt and compares.
fn bench(a: &[String]) -> Result<()> {
    let model = Path::new(&a[1]);
    let n: usize = a[3].parse()?;
    let files: Vec<&String> = a[4..].iter().take_while(|x| !x.starts_with("--")).collect();
    let cfgs: Vec<(usize, usize)> = flag(a, "--configs").unwrap_or("8:7").split(',')
        .map(|c| { let (b, k) = c.split_once(':').unwrap(); (b.parse().unwrap(), k.parse().unwrap()) }).collect();
    let reps: usize = flag(a, "--reps").map(str::parse).transpose()?.unwrap_or(1);
    let check = a.iter().any(|x| x == "--check");
    let prompts: Vec<(String, Vec<u32>)> = files.iter().map(|f| {
        let s = fe_xe::eval::read_ids(Path::new(f.as_str())).unwrap();
        (Path::new(f.as_str()).file_stem().unwrap().to_string_lossy().into_owned(), s[0].clone())
    }).collect();
    let maxp = prompts.iter().map(|p| p.1.len()).max().unwrap_or(0);
    let ctx = (maxp + n + 64).next_multiple_of(1024) as u32;
    let mut e = fe_xe::engine::open(model, ctx, 4096)?;
    let mut d = DFlash::load(&mut e, &draft_dir(None))?;
    for (name, prompt) in &prompts {
        let reference = if check { Some(plain(&mut e, prompt, n, 4096)?) } else { None };
        if let Some((_, tps)) = &reference {
            println!("{name:<14} plain            {tps:6.1} tok/s");
        }
        for &(b, k) in &cfgs {
            for _ in 0..reps {
                d.block = b;
                let cfg = DfCfg { k, rows: 4096, ..Default::default() };
                let mut out = Vec::new();
                let (st, tm, rounds) = fe_xdf::spec::generate(&mut e, &mut d, prompt, n, &cfg, &[], |t| { out.extend_from_slice(t); true })?;
                let ok = match &reference {
                    Some((r, _)) => if *r == out { "identical" } else { "DIFFERS" },
                    None => "",
                };
                let w = st.windows.max(1) as f64 / 1e3;
                println!("{name:<14} block {b:>2} k {k:>2}  {:6.1} tok/s  {:.2} accepted a window ({} windows, {} copies, {} vetoed) draft {:.2} ms verify {:.2} ms hist {:?} {ok}",
                    (st.tokens - 1) as f64 / st.gen_s, st.accepted as f64 / st.windows.max(1) as f64, st.windows, st.copies, st.vetoed,
                    tm.draft / w, tm.verify / w,
                    &tm.hist[..k.min(15) + 1]);
                if let (Some(rs), Ok(dir)) = (rounds.as_ref(), std::env::var("FE_DF_TREESIM_DUMP")) {
                    let mut seq = prompt.clone();
                    seq.extend_from_slice(&out);
                    let f = Path::new(&dir).join(format!("{name}-b{b}.rounds"));
                    fe_xdf::tree::dump(&f, rs, &seq)?;
                }
                if let (Some(rs), Some(cb)) = (rounds.as_ref(), d.cb_host.as_ref()) {
                    let mut seq = prompt.clone();
                    seq.extend_from_slice(&out);
                    let bud: Vec<usize> = [3usize, 5, 7, 11, 15].into_iter().filter(|&x| x < b).collect();
                    let sim = fe_xdf::tree::simulate(rs, &seq, cb, &bud);
                    println!("{name:<14}   treesim (chain / tree accepted a round): {}", sim.iter()
                        .map(|(b, c, t)| format!("{b}: {c:.2} / {t:.2}")).collect::<Vec<_>>().join(", "));
                }
            }
        }
    }
    Ok(())
}

/// xdf <model> dprof <ids.txt> [N]: prefill, then N (default 100) draft passes and N 8-row absorbs at the prompt end, timed on the
/// wall clock (each draft ends in a host sync), and with FE_XE_PROFILE=1 the device time per kernel of drafts + absorbs only.
fn dprof(a: &[String]) -> Result<()> {
    let model = Path::new(&a[1]);
    let seqs = fe_xe::eval::read_ids(Path::new(&a[3]))?;
    let prompt = seqs[0].clone();
    let n: usize = a.get(4).map(|x| x.parse()).transpose()?.unwrap_or(100);
    let ctx = (prompt.len() + 128).next_multiple_of(1024) as u32;
    let mut e = fe_xe::engine::open(model, ctx, 4096)?;
    let mut d = DFlash::load(&mut e, &draft_dir(None))?;
    let mut sp = DfSpec::new(DfCfg::default());
    let first = sp.prefill(&mut e, &mut d, &prompt, 0, &[], &mut |_, _| Ok(()), None)?;
    let pos = prompt.len();
    e.dev.sync()?;
    for _ in 0..5 {
        d.draft(&e, first, pos)?;
    }
    e.kx.profile_reset()?;
    d.profile_reset()?;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        d.draft(&e, first, pos)?;
    }
    let td = t0.elapsed().as_secs_f64() / n as f64;
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        d.absorb(&e, d.taps_buf.ptr(), 8, pos - 8)?;
    }
    e.dev.sync()?;
    let ta = t1.elapsed().as_secs_f64() / n as f64;
    println!("draft {:.3} ms (block {}), absorb of 8 rows {:.3} ms (wall, {n} each)", td * 1e3, d.block, ta * 1e3);
    if let Some(r) = e.kx.profile_report()? {
        print!("engine registry (drafter XQ / kx-core launches): {r}");
    }
    if let Some(r) = d.profile_report()? {
        print!("kx-dflash: {r}");
    }
    Ok(())
}
