//! fe-xe CLI.
//!   fe-xe <model> info [--ctx N] [--kv q8|bf16|q4] [--prepare]   device memory plan (weights, KV, GDN state); --prepare also reads and
//!                                                                packs every tensor on the host (checks the checkpoint end to end)
//!   fe-xe <model> logits <ids.txt> <out.bin> <skip> <rows> [--window W] [--f32] [--seqs A:B] [--bf16-logits]
//!                                                                Ferrite's KL contract: bf16 (or fp32) logits [nseq][rows][vocab] of
//!                                                                positions skip..skip+rows-1 of each sequence (windows of W rows:
//!                                                                <= 16 the verify-window path, up to 4096 the prompt path)
//!   fe-xe <model> fpkl <IN.bin> <OUTDIR> [--rows A:B] [--window W]  fpkl row records (all positions)
//!   fe-xe <model> gen <ids.txt> <n> [--window W]                  greedy generation of n tokens after the first sequence (plain)
//!   fe-xe <model> pfbench <ids.txt> [--len L] [--reps N] [--window W]  prompt prefill time: the first L ids of the file (sequences
//!                                                                concatenated) in W-row windows, N times (the first run also
//!                                                                compiles the kernels); FE_XE_PROFILE=1 reports the last run
//!   fe-xe <model> treelogits <ids.txt> <out.bin> <skip> <par,..> [--linear]  logits of one draft-tree verify window after
//!                                                                the prompt ids[0..skip] (--linear: each row's root path alone)
//! Kernels: FE_XE_KERNELS=ref runs on arc-engine's reference SPIR-V (build with --features ref-spv; test only, see kx.rs);
//! the Rust kernels are the default as kernels-xe delivers them.
use anyhow::{bail, ensure, Context as _, Result};
use fe_xe::config::Config;
use fe_xe::engine::{Engine, KEEP_HIDDEN, MAX_ROWS};
use fe_xe::eval;
use fe_xe::state::{self, KvMode};
use fe_xe::weights::Planner;
use fe_xq::Checkpoint;
use std::path::{Path, PathBuf};

fn flag<'a>(a: &'a [String], name: &str) -> Option<&'a str> {
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).map(String::as_str)
}

fn range(a: &[String], name: &str) -> Result<Option<(usize, usize)>> {
    let Some(v) = flag(a, name) else { return Ok(None) };
    let (x, y) = v.split_once(':').context("A:B")?;
    Ok(Some((x.parse()?, y.parse()?)))
}

/// This process's device memory from the DRM fdinfo of its render-node fds (xe: drm-total-vram* / drm-resident-vram*, KiB),
/// one count per drm-client-id: (total, resident) bytes.
fn vram_now() -> (u64, u64) {
    let mut seen = std::collections::HashSet::new();
    let (mut tot, mut res) = (0u64, 0u64);
    let Ok(dir) = std::fs::read_dir("/proc/self/fdinfo") else { return (0, 0) };
    for e in dir.flatten() {
        let Ok(t) = std::fs::read_to_string(e.path()) else { continue };
        let Some(id) = t.lines().find_map(|l| l.strip_prefix("drm-client-id:")).map(|v| v.trim().to_string()) else { continue };
        if !seen.insert(id) {
            continue;
        }
        for l in t.lines() {
            let kib = |l: &str| l.split(':').nth(1).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) * 1024;
            if l.starts_with("drm-total-vram") {
                tot += kib(l);
            } else if l.starts_with("drm-resident-vram") {
                res += kib(l);
            }
        }
    }
    (tot, res)
}

fn gb(b: usize) -> f64 {
    b as f64 / 1e9
}

fn info(model: &Path, a: &[String]) -> Result<()> {
    let cfg = Config::load(&Config::locate(model))?;
    let ck = Checkpoint::open(model)?;
    let ctx: usize = flag(a, "--ctx").map(str::parse).transpose()?.unwrap_or(131072);
    let kv = KvMode::parse(flag(a, "--kv").unwrap_or("q8")).context("--kv bf16|q8|q4")?;
    ensure!(ctx <= state::MAX_CTX, "ctx {ctx} > {}", state::MAX_CTX);
    let s = state::sizes(&cfg, ctx, kv, kv);
    let emb = cfg.vocab * cfg.hidden * 2;
    println!("model {model:?}: {} layers ({} attention), hidden {}, vocab {}", cfg.n_layers, cfg.n_full(), cfg.hidden, cfg.vocab);
    println!("  XQ projections (packed + su): {:.2} GB; embedding bf16 {:.2} GB", gb(ck.packed_bytes()), gb(emb));
    println!("  KV {kv:?} at ctx {ctx}: {:.2} GB ({:.1} MB a layer per K / V); MTP KV {:.2} GB", gb(s.kv_total), s.kv_layer as f64 / 1e6, gb(s.mtp_kv));
    println!("  GDN state {:.1} MB (= one prefix-cache checkpoint slot), {:.2} MB a layer", s.gdn_total as f64 / 1e6, s.gdn_layer as f64 / 1e6);
    println!("  total before scratch / prompt windows: {:.2} GB", gb(ck.packed_bytes() + emb + s.kv_total + s.mtp_kv + s.gdn_total));
    if a.iter().any(|x| x == "--prepare") {
        let p = Planner::new(&ck, &cfg);
        let t0 = std::time::Instant::now();
        let mut total = 0usize;
        for (_, t) in p.top()? {
            total += t.device_bytes();
        }
        for i in 0..cfg.n_layers {
            for (_, t) in p.layer(i)? {
                total += t.device_bytes();
            }
        }
        for (_, t) in p.mtp()? {
            total += t.device_bytes();
        }
        println!("  prepared every tensor on the host: {:.2} GB in {:.1} s", gb(total), t0.elapsed().as_secs_f64());
    }
    Ok(())
}

fn open(model: &Path, ctx: u32, rcap: u32) -> Result<Engine> {
    fe_xe::engine::open(model, ctx, rcap)
}

/// Runs `ids[0..need]` from position 0 in windows of `win` rows (heads on every row) and hands each row's fp32 logits
/// (row index = position) to `sink` for positions >= `first`.
fn run_rows(e: &mut Engine, ids: &[u32], need: usize, first: usize, win: u32, buf: &mut [f32], mut sink: impl FnMut(usize, &[f32]) -> Result<()>) -> Result<()> {
    e.reset()?;
    let v = e.cfg.vocab;
    let mut t = 0usize;
    while t < need {
        let nb = (win as usize).min(need - t);
        if nb > MAX_ROWS as usize {
            // prompt window: keep the normalised hidden rows, then the head in 16-row batches over the rows that are needed
            let want = t + nb > first;
            e.forward_rows(&ids[t..t + nb], if want { KEEP_HIDDEN } else { 0 }, true)?;
            if want {
                let mut h = first.saturating_sub(t);
                while h < nb {
                    let n = (MAX_ROWS as usize).min(nb - h);
                    e.head_batch(h as u32, n as u32)?;
                    e.fetch_logits(buf)?;
                    for j in 0..n {
                        sink(t + h + j, &buf[j * v..(j + 1) * v])?;
                    }
                    h += n;
                }
            }
            t += nb;
            continue;
        }
        let head = if t + nb > first { nb as u32 } else { 0 };
        e.forward_rows(&ids[t..t + nb], head, true)?;
        if head > 0 {
            e.fetch_logits(buf)?;
            for j in 0..nb {
                if t + j >= first {
                    sink(t + j, &buf[j * v..(j + 1) * v])?;
                }
            }
        }
        t += nb;
    }
    e.dev.sync()
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        bail!("usage: fe-xe <model> info|logits|fpkl|gen ... (see the source header)");
    }
    let model = PathBuf::from(&a[1]);
    let win: u32 = flag(&a, "--window").map(str::parse).transpose()?.unwrap_or(MAX_ROWS);
    ensure!((1..=4096).contains(&win), "--window {win}: 1..4096");
    match a[2].as_str() {
        "info" => info(&model, &a[3..]),
        "logits" => {
            ensure!(a.len() >= 7, "usage: fe-xe <model> logits <ids.txt> <out.bin> <skip> <rows> [--window W] [--f32] [--seqs A:B]");
            let seqs = eval::read_ids(Path::new(&a[3]))?;
            let (skip, rows): (usize, usize) = (a[5].parse()?, a[6].parse()?);
            let (s0, s1) = range(&a, "--seqs")?.unwrap_or((0, seqs.len()));
            let f32out = a.iter().any(|x| x == "--f32");
            let need = skip + rows;
            for (i, s) in seqs[s0..s1].iter().enumerate() {
                ensure!(s.len() >= need, "sequence {}: {} ids < skip + rows = {need}", s0 + i, s.len());
            }
            let mut e = open(&model, (need + 16) as u32, win)?;
            e.bf16_logits = a.iter().any(|x| x == "--bf16-logits");
            let mut out = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&a[4])?);
            let mut buf = vec![0f32; MAX_ROWS as usize * e.cfg.vocab];
            for (i, s) in seqs[s0..s1].iter().enumerate() {
                let t0 = std::time::Instant::now();
                run_rows(&mut e, s, need, skip, win, &mut buf, |_, lg| {
                    use std::io::Write;
                    if f32out {
                        out.write_all(unsafe { std::slice::from_raw_parts(lg.as_ptr() as *const u8, lg.len() * 4) })?;
                    } else {
                        let b: Vec<u8> = lg.iter().flat_map(|&v| eval::to_bf16(v).to_le_bytes()).collect();
                        out.write_all(&b)?;
                    }
                    Ok(())
                })?;
                eprintln!("seq {}: {need} positions in {:.1} s", s0 + i, t0.elapsed().as_secs_f64());
            }
            use std::io::Write;
            out.flush()?;
            eprintln!("fe-xe: {}", e.kx.origin_line());
            if let Some(r) = e.kx.profile_report()? {
                eprint!("{r}");
            }
            Ok(())
        }
        "fpkl" => {
            ensure!(a.len() >= 5, "usage: fe-xe <model> fpkl <IN.bin> <OUTDIR> [--rows A:B] [--window W]");
            let inp = eval::FpklIn::read(Path::new(&a[3]))?;
            let dir = PathBuf::from(&a[4]);
            let (r0, r1) = range(&a, "--rows")?.unwrap_or((0, inp.n));
            let maxlen = inp.lens[r0..r1.min(inp.n)].iter().copied().max().unwrap_or(0) as usize;
            let mut e = open(&model, (maxlen + 16) as u32, win)?;
            let mut buf = vec![0f32; MAX_ROWS as usize * e.cfg.vocab];
            let k = inp.k;
            for n in r0..r1.min(inp.n) {
                if dir.join(format!("row_{n}.bin")).exists() {
                    continue;
                }
                let len = inp.lens[n] as usize;
                let ids: Vec<u32> = inp.row_ids(n).iter().map(|&t| t as u32).collect();
                let mut recs = vec![0f32; len * (4 + k)];
                let t0 = std::time::Instant::now();
                let mut bad = 0u64;
                run_rows(&mut e, &ids, len, 0, win, &mut buf, |t, lg| {
                    bad += eval::fpkl_record(lg, inp.ref_ids(n, t), ids[t + 1] as usize, &mut recs[t * (4 + k)..(t + 1) * (4 + k)]) as u64;
                    Ok(())
                })?;
                eval::write_fpkl_row(&dir, n, len, k, win, &recs)?;
                eprintln!("row {n}: {len} positions in {:.1} s, nonfinite {bad}", t0.elapsed().as_secs_f64());
            }
            Ok(())
        }
        "treelogits" => {
            // treelogits <ids.txt> <out.bin> <skip> <par,par,..> [--linear] [--window W]: ids[0..skip] of the first sequence
            // as prompt windows, then ONE tree verify window of rows ids[skip..skip+m] (row t's token = ids[skip + t]) with
            // parents par (row 0 = the root); fp32 logits [m][vocab] to out.bin. --linear: every row decoded alone along its
            // root path instead (fresh prefill a row, the path as one linear window): must equal the tree's bit for bit.
            ensure!(a.len() >= 7, "usage: fe-xe <model> treelogits <ids.txt> <out.bin> <skip> <par,..> [--linear] [--window W]");
            let seqs = eval::read_ids(Path::new(&a[3]))?;
            let skip: usize = a[5].parse()?;
            let par: Vec<i32> = a[6].split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
            let m = par.len();
            let ids = &seqs[0];
            ensure!(skip + m <= ids.len(), "sequence too short");
            let mut e = open(&model, (skip + m + 64) as u32, win.max(MAX_ROWS))?;
            let v = e.cfg.vocab;
            let mut buf = vec![0f32; MAX_ROWS as usize * v];
            let mut out = vec![0f32; m * v];
            let prefill = |e: &mut Engine| -> Result<()> {
                e.reset()?;
                let mut t = 0;
                while t < skip {
                    let nb = (win as usize).min(skip - t);
                    e.forward_rows(&ids[t..t + nb], 0, true)?;
                    t += nb;
                }
                Ok(())
            };
            let toks: Vec<u32> = ids[skip..skip + m].to_vec();
            if a.iter().any(|x| x == "--linear") {
                let tt = fe_kxcore_tree_paths(&par);
                for (r, path) in tt.iter().enumerate() {
                    prefill(&mut e)?;
                    let pt: Vec<u32> = path.iter().map(|&p| toks[p]).collect();
                    e.forward_rows(&pt, pt.len() as u32, false)?;
                    e.fetch_logits(&mut buf)?;
                    let l = pt.len() - 1;
                    out[r * v..(r + 1) * v].copy_from_slice(&buf[l * v..(l + 1) * v]);
                }
            } else {
                prefill(&mut e)?;
                e.forward_tree(&toks, &par)?;
                e.fetch_logits(&mut buf)?;
                out.copy_from_slice(&buf[..m * v]);
            }
            e.dev.sync()?;
            let bytes: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(&a[4], bytes)?;
            eprintln!("treelogits: {m} rows, kernels {}", e.kx.origin_line());
            Ok(())
        }
        "gdnpregate" => {
            // gdnpregate <ids.txt> <outdir> [--n N]: the FE_XE_GDNPRE gate in one process (one model load). The same windows with
            // Engine::gdnpre on and off must give the same logits bit for bit: w16 (1024 + 32 rows in 16-row windows), w16-0 (64
            // rows), dec0 (64 one-row steps), verify windows (8 rows, not in place, commits of 3 / 8 / 1 / 5 rows: the replays),
            // three tree windows after a 1024-token prompt plus a tree commit and the window after it; then speculative greedy
            // decode of N tokens (ids on == off == plain greedy). Each tree row against its root path run as a linear window is
            // reported too (informational: the GEMV row count differs). Logits go to <outdir>/<name>-{on,off}.bin (fp32).
            ensure!(a.len() >= 5, "usage: fe-xe <model> gdnpregate <ids.txt> <outdir> [--n N]");
            let seqs = eval::read_ids(Path::new(&a[3]))?;
            let dir = PathBuf::from(&a[4]);
            std::fs::create_dir_all(&dir)?;
            let n: usize = flag(&a, "--n").map(str::parse).transpose()?.unwrap_or(500);
            let ids = &seqs[0];
            ensure!(ids.len() >= 1100, "first sequence < 1100 ids");
            let mut e = open(&model, (1024 + n + 64).max(1100) as u32, 4096)?;
            ensure!(e.gdnpre, "gdn_pre_r / gdn_pre_tr not in the kernels (or FE_XE_GDNPRE=0): nothing to gate");
            let v = e.cfg.vocab;
            let mut buf = vec![0f32; MAX_ROWS as usize * v];
            let mut bad = 0usize;
            let mut cmp = |name: &str, on: &[f32], off: &[f32]| -> Result<()> {
                let b = |x: &[f32]| -> Vec<u8> { x.iter().flat_map(|y| y.to_le_bytes()).collect() };
                std::fs::write(dir.join(format!("{name}-on.bin")), b(on))?;
                std::fs::write(dir.join(format!("{name}-off.bin")), b(off))?;
                let same = on.len() == off.len() && on.iter().zip(off).all(|(x, y)| x.to_bits() == y.to_bits());
                if !same {
                    bad += 1;
                }
                println!("{name}: {} rows, gdn_pre {} the old chain", on.len() / v, if same { "IDENTICAL to" } else { "DIFFERS from" });
                Ok(())
            };
            let trees: [(&str, Vec<i32>); 3] = [
                ("tree8", vec![-1, 0, 0, 1, 1, 2, 2, 3]),
                ("chain16", (0..16).map(|t| t - 1).collect()),
                ("tree16", vec![-1, 0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            ];
            let mut res: Vec<Vec<Vec<f32>>> = Vec::new();
            for on in [true, false] {
                e.gdnpre = on;
                let mut r = Vec::new();
                for (need, first, w) in [(1056usize, 1024usize, 16u32), (64, 0, 16), (64, 0, 1)] {
                    let mut out = Vec::new();
                    run_rows(&mut e, ids, need, first, w, &mut buf, |_, lg| {
                        out.extend_from_slice(lg);
                        Ok(())
                    })?;
                    r.push(out);
                }
                // verify windows: 8 rows, not in place, then keep c rows
                let mut out = Vec::new();
                e.reset()?;
                e.forward_rows(&ids[..1024], 0, true)?;
                let mut t = 1024;
                for c in [3u32, 8, 1, 5] {
                    e.forward_rows(&ids[t..t + 8], 8, false)?;
                    e.fetch_logits(&mut buf)?;
                    out.extend_from_slice(&buf[..8 * v]);
                    e.commit(c)?;
                    t += c as usize;
                }
                e.forward_rows(&ids[t..t + 8], 8, false)?;
                e.fetch_logits(&mut buf)?;
                out.extend_from_slice(&buf[..8 * v]);
                r.push(out);
                for (_, par) in &trees {
                    let m = par.len();
                    e.reset()?;
                    e.forward_rows(&ids[..1024], 0, true)?;
                    e.forward_tree(&ids[1024..1024 + m], par)?;
                    e.fetch_logits(&mut buf)?;
                    let mut out = buf[..m * v].to_vec();
                    // tree commit along row 7's / the last row's root path, then a linear window after it
                    let paths = fe_kxcore_tree_paths(par);
                    let path = &paths[7.min(m - 1)];
                    e.commit_tree(path)?;
                    let t = 1024 + path.len();
                    e.forward_rows(&ids[t..t + 6], 6, false)?;
                    e.fetch_logits(&mut buf)?;
                    out.extend_from_slice(&buf[..6 * v]);
                    r.push(out);
                }
                res.push(r);
            }
            for (i, name) in ["w16", "w16-0", "dec0", "verify", "tree8", "chain16", "tree16"].iter().enumerate() {
                cmp(name, &res[0][i], &res[1][i])?;
            }
            // tree rows vs their root paths as linear windows (gdn_pre on)
            e.gdnpre = true;
            for (ti, (name, par)) in trees.iter().enumerate() {
                let m = par.len();
                e.reset()?;
                e.forward_rows(&ids[..1024], 0, true)?;
                let toks = &ids[1024..1024 + m];
                let (mut same, mut am, mut maxd) = (0, 0, 0f32);
                for (r, path) in fe_kxcore_tree_paths(par).iter().enumerate() {
                    let pt: Vec<u32> = path.iter().map(|&p| toks[p]).collect();
                    e.forward_rows(&pt, pt.len() as u32, false)?;
                    e.fetch_logits(&mut buf)?;
                    let l = pt.len() - 1;
                    let lin = &buf[l * v..(l + 1) * v];
                    let tr = &res[0][4 + ti][r * v..(r + 1) * v];
                    same += lin.iter().zip(tr).all(|(x, y)| x.to_bits() == y.to_bits()) as usize;
                    let arg = |x: &[f32]| x.iter().enumerate().fold((0, f32::MIN), |b, (i, &y)| if y > b.1 { (i, y) } else { b }).0;
                    am += (arg(lin) == arg(tr)) as usize;
                    maxd = lin.iter().zip(tr).fold(maxd, |d, (x, y)| d.max((x - y).abs()));
                }
                println!("{name} vs root paths as linear windows: {same}/{m} rows bitwise, argmax {am}/{m}, max |diff| {maxd:.3e}");
            }
            // speculative greedy decode: ids with gdn_pre on == off == plain greedy
            let ck = Checkpoint::open(&model)?;
            let dh = std::env::var("MTP_DHEAD").map_err(|_| anyhow::anyhow!("MTP_DHEAD is not set: the DHQ8 draft head file (dhead_q8.bin)"))?;
            let mut mt = fe_xe::mtp::Mtp::load(&e, &ck, Some(&dh))?;
            let cfg = fe_xe::specgen::SpecCfg { rows: 4096, ..Default::default() };
            let prompt = &ids[..1024];
            let mut outs = Vec::new();
            for (on, k0) in [(true, false), (false, false), (true, true)] {
                e.gdnpre = on;
                let mut c = cfg.clone();
                if k0 {
                    c.k = 0;
                }
                let mut out = Vec::new();
                let st = fe_xe::specgen::generate(&mut e, &mut mt, prompt, n, &c, &[], |t| {
                    out.extend_from_slice(t);
                    true
                })?;
                println!("{}: {} tokens, {:.1} tok/s, windows {} accepted {}/{}", if k0 { "plain greedy" } else if on { "spec gdn_pre on" } else { "spec gdn_pre off" },
                    st.tokens, (st.tokens - 1) as f64 / st.gen_s, st.windows, st.accepted, st.drafted);
                outs.push(out);
            }
            let ok = outs[0] == outs[1] && outs[0] == outs[2];
            println!("spec ids: on {} off, on {} plain greedy", if outs[0] == outs[1] { "==" } else { "!=" }, if outs[0] == outs[2] { "==" } else { "!=" });
            if !ok {
                bad += 1;
            }
            eprintln!("fe-xe: {}", e.kx.origin_line());
            println!("{}", if bad == 0 { "GDNPRE-GATE PASS" } else { "GDNPRE-GATE FAIL" });
            Ok(())
        }
        "gen" => {
            ensure!(a.len() >= 5, "usage: fe-xe <model> gen <ids.txt> <n> [--window W]");
            let seqs = eval::read_ids(Path::new(&a[3]))?;
            let n: usize = a[4].parse()?;
            let prompt = &seqs[0];
            let mut e = open(&model, (prompt.len() + n + 16) as u32, win)?;
            e.reset()?;
            let mut t = 0;
            let mut am = [0i32; MAX_ROWS as usize];
            while t < prompt.len() {
                let nb = (win as usize).min(prompt.len() - t);
                e.forward_rows(&prompt[t..t + nb], if t + nb == prompt.len() { 1 } else { 0 }, true)?;
                t += nb;
            }
            e.argmax(&mut am)?;
            e.kx.profile_reset()?;
            let mut out = vec![am[0] as u32];
            let t0 = std::time::Instant::now();
            while out.len() < n {
                e.forward_rows(&[*out.last().unwrap()], 1, true)?;
                e.argmax(&mut am)?;
                out.push(am[0] as u32);
            }
            let dt = t0.elapsed().as_secs_f64();
            println!("generated ids: {}", out.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
            eprintln!("{} tokens after the prompt in {dt:.2} s: {:.1} tok/s (plain, 1-row windows)", n - 1, (n - 1) as f64 / dt);
            eprintln!("fe-xe: {}", e.kx.origin_line());
            if let Some(r) = e.kx.profile_report()? {
                eprint!("{r}");
            }
            Ok(())
        }
        "pfbench" => {
            ensure!(a.len() >= 4, "usage: fe-xe <model> pfbench <ids.txt> [--len L] [--reps N] [--window W]");
            let seqs = eval::read_ids(Path::new(&a[3]))?;
            let len: usize = flag(&a, "--len").map(str::parse).transpose()?.unwrap_or(seqs[0].len());
            let reps: usize = flag(&a, "--reps").map(str::parse).transpose()?.unwrap_or(3);
            let prompt: Vec<u32> = seqs.iter().flatten().copied().take(len).collect();
            ensure!(prompt.len() == len, "only {} ids in {}", prompt.len(), a[3]);
            let mut e = open(&model, (len + 16) as u32, win)?;
            let mut am = [0i32; MAX_ROWS as usize];
            for rep in 0..reps {
                e.reset()?;
                if rep + 1 == reps {
                    e.kx.profile_reset()?;
                }
                let t0 = std::time::Instant::now();
                let mut t = 0;
                while t < len {
                    let nb = (win as usize).min(len - t);
                    e.forward_rows(&prompt[t..t + nb], if t + nb == len { 1 } else { 0 }, true)?;
                    t += nb;
                }
                e.argmax(&mut am)?;
                let dt = t0.elapsed().as_secs_f64();
                let (vt, vr) = vram_now();
                eprintln!("pfbench rep {rep}: {len} tokens in {dt:.3} s = {:.0} tok/s (window {win}, next id {}), VRAM total {:.2} GB resident {:.2} GB",
                    len as f64 / dt, am[0], vt as f64 / 1e9, vr as f64 / 1e9);
            }
            eprintln!("fe-xe: {}", e.kx.origin_line());
            if let Some(r) = e.kx.profile_report()? {
                eprint!("{r}");
            }
            Ok(())
        }
        "spec" => {
            // speculative greedy generation (arc-engine policy) after the first sequence; --check: plain greedy too, ids must match
            ensure!(a.len() >= 5, "usage: fe-xe <model> spec <ids.txt> <n> [--window W] [--check] [--k K] [--no-hybrid] [--no-adaptive]");
            let seqs = eval::read_ids(Path::new(&a[3]))?;
            let n: usize = a[4].parse()?;
            let prompt = &seqs[0];
            let ctx = (prompt.len() + n + 64) as u32;
            let mut e = open(&model, ctx, win.max(MAX_ROWS))?;
            let ck = Checkpoint::open(&model)?;
            let dh = std::env::var("MTP_DHEAD").map_err(|_| anyhow::anyhow!("MTP_DHEAD is not set: the DHQ8 draft head file (dhead_q8.bin)"))?;
            let mut mt = fe_xe::mtp::Mtp::load(&e, &ck, Some(&dh))?;
            let mut cfg = fe_xe::specgen::SpecCfg { rows: win as usize, ..Default::default() };
            if let Some(k) = flag(&a, "--k") {
                cfg.k = k.parse()?;
            }
            cfg.hybrid = !a.iter().any(|x| x == "--no-hybrid");
            cfg.adaptive = !a.iter().any(|x| x == "--no-adaptive");
            // (generate() by hand: the profile, if on, restarts after the prompt)
            ensure!(prompt.len() >= 2, "prompt too short");
            e.reset()?;
            let mut sp = fe_xe::specgen::Spec::new(cfg.clone());
            let first = sp.prefill(&mut e, &mut mt, prompt, 0, &[], &mut |_, _| Ok(()), None)?;
            e.kx.profile_reset()?;
            let mut out = vec![first];
            while sp.st.tokens < n {
                let (kept, send, hit) = sp.step(&mut e, &mut mt, n - sp.st.tokens, &[], None)?;
                out.extend_from_slice(&kept[..send]);
                if hit {
                    break;
                }
            }
            e.dev.sync()?;
            let st = sp.st.clone();
            let prof = e.kx.profile_report()?;
            println!("spec ids: {}", out.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
            eprintln!("prompt {} in {:.2} s ({:.0} tok/s incl. MTP absorb); {} tokens in {:.2} s = {:.1} tok/s; windows {} plain {} drafted {} accepted {} ({:.0} %), copies {} vetoed {}",
                st.prompt, st.prefill_s, st.prompt as f64 / st.prefill_s, st.tokens, st.gen_s, (st.tokens - 1) as f64 / st.gen_s,
                st.windows, st.plain, st.drafted, st.accepted, 100.0 * st.accepted as f64 / st.drafted.max(1) as f64, st.copies, st.vetoed);
            if a.iter().any(|x| x == "--check") {
                let mut plain = Vec::new();
                let mut cfg0 = cfg.clone();
                cfg0.k = 0;
                fe_xe::specgen::generate(&mut e, &mut mt, prompt, n, &cfg0, &[], |t| {
                    plain.extend_from_slice(t);
                    true
                })?;
                if plain == out {
                    println!("LOSSLESS: speculative ids == plain greedy ids ({} tokens)", out.len());
                } else {
                    let i = plain.iter().zip(&out).position(|(x, y)| x != y).unwrap_or(plain.len().min(out.len()));
                    println!("DIFFERS at token {i}: plain {:?} spec {:?}", &plain[i..(i + 5).min(plain.len())], &out[i..(i + 5).min(out.len())]);
                }
            }
            eprintln!("fe-xe: {}", e.kx.origin_line());
            if let Some(r) = prof {
                eprint!("decode (verify windows + MTP) {r}");
            }
            Ok(())
        }
        c => bail!("unknown command {c} (info | logits | fpkl | gen | spec | pfbench)"),
    }
}

/// Root paths of a tree given by its parents (par[0] = -1, par[t] < t).
fn fe_kxcore_tree_paths(par: &[i32]) -> Vec<Vec<usize>> {
    (0..par.len())
        .map(|t| {
            let mut p = vec![t];
            let mut x = t;
            while par[x] >= 0 {
                x = par[x] as usize;
                p.push(x);
            }
            p.reverse();
            p
        })
        .collect()
}
