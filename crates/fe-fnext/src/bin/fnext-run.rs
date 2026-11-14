//! Fast decode engine driver:
//!   fnext-run greedy <a|b>              teacher-forced vs the REF greedy stream (argmax agreement, top-20 KL)
//!   fnext-run bench <prompt.txt> [n]    prefill + n greedy tokens: tok/s, expert hit rate, CPU miss time
//!   fnext-run counts <out.json> <files..>  routing counts over text files (no experts on the card)
//! env: FNEXT_COUNTS (counts json for the static expert set), FNEXT_EXPERT_MIB, FNEXT_CTX, FNEXT_THREADS,
//!      FNEXT_TOKENIZER (tokenizer.json)
use anyhow::{anyhow, Result};
use fe_fnext::fast::{load_counts, Fast, Spec};
use serde_json::Value;
use std::time::Instant;

/// Reference dumps (named tensors, prompts, greedy streams): FNEXT_REF, default `ref`.
static REF: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| std::env::var("FNEXT_REF").unwrap_or_else(|_| "ref".into()));

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn tokenizer() -> Result<tokenizers::Tokenizer> {
    // FNEXT_TOKENIZER, else tokenizer.json next to the model (scripts/fetch-flashnext.sh puts it there)
    let p = std::env::var("FNEXT_TOKENIZER").unwrap_or_else(|_| {
        fe_fnext::default_model().with_file_name("tokenizer.json").to_string_lossy().into_owned()
    });
    tokenizers::Tokenizer::from_file(&p).map_err(|e| anyhow!("tokenizer {p}: {e}"))
}

fn load(counts: bool) -> Result<Fast> {
    let c = if counts { std::env::var("FNEXT_COUNTS").ok().map(|p| load_counts(&p)).transpose()? } else { None };
    let mib = std::env::var("FNEXT_EXPERT_MIB").ok().and_then(|v| v.parse().ok());
    Fast::load(&fe_fnext::default_model(), env("FNEXT_CTX", 32768), c.as_ref(), mib, env("FNEXT_THREADS", 8))
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
    let ls = s.ln() + m;
    v.iter().map(|&x| x as f64 - ls).collect()
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("greedy") => greedy(a.get(2).map(String::as_str).unwrap_or("a")),
        Some("bench") => bench(&a[2], a.get(3).and_then(|v| v.parse().ok()).unwrap_or(128)),
        Some("counts") => counts(&a[2], &a[3..]),
        Some("mtp") => mtp_bench(&a[2], a.get(3).and_then(|v| v.parse().ok()).unwrap_or(256),
            a.get(4).and_then(|v| v.parse().ok()).unwrap_or(3)),
        Some("pp") => pp(&a[2], &a[3..]),
        Some("ctx") => ctxbench(&a[2], a[3].parse()?, a.get(4).and_then(|v| v.parse().ok()).unwrap_or(200)),
        Some("vs") => vs_truth(a.get(2).map(String::as_str).unwrap_or("a")),
        Some("cache") => cache_test(&a[2], a.get(3).and_then(|v| v.parse().ok()).unwrap_or(48)),
        Some("xfer") => xfer(),
        Some("arbench") => arbench(a.get(2).and_then(|v| v.parse().ok()).unwrap_or(1), a.get(3).and_then(|v| v.parse().ok()).unwrap_or(96)),
        Some("plegather") => {
            // the prompt PLE gather alone: n random tokens of corpus/long.txt-like ids (timing of the host reads)
            let n: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(16384);
            let g = fe_fnext::open_model(&fe_fnext::default_model())?;
            let c = fe_fnext::config::Config::from_gguf(&g)?;
            let mut x = 0x9e3779b97f4a7c15u64 ^ std::process::id() as u64;
            let toks: Vec<u32> = (0..n).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((x >> 33) % 150000) as u32 }).collect();
            let idx = fe_fnext::engine::ple_rows(&c, &[], 0, &toks);
            let pt = g.tensor("per_layer_token_embd.weight").unwrap();
            for _ in 0..2 {
                let t0 = std::time::Instant::now();
                let v = fe_fnext::fast::hostmem::ple_gather(&g, pt, &idx, 4096);
                println!("plegather {} rows: {:.0} ms ({})", idx.len(), t0.elapsed().as_secs_f64() * 1e3, v.len());
            }
            Ok(())
        }
        Some("rowinv") => row_inv(a.get(2).map(String::as_str).unwrap_or("b"), a.get(3).and_then(|v| v.parse().ok()).unwrap_or(24)),
        Some("park") => park_test(&a[2], a.get(3).and_then(|v| v.parse().ok()).unwrap_or(48),
            a.get(4).and_then(|v| v.parse().ok()).unwrap_or(0)),
        Some("lbench") => lanes_bench(&a[2], a.get(3).and_then(|v| v.parse().ok()).unwrap_or(256), &a[4..]),
        Some("lanesm") => lanes_mtp(a.get(2).and_then(|v| v.parse().ok()).unwrap_or(64), &a[3..]),
        Some("lanesn") => lanes_n(a.get(2).and_then(|v| v.parse().ok()).unwrap_or(64), &a[3..]),
        Some("lanes") => lanes_test(a.get(2).map(String::as_str).unwrap_or("a"), a.get(3).map(String::as_str).unwrap_or("b"),
            a.get(4).and_then(|v| v.parse().ok()).unwrap_or(48)),
        Some("det") => det(a.get(2).map(String::as_str).unwrap_or("a"), a.get(3).and_then(|v| v.parse().ok()).unwrap_or(64)),
        Some("lossless") => lossless(a.get(2).map(String::as_str).unwrap_or("a"), a.get(3).and_then(|v| v.parse().ok()).unwrap_or(64)),
        Some("spec") => spec(a.get(2).map(String::as_str).unwrap_or("a"), a.get(3).and_then(|v| v.parse().ok()).unwrap_or(4)),
        Some("gencounts") => gencounts(&a[2], &a[3], a.get(4).and_then(|v| v.parse().ok()).unwrap_or(256)),
        _ => Err(anyhow!("usage: fnext-run greedy|bench|counts ...")),
    }
}

/// fnext-run xfer (two visible cards): device 0 -> device 1 copy of residual-sized blocks, pinned host staging
/// (DtoH on card 0, HtoD on card 1) vs cuMemcpyPeerAsync; mean of 200 (small) / 5 (large) copies.
fn xfer() -> Result<()> {
    use fe_cuda::{Context, DeviceBuffer, HostBuffer, Stream};
    let c0 = Context::new(0)?;
    let c1 = Context::new(1)?;
    for bytes in [40960usize, 8 * 40960, 64 << 20, 671 << 20] {
        let n = bytes / 4;
        c0.bind()?;
        let a = DeviceBuffer::<f32>::zeroed(n)?;
        let s0 = Stream::new()?;
        c1.bind()?;
        let b = DeviceBuffer::<f32>::zeroed(n)?;
        let s1 = Stream::new()?;
        let mut h = HostBuffer::<f32>::alloc(n)?;
        let reps = if bytes < (1 << 20) { 200 } else { 5 };
        let t = Instant::now();
        for _ in 0..reps {
            c0.bind()?;
            unsafe { s0.memcpy_dtoh(h.as_mut_slice(), a.ptr())? };
            s0.sync()?;
            c1.bind()?;
            unsafe { s1.memcpy_htod(b.ptr(), h.as_slice())? };
            s1.sync()?;
        }
        let staged = t.elapsed().as_secs_f64() / reps as f64;
        c1.bind()?;
        let t = Instant::now();
        for _ in 0..reps {
            c1.copy_peer(b.ptr(), &c0, a.ptr(), bytes, &s1)?;
            s1.sync()?;
        }
        let peer = t.elapsed().as_secs_f64() / reps as f64;
        println!("xfer {:>10} B: staged {:8.1} us ({:5.2} GB/s), peer {:8.1} us ({:5.2} GB/s)", bytes, staged * 1e6,
            bytes as f64 / staged / 1e9, peer * 1e6, bytes as f64 / peer / 1e9);
        c0.bind()?;
        drop(a);
        drop(s0);
        c1.bind()?;
    }
    Ok(())
}

/// Two-card all-reduce latency through pinned host memory (no peer access): K reduction points of rows x 2560 f32 in
/// one graph per card, both graphs launched together. Variants: memop (put kernel + stream write/wait value + sum
/// kernel), fused (one kernel: put, fence.sys, flag, spin, sum), copy (DtoH copy + write/wait value + HtoD copy + sum),
/// local (the sum kernels alone, no exchange).
fn arbench(rows: usize, k: usize) -> Result<()> {
    use fe_cuda::{Context, DeviceBuffer, HostBuffer, Module, Stream};
    let n = rows * 2560;
    let cus = [Context::new(0)?, Context::new(1)?];
    let wc = std::env::var("AR_WC").is_ok();
    let hs = if wc { HostBuffer::<f32>::alloc_wc(2 * k * n * 2)? } else { HostBuffer::<f32>::alloc(2 * k * n * 2)? };
    let fl = if wc { HostBuffer::<u32>::alloc_wc(2 * k * 64)? } else { HostBuffer::<u32>::alloc(2 * k * 64)? };
    struct Card { _m: Module, put: fe_cuda::Function, sum: fe_cuda::Function, ar2: fe_cuda::Function, ll: fe_cuda::Function,
        ping: fe_cuda::Function, probe: fe_cuda::Function, sumw: fe_cuda::Function, pack: fe_cuda::Function,
        llsum: fe_cuda::Function, pk: DeviceBuffer<u32>, s: Stream,
        x: DeviceBuffer<f32>, y: DeviceBuffer<f32>, o: DeviceBuffer<f32>, tag: DeviceBuffer<u32> }
    let mut cards = vec![];
    for (ci, cu) in cus.iter().enumerate() {
        cu.bind()?;
        let m = Module::load(fe_fnext::K_FNEXT_CUBIN)?;
        let xv: Vec<f32> = (0..n).map(|i| ((i * 7 + ci * 13) % 1000) as f32 * 0.001 + ci as f32).collect();
        cards.push(Card { put: m.function("fx_ar_put")?, sum: m.function("fx_ar_sum")?, ar2: m.function("fx_ar2")?,
            ll: m.function("fx_ar_ll")?, ping: m.function("fx_ar_ping")?, probe: m.function("fx_ar_probe")?,
            sumw: m.function("fx_ar_sumw")?, pack: m.function("fx_ll_pack")?, llsum: m.function("fx_ll_sum")?,
            pk: DeviceBuffer::zeroed(2 * n)?,
            _m: m, s: Stream::new()?, x: DeviceBuffer::from_slice(&xv)?, y: DeviceBuffer::zeroed(n)?,
            o: DeviceBuffer::zeroed(n)?, tag: DeviceBuffer::from_slice(&[1u32])? });
    }
    let g = (n as u32).div_ceil(1024);
    let variants: Vec<String> = std::env::var("AR_VARIANTS").unwrap_or("local,memop,fused,copy,ll,ping".into())
        .split(',').map(String::from).collect();
    for variant in variants.iter().map(String::as_str) {
        let mut graphs = vec![];
        for ci in 0..2 {
            cus[ci].bind()?;
            let cd = &cards[ci];
            let s = &cd.s;
            s.begin_capture()?;
            for j in 0..k {
                let hm = hs.dev().add::<f32>((ci * k + j) * n);
                let ho = hs.dev().add::<f32>(((1 - ci) * k + j) * n);
                let fm = fl.dev().add::<u32>((ci * k + j) * 64);
                let fo = fl.dev().add::<u32>(((1 - ci) * k + j) * 64);
                unsafe {
                    match variant {
                        "local" => s.launch(&cd.sum, g, 256, 0, (cd.y.ptr(), cd.x.ptr(), cd.o.ptr(), n as u32))?,
                        "memop" => {
                            s.launch(&cd.put, g, 256, 0, (hm, cd.x.ptr(), n as u32))?;
                            s.write_value32(fm, 1)?;
                            s.wait_value32(fo, 1, true)?;
                            s.launch(&cd.sum, g, 256, 0, (cd.y.ptr(), cd.x.ptr(), ho, n as u32))?;
                        }
                        "fused" => s.launch(&cd.ar2, g, 256, 0, (cd.y.ptr(), cd.x.ptr(), hm, ho, fm, fo, n as u32, cd.tag.ptr()))?,
                        "ll" => {
                            let hm = hs.dev().add::<f32>((ci * k + j) * 2 * n);
                            let ho = hs.dev().add::<f32>(((1 - ci) * k + j) * 2 * n);
                            s.launch(&cd.ll, (n as u32).div_ceil(512), 256, 0, (cd.y.ptr(), cd.x.ptr(), hm, ho, n as u32, cd.tag.ptr(), j as u32 + 1))?
                        }
                        "ping" => s.launch(&cd.ping, 1, 32, 0, (fm, fo, cd.tag.ptr(), j as u32 + 1))?,
                        v if v.starts_with("probe") => {
                            let mode: u32 = v[5..].parse().unwrap();
                            s.launch(&cd.probe, g, 256, 0, (cd.o.ptr(), cd.x.ptr(), hm, n as u32, mode))?
                        }
                        "ce1" | "ce2" => {
                            let hsl = std::slice::from_raw_parts_mut((hs.as_ptr() as *mut f32).add((ci * k + j) * n), n);
                            s.memcpy_dtoh(hsl, cd.x.ptr())?;
                            s.write_value32(fm, 1)?;
                            if variant == "ce1" {
                                s.wait_value32(fo, 1, true)?;
                                s.launch(&cd.sum, g, 256, 0, (cd.y.ptr(), cd.x.ptr(), ho, n as u32))?;
                            } else {
                                s.launch(&cd.sumw, g, 256, 0, (cd.y.ptr(), cd.x.ptr(), ho, n as u32, fo, 1u32))?;
                            }
                        }
                        "cell" => {
                            let hm = hs.dev().add::<f32>((ci * k + j) * 2 * n);
                            let ho = hs.dev().add::<f32>(((1 - ci) * k + j) * 2 * n);
                            s.launch(&cd.pack, (n as u32).div_ceil(256), 256, 0, (cd.pk.ptr(), cd.x.ptr(), n as u32, cd.tag.ptr(), j as u32 + 1))?;
                            let hsl = std::slice::from_raw_parts_mut((hs.as_ptr() as *mut u32).add((ci * k + j) * 2 * n), 2 * n);
                            let _ = hm;
                            s.memcpy_dtoh(hsl, cd.pk.ptr())?;
                            s.launch(&cd.llsum, (n as u32).div_ceil(512), 256, 0, (cd.y.ptr(), cd.x.ptr(), ho, n as u32, cd.tag.ptr(), j as u32 + 1))?;
                        }
                        "dtoh" => {
                            let hsl = std::slice::from_raw_parts_mut((hs.as_ptr() as *mut f32).add((ci * k + j) * n), n);
                            s.memcpy_dtoh(hsl, cd.x.ptr())?;
                        }
                        "htod" => {
                            let osl = std::slice::from_raw_parts((hs.as_ptr()).add(((1 - ci) * k + j) * n), n);
                            s.memcpy_htod(cd.o.ptr(), osl)?;
                        }
                        _ => {
                            let hsl = std::slice::from_raw_parts_mut((hs.as_ptr() as *mut f32).add((ci * k + j) * n), n);
                            s.memcpy_dtoh(hsl, cd.x.ptr())?;
                            s.write_value32(fm, 1)?;
                            s.wait_value32(fo, 1, true)?;
                            let osl = std::slice::from_raw_parts((hs.as_ptr()).add(((1 - ci) * k + j) * n), n);
                            s.memcpy_htod(cd.o.ptr(), osl)?;
                            s.launch(&cd.sum, g, 256, 0, (cd.y.ptr(), cd.x.ptr(), cd.o.ptr(), n as u32))?;
                        }
                    }
                }
            }
            graphs.push(s.end_capture()?);
        }
        let mut times = vec![];
        for it in 0..23 {
            unsafe { std::ptr::write_bytes(fl.as_ptr() as *mut u32, 0, 2 * k * 64) };
            unsafe { std::ptr::write_bytes(hs.as_ptr() as *mut f32, 0, 2 * k * n * 2) };
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            let t = Instant::now();
            for ci in 0..2 {
                cus[ci].bind()?;
                graphs[ci].launch(&cards[ci].s)?;
            }
            for ci in 0..2 {
                cus[ci].bind()?;
                cards[ci].s.sync()?;
            }
            if it >= 3 {
                times.push(t.elapsed().as_secs_f64());
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = times[times.len() / 2];
        let mut ok = true;
        if ["memop", "fused", "copy", "ll", "ce1", "ce2", "cell"].contains(&variant) {
            cus[0].bind()?;
            let y0 = cards[0].y.to_vec()?;
            let x0 = cards[0].x.to_vec()?;
            cus[1].bind()?;
            let y1 = cards[1].y.to_vec()?;
            let x1 = cards[1].x.to_vec()?;
            ok = (0..n).all(|i| y0[i].to_bits() == y1[i].to_bits() && y0[i] == x0[i] + x1[i]);
        }
        println!("arbench {variant:>6}: rows {rows} ({} B), {k} reductions: {:.1} us per reduction (graph pair {:.3} ms), results {}",
            n * 4, med * 1e6 / k as f64, med * 1e3, if ok { "OK (identical on both cards)" } else { "WRONG" });
        drop(graphs);
    }
    Ok(())
}

fn greedy(which: &str) -> Result<()> {
    let f = match which { "a" => "a_short", "b" => "b_code600", "c" => "c_long5000", _ => "d_long20k" };
    let toks: Vec<u32> = serde_json::from_str::<Vec<u32>>(&std::fs::read_to_string(format!("{}/prompts/{f}.tokens.json", &*REF))?)?;
    let g: Value = serde_json::from_str(&String::from_utf8_lossy(&std::fs::read(format!("{}/greedy_{which}.json", &*REF))?))?;
    let steps = g["steps"].as_array().unwrap();
    let mut m = load(true)?;
    let t0 = Instant::now();
    let mut tok = m.prefill(&toks)?;
    let tp = t0.elapsed().as_secs_f64();
    if let Ok(path) = std::env::var("FNEXT_DUMP_LOGITS") {
        let lg = m.logits();
        let b: Vec<u8> = lg.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, b)?;
        eprintln!("prefill logits -> {path}");
    }
    let (mut agree, mut kls, mut klm, mut first) = (0, 0f64, 0f64, None);
    let t1 = Instant::now();
    for (i, s) in steps.iter().enumerate() {
        let rt = s["token"].as_u64().unwrap() as u32;
        let lg = m.logits();
        let top = s["top"].as_array().unwrap();
        let ids: Vec<usize> = top.iter().map(|t| t["id"].as_u64().unwrap() as usize).collect();
        let rl: Vec<f32> = top.iter().map(|t| t["logit"].as_f64().unwrap() as f32).collect();
        let ml: Vec<f32> = ids.iter().map(|&j| lg[j]).collect();
        let (lp, lq) = (log_softmax(&rl), log_softmax(&ml));
        let kl: f64 = lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum();
        kls += kl;
        klm = klm.max(kl);
        if tok == rt { agree += 1 } else if first.is_none() { first = Some(i) }
        if i + 1 < steps.len() {
            tok = m.step(&[rt])?;
        }
    }
    let n = steps.len();
    let st = m.stats.lock().unwrap().clone();
    println!("{which}: teacher-forced argmax {agree}/{n}, top-20 KL mean {:.3e} max {:.3e}, stream identical for {} tokens",
        kls / n as f64, klm, first.unwrap_or(n));
    println!("prefill {} tok in {:.2} s ({:.1} tok/s); decode {:.1} tok/s; expert hit rate {:.1}% ({} experts on card)",
        toks.len(), tp, toks.len() as f64 / tp, (n - 1) as f64 / t1.elapsed().as_secs_f64(),
        100.0 * (1.0 - st.misses as f64 / st.pairs.max(1) as f64), m.n_resident);
    Ok(())
}

fn bench(prompt: &str, n: usize) -> Result<()> {
    let tk = tokenizer()?;
    let text = std::fs::read_to_string(prompt)?;
    let chat = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", text.trim());
    let toks = tk.encode(chat.as_str(), false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    let mut m = load(true)?;
    m.want_logits = false;
    let t0 = Instant::now();
    let mut tok = m.prefill(&toks)?;
    let tp = t0.elapsed().as_secs_f64();
    let s0 = m.stats.lock().unwrap().clone();
    let t1 = Instant::now();
    let mut out = vec![tok];
    for _ in 1..n {
        tok = m.step(&[tok])?;
        out.push(tok);
    }
    let td = t1.elapsed().as_secs_f64();
    let st = m.stats.lock().unwrap().clone();
    let (pairs, miss, cpu) = (st.pairs - s0.pairs, st.misses - s0.misses, st.cpu_us - s0.cpu_us);
    let simh = st.sim_hits - s0.sim_hits;
    println!("{}", tk.decode(&out, false).map_err(|e| anyhow!("{e}"))?);
    println!("---\nprefill {} tok {:.2} s ({:.1} tok/s) | decode {} tok {:.2} s = {:.2} tok/s ({:.2} ms/tok) | hit rate {:.1}% \
        ({:.1} misses/tok, {} experts / {:.0} MiB on card) | CPU experts {:.2} ms/tok", toks.len(), tp, toks.len() as f64 / tp,
        n - 1, td, (n - 1) as f64 / td, 1000.0 * td / (n - 1) as f64, 100.0 * (1.0 - miss as f64 / pairs.max(1) as f64),
        miss as f64 / (n - 1) as f64, m.n_resident, m.expert_bytes_on_card as f64 / 1048576.0, cpu as f64 / 1000.0 / (n - 1) as f64);
    println!("dynamic uploads {} ({:.1}/tok); host per step: prepare {:.2} ms, graph {:.2} ms, refill {:.2} ms",
        m.uploads, m.uploads as f64 / (n - 1) as f64, (st.prep_us - s0.prep_us) as f64 / 1000.0 / (n - 1) as f64,
        (st.graph_us - s0.graph_us) as f64 / 1000.0 / (n - 1) as f64, (st.refill_us - s0.refill_us) as f64 / 1000.0 / (n - 1) as f64);
    if std::env::var("FNEXT_MISS_LOG").is_ok() {
        let ml = |i: usize| st.miss_lc.get(i).copied().unwrap_or(0) - s0.miss_lc.get(i).copied().unwrap_or(0);
        let per: Vec<String> = (0..m.c.n_layer).map(|l| format!("{}/{}", ml(l * 2), ml(l * 2 + 1))).collect();
        println!("misses per layer (card0/card1): {}", per.join(" "));
    }
    if simh > 0 {
        println!("simulated LRU: hit rate {:.1}%", 100.0 * (1.0 - (miss - simh) as f64 / pairs.max(1) as f64));
    }
    Ok(())
}

fn counts(out: &str, files: &[String]) -> Result<()> {
    let tk = tokenizer()?;
    let mut m = load(false)?;
    m.want_logits = false;
    let mut total = 0;
    for f in files {
        let text = std::fs::read_to_string(f)?;
        let max: usize = env("FNEXT_MAX_TOK", 4000);
        let mut toks = tk.encode(text.as_str(), false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
        toks.truncate(max);
        m.reset()?;
        let t0 = Instant::now();
        m.prefill(&toks)?;
        total += toks.len();
        eprintln!("{f}: {} tokens in {:.1} s", toks.len(), t0.elapsed().as_secs_f32());
    }
    let c = m.counts_snapshot();
    std::fs::write(out, serde_json::to_string(&c)?)?;
    eprintln!("wrote {out} ({total} tokens)");
    Ok(())
}

/// Routing counts of generated tokens only: each line of `prompts` is a user turn; greedy `n` tokens each.
fn gencounts(out: &str, prompts: &str, n: usize) -> Result<()> {
    let tk = tokenizer()?;
    let mut m = load(true)?;
    m.want_logits = false;
    let nl = m.c.n_layer;
    let ne = m.c.n_expert;
    let mut acc = vec![vec![0u64; ne]; nl];
    for line in std::fs::read_to_string(prompts)?.lines().filter(|l| !l.trim().is_empty()) {
        let chat = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", line.trim());
        let toks = tk.encode(chat.as_str(), false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
        m.reset()?;
        let mut tok = m.prefill(&toks)?;
        let c0 = m.counts_snapshot();
        let t0 = Instant::now();
        for _ in 1..n {
            tok = m.step(&[tok])?;
            if tok == 248046 { break; }
        }
        let c1 = m.counts_snapshot();
        for l in 0..nl { for e in 0..ne { acc[l][e] += c1[l][e] - c0[l][e]; } }
        let st = m.stats.lock().unwrap().clone();
        eprintln!("{:.50}: {:.1} s, hit rate so far {:.1}%", line, t0.elapsed().as_secs_f32(),
            100.0 * (1.0 - st.misses as f64 / st.pairs.max(1) as f64));
    }
    std::fs::write(out, serde_json::to_string(&acc)?)?;
    Ok(())
}

/// Verify / commit check: drafts = the REF greedy stream, `rows` per verify; every prediction must match REF.
fn spec(which: &str, rows: usize) -> Result<()> {
    let f = if which == "a" { "a_short" } else { "b_code600" };
    let toks: Vec<u32> = serde_json::from_str::<Vec<u32>>(&std::fs::read_to_string(format!("{}/prompts/{f}.tokens.json", &*REF))?)?;
    let g: Value = serde_json::from_str(&String::from_utf8_lossy(&std::fs::read(format!("{}/greedy_{which}.json", &*REF))?))?;
    let rt: Vec<u32> = g["steps"].as_array().unwrap().iter().map(|s| s["token"].as_u64().unwrap() as u32).collect();
    let mut m = load(true)?;
    m.want_logits = false;
    let first = m.prefill(&toks)?;
    let mut agree = (first == rt[0]) as usize;
    let mut total = 1;
    let mut i = 0; // rt[i] is the token at the next position
    let t0 = Instant::now();
    let mut steps = 0;
    while i + 1 < rt.len() {
        let n = rows.min(rt.len() - i);
        let rowt: Vec<u32> = rt[i..i + n].to_vec();
        let pred = m.verify(&rowt)?;
        steps += 1;
        // pred[j] predicts rt[i + j + 1]
        let mut keep = 0;
        for j in 0..n {
            keep = j + 1;
            if i + j + 1 >= rt.len() { break; }
            total += 1;
            if pred[j] == rt[i + j + 1] { agree += 1 } else { break }
        }
        m.commit(&rowt, keep)?;
        i += keep;
    }
    println!("{which} spec rows {rows}: predictions matching REF {agree}/{total}, {steps} verify steps, {:.2} s",
        t0.elapsed().as_secs_f64());
    Ok(())
}

fn mtp_bench(prompt: &str, n: usize, nd: usize) -> Result<()> {
    let tk = tokenizer()?;
    let text = std::fs::read_to_string(prompt)?;
    let chat = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", text.trim());
    let toks = tk.encode(chat.as_str(), false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    let mut m = load(true)?;
    m.want_logits = false;
    if !m.has_mtp() {
        return Err(anyhow!("set FNEXT_MTP=1"));
    }
    if std::env::var("FNEXT_BENCH_WARM").map_or(true, |v| v != "0") {
        // capture the step / verify / MTP / commit graphs first (as the server does at start), then start over
        let (f, d) = m.prefill_mtp(&toks)?;
        m.generate_spec(f, d, 96, 7, &[])?;
        m.reset()?;
        m.spec_stats = (0, 0);
        m.spec_ms = [0.0; 5];
    }
    let t0 = Instant::now();
    let (first, d1) = m.prefill_mtp(&toks)?;
    let tp = t0.elapsed().as_secs_f64();
    let s0 = m.stats.lock().unwrap().clone();
    let t1 = Instant::now();
    let out = m.generate_spec(first, d1, n, nd, &[])?;
    let td = t1.elapsed().as_secs_f64();
    let st = m.stats.lock().unwrap().clone();
    println!("{}", tk.decode(&out, false).map_err(|e| anyhow!("{e}"))?);
    let (dr, ac) = m.spec_stats;
    println!("---\nprefill {} tok {:.2} s | decode {} tok {:.2} s = {:.2} tok/s | drafts {nd}/step, accepted {:.1}% ({:.2} tokens/verify) \
        | hit rate {:.1}% | CPU experts {:.2} ms/tok", toks.len(), tp, out.len() - 1, td, (out.len() - 1) as f64 / td,
        100.0 * ac as f64 / dr.max(1) as f64, 1.0 + ac as f64 / (dr as f64 / nd as f64).max(1.0),
        100.0 * (1.0 - (st.misses - s0.misses) as f64 / (st.pairs - s0.pairs).max(1) as f64),
        (st.cpu_us - s0.cpu_us) as f64 / 1000.0 / (out.len() - 1) as f64);
    if let Ok(path) = std::env::var("FNEXT_MTP_COUNT") {
        // merge into the counts file, then print how much routing the top-S experts cover
        let mut tot: Vec<u64> = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_else(|| vec![0; m.mtp_counts.len()]);
        for (a, b) in tot.iter_mut().zip(&m.mtp_counts) {
            *a += b;
        }
        std::fs::write(&path, serde_json::to_string(&tot)?)?;
        let mut v = tot.clone();
        v.sort_unstable_by(|a, b| b.cmp(a));
        let all: u64 = v.iter().sum();
        let cov = |k: usize| v.iter().take(k).sum::<u64>() as f64 / all.max(1) as f64;
        println!("MTP routing: {all} picks, {} experts used; top 64 {:.3}, 128 {:.3}, 192 {:.3}, 256 {:.3}, 320 {:.3}",
            v.iter().filter(|&&x| x > 0).count(), cov(64), cov(128), cov(192), cov(256), cov(320));
    }
    let sm = m.spec_ms;
    let k = sm[4].max(1.0);
    println!("verify steps {}: {:.2} tokens per step ({:.2} drafts a step)", sm[4], (out.len() - 1) as f64 / sm[4].max(1.0),
        dr as f64 / sm[4].max(1.0));
    println!("per verify step: drafts {:.2} ms, verify {:.2} ms, commit {:.2} ms, catch-up {:.2} ms", sm[0] / k, sm[1] / k,
        sm[2] / k, sm[3] / k);
    if std::env::var("FNEXT_TIME_GRAPHS").is_ok() {
        let (a, b) = m.time_graphs(30)?;
        println!("back-to-back: MTP graph {a:.2} ms, trunk graph {b:.2} ms");
    }
    let mm = m.mtp_ms;
    println!("MTP pass: {:.0} passes, host prep {:.2} ms, graph {:.2} ms", mm[2], mm[0] / mm[2].max(1.0), mm[1] / mm[2].max(1.0));
    Ok(())
}

/// Prompt speed: fnext-run pp <text file> <n tokens>...  (each n: fresh sequence, n tokens of the file, repeated)
fn pp(file: &str, ns: &[String]) -> Result<()> {
    let tk = tokenizer()?;
    let text = std::fs::read_to_string(file)?;
    let base = tk.encode(text.as_str(), false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    let mut m = load(true)?;
    m.want_logits = false;
    for n in ns {
        let n: usize = n.parse()?;
        let toks: Vec<u32> = base.iter().cycle().take(n).cloned().collect();
        m.reset()?;
        let t0 = Instant::now();
        // FNEXT_MTP=1: the server path (trunk + the MTP block over the prompt rows)
        let (t, d1) = if std::env::var("FNEXT_MTP").is_ok_and(|v| v == "1") {
            let (t, d) = m.prefill_mtp(&toks)?;
            (t, Some(d))
        } else {
            (m.prefill(&toks)?, None)
        };
        let dt = t0.elapsed().as_secs_f64();
        let dr = d1.map(|d| format!(", first draft {d} p {:.4}", m.draft_p())).unwrap_or_default();
        println!("prompt {n} tokens: {:.2} s = {:.1} tok/s (next token {t}{dr})", dt, n as f64 / dt);
    }
    Ok(())
}

/// Decode speed at a given context: fnext-run ctx <text file> <context tokens> [gen]: a chat turn whose user text is
/// the file (cycled to the length), then `gen` greedy tokens, plain or with MTP (FNEXT_MTP=1).
fn ctxbench(file: &str, n: usize, ngen: usize) -> Result<()> {
    let tk = tokenizer()?;
    let text = std::fs::read_to_string(file)?;
    let base = tk.encode(text.as_str(), false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    let head = tk.encode("<|im_start|>user\n", false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    let tail = tk.encode("\nSummarise the text above, then write a Rust function that counts its words.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", false)
        .map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    let body = n.saturating_sub(head.len() + tail.len());
    let mut toks = head.clone();
    toks.extend(base.iter().cycle().take(body));
    toks.extend(tail);
    let mut m = load(true)?;
    m.want_logits = false;
    let t0 = Instant::now();
    let s0;
    let t1;
    let out;
    if m.has_mtp() {
        let (f, d) = m.prefill_mtp(&toks)?;
        s0 = m.stats.lock().unwrap().clone();
        t1 = Instant::now();
        out = m.generate_spec(f, d, ngen, 4, &[])?;
    } else {
        let mut t = m.prefill(&toks)?;
        s0 = m.stats.lock().unwrap().clone();
        t1 = Instant::now();
        let mut o = vec![t];
        for _ in 1..ngen {
            t = m.step(&[t])?;
            o.push(t);
        }
        out = o;
    }
    let tp = (t1 - t0).as_secs_f64();
    let td = t1.elapsed().as_secs_f64();
    let st = m.stats.lock().unwrap().clone();
    let (pairs, miss) = (st.pairs - s0.pairs, st.misses - s0.misses);
    println!("ctx {} tokens: prefill {:.1} s ({:.0} tok/s) | decode {:.2} tok/s ({}) | hit rate {:.1}% | KV mapped {} cells | {} uploads",
        toks.len(), tp, toks.len() as f64 / tp, (out.len() - 1) as f64 / td, if m.has_mtp() { "MTP" } else { "plain" },
        100.0 * (1.0 - miss as f64 / pairs.max(1) as f64), m.ctx_mapped, m.uploads);
    Ok(())
}

/// Teacher-forced fast path vs the f32 truth logits (fnext-check truth): full-vocab KL and top-1 agreement.
fn vs_truth(which: &str) -> Result<()> {
    let f = if which == "a" { "a_short" } else { "b_code600" };
    let toks: Vec<u32> = serde_json::from_str::<Vec<u32>>(&std::fs::read_to_string(format!("{}/prompts/{f}.tokens.json", &*REF))?)?;
    let g: Value = serde_json::from_str(&String::from_utf8_lossy(&std::fs::read(format!("{}/greedy_{which}.json", &*REF))?))?;
    let rt: Vec<u32> = g["steps"].as_array().unwrap().iter().map(|s| s["token"].as_u64().unwrap() as u32).collect();
    let tb = std::fs::read(format!("{}/truth_{which}.bin", &*REF))?;
    let truth: Vec<f32> = tb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut m = load(true)?;
    m.want_logits = true;
    let v = m.c.n_vocab;
    m.prefill(&toks)?;
    let (mut kl_s, mut kl_m, mut top) = (0f64, 0f64, 0);
    for i in 0..rt.len() {
        let lt = &truth[i * v..(i + 1) * v];
        let lm = m.logits().to_vec();
        let (p, q) = (log_softmax(lt), log_softmax(&lm));
        let kl: f64 = p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum();
        kl_s += kl;
        kl_m = kl_m.max(kl);
        let am = |x: &[f32]| x.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (j, &y)| if y > b.1 { (j, y) } else { b }).0;
        if am(lt) == am(&lm) { top += 1 }
        if i + 1 < rt.len() {
            m.step(&rt[i..i + 1])?;
        }
    }
    println!("{which} vs f32 truth ({}): KL mean {:.3e} max {:.3e}, top-1 {}/{}", std::env::var("FNEXT_SUMMODE").unwrap_or("default".into()),
        kl_s / rt.len() as f64, kl_m, top, rt.len());
    Ok(())
}

fn ref_prompt(which: &str) -> Result<Vec<u32>> {
    let f = match which { "a" => "a_short", "b" => "b_code600", "c" => "c_long5000", _ => "d_long20k" };
    Ok(serde_json::from_str::<Vec<u32>>(&std::fs::read_to_string(format!("{}/prompts/{f}.tokens.json", &*REF))?)?)
}

/// FNV-1a over the bits of a logit row.
fn hash_bits(v: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in v {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Free-running greedy: prefill, then `n` 1-row steps; tokens and the logits bits hash of every step.
fn greedy_run(m: &mut Fast, toks: &[u32], n: usize) -> Result<(Vec<u32>, Vec<u64>)> {
    let tr = std::env::var("FNEXT_DET_TRACE").is_ok();
    let mut t = m.prefill(toks)?;
    if tr {
        eprintln!("det: prefill done ({} tokens)", toks.len());
    }
    let (mut out, mut hs) = (vec![t], vec![hash_bits(m.logits())]);
    for i in 1..n {
        if tr {
            eprintln!("det: step {i}");
        }
        t = m.step(&[t])?;
        out.push(t);
        hs.push(hash_bits(m.logits()));
    }
    Ok((out, hs))
}

/// fnext-run det <a|b|c|d> [n]: the same prompt twice in one process (first with the expert cache as loaded, then
/// as the first run left it): token streams and logits bits must be identical.
fn det(which: &str, n: usize) -> Result<()> {
    let toks = ref_prompt(which)?;
    let mut m = load(true)?;
    m.want_logits = true;
    let (t1, h1) = greedy_run(&mut m, &toks, n)?;
    if let Ok(p) = std::env::var("FNEXT_HASH_OUT") {
        // the first run's tokens + logits hashes (compare runs across builds / FNEXT_GPUS)
        std::fs::write(&p, serde_json::to_string(&serde_json::json!({"tokens": t1, "hashes": h1}))?)?;
    }
    m.reset()?;
    let (t2, h2) = greedy_run(&mut m, &toks, n)?;
    let ft = t1.iter().zip(&t2).position(|(a, b)| a != b);
    let fh = h1.iter().zip(&h2).position(|(a, b)| a != b);
    println!("{which} det {n}: tokens {} (first diff {:?}), logits bits {} (first diff {:?}) -> {}",
        if ft.is_none() { "identical" } else { "DIFFER" }, ft, if fh.is_none() { "identical" } else { "DIFFER" }, fh,
        if ft.is_none() && fh.is_none() { "PASS" } else { "FAIL" });
    std::io::Write::flush(&mut std::io::stdout())?;
    Ok(())
}

/// fnext-run lossless <a|b|c|d> [n] (FNEXT_MTP=1): greedy stream plain vs with MTP drafts must be identical.
fn lossless(which: &str, n: usize) -> Result<()> {
    let toks = ref_prompt(which)?;
    let mut m = load(true)?;
    if !m.has_mtp() {
        return Err(anyhow!("set FNEXT_MTP=1"));
    }
    m.want_logits = true;
    let (plain, _) = greedy_run(&mut m, &toks, n)?;
    // FNEXT_REPEAT=k: k MTP runs against the one plain stream (catches rare races)
    let reps: usize = std::env::var("FNEXT_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut f = None;
    let mut spec = vec![];
    for r in 0..reps {
        m.reset()?;
        let (first, d1) = m.prefill_mtp(&toks)?;
        spec = m.generate_spec(first, d1, n, 4, &[])?;
        f = plain.iter().zip(&spec).position(|(a, b)| a != b);
        if f.is_some() {
            println!("repeat {r}: differs");
            break;
        }
    }
    println!("{which} lossless {n}: plain vs MTP {} (first diff {:?}{}) -> {}", if f.is_none() { "identical" } else { "DIFFER" },
        f, f.map(|i| format!(": {} vs {}", plain[i], spec[i])).unwrap_or_default(), if f.is_none() { "PASS" } else { "FAIL" });
    std::io::Write::flush(&mut std::io::stdout())?;
    Ok(())
}

fn find_all(ids: &[u32], pat: &[u32]) -> Vec<usize> {
    if pat.is_empty() || ids.len() < pat.len() {
        return vec![];
    }
    (0..=ids.len() - pat.len()).filter(|&i| ids[i..i + pat.len()] == *pat).collect()
}

/// Snapshot points as the server marks them: after every `<|im_start|>assistant\n`, and after the system message.
fn snap_points(ids: &[u32], reply: &[u32], sys: &[u32], end: &[u32]) -> Vec<usize> {
    let mut v: Vec<usize> = find_all(ids, reply).into_iter().map(|i| i + reply.len()).collect();
    if ids.starts_with(sys) {
        if let Some(&i) = find_all(ids, end).first() {
            v.push(i + end.len());
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Greedy continuation of `n` tokens after `first` (plain steps, or MTP spec when d1 is given); tokens + logits hashes
/// (hashes only on the plain path).
fn continue_greedy(m: &mut Fast, first: u32, d1: Option<u32>, n: usize) -> Result<(Vec<u32>, Vec<u64>)> {
    if let Some(d) = d1 {
        let out = m.generate_spec(first, d, n, 4, &[])?;
        return Ok((out, vec![]));
    }
    let (mut out, mut hs) = (vec![first], vec![hash_bits(m.logits())]);
    let mut t = first;
    for _ in 1..n {
        t = m.step(&[t])?;
        out.push(t);
        hs.push(hash_bits(m.logits()));
    }
    Ok((out, hs))
}

/// fnext-run cache <text file> [n]: turn 2 of a chat resumed from turn 1's snapshots vs computed fresh: identical
/// tokens (and logits bits on the plain path). FNEXT_MTP=1 runs the MTP path.
fn cache_test(file: &str, n: usize) -> Result<()> {
    let tk = tokenizer()?;
    let enc = |s: &str| -> Result<Vec<u32>> { Ok(tk.encode(s, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec()) };
    let (reply, sys, end) = (enc("<|im_start|>assistant\n")?, enc("<|im_start|>system\n")?, enc("<|im_end|>\n")?);
    let text = std::fs::read_to_string(file)?;
    let doc: String = text.chars().take(12000).collect();
    let p1 = enc(&format!("<|im_start|>system\nYou are a careful coding assistant. Reference material:\n{doc}<|im_end|>\n\
        <|im_start|>user\nSummarise the reference material in two sentences.<|im_end|>\n<|im_start|>assistant\n"))?;
    let mut m = load(true)?;
    m.want_logits = true;
    let mtp = m.has_mtp();
    // turn 1, snapshots kept as the engine would (first + last two)
    let mut snaps: Vec<fe_fnext::fast::Snap> = vec![];
    let pts1 = snap_points(&p1, &reply, &sys, &end);
    let (f1, d1) = m.prefill_cut(&p1, &pts1, mtp, |m| {
        snaps.push(m.snap_save()?);
        while snaps.len() > 3 {
            snaps.remove(1);
        }
        Ok(())
    })?;
    let (gen1, _) = continue_greedy(&mut m, f1, d1, n)?;
    let mut p2 = p1.clone();
    p2.extend_from_slice(&gen1);
    p2.extend(enc("<|im_end|>\n<|im_start|>user\nNow name two functions it defines.<|im_end|>\n<|im_start|>assistant\n")?);
    let pts2 = snap_points(&p2, &reply, &sys, &end);
    // turn 2 resumed
    let lcp = m.tokens.iter().zip(&p2).take_while(|(a, b)| a == b).count();
    let best = snaps.iter().filter(|s| s.pos <= lcp && s.pos < p2.len()).max_by_key(|s| s.pos).map(|s| s.pos);
    let pos = match best {
        Some(pos) => {
            let sn = snaps.iter().find(|s| s.pos == pos).unwrap();
            m.snap_restore(sn)?;
            pos
        }
        None => {
            m.reset()?;
            0
        }
    };
    let t0 = Instant::now();
    let (f2, d2) = m.prefill_cut(&p2[pos..], &pts2, mtp, |_| Ok(()))?;
    let tc = t0.elapsed().as_secs_f64();
    let (c_out, c_h) = continue_greedy(&mut m, f2, d2, n)?;
    // turn 2 fresh
    m.reset()?;
    let t1 = Instant::now();
    let (f3, d3) = m.prefill_cut(&p2, &pts2, mtp, |_| Ok(()))?;
    let tf = t1.elapsed().as_secs_f64();
    let (r_out, r_h) = continue_greedy(&mut m, f3, d3, n)?;
    let ft = c_out.iter().zip(&r_out).position(|(a, b)| a != b);
    let fh = c_h.iter().zip(&r_h).position(|(a, b)| a != b);
    let ok = ft.is_none() && fh.is_none();
    println!("cache{}: turn 2 = {} tokens, resumed at {pos} (points {:?}); prefill {:.2} s resumed vs {:.2} s fresh; tokens {} \
        (first diff {ft:?}), logits bits {} (first diff {fh:?}) -> {}", if mtp { " (MTP)" } else { "" }, p2.len(), pts2, tc, tf,
        if ft.is_none() { "identical" } else { "DIFFER" }, if fh.is_none() { "identical" } else { "DIFFER" },
        if ok { "PASS" } else { "FAIL" });
    std::io::Write::flush(&mut std::io::stdout())?;
    Ok(())
}


/// fnext-run lanes <p> <q> [n] (FNEXT_SLOTS >= 2): prompts p and q decoded alone on lane 0, then interleaved step by
/// step on lanes 0 and 1 (state swapped every step): both streams must equal their solo runs.
fn lanes_test(p: &str, q: &str, n: usize) -> Result<()> {
    let (tp, tq) = (ref_prompt(p)?, ref_prompt(q)?);
    let mut m = load(true)?;
    if m.n_lanes() < 2 {
        return Err(anyhow!("set FNEXT_SLOTS=2"));
    }
    m.want_logits = false;
    let (ap, _) = greedy_run(&mut m, &tp, n)?;
    m.reset()?;
    let (aq, _) = greedy_run(&mut m, &tq, n)?;
    m.reset()?;
    let mut sp = vec![m.prefill(&tp)?];
    m.select_lane(1)?;
    m.reset()?;
    let mut sq = vec![m.prefill(&tq)?];
    for _ in 1..n {
        m.select_lane(0)?;
        let t = m.step(&[*sp.last().unwrap()])?;
        sp.push(t);
        m.select_lane(1)?;
        let t = m.step(&[*sq.last().unwrap()])?;
        sq.push(t);
    }
    let fp = ap.iter().zip(&sp).position(|(a, b)| a != b);
    let fq = aq.iter().zip(&sq).position(|(a, b)| a != b);
    println!("lanes {p}+{q} {n} (swapped steps): lane 0 vs alone {fp:?}, lane 1 vs alone {fq:?} -> {}",
        if fp.is_none() && fq.is_none() { "PASS" } else { "FAIL" });
    // batched: both lanes in one step graph
    m.select_lane(0)?;
    m.reset()?;
    let mut bp = vec![m.prefill(&tp)?];
    m.select_lane(1)?;
    m.reset()?;
    let mut bq = vec![m.prefill(&tq)?];
    let t0 = Instant::now();
    for _ in 1..n {
        let r = m.step_lanes(&[(0, *bp.last().unwrap()), (1, *bq.last().unwrap())])?;
        bp.push(r[0]);
        bq.push(r[1]);
    }
    let dt = t0.elapsed().as_secs_f64();
    let fp = ap.iter().zip(&bp).position(|(a, b)| a != b);
    let fq = aq.iter().zip(&bq).position(|(a, b)| a != b);
    println!("lanes {p}+{q} {n} (batched, {:.1} tok/s total): lane 0 vs alone {fp:?}, lane 1 vs alone {fq:?} -> {}",
        2.0 * (n - 1) as f64 / dt, if fp.is_none() && fq.is_none() { "PASS" } else { "FAIL" });
    std::io::Write::flush(&mut std::io::stdout())?;
    Ok(())
}


/// fnext-run lanesn <n> <p1> <p2> ... (FNEXT_SLOTS >= count): every prompt alone on lane 0, then all of them batched
/// (one lane each, one step graph): streams must match; aggregate and per-lane tok/s of the batched run (after the
/// first step).
fn lanes_n(n: usize, ps: &[String]) -> Result<()> {
    let toks: Vec<Vec<u32>> = ps.iter().map(|p| ref_prompt(p)).collect::<Result<_>>()?;
    let mut m = load(true)?;
    let k = toks.len();
    if m.n_lanes() < k {
        return Err(anyhow!("set FNEXT_SLOTS={k}"));
    }
    m.want_logits = false;
    let mut alone = vec![];
    let mut t_alone = 0.0;
    for t in &toks {
        m.select_lane(0)?;
        m.reset()?;
        let mut cur = m.prefill(t)?;
        let mut out = vec![cur];
        let t0 = Instant::now();
        for _ in 1..n {
            cur = m.step(&[cur])?;
            out.push(cur);
        }
        t_alone += t0.elapsed().as_secs_f64();
        alone.push(out);
    }
    let mut outs: Vec<Vec<u32>> = vec![];
    for (i, t) in toks.iter().enumerate() {
        m.select_lane(i)?;
        m.reset()?;
        outs.push(vec![m.prefill(t)?]);
    }
    let mut t0 = Instant::now();
    let mut s0 = m.stats.lock().unwrap().clone();
    for st in 1..n {
        if st == 2 {
            t0 = Instant::now();
            s0 = m.stats.lock().unwrap().clone();
        }
        let items: Vec<(usize, u32)> = (0..k).map(|i| (i, *outs[i].last().unwrap())).collect();
        let r = m.step_lanes(&items)?;
        for i in 0..k {
            outs[i].push(r[i]);
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    let s1 = m.stats.lock().unwrap().clone();
    let steps = (s1.steps - s0.steps).max(1) as f64;
    println!("batched per step: {:.2} ms, CPU experts {:.2} ms, hit rate {:.1}%, misses {:.1}/step, graph {:.2} ms, prep {:.2} ms",
        dt * 1e3 / steps, (s1.cpu_us - s0.cpu_us) as f64 / 1e3 / steps,
        100.0 * (1.0 - (s1.misses - s0.misses) as f64 / (s1.pairs - s0.pairs).max(1) as f64), (s1.misses - s0.misses) as f64 / steps,
        (s1.graph_us - s0.graph_us) as f64 / 1e3 / steps, (s1.prep_us - s0.prep_us) as f64 / 1e3 / steps);
    let diffs: Vec<Option<usize>> = (0..k).map(|i| alone[i].iter().zip(&outs[i]).position(|(a, b)| a != b)).collect();
    let ok = diffs.iter().all(|d| d.is_none());
    println!("lanesn {k} x {n}: alone {:.1} tok/s; batched {:.1} tok/s total, {:.1} per lane; first diffs {diffs:?} -> {}",
        (k * (n - 1)) as f64 / t_alone, (k * (n - 2)) as f64 / dt, (n - 2) as f64 / dt, if ok { "PASS" } else { "FAIL" });
    std::io::Write::flush(&mut std::io::stdout())?;
    Ok(())
}

/// Chat-wrapped token ids of a prompt file (as bench / mtp).
fn chat_tokens(path: &str) -> Result<Vec<u32>> {
    let tk = tokenizer()?;
    let text = std::fs::read_to_string(path)?;
    let chat = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", text.trim());
    Ok(tk.encode(chat.as_str(), false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec())
}

/// `n` greedy tokens of every prompt on lane 0, one after another (the reference streams).
fn alone_streams(m: &mut Fast, toks: &[Vec<u32>], n: usize) -> Result<Vec<Vec<u32>>> {
    let mut out = vec![];
    for t in toks {
        m.select_lane(0)?;
        m.reset()?;
        let mut cur = m.prefill(t)?;
        let mut v = vec![cur];
        while v.len() < n {
            cur = m.step(&[cur])?;
            v.push(cur);
        }
        out.push(v);
    }
    Ok(out)
}

/// Decode every prompt on its own lane at once until each has `n` tokens: plain (batched 1-row steps) or MTP
/// (spec_lanes). Returns the streams, the decode seconds after the warm-up steps and the steps timed.
fn lanes_decode(m: &mut Fast, toks: &[Vec<u32>], n: usize, mtp: bool, nd: usize) -> Result<(Vec<Vec<u32>>, f64, u64, usize)> {
    lanes_decode_t(m, toks, n, mtp, nd, &mut None)
}

/// lanes_decode with an optional draft tuner (FNEXT_TUNE=1 in lbench; it keeps learning across calls).
fn lanes_decode_t(m: &mut Fast, toks: &[Vec<u32>], n: usize, mtp: bool, nd: usize, tuner: &mut Option<fe_fnext::fast::DraftTuner>)
    -> Result<(Vec<Vec<u32>>, f64, u64, usize)> {
    let k = toks.len();
    let mut outs: Vec<Vec<u32>> = vec![];
    let mut specs: Vec<Spec> = vec![];
    for (i, t) in toks.iter().enumerate() {
        m.select_lane(i)?;
        m.reset()?;
        if mtp {
            let (f, d) = m.prefill_mtp(t)?;
            outs.push(vec![f]);
            specs.push(Spec { cur: f, drafts: vec![d] });
        } else {
            outs.push(vec![m.prefill(t)?]);
        }
    }
    // time from the third step on (graph captures and first uploads before it)
    let mut t0 = Instant::now();
    let mut base: Vec<usize> = outs.iter().map(|o| o.len()).collect();
    let mut steps = 0u64;
    while outs.iter().any(|o| o.len() < n) {
        if steps == 2 {
            t0 = Instant::now();
            base = outs.iter().map(|o| o.len()).collect();
        }
        let live: Vec<usize> = (0..k).filter(|&i| outs[i].len() < n).collect();
        if mtp {
            let mut taken: Vec<Spec> = live.iter().map(|&i| std::mem::replace(&mut specs[i], Spec { cur: 0, drafts: vec![] })).collect();
            let mut items: Vec<(usize, &mut Spec)> = live.iter().copied().zip(taken.iter_mut()).collect();
            let (d, t0s) = match tuner.as_mut() { Some(t) => (t.pick(items.len()), Some(Instant::now())), None => (nd, None) };
            let got = m.spec_lanes(&mut items, d)?;
            if let (Some(t), Some(t0s)) = (tuner.as_mut(), t0s) {
                t.record(got.iter().map(|g| g.len()).sum(), t0s.elapsed().as_secs_f64());
            }
            for (j, &i) in live.iter().enumerate() {
                outs[i].extend_from_slice(&got[j]);
            }
            for (j, sp) in taken.into_iter().enumerate() {
                specs[live[j]] = sp;
            }
        } else {
            let items: Vec<(usize, u32)> = live.iter().map(|&i| (i, *outs[i].last().unwrap())).collect();
            let r = m.step_lanes(&items)?;
            for (j, &i) in live.iter().enumerate() {
                outs[i].push(r[j]);
            }
        }
        steps += 1;
    }
    let dt = t0.elapsed().as_secs_f64();
    let made: usize = outs.iter().zip(&base).map(|(o, b)| o.len().min(n).saturating_sub(*b)).sum();
    for o in &mut outs {
        o.truncate(n);
    }
    Ok((outs, dt, steps.saturating_sub(2), made))
}

/// fnext-run lbench <plain|mtp> <n> <prompt files...> (FNEXT_SLOTS >= files; mtp: FNEXT_MTP=1): every prompt
/// alone first (reference greedy streams), a warm-up round, then all prompts on their own lanes at once; prints
/// total and per-lane tok/s, hit rate, CPU expert ms per step, MTP acceptance, and whether every lane's stream
/// equals its alone stream.
fn lanes_bench(mode: &str, n: usize, files: &[String]) -> Result<()> {
    let toks: Vec<Vec<u32>> = files.iter().map(|f| chat_tokens(f)).collect::<Result<_>>()?;
    let k = toks.len();
    let mut m = load(true)?;
    if m.n_lanes() < k {
        return Err(anyhow!("set FNEXT_SLOTS={k}"));
    }
    let mtp = mode == "mtp";
    if mtp && !m.has_mtp() {
        return Err(anyhow!("set FNEXT_MTP=1"));
    }
    let nd: usize = env("FNEXT_DRAFTS", 4);
    m.want_logits = false;
    let alone = if std::env::var("FNEXT_LB_CHECK").map_or(true, |v| v != "0") { Some(alone_streams(&mut m, &toks, n)?) } else { None };
    let mut tuner = if std::env::var("FNEXT_TUNE").is_ok_and(|v| v == "1") { Some(fe_fnext::fast::DraftTuner::new(nd)) } else { None };
    // warm-up: the graphs of this lane set (and the tuner's first measurements)
    lanes_decode_t(&mut m, &toks, if tuner.is_some() { n } else { 48.min(n) }, mtp, nd, &mut tuner)?;
    let reps: usize = env("FNEXT_LB_REPS", 3);
    for rep in 0..reps {
        let s0 = m.stats.lock().unwrap().clone();
        let sp0 = m.spec_stats;
        let sm0 = m.spec_ms;
        let (outs, dt, steps, made) = lanes_decode_t(&mut m, &toks, n, mtp, nd, &mut tuner)?;
        let s1 = m.stats.lock().unwrap().clone();
        let sp1 = m.spec_stats;
        if let Some(t) = &tuner {
            println!("tuner: {:?}", t.summary());
        }
        let stp = (s1.steps - s0.steps).max(1) as f64;
        let ok = alone.as_ref().map(|a| (0..k).map(|i| a[i].iter().zip(&outs[i]).position(|(x, y)| x != y)).collect::<Vec<_>>());
        let pass = ok.as_ref().map(|v| v.iter().all(|d| d.is_none()));
        let sm1 = m.spec_ms;
        let ns = (sm1[4] - sm0[4]).max(1.0);
        let acc = if mtp { format!(", accepted {:.1}% of {} drafts, ms/step drafts {:.1} verify {:.1} commit {:.1} catch-up {:.1}",
            100.0 * (sp1.1 - sp0.1) as f64 / (sp1.0 - sp0.0).max(1) as f64, sp1.0 - sp0.0, (sm1[0] - sm0[0]) / ns,
            (sm1[1] - sm0[1]) / ns, (sm1[2] - sm0[2]) / ns, (sm1[3] - sm0[3]) / ns) } else { String::new() };
        println!("lbench {mode} {k} lanes x {n} rep {rep}: {:.1} tok/s total, {:.1} per lane | {:.2} ms/step over {steps} steps, hit rate {:.1}%, \
            CPU experts {:.2} ms/step{acc} | vs alone {:?} -> {}",
            made as f64 / dt, made as f64 / dt / k as f64, dt * 1e3 / steps.max(1) as f64,
            100.0 * (1.0 - (s1.misses - s0.misses) as f64 / (s1.pairs - s0.pairs).max(1) as f64),
            (s1.cpu_us - s0.cpu_us) as f64 / 1e3 / stp, ok, match pass { Some(true) => "PASS", Some(false) => "FAIL", None => "unchecked" });
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    std::io::Write::flush(&mut std::io::stdout())?;
    Ok(())
}

/// fnext-run lanesm <n> <a|b|c|d ...> (FNEXT_MTP=1, FNEXT_SLOTS >= count): reference prompts with MTP on their own
/// lanes at once (spec_lanes) vs each prompt's plain greedy stream alone.
fn lanes_mtp(n: usize, ps: &[String]) -> Result<()> {
    let toks: Vec<Vec<u32>> = ps.iter().map(|p| ref_prompt(p)).collect::<Result<_>>()?;
    let mut m = load(true)?;
    if m.n_lanes() < toks.len() || !m.has_mtp() {
        return Err(anyhow!("set FNEXT_SLOTS={} FNEXT_MTP=1", toks.len()));
    }
    m.want_logits = false;
    let alone = alone_streams(&mut m, &toks, n)?;
    let nd: usize = env("FNEXT_DRAFTS", 4);
    let sp0 = m.spec_stats;
    let (outs, dt, steps, made) = lanes_decode(&mut m, &toks, n, true, nd)?;
    let sp1 = m.spec_stats;
    let diffs: Vec<Option<usize>> = (0..toks.len()).map(|i| alone[i].iter().zip(&outs[i]).position(|(a, b)| a != b)).collect();
    let ok = diffs.iter().all(|d| d.is_none());
    println!("lanesm {} x {n}: {:.1} tok/s total over {steps} steps, accepted {:.1}% of {} drafts; first diffs vs alone {diffs:?} -> {}",
        toks.len(), made as f64 / dt, 100.0 * (sp1.1 - sp0.1) as f64 / (sp1.0 - sp0.0).max(1) as f64, sp1.0 - sp0.0,
        if ok { "PASS" } else { "FAIL" });
    std::io::Write::flush(&mut std::io::stdout())?;
    Ok(())
}

/// fnext-run rowinv <a|b|c|d> [max rows]: decode-step batch invariance. After the prompt, verify steps of N = 1..max
/// rows (the greedy continuation as the rows; a verify does not advance the sequence) must give row j the same
/// logits bits for every N > j. Prints the mismatches.
fn row_inv(which: &str, nmax: usize) -> Result<()> {
    let toks = ref_prompt(which)?;
    let mut m = load(true)?;
    m.want_logits = false;
    let t = m.prefill(&toks)?;
    let mut cont = vec![t];
    let snap = m.snap_save()?;
    for _ in 1..nmax {
        let x = m.step(&[*cont.last().unwrap()])?;
        cont.push(x);
    }
    m.snap_restore(&snap)?;
    m.tokens.truncate(toks.len());
    let mut h: Vec<Vec<u64>> = vec![];
    if std::env::var("FNEXT_DUMP").is_ok() {
        // the first (layer, stage) where rows < 8 of an 8-row and a 9-row step differ
        m.dump.clear();
        m.verify(&cont[..8])?;
        let d8 = std::mem::take(&mut m.dump);
        m.verify(&cont[..9])?;
        let d9 = std::mem::take(&mut m.dump);
        for (a, b) in d8.iter().zip(&d9) {
            let rs: Vec<usize> = (0..8).filter(|&r| a.2[r] != b.2[r]).collect();
            if !rs.is_empty() {
                println!("dump: first difference at layer {} stage {} rows {rs:?}", a.0, a.1);
                break;
            }
        }
    }
    for n in 1..=nmax {
        m.verify(&cont[..n])?;
        let mut v = vec![];
        for j in 0..n {
            v.push(hash_bits(&m.logits_row(j)?));
        }
        h.push(v);
    }
    let mut bad = vec![];
    for n in 2..=nmax {
        for j in 0..n - 1 {
            if h[n - 1][j] != h[j][j] {
                bad.push((n, j));
            }
        }
    }
    let first: Vec<usize> = (1..=nmax).filter(|&n| (0..n - 1).any(|j| h[n - 1][j] != h[j][j])).collect();
    for n in 2..=nmax {
        let js: Vec<usize> = (0..n - 1).filter(|&j| h[n - 1][j] != h[j][j]).collect();
        let j8: Vec<usize> = (0..n.min(8)).filter(|&j| h[n - 1][j] != h[7.min(n - 1)][j]).collect();
        if !js.is_empty() || !j8.is_empty() {
            println!("  N={n}: rows differing from their own-size step {js:?}; rows < 8 differing from the 8-row step {j8:?}");
        }
    }
    println!("rowinv {which} 1..{nmax}: {} mismatching (rows, row) pairs; step sizes with any: {first:?} -> {}",
        bad.len(), if bad.is_empty() { "PASS" } else { "FAIL" });
    Ok(())
}

/// fnext-run park <file> [n] [lane]: on `lane`, a two-turn chat as cache does, but between the turns the lane's
/// conversation is parked in host memory (park), another prompt runs on the lane, and it comes back (unpark); turn 2
/// then resumes at a snapshot (as the server does) and must match turn 2 run fresh, tokens and logits bits.
fn park_test(file: &str, n: usize, lane: usize) -> Result<()> {
    let tk = tokenizer()?;
    let enc = |s: &str| -> Result<Vec<u32>> { Ok(tk.encode(s, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec()) };
    let (reply, sys, end) = (enc("<|im_start|>assistant\n")?, enc("<|im_start|>system\n")?, enc("<|im_end|>\n")?);
    let text = std::fs::read_to_string(file)?;
    let doc: String = text.chars().take(12000).collect();
    let p1 = enc(&format!("<|im_start|>system\nYou are a careful coding assistant. Reference material:\n{doc}<|im_end|>\n\
        <|im_start|>user\nSummarise the reference material in two sentences.<|im_end|>\n<|im_start|>assistant\n"))?;
    let mut m = load(true)?;
    m.want_logits = true;
    let mtp = m.has_mtp();
    m.select_lane(lane)?;
    m.reset()?;
    let pts1 = snap_points(&p1, &reply, &sys, &end);
    let (f1, d1) = m.prefill_cut(&p1, &pts1, mtp, |m| {
        let sn = m.snap_save()?;
        m.snaps.push(sn);
        while m.snaps.len() > 3 {
            m.snaps.remove(1);
        }
        Ok(())
    })?;
    let (gen1, _) = continue_greedy(&mut m, f1, d1, n)?;
    let mut p2 = p1.clone();
    p2.extend_from_slice(&gen1);
    p2.extend(enc("<|im_end|>\n<|im_start|>user\nNow name two functions it defines.<|im_end|>\n<|im_start|>assistant\n")?);
    let pts2 = snap_points(&p2, &reply, &sys, &end);
    // park, clobber the lane with another conversation, bring it back
    let t0 = Instant::now();
    let pk = m.park()?.ok_or_else(|| anyhow!("nothing to park (no snapshot)"))?;
    let tpark = t0.elapsed().as_secs_f64();
    let other = ref_prompt("b")?;
    let mut t = m.prefill(&other)?;
    for _ in 0..16 {
        t = m.step(&[t])?;
    }
    let t1 = Instant::now();
    let bytes = pk.bytes();
    m.unpark(pk)?;
    let tun = t1.elapsed().as_secs_f64();
    let lcp = m.tokens.iter().zip(&p2).take_while(|(a, b)| a == b).count();
    let best = m.snaps.iter().filter(|s| s.pos <= lcp && s.pos < p2.len()).map(|s| s.pos).max();
    let pos = match best {
        Some(pos) => {
            let i = m.snaps.iter().position(|s| s.pos == pos).unwrap();
            let sn = m.snaps.remove(i);
            let r = m.snap_restore(&sn);
            m.snaps.insert(i, sn);
            r?;
            pos
        }
        None => return Err(anyhow!("no snapshot to resume from after unpark")),
    };
    let (f2, d2) = m.prefill_cut(&p2[pos..], &pts2, mtp, |_| Ok(()))?;
    let (c_out, c_h) = continue_greedy(&mut m, f2, d2, n)?;
    m.reset()?;
    let (f3, d3) = m.prefill_cut(&p2, &pts2, mtp, |_| Ok(()))?;
    let (r_out, r_h) = continue_greedy(&mut m, f3, d3, n)?;
    let ft = c_out.iter().zip(&r_out).position(|(a, b)| a != b);
    let fh = c_h.iter().zip(&r_h).position(|(a, b)| a != b);
    let ok = ft.is_none() && fh.is_none();
    println!("park{} lane {lane}: parked {:.0} MB in {:.2} s, unparked in {:.2} s, turn 2 resumed at {pos} of {}; tokens {} (first diff {ft:?}), \
        logits bits {} (first diff {fh:?}) -> {}", if mtp { " (MTP)" } else { "" }, bytes as f64 / 1e6, tpark, tun, p2.len(),
        if ft.is_none() { "identical" } else { "DIFFER" }, if fh.is_none() { "identical" } else { "DIFFER" }, if ok { "PASS" } else { "FAIL" });
    std::io::Write::flush(&mut std::io::stdout())?;
    Ok(())
}
