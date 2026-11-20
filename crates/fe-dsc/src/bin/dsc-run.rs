//! The engine driver (development and checks; cards via CUDA_VISIBLE_DEVICES, files as fe-server finds them):
//!   dsc-run gen <prompt text> [n]              greedy completion (token-by-token prompt), prints text + tok/s + split
//!   dsc-run logits <text file> [ntok] <out>    last-row logits (f32 LE) to <out>; DSC_DUMP_H=<file>: per-layer residual of the last row
//!   dsc-run ppl <text file> [ntok]             teacher-forced mean NLL / perplexity, top-1 of every row to DSC_TOP1 (u32 LE) if set
use anyhow::{bail, Result};
use fe_dsc::engine::{argmax, Engine};
use fe_dsc::{config::DscConfig, refmodel::logprob, tok};

fn vmstat(k: &str) -> u64 {
    std::fs::read_to_string("/proc/vmstat").unwrap_or_default().lines().find(|l| l.split_whitespace().next() == Some(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn minflt() -> u64 {
    let t = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let r = t.rsplit(')').next().unwrap_or("");
    r.split_whitespace().nth(7).and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn ensure_eq(a: &[u32], b: &[u32]) {
    assert!(a == b, "the two sequences do not share the kept prefix");
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let path = fe_dsc::model_path();
    let eng = fe_dsc::engram_path(&path);
    let g = fe_dsc::open_model(&path, eng.as_deref(), false)?;
    let c = DscConfig::from_gguf(&g)?;
    let max_ctx: usize = std::env::var("DSC_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let text_ids = |text: &str, ntok: usize| -> Result<Vec<u32>> {
        // "ids: 0 128803 ..." = the token ids themselves (BOS included), anything else is text
        if let Some(r) = text.strip_prefix("ids:") {
            let mut ids: Vec<u32> = r.split_whitespace().map(|v| v.parse().unwrap()).collect();
            ids.truncate(ntok);
            return Ok(ids);
        }
        let t = tok::build(&g, &c)?;
        let mut ids = vec![0u32];
        ids.extend(tok::encode(&t, text)?);
        ids.truncate(ntok);
        Ok(ids)
    };
    match a.first().map(String::as_str) {
        Some("gen") => {
          for _round in 0..std::env::var("DSC_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize) {
            let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
            let t = tok::build(&g, &c)?;
            let ids = text_ids(&a[1], std::env::var("DSC_MAXTOK").ok().and_then(|v| v.parse().ok()).unwrap_or(4096))?;
            let mut e = Engine::new(&g, &c, max_ctx)?;
            let mut out = vec![];
            let t0 = std::time::Instant::now();
            let mut lg = None;
            if std::env::var("DSC_PREFILL").is_ok_and(|v| v == "chunk") {
                lg = e.prefill_chunked(&ids, true)?;
            } else {
                for (i, &id) in ids.iter().enumerate() {
                    lg = e.step(id, i + 1 == ids.len())?;
                }
            }
            let tp = t0.elapsed().as_secs_f32();
            eprintln!("prompt {} tokens {:.2}s ({:.2} tok/s)", ids.len(), tp, ids.len() as f32 / tp);
            if std::env::var("DSC_CPUPROF").is_ok() { fe_ggq::cpu::PROF.store(true, std::sync::atomic::Ordering::Relaxed); }
            let f0 = minflt();
            let vm0 = [vmstat("numa_hint_faults"), vmstat("numa_pages_migrated"), vmstat("thp_split_page"), vmstat("thp_fault_alloc"), vmstat("pgfault")];
            let t1 = std::time::Instant::now();
            let (mut sg, mut sc, mut se, mut st) = (0.0, 0.0, 0.0, 0.0);
            let (mut hit, mut hit0, mut miss) = (0u32, 0u32, 0u32);
            let mut ph = [0f64; 10];
            for _ in 0..n {
                let next = argmax(lg.as_ref().unwrap());
                out.push(next);
                lg = e.step(next, true)?;
                sg += e.split.gpu_ms;
                sc += e.split.cpu_ms;
                se += e.split.engram_ms;
                st += e.split.total_ms;
                hit += e.split.hit;
                for i in 0..10 { ph[i] += e.split.ph[i]; }
                hit0 += e.split.hit0;
                miss += e.split.miss;
            }
            let td = t1.elapsed().as_secs_f32();
            let nf = n as f64;
            eprintln!("decode {n} tokens {td:.2}s ({:.2} tok/s); ms/token: total {:.1} = gpu dense {:.1} + cpu experts {:.1} + engram {:.1}",
                n as f32 / td, st / nf, sg / nf, sc / nf, se / nf);
            eprintln!("phases ms/token: attention-to-ids {:.1}, card enqueue {:.1}, CpuMoe::run {:.1}, tail (card0 wait + combine + post) {:.1}, hot repeat of the CPU jobs {:.1} (with DSC_CARDTIME: expert card 1 GPU ms/token {:.1}, card 2 {:.1})", ph[0] / nf, ph[1] / nf, ph[2] / nf, ph[3] / nf, ph[4] / nf, ph[4] / nf, ph[9] / nf);
            eprintln!("detail ms/token: seg0 sync wait {:.1} (rest of attention-to-ids = seg0 enqueue + ids D2H {:.1}), card wait in tail {:.1}, tail through uploads {:.1}, seg2 enqueue {:.1}", ph[5] / nf, (ph[0] - ph[5]) / nf, ph[6] / nf, ph[7] / nf, ph[8] / nf);
            if std::env::var("DSC_CPUPROF").is_ok() {
                let p: Vec<f64> = fe_ggq::cpu::PROF_NS.iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6 / nf).collect();
                eprintln!("CpuMoe prof ms/token (thread-summed except 3,4,6,7): gu {:.2} wait {:.2} down {:.2} first-claim latency {:.2} last-finish {:.2} threads-in {:.1} pre-dispatch {:.2} pass {:.2}", p[0], p[1], p[2], p[3], p[4], p[5] * nf, p[6], p[7]);
            }
            eprintln!("vmstat deltas (whole box) during decode: numa_hint_faults {} numa_pages_migrated {} thp_split_page {} thp_fault_alloc {} pgfault {}", vmstat("numa_hint_faults") - vm0[0], vmstat("numa_pages_migrated") - vm0[1], vmstat("thp_split_page") - vm0[2], vmstat("thp_fault_alloc") - vm0[3], vmstat("pgfault") - vm0[4]);
            eprintln!("minor faults inside CpuMoe::run: {}", fe_dsc::engine::LANE_FAULTS.load(std::sync::atomic::Ordering::Relaxed));
            eprintln!("minor page faults during decode: {} ({:.0}/token)", minflt() - f0, (minflt() - f0) as f64 / nf);
            {
                let f = *fe_ggq::cpu::PROF_FIT.lock().unwrap();
                if f[0] > 2.0 {
                    let (n, sx, sy, sxy, sxx) = (f[0], f[1], f[2], f[3], f[4]);
                    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
                    let a = (sy - b * sx) / n;
                    eprintln!("lane pass fit over {} calls: time = {:.1} us + {:.1} us per MB stored (marginal {:.1} GB/s); mean call {:.1} MB in {:.0} us", n, a, b, 1e3 / b, sx / n, sy / n);
                }
            }
            eprintln!("expert hits card1 {hit} card0 {hit0} misses {miss} ({:.1}% on the cards, {} resident)", 100.0 * (hit + hit0) as f64 / (hit + hit0 + miss).max(1) as f64, e.resident_experts);
            println!("{}{}", a[1], t.decode(&out, false).map_err(|e| anyhow::anyhow!("{e}"))?);
            eprintln!("ids {:?}", out);
          }
        }
        Some("vocab") => {
            let t = tok::build(&g, &c)?;
            let mut o = String::new();
            for i in 0..c.n_vocab as u32 { o.push_str(&format!("{}\t{:?}\n", i, t.id_to_token(i).unwrap_or_default())); }
            std::fs::write(&a[1], o)?;
        }
        Some("lanes") => {
            // dsc-run lanes <ntok> <prompt tokens> [lanes]: every prompt alone (lane 0) vs all of them at once in separate lanes, one pass a step:
            // the generated ids and a hash of every logits row must be identical
            let ntok: usize = a[1].parse()?;
            let np: usize = a[2].parse()?;
            let nl: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(4);
            // DSC_LANE_PROMPTS = three text files separated by ':' (the fourth lane repeats the first, reworded)
            let lp = std::env::var("DSC_LANE_PROMPTS").map_err(|_| anyhow::anyhow!("dsc-run lanes: set DSC_LANE_PROMPTS=a.txt:b.txt:c.txt"))?;
            let mut files: Vec<String> = lp.split(':').map(String::from).collect();
            anyhow::ensure!(files.len() >= 3, "DSC_LANE_PROMPTS needs three files");
            files.truncate(3);
            files.push(files[0].clone());
            let mut prompts: Vec<Vec<u32>> = vec![];
            for (i, f) in files.iter().take(nl).enumerate() {
                let mut t = std::fs::read_to_string(f)?;
                if i == 3 { t = t.replace("Summarize", "Explain"); }
                let ids = text_ids(&t, np)?;
                prompts.push(ids);
            }
            let hash = |l: &[f32]| -> u64 { l.iter().fold(1469598103934665603u64, |h, v| (h ^ v.to_bits() as u64).wrapping_mul(1099511628211)) };
            let mut e = Engine::new(&g, &c, max_ctx.max(np + ntok + 64))?;
            // solo
            let mut solo: Vec<(Vec<u32>, Vec<u64>)> = vec![];
            for p in &prompts {
                e.activate(0);
                e.reset()?;
                for ch in p[..p.len() - 1].chunks(8) { e.step_rows(ch, false)?; }
                let mut cur = p[p.len() - 1];
                let (mut out, mut hs) = (vec![], vec![]);
                for _ in 0..ntok {
                    let lg = e.step_rows(&[cur], true)?.unwrap().remove(0);
                    cur = argmax(&lg) as u32;
                    out.push(cur);
                    hs.push(hash(&lg));
                }
                solo.push((out, hs));
            }
            // together
            e.activate(0);
            e.reset()?;
            let mut pos = vec![0usize; nl];
            let mut cur: Vec<u32> = prompts.iter().map(|p| p[p.len() - 1]).collect();
            loop {
                // prefill passes: a chunk of each lane that still has prompt (different sizes, rows of several lanes in one pass)
                let mut parts: Vec<(usize, &[u32])> = vec![];
                let mut left = 24usize;
                for l in 0..nl {
                    let pl = prompts[l].len() - 1;
                    if pos[l] < pl && left > 0 {
                        let n = (pl - pos[l]).min(left).min(5 + l);
                        parts.push((l, &prompts[l][pos[l]..pos[l] + n]));
                        left -= n;
                    }
                }
                if parts.is_empty() { break; }
                for &(l, t) in &parts { pos[l] += t.len(); }
                e.step_multi(&parts, false)?;
            }
            let (mut out, mut hs): (Vec<Vec<u32>>, Vec<Vec<u64>>) = (vec![vec![]; nl], vec![vec![]; nl]);
            for _ in 0..ntok {
                let toks: Vec<[u32; 1]> = cur.iter().map(|&t| [t]).collect();
                let parts: Vec<(usize, &[u32])> = (0..nl).map(|l| (l, &toks[l][..])).collect();
                let lg = e.step_multi(&parts, true)?.unwrap();
                for l in 0..nl {
                    cur[l] = argmax(&lg[l]) as u32;
                    out[l].push(cur[l]);
                    hs[l].push(hash(&lg[l]));
                }
            }
            for l in 0..nl {
                let same_ids = out[l] == solo[l].0;
                let same_bits = hs[l] == solo[l].1;
                println!("lane {l}: prompt {} tokens, generated ids {} solo, logits bits {} solo", prompts[l].len(), if same_ids { "==" } else { "!=" }, if same_bits { "==" } else { "!=" });
            }
        }
        Some("tf") => {
            // dsc-run tf <prompt text file> <n> <out log> [forced ids file]: n steps after the prompt (chunked prefill: the big path when DSC_PF_ROWS is set),
            // greedy or teacher-forced; one line a step: the top-5 (id logit) of the logits that chose / precede the token
            let text = std::fs::read_to_string(&a[1])?;
            let n: usize = a[2].parse()?;
            let ids = text_ids(&text, std::env::var("DSC_MAXTOK").ok().and_then(|v| v.parse().ok()).unwrap_or(8192))?;
            let forced: Option<Vec<u32>> = a.get(4).map(|p| std::fs::read_to_string(p).unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect());
            let mut e = Engine::new(&g, &c, max_ctx.max(ids.len() + n + 16))?;
            let mut lg = e.prefill_chunked(&ids, true)?.unwrap();
            let mut log = String::new();
            let mut chosen = vec![];
            for i in 0..n {
                let mut top: Vec<usize> = (0..lg.len()).collect();
                top.select_nth_unstable_by(4, |&x, &y| lg[y].partial_cmp(&lg[x]).unwrap());
                top.truncate(5);
                top.sort_by(|&x, &y| lg[y].partial_cmp(&lg[x]).unwrap());
                log.push_str(&top.iter().map(|&t| format!("{t} {:.4}", lg[t])).collect::<Vec<_>>().join(" "));
                log.push('\n');
                let next = match &forced { Some(f) => f[i], None => top[0] as u32 };
                chosen.push(next);
                lg = e.step(next, true)?.unwrap();
            }
            std::fs::write(&a[3], log)?;
            if forced.is_none() {
                std::fs::write(format!("{}.ids", a[3]), chosen.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "))?;
            }
        }
        Some("logits") => {
            let text = std::fs::read_to_string(&a[1])?;
            let ntok = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
            let ids = text_ids(&text, ntok)?;
            let mut e = Engine::new(&g, &c, max_ctx)?;
            let dump = std::env::var("DSC_DUMP_H").ok();
            let mut last = None;
            if std::env::var("DSC_PREFILL").is_ok_and(|v| v == "chunk") {
                let t0 = std::time::Instant::now();
                e.tap_layers = dump.is_some();
                last = e.prefill_chunked(&ids, true)?;
                eprintln!("chunked prefill {} tokens {:.2}s ({:.1} tok/s)", ids.len(), t0.elapsed().as_secs_f32(), ids.len() as f32 / t0.elapsed().as_secs_f32());
            } else {
                let t0 = std::time::Instant::now();
                for (i, &id) in ids.iter().enumerate() {
                    e.tap_layers = dump.is_some() && i + 1 == ids.len();
                    last = e.step(id, i + 1 == ids.len())?;
                }
                eprintln!("token-by-token prefill {} tokens {:.2}s ({:.1} tok/s)", ids.len(), t0.elapsed().as_secs_f32(), ids.len() as f32 / t0.elapsed().as_secs_f32());
            }
            let lg = last.unwrap();
            if let Some(p) = dump {
                let b: Vec<u8> = e.layer_h.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(p, b)?;
            }
            let out = a.get(3).cloned().unwrap_or_else(|| "/dev/null".into());
            std::fs::write(out, lg.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            let mut top: Vec<usize> = (0..lg.len()).collect();
            top.sort_by(|&x, &y| lg[y].partial_cmp(&lg[x]).unwrap());
            for &t in &top[..10] {
                println!("{t} {:.4}", lg[t]);
            }
        }
        Some("ppl") => {
            let text = std::fs::read_to_string(&a[1])?;
            let ntok = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
            let ids = text_ids(&text, ntok)?;
            let mut e = Engine::new(&g, &c, max_ctx.max(ids.len() + 8))?;
            let t0 = std::time::Instant::now();
            let (mut nll, mut top1) = (0f64, vec![]);
            for i in 0..ids.len() - 1 {
                let lg = e.step(ids[i], true)?.unwrap();
                nll -= logprob(&lg, ids[i + 1] as usize);
                top1.push(argmax(&lg));
            }
            nll /= (ids.len() - 1) as f64;
            println!("tokens {} mean NLL {:.4} perplexity {:.3} ({:.1}s)", ids.len(), nll, nll.exp(), t0.elapsed().as_secs_f32());
            if let Ok(p) = std::env::var("DSC_TOP1") {
                std::fs::write(p, top1.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            }
        }
        Some("trace") => {
            // dsc-run trace <list file> <out.bin> [ntok]: engine routing trace (45-byte records: seq u32, layer u8, row u32, ids 6 x u16, w 6 x f32)
            let list = std::fs::read_to_string(&a[1])?;
            let ntok = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
            // resumable: <out>.done holds the number of finished sequences, the output is appended one whole sequence at a time
            let done_p = format!("{}.done", a[2]);
            let done: usize = std::fs::read_to_string(&done_p).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
            let want = std::fs::metadata(&a[2]).map(|m| m.len()).unwrap_or(0);
            let keep = std::fs::read_to_string(format!("{}.size", a[2])).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0);
            let mut f = if done > 0 && want >= keep {
                let fh = std::fs::OpenOptions::new().write(true).open(&a[2])?;
                fh.set_len(keep)?;
                let mut fh = fh;
                std::io::Seek::seek(&mut fh, std::io::SeekFrom::Start(keep))?;
                std::io::BufWriter::new(fh)
            } else {
                std::io::BufWriter::new(std::fs::File::create(&a[2])?)
            };
            let done = if done > 0 && want >= keep { done } else { 0 };
            let mut written = if done > 0 { keep } else { 0 };
            let mut e = Engine::new(&g, &c, max_ctx.max(ntok + 8))?;
            for (seq, path) in list.lines().filter(|l| !l.trim().is_empty()).enumerate() {
                if seq < done {
                    continue;
                }
                let text = std::fs::read_to_string(path)?;
                let ids = text_ids(&text, ntok)?;
                e.reset()?;
                e.trace = Some(vec![]);
                let t0 = std::time::Instant::now();
                for &id in &ids {
                    e.step(id, false)?;
                }
                for (l, row, i6, w6) in e.trace.take().unwrap() {
                    let mut r = Vec::with_capacity(45);
                    r.extend_from_slice(&((seq + std::env::var("DSC_SEQ0").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0)) as u32).to_le_bytes());
                    r.push(l);
                    r.extend_from_slice(&row.to_le_bytes());
                    for v in i6 { r.extend_from_slice(&v.to_le_bytes()); }
                    for v in w6 { r.extend_from_slice(&v.to_le_bytes()); }
                    std::io::Write::write_all(&mut f, &r)?;
                    written += 45;
                }
                std::io::Write::flush(&mut f)?;
                std::fs::write(format!("{}.size", a[2]), written.to_string())?;
                std::fs::write(&done_p, (seq + 1).to_string())?;
                eprintln!("trace seq {seq} {path}: {} tokens {:.1}s", ids.len(), t0.elapsed().as_secs_f32());
            }
        }
        Some("verify") => {
            // dsc-run verify <text> <prompt tokens> <rows> [reps]: the cost of one R-row pass (feed R real tokens, roll back), phases and distinct experts
            let text = std::fs::read_to_string(&a[1])?;
            let np: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
            let r: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
            let reps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);
            let ids = text_ids(&text, np + 64)?;
            let mut e = Engine::new(&g, &c, max_ctx.max(ids.len() + 64))?;
            for chunk in ids[..np].chunks(6) {
                e.step_rows(chunk, false)?;
            }
            for rows in [1usize, r] {
                let mut ph = [0f64; 10];
                let (mut tot, mut gpu, mut cpu) = (0f64, 0f64, 0f64);
                let (mut dist, mut hit, mut hit0, mut miss) = (0u32, 0u32, 0u32, 0u32);
                for k in 0..reps + 2 {
                    let p = np + (k % 4) * 8;
                    e.truncate(p);
                    e.step_rows(&ids[p..p + rows], std::env::var("DSC_WANT").is_ok())?;
                    if k >= 2 {
                        tot += e.split.total_ms;
                        gpu += e.split.gpu_ms;
                        cpu += e.split.cpu_ms;
                        for i in 0..10 { ph[i] += e.split.ph[i]; }
                        dist += e.split.distinct; hit += e.split.hit; hit0 += e.split.hit0; miss += e.split.miss;
                    }
                }
                let n = reps as f64;
                eprintln!("{rows}-row pass: {:.1} ms (gpu path {:.1}, CPU lane {:.1}); phases attention-to-ids {:.1}, card enqueue {:.1}, CpuMoe::run {:.1}, tail {:.1}; expert-card GPU ms {:.1} / {:.1}; per layer distinct CPU experts {:.2}, pairs hit card1 {:.1} card0 {:.1} miss {:.1} of {}",
                    tot / n, gpu / n, cpu / n, ph[0] / n, ph[1] / n, ph[2] / n, ph[3] / n, ph[4] / n, ph[9] / n, dist as f64 / n / 40.0,
                    hit as f64 / n / 40.0, hit0 as f64 / n / 40.0, miss as f64 / n / 40.0, rows * 6);
            }
        }
        Some("rows") => {
            // all rows' logits (rows 0..n-2, f32 LE, the layout of dsc-ref kl's .bin files) of a text, token by token
            let text = std::fs::read_to_string(&a[1])?;
            let ntok = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
            let ids = text_ids(&text, ntok)?;
            let mut e = Engine::new(&g, &c, max_ctx.max(ids.len() + 8))?;
            let mut f = std::io::BufWriter::new(std::fs::File::create(&a[3])?);
            let t0 = std::time::Instant::now();
            let batch: usize = std::env::var("DSC_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            let mut i = 0;
            while i < ids.len() - 1 {
                let n = batch.min(ids.len() - 1 - i);
                if std::env::var("DSC_ROLLBACK").is_ok() {
                    // a rejected verify: feed wrong tokens at these positions, roll back, then feed the real ones
                    let junk: Vec<u32> = (0..n).map(|k| 1000 + 37 * (i + k) as u32 % 50000).collect();
                    e.step_rows(&junk, false)?;
                    e.truncate(i);
                }
                let lgs = e.step_rows(&ids[i..i + n], true)?.unwrap();
                for lg in lgs {
                    std::io::Write::write_all(&mut f, &lg.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                }
                i += n;
            }
            eprintln!("{} rows in {:.1}s", ids.len() - 1, t0.elapsed().as_secs_f32());
        }
        Some("tfrows") => {
            // dsc-run tfrows <ids file> <prompt tokens P> <out.bin>: the first P-1 tokens by chunked prefill (the big-chunk path when DSC_PF_ROWS is
            // set and the prompt is long enough), then every further token of the file in one-row passes; the logits rows from the last
            // prompt token on go to <out.bin> (row i = after token P-1+i), to compare the big path's prefix with the plain path's / the oracle's
            let ids = text_ids(&std::fs::read_to_string(&a[1])?, 1 << 20)?;
            let np: usize = a[2].parse()?;
            let mut e = Engine::new(&g, &c, max_ctx)?;
            let mut f = std::io::BufWriter::new(std::fs::File::create(&a[3])?);
            if np > 1 {
                let mut i = 0;
                while i < np - 1 {
                    let n = e.chunk_for(np - 1 - i);
                    e.step_rows(&ids[i..i + n], false)?;
                    eprintln!("tfrows: prefill pass of {n} rows");
                    i += n;
                }
            }
            let last = if ids[ids.len() - 1] == 1 { ids.len() - 1 } else { ids.len() };
            for &t in &ids[np - 1..last] {
                let lg = e.step(t, true)?.unwrap();
                std::io::Write::write_all(&mut f, &lg.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            }
        }
        Some("stale") => {
            // dsc-run stale <A ids> <B ids> <keep> [clear mask ...]: does state left by an earlier sequence change a later one? B is fed token by
            // token on a fresh engine (reference), then A is fed (its first <A prompt> tokens as chunked prefill when DSC_STALE_AP is set), the
            // sequence is rolled back to <keep> tokens (the common prefix) and B's rest is fed again. Reports the rows whose logits differ.
            // Further arguments: masks for Engine::clear_caches applied after the rollback (which cache carries the difference).
            let a_ids = text_ids(&std::fs::read_to_string(&a[1])?, 1 << 20)?;
            let b_ids = text_ids(&std::fs::read_to_string(&a[2])?, std::env::var("DSC_STALE_NB").ok().and_then(|v| v.parse().ok()).unwrap_or(120))?;
            let keep: usize = a[3].parse()?;
            ensure_eq(&a_ids[..keep], &b_ids[..keep]);
            let mut e = Engine::new(&g, &c, max_ctx)?;
            let run_b = |e: &mut Engine, from: usize| -> Result<Vec<Vec<f32>>> {
                let mut v = vec![];
                for &t in &b_ids[from..] {
                    v.push(e.step(t, true)?.unwrap());
                }
                Ok(v)
            };
            let feed_a = |e: &mut Engine| -> Result<()> {
                let ap: usize = std::env::var("DSC_STALE_AP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                if ap > 0 {
                    e.prefill_chunked(&a_ids[..ap], false)?;
                }
                let rows: usize = std::env::var("DSC_STALE_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
                let mut i = ap;
                while i < a_ids.len() {
                    let n = rows.min(a_ids.len() - i);
                    e.step_rows(&a_ids[i..i + n], false)?;
                    i += n;
                }
                Ok(())
            };
            e.reset()?;
            let fresh = run_b(&mut e, 0)?;
            let report = |label: &str, got: &[Vec<f32>]| {
                let off = fresh.len() - got.len();
                let (mut nd, mut first, mut maxd, mut flips) = (0usize, None, 0f32, 0usize);
                for (i, row) in got.iter().enumerate() {
                    let r = &fresh[off + i];
                    let d = row.iter().zip(r).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                    if d > 0.0 {
                        nd += 1;
                        if first.is_none() { first = Some(off + i); }
                        maxd = maxd.max(d);
                        if argmax(row) != argmax(r) { flips += 1; }
                    }
                }
                println!("{label}: {} rows compared, {nd} differ (first at position {:?}), max |dlogit| {maxd:.4}, argmax flips {flips}", got.len(), first);
            };
            e.reset()?;
            let again = run_b(&mut e, 0)?;
            report("fresh again (after reset)", &again);
            let masks: Vec<u32> = if a.len() > 4 { a[4..].iter().map(|v| v.parse().unwrap()).collect() } else { vec![0] };
            for m in masks {
                e.reset()?;
                feed_a(&mut e)?;
                e.truncate(keep);
                e.clear_caches(m)?;
                let got = run_b(&mut e, keep)?;
                report(&format!("after A ({} tokens), rollback to {keep}, cleared mask {m}", a_ids.len()), &got);
            }
        }
        Some("p2") => {
            // dsc-run p2 <out file> <rows> <name>:<prompt tokens>:<ids file> ...: measurement for the spilled-expert study. One engine (run it with
            // the launcher's environment so the placement is the server's). Per case the prompt is prefilled in chunks, then the rest of the
            // ids is fed `rows` tokens a pass (teacher-forced, logits wanted, as a verify pass of distinct real tokens). One line a pass:
            // "<name> <position> <rows> <ms> <cold picks> l:e:o ..." (layer, expert, owner of every pick of a host-backed expert).
            use std::io::Write;
            let rows: usize = a[2].parse()?;
            let mut e = Engine::new(&g, &c, max_ctx)?;
            let mut f = std::io::BufWriter::new(std::fs::File::create(&a[1])?);
            let hb = e.host_backed();
            let tot: usize = hb.iter().map(|x| x.0).sum();
            let bytes: usize = hb.iter().map(|x| x.0 * (x.1[0] + x.1[1] + x.1[2])).sum();
            writeln!(f, "# host-backed experts {tot}, {:.3} GiB as stored", bytes as f64 / (1u64 << 30) as f64)?;
            for (l, x) in hb.iter().enumerate() {
                let mut own = [0usize; 4];
                for (_, o) in &x.2 {
                    own[(*o as usize).min(3)] += 1;
                }
                writeln!(f, "# layer {l}: host-backed {} of {} experts, bytes gate/up/down {:?}, by owner {:?}, experts {:?}", x.0, c.n_expert, x.1, own, x.2.iter().map(|v| v.0).collect::<Vec<_>>())?;
            }
            for spec in &a[3..] {
                let mut it = spec.splitn(3, ':');
                let (name, np, path) = (it.next().unwrap(), it.next().unwrap().parse::<usize>()?, it.next().unwrap());
                let ids = text_ids(&std::fs::read_to_string(path)?, 1 << 20)?;
                e.reset()?;
                e.prefill_chunked(&ids[..np], false)?;
                e.heat_on();
                // warm the pass shape (graphs are captured on first use)
                let mut i = np;
                let mut k = 0;
                while i + rows <= ids.len() {
                    let t = std::time::Instant::now();
                    e.step_rows(&ids[i..i + rows], true)?;
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    let cp = e.cold_picks(rows);
                    if k >= 3 {
                        writeln!(f, "{name} {i} {rows} {ms:.3} {} {}", cp.len(), cp.iter().map(|(l, x, o)| format!("{l}:{x}:{o}")).collect::<Vec<_>>().join(" "))?;
                    }
                    i += rows;
                    k += 1;
                }
                e.heat_write(&format!("{}.{name}.cnt", a[1]))?;
                eprintln!("p2 {name}: {} passes of {rows} rows", k);
            }
        }
        Some("diag") => {
            // dsc-run diag <out dir> <name>:<n>:<prompt file> ...: one engine for all cases. Per case the prompt is fed token by token, then up to n
            // greedy steps (ends after the stop id 1). Writes <out>/<name>.ids ("ids: ..." = prompt + continuation), <out>/<name>.eng.bin (the logits
            // row after every fed token, f32 LE) and <out>/<name>.routes (the router's picks: run with DSC_GRAPH=0).
            let out = &a[1];
            let mut e = Engine::new(&g, &c, max_ctx)?;
            for spec in &a[2..] {
                let mut it = spec.splitn(3, ':');
                let (name, n, path) = (it.next().unwrap(), it.next().unwrap().parse::<usize>()?, it.next().unwrap());
                let prompt = text_ids(&std::fs::read_to_string(path)?, 1 << 20)?;
                e.reset()?;
                let rf = format!("{out}/{name}.routes");
                let _ = std::fs::remove_file(&rf);
                unsafe { std::env::set_var("DSC_ROUTES", &rf) };
                let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{out}/{name}.eng.bin"))?);
                let mut put = |lg: &[f32]| std::io::Write::write_all(&mut f, &lg.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                let t0 = std::time::Instant::now();
                let mut ids = prompt.clone();
                let mut lg = vec![];
                for &id in &prompt {
                    lg = e.step(id, true)?.unwrap();
                    put(&lg)?;
                }
                for _ in 0..n {
                    let next = argmax(&lg) as u32;
                    ids.push(next);
                    if next == 1 {
                        break;
                    }
                    lg = e.step(next, true)?.unwrap();
                    put(&lg)?;
                }
                std::fs::write(format!("{out}/{name}.ids"), format!("ids: {}", ids.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")))?;
                eprintln!("diag {name}: prompt {} + {} generated in {:.1}s", prompt.len(), ids.len() - prompt.len(), t0.elapsed().as_secs_f32());
            }
        }
        Some("cmp") => {
            // dsc-run cmp <ref.bin> <eng.bin>: KL(ref || eng) per row and top-1 agreement (rows of the vocabulary)
            let v = c.n_vocab;
            let rd = |p: &str| -> Result<Vec<f32>> {
                let b = std::fs::read(p)?;
                Ok(b.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect())
            };
            let (r, e) = (rd(&a[1])?, rd(&a[2])?);
            let n = r.len().min(e.len()) / v;
            let lsm = |l: &[f32]| -> Vec<f64> {
                let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let z: f64 = l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln();
                l.iter().map(|&x| x as f64 - m - z).collect()
            };
            let (mut kls, mut agree) = (vec![], 0usize);
            for i in 0..n {
                let (lr, le) = (lsm(&r[i * v..(i + 1) * v]), lsm(&e[i * v..(i + 1) * v]));
                kls.push(lr.iter().zip(&le).map(|(p, q)| p.exp() * (p - q)).sum::<f64>());
                agree += (argmax(&r[i * v..(i + 1) * v]) == argmax(&e[i * v..(i + 1) * v])) as usize;
            }
            let mut s = kls.clone();
            s.sort_by(|x, y| x.partial_cmp(y).unwrap());
            println!("rows {n}: mean KL {:.5} median {:.5} p99 {:.5} max {:.5} (worst row {}) top-1 agreement {:.4}",
                kls.iter().sum::<f64>() / n as f64, s[n / 2], s[(n * 99 / 100).min(n - 1)], s[n - 1],
                kls.iter().position(|&x| x == s[n - 1]).unwrap(), agree as f64 / n as f64);
        }
        _ => bail!("usage: dsc-run gen|logits|ppl|rows|cmp ..."),
    }
    Ok(())
}
