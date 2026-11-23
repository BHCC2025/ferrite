//! ink-run: self-test, benchmark and accuracy check for Inkling-2-Mini on Ferrite (one NVIDIA card; CUDA_VISIBLE_DEVICES picks it).
//!
//!   ink-run selftest [--n N] [--deep N]   bit-exactness checks: prompt chunks of 1 / 7 / 20 rows give the same logits as one chunk, decode rows equal
//!                                         prompt rows, an 8-row speculative forward and a rollback equal single decodes, the L2-clip / log-scale kernel
//!                                         against its formula, and drafted generation (draft head on) == plain generation. --deep N also prefills
//!                                         N tokens and checks decode rows against prompt rows at that depth. Exit status 1 when a check fails.
//!   ink-run bench [--ctx N] [--gen G] [--kind code|prose] [--draft on|off]
//!                                         a context of N tokens (repository text, repeated beyond its ~40K distinct tokens; its last 1024 tokens are
//!                                         code or prose, which the generated text continues), prefill and G greedy tokens; prints tokens per second
//!   ink-run kl --ref DIR --set NAME --ids FILE.u32
//!                                         free-running KL(reference || engine) and perplexity against reference logits (DIR/NAME/taps.json + taps.bin,
//!                                         `logprobs_f16` rows of the publisher's reference implementation)
//!   ink-run fold                          compute (or show) the draft head's fold tables
//!
//! Options: --model DIR (default INK_DIR, else $FERRITE_MODELS/Inkling-2-Mini), --tmax N prompt rows per forward (default 256; 0 = sized from free
//! memory). The engine switches of docs/inkling-2-mini.md (INK_G16, INK_KVQ6, ...) apply.
use anyhow::{bail, Context, Result};
use fe_ink::cpuref::{bf16_bits, bf16_f32};
use fe_ink::engine::Ink;
use fe_ink::spec::{argmax, Spec, EOS};
use fe_ink::sampler::SampleCfg;
use fe_ink::*;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------------------------------------------- text
const PROSE: [&str; 2] = [include_str!("../../../../CHANGELOG.md"), include_str!("../../../../README.md")];
const CODE: [&str; 4] = [include_str!("../spec.rs"), include_str!("../draft.rs"), include_str!("../engine.rs"), include_str!("../weights.rs")];

struct Corpus {
    prose: Vec<u32>,
    code: Vec<u32>,
}
impl Corpus {
    fn new(model: &Path) -> Result<Corpus> {
        let tk = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
        let enc = |texts: &[&str]| -> Result<Vec<u32>> {
            let mut v = vec![];
            for t in texts {
                v.extend(tk.encode(*t, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().iter().copied());
            }
            Ok(v)
        };
        Ok(Corpus { prose: enc(&PROSE)?, code: enc(&CODE)? })
    }
    /// `n` tokens: mixed text, then the last 1024 from the `kind` part (so the generated text continues code or prose).
    fn context(&self, n: usize, kind: &str) -> Vec<u32> {
        let tail_src = if kind == "prose" { &self.prose } else { &self.code };
        let tail_n = 1024.min(n / 2);
        let tail = &tail_src[tail_src.len() / 3..tail_src.len() / 3 + tail_n.min(tail_src.len() / 3)];
        let mut mixed: Vec<u32> = vec![];
        let (mut pi, mut ci) = (0usize, 0usize);
        let blk = 512;
        while mixed.len() < n - tail.len() {
            for _ in 0..2 {
                for k in 0..blk { mixed.push(self.prose[(pi + k) % self.prose.len()]); }
                pi += blk;
            }
            for k in 0..blk { mixed.push(self.code[(ci + k) % self.code.len()]); }
            ci += blk;
        }
        mixed.truncate(n - tail.len());
        mixed.extend_from_slice(tail);
        mixed
    }
}

// ---------------------------------------------------------------------------------------------------------------- helpers
fn softcap(x: f32) -> f32 {
    30.0 * (x / 30.0).tanh()
}
fn log_softmax(row: &[f32]) -> Vec<f64> {
    let m = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let z: f64 = row.iter().map(|&x| (x as f64 - m).exp()).sum();
    row.iter().map(|&x| x as f64 - m - z.ln()).collect()
}
fn f16_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 31) as u32;
    let m = (h & 1023) as u32;
    if e == 0 {
        let v = m as f32 / 16777216.0;
        return if s != 0 { -v } else { v };
    }
    if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13));
    }
    f32::from_bits(s | ((e + 112) << 23) | (m << 13))
}
fn prefill(ink: &mut Ink, ids: &[u32], rows_last: usize) -> Result<Vec<u16>> {
    let tmax = ink.tmax;
    ink.prompt = true;
    let (mut pos, mut last) = (0, vec![]);
    while pos < ids.len() {
        let t = (ids.len() - pos).min(tmax);
        last = ink.forward(&ids[pos..pos + t], if pos + t == ids.len() { rows_last.min(t) } else { 1 }, None, None)?;
        pos += t;
    }
    Ok(last)
}
fn fnv(v: &[u32]) -> u64 {
    v.iter().flat_map(|x| x.to_le_bytes()).fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// Greedy generation of `n` tokens after `ids`, with the draft head (`spec`) or plain. Returns (tokens, prefill s, decode s).
fn generate(ink: &mut Ink, spec: Option<&mut Spec>, ids: &[u32], n: usize) -> Result<(Vec<u32>, f64, f64)> {
    let t0 = std::time::Instant::now();
    let mut spec = spec;
    let lg = if let Some(sp) = spec.as_deref_mut() {
        sp.reset(ink)?;
        sp.begin(SampleCfg::greedy(), ids);
        sp.prefill(ink, ids)?
    } else {
        ink.reset()?;
        ink.draft_taps = false;
        prefill(ink, ids, 1)?
    };
    ink.s.sync()?;
    let tp = t0.elapsed().as_secs_f64();
    ink.prompt = false;
    let mut rng = 12345u64;
    let first = if let Some(sp) = spec.as_deref_mut() { sp.first_token(ink, &lg, &mut rng)? } else { argmax(&lg) };
    let mut out = vec![first];
    let t1 = std::time::Instant::now();
    while out.len() < n && *out.last().unwrap() != EOS {
        if let Some(sp) = spec.as_deref_mut() {
            let anchor = *out.last().unwrap();
            let new = sp.step(ink, anchor, &mut rng)?;
            out.extend(new);
            if let Some(p) = out.iter().position(|&x| x == EOS) {
                out.truncate(p + 1);
                break;
            }
        } else {
            let lg = ink.forward(&[*out.last().unwrap()], 1, None, None)?;
            out.push(argmax(&lg));
        }
    }
    out.truncate(n);
    Ok((out, tp, t1.elapsed().as_secs_f64()))
}

struct Checks {
    failed: usize,
}
impl Checks {
    fn report(&mut self, name: &str, ok: bool, detail: String) {
        println!("{} {name}: {detail}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            self.failed += 1;
        }
    }
}

fn load_spec(ink: &mut Ink, model: &Path) -> Result<Spec> {
    let fold = fold::ensure(model)?;
    Spec::load(ink, model, &fold)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let opt = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let model: PathBuf = opt("--model").map(PathBuf::from).unwrap_or_else(model_dir);
    let tmax: usize = opt("--tmax").map(|s| s.parse().unwrap()).unwrap_or(256);
    match cmd {
        "selftest" => {
            let n: usize = opt("--n").map(|s| s.parse().unwrap()).unwrap_or(1500);
            let corpus = Corpus::new(&model)?;
            let ids = corpus.context(n, "code");
            let mut ck = Checks { failed: 0 };
            let mut ink = Ink::load(&model, 0, n.max(2048) + 640, tmax)?;
            let tmax = ink.tmax;
            // 1. prompt family: chunks of 7 / 20 / 1 rows against the chunks the engine uses
            let full = prefill(&mut ink, &ids, 1)?;
            for chunk in [7usize, 20, 1] {
                ink.reset()?;
                ink.prompt = true;
                let (mut pos, mut lg) = (0, vec![]);
                while pos < n {
                    let t = (n - pos).min(chunk);
                    lg = ink.forward(&ids[pos..pos + t], 1, None, None)?;
                    pos += t;
                }
                let d = lg.iter().zip(&full).filter(|(a, b)| a != b).count();
                ck.report(&format!("prompt chunks of {chunk} == chunks of {tmax}"), d == 0, format!("{d} of {VOCAB} logits differ"));
            }
            // 2. decode rows: two runs identical, and equal to the prompt rows of the same positions
            let split = n - 8;
            let mut runs: Vec<Vec<Vec<u16>>> = vec![];
            for _ in 0..2 {
                ink.reset()?;
                prefill(&mut ink, &ids[..split], 1)?;
                ink.prompt = false;
                let mut v = vec![];
                for i in 0..8 {
                    v.push(ink.forward(&ids[split + i..split + i + 1], 1, None, None)?);
                }
                runs.push(v);
            }
            ck.report("decode rows: two runs identical", runs[0] == runs[1], String::new());
            ink.reset()?;
            let rows = prefill(&mut ink, &ids, 8)?;
            let nd = (0..8).filter(|&i| runs[0][i][..] != rows[i * VOCAB..(i + 1) * VOCAB]).count();
            ck.report("decode rows == prompt rows", nd == 0, format!("{nd} of 8 rows differ"));
            // 3. 8-row speculative forward and rollback
            ink.reset()?;
            prefill(&mut ink, &ids[..split], 1)?;
            ink.prompt = false;
            ink.spec = true;
            let sp_rows = ink.forward(&ids[split..split + 8], 8, None, None)?;
            let d1 = (0..8).filter(|&i| sp_rows[i * VOCAB..(i + 1) * VOCAB] != runs[0][i][..]).count();
            ink.rollback(5)?;
            ink.spec = false;
            let mut d2 = 0;
            for i in 5..8 {
                if ink.forward(&ids[split + i..split + i + 1], 1, None, None)? != runs[0][i] {
                    d2 += 1;
                }
            }
            ck.report("8-row speculative forward + rollback(5)", d1 == 0 && d2 == 0, format!("{d1} of 8 rows, then {d2} of 3 decodes differ"));
            // 4. L2 clip + log scale against the formula
            let cker = fe_ink::ckpt::Ckpt::open(&model.join("model-00001-of-00001.safetensors"))?;
            let mut rng = 0x9e3779b97f4a7c15u64;
            let mut nxt = || { rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; (rng.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64 };
            let (mut worst, mut total) = (0usize, 0usize);
            for (name, heads, dh, thr) in [("q_l2_clip", 24usize, 128usize, 22.627417f32), ("rel_l2_clip", 24, 16, 8.0)] {
                let sc = cker.read(&format!("model.llm.layers.5.attn.{name}.scale"))?;
                let scu: Vec<u16> = sc.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                for &pos in &[0u32, 100, 8190, 8191, 8192, 8193, 12000, 16383, 100000] {
                    for trial in 0..20 {
                        let amp = [0.5f32, 4.0, 60.0][trial % 3];
                        let x: Vec<u16> = (0..heads * dh).map(|_| bf16_bits(((nxt() * 2.0 - 1.0) as f32) * amp)).collect();
                        let got = ink.test_headclip(&x, &scu, heads, dh, thr, true, pos)?;
                        let tau = 1.0f32 + 0.1 * ((pos as f32 + 1.0) / 8192.0).max(1.0).ln();
                        for h in 0..heads {
                            let v: Vec<f32> = (0..dh).map(|e| bf16_f32(x[h * dh + e]) * bf16_f32(scu[h * dh + e])).collect();
                            let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
                            let f = (thr / (norm + 1e-6)).min(1.0);
                            for e in 0..dh {
                                let want = bf16_bits(bf16_f32(bf16_bits(v[e] * f)) * tau);
                                worst = worst.max((want as i32 - got[h * dh + e] as i32).abs() as usize);
                                total += 1;
                            }
                        }
                    }
                }
            }
            ck.report("L2 clip + log scale vs formula", worst <= 1, format!("{total} values, worst difference {worst} bf16 ulp"));
            // 5. drafted generation == plain generation (greedy), on a code and a prose prompt
            match load_spec(&mut ink, &model) {
                Ok(mut spec) => {
                    for kind in ["code", "prose"] {
                        let p = corpus.context(1200, kind);
                        let (a, _, ta) = generate(&mut ink, Some(&mut spec), &p, 160)?;
                        let (b, _, tb) = generate(&mut ink, None, &p, 160)?;
                        ck.report(&format!("draft head on == off ({kind} prompt)"), a == b,
                            format!("{} tokens, hash {:016x}; {:.0} tok/s drafted, {:.0} plain", a.len(), fnv(&a), a.len() as f64 / ta, b.len() as f64 / tb));
                    }
                }
                Err(e) => ck.report("draft head", false, format!("{e:#}")),
            }
            // 6. a long context: prefill, decode, decode rows against prompt rows at that depth
            if let Some(deep) = opt("--deep").map(|s| s.parse::<usize>().unwrap()) {
                drop(ink);
                let mut ink = Ink::load(&model, 0, deep + 128, tmax)?;
                let ids = corpus.context(deep, "code");
                let t0 = std::time::Instant::now();
                prefill(&mut ink, &ids[..deep - 8], 1)?;
                let tp = t0.elapsed().as_secs_f64();
                ink.prompt = false;
                let mut a = vec![];
                for i in 0..8 {
                    a.push(ink.forward(&ids[deep - 8 + i..deep - 7 + i], 1, None, None)?);
                }
                ink.reset()?;
                let b = prefill(&mut ink, &ids, 8)?;
                let nd = (0..8).filter(|&i| a[i][..] != b[i * VOCAB..(i + 1) * VOCAB]).count();
                ck.report(&format!("{deep} tokens: decode rows == prompt rows"), nd == 0, format!("{nd} of 8 rows differ, prefill {:.0} tok/s", (deep - 8) as f64 / tp));
            }
            if ck.failed > 0 {
                println!("selftest: {} check(s) FAILED", ck.failed);
                std::process::exit(1);
            }
            println!("selftest: all checks passed");
        }
        "bench" => {
            let ctx: usize = opt("--ctx").map(|s| s.parse().unwrap()).unwrap_or(2048);
            let ngen: usize = opt("--gen").map(|s| s.parse().unwrap()).unwrap_or(200);
            let kind = opt("--kind").unwrap_or("code".into());
            let draft = opt("--draft").map(|v| v != "off").unwrap_or(true);
            let corpus = Corpus::new(&model)?;
            let ids = corpus.context(ctx, &kind);
            let mut ink = Ink::load(&model, 0, ids.len() + ngen + 16, tmax)?;
            let mut spec = if draft { Some(load_spec(&mut ink, &model)?) } else { None };
            // two runs: the first warms the expert residency on cards that keep experts in host memory; the second is the one reported
            for run in 0..2 {
                let (out, tp, td) = generate(&mut ink, spec.as_mut(), &ids, ngen)?;
                let hosted = ink.host_experts();
                println!("run {}: {kind} context {} tokens, chunk {}, draft {}: prefill {:.0} tok/s ({tp:.2} s); {} tokens decoded at {:.1} tok/s{}{}; hash {:016x}",
                    run + 1, ids.len(), ink.tmax, if draft { "on" } else { "off" }, ids.len() as f64 / tp, out.len(), out.len() as f64 / td,
                    if hosted > 0 { format!("; {hosted} experts in host memory") } else { String::new() },
                    spec.as_ref().map(|s| format!("; {:.2} tokens a pass", s.stats.tokens as f64 / s.stats.passes.max(1) as f64)).unwrap_or_default(), fnv(&out));
            }
        }
        "kl" => {
            let root = PathBuf::from(opt("--ref").context("--ref DIR")?);
            let set = opt("--set").context("--set NAME")?;
            let ids: Vec<u32> = std::fs::read(opt("--ids").context("--ids FILE.u32")?)?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let n = ids.len();
            let dir = root.join(&set);
            let j: Value = serde_json::from_slice(&std::fs::read(dir.join("taps.json"))?)?;
            let bin = std::fs::read(dir.join("taps.bin"))?;
            let mut ent: HashMap<String, (String, Vec<usize>, usize, usize)> = HashMap::new();
            for t in j["tensors"].as_array().context("tensors")? {
                let shape = t["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                ent.insert(t["name"].as_str().unwrap().to_string(), (t["dtype"].as_str().unwrap().to_string(), shape, t["offset"].as_u64().unwrap() as usize, t["nbytes"].as_u64().unwrap() as usize));
            }
            let i32s = |name: &str| -> Result<Vec<i32>> {
                let (dt, _, off, nb) = ent.get(name).with_context(|| format!("no tensor {name}"))?;
                let b = &bin[*off..*off + *nb];
                if dt == "i32" { Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()) } else { bail!("{name}: dtype {dt}") }
            };
            let lp_ent = ent.get("logprobs_f16").context("logprobs_f16")?.clone();
            let lp16 = &bin[lp_ent.2..lp_ent.2 + lp_ent.3];
            let rows = lp_ent.1[0];
            let lpos: Vec<usize> = if rows == n { (0..n).collect() } else { i32s("logit_positions")?.into_iter().map(|x| x as usize).collect() };
            let rowof: HashMap<usize, usize> = lpos.iter().enumerate().map(|(i, &p)| (p, i)).collect();
            let top1 = i32s("top64_ids")?;
            let mut ink = Ink::load(&model, 0, n + 16, tmax)?;
            ink.prompt = true;
            let tmax = ink.tmax;
            let (mut klsum, mut klcnt, mut agree, mut nllsum, mut nllcnt) = (0f64, 0usize, 0usize, 0f64, 0usize);
            let t0 = std::time::Instant::now();
            let mut pos = 0;
            while pos < n {
                let t = (n - pos).min(tmax);
                let logits = ink.forward(&ids[pos..pos + t], t, None, None)?;
                let res: Vec<(f64, f64, bool)> = std::thread::scope(|sc| {
                    let hs: Vec<_> = (0..t).step_by(8).map(|r0| {
                        let (logits, ids, lp16, rowof, top1) = (&logits, &ids, lp16, &rowof, &top1);
                        sc.spawn(move || {
                            let mut v = vec![];
                            for i in r0..(r0 + 8).min(t) {
                                let g = pos + i;
                                let mine: Vec<f32> = logits[i * VOCAB..(i + 1) * VOCAB].iter().map(|&h| softcap(bf16_f32(h))).collect();
                                let lq = log_softmax(&mine);
                                let nll = if g + 1 < n { -lq[ids[g + 1] as usize] } else { 0.0 };
                                let (mut k, mut ag) = (f64::NAN, false);
                                if let Some(&row) = rowof.get(&g) {
                                    let lpr = &lp16[row * VOCAB * 2..(row + 1) * VOCAB * 2];
                                    k = 0.0;
                                    for j in 0..VOCAB {
                                        let p = f16_f32(u16::from_le_bytes([lpr[2 * j], lpr[2 * j + 1]])) as f64;
                                        k += p.exp() * (p - lq[j]);
                                    }
                                    let am = mine.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
                                    ag = am as i32 == top1[row * 64];
                                }
                                v.push((k, nll, ag));
                            }
                            v
                        })
                    }).collect();
                    hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
                });
                for (i, (k, nll, ag)) in res.iter().enumerate() {
                    if !k.is_nan() {
                        klsum += k;
                        klcnt += 1;
                        agree += *ag as usize;
                    }
                    if pos + i + 1 < n {
                        nllsum += nll;
                        nllcnt += 1;
                    }
                }
                pos += t;
            }
            println!("{set}: n={n} | KL(reference || engine) on {klcnt} logit rows {:.5}, top-1 agrees on {agree}/{klcnt} | perplexity {:.3} (nll {:.4}) | {:.1} s ({:.0} tok/s)",
                klsum / klcnt.max(1) as f64, (nllsum / nllcnt.max(1) as f64).exp(), nllsum / nllcnt.max(1) as f64, t0.elapsed().as_secs_f64(), n as f64 / t0.elapsed().as_secs_f64());
        }
        "fold" => {
            let p = fold::ensure(&model)?;
            println!("fold tables: {}", p.display());
        }
        _ => bail!("usage: ink-run selftest [--n N] [--deep N] | bench [--ctx N] [--gen G] [--kind code|prose] [--draft on|off] | kl --ref DIR --set NAME --ids FILE.u32 | fold   (--model DIR, --tmax N)"),
    }
    Ok(())
}
