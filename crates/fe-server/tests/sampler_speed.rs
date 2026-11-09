//! Host sampling cost on a full 248320-token vocabulary (run: cargo test --release --test sampler_speed -- --nocapture).
use fe_server::engine::SamplingParams;
use fe_server::sampler::{Rng, Sampler};
use std::time::Instant;

fn logits(seed: u64) -> Vec<u16> {
    // roughly LLM-shaped: N(0, 2.5) background, a handful of strong candidates
    let mut r = Rng::new(seed);
    let mut v: Vec<f32> = (0..248_320)
        .map(|_| {
            let (a, b) = (r.next_f64().max(1e-12), r.next_f64());
            ((-2.0 * a.ln()).sqrt() * (6.283_185 * b).cos() * 2.5) as f32
        })
        .collect();
    for k in 0..8 {
        v[(r.next_u64() % 248_320) as usize] = 14.0 - k as f32;
    }
    v.iter().map(|f| (f.to_bits() >> 16) as u16).collect()
}

#[test]
fn sampling_speed() {
    let rows: Vec<Vec<u16>> = (0..20).map(logits).collect();
    for (name, p) in [
        ("greedy", SamplingParams { temperature: 0.0, ..Default::default() }),
        ("qwen default t1 k20 p0.95", SamplingParams { temperature: 1.0, top_k: 20, top_p: 0.95, ..Default::default() }),
        ("t0.7 top_p 0.9 only", SamplingParams { temperature: 0.7, top_p: 0.9, ..Default::default() }),
        ("t1 no filters", SamplingParams { temperature: 1.0, ..Default::default() }),
        ("t1 min_p 0.05", SamplingParams { temperature: 1.0, min_p: 0.05, ..Default::default() }),
        ("t1 k20 + penalties", SamplingParams { temperature: 1.0, top_k: 20, presence_penalty: 1.5, repetition_penalty: 1.1, ..Default::default() }),
    ] {
        let prompt: Vec<u32> = (0..2000).collect();
        let mut s = Sampler::new(&SamplingParams { seed: Some(1), ..p }, &prompt);
        let t = Instant::now();
        for (i, row) in rows.iter().cycle().take(100).enumerate() {
            let tok = s.sample_bf16(row);
            s.accept(tok);
            std::hint::black_box((i, tok));
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
        println!("{name:>28}: {ms:.3} ms / token");
        assert!(ms < 8.0, "{name} too slow: {ms} ms");
    }
}
