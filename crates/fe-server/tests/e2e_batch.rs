//! The real-engine path end to end (router → BatchEngine → slot model) with the CPU fake model in place of
//! fe_model: usage accounting, stream == non-stream, prompt-cache hits reported, 6 streams on 4 slots,
//! disconnect frees the slot.
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use fe_server::batch::fake::{FakeModel, reference};
use fe_server::batch::{BatchConfig, BatchEngine};
use fe_server::engine::FinishReason;
use fe_server::server::{AppState, GenDefaults, ServerConfig, router};
use fe_server::template::ChatTemplate;
use fe_server::tokenizer::Tok;
use http_body_util::BodyExt;
use serde_json::{Value as Json, json};
use tower::ServiceExt;

struct Setup {
    state: Arc<AppState>,
    eos: u32,
    text_vocab: usize,
}

const PERIOD: u64 = 61;

fn setup(delay_ms: u64) -> Option<Setup> {
    let dir = std::env::var("FE_MODEL_DIR").map_or_else(|_| fe_site::nvfp4_dir(), PathBuf::from);
    if !dir.join("tokenizer.json").exists() {
        eprintln!("skipped: model dir not found (set FE_MODEL_DIR)");
        return None;
    }
    let tok = Arc::new(Tok::from_dir(&dir).unwrap());
    let vocab = tok.vocab_size();
    let eos = tok.token_id("<|im_end|>").unwrap();
    let text_vocab = tok.token_id("<|endoftext|>").unwrap() as usize;
    let engine = BatchEngine::spawn(BatchConfig { name: "fake".into(), prefill_chunk: 256 }, move || {
        let mut m = FakeModel::new(4, 8192, 256 + 4, vocab, eos);
        m.text_vocab = text_vocab;
        m.eos_period = PERIOD;
        m.delay = Duration::from_millis(delay_ms);
        Ok(m)
    })
    .unwrap();
    let cfg = ServerConfig { model_id: "spark".into(), max_concurrent: 4, max_context: 8192, defaults: GenDefaults::from_dir(&dir, &tok) };
    let state = AppState::new(Arc::new(engine), tok, Arc::new(ChatTemplate::from_dir(&dir).unwrap()), cfg);
    Some(Setup { state, eos, text_vocab })
}

async fn post(s: &Arc<AppState>, path: &str, body: Json) -> (StatusCode, Vec<u8>) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = router(s.clone()).oneshot(req).await.unwrap();
    let status = resp.status();
    (status, resp.into_body().collect().await.unwrap().to_bytes().to_vec())
}

fn sse(bytes: &[u8]) -> Vec<Json> {
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let datas: Vec<&str> = text.split("\n\n").filter(|b| !b.trim().is_empty()).map(|b| b.strip_prefix("data: ").unwrap()).collect();
    assert_eq!(datas.last(), Some(&"[DONE]"));
    datas[..datas.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completions_usage_and_prompt_cache() {
    let Some(su) = setup(0) else { return };
    let s = &su.state;
    let p1: Vec<u32> = (0..300u32).map(|i| (i * 37 + 11) % 5000 + 10).collect();
    let (want, reason) = reference(&p1, 40, su.text_vocab, su.eos, PERIOD);
    let (status, bytes) = post(s, "/v1/completions", json!({"prompt": p1, "max_tokens": 40, "temperature": 0})).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
    let v: Json = serde_json::from_slice(&bytes).unwrap();
    let stopped = matches!(reason, FinishReason::Stop(_));
    assert_eq!(v["usage"]["completion_tokens"].as_u64().unwrap() as usize, want.len() + stopped as usize);
    assert_eq!(v["choices"][0]["finish_reason"], if stopped { "stop" } else { "length" });
    assert_eq!(v["choices"][0]["text"].as_str().unwrap(), s.tok.decode(&want, true).unwrap());
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 0);

    // the follow-up prompt extends everything the slot holds: prompt + output (minus the unforwarded last token)
    let mut p2 = p1.clone();
    p2.extend_from_slice(&want);
    p2.extend_from_slice(&[101, 102, 103]);
    let (_, bytes) = post(s, "/v1/completions", json!({"prompt": p2, "max_tokens": 10, "temperature": 0})).await;
    let v: Json = serde_json::from_slice(&bytes).unwrap();
    let in_slot = p1.len() + want.len() - (!stopped) as usize;
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap() as usize, in_slot);
    let (want2, _) = reference(&p2, 10, su.text_vocab, su.eos, PERIOD);
    assert_eq!(v["choices"][0]["text"].as_str().unwrap(), s.tok.decode(&want2, true).unwrap(), "cache hit changed the output");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chat_stream_equals_non_stream_and_sampling_is_seeded() {
    let Some(su) = setup(0) else { return };
    let s = &su.state;
    let body = json!({"messages": [{"role": "user", "content": "hello batch engine"}], "max_tokens": 80, "temperature": 0});
    let (_, bytes) = post(s, "/v1/chat/completions", body.clone()).await;
    let full: Json = serde_json::from_slice(&bytes).unwrap();
    let mut sb = body.clone();
    sb["stream"] = json!(true);
    sb["stream_options"] = json!({"include_usage": true});
    let (_, bytes) = post(s, "/v1/chat/completions", sb).await;
    let events = sse(&bytes);
    let (mut reasoning, mut content) = (String::new(), String::new());
    for e in &events {
        for ch in e["choices"].as_array().unwrap() {
            reasoning.push_str(ch["delta"].get("reasoning_content").and_then(Json::as_str).unwrap_or(""));
            content.push_str(ch["delta"].get("content").and_then(Json::as_str).unwrap_or(""));
        }
    }
    let m = &full["choices"][0]["message"];
    assert_eq!(reasoning, m.get("reasoning_content").and_then(Json::as_str).unwrap_or(""));
    assert_eq!(content, m["content"].as_str().unwrap_or(""));
    assert_eq!(events.last().unwrap()["usage"]["completion_tokens"], full["usage"]["completion_tokens"]);
    // second identical request: served from the same slot history? no: the slot holds prompt + output,
    // which is not a prefix of the same prompt again, so it re-prefills and matches exactly
    let (_, bytes) = post(s, "/v1/chat/completions", body.clone()).await;
    let again: Json = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(again["choices"][0]["message"], full["choices"][0]["message"]);
    // seeded sampling through the host sampler is reproducible
    let seeded = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 30, "temperature": 1.0, "seed": 5});
    let a: Json = serde_json::from_slice(&post(s, "/v1/chat/completions", seeded.clone()).await.1).unwrap();
    let b: Json = serde_json::from_slice(&post(s, "/v1/chat/completions", seeded).await.1).unwrap();
    assert_eq!(a["choices"][0]["message"], b["choices"][0]["message"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn six_streams_four_slots_and_disconnect() {
    let Some(su) = setup(3) else { return };
    let s = su.state.clone();
    let mut hs = Vec::new();
    for i in 0..6 {
        let s = s.clone();
        hs.push(tokio::spawn(async move {
            let (status, bytes) = post(&s, "/v1/chat/completions", json!({"stream": true, "max_tokens": 40, "temperature": 0,
                "messages": [{"role": "user", "content": format!("stream number {i}")}], "ignore_eos": true})).await;
            assert_eq!(status, StatusCode::OK);
            let events = sse(&bytes);
            events.iter().rev().find_map(|e| e["choices"][0]["finish_reason"].as_str().map(str::to_string))
        }));
    }
    let mut peak = 0;
    let t0 = Instant::now();
    while !hs.iter().all(|h| h.is_finished()) {
        peak = peak.max(s.running());
        assert!(t0.elapsed() < Duration::from_secs(30));
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    for h in hs {
        assert_eq!(h.await.unwrap().as_deref(), Some("length"));
    }
    assert!((2..=4).contains(&peak), "peak running {peak}");

    // a dropped stream frees its slot
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(json!({"stream": true, "max_tokens": 5000, "ignore_eos": true, "temperature": 0,
            "messages": [{"role": "user", "content": "long one"}]}).to_string()))
        .unwrap();
    let mut body = router(s.clone()).oneshot(req).await.unwrap().into_body();
    for _ in 0..4 {
        body.frame().await.unwrap().unwrap();
    }
    assert_eq!(s.running(), 1);
    drop(body);
    let t0 = Instant::now();
    while s.running() != 0 {
        assert!(t0.elapsed() < Duration::from_secs(5), "slot not freed after disconnect");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
