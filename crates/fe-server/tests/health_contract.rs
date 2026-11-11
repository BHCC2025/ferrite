//! The `/health` contract and the streamed cache count that clients size themselves from (in process, mock engine,
//! no GPU). A client that fans a job out into parallel requests reads `max_batch_size` (how many to send at once),
//! `max_model_len` (the per-request limit) and `running` + `queued` (other requests active) before every call, and
//! `usage.prompt_tokens_details.cached_tokens` of each streamed reply. These names must not change, and the values
//! must be what the loaded engine really does.
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use fe_server::mock::{MockConfig, MockEngine};
use fe_server::server::{AppState, GenDefaults, ServerConfig, router};
use fe_server::template::ChatTemplate;
use fe_server::tokenizer::Tok;
use http_body_util::BodyExt;
use serde_json::{Value as Json, json};
use tower::ServiceExt;

fn model_dir() -> Option<PathBuf> {
    let d = std::env::var("FE_MODEL_DIR").map_or_else(|_| fe_site::nvfp4_dir(), PathBuf::from);
    if d.join("tokenizer.json").exists() {
        Some(d)
    } else {
        eprintln!("skipped: model dir not found (set FE_MODEL_DIR)");
        None
    }
}

/// A server whose flags ask for `flag_concurrent` requests at once and `flag_context` tokens, over a mock engine that
/// decodes `engine.max_batch` at once with `engine.max_context` tokens.
fn state(engine: MockConfig, flag_concurrent: usize, flag_context: usize) -> Option<Arc<AppState>> {
    let dir = model_dir()?;
    let tok = Arc::new(Tok::from_dir(&dir).unwrap());
    let template = Arc::new(ChatTemplate::from_dir(&dir).unwrap());
    let defaults = GenDefaults::from_dir(&dir, &tok);
    let engine = Arc::new(MockEngine::new(tok.clone(), engine));
    let cfg = ServerConfig { model_id: "spark".into(), max_concurrent: flag_concurrent, max_context: flag_context, defaults };
    Some(AppState::new(engine, tok, template, cfg))
}

async fn get(s: &Arc<AppState>, path: &str) -> Json {
    let resp = router(s.clone()).oneshot(Request::get(path).body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap()
}

async fn post(s: &Arc<AppState>, body: Json) -> Vec<u8> {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = router(s.clone()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    resp.into_body().collect().await.unwrap().to_bytes().to_vec()
}

/// The four fields as unsigned integers, read the way a client does: by name.
fn fields(h: &Json) -> (u64, u64, u64, u64) {
    let f = |k: &str| h.get(k).and_then(Json::as_u64).unwrap_or_else(|| panic!("/health has no integer `{k}`: {h}"));
    (f("max_batch_size"), f("max_model_len"), f("running"), f("queued"))
}

#[tokio::test]
async fn health_field_names_and_idle_values() {
    let Some(s) = state(MockConfig { max_batch: 4, ..Default::default() }, 4, 131_072) else { return };
    for path in ["/health", "/v1/health"] {
        let h = get(&s, path).await;
        assert_eq!(fields(&h), (4, 131_072, 0, 0), "{path}: {h}");
        assert_eq!(h["status"], "ok");
    }
    // /v1/models reports the same per-request limit
    assert_eq!(get(&s, "/v1/models").await["data"][0]["max_model_len"], 131_072);
}

#[tokio::test]
async fn health_never_claims_more_than_the_engine_decodes() {
    // a single-sequence engine (max_batch 1, 32K) behind a server started with --max-concurrent 4 --max-model-len 131072
    let engine = MockConfig { max_batch: 1, max_context: 32_768, ..Default::default() };
    let Some(s) = state(engine, 4, 131_072) else { return };
    assert_eq!(fields(&get(&s, "/health").await), (1, 32_768, 0, 0));
    assert_eq!(get(&s, "/v1/models").await["data"][0]["max_model_len"], 32_768);
    // flags below the engine's limits are the server's real limits, and are reported as such
    let Some(s) = state(MockConfig { max_batch: 4, ..Default::default() }, 2, 8192) else { return };
    assert_eq!(fields(&get(&s, "/health").await), (2, 8192, 0, 0));
    // a zero flag still runs one request at a time
    let Some(s) = state(MockConfig { max_batch: 4, ..Default::default() }, 0, 8192) else { return };
    assert_eq!(fields(&get(&s, "/health").await).0, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn health_counts_running_and_queued_requests() {
    // the engine decodes 2 at once; the server was asked for 4: 5 requests sent together = 2 running + 3 queued
    let engine = MockConfig { delay: Duration::from_millis(4), max_batch: 2, ..Default::default() };
    let Some(s) = state(engine, 4, 131_072) else { return };
    let mut handles = Vec::new();
    for i in 0..5 {
        let s = s.clone();
        handles.push(tokio::spawn(async move {
            post(&s, json!({"stream": true, "max_tokens": 120,
                "messages": [{"role": "user", "content": format!("write stream {i}")}]})).await
        }));
    }
    let (mut peak_running, mut saw_full) = (0, false);
    let t0 = Instant::now();
    while !handles.iter().all(|h| h.is_finished()) {
        let (batch, _, running, queued) = fields(&get(&s, "/health").await);
        assert_eq!(batch, 2);
        assert!(running <= batch, "running {running} above max_batch_size {batch}");
        assert!(running + queued <= 5);
        peak_running = peak_running.max(running);
        saw_full |= running == 2 && queued == 3;
        assert!(t0.elapsed() < Duration::from_secs(60), "streams did not finish");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    for h in handles {
        h.await.unwrap();
    }
    assert_eq!(peak_running, 2, "the engine's two slots were never both in use");
    assert!(saw_full, "never saw 2 running + 3 queued");
    let (_, _, running, queued) = fields(&get(&s, "/health").await);
    assert_eq!((running, queued), (0, 0), "counts must return to zero");
}

fn sse_events(bytes: &[u8]) -> Vec<Json> {
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let datas: Vec<&str> =
        text.split("\n\n").filter(|b| !b.trim().is_empty()).map(|b| b.strip_prefix("data: ").unwrap()).collect();
    assert_eq!(datas.last(), Some(&"[DONE]"));
    datas[..datas.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect()
}

#[tokio::test]
async fn streamed_usage_reports_cached_prompt_tokens() {
    // the engine served 7 prompt tokens from its cache
    let Some(s) = state(MockConfig { cached_prompt_tokens: 7, ..Default::default() }, 4, 131_072) else { return };
    let body = |stream: bool, usage: bool| {
        let mut b = json!({"stream": stream, "max_tokens": 12, "messages": [
            {"role": "system", "content": "You are one lane of a parallel job."},
            {"role": "user", "content": "write a short reply"}]});
        if usage {
            b["stream_options"] = json!({"include_usage": true});
        }
        b
    };
    // streaming with stream_options.include_usage: the last chunk carries usage with the cached count
    let ev = sse_events(&post(&s, body(true, true)).await);
    let last = ev.last().unwrap();
    assert_eq!(last["choices"], json!([]), "the usage chunk has no choices: {last}");
    let u = &last["usage"];
    assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 7, "{u}");
    let prompt = u["prompt_tokens"].as_u64().unwrap();
    assert!(prompt > 7 && u["completion_tokens"].as_u64().unwrap() > 0, "{u}");
    assert_eq!(u["total_tokens"].as_u64().unwrap(), prompt + u["completion_tokens"].as_u64().unwrap());
    // every earlier chunk says usage: null (the OpenAI shape with include_usage)
    assert!(ev[..ev.len() - 1].iter().all(|e| e["usage"].is_null()));
    // not streaming: the same object
    let v: Json = serde_json::from_slice(&post(&s, body(false, false)).await).unwrap();
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 7);
    assert_eq!(v["usage"]["prompt_tokens"].as_u64().unwrap(), prompt);
    // streaming without include_usage: no usage anywhere (as the OpenAI API)
    let ev = sse_events(&post(&s, body(true, false)).await);
    assert!(ev.iter().all(|e| e.get("usage").is_none()), "usage without include_usage");
}
