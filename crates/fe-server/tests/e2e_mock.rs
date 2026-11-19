//! End-to-end through the axum router (in process) with the mock engine: chat + completions, stream
//! and not, reasoning, tool calls, stop strings, max_tokens, usage, concurrency and cancellation.
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

fn state_with(delay_ms: u64, max_concurrent: usize) -> Option<Arc<AppState>> {
    let dir = model_dir()?;
    let tok = Arc::new(Tok::from_dir(&dir).unwrap());
    let template = Arc::new(ChatTemplate::from_dir(&dir).unwrap());
    let defaults = GenDefaults::from_dir(&dir, &tok);
    let engine = Arc::new(MockEngine::new(
        tok.clone(),
        MockConfig { delay: Duration::from_millis(delay_ms), max_batch: max_concurrent, ..Default::default() },
    ));
    let cfg = ServerConfig { model_id: "qwen3.8-27b".into(), max_concurrent, max_context: 131_072, defaults };
    Some(AppState::new(engine, tok, template, cfg))
}

fn state() -> Option<Arc<AppState>> {
    state_with(0, 4)
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
    let bytes = resp.into_body().collect().await.unwrap().to_bytes().to_vec();
    (status, bytes)
}

async fn post_json(s: &Arc<AppState>, path: &str, body: Json) -> Json {
    let (status, bytes) = post(s, path, body).await;
    let v: Json = serde_json::from_slice(&bytes).unwrap_or_else(|_| panic!("not JSON: {}", String::from_utf8_lossy(&bytes)));
    assert_eq!(status, StatusCode::OK, "{v}");
    v
}

/// The `data:` payloads of an SSE body (the last one must be `[DONE]`).
fn sse_events(bytes: &[u8]) -> Vec<Json> {
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let datas: Vec<&str> = text
        .split("\n\n")
        .filter(|b| !b.trim().is_empty())
        .map(|b| b.strip_prefix("data: ").unwrap_or_else(|| panic!("bad SSE block {b:?}")))
        .collect();
    assert_eq!(datas.last(), Some(&"[DONE]"), "stream must end with [DONE]");
    datas[..datas.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect()
}

/// Reassemble a streamed chat response: (reasoning, content, tool_calls, finish_reason, usage).
fn assemble(events: &[Json]) -> (String, String, Vec<(String, String, String)>, Option<String>, Option<Json>) {
    let (mut reasoning, mut content, mut calls, mut finish, mut usage) = (String::new(), String::new(), Vec::new(), None, None);
    for (i, ev) in events.iter().enumerate() {
        assert_eq!(ev["model"], "qwen3.8-27b");
        assert_eq!(ev["object"], "chat.completion.chunk");
        assert!(ev["id"].as_str().unwrap().starts_with("chatcmpl-"));
        if let Some(u) = ev.get("usage").filter(|u| !u.is_null()) {
            assert!(ev["choices"].as_array().unwrap().is_empty());
            usage = Some(u.clone());
        }
        for ch in ev["choices"].as_array().unwrap() {
            let d = &ch["delta"];
            if i == 0 {
                assert_eq!(d["role"], "assistant");
            }
            if let Some(s) = d.get("reasoning_content").and_then(Json::as_str) {
                reasoning.push_str(s);
            }
            if let Some(s) = d.get("content").and_then(Json::as_str) {
                content.push_str(s);
            }
            for tc in d.get("tool_calls").and_then(Json::as_array).into_iter().flatten() {
                let idx = tc["index"].as_u64().unwrap() as usize;
                if idx == calls.len() {
                    calls.push((tc["id"].as_str().unwrap().to_string(), tc["function"]["name"].as_str().unwrap().to_string(), String::new()));
                }
                if let Some(a) = tc["function"].get("arguments").and_then(Json::as_str) {
                    calls[idx].2.push_str(a);
                }
            }
            if let Some(f) = ch["finish_reason"].as_str() {
                finish = Some(f.to_string());
            }
        }
    }
    (reasoning, content, calls, finish, usage)
}

fn weather_tools() -> Json {
    json!([
        {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "days": {"type": "integer"}},
                           "required": ["location", "days"]}}},
        {"type": "function", "function": {"name": "web_search", "description": "Search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    ])
}

#[tokio::test]
async fn health_and_models() {
    let Some(s) = state() else { return };
    let resp = router(s.clone()).oneshot(Request::get("/health").body(Body::empty()).unwrap()).await.unwrap();
    let v: Json = serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(v["max_batch_size"], 4);
    let resp = router(s.clone()).oneshot(Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
    let v: Json = serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(v["data"][0]["id"], "qwen3.8-27b");
    assert_eq!(v["data"][0]["max_model_len"], 131_072);
}

#[tokio::test]
async fn chat_non_stream_with_reasoning() {
    let Some(s) = state() else { return };
    let v = post_json(&s, "/v1/chat/completions", json!({"model": "whatever", "messages": [{"role": "user", "content": "Hi there"}]})).await;
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "qwen3.8-27b");
    let m = &v["choices"][0]["message"];
    assert_eq!(m["role"], "assistant");
    assert!(m["reasoning_content"].as_str().unwrap().starts_with("The user wrote: \"Hi there\""), "{m}");
    let content = m["content"].as_str().unwrap();
    assert!(content.starts_with("Hello from the ferrite mock engine! You said: \"Hi there\"."), "{content}");
    assert!(content.contains("日本語のテキスト") && content.ends_with("🦀🚀✨."));
    assert!(m.get("tool_calls").is_none());
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    // usage: prompt = rendered prompt tokens, completion = generated + the eos token
    let prompt_ids = s.tok.encode("<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>\n").unwrap();
    let u = &v["usage"];
    assert_eq!(u["prompt_tokens"].as_u64().unwrap() as usize, prompt_ids.len());
    let reasoning = m["reasoning_content"].as_str().unwrap();
    let generated = s.tok.encode(&format!("{reasoning}\n</think>\n\n{content}")).unwrap().len();
    assert_eq!(u["completion_tokens"].as_u64().unwrap() as usize, generated + 1);
    assert_eq!(u["total_tokens"], u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap());
}

#[tokio::test]
async fn chat_thinking_off() {
    let Some(s) = state() else { return };
    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Hi"}],
        "chat_template_kwargs": {"enable_thinking": false}})).await;
    let m = &v["choices"][0]["message"];
    assert!(m.get("reasoning_content").is_none(), "{m}");
    assert!(m["content"].as_str().unwrap().starts_with("Hello from the ferrite mock engine!"));
}

#[tokio::test]
async fn no_timings_key_without_engine_timings() {
    // the mock (like the CUDA engines) reports no timings: no `timings` key anywhere, stream or not
    let Some(s) = state() else { return };
    let body = json!({"messages": [{"role": "user", "content": "hello"}]});
    let full = post_json(&s, "/v1/chat/completions", body.clone()).await;
    assert!(full.get("timings").is_none());
    let mut sb = body.clone();
    sb["stream"] = json!(true);
    sb["stream_options"] = json!({"include_usage": true});
    let (_, bytes) = post(&s, "/v1/chat/completions", sb).await;
    assert!(!String::from_utf8_lossy(&bytes).contains("timings"));
}

#[tokio::test]
async fn chat_stream_matches_non_stream() {
    let Some(s) = state() else { return };
    let body = json!({"messages": [{"role": "system", "content": "Be nice."}, {"role": "user", "content": "Ünïcödé please"}]});
    let full = post_json(&s, "/v1/chat/completions", body.clone()).await;
    let mut sbody = body.clone();
    sbody["stream"] = json!(true);
    sbody["stream_options"] = json!({"include_usage": true});
    let (status, bytes) = post(&s, "/v1/chat/completions", sbody).await;
    assert_eq!(status, StatusCode::OK);
    let events = sse_events(&bytes);
    let (reasoning, content, calls, finish, usage) = assemble(&events);
    let m = &full["choices"][0]["message"];
    assert_eq!(reasoning, m["reasoning_content"].as_str().unwrap());
    assert_eq!(content, m["content"].as_str().unwrap());
    assert!(calls.is_empty());
    assert_eq!(finish.as_deref(), Some("stop"));
    assert_eq!(usage.unwrap(), full["usage"]);
    // with include_usage, non-final chunks carry "usage": null
    assert!(events[1].get("usage").is_some_and(Json::is_null));
    // and without it, no usage key at all
    let mut b2 = body.clone();
    b2["stream"] = json!(true);
    let (_, bytes) = post(&s, "/v1/chat/completions", b2).await;
    assert!(sse_events(&bytes).iter().all(|e| e.get("usage").is_none()));
}

#[tokio::test]
async fn tool_calls_non_stream_and_stream() {
    let Some(s) = state() else { return };
    let body = json!({"messages": [{"role": "user", "content": "What's the weather in Oslo?"}], "tools": weather_tools()});
    let v = post_json(&s, "/v1/chat/completions", body.clone()).await;
    let ch = &v["choices"][0];
    assert_eq!(ch["finish_reason"], "tool_calls");
    let m = &ch["message"];
    assert_eq!(m["content"], "I'll use get_weather for this.");
    let calls = m["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["type"], "function");
    assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: Json = serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"location": "Oslo", "days": 3}), "days must be typed as an integer");

    let mut sb = body.clone();
    sb["stream"] = json!(true);
    let (_, bytes) = post(&s, "/v1/chat/completions", sb).await;
    let (_, content, calls, finish, _) = assemble(&sse_events(&bytes));
    assert_eq!(content, "I'll use get_weather for this.");
    assert_eq!(finish.as_deref(), Some("tool_calls"));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].1, "get_weather");
    assert_eq!(serde_json::from_str::<Json>(&calls[0].2).unwrap(), json!({"location": "Oslo", "days": 3}));
}

#[tokio::test]
async fn parallel_tool_calls_and_tool_results() {
    let Some(s) = state() else { return };
    let v = post_json(&s, "/v1/chat/completions", json!({"tools": weather_tools(),
        "messages": [{"role": "user", "content": "Weather in Rome, and do a web_search for pizza"}]})).await;
    let calls = v["choices"][0]["message"]["tool_calls"].as_array().unwrap();
    let names: Vec<&str> = calls.iter().map(|c| c["function"]["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["get_weather", "web_search"]);
    let ids: Vec<&str> = calls.iter().map(|c| c["id"].as_str().unwrap()).collect();
    assert_ne!(ids[0], ids[1]);

    // send the results back as a client would (arguments as JSON strings)
    let mut messages = vec![json!({"role": "user", "content": "Weather in Rome, and do a web_search for pizza"})];
    messages.push(v["choices"][0]["message"].clone());
    messages.push(json!({"role": "tool", "tool_call_id": ids[0], "content": "22 C and sunny"}));
    messages.push(json!({"role": "tool", "tool_call_id": ids[1], "content": "Pizzeria da Michele"}));
    let v = post_json(&s, "/v1/chat/completions", json!({"tools": weather_tools(), "messages": messages})).await;
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["choices"][0]["message"]["content"], "Based on the tool result: 22 C and sunny");

    // parallel_tool_calls=false keeps the first call only
    let v = post_json(&s, "/v1/chat/completions", json!({"tools": weather_tools(), "parallel_tool_calls": false,
        "messages": [{"role": "user", "content": "Weather in Rome, and do a web_search for pizza"}]})).await;
    assert_eq!(v["choices"][0]["message"]["tool_calls"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn tool_choice_none_and_named() {
    let Some(s) = state() else { return };
    let v = post_json(&s, "/v1/chat/completions", json!({"tools": weather_tools(), "tool_choice": "none",
        "messages": [{"role": "user", "content": "What's the weather in Oslo?"}]})).await;
    let m = &v["choices"][0]["message"];
    assert!(m.get("tool_calls").is_none());
    assert!(m["content"].as_str().unwrap().starts_with("Hello from the ferrite mock engine!"));
    // a named function the user also mentions; only that tool is offered
    let v = post_json(&s, "/v1/chat/completions", json!({"tools": weather_tools(),
        "tool_choice": {"type": "function", "function": {"name": "web_search"}},
        "messages": [{"role": "user", "content": "web_search and weather please"}]})).await;
    let calls = v["choices"][0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "web_search");
    let (status, _) = post(&s, "/v1/chat/completions", json!({"tool_choice": "required",
        "messages": [{"role": "user", "content": "x"}]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn stop_strings_and_max_tokens() {
    let Some(s) = state() else { return };
    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Hi"}],
        "chat_template_kwargs": {"enable_thinking": false}, "stop": ["naïve", "never"]})).await;
    let ch = &v["choices"][0];
    assert_eq!(ch["finish_reason"], "stop");
    let c = ch["message"]["content"].as_str().unwrap();
    assert!(c.ends_with("streaming: café, "), "{c:?}");
    // stream: same truncation, stop string never emitted
    let (_, bytes) = post(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Hi"}],
        "chat_template_kwargs": {"enable_thinking": false}, "stop": "naïve", "stream": true})).await;
    let (_, content, _, finish, _) = assemble(&sse_events(&bytes));
    assert_eq!(content, c);
    assert_eq!(finish.as_deref(), Some("stop"));
    // include_stop_str_in_output keeps it
    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Hi"}],
        "chat_template_kwargs": {"enable_thinking": false}, "stop": "naïve", "include_stop_str_in_output": true})).await;
    assert!(v["choices"][0]["message"]["content"].as_str().unwrap().ends_with("café, naïve"));

    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5})).await;
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 5);
    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "write a long story"}],
        "max_completion_tokens": 300, "chat_template_kwargs": {"enable_thinking": false}})).await;
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 300);
    assert!(v["choices"][0]["message"]["content"].as_str().unwrap().contains("Line 3: the quick brown fox"));
}

#[tokio::test]
async fn completions_endpoint() {
    let Some(s) = state() else { return };
    let v = post_json(&s, "/v1/completions", json!({"model": "x", "prompt": "Once upon a time", "max_tokens": 200})).await;
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["model"], "qwen3.8-27b");
    let text = v["choices"][0]["text"].as_str().unwrap();
    assert!(text.starts_with(" and so the mock engine continues after \"Once upon a time\"."), "{text}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], s.tok.encode("Once upon a time").unwrap().len());
    // default max_tokens is 16 (OpenAI)
    let v = post_json(&s, "/v1/completions", json!({"prompt": "Once upon a time"})).await;
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 16);
    // streaming, token-id prompt, echo
    let ids = s.tok.encode("Hello world").unwrap();
    let (_, bytes) = post(&s, "/v1/completions", json!({"prompt": ids, "max_tokens": 50, "stream": true,
        "echo": true, "stream_options": {"include_usage": true}})).await;
    let events = sse_events(&bytes);
    let mut text = String::new();
    let mut finish = None;
    for e in &events {
        assert_eq!(e["object"], "text_completion");
        for ch in e["choices"].as_array().unwrap() {
            text.push_str(ch["text"].as_str().unwrap());
            if let Some(f) = ch["finish_reason"].as_str() {
                finish = Some(f.to_string());
            }
        }
    }
    assert!(text.starts_with("Hello world and so the mock engine continues"), "{text}");
    assert_eq!(finish.as_deref(), Some("stop"));
    assert_eq!(events.last().unwrap()["usage"]["prompt_tokens"], ids.len());
}

#[tokio::test]
async fn bad_requests_are_openai_errors() {
    let Some(s) = state() else { return };
    for body in [
        json!({"messages": []}),
        json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}),
        json!({"messages": [{"role": "user", "content": "x"}], "n": 2}),
        json!({"messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": {"reasoning_effort": "extreme"}}),
        json!({"messages": [{"role": "user", "content": 7}]}),
    ] {
        let (status, bytes) = post(&s, "/v1/chat/completions", body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let v: Json = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(!v["error"]["message"].as_str().unwrap().is_empty());
    }
    let req = Request::builder().method("POST").uri("/v1/chat/completions").body(Body::from("{oops")).unwrap();
    let resp = router(s.clone()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn burst_tokens_like_speculative_decoding() {
    let Some(dir) = model_dir() else { return };
    let tok = Arc::new(Tok::from_dir(&dir).unwrap());
    let engine = Arc::new(MockEngine::new(tok.clone(), MockConfig { delay: Duration::ZERO, burst: 5, ..Default::default() }));
    let cfg = ServerConfig { model_id: "qwen3.8-27b".into(), max_concurrent: 4, max_context: 131_072, defaults: GenDefaults::from_dir(&dir, &tok) };
    let s = AppState::new(engine, tok, Arc::new(ChatTemplate::from_dir(&dir).unwrap()), cfg);
    let reference = post_json(&state().unwrap(), "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Hi"}]})).await;
    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Hi"}]})).await;
    assert_eq!(v["choices"][0]["message"], reference["choices"][0]["message"]);
    assert_eq!(v["usage"], reference["usage"]);
    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 7})).await;
    assert_eq!(v["usage"]["completion_tokens"], 7);
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn six_concurrent_streams_with_four_slots() {
    let Some(s) = state_with(2, 4) else { return };
    let mut handles = Vec::new();
    for i in 0..6 {
        let s = s.clone();
        handles.push(tokio::spawn(async move {
            let (status, bytes) = post(&s, "/v1/chat/completions", json!({"stream": true, "max_tokens": 60,
                "messages": [{"role": "user", "content": format!("write stream {i}")}]})).await;
            assert_eq!(status, StatusCode::OK);
            let (_, _, _, finish, _) = assemble(&sse_events(&bytes));
            finish
        }));
    }
    let mut peak = 0;
    let t0 = Instant::now();
    while !handles.iter().all(|h| h.is_finished()) {
        peak = peak.max(s.running());
        assert!(t0.elapsed() < Duration::from_secs(30), "streams did not finish");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for h in handles {
        assert_eq!(h.await.unwrap().as_deref(), Some("length"));
    }
    assert!(peak <= 4, "more than 4 requests ran at once ({peak})");
    assert!(peak >= 2, "requests did not overlap ({peak})");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_stream_cancels_generation() {
    let Some(s) = state_with(5, 4) else { return };
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(json!({"stream": true, "max_tokens": 100000,
            "messages": [{"role": "user", "content": "write forever"}]}).to_string()))
        .unwrap();
    let resp = router(s.clone()).oneshot(req).await.unwrap();
    let mut body = resp.into_body();
    for _ in 0..5 {
        body.frame().await.unwrap().unwrap();
    }
    assert_eq!(s.running(), 1);
    drop(body);
    let t0 = Instant::now();
    while s.running() != 0 {
        assert!(t0.elapsed() < Duration::from_secs(5), "generation kept running after the client left");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// FE_DUMP_REQUESTS: the body as received in `<id>.json`, the outcome in `<id>.meta.json`; a rejected request is kept too.
#[tokio::test]
async fn request_capture_writes_body_and_meta() {
    let Some(s) = state() else { return };
    let dir = std::env::temp_dir().join(format!("fe-capture-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    s.dump_dir.set(dir.clone()).expect("capture dir not set by the environment");
    // odd spacing and key order: the file must hold the bytes as received, not a re-serialisation
    let raw = r#"{ "messages":[{"role":"user","content":"Say hi."}],   "max_tokens": 24, "model":"qwen3.8-27b","stream":false }"#;
    let req = Request::builder().method("POST").uri("/v1/chat/completions").header("content-type", "application/json").body(Body::from(raw)).unwrap();
    let resp = router(s.clone()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Json = serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let id = v["id"].as_str().unwrap();
    assert_eq!(std::fs::read_to_string(dir.join(format!("{id}.json"))).unwrap(), raw);
    let meta: Json = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{id}.meta.json"))).unwrap()).unwrap();
    assert_eq!(meta["id"], id);
    assert_eq!(meta["finish_reason"], v["choices"][0]["finish_reason"]);
    assert_eq!(meta["completion_tokens"], v["usage"]["completion_tokens"]);
    assert_eq!(meta["prompt_tokens"], v["usage"]["prompt_tokens"]);
    assert_eq!(meta["reasoning_cap_fired"], false);
    assert_eq!(meta["loop_breaker_fired"], false);
    assert_eq!(meta["stream"], false);
    // streaming: same files
    let (st, bytes) = post(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 16, "stream": true})).await;
    assert_eq!(st, StatusCode::OK);
    let ev = sse_events(&bytes);
    let id2 = ev[0]["id"].as_str().unwrap().to_string();
    let meta2: Json = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{id2}.meta.json"))).unwrap()).unwrap();
    assert_eq!(meta2["stream"], true);
    assert!(meta2["finish_reason"].is_string());
    assert!(dir.join(format!("{id2}.json")).exists());
    // a rejected request is kept with the reason
    let (st, _) = post(&s, "/v1/chat/completions", json!({"messages": "not a list"})).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);
    let rejected: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with("rejected-")).collect();
    assert_eq!(rejected.len(), 2, "{rejected:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Off by default: nothing is written anywhere.
#[tokio::test]
async fn request_capture_is_off_by_default() {
    let Some(s) = state() else { return };
    assert!(std::env::var_os("FE_DUMP_REQUESTS").is_some() || s.dump_dir.get().is_none());
}

// ---- sampling profiles, the effective-sampling header, the watchdog, the repeated-sentence rule (mock engine) ----

use fe_server::server::Profile;

fn state_cfg(delay_ms: u64, f: impl FnOnce(&mut GenDefaults)) -> Option<Arc<AppState>> {
    let dir = model_dir()?;
    let tok = Arc::new(Tok::from_dir(&dir).unwrap());
    let template = Arc::new(ChatTemplate::from_dir(&dir).unwrap());
    let mut defaults = GenDefaults::from_dir(&dir, &tok);
    f(&mut defaults);
    let engine = Arc::new(MockEngine::new(tok.clone(), MockConfig { delay: Duration::from_millis(delay_ms), max_batch: 4, ..Default::default() }));
    let cfg = ServerConfig { model_id: "qwen3.8-27b".into(), max_concurrent: 4, max_context: 131_072, defaults };
    Some(AppState::new(engine, tok, template, cfg))
}

fn card_profiles(d: &mut GenDefaults) {
    d.phase.profile_thinking = Profile { temperature: Some(1.0), top_p: Some(0.95), top_k: Some(20), min_p: Some(0.0), presence_penalty: Some(0.0),
        frequency_penalty: None, repetition_penalty: Some(1.0) };
    d.phase.profile_plain = Profile { temperature: Some(0.7), top_p: Some(0.8), top_k: Some(20), min_p: Some(0.0), presence_penalty: Some(1.5),
        frequency_penalty: None, repetition_penalty: Some(1.0) };
    d.phase.greedy_floor = false;
    d.phase.greedy_keeps_fast_path = true;
}

/// POST and return the parsed `x-sampling-effective` header.
async fn effective(s: &Arc<AppState>, body: Json) -> Json {
    let req = Request::builder().method("POST").uri("/v1/chat/completions").header("content-type", "application/json")
        .body(Body::from(body.to_string())).unwrap();
    let resp = router(s.clone()).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let h = resp.headers().get("x-sampling-effective").expect("x-sampling-effective header").to_str().unwrap().to_string();
    let _ = resp.into_body().collect().await;
    serde_json::from_str(&h).unwrap()
}

#[tokio::test]
async fn profile_of_the_request_mode_fills_what_the_client_omits() {
    let Some(s) = state_cfg(0, card_profiles) else { return };
    let msg = json!([{"role": "user", "content": "Hi"}]);
    for stream in [false, true] {
        // thinking: the thinking profile
        let e = effective(&s, json!({"messages": msg, "stream": stream})).await;
        assert_eq!(e["mode"], "thinking", "{e}");
        assert_eq!((e["temperature"].as_f64(), e["top_p"].as_f64(), e["top_k"].as_u64(), e["presence_penalty"].as_f64()), (Some(1.0), Some(0.95), Some(20), Some(0.0)), "{e}");
        assert_eq!(e["from"]["temperature"], "profile");
        // not thinking: the non-thinking profile
        let e = effective(&s, json!({"messages": msg, "stream": stream, "chat_template_kwargs": {"enable_thinking": false}})).await;
        assert_eq!(e["mode"], "non_thinking", "{e}");
        assert_eq!((e["temperature"].as_f64(), e["top_p"].as_f64(), e["top_k"].as_u64(), e["presence_penalty"].as_f64(), e["repetition_penalty"].as_f64()),
            (Some(0.7), Some(0.8), Some(20), Some(1.5), Some(1.0)), "{e}");
        assert_eq!(e["from"]["presence_penalty"], "profile");
    }
    // a field the client sends always wins, field by field (0 included); the others still come from the profile
    let e = effective(&s, json!({"messages": msg, "temperature": 0, "presence_penalty": 0.25, "chat_template_kwargs": {"enable_thinking": false}})).await;
    assert_eq!((e["temperature"].as_f64(), e["presence_penalty"].as_f64(), e["top_p"].as_f64()), (Some(0.0), Some(0.25), Some(0.8)), "{e}");
    assert_eq!((&e["from"]["temperature"], &e["from"]["presence_penalty"], &e["from"]["top_p"]), (&json!("client"), &json!("client"), &json!("profile")));
    let e = effective(&s, json!({"messages": msg, "presence_penalty": 0, "top_k": 0, "chat_template_kwargs": {"enable_thinking": false}})).await;
    assert_eq!((e["presence_penalty"].as_f64(), e["top_k"].as_u64(), e["temperature"].as_f64()), (Some(0.0), Some(0), Some(0.7)), "{e}");
    // a raw completion has no mode: no profile
    let req = Request::builder().method("POST").uri("/v1/completions").header("content-type", "application/json")
        .body(Body::from(json!({"prompt": "Once", "max_tokens": 4}).to_string())).unwrap();
    let resp = router(s.clone()).oneshot(req).await.unwrap();
    let e: Json = serde_json::from_str(resp.headers().get("x-sampling-effective").unwrap().to_str().unwrap()).unwrap();
    assert_eq!((&e["mode"], e["presence_penalty"].as_f64(), &e["from"]["temperature"]), (&json!("none"), Some(0.0), &json!("default")), "{e}");
}

/// A client that sends temperature 0 gets no penalty from the profile (the request stays on the device argmax); its own
/// penalties still apply, a client that omits temperature gets the whole profile, and the key switches the rule off.
#[tokio::test]
async fn greedy_client_keeps_the_fast_path() {
    let Some(s) = state_cfg(0, card_profiles) else { return };
    let msg = json!([{"role": "user", "content": "Hi"}]);
    let off = json!({"enable_thinking": false});
    let pens = |e: &Json| (e["presence_penalty"].as_f64(), e["frequency_penalty"].as_f64(), e["repetition_penalty"].as_f64());
    for stream in [false, true] {
        for t in [json!(0), json!(0.0), json!(0.000001)] {
            let e = effective(&s, json!({"messages": msg, "stream": stream, "temperature": t, "chat_template_kwargs": off})).await;
            assert_eq!(e["mode"], "non_thinking", "{e}");
            assert_eq!(e["temperature"].as_f64(), Some(0.0), "{e}");
            assert_eq!(pens(&e), (Some(0.0), Some(0.0), Some(1.0)), "{e}");
            assert_eq!((&e["from"]["temperature"], &e["from"]["presence_penalty"], &e["from"]["repetition_penalty"], &e["from"]["frequency_penalty"]),
                (&json!("client"), &json!("greedy_fast_path"), &json!("greedy_fast_path"), &json!("default")), "{e}");
            assert_eq!(e["greedy_keeps_fast_path"], true, "{e}");
            // top_p / top_k are still the profile's (unused at temperature 0)
            assert_eq!((e["top_p"].as_f64(), e["top_k"].as_u64()), (Some(0.8), Some(20)), "{e}");
        }
    }
    // the client's own penalties apply, one by one; the ones it omits stay neutral
    let e = effective(&s, json!({"messages": msg, "temperature": 0, "presence_penalty": 1.5, "chat_template_kwargs": off})).await;
    assert_eq!(pens(&e), (Some(1.5), Some(0.0), Some(1.0)), "{e}");
    assert_eq!(e["from"]["presence_penalty"], "client");
    let e = effective(&s, json!({"messages": msg, "temperature": 0, "repetition_penalty": 1.1, "frequency_penalty": 0.2, "chat_template_kwargs": off})).await;
    assert_eq!(pens(&e), (Some(0.0), Some(0.2), Some(1.1)), "{e}");
    // no temperature from the client, or one above 0: the whole profile
    let e = effective(&s, json!({"messages": msg, "chat_template_kwargs": off})).await;
    assert_eq!((e["temperature"].as_f64(), e["top_p"].as_f64(), e["top_k"].as_u64(), e["presence_penalty"].as_f64()), (Some(0.7), Some(0.8), Some(20), Some(1.5)), "{e}");
    assert_eq!(e["from"]["presence_penalty"], "profile");
    let e = effective(&s, json!({"messages": msg, "temperature": 0.2, "chat_template_kwargs": off})).await;
    assert_eq!((e["temperature"].as_f64(), e["presence_penalty"].as_f64(), &e["from"]["presence_penalty"]), (Some(0.2), Some(1.5), &json!("profile")), "{e}");
    let e = effective(&s, json!({"messages": msg, "top_k": 1, "chat_template_kwargs": off})).await;
    assert_eq!(e["presence_penalty"].as_f64(), Some(1.5), "top_k 1 without temperature is not the rule: {e}");
    // thinking mode: the same rule, with a profile that has a penalty there
    let Some(s2) = state_cfg(0, |d| { card_profiles(d); d.phase.profile_thinking.presence_penalty = Some(0.5); d.phase.profile_thinking.repetition_penalty = Some(1.05); }) else { return };
    let e = effective(&s2, json!({"messages": msg, "temperature": 0})).await;
    assert_eq!((e["mode"].as_str(), pens(&e)), (Some("thinking"), (Some(0.0), Some(0.0), Some(1.0))), "{e}");
    let e = effective(&s2, json!({"messages": msg})).await;
    assert_eq!(pens(&e), (Some(0.5), Some(0.0), Some(1.05)), "{e}");
    // switched off: the profile fills every omitted field again
    let Some(s3) = state_cfg(0, |d| { card_profiles(d); d.phase.greedy_keeps_fast_path = false; }) else { return };
    let e = effective(&s3, json!({"messages": msg, "temperature": 0, "chat_template_kwargs": off})).await;
    assert_eq!((e["presence_penalty"].as_f64(), &e["from"]["presence_penalty"], &e["greedy_keeps_fast_path"]), (Some(1.5), &json!("profile"), &json!(false)), "{e}");
}

#[tokio::test]
async fn greedy_floor_is_off_by_default_and_only_touches_the_reasoning_phase() {
    let msg = json!([{"role": "user", "content": "Hi"}]);
    let Some(s) = state_cfg(0, card_profiles) else { return };
    let e = effective(&s, json!({"messages": msg, "temperature": 0})).await;
    assert_eq!(e["greedy_floor"], false);
    assert!(e.get("reasoning_temperature").is_none(), "{e}");
    let Some(s) = state_cfg(0, |d| { card_profiles(d); d.phase.greedy_floor = true; }) else { return };
    // temperature 0 while thinking: the block is sampled at the profile temperature, the answer stays at 0
    let e = effective(&s, json!({"messages": msg, "temperature": 0})).await;
    assert_eq!((e["greedy_floor"].as_bool(), e["temperature"].as_f64(), e["reasoning_temperature"].as_f64()), (Some(true), Some(0.0), Some(1.0)), "{e}");
    // not thinking, or a client temperature above 0: untouched
    let e = effective(&s, json!({"messages": msg, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}})).await;
    assert!(e.get("reasoning_temperature").is_none(), "{e}");
    let e = effective(&s, json!({"messages": msg, "temperature": 0.3})).await;
    assert!(e.get("reasoning_temperature").is_none() && e["temperature"].as_f64() == Some(0.3), "{e}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watchdog_ends_a_request_that_goes_silent() {
    // the engine sends its first token and stalls: the no-token timeout (0.3 s here) ends the request cleanly
    let Some(s) = state_cfg(1, |d| d.phase.no_token_timeout = 0.3) else { return };
    let t = Instant::now();
    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "[[mock:stall]]"}], "chat_template_kwargs": {"enable_thinking": false}})).await;
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
    assert!(v["usage"]["completion_tokens"].as_u64().unwrap() >= 1);
    assert!(t.elapsed() >= Duration::from_millis(300) && t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
    // the wall-clock limit alone (no-token timeout off)
    let Some(s) = state_cfg(1, |d| { d.phase.no_token_timeout = 0.0; d.phase.reply_wall_limit = 0.4; }) else { return };
    let t = Instant::now();
    let (status, bytes) = post(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "[[mock:stall]]"}], "stream": true, "chat_template_kwargs": {"enable_thinking": false}})).await;
    assert_eq!(status, StatusCode::OK);
    let ev = sse_events(&bytes);
    assert!(ev.iter().any(|e| e["choices"][0]["finish_reason"] == "length"), "{ev:?}");
    assert!(t.elapsed() >= Duration::from_millis(400) && t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
    // off (the defaults of a state without a sampling.json watchdog entry keep 60 s): a normal reply is untouched
    let Some(s) = state_cfg(0, |d| d.phase.no_token_timeout = 0.3) else { return };
    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Hi"}]})).await;
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn wandering_reasoning_is_closed_and_a_wandering_answer_is_ended() {
    let Some(s) = state_cfg(0, |d| { d.phase.loop_breaker = true; d.phase.max_reasoning = 0; d.phase.loop_rules = Default::default(); }) else { return };
    let full = s.tok.encode(fe_server::mock::WANDER).unwrap().len();
    // thinking: the block is closed where the rule fires and the answer follows
    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "[[mock:wander]]"}], "max_tokens": 20000})).await;
    let m = &v["choices"][0]["message"];
    let reasoning = s.tok.encode(m["reasoning_content"].as_str().unwrap()).unwrap().len();
    assert!(reasoning >= 600 && reasoning < full / 2, "reasoning {reasoning} of {full} tokens");
    assert_eq!(m["content"], "The closing line: \"The river keeps its silver secrets under stone.\"", "{m}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    // not thinking: boosted once, then ended (finish length) well before the text is out
    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "[[mock:wander]]"}], "max_tokens": 20000,
        "chat_template_kwargs": {"enable_thinking": false}})).await;
    let out = v["usage"]["completion_tokens"].as_u64().unwrap() as usize;
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
    assert!(out >= 2400 && out < 2 * full, "{out} of {} tokens", 3 * full);
    // the rule off: the whole block and the whole answer come through
    let Some(s) = state_cfg(0, |d| { d.phase.loop_breaker = true; d.phase.max_reasoning = 0; d.phase.loop_rules = Default::default(); d.phase.loop_rules.sentence.enabled = false; }) else { return };
    let v = post_json(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "[[mock:wander]]"}], "max_tokens": 20000})).await;
    let reasoning = s.tok.encode(v["choices"][0]["message"]["reasoning_content"].as_str().unwrap()).unwrap().len();
    assert!(reasoning + 8 >= full, "{reasoning} of {full}");
}
