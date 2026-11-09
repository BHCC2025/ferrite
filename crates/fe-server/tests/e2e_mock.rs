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
    let cfg = ServerConfig { model_id: "spark".into(), max_concurrent, max_context: 131_072, defaults };
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
        assert_eq!(ev["model"], "spark");
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
    assert_eq!(v["data"][0]["id"], "spark");
    assert_eq!(v["data"][0]["max_model_len"], 131_072);
}

#[tokio::test]
async fn chat_non_stream_with_reasoning() {
    let Some(s) = state() else { return };
    let v = post_json(&s, "/v1/chat/completions", json!({"model": "whatever", "messages": [{"role": "user", "content": "Hi there"}]})).await;
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "spark");
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
    assert_eq!(v["model"], "spark");
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
    let cfg = ServerConfig { model_id: "spark".into(), max_concurrent: 4, max_context: 131_072, defaults: GenDefaults::from_dir(&dir, &tok) };
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
