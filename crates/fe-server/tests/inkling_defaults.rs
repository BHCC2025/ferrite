//! Request defaults of the Inkling server, through the router with an engine that records what it is asked to run: a request that sends no
//! temperature / reasoning effort / max_tokens gets the publisher's settings (generation_config.json: sampling on, temperature 1.0, no top-p / top-k;
//! the template's reasoning effort 0.9) and the launcher's default cap; what the request sends wins. Skipped without the model dir
//! (INK_DIR, else $FERRITE_MODELS/Inkling-2-Mini).
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::Request;
use fe_server::engine::{Engine, EngineInfo, Finish, FinishReason, GenRequest, SamplingParams, TokenSink};
use fe_server::server::{AppState, GenDefaults, ServerConfig, router};
use fe_server::template::ChatTemplate;
use fe_server::tokenizer::Tok;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

struct Recorder(Mutex<Vec<(Vec<u32>, SamplingParams)>>);

impl Engine for Recorder {
    fn info(&self) -> EngineInfo {
        EngineInfo { name: "recorder".into(), max_batch: 1, max_context: 131_072 }
    }
    fn submit(&self, req: GenRequest, sink: TokenSink) {
        self.0.lock().unwrap().push((req.prompt_ids.clone(), req.params.clone()));
        sink.finish(Finish::new(FinishReason::Length));
    }
}

fn setup(tweak: impl FnOnce(&mut GenDefaults)) -> Option<(Arc<AppState>, Arc<Recorder>, Arc<Tok>)> {
    let dir = PathBuf::from(std::env::var("INK_DIR").unwrap_or_else(|_| format!("{}/Inkling-2-Mini", std::env::var("FERRITE_MODELS").unwrap_or("/var/tmp/models".into()))));
    if !dir.join("chat_template.jinja").exists() {
        eprintln!("skipped: no model dir");
        return None;
    }
    let tok = Arc::new(Tok::from_dir(&dir).unwrap());
    let template = Arc::new(ChatTemplate::from_dir(&dir).unwrap());
    let mut defaults = GenDefaults::from_dir(&dir, &tok);
    defaults.apply_template(&template);
    tweak(&mut defaults);
    let rec = Arc::new(Recorder(Mutex::new(Vec::new())));
    let cfg = ServerConfig { model_id: "inkling-2-mini".into(), max_concurrent: 1, max_context: 131_072, defaults };
    Some((AppState::new(rec.clone(), tok.clone(), template, cfg), rec, tok))
}

async fn ask(s: &Arc<AppState>, rec: &Recorder, tok: &Tok, extra: Value) -> (String, SamplingParams) {
    let mut body = json!({"model": "inkling-2-mini", "messages": [{"role": "user", "content": "hi"}]});
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    let req = Request::builder().method("POST").uri("/v1/chat/completions").header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
    let resp = router(s.clone()).oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    assert!(status.is_success(), "{}", String::from_utf8_lossy(&bytes));
    let (ids, p) = rec.0.lock().unwrap().pop().unwrap();
    (tok.decode(&ids, false).unwrap(), p)
}

#[tokio::test]
async fn publisher_defaults_and_explicit_values() {
    let Some((s, rec, tok)) = setup(|_| {}) else { return };
    // nothing sent: sampling on at temperature 1.0, no top-p / top-k, effort 0.9, the 32768 cap
    let (prompt, p) = ask(&s, &rec, &tok, json!({})).await;
    assert!(prompt.contains("Thinking effort level: 0.9<|end_message|>"), "{prompt}");
    assert_eq!((p.temperature, p.top_p, p.top_k, p.min_p), (1.0, 1.0, 0, 0.0));
    assert_eq!(p.max_tokens, 32768);
    // explicit values win
    let (prompt, p) = ask(&s, &rec, &tok, json!({"temperature": 0.3, "top_p": 0.9, "max_tokens": 77, "reasoning_effort": "low"})).await;
    assert!(prompt.contains("Thinking effort level: 0.2<"), "{prompt}");
    assert_eq!((p.temperature, p.top_p, p.max_tokens), (0.3, 0.9, 77));
    let (prompt, _) = ask(&s, &rec, &tok, json!({"reasoning_effort": "0.5"})).await;
    assert!(prompt.contains("Thinking effort level: 0.5<"), "{prompt}");
    let (prompt, _) = ask(&s, &rec, &tok, json!({"chat_template_kwargs": {"reasoning_effort": 0.99}})).await;
    assert!(prompt.contains("Thinking effort level: 0.99<"), "{prompt}");
    for (name, extra) in [("enable_thinking false", json!({"chat_template_kwargs": {"enable_thinking": false}})), ("top-level enable_thinking", json!({"enable_thinking": false})), ("effort none", json!({"reasoning_effort": "none"}))] {
        let (prompt, _) = ask(&s, &rec, &tok, extra).await;
        assert!(prompt.contains("Thinking effort level: 0<|end_message|>"), "{name}: {prompt}");
    }
    // temperature 0 stays greedy, max_completion_tokens is max_tokens
    let (_, p) = ask(&s, &rec, &tok, json!({"temperature": 0, "max_completion_tokens": 5})).await;
    assert_eq!((p.temperature, p.max_tokens), (0.0, 5));
}

#[tokio::test]
async fn launcher_overrides() {
    let Some((s, rec, tok)) = setup(|d| {
        d.temperature = 0.6;
        d.default_max_tokens = Some(4096);
        d.default_reasoning_effort = Some(json!("medium"));
    }) else { return };
    let (prompt, p) = ask(&s, &rec, &tok, json!({})).await;
    assert!(prompt.contains("Thinking effort level: 0.7<"), "{prompt}");
    assert_eq!((p.temperature, p.max_tokens), (0.6, 4096));
    // the request still wins, and enable_thinking false beats the launcher's effort
    let (prompt, p) = ask(&s, &rec, &tok, json!({"temperature": 1.0, "max_tokens": 9, "reasoning_effort": "high"})).await;
    assert!(prompt.contains("Thinking effort level: 0.9<"), "{prompt}");
    assert_eq!((p.temperature, p.max_tokens), (1.0, 9));
    let (prompt, _) = ask(&s, &rec, &tok, json!({"enable_thinking": false})).await;
    assert!(prompt.contains("Thinking effort level: 0<"), "{prompt}");
}
