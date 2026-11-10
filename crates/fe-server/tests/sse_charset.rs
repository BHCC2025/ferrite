//! The stream declares its charset (`text/event-stream; charset=utf-8`), so a client that decodes by the header (python
//! requests treats text/* without a charset as ISO-8859-1) reads the same text as the non-streamed response: curly quotes,
//! a 4-byte emoji, CJK, and characters the byte-level tokenizer splits over 2-3 tokens.
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use fe_server::mock::{MockConfig, MockEngine};
use fe_server::server::{AppState, GenDefaults, ServerConfig, router};
use fe_server::template::ChatTemplate;
use fe_server::tokenizer::Tok;
use http_body_util::BodyExt;
use serde_json::{Value as Json, json};
use tower::ServiceExt;

const TEXT: &str = "“curly” ‘quotes’ — the bucket’s 🦀 crab, 日本語と中文, rare 𠜎𝔘 and ꙮ";

fn state() -> Option<(Arc<AppState>, Arc<Tok>)> {
    let d = PathBuf::from(
        std::env::var("FE_MODEL_DIR").unwrap_or_else(|_| "/var/tmp/models/nvidia-Qwen3.8-27B-NVFP4".into()),
    );
    if !d.join("tokenizer.json").exists() {
        eprintln!("skipped: model dir not found (set FE_MODEL_DIR)");
        return None;
    }
    let tok = Arc::new(Tok::from_dir(&d).unwrap());
    let template = Arc::new(ChatTemplate::from_dir(&d).unwrap());
    let defaults = GenDefaults::from_dir(&d, &tok);
    let engine = Arc::new(MockEngine::new(tok.clone(), MockConfig { delay: Duration::from_millis(0), ..Default::default() }));
    let cfg = ServerConfig { model_id: "t".into(), max_concurrent: 4, max_context: 131_072, defaults };
    Some((AppState::new(engine, tok.clone(), template, cfg), tok))
}

async fn post(s: &Arc<AppState>, body: Json) -> (Option<String>, Vec<u8>) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = router(s.clone()).oneshot(req).await.unwrap();
    let ct = resp.headers().get("content-type").map(|v| v.to_str().unwrap().to_string());
    (ct, resp.into_body().collect().await.unwrap().to_bytes().to_vec())
}

/// Bytes -> text the way a header-honouring client does: the declared charset, else ISO-8859-1 for text/*.
fn client_decode(content_type: &str, bytes: &[u8]) -> String {
    let utf8 = content_type.split(';').skip(1).any(|p| p.trim().eq_ignore_ascii_case("charset=utf-8"));
    if utf8 { String::from_utf8(bytes.to_vec()).unwrap() } else { bytes.iter().map(|&b| b as char).collect() }
}

#[tokio::test]
async fn stream_declares_utf8_and_reads_back_as_non_stream() {
    let Some((s, tok)) = state() else { return };
    // the probe really has characters split over several tokens (byte fallback)
    let split: Vec<char> = TEXT.chars().filter(|c| c.len_utf8() >= 3 && tok.encode(&c.to_string()).unwrap().len() >= 2).collect();
    assert!(split.len() >= 2, "no multi-token character in the probe");

    let body = json!({"messages": [{"role": "user", "content": TEXT}], "chat_template_kwargs": {"enable_thinking": false}});
    let (_, full) = post(&s, body.clone()).await;
    let full: Json = serde_json::from_slice(&full).unwrap();
    let want = full["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    for probe in ["“curly”", "’", "🦀", "日本語", "𠜎𝔘"] {
        assert!(want.contains(probe), "the mock reply must echo {probe:?}: {want}");
    }

    let mut sb = body.clone();
    sb["stream"] = json!(true);
    let (ct, bytes) = post(&s, sb).await;
    let ct = ct.expect("content-type");
    assert_eq!(ct, "text/event-stream; charset=utf-8");
    let text = client_decode(&ct, &bytes);
    let mut got = String::new();
    for block in text.split("\n\n").filter(|b| !b.trim().is_empty()) {
        let d = block.strip_prefix("data: ").unwrap();
        if d == "[DONE]" {
            break;
        }
        let v: Json = serde_json::from_str(d).unwrap();
        if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
            got.push_str(c);
        }
    }
    assert_eq!(got, want, "streamed text read by a charset-honouring client != non-streamed");
    // the same bytes read without the charset are the mojibake this guards against
    assert_ne!(client_decode("text/event-stream", &bytes), text);
}
