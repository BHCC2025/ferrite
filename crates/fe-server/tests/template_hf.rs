//! Chat-template rendering vs HF `apply_chat_template` references (tests/fixtures/hf_template_refs.json,
//! made by tools/gen_template_refs.py with transformers).
use std::path::PathBuf;

use fe_server::template::{ChatTemplate, normalize_messages};
use fe_server::tokenizer::Tok;
use serde_json::{Map, Value as Json, json};

fn model_dir() -> Option<PathBuf> {
    let d = std::env::var("FE_MODEL_DIR").map_or_else(|_| fe_site::nvfp4_dir(), PathBuf::from);
    d.join("tokenizer.json").exists().then_some(d)
}

fn refs() -> Vec<Json> {
    serde_json::from_str(include_str!("fixtures/hf_template_refs.json")).unwrap()
}

fn fixture_template() -> ChatTemplate {
    let mut special = Map::new();
    special.insert("bos_token".into(), Json::Null);
    special.insert("eos_token".into(), json!("<|im_end|>"));
    special.insert("pad_token".into(), json!("<|im_end|>"));
    special.insert("unk_token".into(), Json::Null);
    ChatTemplate::from_source(include_str!("fixtures/chat_template.jinja").to_string(), special).unwrap()
}

fn render(t: &ChatTemplate, case: &Json) -> String {
    let messages = normalize_messages(case["messages"].as_array().unwrap()).unwrap();
    let tools: Option<Vec<Json>> = case.get("tools").and_then(Json::as_array).cloned();
    let kwargs = case.get("kwargs").and_then(Json::as_object).cloned().unwrap_or_default();
    let agp = case["add_generation_prompt"].as_bool().unwrap();
    t.render(&messages, tools.as_deref(), agp, &kwargs)
        .unwrap_or_else(|e| panic!("{}: {e}", case["name"]))
}

fn assert_same(name: &str, got: &str, want: &str) {
    if got != want {
        let at = got.bytes().zip(want.bytes()).take_while(|(a, b)| a == b).count();
        let lo = at.saturating_sub(80);
        panic!(
            "{name}: render differs at byte {at}\n--- got ---\n{:?}\n--- want ---\n{:?}",
            &got[lo..(at + 80).min(got.len())],
            &want[lo..(at + 80).min(want.len())]
        );
    }
}

#[test]
fn renders_match_hf_byte_for_byte() {
    let t = fixture_template();
    let refs = refs();
    assert!(refs.len() >= 10);
    for case in &refs {
        let name = case["name"].as_str().unwrap();
        assert_same(name, &render(&t, case), case["text"].as_str().unwrap());
    }
}

#[test]
fn token_ids_match_hf() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: model dir not found (set FE_MODEL_DIR)");
        return;
    };
    let tok = Tok::from_dir(&dir).unwrap();
    let t = ChatTemplate::from_dir(&dir).unwrap();
    for case in &refs() {
        let name = case["name"].as_str().unwrap();
        let text = render(&t, case);
        assert_same(name, &text, case["text"].as_str().unwrap());
        let want: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(tok.encode(&text).unwrap(), want, "{name}: token ids differ");
    }
}

#[test]
fn template_errors_surface() {
    let t = fixture_template();
    let no_user = vec![json!({"role": "system", "content": "s"})];
    let e = t.render(&no_user, None, true, &Map::new()).unwrap_err();
    assert!(e.0.contains("No user query found"), "{e}");
    let mut kw = Map::new();
    kw.insert("reasoning_effort".into(), json!("extreme"));
    let e = t.render(&[json!({"role": "user", "content": "hi"})], None, true, &kw).unwrap_err();
    assert!(e.0.contains("Unexpected reasoning effort"), "{e}");
}
