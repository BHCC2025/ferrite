//! The server's minijinja renderer against HF `apply_chat_template` (Python jinja2) on the Inkling chat template, byte for byte
//! (cases from scripts/make_inkling_template_cases.py). Skipped when the model directory is not there (INK_DIR, else $FERRITE_MODELS/Inkling-2-Mini).
use fe_server::template::ChatTemplate;
use fe_server::tokenizer::Tok;
use serde_json::{Map, Value};
use std::path::PathBuf;

#[test]
fn inkling_template_matches_hf() {
    let dir = PathBuf::from(std::env::var("INK_DIR").unwrap_or_else(|_| format!("{}/Inkling-2-Mini", std::env::var("FERRITE_MODELS").unwrap_or("/var/tmp/models".into()))));
    if !dir.join("chat_template.jinja").exists() {
        eprintln!("skipped: no chat template in {}", dir.display());
        return;
    }
    let t = ChatTemplate::from_dir(&dir).unwrap();
    let tok = Tok::from_dir(&dir).unwrap();
    let cases: Vec<Value> = serde_json::from_str(include_str!("data/inkling_template_cases.json")).unwrap();
    let mut bad = Vec::new();
    for c in &cases {
        let messages = c["messages"].as_array().unwrap().clone();
        let tools: Option<Vec<Value>> = c["tools"].as_array().cloned();
        let kwargs: Map<String, Value> = c["kwargs"].as_object().cloned().unwrap_or_default();
        let got = t.render(&messages, tools.as_deref(), true, &kwargs);
        let want = c["expected"].as_str().unwrap();
        match got {
            Ok(g) if g == want => {
                // and the server's tokenizer turns the text into the same ids as HF
                let ids: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
                if tok.encode(&g).unwrap() != ids {
                    bad.push(format!("{}: token ids differ from HF", c["name"]));
                }
            }
            Ok(g) => {
                let i = g.bytes().zip(want.bytes()).take_while(|(a, b)| a == b).count();
                bad.push(format!("{}: differs at byte {i}: got {:?} want {:?}", c["name"], &g[i.saturating_sub(40)..(i + 60).min(g.len())], &want[i.saturating_sub(40)..(i + 60).min(want.len())]));
            }
            Err(e) => bad.push(format!("{}: render error {e}", c["name"])),
        }
    }
    eprintln!("inkling template: {} cases rendered byte-identical to HF", cases.len() - bad.len());
    assert!(bad.is_empty(), "{} of {} cases differ:\n{}", bad.len(), cases.len(), bad.join("\n"));
}
