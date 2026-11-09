//! Incremental detokenization never splits a UTF-8 character and reproduces the full decode.
use std::path::PathBuf;

use fe_server::tokenizer::{IncrementalDecoder, Tok};

fn tok() -> Option<Tok> {
    let d = std::env::var("FE_MODEL_DIR").map_or_else(|_| fe_site::nvfp4_dir(), PathBuf::from);
    if !d.join("tokenizer.json").exists() {
        eprintln!("skipped: model dir not found (set FE_MODEL_DIR)");
        return None;
    }
    Some(Tok::from_dir(&d).unwrap())
}

#[test]
fn streaming_decode_is_exact_and_never_splits_chars() {
    let Some(tok) = tok() else { return };
    let texts = [
        "plain ascii text, with punctuation!",
        "Résumé naïve café Zürich — “quotes” … ✓",
        "日本語のテキストと中文混合，한국어도 있습니다。",
        "emoji 🦀🚀✨👩‍👩‍👧‍👦 flags 🇳🇴🇯🇵 and 𝔘𝔫𝔦𝔠𝔬𝔡𝔢",
        "<think>\nreasoning\n</think>\n\n<tool_call>\n<function=f>\n</function>\n</tool_call>",
        "code:\n\tfn main() { println!(\"{}\", \"π≈3.14\"); }\n",
    ];
    let mut held_back = 0;
    for text in texts {
        let ids = tok.encode(text).unwrap();
        let mut dec = IncrementalDecoder::new(true);
        let mut out = String::new();
        for &id in &ids {
            let piece = dec.push(&tok, id).unwrap();
            assert!(!piece.contains('\u{FFFD}'), "replacement char streamed for {text:?}: {piece:?}");
            if piece.is_empty() {
                held_back += 1;
            }
            out.push_str(&piece);
        }
        out.push_str(&dec.flush(&tok).unwrap());
        assert_eq!(out, text, "stream != text");
        assert_eq!(out, tok.decode(&ids, true).unwrap());
    }
    // the emoji / CJK texts must have forced at least one token to wait for the rest of a character
    assert!(held_back > 0, "no multi-byte character was ever split across tokens");
}

#[test]
fn special_tokens_are_skipped() {
    let Some(tok) = tok() else { return };
    let ids = tok.encode("Hello<|im_end|> world").unwrap();
    let mut dec = IncrementalDecoder::new(true);
    let mut out = String::new();
    for &id in &ids {
        out.push_str(&dec.push(&tok, id).unwrap());
    }
    out.push_str(&dec.flush(&tok).unwrap());
    assert_eq!(out, "Hello world");
}

#[test]
fn byte_table_matches_the_decoder_for_every_token() {
    let Some(tok) = tok() else { return };
    assert!(tok.byte_level(), "Qwen's tokenizer should use the byte-level streaming path");
    for id in 0..tok.vocab_size() as u32 {
        let b = tok.token_bytes(id, false).unwrap();
        assert_eq!(String::from_utf8_lossy(b), tok.decode(&[id], false).unwrap(), "token {id}");
        let skipped = tok.token_bytes(id, true).unwrap();
        assert_eq!(String::from_utf8_lossy(skipped), tok.decode(&[id], true).unwrap(), "token {id} (skip special)");
    }
}

#[test]
fn random_token_streams_equal_full_decode() {
    let Some(tok) = tok() else { return };
    let mut rng = fe_server::sampler::Rng::new(11);
    let v = tok.vocab_size() as u64;
    // byte-fragment heavy: half the ids from the low (single-byte / fragment) range
    for _ in 0..400 {
        let ids: Vec<u32> = (0..40)
            .map(|_| if rng.next_u64() % 2 == 0 { (rng.next_u64() % 512) as u32 } else { (rng.next_u64() % v) as u32 })
            .collect();
        let mut dec = IncrementalDecoder::new(true);
        let mut out = String::new();
        for &id in &ids {
            out.push_str(&dec.push(&tok, id).unwrap());
        }
        out.push_str(&dec.flush(&tok).unwrap());
        assert_eq!(out, tok.decode(&ids, true).unwrap(), "{ids:?}");
    }
}
