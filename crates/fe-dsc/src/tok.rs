//! Tokenizer built from the GGUF (`tokenizer.ggml.tokens` / `merges`, byte-level BPE, DeepSeek-V3 style
//! pre-tokenizer), with the user-defined tokens (the think and DSML marks) as single ids. The serving
//! tokenizer: dsc-mkmodel writes it to tokenizer.json; the CPU reference and dsc-run use it directly.
use crate::config::DscConfig;
use anyhow::{anyhow, Result};
use fe_gguf::{Gguf, Value};
use tokenizers::{models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel, pre_tokenizers::sequence::Sequence,
    pre_tokenizers::split::Split, SplitDelimiterBehavior, Tokenizer, PreTokenizerWrapper};

pub fn build(g: &Gguf, _c: &DscConfig) -> Result<Tokenizer> {
    let strs = |k: &str| -> Result<Vec<String>> {
        Ok(g.get(k).and_then(Value::as_array).ok_or_else(|| anyhow!("{k} missing"))?
            .iter().map(|v| v.as_str().unwrap_or("").to_string()).collect())
    };
    let toks = strs("tokenizer.ggml.tokens")?;
    let merges: Vec<(String, String)> = strs("tokenizer.ggml.merges")?.into_iter()
        .filter_map(|m| m.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string()))).collect();
    let vocab: tokenizers::models::bpe::Vocab = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let bpe = BPE::builder().vocab_and_merges(vocab, merges).build().map_err(|e| anyhow!("bpe: {e}"))?;
    let mut t = Tokenizer::new(bpe);
    let split = |re: &str| -> Result<PreTokenizerWrapper> {
        Ok(Split::new(tokenizers::pre_tokenizers::split::SplitPattern::Regex(re.to_string()), SplitDelimiterBehavior::Isolated, false)
            .map_err(|e| anyhow!("split: {e}"))?.into())
    };
    let seq = Sequence::new(vec![
        split(r"\p{N}{1,3}")?,
        split(r"[一-龥぀-ゟ゠-ヿ]+")?,
        split(r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##)?,
        ByteLevel::new(false, false, false).into(),
    ]);
    t.with_pre_tokenizer(Some(seq));
    // control tokens (type 3: chat markers, BOS/EOS) are matched whole in the text, so a rendered chat template tokenises as served
    if let Some(tt) = g.get("tokenizer.ggml.token_type").and_then(Value::as_array) {
        let sp: Vec<tokenizers::AddedToken> = tt.iter().enumerate()
            .filter(|(_, v)| v.as_i64().is_some_and(|x| x == 3))
            .map(|(i, _)| tokenizers::AddedToken::from(toks[i].clone(), true)).collect();
        t.add_special_tokens(&sp);
        // user-defined tokens (type 4: <think>, </think>, the DSML mark, ...) are single ids in a rendered template too (llama.cpp matches
        // them whole); they are ordinary text on output, so they are added as non-special tokens
        let ud: Vec<tokenizers::AddedToken> = tt.iter().enumerate()
            .filter(|(_, v)| v.as_i64().is_some_and(|x| x == 4))
            .map(|(i, _)| tokenizers::AddedToken::from(toks[i].clone(), false).normalized(false)).collect();
        if std::env::var("DSC_TOK_NO_UD").is_err() {
            t.add_tokens(&ud);
        }
    }
    t.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    Ok(t)
}

/// Token ids of `text` (no BOS added).
pub fn encode(t: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(t.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
}
