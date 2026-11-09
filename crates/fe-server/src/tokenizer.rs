//! `tokenizer.json` loading and incremental (streaming) detokenization.
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};

/// Thin wrapper over a HF `tokenizers::Tokenizer`.
pub struct Tok {
    inner: tokenizers::Tokenizer,
    /// cached: `get_vocab_size(true)` rebuilds the vocabulary map on every call
    vocab_size: usize,
    /// For byte-level BPE (Qwen, GPT-2 style): every id's raw output bytes and whether it is a special
    /// token, so streaming can assemble UTF-8 exactly. `None` for other decoders.
    bytes: Option<ByteTable>,
}

struct ByteTable {
    bytes: Vec<Box<[u8]>>,
    special: Vec<bool>,
}

impl Tok {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_file(&dir.join("tokenizer.json"))
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
        let vocab_size = inner.get_vocab_size(true);
        let bytes = byte_table(&inner, vocab_size);
        Ok(Self { inner, vocab_size, bytes })
    }

    /// Encode text as the chat template rendered it: special-token strings in the text become their
    /// ids, and nothing else is added (no BOS), like HF `apply_chat_template(tokenize=True)`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode_fast(text, false).map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(ids, skip_special_tokens).map_err(|e| anyhow!("detokenize: {e}"))
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The raw bytes token `id` contributes to the output (byte-level tokenizers only); `None` when
    /// streaming falls back to window decoding. Skipped special tokens and unknown ids give `Some(&[])`.
    pub fn token_bytes(&self, id: u32, skip_special_tokens: bool) -> Option<&[u8]> {
        let t = self.bytes.as_ref()?;
        let i = id as usize;
        if i >= t.bytes.len() || (skip_special_tokens && t.special[i]) {
            return Some(&[]);
        }
        Some(&t.bytes[i])
    }

    /// True if streaming uses the exact byte table.
    pub fn byte_level(&self) -> bool {
        self.bytes.is_some()
    }
}

/// GPT-2's byte <-> printable-char mapping, inverted: the char a ByteLevel vocab uses → the byte.
fn byte_decoder() -> HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter().zip(cs).filter_map(|(b, c)| Some((char::from_u32(c)?, b as u8))).collect()
}

/// Per-id output bytes, the way the ByteLevel decoder computes them: a token whose chars all map back
/// to bytes gives those bytes, any other token (e.g. an added `<think>` with unmapped chars) its UTF-8.
fn byte_table(t: &tokenizers::Tokenizer, vocab_size: usize) -> Option<ByteTable> {
    if !matches!(t.get_decoder(), Some(tokenizers::DecoderWrapper::ByteLevel(_))) {
        return None;
    }
    let map = byte_decoder();
    let added = t.get_added_tokens_decoder();
    let n = vocab_size.max(added.keys().map(|&k| k as usize + 1).max().unwrap_or(0));
    let mut bytes = Vec::with_capacity(n);
    let mut special = vec![false; n];
    for id in 0..n as u32 {
        let b: Vec<u8> = match t.id_to_token(id) {
            Some(tokstr) => tokstr
                .chars()
                .map(|c| map.get(&c).copied())
                .collect::<Option<Vec<u8>>>()
                .unwrap_or_else(|| tokstr.as_bytes().to_vec()),
            None => Vec::new(),
        };
        bytes.push(b.into_boxed_slice());
    }
    for (id, tok) in &added {
        special[*id as usize] = tok.special;
    }
    Some(ByteTable { bytes, special })
}

/// Turns a stream of token ids into a stream of text without ever splitting a UTF-8 character.
///
/// Byte-level tokenizers (Qwen): token bytes are appended to a buffer and the longest valid UTF-8 prefix
/// is emitted; an incomplete character waits for its next bytes, bytes that can never become valid are
/// replaced by U+FFFD exactly as a full `decode` would (`from_utf8_lossy` semantics). So the
/// concatenated stream always equals `decode(all ids)`.
///
/// Other tokenizers: TGI / vLLM window method (decode `[prefix_offset..]`, emit the part beyond
/// `[prefix_offset..read_offset]` once it does not end in U+FFFD).
pub struct IncrementalDecoder {
    ids: Vec<u32>,
    pending: Vec<u8>,
    prefix_offset: usize,
    read_offset: usize,
    skip_special_tokens: bool,
}

/// Window method only: after this many tokens without a complete character, emit lossily.
const MAX_PENDING: usize = 32;

impl IncrementalDecoder {
    pub fn new(skip_special_tokens: bool) -> Self {
        Self { ids: Vec::new(), pending: Vec::new(), prefix_offset: 0, read_offset: 0, skip_special_tokens }
    }

    /// All ids pushed so far.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Add one token; returns the newly completed text (often empty while a character is incomplete).
    pub fn push(&mut self, tok: &Tok, id: u32) -> Result<String> {
        self.ids.push(id);
        if let Some(b) = tok.token_bytes(id, self.skip_special_tokens) {
            self.pending.extend_from_slice(b);
            return Ok(take_valid_utf8(&mut self.pending));
        }
        self.step(tok, false)
    }

    /// Emit whatever is still held back (end of generation), lossily if it is incomplete.
    pub fn flush(&mut self, tok: &Tok) -> Result<String> {
        if tok.byte_level() {
            let out = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            return Ok(out);
        }
        self.step(tok, true)
    }

    fn step(&mut self, tok: &Tok, force: bool) -> Result<String> {
        if self.read_offset >= self.ids.len() {
            return Ok(String::new());
        }
        let prefix = tok
            .decode(&self.ids[self.prefix_offset..self.read_offset], self.skip_special_tokens)
            .context("prefix")?;
        let full = tok.decode(&self.ids[self.prefix_offset..], self.skip_special_tokens)?;
        let pending = self.ids.len() - self.read_offset;
        let complete = !full.ends_with('\u{FFFD}');
        if !(complete || force || pending >= MAX_PENDING) {
            return Ok(String::new());
        }
        let delta = if full.starts_with(&prefix) {
            full[prefix.len()..].to_string()
        } else {
            let common = prefix
                .char_indices()
                .zip(full.chars())
                .take_while(|((_, a), b)| a == b)
                .last()
                .map(|((i, c), _)| i + c.len_utf8())
                .unwrap_or(0);
            full[common..].to_string()
        };
        self.prefix_offset = self.read_offset;
        self.read_offset = self.ids.len();
        Ok(delta)
    }
}

/// Remove and return the longest prefix of `buf` that is final: valid UTF-8, with each invalid
/// sequence replaced by U+FFFD; an incomplete character at the end stays in `buf`.
fn take_valid_utf8(buf: &mut Vec<u8>) -> String {
    let mut out = String::new();
    let mut start = 0;
    loop {
        match std::str::from_utf8(&buf[start..]) {
            Ok(s) => {
                out.push_str(s);
                buf.clear();
                return out;
            }
            Err(e) => {
                let good = start + e.valid_up_to();
                out.push_str(std::str::from_utf8(&buf[start..good]).unwrap_or_default());
                match e.error_len() {
                    Some(n) => {
                        out.push('\u{FFFD}');
                        start = good + n;
                    }
                    None => {
                        buf.drain(..good);
                        return out;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::take_valid_utf8;

    #[test]
    fn utf8_assembly_matches_lossy_decoding() {
        let cases: &[&[&[u8]]] = &[
            &["é".as_bytes()],
            &[&[0xE6], &[0x97, 0xA5], &[0xE6, 0x9C], &[0xAC]], // 日本 split across 4 tokens
            &[&[0xF0, 0x9F], &[0xA6], &[0x80, b'x']],         // 🦀 then x
            &[&[0xFF, b'a'], &[0xC3]],                          // invalid byte, then a dangling lead byte
            &[&[0xE6, 0x97], &[b'z']],                          // truncated char followed by ASCII
        ];
        for chunks in cases {
            let all: Vec<u8> = chunks.concat();
            let mut buf = Vec::new();
            let mut out = String::new();
            for c in *chunks {
                buf.extend_from_slice(c);
                let piece = take_valid_utf8(&mut buf);
                out.push_str(&piece);
            }
            out.push_str(&String::from_utf8_lossy(&buf));
            assert_eq!(out, String::from_utf8_lossy(&all), "{chunks:?}");
        }
    }
}
