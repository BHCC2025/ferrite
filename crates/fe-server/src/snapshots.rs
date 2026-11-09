//! Snapshot points: the token positions of a chat prompt where an engine with a prompt cache keeps a resumable state,
//! because a later request is likely to share the prompt up to there and no further:
//! - after each `<|im_start|>assistant\n`: where a turn's reply starts. Each earlier turn's prompt ended there (its
//!   generation prompt), and the history a client resends can differ right after it (the template drops earlier
//!   reasoning, and a reply cut by max_tokens comes back edited);
//! - after the system message's `<|im_end|>\n` when the prompt starts with one: the system prompt + tools that every
//!   conversation of an agent shares.
//! The markers are tokenized once from the template's literal text; a template without them yields no points.

use crate::tokenizer::Tok;
use anyhow::Result;

pub struct Marks {
    /// `<|im_start|>assistant\n`
    reply: Vec<u32>,
    /// `<|im_start|>system\n` and `<|im_end|>\n`
    system: Vec<u32>,
    end: Vec<u32>,
}

fn find(ids: &[u32], pat: &[u32], from: usize) -> Option<usize> {
    if pat.is_empty() || ids.len() < pat.len() {
        return None;
    }
    (from..=ids.len() - pat.len()).find(|&i| ids[i..i + pat.len()] == *pat)
}

impl Marks {
    pub fn new(tok: &Tok) -> Result<Marks> {
        Ok(Marks::from_ids(tok.encode("<|im_start|>assistant\n")?, tok.encode("<|im_start|>system\n")?,
            tok.encode("<|im_end|>\n")?))
    }

    pub fn from_ids(reply: Vec<u32>, system: Vec<u32>, end: Vec<u32>) -> Marks {
        Marks { reply, system, end }
    }

    /// Ascending positions (token counts from the prompt start, each <= ids.len()) of the snapshot points.
    pub fn points(&self, ids: &[u32]) -> Vec<usize> {
        let mut out = Vec::new();
        if !self.system.is_empty() && ids.starts_with(&self.system) {
            if let Some(i) = find(ids, &self.end, self.system.len()) {
                out.push(i + self.end.len());
            }
        }
        let mut from = 0;
        while let Some(i) = find(ids, &self.reply, from) {
            let p = i + self.reply.len();
            if out.last().map_or(true, |&q| q < p) {
                out.push(p);
            }
            from = i + 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ids: 1 = <|im_start|>, 2 = <|im_end|>, 3 = "\n", 4 = "system", 5 = "user", 6 = "assistant", 7.. = text
    fn marks() -> Marks {
        Marks::from_ids(vec![1, 6, 3], vec![1, 4, 3], vec![2, 3])
    }

    #[test]
    fn system_end_and_each_reply_start() {
        // system: 1 4 3 7 7 2 3 | user: 1 5 3 8 2 3 | assistant (history): 1 6 3 9 2 3 | user 1 5 3 8 2 3 | gen 1 6 3
        let ids = [1, 4, 3, 7, 7, 2, 3, 1, 5, 3, 8, 2, 3, 1, 6, 3, 9, 2, 3, 1, 5, 3, 8, 2, 3, 1, 6, 3];
        assert_eq!(marks().points(&ids), vec![7, 16, 28]);
    }

    #[test]
    fn no_system_message_and_a_thinking_off_tail() {
        // user, then the generation prompt and an empty think block (10 11 3 3) after it
        let ids = [1, 5, 3, 8, 2, 3, 1, 6, 3, 10, 11, 3, 3];
        assert_eq!(marks().points(&ids), vec![9]);
    }

    /// With the model's tokenizer (skipped when the model folder is not there): every point ends a marker.
    #[test]
    fn real_tokenizer_points_end_markers() {
        let dir = fe_site::exl3_dir();
        if !dir.join("tokenizer.json").exists() {
            return;
        }
        let tok = Tok::from_dir(&dir).unwrap();
        let m = Marks::new(&tok).unwrap();
        let prompt = "<|im_start|>system\nYou are a coding agent.\n\n# Tools\n...<|im_end|>\n<|im_start|>user\nFix the \
            bug.<|im_end|>\n<|im_start|>assistant\nDone: the loop bound.<|im_end|>\n<|im_start|>user\nNow add a test.\
            <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let ids = tok.encode(prompt).unwrap();
        let pts = m.points(&ids);
        assert_eq!(pts.len(), 3, "{pts:?}");
        let text = |n: usize| tok.decode(&ids[..n], false).unwrap();
        assert!(text(pts[0]).ends_with("...<|im_end|>\n"));
        for &p in &pts[1..] {
            assert!(text(p).ends_with("<|im_start|>assistant\n"), "{:?}", text(p));
        }
        assert!(pts[2] < ids.len());
    }

    #[test]
    fn no_markers() {
        assert!(marks().points(&[7, 8, 9]).is_empty());
        assert!(Marks::from_ids(vec![], vec![], vec![]).points(&[1, 6, 3]).is_empty());
    }
}
