//! Stop strings: truncate the output at the first stop string, holding back text that could still
//! become one.

pub struct StopMatcher {
    stops: Vec<String>,
    include_stop: bool,
    pending: String,
    stopped: bool,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>, include_stop: bool) -> Self {
        let stops = stops.into_iter().filter(|s| !s.is_empty()).collect();
        Self { stops, include_stop, pending: String::new(), stopped: false }
    }

    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Feed new text; returns the text that is safe to emit and whether a stop string was hit
    /// (after which everything else is discarded).
    pub fn push(&mut self, text: &str) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }
        if self.stops.is_empty() {
            return (text.to_string(), false);
        }
        self.pending.push_str(text);
        // earliest match wins; on a tie, the longest stop string
        let mut best: Option<(usize, usize)> = None;
        for s in &self.stops {
            if let Some(pos) = self.pending.find(s.as_str()) {
                if best.is_none_or(|(p, l)| pos < p || (pos == p && s.len() > l)) {
                    best = Some((pos, s.len()));
                }
            }
        }
        if let Some((pos, len)) = best {
            let end = if self.include_stop { pos + len } else { pos };
            let out = self.pending[..end].to_string();
            self.pending.clear();
            self.stopped = true;
            return (out, true);
        }
        let hold = self.holdback();
        let cut = self.pending.len() - hold;
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        (out, false)
    }

    /// End of generation: release whatever was held back.
    pub fn flush(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }

    /// Length of the longest suffix of `pending` that is a proper prefix of some stop string.
    fn holdback(&self) -> usize {
        let p = &self.pending;
        let mut best = 0;
        for s in &self.stops {
            let max = (s.len() - 1).min(p.len());
            for k in (best + 1..=max).rev() {
                let start = p.len() - k;
                if p.is_char_boundary(start) && s.is_char_boundary(k) && s.as_bytes()[..k] == p.as_bytes()[start..] {
                    best = k;
                    break;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], include: bool, chunks: &[&str]) -> (String, bool) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()).collect(), include);
        let mut out = String::new();
        for c in chunks {
            let (t, hit) = m.push(c);
            out.push_str(&t);
            if hit {
                return (out, true);
            }
        }
        out.push_str(&m.flush());
        (out, false)
    }

    #[test]
    fn no_stops_passthrough() {
        assert_eq!(run(&[], false, &["a", "b"]), ("ab".into(), false));
    }

    #[test]
    fn stop_within_chunk() {
        assert_eq!(run(&["END"], false, &["hello END world"]), ("hello ".into(), true));
        assert_eq!(run(&["END"], true, &["hello END world"]), ("hello END".into(), true));
    }

    #[test]
    fn stop_split_across_chunks() {
        assert_eq!(run(&["STOP"], false, &["abc S", "T", "OP def"]), ("abc ".into(), true));
    }

    #[test]
    fn partial_prefix_released_when_not_a_stop() {
        assert_eq!(run(&["STOP"], false, &["ab ST", "x", "y"]), ("ab STxy".into(), false));
        assert_eq!(run(&["STOP"], false, &["ab ST"]), ("ab ST".into(), false));
    }

    #[test]
    fn earliest_of_several() {
        assert_eq!(run(&["b", "a"], false, &["xxab"]), ("xx".into(), true));
        assert_eq!(run(&["\n\n", "User:"], false, &["Hi\n", "\nUser:"]), ("Hi".into(), true));
    }

    #[test]
    fn multibyte_stop() {
        assert_eq!(run(&["日本"], false, &["café 日", "本語"]), ("café ".into(), true));
        assert_eq!(run(&["🦀x"], false, &["a🦀", "y"]), ("a🦀y".into(), false));
    }
}
