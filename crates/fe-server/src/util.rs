//! Small helpers: random ids, timestamps.
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 64 random bits (std's per-process random hash keys + a counter + the clock; not cryptographic).
pub fn random_u64() -> u64 {
    let mut h = RandomState::new().build_hasher();
    h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    h.write_u128(SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.finish()
}

/// `prefix` + `len` random alphanumeric characters, e.g. `call_3fJk...`.
pub fn random_id(prefix: &str, len: usize) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut s = String::with_capacity(prefix.len() + len);
    s.push_str(prefix);
    let mut bits = random_u64();
    let mut left = 10; // 6 bits per char from 64 bits
    for _ in 0..len {
        if left == 0 {
            bits = random_u64();
            left = 10;
        }
        s.push(ALPHABET[(bits % 62) as usize] as char);
        bits /= 62;
        left -= 1;
    }
    s
}

pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn ids_are_unique_and_shaped() {
        let a = super::random_id("call_", 24);
        let b = super::random_id("call_", 24);
        assert_ne!(a, b);
        assert_eq!(a.len(), 29);
        assert!(a[5..].chars().all(|c| c.is_ascii_alphanumeric()));
    }
}
