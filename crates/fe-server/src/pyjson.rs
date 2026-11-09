//! JSON text exactly as Python's `json.dumps(x, ensure_ascii=False)` writes it.
//!
//! Chat templates call `tojson` (HF: `json.dumps`) on tool schemas and tool-call arguments, and the
//! model was trained on that byte format: `", "` / `": "` separators, insertion key order, no
//! HTML escaping, non-ASCII kept, Python float repr (`1.0`, `1e-05`, `1e+16`).
use serde_json::Value;

/// `json.dumps(v, ensure_ascii=False)` or, with `indent`, `json.dumps(v, ensure_ascii=False, indent=n)`.
pub fn dumps(v: &Value, indent: Option<usize>) -> String {
    let mut out = String::new();
    write_value(&mut out, v, indent, 0);
    out
}

fn write_value(out: &mut String, v: &Value, indent: Option<usize>, level: usize) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&py_float_repr(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_str(out, s),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                sep(out, i, indent, level + 1);
                write_value(out, item, indent, level + 1);
            }
            close(out, indent, level);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, item)) in map.iter().enumerate() {
                sep(out, i, indent, level + 1);
                write_str(out, k);
                out.push_str(": ");
                write_value(out, item, indent, level + 1);
            }
            close(out, indent, level);
            out.push('}');
        }
    }
}

fn sep(out: &mut String, i: usize, indent: Option<usize>, level: usize) {
    match indent {
        None => {
            if i > 0 {
                out.push_str(", ");
            }
        }
        Some(n) => {
            if i > 0 {
                out.push(',');
            }
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', n * level));
        }
    }
}

fn close(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', n * level));
    }
}

/// Python's string escaping with `ensure_ascii=False`.
pub fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `repr(float)` (as `json.dumps` uses it): shortest round-trip digits, scientific notation
/// when the decimal exponent is < -4 or >= 16, always a `.0` for integral values.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let s = format!("{:e}", f); // shortest digits, e.g. "-1.2345e-7", "3e0"
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let body = if !(-4..16).contains(&exp) {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{}e{}{:02}", m, if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        let point = exp + 1; // digits before the decimal point
        if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        }
    };
    if neg { format!("-{body}") } else { body }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_like_python() {
        let cases = [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (1234567890123456.0, "1234567890123456.0"),
            (-2.5, "-2.5"),
            (3.14159, "3.14159"),
            (100.0, "100.0"),
            (1.2345678901234568e17, "1.2345678901234568e+17"),
            (-0.0, "-0.0"),
        ];
        for (f, want) in cases {
            assert_eq!(py_float_repr(f), want, "{f}");
        }
    }

    #[test]
    fn dumps_like_python() {
        let v = json!({"b": [1, 2.0, "x\"y\n<z>&é🦀"], "a": {"n": null, "t": true, "e": {}, "l": []}});
        assert_eq!(
            dumps(&v, None),
            r#"{"b": [1, 2.0, "x\"y\n<z>&é🦀"], "a": {"n": null, "t": true, "e": {}, "l": []}}"#
        );
        assert_eq!(dumps(&json!({"a": [1, {"b": 2}]}), Some(2)), "{\n  \"a\": [\n    1,\n    {\n      \"b\": 2\n    }\n  ]\n}");
        assert_eq!(dumps(&json!("\u{01}\t"), None), "\"\\u0001\\t\"");
    }
}
