//! The canonical two-rank split of Qwen3.5 (qwen3_5) EXL3 linears, by attention / GDN head and MLP column, so
//! each rank owns whole heads (its KV cache, GDN state and conv channels) and needs one reduction after each
//! of o_proj / linear_attn.out_proj / mlp.down_proj, plus a logits gather after lm_head.

/// What rank `r` of `world` keeps of a linear.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Split {
    /// output columns, concatenated in order (each range 128-aligned)
    Cols(Vec<(usize, usize)>),
    /// input rows k0..k1 (the rank's partial is summed over the ranks)
    Rows(usize, usize),
    /// the whole tensor on every rank
    Replicate,
}

/// Qwen3.5-27B shapes: 24 q heads / 4 kv heads x 256 (q_proj packs q and the output gate a head: 512 cols a
/// head), GDN 16 key heads / 48 value heads x 128 (in_proj_qkv = [q 2048 | k 2048 | v 6144]), MLP 17408.
pub fn qwen35_split(base: &str, k: usize, n: usize, rank: usize, world: usize) -> Split {
    let part = |len: usize, off: usize| {
        let w = len / world;
        (off + rank * w, off + (rank + 1) * w)
    };
    if world == 1 {
        return Split::Replicate;
    }
    let leaf = base.rsplit('.').next().unwrap_or("");
    match leaf {
        "q_proj" | "k_proj" | "v_proj" | "gate_proj" | "up_proj" | "in_proj_z" => Split::Cols(vec![part(n, 0)]),
        "in_proj_qkv" => {
            // q and k: 16 heads x 128 each, v: the rest (48 heads x 128)
            let qk = (n - (n * 3) / 5) / 2;
            Split::Cols(vec![part(qk, 0), part(qk, qk), part(n - 2 * qk, 2 * qk)])
        }
        "o_proj" | "out_proj" | "down_proj" => {
            let (a, b) = part(k, 0);
            Split::Rows(a, b)
        }
        "lm_head" => Split::Cols(vec![part(n, 0)]),
        _ => Split::Replicate, // mtp.fc and anything unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_tp2() {
        assert_eq!(qwen35_split("model.language_model.layers.3.self_attn.q_proj", 5120, 12288, 1, 2), Split::Cols(vec![(6144, 12288)]));
        assert_eq!(qwen35_split("x.self_attn.k_proj", 5120, 1024, 0, 2), Split::Cols(vec![(0, 512)]));
        assert_eq!(qwen35_split("x.self_attn.o_proj", 6144, 5120, 1, 2), Split::Rows(3072, 6144));
        assert_eq!(
            qwen35_split("x.linear_attn.in_proj_qkv", 5120, 10240, 1, 2),
            Split::Cols(vec![(1024, 2048), (3072, 4096), (7168, 10240)])
        );
        assert_eq!(qwen35_split("x.mlp.down_proj", 17408, 5120, 0, 2), Split::Rows(0, 8704));
        assert_eq!(qwen35_split("lm_head", 5120, 248320, 1, 2), Split::Cols(vec![(124160, 248320)]));
        assert_eq!(qwen35_split("mtp.fc", 10240, 5120, 1, 2), Split::Replicate);
    }
}
