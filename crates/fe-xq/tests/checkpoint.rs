//! fe-xq on the real gptq-mix5.0 checkpoint (FE_XQ_MODEL: its folder): config / shard consistency, and
//! the packed bytes of real tensors against xq.zig pack's output (oracle `xq-mix5.0/xqpack/`, written by
//! `xpu-xq_oracle-test`; FE_XQ_ZIGPACK overrides the directory). Skips when the files are absent.
use fe_xq::{pack, Checkpoint, Rotate};
use std::path::PathBuf;

fn model() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("FE_XQ_MODEL").unwrap_or_default());
    if d.join("xq_config.json").exists() { Some(d) } else {
        eprintln!("SKIP: no XQ checkpoint at {d:?}");
        None
    }
}

#[test]
fn config_and_shards_agree() {
    let Some(d) = model() else { return };
    let ck = Checkpoint::open(&d).unwrap();
    ck.validate(false).unwrap();
    let h = ck.cfg.bits_histogram();
    eprintln!("{} XQ tensors, bits {h:?}, packed {:.2} GB", ck.cfg.tensors.len(), ck.packed_bytes() as f64 / 1e9);
    let lm = ck.cfg.get("lm_head").unwrap();
    assert_eq!((lm.n, lm.k, lm.rotate), (248320, 5120, Rotate::None));
    let qkv = ck.cfg.get("model.language_model.layers.0.linear_attn.in_proj_qkv").unwrap();
    assert_eq!((qkv.n, qkv.k, qkv.rotate, qkv.group), (10240, 5120, Rotate::Had128In, 128));
    assert!(ck.has("model.language_model.embed_tokens.weight"));
}

#[test]
fn real_tensors_pack_like_zig() {
    let Some(d) = model() else { return };
    let zd = PathBuf::from(std::env::var("FE_XQ_ZIGPACK").unwrap_or_default());
    let ck = Checkpoint::open(&d).unwrap();
    let names = ["model.language_model.layers.0.linear_attn.in_proj_b", "model.language_model.layers.3.self_attn.k_proj",
                 "model.language_model.layers.0.linear_attn.out_proj", "model.language_model.layers.0.mlp.down_proj"];
    for name in names {
        let t = ck.xq(name).unwrap();
        t.check_codes().unwrap();
        let p = t.pack().unwrap();
        let (q2, s2) = pack::unpack(t.meta.bits, t.meta.group, t.meta.n, t.meta.k, &p).unwrap();
        assert!(q2 == t.q && s2 == t.s, "{name}: unpack(pack) != input");
        let zf = zd.join(format!("{name}.bin"));
        match std::fs::read(&zf) {
            Ok(z) => {
                let zb: Vec<u32> = z.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                assert!(zb == p, "{name}: packed bytes differ from xq.zig pack");
                eprintln!("{name}: {} bits, {} packed dwords == zig", t.meta.bits, p.len());
            }
            Err(_) => eprintln!("{name}: no zig pack dump at {zf:?} (only the roundtrip checked)"),
        }
    }
}
