//! Tests on the real Coder files (skipped when the files are not on this box). `DSC_MODEL` overrides the path.
//! While the download is in progress the model opens partially; once complete the stricter assertions apply.
use fe_dsc::{config::DscConfig, tensors};

fn model() -> Option<std::path::PathBuf> {
    let p = fe_dsc::model_path();
    p.exists().then_some(p)
}

fn open() -> Option<(fe_gguf::Gguf, DscConfig)> {
    let p = model()?;
    let e = fe_dsc::engram_path(&p);
    let g = fe_dsc::open_model(&p, e.as_deref(), true).unwrap();
    let c = DscConfig::from_gguf(&g).unwrap();
    Some((g, c))
}

#[test]
fn config_and_roles() {
    let Some((_, c)) = open() else { return };
    assert_eq!((c.n_layer, c.n_embd, c.n_vocab, c.n_head, c.head_dim, c.n_head_kv), (40, 5120, 129280, 64, 512, 1));
    assert_eq!((c.n_expert, c.n_expert_used, c.n_expert_shared, c.n_ff_exp), (192, 6, 1, 2304));
    assert_eq!((c.hc, c.hc_iters, c.gating_func, c.weights_scale), (4, 20, 4, 1.5));
    assert_eq!(c.hc_mix(), 24);
    assert_eq!(c.engram_layers, vec![1, 14]);
    assert_eq!(c.engram_width(), 6144);
    // roles: window-only, full (ratio 2), reuse, candidate source, reindex, reuse of a reindex
    assert_eq!(c.role(0).ratio, 0);
    assert!(c.role(1).engram && c.role(14).engram && !c.role(2).engram);
    let r2 = c.role(2);
    assert!(r2.kv_source && r2.index_source && r2.kv_from == Some(2) && r2.sel_from == Some(2));
    let r5 = c.role(5);
    assert!(!r5.kv_source && !r5.index_source && r5.kv_from == Some(2) && r5.sel_from == Some(2));
    let r20 = c.role(20);
    assert!(r20.cand_source && r20.kv_source && r20.ratio == 1 && !r20.cand_masked);
    let r24 = c.role(24);
    assert!(r24.index_source && !r24.kv_source && r24.cand_masked && r24.kv_from == Some(20) && r24.sel_from == Some(24));
    let r27 = c.role(27);
    assert!(!r27.index_source && r27.sel_from == Some(24) && !r27.cand_masked);
    assert_eq!(c.role(39).sel_from, Some(36));
    // the token map is a surjection onto 0..max
    let max = *c.token_map.iter().max().unwrap();
    let mut seen = vec![false; max as usize + 1];
    for &t in &c.token_map {
        seen[t as usize] = true;
    }
    assert!(seen.iter().all(|&x| x), "token map has holes");
    assert_eq!(max + 1, 99092);
    // engram row bookkeeping: layer 1's table is the first 24 head buckets, layer 14 starts at their sum
    let (rows1, off1) = tensors::engram_rows(&c, 0);
    let (rows14, off14) = tensors::engram_rows(&c, 1);
    assert_eq!(off1, 0);
    assert_eq!(off14, rows1);
    assert_eq!(rows1, 384006168);
    assert_eq!(rows14, 384016682);
}

#[test]
fn tensors_match_spec() {
    let Some((g, c)) = open() else { return };
    let rep = tensors::check(&g, &c);
    assert!(rep.errors.is_empty(), "{:#?}", rep.errors);
    assert!(rep.checked > 0);
    // the routed-expert quant mix the kernels must cover
    let mut ty: std::collections::BTreeSet<&str> = Default::default();
    for t in g.tensors.iter().filter(|t| t.name.contains("_exps.")) {
        ty.insert(t.ty.name());
    }
    assert!(ty.iter().all(|t| ["IQ1_M", "IQ2_XXS", "IQ2_XS"].contains(t)), "{ty:?}");
    if g.missing.is_empty() {
        // everything downloaded: 1042 backbone tensors in the five files (split.tensors.count says 1043: the
        // metadata counts one tensor the files do not hold) + the 2 Engram tables, all mapped
        assert_eq!(g.tensors.len(), 1042 + 2, "tensor count");
        assert!(rep.pending.is_empty(), "{:?}", rep.pending);
        assert!(rep.unexpected.is_empty(), "{:#?}", rep.unexpected);
        // Engram tables: rows x 144 B (IQ4_NL, 256 values a row)
        for (i, &l) in c.engram_layers.iter().enumerate() {
            let t = g.tensor(&tensors::engram_table_name(l)).expect("engram table");
            assert_eq!(t.dims, vec![c.ple_dim as u64, tensors::engram_rows(&c, i).0]);
            assert_eq!(t.row_bytes(), 144);
        }
    }
}

#[test]
fn present_tensors_are_readable() {
    let Some((g, c)) = open() else { return };
    // the router and a norm are the right size and finite; a BF16 router row converts
    let t = g.tensor("blk.0.ffn_gate_inp.weight").expect("router");
    let d = g.data(t);
    assert_eq!(d.len(), c.n_embd * c.n_expert * 2);
    let n = g.tensor("blk.0.attn_norm.weight").unwrap();
    let f: Vec<f32> = g.data(n).chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    assert_eq!(f.len(), c.n_embd);
    assert!(f.iter().all(|x| x.is_finite()));
}

#[test]
fn engram_rows_and_vectors() {
    let Some((g, c)) = open() else { return };
    if !g.missing.is_empty() || g.tensor("blk.1.engram_embd.weight").is_none() {
        return;
    }
    let e = fe_dsc::engram::Engram::new(&g, &c).unwrap();
    let ids: Vec<u32> = vec![0, 5, 100, 20000, 129000, 7, 7, 7];
    for li in 0..2 {
        let (rows, _) = fe_dsc::tensors::engram_rows(&c, li);
        for pos in 0..ids.len() {
            let h = e.hashes(&ids, pos, li);
            assert_eq!(h.len(), 24);
            assert!(h.iter().all(|&r| r >= 0 && (r as u64) < rows), "row out of the table");
            // each column stays inside its own bucket range
            let cols = c.engram_cols();
            for (k, &r) in h.iter().enumerate() {
                let lo = c.head_off[li * cols + k] - c.head_off[li * cols];
                assert!(r as u64 >= lo && (r as u64) < lo + c.head_vocab[li * cols + k]);
            }
        }
        // position 0 has no history: a 2..4-gram column hashes the pad token, so the 3 orders agree on (pad, pad...)
        let v = e.vector(&ids, 3, li).unwrap();
        assert_eq!(v.len(), 6144);
        assert!(v.iter().all(|x| x.is_finite()) && v.iter().any(|&x| x != 0.0));
    }
    // deterministic
    assert_eq!(e.hashes(&ids, 4, 0), e.hashes(&ids, 4, 0));
}
