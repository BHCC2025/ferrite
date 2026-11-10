//! fe-xq's CPU reference against arc-engine's B70 XQ kernels on real checkpoint tensors (oracle `xq-mix*/xqmv/*.bin`, written by
//! `xpu-xq_oracle-test`; FE_XQ_ORACLE names the oracle folder; skipped without it). For every dump and row count m:
//! - the rotation prepass xt and xs in the device layout (padding rows included) bit for bit; the xs sums depend on the sub-group
//!   reduction order (cpu::subgroup_sum16 models it as a halving tree);
//! - y fp32 against the float64 product within the kernel oracle's tolerance (1e-4 of max|y|), and the fp32 kernel model's exact-match rate;
//! - y bf16 == RNE(y fp32) (the kernel's two outputs agree).
//! The checkpoint of a dump dir: xq-mix5.0 -> .../xq/gptq-mix5.0 (FE_XQ_ROOT: the parent folder).
use fe_xq::cpu::{self, Act};
use fe_xq::fixture::XqMvDump;
use fe_xq::{pack, Checkpoint};
use std::path::PathBuf;

#[test]
fn kernels_vs_cpu_reference() {
    let root = PathBuf::from(std::env::var("FE_XQ_ORACLE").unwrap_or_default());
    let xqroot = PathBuf::from(std::env::var("FE_XQ_ROOT").unwrap_or_default());
    let Ok(rd) = std::fs::read_dir(&root) else {
        eprintln!("SKIP: no oracle dir {root:?}");
        return;
    };
    let mut dirs: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.join("xqmv").is_dir()).collect();
    dirs.sort();
    if dirs.is_empty() {
        eprintln!("SKIP: no xq-*/xqmv dumps in {root:?} (oracle/make-oracle.sh xq50)");
        return;
    }
    for d in dirs {
        let tag = d.file_name().unwrap().to_string_lossy().to_string();
        let ck = Checkpoint::open(&xqroot.join(tag.replace("xq-", "gptq-"))).unwrap();
        let mut files: Vec<PathBuf> = std::fs::read_dir(d.join("xqmv")).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
        files.sort();
        for f in files {
            let name = f.file_stem().unwrap().to_string_lossy().to_string();
            let dump = XqMvDump::read(&f).unwrap();
            let t = ck.xq(&name).unwrap();
            assert_eq!((dump.bits, dump.group, dump.n, dump.k), (t.meta.bits, t.meta.group, t.meta.n, t.meta.k), "{name}: shape");
            assert_eq!(dump.split, pack::split_for(dump.n, dump.k, dump.group, None), "{name}: split");
            let w = t.pack().unwrap();
            for c in &dump.cases {
                let m = c.m as usize;
                let p = cpu::prep(Act::Bf16(&dump.x), dump.k, m, dump.k, t.su.as_deref());
                let (xt_dev, xs_dev) = cpu::prep_device(Act::Bf16(&dump.x), dump.k, m, dump.k, t.su.as_deref());
                let xt_diff = xt_dev.iter().zip(&c.xt).filter(|(a, b)| a != b).count();
                let xs_diff = xs_dev.iter().zip(&c.xs).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                let xs_rel = xs_dev.iter().zip(&c.xs).fold(0f32, |a, (u, v)| a.max((u - v).abs() / v.abs().max(1e-3)));
                assert_eq!(xt_diff, 0, "{tag}/{name} m {m}: {xt_diff} rotated values differ from the kernel's");
                assert_eq!(xs_diff, 0, "{tag}/{name} m {m}: {xs_diff} group sums differ from the kernel's (max rel {xs_rel:e})");
                let y64 = cpu::gemv_f64(dump.group, dump.n, dump.k, t.q, &t.s, &p.xt, m);
                let mx = y64.iter().fold(0f64, |a, &b| a.max(b.abs()));
                let err = c.y32.iter().zip(&y64).fold(0f64, |a, (&u, &v)| a.max((u as f64 - v).abs()));
                assert!(err <= 1e-4 * mx, "{tag}/{name} m {m}: kernel y err {err:e} of max {mx}");
                let bf_diff = c.y32.iter().zip(&c.y16).filter(|(u, b)| cpu::to_bf(**u) != **b).count();
                assert_eq!(bf_diff, 0, "{tag}/{name} m {m}: bf16 output != RNE(fp32 output) in {bf_diff} places");
                // the fp32 model on the kernel's own prepass (xs from the dump: the model then differs only inside the DPAS chain)
                let mut pk = p.clone();
                if xs_diff != 0 {
                    for row in 0..m {
                        let (mw, _) = pack::VARIANTS[pack::variant_for(c.m).unwrap()];
                        let (rg, r) = (row / mw as usize, row % mw as usize);
                        for j in 0..dump.k / 32 {
                            pk.xs[row * (dump.k / 32) + j] = c.xs[(rg * (dump.k / 32) + j) * mw as usize + r];
                        }
                    }
                }
                let model_exact = if m == 1 && dump.n * dump.k <= 128 << 20 {
                    let ym = cpu::gemv_model(dump.bits, dump.group, dump.n, dump.k, &w, &pk, dump.split);
                    let same = ym.iter().zip(&c.y32).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
                    format!("{:.2} %", 100.0 * same as f64 / ym.len() as f64)
                } else {
                    "skipped".into()
                };
                eprintln!("{tag}/{name} bits {} m {m:>2}: xt equal, xs differ {xs_diff}/{} (max rel {xs_rel:.1e}), y err {:.1e} of max|y| {mx:.2}, model exact {model_exact}",
                    dump.bits, xs_dev.len(), err / mx);
            }
        }
    }
}
