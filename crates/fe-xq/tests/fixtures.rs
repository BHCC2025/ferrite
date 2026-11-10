//! fe-xq against arc-engine's numpy oracle fixtures (`tools/make_xq_fixtures.py`; not in this repository).
//! FE_XQ_FIXTURES names their directory; the test skips without it.
//! - pack: byte-identical to the independent numpy packer (and so to xq.zig pack, which xq_test.zig checks the same way);
//! - prep: rotated rows bit-identical to the numpy emulation of xq_prep;
//! - gemv_f64: the float64 product (1e-12 of max|y|); gemv_model: the decode matvec model within the kernel oracle's 1e-4 of max|y|.
use fe_xq::cpu::{self, Act};
use fe_xq::fixture::XqFixture;
use fe_xq::pack;
use std::path::PathBuf;

fn dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("FE_XQ_FIXTURES").unwrap_or_default());
    if d.is_dir() { Some(d) } else {
        eprintln!("SKIP: no fixtures at {d:?} (python3 tools/make_xq_fixtures.py DIR in arc-engine)");
        None
    }
}

#[test]
fn numpy_fixtures() {
    let Some(d) = dir() else { return };
    let mut files: Vec<_> = std::fs::read_dir(&d).unwrap().filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("xq_")).collect();
    files.sort();
    assert!(!files.is_empty(), "no xq_*.bin in {d:?}");
    for f in files {
        let fx = XqFixture::read(&f).unwrap();
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        // pack
        let p = pack::pack(fx.bits, fx.g, fx.n, fx.k, &fx.q, &fx.s).unwrap();
        assert!(p == fx.packed, "{name}: pack differs from the numpy packer");
        // prep
        let pr = cpu::prep(Act::Bf16(&fx.x), fx.k, fx.m, fx.k, if fx.rot { Some(&fx.su) } else { None });
        let diff = pr.xt.iter().zip(&fx.xt).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "{name}: {diff} rotated values differ from the numpy emulation");
        // float64 product
        let y = cpu::gemv_f64(fx.g, fx.n, fx.k, &fx.q, &fx.s, &pr.xt, fx.m);
        let mx = fx.y.iter().fold(0f64, |a, &b| a.max(b.abs()));
        let e64 = y.iter().zip(&fx.y).fold(0f64, |a, (&u, &v)| a.max((u - v).abs()));
        assert!(e64 <= 1e-12 * mx, "{name}: f64 gemv err {e64:e} of max {mx}");
        // fp32 kernel model on the packed layout
        let ym = cpu::gemv_model(fx.bits, fx.g, fx.n, fx.k, &p, &pr, pack::split_for(fx.n, fx.k, fx.g, None));
        let em = ym.iter().zip(&fx.y).fold(0f64, |a, (&u, &v)| a.max((u as f64 - v).abs()));
        assert!(em <= 1e-4 * mx, "{name}: model err {em:e} of max {mx}");
        eprintln!("{name}: bits {} g {} rot {} [{}x{}] m {}: pack equal, xt equal, f64 err {:.1e}, model err {:.1e} of max|y| {:.3}",
            fx.bits, fx.g, fx.rot, fx.n, fx.k, fx.m, e64 / mx, em / mx, mx);
    }
}
