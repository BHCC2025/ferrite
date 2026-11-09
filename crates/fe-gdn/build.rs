// k-gdn (and k-gdn-tp2) through fe-kbuild, then `.maxntid 512` on fe_gdn_chunk (Rust has no `__launch_bounds__`): ptxas
// keeps it at <= 128 registers so a 512-thread CTA launches; then ptxas again.
use std::path::PathBuf;

fn main() {
    // the whole model and one TP2 rank (k-gdn-tp2: the same source, halved head counts)
    build("k-gdn");
    build("k-gdn-tp2");
}

fn build(name: &str) {
    let cubin = fe_kbuild::build(name);
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let stem = name.replace('-', "_");
    let ptx = std::fs::read_to_string(out.join(format!("{stem}.ptx"))).unwrap();
    let mut patched = String::with_capacity(ptx.len() + 64);
    let mut pending = false;
    for line in ptx.lines() {
        if let Some(rest) = line.strip_prefix(".visible .entry ") {
            pending = rest.starts_with("fe_gdn_chunk(");
        }
        patched.push_str(line);
        patched.push('\n');
        if pending && line.starts_with(')') {
            patched.push_str(".maxntid 512, 1, 1\n.minnctapersm 1\n");
            pending = false;
        }
    }
    let pp = out.join(format!("{stem}.lb.ptx"));
    std::fs::write(&pp, patched).unwrap();
    fe_kbuild::assemble(&pp, &cubin);
}
