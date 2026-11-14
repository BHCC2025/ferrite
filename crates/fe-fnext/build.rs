// k-fnext through fe-kbuild, then `.maxntid 512` on the 512-thread kernels (Rust has no `__launch_bounds__`): ptxas
// keeps them at <= 128 registers so a 512-thread block launches; then ptxas again (as crates/fe-gdn does).
use std::path::PathBuf;

fn main() {
    let cubin = fe_kbuild::build("k-fnext");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx = std::fs::read_to_string(out.join("k_fnext.ptx")).unwrap();
    let mut patched = String::with_capacity(ptx.len() + 256);
    let mut pending = false;
    for line in ptx.lines() {
        if let Some(rest) = line.strip_prefix(".visible .entry ") {
            pending = rest.starts_with("fx_gdn_chunk(") || rest.starts_with("fn_gdn_rec(");
        }
        patched.push_str(line);
        patched.push('\n');
        if pending && line.starts_with(')') {
            patched.push_str(".maxntid 512, 1, 1\n.minnctapersm 1\n");
            pending = false;
        }
    }
    let pp = out.join("k_fnext.lb.ptx");
    std::fs::write(&pp, patched).unwrap();
    fe_kbuild::assemble(&pp, &cubin);
}
