// k-ds through fe-kbuild, then launch bounds (Rust has no __launch_bounds__): the kernels listed below get
// `.maxntid` / `.minnctapersm` after their parameter list and the PTX is assembled again.
use std::path::PathBuf;

const BOUNDS: &[(&str, u32, u32)] = &[("ds_attn16", 512, 2), ("ds_attn16x2", 512, 1), ("ds_attn_mq", 512, 1)];

fn main() {
    let cubin = fe_kbuild::build("k-ds");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx = std::fs::read_to_string(out.join("k_ds.ptx")).unwrap();
    let mut patched = String::with_capacity(ptx.len() + 1024);
    let mut pending: Option<String> = None;
    for line in ptx.lines() {
        if let Some(rest) = line.strip_prefix(".visible .entry ") {
            let name = rest.split('(').next().unwrap_or("");
            pending = BOUNDS.iter().find(|b| b.0 == name).map(|b| format!(".maxntid {}, 1, 1\n.minnctapersm {}", b.1, b.2));
        }
        patched.push_str(line);
        patched.push('\n');
        if pending.is_some() && line.starts_with(')') {
            patched.push_str(&pending.take().unwrap());
            patched.push('\n');
        }
    }
    let pp = out.join("k_ds.lb.ptx");
    std::fs::write(&pp, patched).unwrap();
    fe_kbuild::assemble(&pp, &cubin);
    fe_kbuild::build("k-engram");
    fe_kbuild::build("k-dspark");
}
