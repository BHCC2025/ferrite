// k-exl3 through fe-kbuild, then launch bounds: Rust has no `__launch_bounds__`, so kernels named
// `..._w{warps}_s{stages}_o{programs}` get `.maxntid {32 * warps}, 1, 1` and `.minnctapersm {programs}`
// inserted after their parameter list, and the PTX is assembled again.
use std::path::PathBuf;

fn launch_bounds(name: &str) -> Option<String> {
    let mut warps = None;
    let mut occ = None;
    for part in name.split('_') {
        if let Some(v) = part.strip_prefix('w') {
            warps = v.parse::<u32>().ok();
        }
        if let Some(v) = part.strip_prefix('o') {
            occ = v.parse::<u32>().ok();
        }
    }
    Some(format!(".maxntid {}, 1, 1\n.minnctapersm {}", 32 * warps?, occ?))
}

fn main() {
    let cubin = fe_kbuild::build("k-exl3");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx = std::fs::read_to_string(out.join("k_exl3.ptx")).unwrap();
    let mut patched = String::with_capacity(ptx.len() + 4096);
    let mut pending: Option<String> = None;
    for line in ptx.lines() {
        if let Some(rest) = line.strip_prefix(".visible .entry ") {
            let name = rest.split('(').next().unwrap_or("");
            pending = if name.contains("_o") { launch_bounds(name) } else { None };
        }
        patched.push_str(line);
        patched.push('\n');
        if pending.is_some() && line.starts_with(')') {
            patched.push_str(&pending.take().unwrap());
            patched.push('\n');
        }
    }
    let pp = out.join("k_exl3.lb.ptx");
    std::fs::write(&pp, patched).unwrap();
    fe_kbuild::assemble(&pp, &cubin);
}
