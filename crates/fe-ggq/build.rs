// k-gguf through fe-kbuild, then launch bounds: Rust has no `__launch_bounds__`, so the int8 GEMM entries
// (`ggq_mmq_*`, 256 threads) can get `.maxntid 256, 1, 1` + `.minnctapersm N` (FE_MMQ_MINCTA=N; off by default:
// capping registers at 128 spills badly)
// inserted after their parameter list, and the PTX is assembled again. FE_MMQ_MINCTA overrides the 2.
use std::path::PathBuf;

fn main() {
    let cubin = fe_kbuild::build("k-gguf");
    println!("cargo:rerun-if-env-changed=FE_MMQ_MINCTA");
    let minc = std::env::var("FE_MMQ_MINCTA").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1);
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx = std::fs::read_to_string(out.join("k_gguf.ptx")).unwrap();
    let mut patched = String::with_capacity(ptx.len() + 4096);
    let mut pending: Option<String> = None;
    for line in ptx.lines() {
        if let Some(rest) = line.strip_prefix(".visible .entry ") {
            let name = rest.split('(').next().unwrap_or("");
            pending = if name.starts_with("ggq_mmq_") && minc > 0 {
                Some(format!(".maxntid 384, 1, 1\n.minnctapersm {minc}"))
            } else if name.starts_with("ggq_hc_dw_b") {
                Some(".maxntid 288, 1, 1\n.minnctapersm 2".to_string())
            } else if name == "ggq_fa_tc" || name == "ggq_fa_tc2" {
                Some(".maxntid 128, 1, 1\n.minnctapersm 2".to_string())
            } else {
                None
            };
        }
        patched.push_str(line);
        patched.push('\n');
        if pending.is_some() && line.starts_with(')') {
            patched.push_str(&pending.take().unwrap());
            patched.push('\n');
        }
    }
    let pp = out.join("k_gguf.lb.ptx");
    std::fs::write(&pp, patched).unwrap();
    fe_kbuild::assemble(&pp, &cubin);
}
