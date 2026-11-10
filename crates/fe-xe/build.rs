// The Rust kernels of the forward -> SPIR-V (kernels-xe via fe-kbuild-xe), embedded by src/kx.rs (`embedded`):
// kx_core.spv (norms, GDN window + one-token kernels, attention prep, KV q8 append, chunk merge, embedding, SwiGLU, argmax,
// the MTP draft head), kx_core_dec.spv (feature dec: the matrix-engine decode attention and the prompt-window attention, loaded
// with 256 GRF), kx_xq{3,4,5,6,8}.spv (XQ matvec + prepasses and the prompt GEMM's prep / weight decode, one module per
// code width), kx_xqpf.spv (the prompt GEMM, 256 GRF).
// kx_core_dec gets fe-kxcore's stopgap (copied from crates/fe-kxcore/build.rs, keep the two in step): re-optimized from the
// `.spirv.ll` without the scalarizer, so the DPAS accumulators stay vectors and the attention kernels do not spill;
// KX_SCALARIZE=1 keeps the scalarized module. FE_KBUILD_JOBS limits the nested kernel builds on the shared box.
use std::path::Path;
use std::process::Command;

fn tool(name: &str) -> String {
    for d in std::env::var("PATH").unwrap_or_default().split(':') {
        let p = Path::new(d).join(name);
        if p.is_file() {
            return p.display().to_string();
        }
    }
    format!("/usr/bin/{name}")
}

fn ok(cmd: &mut Command) -> Result<(), String> {
    let o = cmd.output().map_err(|e| e.to_string())?;
    if o.status.success() { Ok(()) } else { Err(String::from_utf8_lossy(&o.stderr).into_owned()) }
}

fn unscalarized(out: &Path, ident: &str) -> Result<(), String> {
    let ll = out.join(format!("{ident}.spirv.ll"));
    let bc = out.join(format!("{ident}.nosc.bc"));
    let dis = out.join(format!("{ident}.nosc.ll"));
    let spv = out.join(format!("{ident}.nosc.spv"));
    ok(Command::new(tool("opt"))
        .args(["-mtriple=spirv64-unknown-unknown", "-passes=function(infer-address-spaces),default<O2>,function(infer-address-spaces,dce)", "-o"])
        .arg(&bc)
        .arg(&ll))?;
    ok(Command::new(tool("llvm-dis")).arg(&bc).arg("-o").arg(&dis))?;
    let text = std::fs::read_to_string(&dis).map_err(|e| e.to_string())?;
    for (i, _) in text.match_indices(" x ") {
        // "<N x T>": the digits before " x " right after a '<'
        let head = &text[..i];
        let Some(start) = head.rfind('<') else { continue };
        if let Ok(n) = head[start + 1..].parse::<u32>() {
            if ![2, 3, 4, 8, 16].contains(&n) {
                return Err(format!("vector width {n} in {}", dis.display()));
            }
        }
    }
    ok(Command::new(tool("llvm-spirv"))
        .args(["--spirv-fp-contract=off", &format!("--spirv-ext={}", fe_kbuild_xe::SPIRV_EXTS), "-o"])
        .arg(&spv)
        .arg(&bc))?;
    ok(Command::new(tool("spirv-val")).arg(&spv))?;
    std::fs::copy(&spv, out.join(format!("{ident}.spv"))).map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    fe_kbuild_xe::build("kx-core");
    let dec = fe_kbuild_xe::build_variant("kx-core", &["dec"], "kx_core_dec");
    println!("cargo:rerun-if-env-changed=KX_SCALARIZE");
    if !std::env::var("KX_SCALARIZE").map(|v| v == "1").unwrap_or(false) {
        if let Err(e) = unscalarized(dec.parent().unwrap(), "kx_core_dec") {
            println!("cargo:warning=kx_core_dec: keeping the scalarized module: {e}");
        }
    }
    for b in [3, 4, 5, 6, 8] {
        fe_kbuild_xe::build_variant("kx-xq", &[&format!("b{b}")], &format!("kx_xq{b}"));
    }
    fe_kbuild_xe::build("kx-xqpf");
}
