//! Rust GPU kernels for the Intel Arc Pro B70 (Xe2): `kernels-xe/kx-*` crate -> SPIR-V module for Level Zero.
//!
//! Use from a host crate's build.rs:
//! ```ignore
//! fn main() { fe_kbuild_xe::build("kx-xq"); }
//! ```
//! and in the crate: `pub static KX_XQ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kx_xq.spv"));`
//!
//! Pipeline (every step's output is kept next to the .spv for inspection):
//! 1. `cargo build --release -p <crate>` inside `kernels-xe/` (rust-toolchain.toml pins the nightly; .cargo/config.toml
//!    selects the amdgcn-amd-amdhsa carrier target, build-std=core, `--emit=llvm-bc`, `-C no-prepopulate-passes`,
//!    `-Cpanic=immediate-abort`): `<ident>.rustc.bc`, the fat-LTO module *without* LLVM optimization. rustc's own
//!    pipeline would optimize for amdgcn; IGC compiles the result best when the optimization is done for spirv64 (step
//!    4), much as ocloc hands IGC barely optimized SPIR-V: on the XQ GEMV the 9..16-row kernel went from 1595 ISA
//!    instructions + 512 B spill (amdgcn-optimized) to 1252, no spill (OpenCL: 1227).
//! 2. The nightly's llvm-tools `opt` (same LLVM as rustc): instnamer, then always-inline (kxcommon helpers and their
//!    markers into the kernels), internalize all but the kernels, globaldce, instnamer: `<ident>.named.ll`.
//! 3. [`retarget::retarget`] (address spaces, kernels, builtin pointer casts, markers): `<ident>.spirv.ll`.
//! 4. system `opt` for spirv64 ([`pipeline`]): a light pipeline (inline, SROA, instcombine, simplifycfg,
//!    infer-address-spaces: generic -> global / local) that leaves the rest to IGC, then the scalarizer + dce: LLVM makes vectors of any width (<7 x i16>, ...) and Kernel SPIR-V allows only 2, 3, 4, 8,
//!    16 components. Arithmetic only first (IGC scalarizes vector arithmetic itself, builtin calls keep their vector
//!    operands); loads / stores too only when the translator still rejects a width. Result: `<ident>.opt.bc` (+ `.opt.ll`).
//! 5. `llvm-spirv --spirv-fp-contract=off` (Rust never contracts: every entry point gets ContractionOff) with the
//!    Intel extensions enabled: `<ident>.spv`; then `spirv-val`.
//! IGC (inside the Level Zero driver) compiles the SPIR-V to Xe2 ISA at `zeModuleCreate`, as ptxas does for PTX.
pub mod retarget;

use std::path::{Path, PathBuf};
use std::process::Command;

/// Extensions llvm-spirv may use. Each one is used only when the IR needs it.
pub const SPIRV_EXTS: &str = "+SPV_INTEL_subgroups,+SPV_INTEL_fp_max_error,+SPV_INTEL_split_barrier,\
+SPV_INTEL_subgroup_buffer_prefetch,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate,\
+SPV_INTEL_bfloat16_conversion,+SPV_KHR_bit_instructions";

/// The system-opt pipeline before legalization, by name. "light" (the default): inline, SROA, instcombine, simplifycfg,
/// full unrolling of constant-trip loops (rustc lowers `[x; N]` to a loop: left in, its array stays in private memory,
/// 8.7 KB a thread on one XQ kernel), address-space inference, and nothing else (no GVN, LICM, partial unrolling), leaving
/// scheduling and register allocation to IGC, as with ocloc's barely optimized SPIR-V. On the XQ GEMV this beats a full
/// `default<O2>` (int4 m=1: 557 ISA instructions vs 578, OpenCL 559; m = 9..16: 1200 vs 1252, OpenCL 1227; int6 m = 1: 941
/// vs 988, OpenCL 958). "o1" | "o2" | "o3": `default<On>`, which some modules need: the matrix-engine decode attention
/// (kx_core_dec) spills 5.2-5.4 KB in the pack kernels on light (IGC splits the float8 accumulators after the key loop)
/// and 0 on o2. None for an unknown name.
pub fn pipeline_named(name: &str) -> Option<String> {
    match name {
        "o1" | "o2" | "o3" => Some(format!("function(infer-address-spaces),default<O{}>,function(infer-address-spaces)", &name[1..])),
        "light" => Some(
            "cgscc(inline),function(sroa,instcombine<no-verify-fixpoint>,simplifycfg,loop(loop-unroll-full),sroa,\
instcombine<no-verify-fixpoint>,infer-address-spaces,simplifycfg)"
                .into(),
        ),
        _ => None,
    }
}

/// The pipeline every module gets unless its crate asks otherwise: FE_KBUILD_PIPELINE (light | o1 | o2 | o3) if set,
/// else light.
pub fn pipeline() -> String {
    std::env::var("FE_KBUILD_PIPELINE").ok().and_then(|n| pipeline_named(&n)).unwrap_or_else(|| pipeline_named("light").unwrap())
}

/// The pipeline NAME for one build variant of `kernels-xe/<crate_name>`, so every build.rs that builds the variant (fe-xe,
/// fe-kxcore, fe-kx-tests) gets the same module. Precedence:
/// 1. FE_KBUILD_PIPELINE (an A/B experiment, applies to every module built);
/// 2. the crate's own `Cargo.toml`, keyed by the variant's cargo features joined by "," ("" = no features):
///    ```toml
///    [package.metadata.kbuild.pipeline]
///    dec = "o2"        # kx_core_dec (--features dec)
///    ```
/// 3. "light".
pub fn pipeline_for(root: &Path, crate_name: &str, features: &[&str]) -> Result<String, String> {
    if let Ok(n) = std::env::var("FE_KBUILD_PIPELINE") {
        return pipeline_named(&n).map(|_| n.clone()).ok_or_else(|| format!("fe-kbuild-xe: FE_KBUILD_PIPELINE={n}: not light | o1 | o2 | o3"));
    }
    let manifest = root.join("kernels-xe").join(crate_name).join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest).map_err(|e| format!("fe-kbuild-xe: {}: {e}", manifest.display()))?;
    let key = features.join(",");
    match metadata_pipeline(&text, &key) {
        Some(n) if pipeline_named(&n).is_some() => Ok(n),
        Some(n) => Err(format!("fe-kbuild-xe: {}: [package.metadata.kbuild.pipeline] {key:?} = {n:?}: not light | o1 | o2 | o3", manifest.display())),
        None => Ok("light".into()),
    }
}

/// `key = "value"` from the `[package.metadata.kbuild.pipeline]` table of a Cargo.toml (bare or quoted keys, `#`
/// comments). A tiny reader instead of a toml dependency: the build helper has none.
fn metadata_pipeline(manifest: &str, key: &str) -> Option<String> {
    let mut inside = false;
    for line in manifest.lines() {
        let l = line.split('#').next().unwrap_or("").trim();
        if l.starts_with('[') {
            inside = l == "[package.metadata.kbuild.pipeline]";
            continue;
        }
        if !inside || l.is_empty() {
            continue;
        }
        let Some((k, v)) = l.split_once('=') else { continue };
        let unq = |x: &str| x.trim().trim_matches('"').to_string();
        if unq(k) == key {
            return Some(unq(v));
        }
    }
    None
}

/// Workspace root (the directory holding `kernels-xe/`), from the calling crate's manifest dir or the current dir.
pub fn workspace_root() -> PathBuf {
    let mut d = std::env::var("CARGO_MANIFEST_DIR").map(PathBuf::from).unwrap_or_else(|_| std::env::current_dir().unwrap());
    while !d.join("kernels-xe").join("Cargo.toml").exists() {
        assert!(d.pop(), "fe-kbuild-xe: no kernels-xe/Cargo.toml above {}", d.display());
    }
    d
}

fn which(tool: &str, fallbacks: &[&str]) -> Result<PathBuf, String> {
    if let Some(paths) = std::env::var_os("PATH") {
        for p in std::env::split_paths(&paths) {
            let c = p.join(tool);
            if c.is_file() {
                return Ok(c);
            }
        }
    }
    for f in fallbacks {
        let c = PathBuf::from(f);
        if c.is_file() {
            return Ok(c);
        }
    }
    Err(format!("fe-kbuild-xe: `{tool}` not found on PATH or in {fallbacks:?}"))
}

fn home() -> String {
    std::env::var("HOME").unwrap_or_default()
}

/// A rustup proxy (`cargo`, `rustc`) that honours kernels-xe/rust-toolchain.toml: ~/.cargo/bin, /usr/bin (system
/// rustup), else PATH unless PATH points into a toolchain directory (inside a build script it may).
fn rustup_proxy(tool: &str) -> Result<PathBuf, String> {
    for c in [format!("{}/.cargo/bin/{tool}", home()), format!("/usr/bin/{tool}")] {
        if Path::new(&c).is_file() {
            return Ok(PathBuf::from(c));
        }
    }
    let p = which(tool, &[])?;
    if p.to_string_lossy().contains("/toolchains/") {
        return Err(format!("fe-kbuild-xe: only a toolchain `{tool}` ({}) found, no rustup proxy", p.display()));
    }
    Ok(p)
}

fn run(cmd: &mut Command, what: &str) -> Result<String, String> {
    let out = cmd.output().map_err(|e| format!("fe-kbuild-xe: running {what}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "fe-kbuild-xe: {what} failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stderr).into_owned())
}

/// A clean environment for the nested nightly cargo (the outer stable build's RUSTFLAGS, RUSTUP_TOOLCHAIN, CARGO_* and
/// target variables must not leak in).
fn nightly_cmd(prog: &Path, kdir: &Path) -> Command {
    let h = home();
    let path = std::env::var("PATH").unwrap_or_default();
    let mut cmd = Command::new(prog);
    cmd.env_clear().env("PATH", format!("{h}/.cargo/bin:{path}")).env("HOME", &h).current_dir(kdir);
    for k in ["CARGO_HOME", "RUSTUP_HOME", "TERM"] {
        if let Ok(v) = std::env::var(k) {
            cmd.env(k, v);
        }
    }
    cmd
}

fn newest_bc(dir: &Path, file: &str, best: &mut Option<(std::time::SystemTime, PathBuf)>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            newest_bc(&p, file, best);
        } else if p.file_name().map(|n| n == file).unwrap_or(false) {
            if let Ok(t) = e.metadata().and_then(|m| m.modified()) {
                if best.as_ref().map(|b| t > b.0).unwrap_or(true) {
                    *best = Some((t, p));
                }
            }
        }
    }
}

/// Build `kernels-xe/<crate_name>` into `<out_dir>/<ident>.spv` (and the intermediates). Returns the .spv path.
pub fn build_to(root: &Path, crate_name: &str, out_dir: &Path) -> Result<PathBuf, String> {
    build_variant_to(root, crate_name, &[], &crate_name.replace('-', "_"), out_dir)
}

/// As [`build_to`] with cargo features, written as `<out_dir>/<name>.spv` (one crate, several modules: kx-xq is built
/// once per code width, `--features b4` -> kx_xq4.spv).
pub fn build_variant_to(root: &Path, crate_name: &str, features: &[&str], name: &str, out_dir: &Path) -> Result<PathBuf, String> {
    let kdir = root.join("kernels-xe");
    let crate_ident = crate_name.replace('-', "_");
    let ident = name.to_string();
    std::fs::create_dir_all(out_dir).map_err(|e| format!("fe-kbuild-xe: {}: {e}", out_dir.display()))?;
    let cargo = rustup_proxy("cargo")?;
    // One cargo target dir per feature set: cargo keeps each feature set's output in its own hash directory, so with
    // several variants of one crate in one target dir "the newest .bc" may belong to another variant (a fresh build
    // writes nothing). The default build uses kernels-xe/target.
    let tdir = if features.is_empty() { kdir.join("target") } else { kdir.join("target").join("v").join(&ident) };
    let mut cb = nightly_cmd(&cargo, &kdir);
    cb.args(["build", "--release", "-p", crate_name, "--target-dir"]).arg(&tdir);
    if !features.is_empty() {
        cb.args(["--features", &features.join(",")]);
    }
    if let Ok(j) = std::env::var("FE_KBUILD_JOBS") {
        cb.args(["-j", &j]); // the shared box: keep the nested build's parallelism down when asked
    }
    run(&mut cb, &format!("cargo build -p {crate_name} {features:?}"))?;
    // the post-LTO bitcode of the cdylib (cargo's build-dir layout puts it under .../build/<crate>/<hash>/out/)
    let mut best = None;
    newest_bc(&tdir.join("amdgcn-amd-amdhsa").join("release"), &format!("{crate_ident}.bc"), &mut best);
    let (_, bc) = best.ok_or_else(|| format!("fe-kbuild-xe: no {crate_ident}.bc under {}", tdir.display()))?;
    let rustc_bc = out_dir.join(format!("{ident}.rustc.bc"));
    std::fs::copy(&bc, &rustc_bc).map_err(|e| format!("copy {}: {e}", bc.display()))?;

    // 2. the nightly's own opt (same LLVM as rustc): name every value, inline the #[inline(always)] helpers (kxcommon
    //    markers must sit in the kernels), keep only the kernels and what they reach
    let sysroot = run_stdout(nightly_cmd(&rustup_proxy("rustc")?, &kdir).args(["--print", "sysroot"]))?;
    let ropt = PathBuf::from(sysroot.trim()).join("lib/rustlib/x86_64-unknown-linux-gnu/bin/opt");
    let named0 = out_dir.join(format!("{ident}.rustc.ll"));
    run(Command::new(&ropt).args(["-passes=instnamer", "-S", "-o"]).arg(&named0).arg(&rustc_bc), "opt -passes=instnamer (rust llvm-tools)")?;
    let kernels: Vec<String> = std::fs::read_to_string(&named0)
        .map_err(|e| format!("{}: {e}", named0.display()))?
        .lines()
        .filter(|l| l.starts_with("define ") && l.contains(" amdgpu_kernel "))
        .filter_map(|l| l.split(" @").nth(1).and_then(|r| r.split('(').next()).map(String::from))
        .collect();
    if kernels.is_empty() {
        return Err(format!("fe-kbuild-xe: no kernels (extern \"gpu-kernel\" fns) in {crate_name}"));
    }
    let named = out_dir.join(format!("{ident}.named.ll"));
    run(
        Command::new(&ropt)
            .args(["-passes=always-inline,internalize,globaldce,instnamer", "-S"])
            .arg(format!("-internalize-public-api-list={}", kernels.join(",")))
            .arg("-o")
            .arg(&named)
            .arg(&rustc_bc),
        "opt always-inline/internalize (rust llvm-tools)",
    )?;

    // 3. retarget
    let ir = std::fs::read_to_string(&named).map_err(|e| format!("{}: {e}", named.display()))?;
    let spir = retarget::retarget(&ir).map_err(|e| format!("fe-kbuild-xe: retarget {crate_name}: {e}"))?;
    let spir_ll = out_dir.join(format!("{ident}.spirv.ll"));
    std::fs::write(&spir_ll, spir).map_err(|e| format!("{}: {e}", spir_ll.display()))?;

    // 4. system opt (has the SPIR-V target) + 5. llvm-spirv + spirv-val, as a cascade. First WITHOUT the scalarizer: it
    //    splits loop-carried <8 x float> DPAS accumulator phis into scalars that IGC re-packs for every DPAS (
    //    attention decode kernels spilled 3.7-7 KB and ran at 0.55x of OpenCL; unscalarized: no spill, 0.96-1.07x). Only
    //    when the translator or the validator rejects the module (LLVM can make vectors of any width, <7 x i16> and the
    //    like; Kernel SPIR-V allows 2, 3, 4, 8, 16 components) the scalarizer legalizes them: on arithmetic first, so legal
    //    vector loads / stores stay single messages (scalarized 64-byte lane loads cost lb_vl16 a third of its rate), and
    //    with loads / stores too as the last resort. FE_KBUILD_SCALARIZE=1 starts at the arithmetic scalarizer (A/B).
    let sopt = which("opt", &["/usr/bin/opt"])?;
    let opt_bc = out_dir.join(format!("{ident}.opt.bc"));
    let spv = out_dir.join(format!("{ident}.spv"));
    let llvm_spirv = which("llvm-spirv", &["/usr/bin/llvm-spirv"])?;
    let spirv_val = which("spirv-val", &["/usr/bin/spirv-val"])?;
    let mut steps = vec!["", "scalarizer<variable-insert-extract>,", "scalarizer<load-store;variable-insert-extract>,"];
    if std::env::var("FE_KBUILD_SCALARIZE").map(|v| v == "1").unwrap_or(false) {
        steps.remove(0);
    }
    let pname = pipeline_for(root, crate_name, features)?;
    let pipe = pipeline_named(&pname).expect("checked by pipeline_for");
    std::fs::write(out_dir.join(format!("{ident}.pipeline.txt")), format!("{pname}\n")).map_err(|e| format!("{e}"))?;
    let mut last_err = String::new();
    for scal in steps {
        let passes = format!("-passes={pipe},function({scal}dce),globaldce");
        run(Command::new(&sopt).args(["-mtriple=spirv64-unknown-unknown", &passes, "-o"]).arg(&opt_bc).arg(&spir_ll), "opt (spirv64)")?;
        let r = run(
            Command::new(&llvm_spirv).args(["--spirv-fp-contract=off", &format!("--spirv-ext={SPIRV_EXTS}"), "-o"]).arg(&spv).arg(&opt_bc),
            "llvm-spirv (\"Unsupported vector type\" / i128 / i256 / i512 even after the scalarizer: an array of vectors indexed in a loop that LLVM did not unroll before SROA (use kxcommon::unroll!), or a byte-wise copy (use kxcommon::bits); see the .opt.ll)",
        )
        .and_then(|_| run(Command::new(&spirv_val).arg(&spv), "spirv-val"));
        match r {
            Ok(_) => {
                last_err.clear();
                if !scal.is_empty() {
                    println!("cargo:warning=fe-kbuild-xe: {crate_name} ({ident}) needed `{}` to be valid SPIR-V", scal.trim_end_matches(','));
                }
                std::fs::write(out_dir.join(format!("{ident}.legalize.txt")), if scal.is_empty() { "none\n" } else { scal })
                    .map_err(|e| format!("{e}"))?;
                break;
            }
            Err(e) => last_err = e,
        }
    }
    if !last_err.is_empty() {
        return Err(last_err);
    }
    let opt_ll = out_dir.join(format!("{ident}.opt.ll"));
    let _ = Command::new(which("llvm-dis", &["/usr/bin/llvm-dis"])?).arg(&opt_bc).arg("-o").arg(&opt_ll).output();
    check_invariants(&spv)?;
    Ok(spv)
}

fn run_stdout(cmd: &mut Command) -> Result<String, String> {
    let out = cmd.output().map_err(|e| format!("fe-kbuild-xe: {e}"))?;
    if !out.status.success() {
        return Err(format!("fe-kbuild-xe: {}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `spirv-dis` of a module with every `%id` replaced by `%`, one instruction a line (for invariant and golden checks).
pub fn disassemble(spv: &Path) -> Result<String, String> {
    let dis = run_stdout(Command::new(which("spirv-dis", &["/usr/bin/spirv-dis"])?).arg(spv))?;
    Ok(dis
        .lines()
        .map(|l| l.split_whitespace().map(|t| if t.starts_with('%') { "%" } else { t }).collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n"))
}

/// Invariants every kernels-xe module must satisfy (checked on every build, so a toolchain change that breaks the route
/// fails the build instead of miscompiling): OpenCL memory model; every kernel entry point has ContractionOff (Rust
/// never contracts) and a SubgroupSize; no kxcommon marker or amdgcn name survived the retarget.
pub fn check_invariants(spv: &Path) -> Result<(), String> {
    let dis = disassemble(spv)?;
    let mut errs = Vec::new();
    if !dis.contains("OpMemoryModel Physical64 OpenCL") {
        errs.push("no `OpMemoryModel Physical64 OpenCL`".to_string());
    }
    let entries = dis.lines().filter(|l| l.starts_with("OpEntryPoint Kernel")).count();
    let modes = |m: &str| dis.lines().filter(|l| l.starts_with("OpExecutionMode") && l.contains(m)).count();
    if entries == 0 {
        errs.push("no kernel entry points".into());
    }
    if modes(" ContractionOff") != entries {
        errs.push(format!("{entries} kernels but {} ContractionOff execution modes", modes(" ContractionOff")));
    }
    if modes(" SubgroupSize ") != entries {
        errs.push(format!("{entries} kernels but {} SubgroupSize execution modes", modes(" SubgroupSize ")));
    }
    // marker functions (the SLM globals `__xe_slm_<bytes>_<id>` are the intended result and stay)
    for bad in ["__xe_attr_", "__xe_fdiv_ocl", "__xe_local_arg", "\"__xe_slm\"", "amdgcn", "amdgpu"] {
        if dis.contains(bad) {
            errs.push(format!("`{bad}` left in the module"));
        }
    }
    if errs.is_empty() { Ok(()) } else { Err(format!("fe-kbuild-xe: {} breaks the route invariants: {}", spv.display(), errs.join("; "))) }
}

/// Golden check: every pattern must appear in the normalized disassembly ([`disassemble`]). fe-kx-tests runs it on the
/// probe module kx_test.spv (ExecutionModes, block IO, shuffles, collectives, DPAS / 2D builtin imports, rounding
/// decorations, SLM storage), so a rustc nightly bump that changes the bitcode fails the build loudly.
pub fn golden(spv: &Path, patterns: &[&str]) -> Result<(), String> {
    let dis = disassemble(spv)?;
    let missing: Vec<&&str> = patterns.iter().filter(|p| !dis.contains(**p)).collect();
    if missing.is_empty() { Ok(()) } else { Err(format!("fe-kbuild-xe golden check of {}: missing {missing:?}", spv.display())) }
}

/// build.rs entry: build `kernels-xe/<crate_name>` into `$OUT_DIR/<ident>.spv` and register rerun triggers.
pub fn build(crate_name: &str) -> PathBuf {
    build_variant(crate_name, &[], &crate_name.replace('-', "_"))
}

/// build.rs entry with cargo features: `$OUT_DIR/<name>.spv`.
pub fn build_variant(crate_name: &str, features: &[&str], name: &str) -> PathBuf {
    let root = workspace_root();
    let kdir = root.join("kernels-xe");
    for p in [
        kdir.join(crate_name),
        kdir.join("kxcommon"),
        kdir.join("Cargo.toml"),
        kdir.join(".cargo"),
        kdir.join("rust-toolchain.toml"),
        root.join("crates/fe-kbuild-xe/src"),
    ] {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    println!("cargo:rerun-if-env-changed=FE_KBUILD_PIPELINE");
    println!("cargo:rerun-if-env-changed=FE_KBUILD_SCALARIZE");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    match build_variant_to(&root, crate_name, features, name, &out_dir) {
        Ok(p) => p,
        Err(e) => panic!("{e}"),
    }
}

#[cfg(test)]
mod pipeline_tests {
    use super::metadata_pipeline;

    #[test]
    fn metadata_pipeline_reader() {
        let m = "[package]\nname = \"kx-core\"\n\n[package.metadata.kbuild.pipeline]\n# per variant\ndec = \"o2\"   # attention\n\"a,b\" = \"o1\"\n\n[dependencies]\ndec = \"x\"\n";
        assert_eq!(metadata_pipeline(m, "dec").as_deref(), Some("o2"));
        assert_eq!(metadata_pipeline(m, "a,b").as_deref(), Some("o1"));
        assert_eq!(metadata_pipeline(m, ""), None);
        assert_eq!(metadata_pipeline("[package]\nname = \"kx-xq\"\n", "b4"), None);
    }
}
