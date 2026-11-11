//! Use from a host crate's build.rs:
//!
//! ```ignore
//! fn main() { fe_kbuild::build("k-core"); }
//! ```
//! and in the crate: `pub static K_CORE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/k_core.cubin"));`
//!
//! Steps: `cargo build --release -p <crate>` inside `kernels/` (its rust-toolchain.toml pins the nightly, its
//! .cargo/config.toml selects nvptx64-nvidia-cuda + target-cpu=sm_121a + build-std=core), then
//! `ptxas -arch=sm_121a -O3` on the PTX. Both PTX and cubin land in OUT_DIR. Only the named crate (and
//! kcommon) are watched, so editing another kernel crate never rebuilds this one.
use std::path::{Path, PathBuf};
use std::process::Command;

/// The default kernel architecture (GB10, DGX Spark). `FE_KARCH` overrides it at build time (e.g. `sm_120a` for
/// RTX 50-series cards): the kernels are then compiled with that `target-cpu` and assembled with `ptxas -arch` of
/// the same name, in their own target directory.
pub const ARCH: &str = "sm_121a";

/// The kernel architecture of this build: `FE_KARCH`, else [`ARCH`].
pub fn arch() -> String {
    match std::env::var("FE_KARCH") {
        Ok(a) if !a.trim().is_empty() => a.trim().to_string(),
        _ => ARCH.to_string(),
    }
}

/// Workspace root (the directory holding `kernels/`), found from the calling crate's manifest dir.
pub fn workspace_root() -> PathBuf {
    let mut d = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    while !d.join("kernels").join("Cargo.toml").exists() {
        assert!(d.pop(), "fe-kbuild: no kernels/Cargo.toml above CARGO_MANIFEST_DIR");
    }
    d
}

fn which(tool: &str, fallbacks: &[&str]) -> PathBuf {
    if let Some(paths) = std::env::var_os("PATH") {
        for p in std::env::split_paths(&paths) {
            let c = p.join(tool);
            if c.is_file() {
                return c;
            }
        }
    }
    for f in fallbacks {
        let c = PathBuf::from(f);
        if c.is_file() {
            return c;
        }
    }
    panic!("fe-kbuild: `{tool}` not found on PATH or in {fallbacks:?}")
}

fn home() -> String {
    std::env::var("HOME").unwrap_or_default()
}

/// Build kernels/<crate_name> and return the cubin path (`$OUT_DIR/<crate_ident>.cubin`).
pub fn build(crate_name: &str) -> PathBuf {
    let root = workspace_root();
    let kdir = root.join("kernels");
    for p in [kdir.join(crate_name), kdir.join("kcommon"), kdir.join("Cargo.toml"), kdir.join(".cargo"),
        kdir.join("rust-toolchain.toml")]
    {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    println!("cargo:rerun-if-env-changed=FE_PTXAS_FLAGS");
    println!("cargo:rerun-if-env-changed=FE_KARCH");
    let arch = arch();
    println!("cargo:rustc-env=FE_KARCH_BUILT={arch}");
    let ident = crate_name.replace('-', "_");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let cubin = out_dir.join(format!("{ident}.cubin"));

    // Nested cargo with a clean environment: the outer (stable, host) build's RUSTFLAGS, RUSTUP_TOOLCHAIN,
    // CARGO_* and target variables must not leak into the nightly nvptx build.
    let h = home();
    // the rustup proxies live in $CARGO_HOME/bin (default ~/.cargo/bin)
    let cbin = match std::env::var("CARGO_HOME") {
        Ok(c) if !c.is_empty() => format!("{c}/bin"),
        _ => format!("{h}/.cargo/bin"),
    };
    let cargo = which("cargo", &[&format!("{cbin}/cargo")]);
    let cargo = if cargo.to_string_lossy().contains("/toolchains/") {
        // PATH points into the stable toolchain dir; use the rustup proxy so rust-toolchain.toml applies.
        PathBuf::from(format!("{cbin}/cargo"))
    } else {
        cargo
    };
    let path = std::env::var("PATH").unwrap_or_default();
    // the default architecture keeps kernels/target; another one gets its own directory so the two never thrash
    let tdir = if arch == ARCH { kdir.join("target") } else { kdir.join(format!("target-{arch}")) };
    let mut cmd = Command::new(&cargo);
    cmd.env_clear()
        .env("PATH", format!("{cbin}:{path}"))
        .env("HOME", &h)
        .current_dir(&kdir)
        .args(["build", "--release", "-p", crate_name, "--target-dir"])
        .arg(&tdir);
    if arch != ARCH {
        // overrides the target-cpu of kernels/.cargo/config.toml
        cmd.env("CARGO_TARGET_NVPTX64_NVIDIA_CUDA_RUSTFLAGS", format!("-C target-cpu={arch}"));
    }
    for k in ["CARGO_HOME", "RUSTUP_HOME", "TERM"] {
        if let Ok(v) = std::env::var(k) {
            cmd.env(k, v);
        }
    }
    let out = cmd.output().unwrap_or_else(|e| panic!("fe-kbuild: running {}: {e}", cargo.display()));
    if !out.status.success() {
        panic!("fe-kbuild: kernel crate `{crate_name}` failed to build:\n{}", String::from_utf8_lossy(&out.stderr));
    }
    let ptx = tdir.join("nvptx64-nvidia-cuda/release").join(format!("{ident}.ptx"));
    assert!(ptx.exists(), "fe-kbuild: expected {}", ptx.display());
    let ptx_out = out_dir.join(format!("{ident}.ptx"));
    std::fs::copy(&ptx, &ptx_out).expect("copy ptx");
    assemble(&ptx_out, &cubin);
    cubin
}

/// `ptxas -arch=sm_121a -O3 [FE_PTXAS_FLAGS] in.ptx -o out.cubin` (prints register/spill info as warnings
/// when FE_PTXAS_FLAGS contains -v).
pub fn assemble(ptx: &Path, cubin: &Path) {
    let ptxas = which("ptxas", &["/usr/local/cuda/bin/ptxas"]);
    let mut cmd = Command::new(ptxas);
    cmd.arg(format!("-arch={}", arch())).arg("-O3");
    if let Ok(f) = std::env::var("FE_PTXAS_FLAGS") {
        cmd.args(f.split_whitespace());
    }
    let out = cmd.arg(ptx).arg("-o").arg(cubin).output().expect("fe-kbuild: running ptxas");
    if !out.status.success() {
        panic!("fe-kbuild: ptxas failed on {}:\n{}", ptx.display(), String::from_utf8_lossy(&out.stderr));
    }
    for line in String::from_utf8_lossy(&out.stderr).lines() {
        println!("cargo:warning={line}");
    }
}
