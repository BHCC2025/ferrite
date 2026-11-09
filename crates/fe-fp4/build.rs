fn main() {
    fe_kbuild::build("k-fp4");
    // cuBLASLt: the measured baseline / fallback GEMM (listed in NON-RUST.md)
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
}
