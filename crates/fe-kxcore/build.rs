// kernels-xe/kx-core -> SPIR-V: kx_core.spv (every kernel but the matrix-engine attention) and kx_core_dec.spv (feature
// dec: decode + prompt-window attention, loaded with the 256-GRF flag). The system-opt pipeline of each variant comes from
// kernels-xe/kx-core/Cargo.toml ([package.metadata.kbuild.pipeline]: dec = o2), so every crate that builds them gets the
// same modules.
fn main() {
    fe_kbuild_xe::build("kx-core");
    fe_kbuild_xe::build_variant("kx-core", &["dec"], "kx_core_dec");
}
