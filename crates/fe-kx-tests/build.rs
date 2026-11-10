// Rust kernels -> SPIR-V (kernels-xe via fe-kbuild-xe): kx-test, kx-bench, kx-xqpf, and kx-xq once per code width.
fn main() {
    let probe = fe_kbuild_xe::build("kx-test");
    // golden forms of the probe module: what ocloc's SPIR-V of the arc-engine kernels uses, and what IGC must see
    let golden = [
        "OpEntryPoint Kernel % \"t_slm\"",
        "OpExecutionMode % LocalSize 256 1 1",
        "OpExecutionMode % LocalSize 16 1 1",
        "OpExecutionMode % SubgroupSize 16",
        "OpExecutionMode % ContractionOff",
        "OpSubgroupBlockReadINTEL",
        "OpSubgroupBlockWriteINTEL",
        "OpSubgroupShuffleINTEL",
        "OpSubgroupShuffleXorINTEL",
        "OpSubgroupShuffleDownINTEL",
        "OpSubgroupShuffleUpINTEL",
        "OpGroupFAdd % % Reduce",
        "OpGroupFMax % % Reduce",
        "OpGroupBroadcast",
        "OpControlBarrier",
        "OpControlBarrierArriveEXT",
        "OpControlBarrierWaitEXT",
        "OpTypePointer Workgroup",
        "LinkageAttributes \"_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f\" Import",
        "LinkageAttributes \"_Z38intel_sub_group_f16_f16_matrix_mad_k16sDv8_if\" Import",
        "LinkageAttributes \"_Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iDv8_f\" Import",
        "LinkageAttributes \"_Z36intel_sub_group_i8_i8_matrix_mad_k32Dv8_sDv8_iS0_\" Import",
        "LinkageAttributes \"_Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt\" Import",
        "LinkageAttributes \"_Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj\" Import",
        "LinkageAttributes \"_Z42intel_sub_group_2d_block_write_16b_8r16x1cPU3AS1viiiDv2_iPt\" Import",
        "LinkageAttributes \"_Z46intel_sub_group_2d_block_prefetch_16b_16r16x2cPU3AS1viiiDv2_i\" Import",
        "FPRoundingMode RTE",
        "FPMaxErrorDecorationINTEL 2.5",
        "OpExtInst % % fma",
        "OpExtInst % % mad",
    ];
    if let Err(e) = fe_kbuild_xe::golden(&probe, &golden) {
        panic!("{e}");
    }
    fe_kbuild_xe::build("kx-bench");
    fe_kbuild_xe::build("kx-xqpf");
    for b in [3, 4, 5, 6, 8] {
        fe_kbuild_xe::build_variant("kx-xq", &[&format!("b{b}")], &format!("kx_xq{b}"));
    }
}
