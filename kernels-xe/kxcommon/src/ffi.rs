//! Raw declarations: OpenCL builtins by their SPIR (Itanium) mangled names, exactly as clang emits them for OpenCL C
//! 3.0 (checked with `clang -cc1 -triple spir64 -finclude-default-header`) and as they appear in arc-engine's ocloc
//! `.spv` files; plus the fe-kbuild-xe markers (`__xe_*`). Pointer parameters are written as plain pointers here: the
//! retarget pass casts each to the address space the mangling names (PU3AS1 global, PU3AS3 local, P private).
//! Use the safe(r) wrappers in the other modules.
use crate::v::*;

unsafe extern "C" {
    // ---- fe-kbuild-xe markers ----
    pub fn __xe_slm(bytes: u32, id: u32) -> *mut u8;
    pub fn __xe_fdiv_ocl(a: f32, b: f32) -> f32;
    pub fn __xe_attr_reqd_wg_size(x: u32, y: u32, z: u32);
    pub fn __xe_attr_sg_size(n: u32);
    pub fn __xe_local_arg(p: *mut u8) -> *mut u8;

    // ---- ids ----
    pub fn _Z22get_sub_group_local_idv() -> u32;
    pub fn _Z16get_sub_group_idv() -> u32;
    pub fn _Z18get_num_sub_groupsv() -> u32;
    pub fn _Z18get_sub_group_sizev() -> u32;
    pub fn _Z22get_max_sub_group_sizev() -> u32;
    pub fn _Z12get_group_idj(d: u32) -> u64;
    pub fn _Z12get_local_idj(d: u32) -> u64;
    pub fn _Z13get_global_idj(d: u32) -> u64;
    pub fn _Z14get_local_sizej(d: u32) -> u64;
    pub fn _Z14get_num_groupsj(d: u32) -> u64;
    pub fn _Z15get_global_sizej(d: u32) -> u64;

    // ---- barriers ----
    pub fn _Z7barrierj(flags: u32);
    pub fn _Z17sub_group_barrierj(flags: u32);
    pub fn _Z31intel_work_group_barrier_arrivej(flags: u32);
    pub fn _Z29intel_work_group_barrier_waitj(flags: u32);

    // ---- sub-group shuffles / collectives ----
    pub fn _Z23intel_sub_group_shufflefj(v: f32, lane: u32) -> f32;
    pub fn _Z23intel_sub_group_shufflejj(v: u32, lane: u32) -> u32;
    pub fn _Z23intel_sub_group_shuffleij(v: i32, lane: u32) -> i32;
    pub fn _Z23intel_sub_group_shuffletj(v: u16, lane: u32) -> u16;
    pub fn _Z23intel_sub_group_shuffleDhj(v: f16, lane: u32) -> f16;
    pub fn _Z27intel_sub_group_shuffle_xorfj(v: f32, m: u32) -> f32;
    pub fn _Z27intel_sub_group_shuffle_xorjj(v: u32, m: u32) -> u32;
    pub fn _Z27intel_sub_group_shuffle_xorij(v: i32, m: u32) -> i32;
    pub fn _Z28intel_sub_group_shuffle_downffj(cur: f32, next: f32, d: u32) -> f32;
    pub fn _Z26intel_sub_group_shuffle_upffj(prev: f32, cur: f32, d: u32) -> f32;
    pub fn _Z19sub_group_broadcastfj(v: f32, lane: u32) -> f32;
    pub fn _Z19sub_group_broadcastjj(v: u32, lane: u32) -> u32;
    pub fn _Z20sub_group_reduce_addf(v: f32) -> f32;
    pub fn _Z20sub_group_reduce_addj(v: u32) -> u32;
    pub fn _Z20sub_group_reduce_addi(v: i32) -> i32;
    pub fn _Z20sub_group_reduce_maxf(v: f32) -> f32;
    pub fn _Z20sub_group_reduce_maxj(v: u32) -> u32;
    pub fn _Z20sub_group_reduce_minf(v: f32) -> f32;
    pub fn _Z20sub_group_reduce_minj(v: u32) -> u32;
    pub fn _Z28sub_group_scan_inclusive_addf(v: f32) -> f32;
    pub fn _Z28sub_group_scan_inclusive_addj(v: u32) -> u32;
    pub fn _Z28sub_group_scan_exclusive_addf(v: f32) -> f32;
    pub fn _Z13sub_group_anyi(p: i32) -> i32;
    pub fn _Z13sub_group_alli(p: i32) -> i32;

    // ---- sub-group block reads / writes (global PU3AS1, local PU3AS3) ----
    pub fn _Z26intel_sub_group_block_readPU3AS1Kj(p: *const u32) -> u32;
    pub fn _Z27intel_sub_group_block_read2PU3AS1Kj(p: *const u32) -> U32x2;
    pub fn _Z27intel_sub_group_block_read4PU3AS1Kj(p: *const u32) -> U32x4;
    pub fn _Z27intel_sub_group_block_read8PU3AS1Kj(p: *const u32) -> U32x8;
    pub fn _Z29intel_sub_group_block_read_usPU3AS1Kt(p: *const u16) -> u16;
    pub fn _Z30intel_sub_group_block_read_us2PU3AS1Kt(p: *const u16) -> U16x2;
    pub fn _Z30intel_sub_group_block_read_us4PU3AS1Kt(p: *const u16) -> U16x4;
    pub fn _Z30intel_sub_group_block_read_us8PU3AS1Kt(p: *const u16) -> U16x8;
    pub fn _Z29intel_sub_group_block_read_ucPU3AS1Kh(p: *const u8) -> u8;
    pub fn _Z30intel_sub_group_block_read_uc2PU3AS1Kh(p: *const u8) -> U8x2;
    pub fn _Z30intel_sub_group_block_read_uc4PU3AS1Kh(p: *const u8) -> U8x4;
    pub fn _Z30intel_sub_group_block_read_uc8PU3AS1Kh(p: *const u8) -> U8x8;
    pub fn _Z31intel_sub_group_block_read_uc16PU3AS1Kh(p: *const u8) -> U8x16;
    pub fn _Z26intel_sub_group_block_readPU3AS3Kj(p: *const u32) -> u32;
    pub fn _Z27intel_sub_group_block_read2PU3AS3Kj(p: *const u32) -> U32x2;
    pub fn _Z27intel_sub_group_block_read4PU3AS3Kj(p: *const u32) -> U32x4;
    pub fn _Z27intel_sub_group_block_read8PU3AS3Kj(p: *const u32) -> U32x8;
    pub fn _Z29intel_sub_group_block_read_usPU3AS3Kt(p: *const u16) -> u16;
    pub fn _Z30intel_sub_group_block_read_us8PU3AS3Kt(p: *const u16) -> U16x8;
    pub fn _Z27intel_sub_group_block_writePU3AS1jj(p: *mut u32, v: u32);
    pub fn _Z28intel_sub_group_block_write2PU3AS1jDv2_j(p: *mut u32, v: U32x2);
    pub fn _Z28intel_sub_group_block_write4PU3AS1jDv4_j(p: *mut u32, v: U32x4);
    pub fn _Z28intel_sub_group_block_write8PU3AS1jDv8_j(p: *mut u32, v: U32x8);
    pub fn _Z30intel_sub_group_block_write_usPU3AS1tt(p: *mut u16, v: u16);
    pub fn _Z31intel_sub_group_block_write_us2PU3AS1tDv2_t(p: *mut u16, v: U16x2);
    pub fn _Z31intel_sub_group_block_write_us4PU3AS1tDv4_t(p: *mut u16, v: U16x4);
    pub fn _Z31intel_sub_group_block_write_us8PU3AS1tDv8_t(p: *mut u16, v: U16x8);
    pub fn _Z27intel_sub_group_block_writePU3AS3jj(p: *mut u32, v: u32);
    pub fn _Z28intel_sub_group_block_write8PU3AS3jDv8_j(p: *mut u32, v: U32x8);
    pub fn _Z31intel_sub_group_block_write_us8PU3AS3tDv8_t(p: *mut u16, v: U16x8);
    pub fn _Z8prefetchPU3AS1Kjm(p: *const u32, n: u64);

    // ---- DPAS (cl_intel_subgroup_matrix_multiply_accumulate): a = M rows of A (one short per lane per row),
    //      b = 8 dwords per lane (the K x 16 B tile, VNNI-packed), c = M accumulator rows (lane = column) ----
    pub fn _Z38intel_sub_group_f16_f16_matrix_mad_k16sDv8_if(a: i16, b: I32x8, c: f32) -> f32;
    pub fn _Z38intel_sub_group_f16_f16_matrix_mad_k16Dv2_sDv8_iDv2_f(a: I16x2, b: I32x8, c: F32x2) -> F32x2;
    pub fn _Z38intel_sub_group_f16_f16_matrix_mad_k16Dv4_sDv8_iDv4_f(a: I16x4, b: I32x8, c: F32x4) -> F32x4;
    pub fn _Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(a: I16x8, b: I32x8, c: F32x8) -> F32x8;
    pub fn _Z40intel_sub_group_bf16_bf16_matrix_mad_k16sDv8_if(a: i16, b: I32x8, c: f32) -> f32;
    pub fn _Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv2_sDv8_iDv2_f(a: I16x2, b: I32x8, c: F32x2) -> F32x2;
    pub fn _Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv4_sDv8_iDv4_f(a: I16x4, b: I32x8, c: F32x4) -> F32x4;
    pub fn _Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iDv8_f(a: I16x8, b: I32x8, c: F32x8) -> F32x8;
    pub fn _Z36intel_sub_group_i8_i8_matrix_mad_k32sDv8_ii(a: i16, b: I32x8, c: i32) -> i32;
    pub fn _Z36intel_sub_group_i8_i8_matrix_mad_k32Dv2_sDv8_iDv2_i(a: I16x2, b: I32x8, c: I32x2) -> I32x2;
    pub fn _Z36intel_sub_group_i8_i8_matrix_mad_k32Dv4_sDv8_iDv4_i(a: I16x4, b: I32x8, c: I32x4) -> I32x4;
    pub fn _Z36intel_sub_group_i8_i8_matrix_mad_k32Dv8_sDv8_iS0_(a: I16x8, b: I32x8, c: I32x8) -> I32x8;

    // ---- 2D block IO (cl_intel_subgroup_2d_block_io): (base, width bytes, height rows, pitch bytes, (x elems, y rows), dst) ----
    pub fn _Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u16);
    pub fn _Z41intel_sub_group_2d_block_read_16b_8r16x2cPU3AS1viiiDv2_iPt(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u16);
    pub fn _Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u16);
    pub fn _Z42intel_sub_group_2d_block_read_16b_16r16x2cPU3AS1viiiDv2_iPt(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u16);
    pub fn _Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u16);
    pub fn _Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u16);
    pub fn _Z41intel_sub_group_2d_block_read_8b_16r32x1cPU3AS1viiiDv2_iPt(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u16);
    pub fn _Z41intel_sub_group_2d_block_read_8b_32r32x1cPU3AS1viiiDv2_iPt(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u16);
    pub fn _Z41intel_sub_group_2d_block_read_8b_32r32x2cPU3AS1viiiDv2_iPt(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u16);
    pub fn _Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u32);
    pub fn _Z52intel_sub_group_2d_block_read_transform_16b_16r16x2cPU3AS1viiiDv2_iPj(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u32);
    pub fn _Z52intel_sub_group_2d_block_read_transform_16b_32r16x2cPU3AS1viiiDv2_iPj(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u32);
    pub fn _Z51intel_sub_group_2d_block_read_transform_8b_32r16x1cPU3AS1viiiDv2_iPj(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u32);
    pub fn _Z51intel_sub_group_2d_block_read_transform_8b_32r16x2cPU3AS1viiiDv2_iPj(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u32);
    pub fn _Z51intel_sub_group_2d_block_read_transform_8b_32r16x4cPU3AS1viiiDv2_iPj(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u32);
    pub fn _Z51intel_sub_group_2d_block_read_transpose_32b_16r8x1cPU3AS1viiiDv2_iPj(b: *const u8, w: i32, h: i32, p: i32, c: I32x2, d: *mut u32);
    pub fn _Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(b: *const u8, w: i32, h: i32, p: i32, c: I32x2);
    pub fn _Z46intel_sub_group_2d_block_prefetch_16b_16r16x2cPU3AS1viiiDv2_i(b: *const u8, w: i32, h: i32, p: i32, c: I32x2);
    pub fn _Z46intel_sub_group_2d_block_prefetch_16b_32r16x2cPU3AS1viiiDv2_i(b: *const u8, w: i32, h: i32, p: i32, c: I32x2);
    pub fn _Z45intel_sub_group_2d_block_prefetch_8b_32r32x1cPU3AS1viiiDv2_i(b: *const u8, w: i32, h: i32, p: i32, c: I32x2);
    pub fn _Z45intel_sub_group_2d_block_prefetch_8b_32r32x2cPU3AS1viiiDv2_i(b: *const u8, w: i32, h: i32, p: i32, c: I32x2);
    pub fn _Z42intel_sub_group_2d_block_write_16b_8r16x1cPU3AS1viiiDv2_iPt(b: *mut u8, w: i32, h: i32, p: i32, c: I32x2, s: *mut u16);

    // ---- conversions ----
    pub fn _Z16convert_half_rtef(x: f32) -> f16;
    pub fn _Z16convert_half_rtzf(x: f32) -> f16;
    pub fn _Z32intel_convert_bfloat16_as_ushortf(x: f32) -> u16;
    pub fn _Z31intel_convert_as_bfloat16_floatt(x: u16) -> f32;

    // ---- math (OpenCL precision) ----
    pub fn _Z3madfff(a: f32, b: f32, c: f32) -> f32;
    pub fn _Z3expf(x: f32) -> f32;
    pub fn _Z4exp2f(x: f32) -> f32;
    pub fn _Z3logf(x: f32) -> f32;
    pub fn _Z4log2f(x: f32) -> f32;
    pub fn _Z5rsqrtf(x: f32) -> f32;
    pub fn _Z4sqrtf(x: f32) -> f32;
    pub fn _Z4tanhf(x: f32) -> f32;
    pub fn _Z4fmaxff(a: f32, b: f32) -> f32;
    pub fn _Z4fminff(a: f32, b: f32) -> f32;
    pub fn _Z10native_expf(x: f32) -> f32;
    pub fn _Z11native_exp2f(x: f32) -> f32;
    pub fn _Z11native_log2f(x: f32) -> f32;
    pub fn _Z12native_recipf(x: f32) -> f32;
    pub fn _Z12native_rsqrtf(x: f32) -> f32;
    pub fn _Z11native_sqrtf(x: f32) -> f32;
    pub fn _Z13native_divideff(a: f32, b: f32) -> f32;

    // ---- vloadN / vstoreN (OpenCL.std vloadn / vstoren: one N-element message per lane at element alignment) ----
    pub fn _Z6vload2mPU3AS1Kj(o: u64, p: *const u32) -> U32x2;
    pub fn _Z6vload4mPU3AS1Kj(o: u64, p: *const u32) -> U32x4;
    pub fn _Z6vload8mPU3AS1Kj(o: u64, p: *const u32) -> U32x8;
    pub fn _Z7vload16mPU3AS1Kj(o: u64, p: *const u32) -> U32x16;
    pub fn _Z6vload2mPU3AS1Kf(o: u64, p: *const f32) -> F32x2;
    pub fn _Z6vload4mPU3AS1Kf(o: u64, p: *const f32) -> F32x4;
    pub fn _Z6vload8mPU3AS1Kf(o: u64, p: *const f32) -> F32x8;
    pub fn _Z6vload4mPU3AS1Kt(o: u64, p: *const u16) -> U16x4;
    pub fn _Z6vload8mPU3AS1Kt(o: u64, p: *const u16) -> U16x8;
    pub fn _Z7vload16mPU3AS1Kt(o: u64, p: *const u16) -> U16x16;
    pub fn _Z6vload4mPU3AS3Kj(o: u64, p: *const u32) -> U32x4;
    pub fn _Z6vload8mPU3AS3Kj(o: u64, p: *const u32) -> U32x8;
    pub fn _Z6vload4mPU3AS3Kf(o: u64, p: *const f32) -> F32x4;
    pub fn _Z6vload8mPU3AS3Kf(o: u64, p: *const f32) -> F32x8;
    pub fn _Z6vload8mPU3AS3Kt(o: u64, p: *const u16) -> U16x8;
    pub fn _Z6vload8mPU3AS3Kh(o: u64, p: *const u8) -> U8x8;
    pub fn _Z6vload8mPKt(o: u64, p: *const u16) -> U16x8;
    pub fn _Z6vload8mPKj(o: u64, p: *const u32) -> U32x8;
    pub fn _Z7vstore2Dv2_jmPU3AS1j(v: U32x2, o: u64, p: *mut u32);
    pub fn _Z7vstore4Dv4_jmPU3AS1j(v: U32x4, o: u64, p: *mut u32);
    pub fn _Z7vstore8Dv8_jmPU3AS1j(v: U32x8, o: u64, p: *mut u32);
    pub fn _Z8vstore16Dv16_jmPU3AS1j(v: U32x16, o: u64, p: *mut u32);
    pub fn _Z7vstore2Dv2_fmPU3AS1f(v: F32x2, o: u64, p: *mut f32);
    pub fn _Z7vstore4Dv4_fmPU3AS1f(v: F32x4, o: u64, p: *mut f32);
    pub fn _Z7vstore8Dv8_fmPU3AS1f(v: F32x8, o: u64, p: *mut f32);
    pub fn _Z7vstore4Dv4_tmPU3AS1t(v: U16x4, o: u64, p: *mut u16);
    pub fn _Z7vstore8Dv8_tmPU3AS1t(v: U16x8, o: u64, p: *mut u16);
    pub fn _Z7vstore8Dv8_jmPU3AS3j(v: U32x8, o: u64, p: *mut u32);
    pub fn _Z7vstore8Dv8_fmPU3AS3f(v: F32x8, o: u64, p: *mut f32);
    pub fn _Z7vstore4Dv4_fmPU3AS3f(v: F32x4, o: u64, p: *mut f32);

    // ---- more conversions / math / collectives ----
    pub fn _Z16convert_char_rtef(x: f32) -> i8;
    pub fn _Z20convert_char_sat_rtef(x: f32) -> i8;
    pub fn _Z15convert_int_rtef(x: f32) -> i32;
    pub fn _Z19convert_int_sat_rtef(x: f32) -> i32;
    pub fn _Z20convert_int8_sat_rteDv8_f(x: F32x8) -> I32x8;
    pub fn _Z3dotDv2_fS_(a: F32x2, b: F32x2) -> f32;
    pub fn _Z3dotDv4_fS_(a: F32x4, b: F32x4) -> f32;
    pub fn _Z5log1pf(x: f32) -> f32;
    pub fn _Z9bitselectjjj(a: u32, b: u32, c: u32) -> u32;
    pub fn _Z4rintf(x: f32) -> f32;
    pub fn _Z5truncf(x: f32) -> f32;
    pub fn _Z5floorf(x: f32) -> f32;
    pub fn _Z4ceilf(x: f32) -> f32;
    pub fn _Z20sub_group_reduce_maxm(v: u64) -> u64;
    pub fn _Z20sub_group_reduce_minm(v: u64) -> u64;
    pub fn _Z20sub_group_reduce_addm(v: u64) -> u64;
}
