# kernels-xe: Rust GPU kernels for the Intel Arc Pro B70 (Xe2 / Battlemage)

Each `kx-*` crate is one SPIR-V module, loaded by Level Zero (`zeModuleCreate`, IL = SPIR-V; crates/fe-ze). IGC, inside
the driver, compiles the SPIR-V to Xe2 ISA. This is the same role ptxas plays for the CUDA kernels in `kernels/`.

## The route: Rust -> LLVM IR -> SPIR-V (crates/fe-kbuild-xe)

rustc's LLVM (23.1.5, the nightly pinned in rust-toolchain.toml) is built without the SPIR-V backend. So the kernels are
compiled for a GPU target rustc does have, and fe-kbuild-xe then moves the IR over to spirv64.

1. `cargo build --release -p kx-foo` here. `.cargo/config.toml` selects the **amdgcn-amd-amdhsa carrier target**
   (build-std core, `--emit=llvm-bc`, `-C no-prepopulate-passes`, `-Cpanic=immediate-abort`). The fat-LTO bitcode is
   used *unoptimized*: rustc's own pipeline would optimize for amdgcn, and IGC compiles worse from that (the XQ GEMV
   9-16-row kernel spilled 512 B and ran up to 1.45x slower). No AMD code is ever used.
   - amdgcn rather than nvptx, because rustc rejects SIMD vector types in `extern "C"` signatures on nvptx, and the
     DPAS / block-IO builtins take `<8 x i32>` and similar types.
   - On amdgcn every address space is explicit in the IR.
2. The nightly's own `opt` (same LLVM as rustc): instnamer, always-inline (the kxcommon helpers and their markers into
   the kernels), internalize everything but the kernels, globaldce.
3. `fe_kbuild_xe::retarget`, a text pass over the named IR:
   - **Address spaces.** One bijection, applied to every type and to every intrinsic suffix:

     | amdgcn | SPIR-V |
     |---|---|
     | flat 0 | generic 4 |
     | private 5 | 0 |
     | global 1 | 1 |
     | local 3 | 3 |
     | constant 4 | 2 |

   - **Kernels.** `amdgpu_kernel` becomes `spir_kernel`. Pointer parameters become global, with a cast to generic at
     entry.
   - **Builtin calls.** OpenCL builtin calls (`_Z..` mangled names) get their pointer arguments cast to the address
     space their mangling names (`PU3AS1` global, `PU3AS3` local, plain `P` private).
   - **kxcommon markers.** These are replaced:
     - SLM regions become Workgroup globals.
     - `local_arg` turns a `__local` parameter into dynamic SLM.
     - `reqd_work_group_size` and the sub-group size become kernel metadata. The sub-group size defaults to 16.
     - `div_ocl` becomes OpenCL's 2.5-ulp `fdiv`.
4. System `opt` for spirv64 ([`fe_kbuild_xe::pipeline`]), a deliberately **light** pipeline:
   - inline, SROA, instcombine, simplifycfg, full unrolling of constant-trip loops, and `infer-address-spaces`, which
     pushes global / local through the generic pointers;
   - no GVN, no LICM, no partial unrolling: IGC does the scheduling, unrolling and register allocation, as it does for
     ocloc's barely optimized SPIR-V. This beat a full `default<O2>` (XQ GEMV int4 m=1: 557 ISA instructions vs 578,
     OpenCL 559; m = 9..16: 1200 vs 1252, OpenCL 1227). The full unroll is needed: rustc lowers `[x; N]` to a loop,
     and left in, its array stays in private memory (8.7 KB a thread on one XQ kernel, 7x slower);
   - legalization as a **cascade**, each step only if the translator or the validator rejects the previous one:
     (a) nothing; (b) the scalarizer on arithmetic; (c) the scalarizer on loads/stores too. Kernel SPIR-V allows only
     2, 3, 4, 8 or 16 vector components and LLVM makes any width, but scalarizing costs: loop-carried `<8 x float>` DPAS
     accumulators split into scalar phis made attention decode spill 3.7-7 KB (0.55x of OpenCL; 0.96-1.07x without),
     and scalarized 64-byte lane loads cost lb_vl16 a third of its rate. `<ident>.legalize.txt` records the step used.
   - **per variant**, a crate may ask for `default<O2>` (or o1 / o3) in its own Cargo.toml, keyed by the variant's cargo
     features joined by ",", so every build.rs that builds the variant (fe-xe, fe-kxcore, fe-kx-tests) gets the same
     module: `[package.metadata.kbuild.pipeline]` / `dec = "o2"` (kx_core_dec: the pack kernels spill 5.2-5.4 KB on
     light, IGC splitting the float8 accumulators after the key loop, and 0 on o2). FE_KBUILD_PIPELINE=light|o1|o2|o3
     overrides it for every module (A/B). `<ident>.pipeline.txt` records the choice.
5. `llvm-spirv --spirv-fp-contract=off` with the Intel extensions, then `spirv-val`, then the route invariants
   (`check_invariants`: OpenCL memory model; ContractionOff and SubgroupSize on every kernel; no marker or amdgcn names
   left). fe-kx-tests' build.rs also checks 30 golden SPIR-V forms of the probe module kx_test.spv, so a nightly bump
   that changes the bitcode fails the build loudly.

Builtins are called exactly as clang + llvm-spirv call them for OpenCL C. The SPIR-V forms are therefore the ones
ocloc puts in arc-engine's `.spv` files:
- `OpSubgroupBlockReadINTEL`, `OpSubgroupShuffle*INTEL`, `OpGroup*`;
- DPAS and 2D block IO as imported `intel_sub_group_*` functions that IGC resolves by name;
- ExecutionMode `SubgroupSize` / `LocalSize` / `ContractionOff`.

Build one module by hand: `cargo run --release -p fe-kbuild-xe --bin kbuild-xe -- kx-foo [--features b4 --name kx_foo4]`.
The intermediates (`.named.ll`, `.spirv.ll`, `.opt.ll`) land next to the `.spv` in `kernels-xe/target/spv`.

## Writing kernels

- Kernels: `#[unsafe(no_mangle)] pub unsafe extern "gpu-kernel" fn name(p: *const T, .., n: u32)`.
  - Pointer and scalar parameters only.
  - Pointers are `__global`. Make a parameter `__local` with `attr::local_arg(p)`, called first.
  - Attributes (`attr::reqd_work_group_size::<X, Y, Z>()`, `attr::sg_size::<N>()`) are the first calls of the body.
- **Floating point is explicit.**
  - `a * b + c` is always two roundings (Rust semantics, and ContractionOff on every entry point).
  - `math::fma` and `math::mad` fuse.
  - `/` is a plain OpFDiv: IGC gives it OpenCL precision (<= 2.5 ulp, bit-identical to OpenCL C's `/`).
    `math::div_ocl` says so explicitly. For IEEE division, build the module with `-cl-fp32-correctly-rounded-divide-sqrt`.
  - `exp` / `rsqrt` / ... are the OpenCL builtins.
  - To match an OpenCL kernel bit for bit, write fma where it wrote `fma()`, `mad` where it wrote `mad()` or where
    clang contracted (no ContractionOff in its SPIR-V), and `div_ocl` for its `/`.
- **Arrays that must live in registers** need constant indices. Use `kxcommon::unroll(n, |i| ..)`: straight-line
  code for a constant `n` (OpenCL's `#pragma unroll`), rather than relying on LLVM's unroller.
- **Bit casts** between vector types go through `kxcommon::bits` (one `bitcast`). Byte-wise copies (`transmute_copy`)
  make LLVM build i128 / i256 values, which SPIR-V cannot express, and llvm-spirv then fails with "Unsupported vector
  type".
- Panics compile to unreachable code. Release kernels have no overflow or debug checks, as in OpenCL C.
- Every helper in kxcommon is `#[inline(always)]`. Kernels call no non-inlined functions.

## Crates

| crate | module | contents |
|---|---|---|
| kxcommon | (lib) | ids, sub-group ops, SLM + (split) barriers, block IO, DPAS, 2D block IO, conversions, math, unroll |
| kx-test | kx_test.spv | X0 feature proofs (crates/fe-kx-tests `kx-feature-test`) |
| kx-bench | kx_bench.spv | X0 ceilings, kernel for kernel arc-engine's probes (`kx-bench`) |
| kx-xq | kx_xq{3,4,5,6,8}.spv | XQ prepasses + decode GEMV (arc-engine xq.cl, `xq-check`); prompt weight decode xq_pfdec16x_128 / xq_pfdec_G and activation prep xq_pfprep_k16 (xq_pfd.cl / xq_pf.cl, `xqpf-check`) |
| kx-xqpf | kx_xqpf.spv (256 GRF) | prompt GEMMs: fp16 work-group-tiled xq_pfgemm_f16_cCtT, int8 exact xq_pfgemm_i8{,w}_G / i8g_128_cCtT, group-free i8c_cCtT (xq_pfg.cl / xq_pf.cl, `xqpf-check`) |
