# Non-Rust components

Every part of the engine that is not Rust: what it is, where it is used, why it is not Rust, and the plan to replace
it where there is one.

**What is listed.** The parts Ferrite needs to build, links or loads at run time, and the non-Rust reference tools its
own checks compare against. Not listed: the Python clients around the server (the benchmark clients in `tools/bench`,
the API checks in `tools/conformance`, the other helper scripts under `tools/` and `scripts/`) and the `hf` download
tool; none of them is part of the engine.

**Used by.** *CUDA* = the NVIDIA builds (DGX Spark, kernels for `sm_121a`; RTX cards, kernels for `sm_120a`).
*Two nodes* = two DGX Sparks working as one server (TP2). *Intel Arc* = the Arc Pro B70 engine. *Test only* = never
in a server build.

| Component | Used by | Why not Rust (yet) | Replacement plan, if any |
|---|---|---|---|
| `ptxas` (CUDA 13.0 toolchain) | CUDA, build time | NVIDIA's assembler from PTX (the GPU's assembly text) to SASS (its machine code): the only way to get `sm_121a` / `sm_120a` machine code.<br>A build-time tool, not engine code: the kernels themselves are Rust, compiled to PTX by LLVM's NVPTX backend. | None needed. The alternative, letting the driver compile PTX at load time, runs the same NVIDIA compiler. |
| libcuda (driver API) | CUDA, run time | The GPU driver interface. Ferrite binds it directly (crates/fe-cuda, no runtime, no cudarc). | n/a |
| `tools/exl3_dump_ref.py` (Python, torch + exllamav3) | Test only | Test references (exllamav3 reconstruct dumps), never in the engine. | Keep as a test tool. |
| cuBLASLt (`libcublasLt.so`, CUDA 13.0), linked by crates/fe-fp4 (`lt.rs`) | CUDA, the Qwen3.8-27B engine | The measured baseline for the Rust NVFP4 / FP8 prompt GEMMs (`fp4-bench`, and the `fp4-check` cross-check).<br>Also an opt-in fallback (`PromptLinear::forward_lt`, needs `load_opts(.., true)`).<br>The engine path (`PromptLinear::forward`) is the Rust k-fp4 GEMM. | Drop the link once the Rust GEMM is at parity everywhere; it already beats cuBLASLt on the MLP down projection at M >= 4096. Bench-only meanwhile. |
| `vllm` / `transformers` Python (a separate venv) | Test only | fe-server test fixtures: reference renderings of the Hugging Face chat template, never in the engine. | Keep as a test tool. |
| NCCL (`libnccl.so.2`, loaded with dlopen; 2.30 or newer, tested with 2.32.6) in fe-fabric | Two nodes | The baseline collective of the two-node mode, and the fallback for messages above the RoCE path's size cap. | The RoCE one-shot path (a Rust kernel and an ibverbs proxy) already serves decode-sized reductions; a Rust large-message path would replace the rest. |
| libibverbs (rdma-core 50, system library) in fe-fabric/src/roce.rs | Two nodes | The RDMA verbs API (queue pairs, memory registration): for the network card what libcuda is for the GPU.<br>`post_send` / `poll_cq` are called through the context ops table (inline in verbs.h). | None needed (it is the OS interface to the network card). |
| `llvm-spirv` (Khronos SPIRV-LLVM-Translator 23) and the system LLVM 23 `opt`, run by crates/fe-kbuild-xe | Intel Arc, build time | rustc's LLVM is built without the SPIR-V backend. So the Rust kernels' LLVM IR (compiled for the amdgcn target as a carrier, then retargeted to spirv64 in Rust) is optimized for spirv64 and translated to SPIR-V by these two tools.<br>The kernels themselves are Rust. This is the role `ptxas` has for `kernels/`. | None needed. If a rustc nightly ships the SPIR-V target, drop the carrier and the retarget step and keep only `llvm-spirv` (or LLVM's SPIR-V backend). |
| IGC (Intel Graphics Compiler, inside the Level Zero driver, compute-runtime 26.35) | Intel Arc, load time | Compiles the SPIR-V to Xe2 machine code when the module is created (`zeModuleCreate`): the counterpart of `ptxas` / the CUDA driver's load-time compiler.<br>It resolves the `intel_sub_group_*` builtins (DPAS matrix products, 2D block IO) by name. | n/a (the vendor's device compiler) |
| Level Zero loader `libze_loader.so.1` (dlopen, crates/fe-ze) | Intel Arc, run time | The GPU driver interface on Intel Arc, like libcuda for fe-cuda. | n/a |
| Reference OpenCL C kernels as SPIR-V (`*.spv` from arc-engine, an unpublished Zig + OpenCL C engine for the Intel Arc Pro B70; not in this repository).<br>Loaded by the check tools in crates/fe-kx-tests and crates/fe-kxcore, and by crates/fe-xe only with the test feature `ref-spv`. | Test only (Intel Arc) | Bitwise references ("oracles") and timing baselines for the Rust Xe kernels and the host forward.<br>The engine build (`--features xe`) has no path to them. | Keep as test tools. They need the reference modules (`FE_XE_REF_SPV`, `ARC_SPV_DIR`) and are skipped without them. |
