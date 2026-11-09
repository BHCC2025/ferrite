# Non-Rust components

Every piece of the engine that is not Rust: what, why, and the plan to replace it.

| component | why not Rust (yet) | replacement plan |
|---|---|---|
| `ptxas` (CUDA 13.0 toolchain) | NVIDIA's PTX -> SASS assembler; the only way to get sm_121a machine code. Build-time tool, not engine code (the kernels themselves are Rust -> PTX via LLVM NVPTX). | none needed; the alternative (driver JIT of PTX) runs the same NVIDIA compiler at load time |
| libcuda (driver API) | the GPU driver interface; ferrite binds it directly (crates/fe-cuda, no runtime, no cudarc) | n/a |
| `tools/exl3_dump_ref.py` (python, torch + exllamav3) | test references only (exllamav3 reconstruct dumps), never in the engine | keep as a test tool |
| cuBLASLt (`libcublasLt.so`, CUDA 13.0), linked by crates/fe-fp4 (`lt.rs`) | measured baseline for the Rust NVFP4 / FP8 prompt GEMMs (`fp4-bench`, `fp4-check` cross-check) and an opt-in fallback (`PromptLinear::forward_lt`, needs `load_opts(.., true)`); the engine path (`PromptLinear::forward`) is the Rust k-fp4 GEMM | drop the link once the Rust GEMM is at parity everywhere (it already beats cuBLASLt on mlp down at M >= 4096); bench-only meanwhile |
| `vllm` / `transformers` python (a separate venv) | fe-server test fixtures only: HF chat-template reference renderings, never in the engine | keep as a test tool |
| NCCL 2.30.4 (libnccl.so.2, dlopen) in fe-fabric | TP2 baseline collective and the fallback above the RoCE cap | the RoCE one-shot path (Rust kernel + ibverbs proxy) already serves decode-sized reductions; a Rust large-message path would replace the rest |
| libibverbs (rdma-core 50, system library) in fe-fabric/src/roce.rs | the RDMA verbs API (queue pairs, memory registration) like libcuda for the GPU; post_send / poll_cq called through the context ops table (inline in verbs.h) | none needed (it is the OS interface to the NIC) |
