# Credits

Ferrite stands on other people's work. This file lists every project, paper and idea it drew on, where the
result lives in this tree, and what kind of debt it is:

- **ported**: upstream code was read and re-implemented in Rust with its structure kept (the file says so in
  its header);
- **design**: an approach or a set of choices taken over, implemented anew;
- **studied**: upstream code was read for its design or used as a reference to check against; Ferrite's version
  is written separately;
- **idea**: a method from a paper or a project, implemented from the description;
- **format**: a file format, checkpoint layout or API behaviour Ferrite reads, writes or matches;
- **used**: a library, tool or model Ferrite links, loads, or runs in its tests.

No upstream source file is copied into this repository: every kernel is Rust (see NON-RUST.md for what is
not). Licenses and the notices they require are in LICENSE and NOTICE.

## TensorFold

**[TensorFold](https://github.com/ashhart/TensorFold)** (Apache-2.0 from 0.6.0, MIT before; NOTICE reproduces
its notice)

Ferrite grew out of work on TensorFold. TensorFold is the mature engine: a native Zig engine with CUDA and Metal
backends, many model families, its own drafters and serving stack. Ferrite is a narrow experiment built on one of
TensorFold's setups, Qwen3.8-27B as an NVFP4 + EXL3 hybrid on two DGX Sparks, to find out how much of it can be
written in Rust, down to the GPU kernels. Most of what Ferrite does was worked out first, and ran first, in
TensorFold. The project plan had everyone working on Ferrite read TensorFold's kernels as reference material and
port ideas from them: `exl3_linear_wc`, `exl3_decode`, `exl3_prompt`, `gdn*`, `qwen35_attn*`, `prefill_attention`,
`prompt_lt`, `prompt16`, `roce_gather` and `norms*` (all in `zig/kernels/cuda/`). The ones not ported outright
are tagged "studied" below: read as reference while writing Ferrite's own kernels.

| from TensorFold | kind | in Ferrite |
|---|---|---|
| the hybrid: prompt rows through NVIDIA's NVFP4 / FP8 checkpoint, decode and verify rows through the EXL3 quant | design | crates/fe-fp4, crates/fe-model (`FE_PROMPT=fp4`) |
| the TP2 approach: every layer split by heads and MLP features over two ranks, the RoCE one-shot exchange for decode-sized reductions and NCCL for large ones, bf16 decode partials, a vocabulary half of lm_head per rank, the DFlash2 drafter split across the ranks | design | crates/fe-model/src/tp.rs, crates/fe-exl3/src/tp.rs, crates/fe-fabric, kernels/k-core-tp2, kernels/k-gdn-tp2, crates/fe-dflash |
| EXL3 decode linear: `zig/kernels/cuda/exl3_linear.cu`, `exl3_linear_wc.cuh`, `exl3_decode.cuh` | ported | kernels/k-exl3/src/gemv.rs: each lane decodes its 8 trellis values of a tile straight into its mma B fragments, x rows as the A fragment, one program per 128-column block and K split, warps over contiguous K ranges, the last split to arrive finishes the block (output Hadamard, `svh`) and re-arms its counter, programmatic dependent launch. The forms of up to 48 rows a pass that keep accumulator groups in shared memory were added in Ferrite. |
| the EXL3 plan rule (warps and K splits per shape): `zig/src/cuda/exl3.zig` `plan` | ported | crates/fe-exl3/src/linear.rs `tf_plan`, used for every shape src/plans.rs does not tune (the tuner always tries it too) |
| two-rank RoCE one-shot gather: `zig/kernels/cuda/roce_gather.cu`, `zig/src/cuda/roce.zig` (after b12x, below) | ported | kernels/k-fabric, crates/fe-fabric/src/roce.rs: a device-resident epoch, a sequence flag written after each payload on the same queue pair, two slots by sequence parity, a host proxy thread that turns the kernel's doorbell into RDMA writes, a timeout that fails the transport, the rank sum fused into the exchange |
| DFlash2 best-first verify tree: `zig/src/families/qwen3_5/cuda_draft_tree.zig` | ported | crates/fe-dflash/src/lib.rs `grow_tree`: the grow logic and its constants, candidates scored log-softmax((logit + 0.6 x selector edge) / 1.5) over each row's top 16, 4 children per taken node, least path cost next, up to the node budget |
| copy rows: `zig/src/native/qwen27_slots.zig` (copy rows past the drafted tree, adaptive length) and `zig/src/core/lanes/proposer.zig` (`SuffixLookup`: 3-token keys, a minimum match of 5, silence after rejected copies, `confident_match` 24) | ported | crates/fe-model/src/copy.rs and the scheduler's copy rows (`FE_COPY_ROWS`): the idea and its constants. How a copy is grafted onto the tree, the chain mode and the cost tiers are Ferrite's. |
| the drafter's linears quantized at load to affine 4-bit, group 64 ([MLX](https://github.com/ml-explore/mlx)'s scheme); a draft head over a prefix of the vocabulary (`draft_vocab`); 16-row DFlash2 blocks past the 8 the drafter was trained with (as exllamav3 also runs) | design | crates/fe-dflash (`fe_df_gemv4t`, `FE_DF_VOCAB`, `FE_DF_BLOCK`), fe-model's MTP draft head (`FE_DRAFT_VOCAB`) |
| the NVFP4 / FP8 prompt GEMMs and the ModelOpt scale layout: `zig/kernels/cuda/prompt_lt.cu`, `prompt16.cu`, `modelopt_pack.cu` | studied | read as reference while writing crates/fe-fp4 and kernels/k-fp4 (the loader, the quantizers, the GEMMs; cuBLASLt kept as the measured baseline, `fe_fp4::lt`) |
| the GDN kernels: `zig/kernels/cuda/gdn.cu`, `gdn_prefill.cu` | studied | read as reference while writing k-core's recurrence and kernels/k-gdn (whose chunked algorithm comes from the papers below) |
| the attention kernels (`qwen35_attn*.cuh`, `prefill_attention.cu`), the norms (`norms*.cu`), the EXL3 prompt helpers (`exl3_prompt.cuh`) and the Qwen3.5 forward (`zig/src/families/qwen3_5/forward.zig`) | studied | read as reference while writing crates/fe-model and kernels/k-core; the methods are credited below |

**[b12x](https://github.com/local-inference-lab/b12x)** RoCEnante (Apache-2.0; `b12x/comm/roce/`, PRs #295 and
#315; NOTICE carries TensorFold's attribution line for it)

- design, through TensorFold's adaptation: the one-shot two-rank RoCE all-gather protocol (a device-resident
  epoch, sequence flags after each payload, two slots by sequence parity, a timeout that fails the transport),
  in kernels/k-fabric and crates/fe-fabric/src/roce.rs.

## Models (loaded at run time, not shipped)

| model | license | kind | what Ferrite takes |
|---|---|---|---|
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Qwen team, Alibaba Cloud | Apache-2.0 | format, used | The model: its `qwen3_5` architecture (Gated DeltaNet linear attention in 3 of every 4 layers, gated full attention with partial interleaved mRoPE, zero-centred RMSNorm, the MTP layer), tokenizer, chat template and generation config. The BF16 weights are the KL reference and the source of the EXL3 quant. |
| [Qwen3.8-27B-EXL3-3.0bpw-mix](docs/model-card.md) | Apache-2.0 | used | The decode weights: our quant of the model above, made with exllamav3 (see its card). |
| [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), NVIDIA, made with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) 0.48 | Apache-2.0 | format, used | The prompt-row weights (`FE_PROMPT=fp4`): crates/fe-fp4 reads ModelOpt's layout (NVFP4 e2m1 codes with e4m3 block scales, `weight_scale_2`, `input_scale`; FP8 e4m3 per-tensor scales; `hf_quant_config.json`). Its NVFP4 lm_head is also the DFlash2 draft head. |
| [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), a mirror of [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | Apache-2.0 (model card) | format, used | The DFlash2 drafter weights and config (DFlash below). |

## EXL3 and its roots

**[exllamav3](https://github.com/turboderp-org/exllamav3)** (turboderp-org, MIT, Copyright (c) 2025 Turboderp)

- format: the EXL3 tensor format as of exllamav3 1.6: the trellis bitstream, 16x16 tiles stored in tensor-core
  fragment order, the `mul1` codebook, the `suh` / `svh` sign-and-scale vectors and the 128-point block Hadamard
  rotations, integer bit widths (crates/fe-exl3/src/format.rs, cpu.rs; kernels/k-exl3/src/util.rs, recon.rs).
  The decoder is checked bit for bit against exllamav3's `reconstruct` (tools/exl3_dump_ref.py,
  `exl3-refcheck`).
- studied: the EXL3 GEMV / GEMM, dequant, codebook and Hadamard kernels (`exllamav3_ext/quant/`: `exl3_gemv*`,
  `exl3_gemm*`, `exl3_dq.cuh`, `codebook.cuh`, `hadamard*`); Ferrite's decode GEMV came to it through
  TensorFold's port (above). Its DFlash2 drafter (`architecture/`, `modules/arch_specific/`), to cross-check
  crates/fe-dflash's forward.
- used: the quant was made with exllamav3's `sc_optimize.py` and `convert.py` from the per-tensor sensitivity
  measurement published in [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
  (docs/model-card.md). The exllamav3 Python package also generates the test references above (test only).

**[QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235)**
(NeurIPS 2024; [code](https://github.com/Cornell-RelaxML/qtip))

- idea: EXL3 is exllamav3's variant of QTIP's trellis-coded quantization with incoherence processing
  (random-sign Hadamard rotations around the weight). Ferrite only decodes the format; it took no QTIP code.

## Speculative decoding

| source | kind | Ferrite |
|---|---|---|
| [DFlash: Block Diffusion for Flash Speculative Decoding](https://github.com/z-lab/dflash) (ICML 2026; z-lab, MIT, Copyright (c) 2026 Z Lab) and [DFlash 2: Keep Drafting Parallel](https://inco.ai/blog/dflash2/) (Inco AI, 2026) | format, idea | crates/fe-dflash, kernels/k-dflash implement the DFlash2 drafter from the checkpoint's config and model card: 5 Qwen3 layers over the target's hidden states (taps after layers 5, 19, 33, 47, 61, projected by `fc`), a block of [anchor, masks] drafted in one pass with attention non-causal inside the block over a 2048-position window, two-tap grouped dynamic convolutions around every sublayer, and the top-16 candidate selector (predecessor / successor codebooks) that walks one path. The forward was cross-checked against exllamav3's and vLLM's DFlash2 implementations; z-lab's code was not read. |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192); [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) | idea | draft, then verify every draft in one target forward. Greedy streams keep the matching prefix plus the target's next token, so their output equals plain greedy decoding; sampled streams use the speculative-sampling rule for a deterministic draft (keep a draft token with probability p(d), else sample from p without it), which keeps the target's distribution (crates/fe-model/src/sched.rs). |
| [SpecInfer](https://arxiv.org/abs/2305.09781) (token-tree verification); [Medusa](https://arxiv.org/abs/2401.10774) (tree attention) | idea | verify trees: a whole draft tree checked in one forward, each row attending only to its ancestors (an ancestor bitmask, `row_tmask`, in k-core's attention; fe-model `commit_tree`). Ferrite's implementation; the tree's shape comes from TensorFold (above). |
| Multi-token prediction ([arXiv 2404.19737](https://arxiv.org/abs/2404.19737)) as a drafter, in the form of [DeepSeek-V3](https://arxiv.org/abs/2412.19437)'s sequential MTP module | idea | the checkpoint's own MTP layer drafts chained tokens (fe-model `mtp_forward`, `FE_MTP`, `FE_SPEC`). |
| [Prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) | idea | the idea behind copy rows: draft by copying what followed an earlier match in the context (Ferrite's version follows TensorFold's, above). |

## Kernels: methods

| source | kind | Ferrite |
|---|---|---|
| [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) (ICLR 2025) | idea | the gated delta rule as the model defines it: k-core's recurrence (`fe_gdn_rec2`: fp32 state, fused with the gated RMSNorm) and its replay at commit |
| [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) (NeurIPS 2024) | idea | kernels/k-gdn, crates/fe-gdn: the chunkwise WY / UT form for prompt rows (per 32-row chunk a unit lower-triangular solve, T = (I + A)^-1), derived from the math with the gate decays of the paper above; the derivation is in kernels/k-gdn's crate docs. [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) implements the same algorithm; its code was not used. |
| the 3xTF32 split (each fp32 operand as a TF32 high part plus a TF32 low part, three products), as in [CUTLASS](https://github.com/NVIDIA/cutlass)'s 3xTF32 GEMMs | idea | kernels/k-gdn `fe_gdn_chunk_tc`: fp32-accurate state products on TF32 tensor cores |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | idea | kernels/k-core/src/flash.rs: tiled causal prompt attention with online softmax on tensor cores (head dim 256, gated epilogue) |
| [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | idea | kernels/k-core/src/attn.rs: split-K attention over the KV cache with a combine pass |
| [CUTLASS](https://github.com/NVIDIA/cutlass) (NVIDIA, BSD-3-Clause): cute's `mma_sm120.hpp`, as bundled with [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | studied | kernels/k-fp4: the operand and block-scale register mapping of the NVFP4 / FP8 `mma.sync` forms was checked against it |
| NVIDIA PTX ISA (mma.sync fragment layouts, block-scaled `mxf4nvf4` and `f8f6f4` forms, cp.async, ldmatrix) | format | kernels/kcommon/src/mma.rs; kernels/k-fp4's GEMMs and quantizers are written from it, tiled like k-core's bf16 GEMM |

Norms, RoPE, the bf16 GEMM / GEMV, the scheduler (continuous batching, verify-row budgets, arrival batching) and
the tensor-core decode / verify attention (`fe_attn_vk`) are Ferrite's own, built on standard techniques.

## Server behaviour

| source | kind | Ferrite |
|---|---|---|
| [OpenAI API](https://platform.openai.com/docs/api-reference) | format | crates/fe-server: chat and text completions, SSE streaming, `tool_calls`, usage |
| [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) | studied, format | crates/fe-server matches vLLM's behaviour: sampler order and greedy below temperature 1e-5, stop ids banned before `min_tokens`, the qwen3 reasoning parser (`</think>`), the qwen3 XML tool-call parser with schema-typed arguments and its type order, the OpenAI-to-HF message normalization, the incremental detokenizer's window method (shared with [TGI](https://github.com/huggingface/text-generation-inference)). vLLM's `qwen3_dflash.py` was a cross-check for crates/fe-dflash. No vLLM code is included; vLLM also renders reference fixtures for the tests. |
| [Hugging Face transformers](https://github.com/huggingface/transformers) (Apache-2.0) | format, used (tests) | crates/fe-server/src/template.rs renders chat templates byte for byte as `apply_chat_template` (jinja2 `trim_blocks`, `lstrip_blocks`, `tojson`); crates/fe-server/tools/gen_template_refs.py makes the reference renderings |

## Libraries and tools (used)

Rust crates (direct dependencies; the other crates in Cargo.lock are their dependencies, all under permissive
licenses):

| crate | license | for |
|---|---|---|
| [tokenizers](https://github.com/huggingface/tokenizers) (Hugging Face) | Apache-2.0 | `tokenizer.json` |
| [minijinja](https://github.com/mitsuhiko/minijinja), minijinja-contrib (pycompat) | Apache-2.0 | `chat_template.jinja` |
| [axum](https://github.com/tokio-rs/axum), [tokio](https://github.com/tokio-rs/tokio), tokio-stream, [tracing](https://github.com/tokio-rs/tracing), tracing-subscriber, [tower](https://github.com/tower-rs/tower) (tests), [http-body-util](https://github.com/hyperium/http-body) (tests) | MIT | the HTTP server, async runtime, logging |
| [serde](https://github.com/serde-rs/serde), [serde_json](https://github.com/serde-rs/json), [anyhow](https://github.com/dtolnay/anyhow), [clap](https://github.com/clap-rs/clap), [futures](https://github.com/rust-lang/futures-rs), [memmap2](https://github.com/RazrFalcon/memmap2-rs), [libc](https://github.com/rust-lang/libc) | MIT OR Apache-2.0 | JSON, errors, CLI, streams, safetensors mapping, system calls |

Toolchain and system libraries (NON-RUST.md has the reasons):

- [Rust](https://www.rust-lang.org) nightly and its LLVM `nvptx64-nvidia-cuda` target: every GPU kernel.
- NVIDIA CUDA 13.0: `ptxas` (build), the driver API `libcuda` (bound directly in crates/fe-cuda), cuBLASLt
  (baseline and opt-in fallback in crates/fe-fp4 only).
- [NCCL](https://github.com/NVIDIA/nccl) 2.30.4 (dlopen'd): the TP2 collectives above the RoCE path's size.
- [rdma-core](https://github.com/linux-rdma/rdma-core) 50 `libibverbs`: the RoCE queue pairs.
- PyTorch, exllamav3, vLLM and transformers (Python): test references only, never in the engine.
