# Qwen3.8-Flash-Next on one or two 16 GB cards

Ferrite 0.6.0 adds a second model: **Qwen3.8-Flash-Next**, Qwen's 48-block hybrid MoE (Gated DeltaNet and QSA sparse
attention, hyper-connections, 256 experts a layer, a PLE n-gram table), in the
[ISTA-DASLab GSQ-RCO Coder IQ1_M](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) quant
(58.4 GB of GGUF), on one **RTX 5060 Ti 16 GB** in a machine with **32 GB** of memory, or on two of them. It is a
separate engine in the same server (`fe-server --engine fnext`, crate `fe-fnext`), with its own Rust kernels for the
GGUF quant types, compiled to PTX like the rest.

The model is far larger than a 16 GB card, so:

- the **always-used weights** (attention, GDN, hyper-connection mixers, shared experts, router, output head: ~4.1 GB),
  the K/V cache, the QSA indexer keys and the GDN state are on the card;
- the **routed experts** (23.4 GB) stay in the memory-mapped GGUF; the card holds as many as fit in slots refilled
  as tokens use them, and the experts a token needs that are not on the card are computed **on the CPU** that step;
- the **PLE table** (27.5 GiB) is read a few rows a token from the GGUF (page cache or disk);
- the **MTP draft head** (a separate 1.4 GiB file, below) drafts tokens for speculative decoding.

With two cards they work in **tensor parallel**: each card holds half the attention heads and owns half of every
layer's experts, so twice the expert slots and almost no CPU work.

> **Status: experimental.** One model, built from source only, on Linux. Measured on one machine (below): CPU-side
> numbers on other CPUs, and the single-card preset on a real 32 GB desktop, are not measured yet. Reports welcome.

## Results

Two RTX 5060 Ti 16 GB (PCIe 4.0 x8 each), AMD EPYC 7313 (16 cores), 128 GB DDR4, NVMe. Qwen3.8-Flash-Next
GSQ-RCO IQ1_M, 128K context configured, greedy decoding, thinking off, a short chat prompt and 256 tokens out
(`fnext-run bench` / `mtp`), three warm runs; prompt processing is the second pass over the same prompt length.

| | Two cards (tensor parallel) | One card, 28 GiB memory limit |
|---|---|---|
| Code, with MTP drafting | 135-136 tok/s | 84.0-85.3 tok/s |
| Prose, with MTP drafting | 110-111 tok/s | 66.7-67.5 tok/s |
| Plain decoding, code / prose | 74.5-74.8 / 74.8-75.3 tok/s | 45.8-46.5 / 44.8-45.2 tok/s |
| Prompt processing, 2K / 16K tokens | 1,434 / 2,467 tok/s | 1,341-1,345 / 1,874-1,888 tok/s |
| Prompt processing with the MTP pass, 2K / 16K | 1,336 / 2,414 tok/s | 1,207-1,215 / 1,864-1,883 tok/s |
| Four requests at once, code, total (plain / MTP) | 177-179 / 194-201 tok/s | not measured under the limit |

- Server, two cards, 4 slots (`fe-server`, no thinking): one code request decodes at 125-142 tok/s with the first
  token in 0.19 s; four at once total 181 tok/s on code and 145 on prose.
- Speculative decoding is lossless: greedy output with the MTP draft is the same token stream as without it (four
  reference prompts of 39 to 20,000 tokens, 64 tokens each, every preset), and requests decoded together give the
  same tokens as alone. About 92-93% of the drafted tokens are accepted on code and 79% on prose.
- A 10-minute soak with four clients (chat turns, growing agent transcripts, prompts up to ~10K tokens, sampled and
  greedy): 260 requests, no errors.
- The one-card column runs inside a 28 GiB memory limit (a cgroup, no swap) to stand in for a 32 GB machine; its CPU
  is still the 16-core EPYC, which computes the experts that miss the card. A desktop CPU with fewer cores or less
  memory bandwidth will be slower there.

## Requirements

- One or two **RTX 5060 Ti 16 GB** (other 16 GB RTX 50 cards were not tried; two cards of the same model).
- An **x86_64 CPU with AVX2, FMA, F16C and BMI2** (any recent Intel or AMD desktop or server CPU): the experts that
  miss the card are computed by AVX2 kernels. Flash-Next does not build on ARM hosts such as the DGX Spark.
- **32 GB of memory** (the PLE table and the experts are read through the page cache; more memory keeps more of
  them cached). The two-card numbers above were measured with 128 GB.
- About 60 GB of disk, on an NVMe drive: cold PLE rows and experts are read from it while generating.
- NVIDIA driver 580 or later (tested with 595.99), Linux (tested on Ubuntu 24.04).
- To build: stable Rust 1.99+, the kernels' pinned nightly (`kernels/rust-toolchain.toml`) with `rust-src`,
  `llvm-tools`, `llvm-bitcode-linker` and the `nvptx64-nvidia-cuda` target, and `ptxas` from the CUDA 13 toolkit
  (tested with 13.3), as in [docs/rtx-5080.md](rtx-5080.md).

## Setup

```bash
# build the server with the Flash-Next engine
cargo build --release -p fe-server --features fnext

# download and verify the model files (~60 GB) into $FERRITE_MODELS/Qwen3.8-Flash-Next (default /var/tmp/models)
scripts/fetch-flashnext.sh

# serve: one or two cards, picked from the visible GPUs
tools/serve-fnext.sh
curl -s localhost:8000/v1/models
```

`scripts/fetch-flashnext.sh` downloads, resumes and checks (size and sha256):

| File | From | Notes |
|---|---|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf`, `-00002-of-00002.gguf` | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) `IQ1_M/` | 58.4 GB, used exactly as published |
| `Qwen3.8-Flash-Next-MTP.gguf` | [`BHCC2025/Qwen3.8-Flash-Next-MTP-GGUF`](https://huggingface.co/BHCC2025/Qwen3.8-Flash-Next-MTP-GGUF) | 1.5 GB, the MTP draft head (Qwen Community License 1.0) |
| `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | |
| `chat_template.jinja` | the first GGUF's own chat template | |

`--no-draft` skips the draft; the model then runs without speculative decoding. `--verify-only` checks a folder,
`--dir` puts it elsewhere.

The ISTA files carry no MTP block. The draft file holds only Qwen's MTP head (`blk.48.*`), made from the official
checkpoint; Ferrite loads it beside the ISTA files (found by `mtp` in its name next to them, or `FNEXT_DRAFT=<path>`).
How it was made is in its model card.

## The presets

`tools/serve-fnext.sh` reads the visible cards (`CUDA_VISIBLE_DEVICES`, else all) and the host memory:

| | One card | Two cards |
|---|---|---|
| Engine | single card: experts in LRU slots on the card, misses on the CPU | tensor parallel (`FNEXT_TP=2`): half the heads and half the experts a card |
| Requests at once (`FERRITE_SLOTS`) | 1 below 40 GiB of memory, else 4 | 4 |
| Pinned host buffers (engine, by memory) | below 40 GiB: one prompt-cache snapshot buffer, no prompt staging; 40-48 GiB: four; 48 GiB and up: prewarmed for every slot | same rule |
| Context | 131072 | 131072 |

`FERRITE_GPUS=1` uses one card when two are visible. With more than two cards the first two are used.

| Setting | Default | |
|---|---|---|
| `FNEXT_DIR` | `$FERRITE_MODELS/Qwen3.8-Flash-Next` | the folder from `scripts/fetch-flashnext.sh` |
| `FERRITE_GPUS` | `auto` | `1` or `2` |
| `FERRITE_HOST`, `FERRITE_PORT` | `127.0.0.1`, `8000` | there is no API key: firewall the port before using `0.0.0.0` |
| `FERRITE_MODEL_ID` | `qwen3.8-flash-next` | the id `/v1/models` reports |
| `FERRITE_CTX` | `131072` | context length |
| `FERRITE_SLOTS` | see above | requests at once, 1-8 |
| `FERRITE_DRAFT` | `mtp` when the draft file is there, else `off` | |
| `FNEXT_COUNTS` | `crates/fe-fnext/data/routing-mix.json` | expert routing counts that pick the experts loaded first |
| `FNEXT_CPU_POOL` | the E-cores on a hybrid Intel CPU, else one thread per physical core not taken by the engine's 3 own threads | CPUs that compute the experts that miss the card |

## Limits

- Experimental: measured on one machine, two 5060 Ti cards. The single-card preset was measured on that machine under
  a 28 GiB memory limit, not yet on a 32 GB desktop; CPU-side speed (expert misses, PLE reads) depends on the host.
- Text only: the vision tower (`mmproj`) is not supported.
- The first request after a start reads experts and PLE rows from disk and is slower; a short prompt right after a
  very long one is ~20% slower for a while (its expert slots refill).
- Requests that sample (temperature > 0) decode one token a step; greedy requests use MTP drafting. Several requests
  at once share a step either way.
- Long prompts run alone: other requests wait while one prefills.
- Same weights as published, never requantized: the IQ1_M files are loaded as they are.

## License

Ferrite is Apache-2.0. The model files have their own licenses: Qwen3.8-Flash-Next is under the Qwen Community
License 1.0, which requires a separate license from Qwen before a "Model as a Service" or "AI Work Assistant"
business uses it commercially (see the draft repository's `LICENSE-NOTE.md`); check the ISTA-DASLab repository for
the terms of the IQ1_M files.
