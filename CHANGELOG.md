# Changelog

Notable changes to Ferrite. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [On main since 0.1.0]

Published and available now: clone `main` and build from source. These changes get a version number with the next
tagged release.

### Added

- **Intel Arc Pro B70 engine (experimental).** Ferrite serves Qwen3.8-27B on one Arc Pro B70 (32 GB) with
  `fe-server --engine xe`. The kernels are Rust compiled to SPIR-V and loaded through Level Zero. Built from source
  with `--features xe`; one request at a time. See [docs/arc-b70.md](docs/arc-b70.md).
  - XQ 5.0 bpw weights for decode and int8 prompt processing.
  - Lossless DFlash2 and MTP drafting, a prefix cache and 128K context.
  - New crates `fe-xe`, `fe-xdf`, `fe-xq`, `fe-ze` and `fe-kbuild-xe`, and the kernel sources in `kernels-xe`.
  - `scripts/serve-xe.sh` starts the server.
- **XQ model.** [`BHCC2025/Qwen3.8-27B-XQ-5.0bpw-mix`](https://huggingface.co/BHCC2025/Qwen3.8-27B-XQ-5.0bpw-mix)
  on Hugging Face, described in [docs/model-card-xq.md](docs/model-card-xq.md). `scripts/fetch-models.sh xq` fetches
  it and checks its sha256 sums.
- `fe-server`: engines may report their own timings in a response.
- This changelog.

### Changed

- `fe-server`: streamed responses declare `charset=utf-8`.

## [0.1.0] - 2026-11-09

First release: an LLM inference engine for NVIDIA DGX Spark (GB10), written in Rust down to the GPU kernels, serving
Qwen3.8-27B behind an OpenAI-compatible API.

### Added

- Serving on one Spark, or on two over RoCE (tensor parallel), with 128K context and up to 4 concurrent requests.
- EXL3 3.0 bpw decode and NVFP4/FP8 prompt processing.
- Lossless speculative decoding: DFlash2 block drafting with verify trees, copy rows, and the model's MTP head as a
  fallback.
- A prompt cache for agents: turn snapshots, parked conversations and one shared system prompt
  ([docs/prompt-cache.md](docs/prompt-cache.md)).
- Model [`BHCC2025/Qwen3.8-27B-EXL3-3.0bpw-mix`](https://huggingface.co/BHCC2025/Qwen3.8-27B-EXL3-3.0bpw-mix),
  fetched and sha256-checked by `scripts/fetch-models.sh`.
- One-line install (`scripts/install.sh`), an agent-led install guide ([AGENTS.md](AGENTS.md)) and a prebuilt
  `ferrite-0.1.0-dgx-spark-aarch64.tar.gz` (aarch64, CUDA 13, kernels for `sm_121a`).

[On main since 0.1.0]: https://github.com/BHCC2025/ferrite/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.1.0
