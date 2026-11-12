# Changelog

Notable changes to Ferrite, newest release first. Every section below is a tagged release that is published and
available. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-11-12

### Changed

- **Faster prompt MLP on the RTX 5060 Ti 16 GB.** The NVFP4 GEMMs take the mbarrier pipeline on that card (36 SMs):
  the same bits as the default pipeline, the MLP 17% faster. Prompt processing goes from about 1,780 to 1,890
  tokens per second at 16K (128K context configured), 1,420 to 1,470 at 56K and 1,020 to 1,040 at 110K.
  `FE_FP4_KERNEL` picks another variant. The other targets are unchanged.

## [0.5.0] - 2026-11-12

### Added

- **Prompt groups on cards that stream their prompt weights** (the RTX 5080 preset, `tools/serve-5080.sh`). A long
  prompt that is alone on the card runs its chunks layer by layer, so the 16.8 GB of NVFP4/FP8 prompt weights cross
  the bus once a group instead of once a chunk. The replies are the same text as before, the prompt cache's
  snapshots inside a group hold the state they would have without it, and a cancelled request stops it between
  layers. `FE_PROMPT_GROUP=0` turns it
  off. See [docs/rtx-5080.md](docs/rtx-5080.md).
  - RTX 5060 Ti 16 GB (4-lane link, 128K context configured): prompt processing goes from about 790 to 1,780
    tokens per second at 16K, 800 to 1,420 at 56K and 650 to 1,020 at 110K, in the memory that is free anyway. In
    an agent-style soak the median time to first token of a turn went from 4.9 s to 2.6 s.
  - RTX 5080: about 8% at 16K, within 1% on longer prompts (its link already keeps up with the computation).
- **Four requests at once on the RTX 5060 Ti 16 GB** (`SLOTS=4`) checked: the lanes gate passes, 155 tokens per
  second in total with a 2K shared prefix.

The DGX Spark and Arc Pro B70 engines are unchanged.

## [0.4.3] - 2026-11-11

### Changed

- **Decode plans for the RTX 5060 Ti 16 GB.** The EXL3 decode kernels have a plan table tuned on the card (36 SMs);
  until now it ran with the DGX Spark's. Code generation goes from 139 to 155 tokens per second at 2K context
  (121 to 136 at 16K, 86 to 97 at 56K) and 500-token edits from 343 to 357. Replies with drafting on and off are
  the same text. See [docs/rtx-5080.md](docs/rtx-5080.md). The DGX Spark and RTX 5080 engines are unchanged.

## [0.4.2] - 2026-11-11

### Changed

- **Model cards use the model id `qwen3.8-27b`.** The cards of both quants on Hugging Face (and their copies in
  `docs/`) showed the old ids `spark` and `b70` in their serve examples. `scripts/models.manifest` pins the
  re-uploaded revisions; only the card and the checksum list changed there, the weights are the same files.

## [0.4.1] - 2026-11-11

### Changed

- **The default model id is `qwen3.8-27b` on every target.** It was `spark` on the CUDA engine and `b70` on the Intel
  Arc Pro B70 engine, names that only made sense on those machines. `/v1/models`, `/health` and the replies report
  the new id; requests may still name any model, so existing clients keep working. `FERRITE_MODEL_ID` (or
  `--served-model-name`) sets another id, for example `FERRITE_MODEL_ID=spark` to keep the old one.

### Fixed

- **RTX 5080 preset without the NVIDIA checkpoint.** `PROMPT=bf16` (also the fallback when the checkpoint is
  missing) ran out of device memory while loading, because bf16 copies of the prompt weights were built on the
  card. On small-memory devices the prompt now goes through the EXL3 weights on the decode kernels, as
  [docs/rtx-5080.md](docs/rtx-5080.md) describes (`FE_BF16=1` brings the copies back).

### Added

- **RTX 5060 Ti 16 GB results** in [docs/rtx-5080.md](docs/rtx-5080.md): the RTX 5080 build runs on it, with
  measured speeds and the prompt chunk size that suits its 8-lane link.

## [0.4.0] - 2026-11-11

### Added

- **Up to four requests at once on the Intel Arc Pro B70** (`FERRITE_SLOTS=4` with `scripts/serve-xe.sh`). Each
  request gets a lane with its own conversation state; lanes that decode together share one verify window, so the
  weights are read once for all of them (four lanes total about 1.4x the speed of one request), and a prompt that
  starts like another lane's reuses that part in whole prompt windows. Every lane's text is what the same request
  gives alone. `/health` reports `max_batch_size` 4 in this mode. See [docs/arc-b70.md](docs/arc-b70.md).

## [0.3.0] - 2026-11-11

### Added

- **NVIDIA RTX 5080 target (experimental).** The CUDA engine serves Qwen3.8-27B on one RTX 5080 (16 GB) with 128K
  context, built from source for `sm_120a` (`FE_KARCH`). See [docs/rtx-5080.md](docs/rtx-5080.md).
  - A 6-bit K/V cache (`FE_KVQ`, new Rust kernels), NVFP4/FP8 prompt weights streamed from pinned host memory, and
    the embedding table and prompt cache pool in host memory.
  - MTP drafting with copy rows, and `tools/serve-5080.sh` to start it.
- **Up to four requests at once on the RTX 5080** from one shared pool of context (`SLOTS=4`, `FE_POOL`): a request
  that fits a unit runs beside others, a longer one takes the whole pool and runs alone.
- **Shared prefix computed once.** Requests that arrive together and start with the same text (parallel lanes of one
  job, agents with one system prompt) prefill that text once, wherever it ends
  ([docs/prompt-cache.md](docs/prompt-cache.md)).
- `tools/bench/lanes.py`: a benchmark and gate for N requests sharing a prefix.
- A "Supported systems" table in the README: Ferrite runs on Linux only.

### Changed

- Three or more streams decoding at once share 32 verify rows with DFlash2 drafting (48 for one or two), which raises
  their total speed. Set `FE_VERIFY_ROWS` / `FE_VERIFY_GAIN` to fix them.
- `/health`: `max_batch_size` is the number of requests the loaded engine decodes at once (a larger
  `--max-concurrent` is clamped to it), and a test pins the field names (`max_batch_size`, `max_model_len`,
  `running`, `queued`).
- The DGX Spark package is `ferrite-0.3.0-dgx-spark-aarch64.tar.gz`; the one-line installer installs 0.3.0.

### Fixed

- Captured CUDA graphs were never released, so many different batch shapes could fill device memory. They are now
  dropped when memory runs low.
- Prompt attention with the quantized K/V cache: a kernel written during this release read staged data too early.

## [0.2.0] - 2026-11-10

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
- Prebuilt DGX Spark package `ferrite-0.2.0-dgx-spark-aarch64.tar.gz`; the one-line installer now installs 0.2.0.
  The Arc Pro B70 engine is built from source.

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

[0.5.1]: https://github.com/BHCC2025/ferrite/releases/tag/v0.5.1
[0.5.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.5.0
[0.4.3]: https://github.com/BHCC2025/ferrite/releases/tag/v0.4.3
[0.4.2]: https://github.com/BHCC2025/ferrite/releases/tag/v0.4.2
[0.4.1]: https://github.com/BHCC2025/ferrite/releases/tag/v0.4.1
[0.4.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.4.0
[0.3.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.3.0
[0.2.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.2.0
[0.1.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.1.0
