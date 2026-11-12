# Ferrite on the Intel Arc Pro B70

Ferrite also serves Qwen3.8-27B on a single **Intel Arc Pro B70** (32 GB, Xe2 / Battlemage). The engine is Rust from the
HTTP server down to the GPU kernels here too: the kernels are compiled by rustc to LLVM IR, translated to SPIR-V and
loaded through Level Zero, with no OpenCL C in the engine.

> **Status: experimental.** One model, one card, one request at a time by default and up to four with
> `FERRITE_SLOTS=4` (more wait in a queue), built from source only. It was developed and measured on a single machine. Reports from other B70 machines are welcome.

## What you get

- An OpenAI-compatible server (`fe-server --engine xe`): streaming, thinking, tool calls, 128K context.
- **XQ weights**: a mixed 4 / 5 / 6 / 8-bit quant, 5.0 bits per weight on average, read at close to the card's memory
  bandwidth ([model card](model-card-xq.md)).
- **Lossless speculative decoding**: DFlash2 block drafting with verify trees and context copy by default, or the
  model's own MTP head. Greedy output is token for token what plain decoding gives.
- **int8 prompt processing** on the matrix engine (DPAS), about 1.5x the fp16 path.
- **A prefix cache**: the server keeps the last conversation's state with checkpoints at turn boundaries, so a
  follow-up turn computes only what is new. A 16K side pool serves short unrelated requests (titles, summaries)
  without evicting a long conversation.
- **Up to four requests at once** (`FERRITE_SLOTS=4`): each gets a lane with its own conversation state, lanes that
  decode together share one verify window, and text that requests sent together have in common is processed once.

## Results

One Arc Pro B70 with the card capped at 180 W and 2000 MHz, in a desktop with a 6-core CPU and a PCIe 3.0 x16 slot.
Qwen3.8-27B, XQ 5.0 bpw, greedy decoding, fixed prompts, 128K context configured. Tokens per second.

| | Ferrite |
|---|---|
| Code, 2K context | 124 |
| Code, 32K / 57K / 100K context | 111 / 91 / 79 |
| Prose, 2K context | 79 |
| Code edits (rewrite a file it was shown) | 201-244 |
| Plain decoding, no speculation | 27.6 |
| Prompt processing, 2K / 8K / 32K / 100K context | ~2,090 / ~2,240 / ~2,060 / ~1,620 |
| Prompt processing, ~120K context | ~1,490 (first token after ~81 s) |
| KL divergence vs BF16 (code / prose / chat / 16K / 120K context), lower is better | 0.0061 / 0.0114 / 0.0065 / 0.0055 / 0.0037 |
| Device memory at load, 128K context | 28.6 GB |

Speculative decoding speed depends on the text: code and edits are faster than prose. A 30-minute soak with two
clients (698 requests, cancels and over-length prompts included) finished without an error.

## Requirements

- An Intel Arc Pro B70 (32 GB). Other Xe2 cards are untested; the kernels are tuned for this one, and 128K context
  needs the 32 GB.
- Linux with the `xe` kernel driver, the Level Zero loader 1.32 or later (`libze_loader.so.1`) and Intel's compute
  runtime 26.35 or later (it contains the IGC device compiler). Resizable BAR enabled.
- About 4 GB of free host memory and 36 GB of disk for the models.
- To build: stable Rust 1.99+, the kernels' pinned nightly (`nightly-2026-11-04`, in `kernels-xe/rust-toolchain.toml`,
  with `rust-src` and `llvm-tools`), LLVM 23 (`opt`, `llvm-dis`), the SPIR-V translator `llvm-spirv` 23 and
  SPIRV-Tools (`spirv-val`). On Arch Linux: `level-zero-loader intel-compute-runtime llvm spirv-llvm-translator
  spirv-tools rustup`.

## Build

```bash
git clone https://github.com/BHCC2025/ferrite && cd ferrite
rustup toolchain install nightly-2026-11-04 --component rust-src --component llvm-tools
cargo build --release --features xe -p fe-server
```

The first build compiles the kernels to SPIR-V (`kernels-xe/`, several minutes). `--features xe` matters: without it
the server builds but has no Arc engine. No CUDA toolchain is needed.

## Models

```bash
python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install -U huggingface_hub
export FERRITE_MODELS=$HOME/models
scripts/fetch-models.sh xq draft        # about 34 GB, sha256-checked
```

| Role | Repository | License |
|---|---|---|
| Weights: XQ mixed 5.0 bpw, with the MTP draft head | [`BHCC2025/Qwen3.8-27B-XQ-5.0bpw-mix`](model-card-xq.md) | Apache-2.0 |
| Draft model: DFlash2 | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | Apache-2.0 |

### Four requests at once

`FERRITE_SLOTS=4`, same card and settings. N streaming requests sent together that share a prefix of P tokens, 800
tokens each (`tools/bench/lanes.py`, medians of 3 sets). Total tokens per second over the whole set, prompt
processing included, and each lane's decode speed:

| N | P | total tok/s | decode tok/s a lane | prompt tokens sent | computed |
|---|---|---|---|---|---|
| 1 | 2K | 90 | 104 | 2,155 | 2,155 |
| 2 | 2K | 110 | 64 | 4,319 | 4,319 |
| 3 | 2K | 113 | 46 | 6,495 | 6,495 |
| 4 | 2K | 126 | 38 | 8,692 | 8,692 |
| 1 | 16K | 46 | 96 | 16,153 | 16,153 |
| 2 | 16K | 66 | 57 | 32,313 | 20,025 |
| 3 | 16K | 77 | 41 | 48,486 | 23,910 |
| 4 | 16K | 85 | 32 | 64,676 | 27,812 |

Every lane's text is what the same request gives alone (greedy and seeded sampling; the gate of `lanes.py` and a
cancel-one-lane check). The shared part of a prompt is reused at 4096-token boundaries, so a 2K prefix is computed
by every lane and 12K of a 16K prefix is computed once.

## Serve

```bash
scripts/serve-xe.sh                     # http://127.0.0.1:8000/v1, model id "qwen3.8-27b"
curl -s localhost:8000/v1/models
```

The first start quantizes the drafter's linears on the CPU (under a minute) and caches them in the drafter folder;
loading the weights takes under a minute once the files are in the page cache, a few minutes from a cold hard disk. `FERRITE_HOST=0.0.0.0` serves the LAN (there is no API key, so firewall it).
`scripts/serve-xe.sh --help` lists the settings:

| Variable | Default | What it sets |
|---|---|---|
| `FERRITE_MODELS` | `/var/tmp/models` | Folder holding the model folders |
| `FERRITE_XQ_DIR` / `FERRITE_DRAFT_DIR` | under `$FERRITE_MODELS` | Each model folder, if stored elsewhere |
| `FERRITE_MODEL_ID` | `qwen3.8-27b` | Model id the API reports |
| `FERRITE_PORT`, `FERRITE_CTX` | `8000`, `131072` | Port and context length (`81920` with `FERRITE_SLOTS` above 1) |
| `FERRITE_SLOTS` | `1` | Requests served at once, 1 to 4 (see below) |
| `FERRITE_DRAFT` | `dflash` when the drafter is present | `dflash` or `mtp` |
| `FERRITE_PREFILL` | `i8` | `i8` (int8 prompt GEMM) or `f16` (fp16, slower, slightly lower KL) |

### Several requests at once

```bash
FERRITE_SLOTS=4 scripts/serve-xe.sh     # /health reports max_batch_size 4
```

The server then keeps four lanes. Lane 0 has `FERRITE_CTX` positions (81,920 by default in this mode) and the others
`FE_XE_LANE_CTX` (28,672), which together with the weights fill the card (30.6 GB); a request that does not fit a
short lane waits for lane 0. Each lane keeps its own conversation, so a client that comes back to its lane computes
only what is new, and a request whose prompt starts like one that is running or kept in another lane copies that
part instead of computing it (in whole 4096-token prompt windows, which keeps the result identical to a request
sent alone).

Lanes that are decoding share one verify window of up to 16 rows: 8 rows each for two lanes, 5 for three, 4 for
four (`FE_XE_LANE_ROWS`). The weights are read once for all rows, which is where the gain over taking turns comes
from; each lane's drafter pass, state update and cache append run on their own. A request that is alone decodes
exactly as with `FERRITE_SLOTS=1`. Prompt processing takes the whole card: while one request processes its prompt
the other lanes wait, and between two prompts that are waiting the decoding lanes get `FE_XE_QUANTUM` (8) windows.

## How it gets its speed

- **XQ decode at the memory bound.** The matrix-vector kernels decode packed 4 to 8-bit codes straight into the
  product; 1 to 16 rows share one weight read and every row stays bit-identical to single-row decoding, which is
  what makes speculation lossless.
- **Rust to SPIR-V.** rustc has no SPIR-V backend for this, so the kernels are compiled for a carrier GPU target,
  retargeted to `spirv64` in Rust and translated by `llvm-spirv`; Intel's device compiler does the rest at load. The
  route, and what it takes to make Rust kernels match hand-written OpenCL C instruction for instruction, is in
  [kernels-xe/README.md](../kernels-xe/README.md).
- **The matrix engine for prompts and attention.** Prompt GEMMs run as int8 DPAS with per-channel weight scales and
  per-token activation scales; decode attention packs a window's rows into matrix tiles so they share key reads.
- **DFlash2 trees at 8 rows.** A 16-row verify window costs more than it returns for one request on this card, so
  drafts and trees stay at 8 rows (several lanes share 16); the drafter's own linears run as 4-bit XQ on the same kernels.
- **Small-kernel fusions.** The many tiny launches of a window (norms, gates, convolutions) are fused where the
  result stays bit-exact.

The design of the engine is written up in [crates/fe-xe/DESIGN.md](../crates/fe-xe/DESIGN.md), and the drafter port in
[crates/fe-xdf/README.md](../crates/fe-xdf/README.md).

## Limits

- One request at a time by default. With `FERRITE_SLOTS=4` the context is split (81,920 + 3 x 28,672 positions)
  instead of one 128K conversation, and four lanes total about 1.4x one request's speed, not 4x: the drafter still
  runs once a lane.
- Built from source only; there is no release tarball or installer for this target yet.
- The check tools under `crates/fe-kx-tests` and `crates/fe-kxcore` compare the Rust kernels bit for bit with the
  OpenCL kernels of the engine Ferrite's Arc port was developed against. Those reference modules are not in this
  repository, so the tools that need them do not run from a public checkout; the Rust kernels and the engine do not
  depend on them ([NON-RUST.md](../NON-RUST.md)).
- Text only: the vision tower is not quantized or loaded.

## Troubleshooting

- `this fe-server was built without the B70 engine`: rebuild with `--features xe`.
- `FE_DRAFT_DIR is not set`: start through `scripts/serve-xe.sh`, or export it (the DFlash2 folder), or set
  `FE_XE_DRAFT=mtp`.
- No device found: check `ls /dev/dri/renderD*` and that the `xe` driver bound the card. Some firmware leaves the
  card's memory regions unassigned at boot; a PCI remove and rescan of the card's root port fixes that until the next
  boot.
- The server exits at load with an out-of-memory error: lower `FERRITE_CTX` (the KV cache is about 35 KB a token).
