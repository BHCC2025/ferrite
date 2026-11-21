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
Qwen3.8-27B, XQ 5.0 bpw, `scripts/serve-xe.sh` defaults (DFlash2, int8 prompt GEMM, 128K context), greedy
decoding, thinking off. Tokens per second, medians of 3 runs (measured 2026-11-14, Ferrite 0.6.0).

| | Ferrite |
|---|---|
| Code, 2K context | 170 |
| Code, 32K / 57K / 100K context | 136 / 122 / 102 |
| Prose, 2K context (clean replies) | 66 |
| Code edits (rewrite a file it was shown; the four edit tasks) | 176-257 |
| Plain decoding, no speculation | 27.6 |
| Prompt processing, 2K / 8K / 32K / 100K context | ~2,130 / ~2,200 / ~2,030 / ~1,600 |
| Prompt processing, ~120K context | ~1,490 (first token after ~80 s) |
| KL divergence vs BF16 (code / prose / chat / 16K / 120K context), lower is better | 0.0061 / 0.0114 / 0.0065 / 0.0055 / 0.0037 |
| Device memory at load, 128K context | 28.6 GB |

Speculative decoding speed depends on the text: code and edits are faster than prose. A 30-minute soak with two
clients (698 requests, cancels and over-length prompts included) finished without an error.

**Measured with** the card's power and clock lowered from their defaults (230 W cap, 460 W spike limit, 2800 MHz)
to 180 W, 230 W and 2000 MHz. The machine the numbers come from has a 300 W power supply, which shut off on the
card's start-of-kernel power spikes at the defaults. Memory bandwidth, which decode is bound by, is the same at
2000 MHz, while the matrix engine is about 28% slower. On a machine with a stronger supply the defaults are fine
and prompts are faster. These are runtime settings that a reboot undoes; as root:

```bash
h=$(for x in /sys/class/drm/card*/device/hwmon/hwmon*; do [ "$(cat $x/name)" = xe ] && echo $x; done)
echo 180000000 | sudo tee $h/power1_cap                        # sustained power cap, microwatts (180 W)
echo 230000000 | sudo tee $h/power1_crit                       # spike limit (230 W)
echo 2000 | sudo tee ${h%/hwmon/*}/tile0/gt0/freq0/max_freq    # GPU clock ceiling, MHz
```

To reproduce the table, run `python3 tools/bench/sweep.py --url http://127.0.0.1:8000/v1 --out sweep` against
`scripts/serve-xe.sh`. That gives the 2K and 32K code rows, the edit rows (its `edit-4s` case runs the four tasks;
one request at a time here) and the prompt rows. The other rows come from separate runs:
- 57K and 100K code: `QB_SEED=1 python3 tools/bench/quickbench.py --mode code --tokens 500 --context 57000 --runs 2`
  (and `--context 100000`).
- Prose: `--mode prose --tokens 500 --context 2000` with `QB_SEED` 1 to 8, the median of the replies it marks clean.
  The sweep's own one-request prose prompt is echoed on this engine, so the sweep reports no clean prose figure
  (`tools/bench/README.md`).
- Plain decoding: the code command against `FERRITE_PLAIN=1 scripts/serve-xe.sh`.
- The lanes table: `tools/bench/lanes.py` (below).

## Requirements

- An Intel Arc Pro B70 (32 GB). Other Xe2 cards are untested; the kernels are tuned for this one, and 128K context
  needs the 32 GB.
- Linux with the `xe` kernel driver, the Level Zero loader 1.32 or later (`libze_loader.so.1`) and Intel's compute
  runtime 26.35 or later (it contains the IGC device compiler). Resizable BAR enabled.
- About 4 GB of free host memory and 36 GB of disk for the models.
- To build: stable Rust 1.99+, the kernels' pinned nightly (`nightly-2026-11-04`, in `kernels-xe/rust-toolchain.toml`,
  with `rust-src` and `llvm-tools`), LLVM 23 (`opt`, `llvm-dis`), the SPIR-V translator `llvm-spirv` 23 and
  SPIRV-Tools (`spirv-val`). On Arch Linux: `sudo pacman -S level-zero-loader intel-compute-runtime llvm
  spirv-llvm-translator spirv-tools rustup git python`.
- About 3 GB of disk for a build (the toolchains and `target/`; the build takes about 10 minutes on a 6-core CPU).

## Build

```bash
git clone https://github.com/BHCC2025/ferrite && cd ferrite
rustup default stable          # Arch's rustup starts with no default toolchain: without this cargo does not run
rustup toolchain install nightly-2026-11-04 --component rust-src --component llvm-tools
cargo build --release --features xe -p fe-server
```

The first build compiles the kernels to SPIR-V (`kernels-xe/`, several minutes). `--features xe` matters: without it
the server builds but has no Arc engine. No CUDA toolchain is needed.

## Models

```bash
python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install -U huggingface_hub
export FERRITE_MODELS=$HOME/models      # a disk with 36 GB free (the default /var/tmp/models is on the root disk)
scripts/fetch-models.sh xq draft        # about 34 GB, sha256-checked; it stops first if the disk is too small
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
| 1 | 2K | 88 | 101 | 2,156 | 2,156 |
| 2 | 2K | 108 | 63 | 4,321 | 4,321 |
| 3 | 2K | 120 | 49 | 6,492 | 6,492 |
| 4 | 2K | 125 | 38 | 8,692 | 8,692 |
| 1 | 16K | 46 | 93 | 16,151 | 16,151 |
| 2 | 16K | 65 | 56 | 32,309 | 20,021 |
| 3 | 16K | 78 | 43 | 48,483 | 23,907 |
| 4 | 16K | 83 | 32 | 64,664 | 27,800 |

Measured with `python3 tools/bench/lanes.py --lanes 1,2,3,4 --prefix 2000,16000 --runs 3` against
`FERRITE_SLOTS=4 scripts/serve-xe.sh`.

Every lane's text is what the same request gives alone (greedy and seeded sampling; the gate of `lanes.py` and a
cancel-one-lane check). The shared part of a prompt is reused at 4096-token boundaries, so a 2K prefix is computed
by every lane and 12K of a 16K prefix is computed once.

## Serve

```bash
scripts/serve-xe.sh                     # http://<this machine>:8000/v1, model id "qwen3.8-27b"
curl -s localhost:8000/v1/models
```

The server listens on every interface, so other machines on your network can use it, and there is no API key: use
it on a trusted network or firewall the port. `FERRITE_HOST=127.0.0.1` keeps it local. It runs in the foreground and
its process becomes the server: Ctrl-C, `kill <pid>` or, after `scripts/serve-xe.sh &`, `kill $!` stop it and free
the card. `scripts/ferritectl.sh start b70` runs it in the background (`stop`, `status`, `log`).

The first start quantizes the drafter's linears on the CPU (under a minute) and caches them in the drafter folder.
Loading the weights takes under a minute once the files are in the page cache, and a few minutes from a cold hard
disk. **The first request after a new build takes about 30 s**: Intel's compute runtime compiles the kernels for
the card then and caches them, and every later start answers at once. Thinking is on unless a request turns it off
(`"chat_template_kwargs": {"enable_thinking": false}`): with thinking on, a small `max_tokens` can be spent entirely
on `reasoning_content`. Requests may name any model id. `scripts/serve-xe.sh --help` lists the settings:

| Variable | Default | What it sets |
|---|---|---|
| `FERRITE_MODELS` | `/var/tmp/models` | Folder holding the model folders |
| `FERRITE_XQ_DIR` / `FERRITE_DRAFT_DIR` | under `$FERRITE_MODELS` | Each model folder, if stored elsewhere |
| `FERRITE_MODEL_ID` | `qwen3.8-27b` | Model id the API reports |
| `FERRITE_PORT`, `FERRITE_CTX` | `8000`, `131072` | Port and context length (`81920` with `FERRITE_SLOTS` above 1) |
| `FERRITE_SLOTS` | `1` | Requests served at once, 1 to 4 (see below) |
| `FERRITE_DRAFT` | `dflash` when the drafter is present | `dflash` or `mtp` |
| `FERRITE_PREFILL` | `i8` | `i8` (int8 prompt GEMM) or `f16` (fp16, slower, slightly lower KL) |
| `FERRITE_PLAIN` | `0` | `1`: plain decoding, no speculation (one request at a time) |
| `FERRITE_HOST` | `0.0.0.0` | Listen address; `127.0.0.1` keeps the server on this machine |
| `FERRITE_GPU_LOCK` | `/var/tmp/ferrite-gpu.lock` | The card's lock, held while the server runs; a second server stops at once and names the holder (`FERRITE_LOCK_WAIT=1` waits) |

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

## Recommended settings

The same as on the other targets ([model card](model-card.md#recommended-settings)): the base model's published
values, which the server applies to the fields a request leaves out.

| | Thinking | Non-thinking |
|---|---|---|
| temperature | 1.0 | 0.7 |
| top_p | 0.95 | 0.80 |
| top_k | 20 | 20 |
| presence_penalty | 0 | 1.5 |
| repetition_penalty | 1.0 | 1.0 |

A value the client sends always wins, field by field. A request that sends `temperature: 0` gets no penalty added
and stays on the greedy path (the device argmax with drafting). The values a request ran with are returned in the
`x-sampling-effective` response header.

The server reads them from `sampling.json` in the XQ folder. Ferrite's copy for this model is
`crates/fe-server/sampling.qwen3.8-27b-xq.json`; `scripts/serve-xe.sh` uses it when the folder has none, and
`FE_SAMPLING_JSON=<file>` names another file.

The protections are the ones of the other engines, through the same code, with the same switches (the table in the
[model card](model-card.md#recommended-settings)): reasoning capped at 8,192 tokens (`FE_MAX_REASONING_TOKENS`), the
loop breaker and the repeated-sentence rule (`FE_LOOP_BREAKER`, `FE_LOOP_RULES`), never a stop token as the first
token, `tool_choice` `required` limited to the offered names, the no-token timeout, and the non-finite check
(`FE_FINITE_CHECK`). A protection acts at the token where it fires: the rows a verify window accepted past that
token are not kept. On this engine:

- The non-finite check covers the rows the host picks from (sampled requests, the first token when a stop token is
  banned, constrained openings). Rows of a plain greedy request are decided on the device and are not checked.
- The loop breaker reads every token of a chat reply. Where nothing fires the reply is unchanged; on a reply that
  repeats its prompt almost verbatim it costs about 4% of decode speed, and nothing measurable on code.
- With `FERRITE_DRAFT=mtp`, a first token the engine replaces (a banned stop token) leaves the MTP head's first
  drafts made for the model's own choice; they are verified like any draft, so the text is not affected.

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
- Built from source; there is no installer for this target yet.
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
- `serve-xe: the model folders do not match scripts/models.manifest`: a file is missing or cut off (the lines above
  name it). Run `scripts/fetch-models.sh xq draft` again, which fetches just those files.
- `serve-xe: port 8000 is already in use (by ...)`, or `another Ferrite GPU job holds ...`: another server is running;
  it names the process.
- The machine powers off under load: the power supply cannot take the card's spikes. Lower the spike limit and
  clock as in [Results](#results).
