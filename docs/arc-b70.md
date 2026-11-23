# Ferrite on the Intel Arc Pro B70

Ferrite serves **Qwen3.8-27B** on a single **Intel Arc Pro B70** (32 GB, Xe2 / Battlemage) behind an
OpenAI-compatible API, with a 128K context.

- **Speed:** about 170 tokens/s on code and 66 on prose at a 2K context, with the card limited to 180 W
  ([Results](#results)).
- **Requests at once:** one by default, up to four with `FERRITE_SLOTS=4`.
- **You need:** the card, Linux with the `xe` kernel driver (tested on Arch Linux), about 36 GB of disk for the models
  and about 4 GB of free host memory ([Requirements](#requirements)).
- **How you get it:** built from source; there is no installer for this target yet.

The terms used on this page are explained in the README:
[Words used in these docs](../README.md#words-used-in-these-docs).

> **Status: experimental.** One model and one card, built from source only. By default one request runs at a time;
> `FERRITE_SLOTS=4` serves up to four, and more wait in a queue. It was developed and measured on a single machine.
> Reports from other B70 machines are welcome.

## What you get

- An OpenAI-compatible server (`fe-server --engine xe`): streaming, thinking, tool calls, 128K context.
- **XQ weights**: a mixed 4 / 5 / 6 / 8-bit quant, 5.0 bits per weight on average, read at close to the card's memory
  bandwidth ([model card](model-card-xq.md)).
- **Lossless speculative decoding**: DFlash2 block drafting with verify trees and copy rows (context copy) by default,
  or the model's own MTP head. Greedy output is token for token what plain decoding gives.
- **int8 prompt processing** on the card's matrix engine (DPAS), about 1.5x the fp16 path.
- **A prompt cache** (a prefix cache): the server keeps the last conversation's state with checkpoints at turn
  boundaries, so a follow-up turn computes only what is new. A 16K side pool serves short unrelated requests (titles,
  summaries) without evicting a long conversation.
- **Up to four requests at once (slots)** with `FERRITE_SLOTS=4`. Each request gets a lane (a lane is one request's
  slot) with its own conversation state, lanes that decode together share one verify window, and text that requests
  sent together have in common is processed once.

## Requirements

- An Intel Arc Pro B70 (32 GB). Other Xe2 cards are untested; the kernels are tuned for this one, and 128K context
  needs the 32 GB.
- Linux with the `xe` kernel driver, the Level Zero loader 1.32 or later (`libze_loader.so.1`) and Intel's compute
  runtime 26.35 or later (it contains the IGC device compiler). Resizable BAR enabled.
- About 4 GB of free host memory and 36 GB of disk for the models (the download is about 34 GB).
- To build: stable Rust 1.99+, the kernels' pinned nightly (`nightly-2026-11-04`, in `kernels-xe/rust-toolchain.toml`,
  with `rust-src` and `llvm-tools`), LLVM 23 (`opt`, `llvm-dis`), the SPIR-V translator `llvm-spirv` 23 and
  SPIRV-Tools (`spirv-val`).
- About 3 GB of disk for a build (the toolchains and `target/`; the build takes about 10 minutes on a 6-core CPU).
- No CUDA toolchain is needed.

On Arch Linux, the only distribution this was tested on, one command installs all of it:

```bash
sudo pacman -S level-zero-loader intel-compute-runtime llvm spirv-llvm-translator spirv-tools rustup git python
```

On another distribution, install the same components at the versions above.

## Build

```bash
git clone https://github.com/BHCC2025/ferrite && cd ferrite
rustup default stable          # Arch's rustup starts with no default toolchain: without this cargo does not run
rustup toolchain install nightly-2026-11-04 --component rust-src --component llvm-tools
cargo build --release --features xe -p fe-server
```

The first build compiles the kernels to SPIR-V (`kernels-xe/`, several minutes). `--features xe` matters: without it
the server builds but has no Arc engine.

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

## Serve

Start it, in the same shell that has `FERRITE_MODELS` set:

```bash
scripts/serve-xe.sh                     # http://<this machine>:8000/v1, model id "qwen3.8-27b"
```

Test it from another terminal:

```bash
curl -s localhost:8000/v1/models

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
  "max_tokens": 300,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

- **Stop.** The launcher runs in the foreground and its process becomes the server: Ctrl-C, `kill <pid>` or, after
  `scripts/serve-xe.sh &`, `kill $!` stop it and free the card.
- **Background.** `scripts/ferritectl.sh start b70` runs it in the background; `scripts/ferritectl.sh stop`, `status`
  and `log` (`log -f` follows it) manage it.
- **Logs.** A foreground server logs to its terminal. A background one logs to `logs/server-8000.log` in the Ferrite
  folder (the previous run's log is kept as `.log.1`).
- **Network.** The server listens on every interface, so other machines on your network can use it, and there is no
  API key: use it on a trusted network or firewall the port. `FERRITE_HOST=127.0.0.1` keeps it local.
- **Model id.** Requests may name any model id; the server always runs this model.

What to expect at the start:

- The first start quantizes the drafter's linears on the CPU (under a minute) and caches them in the drafter folder.
- Loading the weights takes under a minute once the files are in the page cache, and a few minutes from a cold hard
  disk.
- **The first request after a new build takes about 30 s.** Intel's compute runtime compiles the kernels for the card
  then and caches them; every later start answers at once.
- **Thinking is on** unless a request turns it off (`"chat_template_kwargs": {"enable_thinking": false}`, as in the
  test above). With thinking on, a small `max_tokens` can be spent entirely on `reasoning_content`.

## Settings

`scripts/serve-xe.sh --help` lists the settings:

| Variable | Default | What it sets |
|---|---|---|
| `FERRITE_MODELS` | `/var/tmp/models` | Folder holding the model folders |
| `FERRITE_XQ_DIR` / `FERRITE_DRAFT_DIR` | under `$FERRITE_MODELS` | Each model folder, if stored elsewhere |
| `FERRITE_MODEL_ID` | `qwen3.8-27b` | Model id the API reports |
| `FERRITE_PORT`, `FERRITE_CTX` | `8000`, `131072` | Port and context length (`81920` with `FERRITE_SLOTS` above 1) |
| `FERRITE_SLOTS` | `1` | Requests served at once, 1 to 4 ([Several requests at once](#several-requests-at-once)) |
| `FERRITE_DRAFT` | `dflash` when the drafter is present | `dflash` or `mtp` |
| `FERRITE_PREFILL` | `i8` | `i8` (int8 prompt GEMM) or `f16` (fp16, slower, slightly lower KL) |
| `FERRITE_PLAIN` | `0` | `1`: plain decoding, no speculation (one request at a time) |
| `FERRITE_HOST` | `0.0.0.0` | Listen address; `127.0.0.1` keeps the server on this machine |
| `FERRITE_GPU_LOCK` | `/var/tmp/ferrite-gpu.lock` | The card's lock, held while the server runs; a second server stops at once and names the holder (`FERRITE_LOCK_WAIT=1` waits) |

## If the machine powers off under load

The card's start-of-kernel power spikes can shut off a small power supply. The machine this page's figures come from
has a 300 W supply, which shut off at the card's default limits, so **every figure on this page was measured with the
card limited**:

| | Card default | Used for this page |
|---|---|---|
| Sustained power cap | 230 W | 180 W |
| Spike limit | 460 W | 230 W |
| GPU clock ceiling | 2800 MHz | 2000 MHz |

- Memory bandwidth, which decode is bound by, is the same at 2000 MHz, while the matrix engine is about 28% slower.
- On a machine with a stronger supply the defaults are fine and prompts are faster.
- These are runtime settings that a reboot undoes: set them again after each boot. They need root (the commands use
  `sudo`):

```bash
h=$(for x in /sys/class/drm/card*/device/hwmon/hwmon*; do [ "$(cat $x/name)" = xe ] && echo $x; done)
echo 180000000 | sudo tee $h/power1_cap                        # sustained power cap, microwatts (180 W)
echo 230000000 | sudo tee $h/power1_crit                       # spike limit (230 W)
echo 2000 | sudo tee ${h%/hwmon/*}/tile0/gt0/freq0/max_freq    # GPU clock ceiling, MHz
```

## Several requests at once

```bash
FERRITE_SLOTS=4 scripts/serve-xe.sh     # /health reports max_batch_size 4
```

What changes with four slots:

- **Context.** The server keeps four lanes. Lane 0 holds `FERRITE_CTX` positions (81,920 by default in this mode) and
  each of the other three holds `FE_XE_LANE_CTX` (28,672). With the weights this fills the card (30.6 GB, against
  28.6 GB at load with one slot). A request that does not fit a short lane waits for lane 0.
- **Conversations.** Each lane keeps its own conversation, so a client that comes back to its lane computes only
  what is new.
- **Shared text.** A request whose prompt starts like one that is running or kept in another lane copies that part
  instead of computing it. It does so in whole 4096-token prompt windows, which keeps the result identical to a
  request sent alone.
- **Prompts take the whole card.** While one request processes its prompt, the other lanes wait.
- **Drafting.** Four slots need the DFlash2 drafter (`FERRITE_DRAFT=dflash`).

Measured with `FERRITE_SLOTS=4`, same card and settings as [Results](#results): streaming requests sent together
that share a prefix, 800 tokens each (`tools/bench/lanes.py`, medians of 3 sets).

- **Total tokens/s** is all requests' tokens over the time of the whole set, prompt processing included.
- **Decode tokens/s per request** is one request's speed after its first token.
- **Prompt tokens computed** is what the server processed after sharing the prefix.

| Requests at once | Shared prefix | Total tokens/s | Decode tokens/s per request | Prompt tokens sent | Prompt tokens computed |
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
`FERRITE_SLOTS=4 scripts/serve-xe.sh`. The job, its prompts and its tasks are those of `lanes.py`, not the code
prompt of [Results](#results), so the one-request rows here are not the code rows there.

Every lane's text is what the same request gives alone (greedy and seeded sampling; the gate of `lanes.py` and a
cancel-one-lane check). The shared part of a prompt is reused at 4096-token boundaries, so a 2K prefix is computed
by every lane and 12K of a 16K prefix is computed once.

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

- A value the client sends always wins, field by field.
- A request that sends `temperature: 0` gets no penalty added and stays on the greedy path (the device argmax with
  drafting).
- The values a request ran with are returned in the `x-sampling-effective` response header.
- **Where the settings come from.** The server reads them from `sampling.json` in the XQ folder. Ferrite's copy for
  this model is `crates/fe-server/sampling.qwen3.8-27b-xq.json`; `scripts/serve-xe.sh` uses it when the folder has
  none, and `FE_SAMPLING_JSON=<file>` names another file.
- **Protections.** The server's protections (a cap on reasoning, a loop breaker, timeouts and others) are the ones of
  the other engines, with the same switches: the table in the [model card](model-card.md#recommended-settings). What
  differs on this engine is under [Developer notes](#developer-notes).

## Results

Test machine: a desktop with a 6-core CPU and a PCIe 3.0 x16 slot, one Arc Pro B70 limited to 180 W and 2000 MHz
([If the machine powers off under load](#if-the-machine-powers-off-under-load)). Qwen3.8-27B, XQ 5.0 bpw,
`scripts/serve-xe.sh` defaults (DFlash2, int8 prompt GEMM, 128K context), greedy decoding, thinking off. Tokens per
second, medians of 3 runs. Measured 2026-11-14 with Ferrite 0.6.0 and not re-measured since.

| | Ferrite on one Arc Pro B70 |
|---|---|
| Code, 2K context | 170 |
| Code, 32K / 57K / 100K context | 136 / 122 / 102 |
| Prose, 2K context (clean replies) | 66 |
| Code edits (rewrite a file it was shown; the four edit tasks) | 176-257 |
| Plain decoding, no speculation | 27.6 |
| Prompt processing, 2K / 8K / 32K / 100K context | ~2,130 / ~2,200 / ~2,030 / ~1,600 |
| Prompt processing, ~120K context | ~1,490 (first token after ~80 s) |
| KL divergence vs BF16 (code / prose / chat / 16K / 120K context), lower is better | 0.0061 / 0.0114 / 0.0065 / 0.0055 / 0.0037 |
| Device memory at load, 128K context | 28.6 GB (30.6 GB with four slots) |

Speculative decoding speed depends on the text: code and edits are faster than prose. A 30-minute soak with two
clients (698 requests, cancels and over-length prompts included) finished without an error.

To reproduce the table, run `python3 tools/bench/sweep.py --url http://127.0.0.1:8000/v1 --out sweep` against
`scripts/serve-xe.sh`. That gives the 2K and 32K code rows, the edit rows (its `edit-4s` case runs the four tasks;
one request at a time here) and the prompt rows. The other rows come from separate runs:
- 57K and 100K code: `QB_SEED=1 python3 tools/bench/quickbench.py --mode code --tokens 500 --context 57000 --runs 2`
  (and `--context 100000`).
- Prose: `--mode prose --tokens 500 --context 2000` with `QB_SEED` 1 to 8, the median of the replies it marks clean.
  The sweep's own one-request prose prompt is echoed on this engine, so the sweep reports no clean prose figure
  (`tools/bench/README.md`).
- Plain decoding: the code command against `FERRITE_PLAIN=1 scripts/serve-xe.sh`.
- The four-requests table: `tools/bench/lanes.py` ([Several requests at once](#several-requests-at-once)).

## Limits

- One request at a time by default. With `FERRITE_SLOTS=4` the context is split (81,920 + 3 x 28,672 positions)
  instead of one 128K conversation, and four requests total about 1.4x one request's speed at a 2K shared prefix and
  about 1.8x at 16K (the table above), not 4x: the drafter still runs once a lane.
- Built from source; there is no installer for this target yet.
- Text only: the vision tower is not quantized or loaded.
- Measured on one machine, with the card limited to 180 W and 2000 MHz.

## Troubleshooting

- `this fe-server was built without the B70 engine`: rebuild with `--features xe`.
- `FE_DRAFT_DIR is not set`: start through `scripts/serve-xe.sh`, or export it (the DFlash2 folder), or set
  `FE_XE_DRAFT=mtp`.
- No device found: check `ls /dev/dri/renderD*` and that the `xe` driver bound the card. Some firmware leaves the
  card's memory regions unassigned at boot; a PCI remove and rescan of the card's root port fixes that until the next
  boot.
- The server exits at load with an out-of-memory error: lower `FERRITE_CTX` (the K/V cache is about 35 KB a token).
- `serve-xe: the model folders do not match scripts/models.manifest`: a file is missing or cut off (the lines above
  name it). Run `scripts/fetch-models.sh xq draft` again, which fetches just those files.
- `serve-xe: port 8000 is already in use (by ...)`, or `another Ferrite GPU job holds ...`: another server is running;
  it names the process.
- `serve-xe: FERRITE_SLOTS=4 needs the DFlash2 drafter`: several requests at once work only with DFlash2 drafting.
  Fetch the drafter (`scripts/fetch-models.sh draft`) and leave `FERRITE_DRAFT` at `dflash`.
- `serve-xe: FERRITE_PLAIN=1 serves one request at a time (FERRITE_SLOTS=1)`: plain decoding has one slot; unset
  `FERRITE_SLOTS`.
- The machine powers off under load: the power supply cannot take the card's spikes. Lower the spike limit and
  clock as in [If the machine powers off under load](#if-the-machine-powers-off-under-load).

## How it gets its speed

- **XQ decode at the memory bound.** The matrix-vector kernels decode packed 4 to 8-bit codes straight into the
  product; 1 to 16 rows share one weight read and every row stays bit-identical to single-row decoding, which is
  what makes speculation lossless.
- **Rust to SPIR-V.** The engine is Rust from the HTTP server down to the GPU kernels here too, with no OpenCL C in
  the engine. rustc has no SPIR-V backend for this, so the kernels are compiled by rustc to LLVM IR for a carrier GPU
  target, retargeted to `spirv64` in Rust, translated to SPIR-V by `llvm-spirv` and loaded through Level Zero; Intel's
  device compiler does the rest at load. The route, and what it takes to make Rust kernels match hand-written OpenCL C
  instruction for instruction, is in [kernels-xe/README.md](../kernels-xe/README.md).
- **The matrix engine for prompts and attention.** Prompt GEMMs run as int8 DPAS with per-channel weight scales and
  per-token activation scales; decode attention packs a window's rows into matrix tiles so they share key reads.
- **DFlash2 trees at 8 rows.** A 16-row verify window costs more than it returns for one request on this card, so
  drafts and trees stay at 8 rows (several lanes share 16); the drafter's own linears run as 4-bit XQ on the same kernels.
- **Small-kernel fusions.** The many tiny launches of a window (norms, gates, convolutions) are fused where the
  result stays bit-exact.

The design of the engine is written up in [crates/fe-xe/DESIGN.md](../crates/fe-xe/DESIGN.md), and the drafter port in
[crates/fe-xdf/README.md](../crates/fe-xdf/README.md).

## Developer notes

**How several lanes share the card.** Lanes that are decoding share one verify window of up to 16 rows: 8 rows each
for two lanes, 5 for three, 4 for four (`FE_XE_LANE_ROWS`). The weights are read once for all rows, which is where
the gain over taking turns comes from; each lane's drafter pass, state update and cache append run on their own. A
request that is alone decodes exactly as with `FERRITE_SLOTS=1`. Between two prompts that are waiting, the decoding
lanes get `FE_XE_QUANTUM` (8) windows.

**Protections on this engine.** They run through the same code as on the other engines: reasoning capped at 8,192
tokens (`FE_MAX_REASONING_TOKENS`), the loop breaker and the repeated-sentence rule (`FE_LOOP_BREAKER`,
`FE_LOOP_RULES`), never a stop token as the first token, `tool_choice` `required` limited to the offered names, the
no-token timeout, and the non-finite check (`FE_FINITE_CHECK`). A protection takes effect at the token where it
fires: the rows a verify window accepted past that token are not kept. On this engine:

- The non-finite check covers the rows the host picks from (sampled requests, the first token when a stop token is
  banned, constrained openings). Rows of a plain greedy request are decided on the device and are not checked.
- The loop breaker reads every token of a chat reply. Where nothing fires the reply is unchanged; on a reply that
  repeats its prompt almost verbatim it costs about 4% of decode speed, and nothing measurable on code.
- With `FERRITE_DRAFT=mtp`, a first token the engine replaces (a banned stop token) leaves the MTP head's first
  drafts made for the model's own choice; they are verified like any draft, so the text is not affected.

**Kernel check tools.** The tools under `crates/fe-kx-tests` and `crates/fe-kxcore` compare the Rust kernels bit for
bit with the OpenCL kernels of the engine Ferrite's Arc port was developed against. Those reference modules are not
in this repository, so the tools that need them do not run from a public checkout; the Rust kernels and the engine
do not depend on them ([NON-RUST.md](../NON-RUST.md)).
