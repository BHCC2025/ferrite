# Ferrite on the NVIDIA RTX PRO 4000 Blackwell (24 GB)

Ferrite serves **Qwen3.8-27B** on a single **RTX PRO 4000 Blackwell** (24 GB, 70 SMs, compute capability 12.0,
145 W), behind an OpenAI-compatible API on port 8000, with a context of up to 128K tokens. The preset is called
`rtx-pro-4000` and its launcher is `tools/serve-pro4000.sh`.

- **Speed.** Code at 200 tokens/s, prose at 74-79, a code edit at 501, a 2K prompt at 2,712 tokens/s (a 110K prompt
  at 1,490), and the first token of a short prompt after 0.70 s ([Results](#results)).
- **What you need.** The card, NVIDIA driver 580 or later, about 40 GB of free host memory, about 34 GB of disk for
  the models, Linux ([Requirements](#requirements)).
- **How you get it.** Built from source: [Run it](#run-it) has the commands.
- **Why a preset of its own.** The card also runs the 16 GB preset of the RTX 5080 and RTX 5060 Ti
  ([rtx-5080.md](rtx-5080.md)). This preset is that arrangement (EXL3 3.0 bpw decode weights and a 6-bit K/V cache
  on the card, the NVFP4/FP8 prompt weights in pinned host memory, MTP drafting with copy rows) with the extra 8 GB
  spent where the card's time account says they pay: the first token comes sooner, prompts run faster and four
  requests at once get more context. [rtx-5080.md](rtx-5080.md) describes the shared arrangement in detail; what it
  says applies here unless this page says otherwise.

The terms used on this page (K/V cache, copy rows, MTP, slots, `sm_120a`) are explained in
[Words used in these docs](../README.md#words-used-in-these-docs).

> **Status: experimental.** One model, one card, built from source, on Linux. Developed and measured on a single
> machine (a two-socket server, the card in a PCIe 4.0 x8 slot, two other cards at work beside it).

## Requirements

| | |
|---|---|
| Card | One RTX PRO 4000 Blackwell (24 GB). The default preset uses 22.2 GB of the card's 24.5 with a 128K context. |
| Driver | NVIDIA driver 580 or later (tested with 595.99). |
| PCIe slot | Measured in a PCIe 4.0 x8 slot. The prompt weights that are not kept on the card cross the bus for every prompt. |
| Host memory | As for the 16 GB preset ([rtx-5080.md, Requirements](rtx-5080.md#requirements)): about 40 GB free for the pinned prompt weights (16.8 GB), the embedding table and the prompt cache's pool (8 GB by default). Before loading, the launcher checks that about 30 GB is available with the defaults and stops with the numbers if not. |
| Disk | About 34 GB for the two model folders, and about 3 GB more for a build. Put the models on a disk with room: `FERRITE_MODELS` (the default `/var/tmp/models` is on the root disk). |
| Operating system | Linux, x86_64; tested on Ubuntu 24.04. |
| Build tools | `ptxas` (`cuda-nvcc-13-x`) and cuBLASLt (`libcublas-dev-13-x`) from NVIDIA's CUDA repository, and rustup: [quickstart.md, Clean machine](quickstart.md#clean-machine) has the commands for a fresh Ubuntu. `scripts/check-deps.sh --build --install` installs the Rust toolchains. |

**Cooling and power.** Ferrite keeps the card at full load for as long as requests come in. Under load the card
runs at its 145 W limit (86-88 C in the test machine) and decoding is power-bound
([Where the time goes](#where-the-time-goes-measured-on-the-card)). Cards in adjacent slots heat each other, so
give a machine with two or more cards enough case airflow and watch their temperature under a long run
(`nvidia-smi --query-gpu=temperature.gpu --format=csv -l 5`).

## Run it

### 1. Build

```bash
git clone https://github.com/BHCC2025/ferrite && cd ferrite
FE_KARCH=sm_120a scripts/check-deps.sh --build --install
FE_KARCH=sm_120a cargo build --release -p fe-server --features ferrite -p fe-cli
FE_KARCH=sm_120a scripts/check-deps.sh          # the card, the driver, the linked libraries
target/release/ferrite selftest                 # every kernel family against host math, on the card
```

`FE_KARCH=sm_120a` builds the GPU kernels for this card (`sm_121a` is the DGX Spark). On the machine with the card
it can be left out: the build takes the card `nvidia-smi` reports. A binary built for another card stops before
anything loads, with the rebuild command. More about the build: [rtx-5080.md, Build](rtx-5080.md#1-build).

### 2. Download the models

```bash
python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install -U huggingface_hub
export FERRITE_MODELS=$HOME/models
scripts/fetch-models.sh quant prompt            # about 34 GB, sha256-checked
```

These are the two folders of the 16 GB preset: the EXL3 3.0 bpw decode weights
([`BHCC2025/Qwen3.8-27B-EXL3-3.0bpw-mix`](model-card.md)) and NVIDIA's NVFP4 checkpoint for the prompt weights
([`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)), both Apache-2.0.

### 3. Start the server

```bash
tools/serve-pro4000.sh                          # http://<this machine>:8000/v1, model id "qwen3.8-27b" (FERRITE_MODEL_ID)
```

- `scripts/serve.sh` picks this preset by the card's name and starts the same launcher.
- It runs in the foreground, and its process becomes the server: Ctrl-C or `kill <pid>` stops it.
- The launcher says "loading takes a minute".
- The server listens on every interface, so other machines on your network can use it, and there is no API key: use
  it on a trusted network or firewall the port. `FERRITE_HOST=127.0.0.1` keeps it local.
- `tools/serve-pro4000.sh --help` lists the settings. Everything of `scripts/serve.sh --help` applies.

### 4. Test it

```bash
curl -s localhost:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":200}'
```

Thinking is on by default in the Qwen template, so part of a small `max_tokens` goes to the reasoning. Add
`"chat_template_kwargs": {"enable_thinking": false}` to the request to turn it off. Client and sampling settings:
[model-card.md, Recommended settings](model-card.md#recommended-settings) and [harness.md](harness.md).

### 5. Run it in the background, stop it, read the log

```bash
scripts/ferritectl.sh start rtx-pro-4000        # starts tools/serve-pro4000.sh in the background and waits until /health answers
scripts/ferritectl.sh status                    # running or not, and /health
scripts/ferritectl.sh log                       # the log (log -f follows it)
scripts/ferritectl.sh stop                      # stops it and waits until the card is free
```

`scripts/ferritectl.sh start` without a target also picks this preset by the card's name. The log is
`<ferrite folder>/logs/server-8000.log` (the previous run's is kept as `.log.1`; `FERRITE_LOG_DIR` moves it). A
server started by hand with `tools/serve-pro4000.sh` writes to its terminal.

## Settings

The short names are the same settings as their `FERRITE_*` forms. The preset's defaults:

**Settings you may want to change:**

| Variable | Default | What it sets |
|---|---|---|
| `CTX` / `FERRITE_CTX` | the largest that fits: at most `131072`, or `32768` a request with `SLOTS` above 1 | Context length. `CTX=262144` also starts ([below](#a-262144-token-context)). |
| `SLOTS` / `FERRITE_SLOTS` | `1` | Requests at once (slots), 1-4 ([Several requests at once](#several-requests-at-once)); more wait in a queue. |
| `DRAFT` / `FERRITE_DRAFT` | `mtp` | `mtp` (the MTP head with copy rows) or `off` (plain decoding). The DFlash2 drafter is not part of this preset. |
| `PROMPT` / `FERRITE_PROMPT` | `fp4` when the NVIDIA checkpoint is there | `fp4`, or `bf16` (the EXL3 weights on the decode kernels). |
| `RESIDENT` / `FERRITE_RESIDENT` | `auto` | Layers of prompt weights kept on the card, 0-64, or `auto`: what fits beside the model, the K/V cache and `FERRITE_VRAM_WORK_MIB` (1280) of working room ([What the preset does](#what-the-preset-does)). |
| `PARK_GB` | `8` | The prompt cache's pool in host memory. |
| `FERRITE_VRAM_RESERVE_MIB` | `128` (`768` when the card drives a display) | Card memory left free for other programs. |
| `FERRITE_HOST`, `FERRITE_PORT`, `FERRITE_MODEL_ID` | `0.0.0.0`, `8000`, `qwen3.8-27b` | Listen address, port and the model id the API reports. |

**Tuning variables (measured on the card and left at these values:
[What did not pay](#what-did-not-pay-measured-left-at-the-16-gb-presets-values)):**

| Variable | Default | What it sets |
|---|---|---|
| `SPEC`, `COPY` | `8`, `47` (also with `SLOTS` above 1) | MTP drafts a round, copy rows a round. |
| `CHUNK` | `2048` | Prompt rows a forward. |
| `FE_KVQ` | `6` | Bits a K/V value: `6`, `4` or `0` (bf16; no faster on this card, 5.2 GB more at 128K). |
| `FE_PROMPT_RING` | `2` | Streamed layers of prompt weights on the card at once. |
| `FE_EXL3_TABLE` | the card's own decode plans | `gb10` selects the DGX Spark's plans, which gives the numerics of Ferrite 0.7.0 on this card. |

### A 262,144-token context

`CTX=262144` starts: the K/V cache takes 7.1 GB, 13 layers of prompt weights stay resident instead of 27, and 22.2
GB of the card is in use (the same total as the default, with fewer resident layers). A 10-fact needle test over a
240,524-token prompt found 10 of 10; the prompt took 279 s (863 tokens/s) and each further question 3 s from the
prompt cache. It is not the default.

## Several requests at once

`SLOTS=4 tools/serve-pro4000.sh` serves up to four requests at once (slots) from one pool of context.

- The pool is 4 x 32,768 tokens, one unit of 32,768 tokens a slot: the memory of the one-slot 128K context.
- A request that fits a unit runs beside others. A request longer than a unit takes the whole pool and runs alone.
- Each request keeps 47 copy rows (the 16 GB preset drops to 15 for memory). A lone edit runs at 514 tokens/s on the
  four-slot preset, and a lone code request at the one-slot speed.
- Four requests with a shared prefix of 2K / 16K total 255 / 163 tokens/s ([Results](#results)).

## Results

Qwen3.8-27B, EXL3 3.0 bpw mix, the default preset (128K context, one slot), greedy decoding, thinking off, the
repository's bench tools, medians of 3 runs on a warm card (86-88 C at its 145 W limit).

The left column is the same card running the 16 GB preset (`tools/serve-5080.sh`), which is how the card ran before
this preset existed. The right column is this preset, which is what you get by default.

Measured with the changes of Ferrite 0.7.2 built on top of 0.7.0 (right column) and with Ferrite 0.7.0 (left
column, except where noted); not re-measured since.

| | 16 GB preset on this card (before) | `rtx-pro-4000` preset (default) |
|---|---|---|
| Code, 2K / 16K / 56K context, tokens/s | 184 / 160 / 137 (56K: with a 64K context configured) | 200 / 171 / 143 |
| Prose, 2K context (clean replies), tokens/s | 66-78 | 74-79 |
| Code edit, 500 tokens, tokens/s | 462 | 501 |
| Plain decoding, no speculation, tokens/s | 36.5 | 40.1 |
| Prompt processing, 2K / 16K / 56K / 110K prompt, tokens/s | 1,599 / 2,714 / 1,965 / 1,423 | 2,712 / 3,029 / 2,112 / 1,490 |
| First token of a short prompt (44 tokens) | 1.19 s | 0.70 s |
| First token of a 2,118-token prompt (the edit) | 1.23 s | 0.79 s |
| Four requests at once, 2K / 16K shared prefix: total tokens/s | 186 / 149 (measured with Ferrite 0.6.0) | 255 / 163 |
| Device memory, 128K context | 15.5 GB | 22.2 GB of 24.5 GB |

- Four requests at once: `tools/bench/lanes.py --lanes 4 --prefix 2000,16000` against `SLOTS=4`, 800 tokens a
  request. "Total tokens/s" is every request's generated tokens over the wall time of the set, prompt processing
  included (medians of 3 sets). Four code streams decode at 324 tokens/s together.
- "Clean replies": the benchmark's prose prompt tends to make a greedy reply echo the prompt, and the prose figures
  count only the replies the benchmark marks clean (seeds 2, 3 and 6 here; `tools/bench/README.md`).
- Speculative decoding is lossless: with drafting off the server gives the same text (ten requests: code, prose, a
  16K prompt, four file edits, a two-turn conversation with thinking, a tool call). The same ten requests give the
  same text with one slot and with the four-slot preset.
- A 5-fact needle test (five facts placed in a long prompt and asked for afterwards) over a 120K-token prompt found
  5 of 5.

## Limits

- One request at a time by default. With `SLOTS` 2 to 4 each request gets a unit of 32,768 tokens from the shared
  pool ([Several requests at once](#several-requests-at-once)).
- The decode plans change numerics on 70-SM cards: a greedy reply can differ from a Ferrite 0.7.0 server's on this
  card at tokens whose two best candidates are nearly tied ([What the preset does](#what-the-preset-does)).
- The DFlash2 drafter is not part of this preset, and the two-node mode is DGX Spark only.
- The limits of the 16 GB preset that do not depend on the card's memory apply here too
  ([rtx-5080.md, Limits](rtx-5080.md#limits)).

## Troubleshooting

The launcher checks the binary, the port, the GPU lock, the model files and the memory before it loads the model,
and stops with a message that names the problem. The messages and what to do about each are the 16 GB preset's:
[rtx-5080.md, Troubleshooting](rtx-5080.md#troubleshooting). Particular to this preset:

- `FERRITE_DRAFT=dflash: the DFlash2 drafter is not part of this preset`: use `mtp` (the default) or `off`.
- `SLOTS=N: this preset serves 1-4`: set `SLOTS` to 1, 2, 3 or 4.
- A reply is empty or cut short with a small `max_tokens`: thinking is on by default ([Test it](#4-test-it)).

## What the preset does

- **Resident layers of prompt weights** (`FERRITE_RESIDENT`, default `auto`). The first token of a short prompt
  comes after 0.70 s instead of 1.19 s, for 6.7 GB more of the card's memory. On a 16 GB card every prompt of more
  than 16 tokens waits for one pass of the 16.8 GB of prompt weights over the bus: 1.13 s of the 1.19 s to the first
  token here (PCIe 4.0 x8; with the copies skipped the 44-token prompt takes 0.06 s). The preset keeps as many
  layers on the card as fit beside the model and the K/V cache, spread evenly over the stack so the bus works on a
  streamed layer while its resident neighbours compute: 27 of 64 with a 128K context. Each layer takes about 250 MiB
  and 18 ms off the pass. The bits do not change.

  | Resident layers | 0 | 16 | 24 | 27 (128K default) | 30 |
  |---|---|---|---|---|---|
  | First token, 44-token prompt | 1.19 s | 0.91 s | 0.76 s | 0.70 s | 0.64 s |
  | First token, 2K prompt | 1.21 s | 0.93 s | 0.78 s | 0.72 s | 0.66 s |
  | Device memory | 15.5 GB | 19.3 GB | 21.4 GB | 22.2 GB | 23.0 GB |

  A shorter context leaves room for more (`CTX=65536`: 34 layers by the same rule), and `FERRITE_RESIDENT=30` fits at
  128K with 1.5 GB left free. `FERRITE_VRAM_WORK_MIB` (1280) is the room `auto` keeps for prompt groups and graphs.
- **Decode plans for 70 SMs** (`crates/fe-exl3/src/plans.rs`, from `exl3-tune` on the card). A card without a table
  of its own uses the DGX Spark's.
  - With the card's own plans the tuner's sum over the model's GEMVs is 22.6 ms a token (28.1 with the untuned
    rule), and plain decoding goes from 36.5 to 40.3 tokens/s.
  - **This changes numerics on 70-SM cards**: the plans fix the order of the sums. KL between the two on the card is
    0.0004 (code), 0.0010 (text) and 0.0005 (16K context); KL against BF16 stays 0.021 / 0.043 / 0.023.
  - Greedy replies can differ from a 0.7.0 server's on this card at tokens whose two best candidates are nearly
    tied (4 of the 10 gate requests did). `FE_EXL3_TABLE=gb10` gives the old numerics.
- **Prompt kernels for this card.** The NVFP4 GEMMs take the mbarrier pipeline and prompt groups run the next
  chunk's GDN projection beside the recurrence (its 48 head blocks leave 22 of the 70 SMs idle); the chunk is 2048
  rows (2240 on the 16 GB preset). 16K / 56K / 110K prompts run 12 / 7 / 5% faster than with the 16 GB preset's
  settings, the same bits.
- **Several requests at once.** `SLOTS=4` shares one pool of 4 x 32,768 tokens, the memory of the one-slot 128K
  context, and keeps 47 copy rows a request ([above](#several-requests-at-once)).

## Where the time goes (measured on the card)

Engineering notes: none of this is needed to run the server.

**A decoding step** (one verify forward for the drafted rows; `FE_SCHED_PROF=2`, measured with Ferrite 0.7.0,
before the card's decode plans):

| | steps for 500 tokens | tokens a step | draft | forward | commit + rest | a step |
|---|---|---|---|---|---|---|
| Code, 2K | 66 | 7.6 | 5.3 ms | 33.8 ms | 2.4 ms | 41.5 ms |
| Prose, 2K | 225 | 2.2 | 1.7 ms | 28.1 ms | 1.7 ms | 31.5 ms |
| Edit | 18 | 27.8 | 1.5 ms | 50.8 ms | 3.3 ms | 55.5 ms |

The forward takes 28 ms for up to 4 rows, 30 for 5-8, 34 for 9-16, 46 for 17-32 and 66 for 33-48; a draft token
costs 0.95 ms. Of a plain token's 27 ms (also before the plans), 84% is the EXL3 GEMVs of the 64 layers (the MLP
alone 53%) and 5% the output head and the embedding; the norms, the GDN recurrence and attention share the rest. At
57K context attention adds 9 ms. The card runs these at its 145 W limit: `nvidia-smi` reports the power cap
active in 19 of 20 samples under load and an SM clock of about 1,840 MHz of 3,090. Decoding is power-bound.

**A prompt.** One 2048-row chunk computes in about 0.5 s (MLP 41%, the GDN layers 44%, attention 12%); at 16K the
chunks average 0.58 s (MLP 35%, GDN 37%, attention 25%) and at 56K attention is 43%. Prompts of 8K and more are
bound by this computation, shorter ones by the bus.

## What did not pay (measured, left at the 16 GB preset's values)

Engineering notes: settings that were tried on the card and are not worth changing. Each line gives its own
baseline from the same tuning run; these are not the figures of the Results table.

- A bf16 K/V cache (`FE_KVQ=0`): code at 2K / 16K / 56K 180 / 157 / 134 tokens/s against 182 / 161 / 137 with 6 bits,
  for 5.2 GB more at 128K. The 6-bit cache only saves memory.
- A wider or narrower draft head (`FE_DRAFT_VOCAB` 32,768 / 131,072 / 248,192): code 168 / 187 / 167 tokens/s against
  201 with 65,536.
- Draft depth (`SPEC` 6 / 10 / 12 / 16): 191 / 201 / 199 / 199 on code at 2K against 201 with 8. `SPEC=10` is 8% faster
  on the 16K code benchmark and 1% slower on edits and at 56K (3 fresh starts each).
- Copy rows (`COPY` 31 / 63, `FE_COPY_AGREE` 2 / 4): within 3% of 47 / 3 either way.
- A larger ring (`FE_PROMPT_RING` 4, 8, 16): no change.

## About the "before" column's four-request figure

The left column's four-requests figure (186 / 149) was measured with Ferrite 0.6.0. There, in one set of three or
four the requests did not share their prefix. That was fixed in Ferrite 0.7.2: a worker woken from idle now waits
for requests the server is still tokenizing ([CHANGELOG.md](../CHANGELOG.md), 0.7.2, "Requests sent together share
their prefix").
