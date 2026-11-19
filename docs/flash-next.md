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
GSQ-RCO IQ1_M, 128K context configured, `tools/serve-fnext.sh` with its defaults, thinking off, measured at the
server with `tools/bench/quickbench.py` (commands below): ~2K-token prompts, 256 tokens out, three runs after a
warm-up. Prompt speed is the prompt's tokens over the time to the first token of a fresh prompt.

| | Two cards (tensor parallel) | One card, 28 GiB memory limit |
|---|---|---|
| Code, with MTP drafting | 131-146 tok/s | 93-99 tok/s |
| Prose, with MTP drafting | 127-132 tok/s | 74-88 tok/s |
| Plain decoding (`FERRITE_DRAFT=off`), code / prose | not measured at the server yet | 38-43 / 41-46 tok/s |
| Prompt, 2K / 16K tokens | ~1,200 / 2,350-2,410 tok/s | ~710 / 1,820-1,860 tok/s |
| Four requests at once, short prompts, total (code / prose) | 153-193 / 129-137 tok/s | one at a time (1 slot below 40 GiB) |
| Four requests at once, fresh 2K prompts, total (code / prose) | 88 / 81-96 tok/s, all first tokens by ~6.5 s | |

- The prompt rows were measured before Flash-Next started fixing the C heap's malloc thresholds by default, which
  keeps a long-running server's host memory flat. With that default, processing a fresh prompt is about 8-10%
  slower than the table shows. `FERRITE_MALLOC=0` gives the faster prompts back, at the cost of host memory that
  grows slowly over a long run. Decoding speed is unchanged.
- With fresh prompts, four requests at once total less than the short-prompt row: their prompts are processed one
  after another, and the others' decoding waits meanwhile (see Limits). The short-prompt row is what chat
  follow-ups and agent turns with a cached prefix see.
- Speculative decoding is lossless: greedy output with the MTP draft is the same token stream as without it (four
  reference prompts of 39 to 20,000 tokens, 64 tokens each, every preset), and requests decoded together give the
  same tokens as alone. About 92-93% of the drafted tokens are accepted on code and 79% on prose.
- A 75-minute soak with four clients on two cards (chat turns, growing agent transcripts, prompts up to ~30K
  characters, sampled and greedy): 1,182 requests, no errors.
- The one-card column runs inside a 28 GiB memory limit (a cgroup, no swap, page cache dropped first) to stand in
  for a 32 GB machine; its CPU is still the 16-core EPYC, which computes the experts that miss the card. A desktop
  CPU with fewer cores or less memory bandwidth will be slower there.

To reproduce against a running server (`QB_SEED` fixes the prompts; each command prints the median of its runs):

```bash
export QB_SEED=1
B="python3 tools/bench/quickbench.py --url http://localhost:8000/v1"
$B --tokens 64 --context 2000 --mode code --runs 2              # warm-up
$B --tokens 256 --context 2000 --mode code --runs 3             # and --mode prose
$B --tokens 16 --context 16000 --mode code --runs 3             # 16K prompt
$B --tokens 256 --context 0 --streams 4 --mode code --runs 3    # four at once, short prompts
$B --tokens 256 --context 2000 --streams 4 --mode code --runs 3 # four at once, fresh 2K prompts
```

The one-card memory limit:

```bash
sudo sysctl vm.drop_caches=3   # else the page cache already read stays charged outside the limit
sudo systemd-run --scope -p MemoryMax=28G -p MemorySwapMax=0 --uid=$USER env FERRITE_GPUS=1 tools/serve-fnext.sh
```

## Requirements

- One or two **RTX 5060 Ti 16 GB** (other 16 GB RTX 50 cards were not tried; two cards of the same model).
- An **x86_64 CPU with AVX2, FMA, F16C and BMI2** (any recent Intel or AMD desktop or server CPU): the experts that
  miss the card are computed by AVX2 kernels. Flash-Next does not build on ARM hosts such as the DGX Spark.
- **32 GB of memory** (the PLE table and the experts are read through the page cache; more memory keeps more of
  them cached). The two-card numbers above were measured with 128 GB.
- About 60 GB of disk, on an NVMe drive: cold PLE rows and experts are read from it while generating.
- NVIDIA driver 580 or later (tested with 595.99), Linux (tested on Ubuntu 24.04).
- To build: stable Rust 1.99+, the kernels' pinned nightly (`kernels/rust-toolchain.toml`) with `rust-src`,
  `llvm-tools`, `llvm-bitcode-linker` and the `nvptx64-nvidia-cuda` target, `ptxas` from the CUDA 13 toolkit
  (tested with 13.3; NVIDIA's `cuda-nvcc-13-x` package is enough for Flash-Next) and a C toolchain
  (`build-essential`), as in [docs/rtx-5080.md](rtx-5080.md).
- To download: `curl`, `sha256sum` and `python3` (standard library only); the Hugging Face CLI (`hf`) is used
  when it is installed.
- **Cooling.** Ferrite keeps a card at full load for as long as requests come in. Cards in adjacent slots heat each
  other, so give a machine with two or more cards enough case airflow and watch their temperature under a long run
  (`nvidia-smi --query-gpu=temperature.gpu --format=csv -l 5`). A card that overheats can crash the driver or keep
  the machine from booting until it has cooled.

## Setup

```bash
# build the server (both engines: Flash-Next and the 27B); FE_KARCH=sm_120a compiles the kernels for RTX 50 cards
FE_KARCH=sm_120a cargo build --release -p fe-server --features ferrite,fnext

# download and verify the model files (~60 GB) into $FERRITE_MODELS/Qwen3.8-Flash-Next (default /var/tmp/models)
scripts/fetch-flashnext.sh

# serve: one or two cards, picked from the visible GPUs
tools/serve-fnext.sh
curl -s localhost:8000/v1/models
```

`FE_KARCH` names the GPU the kernels are compiled for. Without it the build asks `nvidia-smi` and uses the card it
reports (a build warning says which); with no card to ask it falls back to `sm_121a`, the DGX Spark. A binary whose
kernels do not match the card stops at once with the command to rebuild, before it loads anything. `--features fnext`
alone builds only Flash-Next; building one engine alone later replaces the other's `target/release/fe-server`, so
keep both features in one build. The packages a clean Ubuntu machine needs are in
[docs/quickstart.md, "Clean machine"](quickstart.md#clean-machine).

`scripts/fetch-flashnext.sh` downloads, resumes and checks (size and sha256). It uses the Hugging Face CLI when
`hf` is installed, else curl with eight byte ranges of a file at once (`FNEXT_FETCH_PARTS`); each range resumes
where it stopped and is retried with a growing pause (`FNEXT_FETCH_TRIES`, default 10), so a dropped or stalled
connection does not end the download. A file of the right size whose sha256 differs is moved to `NAME.bad` and
fetched again. `FNEXT_FETCH=curl` or `=hf` picks the downloader.

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

The server holds a lock for each card it uses (next to `FERRITE_GPU_LOCK`, by card UUID) and the node lock shared,
so a 27B server on one card and Flash-Next on the other run side by side. It stops at once when a card is taken
and names the process holding it (`FERRITE_LOCK_WAIT=1` waits for it instead). It also stops in a second, before loading, when the port is taken.

| Setting | Default | |
|---|---|---|
| `FNEXT_DIR` | `$FERRITE_MODELS/Qwen3.8-Flash-Next` | the folder from `scripts/fetch-flashnext.sh` |
| `FERRITE_GPUS` | `auto` | `1` or `2` |
| `FERRITE_HOST`, `FERRITE_PORT` | `0.0.0.0`, `8000` | every interface, so other machines on the network can use it; there is no API key, so use it on a trusted network or firewall the port. `127.0.0.1` keeps it on this machine |
| `FERRITE_MODEL_ID` | `qwen3.8-flash-next` | the id `/v1/models` reports |
| `FERRITE_CTX` | `131072` | context length |
| `FERRITE_SLOTS` | see above | requests at once, 1-8 |
| `FERRITE_DRAFT` | `mtp` when the draft file is there, else `off` | |
| `FERRITE_LOCK_WAIT` | `0` | `1`: wait for a busy card instead of stopping |
| `FNEXT_COUNTS` | `crates/fe-fnext/data/routing-mix.json` | expert routing counts that pick the experts loaded first |
| `FNEXT_CPU_POOL` | the E-cores on a hybrid Intel CPU, else one thread per physical core not taken by the engine's 3 own threads | CPUs that compute the experts that miss the card |

## Recommended settings

The server reads `sampling.json` from the model folder: the base model's published sampling settings, which it
applies to the fields a request leaves out, and the protections below. Ferrite's copy for this model is
`crates/fe-server/sampling.qwen3.8-flash-next.json`; `tools/serve-fnext.sh` uses it when the folder has none, and
`FE_SAMPLING_JSON=<file>` names another file.

| | Thinking | Non-thinking |
|---|---|---|
| temperature | 1.0 | 0.7 |
| top_p | 0.95 | 0.80 |
| top_k | 20 | 20 |
| presence_penalty | 0 | 1.5 |
| repetition_penalty | 1.0 | 1.0 |

- A value the client sends always wins, field by field. The values a request ran with are returned in the
  `x-sampling-effective` response header and written to the log.
- **Temperature 0 stays greedy.** A request that sends `temperature: 0` gets no penalty from the table, so it
  keeps MTP drafting. A penalty the client sends itself still applies and takes the request off that path.
  `"greedy_keeps_fast_path": false` in `sampling.json` (or `FE_GREEDY_FAST_PATH=0`) turns the rule off.
- **A request that leaves temperature out is sampled** with the table's values and decodes one token a step
  (see Limits). A coding agent that wants the drafting speed sends `temperature: 0`.

Protections, the same code as for the 27B model:

| Protection | Default | Switch |
|---|---|---|
| Reasoning cap: a thinking block still open after 8,192 tokens is closed by the engine and the model answers | 8192 | `max_reasoning_tokens`, `FE_MAX_REASONING_TOKENS` (0 = off) |
| Loop breaker: an exact repeating cycle closes a reasoning block; in an answer it boosts the sampling for 96 tokens and ends the reply (`finish_reason: "length"`) if the cycle returns. Tool-call arguments are held to a far stricter rule | on | `loop_breaker`, `FE_LOOP_BREAKER=0`; thresholds `loop_rules`, `FE_LOOP_RULES` |
| Repeated-sentence rule: a reply that keeps coming back to the same sentences is treated like a cycle; never inside a code block or a tool call | on | `loop_rules.sentence` |
| The first token of a reply is never a stop token | on | none |
| `tool_choice` `required` / a named function is enforced, limited to the tools offered | on | none |
| No token for 60 s after the first one ends the reply | 60 s | `FE_NO_TOKEN_TIMEOUT` (seconds, 0 = off) |
| Wall-clock limit of a reply | off | `FE_REPLY_WALL_LIMIT` (seconds) |
| Non-finite logits end the request with an error instead of text | on | `FE_FINITE_CHECK=0` |
| SSE keep-alive comment every 10 s of silence | on | none |
| Request capture for replay | off | `FE_DUMP_REQUESTS=<dir>` |

A protection acts at the token where it fires, whatever the number of drafts a step verified: the tokens accepted
past that point are given back. With protections firing, greedy replies are the same with drafting on and off, and
the same alone as with other requests beside them. A request on which none fires gets the tokens it got before.

Harness settings that were tested end to end are in [harness.md](harness.md).

### The Coder file is for code

The GSQ-RCO Coder file is pruned for code and handles general-chat prompts poorly, non-English words in particular.
Measured on two RTX PRO 4000 Blackwell 24 GB with short prompts and a 1,024-token limit (temperature 0, 0.6 and
the default sampling, thinking on and off): coding prompts, with and without tools, ended on their own in 176 of
176 runs; general-chat prompts ran to the limit in 39 of 112. A request to translate two words into French and
Spanish circled without producing the Spanish. Use this model for code.

## Limits

- Experimental: measured on one machine, two 5060 Ti cards. The single-card preset was measured on that machine under
  a 28 GiB memory limit, not yet on a 32 GB desktop; CPU-side speed (expert misses, PLE reads) depends on the host.
- Text only: the vision tower (`mmproj`) is not supported.
- The first request after a start reads experts and PLE rows from disk and is slower; a short prompt right after a
  very long one is ~20% slower for a while (its expert slots refill).
- Requests that sample (temperature > 0, or no temperature at all) decode one token a step; greedy requests use MTP
  drafting. Several requests at once share a step either way, and while one of them samples, the greedy ones beside
  it decode one token a step too and do not return to drafting during that request.
- After the reasoning cap or a reasoning cycle closes a thinking block, a greedy request keeps drafting. The boosted
  stretch after an answer loop is sampled: a request that runs alone drafts again when the stretch ends, one with
  other requests beside it stays at one token a step.
- The check for non-finite logits covers sampled picks on every preset; greedy picks are covered on two cards only.
- A prompt is processed while the other requests wait: their decoding pauses until it is done. Requests that
  arrive together with fresh prompts therefore start one after another (four 2K-token prompts on two cards: the
  last first token after ~6.5 s, 88 tok/s in total over the whole run, against ~190 with short or cached prompts).
  Chat follow-ups and agent turns mostly hit the prompt cache, so their prompts are short. A request longer than a
  slot's share of the context (32K tokens of the 128K with four slots) waits until the others are done and runs
  alone.
- A client that disconnects stops its request within a fraction of a second, also in the middle of a long prompt
  (it stops between layers); the slot keeps the conversation up to its last prompt-cache point.
- Same weights as published, never requantized: the IQ1_M files are loaded as they are.

## License

Ferrite is Apache-2.0. The model files have their own licenses: Qwen3.8-Flash-Next is under the Qwen Community
License 1.0, which requires a separate license from Qwen before a "Model as a Service" or "AI Work Assistant"
business uses it commercially (see the draft repository's `LICENSE-NOTE.md`); check the ISTA-DASLab repository for
the terms of the IQ1_M files.
