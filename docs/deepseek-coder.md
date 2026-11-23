# DeepSeek-V4.1-Flash Coder on three or four cards

**DeepSeek-V4.1-Flash Coder** is DeepSeek-V4.1-Flash, a large mixture-of-experts model, in a quant made for coding.
Ferrite serves it on **three 24 GB cards** (tested on three RTX PRO 4000 Blackwell), optionally with a **fourth card
of 16 GB or more** (tested with an RTX 5060 Ti), from the
[ISTA-DASLab GSQ-RCO Coder IQ2_XXS](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF)
quant: the experts pruned for code and quantized to about 2 bits, 72 GB of GGUF.

- **Speed:** Python at 103 tokens/s on three cards and 114 on four, with a 128K-token context ([Results](#results)).
- **Use it for code.** The Coder quant was pruned for code: chat in languages other than English runs away or
  answers in the wrong language ([Limits](#limits)).
- **Plan for** about 191 GB of disk on an SSD and 128 GB of host memory (the tested size).

Terms such as experts, drafting, slots and page cache are explained in
[Words used in these docs](../README.md#words-used-in-these-docs).

> **Status: experimental.** One model, built from source only, on Linux. Measured on one machine (see Results). The
> layout needs three cards of 24 GB; other card mixes are not tested. Reports welcome.

## Requirements

| | |
|---|---|
| Cards | **Three 24 GB cards** (tested: three RTX PRO 4000 Blackwell): the launcher requires at least about 23,000 MiB on each of the first three. An optional **fourth card of 16 GB or more** (tested: RTX 5060 Ti; at least about 15,000 MiB). The launcher uses 3 or 4 cards. The first card is the main card and sets the pace: give it your fastest x16 slot. |
| CPU | x86_64. |
| Host memory | Tested with 128 GB. With three cards about 12 GiB of experts are pinned in host memory beside the page cache. Below 60 GiB the launcher warns that the start may fail below 64 GB. |
| Disk | About 191 GB (more while the download keeps a `.bad` copy of a damaged file). Keep the Engram table (111 GB) on an SSD: it is read while generating. |
| OS and driver | Linux (tested on Ubuntu 24.04), NVIDIA driver 580 or later. |
| To build | The Rust toolchains of the other NVIDIA builds and `ptxas` from the CUDA 13 toolkit ([docs/quickstart.md, "Clean machine"](quickstart.md#clean-machine)). |
| To download | `curl`, `sha256sum` and `python3` (standard library only); the Hugging Face CLI (`hf`) is used when it is installed. |
| Optional | `numactl`, on a machine with more than one CPU socket (see `DSC_NUMA` under Settings). |

## Run it

```bash
# 1. build (x86_64, CUDA 13 ptxas; the engine is behind a feature)
cargo build --release -p fe-server --features dsc -p fe-dsc

# 2. the files: ~191 GB into $FERRITE_MODELS/DeepSeek-V4.1-Flash-Coder (FERRITE_MODELS default /var/tmp/models)
scripts/fetch-deepseek.sh

# 3. serve on 0.0.0.0:8000 (the cards: every visible card, the largest first; the first is the main card).
#    Foreground; loading takes two to four minutes
tools/serve-dsc.sh

# 4. from another terminal: test it
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4.1-flash-coder","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":200}'
```

- **The build.** `-p fe-dsc` also builds `dsc-mkmodel`, which `scripts/fetch-deepseek.sh` uses to write the tokenizer
  files from the first GGUF (the script builds it if it is missing). The build compiles the GPU kernels for the card
  `nvidia-smi` reports. Set `FE_KARCH=sm_120a` in front of the command only when the build machine does not hold the
  cards: with no card to ask, the build falls back to `sm_121a`, the DGX Spark.
- **The download.** `scripts/fetch-deepseek.sh` checks every file against its size and sha256, resumes where it
  stopped and retries. The Coder GGUF is five files named `...-0000N-of-00006.gguf`: the repository has no part 6.
  The Engram table is pinned to the revision of its repository before it moved to the llama.cpp tensor layout
  (2026-10-27), which is the layout this engine reads; the script fetches that revision.
- **Stopping and logs.** `tools/serve-dsc.sh` runs in the foreground and its process becomes the server: Ctrl-C or
  `kill <pid>` stops it, and its log is what the terminal shows. `scripts/ferritectl.sh` has no target for this
  engine, so there is no built-in background mode.
- **Thinking is on** unless the request turns it off (`chat_template_kwargs.enable_thinking: false`; a harness's
  thinking setting does this).

## Settings

`tools/serve-dsc.sh --help` lists its settings. The ones you are most likely to need:

| Variable | Default | What it sets |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | every card, largest first | The cards, in role order: the first is the main card, the third holds the drafter. |
| `DSC_CARDS` | 4 when four are visible, else 3 | How many cards to use. |
| `FERRITE_SLOTS` | 1 | Requests at once (slots), 1-4. |
| `FERRITE_CTX` | 131072 | Context length. |
| `FERRITE_HOST`, `FERRITE_PORT` | `0.0.0.0`, `8000` | Every interface, so other machines on the network can use it; there is no API key, so use it on a trusted network or firewall the port. `127.0.0.1` keeps it on this machine. |
| `FERRITE_MODEL_ID` | `deepseek-v4.1-flash-coder` | The id `/v1/models` reports. |
| `DSC_NUMA` | bound to node 0 on a multi-socket machine | The `numactl` binding. Bind it to the CPU the main card is attached to; `""` runs unbound. |
| `DSC_DIR` | `$FERRITE_MODELS/DeepSeek-V4.1-Flash-Coder` | The model folder. |

### Advanced

Engine settings (`DSC_*`) pass through; the launcher sets the tested values.

| Variable | What it sets |
|---|---|
| `DSC_RESERVE_GBS` | Card memory left free per card, in GiB: main card, expert cards, drafter card. |
| `DSC_HEAT` | The placement heat file (default `crates/fe-dsc/heat/coder.cnt`). |
| `DSC_SPEC=0` | No drafting. |
| `DSC_COPY=0` | No drafts copied from the context. |
| `DSC_ENGRAM_THREADS` | Reader threads for Engram rows that are not in the page cache, default 128. |

### Sampling and protections

- The model's settings come from `crates/fe-server/sampling.deepseek-v4.1-flash-coder.json` (copied into the model
  folder as `sampling.json` if you want to change them): temperature 1.0 / top-p 0.95 for sampled requests in both
  modes. A request's own fields win, and a temperature-0 request decodes greedy.
- The protections of [docs/harness.md](harness.md) apply on every request, also with several at once: the reasoning
  cap (8,192 tokens, then the engine closes the block itself), the loop breaker, the repeated-sentence rule, the
  first-token stop ban, `tool_choice`, the watchdog and the finite check.
- Tool calls use the model's DSML format; a tool call that starts to repeat itself is rewound to where the cycle
  began and continued with sampling.

## Results

**What to expect:** code at about 102-115 tokens/s on three cards and 114-129 on four; chat at 43 and 69.

Measured with Ferrite 0.8.1.

| tokens/s | Three cards | Four cards |
|---|---|---|
| Python | 103 | 114 |
| Rust | 102 | 114 |
| HTML / JavaScript | 115 | 129 |
| Chat (prose) | 43 | 69 |
| Average over eight prompts the server had not seen before, each the first reply after a start | 87 | 105 |
| Four code requests at once (`FERRITE_SLOTS=4`), total | 118 | 142 |
| Four requests at once, three code and one chat, total | 97 | 135 |

**How it was measured.** AMD EPYC 7313 x 2, 128 GB DDR4, three RTX PRO 4000 Blackwell (24 GB, 145 W) on PCIe 4.0
(two x16, one x8), and for the four-card column an RTX 5060 Ti 16 GB (x8). `tools/serve-dsc.sh` with its defaults,
128K context configured, temperature 0, thinking off, measured at the server: replies of 900 to 1,500 tokens, the
second of two runs after a short warm-up.

- **Text the server has not seen before** reads its Engram rows from the SSD, not the page cache. Since 0.8.1 those
  reads are issued together, so such replies run close to repeated ones; with 0.8.0 the same eight prompts averaged
  74 (three cards) and 88 tokens/s (four cards), with the same text.
- **Chat gains most from the fourth card:** prose picks far more of the experts that do not fit on three cards.
- **Several requests at once add little on this model.** Only the dense part of each pass is shared; every row of
  every request picks its own experts, and speculative decoding already fills a single request's passes with 5 to 6
  rows. On four cards, four code requests share about 140 tokens/s; one alone gets 114 to 129. With
  `FERRITE_SLOTS=4` a request alone runs a few percent slower than with 1.
- **Reasoning text** (thinking on, the default) decodes slower than code: plan for lower numbers in a thinking
  session.

## Accuracy

- **Against the original model: not measured.** Ferrite has no KL or perplexity figure for this quant against the
  unquantized DeepSeek-V4.1-Flash.
- **What is checked:** speculative decoding is lossless (greedy output with drafting is the same as without), four
  requests at once give the same text as each alone, and the four-card layout gives the same text as three.

## Limits

- **Other languages than English** in chat run away or answer in the wrong language: the Coder quant was pruned for
  code and lost part of that vocabulary. That is the model, not the engine.
- **Very long single tool calls** (one file write of thousands of lines) can still fall into a loop at temperature
  0: about 3 in 10 in our tests. Writing a long file in parts avoids it.
- **Card mixes.** The layout needs three cards of 24 GB; other card mixes are not tested.

## How the model is spread over the cards

The model has 40 MoE blocks of 192 routed experts (pruned from 384), six picked a token plus a shared expert, with
compressed sparse attention with an indexer, hyper-connections and an Engram n-gram table. It runs in a separate
engine in the same server (`fe-server --engine dsc`, crate `fe-dsc`), with its own Rust kernels compiled to PTX like
the rest, and the [DSpark](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) draft model for speculative
decoding.

- The **first card** (the main card) holds the attention, the dense layers, the shared experts and the output head,
  and its share of the routed experts. It sets the pace: give it your fastest slot.
- The **other cards** hold routed experts only, the third also the DSpark drafter. The main card sends each layer's
  input to them and gets the results back card to card.
- **Which experts go on the cards** is decided by a usage-count file shipped with Ferrite, the placement heat
  (`crates/fe-dsc/heat/coder.cnt`: pick counts measured on code, agent sessions, chat and long HTML/JS writes): the
  most used go on the cards first. With three cards about 1,440 of the 7,680 experts do not fit; they stay in
  pinned host memory (~11.6 GiB) and are copied to a card when a token picks one. With a fourth card all of them
  are on the cards.
- The **Engram table** (111 GB) is read 48 rows a token from its file, so keep it on an SSD; rows that are not in
  the page cache yet are read in parallel.

## Models

| Role | Repository | License |
|---|---|---|
| The model: GSQ-RCO Coder IQ2_XXS | [`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF`](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF) | MIT |
| Engram table (IQ4_NL) | [`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) | MIT |
| Draft model: DSpark Q8_0 | [`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | MIT |

All derive from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
