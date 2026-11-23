# Inkling-2-Mini on one RTX PRO 4000 or RTX 5060 Ti

**Inkling-2-Mini** from [Thinking Machines Lab](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Apache-2.0) is a
mixture-of-experts reasoning model of about 20 GB. Ferrite serves it from the checkpoint as published (NVFP4 and MXFP8
experts, FP8 attention) on **one NVIDIA card**: an **RTX PRO 4000 Blackwell (24 GB)** or an **RTX 5060 Ti (16 GB)**,
with a **128K-token context on either**.

- **Speed:** a short code request runs at 138 tokens/s on the 24 GB card and 110 on the 16 GB card
  ([Results](#results)).
- **Text only, one request at a time, no prompt cache.** The published model also reads images; this engine does
  not ([Limits](#limits)).
- **It reasons by default.** Reasoning arrives in `reasoning_content`, and the effort setting lowers it without
  switching it off ([Using the server](#using-the-server)).

Terms such as K/V cache, experts, drafting, prefill and KL are explained in
[Words used in these docs](../README.md#words-used-in-these-docs).

> **Status: experimental.** One model, built from source only, on Linux. Measured on one machine (see Results). Text only,
> one request at a time, no prompt cache. Reports and measurements from other cards are welcome.

## Requirements

| | |
|---|---|
| Card | One **RTX PRO 4000 Blackwell (24 GB)** or one **RTX 5060 Ti (16 GB)**. The launcher uses the first visible card and picks the layout from its memory: the 24 GB layout from about 23,000 MiB, the 16 GB layout from about 15,000 MiB; it stops on a smaller card. |
| CPU | x86_64. |
| Host memory | 16 GB card: 32 GB recommended, because up to about 4 GiB of experts are pinned in host memory (the launcher warns below 24 GiB). 24 GB card: any; the launcher does not check it. |
| Disk | About 20 GB for the model files. |
| OS and driver | Linux (tested on Ubuntu 24.04), NVIDIA driver 580 or later (tested with 595.99). |
| To build | The Rust toolchains of the other NVIDIA builds and `ptxas` from the CUDA 13 toolkit ([docs/quickstart.md, "Clean machine"](quickstart.md#clean-machine)). |
| To download | `curl`, `sha256sum` and `python3` (standard library only); the Hugging Face CLI (`hf`) is used when it is installed. |

What each layout keeps where is under [What it runs on](#what-it-runs-on).

## Run it

```bash
# 1. build; FE_KARCH=sm_120a compiles the kernels for these cards
FE_KARCH=sm_120a cargo build --release -p fe-server --features ink -p fe-ink

# 2. download: ~20 GB into $FERRITE_MODELS/Inkling-2-Mini (default /var/tmp/models), pinned revision, sha256 checked
scripts/fetch-inkling.sh

# 3. serve on :8000: picks the layout from the card (foreground; loading takes one to two minutes)
tools/serve-ink.sh

# 4. from another terminal: test it
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":200}'
```

- **The build.** `-p fe-ink` also builds `target/release/ink-run`, the tool for the self-test and the benchmarks
  named on this page.
- **The first start** also computes the draft head's fold tables (about a minute of CPU) and keeps them beside the
  model as `fold.safetensors`.
- **The test reply.** The model reasons before it answers, so the first part of a short reply is in
  `reasoning_content` and the answer in `content`.
- **Stopping and logs.** `tools/serve-ink.sh` runs in the foreground and its process becomes the server: Ctrl-C or
  `kill <pid>` stops it, and its log is what the terminal shows. `scripts/ferritectl.sh` has no target for this
  engine, so there is no built-in background mode.
- **Network.** It listens on every interface by default (`FERRITE_HOST=127.0.0.1` keeps it on the machine; there is
  no API key) and holds the card's GPU lock.
- **Check an install** with `target/release/ink-run selftest` (about two minutes; `--deep 65536` adds a 64K prompt).
- **Measure it** with `tools/bench-ink.py` (against a running server) or
  `target/release/ink-run bench --ctx 8192 --kind code`.

## Settings

`tools/serve-ink.sh --help` lists the launcher's settings. The ones you are most likely to need:

| Variable | Default | What it sets |
|---|---|---|
| `INK_KV` | `6bit` on both layouts | The K/V cache of the 8 global layers: `6bit` or `bf16`. What bf16 costs is under [Results](#results). |
| `INK_DRAFT` | `on` | The model's own draft head: `on` or `off`. |
| `INK_LAYOUT` | from the card's memory | Force a layout: `24gb` or `16gb`. |
| `FERRITE_CTX` | `131072` | Context length, prompt plus answer. |
| `FERRITE_HOST`, `FERRITE_PORT` | `0.0.0.0`, `8000` | Every interface, so other machines on the network can use it; there is no API key, so use it on a trusted network or firewall the port. `127.0.0.1` keeps it on this machine. |
| `FERRITE_MODEL_ID` | `inkling-2-mini` | The id `/v1/models` reports. |
| `FERRITE_LOCK_WAIT` | `0` | `1`: wait for a busy card instead of stopping. |
| `INK_NUMA` | bound to node 0 on a multi-socket machine with `numactl` | The `numactl` binding. Bind it to the node the card hangs off; `""` runs unbound. |
| `INK_DIR` | `$FERRITE_MODELS/Inkling-2-Mini` | The model folder. |
| `FE_DEFAULT_TEMPERATURE`, `FE_DEFAULT_REASONING_EFFORT`, `FE_DEFAULT_MAX_TOKENS` | 1.0, the template's 0.9, 32768 | Request defaults ([Using the server](#using-the-server)). |

The engine's own settings, which the launcher sets for the layout, are under [Advanced settings](#advanced-settings).

## Using the server

OpenAI-compatible chat and completions API, streaming or not. The model writes typed messages (reasoning, answer, tool calls); the server
parses them, so a client sees:

- **Reasoning** in `reasoning_content` (`message.reasoning_content`, or `delta.reasoning_content` when streaming); the answer is in `content`, clean.
  `usage.completion_tokens` counts everything generated, `usage.completion_tokens_details.reasoning_tokens` the tokens of the reasoning block.
- **Tool calls** as OpenAI `tool_calls` (several in one turn, `finish_reason: "tool_calls"`; `parallel_tool_calls: false` keeps the first), rendered through
  the model's own chat template. Arguments are checked and streamed as the model wrote them. `tool_choice` `required` or a named function is only a
  hint to the model. A call in a request that offered no tools is dropped and ends the turn. Text of a message that spells a control token
  (`<|content_thinking|>`) stays text, and cannot forge a message boundary.
- **Stop strings** and `max_tokens` cut at exactly that token, also inside an accepted draft block.

The chat template is checked against the publisher's: 28 conversations (tools, parallel calls, results by id or name, reasoning kept in the
history) render byte for byte like the Hugging Face template and tokenize to the same ids (`cargo test -p fe-server --test inkling_template`).

**Defaults.** A request that omits a field gets the publisher's settings, and everything a request ran with is reported in the
`x-sampling-effective` response header. What the request sends wins.

| Setting | Default | Change the default (launcher environment) | Request fields |
|---|---|---|---|
| Sampling | temperature 1.0 with sampling on and no top-p / top-k: the publisher's `generation_config.json` | `FE_DEFAULT_TEMPERATURE=0.7` | `temperature`, `top_p`, `top_k`, `min_p` |
| Reasoning effort | 0.9 ("high"), the chat template's own default | `FE_DEFAULT_REASONING_EFFORT=medium`: a name from none, minimal, low, medium, high, xhigh, max, or a number 0 to 0.99 | `reasoning_effort`, `chat_template_kwargs.reasoning_effort`, and `enable_thinking: false` (= effort `none`) |
| Output cap of a request without `max_tokens` | 32,768 tokens, because the model card advises limiting the output: reasoning can run long | `FE_DEFAULT_MAX_TOKENS=8192` (`0` = no cap) | `max_tokens` / `max_completion_tokens` |

An effort the template does not know is a 400 with the template's own message.

**What the effort does.** It is a soft control, and `none` does **not** switch reasoning off (the model card says so too): on short prompts the model still
writes 40 to 380 reasoning tokens.

| Request | Effort | Reasoning tokens | Answer tokens | Seconds | Tokens/s |
|---|---|---|---|---|---|
| short coding request (a function and two tests) | none | 98 | 374 | 3.6 | 130.4 |
| | low | 27 | 351 | 3.5 | 136.6 |
| | medium | 260 | 402 | 5.7 | 117.6 |
| | high (default) | 1,604 | 555 | 14.7 | 130.6 |
| short prose request (a few paragraphs) | none | 230 | 289 | 6.6 | 80.4 |
| | low | 157 | 335 | 5.8 | 79.6 |
| | medium | 141 | 286 | 5.3 | 80.7 |
| | high (default) | 386 | 374 | 9.6 | 79.2 |

(Median of three seeds at temperature 1.0, answered to the end, RTX PRO 4000, draft on, the default 6-bit K/V, measured
through `tools/serve-ink.sh` with `tools/conformance/ink_settings_bench.py`.)

- **Effort changes the length of the reasoning, not reliably in order.** Low can think less than none on one prompt, and one effort varies a lot
  from seed to seed: 749 to 1,792 reasoning tokens for the same coding request at high.
- Reasoning is 40 to 55% of the tokens of a short request at none, medium and high.
- **Drafting follows the text, reasoning included.** A coding request, whose reasoning is mostly about code, runs at 120 to 155 tokens/s at
  temperature 0, 0.7 and 1.0 (drafting does not slow sampled requests). Reasoning and answers about prose-like questions run at the plain speed
  (about 80 tokens/s on this card): the draft head gives no speed-up there.

## Results

**What to expect:** with the draft head on, a short code request runs at 138 tokens/s on the RTX PRO 4000 and 110 on the RTX 5060 Ti; prose
runs at the plain speed, about 83 and 70. At a 126K-token context the RTX PRO 4000 still decodes code at 117 tokens/s and the RTX 5060 Ti at 64.

Measured with Ferrite 0.9.0. "Draft" is the model's own draft head, on by default. It is lossless: the text is the same with it and without it
(`ink-run selftest` checks that). Drafted speed depends on the text; plain speed does not.

**Short prompts** (chat requests of 25 to 160 tokens), tokens per second generated:

| | RTX PRO 4000 24 GB, draft on | RTX PRO 4000, draft off | RTX 5060 Ti 16 GB, draft on | RTX 5060 Ti, draft off |
|---|---|---|---|---|
| Code (a Python function with tests) | 138.1 | 83.2 | 110.4 | 69.2 |
| Agent turn (a tool result to answer from) | 126.2 | 82.6 | 98.8 | 69.1 |
| Prose (a short essay) | 83.9 | 83.3 | 68.5 | 69.9 |

**Long contexts.** Prefill is prompt tokens per second; decode is generated tokens per second for a code and for a prose continuation.

RTX PRO 4000 Blackwell, 24 GB:

| Context | Prefill | Decode, draft on: code | Decode, draft on: prose | Decode, draft off: code | Decode, draft off: prose |
|---|---|---|---|---|---|
| 8K | 1,939 | 158.6 | 102.1 | 78.5 | 78.3 |
| 32K | 1,446 | 141.6 | 72.3 | 74.2 | 74.5 |
| 64K | 1,049 | 141.9 | 137.5 | 68.5 | 68.6 |
| 126K | 703 | 117.0 | 76.8 | 59.6 | 59.6 |

RTX 5060 Ti, 16 GB:

| Context | Prefill, draft on | Prefill, draft off | Decode, draft on: code | Decode, draft on: prose | Decode, draft off: code | Decode, draft off: prose |
|---|---|---|---|---|---|---|
| 8K | 1,163 | 1,275 | 128.0 | 76.5 | 64.6 | 63.5 |
| 32K | 879 | 957 | 97.1 | 61.9 | 58.4 | 59.8 |
| 64K | 638 | 699 | 95.9 | 77.9 | 54.0 | 52.5 |
| 126K | 413 | 453 | 64.0 | 81.3 | 44.5 | 43.6 |

- The contexts are 8,305 / 32,753 / 65,581 / 126,216 tokens for the code columns and 8,017 / 32,465 / 65,293 / 125,928 for the prose columns.
- Both cards run the default 6-bit K/V cache.
- Prefill is the same with the draft head on or off to within 1% on the 24 GB card; on the 16 GB card the head leaves less room for the prompt
  chunk, so it prefills about 8% slower.

**The draft head and the text.** The speed of a drafted request is the model's own acceptance of its draft, and it changes with the text by a
factor of two; the draft-off numbers do not. The long-context tables use mixed real text, and the tail of each prompt is code or prose from this
repository, so the answer sometimes repeats what the prompt contains and sometimes does not: the 64K prose figure on the 24 GB card, 137.5, is an
answer that quotes the text before it, and the 126K prose figure on the 16 GB card, 81.3, likewise. Code that repeats earlier code (an edit, a
rename, a refactor) is the best case, measured with `ink-run bench --kind code` (a source file's text repeated through the context, default
layouts, 6-bit K/V on both cards):

| Context | RTX PRO 4000: draft on / off | RTX 5060 Ti: draft on / off |
|---|---|---|
| 32K | 171.6 / 74.5 | 119.1 / 58.5 |
| 64K | 145.4 / 68.6 | 93.3 / 52.7 |

- **Drafting never costs more than 3% against plain decoding** in any row of the tables above: the 24 GB card's 32K prose answer is 3% behind
  (72.3 against 74.5) and the 16 GB card's short prose answer 2% behind (68.5 against 69.9). A rule suspends drafting when it is slower (the
  plain decode time is measured at long contexts and the draft speed is averaged over many passes), but a short answer has too few passes to
  settle.
- **What drafting gains.** On short prompts it raises code by about 60% and agent turns by about 50%, and on code that repeats earlier code
  by up to 2.3 times (171.6 against 74.5 tokens/s at 32K on the 24 GB card). It does little for prose.
- **The draft head on the 16 GB card** also takes card memory that the experts could use, so more experts are read from host memory; the
  launcher keeps the head at every context.
- **The 16 GB card slows down with the context** because more experts live in host memory (with the draft head off, 134 of the 3,072 at 8K
  tokens, 218 at 32K, 320 at 64K and 492 at 124K, read over PCIe 4.0 x8 when picked; `ink-run bench` prints the count), and every token reads
  K/V through the 24 attention layers. Prefill falls with the context for the same reasons; the 16 GB card prefills in chunks of 256 rows.
- **The 24 GB card's** decode without the head falls from 78 tokens/s at 8K to 60 at 126K because every token reads the keys and values of
  the eight global layers (1.6 GiB of 6-bit K/V at 128K).

**bf16 K/V on the 24 GB card** (`INK_KV=bf16`). Keep the 6-bit default unless short-context drafted speed matters more to you than speed at
depth and 2.5 GiB of card memory. Same server and method:

| RTX PRO 4000, tokens/s unless marked | 6-bit K/V (default) | bf16 K/V |
|---|---|---|
| K/V cache at 128K | 1.6 GiB | 4.1 GiB |
| Plain decode, 64K / 126K | 68.5 / 59.6 | 60.8 / 49.9 |
| Prefill, 64K / 126K | 1,049 / 703 | 1,027 / 636 |
| Drafted, short prompts: code / agent / prose | 138.1 / 126.2 / 83.9 | 149.7 / 169.1 / 89.3 |
| Drafted code, 8K | 158.6 | 184.8 |
| Drafted code, 64K / 126K | 141.9 / 117.0 | 104.8 / 113.1 |
| Drafted prose, 64K / 126K | 137.5 / 76.8 | 138.5 / 100.8 |
| KL on code ([Accuracy](#accuracy)) | 0.0037 | 0.0041 |

At 64K and 126K the drafted figures move with the text.

**How it was measured.** AMD EPYC 7313 x2, 128 GB DDR4; an RTX PRO 4000 Blackwell (24 GB, 145 W) on PCIe 4.0 x16 and an RTX 5060 Ti (16 GB,
180 W) on PCIe 4.0 x8; driver 595.99. `tools/serve-ink.sh` with its defaults for the card, 131,072 tokens of context configured, measured at
the server by `tools/bench-ink.py`: temperature 0, 200 tokens generated, the second of two identical requests, one request at a time, nothing
else on the card or the host. Contexts are real text: this repository's Markdown and Rust files in turn, no block twice, whose last 1,024
tokens are code or prose that the answer continues.

## Accuracy

| | KL code | KL prose | KL long text | perplexity, code / prose (reference) |
|---|---|---|---|---|
| 24 GB layout, as published, 6-bit K/V (the default) | 0.0037 | 0.0074 | 0.0093 (16,384 tokens) | 2.074 / 4.899 (2.068 / 4.889) |
| the same with bf16 K/V (`INK_KV=bf16`) | 0.0041 | 0.0083 | 0.0087 (16,384 tokens) | 2.077 / 4.895 |
| 16 GB layout (layers 20-23 requantised to NVFP4), 6-bit K/V | 0.0050 | 0.0113 | 0.0044 (a 4,096-token text: a shorter text than the rows above, not comparable with them) | 2.079 / 4.921 |

- Requantising the four MXFP8 layers costs about 0.001 to 0.004 of KL; 6-bit K/V costs less than 0.001 (it is not a loss in every text).
- All are far below 0.05, the limit this port set for itself.

**How it was measured.** The cost of the formats is KL(reference || Ferrite) on the next-token distribution over 1,024-token slices of source
code and of Wikipedia text and a 16,384-token mixed text, free-running (the engine's own routing), and perplexity next to the reference's. The
reference is the published model run with Hugging Face transformers (weights as shipped, experts dequantised to bf16, bf16 activations). The
engine's expert GEMMs use the activation quantization the SM120 expert kernels of the model's reference code use, which is why a run with the
shipped formats is not bit-equal to the reference and costs a few thousandths of KL.

**Repeating it.** `ink-run kl` repeats the measurement, but it needs reference logits in the format described at the top of
`crates/fe-ink/src/bin/ink-run.rs`. They are not shipped, so the measurement cannot be repeated from the repository alone.

## Limits

- **Text only.** The published model reads images; the engine has no vision tower yet. A request with an image is answered without it (the prompt tells the model that an image was omitted).
- **One request at a time.** A second request waits in the queue; there are no lanes, and no prompt cache: every request prefills its whole prompt.
- **`tool_choice` is a hint.** `required` or a named function is not enforced. Tool calls follow the model's JSON format; a model that reaches
  for a tool the request did not offer is stopped (see [Using the server](#using-the-server)).
- **It reasons by default.** `reasoning_effort` or `enable_thinking: false` lowers the reasoning without removing it.
- On the 16 GB card, speed falls with the context because more experts are read from host memory ([Results](#results)).
- The context is 131,072 tokens, an eighth of the model's 1,048,576-token maximum.

## What it runs on

| | RTX PRO 4000 Blackwell, 24 GB | RTX 5060 Ti, 16 GB |
|---|---|---|
| Experts | as published: NVFP4 in layers 0-19, MXFP8 in layers 20-23, all on the card | all NVFP4 (layers 20-23 are requantised from MXFP8 when the model loads); the ones that do not fit stay in pinned host memory and are read from there when picked, more of them as the context grows |
| Attention and output head | FP8 (per-row scales), as published | the same |
| K/V cache of the 8 global layers | 6-bit (1.6 GiB at 128K), mapped as the context grows (`INK_KV=bf16` keeps bf16: 4.1 GiB at 128K) | 6-bit (1.6 GiB at 128K), mapped as the context grows |
| Draft head | bf16, 0.75 GiB | FP8, 0.45 GiB (and 4.5 KiB a position), at every context |
| Context | 131,072 | 131,072 |
| Host memory | any | 32 GB recommended: up to about 4 GiB of experts are pinned |

Everything else is the same code on both cards. Prompts and generated tokens go through the same arithmetic, so the logits of a token do not
depend on the chunk it was prefilled in, on speculative rows beside it or on the cache state (`ink-run selftest` checks that), and greedy text
with the draft head is the text without it.

## Advanced settings

These are the engine's own settings, its whole interface. The launcher sets them for the layout, so most users change none of them
(booleans: set and not `0`, `false`, `off` or `no`).

| Variable | What |
|---|---|
| `INK_DIR` | the model folder [`$FERRITE_MODELS/Inkling-2-Mini`] |
| `INK_G16` | all experts NVFP4 (the 16 GB layout's weights) |
| `INK_SLOTS` | experts in a shared pool that can move between the card and host memory (with `INK_G16`) |
| `INK_HOST_EXPERTS=N` | experts parked in host memory at the start, the coldest first by the shipped heat table (`crates/fe-ink/heat/heat_share.json`, or `INK_HEAT_SHARE=file`) |
| `INK_FIT_MB=n` | keep n MiB of the card free: before the K/V cache or the draft head grows, experts move to host memory to make room; when a request ends they move back |
| `INK_KVQ6` | 6-bit K/V cache of the global layers (rotated absmax groups of 32, 104 bytes a head and token) |
| `INK_KV_LAZY` | map the K/V cache as the context grows instead of reserving it |
| `INK_CHUNK=n` | prompt rows a forward [256; `0` sizes it from the card's free memory after loading] |
| `INK_RESERVE_MB=n` | with `INK_CHUNK=0`: memory kept free for the draft head |
| `INK_NODRAFT` | no draft head |
| `INK_DRAFT_FP8` (0, 1 or 2) | draft head precision: 0 bf16, 1 FP8 matrices, 2 FP8 matrices and the Markov head's tables [2 on a card under 20 GB, else 0] |
| `INK_DRAFT_MAX_CTX=n` | the draft head is loaded only for requests of at most n tokens (prompt plus answer) [no limit] |
| `INK_FOLD=file` | where the draft head's fold tables are kept [`$INK_DIR/fold.safetensors`] |
| `INK_MEM_LOG`, `INK_SLOT_LOG` | print the card's memory by step while loading; print each expert-residency change |

## About the model

24 layers of width 2,304, each with sliding-window (1,024) or global attention (24 query heads, 8 key/value heads of 128, a learned
relative-position bias, attention sinks, short causal convolutions on the keys, values and layer inputs) and a mixture of 128 experts, 8 picked a
token (sigmoid router, soft-capped SwiGLU). Layers 6 to 17 run three times (the loop), the layers hand their results to block attention residuals,
and a small gate passes a message from one pass to the next. 200,058 tokens of vocabulary, logits soft-capped at 30. The published model also
reads images; **this engine is text only**.

It runs in a separate engine in the same server (`fe-server --engine ink`, crates `fe-ink` and `fe-inkk`), with its own Rust kernels compiled
to PTX like the rest (`k-ink`, `k-inkm`, `k-draft`), and it uses the model's own draft head for speculative decoding.
