# DeepSeek-V4.1-Flash Coder on three or four cards

Ferrite 0.8.0 adds a third model: **DeepSeek-V4.1-Flash Coder**, DeepSeek-V4.1-Flash (40 MoE blocks of 192 routed
experts, pruned from 384, six picked a token plus a shared expert; compressed sparse attention with an indexer,
hyper-connections, an Engram n-gram table) in the [ISTA-DASLab GSQ-RCO Coder IQ2_XXS](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF)
quant: the experts pruned for code and quantized to about 2 bits, 72 GB of GGUF. It runs on **three 24 GB cards**
(tested on three RTX PRO 4000 Blackwell), optionally with a **fourth card of 16 GB or more** (tested with an RTX
5060 Ti). It is a separate engine in the same server (`fe-server --engine dsc`, crate `fe-dsc`), with its own Rust
kernels compiled to PTX like the rest, and the [DSpark](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)
draft model for speculative decoding.

How the model is spread over the cards:

- the **first card** (the main card) holds the attention, the dense layers, the shared experts and the output head,
  and its share of the routed experts. It sets the pace: give it your fastest slot;
- the **other cards** hold routed experts only, the third also the DSpark drafter. The main card sends each layer's
  input to them and gets the results back card to card;
- which experts go on the cards comes from a **placement heat** this tree ships (`crates/fe-dsc/heat/coder.cnt`:
  pick counts measured on code, agent sessions, chat and long HTML/JS writes), the most used first. With three cards
  about 1,440 of the 7,680 experts do not fit; they stay in pinned host memory (~11.6 GiB) and are copied to a card
  when a token picks one. With a fourth card all of them are on the cards;
- the **Engram table** (111 GB) is read 48 rows a token from its file, so keep it on an SSD; rows that are not in
  the page cache yet are read in parallel.

> **Status: experimental.** One model, built from source only, on Linux. Measured on one machine (below). The
> layout needs three cards of 24 GB; other card mixes are not tested. Reports welcome.

## Results

AMD EPYC 7313 x 2, 128 GB DDR4, three RTX PRO 4000 Blackwell (24 GB, 145 W) on PCIe 4.0 (two x16, one x8), and for
the four-card column an RTX 5060 Ti 16 GB (x8). Ferrite 0.8.1, `tools/serve-dsc.sh` with its defaults, 128K context
configured, temperature 0, thinking off, measured at the server: replies of 900 to 1,500 tokens, the second of two runs
after a short warm-up.

| tokens/s | Three cards | Four cards |
|---|---|---|
| Python | 103 | 114 |
| Rust | 102 | 114 |
| HTML / JavaScript | 115 | 129 |
| Chat (prose) | 43 | 69 |
| Eight prompts the server has not seen before, first reply each after a start (average) | 87 | 105 |
| Four code requests at once (`FERRITE_SLOTS=4`), total | 118 | 142 |
| Four requests at once, three code and one chat, total | 97 | 135 |

- Text the server has not seen before reads its Engram rows from the SSD, not the page cache. Since 0.8.1 those
  reads are issued together, so such replies run close to repeated ones; with 0.8.0 the same eight prompts averaged 74
  (three cards) and 88 tokens/s (four cards), with the same text.

- Chat gains most from the fourth card: prose picks far more of the experts that do not fit on three cards.
- Several requests at once add little on this model. Only the dense part of each pass is shared; every row of every
  request picks its own experts, and speculative decoding already fills a single request's passes with 5 to 6 rows.
  Four requests share about 140 tokens/s; one alone gets 109 to 130. With `FERRITE_SLOTS=4` a request alone runs a few
  percent slower than with 1.
- Reasoning text (thinking on, the default) decodes slower than code: plan for lower numbers in a thinking session.
- Speculative decoding is lossless: greedy output with drafting is the same as without, four requests at once give
  the same text as each alone, and the four-card layout gives the same text as three.

## Setup

```bash
# build (x86_64, CUDA 13 ptxas; the engine is behind a feature)
cargo build --release -p fe-server --features dsc -p fe-dsc

# the files: ~191 GB into $FERRITE_MODELS/DeepSeek-V4.1-Flash-Coder (FERRITE_MODELS default /var/tmp/models)
scripts/fetch-deepseek.sh

# serve on 0.0.0.0:8000 (the cards: every visible card, the largest first; the first is the main card)
tools/serve-dsc.sh
```

`tools/serve-dsc.sh --help` lists its settings. The ones you are most likely to need:

| Variable | Default | What it sets |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | every card, largest first | the cards, in role order: main card first, the drafter on the third |
| `DSC_CARDS` | 4 when four are visible, else 3 | how many cards to use |
| `FERRITE_SLOTS` | 1 | requests at once, 1-4 |
| `FERRITE_CTX` | 131072 | context length |
| `DSC_NUMA` | bound to node 0 on a multi-socket machine | the `numactl` binding; bind it to the CPU the main card is attached to, `""` runs unbound |
| `DSC_DIR` | `$FERRITE_MODELS/DeepSeek-V4.1-Flash-Coder` | the model folder |

Engine settings (`DSC_*`) pass through; the launcher sets the tested values. Among them: `DSC_RESERVE_GBS` (card
memory left free per card, GiB: main, expert cards, drafter card), `DSC_HEAT` (the placement heat file),
`DSC_SPEC=0` (no drafting), `DSC_COPY=0` (no drafts copied from the context), `DSC_ENGRAM_THREADS` (reader threads for
Engram rows that are not in the page cache, default 128).

## Behaviour

The model's settings come from `crates/fe-server/sampling.deepseek-v4.1-flash-coder.json` (copied into the model
folder as `sampling.json` if you want to change them): temperature 1.0 / top-p 0.95 for sampled requests in both
modes, a request's own fields win, and a temperature-0 request decodes greedy. The protections of
[docs/harness.md](harness.md) apply on every request, also with several at once: the reasoning cap (8,192 tokens,
then the engine closes the block itself), the loop breaker, the repeated-sentence rule, the first-token stop ban,
`tool_choice`, the watchdog and the finite check. Tool calls use the model's DSML format; a tool call that starts to
repeat itself is rewound to where the cycle began and continued with sampling.

Thinking is on unless the request turns it off (`chat_template_kwargs.enable_thinking: false`; a harness's thinking
setting does this).

Known limits:

- **Very long single tool calls** (one file write of thousands of lines) can still fall into a loop at temperature
  0: about 3 in 10 in our tests. Writing a long file in parts avoids it.
- **Other languages than English** in chat run away or answer in the wrong language: the Coder quant was pruned for
  code and lost part of that vocabulary. That is the model, not the engine.
- The Engram table is pinned to the revision of its repository before it moved to the llama.cpp tensor layout
  (2026-10-27); `scripts/fetch-deepseek.sh` fetches that revision.

## Models

| Role | Repository | License |
|---|---|---|
| The model: GSQ-RCO Coder IQ2_XXS | [`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF`](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF) | MIT |
| Engram table (IQ4_NL) | [`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) | MIT |
| Draft model: DSpark Q8_0 | [`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | MIT |

All derive from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
