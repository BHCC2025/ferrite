# Benchmarks

What the numbers in the README's [Results](../README.md#results) are, how they were taken, and what they don't tell
you.

**What this page covers.** Qwen3.8-27B on two DGX Sparks; the one-Spark figures are in the last bullet of
[What these numbers don't tell you](#what-these-numbers-dont-tell-you). Every other setup (the RTX cards, the Arc Pro
B70, the other models) has its own figures and method in its guide: the README's
[What runs where](../README.md#what-runs-where) table links them.

**When.** Measured on 2026-11-15 with Ferrite 0.6.1, and not re-measured on later releases.

Terms such as drafting, copy rows, slot and prompt processing are in the README's
[Words used in these docs](../README.md#words-used-in-these-docs).

## Setup

- **Hardware:** two NVIDIA DGX Spark (GB10, 128 GB of unified memory each, of which Linux reports 121 GB), connected port to port over RoCE.
  Single-Spark numbers are labeled as such.
- **Software:** DGX OS, NVIDIA driver 580.178.04 (CUDA 13.0), Ferrite 0.6.1.
- **Model:** Qwen3.8-27B.
  - Decode weights: the EXL3 mixed 3.0 bpw quant.
  - Prompt weights: NVIDIA's NVFP4/FP8 checkpoint.
  - Draft model: DFlash2.
- **Server defaults:** DFlash2 drafting with best-first verify trees, 48 verify rows, copy rows on, 4 slots of 128K
  context. (The drafter proposes a small tree of likely continuations instead of one line; the model checks its
  branches as extra rows of one pass, the verify rows, and keeps the longest branch that matches its own choice.
  Copy rows propose the continuation of text that already appeared in the conversation.)

## How speed is measured

All runs use greedy decoding (temperature 0) against the OpenAI-compatible API.

- **Generation speed, 1 request:** 500 generated tokens after a prompt of the given length. Tokens divided by the
  time from the first token to the last.
- **Generation speed, 4 requests:** four different prompts are sent at once, 500 tokens each. The rate is the number
  of tokens generated after the last request's first token, divided by the time from that moment to the last token.
  Prompt processing is measured separately. `summary.json` also keeps the wall-clock rate that includes it.
- **Prompt processing:** prompt tokens divided by the time to the first token, with a cold prompt cache.
- **Code edits:** the model is shown a source file and asked to return it with a small change: 1,500 tokens, mostly
  copied from the prompt. This is the case speculative decoding helps most, and a common one for coding agents.
  Timed like generation: one request from its first token to its last, four requests as their combined rate.
- **Prose:** short stories. The standard benchmark's prose prompt repeats one passage many times, and models often
  answer by echoing it, which drafts perfectly and inflates the speed. Prose figures count only replies that don't
  do that: a zlib compression ratio of at least 0.4 and no repeating tail. Replies that echo are counted and reported
  separately.

Each figure is the median of 3 runs on fixed seeds; prose with 1 request is the median over 5 seeds. Charts show the
minimum and maximum as bars.

## How accuracy is measured

KL divergence of the model's next-token distribution against the BF16 model (the original weights, unquantized),
over fixed token sequences. The sequences are teacher-forced: at every position the model is given the reference
text so far, not its own earlier picks, and its distribution for the next token is compared.

| Set | Positions |
|---|---|
| Code | 40 sequences, 20,480 positions |
| Text | 20 sequences, 10,240 positions |
| Long context | 3 sequences, 1,536 positions: the 512 that follow a 15,360-token prompt |

Speculative decoding does not change the output: every drafting mode is checked to produce exactly the same text as
plain greedy decoding. A request's output does not depend on what else is running either.

| | KL vs BF16, two Sparks (lower is better) |
|---|---|
| Code | 0.021 |
| Text | 0.040 |
| 16K context | 0.024 |

## Results

See the table and chart in the [README](../README.md#results). The context sweep (2K-120K) is the chart
[`docs/assets/ferrite-bench.png`](assets/ferrite-bench.png), and the numbers behind it are in `docs/bench-data/`:

- `raw.jsonl`: one line per request of the sweep;
- `summary.json`: each case's median, minimum, maximum and counts, the quoted figures under `headline`, and the
  setup under `meta`.

Generation figures are tokens per second while generating; prompt processing is listed separately. With several
requests, the time to the last request's first token matters too. End to end, counting from the moment the requests
are sent:

| 4 requests at once | While generating | End to end, including the time to the first token |
|---|---|---|
| Code, 2K-token prompts | 467 | 312 (all four first tokens by 2.1 s) |
| Code, 32K-token prompts | 335 | 41 (all four first tokens by 43.3 s: four 32K prompts take most of the time) |
| Code edits | 472 | 405 |

These come from the same sweep as the README table: `summary.json`'s `combined`,
`aggregate_with_ttft` and `ttft_last` of the `gen-code-4s-2K`, `gen-code-4s-32K` and `edit-4s` cases.

The prose figure at 32K in [the chart](assets/ferrite-bench.png) rests on 2 clean replies out of 5 (the other three partly repeated the prompt),
so read it as a low-sample point.

## What these numbers don't tell you

- **Speculative decoding speed depends on the text.** Repetitive text and code edits run several times faster than
  fresh prose. Your workload will land somewhere in between.
- **Sampling is slower than greedy.** With temperature above 0, fewer drafted tokens are accepted.
- **One pair of machines.** Thermal state, memory fragmentation and driver versions move results by a few percent.
  Run-to-run spread is shown wherever it was measured.
- **Single Spark.** Most numbers here are two-node. On one Spark (`scripts/serve.sh` defaults, the same sweep),
  generation runs at 58-67% of the two-node speed:
  - code: 167 tok/s for 1 request and 287 for 4 at once (two Sparks: 267 and 467);
  - code, 4 requests, counting the time spent on prompt processing as well: 208 tok/s on one Spark, 312 on two;
  - code edits: 376 for 1 request and 276 for 4 (two Sparks: 564 and 472);
  - prompt processing: ~3,030 tok/s at 2K and ~1,150 at 120K.

## Reproducing

The published figures come from `tools/bench/sweep.sh`, which serves with the defaults above on a test port, runs every
case and writes `docs/bench-data/` (raw runs and `summary.json`) and the chart's data. It starts the two-node server
itself, so it needs two Sparks and node B's ssh name: `FERRITE_PEER=<node B> tools/bench/sweep.sh` on node A
([TP2.md](../TP2.md)). On one Spark, start `scripts/serve.sh` and run `sweep.py` against it, as below.

The benchmark clients are in `tools/bench/`. With a server running:

```bash
# the whole published sweep against a running server (about 20 minutes), medians in sweep/summary.json
python3 tools/bench/sweep.py --url http://localhost:8000/v1 --out sweep
# one case: code, 4 requests at once, 2K context, 500 tokens, median of 3 runs
python3 tools/bench/quickbench.py --url http://localhost:8000/v1 --mode code --tokens 500 --context 2000 --streams 4 --runs 3
python3 tools/bench/edit.py --url http://localhost:8000/v1 --streams 1
```

`quickbench.py` without `--mode` opens an interactive menu. With several streams it prints two rates. The "decode
aggregate" is the figure the tables quote. The "wall" rate counts the prompt processing before the first tokens
too, so it is lower. `edit.py` uses CPython 3.12.3's sources whatever Python runs it: it downloads them once and
checks them by sha256.

Please share results from your own Sparks in an [issue](https://github.com/BHCC2025/ferrite/issues), including the
Ferrite version or commit and the driver version.
