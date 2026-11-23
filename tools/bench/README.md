# Benchmark clients

Plain Python 3, standard library only. Each talks to any OpenAI-compatible endpoint: start a server first, then
point a client at it with `--url http://<server>:8000/v1`. One runnable example (code, 4 requests at once, 2K
context, 500 tokens, median of 3 runs):

```bash
python3 tools/bench/quickbench.py --url http://localhost:8000/v1 --mode code --tokens 500 --context 2000 --streams 4 --runs 3
```

The method behind the published numbers, and what "clean", "partial echo" and "echo" mean for a prose reply, is in
[docs/benchmarks.md](../../docs/benchmarks.md#how-speed-is-measured).

## Speed

- `quickbench.py`: generation and prompt-processing speed at a chosen context size, with one or several
  concurrent requests. `QB_SEED=<n>` fixes the prompts, so runs on different builds see the same text.
  Prose replies are classified as clean, partial echo or echo. Without `--mode` it opens an interactive menu.
- `edit.py`: agent-style code edits. A source file goes in the prompt, with an instruction that keeps most of it;
  the reply is the whole edited file.
- `cleansum.py`: summarizes prose speed over replies that are clean in every run given.
- `sweep.py`: the published sweep against a running server: prompt processing at 2K-120K, generation at 2K and
  32K (code and prose, 1 and 4 requests) and code edits, on fixed seeds, medians of 3. Writes the raw runs, a
  summary and the chart data for `tools/gfx/bench.py`. `sweep.py --plan` lists the cases. It takes about 20 minutes.
- `sweep.sh`: starts the **two-node** server with its defaults on a test port, runs `sweep.py` and stops it again.
  It needs two DGX Sparks and node B's ssh name: `FERRITE_PEER=<node B> tools/bench/sweep.sh` on node A. On one
  Spark, or any other setup, start the server yourself and run `sweep.py` against it.

## Several requests and agents

- `lanes.py`: parallel requests of one job (lanes).
  - What it sends: N streaming requests together that start with the same text (a system prompt and a shared block
    of source files, 2K or 16K tokens) and end with a short task of their own, about 800 tokens of code each.
  - How: like such a client it reads `/health` first and never sends more than `max_batch_size` requests at once.
  - What it reports: total and per-lane output tok/s, prompt tokens computed and cached, and the lowest memory
    headroom, as medians of 3 sets after a warm-up.
  - `--gate` checks that each lane's text equals the same request sent alone, fix round included.
- `agents.py`: a multi-agent workload. N agents run at once and share one ~13K-token system prompt; each then does
  its own task (code, edit or tool call) over 2-3 turns, and later turns resend the history. Greedy, thinking off
  unless `--think`. Prints one summary line; `--json` saves the run. Its prompts quote a fixed copy of repository
  files in `tools/bench/data/agents-corpus`.
- `agsum.py`: turns the `--json` outputs of `agents.py` into markdown table rows (output speed, time to first token
  per turn, prompt tokens computed and cached). The tables of [docs/prompt-cache.md](../../docs/prompt-cache.md)
  come from it.
- `agprefix.py`: from the same `--json` outputs, the prompt tokens computed against what per-slot reuse and a
  shared system prefix would compute.
- `agentturns.py`: one conversation behind a ~5K-token system prompt that grows for `--turns` turns (each turn
  appends the previous reply and a ~1K-token tool result), greedy, streamed. Prints per turn the prompt tokens, the
  cached tokens, the time to first token and a hash of the reply.

## Checks

- `modes.py`: greedy text must not depend on the server's prompt cache (`FERRITE_CACHE`) or its drafting method
  (`FERRITE_DRAFT` `mtp` / `dflash` / `off`). It sends a fixed set of greedy requests to a fresh server and saves a
  hash per case (`--save ref.json`), or checks them against a reference made by another mode (`--compare ref.json`).
- `mixed.py`: greedy and sampled requests together. `repro` runs greedy file edits beside short sampled requests;
  `soak` loops a mix of request kinds for a set time with a fixed greedy probe every 60 s that must give the same
  text every time. Every request must succeed and `/health` must stay ok.

The Inkling-2-Mini client is not in this folder: `tools/bench-ink.py`
([docs/inkling-2-mini.md](../../docs/inkling-2-mini.md)).
