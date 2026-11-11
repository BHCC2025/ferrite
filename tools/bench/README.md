# Benchmark clients

Plain Python 3, standard library only. Each talks to any OpenAI-compatible endpoint.

- `quickbench.py`: generation and prompt-processing speed at a chosen context size, with one or several
  concurrent requests. `QB_SEED=<n>` fixes the prompts, so runs on different builds see the same text.
  Prose replies are classified as clean, partial echo or echo.
- `edit.py`: agent-style code edits. A source file goes in the prompt, with an instruction that keeps most of it;
  the reply is the whole edited file.
- `lanes.py`: parallel lanes of one job. N streaming requests sent together that start with the same text (a
  system prompt and a shared block of source files, 2K or 16K tokens) and end with a short task of their own, about
  800 tokens of code each. Like such a client it reads `/health` first and never sends more than `max_batch_size`
  requests at once. Reports total and per-lane output tok/s, prompt tokens computed and cached, and the lowest
  memory headroom, as medians of 3 sets after a warm-up. `--gate` checks that each lane's text equals the same
  request sent alone, fix round included.
- `cleansum.py`: summarizes prose speed over replies that are clean in every run given.
- `sweep.py`: the published sweep against a running server: prompt processing at 2K-120K, generation at 2K and
  32K (code and prose, 1 and 4 requests) and code edits, on fixed seeds, medians of 3. Writes the raw runs, a
  summary and the chart data for `tools/gfx/bench.py`. `sweep.py --plan` lists the cases.
- `sweep.sh`: starts the two-node server with its defaults on a test port, runs `sweep.py` (about 20 minutes)
  and stops it again.

The method behind the published numbers is in [docs/benchmarks.md](../../docs/benchmarks.md).
