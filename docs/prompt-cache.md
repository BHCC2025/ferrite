# Prompt cache

Coding agents resend the whole conversation on every turn, and several agents usually start from the same long
system prompt (rules, tool definitions, project files). Ferrite's prompt cache keeps what the server has already
computed, so those tokens are not processed again. It is on by default in the launch scripts (`FERRITE_CACHE=0`
turns it off) and works with DFlash2 drafting, the default.

## Results

Four agents at once on two Sparks (`tools/bench/agents.py`: a shared ~13K-token system prompt, then 2-3 turns each
of code, edit and tool-call tasks, greedy), cache off against cache on:

| | Cache off | Cache on |
|---|---|---|
| Prompt tokens computed | all | 74% fewer (thinking on: 86% fewer) |
| Time to first token, first turn | 16.4 s | 5.1 s |
| Time to first token, later turns | the whole history is computed again | about 0.4-0.6 s |
| Combined output over the run, thinking off | 46.5 tok/s | 92 tok/s |
| Combined output over the run, thinking on | 36.9 tok/s | 108 tok/s |

The first turn gains because the shared system prompt is computed once for all four agents. Later turns resume from
the previous turn instead of computing the whole history again. `tools/bench/agsum.py` turns the `--json` outputs of
`agents.py` into these tables.

## How it works

**Snapshots at turn boundaries.** The server marks the end of the system message and the start of every assistant
turn (`<|im_start|>assistant\n`). When a prompt chunk ends on one of those points, the slot's state is saved:

- the recurrent state of the 48 Gated DeltaNet layers (about 77 MB per node), which cannot be rewound, so it has to
  be saved where a later request may resume;
- the DFlash2 drafter's window (about 21 MB);
- the token ids up to that point.

The attention KV rows stay where they are. A slot keeps its first snapshot and its last two.

**Longest-match resume.** A new request resumes from the longest snapshot whose tokens equal the start of its prompt.
Three sources are checked, in this order of preference:

1. a free slot's own snapshots (nothing to copy);
2. a parked conversation;
3. a shared-prefix entry.

With fewer than 1024 matching tokens the request starts cold. The response reports the reused tokens in
`usage.prompt_tokens_details.cached_tokens`.

**Parking.** When a slot is needed for another conversation, the old one's KV rows and snapshots are copied into a
park pool instead of being dropped. A later turn of that conversation restores them into a free slot, which takes a
few milliseconds of GPU copies. The pool is allocated once at start, in KV blocks of 2048 tokens (32 KB a token per
node): 12 GB on one Spark (about 90K parked tokens) and 24 GB on each node of two (`FERRITE_PARK_GB` overrides). As a
backstop the engine shrinks it to keep `FE_PARK_RESERVE_GB` (20 GB) of memory available. The least recently used
entries make room for new ones.

**Shared prefix.** The first request that computes a system prompt of 1024 tokens or more also stores it as a prefix
entry (at most `FE_PREFIX_ENTRIES`, default 4). New conversations with the same system prompt start from it. Requests
that arrive while that prefix is still being computed wait for it, so agents that start together compute it once.

**Two nodes.** Node A's scheduler makes every decision. Each save, restore and park is mirrored to node B, which
applies it to its half of the model.

## Same text, cached or not

A reply resumed from the cache is the same, token for token, as the same request computed in full with the cache on.
Three rules make that hold:

- **Same chunk cuts.** Prompts are cut into chunks at the snapshot points in both paths, and the 32-row grid of the
  chunked Gated DeltaNet kernel restarts at each point. A restored state is therefore bitwise the state the full
  computation would have reached.
- **Resume only at snapshot points.** A request never resumes from the live state after a reply's generated tokens:
  generated rows and prompt rows round differently. The previous reply is computed again as part of the next
  prompt, about 0.1-0.2 s for 400 tokens.
- **No cut near the prompt end.** A point within 16 rows of the end is neither cut at nor resumed from, because such
  a short last chunk would run on the decode kernels instead of the prompt kernels. A turn therefore resumes from the
  turn before it.

Checked on two nodes: every agent turn matched a run with the same cuts and no resume (thinking on and off, and 6
conversations sharing 4 slots, with parking). Fresh single-turn prompts gave the same bytes with the cache on and off.
Turning the cache off changes where long prompts are cut, which can move a late near-tie.

## Settings and limits

| Setting | Default | |
|---|---|---|
| `FERRITE_CACHE` (`scripts/serve.sh`, `tools/serve-tp2.sh`) | `1` | `0` turns the cache off; `fe-server` started by hand takes `FE_CACHE=1` |
| `FERRITE_PARK_GB` (the engine reads `FE_PARK_GB`) | `12` on one node, `24` on two | the park pool per node, allocated at start |
| `FE_PARK_RESERVE_GB` | `20` | memory the pool always leaves available: a smaller pool when the machine runs other things |
| `FE_PREFIX_ENTRIES` | `4` | shared-prefix entries kept |
| `FE_CACHE_TRACE` | off | `1` logs one line per cache decision |
| `FE_CACHE_RESTORE` | `1` | `0` keeps the same cuts and snapshots but never resumes: the uncached reference for tests |

- **4 requests at once.** Ferrite runs at most 4 requests at a time with DFlash2 drafting. More requests wait in a
  first-in, first-out queue. A waiting agent's conversation is parked and restored when it gets a slot.
- **DFlash2 only.** With the model's MTP head loaded (`FERRITE_DRAFT=mtp` or `auto`), there are no snapshots,
  parking or shared prefixes: the server only reuses an identical prefix in the same slot, and says so once in its
  log.
- **Memory.** The park pool comes on top of the model and the KV cache. Measured under agent load: one Spark with
  4 x 128K slots and the 12 GB pool keeps about 34 GB available (a 24 GB pool would go down to about 20 GB); two
  Sparks with 24 GB pools keep 55-58 GB available per node.
