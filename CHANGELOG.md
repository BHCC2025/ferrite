# Changelog

Notable changes to Ferrite, newest release first. Every section below is a tagged release that is published and
available. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-11-23

### Added

- **A fourth model: Inkling-2-Mini** (Thinking Machines Lab, Apache-2.0; the checkpoint as published, NVFP4 and MXFP8 experts and FP8
  attention, 20 GB) on **one card**, an RTX PRO 4000 Blackwell (24 GB) or an RTX 5060 Ti (16 GB), with a 128K context on either:
  `fe-server --engine ink` (`--features ink`), crates `fe-ink` and `fe-inkk`, kernels `k-ink`, `k-inkm` and `k-draft`,
  [docs/inkling-2-mini.md](docs/inkling-2-mini.md). New `scripts/fetch-inkling.sh` (the files pinned to the tested revision, sizes and
  sha256 checked), `tools/serve-ink.sh` (the layout comes from the card's memory) and `tools/bench-ink.py`.
  - The model's own draft head gives lossless speculative decoding: greedy text is the same with it and without it, and the same whatever the
    prompt chunk, the rows beside a token or the cache state (`ink-run selftest` checks all of it, and a 64K and a 128K prompt).
  - Both layouts keep a 6-bit K/V cache mapped as the context grows (1.6 GiB at 128K; `INK_KV=bf16` is the opt-out on the 24 GB card). The 24 GB
    layout keeps the weights as published. The 16 GB layout requantises layers 20-23 to NVFP4 when the model
    loads, parks the experts that do not fit in pinned host memory (more of them as the context
    grows, fewer again when a request ends) and keeps the FP8 draft head at every context up to 128K.
  - Accuracy against the published model (free-running KL on code, prose and a 16K text): 0.004 / 0.008 / 0.009 on the 24 GB layout,
    0.005 / 0.011 on the 16 GB layout.
  - One request, temperature 0, draft head on / off: code 138 / 83 tokens/s on the RTX PRO 4000 and 110 / 69 on the RTX 5060 Ti, an agent
    turn 126 / 83 and 99 / 69, prose 84 / 83 and 69 / 70. A 126K-token prompt prefills at 703 tokens/s on the RTX PRO 4000 and 413 on the RTX
    5060 Ti and then decodes at 117 and 64 tokens/s with the draft head (60 and 45 without it); 8K, 32K and 64K contexts, the effect of the text on
    drafting and the bf16 K/V opt-in in [docs/inkling-2-mini.md](docs/inkling-2-mini.md).
  - The server parses the model's typed replies: reasoning goes to `reasoning_content` and the answer to `content`; tool calls become OpenAI
    `tool_calls` (through the model's own chat template, byte for byte like the Hugging Face one on 28 conversations, with markup spelled in message
    text disarmed); stop strings cut at the exact token. Defaults are the publisher's (temperature 1.0, reasoning effort 0.9) with a 32,768-token cap
    for requests without `max_tokens` (`FE_DEFAULT_TEMPERATURE`, `FE_DEFAULT_REASONING_EFFORT`, `FE_DEFAULT_MAX_TOKENS`); the `inkling-2-mini` profile
    and four Inkling checks are in `tools/conformance`.
  - Known gaps: text only (an image in a request is ignored), one request at a time, no prompt cache.

### Changed

- The draft head's fold tables are computed from two tensors of the checkpoint on the first start and kept beside the model.
- Every engine's chat completion reports `usage.completion_tokens_details.reasoning_tokens` (the tokens inside the reasoning block).
- `fe-cuda`: `Context::device_id`. Version 0.9.0.

## [0.8.1] - 2026-11-21

### Fixed

- **DeepSeek-V4.1-Flash Coder: Engram rows are read in parallel.** Every token reads 48 rows of the 111 GB Engram
  table. They were read one after another, and a row that is not in the page cache waits about 70 us on the SSD, so a
  pass over six drafted tokens could wait some 20 ms before the cards started. Rows in the page cache are now read at
  once on the engine thread and the others together on a pool of reader threads (`DSC_ENGRAM_THREADS`, default 128).
  Eight prompts the server had not seen before, fresh start, temperature 0: 74 -> 87 tokens/s on three cards and
  88 -> 105 on four, with the same text as 0.8.0. Figures in [docs/deepseek-coder.md](docs/deepseek-coder.md)
  re-measured with 0.8.1.

### Changed

- **Source only: Ferrite ships no prebuilt binaries.** `scripts/install.sh` clones the release's tag and builds it
  (rustup, the pinned toolchains and `cargo build`, a few minutes the first time); `--source` is still accepted and
  does the same. The packages attached to releases 0.1.0 to 0.8.0 were withdrawn and `scripts/package-release.sh`
  removed. An install unpacked from one of those packages is upgraded by the same command.

## [0.8.0] - 2026-11-20

### Added

- **A third model: DeepSeek-V4.1-Flash Coder** (the ISTA-DASLab GSQ-RCO Coder IQ2_XXS GGUF, 72 GB) on three 24 GB
  cards, or four with a 16 GB card added: `fe-server --engine dsc` (`--features dsc`), crate `fe-dsc`, kernels
  `k-ds`, `k-dspark` and `k-engram`, [docs/deepseek-coder.md](docs/deepseek-coder.md). New `scripts/fetch-deepseek.sh`
  (the model, the DSpark draft model and the Engram table, ~191 GB, checked against pinned sizes and sha256; it
  writes the tokenizer and serving files with the new `dsc-mkmodel`) and `tools/serve-dsc.sh`.
  - The main card runs the dense part and its share of the experts; the other cards hold experts only and get each
    layer's input card to card. Which experts are on the cards comes from a placement heat measured on real
    workloads (`crates/fe-dsc/heat/coder.cnt`). On three RTX PRO 4000 Blackwell about 1,440 experts stay in pinned
    host memory and are copied to a card when picked; a fourth card holds them all.
  - DSpark speculative decoding and drafts copied from the context, lossless (greedy output is the same as plain
    decoding).
  - One request, thinking off, three / four cards: Python 94 / 109, Rust 96 / 115, HTML and JavaScript 108 / 130,
    chat 43 / 69 tokens/s.
  - `FERRITE_SLOTS=2-4`: several requests at once, each with the same text as alone; one drafter pass serves all of
    them. Four code requests total 116 / 141 tokens/s on three / four cards: on this model several requests add
    little over one.
  - The protections of [docs/harness.md](docs/harness.md) on every request, also with several at once: the
    reasoning cap (the engine closes the block itself), the loop breaker, the repeated-sentence rule, the first-token
    stop ban, enforced `tool_choice` in the DSML tool-call format, the watchdog and the finite check. A tool call
    that starts to repeat itself is rewound to where the cycle began and continued with sampling
    (`loop_rules.call_recover`, on in `crates/fe-server/sampling.deepseek-v4.1-flash-coder.json`).
  - The tokenizer is built from the GGUF with the think and DSML marks as single tokens (split into pieces they
    made replies run away and broke tool calls).
- `tools/conformance/profiles/deepseek-v4.1-flash-coder.json`: the conformance suite for this model.
- GGUF GPU kernels for the IQ2_XXS, IQ2_XS, IQ1_M and MXFP4 types (`k-gguf`, `fe-ggq`). AVX2 CPU kernels for the
  same types are in `fe-ggq` too; the DeepSeek engine does not use them by default (copying a spilled expert to a
  card measured faster).

### Changed

- The server's guard can rewind: an engine that holds back the tokens of a building tool-call cycle can take them
  back and continue with sampling. Engines that do not enable it behave as before.

## [0.7.2] - 2026-11-19

### Added

- **The `rtx-pro-4000` preset: Qwen3.8-27B on one RTX PRO 4000 Blackwell (24 GB).** New `tools/serve-pro4000.sh`
  and [docs/rtx-pro-4000.md](docs/rtx-pro-4000.md). `scripts/serve.sh` and `scripts/ferritectl.sh start` pick it by
  the card's name, and `scripts/ferritectl.sh start rtx-pro-4000` names it. It is the arrangement of the 16 GB
  preset (`tools/serve-5080.sh`, unchanged) with the extra memory put to use:
  - **Prompt-weight layers stay on the card** (`FERRITE_RESIDENT`, default `auto`; `FE_PROMPT_RESIDENT` in the
    engine, default 0). With a 128K context 27 of the 64 layers are resident, so a prompt's pass over the bus moves
    only the others, and the streamed layers upload a ring ahead of the layer at work. On the card, against the
    16 GB preset with 0.7.0: the first token of a short prompt comes after 0.70 s instead of 1.19 s, a 2K prompt is
    processed at about 2,700 tokens/s instead of 1,599, and a 2,118-token prompt reaches its first token in 0.79 s
    instead of 1.23 s. The server uses 22.2 GB of the card. Where a layer's bytes are kept does not change what is
    computed.
  - `SLOTS` 2-4 share one pool of 4 x 32,768 tokens and keep 47 copy rows a request.
- **Decode plans for 70-SM cards** (`crates/fe-exl3/src/plans.rs`, tuned on the RTX PRO 4000 Blackwell at its
  145 W limit). Such a card used the DGX Spark's table before. Same binary, old table against new: plain decoding
  36.5 to 40.3 tokens/s, code at 2K context 183 to 197, a 500-token edit 464 to 484. With the whole preset, against
  the 16 GB preset with 0.7.0: code at 2K / 16K / 56K context 184 / 160 / 137 to 200 / 171 / 143 tokens/s, edits
  462 to 501, plain decoding 40.1. **This changes numerics, on 70-SM cards only**: the plans fix the order of the
  sums. KL divergence against 0.7.0 on the card is 0.0004 (code), 0.0010 (text) and 0.0005 (16K context), and
  against BF16 it is 0.0208 / 0.0427 / 0.0229 (0.7.0: 0.0207 / 0.0427 / 0.0230). A greedy reply can differ from
  0.7.0's at a token whose two best candidates are nearly tied (4 of 10 test requests did). `FE_EXL3_TABLE=gb10`
  selects the 0.7.0 numerics. Cards with 84, 36 or 48 SMs keep their tables.
- **Prompt kernels for 70 SMs.** The NVFP4 GEMMs take the mbarrier pipeline and prompt groups run the next chunk's
  GDN projection beside the recurrence. A 16K / 56K / 110K prompt takes 5.06 / 25.49 / 71.92 s instead of
  5.41 / 26.85 / 74.62 s, with the same bits.
- **A 262,144-token context starts on the 24 GB card** (`CTX=262144`: 13 resident layers, 22.2 GB in use). A
  10-fact needle test over a 240,524-token prompt found 10 of 10; the prompt took 279 s (863 tokens/s) and each
  further question 3 s from the prompt cache. This is a feasibility measurement: the default stays 131,072.

### Changed

- A correction to the 0.7.1 entry, which says the Arc B70 protections were each tested "with greedy and sampled
  requests": that holds for the reasoning cap, the forced repetition and `tool_choice` `required`. The
  named-function call was confirmed to return a valid call; the request modes of that check were not recorded.
- **What this release was verified on.**
  - One RTX PRO 4000 Blackwell 24 GB, on a build of these changes on top of 0.7.0 (the 0.7.2 source itself, which
    also holds the 0.7.1 changes, was not rebuilt on the card). Unit and end-to-end tests: 141 pass, 1 ignored; the
    scheduler's 12 pass; `ferrite selftest` passes. With `FE_EXL3_TABLE=gb10` ten greedy requests give the text of
    the 0.7.0 build, with this preset and with the 16 GB preset: resident layers, the ring ahead, the prompt
    kernels and the arrival fix change no text. With the card's table 6 of the 10 are identical. Drafting on and
    off give the same text, a repeated request the same text, one slot and four slots the same text
    (`tools/bench/lanes.py --gate`: 0 mismatches). A 5-fact needle test over 120,287 tokens found 5 of 5. The
    protections fire.
  - The conformance suite on that build with the card's table: quick tier 53 runs, 1 failed (`special_content`,
    streamed with thinking); heavy tier 12 runs, 1 failed (`long_write_big`). Both cases sample their reasoning,
    and both also fail with the 0.7.0 numerics under other seeds. `long_write_big` was then run with 20 seeds a
    table on three cards: 18 of 20 passed with the card's table and 16 of 20 with the 0.7.0 numerics (Fisher exact
    test, two-sided p = 0.66); with 6 earlier runs a table it is 21 of 26 for each. The failures are of the same
    two kinds with either table (a staged write left open after 6 turns; an edit call with a malformed argument).
    The cards reached 88-90 C with no thermal slowdown.
  - DGX Spark (aarch64), the published source: build and unit tests pass. One node: the package gives the same
    greedy text as the 0.7.1 package on four prompts. `tools/bench/lanes.py --gate` passes with 0 mismatches over
    1, 2 and 4 requests and 2K and 16K shared prefixes, and its texts equal those of the 0.7.1 package. The
    arrival fix makes no difference there: the prefix was shared in every set before and after, four requests
    with a 16K shared prefix total 114-116 tokens/s (0.7.1: 114-115; series of 5 sets, alternating), and the last
    first token comes after 7.2 s with both. Two nodes, started with `tools/serve-tp2.sh`: a greedy reply is
    identical to the 0.7.1 package's, and a request without sampling fields runs with its mode's profile. The
    speed runs, the long gates and the conformance suite were not run again on the Sparks.
  - Not built and not run for this release: the RTX 5080 and RTX 5060 Ti 16 GB (their preset and decode tables are
    unchanged; they get the arrival fix), the Intel Arc Pro B70 and Flash-Next. The shared code that changed is
    the arrival fix: the server now tells an engine when a request arrives (`Engine::incoming`), which the Arc
    and Flash-Next engines ignore, and the 27B scheduler waits on it.

### Fixed

- **Requests sent together share their prefix.** A worker woken from idle started its first forward once no
  request had arrived for 5 ms (20 ms at most). Rendering and tokenizing a 16K-token prompt takes longer than
  that, so of several requests sent at the same moment the first could start alone, and the others then computed
  the text they share again. On the RTX PRO 4000, two requests with a 16K shared prefix: in 2 of 5 sets nothing
  was shared and the last first token came after 22.5 s instead of 7.0-7.4 s. The server now holds a marker from
  the moment it has a request's body until the request is submitted, and a worker woken from idle waits while one
  is held, 250 ms at most (`FE_ARRIVAL_MAX_MS`). A lone request is not delayed. After the fix 5 of 5 sets of two
  and of four requests share the prefix. Four requests with a 2K shared prefix on that card total 255 tokens/s
  with this release's preset; the 16 GB preset gave 186 (measured with 0.6.0), and the difference includes the
  preset's other gains. This is in the scheduler every Qwen3.8-27B target uses.

## [0.7.1] - 2026-11-19

### Added

- **The Intel Arc Pro B70 engine has the protections of the other engines.** `--engine xe` now uses the guard the
  27B and Flash-Next engines share: the reasoning cap, with the block closed by the engine
  (`FE_MAX_REASONING_TOKENS`), the loop breaker and the repeated-sentence rule (`FE_LOOP_BREAKER`, `FE_LOOP_RULES`),
  never a stop token as the first token of a reply, `tool_choice` `required` or a named function limited to the
  offered names, and a non-finite check on the logit rows the host picks from (`FE_FINITE_CHECK`). A protection acts
  at the token where it fires: the rows a verify window accepted past that token are not kept. What fired is
  reported as on the other engines (the request-capture meta file). In 0.7.0 this engine ignored these settings.
- **Sampling defaults per mode on the Arc B70.** New `crates/fe-server/sampling.qwen3.8-27b-xq.json`, with the
  values of the 27B file. `scripts/serve-xe.sh` uses it when the XQ folder has no `sampling.json`
  (`FE_SAMPLING_JSON=<file>` names another). A request that leaves its sampling fields out now gets the profile of
  its mode, `x-sampling-effective` reports it, and `temperature: 0` keeps the greedy path with no penalty added.
- **[docs/arc-b70.md](docs/arc-b70.md#recommended-settings)** has a "Recommended settings" section.

### Changed

- The Arc engine crates take a per-request gate from the server (`fe_xe::specgen::Spec` and `fe_xdf::spec::DfSpec`
  have a `gate` field; `None`, the default, is the 0.7.0 behaviour).
- **What this release was verified on.**
  - One Intel Arc Pro B70 32 GB at the documented settings (180 W cap, 2000 MHz), XQ 5.0 bpw, DFlash2 drafting,
    128K context, one request at a time: greedy replies to a six-prompt set (code, prose, chat, edit, tool call,
    thinking) are byte-identical to 0.7.0. Code at 2K context is 168.8 tok/s before and after. A reply that repeats
    its prompt went from 257.8 to 248.5 tok/s, the loop detector's own cost on repetitive text. The protections
    fire: with `FE_MAX_REASONING_TOKENS=48` the block is closed at 48 reasoning tokens and the answer follows, a
    forced repetition is ended by the loop breaker, and `tool_choice` `required` and a named function return a
    call to an offered function, each with greedy and sampled requests. `x-sampling-effective` shows the profile.
    The conformance suite's quick tier: 53 runs, 0 failed. The 100 unit tests of the Arc build pass.
  - Not exercised on the Arc B70 with protections firing: more than one slot (`FERRITE_SLOTS` above 1), MTP
    drafting (`FERRITE_DRAFT=mtp`) and plain decoding (`FERRITE_PLAIN=1`). The heavy conformance tier, a soak and
    KL divergence were not run for this release.
  - DGX Spark (aarch64), the published source: build and unit tests pass. The package on one node gives the same
    greedy text as the 0.7.0 package on four prompts. On two nodes, started with `tools/serve-tp2.sh` at its
    defaults and no `sampling.json` in the model folder: a request without sampling fields and with thinking off
    runs with the non-thinking profile, a `temperature: 0` request gets no presence penalty, and a greedy reply is
    identical to the 0.7.0 package's on two nodes. The speed runs, the long gates and the conformance suite were
    not run again on the Sparks.
  - RTX cards and Flash-Next: not built and not run for this release. Their engine code is unchanged since 0.7.0.

### Fixed

- **The two-node DGX Spark launcher loads the sampling file.** `tools/serve-tp2.sh` now calls `fl_sampling` like
  `scripts/serve.sh`. In 0.7.0 a two-node server whose model folder had no `sampling.json` ran with the built-in
  defaults; it now gets the 27B profiles and protections from `crates/fe-server/sampling.qwen3.8-27b.json`.

## [0.7.0] - 2026-11-19

### Added

- **Sampling defaults per mode.** The server reads `sampling.json` from the model folder: one set of values for
  replies that think and one for replies that do not, applied to the fields a request leaves out. A field the
  client sends always wins. The files for both models hold the base models' published settings (thinking: temperature
  1.0, top_p 0.95, top_k 20; non-thinking: 0.7, 0.8, 20 and presence penalty 1.5) and are in `crates/fe-server/`
  (`sampling.qwen3.8-27b.json`, `sampling.qwen3.8-flash-next.json`); `scripts/serve.sh` and `tools/serve-fnext.sh`
  use them when the model folder has none, and `FE_SAMPLING_JSON=<file>` names another. The values a request ran with
  are written to the log and returned in the `x-sampling-effective` response header, with the origin of each.
- **A `temperature: 0` request keeps the fast greedy path** (`greedy_keeps_fast_path`, on by default;
  `FE_GREEDY_FAST_PATH=0` turns it off): the defaults add no presence, frequency or repetition penalty to it, and a
  penalty the client sends itself still applies. Qwen3.8-27B on one RTX PRO 4000 Blackwell 24 GB with the defaults
  live: code at 2K context 184 tok/s and edits 462 tok/s, the same as without the defaults; with the non-thinking
  presence penalty applied to such requests they ran at 147-153 and 152-157.
- **Protections for coding agents, in both engines.** One guard per request, shared by the 27B and the Flash-Next
  engine, is fed the accepted tokens one at a time:
  - a reasoning block still open after 8,192 tokens is closed by the engine and the model answers
    (`FE_MAX_REASONING_TOKENS`, 0 = off);
  - a loop breaker for exact repeating cycles: it closes a reasoning block; in an answer it boosts the sampling for
    96 tokens and ends the reply with `finish_reason: "length"` if the cycle returns; tool-call arguments are held
    to a far stricter rule (`FE_LOOP_BREAKER`, `FE_LOOP_RULES`);
  - a repeated-sentence rule for replies that keep coming back to the same sentences without a verbatim cycle,
    never inside a code block or a tool call;
  - the first token of a reply is never a stop token;
  - `tool_choice` `required` and named functions are enforced: the reply starts inside the call, and with several
    tools the name is limited to the ones offered;
  - a watchdog: a reply that produces no token for 60 s after its first one is ended (`FE_NO_TOKEN_TIMEOUT`), with an
    optional wall-clock limit (`FE_REPLY_WALL_LIMIT`); a pick from non-finite logits ends the request with an error
    instead of text (`FE_FINITE_CHECK`).

  A protection acts at the token where it fires: the tokens a verify step accepted past it are given back. With
  protections firing, greedy replies are identical with drafting on and off, and alone and with four requests at
  once (both engines, RTX PRO 4000 Blackwell 24 GB: one card for the 27B, two for Flash-Next), and a request on
  which none fires gets the tokens it got before. Drafting goes on after the cap closes a block: with the cap set to
  48 tokens and a long code answer after it, the 27B decoded the answer at 612 characters a second against 153 with
  drafting off, and Flash-Next at 129-131 tok/s against 94, the same text in both.
- **Tool-call arguments stream as they are written** (Qwen XML calls). On Flash-Next an 86-line file write arrives
  as 414 argument deltas.
- **Request capture** for replay (`FE_DUMP_REQUESTS=<dir>`: each chat-completions body and a small `.meta.json`
  with the outcome), and an **SSE keep-alive** comment every 10 s of silence, from the moment a request is queued.
- **[docs/harness.md](docs/harness.md)**: agent harness settings that were tested end to end, with a Pi extension
  that gives the model's shell commands a default and a maximum run time
  ([docs/harness/pi/bash-timeout.ts](docs/harness/pi/bash-timeout.ts)).
- **An API conformance suite**, [tools/conformance](tools/conformance/README.md): tool calls, streaming, stopping,
  loops, sampling defaults, caching and liveness against any OpenAI-compatible server, Python standard library only,
  one profile file a model. On this release (RTX PRO 4000 Blackwell 24 GB, fresh servers):
  - Qwen3.8-27B, one card: quick tier 53 runs, 0 failed; heavy tier with six runs a loop case 12 runs, 0 failed.
  - Qwen3.8-Flash-Next, two cards: quick tier 53 runs, 3 failed, all three the model: two special-content runs
    with thinking on, where the model ended its reply inside the reasoning block, and one translation prompt that
    ran to the token limit (see [docs/flash-next.md](docs/flash-next.md#the-coder-file-is-for-code)); heavy tier
    12 runs, 0 failed.
- **Recommended settings** in [docs/model-card.md](docs/model-card.md#recommended-settings) and
  [docs/flash-next.md](docs/flash-next.md#recommended-settings): what the server applies by default, the greedy
  rule, the protections and their switches.

### Changed

- **Non-thinking replies get the non-thinking settings.** A request that leaves its sampling fields out used to run
  with the values of `generation_config.json` (temperature 1.0, top_p 0.95, top_k 20) in either mode. With the
  shipped `sampling.json` a reply that does not think now runs at temperature 0.7, top_p 0.8, top_k 20 and presence
  penalty 1.5.
- **Flash-Next applies the same protections and defaults as the 27B**, through the same code. On Flash-Next a
  sampled request decodes one token a step (about 72 tok/s on two RTX PRO 4000 Blackwell 24 GB, against 173-194
  for a greedy request with MTP drafting), so a client that wants the drafting speed sends `temperature: 0`.
- **[docs/flash-next.md](docs/flash-next.md) says what the Coder file is for.** Short coding prompts ended on their
  own in 176 of 176 runs; general-chat prompts ran to a 1,024-token limit in 39 of 112 (two RTX PRO 4000 Blackwell
  24 GB).
- Speed is unchanged where no protection fires. Qwen3.8-27B on one RTX PRO 4000 Blackwell 24 GB: code 184 tok/s at
  2K context and 160 at 16K, prose 66-78, edits 462, prompt processing 1,599 tok/s at 2K and 2,714 at 16K, first
  token of a short prompt after 1.19 s. Flash-Next on two: one code request 173-194 tok/s, one prose request
  132-136, four code requests 206-220 in total, first token after 0.12-0.13 s.
- **What this release was verified on.**
  - RTX PRO 4000 Blackwell 24 GB (x86-64), both engines: every gate and figure in this entry, run on this release's
    changes on top of the 0.6.0 code. The published source also carries the 0.6.1 fixes; that combination was built
    and unit-tested on x86-64 with both engines and was not run on these cards again.
  - DGX Spark (aarch64), the published source: build, unit tests and `ferrite selftest` pass. On one node: greedy
    text identical to 0.6.1 with the sampling file present and absent, with DFlash2 drafting, MTP drafting, drafting
    off and the prompt cache off; repeated requests identical; four requests at once identical to each alone; with
    protections firing, the same text with drafting on and off and with one and four slots; the non-finite check;
    the release gate's API checks; speed as in 0.6.1 (code 160 tok/s, four requests 286, edits 316, prompt
    processing about 2,500 tok/s at 2K). The conformance quick tier on one node: 53 runs, 2 failed, both at the
    default sampling: a `special_content` run with thinking on, where the model wrote an example call before the
    real one, and a `cut_off_call` run, where the model made a short `bash` call instead of the long write the case
    expects. On two nodes: greedy text identical to 0.6.1, repeated requests identical, four requests at once
    identical to each alone, speed as in 0.6.1 (code 268 tok/s, four requests 473); the other two-node checks
    (no sampling file, drafting off, protections firing, the conformance suite, the release gate) were not run.
  - Intel Arc Pro B70: not rebuilt and not run for this release. Its kernels are unchanged since 0.6.1, and the
    server code was type-checked with `--features xe` without building them.

### Fixed

- **Flash-Next: a sampled request could read another request's logits.** The copy of the logit row to the host is
  part of a captured step graph, and a graph captured for a greedy request was reused for sampled ones: a sampled
  request that followed a greedy one on the same slot drew its first token from the previous request's row (a wrong
  first token, or an empty one-token reply), and in two cases every token. Step graphs are now kept apart by that
  copy. Present in 0.6.0, where the regression check fails 82 of 100, 24 of 25 and 2 of 2 cases; all pass now (two
  RTX PRO 4000 Blackwell 24 GB, fresh servers).
- **Text that looks like tool-call markup inside a call argument** stays part of the argument when its tags are
  balanced.

## [0.6.1] - 2026-11-15

### Fixed

- **A crash under mixed traffic.** A request with `temperature` above 0 that arrived while other requests were
  drafting (several greedy file edits at once, say) could crash the engine. The server then answered every request
  with an error while `/health` still said ok. Sampled requests now copy only their own rows to the host, into a
  buffer sized for the step.
- **Greedy text no longer depends on the settings.** `FERRITE_CACHE=0` and every `FERRITE_DRAFT` mode (`dflash`,
  `mtp`, `auto`, `off`) give the same text for every prompt, including prompts with a system message and thinking
  turns. Before, a prompt was cut at its turn boundaries only with the cache on, which could change near ties.
  `tools/bench/modes.py` checks it.
- **Two Sparks with several requests are back to full speed.** Four code requests went from 444 to 467 tok/s and
  four edits from 426 to 472: the verify-row split for three or more streams now applies to one node only.
- **The server's memory levels off** under mixed traffic instead of growing by gigabytes an hour: captured GPU
  graphs are capped (`FE_GRAPH_MAX`, 128), and on Flash-Next the step-graph cache too.
- **The prompt cache no longer fails a request** when the conversation it needed had to leave the cache pool. The
  prompt is computed in full instead.
- **Clear errors instead of late or vague ones.**
  - A taken port fails in a second, before the model loads.
  - Kernels built for another card stop at start with the `FE_KARCH=... cargo build` command that fixes them, on
    both CUDA engines.
  - A missing or cut-off model file is named, and the launchers check the model folders before loading.
  - A prompt in the last positions of the context gets a 400 that says how long a prompt may be, not a 500.
  - `logprobs` / `top_logprobs` get a 400 (not supported) instead of a reply without them. An unknown message role
    gets a 400 that names it. Request bodies above 32 MB get a 413.
  - Flash-Next running out of card memory at load names the settings to lower.
- **Flash-Next stops a dropped prompt.** When a client disconnects during its prompt, the prompt stops between
  layers instead of running to the end. A dropped 100K-token prompt used to hold up the next request for about a
  minute.
- **The installer installs the latest release.** It used to ask for 0.5.2, which was never published, and fell
  back to a source build without saying why. Run over an older install it now upgrades, keeping the old folder as
  `~/ferrite-<version>`. It also no longer reports success when another program answers on the port.
- **Downloads recover by themselves.**
  - `scripts/fetch-models.sh` retries a failed transfer, fetches again any file whose size or sha256 is wrong, and
    deletes the partial files an interruption leaves.
  - It stops before downloading when the disk is too small, and follows symlinked model files.
  - `scripts/fetch-flashnext.sh` downloads in parallel ranges that resume and retry, so a dropped connection no
    longer ends the run.
- **Builds without `FE_KARCH`** compile the kernels for the card `nvidia-smi` reports on the build machine, and say
  so in a build warning. Before, they were built for the DGX Spark, and an RTX build failed only on the card.
- **Fixes for the RTX 5080 / 5060 Ti preset.**
  - `DRAFT=off` now turns drafting off.
  - `FE_KVQ=0` picks a context that fits.
  - A second launcher on a busy card stops with a message instead of waiting silently.
  - `SLOTS=abc` and similar bad settings are refused.
- **Smaller fixes.**
  - The server log has no colour codes when it is written to a file.
  - `fetch-models.sh` names the right launcher for the machine at the end.
  - Source builds are free of compiler warnings.
  - The release tarball's documentation links all resolve.

### Changed

- **Every launcher listens on `0.0.0.0` by default**, so other machines on your network can use the server:
  `scripts/serve.sh`, `tools/serve-tp2.sh`, `tools/serve-5080.sh` (RTX 5080 and 5060 Ti), `tools/serve-fnext.sh`
  and `scripts/serve-xe.sh`. There is no API key: use it on a trusted network or firewall the port.
  `FERRITE_HOST=127.0.0.1` keeps a server on its own machine. Each launcher says so when it starts.
- **The launchers behave the same way on every target.**
  - Before the model loads, they check the port, the GPU lock, the model files and the memory.
  - A busy card stops a second launcher at once and names the process holding it (`FERRITE_LOCK_WAIT=1` waits
    instead).
  - Locks are per card: the node lock is held shared, plus the card's own lock. A 27B server on one card and
    Flash-Next on another run side by side, while whole-node jobs still exclude both.
  - The launcher's process becomes the server, so Ctrl-C, `kill <pid>` or `kill $!` stop it and free the card.
- **`scripts/serve.sh` refuses to start when memory is short**, as the installer does, instead of only warning
  (`FERRITE_FORCE=1` overrides). On an RTX 50 card it starts the RTX preset by itself.
- **The RTX preset sizes the context to the card's free memory**: 128K on an empty card, and less (with a message
  saying so) when a desktop or another program holds part of it. A `FERRITE_CTX` that cannot fit is refused before
  loading.
- **When the engine stops**, after a crash or when the second node of a two-node server is gone, the server
  reacts. `/health` and new requests answer 503 with the reason, and the process exits with status 70 so a
  supervisor can restart it. On two nodes the other node's process exits too.
- **The prompt cache also works with MTP drafting** (`FERRITE_DRAFT=mtp` / `auto`) on every device
  (`FE_CACHE_MTP=0` turns that off).
- **Follow-up turns resume closer to the end of the previous one.** Turn boundaries sit on a 32-row grid, so an
  agent conversation recomputes only the last reply and the new message. On one Spark, 8 turns behind a
  6.5K-token system prompt took 9.3 s of time to first token in total before and 7.2 s now, with the same text.
  This moves where some prompts are cut, so greedy text can differ slightly from 0.6.0's. It is the same in every
  cache and drafting mode.
- **Flash-Next keeps its host memory flat over long runs**: fixed C-heap thresholds, on by default for Flash-Next
  (`FERRITE_MALLOC=1` / `0` turns them on or off for any engine; on the 27B engine they cost about 1.6% of decode
  speed). On Flash-Next they make fresh-prompt processing about 8-10% slower than the figures in
  docs/flash-next.md, which were measured before; `FERRITE_MALLOC=0` restores those speeds, at the cost of slowly
  growing host memory.
- **`/health` reports the server version**, and `fe-server --version` names the release, the engines and the
  kernels' GPU architecture.
- **Documentation.**
  - The speed tables were re-measured with the shipped `tools/bench` clients, each with the command that
    reproduces it: two Sparks, one Spark, agents, RTX 5080, RTX 5060 Ti, Arc B70 and Flash-Next.
  - A "Clean machine" section lists the system packages, how to get `ptxas` and cuBLASLt from NVIDIA's repository
    (no full toolkit), and the `FE_KARCH` of every card. It was tested in clean Ubuntu 24.04 and 26.04 containers.
  - Two Sparks need no sudo for NCCL: NVIDIA's `nvidia-nccl-cu13` wheel works (tested with NCCL 2.32.6).
  - The Arc B70 page lists the power and clock settings its numbers were measured with, and notes that the first
    request after a new build takes about 30 s.
  - A cooling note covers several cards in adjacent slots.
- **The release tarball carries every launcher, every doc page, `CHANGELOG.md` and the benchmark clients.**
  `tools/bench/agents.py` and `mixed.py` quote a fixed copy of the repository files they need, so they run from
  the tarball and their prompts do not change between releases.
- **`tools/bench/quickbench.py --mode code|prose [--runs N]`** runs without the menu and prints the median. With
  several requests it prints the decode aggregate the docs quote, next to the wall-clock rate.
- **`tools/bench/edit.py`** uses CPython 3.12.3's sources whatever Python runs it. **`tools/bench/sweep.py`**
  labels a run with the server's own `/health`.

### Added

- **`scripts/ferritectl.sh start|stop|restart|status|log [spark|tp2|rtx|fnext|b70]`** runs any target's server in
  the background, with one log and pid file per port. The installer uses it.
- **`tools/release-gate.sh TARGET SERVER|URL`** checks a server from another machine on the network: `/health`,
  `/v1/models`, a request, a streamed request, a burst of sampled and greedy requests, and the mixed-load repro.
  Given a host instead of a URL, it also starts and stops the server.
- **`fe-server --version`** and **`ferrite --version`**.
- **`FERRITE_PLAIN=1`** on `scripts/serve-xe.sh` serves plain decoding on the Arc B70.
- **`scripts/check-deps.sh`** reads the kernels' architecture from the binary. With `--build` it names any
  missing system packages in one apt line and checks for cuBLASLt. It also has an Arc B70 mode, and finds a
  pip-installed NCCL.
- **`tools/bench/modes.py`** checks that greedy text is the same across cache and drafting settings.
  **`tools/bench/mixed.py`** sends sampled and greedy requests together, as a crash check and a soak with a fixed
  greedy probe. **`tools/bench/agentturns.py`** measures time to first token turn by turn in a growing agent
  conversation.

## [0.6.0] - 2026-11-14

### Added

- **Qwen3.8-Flash-Next on one or two RTX 5060 Ti 16 GB** (experimental): a second engine in the same server,
  `fe-server --engine fnext` (build with `--features fnext`), for Qwen's 48-block hybrid MoE in the ISTA-DASLab
  GSQ-RCO Coder IQ1_M GGUF (58.4 GB), with Rust kernels for its GGUF quant types. The always-used weights, the K/V
  cache and the GDN state are on the card; the routed experts stay in the memory-mapped GGUF and fill LRU slots on
  the card, the ones that miss are computed on the CPU; the PLE n-gram table is read a few rows a token. Two cards
  run in tensor parallel (half the heads and half the experts each). Lossless MTP drafting, several requests at once
  (up to 8 slots), the prompt cache, 128K context. See [docs/flash-next.md](docs/flash-next.md).
  - Two cards (EPYC 7313 host, 128K context configured, greedy, 256 tokens): code 135 tok/s and prose 110 with MTP
    drafting, 75 without; prompt processing 1,434 tok/s at 2K and 2,467 at 16K; four requests at once 194-201 tok/s
    total on code.
  - One card inside a 28 GiB memory limit (standing in for a 32 GB machine): code 84-85 tok/s and prose 67 with MTP drafting, 45-46 without; prompt processing 1,340 tok/s at 2K and
    1,880 at 16K.
- **`scripts/fetch-flashnext.sh`** downloads the ISTA-DASLab IQ1_M files exactly as published, the MTP draft file
  and the tokenizer files, checks every size and sha256, and lays them out for **`tools/serve-fnext.sh`**, which
  picks the one-card or the two-card preset from the visible cards and the request slots from the host memory.
- **The MTP draft file** (`BHCC2025/Qwen3.8-Flash-Next-MTP-GGUF`, 1.5 GB): Qwen's MTP head, read from the official
  checkpoint and converted with llama.cpp's converter, as one small GGUF beside the ISTA files (which carry no MTP
  block). Without it Flash-Next runs without speculative decoding. `FNEXT_DRAFT=<path>` names another one,
  `FNEXT_DRAFT=none` skips it.

### Changed

- **GDN overlap in prompt groups on the RTX 5060 Ti 16 GB.** The chunked GDN kernel launches one block a value
  head: 48 blocks, which on 36 SMs is a full wave and a second one of 12. In a prompt group the next chunk's input
  norm and GDN projection now start on a second stream as the recurrence is launched and take the SMs that wave
  leaves idle. The same kernels on the same inputs, so the replies are the same text. Prompt processing goes from
  about 1,890 to 1,940 tokens per second at 16K (128K context configured), 1,470 to 1,510 at 56K and 1,040 to
  1,070 at 110K. `FE_GDN_OVERLAP=0` turns it off. Cards with 48 SMs or more are unchanged.

## [0.5.1] - 2026-11-12

### Changed

- **Faster prompt MLP on the RTX 5060 Ti 16 GB.** The NVFP4 GEMMs take the mbarrier pipeline on that card (36 SMs):
  the same bits as the default pipeline, the MLP 17% faster. Prompt processing goes from about 1,780 to 1,890
  tokens per second at 16K (128K context configured), 1,420 to 1,470 at 56K and 1,020 to 1,040 at 110K.
  `FE_FP4_KERNEL` picks another variant. The other targets are unchanged.

## [0.5.0] - 2026-11-12

### Added

- **Prompt groups on cards that stream their prompt weights** (the RTX 5080 preset, `tools/serve-5080.sh`). A long
  prompt that is alone on the card runs its chunks layer by layer, so the 16.8 GB of NVFP4/FP8 prompt weights cross
  the bus once a group instead of once a chunk. The replies are the same text as before, the prompt cache's
  snapshots inside a group hold the state they would have without it, and a cancelled request stops it between
  layers. `FE_PROMPT_GROUP=0` turns it
  off. See [docs/rtx-5080.md](docs/rtx-5080.md).
  - RTX 5060 Ti 16 GB (4-lane link, 128K context configured): prompt processing goes from about 790 to 1,780
    tokens per second at 16K, 800 to 1,420 at 56K and 650 to 1,020 at 110K, in the memory that is free anyway. In
    an agent-style soak the median time to first token of a turn went from 4.9 s to 2.6 s.
  - RTX 5080: about 8% at 16K, within 1% on longer prompts (its link already keeps up with the computation).
- **Four requests at once on the RTX 5060 Ti 16 GB** (`SLOTS=4`) checked: the lanes gate passes, 155 tokens per
  second in total with a 2K shared prefix.

The DGX Spark and Arc Pro B70 engines are unchanged.

## [0.4.3] - 2026-11-11

### Changed

- **Decode plans for the RTX 5060 Ti 16 GB.** The EXL3 decode kernels have a plan table tuned on the card (36 SMs);
  until now it ran with the DGX Spark's. Code generation goes from 139 to 155 tokens per second at 2K context
  (121 to 136 at 16K, 86 to 97 at 56K) and 500-token edits from 343 to 357. Replies with drafting on and off are
  the same text. See [docs/rtx-5080.md](docs/rtx-5080.md). The DGX Spark and RTX 5080 engines are unchanged.

## [0.4.2] - 2026-11-11

### Changed

- **Model cards use the model id `qwen3.8-27b`.** The cards of both quants on Hugging Face (and their copies in
  `docs/`) showed the old ids `spark` and `b70` in their serve examples. `scripts/models.manifest` pins the
  re-uploaded revisions; only the card and the checksum list changed there, the weights are the same files.

## [0.4.1] - 2026-11-11

### Changed

- **The default model id is `qwen3.8-27b` on every target.** It was `spark` on the CUDA engine and `b70` on the Intel
  Arc Pro B70 engine, names that only made sense on those machines. `/v1/models`, `/health` and the replies report
  the new id; requests may still name any model, so existing clients keep working. `FERRITE_MODEL_ID` (or
  `--served-model-name`) sets another id, for example `FERRITE_MODEL_ID=spark` to keep the old one.

### Fixed

- **RTX 5080 preset without the NVIDIA checkpoint.** `PROMPT=bf16` (also the fallback when the checkpoint is
  missing) ran out of device memory while loading, because bf16 copies of the prompt weights were built on the
  card. On small-memory devices the prompt now goes through the EXL3 weights on the decode kernels, as
  [docs/rtx-5080.md](docs/rtx-5080.md) describes (`FE_BF16=1` brings the copies back).

### Added

- **RTX 5060 Ti 16 GB results** in [docs/rtx-5080.md](docs/rtx-5080.md): the RTX 5080 build runs on it, with
  measured speeds and the prompt chunk size that suits its 8-lane link.

## [0.4.0] - 2026-11-11

### Added

- **Up to four requests at once on the Intel Arc Pro B70** (`FERRITE_SLOTS=4` with `scripts/serve-xe.sh`). Each
  request gets a lane with its own conversation state; lanes that decode together share one verify window, so the
  weights are read once for all of them (four lanes total about 1.4x the speed of one request), and a prompt that
  starts like another lane's reuses that part in whole prompt windows. Every lane's text is what the same request
  gives alone. `/health` reports `max_batch_size` 4 in this mode. See [docs/arc-b70.md](docs/arc-b70.md).

## [0.3.0] - 2026-11-11

### Added

- **NVIDIA RTX 5080 target (experimental).** The CUDA engine serves Qwen3.8-27B on one RTX 5080 (16 GB) with 128K
  context, built from source for `sm_120a` (`FE_KARCH`). See [docs/rtx-5080.md](docs/rtx-5080.md).
  - A 6-bit K/V cache (`FE_KVQ`, new Rust kernels), NVFP4/FP8 prompt weights streamed from pinned host memory, and
    the embedding table and prompt cache pool in host memory.
  - MTP drafting with copy rows, and `tools/serve-5080.sh` to start it.
- **Up to four requests at once on the RTX 5080** from one shared pool of context (`SLOTS=4`, `FE_POOL`): a request
  that fits a unit runs beside others, a longer one takes the whole pool and runs alone.
- **Shared prefix computed once.** Requests that arrive together and start with the same text (parallel lanes of one
  job, agents with one system prompt) prefill that text once, wherever it ends
  ([docs/prompt-cache.md](docs/prompt-cache.md)).
- `tools/bench/lanes.py`: a benchmark and gate for N requests sharing a prefix.
- A "Supported systems" table in the README: Ferrite runs on Linux only.

### Changed

- Three or more streams decoding at once share 32 verify rows with DFlash2 drafting (48 for one or two), which raises
  their total speed. Set `FE_VERIFY_ROWS` / `FE_VERIFY_GAIN` to fix them.
- `/health`: `max_batch_size` is the number of requests the loaded engine decodes at once (a larger
  `--max-concurrent` is clamped to it), and a test pins the field names (`max_batch_size`, `max_model_len`,
  `running`, `queued`).
- The DGX Spark package is `ferrite-0.3.0-dgx-spark-aarch64.tar.gz`; the one-line installer installs 0.3.0.

### Fixed

- Captured CUDA graphs were never released, so many different batch shapes could fill device memory. They are now
  dropped when memory runs low.
- Prompt attention with the quantized K/V cache: a kernel written during this release read staged data too early.

## [0.2.0] - 2026-11-10

### Added

- **Intel Arc Pro B70 engine (experimental).** Ferrite serves Qwen3.8-27B on one Arc Pro B70 (32 GB) with
  `fe-server --engine xe`. The kernels are Rust compiled to SPIR-V and loaded through Level Zero. Built from source
  with `--features xe`; one request at a time. See [docs/arc-b70.md](docs/arc-b70.md).
  - XQ 5.0 bpw weights for decode and int8 prompt processing.
  - Lossless DFlash2 and MTP drafting, a prefix cache and 128K context.
  - New crates `fe-xe`, `fe-xdf`, `fe-xq`, `fe-ze` and `fe-kbuild-xe`, and the kernel sources in `kernels-xe`.
  - `scripts/serve-xe.sh` starts the server.
- **XQ model.** [`BHCC2025/Qwen3.8-27B-XQ-5.0bpw-mix`](https://huggingface.co/BHCC2025/Qwen3.8-27B-XQ-5.0bpw-mix)
  on Hugging Face, described in [docs/model-card-xq.md](docs/model-card-xq.md). `scripts/fetch-models.sh xq` fetches
  it and checks its sha256 sums.
- `fe-server`: engines may report their own timings in a response.
- This changelog.
- Prebuilt DGX Spark package `ferrite-0.2.0-dgx-spark-aarch64.tar.gz`; the one-line installer now installs 0.2.0.
  The Arc Pro B70 engine is built from source.

### Changed

- `fe-server`: streamed responses declare `charset=utf-8`.

## [0.1.0] - 2026-11-09

First release: an LLM inference engine for NVIDIA DGX Spark (GB10), written in Rust down to the GPU kernels, serving
Qwen3.8-27B behind an OpenAI-compatible API.

### Added

- Serving on one Spark, or on two over RoCE (tensor parallel), with 128K context and up to 4 concurrent requests.
- EXL3 3.0 bpw decode and NVFP4/FP8 prompt processing.
- Lossless speculative decoding: DFlash2 block drafting with verify trees, copy rows, and the model's MTP head as a
  fallback.
- A prompt cache for agents: turn snapshots, parked conversations and one shared system prompt
  ([docs/prompt-cache.md](docs/prompt-cache.md)).
- Model [`BHCC2025/Qwen3.8-27B-EXL3-3.0bpw-mix`](https://huggingface.co/BHCC2025/Qwen3.8-27B-EXL3-3.0bpw-mix),
  fetched and sha256-checked by `scripts/fetch-models.sh`.
- One-line install (`scripts/install.sh`), an agent-led install guide ([AGENTS.md](AGENTS.md)) and a prebuilt
  `ferrite-0.1.0-dgx-spark-aarch64.tar.gz` (aarch64, CUDA 13, kernels for `sm_121a`).

[0.8.1]: https://github.com/BHCC2025/ferrite/releases/tag/v0.8.1
[0.8.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.8.0
[0.7.2]: https://github.com/BHCC2025/ferrite/releases/tag/v0.7.2
[0.7.1]: https://github.com/BHCC2025/ferrite/releases/tag/v0.7.1
[0.7.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.7.0
[0.6.1]: https://github.com/BHCC2025/ferrite/releases/tag/v0.6.1
[0.6.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.6.0
[0.5.1]: https://github.com/BHCC2025/ferrite/releases/tag/v0.5.1
[0.5.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.5.0
[0.4.3]: https://github.com/BHCC2025/ferrite/releases/tag/v0.4.3
[0.4.2]: https://github.com/BHCC2025/ferrite/releases/tag/v0.4.2
[0.4.1]: https://github.com/BHCC2025/ferrite/releases/tag/v0.4.1
[0.4.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.4.0
[0.3.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.3.0
[0.2.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.2.0
[0.1.0]: https://github.com/BHCC2025/ferrite/releases/tag/v0.1.0
