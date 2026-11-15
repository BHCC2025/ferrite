# Changelog

Notable changes to Ferrite, newest release first. Every section below is a tagged release that is published and
available. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
