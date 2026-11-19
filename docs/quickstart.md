# Ferrite quickstart

Ferrite serves Qwen3.8-27B from a Rust engine (host code and GPU kernels) on NVIDIA DGX Spark, with an
OpenAI-compatible API: streaming, tool calls, thinking on/off, 128K context, a prompt cache, 4 concurrent streams
and speculative decoding (DFlash2 or the model's MTP head). It runs on one Spark or on two over the RoCE link.

Ferrite runs on Linux only; macOS and Windows are not supported. This page is for DGX Spark. The other targets have
their own pages: [rtx-5080.md](rtx-5080.md) and [arc-b70.md](arc-b70.md).

## Requirements

| | |
|---|---|
| Hardware | DGX Spark (GB10, compute capability 12.1, 128 GB unified memory). The GPU kernels are compiled for `sm_121a` only. Two Sparks need the ConnectX-7 cable between them (RoCE). |
| OS / driver | DGX OS (Ubuntu 24.04), NVIDIA driver 580.x with CUDA 13.0. Tested with 580.178.04 on kernel 7.0.0-1019-nvidia. |
| Memory | One node with the defaults (128K context x 4 slots, NVFP4 prompt weights, DFlash2) needs about 80 GB available for the model and the 4 x 128K KV cache, plus the prompt cache's 12 GB pool (`FERRITE_PARK_GB`), so at least 95 GB must be available before starting; with agents running, about 34 GB stays available. The pool also shrinks by itself to keep 20 GB available. The GPU shares system memory: running out of it can stall or reboot the machine, so do not run other large GPU jobs beside the server. |
| Disk | 45 GB free: about 38 GB for the three model folders, plus Ferrite (the release tarball needs almost nothing; a source build about 6 GB under your home folder for rustup, cargo and the build). |
| Libraries | `libcuda.so.1` (the driver) and `libibverbs.so.1` (rdma-core, preinstalled on DGX OS). Two nodes also need NCCL 2.30+ ([Two nodes](#two-nodes-tp2): a user-level `pip` install works, no sudo). |
| Download tool | The Hugging Face CLI (`hf`). DGX OS's Python refuses a system-wide `pip install`, so give it its own environment: `python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install -U huggingface_hub`. `scripts/fetch-models.sh` finds `hf` there or on `PATH`. |

`scripts/check-deps.sh` checks all of this (`--two-node` adds NCCL and the RDMA device).

## The one-command install

```bash
curl -fsSL https://raw.githubusercontent.com/BHCC2025/ferrite/main/scripts/install.sh | bash
```

`scripts/install.sh` does everything below in six printed steps, and running it again skips the finished ones:

1. checks the machine: aarch64, a GB10 GPU (compute capability 12.1) with the NVIDIA driver, at least 95 GB of
   available memory (`FERRITE_MIN_FREE_GB`), and that the port is free or already serves Ferrite;
2. gets Ferrite into `~/ferrite` (`FERRITE_HOME`): the latest published release, which it asks the GitHub API for
   (`FERRITE_VERSION=0.7.1` or `--version 0.7.1` pins one). That is the release tarball with ready-made binaries,
   its sha256 checked. When the release has no tarball (or with `--source`) it says so and builds from source: a
   clone of the release's tag, rustup, the pinned toolchains and `cargo build`. Run inside a checkout or an unpacked
   release, it uses that folder instead;
3. installs the Hugging Face CLI into `~/.venvs/hf`;
4. downloads and verifies the models into `FERRITE_MODELS` (`/var/tmp/models`);
5. starts the server in the background with `scripts/ferritectl.sh start` (log `logs/server-8000.log`;
   `--no-serve` stops before this);
6. sends a test request and prints `OK: Ferrite is serving http://<this machine>:8000/v1 (model id 'qwen3.8-27b')`.

It needs no `sudo`: when a system package is missing it stops and prints the `apt` command to run. Every
`scripts/serve.sh` setting below (`FERRITE_PORT`, `FERRITE_CTX`, ...) applies to the server it starts. Stop the server
with `~/ferrite/scripts/ferritectl.sh stop`.

**Upgrading.** Run the same command again. When `~/ferrite` holds an older release, the installer stops the upgrade
if that release's server is still running (stop it first), moves the old folder to `~/ferrite-<old version>` and
installs the new one. The models and their verification are kept. A source checkout in `~/ferrite` is moved to the
new release's tag when it has no local changes; otherwise the installer says why it left it alone.

To let a coding agent do it, give it this line: `Install Ferrite on this DGX Spark by following https://github.com/BHCC2025/ferrite/blob/main/AGENTS.md exactly; stop and ask me if any check fails.`

The sections below are the same steps by hand.

## 1. Get Ferrite

Either unpack a release tarball on the Spark:

```bash
tar xzf ferrite-<version>-dgx-spark-aarch64.tar.gz
cd ferrite-<version>-dgx-spark-aarch64
scripts/check-deps.sh
```

or build from source (see [Build from source](#build-from-source)) and run the same scripts from the repository.

## 2. Download the models

```bash
export FERRITE_MODELS=/var/tmp/models          # any folder with ~38 GB free; this is the default
scripts/fetch-models.sh
```

Three folders land in `$FERRITE_MODELS`:

| Folder | Hugging Face repo | Used for |
|---|---|---|
| `Qwen3.8-27B-EXL3-3.0bpw-mix` (~13 GB) | `BHCC2025/Qwen3.8-27B-EXL3-3.0bpw-mix` | decode weights (EXL3 3.0 bpw mix), tokenizer, chat template |
| `nvidia-Qwen3.8-27B-NVFP4` (~21 GB) | `nvidia/Qwen3.8-27B-NVFP4` | prompt (prefill) GEMMs in NVFP4/FP8 and the drafter's output head |
| `z-lab-Qwen3.8-27B-DFlash2` (~3.6 GB) | `z-lab/Qwen3.8-27B-DFlash2` | the DFlash2 block drafter |

The script pins the upstream revisions and checks every file's size and sha256 against `scripts/models.manifest`.
`--quick` checks sizes only, `--verify-only` re-checks an existing folder, and roles (`quant`, `prompt`, `draft`)
fetch a subset. A failed transfer is retried (3 attempts). Run again after an interruption, it downloads what is
missing, fetches again any file whose size or sha256 is wrong, and deletes the partial files the interruption left.
A file cut off mid-transfer starts over: the Hugging Face CLI does not continue it. It stops before downloading
when the disk has too little room (`FERRITE_FORCE=1` tries anyway).

## 3. Start the server (one node)

```bash
FERRITE_MODELS=/var/tmp/models scripts/serve.sh
```

It runs in the foreground, and its process becomes the server: Ctrl-C or `kill <pid>` stops it.
`scripts/ferritectl.sh start` runs it in the background instead (`stop`, `restart`, `status`, `log`). Loading takes
about a minute; the server is ready when `/v1/models` answers.

> **The server is open to your network.** It listens on every interface (`0.0.0.0`) by default, so any machine on the
> local network can use it, for example a coding agent on your laptop. There is no API key. Use it on a trusted
> network or firewall the port; `FERRITE_HOST=127.0.0.1` keeps it on the Spark itself.

Before it loads the model, the launcher checks the port and the GPU lock. It checks the model files' sizes against
the manifest, and it checks the memory against what the settings need. It stops at once with a message when one of
them fails: the port is taken (and by what), another Ferrite job holds the GPU (and which), a file is missing or cut
off (which one), or there is not enough memory. `FERRITE_FORCE=1` starts anyway despite the memory and file checks.

Settings are environment variables.
The site settings (model folders, locks, the two-node fabric) are read from `tools/site.sh` by every script and by the
binaries themselves; the README has the full `FERRITE_*` table, and `tools/site.sh` run on its own prints the resolved
values. `scripts/serve.sh` adds the server settings:

| Variable | Default | |
|---|---|---|
| `FERRITE_MODELS` | `/var/tmp/models` | the folder from step 2 |
| `FERRITE_EXL3_DIR`, `FERRITE_NVFP4_DIR`, `FERRITE_DRAFT_DIR` | the three folders under `FERRITE_MODELS` | set one of them to keep that model elsewhere |
| `FERRITE_GPU_LOCK` | `/var/tmp/ferrite-gpu.lock` | the node's GPU lock. A server holds it shared, plus its own card's lock (`ferrite-gpu.<card UUID>.lock` in the same folder) exclusively: servers on different cards run side by side, a second launcher on a busy card stops at once and names the holder, and a whole-node job (a test, a KL run) excludes them all. |
| `FERRITE_LOCK_WAIT` | `0` | `1` waits for a held GPU lock instead of stopping |
| `FERRITE_HOST` | `0.0.0.0` | every interface: other machines on the network can use the server, with no API key. `127.0.0.1` keeps it on this machine. |
| `FERRITE_PORT` | `8000` | checked before the model loads |
| `FERRITE_FORCE` | `0` | `1` starts even when the memory or model-file check fails |
| `FERRITE_MALLOC` | `0` here (`1` for Flash-Next) | `1` fixes glibc's malloc thresholds at start, which keeps a long-running server's host memory flat; on the 27B engine it costs about 1.6% of decode speed, so it is off there by default |
| `FERRITE_MODEL_ID` | `qwen3.8-27b` | the id `/v1/models` reports. Requests may name any model; the server always runs this one. |
| `FERRITE_CTX` | `131072` | context per request. The KV cache takes ~64 KB a token a slot (128K x 4 slots = 34 GB), so lower this to save memory. |
| `FERRITE_SLOTS` | `4` | requests at once: at most 4 with DFlash2 drafting; more requests wait in a queue |
| `FERRITE_PROMPT` | `fp4` | prompt weights: `fp4` (the NVIDIA checkpoint) or `bf16` (see [Without the NVIDIA checkpoint](#without-the-nvidia-checkpoint)) |
| `FERRITE_DRAFT` | `dflash` | speculative decoding: `dflash`, `mtp`, `auto` (DFlash2 up to 2 streams, MTP above) or `off`. Drafting is lossless: every mode gives the same greedy text ([Same text](#same-text)). |
| `FERRITE_CACHE` | `1` | the prompt cache: each conversation's state is kept at its turn boundaries and at the end of the system prompt, so a later turn, or a new conversation with the same system prompt, skips what was already computed. `0` turns it off. It works with every `FERRITE_DRAFT` mode. |
| `FERRITE_PARK_GB` | `12` | memory the prompt cache keeps for conversations moved out of a slot and for shared system prompts, on top of the model and the KV cache (two nodes: 24 a node). The server shrinks it as needed to keep 20 GB of memory available (`FE_PARK_RESERVE_GB`). |

Extra arguments go to `fe-server` (`bin/fe-server --help` lists them; `bin/fe-server --version` names the release).

### Same text

With greedy decoding (temperature 0) a request's text does not depend on what else runs beside it, on the number of
slots, on the drafting mode (`FERRITE_DRAFT` `dflash`, `mtp`, `auto` or `off`), on `FERRITE_CACHE`, or on whether its
prompt came from the prompt cache or was computed in full. A prompt is cut into chunks at its turn boundaries
whatever the settings, so the cuts and therefore the arithmetic are the same in every mode.
`python3 tools/bench/modes.py` checks it against servers started with the different settings.

## 4. Test it

```bash
curl -s http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
  "max_tokens": 300,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Add `"stream": true` for server-sent events. Thinking is on by default in the Qwen template; `enable_thinking: false`
turns it off. When it is on, the reasoning comes back in `reasoning_content`, separate from the answer in
`content`. Tool calls use the OpenAI `tools` / `tool_calls` fields.

## 5. Point a client at it

Any OpenAI-compatible client works. Use base URL `http://<spark-address>:8000/v1`, model `qwen3.8-27b` (or your
`FERRITE_MODEL_ID`) and any non-empty API key.

- Open WebUI: Settings, Connections, add an OpenAI API connection with the base URL above.
- Continue (VS Code / JetBrains): a model with `provider: openai`, `apiBase: http://<spark-address>:8000/v1`,
  `model: qwen3.8-27b`.
- Cline: API provider "OpenAI Compatible", the base URL above, model id `qwen3.8-27b`.
- Pi: an OpenAI-compatible provider with the base URL above; it sends the model id `qwen3.8-27b`.

Other machines reach the server as it is: it listens on every interface by default. To keep it on the Spark, start
it with `FERRITE_HOST=127.0.0.1`. Requests may name any model id: the server always answers with the one it runs.

## Prompt cache

The server keeps what it has computed for each conversation: it snapshots the state at turn boundaries, resumes a
new turn from the longest matching snapshot, parks conversations that lose their slot, and computes a system prompt
shared by several agents once. Four agents sharing a ~13K-token system prompt compute 84% fewer prompt tokens (86%
with thinking on), and later turns start in about half a second. A resumed reply is the same as one computed in full.

- It is on by default, with every drafting mode: `FERRITE_CACHE=0` turns it off (`scripts/serve.sh` and
  `tools/serve-tp2.sh`).
- Its park pool is allocated at start: 12 GB on one Spark, 24 GB per node on two (`FERRITE_PARK_GB` overrides). It
  shrinks by itself if needed to keep 20 GB of memory available (`FE_PARK_RESERVE_GB`).
- The response's `usage.prompt_tokens_details.cached_tokens` shows how much of a prompt came from the cache.

Details and the guarantees: [prompt-cache.md](prompt-cache.md).

## Two nodes (TP2)

Both Sparks hold half of every layer and exchange partial sums over the RoCE link (an ibverbs one-shot path for
decode-sized messages, NCCL for the rest). Node A runs rank 0 and the server; node B runs rank 1, started by node A
over `ssh`. Start it on node A:

```bash
tools/serve-tp2.sh     # FERRITE_PORT / FERRITE_CTX / FERRITE_SLOTS, PROMPT, DRAFT as in tools/serve.sh --help
```

Like `scripts/serve.sh`, it uses the NVFP4 prompt weights when `FERRITE_NVFP4_DIR` is present (else bf16) and a 128K
context. It needs:

- the same Ferrite folder (same path, same build) and the same model folders on both nodes;
- passwordless `ssh` from node A to node B: `FERRITE_PEER` is node B's ssh name;
- NCCL 2.30 or newer on both nodes. DGX OS does not ship it. Without sudo, install NVIDIA's wheel into a venv
  (tested with NCCL 2.32.6) and point `FERRITE_NCCL_LIB` at it:

  ```bash
  python3 -m venv ~/.venvs/nccl && ~/.venvs/nccl/bin/pip install nvidia-nccl-cu13      # on both nodes
  export FERRITE_NCCL_LIB=$(ls ~/.venvs/nccl/lib/python3*/site-packages/nvidia/nccl/lib/libnccl.so.2)
  ```

  With sudo, `sudo apt install libnccl2` puts it on the loader path instead. `tools/serve-tp2.sh` passes
  `FERRITE_NCCL_LIB` on to node B. `scripts/check-deps.sh --two-node` checks NCCL together with the RDMA devices and
  the interface below, and finds the venv by itself;
- the fabric settings in `tools/site.sh`: `FERRITE_TP_HOST` (node A's address on the RoCE link, the rendezvous;
  its port is `FE_TP_PORT`, default 29620), `FERRITE_NET_IF` (the link's interface for NCCL), `FERRITE_ROCE_HCA`
  (one or two RDMA devices) and `FERRITE_ROCE_GID` (their RoCE v2 GID index). While `FERRITE_TP_HOST` is unset
  the two-node scripts detect all of them from both nodes (`tools/fabric-detect.sh`), as `FERRITE_FABRIC=auto`
  does; settings you keep can go in `tools/site.local.sh` (not tracked), which `tools/site.sh` reads first;
- `FERRITE_MIN_FREE_GB` (default 45) available on each node; a TP2 job also takes `FERRITE_TP2_LOCK` on node A.

Like `scripts/serve.sh` it listens on every interface (`FERRITE_HOST=127.0.0.1` keeps it on node A), checks the port
first, and stops at once when another job holds a lock. Its process forwards Ctrl-C and SIGTERM, so stopping it
stops both ranks; `scripts/ferritectl.sh start tp2` runs it in the background. The speed on two nodes is in the
README's [Results](../README.md#results). For reference, with 2K-token prompts and
500 tokens out (DFlash2): code 267 tok/s for one stream and 467 tok/s aggregate for 4 streams.

## Without the NVIDIA checkpoint

`FERRITE_PROMPT=bf16` (the default when `FERRITE_NVFP4_DIR` is missing) runs prompt rows through bf16 weights
that are reconstructed from the EXL3 quant on the GPU at load, so only the quant (and optionally the drafter) has to
be downloaded. What that costs:

- Prefill is about half as fast as with NVFP4. On two nodes, 2K / 8K / 32K prompts ran at 1378 / 1846 / 1673 tok/s
  with bf16 prompt slices (an earlier build), against 3669 / 3466 / 2880 with NVFP4. Single-node bf16 prefill has
  not been measured yet.
- The bf16 copies take about 52 GB on one node instead of 21 GB, so `scripts/serve.sh` lowers the default context to
  32K there (`FERRITE_CTX` overrides it).
- The DFlash2 drafter's output head falls back from the NVFP4 head (the first 128K vocabulary ids) to the EXL3 head
  over the whole vocabulary, which makes each draft slower.
- Accuracy is the same class: KL against BF16 0.0218 (code) / 0.0466 (text) with bf16 prompt rows, against 0.0207 /
  0.0401 with NVFP4.

## Clean machine

What a fresh Ubuntu 24.04 or 26.04 needs before a source build (DGX OS has all of it except Rust). The same steps
were run in clean `ubuntu:24.04` and `ubuntu:26.04` containers for the RTX 50 builds:

```bash
# system packages (the only step that needs sudo)
sudo apt install -y curl ca-certificates git build-essential python3 python3-venv

# ptxas, and libcublasLt for the 27B engine, from NVIDIA's CUDA repository (no full toolkit needed);
# ubuntu2404 or ubuntu2604, x86_64 or sbsa (aarch64)
curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install -y cuda-nvcc-13-5 libcublas-dev-13-5     # any 13.x; Flash-Next alone needs only cuda-nvcc
export PATH=/usr/local/cuda/bin:$PATH

# Rust: rustup with a stable default toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
```

On Arch Linux (the Arc B70 target), `pacman -S rustup` installs rustup with **no** default toolchain: run
`rustup default stable` once, or every `cargo` command stops with "no default is configured".

The kernels are compiled for one GPU architecture, chosen at build time with `FE_KARCH`. Without it the build takes
the card `nvidia-smi` reports on the build machine (and says so in a build warning), or `sm_121a` when there is
none. Set it to build for a card that is not in the build machine:

| Target | `FE_KARCH` | Build |
|---|---|---|
| DGX Spark (GB10) | `sm_121a` (the default on aarch64) | `cargo build --release -p fe-server --features ferrite -p fe-cli` |
| RTX 5080, RTX 5060 Ti (Qwen3.8-27B) | `sm_120a` | `FE_KARCH=sm_120a cargo build --release -p fe-server --features ferrite -p fe-cli` |
| RTX 5060 Ti (Qwen3.8-Flash-Next) | `sm_120a` | `FE_KARCH=sm_120a cargo build --release -p fe-server --features fnext` |
| Both RTX engines in one binary | `sm_120a` | `FE_KARCH=sm_120a cargo build --release -p fe-server --features ferrite,fnext -p fe-cli` |
| Intel Arc Pro B70 | none (SPIR-V) | `cargo build --release --features xe -p fe-server` ([arc-b70.md](arc-b70.md)) |

`target/release/fe-server` is one file: building it with other features replaces the previous build. Build with both
`ferrite,fnext` to keep both RTX engines. A binary built for the wrong card does not start. The launchers and
`scripts/check-deps.sh` read the architecture from the binary and print the `FE_KARCH=... cargo build` command
that fixes it.

## Build from source

On the Spark (aarch64):

```bash
# Rust: rustup with a stable toolchain for the host crates
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"

# the GPU kernels' pinned nightly (kernels/rust-toolchain.toml) with its components and target, checked together
# with a C linker and ptxas (CUDA 13 toolkit, on PATH or at /usr/local/cuda/bin/ptxas)
scripts/check-deps.sh --build --install

cargo build --release -p fe-server --features ferrite -p fe-cli
```

The kernels are built with exactly one nightly, `nightly-2026-11-04` (rustc 1.101.0-nightly 708c94ef0), pinned in
`kernels/rust-toolchain.toml`: the nvptx backend and the unstable features the kernels use change between nightlies.
`scripts/check-deps.sh --build` checks that toolchain, its `rust-src`, `llvm-tools` and `llvm-bitcode-linker`
components and the `nvptx64-nvidia-cuda` target, a stable Rust 1.99 or newer, `cc` and `ptxas`; `--install` installs
the missing toolchains with rustup.

The build compiles every `kernels/k-*` crate to PTX with that nightly (`target-cpu=sm_121a`), assembles it with
`ptxas -arch=sm_121a`, and embeds the cubins in `target/release/fe-server` and `target/release/ferrite`. Tested with
rustc 1.99.1 (host), nightly-2026-11-04 (kernels) and ptxas 13.0.88.
`scripts/package-release.sh` builds and packs the release tarball into `dist/`.

To build for an RTX 50 card instead (x86_64, `sm_120a`), set `FE_KARCH=sm_120a` for the build (the table in
[Clean machine](#clean-machine)); see [rtx-5080.md](rtx-5080.md). A source build prints a few compiler warnings;
they are harmless.

## Troubleshooting

- `check-deps: ERROR: ... the kernels are built for ...` or `serve: ... was built for sm_121a, but this GPU ... needs
  sm_120a`: the binary was built for another card; rebuild with the `FE_KARCH` it names.
- `serve: port 8000 is already in use (by ...)`: another server has the port. Stop it, or set `FERRITE_PORT`.
- `serve: another Ferrite GPU job holds /var/tmp/ferrite-gpu.lock (pid ...)`: a Ferrite server or job is already
  running on that card. Stop it (`scripts/ferritectl.sh stop`, or `kill` the pid named), or wait for it with
  `FERRITE_LOCK_WAIT=1`.
- `serve: only N GB of memory is available; this setup needs about M GB`: stop other GPU jobs, or lower
  `FERRITE_CTX` / `FERRITE_SLOTS` / `FERRITE_PARK_GB`. The GPU shares the memory, and running out of it can stall or
  reboot the machine.
- The server exits during load, or the machine stalls: not enough memory. Lower `FERRITE_CTX` / `FERRITE_SLOTS`,
  or the prompt cache's `FERRITE_PARK_GB`, or stop other GPU work. The pool already shrinks by itself to keep 20 GB
  available (`FE_PARK_RESERVE_GB`).
- `--slots N is not supported with DFlash2 drafting (DRAFT=dflash): one draft pass serves at most 4 streams in this
  build`: with DFlash2 drafting (the default) at most 4 requests run at once. Use `FERRITE_SLOTS=4` (more requests
  wait in a queue), or `FERRITE_DRAFT=auto` / `mtp` for more slots (MTP drafting above 2 streams, not tested above 4
  slots). The launch scripts check this before loading the model.
- Random slow runs under memory pressure: the kernel's proactive compaction (`kcompactd`) fights the pinned GPU
  memory. `sysctl vm.compaction_proactiveness=0` stops it (optional; it is a system-wide setting).
