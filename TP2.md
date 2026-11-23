# TP2: Qwen3.8-27B across two DGX Sparks

TP2 is tensor parallelism over two nodes: two DGX Sparks, joined by a direct RoCE v2 link, serve Qwen3.8-27B as one
server. Each Spark holds half of every layer. Node A (rank 0) runs the server and the scheduler; node B (rank 1) is
started by node A over `ssh` and follows it. The two exchange small partial sums on every step, and the reply is the
same text as on one node.

This page has what a user needs first ([What you need](#what-you-need), [Serving](#serving), [NCCL](#nccl),
[Troubleshooting](#troubleshooting)), then development measurements, then how it works. One-node setup, the models and
the server settings are in [docs/quickstart.md](docs/quickstart.md); terms such as slot, drafting and K/V cache are in
the README's [Words used in these docs](README.md#words-used-in-these-docs).

## What you need

- Two DGX Sparks with the ConnectX-7 cable between them (RoCE), each set up as in
  [docs/quickstart.md](docs/quickstart.md#requirements).
- The same Ferrite folder (same path, same build) and the same model folders on both nodes. Both nodes must run the
  same binary (`tools/tp2.sh` compares sha256s).
- Passwordless `ssh` from node A to node B: `FERRITE_PEER` is node B's ssh name.
- NCCL 2.30 or newer on both nodes ([NCCL](#nccl) below).
- The fabric settings in `tools/site.sh`:
  - `FERRITE_TP_HOST`: node A's address on the RoCE link, the rendezvous (its port is `FE_TP_PORT`, default 29620);
  - `FERRITE_NET_IF`: the link's interface for NCCL;
  - `FERRITE_ROCE_HCA`: one or two RDMA devices;
  - `FERRITE_ROCE_GID`: their RoCE v2 GID index.

  While `FERRITE_TP_HOST` is unset the two-node scripts detect all of them from both nodes
  (`tools/fabric-detect.sh`), as `FERRITE_FABRIC=auto` does: it derives them from the subnets the nodes share. Settings
  you keep can go in `tools/site.local.sh` (not tracked), which `tools/site.sh` reads first.
- `FERRITE_MIN_FREE_GB` (default 45) of memory available on each node. A TP2 job also takes `FERRITE_TP2_LOCK` on
  node A.

`scripts/check-deps.sh --two-node` checks NCCL together with the RDMA devices and the interface, and finds the NCCL
venv by itself.

## Serving

```bash
FERRITE_PEER=nodeb tools/serve-tp2.sh                    # on node A, in the foreground; ~40 s to load
FERRITE_PEER=nodeb scripts/ferritectl.sh start tp2       # or in the background (stop / status / log)
tools/serve-tp2.sh --help                                # the settings and the commands it would run
until curl -sf localhost:8000/v1/models; do sleep 5; done
```

- **Settings.** `FERRITE_PORT`, `FERRITE_CTX`, `FERRITE_SLOTS`, `FERRITE_PROMPT`, `FERRITE_DRAFT` and `FERRITE_CACHE`
  as on one node ([docs/quickstart.md](docs/quickstart.md#3-start-the-server-one-node)). Like `scripts/serve.sh`, it
  uses the NVFP4 prompt weights when `FERRITE_NVFP4_DIR` is present (else bf16) and a 128K context. The prompt
  cache's pool is 24 GB a node (`FERRITE_PARK_GB`).
- **Network.** The server listens on every interface of node A (`FERRITE_HOST=127.0.0.1` keeps it local), and it does
  not check an API key: use it on a trusted network or firewall the port.
- **Locks.** The script checks the port first. It holds the global TP2 lock and both nodes' GPU locks while the
  server is up; a second start stops at once and names the holder.
- **Stopping.** Stopping the script (Ctrl-C, `kill` with the pid of the `tools/serve-tp2.sh` process, or
  `scripts/ferritectl.sh stop`) stops rank 0; the script forwards Ctrl-C and SIGTERM. The follower on node B then
  exits ("rank 0 stopped after N calls"), and both nodes' locks and memory are free again.
- **Do not rebuild on node B while the server runs.** Rank 1 runs the binary at the same path on node B.
- **Log.** In the foreground the log is on the terminal; with `scripts/ferritectl.sh` it is
  `<ferrite folder>/logs/server-<port>.log` on node A (`scripts/ferritectl.sh log`).

### NCCL

Both nodes need NCCL 2.30 or newer for the prompt batches (decode-sized reductions take the RoCE path). DGX OS does
not ship it. Without sudo, NVIDIA's wheel works (tested with NCCL 2.32.6):

```bash
python3 -m venv ~/.venvs/nccl && ~/.venvs/nccl/bin/pip install nvidia-nccl-cu13           # on both nodes
export FERRITE_NCCL_LIB=$(ls ~/.venvs/nccl/lib/python3*/site-packages/nvidia/nccl/lib/libnccl.so.2)
```

`tools/serve-tp2.sh` passes `FERRITE_NCCL_LIB` to node B, so the path must be the same there. With sudo,
`sudo apt install libnccl2` from NVIDIA's CUDA repository puts it on the loader path instead.
`scripts/check-deps.sh --two-node` checks it.

## Speed and accuracy

The current serving figures (code, prose and edits at 1 and 4 requests, prompt processing from 2K to 120K) are the
README's [Results](README.md#results) table. They were measured with `tools/bench/sweep.sh`, which starts
`tools/serve-tp2.sh` with its defaults on a test port and runs `tools/bench/sweep.py` against it. To measure your own
pair, run `FERRITE_PEER=<node B> tools/bench/sweep.sh` on node A (about 20 minutes).

## Troubleshooting

- **Check the pair first.** `scripts/check-deps.sh --two-node` checks NCCL, the RDMA devices and the interface;
  `tools/fabric-detect.sh <node B>` prints the fabric settings it finds from the subnets the two nodes share (it stops
  with `no subnet shared with <peer>` when there is none); `TP2_DRY=1 tools/tp2.sh <any arguments>` prints what both
  ranks would run and runs nothing; `tools/serve-tp2.sh --help` prints the settings and the commands it would run.
- **`no model at ...: run scripts/fetch-models.sh on both nodes`** or **`FERRITE_DRAFT=... needs the DFlash2 drafter
  in ... on both nodes`**: the model folders must be at the same paths on node B.
- **NCCL not found on node B.** `FERRITE_NCCL_LIB` is passed on as it is, so the library must be at the same path on
  both nodes.
- **A second start stops at once and names a holder.** Another Ferrite job holds the TP2 lock or a node's GPU lock.
  Stop it first.
- **The ranks refuse to start together.** Both nodes must run the same binary (`tools/tp2.sh` compares sha256s), and
  in lockstep runs the ranks compare a hash of their arguments at connect. Build the same source on both nodes. The
  connection attempt gives up after `FE_TP_CONNECT_SECS` (900).
- **`FERRITE_SLOTS=N is not supported with DFlash2 drafting`**: at most 4 requests run at once with DFlash2. Use
  `FERRITE_SLOTS=4`, or `FERRITE_DRAFT=auto` / `mtp`.
- **Upgrading.** Stop the server, then bring both nodes to the same release and build on both before starting again.

## Development measurements

The sections below are development measurements of the engine paths: lockstep CLI runs and earlier builds, kept
for the comparisons inside each table. Each one gives the command it was taken with. They are not the current
serving figures; those are the README's [Results](README.md#results).

### Accuracy (development measurement)

KL vs the BF16 reference (tools/kl-tp2.sh: code / text 1024-token prompt, long16k 15360, then 512 teacher-forced
decode rows):

| | TP2 fp4 prompt | TP2 bf16 prompt | TP1 bf16 prompt |
|---|---|---|---|
| KL code (40 seqs x 512) | 0.0207 (top1 96.23) | 0.02186 (96.17) | 0.02181 |
| KL text (20 seqs x 512) | 0.0401 (top1 91.87) | 0.04652 (90.71) | 0.04657 |
| KL long16k (3 seqs x 512) | 0.0240 (top1 95.12), identical x5 | 0.0249 (top1 95.44), identical x2 | 0.025 |

TP2 fp4 = NVFP4/FP8 prompt slices + the 3xTF32 chunked GDN kernel on the ranks (the default; `FE_GDN_CHUNK=0` takes
the recurrence: code 0.0207, text 0.0398). TP1 bf16: bf16 prompt rows, the EXL3 decode GEMV.


Determinism:

- two runs of fp4 TP2 `logits` (8 text sequences, 1024 prompt + 256 decode rows) give bit-identical logits;
- long16k gives the same KL every run (fp4 5 runs, bf16 2);
- every per-step upload is pinned and asynchronous on the engine stream, and the TP / follower / fabric code makes no
  per-call host-to-device copies. A synchronous pageable copy is not ordered with the engine stream, and once made a
  long16k run read 0.0294;
- a request's greedy output does not depend on what it is batched with: alone, batched, or arriving staggered so its
  prompt chunks share forwards with other requests' decode rows, it is the same text, TP1 and TP2, fp4 and bf16
  prompt rows.

### Prefill and plain decode (development measurement)

`TP_ENV="FE_PROMPT=fp4" tools/tp2.sh $FERRITE_EXL3_DIR bench 2048,8192,32768 128 4` (EXL3 decode with the tuned K
splits, NVFP4/FP8 prompt slices, chunked GDN on the ranks; greedy, no draft):

| context | prefill tok/s | decode tok/s (ms/tok) |
|---|---|---|
| 2K | 3669 | 33.7 (29.7) |
| 8K | 3466 | 32.7 (30.6) |
| 32K | 2880 | 28.8 (34.8) |
| 4 streams (512 ctx) | | 119.1 aggregate, 29.8 a stream |

The same bench with bf16 prompt slices: 1378 / 1846 / 1673 tok/s prefill; NVFP4 slices with the GDN recurrence
for prompt rows (`FE_GDN_CHUNK=0`): 3311 / 3103 / 2643.


K splits: the EXL3 matrix-vector kernel (GEMV) splits long input dimensions across thread groups, and the tuned splits
are the default.

- `FE_EXL3_NOSPLIT=1` forces one split, which costs TP2 11% decode: 30.1 / 29.3 / 26.1 tok/s, 4 streams 102.
- The split-K GEMV is deterministic: `exl3-stress` (79 split-K linears in one workspace, random calls, bit-compare)
  ran 1.024M calls on node A and 256K on node B with no difference.
- The fp4 prompt repro (batch 8, 1024 + 256 rows, run twice) was identical in 5 pairs over both nodes.

### MTP speculative decode (development measurement)

TP2, fp4 prompt, greedy, 256 tokens (`TP_ENV="FE_MTP=1 FE_SPEC=d FE_PROMPT=fp4"`):

| | depth 4 | depth 6 |
|---|---|---|
| code (CSV parser prompt) | 93.2 tok/s (2.88 accepted a round) | 90.5 (3.32) |
| prose (story prompt) | 59.5 (1.50) | 52.7 (1.52) |

In the server (`DRAFT=mtp`, fp4 prompt rows, same client bench, 4 x 128K, MTP depth 8):

| | code | prose |
|---|---|---|
| 2K, 1 stream | 117.4 tok/s | 78.0 tok/s |
| 2K, 4 streams | 197.0 aggregate | 129.1 aggregate |
| 32K, 1 stream | 93.2 | 53.4 |

GDN replay (verify rows run the recurrence read-only, commit replays the kept rows; the replay buffers and
kernels are per rank) gives the same text as `FE_GDN_REPLAY=0` (CLI MTP depth 4 / 8, scheduler sched-test), and
in the server takes prose 50.7 -> 78.0 and 4-stream code 183 -> 197.

### Long context and load (development measurement, a 2026-11-04 build)

Server at 4 x 128K (MTP depth 4, thinking off, temperature 0; the figures of each request, from fe-server):

| load | first token | decode |
|---|---|---|
| 4 concurrent short prompts | 0.08-0.23 s | 131.7 tok/s aggregate, 36.5-43.2 a stream |
| one 32K prompt | 11.7 s (~2.7K tok/s) | 45.9 tok/s |
| one 119K prompt | 68.2 s (~1.74K tok/s) | 30.5 tok/s |
| 4 concurrent 30K prompts (121K tokens) | 10.9 / 23.4 / 39.0 / 48.1 s (queued prefills, ~2.5K tok/s) | ~3.5 tok/s while others prefill, 40 alone |

A 118850-token needle prompt (a code sentence at ~75% depth in a repeated paragraph) was answered correctly. A
tool call (`get_weather {"city": "Paris"}`, finish `tool_calls`), thinking on (reasoning + answer) and off (answer
only) work as on one node, and a stream kept decoding at 15.5 chunks/s while a 30K prompt prefilled (18.3 s,
scheduler fairness). Available memory with the model loaded (MTP, 4 x 128K): 74 GB on node A, 78 GB on node B.

## How it works

### Driving the pair

Both ranks run the same `ferrite` command; each ends every forward with identical residuals (the running hidden
state) and logits. There are two ways to drive the pair:

- **Lockstep.** Both ranks run the same command (CLI generate / logits / bench). The ranks exchange a hash of their
  argv at connect and refuse to start if they differ.
- **Follow** (`fe_model::tp::load_engine(.., follower = true)`). Rank 1 takes rank 0's engine sizes, and
  `Engine::follow` replays every state-changing call (forward_ex, commit, reset_slot, mtp_forward) that rank 0 sends
  over the bootstrap link before running it. Read-only calls stay on rank 0. The server runs this way.

Details:

- Both nodes must run the same binary (`tools/tp2.sh` compares sha256s); the bootstrap gives up after
  `FE_TP_CONNECT_SECS` (900).
- NCCL connects lazily, so `Tp::with_fabric` runs one all-reduce per message size class and one all-gather at attach
  (without it the first 2K prompt paid ~0.3 s).
- Both ranks set `oom_score_adj` to 500. A process started over ssh inherits sshd's score, which is -1000 on some
  systems, and an out-of-memory event would then reboot the node instead of killing the process.

The commands (run on node A; the same binary and model folder on both nodes):

```bash
. tools/site.sh
tools/tp2.sh $FERRITE_EXL3_DIR generate "The capital of France is" 64
tools/tp2.sh $FERRITE_EXL3_DIR bench 2048,8192,32768 128 4
tools/kl-tp2.sh $FERRITE_EXL3_DIR code      # | text | long16k
TP_ENV="FE_PROMPT=fp4" tools/tp2.sh $FERRITE_EXL3_DIR bench 2048,8192,32768 128 4   # NVFP4/FP8 prompt rows (rank slices)
TP_ENV="FE_PROMPT=fp4" tools/kl-tp2.sh $FERRITE_EXL3_DIR text
TP_ENV="FE_MTP=1 FE_SPEC=4" tools/tp2.sh $FERRITE_EXL3_DIR generate "def fibonacci(n):" 256     # MTP speculative decode
# serving: rank 0 runs the scheduler / server, rank 1 replays its engine calls (`ferrite <model> follow`)
FOLLOW=$FERRITE_EXL3_DIR tools/tp2.sh target/release/ferrite $FERRITE_EXL3_DIR sched-test 6
tools/serve-tp2.sh                     # the OpenAI server (model "qwen3.8-27b") on node A :8000 as TP2
TP2_DRY=1 tools/tp2.sh <arguments>     # print what both ranks would run, run nothing
# by hand: FE_TP_WORLD=2 FE_TP_RANK=0|1 FERRITE_TP_ADDR=<rank 0 fabric address>:<port> on both ranks, plus the
#          NCCL_IB_GID_INDEX / NCCL_SOCKET_IFNAME / NCCL_IB_HCA and FERRITE_ROCE_* settings tools/site.sh prints
```

### Split (fe_exl3::tp::qwen35_split, fe_model::tp::plain_rows)

GDN is Gated DeltaNet, the linear-attention layer of Qwen3.8 (`linear_attn` below).

| tensor | rank r keeps |
|---|---|
| self_attn.q_proj (24 heads x [q 256 \| gate 256]) | heads 12r..12r+12 (columns) |
| self_attn.k_proj / v_proj (4 kv heads x 256) | kv heads 2r..2r+2 (columns) |
| self_attn.o_proj | input rows of its heads (row-parallel, reduced) |
| linear_attn.in_proj_qkv ([q 16x128 \| k 16x128 \| v 48x128]) | half of each section (8 key heads, 24 value heads) |
| linear_attn.in_proj_z, in_proj_a, in_proj_b, A_log, dt_bias | its 24 value heads |
| linear_attn.conv1d | the channels of its q / k / v halves |
| linear_attn.out_proj | input rows of its value heads (row-parallel, reduced) |
| mlp.gate_proj / up_proj | columns 8704r..8704r+8704 |
| mlp.down_proj | input rows 8704r.. (row-parallel, reduced) |
| lm_head | vocabulary half (gathered) |
| embeddings, norms, q/k norms, GDN norm | whole |

Kernels: the head counts of k-core's attention / flash / GDN kernels live in `kernels/k-core/src/dims.rs`;
`kernels/k-core-tp2` builds the same sources with 12 / 2 attention heads and 8 / 24 GDN heads
(`Kernels::with_tp2`). The same for the chunked GDN prompt kernel: k-gdn's body is `kernels/k-gdn/src/chunk.rs`
with its head counts in `dims.rs`, and `kernels/k-gdn-tp2` builds it with 8 / 24 heads and conv rows of 5120
(`fe_gdn::Gdn::load_tp2`, taken by `Engine::attach_tp`).

### Reductions (`Engine::resid_linear`)

Where the two halves of a layer's output are added, by kind of row:

- **Decode and verify rows** run the EXL3 GEMV with a bf16 partial (`FE_TP_PART=f32`: f32 partials) and the RoCE
  one-shot exchange (`fe_fabric::Fabric::reduce_bf16_to_bf16`). The sum is hidden = bf16(hidden + (p0 + p1)), in rank
  order, with identical bits on both ranks; crates/fe-fabric has its timings.
- **A mixed batch.** The leading decode / verify rows take that same path in passes of up to 64 rows (16 with f32
  partials), so their bits do not depend on the batch.
- **Prompt rows** run the bf16 or NVFP4/FP8 GEMM on the rank's slice with a bf16 partial, an NCCL bf16 all-reduce,
  then the residual add (the fused NVFP4 MLP front included: its down GEMM writes the partial).
- **NVFP4/FP8 prompt weights** (`FE_PROMPT=fp4`) are sliced at load exactly like the EXL3 linears
  (`fe_model::tp::fp4_slice` -> `fe_fp4::Slice`): output rows for column-parallel, k columns of codes and block scales
  for row-parallel. lm_head keeps only its trellis form; logits rows never take the prompt path.
- **lm_head.** Each rank computes its 124160 logits, then an NCCL all-gather and an interleave kernel.
- **MTP.** The MTP layer is split like the others, mtp.fc is replicated, and the draft head is a replicated slice of
  lm_head's first 65536 ids on both ranks, so both ranks draft and accept identically.
- **DFlash2.** The drafter runs split across the ranks too: half the heads and MLP features of every draft layer and
  of the draft K/V cache on each; fc, the convolutions and norms are replicated.

## Future work

Not done yet. The NVFP4 prefill figures quoted here (3622 / 3448 / 2864) are those of this timing comparison, not the
ones of the prefill table above (3669 / 3466 / 2880).

- Prompt-row reductions are 19-23% of TP2 prefill (dropping them, timing only: 4714 / 4420 / 3523 tok/s at 2K / 8K
  / 32K vs 3622 / 3448 / 2864). Overlapping them on a second stream (`Tp::reduce_rows`, up to 4 row chunks,
  bit-identical logits) was slower on GB10 (2654-3050 tok/s at 2K with 4 / 2 / 1 NCCL channels): NCCL's kernel
  competes with the GEMMs for SMs. The next try is an SM-free path for big messages: RDMA writes of the bf16
  partials by the proxy thread (as the decode-row RoCE path does), with only the add on the GPU.
