# fe-xdf: DFlash2 block drafter on the Arc Pro B70

Speculative decoding for fe-xe (Qwen3.8-27B, XQ mix5.0) with z-lab's Qwen3.8-27B-DFlash2 drafter (5 Qwen3 layers, GQA 32/8,
head 128, sliding window 2048, non-causal inside the block, two-tap dynamic convolutions, top-16 candidate selector). The CUDA
original is crates/fe-dflash; this is the B70 port, all Rust:

- `quant`: the drafter linears as XQ tensors, round to nearest (Hadamard-128 rotation with a fixed sign vector per input width,
  symmetric 4-bit codes, fp16 g128 scales): they run on kx-xq's decode GEMV like the target's weights. 0.93 GB packed, cached
  as `ferrite-xq<bits>.bin` in the drafter folder (`FE_DF_CACHE` names another file; `xdf quant`).
- `draft`: device weights + draft K/V rings (4096 positions a layer), `absorb` (target feature taps of layers 5 / 19 / 33 / 47 /
  61 -> fc -> hidden_norm -> per-layer K / V), `draft` (selector chain) and `draft_tree` (best-first tree of up to 7 nodes), the
  drafter-window snapshots that ride along with fe-server's GDN checkpoints.
- `spec`: DfSpec, the decode loop (prefill with taps of the last 2048 positions before every cut, draft, verify window or verify
  tree, commit, absorb), the context-copy rule of the MTP policy with the drafter's first proposal as the veto.
- `tree`: `grow_tree` (the CUDA drafter's best-first grow) and an offline tree simulator (`FE_DF_TREESIM`).
- kernels: `kernels-xe/kx-dflash` (dynamic conv, per-head norm + RoPE + ring write, draft attention on the matrix engine,
  top-16, selector walk, feature tap); norms / embedding / SwiGLU are kx-core's, the tree verify kernels are kx-core's
  (tree.rs / dec.rs), the window forward is fe-xe's (`Engine::taps`, `forward_tree`, `commit_tree`).

Lossless: the target verifies every proposal with the row-invariant window kernels (trees: every row == its root path decoded
alone), so greedy output == plain greedy (`xdf ... --check`, gate.py through fe-server).

```
xdf quant                                   # drafter linears -> XQ cache (CPU)
xdf <XQ model> gen <ids.txt> <n> --check    # one prompt, DFlash2 vs plain greedy
xdf <XQ model> bench <n> <ids.txt>... --configs 8:7 --check   # several prompts, one line each
xdf <XQ model> dprof <ids.txt> [N]          # draft / absorb wall time, FE_XE_PROFILE=1 per kernel
fe-server --engine xe ...                    # the server path (XeEngine): DFlash2 by default, FE_XE_DRAFT=mtp for MTP drafting
```

Knobs: `FE_DRAFT_DIR` (the Qwen3.8-27B-DFlash2 folder, required), `FE_DF_TREE` (default 1; 0 = selector
chains), `FE_DF_TREE_BELOW` (trees only below this accepted-drafts EMA; default always), `FE_DF_COPY` (context copy, default 1),
`FE_DF_ADAPT` (verify length from fe-xe's Adapt over `FE_DF_COST`, default 1), `FE_DF_BLOCK` (default the checkpoint's 8),
`FE_DF_BITS` / `FE_DF_BITS_FC` / `FE_DF_BITS_SMALL` (4 / 4 / 8), `FE_DF_VOCAB` (draft head span, default 131072),
`FE_DF_HEAD=q4` (4-bit copy of the BF16 lm_head as draft head: off, it costs code acceptance), `FE_DF_ATTN` (draft attention:
4 = matrix engine, default), `FE_DF_PROF`, `FE_DF_TREESIM`.

## Results

One Arc Pro B70 (32 GB) capped at 180 W and 2000 MHz, XQ mix5.0, all-Rust kernels, greedy decoding, tokens per second.

Standalone (`xdf bench`, four fixed prompts, 500 tokens, every run identical to plain greedy; plain decoding is 27.6):

| prompt | selector chain of 7 | 7-node verify tree | MTP drafting |
|---|---|---|---|
| code | 114.1 (4.24 accepted a window) | 114.1 (4.31) | 85.1 |
| text | 61.5 (1.84) | 70.3 (2.30) | 55.4 |
| edit, rename (context copy) | 232 | 233 | 179.4 |
| edit, hints (context copy) | 184 | 186 | 161.7 |

Per window (code): draft ~3.8 ms (drafter XQ GEMVs ~1.8 ms + 6-bit head over 131072 ids 0.94 ms + attention 0.3 ms), 8-row
verify 40.0 ms (a plain step is 36.2), commit ~1.1 ms, absorb 0.3 ms. 16-row windows cost ~1.6 plain steps on the B70 (the
8x2 XQ GEMV), so blocks and trees stay at 8 rows.

Through fe-server (`--engine xe`, trees + context copy, 128K context):

| row | DFlash2 (trees) | MTP drafting |
|---|---|---|
| decode, code / text / edit rename / edit hints | 119.7 / 71.7 / 241.8 / 199.6 | 90.0 / 58.8 / 188.8 / 170.2 |
| device memory at load (131072 context, 8 checkpoint slots) | 28.6 GB (drafter 1.42 GB, 8 x 42 MB drafter-window snapshots; tree buffers +28 MB on first use) | 26.9 GB |
| long-context losslessness (`xdf gen --check`, trees) | 57K prompt 300 tokens, 95K prompt 200 tokens: identical to plain greedy | |

Offline tree-constant grid over recorded rounds (`xdf treesim`, 4 prompts x 1000 tokens): branch 4 / edge 0.6 / scale 1.5, the
constants of the CUDA drafter (from TensorFold), are already the best.

## Measured and dropped (all lossless; standalone xdf bench unless noted)

| experiment | result | why dropped |
|---|---|---|
| block 16, verify 15 drafts | code 94.5 (5.30 accepted) vs block 8 113.3 (4.24); text 44.0 vs 60.9 | 16-row verify windows cost ~1.6 plain steps (8x2 XQ GEMV: int5 8x2 runs at 54 % of the DRAM bound, 8x1 at 84 %) |
| block 16, verify 7 | code 105.5 (4.08 accepted), text 58.4 | the extra mask rows change the first 7 predictions (non-causal block) and cost draft time |
| 6-bit drafter linears | code 111.9 (4.30), text 61.3 (1.92); draft +1 ms | acceptance +1.4 / +4 % does not pay for 0.45 GB more a draft |
| 8-bit drafter linears | code 109.6 (4.35), text 59.8 (1.94); draft +2.5 ms | same |
| full-vocabulary draft head (248320 ids) | code 111.3 (4.24), text 60.3 (1.85); draft +0.8 ms | no acceptance gain over the first 131072 ids |
| 4-bit RTN copy of the BF16 lm_head as draft head | draft -0.9 ms, code 110.5 (4.03) vs 114.4 (4.24), text 62.1 vs 61.6 | code acceptance -5 % |
| per-(head, row) draft attention / SLM-shared / block reads | 243 / 110 / 125 us a layer | the matrix-engine version is 60 us at the same acceptance |
| fused q\|k\|v and gate\|up draft GEMVs | draft 4.10 vs 3.83 ms | split_for of the fused shapes is less parallel; only the absorb's k\|v of all layers stays fused (0.29 -> 0.22 ms) |
| drafter on its own Level Zero queue (draft beside the engine's commit) | code 114.0 / text 70.3, identical outputs | the B70 serializes the two lists: no gain |
| trees only below an accepted-drafts EMA of 4 (chains above) | code 112.3 / text 69.5 vs trees always 114.1 / 70.3 | trees are as fast as chains on code once the draft attention runs on the matrix engine |
| tree constants (branch 2..6, edge 0..1, scale 1..2.5) on recorded rounds | 4 / 0.6 / 1.5 (the CUDA drafter's constants) stays | no other setting accepted more |
