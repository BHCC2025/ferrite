# fe-dflash — DFlash2 block drafter

Speculative decoding with z-lab's DFlash2 drafter (`$FERRITE_DRAFT_DIR`, z-lab/Qwen3.8-27B-DFlash2: 5 Qwen3 layers, GQA 32/8, head 128,
sliding window 2048, non-causal inside the block, two-tap grouped dynamic convolutions around every
sublayer, a top-16 candidate selector), on the ferrite engine. Kernels in `kernels/k-dflash`; norms, SwiGLU
and the embedding reuse k-core; the draft logits come from the target's NVFP4 lm_head (fe-fp4).

```
# single stream, own loop (prints text, then timing / acceptance / ids on stderr)
. tools/site.sh                     # FERRITE_EXL3_DIR / FERRITE_DRAFT_DIR / ... (README table)
FE_CHAT=1 FE_MTP=0 FE_PROMPT=fp4 ./target/release/dflash-generate "$FERRITE_EXL3_DIR" "<user message>" 400 [--plain]
FE_DRAFT=dflash FE_MTP=0 FE_PROMPT=fp4 ./target/release/ferrite "$FERRITE_EXL3_DIR" generate "<raw prompt>" 300
# scheduler / server (sched::Drafter, one draft cache per slot; all decoding streams draft in one batched pass)
FE_DRAFT=dflash FE_TREE=1 FE_DF_BLOCK=16 FE_SPEC=16 FE_PROMPT=fp4 ./target/release/ferrite "$FERRITE_EXL3_DIR" sched-test 6
FERRITE_DRAFT=dflash scripts/serve.sh  # one node (or tools/serve-tp2.sh); auto: DFlash2 for <= 2 streams, MTP above
```

Flow per round: draft block [anchor, mask x 7] at positions pos..pos+7 -> NVFP4 lm_head over the first 128K
ids on rows 1..7 -> top-16 -> selector walk from the anchor -> 7 proposals; one target forward over
[anchor, proposals] (8 rows, EXL3 GEMV; the GDN recurrence runs read-only and saves its inputs) -> keep the
matching prefix + the target's next token (greedy accept-while-match) -> `Engine::commit` (replays the kept
rows' GDN recurrence) -> the kept rows' target taps (outputs of layers 5, 19, 33, 47, 61) are projected (fc,
hidden_norm, per-layer k/v + k_norm + RoPE) into the draft K/V cache (`absorb`). Prompt rows are absorbed
after each prefill chunk.

Engine hooks (crates/fe-model): `set_feature_taps(&[layers])`, `feature_taps()`, `lm_head_rows(x, rows)`;
GDN replay (`FE_GDN_REPLAY`, default on).

Drafter weights: the big linears are quantized at load to affine 4-bit, group 64 (MLX style, as TensorFold's
drafter), repacked for `fe_df_gemv4t` (W4A16 on tensor cores, <= 8 rows: 16 weight columns x 8 x rows per
mma, every load a coalesced 16 B per lane); fc / k / v keep a bf16 copy for prompt-size absorbs. The draft
model's RMSNorm is plain `w * x` (Qwen3), not the target's Qwen3.5 `(1 + w) * x`. Draft K/V caches are
rings of 4096 positions per slot (>= window + block).

Knobs: `FE_DF_VOCAB` (draft head span, default 131072; 65536 drops code acceptance 5.14 -> 4.68, 0 = all),
`FE_DF_HEAD=exl3` (EXL3 head, full vocabulary), `FE_DF_W4=0` (bf16 drafter), `FE_DF_GEMV=simt` (warp per
column W4 GEMV), `FE_DF_GEMV_TILES=4`, `FE_DF_ATTN_SPLIT` (split-K draft attention parts, default 8; 1 = one
pass), `FE_DF_RING=0` (linear draft caches), `FE_DF_PROF=1` (per-round draft / verify / commit split),
`FE_DF_TRACE`, `FE_DF_DEBUG`, `FE_DF_BLOCK` (draft block, 2..16; the servers use 16), `FE_DF_ATTN_MMA=0` (the scalar
split-K attention instead of the tensor-core one), `FE_DF_ATTN_BLOCKS` (tensor-core draft attention: splits for
about this many blocks, default 32; 192 = the earlier sizing), `FE_DF_ATTN_MMA_SPLIT` (fixes the split count), `FE_DF_PART=f32` (TP2: f32 instead of bf16
reduction partials), `FE_DF_TP=0` (TP2: rank 0 drafts alone with the whole drafter), `FE_DF_PROF2=1` (CUDA-event
split of every batched draft by phase), `FE_DF_DUMP=<file>` (append every tree pass's top-16 lists, to compare drafts
across builds byte for byte). The checkpoint folder is `FERRITE_DRAFT_DIR` (the older `FE_DRAFT_DIR` still works).

## First results (one node, TP1, block 8 chains, EXL3 3.0 mix target, FE_PROMPT=fp4, greedy, thinking off)

| prompt | plain | DFlash2 | accepted / round (of 7) | output vs plain |
|---|---|---|---|---|
| palindrome function + tests (code, 300 tok) | 19.2 tok/s | 96.8 tok/s | 5.23 | identical |
| lighthouse short story (prose, 400 tok) | 19.2 tok/s | 39.4 tok/s | 1.59 | identical |
| 7.6K-token code prompt, explain (200 tok) | - | 47.3 tok/s | 2.39 | - |

fe-server on :8011 (same prompts, server-reported decode): DFlash2 94.3 code / 39.1 prose, MTP SPEC=8
67.9 / 34.7, identical texts. Round (code): draft 6.9 ms + 8-row verify 54.9 ms + commit/absorb 2.5 ms; the
verify is ~54 ms of EXL3 GEMV (about 190 GB/s over the 3.0 bpw weights), so that is where the next gains are.

History of the round: 83 ms (bf16 drafter, 30 ms draft) -> W4 SIMT GEMV (15 ms draft) -> tensor-core W4
GEMV + 128K head (7.8 ms) -> GDN replay (verify 65 -> 55 ms) -> split-K draft attention (7.0 ms) ->
one-launch replay (commit 3.1 -> 2.5 ms).

## Since then

- 16-row blocks (as TensorFold and exllamav3 run them, past the 8 the checkpoint was trained with), with the
  batch-variance fixes that keep greedy output identical to plain decoding.
- Best-first verify trees (`FE_TREE=1`): the grow logic and constants of TensorFold's `cuda_draft_tree.zig`, with
  tree attention, GDN tree kernels and `commit_tree` in fe-model.
- One batched draft pass for every decoding stream (`draft_tree_many`) and a verify-row budget in the scheduler
  that splits the rows by each stream's recent acceptance.
- TP2: each rank holds half of every draft layer (heads, MLP, K/V) and scores half of the 128K-id NVFP4 draft head (64K ids);
  the row-parallel partials (bf16 by default) are reduced over the RoCE link and the top-16 lists exchanged. Rank 1
  follows through `Call::Draft` / `Call::Absorb`.
- Tensor-core draft attention (`fe_df_attn_mma`: one block per draft kv head and split, the 4 q heads of the group
  sharing every K/V tile through shared memory).

Draft cost per round on two nodes (2K-token context, `FE_DF_PROF2`): 4.9-5.1 ms for one stream, about 7.3 ms for
4 streams. Server throughput on two nodes (2K-token prompts, 500 tokens out, medians of 3): code 256 / prose 77
tok/s for one stream, code 309 / prose 168 tok/s aggregate for 4 streams.

Not done: sampling-aware drafting (sampled streams draft chains, not trees).
