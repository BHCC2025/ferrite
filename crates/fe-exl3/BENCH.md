# fe-exl3 decode GEMV: GB10 measurements

Machine: one DGX Spark (GB10 sm_121a, 48 SMs, 121 GB unified LPDDR5X), CUDA 13.0, driver 580.
Model: Qwen3.8-27B EXL3 3.0 bpw mix (mul1 codebook; 2/3/4-bit layers, 5-bit lm_head, 4-bit MTP).
Date: 2026-11-03 (the plans in the table are today's plans.rs).
The kernel design follows exllamav3's EXL3 kernels and TensorFold's exl3_linear / exl3_linear_wc (CREDITS.md).

## Result

| rows | GEMV, all model linears | rot_in + GEMV |
|---|---|---|
| 1 (decode) | 47.61 ms | 48.27 ms |
| 16 (speculative verify) | 50.50 ms | 50.57 ms |

("all model linears" = every EXL3 linear of one token, count-weighted: 64 layers, lm_head, the MTP layer once;
run-to-run spread of these sums is about +-2%.) Device streaming roof (`exl3-probe`, 16-byte non-allocating
loads over 1 GB): **239 GB/s**. At 1 row the wide 4-bit shapes and lm_head reach 90-98% of it, the 3-bit shapes
~200-215 GB/s (the mul1 decode runs on the half-rate integer pipe: ~64 lanes/clk/SM for IMAD / IDP / LOP3 / HFMA2
on GB10, measured), and 2-bit is decode-bound (~0.9 G weights/s/SM). The narrow k / v_proj (2-2.6 MB a call)
reach 115-171 GB/s: at that size, launch and the split-K handoff are a large share of the time.

## Method

- `exl3-bench <model> "" 300`: every distinct (kind, bits) shape of the checkpoint, the `Linear` with its tuned
  plan (`src/plans.rs`) on the real weights, launched with programmatic dependent launch as the engine runs it.
- Each call cycles over enough device copies of the weight (>= 96 MB) that nothing is served from L2. 300 calls
  back to back, CUDA events, mean (the table: two such rounds, averaged). "GEMV" = the trellis GEMV alone on a
  rotated input; "fwd" = the input rotation + GEMV. GB/s = trellis bytes / GEMV time.
- Plan = `sk` K splits, `wk` warps a program, kernel `v` (1: x as the mma A operand; 8: gemv8, W^T as A and
  8 x rows as B; 9: gemv8 with a two-step register prefetch), `o` launch-bounds variant.

## Per shape

| shape | bits | k x n | uses/token | plan | MB | rows | GEMV us | GB/s | of roof | fwd us |
|---|---|---|---|---|---|---|---|---|---|---|
| lm_head | 5 | 5120 x 248320 | 1 | sk1 wk8 v9 o0 | 794.62 | 1 | 3383.0 | 235 | 98% | 3380.9 |
| lm_head | 5 | 5120 x 248320 | 1 | sk1 wk8 v9 o0 | 794.62 | 16 | 3498.7 | 227 | 95% | 3486.1 |
| layers.linear_attn.in_proj_qkv | 2 | 5120 x 10240 | 1 | sk1 wk8 v8 o2 | 13.11 | 1 | 66.8 | 196 | 82% | 68.1 |
| layers.linear_attn.in_proj_qkv | 2 | 5120 x 10240 | 1 | sk1 wk8 v8 o2 | 13.11 | 16 | 71.3 | 184 | 77% | 73.6 |
| layers.linear_attn.in_proj_qkv | 3 | 5120 x 10240 | 47 | sk1 wk8 v8 o2 | 19.66 | 1 | 94.7 | 208 | 87% | 96.3 |
| layers.linear_attn.in_proj_qkv | 3 | 5120 x 10240 | 47 | sk1 wk8 v8 o2 | 19.66 | 16 | 96.7 | 203 | 85% | 97.9 |
| layers.linear_attn.in_proj_z | 2 | 5120 x 6144 | 1 | sk2 wk8 v8 o2 | 7.86 | 1 | 41.5 | 190 | 79% | 44.0 |
| layers.linear_attn.in_proj_z | 2 | 5120 x 6144 | 1 | sk2 wk8 v8 o2 | 7.86 | 16 | 46.2 | 170 | 71% | 46.4 |
| layers.linear_attn.in_proj_z | 3 | 5120 x 6144 | 47 | sk2 wk8 v8 o2 | 11.80 | 1 | 54.8 | 215 | 90% | 57.2 |
| layers.linear_attn.in_proj_z | 3 | 5120 x 6144 | 47 | sk2 wk8 v8 o2 | 11.80 | 16 | 57.3 | 206 | 86% | 59.1 |
| layers.linear_attn.out_proj | 3 | 6144 x 5120 | 29 | sk2 wk8 v8 o2 | 11.80 | 1 | 58.3 | 202 | 85% | 58.9 |
| layers.linear_attn.out_proj | 3 | 6144 x 5120 | 29 | sk2 wk8 v8 o2 | 11.80 | 16 | 62.0 | 190 | 79% | 61.1 |
| layers.linear_attn.out_proj | 4 | 6144 x 5120 | 19 | sk2 wk8 v8 o2 | 15.73 | 1 | 71.3 | 221 | 92% | 72.5 |
| layers.linear_attn.out_proj | 4 | 6144 x 5120 | 19 | sk2 wk8 v8 o2 | 15.73 | 16 | 75.4 | 209 | 87% | 77.7 |
| layers.mlp.down_proj | 2 | 17408 x 5120 | 2 | sk2 wk8 v8 o2 | 22.28 | 1 | 108.0 | 206 | 86% | 106.5 |
| layers.mlp.down_proj | 2 | 17408 x 5120 | 2 | sk2 wk8 v8 o2 | 22.28 | 16 | 118.5 | 188 | 79% | 120.4 |
| layers.mlp.down_proj | 3 | 17408 x 5120 | 48 | sk2 wk8 v8 o2 | 33.42 | 1 | 157.9 | 212 | 89% | 164.3 |
| layers.mlp.down_proj | 3 | 17408 x 5120 | 48 | sk2 wk8 v8 o2 | 33.42 | 16 | 170.4 | 196 | 82% | 169.3 |
| layers.mlp.down_proj | 4 | 17408 x 5120 | 14 | sk2 wk8 v8 o2 | 44.56 | 1 | 197.5 | 226 | 95% | 196.4 |
| layers.mlp.down_proj | 4 | 17408 x 5120 | 14 | sk2 wk8 v8 o2 | 44.56 | 16 | 217.1 | 205 | 86% | 221.2 |
| layers.mlp.gate_proj | 2 | 5120 x 17408 | 14 | sk2 wk8 v8 o2 | 22.28 | 1 | 106.7 | 209 | 87% | 111.6 |
| layers.mlp.gate_proj | 2 | 5120 x 17408 | 14 | sk2 wk8 v8 o2 | 22.28 | 16 | 124.4 | 179 | 75% | 121.0 |
| layers.mlp.gate_proj | 3 | 5120 x 17408 | 49 | sk1 wk4 v9 o0 | 33.42 | 1 | 160.6 | 208 | 87% | 161.5 |
| layers.mlp.gate_proj | 3 | 5120 x 17408 | 49 | sk1 wk4 v9 o0 | 33.42 | 16 | 168.0 | 199 | 83% | 164.4 |
| layers.mlp.gate_proj | 4 | 5120 x 17408 | 1 | sk1 wk4 v9 o0 | 44.56 | 1 | 199.8 | 223 | 93% | 200.8 |
| layers.mlp.gate_proj | 4 | 5120 x 17408 | 1 | sk1 wk4 v9 o0 | 44.56 | 16 | 217.2 | 205 | 86% | 209.8 |
| layers.mlp.up_proj | 2 | 5120 x 17408 | 14 | sk2 wk8 v8 o2 | 22.28 | 1 | 105.4 | 211 | 88% | 108.0 |
| layers.mlp.up_proj | 2 | 5120 x 17408 | 14 | sk2 wk8 v8 o2 | 22.28 | 16 | 123.1 | 181 | 76% | 122.3 |
| layers.mlp.up_proj | 3 | 5120 x 17408 | 49 | sk1 wk4 v9 o0 | 33.42 | 1 | 160.0 | 209 | 87% | 158.6 |
| layers.mlp.up_proj | 3 | 5120 x 17408 | 49 | sk1 wk4 v9 o0 | 33.42 | 16 | 164.2 | 204 | 85% | 164.8 |
| layers.mlp.up_proj | 4 | 5120 x 17408 | 1 | sk1 wk4 v9 o0 | 44.56 | 1 | 198.2 | 225 | 94% | 198.8 |
| layers.mlp.up_proj | 4 | 5120 x 17408 | 1 | sk1 wk4 v9 o0 | 44.56 | 16 | 204.9 | 218 | 91% | 207.1 |
| layers.self_attn.k_proj | 3 | 5120 x 1024 | 9 | sk10 wk8 v8 o2 | 1.97 | 1 | 14.2 | 139 | 58% | 14.8 |
| layers.self_attn.k_proj | 3 | 5120 x 1024 | 9 | sk10 wk8 v8 o2 | 1.97 | 16 | 17.2 | 115 | 48% | 19.1 |
| layers.self_attn.k_proj | 4 | 5120 x 1024 | 7 | sk5 wk8 v9 o0 | 2.62 | 1 | 15.4 | 171 | 72% | 16.9 |
| layers.self_attn.k_proj | 4 | 5120 x 1024 | 7 | sk5 wk8 v9 o0 | 2.62 | 16 | 17.0 | 154 | 64% | 20.9 |
| layers.self_attn.o_proj | 3 | 6144 x 5120 | 9 | sk2 wk8 v8 o2 | 11.80 | 1 | 55.7 | 212 | 89% | 56.8 |
| layers.self_attn.o_proj | 3 | 6144 x 5120 | 9 | sk2 wk8 v8 o2 | 11.80 | 16 | 60.3 | 196 | 82% | 61.8 |
| layers.self_attn.o_proj | 4 | 6144 x 5120 | 7 | sk2 wk8 v8 o2 | 15.73 | 1 | 71.9 | 219 | 92% | 71.0 |
| layers.self_attn.o_proj | 4 | 6144 x 5120 | 7 | sk2 wk8 v8 o2 | 15.73 | 16 | 80.2 | 196 | 82% | 79.3 |
| layers.self_attn.q_proj | 3 | 5120 x 12288 | 9 | sk1 wk8 v8 o2 | 23.59 | 1 | 110.9 | 213 | 89% | 113.0 |
| layers.self_attn.q_proj | 3 | 5120 x 12288 | 9 | sk1 wk8 v8 o2 | 23.59 | 16 | 115.8 | 204 | 85% | 115.8 |
| layers.self_attn.q_proj | 4 | 5120 x 12288 | 7 | sk1 wk8 v8 o2 | 31.46 | 1 | 139.3 | 226 | 95% | 138.2 |
| layers.self_attn.q_proj | 4 | 5120 x 12288 | 7 | sk1 wk8 v8 o2 | 31.46 | 16 | 155.6 | 202 | 85% | 154.3 |
| layers.self_attn.v_proj | 3 | 5120 x 1024 | 9 | sk10 wk8 v8 o2 | 1.97 | 1 | 13.4 | 147 | 62% | 15.5 |
| layers.self_attn.v_proj | 3 | 5120 x 1024 | 9 | sk10 wk8 v8 o2 | 1.97 | 16 | 15.8 | 125 | 52% | 19.1 |
| layers.self_attn.v_proj | 4 | 5120 x 1024 | 7 | sk5 wk8 v9 o0 | 2.62 | 1 | 15.9 | 164 | 69% | 16.9 |
| layers.self_attn.v_proj | 4 | 5120 x 1024 | 7 | sk5 wk8 v9 o0 | 2.62 | 16 | 17.0 | 154 | 64% | 19.7 |
| mtp.fc | 4 | 10240 x 5120 | 1 | sk10 wk4 v9 o0 | 26.21 | 1 | 121.6 | 216 | 90% | 119.3 |
| mtp.fc | 4 | 10240 x 5120 | 1 | sk10 wk4 v9 o0 | 26.21 | 16 | 135.4 | 194 | 81% | 138.6 |
| mtp.layers.mlp.down_proj | 4 | 17408 x 5120 | 1 | sk2 wk8 v8 o2 | 44.56 | 1 | 199.0 | 224 | 94% | 198.0 |
| mtp.layers.mlp.down_proj | 4 | 17408 x 5120 | 1 | sk2 wk8 v8 o2 | 44.56 | 16 | 216.7 | 206 | 86% | 221.6 |
| mtp.layers.mlp.gate_proj | 4 | 5120 x 17408 | 1 | sk1 wk4 v9 o0 | 44.56 | 1 | 197.4 | 226 | 95% | 198.6 |
| mtp.layers.mlp.gate_proj | 4 | 5120 x 17408 | 1 | sk1 wk4 v9 o0 | 44.56 | 16 | 205.3 | 217 | 91% | 215.3 |
| mtp.layers.mlp.up_proj | 4 | 5120 x 17408 | 1 | sk1 wk4 v9 o0 | 44.56 | 1 | 198.1 | 225 | 94% | 199.3 |
| mtp.layers.mlp.up_proj | 4 | 5120 x 17408 | 1 | sk1 wk4 v9 o0 | 44.56 | 16 | 206.0 | 216 | 90% | 207.1 |
| mtp.layers.self_attn.k_proj | 4 | 5120 x 1024 | 1 | sk5 wk8 v9 o0 | 2.62 | 1 | 15.5 | 170 | 71% | 19.0 |
| mtp.layers.self_attn.k_proj | 4 | 5120 x 1024 | 1 | sk5 wk8 v9 o0 | 2.62 | 16 | 17.1 | 154 | 64% | 19.8 |
| mtp.layers.self_attn.o_proj | 4 | 6144 x 5120 | 1 | sk2 wk8 v8 o2 | 15.73 | 1 | 72.1 | 218 | 91% | 71.2 |
| mtp.layers.self_attn.o_proj | 4 | 6144 x 5120 | 1 | sk2 wk8 v8 o2 | 15.73 | 16 | 74.8 | 210 | 88% | 77.5 |
| mtp.layers.self_attn.q_proj | 4 | 5120 x 12288 | 1 | sk1 wk8 v8 o2 | 31.46 | 1 | 139.2 | 226 | 95% | 146.2 |
| mtp.layers.self_attn.q_proj | 4 | 5120 x 12288 | 1 | sk1 wk8 v8 o2 | 31.46 | 16 | 153.6 | 205 | 86% | 153.4 |
| mtp.layers.self_attn.v_proj | 4 | 5120 x 1024 | 1 | sk5 wk8 v9 o0 | 2.62 | 1 | 16.1 | 163 | 68% | 17.0 |
| mtp.layers.self_attn.v_proj | 4 | 5120 x 1024 | 1 | sk5 wk8 v9 o0 | 2.62 | 16 | 17.1 | 153 | 64% | 20.6 |

## Rows 17..64 (several streams' verify batches)

Model linears per forward (`FE_BENCH_ROWS2=32|64 exl3-bench`): rows 32 63.68 ms, rows 64 116.61 ms (rows 1:
~46.3 ms). Above 16 rows each tile is decoded once for the whole pass: 17-24 rows take g3, 25-32 g4, 33-40 g5,
41-48 g6 (one pass each), 49-64 two 32-row g4 passes; a row's bits stay identical to the 1..16-row calls. README.md
has the per-form costs for a TP2 rank.

## TP2 slices (per rank, one Spark's share of a two-Spark split)

Column-parallel q / k / v / gate / up / in_proj_qkv (by q/k/v sections) / in_proj_z / lm_head take half the
output columns, row-parallel o_proj / out_proj / down_proj half the input rows (`Linear::load_cols` /
`load_rows`, checked against the whole tensor by `exl3-tpcheck`). `FE_TUNE_TP2=1 exl3-tune`, rows 1 / 16 (us);
default = the plan rule (`linear::plan`), tuned = the search's best (now in plans.rs):

| k x n | bits | uses | default plan | us m1 / m16 | tuned plan | us m1 / m16 | tuned gain m1 |
|---|---|---|---|---|---|---|---|
| 3072 x 5120 | 3 | 38 | sk4 wk4 v1 o0 | 36.5 / 39.9 | sk2 wk8 v8 o2 | 29.1 / 31.4 | 1.253 |
| 3072 x 5120 | 4 | 27 | sk4 wk4 v1 o0 | 40.6 / 43.2 | sk2 wk8 v8 o2 | 34.6 / 37.5 | 1.171 |
| 5120 x 512 | 3 | 18 | sk8 wk4 v1 o0 | 11.3 / 16.5 | sk10 wk8 v8 o2 | 7.5 / 9.7 | 1.500 |
| 5120 x 512 | 4 | 16 | sk8 wk4 v1 o0 | 12.0 / 16.7 | sk10 wk8 v8 o2 | 9.0 / 11.4 | 1.339 |
| 5120 x 3072 | 2 | 1 | sk8 wk4 v1 o0 | 28.0 / 33.4 | sk4 wk8 v8 o2 | 21.9 / 25.5 | 1.282 |
| 5120 x 3072 | 3 | 47 | sk8 wk4 v1 o0 | 35.1 / 40.8 | sk4 wk8 v8 o2 | 29.2 / 31.6 | 1.203 |
| 5120 x 5120 | 2 | 1 | sk8 wk4 v1 o0 | 40.0 / 47.3 | sk2 wk8 v8 o2 | 34.7 / 37.7 | 1.152 |
| 5120 x 5120 | 3 | 47 | sk8 wk4 v1 o0 | 50.9 / 58.1 | sk2 wk8 v8 o2 | 45.9 / 48.2 | 1.110 |
| 5120 x 5120 | 4 | 1 | sk8 wk4 v1 o0 | 60.9 / 66.3 | sk2 wk8 v8 o2 | 58.5 / 62.2 | 1.041 |
| 5120 x 6144 | 3 | 9 | sk4 wk4 v1 o0 | 65.6 / 67.3 | sk2 wk8 v8 o2 | 53.7 / 56.3 | 1.221 |
| 5120 x 6144 | 4 | 8 | sk4 wk4 v1 o0 | 74.3 / 77.2 | sk4 wk8 v8 o2 | 66.7 / 73.4 | 1.113 |
| 5120 x 8704 | 2 | 28 | sk1 wk8 v1 o0 | 80.2 / 83.0 | sk5 wk4 v8 o4 | 54.5 / 67.8 | 1.472 |
| 5120 x 8704 | 3 | 98 | sk1 wk8 v1 o0 | 93.7 / 96.3 | sk5 wk4 v8 o0 | 76.1 / 83.0 | 1.232 |
| 5120 x 8704 | 4 | 4 | sk1 wk8 v1 o0 | 111.9 / 111.3 | sk5 wk4 v8 o0 | 95.4 / 109.4 | 1.173 |
| 5120 x 124160 | 5 | 1 | sk1 wk8 v1 o0 | 1701.5 / 1762.6 | sk1 wk8 v1 o0 | 1701.5 / 1762.6 | 1.000 |
| 8704 x 5120 | 2 | 2 | sk8 wk4 v1 o0 | 62.6 / 69.5 | sk2 wk8 v8 o2 | 54.8 / 59.0 | 1.142 |
| 8704 x 5120 | 3 | 48 | sk8 wk4 v1 o0 | 82.8 / 97.2 | sk2 wk8 v8 o2 | 76.7 / 80.7 | 1.079 |
| 8704 x 5120 | 4 | 15 | sk8 wk4 v1 o0 | 99.8 / 115.6 | sk4 wk8 v8 o2 | 93.4 / 103.8 | 1.069 |
count-weighted sum: default 27.41 / 29.78 ms, tuned 23.24 / 25.32 ms (m1 / m16)

Per rank and token the EXL3 linears take ~23 ms at rows 1 (the communication is extra).

## What mattered (and what did not)

| change | effect |
|---|---|
| L1-allocating trellis loads (`ld.global.nc`, not `.L1::no_allocate`): a lane's 2-3 words are reread by its neighbours and by its own next load | 3-bit -15..25% time (the single biggest win) |
| PDL: `griddepcontrol` + L2 prefetch of each warp's first 6 k steps before the wait; `rot_in` triggers at entry | `rot_in` disappears from the critical path (fwd 50.0 -> 46.6 ms on the layers) |
| gemv8: W^T as the mma A operand, 8 x rows as B (1 mma a tile, 32 accumulators, 96-128 registers) | 5-30% on most shapes vs x-as-A (2 mma a tile, 64 accumulators) |
| per-shape (sk, wk, kernel, launch bounds) search, median of 5 | 5-40% on narrow shapes (k/v, in_proj_z, out_proj) |
| `.maxntid` / `.minnctapersm` patched into the PTX (no `__launch_bounds__` in Rust) | 128-register variants, 2 programs/SM at 8 warps |
| cp.async shared-memory ring per warp (3 steps in flight) | slower (10-25%), removed |
| "strips" trellis layout (each column block's K contiguous) | neutral (kept, harmless) |
| depth-2 register prefetch | neutral at rows 1 (v9 = gemv8 + depth 2 kept: equal speed, used where v1 won so 17+ rows get 32-row passes) |
| a dead prefetch buffer kept live at depth 1 (bug, fixed) | had cost 10-15% at rows 16 through spills under launch bounds |

## Reproduce

```
cargo build --release -p fe-exl3
. tools/site.sh                               # FERRITE_EXL3_DIR, FERRITE_GPU_LOCK
M=$FERRITE_EXL3_DIR
flock $FERRITE_GPU_LOCK ./target/release/exl3-probe                                # streaming roof
flock $FERRITE_GPU_LOCK ./target/release/exl3-bench $M "" 300                      # rows 1 / 16, per shape
FE_BENCH_ROWS2=32 flock $FERRITE_GPU_LOCK ./target/release/exl3-bench $M "" 200    # rows 1 / 32
FE_EXL3_UNTUNED=1 flock $FERRITE_GPU_LOCK ./target/release/exl3-tune $M "" 150    # re-derive plans.rs
```
