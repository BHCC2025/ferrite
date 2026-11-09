# fe-fp4 prompt GEMMs: Rust (k-fp4) vs cuBLASLt

GB10 (DGX Spark, sm_121a, 48 SMs), CUDA 13.0, 2026-11-04. Real Qwen3.8-27B ModelOpt weights (layer 3 MLP /
attention, layer 0 GDN), fused shapes as the engine runs them. GEMM-only time (activations already quantized,
same quantized inputs for both paths), mean of 30 back-to-back launches after warm-up, bf16 output.
`quant us` = the bf16 -> NVFP4/FP8 activation quantizer for that M x K (memory bound, shared by both paths).

Tensor-core peak measured with `fp4-peak` (registers only): **NVFP4 m16n8k64 block-scaled 426 TFLOPS, FP8
m16n8k32 213 TFLOPS** (mma.sync; consumer Blackwell has no wgmma/tcgen05).

```
. tools/site.sh
flock $FERRITE_GPU_LOCK ./target/release/fp4-bench $FERRITE_NVFP4_DIR 30 512,2048,4096,8192
flock $FERRITE_GPU_LOCK ./target/release/fp4-peak
```

Kernels = current defaults (03:10): NVFP4 on the 128x256-tile 3-stage bar.sync pipeline (`fe_gemm_nvfp4_n256`),
FP8 on 128x128 tiles / 3 stages / two CTAs a SM (`fe_gemm_fp8_s3`); both compute-sanitizer racecheck clean.
Output bit-identical to cuBLASLt on the same inputs.

| linear | kind | N x K | M | Rust TFLOPS | cuBLASLt TFLOPS | Rust/Lt | quant us |
|---|---|---|---|---|---|---|---|
| mlp gate\|up | NVFP4 | 34816 x 5120 | 512 | 252 | 263 | 0.96 | 6 |
| mlp gate\|up | NVFP4 | 34816 x 5120 | 2048 | 238 | 283 | 0.84 | 74 |
| mlp gate\|up | NVFP4 | 34816 x 5120 | 4096 | 277 | 284 | 0.97 | 238 |
| mlp gate\|up | NVFP4 | 34816 x 5120 | 8192 | 280 | 315 | 0.89 | 473 |
| mlp down | NVFP4 | 5120 x 17408 | 512 | 229 | 281 | 0.82 | 31 |
| mlp down | NVFP4 | 5120 x 17408 | 2048 | 266 | 299 | 0.89 | 414 |
| mlp down | NVFP4 | 5120 x 17408 | 4096 | 273 | 137 | 1.99 | 802 |
| mlp down | NVFP4 | 5120 x 17408 | 8192 | 281 | 111 | 2.52 | 1633 |
| gdn qkv\|z | FP8 | 16384 x 5120 | 512 | 149 | 134 | 1.12 | 6 |
| gdn qkv\|z | FP8 | 16384 x 5120 | 2048 | 167 | 183 | 0.92 | 127 |
| gdn qkv\|z | FP8 | 16384 x 5120 | 4096 | 170 | 191 | 0.89 | 280 |
| gdn qkv\|z | FP8 | 16384 x 5120 | 8192 | 170 | 192 | 0.88 | 553 |
| gdn out | FP8 | 5120 x 6144 | 512 | 129 | 152 | 0.85 | 12 |
| gdn out | FP8 | 5120 x 6144 | 2048 | 157 | 165 | 0.95 | 179 |
| gdn out | FP8 | 5120 x 6144 | 4096 | 162 | 180 | 0.90 | 344 |
| gdn out | FP8 | 5120 x 6144 | 8192 | 168 | 189 | 0.89 | 676 |
| attn q\|k\|v | FP8 | 14336 x 5120 | 512 | 141 | 134 | 1.05 | 6 |
| attn q\|k\|v | FP8 | 14336 x 5120 | 2048 | 163 | 152 | 1.07 | 134 |
| attn q\|k\|v | FP8 | 14336 x 5120 | 4096 | 172 | 187 | 0.92 | 291 |
| attn q\|k\|v | FP8 | 14336 x 5120 | 8192 | 169 | 190 | 0.89 | 556 |
| attn o | FP8 | 5120 x 6144 | 512 | 129 | 136 | 0.95 | 8 |
| attn o | FP8 | 5120 x 6144 | 2048 | 162 | 166 | 0.98 | 168 |
| attn o | FP8 | 5120 x 6144 | 4096 | 162 | 178 | 0.91 | 333 |
| attn o | FP8 | 5120 x 6144 | 8192 | 167 | 187 | 0.89 | 665 |

The opt-in mbarrier pipeline (`FE_FP4_KERNEL=mb`, racecheck-flagged, bitwise identical in practice) reached
277-290 NVFP4 (1.00-1.02x cuBLASLt at M 2048-4096) and 144-165 FP8.

Target was Rust >= 60% of cuBLASLt: met everywhere (NVFP4 0.82-0.97x, 2-2.5x on mlp down at M >= 4096 where
cuBLASLt's heuristic picks a slow algorithm; FP8 0.85-1.12x).

## Fused MLP front (`PromptLinear::forward_swiglu`)

gate|up GEMM + SwiGLU (bf16(bf16(silu(g)) * u), core's fe_silu_mul semantics) + NVFP4 quantization of
down_proj's input in one kernel (weights row-interleaved by 8 so each lane holds gate and up of the same
features); output = down's quantized activations, so down runs `gemm` with no separate quantize.
Bit-identical to GEMM + `fe_fp4_swiglu_quant` (fp4-check).

| M | fused ms | GEMM + swiglu_quant ms | fused TFLOPS | speedup |
|---|---|---|---|---|
| 512 | 0.777 | 1.044 | 235 | 1.34x |
| 2048 | 2.901 | 4.028 | 252 | 1.39x |
| 4096 | 5.715 | 8.500 | 256 | 1.49x |
| 8192 | 11.272 | 16.160 | 259 | 1.43x |

(bar.sync pipeline, the default; on the mbarrier pipeline it was 249-288 TFLOPS.)

(vs the engine's current gate|up GEMM + fe_silu_mul + down quantize, which makes two passes over the act.)

## Engine (one node, EXL3 3.0 mix decode + FE_PROMPT=fp4, TP1, `ferrite bench`)

| prefill | 2K | 8K | 32K |
|---|---|---|---|
| bf16 prompt GEMM (before) | 1065 | - | - |
| fp4, first kernels (23:25) | 2077 | 1933 | 1600 |
| fp4, staged epilogue + mb (00:05) | 2317 | 2141 | 1735 |
| + fused gate\|up + SwiGLU + NVFP4 quant (00:15) | 2498 | 2327 | 1855 |
| + chunked 3xTF32 GDN prefill (crates/fe-gdn, 01:20) | 3024 | 2773 | 2122 |
| race-safe defaults: bar.sync n256 NVFP4 + fused SwiGLU (02:10) | 2868 | 2644 | 2037 |
| + FP8 on 128x128 / 3 stages / 2 CTAs a SM (02:35) | 2974 | 2723 | 2096 |
| + k-gdn batched loads (03:00, current) | 3084 | 2815 | 2152 |

KL (decode rows on EXL3, prompt rows through fp4: fp4 only for calls with > 64 rows): code 0.02045 / text
0.04093. Routing decode batches > 16 rows to fp4 instead gives text 0.0757 (fails
the 0.05 bar): keep decode on EXL3.

## What limits the Rust GEMM (ncu, gate|up M = 4096)

- Inner loop, not memory: the same kernel with no main-loop loads (`fe_gemm_nvfp4_n256_noload`) ran at
  244 TFLOPS when the loaded one ran ~240; L2 hit 91%, L2 throughput ~52%.
- Stall mix per issued instruction: fixed-latency wait 3.9, math-pipe throttle 3.0, long scoreboard 1.6,
  MIO throttle 1.3, barrier 0.6 (of 12.3 cycles). OMMA + the NOPs ptxas pairs with it are ~50% of samples;
  the old per-element epilogue was ~12% (fixed by the smem-staged epilogue: +8-15%).
- 2 warps per SMSP (256 threads, 224-254 registers): the 128 x 256 tile needs 128 accumulator registers;
  a 9th producer warp would cap registers at 168 (16K registers per SMSP / 3 warps) and spill.
- mbarrier stage hand-off instead of a CTA barrier: +5-10% at M >= 2048, slightly worse at 512.

## Determinism / sanitizers

`fp4-determinism [dir] [reps] [m]`: every variant (n128 / n256 / mb, NVFP4 + FP8) and the fused SwiGLU kernel,
30 runs each at M = 2048 on real weights: bitwise identical. compute-sanitizer racecheck flagged the mbarrier
pipeline (mb, swiglu) with "invalid memcpy_async synchronization" while it used
`cp.async.mbarrier.arrive.noinc`; the pipeline now uses the cuda::barrier form (`cp.async.mbarrier.arrive` +
`mbarrier.arrive`, one arrival per thread). Defaults since 02:10 avoid the mbarrier pipeline entirely: NVFP4 GEMMs and the fused SwiGLU run the
`cp.async.wait_group` + `bar.sync` n256 pipeline, FP8 GEMMs the 128 x 128 / 3-stage variant (`fe_gemm_fp8_s3`,
48 KB so two CTAs share a SM and hide each other's barrier bubbles: 157-168 TFLOPS vs 141-156 for n256).
racecheck: 0 hazards on fe_gemm_nvfp4_n256, fe_gemm_nvfp4_swiglu, fe_gemm_fp8_s3 (and k-gdn). Env:
FE_FP4_KERNEL / FE_FP8_KERNEL = n128 | n256 | mb | auto | s (s = s2 for NVFP4, s3 for FP8).

Next: TMA (cp.async.bulk.tensor) loads to cut the LDGSTS issue + address math, BK = 128 bytes per stage on a
128 x 128 tile to halve per-k-tile overhead, fusing the remaining activation quantizers into their producers
(rmsnorm, attention / GDN output), split-row routing so mixed prefill + decode forwards keep decode rows on EXL3.
