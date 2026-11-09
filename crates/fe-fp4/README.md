# fe-fp4 — NVFP4 / FP8 prompt (prefill) linears

Prompt rows of the hybrid run through the ModelOpt checkpoint `$FERRITE_NVFP4_DIR` (nvidia/Qwen3.8-27B-NVFP4)
(MLP gate/up/down NVFP4: e2m1 codes + e4m3 scale per 16 + fp32 `weight_scale_2` + `input_scale`; attention and
GDN projections FP8 e4m3 per-tensor `weight_scale` + `input_scale`). Decode rows stay on EXL3.

Everything on the GPU is Rust (`kernels/k-fp4`): the bf16 -> NVFP4 / FP8 activation quantizers and the GEMMs
(block-scaled `mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` with ue4m3 scales, and
`mma.sync.kind::f8f6f4.m16n8k32` e4m3). cuBLASLt is linked only as the measured baseline / optional fallback
(`forward_lt`, see NON-RUST.md).

## API

```rust
use fe_fp4::{Checkpoint, Kernels, PromptLinear, Workspace};

let ck = Checkpoint::open(&fe_site::nvfp4_dir())?;  // $FERRITE_NVFP4_DIR
let kern = Kernels::load()?;                       // after the CUDA context exists; one per process
let ws = Workspace::new(max_rows, 17408)?;         // quantized-activation scratch shared by all prompt linears
                                                   // (max_rows x max_k bytes + 1/16 of that); one per stream

// one linear, or several tensors with the same input fused along N (each column keeps its own scale):
let gate_up = PromptLinear::load(&ck, &["model.language_model.layers.0.mlp.gate_proj",
                                        "model.language_model.layers.0.mlp.up_proj"])?;   // n = 34816
let down = PromptLinear::load(&ck, &["model.language_model.layers.0.mlp.down_proj"])?;

// y[m][ldy] = x[m][lda] . W^T   (x, y bf16; same contract as fe_model::Linear::forward)
gate_up.forward(&kern, &stream, &ws, y, gate_up.n, x, 5120, m, None)?;
// with a residual: y = bf16(bf16(x . W^T) + res), res may equal y (in place)
down.forward(&kern, &stream, &ws, h, 5120, act, 17408, m, Some(h))?;
```

- `PromptLinear { kind: Nvfp4 | Fp8, n, k, parts: Vec<Part { name, n0, n, alpha, input_scale }>, .. }`. Fused
  parts are laid out side by side: part `i` occupies output columns `parts[i].n0 .. n0 + n` of each row.
- The kind (NVFP4 / FP8) comes from `hf_quant_config.json` (or the tensor dtypes). Fused parts must share
  the kind and K. Requirements: N % 128 == 0, K % 128 == 0 (all 27B shapes satisfy this), x rows 16-byte
  aligned, any M (edges zero-filled). `m == 0` is a no-op.
- Activation scale: static per-tensor `input_scale` from the checkpoint (as vLLM / TRT-LLM): NVFP4 block
  scale = e4m3(amax16 / 6 / input_scale), codes = e2m1(x / (scale * input_scale)); FP8 codes = e4m3(x /
  input_scale). For fused parts the largest `input_scale` is used (no clipping); `alpha[col]` =
  input_scale x `weight_scale_2` (NVFP4) or x `weight_scale` (FP8).
- `forward` = one quantize launch + one GEMM launch on `stream`, graph-capturable (no allocation, no sync).
  The workspace is reused by the next call, so calls sharing a workspace must be stream-ordered.
- `quantize(kern, s, ws, x, lda, m)` + `gemm(kern, s, ws, y, ldy, m, res)`: quantize once, run several GEMMs of
  linears with the same `kind`, `k` and `g`. Lower level: `Kernels::{quant_fp4, quant_fp8, gemm_fp4, gemm_fp8}`.
- Fused MLP front: `PromptLinear::load_swiglu(ck, gate, up)` (gate|up rows interleaved by 8) and
  `forward_swiglu(kern, s, ws, act_ws, x, lda, m, down.g)` = gate|up GEMM + SwiGLU (bf16(bf16(silu(g)) * u)) +
  NVFP4 quantization of down's input into `act_ws`; then `down.gemm(kern, s, act_ws, y, ldy, m, Some(res))`.
  Bit-identical to the unfused path (1.34-1.49x faster than GEMM + a one-pass swiglu_quant).
- Memory: weights stay quantized on the device (whole 27B: ~8.6 GB NVFP4 MLP + ~7.2 GB FP8 attention/GDN).
  Loading reads with pread and drops the page cache behind each tensor.
- Env: `FE_FP4_GROUP` = raster group height in 128-row tiles (default 8; 0 = all rows fastest);
  `FE_FP4_KERNEL` (NVFP4) / `FE_FP8_KERNEL` (FP8) = n128 | n256 | mb | auto | s; defaults n256 / s (s3 = 128x128,
  3 stages, two CTAs a SM). `mb` (mbarrier pipeline) is faster but flagged by racecheck, so opt-in.
- Engine flags (crates/fe-model): `FE_PROMPT=fp4` (the checkpoint folder is `FERRITE_NVFP4_DIR`), `FE_FP4_SWIGLU=0` (unfused MLP),
  `FE_GDN_CHUNK=0` (k-core rec2 instead of crates/fe-gdn's chunked GDN), `FE_GDN_TC=0|1|3`.

## Tests and benches

```
cargo build --release -p fe-fp4
cargo test --release -p fe-fp4 --lib                      # format round trips
. tools/site.sh                                            # FERRITE_* site settings (README table)
flock $FERRITE_GPU_LOCK ./target/release/fp4-check $FERRITE_NVFP4_DIR 3 200
flock $FERRITE_GPU_LOCK ./target/release/fp4-bench $FERRITE_NVFP4_DIR 20
flock $FERRITE_GPU_LOCK ./target/release/fp4-determinism $FERRITE_NVFP4_DIR 30 2048
flock $FERRITE_GPU_LOCK ./target/release/fp4-peak
```

`fp4-check` (layer 3: gate|up, down, q|k|v, o): GPU quantizers bit-exact vs the CPU quantizer, GEMM vs an f64
CPU GEMM over the same quantized operands (max err / row max ~3e-3 = bf16 output rounding), residual epilogue
exact, the Rust GEMM output is bit-identical to cuBLASLt on the same inputs, a canary run shows nothing is
written outside [m][n] (strided output, residual), and the fused SwiGLU kernel matches the unfused GPU path
bit for bit. Results in BENCH.md.
