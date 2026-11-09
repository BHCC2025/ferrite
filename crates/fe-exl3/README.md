# fe-exl3

EXL3 (exllamav3 1.6 trellis, `mul1` codebook) linears for ferrite: format, CPU reference, Rust GPU kernels
(`kernels/k-exl3`, built to PTX by `build.rs` and assembled with `ptxas -arch=sm_121a`).

## Format

`y = x @ W`, W is `[k = in_features][n = out_features]` (the transpose of the HF `weight`), stored as
`W = diag(suh) · H_k · W_hat · H_n · diag(svh)`: H is block-diagonal 128x128 Sylvester Hadamard / sqrt(128),
`suh` fp16 [k], `svh` fp16 [n], `W_hat` trellis-coded: `trellis` int16 `[k/16][n/16][16·bits]`, one 16x16 tile
of 256 values per cell, in mma B-fragment order. Bits come from the trellis shape (`shape[2] / 16`); this
checkpoint uses 2 / 3 / 4 bits (layers, MTP 4) and 5 (lm_head).

## Model-load API (reconstruct to bf16)

```rust
use fe_cuda::{DeviceBuffer, Stream};
use fe_exl3::{cpu::Layout, exl3_bases, gpu::Kernels, st::Checkpoint};

// in the engine's context (fe_cuda::Context::new(0))
let kern = Kernels::load()?;                              // k-exl3 cubin (built by fe-kbuild)
let ck = Checkpoint::open(fe_site::exl3_dir())?;          // FERRITE_EXL3_DIR (tools/site.sh)
let s = Stream::new()?;
let scratch = DeviceBuffer::<f32>::alloc(64 << 20)?;      // 256 MB f32 workspace, reused (min k * 128 floats)
for base in exl3_bases(&ck) {                             // "model.language_model.layers.0.mlp.up_proj", "lm_head", "mtp.fc", ...
    // bf16 bits [n][k] (HF weight layout: out rows of in) or Layout::KN for [k][n]
    let (w, k, n): (DeviceBuffer<u16>, usize, usize) = kern.load_bf16(&s, &ck, &base, Layout::NK, &scratch)?;
}
```

Raw form (caller-owned buffers, asynchronous on a stream):
`kern.reconstruct_bf16(&s, &view, out: DevPtr, Layout::NK, scratch: DevPtr, scratch_bytes)` with
`view: Exl3View { k, n, bits, trellis, suh, svh }` (device pointers; `DeviceExl3::upload(&Exl3Tensor)` makes one).
`kern.reconstruct_hat_f16(&s, &view, out)` gives the rotated-basis W_hat fp16 `[k][n]` (exllamav3 `reconstruct`).

Cost: ~27-40 GB/s of bf16 output (the 64-layer model, ~27 GB bf16, reconstructs in about a second).

## Decode API (GEMV straight from the trellis, 1..16+ rows)

```rust
use fe_exl3::linear::{default_blocks, Dtype, Linear, Workspace};

let blocks = default_blocks(ctx.sm_count()? as u32);          // program target for untuned shapes (4 / SM)
let q = Linear::load(&kern, &ck, "model.language_model.layers.3.self_attn.q_proj", blocks)?;
let o = Linear::load(&kern, &ck, "model.language_model.layers.3.self_attn.o_proj", blocks)?;
let ws = Workspace::new(&[&q, &o /* every Linear that shares the stream */], 16)?;  // max rows a call
// y[rows][n] = x[rows][k] @ W ; x bf16 (or Dtype::F32), y bf16 / f32, row-major contiguous
q.forward(&kern, &stream, x.ptr(), Dtype::Bf16, y.ptr(), Dtype::Bf16, rows, &ws)?;
// fused residual add: resid = bf16(f32(resid) + h @ W_o)
o.forward(&kern, &stream, h.ptr(), Dtype::Bf16, resid.ptr(), Dtype::Bf16Add, rows, &ws)?;
```

- `forward` = `rotate_input` (x -> fp16 H(x * suh) / sqrt(128) into `ws.xh`) + `gemv_rotated` (decode + mma +
  output Hadamard + svh + store/add). Both are programmatic dependent launches (PDL, `FE_EXL3_PDL=0` turns it
  off): the GEMV prefetches its first weight steps into L2 while the previous kernel drains, so `rot_in` is
  hidden. No host syncs, fixed workspaces: capture it in a CUDA graph.
- One `Workspace` per stream (calls on a stream run one at a time; split-K counters re-arm themselves).
- Plans: `crates/fe-exl3/src/plans.rs` has the GB10-tuned (sk, wk, kernel) per (k, n, bits); other shapes use
  TensorFold's rule. A row's result bits never depend on how many rows are in the call (1..64, exl3-rowcheck), so
  speculative verify rows match the draft-time rows exactly. `FE_EXL3_UNTUNED=1` forces the rule.
- `Linear::load` stores the trellis in the "strips" layout (decode only); `Linear::from_view` wraps buffers you own.
- Tensor parallel: `Linear::load_cols(&kern, &ck, base, &[(n0, n1), ...], blocks)` keeps the listed output
  columns (128-aligned ranges, concatenated: e.g. GDN in_proj_qkv rank r = its halves of the q, k and v
  sections); `Linear::load_rows(&kern, &ck, base, k0, k1, blocks)` keeps input rows k0..k1: feed it the rank's
  input columns, and the ranks' outputs (already in the original basis) sum to the full result. TP2 shapes
  are tuned in plans.rs too. `exl3-tpcheck <model> [layer]` verifies each split against the whole tensor.
- Kernels (`kernels/k-exl3/src/gemv.rs`): `exl3_gemv8_b{bits}_w{warps}_g{1,2,4,6}[_o{occ}]` (W^T as the mma A operand,
  8 x rows as B: one mma a tile, 32 accumulators a row group; g4 = 32-row passes, g6 = 48-row passes) and
  `exl3_gemv_b{bits}_w{warps}[_o{occ}]` (x as A, W as B, two mma a tile, TensorFold's form); `_o` = launch bounds
  (`.maxntid` / `.minnctapersm` patched into the PTX by build.rs, Rust has no `__launch_bounds__`).

## 48-row passes (g6)

One GEMV call costs a TP2 rank (sum over all linears, exl3-bench `FE_BENCH_TP2=1 FE_BENCH_ROWS2=m`, rot_in included):

| rows | g4 (32-row passes) | g6 (48-row passes) |
|---|---|---|
| 16 | 26.3 ms | - |
| 32 | 35.7 | - |
| 48 | 56.7 (32 + 16) | **43.8** |
| 64 | **62.1** (32 + 32) | 64.8 (48 + 16) |

`exl3_gemv8_b{bits}_w{4,8}_g6` decodes each k step's A fragments once for six 8-row groups. Groups 0-3 accumulate in
registers exactly as g4 (128 accumulators a lane, at the 255-register cap); groups 4-5 keep theirs in shared memory
after the partial-sum area (WK x 8 KB): each of their mmas loads its C fragment, runs the same `mma_f16_16816` and
stores it back, the same instruction on the same values as a register chain. The shape's tuned (sk, wk), each warp's
k range and k order, the warp order of the sums and the split order are those of g1 / g2 / g4, so a row's bits do
not depend on the form or on the call's row count (exl3-rowcheck 1-64 on TP1 and `FE_ROWCHECK_TP2=1` both ranks).
Dispatch (`Linear::gemv_rotated`): 33-48 rows take g6, 49-64 keep g4 (two 32-row passes win on nearly every shape);
`FE_EXL3_G6=0` turns g6 off. g6 needs 96 KB of dynamic smem at 8 warps and exactly 48 KB at 4, over the default
limit with its static smem, so `from_view` always raises the limit.

64 rows in one pass would need 8 resident groups: at 8 warps that is 64 rows x 128 columns x 8 partial sums = the
whole register file, or 4 shared-memory groups (128 KB) next to the 32 KB partial sums, over GB10's 99 KB a block.
Changing wk for big calls is not an option (a different wk changes the bits). The engine uses `X3_PASS` 64 and a RoCE
slot for 64 bf16 rows (tp.rs), so the scheduler's verify rows can go to 64, but the default is 48 (sched.rs): TP2 4
streams code 286 at 48 vs 264 at 32 and 286 at 64.

## Decode floor (1 row)

At 1 row the GEMVs of a TP2 rank sum to 23.7 ms (24.9 with the input rotations) for 5.08 GB of trellis, the lm_head's
5-bit vocabulary half included: 214 GB/s, ~90% of the 239 GB/s stream roof measured on GB10. So a 1-row decode step
(~30 ms a rank) is bound by the weight bytes; the rest is the RoCE reductions (~1.5-2 ms, 129 a forward) and ~2 ms
of GDN / norms / small kernels.

The input rotations are mostly hidden in a forward: `exl3_rot_in` triggers PDL as its first instruction, so the
following GEMV launches at once and streams its first weight steps into L2 while the rotation runs. One rotation
launch per projection for the linears that share an input (`exl3_rot_in3`: 144 launches a TP2 forward fewer,
bit-identical) measured no gain (TP1 decode 52.2 -> 52.2 ms/tok, 4 streams 58.4 -> 58.1 ms/step) and
was dropped. What would lower the floor is fewer bytes (ruled out: no smaller quant) or hiding the reductions behind
the GEMVs (column-half pipelining: an output-column range is pointer offsets plus a y row stride).

## Accuracy evidence

- `exl3-refcheck <model> <refdir>` (CPU, f64): W_hat bit-exact vs exllamav3 `reconstruct` for every
  kind x bit width (32 tensors: GDN in_proj_qkv / in_proj_z / out_proj, attention q/k/v/o, MLP gate/up/down at
  2/3/4 bits, lm_head 5 bits, MTP fc/attn/mlp 4 bits); W bf16 vs torch f64 -> bf16 bit-exact except at most one
  1-ulp tie per tensor and exact-zero cancellations.
- `exl3-gpucheck <model> <refdir>` (GPU): W_hat bit-exact; W bf16 (f32 math) within 1 ulp everywhere
  (~0.01% of values 1 ulp off, the rest identical; >1 ulp only for |error| < 1e-4 rms near zero); `[n][k]` is the exact
  transpose of `[k][n]`.
- `exl3-gemvcheck <model>` (GPU decode GEMV): every kind x bit width at rows 1, 2, 5, 8, 9, 16 vs an f64 CPU
  reference on the bit-exact W_hat and the GPU's own fp16 rotated input: max |y - ref| / rms <= 3e-5,
  rms error ~2e-7..1.3e-6 of rms (fp32 accumulation order); bf16 outputs within one bf16 rounding; rows past
  `rows` untouched; the `Bf16Add` residual mode checked the same way. `rot_in` alone vs f64 -> fp16: a few
  1-ulp tie flips per 80K values, never more.
- `exl3-detcheck <model> [filter] [iters] [rows,..]` (GPU determinism / race audit): every kind x bit width,
  200 forwards each at rows 1, 5, 16, 33 (tuned plan, split-K up to 10), each interleaved with a second linear
  sharing the workspace: all outputs bit-identical with PDL on and with `FE_EXL3_PDL=0`. Under compute-sanitizer
  (racecheck / synccheck / memcheck, split-K sk5 / sk10 shapes and a wide gate_proj): 0 hazards, 0 errors
  (2026-11-04). The split-K handoff is the release / acquire pattern (partials, `fence.acq_rel.gpu`, barrier,
  `atom.add.acq_rel.gpu`; the last program: barrier, fence, `ld.global.cg` partials) and every access to
  inputs, partials, outputs and counters comes after `griddepcontrol.wait`.
- `exl3-rowcheck <model> [filter] [rows,.. | a-b] [iters]` (GPU batch invariance): every row of an m-row call must be
  bit-identical to the same input row's 1-row call (the 1..8-, 9..16- and 17+-row kernel forms, tuned plans).
  Every kind x bits at m = 1..64, 3 times, PDL on: identical (2026-11-04). The references read their input rows in
  place: per-row uploads with `DeviceBuffer::copy_from` (a synchronous pageable cuMemcpyHtoD) are not ordered with
  the non-blocking stream, and under GPU load a reference then comes from the previous row (whole rows "wrong").
- `exl3-stress <model> [layers] [batches] [calls]` (GPU, engine-style split-K stress; run with FE_EXL3_SPLIT=1 now that
  one split is the default): the linears of the first `layers` layers + MTP share one workspace and a pseudo-random
  sequence of (linear, rows 1..16) forwards runs back to back, every output bit-compared with that pair's first
  result. One GB10, 16 layers (108 linears, 79 split-K), PDL on: 256000 calls, 0 differing (2026-11-04).
- References: `tools/exl3_dump_ref.py <outdir>` (a Python with exllamav3 and torch; reads `FERRITE_EXL3_DIR`).
- Speed: see BENCH.md.
