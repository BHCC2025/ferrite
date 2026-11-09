# fe-gdn — chunked gated delta rule for GDN prompt rows

Drop-in for k-core's `fe_gdn_rec2` on forwards without a state history (prefill chunks): same inputs
(post-conv q|k|v rows, a / b projections, A_log, dt_bias), same raw output `o` (bf16, before the gated
RMSNorm), same fp32 state `[slot][48][128][128]` read and written back. The engine uses it for GDN forwards
with more than 64 rows and no history (`FE_GDN_CHUNK=0` keeps rec2 everywhere).

## Math

Row recurrence (rec2): `S *= g; d = beta (v - S^T k); S += k d^T; o = S^T q`, q / k l2-normalised.
Over a chunk of C = 32 rows with `L_r = sum_{s<=r} ln g_s`, `gamma_r = exp(L_r)`, S0 the incoming state:

```
(I + A) U = R,   A[r][s] = beta_r exp(L_r - L_s) k_r.k_s (s < r),   R_r = beta_r (v_r - gamma_r S0^T k_r)
o_r = gamma_r S0^T q_r + sum_{s<=r} exp(L_r - L_s) (q_r.k_s) u_s
S   = gamma_{C-1} S0 + sum_s exp(L_{C-1} - L_s) k_s u_s^T
```

## Kernels (kernels/k-gdn)

- `fe_gdn_chunk_tc` (default, `FE_GDN_TC=3`): one CTA per head (and segment), 8 warps; warp w owns value
  columns 16w..16w+16 with its 16 x 128 slice of S^T as m16n8 TF32-mma accumulators (64 registers). Per chunk
  `S0^T K^T`, `S0^T Q^T` (the accumulators are reused as A fragments: an accumulator tile is an A fragment
  once its 8 k slots are read as 0,2,4,6 / 1,3,5,7, and the B rows are loaded in that order), `U^T P^T` and
  `(w . U)^T K` run on `mma.sync.m16n8k8.tf32`; the Gram matrices, `T = (I + A)^-1` (warp 0, overlapping the
  other warps' mma) and `U = T R` are fp32 SIMT. Every mma product uses 3xTF32 (each operand split into TF32
  hi + lo, hi.hi + hi.lo + lo.hi), so results match the fp32 algorithm (single TF32 on `U^T P^T` alone cost
  +0.002 text KL). Accumulation fp32. 255 registers, 86 KB smem.
- `fe_gdn_chunk_tc1` (`FE_GDN_TC=1`): single TF32 everywhere (~8% faster, state rel error 4e-4, text KL +0.002).
- `fe_gdn_chunk` (`FE_GDN_TC=0`): all-fp32 SIMT version (512 threads), the reference for the TC kernel.

## Results (GB10)

```
. tools/site.sh
flock $FERRITE_GPU_LOCK ./target/release/gdn-check $FERRITE_NVFP4_DIR 2048,500,3
```

| kernel | o rel rms vs f64 | state rel rms | 2048 rows x 48 heads |
|---|---|---|---|
| fe_gdn_rec2 (k-core) | - | - | ~5 ms (engine nsys) |
| fe_gdn_chunk (fp32 SIMT) | 1.66e-3 | 4.1e-7 | 3.96 ms |
| fe_gdn_chunk_tc1 (TF32 mma) | 1.72e-3 | 3.6e-4 | 1.74-1.92 ms |
| fe_gdn_chunk_tc (3xTF32, default; batched phase-1 loads) | 1.66e-3 | 2.0e-6 | 1.53 ms |

(o error is the bf16 output rounding in both.) Engine prefill on one node (EXL3 3.0 mix decode, FE_PROMPT=fp4,
fused SwiGLU), `ferrite bench`:

| prefill tok/s | 2K | 8K | 32K |
|---|---|---|---|
| rec2 | 2448 | 2319 | 1852 |
| fe_gdn_chunk_tc1 (TF32) | 3142 | 2821 | 2150 |
| fe_gdn_chunk_tc (3xTF32, default) | 3035 | 2776 | 2120 |
| + race-safe GEMM defaults, FP8 s3, batched GDN loads (03:00) | 3084 | 2815 | 2152 |

KL vs BF16 (FE_BATCH=16 so decode rows stay on EXL3; code 40 seqs / text 20 seqs, 1024 prompt + 512 decode):
| GDN prompt kernel | code | text |
|---|---|---|
| fe_gdn_rec2 | (0.02045 at batch 20) | 0.04123 |
| fe_gdn_chunk (fp32 SIMT) | | 0.04106 |
| fe_gdn_chunk_tc1 (single TF32) | 0.0209 | 0.04319 |
| fe_gdn_chunk_tc, 3xTF32 state products only | 0.02073 | 0.04296 |
| fe_gdn_chunk_tc, 3xTF32 everywhere | | 0.04172 |
| fe_gdn_chunk_tc, current default (truncation split, 03:00) | | 0.03997 |
