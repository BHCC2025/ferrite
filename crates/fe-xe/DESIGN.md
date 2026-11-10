# fe-xe design: Qwen3.8-27B on the Arc Pro B70 in Rust

fe-xe is the Intel Arc Pro B70 engine of Ferrite: Rust host code (this crate), Rust GPU kernels (kernels-xe, compiled to SPIR-V by
fe-kbuild-xe), Level Zero through fe-ze, weights in the XQ format (fe-xq). It was written against arc-engine, an unpublished Zig +
OpenCL C engine for the same card and model: every Rust kernel and the whole forward were checked bit for bit against arc-engine's
kernels and logits, and this file maps arc-engine's forward onto fe-xe. File references such as `src/xpu/*.zig` and `kernels/xpu/*.cl`
are to arc-engine and are kept as the names of the algorithms; none of that code is in this repository.

## 0. Status

The engine path is all Rust: 49 kernels built by `build.rs` (kx-core, kx-xq per code width, kx-xqpf) and embedded in the binary.

| piece | where |
|---|---|
| weights on the device (XQ packed + deduplicated su, shifted norms, conv / A_log / dt_bias, embedding) | `engine.rs` Weights::load |
| q8 KV, GDN state, window extras, 16-row scratch, prompt-window buffers | `engine.rs` Engine::new / Prompt |
| window forward m = 1..16 (verify windows, plain decode), commit / replay; rows are bit-identical to one-token decoding | `engine.rs` forward_rows / commit |
| prompt windows up to 4096 rows: int8 prompt GEMM `i8c2` by default (`FE_XE_PF=f16` = the fp16 path), int8 prompt attention, GDN over the window | `engine.rs` proj / pf_gemm / attention |
| bit-exact fusions of the small window kernels (`FE_XE_FUSEM`, `FE_XE_GDNPRE`, both on) | `engine.rs` |
| `logits` / `fpkl` / `gen` / `spec` modes | `bin/fe-xe.rs` |
| kernel registry: names, arguments, grids; per-kernel device time with `FE_XE_PROFILE=1` | `kx.rs` |
| MTP head, ranked int8 draft head, speculation (adaptive chain k <= 7, context copy, verify windows + commit) | `mtp.rs`, `specgen.rs`, `spec.rs` |
| DFlash2 block drafting with verify trees (the server default) | `crates/fe-xdf` |
| fe-server `--engine xe` (FIFO, GDN-state checkpoints, side pool, exact sampling) | `fe-server/src/xe.rs` |
| command-list replay / mutable arguments (not used by the forward yet) | `fe-ze/src/graph.rs` |

Checks that hold on this tree: speculative output == plain greedy (MTP and DFlash2); verify windows == one-token decoding, bitwise;
the fp16 prompt path == the reference logits, bitwise; KL divergence vs BF16 with the default int8 prefill: code 0.0061, prose 0.0114,
chat 0.0065, 16K context 0.0055, 120K context 0.0037.

## 1. Crates and reuse

| crate | role |
|---|---|
| `fe-xq` | XQ checkpoint loader (mmap safetensors + `xq_config.json`), device packing, CPU reference (`xq_prep` bit-exact, f64 GEMV, fp32 matvec model), `split_for` / `variant_for` |
| `fe-xe` | weights preparation, device state, forward (decode / window / prompt), MTP, speculation, logits / fpkl modes, oracle readers |
| `fe-xdf` | the DFlash2 drafter on the B70 |
| `fe-ze` | Level Zero: context, USM, immediate command lists, modules from SPIR-V, kernels, events, timestamps |
| `kernels-xe/*` | the kernels (section 8) |
| `fe-server` | HTTP / OpenAI front: tokenizer, chat template, reasoning + tool-call parsing, streaming, stop strings, sampling; cargo feature `xe` adds the `Engine` impl over fe-xe |

Reused without touching the CUDA crates: `fe-model/src/config.rs` (`#[path]` include: the Qwen3.5 config parser, `check_supported`),
`fe-exl3/src/half.rs` (fp16 / bf16 bit helpers, via fe-xq). fe-server's protocol layer needs no change: its `Engine` trait is
token-level (`submit(GenRequest, TokenSink)`), and `GenRequest.snapshots` already carries the prompt-cache checkpoint hints.
A machine without CUDA builds it with `cargo build -p fe-xe -p fe-xq` (the CUDA crates are not needed).

## 2. Model and numerics conventions (all must match arc-engine bit for bit)

- Qwen3.8-27B (`qwen3_5_text`): 64 layers, layer i is full attention when `i % 4 == 3` (16 layers), else Gated DeltaNet (48); hidden 5120,
  MLP 17408 (SwiGLU), vocab 248320, 1 MTP layer.
- Attention: 24 q heads / 4 kv heads, head_dim 256, `q_proj` is 12288 rows = per head [q 256 | gate 256] (`attn_output_gate`, output
  gated by sigmoid(gate)), q_norm / k_norm (RMSNorm over 256), partial RoPE on the first 64 dims (`rope_half` 32, theta 1e7), scale 1/16.
- GDN: 16 key heads x 128, 48 value heads x 128; qkv 10240 = q 2048 | k 2048 | v 6144 (grouped head order: value head h uses key head
  h / 3); conv1d k=4 over the 10240 channels + silu; q = bf16(rms(q) / 128), k = bf16(rms(k) / sqrt(128)) (no weight, eps 1e-6);
  beta = sigmoid(b), g = exp(-exp(A_log) * softplus(a + dt_bias)) (softplus = x if x > 20 else log1p(exp(x)));
  S = S g; kv = S k; delta = (v - kv) beta; S += k (x) delta; y = S q; out = bf16(silu(z) * rmsnorm(y) * w) per value head.
- Residual stream bf16; every block is `x = bf16(x + block(rmsnorm(x)))`; the add is folded into the next norm (`pending`).
- RMSNorm weights: the checkpoint stores the zero-centred form; arc-engine uploads `bf16(1 + w)` (f32 add, RNE) for input /
  post-attention / final / q / k / MTP norms (`qwen_load.normWs`); the GDN gated-norm weight (`linear_attn.norm`, 128) is used as stored.
- A_log and dt_bias are widened to fp32; conv1d weights stay bf16 `[10240][4]` (tap 0 = oldest).
- RoPE cos / sin are computed on the host in f32 (`inv = f32(theta^(-i/32))` in f64 then cast, `ang = f32(pos) * inv`, `cosf` / `sinf`)
  and uploaded per window (`Win.setRope`), or as a whole-context table for the MTP block (`qwen_mtp.ropeTable`). fe-xe must produce the
  same f32 bits: Zig's f32 `@cos` / `@sin` are its musl-derived routines, Rust's `f32::cos` (glibc) differs in 1.1 % of the 128K table,
  the `libm` crate matches all 8.4 M values (`fe-xe/src/rope.rs`, checked against a Zig dump of the whole table).
- bf16 rounding is RNE everywhere (`to_bf`: `u + 0x7fff + ((u >> 16) & 1)`), fp16 conversions RTE.
- **Contraction.** The reference SPIR-V (ocloc, OpenCL C 3.0) carries `ContractionOff` on almost every kernel and spells each
  contracted `a * b + c` as an explicit `OpExtInst mad` (or `fma` where the source wrote `fma()`): e.g. qwen_rows 19 mad + 1 fma,
  qwen_basic 11 + 2, xq 0 + 3, qwen_attn_long_q8 7 + 0. A Rust kernel reproduces those bits only if it fuses exactly there and
  nowhere else: kernels-xe builds with contraction off and writes every fused site as an explicit `fma` (Intel's fp32 `mad` is a single
  rounding fma). Source of truth per kernel: `spirv-dis kernels/xpu/<file>.spv`. Kernels without `ContractionOff` in the reference
  (qwen_kvq) may have extra driver fusion: check them against the ARC_KDUMP oracle first.
- Math builtins (`exp`, `log1p`, `rsqrt`, `sqrt`, `native_*`, `dot`, `sub_group_reduce_add`, `sub_group_broadcast`) must lower to the
  same SPIR-V instructions (OpenCL.std ext-insts, OpDot, OpGroupFAdd) so IGC picks the same implementations and reduction trees; a Rust
  software exp would not match.

## 3. Device memory (default profile, mix5.0, 128K context, KV q8; `fe-xe <model> info`)

| item | size |
|---|---|
| XQ projections packed + su (505 tensors; su shared by projections with one input in arc-engine) | 16.29 GB |
| embedding bf16 [248320][5120] (device resident) | 2.54 GB |
| KV cache q8, 16 layers x K + V x 131072 x 4 heads x 272 B | 4.56 GB |
| MTP KV cache (one layer, same mode) | 0.29 GB |
| GDN state (48 x (3 x 10240 bf16 + 48 x 128 x 128 fp32)) | 154 MB |
| prefix-cache checkpoint slots (8 x 154 MB) | 1.23 GB |
| side pool (16K positions KV + MTP KV) | ~0.6 GB |
| window scratch (4096-row prompt windows ~0.3-0.5 MB a row), prefill GEMM buffers (fp16 weights 17408 x 5120, k-major activations), head (16 rows x 248320 fp32 = 16 MB), attention partials | ~2-3 GB |
| total at load, MTP drafting | ~26.9 GB |
| total at load, DFlash2 drafting (the default; drafter 1.4 GB, MTP head not loaded) | ~28.6 GB |

## 4. Weights (fe-xe `weights.rs`, done)

Every projection is XQ (`fe-xq`: packed `[N/16][K/g][16 D + 8]` dwords, su fp16 [K]; mix5.0 has 4-, 5-, 6- and 8-bit tensors, mix3.9p adds
3-bit, the lm_head is 6-bit unrotated). `Planner` prepares each tensor on the host (`fe-xe <model> info --prepare` packs the whole
checkpoint: 18.84 GB device plan, ~5 min from a cold HDD). The upload is one `zeCommandListAppendMemoryCopy` per tensor (64 MB pieces
from a pinned staging buffer, as arc-engine). The shape-only K split `split_for(n, k, 128)` travels with each tensor: it fixes the
summation order of the decode matvec (1 for gate/up and lm_head, 2 qkv / q, 4 down / mtp.fc, 5 z, 6 out / o, 20 k / v, 32 b / a).

## 5. Forward paths

### 5.1 One token (`qwen_model.Model.forward`, `qwen_blocks.Ops`; used for plain decode steps, MTP steps)

Per token, `pending` folding the residual add into the next norm:

| block | launches (kernel, file, grid / local) |
|---|---|
| embed | `embed_bf16` (qwen_basic) |
| GDN norm + rotation | `xq_add_rms_prep` (xq.cl, 1 x 512): x = bf16(x + delta), xn = bf16(rms(x) * w), and the XQ prepass of xn for the su of in_proj_qkv |
| GDN projections | `xq_mv_128_1x1_1` qkv (S=2), z (S=5), b (S=32), a (S=32): one prepass reused (same input, same su) |
| GDN core | `gdn_conv_prep` (16 x 16: conv + silu + q/k norm, shifts the conv state of q/k channels), `gdn_step` (48 x 32 groups of 64: delta rule, v conv by lane 0, state in place), `gdn_gate_norm` (48 x 16) |
| GDN out | `xq_prep` + `xq_mv` out_proj (S=6) -> delta |
| attention norm + projections | `xq_add_rms_prep`; `xq_mv` q (S=2) -> qg [12288], k (S=20) -> kraw, v -> vtmp |
| attention core | `attn_prep` (28 groups of 256: q/k norm + RoPE; K row to ktmp), `kvq_quant8` x 2 (K and V rows into the q8 cache at pos), long attention over pos + 1 keys (q8 cache: `al.threshold()` is 0, so always `qwen_attn_long` / ARC_DEC=1: `attn_dec_partial` single row, then `attn_long_merge` with the sigmoid gate) |
| attention out | `xq_prep` + `xq_mv` o (S=6) -> delta |
| MLP | `xq_add_rms_prep`; `xq_mv` gate, up (S=1); `xq_prep_swiglu` (act = bf16(silu(g) u) + down's prepass in one launch); `xq_mv` down (S=4) |
| head | `xq_add_rms_prep` (final norm, lm_head unrotated: prepass is fp16(x)); `xq_mv` lm_head to fp32 [248320] (S=1); `round_bf16_f32` when bf16_logits; `argmax_partial` (64 parts) + `argmax_final` |

982 launches a token (GDN layer 10 + MLP 5, attention layer 11 + MLP 5, head 5 incl. the bf16 logits rounding, embed 1); time is the weight stream (16.3 GB at ~580 GB/s = 28 ms).

### 5.2 Windows of 1..16 rows (`qwen_win.Win.forward`; every speculative verify step)

The same arithmetic per row with row-batched kernels (`qwen_rows.cl`) and the XQ m-row variants; a row's bits equal the one-token path's
(row invariance), which is what makes speculation lossless:

- norms: `add_rmsnorm_r` / `rmsnorm_r` (m groups of 512), then the XQ prepass inside `Layer.forwardRows` (`xq_prep`, grid K/128 x Mpad);
  no fused `xq_add_rms_prep` / `xq_prep_swiglu` in windows (separate `swiglu`).
- XQ: `xq_mv_128_{MW}x{NR}_1` with (MW, NR) = (1,1) m=1, (2,1) m=2, (4,1) m=3-4, (8,1) m=5-8, (8,2) m=9-16; same split S as decode.
- GDN core: `gdn_conv_prep_r` (16 x m), `gdn_vconv_r` (m x 96), `gdn_gates_r` (m), `gdn_step_r2x4` (48 x 8 groups: the recurrence over
  the m rows inside the kernel, state in registers, into `sstate` (in place) or `sstate2` (pending)), `gdn_gate_norm_r` (48 x m),
  `gdn_conv_commit` (conv state after c kept rows).
- attention: `attn_prep_r` (28 x m), `kvq_quant8` x 2, decode attention for the window rows: ARC_DEC=1 packs the window's (row, head) pairs
  8 to a matrix tile (`attn_dec_pack` / `pack2` / `pack4` by group size) so rows share K/V reads; `attn_long_merge`.
- head: `headBatch` on the last `head_rows` rows (XQ m-row lm_head to fp32 [rows][248320], argmax per row).
- 1,271 launches a verify window (48 x 20 + 16 x 19 + 7), + 48 for a full commit or + 240 for a replay.
- decode attention kernel by window size: m = 1 `attn_dec_partial`; tiles = ceil(6m / 8): m = 2-4 `attn_dec_pack2` (2 tiles a work-group), m >= 5
  `attn_dec_pack4` (4 tiles); `attn_dec_pack` (1 tile) only with DEC_TG=1. All give the single-row bits.
- `commit(c)`: all rows kept -> swap `sstate2` in and shift the conv state; fewer -> replay the GDN core of the kept rows in place from
  the saved inputs (`wqkv`, `wb`, `wa` per layer). The KV cache just keeps positions (the next window overwrites the rejected ones).

### 5.3 Prompt windows (> 16 rows; 4096 by default, `XQ_PF_ROWS` / `--prefill`)

- Window cuts (serve_worker): at absolute multiples of 4096, at multiples of 8192 (checkpoint grid) and at n - 8 (`ckpt_tail`, when
  n - cached > 32), so the last 8 prompt rows usually run as a <= 16-row window on the decode kernels. Windows of 17-31 rows use the
  decode matvec (the GEMM needs R >= 32).

- projections: XQ prompt GEMM `XQ_PF=f16` (`xq.Prefill`): `xq_pfprep_k16` (xq_pfd.cl: the rows rotated once into a k-major fp16
  buffer [K][Rp]), `xq_pfdec16x_128` (xq_pfd.cl: the layer's weights decoded to pre-scaled fp16 [N][K] in a scratch, every call),
  `xq_pfgemm_f16_c4t4` (xq_pfg.cl: work-group-tiled DPAS GEMM, 4 x 4 sub-groups of 32 channels x 64 tokens, 2D block loads, 256 GRF).
  Bits differ from the decode kernel (within ~2.5e-4 of max|y|); rows are chunk-invariant. GDN b / a (48 rows) also go through it.
  The weight decode to fp16 runs again on every call (496 a 4096-row window): cacheable in fe-xe if memory allows.
- attention: `al.Pfs` with ARC_PFX=i8 (`qwen_attn_pfxi_q8.spv`): `attn_pfs_prep_i8` (Hadamard-rotated int8 Q), `kvq_expand_k8`, `kmean`,
  `kvq_expand_vt`, `attn_prefill_s` (8 rows x 6 heads a work-group, int8 Q.K^T on the matrix engine, fp16 P.V), causal over the cache
  (the window's own K/V rows are appended first). K and V are re-expanded from the q8 cache in 16384-key ranges for every attention
  layer of every window (softmax state carried between ranges).
- GDN: the window kernels of 5.2 with m = window rows (the recurrence runs row by row inside `gdn_step_r2x4`: not chunked; a chunked
  GDN as on the Sparks would change bits and need a KL gate).
- MTP cache entries of the prompt: `qwen_mtp_rows.zig` batched absorb (fc, MTP input norm, k / v projections, k-norm + RoPE, cache append
  for all window rows in one pass; entries may differ from single-row steps in the last bits: they only steer drafts).

### 5.4 MTP head and drafting (`qwen_mtp.zig`, `qwen_spec.zig`)

- A step at position p takes token u and the target's final-normed hidden row h of the previous row:
  `x = fc([rmsnorm(embed(u)) | rmsnorm(h)])` (rms512 x 2 into `cat`, XQ fc 5120 x 10240), then one full-attention block and an MLP with
  the single-row kernels of 5.1 against the MTP's own KV cache (its mode via `al.setOverride`), then `mtp.norm`; the normed output is the
  next chained step's h.
- Draft head (default profile, `MTP_DHEAD=dhead_q8.bin`): ranked int8 head from the BF16 lm_head (per-row scale max|w| / 127), file:
  "DHQ8", u32 version 1, u32 rows, u32 cols (5120), u32 n_ranked (20-byte header), then u32 ids[rows] (row -> token id, most frequent
  first), f32 scale[rows], int8 q[rows][5120]; the deployed file has 248320 rows, n_ranked 78834, 1.27 GB. The first 65536 rows live on the device; each request adds the rows
  of its prompt tokens outside them (up to 8192). Kernels `add_rmsnorm` (mtp.norm) + `dhead_q8b` + `argmax_partial` + `dhead_final`
  (spec_head.cl; maps back to token ids). ~0.35 GB of int8 rows a draft step: a real share of decode time with long chains.
- Policy (serving): hybrid drafter = MTP chain with adaptive k <= 7 (`Adapt`: expected tokens per cost from a verify-cost table,
  `SPEC_DSTEP=0.035`), replaced by a context copy of up to 15 tokens when the context suffix matches >= 8 tokens earlier (`GramIndex`),
  vetoed when the MTP's first draft disagrees; `observe` absorbs the kept rows into the MTP cache (batched) and chains the next drafts on
  the device. A window drafts kd tokens: verify `[last, d1..dkd]` (kd + 1 rows, non-inplace), keep the agreeing prefix a plus the bonus
  token, `commit(a + 1)`. kd = 0 -> a plain one-token step (5.1). Proposals are capped by room <= spec k = 7, so default windows have at
  most 8 rows even when a copy of 15 tokens fires. The first chain after a prompt is not adaptive (k = 7).
- Sampled requests: the window's logits come to the host, every row is sampled (temperature / top-k / top-p / min-p, seeded); a draft is
  kept while it equals the token sampled at its row (exact sampling). Greedy uses the device argmax.

### 5.5 Serving (`serve_worker.zig` -> fe-server `Engine` impl)

- One sequence at a time (FIFO), plus a 16K side pool (second KV + MTP cache) for short unrelated requests, promoted by device copy.
- Prefix cache: the longest common prefix with the kept sequence; resume from the latest GDN checkpoint inside it (KV is position
  addressed). Checkpoints (8 slots of 154 MB): prompt windows on an 8192 grid, `ckpt_tail` (8) tokens before the prompt end (the next agent
  turn re-renders the assistant header), the end of every generation. The checkpoint is a device-to-device copy of every layer's
  cstate + sstate. With the MTP head the resume point is one token earlier (an MTP entry is written with the following token).
  fe-server's `snapshots` (positions where a turn's reply starts) map onto extra checkpoint positions.
- Warmup before ready: one prompt window + an MTP chain + verify windows (JIT of every kernel variant).

## 6. State layouts (fe-xe `state.rs`)

- KV q8, per attention layer, K and V separate: `[pos][4 kv heads][272 B]` = 256 int8 (8 blocks of 32 dims, dim order) then 8 fp16 scales
  (bytes 256..271). `kvq_quant8`: d = fp16(amax / 127) (RTE through vstore_half), q = clamp(rint(v * (1 / d)), -127, 127) (multiply by the
  reciprocal, not a division). q4: 144 B (8 x 16 B + 8 fp16), bf16: 512 B.
- GDN conv state bf16 `[3][10240]`, oldest row first; delta-rule state fp32 `[48 value heads][128 v rows][128 k]` (`gdn_step`: a 16-lane
  sub-group owns one v row, 8 k values a lane).
- Window extras per GDN layer: `sstate2` (pending state), `wqkv [16][10240]`, `wb`, `wa [16][48]` (replay inputs).

## 7. Correctness gates and oracles

- The oracle set (written by arc-engine, not in this repository): XQ fixtures (numpy), XQ prep + matvec dumps on real tensors for every
  shape x width (+ row counts 1..16), full-model logits on fixed sequences (prompt windows, 16-row windows, decode rows, per-layer
  residual streams), and per-kernel launch dumps (arguments + every buffer before / after) from a real forward: each Rust kernel is
  unit-tested by replaying a dumped launch and comparing the written buffers bit for bit.
- Confirmed on the XQ dumps (fe-xq `tests/xqmv.rs`, 238 tensor x row-count cases): the prepass scratch (xt and xs) is reproduced bit
  for bit on the CPU, which pins two driver behaviours the Rust kernels must match: `sub_group_reduce_add` over 16 lanes is a halving
  tree (lane i += lane i + 8, + 4, + 2, + 1), and the scratch's padding rows (m..MW*NR) are computed from x = 0, so a rotation leaves
  signed zeros (0 * negative su = -0.0) in some slots. The kernel outputs sit within 8.5e-5 of max|y| of the float64 product.
- Row invariance (window row == single row), spec == plain greedy, cached == uncached: bitwise, against the uncached / unspeculated
  reference.
- KL: `fe-xe <model> fpkl` writes per-position row records that are scored against BF16 logits on five sets (code, prose, chat, 16K and
  120K context); bar < 0.05 on every set.

## 8. Kernel inventory

(Cross-checked against an independent read of the arc-engine sources; launch counts: plain token 982, verify window 1,271, MTP
drafting step ~25, absorb-only step ~21, prompt window: xq_pfdec16x_128 + xq_pfgemm_f16_c4t4 496 each, xq_pfprep_k16 256.)

Default profile (XQ mix5.0, KV q8, ARC_DEC=1, ARC_PFX=i8, XQ_PF=f16, 4096-row prompt windows). "Per token" counts are for a one-token
decode step (64 layers); windows launch the same kernels once per window with m rows. Share = estimated share of a 2K-context decode step (weights 16.3 GB a token vs KV / state traffic).

| # | kernel(s) | .cl / .spv | purpose | shapes / grid | per token | share | path | prio |
|---|---|---|---|---|---|---|---|---|
| 1 | `xq_mv_128_{1,2,4,8}x1_1`, `xq_mv_128_8x2_1` | xq.cl / xq{3,4,5,6,8}.spv | XQ decode matvec, 1..16 rows | all projections; grid N/16, local 16 x S | 497 | ~90 % (est.) | both | 1 |
| 2 | `xq_prep` | xq.cl | rotate + fp16 + group sums of 1..16 rows | grid K/128 x Mpad, local 16 | 64 (one-token; windows: every projection group) | small | both | 1 |
| 3 | `xq_add_rms_prep` | xq.cl | residual add + RMSNorm + prepass, one row | 1 x 512 | 129 | small | decode (1 row) | 2 |
| 4 | `xq_prep_swiglu` | xq.cl | SwiGLU + down's prepass, 1..16 rows | K/128 x Mpad | 64 | small | decode (1 row) | 2 |
| 5 | `xq_pfprep_k16`, `xq_pfdec16x_128`, `xq_pfgemm_f16_c4t4` | xq_pfd.cl, xq_pfg.cl | prompt GEMM (prep, weight decode, tiled DPAS GEMM) | R x N x K, R <= 4096 | - | prefill ~70 % | prefill | 2 |
| 6 | `embed_bf16` | qwen_basic.cl | embedding rows | m groups of 64 | 1 | - | both | 1 |
| 7 | `add_rmsnorm_r`, `rmsnorm_r` | qwen_rows.cl | residual add + RMSNorm, m rows | m x 512 | (windows) | small | both | 1 |
| 8 | `add_rmsnorm`, `rmsnorm512`, `add_bf16` | qwen_basic.cl | one-row norms (MTP input norms, fallbacks) | 1 x 512 | MTP | small | decode | 2 |
| 9 | `swiglu` | qwen_basic.cl | act = bf16(silu(g) u), m rows | m*17408/64 | (windows) | small | both | 1 |
| 10 | `gdn_conv_prep_r` | qwen_rows.cl | conv + silu + q/k norm of m rows | 16 x m, local 16 | 48 | small | both | 1 |
| 11 | `gdn_vconv_r` | qwen_rows.cl | v-channel conv + silu, m rows | m x 96 | 48 | small | both | 1 |
| 12 | `gdn_gates_r` | qwen_rows.cl | beta, g per head and row | m x 48 | 48 | small | both | 1 |
| 13 | `gdn_step_r2x4` (r2 / r2x2 variants, same bits) | qwen_rows.cl | delta rule over m rows, state in / out | 48 x 8, local 64 | 48 | ~2 % (151 MB state r+w) | both | 1 |
| 14 | `gdn_gate_norm_r` | qwen_rows.cl | gated RMSNorm per value head | 48 x m | 48 | small | both | 1 |
| 15 | `gdn_conv_commit` | qwen_rows.cl | conv state after c kept rows | 10240/64 | 48 | small | both | 1 |
| 16 | `gdn_conv_prep`, `gdn_step`, `gdn_gate_norm` | qwen_gdn.cl | one-row GDN (bit-identical to the row kernels at m = 1) | 16 / 48 x 32 / 48 | - | - | decode | 3 (only if faster than m=1 windows) |
| 17 | `attn_prep_r` (`attn_prep` one row) | qwen_rows.cl / qwen_attn.cl | q/k norm, RoPE, gate split, K row out | 28 x m, local 256 | 16 | small | both | 1 |
| 18 | `kvq_quant8` | qwen_kvq.cl | append K / V rows to the q8 cache | 32 lanes a (row, head, block) | 32 | small | both | 1 |
| 19 | `attn_dec_partial`, `attn_dec_pack{,2,4}` | qwen_attn_dec.cl / qwen_attn_dec_q8.spv | decode attention partials on the matrix engine (single row / window rows packed 8 a tile), q8 cache, 1024-key chunks | 4 kv heads x chunks | 16 | 2 % at 2K, ~25 % at 100K | both | 1 |
| 20 | `attn_long_merge` | qwen_attn_long.cl (q8 build) | merge chunk partials + output gate | 24 heads x rows | 16 | small | both | 1 |
| 21 | `attn_pfs_prep_i8`, `kvq_expand_k8`, `kmean`, `kvq_expand_vt`, `attn_prefill_s` | qwen_attn_pfx.cl / qwen_attn_pfxi_q8.spv | prompt-window attention (Hadamard-int8 Q.K^T, fp16 P.V) | rows/8 x heads/6 | - | prefill ~15-30 % (grows with context) | prefill | 2 |
| 22 | `round_bf16_f32` | qwen_basic.cl | bf16-rounded logits (gate / harness parity) | V/64 | 0-1 | - | head | 2 |
| 23 | `argmax_partial`, `argmax_final` | qwen_basic.cl | greedy token per row (ties to the lowest id) | 64 parts x rows | 2 | small | head | 1 |
| 24 | `dhead_q8b`, `dhead_final` | spec_head.cl | ranked int8 draft head + id map | (n+31)/32 | per draft | ~0.35 GB a draft step: 10-15 % of decode wall time (est.) | spec | 2 |
| 25 | MTP batched absorb (fc, norms, k/v, k-norm + RoPE, append: XQ + rows kernels) | qwen_mtp_rows.zig | MTP cache entries of prompt / kept rows | rows | per window | small | spec | 2 |
| 26 | GDN checkpoint / side-pool copies | (memory copies) | prefix cache | 154 MB / slot | - | - | serve | 3 |

Not needed on the default path: the EXL3 / MLX / GGUF kernels, `attn_partial` / `attn_merge` / `attn_partial_r` / `attn_merge_r` (bf16
KV below 4096 keys), `attn_long_partial` (ARC_DEC=0), `xq_pfgemm_i8*` / `xq_pfdec_*` (XQ_PF=i8 or XQ_PFD=0), `qwen_attn_pfs*` (ARC_PFX=0),
`mv_f16_rows`, `mv16x`.
