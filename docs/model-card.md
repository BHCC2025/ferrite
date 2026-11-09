---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
tags:
- exl3
- quantized
- dgx-spark
---

# Qwen3.8-27B · EXL3 3.0 bpw mix

A mixed-precision [EXL3](https://github.com/turboderp-org/exllamav3) quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It averages **3.0 bits per weight (13.3 GB)**, but each
tensor gets its own width (2, 3 or 4 bits) based on how sensitive it is. The result is **more accurate than uniform
3.0 bpw on every test set while reading fewer bytes per token**.

It was built as the decode model of [Ferrite](https://github.com/BHCC2025/ferrite), a Rust inference engine for
Qwen3.8-27B on NVIDIA DGX Spark, and it also runs unchanged in exllamav3 and TabbyAPI.

## At a glance

| | |
|---|---|
| Base model | Qwen/Qwen3.8-27B: 64 layers (48 Gated DeltaNet + 16 gated attention), hidden 5120, vocab 248,320, one MTP layer |
| Format | EXL3 (exllamav3 1.6.1 or later) |
| Bits | 3.0 bpw average over the decoder linears (2 / 3 / 4-bit per tensor); output head 5-bit; MTP layer 4-bit |
| Size | 13.3 GB on disk (about 54 GB in BF16) |
| Memory read per token | 9.98 GB (uniform 3.0 bpw: 10.14 GB) |
| Accuracy | KL vs BF16 is 0.86-0.93x that of uniform 3.0 bpw on all five test sets |
| Context | measured to 120K tokens (Ferrite serves 128K) |
| Runs on | Ferrite (DGX Spark, one or two nodes), exllamav3, TabbyAPI |
| License | Apache-2.0, inherited from the base model |

## Which engine to use

| Engine | Hardware | Use it for |
|---|---|---|
| **[Ferrite](https://github.com/BHCC2025/ferrite)** | NVIDIA DGX Spark (GB10), one or two units | built for DGX Spark: Rust kernels, speculative decoding, a prompt cache for agents, an OpenAI-compatible server |
| **exllamav3** | NVIDIA GPUs supported by exllamav3 (made and measured on a 16 GB RTX 5080) | Python use, scripts, MTP drafting |
| **TabbyAPI** | same as exllamav3 | an OpenAI-compatible server on a regular GPU |

## What is in it

Bit widths were chosen by exllamav3's optimizer under a 3.0 bpw budget. Tensors that are fused at decode time keep
one shared width (q+k+v, gate+up, GDN qkv+z), so fused kernels still apply.

| Decoder tensor | 2-bit | 3-bit | 4-bit |
|---|---|---|---|
| attention q / k / v / o_proj (16 layers) | | 9 | 7 |
| GDN in_proj_qkv, in_proj_z (48 layers) | 1 | 47 | |
| GDN out_proj (48 layers) | | 29 | 19 |
| MLP gate_proj, up_proj (64 layers) | 14 | 49 | 1 |
| MLP down_proj (64 layers) | 2 | 48 | 14 |

Kept at higher precision: output head 5-bit, MTP layer 4-bit, embeddings BF16, GDN in_proj_a / in_proj_b FP16,
vision tower 6-bit (exllamav3's default; Ferrite is text-only and does not load it).

## Accuracy

KL divergence from the BF16 model (lower is better). The measurement is the mean per position over BF16's top 128
tokens plus one remainder bucket, run on exllamav3 with an FP16 KV cache on an RTX 5080.

| Test set | Uniform 3.0 bpw | **This quant** | Ratio |
|---|---|---|---|
| code (40 x 2048 tokens) | 0.0250 | **0.0230** | 0.92x |
| text (20 x 2048 tokens) | 0.0500 | **0.0467** | 0.93x |
| chat (held-out trace, 45,930 response tokens) | 0.0330 | **0.0282** | 0.86x |
| long16k (16K-token prompts) | 0.0239 | **0.0215** | 0.90x |
| long120k (120K-token prompts) | 0.0152 | **0.0138** | 0.91x |
| top-1 agreement with BF16, code / text | 95.53 / 90.70% | **95.72 / 91.04%** | |

"Uniform 3.0 bpw" is the plain 3.00 bpw EXL3 quant (6-bit head) from the same upstream measurement repository.

**In Ferrite** (Ferrite's own harness: 1024-token prompts, then 512 teacher-forced decode rows, KL against the same
BF16 model):

| Setup | code | text | long16k |
|---|---|---|---|
| one DGX Spark, bf16 prompt weights | 0.0218 | 0.0466 | 0.025 |
| two DGX Sparks, NVFP4 prompt weights | 0.0208 | 0.0401 | 0.024 |

The two harnesses differ, so compare numbers within a table, not across tables. Ferrite's speculative decoding is
lossless (greedy output is token-for-token identical to plain decoding), and its prompt cache is exact (a resumed
request produces the same tokens as computing the whole prompt again).

## Performance

**Fewer bytes than uniform 3.0 bpw.** Decoding is memory-bound, and this quant reads 1.6% less per token while being
more accurate.

**exllamav3, RTX 5080 16 GB** (one stream, 500 tokens):

| | Uniform 3.0 bpw | This quant | Gain |
|---|---|---|---|
| plain decode | 57.2 tok/s | 57.5 tok/s | +0.5% |
| MTP drafting (4 tokens), code | 127.8 tok/s | 138.6 tok/s | +8.5% |
| MTP drafting (4 tokens), prose | 94.1 tok/s | 101.6 tok/s | +8.0% |

**Ferrite, two DGX Sparks** (tensor-parallel over RoCE; NVFP4 prompt weights; DFlash2 speculative decoding; 4
requests x 128K context; generation rates exclude prompt processing; medians of 3 runs):

| Workload | Result |
|---|---|
| code, 1 request (2K / 32K context) | 273 / 238 tok/s |
| code, 4 requests combined | 473 tok/s |
| code edits (mostly copied from the prompt), 1 / 4 requests | 582 / 481 tok/s |
| prose, 1 request (replies that do not repeat the test prompt) | 76 tok/s |
| prompt processing at 2K / 8K / 32K / 64K / 120K | 3,544 / 3,335 / 2,799 / 2,321 / 1,792 tok/s |

**Ferrite with 4 agents** (shared ~13K-token system prompt, multi-turn, cold start; prompt cache off -> on):

| | Thinking off | Thinking on |
|---|---|---|
| prompt tokens processed | -74% | -86% |
| time to first token, first turn | 16.4 -> 5.1 s | 16.4 -> ~5.9 s |
| time to first token, later turns | 2.5-2.7 -> ~0.5 s | 5.6-10.5 -> ~0.35 s |
| combined output | 46.5 -> 92 tok/s | 36.9 -> 108 tok/s |

## How to run

**Ferrite** (DGX Spark). The full guide is in Ferrite's
[quickstart](https://github.com/BHCC2025/ferrite/blob/main/docs/quickstart.md). In short:

```bash
scripts/check-deps.sh --build --install                    # pinned Rust toolchains, CUDA 13 check
cargo build --release -p fe-server --features ferrite -p fe-cli
export FERRITE_MODELS=/var/tmp/models                      # ~38 GB free
scripts/fetch-models.sh                                    # this quant + NVFP4 prompt weights + DFlash2 drafter, sha256-checked
scripts/serve.sh                                           # one Spark: OpenAI-compatible server on :8000, model id "spark"
```

- One Spark needs about 95 GB of free memory (model + 4 x 128K context + a 12 GB prompt-cache pool).
- Two Sparks run tensor-parallel with `tools/serve-tp2.sh`.
- Optional companions: [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) for faster
  prompt processing, and [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) for speculative
  decoding.

**exllamav3** (1.6.1 or later):

```python
from exllamav3 import Config, Model, Cache, Tokenizer, Generator

config = Config.from_directory("Qwen3.8-27B-EXL3-3.0bpw-mix")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 16384)    # context length; raise it if your GPU has memory to spare
model.load()
tokenizer = Tokenizer.from_config(config)
generator = Generator(model = model, cache = cache, tokenizer = tokenizer)

# ChatML prompt. The empty think block turns reasoning off;
# end the prompt right after "<|im_start|>assistant\n" to let the model reason first.
prompt = (
    "<|im_start|>user\nWrite a haiku about rust.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
print(generator.generate(
    prompt = prompt,
    max_new_tokens = 200,
    encode_special_tokens = True,
    stop_conditions = [tokenizer.eos_token_id, "<|im_end|>"],
    completion_only = True,
))
```

To chat in a terminal with MTP drafting from the model's own MTP layer, use exllamav3's example script:
`python examples/chat.py -m <model dir> -mode qwen35 -mtp` (add `-think` to turn reasoning on).

**TabbyAPI**: put the folder in TabbyAPI's model directory (`model_dir`, default `models`) and set in `config.yml`:

```yaml
model:
  model_name: Qwen3.8-27B-EXL3-3.0bpw-mix
  cache_size: 16384      # context length; the default is only 4096 tokens
draft_model:
  draft_mode: mtp        # optional: speculative decoding with the model's own MTP layer
```

TabbyAPI ships prebuilt exllamav3 wheels for x86_64 Linux and Windows. On aarch64 (for example DGX Spark),
build exllamav3 from source.

## How it was made

| Step | Details |
|---|---|
| Sensitivity | exllamav3's published per-tensor measurement for this model (`noise_attrib.json`, `rfn_2.00bpw.json`, [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3), main branch). Nothing was re-measured |
| Allocation | `sc_optimize.py -m noise_attrib.json -rr rfn_2.00bpw.json -b 3.0 -hb 5 -al 2.0 -mink 2`: whole-number widths only, 5-bit head, default tie groups |
| Conversion | `convert.py -i Qwen3.8-27B -rcp r_3.0_h5.yaml -mb 4`: default calibration, `mul1` codebook, MTP layer at 4 bpw; exllamav3 1.6.1 (commit 53ea1d8) with the two fixes below; about 70 minutes on one RTX 5080 16 GB |

Design choices: half-integer widths (`-ks 0.5`) scored slightly better KL (code 0.0214, text 0.0449) but decoded about
3% slower, so this quant keeps whole numbers. The calibration set decides where gains land: calibrating on the chat
trace instead improved chat KL and made code, text and long16k worse.

<details><summary>Two conversion fixes for a 16 GB card (patch against exllamav3 53ea1d8)</summary>

Quantizing the output head (248,320 x 5120) ran out of memory on 16 GB. Two changes to
`exllamav3/modules/quant/exl3_lib/quantize.py`:

- `preapply_had_r` applies the Hadamard transform chunk by chunk in place instead of concatenating three FP32
  copies. Same arithmetic, less memory.
- The per-channel scale refit ("polish") is skipped for tensors over 4e8 elements, which is the output head only.
  Every other tensor is refit normally. The KL numbers above include this.

```diff
--- a/exllamav3/modules/quant/exl3_lib/quantize.py
+++ b/exllamav3/modules/quant/exl3_lib/quantize.py
@@ -445,10 +445,10 @@
     if rows >= k:
         x = (x.view(k, -1, had_dim) @ had).view(k, n)
     else:
-        x = torch.cat([
-            (x[i:i + rows].reshape(-1, had_dim) @ had).view(-1, n)
-            for i in range(0, k, rows)
-        ], dim = 0)
+        # in place, chunk by chunk (the cat version holds 3 copies of x: OOMs on lm_head with 16 GB)
+        x = x.contiguous()
+        for i in range(0, k, rows):
+            x[i:i + rows] = (x[i:i + rows].reshape(-1, had_dim) @ had).view(-1, n)
     x = x.to(x_dtype)
     return x
 
@@ -1826,7 +1826,8 @@
         weight_q *= sv
 
         # Polish: refit the per-channel fp16 scales to the quantized weight (free at inference)
-        if not q_fallback and H is not None and not quant_args.get("no_refit"):
+        # (16 GB card: skip the scale refit for lm_head-sized tensors, it needs ~3 extra fp32 copies of the weight)
+        if not q_fallback and H is not None and not quant_args.get("no_refit") and weight_q.numel() < 400_000_000:
             H_orig = unrotate_H(H, H_data["su"])
             weight_q, su_f, sv_f, e_before, e_after = refit_scales(weight_orig, weight_q, H_orig, su, sv,
                                                                    damping = damping_share(quant_args))
```

</details>

## Files

Two weight shards (`model-0000x-of-00002.safetensors`), the configs, the tokenizer files, `chat_template.jinja`,
`LICENSE` and this card. `sha256sums.txt` lists the SHA-256 of every file; Ferrite's `scripts/fetch-models.sh`
verifies them after download.

## License and credits

- **Model:** Qwen ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)), Apache-2.0. These weights are a
  quantized derivative, not the original weights; `LICENSE` is the base model's license, unchanged.
- **Format, optimizer, converter and sensitivity data:** exllamav3
  ([turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), MIT). EXL3 builds on
  [QTIP](https://arxiv.org/abs/2406.11235).
- **Engine:** [Ferrite](https://github.com/BHCC2025/ferrite), Apache-2.0.
