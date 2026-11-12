---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- xq
- quantized
- intel-arc
- ferrite
---

# Qwen3.8-27B · XQ 5.0 bpw mix

A mixed-precision quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in the **XQ** format,
the weight format of [Ferrite](https://github.com/BHCC2025/ferrite)'s engine for the **Intel Arc Pro B70** (32 GB).
It averages **5.0 bits per weight (29 GB with the BF16 embedding)**; each tensor has its own width (4, 5, 6 or 8 bits).

XQ is decoded directly by GPU kernels: grouped integer codes with fp16 scales per 128 inputs, after a Hadamard
rotation of the inputs. It is meant for one engine, Ferrite; it does not load in transformers, llama.cpp or
exllamav3.

## At a glance

| | |
|---|---|
| Base model | Qwen/Qwen3.8-27B: 64 layers (48 Gated DeltaNet + 16 gated attention), hidden 5120, vocab 248,320, one MTP layer |
| Format | XQ version 0 (`xq_config.json` + `xq-0000x.safetensors`), read by Ferrite's `fe-xq` crate |
| Bits | 5.0 bpw average over the linears including their scales (4 / 5 / 6 / 8-bit per tensor); output head 6-bit, not rotated |
| Method | GPTQ on 128 x 2048 calibration tokens for 497 tensors, round-to-nearest for 8 |
| Size | 30.3 GB on disk, of which 1.3 GB is the optional MTP draft head `dhead_q8.bin` |
| Accuracy | KL vs BF16: code 0.0054, prose 0.0105, chat 0.0057, 16K context 0.0048, 120K context 0.0033 |
| Context | measured to 120K tokens (Ferrite serves 128K on the 32 GB card) |
| Runs on | Ferrite on an Intel Arc Pro B70 |
| License | Apache-2.0, inherited from the base model |

## Accuracy

KL divergence from the BF16 model, mean per position, lower is better. Measured in Ferrite on the Arc Pro B70 with a
q8 KV cache.

| Test set | fp16 prompt path | int8 prompt path (Ferrite's default) |
|---|---|---|
| code | 0.0054 | 0.0061 |
| prose | 0.0105 | 0.0114 |
| chat | 0.0057 | 0.0065 |
| 16K-token prompts | 0.0048 | 0.0055 |
| 120K-token prompts | 0.0033 | 0.0037 |

The two columns are the same weights. Ferrite processes prompts with an int8 matrix product by default, which is
about 1.5x faster and costs the difference shown; `FERRITE_PREFILL=f16` selects the fp16 path. Decoding is the same
in both. Speculative decoding is lossless: greedy output is token for token what plain decoding gives.

## Performance

Ferrite on one Arc Pro B70 capped at 180 W and 2000 MHz, DFlash2 speculative decoding, 128K context configured,
greedy decoding, tokens per second:

| Workload | Result |
|---|---|
| code, 2K / 32K / 100K context | 124 / 111 / 79 |
| prose, 2K context | 79 |
| code edits (mostly copied from the prompt) | 201-244 |
| plain decoding, no speculation | 27.6 |
| prompt processing at 2K / 8K / 32K / ~120K | ~2,090 / ~2,240 / ~2,060 / ~1,490 |
| device memory at load, 128K context | 28.6 GB |

## How to run

The full guide is Ferrite's [Arc B70 page](https://github.com/BHCC2025/ferrite/blob/main/docs/arc-b70.md). In short:

```bash
git clone https://github.com/BHCC2025/ferrite && cd ferrite
rustup toolchain install nightly-2026-11-04 --component rust-src --component llvm-tools
cargo build --release --features xe -p fe-server
export FERRITE_MODELS=$HOME/models                # ~36 GB free
scripts/fetch-models.sh xq draft                  # this quant + the DFlash2 drafter, sha256-checked
scripts/serve-xe.sh                               # OpenAI-compatible server on :8000, model id "qwen3.8-27b"
```

Companion: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), the draft model for
speculative decoding. Without it Ferrite drafts with the model's own MTP layer and `dhead_q8.bin`.

## What is in it

| Tensor | 4-bit | 5-bit | 6-bit | 8-bit |
|---|---|---|---|---|
| GDN in_proj_qkv, in_proj_z (48 layers each) | 6 | 42 | | |
| GDN out_proj (48 layers) | 1 | 47 | | |
| GDN in_proj_a, in_proj_b (48 layers each) | | | | 48 |
| attention q / k / v_proj (16 layers each) | | 16 | | |
| attention o_proj (16 layers) | | 14 | 2 | |
| MLP gate_proj, up_proj (64 layers each) | 19 | 44 | 1 | |
| MLP down_proj (64 layers) | 10 | 54 | | |
| MTP layer (8 tensors) | | 8 | | |
| output head | | | 1 (unrotated) | |

Tensors that share an input keep one width (qkv with z, gate with up), so their rotated input is prepared once. The
embedding, the norms and the GDN convolutions stay BF16. The codes average 4.8 bits; the fp16 group scales bring the
linears to 5.0 bits per weight.

`xq_config.json` lists every tensor's width, group size, rotation and method; `quant_log.json` has each tensor's
quantization error. The vision tower is not included (Ferrite is text-only).

`dhead_q8.bin` is a ranked int8 copy of the BF16 output head (token ids ordered by frequency, one scale per row),
used only as the draft head of MTP drafting. It never decides an output token.

## Files

Eight weight shards (`xq-0000x.safetensors`), `xq_config.json`, `quant_log.json`, the base model's configs and
tokenizer files, `chat_template.jinja`, `dhead_q8.bin`, `LICENSE` and this card. `sha256sums.txt` lists the SHA-256
of every file; Ferrite's `scripts/fetch-models.sh` verifies them after download.

## License and credits

- **Model:** Qwen ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)), Apache-2.0. These weights are a
  quantized derivative, not the original weights; `LICENSE` is the base model's license, unchanged.
- **Method:** [GPTQ](https://arxiv.org/abs/2210.17323) on inputs rotated by a 128-wide Hadamard transform; the
  calibration data is [exllamav3](https://github.com/turboderp-org/exllamav3)'s default set.
- **Engine:** [Ferrite](https://github.com/BHCC2025/ferrite), Apache-2.0.
