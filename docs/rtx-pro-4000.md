# Ferrite on the NVIDIA RTX PRO 4000 Blackwell (24 GB)

Ferrite serves Qwen3.8-27B on a single **RTX PRO 4000 Blackwell** (24 GB, 70 SMs, compute capability 12.0, 145 W)
with the `rtx-pro-4000` preset, `tools/serve-pro4000.sh`. It is the small-memory arrangement of the 16 GB cards
([rtx-5080.md](rtx-5080.md): EXL3 3.0 bpw decode weights and a 6-bit K/V cache on the card, the NVFP4/FP8 prompt
weights in pinned host memory, MTP drafting with copy rows), with the extra 8 GB spent where the card's time account
says they pay. Read that page first: everything in it applies unless this page says otherwise.

> **Status: experimental.** One model, one card, built from source, on Linux. Developed and measured on a single
> machine (a two-socket server, the card in a PCIe 4.0 x8 slot, two other cards at work beside it).

## Results

Qwen3.8-27B, EXL3 3.0 bpw mix, the default preset (128K context, one slot), greedy decoding, thinking off, the
repository's bench tools, medians of 3 runs on a warm card (86-88 C at its 145 W limit). "16 GB preset" is
`tools/serve-5080.sh` on the same card with Ferrite 0.7.0, which is how the card ran before this preset.

| | 16 GB preset, 0.7.0 | `rtx-pro-4000` preset |
|---|---|---|
| Code, 2K / 16K / 56K context, tokens/s | 184 / 160 / 137 | 200 / 171 / 143 |
| Prose, 2K context (clean replies), tokens/s | 66-78 | 74-79 |
| Code edit, 500 tokens, tokens/s | 462 | 501 |
| Plain decoding, no speculation, tokens/s | 36.5 | 40.1 |
| Prompt processing, 2K / 16K / 56K / 110K prompt, tokens/s | 1,599 / 2,714 / 1,965 / 1,423 | 2,712 / 3,029 / 2,112 / 1,490 |
| First token of a short prompt (44 tokens) | 1.19 s | 0.70 s |
| First token of a 2,118-token prompt (the edit) | 1.23 s | 0.79 s |
| Four requests at once, 2K / 16K shared prefix: total tokens/s | 186 / 149 | 255 / 163 |
| Device memory, 128K context | 15.5 GB | 22.2 GB of 24.5 |

- Code at 56K with the 16 GB preset was measured with a 64K context configured.
- Four requests at once: `tools/bench/lanes.py --lanes 4 --prefix 2000,16000` against `SLOTS=4`, 800 tokens a
  request, total tokens/s over the wall time of the set (medians of 3 sets). The left column was measured with
  0.6.0, where in one set of three or four the requests did not share their prefix (fixed here: a worker woken from
  idle now waits for requests the server is still tokenizing). Four code streams decode at 324 tokens/s together.
- The prose figures are the replies the benchmark marks clean (seeds 2, 3 and 6 here; `tools/bench/README.md`).
- Speculative decoding is lossless: with drafting off the server gives the same text (ten requests: code, prose, a
  16K prompt, four file edits, a two-turn conversation with thinking, a tool call). The same ten requests give the
  same text with one slot and with the four-slot preset.
- A 5-fact needle test over a 120K-token prompt found 5 of 5.

## What the preset does

- **Decode plans for 70 SMs** (`crates/fe-exl3/src/plans.rs`, from `exl3-tune` on the card). A card without a table
  of its own uses the DGX Spark's. With the card's own plans the tuner's sum over the model's GEMVs is 22.6 ms a
  token (28.1 with the untuned rule), and plain decoding goes from 36.5 to 40.3 tokens/s. **This changes numerics on 70-SM cards**: the plans fix the order of the
  sums. KL between the two on the card is 0.0004 (code), 0.0010 (text) and 0.0005 (16K context); KL against BF16
  stays 0.021 / 0.043 / 0.023. Greedy replies can differ from a 0.7.0 server's on this card at tokens whose two
  best candidates are nearly tied (4 of the 10 gate requests did). `FE_EXL3_TABLE=gb10` gives the old numerics.
- **Resident layers of prompt weights** (`FERRITE_RESIDENT`, default `auto`). On a 16 GB card every prompt of more
  than 16 tokens waits for one pass of the 16.8 GB of prompt weights over the bus: 1.13 s of the 1.19 s to the first
  token here (PCIe 4.0 x8; with the copies skipped the 44-token prompt takes 0.06 s). The preset keeps as many
  layers on the card as fit beside the model and the K/V cache, spread evenly over the stack so the bus works on a
  streamed layer while its resident neighbours compute: 27 of 64 with a 128K context. Each layer takes about 250 MiB
  and 18 ms off the pass. The bits do not change.

  | Resident layers | 0 | 16 | 24 | 27 (128K default) | 30 |
  |---|---|---|---|---|---|
  | First token, 44-token prompt | 1.19 s | 0.91 s | 0.76 s | 0.70 s | 0.64 s |
  | First token, 2K prompt | 1.21 s | 0.93 s | 0.78 s | 0.72 s | 0.66 s |
  | Device memory | 15.5 GB | 19.3 GB | 21.4 GB | 22.2 GB | 23.0 GB |

  A shorter context leaves room for more (`CTX=65536`: 34 layers by the same rule), and `FERRITE_RESIDENT=30` fits at
  128K with 1.5 GB left free. `FERRITE_VRAM_WORK_MIB` (1280) is the room `auto` keeps for prompt groups and graphs.
- **Prompt kernels for this card.** The NVFP4 GEMMs take the mbarrier pipeline and prompt groups run the next
  chunk's GDN projection beside the recurrence (its 48 head blocks leave 22 of the 70 SMs idle); the chunk is 2048
  rows (2240 on the 16 GB preset). 16K / 56K / 110K prompts run 12 / 7 / 5% faster than with the 16 GB preset's
  settings, the same bits.
- **Several requests at once.** `SLOTS=4` shares one pool of 4 x 32,768 tokens, the memory of the one-slot 128K
  context, and keeps 47 copy rows a request (the 16 GB preset drops to 15 for memory): a lone edit runs at 514
  tokens/s on the four-slot preset, and a lone code request at the one-slot speed. A request longer than a unit
  takes the whole pool and runs alone.

Settings: `tools/serve-pro4000.sh --help`. `scripts/serve.sh` and `scripts/ferritectl.sh start` pick this preset by
the card's name; `scripts/ferritectl.sh start rtx-pro-4000` names it.

## Where the time goes (measured on the card)

**A decoding step** (one verify forward for the drafted rows; `FE_SCHED_PROF=2`, 0.7.0 before the plans):

| | steps for 500 tokens | tokens a step | draft | forward | commit + rest | a step |
|---|---|---|---|---|---|---|
| Code, 2K | 66 | 7.6 | 5.3 ms | 33.8 ms | 2.4 ms | 41.5 ms |
| Prose, 2K | 225 | 2.2 | 1.7 ms | 28.1 ms | 1.7 ms | 31.5 ms |
| Edit | 18 | 27.8 | 1.5 ms | 50.8 ms | 3.3 ms | 55.5 ms |

The forward takes 28 ms for up to 4 rows, 30 for 5-8, 34 for 9-16, 46 for 17-32 and 66 for 33-48; a draft token
costs 0.95 ms. Of a plain token's 27 ms, 84% is the EXL3 GEMVs of the 64 layers (the MLP alone 53%) and 5% the
output head and the embedding; the norms, the GDN recurrence and attention share the rest. At 57K context attention
adds 9 ms. The card runs these at its 145 W limit: `nvidia-smi` reports the power cap
active in 19 of 20 samples under load and an SM clock of about 1,840 MHz of 3,090. Decoding is power-bound.

**A prompt.** One 2048-row chunk computes in about 0.5 s (MLP 41%, the GDN layers 44%, attention 12%); at 16K the
chunks average 0.58 s (MLP 35%, GDN 37%, attention 25%) and at 56K attention is 43%. Prompts of 8K and more are
bound by this computation, shorter ones by the bus.

## What did not pay (measured, left at the 16 GB preset's values)

- A bf16 K/V cache (`FE_KVQ=0`): code at 2K / 16K / 56K 180 / 157 / 134 tokens/s against 182 / 161 / 137 with 6 bits,
  for 5.2 GB more at 128K. The 6-bit cache only saves memory.
- A wider or narrower draft head (`FE_DRAFT_VOCAB` 32,768 / 131,072 / 248,192): code 168 / 187 / 167 tokens/s against
  201 with 65,536.
- Draft depth (`SPEC` 6 / 10 / 12 / 16): 191 / 201 / 199 / 199 on code at 2K against 201 with 8. `SPEC=10` is 8% faster
  on the 16K code benchmark and 1% slower on edits and at 56K (3 fresh starts each).
- Copy rows (`COPY` 31 / 63, `FE_COPY_AGREE` 2 / 4): within 3% of 47 / 3 either way.
- A larger ring (`FE_PROMPT_RING` 4, 8, 16): no change.

## A 262,144-token context

`CTX=262144` starts (the K/V cache takes 7.1 GB, 13 layers stay resident, 22.2 GB in use) and a 10-fact needle test
over a 240,524-token prompt found 10 of 10; the prompt took 279 s (863 tokens/s) and each further question 3 s from
the prompt cache. It is not the default.
