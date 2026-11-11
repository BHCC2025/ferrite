#!/bin/bash
# ferrite on one RTX 5080 (16 GB, sm_120): the small-memory preset of tools/serve.sh (docs/rtx-5080.md). One slot of
# 128K context: EXL3 3.0 mix decode on the card, a 6-bit K/V cache, the NVFP4/FP8 prompt weights streamed from pinned
# host memory, the embedding table and the prompt cache's pool in host memory, MTP drafting with copy rows.
# The binaries must be built for the card: FE_KARCH=sm_120a cargo build --release -p fe-server --features ferrite -p fe-cli
# usage: tools/serve-5080.sh [--help]     env as tools/serve.sh; this preset's defaults:
#   CTX 131072, SLOTS 1, DRAFT mtp, SPEC 8 (MTP drafts a round), COPY 47 (copy rows), PROMPT fp4 when the NVIDIA
#   checkpoint is there, CHUNK 2240 (prompt rows a forward: 35 x 64 rows fill the 84 SMs in whole waves of attention blocks), PARK_GB 8 (host memory), FE_KVQ 6 (4 = a 4-bit K/V cache: 0.8 GB less at 128K,
#   KL +4-8%), FE_PROMPT_RING 2 (layers of prompt weights on the card at once)
export SLOTS=${SLOTS:-${FERRITE_SLOTS:-1}}
if [ "$SLOTS" -gt 1 ]; then
  # several requests at once (up to 4 lanes): the card holds 4 x 28672 tokens of context beside the weights, with 15
  # copy rows a sequence and a 1280-row prompt chunk (docs/rtx-5080.md, "Several requests at once")
  [ "$SLOTS" -le 4 ] || { echo "serve-5080.sh: SLOTS=$SLOTS: this card fits at most 4" >&2; exit 2; }
  export CTX=${CTX:-${FERRITE_CTX:-28160}} COPY=${COPY:-15} CHUNK=${CHUNK:-2048}
  export FE_BATCH_ROWS=${FE_BATCH_ROWS:-$CHUNK}
  export FE_POOL=${FE_POOL:-1}   # one context pool: a request too long for a slot takes all of them and runs alone
fi
export CTX=${CTX:-${FERRITE_CTX:-131072}}
export DRAFT=${DRAFT:-${FERRITE_DRAFT:-mtp}} SPEC=${SPEC:-8} COPY=${COPY:-47}
export CHUNK=${CHUNK:-2240}
export PARK_GB=${PARK_GB:-${FERRITE_PARK_GB:-8}} FE_PARK_RESERVE_GB=${FE_PARK_RESERVE_GB:-8}
export FE_LOWMEM=${FE_LOWMEM:-1} FE_KVQ=${FE_KVQ:-6}
exec "$(dirname "$0")/serve.sh" "$@"
