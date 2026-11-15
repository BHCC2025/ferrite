#!/bin/bash
# Ferrite on one RTX 5080 or RTX 5060 Ti (16 GB, sm_120): the small-memory preset of scripts/serve.sh
# (docs/rtx-5080.md). One slot of up to 128K context: EXL3 3.0 mix decode on the card, a 6-bit K/V cache, the
# NVFP4/FP8 prompt weights streamed from pinned host memory, the embedding table and the prompt cache's pool in host
# memory, MTP drafting with copy rows. The context is sized to the card's free memory at start (128K on an otherwise
# empty card; less, with a message, when a desktop or another program holds some of it).
# The binaries must be built for the card: FE_KARCH=sm_120a cargo build --release -p fe-server --features ferrite -p fe-cli
#
#   tools/serve-5080.sh [extra fe-server arguments]      tools/serve-5080.sh --help: these settings
#
# Everything of scripts/serve.sh --help applies (FERRITE_HOST [0.0.0.0], FERRITE_PORT [8000], FERRITE_MODEL_ID,
# FERRITE_GPU_LOCK, FERRITE_LOCK_WAIT, FERRITE_FORCE, ...); the short names (CTX, SLOTS, ...) are the same settings.
# This preset's defaults:
#   FERRITE_CTX / CTX       context [the largest that fits: at most 131072, or 28160 a request with SLOTS > 1]
#   FERRITE_SLOTS / SLOTS   requests at once, 1-4 [1]. 2-4 share one pool of 4 x 28160 tokens (FE_POOL=1): a longer
#                           request takes the whole pool and runs alone
#   FERRITE_DRAFT / DRAFT   mtp (the MTP head with copy rows) or off (plain decoding) [mtp]
#   SPEC, COPY              MTP drafts a round [8], copy rows a round [47; 15 with SLOTS > 1]
#   FERRITE_PROMPT / PROMPT fp4 when the NVIDIA checkpoint is there, else bf16 (the EXL3 weights on the decode kernels)
#   CHUNK                   prompt rows a forward [2240: 35 x 64 rows fill the 84 SMs in whole waves; 2048 with SLOTS > 1]
#   PARK_GB                 the prompt cache's pool in host memory [8]
#   FE_KVQ                  bits a K/V value: 6, 4 (0.8 GB less at 128K, KL +4-8%) or 0 (bf16: about 55K fits) [6]
#   FE_PROMPT_RING          layers of prompt weights on the card at once [2]
#   FERRITE_VRAM_RESERVE_MIB  card memory left free for other programs [128; 768 when the card drives a display]
set -euo pipefail
case ${1:-} in -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac
die() { echo "serve-5080: $*" >&2; exit 1; }
# the short names of tools/serve.sh are the FERRITE_* settings
for p in SLOTS CTX PORT PROMPT DRAFT CACHE PARK_GB CHUNK SPEC COPY; do
  if [ -n "${!p:-}" ]; then export "FERRITE_$p=${!p}"; fi
done
export FERRITE_SLOTS=${FERRITE_SLOTS:-1}
case $FERRITE_SLOTS in ''|*[!0-9]*) die "SLOTS / FERRITE_SLOTS must be 1-4, not '$FERRITE_SLOTS'" ;; esac
[ "$FERRITE_SLOTS" -ge 1 ] && [ "$FERRITE_SLOTS" -le 4 ] || die "SLOTS=$FERRITE_SLOTS: this card fits 1-4"
export FERRITE_DRAFT=${FERRITE_DRAFT:-mtp}
case $FERRITE_DRAFT in
  mtp|off) ;;
  dflash|auto) die "FERRITE_DRAFT=$FERRITE_DRAFT: the DFlash2 drafter does not fit beside the weights on a 16 GB card; use mtp (the default) or off" ;;
  *) die "FERRITE_DRAFT is mtp or off, not '$FERRITE_DRAFT'" ;;
esac
if [ "$FERRITE_SLOTS" -gt 1 ]; then
  # several requests at once (up to 4 lanes): the card holds 4 x 28160 tokens of context beside the weights, with 15
  # copy rows a sequence and a 2048-row prompt chunk (docs/rtx-5080.md, "Several requests at once")
  export FERRITE_COPY=${FERRITE_COPY:-15} FERRITE_CHUNK=${FERRITE_CHUNK:-2048}
  export FE_BATCH_ROWS=${FE_BATCH_ROWS:-$FERRITE_CHUNK}
  export FE_POOL=${FE_POOL:-1}   # one context pool: a request too long for a slot takes all of them and runs alone
fi
export FERRITE_SPEC=${FERRITE_SPEC:-8} FERRITE_COPY=${FERRITE_COPY:-47} FERRITE_CHUNK=${FERRITE_CHUNK:-2240}
export FERRITE_PARK_GB=${FERRITE_PARK_GB:-8} FE_PARK_RESERVE_GB=${FE_PARK_RESERVE_GB:-8}
export FE_LOWMEM=${FE_LOWMEM:-1} FE_KVQ=${FE_KVQ:-6}
export FERRITE_PROFILE=rtx
exec "$(dirname "$0")/../scripts/serve.sh" "$@"
