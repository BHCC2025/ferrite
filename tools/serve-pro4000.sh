#!/bin/bash
# Ferrite on one RTX PRO 4000 Blackwell (24 GB, sm_120, 70 SMs): the rtx-pro-4000 preset of scripts/serve.sh
# (docs/rtx-pro-4000.md). It is the small-memory preset of tools/serve-5080.sh (EXL3 3.0 mix decode on the card, a
# 6-bit K/V cache, the NVFP4/FP8 prompt weights in pinned host memory, MTP drafting with copy rows) with what the
# extra 8 GB are best spent on, as measured on the card:
#   - as many layers of prompt weights as fit stay on the card (FERRITE_RESIDENT=auto: about 28 of 64 beside a 128K
#     context), so a prompt's pass over the bus moves only the others: the first token of a short prompt comes after
#     about 0.65 s instead of 1.2 s;
#   - SLOTS > 1 shares one pool of SLOTS x 32768 tokens (4 x 32768 = the one-slot context) instead of 4 x 28160;
#   - a 2048-row prompt chunk (the 16 GB preset's 2240 is sized for 84 SMs).
# The card's decode plans (70 SMs) are in crates/fe-exl3/src/plans.rs. The binaries must be built for the card:
# FE_KARCH=sm_120a cargo build --release -p fe-server --features ferrite -p fe-cli
#
#   tools/serve-pro4000.sh [extra fe-server arguments]      tools/serve-pro4000.sh --help: these settings
#
# Everything of scripts/serve.sh --help applies (FERRITE_HOST [0.0.0.0], FERRITE_PORT [8000], FERRITE_MODEL_ID,
# FERRITE_GPU_LOCK, FERRITE_LOCK_WAIT, FERRITE_FORCE, ...); the short names (CTX, SLOTS, ...) are the same settings.
# This preset's defaults:
#   FERRITE_CTX / CTX       context [the largest that fits: at most 131072, or 32768 a request with SLOTS > 1]
#   FERRITE_SLOTS / SLOTS   requests at once, 1-4 [1]. 2-4 share one pool of SLOTS x 32768 tokens (FE_POOL=1): a longer
#                           request takes the whole pool and runs alone
#   FERRITE_DRAFT / DRAFT   mtp (the MTP head with copy rows) or off (plain decoding) [mtp]
#   SPEC, COPY              MTP drafts a round [8], copy rows a round [47, also with SLOTS > 1]
#   FERRITE_PROMPT / PROMPT fp4 when the NVIDIA checkpoint is there, else bf16 (the EXL3 weights on the decode kernels)
#   CHUNK                   prompt rows a forward [2048]
#   FERRITE_RESIDENT / RESIDENT  layers of prompt weights kept on the card, 0-64, or auto: what fits beside the model,
#                           the K/V cache and FERRITE_VRAM_WORK_MIB [1280] of working room [auto]. Each layer takes
#                           about 250 MiB and takes about 18 ms off a prompt's first token (PCIe 4.0 x8)
#   PARK_GB                 the prompt cache's pool in host memory [8]
#   FE_KVQ                  bits a K/V value: 6, 4 or 0 (bf16; no faster on this card, 5.2 GB more at 128K) [6]
#   FE_PROMPT_RING          streamed layers of prompt weights on the card at once [2]
#   FERRITE_VRAM_RESERVE_MIB  card memory left free for other programs [128; 768 when the card drives a display]
set -euo pipefail
case ${1:-} in -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac
die() { echo "serve-pro4000: $*" >&2; exit 1; }
# the short names of tools/serve.sh are the FERRITE_* settings
for p in SLOTS CTX PORT PROMPT DRAFT CACHE PARK_GB CHUNK SPEC COPY RESIDENT; do
  if [ -n "${!p:-}" ]; then export "FERRITE_$p=${!p}"; fi
done
export FERRITE_SLOTS=${FERRITE_SLOTS:-1}
case $FERRITE_SLOTS in ''|*[!0-9]*) die "SLOTS / FERRITE_SLOTS must be 1-4, not '$FERRITE_SLOTS'" ;; esac
[ "$FERRITE_SLOTS" -ge 1 ] && [ "$FERRITE_SLOTS" -le 4 ] || die "SLOTS=$FERRITE_SLOTS: this preset serves 1-4"
export FERRITE_DRAFT=${FERRITE_DRAFT:-mtp}
case $FERRITE_DRAFT in
  mtp|off) ;;
  dflash|auto) die "FERRITE_DRAFT=$FERRITE_DRAFT: the DFlash2 drafter is not part of this preset; use mtp (the default) or off" ;;
  *) die "FERRITE_DRAFT is mtp or off, not '$FERRITE_DRAFT'" ;;
esac
if [ "$FERRITE_SLOTS" -gt 1 ]; then
  # several requests at once (up to 4 lanes): one pool of SLOTS x 32768 tokens. The 47 copy rows of the one-slot
  # preset stay (the 16 GB preset drops to 15 for memory): 190 MiB, a lone edit 372 -> 514 tok/s, 4 streams the same
  export FERRITE_UNIT_MAX=${FERRITE_UNIT_MAX:-32768}
  export FE_POOL=${FE_POOL:-1}   # one context pool: a request too long for a slot takes all of them and runs alone
fi
export FERRITE_SPEC=${FERRITE_SPEC:-8} FERRITE_COPY=${FERRITE_COPY:-47} FERRITE_CHUNK=${FERRITE_CHUNK:-2048}
if [ "$FERRITE_SLOTS" -gt 1 ]; then export FE_BATCH_ROWS=${FE_BATCH_ROWS:-$FERRITE_CHUNK}; fi
export FERRITE_PARK_GB=${FERRITE_PARK_GB:-8} FE_PARK_RESERVE_GB=${FE_PARK_RESERVE_GB:-8}
export FE_LOWMEM=${FE_LOWMEM:-1} FE_KVQ=${FE_KVQ:-6}
export FERRITE_RESIDENT=${FERRITE_RESIDENT:-auto}
export FERRITE_PROFILE=rtx
exec "$(dirname "$0")/../scripts/serve.sh" "$@"
