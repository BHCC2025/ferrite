#!/usr/bin/env bash
# Start Ferrite's OpenAI-compatible server on one DGX Spark (one command once the models are downloaded):
#
#   FERRITE_MODELS=/path/to/models scripts/serve.sh [extra fe-server arguments]
#
# Serves /v1/chat/completions, /v1/completions, /v1/models and /health (streaming, tool calls, thinking via
# chat_template_kwargs.enable_thinking) at http://$FERRITE_HOST:$FERRITE_PORT/v1 under the model id
# $FERRITE_MODEL_ID. It runs in the foreground; Ctrl-C stops it.
#
# Environment (defaults in brackets). The site settings (model folders, GPU lock, ...) come from tools/site.sh,
# whose FERRITE_* table is in the README; this script adds the server settings below.
#   FERRITE_MODELS    folder with the model folders from scripts/fetch-models.sh [/var/tmp/models]; or set
#                     FERRITE_EXL3_DIR / FERRITE_NVFP4_DIR / FERRITE_DRAFT_DIR one by one
#   FERRITE_HOST      listen address [127.0.0.1; 0.0.0.0 to serve the LAN; there is no API key, so firewall it]
#   FERRITE_PORT      [8000]
#   FERRITE_MODEL_ID  model id reported by /v1/models; requests may name any model [qwen3.8-27b]
#   FERRITE_GPU_LOCK  the node's GPU lock; the server holds it while it runs and refuses to start when another
#                     Ferrite GPU job holds it [/var/tmp/ferrite-gpu.lock]
#   FERRITE_CTX       context length per request [131072 with fp4 prompt weights, 32768 with bf16; lower it to save
#                     memory: the KV cache is ~65 KB a token a slot]
#   FERRITE_SLOTS     requests at once [4; at most 4 with dflash drafting: more requests wait in a queue]
#   FERRITE_PROMPT    prompt (prefill) weights: fp4 (the NVIDIA NVFP4 checkpoint) or bf16 (reconstructed from the
#                     EXL3 quant at load: no NVIDIA download, about half the prefill speed and ~30 GB more memory)
#                     [fp4 when FERRITE_NVFP4_DIR is present, else bf16]
#   FERRITE_DRAFT     speculative decoding: dflash (DFlash2 drafter with verify trees), mtp (the model's own MTP
#                     head), auto (DFlash2 for up to 2 streams, MTP above) or off [dflash when the drafter is
#                     present, else mtp]
#   FERRITE_CACHE     prompt cache: 1 keeps each conversation's state at its turn boundaries (and the shared system
#                     prompt), so a later turn, or another conversation with the same system prompt, skips what was
#                     already computed; 0 turns it off. It needs dflash or off drafting (with mtp / auto it falls
#                     back to reusing an identical prefix in the same slot) [1]
#   FERRITE_PARK_GB   (or FE_PARK_GB) memory for the cache's parked conversations and shared prefixes, on top of the
#                     model and KV cache; the server shrinks it if needed to keep FE_PARK_RESERVE_GB (20) of memory
#                     available [12 on one Spark; tools/serve-tp2.sh takes 24 a node]
#   FERRITE_BIN       fe-server binary [bin/fe-server, else target/release/fe-server]
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
die() { echo "serve: $*" >&2; exit 1; }
case ${1:-} in -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac

user_min_free=${FERRITE_MIN_FREE_GB:-}
# shellcheck source=../tools/site.sh
. "$root/tools/site.sh"
quant=$FERRITE_EXL3_DIR
nvfp4=$FERRITE_NVFP4_DIR
draft=$FERRITE_DRAFT_DIR
bin=${FERRITE_BIN:-}
if [ -z "$bin" ]; then
  if [ -x "$root/bin/fe-server" ]; then bin=$root/bin/fe-server; else bin=$root/target/release/fe-server; fi
fi
[ -x "$bin" ] || die "no fe-server binary at $bin (unpack a release, or cargo build --release -p fe-server --features ferrite)"
[ -f "$quant/config.json" ] || die "no model at $quant: run scripts/fetch-models.sh (FERRITE_MODELS=$FERRITE_MODELS)"

prompt=${FERRITE_PROMPT:-}
[ -n "$prompt" ] || { [ -f "$nvfp4/config.json" ] && prompt=fp4 || prompt=bf16; }
dr=${FERRITE_DRAFT:-}
[ -n "$dr" ] || { [ -f "$draft/config.json" ] && dr=dflash || dr=mtp; }

case $prompt in
  fp4) [ -f "$nvfp4/config.json" ] || die "FERRITE_PROMPT=fp4 needs $nvfp4 (scripts/fetch-models.sh prompt)"
       export FE_PROMPT=fp4 ;;
  bf16) unset FE_PROMPT ;;
  *) die "FERRITE_PROMPT is fp4 or bf16, not '$prompt'" ;;
esac
case $dr in
  dflash|auto)
    [ -f "$draft/config.json" ] || die "FERRITE_DRAFT=$dr needs $draft (scripts/fetch-models.sh draft)"
    export FE_DRAFT=dflash FE_SPEC=16 FE_TREE=1 FE_DF_BLOCK=16 FE_COPY_ROWS=47
    if [ "$dr" = auto ]; then export FE_MTP=1; else export FE_MTP=0; fi ;;
  mtp) unset FE_DRAFT; export FE_MTP=1 FE_SPEC=8 ;;
  off) unset FE_DRAFT; export FE_MTP=0 FE_SPEC=0 ;;
  *) die "FERRITE_DRAFT is dflash, mtp, auto or off, not '$dr'" ;;
esac
if [ "${FERRITE_SLOTS:-4}" -gt 4 ] && [ "$dr" = dflash ]; then
  die "FERRITE_SLOTS=${FERRITE_SLOTS} is not supported with DFlash2 drafting (FERRITE_DRAFT=dflash): one draft pass serves at most 4 streams in this build. Use FERRITE_SLOTS=4, or FERRITE_DRAFT=auto / mtp for more slots (MTP drafting above 2 streams; not tested above 4 slots)"
fi

cache=${FERRITE_CACHE:-${CACHE:-1}}
case $cache in
  0|1) export FE_CACHE=$cache ;;
  *) die "FERRITE_CACHE is 1 or 0, not '$cache'" ;;
esac
# one Spark: a 12 GB pool (24 bottoms out at ~20 GB available under 4 agents with thinking; 12 keeps ~34 GB and
# still parks ~92K tokens); the server also shrinks it if needed to keep 20 GB available
park=${FERRITE_PARK_GB:-${FE_PARK_GB:-12}}
if [ "$cache" = 1 ]; then export FE_PARK_GB=$park; else park=0; fi

# bf16 prompt weights take ~30 GB more than the NVFP4 ones: keep the default KV cache inside one Spark's memory
ctx=${FERRITE_CTX:-}
[ -n "$ctx" ] || { [ "$prompt" = fp4 ] && ctx=131072 || ctx=32768; }
# memory estimate (GB): EXL3 13 + prompt weights (NVFP4 21 | bf16 52) + drafter 4 + KV 64 KB a token a slot + ~10
# + the prompt cache's pool (FE_PARK_GB; the server shrinks it if less is free, keeping 20 GB available)
# (tools/site.sh's FERRITE_MIN_FREE_GB is the two-node figure; a value set by the caller wins)
wts=$([ "$prompt" = fp4 ] && echo 21 || echo 52)
need=${user_min_free:-$(( 13 + wts + 4 + ctx * ${FERRITE_SLOTS:-4} * 64 / 1048576 + 10 + ${park%.*} ))}
FERRITE_MIN_FREE_GB=$need "$root/scripts/check-deps.sh" "$bin" || die "fix the errors above first"
echo "serve: $quant, prompt weights $prompt, drafting $dr, ctx $ctx x ${FERRITE_SLOTS:-4} slots, prompt cache \
$([ "$cache" = 1 ] && echo "on (a pool of up to ${park} GB, keeping 20 GB available)" || echo off)"
echo "serve: http://${FERRITE_HOST:-127.0.0.1}:${FERRITE_PORT:-8000}/v1 as model '${FERRITE_MODEL_ID:-qwen3.8-27b}' (loading takes a minute)"
mkdir -p "$(dirname "$FERRITE_GPU_LOCK")"
flock -n "$FERRITE_GPU_LOCK" true || die "another Ferrite GPU job holds $FERRITE_GPU_LOCK on this node"
exec flock -n -E 75 "$FERRITE_GPU_LOCK" "$bin" --engine ferrite --model-dir "$quant" --weights-dir "$quant" \
  --nvfp4-dir "$nvfp4" --draft-dir "$draft" \
  --host "${FERRITE_HOST:-127.0.0.1}" --port "${FERRITE_PORT:-8000}" --served-model-name "${FERRITE_MODEL_ID:-qwen3.8-27b}" \
  --slots "${FERRITE_SLOTS:-4}" --max-model-len "$ctx" --prefill-chunk 2048 "$@"
