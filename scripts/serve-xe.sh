#!/usr/bin/env bash
# Start Ferrite's OpenAI-compatible server on an Intel Arc Pro B70 (experimental; docs/arc-b70.md):
#
#   FERRITE_MODELS=/path/to/models scripts/serve-xe.sh [extra fe-server arguments]
#
# Serves /v1/chat/completions, /v1/completions, /v1/models and /health (streaming, tool calls, thinking via
# chat_template_kwargs.enable_thinking) at http://$FERRITE_HOST:$FERRITE_PORT/v1 under the model id
# $FERRITE_MODEL_ID. One request runs at a time (FERRITE_SLOTS: up to 4); more wait in a queue. It runs in the
# foreground; Ctrl-C stops it.
#
# Environment (defaults in brackets):
#   FERRITE_MODELS    folder with the model folders from `scripts/fetch-models.sh xq draft` [/var/tmp/models]; or set
#                     FERRITE_XQ_DIR / FERRITE_DRAFT_DIR one by one
#   FERRITE_HOST      listen address [127.0.0.1; 0.0.0.0 to serve the LAN; there is no API key, so firewall it]
#   FERRITE_PORT      [8000]
#   FERRITE_MODEL_ID  model id reported by /v1/models; requests may name any model [qwen3.8-27b]
#   FERRITE_CTX       context length [131072; the q8 KV cache is ~35 KB a token]
#   FERRITE_SLOTS     requests served at once, 1..4 [1]. Above 1 the server keeps that many lanes: lane 0 of FERRITE_CTX
#                     positions [81920 in this mode], the others of FE_XE_LANE_CTX [28672]; needs the DFlash2 drafter
#   FERRITE_DRAFT     speculative decoding: dflash (DFlash2 drafter with verify trees) or mtp (the model's own MTP head
#                     with the draft head dhead_q8.bin from the XQ folder) [dflash when the drafter is present, else mtp]
#   FERRITE_PREFILL   prompt GEMM: i8 (int8, the default) or f16 (fp16: about 1.5x slower prompts, bit-exact with the
#                     decode weights' fp16 path) [i8]
#   FERRITE_GPU_LOCK  the GPU lock; the server holds it while it runs [/var/tmp/ferrite-gpu.lock]
#   FERRITE_BIN       fe-server binary built with --features xe [bin/fe-server, else target/release/fe-server]
# The first start quantizes the drafter's linears on the CPU (under a minute) and caches them as ferrite-xq*.bin in the
# drafter folder (FE_DF_CACHE names another file when that folder is read-only).
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
die() { echo "serve-xe: $*" >&2; exit 1; }
case ${1:-} in -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac

# shellcheck source=../tools/site.sh
. "$root/tools/site.sh"
xq=$FERRITE_XQ_DIR
draft=$FERRITE_DRAFT_DIR
bin=${FERRITE_BIN:-}
if [ -z "$bin" ]; then
  if [ -x "$root/bin/fe-server" ]; then bin=$root/bin/fe-server; else bin=$root/target/release/fe-server; fi
fi
[ -x "$bin" ] || die "no fe-server binary at $bin (cargo build --release --features xe -p fe-server)"
[ -f "$xq/xq_config.json" ] || die "no XQ model at $xq: run scripts/fetch-models.sh xq draft (FERRITE_MODELS=$FERRITE_MODELS)"
# (no grep -q here: under pipefail it would close the pipe early and fail the check)
{ ldconfig -p 2> /dev/null || true; } | grep 'libze_loader\.so\.1' > /dev/null || [ -n "${LD_LIBRARY_PATH:-}" ] \
  || ls /usr/lib/libze_loader.so.1 /usr/lib64/libze_loader.so.1 /usr/lib/x86_64-linux-gnu/libze_loader.so.1 > /dev/null 2>&1 \
  || die "the Level Zero loader (libze_loader.so.1) is not installed: see docs/arc-b70.md"

dr=${FERRITE_DRAFT:-}
[ -n "$dr" ] || { [ -f "$draft/config.json" ] && dr=dflash || dr=mtp; }
case $dr in
  dflash)
    [ -f "$draft/config.json" ] || die "FERRITE_DRAFT=dflash needs $draft (scripts/fetch-models.sh draft)"
    export FE_XE_DRAFT=dflash FE_DRAFT_DIR=$draft ;;
  mtp)
    export FE_XE_DRAFT=mtp
    if [ -z "${MTP_DHEAD:-}" ] && [ -f "$xq/dhead_q8.bin" ]; then export MTP_DHEAD=$xq/dhead_q8.bin; fi ;;
  *) die "FERRITE_DRAFT is dflash or mtp, not '$dr'" ;;
esac
case ${FERRITE_PREFILL:-i8} in
  i8) ;;
  f16) export FE_XE_PF=f16 ;;
  *) die "FERRITE_PREFILL is i8 or f16, not '${FERRITE_PREFILL}'" ;;
esac

slots=${FERRITE_SLOTS:-1}
case $slots in
  1) ctx=${FERRITE_CTX:-131072} ;;
  2|3|4)
    [ "$dr" = dflash ] || die "FERRITE_SLOTS=$slots needs the DFlash2 drafter (FERRITE_DRAFT=dflash)"
    ctx=${FERRITE_CTX:-81920}
    export FE_XE_LANES=$slots ;;
  *) die "FERRITE_SLOTS is 1 to 4, not '$slots'" ;;
esac
echo "serve-xe: $xq, drafting $dr, prompt GEMM ${FERRITE_PREFILL:-i8}, ctx $ctx, requests at once $slots"
echo "serve-xe: http://${FERRITE_HOST:-127.0.0.1}:${FERRITE_PORT:-8000}/v1 as model '${FERRITE_MODEL_ID:-qwen3.8-27b}' (loading takes a few minutes from a hard disk)"
mkdir -p "$(dirname "$FERRITE_GPU_LOCK")"
flock -n "$FERRITE_GPU_LOCK" true || die "another Ferrite GPU job holds $FERRITE_GPU_LOCK"
exec flock -n -E 75 "$FERRITE_GPU_LOCK" "$bin" --engine xe --model-dir "$xq" \
  --host "${FERRITE_HOST:-127.0.0.1}" --port "${FERRITE_PORT:-8000}" --served-model-name "${FERRITE_MODEL_ID:-qwen3.8-27b}" \
  --max-model-len "$ctx" --prefill-chunk 4096 "$@"
