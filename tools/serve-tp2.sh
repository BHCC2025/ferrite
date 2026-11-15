#!/bin/bash
# The OpenAI-compatible server on two DGX Sparks (TP2): rank 0 and the server on this host (node A), rank 1 on the
# peer replaying its engine calls (`ferrite <model> follow`). EXL3 3.0 mix decode + speculative decoding, prompt rows
# via NVFP4/FP8 (the default when $FERRITE_NVFP4_DIR holds a checkpoint; both nodes need it) or bf16. Holds the TP2
# lock and both nodes' GPU locks for its whole life (tools/tp2.sh lock order); a second start stops at once and names
# the holder (FERRITE_LOCK_WAIT=1 waits). Its process forwards Ctrl-C / SIGTERM, so `kill <pid>` stops both ranks.
# Run on node A; bin/ (or target/release/) {ferrite,fe-server} must be the same build at the same path on both nodes.
# usage: tools/serve-tp2.sh [--help: the settings and the commands it would run]
# env (FERRITE_* names as in scripts/serve.sh; the short names are the same settings):
#   FERRITE_HOST [0.0.0.0: every machine on your network can reach it, and there is no API key, so use it on a
#                trusted network or firewall the port; 127.0.0.1 keeps it on node A], FERRITE_PORT / PORT [8000],
#   FERRITE_MODEL_ID [qwen3.8-27b], FERRITE_SLOTS / SLOTS [4], FERRITE_CTX / CTX [131072],
#   FERRITE_PROMPT / PROMPT [fp4|bf16], FERRITE_DRAFT / DRAFT [dflash | mtp | auto: DFlash2 for <= 2 decoding
#   streams, MTP above | off], FERRITE_CACHE / CACHE [1; 0 = off], FERRITE_PARK_GB [24 a node],
#   SPEC [16 DFlash2, 8 MTP], TREE [1], BLOCK [16], COPY [47], FERRITE_NCCL_LIB (libnccl.so.2 when it is not on the
#   loader path; passed to rank 1 too), FERRITE_PEER (node B's ssh host); site settings: tools/site.sh
# SPEC 8 (MTP): in the server depth 8 beats 4 for code; see TP2.md
set -e
cd "$(dirname "$0")/.."
. tools/site.sh
fl_name=serve-tp2
. tools/launch-lib.sh
for p in SLOTS CTX PORT PROMPT DRAFT CACHE PARK_GB; do
  if [ -n "${!p:-}" ]; then export "FERRITE_$p=${!p}"; fi
done
SLOTS=${FERRITE_SLOTS:-4}; CTXV=${FERRITE_CTX:-131072}
fl_int FERRITE_SLOTS "$SLOTS" 1 8
fl_int FERRITE_CTX "$CTXV" 1024 1048576
DRAFT=${FERRITE_DRAFT:-dflash}
if [ "$SLOTS" -gt 4 ] && [ "$DRAFT" = dflash ]; then
  fl_die "FERRITE_SLOTS=$SLOTS is not supported with DFlash2 drafting (FERRITE_DRAFT=dflash): one draft pass serves at most 4 streams in this build. Use FERRITE_SLOTS=4, or FERRITE_DRAFT=auto / mtp for more slots (MTP drafting above 2 streams; not tested above 4 slots)"
fi
M=$FERRITE_EXL3_DIR
BIN=bin/fe-server; [ -x $BIN ] || BIN=target/release/fe-server
[ -x $BIN ] || fl_die "no fe-server at bin/ or target/release/"
export FE_BIN=${FE_BIN:-$(dirname $BIN)/ferrite}
DRY=0
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
  sed -n '2,/^set -e/p' "$0" | sed '$d'
  echo; echo "site settings (tools/site.sh):"; tools/site.sh | sed 's/^/  /'
  echo; echo "would run:"; DRY=1
fi
DF="FE_SPEC=${SPEC:-16} FE_TREE=${TREE:-1} FE_DF_BLOCK=${BLOCK:-16} FE_COPY_ROWS=${COPY:-47}"
case $DRAFT in
  dflash) ENVS="FE_MTP=0 FE_DRAFT=dflash $DF" ;;
  auto) ENVS="FE_MTP=1 FE_DRAFT=dflash $DF" ;;
  mtp) ENVS="FE_MTP=1 FE_SPEC=${SPEC:-8}${COPY:+ FE_COPY_ROWS=$COPY}" ;;
  off) ENVS="FE_MTP=0 FE_SPEC=0 FE_COPY_ROWS=0" ;;
  *) fl_die "FERRITE_DRAFT is dflash, mtp, auto or off, not '$DRAFT'" ;;
esac
case $DRAFT in dflash|auto) [ -f "$FERRITE_DRAFT_DIR/config.json" ] || fl_die "FERRITE_DRAFT=$DRAFT needs the DFlash2 drafter in $FERRITE_DRAFT_DIR on both nodes (scripts/fetch-models.sh draft)" ;; esac
[ -f "$M/config.json" ] || fl_die "no model at $M: run scripts/fetch-models.sh on both nodes (FERRITE_MODELS=$FERRITE_MODELS)"
# prompt rows: NVFP4/FP8 when that checkpoint is there (the fast prefill), else bf16; PROMPT overrides
PROMPT=${FERRITE_PROMPT:-}
if [ -z "$PROMPT" ]; then
  if [ -f "$FERRITE_NVFP4_DIR/config.json" ]; then PROMPT=fp4; else PROMPT=bf16
    fl_warn "no NVFP4 checkpoint at $FERRITE_NVFP4_DIR (FERRITE_NVFP4_DIR): prompt rows in bf16, a slower prefill; PROMPT=fp4 needs it on both nodes"
  fi
fi
case $PROMPT in fp4) ENVS="$ENVS FE_PROMPT=fp4" ;; bf16) ;; *) fl_die "FERRITE_PROMPT is fp4 or bf16, not '$PROMPT'" ;; esac
CACHEV=${FERRITE_CACHE:-1}
case $CACHEV in 0|1) ;; *) fl_die "FERRITE_CACHE is 1 or 0, not '$CACHEV'" ;; esac
ENVS="$ENVS FE_CACHE=$CACHEV"   # prompt cache (turn snapshots, parking, shared prefix); 0 = off
[ -z "${FERRITE_PARK_GB:-}" ] || ENVS="$ENVS FE_PARK_GB=$FERRITE_PARK_GB"   # the cache pool a node (default 24 GB)
if [ $DRY = 0 ]; then
  fl_net
  fl_say "rank 0 here, rank 1 on ${FERRITE_PEER:-<FERRITE_PEER unset>}; drafting $DRAFT, prompt weights $PROMPT, ctx $CTXV x $SLOTS slots, prompt cache $([ "$CACHEV" = 1 ] && echo on || echo off)"
  fl_banner "${FERRITE_MODEL_ID:-qwen3.8-27b}" "two nodes, loading takes about a minute"
else
  FERRITE_HOST=${FERRITE_HOST:-0.0.0.0}; FERRITE_PORT=${FERRITE_PORT:-8000}
fi
[ "${FERRITE_LOCK_WAIT:-0}" = 1 ] || export TP2_NOWAIT=1
TP2_DRY=$DRY TP2_TIMEOUT=0 TP_ENV="$ENVS ${TP_ENV:-}" FOLLOW=$M exec tools/tp2.sh $BIN --engine ferrite \
  --model-dir $M --weights-dir $M --slots "$SLOTS" --max-model-len "$CTXV" --prefill-chunk 2048 \
  --host "$FERRITE_HOST" --port "$FERRITE_PORT" --served-model-name "${FERRITE_MODEL_ID:-qwen3.8-27b}"
