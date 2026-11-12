#!/bin/bash
# ferrite OpenAI-compatible server on :8000 (model id "qwen3.8-27b", FERRITE_MODEL_ID): EXL3 3.0 mix decode + speculative decoding,
# prompt rows via NVFP4/FP8 (PROMPT=fp4, the default when $FERRITE_NVFP4_DIR holds a checkpoint) or bf16 (PROMPT=bf16, the
# default otherwise). Holds the GPU lock for its whole life.
# usage: tools/serve.sh [--help: the settings and the command it would run]
#                                  env: SLOTS (4), CTX (131072), PROMPT (fp4|bf16), SPEC (8; 16 with DFlash2),
#                                  (SLOTS / CTX / PORT may also be given as FERRITE_SLOTS / FERRITE_CTX / FERRITE_PORT;
#                                  FERRITE_HOST sets the listen address, default 0.0.0.0)
#                                  PORT (8000), DRAFT (dflash, the default: DFlash2 block drafter with verify trees | mtp | auto: DFlash2 for
#                                  <= 2 decoding streams, MTP above), TREE (1: greedy DFlash2 streams verify best-first
#                                  trees; 0 = chains), BLOCK (16: DFlash2 draft block), COPY (47: copy rows, greedy
#                                  DFlash2 streams verify copies of their own history too; 0 = off), CACHE / FERRITE_CACHE
#                                  (1: prompt cache, docs/prompt-cache.md; 0 = off), PARK_GB / FERRITE_PARK_GB (12: its park pool on
#                                  this node; two nodes 24), CHUNK (2048: prompt rows a forward). SLOTS > 4
#                                  needs DRAFT=auto or mtp (DFlash2 serves 4 streams a pass)
#                                  site settings (model folders, GPU lock): tools/site.sh
set -e
cd "$(dirname "$0")/.."
. tools/site.sh
DRAFT=${DRAFT:-${FERRITE_DRAFT:-dflash}}   # FERRITE_DRAFT as in scripts/serve.sh
if [ "${SLOTS:-${FERRITE_SLOTS:-4}}" -gt 4 ] && [ "$DRAFT" = dflash ]; then
  echo "$(basename "$0"): SLOTS=${SLOTS:-$FERRITE_SLOTS} is not supported with DFlash2 drafting (DRAFT=dflash): one draft pass serves at most 4 streams in this build. Use SLOTS=4, or DRAFT=auto / DRAFT=mtp for more slots (MTP drafting above 2 streams; not tested above 4 slots)" >&2
  exit 2
fi
M=$FERRITE_EXL3_DIR
DRY=
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
  sed -n '2,/^set -e/p' "$0" | sed '$d'
  echo; echo "site settings (tools/site.sh):"; tools/site.sh | sed 's/^/  /'
  echo; echo "would run:"; DRY=echo
else
  free -g | head -2
fi
# prompt rows: NVFP4/FP8 when that checkpoint is there (the fast prefill), else bf16; PROMPT overrides
if [ -z "${PROMPT:-}" ]; then
  if [ -f "$FERRITE_NVFP4_DIR/config.json" ]; then PROMPT=fp4; else PROMPT=bf16
    echo "serve.sh: no NVFP4 checkpoint at $FERRITE_NVFP4_DIR (FERRITE_NVFP4_DIR): prompt rows in bf16, a slower prefill" >&2
  fi
fi
[ "$PROMPT" = fp4 ] && export FE_PROMPT=fp4
export FE_SPEC=${SPEC:-8}
[ "${DRAFT:-dflash}" = dflash ] && export FE_DRAFT=dflash FE_MTP=0
[ "${DRAFT:-dflash}" = auto ] && export FE_DRAFT=dflash FE_MTP=1
[ "${DRAFT:-dflash}" != mtp ] && export FE_SPEC=${SPEC:-16} FE_TREE=${TREE:-1} FE_DF_BLOCK=${BLOCK:-16} FE_COPY_ROWS=${COPY:-47}
# MTP drafting: copy rows only when COPY is given (the MTP chain and the copy then share a verify tree)
[ "${DRAFT:-dflash}" = mtp ] && [ -n "${COPY:-}" ] && export FE_COPY_ROWS=$COPY
export FE_CACHE=${CACHE:-${FERRITE_CACHE:-1}}   # prompt cache (turn snapshots, parking, shared prefix); CACHE=0 = off
export FE_PARK_GB=${PARK_GB:-${FERRITE_PARK_GB:-${FE_PARK_GB:-12}}}   # park pool on one node (two nodes: 24); the engine shrinks it to keep 20 GB available
[ -n "$DRY" ] && env | grep -E '^FE_' | sort | tr '\n' ' ' && echo
$DRY exec flock $FERRITE_GPU_LOCK ./target/release/fe-server --engine ferrite --model-dir $M --weights-dir $M \
  --slots "${SLOTS:-${FERRITE_SLOTS:-4}}" --max-model-len "${CTX:-${FERRITE_CTX:-131072}}" --prefill-chunk "${CHUNK:-2048}" \
  --port "${PORT:-${FERRITE_PORT:-8000}}" ${FERRITE_HOST:+--host "$FERRITE_HOST"}
