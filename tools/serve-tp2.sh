#!/bin/bash
# tools/serve.sh on TP2: the OpenAI-compatible server (model id "spark", FERRITE_MODEL_ID) on :8000 as rank 0
# (this host), rank 1 on the peer replaying its engine calls (`ferrite <model> follow`). EXL3 3.0 mix decode +
# speculative decoding, prompt rows via NVFP4/FP8 (PROMPT=fp4, the default when $FERRITE_NVFP4_DIR holds a checkpoint; both
# nodes need it) or bf16 (PROMPT=bf16, the default otherwise). Holds the global TP2 lock and both nodes' GPU locks
# for its whole life (tools/tp2.sh lock order).
# Run on rank 0's node; target/release/{ferrite,fe-server} must be the same build on both nodes (tp2.sh checks).
# usage: tools/serve-tp2.sh [--help: the settings and the commands it would run]
#                                  env: SLOTS (4), CTX (131072), PROMPT (fp4|bf16), SPEC (8; 16 with DFlash2),
#                                  (SLOTS / CTX / PORT may also be given as FERRITE_SLOTS / FERRITE_CTX / FERRITE_PORT)
#                                  PORT (8000), DRAFT (dflash, the default: DFlash2 block drafter on rank 0 with verify trees | mtp | auto:
#                                  DFlash2 for <= 2 decoding streams, MTP above), TREE (1: greedy DFlash2 streams verify
#                                  best-first trees; 0 = chains), BLOCK (16: DFlash2 draft block), CACHE / FERRITE_CACHE (1: prompt cache, docs/prompt-cache.md; 0 = off; FERRITE_PARK_GB its pool), COPY (47: copy rows,
#                                  greedy DFlash2 streams verify copies of their own history too; 0 = off)
#                                  site settings (model folders, peer, fabric, locks): tools/site.sh
# SPEC 8 (MTP): in the server depth 8 beats 4 for code (113.6 vs 105.0 tok/s), prose equal; see TP2.md
set -e
if [ "${SLOTS:-${FERRITE_SLOTS:-4}}" -gt 4 ] && [ "${DRAFT:-${FERRITE_DRAFT:-dflash}}" = dflash ]; then
  echo "$(basename "$0"): SLOTS=${SLOTS:-$FERRITE_SLOTS} is not supported with DFlash2 drafting (DRAFT=dflash): one draft pass serves at most 4 streams in this build. Use SLOTS=4, or DRAFT=auto / DRAFT=mtp for more slots (MTP drafting above 2 streams; not tested above 4 slots)" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
. tools/site.sh
M=$FERRITE_EXL3_DIR
DRY=0
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
  sed -n '2,/^set -e/p' "$0" | sed '$d'
  echo; echo "site settings (tools/site.sh):"; tools/site.sh | sed 's/^/  /'
  echo; echo "would run:"; DRY=1
fi
DRAFT=${DRAFT:-${FERRITE_DRAFT:-dflash}}   # FERRITE_DRAFT as in scripts/serve.sh
ENVS="FE_MTP=1 FE_SPEC=${SPEC:-8}"
DF="FE_SPEC=${SPEC:-16} FE_TREE=${TREE:-1} FE_DF_BLOCK=${BLOCK:-16} FE_COPY_ROWS=${COPY:-47}"
[ "${DRAFT:-dflash}" = dflash ] && ENVS="FE_MTP=0 FE_DRAFT=dflash $DF"
[ "${DRAFT:-dflash}" = auto ] && ENVS="FE_MTP=1 FE_DRAFT=dflash $DF"
# prompt rows: NVFP4/FP8 when that checkpoint is there (the fast prefill), else bf16; PROMPT overrides
if [ -z "${PROMPT:-}" ]; then
  if [ -f "$FERRITE_NVFP4_DIR/config.json" ]; then PROMPT=fp4; else PROMPT=bf16
    echo "serve-tp2.sh: no NVFP4 checkpoint at $FERRITE_NVFP4_DIR (FERRITE_NVFP4_DIR): prompt rows in bf16, a slower prefill; PROMPT=fp4 needs it on both nodes" >&2
  fi
fi
[ "$PROMPT" = fp4 ] && ENVS="$ENVS FE_PROMPT=fp4"
ENVS="$ENVS FE_CACHE=${CACHE:-${FERRITE_CACHE:-1}}"   # prompt cache (turn snapshots, parking, shared prefix); CACHE=0 = off
[ -z "${FERRITE_PARK_GB:-}" ] || ENVS="$ENVS FE_PARK_GB=$FERRITE_PARK_GB"   # the cache pool a node (default 24 GB)
TP2_DRY=$DRY TP2_TIMEOUT=0 TP_ENV="$ENVS $TP_ENV" FOLLOW=$M exec tools/tp2.sh target/release/fe-server --engine ferrite \
  --model-dir $M --weights-dir $M --slots "${SLOTS:-${FERRITE_SLOTS:-4}}" \
  --max-model-len "${CTX:-${FERRITE_CTX:-131072}}" --prefill-chunk 2048 --port "${PORT:-${FERRITE_PORT:-8000}}"
