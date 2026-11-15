#!/bin/bash
# tools/serve.sh: scripts/serve.sh with the short setting names of the tools/ scripts (SLOTS, CTX, PORT, PROMPT,
# DRAFT, CACHE, PARK_GB, CHUNK, SPEC, COPY, TREE, BLOCK are FERRITE_SLOTS, FERRITE_CTX, ...). Everything else,
# including the defaults (FERRITE_HOST 0.0.0.0, port 8000, model id qwen3.8-27b) and the GPU lock, is scripts/serve.sh's.
# usage: tools/serve.sh [--help] [extra fe-server arguments]
set -euo pipefail
for p in SLOTS CTX PORT PROMPT DRAFT CACHE PARK_GB CHUNK SPEC COPY TREE BLOCK; do
  if [ -n "${!p:-}" ]; then export "FERRITE_$p=${!p}"; fi
done
exec "$(dirname "$0")/../scripts/serve.sh" "$@"
