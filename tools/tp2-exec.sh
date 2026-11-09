#!/bin/bash
# tools/tp2-exec.sh <binary> <base64 of NUL-separated args>: run the binary with exactly those arguments
# (tools/tp2.sh ships the command line this way so prompts with spaces, quotes or newlines reach both ranks intact).
# a runaway rank must be killable: where sshd runs with oom_score_adj -1000 everything started over ssh inherits it,
# and then a GPU OOM takes the box down instead of the process
echo 500 > /proc/self/oom_score_adj 2>/dev/null || true
BIN=$1
mapfile -d '' -t A < <(printf '%s' "$2" | base64 -d)
exec "$BIN" "${A[@]}"
