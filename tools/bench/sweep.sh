#!/bin/bash
# The published benchmark sweep (docs/benchmarks.md, docs/assets/ferrite-bench.png) in one go: starts the TP2 server
# the way tools/serve-tp2.sh does by default (DRAFT=dflash, PROMPT=fp4, 4 slots, 128K context, COPY=47) on a test port,
# waits until it is ready, runs tools/bench/sweep.py against it (prompt processing 2K-120K, generation at 2K and 32K,
# code edits; medians of 3; ~20 min) and stops the server again. Run it on rank 0's node with the GPUs free.
# usage: tools/bench/sweep.sh [OUT_DIR]     (default $FERRITE_STATE_DIR/sweep-<date-time>)
# env:   FERRITE_DIR  the build to serve: a checkout with target/release/{ferrite,fe-server} on both nodes and
#                     tools/serve-tp2.sh (default: this repository)
#        PORT (8050), FE_TP_PORT (29890), WAIT (seconds to ready, 1200), QUICK=1 (sweep.py --quick, a smoke test),
#        URL (measure a server that is already running instead; nothing is started or stopped),
#        META ("key=value;key=value" for the summaries, e.g. the final gate's KL:
#              META="kl_code=0.0207;kl_text=0.0401;kl_long=0.024")
# Output: OUT_DIR/raw.jsonl, summary.json (its "headline": the quoted figures), bench-data.json, sweep.log, server.log,
# run-info.txt (commit / binary / host: not published). Then, on a machine with the fonts
# (tools/gfx/README.md):  python3 tools/gfx/bench.py OUT_DIR/bench-data.json > ferrite-bench.svg &&
#                         rsvg-convert -w 2560 ferrite-bench.svg -o docs/assets/ferrite-bench.png
#        and keep the run: OUT_DIR/{raw.jsonl,summary.json} -> docs/bench-data/, bench-data.json -> tools/gfx/
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
FD=$(cd "${FERRITE_DIR:-$repo}" && pwd)
. "$repo/tools/site.sh"
OUT=${1:-$FERRITE_STATE_DIR/sweep-$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
PORT=${PORT:-8050}
commit=$(git -C "$FD" describe --always --dirty 2>/dev/null || echo unknown)
# the build's ids stay in run-info.txt (for the record, not published); the summaries get the date and driver only
{ echo "commit $commit"; echo "fe-server sha256 $(sha256sum < "$FD/target/release/fe-server" 2>/dev/null | cut -c1-64)"
  echo "host $(hostname)"; echo "started $(date '+%Y-%m-%d %H:%M %Z')"; } > "$OUT/run-info.txt"
meta=(date="$(date +'%b %Y')" driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)")
IFS=';' read -r -a extra <<< "${META:-}"
for kv in "${extra[@]}"; do [ -n "$kv" ] && meta+=("$kv"); done
margs=(); for kv in "${meta[@]}"; do margs+=(--meta "$kv"); done
quick=(); [ "${QUICK:-0}" = 1 ] && quick=(--quick)

if [ -n "${URL:-}" ]; then
  python3 "$here/sweep.py" --url "$URL" --out "$OUT" "${quick[@]}" "${margs[@]}" 2>&1 | tee "$OUT/sweep.log"
  exit "${PIPESTATUS[0]}"
fi

LOG=$OUT/server.log
echo "sweep.sh: serving $FD ($commit) on :$PORT, results in $OUT"
( cd "$FD" && DRAFT=dflash PROMPT=fp4 SLOTS=4 CTX=131072 COPY=47 PORT=$PORT FE_TP_PORT=${FE_TP_PORT:-29890} \
    setsid tools/serve-tp2.sh > "$LOG" 2>&1 < /dev/null ) &
SRV=$!
stop() { # this sweep's fe-server (by its port); the follower on the peer exits when the link closes
  local p
  p=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | head -n 1 | cut -d= -f2)
  [ -n "$p" ] && kill "$p" && timeout 120 bash -c "while kill -0 $p 2>/dev/null; do sleep 1; done"
  wait $SRV 2>/dev/null
  # a follower whose rank 0 died before the link came up waits for it forever (holding the peer's GPU lock):
  # stop any follower of this build still left on the peer (one TP2 job at a time, so it is this one's)
  [ -n "${FERRITE_PEER:-}" ] || return 0
  sleep 10
  ssh -n "$FERRITE_PEER" "ps -eo pid=,pgid=,args= | awk -v b='$FD/target/release/ferrite' \
    'index(\$0, b) && index(\$0, \" follow\") && !index(\$0, \"awk\") { print \$2 }' | sort -u | while read -r g; do
      echo \"sweep.sh: stopping a leftover follower on \$(hostname) (process group \$g)\"; kill -- -\$g; done"
}
trap stop EXIT
deadline=$((SECONDS + ${WAIT:-1200}))
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 $SRV 2>/dev/null || { echo "sweep.sh: the server exited before it was ready"; tail -n 20 "$LOG"; exit 1; }
  [ $SECONDS -lt $deadline ] || { echo "sweep.sh: server not ready after ${WAIT:-1200} s"; tail -n 20 "$LOG"; exit 1; }
  sleep 5
done
echo "sweep.sh: server ready after $SECONDS s ($(grep -h 'engine ready' "$LOG" | tail -n 1))"
python3 "$here/sweep.py" --url "http://127.0.0.1:$PORT/v1" --out "$OUT" "${quick[@]}" "${margs[@]}" 2>&1 | tee "$OUT/sweep.log"
rc=${PIPESTATUS[0]}
grep -q -E "^Error|step failed|panicked" "$LOG" && { echo "sweep.sh: errors in the server log ($LOG)"; rc=1; }
echo "sweep.sh: done (rc $rc). Chart: python3 tools/gfx/bench.py $OUT/bench-data.json > ferrite-bench.svg"
exit $rc
