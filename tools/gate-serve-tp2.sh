#!/bin/bash
# Gate for the TP2 server path that sched-test does not reach: the follower's DFlash2 half, draft rings at long
# context, the scheduler's batch rows. Starts tools/serve-tp2.sh on a test port (DRAFT=dflash, CTX 131072, SLOTS 4,
# PROMPT fp4 by default), waits for /v1/models, sends 4 chat requests at once (greedy, thinking off) and passes when
# all four return text. Then it stops the server it started; the follower on the peer exits when the link closes.
# Takes the TP2 + GPU locks through serve-tp2.sh while it runs: never start it beside a live server.
# usage: tools/gate-serve-tp2.sh     env: PORT (8019), DRAFT, CTX, SLOTS, PROMPT, WAIT (seconds to ready, 900),
#                                    and anything serve-tp2.sh takes (TP_ENV, SPEC, TREE, BLOCK)
# The failure it was written for: rank 1 sized its drafter by the engine's batch rows (8192) and died with
# "draft ring 4096 too small" right after engine ready; rank 0 then failed the first request (broken pipe).
set -u
cd "$(dirname "$0")/.."
PORT=${PORT:-8019}
. tools/site.sh
OUT=$FERRITE_STATE_DIR/gate-serve-tp2-$PORT; mkdir -p $OUT; rm -f $OUT/*.json
LOG=$OUT/server.log
DRAFT=${DRAFT:-dflash} CTX=${CTX:-131072} SLOTS=${SLOTS:-4} PROMPT=${PROMPT:-fp4} PORT=$PORT \
  setsid tools/serve-tp2.sh > $LOG 2>&1 < /dev/null &
SRV=$!
# stop = kill this gate's fe-server (by its port; TP2.md: the follower and the locks follow it), then reap the script
stop() {
  local p
  p=$(ps -eo pid=,args= | awk -v port="--port $PORT" 'index($0, "target/release/fe-server") && index($0, port) && !index($0, "awk") { print $1 }')
  [ -n "$p" ] && kill $p
  wait $SRV 2>/dev/null
}
fail() { echo "gate-serve-tp2: FAIL: $1"; tail -n 20 $LOG; stop; exit 1; }
deadline=$((SECONDS + ${WAIT:-900}))
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models)" = 200 ]; do
  kill -0 $SRV 2>/dev/null || fail "the server exited before it was ready"
  [ $SECONDS -lt $deadline ] || fail "not ready after ${WAIT:-900} s"
  sleep 5
done
echo "gate-serve-tp2: ready after $SECONDS s ($(grep -h 'engine ready' $LOG | tail -n 1))"
P=("Write a Python function that parses an ISO 8601 date string, with tests."
   "Write a Rust function that merges two sorted vectors."
   "Explain what a write-ahead log is, in three short paragraphs."
   "Write a short story about a lighthouse keeper in winter.")
CURL=()
for i in 0 1 2 3; do
  body=$(python3 -c 'import json,sys; print(json.dumps({"model": "spark", "messages": [{"role": "user", "content": sys.argv[1]}],
    "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}))' "${P[$i]}")
  curl -s --max-time 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$body" \
    > $OUT/r$i.json &
  CURL+=($!)
done
wait "${CURL[@]}" 2>/dev/null
ok=0
for i in 0 1 2 3; do
  n=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"] or ""))' \
    $OUT/r$i.json 2>/dev/null || echo 0)
  echo "gate-serve-tp2: request $i: $n chars"
  [ "$n" -gt 0 ] && ok=$((ok + 1))
done
kill -0 $SRV 2>/dev/null || fail "the server died during the requests"
grep -q -E "^Error|step failed|too small" $LOG && fail "errors in the server log"
stop
# rank 1's own log (tp2.sh names it on exit): its drafter / follower errors never reach rank 0's
r1=$(sed -n 's/^(rank 1 log: \(.*\))$/\1/p' $LOG | tail -n 1)
if [ -n "$r1" ] && ssh -n "${r1%%:*}" "grep -E '^Error|too small' ${r1#*:}"; then
  echo "gate-serve-tp2: FAIL: errors in rank 1's log ($r1)"; exit 1
fi
[ $ok = 4 ] || { echo "gate-serve-tp2: FAIL: $ok of 4 requests returned text"; tail -n 20 $LOG; exit 1; }
echo "gate-serve-tp2: PASS (4 of 4 requests answered; log $LOG)"
