#!/usr/bin/env bash
# Release gate: start a launcher on its target machine and check, from THIS machine (another one on the network),
# that the server answers like a client would see it. Run it once per target before a release.
#
#   tools/release-gate.sh TARGET SERVER [-- extra launcher env, e.g. FERRITE_PEER=nodeb CUDA_VISIBLE_DEVICES=1]
#
# TARGET   spark | tp2 | rtx | fnext | b70 (scripts/ferritectl.sh's targets)
# SERVER   the inference machine's ssh host; the Ferrite folder there is GATE_DIR [~/ferrite], the address this
#          machine uses for it is GATE_ADDR [the first address of `hostname -I` there], the port FERRITE_PORT [8000]
# Steps, each printed PASS or FAIL:
#   1. ferritectl.sh start TARGET on SERVER (it waits for /health); the launcher's defaults, so it listens on 0.0.0.0
#   2. from here: GET /health (status ok, the engine's name), GET /v1/models (one model, its id and context)
#   3. from here: a greedy chat request (thinking off) that must return text, the same request streamed (text and
#      [DONE]), and a burst of 4 at once mixing sampled (temperature 1) and greedy requests, after which /health
#      must still be ok and a greedy request must still answer; then tools/bench/mixed.py repro (greedy edits with
#      sampled requests beside them) when it is next to this script (GATE_MIXED=0 skips it)
#   4. GATE_LOCAL=1 also restarts it with FERRITE_HOST=127.0.0.1 and checks that this machine can NOT reach it
#   5. ferritectl.sh stop on SERVER, and the port is closed again
# Exit status 0 when every step passed. The summary line is "release-gate: TARGET on SERVER: PASS (n checks)".
# When this machine cannot ssh to the server, start the server there yourself (scripts/ferritectl.sh start TARGET)
# and give its URL instead: `tools/release-gate.sh TARGET http://ADDR:PORT` runs steps 2 and 3 only.
set -uo pipefail

[ $# -ge 2 ] || { sed -n '2,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 2; }
target=$1; server=$2; shift 2
[ "${1:-}" = -- ] && shift
extra=("$@")
dir=${GATE_DIR:-'~/ferrite'}
port=${FERRITE_PORT:-8000}
check_only=0
case $server in
  http://*|https://*) check_only=1; base=${server%/}; base=${base%/v1} ;;
  *) addr=${GATE_ADDR:-$(ssh -n "$server" "ip -4 -o route get 1.1.1.1 | sed -n 's/.* src \\([0-9.]*\\).*/\\1/p'")}
     [ -n "$addr" ] || { echo "release-gate: no address for $server (set GATE_ADDR)"; exit 2; }
     base=http://$addr:$port ;;
esac
pass=0; fail=0
ok() { echo "release-gate: PASS  $*"; pass=$((pass + 1)); }
bad() { echo "release-gate: FAIL  $*"; fail=$((fail + 1)); }
remote() { ssh -n "$server" "cd $dir && env FERRITE_PORT=$port ${extra[*]:-} $*"; }
json() { python3 -c "import json, sys; d = json.load(sys.stdin); print($1)" 2> /dev/null; }

echo "release-gate: $target on $server ($base), from $(hostname -s)"
# 1. start
if [ $check_only = 1 ]; then :
elif remote "scripts/ferritectl.sh start $target" 2>&1 | sed 's/^/  | /'; [ "${PIPESTATUS[0]}" = 0 ]; then ok "started ($target)"
else bad "the server did not start"; remote "tail -n 30 logs/server-$port.log" | sed 's/^/  | /'; echo "release-gate: $target on $server: FAIL"; exit 1
fi

check_api() {
  # 2. health and models, from here
  h=$(curl -s -m 10 "$base/health")
  st=$(printf '%s' "$h" | json 'd["status"]'); eng=$(printf '%s' "$h" | json 'd["engine"]')
  [ "$st" = ok ] && ok "/health from $(hostname -s): engine $eng, model $(printf '%s' "$h" | json 'd["model"]'), $(printf '%s' "$h" | json 'd["max_batch_size"]') at once" \
    || bad "/health from $(hostname -s): '$h'"
  m=$(curl -s -m 10 "$base/v1/models")
  id=$(printf '%s' "$m" | json 'd["data"][0]["id"]')
  [ -n "$id" ] && ok "/v1/models: $id, context $(printf '%s' "$m" | json 'd["data"][0].get("context_length")')" || bad "/v1/models: '$m'"
  # 3. requests
  body=$(python3 -c 'import json, sys; print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}], "max_tokens": 120, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}))' "$id")
  r=$(curl -s -m 300 "$base/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
  t=$(printf '%s' "$r" | json 'd["choices"][0]["message"]["content"]')
  [ -n "$t" ] && ok "chat: ${#t} chars, $(printf '%s' "$r" | json 'd["usage"]["completion_tokens"]') tokens" || bad "chat: '$(printf '%s' "$r" | head -c 300)'"
  sb=$(printf '%s' "$body" | python3 -c 'import json, sys; d = json.load(sys.stdin); d["stream"] = True; print(json.dumps(d))')
  s=$(curl -s -N -m 300 "$base/v1/chat/completions" -H 'Content-Type: application/json' -d "$sb")
  sc=$(printf '%s\n' "$s" | grep -c '"content"'); sd=$(printf '%s\n' "$s" | grep -c '\[DONE\]')
  [ "$sc" -gt 0 ] && [ "$sd" = 1 ] && ok "streamed chat: $sc content events, [DONE]" || bad "streamed chat: $sc content events, $sd [DONE]"
  pids=(); out=$(mktemp -d)
  for i in 0 1 2 3; do
    temp=$([ $((i % 2)) = 0 ] && echo 1.0 || echo 0)
    b=$(python3 -c 'import json, sys; print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "Write a short story about a lighthouse keeper." if sys.argv[2] != "0" else "Write a Rust function that merges two sorted vectors."}], "max_tokens": 200, "temperature": float(sys.argv[2]), "chat_template_kwargs": {"enable_thinking": False}}))' "$id" "$temp")
    curl -s -m 600 -o "$out/$i.json" -w '%{http_code}' "$base/v1/chat/completions" -H 'Content-Type: application/json' -d "$b" > "$out/$i.code" &
    pids+=($!)
  done
  wait "${pids[@]}"
  good=0
  for i in 0 1 2 3; do
    [ "$(cat "$out/$i.code")" = 200 ] && [ -n "$(json 'd["choices"][0]["message"]["content"]' < "$out/$i.json")" ] && good=$((good + 1))
  done
  rm -rf "$out"
  [ $good = 4 ] && ok "4 at once (2 sampled, 2 greedy): 4 answered" || bad "4 at once (2 sampled, 2 greedy): $good of 4 answered"
  h=$(curl -s -m 10 "$base/health"); r=$(curl -s -m 300 "$base/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
  [ "$(printf '%s' "$h" | json 'd["status"]')" = ok ] && [ -n "$(printf '%s' "$r" | json 'd["choices"][0]["message"]["content"]')" ] \
    && ok "after the burst: /health ok and a request answers" || bad "after the burst: health '$h', reply '$(printf '%s' "$r" | head -c 200)'"
  # the mixed-load repro (greedy edits with sampled requests beside them), when the bench clients (tools/bench with its
  # data/ folder) are next to this script
  mx=$(dirname "$0")/bench/mixed.py
  if [ "${GATE_MIXED:-1}" = 1 ] && [ -f "$mx" ]; then
    if python3 "$mx" --url "$base/v1" repro > /tmp/release-gate-mixed.$$ 2>&1; then ok "mixed sampled + greedy load (bench/mixed.py repro)"
    else bad "mixed sampled + greedy load: $(tail -n 3 /tmp/release-gate-mixed.$$ | tr '\n' ' ')"; fi
    rm -f /tmp/release-gate-mixed.$$
  fi
}
check_api
if [ $check_only = 1 ]; then
  echo "release-gate: $target at $base: $([ $fail = 0 ] && echo PASS || echo FAIL) ($pass passed, $fail failed)"
  [ $fail = 0 ]; exit
fi

# 4. optional: FERRITE_HOST=127.0.0.1 is not reachable from here
if [ "${GATE_LOCAL:-0}" = 1 ]; then
  remote "scripts/ferritectl.sh stop" > /dev/null
  extra+=(FERRITE_HOST=127.0.0.1)
  if remote "scripts/ferritectl.sh start $target" > /dev/null 2>&1; then
    curl -s -m 5 -o /dev/null "$base/health" && bad "FERRITE_HOST=127.0.0.1: still reachable from $(hostname -s)" \
      || ok "FERRITE_HOST=127.0.0.1: not reachable from $(hostname -s)"
    remote "curl -sf -m 5 http://127.0.0.1:$port/health" > /dev/null && ok "FERRITE_HOST=127.0.0.1: reachable on $server itself" \
      || bad "FERRITE_HOST=127.0.0.1: not reachable on $server itself"
  else
    bad "restart with FERRITE_HOST=127.0.0.1 failed"
  fi
fi

# 5. stop
remote "scripts/ferritectl.sh stop" | sed 's/^/  | /'
sleep 2
curl -s -m 5 -o /dev/null "$base/health" && bad "the port still answers after stop" || ok "stopped; port $port closed"
echo "release-gate: $target on $server: $([ $fail = 0 ] && echo PASS || echo FAIL) ($pass passed, $fail failed)"
[ $fail = 0 ]
