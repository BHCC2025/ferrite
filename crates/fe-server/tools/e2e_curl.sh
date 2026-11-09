#!/usr/bin/env bash
# Smoke-test a running fe-server with curl: chat / completions, stream / non-stream, tools, errors,
# client-disconnect cancellation. usage: tools/e2e_curl.sh [http://host:port]   (default localhost:8090)
set -u
B=${1:-http://localhost:8090}
fail=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; fail=1; }
post() { curl -s -m 60 "$B$1" -H 'Content-Type: application/json' -d "$2"; }
has() { if grep -qF -- "$2" <<<"$3"; then ok "$1"; else bad "$1: $(head -c 400 <<<"$3")"; fi; }

r=$(curl -s -m 5 "$B/health");                       has "/health max_batch_size" '"max_batch_size":' "$r"
r=$(curl -s -m 5 "$B/v1/models");                    has "/v1/models id spark" '"id":"spark"' "$r"

r=$(post /v1/chat/completions '{"model":"x","messages":[{"role":"user","content":"Hello!"}]}')
has "chat: reasoning_content"   '"reasoning_content":"The user wrote' "$r"
has "chat: content"             '"content":"Hello from the ferrite mock engine!' "$r"
has "chat: finish stop"         '"finish_reason":"stop"' "$r"
has "chat: usage"               '"prompt_tokens":' "$r"

r=$(post /v1/chat/completions '{"messages":[{"role":"user","content":"Hi"}],"stream":true,"stream_options":{"include_usage":true},"chat_template_kwargs":{"enable_thinking":false},"max_tokens":12}')
has "stream: role chunk"        '"delta":{"role":"assistant","content":""}' "$r"
has "stream: finish length"     '"finish_reason":"length"' "$r"
has "stream: usage chunk"       '"choices":[],"usage":{"prompt_tokens":' "$r"
has "stream: [DONE]"            'data: [DONE]' "$r"
if grep -q reasoning_content <<<"$r"; then bad "thinking off still streamed reasoning"; else ok "stream: thinking off"; fi

TOOLS='[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"},"days":{"type":"integer"}},"required":["location","days"]}}}]'
r=$(post /v1/chat/completions '{"messages":[{"role":"user","content":"weather in Paris?"}],"tools":'"$TOOLS"'}')
has "tools: tool_calls"         '"tool_calls":[{"id":"call_' "$r"
has "tools: typed arguments"    '"arguments":"{\"location\": \"Paris\", \"days\": 3}"' "$r"
has "tools: finish tool_calls"  '"finish_reason":"tool_calls"' "$r"
r=$(post /v1/chat/completions '{"messages":[{"role":"user","content":"weather in Paris?"}],"stream":true,"tools":'"$TOOLS"'}')
has "stream tools: name delta"  '"function":{"name":"get_weather","arguments":""}' "$r"
has "stream tools: args delta"  '"function":{"arguments":"{\"location\": \"Paris\", \"days\": 3}"}' "$r"

r=$(post /v1/chat/completions '{"messages":[{"role":"user","content":"Hi"}],"stop":["naïve"],"chat_template_kwargs":{"enable_thinking":false}}')
has "stop string truncates"     'streaming: café, "}' "$r"

r=$(post /v1/completions '{"prompt":"The capital of France is","max_tokens":30}')
has "completions"               '"object":"text_completion"' "$r"
r=$(post /v1/completions '{"prompt":"The capital of France is","max_tokens":30,"stream":true}')
has "completions stream"        'data: [DONE]' "$r"

r=$(curl -s -m 5 -w ' HTTP%{http_code}' "$B/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[]}')
has "bad request -> 400 JSON"   'HTTP400' "$r"

# disconnect mid-stream and mid-non-stream: the generation must stop
curl -sN -m 1 "$B/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"write forever"}],"stream":true,"max_tokens":100000}' >/dev/null
curl -s -m 1 "$B/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"write forever"}],"max_tokens":100000}' >/dev/null
sleep 0.5
r=$(curl -s -m 5 "$B/health");                       has "disconnects cancelled (running 0)" '"running":0' "$r"

[ $fail = 0 ] && echo "all curl checks passed against $B" || { echo "some curl checks FAILED"; exit 1; }
