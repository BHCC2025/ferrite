#!/usr/bin/env bash
# Drive a running fe-server (--mock) with the real Pi coding agent, in a throwaway agent dir + work dir
# (the user's ~/.pi config is not touched). Checks a plain reply and a read-tool round trip.
# usage: tools/e2e_pi.sh [base_url]     (default $FERRITE_URL, else http://localhost:8000/v1)
set -u
URL=${1:-${FERRITE_URL:-http://localhost:8000/v1}}
T=$(mktemp -d)
mkdir -p "$T/agent" "$T/work"
cat > "$T/agent/models.json" <<EOF
{"providers": {"ferrite-test": {"baseUrl": "$URL", "api": "openai-completions", "apiKey": "none",
  "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": false, "supportsFinishReason": false,
             "thinkingFormat": "qwen-chat-template"},
  "models": [{"id": "qwen3.8-27b", "name": "ferrite mock", "reasoning": true, "input": ["text"], "contextWindow": 131072,
              "maxTokens": 16384, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}}
EOF
echo "hello file" > "$T/work/notes.txt"
run() { # run pi with a 60 s deadline; output to $T/out
  (cd "$T/work" && PI_CODING_AGENT_DIR="$T/agent" pi -p --no-session --no-extensions --no-skills --no-context-files \
     --no-mcp --no-prompt-templates --offline --model ferrite-test/spark --thinking high --tools read "$@" \
     < /dev/null > "$T/out" 2>&1) & local p=$!
  for _ in $(seq 1 60); do kill -0 $p 2>/dev/null || break; sleep 1; done
  kill $p 2>/dev/null; wait $p
}
fail=0
run "Hello there, who are you?"
if grep -q "Hello from the ferrite mock engine" "$T/out"; then echo "ok   pi: plain reply"; else echo "FAIL pi: plain reply"; cat "$T/out"; fail=1; fi
run --mode json "Please read the file notes.txt and tell me what it says."
python3 - "$T/out" <<'PY' || fail=1
import json, sys
ev = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
calls = [e for e in ev if e.get("type") == "tool_execution_end"]
assert calls and calls[0]["toolName"] == "read" and not calls[0]["isError"], "no successful read tool call"
assert "hello file" in json.dumps(calls[0]["result"]), calls[0]
final = [e for e in ev if e.get("type") == "message_end" and e["message"].get("role") == "assistant"][-1]["message"]
kinds = [c["type"] for c in final["content"]]
text = "".join(c.get("text", "") for c in final["content"])
assert "thinking" in kinds and text == "Based on the tool result: hello file", final
print("ok   pi: thinking + read tool call + tool result round trip")
PY
rm -rf "$T"
[ $fail = 0 ] && echo "all pi checks passed against $URL" || { echo "pi checks FAILED"; exit 1; }
