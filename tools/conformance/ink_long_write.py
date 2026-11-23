#!/usr/bin/env python3
"""One long write call through the Inkling server (what long_write_1k wants when the model does not explore first): streamed, valid JSON, closed file.
usage: ink_long_write.py http://127.0.0.1:8000 [lines]"""
import json, sys, time, urllib.request
BASE = sys.argv[1].rstrip("/"); LINES = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TOOLS = [{"type": "function", "function": {"name": "write_file", "description": "Write a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
         {"type": "function", "function": {"name": "bash", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
body = {"model": "inkling-2-mini", "temperature": 0, "max_tokens": 40000, "stream": True, "stream_options": {"include_usage": True}, "tools": TOOLS, "reasoning_effort": "low",
        "messages": [{"role": "system", "content": "You are a coding agent. Do not explore first: the directory is empty. Save the file with ONE write_file call containing the complete file."},
                     {"role": "user", "content": "Create index.html: a complete self-contained landing page for a bakery with an embedded stylesheet and script, about %d lines. Save it with write_file." % LINES}]}
t0 = time.time(); args = ""; deltas = 0; name = ""; fin = None; usage = None; reasoning = 0; gaps = []; last = t0
resp = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=3600)
for line in resp:
    line = line.decode().strip()
    if not line.startswith("data: ") or line == "data: [DONE]": continue
    d = json.loads(line[6:]); now = time.time(); gaps.append(now - last); last = now
    if d.get("usage"): usage = d["usage"]
    for ch in d.get("choices") or []:
        dl = ch.get("delta") or {}
        reasoning += len(dl.get("reasoning_content") or "")
        for t in dl.get("tool_calls") or []:
            f = t.get("function") or {}; name += f.get("name") or ""; args += f.get("arguments") or ""; deltas += 1
        if ch.get("finish_reason"): fin = ch["finish_reason"]
a = json.loads(args) if args else {}
c = a.get("content", "")
print("finish %s, call %r, %d arg deltas, %.0fs, longest gap %.1fs, usage %s" % (fin, name, deltas, time.time() - t0, max(gaps), usage))
print("file: %d lines, %d chars, closed with </html>: %s" % (len(c.splitlines()), len(c), "</html>" in c))
ok = fin == "tool_calls" and name == "write_file" and "</html>" in c and deltas > 50 and len(c.splitlines()) >= LINES * 0.6
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
