#!/usr/bin/env python3
"""Draft head on vs off (two servers of the same build, one started with INK_NODRAFT=1): greedy chat requests must come back identical
(content, reasoning, tool calls, finish_reason, completion tokens), non-streaming and streaming, and streaming must equal non-streaming.
usage: ink_onoff.py http://127.0.0.1:8000 http://127.0.0.1:8001 [model]"""
import json, sys, urllib.request

A, B = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
MODEL = sys.argv[3] if len(sys.argv) > 3 else "inkling-2-mini"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "write_file", "description": "Write a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]

def post(base, body):
    body = dict(body, model=MODEL, temperature=0)
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=1800)
    if not body.get("stream"):
        r = json.loads(resp.read()); c = r["choices"][0]; m = c["message"]
        calls = [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []]
        return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "calls": calls, "finish": c["finish_reason"], "tokens": r["usage"]["completion_tokens"], "reasoning_tokens": r["usage"]["completion_tokens_details"]["reasoning_tokens"]}
    out = {"content": "", "reasoning": "", "calls": [], "finish": None, "tokens": None, "reasoning_tokens": None}
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:])
        if d.get("usage"): out["tokens"] = d["usage"]["completion_tokens"]; out["reasoning_tokens"] = d["usage"]["completion_tokens_details"]["reasoning_tokens"]
        for ch in d.get("choices") or []:
            dl = ch.get("delta") or {}
            out["content"] += dl.get("content") or ""; out["reasoning"] += dl.get("reasoning_content") or ""
            for t in dl.get("tool_calls") or []:
                i = t["index"]
                while len(out["calls"]) <= i: out["calls"].append(["", ""])
                f = t.get("function") or {}
                if f.get("name"): out["calls"][i][0] += f["name"]
                out["calls"][i][1] += f.get("arguments") or ""
            if ch.get("finish_reason"): out["finish"] = ch["finish_reason"]
    out["calls"] = [tuple(c) for c in out["calls"]]
    return out

U = lambda s: {"role": "user", "content": s}
cases = {
 "think_math": {"messages": [U("What is 17*23? Think it through, answer briefly.")], "max_tokens": 700},
 "nothink_math": {"messages": [U("What is 17*23? Answer briefly.")], "max_tokens": 700, "chat_template_kwargs": {"enable_thinking": False}},
 "effort_low": {"messages": [U("Name three prime numbers above 100.")], "max_tokens": 700, "reasoning_effort": "low"},
 "code": {"messages": [U("Write a Python function that reverses a linked list, with a short docstring.")], "max_tokens": 1200},
 "tool_call": {"messages": [U("Weather in Paris and Tokyo in celsius, please (call the tool).")], "tools": TOOLS, "max_tokens": 1500},
 "tool_write": {"messages": [U("Create hello.py that prints hello world, using the tool.")], "tools": TOOLS, "max_tokens": 1500},
 "stop_string": {"messages": [U("Count from 1 to 20 in words, one per line.")], "max_tokens": 900, "stop": ["seven"], "chat_template_kwargs": {"enable_thinking": False}},
 "stop_in_think": {"messages": [U("Explain briefly why the sky is blue.")], "max_tokens": 600, "stop": ["Rayleigh"]},
 "length_in_think": {"messages": [U("Write a detailed plan to learn Rust in 3 months.")], "max_tokens": 120},
 "multi_stop": {"messages": [U("List the planets of the solar system, one per line.")], "max_tokens": 900, "stop": ["Mars", "Venus"], "chat_template_kwargs": {"enable_thinking": False}},
}
fails = 0
def show(n, tag, ok, extra=""):
    global fails; fails += (not ok)
    print("%-16s %-22s %s %s" % (n, tag, "PASS" if ok else "FAIL", extra), flush=True)
for n, body in cases.items():
    res = {}
    for base, lab in ((A, "A"), (B, "B")):
        for st in (False, True):
            res[(lab, st)] = post(base, dict(body, stream=st, stream_options={"include_usage": True} if st else None) if st else dict(body))
    ref = res[("A", False)]
    for k, v in res.items():
        if k == ("A", False): continue
        ok = v == ref
        show(n, "%s %s == A full" % (k[0], "stream" if k[1] else "full"), ok, "" if ok else "diff: " + ", ".join(f for f in ref if ref[f] != v[f]))
    print("   -> finish %s, %s tokens (%s in reasoning), content %d chars, reasoning %d chars, calls %d" % (ref["finish"], ref["tokens"], ref["reasoning_tokens"], len(ref["content"]), len(ref["reasoning"]), len(ref["calls"])))
# tool loop: the result of the first call goes back, the model answers
msgs = [U("Weather in Paris in celsius, please (call the tool).")]
first = post(A, {"messages": msgs, "tools": TOOLS, "max_tokens": 1500})
if first["calls"]:
    tc = [{"id": "call_%d" % i, "type": "function", "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(first["calls"])]
    msgs2 = msgs + [{"role": "assistant", "content": first["content"] or None, "reasoning_content": first["reasoning"] or None, "tool_calls": tc}] + [{"role": "tool", "tool_call_id": t["id"], "content": "18 C, light rain"} for t in tc]
    for lab, base in (("A", A), ("B", B)):
        r = post(base, {"messages": msgs2, "tools": TOOLS, "max_tokens": 1500})
        if lab == "A": refA = r
        else: show("tool_loop", "answer A == B", r == refA, "" if r == refA else "diff")
    print("   -> answer: %r" % refA["content"][:120])
else:
    show("tool_loop", "first call", False, "no call returned")
print("\n%d mismatches" % fails); sys.exit(1 if fails else 0)
