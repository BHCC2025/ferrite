#!/usr/bin/env python3
"""Generates crates/fe-server/tests/data/inkling_template_cases.json: conversations rendered by the model's own chat template through HF
apply_chat_template (Python jinja2). The Rust test renders the same inputs with the server's minijinja renderer and compares byte for byte.
Needs `transformers` and the model dir (tokenizer + chat_template.jinja); test tooling only, the server does not use Python."""
import json, sys
from transformers import AutoTokenizer
MODEL = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: make_inkling_template_cases.py MODEL_DIR [OUT.json]")
OUT = sys.argv[2] if len(sys.argv) > 2 else "crates/fe-server/tests/data/inkling_template_cases.json"
tok = AutoTokenizer.from_pretrained(MODEL)
U = lambda s: {"role": "user", "content": s}
tools = [
    {"type": "function", "function": {"name": "write_file", "description": "Write a file.", "parameters": {"type": "object",
        "properties": {"path": {"type": "string", "description": "where"}, "content": {"type": "string"}, "mode": {"type": "string", "enum": ["w", "a"]}},
        "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search é 日本 files.", "parameters": {"type": "object",
        "properties": {"q": {"type": "string"}, "opts": {"type": "object", "properties": {"zeta": {"type": "integer"}, "alpha": {"type": "number", "default": 1.5}}}}}}},
    {"type": "function", "function": {"name": "noargs"}},
]
tc = lambda i, n, a: {"id": i, "type": "function", "function": {"name": n, "arguments": a}}
cases = []
def add(name, messages, tools=None, **kw):
    kwargs = dict(kw)
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    cases.append({"name": name, "messages": messages, "tools": tools, "kwargs": kwargs, "expected": text, "ids": ids})
add("plain", [U("Hello there.")])
add("system", [{"role": "system", "content": "Be brief."}, U("Hi")])
add("system_only_then_user_effort", [{"role": "system", "content": "S1"}, {"role": "system", "content": "S2"}, U("x")])
for e in ("none", "minimal", "low", "medium", "high", "xhigh", "max", 0, 0.0, 0.5, 0.7, 0.99, 0.123):
    add("effort_%s" % e, [U("think about it")], reasoning_effort=e)
add("unicode", [U("naïve café 日本語 🦀 \"quotes\" \\ back\nnewline")])
add("content_parts", [{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}])
add("tools_only", [U("find it")], tools=tools)
add("tools_system_effort", [{"role": "system", "content": "You are an agent."}, U("find it")], tools=tools, reasoning_effort="low")
add("assistant_plain", [U("a"), {"role": "assistant", "content": "b"}, U("c")])
add("assistant_reasoning", [U("a"), {"role": "assistant", "content": "answer", "reasoning_content": "because"}, U("c")])
add("assistant_reasoning_only", [U("a"), {"role": "assistant", "content": "", "reasoning_content": "thought"}, U("c")])
add("tool_loop", [U("save it"),
    {"role": "assistant", "content": "", "tool_calls": [tc("call_1", "write_file", {"path": "a.txt", "content": "hi \"there\"\né", "mode": "w"})]},
    {"role": "tool", "tool_call_id": "call_1", "content": "ok, 2 bytes"},
    {"role": "assistant", "content": "Saved."}, U("thanks")], tools=tools)
add("tool_parallel_nested", [U("do both"),
    {"role": "assistant", "content": "On it.", "reasoning_content": "plan", "tool_calls": [
        tc("c1", "search", {"q": "x", "opts": {"zeta": 1, "alpha": 2.5}}), tc("c2", "write_file", {"mode": "a", "content": "z", "path": "p"}), tc("c3", "noargs", {})]},
    {"role": "tool", "tool_call_id": "c1", "content": "r1"},
    {"role": "tool", "tool_call_id": "c2", "name": "write_file", "content": "r2"},
    {"role": "tool", "tool_call_id": "c3", "content": [{"type": "text", "text": "r3 part"}]}], tools=tools)
add("tool_call_null_content", [U("go"), {"role": "assistant", "content": None, "tool_calls": [tc("k", "noargs", {})]}, {"role": "tool", "tool_call_id": "k", "content": "done"}], tools=tools)
add("assistant_name", [U("go"), {"role": "assistant", "name": "agent_a", "content": "hello"}, U("next")])
add("unsorted_nested_args", [U("go"), {"role": "assistant", "content": "", "tool_calls": [tc("k", "search", {"opts": {"zeta": [3, 1, {"b": 1, "a": 2}], "alpha": None, "t": True}, "q": "é"})]},
    {"role": "tool", "tool_call_id": "k", "content": "found"}], tools=tools)
json.dump(cases, open(OUT, "w"), indent=1, ensure_ascii=False)
print(len(cases), "cases ->", OUT)
