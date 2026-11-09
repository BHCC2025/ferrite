#!/usr/bin/env python3
"""End-to-end checks of a running fe-server (mock engine) with the official `openai` client.

usage: e2e_openai.py [base_url]     (default $FERRITE_URL, else http://localhost:8000/v1)
Exits non-zero on the first failed check. Needs the `openai` package.
"""
import json
import os
import sys
import threading
import time

from openai import OpenAI

BASE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FERRITE_URL", "http://localhost:8000/v1")
client = OpenAI(base_url=BASE, api_key="none")
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}},
            "required": ["location", "days", "metric"]}}},
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
passed = 0


def check(cond, what):
    global passed
    if not cond:
        print(f"FAIL: {what}")
        sys.exit(1)
    passed += 1
    print(f"ok   {what}")


def reasoning_of(obj):
    extra = getattr(obj, "model_extra", None) or {}
    return extra.get("reasoning_content") or getattr(obj, "reasoning_content", None)


# models
models = client.models.list()
check([m.id for m in models.data] == ["spark"], "models.list -> ['spark']")

# non-stream chat with reasoning
r = client.chat.completions.create(model="anything", messages=[{"role": "user", "content": "Hello there"}])
m = r.choices[0].message
check(r.model == "spark" and r.object == "chat.completion", "chat: model id spark, object chat.completion")
check(m.content.startswith("Hello from the ferrite mock engine!"), "chat: content")
check((reasoning_of(m) or "").startswith("The user wrote"), "chat: reasoning_content separated")
check(r.choices[0].finish_reason == "stop", "chat: finish_reason stop")
check(r.usage.prompt_tokens > 0 and r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens,
      "chat: usage adds up")

# streaming chat with usage
stream = client.chat.completions.create(model="spark", messages=[{"role": "user", "content": "Hello there"}],
                                        stream=True, stream_options={"include_usage": True})
content, reasoning, finish, usage, n = "", "", None, None, 0
for chunk in stream:
    n += 1
    if chunk.usage:
        usage = chunk.usage
    for ch in chunk.choices:
        content += ch.delta.content or ""
        reasoning += reasoning_of(ch.delta) or ""
        finish = ch.finish_reason or finish
check(content == m.content, "stream: content equals non-stream content")
check(reasoning == reasoning_of(m), "stream: reasoning equals non-stream reasoning")
check(finish == "stop" and usage is not None and usage.completion_tokens == r.usage.completion_tokens,
      f"stream: finish stop + usage chunk ({n} chunks)")

# thinking off via extra_body
r = client.chat.completions.create(model="spark", messages=[{"role": "user", "content": "Hi"}],
                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
check(reasoning_of(r.choices[0].message) is None, "enable_thinking=false: no reasoning_content")

# tool calls, non-stream
r = client.chat.completions.create(model="spark", tools=TOOLS,
                                   messages=[{"role": "user", "content": "What is the weather in Oslo? Also web_search it."}])
c = r.choices[0]
check(c.finish_reason == "tool_calls", "tools: finish_reason tool_calls")
calls = c.message.tool_calls
check([t.function.name for t in calls] == ["get_weather", "web_search"], "tools: two parallel calls")
args = json.loads(calls[0].function.arguments)
check(args == {"location": "Oslo", "days": 3, "metric": True}, f"tools: typed arguments {args}")
check(all(t.id.startswith("call_") and t.type == "function" for t in calls), "tools: ids + type")

# feed the results back
msgs = [{"role": "user", "content": "What is the weather in Oslo? Also web_search it."},
        c.message.model_dump(exclude_none=True)]
for t, out in zip(calls, ["-3 C, snow", "Oslo is the capital of Norway"]):
    msgs.append({"role": "tool", "tool_call_id": t.id, "content": out})
r = client.chat.completions.create(model="spark", tools=TOOLS, messages=msgs)
check(r.choices[0].message.content == "Based on the tool result: -3 C, snow", "tools: answer after tool results")

# tool calls, stream
stream = client.chat.completions.create(model="spark", tools=TOOLS, stream=True,
                                        messages=[{"role": "user", "content": "weather in Rome please"}])
acc, text, finish = {}, "", None
for chunk in stream:
    for ch in chunk.choices:
        text += ch.delta.content or ""
        for tc in ch.delta.tool_calls or []:
            slot = acc.setdefault(tc.index, {"id": None, "name": None, "args": ""})
            slot["id"] = tc.id or slot["id"]
            if tc.function:
                slot["name"] = tc.function.name or slot["name"]
                slot["args"] += tc.function.arguments or ""
        finish = ch.finish_reason or finish
check(finish == "tool_calls" and len(acc) == 1, "stream tools: one call, finish tool_calls")
check(acc[0]["name"] == "get_weather" and json.loads(acc[0]["args"])["location"] == "Rome",
      f"stream tools: {acc[0]}")
check(text == "I'll use get_weather for this.", f"stream tools: content before the call {text!r}")

# tool_choice none
r = client.chat.completions.create(model="spark", tools=TOOLS, tool_choice="none",
                                   messages=[{"role": "user", "content": "weather in Rome please"}])
check(not r.choices[0].message.tool_calls, "tool_choice none: no tool calls")

# stop strings + max_tokens
r = client.chat.completions.create(model="spark", messages=[{"role": "user", "content": "Hi"}], stop=["Zürich"],
                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
check(r.choices[0].message.content.endswith("naïve, ") and r.choices[0].finish_reason == "stop", "stop string")
r = client.chat.completions.create(model="spark", messages=[{"role": "user", "content": "Hi"}], max_tokens=4)
check(r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 4, "max_tokens -> length")

# completions
r = client.completions.create(model="spark", prompt="Once upon a time", max_tokens=64)
check(r.object == "text_completion" and r.choices[0].text.startswith(" and so the mock engine"), "completions")
text = "".join(c.choices[0].text for c in client.completions.create(model="spark", prompt="Once upon a time",
                                                                      max_tokens=64, stream=True) if c.choices)
check(text == r.choices[0].text, "completions stream == non-stream")

# 4 concurrent streams overlap; a 5th queues
spans = [None] * 5


def one(i):
    t0 = time.time()
    s = client.chat.completions.create(model="spark", max_tokens=200, stream=True,
                                       messages=[{"role": "user", "content": f"write something long {i}"}],
                                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    first, n = None, 0
    for chunk in s:
        if chunk.choices and chunk.choices[0].delta.content:
            first = first or time.time()
            n += 1
    spans[i] = (t0, first, time.time(), n)


th = [threading.Thread(target=one, args=(i,)) for i in range(5)]
for t in th:
    t.start()
for t in th:
    t.join()
firsts = sorted(s[1] - s[0] for s in spans)
check(all(s[3] > 0 for s in spans), "5 concurrent streams all produced content")
check(firsts[3] < firsts[4] / 2, f"4 run at once, the 5th waited (first-token s: {[round(f, 2) for f in firsts]})")
print(f"\nall {passed} checks passed against {BASE}")
