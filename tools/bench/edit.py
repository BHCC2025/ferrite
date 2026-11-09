#!/usr/bin/env python3
"""Edit-style (agent) bench: a stdlib module in the prompt, an instruction that keeps most of it, the whole updated
module back (greedy). Streams start together; per stream tok/s, and each reply saved as <out>-<i>.txt (identity
across builds). usage: edit.py --url URL --streams N --tokens T --out PREFIX"""
import argparse, json, threading, time, urllib.request

TASKS = [
    ("/usr/lib/python3.12/textwrap.py", 200, "Rename the class `TextWrapper` to `LineWrapper` everywhere (its uses too) "
     "and add a one-line docstring to every method that lacks one."),
    ("/usr/lib/python3.12/shlex.py", 200, "Add type hints to every function and method signature, and change the "
     "`debug` prints to use f-strings."),
    ("/usr/lib/python3.12/string.py", 200, "Rename the function `capwords` to `capitalize_words` (its uses too) and "
     "add a module-level `__version__ = \"1.0\"` after the imports."),
    ("/usr/lib/python3.12/bisect.py", 118, "Rename the parameter `lo` to `low` and `hi` to `high` in every function "
     "(their uses too)."),
]


def ask(url, i, tokens, res):
    path, lines, task = TASKS[i % len(TASKS)]
    code = "".join(open(path).readlines()[:lines])
    msg = (f"Here is a Python module:\n\n```python\n{code}```\n\n{task} Reply with the complete updated module in one "
           "code block and nothing else.")
    body = {"model": "spark", "messages": [{"role": "user", "content": msg}], "max_tokens": tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); first = None; parts = []; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                t = (ch.get("delta") or {}).get("content") or ""
                if t:
                    first = first or time.time()
                    parts.append(t)
    res[i] = (t0, first or time.time(), time.time(), usage.get("completion_tokens") or len(parts), "".join(parts))


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000/v1")
ap.add_argument("--streams", type=int, default=1)
ap.add_argument("--tokens", type=int, default=1500)
ap.add_argument("--out", default="")
a = ap.parse_args()
res = [None] * a.streams
th = [threading.Thread(target=ask, args=(a.url, i, a.tokens, res)) for i in range(a.streams)]
for t in th:
    t.start()
for t in th:
    t.join()
for i, (t0, f, e, n, text) in enumerate(res):
    print(f"edit {a.streams}s stream {i}: first token {f - t0:.2f} s, {n} tokens in {e - f:.2f} s -> {(n - 1) / max(e - f, 1e-9):.1f} tok/s")
    if a.out:
        open(f"{a.out}-{i}.txt", "w").write(text)
wall = max(r[2] for r in res) - min(r[0] for r in res)
print(f"edit {a.streams}s aggregate: {sum(r[3] for r in res)} tokens in {wall:.2f} s wall -> {sum(r[3] for r in res) / wall:.1f} tok/s")
