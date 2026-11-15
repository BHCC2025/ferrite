#!/usr/bin/env python3
"""Edit-style (agent) bench: a stdlib module in the prompt, an instruction that keeps most of it, the whole updated
module back (greedy). Streams start together; per stream tok/s, and each reply saved as <out>-<i>.txt (identity
across builds). usage: edit.py --url URL --streams N --tokens T --out PREFIX

The four modules are CPython 3.12.3's (the published edit numbers were measured on them), whatever Python runs this
script: /usr/lib/python3.12 when its copy is that exact file, else ~/.cache/ferrite-bench/cpython-3.12.3, filled once
from the v3.12.3 tag on GitHub; every file is checked against its sha256."""
import argparse, hashlib, json, os, threading, time, urllib.request

CPY = "3.12.3"
SHA = {
    "textwrap.py": "62867e40cdea6669b361f72af4d7daf0359f207c92cbeddfc7c7506397c1f31c",
    "shlex.py": "f927227de5ba5b1b2bdd75e3d9c8cb72b602b3bba3cc8edbf8fb554de0dc1fd7",
    "string.py": "24aeae1f0526250f442022022bf98df9a823b1cb330543ee79e70e44907462e9",
    "bisect.py": "f1cf7b85fc36b5da249813fc5ab97d9464f8cc1bc817f7146206fa2713e35999",
}


def source(name):
    """the text of CPython 3.12.3's Lib/<name>, sha256-checked"""
    ok = lambda b: hashlib.sha256(b).hexdigest() == SHA[name]
    cache = os.path.join(os.path.expanduser("~/.cache/ferrite-bench"), f"cpython-{CPY}", name)
    for p in (f"/usr/lib/python3.12/{name}", cache):
        try:
            b = open(p, "rb").read()
            if ok(b):
                return b.decode()
        except OSError:
            pass
    url = f"https://raw.githubusercontent.com/python/cpython/v{CPY}/Lib/{name}"
    b = urllib.request.urlopen(url, timeout=60).read()
    if not ok(b):
        raise SystemExit(f"edit.py: {url} does not match its sha256")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    open(cache, "wb").write(b)
    return b.decode()


TASKS = [
    ("textwrap.py", 200, "Rename the class `TextWrapper` to `LineWrapper` everywhere (its uses too) "
     "and add a one-line docstring to every method that lacks one."),
    ("shlex.py", 200, "Add type hints to every function and method signature, and change the "
     "`debug` prints to use f-strings."),
    ("string.py", 200, "Rename the function `capwords` to `capitalize_words` (its uses too) and "
     "add a module-level `__version__ = \"1.0\"` after the imports."),
    ("bisect.py", 118, "Rename the parameter `lo` to `low` and `hi` to `high` in every function "
     "(their uses too)."),
]


def ask(url, i, tokens, res):
    path, lines, task = TASKS[i % len(TASKS)]
    code = "".join(source(path).splitlines(keepends=True)[:lines])
    msg = (f"Here is a Python module:\n\n```python\n{code}```\n\n{task} Reply with the complete updated module in one "
           "code block and nothing else.")
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": msg}], "max_tokens": tokens, "temperature": 0,
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
