#!/usr/bin/env python3
"""Throughput of a running Inkling-2-Mini server (tools/serve-ink.sh), the way docs/inkling-2-mini.md measures it.

  tools/bench-ink.py [--url http://127.0.0.1:8000] [--short] [--ctx 8192,32768,65536,130000] [--tokens 200]

--short   three chat requests (code, agent-style tool result, prose), temperature 0, each sent twice: the second answer is the warm one
--ctx     one completion request per context length and kind (code, prose): a prompt of about that many tokens made of this repository's
          Markdown and Rust text in turn (not repeated while the repository lasts) whose last 1024 tokens are code or prose, which the answer continues;
          each sent twice, the second is reported. The server's own prompt count (`prompt_n`) is what is printed.
Prints the server's timings: prompt tokens per second (prefill) and generated tokens per second. Standard library only.
"""
import argparse, json, os, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def walk(top, exts, skip=("bench-data", "target", ".git", "assets")):
    out = []
    for dp, dn, fs in os.walk(os.path.join(ROOT, top)):
        dn[:] = sorted(d for d in dn if d not in skip)
        for f in sorted(fs):
            if f.endswith(exts):
                try:
                    out.append(open(os.path.join(dp, f), encoding="utf-8", errors="ignore").read())
                except OSError:
                    pass
    return out


SKIP_FILES = ("CHANGELOG.md", "README.md", "docs/inkling-2-mini.md", "RELEASE")


def corpus():
    """(prose, code): this repository's Markdown files and Rust source files, without the Inkling engine's own files (so the text a model
    continues does not change with the engine under test) and without the changelog and readme (they change with every release)."""
    prose, code = [], []
    for dp, dn, fs in os.walk(ROOT):
        dn[:] = sorted(d for d in dn if d not in ("bench-data", "target", ".git", "assets", "fe-ink", "fe-inkk", "k-ink", "k-inkm", "k-draft", "conformance"))
        for f in sorted(fs):
            rel = os.path.relpath(os.path.join(dp, f), ROOT)
            if rel in SKIP_FILES or f in ("ink.rs", "inkling.rs"):
                continue
            if f.endswith(".md") or f.endswith(".rs"):
                try:
                    txt = open(os.path.join(dp, f), encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                (prose if f.endswith(".md") else code).append(txt)
    return "\n\n".join(prose), "\n\n".join(code)


def context(n_tokens, kind, prose, code, chars_per_token=2.8):
    """About n_tokens of text: blocks of Markdown and Rust text in turn (no block twice while the repository lasts); the last ~1024 tokens are `kind`
    (taken from the middle of that text)."""
    tail_chars = int(1024 * chars_per_token)
    src = prose if kind == "prose" else code
    tail = src[len(src) // 2: len(src) // 2 + tail_chars]
    want = int(n_tokens * chars_per_token) - len(tail)
    parts, total, i, blk = [], 0, 0, 2400
    while total < want:
        p = prose[(i * blk) % max(1, len(prose) - blk):][:blk]
        c = code[(i * blk) % max(1, len(code) - blk):][:blk]
        parts += [p, c]
        total += len(p) + len(c)
        i += 1
    return "".join(parts)[:max(want, 0)] + "\n\n" + tail


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


SHORT = {
    "code": [{"role": "user", "content": "Write a Python function that checks whether a number is prime, with a few tests."}],
    "agent": [
        {"role": "system", "content": "You are a coding agent. You can call the tool read_file(path)."},
        {"role": "user", "content": "What does the function parse_header in src/net.py return for an empty header?"},
        {"role": "assistant", "content": "I will read the file.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"src/net.py\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "def parse_header(raw):\n    if not raw:\n        return {}\n    name, _, value = raw.partition(':')\n    return {name.strip().lower(): value.strip()}\n"},
    ],
    "prose": [{"role": "user", "content": "Write a short essay about the history of tea."}],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--ctx", default="")
    ap.add_argument("--tokens", type=int, default=200)
    a = ap.parse_args()
    base = a.url.rstrip("/")
    model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
    if a.short:
        for name, msgs in SHORT.items():
            for run in (1, 2):
                d = post(base + "/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": a.tokens, "temperature": 0})
                t = d["timings"]
                if run == 2:
                    print(f"short {name}: prompt {t['prompt_n']} tokens at {t['prompt_per_second']:.0f} tok/s, {t['predicted_n']} tokens generated at {t['predicted_per_second']:.1f} tok/s")
    if a.ctx:
        prose, code = corpus()
        for n in [int(x) for x in a.ctx.split(",")]:
            for kind in ("code", "prose"):
                text = context(n, kind, prose, code)
                for run in (1, 2):
                    t0 = time.time()
                    d = post(base + "/v1/completions", {"model": model, "prompt": text, "max_tokens": a.tokens, "temperature": 0})
                    t = d["timings"]
                    if run == 2:
                        print(f"context {t['prompt_n']} tokens, {kind} continuation: prefill {t['prompt_per_second']:.0f} tok/s ({t['prompt_ms'] / 1000:.1f} s), "
                              f"{t['predicted_n']} tokens at {t['predicted_per_second']:.1f} tok/s ({time.time() - t0:.0f} s for the request)", flush=True)


if __name__ == "__main__":
    main()
