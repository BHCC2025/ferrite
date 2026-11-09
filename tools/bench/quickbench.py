#!/usr/bin/env python3
"""Quick streaming bench for the :8000 server: prefill and generation speed, code or prose.

usage: quickbench.py [--url http://host:8000/v1] [--tokens 500] [--context 2000] [--streams 1]
The menu changes the context size (c), tokens to generate (t) and concurrent streams (n); counts take 8k, 32000, ...
"""
import argparse, json, random, sys, threading, time, urllib.request
import os
if os.environ.get("QB_SEED"):
    random.seed(int(os.environ["QB_SEED"]))  # fixed nonces: same prompts across configs (A/B), core copy

CODE_CTX = """def parse_line(line):
    key, _, value = line.partition("=")
    return key.strip(), value.strip()

class Config:
    def __init__(self, path):
        self.path = path
        self.values = {}

    def load(self):
        with open(self.path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    k, v = parse_line(line)
                    self.values[k] = v
        return self.values
"""
CODE_ASK = ("Using the module above as a style reference, write a complete Python LRU cache class "
            "with get, put, delete, resize and a thread-safe wrapper, plus unit tests. Code only, no prose.")

PROSE_CTX = ("The harbor town woke slowly in winter. Fishing boats knocked against the pier, gulls argued "
             "over the market stalls, and the baker opened his shutters to a street still blue with frost. ")
PROSE_ASK = ("Using the passage above as a style reference, write an original short story about a lighthouse "
             "keeper who finds a message in a bottle. Rich description, no headings, no lists.")

MODES = {"1": ("code", CODE_CTX, CODE_ASK), "2": ("prose", PROSE_CTX, PROSE_ASK)}


def build_prompt(ctx, ask, context_tokens):
    # a random nonce up front keeps the prompt cache from serving a repeat run
    nonce = f"[run {random.randrange(10**9)}]\n"
    reps = max(1, context_tokens * 4 // len(ctx)) if context_tokens else 0
    return nonce + ctx * reps + "\n\n" + ask


def kind(text):
    """clean / partial / echo: greedy replies to the prose prompt often repeat its passage or their own first
    paragraph (zlib ratio of normal prose ~0.45-0.5, a repetition loop far lower, or a tail of one block three times)"""
    import zlib
    b = text.encode()
    z = len(zlib.compress(b, 9)) / max(len(b), 1)
    loop = any(text[-p:] == text[-2 * p:-p] == text[-3 * p:-2 * p] for p in range(8, min(600, len(text) // 3) + 1))
    return ("echo" if z < 0.2 or loop else "partial" if z < 0.4 else "clean"), z


def stream(url, mode, max_tokens, context_tokens, echo=True):
    """One streamed request and its timings; echo prints the reply as it arrives."""
    name, ctx, ask = MODES[mode]
    prompt = build_prompt(ctx, ask, context_tokens)
    parts = []
    body = {
        "model": "spark",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; chunks = 0; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                text = (ch.get("delta") or {}).get("content") or ""
                if text:
                    if t_first is None:
                        t_first = time.time()
                    chunks += 1
                    parts.append(text)
                    if echo:
                        sys.stdout.write(text); sys.stdout.flush()
    t_end = time.time()
    usage = usage or {}
    if os.environ.get("QB_TEXT"):  # loop check: the reply, one file a request, named by the nonce
        open(os.environ["QB_TEXT"] + "-" + prompt[5:prompt.index("]")] + ".txt", "w").write("".join(parts))
    k, z = kind("".join(parts))
    return {"kind": k, "zlib": z, "nonce": prompt[5:prompt.index("]")], "t0": t0, "first": t_first or t_end, "end": t_end,
            "prompt": usage.get("prompt_tokens"),
            "gen": usage.get("completion_tokens") or chunks,
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}


def run(url, mode, max_tokens, context_tokens):
    print(f"\n\033[1;36m=== {MODES[mode][0]}: {max_tokens} tokens, ~{context_tokens} context tokens ===\033[0m\n")
    s = stream(url, mode, max_tokens, context_tokens)
    ptoks, gtoks, cached, t0, t_first, t_end = s["prompt"], s["gen"], s["cached"], s["t0"], s["first"], s["end"]
    ttft = t_first - t0
    gen_s = max(t_end - t_first, 1e-9)
    print("\n\n\033[1;33m--- results ---\033[0m")
    if ptoks:
        print(f"prompt   {ptoks:>6} tokens  (cached {cached})")
        print(f"prefill  {ttft:>9.2f} s to first token  ->  {ptoks / ttft:,.0f} tok/s")
    else:
        print(f"prefill  {ttft:>9.2f} s to first token")
    print(f"generate {gtoks:>6} tokens in {gen_s:.2f} s  ->  \033[1;32m{(gtoks - 1) / gen_s:,.1f} tok/s\033[0m")
    print(f"reply    {s['kind']} (zlib {s['zlib']:.2f}, nonce {s['nonce']})")
    print(f"total    {t_end - t0:>9.2f} s")


def concurrent(url, mode, max_tokens, context_tokens, n):
    """n requests at once, each its own fresh prompt: every stream's speeds and the aggregate."""
    print(f"\n\033[1;36m=== {MODES[mode][0]} x{n} concurrent: {max_tokens} tokens each, ~{context_tokens} context "
          f"tokens each ===\033[0m\nrunning {n} streams...")
    results = [None] * n

    def one(i):
        try:
            results[i] = stream(url, mode, max_tokens, context_tokens, echo=False)
        except Exception as e:
            results[i] = e

    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("\n\033[1;33m--- results ---\033[0m")
    ok = []
    for i, r in enumerate(results, 1):
        if isinstance(r, Exception):
            print(f"stream {i:>2}  failed: {r}")
            continue
        ok.append(r)
        ttft = r["first"] - r["t0"]
        gen_s = max(r["end"] - r["first"], 1e-9)
        print(f"stream {i:>2}  prompt {r['prompt'] or '?':>6}  first token {ttft:6.2f} s  "
              f"generate {r['gen']:>5} in {gen_s:6.2f} s  ->  {(r['gen'] - 1) / gen_s:7.1f} tok/s  {r['kind']} {r['nonce']}")
    if not ok:
        return
    t0 = min(r["t0"] for r in ok)
    wall = max(r["end"] for r in ok) - t0
    gen = sum(r["gen"] for r in ok)
    prompt = sum(r["prompt"] or 0 for r in ok)
    last_first = max(r["first"] for r in ok) - t0
    per = [(r["gen"] - 1) / max(r["end"] - r["first"], 1e-9) for r in ok]
    print(f"\nstreams  {len(ok)} of {n} finished")
    if prompt:
        print(f"prefill  {prompt:>6} prompt tokens, all first tokens by {last_first:.2f} s  ->  {prompt / last_first:,.0f} tok/s")
    print(f"generate {gen:>6} tokens in {wall:.2f} s wall  ->  \033[1;32m{gen / wall:,.1f} tok/s aggregate\033[0m"
          f"  (per stream {min(per):.1f} to {max(per):.1f}, mean {sum(per) / len(per):.1f})")


def count(text):
    """A token count as typed: 2000, 8k, 32K, 1.5k (None if it isn't one)."""
    t = text.strip().lower().replace(",", "").replace("_", "")
    scale = 1000 if t.endswith("k") else 1
    try:
        n = int(float(t.rstrip("k")) * scale)
    except ValueError:
        return None
    return n if n >= 0 else None


def context_window(url):
    """The served model's context length from /models (None if the server doesn't say)."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as r:
            m = json.loads(r.read())["data"][0]
        return m.get("context_length") or m.get("max_model_len")
    except Exception:
        return None


def batch_limit(url):
    """How many streams the server decodes at once, from /health (None if it doesn't say)."""
    base = url.rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as r:
            return json.loads(r.read()).get("max_batch_size")
    except Exception:
        return None


def ask_count(label, current, examples, least):
    s = input(f"{label} (e.g. {examples}) [{current}]: ").strip()
    if not s:
        return current
    n = count(s)
    if n is None or n < least:
        print(f"  not a token count of at least {least}; keeping {current}")
        return current
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--tokens", type=count, default=500, help="tokens to generate (500, 2k, ...)")
    ap.add_argument("--context", type=count, default=2000, help="approx. prompt tokens of filler context (2000, 32k, ...)")
    ap.add_argument("--streams", type=int, default=1, help="concurrent requests (1 = a single streamed reply)")
    a = ap.parse_args()
    window = context_window(a.url)
    limit = batch_limit(a.url)
    while True:
        print("\n\033[1mQuick bench\033[0m  (" + a.url + f", {a.tokens} tokens, ~{a.context} ctx, {a.streams} stream"
              + ("s" if a.streams != 1 else "") + (f", model window {window}" if window else "")
              + (f", server runs {limit} at once" if limit else "") + ")")
        print("  1) code\n  2) prose\n  c) change context size\n  t) change tokens to generate\n"
              "  n) change concurrent streams\n  q) quit")
        choice = input("> ").strip().lower()
        if choice == "q":
            return
        if choice == "c":
            a.context = ask_count("context tokens", a.context, "0, 2k, 8k, 32k, 100k", 0)
            continue
        if choice == "t":
            a.tokens = ask_count("tokens to generate", a.tokens, "256, 500, 2k, 8k", 1)
            continue
        if choice == "n":
            a.streams = ask_count("concurrent streams", a.streams, "1, 2, 4, 8", 1)
            if limit and a.streams > limit:
                print(f"  the server decodes {limit} at once; the rest wait in its queue")
            continue
        if choice in MODES:
            if window and a.context + a.tokens > window:
                print(f"  ~{a.context} context + {a.tokens} tokens is over the model's {window}; lower one of them")
                continue
            try:
                if a.streams > 1:
                    concurrent(a.url, choice, a.tokens, a.context, a.streams)
                else:
                    run(a.url, choice, a.tokens, a.context)
            except Exception as e:
                print(f"\nrequest failed: {e}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
