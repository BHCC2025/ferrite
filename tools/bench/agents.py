#!/usr/bin/env python3
"""Multi-agent workload bench (client only, OpenAI chat API, seeded).

N agents run at once. All share one ~13K-token system prompt (a coding-agent harness: rules, tool definitions,
project reference). Each agent then does its own task over 2-3 turns, and later turns resend the history (a server
prompt cache can serve them). Task kinds, cycled by agent index: code (write a module), edit (a 3-6K-token source
file in context, rewrite one function), tool (a short tool-call style answer). Greedy, thinking off unless --think.

usage: agents.py --url http://localhost:8000/v1 --agents 8 [--seed 1] [--tag x] [--think] [--json out.json]
Prints one summary line: tag N agg_tok/s per_agent_decode_mean ttft_p50/p95 per turn, prompt tokens and cached.
"""
import argparse, json, os, random, statistics, sys, threading, time, urllib.request

SRC = os.environ.get("AGENTS_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root: the corpus
REF_FILES = [("crates/fe-model/ENGINE_API.md", None), ("TP2.md", 14000)]
EDIT_FILES = ["crates/fe-model/src/tp.rs", "crates/fe-server/src/toolcall.rs", "crates/fe-exl3/src/linear.rs",
              "crates/fe-server/src/mock.rs", "crates/fe-fabric/src/roce.rs", "crates/fe-cuda/src/lib.rs",
              "crates/fe-fp4/src/lib.rs", "crates/fe-server/src/batch.rs", "crates/fe-server/src/parser.rs"]
TOOLS = ["read_file", "write_file", "edit_file", "list_dir", "grep", "glob", "run_shell", "run_tests", "git_status",
         "git_diff", "git_commit", "git_log", "web_fetch", "web_search", "ask_user", "todo_write", "todo_read",
         "spawn_agent", "send_message", "lint", "format", "build", "bench", "profile", "open_pr", "review_pr",
         "notebook_edit", "image_read", "pdf_read", "kill_process"]


def read(p, n=None):
    with open(os.path.join(SRC, p), errors="replace") as f:
        s = f.read()
    return s if n is None else s[:n]


def system_prompt():
    rng = random.Random(1234)  # the shared prefix is the same for every agent and every run
    parts = ["You are a coding agent working in a Rust inference-engine repository. Follow the project rules, use the "
             "tools below to read and change files, keep answers short, show code in fenced blocks, never invent "
             "file contents you have not read.\n\n# Tools\n"]
    for t in TOOLS:
        args = rng.sample(["path", "pattern", "content", "old", "new", "cmd", "timeout_s", "limit", "offset",
                           "message", "branch", "query", "url", "glob", "recursive", "dry_run", "reason"], 4)
        props = ", ".join(f'"{a}": {{"type": "{rng.choice(["string", "integer", "boolean"])}", "description": '
                          f'"The {a.replace("_", " ")} for {t.replace("_", " ")}; see the rules below for limits."}}'
                          for a in args)
        parts.append(f'## {t}\n{{"name": "{t}", "description": "{t.replace("_", " ").capitalize()} in the '
                     f'workspace. Returns a JSON result or an error string.", "parameters": {{"type": "object", '
                     f'"properties": {{{props}}}, "required": ["{args[0]}"]}}}}\n')
    parts.append("\n# Project reference\n")
    for p, n in REF_FILES:
        parts.append(f"\n## {p}\n```\n{read(p, n)}\n```\n")
    return "".join(parts)


def agent_script(i, rng):
    kind = ["code", "edit", "tool"][i % 3]
    f = EDIT_FILES[rng.randrange(len(EDIT_FILES))]
    if kind == "code":
        topic = rng.choice(["an LRU cache with TTL", "a token-bucket rate limiter", "a ring buffer with overwrite",
                            "a JSON-lines log reader", "a retrying HTTP client wrapper", "an interval tree",
                            "a trie with prefix counts", "a bounded worker pool"])
        return kind, [(f"Write a complete Rust module implementing {topic}, with doc comments and a small test "
                       f"module. Code only.", 400),
                      ("Add two more unit tests for edge cases and show only the new test functions.", 300),
                      ("Which tool from the list would you call next to run the tests? Answer with one JSON object "
                       "and one sentence.", 64)]
    if kind == "edit":
        body = read(f, rng.choice([11000, 15000, 19000]))
        return kind, [(f"Here is {f}:\n```rust\n{body}\n```\nRewrite the first public function in this file to "
                       f"return a Result with a descriptive error instead of panicking. Show the whole rewritten "
                       f"function only.", 400),
                      ("Now list every call site in the file shown that must change, as a short bullet list with "
                       "the line text.", 200)]
    return kind, [(f"Using the tool definitions, call the right tool to show the last 5 commits that touched {f}. "
                   f"Answer with one JSON object only.", 64),
                  ("The tool returned 5 commits. Call the tool that shows the diff of the newest one. One JSON "
                   "object only.", 64),
                  ("Summarize in two sentences what you would check in that diff.", 96)]


def chat(url, messages, max_tokens, think):
    body = {"model": "qwen3.8-27b", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    text = []
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    if first is None:
                        first = time.time()
                    text.append(d.get("content") or "")
    end = time.time()
    first = first or end
    gen = usage.get("completion_tokens") or len(text)
    return {"t0": t0, "first": first, "end": end, "ttft": first - t0, "gen": gen,
            "dec": (gen - 1) / max(end - first, 1e-9) if gen > 1 else 0.0, "prompt": usage.get("prompt_tokens", 0),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0), "text": "".join(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--save", help="write every turn's messages and reply text (for --replay)")
    ap.add_argument("--replay", help="send the saved turns one at a time, in order, and compare the reply texts")
    a = ap.parse_args()
    if a.replay:
        turns = json.load(open(a.replay))
        same = 0
        for t in turns:
            r = chat(a.url, t["messages"], t["max_tokens"], t["think"])
            ok = r["text"] == t["text"]
            same += ok
            if not ok:
                print(f"  DIFF agent {t['agent']} turn {t['turn']}: {len(r['text'])} vs {len(t['text'])} chars", flush=True)
        print(f"{a.tag} replay: {same} of {len(turns)} turns identical", flush=True)
        return
    sysp = system_prompt()
    results = [[] for _ in range(a.agents)]
    errors = []

    def run(i):
        rng = random.Random(a.seed * 1000 + i)
        kind, turns = agent_script(i, rng)
        msgs = [{"role": "system", "content": sysp}]
        for k, (q, mt) in enumerate(turns):
            msgs.append({"role": "user", "content": f"[agent {i}] {q}" if k == 0 else q})
            try:
                sent = [dict(m) for m in msgs]
                r = chat(a.url, msgs, mt, a.think)
            except Exception as e:
                errors.append(f"agent {i} turn {k}: {e}")
                return
            r.update(agent=i, kind=kind, turn=k, messages=sent, max_tokens=mt, think=a.think)
            results[i].append(r)
            msgs.append({"role": "assistant", "content": r["text"]})

    t0 = time.time()
    th = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(a.agents)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    flat = [r for rs in results for r in rs]
    gen = sum(r["gen"] for r in flat)
    pct = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))] if v else 0.0
    turns = []
    for k in range(3):
        tt = [r["ttft"] for r in flat if r["turn"] == k]
        if tt:
            turns.append(f"t{k} ttft {statistics.median(tt):.2f}/{pct(tt, 0.95):.2f}s")
    dec = [r["dec"] for r in flat if r["gen"] > 16]
    prompt = sum(r["prompt"] for r in flat)
    cached = sum(r["cached"] for r in flat)
    print(f"{a.tag} N={a.agents} agg {gen / wall:.1f} tok/s ({gen} tok / {wall:.1f} s) dec/agent "
          f"{statistics.mean(dec) if dec else 0:.1f} | {' | '.join(turns)} | prompt {prompt} cached {cached} "
          f"computed {prompt - cached}" + (f" | ERRORS {len(errors)}: {errors[:2]}" if errors else ""), flush=True)
    if a.save:
        with open(a.save, "w") as f:
            json.dump([{k: r[k] for k in ("agent", "turn", "messages", "max_tokens", "think", "text")} for r in flat], f)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"tag": a.tag, "agents": a.agents, "wall": wall, "results": [
                {k: v for k, v in r.items() if k not in ("text", "messages")} for r in flat], "errors": errors}, f)


if __name__ == "__main__":
    main()
