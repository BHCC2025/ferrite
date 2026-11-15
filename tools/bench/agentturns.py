#!/usr/bin/env python3
"""Agent-style multi-turn timing (client only, OpenAI chat API, Python 3 standard library): one conversation behind a
~5K-token system prompt grows for --turns turns (each turn appends the previous reply and a ~1K-token tool result),
greedy, streamed. Prints per turn: prompt tokens, cached tokens, time to first token, and a hash of the reply
(the same build and settings must give the same hashes).

  agentturns.py --url http://127.0.0.1:8000/v1 [--turns 8] [--max-tokens 200] [--seed 1]
"""
import argparse, hashlib, json, os, random, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
FILES = ["crates/fe-server/src/server.rs", "crates/fe-server/src/parser.rs", "crates/fe-server/src/template.rs",
         "crates/fe-server/src/toolcall.rs", "crates/fe-server/src/stop.rs", "crates/fe-server/src/sampler.rs",
         "crates/fe-server/src/tokenizer.rs", "crates/fe-server/src/batch.rs", "crates/fe-server/src/mock.rs"]


def read(p, a, n):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return f.read()[a:a + n]


def turn(url, messages, max_tokens):
    body = {"model": "m", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; out = []; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            j = json.loads(line[5:])
            if "error" in j:
                raise RuntimeError(j["error"])
            usage = j.get("usage") or usage
            for c in j.get("choices", []):
                d = c.get("delta", {}).get("content")
                if d:
                    ttft = ttft or time.time() - t0
                    out.append(d)
    return "".join(out), ttft or time.time() - t0, time.time() - t0, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tag = f"[job {rng.randrange(10**9)}]"
    system = tag + " You are a coding agent working in a Rust repository. Use the tool results to answer.\n\n" + \
        read("TP2.md", 0, 12000) + "\n\n" + read("crates/fe-server/src/engine.rs", 0, 8000)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "Find how stop strings are handled and propose one improvement."}]
    tot = 0.0
    for t in range(a.turns):
        reply, ttft, wall, u = turn(a.url, msgs, a.max_tokens)
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        tot += ttft
        print(f"turn {t + 1}: prompt {u.get('prompt_tokens')} cached {cached} ttft {ttft:.2f} s wall {wall:.2f} s "
              f"reply {hashlib.sha256(reply.encode()).hexdigest()[:10]}", flush=True)
        f = FILES[t % len(FILES)]
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"Tool result (read_file {f}):\n\n{read(f, 1000 * t, 4000)}\n\nContinue."}]
    print(f"total ttft {tot:.2f} s over {a.turns} turns")


if __name__ == "__main__":
    main()
