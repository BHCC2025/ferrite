#!/usr/bin/env python3
"""Mode gate (client only, OpenAI chat API, Python 3 standard library): greedy text must not depend on the server's
prompt cache (FERRITE_CACHE) or its drafting method (FERRITE_DRAFT mtp / dflash / off).

A fixed set of greedy requests, each sent to a fresh server (nothing cached) in one order: single-turn prompts with
and without a system message (short and ~3K tokens), a thinking-on question and its follow-up turn, a second turn
of a conversation with a ~4K-token system prompt, and a file edit. The script prints a hash per case and either saves
them (--save ref.json) or checks them against a reference made by another mode (--compare ref.json; exit 1 on any
difference, with the first differing characters).

  modes.py --url http://127.0.0.1:8000/v1 --save /tmp/ref-dflash.json      # server 1: the default mode
  modes.py --url http://127.0.0.1:8000/v1 --compare /tmp/ref-dflash.json   # server 2: FERRITE_CACHE=0, mtp, ...
"""
import argparse, hashlib, json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")


def src(path, a=0, b=None):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()[a:b]


def cases():
    review = src("crates/fe-server/src/stop.rs", 0, 12000)
    sysbig = "You are the coding agent of a Rust inference-engine repository. Project notes follow.\n\n" + \
        src("TP2.md", 0, 14000)
    edit = src("crates/fe-server/src/util.rs")
    out = [
        ("sys-short-review", [{"role": "system", "content": "You are a careful code reviewer."},
                              {"role": "user", "content": "Review this Rust file. Name the three most important issues "
                               "with line numbers.\n\n" + review}], False, 400),
        ("nosys-review", [{"role": "user", "content": "Review this Rust file. Name the three most important issues.\n\n"
                           + review}], False, 300),
        ("sys-short-chat", [{"role": "system", "content": "You answer in one short paragraph."},
                            {"role": "user", "content": "Why does a recurrent layer make prompt caching harder?"}], False, 200),
        ("think-17x23", [{"role": "user", "content": "What is 17 * 23? Answer briefly."}], True, 400),
        ("code", [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome, "
                   "ignoring case and punctuation, with three tests."}], False, 300),
        ("edit", [{"role": "user", "content": "Add a doc comment to every function in this file. Reply with the whole "
                   "file only.\n\n" + edit}], False, 900),
    ]
    return out, sysbig


def post(url, messages, think, max_tokens):
    body = {"model": "m", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read())
    m = j["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or ""), j["usage"]


def run(url):
    res = {}
    base, sysbig = cases()
    for name, msgs, think, mt in base:
        res[name], _ = post(url, msgs, think, mt)
        if name == "think-17x23":
            reply = res[name].split("\x00", 1)[1]
            res["think-17x23-turn2"], _ = post(url, msgs + [{"role": "assistant", "content": reply},
                                                            {"role": "user", "content": "Now multiply that by 3."}], True, 400)
    # a 2-turn conversation behind a long shared system prompt (the agent case)
    conv = [{"role": "system", "content": sysbig}, {"role": "user", "content": "In two sentences: what does rank 1 do?"}]
    t1, u1 = post(url, conv, False, 200)
    conv += [{"role": "assistant", "content": t1.split("\x00", 1)[1]},
             {"role": "user", "content": "And what happens when the link between the two nodes drops?"}]
    t2, u2 = post(url, conv, False, 200)
    res["sysbig-turn1"], res["sysbig-turn2"] = t1, t2
    print(f"sysbig prompt {u1['prompt_tokens']} / {u2['prompt_tokens']} tokens, turn 2 cached "
          f"{(u2.get('prompt_tokens_details') or {}).get('cached_tokens')}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--save")
    ap.add_argument("--compare")
    a = ap.parse_args()
    res = run(a.url)
    for k, v in res.items():
        print(f"{k:20s} {hashlib.sha256(v.encode()).hexdigest()[:12]} {len(v)} chars", flush=True)
    if a.save:
        with open(a.save, "w") as f:
            json.dump(res, f)
    bad = 0
    if a.compare:
        with open(a.compare) as f:
            ref = json.load(f)
        for k, v in res.items():
            r = ref.get(k)
            if r != v:
                bad += 1
                i = next((i for i, (x, y) in enumerate(zip(r or "", v)) if x != y), min(len(r or ""), len(v)))
                print(f"DIFF {k} at char {i}: ref {r[max(0, i - 40):i + 60]!r}\n          got {v[max(0, i - 40):i + 60]!r}")
        print("modes gate:", "PASS" if bad == 0 else f"FAIL ({bad} of {len(res)} differ)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
