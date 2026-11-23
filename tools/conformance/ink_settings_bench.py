#!/usr/bin/env python3
"""What the sampling and reasoning settings cost on the Inkling server (drafting on): (A) temperature 0 / 0.7 / 1.0, 600 tokens, 3 seeds; (B) reasoning
effort none / low / medium / high, to the end of the answer, 3 seeds (temperature 1.0). Per request: tok/s (server timings, decode only), tokens per
verify pass (the server's 'ink spec' log line), reasoning tokens, answer tokens, wall seconds.
usage: ink_settings_bench.py http://127.0.0.1:8000 /path/to/server.log [a|b|ab]"""
import json, re, statistics as st, sys, time, urllib.request
BASE = sys.argv[1].rstrip("/"); LOG = sys.argv[2]; WHAT = sys.argv[3] if len(sys.argv) > 3 else "ab"
PROMPTS = {"code": "Write a Python function that merges overlapping intervals in a list of (start, end) tuples, with a short explanation and two test cases.",
           "prose": "Explain in a few paragraphs why the sky is blue at noon and red at sunset, for a curious teenager."}
def log_tail():
    try: lines = [l for l in open(LOG, errors="replace") if "ink spec (" in l]
    except OSError: return None
    return lines[-1] if lines else None
def run(prompt, **kw):
    body = {"model": "inkling-2-mini", "messages": [{"role": "user", "content": PROMPTS[prompt]}]}; body.update(kw)
    before = log_tail(); t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=3600).read())
    wall = time.time() - t0; time.sleep(0.3)
    u = r["usage"]; tm = r.get("timings", {}); c = r["choices"][0]
    line = log_tail(); tpp = None
    if line and line != before:
        m = re.search(r"\((\d+)\.?\d* passes|ink spec \(\w+\): (\d+) passes, (\d+) tokens \(([\d.]+) a pass\)", line)
        m = re.search(r"(\d+) passes, (\d+) tokens \(([\d.]+) a pass\), (\d+) drafts accepted of (\d+) verified, (\d+) plain tokens", line)
        if m: tpp = (float(m.group(3)), int(m.group(1)), int(m.group(6)))
    rt = u["completion_tokens_details"]["reasoning_tokens"]
    return {"tps": tm.get("predicted_per_second"), "tpp": tpp, "reasoning": rt, "answer": u["completion_tokens"] - rt, "total": u["completion_tokens"], "wall": wall, "finish": c["finish_reason"], "answer_chars": len(c["message"].get("content") or "")}
def mean(xs): xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else float("nan")
if "a" in WHAT:
    print("\n(A) temperature, 600 tokens, default reasoning effort (0.9), 3 seeds; decode tok/s from the server's timings; tokens per pass = tokens made by drafted passes / passes (plain-decode tokens counted separately)")
    print("| prompt | temperature | tok/s (3 runs) | tokens per pass | passes | plain tokens | reasoning / total tokens |"); print("|---|---|---|---|---|---|---|")
    for p in PROMPTS:
        for T in (0, 0.7, 1.0):
            rs = [run(p, temperature=T, seed=s, max_tokens=600) for s in (1, 2, 3)]
            print("| %s | %s | %s | %s | %s | %s | %s / %s |" % (p, T, " / ".join("%.1f" % r["tps"] for r in rs), " / ".join("%.2f" % r["tpp"][0] if r["tpp"] else "-" for r in rs),
                  " / ".join(str(r["tpp"][1]) if r["tpp"] else "-" for r in rs), " / ".join(str(r["tpp"][2]) if r["tpp"] else "-" for r in rs), " / ".join(str(r["reasoning"]) for r in rs), " / ".join(str(r["total"]) for r in rs)), flush=True)
if "b" in WHAT:
    print("\n(B) reasoning effort, to the end of the answer (max_tokens 8000), temperature 1.0, seeds 1-3; medians over the 3 runs (min-max in brackets)")
    print("| prompt | effort | reasoning tokens | answer tokens | total s | tok/s | tokens per pass | finish |"); print("|---|---|---|---|---|---|---|---|")
    for p in PROMPTS:
        for e in ("none", "low", "medium", "high"):
            rs = [run(p, temperature=1.0, seed=s, max_tokens=8000, reasoning_effort=e) for s in (1, 2, 3)]
            f = lambda k: "%d (%d-%d)" % (st.median([r[k] for r in rs]), min(r[k] for r in rs), max(r[k] for r in rs))
            g = lambda k: "%.1f (%.1f-%.1f)" % (st.median([r[k] for r in rs]), min(r[k] for r in rs), max(r[k] for r in rs))
            tpps = [r["tpp"][0] for r in rs if r["tpp"]]
            print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (p, e, f("reasoning"), f("answer"), g("wall"), "%.1f" % st.median([r["tps"] for r in rs]), "%.2f" % st.median(tpps) if tpps else "-", "/".join(r["finish"] for r in rs)), flush=True)
