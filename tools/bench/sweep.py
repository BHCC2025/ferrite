#!/usr/bin/env python3
"""The benchmark sweep behind docs/assets/ferrite-bench.png, against a running OpenAI-compatible server (greedy,
thinking off, the prompts of quickbench.py):
  prompt processing   2K / 8K / 32K / 64K / 120K prompt tokens, one request, cold cache
  generation          500 tokens after a 2K and a 32K prompt; code and prose; 1 request and 4 at once
  code edits          1,500 tokens, mostly copied from the prompt (edit.py); 1 request and 4 at once
Generation rates are tokens per second while generating: one request from its first to its last token; several
requests combined (combined(): after the last one's first token, so the prompt processing up front is not counted).
summary.json also keeps the wall-clock rates that include it (*_with_ttft) and the time to the last first token.
Every case runs on fixed seeds (the same prompts on every build, never a repeat within a sweep, so no prompt cache
hits) and is reported as the median, min and max of 3 runs. Prose counts only the replies that don't echo the
prompt (quickbench.py's "clean"), over 5 seeds for 1 request and 3 runs x 4 streams for 4 requests.

usage: sweep.py --url http://localhost:8000/v1 --out DIR [--quick] [--meta KEY=VALUE ...]
       sweep.py --plan                 the cases and a rough time, nothing is sent
       sweep.py --summarize DIR        rebuild summary.json and bench-data.json from DIR/raw.jsonl
Writes DIR/raw.jsonl (one line a request), DIR/summary.json (medians, min, max, counts) and DIR/bench-data.json (the
input of tools/gfx/bench.py; no build ids). --meta values go into summary.json (date=, driver=, kl_code=, kl_text=,
kl_long= for the KL line; title= / subtitle= for the chart); summary.json's "headline" holds the quoted figures.
"""
import argparse, json, os, random, statistics, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quickbench as qb  # noqa: E402

PREFILL_CTX = [2048, 8192, 32768, 65536, 122880]
GEN_CTX = [2048, 32768]
LABEL = {2048: "2K", 8192: "8K", 32768: "32K", 65536: "64K", 122880: "120K"}
MODE = {"code": "1", "prose": "2"}


def plan(quick=False):
    """(case id, kind, mode, context, streams, tokens, runs); a case's seeds are 100 * its index + run"""
    runs = 1 if quick else 3
    c = [("warmup", "warmup", "code", 2048, 1, 64, 1)]
    for ctx in PREFILL_CTX[:2] if quick else PREFILL_CTX:
        c.append((f"prefill-{LABEL[ctx]}", "prefill", "code", ctx, 1, 16, runs))
    for ctx in GEN_CTX[:1] if quick else GEN_CTX:
        for n in (1, 4):
            for mode in ("code", "prose"):
                r = 5 if (mode == "prose" and n == 1 and not quick) else runs
                c.append((f"gen-{mode}-{n}s-{LABEL[ctx]}", "gen", mode, ctx, n, 500, r))
    for n in (1, 4):
        c.append((f"edit-{n}s", "edit", "code", 0, n, 1500, runs))
    return c


def rough_seconds(c):
    """a rough wall time a case: prefill ~2.5K tok/s, code ~250 tok/s a stream, prose ~70, edits ~400"""
    _, kind, mode, ctx, n, tok, runs = c
    pre = n * ctx / 2500
    dec = {"warmup": tok / 250, "prefill": 0.2, "edit": n * tok / 400}.get(kind, tok / (250 if mode == "code" else 70))
    return runs * (pre + dec + 1)


def run_quickbench(url, mode, ctx, n, tokens):
    out = [None] * n

    def one(i):
        try:
            out[i] = qb.stream(url, MODE[mode], tokens, ctx, echo=False)
        except Exception as e:  # noqa: BLE001 - recorded as a failed request
            out[i] = {"error": str(e)}
    th = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def run_edit(url, n, tokens):
    p = subprocess.run([sys.executable, os.path.join(HERE, "edit.py"), "--url", url, "--streams", str(n), "--tokens",
                        str(tokens)], capture_output=True, text=True, timeout=1800)
    rows = []; agg = None
    for line in p.stdout.splitlines():
        # edit 1s stream 0: first token 0.21 s, 1499 tokens in 2.68 s -> 559.0 tok/s
        if " stream " in line and "tok/s" in line:
            w = line.split()
            rows.append({"ttft": float(w[6]), "gen": int(w[8]), "gen_s": float(w[11]), "tps": float(w[14])})
        elif "aggregate" in line:
            w = line.split()
            agg = {"gen": int(w[3]), "wall": float(w[6]), "tps": float(w[10])}
    if p.returncode != 0 or not rows or not agg:
        return {"error": (p.stderr or p.stdout)[-400:]}
    return {"streams": rows, "aggregate": agg}


def measure(url, out, quick, meta):
    os.makedirs(out, exist_ok=True)
    raw = open(os.path.join(out, "raw.jsonl"), "a")
    cases = plan(quick)
    print(f"sweep: {len(cases)} cases, roughly {sum(map(rough_seconds, cases)) / 60:.0f} min, against {url}", flush=True)
    raw.write(json.dumps({"meta": meta, "url": url, "start": time.strftime("%Y-%m-%d %H:%M:%S"), "quick": quick}) + "\n")
    for ci, c in enumerate(cases):
        cid, kind, mode, ctx, n, tokens, runs = c
        for r in range(runs):
            seed = 100 * ci + r
            t = time.time()
            if kind == "edit":
                res = run_edit(url, n, tokens)
            else:
                random.seed(seed)  # quickbench's nonces: the same prompts on every build
                res = run_quickbench(url, mode, ctx, n, tokens)
            rec = {"case": cid, "kind": kind, "mode": mode, "ctx": ctx, "streams": n, "tokens": tokens, "run": r,
                   "seed": seed, "result": res}
            raw.write(json.dumps(rec) + "\n"); raw.flush()
            print(f"  {cid:<22} run {r} ({time.time() - t:5.1f} s): {brief(rec)}", flush=True)
    raw.close()
    summarize(out)


def per_stream_tps(s):
    return (s["gen"] - 1) / max(s["end"] - s["first"], 1e-9)


def brief(rec):
    res = rec["result"]
    if isinstance(res, dict) and "error" in res:
        return "FAILED " + res["error"][:120]
    if rec["kind"] == "edit":
        return f"{res['aggregate']['tps']:.1f} tok/s aggregate"
    bad = [s for s in res if "error" in s]
    if bad:
        return f"{len(bad)} of {len(res)} requests FAILED: {bad[0]['error'][:100]}"
    if rec["kind"] in ("prefill", "warmup"):
        s = res[0]
        return f"{s['prompt']} prompt tokens (cached {s['cached']}) in {s['first'] - s['t0']:.2f} s -> {s['prompt'] / (s['first'] - s['t0']):,.0f} tok/s"
    kinds = " ".join(s["kind"] for s in res) if rec["mode"] == "prose" else ""
    if len(res) == 1:
        return f"{per_stream_tps(res[0]):.1f} tok/s {kinds}"
    wall = max(s["end"] for s in res) - min(s["t0"] for s in res)
    return f"{sum(s['gen'] for s in res) / wall:.1f} tok/s aggregate {kinds}"


def combined(spans):
    """the combined generation rate of streams running at once, from (first token, last token, tokens) per stream:
    the tokens generated after the last stream's first token (each stream's share by its own average rate), over
    the time from then to the last token. It leaves out the prompt processing up front, which a wall-clock rate
    over 4 x 32K prompts is mostly made of."""
    f = max(a for a, _, _ in spans)
    e = max(b for _, b, _ in spans)
    n = sum(g * max(b - max(a, f), 0) / max(b - a, 1e-9) for a, b, g in spans)
    return n / max(e - f, 1e-9)


def stat(v):
    return {"median": round(statistics.median(v), 1), "min": round(min(v), 1), "max": round(max(v), 1), "n": len(v)} if v else None


def summarize(out):
    lines = [json.loads(x) for x in open(os.path.join(out, "raw.jsonl"))]
    meta = {}
    for x in lines:
        if "meta" in x:
            meta.update(x["meta"]); meta.setdefault("url", x["url"]); meta.setdefault("start", x["start"])
    recs = [x for x in lines if "case" in x and x["kind"] != "warmup"]
    cases = {}
    for x in recs:
        cases.setdefault(x["case"], []).append(x)
    summary = {"meta": meta, "cases": {}}
    for cid, rs in cases.items():
        k = rs[0]["kind"]; failed = 0; cached = 0; s = {}
        if k == "edit":
            ok = [r["result"] for r in rs if "error" not in r["result"]]
            failed = len(rs) - len(ok)
            s["aggregate_with_ttft"] = stat([o["aggregate"]["tps"] for o in ok])
            s["per_stream"] = stat([st["tps"] for o in ok for st in o["streams"]])
            # the streams start together (edit.py's threads), so each one's times count from 0
            s["combined"] = stat([combined([(st["ttft"], st["ttft"] + st["gen_s"], st["gen"]) for st in o["streams"]]) for o in ok])
        else:
            runs = [r["result"] for r in rs]
            good = [[st for st in run if "error" not in st] for run in runs]
            failed = sum(len(run) - len(g) for run, g in zip(runs, good))
            full = [g for g, run in zip(good, runs) if len(g) == len(run)]  # runs without a failed request
            cached = sum(st.get("cached") or 0 for g in good for st in g)
            if k == "prefill":
                s["prefill"] = stat([g[0]["prompt"] / (g[0]["first"] - g[0]["t0"]) for g in full])
                s["prompt_tokens"] = stat([g[0]["prompt"] for g in full])
            else:
                streams = [st for g in full for st in g]
                clean = [st for st in streams if st["kind"] == "clean"]
                s["kinds"] = {kk: sum(st["kind"] == kk for st in streams) for kk in ("clean", "partial", "echo")}
                s["prompt_tokens"] = stat([st["prompt"] for st in streams if st["prompt"]])
                s["per_stream"] = stat([per_stream_tps(st) for st in streams])
                s["per_stream_clean"] = stat([per_stream_tps(st) for st in clean])
                # all tokens over the wall time from the requests to the last token (includes the prompt processing)
                s["aggregate_with_ttft"] = stat([sum(st["gen"] for st in g) / (max(st["end"] for st in g) - min(st["t0"] for st in g))
                                                 for g in full])
                s["combined"] = stat([combined([(st["first"], st["end"], st["gen"]) for st in g]) for g in full])
                s["ttft_last"] = stat([max(st["first"] for st in g) - min(st["t0"] for st in g) for g in full])
                s["ttft"] = stat([st["first"] - st["t0"] for st in streams])
        s.update({"kind": k, "mode": rs[0]["mode"], "ctx": rs[0]["ctx"], "streams": rs[0]["streams"], "runs": len(rs),
                  "failed_requests": failed, "cached_tokens": cached})
        summary["cases"][cid] = s
    summary["headline"] = headline(summary)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    json.dump(chart_data(summary), open(os.path.join(out, "bench-data.json"), "w"), indent=1)
    report(summary)
    print(f"sweep: wrote {out}/summary.json and {out}/bench-data.json")


def figure(summary, cid, key):
    """the chart's [median, min, max] of a case's statistic, or None (drawn as 'not measured')"""
    s = (summary["cases"].get(cid) or {}).get(key)
    return [s["median"], s["min"], s["max"]] if s else None


def kl(meta):
    """KL vs BF16 (code / text / 16K context) from --meta kl_code= kl_text= kl_long= (the final gate's tools/kl-tp2.sh
    runs); defaults: the published figures of docs/benchmarks.md"""
    return {k: float(meta.get(f"kl_{k}", d)) for k, d in (("code", 0.021), ("text", 0.040), ("long", 0.024))}


def headline(summary):
    """the figures README / site tiles / hero quote: medians, tok/s (prefill: prompt tok/s), None = not measured"""
    def med(cid, key):
        s = (summary["cases"].get(cid) or {}).get(key)
        return s["median"] if s else None
    m = summary["meta"]
    return {
        # generation rates: tokens / second while generating (1 request: first to last token; 4 requests: combined())
        "code_4s_2k": med("gen-code-4s-2K", "combined"),
        "code_1s_2k": med("gen-code-1s-2K", "per_stream"),
        "code_1s_32k": med("gen-code-1s-32K", "per_stream"),
        "code_4s_32k": med("gen-code-4s-32K", "combined"),
        "prose_1s_2k": med("gen-prose-1s-2K", "per_stream_clean"),
        "prose_4s_2k_per_request": med("gen-prose-4s-2K", "per_stream_clean"),
        "edit_1s": med("edit-1s", "per_stream"),
        "edit_4s": med("edit-4s", "combined"),
        # wall-clock rates including the prompt processing (docs/benchmarks.md's earlier 4-request figure), and the
        # time until all four requests had their first token
        "code_4s_2k_with_ttft": med("gen-code-4s-2K", "aggregate_with_ttft"),
        "code_4s_32k_with_ttft": med("gen-code-4s-32K", "aggregate_with_ttft"),
        "edit_4s_with_ttft": med("edit-4s", "aggregate_with_ttft"),
        "ttft_4s_2k_s": med("gen-code-4s-2K", "ttft_last"),
        "ttft_4s_32k_s": med("gen-code-4s-32K", "ttft_last"),
        "prefill_2k": med("prefill-2K", "prefill"),
        "prefill_32k": med("prefill-32K", "prefill"),
        "prefill_120k": med("prefill-120K", "prefill"),
        "kl_code": kl(m)["code"], "kl_text": kl(m)["text"], "kl_16k": kl(m)["long"],
        "kl_source": "measured (--meta)" if any(f"kl_{k}" in m for k in ("code", "text", "long")) else "docs/benchmarks.md",
        "setup": "two DGX Sparks, Qwen3.8-27B (EXL3 3.0 bpw decode, NVFP4 prompt weights), DFlash2 + copy rows, 4 x 128K",
        "date": m.get("date"),
    }


def nice_step(top):
    for st in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000):
        if top / st <= 6:
            return st
    return 10000


def panel(title, unit, ctxs, pts):
    vals = [v for p in pts if p for v in p]
    top = max(vals) * 1.2 if vals else 100
    step = nice_step(top)
    return {"title": title, "unit": unit, "contexts": ctxs, "ctx_labels": [LABEL[c] for c in ctxs], "ystep": step,
            "ymax": step * -(-top // step), "fe": pts}


def chart_data(summary):
    m = summary["meta"]
    gen = lambda mode, n, key: [figure(summary, f"gen-{mode}-{n}s-{LABEL[c]}", key) for c in GEN_CTX]  # noqa: E731
    panels = [
        panel("Code generation, 1 request", "tokens / second while generating", GEN_CTX, gen("code", 1, "per_stream")),
        panel("Code generation, 4 requests", "tokens / second while generating, all four combined", GEN_CTX,
              gen("code", 4, "combined")),
        panel("Prompt processing", "prompt tokens / second, cold cache", PREFILL_CTX,
              [figure(summary, f"prefill-{LABEL[c]}", "prefill") for c in PREFILL_CTX]),
        panel("Prose generation, 1 request", "tokens / second, replies that don't echo the prompt", GEN_CTX,
              gen("prose", 1, "per_stream_clean")),
        panel("Prose generation, 4 requests", "tokens / second per request, replies that don't echo", GEN_CTX,
              gen("prose", 4, "per_stream_clean")),
    ]
    e1, e4 = figure(summary, "edit-1s", "per_stream"), figure(summary, "edit-4s", "combined")
    tile = {"type": "text", "title": "Code edits", "unit": "1,500 tokens, mostly copied from the prompt",
            "lines": [["1 request", f"{e1[0]:,.0f} tok/s" if e1 else "not measured"],
                      ["4 requests", f"{e4[0]:,.0f} tok/s combined" if e4 else "not measured"]],
            "foot": "the agent case: speculative decoding drafts the copied text"}
    when = m.get("date") or time.strftime("%b %Y")
    return {
        "title": m.get("title", "Ferrite on two DGX Sparks"),
        "subtitle": m.get("subtitle", "QWEN3.8-27B  ·  GREEDY  ·  FIXED PROMPTS  ·  MEDIANS OF 3 RUNS"),
        "fe_name": "Ferrite",
        "grid": [3, 2],
        "contexts": PREFILL_CTX,
        "ctx_labels": [LABEL[c] for c in PREFILL_CTX],
        "panels": panels + [tile],
        "method": "Medians of 3 runs (prose: replies that don't echo the prompt); bars: min to max. Generation rates leave out prompt processing.",
        "method2": f"Two DGX Sparks  ·  Qwen3.8-27B: EXL3 3.0 bpw decode + NVFP4 prompt weights  ·  DFlash2 + copy rows"
                   f"  ·  4 x 128K  ·  {when}",
        "kl": "KL vs BF16: " + "  ·  ".join(f"{lab} {kl(m)[key]:.3f}" for key, lab in (("code", "code"), ("text", "text"), ("long", "16K"))),
        "kl2": m.get("kl2", "lower is better  ·  speculative decoding is lossless"),
    }


def report(summary):
    print(f"\n{'case':<22} {'median':>9} {'min':>9} {'max':>9}  n  notes")
    for cid, s in summary["cases"].items():
        key = {"prefill": "prefill"}.get(s["kind"])
        if not key:
            key = "per_stream" if s["streams"] == 1 else "combined"
            if s["mode"] == "prose":
                key = "per_stream_clean"
        v = s.get(key) or {}
        note = f"{key}"
        if s.get("kinds") and s["mode"] == "prose":
            note += f", clean {s['kinds']['clean']} partial {s['kinds']['partial']} echo {s['kinds']['echo']}"
        if s.get("prompt_tokens"):
            note += f", prompt {s['prompt_tokens']['median']:.0f} tok"
        if s["failed_requests"]:
            note += f", {s['failed_requests']} FAILED"
        if s["cached_tokens"]:
            note += f", {s['cached_tokens']} CACHED prompt tokens"
        print(f"{cid:<22} {v.get('median', float('nan')):>9.1f} {v.get('min', float('nan')):>9.1f} "
              f"{v.get('max', float('nan')):>9.1f} {v.get('n', 0):>2}  {note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--out")
    ap.add_argument("--quick", action="store_true", help="1 run a case, 2K/8K prefill and 2K generation only (a smoke test)")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--summarize", metavar="DIR")
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    if a.plan:
        cs = plan(a.quick)
        for c in cs:
            print(f"  {c[0]:<22} {c[4]} stream(s), {c[5]} tokens, {c[6]} run(s), ~{rough_seconds(c):.0f} s")
        print(f"{len(cs)} cases, roughly {sum(map(rough_seconds, cs)) / 60:.0f} min plus the server start")
        return
    if a.summarize:
        summarize(a.summarize)
        return
    if not a.out:
        ap.error("--out DIR is required")
    meta = dict(kv.split("=", 1) for kv in a.meta)
    measure(a.url, a.out, a.quick, meta)


if __name__ == "__main__":
    main()
