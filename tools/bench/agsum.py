#!/usr/bin/env python3
"""Summarize agents.py --json outputs: one row per file (config, N, seed).
usage: agsum.py <json>...  -> markdown table rows"""
import json, statistics, sys

def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else 0.0

print("| run | N | out tok/s agg | decode tok/s per agent (mean) | TTFT t0 p50/p95 s | TTFT t1 p50/p95 s | TTFT t2 p50/p95 s "
      "| prompt tok | cached | computed | hit turns |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for p in sys.argv[1:]:
    d = json.load(open(p))
    rs = d["results"]
    gen = sum(r["gen"] for r in rs)
    dec = [r["dec"] for r in rs if r["gen"] > 16]
    cells = []
    for k in range(3):
        t = [r["ttft"] for r in rs if r["turn"] == k]
        cells.append(f"{statistics.median(t):.1f} / {pct(t, 0.95):.1f}" if t else "-")
    prompt = sum(r["prompt"] for r in rs)
    cached = sum(r["cached"] for r in rs)
    later = [r for r in rs if r["turn"] > 0]
    hits = sum(1 for r in later if r["cached"] > 0)
    print(f"| {d['tag']} | {d['agents']} | {gen / d['wall']:.1f} | {statistics.mean(dec) if dec else 0:.1f} | "
          f"{cells[0]} | {cells[1]} | {cells[2]} | {prompt} | {cached} | {prompt - cached} | {hits}/{len(later)} |"
          + (f" ERR {len(d['errors'])}" if d["errors"] else ""))
