#!/usr/bin/env python3
"""Prompt tokens computed today vs with (a) per-slot reuse that always hits (slots >= N, prompt-end GDN snapshot +
longest-common-prefix match) and (b) + cross-slot sharing of the common system prefix P.
usage: agprefix.py P <json>..."""
import json, sys
P = int(sys.argv[1])
print("| run | N | prompt tok | computed today | (a) per-slot always hits | (b) + shared prefix once | GPU s at 3.2K tok/s today / (b) |")
print("|---|---|---|---|---|---|---|")
for p in sys.argv[2:]:
    d = json.load(open(p))
    by = {}
    for r in d["results"]:
        by.setdefault(r["agent"], []).append(r)
    prompt = sum(r["prompt"] for r in d["results"])
    today = prompt - sum(r["cached"] for r in d["results"])
    a = sum(max(r["prompt"] for r in rs) for rs in by.values())          # each agent: its last prompt once
    b = P + sum(max(r["prompt"] for r in rs) - P for rs in by.values())   # the shared prefix once
    print(f"| {d['tag']} | {d['agents']} | {prompt} | {today} | {a} | {b} | {today / 3200:.0f} / {b / 3200:.0f} |")
