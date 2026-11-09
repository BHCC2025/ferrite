#!/usr/bin/env python3
"""Prose speed over clean replies only: per engine (tag) and run (1p / 4p) the clean / partial / echo counts, the mean
tok/s of its clean replies, and the mean over the nonces clean in every engine. usage: cleansum.py <qbclean output>..."""
import collections, sys
rows = [l.split() for f in sys.argv[1:] for l in open(f) if len(l.split()) == 6]
by = collections.defaultdict(dict)  # (tag, run) -> nonce -> (tok/s, kind)
for tag, run, seed, v, k, nonce in rows:
    by[(tag, run)][nonce] = (float(v.replace(",", "")), k)
for run in sorted({r for _, r in by}):
    tags = sorted(t for t, r in by if r == run)
    common = set.intersection(*[{n for n, (v, k) in by[(t, run)].items() if k == "clean"} for t in tags])
    for t in tags:
        d = by[(t, run)]
        c = collections.Counter(k for v, k in d.values())
        clean = [v for v, k in d.values() if k == "clean"]
        both = [d[n][0] for n in common]
        print(f"{run} {t:10} {len(d):3} replies: clean {c['clean']:3} partial {c['partial']:3} echo {c['echo']:3} | "
              f"clean mean {sum(clean) / max(len(clean), 1):6.1f} | clean in all ({len(common)}) mean {sum(both) / max(len(both), 1):6.1f}")
