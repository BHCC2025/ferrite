# usage: python3 tools/sched_steps.py <server log with FE_SCHED_PROF=2> <first step> [end step]
# Per-step trace (FE_SCHED_PROF=2) of a server log, between step indices [a, b): pure decode steps grouped by the
# number of decoding streams: rows, accepted, tokens out a step, phase ms, tok/s; prefill steps summarized.
import re, sys, collections
log, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
pat = re.compile(r"step (\d+): dec (\d+) rows \[([^\]]*)\] acc \[([^\]]*)\] prompt (\d+) \| draft ([\d.]+) fwd ([\d.]+) commit ([\d.]+) absorb ([\d.]+) rest ([\d.]+) ms")
steps = []
for line in open(log, errors="replace"):
    m = pat.search(line)
    if m:
        steps.append(m)
sel = steps[a:b]
g = collections.defaultdict(list)
pre = []
for m in sel:
    rows = [int(x) for x in m.group(3).split(",") if x.strip()]
    acc = [int(x) for x in m.group(4).split(",") if x.strip()]
    ph = [float(m.group(i)) for i in range(6, 11)]
    if int(m.group(5)) > 0:
        pre.append((len(rows), int(m.group(5)), sum(ph)))
    else:
        g[len(rows)].append((rows, acc, ph))
print(f"steps {len(sel)}: prefill steps {len(pre)} ({sum(p[2] for p in pre)/1e3:.2f} s, prompt rows {sum(p[1] for p in pre)}, decode streams riding {[p[0] for p in pre]})")
for k in sorted(g):
    L = g[k]
    n = len(L)
    rows = sum(sum(r) for r, _, _ in L) / n
    toks = sum(sum(x + 1 for x in a_) for _, a_, _ in L) / n
    accs = sum(sum(a_) for _, a_, _ in L) / (n * k)
    ph = [sum(p[i] for _, _, p in L) / n for i in range(5)]
    tot = sum(ph)
    print(f"  {k} streams: {n:4d} steps, rows {rows:5.1f} ({rows/k:4.1f}/stream), acc/stream {accs:4.2f}, tok/step {toks:5.2f} | "
          f"draft {ph[0]:5.1f} fwd {ph[1]:5.1f} commit {ph[2]:4.1f} absorb {ph[3]:4.1f} rest {ph[4]:4.1f} = {tot:5.1f} ms -> {1e3*toks/tot:6.1f} tok/s")
