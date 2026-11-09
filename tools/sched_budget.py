# usage: python3 tools/sched_budget.py <server log with FE_SCHED_PROF=2> <first step> <end step>  (FWD: TP2 row costs)
# Counterfactual verify budgets from a FE_SCHED_PROF=2 trace: the trees are grown best-first, so a budget of B nodes
# is the first B of them; a round would have accepted the path's leading nodes with index < B. Per stream (slot) of
# the step range: E[accepted] at budgets B, and the 1-stream rate (accepted + 1) / (fixed + fwd(B + 1 rows)).
import re, sys, json, collections
log, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
# TP2 verify forward + argmax by rows (tree-test chain, one segment), interpolated
FWD = [(2, 29.84), (4, 30.29), (6, 31.38), (8, 32.06), (12, 35.84), (16, 39.35), (24, 48.16), (32, 53.17)]
def fwd(r):
    r = max(2, r)
    for (r0, t0), (r1, t1) in zip(FWD, FWD[1:]):
        if r <= r1:
            return t0 + (t1 - t0) * (r - r0) / (r1 - r0)
    return FWD[-1][1]
pat = re.compile(r"step (\d+): dec (\d+) rows \[([^\]]*)\] acc \[([^\]]*)\] prompt (\d+) \| draft ([\d.]+) fwd ([\d.]+) commit ([\d.]+) absorb ([\d.]+) rest ([\d.]+) ms \| slots \[([^\]]*)\] paths (\[.*?\]\])(?: copies .*)?$")
steps = [m for m in (pat.search(l) for l in open(log, errors="replace")) if m][a:b]
per = collections.defaultdict(list)  # slot -> [(budget used, path nodes)]
fixed1 = []
for m in steps:
    rows = [int(x) for x in m.group(3).split(",") if x.strip()]
    slots = [int(x) for x in m.group(11).split(",") if x.strip()]
    paths = json.loads(m.group(12))
    if int(m.group(5)) == 0 and len(rows) == 1:
        fixed1.append(float(m.group(6)) + float(m.group(8)) + float(m.group(9)))
    for r, s_, p in zip(rows, slots, paths):
        per[s_].append((r - 1, p))
f1 = sum(fixed1) / len(fixed1) if fixed1 else 7.5
print(f"{len(steps)} steps; 1-stream fixed (draft + commit + absorb) {f1:.1f} ms")
BS = [1, 2, 3, 5, 7, 9, 11, 13, 15]
for s_, L in sorted(per.items()):
    accs, rates = [], []
    for B in BS:
        obs = [sum(1 for _ in __import__("itertools").takewhile(lambda x: x < B, p)) for bu, p in L if bu >= B]
        e = sum(obs) / len(obs) if obs else float("nan")
        accs.append(e)
        rates.append(1e3 * (e + 1) / (f1 + fwd(B + 1)))
    best = max(range(len(BS)), key=lambda i: rates[i] if rates[i] == rates[i] else -1)
    print(f"slot {s_}: {len(L):4d} rounds, acc@B " + " ".join(f"{B}:{x:.2f}" for B, x in zip(BS, accs)))
    print(f"         1-stream tok/s@B " + " ".join(f"{B}:{x:.0f}" for B, x in zip(BS, rates)) + f"  best B {BS[best]}")
