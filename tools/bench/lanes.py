#!/usr/bin/env python3
"""Lanes bench and gate (client only, OpenAI chat API, Python 3 standard library).

A "lanes" call fans one job out into N generations on the same server: N streaming chat requests (thinking off)
sent together, all starting with the same text (a system prompt, then a shared block of real source files of about
P tokens) and ending with a short tail of their own (own file + task). Each lane writes code up to --max-tokens
(800: every task asks for more than that, so a lane normally ends on the limit). What counts is the time of the
whole set, not a lane's time to first token.

Like the client, the bench reads GET /health first (max_batch_size, max_model_len, running, queued) and never
sends more than max_batch_size requests at once: larger N are skipped and reported as such.

  bench  lanes.py --url http://127.0.0.1:8000/v1 [--lanes 1,2,4,8] [--prefix 2000,16000] [--runs 3]
         Per point, after one warm-up set: total output tok/s (all lanes' tokens / wall time of the set), per-lane
         tok/s (a lane's tokens / its own time from send to end, mean), prompt tokens sent, cached and computed
         (usage.prompt_tokens and usage.prompt_tokens_details.cached_tokens), and the lowest memory headroom seen
         during the set. Medians over the runs. Every set uses a fresh job id at the start of the prompt, so no
         set is served from an earlier one: each is the cold "all lanes arrive together" case.
  gate   lanes.py --url ... --gate [--seed 7] [--save ref.json | --compare ref.json] [--solo-only]
         Each lane's text in an N-lane call must equal the text of the same request sent alone, greedy; then a
         fix round (the even lanes again, same shared text + the lane's output so far + an error message, sent
         together) against the same requests alone. --solo-only --save writes the alone texts of a reference
         server (e.g. one with the prompt cache's resume off); --compare checks a run against that file. A fixed
         --seed gives the same prompts on every server.

Memory headroom (--mem): "meminfo" = MemAvailable of the machine the bench runs on (a unified-memory machine such
as a DGX Spark; run the bench on the server's host), "cmd:<shell command printing GiB free>" for anything else
(e.g. cmd:"nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1/1024}'"), "none".
The default "auto" is meminfo when the URL is local and /proc/meminfo exists.
"""
import argparse, hashlib, json, os, random, statistics, subprocess, sys, tempfile, threading, time
import urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("LANES_SRC", os.path.join(HERE, "..", ".."))  # the corpus: this repository
FILES = ["crates/fe-server/src/engine.rs", "crates/fe-server/src/stop.rs", "crates/fe-server/src/sampler.rs",
         "crates/fe-server/src/mock.rs", "crates/fe-server/src/tokenizer.rs", "crates/fe-server/src/template.rs",
         "crates/fe-server/src/toolcall.rs", "crates/fe-server/src/parser.rs", "crates/fe-server/src/batch.rs",
         "TP2.md", "docs/quickstart.md"]
SYSTEM = ("You are one lane of a parallel coding job in a Rust inference-engine repository. The job was split into "
          "lanes that run at the same time; every lane sees the same brief and the same shared files, then its own "
          "files and its own task. Rules: write only the files your task names; keep the public names the brief "
          "fixes, since other lanes call them; no placeholder code and no TODOs; no explanations before or after "
          "the code; one fenced block per file, with the path on the line before it; follow the style of the shared "
          "files (error handling with Result, doc comments on public items, tests in a tests module).")
BRIEF = ("# Brief\nAdd a small utilities crate next to the server crate shown below. Each lane writes one module of it "
         "with its tests. Modules must not depend on each other; they may use the standard library only. Keep every "
         "module under 400 lines.\n\n# Shared files\n")
TASKS = ["an LRU cache with per-entry time to live (get, put, remove, resize, purge_expired)",
         "a token-bucket rate limiter with a burst size and an injectable clock",
         "a fixed-capacity ring buffer that overwrites its oldest element, with iterators",
         "an interval tree over half-open u64 ranges with insert, remove and overlap queries",
         "a trie over byte strings with prefix counts, longest-prefix match and removal",
         "a bounded worker pool with a job queue, graceful shutdown and panic isolation",
         "a JSON-lines log reader that tolerates truncated last lines and reports line numbers",
         "a retry helper with exponential backoff, jitter from a seeded generator and a deadline"]
ERRORS = ["error[E0382]: borrow of moved value: `entry`\n  --> src/lib.rs:88:21\n   |\n81 |         let entry = self.map.remove(&key);\n   |             ----- move occurs here\n88 |         self.touch(&entry);\n   |                     ^^^^^^ value borrowed here after move",
          "error[E0308]: mismatched types\n  --> src/lib.rs:57:16\n   |\n57 |         return self.len;\n   |                ^^^^^^^^ expected `Option<usize>`, found `usize`\ntest tests::empty_is_none ... FAILED",
          "error[E0502]: cannot borrow `self.items` as mutable because it is also borrowed as immutable\n  --> src/lib.rs:102:13\ntest tests::wraps_around ... FAILED\nassertion `left == right` failed\n  left: 3\n right: 4",
          "warning: unused variable: `now`\nerror[E0599]: no method named `saturating_duration` found for struct `Instant`\n  --> src/lib.rs:44:30"]


def corpus():
    parts = []
    for p in FILES:
        fp = os.path.join(SRC, p)
        if os.path.isfile(fp):
            with open(fp, errors="replace") as f:
                parts.append(f"## {p}\n```\n{f.read()}\n```\n")
    if not parts:  # not inside the repository: any source tree (LANES_SRC), sorted for a fixed order
        for root, dirs, files in sorted(os.walk(SRC)):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "target")
            for n in sorted(files):
                if n.endswith((".rs", ".py", ".md", ".ts", ".c", ".h", ".go")):
                    with open(os.path.join(root, n), errors="replace") as f:
                        parts.append(f"## {n}\n```\n{f.read()}\n```\n")
            if sum(map(len, parts)) > 400_000:
                break
    text = "".join(parts)
    if len(text) < 20_000:
        sys.exit(f"lanes: no corpus under {SRC} (set LANES_SRC to a source tree)")
    return text


CORPUS = corpus()


def http_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def health(url):
    """The server's /health as the lanes client reads it."""
    u = urllib.parse.urlsplit(url)
    h = http_json(f"{u.scheme}://{u.netloc}/health")
    for k in ("max_batch_size", "max_model_len", "running", "queued"):
        if not isinstance(h.get(k), int):
            sys.exit(f"lanes: /health has no integer '{k}': {h}")
    return h


def messages(job, shared_chars, tail, layout):
    """One lane's messages: the job id and the system prompt, the shared block (the first shared_chars characters
    of the corpus), then the lane's own tail. layout: the shared block opens the user message, or ends the system one."""
    system = f"[job {job}]\n{SYSTEM}"
    shared = BRIEF + CORPUS[:shared_chars]
    if layout == "system":
        return [{"role": "system", "content": system + "\n\n" + shared}, {"role": "user", "content": tail.lstrip("\n")}]
    return [{"role": "system", "content": system}, {"role": "user", "content": shared + tail}]


def lane_tail(i):
    own = CORPUS[-(600 + 97 * i):][:240 + 53 * i]  # a lane's own snippet: different text and length per lane
    return (f"\n\n# Lane {i + 1}: your file\n```\n{own}\n```\n\n# Lane {i + 1}: your task\nWrite src/lane{i + 1}.rs: "
            f"a complete Rust module implementing {TASKS[i % len(TASKS)]}, with doc comments on every public item and "
            f"a tests module with at least eight tests. Code only.")


def fix_tail(i, text):
    return (f"\n\n# Lane {i + 1}: current file\nsrc/lane{i + 1}.rs\n```rust\n{text[:1500]}\n```\n\n# Lane {i + 1}: "
            f"errors\n```\n{ERRORS[i % len(ERRORS)]}\n```\n\nFix the file. Output the whole corrected file, code only.")


def chat(url, model, msgs, max_tokens, gate=None):
    """One streamed request. Returns its timings, usage and text."""
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    if gate is not None:
        gate.wait()  # all lanes of a set are sent together
    t0 = time.time()
    first, text, usage, finish, err = None, [], {}, None, None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("error"):
                    err = str(ev["error"])
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        first = first or time.time()
                        text.append(d.get("content") or "")
    except Exception as e:  # noqa: BLE001 (a failed lane is reported, the set goes on)
        err = f"{type(e).__name__}: {e}"
    end = time.time()
    gen = usage.get("completion_tokens", 0)
    return {"t0": t0, "first": first or end, "end": end, "gen": gen, "prompt": usage.get("prompt_tokens", 0),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "finish": finish,
            "text": "".join(text), "error": err}


class Mem:
    """Lowest memory headroom (GiB) seen while it runs; the probe is pluggable (--mem)."""

    def __init__(self, spec, url):
        host = urllib.parse.urlsplit(url).hostname or ""
        if spec == "auto":
            local = host in ("127.0.0.1", "localhost", "::1") or host == os.uname().nodename
            spec = "meminfo" if local and os.path.exists("/proc/meminfo") else "none"
        self.spec, self.low, self.stop, self.t = spec, None, threading.Event(), None

    def read(self):
        if self.spec == "meminfo":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1048576
        elif self.spec.startswith("cmd:"):
            out = subprocess.run(self.spec[4:], shell=True, capture_output=True, text=True, timeout=20).stdout.split()
            return min(float(x) for x in out) if out else None
        return None

    def __enter__(self):
        self.low = self.read()

        def loop():
            while not self.stop.wait(0.5):
                v = self.read()
                if v is not None:
                    self.low = v if self.low is None else min(self.low, v)
        if self.spec != "none":
            self.t = threading.Thread(target=loop, daemon=True)
            self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        if self.t:
            self.t.join()


def run_set(a, lanes_msgs, mem=True):
    """Send the lanes' requests together; returns (results in lane order, wall seconds, lowest headroom)."""
    n = len(lanes_msgs)
    gate = threading.Barrier(n)
    res = [None] * n

    def one(i):
        res[i] = chat(a.url, a.model, lanes_msgs[i], a.max_tokens, gate)
    th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    with Mem(a.mem if mem else "none", a.url) as m:
        for t in th:
            t.start()
        for t in th:
            t.join()
    for i, r in enumerate(res):
        if r["error"]:
            sys.exit(f"lanes: lane {i + 1} of {n} failed: {r['error']}")
    wall = max(r["end"] for r in res) - min(r["t0"] for r in res)
    return res, wall, m.low


def calibrate(a, p):
    """Characters of the corpus that make the shared prefix (job id + system prompt + shared block, as a request of
    its own) p tokens, within 16: measured with max_tokens 1 requests through the server's own tokenizer."""
    key = hashlib.sha1(f"{a.model}|{a.layout}|{p}|{hashlib.sha1(CORPUS.encode()).hexdigest()}".encode()).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"lanes-calib-{key}.json")
    if os.path.exists(path) and not a.recalibrate:
        with open(path) as f:
            return json.load(f)
    chars = int(p * 3.3)
    got = 0
    for _ in range(8):
        if chars > len(CORPUS):
            sys.exit(f"lanes: the corpus under {SRC} is too short for a {p}-token prefix")
        r = chat(a.url, a.model, messages(f"{random.getrandbits(32):08x}", chars, "", a.layout), 1)  # not from --seed
        if r["error"] or not r["prompt"]:
            sys.exit(f"lanes: calibration request failed: {r['error'] or 'no usage in the stream'}")
        got = r["prompt"]
        if abs(got - p) <= 16:
            break
        chars = max(200, int(chars * p / got))
    c = {"chars": chars, "tokens": got}
    with open(path, "w") as f:
        json.dump(c, f)
    return c


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def bench_point(a, n, p, cal):
    runs = []
    for k in range(a.runs + 1):  # the first set is the warm-up
        job = f"{a.rng.getrandbits(32):08x}"
        res, wall, low = run_set(a, [messages(job, cal["chars"], lane_tail(i), a.layout) for i in range(n)])
        gen = sum(r["gen"] for r in res)
        cached = None if any(r["cached"] is None for r in res) else sum(r["cached"] for r in res)
        run = {"wall": wall, "total_tps": gen / wall, "lane_tps": statistics.mean(r["gen"] / (r["end"] - r["t0"]) for r in res),
               "decode_tps": statistics.mean((r["gen"] - 1) / max(r["end"] - r["first"], 1e-9) for r in res),
               "first_s": max(r["first"] - r["t0"] for r in res), "gen": gen, "prompt": sum(r["prompt"] for r in res),
               "cached": cached, "mem_low": low, "short": sum(r["finish"] != "length" for r in res)}
        tag = "warm-up" if k == 0 else f"run {k}"
        print(f"  N={n} P={p} {tag}: {run['total_tps']:.1f} tok/s total, {run['lane_tps']:.1f} a lane, wall {wall:.2f}s, "
              f"last first token {run['first_s']:.2f}s, prompt {run['prompt']} cached {cached}, {gen} tokens out", flush=True)
        if k:
            runs.append(run)
    out = {"lanes": n, "prefix": p, "prefix_tokens": cal["tokens"], "runs": runs}
    for f in ("wall", "total_tps", "lane_tps", "decode_tps", "first_s", "gen", "prompt", "cached"):
        out[f] = med([r[f] for r in runs])
    lows = [r["mem_low"] for r in runs if r["mem_low"] is not None]
    out["mem_low"] = min(lows) if lows else None
    out["short"] = sum(r["short"] for r in runs)
    return out


def first_diff(x, y):
    i = next((i for i, (c, d) in enumerate(zip(x, y)) if c != d), min(len(x), len(y)))
    return f"at char {i} of {len(x)} / {len(y)}: {x[i:i + 40]!r} vs {y[i:i + 40]!r}"


def gate_point(a, n, p, cal, ref, save):
    """Lanes together vs each alone (and the fix round); returns the number of mismatches."""
    job = f"{a.rng.getrandbits(32):08x}"
    msgs = [messages(job, cal["chars"], lane_tail(i), a.layout) for i in range(n)]
    bad = 0

    def check(name, text, want, what):
        nonlocal bad
        if want is not None and text != want:
            bad += 1
            print(f"  MISMATCH {name} vs {what}: {first_diff(text, want)}", flush=True)

    def solo(ms):
        return [run_set(a, [m], mem=False)[0][0] for m in ms]
    if a.solo_only:
        tog = solo(msgs)
    else:
        tog = run_set(a, msgs, mem=False)[0]
    fix_ids = [i for i in range(n) if i % 2 == 0]
    fmsgs = [messages(job, cal["chars"], fix_tail(i, tog[i]["text"]), a.layout) for i in fix_ids]
    ftog = solo(fmsgs) if a.solo_only else run_set(a, fmsgs, mem=False)[0]
    if not a.solo_only:
        for i, r in enumerate(solo(msgs)):
            check(f"N={n} P={p} lane {i + 1}", tog[i]["text"], r["text"], "the same request alone")
        for i, r in zip(fix_ids, solo(fmsgs)):
            check(f"N={n} P={p} fix lane {i + 1}", ftog[fix_ids.index(i)]["text"], r["text"], "the same request alone")
    for i, r in enumerate(tog):
        k = f"{n}/{p}/lane{i + 1}"
        save[k] = r["text"]
        check(f"N={n} P={p} lane {i + 1}", r["text"], ref.get(k), "the reference file")
    for i, r in zip(fix_ids, ftog):
        k = f"{n}/{p}/fix{i + 1}"
        save[k] = r["text"]
        check(f"N={n} P={p} fix lane {i + 1}", r["text"], ref.get(k), "the reference file")
    cached = [r["cached"] for r in tog], [r["cached"] for r in ftog]
    print(f"  N={n} P={p} job {job}: {'solo only' if a.solo_only else 'lanes'} prompt {[r['prompt'] for r in tog]} cached "
          f"{cached[0]}, fix round prompt {[r['prompt'] for r in ftog]} cached {cached[1]}, {bad} mismatches", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="spark")
    ap.add_argument("--lanes", default="1,2,4,8", help="lane counts (those above the server's max_batch_size are skipped)")
    ap.add_argument("--prefix", default="2000,16000", help="shared prefix sizes in tokens")
    ap.add_argument("--runs", type=int, default=3, help="measured sets per point, after one warm-up set")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--layout", choices=["user", "system"], default="user",
                    help="the shared block opens the user message (default) or ends the system message")
    ap.add_argument("--mem", default="auto", help="memory headroom probe: auto, meminfo, none, cmd:<shell printing GiB>")
    ap.add_argument("--seed", type=int, help="fixed job ids (the same prompts on every server); default: from the clock")
    ap.add_argument("--recalibrate", action="store_true", help="measure the prefix sizes again (cached in the temp folder)")
    ap.add_argument("--json", help="write the points and their runs here")
    ap.add_argument("--tag", default="")
    ap.add_argument("--gate", action="store_true", help="lanes == alone gate instead of the bench")
    ap.add_argument("--solo-only", action="store_true", help="gate: only the alone requests (a reference for --save)")
    ap.add_argument("--save", help="gate: write the texts here")
    ap.add_argument("--compare", help="gate: also compare with the texts of this file")
    a = ap.parse_args()
    a.rng = random.Random(a.seed if a.seed is not None else time.time_ns())
    h = health(a.url)
    busy = h["running"] + h["queued"]
    print(f"lanes: {a.url} /health: max_batch_size {h['max_batch_size']}, max_model_len {h['max_model_len']}, running "
          f"{h['running']}, queued {h['queued']}" + (f" ({busy} other requests active: results will be off)" if busy else ""))
    ns = [int(x) for x in a.lanes.split(",")]
    ps = [int(x) for x in a.prefix.split(",")]
    skipped = [n for n in ns if n > h["max_batch_size"]]
    ns = [n for n in ns if n <= h["max_batch_size"]]
    if skipped:
        print(f"lanes: not sending N = {skipped}: above max_batch_size {h['max_batch_size']}")
    ps = [p for p in ps if p + a.max_tokens + 1024 < h["max_model_len"]] or sys.exit("lanes: no prefix fits max_model_len")
    cals = {p: calibrate(a, p) for p in ps}
    for p in ps:
        print(f"lanes: prefix {p}: {cals[p]['chars']} characters = {cals[p]['tokens']} prompt tokens (the shared part alone)")
    if a.gate:
        ref = json.load(open(a.compare)) if a.compare else {}
        # the prompts are made from this repository's files: texts saved from other files compare nothing
        corpus = hashlib.sha1(CORPUS.encode()).hexdigest()
        if ref.pop("_corpus", corpus) != corpus:
            sys.exit(f"lanes: {a.compare} was saved with other source files (the corpus under {SRC} changed): save the "
                     "reference again, or set LANES_SRC to the tree it was made from")
        save, bad = {"_corpus": corpus}, 0
        for p in ps:
            for n in ns:
                bad += gate_point(a, n, p, cals[p], ref, save)
        if a.save:
            with open(a.save, "w") as f:
                json.dump(save, f)
        print(f"lanes gate: {'PASS' if bad == 0 else 'FAIL'} ({bad} mismatches over N = {ns}, P = {ps}"
              f"{', against ' + a.compare if a.compare else ''})")
        sys.exit(1 if bad else 0)
    pts = []
    for p in ps:
        for n in ns:
            pts.append(bench_point(a, n, p, cals[p]))
    f1 = lambda v, fmt="{:.1f}": "-" if v is None else fmt.format(v)  # noqa: E731
    print(f"\n{a.tag + ': ' if a.tag else ''}medians of {a.runs} sets a point, max_tokens {a.max_tokens}, layout {a.layout}")
    print("| N | P | total tok/s | per-lane tok/s | decode tok/s a lane | set wall s | prompt tokens | cached | computed | "
          "lowest headroom GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for o in pts:
        comp = None if o["cached"] is None else o["prompt"] - o["cached"]
        print(f"| {o['lanes']} | {o['prefix']} | {f1(o['total_tps'])} | {f1(o['lane_tps'])} | {f1(o['decode_tps'])} | "
              f"{f1(o['wall'], '{:.2f}')} | {f1(o['prompt'], '{:.0f}')} | {f1(o['cached'], '{:.0f}')} | "
              f"{f1(comp, '{:.0f}')} | {f1(o['mem_low'])} |")
    if any(o["short"] for o in pts):
        print("(some lanes ended before max_tokens: " + ", ".join(f"N={o['lanes']} P={o['prefix']}: {o['short']}" for o in pts if o["short"]) + ")")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"tag": a.tag, "url": a.url, "health": h, "max_tokens": a.max_tokens, "layout": a.layout, "points": pts}, f)


if __name__ == "__main__":
    main()
