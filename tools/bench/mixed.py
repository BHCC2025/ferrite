#!/usr/bin/env python3
"""Mixed-load check (client only, OpenAI chat API, Python 3 standard library): greedy and sampled requests together.

  mixed.py --url URL repro [--rounds 4]
      Three greedy "return this file unchanged" edits (copy rows / verify trees fill the step's decode rows), and while
      they run, short sampled requests one after another (a sampled prompt's last row lands in a step full of verify
      rows), then a sampled story beside three greedy edits. Every request must succeed and /health must stay ok.
  mixed.py --url URL soak [--minutes 30] [--clients 4] [--pid PID]
      CLIENTS threads loop over a mix (short sampled chats, thinking on, greedy and sampled code, 6K-20K-token prompts,
      a two-turn agent conversation behind a long system prompt, streamed replies, greedy edits, cancelled streams);
      a fixed greedy probe every 60 s must give the same text every time. Reports errors, empty replies (no
      finish_reason), /health failures and, with --pid, the server's RSS. Exit 1 on any failure.
"""
import argparse, hashlib, json, os, random, socket, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# the prompts quote a fixed copy of repository files (tools/bench/data/agents-corpus, as agents.py), so they stay the
# same as the repository changes and the script runs from a release tarball or next to tools/release-gate.sh
ROOT = os.path.join(HERE, "data", "agents-corpus")
FILES = ["crates/fe-server/src/server.rs", "crates/fe-server/src/parser.rs", "crates/fe-server/src/template.rs",
         "crates/fe-model/src/sched.rs", "crates/fe-server/src/toolcall.rs", "TP2.md", "docs/quickstart.md"]


def text(i, a, n):
    with open(os.path.join(ROOT, FILES[i % len(FILES)]) + ".txt", encoding="utf-8") as f:
        t = f.read()
    while len(t) < a + n:
        t += t
    return t[a:a + n]


class Api:
    def __init__(self, url):
        self.url = url

    def chat(self, messages, max_tokens, temperature, think=False, stream=False, cancel_after=None, timeout=1800):
        body = {"model": "m", "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": think}, "stream": stream}
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if not stream:
                j = json.loads(r.read())
                c = j["choices"][0]
                if not c.get("finish_reason"):
                    raise RuntimeError("reply without finish_reason")
                return (c["message"].get("reasoning_content") or "") + (c["message"].get("content") or ""), c["finish_reason"]
            out, fin, t0 = [], None, time.time()
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                j = json.loads(d)
                if "error" in j:
                    raise RuntimeError("stream error event: " + str(j["error"].get("message")))
                for c in j.get("choices", []):
                    dl = c.get("delta", {})
                    out.append((dl.get("reasoning_content") or "") + (dl.get("content") or ""))
                    fin = c.get("finish_reason") or fin
                if cancel_after and time.time() - t0 > cancel_after:
                    return "".join(out), "cancelled"
            if not fin:
                raise RuntimeError("stream ended without finish_reason or error")
            return "".join(out), fin

    def health(self):
        try:
            with urllib.request.urlopen(self.url.rsplit("/v1", 1)[0] + "/health", timeout=10) as r:
                return r.status == 200 and json.loads(r.read()).get("status") == "ok"
        except Exception:
            return False


def repro(api, rounds):
    edit = lambda i, r: [{"role": "user", "content": f"Return this file unchanged, nothing else (copy {i} {r}):\n\n" + text(i, 3000 * i, 6000)}]
    bad = 0
    for rnd in range(rounds):
        res = {}
        th = [threading.Thread(target=lambda i=i: res.__setitem__(i, safe(lambda: api.chat(edit(i, rnd), 1500, 0)))) for i in range(3)]
        [t.start() for t in th]
        time.sleep(2.5 if rnd % 2 == 0 else 0.5)
        s, k = [], 0
        while any(t.is_alive() for t in th):
            if rnd % 2 == 0:
                s.append(safe(lambda: api.chat([{"role": "user", "content": f"Name a color ({k})."}], 4, 1.0)))
            else:
                s.append(safe(lambda: api.chat([{"role": "user", "content": f"Write a short story about two rival bakers ({k})."}], 400, 0.8)))
            k += 1
        [t.join() for t in th]
        errs = [x for x in list(res.values()) + s if isinstance(x, Exception)]
        print(f"round {rnd}: 3 edits {[len(x[0]) if not isinstance(x, Exception) else repr(x)[:60] for x in res.values()]}, "
              f"{len(s)} sampled, {len(errs)} errors, health {'ok' if api.health() else 'FAIL'}", flush=True)
        bad += len(errs) + (not api.health())
    print("repro:", "PASS" if bad == 0 else "FAIL")
    return bad == 0


def safe(f):
    try:
        return f()
    except Exception as e:
        return e


PROBE = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}]


def soak(api, minutes, clients, pid):
    end = time.time() + 60 * minutes
    stats = {"ok": 0, "err": 0, "kinds": {}}
    errors = []
    lock = threading.Lock()
    sysbig = "You are the coding agent of this repository. Notes:\n\n" + text(5, 0, 22000)

    def job(rng):
        k = rng.randrange(9)
        n = rng.randrange(1_000_000)
        if k == 0:
            return "chat-sampled", api.chat([{"role": "user", "content": f"Give me a fun fact about the number {n}."}], 120, 1.0)
        if k == 1:
            return "think-sampled", api.chat([{"role": "user", "content": f"Is {n} prime? Explain briefly."}], 600, 0.7, think=True)
        if k == 2:
            return "code", api.chat([{"role": "user", "content": f"Write a Rust function #{n} that merges two sorted vectors."}], 500, rng.choice([0, 0.7]))
        if k == 3:
            m = rng.choice([24000, 48000, 80000])
            return "long-sampled", api.chat([{"role": "user", "content": f"[{n}] Summarize this in five bullets:\n\n" + text(n, n % 5000, m)}], 200, 1.0)
        if k == 4:
            conv = [{"role": "system", "content": sysbig}, {"role": "user", "content": f"[{n}] What does rank 0 send to rank 1?"}]
            t, _ = api.chat(conv, 200, 0)
            conv += [{"role": "assistant", "content": t}, {"role": "user", "content": "Now list three failure cases."}]
            return "agent-greedy", api.chat(conv, 200, 0)
        if k == 5:
            return "stream-sampled", api.chat([{"role": "user", "content": f"Write a haiku about rivers ({n})."}], 80, 1.0, stream=True)
        if k == 6:
            return "edit-greedy", api.chat([{"role": "user", "content": f"Add type annotations and return the whole file ({n}):\n\n" + text(n, n % 3000, 5000)}], 1500, 0)
        if k == 7:
            return "stream-cancel", api.chat([{"role": "user", "content": f"Count from 1 to 500 ({n})."}], 1500, rng.choice([0, 1.0]), stream=True, cancel_after=rng.uniform(0.3, 3))
        return "stream-greedy", api.chat([{"role": "user", "content": f"Explain borrow checking in Rust ({n})."}], 300, 0, stream=True)

    def worker(ci):
        rng = random.Random(1000 + ci)
        while time.time() < end:
            try:
                kind, _ = job(rng)
                with lock:
                    stats["ok"] += 1
                    stats["kinds"][kind] = stats["kinds"].get(kind, 0) + 1
            except Exception as e:
                with lock:
                    stats["err"] += 1
                    errors.append(f"{time.strftime('%H:%M:%S')} {repr(e)[:200]}")
                time.sleep(1)

    th = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(clients)]
    [t.start() for t in th]
    ref, probes, probe_bad, health_bad, rss = None, 0, 0, 0, []
    while time.time() < end:
        try:
            p, _ = api.chat(PROBE, 300, 0)
            probes += 1
            if ref is None:
                ref = p
            elif p != ref:
                probe_bad += 1
                print("probe DIFFERS:", hashlib.sha256(p.encode()).hexdigest()[:12], flush=True)
        except Exception as e:
            errors.append("probe " + repr(e)[:200])
            probe_bad += 1
        if not api.health():
            health_bad += 1
        if pid:
            try:
                with open(f"/proc/{pid}/status") as f:
                    rss.append(int(next(l for l in f if l.startswith("VmRSS")).split()[1]) // 1024)
            except Exception:
                pass
        with lock:
            print(f"{time.strftime('%H:%M:%S')} ok {stats['ok']} err {stats['err']} probes {probes} (bad {probe_bad})"
                  f" health_bad {health_bad}" + (f" rss {rss[-1]} MB" if rss else ""), flush=True)
        time.sleep(max(0, min(60, end - time.time())))
    [t.join(timeout=1900) for t in th]
    print("kinds:", stats["kinds"])
    for e in errors[:20]:
        print("ERROR", e)
    if rss:
        print(f"RSS MB: start {rss[0]} max {max(rss)} end {rss[-1]}")
    ok = stats["err"] == 0 and probe_bad == 0 and health_bad == 0
    print("soak:", "PASS" if ok else "FAIL", f"({stats['ok']} ok, {stats['err']} errors, {probes} probes, {probe_bad} bad)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("mode", choices=["repro", "soak"])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--pid", type=int)
    a = ap.parse_args()
    api = Api(a.url)
    ok = repro(api, a.rounds) if a.mode == "repro" else soak(api, a.minutes, a.clients, a.pid)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
