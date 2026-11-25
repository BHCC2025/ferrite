#!/usr/bin/env python3
"""Image-input checks against a running fe-server (--engine ink). Standard library only.

  e2e_images.py --url URL --images DIR answers  [--save F] [--question TEXT]
                                                             one greedy question a test picture (DIR/*.png, *.jpg)
  e2e_images.py --url URL --images DIR cache    [--save F]   two turns with one picture (turn 2 must come from the prompt
                                                             cache), then another picture of the same size (no cache hit)
  e2e_images.py --url URL --images DIR pair     [--save F] [--against F]
                                                             two picture requests at once, then each alone; --against: a file
                                                             saved by `answers` on a server that ran them one at a time from an
                                                             empty cache (the replies sent at once must equal those)
  e2e_images.py --url URL --images DIR park     [--save F]   picture conversations in turn (more than the server has lanes), then
                                                             the first one's next turn: it must come back from the park pool
  e2e_images.py --url URL --images DIR deep     [--save F]   a picture that arrives deep in a long conversation (the card's memory
                                                             is in use by the K/V then), a tool result that is a picture, and eight
                                                             pictures asked about twice (the second request resumes from a
                                                             prompt-cache mark that lies inside a picture)
  e2e_images.py --url URL text                  [--save F]   text-only chat prompts (for a comparison with another build)
  e2e_images.py --url URL --images DIR limits   [--big F] [--max-pictures N] [--overflow drop|reject]
                                                             requests the server must refuse with HTTP 400; --big: a picture above
                                                             the per-picture token limit, which the server must shrink and take;
                                                             one picture more than INK_IMAGE_MAX (N, default 64): taken with the
                                                             oldest N / 4 left out (drop, the default) or HTTP 400 (reject)
  e2e_images.py --url URL overflow [--turns N] [--size WxH] [--max-pictures N] [--max-image-tokens N] [--save F]
                                                             one conversation that gains a picture a turn (made here: one plain
                                                             colour each) until it holds N. No turn may be refused; past the
                                                             server's limits the oldest pictures are left out in batches, and
                                                             between two batches every turn must resume the prompt cache
  e2e_images.py compare A.json B.json                        the replies saved by two runs, key by key

Every request is greedy (temperature 0, reasoning effort "none", fixed seed), so two servers that compute the same
thing give the same text. `--save` writes {key: reply text} plus the usage and timings of every request.
"""
import argparse, base64, json, os, struct, sys, threading, time, urllib.error, urllib.request, zlib

QUESTIONS = {
    "chart": "Read the text in this chart: give the title and the value of every bar.",
    "ui": "Read the text in this screenshot: which settings page is shown and what does the warning say?",
    "geom": "Describe this image: which shapes are there and what colours do they have?",
    "photo": "Describe this image in two sentences.",
    "tall": "Describe this image. If it holds text, read it.",
    "alpha": "Describe this image. If it holds text, read it.",
}
TEXT_PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number, with a short docstring.",
    "Explain in three sentences why the sky is blue.",
    "List five prime numbers above 100 and say how you checked one of them.",
    "Write a haiku about a river in winter, then explain its imagery in one sentence.",
    "What is the difference between a mutex and a semaphore? Answer in a short paragraph.",
    "Translate to French: 'The train leaves at seven and arrives before noon.'",
]


def data_url(path):
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return "data:%s;base64,%s" % (mime, base64.b64encode(open(path, "rb").read()).decode())


def white_png(w, h):
    """A white PNG of w x h pixels (a small file even when the picture is huge)."""
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff" * (3 * w)) * h)) + chunk(b"IEND", b""))


COLOURS = [("red", (215, 30, 30)), ("green", (30, 165, 50)), ("blue", (30, 60, 215)), ("yellow", (235, 220, 30)),
           ("black", (12, 12, 12)), ("orange", (240, 140, 20)), ("purple", (130, 40, 170))]


def colour_png(turn, w, h):
    """(name, PNG): a w x h picture of one plain colour. The colour's name follows no order a reader could guess, and its shade
    moves a little with the turn, so no two turns (below 125) send the same pixels."""
    name, rgb = COLOURS[(turn * 2654435761 >> 7) % len(COLOURS)]
    px = bytes(c + 3 * j for c, j in zip(rgb, (turn % 5, turn // 5 % 5, turn // 25 % 5)))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return name, (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                  + chunk(b"IDAT", zlib.compress((b"\0" + px * w) * h)) + chunk(b"IEND", b""))


def expect_omitted(tokens, max_pictures, max_tokens):
    """The server's rule (fe_ink::images::omit_count): the smallest multiple of max(1, max_pictures / 4) that makes the newest fit."""
    n, b = len(tokens), max(1, max_pictures // 4)
    fits = lambda k: n - k <= max_pictures and sum(tokens[k:]) <= max_tokens
    for k in range(0, n, b):
        if fits(k):
            return k
    return next(k for k in range((n - 1) // b * b + 1 if n else 0, n + 1) if fits(k))


def image_part(path):
    return {"type": "image_url", "image_url": {"url": data_url(path)}}


def post(url, body, stream=False):
    """Returns (status, reply dict). Streaming: the reply is assembled; `ttft` = seconds to the first content delta."""
    body = dict(body, stream=stream)
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    if not stream:
        j = json.loads(r.read())
        ch = j["choices"][0]
        return r.status, {"content": ch["message"].get("content") or "", "reasoning": ch["message"].get("reasoning_content") or "",
                          "finish": ch.get("finish_reason"), "usage": j.get("usage"), "timings": j.get("timings"), "secs": time.time() - t0}
    out = {"content": "", "reasoning": "", "ttft": None, "usage": None, "timings": None, "finish": None}
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        j = json.loads(line[5:])
        if j.get("usage"):
            out["usage"] = j["usage"]
        if j.get("timings"):
            out["timings"] = j["timings"]
        for ch in j.get("choices", []):
            d = ch.get("delta", {})
            if (d.get("content") or d.get("reasoning_content")) and out["ttft"] is None:
                out["ttft"] = time.time() - t0
            out["content"] += d.get("content") or ""
            out["reasoning"] += d.get("reasoning_content") or ""
            out["finish"] = ch.get("finish_reason") or out["finish"]
    out["secs"] = time.time() - t0
    return r.status, out


def base(max_tokens):
    return {"model": "inkling-2-mini", "temperature": 0, "seed": 7, "max_tokens": max_tokens, "reasoning_effort": "none"}


def ask(url, messages, max_tokens=160, stream=True):
    st, r = post(url, dict(base(max_tokens), messages=messages), stream)
    if st != 200:
        sys.exit("HTTP %d: %s" % (st, json.dumps(r)[:400]))
    return r


def pictures(d):
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    return [(os.path.splitext(f)[0], os.path.join(d, f)) for f in fs]


def question(name):
    return QUESTIONS.get(name.split("_")[0], "Describe this image.")


def show(key, r):
    u, t = r.get("usage") or {}, r.get("timings") or {}
    print("== %s: prompt %s tokens (cached %s), %s new; first token after %s s, %.2f s in all; finish %s" % (
        key, u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens", t.get("cache_n")), u.get("completion_tokens"),
        "%.2f" % r["ttft"] if r.get("ttft") else "-", r["secs"], r.get("finish")))
    print(r["content"].strip())
    sys.stdout.flush()


def cmd_answers(a, res):
    for name, path in pictures(a.images):
        if a.only and not any(name.startswith(o) for o in a.only.split(",")):
            continue
        r = ask(a.url, [{"role": "user", "content": [image_part(path), {"type": "text", "text": a.question or question(name)}]}], a.max_tokens)
        res[name] = r
        show(name, r)


def cmd_cache(a, res):
    pics = dict(pictures(a.images))
    first, other = pics[a.first], pics[a.other]
    q1, q2 = "Read the text in this chart: give the title and the value of every bar.", "Which bar is the tallest, and by how much does it beat the second tallest?"
    m1 = [{"role": "user", "content": [image_part(first), {"type": "text", "text": q1}]}]
    r1 = ask(a.url, m1, a.max_tokens)
    res["turn1"] = r1
    show("turn 1 (%s)" % a.first, r1)
    m2 = m1 + [{"role": "assistant", "content": r1["content"]}, {"role": "user", "content": q2}]
    r2 = ask(a.url, m2, a.max_tokens)
    res["turn2"] = r2
    show("turn 2 (same picture, one more question)", r2)
    m3 = [{"role": "user", "content": [image_part(other), {"type": "text", "text": q1}]}]
    r3 = ask(a.url, m3, a.max_tokens)
    res["other"] = r3
    show("another picture of the same size (%s), same question" % a.other, r3)
    # the first conversation again: its picture must still be told apart from the other one
    r4 = ask(a.url, m2, a.max_tokens)
    res["turn2_again"] = r4
    show("turn 2 again", r4)


def cmd_pair(a, res):
    pics = dict(pictures(a.images))
    names = a.pair.split(",")
    msgs = {n: [{"role": "user", "content": [image_part(pics[n]), {"type": "text", "text": question(n)}]}] for n in names}
    out = {}

    def run(n):
        out[n] = ask(a.url, msgs[n], a.max_tokens)

    ts = [threading.Thread(target=run, args=(n,)) for n in names]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("%d requests at once: %.2f s" % (len(names), time.time() - t0))
    for n in names:
        res["together_" + n] = out[n]
        show("together: " + n, out[n])
    for n in names:
        r = ask(a.url, msgs[n], a.max_tokens)
        res["alone_" + n] = r
        show("alone: " + n, r)
    bad = [n for n in names if res["together_" + n]["content"] != res["alone_" + n]["content"]]
    print("pair: %s" % ("every reply equals the same request alone" if not bad else "DIFFERENT: " + ", ".join(bad)))
    if a.against:
        ref = json.load(open(a.against))
        bad2 = [n for n in names if (res["together_" + n]["content"], res["together_" + n]["reasoning"]) != (ref[n]["content"], ref[n]["reasoning"])]
        print("pair against %s: %s" % (a.against, "every reply sent at once equals the saved one" if not bad2 else "DIFFERENT: " + ", ".join(bad2)))
        bad += bad2
    if bad:
        sys.exit(1)


def cmd_park(a, res):
    pics = dict(pictures(a.images))
    names = a.park.split(",")
    msgs = {n: [{"role": "user", "content": [image_part(pics[n]), {"type": "text", "text": question(n)}]}] for n in names}
    for n in names:
        res["first_" + n] = ask(a.url, msgs[n], a.max_tokens)
        show("turn 1: " + n, res["first_" + n])
    n = names[0]
    m2 = msgs[n] + [{"role": "assistant", "content": res["first_" + n]["content"]}, {"role": "user", "content": "Now say the same in one short sentence."}]
    res["second_" + n] = ask(a.url, m2, a.max_tokens)
    show("turn 2: " + n, res["second_" + n])


def cmd_deep(a, res):
    pics = dict(pictures(a.images))
    filler = " ".join("Line %d of the log: the pump ran for %d minutes and the pressure was %d." % (i, 5 + i * 7 % 50, 90 + i * 13 % 40) for i in range(a.fill_lines))
    m1 = [{"role": "user", "content": filler + "\n\nKeep this log in mind. Reply with the single word OK."}]
    res["deep1"] = ask(a.url, m1, 64)
    show("long text, turn 1", res["deep1"])
    m2 = m1 + [{"role": "assistant", "content": res["deep1"]["content"]},
               {"role": "user", "content": [image_part(pics["ui_1280x720"]), {"type": "text", "text": "What does the warning in this screenshot say?"}]}]
    res["deep2"] = ask(a.url, m2, a.max_tokens)
    show("long text, turn 2: a picture arrives", res["deep2"])
    m3 = m2 + [{"role": "assistant", "content": res["deep2"]["content"]}, {"role": "user", "content": "And which settings page is it?"}]
    res["deep3"] = ask(a.url, m3, a.max_tokens)
    show("long text, turn 3", res["deep3"])
    tool = [{"role": "user", "content": "Take a screenshot of the chart and tell me its title."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": [image_part(pics["chart_800x600"])]}]
    res["tool"] = ask(a.url, tool, a.max_tokens)
    show("a tool result that is a picture", res["tool"])
    # 7 x 1,296 image tokens, then a chart: the prompt-cache mark at 8,192 tokens lies inside the seventh picture
    many = [image_part(pics["photo_1920x1080"])] * 7 + [image_part(pics["chart_800x600"])]
    for key, q in (("many1", "How many pictures did I send? Answer with a number."), ("many2", "What is the title of the chart in the last picture?")):
        res[key] = ask(a.url, [{"role": "user", "content": many + [{"type": "text", "text": q}]}], a.max_tokens)
        show("eight pictures: " + q, res[key])


def cmd_text(a, res):
    for i, p in enumerate(TEXT_PROMPTS):
        r = ask(a.url, [{"role": "user", "content": p}], a.max_tokens)
        res["text%d" % i] = r
        show("text %d" % i, r)
    # two turns
    m = [{"role": "user", "content": TEXT_PROMPTS[0]}, {"role": "assistant", "content": res["text0"]["content"]}, {"role": "user", "content": "Now write the same function in Rust."}]
    res["text_turn2"] = ask(a.url, m, a.max_tokens)
    show("text, turn 2", res["text_turn2"])


def cmd_limits(a, _res):
    pics = dict(pictures(a.images))
    small = pics[a.first]
    cases = [
        ("an http URL", [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]),
        ("a file URL", [{"type": "image_url", "image_url": {"url": "file:///etc/hostname"}}]),
        ("a GIF data URL", [{"type": "image_url", "image_url": {"url": "data:image/gif;base64,R0lGODlhAQABAAAAACw="}}]),
        ("broken base64", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@@"}}]),
        ("not a picture", [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"hello world, not a png").decode()}}]),
        ("a picture with a side above 16384 pixels", [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(white_png(16385, 8)).decode()}}]),
    ]
    bad = 0
    if a.big:
        # above the per-picture token limit: shrunk by the server, not refused
        st, r = post(a.url, dict(base(8), messages=[{"role": "user", "content": [image_part(a.big), {"type": "text", "text": "What is this?"}]}]))
        ok = st == 200
        bad += not ok
        print("%s a picture above the per-picture token limit is taken (shrunk): HTTP %d, prompt %s tokens" % ("ok  " if ok else "FAIL", st, (r.get("usage") or {}).get("prompt_tokens")))
    # one picture more than INK_IMAGE_MAX
    n = a.max_pictures + 1
    st, r = post(a.url, dict(base(8), messages=[{"role": "user", "content": [image_part(small)] * n + [{"type": "text", "text": "What is this?"}]}]))
    if a.overflow == "reject":
        msg = (r.get("error") or {}).get("message", "")
        ok = st == 400 and "INK_IMAGE_MAX" in msg
        print("%s %d pictures (INK_IMAGE_OVERFLOW=reject): HTTP %d: %s" % ("ok  " if ok else "FAIL", n, st, msg[:260]))
    else:
        out = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("images_omitted")
        ok = st == 200 and out == max(1, a.max_pictures // 4)
        print("%s %d pictures: HTTP %d, the oldest %s left out (expected %d), prompt %s tokens" % (
            "ok  " if ok else "FAIL", n, st, out, max(1, a.max_pictures // 4), (r.get("usage") or {}).get("prompt_tokens")))
    bad += not ok
    for name, parts in cases:
        st, r = post(a.url, dict(base(8), messages=[{"role": "user", "content": parts + [{"type": "text", "text": "What is this?"}]}]))
        msg = (r.get("error") or {}).get("message", "")
        ok = st == 400 and msg
        bad += not ok
        print("%s %s: HTTP %d: %s" % ("ok  " if ok else "FAIL", name, st, msg[:260]))
    st, r = post(a.url, dict(base(8), messages=[{"role": "assistant", "content": [image_part(small)]}, {"role": "user", "content": "hi"}]))
    ok = st == 400
    bad += not ok
    print("%s a picture in an assistant message: HTTP %d: %s" % ("ok  " if ok else "FAIL", st, (r.get("error") or {}).get("message", "")[:260]))
    if bad:
        sys.exit(1)


def cmd_overflow(a, res):
    w, h = (int(x) for x in a.size.lower().split("x"))
    each = -(-w // 40) * -(-h // 40)
    q = "Look only at the picture in this message, the newest one. What colour is it? Reply with one word."
    messages, rows, bad, prev = [], [], [], None
    print("one %d x %d picture (%d image tokens) a turn, %d turns; limits %d pictures, %d image tokens a request" % (w, h, each, a.turns, a.max_pictures, a.max_image_tokens))
    print("%4s %8s %6s %8s %8s %5s %7s %7s  %s" % ("turn", "pictures", "out", "prompt", "cached", "new", "ttft s", "all s", "reply (colour sent)"))
    for t in range(1, a.turns + 1):
        colour, png = colour_png(t, w, h)
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": q}]})
        # room for the short reasoning the model does at effort "none" too (it grows with the conversation)
        st, r = post(a.url, dict(base(max(a.max_tokens, 2048)), messages=messages), True)
        if st != 200:
            print("turn %d: HTTP %d: %s" % (t, st, json.dumps(r)[:400]))
            bad.append("turn %d refused (HTTP %d)" % (t, st))
            break
        u = r.get("usage") or {}
        d = u.get("prompt_tokens_details") or {}
        row = {"turn": t, "pictures": t, "omitted": d.get("images_omitted", 0), "expected": expect_omitted([each] * t, a.max_pictures, a.max_image_tokens),
               "prompt": u.get("prompt_tokens"), "cached": d.get("cached_tokens"), "ttft": r.get("ttft"), "secs": r["secs"],
               "reply": r["content"].strip(), "reasoning": r["reasoning"][:600], "finish": r.get("finish"), "new": u.get("completion_tokens"), "colour": colour, "right": r["content"].strip().strip(".").lower() == colour}
        jump = prev is not None and row["omitted"] != prev["omitted"]
        # between two batches the request starts with the previous turn's whole prompt: the cache must hold all of it
        row["resumed"] = None if prev is None or jump else row["cached"] >= prev["prompt"] - 8
        flags = []
        if row["omitted"] != row["expected"]:
            flags.append("LEFT OUT %d, expected %d" % (row["omitted"], row["expected"]))
        if jump:
            flags.append("batch left out: the prompt is processed again")
        if row["resumed"] is False:
            flags.append("CACHE NOT RESUMED (previous prompt %d)" % prev["prompt"])
        if not row["right"]:
            flags.append("WRONG")
        bad += ["turn %d: %s" % (t, x) for x in flags if x[0].isupper() and x != "WRONG"]
        print("%4d %8d %6d %8s %8s %5s %7s %7.2f  %s (%s)%s" % (t, t, row["omitted"], row["prompt"], row["cached"], row["new"], "%.2f" % row["ttft"] if row["ttft"] else "-",
                                                          row["secs"], row["reply"][:40], colour, "  <- " + "; ".join(flags) if flags else ""))
        sys.stdout.flush()
        rows.append(row)
        prev = row
        messages.append({"role": "assistant", "content": r["content"]})
    res["rows"] = rows
    if rows:
        med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
        jumps = [x for i, x in enumerate(rows) if i and x["omitted"] != rows[i - 1]["omitted"]]
        steady = [x for x in rows if x["resumed"] is not None]
        print("pictures left out first changed at turns %s (now %s)" % ([x["turn"] for x in jumps], [x["omitted"] for x in jumps]))
        print("cache resumed on %d of %d turns between changes; first token after %.2f s (median) on those, %s s on the turns of a change" % (
            sum(1 for x in steady if x["resumed"]), len(steady), med([x["ttft"] for x in steady if x["ttft"]]), ", ".join("%.2f" % x["ttft"] for x in jumps if x["ttft"]) or "-"))
        print("replies right: %d of %d; last turn: %r for a %s picture" % (sum(x["right"] for x in rows), len(rows), rows[-1]["reply"], rows[-1]["colour"]))
        if not rows[-1]["right"]:
            bad.append("the reply of the last turn is wrong")
    print("overflow: %s" % ("pass" if not bad else "FAIL: " + "; ".join(bad)))
    if a.save:
        json.dump(res, open(a.save, "w"), indent=1)
    if bad:
        sys.exit(1)


def cmd_compare(a):
    x, y = json.load(open(a.files[0])), json.load(open(a.files[1]))
    bad = 0
    for k in sorted(set(x) | set(y)):
        if k not in x or k not in y:
            print("%-28s only in one file" % k)
            bad += 1
            continue
        same = x[k]["content"] == y[k]["content"] and x[k].get("reasoning", "") == y[k].get("reasoning", "")
        bad += not same
        print("%-28s %s (%d / %d new tokens)" % (k, "same" if same else "DIFFERENT", (x[k].get("usage") or {}).get("completion_tokens", -1), (y[k].get("usage") or {}).get("completion_tokens", -1)))
    print("compare: %d keys, %d different" % (len(set(x) | set(y)), bad))
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["answers", "cache", "pair", "park", "deep", "text", "limits", "overflow", "compare"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url", default="http://127.0.0.1:8040")
    ap.add_argument("--images")
    ap.add_argument("--save")
    ap.add_argument("--only", help="answers: picture name prefixes, comma separated")
    ap.add_argument("--first", default="chart_800x600", help="cache / limits: the picture of the conversation")
    ap.add_argument("--other", default="chart2_800x600", help="cache: another picture of the same size")
    ap.add_argument("--pair", default="chart_800x600,geom_512x512", help="pair: the pictures of the requests sent at once")
    ap.add_argument("--against", help="pair: replies saved by `answers` to compare the replies sent at once with")
    ap.add_argument("--park", default="ui_1280x720,photo_1920x1080,chart_800x600,tall_300x1000", help="park: the pictures of the conversations, in turn")
    ap.add_argument("--fill-lines", type=int, default=1500, help="deep: lines of filler text in the long conversation (about 20 tokens each)")
    ap.add_argument("--big", help="limits: a picture file above the per-picture token limit (the server shrinks it)")
    ap.add_argument("--question", help="answers: this question for every picture")
    ap.add_argument("--max-pictures", type=int, default=64, help="limits / overflow: the server's INK_IMAGE_MAX")
    ap.add_argument("--max-image-tokens", type=int, default=49152, help="overflow: the server's INK_IMAGE_TOKENS")
    ap.add_argument("--overflow", default="drop", choices=["drop", "reject"], help="limits: the server's INK_IMAGE_OVERFLOW")
    ap.add_argument("--turns", type=int, default=90, help="overflow: turns of the conversation (one new picture each)")
    ap.add_argument("--size", default="1280x800", help="overflow: size of the pictures")
    ap.add_argument("--max-tokens", type=int, default=160)
    a = ap.parse_args()
    if a.cmd == "compare":
        return cmd_compare(a)
    res = {}
    {"answers": cmd_answers, "cache": cmd_cache, "pair": cmd_pair, "park": cmd_park, "deep": cmd_deep, "text": cmd_text, "limits": cmd_limits, "overflow": cmd_overflow}[a.cmd](a, res)
    if a.save:
        json.dump(res, open(a.save, "w"), indent=1)


if __name__ == "__main__":
    main()
