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
  e2e_images.py --url URL --images DIR limits   [--big F]    requests the server must refuse with HTTP 400; --big: a picture above
                                                             the per-picture token limit, which the server must shrink and take
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
        ("17 pictures", [image_part(small)] * 17),
        ("a picture with a side above 16384 pixels", [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(white_png(16385, 8)).decode()}}]),
    ]
    bad = 0
    if a.big:
        # above the per-picture token limit: shrunk by the server, not refused
        st, r = post(a.url, dict(base(8), messages=[{"role": "user", "content": [image_part(a.big), {"type": "text", "text": "What is this?"}]}]))
        ok = st == 200
        bad += not ok
        print("%s a picture above the per-picture token limit is taken (shrunk): HTTP %d, prompt %s tokens" % ("ok  " if ok else "FAIL", st, (r.get("usage") or {}).get("prompt_tokens")))
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
    ap.add_argument("cmd", choices=["answers", "cache", "pair", "park", "deep", "text", "limits", "compare"])
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
    ap.add_argument("--max-tokens", type=int, default=160)
    a = ap.parse_args()
    if a.cmd == "compare":
        return cmd_compare(a)
    res = {}
    {"answers": cmd_answers, "cache": cmd_cache, "pair": cmd_pair, "park": cmd_park, "deep": cmd_deep, "text": cmd_text, "limits": cmd_limits}[a.cmd](a, res)
    if a.save:
        json.dump(res, open(a.save, "w"), indent=1)


if __name__ == "__main__":
    main()
