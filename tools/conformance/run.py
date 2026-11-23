#!/usr/bin/env python3
"""Ferrite API conformance runner.

Talks only to an OpenAI-compatible chat completions API (base URL + model id). Every case runs streaming and
non-streaming, thinking on and off (where the case allows). Raw exchanges are saved for failures. Exit code is
non-zero on any failure. Model-specific facts live in a profile file (profiles/*.json), never in this runner.

  python3 run.py --base-url http://127.0.0.1:8000 --profile profiles/qwen3.8-27b.json [--only edits] [--quick]
"""
import argparse, json, os, re, sys, threading, time, urllib.request, urllib.error, socket, traceback

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- tool schemas

def T(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}

PI_TOOLS = [
    T("read", "Read the contents of a file.", {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"]),
    T("write", "Write content to a file. Creates the file if it does not exist, overwrites if it does.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    T("edit", "Edit a file by replacing exact text. Each oldText must match exactly once.",
      {"path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object", "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}}, "required": ["oldText", "newText"]}}}, ["path", "edits"]),
    T("bash", "Execute a bash command.", {"command": {"type": "string"}, "timeout": {"type": "number"}}, ["command"]),
]
GENERIC_TOOLS = [
    T("create_event", "Create a calendar event.", {
        "title": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}},
        "all_day": {"type": "boolean"}, "priority": {"type": "integer"}, "visibility": {"type": "string", "enum": ["public", "private", "team"]},
        "location": {"type": "object", "properties": {"room": {"type": "string"}, "floor": {"type": "integer"}, "geo": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}}}}},
        "reminder_minutes": {"type": "number"}}, ["title", "attendees"]),
    T("get_weather", "Weather for a city.", {"city": {"type": "string"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in PI_TOOLS + GENERIC_TOOLS}
SYS = ("You are an expert coding assistant operating inside a coding agent harness. You help users by reading files, executing commands, "
       "editing code and writing new files. Use the tools to act; be concise.")


def validate(v, sc, path="$"):
    t = sc.get("type")
    if "enum" in sc and v not in sc["enum"]:
        return "%s: %r not in enum" % (path, v)
    if t == "object":
        if not isinstance(v, dict): return "%s: expected object, got %s" % (path, type(v).__name__)
        for k in sc.get("required", []):
            if k not in v: return "%s: missing %s" % (path, k)
        for k, x in v.items():
            if k in sc.get("properties", {}):
                e = validate(x, sc["properties"][k], path + "." + k)
                if e: return e
    elif t == "array":
        if not isinstance(v, list): return "%s: expected array, got %s" % (path, type(v).__name__)
        for i, x in enumerate(v):
            e = validate(x, sc.get("items", {}), "%s[%d]" % (path, i))
            if e: return e
    elif t == "string" and not isinstance(v, str): return "%s: expected string, got %s" % (path, type(v).__name__)
    elif t == "boolean" and not isinstance(v, bool): return "%s: expected boolean, got %s" % (path, type(v).__name__)
    elif t in ("integer", "number"):
        if isinstance(v, bool) or not isinstance(v, (int, float)): return "%s: expected %s, got %s" % (path, t, type(v).__name__)
        if t == "integer" and isinstance(v, float) and v != int(v): return "%s: expected integer, got %r" % (path, v)
    return None


# ---------------------------------------------------------------- client

class Result:
    pass


class Client:
    def __init__(self, base, model, timeout=3600, defaults=None, seed=None, header="x-sampling-effective"):
        self.base, self.model, self.timeout = base.rstrip("/"), model, timeout
        # --sampling: fields added to every request that does not set them itself; --seed: a different seed for every such request
        self.defaults, self.seed, self.n_sent, self.header = defaults or {}, seed, 0, header

    def chat(self, messages, *, stream, think=None, tools=None, max_tokens=None, extra=None, tool_choice=None, raw_limit=200000):
        body = {"model": self.model, "messages": messages, "stream": stream}
        if max_tokens is not None: body["max_tokens"] = max_tokens
        if tools: body["tools"] = tools
        if tool_choice is not None: body["tool_choice"] = tool_choice
        if think is not None: body["chat_template_kwargs"] = {"enable_thinking": think}
        if stream: body["stream_options"] = {"include_usage": True}
        if extra: body.update(extra)
        for k, v in self.defaults.items(): body.setdefault(k, v)
        if self.seed is not None and "seed" not in body:
            body["seed"] = self.seed * 100003 + self.n_sent
        self.n_sent += 1
        return self.send(body, raw_limit)

    def send(self, body, raw_limit=200000):
        """POST a ready request body (a captured one, or the one chat() built) and collect the reply."""
        stream = bool(body.get("stream"))
        r = Result(); r.body = body; r.status = 200; r.error = None
        r.content = ""; r.reasoning = ""; r.calls = []; r.finish = None; r.usage = None; r.raw = []; r.max_gap = 0.0
        r.ttft = None; r.keepalives = 0; r.http_body = None; r.n_chunks = 0; r.effective = None
        t0 = time.time()
        req = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            r.status = e.code; r.http_body = e.read().decode(errors="replace"); r.wall = time.time() - t0
            return r
        # the sampling the server says it ran the request with (a JSON object in a response header), if it says so
        try: r.effective = json.loads(resp.headers.get(self.header) or "null")
        except ValueError: r.effective = None
        last = time.time()
        if not stream:
            raw = resp.read().decode()
            r.wall = time.time() - t0; r.max_gap = r.wall
            r.raw.append(raw[:raw_limit])
            d = json.loads(raw)
            ch = d["choices"][0]; m = ch["message"]
            r.content = m.get("content") or ""; r.reasoning = m.get("reasoning_content") or ""
            for tc in m.get("tool_calls") or []:
                r.calls.append({"id": tc.get("id"), "name": tc["function"]["name"], "args": tc["function"]["arguments"], "deltas": 1})
            r.finish = ch.get("finish_reason"); r.usage = d.get("usage")
            return r
        calls = {}
        for line in resp:
            now = time.time(); r.max_gap = max(r.max_gap, now - last); last = now
            line = line.decode().rstrip("\n")
            if sum(len(x) for x in r.raw) < raw_limit: r.raw.append(line)
            if line.startswith(":"): r.keepalives += 1; continue
            if not line.startswith("data: "): continue
            if line.endswith("[DONE]"): break
            d = json.loads(line[6:]); r.n_chunks += 1
            if d.get("usage"): r.usage = d["usage"]
            if d.get("error"): r.error = d["error"]
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                if r.ttft is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls")): r.ttft = now - t0
                r.content += dl.get("content") or ""; r.reasoning += dl.get("reasoning_content") or ""
                for tc in dl.get("tool_calls") or []:
                    c = calls.setdefault(tc["index"], {"id": tc.get("id"), "name": "", "args": "", "deltas": 0})
                    f = tc.get("function", {})
                    c["name"] += f.get("name") or ""
                    if f.get("arguments"): c["args"] += f["arguments"]; c["deltas"] += 1
                    if tc.get("id") and not c["id"]: c["id"] = tc["id"]
                if ch.get("finish_reason"): r.finish = ch["finish_reason"]
        r.calls = [calls[k] for k in sorted(calls)]
        r.wall = time.time() - t0
        return r


def cached_tokens(usage):
    if not usage: return None
    d = usage.get("prompt_tokens_details") or {}
    return d.get("cached_tokens", usage.get("cached_tokens"))


# ---------------------------------------------------------------- fake file system / agent loop

class FS:
    def __init__(self): self.files = {}

    def run(self, name, args):
        err = validate(args, SCHEMAS.get(name, {"type": "object"}))
        if err: return "Validation failed for tool %s: %s" % (name, err), False
        if name == "write": self.files[args["path"]] = args["content"]; return "Successfully wrote %d bytes to %s" % (len(args["content"]), args["path"]), True
        if name == "read":
            if args["path"] not in self.files: return "Error: no such file", False
            return self.files[args["path"]][: args.get("limit") or 20000], True
        if name == "edit":
            s = self.files.get(args["path"])
            if s is None: return "Error: no such file", False
            for i, e in enumerate(args["edits"]):
                if s.count(e["oldText"]) != 1: return "Error: edits[%d].oldText must match exactly once (found %d)" % (i, s.count(e["oldText"])), False
            for e in args["edits"]: s = s.replace(e["oldText"], e["newText"], 1)
            self.files[args["path"]] = s
            return "Successfully replaced %d block(s)" % len(args["edits"]), True
        if name == "bash": return "(bash not available in the conformance sandbox)", True
        return "unknown tool", False


def parse_args(c):
    try: return json.loads(c["args"] or "{}"), None
    except Exception as e: return None, "arguments are not JSON: %s" % e


def assistant_msg(r, send_reasoning=True):
    m = {"role": "assistant", "content": r.content or None}
    if send_reasoning and r.reasoning: m["reasoning_content"] = r.reasoning
    if r.calls: m["tool_calls"] = [{"id": c["id"] or "call_%d" % i, "type": "function", "function": {"name": c["name"], "arguments": c["args"]}} for i, c in enumerate(r.calls)]
    return m


# ---------------------------------------------------------------- cases

class Ctx:
    def __init__(self, client, profile, stream, think, args):
        self.c, self.p, self.stream, self.think, self.args = client, profile, stream, think, args
        self.notes = []

    def chat(self, messages, **kw):
        kw.setdefault("stream", self.stream); kw.setdefault("think", self.think)
        return self.c.chat(messages, **kw)


class Fail(Exception):
    def __init__(self, msg, results=()):
        super().__init__(msg); self.results = list(results)


def need(cond, msg, *results):
    if not cond: raise Fail(msg, results)


# Markup of each tool-call format (profile key `tool_format`). When the server's parser loses track of a call, this markup ends up inside
# an argument value: no argument may contain a tag of the profile's own format, unless the request asked for text that holds it.
FORMAT_TAGS = {
    "dsml": ["\uff5cDSML\uff5c"],
    "qwen": ["<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>"],
    "inkling": ["<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>", "<|end_message|>", "<|content_model_end_sampling|>"],
}
# Text that looks like each format's markup, to be written to a file as ordinary content (case special_content, every profile gets all of them).
LOOKALIKE = {
    "dsml": "<\uff5cDSML\uff5c parameter name=\"x\">not a tag</\uff5cDSML\uff5c parameter>",
    "qwen": "<tool_call><function=x><parameter=y>1</parameter></function></tool_call> <div class=\"a\">&amp;</div>",
    "inkling": "<|message_model|>x<|content_invoke_tool_json|>{\"name\":\"x\",\"args\":{}}<|end_message|> <|content_text|>not a message<|end_message|>",
}


def strings_of(v):
    """Every string inside a parsed JSON value, joined."""
    if isinstance(v, str): return v
    if isinstance(v, dict): return "\n".join(strings_of(x) for x in v.values())
    if isinstance(v, list): return "\n".join(strings_of(x) for x in v)
    return ""


def check_calls(ctx, r, names=None, expect=""):
    """Calls are valid JSON, match their schema, and hold no tag text of the profile's tool format (`expect`: text the request asked
    for; tags it contains are allowed)."""
    need(r.status == 200, "HTTP %s %s" % (r.status, (r.http_body or "")[:200]), r)
    need(not r.error, "stream error %s" % (r.error,), r)
    for c in r.calls:
        a, e = parse_args(c); need(a is not None, e, r)
        err = validate(a, SCHEMAS.get(c["name"], {"type": "object"})); need(err is None, "%s %s" % (c["name"], err), r)
        text = strings_of(a)
        for tag in FORMAT_TAGS.get(ctx.p.get("tool_format", ""), []):
            need(tag not in text or tag in expect, "tool-call tag text %r inside the arguments of %s" % (tag, c["name"]), r)
    if names is not None: need([c["name"] for c in r.calls] == names, "calls %s != %s" % ([c["name"] for c in r.calls], names), r)


def case_basic_call(ctx):
    r = ctx.chat([{"role": "system", "content": SYS}, {"role": "user", "content": "Save the text hello world to hello.txt."}], tools=PI_TOOLS, max_tokens=8000)
    check_calls(ctx, r, ["write"])
    need(r.finish == "tool_calls", "finish_reason %r" % r.finish, r)
    a, _ = parse_args(r.calls[0]); need(a["path"].endswith("hello.txt"), "path %r" % a["path"], r)
    need("hello world" in a["content"], "content %r" % a["content"][:80], r)
    if ctx.stream:
        need(r.usage and r.usage.get("completion_tokens", 0) > 0, "no usage in the stream", r)
        need(cached_tokens(r.usage) is not None, "no cached token count in usage", r)


# two addresses in the reserved example.com domain, put together here so the file holds no literal address
ATTENDEES = [n + "@" + "example.com" for n in ("alice", "bob")]


def case_generic_schemas(ctx):
    r = ctx.chat([{"role": "user", "content": "Create an event titled 'Design review' for attendees " + " and ".join(ATTENDEES) + ", all day, priority 2, visibility team, in room B12 on floor 3 at lat 52.5 lon 13.4, reminder 7.5 minutes before."}], tools=GENERIC_TOOLS, max_tokens=8000)
    check_calls(ctx, r, ["create_event"])
    a, _ = parse_args(r.calls[0])
    need(a.get("attendees") == ATTENDEES, "attendees %r" % (a.get("attendees"),), r)
    need(a.get("all_day") is True and a.get("priority") == 2 and a.get("visibility") == "team", "scalar types %r" % a, r)
    need(isinstance(a.get("location"), dict), "location not an object: %r" % (a.get("location"),), r)


def case_parallel_calls(ctx):
    r = ctx.chat([{"role": "user", "content": "What is the weather in Paris and in Tokyo? Call get_weather for both cities in the same reply."}], tools=GENERIC_TOOLS, max_tokens=8000)
    check_calls(ctx, r)
    cities = sorted((parse_args(c)[0] or {}).get("city", "") for c in r.calls)
    need(len(r.calls) == 2 and any("Paris" in c for c in cities) and any("Tokyo" in c for c in cities), "calls %s" % cities, r)
    need(r.finish == "tool_calls", "finish %r" % r.finish, r)


def case_preamble(ctx):
    r = ctx.chat([{"role": "user", "content": "First say in one sentence what you are going to do, then write 'ok' to note.txt with the write tool."}], tools=PI_TOOLS, max_tokens=8000)
    check_calls(ctx, r, ["write"])
    need(r.finish == "tool_calls", "finish %r" % r.finish, r)
    txt = r.content.strip()
    if txt:
        half = len(txt) // 2
        need(not (len(txt) > 40 and txt[:half].strip() == txt[half:].strip()), "preamble duplicated: %r" % txt[:120], r)


def case_tool_choice(ctx):
    r = ctx.chat([{"role": "user", "content": "Tell me about Paris."}], tools=GENERIC_TOOLS, max_tokens=8000, tool_choice="required")
    check_calls(ctx, r); need(len(r.calls) >= 1, "tool_choice required: no call", r)
    r = ctx.chat([{"role": "user", "content": "Tell me about Paris."}], tools=GENERIC_TOOLS, max_tokens=8000, tool_choice={"type": "function", "function": {"name": "get_weather"}})
    check_calls(ctx, r, ["get_weather"])
    r = ctx.chat([{"role": "user", "content": "Say hi in three words."}], tools=GENERIC_TOOLS, max_tokens=8000, tool_choice="auto")
    need(r.status == 200 and r.finish in ("stop", "tool_calls"), "auto: %r" % r.finish, r)


SPECIAL = {
    "quotes": 'He said "hello" and she said \'hi\' and `code`.',
    "backslashes": 'C:\\Users\\x\\file.txt and \\n literal and \\\\ double',
    "unicode": "caf\u00e9 \u00fcber na\u00efve \u65e5\u672c\u8a9e \U0001F600 \u2192",
    "crlf": "line1\r\nline2\r\nline3",
}


# The text to copy sits between two full lines that cannot be mistaken for content (the old `<<<BEGIN` / `END>>>` markers were copied
# into the file by some models: a suite fault, not a server one).
MARK_A = "=========== the content starts on the next line ==========="
MARK_B = "=========== the content ended on the previous line ==========="


def case_special_content(ctx):
    """Every text is tried; the failure names each text that did not arrive byte for byte (the first fault per text)."""
    texts = dict(SPECIAL)
    for fmt in sorted(LOOKALIKE): texts["lookalike_" + fmt] = LOOKALIKE[fmt]
    bad = []; bad_results = []
    for key, text in texts.items():
        r = ctx.chat([{"role": "user", "content": "Use the write tool to save a file named special.txt. Its content must be EXACTLY the text between the two marker lines below, "
                       "byte for byte, with nothing added and nothing changed. The two marker lines (the ones that start and end with equals signs) are not part of the content.\n%s\n%s\n%s" % (MARK_A, text, MARK_B)}],
                     tools=PI_TOOLS, max_tokens=8000)
        try:
            check_calls(ctx, r, ["write"], expect=text)
            a, _ = parse_args(r.calls[0])
            got = a["content"].strip("\n")
            need(got == text or got.replace("\r\n", "\n") == text.replace("\r\n", "\n"), "content differs: %r vs %r" % (got[:160], text[:160]), r)
        except Fail as e:
            bad.append("%s: %s" % (key, e)); bad_results.append(r)
    ctx.notes.append("special texts written byte for byte: %d of %d%s" % (len(texts) - len(bad), len(texts), "".join("\n      " + b for b in bad)))
    need(not bad, "%d of %d texts: %s" % (len(bad), len(texts), "; ".join(bad[:2])), *bad_results)


def case_cut_off(ctx):
    r = ctx.chat([{"role": "user", "content": "Write a long HTML page (1000 lines) to page.html with the write tool."}], tools=PI_TOOLS, max_tokens=300, think=False)
    need(r.status == 200, "HTTP %s" % r.status, r)
    need(r.finish == "length", "finish %r, expected length" % r.finish, r)
    for c in r.calls:
        a, e = parse_args(c); need(a is not None, "cut-off call has invalid JSON arguments: %s" % e, r)


def long_write(ctx, lines):
    fs = FS(); msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": "Create index.html: a complete self-contained landing page for a bakery with an embedded stylesheet and script, about %d lines. Save it with the write tool." % lines}]
    r = ctx.chat(msgs, tools=PI_TOOLS, max_tokens=48000)
    check_calls(ctx, r)
    need(r.finish == "tool_calls" and r.calls and r.calls[0]["name"] == "write", "no write call (finish %r, %d calls, content %r)" % (r.finish, len(r.calls), r.content[:80]), r)
    a, _ = parse_args(r.calls[0]); n = len(a["content"].splitlines())
    need(n >= lines * 0.6, "file has %d lines, wanted about %d" % (n, lines), r)
    need("</html>" in a["content"], "file is truncated (no </html>)", r)
    if ctx.stream: need(r.calls[0]["deltas"] > 5, "arguments not streamed incrementally (%d deltas)" % r.calls[0]["deltas"], r)
    return fs, msgs, r


def case_long_write_1k(ctx): long_write(ctx, ctx.p.get("long_write_lines", [1000, 2500])[0])


def case_long_write_big(ctx):
    """The biggest file (profile long_write_lines[-1], about 2,500 lines). What counts is a complete file on disk through valid calls:
    one write call of any length that ends properly (an overshoot is fine), or a staged write (a first write, then further write /
    edit calls over the next turns that complete it). A call cut off at max_tokens, an invalid call, or a file that never gets its
    closing tag fails."""
    lines = ctx.p.get("long_write_lines", [1000, 2500])[-1]
    fs = FS(); msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": "Create index.html: a complete self-contained landing page for a bakery with an embedded stylesheet and script, about %d lines. Save it with the write tool." % lines}]
    rounds = []; results = []

    def state():
        best = max(fs.files.values(), key=len) if fs.files else ""
        return len(best.splitlines()), "</html>" in best
    cap = ctx.p.get("long_write_max_tokens", 48000)   # per turn; a model that writes far more than it was asked needs room to end its call
    for rnd in range(ctx.p.get("long_write_rounds", 6)):
        r = ctx.chat(msgs, tools=PI_TOOLS, max_tokens=cap); results.append(r)
        check_calls(ctx, r)
        out = (r.usage or {}).get("completion_tokens")
        need(r.finish != "length", "round %d was cut off at max_tokens (%s tokens, %d calls): the call never ended (rounds so far: %s)" % (rnd + 1, out, len(r.calls), "; ".join(rounds) or "none"), *results[-2:])
        msgs.append(assistant_msg(r))
        for c in r.calls:
            a, _ = parse_args(c); res, ok = fs.run(c["name"], a)
            msgs.append({"role": "tool", "tool_call_id": c["id"] or "call_x", "content": res})
        n, closed = state()
        rounds.append("round %d: %s, %s tokens, calls %s -> %d lines%s" % (rnd + 1, r.finish, out, [c["name"] for c in r.calls], n, ", closed" if closed else ""))
        if ctx.stream and rnd == 0 and r.calls: need(r.calls[0]["deltas"] > 5, "arguments not streamed incrementally (%d deltas)" % r.calls[0]["deltas"], r)
        if closed and n >= lines * 0.6: break
        if not r.calls:
            # the model stopped without finishing the file: one nudge, as a harness user would
            if rnd >= 3: break
            msgs.append({"role": "user", "content": "The file is not complete yet. Continue until index.html is finished, using the tools."})
    n, closed = state()
    ctx.notes.append("long write, %d lines asked: %s" % (lines, " | ".join(rounds)))
    need(fs.files, "no file was written (%s)" % "; ".join(rounds), *results[-2:])
    need(closed, "the file is truncated after %d round(s) (no </html>, %d lines)" % (len(rounds), n), *results[-2:])
    need(n >= lines * 0.6, "the file has %d lines, wanted about %d" % (n, lines), *results[-2:])


def case_sampling_profile(ctx):
    """The server's per-mode sampling defaults (profile key `sampling_profiles`: {"thinking": {...}, "non_thinking": {...}}): a request
    that sends no sampling field must run with the profile of its mode, and a field the client sends must win, the others still coming
    from the profile. Checked through the API: the server reports the sampling it ran with in a response header (profile key
    `effective_sampling_header`, default x-sampling-effective, a JSON object). Skipped when the profile has no `sampling_profiles`.
    With profile key `greedy_keeps_fast_path` true: a client that sends temperature 0 and no penalty must get no penalty from the
    profile (presence 0, frequency 0, repetition 1: the request stays greedy), while a penalty it sends with temperature 0 applies."""
    want = ctx.p.get("sampling_profiles")
    if not want:
        ctx.notes.append("skipped: the profile has no sampling_profiles"); return
    exp = want["thinking" if ctx.think else "non_thinking"]
    msgs = [{"role": "user", "content": "Say hi in three words."}]
    same = lambda a, b: a is not None and abs(float(a) - float(b)) < 1e-4

    def ask(extra):
        body = {"model": ctx.c.model, "messages": msgs, "stream": ctx.stream, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": ctx.think}}
        body.update(extra)
        r = ctx.c.send(body)
        need(r.status == 200, "HTTP %s %s" % (r.status, (r.http_body or "")[:200]), r)
        need(isinstance(r.effective, dict), "no %s response header with the effective sampling" % ctx.c.header, r)
        return r
    r = ask({})
    ctx.notes.append("effective sampling without client fields: %s" % json.dumps({k: r.effective.get(k) for k in exp}))
    wrong = ["%s: %r, profile says %r" % (k, r.effective.get(k), v) for k, v in exp.items() if not same(r.effective.get(k), v)]
    need(not wrong, "a request without sampling fields did not get the %s profile: %s" % ("thinking" if ctx.think else "non-thinking", "; ".join(wrong)), r)
    # the client's fields win one by one (0 is a value, not "unset")
    sent = {"temperature": 0, "presence_penalty": 0.25}
    r = ask(sent)
    wrong = ["%s: %r, the client sent %r" % (k, r.effective.get(k), v) for k, v in sent.items() if not same(r.effective.get(k), v)]
    wrong += ["%s: %r, profile says %r" % (k, r.effective.get(k), v) for k, v in exp.items() if k not in sent and not same(r.effective.get(k), v)]
    need(not wrong, "client fields must win and the rest come from the profile: %s" % "; ".join(wrong), r)
    ctx.notes.append("with temperature 0 and presence_penalty 0.25 sent: %s" % json.dumps({k: r.effective.get(k) for k in exp}))
    if ctx.p.get("greedy_keeps_fast_path"):
        # temperature 0 from the client is a request for greedy decoding: the profile adds no penalty to it
        neutral = {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
        r = ask({"temperature": 0})
        wrong = ["%s: %r, wanted %r" % (k, r.effective.get(k), v) for k, v in neutral.items() if not same(r.effective.get(k, v), v)]
        need(same(r.effective.get("temperature"), 0), "temperature 0 was sent, the server ran with %r" % r.effective.get("temperature"), r)
        need(not wrong, "a temperature-0 request without penalties must not get the profile's penalties: %s" % "; ".join(wrong), r)
        ctx.notes.append("with temperature 0 only: %s" % json.dumps({k: r.effective.get(k) for k in ("temperature", *neutral)}))
        # ... and the penalties the client does send with temperature 0 still apply
        sent = {"temperature": 0, "presence_penalty": 1.5, "repetition_penalty": 1.05}
        r = ask(sent)
        wrong = ["%s: %r, the client sent %r" % (k, r.effective.get(k), v) for k, v in sent.items() if not same(r.effective.get(k), v)]
        need(not wrong, "penalties sent with temperature 0 must apply: %s" % "; ".join(wrong), r)


WANDER_LINES = ["The river keeps its silver secrets under stone.", "A lantern swings above the harbour wall at dusk.", "Winter folds the orchard into quiet rows of white.",
                "The old clock in the hallway forgets the hour again.", "Somewhere past the ridge a single engine hums.", "Salt wind carries the gulls across the slate roofs."]


def case_wandering_loop(ctx):
    """A reply that keeps bringing back the same few sentences in changing order (no verbatim cycle, so an exact-repeat detector does
    not see it) must be ended by the server well before the token cap. The prompt asks for exactly that text: six lines, again and
    again, each round in another order, 80 rounds (about 5,000 tokens). Passes when the reply ends by itself or is ended by the server
    below `wander_stop_tokens` (profile, default 3,600); fails when it runs on to the cap. At temperature 0 and at the default sampling."""
    cap = ctx.p.get("wander_max_tokens", 8000); limit = ctx.p.get("wander_stop_tokens", 3600)
    ask = "Here are six lines:\n%s\n\nWrite all six lines out again and again for 80 rounds, each round in a different order from the round before. No numbering, no round headers, no blank lines, no other text." % "\n".join(WANDER_LINES)
    det = []; bad = []; results = []
    for tname, extra in (("temp 0", {"temperature": 0}), ("default sampling", None)):
        r = ctx.chat([{"role": "user", "content": ask}], max_tokens=cap, think=False, extra=extra); results.append(r)
        need(r.status == 200, "HTTP %s" % r.status, r)
        out = (r.usage or {}).get("completion_tokens") or 0
        det.append("%s: finish %s after %d tokens%s" % (tname, r.finish, out, (", exact cycle of %d chars at the end" % tail_cycle(r.content)) if tail_cycle(r.content) else ""))
        if out >= limit: bad.append(det[-1])
    ctx.notes.append("wandering reply (cap %d, must end below %d): %s" % (cap, limit, " | ".join(det)))
    need(not bad, "a wandering reply was not ended: %s" % "; ".join(bad), *results)


def case_edits(ctx):
    fs, msgs, r = long_write(ctx, 400)
    a, _ = parse_args(r.calls[0]); fs.files[a["path"]] = a["content"]
    msgs.append(assistant_msg(r)); msgs.append({"role": "tool", "tool_call_id": r.calls[0]["id"] or "call_0", "content": "Successfully wrote file"})
    asks = ["Use the edit tool directly (the file text is in the conversation): replace the first <style> block's body rule with a larger rule set of at least 40 lines of CSS.",
            "Use the edit tool directly: replace the <footer> element with a longer footer with four columns of links.",
            "Use the edit tool directly: replace the contents of the script with a more thorough version of at least 30 lines."]
    ok_edits = 0
    for k, ask in enumerate(asks):
        msgs.append({"role": "user", "content": ask})
        for rnd in range(4):
            r = ctx.chat(msgs, tools=PI_TOOLS, max_tokens=32000)
            check_calls(ctx, r)
            msgs.append(assistant_msg(r))
            if not r.calls: break
            for c in r.calls:
                a, _ = parse_args(c); res, ok = fs.run(c["name"], a)
                need(ok or c["name"] != "edit" or "must match exactly once" in res, "edit rejected: %s" % res, r)
                if c["name"] == "edit" and ok: ok_edits += 1
                msgs.append({"role": "tool", "tool_call_id": c["id"] or "call_x", "content": res})
            if ok_edits > k: break
    need(ok_edits >= 3, "only %d of 3 edits accepted" % ok_edits)


def case_multiturn_cache(ctx):
    fs = FS(); msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": "Create notes.md with a 40-line project overview (any topic) using the write tool."}]
    table = []
    asks = ["Now read notes.md.", "Use edit to change the first line to a new title.", "Read notes.md again.", "Use edit to append a section titled Next Steps by replacing the last line with itself plus the section.", "Read notes.md once more and tell me how many lines it has."]
    ai = 0; turns = 0
    for rnd in range(10):
        r = ctx.chat(msgs, tools=PI_TOOLS, max_tokens=24000)
        check_calls(ctx, r)
        pt = (r.usage or {}).get("prompt_tokens"); ct = cached_tokens(r.usage)
        table.append((pt, ct)); turns += 1
        msgs.append(assistant_msg(r, send_reasoning=(rnd % 2 == 0)))   # reasoning sent back on even rounds, dropped on odd rounds
        if r.calls:
            for c in r.calls:
                a, _ = parse_args(c); res, ok = fs.run(c["name"], a)
                msgs.append({"role": "tool", "tool_call_id": c["id"] or "call_x", "content": res})
        else:
            if ai >= len(asks): break
            msgs.append({"role": "user", "content": asks[ai]}); ai += 1
        if turns >= 8: break
    ctx.notes.append("cached/prompt per round: " + ", ".join("%s/%s" % (c, p) for p, c in table))
    hits = [c for p, c in table[1:] if p and c is not None and c >= 0.5 * p]
    need(len(table) >= 4, "conversation ended after %d rounds" % len(table))
    need(table[1][1] is not None, "no cached token count reported")
    if ctx.p.get("prompt_cache", True):
        need(len(hits) >= 0.6 * (len(table) - 1), "prefix cache hit in only %d of %d follow-up rounds" % (len(hits), len(table) - 1))
    else:
        ctx.notes.append("profile says prompt_cache false: the cache hit rate is not checked (the loop itself - valid calls, results and history rendered back every round - is)")


STOP_PROMPTS = ["What is 17 times 23?", "Name three primary colors.", "Explain what a mutex is in two sentences.", "Write a haiku about the sea.", "Translate 'good morning' into French and Spanish.",
    "What does HTTP 404 mean?", "Give me a one-line Python expression that reverses a string.", "Summarize the plot of Romeo and Juliet in three sentences.", "What is the capital of Australia?", "List five prime numbers.",
    "Explain big-O notation briefly.", "Write a SQL query that counts rows in a table named users.", "What is the difference between TCP and UDP? Keep it short.", "Convert 100 Fahrenheit to Celsius.", "Write a bash one-liner that lists files by size.",
    "Who wrote Pride and Prejudice?", "Say hello in Japanese.", "What is a closure in JavaScript? One paragraph.", "Name two sorting algorithms and their complexity.", "Give a regex that matches an email address (simple)."]


def case_stopping(ctx):
    cap = 4000; bad = []; bad_results = []
    for p in STOP_PROMPTS:
        r = ctx.chat([{"role": "user", "content": p}], max_tokens=cap)
        need(r.status == 200, "HTTP %s" % r.status, r)
        used = (r.usage or {}).get("completion_tokens", 0)
        if r.finish == "length" or used > cap * ctx.p.get("stop_prompts_max_fraction", 0.5) or not (r.content.strip() or r.reasoning.strip()):
            bad.append("%r: finish %s tokens %s content %d" % (p, r.finish, used, len(r.content)))
            bad_results.append(r)
    # the raw exchange of every failing prompt is saved (request, content, reasoning, usage, raw reply)
    need(not bad, "%d of %d prompts ran away or were empty: %s" % (len(bad), len(STOP_PROMPTS), "; ".join(bad[:3])), *bad_results)


# ---------------------------------------------------------------- prompt-set cases (prompts/*.json next to this file)

def load_prompts(name):
    return json.load(open(os.path.join(HERE, "prompts", name + ".json")))


def stop_fault(r, tools):
    """None when the reply stopped by itself with a non-empty answer or a valid call; else what is wrong with it."""
    if r.status != 200: return "HTTP %s %s" % (r.status, (r.http_body or "")[:120])
    if r.error: return "stream error %s" % (r.error,)
    if r.finish == "length": return "did not stop by itself: finish_reason length after %s tokens (the cap, or the server ended a loop)" % (r.usage or {}).get("completion_tokens")
    if r.calls:
        schemas = {t["function"]["name"]: t["function"].get("parameters", {"type": "object"}) for t in (tools or [])}
        for c in r.calls:
            if c["name"] not in schemas: return "call to %r, which was not offered" % c["name"]
            a, e = parse_args(c)
            if a is None: return e
            err = validate(a, schemas[c["name"]])
            if err: return "%s %s" % (c["name"], err)
        return None if r.finish == "tool_calls" else "calls with finish_reason %r" % r.finish
    if not r.content.strip(): return "empty answer (finish %s, %s tokens, %d reasoning chars)" % (r.finish, (r.usage or {}).get("completion_tokens"), len(r.reasoning))
    return None if r.finish == "stop" else "finish_reason %r" % r.finish


def stop_set(ctx, runs, cap):
    """runs: [(label, messages, tools)], each at temperature 0 and at the server's default sampling. Returns (bad lines, bad results, n)."""
    bad = []; bad_results = []; n = 0
    for label, msgs, tools in runs:
        for tname, extra in (("temp 0", {"temperature": 0}), ("default sampling", None)):
            r = ctx.chat(msgs, tools=tools, max_tokens=cap, extra=extra); n += 1
            f = stop_fault(r, tools)
            if f:
                bad.append("%s, %s: %s" % (label, tname, f)); bad_results.append(r)
    return bad, bad_results, n


def case_coding_stop(ctx):
    """Short coding prompts the way a coding harness sends them (a system prompt always; alone, with tools offered, and tool tasks):
    every reply must stop by itself with a non-empty answer or a valid call."""
    d = load_prompts("coding"); cap = ctx.p.get("stop_set_max_tokens", 4096)
    sysm = {"role": "system", "content": d["system"]}
    runs = [("prompt %d" % i, [sysm, {"role": "user", "content": p}], None) for i, p in enumerate(d["prompts"])]
    runs += [("prompt %d + tools" % i, [sysm, {"role": "user", "content": p}], d["tools"]) for i, p in enumerate(d["prompts"])]
    runs += [("tool prompt %d" % i, [sysm, {"role": "user", "content": p}], d["tools"]) for i, p in enumerate(d["tool_prompts"])]
    bad, bad_results, n = stop_set(ctx, runs, cap)
    ctx.notes.append("coding prompts that did not stop with an answer or a valid call: %d/%d (cap %d tokens)%s" % (len(bad), n, cap, "".join("\n      " + b for b in bad)))
    need(not bad, "%d of %d coding requests did not stop with a non-empty answer or a valid call: %s" % (len(bad), n, "; ".join(bad[:3])), *bad_results)


def case_chat_stop(ctx):
    """Short general chat prompts, without and with a system prompt. Informational: the rate is reported, the run does not fail on it
    (a coding model may ramble on open chat prompts; the raw exchanges are saved for a look)."""
    d = load_prompts("chat"); cap = ctx.p.get("stop_set_max_tokens", 4096)
    runs = [("prompt %d" % i, [{"role": "user", "content": p}], None) for i, p in enumerate(d["prompts"])]
    runs += [("prompt %d + system" % i, [{"role": "system", "content": d["system"]}, {"role": "user", "content": p}], None) for i, p in enumerate(d["prompts"])]
    bad, bad_results, n = stop_set(ctx, runs, cap)
    ctx.notes.append("chat prompts that did not stop with an answer: %d/%d (cap %d tokens)%s" % (len(bad), n, cap, "".join("\n      " + b for b in bad)))
    need(not bad, "%d of %d chat requests did not stop with a non-empty answer: %s" % (len(bad), n, "; ".join(bad[:3])), *bad_results)


COUNT_WORDS = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def case_greedy_then_sampled(ctx):
    """A sampled request whose sampling can only pick the most likely token (profile `argmax_sampling`, default temperature 1 with
    top_k 1) must return exactly the greedy reply. Asked greedy first, then sampled twice, for 24 consecutive prompt lengths, then four
    prompts at once greedy and the same four sampled. A server that hands a sampled request the logits of an earlier request (a decode
    graph or buffer captured for a greedy one) fails here. Most sensitive right after a server start: run it first, without any warm-up.
    Both requests send presence_penalty 0 and frequency_penalty 0, so that a server default cannot give them different penalties."""
    arg = ctx.p.get("argmax_sampling", {"temperature": 1.0, "top_k": 1, "top_p": 1.0})
    heads = ["Continue the list with the next four items and nothing else:", "Next four items of this list, nothing else:", "Write only the four items that follow:", "Only the next four items please:"]

    def ask(h, n, sampled):
        msgs = [{"role": "user", "content": heads[h] + "".join(" " + COUNT_WORDS[i % 20] for i in range(n))}]
        # both requests name their penalties (none): a server whose default sampling adds a penalty to one kind of request only
        # (a per-mode profile that leaves temperature-0 requests greedy) would otherwise compare two different distributions
        extra = {"presence_penalty": 0, "frequency_penalty": 0}
        extra.update(arg if sampled else {"temperature": 0})
        return ctx.c.chat(msgs, stream=False, think=False, max_tokens=8, extra=extra)

    key = lambda r: (r.status, r.content, r.finish, (r.usage or {}).get("completion_tokens"))
    bad = []; bad_results = []; n_checks = 0
    for n in range(1, 25):
        g = ask(0, n, False); s1 = ask(0, n, True); s2 = ask(0, n, True); n_checks += 1
        need(g.status == 200 and s1.status == 200, "HTTP %s / %s" % (g.status, s1.status), g, s1)
        if key(g) != key(s1) or key(g) != key(s2):
            bad.append("%d words: greedy %r, sampled %r, sampled again %r" % (n, g.content, s1.content, s2.content)); bad_results += [g, s1, s2]
    if not ctx.p.get("single_stream"):
        def many(n, sampled):
            out = [None] * 4

            def run(h): out[h] = ask(h, n + h, sampled)
            th = [threading.Thread(target=run, args=(h,)) for h in range(4)]
            [t.start() for t in th]; [t.join() for t in th]
            return out
        for n in range(2, 25, 3):
            g = many(n, False); s = many(n, True); n_checks += 1
            diff = [h for h in range(4) if key(g[h]) != key(s[h])]
            if diff:
                bad.append("4 at once, %d+ words: %s" % (n, "; ".join("greedy %r sampled %r" % (g[h].content, s[h].content) for h in diff)))
                for h in diff: bad_results += [g[h], s[h]]
    ctx.notes.append("greedy vs argmax-only sampling: %d of %d checks differ" % (len(bad), n_checks))
    need(not bad, "%d of %d checks: a sampled request that can only pick the top token differs from the greedy reply: %s" % (len(bad), n_checks, " | ".join(bad[:3])), *bad_results[:12])


def case_empty_reply_long_prompt(ctx):
    body = "\n".join("def f%d(x):\n    return x * %d + %d\n" % (i, i, i * 7) for i in range(400))
    r = ctx.chat([{"role": "user", "content": body + "\n\nWrite a unit test file for the functions above (just the first 5 functions)."}], max_tokens=3000, think=False)
    need(r.status == 200, "HTTP %s" % r.status, r)
    need(r.content.strip() != "", "empty reply (finish %r, %s completion tokens)" % (r.finish, (r.usage or {}).get("completion_tokens")), r)


def case_repetition(ctx):
    r = ctx.chat([{"role": "user", "content": "Write a 600-word essay on why the sky is blue."}], max_tokens=1500)
    need(r.status == 200, "HTTP %s" % r.status, r)
    tail = (r.content or r.reasoning)[-3000:]
    for n in (30, 60):
        grams = {}
        for i in range(0, max(0, len(tail) - n), 5):
            g = tail[i:i + n]; grams[g] = grams.get(g, 0) + 1
        need(not grams or max(grams.values()) < 6, "repetition loop: a %d-char span repeats %d times" % (n, max(grams.values())), r)


def case_liveness_prefill(ctx):
    for n in ctx.p.get("cold_prefill_tokens", [8000, 48000]):
        text = (" ".join("sentence %d about topic %d with some filler words to take up tokens." % (i, i * 7 % 13) for i in range(n // 10)))
        r = ctx.chat([{"role": "user", "content": "Session %f\n%s\n\nHow many sentences were above? Answer with just a number." % (time.time(), text)}], max_tokens=64, think=False)
        need(r.status == 200, "HTTP %s" % r.status, r)
        if ctx.stream:
            need(r.max_gap <= ctx.p.get("max_gap_s", 20), "%d-token prompt: %.1f s without bytes on the connection (limit %s)" % (n, r.max_gap, ctx.p.get("max_gap_s", 20)), r)


def case_liveness_long_call(ctx):
    r = ctx.chat([{"role": "user", "content": "Write a 600-line HTML page to big.html with the write tool."}], tools=PI_TOOLS, max_tokens=24000)
    need(r.status == 200, "HTTP %s" % r.status, r)
    if ctx.stream: need(r.max_gap <= ctx.p.get("max_gap_s", 20), "%.1f s without bytes (limit %s)" % (r.max_gap, ctx.p.get("max_gap_s", 20)), r)


def case_limits(ctx):
    ctxlen = ctx.p.get("context", 8192)
    r = ctx.chat([{"role": "user", "content": "word " * (ctxlen + 1000)}], max_tokens=16)
    need(r.status == 400 and r.http_body and ("context" in r.http_body.lower() or "token" in r.http_body.lower()), "over-context prompt: status %s body %r" % (r.status, (r.http_body or "")[:150]), r)
    r = ctx.chat([{"role": "user", "content": "Count from 1 to 500, one number per line."}], max_tokens=37, think=False)
    need(r.status == 200 and (r.usage or {}).get("completion_tokens") == 37 and r.finish == "length", "max_tokens not honoured exactly: tokens %s finish %s" % ((r.usage or {}).get("completion_tokens"), r.finish), r)
    r = ctx.chat([{"role": "user", "content": "Think very carefully and at length about the best chess opening, then answer."}], max_tokens=60, think=True)
    need(r.status == 200 and r.finish == "length", "reasoning-only reply at the cap: finish %r" % r.finish, r)


def case_reasoning_split(ctx):
    if not ctx.think: return
    r = ctx.chat([{"role": "user", "content": "What is 12 * 13? Think it through."}], max_tokens=4000, think=True)
    need(r.status == 200 and "</think>" not in r.content and "<think>" not in r.content, "reasoning markers leaked into content: %r" % r.content[:80], r)
    need(r.reasoning.strip() != "", "thinking on but no reasoning_content", r)



def char_cycle(text, min_period=60, max_period=3000, reps=3):
    """Return the period of a verbatim cycle that ends the text (repeated `reps` times), else None."""
    n = len(text)
    if n < min_period * reps: return None
    tail = text[-48:]
    j = text.rfind(tail, 0, n - 48)
    while j != -1:
        d = n - 48 - j
        if min_period <= d <= max_period and n >= d * reps and text[-d * reps:] == text[-d:] * reps:
            return d
        j = text.rfind(tail, 0, j)
        if n - 48 - j > max_period: break
    return None


def tail_cycle(text):
    """A cycle at the end of the text, checked at a few cut points (the cycle need not end exactly at the last character)."""
    for cut in (0, 7, 23, 61, 137):
        p = char_cycle(text[: len(text) - cut] if cut else text)
        if p: return p
    return None


def review_prompt(k):
    """A long, bug-ridden source file plus a request that invites long reasoning: the shape of the agent turns that looped."""
    funcs = []
    for i in range(60 + 7 * k):
        funcs.append("function handler%d(state, ev) {\n  const el = document.getElementById('item-%d');\n  if (!el) return state;\n  const next = Object.assign({}, state, {count: (state.count || 0) + %d});\n  el.textContent = String(next.count);\n  if (ev && ev.type === 'copy') { toast('copied %d'); }\n  return next;\n}\n" % (i, i, i % 5 + 1, i))
    src = "".join(funcs)
    asks = ["Review this script for bugs and edge cases (toast, copy handler, status updates); then use the edit tool to fix the three most important ones.",
            "Find places where the status is not updated after a copy, a validation failure or a theme toggle, and fix them with the edit tool.",
            "Walk through every handler, list potential issues one by one, then apply fixes with the edit tool.",
            "Audit this file for state bugs; reason about each handler, then patch it with the edit tool.",
            "Check each function for a missing status update and for DOM lookups that can fail; fix with the edit tool.",
            "Think carefully about edge cases in these handlers (copy, validation, theme) and then edit the file accordingly."]
    return [{"role": "system", "content": SYS}, {"role": "user", "content": "File app.js:\n```js\n%s```\n\n%s" % (src, asks[k % len(asks)])}]


def loop_runs(ctx, temperature, n, max_tokens):
    """n long reasoning requests; returns (loops, results, details)."""
    loops = 0; results = []; det = []
    for k in range(n):
        r = ctx.chat(review_prompt(k), tools=PI_TOOLS, max_tokens=max_tokens, think=True, extra={"temperature": temperature, "seed": 1000 + k})
        results.append(r)
        txt = r.reasoning + "\n" + r.content
        p = tail_cycle(r.reasoning) or tail_cycle(r.content)
        runaway = r.finish == "length" and not r.calls and not r.content.strip()
        det.append("run %d: finish %s, reasoning %d chars, content %d chars, calls %d%s" % (k, r.finish, len(r.reasoning), len(r.content), len(r.calls), (", CYCLE period %d chars" % p) if p else ""))
        if p or runaway: loops += 1
    return loops, results, det


def case_loop_temp0(ctx):
    n = ctx.args.loop_runs; cap = ctx.p.get("loop_max_tokens", 12000)
    loops, results, det = loop_runs(ctx, 0, n, cap)
    ctx.notes.append("loop rate at temperature 0, thinking on: %d/%d  [%s]" % (loops, n, " | ".join(det)))
    need(loops == 0, "%d of %d runs at temperature 0 looped or ran away (no answer inside %d tokens)" % (loops, n, cap), *results)


def case_runaway_reasoning(ctx):
    cap = ctx.p.get("reasoning_cap_tokens", 8192)
    bad = []; results = []
    for k in range(ctx.args.loop_runs):
        r = ctx.chat(review_prompt(k + 10), tools=PI_TOOLS, max_tokens=cap * 3, think=True)
        results.append(r)
        rt = (len(r.reasoning) // 3)   # rough tokens
        if (not r.calls and not r.content.strip()) or rt > cap * 1.5:
            bad.append("run %d: ~%d reasoning tokens, finish %s, %d calls, content %d chars" % (k, rt, r.finish, len(r.calls), len(r.content)))
    ctx.notes.append("runaway reasoning: %d/%d runs" % (len(bad), ctx.args.loop_runs))
    need(not bad, "reasoning is not bounded by the cap (%d tokens): %s" % (cap, "; ".join(bad[:3])), *results)


def replay_bodies(path):
    """Captured request bodies under `path` (a file or a directory): `<id>.json` files as a server's request capture writes them
    (`<id>.meta.json` sidecars are read for the original outcome, never sent), or a failure file of this suite (its `exchanges[].request`).
    Returns [(name, body, meta-or-None)]."""
    files = [path] if os.path.isfile(path) else sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json") and not f.endswith(".meta.json"))
    out = []
    for f in files:
        try:
            d = json.load(open(f))
        except (ValueError, OSError) as e:
            print("replay: skipped %s (%s)" % (f, e)); continue
        name = os.path.basename(f)[:-5] if f.endswith(".json") else os.path.basename(f)
        if isinstance(d, dict) and isinstance(d.get("exchanges"), list):
            for i, ex in enumerate(d["exchanges"]):
                if isinstance(ex.get("request"), dict) and "messages" in ex["request"]: out.append(("%s#%d" % (name, i), ex["request"], None))
            continue
        if not (isinstance(d, dict) and isinstance(d.get("messages"), list)):
            print("replay: skipped %s (not a chat-completions body)" % f); continue
        meta = None
        mf = f[:-5] + ".meta.json"
        if os.path.isfile(mf):
            try: meta = json.load(open(mf))
            except (ValueError, OSError): meta = None
        out.append((name, d, meta))
    return out


def replay(a):
    """--replay: resend each captured body --loop-runs times and report the loop rate (the loop_temp0 detection: a verbatim cycle at the
    end of the reasoning or the answer, or the whole budget spent without an answer). A captured real conversation becomes a suite case."""
    bodies = replay_bodies(a.replay)
    if not bodies:
        print("replay: no chat-completions bodies under %s" % a.replay); sys.exit(2)
    cl = Client(a.base_url, a.model); os.makedirs(a.out, exist_ok=True)
    n = a.loop_runs; total = 0; total_loops = 0; rows = []
    for name, body, meta in bodies:
        if a.model: body = dict(body, model=a.model)
        if body.get("stream") and "stream_options" not in body: body = dict(body, stream_options={"include_usage": True})
        if meta:
            print("%s: captured outcome: finish %s, %s output tokens, reasoning cap fired %s, loop breaker fired %s" % (name, meta.get("finish_reason"), meta.get("completion_tokens"), meta.get("reasoning_cap_fired"), meta.get("loop_breaker_fired")))
        loops = 0; bad = []
        for k in range(n):
            try:
                r = cl.send(json.loads(json.dumps(body)))
            except Exception as e:
                loops += 1; print("  %s run %d: ERROR %s: %s" % (name, k, type(e).__name__, e), flush=True); continue
            if r.status != 200:
                loops += 1; print("  %s run %d: HTTP %s %s" % (name, k, r.status, (r.http_body or "")[:200]), flush=True); continue
            p = tail_cycle(r.reasoning) or tail_cycle(r.content)
            runaway = r.finish == "length" and not r.calls and not r.content.strip()
            looped = bool(p) or runaway
            out_tok = (r.usage or {}).get("completion_tokens")
            print("  %s run %d: finish %s, reasoning %d chars, content %d chars, calls %s, output tokens %s, %.0fs%s%s" % (name, k, r.finish, len(r.reasoning), len(r.content), [c["name"] for c in r.calls], out_tok, r.wall, (", CYCLE period %d chars" % p) if p else "", ", NO ANSWER inside the budget" if runaway else ""), flush=True)
            if looped:
                loops += 1
                bad.append({"run": k, "request": r.body, "status": r.status, "content": r.content, "reasoning": r.reasoning, "calls": r.calls, "finish": r.finish, "usage": r.usage, "cycle_period_chars": p})
        if bad:
            with open(os.path.join(a.out, "replay_%s.json" % re.sub(r"[^A-Za-z0-9_.-]", "_", name)), "w") as f:
                json.dump({"case": "replay", "source": name, "exchanges": bad}, f, indent=1)
        print("%-40s %s  loops %d/%d" % (name, "PASS" if loops == 0 else "FAIL", loops, n), flush=True)
        rows.append((name, "PASS" if loops == 0 else "FAIL", loops, n)); total += n; total_loops += loops
    print("\nreplay: %d bodies x %d runs, loop rate %d/%d (%.0f%%)" % (len(bodies), n, total_loops, total, 100.0 * total_loops / max(total, 1)))
    if total_loops: print("    diagnosis: " + DIAG["loop_temp0"])
    json.dump(rows, open(os.path.join(a.out, "replay_summary.json"), "w"), indent=1)
    sys.exit(1 if total_loops else 0)


DIAG = {
    "greedy_then_sampled": "A sampled request did not start from its own logits (with only the top token allowed it must equal the greedy reply). Typical cause: decode state captured for one request kind and reused for the other (a CUDA graph or host buffer that carries, or lacks, the logits copy): key such caches by everything baked in at capture time.",
    "coding_stop": "Short coding requests ran to the token cap or came back empty. Check, in this order: that a sampled request reads its own logits (case greedy_then_sampled), the stop tokens, a ban on the stop token as the first generated token, the reasoning cap, the loop breaker; then compare the same prompts on a reference implementation of the model.",
    "chat_stop": "Informational: short open chat prompts that ran to the cap or came back empty. For a coding model this is often the model itself (compare with a reference implementation); if greedy_then_sampled or coding_stop fail too, fix those first.",
    "basic_call": "The model did not produce a schema-valid tool call. Check the chat template renders the tool schemas and the server parses the model's call format (parser / template); streaming runs also need usage and cached_tokens in the final chunk.",
    "generic_schemas": "Nested objects, arrays, booleans, numbers or enums arrive with the wrong type: arguments must be typed from the request's JSON schema, not from the model's own type attribute (parser).",
    "parallel_calls": "Two calls in one reply were not both returned: the parser drops or merges later calls (parser), or the model answers one call at a time (prompt).",
    "preamble_then_call": "Text before a call is lost, duplicated or breaks the call: check the parser's content / call boundary and that the preamble is emitted once.",
    "tool_choice": "tool_choice required / named is not enforced: the server must add the instruction to the prompt (or constrain decoding) and still return valid calls.",
    "special_content": "Characters inside string arguments were altered: quotes, backslashes, CRLF or look-alike tags are being mangled by the parser (value end detection, JSON escaping) or by tokenisation.",
    "cut_off_call": "A call cut off at max_tokens must report finish_reason length and never return invalid JSON arguments: close the JSON or drop the call.",
    "reasoning_split": "Reasoning leaks into content or is missing: the parser must split on the think markers and fill reasoning_content.",
    "stopping_20_prompts": "Short prompts run away or come back empty: check the stop tokens, the default max_tokens and for repetition loops (set a sampling floor for greedy requests / enable a loop breaker).",
    "empty_reply_long_prompt": "Empty reply in non-thinking mode: the model emits the end token first. Ban the stop ids on the first generated token (or pick the sampling profile).",
    "repetition_probe": "The output repeats a long span: loops at greedy decoding. Sample the thinking phase at the model's recommended temperature and enable the loop breaker.",
    "limits": "Limits are not honoured: over-context prompts must return HTTP 400 with a clear message, max_tokens must be exact, a reasoning-only reply at the cap must report length.",
    "liveness_cold_prefill": "The connection is silent for too long during a cold prefill: send SSE keep-alive comments (': keep-alive') every ~10 s from the first byte.",
    "liveness_long_call": "No bytes while a long tool call is generated: stream the arguments incrementally and/or send keep-alives.",
    "loop_temp0": "The model loops at temperature 0 with thinking on (a verbatim cycle in the reasoning). Fix: a sampling floor for the reasoning phase (profile thinking temperature / top_p) plus a loop breaker that closes the reasoning block when a token cycle repeats.",
    "runaway_reasoning": "Reasoning is not bounded: configure a reasoning-token cap that closes the block (</think>) and lets the model answer.",
    "multiturn_cache_8": "The prefix cache misses in agent loops: check cached_tokens per turn; the re-rendered history (reasoning included or dropped, argument serialisation) must match the cached tokens.",
    "edits_three": "Edits were rejected: edit arguments must arrive as a real array of objects (parser typing from the schema, missing closing tags) and oldText must match the file.",
    "long_write_1k": "A long write call did not complete as one valid call: check streaming of the arguments, the reasoning cap, and the loop breaker thresholds inside tool arguments.",
    "long_write_big": "A very long file was not completed through valid calls (one long call, or a staged write over several turns): a call that runs to max_tokens never ended (the model does not bound its write: nothing repeats exactly, so a loop breaker inside calls does not apply), or the file was left without its end. See long_write_1k; also context size and max_tokens.",
    "sampling_profile": "The server's default sampling differs from the profile's per-mode values, a client field did not win, or the server does not report its effective sampling in a response header: load the per-mode profile from the model's sampling configuration, apply it only to fields the client omits, and send the header. With `greedy_keeps_fast_path` in the profile: do not add the profile's penalties to a request whose client sent temperature 0.",
    "wandering_loop": "A reply that keeps bringing back the same few sentences in changing order ran on to the token cap: an exact-cycle loop breaker does not see it; add a repeated-sentence rule on the decoded text (outside code fences and tool-call arguments).",
}

# name, function, modes: which (stream, think) combos; heavy flag
# Cases whose failures are reported (status INFO) but never fail the run.
INFORMATIONAL = {"chat_stop"}

CASES = [
    ("greedy_then_sampled", case_greedy_then_sampled, "once", False),
    ("basic_call", case_basic_call, "all", False),
    ("generic_schemas", case_generic_schemas, "all", False),
    ("parallel_calls", case_parallel_calls, "all", False),
    ("preamble_then_call", case_preamble, "all", False),
    ("tool_choice", case_tool_choice, "all", False),
    ("special_content", case_special_content, "all", False),
    ("cut_off_call", case_cut_off, "all", False),
    ("reasoning_split", case_reasoning_split, "think", False),
    ("stopping_20_prompts", case_stopping, "all", False),
    ("coding_stop", case_coding_stop, "stream_both", False),
    ("chat_stop", case_chat_stop, "stream_both", False),
    ("empty_reply_long_prompt", case_empty_reply_long_prompt, "off", False),
    ("repetition_probe", case_repetition, "all", False),
    ("limits", case_limits, "all", False),
    ("sampling_profile", case_sampling_profile, "all", False),
    ("liveness_cold_prefill", case_liveness_prefill, "all", True),
    ("liveness_long_call", case_liveness_long_call, "on_stream", True),
    ("loop_temp0", case_loop_temp0, "on_stream", True),
    ("runaway_reasoning", case_runaway_reasoning, "on_stream", True),
    ("wandering_loop", case_wandering_loop, "off_stream", True),
    ("multiturn_cache_8", case_multiturn_cache, "on", True),
    ("edits_three", case_edits, "on", True),
    ("long_write_1k", case_long_write_1k, "on", True),
    ("long_write_big", case_long_write_big, "on", True),
]


def modes_for(kind, quick):
    allm = [(s, t) for s in (True, False) for t in (True, False)]
    if kind == "all": m = allm
    elif kind == "think": m = [(s, True) for s in (True, False)]
    elif kind == "off": m = [(s, False) for s in (True, False)]
    elif kind == "on": m = [(True, True)]
    elif kind == "on_stream": m = [(True, True)]
    elif kind == "once": m = [(False, False)]
    elif kind == "off_stream": m = [(True, False)]
    elif kind == "stream_both": m = [(True, False), (True, True)]
    else: m = allm
    if quick: m = m[:1]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000"); ap.add_argument("--model", default=None)
    ap.add_argument("--profile", default=None, help="model profile (required unless --replay)"); ap.add_argument("--only", default=None, help="regex on case names")
    ap.add_argument("--tier", choices=["quick", "heavy", "all"], default="quick", help="quick = minutes; heavy = long writes, loops, cache loop, liveness; all = both")
    ap.add_argument("--loop-runs", type=int, default=6, help="requests per loop / runaway case (the report gives the rate)"); ap.add_argument("--quick", action="store_true", help="one mode per case")
    ap.add_argument("--out", default="conformance-out")
    ap.add_argument("--sampling", default=None, metavar="JSON", help="sampling fields added to every request that does not set them itself, e.g. '{\"temperature\": 0}' (what a harness sends); for measuring one sampling setting across the cases")
    ap.add_argument("--seed", type=int, default=None, help="give every request that has no seed its own seed derived from this number (repeatable sampled runs; run the suite with several seeds)")
    ap.add_argument("--modes", choices=["all", "think", "nothink"], default="all", help="only the runs with thinking on / off")
    ap.add_argument("--replay", default=None, metavar="FILE_OR_DIR", help="resend captured chat-completions request bodies (one .json per request; .meta.json sidecars are not sent) --loop-runs times each and report the loop rate; no profile needed")
    a = ap.parse_args()
    if a.replay: replay(a)
    if not a.profile: ap.error("--profile is required (except with --replay)")
    prof = json.load(open(a.profile)); model = a.model or prof["model"]
    cl = Client(a.base_url, model, defaults=json.loads(a.sampling) if a.sampling else None, seed=a.seed, header=prof.get("effective_sampling_header", "x-sampling-effective"))
    os.makedirs(a.out, exist_ok=True)
    rows = []; failed = 0; info = 0
    for name, fn, kind, heavy in CASES:
        if a.only and not re.search(a.only, name): continue
        if heavy and a.tier == "quick" and not a.only: continue
        if not heavy and a.tier == "heavy" and not a.only: continue
        for stream, think in modes_for(kind, a.quick):
            if a.modes != "all" and think != (a.modes == "think"): continue
            ctx = Ctx(cl, prof, stream, think, a); t0 = time.time(); status = "PASS"; detail = ""
            try:
                fn(ctx)
            except Fail as e:
                status = "INFO" if name in INFORMATIONAL else "FAIL"; detail = str(e)
                tag = "%s_%s_%s" % (name, "stream" if stream else "full", "think" if think else "nothink")
                with open(os.path.join(a.out, tag + ".json"), "w") as f:
                    json.dump({"case": name, "detail": detail, "exchanges": [{"request": r.body, "status": r.status, "content": r.content, "reasoning": r.reasoning, "calls": r.calls, "finish": r.finish, "usage": r.usage, "effective_sampling": getattr(r, "effective", None), "raw": r.raw, "http_body": r.http_body} for r in e.results]}, f, indent=1)
            except Exception as e:
                status = "ERROR"; detail = "%s: %s" % (type(e).__name__, e); traceback.print_exc()
            failed += status not in ("PASS", "INFO"); info += status == "INFO"
            rows.append((name, "stream" if stream else "full", "think" if think else "nothink", status, "%.0fs" % (time.time() - t0), detail, "; ".join(ctx.notes)))
            print("%-26s %-6s %-8s %-5s %6s  %s" % rows[-1][:6], flush=True)
            if status != "PASS" and name in DIAG: print("    diagnosis: " + DIAG[name])
            if status != "PASS" and status != "ERROR": print("    raw exchanges: " + os.path.join(a.out, tag + ".json"))
            for n in ctx.notes: print("    " + n)
    print("\n%s: %d run(s), %d failed%s" % (model, len(rows), failed, (", %d informational (not counted)" % info) if info else ""))
    json.dump(rows, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
