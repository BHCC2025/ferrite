# API conformance suite

Checks that an OpenAI-compatible chat-completions server is usable by a real coding agent: tool calls, streaming,
reasoning, limits, loops, caching, liveness. It talks only to the HTTP API (base URL + model id), needs only the Python
3 standard library, and keeps every model-specific fact in a profile file. It contains no paths or imports from the
server it was written next to, so it can be moved to its own repository unchanged.

## Usage

    python3 run.py --base-url http://127.0.0.1:8000 --profile profiles/<model>.json              # quick tier, minutes
    python3 run.py --base-url ... --profile ... --model <id>                                      # when the server reports another model id than the profile's
    python3 run.py --base-url ... --profile ... --tier heavy --loop-runs 6                        # long writes, loops, cache, liveness
    python3 run.py --base-url ... --profile ... --tier all
    python3 run.py ... --only loop_temp0 --loop-runs 10                                           # one case (regex on the name)
    python3 run.py ... --quick                                                                    # one mode per case
    python3 run.py --base-url ... --replay <file-or-dir> --loop-runs 10                           # captured request bodies (see Replay)

Every case runs streaming and non-streaming, thinking on and off where the case allows it. The output is one line per run
(PASS / FAIL / ERROR, seconds) with a plain-language diagnosis and the likely fix under each failure. Failing runs save the raw
exchange (request, content, reasoning, calls, usage, raw SSE) to `--out` (default `conformance-out/`). The exit code is non-zero if
anything failed.

## Tiers

* quick: `greedy_then_sampled` first (see below), tool-call basics with a harness-style schema set and generic schemas (nested objects,
  arrays, enums, numbers, booleans), parallel calls, preamble + call, tool_choice, special characters byte-for-byte, a call cut off at
  max_tokens, reasoning split, 20 stop prompts, `coding_stop`, `chat_stop` (informational), empty-reply check, repetition probe, limits
  (over-context, exact max_tokens, reasoning-only reply at the cap).
* heavy: cold-prefill and long-call liveness (longest gap without bytes), `loop_temp0` and `runaway_reasoning` (n requests each,
  the report gives the loop rate), an 8-round write / read / edit loop with cached tokens per round (reasoning sent back on even rounds,
  dropped on odd rounds), three multi-KB edits, 1K and 2.5K-line writes, `wandering_loop`.
* quick also holds `sampling_profile` (skipped for profiles without `sampling_profiles`).

## Sampling: profile defaults, one setting across the suite, seeds

`sampling_profile` (quick, gating): a request without sampling fields must run with the server's profile for its mode (profile key
`sampling_profiles`: `{"thinking": {...}, "non_thinking": {...}}`, e.g. a model card's recommended values), and a field the client sends
must win while the others still come from the profile. The suite reads what the server ran with from a response header holding a JSON
object (`effective_sampling_header`, default `x-sampling-effective`): any server or proxy can add it. Failure files record it as
`effective_sampling`. With profile key `greedy_keeps_fast_path: true` the case also asserts that a request sending `temperature: 0` and
no penalty runs with presence 0 / frequency 0 / repetition 1 (the profile's penalties would take a greedy request off a server's
fast greedy path), and that penalties sent together with `temperature: 0` still apply.

`--sampling '{"temperature": 0}'` adds the given fields to every request that does not set them itself: the suite then measures one
client setting (what a harness sends) across all cases. `--seed N` gives every request without a seed its own seed derived from N, so a
sampled run is repeatable; run several seeds and compare the pass counts. `--modes think|nothink` keeps only the runs of one mode.

## Long writes

`long_write_1k` wants one valid write call of about the asked size. `long_write_big` (the profile's largest size) wants a complete file
on disk through valid calls, however the model gets there: one write call of any length that ends properly (an overshoot is accepted;
`long_write_max_tokens`, default 48000, is the per-turn cap), or a staged write (a first write, then write / edit calls over the next
turns, at most `long_write_rounds`, default 6; the suite applies the calls to its in-memory files and nudges once if the model stops
early). It fails when a turn is cut off at the cap inside a call, when a call is invalid, or when the file never gets its closing tag.

## Wandering loops

`wandering_loop` (heavy): asks for six given lines again and again, each round in another order, 80 rounds (about 5,000 tokens): text
that keeps bringing back the same sentences without ever forming a verbatim cycle. A server with only an exact-cycle loop breaker
writes all of it; one that also watches for recurring sentences ends the reply early. Passes when the reply ends below
`wander_stop_tokens` (default 3600) of `wander_max_tokens` (default 8000), at temperature 0 and at the default sampling. This is the
answer-side stand-in for the reasoning loop it is named after (candidates tried and rejected in shifting order), which no prompt
produces on demand; a server's own tests should replay a captured transcript against its detector.

## Stop cases and prompt sets

`prompts/coding.json` and `prompts/chat.json` hold short prompts (a system prompt, user prompts, for coding also tool prompts and tool
schemas). Add prompts there, not in the runner.

`coding_stop` (gating): every coding prompt alone, with the tools offered, and every tool prompt with the tools, each at temperature 0 and
at the server's default sampling, streaming, thinking off and on. Each reply must stop by itself (`finish_reason` stop or tool_calls)
inside `stop_set_max_tokens` (profile, default 4096) with a non-empty answer or a schema-valid call to an offered tool.
`chat_stop` (informational): the chat prompts without and with the system prompt, same matrix. The rate is printed and the raw exchanges
are saved, but the status is `INFO` and the exit code does not depend on it: a coding model may ramble on open chat prompts.
`stopping_20_prompts` now saves the raw exchange of every prompt that ran away or came back empty.

`greedy_then_sampled` (gating, runs first): a request whose sampling can only pick the most likely token (profile `argmax_sampling`,
default `{"temperature": 1.0, "top_k": 1, "top_p": 1.0}`) must return exactly the greedy reply. Asked greedy first, then sampled twice,
for 24 consecutive prompt lengths, then four prompts at once (skipped for `single_stream` profiles). It catches a server that hands a
sampled request another request's logits (decode state captured for a greedy request and reused for a sampled one). It is most
sensitive right after a server start, before anything else has run: do not warm the server up before the quick tier. Both requests
send `presence_penalty: 0` and `frequency_penalty: 0`: a server default that adds a penalty to sampled requests only (a per-mode profile
with `greedy_keeps_fast_path`) would otherwise make the two replies differ for a reason this case is not about.

## Special content and tool-format tags

`special_content` asks for texts to be written to a file byte for byte: quotes, backslashes, Unicode, CRLF, and one text per known
tool-call format that looks like that format's markup (`LOOKALIKE` in the runner; every profile gets all of them). The text sits between
two full marker lines that cannot be taken for content. Every call in every case is also checked for tag text of the profile's own
`tool_format` inside its arguments (`FORMAT_TAGS`: the parser lost track of a call); tags that the requested text itself contains are
allowed. A new tool format needs one entry in each of the two tables.

## Loop cases

`loop_temp0`: n long, reasoning-heavy requests (a large source file, a review-and-fix request, tools offered) at `temperature: 0`
with thinking on. A run counts as a loop when its reasoning or answer ends in a verbatim cycle (a span repeated three times), or when
it spends the whole budget without answering. The rate is printed as `loops k/n`. Run it before and after a server change to see the effect.
`runaway_reasoning`: the same requests at the default settings; reasoning must stay within the profile's cap.

## Replay: a captured conversation as a case

    python3 run.py --base-url http://127.0.0.1:8000 --replay captures/ --loop-runs 10            # every body in a directory
    python3 run.py --base-url ... --replay captures/chatcmpl-abc.json --loop-runs 20 --model m    # one body, model id replaced

`--replay <file-or-dir>` resends captured chat-completions request bodies, each `--loop-runs` times, and reports the loop rate per body
and in total, with the same detection as `loop_temp0` (a verbatim cycle at the end of the reasoning or the answer, or the whole budget
spent without an answer or a call). No profile is needed. Each body is sent as captured (messages, tools, sampling, `stream`); only
`model` is replaced when `--model` is given, and `stream_options.include_usage` is added to streaming bodies that lack it. Looping
exchanges are saved to `--out` as `replay_<name>.json`, a summary as `replay_summary.json`; the exit code is non-zero if any run looped,
errored or returned a non-200 status.

Input files: one JSON file per request holding the body (`<id>.json`). A `<id>.meta.json` file next to it is never sent; if present its
recorded outcome (finish reason, output tokens, whether a reasoning cap or loop breaker fired) is printed before the runs. A failure file
written by this suite (`exchanges[].request`) is accepted too. Any server or proxy that can write request bodies to files produces usable
input; the server this suite was written next to does it with the environment variable `FE_DUMP_REQUESTS=<dir>`.

Workflow for a real harness failure: turn request capture on, reproduce the failure with the harness, copy the request that went wrong
(the meta file names the finish reason) into a fixtures directory, and run `--replay` on it before and after the server change.

## Profiles

A profile is JSON: model id, tool format (`tool_format`: `dsml` or `qwen`), context size, long-write sizes, the longest allowed silent
gap, the reasoning cap, cold-prefill sizes, optionally `stop_set_max_tokens`, `argmax_sampling`, `single_stream`, `sampling_profiles`,
`effective_sampling_header`, `long_write_max_tokens`, `long_write_rounds`, `wander_max_tokens` and `wander_stop_tokens`. Add one file per
model. Model-specific facts belong there and nowhere else.

### Tested harness settings

Settings of the agent harness that go with each profile (what passed the suite end to end):

* `qwen3.8-27b` (Qwen3.8-27B on Ferrite's 27B engine, one card, one stream): Qwen XML tool calls; the server closes a reasoning
  block at 8192 tokens, cuts exact cycles and recurring sentences short, never returns a stop token as the first token, enforces
  `tool_choice` required / named by starting the reply inside the call, streams call arguments as they are written, and applies the
  model card's per-mode sampling to the fields the client omits (thinking 1.0 / 0.95 / 20, presence 0; non-thinking 0.7 / 0.8 / 20,
  presence 1.5). A client that sends `temperature: 0` gets no penalty from those defaults.
* `qwen3.8-flash-next` (Qwen3.8-Flash-Next on Ferrite's Flash-Next engine, two cards, four requests at once): the same protections and
  the same per-mode sampling; tool requests do not think by default. The server on the test machine reports the model id `flashnext`,
  so the runs there add `--model flashnext`.

The harness these were tested with sends `temperature: 0` and aborts a request after 300 s without bytes, so the server must send
keep-alive comments during queueing and prompt processing (case `liveness_cold_prefill`). The harness-side settings are in
[`docs/harness.md`](../../docs/harness.md).
