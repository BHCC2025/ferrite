# Agent harness settings

A server can pass every API test and a coding session can still hang or fail on the harness side. This page lists
the harness settings that were tested end to end with Ferrite, and the failure each one prevents.

The API-side checks (tool calls, streaming, stopping, loops, sampling defaults, caching, liveness) are covered by
the conformance suite in [`tools/conformance`](../tools/conformance/README.md). This page covers what the suite
cannot see: the harness's own behaviour.

## Pi coding agent

Tested with Pi 1.5.2 against `fe-server` (OpenAI-compatible chat completions).

### Provider entry (`~/.pi/agent/models.json`)

```json
{
  "baseUrl": "http://<server>:8000/v1",
  "api": "openai-completions",
  "compat": {
    "supportsDeveloperRole": false,
    "supportsReasoningEffort": false,
    "supportsFinishReason": false,
    "thinkingFormat": "qwen-chat-template"
  },
  "models": [
    { "id": "<served model id>", "reasoning": true, "input": ["text"],
      "contextWindow": 131072, "maxTokens": 32768 }
  ]
}
```

Set `contextWindow` to the context the server was started with and keep `maxTokens` well below it.

### Shell commands need a time limit

Pi's `bash` tool takes an optional `timeout` in seconds and has no default. A command that never exits blocks the
session until someone presses Esc: a test script kept alive by a timer or an open handle, a server started in the
foreground, a prompt waiting for input. The model server is idle the whole time, so nothing on the server side can
detect it.

Observed: after finishing a 2,700-line page the model wrote its own test script and ran it; the script never
exited and the session sat for 16 minutes.

Fix: the extension in [`harness/pi/bash-timeout.ts`](harness/pi/bash-timeout.ts). Copy it to
`~/.pi/agent/extensions/` and restart Pi (a new extension file is picked up at start, not by `/reload`). It

- sets a default limit when the model passes no `timeout` (120 s),
- caps a limit the model asks for (600 s),
- appends a note to the tool result when a command is stopped, so the model knows what happened and how to
  proceed (ask for longer, or make the command end on its own).

Optional settings in `~/.pi/agent/bash-timeout.json`, re-read on every call:

```json
{ "defaultSeconds": 120, "maxSeconds": 600, "enabled": true }
```

Check: ask the model to run `sleep 300` without a timeout. The tool result must read
`Command timed out after 120 seconds` followed by the note, and the model must report the timeout.

### Sampling

Pi sends `temperature: 0` on every request and no other sampling field.

`fe-server` fills the fields a client leaves out from the model's `sampling.json`: one set of values for replies
that think and one for replies that do not (the base model's published settings; the tables are in
[model-card.md](model-card.md#recommended-settings) and [flash-next.md](flash-next.md#recommended-settings)). A field
the client sends always wins.

A request that sends `temperature: 0` is decoded greedily, and the server adds no presence, frequency or repetition
penalty to it (`greedy_keeps_fast_path`, on by default). Such a request keeps the fast greedy path with speculative
decoding, so Pi needs no extra setting. A penalty that the client sends together with `temperature: 0` still
applies, and takes the request off that path.

The values a request ran with are returned in the `x-sampling-effective` response header and written to the server
log, with the origin of each (client, profile, default, or withheld by the greedy rule).

### Idle timeout

Pi aborts a request after 300 s without bytes. `fe-server` sends an SSE keep-alive comment (`: keep-alive`) every
10 s of silence on a streamed reply, from the moment the request is queued, so long prompts and queued requests stay
alive.

The server has a limit of its own in the other direction: a reply that has started and then produces no token for
60 s is ended with `finish_reason: "length"` (`FE_NO_TOKEN_TIMEOUT`, seconds; `0` turns it off).

### What the server does about runaway replies

These apply to every harness, and none of them needs a harness setting:

- a reasoning block still open after 8,192 tokens is closed by the server and the model answers
  (`FE_MAX_REASONING_TOKENS`);
- a reply that repeats one cycle of tokens, or keeps coming back to the same sentences, is cut short; code blocks
  are exempt and tool-call arguments are held to a far stricter rule (`FE_LOOP_BREAKER=0` turns it off);
- the first token of a reply is never a stop token, so a reply is never empty;
- `tool_choice` `required` or a named function is enforced, and limited to the tools offered.

`FE_DUMP_REQUESTS=<dir>` writes every chat-completions request body to a file, with a small `.meta.json` beside it
when the request ends (finish reason, tokens, whether a protection fired). A captured body can be replayed with the
conformance suite's `--replay`.

## Other harnesses

Not tested yet. The same three questions apply to any harness: does a shell command have a default time limit,
what sampling values does it send, and how long does it wait without bytes before it aborts.
