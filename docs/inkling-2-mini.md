# Inkling-2-Mini on one RTX PRO 4000, RTX 5060 Ti or DGX Spark

**Inkling-2-Mini** from [Thinking Machines Lab](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Apache-2.0) is a
mixture-of-experts reasoning model of about 20 GB. Ferrite serves it from the checkpoint as published (NVFP4 and MXFP8
experts, FP8 attention) on **one NVIDIA GPU**: an **RTX PRO 4000 Blackwell (24 GB)**, an **RTX 5060 Ti (16 GB)** or a
**DGX Spark (GB10, 128 GB of unified memory)**, with a **128K-token context on each**.

- **Speed:** a short code request runs at 138 tokens/s on the 24 GB card, 110 on the 16 GB card and 72 on a DGX Spark
  ([Results](#results)).
- **It reads pictures.** A chat request may hold PNG and JPEG pictures; each is encoded once and then lives in the prompt cache
  ([Images](#images)).
- **Four requests at once on the 24 GB card and on a DGX Spark.** Each stream has its own caches; 4 short requests run at 160 to 190
  tokens/s together on the 24 GB card and 81 to 92 on a DGX Spark, and every stream gets the same text as alone
  ([Several requests at once](#several-requests-at-once-lanes)).
- **A prompt cache for each conversation in progress.** A follow-up turn prefills only what is new
  ([Prompt cache](#prompt-cache)).
- **It reasons by default.** Reasoning arrives in `reasoning_content`, and the effort setting lowers it without
  switching it off ([Using the server](#using-the-server)).

Terms such as K/V cache, experts, drafting, prefill and KL are explained in
[Words used in these docs](../README.md#words-used-in-these-docs).

> **Status: experimental.** One model, built from source only, on Linux. Measured on two machines: an x86 workstation with the RTX cards
> and a DGX Spark (see Results). Text and pictures in, text out; up to four requests at once on the 24 GB card and on a DGX Spark (one on
> the 16 GB card by default), with a prompt cache for each conversation in progress. Image input was measured on the RTX cards only.
> Reports and measurements from other cards are welcome.

## Requirements

| | |
|---|---|
| GPU | One **RTX PRO 4000 Blackwell (24 GB)**, one **RTX 5060 Ti (16 GB)** or a **DGX Spark** (GB10). The launcher uses the first visible GPU and picks the layout from its memory: the 24 GB layout from about 23,000 MiB, the 16 GB layout from about 15,000 MiB; it stops on a smaller card. On a DGX Spark, whose GPU shares the system memory, it picks the 24 GB layout. |
| CPU | x86_64, or the DGX Spark's aarch64. |
| Host memory | 16 GB card: 32 GB recommended, because up to about 4 GiB of experts are pinned in host memory (the launcher warns below 24 GiB). 24 GB card: any; the launcher does not check it. DGX Spark: the server takes about 25 GB of the unified memory with its four lanes, plus about 17 KiB a token of context and what the park pool holds (up to 16 GB). |
| Disk | About 20 GB for the model files. |
| OS and driver | Linux (tested on Ubuntu 24.04 and DGX OS), NVIDIA driver 580 or later (tested with 595.99 on the RTX cards, 580.178 on a DGX Spark). |
| To build | The Rust toolchains of the other NVIDIA builds and `ptxas` from the CUDA 13 toolkit ([docs/quickstart.md, "Clean machine"](quickstart.md#clean-machine)). On a DGX Spark the CUDA 13.0 toolkit it ships with works. |
| To download | `curl`, `sha256sum` and `python3` (standard library only); the Hugging Face CLI (`hf`) is used when it is installed. |

What each layout keeps where is under [What it runs on](#what-it-runs-on).

## Run it

```bash
# 1. build; FE_KARCH=sm_120a compiles the kernels for the RTX cards (on a DGX Spark leave it out: sm_121a is the default there)
FE_KARCH=sm_120a cargo build --release -p fe-server --features ink -p fe-ink

# 2. download: ~20 GB into $FERRITE_MODELS/Inkling-2-Mini (default /var/tmp/models), pinned revision, sha256 checked
scripts/fetch-inkling.sh

# 3. serve on :8000: picks the layout from the card (foreground; loading takes one to two minutes)
tools/serve-ink.sh

# 4. from another terminal: test it
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":200}'
```

- **The build.** `-p fe-ink` also builds `target/release/ink-run`, the tool for the self-test and the benchmarks
  named on this page.
- **The first start** also computes the draft head's fold tables (about a minute of CPU) and keeps them beside the
  model as `fold.safetensors`.
- **The test reply.** The model reasons before it answers, so the first part of a short reply is in
  `reasoning_content` and the answer in `content`.
- **Stopping and logs.** `tools/serve-ink.sh` runs in the foreground and its process becomes the server: Ctrl-C or
  `kill <pid>` stops it, and its log is what the terminal shows. `scripts/ferritectl.sh` has no target for this
  engine, so there is no built-in background mode.
- **Network.** It listens on every interface by default (`FERRITE_HOST=127.0.0.1` keeps it on the machine; there is
  no API key) and holds the card's GPU lock.
- **Check an install** with `target/release/ink-run selftest` (about two minutes; `--deep 65536` adds a 64K prompt).
- **Measure it** with `tools/bench-ink.py` (against a running server) or
  `target/release/ink-run bench --ctx 8192 --kind code`.

## Settings

`tools/serve-ink.sh --help` lists the launcher's settings. The ones you are most likely to need:

| Variable | Default | What it sets |
|---|---|---|
| `INK_KV` | `6bit` on both layouts | The K/V cache of the 8 global layers: `6bit` or `bf16`. What bf16 costs is under [Results](#results). |
| `INK_DRAFT` | `on` | The model's own draft head: `on` or `off`. |
| `INK_LANES` | `4` on the 24 GB layout and on a DGX Spark, `1` on the 16 GB layout | Requests decoded at once ([Several requests at once](#several-requests-at-once-lanes)). `1` serves one request at a time; on a DGX Spark up to `8` raise the total throughput further. |
| `FERRITE_PARK_GB` | `8` (pinned host memory); `16` on a DGX Spark (kept in its unified memory) | The park pool for conversations a lane gives up ([Park pool](#park-pool)); `0` turns it off. |
| `INK_LANE_PREFILL_ROWS` | `128`; `256` on a DGX Spark | Prompt rows a new request adds to each pass while other lanes decode ([Several requests at once](#several-requests-at-once-lanes)). |
| `INK_LAYOUT` | from the card's memory; `24gb` on a DGX Spark | Force a layout: `24gb` or `16gb`. |
| `FERRITE_CTX` | `131072` | Context length, prompt plus answer. |
| `FERRITE_HOST`, `FERRITE_PORT` | `0.0.0.0`, `8000` | Every interface, so other machines on the network can use it; there is no API key, so use it on a trusted network or firewall the port. `127.0.0.1` keeps it on this machine. |
| `FERRITE_MODEL_ID` | `inkling-2-mini` | The id `/v1/models` reports. |
| `FERRITE_LOCK_WAIT` | `0` | `1`: wait for a busy card instead of stopping. |
| `INK_NUMA` | bound to node 0 on a multi-socket machine with `numactl` | The `numactl` binding. Bind it to the node the card hangs off; `""` runs unbound. |
| `INK_DIR` | `$FERRITE_MODELS/Inkling-2-Mini` | The model folder. |
| `FE_DEFAULT_TEMPERATURE`, `FE_DEFAULT_REASONING_EFFORT`, `FE_DEFAULT_MAX_TOKENS` | 1.0, the template's 0.9, 32768 | Request defaults ([Using the server](#using-the-server)). |

The engine's own settings, which the launcher sets for the layout, are under [Advanced settings](#advanced-settings).

## Using the server

OpenAI-compatible chat and completions API, streaming or not. The model writes typed messages (reasoning, answer, tool calls); the server
parses them, so a client sees:

- **Reasoning** in `reasoning_content` (`message.reasoning_content`, or `delta.reasoning_content` when streaming); the answer is in `content`, clean.
  `usage.completion_tokens` counts everything generated, `usage.completion_tokens_details.reasoning_tokens` the tokens of the reasoning block.
- **Tool calls** as OpenAI `tool_calls` (several in one turn, `finish_reason: "tool_calls"`; `parallel_tool_calls: false` keeps the first), rendered through
  the model's own chat template. Arguments are checked and streamed as the model wrote them. `tool_choice` `required` or a named function is only a
  hint to the model. A call in a request that offered no tools is dropped and ends the turn. Text of a message that spells a control token
  (`<|content_thinking|>`) stays text, and cannot forge a message boundary.
- **Stop strings** and `max_tokens` cut at exactly that token, also inside an accepted draft block.

The chat template is checked against the publisher's: 28 conversations (tools, parallel calls, results by id or name, reasoning kept in the
history) render byte for byte like the Hugging Face template and tokenize to the same ids (`cargo test -p fe-server --test inkling_template`).

**Defaults.** A request that omits a field gets the publisher's settings, and everything a request ran with is reported in the
`x-sampling-effective` response header. What the request sends wins.

| Setting | Default | Change the default (launcher environment) | Request fields |
|---|---|---|---|
| Sampling | temperature 1.0 with sampling on and no top-p / top-k: the publisher's `generation_config.json` | `FE_DEFAULT_TEMPERATURE=0.7` | `temperature`, `top_p`, `top_k`, `min_p` |
| Reasoning effort | 0.9 ("high"), the chat template's own default | `FE_DEFAULT_REASONING_EFFORT=medium`: a name from none, minimal, low, medium, high, xhigh, max, or a number 0 to 0.99 | `reasoning_effort`, `chat_template_kwargs.reasoning_effort`, and `enable_thinking: false` (= effort `none`) |
| Output cap of a request without `max_tokens` | 32,768 tokens, because the model card advises limiting the output: reasoning can run long | `FE_DEFAULT_MAX_TOKENS=8192` (`0` = no cap) | `max_tokens` / `max_completion_tokens` |

An effort the template does not know is a 400 with the template's own message.

**What the effort does.** It is a soft control, and `none` does **not** switch reasoning off (the model card says so too): on short prompts the model still
writes 40 to 380 reasoning tokens.

| Request | Effort | Reasoning tokens | Answer tokens | Seconds | Tokens/s |
|---|---|---|---|---|---|
| short coding request (a function and two tests) | none | 98 | 374 | 3.6 | 130.4 |
| | low | 27 | 351 | 3.5 | 136.6 |
| | medium | 260 | 402 | 5.7 | 117.6 |
| | high (default) | 1,604 | 555 | 14.7 | 130.6 |
| short prose request (a few paragraphs) | none | 230 | 289 | 6.6 | 80.4 |
| | low | 157 | 335 | 5.8 | 79.6 |
| | medium | 141 | 286 | 5.3 | 80.7 |
| | high (default) | 386 | 374 | 9.6 | 79.2 |

(Median of three seeds at temperature 1.0, answered to the end, RTX PRO 4000, draft on, the default 6-bit K/V, measured
through `tools/serve-ink.sh` with `tools/conformance/ink_settings_bench.py`.)

- **Effort changes the length of the reasoning, not reliably in order.** Low can think less than none on one prompt, and one effort varies a lot
  from seed to seed: 749 to 1,792 reasoning tokens for the same coding request at high.
- Reasoning is 40 to 55% of the tokens of a short request at none, medium and high.
- **Drafting follows the text, reasoning included.** A coding request, whose reasoning is mostly about code, runs at 120 to 155 tokens/s at
  temperature 0, 0.7 and 1.0 (drafting does not slow sampled requests). Reasoning and answers about prose-like questions run at the plain speed
  (about 80 tokens/s on this card): the draft head gives no speed-up there.

### Prompt cache

A request whose prompt starts with the text of an earlier request resumes there and prefills only the rest: in a chat or agent session
every turn repeats the previous prompt, so only the reply and the new message are prefilled. The server reports the reused tokens as
`cached` in its log line, `usage.prompt_tokens_details.cached_tokens` and `timings.cache_n`.

- **Measured** on an RTX PRO 4000 (draft on, 6-bit K/V, thinking on, tool calls): an agent conversation of 104,573 tokens takes 131 s
  to prefill once. Each of the next seven turns (105K to 108K tokens, 200 to 616 new tokens each) started its answer after 0.33 to
  1.34 s instead of 133 to 139 s, with replies identical to the same server with `INK_PCACHE=0`.
- **On a DGX Spark** (the same settings, thinking at low effort): a conversation of 93,538 tokens takes 153 s to prefill once; each of the
  next four turns (93,752 to 94,152 tokens, 98 to 215 new tokens each) started its answer after 0.45 to 0.78 s instead of 152 to 154 s,
  with replies identical to the same server with `INK_PCACHE=0`.
- **What is kept.** The global layers' K/V stays in place on the card. What later rows overwrite (the sliding-window K/V, the
  short-convolution states and the draft head's sliding layers) is copied to pinned host memory at the prompt end and every 8,192
  tokens: 172 MiB each, about 7 ms (about 5 ms on a DGX Spark, about 65 ms there the first time a copy's buffer is allocated), up to
  `INK_PCACHE_GB` (2 GiB by default, 11 copies). No card memory is added; on a DGX Spark the copies come out of the same unified memory.
- **What resumes.** A prompt that changes earlier in the history resumes from the last of those points before the change. Each lane
  keeps its own conversation, and a request goes to the lane that holds the longest start of its prompt; a conversation that finds none
  starts over on the lane used longest ago, replacing what it held. A reply the model generated is prefilled again when the next request
  repeats it: only prompt rows are reused.
- **Same answers.** `ink-run pcache` checks resumed requests against requests prefilled from 0, greedy and sampled, draft head on and
  off; they are bit-identical.
- `INK_PCACHE=0` turns it off. The cache is in host memory only and is empty after a restart. With several lanes the snapshot memory
  (`INK_PCACHE_GB`) is shared among them: 2 snapshots a lane at 4 lanes.

### Images

A chat request may hold pictures: `image_url` content parts with a `data:` URL, in user and tool messages.

```json
{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}},
  {"type": "text", "text": "Read the text in this chart."}]}
```

- **Formats.** PNG and JPEG, base64 in a `data:image/png;base64,` or `data:image/jpeg;base64,` URL. `input_image` and `image` parts that
  carry such a URL are read the same way.
- **The engine only takes pictures in with a request.** It does not fetch `http(s)` URLs or files (HTTP 400), take screenshots or look
  again by itself. A loop of look, fix and look again is the client's job: each look is a new request with a new picture.
- **Size.** One image token covers 40 x 40 pixels, rounded up: 800 x 600 is 300 tokens, 1280 x 720 is 576, 1920 x 1080 is 1,296,
  2560 x 1440 is 2,304. Image tokens count in the context and in `usage.prompt_tokens`. Send a smaller picture when the detail is
  not needed: it is faster and leaves more context.
- **A large picture is shrunk, not refused.** A picture of up to 2,304 image tokens (`INK_IMAGE_TOKENS_EACH`; 2560 x 1440) goes to
  the model pixel for pixel. A larger one is shrunk by the server first, with its aspect ratio kept, until it fits: 3840 x 2160
  becomes 2560 x 1440 (2,304 tokens), 8000 x 600 becomes 6933 x 520 (2,262 tokens). A picture is never enlarged. The model itself
  has no resize rule, so this is the same as a client that shrinks a screenshot before it sends it. The filter is Lanczos3, which
  keeps small text sharp (0.3 s for a 3840 x 2160 picture); the log says `ink vision: picture of 3840 x 2160 pixels (5184 image
  tokens) shrunk to 2560 x 1440`. On the test pictures (a chart and a settings screen of 3840 x 2160 with text 24 to 96 pixels
  high) every line of text is read correctly after the shrink. For the full detail of a larger picture set
  `INK_IMAGE_TOKENS_EACH` higher (5184 takes 3840 x 2160 whole); the encoder then needs much longer (table below).
- **Limits of a request:**

  | Setting | Default | What it limits | Beyond it |
  |---|---|---|---|
  | `INK_IMAGE_MAX` | 64 | pictures in one request | the oldest pictures are left out |
  | `INK_IMAGE_TOKENS` | 49152 | image tokens in one request, all pictures together | the oldest pictures are left out |
  | `INK_IMAGE_TOKENS_EACH` | 2304 | image tokens of one picture (2560 x 1440) | the picture is shrunk to fit |
  | `INK_IMAGE_MB` | 20 | size of one picture file, MiB | HTTP 400 |
  | `INK_IMAGE_OVERFLOW` | `drop` | what happens beyond the first two limits | `reject`: HTTP 400 instead |

  A picture file with a side above 16,384 pixels or with more than 100 million pixels is refused (HTTP 400) before it is decoded.
  The request body may be up to 256 MiB.
- **A long session is never refused for its pictures.** An agent sends the whole conversation again with every turn, with every
  screenshot it ever looked at. When a request holds more than `INK_IMAGE_MAX` pictures or more than `INK_IMAGE_TOKENS` image
  tokens, the server leaves out the oldest pictures of the conversation (message order, then part order). Each one becomes the
  text `[earlier picture omitted]` in its place; the text around it stays. The newest pictures are always kept.
  - **They go in batches, for the prompt cache.** The number left out is the smallest multiple of B that makes the rest fit, with
    B = `INK_IMAGE_MAX` / 4 (16 by default; at least 1). With the defaults and screenshots of 1280 x 800: up to 64 pictures none is
    left out, with 65 to 80 pictures the oldest 16, with 81 to 96 the oldest 32. Dropping one picture a turn would change the
    start of the prompt on every turn, and every turn would process the whole conversation again. In batches the start stays
    the same for 16 turns, and the prompt cache resumes each of them.
  - **Measured** (a conversation that gains one 1280 x 800 screenshot a turn, up to 90 pictures; no turn refused): on the
    RTX 5060 Ti the turns between two batches start their answer after 1.6 s (median; 2.0 s at 40,000 tokens; prompts of up to
    43,800 tokens, all but the new picture from the cache). The turn that drops a batch (65 and 81) processes its 33,700 to
    34,400 tokens again once: 32 s. The kept pictures are not encoded again there (their rows are in host memory). On the
    RTX PRO 4000 (70 pictures): 1.1 s, and 20 s for the turn that drops a batch.
  - The log says `image overflow: 16 of 65 pictures left out, oldest first (k = 16, batch B = 16); kept 49 pictures, 31360 image
    tokens`. The response says it in `usage.prompt_tokens_details.images_omitted` (the key is absent when nothing was left out).
  - If the limits are smaller than one batch of the request's pictures, the server leaves out as many as needed, one by one, and
    logs a warning. A picture that is left out is still checked from its file header (format, size), but it is not decoded.
  - `INK_IMAGE_OVERFLOW=reject` brings back the refusal: HTTP 400 with the limit's name.
- **Each picture is encoded once.** The vision encoder runs when the picture first arrives:

  | picture | image tokens | RTX 5060 Ti | RTX PRO 4000 |
  |---|---|---|---|
  | 512 x 512 | 169 | 0.19 s | 0.14 s |
  | 800 x 600 | 300 | 0.29 s | 0.23 s |
  | 1280 x 720 | 576 | 0.56 s | 0.43 s |
  | 1920 x 1080 | 1,296 | 1.8 s | 1.4 s |
  | 2560 x 1440 (the default limit; also 3840 x 2160 after the shrink) | 2,304 | 5.7 s | not measured |
  | 3840 x 2160 with `INK_IMAGE_TOKENS_EACH=5184` | 5,184 | 40 s | 39 s |

  The time grows with the square of the size. After that the picture's rows are part of the prompt and the
  [prompt cache](#prompt-cache) keeps them like text: a later turn of the conversation does not encode or prefill the picture again
  (the log line `ink vision` says `inside the resumed prompt cache (no encode)`). The same holds for a conversation that comes back
  from the [park pool](#park-pool).
- **Two pictures are never mixed up.** The cache knows a picture by the SHA-256 of its pixels and size. Another picture of the same
  size gets no cache hit; the same picture sent again as another file does.
- **Measured on the RTX 5060 Ti** (16 GB layout, 128K context, replies identical to the same server with `INK_PCACHE=0`):
  a question about a 1920 x 1080 picture starts its answer after 3.6 s (1.8 s of it the encoder), the next turn after 0.27 s. For a
  1280 x 720 screenshot: 1.5 s, then 0.29 s. A screenshot that arrives 32,000 tokens into a conversation: 2.9 s, and the turn after
  it 0.41 s (28 s without the cache). Eight pictures (9,421 tokens): 9.7 s for the first question, 1.7 s for a second, different
  question (7.4 s without the cache).
- **Rows of recent pictures are also kept in host memory** (`INK_IMAGE_CACHE_MB`, 256 MiB by default, 6 MiB for a 1920 x 1080 picture),
  so a picture that must be prefilled again (another lane, an edit in front of it, `INK_PCACHE=0`) is not encoded again.
- **Memory.** The encoder's weights (605 MiB) are read into pinned host memory with the first picture (0.6 s) and stay there. Card
  memory is used only while a picture is encoded: 77 MiB for 800 x 600, 215 MiB for 1920 x 1080, 540 MiB for 3840 x 2160; it is
  freed before the prompt is prefilled. **On the 16 GB card** the free memory may be short of that; then some experts move to host
  memory for the encode and move back after it (0 on an idle engine up to 1920 x 1080; 23 experts for a 1280 x 720 picture at 32,000
  tokens of context, 36 for 1920 x 1080 after other requests, 110 for 3840 x 2160; 0.2 to 0.7 s for both moves). The context limit
  does not change.
- **With lanes** the encoder runs between two passes, so the other streams pause for the encode time.
- **Drafting works as with text.** Replies with the draft head on equal replies with it off.
- **EXIF orientation is not applied** (the model's reference processor does not apply it either). Rotate a photo before sending it
  if its orientation is stored in EXIF.
- **JPEG.** The Rust decoder gives slightly other pixels than libjpeg-turbo, which the reference uses: on the test photo 23% of the
  values differ, by up to 4 of 255 levels. See [Accuracy](#accuracy) for what that does to the answers.
- **Transparency** is drawn on a grey checkerboard, as the reference does. 16-bit PNG and CMYK JPEG are converted to 8-bit RGB;
  that conversion was not compared with the reference.

### Several requests at once (lanes)
With `INK_LANES=4` (the default on the 24 GB layout and on a DGX Spark) four requests decode at once. Each lane has its own K/V cache, short-convolution
states, draft-head caches and prompt cache; one forward carries the rows of every lane, so the weights are read once for all of them. Every
stream's tokens are the same as the same request alone, whatever the other lanes do (`ink-run lanescheck` and the lanes gate check it row
for row and stream for stream). A fifth request waits until a lane is free.
A sampled request with a `seed` gets the same tokens on any lane and beside any other requests, with the draft head on or off.

Measured on an RTX PRO 4000 through the server: four different prompts started together, 256 tokens each at temperature 0, tokens per
second over the whole batch (mean of 6 batches):

| | 1 stream | 2 streams | 4 streams | 4 streams, one lane (queued) |
|---|---|---|---|---|
| short code, draft on | 128 | 169 | 188 | 119 |
| short prose, draft on | 84 | 110 | 159 | 83 |
| short code, draft off | 83 | 110 | 163 | 82 |
| short prose, draft off | 83 | 113 | 170 | 82 |

- **Each stream** runs at 41 to 57 tokens/s when four decode together (code with the draft head at the top of that range).
- **Time to first token** with four short requests arriving together: 0.16 s at most, against 6.3 s for the last one when they queue.
- **Alternating conversations.** Four conversations of 10K tokens, one request at a time, switching every turn: each switch answers after
  0.06 to 0.09 s (also at 41K tokens); with one lane every switch prefills the whole conversation again (5.6 s; 32 s at 41K tokens).
- **One stream** runs as fast with four lanes as with one (within 1%).
- **Drafting with several lanes.** A lane drafts only when the drafts it adds are worth more than the time they add to the shared pass:
  prose usually decodes plainly, code drafts. `INK_LANE_DRAFT=all` drafts as a single stream would.
- **A long prompt arriving while others decode** is prefilled in chunks of `INK_LANE_PREFILL_ROWS` rows (128 on the RTX cards) beside them: a 30K-token
  prompt arriving while three streams decode starts its answer after 45 to 48 s, and the three streams slow to about 11 tokens/s meanwhile.
  With 256 rows: 34 to 36 s, the others about 8 tokens/s; with 64: 73 to 75 s, the others about 16. With one lane the same prompt waits for
  the others to finish (77 s in this test). Alone, a prompt is prefilled at full speed.

#### Park pool

When a lane has to take a new conversation, the one it held is parked instead of dropped: its global K/V and draft
context for the prompt it cached, with the newest prompt-cache snapshot, move to pinned host memory (`FERRITE_PARK_GB`, 8 GB by default;
16 GB of host memory stays available). On a DGX Spark nothing is copied: the conversation's memory stays where it is and is mapped back
when it returns (16 GB by default). A later request that continues a parked conversation brings it back into a lane and resumes
there, with the same tokens as a fresh run. The least recently used conversations leave the pool when it is full; one that has left
is prefilled again from the start. Measured on an RTX PRO 4000 with 8 conversations taking turns on the four lanes, one request at a time:

| conversation | parked size | park / bring back | answer starts after a switch, pool on | pool off |
|---|---|---|---|---|
| 10K tokens | 364 MiB | 15 / 15 ms | 0.15 s | 5.8 s |
| 41K tokens | 876 MiB | 55 / 54 ms | 0.29 s | 32 s |
| 94K tokens | 1.7 GiB | 122 / 121 ms | 0.52 s | 111 s |

`FERRITE_PARK_GB=0` turns the pool off. It is in memory only and is empty after a restart.

#### Memory

On the RTX cards each lane beyond the first takes 0.31 GiB of card memory, plus about 17 KiB a token of its context (13 KiB of 6-bit K/V and
4 KiB for the draft head). A request reserves its prompt plus `max_tokens` (32,768 when the request sends none) and starts when the lanes'
reservations fit; the idle lanes' cached conversations are dropped first. One request alone always gets the whole 131,072-token context.

| card | lanes | what fits |
|---|---|---|
| RTX PRO 4000, 24 GB | 4 | 210,000 tokens over all lanes: one stream at 131,072, or four at about 52,000 each (prompt plus answer) |
| RTX 5060 Ti, 16 GB | 1 (default) | each extra lane moves about 125 experts to host memory; the lanes share one 131,072-token context |

On the 16 GB card lanes trade single-stream speed for concurrency (short code, draft on): one lane 108 tokens/s alone and 101 for four
queued streams; two lanes 106 alone and 117 for four; four lanes 95 alone and 136 for four. `INK_LANES=2` or `4` turns them on.

#### On a DGX Spark

Four lanes by default, the same way. Four different prompts started together, 256 tokens each at temperature 0, tokens per second over
the whole batch (median of 6 batches), against the same server with `INK_LANES=1`:

| | 1 stream | 2 streams | 4 streams | 4 streams, one lane (queued) | each of the four |
|---|---|---|---|---|---|
| short code, draft on | 63 | 80 | 92 | 57 | 23 to 38 |
| short prose, draft on | 42 | 56 | 81 | 42 | 20 to 21 |
| 8K code, draft on | 62 | 69 | 76 | 31 | 19 to 30 |
| short code, draft off | 42 | 57 | 82 | 42 | 21 |
| short prose, draft off | 42 | 59 | 86 | 42 | 22 |
| 8K code, draft off | 40 | 53 | 71 | 25 | 18 |

- **Time to first token** with four requests arriving together: 0.32 s at most, against up to 19 s (short) and 35 s (8K) for the last
  one when they queue. The 8K prompts stay in their lanes' prompt caches from batch to batch; with one lane they replace each other.
- **One stream** runs as fast with four lanes as with one.
- **More lanes:** `INK_LANES=8` raises the total further while each stream slows: 8K prose with the draft head off runs at 73 tokens/s
  for four streams, 84 for six and 94 for eight (12 each); short code with the draft head on at 92, 100 and 108. Each lane takes about
  0.53 GiB plus about 17 KiB a token of its context; memory does not limit the lanes on a DGX Spark.
- **A long prompt arriving while others decode** (a 30K-token prompt, three streams decoding, which run at 20 to 28 tokens/s before it):

  | `INK_LANE_PREFILL_ROWS` | the long prompt's first token | the three streams meanwhile |
  |---|---|---|
  | 128 | 77 to 84 s | 6 to 7 tokens/s |
  | 256 (the default on a DGX Spark) | 53 to 55 s | 4 to 6 tokens/s |
  | 512 | 39 to 40 s | 3 to 4 tokens/s |
  | 1024 | 32 to 33 s | 2 to 3 tokens/s |
- **Park pool**, 8 conversations taking turns on the four lanes, one request at a time:

  | conversation | parked size | park / bring back | answer starts after a switch, pool on | pool off |
  |---|---|---|---|---|
  | 10K tokens | 380 MiB | 6 to 19 / 4 ms | 0.15 s | 5.1 s |
  | 51K tokens | 1.05 GiB | 17 / 16 ms | 0.18 s | 39 s |
  | 118K tokens | 2.1 to 2.2 GiB | 34 to 61 / 32 to 46 ms | 0.26 s | 152 s |

  At 118K tokens, four conversations sit in the lanes and four or five in the pool (10.7 GiB).

## Results

**What to expect:** with the draft head on, a short code request runs at 138 tokens/s on the RTX PRO 4000, 110 on the RTX 5060 Ti and 72 on a
DGX Spark; prose runs at the plain speed, about 83, 70 and 43. At a 126K-token context the RTX PRO 4000 still decodes code at 117 tokens/s and the
RTX 5060 Ti at 64.

Measured with Ferrite 0.9.0. "Draft" is the model's own draft head, on by default. It is lossless: the text is the same with it and without it
(`ink-run selftest` checks that), and a sampled request with a `seed` gives the same text with the draft head on or off, whatever the
draft length and the other lanes. Drafted speed depends on the text; plain speed does not.

**Short prompts** (chat requests of 25 to 160 tokens), tokens per second generated:

| | RTX PRO 4000 24 GB, draft on | RTX PRO 4000, draft off | RTX 5060 Ti 16 GB, draft on | RTX 5060 Ti, draft off |
|---|---|---|---|---|
| Code (a Python function with tests) | 138.1 | 83.2 | 110.4 | 69.2 |
| Agent turn (a tool result to answer from) | 126.2 | 82.6 | 98.8 | 69.1 |
| Prose (a short essay) | 83.9 | 83.3 | 68.5 | 69.9 |

**Long contexts.** Prefill is prompt tokens per second; decode is generated tokens per second for a code and for a prose continuation.

RTX PRO 4000 Blackwell, 24 GB:

| Context | Prefill | Decode, draft on: code | Decode, draft on: prose | Decode, draft off: code | Decode, draft off: prose |
|---|---|---|---|---|---|
| 8K | 1,939 | 158.6 | 102.1 | 78.5 | 78.3 |
| 32K | 1,446 | 141.6 | 72.3 | 74.2 | 74.5 |
| 64K | 1,049 | 141.9 | 137.5 | 68.5 | 68.6 |
| 126K | 703 | 117.0 | 76.8 | 59.6 | 59.6 |

RTX 5060 Ti, 16 GB:

| Context | Prefill, draft on | Prefill, draft off | Decode, draft on: code | Decode, draft on: prose | Decode, draft off: code | Decode, draft off: prose |
|---|---|---|---|---|---|---|
| 8K | 1,163 | 1,275 | 128.0 | 76.5 | 64.6 | 63.5 |
| 32K | 879 | 957 | 97.1 | 61.9 | 58.4 | 59.8 |
| 64K | 638 | 699 | 95.9 | 77.9 | 54.0 | 52.5 |
| 126K | 413 | 453 | 64.0 | 81.3 | 44.5 | 43.6 |

- The contexts are 8,305 / 32,753 / 65,581 / 126,216 tokens for the code columns and 8,017 / 32,465 / 65,293 / 125,928 for the prose columns.
- Both cards run the default 6-bit K/V cache.

DGX Spark (GB10), the launcher's defaults (24 GB layout, 6-bit K/V, prompts in chunks of 2,048 rows), every request prefilled from the start
(`INK_PCACHE=0`):

| Context | Prefill (first token) | Decode, draft on: code | Decode, draft on: prose | Decode, draft off: code | Decode, draft off: prose |
|---|---|---|---|---|---|
| short (23 to 30 tokens) | (0.16 s) | 73.5 | 42.8 | 43.1 | 43.1 |
| 8K | 2,174 to 2,199 (3.6 to 3.8 s) | 66.4 | 51.0 | 40.3 | 40.3 |
| 34K | 1,553 to 1,561 (21.6 to 21.9 s) | 78.6 | 40.9 | 37.4 | 37.5 |
| 100K | 890 to 892 (112 s) | | | | |

- The contexts are 8,193 / 33,955 tokens for code and 7,850 / 33,612 for prose, and 99,963 tokens of prose for the 100K row; 256 tokens
  generated, median of three after one unscored request.
- A DGX Spark decodes at about half the 24 GB card's speed: every token reads about 3.6 GB of weights and K/V, and its memory is slower. The
  draft head gains less there for the same reason: each draft it verifies reads more experts.
- Measured with the GPU clock limited to 2,200 MHz (`nvidia-smi -lgc 0,2200`). Without the limit the GPU settles near 2,440 MHz: decode 2%
  and prefill 6% faster, at about 67 W instead of 43 W.
- Prefill is the same with the draft head on or off to within 1% on the 24 GB card; on the 16 GB card the head leaves less room for the prompt
  chunk, so it prefills about 8% slower.

**The draft head and the text.** The speed of a drafted request is the model's own acceptance of its draft, and it changes with the text by a
factor of two; the draft-off numbers do not. The long-context tables use mixed real text, and the tail of each prompt is code or prose from this
repository, so the answer sometimes repeats what the prompt contains and sometimes does not: the 64K prose figure on the 24 GB card, 137.5, is an
answer that quotes the text before it, and the 126K prose figure on the 16 GB card, 81.3, likewise. Code that repeats earlier code (an edit, a
rename, a refactor) is the best case, measured with `ink-run bench --kind code` (a source file's text repeated through the context, default
layouts, 6-bit K/V on both cards):

| Context | RTX PRO 4000: draft on / off | RTX 5060 Ti: draft on / off |
|---|---|---|
| 32K | 171.6 / 74.5 | 119.1 / 58.5 |
| 64K | 145.4 / 68.6 | 93.3 / 52.7 |

- **Drafting never costs more than 3% against plain decoding** in any row of the tables above: the 24 GB card's 32K prose answer is 3% behind
  (72.3 against 74.5) and the 16 GB card's short prose answer 2% behind (68.5 against 69.9). A rule suspends drafting when it is slower (the
  plain decode time is measured at long contexts and the draft speed is averaged over many passes), but a short answer has too few passes to
  settle.
- **What drafting gains.** On short prompts it raises code by about 60% and agent turns by about 50%, and on code that repeats earlier code
  by up to 2.3 times (171.6 against 74.5 tokens/s at 32K on the 24 GB card). It does little for prose.
- **The draft head on the 16 GB card** also takes card memory that the experts could use, so more experts are read from host memory; the
  launcher keeps the head at every context.
- **The 16 GB card slows down with the context** because more experts live in host memory (with the draft head off, 134 of the 3,072 at 8K
  tokens, 218 at 32K, 320 at 64K and 492 at 124K, read over PCIe 4.0 x8 when picked; `ink-run bench` prints the count), and every token reads
  K/V through the 24 attention layers. Prefill falls with the context for the same reasons; the 16 GB card prefills in chunks of 256 rows.
- **The 24 GB card's** decode without the head falls from 78 tokens/s at 8K to 60 at 126K because every token reads the keys and values of
  the eight global layers (1.6 GiB of 6-bit K/V at 128K).

**bf16 K/V on the 24 GB card** (`INK_KV=bf16`). Keep the 6-bit default unless short-context drafted speed matters more to you than speed at
depth and 2.5 GiB of card memory. Same server and method:

| RTX PRO 4000, tokens/s unless marked | 6-bit K/V (default) | bf16 K/V |
|---|---|---|
| K/V cache at 128K | 1.6 GiB | 4.1 GiB |
| Plain decode, 64K / 126K | 68.5 / 59.6 | 60.8 / 49.9 |
| Prefill, 64K / 126K | 1,049 / 703 | 1,027 / 636 |
| Drafted, short prompts: code / agent / prose | 138.1 / 126.2 / 83.9 | 149.7 / 169.1 / 89.3 |
| Drafted code, 8K | 158.6 | 184.8 |
| Drafted code, 64K / 126K | 141.9 / 117.0 | 104.8 / 113.1 |
| Drafted prose, 64K / 126K | 137.5 / 76.8 | 138.5 / 100.8 |
| KL on code ([Accuracy](#accuracy)) | 0.0037 | 0.0041 |

At 64K and 126K the drafted figures move with the text.

**How it was measured.** AMD EPYC 7313 x2, 128 GB DDR4; an RTX PRO 4000 Blackwell (24 GB, 145 W) on PCIe 4.0 x16 and an RTX 5060 Ti (16 GB,
180 W) on PCIe 4.0 x8; driver 595.99. `tools/serve-ink.sh` with its defaults for the card, 131,072 tokens of context configured, measured at
the server by `tools/bench-ink.py`: temperature 0, 200 tokens generated, the second of two identical requests, one request at a time, nothing
else on the card or the host. Contexts are real text: this repository's Markdown and Rust files in turn, no block twice, whose last 1,024
tokens are code or prose that the answer continues. DGX Spark: DGX OS, driver 580.178, CUDA 13.0; the same server and client (streamed,
256 tokens generated, the median of three after one unscored request).

## Accuracy

| | KL code | KL prose | KL long text | perplexity, code / prose (reference) |
|---|---|---|---|---|
| 24 GB layout, as published, 6-bit K/V (the default) | 0.0037 | 0.0074 | 0.0093 (16,384 tokens) | 2.074 / 4.899 (2.068 / 4.889) |
| the same with bf16 K/V (`INK_KV=bf16`) | 0.0041 | 0.0083 | 0.0087 (16,384 tokens) | 2.077 / 4.895 |
| 16 GB layout (layers 20-23 requantised to NVFP4), 6-bit K/V | 0.0050 | 0.0113 | 0.0044 (a 4,096-token text: a shorter text than the rows above, not comparable with them) | 2.079 / 4.921 |

- Requantising the four MXFP8 layers costs about 0.001 to 0.004 of KL; 6-bit K/V costs less than 0.001 (it is not a loss in every text).
- All are far below 0.05, the limit this port set for itself.
- A DGX Spark gives the same figures to the fourth digit (0.0037 / 0.0074 with 6-bit K/V, 0.0041 / 0.0083 with bf16 K/V), and its prompt
  logits are the same bits at every prompt chunk size.

**How it was measured.** The cost of the formats is KL(reference || Ferrite) on the next-token distribution over 1,024-token slices of source
code and of Wikipedia text and a 16,384-token mixed text, free-running (the engine's own routing), and perplexity next to the reference's. The
reference is the published model run with Hugging Face transformers (weights as shipped, experts dequantised to bf16, bf16 activations). The
engine's expert GEMMs use the activation quantization the SM120 expert kernels of the model's reference code use, which is why a run with the
shipped formats is not bit-equal to the reference and costs a few thousandths of KL.

**Pictures.** Measured on 8 prompts with pictures (a chart, a screenshot, shapes, two photos, a tall list, a picture with
transparency, two charts in one request), each with a question and its reply.

| KL over the text rows behind the picture | lowest | highest | mean |
|---|---|---|---|
| 24 GB layout | 0.010 | 0.037 | 0.017 |
| 16 GB layout | 0.016 | 0.045 | 0.025 |
| the reference against itself with the activation quantization of its own SM120 expert kernels | 0.011 | 0.031 | 0.021 |
| the reference against itself with 1% of the picture's row values moved by one bf16 step | 0.002 | 0.008 | 0.004 |

- Prompts with pictures are more sensitive than text: the reference itself moves by 0.021 on them when it runs the experts the way
  the engine does (0.004 to 0.007 on text). The engine is in that range, and below the 0.05 limit on every prompt.
- The most likely next token agrees with the reference on 95 to 99% of those rows.
- The vision encoder is exact: its rows equal the reference's fp32 encoder to 3e-5 (`ink-run vision-check`). The reference for the
  table is that fp32 encoder, run on the pixels the engine decoded, in front of the reference text model.
- A JPEG decoded by the engine instead of libjpeg-turbo moves the reference by 0.009 on the test photo: less than the engine's own
  distance.

**Repeating it.** `ink-run kl` repeats the measurement, but it needs reference logits in the format described at the top of
`crates/fe-ink/src/bin/ink-run.rs`. They are not shipped, so the measurement cannot be repeated from the repository alone.

## Limits

- **Pictures:** PNG and JPEG in `data:` URLs only. The model sees at most 64 pictures and 49,152 image tokens a request; a request
  with more is taken, and its oldest pictures are left out (16 at a time). A picture above 2,304
  image tokens (2560 x 1440) is shrunk to that size; the encoder takes 5.7 s for it on the RTX 5060 Ti ([Images](#images)). No audio, no video. Image input was not run on a DGX Spark yet.
- **Four requests at once** on the 24 GB card and on a DGX Spark (up to eight there with `INK_LANES`), one on the 16 GB card by default;
  more wait in the queue. Each lane's prompt cache follows
  one conversation; more conversations than lanes take turns through the park pool, and one that has left the pool is prefilled again
  (see [Several requests at once](#several-requests-at-once-lanes)). A prompt arriving while
  others decode is prefilled more slowly, and slows them (see [Several requests at once](#several-requests-at-once-lanes)).
- **`tool_choice` is a hint.** `required` or a named function is not enforced. Tool calls follow the model's JSON format; a model that reaches
  for a tool the request did not offer is stopped (see [Using the server](#using-the-server)).
- **It reasons by default.** `reasoning_effort` or `enable_thinking: false` lowers the reasoning without removing it.
- On the 16 GB card, speed falls with the context because more experts are read from host memory ([Results](#results)).
- The context is 131,072 tokens, an eighth of the model's 1,048,576-token maximum.

## What it runs on

| | RTX PRO 4000 Blackwell, 24 GB | RTX 5060 Ti, 16 GB | DGX Spark (GB10) |
|---|---|---|---|
| Experts | as published: NVFP4 in layers 0-19, MXFP8 in layers 20-23, all on the card | all NVFP4 (layers 20-23 are requantised from MXFP8 when the model loads); the ones that do not fit stay in pinned host memory and are read from there when picked, more of them as the context grows | as on the 24 GB card |
| Attention and output head | FP8 (per-row scales), as published | the same | the same |
| K/V cache of the 8 global layers | 6-bit (1.6 GiB at 128K), mapped as the context grows (`INK_KV=bf16` keeps bf16: 4.1 GiB at 128K) | 6-bit (1.6 GiB at 128K), mapped as the context grows | as on the 24 GB card |
| Draft head | bf16, 0.75 GiB | FP8, 0.45 GiB (and 4.5 KiB a position), at every context | bf16, 0.75 GiB |
| Prompt chunk | 256 rows | 256 rows | 2,048 rows |
| Lanes | 4 | 1 | 4 |
| Context | 131,072 | 131,072 | 131,072 |
| Host memory | any | 32 GB recommended: up to about 4 GiB of experts are pinned | the GPU's memory is the system memory: about 25 GB with four lanes, plus the context and the park pool |

Everything else is the same code on every GPU. Prompts and generated tokens go through the same arithmetic, so the logits of a token do not
depend on the chunk it was prefilled in, on speculative rows beside it or on the cache state (`ink-run selftest` checks that), and greedy text
with the draft head is the text without it.

## Advanced settings

These are the engine's own settings, its whole interface. The launcher sets them for the layout, so most users change none of them
(booleans: set and not `0`, `false`, `off` or `no`).

| Variable | What |
|---|---|
| `INK_DIR` | the model folder [`$FERRITE_MODELS/Inkling-2-Mini`] |
| `INK_G16` | all experts NVFP4 (the 16 GB layout's weights) |
| `INK_SLOTS` | experts in a shared pool that can move between the card and host memory (with `INK_G16`) |
| `INK_HOST_EXPERTS=N` | experts parked in host memory at the start, the coldest first by the shipped heat table (`crates/fe-ink/heat/heat_share.json`, or `INK_HEAT_SHARE=file`) |
| `INK_FIT_MB=n` | keep n MiB of the card free: before the K/V cache or the draft head grows, experts move to host memory to make room; when a request ends they move back |
| `INK_KVQ6` | 6-bit K/V cache of the global layers (rotated absmax groups of 32, 104 bytes a head and token) |
| `INK_KV_LAZY` | map the K/V cache as the context grows instead of reserving it |
| `INK_CHUNK=n` | prompt rows a forward [256; 2,048 on a DGX Spark (the launcher); `0` sizes it from the card's free memory after loading] |
| `INK_RESERVE_MB=n` | with `INK_CHUNK=0`: memory kept free for the draft head |
| `INK_NODRAFT` | no draft head |
| `INK_DRAFT_FP8` (0, 1 or 2) | draft head precision: 0 bf16, 1 FP8 matrices, 2 FP8 matrices and the Markov head's tables [2 on a card under 20 GB, else 0] |
| `INK_DRAFT_MAX_CTX=n` | the draft head is loaded only for requests of at most n tokens (prompt plus answer) [no limit] |
| `INK_FOLD=file` | where the draft head's fold tables are kept [`$INK_DIR/fold.safetensors`] |
| `INK_PCACHE=0` | prompt cache off: every request prefills its whole prompt [on] |
| `INK_IMAGE_MAX`, `INK_IMAGE_TOKENS`, `INK_IMAGE_TOKENS_EACH`, `INK_IMAGE_MB` | the limits of image input [64 pictures, 49152 image tokens a request, 2304 a picture (a larger picture is shrunk to fit), 20 MiB a file] ([Images](#images)) |
| `INK_IMAGE_OVERFLOW=drop\|reject` | a request with more pictures or image tokens than the limits: leave out its oldest pictures, a quarter of `INK_IMAGE_MAX` at a time, or refuse it with HTTP 400 [`drop`] |
| `INK_IMAGE_CACHE_MB=n` | host memory for the rows of recent pictures [256; `0` keeps none] |
| `INK_VISION_SCORE_MB=n` | card memory for the vision encoder's attention scores while a picture is encoded [64] |
| `INK_PCACHE_EVERY=n` | prompt-cache marks every n positions besides the prompt end [8192; `0` = the prompt end only] |
| `INK_PCACHE_GB=x` | pinned host memory for prompt-cache snapshots, 172 MiB each [2: 11 snapshots] |
| `INK_LANES=n` | requests decoded at once, 1 to 8 (needs `INK_KV_LAZY`) [the launcher: 4 on the 24 GB layout, 1 on the 16 GB layout] |
| `INK_LANE_PREFILL_ROWS=n` | rows of a prompt chunk prefilled beside decoding lanes [128; 256 on an integrated GPU such as the DGX Spark's] |
| `INK_LANE_SPEC_ROWS=n` | speculative rows the conv buffers of lanes 2 and up hold [16] |
| `INK_LANE_DRAFT=all` | with several lanes, draft as a single stream would instead of by the shared pass's cost |
| `INK_DRAFT_FIXED=n` | always verify n drafts and never suspend drafting (tests; output does not change) |
| `INK_LANES_PROF` | log the mean planning, forward and finishing time of the multi-lane passes |
| `INK_PARK_GB=x` | the park pool with several lanes, GB (the launcher's `FERRITE_PARK_GB`) [8; 16 with unified memory; `0` = off] |
| `INK_PARK_RESERVE_GB=x` | host memory the pool leaves available [16] |
| `INK_PARK_RESIDENT=0\|1` | keep parked conversations in device memory and map them back instead of copying [on with unified memory, else off] |
| `INK_PARK_MIN=n` | conversations shorter than n tokens are not parked [512] |
| `INK_SGEMM_WIDE=0` | the small decode GEMMs stream 64 instead of 256 k-columns a pipeline stage (same output; for comparisons) |
| `INK_SCONV_PAR=0` | the short convolution of a prompt chunk walks its rows in one thread a channel (same output; for comparisons) |
| `INK_ATTN_GROUP_MIN` | rows from which a batch takes the grouped prompt attention: 64, because smaller batches (draft verification) are faster on the earlier path (same output either way) |
| `INK_ATTN_GROUP=0`, `INK_ATTN_SPLIT=0`, `INK_ATTN_ORDER=0` | the prompt attention runs one block a key tile instead of one a group of tiles, both passes in one kernel, or in the earlier block order (same output; for comparisons) |
| `INK_ATTN_UNPACK=0` | the prompt attention unpacks the 6-bit K/V per tile instead of once a layer (on integrated GPUs, where it is on: a scratch of 2 x context x 2 KiB) |
| `INK_MEM_LOG`, `INK_SLOT_LOG` | print the card's memory by step while loading; print each expert-residency change |

## About the model

24 layers of width 2,304, each with sliding-window (1,024) or global attention (24 query heads, 8 key/value heads of 128, a learned
relative-position bias, attention sinks, short causal convolutions on the keys, values and layer inputs) and a mixture of 128 experts, 8 picked a
token (sigmoid router, soft-capped SwiGLU). Layers 6 to 17 run three times (the loop), the layers hand their results to block attention residuals,
and a small gate passes a message from one pass to the next. 200,058 tokens of vocabulary, logits soft-capped at 30. The published model also
reads images, and so does this engine ([Images](#images)).

It runs in a separate engine in the same server (`fe-server --engine ink`, crates `fe-ink` and `fe-inkk`), with its own Rust kernels compiled
to PTX like the rest (`k-ink`, `k-inkm`, `k-draft`), and it uses the model's own draft head for speculative decoding.
