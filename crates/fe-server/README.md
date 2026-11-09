# fe-server

OpenAI-compatible HTTP server for ferrite (Qwen3.8-27B on the DGX Sparks). axum + tokio; the HF
`tokenizers` crate for `tokenizer.json`; minijinja (+ minijinja-contrib pycompat) for the model's
`chat_template.jinja`. The server owns everything text-shaped; the engine behind the `Engine` trait only
turns prompt token ids into generated token ids.

## Build and run

```bash
export PATH=$HOME/.cargo/bin:$PATH
cd crates/fe-server
cargo build --release
# mock engine (no GPU), on the port the clients expect
./target/release/fe-server --mock --model-dir "$FERRITE_EXL3_DIR" --port 8000   # any folder with the tokenizer
# or: cargo run --release -- --mock --model-dir "$FERRITE_EXL3_DIR" --port 8090
```

Real model (inside the ferrite workspace, after folding, see the end; loads the weights on the GPU, so
`free -g` first and keep >= 20 GB free; KV is `slots x max-model-len` bf16 = 32 GB at 4 x 128K):

```bash
cargo build --release -p fe-server --features ferrite
# the EXL3 dir has the same tokenizer.json / chat_template.jinja / generation_config.json as the NVFP4 dir;
# --model-dir defaults to $FERRITE_EXL3_DIR, --nvfp4-dir / --draft-dir to $FERRITE_NVFP4_DIR / $FERRITE_DRAFT_DIR
. tools/site.sh
flock $FERRITE_GPU_LOCK ./target/release/fe-server --engine ferrite \
  --model-dir "$FERRITE_EXL3_DIR" --slots 4 --max-model-len 131072 --port 8000
# or the one-command start with the defaults filled in: scripts/serve.sh (docs/quickstart.md)
```

| flag | default | |
|---|---|---|
| `--model-dir DIR` | (required) | `tokenizer.json`, `chat_template.jinja` (or `tokenizer_config.json` `chat_template`), `generation_config.json` |
| `--host` / `--port` | `0.0.0.0` / `8000` | |
| `--served-model-name` | `spark` | the id reported everywhere; requests may name any model |
| `--max-concurrent N` | engine `max_batch` (mock: 4) | requests generating at once; the rest wait in a FIFO queue (never 503) |
| `--max-model-len N` | `131072` | prompt + output tokens; `max_tokens` is clamped to what is left |
| `--mock` | off | deterministic mock engine (below) |
| `--engine` | `ferrite` | without `--mock`: `ferrite` = core's `fe_model::sched::Scheduler`, `ferrite-batch` = fe-server's `BatchEngine` over `fe_model::model::Engine` (both need `--features ferrite`); `fake` = `BatchEngine` over a CPU fake model (random tokens, real vocab; the real-engine path without a GPU, `--mock-delay-ms` per step, `--mock-prefill-tps` per row) |
| `--weights-dir DIR` | `--model-dir` | real engine weights (config.json + safetensors, BF16 or EXL3) |
| `--slots N` / `--prefill-chunk N` | `4` / `2048` | real engine: cache slots (= concurrent requests), prompt tokens per prefill chunk |
| `--mock-delay-ms` / `--mock-burst` / `--mock-prefill-tps` | `5` / `1` / `0` | mock step delay, tokens per step (>1 imitates spec decoding), simulated prefill speed |

Logging: `RUST_LOG=info` (default) prints one line per request queued / done (prompt and output tokens,
cached tokens, finish reason, time to first token, tok/s) and one per client disconnect;
`RUST_LOG=fe_server=debug` also logs the tail of every rendered prompt and the scheduler's slot choices.
SIGINT / SIGTERM stop accepting requests and exit after at most 5 s.

## Endpoints

- `GET /health` (also `/v1/health`): `{"status":"ok","max_batch_size":4,"max_model_len":131072,"running":n,"queued":m,...}`
  (quickbench reads `max_batch_size`).
- `GET /v1/models`, `GET /v1/models/{id}`: one model, id `spark`, with `max_model_len` / `context_length`.
- `POST /v1/chat/completions`: stream (SSE `data: {...}\n\n`, final `data: [DONE]`) and non-stream.
- `POST /v1/completions`: raw prompt (string, `[string]`, token ids or `[[ids]]`; one prompt), stream and
  non-stream, `echo`. Default `max_tokens` 16 (OpenAI); chat defaults to the rest of the context window.

Response shapes follow OpenAI: `id` (`chatcmpl-…` / `cmpl-…`), `object`, `created`, `model`,
`choices[].index/message|delta|text/logprobs(null)/finish_reason`, `usage.prompt_tokens /
completion_tokens / total_tokens / prompt_tokens_details.cached_tokens`. Streams start with a
`{"role":"assistant","content":""}` delta, end with an empty delta carrying `finish_reason`, then (with
`stream_options.include_usage`) a `choices: []` usage chunk; with include_usage every other chunk has
`"usage": null`. Errors are `{"error":{"message","type","param","code"}}` with HTTP 400 / 500.

### What is supported

- **Chat template**: `chat_template.jinja` rendered with jinja2 semantics (`trim_blocks`, `lstrip_blocks`, no
  autoescape), `raise_exception`, a `tojson` that writes exactly Python's `json.dumps(ensure_ascii=False)`,
  `strftime_now`, the tokenizer's special tokens as variables. Variables: `messages`, `tools`,
  `add_generation_prompt` (request field, default true), every key of `chat_template_kwargs`. Messages are
  normalized like vLLM: tool-call `arguments` JSON strings → objects, `developer` → `system`, an assistant
  `reasoning` field → `reasoning_content`, non-text content parts → a text placeholder.
  `continue_final_message` is supported. Byte-for-byte equal to HF `apply_chat_template` on 16 cases (below).
- **Thinking**: on by default (prompt ends `<think>\n`); `chat_template_kwargs.enable_thinking=false` (or
  top-level `enable_thinking: false`, or `reasoning_effort: "none"`) pre-fills `<think>\n\n</think>\n\n`.
  `chat_template_kwargs.reasoning_effort` (`xhigh` / `medium` / `low`) goes to the template; an OpenAI
  top-level `reasoning_effort` is mapped (`minimal`/`low` → low, `medium` → medium, `high`… → xhigh).
  Reasoning text is returned as `reasoning_content` (message or delta), content after `</think>` as
  `content`, like vLLM's qwen3 parser (`<tool_call>` inside thinking also ends it). Whether output starts
  in thinking is read off the rendered prompt, so `continue_final_message` works too.
- **Tool calls**: the template's `<tool_call><function=NAME><parameter=ARG>VALUE</parameter></function></tool_call>`
  format → OpenAI `tool_calls` (`call_<24 random>`, type `function`, `arguments` a JSON string). Values are
  typed with the tool's JSON schema exactly like vLLM's qwen3 parser (`coerce_to_schema_type`: null >
  integer > number > boolean > object > array > string, `anyOf`/`enum` inference, fallbacks). Several calls
  per reply (parallel), text before a call is content, a bare `<function=` and a JSON body inside
  `<tool_call>` are accepted, a cut-off call at the end of output still reports its parameters.
  `finish_reason` becomes `tool_calls`. `tool_choice`: `auto` (default), `none` (tools not shown to the
  model, no parsing), `required` (system-prompt hint), `{"type":"function","function":{"name":…}}` (only
  that tool offered + hint); `parallel_tool_calls: false` reports only the first call. Streaming: one delta
  with `id`, `type`, `function.name` and `arguments: ""` as soon as `<function=NAME>` is complete, then one
  delta with the full `arguments` when `</function>` arrives; text before the call streams as content.
- **Sampling** (passed to the engine): `max_tokens` / `max_completion_tokens`, `min_tokens`, `temperature`,
  `top_p`, `top_k` (≤0 = off), `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`,
  `seed`, `stop_token_ids`, `ignore_eos`. Defaults from `generation_config.json` (temperature 1.0, top_p 0.95,
  top_k 20); stop ids = its `eos_token_id` list (248046, 248044) + `<|im_end|>`. Validated (400 otherwise):
  temperature >= 0 (below 1e-5 = greedy, as vLLM), top_p in (0, 1], min_p in [0, 1],
  repetition_penalty > 0, finite penalties. Before `min_tokens` the stop ids are banned in sampling
  (vLLM behaviour) rather than passed through.
- **Stop strings** (`stop`, string or list) are handled in the server on the detokenized text: output is cut
  before the first stop string (text that could still become one is held back), `finish_reason` `stop`,
  the engine request is cancelled. `include_stop_str_in_output` keeps it. `max_tokens` → `length`.
- **Usage**: `prompt_tokens` = tokens of the rendered prompt; `completion_tokens` = generated tokens, plus
  one for the stop token when generation ended on eos (as vLLM counts); `cached_tokens` from the engine.
- **Streaming detokenization** never splits a UTF-8 character and always concatenates to exactly
  `decode(all ids)`: for byte-level tokenizers (Qwen) every id's raw bytes are tabled at load (checked
  against the decoder for all 248,077 ids) and the longest valid UTF-8 prefix is emitted, invalid bytes
  becoming U+FFFD as in a full decode; other tokenizers use the TGI/vLLM window method. Special tokens are
  skipped (`skip_special_tokens`, default true).
- **Concurrency**: at most `--max-concurrent` (4) requests run; more queue in FIFO order and stream as soon as
  a slot frees. The slot is owned by the engine's `TokenSink`, so it frees only when the engine has really
  let go of the request.
- **Cancellation**: a client disconnect (stream or not, also while queued) drops the request's driver,
  which sets the `CancelToken` and closes the sink (`send_tokens` then returns false).

## Engine trait (`src/engine.rs`)

```rust
pub struct SamplingParams {
    pub max_tokens: usize,          // hard cap on generated tokens (stop token not counted)
    pub min_tokens: usize,          // no stop-token stop before this many
    pub temperature: f32,           // 0.0 = greedy
    pub top_p: f32,                 // 1.0 = off
    pub top_k: usize,               // 0 = off
    pub min_p: f32,                 // 0.0 = off
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,    // 1.0 = off
    pub seed: Option<u64>,
    pub stop_token_ids: Vec<u32>,   // eos ids + request stop_token_ids
}

#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);          // .cancel(), .is_cancelled()

pub struct GenRequest {
    pub id: String,                 // for logs
    pub prompt_ids: Vec<u32>,       // the full rendered prompt (prefix-cache key)
    pub params: SamplingParams,
    pub cancel: CancelToken,
}

pub enum FinishReason { Stop(u32), Length, Cancelled, Error(String) }
pub struct Finish { pub reason: FinishReason, pub cached_prompt_tokens: usize }
pub enum GenEvent { Tokens(Vec<u32>), Finished(Finish) }

pub struct TokenSink { /* tokio unbounded sender + the request's concurrency slot */ }
impl TokenSink {
    pub fn send_tokens(&self, ids: Vec<u32>) -> bool;  // false = nobody listens any more: stop
    pub fn is_closed(&self) -> bool;
    pub fn finish(self, finish: Finish);              // exactly once; frees the slot
}

pub struct EngineInfo { pub name: String, pub max_batch: usize, pub max_context: usize }

pub trait Engine: Send + Sync + 'static {
    fn info(&self) -> EngineInfo;
    /// Start generating; return immediately. Results go to `sink`: zero or more
    /// `send_tokens`, then exactly one `finish`.
    fn submit(&self, req: GenRequest, sink: TokenSink);
}
```

Contract, in short: `submit` is called concurrently from many tasks (at most `max_batch` requests are in
flight); the engine generates on its own threads and calls the sink synchronously (it never blocks and
needs no async runtime). It stops when it samples one of `stop_token_ids` (→ `Finish{Stop(id)}`, the id is
not sent), after `max_tokens` (→ `Length`), or when `cancel.is_cancelled()` / `send_tokens` returns false
(→ `Cancelled`). Several tokens per `send_tokens` call are fine (speculative decoding). Dropping the sink
without `finish` is reported to the client as an engine error. The server does all text work, so the
engine never needs the tokenizer.

### Real engine adapters (`src/ferrite.rs`, feature `ferrite`)

- **`--engine ferrite`** (default): `sched::FerriteEngine`, the 1:1 adapter over core's
  `fe_model::sched::Scheduler` exactly as `crates/fe-model/ENGINE_API.md` section 2 describes (sink
  closure → `TokenSink`, `CancelToken::flag()` → `Arc<AtomicBool>`, `SamplingParams` → `Sampling`).
- **`--engine ferrite-batch`**: fe-server's own `BatchEngine` (`src/batch.rs`) over `FeModel`, a
  `StepModel` impl over the slot-level `fe_model::model::Engine` (CUDA context created on the engine
  thread). Continuous batching: each step admits queued requests into free slots (a slot whose cached
  tokens are a strict prefix of the new prompt is reused: prompt cache, only the rest is prefilled,
  `cached_tokens` reported), runs one forward with one decode row per generating request plus prefill
  chunks, then samples (device argmax when all rows are plain greedy, else bf16 logits to the host and
  `src/sampler.rs`: penalties, temperature, top-k, top-p, min-p, seeded SplitMix64). Stops on stop ids /
  `max_tokens` / context end / cancellation; a failed step fails the running requests and forces resets.
  Same design as core's scheduler; kept as a fallback and for `--engine fake`.
- Both **compile-checked against core d7cbe99** in a local scratch merge with `Cargo.workspace.toml`
  (`cargo build/test --release -p fe-server --features ferrite`: builds, 75/75 tests); neither has been run
  on a GPU from here.

## MockEngine (`--mock`)

Deterministic; sees only prompt ids (decodes them back) and answers from what the rendered prompt shows:
tools offered and the last user message names one (`get_weather` matches "weather") → optional reasoning,
`I'll use NAME for this.` and one `<tool_call>` per named tool with arguments made up from the schema (a
place after "in", 3 for integers, true for booleans, …); last turn a tool result → `Based on the tool
result: …`; otherwise reasoning (if thinking is on) + `Hello from the ferrite mock engine! You said: "…"`
with multi-byte UTF-8. If the user text contains "write" or "long" (quickbench's prompts do) it continues
with numbered filler lines until `max_tokens`. Tokens come one per `--mock-delay-ms` (5 ms ≈ 185 tok/s).

## Tests

```bash
cargo test --release             # unit + fixtures + in-process e2e (needs the model dir for the tokenizer;
                                 # FE_MODEL_DIR overrides $FERRITE_NVFP4_DIR, the folder the tests read)
tools/e2e_curl.sh http://localhost:8090                         # against a running --mock server
python3 tools/e2e_openai.py "$FERRITE_URL"                      # the official openai client (pip install openai)
tools/e2e_pi.sh "$FERRITE_URL"                                  # the Pi agent, with a throwaway PI_CODING_AGENT_DIR
python3 tools/gen_template_refs.py <model_dir> tests/fixtures/hf_template_refs.json  # regenerate the HF references
```

- `src/*` unit tests (51): a 3000-case parser fuzz (random markup fragments, whole vs 1/3/7-char vs random
  chunks must agree), UTF-8 assembly vs lossy decoding, degenerate sampler params (min_p > 1, T = 1e-40,
  NaN logits) never panic, stop ids banned before `min_tokens`, `BatchEngine` over a CPU fake model whose next token hashes the whole slot
  history (solo greedy = reference with chunked prefill, 7 concurrent requests on 4 slots = their solo
  runs and share forwards, slot prefix reuse is exact and prefills only the tail, cancellation, too-long
  prompt, seeded sampling), the sampler (greedy / top-k 1, seeded determinism + distribution, top-p, min-p,
  penalties, bf16 rows), Python-exact JSON / float repr, stop strings (split across chunks, multi-byte,
  earliest of several, include), schema typing of tool arguments, the reasoning / tool-call parser (each
  case fed whole, char by char and in 3/7-char chunks, results must agree: reasoning split, thinking off,
  unterminated thinking, single / parallel calls, text before a call, call inside thinking, tools off, bare
  `<function=`, two functions in one block, cut-off call, JSON body, parallel off, name-before-args
  streaming), message normalization, tool_choice, thinking detection, mock plans.
- `tests/template_hf.rs`: 16 HF references (plain, system, multi-turn with `reasoning_content`, tools,
  tools + system, assistant tool_calls + tool results (typed, nested args), tool call with text then answer
  then a new query, thinking off, thinking off + tools, reasoning_effort low / medium, content parts,
  unicode / specials, empty system, no generation prompt) match **byte for byte**, and the token ids match
  HF's; template exceptions surface as 400s.
- `tests/e2e_batch.rs` (3): router → `BatchEngine` → fake slot model with the real vocab: completions usage
  and text equal the fake's solo reference, a follow-up prompt that extends the slot reports
  `cached_tokens` = prompt + output - 1 and still produces the reference, chat stream == non-stream,
  identical greedy requests repeat, seeded host sampling repeats, 6 streams on 4 slots (peak 2-4), a
  dropped stream frees its slot.
- `tests/sampler_speed.rs`: host sampling on the full 248,320 vocab: greedy 0.11 ms, Qwen defaults
  (t 1, top-k 20, top-p 0.95) 0.47 ms, top-p only 1.0 ms, no filters 1.1 ms, min-p 0.26 ms per token.
- `tests/utf8_stream.rs` (4): CJK / emoji / ZWJ / flags / math letters streamed token by token reproduce the
  full decode with no U+FFFD ever emitted; the byte table equals `decode([id])` for every id (with and
  without special tokens); 400 random byte-fragment-heavy id sequences stream to exactly `decode(ids)`.
- `tests/e2e_mock.rs` (13, in process through the router): non-stream + stream chat equal, reasoning split,
  usage arithmetic, thinking off, tool calls stream + non-stream with typed args, parallel calls, tool results
  round trip, tool_choice none / named / required-without-tools, stop strings (stream / not / include),
  max_tokens, completions (string, ids, echo, stream, default 16), OpenAI errors, 5-token bursts, 6 streams on
  4 slots (peak ≤ 4), dropped stream cancels the engine.

Results 2026-11-03 on node B: `cargo test --release` 75 passed / 0 failed (standalone, and again inside
a scratch merge with core d7cbe99 with `--features ferrite`); `e2e_curl.sh` all 21 checks
passed; `e2e_openai.py` (openai 3.14.1) all 25 checks passed (incl. 5 concurrent streams: 4 start at once,
the 5th waits for a slot); `e2e_pi.sh` with Pi 1.2.0 (openai-completions, thinkingFormat
qwen-chat-template, as the owner's spark providers): plain reply, and a full agent turn: thinking →
`read` tool call → Pi runs it → tool result (with `reasoning_content` sent back) → final answer; quickbench code / prose / 4 concurrent streams: prompt token counts, TTFT,
~185 tok/s per stream, 740 tok/s aggregate (mock at 5 ms/token). quickbench against `--engine fake`
(BatchEngine, 20 ms/step, 3000 rows/s simulated prefill), 8K context: 1 stream 2,890 tok/s prefill /
47.6 tok/s decode; 4 streams all finish, prefills run one after another (first tokens at 2.8 / 5.6 / 8.5 /
10.9 s) and decode rows ride along with the prefill chunks.

## Folding into the ferrite workspace

The root `Cargo.toml` on `core` has `members = ["crates/*"]`. In this directory:
`mv Cargo.workspace.toml Cargo.toml && rm Cargo.lock` (drops the standalone `[workspace]` table, takes
`anyhow` / `tokenizers` from the workspace, adds optional `fe-cuda` / `fe-model` behind the `ferrite`
feature). Verified: with that manifest on top of core d7cbe99, `cargo build --release -p fe-server
--features ferrite` and `cargo test --release -p fe-server --features ferrite` pass. `serde_json`'s
`preserve_order` feature (tool-schema key order) is then unified across the workspace build.

## Known gaps

- The real engine paths (`--engine ferrite` / `ferrite-batch`) are compile-checked only; neither has run
  on a GPU from here.
- Prompt cache = whole-slot prefix reuse only (the GDN state cannot be rewound): a request whose prompt
  diverges from the slot's history anywhere re-prefills from scratch.
- `n > 1`, `logprobs`, `response_format` / guided decoding, `suffix`, batched completion prompts: not
  supported (`n`/`suffix`/batched return 400, the rest are ignored).
- `tool_choice: required` / named function is a prompt hint, not constrained decoding.
- Tool-call arguments stream in one piece at `</function>` (not character by character).
- Images / audio in messages become a text placeholder (text-only model).
