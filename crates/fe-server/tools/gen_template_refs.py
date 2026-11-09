#!/usr/bin/env python3
"""Generate HF reference renders of the model's chat template for fe-server's template tests.

usage: gen_template_refs.py <model_dir> <out.json>

Each case holds OpenAI-format request pieces (messages with tool_call arguments as JSON strings,
tools, chat_template_kwargs) and the HF `apply_chat_template` output text plus its token ids.
Tool-call arguments are parsed to dicts before rendering, as vLLM does (the template iterates them).
"""
import copy
import json
import sys

from transformers import AutoTokenizer

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns <temp> & \"conditions\".",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Paris or Zürich"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1},
                "safe": {"type": "boolean"},
                "filters": {"type": "object", "properties": {"site": {"type": "string"}}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "threshold": {"type": "number"},
            },
            "required": ["query"],
        },
    },
}

CASES = [
    {"name": "plain", "messages": [{"role": "user", "content": "Hello, who are you?"}]},
    {"name": "system", "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}]},
    {"name": "multi_turn_reasoning", "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Name a prime number."},
        {"role": "assistant", "content": "7", "reasoning_content": "The user wants a prime. 7 is prime.\n"},
        {"role": "user", "content": "Another one?"}]},
    {"name": "multi_turn_no_reasoning_field", "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."}]},
    {"name": "tools_present", "tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "What's the weather in Paris?"}]},
    {"name": "tools_with_system", "tools": [WEATHER], "messages": [
        {"role": "system", "content": "You are a weather bot."},
        {"role": "user", "content": "Weather in Oslo?"}]},
    {"name": "tool_calls_and_results", "tools": [WEATHER, SEARCH], "messages": [
        {"role": "system", "content": "Use tools when useful."},
        {"role": "user", "content": "Weather in Paris, and search for cafés there."},
        {"role": "assistant", "content": "", "reasoning_content": "Two tools needed.",
         "tool_calls": [
             {"id": "call_1", "type": "function",
              "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\", \"unit\": \"celsius\"}"}},
             {"id": "call_2", "type": "function",
              "function": {"name": "search", "arguments": json.dumps(
                  {"query": "cafés in Paris", "max_results": 3, "safe": True, "filters": {"site": "example.com"},
                   "tags": ["food", "coffee"], "threshold": 0.5})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18, \"conditions\": \"sunny\"}"},
        {"role": "tool", "tool_call_id": "call_2", "content": "1. Café de Flore\n2. Les Deux Magots"}]},
    {"name": "tool_call_with_text_then_answer_then_new_query", "tools": [WEATHER], "messages": [
        {"role": "user", "content": "Is it cold in Oslo?"},
        {"role": "assistant", "content": "Let me check the weather.", "reasoning_content": "Need the weather.",
         "tool_calls": [{"id": "call_a", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"location\": \"Oslo\"}"}}]},
        {"role": "tool", "tool_call_id": "call_a", "content": "-3 C, snow"},
        {"role": "assistant", "content": "Yes, it is -3 C with snow.", "reasoning_content": "Tool says -3 C."},
        {"role": "user", "content": "And in Rome?"}]},
    {"name": "thinking_off", "kwargs": {"enable_thinking": False}, "messages": [
        {"role": "user", "content": "Say hi."}]},
    {"name": "thinking_off_tools", "kwargs": {"enable_thinking": False}, "tools": [SEARCH], "messages": [
        {"role": "system", "content": "Short answers."},
        {"role": "user", "content": "Search for rust tokio."}]},
    {"name": "reasoning_effort_low", "kwargs": {"reasoning_effort": "low"}, "messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Capital of France?"}]},
    {"name": "reasoning_effort_medium", "kwargs": {"reasoning_effort": "medium"}, "messages": [
        {"role": "user", "content": "Capital of Spain?"}]},
    {"name": "content_parts", "messages": [
        {"role": "system", "content": [{"type": "text", "text": "System as parts."}]},
        {"role": "user", "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]}]},
    {"name": "unicode_and_specials", "messages": [
        {"role": "user", "content": "  Résumé: naïve café — 日本語 🦀 \"quotes\" <tags> & {braces}  "}]},
    {"name": "empty_system", "messages": [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hi"}]},
    {"name": "no_generation_prompt", "add_generation_prompt": False, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello.", "reasoning_content": "greet"}]},
]


def to_hf(messages):
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"] else {}
    return out


def main():
    model_dir, out_path = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(model_dir)
    refs = []
    for case in CASES:
        agp = case.get("add_generation_prompt", True)
        text = tok.apply_chat_template(to_hf(case["messages"]), tools=case.get("tools"), tokenize=False,
                                       add_generation_prompt=agp, **case.get("kwargs", {}))
        ids = tok(text, add_special_tokens=False)["input_ids"]
        refs.append({**case, "add_generation_prompt": agp, "text": text, "ids": ids})
        print(f"{case['name']}: {len(text)} chars, {len(ids)} tokens")
    with open(out_path, "w") as f:
        json.dump(refs, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
