"""Offline tests for the OpenAI <-> Glean translation layer.

No network access required. Run: python test_translation.py
"""
import sys
from pathlib import Path

# Allow running this script directly from its subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import proxy

failures = []

# Structural tests below read turns in chronological order for clarity, so the
# newest-first reversal is exercised separately at the end.
proxy.REVERSE_MESSAGES = False


def check(label, actual, expected):
    ok = actual == expected
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"      expected: {expected!r}")
        print(f"      actual  : {actual!r}")
        failures.append(label)


def check_true(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        if detail:
            print(f"      {detail}")
        failures.append(label)


print("--- content flattening ---")
check("plain string", proxy._text_of("hello"), "hello")
check(
    "content blocks (Claude Code shape)",
    proxy._text_of([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]),
    "a\nb",
)
check("none", proxy._text_of(None), "")
check_true(
    "image block does not crash",
    "image omitted" in proxy._text_of([{"type": "image_url", "image_url": {"url": "x"}}]),
)

print("\n--- message building ---")
msgs = proxy.build_glean_messages(
    [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ],
    [],
)
check("turn count", len(msgs), 3)
check("authors", [m["author"] for m in msgs], ["USER", "GLEAN_AI", "USER"])
check_true(
    "system prompt rides on the LATEST user turn",
    "Be terse." in msgs[-1]["fragments"][0]["text"]
    and "second" in msgs[-1]["fragments"][0]["text"],
    msgs[-1]["fragments"][0]["text"][:120],
)
check_true(
    "oldest user turn is left clean",
    "Be terse." not in msgs[0]["fragments"][0]["text"],
)
check_true("messageType set", all(m["messageType"] == "CONTENT" for m in msgs))

tools = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}]
with_tools = proxy.build_glean_messages([{"role": "user", "content": "read app.js"}], tools)
text = with_tools[-1]["fragments"][0]["text"]
check_true("tool schema injected", "read_file" in text and proxy.TOOL_CALL_OPEN in text)
check_true("user request preserved", "read app.js" in text)

# A tool result must come back as a USER turn, since Glean has no tool role.
after_tool = proxy.build_glean_messages(
    [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "app.js"}'}}
        ]},
        {"role": "tool", "name": "read_file", "content": "file contents here"},
    ],
    tools,
)
check("tool result becomes USER", after_tool[-1]["author"], "USER")
check_true("tool output carried", "file contents here" in after_tool[-1]["fragments"][0]["text"])
check_true(
    "assistant tool call replayed",
    "read_file" in after_tool[1]["fragments"][0]["text"],
)
check("assistant author", after_tool[1]["author"], "GLEAN_AI")

print("\n--- response extraction (real Glean shape) ---")
real = {
    "messages": [{
        "author": "GLEAN_AI",
        "fragments": [{"text": "PROXY_OK"}, {}],
        "messageType": "CONTENT",
    }],
    "chatId": "abc",
}
check("real response", proxy.extract_text(real), "PROXY_OK")
check(
    "status messages skipped",
    proxy.extract_text({"messages": [
        {"author": "GLEAN_AI", "messageType": "UPDATE", "fragments": [{"text": "Searching..."}]},
        {"author": "GLEAN_AI", "messageType": "CONTENT", "fragments": [{"text": "answer"}]},
    ]}),
    "answer",
)
check(
    "user echo skipped",
    proxy.extract_text({"messages": [
        {"author": "USER", "messageType": "CONTENT", "fragments": [{"text": "my question"}]},
        {"author": "GLEAN_AI", "messageType": "CONTENT", "fragments": [{"text": "the reply"}]},
    ]}),
    "the reply",
)
check("empty fragments tolerated", proxy.extract_text({"messages": [
    {"author": "GLEAN_AI", "fragments": [{}, {"text": "x"}, {}]}]}), "x")

print("\n--- tool call parsing ---")
one = proxy.extract_tool_calls(
    f'{proxy.TOOL_CALL_OPEN}\n{{"name": "read_file", "arguments": {{"path": "a.js"}}}}\n{proxy.TOOL_CALL_CLOSE}'
)
check("single call parsed", one, [{"name": "read_file", "arguments": {"path": "a.js"}}])

many = proxy.extract_tool_calls(
    f'{proxy.TOOL_CALL_OPEN}{{"name":"a","arguments":{{}}}}{proxy.TOOL_CALL_CLOSE}'
    f'{proxy.TOOL_CALL_OPEN}{{"name":"b","arguments":{{"x":1}}}}{proxy.TOOL_CALL_CLOSE}'
)
check("parallel calls parsed", [c["name"] for c in many], ["a", "b"])
check("prose is not a tool call", proxy.extract_tool_calls("Just a normal answer."), [])
check_true(
    "fenced json fallback",
    proxy.extract_tool_calls('```json\n{"name": "ls", "arguments": {}}\n```')
    == [{"name": "ls", "arguments": {}}],
)

print("\n--- malformed tool markup salvage ---")
# The opener is right but the closer is borrowed from another dialect, or is
# missing entirely. The payload is still valid JSON, so the call is recovered
# rather than delivered to the client as raw markup in assistant text.
check(
    "mismatched closing tag recovered",
    proxy.extract_tool_calls(
        f'{proxy.TOOL_CALL_OPEN}\n{{"name": "read", "arguments": {{"filePath": "x"}}}}\n</invoke>'
    ),
    [{"name": "read", "arguments": {"filePath": "x"}}],
)
check(
    "missing closing tag recovered",
    proxy.extract_tool_calls(
        f'{proxy.TOOL_CALL_OPEN}\n{{"name": "read", "arguments": {{"filePath": "x"}}}}'
    ),
    [{"name": "read", "arguments": {"filePath": "x"}}],
)
check(
    "two malformed calls recovered in order",
    [c["arguments"]["filePath"] for c in proxy.extract_tool_calls(
        f'{proxy.TOOL_CALL_OPEN}\n{{"name": "read", "arguments": {{"filePath": "a"}}}}\n</invoke>\n'
        f'{proxy.TOOL_CALL_OPEN}\n{{"name": "read", "arguments": {{"filePath": "b"}}}}\n</invoke>'
    )],
    ["a", "b"],
)
# Brace matching, not a lazy regex: a `}` inside a JSON string must not be
# mistaken for the end of the object, or nested arguments get truncated.
check(
    "brace inside a JSON string",
    proxy.extract_tool_calls(
        f'{proxy.TOOL_CALL_OPEN}\n{{"name": "bash", "arguments": {{"command": "echo }}"}}}}\n</invoke>'
    ),
    [{"name": "bash", "arguments": {"command": "echo }"}}],
)
# The JSON payload here is `"echo \"}\""`: an escaped quote, then a brace that
# must not be read as the end of the object. Doubling the backslashes again would
# emit an escaped backslash followed by a bare quote, which is invalid JSON and
# unparseable by any tier.
check(
    "escaped quote then brace",
    proxy.extract_tool_calls(
        f'{proxy.TOOL_CALL_OPEN}\n{{"name": "bash", "arguments": {{"command": "echo \\"}}\\""}}}}\n</invoke>'
    ),
    [{"name": "bash", "arguments": {"command": 'echo "}"'}}],
)
check(
    "runaway trailing tags do not duplicate the call",
    proxy.extract_tool_calls(
        f'{proxy.TOOL_CALL_OPEN}\n{{"name": "read", "arguments": {{"filePath": "x"}}}}\n</invoke>'
        + "</invoke>" * 50
    ),
    [{"name": "read", "arguments": {"filePath": "x"}}],
)
check(
    "well-formed markup still takes the strict path",
    proxy.extract_tool_calls(
        f'{proxy.TOOL_CALL_OPEN}\n{{"name": "read", "arguments": {{"filePath": "x"}}}}\n{proxy.TOOL_CALL_CLOSE}'
    ),
    [{"name": "read", "arguments": {"filePath": "x"}}],
)

print("\n--- completion assembly ---")
comp = proxy._completion("hello", [])
check("text finish_reason", comp["choices"][0]["finish_reason"], "stop")
check("content passed through", comp["choices"][0]["message"]["content"], "hello")
check_true("usage present", "usage" in comp)

tool_comp = proxy._completion(
    f'{proxy.TOOL_CALL_OPEN}{{"name":"ls","arguments":{{}}}}{proxy.TOOL_CALL_CLOSE}',
    [{"name": "ls", "arguments": {}}],
)
check("tool finish_reason", tool_comp["choices"][0]["finish_reason"], "tool_calls")
check_true(
    "arguments serialized as JSON string",
    isinstance(tool_comp["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], str),
)
check("markup stripped from content", tool_comp["choices"][0]["message"]["content"], None)

print("\n--- stream delta handling ---")
d = proxy.Delta()
check("incremental chunks", [d.push("Hel"), d.push("lo"), d.push("!")], ["Hel", "lo", "!"])
d2 = proxy.Delta(cumulative=True)
check(
    "cumulative chunks de-duplicated (opt-in)",
    [d2.push("Hel"), d2.push("Hello"), d2.push("Hello!")],
    ["Hel", "lo", "!"],
)
check("accumulated text", d2.text, "Hello!")
d3 = proxy.Delta()
check("repeat ignored", [d3.push("a"), d3.push("a")], ["a", ""])

# Glean's fragments are incremental, so a chunk that happens to prefix the next
# one is real text, not a resend. Treating it as cumulative dropped the overlap.
d4 = proxy.Delta()
check(
    "prefix-shaped chunks kept whole when incremental (the default)",
    [d4.push("I"), d4.push("I'll")],
    ["I", "I'll"],
)

print("\n--- SSE / NDJSON parsing ---")
check("sse line", list(proxy._iter_json_objects('data: {"a": 1}')), [{"a": 1}])
check("ndjson line", list(proxy._iter_json_objects('{"a": 2}')), [{"a": 2}])
check("done sentinel", list(proxy._iter_json_objects("data: [DONE]")), [])
check("blank line", list(proxy._iter_json_objects("")), [])
check("garbage tolerated", list(proxy._iter_json_objects("not json")), [])

print("\n--- newest-first ordering (Glean answers messages[0]) ---")
proxy.REVERSE_MESSAGES = True
ordered = proxy.build_glean_messages(
    [
        {"role": "user", "content": "oldest"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "newest"},
    ],
    [],
)
check_true(
    "newest turn is sent first",
    ordered[0]["fragments"][0]["text"].endswith("newest"),
    ordered[0]["fragments"][0]["text"][:80],
)
check("oldest turn is sent last", ordered[-1]["fragments"][0]["text"], "oldest")
check("authors reversed", [m["author"] for m in ordered], ["USER", "GLEAN_AI", "USER"])

proxy.REVERSE_MESSAGES = False
chrono = proxy.build_glean_messages(
    [{"role": "user", "content": "oldest"}, {"role": "user", "content": "newest"}], []
)
check("chronological mode still available", chrono[0]["fragments"][0]["text"], "oldest")

print("\n--- model resolution ---")
# A Claude name must never reach Glean verbatim: it rejects the unknown value by
# discarding the whole agentConfig, silently re-enabling company retrieval.
check("glean id passthrough", proxy.resolve_model_set("OPUS_5_MS")[0], "OPUS_5_MS")
check("glean/ prefix stripped", proxy.resolve_model_set("glean/SONNET_5_MS")[0], "SONNET_5_MS")
check("suffixed claude name", proxy.resolve_model_set("claude-opus-5-fast"), ("OPUS_5_MS", "FAST"))
check("base claude name", proxy.resolve_model_set("claude-sonnet-5")[0], "SONNET_5_MS")
check("dated claude name", proxy.resolve_model_set("claude-opus-4-5-20251101")[0], "OPUS_5_MS")
check(
    "longest prefix wins over base name",
    proxy.resolve_model_set("claude-opus-5-advanced"),
    ("OPUS_5_MS", "ADVANCED"),
)
check(
    "unknown name falls back to the configured default",
    proxy.resolve_model_set("gpt-4o")[0],
    proxy.GLEAN_MODEL_SET_ID,
)

print("\n--- token estimates ---")
check("chars to tokens", proxy._estimate_tokens_from_chars(400), 100)
check("empty text still costs 1", proxy._estimate_tokens(""), 1)
check(
    "prompt chars counted from the assembled payload",
    proxy._prompt_chars({"messages": [
        {"fragments": [{"text": "abc"}, {}]},
        {"fragments": [{"text": "de"}]},
    ]}),
    5,
)

usage = proxy._completion("hello there", [], "claude-opus-5-fast", 400)
check("model echoed back", usage["model"], "claude-opus-5-fast")
check("prompt tokens reported", usage["usage"]["prompt_tokens"], 100)
check_true(
    "total is the sum of both sides",
    usage["usage"]["total_tokens"]
    == usage["usage"]["prompt_tokens"] + usage["usage"]["completion_tokens"],
    str(usage["usage"]),
)

print("\n--- persona sync with SKILL.md ---")
# Behaviour otherwise depends on which persona path fired.
skill = (proxy.ROOT / "coding-agent-backend" / "SKILL.md").read_text(encoding="utf-8")
skill_body = skill.split("---", 2)[-1].strip()
check_true(
    "HARNESS_PROMPT matches the SKILL.md body",
    skill_body == proxy.HARNESS_PROMPT.strip(),
    "SKILL.md and HARNESS_PROMPT have diverged",
)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("All translation tests passed.")
