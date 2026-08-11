"""End-to-end tests for the Anthropic /v1/messages endpoint (Claude Code).

Exercises the request shapes Claude Code actually sends, and validates the SSE
event sequence, which Claude Code will reject if it is malformed.

Run: python tests/test_anthropic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import json

import httpx

from proxy import app

failures = []

TOOLS = [{
    "name": "bash",
    "description": "Run a shell command on the user's machine and return its output.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The command"}},
        "required": ["command"],
    },
}]


def report(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        print(f"      {detail}")
    if not ok:
        failures.append(label)


async def sse_events(client, body):
    """Collect (event_name, payload) pairs from a streaming response."""
    events = []
    async with client.stream("POST", "/v1/messages", json=body) as resp:
        assert resp.status_code == 200, await resp.aread()
        name = None
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                name = line[7:].strip()
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
    return events


async def main():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://p", timeout=300) as c:

        print("--- non-streaming text ---")
        r = await c.post("/v1/messages", json={
            "model": "claude-opus-4-5-20251101",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Reply with exactly: ANTHROPIC_OK"}],
        })
        ok = r.status_code == 200
        report("status 200", ok, "" if ok else r.text[:300])
        if ok:
            d = r.json()
            report("type=message", d.get("type") == "message")
            report("role=assistant", d.get("role") == "assistant")
            report("model echoed back", d.get("model") == "claude-opus-4-5-20251101",
                   f"model={d.get('model')}")
            report("stop_reason=end_turn", d.get("stop_reason") == "end_turn")
            blocks = d.get("content") or []
            report("has a text block", bool(blocks) and blocks[0]["type"] == "text",
                   f"content={json.dumps(blocks)[:120]}")
            report("usage reported", "input_tokens" in (d.get("usage") or {}))

        print("\n--- system prompt as blocks (Claude Code shape) ---")
        r = await c.post("/v1/messages", json={
            "model": "claude-opus-4-5-20251101",
            "max_tokens": 100,
            "system": [{"type": "text", "text": "You always answer in one word."}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What color is the sky on a clear day?"}
            ]}],
        })
        report("block system + content accepted", r.status_code == 200,
               "" if r.status_code == 200 else r.text[:200])

        print("\n--- streaming event sequence ---")
        events = await sse_events(c, {
            "model": "claude-opus-4-5-20251101",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "stream": True,
        })
        names = [n for n, _ in events]
        report("starts with message_start", names[:1] == ["message_start"], f"{names[:6]}")
        report("ends with message_stop", names[-1:] == ["message_stop"], f"{names[-4:]}")
        report("has content_block_start", "content_block_start" in names)
        report("has content_block_delta", "content_block_delta" in names)
        report("has content_block_stop", "content_block_stop" in names)
        report("has message_delta", "message_delta" in names)
        text = "".join(
            p["delta"]["text"] for n, p in events
            if n == "content_block_delta" and p.get("delta", {}).get("type") == "text_delta"
        )
        report("streamed text non-empty", bool(text.strip()), f"text={text[:80]!r}")
        stops = [p["delta"]["stop_reason"] for n, p in events if n == "message_delta"]
        report("stop_reason=end_turn", stops == ["end_turn"], f"{stops}")
        starts = [p for n, p in events if n == "content_block_start"]
        report("block indexes start at 0", starts and starts[0]["index"] == 0)

        print("\n--- tool use (non-streaming) ---")
        r = await c.post("/v1/messages", json={
            "model": "claude-opus-4-5-20251101",
            "max_tokens": 300,
            "tools": TOOLS,
            "messages": [{"role": "user", "content": "How many files are in this directory?"}],
        })
        ok = r.status_code == 200
        report("status 200", ok, "" if ok else r.text[:300])
        tool_use = None
        if ok:
            d = r.json()
            tool_use = next((b for b in d.get("content", []) if b["type"] == "tool_use"), None)
            report("returned a tool_use block", tool_use is not None,
                   f"content={json.dumps(d.get('content'))[:160]}")
            if tool_use:
                report("tool name is bash", tool_use["name"] == "bash", tool_use["name"])
                report("input is an object", isinstance(tool_use["input"], dict),
                       f"input={tool_use['input']}")
                report("stop_reason=tool_use", d.get("stop_reason") == "tool_use")
                report("id looks like toolu_*", str(tool_use["id"]).startswith("toolu_"))

        print("\n--- tool use (streaming) ---")
        events = await sse_events(c, {
            "model": "claude-opus-4-5-20251101",
            "max_tokens": 300,
            "tools": TOOLS,
            "messages": [{"role": "user", "content": "List the files here."}],
            "stream": True,
        })
        blocks = [p["content_block"] for n, p in events if n == "content_block_start"]
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        report("streamed a tool_use block", bool(tool_blocks),
               f"blocks={[b['type'] for b in blocks]}")
        json_deltas = [
            p["delta"]["partial_json"] for n, p in events
            if n == "content_block_delta" and p["delta"].get("type") == "input_json_delta"
        ]
        report("sent input_json_delta", bool(json_deltas), f"{json_deltas[:1]}")
        if json_deltas:
            try:
                parsed = json.loads("".join(json_deltas))
                report("input JSON parses", isinstance(parsed, dict), f"{parsed}")
            except json.JSONDecodeError as exc:
                report("input JSON parses", False, str(exc))
        stops = [p["delta"]["stop_reason"] for n, p in events if n == "message_delta"]
        report("stop_reason=tool_use", stops == ["tool_use"], f"{stops}")
        # Every started block must be closed, or Claude Code errors.
        opened = [p["index"] for n, p in events if n == "content_block_start"]
        closed = [p["index"] for n, p in events if n == "content_block_stop"]
        report("all blocks closed", sorted(opened) == sorted(closed),
               f"opened={opened} closed={closed}")

        print("\n--- tool result round-trip ---")
        r = await c.post("/v1/messages", json={
            "model": "claude-opus-4-5-20251101",
            "max_tokens": 300,
            "tools": TOOLS,
            "messages": [
                {"role": "user", "content": "How many files are here?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_abc123", "name": "bash",
                     "input": {"command": "ls -1 | wc -l"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_abc123", "content": "7"}
                ]},
            ],
        })
        text = ""
        if r.status_code == 200:
            text = " ".join(
                b.get("text", "") for b in r.json().get("content", []) if b["type"] == "text"
            )
        report("uses the tool result", "7" in text, f"reply={text[:160]!r}")

        print("\n--- count_tokens ---")
        r = await c.post("/v1/messages/count_tokens", json={
            "model": "claude-opus-4-5-20251101",
            "messages": [{"role": "user", "content": "hello world"}],
        })
        ok = r.status_code == 200 and isinstance(r.json().get("input_tokens"), int)
        report("returns input_tokens", ok, r.text[:120])

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        raise SystemExit(1)
    print("All Anthropic endpoint tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
