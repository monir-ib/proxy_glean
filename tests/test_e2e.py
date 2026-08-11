"""End-to-end test of the proxy against the live Glean API.

Calls the FastAPI app in-process (no separate server needed) exactly the way a
coding harness would. Requires a valid .env from get_credentials.py.

Run: python test_e2e.py
"""
import sys
from pathlib import Path

# Allow running this script directly from its subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import json

import httpx

from proxy import app

BASE = "http://proxy"
failures = []


def report(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        print(f"      {detail}")
    if not ok:
        failures.append(label)


async def main():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE, timeout=300) as c:

        print("--- health / models ---")
        r = await c.get("/health")
        cfg = r.json()
        report("health ok", r.status_code == 200 and cfg.get("configured"), str(cfg))
        r = await c.get("/v1/models")
        report("models lists glean", r.json()["data"][0]["id"] == "glean")

        print("\n--- non-streaming completion ---")
        r = await c.post("/v1/chat/completions", json={
            "model": "glean",
            "messages": [{"role": "user", "content": "Reply with exactly: ALPHA"}],
        })
        ok = r.status_code == 200
        content = ""
        if ok:
            content = r.json()["choices"][0]["message"]["content"] or ""
        report("returns 200", ok, "" if ok else r.text[:300])
        report("content present", bool(content.strip()), f"content={content[:120]!r}")

        print("\n--- multi-turn context (verifies newest-first ordering) ---")
        r = await c.post("/v1/chat/completions", json={
            "model": "glean",
            "messages": [
                {"role": "user", "content": "My project codename is Falcon. Remember it."},
                {"role": "assistant", "content": "Noted, the codename is Falcon."},
                {"role": "user", "content": "What is my project codename? Reply with one word."},
            ],
        })
        text = ""
        if r.status_code == 200:
            text = (r.json()["choices"][0]["message"]["content"] or "")
        # A correct answer names the codename without re-acknowledging the
        # memory instruction, which is what replying to the oldest turn does.
        good = "falcon" in text.lower() and "remember" not in text.lower()
        report("answers the newest turn", good, f"reply={text[:160]!r}")

        print("\n--- content blocks (Claude Code message shape) ---")
        r = await c.post("/v1/chat/completions", json={
            "model": "glean",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Reply with exactly: BLOCKS_OK"}
            ]}],
        })
        report("block content accepted", r.status_code == 200,
               "" if r.status_code == 200 else r.text[:200])

        print("\n--- streaming ---")
        chunks, contents = [], []
        async with c.stream("POST", "/v1/chat/completions", json={
            "model": "glean",
            "messages": [{"role": "user", "content": "Count: one two three four five"}],
            "stream": True,
        }) as resp:
            report("stream status 200", resp.status_code == 200)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    chunks.append("[DONE]")
                    break
                obj = json.loads(body)
                chunks.append(obj)
                piece = obj["choices"][0]["delta"].get("content")
                if piece:
                    contents.append(piece)
        report("got multiple chunks", len(chunks) > 2, f"{len(chunks)} chunks")
        report("terminated with [DONE]", chunks and chunks[-1] == "[DONE]")
        streamed = "".join(contents)
        report("streamed text non-empty", bool(streamed.strip()), f"text={streamed[:160]!r}")
        finishes = [
            ch["choices"][0]["finish_reason"] for ch in chunks
            if isinstance(ch, dict) and ch["choices"][0]["finish_reason"]
        ]
        report("finish_reason sent once", finishes == ["stop"], f"finishes={finishes}")

        print("\n--- tool calling ---")
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        }]
        r = await c.post("/v1/chat/completions", json={
            "model": "glean",
            "messages": [{"role": "user", "content": "What is the weather in Tokyo right now?"}],
            "tools": tools,
        })
        ok = r.status_code == 200
        msg = r.json()["choices"][0] if ok else {}
        calls = (msg.get("message") or {}).get("tool_calls") or []
        report("tool request ok", ok, "" if ok else r.text[:300])
        if calls:
            fn = calls[0]["function"]
            args_ok = isinstance(fn["arguments"], str)
            parsed = {}
            try:
                parsed = json.loads(fn["arguments"])
            except json.JSONDecodeError:
                pass
            report("emitted a tool_call", fn["name"] == "get_weather", f"name={fn['name']}")
            report("arguments are a JSON string", args_ok, f"args={fn['arguments'][:120]}")
            report("finish_reason=tool_calls", msg.get("finish_reason") == "tool_calls")
            report("city argument present", "city" in parsed, f"parsed={parsed}")
        else:
            content = (msg.get("message") or {}).get("content") or ""
            report("emitted a tool_call", False,
                   f"no tool_calls; model replied with text instead: {content[:200]!r}")

        print("\n--- tool result round-trip ---")
        r = await c.post("/v1/chat/completions", json={
            "model": "glean",
            "messages": [
                {"role": "user", "content": "What is the weather in Tokyo?"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                }]},
                {"role": "tool", "tool_call_id": "call_1", "name": "get_weather",
                 "content": "18C and raining"},
            ],
            "tools": tools,
        })
        text = ""
        if r.status_code == 200:
            text = (r.json()["choices"][0]["message"]["content"] or "")
        report("uses the tool result", "18" in text or "rain" in text.lower(),
               f"reply={text[:200]!r}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        raise SystemExit(1)
    print("All end-to-end tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
