"""Compare tool-call reliability across both API dialects.

Sends the same tool-enabled request through /v1/chat/completions (OpenAI) and
/v1/messages (Anthropic) several times. Similar rates mean any misses come from
Glean's nondeterminism; a gap means the Anthropic conversion is at fault.

Run: python tools/compare_dialects.py [trials]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio

import httpx

from proxy import app

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

PROMPTS = [
    "How many files are in this directory?",
    "List the files here.",
    "Find every TODO comment in this repository.",
]

OPENAI_TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command on the user's machine and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]

ANTHROPIC_TOOLS = [{
    "name": "bash",
    "description": "Run a shell command on the user's machine and return its output.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


async def try_openai(client, prompt) -> tuple[bool, str]:
    r = await client.post("/v1/chat/completions", json={
        "model": "glean",
        "messages": [{"role": "user", "content": prompt}],
        "tools": OPENAI_TOOLS,
    })
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    choice = r.json()["choices"][0]
    calls = (choice["message"] or {}).get("tool_calls") or []
    if calls:
        return True, calls[0]["function"]["arguments"][:60]
    return False, (choice["message"].get("content") or "")[:70]


async def try_anthropic(client, prompt) -> tuple[bool, str]:
    r = await client.post("/v1/messages", json={
        "model": "claude-opus-4-5-20251101",
        "max_tokens": 300,
        "tools": ANTHROPIC_TOOLS,
        "messages": [{"role": "user", "content": prompt}],
    })
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    blocks = r.json().get("content") or []
    for block in blocks:
        if block.get("type") == "tool_use":
            return True, str(block.get("input"))[:60]
    text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return False, text[:70]


async def main():
    transport = httpx.ASGITransport(app=app)
    tally = {"openai": 0, "anthropic": 0}
    total = 0

    async with httpx.AsyncClient(transport=transport, base_url="http://p", timeout=300) as c:
        for prompt in PROMPTS:
            print(f"\n=== {prompt!r} ===")
            for trial in range(TRIALS):
                total += 1
                for name, fn in (("openai", try_openai), ("anthropic", try_anthropic)):
                    called, detail = await fn(c, prompt)
                    tally[name] += int(called)
                    tag = "TOOL" if called else "text"
                    print(f"  [{trial + 1}] {name:10} {tag}  {detail!r}")

    print("\n" + "=" * 60)
    print(f"tool-call rate over {total} requests per dialect:")
    for name, hits in tally.items():
        print(f"  {name:10} {hits}/{total}")
    gap = abs(tally["openai"] - tally["anthropic"])
    if gap <= max(1, total // 4):
        print("\nRates are comparable => misses are Glean nondeterminism,")
        print("not a bug in the Anthropic conversion.")
    else:
        print("\nSignificant gap => the Anthropic conversion likely has a bug.")


if __name__ == "__main__":
    asyncio.run(main())
