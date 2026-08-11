"""Investigate whether Glean supports native client-side tool calling.

The web app sends a `clientTools: []` field. If it accepts real schemas and
returns structured tool invocations, the proxy can stop emulating tool calls
through prompt injection.

Run: python probe_clienttools.py
"""
import sys
from pathlib import Path

# Allow running this script directly from its subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import httpx

import proxy

QUESTION = "List the files in the current directory. Use the bash tool."

# Candidate schema shapes, since the field's format is undocumented here.
SHAPES = {
    "openai-style": [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command on the user's machine.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }],
    "flat-parameters": [{
        "name": "bash",
        "description": "Run a shell command on the user's machine.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }],
    "mcp-inputSchema": [{
        "name": "bash",
        "description": "Run a shell command on the user's machine.",
        "inputSchema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }],
    "displayName-variant": [{
        "name": "bash",
        "displayName": "Bash",
        "description": "Run a shell command on the user's machine.",
        "parameters": [
            {"name": "command", "type": "STRING", "isRequired": True,
             "description": "The command to run"}
        ],
    }],
}

TOOL_HINT_KEYS = (
    "toolcall", "tool_call", "toolinvocation", "action", "functioncall",
    "toolname", "tool", "clienttool",
)


def find_tool_hints(obj, path=""):
    """Surface anything in the response that looks like a structured tool call."""
    hits = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else key
            if key.lower().replace("_", "") in [k.replace("_", "") for k in TOOL_HINT_KEYS]:
                hits.append((here, json.dumps(value)[:200]))
            hits.extend(find_tool_hints(value, here))
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:8]):
            hits.extend(find_tool_hints(item, f"{path}[{i}]"))
    return hits


def probe_listtools(client: httpx.Client):
    """Glean's own tool list shows the naming convention it expects."""
    print("=== /api/v1/listtools ===")
    try:
        r = client.post(
            f"{proxy.GLEAN_BACKEND_URL}/api/v1/listtools",
            headers=proxy.glean_headers(),
            json={},
        )
        print(f"status: {r.status_code}")
        if r.status_code != 200:
            print(r.text[:300])
            return
        data = r.json()
        print(f"top-level keys: {list(data)}")
        tools = data.get("tools") or data.get("toolDefinitions") or []
        print(f"tool count: {len(tools)}")
        if tools:
            print("first tool schema:")
            print(json.dumps(tools[0], indent=2)[:1200])
            print(f"\nfield names used: {sorted(tools[0])}")
    except Exception as exc:
        print(f"failed: {type(exc).__name__}: {exc}")


def probe_shape(client: httpx.Client, label: str, tools: list):
    print(f"\n=== clientTools shape: {label} ===")
    payload = proxy.build_payload(
        [{"role": "user", "content": QUESTION}], [], False, None
    )
    payload["clientTools"] = tools

    try:
        r = client.post(proxy.glean_url(), headers=proxy.glean_headers(), json=payload)
    except Exception as exc:
        print(f"  request failed: {type(exc).__name__}: {exc}")
        return

    print(f"  status: {r.status_code}")
    if r.status_code != 200:
        # A validation error is informative: it usually names the expected field.
        print(f"  body: {r.text[:400]}")
        return

    data = r.json()
    text = proxy.extract_text(data)
    print(f"  text reply: {text[:160]!r}")

    hints = find_tool_hints(data)
    if hints:
        print("  structured tool-call fields found:")
        for path, value in hints[:8]:
            print(f"    {path} = {value}")
    else:
        print("  no structured tool-call fields in response")

    kinds = {
        (m.get("author"), m.get("messageType"))
        for m in data.get("messages", []) if isinstance(m, dict)
    }
    print(f"  message kinds: {sorted(str(k) for k in kinds)}")


def main():
    with httpx.Client(timeout=300) as client:
        probe_listtools(client)
        for label, tools in SHAPES.items():
            probe_shape(client, label, tools)

    print(
        "\nInterpretation:\n"
        "  * A 200 plus structured tool-call fields => native tool calling works.\n"
        "  * A 4xx naming a field => that field name is the correct schema key.\n"
        "  * 200 but only prose => clientTools is ignored; keep prompt injection."
    )


if __name__ == "__main__":
    main()
