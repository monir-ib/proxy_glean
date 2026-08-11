"""Check whether Glean obeys a Cline-style prompt protocol.

Cline defines its own XML tool syntax in the system prompt and sends no `tools`
parameter, so success depends entirely on Glean following instructions instead
of replying as the Glean Assistant. This measures that, with the proxy's harness
prompt on and off, so the difference is visible.

Run: python tools/test_cline_protocol.py [trials]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import re

import httpx

import proxy

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 2

# Condensed but structurally faithful to Cline's real system prompt.
CLINE_SYSTEM = """\
You are Cline, a highly skilled software engineer.

====

TOOL USE

You have access to a set of tools. You use one tool per message, and receive the
result in the user's response. You must use exactly one tool in every message.

# Tool Use Formatting

Tool use is formatted using XML-style tags. The tool name is enclosed in opening
and closing tags, and each parameter likewise:

<tool_name>
<parameter1_name>value1</parameter1_name>
</tool_name>

# Tools

## execute_command
Description: Request to execute a CLI command on the system.
Parameters:
- command: (required) The CLI command to execute.
Usage:
<execute_command>
<command>Your command here</command>
</execute_command>

## read_file
Description: Request to read the contents of a file.
Parameters:
- path: (required) The path of the file to read.
Usage:
<read_file>
<path>File path here</path>
</read_file>

## ask_followup_question
Description: Ask the user a question to gather additional information.
Parameters:
- question: (required) The question to ask.
Usage:
<ask_followup_question>
<question>Your question here</question>
</ask_followup_question>

====

RULES

- You must respond with exactly one tool use per message.
- NEVER start your messages with a greeting like "Great", "Certainly", or "Sure".
- You are STRICTLY FORBIDDEN from starting a message with a conversational filler.
- Do not ask unnecessary questions; use the tools to accomplish the task.

====

SYSTEM INFORMATION

Operating System: Windows 11
Default Shell: PowerShell
Current Working Directory: C:/Users/mfathalla/Desktop/proxy_glean
"""

TASKS = [
    "<task>List the files in this project.</task>",
    "<task>Show me what is in requirements.txt</task>",
    "<task>How many Python files are in this repo?</task>",
]

VALID_TOOLS = ("execute_command", "read_file", "ask_followup_question")

# Signs Glean answered as itself rather than as the tool's backend model.
PERSONA_LEAKS = (
    "hi monir", "hello monir", "hi ", "glean", "document reader",
    "happy to help", "what are you working on", "👋",
)


def score(reply: str) -> tuple[bool, bool, str]:
    """Return (used_a_tool, persona_leaked, note)."""
    used = None
    for tool in VALID_TOOLS:
        if re.search(rf"<{tool}>.*?</{tool}>", reply, re.DOTALL):
            used = tool
            break
    low = reply.lower()
    leaked = any(sign in low for sign in PERSONA_LEAKS)
    note = used if used else reply.strip()[:80].replace("\n", " ")
    return bool(used), leaked, note


async def run(client, harness: bool) -> tuple[int, int, int]:
    proxy.HARNESS_MODE = harness
    tools_used = leaks = total = 0

    for task in TASKS:
        for _ in range(TRIALS):
            total += 1
            payload = proxy.build_payload(
                [
                    {"role": "system", "content": CLINE_SYSTEM},
                    {"role": "user", "content": task},
                ],
                [],
                False,
                None,
            )
            try:
                r = await client.post(
                    proxy.glean_url(), headers=proxy.glean_headers(), json=payload
                )
                reply = proxy.extract_text(r.json()) if r.status_code == 200 else \
                    f"[HTTP {r.status_code}]"
            except Exception as exc:
                reply = f"[{type(exc).__name__}]"

            used, leaked, note = score(reply)
            tools_used += used
            leaks += leaked
            flag = "TOOL" if used else "MISS"
            extra = " (persona leak)" if leaked else ""
            print(f"    {flag} {task[6:40]:36} -> {note}{extra}")

    return tools_used, leaks, total


async def main():
    async with httpx.AsyncClient(timeout=300) as client:
        print("=== harness prompt OFF (raw Glean) ===")
        off_used, off_leaks, total = await run(client, False)

        print("\n=== harness prompt ON (proxy default) ===")
        on_used, on_leaks, _ = await run(client, True)

    print("\n" + "=" * 62)
    print(f"{'':22} {'tool used':>12} {'persona leaks':>15}")
    print(f"{'harness OFF':22} {f'{off_used}/{total}':>12} {f'{off_leaks}/{total}':>15}")
    print(f"{'harness ON':22} {f'{on_used}/{total}':>12} {f'{on_leaks}/{total}':>15}")

    if on_used > off_used or on_leaks < off_leaks:
        print("\nThe harness prompt helps; keep GLEAN_HARNESS_MODE=true.")
    elif on_used == total:
        print("\nBoth pass; Glean follows Cline's protocol either way.")
    else:
        print(
            "\nGlean does not reliably follow Cline's protocol.\n"
            "Cline may not be workable with Glean; prefer a client that uses\n"
            "native tool calling (Continue.dev, Cursor) via /v1/chat/completions."
        )


if __name__ == "__main__":
    asyncio.run(main())
