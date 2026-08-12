"""Check whether the coding-agent-backend Personal Skill actually loads when
slash-invoked over the raw chat API, as opposed to the invocation token just
showing up as literal, unrecognized text in the reply.

Sends the same request under three conditions and compares behaviour:
  A. baseline    - no HARNESS_PROMPT, no skill invocation
  B. skill-only  - slash-invoke the Skill, HARNESS_PROMPT omitted
  C. harness     - HARNESS_PROMPT present, no skill invocation (today's default)

If the Skill is loading, B should behave like C (no "I don't have access to
your files" disclaimer) even though the actual instruction text isn't in the
payload. If the Skill isn't loading, B behaves like A and/or echoes the
"/coding-agent-backend" token back as confused literal text.

Run: python tools/probe_skill.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from dotenv import load_dotenv

import proxy

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REQUEST = "What files are in the current directory?"

CONDITIONS = {
    "A baseline   (no harness, no skill)": (False, False),
    "B skill-only (skill invoke, no harness prompt)": (False, True),
    "C harness    (harness prompt, no skill invoke)": (True, False),
}


def ask(client: httpx.Client, harness: bool, skill: bool) -> str:
    proxy.HARNESS_MODE = harness
    proxy.FORCE_SKILL_INVOKE = skill
    payload = proxy.build_payload(
        openai_messages=[{"role": "user", "content": REQUEST}],
        tools=[],
        stream=False,
        model=None,
    )
    r = client.post(proxy.glean_url(), json=payload, headers=proxy.glean_headers())
    if r.status_code != 200:
        return f"[HTTP {r.status_code}] {r.text[:200]}"
    return proxy.extract_text(r.json())


def main():
    orig_harness, orig_skill = proxy.HARNESS_MODE, proxy.FORCE_SKILL_INVOKE
    try:
        with httpx.Client(timeout=300) as client:
            for label, (harness, skill) in CONDITIONS.items():
                print(f"\n=== {label} ===")
                reply = ask(client, harness, skill)
                print(reply[:600])
    finally:
        proxy.HARNESS_MODE, proxy.FORCE_SKILL_INVOKE = orig_harness, orig_skill


if __name__ == "__main__":
    sys.exit(main())
