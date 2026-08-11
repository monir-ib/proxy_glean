import sys
from pathlib import Path

# Allow running this script directly from its subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#!/usr/bin/env python3
"""
Probe the Glean internal web API to discover its request/response contract.

Run this after get_credentials.py. It sends a minimal chat request using your
captured session and prints the raw response so the proxy can be built to match.
Secrets are never printed.

Usage:
    python probe_api.py                 # non-streaming probe
    python probe_api.py --stream        # streaming probe
    python probe_api.py --order         # test message ordering (chrono vs reversed)
"""
import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

BACKEND = os.getenv("GLEAN_BACKEND_URL", "").rstrip("/")
COOKIE = os.getenv("GLEAN_COOKIE", "")
EMAIL = os.getenv("GLEAN_EMAIL", "")
CLIENT_VERSION = os.getenv("GLEAN_CLIENT_VERSION", "")

STREAM = "--stream" in sys.argv
ORDER = "--order" in sys.argv

if not BACKEND or not COOKIE:
    sys.exit("Missing GLEAN_BACKEND_URL or GLEAN_COOKIE. Run get_credentials.py first.")


def headers() -> dict:
    h = {
        "Cookie": COOKIE,
        "Content-Type": "application/json",
        "Origin": "https://app.glean.com",
        "Referer": "https://app.glean.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
    }
    if EMAIL:
        h["X-Scio-Actas"] = EMAIL
    return h


def url(path: str) -> str:
    params = ["timezoneOffset=420", "locale=en"]
    if CLIENT_VERSION:
        params.append(f"clientVersion={CLIENT_VERSION}")
    return f"{BACKEND}{path}?{'&'.join(params)}"


def show(label: str, resp: httpx.Response, body: str):
    print(f"\n=== {label} ===")
    print(f"status: {resp.status_code}")
    ct = resp.headers.get("content-type", "?")
    print(f"content-type: {ct}")
    print(f"body ({len(body)} chars):")
    print(body[:4000])
    if len(body) > 4000:
        print(f"... [{len(body) - 4000} more chars]")


def probe_checkauth(client: httpx.Client):
    """Cheapest way to confirm the session cookie is accepted."""
    try:
        r = client.post(f"{BACKEND}/api/v1/checkauth", headers=headers(), json={})
        show("checkauth", r, r.text)
    except Exception as exc:
        print(f"checkauth failed: {type(exc).__name__}: {exc}")


def probe_chat(client: httpx.Client):
    payload = {
        "messages": [
            {
                "author": "USER",
                "messageType": "CONTENT",
                "fragments": [{"text": "Reply with exactly: PROXY_OK"}],
            }
        ],
        "stream": STREAM,
    }

    target = url("/api/v1/chat")
    print(f"POST {target.split('?')[0]}  (stream={STREAM})")

    if not STREAM:
        r = client.post(target, headers=headers(), json=payload)
        show("chat (non-streaming)", r, r.text)
        _try_parse(r.text)
        return

    with client.stream("POST", target, headers=headers(), json=payload) as r:
        print(f"\n=== chat (streaming) ===\nstatus: {r.status_code}")
        print(f"content-type: {r.headers.get('content-type', '?')}")
        print("--- lines ---")
        for i, line in enumerate(r.iter_lines()):
            if i >= 40:
                print("... [truncated]")
                break
            print(f"[{i}] {line[:300]}")


def _try_parse(text: str):
    """Show the top-level shape so the proxy can target the right fields."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("\n(body is not a single JSON object — likely newline-delimited JSON)")
        return
    print("\n--- parsed shape ---")
    print(f"top-level keys: {list(data)}")
    msgs = data.get("messages")
    if isinstance(msgs, list):
        print(f"messages: {len(msgs)}")
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                continue
            texts = [
                f.get("text", "")[:120]
                for f in (m.get("fragments") or [])
                if isinstance(f, dict) and f.get("text")
            ]
            print(
                f"  [{i}] author={m.get('author')} "
                f"type={m.get('messageType')} "
                f"stepId={m.get('stepId')} texts={texts}"
            )


def _msg(author: str, text: str) -> dict:
    return {
        "author": author,
        "messageType": "CONTENT",
        "fragments": [{"text": text}],
    }


def probe_order(client: httpx.Client):
    """Determine whether Glean wants chronological or reverse-chronological order.

    A multi-turn conversation is sent both ways; only the correctly ordered one
    lets Glean see "blue" as the answer to the final question.
    """
    chrono = [
        _msg("USER", "My favorite color is blue. Remember it."),
        _msg("GLEAN_AI", "Got it, your favorite color is blue."),
        _msg("USER", "What is my favorite color? Reply with just the color word."),
    ]

    for label, messages in (
        ("chronological (oldest first)", chrono),
        ("reversed (newest first)", list(reversed(chrono))),
    ):
        try:
            r = client.post(
                url("/api/v1/chat"),
                headers=headers(),
                json={"messages": messages, "stream": False},
            )
            text = ""
            if r.status_code == 200:
                for m in r.json().get("messages", []):
                    if m.get("author") == "GLEAN_AI":
                        for f in m.get("fragments") or []:
                            if isinstance(f, dict):
                                text += f.get("text", "")
            else:
                text = f"HTTP {r.status_code}: {r.text[:200]}"
            low = text.lower()
            # Merely containing "blue" is not enough — echoing the oldest
            # message ("I'll remember ... blue") also does. A correct answer
            # responds to the newest turn, so it states the color without
            # acknowledging the memory instruction.
            answered_question = "blue" in low and "remember" not in low
            verdict = "<-- answered the NEWEST turn" if answered_question else \
                      "(responded to the wrong turn)"
            print(f"\n[{label}]\n  reply: {text[:300]!r}\n  {verdict}")
        except Exception as exc:
            print(f"\n[{label}] failed: {type(exc).__name__}: {exc}")

    print(
        "\nThe ordering that answered the NEWEST turn is the one to use.\n"
        "Verified result: reversed (newest first) — GLEAN_REVERSE_MESSAGES=true."
    )


def main():
    print(f"backend: {BACKEND}")
    print(f"clientVersion: {CLIENT_VERSION or '(none)'}")
    with httpx.Client(timeout=120.0, follow_redirects=False) as client:
        if ORDER:
            probe_order(client)
            return
        probe_checkauth(client)
        probe_chat(client)


if __name__ == "__main__":
    main()
