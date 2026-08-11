"""Dump full streaming lines from Glean to verify nothing is filtered out.

Shows each chunk's author/messageType and its fragment text untruncated, then
compares the proxy's extraction against a naive "take every fragment" pass so
any dropped content is obvious.

Run: python dump_stream.py
"""
import sys
from pathlib import Path

# Allow running this script directly from its subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os

import httpx
from dotenv import load_dotenv

import proxy

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROMPT = os.getenv("DUMP_PROMPT", "List exactly five fruits, one per line, no commentary.")

payload = {
    "messages": [
        {"author": "USER", "messageType": "CONTENT", "fragments": [{"text": PROMPT}]}
    ],
    "stream": True,
}

proxy_text = []
naive_text = []
kinds = {}

with httpx.Client(timeout=300) as client:
    with client.stream(
        "POST", proxy.glean_url(), json=payload, headers=proxy.glean_headers()
    ) as resp:
        print(f"status: {resp.status_code}  content-type: {resp.headers.get('content-type')}")
        print("-" * 70)
        for line in resp.iter_lines():
            for obj in proxy._iter_json_objects(line):
                for msg in obj.get("messages", []) or []:
                    author = msg.get("author")
                    mtype = msg.get("messageType")
                    frags = [
                        f.get("text", "")
                        for f in (msg.get("fragments") or [])
                        if isinstance(f, dict)
                    ]
                    texts = [t for t in frags if t]
                    kinds[(author, mtype)] = kinds.get((author, mtype), 0) + 1
                    if texts:
                        print(f"author={author} messageType={mtype} text={texts!r}")
                        naive_text.extend(texts)
                    else:
                        print(f"author={author} messageType={mtype} (no text)")
                extracted = proxy.extract_text(obj)
                if extracted:
                    proxy_text.append(extracted)

print("-" * 70)
print("message kinds seen (author, messageType) -> count:")
for k, v in kinds.items():
    print(f"  {k}: {v}")

naive = "".join(naive_text)
mine = "".join(proxy_text)
print(f"\nnaive  (every fragment) : {len(naive)} chars\n{naive!r}")
print(f"\nproxy  (filtered)       : {len(mine)} chars\n{mine!r}")

if naive == mine:
    print("\nOK: proxy extraction keeps all fragment text.")
else:
    lost = len(naive) - len(mine)
    print(f"\nMISMATCH: proxy extraction differs by {lost} chars — filter is dropping text.")
