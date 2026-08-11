"""Discover available Glean model sets and tool toggles.

Looks through /api/v1/config for modelSetId values and any switches governing
Glean's built-in tools, so the proxy can expose real model names and turn off
the server-side shell sandbox if that is configurable.

Run: python discover_models.py
"""
import sys
from pathlib import Path

# Allow running this script directly from its subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import re

import httpx

import proxy

PATTERN = re.compile(
    r"model|toolset|codeinterp|shell|sandbox|interpreter|agent|thinking|reason",
    re.I,
)


def walk(obj, path=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else key
            if PATTERN.search(key):
                if isinstance(value, (str, int, float, bool)) or value is None:
                    yield here, value
                else:
                    yield here, json.dumps(value)[:400]
            yield from walk(value, here)
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:15]):
            yield from walk(item, f"{path}[{i}]")


def collect_model_ids(obj) -> set:
    """Gather strings that look like modelSetId values, e.g. OPUS_5_MS."""
    found = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and re.fullmatch(r"[A-Z0-9]+(?:_[A-Z0-9]+)+", value):
                if re.search(r"model", key, re.I) or re.search(
                    r"OPUS|SONNET|HAIKU|GPT|GEMINI|LLAMA|MS$", value
                ):
                    found.add(value)
            found |= collect_model_ids(value)
    elif isinstance(obj, list):
        for item in obj:
            found |= collect_model_ids(item)
    return found


def probe(path: str, client: httpx.Client):
    print(f"\n=== {path} ===")
    try:
        r = client.post(f"{proxy.GLEAN_BACKEND_URL}{path}", headers=proxy.glean_headers(), json={})
    except Exception as exc:
        print(f"  failed: {type(exc).__name__}: {exc}")
        return None
    print(f"  status: {r.status_code}  bytes: {len(r.text)}")
    if r.status_code != 200:
        print(f"  {r.text[:200]}")
        return None
    try:
        data = r.json()
    except json.JSONDecodeError:
        print("  (not JSON)")
        return None

    seen = set()
    rows = 0
    for key, value in walk(data):
        if key in seen:
            continue
        seen.add(key)
        print(f"  {key} = {value}")
        rows += 1
        if rows >= 45:
            print("  ... (truncated)")
            break
    if not rows:
        print(f"  no matching keys; top-level: {list(data)[:20]}")
    return data


def main():
    all_ids = set()
    with httpx.Client(timeout=180) as client:
        for path in ("/api/v1/config", "/api/v1/publicclientconfig"):
            data = probe(path, client)
            if data:
                all_ids |= collect_model_ids(data)

    print("\n" + "=" * 60)
    if all_ids:
        print("Model-set-like identifiers found:")
        for mid in sorted(all_ids):
            print(f"  {mid}")
        print("\nTry one with:  GLEAN_MODEL_SET_ID=<id> in .env")
        print('or per request: {"model": "<id>"}')
    else:
        print(
            "No model identifiers found in config.\n"
            "Switch model in the Glean UI and re-run capture_payload.py to read\n"
            "the modelSetId directly from the request."
        )


if __name__ == "__main__":
    main()
