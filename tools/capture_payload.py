import sys
from pathlib import Path

# Allow running this script directly from its subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#!/usr/bin/env python3
"""
Capture the full chat request payload that the Glean web app sends.

Use this to discover the fields behind UI settings — model choice, thinking /
reasoning mode, and restricting or disabling company sources. Configure the
setting in the Glean UI, send a prompt, and the exact JSON body is saved here.

Send as many prompts as you like with different settings; each is saved as a
numbered file in captured_payloads/. Close the browser window when done.

Usage:
    python capture_payload.py
    python capture_payload.py --diff    # after 2+ captures, show what changed
"""
import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from playwright.async_api import async_playwright
except ImportError:
    sys.exit("Run: pip install playwright")

HERE = Path(__file__).resolve().parents[1]
PROFILE_DIR = HERE / ".glean_profile"
OUT_DIR = HERE / "captured_payloads"
GLEAN_CHAT_URL = "https://app.glean.com/chat"

# Field names worth surfacing: these are where model, reasoning mode, and
# source-restriction settings are likely to live.
INTERESTING = (
    "model", "agent", "source", "restrict", "think", "reason", "mode",
    "tool", "datasource", "app", "index", "web", "config", "capabilit",
    "filter", "scope", "search", "retriev",
)


def scan(obj, path=""):
    """Yield (path, value) for keys that look like settings."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else key
            if any(word in key.lower() for word in INTERESTING):
                if isinstance(value, (str, int, float, bool)) or value is None:
                    yield here, value
                else:
                    yield here, f"<{type(value).__name__}>"
            yield from scan(value, here)
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:6]):
            yield from scan(item, f"{path}[{i}]")


def summarize(body: dict, query: dict):
    print(f"  top-level keys: {sorted(body)}")
    if query:
        print(f"  query params  : { {k: v[0] for k, v in query.items()} }")

    prompt = ""
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and str(msg.get("author", "")).upper() == "USER":
            for frag in msg.get("fragments") or []:
                if isinstance(frag, dict) and frag.get("text"):
                    prompt = frag["text"]
    if prompt:
        print(f"  prompt        : {prompt[:100]!r}")

    settings = list(scan(body))
    if settings:
        print("  settings-ish fields:")
        seen = set()
        for path, value in settings:
            if path in seen:
                continue
            seen.add(path)
            print(f"    {path} = {json.dumps(value)[:120]}")
    else:
        print("  (no settings-like fields found)")


def show_diff():
    files = sorted(OUT_DIR.glob("*.json"))
    if len(files) < 2:
        sys.exit("Need at least 2 captures in captured_payloads/ to diff.")

    def flat(obj, path=""):
        out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.update(flat(v, f"{path}.{k}" if path else k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                out.update(flat(v, f"{path}[{i}]"))
        else:
            out[path] = obj
        return out

    print(f"Comparing {len(files)} captures\n")
    flats = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        flats.append((f.name, flat(data.get("body", data))))

    all_keys = sorted({k for _, fl in flats for k in fl})
    for key in all_keys:
        values = [fl.get(key, "<absent>") for _, fl in flats]
        # Skip noise: message text and fields identical everywhere.
        if len(set(map(str, values))) == 1:
            continue
        if "fragments" in key or key.endswith(".text"):
            continue
        print(f"{key}")
        for (name, _), value in zip(flats, values):
            print(f"    {name}: {json.dumps(value)[:110]}")
        print()


async def main():
    OUT_DIR.mkdir(exist_ok=True)
    existing = len(list(OUT_DIR.glob("*.json")))
    counter = {"n": existing}
    closed = asyncio.Event()

    async with async_playwright() as p:
        context = None
        for channel in ("msedge", "chrome", None):
            try:
                kwargs = {
                    "user_data_dir": str(PROFILE_DIR),
                    "headless": False,
                    "args": ["--start-maximized"],
                    "no_viewport": True,
                }
                if channel:
                    kwargs["channel"] = channel
                context = await p.chromium.launch_persistent_context(**kwargs)
                break
            except Exception:
                continue
        if context is None:
            sys.exit("Could not launch a browser.")

        def on_request(request):
            if request.method != "POST" or "/api/v1/chat" not in request.url:
                return
            try:
                raw = request.post_data
            except Exception:
                return
            if not raw:
                return
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                return

            # Ignore the automatic page-load calls that carry no user text.
            has_text = any(
                str(m.get("author", "")).upper() == "USER"
                and any(f.get("text", "").strip() for f in (m.get("fragments") or []))
                for m in (body.get("messages") or [])
                if isinstance(m, dict)
            )
            if not has_text:
                return

            counter["n"] += 1
            n = counter["n"]
            query = parse_qs(urlparse(request.url).query)
            path = OUT_DIR / f"{n:03d}.json"
            path.write_text(
                json.dumps(
                    {"url": request.url, "query": {k: v[0] for k, v in query.items()},
                     "body": body},
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\n=== capture {n:03d} -> {path.name} ===")
            summarize(body, query)
            print("  (send another prompt with different settings, or close the window)")

        context.on("request", on_request)
        context.on("close", lambda *_: closed.set())
        page = context.pages[0] if context.pages else await context.new_page()
        page.on("close", lambda *_: closed.set())

        print("=" * 68)
        print("  Glean payload capture")
        print("=" * 68)
        print("  For each configuration you want captured:")
        print("    1. Change the setting in the Glean UI, for example:")
        print("       - turn company sources off / restrict to no sources")
        print("       - pick a specific model")
        print("       - enable a thinking / reasoning mode")
        print("    2. Send a prompt.")
        print("  Repeat for each combination, then close the browser window.")
        print("=" * 68)

        try:
            await page.goto(GLEAN_CHAT_URL, wait_until="domcontentloaded", timeout=0)
        except Exception as exc:
            print(f"(navigation notice: {type(exc).__name__})")

        await closed.wait()

    total = counter["n"] - existing
    print(f"\nCaptured {total} payload(s) in {OUT_DIR}")
    if total >= 2 or counter["n"] >= 2:
        print("Compare them with:  python capture_payload.py --diff")


if __name__ == "__main__":
    if "--diff" in sys.argv:
        show_diff()
    else:
        asyncio.run(main())
