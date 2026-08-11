"""Search Glean's frontend bundles for the real tool-calling contract.

The web client's JavaScript contains the field names and enum values the API
uses, so this looks for clientTools schemas, tool-call message types, and
anything indicating client-side tool execution — rather than guessing shapes.

Bundles are cached in .bundle_cache/.

Usage:
    python tools/inspect_bundle.py                 # search default keywords
    python tools/inspect_bundle.py clientTool MCP  # search specific terms
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import re

import httpx

import proxy

CACHE = Path(__file__).resolve().parents[1] / ".bundle_cache"
APP = "https://app.glean.com"

DEFAULT_TERMS = [
    "clientTools",
    "CLIENT_TOOL",
    "toolCall",
    "TOOL_CALL",
    "functionCall",
    "inputSchema",
    "toolInvocation",
    "CLIENT_SIDE",
    "messageType",
]

CONTEXT = 260
MAX_HITS_PER_TERM = 6


async def fetch_bundles() -> list[Path]:
    """Download the root bundles, then the webpack chunks they reference.

    Only a handful of scripts appear in the HTML; the chat code lives in lazily
    loaded chunks whose names and hashes come from a map inside the runtime.
    """
    CACHE.mkdir(exist_ok=True)
    headers = {"Cookie": proxy.GLEAN_COOKIE, "User-Agent": "Mozilla/5.0"}
    paths: list[Path] = []

    async with httpx.AsyncClient(timeout=120, headers=headers, follow_redirects=True) as client:
        sem = asyncio.Semaphore(8)

        async def grab(src: str) -> Path | None:
            name = src.rsplit("/", 1)[-1]
            dest = CACHE / name
            if dest.exists() and dest.stat().st_size > 0:
                return dest
            async with sem:
                try:
                    r = await client.get(f"{APP}{src}")
                    if r.status_code == 200 and r.content:
                        dest.write_bytes(r.content)
                        return dest
                except Exception:
                    return None
            return None

        page = await client.get(f"{APP}/chat")
        roots = set(re.findall(r'"(/static/[^"]+\.js)"', page.text))
        print(f"page {page.status_code}: {len(roots)} root scripts")
        if not roots:
            print("No scripts found — is the session valid? Run: python doctor.py")
            return []

        for result in await asyncio.gather(*(grab(s) for s in sorted(roots))):
            if result:
                paths.append(result)

        # Webpack embeds chunk name -> content hash maps; rebuild the URLs.
        chunks: set[str] = set()
        for path in list(paths):
            text = path.read_text(encoding="utf-8", errors="replace")
            # Keys may be quoted names ("chat-routes") or bare numeric chunk ids.
            for name, hash_ in re.findall(
                r'"?([A-Za-z0-9_$-]{1,60})"?\s*:\s*"([0-9a-f]{16})"', text
            ):
                chunks.add(f"/static/{name}-{hash_}.js")
            for direct in re.findall(r'"(/static/[A-Za-z0-9_-]+-[0-9a-f]{16}\.js)"', text):
                chunks.add(direct)

        chunks -= roots
        print(f"discovered {len(chunks)} lazy chunks; downloading...")
        results = await asyncio.gather(*(grab(s) for s in sorted(chunks)))
        got = [r for r in results if r]
        paths.extend(got)
        print(f"fetched {len(got)}/{len(chunks)} chunks")

    total_mb = sum(p.stat().st_size for p in paths) / 1e6
    print(f"cached {len(paths)} bundles ({total_mb:.1f} MB) in {CACHE.name}/")
    return paths


def search(paths: list[Path], terms: list[str]):
    for term in terms:
        print("\n" + "=" * 72)
        print(f"  {term}")
        print("=" * 72)
        hits = 0
        seen: set[str] = set()

        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for match in re.finditer(re.escape(term), text):
                start = max(0, match.start() - CONTEXT // 2)
                snippet = text[start:match.start() + CONTEXT // 2]
                snippet = re.sub(r"\s+", " ", snippet).strip()
                key = snippet[:110]
                if key in seen:
                    continue
                seen.add(key)
                print(f"\n  [{path.name}]")
                print(f"    ...{snippet}...")
                hits += 1
                if hits >= MAX_HITS_PER_TERM:
                    break
            if hits >= MAX_HITS_PER_TERM:
                break
        if not hits:
            print("  (no matches)")


def enum_scan(paths: list[Path]):
    """Collect SCREAMING_CASE values near tool-related words: likely enums."""
    print("\n" + "=" * 72)
    print("  tool-related enum candidates")
    print("=" * 72)
    found: dict[str, int] = {}
    pattern = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+){1,4})\b")
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for match in pattern.finditer(text):
            value = match.group(1)
            if any(word in value for word in ("TOOL", "CLIENT", "FUNCTION", "ACTION", "SHELL")):
                found[value] = found.get(value, 0) + 1
    for value, count in sorted(found.items(), key=lambda kv: -kv[1])[:40]:
        print(f"  {value:44} x{count}")
    if not found:
        print("  (none)")


async def main():
    terms = [a for a in sys.argv[1:] if not a.startswith("-")] or DEFAULT_TERMS
    paths = await fetch_bundles()
    if not paths:
        return
    search(paths, terms)
    enum_scan(paths)
    print(
        "\nLooking for: a clientTools entry shape, a messageType used for tool\n"
        "calls, and any field carrying tool results back to Glean."
    )


if __name__ == "__main__":
    asyncio.run(main())
