"""Measure Glean's effective context window empirically.

Glean does not publish an input limit, and `modelSetId: OPUS_5_MS` says nothing
about how much of a prompt Glean actually forwards. This plants a needle at the
very start of a large prompt and asks for it back: if the needle survives, the
beginning of the prompt was still in context at that size.

Silent truncation is the danger — Glean answering from a fragment looks like a
correct answer, not an error — so this distinguishes three outcomes per size:
kept the needle, lost the needle (truncated), or refused the request.

Usage:
    python tools/probe_context.py              # default ladder
    python tools/probe_context.py 500000       # test one size in characters
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uuid

import httpx

import proxy

# Roughly 4 characters per token.
LADDER = [20_000, 100_000, 400_000, 800_000, 1_600_000, 3_200_000]

FILLER = (
    "The quick brown fox jumps over the lazy dog while the engineer reviews "
    "logs, refactors a module, and documents the deployment procedure. "
)


def build_prompt(size: int, needle: str) -> str:
    """Needle first, filler after: this tests retention of the prompt's start."""
    head = (
        f"MEMO-ID: {needle}\n"
        "Remember the MEMO-ID above; you will be asked to repeat it.\n\n"
        "=== BEGIN LOG ===\n"
    )
    tail = (
        "\n=== END LOG ===\n\n"
        "Ignore the log content entirely. Reply with ONLY the MEMO-ID stated at "
        "the very top of this message, nothing else. If you cannot see it, reply "
        "exactly: MISSING"
    )
    body_len = max(0, size - len(head) - len(tail))
    reps = body_len // len(FILLER) + 1
    body = "".join(f"[{i}] {FILLER}" for i in range(reps))[:body_len]
    return head + body + tail


def probe(client: httpx.Client, size: int) -> str:
    needle = uuid.uuid4().hex[:12].upper()
    prompt = build_prompt(size, needle)
    approx_tokens = len(prompt) // 4

    payload = proxy.build_payload(
        [{"role": "user", "content": prompt}], [], False, None
    )

    label = f"{len(prompt):>9,} chars (~{approx_tokens:>7,} tokens)"
    try:
        r = client.post(proxy.glean_url(), headers=proxy.glean_headers(), json=payload)
    except Exception as exc:
        print(f"  {label}  ERROR      {type(exc).__name__}: {str(exc)[:70]}")
        return "error"

    if r.status_code != 200:
        detail = r.text[:100].replace("\n", " ")
        print(f"  {label}  HTTP {r.status_code}   {detail}")
        return "rejected"

    reply = proxy.extract_text(r.json()).strip()
    if needle in reply:
        print(f"  {label}  KEPT       needle returned")
        return "kept"
    if not reply:
        print(f"  {label}  EMPTY      no reply text")
        return "empty"
    print(f"  {label}  LOST       reply: {reply[:70]!r}")
    return "lost"


def main():
    sizes = [int(sys.argv[1])] if len(sys.argv) > 1 else LADDER

    print(f"backend: {proxy.GLEAN_BACKEND_URL}")
    print(f"modelSetId: {proxy.GLEAN_MODEL_SET_ID}  agent: {proxy.GLEAN_AGENT}")
    print("\nplanting a MEMO-ID at the start of a growing prompt:\n")

    results = {}
    largest_kept = 0
    with httpx.Client(timeout=600) as client:
        for size in sizes:
            outcome = probe(client, size)
            results[size] = outcome
            if outcome == "kept":
                largest_kept = max(largest_kept, size)
            elif outcome in ("rejected", "error"):
                print("  (stopping: the request itself failed)")
                break

    print("\n" + "=" * 64)
    if largest_kept:
        tokens = largest_kept // 4
        print(f"Largest prompt whose start survived: ~{tokens:,} tokens")
        print("\nA safe setting is somewhat below the largest KEPT size:")
        print(f"  GLEAN_CONTEXT_WINDOW={int(tokens * 0.8) // 1000 * 1000}")
    else:
        print("No size retained the needle; try a smaller starting size.")

    if "lost" in results.values():
        print(
            "\nAt least one size returned a reply while dropping the start of the\n"
            "prompt: Glean truncates silently there, so keep the configured window\n"
            "below that point or answers will be based on partial context."
        )
    print("=" * 64)


if __name__ == "__main__":
    main()
