#!/usr/bin/env python3
"""
Diagnose the Glean proxy and say exactly what to do about each problem.

Checks configuration, session validity, whether Glean's frontend version has
moved on, and whether a real chat request still works. Every failure prints the
command that fixes it.

Usage:
    python doctor.py              # full check
    python doctor.py --offline    # config checks only, no network
"""
import base64
import json
import re
import sys
import time
from pathlib import Path

import httpx

import proxy

OFFLINE = "--offline" in sys.argv

OK, WARN, BAD = "OK", "WARN", "FAIL"
results = []


def record(status: str, title: str, detail: str = "", fix: str = ""):
    results.append((status, title, detail, fix))
    mark = {OK: "  OK  ", WARN: " WARN ", BAD: " FAIL "}[status]
    print(f"[{mark}] {title}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")
    if fix and status != OK:
        for line in fix.splitlines():
            print(f"         -> {line}")


CAPTURE = "python get_credentials.py"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def check_env():
    if not proxy.ENV_FILE.exists():
        record(BAD, ".env present", f"{proxy.ENV_FILE} not found",
               f"Create it by running: {CAPTURE}")
        return False
    record(OK, ".env present", str(proxy.ENV_FILE))

    required = {
        "GLEAN_BACKEND_URL": proxy.GLEAN_BACKEND_URL,
        "GLEAN_COOKIE": proxy.GLEAN_COOKIE,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        record(BAD, "required settings", f"missing: {', '.join(missing)}",
               f"Re-capture them: {CAPTURE}")
        return False
    record(OK, "required settings", "backend URL and cookie are set")

    if not proxy.GLEAN_EMAIL:
        record(WARN, "GLEAN_EMAIL", "not set; the X-Scio-Actas header is omitted",
               "Set GLEAN_EMAIL=you@company.com in .env")
    else:
        record(OK, "GLEAN_EMAIL", proxy.GLEAN_EMAIL)

    if "-be." not in proxy.GLEAN_BACKEND_URL:
        record(WARN, "backend host", f"{proxy.GLEAN_BACKEND_URL} is not a *-be host",
               f"Re-run {CAPTURE} and send a chat message so the real\n"
               "backend host is captured")
    else:
        record(OK, "backend host", proxy.GLEAN_BACKEND_URL)
    return True


def session_timestamp() -> int | None:
    """Read the creation time encoded at the start of the session cookie.

    glean-session-store begins with base64 of "<unix seconds>|...", so the
    session's age can be reported without contacting Glean.
    """
    match = re.search(r"glean-session-store=([^;]+)", proxy.GLEAN_COOKIE)
    if not match:
        return None
    value = match.group(1)
    for length in (20, 16, 12):
        chunk = value[:length]
        try:
            decoded = base64.b64decode(chunk + "=" * (-len(chunk) % 4))
        except Exception:
            continue
        digits = re.match(rb"^(\d{10})", decoded)
        if digits:
            return int(digits.group(1))
    return None


def check_cookie():
    names = sorted(
        c.split("=", 1)[0].strip()
        for c in proxy.GLEAN_COOKIE.split(";")
        if "=" in c
    )
    if "glean-session-store" not in names:
        record(BAD, "session cookie", f"glean-session-store missing; have: {names}",
               f"Re-capture: {CAPTURE}")
        return
    record(OK, "session cookie", f"{len(names)} cookies: {', '.join(names)}")

    started = session_timestamp()
    if started is None:
        record(WARN, "session age", "could not decode the session timestamp")
        return
    age_days = (time.time() - started) / 86400
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(started))
    detail = f"issued {when} ({age_days:.1f} days ago)"
    if age_days > 6:
        record(WARN, "session age", detail,
               f"Sessions usually last about a week. Refresh soon:\n{CAPTURE}")
    else:
        record(OK, "session age", detail)


# --------------------------------------------------------------------------
# Live checks
# --------------------------------------------------------------------------

def check_auth(client: httpx.Client):
    try:
        r = client.post(
            f"{proxy.GLEAN_BACKEND_URL}/api/v1/checkauth",
            headers=proxy.glean_headers(),
            json={},
        )
    except Exception as exc:
        record(BAD, "session valid", f"{type(exc).__name__}: {exc}",
               "Check your network or VPN, then retry")
        return False

    if r.status_code in (401, 403):
        record(BAD, "session valid", f"HTTP {r.status_code} - cookie rejected",
               f"Your session expired. Refresh it:\n{CAPTURE}")
        return False
    if r.status_code != 200:
        record(BAD, "session valid", f"HTTP {r.status_code}: {r.text[:160]}",
               f"If this persists, re-capture: {CAPTURE}")
        return False

    try:
        data = r.json()
    except json.JSONDecodeError:
        record(WARN, "session valid", "checkauth returned non-JSON")
        return True

    if not data.get("isValid"):
        record(BAD, "session valid", "checkauth reports isValid=false",
               f"Refresh your session:\n{CAPTURE}")
        return False

    email = (data.get("user") or {}).get("metadata", {}).get("email", "")
    record(OK, "session valid", f"authenticated as {email or 'unknown'}")

    if email and proxy.GLEAN_EMAIL and email.lower() != proxy.GLEAN_EMAIL.lower():
        record(WARN, "email matches session",
               f"GLEAN_EMAIL={proxy.GLEAN_EMAIL} but session is {email}",
               f"Set GLEAN_EMAIL={email} in .env")
    return True


def check_client_version(client: httpx.Client):
    """Compare the configured frontend build with the one Glean serves now."""
    configured = proxy.GLEAN_CLIENT_VERSION
    try:
        r = client.get(
            "https://app.glean.com/chat",
            headers={"Cookie": proxy.GLEAN_COOKIE, "User-Agent": "Mozilla/5.0"},
            follow_redirects=True,
        )
        found = set(re.findall(r"fe-release-\d{4}-\d{2}-\d{2}-[0-9a-f]+", r.text))
    except Exception as exc:
        record(WARN, "client version", f"could not check: {type(exc).__name__}")
        return

    if not found:
        record(WARN, "client version",
               f"configured {configured or '(none)'}; could not read the live version",
               "If requests start failing, refresh it with:\n"
               "python tools/capture_payload.py")
        return

    if configured in found:
        record(OK, "client version", f"{configured} matches the live frontend")
        return

    live = sorted(found)[-1]
    record(WARN, "client version",
           f"configured {configured or '(none)'}\nlive       {live}",
           f"Glean shipped a frontend update. Update .env:\n"
           f"GLEAN_CLIENT_VERSION={live}\n"
           f"(or re-run {CAPTURE} to refresh everything)")


def check_chat(client: httpx.Client):
    payload = proxy.build_payload(
        [{"role": "user", "content": "Reply with exactly: DOCTOR_OK"}], [], False, None
    )
    try:
        r = client.post(proxy.glean_url(), headers=proxy.glean_headers(), json=payload)
    except Exception as exc:
        record(BAD, "chat request", f"{type(exc).__name__}: {exc}")
        return

    if r.status_code in (401, 403):
        record(BAD, "chat request", f"HTTP {r.status_code}",
               f"Session expired. Refresh:\n{CAPTURE}")
        return
    if r.status_code >= 400:
        record(BAD, "chat request", f"HTTP {r.status_code}: {r.text[:200]}",
               "A 500 here often means agentConfig was rejected. Try:\n"
               "  - GLEAN_MODEL_SET_ID=OPUS_5_MS (a known-good value)\n"
               "  - GLEAN_AGENT=ADVANCED\n"
               "  - re-capture the payload: python tools/capture_payload.py")
        return

    text = proxy.extract_text(r.json())
    if not text.strip():
        record(WARN, "chat request", "200 but the reply was empty",
               "Retry; if it persists the model set may be unavailable.\n"
               "Try GLEAN_MODEL_SET_ID=OPUS_5_MS")
        return
    record(OK, "chat request", f"replied {text[:60]!r}")


def check_model(client: httpx.Client):
    record(
        OK if proxy.GLEAN_MODEL_SET_ID else WARN,
        "model config",
        f"agent={proxy.GLEAN_AGENT} modelSetId={proxy.GLEAN_MODEL_SET_ID}",
        "Discover valid IDs with: python tools/discover_models.py",
    )
    if proxy.ENABLE_COMPANY_TOOLS:
        record(WARN, "company tools",
               "enabled - Glean may answer about its own indexed repos instead\n"
               "of calling tools against your machine",
               "For coding use set GLEAN_ENABLE_COMPANY_TOOLS=false in .env")
    else:
        record(OK, "company tools", "disabled (correct for coding use)")


def check_proxy_running():
    for port in (8000,):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                record(OK, "proxy server", f"running on port {port}: {r.json()}")
                return
        except Exception:
            pass
    record(WARN, "proxy server", "not responding on port 8000",
           "Start it with:\npython -m uvicorn proxy:app --host 127.0.0.1 --port 8000")


def main():
    print("=" * 68)
    print("  Glean proxy diagnostics")
    print("=" * 68)

    print("\n-- configuration --")
    if not check_env():
        _summary()
        return 1
    check_cookie()
    check_model(None)

    if not OFFLINE:
        print("\n-- live checks --")
        with httpx.Client(timeout=120) as client:
            if check_auth(client):
                check_client_version(client)
                check_chat(client)
        print("\n-- local server --")
        check_proxy_running()
    else:
        print("\n(--offline: skipped live checks)")

    return _summary()


def _summary():
    bad = [r for r in results if r[0] == BAD]
    warn = [r for r in results if r[0] == WARN]

    print("\n" + "=" * 68)
    if bad:
        print(f"  {len(bad)} problem(s) need fixing:")
        for _, title, _, fix in bad:
            first = fix.splitlines()[0] if fix else "see above"
            print(f"    - {title}: {first}")
    if warn:
        print(f"  {len(warn)} warning(s):")
        for _, title, _, fix in warn:
            first = fix.splitlines()[0] if fix else "see above"
            print(f"    - {title}: {first}")
    if not bad and not warn:
        print("  Everything looks healthy.")
    print("=" * 68)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
