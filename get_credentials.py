#!/usr/bin/env python3
"""
Automatically capture Glean backend URL and session cookie.

Opens Glean in a browser window and watches network traffic to learn your
tenant's backend host, then reads the session cookies straight out of the
browser cookie jar and writes them to .env.

Send one chat message (best — it reveals the exact backend host), or just
close the window once you are logged in.

The browser profile is kept in .glean_profile/ so you only log in once.

Usage:
    python get_credentials.py            # normal run
    python get_credentials.py --debug    # log every request observed
    python get_credentials.py --fresh    # wipe saved login profile first
"""
import asyncio
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

try:
    from playwright.async_api import async_playwright
except ImportError:
    sys.exit(
        "Playwright not installed.\n"
        "Run: pip install playwright && playwright install chromium"
    )

HERE = Path(__file__).parent
ENV_FILE = HERE / ".env"
PROFILE_DIR = HERE / ".glean_profile"
GLEAN_CHAT_URL = "https://app.glean.com/chat"
TIMEOUT_SECONDS = 900  # 15 minutes — plenty of time for SSO

DEBUG = "--debug" in sys.argv
FRESH = "--fresh" in sys.argv


class Watcher:
    """Observes network traffic to identify the Glean backend host.

    Cookies are deliberately NOT read from request headers: Playwright's
    synchronous `request.headers` omits the Cookie header, so cookies are
    pulled from the browser cookie jar after capture instead.
    """

    def __init__(self):
        self.hosts: set[str] = set()
        self.chat_request = None           # a genuine user-initiated chat POST
        self.chat_host: str | None = None
        self.found = asyncio.Event()

    def on_request(self, request):
        host = urlparse(request.url).hostname or ""
        if not host.endswith("glean.com"):
            return

        self.hosts.add(host)
        if DEBUG:
            print(f"  [req] {request.method} {request.url[:110]}")

        if self.found.is_set():
            return
        if request.method != "POST" or "/chat" not in request.url:
            return
        if not self._has_user_message(request):
            return

        self.chat_request = request
        self.chat_host = host
        self.found.set()
        print(f"\n>> Detected chat message to {host}")

    @staticmethod
    def _has_user_message(request) -> bool:
        """True if the POST body carries a USER message with real text.

        Glean fires automatic /chat POSTs on page load (history restore and
        similar) that contain no user-authored text; those must not count.
        """
        try:
            raw = request.post_data
        except Exception:
            return False
        if not raw:
            return False
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if not isinstance(body, dict):
            return False

        messages = body.get("messages")
        if not isinstance(messages, list):
            return False

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("author", "")).upper() != "USER":
                continue
            for frag in msg.get("fragments") or []:
                if isinstance(frag, dict) and frag.get("text", "").strip():
                    return True
        return False

    def backend_host(self) -> str | None:
        """Best guess at the backend host, most reliable source first."""
        if self.chat_host:
            return self.chat_host
        backend = [h for h in self.hosts if "-be." in h]
        if backend:
            return sorted(backend)[0]
        others = [h for h in self.hosts if h != "app.glean.com"]
        if others:
            return sorted(others)[0]
        return None


def _cookie_header(cookies: list[dict], host: str) -> str:
    """Build a Cookie header for `host` from a Playwright cookie jar."""
    # Later entries win, preferring the most specific (longest) domain match.
    chosen: dict[str, tuple[int, str]] = {}
    for c in cookies:
        domain = str(c.get("domain", "")).lstrip(".")
        if not domain:
            continue
        if host != domain and not host.endswith("." + domain):
            continue
        name, value = c.get("name"), c.get("value")
        if not name:
            continue
        specificity = len(domain)
        prev = chosen.get(name)
        if prev is None or specificity >= prev[0]:
            chosen[name] = (specificity, value or "")
    return "; ".join(f"{n}={v}" for n, (_, v) in chosen.items())


async def _extract_email(watcher: Watcher, page) -> str:
    """Find the acting user's email from request headers, else from the page."""
    if watcher.chat_request is not None:
        try:
            headers = await watcher.chat_request.all_headers()
            email = headers.get("x-scio-actas", "").strip()
            if email:
                return email
        except Exception:
            pass

    # Fall back to scraping any email-looking string the app exposes.
    try:
        found = await page.evaluate(
            """() => {
                const re = /[\\w.+-]+@[\\w-]+\\.[\\w.-]+/;
                for (const store of [localStorage, sessionStorage]) {
                    for (let i = 0; i < store.length; i++) {
                        const m = re.exec(store.getItem(store.key(i)) || '');
                        if (m) return m[0];
                    }
                }
                return '';
            }"""
        )
        if found:
            return found.strip()
    except Exception:
        pass
    return ""


async def capture_credentials() -> dict | None:
    watcher = Watcher()
    closed = asyncio.Event()

    if FRESH and PROFILE_DIR.exists():
        shutil.rmtree(PROFILE_DIR, ignore_errors=True)
        print("Wiped saved browser profile.")

    async with async_playwright() as p:
        context = await _launch_context(p)

        # Listen at the context level so popups and SSO tabs are covered too.
        context.on("request", watcher.on_request)
        context.on("close", lambda *_: closed.set())

        page = context.pages[0] if context.pages else await context.new_page()
        page.on("close", lambda *_: closed.set())

        _print_instructions()

        # A slow SSO redirect chain must not abort the run, so navigation
        # errors are non-fatal — you can always navigate manually.
        try:
            await page.goto(GLEAN_CHAT_URL, wait_until="domcontentloaded", timeout=0)
        except Exception as exc:
            print(f"(navigation notice: {type(exc).__name__} — continuing anyway)")

        await _wait_for_capture(watcher, closed)

        # Read cookies and email while the browser is still alive.
        cookies = []
        try:
            cookies = await context.cookies()
        except Exception as exc:
            print(f"Could not read cookie jar: {exc}")

        email = ""
        if not closed.is_set():
            try:
                email = await _extract_email(watcher, page)
            except Exception:
                pass

        host = watcher.backend_host()

        if DEBUG:
            print(f"\n  hosts seen: {sorted(watcher.hosts) or '(none)'}")
            print(f"  cookies in jar: {len(cookies)}")
            names = sorted({c.get('name', '') for c in cookies})
            print(f"  cookie names: {names}")

        if not closed.is_set() and watcher.found.is_set():
            print("Closing browser in 3 seconds...")
            try:
                await page.wait_for_timeout(3000)
            except Exception:
                pass

        try:
            await context.close()
        except Exception:
            pass

    return _assemble(watcher, cookies, host, email)


def _assemble(watcher: Watcher, cookies: list, host: str | None, email: str) -> dict | None:
    if not host:
        print("\nNo Glean host was observed — it looks like Glean never loaded.")
        _diagnose(watcher, cookies)
        return None

    cookie_header = _cookie_header(cookies, host)
    if not cookie_header:
        print(f"\nNo cookies found for {host} — you are probably not logged in yet.")
        _diagnose(watcher, cookies)
        return None

    return {"backend_url": f"https://{host}", "cookie": cookie_header, "email": email}


def _diagnose(watcher: Watcher, cookies: list):
    print("\nWhat was observed:")
    print(f"  Glean hosts contacted : {sorted(watcher.hosts) or '(none)'}")
    print(f"  Cookies in jar        : {len(cookies)}")
    print("\nThings to try:")
    print("  * Re-run and make sure you finish the SSO login, then send a chat message.")
    print("  * Re-run with --debug to see every request observed.")
    print("  * Re-run with --fresh if the saved profile is in a bad state.")


async def _wait_for_capture(watcher: Watcher, closed: asyncio.Event):
    """Wait until a chat message is captured, the window closes, or timeout."""
    waiters = [
        asyncio.create_task(watcher.found.wait()),
        asyncio.create_task(closed.wait()),
    ]
    try:
        done, _ = await asyncio.wait(
            waiters, timeout=TIMEOUT_SECONDS, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            print(f"\nTimed out after {TIMEOUT_SECONDS // 60} minutes.")
    finally:
        for task in waiters:
            task.cancel()


async def _launch_context(p):
    """Launch a persistent context so the login survives between runs."""
    PROFILE_DIR.mkdir(exist_ok=True)
    last_error = None
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
            print(f"Launched {channel or 'bundled Chromium'}.")
            return context
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Could not launch a browser ({last_error}).\nTry: playwright install chromium"
    )


def _print_instructions():
    print()
    print("=" * 64)
    print("  Glean Credential Capture")
    print("=" * 64)
    print("  1. Log in to Glean if prompted (SSO is fine — take your time).")
    print("  2. Send any chat message, e.g. \"hello\".")
    print("  3. The browser closes automatically once captured.")
    print()
    print("  Sending a message is preferred: it reveals the exact backend")
    print("  host. Closing the window also works once you are logged in.")
    print("=" * 64)
    print()


def write_env(data: dict):
    existing: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, _, v = stripped.partition("=")
                existing[k.strip()] = v.strip()

    existing["GLEAN_BACKEND_URL"] = data["backend_url"]
    existing["GLEAN_COOKIE"] = data["cookie"]
    if data.get("email"):
        existing["GLEAN_EMAIL"] = data["email"]
    existing.setdefault("GLEAN_EMAIL", "")

    content = "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
    ENV_FILE.write_text(content, encoding="utf-8")


async def async_main():
    data = await capture_credentials()
    if not data:
        sys.exit(1)

    write_env(data)

    cookie_count = data["cookie"].count("=")
    print(f"\nSaved to {ENV_FILE}")
    print(f"  GLEAN_BACKEND_URL = {data['backend_url']}")
    print(f"  GLEAN_EMAIL       = {data.get('email') or '(not found — set manually in .env)'}")
    print(f"  GLEAN_COOKIE      = {cookie_count} cookies, {len(data['cookie'])} chars")

    if "-be." not in data["backend_url"]:
        print(
            f"\nNote: {data['backend_url']} does not look like a backend host"
            " (*-be.glean.com).\nIf the proxy errors, re-run and send a chat"
            " message so the real backend host is captured."
        )

    print("\nStart the proxy:  uvicorn proxy:app --host 127.0.0.1 --port 8000")


if __name__ == "__main__":
    asyncio.run(async_main())
