# Glean Proxy for Kilo Code

Use Glean as the model behind **Kilo Code** in VS Code, with working tool use — Glean can
run `grep`, read files, and execute commands on your machine through Kilo Code.

It works by reusing your normal Glean browser login, so no admin console or API token is
needed. A local server translates between Kilo Code's OpenAI-style requests and Glean's
internal web API.

---

## Setup

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

If you have neither Edge nor Chrome installed, also run:

```powershell
python -m playwright install chromium
```

> `python -m` is used throughout because pip's `Scripts` folder is often missing from
> PATH on Windows, which makes bare `uvicorn` or `playwright` fail with
> "not recognized as the name of a cmdlet".

### 2. Capture your Glean session

```powershell
python get_credentials.py
```

A browser window opens on Glean. **Log in** (SSO is fine, take your time), then **send
one chat message** such as "hello". The script captures your session and closes.

Your login is saved in `.glean_profile/`, so later refreshes usually skip the login step.

### 3. Verify it works

```powershell
python doctor.py
```

Every line should read `OK`. If anything fails, the doctor prints the exact command that
fixes it. See [When something breaks](#when-something-breaks).

### 4. Start the proxy

```powershell
python -m uvicorn proxy:app --host 127.0.0.1 --port 8000
```

**Leave this terminal open** — the server must keep running while you use Kilo Code. It
logs every request, which is the quickest way to confirm Kilo Code is reaching it.

### 5. Install Kilo Code

In VS Code: Extensions (`Ctrl+Shift+X`) → search **Kilo Code** → Install. Open it from
the Kilo icon in the sidebar.

### 6. Point Kilo Code at the proxy

Open **Settings** (gear icon) → **Providers** tab, scroll to the bottom and click
**Custom provider**. Fill in:

| Field | Value |
|---|---|
| Provider ID | `glean-proxy` |
| Display name | `Glean Proxy` |
| Provider API | `OpenAI Compatible` |
| Base URL | `http://localhost:8000/v1` |
| API key | `dummy` (any non-empty string; may also be left empty) |
| Models | `glean` |

Kilo Code auto-fetches the model list from the proxy's `/v1/models`, so `glean` should
appear on its own once the Base URL is entered — the proxy must be running for that.
Submit, then pick **Glean Proxy → glean** in the model picker.

Then set the context window in `kilo.json` — Kilo Code does **not** take it from the
proxy. See [Setting the context window](#setting-the-context-window), which also covers
turning tool use on. Without `tool_call: true`, Kilo Code will not let Glean edit files
or run commands.

This setup is text-only, so leave image and browser/computer-use options off.

### 7. Open your project and run a task

**File → Open Folder** and pick the project you want to work on. Kilo Code needs a
workspace folder and will not run without one.

Give it a real task:

```
list the files in this project
```

Kilo Code proposes a command and waits for approval. Click **Approve** and **let the
command finish**. The proxy terminal should log something like:

```
-> glean: 1 msgs, 18 tools, stream=True, model=OPUS_5_MS
```

That means Glean is answering. The tool count is non-zero because Kilo Code sends real
OpenAI tool schemas — see [How tool use works](#how-tool-use-works).

---

## Using it day to day

- The proxy must be running. If Kilo Code reports `ECONNREFUSED 127.0.0.1:8000`, start it.
- Restart the proxy after editing `.env` — settings are read once at startup. This
  matters most after refreshing your session.
- Restarting mid-conversation is otherwise harmless; the proxy keeps no state.

Two habits make a real difference:

**Let commands finish.** Approving a command but not waiting for it returns no output.
Starved of the data it asked for, Glean invents an explanation — commonly that it is
"sandboxed" and cannot reach your files. Always let the command complete.

**Give tasks, not greetings.** Opening with "hi" invites a conversational reply with no
tool call in it, which is wasted turn. Start with something actionable.

---

## When something breaks

Start here — it names the problem and the fix:

```powershell
python doctor.py              # full check
python doctor.py --offline    # config only, no network
```

It checks your `.env`, the cookies, your **session age** (decoded from the cookie
itself), whether Glean still accepts the session, whether **Glean's frontend version has
moved on**, whether a real chat request succeeds, and whether the proxy is running.

| Symptom | Cause | Fix |
|---|---|---|
| `ECONNREFUSED 127.0.0.1:8000` in Kilo Code | Proxy not running | Start it (step 4) |
| `401`, or `[Glean session expired]` in a reply | Session expired (roughly weekly) | `python get_credentials.py`, then restart the proxy |
| `500` from Glean on every request | `agentConfig` rejected, usually a bad `GLEAN_MODEL_SET_ID` | Set `GLEAN_MODEL_SET_ID=OPUS_5_MS`, restart |
| Replies mention being "sandboxed" or in a container | Glean's own shell tool confusing it, usually after empty command output | Let commands finish; keep `GLEAN_HARNESS_MODE=true` |
| Answers describe files you do not have | Company retrieval, or Glean's internal sandbox | Keep `GLEAN_ENABLE_COMPANY_TOOLS=false` |
| Greeted by name, or Glean features mentioned | Personalization leaking in | Keep `GLEAN_INCOGNITO_MODE=true` |
| Replies are empty | Model set unavailable | Try `GLEAN_MODEL_SET_ID=OPUS_5_MS` |
| Doctor warns the client version differs | Glean shipped a frontend update | Copy the version doctor prints into `GLEAN_CLIENT_VERSION` |
| `glean` missing from Kilo Code's model picker | Proxy was down when Kilo fetched models, or the model was never added | Start the proxy, then **Edit provider** and add `glean` |
| Kilo Code will not edit files or run commands | `tool_call` not set on the model | Add `"tool_call": true` in `kilo.json` |
| Conversations grow until Glean returns `500` | `limit.context` unset, so compaction is disabled | Set `limit.context` in `kilo.json` |

---

## How tool use works

Kilo Code uses **native OpenAI function calling**: it sends its tool schemas in the
`tools` parameter and expects `tool_calls` back. Glean's API has no usable `tools`
parameter, so the proxy injects the schemas into the prompt and parses `<tool_call>`
blocks out of the reply into a proper `tool_calls` response. This is the same path used
by Continue.dev's agent mode and Cursor.

The proxy also supports clients that define their own XML tool syntax in the system
prompt and parse replies themselves. Those send **no** `tools` parameter, so nothing
needs translating — Glean just has to follow the client's format, which
`tools/test_cline_protocol.py` checks (3/3 on real tasks). An Anthropic-dialect endpoint
(`/v1/messages`) is also served for tools that expect that shape.

### Keeping Glean in character

Glean is an assistant with its own persona, not a bare model, so two settings keep it
behaving like a backend. Both default to on.

- **`GLEAN_HARNESS_MODE`** injects an instruction to follow the client's format, and
  states that Glean has no shell or filesystem of its own and that command output comes
  from your real machine. Without it, Glean answers as the Glean Assistant or claims it
  is sandboxed.
- **`GLEAN_INCOGNITO_MODE`** sets Glean's `incognitoMode` request field, suppressing the
  personalization that otherwise appears as greetings by name or references to your
  documents.

### Company retrieval must stay off

**`GLEAN_ENABLE_COMPANY_TOOLS=false` is load-bearing, not a preference.** With retrieval
on, "Find every TODO comment in this repository" returned a confident answer citing
`skills/create-skill/scripts/init_skill.py:118` — a file in a *different* indexed repo.
An agent would treat that as fact. With it off, the same request correctly produces
`grep -rn -E 'TODO' .`.

Glean can fabricate local answers two ways: its **enterprise index** (fixed by that
setting) and its **own server-side Linux sandbox** (`/home/user`, `.duckdb`,
`tool_sdk.py`), which the harness prompt steers it away from. If a reply mentions files
you do not recognise, suspect these before believing it.

---

## Configuration (`.env`)

Written by `get_credentials.py`; edit by hand for the rest. Restart the proxy after any
change.

| Variable | Purpose |
|---|---|
| `GLEAN_BACKEND_URL` | Tenant backend, e.g. `https://infoblox-be.glean.com` |
| `GLEAN_COOKIE` | Session cookies (captured automatically) |
| `GLEAN_EMAIL` | Sent as `X-Scio-Actas` |
| `GLEAN_CLIENT_VERSION` | Glean frontend build string |
| `GLEAN_REVERSE_MESSAGES` | Newest-first ordering; leave `true` |
| `GLEAN_AGENT` | `ADVANCED` (reasoning) or `FAST` |
| `GLEAN_MODEL_SET_ID` | Model, e.g. `OPUS_5_MS` |
| `GLEAN_ENABLE_COMPANY_TOOLS` | Enterprise retrieval; keep `false` |
| `GLEAN_ENABLE_WEB_SEARCH` | Web search, default `false` |
| `GLEAN_SAVE_CHAT` | Save to your Glean chat history, default `false` |
| `GLEAN_INCOGNITO_MODE` | Suppress personalization, default `true` |
| `GLEAN_HARNESS_MODE` | Keep Glean acting as a backend model, default `true` |
| `GLEAN_CONTEXT_WINDOW` | Context limit advertised on `/v1/models`, default `400000` |
| `GLEAN_MAX_OUTPUT_TOKENS` | Output limit advertised, default `8192` |
| `GLEAN_TIMEZONE_OFFSET` | Minutes, default `420` |
| `GLEAN_TIMEOUT` | Upstream timeout in seconds, default `300` |
| `LOG_LEVEL` | e.g. `DEBUG` |

### Setting the context window

Kilo Code **ignores** the limits the proxy advertises on `/v1/models`. It resolves them
from its own config first, then from its bundled [models.dev](https://models.dev)
snapshot, and `glean` is not in that catalog — so if you set nothing, `context` and
`output` both resolve to `0`. That has real consequences: **compaction is disabled**, so
conversations grow unbounded until Glean rejects the request, output silently falls back
to Kilo's internal 32,000-token default, and context usage tracking stops working.

So set them explicitly in `kilo.json`, using the provider ID you chose in step 6:

```jsonc
{
  "$schema": "https://app.kilo.ai/config.json",
  "model": "glean-proxy/glean",
  "provider": {
    "glean-proxy": {
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "dummy"
      },
      "models": {
        "glean": {
          "name": "Glean",
          "tool_call": true,
          "limit": {
            "context": 400000,
            "output": 8192
          }
        }
      }
    }
  }
}
```

- **Global config** (recommended, and the only place `{env:VAR}` references resolve):
  `C:\Users\<you>\.config\kilo\kilo.json`.
- **Per project**: `kilo.json` in the workspace root.

`limit.context` is what Kilo Code compacts against; `limit.output` is sent upstream as
`max_tokens`. There is also `limit.input`, for the case where a provider's input ceiling
is lower than its full window — set it and compaction triggers against that instead.

Keep the two sides roughly in step: `GLEAN_CONTEXT_WINDOW` in `.env` is what the proxy
reports to any client that asks, and `limit.context` is what Kilo Code actually enforces.
Raising one without the other just means Kilo Code and the proxy disagree. Restart the
proxy after editing `.env`; reload the VS Code window after editing `kilo.json`.

### About the measured limit

Measured with `tools/probe_context.py`, which plants a marker at the start of a growing
prompt and asks for it back:

| prompt size | result |
|---|---|
| 200k tokens | marker kept |
| 400k tokens | marker kept |
| 600k tokens | marker kept |
| 800k tokens | HTTP 500 |

Glean never truncated silently — it either preserved the whole prompt or failed outright,
which is the safer behaviour. The default of `400000` leaves headroom below the measured
ceiling. Lower it if long sessions start failing.

Selecting a model per request works only with Glean-style IDs (`UPPER_SNAKE`, e.g.
`OPUS_5_MS`). Client model names like `gpt-4o` are ignored on purpose: forwarding them
made Glean reject the value and discard the whole `agentConfig`, silently re-enabling
company retrieval.

---

## Layout

```
proxy.py              the server
get_credentials.py    capture/refresh your Glean session
doctor.py             diagnose problems and print the fix
tests/                automated tests (translation, live, Anthropic dialect)
tools/                probes for Glean's API behaviour
.env                  your captured session (gitignored)
```

## Diagnostics

```powershell
python doctor.py                        # start here
python tests/test_translation.py        # offline unit tests, no network
python tests/test_e2e.py                # live: streaming, tools, multi-turn
python tools/test_cline_protocol.py     # does Glean obey a prompt-defined tool format?
python tools/probe_context.py           # measure the context window
python tools/probe_api.py               # raw request/response shape
python tools/dump_stream.py             # confirm no stream content is dropped
python tools/tune_tools.py              # score tool-call prompt wordings
python tools/capture_payload.py         # capture payloads behind Glean UI settings
python tools/inspect_bundle.py          # search Glean's frontend for API details
python tools/discover_models.py         # hunt for available model IDs
curl http://localhost:8000/health
```

`capture_payload.py` is the one to reach for when Glean's UI gains a setting you want:
turn it on in the browser, send a prompt, and it prints the JSON fields that changed.

## Verified API behaviour

Established against the live API, not assumed:

| Detail | Value |
|---|---|
| Endpoint | `POST /api/v1/chat` (the web app's API, **not** `/rest/api/v1/chat`) |
| Query params | `timezoneOffset`, `locale`, `clientVersion` |
| Auth | `glean-session-store` + `okta-saml-hosted-login-session-store` cookies (no `act` cookie) |
| Message order | **Newest first** — Glean answers `messages[0]` |
| Roles | `USER` / `GLEAN_AI`; no system role, so system text folds into the newest user turn |
| Streaming | newline-delimited JSON (`text/plain`), fragments are **incremental** |
| Stream noise | `messageType` `UPDATE`/`CONTROL` carry no text; empty `{}` fragments occur |
| `agentConfig` | `agent`, `modelSetId`, `toolSets.enableCompanyTools`, `toolSets.enableWebSearch` |
| Other request fields | `incognitoMode`, `actionHints`, `agentId`, `inclusions.datasourceInstances` |
| Effective context | at least 600k tokens; 800k returns HTTP 500 |
| Built-in tools | `/api/v1/listtools` lists 18 server-side tools, including `Glean Search` and a `Shell` sandbox |

Ordering matters more than it looks: sent chronologically, Glean replies to the *oldest*
message, which reads as the model ignoring you rather than as a proxy bug.

## Running on another machine (Linux VM, container, server)

The session cookie is a bearer credential and is **not tied to your device**:

```bash
# On your desktop, where a GUI browser exists:
python get_credentials.py

# Copy .env to the VM, then there:
pip install -r requirements.txt
python doctor.py
python -m uvicorn proxy:app --host 127.0.0.1 --port 8000
```

- Playwright is not needed on the VM; only `get_credentials.py` uses a browser.
- The VM needs network access to your Glean backend host (VPN if applicable).
- `.env` is a live credential: copy it over SSH/SCP, and `chmod 600 .env`.
- One session works from several machines at once. Refreshing means re-copying `.env`.
- To reach it from another machine, prefer an SSH tunnel
  (`ssh -L 8000:localhost:8000 user@vm`). Binding `--host 0.0.0.0` exposes an endpoint
  with **no authentication** — anyone who can reach the port can use your Glean account.

## Notes

- `.env` holds live session cookies; anyone with them can act as you in Glean.
- This uses an internal endpoint with no stability guarantees, and web sessions may be
  rate-limited more aggressively than the official API.
- Requests count as your Glean usage and are subject to your organisation's policies.
