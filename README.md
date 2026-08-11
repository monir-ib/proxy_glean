# Glean Proxy

Turns your Glean web session into a local API that coding tools can use as a model —
with **tool calling**, so Glean can run `grep`, read files, and execute commands on your
machine through the harness.

Speaks two dialects, so most tools work without a translation layer:

| Endpoint | Dialect | Used by |
|---|---|---|
| `/v1/chat/completions` | OpenAI | Cursor, Continue.dev, Aider, most tools |
| `/v1/messages` | Anthropic | Claude Code |

No admin console or API token needed — it reuses your normal browser login.

## Quick start

```bash
pip install -r requirements.txt
python -m playwright install chromium   # only if you have neither Edge nor Chrome

python get_credentials.py               # opens a browser: log in, send one chat message
python doctor.py                        # verify everything is healthy
python -m uvicorn proxy:app --host 127.0.0.1 --port 8000
```

`python -m` is used throughout because pip's `Scripts` directory is often missing from
PATH on Windows, which makes bare `uvicorn` / `playwright` fail with "not recognized".

The server keeps running in that terminal, so open a second one for your editor.

## Connecting your tools

### Claude Code

Start the proxy in one terminal, then in a **second** terminal:

PowerShell (Windows):

```powershell
$env:ANTHROPIC_BASE_URL = "http://localhost:8000"
$env:ANTHROPIC_API_KEY  = "dummy"
claude
```

bash / zsh (macOS, Linux):

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=dummy claude
```

PowerShell has no inline `VAR=value cmd` form — using the bash version there leaves
you at a `>>` continuation prompt (press Ctrl+C to escape).

These variables last only for that shell session. To make them permanent on Windows:

```powershell
setx ANTHROPIC_BASE_URL "http://localhost:8000"
setx ANTHROPIC_API_KEY "dummy"
```

No LiteLLM or other proxy required; `/v1/messages` is served natively, including
streaming tool calls and `count_tokens`.

**Glean does not appear in the model picker, and should not.** `ANTHROPIC_BASE_URL`
redirects *all* of Claude Code's traffic to the proxy; the picker still lists the usual
Claude models, but whichever is selected, the request is answered by Glean. The model
name is ignored on purpose (see the model section below).

To confirm it is working, watch the proxy terminal while sending a message:

```
-> glean (anthropic): 1 msgs, 14 tools, stream=True, model=OPUS_5_MS
```

Optional touches:

```powershell
$env:ANTHROPIC_MODEL = "glean"        # label the UI "glean"
$env:ANTHROPIC_AUTH_TOKEN = "dummy"   # if requests never reach the proxy
```

`/status` inside Claude Code shows which base URL it is really using.

### VS Code — Cline or Roo Code (recommended)

These are the most reliable clients here, because they do **not** use OpenAI function
calling: they describe their tools in the prompt and parse the reply themselves. That
bypasses this proxy's tool-call emulation — the reply is just text — so there is one
less probabilistic layer between Glean and your editor.

**1. Start the proxy** in a terminal and leave it running:

```powershell
python -m uvicorn proxy:app --host 127.0.0.1 --port 8000
```

**2. Install Cline**: Extensions view (`Ctrl+Shift+X`) → search "Cline" → Install.
Open it from the robot icon in the sidebar.

**3. Configure it** via the gear icon in the Cline panel:

| Field | Value |
|---|---|
| API Provider | `OpenAI Compatible` |
| Base URL | `http://localhost:8000/v1` |
| API Key | `dummy` (any non-empty string) |
| Model ID | `glean` |

Then expand the model options below and set:

| Field | Value |
|---|---|
| Context Window | `200000` |
| Max Output Tokens | `8192` |
| Supports Images | off |
| Supports Browser Use / Computer Use | off |
| Input / Output Price | `0` |

Cline asks for these rather than reading `/v1/models`, so enter them by hand. They must
match `GLEAN_CONTEXT_WINDOW` and `GLEAN_MAX_OUTPUT_TOKENS` in `.env`.

**4. Open a folder** in VS Code — Cline works against a workspace, and refuses to run
without one.

**5. Send a task**, e.g. *"list the files in this project"*. Cline proposes each command
and waits for **Approve**. Read-only commands can be auto-approved in its settings.

Use **Act** mode for editing files and running commands; **Plan** mode only discusses.

**6. Confirm it is reaching the proxy** — the proxy terminal should log:

```
-> glean: 1 msgs, 0 tools, stream=True, model=OPUS_5_MS
```

`0 tools` is correct here: Cline sends its tool instructions as prose, so this proxy's
emulation stays out of the way.

### Cursor / Continue.dev / Aider

Provider **OpenAI**, base URL `http://localhost:8000/v1`, any dummy API key, model
`glean`.

Continue.dev (`~/.continue/config.json`):

```json
"models": [
  {
    "title": "Glean",
    "provider": "openai",
    "model": "glean",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "dummy"
  }
]
```

Continue.dev's agent mode and Cursor do use native tool calling, which goes through the
emulation described under [Tool calling](#tool-calling).

## Layout

```
proxy.py              the server (both API dialects)
get_credentials.py    capture/refresh your Glean session
doctor.py             diagnose problems and print the fix
tests/                test_translation.py (offline), test_e2e.py, test_anthropic.py
tools/                probes and diagnostics for Glean's API
.env                  your captured session (gitignored)
```

## When something breaks

Run the doctor first — it names the problem and the command that fixes it:

```bash
python doctor.py              # full check
python doctor.py --offline    # config only, no network
```

It checks the `.env`, cookie contents, **session age** (decoded from the cookie),
whether Glean still accepts the session, whether **Glean's frontend version has
moved on**, whether a real chat request works, and whether the proxy is running.

| Symptom | Cause | Fix |
|---|---|---|
| `401`, or `[Glean session expired]` in a reply | Session expired (about weekly) | `python get_credentials.py` |
| `500` from Glean on every request | `agentConfig` rejected, usually a bad `GLEAN_MODEL_SET_ID` | Set `GLEAN_MODEL_SET_ID=OPUS_5_MS`, or re-capture with `python tools/capture_payload.py` |
| Answers describe files you do not have | Company retrieval or Glean's own sandbox | Ensure `GLEAN_ENABLE_COMPANY_TOOLS=false` |
| Replies are empty | Model set unavailable | Try `GLEAN_MODEL_SET_ID=OPUS_5_MS` |
| Doctor warns the client version differs | Glean shipped a frontend update | Copy the version doctor prints into `GLEAN_CLIENT_VERSION` |
| Browser closes before you can log in | — | Fixed; `get_credentials.py` waits up to 15 minutes and saves your login |

## Model, thinking mode, and sources

The proxy sends the same `agentConfig` the web app sends, so UI settings are available
as configuration:

```bash
GLEAN_AGENT=ADVANCED               # ADVANCED (reasoning) or FAST
GLEAN_MODEL_SET_ID=OPUS_5_MS       # model selection
GLEAN_ENABLE_COMPANY_TOOLS=false   # company/enterprise retrieval
GLEAN_ENABLE_WEB_SEARCH=false      # web search
GLEAN_SAVE_CHAT=false              # keep proxy traffic out of your chat history
```

Select a model per request with the `model` field, but **only Glean-style IDs**
(`UPPER_SNAKE`, e.g. `OPUS_5_MS`) are honoured:

```json
{"model": "OPUS_5_MS", "messages": [...]}
```

Client model names such as `claude-opus-4-5-20251101` or `gpt-4o` are deliberately
ignored in favour of `GLEAN_MODEL_SET_ID`. Forwarding them made Glean reject the
unknown value by **discarding the entire `agentConfig`**, which silently re-enabled
company retrieval — tool calling dropped from 6/6 to 3/6 before this was fixed.

### Keep company tools off for coding

**`GLEAN_ENABLE_COMPANY_TOOLS=false` is load-bearing, not a preference.** With
retrieval on, "Find every TODO comment in this repository" returned a confident answer
citing `skills/create-skill/scripts/init_skill.py:118` — a file in some *other* indexed
repo. A coding agent would take that as fact. With it off, the same request correctly
produces `bash({"command": "grep -rn -E 'TODO' ."})`.

Glean can fabricate local answers two ways:

1. **Enterprise index** — fixed by `GLEAN_ENABLE_COMPANY_TOOLS=false`.
2. **Its own server-side shell sandbox** — Glean has an internal `Shell` tool and will
   describe *its* filesystem (`/home/user`, `.duckdb`, `tool_sdk.py`). No flag for this
   is known; the injected tool prompt steers away from it.

If a reply mentions files you do not recognise, suspect these before believing it.

## Tool calling

Glean's API has no usable `tools` parameter, so tool calling is emulated: schemas are
injected into the prompt and Glean is asked to reply with

```
<tool_call>
{"name": "bash", "arguments": {"command": "grep -rn TODO ."}}
</tool_call>
```

which the proxy converts into a native OpenAI `tool_calls` or Anthropic `tool_use`
response. The harness runs it locally and returns the output, which the proxy relays
back to Glean as a user turn.

The wording was tuned against the live API (`tools/tune_tools.py`), because Glean's
persona otherwise replies "I can't access that":

| framing | tool-call rate |
|---|---|
| firm ("you are the reasoning engine; never claim you lack access") | 3/3 |
| plain tool listing | 2/3 |
| few-shot examples | 0/3 — returned *empty* replies |

Because adherence is prompt-based rather than enforced, it is probabilistic. The parser
also accepts bare and fenced JSON as a fallback. Compare dialects with
`python tools/compare_dialects.py`.

### Staying in character

Glean is an assistant with its own persona, not a bare model, so two settings keep it
behaving like a backend:

- **`GLEAN_HARNESS_MODE=true`** injects a short instruction to follow the client's format
  and not greet the user, introduce itself, or mention Glean features. This matters most
  for clients that send **no** `tools` parameter (Cline, Roo Code) — without it Glean gets
  no instruction to stay in character and replies conversationally.
- **`GLEAN_INCOGNITO_MODE=true`** sets the `incognitoMode` request field, suppressing
  personalization that otherwise leaks in as greetings by name or references to your
  documents.

Check Cline-style protocol adherence with `python tools/test_cline_protocol.py`, which
scores tool-tag usage and persona leakage with the harness prompt on and off.

Note that a bare "hi" is a poor first message for Cline: its protocol requires a tool
call in every message, and a greeting invites a conversational reply. Give it a real
task instead.

### Why not native tool calling?

The request has a `clientTools` field and the protocol has `TOOL_USE`, `TOOL_RESULT`,
and `SERVER_TOOL` message types, so a native path exists — but it is not reachable:
three plausible `clientTools` schemas returned opaque 500s and a fourth was accepted
and ignored. Registering real tools appears to require admin-configured MCP servers or
action packs. Re-check with `tools/probe_clienttools.py` and `tools/inspect_bundle.py`.

## Running on another machine (Linux VM, container, server)

The session cookie is a bearer credential and is **not tied to your device**, so:

```bash
# On the machine where you can log in (needs a GUI browser):
python get_credentials.py

# Copy .env to the VM, then there:
pip install -r requirements.txt
python doctor.py
python -m uvicorn proxy:app --host 127.0.0.1 --port 8000
```

Notes:

- `get_credentials.py` needs a real browser, so run it on your desktop and copy `.env`
  over. Playwright is not needed on the VM at all.
- The VM needs network access to your Glean backend host (VPN if applicable).
- `.env` is a live credential: copy it over SSH/SCP rather than a shared drive, and
  `chmod 600 .env`.
- One session works from several machines at once; expiry is unchanged, and refreshing
  means re-copying `.env`.
- To serve other machines, bind `--host 0.0.0.0` — but the proxy has **no
  authentication**, so anyone who can reach the port can use your Glean account. Prefer
  an SSH tunnel: `ssh -L 8000:localhost:8000 user@vm`.

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
| Built-in tools | `/api/v1/listtools` lists 18 server-side tools, including `Glean Search` and a `Shell` sandbox |

Ordering matters more than it looks: sent chronologically, Glean replies to the *oldest*
message, which reads as the model ignoring you rather than as a proxy bug.

## Diagnostics

```bash
python doctor.py                      # start here
python tests/test_translation.py      # offline unit tests, no network
python tests/test_e2e.py              # live: OpenAI endpoint, streaming, tools
python tests/test_anthropic.py        # live: Anthropic endpoint for Claude Code
python tools/probe_api.py             # raw request/response shape
python tools/probe_api.py --stream    # raw streaming format
python tools/probe_api.py --order     # confirm message ordering
python tools/dump_stream.py           # confirm no stream content is dropped
python tools/tune_tools.py            # score tool-call prompt wordings
python tools/compare_dialects.py      # tool-call rate, OpenAI vs Anthropic
python tools/capture_payload.py       # capture payloads behind UI settings
python tools/inspect_bundle.py        # search Glean's frontend for API details
python tools/probe_clienttools.py     # re-test native clientTools support
python tools/discover_models.py       # hunt for available model IDs
curl http://localhost:8000/health
```

`capture_payload.py` is the tool to reach for when Glean's UI gains a setting you want:
enable it in the browser, send a prompt, and it prints the JSON fields that changed.

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `GLEAN_BACKEND_URL` | Tenant backend, e.g. `https://infoblox-be.glean.com` |
| `GLEAN_COOKIE` | Session cookies (captured automatically) |
| `GLEAN_EMAIL` | Sent as `X-Scio-Actas` |
| `GLEAN_CLIENT_VERSION` | Frontend build string from the captured request |
| `GLEAN_REVERSE_MESSAGES` | Newest-first ordering; `true` (leave it) |
| `GLEAN_AGENT` | `ADVANCED` or `FAST` |
| `GLEAN_MODEL_SET_ID` | Default model, e.g. `OPUS_5_MS` |
| `GLEAN_ENABLE_COMPANY_TOOLS` | Enterprise retrieval; keep `false` for coding |
| `GLEAN_ENABLE_WEB_SEARCH` | Web search, default `false` |
| `GLEAN_SAVE_CHAT` | Save to Glean chat history, default `false` |
| `GLEAN_INCOGNITO_MODE` | Suppress Glean personalization, default `true` |
| `GLEAN_HARNESS_MODE` | Tell Glean to act as a backend model, default `true` |
| `GLEAN_CONTEXT_WINDOW` | Advisory context limit in `/v1/models`, default `200000` |
| `GLEAN_MAX_OUTPUT_TOKENS` | Advisory output limit, default `8192` |
| `GLEAN_TIMEZONE_OFFSET` | Minutes, default `420` |
| `GLEAN_TIMEOUT` | Upstream timeout in seconds, default `300` |
| `LOG_LEVEL` | e.g. `DEBUG` |

`GLEAN_CONTEXT_WINDOW` is advisory — Glean does not publish its real limit, so 200k is
an estimate for an Opus-class model set. If long conversations start failing or replying
oddly, lower it (try `128000`); clients use it to decide when to truncate.

## Notes

- `.env` holds live session cookies — anyone with them can act as you in Glean.
- This uses an internal endpoint with no stability guarantees, and web sessions may be
  rate-limited more aggressively than the official API.
- Requests count as your Glean usage and are subject to your organisation's policies.
