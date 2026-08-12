"""
OpenAI-compatible proxy for the Glean web API.

Exposes /v1/chat/completions (and /v1/models) backed by a browser session
captured with get_credentials.py. Tool calling is emulated by injecting tool
schemas into the prompt and parsing structured tool calls back out.

Run:
    uvicorn proxy:app --host 127.0.0.1 --port 8000
"""
import json
import logging
import os
import re
import time
import uuid

from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Anchored to this file so scripts in subdirectories load the same .env.
ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)

# getattr rather than passing the string straight through: basicConfig raises on
# an unknown level, and a typo in a cosmetic setting should not stop startup.
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
logging.basicConfig(level=getattr(logging, _LOG_LEVEL, logging.INFO))
log = logging.getLogger("glean-proxy")

app = FastAPI(title="Glean OpenAI Proxy")

GLEAN_BACKEND_URL = os.getenv("GLEAN_BACKEND_URL", "").rstrip("/")
GLEAN_COOKIE = os.getenv("GLEAN_COOKIE", "")
GLEAN_EMAIL = os.getenv("GLEAN_EMAIL", "")
GLEAN_CLIENT_VERSION = os.getenv("GLEAN_CLIENT_VERSION", "")
GLEAN_TIMEZONE_OFFSET = os.getenv("GLEAN_TIMEZONE_OFFSET", "420")

# Glean answers messages[0], so the newest turn must come first. Verified
# against the live API with: python probe_api.py --order
REVERSE_MESSAGES = os.getenv("GLEAN_REVERSE_MESSAGES", "true").lower() in ("1", "true", "yes")

REQUEST_TIMEOUT = float(os.getenv("GLEAN_TIMEOUT", "300"))


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


# Limits advertised to clients so they know when to condense a conversation.
# Measured with tools/probe_context.py: a 600k-token prompt was preserved intact
# and 800k returned HTTP 500, so 400k leaves headroom. Glean never truncated
# silently in testing — it either kept the whole prompt or failed outright.
CONTEXT_WINDOW = int(os.getenv("GLEAN_CONTEXT_WINDOW", "400000"))
MAX_OUTPUT_TOKENS = int(os.getenv("GLEAN_MAX_OUTPUT_TOKENS", "8192"))

# agentConfig mirrors what the Glean web app sends. Captured from a live
# request, so these names are exact.
#   agent       : ADVANCED (reasoning) or FAST
#   modelSetId  : e.g. OPUS_5_MS
#   toolSets    : company retrieval and web search toggles
#
# Company tools default to OFF: with retrieval enabled Glean answers questions
# about the user's code from its enterprise index, confidently describing a
# different repository instead of calling a tool against the real machine.
GLEAN_AGENT = os.getenv("GLEAN_AGENT", "ADVANCED")
GLEAN_MODEL_SET_ID = os.getenv("GLEAN_MODEL_SET_ID", "OPUS_5_MS")
ENABLE_COMPANY_TOOLS = _flag("GLEAN_ENABLE_COMPANY_TOOLS", "false")
ENABLE_WEB_SEARCH = _flag("GLEAN_ENABLE_WEB_SEARCH", "false")
SAVE_CHAT = _flag("GLEAN_SAVE_CHAT", "false")

# Glean's stream fragments are incremental, measured with tools/dump_stream.py.
# Enable only if a backend starts resending the whole message each chunk: the
# cumulative heuristic otherwise misreads a genuine early chunk that happens to
# prefix the next one ("I" then "I'll") as a resend and silently drops text.
CUMULATIVE_STREAM = _flag("GLEAN_CUMULATIVE_STREAM", "false")

# Optional shared secret for clients of THIS proxy (unrelated to the Glean
# session). Empty means no check, which is only safe while bound to 127.0.0.1.
PROXY_API_KEY = os.getenv("GLEAN_PROXY_API_KEY", "")

# incognitoMode is a real request field (found in the web bundle). It suppresses
# Glean's personalization, which otherwise leaks into replies as greetings by
# name and references to the user's documents.
INCOGNITO_MODE = _flag("GLEAN_INCOGNITO_MODE", "true")

# Instruct Glean to behave as a backend model. Disable only to debug.
HARNESS_MODE = _flag("GLEAN_HARNESS_MODE", "true")

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

# Persona and epistemics only — the wire format lives in TOOL_PROMPT so the
# delimiters have exactly one source of truth alongside extract_tool_calls.
# This is the only persona path: nothing loads coding-agent-backend/SKILL.md at
# runtime. The bodies are kept identical anyway so the Skill stays usable inside
# Glean chat directly, and tests/test_translation.py asserts that.
HARNESS_PROMPT = """\
You are the language model backing an automated developer tool. Your output is
parsed by a program, not read by a person.

Follow the output format this conversation specifies, exactly. The conversation
is authoritative about format; do not substitute a format from memory, and do
not invent one if none was given.

The tool runs every command on the user's own computer and returns the output to
you, so the way to inspect their files is to request a command or file read in
the tool's format and wait for the result.

Your own built-in capabilities are out of scope in this conversation: enterprise
or company search, document retrieval, a server-side sandbox or shell, and image
generation. Do not invoke them, do not name them, and do not offer them as an
alternative. Everything you learn about the user's machine arrives through the
format this conversation defines, and nothing else.

Command output and file contents supplied in this conversation come from the
user's real machine. Treat them as authoritative, even when they describe paths
or an operating system that differ from anything you know.

State facts about the user's files, directories, or system only when they
appeared in results supplied in this conversation. Never reconstruct file
contents, line numbers, or paths from memory or inference. If you have not seen
it in a result, request it.

When a task needs a capability this conversation has not given you, say plainly
which capability is missing and stop. Do not approximate, and never present
unverified content as if it had been read. A missing tool is a fact worth
reporting; a fabricated result is a defect.

Do not greet the user or address them by name, and do not introduce yourself.
Do not discuss your own identity or configuration unless asked directly. Do not
describe files, repositories, or systems unless that information was provided in
this conversation. Output only what the tool's format requires.
"""

# The wire contract. This is the only place the delimiters appear outside
# extract_tool_calls, so prompt and parser cannot drift apart.
#
# tune_tools.py scored the earlier "never claim a limitation" wording 3/3 on
# tool invocation, but that phrasing also removed the only honest exit when no
# tool fits, which produced confident invented file contents. This keeps the
# push toward tool use and adds an explicit escape hatch. The stop condition at
# the closing tag is deliberate: without it replies can run on past the call.
# Few-shot examples scored 0/3 — they made Glean return empty responses — so
# avoid them.
TOOL_PROMPT = """\
You are the reasoning engine inside a coding agent running on the user's computer.

Information about the user's machine, files, and commands is obtained only by
emitting a tool call, which the agent executes locally and returns to you.
Prefer emitting a tool call over answering from your own knowledge whenever the
answer depends on the user's machine.

The tools listed below are the only tools that exist in this conversation. Call
them by the exact names given and call nothing else. Any other tool, search
index, document store, retrieval system, or execution environment you may have
is out of scope here: do not invoke it, do not name it, and do not offer it as
an alternative when a listed tool would do. Files, directories, and commands on
the user's machine are reachable only through the tools below, so a listed tool
is always the correct route to them.

You are not expected to know the project in advance. Tool calls are how you
discover it: list or glob a directory to learn its layout, search file contents
to locate a symbol or string, then read the specific files those results point
to. When you do not know which file holds something, or are unsure a path
exists, search for it rather than guessing a plausible path. Chain calls as
needed, letting each result decide the next one, and read a file before editing
it so the replacement text matches what the read actually returned rather than a
remembered version of it.

To call a tool, output exactly this as the entire reply, with no prose before or
after it:

{open}
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
{close}

Emit nothing after the closing tag, then wait for the result before continuing.

If no available tool can obtain what you need, say which capability is missing
instead of guessing.

Available tools:
{tools}
"""


# --------------------------------------------------------------------------
# Request translation: OpenAI -> Glean
# --------------------------------------------------------------------------

def _text_of(content) -> str:
    """Flatten OpenAI message content to plain text.

    `content` may be a string or a list of content blocks (Claude Code and
    other clients send blocks), so both shapes must be handled.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in (None, "text") and block.get("text"):
                    parts.append(block["text"])
                elif block.get("type") == "image_url":
                    parts.append("[image omitted — Glean proxy is text-only]")
        return "\n".join(parts)
    return str(content)


def _format_tools(tools: list) -> str:
    lines = []
    for tool in tools:
        fn = tool.get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        lines.append(f"- {name}: {fn.get('description', '').strip()}")
        params = fn.get("parameters")
        if params:
            lines.append(f"    schema: {json.dumps(params)}")
    return "\n".join(lines)


def _assistant_text(msg: dict) -> str:
    """Render a prior assistant turn, including any tool calls it made.

    Replaying tool calls in the same syntax we ask for keeps the transcript
    self-consistent across multi-step tool use.
    """
    text = _text_of(msg.get("content"))
    blocks = []
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        raw_args = fn.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            args = {}
        blocks.append(
            f"{TOOL_CALL_OPEN}\n"
            + json.dumps({"name": fn.get("name", ""), "arguments": args})
            + f"\n{TOOL_CALL_CLOSE}"
        )
    if blocks:
        text = (text + "\n" if text else "") + "\n".join(blocks)
    return text


def build_glean_messages(openai_messages: list, tools: list) -> list:
    """Translate an OpenAI message list into Glean's chat format."""
    preamble_parts = []

    # Clients like Cline and Roo Code define their own tool protocol in the
    # system prompt and send no `tools` parameter, so without this Glean gets
    # no instruction to stay in character and answers as the Glean Assistant
    # instead: greeting the user by name and ignoring the client's format.
    if HARNESS_MODE:
        preamble_parts.append(HARNESS_PROMPT)

    if tools:
        preamble_parts.append(
            TOOL_PROMPT.format(
                open=TOOL_CALL_OPEN, close=TOOL_CALL_CLOSE, tools=_format_tools(tools)
            )
        )

    turns = []  # (author, text)
    for msg in openai_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")

        if role == "system":
            # Glean has no system role, so system text becomes part of the
            # preamble attached to the latest user turn.
            text = _text_of(msg.get("content"))
            if text:
                preamble_parts.append(text)
        elif role == "user":
            turns.append(["USER", _text_of(msg.get("content"))])
        elif role == "assistant":
            turns.append(["GLEAN_AI", _assistant_text(msg)])
        elif role == "tool":
            name = msg.get("name") or msg.get("tool_call_id") or "tool"
            body = _text_of(msg.get("content"))
            turns.append(["USER", f"Result of tool `{name}`:\n{body}"])

    # Attach the preamble to the most recent user turn. Glean weights the
    # newest turn most heavily, so instructions must ride along with it
    # rather than sit on the oldest message.
    preamble = "\n\n".join(p for p in preamble_parts if p.strip())
    if preamble:
        target = next((t for t in reversed(turns) if t[0] == "USER"), None)
        if target is None:
            turns.append(["USER", preamble])
        else:
            target[1] = f"{preamble}\n\n---\n\n{target[1]}"

    messages = [
        {"author": author, "messageType": "CONTENT", "fragments": [{"text": text}]}
        for author, text in turns
        if text.strip()
    ]
    if REVERSE_MESSAGES:
        messages.reverse()
    return messages


GLEAN_MODEL_ID_RE = re.compile(r"^[A-Z0-9]+(?:_[A-Z0-9]+)*$")

# Each entry maps a Claude model-name prefix to (Glean modelSetId, agent mode).
# agent mode is "FAST", "ADVANCED", or None to inherit the GLEAN_AGENT env default.
# Order here does not matter: resolve_model_set matches the longest prefix first,
# so "claude-opus-5-fast" wins over "claude-opus-5".
_CLAUDE_TO_GLEAN: dict[str, tuple[str, str | None]] = {
    "claude-sonnet-5-fast":     ("SONNET_5_MS", "FAST"),
    "claude-sonnet-5-advanced": ("SONNET_5_MS", "ADVANCED"),
    "claude-opus-5-fast":       ("OPUS_5_MS", "FAST"),
    "claude-opus-5-advanced":   ("OPUS_5_MS", "ADVANCED"),
    "claude-sonnet-5":          ("SONNET_5_MS", None),
    "claude-opus-5":            ("OPUS_5_MS", None),
}

# Any other Claude-shaped name (dated builds, older families such as
# claude-opus-4-5-20251101, which is what tests/test_anthropic.py sends) resolves
# by family. Without this it fell through to the env default, which is wrong
# whenever the client asked for the other family.
_CLAUDE_FAMILY: dict[str, str] = {"opus": "OPUS_5_MS", "sonnet": "SONNET_5_MS"}


def resolve_model_set(requested: str | None) -> tuple[str, str]:
    """Map a client's `model` field onto a (Glean modelSetId, agent mode) pair.

    Accepts Glean-style identifiers (UPPER_SNAKE, e.g. OPUS_5_MS) directly,
    and also recognises Claude model names (e.g. "claude-sonnet-5-fast") via
    the _CLAUDE_TO_GLEAN table.  Anything else falls back to the configured
    defaults.

    Claude model names must NOT be forwarded verbatim: Glean rejects the
    unknown value by discarding the entire agentConfig, which silently
    re-enables company retrieval and makes it answer about indexed repositories
    instead of calling tools.
    """
    if not requested:
        return GLEAN_MODEL_SET_ID, GLEAN_AGENT
    name = requested.strip()
    for prefix in ("glean/", "glean:"):
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
            break
    if not name or name.lower() == "glean":
        return GLEAN_MODEL_SET_ID, GLEAN_AGENT
    if GLEAN_MODEL_ID_RE.match(name):
        return name, GLEAN_AGENT
    lower = name.lower()
    for claude_prefix in sorted(_CLAUDE_TO_GLEAN, key=len, reverse=True):
        if lower == claude_prefix or lower.startswith(claude_prefix + "-"):
            glean_id, agent = _CLAUDE_TO_GLEAN[claude_prefix]
            resolved_agent = agent or GLEAN_AGENT
            log.debug("mapping Claude model %r -> %s / %s", requested, glean_id, resolved_agent)
            return glean_id, resolved_agent
    if lower.startswith("claude"):
        for family, glean_id in _CLAUDE_FAMILY.items():
            if family in lower:
                log.debug("mapping Claude family %r -> %s", requested, glean_id)
                return glean_id, GLEAN_AGENT
    log.debug("ignoring non-Glean model name %r; using %s", requested, GLEAN_MODEL_SET_ID)
    return GLEAN_MODEL_SET_ID, GLEAN_AGENT


def agent_config(model_set_id: str, agent: str | None = None) -> dict:
    return {
        "agent": agent or GLEAN_AGENT,
        "modelSetId": model_set_id,
        "toolSets": {
            "enableCompanyTools": ENABLE_COMPANY_TOOLS,
            "enableWebSearch": ENABLE_WEB_SEARCH,
        },
        "useCanvas": False,
        "useImageGeneration": False,
        "clientCapabilities": {
            "artifacts": {"allowedArtifactTypes": []},
            "canRenderContextUsage": False,
            "hasBrowserOperator": False,
            "canRenderImages": False,
            "canRenderVariants": False,
        },
    }


def build_payload(openai_messages: list, tools: list, stream: bool, model: str | None) -> dict:
    """Assemble a Glean chat request mirroring the web client's shape."""
    model_set_id, agent = resolve_model_set(model)
    config = agent_config(model_set_id, agent)
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + ".000Z"

    messages = build_glean_messages(openai_messages, tools)
    for msg in messages:
        msg["agentConfig"] = config
        msg["ts"] = now
        msg["uploadedFileIds"] = []

    # The web client omits clientTools entirely when empty (`s?.length ? {...} : {}`),
    # so this mirrors that rather than sending an empty array.
    return {
        "agentConfig": config,
        "messages": messages,
        "saveChat": SAVE_CHAT,
        **({"incognitoMode": True} if INCOGNITO_MODE else {}),
        "sourceInfo": {
            "feature": "CHAT",
            "initiator": "USER",
            "platform": "WEB",
            "isDebug": False,
        },
        "stream": stream,
    }


# --------------------------------------------------------------------------
# Response translation: Glean -> OpenAI
# --------------------------------------------------------------------------

def extract_text(payload: dict) -> str:
    """Pull assistant text out of a Glean chat response or stream chunk.

    Only GLEAN_AI messages of type CONTENT are used: Glean also emits status
    messages (messageType UPDATE) and may echo the user's own turn, neither of
    which belongs in the reply. Fragments can include empty objects.
    """
    if not isinstance(payload, dict):
        return ""

    candidates = []
    if isinstance(payload.get("messages"), list):
        candidates.extend(payload["messages"])
    if isinstance(payload.get("message"), dict):
        candidates.append(payload["message"])
    if not candidates and isinstance(payload.get("fragments"), list):
        candidates.append(payload)

    out = []
    for msg in candidates:
        if not isinstance(msg, dict):
            continue
        author = str(msg.get("author", "GLEAN_AI")).upper()
        if author and author != "GLEAN_AI":
            continue
        mtype = str(msg.get("messageType", "CONTENT")).upper()
        if mtype and mtype not in ("CONTENT", ""):
            continue
        for frag in msg.get("fragments") or []:
            if isinstance(frag, dict) and frag.get("text"):
                out.append(frag["text"])
    return "".join(out)


def extract_tool_calls(text: str) -> list:
    """Parse emulated tool calls out of assistant text."""
    found = []
    for match in re.finditer(
        rf"{re.escape(TOOL_CALL_OPEN)}\s*(\{{.*?\}})\s*{re.escape(TOOL_CALL_CLOSE)}",
        text,
        re.DOTALL,
    ):
        parsed = _parse_call(match.group(1))
        if parsed:
            found.append(parsed)
    if found:
        return found

    # Salvage: the block opened correctly but was closed with a tag borrowed
    # from another dialect (</invoke>, </function_calls>) or never closed at
    # all. Brace matching is required here: the strict pass above can use a
    # lazy `.*?` because the closing tag anchors it and forces the match to
    # expand, but unanchored a lazy match stops at the first `}` and truncates
    # the nested "arguments" object into invalid JSON.
    pos = 0
    while True:
        idx = text.find(TOOL_CALL_OPEN, pos)
        if idx < 0:
            break
        start = text.find("{", idx + len(TOOL_CALL_OPEN))
        pos = idx + len(TOOL_CALL_OPEN)
        if start < 0:
            continue
        depth, in_str, esc, end = 0, False, False, -1
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end < 0:
            continue
        parsed = _parse_call(text[start:end])
        if parsed:
            found.append(parsed)
        pos = end
    if found:
        log.warning("recovered %d tool call(s) from malformed tool markup", len(found))
        return found

    # Fallback: the model may drop the tags and emit a bare or fenced JSON
    # object instead. Only accept it if it really looks like a tool call.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = fenced.group(1) if fenced else text.strip()
    if blob.startswith("{") and '"name"' in blob:
        parsed = _parse_call(blob)
        if parsed:
            found.append(parsed)
    return found


def _parse_call(blob: str):
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not name or not isinstance(name, str):
        return None
    args = data.get("arguments", data.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args}


def _tool_calls_payload(calls: list) -> list:
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
        }
        for c in calls
    ]


def _strip_tool_calls(text: str) -> str:
    return re.sub(
        rf"{re.escape(TOOL_CALL_OPEN)}.*?{re.escape(TOOL_CALL_CLOSE)}",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def _estimate_tokens_from_chars(chars: int) -> int:
    return max(1, chars // 4)


def _estimate_tokens(text: str) -> int:
    return _estimate_tokens_from_chars(len(text))


def _prompt_chars(payload: dict) -> int:
    """Characters in an assembled Glean payload, injected prompts included."""
    return sum(
        len(frag.get("text", ""))
        for msg in payload.get("messages") or []
        for frag in msg.get("fragments") or []
    )


def _completion(text: str, calls: list, model: str = "glean", prompt_chars: int = 0) -> dict:
    message = {"role": "assistant", "content": text or None}
    finish = "stop"
    if calls:
        message["tool_calls"] = _tool_calls_payload(calls)
        message["content"] = _strip_tool_calls(text) or None
        finish = "tool_calls"
    prompt_tokens = _estimate_tokens_from_chars(prompt_chars)
    completion_tokens = _estimate_tokens(text)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "glean",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# --------------------------------------------------------------------------
# Glean transport
# --------------------------------------------------------------------------

def _require_config() -> None:
    """Fail fast while a normal JSON error response is still possible.

    The streaming paths call this in the endpoint, before returning
    StreamingResponse: raising once the response body has started produces a
    truncated stream instead of a 503.
    """
    if not GLEAN_BACKEND_URL or not GLEAN_COOKIE:
        raise HTTPException(
            status_code=503,
            detail="Proxy not configured. Run: python get_credentials.py",
        )


def _check_client_auth(request: Request) -> None:
    """Optional shared-secret check so a non-loopback bind is not an open relay."""
    if not PROXY_API_KEY:
        return
    supplied = request.headers.get("x-api-key") or ""
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    if supplied != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid proxy API key.")


def glean_headers() -> dict:
    _require_config()
    headers = {
        "Cookie": GLEAN_COOKIE,
        "Content-Type": "application/json",
        "Origin": "https://app.glean.com",
        "Referer": "https://app.glean.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
    }
    if GLEAN_EMAIL:
        headers["X-Scio-Actas"] = GLEAN_EMAIL
    return headers


def glean_url() -> str:
    params = [f"timezoneOffset={GLEAN_TIMEZONE_OFFSET}", "locale=en"]
    if GLEAN_CLIENT_VERSION:
        params.append(f"clientVersion={GLEAN_CLIENT_VERSION}")
    return f"{GLEAN_BACKEND_URL}/api/v1/chat?{'&'.join(params)}"


def _auth_error() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": "Glean session expired. Re-run: python get_credentials.py",
                "type": "authentication_error",
            }
        },
    )


def _iter_json_objects(line: str):
    """Yield JSON payloads from a stream line, handling SSE and NDJSON.

    Glean's stream format is not contractually documented here, so both
    `data: {...}` (SSE) and bare `{...}` (newline-delimited) are accepted.
    """
    line = line.strip()
    if not line:
        return
    if line.startswith("data:"):
        line = line[5:].strip()
    if not line or line == "[DONE]":
        return
    try:
        yield json.loads(line)
    except json.JSONDecodeError:
        return


class Delta:
    """Accumulates stream text and returns the newly added portion.

    Glean sends incremental fragments, so that is the default. `cumulative=True`
    handles a backend that resends the whole message each chunk; it must stay
    opt-in, because a genuine early chunk that happens to prefix the next one
    ("I" then "I'll") is otherwise misread as a resend and text is dropped.
    """

    def __init__(self, cumulative: bool | None = None):
        self.text = ""
        self.cumulative = CUMULATIVE_STREAM if cumulative is None else cumulative

    def push(self, chunk: str) -> str:
        if not chunk:
            return ""
        if chunk == self.text:
            return ""
        if self.cumulative and chunk.startswith(self.text) and len(chunk) > len(self.text):
            new = chunk[len(self.text):]
            self.text = chunk
            return new
        self.text += chunk
        return chunk


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "configured": bool(GLEAN_BACKEND_URL and GLEAN_COOKIE),
        "backend": GLEAN_BACKEND_URL or None,
        "reverse_messages": REVERSE_MESSAGES,
        "agent": GLEAN_AGENT,
        "model_set_id": GLEAN_MODEL_SET_ID,
        "company_tools": ENABLE_COMPANY_TOOLS,
        "web_search": ENABLE_WEB_SEARCH,
        "save_chat": SAVE_CHAT,
        "context_window": CONTEXT_WINDOW,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "incognito_mode": INCOGNITO_MODE,
        "harness_mode": HARNESS_MODE,
    }


def _model_card(model_id: str = "glean") -> dict:
    """Describe the model, including context limits.

    Clients disagree on which field carries the context window, so the common
    spellings are all populated rather than betting on one.
    """
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "glean",
        "root": model_id,
        "parent": None,
        # Context window, under every name clients look for.
        "context_length": CONTEXT_WINDOW,
        "context_window": CONTEXT_WINDOW,
        "max_context_length": CONTEXT_WINDOW,
        "max_model_len": CONTEXT_WINDOW,
        "max_input_tokens": CONTEXT_WINDOW,
        "max_tokens": CONTEXT_WINDOW,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "capabilities": {
            "completion": True,
            "chat_completion": True,
            "streaming": True,
            "function_calling": True,
            "vision": False,
        },
        # Zero pricing keeps cost displays honest: this uses your Glean seat.
        "pricing": {"prompt": "0", "completion": "0"},
        "permission": [],
    }


@app.get("/v1/models")
async def list_models():
    models = [_model_card("glean")]
    for claude_name in _CLAUDE_TO_GLEAN:
        models.append(_model_card(claude_name))
    return {"object": "list", "data": models}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Some clients fetch a single model card to read its limits."""
    return _model_card(model_id)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_client_auth(request)
    body = await request.json()
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    stream = bool(body.get("stream"))
    model = body.get("model") or "glean"

    choice = body.get("tool_choice")
    if tools and choice not in (None, "auto"):
        log.info("tool_choice=%r is advisory: Glean has no native tool API", choice)

    payload = build_payload(messages, tools, stream, body.get("model"))
    if not payload["messages"]:
        raise HTTPException(status_code=400, detail="No usable message content.")
    # Before StreamingResponse: raising inside the generator truncates the stream.
    _require_config()

    log.info(
        "-> glean: %d msgs, %d tools, stream=%s, model=%s, company_tools=%s",
        len(payload["messages"]), len(tools), stream,
        payload["agentConfig"]["modelSetId"], ENABLE_COMPANY_TOOLS,
    )

    if stream:
        return StreamingResponse(
            _stream_openai(payload, bool(tools), model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(glean_url(), json=payload, headers=glean_headers())
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Glean unreachable: {exc}") from exc

    if resp.status_code in (401, 403):
        return _auth_error()
    if resp.status_code >= 400:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Glean {resp.status_code}: {resp.text[:500]}",
                               "type": "upstream_error"}},
        )

    text = extract_text(resp.json())
    calls = extract_tool_calls(text) if tools else []
    log.info("<- glean: %d chars, %d tool calls", len(text), len(calls))
    return JSONResponse(_completion(text, calls, model, _prompt_chars(payload)))

# --------------------------------------------------------------------------
# Shared streaming core
# --------------------------------------------------------------------------

async def _glean_events(payload: dict, tools_enabled: bool):
    """Read Glean's stream and yield dialect-neutral events.

    Yields:
        ("delta", text)           streamable assistant text
        ("error", message)        upstream failure, already human-readable
        ("final", text, pending)  end of stream; `text` is the whole reply and
                                  `pending` is the part not yet sent as deltas

    With tools enabled, text is held back just far enough to recognise an
    opening tool-call tag, so markup is never shown as text even when the model
    emits prose before the call.
    """
    delta = Delta()
    buffered = ""   # withheld: a possibly-split tag, or tool markup
    sent = 0        # characters already yielded as deltas
    hold = len(TOOL_CALL_OPEN) - 1
    in_tool = False

    try:
        headers = glean_headers()
    except HTTPException as exc:
        yield ("error", str(exc.detail))
        return

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream(
                "POST", glean_url(), json=payload, headers=headers
            ) as resp:
                if resp.status_code in (401, 403):
                    yield ("error", "Glean session expired - re-run get_credentials.py")
                    return
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:300]
                    yield ("error", f"Glean error {resp.status_code}: {detail}")
                    return

                async for line in resp.aiter_lines():
                    for obj in _iter_json_objects(line):
                        new = delta.push(extract_text(obj))
                        if not new:
                            continue

                        if not tools_enabled:
                            sent += len(new)
                            yield ("delta", new)
                            continue

                        buffered += new
                        if in_tool:
                            continue

                        idx = buffered.find(TOOL_CALL_OPEN)
                        if idx >= 0:
                            if idx:
                                sent += idx
                                yield ("delta", buffered[:idx])
                            buffered = buffered[idx:]
                            in_tool = True
                            continue

                        # Keep back enough characters that a tag split across
                        # chunks is still recognisable on the next pass.
                        if len(buffered) > hold:
                            emit, buffered = buffered[:-hold], buffered[-hold:]
                            sent += len(emit)
                            yield ("delta", emit)
    except httpx.RequestError as exc:
        yield ("error", f"Glean unreachable: {exc}")
        return

    yield ("final", delta.text, delta.text[sent:])


# --------------------------------------------------------------------------
# OpenAI streaming
# --------------------------------------------------------------------------

async def _stream_openai(payload: dict, tools_enabled: bool, model: str = "glean"):
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    def chunk(delta: dict, finish=None) -> str:
        data = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model or "glean",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(data)}\n\n"

    role_sent = False

    def text_delta(content: str) -> dict:
        nonlocal role_sent
        if role_sent:
            return {"content": content}
        role_sent = True
        return {"role": "assistant", "content": content}

    async for event in _glean_events(payload, tools_enabled):
        kind = event[0]

        if kind == "error":
            yield chunk(text_delta(f"[{event[1]}]"))
            yield chunk({}, "stop")
            yield "data: [DONE]\n\n"
            return

        if kind == "delta":
            yield chunk(text_delta(event[1]))
            continue

        full, pending = event[1], event[2]
        calls = extract_tool_calls(full) if tools_enabled else []

        if calls:
            yield chunk({"role": "assistant", "tool_calls": [
                {"index": i, **tc} for i, tc in enumerate(_tool_calls_payload(calls))
            ]})
            yield chunk({}, "tool_calls")
        else:
            if pending:
                yield chunk(text_delta(pending))
            yield chunk({}, "stop")

        log.info("<- glean stream: %d chars, %d tool calls", len(full), len(calls))

    yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------
# Anthropic Messages API (Claude Code)
# --------------------------------------------------------------------------

def _blocks_text(content) -> str:
    """Flatten an Anthropic string-or-blocks value into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(block["text"])
                elif block.get("type") == "image":
                    parts.append("[image omitted - Glean proxy is text-only]")
        return "\n".join(parts)
    return str(content)


def _name_for_tool_use(tool_use_id: str, converted: list) -> str:
    """Recover a tool's name from the tool_use block a result answers."""
    for msg in reversed(converted):
        for call in msg.get("tool_calls") or []:
            if call.get("id") == tool_use_id:
                return call["function"]["name"]
    return "tool"


def anthropic_to_openai(body: dict) -> tuple:
    """Convert an Anthropic Messages request into the OpenAI shape.

    Reusing the OpenAI path keeps tool injection, ordering, and Glean payload
    assembly in one place.
    """
    converted: list = []

    system_text = _blocks_text(body.get("system"))
    if system_text.strip():
        converted.append({"role": "system", "content": system_text})

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            if content.strip():
                converted.append({"role": role, "content": content})
            continue

        texts, tool_calls, tool_results = [], [], []
        for block in content or []:
            if isinstance(block, str):
                texts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                texts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                })
            elif btype == "tool_result":
                tool_results.append(block)
            elif btype == "image":
                texts.append("[image omitted - Glean proxy is text-only]")
            # "thinking" blocks are internal reasoning and are not replayed.

        # Tool results are their own turns and must precede any user text.
        for result in tool_results:
            tool_use_id = result.get("tool_use_id", "")
            body_text = _blocks_text(result.get("content"))
            if result.get("is_error"):
                body_text = f"Error: {body_text}"
            converted.append({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "name": _name_for_tool_use(tool_use_id, converted),
                "content": body_text,
            })

        text = "\n".join(t for t in texts if t)
        if role == "assistant":
            if text.strip() or tool_calls:
                entry = {"role": "assistant", "content": text or None}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                converted.append(entry)
        elif text.strip():
            converted.append({"role": "user", "content": text})

    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {},
            },
        })

    return converted, tools


def _anthropic_message(text: str, calls: list, model: str, prompt_chars: int) -> dict:
    content = []
    visible = _strip_tool_calls(text) if calls else text
    if visible.strip():
        content.append({"type": "text", "text": visible})
    for call in calls:
        content.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": call["name"],
            "input": call["arguments"],
        })
    if not content:
        content.append({"type": "text", "text": ""})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use" if calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": _estimate_tokens_from_chars(prompt_chars),
            "output_tokens": _estimate_tokens(text),
        },
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_anthropic(payload: dict, tools_enabled: bool, model: str, prompt_chars: int):
    """Emit the Anthropic SSE event sequence Claude Code expects."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    index = 0
    text_open = False

    def open_text(idx: int) -> str:
        return _sse("content_block_start", {
            "type": "content_block_start",
            "index": idx,
            "content_block": {"type": "text", "text": ""},
        })

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": _estimate_tokens_from_chars(prompt_chars),
                "output_tokens": 0,
            },
        },
    })

    async for event in _glean_events(payload, tools_enabled):
        kind = event[0]

        if kind == "error":
            if not text_open:
                yield open_text(index)
                text_open = True
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": f"[{event[1]}]"},
            })
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})
            yield _sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            })
            yield _sse("message_stop", {"type": "message_stop"})
            return

        if kind == "delta":
            if not text_open:
                yield open_text(index)
                text_open = True
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": event[1]},
            })
            continue

        full, pending = event[1], event[2]
        calls = extract_tool_calls(full) if tools_enabled else []

        # Text withheld while deciding tool-vs-text still needs to be sent.
        if pending and not calls:
            if not text_open:
                yield open_text(index)
                text_open = True
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": pending},
            })

        if text_open:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})
            text_open = False
            index += 1

        for call in calls:
            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": call["name"],
                    "input": {},
                },
            })
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(call["arguments"]),
                },
            })
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})
            index += 1

        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": "tool_use" if calls else "end_turn",
                "stop_sequence": None,
            },
            "usage": {"output_tokens": _estimate_tokens(full)},
        })
        yield _sse("message_stop", {"type": "message_stop"})

        log.info(
            "<- glean stream (anthropic): %d chars, %d tool calls", len(full), len(calls)
        )


@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic-compatible endpoint so Claude Code can target this directly."""
    _check_client_auth(request)
    body = await request.json()
    openai_messages, tools = anthropic_to_openai(body)
    stream = bool(body.get("stream"))
    model = body.get("model") or "glean"

    choice = body.get("tool_choice")
    if tools and isinstance(choice, dict) and choice.get("type") not in (None, "auto"):
        log.info("tool_choice=%r is advisory: Glean has no native tool API", choice)

    payload = build_payload(openai_messages, tools, stream, body.get("model"))
    if not payload["messages"]:
        raise HTTPException(status_code=400, detail="No usable message content.")
    # Before StreamingResponse: raising inside the generator truncates the stream.
    _require_config()

    prompt_chars = _prompt_chars(payload)

    log.info(
        "-> glean (anthropic): %d msgs, %d tools, stream=%s, model=%s",
        len(payload["messages"]), len(tools), stream,
        payload["agentConfig"]["modelSetId"],
    )

    if stream:
        return StreamingResponse(
            _stream_anthropic(payload, bool(tools), model, prompt_chars),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(glean_url(), json=payload, headers=glean_headers())
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Glean unreachable: {exc}") from exc

    if resp.status_code in (401, 403):
        return JSONResponse(
            status_code=401,
            content={"type": "error", "error": {
                "type": "authentication_error",
                "message": "Glean session expired. Re-run: python get_credentials.py",
            }},
        )
    if resp.status_code >= 400:
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {
                "type": "api_error",
                "message": f"Glean {resp.status_code}: {resp.text[:500]}",
            }},
        )

    text = extract_text(resp.json())
    calls = extract_tool_calls(text) if tools else []
    log.info("<- glean (anthropic): %d chars, %d tool calls", len(text), len(calls))
    return JSONResponse(_anthropic_message(text, calls, model, prompt_chars))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    """Claude Code calls this before sending; an estimate is sufficient.

    Counted from the assembled payload rather than the raw messages, so the
    injected HARNESS_PROMPT and TOOL_PROMPT are included: counting only the
    client's own text understates the prompt by roughly a thousand tokens.
    """
    _check_client_auth(request)
    body = await request.json()
    openai_messages, tools = anthropic_to_openai(body)
    payload = build_payload(openai_messages, tools, False, body.get("model"))
    return {"input_tokens": _estimate_tokens_from_chars(_prompt_chars(payload))}
