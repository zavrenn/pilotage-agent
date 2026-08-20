"""Request shaping and stream consumption for the Codex Responses API.

Two things here decide how fast the agent feels:

* **The prompt cache key.** It is derived from the content that stays the same
  between turns — the instructions and the tool set — scoped per conversation.
  A stable key is the single largest contributor to time-to-first-token.
* **Encrypted reasoning replay.** The model's reasoning is returned encrypted
  and handed back on the next turn, so it does not have to think it again.

A third thing decides how bad it feels when the backend misbehaves. The Codex
endpoint has two observed failure modes that are not errors: it accepts the
connection and never sends a single event, or it sends an opening frame and
then stops. Neither closes the socket, and the HTTP client's own read timeout
is far longer, so the stream is watched here and dropped when it goes quiet. A
fresh connection normally succeeds within seconds.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .call_ids import clamp_call_id, deterministic_call_id

_TERMINAL_EVENT_TYPES = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)

# Waiting for the first event. Long on purpose: a subscription-backed request
# can spend a minute in backend admission and prompt prefill before it says
# anything, and killing that would be killing a healthy request.
DEFAULT_TTFB_TIMEOUT_SECONDS = 120.0

# Waiting between events, once the stream has proved it is alive. Any event
# counts as life, including the backend's own keepalives.
DEFAULT_IDLE_TIMEOUT_SECONDS = 12.0


class CodexStreamError(RuntimeError):
    """The API sent an error event, or the stream ended without a response."""

    def __init__(self, message: str, *, code: str = "", param: str = ""):
        super().__init__(message)
        self.code = code
        self.param = param


class CodexStreamTimeout(CodexStreamError):
    """The stream went quiet. Nothing is wrong with the request — reconnect."""


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def content_cache_key(instructions: str, tools: Optional[List[Dict[str, Any]]], scope_id: str = "") -> Optional[str]:
    """A key that changes only when the cacheable prefix changes.

    Tools are sorted by name so the hash does not depend on insertion order.
    """
    if not instructions and not tools:
        return None
    tools_part = ""
    if tools:
        sorted_tools = sorted(
            (t for t in tools if isinstance(t, dict)),
            key=lambda t: str(t.get("name") or t.get("type") or ""),
        )
        tools_part = json.dumps(sorted_tools, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    content = f"{scope_id}\x00{instructions or ''}\x00{tools_part}"
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"pck_{digest}"


def build_request(
    *,
    model: str,
    instructions: str,
    input_items: List[Dict[str, Any]],
    session_id: str,
    reasoning_effort: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the kwargs for ``responses.create``.

    An empty tool set omits the ``tools`` key entirely rather than passing
    ``[]``: the SDK iterates ``tools`` without a None guard. No
    ``max_output_tokens`` either — the Codex backend rejects it.
    """
    cache_key = content_cache_key(instructions, tools, session_id) or session_id
    # The session id is a WhatsApp chat id — a phone number. It is only ever an
    # opaque conversation handle to the backend, so it goes out hashed.
    opaque_session = "s_" + hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]

    request: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        # We keep the conversation ourselves; nothing is stored server-side.
        "store": False,
        "prompt_cache_key": cache_key,
        "reasoning": {"effort": reasoning_effort, "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        # These are real HTTP headers. The same fields inside the body are
        # rejected with HTTP 400.
        "extra_headers": {
            "session_id": opaque_session,
            "x-client-request-id": cache_key,
        },
    }

    if tools:
        request["tools"] = tools
        # The model decides whether it needs a tool, and may ask for several
        # at once when they do not depend on each other.
        request["tool_choice"] = "auto"
        request["parallel_tool_calls"] = True

    return request


# What one picture costs the model to read. A picture is not its base64 text:
# the encoded blob is megabytes of characters, but the model reads an image,
# not the characters. Counting the characters turns a single photo into a
# 300,000-token conversation and buys it three minutes of patience it has not
# earned. This number is an estimate, and deliberately on the high side —
# over-counting grants a request more patience, under-counting kills a healthy
# one.
IMAGE_TOKEN_COST = 1500

# How much longer than the between-events wait a large request may take to say
# anything at all. Prefill is where a big request spends its time, so the first
# event is exactly what a long conversation delays. The reference
# implementation gave up on scaling this and switched the watchdog off above
# 10k tokens; switching it off restores the hang it exists to catch, so we
# scale instead. The factor is a judgement call — we have no measurement of
# real prefill times yet.
FIRST_EVENT_BRACKET_FACTOR = 3.0

# The most silence we will ever sit through before deciding nothing is coming.
# The scaling above is guesswork, and unbounded guesswork ends with someone
# staring at a chat for nine minutes. Past this point the difference between a
# slow backend and a dead one stops mattering: either way the person is owed a
# reconnect. A cutoff set explicitly in configuration is honoured as given —
# this only bounds what we add on our own.
MAX_FIRST_EVENT_TIMEOUT_SECONDS = 180.0


def estimate_context_tokens(request: Dict[str, Any]) -> int:
    """Roughly how much this request asks the backend to read, in tokens.

    Four characters to the token over the text, a flat cost per picture. It
    only has to be right enough to pick a bracket below.
    """
    chars = len(str(request.get("instructions") or ""))
    images = 0

    for item in request.get("input") or []:
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, list):
            # A plain text turn, or a reasoning item carrying its encrypted
            # payload. Both are read as text, so their characters count.
            chars += len(str(item))
            continue
        for part in content:
            if not isinstance(part, dict):
                chars += len(str(part))
            elif part.get("type") == "input_image":
                images += 1
            else:
                chars += len(str(part.get("text") or ""))

    return chars // 4 + images * IMAGE_TOKEN_COST


def stream_timeouts(
    request: Dict[str, Any],
    *,
    ttfb_timeout: float = DEFAULT_TTFB_TIMEOUT_SECONDS,
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
) -> Tuple[float, float]:
    """Pick the two cutoffs for one request.

    A big request — long history, a photo — legitimately takes longer to start
    and pauses longer between events, so both waits grow with it. The brackets
    are the reference implementation's; the configured value is the floor, so
    raising it in configuration is never undone here. What we add on our own is
    capped, because nobody waits out nine minutes of silence. Zero stays zero: a
    disabled watchdog is not re-enabled by a large request.
    """
    estimate = estimate_context_tokens(request)
    if estimate > 100_000:
        bracket = 180.0
    elif estimate > 50_000:
        bracket = 120.0
    elif estimate > 10_000:
        bracket = 60.0
    else:
        bracket = 0.0
    idle = idle_timeout if idle_timeout <= 0 else max(idle_timeout, bracket)
    scaled = min(bracket * FIRST_EVENT_BRACKET_FACTOR, MAX_FIRST_EVENT_TIMEOUT_SECONDS)
    ttfb = ttfb_timeout if ttfb_timeout <= 0 else max(ttfb_timeout, scaled)
    return ttfb, idle


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


@dataclass
class StreamResult:
    text: str = ""
    reasoning_items: List[Dict[str, Any]] = field(default_factory=list)
    # What the model asked to run, in the order it asked. Each is
    # ``{"call_id", "name", "arguments"}`` — arguments as the JSON text the
    # model wrote, passed through untouched so the request we replay next turn
    # is byte-identical to the one the cache already holds.
    tool_calls: List[Dict[str, str]] = field(default_factory=list)
    status: str = ""
    usage: Any = None
    response_id: str = ""


def _field(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(name, default)
    return default if value is None else value


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except TypeError:
            return dump()
    return {}


def _raise_stream_error(event: Any) -> None:
    message = _field(event, "message", "") or ""
    code = _field(event, "code", "") or ""
    param = _field(event, "param", "") or ""
    if not message:
        nested = _field(event, "error")
        if nested is not None:
            message = _field(nested, "message", "") or ""
            code = code or (_field(nested, "code", "") or "")
            param = param or (_field(nested, "param", "") or "")
    raise CodexStreamError(message or "The Codex stream reported an error.", code=code, param=param)


async def consume_stream(
    stream: Any,
    *,
    on_text_delta: Optional[Callable[[str], None]] = None,
    ttfb_timeout: float = DEFAULT_TTFB_TIMEOUT_SECONDS,
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
) -> StreamResult:
    """Drain a Responses stream into text plus the reasoning we replay next turn.

    Raises ``CodexStreamTimeout`` if the stream stops speaking for longer than
    the caller allows. Either cutoff can be set to zero to wait forever.
    """
    text_deltas: List[str] = []
    output_items: List[Dict[str, Any]] = []
    result = StreamResult()
    terminal_seen = False
    first_event_seen = False

    events = stream.__aiter__()
    while True:
        cutoff = idle_timeout if first_event_seen else ttfb_timeout
        try:
            if cutoff > 0:
                event = await asyncio.wait_for(events.__anext__(), cutoff)
            else:
                event = await events.__anext__()
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            raise _silence_error(first_event_seen, cutoff) from None
        first_event_seen = True

        event_type = str(_field(event, "type", "") or "")

        if event_type == "error":
            _raise_stream_error(event)

        if event_type.endswith("output_text.delta"):
            delta = _field(event, "delta", "") or ""
            if delta:
                text_deltas.append(str(delta))
                if on_text_delta is not None:
                    on_text_delta(str(delta))
            continue

        if event_type == "response.output_item.done":
            item = _field(event, "item")
            if item is not None:
                output_items.append(_as_dict(item))
            continue

        if event_type in _TERMINAL_EVENT_TYPES:
            response = _field(event, "response")
            if response is not None:
                result.status = str(_field(response, "status", "") or "")
                result.usage = _field(response, "usage")
                result.response_id = str(_field(response, "id", "") or "")
                error = _field(response, "error")
                if error is not None:
                    _raise_stream_error(error)
            terminal_seen = True
            break

    result.text = "".join(text_deltas).strip()
    if not result.text:
        result.text = _text_from_items(output_items)
    result.reasoning_items = [
        item
        for item in output_items
        if item.get("type") == "reasoning" and item.get("encrypted_content")
    ]
    result.tool_calls = _tool_calls_from_items(output_items)

    if not terminal_seen and not result.text and not output_items:
        raise CodexStreamError("The Codex stream ended without a response.")

    return result


def _silence_error(first_event_seen: bool, cutoff: float) -> CodexStreamTimeout:
    if first_event_seen:
        return CodexStreamTimeout(
            f"The Codex stream sent nothing for {cutoff:g}s after starting.",
            code="codex_stream_stalled",
        )
    return CodexStreamTimeout(
        f"The Codex stream sent nothing at all within {cutoff:g}s.",
        code="codex_stream_no_first_byte",
    )


def _tool_calls_from_items(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """The calls the model made, ready to be run and then replayed.

    A call without an id gets one derived from its own name and arguments.
    That is not cosmetic: the id goes back to the API inside the request
    prefix, so a random one would give every turn a unique prefix and throw
    away the prompt cache. The index keeps two identical calls in one response
    apart.
    """
    calls: List[Dict[str, str]] = []
    for index, item in enumerate(items):
        if item.get("type") != "function_call":
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            # Nothing to run and nothing to answer with; the model will be
            # told what happened when it sees no result for it.
            continue
        arguments = item.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        arguments = str(arguments or "").strip() or "{}"
        call_id = str(item.get("call_id") or item.get("id") or "").strip()
        if not call_id:
            call_id = deterministic_call_id(name, arguments, index)
        calls.append(
            {"call_id": clamp_call_id(call_id), "name": name, "arguments": arguments}
        )
    return calls


def _text_from_items(items: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in items:
        if item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "".join(parts).strip()
