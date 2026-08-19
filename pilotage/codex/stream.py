"""Request shaping and stream consumption for the Codex Responses API.

Two things here decide how fast the agent feels:

* **The prompt cache key.** It is derived from the content that stays the same
  between turns — the instructions and the tool set — scoped per conversation.
  A stable key is the single largest contributor to time-to-first-token.
* **Encrypted reasoning replay.** The model's reasoning is returned encrypted
  and handed back on the next turn, so it does not have to think it again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_TERMINAL_EVENT_TYPES = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)


class CodexStreamError(RuntimeError):
    """The API sent an error event, or the stream ended without a response."""

    def __init__(self, message: str, *, code: str = "", param: str = ""):
        super().__init__(message)
        self.code = code
        self.param = param


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
) -> Dict[str, Any]:
    """Build the kwargs for ``responses.create``.

    No ``tools`` key is set: the SDK iterates ``tools`` without a None guard, so
    an empty tool set must be omitted entirely rather than passed as ``[]``.
    No ``max_output_tokens`` either — the Codex backend rejects it.
    """
    cache_key = content_cache_key(instructions, None, session_id) or session_id
    # The session id is a WhatsApp chat id — a phone number. It is only ever an
    # opaque conversation handle to the backend, so it goes out hashed.
    opaque_session = "s_" + hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]

    return {
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


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


@dataclass
class StreamResult:
    text: str = ""
    reasoning_items: List[Dict[str, Any]] = field(default_factory=list)
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
) -> StreamResult:
    """Drain a Responses stream into text plus the reasoning we replay next turn."""
    text_deltas: List[str] = []
    output_items: List[Dict[str, Any]] = []
    result = StreamResult()
    terminal_seen = False

    async for event in stream:
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

    if not terminal_seen and not result.text and not output_items:
        raise CodexStreamError("The Codex stream ended without a response.")

    return result


def _text_from_items(items: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in items:
        if item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "".join(parts).strip()
