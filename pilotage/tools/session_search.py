"""Long-term conversation recall.

This is the production-used Hermes session_search behavior adapted to
Genesis' smaller conversation store. It keeps Hermes' FTS5 retrieval and four
calling shapes (discover, scroll, read, browse), but not Hermes' broader
session lineage, generated titles, UI links, or cross-profile lookup.

The store is supplied explicitly by the active Agent, so a profile can search
only its own durable history. Profiles serving several unrelated users can
disable this tool group, as the FLConnect operations profile does in production.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..history import ConversationSearchError
from .ansi_strip import strip_ansi
from .registry import Tool, ToolContext, tool_error

MAX_FTS5_QUERY_CHARS = 2_048
DISCOVER_SCAN_LIMIT = 300
MESSAGE_CONTENT_LIMIT = 4_000
BOOKEND_CONTENT_LIMIT = 1_200
PREVIEW_LIMIT = 240
TRUNCATION_MARKER = "… [truncated]"

# Ported from Hermes' FTS sanitizer. These characters are rejected by FTS5's
# query grammar when they occur outside a balanced quoted phrase.
_FTS5_SPECIAL_CHARS = '+{}():"^@/#&|~[]<>,;!?$=\\\''
_FTS5_SPECIAL_RE = re.compile(f"[{re.escape(_FTS5_SPECIAL_CHARS)}]")
_VALID_ROLES = frozenset({"user", "assistant"})


def _contains_cjk(text: str) -> bool:
    for char in text:
        codepoint = ord(char)
        if (
            0x4E00 <= codepoint <= 0x9FFF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x20000 <= codepoint <= 0x2A6DF
            or 0x3000 <= codepoint <= 0x303F
            or 0x3040 <= codepoint <= 0x309F
            or 0x30A0 <= codepoint <= 0x30FF
            or 0xAC00 <= codepoint <= 0xD7AF
        ):
            return True
    return False


def sanitize_fts5_query(query: str) -> str:
    """Make model-supplied search text safe for SQLite FTS5 MATCH."""
    query = query[:MAX_FTS5_QUERY_CHARS]

    quoted_parts: List[str] = []
    pieces: List[str] = []
    index = 0
    while index < len(query):
        if query[index] != '"':
            pieces.append(query[index])
            index += 1
            continue
        end = query.find('"', index + 1)
        if end == -1:
            pieces.append(" ")
            index += 1
            continue
        quoted_parts.append(query[index : end + 1])
        pieces.append(f"\x00Q{len(quoted_parts) - 1}\x00")
        index = end + 1

    sanitized = _FTS5_SPECIAL_RE.sub(" ", "".join(pieces))
    if "%" in sanitized and not _contains_cjk(sanitized):
        sanitized = sanitized.replace("%", " ")
    sanitized = re.sub(r"\*+", "*", sanitized)
    sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)
    sanitized = re.sub(
        r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip()
    )
    sanitized = re.sub(
        r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip()
    )
    sanitized = re.sub(
        r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized
    )
    for index, quoted in enumerate(quoted_parts):
        sanitized = sanitized.replace(f"\x00Q{index}\x00", quoted)
    return sanitized.strip()


def _timestamp(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value)).astimezone().isoformat(
            timespec="seconds"
        )
    except (OSError, OverflowError, TypeError, ValueError):
        return "unknown"


def _cap(text: Any, limit: int) -> str:
    # Stored history predates current input/output sanitizers and may contain
    # terminal escapes. Strip before truncation so hidden bytes cannot consume
    # the visible budget or re-enter model context through recall.
    value = strip_ansi(str(text or ""))
    if len(value) <= limit:
        return value
    keep = max(0, limit - len(TRUNCATION_MARKER))
    return value[:keep] + TRUNCATION_MARKER


def _preview(text: Any) -> str:
    return _cap(" ".join(str(text or "").split()), PREVIEW_LIMIT)


def _shape_message(
    message: Dict[str, Any],
    *,
    anchor_id: Optional[int] = None,
    content_limit: int = MESSAGE_CONTENT_LIMIT,
) -> Dict[str, Any]:
    shaped = {
        "id": int(message["id"]),
        "role": str(message["role"]),
        "content": _cap(message.get("content"), content_limit),
        "timestamp": _timestamp(message.get("written_at")),
    }
    if anchor_id is not None and shaped["id"] == anchor_id:
        shaped["match"] = True
    return shaped


def _integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _roles(value: Any) -> Tuple[str, ...]:
    if value is None or not str(value).strip():
        return ("user", "assistant")
    requested = tuple(
        part.strip().lower()
        for part in str(value).split(",")
        if part.strip()
    )
    unknown = sorted(set(requested) - _VALID_ROLES)
    if unknown:
        raise ValueError(
            "role_filter contains unsupported roles: " + ", ".join(unknown)
        )
    if not requested:
        raise ValueError("role_filter must include user or assistant")
    return requested


def _browse(store: Any, current_chat_id: str, limit: int) -> str:
    sessions = store.recent_sessions(
        current_chat_id=current_chat_id,
        limit=limit,
    )
    results = [
        {
            "session_id": item["session_id"],
            "started_at": _timestamp(item["started_at"]),
            "last_active": _timestamp(item["last_active"]),
            "message_count": item["message_count"],
            "preview": _preview(item["preview"]),
        }
        for item in sessions
    ]
    return json.dumps(
        {
            "success": True,
            "mode": "browse",
            "results": results,
            "count": len(results),
            "message": (
                f"Showing {len(results)} most recent sessions. "
                "Pass query to search, or session_id to read one."
            ),
        },
        ensure_ascii=False,
    )


def _read(store: Any, session_id: str) -> str:
    session = store.read_session(session_id)
    if session is None:
        return tool_error(
            f"session_id not found: {session_id}",
            success=False,
        )
    messages = [
        _shape_message(message) for message in session["messages"]
    ]
    total = len(messages)
    truncated = total > 30
    visible = messages[:20] + messages[-10:] if truncated else messages
    response: Dict[str, Any] = {
        "success": True,
        "mode": "read",
        "session_id": session["session_id"],
        "started_at": _timestamp(session["started_at"]),
        "last_active": _timestamp(session["last_active"]),
        "message_count": total,
        "truncated": truncated,
        "messages": visible,
    }
    if truncated:
        response["message"] = (
            f"Session has {total} messages; showing first 20 and last 10. "
            "Pass session_id with around_message_id to scroll the middle."
        )
    return json.dumps(response, ensure_ascii=False)


def _scroll(
    store: Any,
    session_id: str,
    around_message_id: Any,
    window: int,
) -> str:
    try:
        anchor = int(around_message_id)
    except (TypeError, ValueError):
        return tool_error(
            "scroll requires integer around_message_id",
            success=False,
        )
    view = store.anchored_view(
        session_id,
        anchor,
        window=window,
    )
    if view is None:
        return tool_error(
            f"around_message_id {anchor} not found in session_id {session_id}",
            success=False,
        )
    return json.dumps(
        {
            "success": True,
            "mode": "scroll",
            "session_id": view["session_id"],
            "around_message_id": anchor,
            "window": window,
            "messages": [
                _shape_message(message, anchor_id=anchor)
                for message in view["messages"]
            ],
            "messages_before": view["messages_before"],
            "messages_after": view["messages_after"],
        },
        ensure_ascii=False,
    )


def _discover(
    store: Any,
    current_chat_id: str,
    query: str,
    roles: Sequence[str],
    limit: int,
    sort: Optional[str],
    detail: str,
) -> str:
    sanitized = sanitize_fts5_query(query)
    if not sanitized:
        raw_results: List[Dict[str, Any]] = []
    else:
        raw_results = store.search_messages(
            sanitized,
            current_chat_id=current_chat_id,
            roles=roles,
            scan_limit=DISCOVER_SCAN_LIMIT,
            sort=sort,
        )

    seen: set[str] = set()
    results: List[Dict[str, Any]] = []
    for hit in raw_results:
        session_id = str(hit["session_id"])
        if session_id in seen:
            continue
        seen.add(session_id)
        result_detail = "full" if detail == "full" or not results else "compact"
        anchor = int(hit["id"])
        view = store.anchored_view(
            session_id,
            anchor,
            window=5,
            bookend=3,
        )
        if view is None:
            continue

        if result_detail == "full":
            messages = [
                _shape_message(message, anchor_id=anchor)
                for message in view["messages"]
            ]
            bookend_start = [
                _shape_message(
                    message,
                    content_limit=BOOKEND_CONTENT_LIMIT,
                )
                for message in view["bookend_start"]
            ]
            bookend_end = [
                _shape_message(
                    message,
                    content_limit=BOOKEND_CONTENT_LIMIT,
                )
                for message in view["bookend_end"]
            ]
        else:
            messages = [
                _shape_message(hit, anchor_id=anchor)
            ]
            bookend_start = []
            bookend_end = []

        results.append(
            {
                "session_id": view["session_id"],
                "when": _timestamp(hit["written_at"]),
                "matched_role": hit["role"],
                "match_message_id": anchor,
                "snippet": _cap(hit["snippet"], MESSAGE_CONTENT_LIMIT),
                "detail": result_detail,
                "bookend_start": bookend_start,
                "messages": messages,
                "bookend_end": bookend_end,
                "messages_before": view["messages_before"],
                "messages_after": view["messages_after"],
            }
        )
        if len(results) >= limit:
            break

    return json.dumps(
        {
            "success": True,
            "mode": "discover",
            "query": query,
            "detail": detail,
            "results": results,
            "count": len(results),
            "message": (
                "No matching sessions found."
                if not results
                else f"Found {len(results)} matching sessions."
            ),
        },
        ensure_ascii=False,
    )


def _run(args: Dict[str, Any], store: Any, current_chat_id: str) -> str:
    session_id_value = args.get("session_id")
    session_id = (
        str(session_id_value).strip()
        if session_id_value is not None
        else ""
    )
    around_message_id = args.get("around_message_id")
    window = _integer(args.get("window"), 5, 1, 20)

    if around_message_id is not None and not session_id:
        return tool_error(
            "around_message_id requires session_id",
            success=False,
        )
    if session_id and around_message_id is not None:
        return _scroll(store, session_id, around_message_id, window)
    if session_id:
        return _read(store, session_id)

    query_value = args.get("query")
    query = str(query_value or "").strip()
    limit = _integer(args.get("limit"), 3, 1, 10)
    if not query:
        return _browse(store, current_chat_id, limit)

    sort_value = args.get("sort")
    sort = str(sort_value).strip().lower() if sort_value is not None else None
    if sort not in {None, "", "newest", "oldest"}:
        return tool_error(
            "sort must be newest or oldest",
            success=False,
        )
    detail_value = str(args.get("detail") or "adaptive").strip().lower()
    if detail_value not in {"adaptive", "full"}:
        return tool_error(
            "detail must be adaptive or full",
            success=False,
        )
    try:
        roles = _roles(args.get("role_filter"))
    except ValueError as exc:
        return tool_error(str(exc), success=False)
    return _discover(
        store,
        current_chat_id,
        query,
        roles,
        limit,
        sort or None,
        detail_value,
    )


async def handle(args: Dict[str, Any], context: ToolContext) -> str:
    store = context.conversation_store
    if store is None:
        return tool_error(
            "Conversation history is unavailable",
            success=False,
        )
    try:
        return await asyncio.to_thread(
            _run,
            args,
            store,
            context.chat_id,
        )
    except ConversationSearchError as exc:
        return tool_error(str(exc), success=False)


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search past conversations stored in this profile's local SQLite "
        "history, or read and scroll inside one. FTS5-backed; makes no LLM "
        "calls and returns actual stored messages. Use this as historical "
        "context, not as proof of a current external source. Four shapes: "
        "(1) pass query to discover matching sessions; adaptive detail fully "
        "hydrates the top result and keeps later results compact, while "
        "detail='full' hydrates all. (2) pass session_id and "
        "around_message_id to scroll around an anchor. (3) pass session_id "
        "alone to read it, bounded to first 20 and last 10 messages when "
        "large. (4) pass no arguments to browse recent sessions. Discovery "
        "uses FTS5 syntax: terms are ANDed by default; use OR, quoted phrases, "
        "NOT, or a suffix wildcard for broader or exact recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords, phrase, or FTS5 expression to find.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum sessions to return (default 3, max 10).",
                "default": 3,
            },
            "sort": {
                "type": "string",
                "enum": ["newest", "oldest"],
                "description": "Optional temporal ordering instead of pure relevance.",
            },
            "detail": {
                "type": "string",
                "enum": ["adaptive", "full"],
                "description": "Discovery hydration; adaptive is the default.",
                "default": "adaptive",
            },
            "session_id": {
                "type": "string",
                "description": "Session handle returned by discovery or browse.",
            },
            "around_message_id": {
                "type": "integer",
                "description": "Message id to center a scroll window on.",
            },
            "window": {
                "type": "integer",
                "description": "Messages on each side of the anchor, clamped 1–20.",
                "default": 5,
            },
            "role_filter": {
                "type": "string",
                "description": "Comma-separated user and/or assistant roles.",
            },
        },
        "required": [],
    },
}

SESSION_SEARCH_TOOL = Tool(
    name="session_search",
    group="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=handle,
    emoji="🔍",
)
