"""Message and tool-payload sanitization helpers.

Pure functions extracted from ``run_agent.py`` so the AIAgent module can
stay focused on the conversation loop.  These walk OpenAI-format message
lists and structured payloads, repairing or stripping problematic
characters that would otherwise crash ``json.dumps`` inside the OpenAI
SDK or be rejected by upstream APIs.

All helpers are stateless and side-effect-free except for in-place
mutation of their input (where documented).  Backward-compatible
re-exports from ``run_agent`` remain in place so existing imports
``from run_agent import _sanitize_surrogates`` keep working.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Lone surrogate code points are invalid in UTF-8 and crash json.dumps
# inside the OpenAI SDK.  Used by every surrogate-sanitization helper
# below as well as by run_agent and the CLI for paste-from-clipboard
# scrubbing.
_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def _sanitize_surrogates(text: str) -> str:
    """Replace lone surrogate code points with U+FFFD (replacement character).

    Surrogates are invalid in UTF-8 and will crash ``json.dumps()`` inside the
    OpenAI SDK.  This is a fast no-op when the text contains no surrogates.
    """
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub('\ufffd', text)
    return text


def _sanitize_structure_surrogates(payload: Any) -> bool:
    """Replace surrogate code points in nested dict/list payloads in-place.

    Mirror of ``_sanitize_structure_non_ascii`` but for surrogate recovery.
    Used to scrub nested structured fields (e.g. ``reasoning_details`` — an
    array of dicts with ``summary``/``text`` strings) that flat per-field
    checks don't reach.  Returns True if any surrogates were replaced.
    """
    found = False

    def _walk(node):
        nonlocal found
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    if _SURROGATE_RE.search(value):
                        node[key] = _SURROGATE_RE.sub('\ufffd', value)
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                if isinstance(value, str):
                    if _SURROGATE_RE.search(value):
                        node[idx] = _SURROGATE_RE.sub('\ufffd', value)
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)

    _walk(payload)
    return found


def _sanitize_messages_surrogates(messages: list) -> bool:
    """Sanitize surrogate characters from all string content in a messages list.

    Walks message dicts in-place. Returns True if any surrogates were found
    and replaced, False otherwise. Covers content/text, name, tool call
    metadata/arguments, AND any additional string or nested structured fields
    (``reasoning``, ``reasoning_content``, ``reasoning_details``, etc.) so
    retries don't fail on a non-content field.  Byte-level reasoning models
    can emit lone surrogates in reasoning output
    that flow through to ``api_messages["reasoning_content"]`` on the next
    turn and crash json.dumps inside the OpenAI SDK.
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _SURROGATE_RE.search(content):
            msg["content"] = _SURROGATE_RE.sub('\ufffd', content)
            found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and _SURROGATE_RE.search(text):
                        part["text"] = _SURROGATE_RE.sub('\ufffd', text)
                        found = True
        name = msg.get("name")
        if isinstance(name, str) and _SURROGATE_RE.search(name):
            msg["name"] = _SURROGATE_RE.sub('\ufffd', name)
            found = True
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and _SURROGATE_RE.search(tc_id):
                    tc["id"] = _SURROGATE_RE.sub('\ufffd', tc_id)
                    found = True
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn_name = fn.get("name")
                    if isinstance(fn_name, str) and _SURROGATE_RE.search(fn_name):
                        fn["name"] = _SURROGATE_RE.sub('\ufffd', fn_name)
                        found = True
                    fn_args = fn.get("arguments")
                    if isinstance(fn_args, str) and _SURROGATE_RE.search(fn_args):
                        fn["arguments"] = _SURROGATE_RE.sub('\ufffd', fn_args)
                        found = True
        # Walk any additional string / nested fields (reasoning,
        # reasoning_content, reasoning_details, etc.) — surrogates from
        # byte-level reasoning models can lurk
        # in these fields and aren't covered by the per-field checks above.
        # Matches _sanitize_messages_non_ascii's coverage.
        for key, value in msg.items():
            if key in {"content", "name", "tool_calls", "role"}:
                continue
            if isinstance(value, str):
                if _SURROGATE_RE.search(value):
                    msg[key] = _SURROGATE_RE.sub('\ufffd', value)
                    found = True
            elif isinstance(value, (dict, list)):
                if _sanitize_structure_surrogates(value):
                    found = True
    return found


def _escape_invalid_chars_in_json_strings(raw: str) -> str:
    """Escape unescaped control chars inside JSON string values.

    Walks the raw JSON character-by-character, tracking whether we are
    inside a double-quoted string. Inside strings, replaces literal
    control characters (0x00-0x1F) that aren't already part of an escape
    sequence with their ``\\uXXXX`` equivalents. Pass-through for everything
    else.

    Ported from — complements the other repair passes in
    ``_repair_tool_call_arguments`` when ``json.loads(strict=False)`` is
    not enough (e.g. llama.cpp backends that emit literal apostrophes or
    tabs alongside other malformations).
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if in_string:
            if ch == "\\" and i + 1 < n:
                # Already-escaped char — pass through as-is
                out.append(ch)
                out.append(raw[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
            elif ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
        i += 1
    return "".join(out)


# When a repair is about to destroy the only copy of a tool call's original
# argument bytes (rewriting them to "{}"), the WARNING log is the last
# surviving copy of content that can hold real user data. Bound the
# logged string at this size instead of a short preview so it stays
# recoverable from agent.log without letting a pathological payload flood
# the log.
_FULL_ARGS_LOG_BOUND = 100_000


def _repair_tool_call_arguments(raw_args: str, tool_name: str = "?") -> str:
    """Attempt to repair malformed tool_call argument JSON.

    Models like GLM-5.1 via Ollama can produce truncated JSON, trailing
    commas, Python ``None``, etc.  The API proxy rejects these with HTTP 400
    "invalid tool call arguments".  This function applies common repairs;
    if all fail it returns ``"{}"`` so the request succeeds (better than
    crashing the session).  All repairs are logged at WARNING level.
    """
    raw_stripped = raw_args.strip() if isinstance(raw_args, str) else ""

    # Fast-path: empty / whitespace-only -> empty object
    if not raw_stripped:
        logger.warning("Sanitized empty tool_call arguments for %s", tool_name)
        return "{}"

    # Python-literal None -> normalise to {}
    if raw_stripped == "None":
        logger.warning("Sanitized Python-None tool_call arguments for %s", tool_name)
        return "{}"

    # Repair pass 0: llama.cpp backends sometimes emit literal control
    # characters (tabs, newlines) inside JSON string values. json.loads
    # with strict=False accepts these and lets us re-serialise the
    # result into wire-valid JSON without any string surgery. This is
    # the most common local-model repair case.
    try:
        parsed = json.loads(raw_stripped, strict=False)
        reserialised = json.dumps(parsed, separators=(",", ":"))
        if reserialised != raw_stripped:
            logger.warning(
                "Repaired unescaped control chars in tool_call arguments for %s",
                tool_name,
            )
        return reserialised
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Attempt common JSON repairs
    fixed = raw_stripped
    # 1. Strip trailing commas before } or ]
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    # 2. Close unclosed structures
    open_curly = fixed.count('{') - fixed.count('}')
    open_bracket = fixed.count('[') - fixed.count(']')
    if open_curly > 0:
        fixed += '}' * open_curly
    if open_bracket > 0:
        fixed += ']' * open_bracket
    # 3. Remove excess closing braces/brackets (bounded to 50 iterations)
    for _ in range(50):
        try:
            json.loads(fixed)
            break
        except json.JSONDecodeError:
            if fixed.endswith('}') and fixed.count('}') > fixed.count('{'):
                fixed = fixed[:-1]
            elif fixed.endswith(']') and fixed.count(']') > fixed.count('['):
                fixed = fixed[:-1]
            else:
                break

    try:
        json.loads(fixed)
        logger.warning(
            "Repaired malformed tool_call arguments for %s: %s → %s",
            tool_name, raw_stripped[:80], fixed[:80],
        )
        return fixed
    except json.JSONDecodeError:
        pass

    # Repair pass 4: escape unescaped control chars inside JSON strings,
    # then retry. Catches cases where strict=False alone fails because
    # other malformations are present too.
    try:
        escaped = _escape_invalid_chars_in_json_strings(fixed)
        if escaped != fixed:
            json.loads(escaped)
            logger.warning(
                "Repaired control-char-laced tool_call arguments for %s: %s → %s",
                tool_name, raw_stripped[:80], escaped[:80],
            )
            return escaped
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Last resort: replace with empty object so the API request doesn't
    # crash the entire session. Log the FULL original string (bounded) —
    # for callers that discard the original (e.g. the pre-send transcript
    # sanitizer), this WARNING is the last surviving copy of bytes that can
    # contain real user content (: a truncated write_file call's
    # streamed file content).
    logger.warning(
        "Unrepairable tool_call arguments for %s — "
        "replaced with empty object (was: %s)",
        tool_name, raw_stripped[:_FULL_ARGS_LOG_BOUND],
    )
    return "{}"


def close_interrupted_tool_sequence(messages: list, final_response: Any = None) -> bool:
    """Append a synthetic assistant turn when an interrupted tail is a tool result.

    A turn cut short by ``/stop`` can leave the transcript ending on a raw
    ``tool`` message (a tool finished, or its execution was cancelled, but the
    model never streamed a closing assistant turn). Persisting that tail means
    the next user message lands as ``… tool → user`` — a role-alternation
    violation that strict providers (Gemini, Claude) react to by hallucinating
    a continuation of the user's message and ignoring prior context, which
    reads to the user as "lost context".

    ``finalize_turn`` closes this on the happy interrupt path, but the
    retry/backoff/error interrupt aborts in ``conversation_loop`` ``return``
    early and never reach it — this shared helper closes the sequence on all of
    them. ``final_response`` is usually empty on an interrupt, so an explicit
    placeholder is used rather than an empty-content assistant turn.

    Mutates ``messages`` in place. Returns True if a closing turn was appended.
    """
    if not messages:
        return False
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "tool":
        return False
    text = final_response if isinstance(final_response, str) else ""
    from agent.message_metadata import append_message

    append_message(messages, {
        "role": "assistant",
        "content": text.strip() or "Operation interrupted.",
    })
    return True


def _strip_non_ascii(text: str) -> str:
    """Remove non-ASCII characters, replacing with closest ASCII equivalent or removing.

    Used as a last resort when the system encoding is ASCII and can't handle
    any non-ASCII characters (e.g. LANG=C on Chromebooks).
    """
    return text.encode('ascii', errors='ignore').decode('ascii')


def _sanitize_messages_non_ascii(messages: list) -> bool:
    """Strip non-ASCII characters from all string content in a messages list.

    This is a last-resort recovery for systems with ASCII-only encoding
    (LANG=C, Chromebooks, minimal containers).  Returns True if any
    non-ASCII content was found and sanitized.
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # Sanitize content (string)
        content = msg.get("content")
        if isinstance(content, str):
            sanitized = _strip_non_ascii(content)
            if sanitized != content:
                msg["content"] = sanitized
                found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        sanitized = _strip_non_ascii(text)
                        if sanitized != text:
                            part["text"] = sanitized
                            found = True
        # Sanitize name field (can contain non-ASCII in tool results)
        name = msg.get("name")
        if isinstance(name, str):
            sanitized = _strip_non_ascii(name)
            if sanitized != name:
                msg["name"] = sanitized
                found = True
        # Sanitize tool_calls
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if isinstance(fn, dict):
                        fn_args = fn.get("arguments")
                        if isinstance(fn_args, str):
                            sanitized = _strip_non_ascii(fn_args)
                            if sanitized != fn_args:
                                fn["arguments"] = sanitized
                                found = True
        # Sanitize any additional top-level string fields (e.g. reasoning_content)
        for key, value in msg.items():
            if key in {"content", "name", "tool_calls", "role"}:
                continue
            if isinstance(value, str):
                sanitized = _strip_non_ascii(value)
                if sanitized != value:
                    msg[key] = sanitized
                    found = True
    return found


def _sanitize_tools_non_ascii(tools: list) -> bool:
    """Strip non-ASCII characters from tool payloads in-place."""
    return _sanitize_structure_non_ascii(tools)


def _strip_images_from_messages(messages: list) -> bool:
    """Remove image_url content parts from all messages in-place.

    Called when a server signals it does not support images (e.g.
    "Only 'text' content type is supported.").  Mutates messages so the
    next API call sends text only.

    Preserves message alternation invariants:
      * ``tool``-role messages whose content was entirely images are replaced
        with a plaintext placeholder, NOT deleted — deleting them would leave
        the paired ``tool_call_id`` on the prior assistant message unmatched,
        which providers reject with HTTP 400.
      * Non-tool messages whose content becomes empty are dropped.  In
        practice this only hits synthetic image-only user messages appended
        for attachment delivery; real user turns always include text.

    Returns True if any image parts were removed.
    """
    found = False
    to_delete = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "image", "input_image"}:
                found = True
            else:
                new_parts.append(part)
        if len(new_parts) < len(content):
            if new_parts:
                msg["content"] = new_parts
            elif msg.get("role") == "tool":
                # Preserve tool_call_id linkage — providers require every
                # assistant tool_call to have a matching tool response.
                msg["content"] = "[image content removed — server does not support images]"
            else:
                # Synthetic image-only user/assistant message with no text;
                # safe to drop.
                to_delete.append(i)
    for i in reversed(to_delete):
        del messages[i]
    return found


def _sanitize_structure_non_ascii(payload: Any) -> bool:
    """Strip non-ASCII characters from nested dict/list payloads in-place."""
    found = False

    def _walk(node):
        nonlocal found
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    sanitized = _strip_non_ascii(value)
                    if sanitized != value:
                        node[key] = sanitized
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                if isinstance(value, str):
                    sanitized = _strip_non_ascii(value)
                    if sanitized != value:
                        node[idx] = sanitized
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)

    _walk(payload)
    return found


__all__ = [
    "_SURROGATE_RE",
    "close_interrupted_tool_sequence",
    "_sanitize_surrogates",
    "_sanitize_structure_surrogates",
    "_sanitize_messages_surrogates",
    "_escape_invalid_chars_in_json_strings",
    "_repair_tool_call_arguments",
    "_strip_non_ascii",
    "_sanitize_messages_non_ascii",
    "_sanitize_tools_non_ascii",
    "_strip_images_from_messages",
    "_sanitize_structure_non_ascii",
    # call_id policy owners (F4 consolidation)
    "deterministic_call_id",
    "coalesce_tool_call_id",
    "uniquify_tool_call_ids",
]


# ---------------------------------------------------------------------------
# call_id policy — single owner (audit F4, incident chain I4)
# ---------------------------------------------------------------------------
#
# Three forked policy sites converged here:
#   * agent/codex_responses_adapter.py `_deterministic_call_id` — hash
#     synthesis when a provider omits call_id (fa3ab2ffd0 → e45f2b39e2).
#   * run_agent.AIAgent._get_tool_call_id_static — `call_id or id`
#     coalescing for dicts and SDK objects.
#   * run_agent.AIAgent._uniquify_tool_call_ids — duplicate-id repair with
# deterministic `_d<n>` suffixes loss class).
#
# NOT consolidated (different scheme on purpose):
#   agent/transports/codex_event_projector._deterministic_call_id maps codex
#   app-server ITEM ids (`codex_<type>_<item_id>`), not chat tool-call
#   content; merging the two would change ids and invalidate prompt caches.
#
# HARD INVARIANT: everything here must stay deterministic (never uuid4) and
# byte-identical for existing inputs — these ids feed prompt-cache prefixes.


def deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """Generate a deterministic call_id from tool call content.

    Used as a fallback when the API doesn't provide a call_id.
    Deterministic IDs prevent cache invalidation — random UUIDs would
    make every API call's prefix unique, breaking OpenAI's prompt cache.
    """
    seed = f"{fn_name}:{arguments}:{index}"
    digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"call_{digest}"


def coalesce_tool_call_id(tc: Any) -> str:
    """Extract the effective call ID from a tool_call entry (dict or object).

    Single owner for the ``call_id or id`` coalescing rule: Codex Responses
    tool calls carry ``call_id`` (authoritative pairing key), Chat
    Completions ones carry ``id`` only. Returns ``""`` when neither is set.
    """
    if isinstance(tc, dict):
        return (tc.get("call_id", "") or tc.get("id", "") or "").strip()
    return (getattr(tc, "call_id", "") or getattr(tc, "id", "") or "").strip()


def uniquify_tool_call_ids(tool_calls: list) -> list:
    """Ensure every tool call in a single assistant turn has a distinct id.

    Some models/providers reuse one call id across different calls in a
    single batch (observed with OpenAI-compatible endpoints and degraded models
    at long context). Duplicate ids are lossy downstream: the pre-API
    sanitizer keeps only the first call/result pair per id, so the later
    call's result silently vanishes from every replayed payload, and
    strict providers reject duplicate ids outright.

    The first occurrence keeps its id; later collisions get a
    deterministic ``<id>_d<n>`` suffix — never a random UUID, which would
    break prompt-cache prefix stability across replays. Mutates the
    entries in place (SDK models / SimpleNamespace / dicts) and returns
    the same list. Blank/missing ids are left for the deterministic
    fallback in ``build_assistant_message``.
    """
    seen: set = set()
    for tc in tool_calls or []:
        # Same coalescing rule as ``coalesce_tool_call_id`` but tolerant of
        # non-string ids (degraded models can emit ints/None here).
        if isinstance(tc, dict):
            raw = tc.get("call_id") or tc.get("id") or ""
        else:
            raw = getattr(tc, "call_id", None) or getattr(tc, "id", None) or ""
        raw = raw.strip() if isinstance(raw, str) else ""
        if not raw:
            continue
        # Composite Responses ids ("call_x|fc_y") collide on the call
        # half — that's the pairing key providers enforce per turn.
        cid = raw.split("|", 1)[0]
        if not cid:
            continue
        if cid not in seen:
            seen.add(cid)
            continue
        n = 2
        new_id = f"{cid}_d{n}"
        while new_id in seen:
            n += 1
            new_id = f"{cid}_d{n}"
        seen.add(new_id)

        def _renamed(value):
            # Preserve a composite id's response-item half so the
            # provider's real fc_/item id survives the rename.
            if isinstance(value, str) and "|" in value:
                return f"{new_id}|{value.split('|', 1)[1]}"
            return new_id

        try:
            if isinstance(tc, dict):
                if tc.get("id"):
                    tc["id"] = _renamed(tc["id"])
                else:
                    tc["id"] = new_id
                if tc.get("call_id"):
                    tc["call_id"] = new_id
            else:
                tc.id = _renamed(getattr(tc, "id", None))
                if getattr(tc, "call_id", None):
                    tc.call_id = new_id
        except Exception:
            logger.warning(
                "Could not uniquify duplicate tool call id %s", cid
            )
            continue
        _fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
        _fn_name = (_fn.get("name") if isinstance(_fn, dict) else getattr(_fn, "name", None)) or "?"
        logger.warning(
            "Model reused tool call id %s within one turn; renamed the "
            "duplicate to %s (tool=%s) to keep call/result pairing "
            "lossless.", cid, new_id, _fn_name,
        )
    return tool_calls


# Image / multimodal parts — evaluated, NOT consolidated (verdict: syntax)
# ---------------------------------------------------------------------------
#
# The per-adapter image handling is format-specific SYNTAX, not shared policy:
#   * codex_responses_adapter (~113/165/812): chat `image_url` parts →
#     Responses `input_image` items and image counting for log summaries —
#     Responses wire shape only.
#   * transports/chat_completions: pass-through (native format).
# The one genuinely shared image POLICY — removing images when a server
# rejects them while preserving tool_call_id pairing — already has a single
# owner here: ``_strip_images_from_messages`` above.
