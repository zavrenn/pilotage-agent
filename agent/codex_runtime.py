"""Codex API runtime — Responses-API streaming paths.

Extracted from :class:`AIAgent` to keep the agent loop file focused.
Each function takes the parent ``AIAgent`` as its first argument
(``agent``).  AIAgent keeps thin forwarder methods for backward
compatibility.

* ``run_codex_stream`` — streams a Codex Responses API call (the
  ``codex_responses`` api_mode).
* ``run_codex_create_stream_fallback`` — recovery path when the
  Responses ``stream=True`` initial create fails.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any, List

from agent.stream_single_writer import claim_stream_writer, stream_writer_is_current

logger = logging.getLogger(__name__)


def _codex_request_failure_details(error: BaseException) -> tuple[int | None, str]:
    """Return the serialized request size and exception class chain.

    OpenAI connection exceptions retain the final ``httpx.Request``. Reading
    its already-buffered content gives us the exact byte count handed to the
    transport without logging any request content. The class-only chain keeps
    the underlying transport failure visible without exposing URLs or payloads
    from exception messages.
    """
    request_body_bytes: int | None = None
    exception_classes: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        exception_classes.append(type(current).__name__)

        if request_body_bytes is None:
            try:
                request = getattr(current, "request", None)
            except Exception:
                request = None
            if request is not None:
                try:
                    content = request.content
                except Exception:
                    content = None
                if isinstance(content, str):
                    request_body_bytes = len(content.encode("utf-8"))
                elif isinstance(content, (bytes, bytearray, memoryview)):
                    request_body_bytes = len(content)

        cause = current.__cause__
        if cause is None and not current.__suppress_context__:
            cause = current.__context__
        current = cause

    return request_body_bytes, " <- ".join(exception_classes)


def _log_codex_request_failure(
    agent: Any,
    error: BaseException,
    *,
    stream_opened: bool,
) -> None:
    request_body_bytes, exception_chain = _codex_request_failure_details(error)
    logger.warning(
        "Codex Responses request failed: "
        "serialized_request_body_bytes=%s stream_opened=%s "
        "exception_chain=%s model=%s",
        request_body_bytes if request_body_bytes is not None else "unknown",
        str(stream_opened).lower(),
        exception_chain,
        getattr(agent, "model", "unknown"),
    )


# ---------------------------------------------------------------------------
# Event-driven Responses streaming
#
# OpenAI ships its consumer Codex backend (chatgpt.com/backend-api/codex) on
# a different schedule from the openai Python SDK.  The high-level
# ``client.responses.stream(...)`` helper reconstructs a typed Response from
# the terminal ``response.completed`` event's ``response.output`` field, and
# when that field drifts to ``null`` (gpt-5.5, May 2026) the SDK raises
# ``TypeError: 'NoneType' object is not iterable`` mid-iteration.
#
# We sidestep the whole class of failure by going one level lower:
# ``client.responses.create(stream=True)`` returns the raw AsyncIterable of
# SSE events, and we assemble the final response object purely from
# ``response.output_item.done`` events as they arrive.  We never read
# ``response.completed.response.output`` for content reconstruction, so the
# backend can return ``null``, ``[]``, a string, or omit the field entirely
# and we don't care.
#
# This mirrors what the OpenClaw TS implementation does for the same backend
# and is structurally immune to the bug class rather than patched.
# ---------------------------------------------------------------------------


_TERMINAL_EVENT_TYPES = frozenset({
    "response.completed",
    "response.incomplete",
    "response.failed",
})


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    """Field access that handles both attr-style (SDK objects) and dict (raw JSON) events."""
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        value = event.get(name, default)
    return value if value is not None else default


def _item_field(item: Any, name: str, default: Any = None) -> Any:
    """Field access for nested Response items (attr-style SDK object or dict)."""
    value = getattr(item, name, None)
    if value is None and isinstance(item, dict):
        value = item.get(name, default)
    return value if value is not None else default


def _raise_stream_error(event: Any) -> None:
    """Raise a ``_StreamErrorEvent`` from a ``type=error`` SSE frame.

    The Responses spec puts the failure details at the top level of the
    frame (``{"type": "error", "code": ..., "message": ..., "param": ...}``),
    but the official OpenAI SDK and several OpenAI-compatible proxies wrap
    them in an HTTP-style nested envelope instead
    (``{"type": "error", "error": {"code": ..., "message": ..., "param": ...}}``).
    Read the top-level fields first, then fall back to the nested envelope so
    the error classifier sees the provider's real code/message (rate-limit vs
    context-overflow vs entitlement) rather than the generic placeholder.
    Port of

    Imported lazily so this module stays importable from places that don't
    pull in ``run_agent`` (e.g. plugin code, doc tools).
    """
    from run_agent import _StreamErrorEvent

    nested = _event_field(event, "error")

    def _error_field(name: str) -> Any:
        value = _event_field(event, name)
        if value is None and nested is not None:
            value = _item_field(nested, name)
        return value

    raw_message = _error_field("message")
    if raw_message is not None and not isinstance(raw_message, str):
        raw_message = str(raw_message)
    message = (raw_message or "stream emitted error event").strip() or "stream emitted error event"
    raise _StreamErrorEvent(
        message,
        code=_error_field("code"),
        param=_error_field("param"),
    )


def _consume_codex_event_stream(
    event_iter: Any,
    *,
    model: str,
    on_text_delta=None,
    on_reasoning_delta=None,
    on_commentary_message=None,
    on_first_delta=None,
    on_event=None,
    interrupt_check=None,
) -> SimpleNamespace:
    """Consume a Codex Responses SSE event stream and return a final response.

    The returned object is a ``SimpleNamespace`` shaped like the SDK's typed
    ``Response`` for the fields downstream code actually reads:

    * ``output``: list of output items, assembled from ``response.output_item.done``.
      For tool-call turns this contains the function_call items; for plain-text
      turns it contains a synthesized ``message`` item built from streamed deltas
      if no message item was emitted directly.
    * ``output_text``: assembled text from ``response.output_text.delta`` deltas.
    * ``usage``: copied from the terminal event's ``response.usage`` (when present).
    * ``status``: ``completed`` / ``incomplete`` / ``failed`` (or ``completed`` if
      the stream ended without a terminal frame but produced content).
    * ``id``: ``response.id`` when present.
    * ``incomplete_details``: passed through for ``response.incomplete`` frames.
    * ``error``: passed through for ``response.failed`` frames.
    * ``model``: from kwargs (the wire model name is not authoritative).

    Critically, we never read ``response.output`` from the terminal event for
    content reconstruction — only ``usage``, ``status``, ``id``.  That field
    being ``null`` / ``[]`` / missing is fine.

    Callbacks:

    * ``on_text_delta(str)`` — fires per ``response.output_text.delta``, suppressed
      once a function_call event is seen (so tool-call turns don't bleed text
      into the chat).
    * ``on_reasoning_delta(str)`` — fires per ``response.reasoning.*.delta`` and
      ``phase=analysis`` message deltas. When no dedicated commentary callback
      is supplied, commentary also uses this legacy fallback.
    * ``on_commentary_message(str)`` — fires once per completed
      ``phase=commentary`` message, before any following tool item executes.
    * ``on_first_delta()`` — one-shot, fires on the first text delta only.
    * ``on_event(event)`` — fires for every event before any other processing.
      Used for watchdog activity, debug logging, anything wire-shape-agnostic.
    * ``interrupt_check()`` — returns True to break the loop early.
    """
    collected_output_items: List[Any] = []
    collected_text_deltas: List[str] = []
    has_tool_calls = False
    first_delta_fired = False
    active_message_phase: str | None = None
    commentary_text_deltas: List[str] = []
    # Last reasoning summary_index seen. The Responses stream delimits summary
    # parts by this index and gives each part no separator of its own, so a
    # change of index is where the blank line belongs.
    active_summary_index: Any = None
    terminal_status: str = "completed"
    terminal_usage: Any = None
    terminal_response_id: str = None
    terminal_incomplete_details: Any = None
    terminal_error: Any = None
    saw_terminal = False

    for event in event_iter:
        if on_event is not None:
            try:
                on_event(event)
            except (TimeoutError, InterruptedError):
                # Control-flow signals from watchdog/cancellation hooks must
                # propagate, not get swallowed as "debug noise".
                raise
            except Exception:
                # Genuine bugs in third-party debug/log hooks shouldn't break
                # stream consumption.
                logger.debug("Codex stream on_event hook raised", exc_info=True)
        if interrupt_check is not None and interrupt_check():
            break

        event_type = _event_field(event, "type", "")
        if not isinstance(event_type, str):
            event_type = ""

        # ``error`` SSE frames carry the provider's real failure reason
        # (subscription / quota / model-not-available / rejected-reasoning-replay)
        # but never appear in the terminal set.  Surface them as a structured
        # exception so the credential pool + error classifier see the body.
        if event_type == "error":
            _raise_stream_error(event)

        # Track the phase of the active streamed message item.  Codex/Harmony
        # ``commentary``/``analysis`` text is mid-turn preamble/progress
        # narration, never the final answer.  We still collect completed output
        # items for replay, but route those deltas to the reasoning callback so
        # they display like thinking text instead of assistant content.
        if event_type == "response.output_item.added":
            item = _event_field(event, "item")
            item_type = _item_field(item, "type", "")
            if item_type == "message":
                phase = _item_field(item, "phase", None)
                active_message_phase = phase.strip().lower() if isinstance(phase, str) else None
                if active_message_phase == "commentary":
                    commentary_text_deltas = []
            else:
                active_message_phase = None
            if "function_call" in str(item_type):
                has_tool_calls = True
            continue

        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
            delta_text = _event_field(event, "delta", "")
            if delta_text and active_message_phase == "commentary":
                commentary_text_deltas.append(delta_text)
                # Preserve CLI/backward compatibility when no first-class
                # commentary consumer is installed.
                if on_commentary_message is None and on_reasoning_delta is not None:
                    try:
                        on_reasoning_delta(delta_text)
                    except Exception:
                        logger.debug("Codex stream on_reasoning_delta raised", exc_info=True)
            elif delta_text and active_message_phase == "analysis":
                if on_reasoning_delta is not None:
                    try:
                        on_reasoning_delta(delta_text)
                    except Exception:
                        logger.debug("Codex stream on_reasoning_delta raised", exc_info=True)
            elif delta_text:
                collected_text_deltas.append(delta_text)
                if not has_tool_calls:
                    if not first_delta_fired:
                        first_delta_fired = True
                        if on_first_delta is not None:
                            try:
                                on_first_delta()
                            except Exception:
                                logger.debug("Codex stream on_first_delta raised", exc_info=True)
                    if on_text_delta is not None:
                        try:
                            on_text_delta(delta_text)
                        except Exception:
                            logger.debug("Codex stream on_text_delta raised", exc_info=True)
            continue

        if "function_call" in event_type:
            has_tool_calls = True
            # fall through — function_call items still get added on output_item.done

        if "reasoning" in event_type and "delta" in event_type:
            reasoning_text = _event_field(event, "delta", "")
            if reasoning_text and on_reasoning_delta is not None:
                # Summary parts stream one after another with no separator of
                # their own; summary_index is the boundary the wire gives us.
                summary_index = _event_field(event, "summary_index")
                if (
                    summary_index is not None
                    and active_summary_index is not None
                    and summary_index != active_summary_index
                ):
                    reasoning_text = f"\n\n{reasoning_text}"
                if summary_index is not None:
                    active_summary_index = summary_index
                try:
                    on_reasoning_delta(reasoning_text)
                except Exception:
                    logger.debug("Codex stream on_reasoning_delta raised", exc_info=True)
            continue

        if event_type == "response.output_item.done":
            done_item = _event_field(event, "item")
            if done_item is not None:
                collected_output_items.append(done_item)
                done_phase = _item_field(done_item, "phase", None)
                done_phase = done_phase.strip().lower() if isinstance(done_phase, str) else None
                if done_phase == "commentary" and on_commentary_message is not None:
                    commentary_text = "".join(commentary_text_deltas).strip()
                    if not commentary_text:
                        content_parts = _item_field(done_item, "content", [])
                        if isinstance(content_parts, list):
                            commentary_text = "".join(
                                str(_item_field(part, "text", "") or "")
                                for part in content_parts
                                if _item_field(part, "type", "") == "output_text"
                            ).strip()
                    if commentary_text:
                        try:
                            on_commentary_message(commentary_text)
                        except Exception:
                            logger.debug(
                                "Codex stream on_commentary_message raised",
                                exc_info=True,
                            )
                    commentary_text_deltas = []
            continue

        if event_type in _TERMINAL_EVENT_TYPES:
            saw_terminal = True
            resp_obj = _event_field(event, "response")
            if resp_obj is not None:
                terminal_usage = getattr(resp_obj, "usage", None)
                if terminal_usage is None and isinstance(resp_obj, dict):
                    terminal_usage = resp_obj.get("usage")
                rid = getattr(resp_obj, "id", None)
                if rid is None and isinstance(resp_obj, dict):
                    rid = resp_obj.get("id")
                terminal_response_id = rid
                rstatus = getattr(resp_obj, "status", None)
                if rstatus is None and isinstance(resp_obj, dict):
                    rstatus = resp_obj.get("status")
                if isinstance(rstatus, str):
                    terminal_status = rstatus
                if event_type == "response.incomplete":
                    terminal_incomplete_details = getattr(resp_obj, "incomplete_details", None)
                    if terminal_incomplete_details is None and isinstance(resp_obj, dict):
                        terminal_incomplete_details = resp_obj.get("incomplete_details")
                if event_type == "response.failed":
                    terminal_error = getattr(resp_obj, "error", None)
                    if terminal_error is None and isinstance(resp_obj, dict):
                        terminal_error = resp_obj.get("error")
            if event_type == "response.completed":
                terminal_status = terminal_status or "completed"
            elif event_type == "response.incomplete":
                terminal_status = terminal_status or "incomplete"
            elif event_type == "response.failed":
                terminal_status = terminal_status or "failed"
            # Stop on terminal event.
            break

    # Build the final output list.  Prefer items observed via output_item.done;
    # if none arrived but we streamed plain text deltas (no tool calls), synthesize
    # a single message item so downstream normalization has something to work with.
    if collected_output_items:
        output = list(collected_output_items)
    elif collected_text_deltas and not has_tool_calls:
        assembled = "".join(collected_text_deltas)
        output = [SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text=assembled)],
        )]
    else:
        output = []

    # If the stream ended without any terminal event AND produced no usable
    # content (no items, no text deltas), surface that as a RuntimeError so
    # callers can distinguish "stream truncated mid-flight / provider rejected
    # the call" from "stream completed with empty body".  This preserves the
    # signal the SDK's high-level helper used to raise as
    # ``RuntimeError("Didn't receive a `response.completed` event.")``.
    if not saw_terminal and not output:
        raise RuntimeError(
            "Codex Responses stream did not emit a terminal response"
        )

    assembled_text = "".join(collected_text_deltas)

    final = SimpleNamespace(
        output=output,
        output_text=assembled_text,
        usage=terminal_usage,
        status=terminal_status,
        id=terminal_response_id,
        model=model,
        incomplete_details=terminal_incomplete_details,
        error=terminal_error,
    )
    return final


def run_codex_stream(agent, api_kwargs: dict, client: Any = None, on_first_delta=None):
    """Execute one streaming Responses API request and return the final response.

    Uses ``responses.create(stream=True)`` (low-level raw event iteration)
    rather than the high-level ``responses.stream(...)`` helper.  This makes
    us structurally immune to backend drift in the ``response.completed``
    payload shape — we never let the SDK reconstruct a typed object from
    the terminal event's ``output`` field.
    """
    import httpx as _httpx
    from openai import APIConnectionError as _APIConnectionError

    from agent import relay_llm

    active_client = client or agent._ensure_primary_openai_client(reason="codex_stream_direct")
    max_stream_retries = 1
    # Accumulate streamed text so callers / compat shims can read it.
    agent._codex_streamed_text_parts: list = []

    def _on_text_delta(text: str) -> None:
        agent._codex_streamed_text_parts.append(text)
        agent._fire_stream_delta(text)

    def _on_reasoning_delta(text: str) -> None:
        agent._fire_reasoning_delta(text)

    def _on_commentary_message(text: str) -> None:
        agent._fire_streamed_codex_commentary(text)

    def _on_event(event: Any) -> None:
        # TTFB watchdog and activity touch — runs once per SSE event.
        agent._codex_stream_last_event_ts = time.time()
        agent._touch_activity("receiving stream response")

    for attempt in range(max_stream_retries + 1):
        if agent._interrupt_requested:
            raise InterruptedError("Agent interrupted before Codex stream retry")

        intercepted_events = []
        writer_token = {"value": None}

        def _open_codex_stream(next_api_kwargs: dict[str, Any]):
            stream_kwargs = dict(next_api_kwargs)
            stream_kwargs["stream"] = True
            return active_client.responses.create(**stream_kwargs)

        def _codex_stream_created(_raw_stream: Any) -> None:
            # Claim the delta sink for THIS physical attempt. A newer attempt
            # supersedes this token and fences late deltas out of the turn.
            writer_token["value"] = claim_stream_writer(agent)

        def _accept_codex_chunk(_chunk: Any) -> bool:
            token = writer_token["value"]
            if token is None or stream_writer_is_current(agent, token):
                return True
            logger.warning(
                "Codex streaming attempt superseded by a newer stream; "
                "stopping consumption to preserve the single-writer "
                "invariant (model=%s).",
                api_kwargs.get("model", "unknown"),
            )
            return False

        def _finalize_codex_stream() -> Any:
            return _consume_codex_event_stream(
                list(intercepted_events),
                model=api_kwargs.get("model"),
            )

        try:
            event_stream = relay_llm.stream(
                dict(api_kwargs),
                _open_codex_stream,
                session_id=str(getattr(agent, "session_id", "") or ""),
                name=str(getattr(agent, "provider", "") or "codex"),
                model_name=str(api_kwargs.get("model") or ""),
                finalizer=_finalize_codex_stream,
                on_stream_created=_codex_stream_created,
                on_chunk=intercepted_events.append,
                chunk_adapter=lambda chunk: chunk,
                accept_chunk=_accept_codex_chunk,
                completed_response_predicate=lambda response: bool(
                    hasattr(response, "output") and not hasattr(response, "__iter__")
                ),
                metadata={
                    "api_mode": "codex_responses",
                    "api_request_id": getattr(agent, "_current_api_request_id", None),
                    "call_role": (
                        "delegated"
                        if getattr(agent, "is_subagent", False)
                        else "primary"
                    ),
                    "retry_count": attempt,
                },
                defer_logical_completion=True,
            )
        except (
            _httpx.RemoteProtocolError,
            _httpx.ReadTimeout,
            _httpx.ConnectError,
            ConnectionError,
        ) as exc:
            if attempt < max_stream_retries:
                logger.debug(
                    "Codex Responses stream connect failed (attempt %s/%s); "
                    "retrying. %s error=%s",
                    attempt + 1,
                    max_stream_retries + 1,
                    agent._client_log_context(),
                    exc,
                )
                continue
            _log_codex_request_failure(
                agent,
                exc,
                stream_opened=writer_token["value"] is not None,
            )
            raise
        except _APIConnectionError as exc:
            _log_codex_request_failure(
                agent,
                exc,
                stream_opened=writer_token["value"] is not None,
            )
            raise

        def _interrupt_or_superseded() -> bool:
            return bool(agent._interrupt_requested)

        try:
            try:
                final = _consume_codex_event_stream(
                    event_stream,
                    model=api_kwargs.get("model"),
                    on_text_delta=_on_text_delta,
                    on_reasoning_delta=_on_reasoning_delta,
                    on_commentary_message=(
                        _on_commentary_message
                        if (
                            getattr(agent, "interim_assistant_callback", None) is not None
                            and getattr(agent, "show_commentary", True)
                        )
                        else None
                    ),
                    on_first_delta=on_first_delta,
                    on_event=_on_event,
                    interrupt_check=_interrupt_or_superseded,
                )
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "Codex Responses stream transport failed mid-iteration "
                        "(attempt %s/%s); retrying. %s error=%s",
                        attempt + 1, max_stream_retries + 1,
                        agent._client_log_context(), exc,
                    )
                    continue
                _log_codex_request_failure(
                    agent,
                    exc,
                    stream_opened=writer_token["value"] is not None,
                )
                raise
            except RuntimeError:
                if event_stream.final_response is not None:
                    return event_stream.final_response
                raise
            except _APIConnectionError as exc:
                _log_codex_request_failure(
                    agent,
                    exc,
                    stream_opened=writer_token["value"] is not None,
                )
                raise

            # A terminal response has already been assembled at this point
            # (``final`` is built), so a transport error while draining the
            # rest of the iterator — done only to let Relay run its response
            # finalizer — must NOT discard it or trigger a new physical
            # request. Record it as a non-fatal finalization warning and
            # still return the already-completed, already-billed response.
            if not agent._interrupt_requested:
                try:
                    for _ignored in event_stream:
                        pass
                except (
                    _httpx.RemoteProtocolError,
                    _httpx.ReadTimeout,
                    _httpx.ConnectError,
                    ConnectionError,
                ) as exc:
                    logger.warning(
                        "Codex Responses stream transport finalization failed "
                        "after a terminal response was already received; "
                        "returning the completed response instead of "
                        "retrying. %s error=%s",
                        agent._client_log_context(), exc,
                    )
                except _APIConnectionError as exc:
                    _log_codex_request_failure(
                        agent,
                        exc,
                        stream_opened=writer_token["value"] is not None,
                    )
                    logger.warning(
                        "Codex Responses stream transport finalization failed "
                        "after a terminal response was already received; "
                        "returning the completed response instead of "
                        "retrying. %s error=%s",
                        agent._client_log_context(), exc,
                    )

            if final.status in {"incomplete", "failed"}:
                logger.warning(
                    "Codex Responses stream terminal status=%s "
                    "(incomplete_details=%s, error=%s, streamed_chars=%d). %s",
                    final.status, final.incomplete_details, final.error,
                    sum(len(p) for p in agent._codex_streamed_text_parts),
                    agent._client_log_context(),
                )

            return final
        finally:
            close_fn = getattr(event_stream, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    # A failed close can leave this response's connection
                    # checked out of the httpx pool while the caller's finally
                    # reports a reuse-reason close (e.g. interrupt_check broke
                    # the event loop with collected output) — caching the
                    # client with the leaked connection. Poison the slot so
                    # that close really closes the pool (owner-thread abort;
                    # mirrors the chat-streaming interrupt-break handling).
                    # ``client is None`` means the shared primary client,
                    # which is never reuse-cached and must not have its
                    # sockets force-shut here.
                    if client is not None:
                        agent._abort_request_openai_client(
                            active_client, reason="codex_stream_close_failed"
                        )


def run_codex_create_stream_fallback(agent, api_kwargs: dict, client: Any = None):
    """Backward-compatible alias for the unified event-driven path.

    Historically this was the fallback when the SDK's high-level
    ``responses.stream(...)`` helper raised on shape drift.  The primary
    path now does exactly what the fallback did, so this just forwards.
    Kept as a public symbol because tests and a small number of call sites
    still reference it by name.
    """
    return run_codex_stream(agent, api_kwargs, client=client)


__all__ = [
    "run_codex_stream",
    "run_codex_create_stream_fallback",
    "_consume_codex_event_stream",
]
