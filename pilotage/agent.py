"""One turn: a message arrives, the model works, the model answers.

A turn is not one call any more. The model may ask to run tools, read what
they returned and go again, as many times as the work takes; only when it
stops asking is there an answer to send. Everything that happened in between —
its reasoning, the calls it made, what came back — is kept and replayed on the
next turn, so the conversation the model sees is the one it actually had.

History is per chat. The working copy is in memory; every turn is also written
to disk so a restart does not silently end every conversation at once. See
``history.py`` for what is kept and what is deliberately not.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from openai import APIStatusError, AsyncOpenAI

from . import media
from .codex import auth, client as codex_client, stream as codex_stream
from .config import Config
from .history import ConversationStore
from .tools import (
    ToolContext,
    build_registry,
    build_skills_prompt,
    enabled_groups,
    run_calls,
)

logger = logging.getLogger(__name__)

# A silent stream is retried once. A fresh connection almost always works
# straight away, and a second failure is the backend, not the socket. The
# budget is per model call, not per turn: a turn that runs thirty tools makes
# thirty calls, and one bad socket at step two should not spend the whole
# turn's patience.
MAX_STREAM_RECONNECTS = 1

# What the person waiting is told when we drop a quiet connection. They have
# been watching the typing indicator for two minutes by then; silence would
# read as the agent having given up.
RECONNECT_NOTICE = "Still nothing back from the model. Reconnecting…"

# What the model is told when it has used every step it is allowed. Hermes'
# wording. It is asked to answer rather than cut off, so the person waiting
# gets what was found instead of nothing.
MAX_ITERATIONS_SUMMARY_REQUEST = (
    "You've reached the maximum number of tool-calling iterations allowed. "
    "Please provide a final response summarizing what you've found and accomplished so far, "
    "without calling any more tools."
)

# Called with a line to show the person waiting, mid-turn.
Notice = Callable[[str], Awaitable[None]]


@dataclass
class Turn:
    role: str
    content: str
    # For an assistant turn: everything it did, already in the shape the API
    # takes back — its encrypted reasoning, its message, the tools it called
    # and what they returned. Replaying reasoning saves it thinking the same
    # thoughts again; replaying the calls is what makes the next question about
    # "that file you read" answerable. Empty on a turn restored from disk,
    # which keeps only the words.
    items: List[Dict[str, Any]] = field(default_factory=list)
    # Images sent with a user turn, as `input_image` parts.
    image_parts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TurnResult:
    """What a whole turn produced: the answer, and the record of getting there."""

    text: str = ""
    items: List[Dict[str, Any]] = field(default_factory=list)


class Agent:
    def __init__(self, config: Config, store: Optional[ConversationStore] = None):
        self._config = config
        self._store = store or ConversationStore(config.conversations_path)
        self._history: Dict[str, List[Turn]] = {}
        # Chats whose stored history has already been looked for. Reading is
        # worth doing once per chat, not once per message.
        self._restored: set[str] = set()
        self._credentials: Optional[auth.Credentials] = None
        self._client: Optional[AsyncOpenAI] = None
        self._auth_lock = asyncio.Lock()
        # One turn at a time per chat, so two fast messages cannot interleave
        # their history writes.
        self._chat_locks: Dict[str, asyncio.Lock] = {}
        # What this build can do, and what this agent is allowed to do with it.
        # Decided once, at startup: a tool list that changed under a running
        # conversation would invalidate its prompt cache and confuse the model
        # about what it just used.
        self._registry = build_registry()
        self._tool_groups = enabled_groups(config.settings, self._registry)
        self._tools = self._registry.definitions(self._tool_groups)
        self._instructions = config.instructions
        if "skills" in self._tool_groups:
            skills_prompt = build_skills_prompt(config)
            if skills_prompt:
                self._instructions = f"{self._instructions}\n\n{skills_prompt}"
        if self._tools:
            logger.info("Tools enabled: %s", ", ".join(self._registry.names(self._tool_groups)))
        # Whatever the tools of one chat need to remember between calls.
        self._tool_state: Dict[str, Dict[str, Any]] = {}

    # -- credentials --------------------------------------------------------

    async def _ensure_client(self, *, force_refresh: bool = False) -> AsyncOpenAI:
        async with self._auth_lock:
            if self._client is not None and not force_refresh:
                # The access token is a short-lived JWT; check before every call
                # rather than discovering the expiry as a 401 mid-answer.
                assert self._credentials is not None
                if not auth.access_token_is_expiring(self._credentials.access_token):
                    return self._client

            credentials = await asyncio.to_thread(
                auth.resolve_credentials,
                self._config.credentials_path,
                force_refresh=force_refresh,
            )
            if self._client is None or credentials.access_token != (
                self._credentials.access_token if self._credentials else None
            ):
                self._credentials = credentials
                self._client = codex_client.build_client(
                    credentials, timeout_seconds=self._config.request_timeout_seconds
                )
            return self._client

    # -- history ------------------------------------------------------------

    def _build_input(
        self, chat_id: str, user_text: str, image_parts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        # Only the newest user turn still carries its pictures; `_remember`
        # clears the rest.
        for turn in self._history.get(chat_id, []):
            if turn.role == "user":
                items.append(_user_item(turn.content, turn.image_parts))
                continue
            if turn.items:
                items.extend(turn.items)
                continue
            # Restored from disk: only the words survived.
            items.append({"role": "assistant", "content": turn.content})
        items.append(_user_item(user_text, image_parts))
        return items

    def _remember(
        self,
        chat_id: str,
        user_text: str,
        image_parts: List[Dict[str, Any]],
        result: TurnResult,
    ) -> None:
        history = self._history.setdefault(chat_id, [])
        # Images are heavy — a base64 photo is a third larger than the file —
        # and replaying every one of them on every turn would grow the request
        # and the process without bound. Only the newest user turn keeps its
        # pictures, so a follow-up about the photo just sent still works and
        # nothing older is held in memory. Hermes strips old image parts the
        # same way, in its context compressor.
        for turn in history:
            turn.image_parts = []
        history.append(Turn(role="user", content=user_text, image_parts=image_parts))
        history.append(Turn(role="assistant", content=result.text, items=result.items))
        limit = self._history_limit()
        if len(history) > limit:
            del history[: len(history) - limit]

    async def _restore(self, chat_id: str) -> None:
        """Bring a chat's stored history back, once, after a restart.

        Only the words come back. A turn restored this way has no reasoning
        items and no pictures, which is what ``history.py`` explains.
        """
        if chat_id in self._restored:
            return
        self._restored.add(chat_id)
        if chat_id in self._history:
            return
        stored = await asyncio.to_thread(
            self._store.load, chat_id, self._history_limit()
        )
        if not stored:
            return
        self._history[chat_id] = [Turn(role=role, content=content) for role, content in stored]
        logger.info("Picked up %d stored turns for %s", len(stored), chat_id)

    def _history_limit(self) -> int:
        """Turns kept per chat — a question and its answer count as two."""
        return max(2, self._config.history_turns * 2)

    async def forget(self, chat_id: str) -> None:
        """Drop a conversation's history.

        Takes the chat's lock rather than clearing outright. A turn already
        running writes its question and answer back when it finishes, so
        clearing underneath it would hand back the conversation the person
        just asked to end.
        """
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            self._history.pop(chat_id, None)
            # The task list belonged to work that has just been abandoned.
            self._tool_state.pop(chat_id, None)
            # The stored conversation is left where it is and a new one
            # begins, so a restart cannot hand back what was just ended.
            await asyncio.to_thread(self._store.new_session, chat_id)
            # Held even if that write failed: a chat cannot be un-forgotten by
            # the next message going looking for it on disk.
            self._restored.add(chat_id)

    # -- the turn -----------------------------------------------------------

    async def respond(
        self,
        chat_id: str,
        user_text: str,
        attachments: Sequence[media.Attachment] = (),
        on_notice: Optional[Notice] = None,
    ) -> str:
        # Reading the files is blocking I/O and base64 of a few megabytes is not
        # free, so it happens off the event loop.
        image_parts = (
            await asyncio.to_thread(media.image_parts, attachments) if attachments else []
        )
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            await self._restore(chat_id)
            result = await self._run_turn(chat_id, user_text, image_parts, on_notice)
            self._remember(chat_id, user_text, image_parts, result)
            await asyncio.to_thread(
                self._store.append,
                chat_id,
                [("user", user_text), ("assistant", result.text)],
            )
            return result.text

    async def _run_turn(
        self,
        chat_id: str,
        user_text: str,
        image_parts: List[Dict[str, Any]],
        on_notice: Optional[Notice] = None,
    ) -> TurnResult:
        """Call the model, run what it asks for, call it again — until it answers."""
        history = self._build_input(chat_id, user_text, image_parts)
        context = ToolContext(
            chat_id=chat_id,
            config=self._config,
            state=self._tool_state.setdefault(chat_id, {}),
        )
        # Everything the assistant does this turn, in order, ready to be sent
        # back on the next call and kept as history afterwards.
        items: List[Dict[str, Any]] = []
        limit = max(1, self._config.max_tool_iterations)

        for step in range(limit):
            result = await self._call_model(chat_id, history + items, self._tools, on_notice)
            items.extend(_assistant_items(result))
            if not result.tool_calls:
                return TurnResult(text=result.text, items=items)

            names = ", ".join(call["name"] for call in result.tool_calls)
            logger.info("Step %d for %s: %s", step + 1, chat_id, names)
            outputs = await run_calls(
                self._registry,
                result.tool_calls,
                context,
                allowed_groups=self._tool_groups,
                max_result_chars=self._config.max_tool_result_chars,
                step_budget_chars=self._config.max_tool_step_chars,
            )
            for call, output in zip(result.tool_calls, outputs):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": output,
                    }
                )

        # Out of steps. Ask for what it has rather than sending nothing: the
        # work is already done and paid for, and the person is still waiting.
        logger.warning("Chat %s used all %d tool steps; asking for a summary", chat_id, limit)
        items.append({"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST})
        result = await self._call_model(chat_id, history + items, None, on_notice)
        items.extend(_assistant_items(result))
        return TurnResult(text=result.text, items=items)

    async def _call_model(
        self,
        chat_id: str,
        input_items: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        on_notice: Optional[Notice] = None,
    ) -> codex_stream.StreamResult:
        """One call to the model, retried when the connection rather than the request fails."""
        request = codex_stream.build_request(
            model=self._config.model,
            instructions=self._instructions,
            input_items=input_items,
            session_id=chat_id,
            reasoning_effort=self._config.reasoning_effort,
            tools=tools,
        )
        # Recomputed per call: a turn grows as tools return, and a request that
        # has grown legitimately takes longer to start.
        ttfb_timeout, idle_timeout = codex_stream.stream_timeouts(
            request,
            ttfb_timeout=self._config.codex_first_event_timeout_seconds,
            idle_timeout=self._config.codex_quiet_stream_timeout_seconds,
        )

        force_refresh = False
        refreshed = False
        reconnects = 0
        while True:
            try:
                return await self._stream_once(
                    request,
                    force_refresh=force_refresh,
                    ttfb_timeout=ttfb_timeout,
                    idle_timeout=idle_timeout,
                )
            except APIStatusError as exc:
                if exc.status_code not in (401, 403) or refreshed:
                    raise
                # The token went stale between the expiry check and the request.
                logger.info(
                    "Codex returned %s; refreshing credentials and retrying once", exc.status_code
                )
                force_refresh = True
                refreshed = True
            except codex_stream.CodexStreamTimeout as exc:
                if reconnects >= MAX_STREAM_RECONNECTS:
                    raise
                reconnects += 1
                # The credentials are not the problem here; keep the ones we have.
                force_refresh = False
                logger.warning(
                    "%s Dropping the connection and reconnecting (%d/%d).",
                    exc,
                    reconnects,
                    MAX_STREAM_RECONNECTS,
                )
                await _notify(on_notice, RECONNECT_NOTICE)

    async def _stream_once(
        self,
        request: Dict[str, Any],
        *,
        force_refresh: bool,
        ttfb_timeout: float,
        idle_timeout: float,
    ) -> codex_stream.StreamResult:
        client = await self._ensure_client(force_refresh=force_refresh)
        stream = await client.responses.create(**request, stream=True)
        try:
            return await codex_stream.consume_stream(
                stream, ttfb_timeout=ttfb_timeout, idle_timeout=idle_timeout
            )
        finally:
            # Closing the response is what actually lets go of a wedged
            # connection, and it must not replace the error that got us here.
            try:
                await stream.close()
            except Exception:  # noqa: BLE001 - the turn already has its outcome
                logger.debug("Closing the Codex stream failed", exc_info=True)


async def _notify(on_notice: Optional[Notice], text: str) -> None:
    """Tell the person waiting what is happening. Never at the cost of the turn."""
    if on_notice is None:
        return
    try:
        await on_notice(text)
    except Exception:  # noqa: BLE001 - a failed notice must not fail the answer
        logger.debug("Could not deliver the notice %r", text, exc_info=True)


def _assistant_items(result: codex_stream.StreamResult) -> List[Dict[str, Any]]:
    """One model call, in the shape the next request takes it back as."""
    items: List[Dict[str, Any]] = []
    for item in result.reasoning_items:
        # `store: False` means the server cannot resolve item ids, so a
        # replayed id is a 404. `_issuer_kind` is not part of the API.
        items.append({k: v for k, v in item.items() if k not in ("id", "_issuer_kind")})
    # A reasoning item must be followed by a message, even an empty one, or the
    # API rejects the input with `missing_following_item`. A call with neither
    # reasoning nor words leaves nothing to say.
    if items or result.text:
        items.append({"role": "assistant", "content": result.text})
    for call in result.tool_calls:
        items.append(
            {
                "type": "function_call",
                "call_id": call["call_id"],
                "name": call["name"],
                "arguments": call["arguments"],
            }
        )
    return items


def _user_item(text: str, image_parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A user message, as plain text or as text plus pictures."""
    if not image_parts:
        return {"role": "user", "content": text}
    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"type": "input_text", "text": text})
    parts.extend(image_parts)
    return {"role": "user", "content": parts}
