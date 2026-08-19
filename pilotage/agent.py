"""One turn: a message arrives, the model answers.

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

logger = logging.getLogger(__name__)

# A silent stream is retried once. A fresh connection almost always works
# straight away, and a second failure is the backend, not the socket.
MAX_STREAM_RECONNECTS = 1

# What the person waiting is told when we drop a quiet connection. They have
# been watching the typing indicator for two minutes by then; silence would
# read as the agent having given up.
RECONNECT_NOTICE = "Still nothing back from the model. Reconnecting…"

# Called with a line to show the person waiting, mid-turn.
Notice = Callable[[str], Awaitable[None]]


@dataclass
class Turn:
    role: str
    content: str
    # Encrypted reasoning from the model, replayed on the next turn so it does
    # not have to think the same thoughts again.
    reasoning_items: List[Dict[str, Any]] = field(default_factory=list)
    # Images sent with a user turn, as `input_image` parts.
    image_parts: List[Dict[str, Any]] = field(default_factory=list)


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
            for item in turn.reasoning_items:
                # `store: False` means the server cannot resolve item ids, so a
                # replayed id is a 404. `_issuer_kind` is not part of the API.
                items.append({k: v for k, v in item.items() if k not in ("id", "_issuer_kind")})
            # A reasoning item must be followed by a message, even an empty one,
            # or the API rejects the input with `missing_following_item`.
            items.append({"role": "assistant", "content": turn.content})
        items.append(_user_item(user_text, image_parts))
        return items

    def _remember(
        self,
        chat_id: str,
        user_text: str,
        image_parts: List[Dict[str, Any]],
        result: codex_stream.StreamResult,
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
        history.append(
            Turn(role="assistant", content=result.text, reasoning_items=result.reasoning_items)
        )
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
    ) -> codex_stream.StreamResult:
        request = codex_stream.build_request(
            model=self._config.model,
            instructions=self._config.instructions,
            input_items=self._build_input(chat_id, user_text, image_parts),
            session_id=chat_id,
            reasoning_effort=self._config.reasoning_effort,
        )
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


def _user_item(text: str, image_parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A user message, as plain text or as text plus pictures."""
    if not image_parts:
        return {"role": "user", "content": text}
    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"type": "input_text", "text": text})
    parts.extend(image_parts)
    return {"role": "user", "content": parts}
