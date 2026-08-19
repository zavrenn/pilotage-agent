"""One turn: a message arrives, the model answers.

History is per chat and lives in memory only — restarting the process forgets
every conversation. Persistence is a later slice, deliberately.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import APIStatusError, AsyncOpenAI

from .codex import auth, client as codex_client, stream as codex_stream
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    role: str
    content: str
    # Encrypted reasoning from the model, replayed on the next turn so it does
    # not have to think the same thoughts again.
    reasoning_items: List[Dict[str, Any]] = field(default_factory=list)


class Agent:
    def __init__(self, config: Config):
        self._config = config
        self._history: Dict[str, List[Turn]] = {}
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

    def _build_input(self, chat_id: str, user_text: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for turn in self._history.get(chat_id, []):
            if turn.role == "user":
                items.append({"role": "user", "content": turn.content})
                continue
            for item in turn.reasoning_items:
                # `store: False` means the server cannot resolve item ids, so a
                # replayed id is a 404. `_issuer_kind` is not part of the API.
                items.append({k: v for k, v in item.items() if k not in ("id", "_issuer_kind")})
            # A reasoning item must be followed by a message, even an empty one,
            # or the API rejects the input with `missing_following_item`.
            items.append({"role": "assistant", "content": turn.content})
        items.append({"role": "user", "content": user_text})
        return items

    def _remember(self, chat_id: str, user_text: str, result: codex_stream.StreamResult) -> None:
        history = self._history.setdefault(chat_id, [])
        history.append(Turn(role="user", content=user_text))
        history.append(
            Turn(role="assistant", content=result.text, reasoning_items=result.reasoning_items)
        )
        limit = max(2, self._config.history_turns * 2)
        if len(history) > limit:
            del history[: len(history) - limit]

    def forget(self, chat_id: str) -> None:
        self._history.pop(chat_id, None)

    # -- the turn -----------------------------------------------------------

    async def respond(self, chat_id: str, user_text: str) -> str:
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            result = await self._run_turn(chat_id, user_text)
            self._remember(chat_id, user_text, result)
            return result.text

    async def _run_turn(self, chat_id: str, user_text: str) -> codex_stream.StreamResult:
        request = codex_stream.build_request(
            model=self._config.model,
            instructions=self._config.instructions,
            input_items=self._build_input(chat_id, user_text),
            session_id=chat_id,
            reasoning_effort=self._config.reasoning_effort,
        )

        try:
            return await self._stream_once(request, force_refresh=False)
        except APIStatusError as exc:
            if exc.status_code not in (401, 403):
                raise
            # The token went stale between the expiry check and the request.
            logger.info("Codex returned %s; refreshing credentials and retrying once", exc.status_code)
            return await self._stream_once(request, force_refresh=True)

    async def _stream_once(self, request: Dict[str, Any], *, force_refresh: bool) -> codex_stream.StreamResult:
        client = await self._ensure_client(force_refresh=force_refresh)
        stream = await client.responses.create(**request, stream=True)
        try:
            return await codex_stream.consume_stream(stream)
        finally:
            await stream.close()
