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
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from openai import APIStatusError, AsyncOpenAI

from . import media
from .codex import (
    auth,
    client as codex_client,
    compaction,
    stream as codex_stream,
)
from .config import Config
from .context_files import build_context_files_prompt
from .cron.jobs import CronStore
from .history import ConversationStore
from .tools.memory import MemoryStore
from .tools import (
    ToolContext,
    build_registry,
    build_skills_prompt,
    enabled_groups,
    responses_tool_output,
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

# Codex can finish a response after emitting only a commentary/analysis message.
# Hermes proved that this is a paused turn, not an empty answer: replay its exact
# message item and let it continue, but stop a response loop after three tries.
MAX_CODEX_INCOMPLETE_RESPONSES = 3
CODEX_INCOMPLETE_RESPONSE = (
    "Codex response remained incomplete after 3 continuation attempts"
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


def _retire_tool_result_images(items: List[Dict[str, Any]]) -> None:
    """Keep one follow-up turn, then replace old tool images with their text."""
    for item in items:
        if (
            not isinstance(item, dict)
            or item.get("type") != "function_call_output"
        ):
            continue
        output = item.get("output")
        if not isinstance(output, list) or not any(
            isinstance(part, dict) and part.get("type") == "input_image"
            for part in output
        ):
            continue
        text_parts = [
            str(part.get("text") or "")
            for part in output
            if (
                isinstance(part, dict)
                and part.get("type") == "input_text"
                and isinstance(part.get("text"), str)
            )
        ]
        item["output"] = (
            "\n".join(filter(None, text_parts))
            or "[vision image removed after the follow-up turn]"
        )


class Agent:
    def __init__(
        self,
        config: Config,
        store: Optional[ConversationStore] = None,
        *,
        cron_store: Optional[CronStore] = None,
        cron_wake: Optional[Callable[[], None]] = None,
        disabled_tool_groups: Sequence[str] = (),
    ):
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
        disabled = set(disabled_tool_groups)
        self._tool_groups = [
            group for group in enabled_groups(config.settings, self._registry)
            if group not in disabled
        ]
        self._tools = self._registry.definitions(self._tool_groups)
        self._base_instructions = config.instructions
        self._skills_prompt = ""
        self._instructions = self._base_instructions
        if "skills" in self._tool_groups:
            self._skills_prompt = build_skills_prompt(config)
            if self._skills_prompt:
                self._instructions = (
                    f"{self._instructions}\n\n{self._skills_prompt}"
                )

        configured_cwd = config.settings.text("terminal.cwd", "")
        default_workspace = getattr(
            config, "workspace_dir", Path(config.state_dir) / "workspace"
        )
        self._context_cwd = (
            Path(configured_cwd).expanduser()
            if configured_cwd
            else default_workspace
        )

        self._memory_store: Optional[MemoryStore] = None
        if "memory" in self._tool_groups:
            self._memory_store = MemoryStore(
                config.memory_dir,
                memory_char_limit=config.memory_char_limit,
                user_char_limit=config.user_memory_char_limit,
            )
            self._memory_store.load_from_disk()
        # One frozen instruction snapshot per chat session. Pilotage serves
        # several chats with one Agent instance; workspace instructions and
        # memory written mid-conversation become visible after /new, not in the
        # middle of an established prompt-cache prefix.
        self._session_instructions: Dict[str, str] = {}

        self._cron_store: Optional[CronStore] = None
        self._cron_wake = cron_wake
        if "cron" in self._tool_groups:
            self._cron_store = cron_store or CronStore(
                config.state_dir,
                timezone_name=config.cron_timezone,
                claim_ttl_seconds=config.cron_claim_ttl_seconds,
                output_retention=config.cron_output_retention,
            )

        if self._tools:
            logger.info("Tools enabled: %s", ", ".join(self._registry.names(self._tool_groups)))
        # Whatever the tools of one chat need to remember between calls.
        self._tool_state: Dict[str, Dict[str, Any]] = {}

    def _instructions_for_session(self, chat_id: str) -> str:
        cached = self._session_instructions.get(chat_id)
        if cached is not None:
            return cached

        blocks = [self._base_instructions]
        workspace_context = build_context_files_prompt(self._context_cwd)
        if workspace_context:
            blocks.append(workspace_context)
        if self._skills_prompt:
            blocks.append(self._skills_prompt)

        if self._memory_store is not None:
            snapshot = MemoryStore(
                self._config.memory_dir,
                memory_char_limit=self._config.memory_char_limit,
                user_char_limit=self._config.user_memory_char_limit,
            )
            snapshot.load_from_disk()
            blocks.extend(
                block
                for target in ("memory", "user")
                if (block := snapshot.format_for_system_prompt(target))
            )

        instructions = "\n\n".join(blocks)
        self._session_instructions[chat_id] = instructions
        return instructions

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
                fallback_path=self._config.main_credentials_path,
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
        # and replaying every one on every turn would grow the request and the
        # process without bound. Only the newest completed turn keeps pictures,
        # whether they came from the user or vision_analyze, so one follow-up
        # still sees them. Hermes retires old image parts in its context
        # compressor for the same reason.
        for turn in history:
            turn.image_parts = []
            if turn.items:
                _retire_tool_result_images(turn.items)
        history.append(Turn(role="user", content=user_text, image_parts=image_parts))
        history.append(Turn(role="assistant", content=result.text, items=result.items))
        if self._native_compaction_active():
            self._prune_native_history(history)
        else:
            limit = self._history_limit()
            if len(history) > limit:
                del history[: len(history) - limit]

    def _native_compaction_active(self) -> bool:
        return (
            self._config.codex_native_compaction
            and compaction.is_native_compaction_model(self._config.model)
        )

    def _prune_native_history(self, history: List[Turn]) -> None:
        """Bound local state around the newest opaque compaction checkpoint."""
        checkpoint_index: Optional[int] = None
        for index, turn in enumerate(history):
            if turn.role == "assistant" and compaction.has_compaction_checkpoint(
                turn.items
            ):
                checkpoint_index = index
        if checkpoint_index is None:
            return

        remaining = compaction.RETAINED_USER_MESSAGE_TOKEN_BUDGET
        retained_reversed: List[Turn] = []
        for turn in reversed(history[:checkpoint_index]):
            if turn.role != "user" or not turn.content.strip():
                continue
            if remaining <= 0:
                break
            cost = max(1, len(turn.content) // 4)
            if cost <= remaining:
                retained_reversed.append(turn)
                remaining -= cost
                continue
            truncated = turn.content[: remaining * 4]
            if truncated.strip():
                retained_reversed.append(
                    Turn(
                        role="user",
                        content=truncated,
                        image_parts=list(turn.image_parts),
                    )
                )
            remaining = 0

        history[:] = list(reversed(retained_reversed)) + history[checkpoint_index:]

    async def _restore(self, chat_id: str) -> None:
        """Bring a chat's stored history back, once, after a restart.

        Normal reasoning and pictures stay process-local. Eligible sessions
        also restore the opaque compaction checkpoint required for continuity.
        """
        if chat_id in self._restored:
            return
        self._restored.add(chat_id)
        if chat_id in self._history:
            return
        if self._native_compaction_active():
            stored = await asyncio.to_thread(self._store.load_with_replay, chat_id)
            history: List[Turn] = []
            for role, content, replay in stored:
                items: List[Dict[str, Any]] = []
                if role == "assistant" and replay:
                    items = [dict(item) for item in replay]
                    items.append({"role": "assistant", "content": content})
                history.append(Turn(role=role, content=content, items=items))
            self._prune_native_history(history)
        else:
            stored = await asyncio.to_thread(
                self._store.load, chat_id, self._history_limit()
            )
            history = [Turn(role=role, content=content) for role, content in stored]
        if history:
            self._history[chat_id] = history
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
            # Record the boundary before changing live state. If this fails,
            # /new reports failure and the old conversation remains intact.
            await asyncio.to_thread(self._store.new_session, chat_id)
            self._history.pop(chat_id, None)
            # The task list belonged to work that has just been abandoned.
            self._tool_state.pop(chat_id, None)
            # Memory written during the old session becomes visible in the
            # frozen prompt of the next one.
            self._session_instructions.pop(chat_id, None)
            # Do not reload the conversation that the durable boundary ended.
            self._restored.add(chat_id)

    # -- the turn -----------------------------------------------------------

    async def respond(
        self,
        chat_id: str,
        user_text: str,
        attachments: Sequence[media.Attachment] = (),
        on_notice: Optional[Notice] = None,
        *,
        origin: Optional[Dict[str, str]] = None,
    ) -> str:
        # Reading the files is blocking I/O and base64 of a few megabytes is not
        # free, so it happens off the event loop.
        if attachments:
            image_parts, attached_image_paths = await asyncio.to_thread(
                media.image_parts_with_paths, attachments
            )
        else:
            image_parts, attached_image_paths = [], []
        if attached_image_paths:
            # Hermes gives the model a string handle alongside the pixels so
            # the same image can be passed to image_generate for editing.
            base_text = user_text.strip() or "What do you see in this image?"
            hints = "\n".join(
                f"[Image attached at: {path}]" for path in attached_image_paths
            )
            user_text = f"{base_text}\n\n{hints}"
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            await self._restore(chat_id)
            if self._memory_store is not None:
                self._memory_store.reset_consolidation_failures()
            result = await self._run_turn(
                chat_id, user_text, image_parts, on_notice, origin=origin
            )
            self._remember(chat_id, user_text, image_parts, result)
            checkpoints = (
                compaction.persistent_compaction_items(result.items)
                if self._native_compaction_active()
                else []
            )
            await asyncio.to_thread(
                self._store.append_with_replay,
                chat_id,
                [
                    ("user", user_text, []),
                    ("assistant", result.text, checkpoints),
                ],
            )
            return result.text

    async def _run_turn(
        self,
        chat_id: str,
        user_text: str,
        image_parts: List[Dict[str, Any]],
        on_notice: Optional[Notice] = None,
        *,
        origin: Optional[Dict[str, str]] = None,
    ) -> TurnResult:
        """Call the model, run what it asks for, call it again — until it answers."""
        self._instructions_for_session(chat_id)
        history = self._build_input(chat_id, user_text, image_parts)
        context = ToolContext(
            chat_id=chat_id,
            config=self._config,
            state=self._tool_state.setdefault(chat_id, {}),
            conversation_store=self._store,
            memory_store=self._memory_store,
            cron_store=self._cron_store,
            origin=origin,
            cron_wake=self._cron_wake,
        )
        # Everything the assistant does this turn, in order, ready to be sent
        # back on the next call and kept as history afterwards.
        items: List[Dict[str, Any]] = []
        limit = max(1, self._config.max_tool_iterations)

        def _finish(text: str) -> TurnResult:
            return TurnResult(
                text=_append_generated_media(
                    text,
                    items,
                    self._config.workspace_dir,
                ),
                items=items,
            )

        async def _next_action_or_answer(
            offered_tools: Optional[List[Dict[str, Any]]],
        ) -> Optional[codex_stream.StreamResult]:
            for attempt in range(1, MAX_CODEX_INCOMPLETE_RESPONSES + 1):
                result = await self._call_model(
                    chat_id, history + items, offered_tools, on_notice
                )
                items.extend(_assistant_items(result))
                # Tool calls take precedence: commentary commonly introduces a
                # tool and must be replayed, but it must not delay execution.
                if result.tool_calls or not result.needs_continuation:
                    return result
                if attempt < MAX_CODEX_INCOMPLETE_RESPONSES:
                    logger.info(
                        "Codex response for %s paused after commentary; continuing (%d/%d)",
                        chat_id,
                        attempt,
                        MAX_CODEX_INCOMPLETE_RESPONSES,
                    )
            return None

        for step in range(limit):
            result = await _next_action_or_answer(self._tools)
            if result is None:
                logger.warning(
                    "Codex response for %s remained incomplete after %d attempts",
                    chat_id,
                    MAX_CODEX_INCOMPLETE_RESPONSES,
                )
                return _finish(CODEX_INCOMPLETE_RESPONSE)
            if not result.tool_calls:
                return _finish(result.text)

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
                        "output": responses_tool_output(output),
                    }
                )

        # Out of steps. Ask for what it has rather than sending nothing: the
        # work is already done and paid for, and the person is still waiting.
        logger.warning("Chat %s used all %d tool steps; asking for a summary", chat_id, limit)
        items.append({"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST})
        result = await _next_action_or_answer(None)
        if result is None:
            return _finish(CODEX_INCOMPLETE_RESPONSE)
        return _finish(result.text)

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
            instructions=self._instructions_for_session(chat_id),
            input_items=input_items,
            session_id=chat_id,
            reasoning_effort=self._config.reasoning_effort,
            tools=tools,
            native_compaction_enabled=self._config.codex_native_compaction,
            compact_threshold=self._config.codex_compact_threshold,
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
    message_items = codex_stream.message_items_for_replay(result.message_items)
    if message_items:
        items.extend(message_items)
    # A reasoning item must be followed by a message, even an empty one, or the
    # API rejects the input with `missing_following_item`. Older/unphased
    # responses still use the plain assistant-message fallback.
    elif items or result.text:
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


def _append_generated_media(
    text: str,
    items: List[Dict[str, Any]],
    workspace: Path,
) -> str:
    """Port Hermes' deterministic image delivery from current-turn outputs."""
    image_call_ids = {
        str(item.get("call_id") or "")
        for item in items
        if item.get("type") == "function_call"
        and item.get("name") == "image_generate"
    }
    if not image_call_ids:
        return text

    existing, _ = media.extract_outbound(text or "", (workspace,))
    seen = {attachment.path for attachment in existing}
    tags: List[str] = []
    for item in items:
        if (
            item.get("type") != "function_call_output"
            or str(item.get("call_id") or "") not in image_call_ids
        ):
            continue
        try:
            payload = json.loads(str(item.get("output") or ""))
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or not payload.get("success"):
            continue
        path = payload.get("image")
        if not isinstance(path, str):
            continue
        attachments, _ = media.extract_outbound(
            f"MEDIA:{path}",
            (workspace,),
        )
        for attachment in attachments:
            if attachment.path in seen:
                continue
            seen.add(attachment.path)
            tags.append(f"MEDIA:{attachment.path}")

    if not tags:
        return text
    visible = (text or "").strip()
    suffix = "\n".join(tags)
    return f"{visible}\n{suffix}" if visible else suffix


def _user_item(text: str, image_parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A user message, as plain text or as text plus pictures."""
    if not image_parts:
        return {"role": "user", "content": text}
    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"type": "input_text", "text": text})
    parts.extend(image_parts)
    return {"role": "user", "content": parts}
