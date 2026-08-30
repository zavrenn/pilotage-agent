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
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Sequence

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError

from . import media
from .approvals import ApprovalManager
from .codex import (
    auth,
    client as codex_client,
    compaction,
    stream as codex_stream,
)
from .config import Config
from .context_files import build_context_files_prompt
from .cron.jobs import CronStore, timezone_for_name
from .history import (
    ActiveTurn,
    ConversationError,
    ConversationStore,
    StopCheckpoint,
    session_workspace_path,
)
from .i18n import DEFAULT_LANGUAGE, t
from .persistence import PersistenceAuditStore, build_persistence_policy
from .redact import identity_pseudonym
from .tools.memory import MemoryStore
from .tools.skills import reset_skill_view_dedup
from .tools import (
    ToolContext,
    build_registry,
    build_skills_prompt,
    enabled_groups,
    frame_untrusted_tool_result,
    responses_tool_output,
    run_calls,
)

logger = logging.getLogger(__name__)

SCHEDULED_PERSISTENCE_BOUNDARY = (
    "## Scheduled persistence\n"
    "Memory and skills are read-only during this scheduled run. Never create, "
    "edit, or delete either."
)

# A silent stream is retried once. A fresh connection almost always works
# straight away, and a second failure is the backend, not the socket. The
# budget is per model call, not per turn: a turn that runs thirty tools makes
# thirty calls, and one bad socket at step two should not spend the whole
# turn's patience.
MAX_STREAM_RECONNECTS = 1
STREAM_CLOSE_TIMEOUT_SECONDS = 5.0
COSMETIC_CLEANUP_TIMEOUT_SECONDS = 0.25

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

_NOTICE_SENT_WITHOUT_ID = "\x00pilotage-notice-without-id"


def _seconds(value: Optional[float]) -> str:
    """Format optional timing evidence without leaking request content."""

    return "-" if value is None else f"{max(0.0, float(value)):.2f}s"


def _is_masked_codex_replay_rejection(exc: APIStatusError) -> bool:
    """Match only Codex's exact stale-encrypted-replay 400 envelope."""

    if exc.status_code != 400 or not isinstance(exc.body, dict):
        return False
    error = exc.body.get("error")
    payload = error if isinstance(error, dict) else exc.body
    return (
        payload.get("code") == "invalid_prompt"
        and payload.get("message") == "Request blocked."
    )

ISOLATED_WORKSPACE_NOTE = (
    "This conversation's restricted working directory is {root}. "
    "Use inputs for inbound copies, tmp for temporary work, and exports for "
    "files intended for the user. Only files inside {exports} can be delivered "
    "with MEDIA. Put every deliverable there first."
)

# Called with a line and an optional message to replace while a turn is active.
Notice = Callable[[str, str], Awaitable[Any]]
ApprovalNotice = Callable[[str], Awaitable[Any]]
PreStopFence = Callable[[], Awaitable[None]]


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
    terminal_completed: bool = False


class TurnRecoveryRejected(ConversationError):
    """A durable interrupted turn is semantically unsafe to resume."""


class StopStatus(str, Enum):
    """Channel-neutral outcome of one exact-session stop request."""

    NOT_RUNNING = "not_running"
    STOPPED = "stopped"
    UNKNOWN = "unknown"
    TOO_LATE = "too_late"


@dataclass(frozen=True)
class StopOutcome:
    status: StopStatus
    checkpoint: Optional[StopCheckpoint] = None

    @property
    def previous_phase(self) -> str:
        return self.checkpoint.previous_phase if self.checkpoint else ""


class TurnStopped(Exception):
    """Expected control-flow signal for a durably stopped channel turn."""

    def __init__(self, outcome: StopOutcome):
        super().__init__(outcome.status.value)
        self.outcome = outcome


def _stopped_history_text(outcome: StopOutcome, language: str) -> str:
    key = (
        "runtime.stopped_after_actions"
        if outcome.previous_phase == "tool_completed"
        else "runtime.stopped"
    )
    return t(key, language)


@dataclass
class _ActiveExecution:
    """Identity guard shared by the model owner and concurrent /stop."""

    session: int
    claim_ids: tuple[str, ...] = ()
    task: Optional[asyncio.Task] = None
    owner_task: Optional[asyncio.Task] = None
    notice_task: Optional[asyncio.Task] = None
    before_preparation_stop: Optional[PreStopFence] = None
    preparing: bool = False
    transition_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stop_ready: asyncio.Event = field(default_factory=asyncio.Event)
    stop_task: Optional["asyncio.Task[StopOutcome]"] = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    outcome: Optional[StopOutcome] = None


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
        enabled_tool_groups: Optional[Sequence[str]] = None,
        enabled_skills: Optional[Sequence[str]] = None,
        working_directory: Optional[Path] = None,
        allow_persistence_writes: bool = False,
        persistence_audit: Optional[PersistenceAuditStore] = None,
        scheduled_run: bool = False,
    ):
        self._config = config
        self._store = store or ConversationStore(config.conversations_path)
        self._history: Dict[str, List[Turn]] = {}
        # Chats whose stored history has already been looked for. Reading is
        # worth doing once per chat, not once per message.
        self._restored: set[str] = set()
        self._credentials: Optional[auth.Credentials] = None
        self._client: Optional[AsyncOpenAI] = None
        # Replaced pools live only while an in-flight stream still owns them.
        # This is Pilotage's async ownership equivalent of Hermes deferring FD
        # release until every borrower has unwound.
        self._client_leases: Dict[int, int] = {}
        self._retired_clients: Dict[int, AsyncOpenAI] = {}
        self._auth_lock = asyncio.Lock()
        self._approvals = ApprovalManager(
            getattr(config, "approval_timeout_seconds", 300.0),
            language=getattr(config, "language", DEFAULT_LANGUAGE),
        )
        # One turn at a time per chat, so two fast messages cannot interleave
        # their history writes.
        self._chat_locks: Dict[str, asyncio.Lock] = {}
        # Channel turns keep their exact answer fenced until the delivery
        # obligation (and WhatsApp input identity) are durable.
        self._ready_turns: Dict[
            str, tuple[str, List[Dict[str, Any]], TurnResult]
        ] = {}
        self._completion_fences: Dict[str, asyncio.Event] = {}
        # Exact live owners for /stop. A control entry outlives model return
        # until the answer checkpoint wins, so completion and cancellation
        # cannot both become terminal outcomes.
        self._active_executions: Dict[str, _ActiveExecution] = {}
        self._stopped_turns: Dict[
            str, tuple[int, str, List[Dict[str, Any]], TurnResult]
        ] = {}
        self._owned_tasks: set[asyncio.Task] = set()
        # What this build can do, and what this agent is allowed to do with it.
        # Decided once, at startup: a tool list that changed under a running
        # conversation would invalidate its prompt cache and confuse the model
        # about what it just used.
        self._registry = build_registry()
        disabled = set(disabled_tool_groups)
        configured_groups = enabled_groups(config.settings, self._registry)
        if enabled_tool_groups is not None:
            requested = {
                str(group).strip()
                for group in enabled_tool_groups
                if str(group).strip()
            }
            unavailable = sorted(requested - set(configured_groups))
            if unavailable:
                raise ValueError(
                    "Requested tool groups are unavailable in this profile: "
                    + ", ".join(unavailable)
                )
        else:
            requested = set(configured_groups)
        self._tool_groups = [
            group
            for group in configured_groups
            if group in requested and group not in disabled
        ]
        self._tools = self._registry.definitions(self._tool_groups)
        self._allow_persistence_writes = bool(allow_persistence_writes)
        self._scheduled_run = bool(scheduled_run)
        memory_writes = self._allow_persistence_writes and "memory" in self._tool_groups
        skill_writes = self._allow_persistence_writes and {
            "file",
            "skills",
        }.issubset(self._tool_groups)
        self._persistence_writes_enabled = memory_writes or skill_writes
        self._persistence_policy = (
            build_persistence_policy(
                memory=memory_writes,
                skills=skill_writes,
            )
            if self._persistence_writes_enabled
            else ""
        )
        self._persistence_audit: Optional[PersistenceAuditStore] = None
        if self._persistence_writes_enabled:
            self._persistence_audit = persistence_audit or PersistenceAuditStore(
                config.state_dir
            )
        self._base_instructions = config.instructions
        self._enabled_skills = (
            None
            if enabled_skills is None
            else frozenset(
                str(name).strip()
                for name in enabled_skills
                if str(name).strip()
            )
        )

        configured_cwd = config.settings.text("terminal.cwd", "")
        default_workspace = getattr(
            config, "workspace_dir", Path(config.state_dir) / "workspace"
        )
        self._context_cwd = (
            Path(working_directory).expanduser()
            if working_directory is not None
            else (
                Path(configured_cwd).expanduser()
                if configured_cwd
                else default_workspace
            )
        )
        self._fixed_working_directory = working_directory is not None
        self._session_workdirs: Dict[str, tuple[int, Path]] = {}

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

    def _instructions_for_session(
        self,
        chat_id: str,
        working_directory: Optional[Path] = None,
    ) -> str:
        cached = self._session_instructions.get(chat_id)
        if cached is not None:
            return cached

        blocks = [self._base_instructions]
        workspace_context = build_context_files_prompt(self._context_cwd)
        if workspace_context:
            blocks.append(workspace_context)
        if (
            getattr(self._config, "session_isolated_workspaces", False)
            and working_directory is not None
        ):
            blocks.append(
                ISOLATED_WORKSPACE_NOTE.format(
                    root=working_directory,
                    exports=working_directory / "exports",
                )
            )
        if "skills" in self._tool_groups:
            # Keep an established session's prompt prefix frozen, but discover
            # newly accepted skills when the next session snapshot is created.
            live_skills_prompt = build_skills_prompt(
                self._config,
                self._enabled_skills,
            )
            if live_skills_prompt:
                blocks.append(live_skills_prompt)
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
        if self._persistence_policy:
            # Keep the autonomy ceiling after mutable legacy memory. A clean
            # but over-broad old entry must not outrank the current contract.
            blocks.append(self._persistence_policy)
        if self._scheduled_run:
            blocks.append(SCHEDULED_PERSISTENCE_BOUNDARY)

        instructions = "\n\n".join(blocks)
        self._session_instructions[chat_id] = instructions
        return instructions

    @property
    def session_workspace_root(self) -> Optional[Path]:
        """Root whose isolated session folders are owned by this Agent."""

        if self._fixed_working_directory or not getattr(
            self._config, "session_isolated_workspaces", False
        ):
            return None
        return self._context_cwd.expanduser().resolve(strict=False)

    def _session_working_directory(self, chat_id: str) -> Path:
        """Route one durable conversation generation to a private workspace."""

        if not getattr(
            self._config, "session_isolated_workspaces", False
        ):
            return self._context_cwd
        generation = self._store.current_session(chat_id)
        cached = self._session_workdirs.get(chat_id)
        if cached is not None and cached[0] == generation:
            return cached[1]

        base = self._context_cwd.expanduser().resolve(strict=False)
        if self._fixed_working_directory:
            candidate = base
        else:
            candidate = session_workspace_path(base, chat_id, generation)

        resolved_candidate = candidate.resolve(strict=False)
        try:
            resolved_candidate.relative_to(base)
        except ValueError as exc:
            raise RuntimeError(
                "Restricted session workspace escaped its configured root"
            ) from exc

        candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
        root = candidate.resolve(strict=True)
        try:
            root.relative_to(base)
        except ValueError as exc:
            raise RuntimeError(
                "Restricted session workspace escaped its configured root"
            ) from exc

        root.chmod(0o700)
        for name in ("inputs", "tmp", "exports"):
            child = root / name
            child.mkdir(mode=0o700, exist_ok=True)
            child.chmod(0o700)
        self._session_workdirs[chat_id] = (generation, root)
        return root

    # -- credentials --------------------------------------------------------

    def _lease_client(self, client: AsyncOpenAI) -> AsyncOpenAI:
        key = id(client)
        self._client_leases[key] = self._client_leases.get(key, 0) + 1
        return client

    async def _ensure_client(self, *, force_refresh: bool = False) -> AsyncOpenAI:
        async with self._auth_lock:
            if self._client is not None and not force_refresh:
                # The access token is a short-lived JWT; check before every call
                # rather than discovering the expiry as a 401 mid-answer.
                assert self._credentials is not None
                if not auth.access_token_is_expiring(self._credentials.access_token):
                    return self._lease_client(self._client)

            credentials = await asyncio.to_thread(
                auth.resolve_credentials,
                self._config.credentials_path,
                fallback_path=self._config.main_credentials_path,
                force_refresh=force_refresh,
            )
            if self._client is None or credentials.access_token != (
                self._credentials.access_token if self._credentials else None
            ):
                replacement = codex_client.build_client(
                    credentials, timeout_seconds=self._config.request_timeout_seconds
                )
                old_client = self._client
                self._credentials = credentials
                self._client = replacement
                if old_client is not None:
                    old_key = id(old_client)
                    if self._client_leases.get(old_key, 0) > 0:
                        self._retired_clients[old_key] = old_client
                    else:
                        await self._close_client(old_client)
            return self._lease_client(self._client)

    @staticmethod
    async def _close_client(client: Any) -> None:
        try:
            await client.close()
        except Exception:  # noqa: BLE001 - retirement must not fail a turn
            logger.debug("Closing a Codex client failed", exc_info=True)

    async def _release_client(self, client: Any) -> None:
        key = id(client)
        async with self._auth_lock:
            count = self._client_leases.get(key, 0)
            if count <= 0:
                return
            if count > 1:
                self._client_leases[key] = count - 1
                return
            self._client_leases.pop(key, None)
            retired = self._retired_clients.get(key)
            if retired is not None:
                await self._close_client(retired)
                self._retired_clients.pop(key, None)

    async def close(self) -> None:
        """Close every resident Codex pool owned by this Agent."""

        owned = list(self._owned_tasks)
        for task in owned:
            task.cancel()
        if owned:
            _done, pending = await asyncio.wait(
                owned,
                timeout=COSMETIC_CLEANUP_TIMEOUT_SECONDS,
            )
            if pending:
                logger.warning(
                    "%d Agent-owned task(s) ignored shutdown cancellation",
                    len(pending),
                )
        async with self._auth_lock:
            clients = list(self._retired_clients.values())
            if self._client is not None and all(
                client is not self._client for client in clients
            ):
                clients.append(self._client)
            self._retired_clients.clear()
            self._client_leases.clear()
            self._client = None
            self._credentials = None
        for client in clients:
            await self._close_client(client)

    def _observe_task(self, task: asyncio.Task) -> None:
        if task.done():
            self._finish_owned_task(task)
            return
        self._owned_tasks.add(task)
        task.add_done_callback(self._finish_owned_task)

    def _finish_owned_task(self, task: asyncio.Task) -> None:
        self._owned_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

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
        if chat_id in self._history:
            self._restored.add(chat_id)
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
            logger.info(
                "Picked up %d stored turns for %s",
                len(stored),
                identity_pseudonym(chat_id, "session"),
            )
        # A failed durable read must remain retryable. Marking this chat before
        # load/build succeeded would make the next message silently continue
        # from empty process memory after a transient or corrupt-history error.
        self._restored.add(chat_id)

    def _history_limit(self) -> int:
        """Turns kept per chat — a question and its answer count as two."""
        return max(2, self._config.history_turns * 2)

    def _clear_live_session(self, chat_id: str) -> None:
        """Clear every process-local value owned by one conversation."""

        self._history.pop(chat_id, None)
        self._ready_turns.pop(chat_id, None)
        self._stopped_turns.pop(chat_id, None)
        self._tool_state.pop(chat_id, None)
        self._session_instructions.pop(chat_id, None)
        self._session_workdirs.pop(chat_id, None)
        # The durable boundary already selected an empty current session.
        self._restored.add(chat_id)

    async def forget(self, chat_id: str) -> bool:
        """Drop a conversation's history.

        Refuse immediately while a turn or delivery fence owns the chat. This
        keeps a later /stop reachable on channels that serialize controls.
        """

        execution = self._active_executions.get(chat_id)
        if execution is not None and not execution.finished.is_set():
            return False
        completion = self._completion_fences.get(chat_id)
        if completion is not None and not completion.is_set():
            return False

        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        if lock.locked():
            return False
        self._approvals.block(chat_id)
        try:
            async with lock:
                execution = self._active_executions.get(chat_id)
                if execution is not None and not execution.finished.is_set():
                    return False
                completion = self._completion_fences.get(chat_id)
                if completion is not None and not completion.is_set():
                    return False
                # Record the boundary before changing live state. If this fails,
                # /new reports failure and the old conversation remains intact.
                await asyncio.to_thread(self._store.new_session, chat_id)
                self._clear_live_session(chat_id)
                return True
        finally:
            self._approvals.unblock(chat_id)

    def _begin_completion_fence(self, chat_id: str) -> None:
        current = self._completion_fences.get(chat_id)
        if current is not None and not current.is_set():
            raise ConversationError(
                "The previous answer is still completing its delivery fence"
            )
        self._completion_fences[chat_id] = asyncio.Event()

    def _release_completion_fence(self, chat_id: str) -> None:
        completion = self._completion_fences.pop(chat_id, None)
        if completion is not None:
            completion.set()

    def resolve_approval(
        self, chat_id: str, *, approved: bool, reason: str = ""
    ) -> bool:
        """Resolve this conversation's oldest live approval request."""

        return self._approvals.resolve(
            chat_id, approved=approved, reason=reason
        )

    def _register_execution(
        self,
        chat_id: str,
        session: int,
        *,
        claim_ids: Sequence[str] = (),
    ) -> _ActiveExecution:
        current = self._active_executions.get(chat_id)
        if current is not None:
            if not current.finished.is_set():
                raise ConversationError("This conversation already has an active owner")
            self._active_executions.pop(chat_id, None)
        execution = _ActiveExecution(
            session=session,
            claim_ids=tuple(dict.fromkeys(str(value) for value in claim_ids if value)),
        )
        self._active_executions[chat_id] = execution
        return execution

    @asynccontextmanager
    async def prepare_turn(
        self,
        chat_id: str,
        *,
        before_stop: Optional[PreStopFence] = None,
    ) -> AsyncIterator[_ActiveExecution]:
        """Expose accepted preprocessing to the same exact-session stop owner."""

        execution = self._register_execution(
            chat_id,
            0,
        )
        owner = asyncio.current_task()
        execution.owner_task = owner
        execution.task = owner
        execution.before_preparation_stop = before_stop
        execution.preparing = True
        try:
            yield execution
        except TurnStopped:
            raise
        finally:
            if self._active_executions.get(chat_id) is execution:
                self._release_execution(chat_id, execution)

    async def run_preparation_step(
        self,
        chat_id: str,
        execution: _ActiveExecution,
        run: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Bound an external preprocessing child without abandoning its owner."""

        self._require_live_execution(chat_id, execution)
        owner = asyncio.current_task()
        if owner is None or execution.owner_task is not owner:
            raise ConversationError("The prepared turn no longer owns this task")
        task = asyncio.create_task(run())
        try:
            result = await self._await_execution_child(execution, task)
            await self._stop_barrier(execution)
            self._require_live_execution(chat_id, execution)
            return result
        finally:
            if execution.task is task:
                execution.task = owner

    async def preparation_stop_barrier(
        self,
        chat_id: str,
        execution: _ActiveExecution,
    ) -> None:
        """Reject late preprocessing before it can emit or begin the turn."""

        await self._stop_barrier(execution)
        self._require_live_execution(chat_id, execution)

    def _release_execution(
        self,
        chat_id: str,
        execution: _ActiveExecution,
    ) -> None:
        if self._active_executions.get(chat_id) is execution:
            self._active_executions.pop(chat_id, None)
        execution.finished.set()

    async def _stop_barrier(self, execution: _ActiveExecution) -> None:
        """Let an in-flight durable stop win or lose before completion."""

        stop_task = execution.stop_task
        if stop_task is None:
            return
        try:
            outcome = await asyncio.shield(stop_task)
        except Exception:
            # Persistence failure leaves the exact owner authoritative. The
            # command reports the error; ordinary work may continue safely.
            return
        if outcome is not None and outcome.status in {
            StopStatus.STOPPED,
            StopStatus.UNKNOWN,
        }:
            raise TurnStopped(outcome)

    async def _await_execution_child(
        self,
        execution: _ActiveExecution,
        task: asyncio.Task,
    ) -> Any:
        """Race one cooperative child against the authoritative stop result."""

        execution.task = task
        self._observe_task(task)
        stop_wait = asyncio.create_task(execution.stop_ready.wait())
        self._observe_task(stop_wait)
        try:
            await self._stop_barrier(execution)
            done, _pending = await asyncio.wait(
                {task, stop_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if task in done:
                try:
                    return task.result()
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is None or not current.cancelling():
                        await self._stop_barrier(execution)
                    raise
            await self._stop_barrier(execution)
            return await task
        finally:
            stop_wait.cancel()
            if not task.done():
                task.cancel()

    async def _run_owned_turn(
        self,
        chat_id: str,
        execution: _ActiveExecution,
        run: Awaitable[TurnResult],
        on_notice: Optional[Notice],
    ) -> TurnResult:
        """Run the exact cancellable child while its cosmetic heartbeat lives."""

        task = asyncio.create_task(run)
        guarded_notice = self._guard_execution_notice(
            chat_id,
            execution,
            on_notice,
        )
        working_notice_task = asyncio.create_task(
            self._working_notice_loop(chat_id, guarded_notice)
        )
        self._observe_task(working_notice_task)
        execution.notice_task = working_notice_task
        try:
            return await self._await_execution_child(execution, task)
        finally:
            working_notice_task.cancel()
            if not working_notice_task.done():
                await asyncio.wait(
                    {working_notice_task},
                    timeout=COSMETIC_CLEANUP_TIMEOUT_SECONDS,
                )
            if execution.notice_task is working_notice_task:
                execution.notice_task = None

    def _guard_execution_notice(
        self,
        chat_id: str,
        execution: _ActiveExecution,
        on_notice: Optional[Notice],
    ) -> Optional[Notice]:
        if on_notice is None:
            return None

        async def guarded(text: str, replace_id: str = "") -> Any:
            outcome = execution.outcome
            if (
                self._active_executions.get(chat_id) is not execution
                or (
                    outcome is not None
                    and outcome.status in {StopStatus.STOPPED, StopStatus.UNKNOWN}
                )
            ):
                return False
            return await on_notice(text, replace_id)

        return guarded

    async def stop(self, chat_id: str) -> StopOutcome:
        """Durably stop only the exact live owner for this conversation."""

        execution = self._active_executions.get(chat_id)
        if execution is not None and execution.finished.is_set():
            execution = None
        if execution is None:
            completion = self._completion_fences.get(chat_id)
            if completion is not None and not completion.is_set():
                return StopOutcome(StopStatus.TOO_LATE)
            return StopOutcome(StopStatus.NOT_RUNNING)
        transition = execution.stop_task
        if transition is not None and transition.done():
            try:
                failed = transition.cancelled() or transition.exception() is not None
            except asyncio.CancelledError:
                failed = True
            if failed:
                transition = None
        if transition is None:
            execution.outcome = None
            execution.stop_ready.clear()
            transition = asyncio.create_task(
                self._run_stop_transition(chat_id, execution)
            )
            execution.stop_task = transition
            self._observe_task(transition)
        return await asyncio.shield(transition)

    async def _run_stop_transition(
        self,
        chat_id: str,
        execution: _ActiveExecution,
    ) -> StopOutcome:
        """Publish one durable stop result and cancel its exact owner atomically."""

        try:
            async with execution.transition_lock:
                if execution.finished.is_set() and execution.preparing:
                    outcome = StopOutcome(StopStatus.NOT_RUNNING)
                elif execution.outcome is not None:
                    outcome = execution.outcome
                elif execution.preparing:
                    before_stop = execution.before_preparation_stop
                    if before_stop is not None:
                        await before_stop()
                    outcome = StopOutcome(StopStatus.STOPPED)
                else:
                    checkpoint = await asyncio.to_thread(
                        self._store.request_stop,
                        chat_id,
                        t("runtime.stopped", self._config.language),
                        stopped_after_actions_text=t(
                            "runtime.stopped_after_actions",
                            self._config.language,
                        ),
                        expected_session=execution.session,
                    )
                    if checkpoint is None:
                        outcome = StopOutcome(StopStatus.NOT_RUNNING)
                    elif checkpoint.state == "stopped":
                        outcome = StopOutcome(
                            StopStatus.STOPPED,
                            checkpoint,
                        )
                    elif checkpoint.state == "unknown":
                        outcome = StopOutcome(
                            StopStatus.UNKNOWN,
                            checkpoint,
                        )
                    elif checkpoint.state == "answer_ready":
                        outcome = StopOutcome(StopStatus.TOO_LATE, checkpoint)
                    else:
                        raise ConversationError(
                            f"Unsupported durable stop state: {checkpoint.state}"
                        )
                execution.outcome = outcome
            if outcome.status in {StopStatus.STOPPED, StopStatus.UNKNOWN}:
                self._approvals.clear(
                    chat_id,
                    reason="The conversation was stopped.",
                )
                task = execution.task
                if (
                    task is not None
                    and task is not execution.owner_task
                    and not task.done()
                ):
                    task.cancel()
                notice_task = execution.notice_task
                if notice_task is not None and not notice_task.done():
                    notice_task.cancel()
        except BaseException:
            # A persistence failure is not an authoritative stop outcome. Keep
            # the cooperative child waiting so a later /stop can retry safely.
            raise
        execution.stop_ready.set()
        return outcome

    async def finalize_stop(self, outcome: StopOutcome) -> None:
        """Retire a safe tombstone after the channel retires its input claim."""

        checkpoint = outcome.checkpoint
        if outcome.status != StopStatus.STOPPED or checkpoint is None:
            return
        lock = self._chat_locks.setdefault(checkpoint.chat_id, asyncio.Lock())
        async with lock:
            user_text, stopped_text = await asyncio.to_thread(
                self._store.complete_stopped_turn,
                checkpoint,
            )
            pending = self._stopped_turns.pop(checkpoint.chat_id, None)
            if pending is not None and pending[0] == checkpoint.session:
                self._remember(
                    checkpoint.chat_id,
                    pending[1],
                    pending[2],
                    pending[3],
                )
            elif user_text or stopped_text:
                self._remember(
                    checkpoint.chat_id,
                    user_text,
                    [],
                    TurnResult(text=stopped_text),
                )

    async def _working_notice_loop(
        self,
        chat_id: str,
        on_notice: Optional[Notice],
    ) -> None:
        """Send the configured generic heartbeat while a long turn is active."""

        interval = float(
            getattr(
                self._config,
                "working_notice_interval_seconds",
                180.0,
            )
        )
        if on_notice is None or interval <= 0:
            return
        text = str(
            getattr(
                self._config,
                "working_notice_text",
                "Still working.",
            )
            or ""
        ).strip()
        if not text:
            return
        started_at = time.monotonic()
        message_id = ""
        last_elapsed_bucket = -1
        delay = interval
        while True:
            await asyncio.sleep(delay)
            delay = interval
            if self._approvals.has_pending(chat_id):
                continue
            elapsed_seconds = max(0.0, time.monotonic() - started_at)
            elapsed_minutes = int(elapsed_seconds // 60)
            if elapsed_minutes == last_elapsed_bucket:
                delay = max(interval, 60.0 - (elapsed_seconds % 60.0))
                continue
            last_elapsed_bucket = elapsed_minutes
            if message_id == _NOTICE_SENT_WITHOUT_ID:
                return
            language = getattr(self._config, "language", DEFAULT_LANGUAGE)
            if elapsed_minutes < 1:
                notice_text = t(
                    "runtime.working_elapsed_under_minute",
                    language,
                    text=text,
                )
            else:
                notice_text = t(
                    "runtime.working_elapsed_minutes",
                    language,
                    text=text,
                    minutes=elapsed_minutes,
                )
            message_id = await _notify(
                on_notice,
                notice_text,
                replace_id=message_id,
            )

    # -- the turn -----------------------------------------------------------

    async def respond(
        self,
        chat_id: str,
        user_text: str,
        attachments: Sequence[media.Attachment] = (),
        on_notice: Optional[Notice] = None,
        *,
        origin: Optional[Dict[str, str]] = None,
        approval_notify: Optional[ApprovalNotice] = None,
        claim_ids: Sequence[str] = (),
        defer_completion: bool = False,
        prepared_execution: Optional[_ActiveExecution] = None,
    ) -> str:
        result = await self.respond_result(
            chat_id,
            user_text,
            attachments,
            on_notice,
            origin=origin,
            approval_notify=approval_notify,
            claim_ids=claim_ids,
            defer_completion=defer_completion,
            prepared_execution=prepared_execution,
        )
        return result.text

    async def respond_result(
        self,
        chat_id: str,
        user_text: str,
        attachments: Sequence[media.Attachment] = (),
        on_notice: Optional[Notice] = None,
        *,
        origin: Optional[Dict[str, str]] = None,
        approval_notify: Optional[ApprovalNotice] = None,
        claim_ids: Sequence[str] = (),
        defer_completion: bool = False,
        prepared_execution: Optional[_ActiveExecution] = None,
    ) -> TurnResult:
        """Return a persisted turn plus its positive terminal-completion proof."""

        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            reset_mode = getattr(self._config, "session_reset_mode", "none")
            reset = None
            if reset_mode != "none":
                reset = await asyncio.to_thread(
                    self._store.prepare_session,
                    chat_id,
                    mode=reset_mode,
                    idle_minutes=getattr(
                        self._config, "session_reset_idle_minutes", 1440
                    ),
                    at_hour=getattr(self._config, "session_reset_at_hour", 4),
                    tzinfo=timezone_for_name(
                        getattr(self._config, "timezone", "")
                    ),
                )
            reset_note = ""
            if reset is not None:
                self._clear_live_session(chat_id)
                reason_text = (
                    "the daily reset schedule"
                    if reset.reason == "daily"
                    else "inactivity"
                )
                reset_note = (
                    "[System note: The user's previous session was automatically "
                    f"reset because of {reason_text}. This is a fresh conversation "
                    "with no prior context.]"
                )
                if (
                    reset.had_activity
                    and getattr(self._config, "session_reset_notify", True)
                ):
                    reset_notice = t(
                        f"session.auto_reset_{reset.reason}",
                        getattr(
                            self._config,
                            "language",
                            DEFAULT_LANGUAGE,
                        ),
                    )
                    if prepared_execution is not None:
                        guarded_notice = self._guard_execution_notice(
                            chat_id,
                            prepared_execution,
                            on_notice,
                        )
                        await self.run_preparation_step(
                            chat_id,
                            prepared_execution,
                            lambda: _notify(
                                guarded_notice,
                                reset_notice,
                            ),
                        )
                    else:
                        await _notify(on_notice, reset_notice)
            await self._restore(chat_id)
            working_directory = await asyncio.to_thread(
                self._session_working_directory,
                chat_id,
            )
            image_manifest: List[Dict[str, Any]] = []
            if attachments:
                original_attachments = list(attachments)
                staged_attachments = await asyncio.to_thread(
                    media.stage_inbound,
                    original_attachments,
                    working_directory / "inputs",
                )
                for original, staged in zip(
                    original_attachments,
                    staged_attachments,
                ):
                    user_text = user_text.replace(
                        str(original.path.resolve(strict=False)),
                        str(staged.path),
                    )
                (
                    image_parts,
                    attached_image_paths,
                    image_manifest,
                ) = await asyncio.to_thread(
                    media.image_parts_with_manifest,
                    staged_attachments,
                    working_directory / "inputs",
                )
            else:
                image_parts, attached_image_paths = [], []
            if attached_image_paths:
                # Keep a local path handle alongside the pixels so the same
                # session input can be used for a later image edit.
                base_text = (
                    user_text.strip()
                    or "What do you see in this image?"
                )
                hints = "\n".join(
                    f"[Image attached at: {path}]"
                    for path in attached_image_paths
                )
                user_text = f"{base_text}\n\n{hints}"
            if self._memory_store is not None:
                self._memory_store.reset_consolidation_failures()
            turn_begun = False
            execution: Optional[_ActiveExecution] = prepared_execution
            owns_execution = execution is None
            try:
                if execution is not None:
                    async with execution.transition_lock:
                        if self._active_executions.get(chat_id) is not execution:
                            raise ConversationError(
                                "The prepared turn no longer owns this conversation"
                            )
                        if execution.stop_ready.is_set():
                            outcome = execution.outcome
                            if outcome is not None and outcome.status in {
                                StopStatus.STOPPED,
                                StopStatus.UNKNOWN,
                            }:
                                raise TurnStopped(outcome)
                        turn_session = await asyncio.to_thread(
                            self._store.begin_turn,
                            chat_id,
                            user_text,
                            origin=origin,
                            claim_ids=claim_ids,
                            image_manifest=image_manifest,
                        )
                        execution.session = turn_session
                        execution.claim_ids = tuple(
                            dict.fromkeys(str(value) for value in claim_ids if value)
                        )
                        execution.preparing = False
                else:
                    turn_session = await asyncio.to_thread(
                        self._store.begin_turn,
                        chat_id,
                        user_text,
                        origin=origin,
                        claim_ids=claim_ids,
                        image_manifest=image_manifest,
                    )
                turn_begun = True
                if execution is None:
                    execution = self._register_execution(
                        chat_id,
                        turn_session,
                        claim_ids=claim_ids,
                    )
                execution.owner_task = asyncio.current_task()
                result = await self._run_owned_turn(
                    chat_id,
                    execution,
                    self._run_turn(
                        chat_id,
                        user_text,
                        image_parts,
                        on_notice,
                        origin=origin,
                        approval_notify=approval_notify,
                        turn_note=reset_note,
                        working_directory=working_directory,
                        execution=execution,
                    ),
                    on_notice,
                )
                await self._stop_barrier(execution)
                checkpoints = (
                    compaction.persistent_compaction_items(result.items)
                    if self._native_compaction_active()
                    else []
                )
                async with execution.transition_lock:
                    self._require_live_execution(chat_id, execution)
                    if defer_completion:
                        self._begin_completion_fence(chat_id)
                        try:
                            await asyncio.to_thread(
                                self._store.checkpoint_answer,
                                chat_id,
                                user_text,
                                result.text,
                                checkpoints,
                                terminal_completed=result.terminal_completed,
                            )
                            self._ready_turns[chat_id] = (
                                user_text,
                                image_parts,
                                result,
                            )
                        except BaseException:
                            self._release_completion_fence(chat_id)
                            raise
                    else:
                        await asyncio.to_thread(
                            self._store.complete_turn,
                            chat_id,
                            user_text,
                            result.text,
                            checkpoints,
                        )
                if not defer_completion:
                    self._remember(chat_id, user_text, image_parts, result)
                return result
            except TurnStopped as stopped:
                outcome = stopped.outcome
                if outcome.status == StopStatus.STOPPED and outcome.checkpoint:
                    self._stopped_turns[chat_id] = (
                        outcome.checkpoint.session,
                        user_text,
                        image_parts,
                        TurnResult(
                            text=_stopped_history_text(
                                outcome,
                                self._config.language,
                            )
                        ),
                    )
                raise
            except BaseException:
                if turn_begun:
                    try:
                        await asyncio.to_thread(
                            self._store.discard_unstarted_turn,
                            chat_id,
                        )
                    except Exception:
                        logger.warning(
                            "Could not clear the unstarted turn for %s",
                            identity_pseudonym(chat_id, "session"),
                            exc_info=True,
                        )
                raise
            finally:
                if owns_execution and execution is not None:
                    self._release_execution(chat_id, execution)

    async def recover_turn(
        self,
        active: ActiveTurn,
        on_notice: Optional[Notice] = None,
        *,
        approval_notify: Optional[ApprovalNotice] = None,
        defer_completion: bool = False,
    ) -> TurnResult:
        """Resume one crash-interrupted turn without repeating durable tool work."""

        if active.phase not in {"started", "tool_completed", "answer_ready"}:
            raise ConversationError(
                "The interrupted turn has an ambiguous tool outcome and cannot resume"
            )
        if active.phase == "started":
            if active.trajectory or active.iteration:
                raise TurnRecoveryRejected(
                    "The interrupted turn's starting checkpoint is inconsistent"
                )
            resume_items: List[Dict[str, Any]] = []
        elif active.phase == "tool_completed":
            resume_items = [dict(item) for item in active.trajectory]
            _validate_completed_tool_trajectory(
                resume_items,
                expected_iterations=active.iteration,
                max_iterations=max(1, self._config.max_tool_iterations),
            )
        else:
            resume_items = []

        lock = self._chat_locks.setdefault(active.chat_id, asyncio.Lock())
        async with lock:
            execution = self._register_execution(
                active.chat_id,
                active.session,
                claim_ids=active.claim_ids,
            )
            execution.owner_task = asyncio.current_task()
            try:
                await self._stop_barrier(execution)
            except TurnStopped:
                self._release_execution(active.chat_id, execution)
                raise
            try:
                await self._restore(active.chat_id)
                working_directory = await asyncio.to_thread(
                    self._session_working_directory,
                    active.chat_id,
                )
            except BaseException:
                self._release_execution(active.chat_id, execution)
                raise
            if active.phase == "answer_ready":
                try:
                    image_parts = await asyncio.to_thread(
                        media.restore_image_parts,
                        active.image_manifest,
                        working_directory / "inputs",
                    )
                except (OSError, ValueError):
                    # The exact answer no longer needs model input. Missing old
                    # pixels may reduce later live context, but must never lose
                    # an answer that is already complete and durable.
                    logger.warning(
                        "Could not restore an answered turn's staged image",
                        exc_info=True,
                    )
                    image_parts = []
                except BaseException:
                    self._release_execution(active.chat_id, execution)
                    raise
                restored_items = [dict(item) for item in active.answer_replay]
                if restored_items:
                    restored_items.append(
                        {"role": "assistant", "content": active.answer_content}
                    )
                result = TurnResult(
                    text=active.answer_content,
                    items=restored_items,
                    terminal_completed=active.terminal_completed,
                )
            else:
                try:
                    image_parts = await asyncio.to_thread(
                        media.restore_image_parts,
                        active.image_manifest,
                        working_directory / "inputs",
                    )
                except (OSError, ValueError) as exc:
                    self._release_execution(active.chat_id, execution)
                    raise ConversationError(
                        "The interrupted turn's staged image cannot be restored safely"
                    ) from exc
                except BaseException:
                    self._release_execution(active.chat_id, execution)
                    raise
                try:
                    result = await self._run_owned_turn(
                        active.chat_id,
                        execution,
                        self._run_turn(
                            active.chat_id,
                            active.user_content,
                            image_parts,
                            on_notice,
                            origin=active.origin,
                            approval_notify=approval_notify,
                            working_directory=working_directory,
                            resume_items=resume_items,
                            completed_steps=active.iteration,
                            execution=execution,
                        ),
                        on_notice,
                    )
                    await self._stop_barrier(execution)
                except TurnStopped as stopped:
                    outcome = stopped.outcome
                    if outcome.status == StopStatus.STOPPED and outcome.checkpoint:
                        self._stopped_turns[active.chat_id] = (
                            outcome.checkpoint.session,
                            active.user_content,
                            image_parts,
                            TurnResult(
                                text=_stopped_history_text(
                                    outcome,
                                    self._config.language,
                                )
                            ),
                        )
                    self._release_execution(active.chat_id, execution)
                    raise
                except BaseException:
                    self._release_execution(active.chat_id, execution)
                    raise

            try:
                async with execution.transition_lock:
                    self._require_live_execution(active.chat_id, execution)
                    if defer_completion:
                        # Create the completion fence only once an exact answer
                        # exists and delivery is about to own it.
                        self._begin_completion_fence(active.chat_id)
                        try:
                            if active.phase != "answer_ready":
                                checkpoints = (
                                    compaction.persistent_compaction_items(result.items)
                                    if self._native_compaction_active()
                                    else []
                                )
                                await asyncio.to_thread(
                                    self._store.checkpoint_answer,
                                    active.chat_id,
                                    active.user_content,
                                    result.text,
                                    checkpoints,
                                    terminal_completed=result.terminal_completed,
                                )
                            self._ready_turns[active.chat_id] = (
                                active.user_content,
                                image_parts,
                                result,
                            )
                        except BaseException:
                            self._release_completion_fence(active.chat_id)
                            raise
                    elif active.phase == "answer_ready":
                        await asyncio.to_thread(
                            self._store.complete_ready_turn,
                            active.chat_id,
                        )
                    else:
                        checkpoints = (
                            compaction.persistent_compaction_items(result.items)
                            if self._native_compaction_active()
                            else []
                        )
                        await asyncio.to_thread(
                            self._store.complete_turn,
                            active.chat_id,
                            active.user_content,
                            result.text,
                            checkpoints,
                        )
                if not defer_completion:
                    self._remember(
                        active.chat_id,
                        active.user_content,
                        image_parts,
                        result,
                    )
                return result
            except TurnStopped as stopped:
                outcome = stopped.outcome
                if outcome.status == StopStatus.STOPPED and outcome.checkpoint:
                    self._stopped_turns[active.chat_id] = (
                        outcome.checkpoint.session,
                        active.user_content,
                        image_parts,
                        TurnResult(
                            text=_stopped_history_text(
                                outcome,
                                self._config.language,
                            )
                        ),
                    )
                raise
            except BaseException:
                raise
            finally:
                self._release_execution(active.chat_id, execution)

    async def finalize_ready_turn(self, chat_id: str) -> None:
        """Move one delivery-fenced answer into canonical conversation history."""

        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            user_text, answer, replay, terminal_completed = await asyncio.to_thread(
                self._store.complete_ready_turn,
                chat_id,
            )
            try:
                pending = self._ready_turns.pop(chat_id, None)
                if pending is not None and pending[0] == user_text:
                    self._remember(chat_id, pending[0], pending[1], pending[2])
                    return
                items = [dict(item) for item in replay]
                if items:
                    items.append({"role": "assistant", "content": answer})
                self._remember(
                    chat_id,
                    user_text,
                    [],
                    TurnResult(
                        text=answer,
                        items=items,
                        terminal_completed=terminal_completed,
                    ),
                )
            finally:
                self._release_completion_fence(chat_id)

    def _require_live_execution(
        self,
        chat_id: str,
        execution: Optional[_ActiveExecution],
    ) -> None:
        if execution is None:
            return
        outcome = execution.outcome
        if outcome is not None and outcome.status in {
            StopStatus.STOPPED,
            StopStatus.UNKNOWN,
        }:
            raise TurnStopped(outcome)
        if self._active_executions.get(chat_id) is execution:
            return
        raise ConversationError("The turn no longer owns this conversation")

    async def _run_turn(
        self,
        chat_id: str,
        user_text: str,
        image_parts: List[Dict[str, Any]],
        on_notice: Optional[Notice] = None,
        *,
        origin: Optional[Dict[str, str]] = None,
        approval_notify: Optional[ApprovalNotice] = None,
        turn_note: str = "",
        working_directory: Optional[Path] = None,
        resume_items: Sequence[Dict[str, Any]] = (),
        completed_steps: int = 0,
        execution: Optional[_ActiveExecution] = None,
    ) -> TurnResult:
        """Call the model, run what it asks for, call it again — until it answers."""
        active_working_directory = working_directory or self._context_cwd
        self._instructions_for_session(chat_id, active_working_directory)
        model_user_text = (
            f"{turn_note}\n\n{user_text}" if turn_note else user_text
        )
        history = self._build_input(chat_id, model_user_text, image_parts)

        async def request_approval(category: str, summary: str):
            return await self._approvals.request(
                chat_id, category, summary, approval_notify
            )

        context = ToolContext(
            chat_id=chat_id,
            config=self._config,
            state=self._tool_state.setdefault(chat_id, {}),
            conversation_store=self._store,
            memory_store=self._memory_store,
            cron_store=self._cron_store,
            origin=origin,
            cron_wake=self._cron_wake,
            working_directory=active_working_directory,
            allowed_skills=self._enabled_skills,
            approval_request=request_approval,
            persistence_audit=self._persistence_audit,
            persistence_writes_allowed=self._persistence_writes_enabled,
            # One opaque reference per accepted turn. Conversation sessions are
            # generations, so they cannot distinguish repeated messages.
            turn_reference=uuid.uuid4().hex,
        )
        # Everything the assistant does this turn, in order, ready to be sent
        # back on the next call and kept as history afterwards.
        items: List[Dict[str, Any]] = [dict(item) for item in resume_items]
        limit = max(1, self._config.max_tool_iterations)

        def _finish(text: str, *, terminal_completed: bool = False) -> TurnResult:
            text = text or t(
                "runtime.failure",
                getattr(self._config, "language", DEFAULT_LANGUAGE),
            )
            isolated = getattr(
                self._config, "session_isolated_workspaces", False
            )
            if isolated:
                outbound_roots = (
                    active_working_directory / "exports",
                )
            else:
                outbound_roots = getattr(
                    self._config, "outbound_media_roots", None
                )
                if not outbound_roots:
                    workspace = getattr(
                        self._config,
                        "workspace_dir",
                        Path(self._config.state_dir) / "workspace",
                    )
                    outbound_roots = (workspace,)
            finished_text = _append_generated_media(
                text,
                items,
                outbound_roots,
            )
            if isolated:
                finished_text = media.confine_outbound(
                    finished_text,
                    outbound_roots,
                )
            return TurnResult(
                text=finished_text,
                items=items,
                terminal_completed=terminal_completed,
            )

        async def _next_action_or_answer(
            offered_tools: Optional[List[Dict[str, Any]]],
        ) -> Optional[codex_stream.StreamResult]:
            for attempt in range(1, MAX_CODEX_INCOMPLETE_RESPONSES + 1):
                self._require_live_execution(chat_id, execution)
                try:
                    result = await self._call_model(
                        chat_id, history + items, offered_tools
                    )
                    self._require_live_execution(chat_id, execution)
                except (
                    OpenAIError,
                    httpx.TransportError,
                    codex_stream.CodexStreamError,
                    auth.AuthError,
                ):
                    logger.warning(
                        "The model request failed for %s",
                        identity_pseudonym(chat_id, "session"),
                        exc_info=True,
                    )
                    failure_key = (
                        "runtime.interrupted_unknown"
                        if any(
                            item.get("type") == "function_call_output"
                            for item in items
                        )
                        else "runtime.failure"
                    )
                    return codex_stream.StreamResult(
                        text=t(
                            failure_key,
                            getattr(
                                self._config,
                                "language",
                                DEFAULT_LANGUAGE,
                            ),
                        ),
                        terminal_completed=False,
                    )
                assistant_items = _assistant_items(result)
                items.extend(assistant_items)
                if compaction.has_compaction_checkpoint(assistant_items):
                    reset_skill_view_dedup(context)
                # Tool calls take precedence: commentary commonly introduces a
                # tool and must be replayed, but it must not delay execution.
                if result.tool_calls or not result.needs_continuation:
                    return result
                if attempt < MAX_CODEX_INCOMPLETE_RESPONSES:
                    logger.info(
                        "Codex response for %s paused after commentary; continuing (%d/%d)",
                        identity_pseudonym(chat_id, "session"),
                        attempt,
                        MAX_CODEX_INCOMPLETE_RESPONSES,
                    )
            return None

        for step in range(max(0, int(completed_steps)), limit):
            result = await _next_action_or_answer(self._tools)
            if result is None:
                logger.warning(
                    "Codex response for %s remained incomplete after %d attempts",
                    identity_pseudonym(chat_id, "session"),
                    MAX_CODEX_INCOMPLETE_RESPONSES,
                )
                return _finish(CODEX_INCOMPLETE_RESPONSE)
            if result.terminal_completed is False:
                logger.warning(
                    "Codex response for %s ended without positive completion proof",
                    identity_pseudonym(chat_id, "session"),
                )
                return _finish(result.text or CODEX_INCOMPLETE_RESPONSE)
            if not result.tool_calls:
                return _finish(
                    result.text,
                    terminal_completed=result.terminal_completed is True,
                )

            names = ", ".join(call["name"] for call in result.tool_calls)
            logger.info(
                "Step %d for %s: %s",
                step + 1,
                identity_pseudonym(chat_id, "session"),
                names,
            )
            step_started_at = time.monotonic()
            if execution is None:
                await asyncio.to_thread(
                    self._store.checkpoint_turn,
                    chat_id,
                    user_text,
                    items,
                    phase="tool_requested",
                    iteration=step + 1,
                )
            else:
                async with execution.transition_lock:
                    self._require_live_execution(chat_id, execution)
                    await asyncio.to_thread(
                        self._store.checkpoint_turn,
                        chat_id,
                        user_text,
                        items,
                        phase="tool_requested",
                        iteration=step + 1,
                        expected_session=execution.session,
                        expected_claim_ids=execution.claim_ids,
                    )
            self._require_live_execution(chat_id, execution)
            outputs = await run_calls(
                self._registry,
                result.tool_calls,
                context,
                allowed_groups=self._tool_groups,
                max_result_chars=self._config.max_tool_result_chars,
                step_budget_chars=self._config.max_tool_step_chars,
            )
            self._require_live_execution(chat_id, execution)
            for call, output in zip(result.tool_calls, outputs):
                if isinstance(output, str):
                    output = frame_untrusted_tool_result(
                        str(call.get("name") or ""),
                        output,
                    )
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": responses_tool_output(output),
                    }
                )
            if execution is None:
                await asyncio.to_thread(
                    self._store.checkpoint_turn,
                    chat_id,
                    user_text,
                    items,
                    phase="tool_completed",
                    iteration=step + 1,
                )
            else:
                async with execution.transition_lock:
                    self._require_live_execution(chat_id, execution)
                    await asyncio.to_thread(
                        self._store.checkpoint_turn,
                        chat_id,
                        user_text,
                        items,
                        phase="tool_completed",
                        iteration=step + 1,
                        expected_session=execution.session,
                        expected_claim_ids=execution.claim_ids,
                    )
            self._require_live_execution(chat_id, execution)
            logger.info(
                "Step %d completed for %s in %.3fs: %s",
                step + 1,
                identity_pseudonym(chat_id, "session"),
                time.monotonic() - step_started_at,
                names,
            )

        # Out of steps. Ask for what it has rather than sending nothing: the
        # work is already done and paid for, and the person is still waiting.
        logger.warning(
            "Chat %s used all %d tool steps; asking for a summary",
            identity_pseudonym(chat_id, "session"),
            limit,
        )
        items.append({"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST})
        result = await _next_action_or_answer(None)
        if result is None:
            return _finish(CODEX_INCOMPLETE_RESPONSE)
        return _finish(
            result.text,
            terminal_completed=result.terminal_completed is True,
        )

    async def _call_model(
        self,
        chat_id: str,
        input_items: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
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
        replay_retried = False
        stream_attempt = 0
        session_label = identity_pseudonym(chat_id, "session")
        while True:
            stream_attempt += 1
            logger.info(
                "Model stream starting for %s "
                "(attempt=%d, model=%s, estimated_context_tokens=%d, "
                "input_items=%d, tools=%d, first_event_timeout=%.1fs, "
                "quiet_timeout=%.1fs)",
                session_label,
                stream_attempt,
                self._config.model,
                codex_stream.estimate_context_tokens(request),
                len(request.get("input") or []),
                len(request.get("tools") or []),
                ttfb_timeout,
                idle_timeout,
            )
            try:
                result = await self._stream_once(
                    request,
                    force_refresh=force_refresh,
                    ttfb_timeout=ttfb_timeout,
                    idle_timeout=idle_timeout,
                )
                timing = result.timing
                logger.info(
                    "Model stream completed for %s "
                    "(attempt=%d, elapsed=%s, first_event=%s, events=%s, "
                    "max_event_gap=%s, status=%s, terminal=%s, "
                    "tool_calls=%d, text_chars=%d)",
                    session_label,
                    stream_attempt,
                    _seconds(timing.elapsed_seconds if timing else None),
                    _seconds(timing.first_event_seconds if timing else None),
                    timing.event_count if timing else "-",
                    _seconds(timing.max_event_gap_seconds if timing else None),
                    result.status or "-",
                    result.terminal_completed,
                    len(result.tool_calls),
                    len(result.text),
                )
                return result
            except APIStatusError as exc:
                request_input = request.get("input")
                if (
                    not replay_retried
                    and _is_masked_codex_replay_rejection(exc)
                    and compaction.has_opaque_replay(request_input)
                ):
                    request = dict(request)
                    request["input"] = compaction.strip_opaque_replay(
                        request_input
                    )
                    replay_retried = True
                    force_refresh = False
                    logger.warning(
                        "Codex rejected opaque replay for %s; stripping it and retrying once",
                        identity_pseudonym(chat_id, "session"),
                    )
                    continue
                if exc.status_code not in (401, 403) or refreshed:
                    raise
                # The token went stale between the expiry check and the request.
                logger.info(
                    "Codex returned %s; refreshing credentials and retrying once", exc.status_code
                )
                force_refresh = True
                refreshed = True
            except (
                codex_stream.CodexStreamTimeout,
                APIConnectionError,
                httpx.TransportError,
            ) as exc:
                if isinstance(exc, codex_stream.CodexStreamTimeout):
                    timing = exc.timing
                    logger.warning(
                        "Model stream timed out for %s "
                        "(attempt=%d, code=%s, elapsed=%s, first_event=%s, "
                        "events=%s, max_event_gap=%s, silence=%s)",
                        session_label,
                        stream_attempt,
                        exc.code or "-",
                        _seconds(timing.elapsed_seconds if timing else None),
                        _seconds(timing.first_event_seconds if timing else None),
                        timing.event_count if timing else "-",
                        _seconds(timing.max_event_gap_seconds if timing else None),
                        _seconds(timing.last_event_gap_seconds if timing else None),
                    )
                if reconnects >= MAX_STREAM_RECONNECTS:
                    raise
                reconnects += 1
                # The credentials are not the problem here; keep the ones we have.
                force_refresh = False
                logger.warning(
                    "%s Dropping the Codex stream and reconnecting (%d/%d).",
                    exc,
                    reconnects,
                    MAX_STREAM_RECONNECTS,
                )
                # A reconnect is an internal recovery attempt, not a message
                # for the person using the channel. Long-running turns still
                # use the configured generic heartbeat.

    async def _stream_once(
        self,
        request: Dict[str, Any],
        *,
        force_refresh: bool,
        ttfb_timeout: float,
        idle_timeout: float,
    ) -> codex_stream.StreamResult:
        client = await self._ensure_client(force_refresh=force_refresh)
        try:
            wire_request = codex_stream._bypass_sdk_request_transform(request)
            # Admission in responses.create() and the first SSE event consume
            # one wall-clock budget. Client acquisition is separate setup.
            request_started_at = time.monotonic()
            create_stream = client.responses.create(**wire_request, stream=True)
            try:
                if ttfb_timeout > 0:
                    stream = await asyncio.wait_for(create_stream, timeout=ttfb_timeout)
                else:
                    stream = await create_stream
            except asyncio.TimeoutError:
                elapsed = max(0.0, time.monotonic() - request_started_at)
                raise codex_stream.CodexStreamTimeout(
                    f"Codex stream produced no bytes within {ttfb_timeout:g}s.",
                    code="codex_stream_no_first_byte",
                    timing=codex_stream.StreamTiming(
                        elapsed_seconds=elapsed,
                        first_event_seconds=None,
                        event_count=0,
                        max_event_gap_seconds=0.0,
                        last_event_gap_seconds=elapsed,
                    ),
                ) from None
            try:
                return await codex_stream.consume_stream(
                    stream,
                    ttfb_timeout=ttfb_timeout,
                    idle_timeout=idle_timeout,
                    started_at=request_started_at,
                )
            finally:
                # Closing the response is what actually lets go of a wedged
                # connection, and it must not replace the error that got us here.
                try:
                    await asyncio.wait_for(
                        stream.close(),
                        timeout=STREAM_CLOSE_TIMEOUT_SECONDS,
                    )
                except Exception:  # noqa: BLE001 - the turn already has its outcome
                    logger.debug("Closing the Codex stream failed", exc_info=True)
        finally:
            await self._release_client(client)


async def _notify(
    on_notice: Optional[Notice],
    text: str,
    *,
    replace_id: str = "",
) -> str:
    """Tell the person waiting what is happening. Never at the cost of the turn."""
    if on_notice is None:
        return replace_id
    try:
        result = await on_notice(text, replace_id)
        message_id = str(getattr(result, "message_id", "") or "")
        if message_id:
            return message_id
        if result is None or bool(result):
            return replace_id if replace_id else _NOTICE_SENT_WITHOUT_ID
        if bool(getattr(result, "retryable", False)):
            return replace_id
        return _NOTICE_SENT_WITHOUT_ID
    except Exception:  # noqa: BLE001 - a failed notice must not fail the answer
        logger.debug("Could not deliver the notice %r", text, exc_info=True)
        return _NOTICE_SENT_WITHOUT_ID


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


def _validate_completed_tool_trajectory(
    items: Sequence[Dict[str, Any]],
    *,
    expected_iterations: int,
    max_iterations: int,
) -> None:
    """Accept only exact, ordered call/result rounds emitted by Pilotage."""

    if not 1 <= expected_iterations <= max_iterations:
        raise TurnRecoveryRejected("The interrupted turn's iteration is invalid")
    pending: List[str] = []
    seen: set[str] = set()
    batches = 0
    batch_outputs_started = False
    for item in items:
        if not isinstance(item, dict):
            raise TurnRecoveryRejected(
                "The interrupted turn's completed-tool checkpoint is malformed"
            )
        kind = item.get("type")
        if kind == "function_call":
            call_id = item.get("call_id")
            if (
                set(item) != {"type", "call_id", "name", "arguments"}
                or not isinstance(call_id, str)
                or not call_id
                or call_id in seen
                or not isinstance(item.get("name"), str)
                or not item.get("name")
                or not isinstance(item.get("arguments"), str)
            ):
                raise TurnRecoveryRejected(
                    "The interrupted turn's completed-tool checkpoint is malformed"
                )
            if pending and batch_outputs_started:
                raise TurnRecoveryRejected(
                    "The interrupted turn's tool calls and results are out of order"
                )
            if not pending:
                batches += 1
                batch_outputs_started = False
            pending.append(call_id)
            seen.add(call_id)
            continue
        if kind == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output")
            if (
                set(item) != {"type", "call_id", "output"}
                or not pending
                or call_id != pending[0]
                or not _valid_recovered_tool_output(output)
            ):
                raise TurnRecoveryRejected(
                    "The interrupted turn's tool calls and results do not match"
                )
            batch_outputs_started = True
            pending.pop(0)
            continue
        if pending:
            raise TurnRecoveryRejected(
                "The interrupted turn's tool calls and results are out of order"
            )
        if kind == "compaction":
            if set(item) != {"type", "encrypted_content"} or not isinstance(
                item.get("encrypted_content"), str
            ) or not item.get("encrypted_content"):
                raise TurnRecoveryRejected(
                    "The interrupted turn's compaction checkpoint is malformed"
                )
            continue
        if kind == "reasoning":
            if (
                not set(item).issubset(
                    {"type", "encrypted_content", "summary", "status"}
                )
                or not isinstance(item.get("encrypted_content"), str)
                or not item.get("encrypted_content")
                or (
                    "status" in item
                    and not isinstance(item.get("status"), str)
                )
                or (
                    "summary" in item
                    and (
                        not isinstance(item.get("summary"), list)
                        or not all(
                            isinstance(part, dict)
                            and set(part) == {"type", "text"}
                            and part.get("type") == "summary_text"
                            and isinstance(part.get("text"), str)
                            for part in item["summary"]
                        )
                    )
                )
            ):
                raise TurnRecoveryRejected(
                    "The interrupted turn's reasoning checkpoint is malformed"
                )
            continue
        if kind == "message":
            if codex_stream.message_items_for_replay([item]) != [item]:
                raise TurnRecoveryRejected(
                    "The interrupted turn's assistant message is malformed"
                )
            continue
        if (
            kind is None
            and set(item) == {"role", "content"}
            and item.get("role") == "assistant"
            and isinstance(item.get("content"), str)
        ):
            continue
        raise TurnRecoveryRejected(
            "The interrupted turn contains an unsupported replay item"
        )
    if pending or not seen or batches != expected_iterations:
        raise TurnRecoveryRejected(
            "The interrupted turn's tool calls and results do not match"
        )


def _valid_recovered_tool_output(output: Any) -> bool:
    if isinstance(output, str):
        return True
    if not isinstance(output, list):
        return False
    for part in output:
        if not isinstance(part, dict):
            return False
        if part.get("type") == "input_text":
            if set(part) != {"type", "text"} or not isinstance(
                part.get("text"), str
            ):
                return False
            continue
        if part.get("type") == "input_image":
            if (
                not {"type", "image_url"}.issubset(part)
                or not set(part).issubset({"type", "image_url", "detail"})
                or not isinstance(part.get("image_url"), str)
                or not part.get("image_url")
                or (
                    "detail" in part
                    and not isinstance(part.get("detail"), str)
                )
            ):
                return False
            continue
        return False
    return True


def _append_generated_media(
    text: str,
    items: List[Dict[str, Any]],
    roots: Sequence[Path],
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

    existing, _ = media.extract_outbound(text or "", roots)
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
            roots,
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
