"""Minimal provenance and rollback for foreground memory and skill changes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, Dict, Iterable, Iterator, Mapping, Optional

from .redact import redact_channel_identities, redact_sensitive_text


MAX_REASON_CHARS = 240
MAX_COMMITTED_EVENTS = 500


SCHEMA = """
CREATE TABLE IF NOT EXISTS persistence_events (
    event_id         TEXT PRIMARY KEY,
    status           TEXT NOT NULL CHECK (status IN ('prepared', 'committed')),
    prepared_at      REAL NOT NULL,
    committed_at     REAL,
    category         TEXT NOT NULL DEFAULT '',
    operation        TEXT NOT NULL,
    turn_ref         TEXT NOT NULL,
    change_reason    TEXT NOT NULL,
    reverts_event_id TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS one_prepared_persistence_event
ON persistence_events (status) WHERE status = 'prepared';

CREATE TABLE IF NOT EXISTS persistence_targets (
    event_id       TEXT NOT NULL REFERENCES persistence_events(event_id) ON DELETE CASCADE,
    path           TEXT NOT NULL,
    before_exists  INTEGER NOT NULL CHECK (before_exists IN (0, 1)),
    before_sha256  TEXT,
    before_bytes   BLOB,
    before_mode    INTEGER,
    after_exists   INTEGER CHECK (after_exists IN (0, 1)),
    after_sha256   TEXT,
    PRIMARY KEY (event_id, path)
);
"""


class PersistenceAuditError(RuntimeError):
    """The private persistence boundary could not complete safely."""


class PersistenceChangeRejected(PersistenceAuditError):
    """A persistent change was refused and its prior bytes were restored."""


@dataclass(frozen=True)
class _FileState:
    exists: bool
    digest: Optional[str]
    content: Optional[bytes]
    mode: Optional[int]


def build_persistence_policy(*, memory: bool, skills: bool) -> str:
    """Return one compact judgment policy for the enabled write surface."""

    if not memory and not skills:
        return ""
    if memory and skills:
        eligibility = (
            "when the user explicitly asks or states a durable fact, clearly corrects "
            "one, or the same preference or procedure is demonstrated across distinct "
            "tasks"
        )
        existing_home = "entry or skill"
    elif memory:
        eligibility = (
            "when the user explicitly asks to remember or states a durable personal "
            "fact, clearly corrects one, or the same preference is demonstrated "
            "across distinct tasks"
        )
        existing_home = "entry"
    else:
        eligibility = (
            "when the user explicitly asks to create or update a skill, clearly "
            "corrects an existing skill, or the same reusable procedure is "
            "demonstrated across distinct tasks"
        )
        existing_home = "skill"
    homes = []
    if memory:
        homes.append("memory for durable personal facts, preferences, and constraints")
    if skills:
        homes.append("skills for reusable task procedures")
    tools = []
    if memory:
        tools.append("memory")
    if skills:
        tools.append("file")
    return (
        "## Persistent learning\n"
        "No change is the default. In a foreground conversation only, persist "
        + eligibility
        + ". Inspect the live target first and update, merge, or remove before "
        "adding; create only when no existing "
        + existing_home
        + " is the right home. Use "
        + " and ".join(homes)
        + ". Change or remove only what the evidence supersedes. Never persist "
        "guesses, generic or rediscoverable facts, one-off task "
        "state, outputs, logs, or temporary workflows. Use only the canonical "
        + " and ".join(tools)
        + " tools and include a short factual change_reason."
    )


def normalize_change_reason(value: Any) -> str:
    text = "" if value is None else str(value)
    text = "".join(char for char in text if char >= " " or char == "\t")
    text = re.sub(r"\s+", " ", text).strip()
    return redact_channel_identities(redact_sensitive_text(text))[:MAX_REASON_CHARS]


def change_reason_error(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return "change_reason must be a short factual string."
    if len(value) > MAX_REASON_CHARS:
        return f"change_reason cannot exceed {MAX_REASON_CHARS} characters."
    if not normalize_change_reason(value):
        return (
            "Persistent memory or skill changes require change_reason: one short "
            "factual sentence naming the durable evidence, not private reasoning."
        )
    return None


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _tool_base(context: Any) -> Path:
    terminal = getattr(context, "state", {}).get("terminal")
    shell = getattr(terminal, "shell", None)
    if getattr(shell, "cwd", None):
        return Path(str(shell.cwd)).expanduser()
    if getattr(context, "working_directory", None) is not None:
        return Path(context.working_directory).expanduser()
    settings = getattr(context.config, "settings", None)
    if settings is not None:
        configured = str(settings.text("terminal.cwd", "") or "").strip()
        if configured:
            return Path(configured).expanduser()
    return Path(context.config.state_dir) / "workspace"


def _skill_target(value: Any, context: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(os.path.expanduser(value.strip()))
    if not candidate.is_absolute():
        candidate = _tool_base(context) / candidate
    lexical = Path(os.path.abspath(str(candidate)))
    root = Path(context.config.state_dir).expanduser() / "skills"
    lexical_root = Path(os.path.abspath(str(root)))
    try:
        resolved = candidate.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PersistenceChangeRejected(
            "The requested skill path could not be resolved safely."
        ) from exc
    lexical_skill = _within(lexical, lexical_root)
    resolved_skill = _within(resolved, resolved_root)
    if lexical_skill and not resolved_skill:
        raise PersistenceChangeRejected(
            "Skill changes cannot escape the profile skills directory through a link."
        )
    if not resolved_skill:
        return None
    relative = resolved.relative_to(resolved_root)
    if not relative.parts:
        return None
    return f"skills/{relative.as_posix()}"


def persistence_targets(
    tool_name: str, args: Mapping[str, Any], context: Any
) -> tuple[str, ...]:
    """Return the canonical files one supported mutation can affect."""

    if tool_name == "memory":
        mutates = bool(args.get("operations")) or args.get("action") in {
            "add",
            "replace",
            "remove",
        }
        if not mutates:
            return ()
        target = args.get("target") or "memory"
        if target == "memory":
            return ("memories/MEMORY.md",)
        if target == "user":
            return ("memories/USER.md",)
        return ()

    candidates: list[Any] = []
    if tool_name == "write_file":
        candidates.append(args.get("path"))
    elif tool_name == "patch" and args.get("mode", "replace") == "replace":
        candidates.append(args.get("path"))
    elif tool_name == "patch":
        patch = args.get("patch")
        if not isinstance(patch, str):
            return ()
        from .tools.patch_parser import OperationType, parse_v4a_patch

        operations, error = parse_v4a_patch(patch)
        if error:
            return ()
        for operation in operations:
            candidates.append(operation.file_path)
            if operation.operation == OperationType.MOVE and operation.new_path:
                candidates.append(operation.new_path)
    else:
        return ()

    classified = [_skill_target(candidate, context) for candidate in candidates]
    targets = {target for target in classified if target is not None}
    if targets and any(target is None for target in classified):
        raise PersistenceChangeRejected(
            "Keep skill and ordinary workspace changes in separate file-tool calls."
        )
    return tuple(sorted(targets))


def should_observe_persistence(
    tool_name: str, args: Mapping[str, Any], context: Any
) -> bool:
    return bool(persistence_targets(tool_name, args, context))


async def _finish_in_thread(
    function: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, bool]:
    """Return a worker's result and whether cancellation arrived while it ran."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            cancelled = True


class PersistenceAuditStore:
    """One profile's minimal prepared/committed persistence journal."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir).expanduser().resolve(strict=False)
        self.path = self.state_dir / "persistence-audit.db"
        self._lock = asyncio.Lock()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise PersistenceAuditError(
                "The private persistence journal cannot be a symbolic link."
            )
        connection: Optional[sqlite3.Connection] = None
        try:
            connection = sqlite3.connect(self.path, timeout=5.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA secure_delete=ON")
            connection.executescript(SCHEMA)
            target_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(persistence_targets)"
                ).fetchall()
            }
            if "before_mode" not in target_columns:
                connection.execute(
                    "ALTER TABLE persistence_targets ADD COLUMN before_mode INTEGER"
                )
            if os.name != "nt":
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
            with connection:
                yield connection
        except PersistenceAuditError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise PersistenceAuditError(
                f"Private persistence journal I/O failed: {type(exc).__name__}."
            ) from exc
        finally:
            if connection is not None:
                connection.close()

    def _target_path(self, relative: str) -> Path:
        written = str(relative or "")
        pure = PurePosixPath(written)
        if (
            "\x00" in written
            or "\\" in written
            or pure.is_absolute()
            or not pure.parts
            or ".." in pure.parts
            or pure.as_posix() != written
        ):
            raise PersistenceAuditError(f"Unsafe persistence path: {written!r}")

        if tuple(pure.parts) in {
            ("memories", "MEMORY.md"),
            ("memories", "USER.md"),
        }:
            root = self.state_dir / "memories"
        elif len(pure.parts) >= 2 and pure.parts[0] == "skills":
            root = self.state_dir / "skills"
        else:
            raise PersistenceAuditError(f"Non-canonical persistence path: {written!r}")

        candidate = self.state_dir.joinpath(*pure.parts)
        current = candidate
        while current != self.state_dir:
            if current.is_symlink():
                raise PersistenceAuditError(
                    f"Persistence path crosses a symbolic link: {written!r}"
                )
            current = current.parent
        try:
            resolved = candidate.resolve(strict=False)
            resolved_root = root.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise PersistenceAuditError(
                f"Persistence path could not be resolved: {written!r}"
            ) from exc
        if not _within(resolved, resolved_root):
            raise PersistenceAuditError(f"Persistence path escapes its root: {written!r}")
        return candidate

    def _snapshot(self, targets: Iterable[str]) -> Dict[str, _FileState]:
        states: Dict[str, _FileState] = {}
        for relative in sorted(set(targets)):
            path = self._target_path(relative)
            if not path.exists():
                states[relative] = _FileState(False, None, None, None)
                continue
            if not path.is_file() or path.is_symlink():
                raise PersistenceAuditError(
                    f"Persistent target is not a regular file: {relative}"
                )
            try:
                details = path.stat()
                content = path.read_bytes()
            except OSError as exc:
                raise PersistenceAuditError(
                    f"Could not read persistent target {relative}: {exc}"
                ) from exc
            states[relative] = _FileState(
                True,
                _digest(content),
                content,
                stat.S_IMODE(details.st_mode),
            )
        return states

    @staticmethod
    def _changed(
        before: Mapping[str, _FileState], after: Mapping[str, _FileState]
    ) -> list[str]:
        return sorted(
            path
            for path in set(before) | set(after)
            if before.get(path) is None
            or after.get(path) is None
            or before[path].exists != after[path].exists
            or before[path].content != after[path].content
        )

    @staticmethod
    def _category(paths: Iterable[str]) -> str:
        paths = tuple(paths)
        categories = []
        if any(path.startswith("memories/") for path in paths):
            categories.append("memory")
        if any(path.startswith("skills/") for path in paths):
            categories.append("skills")
        return "+".join(categories)

    def _prepare(
        self,
        *,
        operation: str,
        turn_ref: str,
        reason: str,
        before: Mapping[str, _FileState],
        reverts_event_id: Optional[str] = None,
    ) -> str:
        event_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO persistence_events "
                "(event_id, status, prepared_at, operation, turn_ref, change_reason, "
                " reverts_event_id) VALUES (?, 'prepared', ?, ?, ?, ?, ?)",
                (
                    event_id,
                    time.time(),
                    str(operation)[:80],
                    str(turn_ref)[:80],
                    reason,
                    reverts_event_id,
                ),
            )
            connection.executemany(
                "INSERT INTO persistence_targets "
                "(event_id, path, before_exists, before_sha256, before_bytes, "
                " before_mode) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    (
                        event_id,
                        path,
                        int(state.exists),
                        state.digest,
                        state.content,
                        state.mode,
                    )
                    for path, state in before.items()
                ),
            )
        return event_id

    def _discard(self, event_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM persistence_events "
                "WHERE event_id = ? AND status = 'prepared'",
                (event_id,),
            )

    def _commit(
        self,
        event_id: str,
        *,
        reason: str,
        before: Mapping[str, _FileState],
        after: Mapping[str, _FileState],
    ) -> None:
        changed = self._changed(before, after)
        with self._connect() as connection:
            placeholders = ",".join("?" for _ in changed)
            connection.execute(
                f"DELETE FROM persistence_targets WHERE event_id = ? "
                f"AND path NOT IN ({placeholders})",
                (event_id, *changed),
            )
            for path in changed:
                state = after[path]
                connection.execute(
                    "UPDATE persistence_targets SET after_exists = ?, after_sha256 = ? "
                    "WHERE event_id = ? AND path = ?",
                    (int(state.exists), state.digest, event_id, path),
                )
            cursor = connection.execute(
                "UPDATE persistence_events SET status = 'committed', committed_at = ?, "
                "category = ?, change_reason = ? "
                "WHERE event_id = ? AND status = 'prepared'",
                (time.time(), self._category(changed), reason, event_id),
            )
            if cursor.rowcount != 1:
                raise PersistenceAuditError(
                    "The prepared persistence event no longer exists."
                )
            self._prune(connection)

    @staticmethod
    def _prune(connection: sqlite3.Connection) -> None:
        old = connection.execute(
            "SELECT event_id FROM persistence_events "
            "WHERE status = 'committed' ORDER BY committed_at DESC, event_id DESC "
            "LIMIT -1 OFFSET ?",
            (MAX_COMMITTED_EVENTS,),
        ).fetchall()
        if old:
            connection.executemany(
                "DELETE FROM persistence_events WHERE event_id = ?",
                ((row["event_id"],) for row in old),
            )

    def _load_states(
        self, event_id: str
    ) -> tuple[Dict[str, _FileState], Dict[str, tuple[bool, Optional[str]]]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM persistence_targets WHERE event_id = ? ORDER BY path",
                (event_id,),
            ).fetchall()
        if not rows:
            raise PersistenceAuditError(
                f"Persistence event has no rollback targets: {event_id}"
            )
        before: Dict[str, _FileState] = {}
        after: Dict[str, tuple[bool, Optional[str]]] = {}
        for row in rows:
            relative = str(row["path"])
            self._target_path(relative)
            before_exists = bool(row["before_exists"])
            before_digest = row["before_sha256"]
            before_content = (
                bytes(row["before_bytes"])
                if row["before_bytes"] is not None
                else None
            )
            before_mode = row["before_mode"]
            if before_exists:
                if (
                    before_content is None
                    or _digest(before_content) != before_digest
                    or (
                        before_mode is not None
                        and (
                            not isinstance(before_mode, int)
                            or before_mode < 0
                            or before_mode > 0o7777
                        )
                    )
                ):
                    raise PersistenceAuditError(
                        f"Rollback bytes failed integrity validation for {relative}."
                    )
            elif (
                before_digest is not None
                or before_content is not None
                or before_mode is not None
            ):
                raise PersistenceAuditError(
                    f"Rollback state is inconsistent for {relative}."
                )
            before[relative] = _FileState(
                before_exists, before_digest, before_content, before_mode
            )

            if row["after_exists"] is not None:
                after_exists = bool(row["after_exists"])
                after_digest = row["after_sha256"]
                if after_exists != (after_digest is not None):
                    raise PersistenceAuditError(
                        f"Committed state is inconsistent for {relative}."
                    )
                after[relative] = (after_exists, after_digest)
        return before, after

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_restore(
        self, path: Path, content: bytes, *, mode: Optional[int]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=str(path.parent), prefix=".persistence-restore-", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                if mode is not None and os.name != "nt":
                    os.fchmod(handle.fileno(), mode)
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._fsync_directory(path.parent)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def _restore(self, states: Mapping[str, _FileState]) -> None:
        for relative, state in states.items():
            self._target_path(relative)
            if state.exists:
                if (
                    state.content is None
                    or _digest(state.content) != state.digest
                ):
                    raise PersistenceAuditError(
                        f"Rollback bytes failed integrity validation for {relative}."
                    )
            elif (
                state.content is not None
                or state.digest is not None
                or state.mode is not None
            ):
                raise PersistenceAuditError(
                    f"Rollback state is inconsistent for {relative}."
                )

        current = self._snapshot(states)
        for relative, state in states.items():
            path = self._target_path(relative)
            live = current[relative]
            if live.exists == state.exists and live.content == state.content:
                if (
                    os.name != "nt"
                    and state.exists
                    and state.mode is not None
                    and live.mode != state.mode
                ):
                    try:
                        os.chmod(path, state.mode)
                        with path.open("rb") as handle:
                            os.fsync(handle.fileno())
                    except OSError as exc:
                        raise PersistenceAuditError(
                            f"Could not restore persistent target {relative}: {exc}"
                        ) from exc
                continue
            try:
                if state.exists:
                    restore_mode = state.mode
                    if restore_mode is None and live.exists:
                        # Journals created before permission capture preserve the
                        # live file's mode while restoring their recorded bytes.
                        restore_mode = live.mode
                    self._atomic_restore(
                        path,
                        state.content or b"",
                        mode=restore_mode,
                    )
                elif path.exists():
                    if not path.is_file() or path.is_symlink():
                        raise OSError("persistent target became a non-file path")
                    path.unlink()
                    self._fsync_directory(path.parent)
            except OSError as exc:
                raise PersistenceAuditError(
                    f"Could not restore persistent target {relative}: {exc}"
                ) from exc

        restored = self._snapshot(states)
        if any(
            restored[path].exists != state.exists
            or restored[path].content != state.content
            or (
                os.name != "nt"
                and state.exists
                and state.mode is not None
                and restored[path].mode != state.mode
            )
            for path, state in states.items()
        ):
            raise PersistenceAuditError(
                "Restored persistent bytes do not match their prior state."
            )

    def _flush(self, targets: Iterable[str]) -> None:
        directories: set[Path] = set()
        for relative in targets:
            path = self._target_path(relative)
            directories.add(path.parent)
            if path.exists():
                if not path.is_file() or path.is_symlink():
                    raise PersistenceAuditError(
                        f"Persistent target is not a regular file: {relative}"
                    )
                try:
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
                except OSError as exc:
                    raise PersistenceAuditError(
                        f"Could not flush persistent target {relative}: {exc}"
                    ) from exc
        for directory in directories:
            try:
                self._fsync_directory(directory)
            except OSError as exc:
                raise PersistenceAuditError(
                    f"Could not flush persistence directory {directory.name}: {exc}"
                ) from exc

    @staticmethod
    def _refresh_memory(context: Any, targets: Iterable[str]) -> None:
        store = getattr(context, "memory_store", None)
        refresh = getattr(store, "refresh_live_from_disk", None)
        if not callable(refresh):
            return
        memory_targets = tuple(
            "memory" if path == "memories/MEMORY.md" else "user"
            for path in targets
            if path in {"memories/MEMORY.md", "memories/USER.md"}
        )
        if memory_targets:
            refresh(memory_targets)

    def _recover_change_sync(
        self,
        *,
        event_id: str,
        before: Mapping[str, _FileState],
        context: Any,
    ) -> None:
        self._restore(before)
        refresh_error: Optional[Exception] = None
        try:
            self._refresh_memory(context, before)
        except Exception as exc:
            refresh_error = exc
        self._discard(event_id)
        if refresh_error is not None:
            raise PersistenceAuditError(
                "Persistent bytes were restored, but live memory could not refresh."
            ) from refresh_error

    async def _recover_change(
        self,
        *,
        event_id: str,
        before: Mapping[str, _FileState],
        context: Any,
    ) -> None:
        _result, cancelled = await _finish_in_thread(
            self._recover_change_sync,
            event_id=event_id,
            before=before,
            context=context,
        )
        if cancelled:
            raise asyncio.CancelledError

    @staticmethod
    def _result_failed(result: Any) -> bool:
        if not isinstance(result, str):
            return True
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return True
        return not isinstance(payload, dict) or bool(payload.get("error")) or (
            payload.get("success") is False
        )

    async def observe(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any],
        context: Any,
        targets: Optional[tuple[str, ...]] = None,
        invoke: Callable[[], Awaitable[Any]],
    ) -> Any:
        if not context.persistence_writes_allowed:
            raise PersistenceChangeRejected(
                "Scheduled or unattended runs cannot change persistent memory or skills."
            )
        if reason_error := change_reason_error(args.get("change_reason")):
            raise PersistenceChangeRejected(reason_error)
        reason = normalize_change_reason(args.get("change_reason"))
        canonical_targets = (
            tuple(targets)
            if targets is not None
            else persistence_targets(tool_name, args, context)
        )
        if not canonical_targets:
            raise PersistenceChangeRejected(
                "The requested operation has no canonical memory or skill target."
            )
        turn_ref = str(getattr(context, "turn_reference", "") or "").strip()
        if not turn_ref:
            raise PersistenceChangeRejected(
                "The source turn could not be identified; no persistent change was attempted."
            )

        async with self._lock:
            before, cancelled = await _finish_in_thread(
                self._snapshot, canonical_targets
            )
            if cancelled:
                raise asyncio.CancelledError
            operation = tool_name
            if tool_name == "memory":
                operation += (
                    ":batch"
                    if args.get("operations")
                    else f":{str(args.get('action') or '')[:32]}"
                )
            event_id, cancelled = await _finish_in_thread(
                self._prepare,
                operation=operation,
                turn_ref=turn_ref,
                reason=reason,
                before=before,
            )
            if cancelled:
                await _finish_in_thread(self._discard, event_id)
                raise asyncio.CancelledError
            finished = False
            try:
                result = await invoke()
                after, cancelled = await _finish_in_thread(
                    self._snapshot, canonical_targets
                )
                if cancelled:
                    raise asyncio.CancelledError
                changed = self._changed(before, after)
                if not changed:
                    _discarded, cancelled = await _finish_in_thread(
                        self._discard, event_id
                    )
                    finished = True
                    if cancelled:
                        raise asyncio.CancelledError
                    return result
                if self._result_failed(result):
                    raise PersistenceChangeRejected(
                        "The canonical tool reported failure after changing persistent "
                        "state; the prior bytes were restored."
                    )
                _flushed, cancelled = await _finish_in_thread(self._flush, changed)
                if cancelled:
                    raise asyncio.CancelledError
                _committed, cancelled = await _finish_in_thread(
                    self._commit,
                    event_id,
                    reason=reason,
                    before=before,
                    after=after,
                )
                finished = True
                if cancelled:
                    raise asyncio.CancelledError
                return result
            except BaseException:
                if finished:
                    raise
                try:
                    await self._recover_change(
                        event_id=event_id,
                        before=before,
                        context=context,
                    )
                except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                    raise
                except PersistenceAuditError:
                    raise
                except BaseException as recovery_error:
                    raise PersistenceAuditError(
                        "Persistent mutation recovery failed; canonical bytes are "
                        "uncertain and require operator inspection."
                    ) from recovery_error
                raise

    def recover_prepared(self) -> int:
        """Restore the exact prior bytes of an interrupted foreground mutation."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event_id FROM persistence_events "
                "WHERE status = 'prepared' ORDER BY prepared_at, event_id"
            ).fetchall()
        recovered = 0
        for row in rows:
            event_id = str(row["event_id"])
            before, _after = self._load_states(event_id)
            self._restore(before)
            self._discard(event_id)
            recovered += 1
        return recovered

    def events(self) -> list[Dict[str, Any]]:
        """Return bounded operator metadata without rollback bytes."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM persistence_events WHERE status = 'committed' "
                "ORDER BY committed_at, event_id"
            ).fetchall()
        events: list[Dict[str, Any]] = []
        for row in rows:
            before, after = self._load_states(str(row["event_id"]))
            events.append(
                {
                    "event_id": row["event_id"],
                    "status": row["status"],
                    "prepared_at": row["prepared_at"],
                    "committed_at": row["committed_at"],
                    "category": row["category"],
                    "operation": row["operation"],
                    "turn_ref": row["turn_ref"],
                    "change_reason": row["change_reason"],
                    "reverts_event_id": row["reverts_event_id"],
                    "paths": sorted(before),
                    "before": {
                        path: {"exists": state.exists, "sha256": state.digest}
                        for path, state in before.items()
                    },
                    "after": {
                        path: {"exists": state[0], "sha256": state[1]}
                        for path, state in after.items()
                    },
                }
            )
        return events

    def rollback(self, event_id: str, *, reason: str = "Operator rollback") -> str:
        """Restore one committed event when its affected bytes have not drifted."""

        reason = normalize_change_reason(reason) or "Operator rollback"
        with self._connect() as connection:
            row = connection.execute(
                "SELECT event_id FROM persistence_events "
                "WHERE event_id = ? AND status = 'committed'",
                (event_id,),
            ).fetchone()
        if row is None:
            raise PersistenceAuditError(f"Unknown committed event: {event_id}")
        original_before, expected_after = self._load_states(event_id)
        current = self._snapshot(expected_after)
        for path, expected in expected_after.items():
            state = current[path]
            if (state.exists, state.digest) != expected:
                raise PersistenceAuditError(
                    f"Rollback refused: {path} has changed since {event_id}."
                )

        rollback_id = self._prepare(
            operation="rollback",
            turn_ref="operator",
            reason=reason,
            before=current,
            reverts_event_id=event_id,
        )
        try:
            self._restore(original_before)
            restored = self._snapshot(original_before)
            self._flush(restored)
            self._commit(
                rollback_id,
                reason=reason,
                before=current,
                after=restored,
            )
            return rollback_id
        except BaseException:
            try:
                self._restore(current)
                self._discard(rollback_id)
            except BaseException as recovery_error:
                raise PersistenceAuditError(
                    "Rollback recovery failed; canonical bytes require operator inspection."
                ) from recovery_error
            raise


__all__ = [
    "MAX_REASON_CHARS",
    "PersistenceAuditError",
    "PersistenceAuditStore",
    "PersistenceChangeRejected",
    "build_persistence_policy",
    "change_reason_error",
    "normalize_change_reason",
    "persistence_targets",
    "should_observe_persistence",
]
