"""Conversations on disk.

A WhatsApp chat is never over. Someone writes again three days later and
expects the thread to still be there. History held in the process cannot
honour that: an upgrade, a crash or a reboot of the container starts every
conversation again from nothing, and the person on the other end has no way of
knowing it happened.

So every turn is written to a small SQLite file beside the credentials. The
words and opaque native-compaction checkpoints are kept. Ordinary encrypted
reasoning is deliberately left out — it is a speed optimisation belonging to a
response that no longer exists after a restart, and replaying a stale one risks
the API refusing the whole request.
Pictures are left out for the same practical reason they are dropped from all
but the newest turn while the process runs: they are enormous next to the text.

"/new" starts a new session rather than erasing the old one, so ending a
conversation stays cheap and reversible. Optional retention later removes only
ended sessions that have passed their configured age.

Tool work is checkpointed before execution and again after its result. A write
failure therefore stops the turn instead of allowing side effects whose only
record lived in one process. ``/new`` likewise succeeds on disk before the live
conversation is cleared, or a restart could silently resurrect what the user
ended.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# One writer and no readers outside this process, so a lock is only ever held
# for the moment a turn is appended. Waiting seconds for it would mean the file
# is held by something that should not exist.
BUSY_TIMEOUT_SECONDS = 5.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT    NOT NULL,
    session    INTEGER NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    replay     TEXT    NOT NULL DEFAULT '',
    written_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_by_session ON turns (chat_id, session, id);

-- Which session a chat is currently in. A row appears the first time "/new" is
-- used; a chat that has never been reset is simply in session 1.
CREATE TABLE IF NOT EXISTS chats (
    chat_id     TEXT PRIMARY KEY,
    session     INTEGER NOT NULL,
    last_active REAL    NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS history_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One unfinished model turn per durable chat session. The exact Responses
-- trajectory is temporary: it protects tool-call/result boundaries while a
-- turn is running, then is deleted atomically with the completed text turns.
CREATE TABLE IF NOT EXISTS active_turns (
    chat_id      TEXT    NOT NULL,
    session      INTEGER NOT NULL,
    user_content TEXT    NOT NULL,
    trajectory   TEXT    NOT NULL DEFAULT '[]',
    phase        TEXT    NOT NULL DEFAULT 'started',
    updated_at   REAL    NOT NULL,
    PRIMARY KEY (chat_id, session)
);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    content,
    content='turns',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS turns_fts_insert AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS turns_fts_delete AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS turns_fts_update
AFTER UPDATE OF content ON turns
WHEN old.content IS NOT new.content
BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO turns_fts(rowid, content) VALUES (new.id, new.content);
END;
"""

# A stored turn: what was said, and by whom.
StoredTurn = Tuple[str, str]
StoredReplayTurn = Tuple[str, str, List[Dict[str, str]]]


class ConversationError(RuntimeError):
    """Canonical conversation state could not be read or written."""


class ConversationSearchError(RuntimeError):
    """The durable history could not answer a conversation-search request."""


@dataclass(frozen=True)
class SessionReset:
    """One automatic conversation boundary created before an inbound turn."""

    reason: str
    had_activity: bool


def _automatic_reset_reason(
    last_active: float,
    now: float,
    *,
    mode: str,
    idle_minutes: int,
    at_hour: int,
    tzinfo=None,
) -> Optional[str]:
    """Return Hermes' idle-first reset reason for one activity timestamp."""

    if mode == "none":
        return None
    if mode in {"idle", "both"} and now > last_active + (idle_minutes * 60):
        return "idle"
    if mode in {"daily", "both"}:
        zone = tzinfo or datetime.now().astimezone().tzinfo
        local_now = datetime.fromtimestamp(now, tz=zone)
        boundary = local_now.replace(
            hour=at_hour, minute=0, second=0, microsecond=0
        )
        if local_now.hour < at_hour:
            boundary -= timedelta(days=1)
        if datetime.fromtimestamp(last_active, tz=zone) < boundary:
            return "daily"
    return None


def _compaction_items(value: Any) -> List[Dict[str, str]]:
    return [
        {"type": "compaction", "encrypted_content": item["encrypted_content"]}
        for item in (value if isinstance(value, list) else ())
        if isinstance(item, dict)
        and item.get("type") == "compaction"
        and isinstance(item.get("encrypted_content"), str)
        and bool(item["encrypted_content"])
    ]


def _decode_replay(raw: str) -> List[Dict[str, str]]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed replay data in conversation history")
        return []
    return _compaction_items(value)


def session_workspace_path(
    workspace_root: Path, chat_id: str, session: int
) -> Path:
    """Return the private workspace owned by one durable session."""

    opaque_chat = hashlib.sha256(chat_id.encode("utf-8")).hexdigest()[:20]
    return (
        Path(workspace_root)
        / "sessions"
        / f"c-{opaque_chat}"
        / f"session-{int(session)}"
    )


class ConversationStore:
    """Every turn of every chat, in one file.

    Built without a path it keeps nothing, on purpose. `pilotage ask` is the
    one caller that wants that: it is the check you run when the agent goes
    quiet, and a check that carries yesterday's questions into today's answer
    tells you less, besides writing into the conversations of an agent that is
    still running.
    """

    def __init__(self, path: Optional[Path]):
        self._path = path
        self._schema_lock = threading.Lock()
        self._fts_error: Optional[str] = None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection for one piece of work, committed and closed after it.

        Closing matters: `sqlite3.connect` as a context manager ends the
        transaction but leaves the handle open, and a leaked handle keeps a
        lock on the file for as long as the process runs.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        try:
            with self._schema_lock:
                # Cheap against a database that already has them, and the file
                # coming back empty after a mishap then repairs itself.
                connection.executescript(SCHEMA)
                # Existing installations predate opaque compaction checkpoints.
                # SQLite's CREATE TABLE IF NOT EXISTS does not add new columns.
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(turns)")
                }
                if "replay" not in columns:
                    connection.execute(
                        "ALTER TABLE turns ADD COLUMN replay TEXT NOT NULL DEFAULT ''"
                    )
                chat_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(chats)")
                }
                if "last_active" not in chat_columns:
                    connection.execute(
                        "ALTER TABLE chats"
                        " ADD COLUMN last_active REAL NOT NULL DEFAULT 0"
                    )

                if self._fts_error is None:
                    had_fts = connection.execute(
                        "SELECT 1 FROM sqlite_master"
                        " WHERE type = 'table' AND name = 'turns_fts'"
                    ).fetchone()
                    backfilled = connection.execute(
                        "SELECT 1 FROM history_meta"
                        " WHERE key = 'fts_storage_version' AND value = '1'"
                    ).fetchone()
                    try:
                        connection.executescript(FTS_SCHEMA)
                        if not had_fts or not backfilled:
                            # Hermes' external-content FTS rebuild also makes
                            # existing installations searchable. The marker is
                            # committed with it, so a crash safely retries.
                            connection.execute(
                                "INSERT INTO turns_fts(turns_fts) VALUES ('rebuild')"
                            )
                            connection.execute(
                                "INSERT OR REPLACE INTO history_meta (key, value)"
                                " VALUES ('fts_storage_version', '1')"
                            )
                    except sqlite3.OperationalError as exc:
                        if "fts5" not in str(exc).lower():
                            raise
                        self._fts_error = str(exc)
                        logger.error("Conversation search is unavailable: %s", exc)
                # Schema and backfill must survive an error in the operation
                # that follows; otherwise the marker could outlive the index.
                connection.commit()
            with connection:
                yield connection
        finally:
            connection.close()

    def _session(self, connection: sqlite3.Connection, chat_id: str) -> int:
        row = connection.execute(
            "SELECT session FROM chats WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return row[0] if row else 1

    def current_session(self, chat_id: str) -> int:
        """Return the durable generation currently selected for one chat."""

        if self._path is None:
            return 1
        try:
            with self._connect() as connection:
                return int(self._session(connection, chat_id))
        except (sqlite3.Error, OSError) as exc:
            logger.warning(
                "Could not read the current session for %s",
                chat_id,
                exc_info=True,
            )
            raise ConversationError(
                f"Could not read the current session for {chat_id}"
            ) from exc

    def prepare_session(
        self,
        chat_id: str,
        *,
        mode: str = "none",
        idle_minutes: int = 1440,
        at_hour: int = 4,
        now: Optional[float] = None,
        tzinfo=None,
    ) -> Optional[SessionReset]:
        """Apply the configured automatic reset before an inbound user turn.

        The boundary and activity timestamp are one SQLite transaction, so two
        resident channel tasks cannot both reset the same conversation.
        """

        if self._path is None:
            return None
        current_time = float(time.time() if now is None else now)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT session, last_active FROM chats WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()
                if row is None:
                    session = 1
                    activity = connection.execute(
                        "SELECT MAX(written_at) FROM turns"
                        " WHERE chat_id = ? AND session = 1",
                        (chat_id,),
                    ).fetchone()[0]
                    last_active = (
                        float(activity) if activity is not None else current_time
                    )
                    connection.execute(
                        "INSERT INTO chats (chat_id, session, last_active)"
                        " VALUES (?, 1, ?)",
                        (chat_id, current_time),
                    )
                else:
                    session = int(row["session"])
                    last_active = float(row["last_active"] or 0)
                    if last_active <= 0:
                        activity = connection.execute(
                            "SELECT MAX(written_at) FROM turns"
                            " WHERE chat_id = ? AND session = ?",
                            (chat_id, session),
                        ).fetchone()[0]
                        last_active = (
                            float(activity) if activity is not None else current_time
                        )

                # An automatic clock boundary must not discard an ambiguous
                # interrupted tool turn. Only the user's explicit /new can
                # abandon that checkpoint.
                if connection.execute(
                    "SELECT 1 FROM active_turns"
                    " WHERE chat_id = ? AND session = ?",
                    (chat_id, session),
                ).fetchone():
                    return None

                reason = _automatic_reset_reason(
                    last_active,
                    current_time,
                    mode=mode,
                    idle_minutes=idle_minutes,
                    at_hour=at_hour,
                    tzinfo=tzinfo,
                )
                had_activity = bool(
                    connection.execute(
                        "SELECT 1 FROM turns"
                        " WHERE chat_id = ? AND session = ? LIMIT 1",
                        (chat_id, session),
                    ).fetchone()
                )
                if reason is not None:
                    connection.execute(
                        "UPDATE chats"
                        " SET session = session + 1, last_active = ?"
                        " WHERE chat_id = ?",
                        (current_time, chat_id),
                    )
                    return SessionReset(reason, had_activity)

                connection.execute(
                    "UPDATE chats SET last_active = ? WHERE chat_id = ?",
                    (current_time, chat_id),
                )
                return None
        except (sqlite3.Error, OSError) as exc:
            logger.warning(
                "Could not apply session reset policy for %s",
                chat_id,
                exc_info=True,
            )
            raise ConversationError(
                f"Could not durably prepare session for {chat_id}"
            ) from exc

    def load(self, chat_id: str, limit: int) -> List[StoredTurn]:
        """The last *limit* turns of the chat's current session, oldest first."""
        if self._path is None or limit <= 0:
            return []
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT role, content FROM turns"
                    " WHERE chat_id = ? AND session = ?"
                    " ORDER BY id DESC LIMIT ?",
                    (chat_id, self._session(connection, chat_id), limit),
                ).fetchall()
        except (sqlite3.Error, OSError):
            logger.warning("Could not read the history of %s", chat_id, exc_info=True)
            return []
        return [(role, content) for role, content in reversed(rows)]

    def load_with_replay(
        self, chat_id: str, limit: Optional[int] = None
    ) -> List[StoredReplayTurn]:
        """Load words plus opaque compaction checkpoints, oldest first."""
        if self._path is None or (limit is not None and limit <= 0):
            return []
        try:
            with self._connect() as connection:
                session = self._session(connection, chat_id)
                if limit is None:
                    rows = connection.execute(
                        "SELECT role, content, replay FROM turns"
                        " WHERE chat_id = ? AND session = ?"
                        " ORDER BY id ASC",
                        (chat_id, session),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT role, content, replay FROM turns"
                        " WHERE chat_id = ? AND session = ?"
                        " ORDER BY id DESC LIMIT ?",
                        (chat_id, session, limit),
                    ).fetchall()
                    rows.reverse()
        except (sqlite3.Error, OSError):
            logger.warning("Could not read the history of %s", chat_id, exc_info=True)
            return []
        return [
            (role, content, _decode_replay(replay))
            for role, content, replay in rows
        ]

    def append(self, chat_id: str, turns: Sequence[StoredTurn]) -> None:
        """Add turns to the chat's current session."""
        self.append_with_replay(
            chat_id, [(role, content, []) for role, content in turns]
        )

    def append_with_replay(
        self, chat_id: str, turns: Sequence[StoredReplayTurn]
    ) -> None:
        """Add words and validated opaque compaction checkpoints."""
        if self._path is None or not turns:
            return
        now = time.time()
        try:
            with self._connect() as connection:
                session = self._session(connection, chat_id)
                connection.execute(
                    "INSERT INTO chats (chat_id, session, last_active)"
                    " VALUES (?, ?, ?)"
                    " ON CONFLICT (chat_id)"
                    " DO UPDATE SET last_active = excluded.last_active",
                    (chat_id, session, now),
                )
                connection.executemany(
                    "INSERT INTO turns"
                    " (chat_id, session, role, content, replay, written_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            chat_id,
                            session,
                            role,
                            content,
                            json.dumps(_compaction_items(replay), separators=(",", ":")),
                            now,
                        )
                        for role, content, replay in turns
                    ],
                )
        except (sqlite3.Error, OSError, TypeError) as exc:
            logger.warning("Could not write the history of %s", chat_id, exc_info=True)
            raise ConversationError(
                f"Could not write the history of {chat_id}"
            ) from exc

    def begin_turn(self, chat_id: str, user_content: str) -> None:
        """Durably accept one user turn before the model can act on it."""

        if self._path is None:
            return
        now = time.time()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                session = self._session(connection, chat_id)
                existing = connection.execute(
                    "SELECT 1 FROM active_turns"
                    " WHERE chat_id = ? AND session = ?",
                    (chat_id, session),
                ).fetchone()
                if existing is not None:
                    raise ConversationError(
                        "The previous turn is incomplete; use /new after "
                        "checking whether its tool work already happened."
                    )
                connection.execute(
                    "INSERT INTO chats (chat_id, session, last_active)"
                    " VALUES (?, ?, ?)"
                    " ON CONFLICT (chat_id)"
                    " DO UPDATE SET last_active = excluded.last_active",
                    (chat_id, session, now),
                )
                connection.execute(
                    "INSERT INTO active_turns"
                    " (chat_id, session, user_content, trajectory, phase, updated_at)"
                    " VALUES (?, ?, ?, '[]', 'started', ?)",
                    (chat_id, session, user_content, now),
                )
        except ConversationError:
            raise
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not begin the turn for %s", chat_id, exc_info=True)
            raise ConversationError(
                f"Could not begin the turn for {chat_id}"
            ) from exc

    def checkpoint_turn(
        self,
        chat_id: str,
        user_content: str,
        items: Sequence[Dict[str, Any]],
        *,
        phase: str,
    ) -> None:
        """Persist a tool boundary before execution or model continuation."""

        if self._path is None:
            return
        if phase not in {"tool_requested", "tool_completed"}:
            raise ValueError(f"Unsupported active-turn phase: {phase}")
        try:
            encoded = json.dumps(
                list(items), ensure_ascii=False, separators=(",", ":")
            )
            with self._connect() as connection:
                session = self._session(connection, chat_id)
                cursor = connection.execute(
                    "UPDATE active_turns"
                    " SET trajectory = ?, phase = ?, updated_at = ?"
                    " WHERE chat_id = ? AND session = ? AND user_content = ?",
                    (encoded, phase, time.time(), chat_id, session, user_content),
                )
                if cursor.rowcount != 1:
                    raise ConversationError(
                        f"No active turn can be checkpointed for {chat_id}"
                    )
        except ConversationError:
            raise
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            logger.warning(
                "Could not checkpoint the %s turn for %s",
                phase,
                chat_id,
                exc_info=True,
            )
            raise ConversationError(
                f"Could not checkpoint the turn for {chat_id}"
            ) from exc

    def discard_unstarted_turn(self, chat_id: str) -> None:
        """Forget a failed model attempt only when no tool call was persisted."""

        if self._path is None:
            return
        try:
            with self._connect() as connection:
                session = self._session(connection, chat_id)
                connection.execute(
                    "DELETE FROM active_turns"
                    " WHERE chat_id = ? AND session = ? AND phase = 'started'",
                    (chat_id, session),
                )
        except (sqlite3.Error, OSError) as exc:
            logger.warning(
                "Could not discard the unstarted turn for %s",
                chat_id,
                exc_info=True,
            )
            raise ConversationError(
                f"Could not discard the unstarted turn for {chat_id}"
            ) from exc

    def complete_turn(
        self,
        chat_id: str,
        user_content: str,
        assistant_content: str,
        replay: Sequence[Dict[str, Any]] = (),
    ) -> None:
        """Atomically replace the active checkpoint with completed text turns."""

        if self._path is None:
            return
        now = time.time()
        try:
            encoded_replay = json.dumps(
                _compaction_items(replay), separators=(",", ":")
            )
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                session = self._session(connection, chat_id)
                active = connection.execute(
                    "SELECT 1 FROM active_turns"
                    " WHERE chat_id = ? AND session = ? AND user_content = ?",
                    (chat_id, session, user_content),
                ).fetchone()
                if active is None:
                    raise ConversationError(
                        f"No active turn can be completed for {chat_id}"
                    )
                connection.execute(
                    "INSERT INTO chats (chat_id, session, last_active)"
                    " VALUES (?, ?, ?)"
                    " ON CONFLICT (chat_id)"
                    " DO UPDATE SET last_active = excluded.last_active",
                    (chat_id, session, now),
                )
                connection.executemany(
                    "INSERT INTO turns"
                    " (chat_id, session, role, content, replay, written_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (chat_id, session, "user", user_content, "", now),
                        (
                            chat_id,
                            session,
                            "assistant",
                            assistant_content,
                            encoded_replay,
                            now,
                        ),
                    ],
                )
                connection.execute(
                    "DELETE FROM active_turns"
                    " WHERE chat_id = ? AND session = ?",
                    (chat_id, session),
                )
        except ConversationError:
            raise
        except (sqlite3.Error, OSError, TypeError) as exc:
            logger.warning("Could not complete the turn for %s", chat_id, exc_info=True)
            raise ConversationError(
                f"Could not complete the turn for {chat_id}"
            ) from exc

    def new_session(self, chat_id: str) -> None:
        """Leave the current conversation behind without deleting it.

        Written down rather than kept in memory: a restart right after "/new"
        would otherwise hand back the conversation the person just ended.
        """
        if self._path is None:
            return
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                previous_session = self._session(connection, chat_id)
                connection.execute(
                    "INSERT INTO chats (chat_id, session, last_active)"
                    " VALUES (?, 2, ?)"
                    " ON CONFLICT (chat_id) DO UPDATE"
                    " SET session = session + 1, last_active = excluded.last_active",
                    (chat_id, time.time()),
                )
                connection.execute(
                    "DELETE FROM active_turns"
                    " WHERE chat_id = ? AND session = ?",
                    (chat_id, previous_session),
                )
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not start a new session for %s", chat_id, exc_info=True)
            raise ConversationError(
                f"Could not durably start a new session for {chat_id}"
            ) from exc

    def prune_old_sessions(
        self,
        retention_days: int,
        *,
        now: Optional[float] = None,
        workspace_roots: Sequence[Path] = (),
    ) -> int:
        """Delete expired ended sessions and their isolated workspaces."""

        if self._path is None or retention_days <= 0:
            return 0
        cutoff = float(time.time() if now is None else now) - (
            int(retention_days) * 86400
        )
        pruned_sessions: List[Tuple[str, int]] = []
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    "SELECT t.chat_id, t.session"
                    " FROM turns t"
                    " LEFT JOIN chats c ON c.chat_id = t.chat_id"
                    " GROUP BY t.chat_id, t.session"
                    " HAVING t.session != COALESCE(MAX(c.session), 1)"
                    " AND MAX(t.written_at) < ?",
                    (cutoff,),
                ).fetchall()
                pruned_sessions = [
                    (str(row["chat_id"]), int(row["session"]))
                    for row in rows
                ]
                connection.executemany(
                    "DELETE FROM turns WHERE chat_id = ? AND session = ?",
                    pruned_sessions,
                )
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not prune old conversation sessions", exc_info=True)
            raise ConversationError("Could not prune old conversation sessions") from exc

        # Hermes removes filesystem state only after the DB transaction has
        # committed, and a filesystem hiccup never rolls the DB prune back.
        for workspace_root in dict.fromkeys(workspace_roots):
            for chat_id, session in pruned_sessions:
                self._remove_session_workspace(
                    workspace_root,
                    chat_id,
                    session,
                )
        return len(pruned_sessions)

    @staticmethod
    def _remove_session_workspace(
        workspace_root: Path, chat_id: str, session: int
    ) -> None:
        """Best-effort removal of one exact Pilotage session directory."""

        try:
            root = Path(workspace_root).expanduser().resolve(strict=False)
            sessions_root = (root / "sessions").resolve(strict=False)
            sessions_root.relative_to(root)
            target = session_workspace_path(root, chat_id, session)
            target.resolve(strict=False).relative_to(sessions_root)
            if target.is_symlink():
                target.unlink(missing_ok=True)
            elif target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        except (OSError, RuntimeError, ValueError):
            logger.debug(
                "Could not prune the session workspace for %s generation %d",
                chat_id,
                session,
                exc_info=True,
            )

    # -- conversation search ----------------------------------------------

    @staticmethod
    def _stored_message(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": int(row["id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "written_at": float(row["written_at"]),
        }

    @staticmethod
    def _session_reference(
        connection: sqlite3.Connection, chat_id: str, session: int
    ) -> str:
        row = connection.execute(
            "SELECT MIN(id) FROM turns WHERE chat_id = ? AND session = ?",
            (chat_id, session),
        ).fetchone()
        if row is None or row[0] is None:
            raise ConversationSearchError("Conversation session has no stored turns")
        # Genesis sessions are a composite (chat, generation). The first
        # immutable turn id is their compact, stable public handle.
        return str(int(row[0]))

    @staticmethod
    def _resolve_session(
        connection: sqlite3.Connection, session_id: str
    ) -> Optional[Tuple[str, int]]:
        try:
            anchor = int(str(session_id).strip())
        except (TypeError, ValueError):
            return None
        if anchor <= 0:
            return None
        row = connection.execute(
            "SELECT chat_id, session FROM turns WHERE id = ?",
            (anchor,),
        ).fetchone()
        if row is None:
            return None
        return str(row["chat_id"]), int(row["session"])

    def search_messages(
        self,
        query: str,
        *,
        current_chat_id: str,
        roles: Sequence[str],
        scan_limit: int,
        sort: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search actual stored words with Hermes' external-content FTS shape."""
        if self._path is None:
            raise ConversationSearchError(
                "Conversation search is unavailable in one-shot mode"
            )
        if not query:
            return []

        # Hermes demotes automation below interactive recall so repetitive
        # scheduled runs cannot starve out the user's conversations.
        order_sql = "automation_rank, score"
        if sort == "newest":
            order_sql = "automation_rank, t.written_at DESC, score"
        elif sort == "oldest":
            order_sql = "automation_rank, t.written_at ASC, score"

        try:
            with self._connect() as connection:
                if self._fts_error is not None:
                    raise ConversationSearchError(
                        f"SQLite FTS5 is unavailable: {self._fts_error}"
                    )
                current_session = self._session(connection, current_chat_id)
                role_placeholders = ",".join("?" for _ in roles)
                rows = connection.execute(
                    "SELECT t.id, t.chat_id, t.session, t.role, t.content,"
                    " t.written_at,"
                    " snippet(turns_fts, 0, '>>>', '<<<', '...', 40) AS snippet,"
                    " bm25(turns_fts) AS score,"
                    " CASE WHEN t.chat_id LIKE 'cron:%' THEN 1"
                    " ELSE 0 END AS automation_rank"
                    " FROM turns_fts"
                    " JOIN turns t ON t.id = turns_fts.rowid"
                    " WHERE turns_fts MATCH ?"
                    " AND NOT (t.chat_id = ? AND t.session = ?)"
                    f" AND t.role IN ({role_placeholders})"
                    f" ORDER BY {order_sql} LIMIT ?",
                    [
                        query,
                        current_chat_id,
                        current_session,
                        *roles,
                        max(1, min(int(scan_limit), 1000)),
                    ],
                ).fetchall()

                references: Dict[Tuple[str, int], str] = {}
                results: List[Dict[str, Any]] = []
                for row in rows:
                    key = (str(row["chat_id"]), int(row["session"]))
                    reference = references.get(key)
                    if reference is None:
                        reference = self._session_reference(connection, *key)
                        references[key] = reference
                    results.append(
                        {
                            "id": int(row["id"]),
                            "session_id": reference,
                            "role": str(row["role"]),
                            "content": str(row["content"]),
                            "snippet": str(row["snippet"] or ""),
                            "written_at": float(row["written_at"]),
                        }
                    )
                return results
        except ConversationSearchError:
            raise
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not search conversation history", exc_info=True)
            raise ConversationSearchError("Could not search conversation history") from exc

    def recent_sessions(
        self, *, current_chat_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """List recent stored sessions, excluding the caller's active context."""
        if self._path is None:
            raise ConversationSearchError(
                "Conversation search is unavailable in one-shot mode"
            )
        try:
            with self._connect() as connection:
                current_session = self._session(connection, current_chat_id)
                rows = connection.execute(
                    "SELECT chat_id, session, MIN(id) AS session_id,"
                    " MIN(written_at) AS started_at,"
                    " MAX(written_at) AS last_active, COUNT(*) AS message_count"
                    " FROM turns"
                    " WHERE NOT (chat_id = ? AND session = ?)"
                    " GROUP BY chat_id, session"
                    " ORDER BY last_active DESC, session_id DESC LIMIT ?",
                    (
                        current_chat_id,
                        current_session,
                        max(1, min(int(limit), 50)),
                    ),
                ).fetchall()
                results: List[Dict[str, Any]] = []
                for row in rows:
                    preview = connection.execute(
                        "SELECT content FROM turns"
                        " WHERE chat_id = ? AND session = ?"
                        " ORDER BY CASE role WHEN 'user' THEN 0 ELSE 1 END, id"
                        " LIMIT 1",
                        (row["chat_id"], row["session"]),
                    ).fetchone()
                    results.append(
                        {
                            "session_id": str(int(row["session_id"])),
                            "started_at": float(row["started_at"]),
                            "last_active": float(row["last_active"]),
                            "message_count": int(row["message_count"]),
                            "preview": str(preview[0]) if preview else "",
                        }
                    )
                return results
        except ConversationSearchError:
            raise
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not browse conversation history", exc_info=True)
            raise ConversationSearchError("Could not browse conversation history") from exc

    def read_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return one complete stored session by its stable turn handle."""
        if self._path is None:
            raise ConversationSearchError(
                "Conversation search is unavailable in one-shot mode"
            )
        try:
            with self._connect() as connection:
                key = self._resolve_session(connection, session_id)
                if key is None:
                    return None
                chat_id, session = key
                rows = connection.execute(
                    "SELECT id, role, content, written_at FROM turns"
                    " WHERE chat_id = ? AND session = ? ORDER BY id",
                    key,
                ).fetchall()
                if not rows:
                    return None
                return {
                    "session_id": self._session_reference(
                        connection, chat_id, session
                    ),
                    "started_at": float(rows[0]["written_at"]),
                    "last_active": float(rows[-1]["written_at"]),
                    "messages": [self._stored_message(row) for row in rows],
                }
        except ConversationSearchError:
            raise
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not read conversation session", exc_info=True)
            raise ConversationSearchError("Could not read conversation session") from exc

    def anchored_view(
        self,
        session_id: str,
        around_message_id: int,
        *,
        window: int,
        bookend: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Return a bounded Hermes-style window around one stored message."""
        if self._path is None:
            raise ConversationSearchError(
                "Conversation search is unavailable in one-shot mode"
            )
        window = max(1, min(int(window), 20))
        bookend = max(0, min(int(bookend), 10))
        try:
            anchor = int(around_message_id)
        except (TypeError, ValueError):
            return None

        try:
            with self._connect() as connection:
                key = self._resolve_session(connection, session_id)
                if key is None:
                    return None
                owns_anchor = connection.execute(
                    "SELECT 1 FROM turns"
                    " WHERE id = ? AND chat_id = ? AND session = ?",
                    (anchor, *key),
                ).fetchone()
                if owns_anchor is None:
                    return None

                before = connection.execute(
                    "SELECT id, role, content, written_at FROM turns"
                    " WHERE chat_id = ? AND session = ? AND id <= ?"
                    " ORDER BY id DESC LIMIT ?",
                    (*key, anchor, window + 1),
                ).fetchall()
                before.reverse()
                after = connection.execute(
                    "SELECT id, role, content, written_at FROM turns"
                    " WHERE chat_id = ? AND session = ? AND id > ?"
                    " ORDER BY id LIMIT ?",
                    (*key, anchor, window),
                ).fetchall()
                rows = [*before, *after]
                first_id = int(rows[0]["id"])
                last_id = int(rows[-1]["id"])
                messages_before = connection.execute(
                    "SELECT COUNT(*) FROM turns"
                    " WHERE chat_id = ? AND session = ? AND id < ?",
                    (*key, first_id),
                ).fetchone()[0]
                messages_after = connection.execute(
                    "SELECT COUNT(*) FROM turns"
                    " WHERE chat_id = ? AND session = ? AND id > ?",
                    (*key, last_id),
                ).fetchone()[0]

                start_rows: Sequence[sqlite3.Row] = ()
                end_rows: Sequence[sqlite3.Row] = ()
                if bookend:
                    start_rows = connection.execute(
                        "SELECT id, role, content, written_at FROM turns"
                        " WHERE chat_id = ? AND session = ?"
                        " ORDER BY id LIMIT ?",
                        (*key, bookend),
                    ).fetchall()
                    end_rows = connection.execute(
                        "SELECT id, role, content, written_at FROM turns"
                        " WHERE chat_id = ? AND session = ?"
                        " ORDER BY id DESC LIMIT ?",
                        (*key, bookend),
                    ).fetchall()
                    end_rows.reverse()

                return {
                    "session_id": self._session_reference(connection, *key),
                    "messages": [self._stored_message(row) for row in rows],
                    "bookend_start": [
                        self._stored_message(row) for row in start_rows
                    ],
                    "bookend_end": [
                        self._stored_message(row) for row in end_rows
                    ],
                    "messages_before": int(messages_before),
                    "messages_after": int(messages_after),
                }
        except ConversationSearchError:
            raise
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not scroll conversation session", exc_info=True)
            raise ConversationSearchError("Could not scroll conversation session") from exc
