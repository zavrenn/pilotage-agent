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

Nothing older is ever deleted. "/new" starts a new session rather than erasing
the old one, so ending a conversation stays cheap and reversible, and the
history is already there when there is something to do with it.

A completed answer is still delivered if its history append fails. ``/new`` is
the deliberate exception: its durable session boundary must succeed before the
live conversation is cleared, or a restart could silently resurrect what the
user ended.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
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
    chat_id TEXT PRIMARY KEY,
    session INTEGER NOT NULL
);
"""

# A stored turn: what was said, and by whom.
StoredTurn = Tuple[str, str]
StoredReplayTurn = Tuple[str, str, List[Dict[str, str]]]


class ConversationError(RuntimeError):
    """A durable conversation boundary could not be recorded."""


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

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection for one piece of work, committed and closed after it.

        Closing matters: `sqlite3.connect` as a context manager ends the
        transaction but leaves the handle open, and a leaked handle keeps a
        lock on the file for as long as the process runs.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_SECONDS)
        try:
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
            with connection:
                yield connection
        finally:
            connection.close()

    def _session(self, connection: sqlite3.Connection, chat_id: str) -> int:
        row = connection.execute(
            "SELECT session FROM chats WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return row[0] if row else 1

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
        except (sqlite3.Error, OSError):
            logger.warning("Could not write the history of %s", chat_id, exc_info=True)

    def new_session(self, chat_id: str) -> None:
        """Leave the current conversation behind without deleting it.

        Written down rather than kept in memory: a restart right after "/new"
        would otherwise hand back the conversation the person just ended.
        """
        if self._path is None:
            return
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO chats (chat_id, session) VALUES (?, 2)"
                    " ON CONFLICT (chat_id) DO UPDATE SET session = session + 1",
                    (chat_id,),
                )
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Could not start a new session for %s", chat_id, exc_info=True)
            raise ConversationError(
                f"Could not durably start a new session for {chat_id}"
            ) from exc
