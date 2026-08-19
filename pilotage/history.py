"""Conversations on disk.

A WhatsApp chat is never over. Someone writes again three days later and
expects the thread to still be there. History held in the process cannot
honour that: an upgrade, a crash or a reboot of the container starts every
conversation again from nothing, and the person on the other end has no way of
knowing it happened.

So every turn is written to a small SQLite file beside the credentials. Only
the words are kept. The model's encrypted reasoning is deliberately left out —
it is a speed optimisation belonging to a response that no longer exists after
a restart, and replaying a stale one risks the API refusing the whole request.
Pictures are left out for the same practical reason they are dropped from all
but the newest turn while the process runs: they are enormous next to the text.

Nothing older is ever deleted. "/new" starts a new session rather than erasing
the old one, so ending a conversation stays cheap and reversible, and the
history is already there when there is something to do with it.

Nothing here may fail a turn. A conversation that cannot be written down is
worth a line in the log; it is not worth refusing to answer.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

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

    def append(self, chat_id: str, turns: Sequence[StoredTurn]) -> None:
        """Add turns to the chat's current session."""
        if self._path is None or not turns:
            return
        now = time.time()
        try:
            with self._connect() as connection:
                session = self._session(connection, chat_id)
                connection.executemany(
                    "INSERT INTO turns (chat_id, session, role, content, written_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    [(chat_id, session, role, content, now) for role, content in turns],
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
        except (sqlite3.Error, OSError):
            logger.warning("Could not start a new session for %s", chat_id, exc_info=True)
