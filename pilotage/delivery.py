"""Durable final-response delivery, reduced from Hermes' gateway mechanism."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import math
import os
import random
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterator, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_ROWS = 500
LIVE_RETRY_MIN_AGE_SECONDS = 60.0
MAX_RECOVERY_BATCH = 20
MAX_INLINE_RETRY_AFTER_SECONDS = 5.0
RECOVERED_MARKER = (
    "♻️ Recovered reply — the agent restarted during delivery, "
    "so this may be a duplicate:\n\n"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS delivery_obligations (
    obligation_id TEXT PRIMARY KEY,
    session_key   TEXT NOT NULL,
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    thread_id     TEXT,
    reply_to      TEXT,
    content       TEXT NOT NULL,
    state         TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    owner_token   TEXT NOT NULL,
    last_error    TEXT,
    retry_safe    INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS delivery_units (
    obligation_id TEXT NOT NULL,
    unit_id       TEXT NOT NULL,
    position      INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    state         TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    retry_safe    INTEGER NOT NULL DEFAULT 0,
    evidence      TEXT,
    last_error    TEXT,
    updated_at    REAL NOT NULL,
    PRIMARY KEY (obligation_id, unit_id),
    UNIQUE (obligation_id, position)
);
CREATE TABLE IF NOT EXISTS command_outcomes (
    command_id     TEXT PRIMARY KEY,
    platform       TEXT NOT NULL,
    claim_id       TEXT NOT NULL,
    session_key    TEXT NOT NULL,
    command_name   TEXT NOT NULL,
    arguments      TEXT NOT NULL,
    state          TEXT NOT NULL CHECK (state IN ('started', 'completed')),
    response       TEXT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    UNIQUE (platform, claim_id)
);
"""


@dataclass(frozen=True)
class SendResult:
    """The platform-neutral part of Hermes' outbound send result."""

    success: bool
    error: str = ""
    retryable: bool = False
    retry_after: Optional[float] = None
    message_id: str = ""
    _unit_failure_recorded: bool = False

    def __bool__(self) -> bool:
        return self.success


def as_send_result(value: Any) -> SendResult:
    if isinstance(value, SendResult):
        return value
    return SendResult(bool(value), error="send rejected" if not value else "")


def compute_obligation_id(
    session_key: str, message_ref: str, content: str
) -> str:
    payload = f"{session_key}|{message_ref}|{content}"
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]


def compute_command_id(platform: str, claim_id: str) -> str:
    """Stable cross-channel identity for one durable inbound command claim."""

    digest = hashlib.sha256()
    for value in ("management-command-v1", platform, claim_id):
        encoded = str(value).encode("utf-8", "replace")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()[:32]


def delivery_fingerprint(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8", "replace")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def file_delivery_fingerprint(path: Path, *metadata: str) -> str:
    digest = hashlib.sha256()
    for value in ("file-v1", *metadata):
        encoded = str(value).encode("utf-8", "replace")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class DeliveryPlanError(RuntimeError):
    """A retry no longer describes the exact units first recorded."""


@dataclass(frozen=True)
class CommandOutcome:
    """Durable execution decision for one claimed management command."""

    execute: bool
    completed: bool
    response: str = ""


@dataclass(frozen=True)
class DeliveryUnit:
    unit_id: str
    position: int
    kind: str
    fingerprint: str


class DeliveryStore:
    """Profile-local delivery obligations with bounded restart recovery."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._owner_token = f"{os.getpid()}:{secrets.token_hex(12)}"
        self._active_recovery_claims: set[str] = set()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(_SCHEMA)
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(delivery_obligations)"
                )
            }
            if "retry_safe" not in columns:
                # Legacy failures carry no proof that platform acceptance was
                # impossible, so migration deliberately defaults them unsafe.
                connection.execute(
                    "ALTER TABLE delivery_obligations"
                    " ADD COLUMN retry_safe INTEGER NOT NULL DEFAULT 0"
                )
            if "reply_to" not in columns:
                connection.execute(
                    "ALTER TABLE delivery_obligations ADD COLUMN reply_to TEXT"
                )
            if "next_attempt_at" not in columns:
                connection.execute(
                    "ALTER TABLE delivery_obligations"
                    " ADD COLUMN next_attempt_at REAL NOT NULL DEFAULT 0"
                )
            connection.commit()
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
            with connection:
                yield connection
        finally:
            connection.close()

    def verify_writable(self) -> None:
        """Prove schema initialization and one rollback-safe write transaction."""

        now = time.time()
        probe_id = f"readiness:{self._owner_token}:{secrets.token_hex(8)}"
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO delivery_obligations"
                    " (obligation_id, session_key, platform, chat_id, thread_id,"
                    " reply_to, content, state, attempts, created_at, updated_at,"
                    " owner_token, last_error, retry_safe)"
                    " VALUES (?, 'readiness', 'readiness', 'readiness', NULL,"
                    " NULL, '', 'pending', 0, ?, ?, ?, NULL, 0)",
                    (probe_id, now, now, self._owner_token),
                )
            finally:
                connection.rollback()

    def release_recovery_claim(self, obligation_id: str) -> None:
        with self._lock:
            self._active_recovery_claims.discard(str(obligation_id))

    def record(
        self,
        *,
        obligation_id: str,
        session_key: str,
        platform: str,
        chat_id: str,
        thread_id: str,
        content: str,
        reply_to: str = "",
    ) -> str:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO delivery_obligations"
                " (obligation_id, session_key, platform, chat_id, thread_id,"
                " reply_to, content, state, attempts, created_at, updated_at,"
                " owner_token, last_error, retry_safe)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, NULL, 0)",
                (
                    obligation_id,
                    session_key,
                    platform,
                    chat_id,
                    thread_id or None,
                    reply_to or None,
                    content,
                    now,
                    now,
                    self._owner_token,
                ),
            )
            row = connection.execute(
                "SELECT session_key, platform, chat_id, thread_id, reply_to, content,"
                " state"
                " FROM delivery_obligations WHERE obligation_id = ?",
                (obligation_id,),
            ).fetchone()
            expected = (
                session_key,
                platform,
                chat_id,
                thread_id or "",
                reply_to or "",
                content,
            )
            recorded = (
                str(row["session_key"]),
                str(row["platform"]),
                str(row["chat_id"]),
                str(row["thread_id"] or ""),
                str(row["reply_to"] or ""),
                str(row["content"]),
            ) if row is not None else None
            if recorded != expected:
                raise DeliveryPlanError(
                    "delivery obligation identity collides with different durable data"
                )
            self._prune(connection, now)
            return str(row["state"])

    def begin_command(
        self,
        *,
        command_id: str,
        platform: str,
        claim_id: str,
        session_key: str,
        command_name: str,
        arguments: str,
    ) -> CommandOutcome:
        """Reserve one command before side effects, or return its saved outcome."""

        now = time.time()
        expected = (
            str(platform),
            str(claim_id),
            str(session_key),
            str(command_name),
            str(arguments),
        )
        if not command_id or not all(expected[:4]):
            raise DeliveryPlanError("management command identity is incomplete")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO command_outcomes"
                " (command_id, platform, claim_id, session_key, command_name,"
                " arguments, state, response, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'started', NULL, ?, ?)",
                (command_id, *expected, now, now),
            )
            row = connection.execute(
                "SELECT platform, claim_id, session_key, command_name, arguments,"
                " state, response FROM command_outcomes WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            recorded = (
                str(row["platform"]),
                str(row["claim_id"]),
                str(row["session_key"]),
                str(row["command_name"]),
                str(row["arguments"]),
            ) if row is not None else None
            if recorded != expected:
                raise DeliveryPlanError(
                    "management command identity collides with different durable data"
                )
            if not cursor.rowcount and str(row["state"]) == "completed":
                # Replaying the durable response proves this inbound claim is
                # not settled yet. Refresh its fence before age/count pruning
                # so this replay cannot delete the only at-most-once record.
                connection.execute(
                    "UPDATE command_outcomes SET updated_at = ?"
                    " WHERE command_id = ? AND state = 'completed'",
                    (now, command_id),
                )
            self._prune_commands(connection, now)
            if cursor.rowcount:
                return CommandOutcome(execute=True, completed=False)
            if str(row["state"]) == "completed":
                response = row["response"]
                if not isinstance(response, str):
                    raise DeliveryPlanError(
                        "completed management command has no durable response"
                    )
                return CommandOutcome(
                    execute=False,
                    completed=True,
                    response=response,
                )
            if str(row["state"]) != "started" or row["response"] is not None:
                raise DeliveryPlanError("management command outcome is malformed")
            # The previous handler crossed the durable execution boundary but
            # did not save a result. Repeating a side effect would be unsafe.
            return CommandOutcome(execute=False, completed=False)

    def complete_command(self, command_id: str, response: str) -> None:
        """Persist the exact response without ever changing an existing result."""

        written = str(response)
        now = time.time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE command_outcomes SET state = 'completed', response = ?,"
                " updated_at = ? WHERE command_id = ? AND state = 'started'",
                (written, now, command_id),
            )
            row = connection.execute(
                "SELECT state, response FROM command_outcomes WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if row is None:
                raise DeliveryPlanError("management command reservation is missing")
            if str(row["state"]) != "completed" or str(row["response"]) != written:
                raise DeliveryPlanError(
                    "management command response changed after completion"
                )
            if cursor.rowcount:
                self._prune_commands(connection, now)

    def mark_attempting(self, obligation_id: str) -> bool:
        return self._update(
            obligation_id,
            "attempting",
            expected_states=("pending",),
        )

    def activate_unit_plan(self, obligation_id: str) -> bool:
        """Claim a parent only after its exact non-empty unit plan is durable."""

        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT owner_token, state FROM delivery_obligations"
                " WHERE obligation_id = ?",
                (obligation_id,),
            ).fetchone()
            if (
                row is None
                or row["owner_token"] != self._owner_token
                or row["state"] not in {"pending", "attempting"}
            ):
                return False
            planned = connection.execute(
                "SELECT 1 FROM delivery_units"
                " WHERE obligation_id = ? LIMIT 1",
                (obligation_id,),
            ).fetchone()
            if planned is None:
                return False
            if row["state"] == "attempting":
                return True
            cursor = connection.execute(
                "UPDATE delivery_obligations SET state = 'attempting',"
                " updated_at = ?, last_error = NULL, retry_safe = 0,"
                " next_attempt_at = 0"
                " WHERE obligation_id = ? AND owner_token = ?"
                " AND state = 'pending'"
                " AND EXISTS (SELECT 1 FROM delivery_units"
                " WHERE obligation_id = ?)",
                (
                    time.time(),
                    obligation_id,
                    self._owner_token,
                    obligation_id,
                ),
            )
            return bool(cursor.rowcount)

    def discard_unplanned(self, obligation_id: str) -> bool:
        """Remove a known-unsent reservation after unit planning failed."""

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM delivery_obligations"
                " WHERE obligation_id = ? AND owner_token = ?"
                " AND state = 'pending'"
                " AND NOT EXISTS (SELECT 1 FROM delivery_units"
                " WHERE obligation_id = ?)",
                (obligation_id, self._owner_token, obligation_id),
            )
            return bool(cursor.rowcount)

    def quarantine_unplanned(self, obligation_id: str, error: str) -> bool:
        """Fence a ledger callback that may have bypassed unit planning."""

        return self._update(
            obligation_id,
            "failed",
            error,
            retry_safe=False,
            expected_states=("pending", "attempting"),
        )

    def exact_obligation_exists(
        self,
        obligation_id: str,
        *,
        session_key: str,
        platform: str,
        chat_id: str,
        thread_id: str,
        reply_to: str,
        content: str,
    ) -> bool:
        """Prove that the exact final response already has a durable ledger row."""

        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT session_key, platform, chat_id, thread_id, reply_to,"
                " content, state, EXISTS (SELECT 1 FROM delivery_units u"
                " WHERE u.obligation_id = delivery_obligations.obligation_id)"
                " AS unitized FROM delivery_obligations"
                " WHERE obligation_id = ?",
                (obligation_id,),
            ).fetchone()
        if row is None:
            return False
        expected = (
            session_key,
            platform,
            chat_id,
            thread_id or "",
            reply_to or "",
            content,
        )
        recorded = (
            str(row["session_key"]),
            str(row["platform"]),
            str(row["chat_id"]),
            str(row["thread_id"] or ""),
            str(row["reply_to"] or ""),
            str(row["content"]),
        )
        if recorded != expected or row["state"] not in {
            "pending",
            "attempting",
            "failed",
            "delivered",
            "abandoned",
        }:
            raise DeliveryPlanError(
                "delivery obligation identity collides with different durable data"
            )
        if row["state"] == "pending" and not bool(row["unitized"]):
            # A reservation with no exact channel plan has not yet accepted
            # responsibility for this reply. Its inbound claim must stay open.
            return False
        return True

    def mark_delivered(self, obligation_id: str) -> bool:
        return self._update(
            obligation_id,
            "delivered",
            expected_states=("attempting",),
            require_units_complete=True,
        )

    def mark_failed(
        self,
        obligation_id: str,
        error: str = "",
        *,
        retry_safe: bool = False,
        retry_after: Optional[float] = None,
    ) -> bool:
        now = time.time()
        next_attempt_at = 0.0
        if retry_safe and retry_after is not None:
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = 0.0
            if math.isfinite(delay) and delay > 0:
                next_attempt_at = now + delay
        return self._update(
            obligation_id,
            "failed",
            error,
            retry_safe=retry_safe,
            next_attempt_at=next_attempt_at,
            expected_states=("attempting",),
        )

    def mark_planned_failed(
        self,
        obligation_id: str,
        error: str = "",
        *,
        retry_safe: bool = False,
        retry_after: Optional[float] = None,
    ) -> bool:
        """Settle a unitized parent only after its units prove no send is live."""

        now = time.time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE delivery_obligations"
                " SET state = 'failed', updated_at = ?, last_error = ?,"
                " retry_safe = CASE WHEN EXISTS (SELECT 1 FROM delivery_units"
                " WHERE obligation_id = ? AND state = 'failed'"
                " AND retry_safe IS NOT 1) THEN 0 ELSE 1 END"
                " WHERE obligation_id = ? AND owner_token = ?"
                " AND state = 'attempting'"
                " AND EXISTS (SELECT 1 FROM delivery_units"
                " WHERE obligation_id = ? AND state = 'failed')"
                " AND NOT EXISTS (SELECT 1 FROM delivery_units"
                " WHERE obligation_id = ? AND state = 'attempting')"
                " AND NOT EXISTS (SELECT 1 FROM delivery_units"
                " WHERE obligation_id = ?"
                " AND state NOT IN ('pending', 'delivered', 'failed'))",
                (
                    now,
                    error[:500] or None,
                    obligation_id,
                    obligation_id,
                    self._owner_token,
                    obligation_id,
                    obligation_id,
                    obligation_id,
                ),
            )
            return bool(cursor.rowcount)

    def _update(
        self,
        obligation_id: str,
        state: str,
        error: str = "",
        *,
        retry_safe: bool = False,
        next_attempt_at: float = 0.0,
        expected_states: tuple[str, ...],
        require_units_complete: bool = False,
    ) -> bool:
        placeholders = ",".join("?" for _ in expected_states)
        with self._lock, self._connect() as connection:
            if require_units_complete:
                unit_counts = connection.execute(
                    "SELECT COUNT(*) AS total,"
                    " SUM(CASE WHEN state != 'delivered' THEN 1 ELSE 0 END)"
                    " AS incomplete FROM delivery_units WHERE obligation_id = ?",
                    (obligation_id,),
                ).fetchone()
                if (
                    unit_counts is not None
                    and int(unit_counts["total"] or 0) > 0
                    and int(unit_counts["incomplete"] or 0) > 0
                ):
                    return False
            cursor = connection.execute(
                "UPDATE delivery_obligations"
                " SET state = ?, updated_at = ?, last_error = ?, retry_safe = ?,"
                " next_attempt_at = ?"
                " WHERE obligation_id = ? AND owner_token = ?"
                f" AND state IN ({placeholders})",
                (
                    state,
                    time.time(),
                    error[:500] or None,
                    int(bool(retry_safe)),
                    float(next_attempt_at),
                    obligation_id,
                    self._owner_token,
                    *expected_states,
                ),
            )
            return bool(cursor.rowcount)

    def record_units(
        self,
        obligation_id: str,
        units: Sequence[DeliveryUnit],
    ) -> None:
        expected = [
            (unit.unit_id, unit.position, unit.kind, unit.fingerprint)
            for unit in units
        ]
        if not expected:
            raise DeliveryPlanError("delivery plan has no units")
        if len({unit_id for unit_id, *_ in expected}) != len(expected):
            raise DeliveryPlanError("delivery plan contains duplicate unit ids")

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            parent = connection.execute(
                "SELECT owner_token, state FROM delivery_obligations"
                " WHERE obligation_id = ?",
                (obligation_id,),
            ).fetchone()
            if (
                parent is None
                or parent["owner_token"] != self._owner_token
                or parent["state"] not in {"pending", "attempting"}
            ):
                raise DeliveryPlanError("delivery obligation is not owned and planable")

            existing = connection.execute(
                "SELECT unit_id, position, kind, fingerprint"
                " FROM delivery_units WHERE obligation_id = ?"
                " ORDER BY position ASC",
                (obligation_id,),
            ).fetchall()
            if existing:
                recorded = [
                    (
                        row["unit_id"],
                        int(row["position"]),
                        row["kind"],
                        row["fingerprint"],
                    )
                    for row in existing
                ]
                if recorded != expected:
                    raise DeliveryPlanError(
                        "delivery plan changed after its first attempt"
                    )
                return

            now = time.time()
            connection.executemany(
                "INSERT INTO delivery_units"
                " (obligation_id, unit_id, position, kind, fingerprint, state,"
                " attempts, retry_safe, evidence, last_error, updated_at)"
                " VALUES (?, ?, ?, ?, ?, 'pending', 0, 0, NULL, NULL, ?)",
                [
                    (
                        obligation_id,
                        unit_id,
                        position,
                        kind,
                        fingerprint,
                        now,
                    )
                    for unit_id, position, kind, fingerprint in expected
                ],
            )

    def unit_state(self, obligation_id: str, unit_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT state, retry_safe, evidence, attempts FROM delivery_units"
                " WHERE obligation_id = ? AND unit_id = ?",
                (obligation_id, unit_id),
            ).fetchone()
            return dict(row) if row is not None else None

    def mark_unit_attempting(self, obligation_id: str, unit_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE delivery_units SET state = 'attempting',"
                " attempts = attempts + 1, retry_safe = 0, last_error = NULL,"
                " updated_at = ? WHERE obligation_id = ? AND unit_id = ?"
                " AND (state = 'pending' OR (state = 'failed' AND retry_safe = 1))"
                " AND EXISTS (SELECT 1 FROM delivery_obligations"
                " WHERE obligation_id = ? AND owner_token = ?"
                " AND state = 'attempting')",
                (
                    time.time(),
                    obligation_id,
                    unit_id,
                    obligation_id,
                    self._owner_token,
                ),
            )
            return bool(cursor.rowcount)

    def mark_unit_delivered(
        self,
        obligation_id: str,
        unit_id: str,
        evidence: str,
    ) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE delivery_units SET state = 'delivered', retry_safe = 0,"
                " evidence = ?, last_error = NULL, updated_at = ?"
                " WHERE obligation_id = ? AND unit_id = ? AND state = 'attempting'"
                " AND EXISTS (SELECT 1 FROM delivery_obligations"
                " WHERE obligation_id = ? AND owner_token = ?"
                " AND state = 'attempting')",
                (
                    evidence[:200],
                    time.time(),
                    obligation_id,
                    unit_id,
                    obligation_id,
                    self._owner_token,
                ),
            )
            return bool(cursor.rowcount)

    def mark_unit_failed(
        self,
        obligation_id: str,
        unit_id: str,
        error: str,
        *,
        retry_safe: bool,
        retry_after: Optional[float] = None,
    ) -> bool:
        now = time.time()
        next_attempt_at = 0.0
        if retry_safe and retry_after is not None:
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = 0.0
            if math.isfinite(delay) and delay > 0:
                next_attempt_at = now + delay
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE delivery_units SET state = 'failed', retry_safe = ?,"
                " evidence = NULL, last_error = ?, updated_at = ?"
                " WHERE obligation_id = ? AND unit_id = ? AND state = 'attempting'"
                " AND EXISTS (SELECT 1 FROM delivery_obligations"
                " WHERE obligation_id = ? AND owner_token = ?"
                " AND state = 'attempting')",
                (
                    int(bool(retry_safe)),
                    error[:500] or None,
                    now,
                    obligation_id,
                    unit_id,
                    obligation_id,
                    self._owner_token,
                ),
            )
            if cursor.rowcount and next_attempt_at > 0:
                parent_cursor = connection.execute(
                    "UPDATE delivery_obligations"
                    " SET next_attempt_at = MAX(next_attempt_at, ?)"
                    " WHERE obligation_id = ? AND owner_token = ?"
                    " AND state = 'attempting'",
                    (
                        next_attempt_at,
                        obligation_id,
                        self._owner_token,
                    ),
                )
                if parent_cursor.rowcount != 1:
                    raise sqlite3.IntegrityError(
                        "delivery unit retry deadline lost its parent obligation"
                    )
            return bool(cursor.rowcount)

    def has_unit_plan(self, obligation_id: str) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM delivery_units WHERE obligation_id = ? LIMIT 1",
                (obligation_id,),
            ).fetchone()
            return row is not None

    def claim_recoverable(
        self,
        platforms: set[str],
        *,
        now: Optional[float] = None,
    ) -> list[Dict[str, Any]]:
        current_time = time.time() if now is None else float(now)
        claimed: list[Dict[str, Any]] = []
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT obligation_id, session_key, platform, chat_id,"
                " thread_id, reply_to, content, state, attempts, created_at, updated_at,"
                " owner_token, retry_safe, next_attempt_at"
                " FROM delivery_obligations"
                " WHERE state IN ('pending', 'attempting', 'failed')"
                " ORDER BY updated_at ASC",
            ).fetchall()
            for row in rows:
                if len(claimed) >= MAX_RECOVERY_BATCH:
                    break
                if row["owner_token"] == self._owner_token:
                    continue
                if float(row["next_attempt_at"] or 0) > current_time:
                    continue
                unit_rows = connection.execute(
                    "SELECT state, retry_safe FROM delivery_units"
                    " WHERE obligation_id = ?",
                    (row["obligation_id"],),
                ).fetchall()
                unitized = bool(unit_rows)
                if row["state"] == "attempting" and (
                    not unitized
                    or any(
                        not (
                            unit["state"] in {"pending", "delivered"}
                            or (
                                unit["state"] == "failed"
                                and unit["retry_safe"] == 1
                            )
                        )
                        for unit in unit_rows
                    )
                ):
                    # No unit plan, an in-flight unit, or an unsafe unit leaves
                    # platform acceptance unknown. Never turn a restart into a
                    # duplicate.
                    continue
                if row["state"] == "failed" and row["retry_safe"] != 1:
                    # The platform may already have accepted this send. A
                    # restart is not evidence that duplicating it is safe.
                    continue
                unattempted = row["state"] == "pending"
                counted_attempt = row["state"] == "failed" or any(
                    unit["state"] == "failed" for unit in unit_rows
                )
                if not unattempted and (
                    (
                        counted_attempt
                        and int(row["attempts"]) >= MAX_ATTEMPTS
                    )
                    or current_time - float(row["created_at"])
                    > STALE_AFTER_SECONDS
                ):
                    connection.execute(
                        "UPDATE delivery_obligations"
                        " SET state = 'abandoned', updated_at = ?"
                        " WHERE obligation_id = ?",
                        (current_time, row["obligation_id"]),
                    )
                    continue
                if row["platform"] not in platforms:
                    continue
                claimed_state = (
                    "pending" if unattempted or not unitized else "attempting"
                )
                attempt_increment = int(counted_attempt)
                cursor = connection.execute(
                    "UPDATE delivery_obligations"
                    " SET owner_token = ?, state = ?,"
                    " attempts = attempts + ?, updated_at = ?, next_attempt_at = 0,"
                    " retry_safe = 0, last_error = NULL"
                    " WHERE obligation_id = ? AND owner_token = ?"
                    " AND state = ? AND updated_at = ?",
                    (
                        self._owner_token,
                        claimed_state,
                        attempt_increment,
                        current_time,
                        row["obligation_id"],
                        row["owner_token"],
                        row["state"],
                        row["updated_at"],
                    ),
                )
                if cursor.rowcount:
                    if claimed_state == "pending":
                        self._active_recovery_claims.add(str(row["obligation_id"]))
                    claimed.append(
                        {
                            "obligation_id": row["obligation_id"],
                            "session_key": row["session_key"],
                            "platform": row["platform"],
                            "chat_id": row["chat_id"],
                            "thread_id": row["thread_id"] or "",
                            "reply_to": row["reply_to"] or "",
                            "content": row["content"],
                            # Every claimable ununitized failure is explicitly
                            # retry-safe, so platform acceptance was disproved.
                            # Keeping the exact original content also makes a
                            # plan-before-activation crash replay-identical.
                            "needs_marker": False,
                            "unitized": unitized,
                            "attempts": int(row["attempts"]) + attempt_increment,
                        }
                    )
            self._prune(connection, current_time)
        return claimed

    def claim_live_failed(
        self,
        platforms: set[str],
        *,
        now: Optional[float] = None,
        min_age_seconds: float = LIVE_RETRY_MIN_AGE_SECONDS,
    ) -> list[Dict[str, Any]]:
        """Claim proven-unsent work owned by this live runtime."""

        current_time = time.time() if now is None else float(now)
        minimum_age = max(0.0, float(min_age_seconds))
        claimed: list[Dict[str, Any]] = []
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT obligation_id, session_key, platform, chat_id,"
                " thread_id, reply_to, content, state, attempts, created_at, updated_at,"
                " owner_token, next_attempt_at FROM delivery_obligations"
                " WHERE (state = 'pending' OR"
                " state = 'attempting' OR"
                " (state = 'failed' AND retry_safe = 1))"
                " AND owner_token = ? ORDER BY updated_at ASC",
                (self._owner_token,),
            ).fetchall()
            for row in rows:
                if len(claimed) >= MAX_RECOVERY_BATCH:
                    break
                if str(row["obligation_id"]) in self._active_recovery_claims:
                    continue
                if float(row["next_attempt_at"] or 0) > current_time:
                    continue
                if current_time - float(row["updated_at"]) < minimum_age:
                    continue
                unit_rows = connection.execute(
                    "SELECT state, retry_safe FROM delivery_units"
                    " WHERE obligation_id = ?",
                    (row["obligation_id"],),
                ).fetchall()
                unitized = bool(unit_rows)
                if row["state"] == "attempting" and (
                    not unitized
                    or any(
                        not (
                            unit["state"] in {"pending", "delivered"}
                            or (
                                unit["state"] == "failed"
                                and unit["retry_safe"] == 1
                            )
                        )
                        for unit in unit_rows
                    )
                ):
                    # An in-flight or unsafe unit may already have reached the
                    # platform. Only a fully safe plan can be retried live.
                    continue
                unattempted = row["state"] == "pending"
                counted_attempt = row["state"] == "failed" or any(
                    unit["state"] == "failed" for unit in unit_rows
                )
                if not unattempted and (
                    (
                        counted_attempt
                        and int(row["attempts"]) >= MAX_ATTEMPTS
                    )
                    or current_time - float(row["created_at"])
                    > STALE_AFTER_SECONDS
                ):
                    connection.execute(
                        "UPDATE delivery_obligations"
                        " SET state = 'abandoned', updated_at = ?"
                        " WHERE obligation_id = ? AND state = ?"
                        " AND owner_token = ? AND updated_at = ?",
                        (
                            current_time,
                            row["obligation_id"],
                            row["state"],
                            self._owner_token,
                            row["updated_at"],
                        ),
                    )
                    continue
                if row["platform"] not in platforms:
                    continue
                claimed_state = (
                    "pending" if unattempted or not unitized else "attempting"
                )
                attempt_increment = int(counted_attempt)
                cursor = connection.execute(
                    "UPDATE delivery_obligations"
                    " SET state = ?, attempts = attempts + ?,"
                    " updated_at = ?, next_attempt_at = 0, retry_safe = 0,"
                    " last_error = NULL"
                    " WHERE obligation_id = ? AND state = ?"
                    " AND owner_token = ? AND updated_at = ?",
                    (
                        claimed_state,
                        attempt_increment,
                        current_time,
                        row["obligation_id"],
                        row["state"],
                        self._owner_token,
                        row["updated_at"],
                    ),
                )
                if cursor.rowcount:
                    if claimed_state == "pending":
                        self._active_recovery_claims.add(str(row["obligation_id"]))
                    claimed.append(
                        {
                            "obligation_id": row["obligation_id"],
                            "session_key": row["session_key"],
                            "platform": row["platform"],
                            "chat_id": row["chat_id"],
                            "thread_id": row["thread_id"] or "",
                            "reply_to": row["reply_to"] or "",
                            "content": row["content"],
                            # This owner is still live and the failure was
                            # proven unsent. A restart/duplicate warning would
                            # be false and is not needed for safety.
                            "needs_marker": False,
                            "unitized": unitized,
                            "attempts": int(row["attempts"]) + attempt_increment,
                        }
                    )
            self._prune(connection, current_time)
        return claimed

    @staticmethod
    def _prune(connection: sqlite3.Connection, now: float) -> None:
        # No non-pending send remains live beyond the bounded recovery window.
        # This global sweep also reaches unsafe outcomes, disabled platforms,
        # old owners, and retry-after deadlines that the claim loops must skip.
        # Pending work stays intact because it is proven unsent.
        connection.execute(
            "UPDATE delivery_obligations"
            " SET state = 'abandoned', updated_at = ?, retry_safe = 0,"
            " next_attempt_at = 0"
            " WHERE state IN ('attempting', 'failed') AND created_at < ?",
            (now, now - STALE_AFTER_SECONDS),
        )
        connection.execute(
            "DELETE FROM delivery_obligations"
            " WHERE state IN ('delivered', 'abandoned') AND updated_at < ?",
            (now - RETENTION_SECONDS,),
        )
        total = int(
            connection.execute(
                "SELECT COUNT(*) FROM delivery_obligations"
            ).fetchone()[0]
        )
        excess = max(0, total - MAX_ROWS)
        if excess:
            connection.execute(
                "DELETE FROM delivery_obligations WHERE obligation_id IN ("
                " SELECT obligation_id FROM delivery_obligations"
                " WHERE state = 'delivered'"
                " ORDER BY updated_at ASC LIMIT ?"
                ")",
                (excess,),
            )
        unresolved = int(
            connection.execute(
                "SELECT COUNT(*) FROM delivery_obligations"
                " WHERE state IN ('pending', 'attempting', 'failed')"
            ).fetchone()[0]
        )
        # Retained abandoned rows are dedupe evidence, not recoverable work.
        # Counting them here would let safety fences permanently prevent new
        # replies; the hard cap still bounds every active obligation.
        if unresolved > MAX_ROWS:
            raise DeliveryPlanError(
                "delivery ledger capacity is exhausted by unresolved obligations"
            )
        connection.execute(
            "DELETE FROM delivery_units WHERE obligation_id NOT IN"
            " (SELECT obligation_id FROM delivery_obligations)"
        )

    @staticmethod
    def _prune_commands(connection: sqlite3.Connection, now: float) -> None:
        connection.execute(
            "DELETE FROM command_outcomes"
            " WHERE state = 'completed' AND updated_at < ?",
            (now - RETENTION_SECONDS,),
        )
        total = int(
            connection.execute("SELECT COUNT(*) FROM command_outcomes").fetchone()[0]
        )
        excess = max(0, total - MAX_ROWS)
        if excess:
            connection.execute(
                "DELETE FROM command_outcomes WHERE command_id IN ("
                " SELECT command_id FROM command_outcomes"
                " WHERE state = 'completed' ORDER BY updated_at ASC LIMIT ?"
                ")",
                (excess,),
            )
        remaining = int(
            connection.execute("SELECT COUNT(*) FROM command_outcomes").fetchone()[0]
        )
        if remaining > MAX_ROWS:
            raise DeliveryPlanError(
                "command ledger capacity is exhausted by unresolved commands"
            )


class DeliveryUnitLedger:
    """Durable proof for the exact chunks/files inside one obligation."""

    def __init__(self, store: DeliveryStore, obligation_id: str):
        self.store = store
        self.obligation_id = obligation_id
        self._prepared = False
        self._preparation_failed = False
        self._activated = False

    @property
    def prepared(self) -> bool:
        return self._prepared

    @property
    def preparation_failed(self) -> bool:
        return self._preparation_failed

    async def prepare(
        self,
        descriptors: Sequence[tuple[str, str]],
    ) -> list[DeliveryUnit]:
        units = [
            DeliveryUnit(
                unit_id=hashlib.sha256(
                    (
                        f"{self.obligation_id}\0{position}\0{kind}\0{fingerprint}"
                    ).encode("utf-8", "replace")
                ).hexdigest()[:32],
                position=position,
                kind=str(kind),
                fingerprint=str(fingerprint),
            )
            for position, (kind, fingerprint) in enumerate(descriptors)
        ]
        try:
            await asyncio.to_thread(
                self.store.record_units,
                self.obligation_id,
                units,
            )
        except Exception:
            self._preparation_failed = True
            raise
        self._prepared = True
        return units

    async def run(
        self,
        unit: DeliveryUnit,
        send: Callable[[], Awaitable[Any]],
    ) -> SendResult:
        if not self._prepared:
            return SendResult(False, "delivery unit plan was not prepared")
        try:
            state = await asyncio.to_thread(
                self.store.unit_state,
                self.obligation_id,
                unit.unit_id,
            )
            if state is None:
                return SendResult(False, "delivery unit is missing from its ledger")
            if state["state"] == "delivered":
                return SendResult(
                    True,
                    message_id=str(state.get("evidence") or ""),
                )
            if not self._activated:
                activated = await asyncio.to_thread(
                    self.store.activate_unit_plan,
                    self.obligation_id,
                )
                if not activated:
                    return SendResult(
                        False,
                        "delivery unit plan could not be durably activated",
                    )
                self._activated = True
            claimed = await asyncio.to_thread(
                self.store.mark_unit_attempting,
                self.obligation_id,
                unit.unit_id,
            )
            if not claimed:
                return SendResult(
                    False,
                    "delivery unit is not safely retryable",
                )
        except Exception as exc:
            logger.exception("Could not claim delivery unit %s", unit.unit_id)
            return SendResult(False, str(exc))

        try:
            result = as_send_result(await send())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = SendResult(False, str(exc))

        try:
            if result.success:
                evidence = result.message_id or f"accepted:{unit.unit_id}"
                updated = await asyncio.to_thread(
                    self.store.mark_unit_delivered,
                    self.obligation_id,
                    unit.unit_id,
                    evidence,
                )
            else:
                updated = await asyncio.to_thread(
                    self.store.mark_unit_failed,
                    self.obligation_id,
                    unit.unit_id,
                    result.error or "send failed",
                    retry_safe=result.retryable,
                    retry_after=result.retry_after,
                )
        except Exception:
            logger.exception("Could not settle delivery unit %s", unit.unit_id)
            updated = False

        if not updated:
            # The platform may have accepted the unit while local proof failed.
            # Do not retry it or advance to later units.
            return SendResult(False, "delivery unit settlement was not durable")
        if not result.success:
            # Parent failure authority travels only with the exact result whose
            # unit failure this ledger instance durably persisted. A duplicate
            # caller which lost the unit claim must never settle the parent.
            return replace(result, _unit_failure_recorded=True)
        return result


async def send_with_retry(
    send: Callable[[], Awaitable[Any]],
    *,
    max_retries: int = 2,
    base_delay: float = 2.0,
    deadline: Optional[float] = None,
) -> SendResult:
    """Retry only proven-safe failures, optionally within one deadline."""

    loop = asyncio.get_running_loop()

    async def attempt() -> SendResult:
        try:
            if deadline is None:
                return as_send_result(await send())
            remaining = deadline - loop.time()
            if remaining <= 0:
                return SendResult(False, "delivery deadline expired")
            return as_send_result(
                await asyncio.wait_for(send(), timeout=remaining)
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            # Cancellation can race with platform acceptance.  The result is
            # deliberately unsafe so the caller cannot retry an unknown send.
            return SendResult(False, "delivery deadline expired")
        except Exception as exc:
            logger.warning("Outbound send raised: %s", exc)
            # Only a channel result can prove that retrying is safe. An
            # unclassified exception may have happened after platform accept.
            return SendResult(False, str(exc))

    result = await attempt()
    for retry in range(1, max_retries + 1):
        if result.success or not result.retryable:
            break
        if result.retry_after is not None:
            try:
                server_delay = float(result.retry_after)
            except (TypeError, ValueError):
                break
            if not math.isfinite(server_delay) or server_delay < 0:
                break
        else:
            server_delay = None
        if server_delay is not None and server_delay > MAX_INLINE_RETRY_AFTER_SECONDS:
            logger.warning(
                "Outbound send retry_after %.1fs exceeds the %.1fs inline cap; "
                "deferring through the durable delivery obligation.",
                server_delay,
                MAX_INLINE_RETRY_AFTER_SECONDS,
            )
            break
        delay = (
            server_delay + random.uniform(0, 1)
            if server_delay is not None
            else base_delay * (2 ** (retry - 1)) + random.uniform(0, 1)
        )
        delay = max(0.0, delay)
        if deadline is not None and delay >= deadline - loop.time():
            logger.warning(
                "Outbound send retry would exceed its deadline: %s",
                result.error,
            )
            break
        logger.warning(
            "Outbound send failed; retrying in %.1fs (%d/%d): %s",
            delay,
            retry,
            max_retries,
            result.error,
        )
        await asyncio.sleep(delay)
        result = await attempt()
    return result


async def deliver_final(
    store: DeliveryStore,
    *,
    session_key: str,
    message_ref: str,
    platform: str,
    chat_id: str,
    thread_id: str,
    content: str,
    send: Callable[[], Awaitable[Any]],
    reply_to: str = "",
    ledger_send: Optional[
        Callable[[DeliveryUnitLedger], Awaitable[Any]]
    ] = None,
) -> SendResult:
    """Record, attempt, and resolve one final-response obligation."""

    obligation_id = compute_obligation_id(session_key, message_ref, content)
    ledger = (
        DeliveryUnitLedger(store, obligation_id)
        if ledger_send is not None
        else None
    )
    recorded = False
    try:
        recorded_state = await asyncio.to_thread(
            store.record,
            obligation_id=obligation_id,
            session_key=session_key,
            platform=platform,
            chat_id=chat_id,
            thread_id=thread_id,
            content=content,
            reply_to=reply_to,
        )
        # Production channel sends first persist their exact text/file plan.
        # The ledger activates this parent immediately before its first unit can
        # reach the network. Generic callers retain the original parent fence.
        recorded = (
            recorded_state == "pending"
            if ledger is not None
            else bool(await asyncio.to_thread(store.mark_attempting, obligation_id))
        )
    except Exception:
        logger.exception("Could not record final-response delivery obligation")

    if not recorded:
        return SendResult(
            False,
            "final-response delivery obligation was not durably claimed",
        )

    result = await send_with_retry(
        (lambda: ledger_send(ledger)) if ledger is not None else send
    )
    if ledger is not None and ledger.preparation_failed:
        try:
            await asyncio.to_thread(store.discard_unplanned, obligation_id)
        except Exception:
            logger.exception("Could not discard an unplanned delivery obligation")
        return SendResult(False, "final-response delivery plan was not durable")

    if ledger is not None and not ledger.prepared:
        # A callback which accepts a ledger but does not prepare it violated the
        # production contract and may already have touched the network. Fence
        # it unsafe; never convert that unknown outcome into a retryable send.
        try:
            await asyncio.to_thread(
                store.quarantine_unplanned,
                obligation_id,
                "delivery callback returned without preparing its unit plan",
            )
        except Exception:
            logger.exception("Could not quarantine an unplanned delivery callback")
        return SendResult(False, "final-response delivery plan was not prepared")
    try:
        if result.success:
            updated = await asyncio.to_thread(
                store.mark_delivered, obligation_id
            )
            if not updated:
                if ledger is None:
                    await asyncio.to_thread(
                        store.mark_failed,
                        obligation_id,
                        "delivery completion was not durable",
                    )
                result = SendResult(
                    False,
                    "delivery completion was not durable",
                )
        else:
            if ledger is not None and not result._unit_failure_recorded:
                # The callback did not durably settle a failed unit. It may be
                # a duplicate loser or a pre-run exception; either way another
                # owner (or recovery) remains responsible for the parent.
                return result
            updated = await asyncio.to_thread(
                store.mark_planned_failed if ledger is not None else store.mark_failed,
                obligation_id,
                result.error or "send failed",
                retry_safe=result.retryable,
                retry_after=result.retry_after,
            )
            if not updated:
                result = SendResult(
                    False,
                    "delivery failure was not durably recorded",
                )
    except Exception:
        logger.exception("Could not update final-response delivery obligation")
        result = SendResult(
            False,
            "delivery settlement was not durable",
        )
    return result


async def claim_deliveries(
    store: DeliveryStore,
    platforms: set[str],
    *,
    now: Optional[float] = None,
) -> list[Dict[str, Any]]:
    """Claim obligations left by an earlier runtime process."""

    return await asyncio.to_thread(
        store.claim_recoverable,
        platforms,
        now=now,
    )


async def claim_live_deliveries(
    store: DeliveryStore,
    platforms: set[str],
    *,
    now: Optional[float] = None,
    min_age_seconds: float = LIVE_RETRY_MIN_AGE_SECONDS,
) -> list[Dict[str, Any]]:
    """Claim bounded, proven-unsent work from this running process."""

    try:
        return await asyncio.to_thread(
            store.claim_live_failed,
            platforms,
            now=now,
            min_age_seconds=min_age_seconds,
        )
    except Exception:
        logger.exception("Could not claim live final-response retries")
        return []


def _accepts_delivery_ledger(channel: Any) -> bool:
    """Keep legacy/local adapters usable without weakening production sends."""

    try:
        parameters = inspect.signature(channel.send).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "delivery_ledger"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


async def redeliver_claimed_deliveries(
    store: DeliveryStore,
    channels: Mapping[str, Any],
    rows: list[Dict[str, Any]],
) -> int:
    """Redeliver already claimed obligations."""

    delivered = 0
    for row in rows:
        channel = channels.get(row["platform"])
        if channel is None:
            store.release_recovery_claim(row["obligation_id"])
            continue
        try:
            content = row["content"]
            if row["needs_marker"]:
                content = RECOVERED_MARKER + content
            ledger = (
                DeliveryUnitLedger(store, row["obligation_id"])
                if _accepts_delivery_ledger(channel)
                else None
            )
            if ledger is None and row.get("unitized"):
                logger.error(
                    "Refusing to recover a unitized delivery through an adapter"
                    " without delivery-ledger support"
                )
                continue
            if ledger is None and not row.get("unitized"):
                # Compatibility for non-channel/local adapters. WhatsApp and
                # Telegram always take the planned path below.
                try:
                    claimed = await asyncio.to_thread(
                        store.mark_attempting,
                        row["obligation_id"],
                    )
                except Exception:
                    logger.exception("Could not claim generic recovered delivery")
                    continue
                if not claimed:
                    continue
            try:
                if row["platform"] == "telegram":
                    kwargs: Dict[str, Any] = {"thread_id": row["thread_id"]}
                    if ledger is not None:
                        kwargs["delivery_ledger"] = ledger
                    if row.get("reply_to"):
                        value = await channel.send(
                            row["chat_id"], content, row["reply_to"], **kwargs
                        )
                    else:
                        value = await channel.send(row["chat_id"], content, **kwargs)
                else:
                    kwargs = {}
                    if ledger is not None:
                        kwargs["delivery_ledger"] = ledger
                    if row.get("reply_to"):
                        value = await channel.send(
                            row["chat_id"], content, row["reply_to"], **kwargs
                        )
                    else:
                        value = await channel.send(row["chat_id"], content, **kwargs)
                result = as_send_result(value)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = SendResult(False, str(exc))
            if ledger is not None and ledger.preparation_failed:
                # Planning happens before any network unit. A safe ununitized
                # row remains pending for a later healthy sweep.
                continue
            if ledger is not None and not ledger.prepared:
                # It accepted the production ledger contract but may have sent
                # without using it. Quarantine that unknown outcome.
                try:
                    await asyncio.to_thread(
                        store.quarantine_unplanned,
                        row["obligation_id"],
                        "recovery callback returned without preparing its unit plan",
                    )
                except Exception:
                    logger.exception("Could not quarantine recovered delivery")
                continue
            try:
                if result.success:
                    updated = await asyncio.to_thread(
                        store.mark_delivered, row["obligation_id"]
                    )
                    if updated:
                        delivered += 1
                    elif ledger is None:
                        await asyncio.to_thread(
                            store.mark_failed,
                            row["obligation_id"],
                            "recovered delivery completion was not durable",
                        )
                else:
                    if ledger is not None and not result._unit_failure_recorded:
                        # Losing a unit claim or failing before ledger.run is
                        # not authority to change the shared parent state.
                        continue
                    await asyncio.to_thread(
                        (
                            store.mark_planned_failed
                            if ledger is not None
                            else store.mark_failed
                        ),
                        row["obligation_id"],
                        result.error or "send failed",
                        retry_safe=result.retryable,
                        retry_after=result.retry_after,
                    )
            except Exception:
                logger.exception("Could not update recovered delivery obligation")
        finally:
            store.release_recovery_claim(row["obligation_id"])
    return delivered


async def recover_deliveries(
    store: DeliveryStore,
    channels: Mapping[str, Any],
) -> int:
    """Redeliver obligations left by an earlier runtime process."""

    rows = await claim_deliveries(store, set(channels))
    return await redeliver_claimed_deliveries(store, channels, rows)


async def recover_live_deliveries(
    store: DeliveryStore,
    channels: Mapping[str, Any],
    *,
    now: Optional[float] = None,
    min_age_seconds: float = LIVE_RETRY_MIN_AGE_SECONDS,
) -> int:
    """Retry due, proven-unsent work while the runtime stays live."""

    rows = await claim_live_deliveries(
        store,
        set(channels),
        now=now,
        min_age_seconds=min_age_seconds,
    )
    # A restart can happen before a long server-requested delay expires. The
    # resident sweep must claim that prior owner's row once it becomes due;
    # startup-only recovery would otherwise strand it until another restart.
    rows.extend(await claim_deliveries(store, set(channels), now=now))
    return await redeliver_claimed_deliveries(store, channels, rows)


__all__ = [
    "CommandOutcome",
    "DeliveryPlanError",
    "DeliveryStore",
    "DeliveryUnit",
    "DeliveryUnitLedger",
    "RECOVERED_MARKER",
    "SendResult",
    "as_send_result",
    "claim_deliveries",
    "claim_live_deliveries",
    "compute_command_id",
    "compute_obligation_id",
    "delivery_fingerprint",
    "deliver_final",
    "file_delivery_fingerprint",
    "redeliver_claimed_deliveries",
    "recover_deliveries",
    "recover_live_deliveries",
    "send_with_retry",
]
