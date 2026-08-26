"""Durable final-response delivery, reduced from Hermes' gateway mechanism."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterator, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_ROWS = 500
LIVE_RETRY_MIN_AGE_SECONDS = 60.0
MAX_RECOVERY_BATCH = 20
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
    retry_safe    INTEGER NOT NULL DEFAULT 0
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
"""


@dataclass(frozen=True)
class SendResult:
    """The platform-neutral part of Hermes' outbound send result."""

    success: bool
    error: str = ""
    retryable: bool = False
    retry_after: Optional[float] = None
    message_id: str = ""

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
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO delivery_obligations"
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
            self._prune(connection, now)

    def mark_attempting(self, obligation_id: str) -> bool:
        return self._update(
            obligation_id,
            "attempting",
            expected_states=("pending",),
        )

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
    ) -> bool:
        return self._update(
            obligation_id,
            "failed",
            error,
            retry_safe=retry_safe,
            expected_states=("attempting",),
        )

    def _update(
        self,
        obligation_id: str,
        state: str,
        error: str = "",
        *,
        retry_safe: bool = False,
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
                " SET state = ?, updated_at = ?, last_error = ?, retry_safe = ?"
                " WHERE obligation_id = ? AND owner_token = ?"
                f" AND state IN ({placeholders})",
                (
                    state,
                    time.time(),
                    error[:500] or None,
                    int(bool(retry_safe)),
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
            parent = connection.execute(
                "SELECT owner_token, state FROM delivery_obligations"
                " WHERE obligation_id = ?",
                (obligation_id,),
            ).fetchone()
            if (
                parent is None
                or parent["owner_token"] != self._owner_token
                or parent["state"] != "attempting"
            ):
                raise DeliveryPlanError("delivery obligation is not owned and active")

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
    ) -> bool:
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
                    time.time(),
                    obligation_id,
                    unit_id,
                    obligation_id,
                    self._owner_token,
                ),
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
                " owner_token, retry_safe"
                " FROM delivery_obligations"
                " WHERE state IN ('pending', 'attempting', 'failed')"
                " ORDER BY updated_at ASC",
            ).fetchall()
            for row in rows:
                if len(claimed) >= MAX_RECOVERY_BATCH:
                    break
                if row["owner_token"] == self._owner_token:
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
                        unit["state"] == "attempting"
                        or (
                            unit["state"] == "failed"
                            and not bool(unit["retry_safe"])
                        )
                        for unit in unit_rows
                    )
                ):
                    # No unit plan, an in-flight unit, or an unsafe unit leaves
                    # platform acceptance unknown. Never turn a restart into a
                    # duplicate.
                    continue
                if row["state"] == "failed" and not bool(row["retry_safe"]):
                    # The platform may already have accepted this send. A
                    # restart is not evidence that duplicating it is safe.
                    continue
                if (
                    int(row["attempts"]) >= MAX_ATTEMPTS
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
                cursor = connection.execute(
                    "UPDATE delivery_obligations"
                    " SET owner_token = ?, state = 'attempting',"
                    " attempts = attempts + 1, updated_at = ?"
                    " WHERE obligation_id = ? AND owner_token = ?"
                    " AND state = ? AND updated_at = ?",
                    (
                        self._owner_token,
                        current_time,
                        row["obligation_id"],
                        row["owner_token"],
                        row["state"],
                        row["updated_at"],
                    ),
                )
                if cursor.rowcount:
                    claimed.append(
                        {
                            "obligation_id": row["obligation_id"],
                            "session_key": row["session_key"],
                            "platform": row["platform"],
                            "chat_id": row["chat_id"],
                            "thread_id": row["thread_id"] or "",
                            "reply_to": row["reply_to"] or "",
                            "content": row["content"],
                            "needs_marker": (
                                row["state"] != "pending" and not unitized
                            ),
                            "unitized": unitized,
                            "attempts": int(row["attempts"]) + 1,
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
        """Claim proven retry-safe failures owned by this live runtime."""

        current_time = time.time() if now is None else float(now)
        minimum_age = max(0.0, float(min_age_seconds))
        claimed: list[Dict[str, Any]] = []
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT obligation_id, session_key, platform, chat_id,"
                " thread_id, reply_to, content, state, attempts, created_at, updated_at,"
                " owner_token FROM delivery_obligations"
                " WHERE state = 'failed' AND retry_safe = 1"
                " AND owner_token = ? ORDER BY updated_at ASC LIMIT ?",
                (self._owner_token, MAX_RECOVERY_BATCH),
            ).fetchall()
            for row in rows:
                if current_time - float(row["updated_at"]) < minimum_age:
                    continue
                if (
                    int(row["attempts"]) >= MAX_ATTEMPTS
                    or current_time - float(row["created_at"])
                    > STALE_AFTER_SECONDS
                ):
                    connection.execute(
                        "UPDATE delivery_obligations"
                        " SET state = 'abandoned', updated_at = ?"
                        " WHERE obligation_id = ? AND state = 'failed'"
                        " AND owner_token = ? AND updated_at = ?",
                        (
                            current_time,
                            row["obligation_id"],
                            self._owner_token,
                            row["updated_at"],
                        ),
                    )
                    continue
                if row["platform"] not in platforms:
                    continue
                cursor = connection.execute(
                    "UPDATE delivery_obligations"
                    " SET state = 'attempting', attempts = attempts + 1,"
                    " updated_at = ?"
                    " WHERE obligation_id = ? AND state = 'failed'"
                    " AND owner_token = ? AND updated_at = ?",
                    (
                        current_time,
                        row["obligation_id"],
                        self._owner_token,
                        row["updated_at"],
                    ),
                )
                if cursor.rowcount:
                    unitized = bool(
                        connection.execute(
                            "SELECT 1 FROM delivery_units"
                            " WHERE obligation_id = ? LIMIT 1",
                            (row["obligation_id"],),
                        ).fetchone()
                    )
                    claimed.append(
                        {
                            "obligation_id": row["obligation_id"],
                            "session_key": row["session_key"],
                            "platform": row["platform"],
                            "chat_id": row["chat_id"],
                            "thread_id": row["thread_id"] or "",
                            "reply_to": row["reply_to"] or "",
                            "content": row["content"],
                            "needs_marker": not unitized,
                            "unitized": unitized,
                            "attempts": int(row["attempts"]) + 1,
                        }
                    )
            self._prune(connection, current_time)
        return claimed

    @staticmethod
    def _prune(connection: sqlite3.Connection, now: float) -> None:
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
                " ORDER BY CASE state WHEN 'delivered' THEN 0"
                " WHEN 'abandoned' THEN 1 ELSE 2 END, updated_at ASC LIMIT ?"
                ")",
                (excess,),
            )
        connection.execute(
            "DELETE FROM delivery_units WHERE obligation_id NOT IN"
            " (SELECT obligation_id FROM delivery_obligations)"
        )


class DeliveryUnitLedger:
    """Durable proof for the exact chunks/files inside one obligation."""

    def __init__(self, store: DeliveryStore, obligation_id: str):
        self.store = store
        self.obligation_id = obligation_id

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
        await asyncio.to_thread(
            self.store.record_units,
            self.obligation_id,
            units,
        )
        return units

    async def run(
        self,
        unit: DeliveryUnit,
        send: Callable[[], Awaitable[Any]],
    ) -> SendResult:
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
                )
        except Exception:
            logger.exception("Could not settle delivery unit %s", unit.unit_id)
            updated = False

        if not updated:
            # The platform may have accepted the unit while local proof failed.
            # Do not retry it or advance to later units.
            return SendResult(False, "delivery unit settlement was not durable")
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
        delay = (
            result.retry_after + random.uniform(0, 1)
            if result.retry_after is not None
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
    recorded = False
    try:
        await asyncio.to_thread(
            store.record,
            obligation_id=obligation_id,
            session_key=session_key,
            platform=platform,
            chat_id=chat_id,
            thread_id=thread_id,
            content=content,
            reply_to=reply_to,
        )
        recorded = bool(
            await asyncio.to_thread(store.mark_attempting, obligation_id)
        )
    except Exception:
        logger.exception("Could not record final-response delivery obligation")

    if not recorded:
        return SendResult(
            False,
            "final-response delivery obligation was not durably claimed",
        )

    ledger = (
        DeliveryUnitLedger(store, obligation_id)
        if ledger_send is not None
        else None
    )
    result = await send_with_retry(
        (lambda: ledger_send(ledger)) if ledger is not None else send
    )
    try:
        if result.success:
            updated = await asyncio.to_thread(
                store.mark_delivered, obligation_id
            )
            if not updated:
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
            updated = await asyncio.to_thread(
                store.mark_failed,
                obligation_id,
                result.error or "send failed",
                retry_safe=result.retryable,
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
) -> list[Dict[str, Any]]:
    """Claim obligations left by an earlier runtime process."""

    try:
        return await asyncio.to_thread(store.claim_recoverable, platforms)
    except Exception:
        logger.exception("Could not claim pending final-response deliveries")
        return []


async def claim_live_deliveries(
    store: DeliveryStore,
    platforms: set[str],
    *,
    now: Optional[float] = None,
    min_age_seconds: float = LIVE_RETRY_MIN_AGE_SECONDS,
) -> list[Dict[str, Any]]:
    """Claim bounded, proven-safe failures from this running process."""

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
            continue
        content = row["content"]
        if row["needs_marker"]:
            content = RECOVERED_MARKER + content
        ledger = (
            DeliveryUnitLedger(store, row["obligation_id"])
            if row.get("unitized")
            else None
        )
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
        try:
            if result.success:
                updated = await asyncio.to_thread(
                    store.mark_delivered, row["obligation_id"]
                )
                if updated:
                    delivered += 1
                else:
                    await asyncio.to_thread(
                        store.mark_failed,
                        row["obligation_id"],
                        "recovered delivery completion was not durable",
                    )
            else:
                await asyncio.to_thread(
                    store.mark_failed,
                    row["obligation_id"],
                    result.error or "send failed",
                    retry_safe=result.retryable,
                )
        except Exception:
            logger.exception("Could not update recovered delivery obligation")
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
    """Retry only safely classed failures while their owner is still live."""

    rows = await claim_live_deliveries(
        store,
        set(channels),
        now=now,
        min_age_seconds=min_age_seconds,
    )
    return await redeliver_claimed_deliveries(store, channels, rows)


__all__ = [
    "DeliveryPlanError",
    "DeliveryStore",
    "DeliveryUnit",
    "DeliveryUnitLedger",
    "RECOVERED_MARKER",
    "SendResult",
    "as_send_result",
    "claim_deliveries",
    "claim_live_deliveries",
    "compute_obligation_id",
    "delivery_fingerprint",
    "deliver_final",
    "file_delivery_fingerprint",
    "redeliver_claimed_deliveries",
    "recover_deliveries",
    "recover_live_deliveries",
    "send_with_retry",
]
