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
from typing import Any, Awaitable, Callable, Dict, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_ROWS = 500
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
    content       TEXT NOT NULL,
    state         TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    owner_token   TEXT NOT NULL,
    last_error    TEXT
);
"""


@dataclass(frozen=True)
class SendResult:
    """The platform-neutral part of Hermes' outbound send result."""

    success: bool
    error: str = ""
    retryable: bool = False
    retry_after: Optional[float] = None

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
            connection.commit()
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
            with connection:
                yield connection
        finally:
            connection.close()

    def record(
        self,
        *,
        obligation_id: str,
        session_key: str,
        platform: str,
        chat_id: str,
        thread_id: str,
        content: str,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO delivery_obligations"
                " (obligation_id, session_key, platform, chat_id, thread_id,"
                " content, state, attempts, created_at, updated_at,"
                " owner_token, last_error)"
                " VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, NULL)",
                (
                    obligation_id,
                    session_key,
                    platform,
                    chat_id,
                    thread_id or None,
                    content,
                    now,
                    now,
                    self._owner_token,
                ),
            )
            self._prune(connection, now)

    def mark_attempting(self, obligation_id: str) -> None:
        self._update(obligation_id, "attempting")

    def mark_delivered(self, obligation_id: str) -> None:
        self._update(obligation_id, "delivered")

    def mark_failed(self, obligation_id: str, error: str = "") -> None:
        self._update(obligation_id, "failed", error)

    def _update(self, obligation_id: str, state: str, error: str = "") -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE delivery_obligations"
                " SET state = ?, updated_at = ?, last_error = ?"
                " WHERE obligation_id = ?",
                (state, time.time(), error[:500] or None, obligation_id),
            )

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
                " thread_id, content, state, attempts, created_at, owner_token"
                " FROM delivery_obligations"
                " WHERE state IN ('pending', 'attempting', 'failed')"
            ).fetchall()
            for row in rows:
                if row["owner_token"] == self._owner_token:
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
                    " SET owner_token = ?, attempts = attempts + 1,"
                    " updated_at = ?"
                    " WHERE obligation_id = ? AND owner_token = ?",
                    (
                        self._owner_token,
                        current_time,
                        row["obligation_id"],
                        row["owner_token"],
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
                            "content": row["content"],
                            "needs_marker": row["state"] != "pending",
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


async def send_with_retry(
    send: Callable[[], Awaitable[Any]],
    *,
    max_retries: int = 2,
    base_delay: float = 2.0,
) -> SendResult:
    """Retry only failures the channel classified as safe and transient."""

    async def attempt() -> SendResult:
        try:
            return as_send_result(await send())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Final-response send raised: %s", exc)
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
        logger.warning(
            "Final-response send failed; retrying in %.1fs (%d/%d): %s",
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
        )
        await asyncio.to_thread(store.mark_attempting, obligation_id)
        recorded = True
    except Exception:
        logger.exception("Could not record final-response delivery obligation")

    result = await send_with_retry(send)
    if recorded:
        try:
            if result.success:
                await asyncio.to_thread(store.mark_delivered, obligation_id)
            else:
                await asyncio.to_thread(
                    store.mark_failed,
                    obligation_id,
                    result.error or "send failed",
                )
        except Exception:
            logger.exception("Could not update final-response delivery obligation")
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
        try:
            if row["platform"] == "telegram":
                value = await channel.send(
                    row["chat_id"],
                    content,
                    thread_id=row["thread_id"],
                )
            else:
                value = await channel.send(row["chat_id"], content)
            result = as_send_result(value)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = SendResult(False, str(exc))
        try:
            if result.success:
                await asyncio.to_thread(
                    store.mark_delivered, row["obligation_id"]
                )
                delivered += 1
            else:
                await asyncio.to_thread(
                    store.mark_failed,
                    row["obligation_id"],
                    result.error or "send failed",
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


__all__ = [
    "DeliveryStore",
    "RECOVERED_MARKER",
    "SendResult",
    "as_send_result",
    "claim_deliveries",
    "compute_obligation_id",
    "deliver_final",
    "redeliver_claimed_deliveries",
    "recover_deliveries",
    "send_with_retry",
]
