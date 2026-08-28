"""Telegram channel adapted from Hermes' proven Telegram adapter.

The Bot API transport stays deliberately small: authorization happens before
downloads, bursts become one agent turn, topics keep their routing lane, and
outbound Markdown/media use Telegram-native delivery.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
)
from urllib.parse import urlparse

import httpx

from .. import media
from ..commands import CommandInvocation, parse_command
from ..delivery import (
    DeliveryUnitLedger,
    SendResult,
    as_send_result,
    delivery_fingerprint,
    file_delivery_fingerprint,
)
from ..redact import identity_pseudonym, redact_channel_identities
from ..settings import ConfigError, Settings
from .dedup import MessageDeduplicator
from .telegram_formatting import (
    split_telegram_message,
    strip_telegram_markdown,
    to_telegram,
)

logger = logging.getLogger(__name__)

try:  # Optional until the Telegram channel is enabled.
    from telegram import LinkPreviewOptions, Update
    from telegram.constants import ChatAction, ParseMode
    from telegram.error import (
        BadRequest,
        Conflict,
        Forbidden,
        InvalidToken,
        NetworkError,
        RetryAfter,
        TimedOut,
    )
    from telegram.ext import Application, MessageHandler, filters
    from telegram.request import HTTPXRequest

    TELEGRAM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised through preflight
    LinkPreviewOptions = Update = ChatAction = ParseMode = Any
    BadRequest = Conflict = Forbidden = InvalidToken = RuntimeError
    NetworkError = RetryAfter = TimedOut = RuntimeError
    Application = MessageHandler = HTTPXRequest = Any
    filters = None
    TELEGRAM_AVAILABLE = False


MAX_MESSAGE_LENGTH = 4096
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
SPLIT_THRESHOLD = 4000
QUOTE_SNIPPET_LIMIT = 500
TYPING_REFRESH_SECONDS = 5.0
SHUTDOWN_STEP_SECONDS = 10.0
STARTUP_FAILURE_DRAIN_SECONDS = 30.0
MEDIA_REGISTRATION_GRACE_SECONDS = 0.01
MEDIA_DOWNLOAD_GRACE_SECONDS = 1.0
POLLING_STARTUP_PROGRESS_SECONDS = 60.0
POLLING_WATCHDOG_INTERVAL_SECONDS = 30.0
POLLING_STALL_SECONDS = 150.0
INBOUND_SPOOL_MAX_ROWS = 10_000
INBOUND_SPOOL_MAX_BYTES = 64 * 1024 * 1024
INBOUND_SPOOL_MAX_UPDATE_BYTES = 1024 * 1024
INBOUND_SPOOL_RETENTION_SECONDS = 7 * 24 * 60 * 60

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})
_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"})
_VOICE_SUFFIXES = frozenset({".ogg", ".opus"})
_AUDIO_SUFFIXES = frozenset({".mp3", ".m4a", ".aac"})
_FOREIGN_BOT_HANDLE_RE = re.compile(r"[a-z0-9_]{2,29}bot", re.IGNORECASE)
_BOT_TOKEN_RE = re.compile(r"[1-9][0-9]*:[A-Za-z0-9_-]{30,}")


class ChannelError(RuntimeError):
    """The Telegram channel cannot start or continue running."""


def _split_env(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


def normalize_telegram_chat_id(value: Any) -> int | str:
    """Return Telegram's accepted numeric-id or @username representation."""
    written = str(value or "").strip()
    try:
        return int(written)
    except (TypeError, ValueError):
        return written


def normalize_telegram_bot_token(value: str) -> str:
    """Validate the BotFather token shape without exposing the secret."""
    token = str(value or "").strip()
    if _BOT_TOKEN_RE.fullmatch(token) is None:
        raise ValueError("Telegram bot token format is invalid")
    return token


def normalize_telegram_allowed_users(value: str) -> tuple[str, ...]:
    """Return a deduplicated, explicit Telegram user allowlist."""
    users: list[str] = []
    seen: set[str] = set()
    for part in str(value or "").replace(";", ",").split(","):
        user_id = part.strip()
        if not user_id:
            continue
        if user_id == "*":
            raise ValueError(
                "TELEGRAM_ALLOWED_USERS must name explicit users; '*' is not allowed"
            )
        if re.fullmatch(r"[1-9][0-9]*", user_id) is None:
            raise ValueError(
                "TELEGRAM_ALLOWED_USERS must contain numeric Telegram user IDs only"
            )
        if user_id not in seen:
            users.append(user_id)
            seen.add(user_id)
    if not users:
        raise ValueError("At least one allowed Telegram user ID is required")
    return tuple(users)


def normalize_telegram_home_chat_id(value: str) -> str:
    """Validate a Telegram user, group, or supergroup destination."""
    chat_id = str(value or "").strip()
    if re.fullmatch(r"-?[1-9][0-9]*", chat_id) is None:
        raise ValueError("Telegram home chat must be a non-zero numeric chat ID")
    return chat_id


def normalize_telegram_topic_id(value: str) -> str:
    """Validate an optional Telegram forum topic ID."""
    topic_id = str(value or "").strip()
    if topic_id and re.fullmatch(r"[1-9][0-9]*", topic_id) is None:
        raise ValueError("Telegram topic ID must be a positive number")
    return topic_id


def _nonnegative_number(settings: Settings, name: str, default: float) -> float:
    value = settings.number(name, default)
    if not math.isfinite(value) or value < 0:
        raise ConfigError(f"{name} must be at least 0, not {value!r}")
    return value


def validate_settings(settings: Settings) -> None:
    """Validate Telegram settings without importing the optional transport."""
    if settings.get("telegram.bot_token") is not None:
        raise ConfigError(
            "telegram.bot_token is a secret; set TELEGRAM_BOT_TOKEN in "
            "~/.pilotage-agent/.env instead"
        )
    if settings.get("telegram.allowed_users") is not None:
        raise ConfigError(
            "telegram.allowed_users contains sensitive identities; set "
            "TELEGRAM_ALLOWED_USERS in ~/.pilotage-agent/.env instead"
        )
    raw_allowed_users = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
    allowed_users: tuple[str, ...] = ()
    if raw_allowed_users:
        try:
            allowed_users = normalize_telegram_allowed_users(raw_allowed_users)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    enabled = settings.flag("telegram.enabled", False)
    if enabled and not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
        raise ConfigError(
            "telegram.enabled is true but TELEGRAM_BOT_TOKEN is not configured"
        )
    if enabled and not allowed_users:
        raise ConfigError(
            "telegram.enabled is true but TELEGRAM_ALLOWED_USERS is not configured"
        )
    home_chat_id = os.environ.get("TELEGRAM_HOME_CHANNEL", "").strip()
    if home_chat_id:
        try:
            normalize_telegram_home_chat_id(home_chat_id)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    topic_id = os.environ.get("TELEGRAM_HOME_CHANNEL_THREAD_ID", "").strip()
    if topic_id:
        try:
            normalize_telegram_topic_id(topic_id)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if not home_chat_id.startswith("-"):
            raise ConfigError(
                "TELEGRAM_HOME_CHANNEL_THREAD_ID requires a negative Telegram "
                "group or supergroup home chat ID"
            )
    webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL", "").strip()
    if enabled and webhook_url:
        parsed = urlparse(webhook_url)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise ConfigError(
                "TELEGRAM_WEBHOOK_URL must be a public https:// URL"
            )
        secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", secret):
            raise ConfigError(
                "TELEGRAM_WEBHOOK_SECRET is required with TELEGRAM_WEBHOOK_URL "
                "and must contain 1-256 letters, digits, underscores, or hyphens"
            )
        try:
            webhook_port = int(os.environ.get("TELEGRAM_WEBHOOK_PORT", "8443"))
        except ValueError:
            webhook_port = 0
        if not 1 <= webhook_port <= 65535:
            raise ConfigError(
                "TELEGRAM_WEBHOOK_PORT must be between 1 and 65535"
            )

    for retired_group_setting in (
        "telegram.group_policy",
        "telegram.group_allow_from",
    ):
        if settings.get(retired_group_setting) is not None:
            raise ConfigError(
                f"{retired_group_setting} is no longer supported; "
                "TELEGRAM_ALLOWED_USERS now authorizes each person in both "
                "DMs and groups"
            )
    settings.flag("telegram.require_mention", True)
    settings.flag("telegram.disable_link_previews", False)

    reply_mode = settings.text("telegram.reply_to_mode", "first").lower()
    if reply_mode not in {"off", "first", "all"}:
        raise ConfigError(
            "telegram.reply_to_mode must be 'off', 'first', or 'all', "
            f"not {reply_mode!r}"
        )
    _nonnegative_number(settings, "telegram.batch_delay", 0.3)
    _nonnegative_number(settings, "telegram.batch_split_delay", 1.0)
    _nonnegative_number(settings, "telegram.media_batch_delay", 0.8)
    hard_cap = settings.number("telegram.batch_hard_cap", 20.0)
    if not math.isfinite(hard_cap) or hard_cap <= 0:
        raise ConfigError(
            f"telegram.batch_hard_cap must be greater than 0, not {hard_cap!r}"
        )


@dataclass
class InboundMessage:
    chat_id: str
    session_id: str
    user_id: str
    user_name: str
    text: str
    is_group: bool
    thread_id: str = ""
    message_ids: List[str] = field(default_factory=list)
    claim_ids: List[str] = field(default_factory=list)
    attachments: List[media.Attachment] = field(default_factory=list)


@dataclass
class _MediaDownloadFence:
    count: int = 0
    generation: int = 0
    spent_generation: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)


Handler = Callable[[InboundMessage], Awaitable[None]]
CommandHandler = Callable[
    [str, str, str, str, CommandInvocation, str], Awaitable[None]
]


def _chat_type(message: Any) -> str:
    chat = getattr(message, "chat", None)
    return str(getattr(chat, "type", "")).split(".")[-1].lower()


def _is_group(message: Any) -> bool:
    return _chat_type(message) in {"group", "supergroup"}


def _effective_thread_id(message: Any) -> str:
    """Keep real topic IDs; ignore ordinary reply-UI anchor IDs. (Hermes)"""
    chat = getattr(message, "chat", None)
    chat_type = _chat_type(message)
    raw = getattr(message, "message_thread_id", None)
    topic_message = bool(getattr(message, "is_topic_message", False))
    forum = (
        chat_type in {"group", "supergroup"}
        and bool(getattr(chat, "is_forum", False))
    )
    if raw is not None:
        if forum or topic_message:
            return str(raw)
        return ""
    return "1" if forum else ""


def _session_id(chat_id: str, user_id: str, is_group: bool, thread_id: str) -> str:
    if not is_group:
        suffix = f":{thread_id}" if thread_id else ""
        return f"telegram:dm:{chat_id}{suffix}"
    if thread_id:
        return f"telegram:group:{chat_id}:{thread_id}:{user_id}"
    return f"telegram:group:{chat_id}:{user_id}"


def _telegram_entity_text(source: str, offset: int, length: int) -> str:
    """Extract a Telegram entity using its UTF-16 code-unit offsets."""
    if offset < 0 or length <= 0:
        return ""
    try:
        encoded = source.encode("utf-16-le")
        return encoded[offset * 2 : (offset + length) * 2].decode("utf-16-le")
    except UnicodeDecodeError:
        return ""


def _expand_telegram_text_links(message: Any) -> str:
    """Expose Telegram URLs stored outside the visible text/caption."""

    text = str(getattr(message, "text", None) or "")
    entities = getattr(message, "entities", None) or []
    if not text:
        text = str(getattr(message, "caption", None) or "")
        entities = getattr(message, "caption_entities", None) or []
    if not text or not entities:
        return text

    encoded = text.encode("utf-16-le")
    links: list[tuple[int, bytes]] = []
    for entity in entities:
        kind = str(getattr(entity, "type", "")).split(".")[-1].lower()
        raw_url = getattr(entity, "url", None)
        url = raw_url.strip() if isinstance(raw_url, str) else ""
        if kind != "text_link" or not url:
            continue
        try:
            offset = int(getattr(entity, "offset", -1))
            length = int(getattr(entity, "length", 0))
        except (TypeError, ValueError):
            continue
        start = offset * 2
        end = (offset + length) * 2
        if start < 0 or end <= start or end > len(encoded):
            continue
        try:
            encoded[start:end].decode("utf-16-le")
        except UnicodeDecodeError:
            continue
        inline = f" ({url})".encode("utf-16-le")
        links.append((end, inline))

    expanded = encoded
    for end, inline in sorted(links, reverse=True):
        if expanded[end : end + len(inline)] == inline:
            continue
        expanded = expanded[:end] + inline + expanded[end:]
    return expanded.decode("utf-16-le")


def _safe_filename(value: str, fallback: str) -> str:
    name = Path(str(value or "")).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return cleaned[:180] or fallback


def _redact_error(exc: BaseException, token: str) -> str:
    text = str(exc)
    if token:
        text = text.replace(token, "<redacted Telegram token>")
    text = re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot<redacted>", text)
    return redact_channel_identities(text)


def _send_failure(exc: BaseException, token: str) -> SendResult:
    written = _redact_error(exc, token)
    if isinstance(exc, RetryAfter):
        value = getattr(exc, "retry_after", None)
        total_seconds = getattr(value, "total_seconds", None)
        if callable(total_seconds):
            value = total_seconds()
        try:
            delay = float(value)
        except (TypeError, ValueError):
            delay = None
        if delay is not None and (not math.isfinite(delay) or delay < 0):
            delay = None
        return SendResult(False, written, retryable=True, retry_after=delay)
    if isinstance(exc, TimedOut):
        # PTB uses TimedOut for connect, pool, read, and write timeouts. Only
        # failures before a connection/request exists prove Telegram did not
        # accept the send.
        retryable = isinstance(
            exc.__cause__, (httpx.ConnectTimeout, httpx.PoolTimeout)
        )
        return SendResult(False, written, retryable=retryable)
    if isinstance(exc, NetworkError):
        # PTB collapses every other httpx.HTTPError into NetworkError. A
        # ConnectError is pre-acceptance; read/write/protocol errors are not.
        return SendResult(
            False,
            written,
            retryable=isinstance(exc.__cause__, httpx.ConnectError),
        )
    return SendResult(False, written)


class _PollingProgressRequest(HTTPXRequest):
    """Record successful getUpdates round-trips without PTB internals."""

    def __init__(self, on_progress: Callable[[], None], **kwargs: Any):
        super().__init__(**kwargs)
        self._on_progress = on_progress

    async def do_request(self, *args: Any, **kwargs: Any) -> tuple[int, bytes]:
        code, payload = await super().do_request(*args, **kwargs)
        if 200 <= code < 300:
            try:
                response = self.parse_json_payload(payload)
            except Exception:
                response = None
            if isinstance(response, dict) and response.get("ok") is True:
                self._on_progress()
        return code, payload


class _InboundSpoolError(RuntimeError):
    """Telegram input could not be accepted without losing its identity."""


def _update_message(update: Any) -> Any:
    return getattr(update, "effective_message", None) or getattr(
        update, "message", None
    )


def _durable_update_required(update: Any) -> bool:
    """Whether Pilotage has a handler that can consume this update."""

    message = _update_message(update)
    if message is None:
        return False
    if getattr(message, "text", None):
        return True
    return any(
        getattr(message, name, None)
        for name in (
            "location",
            "venue",
            "photo",
            "video",
            "audio",
            "voice",
            "document",
            "sticker",
        )
    )


def _update_sender_is_allowed(update: Any, allowed_users: Collection[str]) -> bool:
    message = _update_message(update)
    user = getattr(message, "from_user", None) if message is not None else None
    user_id = str(getattr(user, "id", "") or "").strip()
    return bool(user_id and user_id in allowed_users)


class _TelegramInboundStore:
    """Profile-local write-ahead spool for Telegram updates."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS telegram_updates (
        claim_id      TEXT PRIMARY KEY,
        namespace     TEXT NOT NULL,
        update_id     INTEGER NOT NULL,
        payload       TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL,
        state         TEXT NOT NULL CHECK (state IN ('pending', 'completed')),
        created_at    REAL NOT NULL,
        updated_at    REAL NOT NULL,
        UNIQUE (namespace, update_id)
    );
    """

    _MIGRATION_SCHEMA = """
    CREATE TABLE telegram_updates_v2 (
        claim_id      TEXT PRIMARY KEY,
        namespace     TEXT NOT NULL,
        update_id     INTEGER NOT NULL,
        payload       TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL,
        state         TEXT NOT NULL CHECK (state IN ('pending', 'completed')),
        created_at    REAL NOT NULL,
        updated_at    REAL NOT NULL,
        UNIQUE (namespace, update_id)
    )
    """

    def __init__(self, path: Path, token: str):
        self.path = Path(path)
        bot_id = str(token).partition(":")[0]
        identity = (
            f"bot:{bot_id}"
            if re.fullmatch(r"[1-9][0-9]*", bot_id) is not None
            else "unconfigured:"
            + hashlib.sha256(str(token).encode("utf-8", "replace")).hexdigest()
        )
        self._namespace = hashlib.sha256(
            f"telegram-bot-v1\0{identity}".encode("ascii")
        ).hexdigest()
        self._lock = threading.Lock()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(self._SCHEMA)
            self._migrate_namespace_uniqueness(connection)
            connection.commit()
            with contextlib.suppress(OSError):
                os.chmod(self.path, 0o600)
            with connection:
                yield connection
        finally:
            connection.close()

    @classmethod
    def _migrate_namespace_uniqueness(
        cls, connection: sqlite3.Connection
    ) -> None:
        """Replace the legacy global update-id constraint transactionally."""

        has_composite = False
        has_global_update_id = False
        for index in connection.execute(
            "PRAGMA index_list('telegram_updates')"
        ).fetchall():
            if not bool(index["unique"]):
                continue
            name = str(index["name"]).replace('"', '""')
            columns = [
                str(row["name"])
                for row in connection.execute(
                    f'PRAGMA index_info("{name}")'
                ).fetchall()
            ]
            if (
                columns == ["namespace", "update_id"]
                and not bool(index["partial"])
            ):
                has_composite = True
            elif columns == ["update_id"]:
                has_global_update_id = True
        if has_composite and not has_global_update_id:
            return

        with connection:
            connection.execute("DROP TABLE IF EXISTS telegram_updates_v2")
            connection.execute(cls._MIGRATION_SCHEMA)
            connection.execute(
                "INSERT INTO telegram_updates_v2"
                " (claim_id, namespace, update_id, payload, payload_sha256,"
                " state, created_at, updated_at)"
                " SELECT claim_id, namespace, update_id, payload, payload_sha256,"
                " state, created_at, updated_at FROM telegram_updates"
                " ORDER BY rowid ASC"
            )
            connection.execute("DROP TABLE telegram_updates")
            connection.execute(
                "ALTER TABLE telegram_updates_v2 RENAME TO telegram_updates"
            )

    def claim_id(self, update_id: Any) -> str:
        if isinstance(update_id, bool):
            raise _InboundSpoolError("Telegram update identity is invalid")
        try:
            written = int(update_id)
        except (TypeError, ValueError) as exc:
            raise _InboundSpoolError("Telegram update identity is invalid") from exc
        if written < 0:
            raise _InboundSpoolError("Telegram update identity is invalid")
        return self._claim_id_for_namespace(self._namespace, written)

    @staticmethod
    def _claim_id_for_namespace(namespace: str, update_id: int) -> str:
        return hashlib.sha256(
            f"telegram-update-v1\0{namespace}\0{update_id}".encode("ascii")
        ).hexdigest()

    @staticmethod
    def _payload(update: Any) -> str:
        to_dict = getattr(update, "to_dict", None)
        if not callable(to_dict):
            raise _InboundSpoolError("Telegram update cannot be serialized safely")
        value = to_dict()
        if not isinstance(value, dict):
            raise _InboundSpoolError("Telegram update payload is malformed")
        try:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise _InboundSpoolError(
                "Telegram update cannot be serialized safely"
            ) from exc
        if len(payload.encode("utf-8")) > INBOUND_SPOOL_MAX_UPDATE_BYTES:
            raise _InboundSpoolError("Telegram update payload exceeds the durable limit")
        return payload

    def _validate_rows(
        self,
        rows: Sequence[sqlite3.Row],
        *,
        current_namespace_only: bool = False,
    ) -> None:
        for row in rows:
            claim_id = str(row["claim_id"])
            namespace = str(row["namespace"])
            payload = str(row["payload"])
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            try:
                decoded = json.loads(payload)
                update_id = int(row["update_id"])
                created_at = float(row["created_at"])
                updated_at = float(row["updated_at"])
            except (TypeError, ValueError) as exc:
                raise _InboundSpoolError(
                    "Telegram inbound spool contains malformed durable data"
                ) from exc
            if (
                re.fullmatch(r"[a-f0-9]{64}", claim_id) is None
                or re.fullmatch(r"[a-f0-9]{64}", namespace) is None
                or (
                    current_namespace_only
                    and namespace != self._namespace
                )
                or self._claim_id_for_namespace(
                    namespace, update_id
                )
                != claim_id
                or str(row["payload_sha256"]) != digest
                or str(row["state"]) not in {"pending", "completed"}
                or not isinstance(decoded, dict)
                or update_id < 0
                or not math.isfinite(created_at)
                or not math.isfinite(updated_at)
            ):
                raise _InboundSpoolError(
                    "Telegram inbound spool contains corrupt durable data"
                )

    def _prune(self, connection: sqlite3.Connection, now: float) -> None:
        columns = (
            "claim_id, namespace, update_id, payload, payload_sha256, state,"
            " created_at, updated_at"
        )
        expired = connection.execute(
            f"SELECT {columns} FROM telegram_updates"
            " WHERE state = 'completed' AND updated_at < ?",
            (now - INBOUND_SPOOL_RETENTION_SECONDS,),
        ).fetchall()
        self._validate_rows(expired)
        if expired:
            claims = [str(row["claim_id"]) for row in expired]
            placeholders = ",".join("?" for _ in claims)
            connection.execute(
                f"DELETE FROM telegram_updates WHERE claim_id IN ({placeholders})",
                claims,
            )
        remaining, payload_bytes = connection.execute(
            "SELECT COUNT(*),"
            " COALESCE(SUM(LENGTH(CAST(payload AS BLOB))), 0)"
            " FROM telegram_updates WHERE namespace = ?",
            (self._namespace,),
        ).fetchone()
        if (
            int(remaining) > INBOUND_SPOOL_MAX_ROWS
            or int(payload_bytes) > INBOUND_SPOOL_MAX_BYTES
        ):
            raise _InboundSpoolError(
                "Telegram inbound spool capacity is exhausted by retained updates"
            )

    def record(self, update: Any) -> tuple[str, str]:
        update_id = getattr(update, "update_id", None)
        claim_id = self.claim_id(update_id)
        payload = self._payload(update)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO telegram_updates"
                " (claim_id, namespace, update_id, payload, payload_sha256, state,"
                " created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                (
                    claim_id,
                    self._namespace,
                    int(update_id),
                    payload,
                    digest,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT claim_id, namespace, update_id, payload, payload_sha256, state"
                " FROM telegram_updates WHERE namespace = ? AND update_id = ?",
                (self._namespace, int(update_id)),
            ).fetchone()
            if row is None:
                raise _InboundSpoolError("Telegram update was not recorded")
            recorded = (
                str(row["claim_id"]),
                str(row["namespace"]),
                int(row["update_id"]),
                str(row["payload"]),
                str(row["payload_sha256"]),
            )
            expected = (
                claim_id,
                self._namespace,
                int(update_id),
                payload,
                digest,
            )
            state = str(row["state"])
            if recorded != expected or state not in {"pending", "completed"}:
                raise _InboundSpoolError(
                    "Telegram update identity collides with different durable data"
                )
            self._prune(connection, now)
            return claim_id, state

    def pending(self) -> list[tuple[str, str]]:
        now = time.time()
        with self._lock, self._connect() as connection:
            self._prune(connection, now)
            rows = connection.execute(
                "SELECT claim_id, namespace, update_id, payload, payload_sha256,"
                " state, created_at, updated_at"
                " FROM telegram_updates WHERE state = 'pending'"
                " ORDER BY rowid ASC"
            ).fetchall()
            self._validate_rows(rows)
            if any(str(row["namespace"]) != self._namespace for row in rows):
                raise _InboundSpoolError(
                    "Telegram inbound spool has pending updates for a different bot"
                )
            pending: list[tuple[str, str]] = []
            for row in rows:
                claim_id = str(row["claim_id"])
                payload = str(row["payload"])
                pending.append((claim_id, payload))
            return pending

    def complete(self, claim_ids: Sequence[str]) -> None:
        claims = list(dict.fromkeys(str(value) for value in claim_ids if value))
        if not claims:
            return
        if any(re.fullmatch(r"[a-f0-9]{64}", value) is None for value in claims):
            raise _InboundSpoolError("Telegram inbound claim identity is invalid")
        now = time.time()
        with self._lock, self._connect() as connection:
            placeholders = ",".join("?" for _ in claims)
            rows = connection.execute(
                "SELECT claim_id, namespace, update_id, payload, payload_sha256,"
                " state, created_at, updated_at FROM telegram_updates"
                f" WHERE claim_id IN ({placeholders})",
                claims,
            ).fetchall()
            self._validate_rows(rows, current_namespace_only=True)
            found = {str(row["claim_id"]) for row in rows}
            if found != set(claims) or any(
                str(row["namespace"]) != self._namespace
                or str(row["state"]) not in {"pending", "completed"}
                for row in rows
            ):
                raise _InboundSpoolError(
                    "Telegram inbound claim has no exact durable update"
                )
            connection.execute(
                "UPDATE telegram_updates SET state = 'completed', updated_at = ?"
                f" WHERE claim_id IN ({placeholders})",
                (now, *claims),
            )
            self._prune(connection, now)

    def discard_pending(self, claim_id: str) -> None:
        """Forget an update that current authorization explicitly rejects."""

        written = str(claim_id or "")
        if re.fullmatch(r"[a-f0-9]{64}", written) is None:
            raise _InboundSpoolError("Telegram inbound claim identity is invalid")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT claim_id, namespace, update_id, payload, payload_sha256,"
                " state, created_at, updated_at FROM telegram_updates"
                " WHERE claim_id = ?",
                (written,),
            ).fetchone()
            if row is None:
                raise _InboundSpoolError(
                    "Telegram rejected update has no exact durable record"
                )
            self._validate_rows([row], current_namespace_only=True)
            if str(row["state"]) == "completed":
                return
            removed = connection.execute(
                "DELETE FROM telegram_updates"
                " WHERE claim_id = ? AND state = 'pending'",
                (written,),
            ).rowcount
            if removed != 1:
                raise _InboundSpoolError(
                    "Telegram rejected update could not be discarded safely"
                )


class _DurableUpdateQueue(asyncio.Queue):
    """Commit Telegram work before PTB can advance its server offset."""

    def __init__(
        self,
        store: _TelegramInboundStore,
        on_failure: Callable[[str], None],
        allowed_users: Collection[str],
    ):
        super().__init__()
        self._store = store
        self._on_failure = on_failure
        self._allowed_users = frozenset(str(value) for value in allowed_users)
        self._enqueued_claims: set[str] = set()
        self._enqueue_lock = asyncio.Lock()

    async def put(self, item: Any) -> None:
        if not _durable_update_required(item):
            await super().put(item)
            return
        if not _update_sender_is_allowed(item, self._allowed_users):
            return
        try:
            claim_id, state = await asyncio.to_thread(self._store.record, item)
        except (OSError, sqlite3.Error, _InboundSpoolError) as exc:
            self._on_failure("The Telegram durable inbound spool failed.")
            raise ChannelError("Telegram could not durably accept an update") from exc
        if state == "completed":
            return
        async with self._enqueue_lock:
            if claim_id in self._enqueued_claims:
                return
            self._enqueued_claims.add(claim_id)
            try:
                await super().put(item)
            except BaseException:
                self._enqueued_claims.discard(claim_id)
                raise

    def forget_completed(self, claim_ids: Sequence[str]) -> None:
        self._enqueued_claims.difference_update(claim_ids)

    async def replay_pending(self, bot: Any) -> None:
        try:
            pending = await asyncio.to_thread(self._store.pending)
            for claim_id, payload in pending:
                if claim_id in self._enqueued_claims:
                    continue
                update = Update.de_json(json.loads(payload), bot)
                if (
                    not _durable_update_required(update)
                    or self._store.claim_id(update.update_id) != claim_id
                ):
                    raise _InboundSpoolError(
                        "Telegram pending update cannot be reconstructed safely"
                    )
                if not _update_sender_is_allowed(update, self._allowed_users):
                    await asyncio.to_thread(self._store.discard_pending, claim_id)
                    continue
                self._enqueued_claims.add(claim_id)
                await super().put(update)
        except (OSError, sqlite3.Error, TypeError, ValueError, _InboundSpoolError) as exc:
            self._on_failure("The Telegram durable inbound spool failed.")
            raise ChannelError("Telegram pending updates are not recoverable") from exc


class TelegramChannel:
    """One Telegram bot connected through a webhook or long polling."""

    def __init__(self, config: Any, handler: Handler, on_command: CommandHandler):
        self._config = config
        self._handler = handler
        self._on_command = on_command
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self._webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL", "").strip()
        self._webhook_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
        self._webhook_host = os.environ.get("TELEGRAM_WEBHOOK_HOST", "").strip()
        self._webhook_port = (
            int(os.environ.get("TELEGRAM_WEBHOOK_PORT", "8443"))
            if self._webhook_url
            else 8443
        )
        self._webhook_mode = False
        self._allowed_users = frozenset(_split_env("TELEGRAM_ALLOWED_USERS"))
        settings = config.settings
        self._require_mention = settings.flag(
            "telegram.require_mention", True
        )
        self._reply_to_mode = settings.text(
            "telegram.reply_to_mode", "first"
        ).lower()
        self._disable_link_previews = settings.flag(
            "telegram.disable_link_previews", False
        )
        self._text_batch_delay = settings.number(
            "telegram.batch_delay", 0.3
        )
        self._text_batch_split_delay = max(
            self._text_batch_delay,
            settings.number("telegram.batch_split_delay", 1.0),
        )
        self._media_batch_delay = settings.number(
            "telegram.media_batch_delay", 0.8
        )
        self._batch_hard_cap = settings.number(
            "telegram.batch_hard_cap", 20.0
        )

        self._app: Any = None
        self._bot: Any = None
        self._bot_username = ""
        self._running = False
        self._intake_stopped = False
        self._intake_started = False
        self._drop_delayed_deliveries = False
        self._startup_hold_closed = False
        self._startup_approvals_enabled = False
        self.stopped = asyncio.Event()
        self.failure: Optional[str] = None

        self._seen = MessageDeduplicator()
        self._completed_claims = MessageDeduplicator(
            max_size=INBOUND_SPOOL_MAX_ROWS * 2,
            ttl_seconds=INBOUND_SPOOL_RETENTION_SECONDS,
        )
        self._completed_claims_lock = threading.Lock()
        self._reported_blocked: set[str] = set()
        self._pending: Dict[str, InboundMessage] = {}
        self._pending_started: Dict[str, float] = {}
        self._pending_tasks: Dict[str, asyncio.Task] = {}
        self._queued: Dict[str, InboundMessage] = {}
        self._turn_tasks: Dict[str, asyncio.Task] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._intake_tasks: set[asyncio.Task] = set()
        self._startup_updates: List[tuple[str, Any, Any]] = []
        self._startup_held_claims: set[str] = set()
        self._held_inbound: Dict[str, InboundMessage] = {}
        self._inbound_store = _TelegramInboundStore(
            config.state_dir / "telegram-inbound.db",
            self._token,
        )
        self._update_queue = _DurableUpdateQueue(
            self._inbound_store,
            self._fail,
            self._allowed_users,
        )
        self._polling_progress: Optional[asyncio.Event] = None
        self._polling_last_progress = 0.0
        self._polling_watchdog_task: Optional[asyncio.Task] = None
        self._media_downloads: Dict[str, _MediaDownloadFence] = {}

    # -- lifetime -------------------------------------------------------

    def hold_inbound(self) -> None:
        self._startup_hold_closed = True
        self._startup_approvals_enabled = False

    async def enable_startup_approvals(self) -> None:
        """Allow only approval control through the still-closed startup gate."""

        if not self._startup_hold_closed:
            return
        self._startup_approvals_enabled = True

    @property
    def startup_approval_available(self) -> bool:
        """Whether startup recovery can safely receive an approval command."""

        return self._intake_started and self._startup_approvals_enabled

    async def release_inbound(self) -> None:
        if not self._startup_hold_closed and not self._startup_updates:
            return
        self._startup_approvals_enabled = False
        try:
            # Keep the gate closed while draining. Fresh callbacks append to
            # the same FIFO, so they cannot overtake an older held update.
            while self._startup_updates:
                kind, update, context = self._startup_updates.pop(0)
                await self._replay_startup_update(kind, update, context)
        finally:
            self._startup_hold_closed = False
            self._startup_approvals_enabled = False
            self._startup_held_claims.clear()

    def _preflight(self) -> None:
        if not TELEGRAM_AVAILABLE:
            raise ChannelError(
                "Telegram is enabled but python-telegram-bot is not installed. "
                "Install the project dependencies and try again."
            )
        if not self._token:
            raise ChannelError("TELEGRAM_BOT_TOKEN is not configured.")
        if not self._allowed_users:
            logger.warning(
                "TELEGRAM_ALLOWED_USERS is empty - every Telegram message "
                "will be ignored."
            )

    def _build_application(self) -> Any:
        request_options = {
            "connection_pool_size": 512,
            "pool_timeout": 8.0,
            "connect_timeout": 10.0,
            "read_timeout": 20.0,
            "write_timeout": 20.0,
            "media_write_timeout": 60.0,
        }
        request = HTTPXRequest(**request_options)
        updates_request = _PollingProgressRequest(
            self._record_polling_progress, **request_options
        )
        app = (
            Application.builder()
            .token(self._token)
            .request(request)
            .get_updates_request(updates_request)
            .update_queue(self._update_queue)
            .build()
        )
        self._register_handlers(app)
        app.add_error_handler(self._handle_update_error)
        return app

    def _register_handlers(self, app: Any) -> None:
        assert filters is not None
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )
        app.add_handler(
            MessageHandler(filters.COMMAND, self._handle_command)
        )
        app.add_handler(
            MessageHandler(
                filters.LOCATION | getattr(filters, "VENUE", filters.LOCATION),
                self._handle_location,
            )
        )
        app.add_handler(
            MessageHandler(
                filters.PHOTO
                | filters.VIDEO
                | filters.AUDIO
                | filters.VOICE
                | filters.Document.ALL
                | filters.Sticker.ALL,
                self._handle_media,
            )
        )

    async def start(self) -> None:
        self._preflight()
        self._config.media_dir.joinpath("telegram").mkdir(
            parents=True, exist_ok=True
        )
        self.stopped.clear()
        self.failure = None
        self._intake_stopped = False
        self._drop_delayed_deliveries = False
        self._intake_started = False
        self._seen = MessageDeduplicator()
        # The SQLite spool, not shutdown RAM, is authoritative for identified
        # Telegram work when the same channel object is started again.
        self._held_inbound = {
            key: message
            for key, message in self._held_inbound.items()
            if not message.claim_ids
        }
        self._polling_progress = asyncio.Event()
        self._polling_last_progress = 0.0
        self._update_queue = _DurableUpdateQueue(
            self._inbound_store,
            self._fail,
            self._allowed_users,
        )
        app = self._build_application()
        self._app = app
        try:
            await asyncio.wait_for(app.initialize(), timeout=30.0)
            self._bot = app.bot
            self._bot_username = (
                str(getattr(self._bot, "username", "") or "")
                .lstrip("@")
                .lower()
            )
            # Replay locally committed work before accepting new server work.
            await self._update_queue.replay_pending(self._bot)
            await app.start()
            self._running = True
            await self._start_update_intake()
        except asyncio.CancelledError:
            cleanup = (
                self.abort_startup()
                if self._startup_hold_closed
                else self.stop(
                    drain_timeout_seconds=STARTUP_FAILURE_DRAIN_SECONDS
                )
            )
            await asyncio.shield(cleanup)
            raise
        except BaseException as exc:
            cleanup = (
                self.abort_startup()
                if self._startup_hold_closed
                else self.stop(
                    drain_timeout_seconds=STARTUP_FAILURE_DRAIN_SECONDS
                )
            )
            await asyncio.shield(cleanup)
            if isinstance(exc, InvalidToken):
                raise ChannelError("Telegram rejected TELEGRAM_BOT_TOKEN.") from exc
            if isinstance(exc, ChannelError):
                raise
            raise ChannelError(
                "Telegram could not start: " + _redact_error(exc, self._token)
            ) from exc

        self._log_ready()

    async def _start_update_intake(self) -> None:
        if self._intake_started:
            return
        app = self._app
        if app is None or app.updater is None:
            raise ChannelError("Telegram update delivery is unavailable.")
        if self._webhook_url:
            webhook_path = urlparse(self._webhook_url).path or "/telegram"
            await asyncio.wait_for(
                app.updater.start_webhook(
                    listen=self._webhook_host,
                    port=self._webhook_port,
                    url_path=webhook_path,
                    webhook_url=self._webhook_url,
                    secret_token=self._webhook_secret,
                    allowed_updates=Update.ALL_TYPES,
                    # A message accepted by Telegram while Pilotage is stopped
                    # remains work owed to the user.
                    drop_pending_updates=False,
                ),
                timeout=30.0,
            )
            self._webhook_mode = True
        else:
            await app.bot.delete_webhook(drop_pending_updates=False)
            await asyncio.wait_for(
                app.updater.start_polling(
                    drop_pending_updates=False,
                    error_callback=self._polling_error_callback,
                ),
                timeout=30.0,
            )
            try:
                assert self._polling_progress is not None
                await asyncio.wait_for(
                    self._polling_progress.wait(),
                    timeout=POLLING_STARTUP_PROGRESS_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise ChannelError(
                    "Telegram polling made no successful getUpdates progress "
                    f"within {POLLING_STARTUP_PROGRESS_SECONDS:.0f}s."
                ) from exc
            self._webhook_mode = False
        self._intake_started = True
        if not self._webhook_mode:
            self._polling_watchdog_task = asyncio.create_task(
                self._polling_watchdog(),
                name="pilotage-telegram-polling-watchdog",
            )

    def _log_ready(self) -> None:
        self._release_held_inbound()
        logger.info(
            "Telegram channel ready as %s (%s)",
            identity_pseudonym(self._bot_username or "unknown", "tg-bot"),
            "webhook" if self._webhook_mode else "polling",
        )

    async def _bounded_step(self, awaitable: Any, label: str) -> None:
        try:
            await asyncio.wait_for(awaitable, timeout=SHUTDOWN_STEP_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Telegram %s failed during shutdown: %s",
                label,
                _redact_error(exc, self._token),
            )

    async def stop_intake(self, *, release_held_inbound: bool = True) -> None:
        """Stop receiving updates and dispatch every batch already accepted."""

        if self._intake_stopped:
            if release_held_inbound:
                await self.release_inbound()
                self._release_held_inbound()
            else:
                self._startup_updates.clear()
                self._held_inbound.clear()
            return
        self._intake_stopped = True
        # Hermes fences delayed delivery before the first teardown await. PTB
        # may already have advanced the update offset, so a late handler must
        # be held instead of scheduling work that teardown will never observe.
        self._drop_delayed_deliveries = True
        self._running = False
        watchdog = self._polling_watchdog_task
        self._polling_watchdog_task = None
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)
        app = self._app
        updater = getattr(app, "updater", None) if app is not None else None
        if updater is not None and getattr(updater, "running", False):
            await self._bounded_step(updater.stop(), "update receiver stop")

        if release_held_inbound:
            await self.release_inbound()
        else:
            # The durable spool remains pending. Drop only the in-memory PTB
            # callback so restart replay, not failed-startup teardown, owns it.
            self._startup_updates.clear()
        timers = set(self._pending_tasks.values())
        for key in list(self._pending):
            self._flush_pending_now(key)
        if release_held_inbound:
            self._release_held_inbound()
        else:
            self._held_inbound.clear()
        for task in timers:
            task.cancel()
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)
        self._pending_tasks.clear()
        if app is not None and getattr(app, "running", False):
            # PTB owns the update callbacks. Stop it before taking our task
            # snapshot so callbacks already accepted by PTB can reach the hold
            # queue and then join Pilotage's bounded drain.
            await self._bounded_step(app.stop(), "application stop")
        if release_held_inbound:
            self._release_held_inbound()
        else:
            self._held_inbound.clear()

    async def stop(
        self,
        *,
        drain_timeout_seconds: float = 0.0,
        release_held_inbound: bool = True,
    ) -> None:
        """Stop intake, then give already accepted work a bounded drain."""

        await self.stop_intake(release_held_inbound=release_held_inbound)

        loop = asyncio.get_running_loop()
        # Telegram may already have advanced its server offset when a channel
        # error is reported. Accepted work therefore keeps the caller's bounded
        # drain even on failure.
        timeout = max(0.0, float(drain_timeout_seconds))
        deadline = loop.time() + timeout
        current = asyncio.current_task()

        # A media callback can still be finishing a download when PTB teardown
        # begins. Let it reach the fenced hold queue before draining turns.
        intake = {
            task
            for task in self._intake_tasks
            if task is not current and not task.done()
        }
        pending_intake = intake
        if intake and timeout > 0:
            _, pending_intake = await asyncio.wait(
                intake, timeout=max(0.0, deadline - loop.time())
            )
        if pending_intake:
            logger.warning(
                "Telegram shutdown drain expired with %d in-progress update(s)",
                len(pending_intake),
            )
        for task in pending_intake:
            task.cancel()
        if intake:
            await asyncio.gather(*intake, return_exceptions=True)
        if release_held_inbound:
            self._release_held_inbound()
        else:
            self._held_inbound.clear()

        owned = set(self._pending_tasks.values())
        owned.update(self._turn_tasks.values())
        owned.update(self._background_tasks)
        owned.discard(current)
        live = {task for task in owned if not task.done()}
        pending = live
        if live and timeout > 0:
            _, pending = await asyncio.wait(
                live, timeout=max(0.0, deadline - loop.time())
            )
        if pending:
            logger.warning(
                "Telegram shutdown drain expired with %d accepted task(s)",
                len(pending),
            )
        for task in pending:
            task.cancel()
        if owned:
            await asyncio.gather(*owned, return_exceptions=True)

        self._pending.clear()
        self._pending_started.clear()
        self._pending_tasks.clear()
        self._queued.clear()
        self._turn_tasks.clear()
        self._background_tasks.clear()
        self._intake_tasks.clear()
        self._startup_updates.clear()
        self._startup_held_claims.clear()
        self._startup_hold_closed = False
        self._startup_approvals_enabled = False
        self._intake_started = False
        self._polling_progress = None
        self._polling_last_progress = 0.0
        for fence in self._media_downloads.values():
            fence.changed.set()
        self._media_downloads.clear()

        app = self._app
        self._app = None
        self._bot = None
        if app is None:
            self.stopped.set()
            return
        try:
            await self._bounded_step(app.shutdown(), "application shutdown")
        finally:
            if self._held_inbound:
                logger.warning(
                    "Telegram retained %d late inbound message(s) for restart",
                    len(self._held_inbound),
                )
            self.stopped.set()

    async def abort_startup(self) -> None:
        """Stop without dispatching or completing startup-spooled updates."""

        await self.stop(
            drain_timeout_seconds=0.0,
            release_held_inbound=False,
        )

    def _polling_error_callback(self, error: BaseException) -> None:
        written = _redact_error(error, self._token)
        if isinstance(error, Conflict) or "conflict" in written.lower():
            self._fail(
                "Telegram polling stopped because another process is using "
                "this bot token."
            )
            return
        if isinstance(error, (InvalidToken, Forbidden)):
            self._fail("Telegram polling lost authorization: " + written)
            return
        logger.warning("Telegram polling error: %s", written)

    def _record_polling_progress(self) -> None:
        self._polling_last_progress = time.monotonic()
        progress = self._polling_progress
        if progress is not None:
            progress.set()

    async def _polling_watchdog(self) -> None:
        """Let PTB recover briefly, then fail loudly if polling stays deaf."""

        try:
            while self._running and not self._webhook_mode:
                await asyncio.sleep(POLLING_WATCHDOG_INTERVAL_SECONDS)
                if not self._running or self._webhook_mode:
                    return
                app = self._app
                updater = getattr(app, "updater", None) if app is not None else None
                if updater is None or not getattr(updater, "running", False):
                    self._fail("Telegram polling stopped while Pilotage was running.")
                    return
                stalled_for = time.monotonic() - self._polling_last_progress
                if stalled_for > POLLING_STALL_SECONDS:
                    self._fail(
                        "Telegram polling made no successful getUpdates progress "
                        f"for {stalled_for:.0f}s."
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(
                "Telegram polling watchdog failed: "
                + _redact_error(exc, self._token)
            )

    async def _handle_update_error(self, update: object, context: Any) -> None:
        error = getattr(context, "error", RuntimeError("unknown update error"))
        logger.error(
            "Telegram update handling failed: %s",
            _redact_error(error, self._token),
        )
        if _durable_update_required(update):
            self._fail(
                "Telegram stopped after an accepted update failed; its durable "
                "input will replay after restart."
            )

    def _fail(self, message: str) -> None:
        if self.failure:
            return
        self.failure = message
        self._running = False
        logger.error("%s", message)
        self.stopped.set()

    # -- authorization and routing -------------------------------------

    def _is_user_allowed(self, user: Any) -> bool:
        user_id = str(getattr(user, "id", "") or "").strip()
        return bool(user_id and user_id in self._allowed_users)

    def _report_blocked(self, message: Any) -> None:
        user = getattr(message, "from_user", None)
        identity = str(getattr(user, "id", "") or "unknown")
        if identity in self._reported_blocked:
            return
        self._reported_blocked.add(identity)
        logger.warning(
            "Ignored a Telegram message from %s.",
            identity_pseudonym(identity, "tg"),
        )

    @staticmethod
    def _iter_text_sources(message: Any):
        yield (
            str(getattr(message, "text", "") or ""),
            getattr(message, "entities", None) or [],
        )
        yield (
            str(getattr(message, "caption", "") or ""),
            getattr(message, "caption_entities", None) or [],
        )

    def _extract_bot_mentions(self, message: Any) -> set[str]:
        mentioned: set[str] = set()
        own = self._bot_username

        def is_bot_handle(handle: str) -> bool:
            return bool(
                handle
                and (
                    handle == own
                    or _FOREIGN_BOT_HANDLE_RE.fullmatch(handle)
                )
            )

        for source, entities in self._iter_text_sources(message):
            for entity in entities:
                kind = str(getattr(entity, "type", "")).split(".")[-1].lower()
                if kind not in {"mention", "bot_command"}:
                    continue
                span = _telegram_entity_text(
                    source,
                    int(getattr(entity, "offset", -1)),
                    int(getattr(entity, "length", 0)),
                ).strip()
                if kind == "mention":
                    handle = span.lstrip("@").lower()
                elif "@" in span:
                    handle = span.rsplit("@", 1)[1].lower()
                else:
                    continue
                if is_bot_handle(handle):
                    mentioned.add(handle)

        for source, entities in self._iter_text_sources(message):
            if not source or entities:
                continue
            for match in re.finditer(
                r"(?i)(?<![A-Za-z0-9_/])@([A-Za-z0-9_]{2,31})\b",
                source,
            ):
                handle = match.group(1).lower()
                if is_bot_handle(handle):
                    mentioned.add(handle)
        return mentioned

    def _message_mentions_self(self, message: Any) -> bool:
        bot_id = getattr(self._bot, "id", None)
        expected = self._bot_username
        for source, entities in self._iter_text_sources(message):
            for entity in entities:
                kind = str(getattr(entity, "type", "")).split(".")[-1].lower()
                if kind == "text_mention":
                    user = getattr(entity, "user", None)
                    if user is not None and getattr(user, "id", None) == bot_id:
                        return True
                if kind not in {"mention", "bot_command"}:
                    continue
                span = _telegram_entity_text(
                    source,
                    int(getattr(entity, "offset", -1)),
                    int(getattr(entity, "length", 0)),
                ).strip()
                handle = ""
                if kind == "mention":
                    handle = span.lstrip("@").lower()
                elif "@" in span:
                    handle = span.rsplit("@", 1)[1].lower()
                if expected and handle == expected:
                    return True
        return bool(
            expected
            and expected in self._extract_bot_mentions(message)
        )

    def _is_reply_to_bot(self, message: Any) -> bool:
        reply = getattr(message, "reply_to_message", None)
        reply_user = getattr(reply, "from_user", None)
        return bool(
            reply_user
            and self._bot is not None
            and getattr(reply_user, "id", None) == getattr(self._bot, "id", None)
        )

    def _group_message_is_triggered(self, message: Any) -> bool:
        mentions = self._extract_bot_mentions(message)
        if mentions and self._bot_username not in mentions:
            return False
        if not self._require_mention:
            return True
        return self._message_mentions_self(message) or self._is_reply_to_bot(message)

    def _authorized(self, message: Any) -> bool:
        user = getattr(message, "from_user", None)
        chat = getattr(message, "chat", None)
        chat_id = str(getattr(chat, "id", "") or "")
        if user is None or not chat_id:
            self._report_blocked(message)
            return False
        if not self._is_user_allowed(user):
            self._report_blocked(message)
            return False
        if _is_group(message):
            if not self._group_message_is_triggered(message):
                return False
            return True
        return True

    def _clean_routing_mention(self, text: str) -> str:
        if not text or not self._bot_username:
            return text.strip()
        cleaned = re.sub(
            rf"(?i)(/[A-Za-z0-9_]+)@{re.escape(self._bot_username)}\b",
            r"\1",
            text,
        )
        cleaned = re.sub(
            rf"(?i)(?<![A-Za-z0-9_])@{re.escape(self._bot_username)}\b[,:\-]*\s*",
            "",
            cleaned,
        )
        return cleaned.strip() or text.strip()

    # -- inbound --------------------------------------------------------

    @asynccontextmanager
    async def _track_intake(self):
        task = asyncio.current_task()
        if task is not None:
            self._intake_tasks.add(task)
        try:
            yield
        finally:
            if task is not None:
                self._intake_tasks.discard(task)

    @staticmethod
    def _message_from_update(update: Any) -> Any:
        return _update_message(update)

    def _claim_ids_for_update(self, update: Any) -> List[str]:
        update_id = getattr(update, "update_id", None)
        if update_id is None:
            # Direct handler calls in tests have no PTB update identity. Every
            # production callback enters through the durable update queue.
            return []
        try:
            return [self._inbound_store.claim_id(update_id)]
        except _InboundSpoolError as exc:
            self._fail("Telegram delivered an update without a stable identity.")
            raise ChannelError("Telegram update identity is invalid") from exc

    def persist_completed_claims(self, claim_ids: Sequence[str]) -> None:
        """Fence recovered/model-complete input before its handler returns."""

        claims = list(claim_ids)
        self._inbound_store.complete(claims)
        self._cache_completed_claims(claims)

    def _cache_completed_claims(self, claim_ids: Sequence[str]) -> None:
        with self._completed_claims_lock:
            for claim_id in claim_ids:
                self._completed_claims.is_duplicate(claim_id)

    def _claims_are_completed(self, claim_ids: Sequence[str]) -> bool:
        if not claim_ids:
            return False
        with self._completed_claims_lock:
            completed = all(
                self._completed_claims.contains(claim_id) for claim_id in claim_ids
            )
        if completed:
            self._update_queue.forget_completed(claim_ids)
        return completed

    async def _complete_claims(self, claim_ids: Sequence[str]) -> None:
        claims = list(dict.fromkeys(value for value in claim_ids if value))
        if not claims:
            return
        if self._claims_are_completed(claims):
            return
        try:
            await asyncio.to_thread(self._inbound_store.complete, claims)
        except (OSError, sqlite3.Error, _InboundSpoolError) as exc:
            logger.exception("Persisting completed Telegram claims failed")
            self._fail("The Telegram durable inbound spool failed.")
            raise ChannelError("Telegram claims could not be completed") from exc
        self._cache_completed_claims(claims)
        self._update_queue.forget_completed(claims)

    async def _handle_text(
        self,
        update: Any,
        context: Any,
        *,
        _startup_replay: bool = False,
    ) -> None:
        async with self._track_intake():
            claim_ids = self._claim_ids_for_update(update)
            if self._claims_are_completed(claim_ids):
                return
            message = self._message_from_update(update)
            if message is None or not self._authorized(message):
                await self._complete_claims(claim_ids)
                return
            if not _startup_replay and self._hold_startup_update(
                "text", update, context, claim_ids
            ):
                return
            await self._accept_message(message, [], claim_ids=claim_ids)

    async def _handle_command(
        self,
        update: Any,
        context: Any,
        *,
        _startup_replay: bool = False,
    ) -> None:
        async with self._track_intake():
            claim_ids = self._claim_ids_for_update(update)
            if self._claims_are_completed(claim_ids):
                return
            message = self._message_from_update(update)
            if message is None or not self._authorized(message):
                await self._complete_claims(claim_ids)
                return
            if not _startup_replay and self._hold_startup_update(
                "command", update, context, claim_ids
            ):
                return
            await self._accept_message(message, [], claim_ids=claim_ids)

    async def _handle_location(
        self,
        update: Any,
        context: Any,
        *,
        _startup_replay: bool = False,
    ) -> None:
        async with self._track_intake():
            claim_ids = self._claim_ids_for_update(update)
            if self._claims_are_completed(claim_ids):
                return
            message = self._message_from_update(update)
            if message is None or not self._authorized(message):
                await self._complete_claims(claim_ids)
                return
            if not _startup_replay and self._hold_startup_update(
                "location", update, context, claim_ids
            ):
                return
            venue = getattr(message, "venue", None)
            location = (
                getattr(venue, "location", None)
                if venue is not None
                else getattr(message, "location", None)
            )
            if location is None:
                await self._complete_claims(claim_ids)
                return
            latitude = getattr(location, "latitude", None)
            longitude = getattr(location, "longitude", None)
            if latitude is None or longitude is None:
                await self._complete_claims(claim_ids)
                return
            parts = ["[The sender shared a location pin.]"]
            if venue is not None:
                title = str(getattr(venue, "title", "") or "").strip()
                address = str(getattr(venue, "address", "") or "").strip()
                if title:
                    parts.append(f"Venue: {title}")
                if address:
                    parts.append(f"Address: {address}")
            parts.extend(
                [
                    f"latitude: {latitude}",
                    f"longitude: {longitude}",
                    "Map: https://www.google.com/maps/search/?api=1&query="
                    f"{latitude},{longitude}",
                ]
            )
            await self._accept_message(
                message,
                [],
                claim_ids=claim_ids,
                text_override="\n".join(parts),
            )

    async def _handle_media(
        self,
        update: Any,
        context: Any,
        *,
        _startup_replay: bool = False,
    ) -> None:
        async with self._track_intake():
            claim_ids = self._claim_ids_for_update(update)
            if self._claims_are_completed(claim_ids):
                return
            message = self._message_from_update(update)
            # Authorization must precede get_file/download_as_bytearray.
            if message is None or not self._authorized(message):
                await self._complete_claims(claim_ids)
                return
            if not _startup_replay and self._hold_startup_update(
                "media", update, context, claim_ids
            ):
                return
            session_id = self._message_session_id(message)
            self._begin_media_download(session_id)
            try:
                try:
                    attachments, notes = await self._download_attachments(message)
                except Exception as exc:
                    logger.warning(
                        "Telegram media download failed: %s",
                        _redact_error(exc, self._token),
                    )
                    attachments = []
                    notes = ["[A Telegram attachment could not be downloaded.]"]
                await self._accept_message(
                    message,
                    attachments,
                    claim_ids=claim_ids,
                    notes=notes,
                )
            finally:
                self._end_media_download(session_id)

    def _message_key(self, message: Any) -> str:
        chat_id = str(getattr(getattr(message, "chat", None), "id", "") or "")
        message_id = str(getattr(message, "message_id", "") or "")
        return f"{chat_id}:{message_id}" if message_id else ""

    @staticmethod
    def _message_session_id(message: Any) -> str:
        chat = getattr(message, "chat", None)
        user = getattr(message, "from_user", None)
        return _session_id(
            str(getattr(chat, "id", "") or ""),
            str(getattr(user, "id", "") or ""),
            _is_group(message),
            _effective_thread_id(message),
        )

    def _begin_media_download(self, session_id: str) -> None:
        fence = self._media_downloads.get(session_id)
        if fence is None:
            fence = _MediaDownloadFence()
            self._media_downloads[session_id] = fence
        if fence.count == 0:
            fence.changed = asyncio.Event()
        fence.count += 1
        fence.generation += 1

    def _end_media_download(self, session_id: str) -> None:
        fence = self._media_downloads.get(session_id)
        if fence is None:
            return
        fence.count = max(0, fence.count - 1)
        if fence.count == 0:
            fence.changed.set()
            if session_id not in self._pending:
                self._media_downloads.pop(session_id, None)

    def _hold_startup_update(
        self,
        kind: str,
        update: Any,
        context: Any,
        claim_ids: Sequence[str],
    ) -> bool:
        if not self._startup_hold_closed:
            return False
        claim_id = claim_ids[0] if claim_ids else ""
        if claim_id and claim_id in self._startup_held_claims:
            return True
        if (
            self._startup_approvals_enabled
            and kind == "command"
            and self._startup_approval_command(update)
        ):
            return False
        self._startup_updates.append((kind, update, context))
        if claim_id:
            self._startup_held_claims.add(claim_id)
        return True

    def _startup_approval_command(self, update: Any) -> bool:
        """Let approval control reach a turn resumed behind the startup gate."""

        message = getattr(update, "effective_message", None)
        if message is None:
            return False
        text = _expand_telegram_text_links(message)
        if _is_group(message):
            text = self._clean_routing_mention(text)
        invocation = parse_command(text.strip())
        return bool(
            invocation is not None
            and invocation.command.name in {"approve", "deny"}
        )

    async def _replay_startup_update(
        self, kind: str, update: Any, context: Any
    ) -> None:
        if kind == "text":
            await self._handle_text(update, context, _startup_replay=True)
            return
        if kind == "command":
            await self._handle_command(update, context, _startup_replay=True)
            return
        if kind == "location":
            await self._handle_location(update, context, _startup_replay=True)
            return
        if kind == "media":
            await self._handle_media(update, context, _startup_replay=True)
            return
        logger.warning("Telegram dropped an unknown held update kind: %s", kind)

    async def _accept_message(
        self,
        message: Any,
        attachments: List[media.Attachment],
        *,
        claim_ids: Sequence[str] = (),
        text_override: Optional[str] = None,
        notes: Optional[List[str]] = None,
    ) -> None:
        message_key = self._message_key(message)
        if message_key and self._seen.is_duplicate(message_key):
            await self._complete_claims(claim_ids)
            return

        chat = getattr(message, "chat", None)
        user = getattr(message, "from_user", None)
        chat_id = str(getattr(chat, "id", "") or "")
        user_id = str(getattr(user, "id", "") or "")
        thread_id = _effective_thread_id(message)
        is_group = _is_group(message)
        raw_text = (
            text_override
            if text_override is not None
            else _expand_telegram_text_links(message)
        )
        clean_text = (
            self._clean_routing_mention(raw_text)
            if is_group
            else raw_text.strip()
        )

        invocation = parse_command(clean_text) if not attachments else None
        session_id = self._message_session_id(message)
        message_id = str(getattr(message, "message_id", "") or "")
        if invocation is not None:
            if invocation.command.name == "new" and not invocation.arguments:
                dropped_claims = self._drop_pending_session(session_id)
                await self._complete_claims(dropped_claims)
            task = asyncio.create_task(
                self._run_command(
                    chat_id,
                    session_id,
                    message_id,
                    thread_id,
                    invocation,
                    list(claim_ids),
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return

        body = self._compose_text(
            message,
            clean_text,
            attachments,
            notes or [],
            is_group=is_group,
            user_id=user_id,
        )
        if not body and not attachments:
            await self._complete_claims(claim_ids)
            return

        username = str(getattr(user, "username", "") or "")
        full_name = str(getattr(user, "full_name", "") or "").strip()
        inbound = InboundMessage(
            chat_id=chat_id,
            session_id=session_id,
            user_id=user_id,
            user_name=full_name or (f"@{username}" if username else user_id),
            text=body,
            is_group=is_group,
            thread_id=thread_id,
            message_ids=[message_id] if message_id else [],
            claim_ids=list(claim_ids),
            attachments=attachments,
        )
        self._enqueue(inbound)

    def _compose_text(
        self,
        message: Any,
        body: str,
        attachments: List[media.Attachment],
        notes: List[str],
        *,
        is_group: bool,
        user_id: str,
    ) -> str:
        body = media.inline_documents(body, attachments)
        unreadable = media.describe_unreadable(attachments)
        all_notes = [note for note in [*notes, unreadable] if note]
        if all_notes:
            note_text = "\n".join(dict.fromkeys(all_notes))
            body = f"{note_text}\n\n{body}" if body else note_text

        reply = getattr(message, "reply_to_message", None)
        quote = getattr(message, "quote", None)
        quoted = str(getattr(quote, "text", "") or "").strip()
        if not quoted and reply is not None:
            quoted = str(
                getattr(reply, "text", None)
                or getattr(reply, "caption", None)
                or ""
            ).strip()
        if quoted:
            snippet = quoted[:QUOTE_SNIPPET_LIMIT]
            prefix = (
                f'[Replying to your previous message: "{snippet}"]'
                if self._is_reply_to_bot(message)
                else f'[Replying to: "{snippet}"]'
            )
            body = f"{prefix}\n\n{body}" if body else prefix

        if is_group:
            user = getattr(message, "from_user", None)
            full_name = str(getattr(user, "full_name", "") or "").strip()
            username = str(getattr(user, "username", "") or "").strip()
            name = full_name or (f"@{username}" if username else user_id)
            attribution = f"[Telegram group sender: {name} (user_id={user_id})]"
            body = f"{attribution}\n\n{body}" if body else attribution
        return body.strip()

    async def _run_command(
        self,
        chat_id: str,
        session_id: str,
        message_id: str,
        thread_id: str,
        invocation: CommandInvocation,
        claim_ids: List[str],
    ) -> None:
        try:
            claim_id = claim_ids[-1] if claim_ids else ""
            await self._on_command(
                chat_id,
                session_id,
                message_id,
                thread_id,
                invocation,
                claim_id,
            )
            await self._complete_claims(claim_ids)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Telegram command /%s failed for %s",
                invocation.command.name,
                identity_pseudonym(chat_id, "tg"),
            )
            if claim_ids:
                self._fail("A durable Telegram command could not be completed.")

    def _drop_pending_session(self, session_id: str) -> List[str]:
        task = self._pending_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
        dropped = self._pending.pop(session_id, None)
        self._pending_started.pop(session_id, None)
        queued = self._queued.pop(session_id, None)
        fence = self._media_downloads.get(session_id)
        if fence is not None and fence.count == 0:
            self._media_downloads.pop(session_id, None)
        claims: List[str] = []
        for message in (dropped, queued):
            if message is not None:
                claims.extend(message.claim_ids)
        return claims

    async def _download_attachments(
        self, message: Any
    ) -> tuple[List[media.Attachment], List[str]]:
        attachments: List[media.Attachment] = []
        notes: List[str] = []

        photos = getattr(message, "photo", None) or []
        if photos:
            attachment = await self._download_one(
                message,
                photos[-1],
                filename="photo.jpg",
                mime="image/jpeg",
                media_type="image",
            )
            attachments.append(attachment)
            return attachments, notes

        voice = getattr(message, "voice", None)
        if voice is not None:
            attachment = await self._download_one(
                message,
                voice,
                filename="voice.ogg",
                mime=str(getattr(voice, "mime_type", "") or "audio/ogg"),
                media_type="ptt",
            )
            attachments.append(attachment)
            return attachments, notes

        audio = getattr(message, "audio", None)
        if audio is not None:
            attachment = await self._download_one(
                message,
                audio,
                filename=str(
                    getattr(audio, "file_name", "") or "audio.mp3"
                ),
                mime=str(getattr(audio, "mime_type", "") or "audio/mpeg"),
                media_type="audio",
            )
            attachments.append(attachment)
            return attachments, notes

        video = getattr(message, "video", None)
        if video is not None:
            attachment = await self._download_one(
                message,
                video,
                filename=str(
                    getattr(video, "file_name", "") or "video.mp4"
                ),
                mime=str(getattr(video, "mime_type", "") or "video/mp4"),
                media_type="video",
            )
            attachments.append(attachment)
            return attachments, notes

        sticker = getattr(message, "sticker", None)
        if sticker is not None:
            emoji = str(getattr(sticker, "emoji", "") or "").strip()
            if bool(getattr(sticker, "is_animated", False)) or bool(
                getattr(sticker, "is_video", False)
            ):
                notes.append(
                    "[The sender sent an animated Telegram sticker"
                    + (f" {emoji}" if emoji else "")
                    + ".]"
                )
                return attachments, notes
            attachment = await self._download_one(
                message,
                sticker,
                filename="sticker.webp",
                mime="image/webp",
                media_type="sticker",
            )
            attachments.append(attachment)
            if emoji:
                notes.append(f"[Telegram sticker emoji: {emoji}]")
            return attachments, notes

        document = getattr(message, "document", None)
        if document is not None:
            size = int(getattr(document, "file_size", 0) or 0)
            if not 0 < size <= MAX_DOWNLOAD_BYTES:
                notes.append(
                    "[The Telegram document is too large or its size could "
                    "not be verified. Maximum: 20 MB.]"
                )
                return attachments, notes
            filename = str(
                getattr(document, "file_name", "") or "document.bin"
            )
            mime = str(
                getattr(document, "mime_type", "")
                or mimetypes.guess_type(filename)[0]
                or "application/octet-stream"
            ).lower()
            suffix = Path(filename).suffix.lower()
            if mime.startswith("image/") or suffix in _IMAGE_SUFFIXES:
                media_type = "image"
            elif mime.startswith("audio/"):
                media_type = "audio"
            elif mime.startswith("video/") or suffix in _VIDEO_SUFFIXES:
                media_type = "video"
            else:
                media_type = "document"
            attachment = await self._download_one(
                message,
                document,
                filename=filename,
                mime=mime,
                media_type=media_type,
            )
            attachments.append(attachment)
        return attachments, notes

    async def _download_one(
        self,
        message: Any,
        source: Any,
        *,
        filename: str,
        mime: str,
        media_type: str,
    ) -> media.Attachment:
        advertised_size = int(getattr(source, "file_size", 0) or 0)
        if advertised_size > MAX_DOWNLOAD_BYTES:
            raise ValueError("Telegram attachment exceeds the 20 MB download cap")

        telegram_file = await source.get_file()
        payload = bytes(await telegram_file.download_as_bytearray())
        if len(payload) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Telegram attachment exceeds the 20 MB download cap")

        fallback_suffix = (
            mimetypes.guess_extension(mime.split(";", 1)[0].strip()) or ".bin"
        )
        safe_name = _safe_filename(filename, f"attachment{fallback_suffix}")
        if not Path(safe_name).suffix:
            safe_name += fallback_suffix
        unique = _safe_filename(
            str(getattr(source, "file_unique_id", "") or ""),
            str(getattr(message, "message_id", "") or "message"),
        )
        target = (
            self._config.media_dir
            / "telegram"
            / f"{unique}_{safe_name}"
        )
        await asyncio.to_thread(target.write_bytes, payload)
        return media.Attachment(
            path=target.resolve(),
            mime=mime,
            media_type=media_type,
            file_name=safe_name,
        )

    # -- batching -------------------------------------------------------

    def _hold_inbound(self, message: InboundMessage) -> None:
        existing = self._held_inbound.get(message.session_id)
        if existing is None:
            self._held_inbound[message.session_id] = message
        else:
            self._merge_message(existing, message)

    def _release_held_inbound(self) -> None:
        held = list(self._held_inbound.values())
        self._held_inbound.clear()
        for message in held:
            self._queue_turn(message)

    @staticmethod
    def _merge_message(
        pending: InboundMessage, message: InboundMessage
    ) -> None:
        if message.text:
            pending.text = (
                f"{pending.text}\n{message.text}"
                if pending.text
                else message.text
            )
        pending.message_ids.extend(message.message_ids)
        pending.claim_ids.extend(message.claim_ids)
        pending.attachments.extend(message.attachments)
        pending.chat_id = message.chat_id
        pending.user_id = message.user_id
        pending.user_name = message.user_name
        pending.is_group = message.is_group
        pending.thread_id = message.thread_id

    def _enqueue(self, message: InboundMessage) -> None:
        if self._drop_delayed_deliveries:
            self._hold_inbound(message)
            return
        key = message.session_id
        loop = asyncio.get_running_loop()
        now = loop.time()
        pending = self._pending.get(key)
        started = self._pending_started.get(key)
        if (
            pending is not None
            and started is not None
            and now >= started + self._batch_hard_cap
        ):
            self._flush_pending_now(key)
            pending = None

        if pending is None:
            self._pending[key] = message
            self._pending_started[key] = now
        else:
            self._merge_message(pending, message)

        self._cancel_pending_timer(key)
        self._pending_tasks[key] = asyncio.create_task(
            self._flush_after_quiet(
                key,
                len(message.text),
                bool(message.attachments),
            )
        )

    def _cancel_pending_timer(self, key: str) -> None:
        task = self._pending_tasks.pop(key, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    def _flush_pending_now(self, key: str) -> None:
        self._cancel_pending_timer(key)
        self._pending_started.pop(key, None)
        message = self._pending.pop(key, None)
        if message is not None:
            if self._drop_delayed_deliveries:
                self._hold_inbound(message)
            else:
                self._queue_turn(message)
        fence = self._media_downloads.get(key)
        if fence is not None and fence.count == 0:
            self._media_downloads.pop(key, None)

    async def _wait_for_media_download(
        self, key: str, hard_remaining: float
    ) -> None:
        fence = self._media_downloads.get(key)
        if (
            fence is None
            or fence.count == 0
            or fence.generation <= fence.spent_generation
            or hard_remaining <= 0
        ):
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + min(
            MEDIA_DOWNLOAD_GRACE_SECONDS,
            hard_remaining,
        )
        try:
            while fence.count > 0:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                changed = fence.changed
                try:
                    await asyncio.wait_for(changed.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
        finally:
            # One stuck download may delay one batch once, never every later
            # text turn in the same session.
            fence.spent_generation = fence.generation

    async def _flush_after_quiet(
        self, key: str, last_text_length: int, has_media: bool
    ) -> None:
        if has_media:
            quiet_delay = self._media_batch_delay
        elif last_text_length >= SPLIT_THRESHOLD:
            quiet_delay = self._text_batch_split_delay
        else:
            quiet_delay = self._text_batch_delay
        loop = asyncio.get_running_loop()
        started = self._pending_started.get(key, loop.time())
        hard_remaining = max(
            0.0,
            started + self._batch_hard_cap - loop.time(),
        )
        try:
            await asyncio.sleep(min(quiet_delay, hard_remaining))
        except asyncio.CancelledError:
            return
        if self._pending_tasks.get(key) is not asyncio.current_task():
            return
        if not has_media:
            hard_remaining = max(
                0.0,
                started + self._batch_hard_cap - loop.time(),
            )
            try:
                await asyncio.sleep(
                    min(MEDIA_REGISTRATION_GRACE_SECONDS, hard_remaining)
                )
            except asyncio.CancelledError:
                return
            if self._pending_tasks.get(key) is not asyncio.current_task():
                return
        hard_remaining = max(
            0.0,
            started + self._batch_hard_cap - loop.time(),
        )
        try:
            await self._wait_for_media_download(key, hard_remaining)
        except asyncio.CancelledError:
            return
        if self._pending_tasks.get(key) is not asyncio.current_task():
            return
        self._flush_pending_now(key)

    def _queue_turn(self, message: InboundMessage) -> None:
        key = message.session_id
        active = self._turn_tasks.get(key)
        if active is not None and not active.done():
            queued = self._queued.get(key)
            if queued is None:
                self._queued[key] = message
            else:
                self._merge_message(queued, message)
            return
        task = asyncio.create_task(self._run_turn_queue(key, message))
        self._turn_tasks[key] = task

    async def _run_turn_queue(
        self, key: str, message: Optional[InboundMessage]
    ) -> None:
        current = asyncio.current_task()
        try:
            while message is not None:
                try:
                    await self._handler(message)
                    await self._complete_claims(message.claim_ids)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Handling a Telegram message from %s failed",
                        identity_pseudonym(message.chat_id, "tg"),
                    )
                    if message.claim_ids:
                        self._fail(
                            "A durable Telegram message could not be completed."
                        )
                if self.failure:
                    self._queued.pop(key, None)
                    break
                message = self._queued.pop(key, None)
        finally:
            if self._turn_tasks.get(key) is current:
                self._turn_tasks.pop(key, None)

    # -- outbound -------------------------------------------------------

    @staticmethod
    def _integer_id(value: str) -> Optional[int]:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _thread_kwargs(thread_id: str, *, typing: bool = False) -> Dict[str, Any]:
        if not thread_id or (thread_id == "1" and not typing):
            return {}
        try:
            return {"message_thread_id": int(thread_id)}
        except (TypeError, ValueError):
            return {}

    def _reply_kwargs(self, reply_to: str, delivery_index: int) -> Dict[str, Any]:
        if (
            not reply_to
            or self._reply_to_mode == "off"
            or (self._reply_to_mode == "first" and delivery_index > 0)
        ):
            return {}
        message_id = self._integer_id(reply_to)
        return {"reply_to_message_id": message_id} if message_id is not None else {}

    def _preview_kwargs(self) -> Dict[str, Any]:
        if not self._disable_link_previews:
            return {}
        return {
            "link_preview_options": LinkPreviewOptions(is_disabled=True)
        }

    @asynccontextmanager
    async def typing(self, chat_id: str, thread_id: str = ""):
        task = asyncio.create_task(self._typing_loop(chat_id, thread_id))
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _typing_loop(self, chat_id: str, thread_id: str) -> None:
        while True:
            await self._send_typing(chat_id, thread_id)
            await asyncio.sleep(TYPING_REFRESH_SECONDS)

    async def _send_typing(self, chat_id: str, thread_id: str) -> None:
        if self._bot is None:
            return
        try:
            await self._bot.send_chat_action(
                chat_id=normalize_telegram_chat_id(chat_id),
                action=ChatAction.TYPING,
                **self._thread_kwargs(thread_id, typing=True),
            )
        except Exception:
            pass  # Cosmetic; a typing failure must never cost the reply.

    async def send(
        self,
        chat_id: str,
        text: str,
        reply_to: str = "",
        *,
        thread_id: str = "",
        deliver_media: bool = True,
        delivery_ledger: Optional[DeliveryUnitLedger] = None,
    ) -> SendResult:
        if self._bot is None:
            return SendResult(False, "Telegram transport is not connected")

        if deliver_media:
            attachments, cleaned = media.extract_outbound(
                text or "", self._config.outbound_media_roots
            )
        else:
            attachments, cleaned = [], text or ""

        formatted = to_telegram(cleaned).strip()
        chunks = (
            [
                chunk
                for chunk in split_telegram_message(
                    formatted, MAX_MESSAGE_LENGTH
                )
                if chunk.strip()
            ]
            if formatted
            else []
        )
        if not chunks and not attachments:
            return SendResult(False, "response had no deliverable content")

        units = []
        if delivery_ledger is not None:
            descriptors = [
                (
                    "text",
                    delivery_fingerprint(
                        "telegram-text-v2",
                        chunk,
                        thread_id,
                        reply_to
                        if self._reply_kwargs(reply_to, index)
                        else "",
                        "no-preview" if self._disable_link_previews else "preview",
                    ),
                )
                for index, chunk in enumerate(chunks)
            ]
            for attachment in attachments:
                fingerprint = await asyncio.to_thread(
                    file_delivery_fingerprint,
                    attachment.path,
                    "telegram-file-v1",
                    attachment.media_type,
                    attachment.file_name,
                    thread_id,
                    reply_to
                    if self._reply_kwargs(reply_to, len(descriptors))
                    else "",
                )
                descriptors.append(("file", fingerprint))
            units = await delivery_ledger.prepare(descriptors)

        delivery_index = 0
        for chunk in chunks:
            kwargs = {
                **self._thread_kwargs(thread_id),
                **self._reply_kwargs(reply_to, delivery_index),
                **self._preview_kwargs(),
            }
            send = lambda chunk=chunk, kwargs=kwargs: self._send_text_chunk(
                chat_id, chunk, kwargs
            )
            result = (
                await delivery_ledger.run(units[delivery_index], send)
                if delivery_ledger is not None
                else as_send_result(await send())
            )
            if not result:
                if delivery_ledger is None and delivery_index:
                    return SendResult(False, result.error)
                return result
            delivery_index += 1

        for attachment in attachments:
            kwargs = {
                **self._thread_kwargs(thread_id),
                **self._reply_kwargs(reply_to, delivery_index),
            }
            send = lambda attachment=attachment, kwargs=kwargs: self._send_attachment(
                chat_id, attachment.path, kwargs
            )
            result = (
                await delivery_ledger.run(units[delivery_index], send)
                if delivery_ledger is not None
                else as_send_result(await send())
            )
            if not result:
                if delivery_ledger is None and delivery_index:
                    return SendResult(False, result.error)
                return result
            delivery_index += 1
        return SendResult(True)

    @staticmethod
    def _sent_message_id(value: Any) -> str:
        message_id = getattr(value, "message_id", "")
        return str(message_id) if isinstance(message_id, (str, int)) else ""

    async def _send_text_chunk(
        self, chat_id: str, chunk: str, kwargs: Dict[str, Any]
    ) -> SendResult:
        target = normalize_telegram_chat_id(chat_id)
        try:
            sent = await self._bot.send_message(
                chat_id=target,
                text=chunk,
                parse_mode=ParseMode.MARKDOWN_V2,
                **kwargs,
            )
            return SendResult(True, message_id=self._sent_message_id(sent))
        except BadRequest as exc:
            written = str(exc).lower()
            retry_kwargs = dict(kwargs)
            if "message to be replied not found" in written:
                retry_kwargs.pop("reply_to_message_id", None)
                return await self._send_text_once(
                    target,
                    chunk,
                    ParseMode.MARKDOWN_V2,
                    retry_kwargs,
                )
            if "thread" in written and (
                "not found" in written or "invalid" in written
            ):
                retry_kwargs.pop("message_thread_id", None)
                return await self._send_text_once(
                    target,
                    chunk,
                    ParseMode.MARKDOWN_V2,
                    retry_kwargs,
                )
            if "parse" in written or "markdown" in written:
                logger.warning(
                    "Telegram rejected MarkdownV2; sending plaintext instead."
                )
                return await self._send_text_once(
                    target,
                    strip_telegram_markdown(chunk),
                    None,
                    kwargs,
                )
            logger.error(
                "Telegram rejected a message to %s: %s",
                identity_pseudonym(chat_id, "tg"),
                _redact_error(exc, self._token),
            )
            return SendResult(False, _redact_error(exc, self._token))
        except Exception as exc:
            logger.error(
                "Sending to Telegram chat %s failed: %s",
                identity_pseudonym(chat_id, "tg"),
                _redact_error(exc, self._token),
            )
            return _send_failure(exc, self._token)

    async def _send_text_once(
        self,
        chat_id: int | str,
        text: str,
        parse_mode: Any,
        kwargs: Dict[str, Any],
    ) -> SendResult:
        try:
            sent = await self._bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                **kwargs,
            )
            return SendResult(True, message_id=self._sent_message_id(sent))
        except Exception as exc:
            logger.error(
                "Telegram fallback delivery failed: %s",
                _redact_error(exc, self._token),
            )
            return _send_failure(exc, self._token)

    async def _send_attachment(
        self,
        chat_id: str,
        path: Path,
        kwargs: Dict[str, Any],
    ) -> SendResult:
        suffix = path.suffix.lower()
        target = normalize_telegram_chat_id(chat_id)
        try:
            if suffix == ".gif":
                method = self._bot.send_animation
                argument = "animation"
            elif suffix in _IMAGE_SUFFIXES:
                method = self._bot.send_photo
                argument = "photo"
            elif suffix in _VIDEO_SUFFIXES:
                method = self._bot.send_video
                argument = "video"
            elif suffix in _VOICE_SUFFIXES:
                method = self._bot.send_voice
                argument = "voice"
            elif suffix in _AUDIO_SUFFIXES:
                method = self._bot.send_audio
                argument = "audio"
            else:
                method = self._bot.send_document
                argument = "document"

            try:
                with path.open("rb") as stream:
                    sent = await method(
                        chat_id=target,
                        **{argument: stream},
                        **kwargs,
                    )
                return SendResult(True, message_id=self._sent_message_id(sent))
            except BadRequest:
                if argument not in {"photo", "animation", "video"}:
                    raise
                # Hermes falls back to a document when Telegram cannot render
                # a valid workspace file in its native media endpoint.
                with path.open("rb") as stream:
                    sent = await self._bot.send_document(
                        chat_id=target,
                        document=stream,
                        **kwargs,
                    )
                return SendResult(True, message_id=self._sent_message_id(sent))
        except Exception as exc:
            logger.error(
                "Sending Telegram attachment failed: %s",
                _redact_error(exc, self._token),
            )
            return _send_failure(exc, self._token)


__all__ = [
    "ChannelError",
    "InboundMessage",
    "TelegramChannel",
    "normalize_telegram_allowed_users",
    "normalize_telegram_bot_token",
    "normalize_telegram_chat_id",
    "normalize_telegram_home_chat_id",
    "normalize_telegram_topic_id",
    "validate_settings",
]
