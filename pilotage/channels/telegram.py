"""Telegram channel adapted from Hermes' proven Telegram adapter.

The Bot API transport stays deliberately small: authorization happens before
downloads, bursts become one agent turn, topics keep their routing lane, and
outbound Markdown/media use Telegram-native delivery.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import mimetypes
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlparse

from .. import media
from ..commands import CommandInvocation, parse_command
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
    from telegram.error import BadRequest, Conflict, Forbidden, InvalidToken
    from telegram.ext import Application, MessageHandler, filters
    from telegram.request import HTTPXRequest

    TELEGRAM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised through preflight
    LinkPreviewOptions = Update = ChatAction = ParseMode = Any
    BadRequest = Conflict = Forbidden = InvalidToken = RuntimeError
    Application = MessageHandler = HTTPXRequest = Any
    filters = None
    TELEGRAM_AVAILABLE = False


MAX_MESSAGE_LENGTH = 4096
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
SPLIT_THRESHOLD = 4000
QUOTE_SNIPPET_LIMIT = 500
TYPING_REFRESH_SECONDS = 5.0
SHUTDOWN_STEP_SECONDS = 10.0

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})
_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"})
_VOICE_SUFFIXES = frozenset({".ogg", ".opus"})
_AUDIO_SUFFIXES = frozenset({".mp3", ".m4a", ".aac"})
_FOREIGN_BOT_HANDLE_RE = re.compile(r"[a-z0-9_]{2,29}bot", re.IGNORECASE)


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
    allowed_users = _split_env("TELEGRAM_ALLOWED_USERS")
    if "*" in allowed_users:
        raise ConfigError(
            "TELEGRAM_ALLOWED_USERS must name explicit users; '*' is not allowed"
        )
    if any(
        re.fullmatch(r"[1-9][0-9]*", user_id) is None
        for user_id in allowed_users
    ):
        raise ConfigError(
            "TELEGRAM_ALLOWED_USERS must contain numeric Telegram user IDs only"
        )

    enabled = settings.flag("telegram.enabled", False)
    if enabled and not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
        raise ConfigError(
            "telegram.enabled is true but TELEGRAM_BOT_TOKEN is not configured"
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

    policy = settings.text("telegram.group_policy", "disabled").lower()
    if policy not in {"disabled", "allowlist"}:
        raise ConfigError(
            "telegram.group_policy must be 'disabled' or 'allowlist', "
            f"not {policy!r}"
        )
    group_allow_from = settings.names("telegram.group_allow_from")
    if any(
        chat_id != "*"
        and re.fullmatch(r"-?[1-9][0-9]*", chat_id) is None
        for chat_id in group_allow_from
    ):
        raise ConfigError(
            "telegram.group_allow_from must contain numeric Telegram chat IDs or '*'"
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
    attachments: List[media.Attachment] = field(default_factory=list)


Handler = Callable[[InboundMessage], Awaitable[None]]
CommandHandler = Callable[
    [str, str, str, str, CommandInvocation], Awaitable[None]
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


def _safe_filename(value: str, fallback: str) -> str:
    name = Path(str(value or "")).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return cleaned[:180] or fallback


def _redact_error(exc: BaseException, token: str) -> str:
    text = str(exc)
    if token:
        text = text.replace(token, "<redacted Telegram token>")
    return re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot<redacted>", text)


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
        self._group_policy = settings.text(
            "telegram.group_policy", "disabled"
        ).lower()
        self._group_allow_from = frozenset(
            settings.names("telegram.group_allow_from")
        )
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
        self._drop_delayed_deliveries = False
        self.stopped = asyncio.Event()
        self.failure: Optional[str] = None

        self._seen = MessageDeduplicator()
        self._reported_blocked: set[str] = set()
        self._pending: Dict[str, InboundMessage] = {}
        self._pending_started: Dict[str, float] = {}
        self._pending_tasks: Dict[str, asyncio.Task] = {}
        self._queued: Dict[str, InboundMessage] = {}
        self._turn_tasks: Dict[str, asyncio.Task] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._intake_tasks: set[asyncio.Task] = set()
        self._held_inbound: Dict[str, InboundMessage] = {}

    # -- lifetime -------------------------------------------------------

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
        if self._group_policy == "allowlist" and not self._group_allow_from:
            logger.warning(
                "Telegram group policy is allowlist but group_allow_from is "
                "empty - every group message will be ignored."
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
        updates_request = HTTPXRequest(**request_options)
        app = (
            Application.builder()
            .token(self._token)
            .request(request)
            .get_updates_request(updates_request)
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
            await app.start()
            if app.updater is None:
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
                        drop_pending_updates=True,
                    ),
                    timeout=30.0,
                )
                self._webhook_mode = True
            else:
                await app.bot.delete_webhook(drop_pending_updates=True)
                await asyncio.wait_for(
                    app.updater.start_polling(
                        drop_pending_updates=True,
                        error_callback=self._polling_error_callback,
                    ),
                    timeout=30.0,
                )
                self._webhook_mode = False
        except asyncio.CancelledError:
            await asyncio.shield(self.stop())
            raise
        except BaseException as exc:
            await asyncio.shield(self.stop())
            if isinstance(exc, InvalidToken):
                raise ChannelError("Telegram rejected TELEGRAM_BOT_TOKEN.") from exc
            if isinstance(exc, ChannelError):
                raise
            raise ChannelError(
                "Telegram could not start: " + _redact_error(exc, self._token)
            ) from exc

        self._running = True
        self._release_held_inbound()
        logger.info(
            "Telegram channel ready as @%s (%s)",
            self._bot_username or "unknown",
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

    async def stop_intake(self) -> None:
        """Stop receiving updates and dispatch every batch already accepted."""

        if self._intake_stopped:
            self._release_held_inbound()
            return
        self._intake_stopped = True
        # Hermes fences delayed delivery before the first teardown await. PTB
        # may already have advanced the update offset, so a late handler must
        # be held instead of scheduling work that teardown will never observe.
        self._drop_delayed_deliveries = True
        self._running = False
        app = self._app
        updater = getattr(app, "updater", None) if app is not None else None
        if updater is not None and getattr(updater, "running", False):
            await self._bounded_step(updater.stop(), "update receiver stop")

        timers = set(self._pending_tasks.values())
        for key in list(self._pending):
            self._flush_pending_now(key)
        self._release_held_inbound()
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
        self._release_held_inbound()

    async def stop(self, *, drain_timeout_seconds: float = 0.0) -> None:
        """Stop intake, then give already accepted work a bounded drain."""

        await self.stop_intake()

        loop = asyncio.get_running_loop()
        timeout = (
            0.0
            if self.failure
            else max(0.0, float(drain_timeout_seconds))
        )
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
        self._release_held_inbound()

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

    async def _handle_update_error(self, update: object, context: Any) -> None:
        error = getattr(context, "error", RuntimeError("unknown update error"))
        logger.error(
            "Telegram update handling failed: %s",
            _redact_error(error, self._token),
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

    def _is_group_allowed(self, chat_id: str) -> bool:
        return bool(
            self._group_policy == "allowlist"
            and chat_id
            and (
                "*" in self._group_allow_from
                or chat_id in self._group_allow_from
            )
        )

    def _report_blocked(self, message: Any) -> None:
        user = getattr(message, "from_user", None)
        identity = str(getattr(user, "id", "") or "unknown")
        if identity in self._reported_blocked:
            return
        self._reported_blocked.add(identity)
        logger.warning("Ignored a Telegram message from %s.", identity)

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
        if _is_group(message):
            if not self._is_group_allowed(chat_id):
                return False
            if not self._group_message_is_triggered(message):
                return False
            return True
        if not self._is_user_allowed(user):
            self._report_blocked(message)
            return False
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
        return getattr(update, "effective_message", None) or getattr(
            update, "message", None
        )

    async def _handle_text(self, update: Any, context: Any) -> None:
        async with self._track_intake():
            message = self._message_from_update(update)
            if message is None or not self._authorized(message):
                return
            await self._accept_message(message, [])

    async def _handle_command(self, update: Any, context: Any) -> None:
        async with self._track_intake():
            message = self._message_from_update(update)
            if message is None or not self._authorized(message):
                return
            await self._accept_message(message, [])

    async def _handle_location(self, update: Any, context: Any) -> None:
        async with self._track_intake():
            message = self._message_from_update(update)
            if message is None or not self._authorized(message):
                return
            venue = getattr(message, "venue", None)
            location = (
                getattr(venue, "location", None)
                if venue is not None
                else getattr(message, "location", None)
            )
            if location is None:
                return
            latitude = getattr(location, "latitude", None)
            longitude = getattr(location, "longitude", None)
            if latitude is None or longitude is None:
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
            await self._accept_message(message, [], text_override="\n".join(parts))

    async def _handle_media(self, update: Any, context: Any) -> None:
        async with self._track_intake():
            message = self._message_from_update(update)
            # Authorization must precede get_file/download_as_bytearray.
            if message is None or not self._authorized(message):
                return
            try:
                attachments, notes = await self._download_attachments(message)
            except Exception as exc:
                logger.warning(
                    "Telegram media download failed: %s",
                    _redact_error(exc, self._token),
                )
                attachments = []
                notes = ["[A Telegram attachment could not be downloaded.]"]
            await self._accept_message(message, attachments, notes=notes)

    def _message_key(self, message: Any) -> str:
        chat_id = str(getattr(getattr(message, "chat", None), "id", "") or "")
        message_id = str(getattr(message, "message_id", "") or "")
        return f"{chat_id}:{message_id}" if message_id else ""

    async def _accept_message(
        self,
        message: Any,
        attachments: List[media.Attachment],
        *,
        text_override: Optional[str] = None,
        notes: Optional[List[str]] = None,
    ) -> None:
        message_key = self._message_key(message)
        if message_key and self._seen.is_duplicate(message_key):
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
            else str(
                getattr(message, "text", None)
                or getattr(message, "caption", None)
                or ""
            )
        )
        clean_text = (
            self._clean_routing_mention(raw_text)
            if is_group
            else raw_text.strip()
        )

        invocation = parse_command(clean_text) if not attachments else None
        session_id = _session_id(chat_id, user_id, is_group, thread_id)
        message_id = str(getattr(message, "message_id", "") or "")
        if invocation is not None:
            if invocation.command.name == "new":
                self._drop_pending_session(session_id)
            task = asyncio.create_task(
                self._run_command(
                    chat_id,
                    session_id,
                    message_id,
                    thread_id,
                    invocation,
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
    ) -> None:
        try:
            await self._on_command(
                chat_id,
                session_id,
                message_id,
                thread_id,
                invocation,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Telegram command /%s failed for %s",
                invocation.command.name,
                chat_id,
            )

    def _drop_pending_session(self, session_id: str) -> None:
        task = self._pending_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
        self._pending.pop(session_id, None)
        self._pending_started.pop(session_id, None)
        self._queued.pop(session_id, None)

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
        target.write_bytes(payload)
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
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Handling a Telegram message from %s failed",
                        message.chat_id,
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
    ) -> bool:
        if self._bot is None:
            return False

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
            return False

        delivery_index = 0
        for chunk in chunks:
            kwargs = {
                **self._thread_kwargs(thread_id),
                **self._reply_kwargs(reply_to, delivery_index),
                **self._preview_kwargs(),
            }
            if not await self._send_text_chunk(chat_id, chunk, kwargs):
                return False
            delivery_index += 1

        for attachment in attachments:
            kwargs = {
                **self._thread_kwargs(thread_id),
                **self._reply_kwargs(reply_to, delivery_index),
            }
            if not await self._send_attachment(chat_id, attachment.path, kwargs):
                return False
            delivery_index += 1
        return True

    async def _send_text_chunk(
        self, chat_id: str, chunk: str, kwargs: Dict[str, Any]
    ) -> bool:
        target = normalize_telegram_chat_id(chat_id)
        try:
            await self._bot.send_message(
                chat_id=target,
                text=chunk,
                parse_mode=ParseMode.MARKDOWN_V2,
                **kwargs,
            )
            return True
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
                chat_id,
                _redact_error(exc, self._token),
            )
            return False
        except Exception as exc:
            logger.error(
                "Sending to Telegram chat %s failed: %s",
                chat_id,
                _redact_error(exc, self._token),
            )
            return False

    async def _send_text_once(
        self,
        chat_id: int | str,
        text: str,
        parse_mode: Any,
        kwargs: Dict[str, Any],
    ) -> bool:
        try:
            await self._bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                **kwargs,
            )
            return True
        except Exception as exc:
            logger.error(
                "Telegram fallback delivery failed: %s",
                _redact_error(exc, self._token),
            )
            return False

    async def _send_attachment(
        self,
        chat_id: str,
        path: Path,
        kwargs: Dict[str, Any],
    ) -> bool:
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
                    await method(
                        chat_id=target,
                        **{argument: stream},
                        **kwargs,
                    )
                return True
            except BadRequest:
                if argument not in {"photo", "animation", "video"}:
                    raise
                # Hermes falls back to a document when Telegram cannot render
                # a valid workspace file in its native media endpoint.
                with path.open("rb") as stream:
                    await self._bot.send_document(
                        chat_id=target,
                        document=stream,
                        **kwargs,
                    )
                return True
        except Exception as exc:
            logger.error(
                "Sending Telegram attachment %s failed: %s",
                path.name,
                _redact_error(exc, self._token),
            )
            return False


__all__ = [
    "ChannelError",
    "InboundMessage",
    "TelegramChannel",
    "normalize_telegram_chat_id",
    "validate_settings",
]
