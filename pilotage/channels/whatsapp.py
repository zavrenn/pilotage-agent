"""The WhatsApp channel: supervise the Node bridge, poll it, answer through it.

The bridge owns the WhatsApp connection. This module owns its lifetime, decides
which messages are ours to answer, groups a burst of messages into one turn, and
sends the reply back.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from .. import media
from ..commands import CommandInvocation, parse_command
from ..config import Config
from ..delivery import SendResult
from .dedup import MessageDeduplicator
from .formatting import to_whatsapp

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 1.0
POLL_ERROR_BACKOFF_SECONDS = 5.0
# WhatsApp clears the "typing…" indicator by itself after a few seconds, so it
# has to be renewed for as long as the model is still thinking.
TYPING_REFRESH_SECONDS = 8.0
HTTP_TIMEOUT_SECONDS = 30.0
MEDIA_HTTP_TIMEOUT_SECONDS = 120.0
BRIDGE_READY_TIMEOUT_SECONDS = 120.0
BRIDGE_RESTART_ATTEMPTS = 3
BRIDGE_RESTART_DELAY_SECONDS = 5.0
BRIDGE_TOKEN_HEADER = "X-Pilotage-Bridge-Token"
# The Node bridge needs process basics, not the agent's model, database,
# transcription, search, Telegram, or profile credentials.
BRIDGE_INHERITED_ENV = frozenset(
    {
        "APPDATA",
        "COLORTERM",
        "COMSPEC",
        "FORCE_COLOR",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "LANGUAGE",
        "LC_ADDRESS",
        "LC_ALL",
        "LC_COLLATE",
        "LC_CTYPE",
        "LC_IDENTIFICATION",
        "LC_MEASUREMENT",
        "LC_MESSAGES",
        "LC_MONETARY",
        "LC_NAME",
        "LC_NUMERIC",
        "LC_PAPER",
        "LC_TELEPHONE",
        "LC_TIME",
        "LOCALAPPDATA",
        "LOGNAME",
        "NODE_EXTRA_CA_CERTS",
        "NODE_OPTIONS",
        "NODE_PATH",
        "NO_COLOR",
        "PATH",
        "PATHEXT",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "TZ",
        "USER",
        "USERPROFILE",
        "WINDIR",
    }
)
# A single message this long is almost certainly one chunk of a longer paste, so
# we wait longer for the rest of it.
SPLIT_THRESHOLD = 6000
# How much of a quoted message is shown back to the agent. (Hermes)
QUOTE_SNIPPET_LIMIT = 500
# Start the conversation over. Typed by a person on a phone keyboard, so case
# and a trailing space are not mistakes worth punishing.
RESET_COMMAND = "/new"
_BARE_PHONE_RE = re.compile(r"^\+?[\d \t().-]+$")
_DIRECT_JID_DOMAINS = frozenset({"s.whatsapp.net", "lid", "c.us"})
_HOME_JID_DOMAINS = _DIRECT_JID_DOMAINS | {"g.us"}


class WhatsAppSessionError(ValueError):
    """The linked-device credential file is absent, unreadable, or incomplete."""


def validate_whatsapp_session(session_dir: Path) -> None:
    """Require credentials produced by a completed Baileys pairing flow."""

    path = Path(session_dir) / "creds.json"
    try:
        credentials = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise WhatsAppSessionError(
            "WhatsApp linked-device credentials are missing or unreadable"
        ) from exc
    if not isinstance(credentials, dict):
        raise WhatsAppSessionError(
            "WhatsApp linked-device credentials are incomplete"
        )

    me = credentials.get("me")
    has_linked_identity = (
        isinstance(me, dict)
        and isinstance(me.get("id"), str)
        and bool(me["id"].strip())
    )
    # Baileys 7 QR pairing persists the signed account and signal identity but
    # leaves `registered` false. Pairing by phone-number code sets that flag.
    qr_pairing_complete = (
        isinstance(credentials.get("account"), dict)
        and bool(credentials["account"])
        and isinstance(credentials.get("signalIdentities"), list)
        and bool(credentials["signalIdentities"])
    )
    pairing_code_complete = credentials.get("registered") is True
    if not has_linked_identity or not (
        qr_pairing_complete or pairing_code_complete
    ):
        raise WhatsAppSessionError(
            "WhatsApp linked-device credentials are incomplete"
        )


def normalize_whatsapp_chat_id(value: str) -> str:
    """Return a bridge-safe direct/group JID or reject an invalid target."""

    written = str(value or "").strip()
    if not written:
        raise ValueError("A WhatsApp home number or chat ID is required")

    if "@" in written:
        local, _, domain = written.partition("@")
        local = local.split(":", 1)[0]
        domain = domain.lower()
        if domain in _DIRECT_JID_DOMAINS:
            local = local.removeprefix("+")
            valid_local = local.isascii() and local.isdigit()
        elif domain == "g.us":
            valid_local = bool(re.fullmatch(r"\d+(?:-\d+)?", local))
        else:
            valid_local = False
        if domain not in _HOME_JID_DOMAINS or not valid_local:
            raise ValueError(f"Invalid WhatsApp chat ID: {written!r}")
        if domain == "c.us":
            domain = "s.whatsapp.net"
        return f"{local}@{domain}"

    if _BARE_PHONE_RE.fullmatch(written):
        digits = re.sub(r"\D+", "", written)
        if digits.isascii() and digits.isdigit():
            return f"{digits}@s.whatsapp.net"
    raise ValueError(
        "Invalid WhatsApp number; use the country code and digits"
    )


def build_bridge_environment(
    *,
    base: Optional[Dict[str, str]] = None,
    token: str = "",
    allowed_senders: Optional[List[str]] = None,
    allowed_groups: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Build the bridge's complete, intentionally small child environment."""

    source = os.environ if base is None else base
    child = {
        name: value
        for name, value in source.items()
        if name.upper() in BRIDGE_INHERITED_ENV
    }
    child["PILOTAGE_ALLOWED_SENDERS"] = ",".join(
        sorted(str(value) for value in (allowed_senders or ()))
    )
    child["PILOTAGE_ALLOWED_GROUPS"] = ",".join(
        sorted(str(value) for value in (allowed_groups or ()))
    )
    if token:
        child["PILOTAGE_BRIDGE_TOKEN"] = token
    return child


def _bare_whatsapp_id(value: str) -> str:
    """Normalize a JID or bridge-resolved identity to its stable bare id."""
    written = str(value or "").strip().lstrip("+")
    if "@" in written:
        written = written.split("@", 1)[0]
    if ":" in written:
        written = written.split(":", 1)[0]
    return written


def _canonical_whatsapp_identity(*values: str) -> str:
    """Collapse phone-JID/LID aliases with Hermes' stable selection rule."""
    aliases = {
        normalized
        for value in values
        if (normalized := _bare_whatsapp_id(value))
    }
    return min(aliases, key=lambda candidate: (len(candidate), candidate)) if aliases else ""


def _session_id(
    chat_id: str,
    is_group: bool,
    sender_id: str,
    sender_number: str,
    identities: List[str],
) -> str:
    """Build the production conversation boundary for one inbound message.

    WhatsApp can flip both DM chat ids and group participant ids between phone
    and LID forms. The bridge supplies their resolved alias set.
    """
    candidates = [sender_number, sender_id, *identities]
    if not is_group:
        candidates.insert(0, chat_id)
        return _canonical_whatsapp_identity(*candidates) or chat_id

    participant = _canonical_whatsapp_identity(*candidates)
    return f"{chat_id}:{participant}" if participant else chat_id


def _bot_ids_from_message(data: Dict[str, Any]) -> set[str]:
    raw = data.get("botIds")
    if not isinstance(raw, list):
        return set()
    return {
        normalized
        for value in raw
        if (normalized := _bare_whatsapp_id(str(value)))
    }


def _message_is_reply_to_bot(data: Dict[str, Any]) -> bool:
    quoted = _bare_whatsapp_id(str(data.get("quotedParticipant") or ""))
    return bool(quoted and quoted in _bot_ids_from_message(data))


def _message_mentions_bot(data: Dict[str, Any]) -> bool:
    bot_ids = _bot_ids_from_message(data)
    if not bot_ids:
        return False
    mentioned = data.get("mentionedIds")
    mentioned_ids = (
        {
            normalized
            for value in mentioned
            if (normalized := _bare_whatsapp_id(str(value)))
        }
        if isinstance(mentioned, list)
        else set()
    )
    if mentioned_ids.intersection(bot_ids):
        return True

    # Hermes keeps this fallback because some WhatsApp message forms lose the
    # structured mention list while retaining the visible @number.
    body = str(data.get("body") or "").lower()
    return any(
        f"@{bot_id.lower()}" in body or bot_id.lower() in body
        for bot_id in bot_ids
    )


def _clean_bot_mention_text(text: str, data: Dict[str, Any]) -> str:
    """Remove the bot's routing mention from accepted group text. (Hermes)"""
    if not text:
        return text
    cleaned = text
    for bot_id in _bot_ids_from_message(data):
        cleaned = re.sub(
            rf"@{re.escape(bot_id)}\b[,:\-]*\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    return cleaned.strip() or text


class ChannelError(RuntimeError):
    """The channel cannot start, or cannot keep running."""


@dataclass
class InboundMessage:
    chat_id: str
    # Conversation state key. DMs use the chat; groups add the canonical
    # participant so one room can never share history or tool state.
    session_id: str
    sender_id: str
    sender_number: str
    push_name: str
    text: str
    is_group: bool
    message_ids: List[str] = field(default_factory=list)
    # Files the bridge downloaded for this turn, already checked against the
    # media cache roots.
    attachments: List[media.Attachment] = field(default_factory=list)


Handler = Callable[[InboundMessage], Awaitable[None]]
# Called with the delivery chat, isolated session, requesting message id, and
# the registry-resolved management command.
CommandHandler = Callable[[str, str, str, CommandInvocation], Awaitable[None]]


class WhatsAppChannel:
    def __init__(self, config: Config, handler: Handler, on_command: CommandHandler):
        self._config = config
        self._handler = handler
        self._on_command = on_command
        self._process: Optional[subprocess.Popen] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False
        self._startup_hold_closed = False
        self._startup_events: List[Dict[str, Any]] = []
        self._pending: Dict[str, InboundMessage] = {}
        self._pending_tasks: Dict[str, asyncio.Task] = {}
        self._reported_blocked: set[str] = set()
        # The bridge replays its queue on reconnect, and a restart can overlap
        # with messages already answered.
        self._seen = MessageDeduplicator()
        self._pending_started: Dict[str, float] = {}
        # One active model turn and at most one merged follow-up per isolated
        # session. This bounds work even when messages arrive faster than replies.
        self._queued: Dict[str, InboundMessage] = {}
        self._turn_tasks: Dict[str, asyncio.Task] = {}
        # Read receipts run detached; hold a reference so they are not
        # garbage-collected mid-flight.
        self._pending_tasks_background: set[asyncio.Task] = set()
        self._bridge_token = secrets.token_urlsafe(32)
        self._base_url = f"http://127.0.0.1:{config.bridge_port}"
        # Set when the channel has given up. Whoever owns the process waits on
        # this, so a dead bridge stops the agent instead of leaving it idle.
        self.stopped = asyncio.Event()
        self.failure: Optional[str] = None

    # -- lifetime -----------------------------------------------------------

    def hold_inbound(self) -> None:
        self._startup_hold_closed = True

    def release_inbound(self) -> None:
        if not self._startup_hold_closed and not self._startup_events:
            return
        held = list(self._startup_events)
        self._startup_hold_closed = False
        self._startup_events.clear()
        for event in held:
            self._accept(event)

    async def start(self) -> None:
        self._preflight()
        await self._stop_stale_bridge()
        self._http = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={BRIDGE_TOKEN_HEADER: self._bridge_token},
        )
        try:
            self._spawn_bridge()
            await self._wait_until_connected()
        except BaseException:
            await asyncio.shield(self.stop())
            raise
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        if not self._config.allowed_senders:
            logger.warning(
                "No allowed senders configured — every message will be ignored. "
                "Message the agent once, then add the number it logs to PILOTAGE_ALLOWED_SENDERS."
            )
        if self._config.answer_groups and not self._config.group_allow_from:
            logger.warning(
                "WhatsApp group policy is allowlist but group_allow_from is empty — "
                "every group message will be ignored."
            )
        logger.info("WhatsApp channel ready")

    async def stop_intake(self) -> None:
        """Stop accepting bridge events and dispatch every accepted batch."""

        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None

        self.release_inbound()
        timers = set(self._pending_tasks.values())
        for key in list(self._pending):
            self._flush_pending_now(key)
        for task in timers:
            task.cancel()
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)
        self._pending_tasks.clear()

    async def stop(self, *, drain_timeout_seconds: float = 0.0) -> None:
        """Stop intake, then give already accepted work a bounded drain."""

        await self.stop_intake()

        owned = set(self._pending_tasks.values())
        owned.update(self._pending_tasks_background)
        owned.update(self._turn_tasks.values())
        owned.discard(asyncio.current_task())
        live = {task for task in owned if not task.done()}
        timeout = (
            0.0
            if self.failure
            else max(0.0, float(drain_timeout_seconds))
        )
        pending = live
        if live and timeout > 0:
            _, pending = await asyncio.wait(live, timeout=timeout)
        if pending:
            logger.warning(
                "WhatsApp shutdown drain expired with %d accepted task(s)",
                len(pending),
            )
        for task in pending:
            task.cancel()
        if owned:
            await asyncio.gather(*owned, return_exceptions=True)

        self._pending_tasks.clear()
        self._pending_tasks_background.clear()
        self._turn_tasks.clear()
        self._pending.clear()
        self._pending_started.clear()
        self._queued.clear()
        try:
            if self._http is not None:
                await self._http.aclose()
                self._http = None
        finally:
            try:
                await asyncio.to_thread(self._terminate_bridge)
            finally:
                self.stopped.set()

    def _preflight(self) -> None:
        if shutil.which("node") is None:
            raise ChannelError("Node.js is not on PATH. The WhatsApp bridge needs Node 20 or newer.")
        script = self._config.bridge_script
        if not script.exists():
            raise ChannelError(f"The bridge script is missing at {script}.")
        # Dependencies are installed by the operator, never by the runtime.
        if not (self._config.bridge_dir / "node_modules").exists():
            raise ChannelError(
                f"The bridge dependencies are not installed. Run `npm install` in {self._config.bridge_dir}."
            )
        self._config.session_dir.mkdir(parents=True, exist_ok=True)
        self._config.media_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _pidfile(self) -> Path:
        return self._config.state_dir / "bridge.pid"

    async def _stop_stale_bridge(self) -> None:
        """Ask a bridge we own to stop without trusting a reusable OS pid."""
        try:
            record = json.loads(self._pidfile.read_text(encoding="utf-8"))
            pid = int(record["pid"])
            port = int(record["port"])
            token = str(record["token"])
            if pid <= 0 or not 1 <= port <= 65535 or not token:
                raise ValueError("invalid bridge ownership record")
        except (OSError, ValueError, KeyError, TypeError):
            return

        base_url = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient(
                timeout=2.0,
                headers={BRIDGE_TOKEN_HEADER: token},
            ) as client:
                response = await client.get(f"{base_url}/health")
                response.raise_for_status()
                if int(response.json().get("pid", 0)) != pid:
                    raise ValueError("bridge pid does not match its ownership record")
                logger.info("Stopping a bridge left over from an earlier run (pid %s)", pid)
                response = await client.post(f"{base_url}/shutdown")
                response.raise_for_status()
                for _ in range(30):
                    await asyncio.sleep(0.1)
                    try:
                        await client.get(f"{base_url}/health")
                    except httpx.TransportError:
                        break
                else:
                    raise ChannelError(
                        f"Owned stale bridge on port {port} did not stop."
                    )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning(
                "Could not verify the stale bridge; leaving its process untouched: %s",
                exc,
            )
        with contextlib.suppress(OSError):
            self._pidfile.unlink()

    def _spawn_bridge(self) -> None:
        command = [
            "node",
            str(self._config.bridge_script),
            "--port",
            str(self._config.bridge_port),
            "--session",
            str(self._config.session_dir),
            "--media",
            str(self._config.media_dir),
            "--read-receipts",
            "1" if self._config.send_read_receipts else "0",
            "--answer-groups",
            "1" if self._config.answer_groups else "0",
        ]
        # stdout is inherited on purpose: the pairing QR code is printed there.
        child_env = build_bridge_environment(
            token=self._bridge_token,
            allowed_senders=list(self._config.allowed_senders),
            allowed_groups=list(self._config.group_allow_from),
        )
        self._process = subprocess.Popen(
            command,
            cwd=str(self._config.bridge_dir),
            env=child_env,
        )
        self._config.state_dir.mkdir(parents=True, exist_ok=True)
        temp = self._pidfile.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {
                    "pid": self._process.pid,
                    "port": self._config.bridge_port,
                    "token": self._bridge_token,
                }
            ),
            encoding="utf-8",
        )
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        os.replace(temp, self._pidfile)

    def _terminate_bridge(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            record = json.loads(self._pidfile.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            record = {}
        if record.get("token") == self._bridge_token:
            with contextlib.suppress(OSError):
                self._pidfile.unlink()
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    async def _wait_until_connected(self) -> None:
        assert self._http is not None
        deadline = asyncio.get_running_loop().time() + BRIDGE_READY_TIMEOUT_SECONDS
        announced = False
        while asyncio.get_running_loop().time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise ChannelError(f"The bridge exited with code {self._process.returncode} before connecting.")
            try:
                response = await self._http.get(f"{self._base_url}/health", timeout=5.0)
                if response.status_code == 200 and response.json().get("connected"):
                    return
                if not announced:
                    logger.info("Waiting for the bridge to connect to WhatsApp...")
                    announced = True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
        raise ChannelError(
            "The bridge did not connect to WhatsApp in time. "
            "If this is the first run, scan the QR code printed above."
        )

    # -- inbound ------------------------------------------------------------

    async def _poll_loop(self) -> None:
        assert self._http is not None
        while self._running:
            try:
                if self._process is not None and self._process.poll() is not None:
                    if not await self._restart_bridge():
                        break
                    continue
                response = await self._http.get(f"{self._base_url}/messages")
                response.raise_for_status()
                for event in response.json() or []:
                    self._accept(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                logger.warning("Polling the bridge failed: %s", exc)
                await asyncio.sleep(POLL_ERROR_BACKOFF_SECONDS)
                continue
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _restart_bridge(self) -> bool:
        """Bring the bridge back after it died. Give up loudly rather than idle.

        A logged-out session or a broken install will not fix itself by being
        restarted, so the attempts are capped and the channel then fails.
        """
        code = self._process.returncode if self._process is not None else None
        logger.error("The bridge exited with code %s", code)
        for attempt in range(1, BRIDGE_RESTART_ATTEMPTS + 1):
            await asyncio.sleep(BRIDGE_RESTART_DELAY_SECONDS)
            logger.info("Restarting the bridge (attempt %s of %s)", attempt, BRIDGE_RESTART_ATTEMPTS)
            self._terminate_bridge()
            try:
                self._spawn_bridge()
                await self._wait_until_connected()
            except (ChannelError, OSError) as exc:
                logger.error("Restart failed: %s", exc)
                continue
            logger.info("The bridge is back")
            return True

        self._fail(
            f"The WhatsApp bridge keeps dying (last exit code {code}). "
            "Check its output above — a logged-out session needs pairing again."
        )
        return False

    def _fail(self, message: str) -> None:
        self.failure = message
        self._running = False
        logger.error(message)
        self.stopped.set()

    def _accept(self, event: Dict[str, Any]) -> None:
        if self._startup_hold_closed:
            self._startup_events.append(dict(event))
            return
        chat_id = str(event.get("chatId") or "")
        if not chat_id:
            return
        is_group = bool(event.get("isGroup"))
        if is_group and (
            not self._is_group_allowed(chat_id)
            or not self._group_message_is_triggered(event)
        ):
            return

        message_id = str(event.get("messageId") or "")
        if message_id and self._seen.is_duplicate(message_id):
            return

        sender_id = str(event.get("senderId") or "")
        sender_number = str(event.get("senderNumber") or "")
        raw_identities = event.get("identities")
        identities = (
            [str(value) for value in raw_identities]
            if isinstance(raw_identities, list)
            else []
        )
        if not self._is_allowed(sender_id, sender_number, identities):
            self._report_blocked(sender_id, sender_number)
            return

        attachments = media.collect(event, self._config.media_roots)
        text = self._compose_text(event, attachments)
        if not text and not attachments:
            return

        session_id = _session_id(chat_id, is_group, sender_id, sender_number, identities)

        # Fire and forget: a slow bridge must not hold up the answer. (Hermes)
        self._pending_tasks_background.add(
            asyncio.create_task(self._mark_read(event.get("readReceiptKey")))
        )

        invocation = parse_command(text)
        if invocation is not None:
            if invocation.command.name == "new":
                # Drop a half-written batch on the spot. Sending it after /new
                # would answer the conversation the person just ended.
                waiting = self._pending_tasks.pop(session_id, None)
                if waiting is not None:
                    waiting.cancel()
                self._pending.pop(session_id, None)
                self._pending_started.pop(session_id, None)
                self._queued.pop(session_id, None)
            self._pending_tasks_background.add(
                asyncio.create_task(
                    self._run_command(
                        chat_id, session_id, message_id, invocation
                    )
                )
            )
            return

        message = InboundMessage(
            chat_id=chat_id,
            session_id=session_id,
            sender_id=sender_id,
            sender_number=sender_number,
            push_name=str(event.get("pushName") or ""),
            text=text,
            is_group=is_group,
            message_ids=[message_id],
            attachments=attachments,
        )
        self._enqueue(message)

    def _compose_text(
        self, event: Dict[str, Any], attachments: List[media.Attachment]
    ) -> str:
        """Turn a bridge event into the text the model actually sees."""
        body = str(event.get("body") or "").strip()
        if event.get("isGroup"):
            body = _clean_bot_mention_text(body, event)
        media_type = str(event.get("mediaType") or "")

        # The bridge writes "[image received]" when media arrives with no
        # caption. That is a placeholder, not something the sender said, so it
        # gives way to a description of what we can and cannot read.
        if body == f"[{media_type} received]":
            body = ""

        body = media.inline_documents(body, attachments)

        note = media.describe_unreadable(attachments)
        if note:
            body = f"{note}\n\n{body}" if body else note

        # Which earlier message a reply points at is not recoverable from
        # history: the same text can appear more than once. (Hermes)
        quoted = str(event.get("quotedText") or "").strip()
        if quoted and event.get("quotedMessageId"):
            snippet = quoted[:QUOTE_SNIPPET_LIMIT]
            own = _message_is_reply_to_bot(event)
            prefix = (
                f'[Replying to your previous message: "{snippet}"]'
                if own
                else f'[Replying to: "{snippet}"]'
            )
            body = f"{prefix}\n\n{body}" if body else prefix

        return body.strip()

    async def _mark_read(self, key: Any) -> None:
        """Show the blue ticks, if the operator turned them on."""
        try:
            if not self._config.send_read_receipts or self._http is None:
                return
            if not isinstance(key, dict):
                return
            try:
                response = await self._http.post(
                    f"{self._base_url}/read", json={"key": key}, timeout=5.0
                )
                if response.status_code != 200:
                    logger.warning(
                        "Marking a message read failed with HTTP %s", response.status_code
                    )
            except httpx.HTTPError as exc:
                logger.warning("Marking a message read failed: %s", exc)
        finally:
            self._pending_tasks_background.discard(asyncio.current_task())

    def _is_allowed(
        self, sender_id: str, sender_number: str, identities: List[str]
    ) -> bool:
        allowed = {
            normalized
            for value in self._config.allowed_senders
            if (normalized := _bare_whatsapp_id(value))
        }
        if not allowed or "*" in allowed:
            # An empty allowlist means nobody, never everybody.
            return False
        candidates = {
            normalized
            for value in [sender_id, sender_number, *identities]
            if (normalized := _bare_whatsapp_id(value))
        }
        # The bridge already made the same decision before media extraction.
        # Keep this independent check at the Python trust boundary.
        return bool(allowed.intersection(candidates))

    def _is_group_allowed(self, chat_id: str) -> bool:
        """Apply the group-chat location gate after the global sender gate."""
        if self._config.group_policy != "allowlist":
            return False
        allowed = {
            normalized
            for value in self._config.group_allow_from
            if (normalized := _bare_whatsapp_id(value))
        }
        candidate = _bare_whatsapp_id(chat_id)
        return bool(candidate and ("*" in allowed or candidate in allowed))

    def _group_message_is_triggered(self, event: Dict[str, Any]) -> bool:
        """Require a direct trigger when configured, matching Hermes."""
        if not self._config.require_mention:
            return True
        if str(event.get("body") or "").strip().startswith("/"):
            return True
        if _message_is_reply_to_bot(event):
            return True
        return _message_mentions_bot(event)

    def _report_blocked(self, sender_id: str, sender_number: str) -> None:
        """Note once, per sender, that a message was ignored.

        Nothing more. WhatsApp does not always send a sender's phone number —
        sometimes it sends an internal alias, which is digits too and reads
        exactly like a number. Telling the operator to allowlist whatever is
        printed here would sooner or later be telling them to allowlist that.
        """
        identity = sender_number or sender_id
        if identity in self._reported_blocked:
            return
        self._reported_blocked.add(identity)
        logger.warning("Ignored a message from %s.", identity)

    async def _run_command(
        self,
        chat_id: str,
        session_id: str,
        message_id: str,
        invocation: CommandInvocation,
    ) -> None:
        """Dispatch one recognized command without involving the model."""
        try:
            await self._on_command(chat_id, session_id, message_id, invocation)
        except Exception:  # noqa: BLE001 - one bad command must not stop the channel
            logger.exception("Command /%s failed for %s", invocation.command.name, chat_id)
        finally:
            self._pending_tasks_background.discard(asyncio.current_task())

    @staticmethod
    def _merge_message(pending: InboundMessage, message: InboundMessage) -> None:
        if message.text:
            pending.text = (
                f"{pending.text}\n{message.text}" if pending.text else message.text
            )
        pending.message_ids.extend(message.message_ids)
        # A picture and the question about it can arrive separately. (Hermes)
        pending.attachments.extend(message.attachments)
        # Replies and quotes belong to the newest delivery address and sender
        # representation, even when LID/phone aliases share one session.
        pending.chat_id = message.chat_id
        pending.sender_id = message.sender_id
        pending.sender_number = message.sender_number
        pending.push_name = message.push_name
        pending.is_group = message.is_group

    def _enqueue(self, message: InboundMessage) -> None:
        """Merge into a quiet-period batch bounded by a hard deadline.

        People send three messages in a row and mean one question. Answering the
        first before the other two land wastes a turn; waiting forever under a
        steady stream would starve the conversation.
        """
        key = message.session_id
        loop = asyncio.get_running_loop()
        now = loop.time()
        pending = self._pending.get(key)
        started = self._pending_started.get(key)
        if (
            pending is not None
            and started is not None
            and now >= started + self._config.text_batch_hard_cap_seconds
        ):
            self._flush_pending_now(key)
            pending = None

        if pending is None:
            self._pending[key] = message
            self._pending_started[key] = now
        else:
            self._merge_message(pending, message)

        self._cancel_pending_timer(key)
        self._pending_tasks[key] = asyncio.create_task(self._flush_after_quiet(key, message.text))

    def _cancel_pending_timer(self, key: str) -> None:
        task = self._pending_tasks.pop(key, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    def _flush_pending_now(self, key: str) -> None:
        self._cancel_pending_timer(key)
        self._pending_started.pop(key, None)
        message = self._pending.pop(key, None)
        if message is not None:
            self._queue_turn(message)

    async def _flush_after_quiet(self, key: str, last_text: str) -> None:
        quiet_delay = (
            self._config.text_batch_split_delay_seconds
            if len(last_text) >= SPLIT_THRESHOLD
            else self._config.text_batch_delay_seconds
        )
        loop = asyncio.get_running_loop()
        started = self._pending_started.get(key, loop.time())
        hard_remaining = max(
            0.0,
            started + self._config.text_batch_hard_cap_seconds - loop.time(),
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

    async def _run_turn_queue(self, key: str, message: InboundMessage) -> None:
        current = asyncio.current_task()
        try:
            while message is not None:
                try:
                    await self._handler(message)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - one turn must not stop the channel
                    logger.exception("Handling a message from %s failed", message.chat_id)
                if self.failure:
                    self._queued.pop(key, None)
                    break
                message = self._queued.pop(key, None)
        finally:
            if self._turn_tasks.get(key) is current:
                self._turn_tasks.pop(key, None)

    # -- outbound -----------------------------------------------------------

    @contextlib.asynccontextmanager
    async def typing(self, chat_id: str):
        """Show "typing…" in the chat for as long as the block runs.

        The wait for a model is long enough that silence reads as a broken
        agent. The indicator is cosmetic, so every failure here is swallowed:
        it must never cost a reply.
        """
        task = asyncio.create_task(self._typing_loop(chat_id))
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _typing_loop(self, chat_id: str) -> None:
        while True:
            await self._send_typing(chat_id)
            await asyncio.sleep(TYPING_REFRESH_SECONDS)

    async def _send_typing(self, chat_id: str) -> None:
        if self._http is None:
            return
        try:
            await self._http.post(
                f"{self._base_url}/typing", json={"chatId": chat_id}, timeout=5.0
            )
        except httpx.HTTPError:
            pass  # Cosmetic — never worth interrupting a turn.

    async def send(
        self,
        chat_id: str,
        text: str,
        reply_to: str = "",
        *,
        deliver_media: bool = True,
    ) -> SendResult:
        if self._http is None:
            return SendResult(False, "WhatsApp transport is not connected")
        try:
            target_chat_id = normalize_whatsapp_chat_id(chat_id)
        except ValueError as exc:
            return SendResult(False, str(exc))

        if deliver_media:
            attachments, cleaned = media.extract_outbound(
                text or "", self._config.outbound_media_roots
            )
        else:
            attachments, cleaned = [], text or ""
        # Model replies take Hermes' send order: visible text first, then each
        # extracted attachment. System echoes opt out because user-derived text
        # must never turn a spoken MEDIA phrase into a file delivery.
        body = to_whatsapp(cleaned).strip()
        if not body and not attachments:
            return SendResult(False, "response had no deliverable content")

        sent_any = False
        try:
            if body:
                payload: Dict[str, Any] = {
                    "chatId": target_chat_id,
                    "message": body,
                }
                if reply_to:
                    # Quote the message being answered. One agent can be talking
                    # to several people at once, and an answer that arrives a
                    # minute after the question is otherwise guesswork.
                    payload["replyTo"] = reply_to
                response = await self._http.post(
                    f"{self._base_url}/send", json=payload
                )
                response.raise_for_status()
                sent_any = True

            for attachment in attachments:
                payload = {
                    "chatId": target_chat_id,
                    "filePath": str(attachment.path),
                    "mediaType": attachment.media_type,
                }
                if attachment.media_type == "document":
                    payload["fileName"] = attachment.file_name
                response = await self._http.post(
                    f"{self._base_url}/send-media",
                    json=payload,
                    timeout=MEDIA_HTTP_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                sent_any = True
        except httpx.HTTPError as exc:
            logger.error("Sending to %s failed: %s", target_chat_id, exc)
            retryable = False
            retry_after = None
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                # The bridge can surface 500 after earlier internal chunks were
                # already delivered, so only the explicit backpressure and
                # pre-send not-connected guard are safe to retry.
                retryable = status in {429, 503}
                if status == 429:
                    written = exc.response.headers.get("retry-after", "")
                    try:
                        retry_after = float(written)
                    except (TypeError, ValueError):
                        retry_after = None
            elif isinstance(exc, httpx.ConnectTimeout):
                retryable = True
            elif isinstance(exc, httpx.TimeoutException):
                # The bridge may have accepted a timed-out request.
                retryable = False
            elif isinstance(exc, httpx.TransportError):
                retryable = True
            if sent_any:
                # Retrying the whole response would duplicate the already
                # delivered text or an earlier attachment.
                retryable = False
            return SendResult(
                False,
                str(exc),
                retryable=retryable,
                retry_after=retry_after,
            )
        return SendResult(True)
