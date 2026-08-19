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
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from .. import media
from ..config import Config
from .dedup import MessageDeduplicator
from .formatting import to_whatsapp

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 1.0
POLL_ERROR_BACKOFF_SECONDS = 5.0
# WhatsApp clears the "typing…" indicator by itself after a few seconds, so it
# has to be renewed for as long as the model is still thinking.
TYPING_REFRESH_SECONDS = 8.0
HTTP_TIMEOUT_SECONDS = 30.0
BRIDGE_READY_TIMEOUT_SECONDS = 120.0
BRIDGE_RESTART_ATTEMPTS = 3
BRIDGE_RESTART_DELAY_SECONDS = 5.0
# A single message this long is almost certainly one chunk of a longer paste, so
# we wait longer for the rest of it.
SPLIT_THRESHOLD = 6000
# How much of a quoted message is shown back to the agent. (Hermes)
QUOTE_SNIPPET_LIMIT = 500
# Start the conversation over. Typed by a person on a phone keyboard, so case
# and a trailing space are not mistakes worth punishing.
RESET_COMMAND = "/new"


class ChannelError(RuntimeError):
    """The channel cannot start, or cannot keep running."""


@dataclass
class InboundMessage:
    chat_id: str
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
# Called with a chat id, and the id of the message that asked, when a
# conversation should start over.
Reset = Callable[[str, str], Awaitable[None]]


class WhatsAppChannel:
    def __init__(self, config: Config, handler: Handler, on_reset: Reset):
        self._config = config
        self._handler = handler
        self._on_reset = on_reset
        self._process: Optional[subprocess.Popen] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False
        self._pending: Dict[str, InboundMessage] = {}
        self._pending_tasks: Dict[str, asyncio.Task] = {}
        self._reported_blocked: set[str] = set()
        # The bridge replays its queue on reconnect, and a restart can overlap
        # with messages already answered.
        self._seen = MessageDeduplicator()
        # Read receipts run detached; hold a reference so they are not
        # garbage-collected mid-flight.
        self._pending_tasks_background: set[asyncio.Task] = set()
        self._base_url = f"http://127.0.0.1:{config.bridge_port}"
        # Set when the channel has given up. Whoever owns the process waits on
        # this, so a dead bridge stops the agent instead of leaving it idle.
        self.stopped = asyncio.Event()
        self.failure: Optional[str] = None

    # -- lifetime -----------------------------------------------------------

    async def start(self) -> None:
        self._preflight()
        self._kill_stale_bridge()
        self._http = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        self._spawn_bridge()
        await self._wait_until_connected()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        if not self._config.allowed_senders:
            logger.warning(
                "No allowed senders configured — every message will be ignored. "
                "Message the agent once, then add the number it logs to PILOTAGE_ALLOWED_SENDERS."
            )
        logger.info("WhatsApp channel ready")

    async def stop(self) -> None:
        self._running = False
        self.stopped.set()
        for task in list(self._pending_tasks.values()):
            task.cancel()
        self._pending_tasks.clear()
        self._pending.clear()
        for task in list(self._pending_tasks_background):
            task.cancel()
        self._pending_tasks_background.clear()
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._terminate_bridge()

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

    def _kill_stale_bridge(self) -> None:
        """Kill a bridge we started before and never cleaned up.

        Only pids we wrote ourselves are touched.
        """
        try:
            record = json.loads(self._pidfile.read_text(encoding="utf-8"))
            pid = int(record["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            return
        logger.info("Stopping a bridge left over from an earlier run (pid %s)", pid)
        _kill_pid(pid)
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
        ]
        # stdout is inherited on purpose: the pairing QR code is printed there.
        self._process = subprocess.Popen(command, cwd=str(self._config.bridge_dir))
        self._config.state_dir.mkdir(parents=True, exist_ok=True)
        self._pidfile.write_text(
            json.dumps({"pid": self._process.pid, "port": self._config.bridge_port}),
            encoding="utf-8",
        )

    def _terminate_bridge(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
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
        chat_id = str(event.get("chatId") or "")
        if not chat_id:
            return
        is_group = bool(event.get("isGroup"))
        if is_group and not self._config.answer_groups:
            return

        message_id = str(event.get("messageId") or "")
        if message_id and self._seen.is_duplicate(message_id):
            return

        sender_id = str(event.get("senderId") or "")
        sender_number = str(event.get("senderNumber") or "")
        raw_identities = event.get("identities")
        identities = [str(value) for value in raw_identities] if isinstance(raw_identities, list) else []
        if not self._is_allowed(sender_id, sender_number, identities):
            self._report_blocked(sender_id, sender_number)
            return

        attachments = media.collect(event, self._config.media_roots)
        text = self._compose_text(event, attachments)
        if not text and not attachments:
            return

        # Fire and forget: a slow bridge must not hold up the answer. (Hermes)
        self._pending_tasks_background.add(
            asyncio.create_task(self._mark_read(event.get("readReceiptKey")))
        )

        if text.strip().lower() == RESET_COMMAND:
            # Drop the batch on the spot rather than inside the task below.
            # Someone who types this mid-thought has changed their mind about
            # the thought, and sending it afterwards would answer the
            # conversation they just ended.
            waiting = self._pending_tasks.pop(chat_id, None)
            if waiting is not None:
                waiting.cancel()
            self._pending.pop(chat_id, None)
            self._pending_tasks_background.add(
                asyncio.create_task(self._reset(chat_id, message_id))
            )
            return

        message = InboundMessage(
            chat_id=chat_id,
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
            bot_ids = event.get("botIds")
            own = isinstance(bot_ids, list) and event.get("quotedParticipant") in bot_ids
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

    def _is_allowed(self, sender_id: str, sender_number: str, identities: List[str]) -> bool:
        allowed = self._config.allowed_senders
        if not allowed:
            # An empty allowlist means nobody, never everybody.
            return False
        # WhatsApp addresses the same person by phone number or by @lid alias,
        # and which one arrives is not ours to choose. The bridge resolves both
        # forms from the session's mapping files and sends them as `identities`,
        # so an operator can write either one in the allowlist.
        if any(identity in allowed for identity in identities):
            return True
        return sender_number in allowed or sender_id in allowed

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

    async def _reset(self, chat_id: str, message_id: str) -> None:
        """Start the conversation over, once whatever is running has finished."""
        try:
            await self._on_reset(chat_id, message_id)
        except Exception:  # noqa: BLE001 - one bad command must not stop the channel
            logger.exception("Resetting %s failed", chat_id)
        finally:
            self._pending_tasks_background.discard(asyncio.current_task())

    def _enqueue(self, message: InboundMessage) -> None:
        """Merge a message into the pending batch for its chat and restart the timer.

        People send three messages in a row and mean one question. Answering the
        first one before the other two land wastes a turn and reads as rude.
        """
        key = message.chat_id
        pending = self._pending.get(key)
        if pending is None:
            self._pending[key] = message
        else:
            if message.text:
                pending.text = (
                    f"{pending.text}\n{message.text}" if pending.text else message.text
                )
            pending.message_ids.extend(message.message_ids)
            # A picture and the question about it arrive as two messages.
            # Losing the picture here loses the point of the turn. (Hermes)
            pending.attachments.extend(message.attachments)

        task = self._pending_tasks.pop(key, None)
        if task is not None:
            task.cancel()
        self._pending_tasks[key] = asyncio.create_task(self._flush_after_quiet(key, message.text))

    async def _flush_after_quiet(self, key: str, last_text: str) -> None:
        delay = (
            self._config.text_batch_split_delay_seconds
            if len(last_text) >= SPLIT_THRESHOLD
            else self._config.text_batch_delay_seconds
        )
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        self._pending_tasks.pop(key, None)
        message = self._pending.pop(key, None)
        if message is None:
            return
        try:
            await self._handler(message)
        except Exception:  # noqa: BLE001 - one bad turn must not stop the channel
            logger.exception("Handling a message from %s failed", message.chat_id)

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

    async def send(self, chat_id: str, text: str, reply_to: str = "") -> bool:
        if self._http is None:
            return False
        # Everything leaves through here, so this is the one place the model's
        # markdown has to become WhatsApp's.
        body = to_whatsapp(text or "").strip()
        if not body:
            return False
        payload: Dict[str, Any] = {"chatId": chat_id, "message": body}
        if reply_to:
            # Quote the message being answered. One agent can be talking to
            # several people at once, and an answer that arrives a minute
            # after the question is otherwise guesswork.
            payload["replyTo"] = reply_to
        try:
            response = await self._http.post(
                f"{self._base_url}/send", json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Sending to %s failed: %s", chat_id, exc)
            return False
        return True


def _kill_pid(pid: int) -> None:
    if pid <= 0:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                check=False,
                capture_output=True,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        pass  # Already gone.
