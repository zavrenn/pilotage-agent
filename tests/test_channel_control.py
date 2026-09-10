from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilotage.channels import telegram
from pilotage.channels.whatsapp import (
    InboundMessage as WhatsAppInboundMessage,
    WhatsAppChannel,
)
from pilotage.config import Config


WA_SESSION = "212600000000"
WA_CHAT = f"{WA_SESSION}@s.whatsapp.net"
TG_SESSION = telegram._session_id("42", "42", False, "")


def _wa_message(
    text: str,
    message_id: str,
    *,
    claim_id: str = "",
) -> WhatsAppInboundMessage:
    return WhatsAppInboundMessage(
        chat_id=WA_CHAT,
        session_id=WA_SESSION,
        sender_id=WA_CHAT,
        sender_number=WA_SESSION,
        push_name="Owner",
        text=text,
        is_group=False,
        message_ids=[message_id],
        claim_ids=[claim_id] if claim_id else [],
    )


def _wa_event(message_id: str, body: str, claim_digit: str) -> dict[str, object]:
    return {
        "chatId": WA_CHAT,
        "senderId": WA_CHAT,
        "senderNumber": WA_SESSION,
        "messageId": message_id,
        "_pilotageClaimId": claim_digit * 64,
        "body": body,
        "isGroup": False,
    }


def _tg_message(text: str, message_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(id=42, type="private", is_forum=False),
        from_user=SimpleNamespace(id=42, username="owner", full_name="Owner"),
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        reply_to_message=None,
        quote=None,
        message_id=message_id,
        message_thread_id=None,
        is_topic_message=False,
        photo=[],
        voice=None,
        audio=None,
        video=None,
        sticker=None,
        document=None,
        venue=None,
        location=None,
    )


def _tg_update(update_id: int, text: str) -> object:
    return telegram.Update.de_json(
        {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "date": 0,
                "chat": {"id": 42, "type": "private"},
                "from": {
                    "id": 42,
                    "is_bot": False,
                    "first_name": "Owner",
                },
                "text": text,
            },
        },
        None,
    )


def _tg_inbound(
    text: str,
    message_id: int,
    *,
    claim_id: str = "",
) -> telegram.InboundMessage:
    return telegram.InboundMessage(
        chat_id="42",
        session_id=TG_SESSION,
        user_id="42",
        user_name="Owner",
        text=text,
        is_group=False,
        message_ids=[str(message_id)],
        claim_ids=[claim_id] if claim_id else [],
    )


class ChannelControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_bypasses_slow_effort_checks_without_releasing_followup_early(self):
        for platform in ("whatsapp", "telegram"):
            for queued_new, last_completion in (
                (False, "effort"), (True, "effort"),
                (False, "stop"), (True, "stop"),
            ):
                with self.subTest(platform=platform, queued_new=queued_new, last=last_completion):
                    active = asyncio.Event()
                    stopped = asyncio.Event()
                    selecting = asyncio.Event()
                    selected = asyncio.Event()
                    release_selection = asyncio.Event()
                    release_stop_reply = asyncio.Event()
                    if last_completion == "effort":
                        release_stop_reply.set()
                    followup = asyncio.Event()

                    async def handler(message):
                        if message.text == "active":
                            active.set()
                            await stopped.wait()
                        else:
                            self.assertTrue(selected.is_set())
                            followup.set()

                    async def command(*args):
                        invocation = args[3] if platform == "whatsapp" else args[4]
                        if invocation.command.name == "effort":
                            selecting.set()
                            await release_selection.wait()
                            selected.set()
                        elif invocation.command.name == "stop":
                            stopped.set()
                            await release_stop_reply.wait()
                        else:
                            self.assertEqual(invocation.command.name, "new")
                            self.assertTrue(selected.is_set())

                    if platform == "whatsapp":
                        channel = self._whatsapp(handler, command)
                        channel._queue_turn(_wa_message("active", "m1"))
                    else:
                        channel = self._telegram(handler, command)
                        channel._queue_turn(_tg_inbound("active", 1))
                    self.addAsyncCleanup(channel.stop)

                    async def accept(text, number):
                        if platform == "whatsapp":
                            channel._accept(_wa_event(f"m{number}", text, str(number)))
                        else:
                            await channel._accept_message(_tg_message(text, number), [])

                    try:
                        await asyncio.wait_for(active.wait(), 1)
                        await accept("/effort max", 2)
                        await asyncio.wait_for(selecting.wait(), 1)
                        if queued_new:
                            await accept("/new", 3)
                        await accept("/stop", 4)
                        await asyncio.wait_for(stopped.wait(), 0.5)
                        if last_completion == "stop":
                            release_selection.set()
                            await asyncio.wait_for(selected.wait(), 1)
                        if platform == "whatsapp":
                            channel._queue_turn(_wa_message("followup", "m5"))
                        else:
                            channel._queue_turn(_tg_inbound("followup", 5))
                        await asyncio.sleep(0)
                        await asyncio.sleep(0)
                        self.assertFalse(followup.is_set())
                    finally:
                        stopped.set()
                        release_selection.set()
                        release_stop_reply.set()
                    await asyncio.wait_for(followup.wait(), 1)
                    await channel.stop()

    async def test_shutdown_during_effort_and_stop_retires_all_gates(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                selecting = asyncio.Event()
                stopped = asyncio.Event()
                followup = mock.AsyncMock()

                async def command(*args):
                    invocation = args[3] if platform == "whatsapp" else args[4]
                    if invocation.command.name == "effort":
                        selecting.set()
                        await asyncio.Event().wait()
                    else:
                        stopped.set()

                if platform == "whatsapp":
                    channel = self._whatsapp(followup, command)
                    channel._accept(_wa_event("m1", "/effort max", "a"))
                else:
                    channel = self._telegram(followup, command)
                    await channel._accept_message(_tg_message("/effort max", 1), [])
                self.addAsyncCleanup(channel.stop)
                await asyncio.wait_for(selecting.wait(), 1)
                if platform == "whatsapp":
                    channel._accept(_wa_event("m2", "/stop", "b"))
                    channel._queue_turn(_wa_message("followup", "m3"))
                    tasks = set(channel._pending_tasks_background)
                else:
                    await channel._accept_message(_tg_message("/stop", 2), [])
                    channel._queue_turn(_tg_inbound("followup", 3))
                    tasks = set(channel._background_tasks)
                await asyncio.wait_for(stopped.wait(), 1)
                await channel.stop()
                followup.assert_not_awaited()
                self.assertEqual(channel._control_tasks, {})
                self.assertEqual(channel._effort_tasks, {})
                self.assertTrue(all(task.done() for task in tasks))

    async def test_effort_preserves_pending_input_and_orders_followup_on_both_channels(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                selecting = asyncio.Event()
                release = asyncio.Event()
                answered = asyncio.Event()
                seen = []

                async def handler(message):
                    seen.append(message.text)
                    answered.set()

                async def command(*args):
                    invocation = args[3] if platform == "whatsapp" else args[4]
                    self.assertEqual(invocation.command.name, "effort")
                    selecting.set()
                    await release.wait()

                if platform == "whatsapp":
                    channel = self._whatsapp(handler, command)
                    channel._accept(_wa_event("m1", "pending question", "a"))
                    pending = dict(channel._pending)
                    channel._accept(_wa_event("m2", "/effort high", "b"))
                else:
                    channel = self._telegram(handler, command)
                    await channel._accept_message(_tg_message("pending question", 1), [])
                    pending = dict(channel._pending)
                    await channel._accept_message(_tg_message("/effort high", 2), [])
                self.addAsyncCleanup(channel.stop)
                try:
                    await asyncio.wait_for(selecting.wait(), 1)
                    self.assertTrue(pending)
                    self.assertEqual(channel._pending, pending)
                    if platform == "whatsapp":
                        channel._queue_turn(_wa_message("followup", "m3"))
                    else:
                        channel._queue_turn(_tg_inbound("followup", 3))
                    await asyncio.sleep(0)
                    self.assertFalse(answered.is_set())
                finally:
                    release.set()
                await asyncio.wait_for(answered.wait(), 1)
                self.assertEqual(seen[0], "followup")
                await channel.stop()

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(
            os.environ,
            {
                "PILOTAGE_HOME": str(self.root),
                "TELEGRAM_BOT_TOKEN": "123456:test-token",
                "TELEGRAM_ALLOWED_USERS": "42",
                "TELEGRAM_WEBHOOK_URL": "",
                "TELEGRAM_WEBHOOK_SECRET": "",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _whatsapp(self, handler, command) -> WhatsAppChannel:
        config = Config.load(channel="whatsapp")
        object.__setattr__(config, "allowed_senders", frozenset({WA_SESSION}))
        object.__setattr__(config, "text_batch_delay_seconds", 60.0)
        object.__setattr__(config, "text_batch_hard_cap_seconds", 60.0)
        channel = WhatsAppChannel(config, handler, command)
        channel._mark_claims_completed = mock.Mock()
        channel._settle_claims = mock.AsyncMock()
        channel._ack_later = mock.Mock()
        return channel

    def _telegram(self, handler, command) -> telegram.TelegramChannel:
        config = Config.load(channel="telegram")
        channel = telegram.TelegramChannel(config, handler, command)
        channel._bot = SimpleNamespace(id=999, username="pilotage_bot")
        channel._bot_username = "pilotage_bot"
        channel._complete_claims = mock.AsyncMock()
        return channel

    async def test_whatsapp_control_registers_after_older_turn_and_fences_followup(
        self,
    ) -> None:
        older_registered = asyncio.Event()
        release_older = asyncio.Event()
        control_started = asyncio.Event()
        release_control = asyncio.Event()
        followup_seen = asyncio.Event()

        async def handler(message: WhatsAppInboundMessage) -> None:
            if message.text == "older":
                older_registered.set()
                await release_older.wait()
            else:
                followup_seen.set()

        async def command(_chat, _session, _message, invocation, _claim) -> None:
            self.assertEqual(invocation.command.name, "stop")
            self.assertTrue(older_registered.is_set())
            control_started.set()
            await release_control.wait()

        channel = self._whatsapp(handler, command)
        self.addAsyncCleanup(channel.stop)

        # Models a bridge poll that publishes a turn and /stop synchronously.
        channel._queue_turn(_wa_message("older", "m1"))
        channel._accept(_wa_event("m2", "/stop", "a"))
        await asyncio.wait_for(control_started.wait(), timeout=0.5)

        channel._queue_turn(_wa_message("followup", "m3"))
        release_older.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertFalse(followup_seen.is_set())

        release_control.set()
        await asyncio.wait_for(followup_seen.wait(), timeout=0.5)

    async def test_telegram_control_registers_after_older_turn_and_fences_followup(
        self,
    ) -> None:
        older_registered = asyncio.Event()
        release_older = asyncio.Event()
        control_started = asyncio.Event()
        release_control = asyncio.Event()
        followup_seen = asyncio.Event()

        async def handler(message: telegram.InboundMessage) -> None:
            if message.text == "older":
                older_registered.set()
                await release_older.wait()
            else:
                followup_seen.set()

        async def command(
            _chat, _session, _message, _thread, invocation, _claim
        ) -> None:
            self.assertEqual(invocation.command.name, "stop")
            self.assertTrue(older_registered.is_set())
            control_started.set()
            await release_control.wait()

        channel = self._telegram(handler, command)
        self.addAsyncCleanup(channel.stop)

        channel._queue_turn(_tg_inbound("older", 1))
        await channel._accept_message(_tg_message("/stop", 2), [])
        await asyncio.wait_for(control_started.wait(), timeout=0.5)

        channel._queue_turn(_tg_inbound("followup", 3))
        release_older.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertFalse(followup_seen.is_set())

        release_control.set()
        await asyncio.wait_for(followup_seen.wait(), timeout=0.5)

    async def test_telegram_control_fences_followup_before_dropped_claim_retirement(
        self,
    ) -> None:
        active_started = asyncio.Event()
        release_active = asyncio.Event()
        retirement_started = asyncio.Event()
        release_retirement = asyncio.Event()
        stop_seen = asyncio.Event()
        followup_seen = asyncio.Event()
        dropped_claim = "b" * 64

        async def handler(message: telegram.InboundMessage) -> None:
            if message.text == "active":
                active_started.set()
                await release_active.wait()
            else:
                followup_seen.set()

        async def command(
            _chat, _session, _message, _thread, invocation, _claim
        ) -> None:
            self.assertEqual(invocation.command.name, "stop")
            stop_seen.set()

        async def complete_claims(claim_ids) -> None:
            if list(claim_ids) == [dropped_claim]:
                retirement_started.set()
                await release_retirement.wait()

        channel = self._telegram(handler, command)
        self.addAsyncCleanup(channel.stop)
        channel._complete_claims = complete_claims
        channel._queue_turn(_tg_inbound("active", 1))
        await asyncio.wait_for(active_started.wait(), timeout=0.5)
        channel._queue_turn(
            _tg_inbound("superseded", 2, claim_id=dropped_claim)
        )

        await channel._accept_message(
            _tg_message("/stop", 3),
            [],
            claim_ids=["c" * 64],
        )
        await asyncio.wait_for(retirement_started.wait(), timeout=0.5)
        channel._queue_turn(_tg_inbound("followup", 4))
        release_active.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertFalse(followup_seen.is_set())
        self.assertFalse(stop_seen.is_set())

        release_retirement.set()
        await asyncio.wait_for(stop_seen.wait(), timeout=0.5)
        await asyncio.wait_for(followup_seen.wait(), timeout=0.5)

    async def test_whatsapp_only_bare_stop_drops_pending_and_queued_claims(
        self,
    ) -> None:
        active_started = asyncio.Event()
        release_active = asyncio.Event()
        commands: list[tuple[str, str]] = []
        command_seen = asyncio.Event()

        async def handler(_message: WhatsAppInboundMessage) -> None:
            active_started.set()
            await release_active.wait()

        async def command(_chat, _session, _message, invocation, _claim) -> None:
            commands.append((invocation.command.name, invocation.arguments))
            command_seen.set()

        channel = self._whatsapp(handler, command)
        self.addAsyncCleanup(channel.stop)
        channel._queue_turn(_wa_message("active", "m1"))
        await asyncio.wait_for(active_started.wait(), timeout=0.5)
        channel._queue_turn(_wa_message("queued", "m2", claim_id="b" * 64))
        channel._enqueue(_wa_message("pending", "m3", claim_id="c" * 64))

        channel._accept(_wa_event("m4", "/stop later", "d"))
        await asyncio.wait_for(command_seen.wait(), timeout=0.5)
        self.assertIn(WA_SESSION, channel._pending)
        self.assertIn(WA_SESSION, channel._queued)
        channel._ack_later.assert_not_called()

        command_seen.clear()
        channel._accept(_wa_event("m5", "  /STOP  ", "e"))
        await asyncio.wait_for(command_seen.wait(), timeout=0.5)
        self.assertNotIn(WA_SESSION, channel._pending)
        self.assertNotIn(WA_SESSION, channel._queued)
        channel._ack_later.assert_called_once_with(["c" * 64, "b" * 64])
        self.assertEqual(commands, [("stop", "later"), ("stop", "")])
        release_active.set()

    async def test_telegram_only_bare_stop_drops_pending_and_queued_claims(
        self,
    ) -> None:
        active_started = asyncio.Event()
        release_active = asyncio.Event()
        commands: list[tuple[str, str]] = []
        command_seen = asyncio.Event()

        async def handler(_message: telegram.InboundMessage) -> None:
            active_started.set()
            await release_active.wait()

        async def command(
            _chat, _session, _message, _thread, invocation, _claim
        ) -> None:
            commands.append((invocation.command.name, invocation.arguments))
            command_seen.set()

        channel = self._telegram(handler, command)
        self.addAsyncCleanup(channel.stop)
        channel._queue_turn(_tg_inbound("active", 1))
        await asyncio.wait_for(active_started.wait(), timeout=0.5)
        channel._queue_turn(_tg_inbound("queued", 2, claim_id="b" * 64))
        channel._enqueue(_tg_inbound("pending", 3, claim_id="c" * 64))

        await channel._accept_message(
            _tg_message("/stop later", 4),
            [],
            claim_ids=["d" * 64],
        )
        await asyncio.wait_for(command_seen.wait(), timeout=0.5)
        self.assertIn(TG_SESSION, channel._pending)
        self.assertIn(TG_SESSION, channel._queued)

        command_seen.clear()
        await channel._accept_message(
            _tg_message("  /STOP  ", 5),
            [],
            claim_ids=["e" * 64],
        )
        await asyncio.wait_for(command_seen.wait(), timeout=0.5)
        self.assertNotIn(TG_SESSION, channel._pending)
        self.assertNotIn(TG_SESSION, channel._queued)
        self.assertIn(
            mock.call(["c" * 64, "b" * 64]),
            channel._complete_claims.await_args_list,
        )
        self.assertEqual(commands, [("stop", "later"), ("stop", "")])
        release_active.set()

    async def test_whatsapp_controls_serialize_before_followup(
        self,
    ) -> None:
        new_started = asyncio.Event()
        release_new = asyncio.Event()
        stop_started = asyncio.Event()
        release_stop = asyncio.Event()
        followup_seen = asyncio.Event()

        async def handler(_message: WhatsAppInboundMessage) -> None:
            followup_seen.set()

        async def command(_chat, _session, _message, invocation, _claim) -> None:
            if invocation.command.name == "new":
                new_started.set()
                await release_new.wait()
            else:
                stop_started.set()
                await release_stop.wait()

        channel = self._whatsapp(handler, command)
        self.addAsyncCleanup(channel.stop)
        channel._accept(_wa_event("m1", "/new", "a"))
        await asyncio.wait_for(new_started.wait(), timeout=0.5)

        channel._accept(_wa_event("m2", "/stop", "b"))
        channel._queue_turn(_wa_message("followup", "m3"))
        await asyncio.sleep(0)
        self.assertFalse(stop_started.is_set())

        release_new.set()
        await asyncio.wait_for(stop_started.wait(), timeout=0.5)
        self.assertFalse(followup_seen.is_set())

        release_stop.set()
        await asyncio.wait_for(followup_seen.wait(), timeout=0.5)

    async def test_telegram_controls_serialize_before_followup(
        self,
    ) -> None:
        new_started = asyncio.Event()
        release_new = asyncio.Event()
        stop_started = asyncio.Event()
        release_stop = asyncio.Event()
        followup_seen = asyncio.Event()

        async def handler(_message: telegram.InboundMessage) -> None:
            followup_seen.set()

        async def command(
            _chat, _session, _message, _thread, invocation, _claim
        ) -> None:
            if invocation.command.name == "new":
                new_started.set()
                await release_new.wait()
            else:
                stop_started.set()
                await release_stop.wait()

        channel = self._telegram(handler, command)
        self.addAsyncCleanup(channel.stop)
        await channel._accept_message(_tg_message("/new", 1), [])
        await asyncio.wait_for(new_started.wait(), timeout=0.5)

        await channel._accept_message(_tg_message("/stop", 2), [])
        channel._queue_turn(_tg_inbound("followup", 3))
        await asyncio.sleep(0)
        self.assertFalse(stop_started.is_set())

        release_new.set()
        await asyncio.wait_for(stop_started.wait(), timeout=0.5)
        self.assertFalse(followup_seen.is_set())

        release_stop.set()
        await asyncio.wait_for(followup_seen.wait(), timeout=0.5)

    async def test_whatsapp_startup_stop_waits_for_recovery_release(self) -> None:
        stop_seen = asyncio.Event()

        async def handler(_message: WhatsAppInboundMessage) -> None:
            return None

        async def command(_chat, _session, _message, invocation, _claim) -> None:
            self.assertEqual(invocation.command.name, "stop")
            stop_seen.set()

        channel = self._whatsapp(handler, command)
        self.addAsyncCleanup(channel.stop)
        channel.hold_inbound()
        channel._accept(_wa_event("m1", "/stop", "a"))

        channel.enable_startup_approvals()
        await asyncio.sleep(0)
        self.assertFalse(stop_seen.is_set())
        self.assertEqual(
            [str(event.get("body") or "") for event in channel._startup_events],
            ["/stop"],
        )

        channel.release_inbound()
        await asyncio.wait_for(stop_seen.wait(), timeout=0.5)

    async def test_telegram_startup_stop_waits_for_recovery_release(self) -> None:
        stop_seen = asyncio.Event()

        async def handler(_message: telegram.InboundMessage) -> None:
            return None

        async def command(
            _chat, _session, _message, _thread, invocation, _claim
        ) -> None:
            self.assertEqual(invocation.command.name, "stop")
            stop_seen.set()

        channel = self._telegram(handler, command)
        self.addAsyncCleanup(channel.stop)
        stop = _tg_update(1, "/stop")
        channel._inbound_store.record(stop)
        channel.hold_inbound()
        await channel._handle_command(stop, None)

        await channel.enable_startup_approvals()
        await asyncio.sleep(0)
        self.assertFalse(stop_seen.is_set())
        self.assertEqual(len(channel._startup_updates), 1)

        await channel.release_inbound()
        await asyncio.wait_for(stop_seen.wait(), timeout=0.5)

    async def test_whatsapp_shutdown_does_not_release_control_fenced_followup(
        self,
    ) -> None:
        control_started = asyncio.Event()
        followup = mock.AsyncMock()

        async def command(_chat, _session, _message, _invocation, _claim) -> None:
            control_started.set()
            await asyncio.Event().wait()

        channel = self._whatsapp(followup, command)
        channel._accept(_wa_event("m1", "/stop", "a"))
        await asyncio.wait_for(control_started.wait(), timeout=0.5)
        channel._queue_turn(_wa_message("followup", "m2"))

        await channel.stop()
        await asyncio.sleep(0)

        followup.assert_not_awaited()

    async def test_telegram_shutdown_does_not_release_control_fenced_followup(
        self,
    ) -> None:
        control_started = asyncio.Event()
        followup = mock.AsyncMock()

        async def command(
            _chat, _session, _message, _thread, _invocation, _claim
        ) -> None:
            control_started.set()
            await asyncio.Event().wait()

        channel = self._telegram(followup, command)
        await channel._accept_message(_tg_message("/stop", 1), [])
        await asyncio.wait_for(control_started.wait(), timeout=0.5)
        channel._queue_turn(_tg_inbound("followup", 2))

        await channel.stop()
        await asyncio.sleep(0)

        followup.assert_not_awaited()
