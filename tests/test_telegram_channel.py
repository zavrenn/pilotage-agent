from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilotage.channels import telegram
from pilotage.config import (
    Config,
    TELEGRAM_DEFAULT_INSTRUCTIONS,
    TELEGRAM_FORMATTING_NOTE,
    TELEGRAM_MEDIA_NOTE,
)
from pilotage.delivery import (
    DeliveryPlanError,
    DeliveryStore,
    DeliveryUnitLedger,
    compute_obligation_id,
)
from pilotage.settings import ConfigError


def _entity(kind: str, text: str, *, offset: int = 0):
    return SimpleNamespace(
        type=kind,
        offset=offset,
        length=len(text),
        user=None,
    )


def _message(
    *,
    user_id: int = 42,
    username: str = "owner",
    chat_id: int = 42,
    chat_type: str = "private",
    text: str = "hello",
    entities=None,
    reply_to=None,
    message_id: int = 1,
    is_forum: bool = False,
    thread_id=None,
    is_topic_message: bool = False,
    **media_fields,
):
    values = {
        "chat": SimpleNamespace(
            id=chat_id,
            type=chat_type,
            is_forum=is_forum,
        ),
        "from_user": SimpleNamespace(
            id=user_id,
            username=username,
            full_name="Owner",
        ),
        "text": text,
        "caption": None,
        "entities": list(entities or []),
        "caption_entities": [],
        "reply_to_message": reply_to,
        "quote": None,
        "message_id": message_id,
        "message_thread_id": thread_id,
        "is_topic_message": is_topic_message,
        "photo": [],
        "voice": None,
        "audio": None,
        "video": None,
        "sticker": None,
        "document": None,
        "venue": None,
        "location": None,
    }
    values.update(media_fields)
    return SimpleNamespace(**values)


class TelegramChannelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
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

    def _write_config(self, content: str) -> None:
        (self.root / "config.yaml").write_text(content, encoding="utf-8")

    def _channel(self, config_text: str = ""):
        if config_text:
            self._write_config(config_text)
        config = Config.load(channel="telegram")
        handler = mock.AsyncMock()
        command = mock.AsyncMock()
        channel = telegram.TelegramChannel(config, handler, command)
        channel._bot = SimpleNamespace(id=999, username="pilotage_bot")
        channel._bot_username = "pilotage_bot"
        return channel, handler, command

    def test_telegram_is_disabled_by_default(self):
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": ""}):
            config = Config.load(channel="telegram")

        self.assertFalse(config.settings.flag("telegram.enabled", False))

    def test_enabled_telegram_requires_a_token(self):
        self._write_config("telegram:\n  enabled: true\n")

        with (
            mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": ""}),
            self.assertRaisesRegex(ConfigError, "TELEGRAM_BOT_TOKEN"),
        ):
            Config.load(channel="telegram")

    def test_enabled_telegram_requires_explicit_allowed_users(self):
        self._write_config("telegram:\n  enabled: true\n")

        with (
            mock.patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}),
            self.assertRaisesRegex(ConfigError, "TELEGRAM_ALLOWED_USERS"),
        ):
            Config.load(channel="telegram")

    def test_telegram_home_must_be_a_numeric_chat(self):
        with (
            mock.patch.dict(os.environ, {"TELEGRAM_HOME_CHANNEL": "@owner"}),
            self.assertRaisesRegex(ConfigError, "non-zero numeric chat ID"),
        ):
            Config.load(channel="telegram")

    def test_telegram_topic_requires_a_group_home(self):
        with (
            mock.patch.dict(
                os.environ,
                {
                    "TELEGRAM_HOME_CHANNEL": "42",
                    "TELEGRAM_HOME_CHANNEL_THREAD_ID": "7",
                },
            ),
            self.assertRaisesRegex(ConfigError, "negative Telegram group"),
        ):
            Config.load(channel="telegram")

    def test_secrets_wildcard_users_and_usernames_are_rejected(self):
        self._write_config("telegram:\n  bot_token: secret\n")
        with self.assertRaisesRegex(ConfigError, "secret"):
            Config.load()

        self._write_config("")
        with (
            mock.patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}),
            self.assertRaisesRegex(ConfigError, "explicit users"),
        ):
            Config.load()

        with (
            mock.patch.dict(
                os.environ,
                {"TELEGRAM_ALLOWED_USERS": "42,@backup"},
            ),
            self.assertRaisesRegex(ConfigError, "numeric Telegram user IDs"),
        ):
            Config.load()

    def test_telegram_channel_gets_its_own_prompt_notes(self):
        self._write_config("telegram:\n  enabled: true\n")

        config = Config.load(channel="telegram")

        self.assertIn(TELEGRAM_DEFAULT_INSTRUCTIONS, config.instructions)
        self.assertIn(TELEGRAM_FORMATTING_NOTE, config.instructions)
        self.assertIn(TELEGRAM_MEDIA_NOTE, config.instructions)

    def test_missing_optional_dependency_fails_only_at_channel_preflight(self):
        channel, _, _ = self._channel()

        with (
            mock.patch.object(telegram, "TELEGRAM_AVAILABLE", False),
            self.assertRaisesRegex(telegram.ChannelError, "not installed"),
        ):
            channel._preflight()

    def test_dm_authorization_is_fail_closed(self):
        channel, _, _ = self._channel()

        self.assertTrue(channel._authorized(_message()))
        self.assertFalse(
            channel._authorized(_message(user_id=77, username="backup"))
        )
        self.assertFalse(channel._authorized(_message(user_id=77, username="other")))

    def test_group_location_settings_are_rejected(self):
        for setting in ("group_policy: disabled", "group_allow_from: []"):
            with self.subTest(setting=setting):
                with self.assertRaisesRegex(ConfigError, "no longer supported"):
                    self._channel(f"telegram:\n  {setting}\n")

    def test_allowed_user_can_use_any_group_with_direct_bot_trigger(self):
        channel, _, _ = self._channel(
            "telegram:\n"
            "  require_mention: true\n"
        )
        mention = "@pilotage_bot"
        addressed = _message(
            chat_id=-100,
            chat_type="supergroup",
            text=f"{mention} hello",
            entities=[_entity("mention", mention)],
        )
        unmentioned = _message(
            chat_id=-100,
            chat_type="supergroup",
            text="hello",
        )
        foreign = "@another_bot"
        addressed_elsewhere = _message(
            chat_id=-100,
            chat_type="supergroup",
            text=f"{foreign} hello",
            entities=[_entity("mention", foreign)],
        )
        reply = SimpleNamespace(from_user=SimpleNamespace(id=999), text="earlier")
        replying = _message(
            chat_id=-100,
            chat_type="supergroup",
            text="hello",
            reply_to=reply,
        )
        other_member = _message(
            user_id=77,
            chat_id=-100,
            chat_type="supergroup",
            text=f"{mention} hello",
            entities=[_entity("mention", mention)],
        )

        self.assertTrue(channel._authorized(addressed))
        self.assertFalse(channel._authorized(other_member))
        self.assertFalse(channel._authorized(unmentioned))
        self.assertFalse(channel._authorized(addressed_elsewhere))
        self.assertTrue(channel._authorized(replying))
        self.assertTrue(
            channel._authorized(
                _message(
                    chat_id=-200,
                    chat_type="supergroup",
                    text=f"{mention} hello",
                    entities=[_entity("mention", mention)],
                )
            )
        )

    def test_group_authorization_is_based_only_on_the_user(self):
        channel, _, _ = self._channel(
            "telegram:\n"
            "  require_mention: false\n"
        )

        self.assertTrue(
            channel._authorized(
                _message(
                    user_id=42,
                    chat_id=-200,
                    chat_type="supergroup",
                )
            )
        )
        self.assertFalse(
            channel._authorized(
                _message(
                    user_id=77,
                    chat_id=-200,
                    chat_type="supergroup",
                )
            )
        )
        self.assertFalse(channel._authorized(_message(user_id=77)))

    def test_targeted_group_command_is_cleaned_for_management_registry(self):
        channel, _, _ = self._channel(
            "telegram:\n"
            "  require_mention: true\n"
        )
        command = "/new@pilotage_bot"
        message = _message(
            chat_id=-100,
            chat_type="supergroup",
            text=command,
            entities=[_entity("bot_command", command)],
        )

        self.assertTrue(channel._authorized(message))
        self.assertEqual(channel._clean_routing_mention(command), "/new")

    def test_entity_offsets_are_utf16_units(self):
        source = "😀 @pilotage_bot"

        self.assertEqual(
            telegram._telegram_entity_text(source, 3, len("@pilotage_bot")),
            "@pilotage_bot",
        )

    def test_session_boundaries_separate_group_participants(self):
        self.assertEqual(
            telegram._session_id("42", "42", False, ""),
            "telegram:dm:42",
        )
        self.assertNotEqual(
            telegram._session_id("-100", "42", True, ""),
            telegram._session_id("-100", "43", True, ""),
        )
        self.assertNotEqual(
            telegram._session_id("-100", "42", True, "9"),
            telegram._session_id("-100", "43", True, "9"),
        )
        self.assertEqual(
            telegram._session_id("-100", "42", True, "9"),
            "telegram:group:-100:9:42",
        )
        forum = _message(
            chat_id=-100,
            chat_type="supergroup",
            is_forum=True,
            thread_id=None,
        )
        ordinary_reply = _message(thread_id=123, is_topic_message=False)

        self.assertEqual(telegram._effective_thread_id(forum), "1")
        self.assertEqual(telegram._effective_thread_id(ordinary_reply), "")

    async def test_unauthorized_media_is_never_downloaded(self):
        channel, handler, _ = self._channel()
        source = SimpleNamespace(get_file=mock.AsyncMock())
        message = _message(user_id=77, username="other", voice=source)

        await channel._handle_media(
            SimpleNamespace(effective_message=message),
            None,
        )

        source.get_file.assert_not_awaited()
        handler.assert_not_awaited()

    async def test_native_voice_note_downloads_as_ptt_for_existing_stt(self):
        channel, _, _ = self._channel()
        (channel._config.media_dir / "telegram").mkdir(parents=True)
        telegram_file = SimpleNamespace(
            download_as_bytearray=mock.AsyncMock(
                return_value=bytearray(b"voice")
            )
        )
        source = SimpleNamespace(
            file_size=5,
            file_unique_id="voice-id",
            mime_type="audio/ogg",
            get_file=mock.AsyncMock(return_value=telegram_file),
        )
        message = _message(text="", voice=source)

        with mock.patch.object(
            telegram.asyncio,
            "to_thread",
            wraps=asyncio.to_thread,
        ) as offload:
            attachments, notes = await channel._download_attachments(message)

        self.assertEqual(notes, [])
        offload.assert_awaited_once()
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].media_type, "ptt")
        self.assertTrue(attachments[0].is_voice_message)
        self.assertEqual(attachments[0].path.read_bytes(), b"voice")

    async def test_uploaded_audio_does_not_enter_voice_note_stt(self):
        channel, _, _ = self._channel()
        (channel._config.media_dir / "telegram").mkdir(parents=True)
        telegram_file = SimpleNamespace(
            download_as_bytearray=mock.AsyncMock(
                return_value=bytearray(b"audio")
            )
        )
        source = SimpleNamespace(
            file_size=5,
            file_unique_id="audio-id",
            file_name="recording.mp3",
            mime_type="audio/mpeg",
            get_file=mock.AsyncMock(return_value=telegram_file),
        )

        attachments, _ = await channel._download_attachments(
            _message(text="", audio=source)
        )

        self.assertEqual(attachments[0].media_type, "audio")
        self.assertFalse(attachments[0].is_voice_message)

    async def test_text_documents_use_the_shared_inline_media_contract(self):
        channel, _, _ = self._channel()
        (channel._config.media_dir / "telegram").mkdir(parents=True)
        telegram_file = SimpleNamespace(
            download_as_bytearray=mock.AsyncMock(
                return_value=bytearray(b"document text")
            )
        )
        source = SimpleNamespace(
            file_size=13,
            file_unique_id="document-id",
            file_name="notes.txt",
            mime_type="text/plain",
            get_file=mock.AsyncMock(return_value=telegram_file),
        )
        message = _message(text="question", document=source)

        attachments, notes = await channel._download_attachments(message)
        composed = channel._compose_text(
            message,
            "question",
            attachments,
            notes,
            is_group=False,
            user_id="42",
        )

        self.assertIn("[Content of notes.txt]:", composed)
        self.assertIn("document text", composed)
        self.assertIn("question", composed)

    async def test_binary_document_path_is_exposed_to_the_agent(self):
        channel, _, _ = self._channel()
        (channel._config.media_dir / "telegram").mkdir(parents=True)
        telegram_file = SimpleNamespace(
            download_as_bytearray=mock.AsyncMock(
                return_value=bytearray(b"%PDF")
            )
        )
        source = SimpleNamespace(
            file_size=4,
            file_unique_id="pdf-id",
            file_name="report.pdf",
            mime_type="application/pdf",
            get_file=mock.AsyncMock(return_value=telegram_file),
        )
        message = _message(text="summarize it", document=source)

        attachments, notes = await channel._download_attachments(message)
        composed = channel._compose_text(
            message,
            "summarize it",
            attachments,
            notes,
            is_group=False,
            user_id="42",
        )

        self.assertIn("report.pdf", composed)
        self.assertIn(str(attachments[0].path.resolve()), composed)
        self.assertIn("document-reading skill", composed)
        self.assertNotIn("cannot read", composed)

    async def test_rapid_text_messages_become_one_turn(self):
        channel, handler, _ = self._channel(
            "telegram:\n"
            "  batch_delay: 0\n"
            "  batch_hard_cap: 1\n"
        )
        await channel._accept_message(_message(text="one", message_id=1), [])
        await channel._accept_message(_message(text="two", message_id=2), [])
        await asyncio.sleep(0.05)

        handler.assert_awaited_once()
        inbound = handler.await_args.args[0]
        self.assertEqual(inbound.text, "one\ntwo")
        await channel.stop()

    async def test_text_waits_for_same_session_media_download(self):
        delivered = []
        handled = asyncio.Event()
        download_started = asyncio.Event()
        release_download = asyncio.Event()
        channel, _, _ = self._channel(
            "telegram:\n"
            "  batch_delay: 0.01\n"
            "  media_batch_delay: 0.01\n"
            "  batch_hard_cap: 1\n"
        )
        self.addAsyncCleanup(channel.stop)

        async def handler(message):
            delivered.append(message)
            handled.set()

        async def download(_message):
            download_started.set()
            await release_download.wait()
            return [
                telegram.media.Attachment(
                    path=self.root / "photo.jpg",
                    mime="image/jpeg",
                    media_type="image",
                    file_name="photo.jpg",
                )
            ], []

        channel._handler = handler
        channel._download_attachments = download
        await channel._accept_message(_message(text="question", message_id=1), [])
        media_task = asyncio.create_task(
            channel._handle_media(
                SimpleNamespace(
                    effective_message=_message(
                        text="",
                        message_id=2,
                        voice=SimpleNamespace(),
                    )
                ),
                None,
            )
        )
        await asyncio.wait_for(download_started.wait(), timeout=0.5)
        await asyncio.sleep(0.05)

        self.assertFalse(handled.is_set())
        release_download.set()
        await asyncio.wait_for(media_task, timeout=0.5)
        await asyncio.wait_for(handled.wait(), timeout=0.5)

        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].text, "question")
        self.assertEqual(delivered[0].message_ids, ["1", "2"])
        self.assertEqual(len(delivered[0].attachments), 1)

    async def test_stuck_media_download_cannot_exceed_batch_hard_cap(self):
        handled = asyncio.Event()
        channel, _, _ = self._channel(
            "telegram:\n"
            "  batch_delay: 0\n"
            "  batch_hard_cap: 0.05\n"
        )
        self.addAsyncCleanup(channel.stop)
        channel._handler = mock.AsyncMock(side_effect=lambda _message: handled.set())
        session_id = telegram._session_id("42", "42", False, "")
        channel._begin_media_download(session_id)

        await channel._accept_message(_message(text="question", message_id=1), [])
        await asyncio.wait_for(handled.wait(), timeout=0.3)

        channel._end_media_download(session_id)

    async def test_management_commands_bypass_the_model_batch(self):
        channel, handler, command_handler = self._channel()
        await channel._accept_message(_message(text="/new"), [])
        await asyncio.sleep(0)

        handler.assert_not_awaited()
        command_handler.assert_awaited_once()
        invocation = command_handler.await_args.args[-1]
        self.assertEqual(invocation.command.name, "new")
        await channel.stop()

    async def test_startup_gate_holds_authorized_command_and_media_until_release(self):
        command_started = asyncio.Event()
        media_download_started = asyncio.Event()
        release_media = asyncio.Event()
        media_delivered = asyncio.Event()
        channel, _, _ = self._channel(
            "telegram:\n"
            "  batch_delay: 0\n"
            "  batch_hard_cap: 1\n"
        )
        self.addAsyncCleanup(channel.stop)

        async def handler(message):
            self.assertEqual(message.message_ids, ["12"])
            media_delivered.set()

        async def command(_chat, _session, _message, _thread, invocation):
            self.assertEqual(invocation.command.name, "approve")
            command_started.set()

        async def download(message):
            self.assertEqual(getattr(message, "message_id", None), 12)
            media_download_started.set()
            await release_media.wait()
            return [], []

        channel._handler = handler
        channel._on_command = command
        channel._download_attachments = download
        channel.hold_inbound()

        await channel._handle_media(
            SimpleNamespace(
                effective_message=_message(
                    text="photo",
                    message_id=12,
                    voice=SimpleNamespace(),
                )
            ),
            None,
        )
        await channel._handle_command(
            SimpleNamespace(
                effective_message=_message(text="/approve", message_id=11)
            ),
            None,
        )
        await asyncio.sleep(0)
        self.assertFalse(command_started.is_set())
        self.assertFalse(media_download_started.is_set())

        releasing = asyncio.create_task(channel.release_inbound())
        await asyncio.wait_for(media_download_started.wait(), timeout=0.5)
        await asyncio.sleep(0)
        self.assertFalse(command_started.is_set())
        release_media.set()
        await asyncio.wait_for(media_delivered.wait(), timeout=0.5)
        await asyncio.wait_for(command_started.wait(), timeout=0.5)
        await asyncio.wait_for(releasing, timeout=0.5)
        await channel.stop()

    async def test_approval_command_bypasses_an_active_turn(self):
        channel, _, _ = self._channel(
            "telegram:\n  batch_delay: 0\n  batch_hard_cap: 1\n"
        )
        started = asyncio.Event()
        release = asyncio.Event()
        command_seen = asyncio.Event()

        async def handler(_message):
            started.set()
            await release.wait()

        async def command(_chat, _session, _message, _thread, invocation):
            self.assertEqual(invocation.command.name, "approve")
            command_seen.set()

        channel._handler = handler
        channel._on_command = command
        await channel._accept_message(_message(text="working", message_id=1), [])
        await asyncio.wait_for(started.wait(), timeout=0.5)

        await channel._accept_message(_message(text="/approve", message_id=2), [])
        await asyncio.wait_for(command_seen.wait(), timeout=0.5)

        self.assertTrue(any(not task.done() for task in channel._turn_tasks.values()))
        release.set()
        await channel.stop()

    async def test_stop_drains_every_message_accepted_before_intake_closes(self):
        delivered = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(message):
            delivered.append(message.text)
            if len(delivered) == 1:
                started.set()
                await release.wait()

        channel, _, _ = self._channel()
        channel._handler = handler

        def inbound(text, message_id):
            return telegram.InboundMessage(
                chat_id="42",
                session_id="telegram:dm:42",
                user_id="42",
                user_name="Owner",
                text=text,
                is_group=False,
                message_ids=[str(message_id)],
            )

        channel._queue_turn(inbound("one", 1))
        await asyncio.wait_for(started.wait(), timeout=0.5)
        channel._queue_turn(inbound("two", 2))
        channel._enqueue(inbound("three", 3))

        stopping = asyncio.create_task(
            channel.stop(drain_timeout_seconds=0.5)
        )
        await asyncio.sleep(0)
        self.assertFalse(stopping.done())
        release.set()
        await asyncio.wait_for(stopping, timeout=1)

        self.assertEqual(delivered, ["one", "two\nthree"])
        self.assertEqual(channel._turn_tasks, {})

    async def test_shutdown_holds_and_drains_an_in_progress_media_update(self):
        delivered = []
        download_started = asyncio.Event()
        release_download = asyncio.Event()

        async def download(_message):
            download_started.set()
            await release_download.wait()
            return [], []

        async def handler(message):
            delivered.append(message.text)

        channel, _, _ = self._channel()
        channel._handler = handler
        channel._download_attachments = download
        channel._app = SimpleNamespace(
            updater=SimpleNamespace(running=True, stop=mock.AsyncMock()),
            running=True,
            stop=mock.AsyncMock(),
            shutdown=mock.AsyncMock(),
        )

        update = SimpleNamespace(effective_message=_message(text="late media"))
        handling = asyncio.create_task(channel._handle_media(update, None))
        await asyncio.wait_for(download_started.wait(), timeout=0.5)

        stopping = asyncio.create_task(
            channel.stop(drain_timeout_seconds=0.5)
        )
        await asyncio.sleep(0)
        release_download.set()
        await asyncio.wait_for(stopping, timeout=1)
        await handling

        self.assertEqual(delivered, ["late media"])
        self.assertEqual(channel._held_inbound, {})

    async def test_markdown_reply_and_topic_are_sent_natively(self):
        channel, _, _ = self._channel(
            "telegram:\n"
            "  reply_to_mode: first\n"
        )
        bot = SimpleNamespace(send_message=mock.AsyncMock())
        channel._bot = bot
        parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")

        with mock.patch.object(telegram, "ParseMode", parse_mode):
            delivered = await channel.send(
                "42",
                "**hello**",
                "7",
                thread_id="9",
            )

        self.assertTrue(delivered)
        kwargs = bot.send_message.await_args.kwargs
        self.assertEqual(kwargs["text"], "*hello*")
        self.assertEqual(kwargs["reply_to_message_id"], 7)
        self.assertEqual(kwargs["message_thread_id"], 9)

    async def test_markdown_parse_error_falls_back_to_plaintext(self):
        channel, _, _ = self._channel()
        bot = SimpleNamespace(
            send_message=mock.AsyncMock(
                side_effect=[
                    RuntimeError("Can't parse entities"),
                    SimpleNamespace(message_id=1),
                ]
            )
        )
        channel._bot = bot
        parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")

        with (
            mock.patch.object(telegram, "ParseMode", parse_mode),
            mock.patch.object(telegram, "BadRequest", RuntimeError),
        ):
            delivered = await channel.send("42", "**hello**")

        self.assertTrue(delivered)
        self.assertEqual(bot.send_message.await_count, 2)
        fallback = bot.send_message.await_args.kwargs
        self.assertEqual(fallback["text"], "hello")
        self.assertIsNone(fallback["parse_mode"])

    async def test_workspace_media_directive_sends_a_native_document(self):
        channel, _, _ = self._channel()
        channel._config.workspace_dir.mkdir(parents=True)
        document = channel._config.workspace_dir / "report.pdf"
        document.write_bytes(b"%PDF")
        bot = SimpleNamespace(
            send_message=mock.AsyncMock(),
            send_document=mock.AsyncMock(),
        )
        channel._bot = bot

        delivered = await channel.send(
            "42",
            f"MEDIA:{document.as_posix()}",
        )

        self.assertTrue(delivered)
        bot.send_document.assert_awaited_once()

    async def test_unit_ledger_retries_only_missing_attachment_with_fresh_handle(self):
        channel, _, _ = self._channel()
        channel._config.workspace_dir.mkdir(parents=True)
        document = channel._config.workspace_dir / "report.pdf"
        document.write_bytes(b"%PDF")
        streams = []

        async def send_document(**kwargs):
            streams.append(kwargs["document"])
            if len(streams) == 1:
                raise telegram.NetworkError("offline")
            return SimpleNamespace(message_id=22)

        bot = SimpleNamespace(
            send_message=mock.AsyncMock(return_value=SimpleNamespace(message_id=21)),
            send_document=send_document,
        )
        channel._bot = bot
        content = f"ready\nMEDIA:{document}"
        obligation_id = compute_obligation_id("session", "message", content)
        store = DeliveryStore(self.root / "delivery-ledger.db")
        store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content=content,
        )
        self.assertTrue(store.mark_attempting(obligation_id))
        ledger = DeliveryUnitLedger(store, obligation_id)

        first = await channel.send("42", content, delivery_ledger=ledger)
        second = await channel.send("42", content, delivery_ledger=ledger)

        self.assertFalse(first)
        self.assertTrue(first.retryable)
        self.assertTrue(second)
        bot.send_message.assert_awaited_once()
        self.assertEqual(len(streams), 2)
        self.assertIsNot(streams[0], streams[1])
        self.assertTrue(all(stream.closed for stream in streams))
        self.assertTrue(store.mark_delivered(obligation_id))

    async def test_unit_ledger_fingerprints_the_effective_reply_mode(self):
        channel, _, _ = self._channel(
            "telegram:\n"
            "  reply_to_mode: all\n"
        )
        channel._bot = SimpleNamespace(
            send_message=mock.AsyncMock(
                return_value=SimpleNamespace(message_id=101)
            )
        )
        content = "reply"
        obligation_id = compute_obligation_id("session", "message", content)
        store = DeliveryStore(self.root / "reply-mode-ledger.db")
        store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            reply_to="7",
            content=content,
        )
        self.assertTrue(store.mark_attempting(obligation_id))
        ledger = DeliveryUnitLedger(store, obligation_id)

        self.assertTrue(
            await channel.send("42", content, "7", delivery_ledger=ledger)
        )
        channel._reply_to_mode = "off"

        with self.assertRaisesRegex(DeliveryPlanError, "plan changed"):
            await channel.send("42", content, "7", delivery_ledger=ledger)

    async def test_system_echo_cannot_turn_media_text_into_delivery(self):
        channel, _, _ = self._channel()
        channel._config.workspace_dir.mkdir(parents=True)
        document = channel._config.workspace_dir / "report.pdf"
        document.write_bytes(b"%PDF")
        bot = SimpleNamespace(
            send_message=mock.AsyncMock(),
            send_document=mock.AsyncMock(),
        )
        channel._bot = bot
        parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")

        with mock.patch.object(telegram, "ParseMode", parse_mode):
            delivered = await channel.send(
                "42",
                f"MEDIA:{document.as_posix()}",
                deliver_media=False,
            )

        self.assertTrue(delivered)
        bot.send_message.assert_awaited_once()
        bot.send_document.assert_not_awaited()


class TelegramLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_polling_startup_preserves_pending_updates_and_shutdown_is_owned(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ,
                {
                    "PILOTAGE_HOME": temporary,
                    "TELEGRAM_BOT_TOKEN": "123456:test-token",
                    "TELEGRAM_ALLOWED_USERS": "42",
                    "TELEGRAM_WEBHOOK_URL": "",
                    "TELEGRAM_WEBHOOK_SECRET": "",
                },
            ):
                config = Config.load(channel="telegram")
                channel = telegram.TelegramChannel(
                    config,
                    mock.AsyncMock(),
                    mock.AsyncMock(),
                )
                bot = SimpleNamespace(
                    id=999,
                    username="pilotage_bot",
                    delete_webhook=mock.AsyncMock(),
                )
                updater = SimpleNamespace(
                    running=True,
                    start_polling=mock.AsyncMock(),
                    stop=mock.AsyncMock(),
                )
                app = SimpleNamespace(
                    bot=bot,
                    updater=updater,
                    running=True,
                    initialize=mock.AsyncMock(),
                    start=mock.AsyncMock(),
                    stop=mock.AsyncMock(),
                    shutdown=mock.AsyncMock(),
                )

                with (
                    mock.patch.object(telegram, "TELEGRAM_AVAILABLE", True),
                    mock.patch.object(
                        channel,
                        "_build_application",
                        return_value=app,
                    ),
                ):
                    await channel.start()
                    await channel.stop()

        app.initialize.assert_awaited_once()
        bot.delete_webhook.assert_awaited_once_with(
            drop_pending_updates=False
        )
        updater.start_polling.assert_awaited_once()
        self.assertFalse(
            updater.start_polling.await_args.kwargs[
                "drop_pending_updates"
            ]
        )
        updater.stop.assert_awaited_once()
        app.stop.assert_awaited_once()
        app.shutdown.assert_awaited_once()

    async def test_failed_start_drains_updates_already_accepted_by_polling(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ,
                {
                    "PILOTAGE_HOME": temporary,
                    "TELEGRAM_BOT_TOKEN": "123456:test-token",
                    "TELEGRAM_ALLOWED_USERS": "42",
                    "TELEGRAM_WEBHOOK_URL": "",
                    "TELEGRAM_WEBHOOK_SECRET": "",
                },
            ):
                channel = telegram.TelegramChannel(
                    Config.load(channel="telegram"),
                    mock.AsyncMock(),
                    mock.AsyncMock(),
                )

        completed = asyncio.Event()
        cancelled = asyncio.Event()

        async def accepted_work():
            try:
                await asyncio.sleep(0)
                completed.set()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def replay(_kind, _update, _context):
            channel._turn_tasks["held"] = asyncio.create_task(accepted_work())

        async def fail_after_accepting(**_kwargs):
            channel._startup_updates.append(("text", object(), object()))
            raise RuntimeError("polling startup failed")

        bot = SimpleNamespace(
            username="pilotage_bot",
            delete_webhook=mock.AsyncMock(),
        )
        updater = SimpleNamespace(
            running=True,
            start_polling=mock.AsyncMock(side_effect=fail_after_accepting),
            stop=mock.AsyncMock(),
        )
        app = SimpleNamespace(
            bot=bot,
            updater=updater,
            running=True,
            initialize=mock.AsyncMock(),
            start=mock.AsyncMock(),
            stop=mock.AsyncMock(),
            shutdown=mock.AsyncMock(),
        )
        channel.hold_inbound()

        with (
            mock.patch.object(telegram, "TELEGRAM_AVAILABLE", True),
            mock.patch.object(channel, "_build_application", return_value=app),
            mock.patch.object(channel, "_replay_startup_update", side_effect=replay),
            self.assertRaisesRegex(telegram.ChannelError, "could not start"),
        ):
            await channel.start()

        self.assertTrue(completed.is_set())
        self.assertFalse(cancelled.is_set())

    async def test_polling_conflict_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ,
                {
                    "PILOTAGE_HOME": temporary,
                    "TELEGRAM_BOT_TOKEN": "123456:test-token",
                    "TELEGRAM_ALLOWED_USERS": "42",
                    "TELEGRAM_WEBHOOK_URL": "",
                    "TELEGRAM_WEBHOOK_SECRET": "",
                },
            ):
                channel = telegram.TelegramChannel(
                    Config.load(channel="telegram"),
                    mock.AsyncMock(),
                    mock.AsyncMock(),
                )

        class FakeConflict(Exception):
            pass

        with mock.patch.object(telegram, "Conflict", FakeConflict):
            channel._polling_error_callback(
                FakeConflict("terminated by another getUpdates")
            )

        self.assertTrue(channel.stopped.is_set())
        self.assertIn("another process", channel.failure)

    async def test_update_errors_redact_bot_token(self):
        secret = "123456:test-token"
        channel = telegram.TelegramChannel.__new__(telegram.TelegramChannel)
        channel._token = secret
        context = SimpleNamespace(
            error=RuntimeError(f"https://api.telegram.org/bot{secret}/getMe")
        )

        with self.assertLogs("pilotage.channels.telegram", level="ERROR") as logs:
            await channel._handle_update_error(None, context)

        output = "\n".join(logs.output)
        self.assertNotIn(secret, output)
        self.assertIn("<redacted Telegram token>", output)



class TelegramWebhookTests(unittest.IsolatedAsyncioTestCase):
    def test_enabled_webhook_requires_a_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "config.yaml").write_text(
                "telegram:\n  enabled: true\n",
                encoding="utf-8",
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "PILOTAGE_HOME": temporary,
                        "TELEGRAM_BOT_TOKEN": "123456:test-token",
                        "TELEGRAM_ALLOWED_USERS": "42",
                        "TELEGRAM_WEBHOOK_URL": "https://agent.example/telegram",
                        "TELEGRAM_WEBHOOK_SECRET": "",
                    },
                ),
                self.assertRaisesRegex(
                    ConfigError, "TELEGRAM_WEBHOOK_SECRET"
                ),
            ):
                Config.load(channel="telegram")

    def test_enabled_webhook_requires_a_public_https_url(self):
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "config.yaml").write_text(
                "telegram:\n  enabled: true\n",
                encoding="utf-8",
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "PILOTAGE_HOME": temporary,
                        "TELEGRAM_BOT_TOKEN": "123456:test-token",
                        "TELEGRAM_ALLOWED_USERS": "42",
                        "TELEGRAM_WEBHOOK_URL": "http://agent.example/telegram",
                        "TELEGRAM_WEBHOOK_SECRET": "long_random_secret",
                    },
                ),
                self.assertRaisesRegex(ConfigError, "https"),
            ):
                Config.load(channel="telegram")

    async def test_secure_webhook_mode_uses_hermes_startup_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "config.yaml").write_text(
                "telegram:\n  enabled: true\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "PILOTAGE_HOME": temporary,
                    "TELEGRAM_BOT_TOKEN": "123456:test-token",
                    "TELEGRAM_ALLOWED_USERS": "42",
                    "TELEGRAM_WEBHOOK_URL": "https://agent.example/telegram",
                    "TELEGRAM_WEBHOOK_SECRET": "long_random_secret",
                    "TELEGRAM_WEBHOOK_HOST": "127.0.0.1",
                    "TELEGRAM_WEBHOOK_PORT": "9443",
                },
            ):
                channel = telegram.TelegramChannel(
                    Config.load(channel="telegram"),
                    mock.AsyncMock(),
                    mock.AsyncMock(),
                )
                bot = SimpleNamespace(
                    id=999,
                    username="pilotage_bot",
                    delete_webhook=mock.AsyncMock(),
                )
                updater = SimpleNamespace(
                    running=True,
                    start_webhook=mock.AsyncMock(),
                    start_polling=mock.AsyncMock(),
                    stop=mock.AsyncMock(),
                )
                app = SimpleNamespace(
                    bot=bot,
                    updater=updater,
                    running=True,
                    initialize=mock.AsyncMock(),
                    start=mock.AsyncMock(),
                    stop=mock.AsyncMock(),
                    shutdown=mock.AsyncMock(),
                )
                update_type = SimpleNamespace(ALL_TYPES=("message",))

                with (
                    mock.patch.object(telegram, "TELEGRAM_AVAILABLE", True),
                    mock.patch.object(telegram, "Update", update_type),
                    mock.patch.object(
                        channel,
                        "_build_application",
                        return_value=app,
                    ),
                ):
                    await channel.start()
                    await channel.stop()

        updater.start_webhook.assert_awaited_once_with(
            listen="127.0.0.1",
            port=9443,
            url_path="/telegram",
            webhook_url="https://agent.example/telegram",
            secret_token="long_random_secret",
            allowed_updates=("message",),
            drop_pending_updates=False,
        )
        updater.start_polling.assert_not_awaited()
        bot.delete_webhook.assert_not_awaited()



if __name__ == "__main__":  # pragma: no cover
    unittest.main()
