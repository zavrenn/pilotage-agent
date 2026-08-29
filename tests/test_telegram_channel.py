from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import timedelta
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


def _telegram_update(update_id: int, text: str = "hello", *, user_id: int = 42):
    return telegram.Update.de_json(
        {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "date": 0,
                "chat": {"id": 42, "type": "private"},
                "from": {
                    "id": user_id,
                    "is_bot": False,
                    "first_name": "Owner",
                },
                "text": text,
            },
        },
        None,
    )


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

    async def test_hidden_text_link_reaches_agent_using_utf16_offsets(self):
        channel, handler, _ = self._channel(
            "telegram:\n"
            "  batch_delay: 0\n"
            "  batch_hard_cap: 1\n"
        )
        message = _message(
            text="😀open this",
            entities=[
                SimpleNamespace(
                    type="text_link",
                    offset=2,
                    length=4,
                    url="https://example.test/hidden",
                )
            ],
        )

        await channel._accept_message(message, [])
        await asyncio.sleep(0.05)

        handler.assert_awaited_once()
        self.assertEqual(
            handler.await_args.args[0].text,
            "😀open (https://example.test/hidden) this",
        )
        await channel.stop()

    def test_hidden_caption_link_is_exposed(self):
        message = _message(
            text="",
            caption="See map",
            caption_entities=[
                SimpleNamespace(
                    type="text_link",
                    offset=4,
                    length=3,
                    url="https://example.test/map",
                )
            ],
        )

        self.assertEqual(
            telegram._expand_telegram_text_links(message),
            "See map (https://example.test/map)",
        )

    def test_outbound_retry_requires_proven_unsent_connect_failure(self):
        connect_error = telegram.NetworkError("offline")
        connect_error.__cause__ = telegram.httpx.ConnectError("offline")
        read_error = telegram.NetworkError("response lost")
        read_error.__cause__ = telegram.httpx.ReadError("response lost")
        pool_timeout = telegram.TimedOut("pool busy")
        pool_timeout.__cause__ = telegram.httpx.PoolTimeout("pool busy")
        read_timeout = telegram.TimedOut("response timed out")
        read_timeout.__cause__ = telegram.httpx.ReadTimeout("response timed out")

        self.assertTrue(telegram._send_failure(connect_error, "").retryable)
        self.assertTrue(telegram._send_failure(pool_timeout, "").retryable)
        self.assertFalse(telegram._send_failure(read_error, "").retryable)
        self.assertFalse(telegram._send_failure(read_timeout, "").retryable)
        self.assertFalse(
            telegram._send_failure(telegram.NetworkError("ambiguous"), "").retryable
        )

    def test_retry_after_timedelta_is_preserved_as_seconds(self):
        with mock.patch.dict(os.environ, {"PTB_TIMEDELTA": "true"}):
            error = telegram.RetryAfter(timedelta(minutes=2))
            self.assertIsInstance(error.retry_after, timedelta)
            result = telegram._send_failure(error, "")

        self.assertTrue(result.retryable)
        self.assertEqual(result.retry_after, 120.0)

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
        invocation = command_handler.await_args.args[-2]
        self.assertEqual(invocation.command.name, "new")
        self.assertEqual(command_handler.await_args.args[-1], "")
        await channel.stop()

    async def test_invalid_new_command_does_not_discard_pending_input(self):
        channel, handler, command_handler = self._channel(
            "telegram:\n"
            "  batch_delay: 60\n"
            "  batch_hard_cap: 60\n"
        )
        self.addAsyncCleanup(channel.stop)
        session_id = telegram._session_id("42", "42", False, "")

        await channel._accept_message(
            _message(text="keep me", message_id=1),
            [],
        )
        self.assertEqual(channel._pending[session_id].text, "keep me")

        await channel._accept_message(
            _message(text="/new later", message_id=2),
            [],
        )
        await asyncio.sleep(0)
        await asyncio.gather(*list(channel._background_tasks))
        self.assertEqual(channel._pending[session_id].text, "keep me")
        self.assertEqual(
            command_handler.await_args_list[0].args[-2].arguments,
            "later",
        )
        handler.assert_not_awaited()

        await channel._accept_message(
            _message(text="/new", message_id=3),
            [],
        )
        await asyncio.sleep(0)
        await asyncio.gather(*list(channel._background_tasks))
        self.assertNotIn(session_id, channel._pending)
        self.assertEqual(
            command_handler.await_args_list[1].args[-2].arguments,
            "",
        )

    async def test_durable_management_command_receives_its_claim_identity(self):
        channel, handler, command_handler = self._channel()
        update = _telegram_update(704, "/new")
        claim_id, _ = channel._inbound_store.record(update)

        await channel._handle_command(update, None)
        await asyncio.sleep(0)
        await asyncio.gather(*list(channel._background_tasks))

        handler.assert_not_awaited()
        command_handler.assert_awaited_once()
        self.assertEqual(command_handler.await_args.args[-1], claim_id)
        self.assertEqual(channel._inbound_store.pending(), [])
        await channel.stop()

    async def test_startup_gate_allows_approval_but_holds_media_until_release(self):
        command_started = asyncio.Event()
        commands = []
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

        async def command(
            _chat, _session, message_id, _thread, invocation, _claim_id
        ):
            self.assertEqual(invocation.command.name, "approve")
            commands.append(message_id)
            command_started.set()

        async def download(message):
            self.assertEqual(getattr(message, "message_id", None), 12)
            media_download_started.set()
            await release_media.wait()
            return [], []

        channel._handler = handler
        channel._on_command = command
        channel._download_attachments = download
        old_approval = _telegram_update(720, "/approve")
        fresh_approval = _telegram_update(721, "/approve")
        channel._inbound_store.record(old_approval)
        channel._inbound_store.record(fresh_approval)
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
        await channel._handle_command(old_approval, None)
        await asyncio.sleep(0)
        self.assertFalse(command_started.is_set())
        self.assertFalse(media_download_started.is_set())
        self.assertEqual(len(channel._startup_updates), 2)

        await channel.enable_startup_approvals()
        await channel._handle_command(old_approval, None)
        await asyncio.sleep(0)
        self.assertFalse(command_started.is_set())
        self.assertEqual(len(channel._startup_updates), 2)
        await channel._handle_command(fresh_approval, None)
        await asyncio.wait_for(command_started.wait(), timeout=0.5)
        self.assertEqual(commands, ["721"])
        self.assertEqual(len(channel._startup_updates), 2)

        releasing = asyncio.create_task(channel.release_inbound())
        await asyncio.wait_for(media_download_started.wait(), timeout=0.5)
        release_media.set()
        await asyncio.wait_for(media_delivered.wait(), timeout=0.5)
        await asyncio.wait_for(releasing, timeout=0.5)
        await asyncio.gather(*list(channel._background_tasks))
        self.assertEqual(commands, ["721", "720"])
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

        async def command(
            _chat, _session, _message, _thread, invocation, _claim_id
        ):
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

    async def test_text_send_preserves_the_telegram_message_id(self):
        channel, _, _ = self._channel()
        channel._bot = SimpleNamespace(
            send_message=mock.AsyncMock(return_value=SimpleNamespace(message_id=21))
        )

        delivered = await channel.send("42", "hello")

        self.assertTrue(delivered)
        self.assertEqual(delivered.message_id, "21")

    async def test_progress_edit_targets_the_exact_message(self):
        channel, _, _ = self._channel()
        edit = mock.AsyncMock(return_value=SimpleNamespace(message_id=21))
        channel._bot = SimpleNamespace(edit_message_text=edit)
        parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")

        with mock.patch.object(telegram, "ParseMode", parse_mode):
            edited = await channel.edit_message(
                "42",
                "21",
                "Still working. (2 min)",
            )

        self.assertTrue(edited)
        self.assertEqual(edited.message_id, "21")
        edit.assert_awaited_once_with(
            chat_id=42,
            message_id=21,
            text="Still working\. \(2 min\)",
            parse_mode="MarkdownV2",
        )

    async def test_progress_edit_parse_error_falls_back_in_place(self):
        class FakeBadRequest(Exception):
            pass

        channel, _, _ = self._channel()
        edit = mock.AsyncMock(
            side_effect=[FakeBadRequest("Can't parse entities"), object()]
        )
        channel._bot = SimpleNamespace(edit_message_text=edit)
        parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")

        with (
            mock.patch.object(telegram, "ParseMode", parse_mode),
            mock.patch.object(telegram, "BadRequest", FakeBadRequest),
        ):
            edited = await channel.edit_message("42", "21", "**working**")

        self.assertTrue(edited)
        self.assertEqual(edit.await_count, 2)
        fallback = edit.await_args.kwargs
        self.assertEqual(fallback["chat_id"], 42)
        self.assertEqual(fallback["message_id"], 21)
        self.assertEqual(fallback["text"], "working")
        self.assertIsNone(fallback["parse_mode"])

    async def test_progress_edit_network_failure_retries_the_same_edit(self):
        channel, _, _ = self._channel()
        error = telegram.NetworkError("offline")
        error.__cause__ = telegram.httpx.ReadTimeout("late response")
        channel._bot = SimpleNamespace(
            edit_message_text=mock.AsyncMock(side_effect=error)
        )

        edited = await channel.edit_message("42", "21", "working")

        self.assertFalse(edited)
        self.assertTrue(edited.retryable)

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
                error = telegram.NetworkError("offline")
                error.__cause__ = telegram.httpx.ConnectError("offline")
                raise error
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
        ledger = DeliveryUnitLedger(store, obligation_id)

        self.assertTrue(
            await channel.send("42", content, "7", delivery_ledger=ledger)
        )
        channel._reply_to_mode = "off"

        with self.assertRaisesRegex(DeliveryPlanError, "plan changed"):
            await channel.send("42", content, "7", delivery_ledger=ledger)

    async def test_unit_plan_failure_happens_before_first_bot_api_call(self):
        channel, _, _ = self._channel()
        bot = SimpleNamespace(
            send_message=mock.AsyncMock(
                return_value=SimpleNamespace(message_id=101)
            )
        )
        channel._bot = bot
        content = "planned reply"
        obligation_id = compute_obligation_id("session", "message", content)
        store = DeliveryStore(self.root / "plan-failure.db")
        store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content=content,
        )
        ledger = DeliveryUnitLedger(store, obligation_id)

        with (
            mock.patch.object(
                store,
                "record_units",
                side_effect=sqlite3.OperationalError("plan write failed"),
            ),
            self.assertRaisesRegex(sqlite3.OperationalError, "plan write failed"),
        ):
            await channel.send("42", content, delivery_ledger=ledger)

        bot.send_message.assert_not_awaited()
        self.assertFalse(store.has_unit_plan(obligation_id))

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

    async def test_durable_claims_survive_batching_until_handler_success(self):
        channel, handler, _ = self._channel(
            "telegram:\n  batch_delay: 0\n  batch_hard_cap: 1\n"
        )
        self.addAsyncCleanup(channel.stop)
        updates = [_telegram_update(701, "one"), _telegram_update(702, "two")]
        expected = [channel._inbound_store.record(update)[0] for update in updates]

        await channel._handle_text(updates[0], None)
        await channel._handle_text(updates[1], None)
        await asyncio.sleep(0.05)

        handler.assert_awaited_once()
        inbound = handler.await_args.args[0]
        self.assertEqual(inbound.claim_ids, expected)
        self.assertTrue(all(len(claim) == 64 for claim in inbound.claim_ids))
        self.assertEqual(channel._inbound_store.pending(), [])

    async def test_recovery_completion_skips_update_already_held_in_ram(self):
        channel, handler, _ = self._channel()
        update = _telegram_update(703, "recover me")
        claim_id, _ = channel._inbound_store.record(update)
        channel.hold_inbound()

        await channel._handle_text(update, None)
        self.assertEqual(len(channel._startup_updates), 1)
        await asyncio.to_thread(channel.persist_completed_claims, [claim_id])
        await channel.release_inbound()

        handler.assert_not_awaited()
        self.assertEqual(channel._inbound_store.pending(), [])

    async def test_startup_abort_retains_spooled_updates_without_dispatch(self):
        channel, handler, command = self._channel()
        updates = [
            _telegram_update(705, "ordinary"),
            _telegram_update(706, "/new"),
            _telegram_update(707, "late ordinary"),
        ]
        expected_claims = [
            channel._inbound_store.record(update)[0] for update in updates
        ]
        channel.hold_inbound()

        await channel._handle_text(updates[0], None)
        await channel._handle_command(updates[1], None)
        self.assertEqual(len(channel._startup_updates), 2)

        async def stop_app():
            await channel._handle_text(updates[2], None)

        channel._app = SimpleNamespace(
            updater=SimpleNamespace(running=False),
            running=True,
            stop=mock.AsyncMock(side_effect=stop_app),
            shutdown=mock.AsyncMock(),
        )

        await channel.abort_startup()
        await asyncio.sleep(0)

        handler.assert_not_awaited()
        command.assert_not_awaited()
        self.assertEqual(
            [claim_id for claim_id, _payload in channel._inbound_store.pending()],
            expected_claims,
        )
        self.assertEqual(channel._startup_updates, [])
        self.assertEqual(channel._startup_held_claims, set())
        self.assertTrue(channel.stopped.is_set())


class TelegramInboundSpoolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "telegram-inbound.db"

    async def test_queue_does_not_publish_before_durable_record_returns(self):
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def record(_update):
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=1)
            return "a" * 64, "pending"

        queue = telegram._DurableUpdateQueue(
            SimpleNamespace(record=record),
            mock.Mock(),
            {"42"},
        )
        putting = asyncio.create_task(queue.put(_telegram_update(801)))
        await asyncio.wait_for(started.wait(), timeout=0.5)
        self.assertTrue(queue.empty())

        release.set()
        await asyncio.wait_for(putting, timeout=0.5)
        self.assertEqual((await queue.get()).update_id, 801)

    async def test_unauthorized_sender_is_rejected_before_durable_record(self):
        store = SimpleNamespace(record=mock.Mock())
        queue = telegram._DurableUpdateQueue(store, mock.Mock(), {"42"})

        await queue.put(_telegram_update(810, user_id=99))

        store.record.assert_not_called()
        self.assertTrue(queue.empty())

    async def test_replay_discards_preexisting_unauthorized_pending_update(self):
        store = telegram._TelegramInboundStore(self.path, "123456:old-secret")
        store.record(_telegram_update(811, user_id=99))
        queue = telegram._DurableUpdateQueue(store, mock.Mock(), {"42"})

        await queue.replay_pending(None)

        self.assertTrue(queue.empty())
        self.assertEqual(store.pending(), [])

    async def test_queue_rechecks_payload_and_serializes_duplicate_puts(self):
        store = telegram._TelegramInboundStore(self.path, "123456:old-secret")
        failures = []
        queue = telegram._DurableUpdateQueue(store, failures.append, {"42"})
        update = _telegram_update(802)

        await asyncio.gather(queue.put(update), queue.put(update))
        self.assertEqual(queue.qsize(), 1)
        with self.assertRaises(telegram.ChannelError):
            await queue.put(_telegram_update(802, "different"))
        self.assertEqual(len(failures), 1)

    async def test_pending_update_replays_until_exact_claim_is_completed(self):
        store = telegram._TelegramInboundStore(self.path, "123456:old-secret")
        first = telegram._DurableUpdateQueue(store, mock.Mock(), {"42"})
        await first.put(_telegram_update(803, "owed"))
        claim_id = store.pending()[0][0]

        rotated = telegram._TelegramInboundStore(
            self.path,
            "123456:new-secret",
        )
        replay = telegram._DurableUpdateQueue(rotated, mock.Mock(), {"42"})
        await replay.replay_pending(None)
        self.assertEqual((await replay.get()).update_id, 803)
        self.assertEqual(rotated.pending()[0][0], claim_id)

        rotated.complete([claim_id])
        after_completion = telegram._DurableUpdateQueue(
            rotated,
            mock.Mock(),
            {"42"},
        )
        await after_completion.replay_pending(None)
        self.assertTrue(after_completion.empty())

    def test_replacement_bot_can_reuse_an_update_id_completed_by_old_bot(self):
        update = _telegram_update(812, "same platform update id")
        old = telegram._TelegramInboundStore(self.path, "123456:old-secret")
        old_claim, _ = old.record(update)
        old.complete([old_claim])

        replacement = telegram._TelegramInboundStore(
            self.path,
            "654321:new-secret",
        )
        replacement_claim, state = replacement.record(update)

        self.assertEqual(state, "pending")
        self.assertNotEqual(replacement_claim, old_claim)
        self.assertEqual(
            replacement.pending(),
            [(replacement_claim, replacement._payload(update))],
        )

    async def test_replacement_bot_refuses_to_execute_foreign_pending_work(self):
        old = telegram._TelegramInboundStore(self.path, "123456:old-secret")
        old.record(_telegram_update(813, "owed by the old bot"))
        replacement = telegram._TelegramInboundStore(
            self.path,
            "654321:new-secret",
        )
        failure = mock.Mock()
        replay = telegram._DurableUpdateQueue(replacement, failure, {"42"})

        with self.assertRaisesRegex(telegram.ChannelError, "not recoverable"):
            await replay.replay_pending(None)

        self.assertTrue(replay.empty())
        failure.assert_called_once_with(
            "The Telegram durable inbound spool failed."
        )

    def test_legacy_global_unique_is_removed_even_when_composite_already_exists(self):
        update = _telegram_update(814, "legacy completion")
        old = telegram._TelegramInboundStore(self.path, "123456:old-secret")
        payload = old._payload(update)
        old_claim = old.claim_id(update.update_id)
        connection = sqlite3.connect(self.path)
        try:
            connection.executescript(
                """
                CREATE TABLE telegram_updates (
                    claim_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    update_id INTEGER NOT NULL UNIQUE,
                    payload TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'completed')),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX telegram_namespace_update_id
                    ON telegram_updates(namespace, update_id);
                """
            )
            connection.execute(
                "INSERT INTO telegram_updates VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    old_claim,
                    old._namespace,
                    update.update_id,
                    payload,
                    hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    "completed",
                    time.time(),
                    time.time(),
                ),
            )
            connection.commit()
        finally:
            connection.close()

        replacement = telegram._TelegramInboundStore(
            self.path,
            "654321:new-secret",
        )
        replacement_claim, state = replacement.record(update)

        self.assertEqual(state, "pending")
        self.assertNotEqual(replacement_claim, old_claim)
        connection = sqlite3.connect(self.path)
        try:
            indexes = connection.execute(
                "PRAGMA index_list('telegram_updates')"
            ).fetchall()
            unique_columns = []
            for _sequence, name, unique, *_rest in indexes:
                if unique:
                    unique_columns.append(
                        [
                            row[2]
                            for row in connection.execute(
                                f'PRAGMA index_info("{name}")'
                            ).fetchall()
                        ]
                    )
        finally:
            connection.close()
        self.assertIn(["namespace", "update_id"], unique_columns)
        self.assertNotIn(["update_id"], unique_columns)

    def test_foreign_completions_do_not_consume_new_bot_capacity(self):
        old = telegram._TelegramInboundStore(self.path, "123456:old-secret")
        old_claim, _ = old.record(_telegram_update(815, "old"))
        old.complete([old_claim])
        replacement = telegram._TelegramInboundStore(
            self.path,
            "654321:new-secret",
        )

        with mock.patch.object(telegram, "INBOUND_SPOOL_MAX_ROWS", 1):
            replacement_claim, state = replacement.record(
                _telegram_update(816, "new")
            )

        self.assertEqual(state, "pending")
        self.assertEqual(replacement.pending()[0][0], replacement_claim)

    def test_new_bot_prunes_expired_completion_from_old_namespace(self):
        old = telegram._TelegramInboundStore(self.path, "123456:old-secret")
        old_claim, _ = old.record(_telegram_update(817, "expired"))
        old.complete([old_claim])
        connection = sqlite3.connect(self.path)
        try:
            connection.execute(
                "UPDATE telegram_updates SET updated_at = 0 WHERE claim_id = ?",
                (old_claim,),
            )
            connection.commit()
        finally:
            connection.close()

        replacement = telegram._TelegramInboundStore(
            self.path,
            "654321:new-secret",
        )
        self.assertEqual(replacement.pending(), [])
        connection = sqlite3.connect(self.path)
        try:
            remaining = connection.execute(
                "SELECT COUNT(*) FROM telegram_updates WHERE claim_id = ?",
                (old_claim,),
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(remaining, 0)

    def test_capacity_never_prunes_fresh_completion_or_pending_proofs(self):
        store = telegram._TelegramInboundStore(self.path, "123456:secret")
        with mock.patch.object(telegram, "INBOUND_SPOOL_MAX_ROWS", 2):
            first_update = _telegram_update(804)
            first = store.record(first_update)[0]
            store.record(_telegram_update(805))
            store.complete([first])
            with self.assertRaises(telegram._InboundSpoolError):
                store.record(_telegram_update(806))
            self.assertEqual(
                [telegram.json.loads(payload)["update_id"] for _, payload in store.pending()],
                [805],
            )
            self.assertEqual(store.record(first_update)[1], "completed")

    def test_payload_and_total_byte_capacity_fail_closed(self):
        update = _telegram_update(808, "bounded")
        with mock.patch.object(telegram, "INBOUND_SPOOL_MAX_UPDATE_BYTES", 16):
            store = telegram._TelegramInboundStore(self.path, "123456:secret")
            with self.assertRaises(telegram._InboundSpoolError):
                store.record(update)
            self.assertEqual(store.pending(), [])

        second_path = self.path.with_name("telegram-total.db")
        store = telegram._TelegramInboundStore(second_path, "123456:secret")
        payload_bytes = len(store._payload(update).encode("utf-8"))
        with mock.patch.object(
            telegram,
            "INBOUND_SPOOL_MAX_BYTES",
            payload_bytes,
        ):
            store.record(update)
            with self.assertRaises(telegram._InboundSpoolError):
                store.record(_telegram_update(809, "bounded"))
            self.assertEqual(
                [telegram.json.loads(payload)["update_id"] for _, payload in store.pending()],
                [808],
            )

    def test_caption_on_unsupported_message_is_not_spooled(self):
        update = SimpleNamespace(
            effective_message=SimpleNamespace(
                text=None,
                caption="animation",
                animation=SimpleNamespace(),
                location=None,
                venue=None,
                photo=[],
                video=None,
                audio=None,
                voice=None,
                document=None,
                sticker=None,
            )
        )
        self.assertFalse(telegram._durable_update_required(update))


class TelegramLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_polling_progress_requires_successful_bot_api_response(self):
        progress = mock.Mock()
        request = telegram._PollingProgressRequest.__new__(
            telegram._PollingProgressRequest
        )
        request._on_progress = progress

        with mock.patch.object(
            telegram.HTTPXRequest,
            "do_request",
            new=mock.AsyncMock(return_value=(200, b'{"ok":true,"result":[]}')),
        ):
            self.assertEqual(
                await request.do_request("https://example.test", "POST"),
                (200, b'{"ok":true,"result":[]}'),
            )
        progress.assert_called_once_with()

        progress.reset_mock()
        with mock.patch.object(
            telegram.HTTPXRequest,
            "do_request",
            new=mock.AsyncMock(return_value=(429, b'{"ok":false}')),
        ):
            await request.do_request("https://example.test", "POST")
        progress.assert_not_called()

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

                async def start_polling(**_kwargs):
                    channel._record_polling_progress()

                updater = SimpleNamespace(
                    running=True,
                    start_polling=mock.AsyncMock(side_effect=start_polling),
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
                    mock.patch.object(
                        channel,
                        "_build_application",
                        return_value=app,
                    ),
                ):
                    await channel.start()
                    self.assertFalse(channel.startup_approval_available)
                    await channel.enable_startup_approvals()
                    self.assertTrue(channel.startup_approval_available)
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

    async def test_polling_startup_requires_successful_get_updates_progress(self):
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

        bot = SimpleNamespace(
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
            mock.patch.object(channel, "_build_application", return_value=app),
            mock.patch.object(telegram, "POLLING_STARTUP_PROGRESS_SECONDS", 0.01),
            self.assertRaisesRegex(telegram.ChannelError, "no successful getUpdates"),
        ):
            await channel.start()

        self.assertFalse(
            updater.start_polling.await_args.kwargs["drop_pending_updates"]
        )
        updater.stop.assert_awaited_once()
        app.shutdown.assert_awaited_once()

    async def test_polling_watchdog_fails_after_bounded_stall(self):
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

        channel._app = SimpleNamespace(updater=SimpleNamespace(running=True))
        channel._running = True
        channel._webhook_mode = False
        channel._polling_last_progress = 0.0
        with (
            mock.patch.object(telegram, "POLLING_WATCHDOG_INTERVAL_SECONDS", 0),
            mock.patch.object(telegram, "POLLING_STALL_SECONDS", 0),
        ):
            await channel._polling_watchdog()

        self.assertTrue(channel.stopped.is_set())
        self.assertIn("no successful getUpdates", channel.failure)

    async def test_failed_start_retains_updates_accepted_under_startup_hold(self):
        handler = mock.AsyncMock()
        command = mock.AsyncMock()
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
                    handler,
                    command,
                )

        update = _telegram_update(708, "accepted before polling failed")
        claim_id, _ = channel._inbound_store.record(update)

        async def fail_after_accepting(**_kwargs):
            channel._startup_updates.append(("text", update, None))
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
        replay = mock.AsyncMock()

        with (
            mock.patch.object(telegram, "TELEGRAM_AVAILABLE", True),
            mock.patch.object(channel, "_build_application", return_value=app),
            mock.patch.object(channel, "_replay_startup_update", replay),
            self.assertRaisesRegex(telegram.ChannelError, "could not start"),
        ):
            await channel.start()

        replay.assert_not_awaited()
        handler.assert_not_awaited()
        command.assert_not_awaited()
        self.assertEqual(
            [pending_claim for pending_claim, _ in channel._inbound_store.pending()],
            [claim_id],
        )
        self.assertEqual(channel._startup_updates, [])
        self.assertTrue(channel.stopped.is_set())

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

    async def test_durable_update_error_fails_channel_for_restart_replay(self):
        channel = telegram.TelegramChannel.__new__(telegram.TelegramChannel)
        channel._token = "123456:test-token"
        channel.failure = None
        channel._running = True
        channel.stopped = asyncio.Event()
        context = SimpleNamespace(error=RuntimeError("handler failed"))

        await channel._handle_update_error(_telegram_update(812), context)

        self.assertFalse(channel._running)
        self.assertTrue(channel.stopped.is_set())
        self.assertIn("durable input will replay", channel.failure)



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
