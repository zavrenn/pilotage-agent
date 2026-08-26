from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pilotage import main
from pilotage.channels.telegram import InboundMessage
from pilotage.channels.whatsapp import ChannelError
from pilotage.config import Config, TELEGRAM_FORMATTING_NOTE
from pilotage.cron.jobs import _normalize_origin
from pilotage.delivery import DeliveryUnitLedger


class TelegramRuntimeTests(unittest.IsolatedAsyncioTestCase):
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
        (self.root / "config.yaml").write_text(
            "whatsapp:\n"
            "  enabled: false\n"
            "telegram:\n"
            "  enabled: true\n"
            "cron:\n"
            "  enabled: false\n"
            "stt:\n"
            "  enabled: true\n"
            "  echo_transcripts: true\n",
            encoding="utf-8",
        )

    async def test_telegram_uses_its_channel_config_and_preserves_topic_origin(self):
        config = Config.load(channel="whatsapp")
        seen = {}
        sent = []
        delivery = {"accepted": True}

        class FakeAgent:
            def __init__(self, channel_config, **_runtime_dependencies):
                seen["config"] = channel_config

            async def close(self):
                seen["agent_closed"] = True

            async def respond(
                self,
                session_id,
                text,
                attachments,
                *,
                on_notice,
                origin,
                approval_notify,
            ):
                seen["session_id"] = session_id
                seen["text"] = text
                seen["attachments"] = attachments
                seen["origin"] = origin
                seen["approval_notify"] = approval_notify
                return "answer"

        class FakeTelegramChannel:
            def __init__(self, channel_config, handler, manage):
                seen["channel_config"] = channel_config
                self.handler = handler
                self.manage = manage
                self.stopped = asyncio.Event()
                self.failure = None

            @contextlib.asynccontextmanager
            async def typing(self, chat_id, thread_id):
                seen["typing"] = (chat_id, thread_id)
                yield

            async def send(self, *args, **kwargs):
                sent.append((args, kwargs))
                return delivery["accepted"]

            async def start(self):
                await self.handler(
                    InboundMessage(
                        chat_id="42",
                        session_id="telegram:dm:42:9",
                        user_id="42",
                        user_name="Owner",
                        text="voice",
                        is_group=False,
                        thread_id="9",
                        message_ids=["99"],
                    )
                )
                self.stopped.set()

            async def stop_intake(self):
                seen["intake_stopped"] = True

            async def stop(self, *, drain_timeout_seconds=0):
                seen["drain_timeout"] = drain_timeout_seconds
                pass

        with (
            mock.patch.object(main, "Agent", FakeAgent),
            mock.patch.object(
                main,
                "TelegramChannel",
                FakeTelegramChannel,
            ),
            mock.patch.object(main, "WhatsAppChannel") as whatsapp,
            mock.patch.object(main.auth, "read_credentials"),
            mock.patch.object(
                main.transcription,
                "enrich_message",
                new=mock.AsyncMock(
                    return_value=('"spoken"', ["spoken"])
                ),
            ),
        ):
            result = await main.command_run(config)

        self.assertEqual(result, 0)
        whatsapp.assert_not_called()
        self.assertEqual(seen["config"].settings.channel, "telegram")
        self.assertIs(seen["config"], seen["channel_config"])
        self.assertIn(TELEGRAM_FORMATTING_NOTE, seen["config"].instructions)
        self.assertEqual(seen["typing"], ("42", "9"))
        self.assertEqual(
            seen["origin"],
            {
                "channel": "telegram",
                "chat_id": "42",
                "thread_id": "9",
            },
        )
        self.assertEqual(seen["session_id"], "telegram:dm:42:9")
        self.assertEqual(seen["text"], '"spoken"')
        self.assertTrue(seen["intake_stopped"])
        self.assertTrue(seen["agent_closed"])
        self.assertEqual(
            sent[0],
            (
                ("42", '\U0001f399\ufe0f "spoken"', "99"),
                {"thread_id": "9", "deliver_media": False},
            ),
        )
        self.assertEqual(sent[1][0], ("42", "answer", "99"))
        self.assertEqual(sent[1][1]["thread_id"], "9")
        self.assertIsInstance(
            sent[1][1]["delivery_ledger"], DeliveryUnitLedger
        )
        with contextlib.closing(
            sqlite3.connect(self.root / "delivery.db")
        ) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT platform, chat_id, thread_id, state"
                    " FROM delivery_obligations"
                ).fetchall(),
                [("telegram", "42", "9", "delivered")],
        )
        delivery["accepted"] = False
        self.assertFalse(
            await seen["approval_notify"]("Approve this change")
        )

    async def test_dual_channel_cron_receives_both_channel_configs(self):
        (self.root / "config.yaml").write_text(
            "whatsapp:\n"
            "  enabled: true\n"
            "telegram:\n"
            "  enabled: true\n"
            "cron:\n"
            "  enabled: true\n",
            encoding="utf-8",
        )
        config = Config.load(channel="whatsapp")
        seen = {}

        class FakeAgent:
            def __init__(self, _config, **_runtime_dependencies):
                pass

            async def close(self):
                pass

        class FakeChannel:
            def __init__(self, channel_config, _handler, _manage):
                self.channel_config = channel_config
                self.stopped = asyncio.Event()
                self.failure = None

            async def start(self):
                self.stopped.set()

            async def stop_intake(self):
                pass

            async def stop(self, *, drain_timeout_seconds=0):
                pass

        class FakeScheduler:
            def __init__(
                self,
                scheduler_config,
                _store,
                *,
                deliver,
                channel_configs,
            ):
                seen["scheduler_config"] = scheduler_config
                seen["channel_configs"] = channel_configs
                seen["deliver"] = deliver
                self.stopped = asyncio.Event()
                self.failure = None

            def wake(self):
                pass

            async def start(self):
                pass

            async def stop(self, *, drain_timeout_seconds=0):
                self.stopped.set()

        with (
            mock.patch.object(main, "Agent", FakeAgent),
            mock.patch.object(main, "WhatsAppChannel", FakeChannel),
            mock.patch.object(main, "TelegramChannel", FakeChannel),
            mock.patch.object(main, "CronScheduler", FakeScheduler),
            mock.patch.object(main.auth, "read_credentials"),
        ):
            result = await main.command_run(config)

        self.assertEqual(result, 0)
        self.assertEqual(seen["scheduler_config"].settings.channel, "whatsapp")
        self.assertEqual(
            set(seen["channel_configs"]),
            {"whatsapp", "telegram"},
        )
        self.assertEqual(
            seen["channel_configs"]["whatsapp"].settings.channel,
            "whatsapp",
        )
        self.assertEqual(
            seen["channel_configs"]["telegram"].settings.channel,
            "telegram",
        )

    async def test_whatsapp_failure_cannot_consume_telegram_startup_updates(self):
        (self.root / "config.yaml").write_text(
            "whatsapp:\n"
            "  enabled: true\n"
            "telegram:\n"
            "  enabled: true\n"
            "cron:\n"
            "  enabled: false\n",
            encoding="utf-8",
        )
        config = Config.load(channel="whatsapp")
        starts = []

        class FakeAgent:
            def __init__(self, _config, **_runtime_dependencies):
                pass

            async def close(self):
                pass

        class BaseChannel:
            def __init__(self, _config, _handler, _manage):
                self.stopped = asyncio.Event()
                self.failure = None

            def hold_inbound(self):
                pass

            async def stop(self, *, drain_timeout_seconds=0):
                pass

        class FailingWhatsApp(BaseChannel):
            async def start(self):
                starts.append("whatsapp")
                raise ChannelError("WhatsApp unavailable")

        class ObservingTelegram(BaseChannel):
            async def start(self):
                starts.append("telegram")

        with (
            mock.patch.object(main, "Agent", FakeAgent),
            mock.patch.object(main, "WhatsAppChannel", FailingWhatsApp),
            mock.patch.object(main, "TelegramChannel", ObservingTelegram),
            mock.patch.object(main.auth, "read_credentials"),
        ):
            result = await main.command_run(config)

        self.assertEqual(result, 1)
        self.assertEqual(starts, ["whatsapp"])

    def test_cron_origin_keeps_telegram_topic_id(self):
        self.assertEqual(
            _normalize_origin(
                {
                    "channel": "telegram",
                    "chat_id": "-100",
                    "thread_id": "17",
                }
            ),
            {
                "channel": "telegram",
                "chat_id": "-100",
                "thread_id": "17",
            },
        )
        self.assertEqual(
            _normalize_origin(
                {
                    "channel": "whatsapp",
                    "chat_id": "123@c.us",
                    "thread_id": "ignored",
                }
            ),
            {
                "channel": "whatsapp",
                "chat_id": "123@c.us",
            },
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
