"""Omitted attachments stay visible and durable on both client channels."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx

from pilotage import media
from pilotage.channels import telegram, whatsapp
from pilotage.config import Config
from pilotage.delivery import (
    DeliveryPlanError,
    DeliveryStore,
    DeliveryUnitLedger,
    compute_obligation_id,
    recover_deliveries,
)
from pilotage.i18n import SUPPORTED_LANGUAGES, t


class OutboundAttachmentNoticeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.counter = 0

    def channel(self, platform, language="en", *, declared_root=False):
        self.counter += 1
        home = self.root / str(self.counter)
        home.mkdir()
        reports = home / "reports"
        reports.mkdir()
        settings = f"display:\n  language: {language}\n"
        if declared_root:
            settings += (
                "gateway:\n"
                f"  media_delivery_allow_dirs: ['{reports.as_posix()}']\n"
            )
        (home / "config.yaml").write_text(settings, encoding="utf-8")
        with mock.patch.dict(os.environ, {
            "PILOTAGE_HOME": str(home),
            "PILOTAGE_ALLOWED_SENDERS": "212600000000",
            "TELEGRAM_BOT_TOKEN": "123456:test-token",
            "TELEGRAM_ALLOWED_USERS": "42",
            "TELEGRAM_WEBHOOK_URL": "",
            "TELEGRAM_WEBHOOK_SECRET": "",
        }):
            config = Config.load(channel=platform)
        config.workspace_dir.mkdir(parents=True)
        fixture = SimpleNamespace(
            home=home, reports=reports, config=config, platform=platform,
            calls=[], reject_file=False,
            reject_text_number=None,
        )
        if platform == "whatsapp":
            fixture.channel = whatsapp.WhatsAppChannel(
                config, mock.AsyncMock(), mock.AsyncMock(),
            )
            fixture.chat_id = "212600000000@s.whatsapp.net"
            fixture.send_kwargs = {}
            fixture.format_text = whatsapp.to_whatsapp

            async def post(url, **kwargs):
                payload = kwargs["json"]
                kind = "file" if url.endswith("/send-media") else "text"
                fixture.calls.append((kind, payload))
                request = httpx.Request("POST", url)
                reject_text = (
                    kind == "text"
                    and len(self.sent_text(fixture)) == fixture.reject_text_number
                )
                status = 503 if reject_text or (kind == "file" and fixture.reject_file) else 200
                return httpx.Response(
                    status, request=request,
                    json={"success": True, "messageId": str(len(fixture.calls))},
                )

            fixture.channel._http = SimpleNamespace(post=post)
        else:
            fixture.channel = telegram.TelegramChannel(
                config, mock.AsyncMock(), mock.AsyncMock(),
            )
            fixture.chat_id = "-10042"
            fixture.send_kwargs = {"thread_id": "9"}
            fixture.format_text = telegram.to_telegram

            async def send_message(**kwargs):
                fixture.calls.append(("text", kwargs))
                if len(self.sent_text(fixture)) == fixture.reject_text_number:
                    error = telegram.NetworkError("offline")
                    error.__cause__ = httpx.ConnectError("offline")
                    raise error
                return SimpleNamespace(message_id=len(fixture.calls))

            async def send_document(**kwargs):
                fixture.calls.append(("file", kwargs))
                if fixture.reject_file:
                    error = telegram.NetworkError("offline")
                    error.__cause__ = httpx.ConnectError("offline")
                    raise error
                return SimpleNamespace(message_id=len(fixture.calls))

            fixture.channel._bot = SimpleNamespace(
                send_message=send_message, send_document=send_document,
            )
        return fixture

    async def send(self, fixture, text, **kwargs):
        return await fixture.channel.send(
            fixture.chat_id, text, "7", **fixture.send_kwargs, **kwargs,
        )

    @staticmethod
    def sent_text(fixture):
        return [
            payload.get("message", payload.get("text"))
            for kind, payload in fixture.calls if kind == "text"
        ]

    def record(self, fixture, content):
        path = fixture.home / "delivery.db"
        store = DeliveryStore(path)
        obligation_id = compute_obligation_id("session", "message", content)
        store.record(
            obligation_id=obligation_id, session_key="session",
            platform=fixture.platform, chat_id=fixture.chat_id,
            thread_id=fixture.send_kwargs.get("thread_id", ""),
            reply_to="7", content=content,
        )
        return store, obligation_id

    async def test_mixed_files_keep_allowed_attachment_and_one_plain_notice(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                fixture = self.channel(platform, "fr")
                report = fixture.config.workspace_dir / "report.pdf"
                report.write_bytes(b"%PDF")
                outside = fixture.home / "private.pdf"
                outside.write_bytes(b"private")
                missing = fixture.config.workspace_dir / "missing.pdf"
                content = (
                    f"Your reports are attached\nMEDIA:{report}\n"
                    f"MEDIA:{outside}\nMEDIA:{missing}"
                )

                self.assertTrue(await self.send(fixture, content))

                self.assertEqual([kind for kind, _ in fixture.calls], ["text", "file"])
                self.assertEqual(self.sent_text(fixture), [fixture.format_text(
                    "Your reports are attached\n\n" + t("media.delivery_unavailable", "fr"),
                )])
                payload = fixture.calls[1][1]
                delivered_path = payload.get("filePath") or payload["document"].name
                self.assertEqual(Path(delivered_path), report.resolve())

    async def test_all_denied_files_send_a_localized_notice_without_paths(self):
        for platform in ("whatsapp", "telegram"):
            for language in SUPPORTED_LANGUAGES:
                with self.subTest(platform=platform, language=language):
                    fixture = self.channel(platform, language)
                    content = f"MEDIA:{fixture.home / 'private.pdf'}"

                    self.assertTrue(await self.send(fixture, content))

                    self.assertEqual([kind for kind, _ in fixture.calls], ["text"])
                    self.assertEqual(self.sent_text(fixture), [fixture.format_text(
                        t("media.delivery_unavailable", language),
                    )])

    async def test_already_confined_reply_does_not_duplicate_its_notice(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                fixture = self.channel(platform)
                content = media.confine_outbound(
                    f"Ready\nMEDIA:{fixture.home / 'private.pdf'}",
                    fixture.config.outbound_media_roots, language="en",
                )

                self.assertTrue(await self.send(fixture, content))

                self.assertEqual(self.sent_text(fixture), [fixture.format_text(
                    "Ready\n\n" + t("media.delivery_unavailable", "en"),
                )])

    async def test_system_echo_keeps_media_text_without_a_denial_notice(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                fixture = self.channel(platform)
                content = f"MEDIA:{fixture.home / 'private.pdf'}"

                self.assertTrue(await self.send(fixture, content, deliver_media=False))

                self.assertEqual([kind for kind, _ in fixture.calls], ["text"])
                self.assertEqual(self.sent_text(fixture), [fixture.format_text(content)])

    async def test_declared_delivery_root_is_accepted_without_notice(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                fixture = self.channel(platform, declared_root=True)
                report = fixture.reports / "report.pdf"
                report.write_bytes(b"%PDF")

                self.assertTrue(await self.send(fixture, f"MEDIA:{report}"))

                self.assertEqual([kind for kind, _ in fixture.calls], ["file"])

    async def test_language_change_retries_attachment_without_repeating_accepted_notice(self):
        for platform in ("whatsapp", "telegram"):
            for language in SUPPORTED_LANGUAGES:
                with self.subTest(platform=platform, language=language):
                    await self.check_attachment_restart(platform, language)

    async def check_attachment_restart(self, platform, language):
        fixture = self.channel(platform, language)
        report = fixture.config.workspace_dir / "report.pdf"
        report.write_bytes(b"%PDF")
        content = f"Ready\nMEDIA:{report}\nMEDIA:{fixture.home / 'private.pdf'}"
        store, obligation_id = self.record(fixture, content)
        fixture.reject_file = True

        first = await self.send(
            fixture, content, delivery_ledger=DeliveryUnitLedger(store, obligation_id),
        )
        self.assertFalse(first)
        self.assertTrue(first.retryable)
        self.assertTrue(store.mark_planned_failed(
            obligation_id, first.error, retry_safe=True,
        ))
        with closing(sqlite3.connect(store.path)) as connection:
            before = connection.execute(
                "SELECT unit_id, fingerprint, evidence FROM delivery_units"
                " WHERE state = 'delivered'",
            ).fetchall()
        fixture.reject_file = False
        fixture.channel._config = replace(
            fixture.config, language="fr" if language != "fr" else "ar",
        )
        restarted = DeliveryStore(store.path)

        self.assertEqual(await recover_deliveries(
            restarted, {platform: fixture.channel},
        ), 1)

        self.assertEqual(
            [kind for kind, _ in fixture.calls], ["text", "file", "file"],
        )
        self.assertEqual(self.sent_text(fixture), [fixture.format_text(
            "Ready\n\n" + t("media.delivery_unavailable", language),
        )])
        with closing(sqlite3.connect(store.path)) as connection:
            after = connection.execute(
                "SELECT unit_id, fingerprint, evidence FROM delivery_units"
                " WHERE kind = 'text'",
            ).fetchall()
        self.assertEqual(after, before)

    async def test_language_change_preserves_chunk_boundaries_after_partial_text_send(self):
        for platform in ("whatsapp", "telegram"):
            for language in SUPPORTED_LANGUAGES:
                with self.subTest(platform=platform, language=language):
                    fixture = self.channel(platform, language)
                    report = fixture.config.workspace_dir / "report.pdf"
                    report.write_bytes(b"%PDF")
                    prefix = "A" * (4096 - len(t("media.delivery_unavailable", language)) // 2)
                    content = f"{prefix}\nMEDIA:{report}\nMEDIA:{fixture.home / 'private.pdf'}"
                    store, obligation_id = self.record(fixture, content)
                    fixture.reject_text_number = 2

                    with mock.patch.object(whatsapp, "OUTBOUND_CHUNK_DELAY_SECONDS", 0):
                        first = await self.send(
                            fixture, content, delivery_ledger=DeliveryUnitLedger(store, obligation_id),
                        )
                    self.assertFalse(first)
                    self.assertTrue(first.retryable)
                    self.assertTrue(store.mark_planned_failed(
                        obligation_id, first.error, retry_safe=True,
                    ))
                    original_chunks = self.sent_text(fixture)
                    self.assertEqual(len(original_chunks), 2)
                    with closing(sqlite3.connect(store.path)) as connection:
                        accepted = connection.execute(
                            "SELECT unit_id, fingerprint, evidence FROM delivery_units"
                            " WHERE state = 'delivered'",
                        ).fetchall()
                    self.assertEqual(len(accepted), 1)
                    fixture.reject_text_number = None
                    fixture.channel._config = replace(
                        fixture.config, language="ar" if language != "ar" else "en",
                    )
                    with mock.patch.object(whatsapp, "OUTBOUND_CHUNK_DELAY_SECONDS", 0):
                        self.assertEqual(await recover_deliveries(
                            DeliveryStore(store.path), {platform: fixture.channel},
                        ), 1)

                    self.assertEqual(
                        self.sent_text(fixture), original_chunks + [original_chunks[1]],
                    )
                    self.assertEqual([kind for kind, _ in fixture.calls], ["text", "text", "text", "file"])
                    with closing(sqlite3.connect(store.path)) as connection:
                        recovered = connection.execute(
                            "SELECT unit_id, fingerprint, evidence FROM delivery_units"
                            " WHERE position = 0",
                        ).fetchall()
                    self.assertEqual(recovered, accepted)

    async def test_language_change_after_pretransport_crash_keeps_original_notice(self):
        for platform in ("whatsapp", "telegram"):
            for language in SUPPORTED_LANGUAGES:
                for crash_at in ("before_plan", "before_send"):
                    with self.subTest(platform=platform, language=language, crash_at=crash_at):
                        fixture = self.channel(platform, language)
                        report = fixture.config.workspace_dir / "report.pdf"
                        report.write_bytes(b"%PDF")
                        content = f"Ready\nMEDIA:{report}\nMEDIA:{fixture.home / 'private.pdf'}"
                        store, obligation_id = self.record(fixture, content)
                        ledger = DeliveryUnitLedger(store, obligation_id)
                        target, method = (store, "record_units") if crash_at == "before_plan" else (ledger, "run")
                        with (
                            mock.patch.object(target, method, side_effect=asyncio.CancelledError),
                            self.assertRaises(asyncio.CancelledError),
                        ):
                            await self.send(fixture, content, delivery_ledger=ledger)
                        self.assertEqual(fixture.calls, [])
                        fixture.channel._config = replace(
                            fixture.config, language="en" if language != "en" else "fr",
                        )

                        self.assertEqual(await recover_deliveries(
                            DeliveryStore(store.path), {platform: fixture.channel},
                        ), 1)

                        self.assertEqual(self.sent_text(fixture), [fixture.format_text(
                            "Ready\n\n" + t("media.delivery_unavailable", language),
                        )])
                        self.assertEqual([kind for kind, _ in fixture.calls], ["text", "file"])

    async def test_notice_binding_failure_prevents_transport_and_marks_preparation_failed(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                fixture = self.channel(platform)
                content = f"MEDIA:{fixture.home / 'private.pdf'}"
                store, obligation_id = self.record(fixture, content)
                ledger = DeliveryUnitLedger(store, obligation_id)
                with (
                    mock.patch.object(store, "record_attachment_notice", side_effect=sqlite3.OperationalError("unavailable")),
                    self.assertRaises(sqlite3.OperationalError),
                ):
                    await self.send(fixture, content, delivery_ledger=ledger)
                self.assertTrue(ledger.preparation_failed)
                self.assertFalse(ledger.prepared)
                self.assertEqual(fixture.calls, [])

    async def test_plain_text_does_not_reserve_an_attachment_notice(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                fixture = self.channel(platform)
                store, obligation_id = self.record(fixture, "Ready")
                with mock.patch.object(store, "record_attachment_notice") as reserve:
                    self.assertTrue(await self.send(
                        fixture, "Ready", delivery_ledger=DeliveryUnitLedger(store, obligation_id),
                    ))
                reserve.assert_not_called()

    async def test_concurrent_notice_preparation_freezes_one_value_without_changing_identity(self):
        fixture = self.channel("whatsapp")
        content = f"MEDIA:{fixture.home / 'private.pdf'}"
        store, obligation_id = self.record(fixture, content)
        choices = [t("media.delivery_unavailable", language) for language in SUPPORTED_LANGUAGES]
        results = await asyncio.gather(*[
            DeliveryUnitLedger(store, obligation_id).attachment_notice(content, notice)
            for notice in choices
        ])
        self.assertEqual(len(set(results)), 1)
        self.assertIn(results[0], choices)
        for notice in choices:
            self.assertEqual(store.record_attachment_notice(obligation_id, content, notice), results[0])
        with closing(sqlite3.connect(store.path)) as connection:
            self.assertEqual(connection.execute(
                "SELECT obligation_id, content, attachment_notice FROM delivery_obligations",
            ).fetchone(), (obligation_id, content, results[0]))
        with self.assertRaises(DeliveryPlanError):
            store.record_attachment_notice(obligation_id, "changed content", choices[0])
        with self.assertRaises(DeliveryPlanError):
            DeliveryStore(store.path).record_attachment_notice(obligation_id, content, choices[0])

    async def test_denial_notice_is_planned_before_any_transport_call(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                fixture = self.channel(platform)
                content = f"MEDIA:{fixture.home / 'private.pdf'}"
                store, obligation_id = self.record(fixture, content)
                with (
                    mock.patch.object(store, "record_units", side_effect=sqlite3.OperationalError("unavailable")),
                    self.assertRaises(sqlite3.OperationalError),
                ):
                    await self.send(
                        fixture, content, delivery_ledger=DeliveryUnitLedger(store, obligation_id),
                    )

                self.assertEqual(fixture.calls, [])


if __name__ == "__main__":
    unittest.main()
