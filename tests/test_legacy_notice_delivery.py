"""Recovery of old client replies through the real channel delivery planners."""

from __future__ import annotations

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
    DeliveryStore, DeliveryUnitLedger, claim_deliveries,
    compute_obligation_id, recover_deliveries,
)
from pilotage.i18n import SUPPORTED_LANGUAGES, t
from pilotage.legacy_notices import LEGACY_ATTACHMENT_NOTICE


# Exact storage-failure replies emitted before the client-message change.
STORAGE_NOTICES = (
    "I stopped because conversation state could not be saved safely. Check storage first. "
    "If an earlier request may have acted, verify it before using /new.",
    "Je me suis arrêté car l'état de la conversation n'a pas pu être enregistré en toute "
    "sécurité. Vérifiez d'abord le stockage. Si une demande précédente a pu agir, "
    "vérifiez-la avant d'utiliser /new.",
    "توقفت لأن حالة المحادثة لم تُحفظ بأمان. تحقّق من التخزين أولاً. "
    "إذا كان الطلب السابق ربما نفّذ إجراءً، فتحقّق منه قبل استخدام /new.",
)


class LegacyNoticeDeliveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)
        delay = mock.patch.object(whatsapp, "OUTBOUND_CHUNK_DELAY_SECONDS", 0)
        delay.start()
        self.addCleanup(delay.stop)
        self.counter = 0

    def channel(self, platform, language="fr"):
        config = replace(Config.load(channel=platform), language=language)
        config.workspace_dir.mkdir(parents=True, exist_ok=True)
        fixture = SimpleNamespace(platform=platform, calls=[], reject=set())
        if platform == "whatsapp":
            fixture.module = whatsapp
            fixture.chat_id, fixture.thread_id = "212600000000@s.whatsapp.net", ""
            fixture.channel = whatsapp.WhatsAppChannel(config, mock.AsyncMock(), mock.AsyncMock())

            async def post(url, **kwargs):
                payload = kwargs["json"]
                fixture.calls.append(("text" if url.endswith("/send") else "file", payload))
                return httpx.Response(
                    503 if len(fixture.calls) in fixture.reject else 200,
                    request=httpx.Request("POST", url),
                    json={"messageId": f"accepted-{len(fixture.calls)}"},
                )

            fixture.channel._http = SimpleNamespace(post=post)
            fixture.format_text = whatsapp.to_whatsapp
        else:
            fixture.module = telegram
            fixture.chat_id, fixture.thread_id = "42", "17"
            fixture.channel = telegram.TelegramChannel(config, mock.AsyncMock(), mock.AsyncMock())
            fixture.channel._reply_to_mode = "all"

            async def send(kind, kwargs):
                fixture.calls.append((kind, kwargs))
                if len(fixture.calls) in fixture.reject:
                    error = telegram.NetworkError("offline")
                    error.__cause__ = httpx.ConnectError("offline")
                    raise error
                return SimpleNamespace(message_id=len(fixture.calls))

            async def send_message(**kwargs):
                return await send("text", kwargs)

            async def send_document(**kwargs):
                return await send("file", kwargs)

            fixture.channel._bot = SimpleNamespace(send_message=send_message, send_document=send_document)
            fixture.format_text = telegram.to_telegram
        return fixture

    def record(self, fixture, content):
        self.counter += 1
        path = self.root / f"delivery-{self.counter}.db"
        store = DeliveryStore(path)
        oid = compute_obligation_id("session", str(self.counter), content)
        store.record(
            obligation_id=oid, session_key="session", platform=fixture.platform,
            chat_id=fixture.chat_id, thread_id=fixture.thread_id,
            reply_to="7", content=content,
        )
        return store, path, oid

    async def send(self, fixture, content, ledger):
        kwargs = {"thread_id": fixture.thread_id} if fixture.platform == "telegram" else {}
        return await fixture.channel.send(fixture.chat_id, content, "7", delivery_ledger=ledger, **kwargs)

    @staticmethod
    def rows(path, table):
        assert table in {"delivery_units", "delivery_obligations"}
        with closing(sqlite3.connect(path)) as connection:
            connection.row_factory = sqlite3.Row
            order = " ORDER BY position" if table == "delivery_units" else ""
            return [dict(row) for row in connection.execute(f"SELECT * FROM {table}{order}")]

    @staticmethod
    def text(call):
        return call[1].get("message", call[1].get("text"))

    def old_reply(self, fixture, *, long=False, attachment=False):
        business = ("📈" * 2100 + "\n\n" if long else "") + "**Revenue: 400 MAD**"
        report = fixture.channel._config.workspace_dir / "report.pdf"
        report.write_bytes(b"%PDF")
        if attachment:
            business += f"\nMEDIA:{report}"
        return media.confine_outbound(
            business + f"\nMEDIA:{self.root / 'unavailable.pdf'}",
            (fixture.channel._config.workspace_dir,), denied_notice=LEGACY_ATTACHMENT_NOTICE,
        )

    async def seed_old_plan(self, fixture, content, rejected_unit):
        store, path, oid = self.record(fixture, content)
        fixture.reject = {rejected_unit}
        # The unchanged original formatter/planner records the pre-upgrade units.
        with mock.patch.object(fixture.module, "attachment_notice_replacements", return_value={}):
            result = await self.send(fixture, content, DeliveryUnitLedger(store, oid))
        self.assertFalse(result)
        self.assertTrue(result.retryable)
        self.assertTrue(store.mark_planned_failed(oid, result.error, retry_safe=True))
        return store, path, oid

    async def test_old_storage_failures_are_retired_without_transport_or_identity_changes(self):
        for platform in ("whatsapp", "telegram"):
            fixture = self.channel(platform)
            for content in STORAGE_NOTICES:
                with self.subTest(platform=platform, content=content):
                    _, path, oid = self.record(fixture, content)
                    count = await recover_deliveries(DeliveryStore(path), {platform: fixture.channel})
                    self.assertEqual(count, 0)
                    self.assertEqual(fixture.calls, [])
                    row = self.rows(path, "delivery_obligations")[0]
                    self.assertEqual((row["obligation_id"], row["content"], row["state"]), (oid, content, "abandoned"))

    async def test_unplanned_mixed_reply_keeps_business_and_uses_profile_language(self):
        for platform in ("whatsapp", "telegram"):
            for language in SUPPORTED_LANGUAGES:
                with self.subTest(platform=platform, language=language):
                    fixture = self.channel(platform, language)
                    content = self.old_reply(fixture)
                    _, path, oid = self.record(fixture, content)
                    count = await recover_deliveries(DeliveryStore(path), {platform: fixture.channel})
                    self.assertEqual(count, 1)
                    self.assertEqual([kind for kind, _ in fixture.calls], ["text"])
                    written = self.text(fixture.calls[0])
                    self.assertIn("Revenue: 400 MAD", written)
                    self.assertIn(fixture.format_text(t("media.delivery_unavailable", language)), written)
                    self.assertNotIn("File delivery blocked", written)
                    self.assertEqual(self.rows(path, "delivery_obligations")[0]["content"], content)
                    self.assertEqual(self.rows(path, "delivery_units")[0]["state"], "delivered")
                    self.assertEqual(await recover_deliveries(DeliveryStore(path), {platform: fixture.channel}), 0)

    async def test_recovery_preserves_accepted_prefix_and_attachment_routing(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                fixture = self.channel(platform)
                content = self.old_reply(fixture, long=True, attachment=True)
                _, path, oid = await self.seed_old_plan(fixture, content, rejected_unit=2)
                original = self.rows(path, "delivery_units")
                self.assertEqual([row["state"] for row in original], ["delivered", "failed", "pending"])
                old_last = self.text(fixture.calls[-1])
                fixture.calls.clear()
                fixture.reject.clear()
                count = await recover_deliveries(DeliveryStore(path), {platform: fixture.channel})
                self.assertEqual(count, 1)
                self.assertEqual([kind for kind, _ in fixture.calls], ["text", "file"])
                expected = old_last.replace(
                    fixture.format_text(LEGACY_ATTACHMENT_NOTICE),
                    fixture.format_text(t("media.delivery_unavailable", "fr")),
                )
                self.assertEqual(self.text(fixture.calls[0]), expected)
                self.assertIn("Revenue: 400 MAD", expected)
                updated = self.rows(path, "delivery_units")
                self.assertEqual(updated[0], original[0])
                self.assertEqual([row["unit_id"] for row in updated], [row["unit_id"] for row in original])
                self.assertEqual(updated[2]["fingerprint"], original[2]["fingerprint"])
                if platform == "telegram":
                    for _, payload in fixture.calls:
                        self.assertEqual(payload["message_thread_id"], 17)
                        self.assertEqual(payload["reply_to_message_id"], 7)
                    self.assertIn(r"\(2/2\)", expected)
                else:
                    self.assertNotIn("replyTo", fixture.calls[0][1])
                    self.assertEqual(fixture.calls[1][1]["filePath"], str(fixture.channel._config.workspace_dir / "report.pdf"))
                self.assertEqual(self.rows(path, "delivery_obligations")[0]["obligation_id"], oid)

    async def test_already_accepted_notice_is_never_resent_while_recovering_a_file(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                fixture = self.channel(platform)
                content = self.old_reply(fixture, attachment=True)
                _, path, _ = await self.seed_old_plan(fixture, content, rejected_unit=2)
                before = self.rows(path, "delivery_units")[0]
                fixture.calls.clear()
                fixture.reject.clear()
                self.assertEqual(await recover_deliveries(DeliveryStore(path), {platform: fixture.channel}), 1)
                self.assertEqual([kind for kind, _ in fixture.calls], ["file"])
                self.assertEqual(self.rows(path, "delivery_units")[0], before)

    async def test_retry_after_language_change_keeps_the_recorded_plain_reply(self):
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                fixture = self.channel(platform)
                content = self.old_reply(fixture)
                _, path, oid = await self.seed_old_plan(fixture, content, rejected_unit=1)
                resumed = DeliveryStore(path)
                self.assertEqual(len(await claim_deliveries(resumed, {platform})), 1)
                fixture.calls.clear()
                result = await self.send(fixture, content, DeliveryUnitLedger(resumed, oid))
                self.assertFalse(result)
                self.assertTrue(result.retryable)
                plain_french = self.text(fixture.calls[0])
                self.assertNotIn("File delivery blocked", plain_french)
                self.assertTrue(resumed.mark_planned_failed(oid, result.error, retry_safe=True))
                before = self.rows(path, "delivery_units")[0]
                fixture.channel._config = replace(fixture.channel._config, language="en")
                fixture.calls.clear()
                fixture.reject.clear()
                self.assertEqual(await recover_deliveries(DeliveryStore(path), {platform: fixture.channel}), 1)
                self.assertEqual(self.text(fixture.calls[0]), plain_french)
                after = self.rows(path, "delivery_units")[0]
                self.assertEqual(after["unit_id"], before["unit_id"])
                self.assertEqual(after["fingerprint"], before["fingerprint"])
