"""Native WhatsApp delivery for files produced by skills."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import httpx

from pilotage import media
from pilotage.channels.whatsapp import InboundMessage, WhatsAppChannel
from pilotage.config import Config, WHATSAPP_MEDIA_NOTE


CHAT_ID = "212600000000@s.whatsapp.net"


class FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        pass


class FakeHttp:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.posts.append(
            {
                "url": url,
                "json": kwargs.get("json") or {},
                "timeout": kwargs.get("timeout"),
            }
        )
        return FakeResponse()


async def _handle(_message: InboundMessage) -> None:  # pragma: no cover
    raise AssertionError("no turn should run")


async def _command(*_args: Any) -> None:  # pragma: no cover
    raise AssertionError("no command should run")


class OutboundExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def test_valid_workspace_file_is_extracted_and_directive_is_hidden(self):
        chart = self.workspace / "chart.png"
        chart.write_bytes(b"png")

        attachments, cleaned = media.extract_outbound(
            f"Ready\nMEDIA:{chart}\nDone", (self.workspace,)
        )

        self.assertEqual(cleaned, "Ready\n\nDone")
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].path, chart.resolve())
        self.assertEqual(attachments[0].media_type, "image")

    def test_quoted_path_with_spaces_matches_demo_skill_output(self):
        report = self.workspace / "weekly report.pdf"
        report.write_bytes(b"pdf")

        attachments, cleaned = media.extract_outbound(
            f'Attached\nMEDIA:"{report}"', (self.workspace,)
        )

        self.assertEqual(cleaned, "Attached")
        self.assertEqual([item.path for item in attachments], [report.resolve()])

    def test_file_outside_workspace_is_never_delivered(self):
        outside = self.root / "outside.pdf"
        outside.write_bytes(b"pdf")

        attachments, cleaned = media.extract_outbound(
            f"Done\nMEDIA:{outside}", (self.workspace,)
        )

        self.assertEqual(attachments, [])
        self.assertEqual(cleaned, "Done")

    def test_fenced_example_is_not_a_live_attachment(self):
        chart = self.workspace / "chart.png"
        chart.write_bytes(b"png")
        answer = f"Example:\n```text\nMEDIA:{chart}\n```"

        attachments, cleaned = media.extract_outbound(answer, (self.workspace,))

        self.assertEqual(attachments, [])
        self.assertEqual(cleaned, answer)

    def test_json_tool_output_is_not_replayed_as_an_attachment(self):
        chart = self.workspace / "chart.png"
        chart.write_bytes(b"png")
        answer = f'{{"result": "MEDIA:{chart}"}}'

        attachments, cleaned = media.extract_outbound(answer, (self.workspace,))

        self.assertEqual(attachments, [])
        self.assertEqual(cleaned, answer)

    def test_duplicate_directive_uploads_once(self):
        report = self.workspace / "report.xlsx"
        report.write_bytes(b"xlsx")

        attachments, cleaned = media.extract_outbound(
            f"MEDIA:{report}\nMEDIA:{report}", (self.workspace,)
        )

        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].media_type, "document")
        self.assertEqual(cleaned, "")

    def test_restricted_filter_keeps_current_exports_and_refuses_other_files(self):
        exports = self.workspace / "session-2" / "exports"
        exports.mkdir(parents=True)
        current = exports / "current.pdf"
        current.write_bytes(b"current")
        outside = self.workspace / "old-session.pdf"
        outside.write_bytes(b"old")

        confined = media.confine_outbound(
            f"Ready\nMEDIA:{current}\nMEDIA:{outside}",
            (exports,),
        )

        self.assertIn(f"MEDIA:{current.resolve()}", confined)
        self.assertNotIn(str(outside), confined)
        self.assertIn("File delivery blocked", confined)
        attachments, cleaned = media.extract_outbound(confined, (exports,))
        self.assertEqual(
            [attachment.path for attachment in attachments],
            [current.resolve()],
        )
        self.assertIn("File delivery blocked", cleaned)


class InboundLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_inbound_file_is_copied_into_session_inputs(self):
        cached = self.root / "cache" / "report.pdf"
        cached.parent.mkdir()
        cached.write_bytes(b"report")
        attachment = media.Attachment(
            path=cached,
            mime="application/pdf",
            media_type="document",
            file_name="Quarterly report.pdf",
        )

        staged = media.stage_inbound(
            [attachment],
            self.root / "session" / "inputs",
        )

        self.assertEqual(staged[0].path.read_bytes(), b"report")
        self.assertEqual(staged[0].file_name, "Quarterly report.pdf")
        self.assertNotEqual(staged[0].path, cached)
        self.assertEqual(
            staged[0].path.parent,
            (self.root / "session" / "inputs").resolve(),
        )

    def test_cache_cleanup_removes_only_files_older_than_one_day(self):
        cache = self.root / "media"
        cache.mkdir()
        old = cache / "old.bin"
        fresh = cache / "fresh.bin"
        old.write_bytes(b"old")
        fresh.write_bytes(b"fresh")
        now = 200_000.0
        os.utime(old, (now - 90_000, now - 90_000))
        os.utime(fresh, (now - 10, now - 10))

        removed = media.cleanup_cache(cache, now=now)

        self.assertEqual(removed, 1)
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists())


class WhatsAppMediaSendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(
            os.environ,
            {
                "PILOTAGE_HOME": str(self.root),
                "PILOTAGE_ALLOWED_SENDERS": "212600000000",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.config = Config.load(channel="whatsapp")
        self.config.workspace_dir.mkdir(parents=True)
        self.channel = WhatsAppChannel(self.config, _handle, _command)
        self.http = FakeHttp()
        self.channel._http = self.http

    def test_whatsapp_prompt_carries_hermes_media_contract(self):
        self.assertIn(WHATSAPP_MEDIA_NOTE, self.config.instructions)
        self.assertNotIn(WHATSAPP_MEDIA_NOTE, Config.load().instructions)

    async def test_text_is_sent_before_native_image(self):
        chart = self.config.workspace_dir / "chart.png"
        chart.write_bytes(b"png")

        sent = await self.channel.send(
            CHAT_ID, f"**Chart ready.**\nMEDIA:{chart}", "m7"
        )

        self.assertTrue(sent)
        self.assertEqual(len(self.http.posts), 2)
        self.assertTrue(self.http.posts[0]["url"].endswith("/send"))
        self.assertEqual(
            self.http.posts[0]["json"],
            {"chatId": CHAT_ID, "message": "*Chart ready.*", "replyTo": "m7"},
        )
        self.assertTrue(self.http.posts[1]["url"].endswith("/send-media"))
        self.assertEqual(
            self.http.posts[1]["json"],
            {
                "chatId": CHAT_ID,
                "filePath": str(chart.resolve()),
                "mediaType": "image",
            },
        )
        self.assertEqual(self.http.posts[1]["timeout"], 120.0)

    async def test_media_only_document_is_sent_without_empty_text(self):
        report = self.config.workspace_dir / "report.xlsx"
        report.write_bytes(b"xlsx")

        self.assertTrue(await self.channel.send(CHAT_ID, f"MEDIA:{report}"))

        self.assertEqual(len(self.http.posts), 1)
        self.assertTrue(self.http.posts[0]["url"].endswith("/send-media"))
        self.assertEqual(
            self.http.posts[0]["json"],
            {
                "chatId": CHAT_ID,
                "filePath": str(report.resolve()),
                "mediaType": "document",
                "fileName": "report.xlsx",
            },
        )

    async def test_operator_declared_directory_is_deliverable(self):
        reports = self.root / "reports"
        reports.mkdir()
        (self.root / "config.yaml").write_text(
            "gateway:\n"
            f"  media_delivery_allow_dirs: ['{reports.as_posix()}']\n",
            encoding="utf-8",
        )
        config = Config.load(channel="whatsapp")
        channel = WhatsAppChannel(config, _handle, _command)
        channel._http = self.http
        report = reports / "outside-workspace.pdf"
        report.write_bytes(b"pdf")

        self.assertTrue(await channel.send(CHAT_ID, f"MEDIA:{report}"))
        self.assertEqual(
            self.http.posts[0]["json"]["filePath"],
            str(report.resolve()),
        )

    async def test_plain_text_echo_cannot_turn_media_text_into_an_attachment(self):
        report = self.config.workspace_dir / "report.xlsx"
        report.write_bytes(b"xlsx")

        self.assertTrue(
            await self.channel.send(
                CHAT_ID,
                f"MEDIA:{report}",
                deliver_media=False,
            )
        )

        self.assertEqual(len(self.http.posts), 1)
        self.assertTrue(self.http.posts[0]["url"].endswith("/send"))

    async def test_bridge_500_status_error_is_not_retryable(self):
        request = httpx.Request("POST", "http://127.0.0.1:8123/send")
        response = httpx.Response(500, request=request)
        self.channel._http = mock.Mock(
            post=mock.AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "bridge 500",
                    request=request,
                    response=response,
                )
            )
        )

        sent = await self.channel.send(CHAT_ID, "hello")

        self.assertFalse(sent)
        self.assertFalse(sent.retryable)

    async def test_bridge_503_status_error_is_retryable(self):
        request = httpx.Request("POST", "http://127.0.0.1:8123/send")
        response = httpx.Response(503, request=request)
        self.channel._http = mock.Mock(
            post=mock.AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "bridge 503",
                    request=request,
                    response=response,
                )
            )
        )

        sent = await self.channel.send(CHAT_ID, "hello")

        self.assertFalse(sent)
        self.assertTrue(sent.retryable)

    async def test_bare_home_number_is_normalized_before_delivery(self):
        sent = await self.channel.send("+212 600 000 000", "hello")

        self.assertTrue(sent)
        self.assertEqual(self.http.posts[0]["json"]["chatId"], CHAT_ID)

    async def test_legacy_direct_jid_is_normalized_before_delivery(self):
        sent = await self.channel.send("212600000000@c.us", "hello")

        self.assertTrue(sent)
        self.assertEqual(self.http.posts[0]["json"]["chatId"], CHAT_ID)

    async def test_plus_prefixed_direct_jid_is_normalized_before_delivery(self):
        sent = await self.channel.send("+212600000000@s.whatsapp.net", "hello")

        self.assertTrue(sent)
        self.assertEqual(self.http.posts[0]["json"]["chatId"], CHAT_ID)

    async def test_invalid_home_target_fails_before_bridge_delivery(self):
        sent = await self.channel.send("not-a-chat", "hello")

        self.assertFalse(sent)
        self.assertEqual(self.http.posts, [])

if __name__ == "__main__":
    unittest.main()
