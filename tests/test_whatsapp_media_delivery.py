"""Native WhatsApp delivery for files produced by skills."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from pilotage import media
from pilotage.channels.whatsapp import InboundMessage, WhatsAppChannel
from pilotage.config import Config, WHATSAPP_MEDIA_NOTE


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
            "chat", f"**Chart ready.**\nMEDIA:{chart}", "m7"
        )

        self.assertTrue(sent)
        self.assertEqual(len(self.http.posts), 2)
        self.assertTrue(self.http.posts[0]["url"].endswith("/send"))
        self.assertEqual(
            self.http.posts[0]["json"],
            {"chatId": "chat", "message": "*Chart ready.*", "replyTo": "m7"},
        )
        self.assertTrue(self.http.posts[1]["url"].endswith("/send-media"))
        self.assertEqual(
            self.http.posts[1]["json"],
            {"chatId": "chat", "filePath": str(chart.resolve()), "mediaType": "image"},
        )
        self.assertEqual(self.http.posts[1]["timeout"], 120.0)

    async def test_media_only_document_is_sent_without_empty_text(self):
        report = self.config.workspace_dir / "report.xlsx"
        report.write_bytes(b"xlsx")

        self.assertTrue(await self.channel.send("chat", f"MEDIA:{report}"))

        self.assertEqual(len(self.http.posts), 1)
        self.assertTrue(self.http.posts[0]["url"].endswith("/send-media"))
        self.assertEqual(
            self.http.posts[0]["json"],
            {
                "chatId": "chat",
                "filePath": str(report.resolve()),
                "mediaType": "document",
                "fileName": "report.xlsx",
            },
        )

    async def test_plain_text_echo_cannot_turn_media_text_into_an_attachment(self):
        report = self.config.workspace_dir / "report.xlsx"
        report.write_bytes(b"xlsx")

        self.assertTrue(
            await self.channel.send(
                "chat",
                f"MEDIA:{report}",
                deliver_media=False,
            )
        )

        self.assertEqual(len(self.http.posts), 1)
        self.assertTrue(self.http.posts[0]["url"].endswith("/send"))

if __name__ == "__main__":
    unittest.main()
