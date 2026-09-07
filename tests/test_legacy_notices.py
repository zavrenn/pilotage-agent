"""The upgrade filter recognizes complete runtime notices, not business text."""

from pathlib import Path
from types import SimpleNamespace
import unittest

from pilotage.commands import help_text, profile_text, status_text
from pilotage.channels.whatsapp import split_whatsapp_message, to_whatsapp
from pilotage.channels.telegram_formatting import split_telegram_message, to_telegram
from pilotage.i18n import SUPPORTED_LANGUAGES, t
from pilotage.legacy_notices import (
    LEGACY_ATTACHMENT_NOTICE, attachment_notice_replacements, is_legacy_technical_reply,
)
from pilotage.settings import Settings


class LegacyNoticeTests(unittest.TestCase):
    def test_old_diagnostics_are_recognized_in_every_profile_language(self):
        config = SimpleNamespace(
            credentials_path=Path("unavailable-primary-auth"),
            main_credentials_path=Path("unavailable-shared-auth"),
            state_dir=Path("/private/client-state"), settings=Settings({}),
            model="operator-model", cron_enabled=True, cron_timezone="UTC",
        )
        for language in SUPPORTED_LANGUAGES:
            config.language = language
            for content in (profile_text(config, "client"), status_text(config, "client")):
                with self.subTest(language=language, content=content):
                    self.assertTrue(is_legacy_technical_reply(content))
                    self.assertFalse(is_legacy_technical_reply("Business report:\n" + content))
                    self.assertFalse(is_legacy_technical_reply(content + "\nRevenue: 400 MAD"))

    def test_only_complete_old_failures_are_recognized(self):
        for content in (
            "Codex response remained incomplete after 3 continuation attempts",
            "Scheduled job 'Morning report' failed. Check the agent logs.",
        ):
            self.assertTrue(is_legacy_technical_reply(content))
            self.assertFalse(is_legacy_technical_reply(f'Example: "{content}"'))

    def test_business_replies_and_current_notices_are_preserved(self):
        for content in (
            "Revenue: 400 MAD", "Approved. I’ll continue.",
            "Profile: Standard\nState: Confirmed\nRevenue: 400 MAD",
            "Codex sales: 400 MAD", "MEDIA:/workspace/report.pdf",
            "Report complete.\n\n[File delivery blocked: restricted sessions can only "
            "deliver files from the current session's exports directory.]",
        ):
            self.assertFalse(is_legacy_technical_reply(content))
        for language in SUPPORTED_LANGUAGES:
            self.assertFalse(is_legacy_technical_reply(help_text(language)))
            for key in ("runtime.incomplete_response", "capability.unavailable", "cron.failure"):
                self.assertFalse(is_legacy_technical_reply(t(key, language)))

    def test_attachment_replacement_keeps_original_split_even_when_clean_reply_fits(self):
        for format_text, split in ((to_whatsapp, split_whatsapp_message), (to_telegram, split_telegram_message)):
            with self.subTest(format=format_text.__name__):
                business = "A" * 4000
                content = business + "\n\n" + LEGACY_ATTACHMENT_NOTICE
                chunks = split(format_text(content))
                self.assertEqual(len(chunks), 2)
                self.assertEqual(len(split(format_text(
                    business + "\n\n" + t("media.delivery_unavailable", "fr")
                ))), 1)
                choices = attachment_notice_replacements(content, chunks, format_text, "fr")
                self.assertEqual(set(choices), {1})
                self.assertEqual(chunks[0], split(format_text(content))[0])
                self.assertTrue(all("File delivery blocked" not in candidate for candidate in choices[1]))
                if format_text is to_telegram:
                    self.assertTrue(all(candidate.endswith(r"\(2/2\)") for candidate in choices[1]))

    def test_attachment_replacement_ignores_quoted_and_nonterminal_examples(self):
        for content in (
            f'Example: "{LEGACY_ATTACHMENT_NOTICE}"',
            LEGACY_ATTACHMENT_NOTICE + "\n\nRevenue: 400 MAD",
        ):
            self.assertEqual(attachment_notice_replacements(content, [content], str, "fr"), {})

    def test_fragmented_attachment_notice_is_refused_without_changing_chunks(self):
        content = "Revenue: 400 MAD\n\n" + LEGACY_ATTACHMENT_NOTICE
        chunks = [content[:-30], content[-30:]]
        before = list(chunks)
        with self.assertRaises(ValueError):
            attachment_notice_replacements(content, chunks, str, "fr")
        self.assertEqual(chunks, before)

    def test_old_help_with_retired_commands_is_recognized(self):
        content = (
            "Management commands:\n"
            "/help — Show the available management commands (alias: /commands)\n"
            "/new — Start a fresh conversation (alias: /reset)\n"
            "/stop — Stop the active request\n"
            "/approve — Allow the oldest pending change once\n"
            "/deny — Refuse the oldest pending change\n"
            "/status — Show the running agent's essential status\n"
            "/profile — Show the active profile and state directory"
        )
        self.assertTrue(is_legacy_technical_reply(content))
        self.assertFalse(is_legacy_technical_reply(content + "\nRevenue: 400 MAD"))
