from __future__ import annotations

import unittest

from pilotage.channels.telegram_formatting import (
    split_telegram_message,
    strip_telegram_markdown,
    to_telegram,
    utf16_len,
)


class TelegramFormattingTests(unittest.TestCase):
    def test_telegram_length_uses_utf16_code_units(self):
        self.assertEqual(utf16_len("plain"), 5)
        self.assertEqual(utf16_len("😀"), 2)

    def test_common_markdown_becomes_markdown_v2(self):
        formatted = to_telegram(
            "# Title\n**bold** and *italic*\n"
            "[site](https://example.com/a_(b)) and `x_y`"
        )

        self.assertIn("*Title*", formatted)
        self.assertIn("*bold*", formatted)
        self.assertIn("_italic_", formatted)
        self.assertIn("[site](https://example.com/a_(b\\))", formatted)
        self.assertIn("`x_y`", formatted)

    def test_tables_become_mobile_readable_row_groups(self):
        formatted = to_telegram(
            "| Name | Value |\n"
            "| --- | --- |\n"
            "| Alpha | 1 |\n"
            "| Beta | 2 |"
        )

        self.assertIn("*Alpha*", formatted)
        self.assertIn("• Value: 1", formatted)
        self.assertIn("*Beta*", formatted)
        self.assertNotIn("| --- |", formatted)

    def test_fenced_code_is_not_rewritten_as_a_table(self):
        formatted = to_telegram(
            "```text\n"
            "| A | B |\n"
            "|---|---|\n"
            "```"
        )

        self.assertIn("| A | B |", formatted)
        self.assertIn("|---|---|", formatted)

    def test_plaintext_fallback_removes_telegram_marks(self):
        self.assertEqual(
            strip_telegram_markdown(r"\(notice\) *bold* \!"),
            "(notice) bold !",
        )

    def test_long_messages_are_chunked_with_code_fences_preserved(self):
        content = "```python\n" + "print('hello')\n" * 80 + "```"
        chunks = split_telegram_message(content, max_length=300)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(utf16_len(chunk) <= 300 for chunk in chunks))
        self.assertTrue(all(chunk.count("```") % 2 == 0 for chunk in chunks))
        self.assertIn(r"\(1/", chunks[0])

    def test_emoji_messages_respect_the_real_telegram_limit(self):
        chunks = split_telegram_message("😀" * 2500)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(utf16_len(chunk) <= 4096 for chunk in chunks))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
