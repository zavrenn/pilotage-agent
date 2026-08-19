"""What the model writes, and what WhatsApp is given.

Two halves of one fix, and each is useless alone: the reply is translated on
the way out, and the model is told that it will be. The cases that matter are
the ones where translating goes too far — a bullet, a formula, a line of code.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from pilotage.channels.formatting import to_whatsapp
from pilotage.config import DEFAULT_INSTRUCTIONS, FORMATTING_NOTE, Config


class EmphasisTests(unittest.TestCase):
    def test_bold_becomes_a_single_asterisk(self):
        self.assertEqual(to_whatsapp("Sales are **up**."), "Sales are *up*.")

    def test_italic_becomes_an_underscore(self):
        self.assertEqual(to_whatsapp("That is *roughly* right."), "That is _roughly_ right.")

    def test_bold_and_italic_together_keep_their_meaning(self):
        """The two collide: one asterisk means italic here and bold there."""
        self.assertEqual(to_whatsapp("**Firm** but *soft*."), "*Firm* but _soft_.")

    def test_strikethrough_loses_one_tilde(self):
        self.assertEqual(to_whatsapp("~~cancelled~~"), "~cancelled~")

    def test_underscore_italic_is_already_right(self):
        self.assertEqual(to_whatsapp("_already_"), "_already_")

    def test_a_double_underscore_name_is_left_alone(self):
        """__init__ is a word in a technical answer, not an emphasis mark."""
        self.assertEqual(to_whatsapp("Call __init__ first."), "Call __init__ first.")


class StructureTests(unittest.TestCase):
    def test_a_heading_becomes_a_bold_line(self):
        self.assertEqual(to_whatsapp("## Results\nGood."), "*Results*\nGood.")

    def test_a_bold_heading_is_not_bolded_twice(self):
        self.assertEqual(to_whatsapp("# **Results**"), "*Results*")

    def test_a_link_keeps_its_address(self):
        self.assertEqual(
            to_whatsapp("See [the report](http://x.y/a)."),
            "See the report (http://x.y/a).",
        )

    def test_a_bullet_list_is_not_read_as_italic(self):
        self.assertEqual(to_whatsapp("* one\n* two"), "* one\n* two")


class CodeTests(unittest.TestCase):
    def test_inline_code_is_untouched(self):
        self.assertEqual(to_whatsapp("Run `a **b** c`."), "Run `a **b** c`.")

    def test_a_fenced_block_is_untouched(self):
        source = "```py\nx = **2**\n# not a heading\n```"
        self.assertEqual(to_whatsapp(source), source)

    def test_text_around_a_fence_is_still_converted(self):
        self.assertEqual(to_whatsapp("```\nraw\n```\n**after**"), "```\nraw\n```\n*after*")


class SanitizingTests(unittest.TestCase):
    def test_invisible_characters_are_dropped(self):
        """WhatsApp shows them as a blob instead of hiding them."""
        self.assertEqual(to_whatsapp("a\u2060\u200bb"), "ab")

    def test_odd_spaces_become_ordinary_ones(self):
        self.assertEqual(to_whatsapp("a\u202fb"), "a b")

    def test_a_forged_placeholder_cannot_survive(self):
        """The parked code is restored by marker, so the text must not carry one."""
        self.assertEqual(to_whatsapp("\x000\x00 `real`"), "0 `real`")

    def test_empty_text_stays_empty(self):
        self.assertEqual(to_whatsapp(""), "")


class InstructionTests(unittest.TestCase):
    """A converter nobody told the model about converts nothing."""

    def test_the_note_is_there_by_default(self):
        self.assertIn(FORMATTING_NOTE, Config.load().instructions)

    def test_custom_instructions_cannot_drop_the_note(self):
        with mock.patch.dict(os.environ, {"PILOTAGE_INSTRUCTIONS": "Be terse."}):
            instructions = Config.load().instructions
        self.assertTrue(instructions.startswith("Be terse."))
        self.assertIn(FORMATTING_NOTE, instructions)

    def test_emptied_instructions_fall_back_and_keep_the_note(self):
        with mock.patch.dict(os.environ, {"PILOTAGE_INSTRUCTIONS": "   "}):
            instructions = Config.load().instructions
        self.assertTrue(instructions.startswith(DEFAULT_INSTRUCTIONS))
        self.assertIn(FORMATTING_NOTE, instructions)


if __name__ == "__main__":
    unittest.main()
