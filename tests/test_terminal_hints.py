"""Conservative detection of shell suffixes that mask build/test failures."""

from __future__ import annotations

import unittest

from pilotage.tools.terminal_hints import masked_success_advisory


class MaskedSuccessAdvisoryTests(unittest.TestCase):
    def test_build_piped_to_tail_is_flagged(self):
        hint = masked_success_advisory(
            "cargo build --release 2>&1 | tail -20",
            "error[E0308]: mismatched types\nerror: could not compile `app`\n",
        )
        self.assertIn("last pipeline command", hint or "")

    def test_test_fallback_is_flagged(self):
        hint = masked_success_advisory(
            "pytest tests || true",
            "FAILED tests/test_app.py::test_run\n3 failed in 1.2s\n",
        )
        self.assertIn("`||` fallback", hint or "")

    def test_bare_failure_output_is_not_second_guessed(self):
        self.assertIsNone(
            masked_success_advisory(
                "cargo build",
                "error: could not compile `app`\n",
            )
        )

    def test_clean_pipeline_is_not_flagged(self):
        self.assertIsNone(
            masked_success_advisory(
                "cargo build | tail -20",
                "Compiling app\nFinished release\n",
            )
        )

    def test_search_pipeline_is_not_flagged(self):
        self.assertIsNone(
            masked_success_advisory(
                "rg 'could not compile' logs | head -20",
                "old.log:error: could not compile `app`\n",
            )
        )

    def test_generic_error_word_is_not_a_failure_signal(self):
        self.assertIsNone(
            masked_success_advisory(
                "make install | tail -20",
                "checking error handling support... yes\nInstall complete\n",
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
