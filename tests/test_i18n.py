"""The small static-message language boundary."""

from __future__ import annotations

import unittest

from pilotage.i18n import SUPPORTED_LANGUAGES, normalize_language, t


class I18nTests(unittest.TestCase):
    def test_production_languages_and_common_bcp47_variants_are_supported(self):
        self.assertEqual(SUPPORTED_LANGUAGES, ("en", "fr", "ar"))
        self.assertEqual(normalize_language("en-GB"), "en")
        self.assertEqual(normalize_language("fr_CA"), "fr")
        self.assertEqual(normalize_language("ar-MA"), "ar")
        self.assertEqual(normalize_language("français"), "fr")

    def test_unknown_language_is_an_operator_error(self):
        with self.assertRaisesRegex(ValueError, "display.language"):
            normalize_language("xx")

    def test_each_catalog_serves_runtime_messages_and_formats_values(self):
        self.assertIn("try again", t("runtime.failure", "en"))
        self.assertIn("Réessayez", t("runtime.failure", "fr"))
        self.assertIn("حاول", t("runtime.failure", "ar"))
        self.assertIn(
            "memory",
            t("approval.required", "ar", category="memory"),
        )

    def test_missing_key_degrades_to_the_key_without_crashing(self):
        self.assertEqual(t("missing.key", "fr"), "missing.key")


if __name__ == "__main__":
    unittest.main()
