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
        self.assertEqual(t("runtime.failure", "en"), "I couldn't answer just now.")
        self.assertEqual(t("runtime.failure", "fr"), "Je n'ai pas pu répondre pour le moment.")
        self.assertEqual(t("runtime.failure", "ar"), "تعذّر عليّ الرد الآن.")
        for language in SUPPORTED_LANGUAGES:
            self.assertIn(
                "/new",
                t("runtime.storage_failure", language),
            )
            self.assertIn(
                "/new",
                t("runtime.interrupted_unknown", language),
            )
        self.assertIn(
            "memory",
            t("approval.required", "ar", category="memory"),
        )

    def test_uncertain_outcomes_keep_recovery_without_internal_details_or_retry(self):
        for language, uncertainty, check_result in (
            ("en", "confirm", "Check the result"),
            ("fr", "confirmer", "Vérifiez le résultat"),
            ("ar", "التأكد", "تحقّق من النتيجة"),
        ):
            for key in (
                "runtime.storage_failure", "runtime.interrupted_unknown", "commands.stop_unknown",
            ):
                with self.subTest(language=language, key=key):
                    message = t(key, language)
                    self.assertIn(uncertainty, message)
                    self.assertIn(check_result, message)
                    self.assertIn("/new", message)
                    for technical_or_retry in (
                        "storage", "state", "tool", "stockage", "état", "outil",
                        "التخزين", "الأداة", "try again", "Réessayez", "حاول مرة أخرى",
                    ):
                        self.assertNotIn(technical_or_retry, message)

    def test_incomplete_replies_preserve_possible_completed_actions(self):
        for language, possible_actions in (
            ("en", "Some actions may already be complete"),
            ("fr", "Certaines actions ont peut-être déjà été effectuées"),
            ("ar", "قد تكون بعض الإجراءات قد نُفّذت بالفعل"),
        ):
            with self.subTest(language=language):
                message = t("runtime.incomplete_response", language)
                self.assertIn(possible_actions, message)
                self.assertNotIn("Codex", message)
                self.assertNotIn("continuation", message)

    def test_new_client_notices_are_localized_in_each_catalog(self):
        for key in (
            "media.delivery_unavailable", "cron.failure", "cron.unavailable",
            "capability.unavailable", "commands.ready", "commands.description_help",
            "commands.description_new", "commands.description_stop", "commands.description_status",
        ):
            with self.subTest(key=key):
                notices = {t(key, language) for language in SUPPORTED_LANGUAGES}
                self.assertNotIn(key, notices)
                self.assertEqual(len(notices), len(SUPPORTED_LANGUAGES))

    def test_missing_key_degrades_to_the_key_without_crashing(self):
        self.assertEqual(t("missing.key", "fr"), "missing.key")


if __name__ == "__main__":
    unittest.main()
