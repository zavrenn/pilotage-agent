"""Secret redaction follows the current Hermes log and terminal contract."""

from __future__ import annotations

import logging
import unittest

from pilotage.redact import (
    RedactingFormatter,
    is_env_dump_command,
    redact_sensitive_text,
    redact_terminal_output,
)


class SensitiveTextTests(unittest.TestCase):
    def test_known_provider_token_is_masked(self):
        secret = "sk-proj-abcdefghijklmnopqrstuv"
        redacted = redact_sensitive_text(f"request failed for {secret}")
        self.assertNotIn(secret, redacted)
        self.assertIn("sk-pro...stuv", redacted)

    def test_opaque_secret_assignments_and_json_fields_are_masked(self):
        opaque = "opaque-value-1234567890"
        for text in (
            f"MY_SERVICE_TOKEN={opaque}",
            f'{{"refresh_token": "{opaque}"}}',
            f"database_password: {opaque}",
        ):
            with self.subTest(text=text):
                self.assertNotIn(opaque, redact_sensitive_text(text))

    def test_auth_headers_telegram_tokens_and_database_passwords_are_masked(self):
        telegram = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
        bearer = "opaque-bearer-value-1234567890"
        password = "database-password-123"
        text = (
            f"Authorization: Bearer {bearer}\n"
            f"https://api.telegram.org/bot{telegram}/sendMessage\n"
            f"postgresql://pilotage:{password}@db.internal/pilotage"
        )

        redacted = redact_sensitive_text(text)

        self.assertNotIn(bearer, redacted)
        self.assertNotIn(telegram, redacted)
        self.assertNotIn(password, redacted)

    def test_private_keys_jwts_and_e164_numbers_are_masked(self):
        private_key = (
            "-----BEGIN PRIVATE KEY-----\n"
            "opaque-private-key-material\n"
            "-----END PRIVATE KEY-----"
        )
        jwt = "eyJabcdefghijklmnop.abcdefghijklmnop.signature1234"
        phone = "+212600123456"

        redacted = redact_sensitive_text(f"{private_key}\n{jwt}\n{phone}")

        self.assertNotIn("opaque-private-key-material", redacted)
        self.assertNotIn(jwt, redacted)
        self.assertNotIn(phone, redacted)

    def test_normal_text_and_navigation_urls_remain_usable(self):
        text = (
            "Secretary: J.Smith\n"
            "tokenizer: cl100k_base\n"
            "https://example.test/callback?code=round-trip-value&state=ok"
        )
        self.assertEqual(redact_sensitive_text(text), text)

    def test_formatter_redacts_after_log_arguments_are_rendered(self):
        secret = "fc-abcdefghijklmnopqrst"
        record = logging.LogRecord(
            "pilotage.test",
            logging.ERROR,
            __file__,
            1,
            "Firecrawl failed with %s",
            (secret,),
            None,
        )

        rendered = RedactingFormatter("%(levelname)s %(message)s").format(record)

        self.assertNotIn(secret, rendered)
        self.assertIn("ERROR Firecrawl failed", rendered)


class TerminalRedactionTests(unittest.TestCase):
    def test_environment_dump_masks_an_opaque_value(self):
        secret = "opaque-value-1234567890"
        output = redact_terminal_output(
            f"PATH=/usr/bin\nMY_SERVICE_TOKEN={secret}\n",
            "printenv",
        )
        self.assertNotIn(secret, output)
        self.assertIn("PATH=/usr/bin", output)

    def test_env_file_reads_use_assignment_redaction(self):
        secret = "opaque-value-1234567890"
        output = redact_terminal_output(
            f"DATABASE_PASSWORD={secret}\n",
            "cat .env",
        )
        self.assertNotIn(secret, output)

    def test_regular_source_output_avoids_assignment_false_positives(self):
        source = 'MAX_TOKENS=100\nconfig = {"apiKey": "test-fixture"}\n'
        self.assertEqual(redact_terminal_output(source, "cat settings.py"), source)

    def test_environment_command_detection_handles_pipelines_and_sequences(self):
        self.assertTrue(is_env_dump_command("pwd; printenv | sort"))
        self.assertFalse(is_env_dump_command("python report.py"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
