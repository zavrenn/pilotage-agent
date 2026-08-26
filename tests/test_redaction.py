"""Secret redaction follows the current Hermes log and terminal contract."""

from __future__ import annotations

import ast
import concurrent.futures
import logging
import tempfile
import unittest
from pathlib import Path

from pilotage.redact import (
    RedactingFormatter,
    configure_identity_pseudonyms,
    identity_pseudonym,
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

    def test_channel_identities_become_stable_profile_keyed_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            configure_identity_pseudonyms(Path(directory))
            first = identity_pseudonym("212600123456@s.whatsapp.net", "wa")
            second = identity_pseudonym("212600123456@s.whatsapp.net", "wa")
            telegram = identity_pseudonym("987654321", "tg")

        self.assertEqual(first, second)
        self.assertNotEqual(first, telegram)
        self.assertNotIn("212600123456", first)
        self.assertNotIn("987654321", telegram)

    def test_concurrent_key_initialization_cannot_rotate_the_profile_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(
                    pool.map(
                        configure_identity_pseudonyms,
                        [state, state, state, state],
                    )
                )
            key = paths[0].read_bytes()

        self.assertTrue(all(path == paths[0] for path in paths))
        self.assertEqual(len(key), 32)

    def test_formatter_pseudonymizes_jids_and_labeled_telegram_ids(self):
        jid = "212600123456@s.whatsapp.net"
        telegram_id = "987654321"
        record = logging.LogRecord(
            "pilotage.test",
            logging.ERROR,
            __file__,
            1,
            "failed for %s with chat_id=%s",
            (jid, telegram_id),
            None,
        )

        rendered = RedactingFormatter("%(message)s").format(record)

        self.assertNotIn(jid, rendered)
        self.assertNotIn(telegram_id, rendered)
        self.assertIn("[wa:", rendered)
        self.assertIn("[tg:", rendered)

    def test_routine_log_calls_never_receive_raw_channel_identity_variables(self):
        root = Path(__file__).resolve().parent.parent / "pilotage"
        paths = [
            root / "agent.py",
            root / "delivery.py",
            root / "history.py",
            root / "main.py",
            *sorted((root / "channels").glob("*.py")),
            *sorted((root / "cron").glob("*.py")),
        ]
        sensitive = {
            "chat_id",
            "sender_id",
            "sender_number",
            "session_id",
            "thread_id",
        }
        failures = []

        def raw_references(node):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "identity_pseudonym"
            ):
                return []
            found = []
            if isinstance(node, ast.Name) and node.id in sensitive:
                found.append(node.id)
            elif isinstance(node, ast.Attribute) and node.attr in sensitive:
                found.append(node.attr)
            for child in ast.iter_child_nodes(node):
                found.extend(raw_references(child))
            return found

        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "logger"
                ):
                    continue
                raw = []
                for argument in [*node.args, *[item.value for item in node.keywords]]:
                    raw.extend(raw_references(argument))
                if raw:
                    failures.append(
                        f"{path.relative_to(root.parent)}:{node.lineno}: {sorted(set(raw))}"
                    )

        self.assertEqual(failures, [])


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
