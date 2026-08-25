"""Profile-scoped Telegram setup without starting the agent runtime."""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx

from pilotage import main
from pilotage.main import command_telegram_setup
from pilotage.settings import Settings


TOKEN = "123456789:" + "A" * 35
NEW_TOKEN = "987654321:" + "b" * 35


def _accepted_bot(username: str = "pilotage_bot") -> SimpleNamespace:
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"ok": True, "result": {"username": username}},
    )


class TelegramSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env_path = self.root / ".env"
        self.settings_path = self.root / "config.yaml"
        self.settings_path.write_text(
            "# operator comment\n"
            "whatsapp:\n"
            "  enabled: false\n"
            "telegram:\n"
            "  enabled: false\n",
            encoding="utf-8",
        )
        environment = mock.patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "",
                "TELEGRAM_ALLOWED_USERS": "",
                "TELEGRAM_HOME_CHANNEL": "",
                "TELEGRAM_HOME_CHANNEL_THREAD_ID": "",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _run(self, *, external_env: frozenset[str] = frozenset()) -> int:
        return command_telegram_setup(
            self.root,
            env_path=self.env_path,
            settings_path=self.settings_path,
            external_env=external_env,
        )

    def test_new_setup_saves_verifies_and_enables_only_telegram(self):
        output, error = StringIO(), StringIO()
        with (
            mock.patch("pilotage.main.getpass.getpass", return_value=TOKEN),
            mock.patch("builtins.input", side_effect=["42,43", ""]),
            mock.patch("pilotage.main.httpx.get", return_value=_accepted_bot()) as get,
            redirect_stdout(output),
            redirect_stderr(error),
        ):
            code = self._run()

        self.assertEqual((code, error.getvalue()), (0, ""))
        get.assert_called_once_with(
            f"https://api.telegram.org/bot{TOKEN}/getMe",
            timeout=10,
        )
        self.assertNotIn(TOKEN, output.getvalue())
        self.assertIn("@pilotage_bot", output.getvalue())
        self.assertEqual(
            self.env_path.read_text(encoding="utf-8"),
            f"TELEGRAM_BOT_TOKEN={TOKEN}\n"
            "TELEGRAM_ALLOWED_USERS=42,43\n"
            "TELEGRAM_HOME_CHANNEL=42\n",
        )
        settings = Settings.load(self.settings_path)
        self.assertFalse(settings.flag("whatsapp.enabled", True))
        self.assertTrue(settings.flag("telegram.enabled", False))
        self.assertIn("# operator comment\n", self.settings_path.read_text())

    def test_existing_values_and_file_are_kept_by_default(self):
        original_env = (
            "# keep\n"
            f"TELEGRAM_BOT_TOKEN={TOKEN}\n"
            "TELEGRAM_ALLOWED_USERS=42,43\n"
            "TELEGRAM_HOME_CHANNEL=42\n"
        )
        self.env_path.write_text(original_env, encoding="utf-8")
        existing = {
            "TELEGRAM_BOT_TOKEN": TOKEN,
            "TELEGRAM_ALLOWED_USERS": "42,43",
            "TELEGRAM_HOME_CHANNEL": "42",
        }
        with (
            mock.patch.dict(os.environ, existing),
            mock.patch("pilotage.main.getpass.getpass") as secret,
            mock.patch("builtins.input", side_effect=["", "", ""]),
            mock.patch("pilotage.main.httpx.get", return_value=_accepted_bot()),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = self._run()

        self.assertEqual((code, error.getvalue()), (0, ""))
        secret.assert_not_called()
        self.assertEqual(self.env_path.read_text(encoding="utf-8"), original_env)
        self.assertTrue(
            Settings.load(self.settings_path).flag("telegram.enabled", False)
        )

    def test_interruption_cancels_without_enabling_telegram(self):
        existing = {
            "TELEGRAM_BOT_TOKEN": TOKEN,
            "TELEGRAM_ALLOWED_USERS": "42",
            "TELEGRAM_HOME_CHANNEL": "42",
        }
        with (
            mock.patch.dict(os.environ, existing),
            mock.patch("builtins.input", side_effect=KeyboardInterrupt),
            mock.patch("pilotage.main.httpx.get") as get,
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = self._run()

        self.assertEqual(code, 1)
        get.assert_not_called()
        self.assertFalse(self.env_path.exists())
        self.assertFalse(
            Settings.load(self.settings_path).flag("telegram.enabled", True)
        )
        self.assertIn("setup cancelled", error.getvalue())

    def test_existing_values_can_be_replaced_independently(self):
        existing = {
            "TELEGRAM_BOT_TOKEN": TOKEN,
            "TELEGRAM_ALLOWED_USERS": "42",
            "TELEGRAM_HOME_CHANNEL": "-1001",
            "TELEGRAM_HOME_CHANNEL_THREAD_ID": "7",
        }
        with (
            mock.patch.dict(os.environ, existing),
            mock.patch("pilotage.main.getpass.getpass", return_value=NEW_TOKEN),
            mock.patch(
                "builtins.input",
                side_effect=["yes", "", "yes", "-1002", "yes", "88"],
            ),
            mock.patch("pilotage.main.httpx.get", return_value=_accepted_bot()),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = self._run()

        self.assertEqual((code, error.getvalue()), (0, ""))
        text = self.env_path.read_text(encoding="utf-8")
        self.assertIn(f"TELEGRAM_BOT_TOKEN={NEW_TOKEN}\n", text)
        self.assertNotIn("TELEGRAM_ALLOWED_USERS=", text)
        self.assertIn("TELEGRAM_HOME_CHANNEL=-1002\n", text)
        self.assertIn("TELEGRAM_HOME_CHANNEL_THREAD_ID=88\n", text)

    def test_switching_home_to_a_dm_clears_the_group_topic(self):
        original_env = (
            f"TELEGRAM_BOT_TOKEN={TOKEN}\n"
            "TELEGRAM_ALLOWED_USERS=42\n"
            "TELEGRAM_HOME_CHANNEL=-1001\n"
            "TELEGRAM_HOME_CHANNEL_THREAD_ID=7\n"
        )
        self.env_path.write_text(original_env, encoding="utf-8")
        existing = {
            "TELEGRAM_BOT_TOKEN": TOKEN,
            "TELEGRAM_ALLOWED_USERS": "42",
            "TELEGRAM_HOME_CHANNEL": "-1001",
            "TELEGRAM_HOME_CHANNEL_THREAD_ID": "7",
        }
        with (
            mock.patch.dict(os.environ, existing),
            mock.patch("builtins.input", side_effect=["", "", "yes", "42"]),
            mock.patch("pilotage.main.httpx.get", return_value=_accepted_bot()),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = self._run()

        self.assertEqual((code, error.getvalue()), (0, ""))
        text = self.env_path.read_text(encoding="utf-8")
        self.assertIn("TELEGRAM_HOME_CHANNEL=42\n", text)
        self.assertIn("TELEGRAM_HOME_CHANNEL_THREAD_ID=\n", text)

    def test_invalid_values_are_reprompted_before_any_write(self):
        with (
            mock.patch(
                "pilotage.main.getpass.getpass",
                side_effect=["not-a-token", TOKEN],
            ),
            mock.patch(
                "builtins.input",
                side_effect=["*", "username", "42", "not-a-chat", ""],
            ),
            mock.patch("pilotage.main.httpx.get", return_value=_accepted_bot()),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = self._run()

        self.assertEqual(code, 0)
        self.assertIn("token format is invalid", error.getvalue())
        self.assertIn("explicit users", error.getvalue())
        self.assertIn("numeric Telegram user IDs", error.getvalue())
        self.assertIn("non-zero numeric chat ID", error.getvalue())

    def test_failed_bot_verification_writes_nothing_and_does_not_enable(self):
        with (
            mock.patch("pilotage.main.getpass.getpass", return_value=TOKEN),
            mock.patch("builtins.input", side_effect=["42", ""]),
            mock.patch(
                "pilotage.main.httpx.get",
                side_effect=httpx.ConnectError("network unavailable"),
            ),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = self._run()

        self.assertEqual(code, 1)
        self.assertFalse(self.env_path.exists())
        self.assertFalse(
            Settings.load(self.settings_path).flag("telegram.enabled", True)
        )
        self.assertNotIn(TOKEN, error.getvalue())
        self.assertIn("Could not verify", error.getvalue())

    def test_external_values_cannot_be_falsely_overwritten(self):
        existing = {
            "TELEGRAM_BOT_TOKEN": TOKEN,
            "TELEGRAM_ALLOWED_USERS": "42",
            "TELEGRAM_HOME_CHANNEL": "42",
        }
        with (
            mock.patch.dict(os.environ, existing),
            mock.patch("builtins.input", side_effect=["", "yes", "43", ""]),
            mock.patch("pilotage.main.httpx.get") as get,
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = self._run(
                external_env=frozenset({"TELEGRAM_ALLOWED_USERS"})
            )

        self.assertEqual(code, 1)
        get.assert_not_called()
        self.assertFalse(self.env_path.exists())
        self.assertIn("externally supplied", error.getvalue())

    def test_cli_dispatches_setup_before_strict_config_loading(self):
        selected_env = self.root / "deployment.env"
        selected_env.write_text("", encoding="utf-8")
        with (
            mock.patch.dict(
                os.environ,
                {"PILOTAGE_ENV_FILE": str(selected_env)},
                clear=True,
            ),
            mock.patch.object(
                main.profiles,
                "activate_for_process",
                return_value=("default", self.root),
            ),
            mock.patch.object(
                main.Config,
                "load",
                side_effect=AssertionError("Telegram setup must bypass Config.load"),
            ),
            mock.patch.object(main, "command_telegram_setup", return_value=0) as setup,
        ):
            code = main.main(["telegram"])

        self.assertEqual(code, 0)
        setup.assert_called_once_with(
            self.root,
            env_path=selected_env,
            settings_path=self.settings_path,
            external_env=frozenset(),
        )


if __name__ == "__main__":
    unittest.main()
