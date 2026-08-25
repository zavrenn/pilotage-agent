"""Profile-scoped WhatsApp pairing without starting the full runtime."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilotage.main import command_whatsapp_pair
from pilotage.runtime_lock import ProfileRuntimeLock


class WhatsAppPairingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bridge = self.root / "bridge"
        self.bridge.mkdir()
        (self.bridge / "node_modules").mkdir()
        (self.bridge / "bridge.js").write_text("// bridge", encoding="utf-8")
        self.session = self.root / "state" / "whatsapp"
        self.config = SimpleNamespace(
            bridge_script=self.bridge / "bridge.js",
            bridge_dir=self.bridge,
            session_dir=self.session,
            state_dir=self.root / "state",
            allowed_senders=frozenset(),
            home_chat_id="",
            group_allow_from=frozenset(),
        )

    def test_pair_only_bridge_saves_credentials_and_releases_profile(self):
        commands = []

        def run(command, **kwargs):
            commands.append((command, kwargs))
            self.session.mkdir(parents=True, exist_ok=True)
            (self.session / "creds.json").write_text(
                json.dumps({"registered": True}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)

        output, error = StringIO(), StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "PILOTAGE_CODEX_BASE_URL": "sentinel-codex",
                    "VOICE_TOOLS_OPENAI_KEY": "sentinel-openai",
                },
            ),
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch("pilotage.main.subprocess.run", side_effect=run),
            mock.patch("builtins.input", side_effect=["+212 600 000 000", ""]),
            redirect_stdout(output),
            redirect_stderr(error),
        ):
            code = command_whatsapp_pair(self.config)

        self.assertEqual((code, error.getvalue()), (0, ""))
        self.assertIn("connected", output.getvalue())
        self.assertEqual(
            commands[0][0],
            [
                "node",
                str(self.config.bridge_script),
                "--pair-only",
                "--session",
                str(self.session),
            ],
        )
        self.assertNotIn("PILOTAGE_CODEX_BASE_URL", commands[0][1]["env"])
        self.assertNotIn("VOICE_TOOLS_OPENAI_KEY", commands[0][1]["env"])
        self.assertEqual(
            commands[0][1]["env"]["PILOTAGE_ALLOWED_SENDERS"],
            "212600000000",
        )
        env_text = (self.config.state_dir / ".env").read_text(encoding="utf-8")
        self.assertIn("PILOTAGE_ALLOWED_SENDERS=212600000000\n", env_text)
        self.assertIn(
            "WHATSAPP_HOME_CHANNEL=212600000000@s.whatsapp.net\n",
            env_text,
        )
        lock = ProfileRuntimeLock(self.config.state_dir)
        lock.acquire()
        lock.release()

    def test_invalid_existing_values_must_be_replaced(self):
        self.config.allowed_senders = frozenset({"not-a-number"})
        self.config.home_chat_id = "not-a-chat"

        def run(command, **kwargs):
            self.session.mkdir(parents=True, exist_ok=True)
            (self.session / "creds.json").write_text(
                json.dumps({"registered": True}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch("pilotage.main.subprocess.run", side_effect=run),
            mock.patch("builtins.input", side_effect=["212600000000", ""]),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(self.config)

        self.assertEqual(code, 0)
        self.assertIn("Existing WhatsApp allowlist is invalid", error.getvalue())
        self.assertIn("Existing WhatsApp home chat is invalid", error.getvalue())
        env_text = (self.config.state_dir / ".env").read_text(encoding="utf-8")
        self.assertIn("PILOTAGE_ALLOWED_SENDERS=212600000000\n", env_text)
        self.assertIn(
            "WHATSAPP_HOME_CHANNEL=212600000000@s.whatsapp.net\n",
            env_text,
        )

    def test_lid_allowlist_default_preserves_lid_home_target(self):
        lid = "130631430344750@lid"

        def run(command, **kwargs):
            self.session.mkdir(parents=True, exist_ok=True)
            (self.session / "creds.json").write_text(
                json.dumps({"registered": True}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch("pilotage.main.subprocess.run", side_effect=run),
            mock.patch("builtins.input", side_effect=[lid, ""]),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(self.config)

        self.assertEqual((code, error.getvalue()), (0, ""))
        env_text = (self.config.state_dir / ".env").read_text(encoding="utf-8")
        self.assertIn(f"PILOTAGE_ALLOWED_SENDERS={lid}\n", env_text)
        self.assertIn(f"WHATSAPP_HOME_CHANNEL={lid}\n", env_text)

    def test_updates_use_the_selected_environment_file(self):
        selected = self.root / "deployment.env"
        selected.write_text("UNRELATED=kept\n", encoding="utf-8")

        def run(command, **kwargs):
            self.session.mkdir(parents=True, exist_ok=True)
            (self.session / "creds.json").write_text(
                json.dumps({"registered": True}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch("pilotage.main.subprocess.run", side_effect=run),
            mock.patch("builtins.input", side_effect=["212600000000", ""]),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(self.config, env_path=selected)

        self.assertEqual((code, error.getvalue()), (0, ""))
        self.assertFalse((self.config.state_dir / ".env").exists())
        env_text = selected.read_text(encoding="utf-8")
        self.assertIn("UNRELATED=kept\n", env_text)
        self.assertIn("PILOTAGE_ALLOWED_SENDERS=212600000000\n", env_text)

    def test_main_routes_setup_updates_to_the_loaded_override_file(self):
        from pilotage import main as main_module

        selected = self.root / "deployment.env"
        selected.write_text(
            "PILOTAGE_ALLOWED_SENDERS=212600000000\n"
            "WHATSAPP_HOME_CHANNEL=212600000000@s.whatsapp.net\n",
            encoding="utf-8",
        )

        with (
            mock.patch.dict(
                os.environ,
                {"PILOTAGE_ENV_FILE": str(selected)},
                clear=True,
            ),
            mock.patch.object(
                main_module.profiles,
                "activate_for_process",
                return_value=("default", self.root),
            ),
            mock.patch.object(main_module.Config, "load", return_value=self.config),
            mock.patch.object(
                main_module,
                "command_whatsapp_pair",
                return_value=0,
            ) as pair,
        ):
            code = main_module.main(["whatsapp"])

        self.assertEqual(code, 0)
        pair.assert_called_once_with(
            self.config,
            env_path=selected,
            external_env=frozenset(),
        )

    def test_externally_supplied_values_cannot_be_falsely_overwritten(self):
        self.config.allowed_senders = frozenset({"212600000000"})
        self.config.home_chat_id = "212600000000@s.whatsapp.net"
        env_path = self.config.state_dir / ".env"

        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch(
                "builtins.input",
                side_effect=["yes", "212611111111", ""],
            ),
            mock.patch("pilotage.main.subprocess.run") as run,
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(
                self.config,
                external_env=frozenset({"PILOTAGE_ALLOWED_SENDERS"}),
            )

        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertFalse(env_path.exists())
        self.assertIn("externally supplied", error.getvalue())

    def test_success_exit_without_credentials_fails_closed(self):
        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch(
                "pilotage.main.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
            mock.patch("builtins.input", side_effect=["212600000000", ""]),
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(self.config)
        self.assertEqual(code, 1)
        self.assertIn("did not register a linked device", error.getvalue())

    def test_unregistered_credentials_do_not_count_as_pairing_success(self):
        def run(command, **kwargs):
            self.session.mkdir(parents=True, exist_ok=True)
            (self.session / "creds.json").write_text(
                json.dumps({"registered": False}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch("pilotage.main.subprocess.run", side_effect=run),
            mock.patch("builtins.input", side_effect=["212600000000", ""]),
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(self.config)

        self.assertEqual(code, 1)
        self.assertIn("session is not registered", error.getvalue())

    def test_existing_configuration_and_pairing_are_kept_by_default(self):
        self.config.allowed_senders = frozenset({"212600000000", "212611111111"})
        self.config.home_chat_id = "212600000000@s.whatsapp.net"
        self.config.state_dir.mkdir(parents=True)
        env_path = self.config.state_dir / ".env"
        original_env = (
            "# unchanged\n"
            "PILOTAGE_ALLOWED_SENDERS=212611111111,212600000000\n"
            "WHATSAPP_HOME_CHANNEL=212600000000@s.whatsapp.net\n"
        )
        env_path.write_text(original_env, encoding="utf-8")
        self.session.mkdir(parents=True)
        (self.session / "creds.json").write_text(
            json.dumps({"registered": True}),
            encoding="utf-8",
        )

        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch("builtins.input", side_effect=["", "", ""]),
            mock.patch("pilotage.main.subprocess.run") as run,
            redirect_stdout(StringIO()) as output,
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(self.config)

        self.assertEqual((code, error.getvalue()), (0, ""))
        run.assert_not_called()
        self.assertIn("existing pairing kept", output.getvalue())
        self.assertEqual(env_path.read_text(encoding="utf-8"), original_env)

    def test_invalid_existing_session_is_not_kept_as_paired(self):
        self.config.allowed_senders = frozenset({"212600000000"})
        self.config.home_chat_id = "212600000000@s.whatsapp.net"
        self.session.mkdir(parents=True)
        credentials = self.session / "creds.json"
        credentials.write_text("{}", encoding="utf-8")

        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch("builtins.input", side_effect=["", "", ""]),
            mock.patch("pilotage.main.subprocess.run") as run,
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(self.config)

        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertTrue(credentials.is_file())
        self.assertIn("session is not registered", error.getvalue())
        self.assertIn("remains unpaired", error.getvalue())

    def test_existing_values_can_be_updated_and_session_repaired(self):
        self.config.allowed_senders = frozenset({"212600000000"})
        self.config.home_chat_id = "212600000000@s.whatsapp.net"
        self.config.state_dir.mkdir(parents=True)
        (self.config.state_dir / ".env").write_text(
            "# keep this comment\n"
            "UNRELATED=value\n"
            "PILOTAGE_ALLOWED_SENDERS=old\n"
            "PILOTAGE_ALLOWED_SENDERS=duplicate\n"
            "WHATSAPP_HOME_CHANNEL=old@s.whatsapp.net\n",
            encoding="utf-8",
        )
        self.session.mkdir()
        (self.session / "creds.json").write_text(
            json.dumps({"registered": True}),
            encoding="utf-8",
        )
        (self.session / "old-session-file").write_text("old", encoding="utf-8")

        def run(command, **kwargs):
            self.session.mkdir(parents=True, exist_ok=True)
            (self.session / "creds.json").write_text(
                json.dumps({"registered": True}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch("pilotage.main.subprocess.run", side_effect=run),
            mock.patch(
                "builtins.input",
                side_effect=[
                    "yes",
                    "+212 622 222 222, 212633333333",
                    "yes",
                    "+212 622 222 222",
                    "yes",
                ],
            ),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(self.config)

        self.assertEqual((code, error.getvalue()), (0, ""))
        self.assertFalse((self.session / "old-session-file").exists())
        env_text = (self.config.state_dir / ".env").read_text(encoding="utf-8")
        self.assertIn("# keep this comment\n", env_text)
        self.assertIn("UNRELATED=value\n", env_text)
        self.assertEqual(env_text.count("PILOTAGE_ALLOWED_SENDERS="), 1)
        self.assertIn(
            "PILOTAGE_ALLOWED_SENDERS=212622222222,212633333333\n",
            env_text,
        )
        self.assertIn(
            "WHATSAPP_HOME_CHANNEL=212622222222@s.whatsapp.net\n",
            env_text,
        )

    def test_cancelled_required_configuration_does_not_write_env(self):
        self.config.state_dir.mkdir(parents=True)
        env_path = self.config.state_dir / ".env"
        env_path.write_text("UNCHANGED=yes\n", encoding="utf-8")

        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch("builtins.input", side_effect=KeyboardInterrupt),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(self.config)

        self.assertEqual(code, 1)
        self.assertIn("setup cancelled", error.getvalue())
        self.assertEqual(env_path.read_text(encoding="utf-8"), "UNCHANGED=yes\n")

    def test_invalid_numbers_are_rejected_before_configuration_is_saved(self):
        def run(command, **kwargs):
            self.session.mkdir(parents=True, exist_ok=True)
            (self.session / "creds.json").write_text(
                json.dumps({"registered": True}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch("pilotage.main.subprocess.run", side_effect=run),
            mock.patch(
                "builtins.input",
                side_effect=["*", "212600000000", "not-a-number", ""],
            ),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(self.config)

        self.assertEqual(code, 0)
        self.assertIn("Invalid WhatsApp number", error.getvalue())
        self.assertIn(
            "Invalid WhatsApp number; use the country code and digits",
            error.getvalue(),
        )
        env_text = (self.config.state_dir / ".env").read_text(encoding="utf-8")
        self.assertIn("PILOTAGE_ALLOWED_SENDERS=212600000000\n", env_text)
        self.assertIn(
            "WHATSAPP_HOME_CHANNEL=212600000000@s.whatsapp.net\n",
            env_text,
        )


if __name__ == "__main__":
    unittest.main()
