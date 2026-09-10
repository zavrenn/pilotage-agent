"""Single-agent state and authentication; no real credentials or services."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pilotage import main
from pilotage.codex import auth
from pilotage.config import Config, state_dir


def credentials(token="local"):
    return auth.Credentials(token, "refresh-" + token, auth.DEFAULT_CODEX_BASE_URL, "now")


class SingleAgentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = self.root / "agent"
        self.state.mkdir()
        patch = mock.patch.dict(os.environ, {
            "PILOTAGE_HOME": str(self.state),
            "PILOTAGE_ENV_FILE": str(self.state / ".env"),
        }, clear=True)
        patch.start()
        self.addCleanup(patch.stop)

    def test_default_state_uses_agent_home(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(Path, "home", return_value=self.root):
            self.assertEqual(state_dir(), self.root / ".pilotage-agent")

    def test_all_existing_state_paths_remain_in_place(self):
        config = Config.load()
        self.assertEqual(config.state_dir, self.state)
        self.assertEqual(config.credentials_path, self.state / "codex-auth.json")
        self.assertEqual(config.conversations_path, self.state / "conversations.db")
        self.assertEqual(config.memory_dir, self.state / "memories")
        self.assertEqual(config.workspace_dir, self.state / "workspace")
        self.assertFalse(hasattr(config, "main_credentials_path"))
        self.assertFalse((self.state / "profiles").exists())

    def test_missing_auth_never_borrows_parent_credentials(self):
        parent_auth = self.root / "codex-auth.json"
        auth.write_credentials(parent_auth, credentials("parent"))
        before = parent_auth.read_bytes()
        for read in (auth.read_credentials, auth.resolve_credentials):
            with self.subTest(read=read.__name__), self.assertRaises(auth.AuthError) as caught:
                read(self.state / "codex-auth.json")
            self.assertEqual(caught.exception.code, "codex_auth_missing")
        self.assertEqual(parent_auth.read_bytes(), before)

    def test_auth_api_no_longer_accepts_fallback(self):
        for read in (auth.read_credentials, auth.resolve_credentials):
            with self.assertRaises(TypeError):
                read(self.state / "codex-auth.json", fallback_path=self.root / "codex-auth.json")

    def test_refresh_changes_only_this_agents_credentials(self):
        local = self.state / "codex-auth.json"
        parent = self.root / "codex-auth.json"
        auth.write_credentials(local, credentials("old"))
        auth.write_credentials(parent, credentials("parent"))
        before = parent.read_bytes()
        with mock.patch.object(auth, "refresh_credentials", return_value=credentials("new")):
            found = auth.resolve_credentials(local, force_refresh=True)
        self.assertEqual(found.access_token, "new")
        self.assertEqual(auth.read_credentials(local).access_token, "new")
        self.assertEqual(parent.read_bytes(), before)

    def test_malformed_local_auth_fails_without_using_parent(self):
        auth.write_credentials(self.root / "codex-auth.json", credentials("parent"))
        local = self.state / "codex-auth.json"
        local.write_text("broken", encoding="utf-8")
        with self.assertRaises(auth.AuthError):
            auth.read_credentials(local)
        self.assertEqual(local.read_text(), "broken")

    def test_login_still_works_before_channel_credentials_exist(self):
        settings = self.state / "config.yaml"
        settings.write_text("telegram:\n  enabled: true\n", encoding="utf-8")
        with mock.patch.object(auth, "device_code_login", return_value=credentials()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.main(["login"]), 0)
        self.assertEqual(auth.read_credentials(self.state / "codex-auth.json").access_token, "local")
        self.assertEqual(settings.read_text(), "telegram:\n  enabled: true\n")

    def test_run_still_requires_enabled_channel_credentials(self):
        (self.state / "config.yaml").write_text("telegram:\n  enabled: true\n", encoding="utf-8")
        with mock.patch.object(main, "command_run") as run, self.assertLogs("pilotage", level="ERROR") as messages:
            self.assertEqual(main.main(["run"]), 1)
        run.assert_not_called()
        self.assertTrue(any("TELEGRAM_BOT_TOKEN" in message for message in messages.output))

    def test_env_cannot_redirect_state_after_login_target_is_chosen(self):
        (self.state / ".env").write_text(f"PILOTAGE_HOME={self.root / 'other'}\n", encoding="utf-8")
        with mock.patch.object(auth, "device_code_login", return_value=credentials()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.main(["login"]), 0)
        self.assertEqual(state_dir(), self.state)
        self.assertFalse((self.root / "other").exists())

    def test_named_profile_options_and_management_are_rejected_before_loading(self):
        for argv in (["--profile", "default", "run"], ["--profile", "work", "run"], ["-p", "work", "run"], ["profile", "list"], ["profile", "create", "work"], ["profile", "delete", "work", "--yes"]):
            with self.subTest(argv=argv), mock.patch.object(main, "load_env_files") as load, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main.main(argv)
            self.assertEqual(caught.exception.code, 2)
            load.assert_not_called()

    def test_run_uses_the_agent_state(self):
        with mock.patch.object(main, "command_run", new_callable=mock.AsyncMock, return_value=0) as run:
            self.assertEqual(main.main(["run"]), 0)
        run.assert_awaited_once()
        self.assertEqual(len(run.call_args.args), 1)
        self.assertEqual(run.call_args.args[0].state_dir, self.state)

    def test_unrelated_state_files_do_not_select_an_agent_or_block_startup(self):
        for name in ("profiles", "active_profile"):
            (self.state / name).write_text("ordinary content", encoding="utf-8")
        with mock.patch.object(main, "command_run", new_callable=mock.AsyncMock, return_value=0) as run:
            self.assertEqual(main.main(["run"]), 0)
        self.assertEqual(run.call_args.args[0].state_dir, self.state)
        self.assertEqual(run.call_args.args[0].credentials_path, self.state / "codex-auth.json")

    def test_help_has_no_profile_interface(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            main.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertNotIn("profile", output.getvalue())

    def test_interactive_terminal_conversation_is_still_not_a_command(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main.main(["chat"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
