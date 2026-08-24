"""Contract for the small model-independent management command surface."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilotage import main
from pilotage.codex.auth import AuthError
from pilotage.commands import (
    COMMAND_REGISTRY,
    execute_command,
    help_text,
    parse_command,
    profile_text,
    resolve_command,
    status_text,
)
from pilotage.settings import Settings


class RegistryTests(unittest.TestCase):
    def test_aliases_resolve_to_the_canonical_definition(self):
        self.assertIs(resolve_command("/reset"), resolve_command("new"))
        self.assertIs(resolve_command("/commands"), resolve_command("help"))

    def test_only_a_whole_message_known_slash_command_is_intercepted(self):
        invocation = parse_command("  /STATUS  ")
        self.assertIsNotNone(invocation)
        self.assertEqual(invocation.command.name, "status")
        self.assertIsNone(parse_command("what does /status mean?"))
        self.assertIsNone(parse_command("/not-a-command"))

    def test_arguments_are_preserved_for_usage_validation(self):
        invocation = parse_command("/new unexpected")
        self.assertEqual(invocation.arguments, "unexpected")

    def test_help_is_derived_from_the_registry(self):
        rendered = help_text()
        for command in COMMAND_REGISTRY:
            self.assertEqual(rendered.count(f"/{command.name} "), 1)


class FormattingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = self.root / "profiles" / "work"
        self.profile.mkdir(parents=True)
        self.config = SimpleNamespace(
            settings=Settings(
                {"tools": {"enabled": ["todo"]}},
                channel="whatsapp",
            ),
            model="gpt-test",
            state_dir=self.profile,
            credentials_path=self.profile / "codex-auth.json",
            main_credentials_path=self.root / "codex-auth.json",
        )

    def test_status_reports_the_actual_profile_channel_model_tools_and_shared_auth(self):
        self.config.main_credentials_path.write_text("{}", encoding="utf-8")
        rendered = status_text(self.config, "work")
        self.assertIn("Profile: work", rendered)
        self.assertIn("Model: gpt-test", rendered)
        self.assertIn("Channel: whatsapp", rendered)
        self.assertIn("Tools: todo", rendered)
        self.assertIn("shared from default profile", rendered)

    def test_profile_auth_shadows_shared_auth_in_status(self):
        self.config.main_credentials_path.write_text("{}", encoding="utf-8")
        self.config.credentials_path.write_text("{}", encoding="utf-8")
        self.assertIn("ChatGPT auth: this profile", status_text(self.config, "work"))

    def test_profile_command_reports_the_isolated_state_root(self):
        rendered = profile_text(self.config, "work")
        self.assertIn(f"State: {self.profile}", rendered)

    def test_static_status_labels_follow_the_profile_language(self):
        self.config.language = "fr"
        rendered = status_text(self.config, "work")
        self.assertIn("Profil : work", rendered)
        self.assertIn("Modèle : gpt-test", rendered)
        self.assertIn("Outils : todo", rendered)


class StatusHealthTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.config = SimpleNamespace(
            settings=Settings({"tools": {"enabled": []}}),
            model="gpt-test",
            state_dir=root,
            credentials_path=root / "codex-auth.json",
            main_credentials_path=root / "main-auth.json",
        )

    def test_status_fails_when_authentication_is_unusable(self):
        output = io.StringIO()
        errors = io.StringIO()
        with (
            mock.patch.object(
                main.auth,
                "read_credentials",
                side_effect=AuthError("broken credentials"),
            ),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(errors),
        ):
            self.assertEqual(main.command_status(self.config, "work"), 1)

        self.assertIn("Profile: work", output.getvalue())
        self.assertIn("broken credentials", errors.getvalue())

    def test_status_succeeds_after_authentication_verification(self):
        with (
            mock.patch.object(main.auth, "read_credentials") as read,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main.command_status(self.config, "work"), 0)

        read.assert_called_once_with(
            self.config.credentials_path,
            fallback_path=self.config.main_credentials_path,
        )


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    class FakeAgent:
        def __init__(self):
            self.forgotten = []
            self.approval_waiting = False
            self.approval_resolutions = []

        async def forget(self, session_id):
            self.forgotten.append(session_id)

        def resolve_approval(self, session_id, *, approved, reason=""):
            self.approval_resolutions.append((session_id, approved, reason))
            was_waiting = self.approval_waiting
            self.approval_waiting = False
            return was_waiting

    def setUp(self):
        self.agent = self.FakeAgent()
        self.config = SimpleNamespace(
            settings=Settings({"tools": {"enabled": []}}, channel="whatsapp"),
            model="gpt-test",
            state_dir=Path("/profile"),
            credentials_path=Path("/profile/codex-auth.json"),
            main_credentials_path=Path("/main/codex-auth.json"),
        )

    async def execute(self, text):
        return await execute_command(
            parse_command(text),
            agent=self.agent,
            config=self.config,
            profile_name="work",
            session_id="wa-chat",
            reset_reply="reset done",
        )

    async def test_new_resets_the_exact_isolated_session(self):
        self.assertEqual(await self.execute("/reset"), "reset done")
        self.assertEqual(self.agent.forgotten, ["wa-chat"])

    async def test_arguments_are_rejected_without_resetting(self):
        self.assertEqual(await self.execute("/new named"), "Usage: /new")
        self.assertEqual(self.agent.forgotten, [])

    async def test_info_commands_do_not_touch_the_session(self):
        self.assertIn("/new", await self.execute("/help"))
        self.assertIn("Profile: work", await self.execute("/profile"))
        self.assertIn("Pilotage", await self.execute("/status"))
        self.assertEqual(self.agent.forgotten, [])

    async def test_approve_and_deny_resolve_only_this_session(self):
        self.assertEqual(await self.execute("/approve"), "No approval is waiting.")
        self.agent.approval_waiting = True
        self.assertIn("Approved", await self.execute("/approve"))
        self.agent.approval_waiting = True
        self.assertIn("Denied", await self.execute("/deny not this change"))
        self.assertEqual(
            self.agent.approval_resolutions,
            [
                ("wa-chat", True, ""),
                ("wa-chat", True, ""),
                ("wa-chat", False, "not this change"),
            ],
        )

    async def test_command_replies_follow_the_profile_language(self):
        self.config.language = "ar"
        self.assertIn("لا توجد موافقة", await self.execute("/approve"))
        self.assertIn("أوامر الإدارة", await self.execute("/help"))


if __name__ == "__main__":
    unittest.main()
