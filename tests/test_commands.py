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
    resolve_command,
    status_text,
)
from pilotage.settings import Settings
from pilotage.i18n import t


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
        self.assertIsNone(parse_command("/profile"))

    def test_arguments_are_preserved_for_usage_validation(self):
        invocation = parse_command("/new unexpected")
        self.assertEqual(invocation.arguments, "unexpected")

    def test_help_is_derived_from_the_registry(self):
        rendered = help_text()
        for command in COMMAND_REGISTRY:
            self.assertEqual(rendered.count(f"/{command.name} "), 1)
        for retired in ("/approve", "/deny"):
            self.assertNotIn(retired, rendered)


class FormattingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = self.root / "agent"
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

    def test_status_reports_local_configuration_without_shared_auth(self):
        self.config.main_credentials_path.write_text("{}", encoding="utf-8")
        rendered = status_text(self.config)
        self.assertNotIn("Profile:", rendered)
        self.assertIn("Model: gpt-test", rendered)
        self.assertIn("Channel: whatsapp", rendered)
        self.assertIn("Tools: todo", rendered)
        self.assertIn("not signed in", rendered)

    def test_status_uses_only_this_agents_auth(self):
        self.config.main_credentials_path.write_text("{}", encoding="utf-8")
        self.config.credentials_path.write_text("{}", encoding="utf-8")
        self.assertIn("ChatGPT auth: this agent", status_text(self.config))


    def test_static_status_labels_follow_the_profile_language(self):
        self.config.language = "fr"
        rendered = status_text(self.config)
        self.assertNotIn("Profil :", rendered)
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
            self.assertEqual(main.command_status(self.config), 1)

        self.assertNotIn("Profile:", output.getvalue())
        self.assertIn("broken credentials", errors.getvalue())

    def test_status_succeeds_after_authentication_verification(self):
        with (
            mock.patch.object(main.auth, "read_credentials") as read,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main.command_status(self.config), 0)

        read.assert_called_once_with(
            self.config.credentials_path,
        )


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    class FakeAgent:
        def __init__(self):
            self.forgotten = []
            self.forget_result = True
            self.stopped = []
            self.stop_outcome = SimpleNamespace(
                status="not_running",
                previous_phase="",
            )
            self.approval_waiting = False
            self.approval_resolutions = []

        async def forget(self, session_id):
            self.forgotten.append(session_id)
            return self.forget_result

        async def stop(self, session_id):
            self.stopped.append(session_id)
            return self.stop_outcome

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
            session_id="wa-chat",
            reset_reply="reset done",
        )

    async def test_new_resets_the_exact_isolated_session(self):
        self.assertEqual(await self.execute("/reset"), "reset done")
        self.assertEqual(self.agent.forgotten, ["wa-chat"])

    async def test_new_keeps_stop_reachable_while_a_request_runs(self):
        self.agent.forget_result = False

        self.assertEqual(
            await self.execute("/new"),
            t("commands.reset_running", "en"),
        )

    async def test_arguments_are_rejected_without_resetting(self):
        self.assertEqual(await self.execute("/new named"), "Usage: /new")
        self.assertEqual(self.agent.forgotten, [])

    async def test_stop_targets_the_exact_session_without_model_input(self):
        self.agent.stop_outcome = SimpleNamespace(
            status="stopped",
            previous_phase="started",
        )

        self.assertEqual(await self.execute("/stop"), "Stopped.")
        self.assertEqual(self.agent.stopped, ["wa-chat"])

    async def test_stop_reports_unsafe_and_completed_races(self):
        self.agent.stop_outcome = SimpleNamespace(
            status="unknown",
            previous_phase="tool_requested",
        )
        self.assertEqual(await self.execute("/stop"), t("commands.stop_unknown", "en"))
        self.agent.stop_outcome = SimpleNamespace(
            status="too_late",
            previous_phase="answer_ready",
        )
        self.assertIn("already complete", await self.execute("/stop"))

    async def test_stop_arguments_return_usage_without_stopping(self):
        self.assertEqual(await self.execute("/stop later"), "Usage: /stop")
        self.assertEqual(self.agent.stopped, [])

    async def test_info_commands_do_not_touch_the_session(self):
        self.assertIn("/new", await self.execute("/help"))
        self.assertEqual(await self.execute("/status"), t("commands.ready", "en"))
        self.assertEqual(self.agent.forgotten, [])

    async def test_legacy_commands_never_authorize_or_expose_operator_details(self):
        for written in ("/approve", "/approve anything", "/deny", "/deny not this change"):
            with self.subTest(command=written):
                self.agent.approval_waiting = True
                self.assertEqual(await self.execute(written), t("capability.unavailable", "en"))
                self.assertTrue(self.agent.approval_waiting)
        self.assertEqual(self.agent.approval_resolutions, [])

    async def test_status_does_not_read_or_send_operator_configuration(self):
        with (
            mock.patch("pilotage.commands.status_text", side_effect=AssertionError("operator only")),
        ):
            self.assertEqual(await self.execute("/status"), t("commands.ready", "en"))

    async def test_command_replies_follow_the_agent_language(self):
        self.config.language = "ar"
        self.assertEqual(await self.execute("/approve"), t("capability.unavailable", "ar"))
        self.assertIn(t("commands.header", "ar"), await self.execute("/help"))


class DurableCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = main.DeliveryStore(Path(temporary.name) / "delivery.db")
        self.invocation = parse_command("/new")

    async def test_completed_command_reuses_response_without_reexecution(self):
        execute = mock.AsyncMock(return_value="reset done")
        arguments = {
            "platform": "whatsapp",
            "claim_id": "claim-1",
            "session_key": "session",
            "invocation": self.invocation,
            "uncertain_reply": "unknown",
            "execute": execute,
        }

        first = await main._durable_command_result(self.store, **arguments)
        second = await main._durable_command_result(self.store, **arguments)

        self.assertEqual((first, second), ("reset done", "reset done"))
        execute.assert_awaited_once()

    async def test_cached_failure_reply_keeps_its_fence_without_reaching_the_client(self):
        content = "Codex response remained incomplete after 3 continuation attempts"
        for platform in ("whatsapp", "telegram"):
            with self.subTest(platform=platform):
                claim_id = f"legacy-{platform}"
                command_id = main.compute_command_id(platform, claim_id)
                self.store.begin_command(
                    command_id=command_id, platform=platform, claim_id=claim_id,
                    session_key="session", command_name="status", arguments="",
                )
                self.store.complete_command(command_id, content)
                execute, send, ledger_send = mock.AsyncMock(), mock.AsyncMock(), mock.AsyncMock()
                for _restart in range(2):
                    restarted = main.DeliveryStore(self.store.path)
                    answer = await main._durable_command_result(
                        restarted, platform=platform, claim_id=claim_id, session_key="session",
                        invocation=parse_command("/status"), uncertain_reply="unknown", execute=execute,
                    )
                    await main.deliver_final(
                        restarted, session_key="session", message_ref=claim_id,
                        platform=platform, chat_id="42", thread_id="", content=answer,
                        send=send, ledger_send=ledger_send,
                    )
                    self.assertTrue(await main._exact_delivery_obligation_exists(
                        restarted, session_key="session", message_ref=claim_id,
                        platform=platform, chat_id="42", thread_id="", reply_to="", content=content,
                    ))
                execute.assert_not_awaited()
                send.assert_not_awaited()
                ledger_send.assert_not_awaited()

    async def test_interrupted_command_returns_warning_without_reexecution(self):
        command_id = main.compute_command_id("telegram", "claim-2")
        self.store.begin_command(
            command_id=command_id,
            platform="telegram",
            claim_id="claim-2",
            session_key="session",
            command_name="new",
            arguments="",
        )
        execute = mock.AsyncMock(return_value="must not run")

        response = await main._durable_command_result(
            self.store,
            platform="telegram",
            claim_id="claim-2",
            session_key="session",
            invocation=self.invocation,
            uncertain_reply="verify the prior command",
            execute=execute,
        )

        self.assertEqual(response, "verify the prior command")
        execute.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
