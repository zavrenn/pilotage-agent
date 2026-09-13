"""Workspace defaults and native file delivery, without live integrations."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pilotage import media
from pilotage.agent import Agent
from pilotage.codex.stream import StreamResult
from pilotage.config import Config
from pilotage.history import ConversationStore, session_workspace_path
from pilotage.tools import ToolContext
from pilotage.tools.image import _output_root


class WorkspaceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.state = self.root / ".pilotage-agent"
        self.state.mkdir()
        # Windows TLS initialization needs these even in an isolated test env.
        values = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR") if name in os.environ}
        values["PILOTAGE_HOME"] = str(self.state)
        environment = mock.patch.dict(os.environ, values, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def config(self, text=""):
        (self.state / "config.yaml").write_text(text, encoding="utf-8")
        return Config.load()

    def file(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF test")
        return path

    def delivered(self, config, *files):
        content = "\n".join(f"MEDIA:{path}" for path in files)
        return media.extract_outbound(content, media.delivery_roots(config, content))[0]

    def test_default_delivery_accepts_only_export_subtree(self):
        config = self.config()
        self.assertEqual(config.workspace_dir, self.root / "workspace")
        report = self.file(config.workspace_dir / "exports/reports/report.pdf")
        private = [
            self.file(config.workspace_dir / name)
            for name in ("inputs/source.pdf", "tmp/draft.pdf", "root.pdf", "export/other.pdf")
        ]
        self.assertEqual(
            [item.path for item in self.delivered(config, report, *private)],
            [report],
        )
        self.assertFalse((self.state / "workspace").exists())

    def test_terminal_override_controls_default_export_and_image_output(self):
        working = self.root / "company work"
        working.mkdir()
        config = self.config(f"terminal:\n  cwd: '{working.as_posix()}'\n")
        report = self.file(working / "exports/report.pdf")
        wrong = self.file(config.workspace_dir / "exports/wrong.pdf")
        self.assertEqual(config.outbound_media_roots, (working / "exports",))
        self.assertEqual(
            [item.path for item in self.delivered(config, report, wrong)],
            [report],
        )
        context = ToolContext("chat", config, working_directory=working / "tmp")
        self.assertEqual(_output_root(context), working / "exports")
        self.assertEqual(config.outbound_media_roots, (working / "exports",))

    def test_channel_cwd_override_gets_its_own_default_export(self):
        common = self.root / "common"
        telegram = self.root / "telegram"
        common.mkdir()
        telegram.mkdir()
        self.config(
            f"terminal:\n  cwd: '{common.as_posix()}'\n"
            f"channels:\n  telegram:\n    terminal:\n      cwd: '{telegram.as_posix()}'\n"
        )
        self.assertEqual(Config.load().outbound_media_roots, (common / "exports",))
        self.assertEqual(
            Config.load(channel="telegram").outbound_media_roots,
            (telegram / "exports",),
        )

    def test_explicit_allowlist_replaces_all_default_export_roots(self):
        allowed = self.root / "approved"
        allowed.mkdir()
        config = self.config(
            "sessions:\n  isolated_workspaces: true\n"
            f"gateway:\n  media_delivery_allow_dirs: ['{allowed.as_posix()}']\n"
        )
        approved = self.file(allowed / "report.pdf")
        default = self.file(config.workspace_dir / "exports/default.pdf")
        session = session_workspace_path(config.workspace_dir, "chat", 1)
        isolated = self.file(session / "exports/isolated.pdf")
        self.assertEqual(
            [item.path for item in self.delivered(config, approved, default, isolated)],
            [approved],
        )

    def test_explicit_empty_disables_regular_and_isolated_file_delivery(self):
        for isolated in (False, True):
            with self.subTest(isolated=isolated):
                config = self.config(
                    f"sessions:\n  isolated_workspaces: {str(isolated).lower()}\n"
                    "gateway:\n  media_delivery_allow_dirs: []\n"
                )
                report = self.file(config.workspace_dir / "exports/report.pdf")
                session = session_workspace_path(config.workspace_dir, "chat", 1)
                private = self.file(session / "exports/report.pdf")
                self.assertEqual(self.delivered(config, report, private), [])
                with self.assertRaisesRegex(ValueError, "disabled"):
                    _output_root(ToolContext("chat", config, working_directory=session))

    def test_isolated_transport_admits_export_only_under_valid_session_paths(self):
        config = self.config("sessions:\n  isolated_workspaces: true\n")
        session = session_workspace_path(config.workspace_dir, "chat", 1)
        report = self.file(session / "exports/report.pdf")
        denied = [
            self.file(session / "inputs/source.pdf"),
            self.file(session / "tmp/draft.pdf"),
            self.file(config.workspace_dir / "sessions/arbitrary/exports/report.pdf"),
        ]
        self.assertEqual(
            [item.path for item in self.delivered(config, report, *denied)],
            [report],
        )
        self.assertEqual(
            _output_root(ToolContext("chat", config, working_directory=session)),
            session / "exports",
        )

    async def test_isolated_turn_confines_reply_to_its_own_export(self):
        config = self.config("sessions:\n  isolated_workspaces: true\ntools:\n  enabled: [todo]\n")
        agent = Agent(config, ConversationStore(self.state / "conversations.db"))
        self.addAsyncCleanup(agent.close)
        own = agent._session_working_directory("chat")
        other = agent._session_working_directory("other")
        report = self.file(own / "exports/report.pdf")
        wrong = self.file(other / "exports/other.pdf")
        draft = self.file(own / "tmp/draft.pdf")

        async def respond(*args, **kwargs):
            return StreamResult(text=f"MEDIA:{report}\nMEDIA:{wrong}\nMEDIA:{draft}")

        agent._stream_once = respond
        answer = await agent.respond("chat", "Send my report.")
        attachments, _ = media.extract_outbound(answer, media.delivery_roots(config, answer))
        self.assertEqual([item.path for item in attachments], [report])
        self.assertIn(str(own / "exports"), agent._instructions_for_session("chat", own))

    def test_isolated_cwd_override_keeps_delivery_policy_independent(self):
        working = self.root / "custom"
        working.mkdir()
        prefix = (
            "sessions:\n  isolated_workspaces: true\n"
            f"terminal:\n  cwd: '{working.as_posix()}'\n"
        )
        config = self.config(prefix)
        session = session_workspace_path(working, "chat", 1)
        report = self.file(session / "exports/report.pdf")
        draft = self.file(session / "tmp/draft.pdf")
        self.assertEqual(
            [item.path for item in self.delivered(config, report, draft)],
            [report],
        )
        config = self.config(prefix + "gateway:\n  media_delivery_allow_dirs: []\n")
        self.assertEqual(self.delivered(config, report), [])

    def test_default_export_symlink_preserves_the_real_session_root(self):
        for override in (False, True):
            with self.subTest(terminal_override=override):
                working = self.root / ("custom" if override else "workspace")
                working.mkdir()
                prefix = "sessions:\n  isolated_workspaces: true\n"
                if override:
                    prefix += f"terminal:\n  cwd: '{working.as_posix()}'\n"
                config = self.config(prefix)
                shared = self.root / f"shared-{override}" / "public"
                shared.mkdir(parents=True)
                try:
                    (working / "exports").symlink_to(shared, target_is_directory=True)
                except OSError:
                    self.skipTest("Creating symlinks is unavailable")
                session = session_workspace_path(working, "chat", 1)
                report = self.file(session / "exports/report.pdf")
                wrong_session = session_workspace_path(shared.parent, "chat", 1)
                wrong = self.file(wrong_session / "exports/wrong.pdf")
                self.assertEqual(config.outbound_media_roots, (shared,))
                self.assertEqual(
                    [item.path for item in self.delivered(config, report, wrong)],
                    [report],
                )

    def test_session_export_symlink_cannot_expose_files_outside_workspace(self):
        config = self.config("sessions:\n  isolated_workspaces: true\n")
        session = session_workspace_path(config.workspace_dir, "chat", 1)
        session.mkdir(parents=True)
        private = self.file(self.root / "private/secret.pdf")
        try:
            (session / "exports").symlink_to(private.parent, target_is_directory=True)
        except OSError:
            self.skipTest("Creating symlinks is unavailable")
        self.assertEqual(self.delivered(config, session / "exports/secret.pdf"), [])

    def test_example_configuration_uses_export_default(self):
        example = Path(__file__).resolve().parent.parent / "config.yaml.example"
        config = self.config(example.read_text(encoding="utf-8"))
        self.assertIsNone(config.settings.get("gateway.media_delivery_allow_dirs"))
        self.assertEqual(config.outbound_media_roots, (self.root / "workspace/exports",))
        instructions = Agent(config, ConversationStore(None))._instructions_for_session("chat")
        self.assertIn(str(config.workspace_dir / "exports"), instructions)


if __name__ == "__main__":
    unittest.main()
