"""Profile-scoped WhatsApp pairing without starting the full runtime."""

from __future__ import annotations

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
        )

    def test_pair_only_bridge_saves_credentials_and_releases_profile(self):
        commands = []

        def run(command, **kwargs):
            commands.append((command, kwargs))
            self.session.mkdir(parents=True, exist_ok=True)
            (self.session / "creds.json").write_text("{}", encoding="utf-8")
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
        lock = ProfileRuntimeLock(self.config.state_dir)
        lock.acquire()
        lock.release()

    def test_success_exit_without_credentials_fails_closed(self):
        with (
            mock.patch("pilotage.main.shutil.which", return_value="node"),
            mock.patch(
                "pilotage.main.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
            redirect_stderr(StringIO()) as error,
        ):
            code = command_whatsapp_pair(self.config)
        self.assertEqual(code, 1)
        self.assertIn("without saved credentials", error.getvalue())


if __name__ == "__main__":
    unittest.main()
