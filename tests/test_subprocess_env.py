"""The Hermes-derived environment boundary for model-controlled children."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from pilotage.tools.shell import Shell
from pilotage.tools.subprocess_env import build_subprocess_env


class SubprocessEnvironmentTests(unittest.TestCase):
    def test_pilotage_credentials_and_private_identities_are_removed(self):
        planted = {
            "OPENAI_API_KEY": "openai-secret",
            "PILOTAGE_CODEX_BASE_URL": "https://private-model.invalid",
            "VOICE_TOOLS_OPENAI_KEY": "voice-secret",
            "FIRECRAWL_API_KEY": "firecrawl-secret",
            "TELEGRAM_BOT_TOKEN": "telegram-secret",
            "TELEGRAM_WEBHOOK_SECRET": "webhook-secret",
            "TELEGRAM_ALLOWED_USERS": "12345",
            "PILOTAGE_ALLOWED_SENDERS": "212600000000",
            "PILOTAGE_BRIDGE_TOKEN": "bridge-secret",
            "GH_TOKEN": "github-secret",
        }

        env = build_subprocess_env(planted)

        self.assertTrue(set(planted).isdisjoint(env))

    def test_normal_operator_environment_is_preserved(self):
        env = build_subprocess_env(
            {
                "PATH": "/usr/local/bin:/usr/bin",
                "LANG": "C.UTF-8",
                "AWS_PROFILE": "operator",
                "REPORT_REGION": "morocco",
            }
        )

        self.assertEqual(env["PATH"], "/usr/local/bin:/usr/bin")
        self.assertEqual(env["AWS_PROFILE"], "operator")
        self.assertEqual(env["REPORT_REGION"], "morocco")

    def test_runtime_environment_markers_do_not_leak(self):
        env = build_subprocess_env(
            {
                "PATH": "/opt/pilotage/.venv/bin:/usr/bin",
                "VIRTUAL_ENV": "/opt/pilotage/.venv",
                "CONDA_PREFIX": "/opt/conda",
                "PYTHONHOME": "/opt/pilotage/python",
            }
        )

        self.assertEqual(env["PATH"], "/opt/pilotage/.venv/bin:/usr/bin")
        self.assertNotIn("VIRTUAL_ENV", env)
        self.assertNotIn("CONDA_PREFIX", env)
        self.assertNotIn("PYTHONHOME", env)

    def test_explicit_session_values_are_added_after_scrubbing(self):
        env = build_subprocess_env(
            {"PATH": "/usr/bin", "HERMES_SESSION_ID": "stale"},
            extra={
                "HERMES_SESSION_ID": "current",
                "HERMES_HOME": "/srv/profile",
                "PILOTAGE_HOME": "/srv/profile",
            },
        )

        self.assertEqual(env["HERMES_SESSION_ID"], "current")
        self.assertEqual(env["HERMES_HOME"], "/srv/profile")
        self.assertEqual(env["PILOTAGE_HOME"], "/srv/profile")

    def test_explicit_extra_cannot_reinsert_a_protected_value(self):
        env = build_subprocess_env(
            {"PATH": "/usr/bin"},
            extra={"TELEGRAM_BOT_TOKEN": "secret", "SAFE_SETTING": "yes"},
        )

        self.assertNotIn("TELEGRAM_BOT_TOKEN", env)
        self.assertEqual(env["SAFE_SETTING"], "yes")

    def test_default_base_snapshots_the_current_process_environment(self):
        with mock.patch.dict(
            os.environ,
            {"PILOTAGE_TEST_VISIBLE": "yes", "FIRECRAWL_API_KEY": "secret"},
            clear=True,
        ):
            env = build_subprocess_env()

        self.assertEqual(env["PILOTAGE_TEST_VISIBLE"], "yes")
        self.assertNotIn("FIRECRAWL_API_KEY", env)

    def test_shell_spawn_uses_the_scrubbed_environment(self):
        shell = Shell.__new__(Shell)
        shell.cwd = os.getcwd()
        shell.env = {
            "HERMES_SESSION_ID": "current",
            "TELEGRAM_BOT_TOKEN": "must-not-leak",
        }
        shell._ensure_cwd = mock.Mock()
        process = mock.Mock(pid=123)

        with (
            mock.patch("pilotage.tools.shell.find_bash", return_value="bash"),
            mock.patch(
                "pilotage.tools.shell.subprocess.Popen", return_value=process
            ) as popen,
            mock.patch("pilotage.tools.shell.os.getpgid", return_value=123, create=True),
            mock.patch.dict(
                os.environ,
                {
                    "PATH": os.environ.get("PATH", ""),
                    "FIRECRAWL_API_KEY": "must-not-leak",
                },
                clear=True,
            ),
        ):
            returned = shell._run_bash("printf ok")

        self.assertIs(returned, process)
        child_env = popen.call_args.kwargs["env"]
        self.assertEqual(child_env["HERMES_SESSION_ID"], "current")
        self.assertNotIn("TELEGRAM_BOT_TOKEN", child_env)
        self.assertNotIn("FIRECRAWL_API_KEY", child_env)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
