"""Sudo listing success is distinct from permission to execute commands."""

import os
from pathlib import Path
import runpy
import subprocess
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify-agent-sudo.py"
verify_agent_sudo = runpy.run_path(str(SCRIPT))["verify_agent_sudo"]
DENIED = "User agent is not allowed to run sudo on pilotage-demo.\n"


class SudoPolicyTests(unittest.TestCase):
    def listing(self, stdout=DENIED, stderr="", returncode=0):
        return mock.patch.object(
            subprocess, "run",
            return_value=subprocess.CompletedProcess([], returncode, stdout, stderr),
        )

    def test_successful_query_with_explicit_denial_passes(self):
        # The real Ubuntu failure: exit zero, but no grants for the account.
        for hostname in ("pilotage-demo", "pilotage-demo.example.test"):
            with self.subTest(hostname=hostname), self.listing(
                f"User agent is not allowed to run sudo on {hostname}.\n"
            ):
                verify_agent_sudo()

    def test_broad_and_limited_sudo_rules_are_rejected(self):
        for rule in ("(ALL : ALL) ALL", "(root) NOPASSWD: /usr/bin/id", "(backup) /usr/bin/true"):
            output = (
                "Matching Defaults entries for agent on pilotage-demo:\n"
                "    env_reset, use_pty\n\n"
                "User agent may run the following commands on pilotage-demo:\n"
                f"    {rule}\n"
            )
            with self.subTest(rule=rule), self.listing(output):
                with self.assertRaisesRegex(RuntimeError, "sudo rules"):
                    verify_agent_sudo()

    def test_empty_unknown_and_mixed_output_cannot_prove_denial(self):
        for output in (
            "", "Matching Defaults entries for agent on pilotage-demo:\n    env_reset\n",
            DENIED.replace("agent", "operator"),
            "Utilisateur agent non autorise.",
            DENIED + "User agent may run the following commands: (ALL) ALL\n",
        ):
            with self.subTest(output=output), self.listing(output):
                with self.assertRaises(RuntimeError):
                    verify_agent_sudo()

    def test_failed_query_is_not_treated_as_no_permissions(self):
        for status in (1, 2, 127, -9):
            with self.subTest(status=status), self.listing(returncode=status):
                with self.assertRaisesRegex(RuntimeError, "query failed"):
                    verify_agent_sudo()

    def test_policy_diagnostics_stop_verification_even_with_denial(self):
        with self.listing(stderr="sudo: /etc/sudoers: syntax error\n"):
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                verify_agent_sudo()

    def test_missing_command_timeout_and_decode_failure_stop_verification(self):
        for error in (
            FileNotFoundError("sudo"),
            subprocess.TimeoutExpired("sudo", 15),
            UnicodeError("invalid policy output"),
        ):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(subprocess, "run", side_effect=error):
                    with self.assertRaisesRegex(RuntimeError, "could not query"):
                        verify_agent_sudo()

    def test_query_is_noninteractive_bounded_and_uses_fixed_locale(self):
        with mock.patch.dict(os.environ, {"LC_ALL": "fr_FR.UTF-8"}), self.listing() as run:
            verify_agent_sudo()
            self.assertEqual(run.call_args.args[0], ["sudo", "-n", "-l", "-U", "agent"])
            self.assertEqual(run.call_args.kwargs["env"]["LC_ALL"], "C")
            self.assertGreater(run.call_args.kwargs["timeout"], 0)
            self.assertLessEqual(run.call_args.kwargs["timeout"], 15)
            self.assertEqual(os.environ["LC_ALL"], "fr_FR.UTF-8")

    def test_cli_failure_returns_an_operator_error(self):
        with self.listing(stderr="sudo: policy plugin failed\n"):
            with self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(SCRIPT), run_name="__main__")
        self.assertTrue(str(stopped.exception).startswith("error: sudo policy query failed"))


if __name__ == "__main__":
    unittest.main()
