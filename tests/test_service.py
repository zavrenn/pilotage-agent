"""Operator start, stop, and inspection of one installed profile service."""

from __future__ import annotations

import subprocess
import unittest
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from pilotage.service import run_service_command, unit_name


class ServiceCommandTests(unittest.TestCase):
    def run_command(self, action, result):
        commands = []

        def run(command):
            commands.append(list(command))
            return result

        stdout, stderr = StringIO(), StringIO()
        with (
            mock.patch("pilotage.service.shutil.which", return_value="/bin/systemctl"),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = run_service_command(action, "work", run=run)
        return code, stdout.getvalue(), stderr.getvalue(), commands

    def test_unit_name_is_profile_scoped(self):
        self.assertEqual(unit_name("work"), "pilotage-agent@work.service")

    def test_start_and_stop_target_only_the_selected_user_unit(self):
        result = subprocess.CompletedProcess([], 0, "", "")
        for action in ("start", "stop"):
            with self.subTest(action=action):
                code, output, error, commands = self.run_command(action, result)
                self.assertEqual((code, error), (0, ""))
                self.assertIn(action.title(), output)
                self.assertEqual(
                    commands,
                    [["systemctl", "--user", action, "pilotage-agent@work.service"]],
                )

    def test_status_reports_active_state_and_pid(self):
        result = subprocess.CompletedProcess(
            [],
            0,
            "LoadState=loaded\nUnitFileState=enabled\nActiveState=active\n"
            "SubState=running\nMainPID=123\n",
            "",
        )
        code, output, error, commands = self.run_command("status", result)
        self.assertEqual((code, error), (0, ""))
        self.assertIn("active (running)", output)
        self.assertIn("pid=123", output)
        self.assertEqual(commands[0][:4], ["systemctl", "--user", "show", "pilotage-agent@work.service"])

    def test_missing_systemd_fails_clearly(self):
        stderr = StringIO()
        with (
            mock.patch("pilotage.service.shutil.which", return_value=None),
            redirect_stderr(stderr),
        ):
            code = run_service_command("status", "default")
        self.assertEqual(code, 1)
        self.assertIn("requires Ubuntu systemd", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
