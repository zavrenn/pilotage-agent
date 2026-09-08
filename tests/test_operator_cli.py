"""Operator shortcuts dispatch without depending on valid runtime config."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from pilotage import main


class OperatorDispatchTests(unittest.TestCase):
    def test_shortcuts_do_not_load_agent_config_or_credentials(self):
        cases = (
            (["restart"], "pilotage.main.run_service_command", ("restart", "work"), {}),
            (["service", "restart"], "pilotage.main.run_service_command", ("restart", "work"), {}),
            (["update", "--check"], "pilotage.update.run_update", ("work",), {"check": True}),
            (["logs", "-f", "--level", "warning", "--since", "1h", "-n", "100"],
             "pilotage.logs.run_logs", ("work",),
             {"follow": True, "lines": 100, "level": "WARNING", "since": "1h"}),
        )
        for argv, target, positional, keywords in cases:
            with (
                self.subTest(argv=argv),
                mock.patch.object(main.profiles, "activate_for_process", return_value=("work", Path("/state"))),
                mock.patch.object(main.Config, "load", side_effect=AssertionError("config loaded")),
                mock.patch.object(main, "load_env_files", side_effect=AssertionError("env loaded")),
                mock.patch(target, return_value=7) as run,
            ):
                self.assertEqual(main.main(["--profile", "work", *argv]), 7)
                run.assert_called_once_with(*positional, **keywords)

    def test_negative_line_count_is_rejected(self):
        with (
            mock.patch.object(main.profiles, "activate_for_process", return_value=("default", Path("/state"))),
            redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            main.main(["logs", "-n", "-1"])
        self.assertEqual(caught.exception.code, 2)
