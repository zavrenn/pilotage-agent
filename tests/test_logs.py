import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock

from pilotage import logs


class LogTests(unittest.TestCase):
    def test_filters_python_stderr_severity_and_keeps_traceback(self):
        messages = (
            ("1", "21:00:00 INFO pilotage: ordinary message", "3"),
            ("1", "21:00:01 ERROR pilotage: failed", "3"),
            ("1", "Traceback (most recent call last):", "3"),
            ("1", "  failing_call()", "3"),
            ("1", "ValueError: broken", "3"),
            ("2", "21:00:02 INFO pilotage: another process", "3"),
            ("3", "kernel warning", "4"),
            ("3", "normal journal event", "6"),
            ("1", "21:00:03 INFO pilotage: recovered", "3"),
        )
        data = "".join(json.dumps({
            "__REALTIME_TIMESTAMP": "1700000000000000", "MESSAGE": message,
            "_PID": pid, "PRIORITY": priority,
            "_TRANSPORT": "journal" if pid == "3" else "stdout",
        }) + "\n" for pid, message, priority in messages)
        process = mock.Mock(stdout=io.StringIO(data))
        process.wait.return_value = 0
        process.poll.return_value = 0
        output = io.StringIO()
        with (
            mock.patch.object(logs.shutil, "which", return_value="journalctl"),
            mock.patch.object(logs.subprocess, "Popen", return_value=process) as launch,
            redirect_stdout(output),
        ):
            self.assertEqual(logs.run_logs("work", follow=True, lines=100, level="WARNING", since="1h"), 0)
        written = output.getvalue()
        for expected in ("failed", "Traceback", "failing_call", "ValueError", "kernel warning"):
            self.assertIn(expected, written)
        for unwanted in ("ordinary message", "another process", "normal journal event", "recovered"):
            self.assertNotIn(unwanted, written)
        command = launch.call_args.args[0]
        self.assertIn("pilotage-agent@work.service", command)
        self.assertIn("--follow", command)
        self.assertEqual(command[-2:], ["--since", "1 hours ago"])
        self.assertNotIn("--priority", command)

    def test_unformatted_failures_remain_visible_after_info_at_every_error_filter(self):
        messages = (
            "21:00:00 INFO pilotage: Read environment from /state/.env",
            "Could not initialize private log identities: Permission denied",
            "21:00:01 CRITICAL pilotage: startup failed",
            "Traceback (most recent call last):",
            "  initialize()",
            "PermissionError: denied",
        )
        data = "".join(json.dumps({
            "__REALTIME_TIMESTAMP": "1700000000000000", "MESSAGE": message,
            "_PID": "1", "_TRANSPORT": "stdout", "PRIORITY": "3",
        }) + "\n" for message in messages)
        for level in ("WARNING", "ERROR", "CRITICAL"):
            process = mock.Mock(stdout=io.StringIO(data))
            process.wait.return_value = process.poll.return_value = 0
            output = io.StringIO()
            with (
                self.subTest(level=level),
                mock.patch.object(logs.shutil, "which", return_value="journalctl"),
                mock.patch.object(logs.subprocess, "Popen", return_value=process),
                redirect_stdout(output),
            ):
                self.assertEqual(logs.run_logs("default", level=level), 0)
            self.assertNotIn(messages[0], output.getvalue())
            for message in messages[1:]:
                self.assertIn(message, output.getvalue())

    def test_absolute_since_is_preserved(self):
        stamp = "2026-09-08 20:39:00"
        self.assertEqual(logs.journal_since(stamp), stamp)

    def test_ctrl_c_terminates_the_follower(self):
        stream = mock.MagicMock()
        stream.__iter__.side_effect = KeyboardInterrupt
        process = mock.Mock(stdout=stream)
        process.poll.return_value = None
        with (
            mock.patch.object(logs.shutil, "which", return_value="journalctl"),
            mock.patch.object(logs.subprocess, "Popen", return_value=process),
        ):
            self.assertEqual(logs.run_logs("default", follow=True), 130)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)

    def test_journal_failure_is_not_reported_as_success(self):
        process = mock.Mock(stdout=io.StringIO(""))
        process.poll.return_value = 1
        process.wait.return_value = 1
        with (
            mock.patch.object(logs.shutil, "which", return_value="journalctl"),
            mock.patch.object(logs.subprocess, "Popen", return_value=process),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(logs.run_logs("default"), 1)
