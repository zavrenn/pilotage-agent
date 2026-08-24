"""Operator cron commands remain usable without a model connection."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from pilotage import main
from pilotage.config import Config
from pilotage.cron.jobs import CronStore


class CronCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = mock.patch.dict(
            "os.environ", {"PILOTAGE_HOME": str(self.root)}, clear=False
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *arguments):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main.main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_full_management_flow_needs_no_auth_or_model(self):
        code, output, error = self.run_cli(
            "cron", "create", "1h", "--prompt", "Send status", "--name", "report"
        )
        self.assertEqual((code, error), (0, ""))
        job_id = output.split("\t", 1)[0]
        self.assertEqual(len(job_id), 12)

        code, output, _ = self.run_cli("cron", "list")
        self.assertEqual(code, 0)
        self.assertIn(job_id, output)

        self.assertEqual(self.run_cli("cron", "pause", job_id)[0], 0)
        code, output, _ = self.run_cli("cron", "list", "--all")
        self.assertEqual(code, 0)
        self.assertIn("paused", output)
        self.assertEqual(self.run_cli("cron", "resume", job_id)[0], 0)
        self.assertEqual(self.run_cli("cron", "run", job_id)[0], 0)

        config = Config.load()
        store = CronStore(config.state_dir, timezone_name=config.cron_timezone)
        store.save_output(job_id, "saved result")
        code, output, _ = self.run_cli("cron", "output", job_id)
        self.assertEqual(code, 0)
        self.assertEqual(output.strip(), "saved result")

        self.assertEqual(self.run_cli("cron", "remove", job_id)[0], 0)
        self.assertNotIn(job_id, self.run_cli("cron", "list", "--all")[1])

    def test_disabled_scheduler_refuses_new_work_but_allows_listing(self):
        (self.root / "config.yaml").write_text(
            "cron:\n  enabled: false\n", encoding="utf-8"
        )
        code, _, error = self.run_cli(
            "cron", "create", "1h", "--prompt", "Send status"
        )
        self.assertEqual(code, 1)
        self.assertIn("disabled", error)
        self.assertEqual(self.run_cli("cron", "list")[0], 0)

    def test_operator_can_set_and_clear_workdir_and_toolsets(self):
        workdir = self.root / "project"
        workdir.mkdir()
        code, output, error = self.run_cli(
            "cron",
            "create",
            "1h",
            "--prompt",
            "Send status",
            "--tool",
            "web",
            "--workdir",
            str(workdir),
        )
        self.assertEqual((code, error), (0, ""))
        job_id = output.split("\t", 1)[0]
        store = CronStore(Config.load().state_dir)
        job = store.resolve_job(job_id)
        self.assertEqual(job["enabled_toolsets"], ["web"])
        self.assertEqual(job["workdir"], str(workdir.resolve()))

        self.assertEqual(
            self.run_cli(
                "cron",
                "update",
                job_id,
                "--clear-tools",
                "--clear-workdir",
            )[0],
            0,
        )
        cleared = store.resolve_job(job_id)
        self.assertIsNone(cleared["enabled_toolsets"])
        self.assertIsNone(cleared["workdir"])

    def test_operator_can_choose_a_declared_home_delivery(self):
        code, output, error = self.run_cli(
            "cron",
            "create",
            "1h",
            "--prompt",
            "Send status",
            "--deliver",
            "telegram",
        )
        self.assertEqual((code, error), (0, ""))
        job_id = output.split("\t", 1)[0]
        job = CronStore(Config.load().state_dir).resolve_job(job_id)
        self.assertEqual(job["deliver"], "telegram")

        self.assertEqual(
            self.run_cli("cron", "update", job_id, "--deliver", "local")[0],
            0,
        )
        self.assertEqual(
            CronStore(Config.load().state_dir).resolve_job(job_id)["deliver"],
            "local",
        )


if __name__ == "__main__":
    unittest.main()
