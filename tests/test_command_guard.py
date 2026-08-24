"""Hermes-derived unconditional command guard contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pilotage.tools.command_guard import (
    find_blocked_command,
    find_blocked_python_source,
    find_embedded_self_lifecycle,
)


class HardlineCommandTests(unittest.TestCase):
    def test_catastrophic_commands_are_blocked(self):
        commands = (
            "rm -rf /",
            "rm -rf //",
            "rm -rf /./",
            "rm -rf /etc",
            "rm -rf ~",
            'sudo rm -rf "${HOME}"',
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/nvme0n1",
            "cat image > /dev/sda",
            ":(){ :|:& };:",
            "kill -9 -1",
            "shutdown -h now",
            "reboot",
            "init 6",
            "telinit 0",
            "systemctl poweroff",
            "true && (sudo reboot)",
            'bash -c "mkfs.xfs /dev/sdb"',
            "python -c 'import os; os.system(\"reboot\")'",
            "env -S 'bash -c reboot'",
            "sudo env -S 'bash -c' reboot",
            'eval "cat image > /dev/sda"',
            'echo "$(reboot)"',
            "`shutdown now`",
            "rm${IFS}-rf${IFS}/",
        )
        for command in commands:
            with self.subTest(command=command):
                finding = find_blocked_command(command)
                self.assertIsNotNone(finding)
                self.assertEqual(finding.category, "catastrophic")

    def test_recoverable_commands_and_quoted_prose_are_allowed(self):
        commands = (
            "rm -rf /tmp/build",
            "rm -rf /home/user/scratch",
            "rm -rf ~/Downloads/old",
            "rm -rf /...",
            "dd if=/dev/zero of=./image.bin",
            "echo done > /dev/null",
            "systemctl restart nginx",
            "kill -9 12345",
            'git commit -m "block rm -rf / spellings"',
            'echo "does this workflow use mkfs?"',
            'echo "cat file > /dev/sda is destructive"',
            'echo "classic fork bomb: :(){ :|:& };:"',
            'echo "reboot"',
            "python -c 'print(\"reboot\")'",
            "env -S 'bash -c \"echo reboot\"'",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNone(find_blocked_command(command))

    def test_only_provably_inert_heredoc_bodies_are_treated_as_data(self):
        inert = "cat > /tmp/runbook <<'EOF'\nreboot\nEOF"
        executable = "bash <<'EOF'\nreboot\nEOF"
        expansion_capable = "cat > /tmp/runbook <<EOF\nreboot\nEOF"

        self.assertIsNone(find_blocked_command(inert))
        self.assertIsNotNone(find_blocked_command(executable))
        self.assertIsNotNone(find_blocked_command(expansion_capable))


class SelfLifecycleTests(unittest.TestCase):
    def test_active_service_lifecycle_commands_are_blocked(self):
        commands = (
            "pilotage service stop",
            "/usr/local/bin/pilotage service restart",
            "pilotage -p default service stop",
            "pilotage --profile=default service restart",
            "systemctl --user stop pilotage-agent@default.service",
            "sudo systemctl --user restart pilotage-agent@default",
            "pkill -f pilotage-agent",
            'bash -c "pilotage service stop"',
            "python -c 'import os; os.system(\"pilotage service stop\")'",
            "env -S 'bash -c \"pilotage service stop\"'",
        )
        for command in commands:
            with self.subTest(command=command):
                finding = find_blocked_command(command, current_profile="default")
                self.assertIsNotNone(finding)
                self.assertEqual(finding.category, "self_lifecycle")

    def test_sibling_profiles_and_non_lifecycle_commands_are_allowed(self):
        commands = (
            "pilotage service start",
            "pilotage service status",
            "pilotage -p sibling service stop",
            "systemctl --user restart pilotage-agent@sibling.service",
            "systemctl --user status pilotage-agent@default.service",
            'echo "pilotage service stop"',
            "pkill python",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNone(
                    find_blocked_command(command, current_profile="default")
                )

    def test_referenced_shell_scripts_are_scanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocked = root / "blocked.sh"
            blocked.write_text("#!/bin/sh\npilotage service stop\n", encoding="utf-8")
            safe = root / "safe.sh"
            safe.write_text("#!/bin/sh\nprintf 'healthy\\n'\n", encoding="utf-8")
            long_safe = root / "long-safe.sh"
            long_safe.write_text("printf %s " + "x" * 5_000, encoding="utf-8")

            self.assertIsNotNone(
                find_blocked_command(
                    "bash blocked.sh", cwd=str(root), current_profile="default"
                )
            )
            self.assertIsNone(
                find_blocked_command(
                    ". ./safe.sh", cwd=str(root), current_profile="default"
                )
            )
            self.assertIsNone(
                find_blocked_command(
                    "bash long-safe.sh", cwd=str(root), current_profile="default"
                )
            )

    def test_embedded_cron_command_shape_is_blocked_but_sibling_is_allowed(self):
        self.assertIsNotNone(
            find_embedded_self_lifecycle(
                "At midnight, run pilotage service stop",
                current_profile="default",
            )
        )
        self.assertIsNone(
            find_embedded_self_lifecycle(
                "At midnight, run pilotage -p sibling service stop",
                current_profile="default",
            )
        )
        self.assertIsNone(
            find_embedded_self_lifecycle(
                "x" * 5_000,
                current_profile="default",
            )
        )

    def test_inert_heredoc_lifecycle_prose_is_not_executable(self):
        inert = "cat > /tmp/runbook <<'EOF'\npilotage service stop\nEOF"
        executable = "bash <<'EOF'\npilotage service stop\nEOF"

        self.assertIsNone(find_blocked_command(inert))
        self.assertIsNotNone(find_blocked_command(executable))


class PythonSourceTests(unittest.TestCase):
    def test_literal_process_launches_are_guarded(self):
        sources = (
            'import subprocess\nsubprocess.run(["pilotage", "service", "stop"])',
            'import os\nos.system("systemctl --user restart pilotage-agent@default.service")',
            'import subprocess\nsubprocess.Popen(["mkfs.ext4", "/dev/sda1"])',
            'import subprocess\ncommand = "reboot"\nsubprocess.run(command)',
            (
                'import subprocess\ndef stop():\n'
                '    command = ["pilotage", "service", "stop"]\n'
                '    subprocess.run(command)\nstop()'
            ),
            'import os\nos.system(f"pilotage service stop")',
            (
                'from subprocess import run as launch\ndef stop():\n'
                '    command = "reboot"\n    launch(command)\nstop()'
            ),
            'import asyncio\nasyncio.create_subprocess_exec("kill", "-1")',
        )
        for source in sources:
            with self.subTest(source=source):
                self.assertIsNotNone(
                    find_blocked_python_source(source, current_profile="default")
                )

    def test_non_executed_prose_and_sibling_service_are_allowed(self):
        sources = (
            'print("reboot")',
            'notes = "pilotage service stop"\nprint(notes)',
            'import subprocess\nsubprocess.run(["pilotage", "-p", "sibling", "service", "stop"])',
            (
                'class Report:\n'
                '    def run(self, label):\n        return label\n'
                'report = Report()\nreport.run("reboot")'
            ),
            'def call(label):\n    return label\ncall("reboot")',
        )
        for source in sources:
            with self.subTest(source=source):
                self.assertIsNone(
                    find_blocked_python_source(source, current_profile="default")
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
