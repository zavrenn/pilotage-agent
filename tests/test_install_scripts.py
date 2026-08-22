from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class InstallScriptTests(unittest.TestCase):
    def test_bridge_manifest_and_lockfile_describe_the_same_root_package(self):
        manifest = json.loads((ROOT / "bridge" / "package.json").read_text(encoding="utf-8"))
        lock = json.loads((ROOT / "bridge" / "package-lock.json").read_text(encoding="utf-8"))
        locked_root = lock["packages"][""]

        self.assertEqual(lock["name"], manifest["name"])
        self.assertEqual(lock["version"], manifest["version"])
        self.assertEqual(locked_root["dependencies"], manifest["dependencies"])
        self.assertEqual(locked_root["name"], manifest["name"])
        self.assertEqual(locked_root["version"], manifest["version"])
        self.assertEqual(lock["lockfileVersion"], 3)

    def test_installer_uses_the_lockfile(self):
        source = (ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
        self.assertIn("npm ci --silent --no-fund --no-audit", source)
        self.assertNotIn("npm install --silent", source)

    def test_service_is_profile_scoped_and_preflighted(self):
        source = (ROOT / "scripts" / "install-service.sh").read_text(encoding="utf-8")

        for expected in (
            "pilotage-agent@.service",
            "--profile %i run",
            'PILOTAGE_HOME="$state_root"',
            '--profile "$profile" status',
            "realpath -m",
            "Restart=always",
            "KillMode=mixed",
            "TimeoutStopSec=60",
            "systemctl --user enable --now",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)

        self.assertNotIn("\nsudo loginctl enable-linger", source)
        self.assertIn("WorkingDirectory=$escaped_repo", source)
        self.assertNotIn('WorkingDirectory="$escaped_repo"', source)

    @unittest.skipIf(os.name == "nt", "Windows bash does not consume Windows paths")
    def test_shell_scripts_parse(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not installed")
        subprocess.run(
            [
                bash,
                "-n",
                str(ROOT / "scripts" / "install.sh"),
                str(ROOT / "scripts" / "install-service.sh"),
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
