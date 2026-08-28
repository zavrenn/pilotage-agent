from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
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
        self.assertIn("uv sync", source)
        self.assertIn("--locked", source)
        self.assertIn("--no-python-downloads", source)
        self.assertIn('sync_environment "$repo_root/.venv" --no-dev', source)
        self.assertNotIn("pip install --editable", source)

    def test_installer_builds_all_prepared_environments_before_runtime(self):
        installer = (ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
        setup = (
            ROOT / "scripts" / "setup-runtime-environments.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("bash scripts/setup-runtime-environments.sh", installer)
        self.assertIn("--only-group", setup)
        self.assertIn("--no-install-project", setup)
        self.assertIn("--locked", setup)
        self.assertNotIn("requirements-", setup)
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        groups = project["dependency-groups"]
        self.assertEqual(set(groups), {"chart", "docs", "excel", "pdf"})
        for name in ("chart", "docs", "excel", "pdf"):
            self.assertIn(f"install_environment {name}", setup)
            self.assertTrue(groups[name])
            self.assertTrue(all("==" in dependency for dependency in groups[name]))

    def test_python_lock_covers_the_project_and_every_prepared_environment(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
        self.assertEqual(
            lock["requires-python"].replace(" ", ""),
            project["project"]["requires-python"].replace(" ", ""),
        )
        package = next(
            package for package in lock["package"]
            if package["name"] == project["project"]["name"]
        )
        self.assertEqual(
            set(package["dev-dependencies"]),
            set(project["dependency-groups"]),
        )
        self.assertGreater(len(lock["package"]), 1)

    def test_system_builder_supplies_every_required_runtime(self):
        source = (
            ROOT / "scripts" / "install-system-dependencies.sh"
        ).read_text(encoding="utf-8")

        for expected in (
            'VERSION_ID:-}" = "24.04"',
            "setup_${NODE_MAJOR}.x",
            "google-chrome-stable",
            "tesseract-ocr-ara",
            "tesseract-ocr-eng",
            "tesseract-ocr-fra",
            "libreoffice-writer",
            "pandoc",
            "poppler-utils",
            "qpdf",
            "fonts-noto-core",
            "mssql-tools18",
            "sqlcmd",
            'UV_VERSION="0.12.0"',
            '"uv==$UV_VERSION"',
            "/opt/pilotage-uv/bin/uv",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)

        self.assertIn("run this script as root", source)
        self.assertIn("unprivileged service user", source)

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
            "TimeoutStopSec=$timeout_stop_sec",
            "systemctl --user enable --now",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)

        self.assertNotIn("\nsudo loginctl enable-linger", source)
        self.assertIn("WorkingDirectory=$escaped_repo", source)
        self.assertNotIn('WorkingDirectory="$escaped_repo"', source)

        budgets = {
            name: int(value)
            for name, value in re.findall(
                r"^(stop_intake_budget|shutdown_drain_budget|"
                r"channel_cleanup_budget|shutdown_headroom)=([0-9]+)$",
                source,
                flags=re.MULTILINE,
            )
        }
        self.assertEqual(
            set(budgets),
            {
                "stop_intake_budget",
                "shutdown_drain_budget",
                "channel_cleanup_budget",
                "shutdown_headroom",
            },
        )
        self.assertGreaterEqual(sum(budgets.values()), 90)

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
                str(ROOT / "scripts" / "install-system-dependencies.sh"),
                str(ROOT / "scripts" / "setup-runtime-environments.sh"),
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
