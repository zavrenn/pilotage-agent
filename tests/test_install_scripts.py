from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

from tests.test_bootstrap import BASH


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
            'https://astral.sh/uv/$UV_VERSION/install.sh',
            "/opt/pilotage-uv/bin/uv",
            "uv python install 3.13",
            "UV_PYTHON_INSTALL_DIR=/opt/pilotage-python",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)

        self.assertIn("run this script as root", source)
        self.assertIn("unprivileged service user", source)

    def test_service_is_single_agent_and_preflighted(self):
        source = (ROOT / "scripts" / "install-service.sh").read_text(encoding="utf-8")

        for expected in (
            "pilotage-agent.service",
            'ExecStart="$escaped_bin" run',
            'PILOTAGE_HOME="$state_root"',
            '"$pilotage_bin" status',
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

    @unittest.skipUnless(BASH, "A local Bash interpreter is required")
    def test_shell_scripts_parse(self):
        for script in [ROOT / "install.sh", *(ROOT / "scripts").glob("*.sh")]:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [BASH, "-n"], input=script.read_text(encoding="utf-8"),
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(BASH, "A local Bash interpreter is required")
class UbuntuDependencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pilotage-ubuntu-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trace = self.root / "trace"

    def path(self, path):
        value = path.as_posix()
        return "/" + value[0].lower() + value[2:] if os.name == "nt" else value

    def shell(self, source):
        environment = os.environ.copy()
        for key in ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"):
            environment.pop(key, None)
        environment = {key: value for key, value in environment.items() if not key.startswith("BASH_FUNC_")}
        return subprocess.run(
            [BASH, "--noprofile", "--norc"],
            input=f"set -euo pipefail\nexport TRACE={shlex.quote(self.path(self.trace))}\n" + source,
            cwd=self.root, env=environment, text=True, capture_output=True, timeout=20,
        )

    def executable(self, directory, name, body):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def test_every_environment_uses_managed_python_with_a_system_fallback(self):
        host = self.root / "host"
        managed = self.root / "managed"
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("install.sh", "setup-runtime-environments.sh"):
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            source = source.replace("/opt/pilotage-python/bin", self.path(managed))
            (scripts / name).write_text(source, encoding="utf-8", newline="\n")
        self.executable(host, "uv", 'printf "%s\\n" "$*" >> "$TRACE"')
        self.executable(host, "node", "echo 22")
        self.executable(host, "npm", "exit 0")
        (self.root / "bridge").mkdir()
        for managed_present, system_version, succeeds in ((True, "3.14", True), (False, "3.12", True), (False, "3.14", False)):
            with self.subTest(managed=managed_present, system=system_version):
                self.trace.write_text("", encoding="utf-8")
                if managed_present:
                    self.executable(managed, "python3", '[[ "$*" != *print* ]] || echo 3.13\nexit 0')
                elif (managed / "python3").exists():
                    (managed / "python3").unlink()
                self.executable(host, "python3", f'if [[ "$*" == *print* ]]; then echo {system_version}; exit 0; fi\nexit {int(system_version == "3.14")}')
                result = self.shell(
                    f'export PATH={shlex.quote(self.path(host))}:"$PATH"\n'
                    "bash scripts/install.sh --dependencies-only\n"
                )
                calls = self.trace.read_text(encoding="utf-8").splitlines()
                if succeeds:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(len(calls), 5)
                    python = self.path((managed if managed_present else host) / "python3")
                    for call in calls:
                        self.assertIn(f"--python {python} --no-python-downloads", call)
                        self.assertIn("--locked", call)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Python 3.11 through 3.13", result.stderr)
                    self.assertEqual(calls, [])

    def sql_install(self, *, download_status=0, candidate="18.6.1", install_status=0):
        source = (ROOT / "scripts" / "install-system-dependencies.sh").read_text(encoding="utf-8")
        function = "install_sql_tools() {" + source.split("install_sql_tools() {", 1)[1].split("\ninstall_sql_tools\n", 1)[0]
        package = self.root / "microsoft.deb"
        package.touch()
        return self.shell(function + f'''
VERSION_ID=26.04
dpkg-query() {{ return 1; }}
mktemp() {{ printf '%s\\n' {shlex.quote(self.path(package))}; }}
curl() {{ printf 'download:%s\\n' "${{@: -1}}" >> "$TRACE"; return {download_status}; }}
dpkg() {{ printf 'dpkg:%s\\n' "$*" >> "$TRACE"; }}
apt-cache() {{ printf 'Candidate: %s\\n' {shlex.quote(candidate)}; }}
apt-get() {{ printf 'apt:%s\\n' "$*" >> "$TRACE"; [[ "$1" != install ]] || return {install_status}; }}
ln() {{ printf 'link:%s\\n' "$*" >> "$TRACE"; }}
install_sql_tools
''')

    def test_sql_repository_matches_the_host_release(self):
        result = self.sql_install()
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.trace.read_text(encoding="utf-8")
        self.assertIn("/ubuntu/26.04/packages-microsoft-prod.deb", events)
        self.assertIn("apt:install -y mssql-tools18 unixodbc-dev", events)
        self.assertNotIn("24.04", events)

    def test_unavailable_optional_sql_does_not_block_runtime_installation(self):
        for download_status, candidate in ((22, "18.6.1"), (0, "(none)")):
            with self.subTest(download=download_status, candidate=candidate):
                self.trace.write_text("", encoding="utf-8")
                result = self.sql_install(download_status=download_status, candidate=candidate)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Optional SQL tools skipped", result.stdout)
                events = self.trace.read_text(encoding="utf-8")
                self.assertNotIn("apt:install", events)
                self.assertNotIn("link:", events)

    def test_package_installation_failure_is_reported(self):
        result = self.sql_install(install_status=42)
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertNotIn("Optional SQL tools skipped", result.stdout)
        self.assertNotIn("link:", self.trace.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
