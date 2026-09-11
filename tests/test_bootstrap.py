"""Exercise the public installer without installing packages or touching accounts."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INSTALLER = ROOT / "install.sh"


def find_bash() -> str | None:
    if os.name == "nt":
        # Windows' bash.exe can be a WSL launcher, not a usable local shell.
        git = shutil.which("git")
        if git:
            candidate = Path(git).resolve().parent.parent / "bin" / "bash.exe"
            if candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("bash")


BASH = find_bash()


@unittest.skipUnless(BASH, "A local Bash interpreter is required")
class BootstrapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pilotage-bootstrap-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trace = self.root / "trace"
        self.answers = self.root / "answers"
        self.prompts = self.root / "prompts"

    def shell(self, body: str, *, source: bool = True):
        script = f"source {shlex.quote(INSTALLER.as_posix())}\n" if source else ""
        script += f"trace={shlex.quote(self.trace.as_posix())}\n"
        script += body
        environment = os.environ.copy()
        for key in ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"):
            environment.pop(key, None)
        environment = {
            key: value for key, value in environment.items()
            if not key.startswith("BASH_FUNC_")
        }
        return subprocess.run(
            [BASH, "--noprofile", "--norc"],
            input=script, text=True, capture_output=True, env=environment,
            cwd=self.root, timeout=20,
        )

    def events(self):
        return self.trace.read_text(encoding="utf-8").splitlines() if self.trace.exists() else []

    def main_stubs(self):
        # These replace every mutation boundary. All filesystem writes below
        # are test-owned logs; none of the real provisioning commands run.
        return r'''
id() { printf '0\n'; }
pilotage_platform() { printf 'platform\n' >> "$trace"; }
pilotage_lock() { printf 'lock\n' >> "$trace"; }
pilotage_preflight() { printf 'preflight\n' >> "$trace"; }
pilotage_checkout() { printf 'checkout\n' >> "$trace"; }
pilotage_root() { printf 'root:%s\n' "$*" >> "$trace"; }
pilotage_operator() { printf 'operator:%s\n' "$*" >> "$trace"; }
pilotage_setup() { printf 'setup\n' >> "$trace"; }
'''

    def test_script_parses_and_help_works_through_stdin(self):
        parsed = subprocess.run(
            [BASH, "-n"], input=INSTALLER.read_text(encoding="utf-8"),
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = subprocess.run(
            [BASH, "--noprofile", "--norc", "-s", "--", "--help"],
            input=INSTALLER.read_text(encoding="utf-8"), capture_output=True,
            text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LXC/LXD is not required", result.stdout)
        self.assertIn("--skip-setup", result.stdout)

    def test_partial_download_never_starts_installation(self):
        truncated = INSTALLER.read_text(encoding="utf-8").rsplit("}", 1)[0]
        result = self.shell(
            'uname() { printf "started\\n" >> "$trace"; printf "Darwin\\n"; }\n'
            + truncated,
            source=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.events(), [])

    def test_unknown_options_do_not_reach_provisioning(self):
        result = self.shell(self.main_stubs() + 'pilotage_main --unknown\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown option", result.stderr)
        self.assertEqual(self.events(), [])

    def test_platform_rejects_wrong_os_architecture_and_missing_systemd(self):
        release = self.root / "os-release"
        for system, distribution, version, architecture, systemd, message in (
            ("Darwin", "ubuntu", "24.04", "amd64", 0, "Ubuntu Server"),
            ("Linux", "debian", "12", "amd64", 0, "Ubuntu Server"),
            ("Linux", "ubuntu", "22.04", "amd64", 0, "Ubuntu Server"),
            ("Linux", "ubuntu", "24.04", "arm64", 0, "amd64"),
            ("Linux", "ubuntu", "24.04", "amd64", 1, "systemd"),
        ):
            with self.subTest(system=system, distribution=distribution, version=version, architecture=architecture):
                release.write_text(f'ID={distribution}\nVERSION_ID="{version}"\n', encoding="utf-8", newline="\n")
                result = self.shell(
                    f"uname() {{ printf '%s\\n' {shlex.quote(system)}; }}\n"
                    f"dpkg() {{ printf '%s\\n' {shlex.quote(architecture)}; }}\n"
                    f"systemctl() {{ return {systemd}; }}\n"
                    f"pilotage_platform {shlex.quote(release.as_posix())}\n"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_platform_accepts_ubuntu_without_container_tools(self):
        release = self.root / "os-release"
        release.write_text('ID=ubuntu\nVERSION_ID="24.04"\n', encoding="utf-8", newline="\n")
        result = self.shell(
            "uname() { printf 'Linux\\n'; }\n"
            "dpkg() { printf 'amd64\\n'; }\n"
            "systemctl() { return 0; }\n"
            "lxc() { exit 91; }; lxd() { exit 92; }\n"
            f"pilotage_platform {shlex.quote(release.as_posix())}\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_protection_and_verification_precede_setup(self):
        result = self.shell(self.main_stubs() + 'pilotage_main --skip-setup\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.events(), [
            "platform", "lock", "preflight", "checkout",
            "root:/bin/bash /opt/pilotage-agent/scripts/install-system-dependencies.sh",
            "root:/bin/bash /opt/pilotage-agent/scripts/setup-accounts.sh operator",
            "operator:/bin/bash -c cd /opt/pilotage-agent && bash scripts/install.sh --dependencies-only",
            "root:/bin/bash /opt/pilotage-agent/scripts/install-protection.sh operator",
            "root:systemctl disable pilotage-agent.service",
            "root:/opt/pilotage-agent/.venv/bin/python -I -B /opt/pilotage-agent/scripts/verify-protection.py",
            "setup",
        ])

    def test_failed_dependency_install_never_protects_or_starts_service(self):
        result = self.shell(self.main_stubs() + r'''
pilotage_operator() { printf 'dependency-failure\n' >> "$trace"; return 42; }
pilotage_main
''')
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertIn("runtime dependencies", result.stderr)
        self.assertIn("--resume", result.stderr)
        self.assertFalse(any("install-protection" in event for event in self.events()))
        self.assertNotIn("setup", self.events())

    def test_failed_protection_prints_manual_recovery_and_never_runs_setup(self):
        result = self.shell(self.main_stubs() + r'''
pilotage_root() {
  printf 'root:%s\n' "$*" >> "$trace"
  case "$*" in *install-protection.sh*) return 43 ;; esac
}
pilotage_main
''')
        self.assertEqual(result.returncode, 43, result.stderr)
        self.assertIn("run as root", result.stderr)
        self.assertIn("systemctl disable pilotage-agent.service", result.stderr)
        self.assertIn("verify-protection.py", result.stderr)
        self.assertNotIn("--resume", result.stderr)
        self.assertNotIn("setup", self.events())

    def test_sudo_handoff_clears_environment_and_preserves_options(self):
        result = self.shell(self.main_stubs() + r'''
id() { printf '1000\n'; }
sudo() {
  [[ "$1" == /usr/bin/env && "$2" == -i && "$3" == HOME=/root ]]
  [[ "$5" == /bin/bash && "$6" == -c ]]
  [[ "$7" == *'pilotage_main "$@"'* ]]
  [[ "$8" == pilotage-installer && "$9" == --resume && "${10}" == --skip-setup ]]
  bash -n -c "$7"
  printf 'sudo\n' >> "$trace"
}
pilotage_main --resume --skip-setup
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.events(), ["platform", "sudo"])

    def test_privileged_commands_receive_a_clean_environment(self):
        result = self.shell(r'''
export PILOTAGE_BOOTSTRAP_TEST_SECRET=private
export PYTHONPATH=/untrusted
export NODE_OPTIONS=--untrusted
pilotage_root /usr/bin/env
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("HOME=/root", result.stdout)
        self.assertIn("DEBIAN_FRONTEND=noninteractive", result.stdout)
        self.assertNotIn("PILOTAGE_BOOTSTRAP_TEST_SECRET", result.stdout)
        self.assertNotIn("PYTHONPATH", result.stdout)
        self.assertNotIn("NODE_OPTIONS", result.stdout)

    def test_operator_commands_drop_privileges_and_use_the_operator_home(self):
        result = self.shell(r'''
operator_home=/home/operator
runuser() { printf '%s\n' "$@"; }
pilotage_operator git --version
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = result.stdout.splitlines()
        self.assertEqual(arguments[:5], ["-u", "operator", "--", "/usr/bin/env", "-i"])
        self.assertIn("HOME=/home/operator", arguments)
        self.assertIn("USER=operator", arguments)
        self.assertIn("GIT_TERMINAL_PROMPT=0", arguments)
        self.assertEqual(arguments[-2:], ["git", "--version"])

    def preflight_context(self):
        paths = {
            "manifest": self.root / "deployment.json",
            "launcher": self.root / "bin" / "pilotage",
            "state_dir": self.root / "agent" / ".pilotage-agent",
            "checkout": self.root / "runtime",
            "operator_home": self.root / "operator",
            "agent_home": self.root / "agent",
        }
        # Bash's native path avoids a Windows drive colon in mocked passwd rows.
        return 'fixture_root="$(pwd)"\n' + "\n".join(
            f'{key}="$fixture_root/{path.relative_to(self.root).as_posix()}"'
            for key, path in paths.items()
        ) + "\nresume=false\nsystemctl() { printf 'not-found\\n'; }\n"

    def test_existing_deployment_or_state_is_preserved(self):
        for relative, message in (("deployment.json", "pilotage update"), ("agent/.pilotage-agent", "does not migrate")):
            with self.subTest(path=relative):
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("keep this", encoding="utf-8")
                result = self.shell(self.preflight_context() + "pilotage_preflight\n")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual(path.read_text(encoding="utf-8"), "keep this")
                path.unlink()

    def test_existing_checkout_requires_explicit_resume(self):
        (self.root / "runtime").mkdir()
        result = self.shell(self.preflight_context() + "pilotage_preflight\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--resume", result.stderr)
        self.assertTrue((self.root / "runtime").is_dir())

    def test_installed_runtime_points_to_setup_commands_when_resumed(self):
        (self.root / "deployment.json").write_text("installed", encoding="utf-8")
        result = self.shell(self.preflight_context() + "resume=true\npilotage_preflight\n")
        self.assertNotEqual(result.returncode, 0)
        for command in ("sudo -iu operator", "pilotage login", "pilotage model",
                        "pilotage whatsapp", "pilotage telegram",
                        "sudo systemctl enable --now pilotage-agent.service", "pilotage doctor"):
            self.assertIn(command, result.stderr)

    def test_resume_refuses_unexpected_remote_and_local_edits(self):
        checkout = self.root / "runtime"
        (checkout / ".git").mkdir(parents=True)
        operator = self.root / "operator"
        operator.mkdir()
        preserved = checkout / "local.txt"
        preserved.write_text("do not overwrite", encoding="utf-8")
        for remote, status, message in (
            ("https://example.invalid/wrong.git", "", "unexpected origin"),
            ("https://github.com/zavrenn/pilotage-agent.git", " M local.txt", "local changes"),
        ):
            with self.subTest(remote=remote, status=status):
                result = self.shell(
                    f"checkout={shlex.quote(checkout.as_posix())}\n"
                    f"operator_home={shlex.quote(operator.as_posix())}\n"
                    f"remote={shlex.quote(remote)}\nstatus={shlex.quote(status)}\n"
                    + r'''
pilotage_root() { :; }
id() { return 0; }
stat() { printf 'operator\n'; }
find() { return 0; }
pilotage_operator() {
  case "$*" in
    *'remote get-url origin') printf '%s\n' "$remote" ;;
    *'symbolic-ref --short HEAD') printf 'main\n' ;;
    *'rev-parse '*) printf 'same-commit\n' ;;
    *'status --porcelain'*) printf '%s\n' "$status" ;;
    *) exit 90 ;;
  esac
}
pilotage_checkout
'''
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual(preserved.read_text(encoding="utf-8"), "do not overwrite")

    def test_untrusted_checkout_permissions_are_rejected_before_git(self):
        checkout = self.root / "runtime"
        (checkout / ".git").mkdir(parents=True)
        operator = self.root / "operator"
        operator.mkdir()
        result = self.shell(
            f"checkout={shlex.quote(checkout.as_posix())}\n"
            f"operator_home={shlex.quote(operator.as_posix())}\n"
            + r'''
pilotage_root() { :; }
id() { return 0; }
stat() { printf 'operator\n'; }
find() { printf '%s/unsafe\n' "$checkout"; }
pilotage_operator() { printf 'git-called\n' >> "$trace"; exit 90; }
pilotage_checkout
'''
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not writable by other accounts", result.stderr)
        self.assertEqual(self.events(), [])

    def test_process_inspection_error_cannot_be_treated_as_an_idle_agent(self):
        result = self.shell(self.preflight_context() + r'''
resume=true
id() {
  case "$*" in
    operator) return 1 ;;
    '-nG agent') printf 'agent\n' ;;
    *) printf '1002\n' ;;
  esac
}
getent() { printf 'agent:x:1002:1002::%s:/bin/bash\n' "$agent_home"; }
pgrep() { return 2; }
pilotage_preflight
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Could not verify", result.stderr)

    def setup_stubs(self, answers: str):
        self.answers.write_text(answers, encoding="utf-8", newline="\n")
        return (
            f"answers={shlex.quote(self.answers.as_posix())}\n"
            f"prompts={shlex.quote(self.prompts.as_posix())}\n"
            + r'''
skip_setup=false
pilotage_terminal() { exec 3<"$answers" 4>"$prompts"; }
passwd() { printf 'operator P\n'; }
pilotage_operator() { printf '%s\n' "$*" >> "$trace"; }
'''
        )

    def test_guided_setup_does_not_start_without_explicit_choice(self):
        result = self.shell(self.setup_stubs("y\n2\nn\n") + "pilotage_setup\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.events(), [
            "/usr/local/bin/pilotage login", "/usr/local/bin/pilotage model",
            "/usr/local/bin/pilotage telegram",
        ])

    def test_guided_setup_can_configure_both_channels_then_check_readiness(self):
        result = self.shell(self.setup_stubs("y\ninvalid\n3\ny\n") + "pilotage_setup\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.events(), [
            "/usr/local/bin/pilotage login", "/usr/local/bin/pilotage model",
            "/usr/local/bin/pilotage whatsapp", "/usr/local/bin/pilotage telegram",
            "sudo systemctl enable --now pilotage-agent.service", "/usr/local/bin/pilotage doctor",
        ])
        self.assertIn("Choose 0, 1, 2 or 3", self.prompts.read_text(encoding="utf-8"))

    def test_authentication_failure_stops_guided_setup(self):
        result = self.shell(self.setup_stubs("y\n1\ny\n") + r'''
pilotage_operator() { printf '%s\n' "$*" >> "$trace"; return 23; }
pilotage_setup
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.events(), ["/usr/local/bin/pilotage login"])

    def test_skip_setup_never_requests_a_terminal_or_credentials(self):
        result = self.shell(r'''
skip_setup=true
pilotage_terminal() { exit 91; }
passwd() { exit 92; }
pilotage_operator() { exit 93; }
pilotage_setup
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("service is stopped", result.stdout)
        self.assertIn("disabled at boot", result.stdout)
        self.assertIn("passwd operator", result.stdout)


if __name__ == "__main__":
    unittest.main()
