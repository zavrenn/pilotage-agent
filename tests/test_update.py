"""Update safety with local Git and disposable processes; no dependencies installed."""

import io
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilotage import process_tree, update
from pilotage.runtime_lock import ProfileRuntimeLock, RuntimeAlreadyRunning, runtime_lock_is_held


@unittest.skipUnless(shutil.which("git"), "Git required")
class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.checkout = self.root / "checkout"
        self.real_run = subprocess.run
        self.git(self.upstream, "init", "-b", "main")
        (self.upstream / "version.txt").write_text("one\n")
        self.commit("one")
        self.git(self.root, "clone", str(self.upstream), str(self.checkout))
        self.old_head = self.git(self.checkout, "rev-parse", "HEAD")
        (self.upstream / "version.txt").write_text("two\n")
        self.commit("two")
        self.new_head = self.git(self.upstream, "rev-parse", "HEAD")
        self.state = self.root / "state"
        self.profile = SimpleNamespace(name="default", path=self.state)
        self.events = []

    def git(self, root, *args):
        return self.real_run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()

    def commit(self, message):
        self.git(self.upstream, "add", "version.txt")
        self.git(self.upstream, "commit", "-m", message)

    def run_update(self, *, check=False, running=True, install_code=0, extra_profiles=()):
        def run(command, **kwargs):
            if command[0] == "git":
                return self.real_run(command, **kwargs)
            self.events.append("verify")
            return subprocess.CompletedProcess(command, 0)

        def install(root, *, lock_fds):
            self.events.append("install")
            self.assertEqual(len(lock_fds), 2 + len(extra_profiles))
            for fd in lock_fds:
                os.fstat(fd)  # Every inherited descriptor must still be open.
            if isinstance(install_code, BaseException):
                raise install_code
            return install_code

        def service(action, profile):
            self.events.append(action)
            return 0

        self.output, self.errors = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(update, "__file__", str(self.checkout / "pilotage" / "update.py")))
            stack.enter_context(mock.patch.object(update, "sys", SimpleNamespace(platform="linux", stderr=self.errors)))
            stack.enter_context(mock.patch.object(update.shutil, "which", return_value="available"))
            stack.enter_context(mock.patch.object(update.subprocess, "run", side_effect=run))
            stack.enter_context(mock.patch.object(update, "_install", side_effect=install))
            stack.enter_context(mock.patch.object(update, "_service_running", return_value=running))
            stack.enter_context(mock.patch.object(update, "run_service_command", side_effect=service))
            stack.enter_context(mock.patch.object(update.profiles, "list_profiles", return_value=[self.profile, *extra_profiles]))
            stack.enter_context(redirect_stdout(self.output))
            stack.enter_context(redirect_stderr(self.errors))
            return update.run_update("default", check=check)

    def test_check_fetches_without_changing_checkout_or_service(self):
        self.assertEqual(self.run_update(check=True), 0)
        self.assertEqual(self.git(self.checkout, "rev-parse", "HEAD"), self.old_head)
        self.assertEqual(self.events, [])
        self.assertIn("1 new commit", self.output.getvalue())

    def test_updates_then_restarts_only_after_install_and_verification(self):
        self.assertEqual(self.run_update(), 0)
        self.assertEqual(self.git(self.checkout, "rev-parse", "HEAD"), self.new_head)
        self.assertEqual(self.events, ["stop", "install", "verify", "start"])

    def test_local_edits_are_preserved_and_service_is_not_stopped(self):
        (self.checkout / "version.txt").write_text("local edit\n")
        self.assertEqual(self.run_update(), 1)
        self.assertEqual((self.checkout / "version.txt").read_text(), "local edit\n")
        self.assertEqual(self.events, [])

    def test_diverged_branch_is_not_reset_or_merged(self):
        (self.checkout / "local.txt").write_text("local\n")
        self.git(self.checkout, "add", "local.txt")
        self.git(self.checkout, "commit", "-m", "local commit")
        before = self.git(self.checkout, "rev-parse", "HEAD")
        self.assertEqual(self.run_update(), 1)
        self.assertEqual(self.git(self.checkout, "rev-parse", "HEAD"), before)
        self.assertEqual(self.events, [])

    def test_failed_install_leaves_service_stopped_and_can_be_retried(self):
        self.assertEqual(self.run_update(install_code=1), 1)
        self.assertEqual(self.events, ["stop", "install"])
        self.assertIn("stopped", self.errors.getvalue())
        self.events.clear()
        self.assertEqual(self.run_update(running=False), 0)
        self.assertEqual(self.events, ["install", "verify"])

    def test_another_running_profile_blocks_the_update(self):
        other = SimpleNamespace(name="work", path=self.root / "other")
        lock = ProfileRuntimeLock(other.path)
        lock.acquire()
        self.addCleanup(lock.release)
        self.assertEqual(self.run_update(extra_profiles=[other]), 1)
        self.assertEqual(self.events, [])
        self.assertEqual(self.git(self.checkout, "rev-parse", "HEAD"), self.old_head)

    def test_signaled_install_leaves_the_service_stopped(self):
        self.assertEqual(self.run_update(install_code=update.UpdateInterrupted(signal.SIGTERM)), 143)
        self.assertEqual(self.events, ["stop", "install"])
        self.assertIn("Update interrupted", self.errors.getvalue())
        self.assertIn("stopped", self.errors.getvalue())

    def test_foreground_runtime_blocks_dependency_mutation(self):
        lock = ProfileRuntimeLock(self.state)
        lock.acquire()
        self.addCleanup(lock.release)
        self.assertEqual(self.run_update(running=False), 1)
        self.assertEqual(self.events, [])
        self.assertEqual(self.git(self.checkout, "rev-parse", "HEAD"), self.old_head)

    def test_missing_upstream_does_not_select_a_branch_automatically(self):
        self.git(self.checkout, "branch", "--unset-upstream")
        self.assertEqual(self.run_update(), 1)
        self.assertEqual(self.events, [])


class ServicePreflightTests(unittest.TestCase):
    def test_termination_signal_allows_cleanup_and_restores_handlers(self):
        managed = [getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, name)]
        before = {signum: signal.getsignal(signum) for signum in managed}
        with self.assertRaises(update.UpdateInterrupted) as caught:
            with update._update_signals():
                try:
                    signal.raise_signal(signal.SIGTERM)
                finally:
                    for signum in managed:
                        self.assertEqual(signal.getsignal(signum), signal.SIG_IGN)
        self.assertEqual(caught.exception.signum, signal.SIGTERM)
        self.assertEqual({signum: signal.getsignal(signum) for signum in managed}, before)

    def test_interrupted_or_failed_installer_cleans_up_its_process_tree(self):
        for outcome in (subprocess.TimeoutExpired("bash", 1800), KeyboardInterrupt(), 1):
            process = mock.Mock(pid=4321)
            if isinstance(outcome, BaseException):
                process.wait.side_effect = outcome
            else:
                process.wait.return_value = outcome
            with (
                mock.patch.object(update.subprocess, "Popen", return_value=process) as launch,
                mock.patch.object(update, "terminate_process_tree") as terminate,
            ):
                if isinstance(outcome, BaseException):
                    with self.assertRaises(type(outcome)):
                        update._install(Path("/checkout"), lock_fds=(10, 11))
                else:
                    self.assertEqual(update._install(Path("/checkout"), lock_fds=(10, 11)), 1)
                terminate.assert_called_once_with(process, pgid=4321)
                self.assertTrue(launch.call_args.kwargs["start_new_session"])
                self.assertEqual(launch.call_args.kwargs["pass_fds"], (10, 11))

    def test_missing_unit_is_allowed_but_unknown_or_transitioning_states_fail(self):
        cases = (
            ("LoadState=not-found\nActiveState=inactive\n", False),
            ("LoadState=loaded\nActiveState=inactive\n", False),
            ("LoadState=loaded\nActiveState=active\n", True),
            ("LoadState=loaded\nActiveState=activating\n", None),
            ("", None),
        )
        for output, expected in cases:
            with mock.patch.object(update.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output, "")):
                if expected is None:
                    with self.assertRaises(update.UpdateError):
                        update._service_running("default")
                else:
                    self.assertEqual(update._service_running("default"), expected)


@unittest.skipUnless(sys.platform == "linux" and shutil.which("bash"), "requires Linux and bash")
class InstallerInterruptionTests(unittest.TestCase):
    """Real updater/installer processes and OS locks; no dependencies installed."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        self.states = [self.root / "git-lock", self.root / "profile"]
        (self.root / "scripts").mkdir()
        (self.root / "scripts" / "install.sh").write_text(
            "trap '' INT TERM HUP\n"
            "printf '%s' \"$$\" > installer.pid\n"
            "while [[ ! -f release && $SECONDS -lt 15 ]]; do sleep 0.05; done\n",
            encoding="utf-8",
        )
        code = """
import sys
from contextlib import ExitStack
from pathlib import Path
from pilotage import update
from pilotage.runtime_lock import ProfileRuntimeLock
root = Path(sys.argv[1])
try:
    with update._update_signals(), ExitStack() as stack:
        locks = [ProfileRuntimeLock(root / name) for name in ('git-lock', 'profile')]
        for lock in locks:
            lock.acquire()
            stack.callback(lock.release)
        update._install(root, lock_fds=tuple(lock.fileno() for lock in locks))
except update.UpdateInterrupted as exc:
    sys.exit(128 + exc.signum)
"""
        self.parent = subprocess.Popen(
            [sys.executable, "-B", "-c", code, str(self.root)],
            cwd=Path(update.__file__).resolve().parent.parent,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        self.installer = None
        self.addCleanup(self.cleanup_processes)
        self.wait_until(lambda: (self.root / "installer.pid").exists())
        self.wait_until(lambda: (self.root / "installer.pid").read_text().strip())
        pid = int((self.root / "installer.pid").read_text())
        stat = process_tree._read_linux_stat(pid)
        self.assertIsNotNone(stat)
        self.installer = process_tree.ProcessIdentity(pid, stat[1])
        for state in self.states:
            self.assertTrue(runtime_lock_is_held(state))

    def wait_until(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail("installer condition did not become ready")

    def cleanup_processes(self):
        if self.parent.poll() is None:
            process_tree.terminate_process_tree(self.parent, pgid=self.parent.pid)
        if self.installer and process_tree._identity_alive(self.installer):
            os.killpg(self.installer.pid, signal.SIGKILL)
        if self.parent.poll() is None:
            self.parent.kill()
        self.parent.wait(timeout=5)
        self.parent.stderr.close()

    def test_killed_updater_cannot_unlock_a_live_installer(self):
        self.parent.kill()
        self.parent.wait(timeout=5)
        self.assertTrue(process_tree._identity_alive(self.installer))
        for state in self.states:
            self.assertTrue(runtime_lock_is_held(state))
            with self.assertRaises(RuntimeAlreadyRunning):
                ProfileRuntimeLock(state).acquire()
        (self.root / "release").touch()
        self.wait_until(lambda: all(not runtime_lock_is_held(state) for state in self.states))

    def test_sigterm_cleans_up_before_releasing_locks(self):
        self.assert_clean_interruption(signal.SIGTERM)

    def test_ssh_hangup_cleans_up_before_releasing_locks(self):
        self.assert_clean_interruption(signal.SIGHUP)

    def assert_clean_interruption(self, signum):
        self.parent.send_signal(signum)
        self.assertEqual(self.parent.wait(timeout=5), 128 + signum)
        self.assertFalse(process_tree._identity_alive(self.installer))
        for state in self.states:
            self.assertFalse(runtime_lock_is_held(state))
