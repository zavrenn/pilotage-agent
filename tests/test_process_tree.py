import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilotage import process_tree
from pilotage.process_tree import ProcessIdentity, snapshot_linux_descendants


def _stat(pid: int, parent: int, started: int, name: str = "worker") -> str:
    # Linux fields 3..22: state, ppid, then 17 values before starttime.
    tail = ["S", str(parent), *("0" for _ in range(17)), str(started)]
    return f"{pid} ({name}) " + " ".join(tail)


class ProcessSnapshotTests(unittest.TestCase):
    def test_descendants_are_snapshotted_with_pid_reuse_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            for pid, parent, started, name in (
                (100, 1, 10, "root"),
                (101, 100, 11, "child (one)"),
                (102, 101, 12, "grand child"),
                (200, 1, 20, "unrelated"),
            ):
                target = proc / str(pid)
                target.mkdir()
                (target / "stat").write_text(
                    _stat(pid, parent, started, name), encoding="utf-8"
                )

            found = snapshot_linux_descendants(100, proc)

        self.assertEqual(
            set(found),
            {ProcessIdentity(101, 11), ProcessIdentity(102, 12)},
        )

    def test_escaped_descendant_gets_term_then_identity_checked_kill(self):
        child = ProcessIdentity(101, 11)
        process = SimpleNamespace(
            pid=100,
            poll=lambda: None,
            wait=mock.Mock(side_effect=subprocess.TimeoutExpired("test", 0)),
        )
        killpg = mock.Mock()
        kill = mock.Mock()
        with (
            mock.patch.object(process_tree.os, "name", "posix"),
            mock.patch.object(process_tree.os, "killpg", killpg, create=True),
            mock.patch.object(process_tree.os, "kill", kill),
            mock.patch.object(
                process_tree,
                "snapshot_linux_descendants",
                return_value=[child],
            ),
            mock.patch.object(process_tree, "_identity_alive", return_value=True),
        ):
            process_tree.terminate_process_tree(
                process,
                pgid=100,
                term_grace_seconds=0,
                kill_grace_seconds=0,
            )

        self.assertEqual(
            killpg.call_args_list,
            [mock.call(100, process_tree._SIGTERM), mock.call(100, process_tree._SIGKILL)],
        )
        self.assertEqual(
            kill.call_args_list,
            [mock.call(101, process_tree._SIGTERM), mock.call(101, process_tree._SIGKILL)],
        )

    @unittest.skipUnless(
        sys.platform.startswith("linux") and Path("/proc").is_dir(),
        "requires a live Linux procfs",
    )
    def test_live_linux_tree_removes_setsid_child_and_its_grandchild(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "descendants"
            grandchild_code = "import time; time.sleep(60)"
            child_code = (
                "import os, pathlib, subprocess, sys, time\n"
                f"grandchild = subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
                f"pathlib.Path({str(registry)!r}).write_text("
                "f'{os.getpid()} {grandchild.pid}', encoding='ascii')\n"
                "time.sleep(60)\n"
            )
            root_code = (
                "import subprocess, sys, time\n"
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
                "start_new_session=True)\n"
                "time.sleep(60)\n"
            )
            root = subprocess.Popen(
                [sys.executable, "-c", root_code],
                start_new_session=True,
            )
            descendants = []
            try:
                deadline = time.monotonic() + 5.0
                raw = ""
                while time.monotonic() < deadline:
                    try:
                        raw = registry.read_text(encoding="ascii").strip()
                    except OSError:
                        raw = ""
                    if len(raw.split()) == 2:
                        break
                    time.sleep(0.05)
                self.assertEqual(len(raw.split()), 2, "descendants did not start")

                for written_pid in raw.split():
                    pid = int(written_pid)
                    stat = process_tree._read_linux_stat(pid)
                    self.assertIsNotNone(stat)
                    descendants.append(ProcessIdentity(pid, stat[1]))

                process_tree.terminate_process_tree(
                    root,
                    pgid=root.pid,
                    term_grace_seconds=0.5,
                    kill_grace_seconds=1.0,
                )

                self.assertIsNotNone(root.poll())
                self.assertFalse(
                    any(process_tree._identity_alive(item) for item in descendants),
                    "a setsid descendant survived tree termination",
                )
            finally:
                for item in descendants:
                    if process_tree._identity_alive(item):
                        try:
                            os.kill(item.pid, signal.SIGKILL)
                        except OSError:
                            pass
                if root.poll() is None:
                    try:
                        os.killpg(root.pid, signal.SIGKILL)
                    except OSError:
                        root.kill()
                try:
                    root.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    root.kill()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
