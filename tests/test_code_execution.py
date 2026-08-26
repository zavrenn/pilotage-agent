"""Contracts for prepared-environment code execution."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilotage.settings import Settings
from pilotage.tools import ToolContext, build_registry
from pilotage.tools import code_execution


class _Config:
    def __init__(self, workspace: Path, settings: dict | None = None):
        self.state_dir = workspace
        self.workspace_dir = workspace
        self.settings = Settings(settings or {})
        self.max_tool_result_chars = 100_000


def _context(workspace: Path, settings: dict | None = None) -> ToolContext:
    return ToolContext("chat", _Config(workspace, settings))


class SchemaTests(unittest.TestCase):
    def test_registry_exposes_one_code_execution_tool(self):
        registry = build_registry()
        self.assertEqual(
            registry.names(["code_execution"]),
            ["execute_code"],
        )
        schema = registry.get("execute_code").schema
        self.assertEqual(
            schema["parameters"]["required"],
            ["code", "environment"],
        )
        self.assertEqual(
            schema["parameters"]["properties"]["environment"]["enum"],
            ["chart", "docs", "excel", "pdf"],
        )

    def test_relative_environment_root_is_anchored_to_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            context = _context(
                Path(directory),
                {"code_execution": {"root": "prepared"}},
            )
            root = code_execution.environment_root(context)
        self.assertEqual(root, code_execution.REPO_ROOT / "prepared")


class EnvironmentScrubTests(unittest.TestCase):
    def test_child_gets_operating_context_but_no_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if os.name == "nt":
                interpreter = root / "chart" / "Scripts" / "python.exe"
            else:
                interpreter = root / "chart" / "bin" / "python"
            workspace = root / "workspace"
            source = {
                "PATH": "/usr/bin",
                "HOME": "/home/agent",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "OPENAI_API_KEY": "secret",
                "DATABASE_PASSWORD": "secret",
                "PILOTAGE_BRIDGE_TOKEN": "secret",
                "AWS_REGION": "eu-west-1",
                "VIRTUAL_ENV": "/wrong/environment",
            }
            child = code_execution._build_child_env(
                interpreter,
                workspace,
                source,
            )

        self.assertEqual(child["HOME"], "/home/agent")
        self.assertEqual(child["LANG"], "C.UTF-8")
        self.assertEqual(child["LC_ALL"], "C.UTF-8")
        self.assertNotIn("OPENAI_API_KEY", child)
        self.assertNotIn("DATABASE_PASSWORD", child)
        self.assertNotIn("PILOTAGE_BRIDGE_TOKEN", child)
        self.assertNotIn("AWS_REGION", child)
        self.assertEqual(child["VIRTUAL_ENV"], str(interpreter.parent.parent))
        self.assertEqual(child["PYTHONPATH"], str(workspace))
        self.assertTrue(child["PATH"].startswith(str(interpreter.parent)))


class ExecutionTests(unittest.TestCase):
    def _run(
        self,
        code: str,
        workspace: Path,
        *,
        settings: dict | None = None,
    ) -> dict:
        context = _context(workspace, settings)
        with mock.patch.object(
            code_execution,
            "interpreter_path",
            return_value=Path(sys.executable),
        ):
            return json.loads(
                code_execution._execute(
                    code,
                    "chart",
                    context,
                    workspace=workspace,
                )
            )

    def test_script_runs_in_session_workspace_and_can_write_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            result = self._run(
                "from pathlib import Path\n"
                "Path('artifact.txt').write_text('ready', encoding='utf-8')\n"
                "print(Path.cwd())\n",
                workspace,
            )
            written = (workspace / "artifact.txt").read_text(encoding="utf-8")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["environment"], "chart")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn(str(workspace), result["output"])
        self.assertEqual(written, "ready")

    def test_nonzero_exit_surfaces_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(
                "import sys\nprint('broken', file=sys.stderr)\nsys.exit(7)\n",
                Path(directory),
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_code"], 7)
        self.assertIn("broken", result["stderr"])
        self.assertIn("code 7", result["error"])

    def test_stdout_keeps_head_and_tail_with_explicit_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(code_execution, "MAX_STDOUT_BYTES", 100):
                result = self._run(
                    "print('HEAD-' + 'x' * 500 + '-TAIL')",
                    Path(directory),
                )

        self.assertTrue(result["stdout_truncated"])
        self.assertGreater(result["stdout_bytes_omitted"], 0)
        self.assertIn("HEAD-", result["output"])
        self.assertIn("-TAIL", result["output"])
        self.assertIn("OUTPUT TRUNCATED", result["output"])

    def test_output_is_ansi_cleaned_and_secret_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(
                "print('\\x1b[31msk-proj-abcdefghijklmnopqrstuvwxyz123456\\x1b[0m')",
                Path(directory),
            )

        self.assertNotIn("\x1b", result["output"])
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", result["output"])
        self.assertIn("sk-pro...", result["output"])

    def test_timeout_kills_the_script_and_reports_it(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(
                "import time\ntime.sleep(5)\n",
                Path(directory),
                settings={"code_execution": {"timeout": 1}},
            )

        self.assertEqual(result["status"], "timeout")
        self.assertIn("timed out after 1s", result["error"])

    def test_cancellation_kills_the_script_and_reports_it(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            context = _context(workspace)
            cancel = threading.Event()
            timer = threading.Timer(0.1, cancel.set)
            timer.start()
            self.addCleanup(timer.cancel)
            with mock.patch.object(
                code_execution,
                "interpreter_path",
                return_value=Path(sys.executable),
            ):
                result = json.loads(
                    code_execution._execute(
                        "import time\ntime.sleep(30)\n",
                        "chart",
                        context,
                        workspace=workspace,
                        cancel_event=cancel,
                    )
                )

        self.assertEqual(result["status"], "cancelled")
        self.assertIn("process tree was killed", result["error"])

    @unittest.skipUnless(os.name == "posix", "setsid is a POSIX boundary")
    def test_timeout_kills_a_setsid_grandchild(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            marker = workspace / "setsid-grandchild-survived"
            result = self._run(
                "import os, pathlib, time\n"
                "if os.fork() == 0:\n"
                "    os.setsid()\n"
                "    if os.fork() == 0:\n"
                "        time.sleep(2)\n"
                f"        pathlib.Path({str(marker)!r}).touch()\n"
                "        os._exit(0)\n"
                "    time.sleep(30)\n"
                "    os._exit(0)\n"
                "time.sleep(30)\n",
                workspace,
                settings={"code_execution": {"timeout": 1}},
            )
            time.sleep(3.0)
            self.assertEqual(result["status"], "timeout")
            self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "posix", "setsid is a POSIX boundary")
    def test_cancellation_kills_a_setsid_grandchild(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            marker = workspace / "cancelled-grandchild-survived"
            ready = workspace / "cancel-tree-ready"
            context = _context(workspace)
            cancel = threading.Event()

            def cancel_after_tree_starts():
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not ready.exists():
                    time.sleep(0.01)
                cancel.set()

            trigger = threading.Thread(
                target=cancel_after_tree_starts,
                daemon=True,
            )
            trigger.start()
            with mock.patch.object(
                code_execution,
                "interpreter_path",
                return_value=Path(sys.executable),
            ):
                result = json.loads(
                    code_execution._execute(
                        "import os, pathlib, time\n"
                        "if os.fork() == 0:\n"
                        "    os.setsid()\n"
                        "    if os.fork() == 0:\n"
                        "        time.sleep(2)\n"
                        f"        pathlib.Path({str(marker)!r}).touch()\n"
                        "        os._exit(0)\n"
                        f"    pathlib.Path({str(ready)!r}).touch()\n"
                        "    time.sleep(30)\n"
                        "    os._exit(0)\n"
                        "time.sleep(30)\n",
                        "chart",
                        context,
                        workspace=workspace,
                        cancel_event=cancel,
                    )
                )
            trigger.join(timeout=1)
            time.sleep(3.0)

            self.assertTrue(ready.exists(), "the escaped tree never started")
            self.assertEqual(result["status"], "cancelled")
            self.assertFalse(marker.exists())

    def test_missing_prepared_environment_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            context = _context(workspace)
            with mock.patch.object(
                code_execution,
                "interpreter_path",
                return_value=workspace / "missing" / "python",
            ):
                result = json.loads(
                    code_execution._execute(
                        "print('never')",
                        "docs",
                        context,
                        workspace=workspace,
                    )
                )

        self.assertIn("Prepared environment 'docs' is unavailable", result["error"])
        self.assertIn("pilotage doctor", result["error"])

    def test_literal_catastrophic_and_self_lifecycle_subprocesses_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for source in (
                'import subprocess\nsubprocess.run(["reboot"])',
                'import subprocess\nsubprocess.run(["pilotage", "service", "stop"])',
            ):
                with self.subTest(source=source):
                    result = self._run(source, workspace)
                    self.assertIn("Blocked", result["error"])
                    self.assertNotIn("status", result)

    def test_command_words_printed_as_data_are_not_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._run('print("reboot")', Path(directory))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["output"].strip(), "reboot")


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_shell_command_shape_and_unknown_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            context = _context(Path(directory))
            command = json.loads(
                await code_execution.handle({"command": "echo hi"}, context)
            )
            unknown = json.loads(
                await code_execution.handle(
                    {"code": "print(1)", "environment": "general"},
                    context,
                )
            )

        self.assertIn("terminal", command["error"])
        self.assertIn("chart, docs, excel, pdf", unknown["error"])

    async def test_cancelling_the_tool_stops_its_worker_before_returning(self):
        started = threading.Event()
        cancelled = threading.Event()

        def fake_execute(*args, cancel_event, **kwargs):
            started.set()
            cancel_event.wait(timeout=1)
            if cancel_event.is_set():
                cancelled.set()
            return json.dumps({"status": "cancelled"})

        with tempfile.TemporaryDirectory() as directory:
            context = _context(Path(directory))
            with mock.patch.object(
                code_execution,
                "_execute",
                side_effect=fake_execute,
            ):
                task = asyncio.create_task(
                    code_execution.handle(
                        {"code": "print(1)", "environment": "chart"},
                        context,
                    )
                )
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.005)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)

        self.assertTrue(cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
