"""The terminal tool boundary around the persistent Linux shell."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from pilotage.settings import Settings
from pilotage.tools import ToolContext, build_registry
from pilotage.tools.terminal import TERMINAL_SCHEMA, TerminalSession, handle


class _Config:
    def __init__(self, data=None, result_limit=321, state_dir=None):
        self.settings = Settings(data or {})
        self.max_tool_result_chars = result_limit
        self.state_dir = state_dir


def _context(chat_id="chat", data=None, result_limit=321, state_dir=None):
    return ToolContext(
        chat_id=chat_id,
        config=_Config(data, result_limit, state_dir),
    )


class _FakeShell:
    instances = []
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    cancel_seen = threading.Event()

    def __init__(self, cwd="", timeout=0, env=None):
        self.cwd = cwd or "/default"
        self.timeout = timeout
        self.env = dict(env or {})
        self.calls = []
        type(self).instances.append(self)

    def execute(self, command, cwd="", **kwargs):
        self.calls.append((command, cwd, kwargs))
        with type(self).active_lock:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
        try:
            if command == "slow":
                time.sleep(0.03)
            if command == "cancel":
                cancel_event = kwargs["cancel_event"]
                cancel_event.wait(timeout=1)
                if cancel_event.is_set():
                    type(self).cancel_seen.set()
            if command.startswith("cd ") and not cwd:
                self.cwd = command[3:]
                return {"output": "", "returncode": 0, "cwd_observed": True}
            if command == "printenv":
                return {
                    "output": "MY_SERVICE_TOKEN=opaque-value-1234567890\n",
                    "returncode": 0,
                }
            return {"output": f"ran {command}", "returncode": 0}
        finally:
            with type(self).active_lock:
                type(self).active -= 1


class SchemaTests(unittest.TestCase):
    def test_the_build_exposes_the_terminal_group(self):
        registry = build_registry()
        self.assertIsNotNone(registry.get("terminal"))
        self.assertIn("terminal", registry.groups())

    def test_the_schema_asks_only_for_the_interface_we_implement(self):
        parameters = TERMINAL_SCHEMA["parameters"]
        self.assertEqual(parameters["required"], ["command"])
        self.assertEqual(
            set(parameters["properties"]),
            {"command", "timeout", "workdir"},
        )


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _FakeShell.instances = []
        _FakeShell.active = 0
        _FakeShell.max_active = 0
        _FakeShell.cancel_seen = threading.Event()
        patch = mock.patch("pilotage.tools.terminal.Shell", _FakeShell)
        patch.start()
        self.addCleanup(patch.stop)

    async def _run(self, args, context=None):
        return json.loads(await handle(args, context or _context()))

    async def test_a_command_runs_in_the_configured_workspace(self):
        context = _context(
            data={"terminal": {"cwd": "/workspace", "timeout": 45}},
            result_limit=700,
        )
        result = await self._run({"command": "printf hello"}, context)
        self.assertEqual(result, {"output": "ran printf hello", "exit_code": 0})
        shell = _FakeShell.instances[0]
        self.assertEqual((shell.cwd, shell.timeout), ("/workspace", 45))
        self.assertEqual(shell.calls[0][2]["capture_limit"], 700)

    async def test_default_workspace_and_hermes_runtime_env_are_profile_scoped(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_dir = Path(temporary.name)
        context = _context(chat_id="wa-session", state_dir=state_dir)

        await self._run({"command": "pwd"}, context)

        shell = _FakeShell.instances[0]
        self.assertEqual(Path(shell.cwd), state_dir / "workspace")
        self.assertTrue((state_dir / "workspace").is_dir())
        self.assertEqual(
            shell.env,
            {
                "HERMES_SESSION_ID": "wa-session",
                "PILOTAGE_HOME": str(state_dir),
                "HERMES_HOME": str(state_dir),
            },
        )

    async def test_the_shell_persists_for_the_chat(self):
        context = _context()
        await self._run({"command": "export FOO=one"}, context)
        await self._run({"command": "printf $FOO"}, context)
        self.assertEqual(len(_FakeShell.instances), 1)
        self.assertEqual(len(_FakeShell.instances[0].calls), 2)

    async def test_chats_do_not_share_a_shell(self):
        await self._run({"command": "one"}, _context("one"))
        await self._run({"command": "two"}, _context("two"))
        self.assertEqual(len(_FakeShell.instances), 2)

    async def test_a_one_command_workdir_is_passed_without_moving_the_session(self):
        context = _context(data={"terminal": {"cwd": "/workspace"}})
        result = await self._run(
            {"command": "pwd", "workdir": "/srv/report"}, context
        )
        self.assertNotIn("cwd", result)
        shell = _FakeShell.instances[0]
        self.assertEqual(shell.calls[0][1], "/srv/report")
        self.assertEqual(shell.cwd, "/workspace")

    async def test_a_real_directory_change_is_reported(self):
        result = await self._run({"command": "cd /srv/report"})
        self.assertEqual(result["cwd"], "/srv/report")

    async def test_the_call_timeout_overrides_the_session_default(self):
        context = _context(data={"terminal": {"timeout": 45}})
        await self._run({"command": "build", "timeout": 300}, context)
        self.assertEqual(_FakeShell.instances[0].calls[0][2]["timeout"], 300)

    async def test_terminal_output_is_redacted_before_the_model_receives_it(self):
        result = await self._run({"command": "printenv"})
        self.assertNotIn("opaque-value-1234567890", result["output"])
        self.assertIn("MY_SERVICE_TOKEN=", result["output"])

    async def test_exit_zero_with_masked_build_failure_gets_an_advisory(self):
        def _masked_failure(_shell, _command, _cwd="", **_kwargs):
            return {
                "output": "error: could not compile `pilotage` due to 2 errors\n",
                "returncode": 0,
            }

        with mock.patch.object(_FakeShell, "execute", _masked_failure):
            result = await self._run({"command": "cargo build | tail -20"})

        self.assertEqual(result["exit_code"], 0)
        self.assertIn("Treat this run as failed", result["hint"])

    async def test_bad_arguments_are_errors_the_model_can_fix(self):
        for args in (
            {},
            {"command": ""},
            {"command": "ok", "timeout": 0},
            {"command": "ok", "timeout": 1.5},
            {"command": "ok", "workdir": 3},
        ):
            with self.subTest(args=args):
                self.assertIn("error", await self._run(args))

    async def test_catastrophic_and_self_lifecycle_commands_never_reach_the_shell(self):
        for command in (
            "rm -rf /",
            "mkfs.ext4 /dev/sda1",
            "pilotage service stop",
            "systemctl --user restart pilotage-agent@default.service",
        ):
            with self.subTest(command=command):
                result = await self._run({"command": command})
                self.assertIn("Blocked", result["error"])
        self.assertEqual(_FakeShell.instances, [])

    async def test_persistent_store_mutation_never_reaches_the_shell(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_dir = Path(temporary.name).resolve()
        target = state_dir / "memories" / "MEMORY.md"

        result = await self._run(
            {"command": f'echo changed > "{target}"'},
            _context(state_dir=state_dir),
        )

        self.assertIn("Blocked direct persistent-store access", result["error"])
        self.assertEqual(_FakeShell.instances, [])

    async def test_skill_script_execution_and_project_skill_listing_reach_shell(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state_dir = Path(temporary.name).resolve()
        script = state_dir / "skills" / "demo" / "scripts" / "run.py"
        context = _context(state_dir=state_dir)

        executed = await self._run(
            {"command": f'python "{script}"'},
            context,
        )
        listed = await self._run(
            {"command": "cd project && ls skills"},
            context,
        )

        self.assertEqual(executed["exit_code"], 0)
        self.assertEqual(listed["exit_code"], 0)
        self.assertEqual(len(_FakeShell.instances[0].calls), 2)

    async def test_parallel_calls_are_ordered_on_the_same_shell(self):
        context = _context()
        await asyncio.gather(
            handle({"command": "slow"}, context),
            handle({"command": "slow"}, context),
        )
        self.assertEqual(_FakeShell.max_active, 1)
        self.assertEqual(len(_FakeShell.instances), 1)

    async def test_cancelling_the_tool_stops_its_worker_before_returning(self):
        task = asyncio.create_task(handle({"command": "cancel"}, _context()))
        for _ in range(100):
            if _FakeShell.active:
                break
            await asyncio.sleep(0.005)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)

        self.assertTrue(_FakeShell.cancel_seen.is_set())
        self.assertEqual(_FakeShell.active, 0)


posix_only = unittest.skipUnless(os.name == "posix", "the terminal is POSIX-only")


@posix_only
class PosixIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.workspace = Path(tmp.name)
        self.context = _context(data={"terminal": {"cwd": str(self.workspace)}})

    def tearDown(self):
        session = self.context.state.get("terminal")
        if isinstance(session, TerminalSession) and session.shell is not None:
            session.shell.close()

    async def test_the_real_shell_runs_through_the_tool(self):
        result = json.loads(
            await handle({"command": "printf 'hello from terminal'"}, self.context)
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["output"], "hello from terminal")

    async def test_the_real_shell_keeps_and_reports_its_directory(self):
        child = self.workspace / "child"
        child.mkdir()
        moved = json.loads(
            await handle({"command": f"cd {shlex.quote(str(child))}"}, self.context)
        )
        self.assertEqual(Path(moved["cwd"]), child)
        after = json.loads(await handle({"command": "pwd"}, self.context))
        self.assertEqual(Path(after["output"].strip()), child)
        self.assertNotIn("cwd", after)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
