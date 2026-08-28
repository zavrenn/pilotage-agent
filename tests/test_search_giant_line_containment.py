"""Content search bounds giant matching lines before subprocess transport."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from pilotage.tools.file_operations import ShellFileOperations

GIANT_LINE_CHARS = 2 * 1024 * 1024
STDOUT_CEILING = 256 * 1024


class _CommandRecorder:
    cwd = "/workspace"

    def __init__(self):
        self.commands = []

    def execute(self, command, **_kwargs):
        self.commands.append(command)
        return {"output": "", "returncode": 1}


class SearchPipelineConstructionTests(unittest.TestCase):
    def setUp(self):
        self.env = _CommandRecorder()
        self.operations = ShellFileOperations(self.env)

    def test_rg_caps_content_but_not_path_or_count_rows(self):
        for mode, expected in (
            ("content", True),
            ("files_only", False),
            ("count", False),
        ):
            with self.subTest(mode=mode):
                self.env.commands.clear()
                self.operations._search_with_rg("needle", ".", None, 10, 0, mode, 0)
                command = self.env.commands[-1]
                self.assertEqual("--max-columns" in command, expected)
                self.assertEqual("--max-columns-preview" in command, expected)

    def test_grep_clips_content_but_not_path_or_count_rows(self):
        for mode, expected in (
            ("content", True),
            ("files_only", False),
            ("count", False),
        ):
            with self.subTest(mode=mode):
                self.env.commands.clear()
                self.operations._search_with_grep("needle", ".", None, 10, 0, mode, 0)
                command = self.env.commands[-1]
                self.assertEqual("cut -c1-2000" in command, expected)


@unittest.skipUnless(os.name == "posix", "real search pipelines require POSIX bash")
class PosixGiantLineIntegrationTests(unittest.TestCase):
    class _Environment:
        def __init__(self, cwd: Path):
            self.cwd = str(cwd)
            self.max_stdout = 0

        def execute(self, command, cwd=None, **_kwargs):
            completed = subprocess.run(
                ["bash", "-c", command],
                cwd=cwd or self.cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            output = completed.stdout + completed.stderr
            self.max_stdout = max(self.max_stdout, len(output))
            return {"output": output, "returncode": completed.returncode}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "trace.json").write_text(
            '{"needle":"' + "x" * GIANT_LINE_CHARS + '"}',
            encoding="utf-8",
        )
        (self.root / "small.py").write_text("needle = 1\n", encoding="utf-8")

    def test_real_available_engines_keep_the_hit_without_transporting_the_line(self):
        if shutil.which("bash") is None:
            self.skipTest("bash is not installed")
        tested = 0
        for engine in ("rg", "grep"):
            if shutil.which(engine) is None:
                continue
            if engine == "grep" and shutil.which("cut") is None:
                continue
            with self.subTest(engine=engine):
                environment = self._Environment(self.root)
                operations = ShellFileOperations(environment)
                operations._has_command = lambda command, selected=engine: command == selected

                result = operations.search(
                    "needle",
                    path=str(self.root),
                    target="content",
                )

                self.assertIsNone(result.error)
                self.assertEqual(
                    {Path(match.path).name for match in result.matches},
                    {"small.py", "trace.json"},
                )
                self.assertTrue(all(len(match.content) <= 500 for match in result.matches))
                self.assertLess(environment.max_stdout, STDOUT_CEILING)
                tested += 1
        if not tested:
            self.skipTest("neither rg nor grep/cut is installed")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
