"""Hermes-derived unconditional command guard contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pilotage.tools.command_guard import (
    find_blocked_command,
    find_blocked_python_source,
    find_embedded_self_lifecycle,
    find_persistence_store_reference,
)


class PersistencePathGuardTests(unittest.TestCase):
    def test_profile_learning_store_mutations_and_private_state_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            memory_path = state / "memories" / "MEMORY.md"
            skill_path = state / "skills" / "demo" / "SKILL.md"
            audit_path = state / "persistence-audit.db"
            for text in (
                f'echo x > "{memory_path}"',
                f'echo x > "{skill_path}"',
                "echo x > $HERMES_HOME/skills/demo/SKILL.md",
                "echo x > $PILOTAGE_HOME/memories/MEMORY.md",
                f'open("{audit_path}", "wb")',
            ):
                with self.subTest(text=text):
                    finding = find_persistence_store_reference(
                        text,
                        cwd=str(workspace),
                        state_dir=state,
                    )
                    self.assertIsNotNone(finding)
                    self.assertEqual(finding.category, "persistence")

    def test_skill_reads_and_execution_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            skill = state / "skills" / "demo"
            for text, source_kind in (
                (f'type "{skill / "SKILL.md"}"', "shell"),
                (f'python "{skill / "scripts" / "run.py"}"', "shell"),
                (f'cp "{skill / "scripts" / "run.py"}" /tmp/run.py', "shell"),
                (f"runpy.run_path({str(skill / 'scripts' / 'run.py')!r})", "python"),
                (f"Path({str(skill / 'SKILL.md')!r}).read_text()", "python"),
                (
                    f"content = Path({str(skill / 'SKILL.md')!r}).read_text(); "
                    f"Path({str(workspace / 'report.txt')!r}).write_text(content)",
                    "python",
                ),
                (
                    f"Path({str(workspace / 'report.txt')!r}).write_text("
                    f"Path({str(skill / 'SKILL.md')!r}).read_text())",
                    "python",
                ),
            ):
                with self.subTest(text=text):
                    self.assertIsNone(
                        find_persistence_store_reference(
                            text,
                            cwd=str(workspace),
                            state_dir=state,
                            source_kind=source_kind,
                        )
                    )

    def test_python_path_binding_cannot_hide_a_store_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            skill_path = state / "skills" / "demo" / "SKILL.md"
            source = (
                f"target = Path({str(skill_path)!r})\n"
                "target.write_text('changed')"
            )

            finding = find_persistence_store_reference(
                source,
                cwd=str(state / "workspace"),
                state_dir=state,
                source_kind="python",
            )

            self.assertIsNotNone(finding)
            self.assertEqual(finding.category, "persistence")

    def test_shell_carriers_cannot_hide_store_mutations_or_block_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            skill = state / "skills" / "demo"
            mutation = f'bash -lc "echo changed > {skill.as_posix()}/SKILL.md"'
            read = f'sh -c "cat {skill.as_posix()}/SKILL.md"'

            self.assertIsNotNone(
                find_persistence_store_reference(
                    mutation,
                    cwd=str(workspace),
                    state_dir=state,
                )
            )
            self.assertIsNone(
                find_persistence_store_reference(
                    read,
                    cwd=str(workspace),
                    state_dir=state,
                )
            )

    def test_python_keyword_and_aliased_path_mutations_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            skill_path = state / "skills" / "demo" / "SKILL.md"
            for source in (
                f"open(file={str(skill_path)!r}, mode='w').write('changed')",
                (
                    "import shutil\n"
                    f"shutil.copyfile(src='source', dst={str(skill_path)!r})"
                ),
                (
                    "from pathlib import Path as P\n"
                    f"target = {str(skill_path)!r}\n"
                    "P(target).write_text('changed')"
                ),
                (
                    "import pathlib as paths\n"
                    f"target = {str(skill_path)!r}\n"
                    "paths.Path(target).write_text('changed')"
                ),
            ):
                with self.subTest(source=source):
                    self.assertIsNotNone(
                        find_persistence_store_reference(
                            source,
                            cwd=str(workspace),
                            state_dir=state,
                            source_kind="python",
                        )
                    )

    def test_function_local_path_bindings_do_not_leak_to_module_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            source = (
                "from pathlib import Path\n"
                f"target = Path({str(workspace / 'report.txt')!r})\n"
                "def read_skill():\n"
                f"    target = Path({str(state / 'skills/demo/SKILL.md')!r})\n"
                "    return target.read_text()\n"
                "target.write_text('report')"
            )

            self.assertIsNone(
                find_persistence_store_reference(
                    source,
                    cwd=str(workspace),
                    state_dir=state,
                    source_kind="python",
                )
            )

    def test_shell_variable_targets_are_resolved_only_for_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            source = "$PILOTAGE_HOME/skills/demo/SKILL.md"
            mutation = f'target="{source}"; echo changed > "$target"'
            read = f'source="{source}"; cat "$source" > report.txt'

            self.assertIsNotNone(
                find_persistence_store_reference(
                    mutation,
                    cwd=str(workspace),
                    state_dir=state,
                )
            )
            self.assertIsNone(
                find_persistence_store_reference(
                    read,
                    cwd=str(workspace),
                    state_dir=state,
                )
            )

    def test_directly_imported_python_mutators_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            skill_path = state / "skills" / "demo" / "SKILL.md"
            for source in (
                (
                    "from shutil import copyfile\n"
                    f"copyfile(src='source', dst={str(skill_path)!r})"
                ),
                (
                    "from os import remove as delete_file\n"
                    f"delete_file(path={str(skill_path)!r})"
                ),
            ):
                with self.subTest(source=source):
                    self.assertIsNotNone(
                        find_persistence_store_reference(
                            source,
                            cwd=str(workspace),
                            state_dir=state,
                            source_kind="python",
                        )
                    )

    def test_function_parameters_shadow_outer_path_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            source = (
                "from pathlib import Path\n"
                f"target = Path({str(state / 'skills/demo/SKILL.md')!r})\n"
                "def write_report(target):\n"
                "    target.write_text('report')\n"
                f"write_report(Path({str(workspace / 'report.txt')!r}))"
            )

            self.assertIsNone(
                find_persistence_store_reference(
                    source,
                    cwd=str(workspace),
                    state_dir=state,
                    source_kind="python",
                )
            )

    def test_class_bindings_do_not_leak_to_module_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            source = (
                "from pathlib import Path\n"
                f"target = Path({str(workspace / 'report.txt')!r})\n"
                "class SkillReader:\n"
                f"    target = Path({str(state / 'skills/demo/SKILL.md')!r})\n"
                "    content = target.read_text()\n"
                "target.write_text('report')"
            )

            self.assertIsNone(
                find_persistence_store_reference(
                    source,
                    cwd=str(workspace),
                    state_dir=state,
                    source_kind="python",
                )
            )

    def test_function_local_chdir_does_not_leak_to_module_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            skill = state / "skills" / "demo"
            source = (
                "from pathlib import Path\n"
                "import os\n"
                "def inspect_skill():\n"
                f"    os.chdir({str(skill)!r})\n"
                "    Path('SKILL.md').read_text()\n"
                "Path('report.md').write_text('safe workspace output')\n"
            )

            self.assertIsNone(
                find_persistence_store_reference(
                    source,
                    cwd=str(workspace),
                    state_dir=state,
                    source_kind="python",
                )
            )

    def test_cwd_change_cannot_hide_a_direct_relative_store_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            for text, source_kind in (
                ("cd .. && echo poisoned > memories/MEMORY.md", "shell"),
                ("cd .. && touch skills/demo/SKILL.md", "shell"),
                (
                    "os.chdir('..'); open('memories/USER.md', 'w').write('x')",
                    "python",
                ),
            ):
                with self.subTest(text=text):
                    finding = find_persistence_store_reference(
                        text,
                        cwd=str(workspace),
                        state_dir=state,
                        source_kind=source_kind,
                    )
                    self.assertIsNotNone(finding)
                    self.assertEqual(finding.category, "persistence")

    def test_unrelated_workspace_skills_are_not_profile_state(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            self.assertIsNone(
                find_persistence_store_reference(
                    "python scripts/build_skills.py",
                    cwd=str(workspace),
                    state_dir=state,
                )
            )

    def test_working_inside_profile_store_allows_execution_but_not_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            cwd = state / "skills" / "demo"
            self.assertIsNone(
                find_persistence_store_reference(
                    "python scripts/run.py",
                    cwd=str(cwd),
                    state_dir=state,
                )
            )
            self.assertIsNotNone(
                find_persistence_store_reference(
                    "echo changed > SKILL.md",
                    cwd=str(cwd),
                    state_dir=state,
                )
            )

    def test_changing_to_a_project_and_listing_its_skills_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            self.assertIsNone(
                find_persistence_store_reference(
                    "cd project && ls skills",
                    cwd=str(state / "workspace"),
                    state_dir=state,
                )
            )

    def test_relative_paths_resolving_into_profile_stores_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            workspace = state / "workspace"
            for text, cwd, source_kind in (
                ("echo x > ../skills/demo/SKILL.md", workspace, "shell"),
                (
                    "Path('../memories/USER.md').write_text('x')",
                    workspace,
                    "python",
                ),
                ("echo x > skills/demo/SKILL.md", state, "shell"),
                ("echo x > memories/MEMORY.md", state, "shell"),
            ):
                with self.subTest(text=text, cwd=cwd):
                    finding = find_persistence_store_reference(
                        text,
                        cwd=str(cwd),
                        state_dir=state,
                        source_kind=source_kind,
                    )
                    self.assertIsNotNone(finding)
                    self.assertEqual(finding.category, "persistence")


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
