"""The shell session, and the four bugs its shape exists to avoid.

Most of these are regression tests for things that are invisible when they go
wrong: a command that hangs forever because a grandchild is holding a pipe, a
file written with the wrong newlines, a killed command whose children keep
running, a session that never recovers after it deletes its own directory.

The session runs bash on a POSIX host, which is what the agent runs on. On this
Windows development machine the whole module skips; run it under WSL to see it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from pilotage.tools.shell import (
    BoundedOutput,
    Shell,
    read_shell_token,
    rewrite_compound_background,
)

posix_only = unittest.skipUnless(os.name == "posix", "the shell session is POSIX-only")


class BoundedOutputTests(unittest.TestCase):
    """Bounding happens while the stream is read, not after it is finished."""

    def test_output_that_fits_comes_back_whole(self):
        collector = BoundedOutput(100)
        collector.append("hello")
        self.assertEqual(collector.render(), "hello")

    def test_the_middle_is_dropped_and_both_ends_are_kept(self):
        """A listing answers at the top; a build answers at the very bottom."""
        collector = BoundedOutput(200)
        collector.append("START" + "x" * 5000 + "END")
        rendered = collector.render()
        self.assertTrue(rendered.startswith("START"))
        self.assertTrue(rendered.endswith("END"))

    def test_it_says_how_much_it_dropped(self):
        collector = BoundedOutput(200)
        collector.append("y" * 5000)
        self.assertIn("OUTPUT TRUNCATED", collector.render())

    def test_the_limit_holds_however_it_was_written(self):
        for chunk in (1, 7, 500):
            with self.subTest(chunk=chunk):
                collector = BoundedOutput(300)
                text = "z" * 5000
                for i in range(0, len(text), chunk):
                    collector.append(text[i : i + chunk])
                self.assertLessEqual(len(collector.render()), 300)

    def test_the_suffix_is_part_of_the_budget(self):
        """The timeout note has to fit too, or the limit is not a limit."""
        collector = BoundedOutput(300)
        collector.append("q" * 5000)
        self.assertLessEqual(len(collector.render(suffix="\n[timed out]")), 300)

    def test_a_suffix_that_fills_the_budget_wins(self):
        collector = BoundedOutput(10)
        collector.append("ignored")
        self.assertEqual(collector.render(suffix="0123456789ABC"), "3456789ABC")

    def test_it_counts_everything_it_was_given(self):
        collector = BoundedOutput(50)
        collector.append("a" * 400)
        collector.append("b" * 600)
        self.assertEqual(collector.total_chars, 1000)


class BackgroundRewriteTests(unittest.TestCase):
    """`A && B &` backgrounds the whole chain, which is not what it looks like.

    Bash binds `&&` tighter than `&`, so it forks a subshell for the compound
    and runs B inside it in the foreground. The subshell then waits for B —
    forever, when B is a server — and holds our stdout pipe open the whole time.
    """

    def test_a_chained_background_command_is_grouped(self):
        self.assertEqual(
            rewrite_compound_background("cd /tmp && npm start &"),
            "cd /tmp && { npm start & }",
        )

    def test_a_plain_background_command_is_left_alone(self):
        self.assertEqual(rewrite_compound_background("npm start &"), "npm start &")

    def test_a_command_with_no_background_is_left_alone(self):
        self.assertEqual(rewrite_compound_background("a && b"), "a && b")

    def test_a_redirection_is_not_mistaken_for_backgrounding(self):
        for command in ("a && b &> log", "a && b 2>&1", "a && b >&2"):
            with self.subTest(command=command):
                self.assertEqual(rewrite_compound_background(command), command)

    def test_an_ampersand_inside_quotes_is_only_text(self):
        command = 'a && echo "x && y &"'
        self.assertEqual(rewrite_compound_background(command), command)

    def test_rewriting_twice_changes_nothing_the_second_time(self):
        once = rewrite_compound_background("cd /tmp && npm start &")
        self.assertEqual(rewrite_compound_background(once), once)

    def test_a_semicolon_ends_the_chain(self):
        self.assertEqual(rewrite_compound_background("a && b; c &"), "a && b; c &")

    def test_each_line_is_judged_on_its_own(self):
        self.assertEqual(
            rewrite_compound_background("a && b\nc &"),
            "a && b\nc &",
        )

    def test_every_chain_in_a_line_is_rewritten(self):
        self.assertEqual(
            rewrite_compound_background("a && b & c && d &"),
            "a && { b & } c && { d & }",
        )


class ShellTokenTests(unittest.TestCase):
    def test_a_quoted_token_is_read_whole(self):
        token, end = read_shell_token("'a b c' rest", 0)
        self.assertEqual(token, "'a b c'")
        self.assertEqual(end, 7)

    def test_an_operator_ends_a_token(self):
        token, _ = read_shell_token("cmd&& more", 0)
        self.assertEqual(token, "cmd")


@posix_only
class SessionTests(unittest.TestCase):
    """The session survives between commands, because there is no session."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.shell = Shell(cwd=str(self.tmp), timeout=30)
        self.addCleanup(self.shell.close)

    def test_a_command_runs_and_its_output_comes_back(self):
        result = self.shell.execute("echo hello")
        self.assertEqual(result["output"].strip(), "hello")
        self.assertEqual(result["returncode"], 0)

    def test_a_failing_command_reports_its_own_exit_code(self):
        self.assertEqual(self.shell.execute("exit 3")["returncode"], 3)

    def test_stderr_arrives_with_stdout(self):
        result = self.shell.execute("echo oops >&2")
        self.assertIn("oops", result["output"])

    def test_the_marker_never_reaches_the_reader(self):
        """It is bookkeeping; the model must not see it, or it will copy it."""
        result = self.shell.execute("echo hello")
        self.assertNotIn("PILOTAGE_CWD", result["output"])
        self.assertEqual(result["output"], "hello\n")

    def test_an_exported_variable_is_still_set_on_the_next_command(self):
        self.shell.execute("export PILOTAGE_TEST_VAR=carried")
        self.assertEqual(
            self.shell.execute("echo \"$PILOTAGE_TEST_VAR\"")["output"].strip(), "carried"
        )

    def test_a_defined_function_survives_to_the_next_command(self):
        """Filtering the dump by line would leave an orphaned body behind."""
        self.shell.execute("pilotage_greet() { echo greeted; }")
        result = self.shell.execute("pilotage_greet")
        self.assertEqual(result["output"].strip(), "greeted")
        self.assertEqual(result["returncode"], 0)

    def test_a_function_is_not_lost_by_the_commands_that_follow_it(self):
        """Saving only exports would make nvm and pyenv work exactly once."""
        self.shell.execute("pilotage_greet() { echo greeted; }")
        self.shell.execute("echo something else entirely")
        self.assertEqual(self.shell.execute("pilotage_greet")["output"].strip(), "greeted")

    def test_an_alias_survives_to_the_next_command(self):
        self.shell.execute("alias pilotage_ll='echo aliased'")
        self.assertEqual(self.shell.execute("pilotage_ll")["output"].strip(), "aliased")

    def test_changing_directory_sticks(self):
        target = self.tmp / "sub"
        target.mkdir()
        moved = self.shell.execute(f"cd {target}")
        self.assertTrue(moved.get("cwd_observed"))
        steady = self.shell.execute("pwd")
        self.assertEqual(steady["output"].strip(), str(target.resolve()))
        self.assertNotIn("cwd_observed", steady)
        self.assertEqual(Path(self.shell.cwd).resolve(), target.resolve())

    def test_a_directory_given_for_one_command_does_not_stick(self):
        other = self.tmp / "elsewhere"
        other.mkdir()
        self.shell.execute("pwd", cwd=str(other))
        self.assertEqual(Path(self.shell.cwd).resolve(), self.tmp.resolve())

    def test_a_directory_that_cannot_be_entered_is_reported_not_raised(self):
        result = self.shell.execute("pwd", cwd="/no/such/place")
        self.assertEqual(result["returncode"], 126)

    def test_the_session_recovers_after_deleting_its_own_directory(self):
        """Otherwise every later command fails inside the spawn, silently."""
        doomed = self.tmp / "doomed"
        doomed.mkdir()
        self.shell.execute(f"cd {doomed}")
        shutil.rmtree(doomed)
        result = self.shell.execute("echo still here")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["output"].strip(), "still here")

    def test_closing_removes_the_snapshot(self):
        snapshot = self.shell._snapshot_path
        self.assertTrue(os.path.exists(snapshot))
        self.shell.close()
        self.assertFalse(os.path.exists(snapshot))


@posix_only
class StdinTests(unittest.TestCase):
    """Everything the model writes to a file goes through here."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.shell = Shell(cwd=str(self.tmp), timeout=30)
        self.addCleanup(self.shell.close)

    def test_input_reaches_the_command(self):
        result = self.shell.execute("cat", stdin_data="from stdin\n")
        self.assertEqual(result["output"], "from stdin\n")

    def test_newlines_are_not_translated(self):
        """Text-mode stdin would turn every \\n into \\r\\n and corrupt writes."""
        target = self.tmp / "written.txt"
        self.shell.execute(f"cat > {target}", stdin_data="one\ntwo\nthree\n")
        self.assertEqual(target.read_bytes(), b"one\ntwo\nthree\n")

    def test_a_command_that_ignores_its_input_still_finishes(self):
        """A writer left blocked on a full pipe would hang the whole agent."""
        result = self.shell.execute("echo done", stdin_data="x" * 200_000)
        self.assertEqual(result["output"].strip(), "done")
        self.assertEqual(result["returncode"], 0)


@posix_only
class TerminationTests(unittest.TestCase):
    """Commands that do not end on their own, and commands that pretend to."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.shell = Shell(cwd=str(self.tmp), timeout=30)
        self.addCleanup(self.shell.close)

    def test_a_backgrounded_process_does_not_hold_the_command_open(self):
        """It inherits our stdout pipe, so waiting for end-of-file waits for it."""
        started = time.monotonic()
        result = self.shell.execute("sleep 20 & echo started", timeout=15)
        elapsed = time.monotonic() - started
        self.assertEqual(result["output"].strip(), "started")
        self.assertLess(elapsed, 10, "the finished command waited for its background child")

    def test_a_command_that_runs_too_long_is_stopped(self):
        result = self.shell.execute("sleep 30", timeout=1)
        self.assertEqual(result["returncode"], 124)
        self.assertIn("timed out", result["output"])

    def test_the_output_so_far_survives_the_timeout(self):
        result = self.shell.execute("echo partial; sleep 30", timeout=2)
        self.assertIn("partial", result["output"])

    def test_stopping_a_command_stops_its_children_too(self):
        """We spawn into a new session, so signalling bash alone orphans them."""
        marker = self.tmp / "child-survived"
        self.shell.execute(f"( sleep 2; touch {marker} ) & sleep 30", timeout=1)
        time.sleep(3.5)
        self.assertFalse(marker.exists(), "a child outlived the command that started it")

    def test_the_session_still_works_after_a_timeout(self):
        self.shell.execute("sleep 30", timeout=1)
        self.assertEqual(self.shell.execute("echo alive")["output"].strip(), "alive")

    def test_a_stopped_command_is_not_credited_with_a_directory(self):
        """It printed no marker, so its directory is unknown, not unchanged."""
        result = self.shell.execute("cd /; sleep 30", timeout=1)
        self.assertNotIn("cwd_observed", result)
        self.assertEqual(Path(self.shell.cwd).resolve(), self.tmp.resolve())


@posix_only
class CaptureLimitTests(unittest.TestCase):
    """What is kept of a command that prints far too much."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.shell = Shell(cwd=tmp.name, timeout=30)
        self.addCleanup(self.shell.close)
        self.noisy = "for i in $(seq 1 4000); do echo \"line $i\"; done"

    def test_nothing_is_dropped_unless_a_limit_is_asked_for(self):
        """A truncated read is not a shorter file, it is a corrupted one."""
        result = self.shell.execute(self.noisy)
        self.assertIn("line 1\n", result["output"])
        self.assertIn("line 4000", result["output"])

    def test_a_limit_is_honoured_and_keeps_both_ends(self):
        result = self.shell.execute(self.noisy, capture_limit=2000)
        self.assertLessEqual(len(result["output"]), 2000)
        self.assertIn("line 1\n", result["output"])
        self.assertIn("line 4000", result["output"])
        self.assertIn("OUTPUT TRUNCATED", result["output"])

    def test_the_directory_is_still_read_from_truncated_output(self):
        """The marker lives in the tail, which is the half that is kept."""
        result = self.shell.execute(f"cd /tmp; {self.noisy}", capture_limit=2000)
        self.assertTrue(result.get("cwd_observed"))
        self.assertNotIn("PILOTAGE_CWD", result["output"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
