"""Integration contract for the extracted Hermes file-tool slice."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilotage import media
from pilotage.approvals import ApprovalOutcome
from pilotage.settings import Settings
from pilotage.tools import ToolContext, build_registry
from pilotage.tools.file_operations import (
    PatchResult,
    SearchMatch,
    SearchResult,
    ShellFileOperations,
    WriteResult,
)
from pilotage.tools import file_safety
from pilotage.tools.files import (
    FileSession,
    PATCH_SCHEMA,
    READ_FILE_SCHEMA,
    SEARCH_FILES_SCHEMA,
    WRITE_FILE_SCHEMA,
    _bounded_search_dict,
    _patch,
    _ApprovalRequired,
    _run,
    _write,
    handle_patch,
    handle_read_file,
    handle_search_files,
    handle_write_file,
)
from pilotage.tools.terminal import TerminalSession, handle as handle_terminal


VALID_SKILL = (
    "---\n"
    "name: demo\n"
    "description: Demo workflow.\n"
    "version: 1.0.0\n"
    "---\n\n"
    "# Demo\n\nFollow this workflow.\n"
)


class _Config:
    def __init__(self, workspace: Path):
        self.settings = Settings({"terminal": {"cwd": str(workspace)}})
        self.max_tool_result_chars = 100_000
        self.state_dir = workspace
        self.workspace_dir = workspace


class SchemaTests(unittest.TestCase):
    def test_the_file_group_contains_the_complete_slice(self):
        self.assertEqual(
            build_registry().names(["file"]),
            ["patch", "read_file", "search_files", "write_file"],
        )

    def test_schemas_match_the_four_interfaces(self):
        self.assertEqual(READ_FILE_SCHEMA["parameters"]["required"], ["path"])
        self.assertEqual(WRITE_FILE_SCHEMA["parameters"]["required"], ["path", "content"])
        self.assertEqual(PATCH_SCHEMA["parameters"]["required"], ["mode"])
        self.assertEqual(SEARCH_FILES_SCHEMA["parameters"]["required"], ["pattern"])

    def test_whatsapp_auth_state_is_read_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            credential = state / "whatsapp" / "creds.json"
            with mock.patch.object(file_safety, "state_dir", return_value=state):
                error = file_safety.get_read_block_error(str(credential))
        self.assertIn("authentication state", error)

    def test_named_profile_cannot_use_file_tools_on_default_or_sibling_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            selected = root / "profiles" / "selected"
            sibling = root / "profiles" / "sibling"
            workspace = selected / "workspace"
            for path in (workspace, sibling):
                path.mkdir(parents=True)

            with (
                mock.patch.object(file_safety, "state_dir", return_value=selected),
                mock.patch.object(
                    file_safety,
                    "default_state_root",
                    return_value=root,
                ),
            ):
                self.assertIn(
                    "another agent profile",
                    file_safety.get_read_block_error(
                        str(root / "codex-auth.json")
                    ),
                )
                self.assertIn(
                    "another agent profile",
                    file_safety.get_read_block_error(
                        str(sibling / "memories" / "MEMORY.md")
                    ),
                )
                self.assertIn(
                    "another agent profile",
                    file_safety.get_write_denied_error(
                        str(root / "workspace" / "file.txt")
                    ),
                )
                self.assertIsNone(
                    file_safety.get_read_block_error(str(workspace / "own.txt"))
                )
                self.assertIsNone(
                    file_safety.get_write_denied_error(str(workspace / "own.txt"))
                )

    def test_default_profile_cannot_use_file_tools_on_named_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            named = root / "profiles" / "named" / "workspace" / "file.txt"
            with (
                mock.patch.object(file_safety, "state_dir", return_value=root),
                mock.patch.object(
                    file_safety,
                    "default_state_root",
                    return_value=root,
                ),
            ):
                self.assertIn(
                    "another agent profile",
                    file_safety.get_read_block_error(str(named)),
                )
                self.assertIn(
                    "another agent profile",
                    file_safety.get_write_denied_error(str(named)),
                )

    def test_own_control_plane_requires_its_dedicated_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            with mock.patch.object(file_safety, "state_dir", return_value=state):
                for relative in (
                    ".runtime.lock",
                    "active_profile",
                    "bridge.pid",
                    "codex-auth.json.lock",
                    "conversations.db-wal",
                    "SOUL.md",
                    "cron/.jobs.lock",
                    "cron/jobs.json",
                ):
                    path = state / relative
                    self.assertIsNotNone(file_safety.get_read_block_error(str(path)))
                    self.assertIsNotNone(file_safety.get_write_denied_error(str(path)))

                for relative in ("cron/output/result.md", "memories/MEMORY.md"):
                    self.assertIsNotNone(
                        file_safety.get_write_denied_error(str(state / relative))
                    )

    def test_auto_loaded_agents_files_fail_closed_until_approvals_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ("AGENTS.md", "agents.md", "AgEnTs.Md"):
                error = file_safety.get_write_denied_error(str(root / name))
                self.assertIn("requires approval", error)
            self.assertIsNone(
                file_safety.get_write_denied_error(str(root / "notes.md"))
            )

    def test_skill_write_requires_an_explicit_category_grant(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            target = home / "skills" / "demo" / "SKILL.md"
            context = ToolContext("chat", _Config(home))
            shell = SimpleNamespace(cwd=str(home))
            operations = mock.Mock()

            def write(path, content):
                written = Path(path)
                written.parent.mkdir(parents=True, exist_ok=True)
                written.write_text(content, encoding="utf-8")
                return WriteResult(bytes_written=len(content), verified=True)

            operations.write_file.side_effect = write
            with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(home)}):
                with self.assertRaisesRegex(
                    RuntimeError, "Write skill file"
                ) as approval:
                    _write(
                        {"path": str(target), "content": VALID_SKILL},
                        context,
                        shell,
                        operations,
                    )
                self.assertNotIn("Follow this workflow", approval.exception.summary)
                self.assertIn(VALID_SKILL, approval.exception.review_content)
                result = json.loads(
                    _write(
                        {"path": str(target), "content": VALID_SKILL},
                        context,
                        shell,
                        operations,
                        approved_categories=frozenset({"skills"}),
                    )
                )

            self.assertTrue(result["verified"])
            self.assertEqual(target.read_text(encoding="utf-8"), VALID_SKILL)

    def test_invalid_skill_write_is_rejected_before_approval_or_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            target = home / "skills" / "demo" / "SKILL.md"
            context = ToolContext("chat", _Config(home))
            shell = SimpleNamespace(cwd=str(home))
            operations = mock.Mock()
            invalid = VALID_SKILL.replace("version: 1.0.0\n", "")

            with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(home)}):
                result = json.loads(
                    _write(
                        {"path": str(target), "content": invalid},
                        context,
                        shell,
                        operations,
                    )
                )

            self.assertIn("version", result["error"])
            self.assertFalse(target.exists())
            operations.write_file.assert_not_called()

    def test_invalid_skill_patch_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            target = home / "skills" / "demo" / "SKILL.md"
            target.parent.mkdir(parents=True)
            target.write_text(VALID_SKILL, encoding="utf-8")
            context = ToolContext("chat", _Config(home))
            shell = SimpleNamespace(cwd=str(home))
            operations = mock.Mock()

            def patch(path, old, new, _replace_all):
                destination = Path(path)
                replacement = destination.with_name("replacement.tmp")
                replacement.write_text(
                    destination.read_text(encoding="utf-8").replace(old, new),
                    encoding="utf-8",
                )
                os.replace(replacement, destination)
                return PatchResult(success=True, files_modified=[str(destination)])

            operations.patch_replace.side_effect = patch
            with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(home)}):
                result = json.loads(
                    _patch(
                        {
                            "mode": "replace",
                            "path": str(target),
                            "old_string": "version: 1.0.0\n",
                            "new_string": "",
                        },
                        context,
                        shell,
                        operations,
                        approved_categories=frozenset({"skills"}),
                    )
                )

            self.assertFalse(result["success"])
            self.assertTrue(result["restored"])
            self.assertIn("version", result["error"])
            self.assertEqual(target.read_text(encoding="utf-8"), VALID_SKILL)

    def test_invalid_v4a_skill_patch_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            target = home / "skills" / "demo" / "SKILL.md"
            target.parent.mkdir(parents=True)
            target.write_text(VALID_SKILL, encoding="utf-8")
            context = ToolContext("chat", _Config(home))
            shell = SimpleNamespace(cwd=str(home))

            def apply(_parsed, _operations):
                replacement = target.with_name("replacement.tmp")
                replacement.write_text(
                    VALID_SKILL.replace("version: 1.0.0\n", ""),
                    encoding="utf-8",
                )
                os.replace(replacement, target)
                return PatchResult(success=True, files_modified=[str(target)])

            patch = """*** Begin Patch
*** Update File: skills/demo/SKILL.md
@@
-version: 1.0.0
*** End Patch"""
            with (
                mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(home)}),
                mock.patch(
                    "pilotage.tools.files.apply_v4a_operations",
                    side_effect=apply,
                ),
            ):
                result = json.loads(
                    _patch(
                        {"mode": "patch", "patch": patch},
                        context,
                        shell,
                        mock.Mock(),
                        approved_categories=frozenset({"skills"}),
                    )
                )

            self.assertFalse(result["success"])
            self.assertTrue(result["restored"])
            self.assertIn("version", result["error"])
            self.assertEqual(target.read_text(encoding="utf-8"), VALID_SKILL)

    def test_agents_symlink_name_is_checked_before_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "rules.md"
            target.write_text("Operator authority.", encoding="utf-8")
            try:
                (root / "AGENTS.md").symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            context = ToolContext("chat", _Config(root))
            shell = SimpleNamespace(cwd=str(root))
            operations = mock.Mock()

            written = json.loads(
                _write(
                    {"path": "AGENTS.md", "content": "Agent authority."},
                    context,
                    shell,
                    operations,
                )
            )
            patched = json.loads(
                _patch(
                    {
                        "mode": "replace",
                        "path": "AGENTS.md",
                        "old_string": "Operator",
                        "new_string": "Agent",
                    },
                    context,
                    shell,
                    operations,
                )
            )

            self.assertIn("requires approval", written["error"])
            self.assertIn("requires approval", patched["error"])
            self.assertEqual(
                target.read_text(encoding="utf-8"), "Operator authority."
            )
            operations.write_file.assert_not_called()
            operations.patch_replace.assert_not_called()


class OutputBoundTests(unittest.TestCase):
    def test_large_search_pages_keep_valid_json_and_advance(self):
        result = SearchResult(
            matches=[SearchMatch(f"file-{i}.txt", i + 1, "\\" * 500) for i in range(1000)],
            total_count=1000,
            truncated=True,
        )
        data = _bounded_search_dict(result, 0)
        encoded = json.dumps(data, ensure_ascii=False)
        self.assertLessEqual(len(encoded), 95_000)
        self.assertGreater(data["next_offset"], 0)
        self.assertTrue(data["truncated"])

    def test_one_oversized_search_item_still_advances(self):
        result = SearchResult(
            matches=[SearchMatch("x" * 100_000, 1, "match")],
            total_count=1,
            truncated=True,
        )
        data = _bounded_search_dict(result, 0)
        self.assertEqual(data["next_offset"], 1)


class ApprovalReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_skill_proposal_is_attached_then_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            context = ToolContext("chat", _Config(workspace))
            shell = SimpleNamespace(cwd=str(workspace))
            context.state["terminal"] = TerminalSession(shell=shell)
            context.state["file"] = FileSession(shell=shell, operations=object())
            proposal = "complete proposal\n" + "x" * 2_000 + "\nTAIL-MARKER"
            review_paths = []

            async def approve(category, summary):
                self.assertEqual(category, "skills")
                self.assertIn("Full proposal attached", summary)
                review = Path(summary.rsplit("MEDIA:", 1)[1].strip())
                review_paths.append(review)
                self.assertTrue(review.is_file())
                self.assertEqual(review.read_text(encoding="utf-8"), proposal)
                attachments, cleaned = media.extract_outbound(summary, (workspace,))
                self.assertEqual([item.path for item in attachments], [review])
                self.assertNotIn("MEDIA:", cleaned)
                return ApprovalOutcome(True, "approved")

            def handler(_args, _context, _shell, _operations, *, approved_categories):
                if "skills" not in approved_categories:
                    raise _ApprovalRequired("skills", "Short summary", proposal)
                return json.dumps({"success": True})

            context.approval_request = approve
            result = json.loads(await _run(handler, {}, context))

            self.assertTrue(result["success"])
            self.assertEqual(len(review_paths), 1)
            self.assertFalse(review_paths[0].exists())

@unittest.skipIf(os.name == "nt", "The production file stack requires Linux/bash")
class FileToolCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.context = ToolContext("chat", _Config(self.workspace))

    async def call(self, handler, args):
        return json.loads(await handler(args, self.context))

    def tearDown(self):
        terminal = self.context.state.get("terminal")
        shell = getattr(terminal, "shell", None)
        if shell is not None:
            shell.close()


class SessionAndGuardTests(FileToolCase):
    async def test_relative_paths_follow_the_live_terminal_cwd(self):
        child = self.workspace / "child"
        child.mkdir()
        terminal = await self.call(handle_terminal, {"command": "cd child"})
        self.assertEqual(Path(terminal["cwd"]), child)
        result = await self.call(
            handle_write_file, {"path": "answer.txt", "content": "inside"}
        )
        self.assertNotIn("error", result)
        self.assertEqual((child / "answer.txt").read_text(), "inside")

    async def test_absolute_and_parent_paths_match_hermes_behavior(self):
        sibling = self.workspace.parent / f"pilotage-{self.workspace.name}.txt"
        self.addCleanup(lambda: sibling.unlink(missing_ok=True))
        result = await self.call(
            handle_write_file,
            {"path": str(sibling), "content": "allowed because terminal has the same reach"},
        )
        self.assertNotIn("error", result)
        self.assertEqual(sibling.read_text(), "allowed because terminal has the same reach")

    async def test_environment_files_are_neither_read_nor_written(self):
        secret = self.workspace / ".env"
        secret.write_text("TOKEN=secret")
        read = await self.call(handle_read_file, {"path": ".env"})
        write = await self.call(
            handle_write_file, {"path": ".env", "content": "TOKEN=lost"}
        )
        self.assertIn("secrets", read["error"])
        self.assertIn("secrets", write["error"])
        self.assertEqual(secret.read_text(), "TOKEN=secret")

    async def test_auto_loaded_agents_file_cannot_be_written_or_patched(self):
        created = await self.call(
            handle_write_file,
            {"path": "AGENTS.md", "content": "New authority."},
        )
        self.assertIn("requires approval", created["error"])
        self.assertFalse((self.workspace / "AGENTS.md").exists())

        path = self.workspace / "AGENTS.md"
        path.write_text("Operator authority.", encoding="utf-8")
        patched = await self.call(
            handle_patch,
            {
                "mode": "replace",
                "path": "AGENTS.md",
                "old_string": "Operator",
                "new_string": "Agent",
            },
        )
        self.assertIn("requires approval", patched["error"])
        self.assertEqual(path.read_text(encoding="utf-8"), "Operator authority.")

    async def test_skill_write_uses_live_approval_and_denial_writes_nothing(self):
        requests = []

        async def deny(category, summary):
            requests.append((category, summary))
            return ApprovalOutcome(False, "denied", "Keep the old skill")

        self.context.approval_request = deny
        target = self.workspace / "skills" / "demo" / "SKILL.md"
        with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.workspace)}):
            result = await self.call(
                handle_write_file,
                {"path": str(target), "content": VALID_SKILL},
            )

        self.assertEqual(result["approval"], "denied")
        self.assertFalse(target.exists())
        self.assertEqual(requests[0][0], "skills")
        self.assertIn(str(target), requests[0][1])

    async def test_approved_skill_write_runs_once(self):
        async def approve(_category, _summary):
            return ApprovalOutcome(True, "approved")

        self.context.approval_request = approve
        target = self.workspace / "skills" / "demo" / "SKILL.md"
        with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.workspace)}):
            result = await self.call(
                handle_write_file,
                {"path": str(target), "content": VALID_SKILL},
            )

        self.assertNotIn("error", result)
        self.assertTrue(result["verified"])
        self.assertTrue(target.is_file())

    async def test_agents_symlink_cannot_redirect_a_write_to_an_unprotected_name(self):
        target = self.workspace / "rules.md"
        target.write_text("Operator authority.", encoding="utf-8")
        try:
            (self.workspace / "AGENTS.md").symlink_to(target)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

        written = await self.call(
            handle_write_file,
            {"path": "AGENTS.md", "content": "Agent authority."},
        )
        patched = await self.call(
            handle_patch,
            {
                "mode": "replace",
                "path": "AGENTS.md",
                "old_string": "Operator",
                "new_string": "Agent",
            },
        )

        self.assertIn("requires approval", written["error"])
        self.assertIn("requires approval", patched["error"])
        self.assertEqual(
            target.read_text(encoding="utf-8"), "Operator authority."
        )


class ReadTests(FileToolCase):
    async def test_read_is_numbered_and_paginated(self):
        (self.workspace / "notes.txt").write_text("one\ntwo\nthree\n")
        first = await self.call(
            handle_read_file, {"path": "notes.txt", "offset": 2, "limit": 1}
        )
        self.assertTrue(first["content"].startswith("2|two"))
        self.assertEqual(first["total_lines"], 3)
        self.assertTrue(first["truncated"])

    async def test_empty_and_utf16_files_are_text(self):
        (self.workspace / "empty.txt").write_bytes(b"")
        empty = await self.call(handle_read_file, {"path": "empty.txt"})
        self.assertEqual((empty["content"], empty["total_lines"]), ("", 0))
        (self.workspace / "wide.txt").write_bytes(
            b"\xff\xfe" + "bonjour".encode("utf-16-le")
        )
        wide = await self.call(handle_read_file, {"path": "wide.txt"})
        self.assertEqual(wide["content"], "1|bonjour")

    async def test_binary_documents_are_outside_this_text_slice(self):
        (self.workspace / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        result = await self.call(handle_read_file, {"path": "image.png"})
        self.assertIn("binary", result["error"])

    async def test_pagination_is_clamped_like_hermes(self):
        (self.workspace / "one.txt").write_text("one\n")
        result = await self.call(
            handle_read_file, {"path": "one.txt", "offset": -10, "limit": 0}
        )
        self.assertTrue(result["content"].startswith("1|one"))

    async def test_long_lines_are_bounded_before_the_registry_cap(self):
        (self.workspace / "long.txt").write_text("x" * 50_000)
        result = await self.call(handle_read_file, {"path": "long.txt"})
        self.assertIn("... [truncated]", result["content"])
        self.assertLess(len(result["content"]), 3_000)

    async def test_heavily_escaped_content_still_returns_valid_json(self):
        (self.workspace / "slashes.txt").write_text(
            "\n".join("\\" * 1900 for _ in range(100))
        )
        raw = await handle_read_file({"path": "slashes.txt"}, self.context)
        parsed = json.loads(raw)
        self.assertIn("content", parsed)
        self.assertGreater(parsed["next_offset"], 1)
        self.assertLess(len(raw), 100_000)


class WriteTests(FileToolCase):
    async def test_write_creates_parents_and_verifies_exact_bytes(self):
        result = await self.call(
            handle_write_file, {"path": "reports/today.txt", "content": "complete\n"}
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["bytes_written"], len(b"complete\n"))
        self.assertEqual((self.workspace / "reports/today.txt").read_bytes(), b"complete\n")

    async def test_invalid_structured_content_never_touches_the_file(self):
        for name, invalid in (("data.json", "{"), ("data.yaml", "key: [")):
            with self.subTest(name=name):
                path = self.workspace / name
                path.write_text('{"valid": true}' if name.endswith("json") else "valid: true\n")
                before = path.read_bytes()
                result = await self.call(
                    handle_write_file, {"path": name, "content": invalid}
                )
                self.assertIn("syntax validation", result["error"])
                self.assertEqual(path.read_bytes(), before)

    async def test_read_display_text_is_not_written_back_as_source(self):
        result = await self.call(
            handle_write_file, {"path": "bad.txt", "content": "1|first\n2|second\n"}
        )
        self.assertIn("line-numbered", result["error"])
        self.assertFalse((self.workspace / "bad.txt").exists())

    async def test_opaque_documents_and_existing_pdfs_are_protected(self):
        document = await self.call(
            handle_write_file, {"path": "report.docx", "content": "text"}
        )
        self.assertIn("opaque", document["error"])
        pdf = self.workspace / "report.pdf"
        pdf.write_bytes(b"%PDF-old")
        result = await self.call(
            handle_write_file, {"path": "report.pdf", "content": "%PDF-new"}
        )
        self.assertIn("PDF", result["error"])
        self.assertEqual(pdf.read_bytes(), b"%PDF-old")

    async def test_unknown_extension_binary_content_is_not_overwritten(self):
        path = self.workspace / "payload.unknown"
        path.write_bytes(b"\x00\x01\x02private")
        result = await self.call(
            handle_write_file, {"path": "payload.unknown", "content": "replacement"}
        )
        self.assertIn("binary", result["error"])
        self.assertEqual(path.read_bytes(), b"\x00\x01\x02private")

    async def test_existing_bom_line_endings_and_mode_survive_rewrite(self):
        path = self.workspace / "windows.txt"
        path.write_bytes(b"\xef\xbb\xbfold\r\nline\r\n")
        path.chmod(0o755)
        result = await self.call(
            handle_write_file, {"path": "windows.txt", "content": "new\nline\n"}
        )
        self.assertTrue(result["verified"])
        self.assertEqual(path.read_bytes(), b"\xef\xbb\xbfnew\r\nline\r\n")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755)

    async def test_a_post_write_failure_restores_the_original(self):
        path = self.workspace / "restore.txt"
        path.write_text("original")
        original_write = ShellFileOperations.write_file

        def write_then_report_failure(instance, target, content, pre_content=None):
            written = original_write(instance, target, content, pre_content=pre_content)
            self.assertIsNone(written.error)
            return WriteResult(error="simulated verification failure")

        with mock.patch.object(
            ShellFileOperations, "write_file", new=write_then_report_failure
        ):
            result = await self.call(
                handle_write_file, {"path": "restore.txt", "content": "changed"}
            )
        self.assertTrue(result["restored"])
        self.assertEqual(path.read_text(), "original")


class ReplacePatchTests(FileToolCase):
    async def test_unique_replace_returns_a_diff_and_preserves_crlf(self):
        path = self.workspace / "app.py"
        path.write_bytes(b"first\r\nvalue = 1\r\nlast\r\n")
        result = await self.call(
            handle_patch,
            {
                "mode": "replace",
                "path": "app.py",
                "old_string": "value = 1",
                "new_string": "value = 2",
            },
        )
        self.assertTrue(result["success"])
        self.assertIn("+value = 2", result["diff"])
        self.assertEqual(path.read_bytes(), b"first\r\nvalue = 2\r\nlast\r\n")

    async def test_non_unique_replace_requires_replace_all(self):
        path = self.workspace / "many.txt"
        path.write_text("same\nsame\n")
        one = await self.call(
            handle_patch,
            {"mode": "replace", "path": "many.txt", "old_string": "same", "new_string": "new"},
        )
        self.assertIn("Found 2 matches", one["error"])
        all_result = await self.call(
            handle_patch,
            {
                "mode": "replace",
                "path": "many.txt",
                "old_string": "same",
                "new_string": "new",
                "replace_all": True,
            },
        )
        self.assertTrue(all_result["success"])
        self.assertEqual(path.read_text(), "new\nnew\n")

    async def test_fuzzy_matching_accepts_indentation_differences(self):
        path = self.workspace / "app.py"
        path.write_text("def run():\n    old()\n")
        result = await self.call(
            handle_patch,
            {
                "mode": "replace",
                "path": "app.py",
                "old_string": "def run():\n  old()",
                "new_string": "def run():\n  new()",
            },
        )
        self.assertTrue(result["success"])
        self.assertEqual(path.read_text(), "def run():\n  new()\n")

    async def test_repeating_an_applied_patch_is_a_successful_noop(self):
        (self.workspace / "done.txt").write_text("new value")
        result = await self.call(
            handle_patch,
            {
                "mode": "replace",
                "path": "done.txt",
                "old_string": "old value",
                "new_string": "new value",
            },
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["no_change"])


class V4APatchTests(FileToolCase):
    async def test_skill_patch_that_breaks_metadata_is_rolled_back(self):
        async def approve(_category, _summary):
            return ApprovalOutcome(True, "approved")

        self.context.approval_request = approve
        target = self.workspace / "skills" / "demo" / "SKILL.md"
        target.parent.mkdir(parents=True)
        target.write_text(VALID_SKILL, encoding="utf-8")
        patch = """*** Begin Patch
*** Update File: skills/demo/SKILL.md
@@
-version: 1.0.0
*** End Patch"""
        with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.workspace)}):
            result = await self.call(handle_patch, {"mode": "patch", "patch": patch})

        self.assertFalse(result["success"])
        self.assertTrue(result["restored"])
        self.assertIn("version", result["error"])
        self.assertEqual(target.read_text(encoding="utf-8"), VALID_SKILL)

    async def test_one_patch_can_add_update_move_and_delete(self):
        (self.workspace / "update.txt").write_text("before\n")
        moved = self.workspace / "move.txt"
        moved.write_text("moving\n")
        moved.chmod(0o755)
        (self.workspace / "delete.txt").write_text("gone\n")
        patch = """*** Begin Patch
*** Add File: added.txt
+created
*** Update File: update.txt
@@
-before
+after
*** Move File: move.txt -> moved.txt
*** Delete File: delete.txt
*** End Patch"""
        result = await self.call(handle_patch, {"mode": "patch", "patch": patch})
        self.assertTrue(result["success"])
        self.assertEqual((self.workspace / "added.txt").read_text(), "created")
        self.assertEqual((self.workspace / "update.txt").read_text(), "after\n")
        self.assertEqual((self.workspace / "moved.txt").read_text(), "moving\n")
        self.assertEqual(stat.S_IMODE((self.workspace / "moved.txt").stat().st_mode), 0o755)
        self.assertFalse((self.workspace / "delete.txt").exists())

    async def test_validation_failure_changes_nothing(self):
        first = self.workspace / "first.txt"
        second = self.workspace / "second.txt"
        first.write_text("old first\n")
        second.write_text("old second\n")
        patch = """*** Begin Patch
*** Update File: first.txt
@@
-old first
+new first
*** Update File: second.txt
@@
-missing
+new second
*** End Patch"""
        result = await self.call(handle_patch, {"mode": "patch", "patch": patch})
        self.assertIn("no files were modified", result["error"])
        self.assertEqual(first.read_text(), "old first\n")
        self.assertEqual(second.read_text(), "old second\n")

    async def test_apply_failure_restores_files_already_written(self):
        first = self.workspace / "first.txt"
        second = self.workspace / "second.txt"
        first.write_text("old first\n")
        second.write_text("old second\n")
        patch = """*** Begin Patch
*** Update File: first.txt
@@
-old first
+new first
*** Update File: second.txt
@@
-old second
+new second
*** End Patch"""
        original_write = ShellFileOperations.write_file

        def fail_second(instance, target, content, pre_content=None):
            if target == str(second):
                return WriteResult(error="simulated disk failure")
            return original_write(instance, target, content, pre_content=pre_content)

        with mock.patch.object(ShellFileOperations, "write_file", new=fail_second):
            result = await self.call(handle_patch, {"mode": "patch", "patch": patch})
        self.assertTrue(result["restored"])
        self.assertEqual(first.read_text(), "old first\n")
        self.assertEqual(second.read_text(), "old second\n")

    async def test_existing_add_destination_is_not_overwritten(self):
        path = self.workspace / "exists.txt"
        path.write_text("keep")
        patch = """*** Begin Patch
*** Add File: exists.txt
+replace
*** End Patch"""
        result = await self.call(handle_patch, {"mode": "patch", "patch": patch})
        self.assertIn("already exists", result["error"])
        self.assertEqual(path.read_text(), "keep")

    async def test_v4a_parent_traversal_is_rejected(self):
        patch = """*** Begin Patch
*** Add File: ../escaped.txt
+no
*** End Patch"""
        result = await self.call(handle_patch, {"mode": "patch", "patch": patch})
        self.assertIn("traversal", result["error"])


class SearchTests(FileToolCase):
    def setUp(self):
        super().setUp()
        (self.workspace / "a.py").write_text("zero\nneedle one\nafter\n")
        (self.workspace / "b.txt").write_text("needle two\nneedle three\n")
        hidden = self.workspace / ".git"
        hidden.mkdir()
        (hidden / "secret.txt").write_text("needle hidden")
        (self.workspace / ".env").write_text("needle secret")

    async def test_content_search_has_lines_context_and_pagination(self):
        first = await self.call(
            handle_search_files, {"pattern": "needle", "limit": 1, "context": 1}
        )
        self.assertGreaterEqual(first["total_count"], 3)
        self.assertEqual(first["matches"][0]["line"], 1)
        self.assertTrue(first["truncated"])

    async def test_content_search_supports_filters_and_modes(self):
        files = await self.call(
            handle_search_files,
            {"pattern": "needle", "file_glob": "*.txt", "output_mode": "files_only"},
        )
        self.assertEqual([Path(path).name for path in files["files"]], ["b.txt"])
        counts = await self.call(
            handle_search_files,
            {"pattern": "needle", "file_glob": "*.txt", "output_mode": "count"},
        )
        self.assertEqual(sum(counts["counts"].values()), 2)

    async def test_file_search_uses_globs(self):
        result = await self.call(
            handle_search_files, {"pattern": "*.py", "target": "files"}
        )
        self.assertEqual([Path(path).name for path in result["files"]], ["a.py"])

    async def test_hidden_and_secret_files_are_not_results(self):
        result = await self.call(handle_search_files, {"pattern": "secret"})
        self.assertEqual(result.get("matches", []), [])

    async def test_invalid_regex_is_an_error(self):
        result = await self.call(handle_search_files, {"pattern": "["})
        self.assertIn("Search failed", result["error"])


if __name__ == "__main__":
    unittest.main()
