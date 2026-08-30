"""Contract for the Hermes-derived local skills slice."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from pilotage.settings import Settings
from pilotage.tools import ToolContext
from pilotage.tools.file_safety import (
    get_write_approval_category,
    get_write_denied_error,
)
from pilotage.tools.skill_utils import parse_frontmatter
from pilotage.tools.skills import (
    build_skills_prompt,
    discover_skills,
    handle_skill_view,
    handle_skills_list,
    reset_skill_view_dedup,
    skills_directory,
)


class _Config:
    def __init__(self, state_dir: Path, *, channel: str = "whatsapp", enabled=None):
        self.state_dir = state_dir
        data = {"tools": {"enabled": enabled or ["skills"]}}
        self.settings = Settings(data).for_channel(channel) if channel else Settings(data)
        self.instructions = "Base instructions."


def _make_skill(
    root: Path,
    name: str,
    *,
    category: str = "",
    directory: str = "",
    description: str = "A useful workflow.",
    version: str = "1.0.0",
    channels: str = "[whatsapp, telegram]",
    extra: str = "",
    body: str = "Follow this exact workflow.",
) -> Path:
    skill_dir = root / category / (directory or name)
    skill_dir.mkdir(parents=True, exist_ok=True)
    version_line = f"version: {version}\n" if version else ""
    channels_line = f"channels: {channels}\n" if channels else ""
    content = (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"{version_line}"
        f"{channels_line}"
        f"{extra}"
        "---\n\n"
        f"# {name}\n\n{body}\n"
    )
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


class SkillCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name).resolve()
        self.config = _Config(self.home)
        self.root = skills_directory(self.config)
        self.context = ToolContext("chat", self.config)
        env = mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)

    async def call(self, handler, args):
        return json.loads(await handler(args, self.context))


class FrontmatterTests(unittest.TestCase):
    def test_nested_yaml_and_utf8_bom_match_hermes(self):
        content = (
            "\ufeff---\nname: demo\ndescription: Demo.\nversion: 1\n"
            "metadata:\n  hermes:\n    tags: [a, b]\n---\n\nBody.\n"
        )
        frontmatter, body = parse_frontmatter(content)
        self.assertEqual(frontmatter["name"], "demo")
        self.assertEqual(frontmatter["metadata"]["hermes"]["tags"], ["a", "b"])
        self.assertEqual(body.strip(), "Body.")


class DiscoveryTests(SkillCase):
    async def test_discovery_prunes_support_and_dependency_trees(self):
        _make_skill(self.root, "real", category="data")
        archived = self.root / "data" / "real" / "references" / "archived"
        archived.mkdir(parents=True)
        (archived / "SKILL.md").write_text(
            "---\nname: archived\ndescription: no\nversion: 1\n---\n",
            encoding="utf-8",
        )
        dependency = self.root / "node_modules" / "fake"
        dependency.mkdir(parents=True)
        (dependency / "SKILL.md").write_text(
            "---\nname: fake\ndescription: no\nversion: 1\n---\n",
            encoding="utf-8",
        )
        self.assertEqual([skill["name"] for skill in discover_skills(self.config)], ["real"])

    async def test_platform_and_channel_filters_apply_before_the_prompt(self):
        _make_skill(self.root, "wrong-platform", extra="platforms: [never-os]\n")
        _make_skill(self.root, "telegram-only", channels="[telegram]")
        _make_skill(self.root, "whatsapp", channels="[whatsapp]")
        self.assertEqual([skill["name"] for skill in discover_skills(self.config)], ["whatsapp"])

    async def test_missing_or_unknown_channels_fail_loudly(self):
        _make_skill(self.root, "missing-channels", channels="")
        _make_skill(self.root, "unknown-channel", channels="[email]")
        skills = discover_skills(self.config)
        self.assertEqual(len(skills), 2)
        self.assertTrue(all(not skill["available"] for skill in skills))
        self.assertTrue(all("channels" in skill["error"] for skill in skills))

    async def test_missing_required_credential_is_unavailable_until_present(self):
        _make_skill(
            self.root,
            "database",
            extra="required_credential_files:\n  - database-token.json\n",
        )
        skill = discover_skills(self.config)[0]
        self.assertFalse(skill["available"])
        self.assertIn("database-token.json", skill["error"])
        self.assertNotIn("database", build_skills_prompt(self.config))

        (self.home / "database-token.json").write_text("{}", encoding="utf-8")
        self.assertTrue(discover_skills(self.config)[0]["available"])
        self.assertIn("database", build_skills_prompt(self.config))

    async def test_master_credentials_cannot_satisfy_a_skill_requirement(self):
        (self.home / "codex-auth.json").write_text("secret", encoding="utf-8")
        _make_skill(
            self.root,
            "steal-auth",
            extra="required_credential_files: [codex-auth.json]\n",
        )
        skill = discover_skills(self.config)[0]
        self.assertFalse(skill["available"])
        self.assertIn("protected credential", skill["error"])

    async def test_required_frontmatter_fields_fail_loudly(self):
        _make_skill(self.root, "unversioned", version="")
        skill = discover_skills(self.config)[0]
        self.assertFalse(skill["available"])
        self.assertIn("version is required", skill["error"])

    async def test_colliding_names_are_not_silently_shadowed(self):
        _make_skill(self.root, "same", category="one")
        _make_skill(self.root, "same", category="two")
        skills = discover_skills(self.config)
        self.assertEqual(len(skills), 2)
        self.assertTrue(all(not skill["available"] for skill in skills))
        self.assertTrue(all("Ambiguous" in skill["error"] for skill in skills))

    async def test_disabled_skill_is_absent(self):
        _make_skill(self.root, "hidden")
        self.config.settings = Settings({"skills": {"disabled": ["hidden"]}}).for_channel(
            "whatsapp"
        )
        self.assertEqual(discover_skills(self.config), [])


class PromptAndListTests(SkillCase):
    async def test_prompt_contains_only_short_metadata_not_skill_content(self):
        description = "d" * 100
        _make_skill(
            self.root,
            "reporting",
            category="data",
            description=description,
            body="PRIVATE WORKFLOW BODY",
        )
        prompt = build_skills_prompt(self.config)
        self.assertIn("reporting", prompt)
        self.assertIn("d" * 57 + "...", prompt)
        self.assertNotIn("PRIVATE WORKFLOW BODY", prompt)
        self.assertLess(prompt.count("d"), 70)

    async def test_skills_list_creates_the_profile_directory_and_filters_category(self):
        empty = await self.call(handle_skills_list, {})
        self.assertTrue(empty["success"])
        self.assertTrue(self.root.is_dir())
        _make_skill(self.root, "chart", category="data")
        _make_skill(self.root, "deploy", category="ops")
        result = await self.call(handle_skills_list, {"category": "data"})
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["skills"][0]["name"], "chart")

    async def test_agent_injects_index_only_when_skills_group_is_enabled(self):
        try:
            from pilotage.agent import Agent
            from pilotage.history import ConversationStore
        except ModuleNotFoundError as exc:
            if exc.name in {"openai", "httpx"}:
                self.skipTest(f"runtime dependency unavailable: {exc.name}")
            raise
        _make_skill(self.root, "reporting")
        enabled = Agent(
            self.config,
            ConversationStore(path=None),
            allow_persistence_writes=True,
        )
        enabled_instructions = enabled._instructions_for_session("chat")
        self.assertIn("<available_skills>", enabled_instructions)
        self.assertNotIn("## Persistent learning", enabled_instructions)

        editable_config = _Config(self.home, enabled=["skills", "file"])
        editable = Agent(
            editable_config,
            ConversationStore(path=None),
            allow_persistence_writes=True,
        )
        self.assertIn(
            "## Persistent learning", editable._instructions_for_session("chat")
        )
        self.assertTrue(editable._persistence_writes_enabled)

        file_only_config = _Config(self.home, enabled=["file"])
        file_only = Agent(
            file_only_config,
            ConversationStore(path=None),
            allow_persistence_writes=True,
        )
        file_only_instructions = file_only._instructions_for_session("chat")
        self.assertNotIn("## Persistent learning", file_only_instructions)
        self.assertFalse(file_only._persistence_writes_enabled)
        self.assertNotIn("<available_skills>", file_only_instructions)

        disabled_config = _Config(self.home, enabled=["todo"])
        disabled = Agent(disabled_config, ConversationStore(path=None))
        self.assertNotIn(
            "<available_skills>", disabled._instructions_for_session("chat")
        )

    async def test_new_session_refreshes_skill_index_without_mutating_old_prefix(self):
        try:
            from pilotage.agent import Agent
            from pilotage.history import ConversationStore
        except ModuleNotFoundError as exc:
            if exc.name in {"openai", "httpx"}:
                self.skipTest(f"runtime dependency unavailable: {exc.name}")
            raise

        _make_skill(self.root, "existing")
        agent = Agent(self.config, ConversationStore(path=None))
        first = agent._instructions_for_session("first")
        self.assertIn("existing", first)
        self.assertNotIn("new-skill", first)

        _make_skill(self.root, "new-skill")

        self.assertNotIn("new-skill", agent._instructions_for_session("first"))
        self.assertIn("new-skill", agent._instructions_for_session("second"))

    async def test_scheduled_allowlist_hides_and_blocks_other_skills(self):
        _make_skill(self.root, "allowed")
        _make_skill(self.root, "blocked")
        self.context.allowed_skills = frozenset({"allowed"})

        prompt = build_skills_prompt(
            self.config,
            self.context.allowed_skills,
        )
        self.assertIn("allowed", prompt)
        self.assertNotIn("blocked", prompt)

        listed = await self.call(handle_skills_list, {})
        self.assertEqual(
            [skill["name"] for skill in listed["skills"]],
            ["allowed"],
        )
        denied = await self.call(
            handle_skill_view,
            {"name": "blocked"},
        )
        self.assertFalse(denied["success"])
        self.assertIn("not allowed", denied["error"])


class ViewTests(SkillCase):
    async def test_view_resolves_directory_and_frontmatter_names(self):
        _make_skill(self.root, "public-name", directory="disk-name")
        by_directory = await self.call(handle_skill_view, {"name": "disk-name"})
        self.context.state.clear()
        by_name = await self.call(handle_skill_view, {"name": "public-name"})
        self.assertTrue(by_directory["success"])
        self.assertTrue(by_name["success"])
        self.assertIn("Follow this exact workflow", by_name["content"])

    async def test_main_view_advertises_and_loads_linked_files(self):
        skill = _make_skill(self.root, "reporting")
        reference = skill / "references" / "format.md"
        reference.parent.mkdir()
        reference.write_text("Exact report format.", encoding="utf-8")
        main = await self.call(handle_skill_view, {"name": "reporting"})
        self.assertEqual(main["linked_files"]["references"], ["references/format.md"])

        linked = await self.call(
            handle_skill_view,
            {"name": "reporting", "file_path": "references/format.md"},
        )
        self.assertEqual(linked["content"], "Exact report format.")

    async def test_name_and_linked_file_traversal_are_rejected(self):
        _make_skill(self.root, "safe")
        escaped_name = await self.call(handle_skill_view, {"name": "../safe"})
        escaped_file = await self.call(
            handle_skill_view,
            {"name": "safe", "file_path": "references/../../../secret"},
        )
        windows_path = await self.call(handle_skill_view, {"name": "C:\\secret"})
        self.assertIn("traversal", escaped_name["error"])
        self.assertIn("traversal", escaped_file["error"])
        self.assertIn("relative", windows_path["error"])

    async def test_symlink_escape_is_rejected(self):
        skill = _make_skill(self.root, "safe")
        outside = self.home / "outside.txt"
        outside.write_text("do not leak", encoding="utf-8")
        link = skill / "references" / "outside.txt"
        link.parent.mkdir()
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        result = await self.call(
            handle_skill_view,
            {"name": "safe", "file_path": "references/outside.txt"},
        )
        self.assertIn("escapes", result["error"])
        self.assertNotIn("do not leak", json.dumps(result))

    async def test_missing_credential_blocks_content_load(self):
        _make_skill(
            self.root,
            "database",
            extra="required_credential_files: [database-token.json]\n",
            body="Never reveal this until ready.",
        )
        result = await self.call(handle_skill_view, {"name": "database"})
        self.assertIn("missing required credential", result["error"])
        self.assertNotIn("Never reveal", json.dumps(result))

    async def test_binary_linked_file_returns_metadata_not_bytes(self):
        skill = _make_skill(self.root, "assets")
        asset = skill / "assets" / "image.bin"
        asset.parent.mkdir()
        asset.write_bytes(b"\x00\x01private")
        result = await self.call(
            handle_skill_view,
            {"name": "assets", "file_path": "assets/image.bin"},
        )
        self.assertTrue(result["is_binary"])
        self.assertNotIn("private", json.dumps(result))

    async def test_secret_bearing_linked_file_is_neither_listed_nor_read(self):
        skill = _make_skill(self.root, "secrets")
        secret = skill / "references" / ".env"
        secret.parent.mkdir()
        secret.write_text("TOKEN=do-not-leak", encoding="utf-8")
        main = await self.call(handle_skill_view, {"name": "secrets"})
        self.assertIsNone(main["linked_files"])
        result = await self.call(
            handle_skill_view,
            {"name": "secrets", "file_path": "references/.env"},
        )
        self.assertIn("Access denied", result["error"])
        self.assertNotIn("do-not-leak", json.dumps(result))

    async def test_unchanged_repeat_is_deduplicated_but_edit_reloads(self):
        skill = _make_skill(self.root, "repeat")
        first = await self.call(handle_skill_view, {"name": "repeat"})
        second = await self.call(handle_skill_view, {"name": "repeat"})
        self.assertIn("content", first)
        self.assertTrue(second["dedup"])
        self.assertNotIn("content", second)

        time.sleep(0.01)
        path = skill / "SKILL.md"
        path.write_text(path.read_text(encoding="utf-8") + "\nNew step.\n", encoding="utf-8")
        third = await self.call(handle_skill_view, {"name": "repeat"})
        self.assertIn("New step", third["content"])

    async def test_compaction_reset_makes_the_next_view_full_again(self):
        _make_skill(self.root, "repeat")
        await self.call(handle_skill_view, {"name": "repeat"})
        self.assertTrue(
            (await self.call(handle_skill_view, {"name": "repeat"}))["dedup"]
        )
        reset_skill_view_dedup(self.context)
        reloaded = await self.call(handle_skill_view, {"name": "repeat"})
        self.assertIn("Follow this exact workflow", reloaded["content"])


class WriteSafetyTests(unittest.TestCase):
    def test_skill_writes_are_classified_for_approval_not_hard_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(home)}):
                path = str(home / "skills" / "demo" / "SKILL.md")
                error = get_write_denied_error(path)
                category = get_write_approval_category(path)
        self.assertIsNone(error)
        self.assertEqual(category, "skills")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
