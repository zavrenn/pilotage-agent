"""Contract for inert skill template preprocessing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pilotage.settings import Settings
from pilotage.tools import ToolContext
from pilotage.tools.skill_preprocessing import (
    preprocess_skill_content,
    substitute_template_vars,
)
from pilotage.tools.skills import handle_skill_view, skill_view


class _Config:
    def __init__(self, root: Path, skills: dict | None = None):
        self.state_dir = root
        self.settings = Settings(
            {
                "tools": {"enabled": ["skills"]},
                "skills": skills or {},
            }
        )


def _skill(root: Path, body: str) -> Path:
    directory = root / "skills" / "example"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\n"
        "name: example\n"
        "description: Example workflow.\n"
        "version: 1.0.0\n"
        "channels: [whatsapp, telegram]\n"
        "---\n\n"
        + body,
        encoding="utf-8",
    )
    return directory


class TemplateVariableTests(unittest.TestCase):
    def test_hermes_skill_placeholders_remain_compatible(self):
        root = Path("/profile/skills/demo")
        content = "${HERMES_SKILL_DIR}/run --session ${HERMES_SESSION_ID}"

        rendered = substitute_template_vars(content, root, "chat-42")

        self.assertEqual(rendered, f"{root}/run --session chat-42")

    def test_unresolved_values_stay_visible_for_debugging(self):
        content = "${HERMES_SKILL_DIR} ${HERMES_SESSION_ID}"
        self.assertEqual(
            substitute_template_vars(content, None, None),
            content,
        )

    def test_template_expansion_can_be_disabled(self):
        rendered = preprocess_skill_content(
            "${HERMES_SESSION_ID}",
            Path("/skill"),
            session_id="chat",
            skills_cfg={"template_vars": False},
        )
        self.assertEqual(rendered, "${HERMES_SESSION_ID}")


class ExecutablePreprocessingTests(unittest.TestCase):
    def test_inline_shell_syntax_is_always_inert(self):
        content = "Today is !`date +%F`"
        for config in ({}, {"inline_shell": True, "inline_shell_timeout": 1}):
            with self.subTest(config=config):
                self.assertEqual(
                    preprocess_skill_content(
                        content,
                        Path("/skill"),
                        skills_cfg=config,
                    ),
                    content,
                )


class SkillViewPreprocessingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.skill_dir = _skill(
            self.root,
            "Run ${HERMES_SKILL_DIR}/scripts/task.sh "
            "for ${HERMES_SESSION_ID}.",
        )

    async def test_main_skill_view_receives_profile_path_and_chat_session(self):
        config = _Config(self.root)
        context = ToolContext("whatsapp-chat", config)

        result = json.loads(await handle_skill_view({"name": "example"}, context))

        self.assertIn(str(self.skill_dir), result["content"])
        self.assertIn("whatsapp-chat", result["content"])
        self.assertNotIn("${HERMES_", result["content"])

    async def test_direct_view_without_session_leaves_that_token_visible(self):
        config = _Config(self.root)
        result = json.loads(skill_view(config, "example"))

        self.assertIn(str(self.skill_dir), result["content"])
        self.assertIn("${HERMES_SESSION_ID}", result["content"])

    async def test_linked_files_are_read_verbatim_not_executed(self):
        linked = self.skill_dir / "references" / "literal.md"
        linked.parent.mkdir()
        linked.write_text(
            "${HERMES_SESSION_ID} !`dangerous command`",
            encoding="utf-8",
        )
        config = _Config(self.root, {"inline_shell": True})
        context = ToolContext("chat", config)

        result = json.loads(
            await handle_skill_view(
                {"name": "example", "file_path": "references/literal.md"},
                context,
            )
        )

        self.assertEqual(
            result["content"],
            "${HERMES_SESSION_ID} !`dangerous command`",
        )

    async def test_main_skill_never_executes_inline_shell_syntax(self):
        (self.skill_dir / "SKILL.md").write_text(
            "---\nname: example\ndescription: Example.\nversion: 1\n"
            "channels: [whatsapp, telegram]\n---\n"
            "value=!`printf ready`",
            encoding="utf-8",
        )
        config = _Config(
            self.root,
            {"inline_shell": True, "inline_shell_timeout": 3},
        )
        context = ToolContext("chat", config)
        result = json.loads(await handle_skill_view({"name": "example"}, context))

        self.assertIn("value=!`printf ready`", result["content"])


if __name__ == "__main__":
    unittest.main()
