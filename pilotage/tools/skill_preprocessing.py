"""Inert SKILL.md template substitution.

Skill content is data shown to the model. Loading it must never execute code;
all executable work stays behind the normal terminal/code tool boundaries.
"""

import re
from pathlib import Path

# Matches ${HERMES_SKILL_DIR} / ${HERMES_SESSION_ID} tokens in SKILL.md.
# Tokens that don't resolve (e.g. ${HERMES_SESSION_ID} with no session) are
# left as-is so the user can debug them.
_SKILL_TEMPLATE_RE = re.compile(r"\$\{(HERMES_SKILL_DIR|HERMES_SESSION_ID)\}")


def load_skills_config() -> dict:
    """Return defaults when preprocessing is called outside a live Config."""
    return {}


def substitute_template_vars(
    content: str,
    skill_dir: Path | None,
    session_id: str | None,
) -> str:
    """Replace ${HERMES_SKILL_DIR} / ${HERMES_SESSION_ID} in skill content.

    Only substitutes tokens for which a concrete value is available --
    unresolved tokens are left in place so the author can spot them.
    """
    if not content:
        return content

    skill_dir_str = str(skill_dir) if skill_dir else None

    def _replace(match: re.Match) -> str:
        token = match.group(1)
        if token == "HERMES_SKILL_DIR" and skill_dir_str:
            return skill_dir_str
        if token == "HERMES_SESSION_ID" and session_id:
            return str(session_id)
        return match.group(0)

    return _SKILL_TEMPLATE_RE.sub(_replace, content)


def preprocess_skill_content(
    content: str,
    skill_dir: Path | None,
    session_id: str | None = None,
    skills_cfg: dict | None = None,
) -> str:
    """Apply inert SKILL.md template substitution only.

    ``skills_cfg`` is accepted while older configuration files are migrated,
    but executable preprocessing settings are intentionally ignored.
    """
    if not content:
        return content

    cfg = skills_cfg if isinstance(skills_cfg, dict) else load_skills_config()
    if cfg.get("template_vars", True):
        content = substitute_template_vars(content, skill_dir, session_id)
    return content

