"""Hermes' lightweight skill metadata and discovery mechanisms.

This is the production slice from ``tmp/hermes-agent/agent/skill_utils.py``:
frontmatter parsing, platform filtering, support-directory exclusion and
ordered SKILL.md discovery. Org mirrors, project trust, external directories,
marketplaces and runtime-environment detection are intentionally outside the
Pilotage contract.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}

EXCLUDED_SKILL_DIRS = frozenset(
    (
        ".git",
        ".github",
        ".hub",
        ".archive",
        ".venv",
        "venv",
        "node_modules",
        "site-packages",
        "__pycache__",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    )
)

SKILL_SUPPORT_DIRS = frozenset(("references", "templates", "assets", "scripts"))


def is_skill_support_path(path, *, root: Optional[Path] = None) -> bool:
    """True if *path* is under a support dir of an actual skill root."""
    path_obj = path if isinstance(path, Path) else Path(str(path))
    parts = path_obj.parts
    for idx, part in enumerate(parts[:-1]):
        if part not in SKILL_SUPPORT_DIRS or idx == 0:
            continue
        skill_root = Path(*parts[:idx])
        if root is not None and not path_obj.is_absolute():
            skill_root = root / skill_root
        if (skill_root / "SKILL.md").exists():
            return True
    return False


_yaml_load_fn = None


def yaml_load(content: str):
    """Parse YAML with lazy import and CSafeLoader preference."""
    global _yaml_load_fn
    if _yaml_load_fn is None:
        import yaml

        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader

        def _load(value: str):
            return yaml.load(value, Loader=loader)

        _yaml_load_fn = _load
    return _yaml_load_fn(content)


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown, including Windows BOM files."""
    frontmatter: Dict[str, Any] = {}
    if content.startswith("\ufeff"):
        content = content[1:]
    body = content
    if not content.startswith("---"):
        return frontmatter, body

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return frontmatter, body

    yaml_content = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]
    try:
        parsed = yaml_load(yaml_content)
        if isinstance(parsed, dict):
            frontmatter = parsed
    except Exception:
        for line in yaml_content.strip().split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()
    return frontmatter, body


def skill_matches_platform_list(platforms: Any) -> bool:
    """Return True when *platforms* is compatible with the current OS."""
    if not platforms:
        return True
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = sys.platform
    for platform in platforms:
        normalized = str(platform).lower().strip()
        mapped = PLATFORM_MAP.get(normalized, normalized)
        if current.startswith(mapped):
            return True
    return False


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """Return True when the skill is compatible with the current OS."""
    return skill_matches_platform_list(frontmatter.get("platforms"))


def skill_matches_channel(frontmatter: Dict[str, Any], channel: str) -> bool:
    """Pilotage adaptation: enforce a skill's declared messaging channels."""
    channels = frontmatter.get("channels")
    if not channels or not channel:
        return True
    if not isinstance(channels, list):
        channels = [channels]
    wanted = channel.lower().strip()
    return any(str(item).lower().strip() == wanted for item in channels)


SKILL_PROMPT_DESC_LIMIT = 60


def extract_skill_description(frontmatter: Dict[str, Any]) -> str:
    """Extract Hermes' system-prompt-length description."""
    raw_desc = frontmatter.get("description", "")
    desc = str(raw_desc).strip().strip("'\"") if raw_desc else ""
    if not desc:
        return ""
    if len(desc) > SKILL_PROMPT_DESC_LIMIT:
        return desc[: SKILL_PROMPT_DESC_LIMIT - 3] + "..."
    return desc


def iter_skill_index_files(skills_dir: Path, filename: str = "SKILL.md"):
    """Walk *skills_dir* yielding sorted active skill files.

    This preserves Hermes' pruning rules. Pilotage does not follow directory
    symlinks because its one profile directory is the complete trusted root;
    linked external skill trees are not a product requirement.
    """
    matches: list[str] = []
    for root, dirs, files in os.walk(str(skills_dir), followlinks=False):
        has_skill_md = "SKILL.md" in files
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in EXCLUDED_SKILL_DIRS
            and not (has_skill_md and directory in SKILL_SUPPORT_DIRS)
        ]
        if filename in files:
            matches.append(os.path.join(root, filename))
    for path in sorted(matches):
        yield Path(path)


__all__ = [
    "EXCLUDED_SKILL_DIRS",
    "SKILL_SUPPORT_DIRS",
    "extract_skill_description",
    "is_skill_support_path",
    "iter_skill_index_files",
    "parse_frontmatter",
    "skill_matches_channel",
    "skill_matches_platform",
    "skill_matches_platform_list",
]
