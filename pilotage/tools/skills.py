"""Local skills with Hermes' progressive-disclosure behavior.

The discovery, frontmatter, sorting, linked-file and path-containment behavior
comes from ``tmp/hermes-agent/tools/skills_tool.py`` and
``tmp/hermes-agent/agent/prompt_builder.py``. Pilotage keeps one trusted,
profile-local skills directory. Plugin skills, external trees, marketplaces,
syncing and mutation are deliberately not carried over. Hermes' small,
optional SKILL.md preprocessor is retained for compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

from .file_safety import get_read_block_error
from .path_security import has_traversal_component, validate_within_dir
from .registry import Tool, ToolContext, tool_error
from .skill_preprocessing import preprocess_skill_content
from .skill_utils import (
    extract_skill_description,
    iter_skill_index_files,
    parse_frontmatter,
    skill_matches_channel,
    skill_matches_platform,
)

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
_LINKED_DIRS = ("references", "templates", "assets", "scripts")
_DEDUP_LOCK = threading.Lock()
_SKILL_VIEW_DEDUP_CAP = 200
_SKILL_VIEW_DEDUP_MESSAGE = (
    "Skill content unchanged since it was loaded earlier in this conversation — "
    "refer to the earlier skill_view result; it is still current and complete."
)


def _json(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False)


def skills_directory(config: Any) -> Path:
    """The current profile's only skill root."""
    return Path(config.state_dir) / "skills"


def _channel(config: Any) -> str:
    return str(getattr(getattr(config, "settings", None), "channel", "") or "")


def _parse_tags(tags_value) -> List[str]:
    """Hermes' accepted list, bracketed-string and comma-string forms."""
    if not tags_value:
        return []
    if isinstance(tags_value, list):
        return [str(tag).strip() for tag in tags_value if tag]
    tags_value = str(tags_value).strip()
    if tags_value.startswith("[") and tags_value.endswith("]"):
        tags_value = tags_value[1:-1]
    return [tag.strip().strip("\"'") for tag in tags_value.split(",") if tag.strip()]


def _category(skill_md: Path, root: Path) -> Optional[str]:
    try:
        parts = skill_md.relative_to(root).parts
    except ValueError:
        return None
    return parts[0] if len(parts) >= 3 else None


def _identifier(skill_md: Path, root: Path) -> str:
    return skill_md.parent.relative_to(root).as_posix()


def _required_credential_files(frontmatter: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Normalize Hermes' string-or-{path: ...} credential declarations."""
    raw = frontmatter.get("required_credential_files", [])
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], ["required_credential_files must be a list"]
    paths: List[str] = []
    errors: List[str] = []
    for entry in raw:
        if isinstance(entry, str):
            relative = entry.strip()
        elif isinstance(entry, dict):
            relative = str(entry.get("path") or entry.get("name") or "").strip()
        else:
            errors.append(f"invalid credential declaration: {entry!r}")
            continue
        if not relative:
            errors.append("empty credential declaration")
            continue
        paths.append(relative)
    return paths, errors


def _credential_status(
    frontmatter: Dict[str, Any], state_root: Path
) -> Tuple[List[str], List[str], List[str]]:
    required, errors = _required_credential_files(frontmatter)
    missing: List[str] = []
    invalid: List[str] = list(errors)
    for relative in required:
        if _relative_path_error(relative):
            invalid.append(relative)
            continue
        candidate = state_root / relative
        if validate_within_dir(candidate, state_root):
            invalid.append(relative)
            continue
        try:
            blocked = get_read_block_error(str(candidate.resolve()))
        except Exception:
            blocked = "credential guard failed"
        if blocked:
            invalid.append(relative)
            continue
        if not candidate.is_file():
            missing.append(relative)
    return required, missing, invalid


def _metadata(skill_md: Path, root: Path, config: Any) -> Optional[Dict[str, Any]]:
    try:
        content = skill_md.read_text(encoding="utf-8-sig", errors="replace")
        frontmatter, _ = parse_frontmatter(content)
    except Exception as exc:
        logger.error("Could not read skill %s: %s", skill_md, exc)
        return {
            "name": skill_md.parent.name,
            "description": "",
            "category": _category(skill_md, root),
            "identifier": _identifier(skill_md, root),
            "available": False,
            "error": f"Could not read SKILL.md: {exc}",
            "_skill_md": skill_md,
            "_skill_dir": skill_md.parent,
            "_frontmatter": {},
        }

    if not skill_matches_platform(frontmatter):
        return None
    if not skill_matches_channel(frontmatter, _channel(config)):
        return None

    raw_name = frontmatter.get("name")
    name = str(raw_name).strip() if raw_name is not None else skill_md.parent.name
    raw_description = frontmatter.get("description")
    description = str(raw_description).strip() if raw_description is not None else ""
    version = str(frontmatter.get("version") or "").strip()
    problems: List[str] = []
    if not raw_name or not name:
        problems.append("frontmatter name is required")
    elif len(name) > MAX_NAME_LENGTH:
        problems.append(f"frontmatter name exceeds {MAX_NAME_LENGTH} characters")
    if not description:
        problems.append("frontmatter description is required")
    if not version:
        problems.append("frontmatter version is required")
    if len(description) > MAX_DESCRIPTION_LENGTH:
        description = description[: MAX_DESCRIPTION_LENGTH - 3] + "..."

    required, missing, invalid = _credential_status(frontmatter, Path(config.state_dir))
    if invalid:
        problems.append("invalid or protected credential declaration(s): " + ", ".join(invalid))
    if missing:
        problems.append("missing required credential file(s): " + ", ".join(missing))

    disabled = set(config.settings.names("skills.disabled"))
    if name in disabled or _identifier(skill_md, root) in disabled:
        return None

    entry: Dict[str, Any] = {
        "name": name,
        "description": description,
        "category": _category(skill_md, root),
        "identifier": _identifier(skill_md, root),
        "version": version,
        "available": not problems,
        "required_credential_files": required,
        "missing_credential_files": missing,
        "_skill_md": skill_md,
        "_skill_dir": skill_md.parent,
        "_frontmatter": frontmatter,
    }
    if problems:
        entry["error"] = "; ".join(problems)
        logger.warning("Skill %s is unavailable: %s", entry["identifier"], entry["error"])
    return entry


def discover_skills(config: Any) -> List[Dict[str, Any]]:
    """Return the profile's channel-compatible skill metadata."""
    root = skills_directory(config)
    if not root.exists():
        return []
    skills = [
        entry
        for skill_md in iter_skill_index_files(root)
        if (entry := _metadata(skill_md, root, config)) is not None
    ]

    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for entry in skills:
        by_name.setdefault(entry["name"], []).append(entry)
    for name, matches in by_name.items():
        if len(matches) < 2:
            continue
        paths = ", ".join(match["identifier"] for match in matches)
        for match in matches:
            match["available"] = False
            match["error"] = f"Ambiguous skill name '{name}': {paths}"
        logger.warning("Skill name collision for %s: %s", name, paths)

    return sorted(
        skills,
        key=lambda entry: (entry.get("category") or "", entry["name"], entry["identifier"]),
    )


def _public_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    public: Dict[str, Any] = {
        "name": entry["name"],
        "description": entry["description"],
        "category": entry.get("category"),
        "available": entry["available"],
    }
    if not entry["available"]:
        public["error"] = entry.get("error", "Skill is unavailable")
        if entry.get("missing_credential_files"):
            public["missing_credential_files"] = entry["missing_credential_files"]
    return public


def skills_list(config: Any, category: Optional[str] = None) -> str:
    """List minimal skill metadata, Hermes progressive-disclosure tier one."""
    root = skills_directory(config)
    root.mkdir(parents=True, exist_ok=True)
    skills = discover_skills(config)
    if category:
        skills = [skill for skill in skills if skill.get("category") == category]
    public = [_public_metadata(skill) for skill in skills]
    categories = sorted({skill["category"] for skill in public if skill.get("category")})
    return _json(
        {
            "success": True,
            "skills": public,
            "categories": categories,
            "count": len(public),
            "hint": "Use skill_view(name) to load full content and linked files",
        }
    )


def build_skills_prompt(config: Any) -> str:
    """Build Hermes' compact name/description index for the system prompt."""
    skills = [entry for entry in discover_skills(config) if entry["available"]]
    if not skills:
        return ""

    by_category: Dict[str, List[Tuple[str, str]]] = {}
    for entry in skills:
        category = entry.get("category") or "general"
        by_category.setdefault(category, []).append(
            (entry["name"], extract_skill_description(entry["_frontmatter"]))
        )

    index_lines: List[str] = []
    for category in sorted(by_category):
        index_lines.append(f"  {category}:")
        for name, description in sorted(by_category[category], key=lambda item: item[0]):
            index_lines.append(f"    - {name}: {description}" if description else f"    - {name}")

    return (
        "## Skills (mandatory)\n"
        "Before replying, scan the skills below. If a skill matches or is even partially "
        "relevant to the task, you MUST load it with skill_view(name) and follow its "
        "instructions. Skills contain established workflows and quality standards; load "
        "them even when you already know a general approach.\n\n"
        "<available_skills>\n"
        + "\n".join(index_lines)
        + "\n</available_skills>\n\n"
        "Only proceed without loading a skill if genuinely none are relevant to the task."
    )


def _relative_path_error(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return "Path must be a string."
    candidate = value.strip()
    if not candidate:
        return "Path must not be empty."
    if (
        PurePosixPath(candidate).is_absolute()
        or PureWindowsPath(candidate).is_absolute()
        or PureWindowsPath(candidate).drive
    ):
        return "Path must be relative to the skills directory."
    if has_traversal_component(candidate):
        return "Path cannot contain '..' traversal components."
    return None


def _resolve_skill(config: Any, name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    error = _relative_path_error(name)
    if error:
        return None, error
    root = skills_directory(config)
    if not root.exists():
        return None, "Skills directory does not exist."

    candidates: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    direct = root / name / "SKILL.md"
    if direct.is_file() and validate_within_dir(direct, root) is None:
        metadata = _metadata(direct, root, config)
        if metadata is not None:
            candidates.append(metadata)
            seen.add(direct.resolve())

    if "/" not in name and "\\" not in name:
        for skill_md in iter_skill_index_files(root):
            try:
                resolved = skill_md.resolve()
            except OSError:
                continue
            if resolved in seen or validate_within_dir(skill_md, root):
                continue
            metadata = _metadata(skill_md, root, config)
            if metadata is None:
                continue
            if skill_md.parent.name == name or metadata["name"] == name:
                candidates.append(metadata)
                seen.add(resolved)

    if len(candidates) > 1:
        paths = [candidate["identifier"] for candidate in candidates]
        return None, (
            f"Ambiguous skill name '{name}': {', '.join(paths)}. "
            "Use the categorized relative path."
        )
    if not candidates:
        available = [skill["name"] for skill in discover_skills(config) if skill["available"]]
        suffix = f" Available skills: {', '.join(available[:20])}." if available else ""
        return None, f"Skill '{name}' not found.{suffix}"
    candidate = candidates[0]
    if not candidate["available"]:
        return None, candidate.get("error", f"Skill '{name}' is unavailable.")
    return candidate, None


def _linked_files(skill_dir: Path) -> Optional[Dict[str, List[str]]]:
    linked: Dict[str, List[str]] = {}
    for category in _LINKED_DIRS:
        base = skill_dir / category
        if not base.is_dir():
            continue
        files = [
            str(path.relative_to(skill_dir)).replace(os.sep, "/")
            for path in sorted(base.rglob("*"))
            if path.is_file()
            and validate_within_dir(path, skill_dir) is None
            and not _read_blocked(path)
        ]
        if files:
            linked[category] = files
    return linked or None


def _available_linked_files(skill_dir: Path) -> Dict[str, List[str]]:
    return _linked_files(skill_dir) or {}


def _read_skill_file(path: Path) -> Tuple[Optional[str], bool]:
    raw = path.read_bytes()
    if b"\x00" in raw:
        return None, True
    try:
        return raw.decode("utf-8-sig"), False
    except UnicodeDecodeError:
        return None, True


def _read_blocked(path: Path) -> Optional[str]:
    """Fail closed when the canonical read guard cannot classify a file."""
    try:
        return get_read_block_error(str(path.resolve()))
    except Exception as exc:
        return f"Access denied: the credential guard failed: {exc}"


def skill_view(
    config: Any,
    name: str,
    file_path: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Load one SKILL.md or one contained linked file."""
    entry, error = _resolve_skill(config, name)
    if error or entry is None:
        return tool_error(error or "Skill lookup failed", success=False)

    skill_dir: Path = entry["_skill_dir"]
    target = entry["_skill_md"]
    if file_path is not None:
        path_error = _relative_path_error(file_path)
        if path_error:
            return tool_error(path_error, success=False)
        target = skill_dir / file_path
        containment_error = validate_within_dir(target, skill_dir)
        if containment_error:
            return tool_error(containment_error, success=False)
        if not target.is_file():
            return _json(
                {
                    "success": False,
                    "error": f"File '{file_path}' not found in skill '{name}'.",
                    "available_files": _available_linked_files(skill_dir),
                }
            )

    blocked = _read_blocked(target)
    if blocked:
        return tool_error(blocked, success=False)

    try:
        content, is_binary = _read_skill_file(target)
    except OSError as exc:
        return tool_error(f"Failed to read skill '{name}': {exc}", success=False)
    if is_binary:
        return _json(
            {
                "success": True,
                "name": entry["name"],
                "file": file_path,
                "content": f"[Binary file: {target.name}, size: {target.stat().st_size} bytes]",
                "is_binary": True,
                "_source_path": str(target),
            }
        )

    if file_path is not None:
        return _json(
            {
                "success": True,
                "name": entry["name"],
                "file": file_path,
                "content": content,
                "file_type": target.suffix,
                "_source_path": str(target),
            }
        )

    rendered_content = content
    try:
        rendered_content = preprocess_skill_content(
            content,
            skill_dir,
            session_id=session_id,
            skills_cfg=config.settings.section("skills"),
        )
    except Exception:
        logger.debug("Could not preprocess skill content for %s", name, exc_info=True)

    frontmatter = entry["_frontmatter"]
    metadata = frontmatter.get("metadata")
    hermes_meta = metadata.get("hermes", {}) if isinstance(metadata, dict) else {}
    tags = _parse_tags(hermes_meta.get("tags") or frontmatter.get("tags", ""))
    related = _parse_tags(
        hermes_meta.get("related_skills") or frontmatter.get("related_skills", "")
    )
    linked = _linked_files(skill_dir)
    result: Dict[str, Any] = {
        "success": True,
        "name": entry["name"],
        "description": entry["description"],
        "version": entry["version"],
        "tags": tags,
        "related_skills": related,
        "content": rendered_content,
        "path": entry["identifier"] + "/SKILL.md",
        "skill_dir": str(skill_dir),
        "linked_files": linked,
        "required_credential_files": entry["required_credential_files"],
        "readiness_status": "available",
        "_source_path": str(target),
    }
    if linked:
        result["usage_hint"] = (
            "To view a linked file, call skill_view(name, file_path), for example "
            "file_path='references/api.md'."
        )
    return _json(result)


def _fingerprint(path: str) -> Optional[Tuple[int, int]]:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _deduplicate_view(raw: str, args: Dict[str, Any], context: ToolContext) -> str:
    try:
        payload = json.loads(raw)
    except ValueError:
        return raw
    source = payload.pop("_source_path", None)
    if not payload.get("success") or not source:
        return _json(payload)
    fingerprint = _fingerprint(source)
    if fingerprint is None:
        return _json(payload)

    key = (str(args.get("name", "")), str(args.get("file_path") or ""))
    with _DEDUP_LOCK:
        tracker = context.state.setdefault("skill_views", {})
        previous = tracker.get(key)
        tracker[key] = fingerprint
        while len(tracker) > _SKILL_VIEW_DEDUP_CAP:
            tracker.pop(next(iter(tracker)))
    if previous == fingerprint:
        return _json(
            {
                "success": True,
                "name": args.get("name"),
                "file": args.get("file_path"),
                "dedup": True,
                "content_returned": False,
                "message": _SKILL_VIEW_DEDUP_MESSAGE,
            }
        )
    return _json(payload)


async def handle_skills_list(args: Dict[str, Any], context: ToolContext) -> str:
    category = args.get("category")
    if category is not None and not isinstance(category, str):
        return tool_error("category must be text")
    return await asyncio.to_thread(skills_list, context.config, category)


async def handle_skill_view(args: Dict[str, Any], context: ToolContext) -> str:
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return tool_error("name must be non-empty text")
    file_path = args.get("file_path")
    if file_path is not None and not isinstance(file_path, str):
        return tool_error("file_path must be text")
    raw = await asyncio.to_thread(
        skill_view,
        context.config,
        name,
        file_path,
        context.chat_id,
    )
    return _deduplicate_view(raw, args, context)


SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": "List available local skills (name and description). Use skill_view(name) to load full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter to narrow results.",
            }
        },
        "required": [],
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "Load a local skill's full instructions or one of its linked references, templates, assets, or scripts.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name. Use skills_list to see available skills.",
            },
            "file_path": {
                "type": "string",
                "description": "Optional linked path within the skill, such as references/api.md.",
            },
        },
        "required": ["name"],
    },
}

SKILLS_TOOLS = (
    Tool("skills_list", "skills", SKILLS_LIST_SCHEMA, handle_skills_list),
    Tool("skill_view", "skills", SKILL_VIEW_SCHEMA, handle_skill_view),
)


__all__ = [
    "SKILLS_LIST_SCHEMA",
    "SKILLS_TOOLS",
    "SKILL_VIEW_SCHEMA",
    "build_skills_prompt",
    "discover_skills",
    "handle_skill_view",
    "handle_skills_list",
    "skill_view",
    "skills_directory",
    "skills_list",
]
