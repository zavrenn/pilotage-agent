"""Working-directory instructions for one agent session.

Adapted from Hermes ``agent/prompt_builder.py``. Hermes recognizes several
framework and editor formats; Pilotage's production contract needs only
``AGENTS.md``. The useful mechanism remains the same: load instructions from
the git root down to the configured working directory, scan every file before
prompt injection, deduplicate copies, and bound what enters the prompt.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .tools.threat_patterns import scan_for_threats

logger = logging.getLogger(__name__)

CONTEXT_FILE_MAX_CHARS = 20_000
CONTEXT_TRUNCATE_HEAD_RATIO = 0.7
CONTEXT_TRUNCATE_TAIL_RATIO = 0.2


def _scan_context_content(content: str, filename: str) -> str:
    """Block promptware before a workspace file enters the system prompt."""
    # A leading UTF-8 BOM is an editor artifact. Hermes strips only the
    # leading one; invisible characters elsewhere remain subject to scanning.
    content = content.removeprefix("\ufeff")
    findings = scan_for_threats(content, scope="context")
    if findings:
        joined = ", ".join(findings)
        logger.warning("Context file %s blocked: %s", filename, joined)
        return (
            f"[BLOCKED: {filename} contained potential prompt injection "
            f"({joined}). Content not loaded.]"
        )
    return content


def _find_git_root(start: Path) -> Path | None:
    """Return the nearest ancestor containing ``.git``, if there is one."""
    current = start.resolve()
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _truncate_content(
    content: str,
    filename: str,
    *,
    max_chars: int = CONTEXT_FILE_MAX_CHARS,
    read_path: str | None = None,
) -> str:
    """Keep Hermes's head/tail prompt budget and recovery hint."""
    if len(content) <= max_chars:
        return content
    target = read_path or filename
    logger.warning(
        "Context file %s truncated: %d chars exceeds limit of %d",
        filename,
        len(content),
        max_chars,
    )
    head_chars = int(max_chars * CONTEXT_TRUNCATE_HEAD_RATIO)
    tail_chars = int(max_chars * CONTEXT_TRUNCATE_TAIL_RATIO)
    marker = (
        f"\n\n[...truncated {filename}: kept {head_chars}+{tail_chars} of "
        f"{len(content)} chars. The middle is omitted — if you need the full "
        f"instructions, read the complete file with the read_file tool: "
        f"{target}]\n\n"
    )
    return content[:head_chars] + marker + content[-tail_chars:]


def _agents_md_directory_chain(cwd_path: Path) -> list[Path]:
    """Directories from the git root through ``cwd``, or only ``cwd``."""
    current = cwd_path.resolve()
    root = _find_git_root(current)
    if root is None or root == current:
        return [current]
    try:
        relative = current.relative_to(root)
    except ValueError:
        return [current]
    chain = [root]
    directory = root
    for part in relative.parts:
        directory = directory / part
        chain.append(directory)
    return chain


def _load_agents_md(
    cwd_path: Path, *, max_chars: int = CONTEXT_FILE_MAX_CHARS
) -> str:
    """Load and merge ``AGENTS.md`` from git root to working directory."""
    cwd_resolved = cwd_path.resolve()
    sections: list[str] = []
    seen_content: set[str] = set()

    for directory in _agents_md_directory_chain(cwd_resolved):
        # Pilotage keeps only the production instruction format. The lowercase
        # spelling is Hermes's compatibility alias; the first match wins.
        for name in ("AGENTS.md", "agents.md"):
            candidate = directory / name
            if not candidate.exists():
                continue
            try:
                content = candidate.read_text(encoding="utf-8").strip()
            except Exception as exc:  # noqa: BLE001 - Hermes skips unreadable hints
                logger.debug("Could not read %s: %s", candidate, exc)
                continue
            if not content:
                continue
            if content in seen_content:
                break
            seen_content.add(content)
            label = (
                name
                if directory == cwd_resolved
                else os.path.relpath(candidate, cwd_resolved)
            )
            scanned = _scan_context_content(content, label)
            section = _truncate_content(
                f"## {label}\n\n{scanned}",
                label,
                max_chars=max_chars,
                read_path=str(candidate),
            )
            sections.append(section)
            break

    if not sections:
        return ""
    if len(sections) == 1:
        return sections[0]
    return _truncate_content(
        "\n\n".join(sections),
        "AGENTS.md (directory chain)",
        max_chars=max_chars,
        read_path=str(cwd_resolved / "AGENTS.md"),
    )


def build_context_files_prompt(
    cwd: str | os.PathLike[str], *, max_chars: int = CONTEXT_FILE_MAX_CHARS
) -> str:
    """Build the optional Hermes-shaped workspace-context prompt block."""
    project_context = _load_agents_md(
        Path(cwd).expanduser().resolve(), max_chars=max_chars
    )
    if not project_context:
        return ""
    return (
        "# Project Context\n\n"
        "The following project context files have been loaded and should be "
        "followed:\n\n"
        f"{project_context}"
    )


__all__ = ["CONTEXT_FILE_MAX_CHARS", "build_context_files_prompt"]
