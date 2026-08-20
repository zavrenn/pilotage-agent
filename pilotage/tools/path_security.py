"""Shared path validation helpers for tool implementations.

Copied from ``tmp/hermes-agent/tools/path_security.py``. These two checks are
the proven boundary used by Hermes' skill tools and are kept unchanged.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def validate_within_dir(path: Path, root: Path) -> Optional[str]:
    """Ensure *path* resolves to a location within *root*.

    Returns an error message string if validation fails, or ``None`` if the
    path is safe. Uses ``Path.resolve()`` to follow symlinks and normalize
    ``..`` components.
    """
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        resolved.relative_to(root_resolved)
    except (ValueError, OSError) as exc:
        return f"Path escapes allowed directory: {exc}"
    return None


def has_traversal_component(path_str: str) -> bool:
    """Return True if *path_str* contains ``..`` traversal components."""
    parts = Path(path_str).parts
    return ".." in parts

