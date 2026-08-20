"""Reading sensitive deployment values from a `.env` file.

The allowlisted identities live beside the agent state rather than in the
behavioral configuration. Real environment variables always win over the
file, which keeps service-manager secret injection possible.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .config import REPO_ROOT, state_dir


def candidate_env_files() -> List[Path]:
    override = os.environ.get("PILOTAGE_ENV_FILE", "").strip()
    if override:
        return [Path(override).expanduser()]
    return [state_dir() / ".env", REPO_ROOT / ".env"]


def load_env_files() -> List[Path]:
    """Load the first `.env` that exists. Returns the files actually read."""
    loaded: List[Path] = []
    for path in candidate_env_files():
        if path.is_file():
            _load(path)
            loaded.append(path)
            break
    return loaded


def _load(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value
