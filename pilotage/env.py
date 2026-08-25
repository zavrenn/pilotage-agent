"""Reading sensitive deployment values from a `.env` file.

The allowlisted identities live beside the agent state rather than in the
behavioral configuration. Real environment variables always win over the
file, which keeps service-manager secret injection possible.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import List

from .config import REPO_ROOT, state_dir


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")


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


def update_env_values(path: Path, values: Mapping[str, str]) -> None:
    """Atomically replace selected assignments while preserving the env file."""

    replacements = {str(key): str(value) for key, value in values.items()}
    if not replacements:
        return
    for key, value in replacements.items():
        if not _ENV_KEY_RE.fullmatch(key):
            raise ValueError(f"Invalid environment key: {key!r}")
        if "\n" in value or "\r" in value:
            raise ValueError(f"Environment value for {key} cannot contain newlines")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []

    written: set[str] = set()
    updated: list[str] = []
    for line in lines:
        match = _ENV_ASSIGNMENT_RE.match(line)
        key = match.group(1) if match else ""
        if key not in replacements:
            updated.append(line)
            continue
        if key not in written:
            updated.append(f"{key}={replacements[key]}")
            written.add(key)

    for key, value in replacements.items():
        if key not in written:
            updated.append(f"{key}={value}")

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(updated) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
