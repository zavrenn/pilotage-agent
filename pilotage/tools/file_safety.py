"""Paths the file tools refuse to touch.

Copied from Hermes (agent/file_safety.py), keeping its deny lists and its
honesty about what they are worth. Hermes' own words, which apply here
unchanged:

    **This is NOT a security boundary.** The terminal tool runs as the same
    OS user with shell access; the agent can still ``cat`` the file and
    exfiltrate it.

It exists because it does two useful things anyway. A model that respects a
tool error stops instead of reaching for the shell — which is most of them,
most of the time. And an attempt to read credentials leaves a named refusal in
the log, which is far easier to notice than a `cat` among a hundred other
commands. Treat it as "usually helps", never as "cannot be bypassed".

Dropped from the Hermes version: sandbox and container path classification, and
the approval gate on `~/.ssh/config`. Pilotage adds a small profile-routing
guard so ordinary file calls cannot enter default or sibling profile state.
Skill paths are classified separately for the live approval gate. Auto-loaded
``AGENTS.md`` files remain denied: they are not one of the three approved write
classes in the current production contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..config import state_dir
from ..profiles import default_state_root


_PROTECTED_INSTRUCTION_BASENAMES = frozenset({"agents.md"})


def _real(path: str) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_other_profile_state(path: str) -> bool:
    """Keep the file tool inside the selected profile's state ownership."""
    selected = Path(_real(str(state_dir())))
    main = Path(_real(str(default_state_root())))
    resolved = Path(_real(path))
    if selected == main:
        return _within(resolved, main / "profiles")
    return _within(resolved, main) and not _within(resolved, selected)


def _is_protected_instruction_file(path: str) -> bool:
    """Match Hermes's case-insensitive normalized-path and realpath guard."""
    normalized = os.path.normpath(os.path.expanduser(str(path)))
    resolved = _real(path)
    return any(
        os.path.basename(candidate).casefold()
        in _PROTECTED_INSTRUCTION_BASENAMES
        for candidate in (normalized, resolved)
    )


def _state_files() -> tuple[str, ...]:
    """Our own state that a file tool must never rewrite.

    The credentials are obvious. The conversation database is here because a
    tool that can edit history can make the agent remember something that was
    never said, and the configuration is here because an agent that can edit
    its own configuration can switch its own guards off.
    """
    home = state_dir()
    return (
        _real(str(home / ".runtime.lock")),
        _real(str(home / "active_profile")),
        _real(str(home / "bridge.pid")),
        _real(str(home / "codex-auth.json")),
        _real(str(home / "codex-auth.json.lock")),
        _real(str(home / "config.yaml")),
        _real(str(home / "SOUL.md")),
        _real(str(home / "conversations.db")),
        _real(str(home / "conversations.db-journal")),
        _real(str(home / "conversations.db-shm")),
        _real(str(home / "conversations.db-wal")),
        _real(str(home / "delivery.db")),
        _real(str(home / "delivery.db-journal")),
        _real(str(home / "delivery.db-shm")),
        _real(str(home / "delivery.db-wal")),
        _real(str(home / "persistence-audit.db")),
        _real(str(home / "persistence-audit.db-journal")),
        _real(str(home / "persistence-audit.db-shm")),
        _real(str(home / "persistence-audit.db-wal")),
        _real(str(home / "cron" / ".jobs.lock")),
        _real(str(home / "cron" / "jobs.json")),
    )


def _state_prefixes() -> tuple[str, ...]:
    return (_real(str(state_dir() / "whatsapp")) + os.sep,)


def build_write_denied_paths(home: str) -> set[str]:
    """Exact paths that must never be written."""
    return {
        _real(p)
        for p in [
            os.path.join(home, ".ssh", "authorized_keys"),
            os.path.join(home, ".ssh", "id_rsa"),
            os.path.join(home, ".ssh", "id_ed25519"),
            os.path.join(home, ".netrc"),
            os.path.join(home, ".pgpass"),
            os.path.join(home, ".npmrc"),
            os.path.join(home, ".pypirc"),
            os.path.join(home, ".git-credentials"),
            "/etc/sudoers",
            "/etc/passwd",
            "/etc/shadow",
        ]
    } | set(_state_files())


def build_write_denied_prefixes(home: str) -> list[str]:
    """Directory prefixes that must never be written."""
    return [
        _real(p) + os.sep
        for p in [
            os.path.join(home, ".ssh"),
            os.path.join(home, ".aws"),
            os.path.join(home, ".gnupg"),
            os.path.join(home, ".kube"),
            "/etc/sudoers.d",
            "/etc/systemd",
            os.path.join(home, ".docker"),
            os.path.join(home, ".azure"),
            os.path.join(home, ".config", "gh"),
            os.path.join(home, ".config", "gcloud"),
            # These stores have dedicated validated mutation tools. Direct file
            # writes would bypass their threat scans, ownership, and locking.
            str(state_dir() / "cron"),
            str(state_dir() / "memories"),
        ]
    ] + list(_state_prefixes())


def resolve_safe_roots(roots: Iterable[str]) -> tuple[str, ...]:
    """Turn what the operator wrote into resolved directories.

    Empty means no restriction: writes are judged by the deny list alone.
    """
    resolved: list[str] = []
    for root in roots or ():
        text = str(root).strip()
        if not text:
            continue
        try:
            real = _real(text)
        except (OSError, ValueError):
            continue
        if real not in resolved:
            resolved.append(real)
    return tuple(resolved)


def _classify_write_denial(path: str, safe_roots: Sequence[str] = ()) -> Optional[str]:
    """Classify a refused write, or return ``None`` when it is allowed."""
    home = _real("~")
    resolved = _real(path)

    if _is_other_profile_state(resolved):
        return "profile"
    if resolved in build_write_denied_paths(home):
        return "credential"
    for prefix in build_write_denied_prefixes(home):
        if resolved.startswith(prefix):
            return "credential"
    if _is_protected_instruction_file(path):
        return "instruction"

    if safe_roots:
        for root in safe_roots:
            if resolved == root or resolved.startswith(root + os.sep):
                return None
        return "safe_root"

    return None


def is_write_denied(path: str, safe_roots: Sequence[str] = ()) -> bool:
    return _classify_write_denial(path, safe_roots) is not None


def get_write_approval_category(path: str) -> Optional[str]:
    """Return the persistent-write category for a path, if it has one.

    Check both the lexical path and its real target. This prevents a symlink
    either into or out of the profile skill tree from losing the approval.
    The ordinary denial guard still runs first and rejects sibling profiles.
    """

    written = Path(os.path.abspath(os.path.expanduser(str(path))))
    resolved = Path(_real(path))
    skill_written = Path(
        os.path.abspath(os.path.expanduser(str(state_dir() / "skills")))
    )
    skill_resolved = Path(_real(str(state_dir() / "skills")))
    if _within(written, skill_written) or _within(resolved, skill_resolved):
        return "skills"
    return None


def get_write_denied_error(
    path: str, *, verb: str = "Write", safe_roots: Sequence[str] = ()
) -> Optional[str]:
    """The message the model reads when a write is refused, or None."""
    denial = _classify_write_denial(path, safe_roots)
    if denial is None:
        return None
    if denial == "safe_root":
        listed = ", ".join(sorted(safe_roots))
        return (
            f"{verb} denied: '{path}' is outside the directories this agent may "
            f"write to ({listed}). Change tools.write_safe_roots to widen it."
        )
    if denial == "profile":
        return f"{verb} denied: '{path}' belongs to another agent profile."
    if denial == "instruction":
        return (
            f"{verb} denied: '{path}' is an agent instruction file. "
            "Changing it requires approval outside this tool and is therefore denied."
        )
    return f"{verb} denied: '{path}' is a protected system or credential file."


# Project-local environment files. They routinely hold API keys and database
# passwords, and .env.example is the documented-shape substitute the model can
# read instead.
BLOCKED_ENV_BASENAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
        ".env.test",
        ".env.staging",
        ".envrc",
    }
)

_NOT_A_BOUNDARY = (
    "(Defense in depth, not a security boundary — the terminal can still read it.)"
)


def get_read_block_error(path: str) -> Optional[str]:
    """The message the model reads when a read is refused, or None.

    Callers that resolve relative paths against a shell cwd of their own must
    pass the absolute path: this function resolves against the process cwd,
    which is not the same directory.
    """
    resolved = Path(path).expanduser().resolve()

    if _is_other_profile_state(str(resolved)):
        return (
            f"Access denied: {path} belongs to another agent profile. "
            f"{_NOT_A_BOUNDARY}"
        )

    for blocked in _state_files():
        if str(resolved) == blocked:
            return (
                f"Access denied: {path} is agent state and cannot be read directly. "
                f"{_NOT_A_BOUNDARY}"
            )

    for blocked_prefix in _state_prefixes():
        if str(resolved).startswith(blocked_prefix):
            return (
                f"Access denied: {path} is agent authentication state and cannot "
                f"be read directly. {_NOT_A_BOUNDARY}"
            )

    if resolved.name.lower() in BLOCKED_ENV_BASENAMES:
        return (
            f"Access denied: {path} carries secrets and cannot be read. Read "
            f".env.example instead if you need the shape of it. {_NOT_A_BOUNDARY}"
        )

    return None


def raise_if_read_blocked(path: str) -> None:
    """Raise ValueError if the read is refused. Never raises anything else."""
    try:
        blocked = get_read_block_error(path)
    except Exception:  # noqa: BLE001 - a broken guard must not break reading
        return
    if blocked:
        raise ValueError(blocked)


__all__ = [
    "BLOCKED_ENV_BASENAMES",
    "build_write_denied_paths",
    "build_write_denied_prefixes",
    "get_read_block_error",
    "get_write_approval_category",
    "get_write_denied_error",
    "is_write_denied",
    "raise_if_read_blocked",
    "resolve_safe_roots",
]
