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

Dropped from the Hermes version: profile mirroring, sandbox and container path
classification, and the approval gate on `~/.ssh/config`. We have one home
directory and no approval prompts — `approvals.mode: off` — so an
approval-gated path would only ever fail closed, which is a deny with extra
steps. `~/.ssh` is denied by prefix instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..config import state_dir


def _real(path: str) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def _state_files() -> tuple[str, ...]:
    """Our own state that a file tool must never rewrite.

    The credentials are obvious. The conversation database is here because a
    tool that can edit history can make the agent remember something that was
    never said, and the configuration is here because an agent that can edit
    its own configuration can switch its own guards off.
    """
    home = state_dir()
    return (
        _real(str(home / "codex-auth.json")),
        _real(str(home / "conversations.db")),
        _real(str(home / "config.yaml")),
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
    """Return 'credential', 'safe_root', or None when the write is allowed."""
    home = _real("~")
    resolved = _real(path)

    if resolved in build_write_denied_paths(home):
        return "credential"
    for prefix in build_write_denied_prefixes(home):
        if resolved.startswith(prefix):
            return "credential"

    if safe_roots:
        for root in safe_roots:
            if resolved == root or resolved.startswith(root + os.sep):
                return None
        return "safe_root"

    return None


def is_write_denied(path: str, safe_roots: Sequence[str] = ()) -> bool:
    return _classify_write_denial(path, safe_roots) is not None


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

    for blocked in _state_files():
        if str(resolved) == blocked:
            return (
                f"Access denied: {path} is agent state and cannot be read directly. "
                f"{_NOT_A_BOUNDARY}"
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
    "get_write_denied_error",
    "is_write_denied",
    "raise_if_read_blocked",
    "resolve_safe_roots",
]
