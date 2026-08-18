"""Resolve gateway ``terminal.cwd`` placeholder values to ``TERMINAL_CWD``.

When ``terminal.cwd`` is unset or a placeholder (``.``, ``auto``, ``cwd``),
fall back to ``MESSAGING_CWD`` and then to the host home directory.
"""

from __future__ import annotations

CWD_PLACEHOLDERS = frozenset({".", "auto", "cwd"})


def resolve_placeholder_terminal_cwd(
    *,
    configured_cwd: str,
    messaging_cwd: str | None,
    home_fallback: str,
) -> str | None:
    """Return the ``TERMINAL_CWD`` value to set, or ``None`` to leave it unset."""
    if configured_cwd and configured_cwd not in CWD_PLACEHOLDERS:
        return configured_cwd

    messaging = (messaging_cwd or "").strip()
    return messaging or home_fallback
