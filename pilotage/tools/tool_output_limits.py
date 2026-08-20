"""Hermes file-output limits used by the extracted local file stack."""

from __future__ import annotations

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_LINE_LENGTH = 2000


def get_max_lines() -> int:
    return DEFAULT_MAX_LINES


def get_max_line_length() -> int:
    return DEFAULT_MAX_LINE_LENGTH


__all__ = ["get_max_line_length", "get_max_lines"]
