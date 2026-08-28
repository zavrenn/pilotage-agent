"""Conservative advisories for terminal results that look falsely successful."""

from __future__ import annotations

import re
from typing import Optional

_SCAN_CHARS = 32_000
_PASSTHROUGH_CONSUMERS = r"(?:tail|head|cat|tee|less|more|wc|sort|uniq)"
_MASKING_PIPE_RE = re.compile(
    r"(?<!\|)\|(?!\|)\s*" + _PASSTHROUGH_CONSUMERS + r"\b[^|]*$"
)
_MASKING_OR_RE = re.compile(r"\|\|\s*(?:echo\b|printf\b|true\b|:\s|:$)")

# These commands intentionally emit or search arbitrary text. Seeing a build
# failure string in their piped output says nothing about their own success.
_READ_ONLY_HEADS = frozenset(
    {
        "ag",
        "awk",
        "cat",
        "dmesg",
        "echo",
        "find",
        "grep",
        "head",
        "journalctl",
        "jq",
        "ls",
        "printf",
        "rg",
        "sed",
        "strings",
        "tail",
        "zcat",
    }
)

# Require a tool-specific failure shape, not the generic word "error".
_FAILURE_SHAPES = re.compile(
    r"(?:"
    r"error\[E\d+\]"
    r"|error: could not compile"
    r"|error: aborting due to"
    r"|Traceback \(most recent call last\)"
    r"|(?m:^(?:=+ )?\d+ failed)"
    r"|(?m:^FAILED (?:\S+::|\S+\.py))"
    r"|compilation terminated\."
    r"|npm ERR!"
    r"|BUILD FAILED|Build FAILED"
    r"|FAILED: "
    r"|(?m:^make(?:\[\d+\])?: \*\*\*)"
    r")"
)


def _first_token(command: str) -> str:
    for token in (command or "").strip().split():
        if "=" in token and not token.startswith(("=", "./", "/")):
            continue
        return token.rsplit("/", 1)[-1]
    return ""


def masked_success_advisory(command: str, output: str) -> Optional[str]:
    """Warn when exit zero likely belongs to a status-masking shell suffix."""

    command = command or ""
    window = (output or "")[:_SCAN_CHARS]
    if not command or not window:
        return None
    if _first_token(command) in _READ_ONLY_HEADS:
        return None
    if not _FAILURE_SHAPES.search(window):
        return None
    if _MASKING_PIPE_RE.search(command):
        return (
            "exit_code 0 is from the last pipeline command, not necessarily "
            "the build/test before it, and the output contains strong failure "
            "markers. Treat this run as failed until verified; rerun the "
            "build/test command without the pipe. Terminal output is already bounded."
        )
    if _MASKING_OR_RE.search(command):
        return (
            "exit_code 0 is from the `||` fallback, not necessarily the "
            "build/test before it, and the output contains strong failure "
            "markers. Treat this run as failed until verified; rerun the "
            "build/test command without the fallback."
        )
    return None


__all__ = ["masked_success_advisory"]
