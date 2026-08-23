"""Hermes-derived Telegram MarkdownV2 formatting and message chunking."""

from __future__ import annotations

import re
from typing import Callable, List, Optional


_MDV2_ESCAPE_RE = re.compile(r"([_*\[\]()~`>#\+\-=|{}.!\\])")
_CHUNK_INDICATOR_ON_FENCE_RE = re.compile(
    r"(?m)^``` (?P<indicator>(?:\\)?\(\d+/\d+(?:\\)?\))$"
)
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*){1,}\|?\s*$"
)


def utf16_len(text: str) -> int:
    """Count the UTF-16 code units Telegram uses for message limits. (Hermes)"""
    return len(text.encode("utf-16-le")) // 2


def _custom_unit_to_codepoints(
    text: str, budget: int, length: Callable[[str], int]
) -> int:
    if length(text) <= budget:
        return len(text)
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if length(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def _split_table_row(row: str) -> List[str]:
    written = row.strip()
    if written.startswith("|"):
        written = written[1:]
    if written.endswith("|"):
        written = written[:-1]
    return [cell.strip() for cell in written.split("|")]


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and "|" in stripped)


def _render_table_block(block: List[str]) -> str:
    if len(block) < 3:
        return "\n".join(block)
    headers = _split_table_row(block[0])
    if len(headers) < 2:
        return "\n".join(block)

    first_row = _split_table_row(block[2])
    has_row_label = len(first_row) == len(headers) + 1
    groups: List[str] = []
    for index, row in enumerate(block[2:], start=1):
        cells = _split_table_row(row)
        if has_row_label:
            heading = cells[0] if cells and cells[0] else f"Row {index}"
            values = cells[1:]
        else:
            heading = next((cell for cell in cells if cell), f"Row {index}")
            values = cells
        values = (values + [""] * len(headers))[: len(headers)]
        bullets = [
            f"\u2022 {header}: {value}"
            for header, value in zip(headers, values)
            if has_row_label or value != heading
        ]
        groups.append("\n".join([f"**{heading}**", *bullets]))
    return "\n\n".join(groups)


def _wrap_markdown_tables(text: str) -> str:
    """Rewrite GFM tables into Telegram-readable row groups. (Hermes)"""
    if "|" not in text or "-" not in text:
        return text
    lines = text.split("\n")
    output: List[str] = []
    in_fence = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            output.append(line)
            index += 1
            continue
        if (
            not in_fence
            and "|" in line
            and index + 1 < len(lines)
            and _TABLE_SEPARATOR_RE.match(lines[index + 1])
        ):
            block = [line, lines[index + 1]]
            cursor = index + 2
            while cursor < len(lines) and _is_table_row(lines[cursor]):
                block.append(lines[cursor])
                cursor += 1
            output.append(_render_table_block(block))
            index = cursor
            continue
        output.append(line)
        index += 1
    return "\n".join(output)


def _escape_mdv2(text: str) -> str:
    return _MDV2_ESCAPE_RE.sub(r"\\\1", text)


def strip_telegram_markdown(text: str) -> str:
    """Produce readable plaintext after Telegram rejects MarkdownV2. (Hermes)"""
    cleaned = re.sub(r"\\([_*\[\]()~`>#\+\-=|{}.!\\])", r"\1", text)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"~([^~]+)~", r"\1", cleaned)
    return re.sub(r"\|\|([^|]+)\|\|", r"\1", cleaned)


def to_telegram(text: str) -> str:
    """Convert ordinary model Markdown to Telegram MarkdownV2. (Hermes)"""
    if not text:
        return text

    placeholders: dict[str, str] = {}

    def protect(value: str) -> str:
        key = f"\x00PH{len(placeholders)}\x00"
        placeholders[key] = value
        return key

    result = _wrap_markdown_tables(text)

    def protect_fence(match: "re.Match[str]") -> str:
        raw = match.group(0)
        opening_end = raw.index("\n") + 1 if "\n" in raw[3:] else 3
        opening = raw[:opening_end]
        body = raw[opening_end:-3]
        body = body.replace("\\", "\\\\").replace("`", "\\`")
        return protect(opening + body + "```")

    result = re.sub(
        r"(```(?:[^\n]*\n)?[\s\S]*?```)",
        protect_fence,
        result,
    )
    result = re.sub(
        r"(`[^`]+`)",
        lambda match: protect(match.group(0).replace("\\", "\\\\")),
        result,
    )

    def convert_link(match: "re.Match[str]") -> str:
        display = _escape_mdv2(match.group(1))
        url = match.group(2).replace("\\", "\\\\").replace(")", "\\)")
        return protect(f"[{display}]({url})")

    result = re.sub(
        r"\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)",
        convert_link,
        result,
    )

    def convert_header(match: "re.Match[str]") -> str:
        inner = re.sub(r"\*\*(.+?)\*\*", r"\1", match.group(1).strip())
        return protect(f"*{_escape_mdv2(inner)}*")

    result = re.sub(
        r"^#{1,6}\s+(.+)$", convert_header, result, flags=re.MULTILINE
    )
    result = re.sub(
        r"\*\*(.+?)\*\*",
        lambda match: protect(f"*{_escape_mdv2(match.group(1))}*"),
        result,
    )
    result = re.sub(
        r"\*([^*\n]+)\*",
        lambda match: protect(f"_{_escape_mdv2(match.group(1))}_"),
        result,
    )
    result = re.sub(
        r"~~(.+?)~~",
        lambda match: protect(f"~{_escape_mdv2(match.group(1))}~"),
        result,
    )
    result = re.sub(
        r"\|\|(.+?)\|\|",
        lambda match: protect(f"||{_escape_mdv2(match.group(1))}||"),
        result,
    )

    def convert_quote(match: "re.Match[str]") -> str:
        prefix, content = match.group(1), match.group(2)
        if prefix.startswith("**") and content.endswith("||"):
            return protect(f"{prefix} {_escape_mdv2(content[:-2])}||")
        return protect(f"{prefix} {_escape_mdv2(content)}")

    result = re.sub(
        r"^((?:\*\*)?>{1,3}) (.+)$",
        convert_quote,
        result,
        flags=re.MULTILINE,
    )
    result = _escape_mdv2(result)
    for key in reversed(list(placeholders)):
        result = result.replace(key, placeholders[key])
    return result


def _separate_chunk_indicator_from_fence(text: str) -> str:
    return _CHUNK_INDICATOR_ON_FENCE_RE.sub(
        r"```\n\g<indicator>", text
    )


def split_telegram_message(
    content: str,
    max_length: int = 4096,
    length: Callable[[str], int] = utf16_len,
) -> List[str]:
    """Split long replies while preserving fenced code blocks. (Hermes)"""
    if length(content) <= max_length:
        return [content]

    indicator_reserve = 10
    fence_close = "\n```"
    chunks: List[str] = []
    remaining = content
    carry_language: Optional[str] = None

    while remaining:
        prefix = (
            f"```{carry_language}\n"
            if carry_language is not None
            else ""
        )
        headroom = (
            max_length
            - indicator_reserve
            - length(prefix)
            - length(fence_close)
        )
        if headroom < 1:
            headroom = max(1, max_length // 2)

        if length(prefix) + length(remaining) <= max_length - indicator_reserve:
            final = prefix + remaining
            in_code = carry_language is not None
            if in_code:
                for line in remaining.split("\n"):
                    if line.strip().startswith("```"):
                        in_code = not in_code
                if in_code:
                    final += fence_close
            chunks.append(final)
            break

        codepoint_limit = _custom_unit_to_codepoints(remaining, headroom, length)
        region = remaining[:codepoint_limit]
        split_at = region.rfind("\n")
        if split_at < codepoint_limit // 2:
            split_at = region.rfind(" ")
        if split_at < 1:
            split_at = max(1, codepoint_limit)

        candidate = remaining[:split_at]
        if (
            candidate.count("`") - candidate.count("\\`")
        ) % 2 == 1:
            last_backtick = candidate.rfind("`")
            while (
                last_backtick > 0
                and candidate[last_backtick - 1] == "\\"
            ):
                last_backtick = candidate.rfind("`", 0, last_backtick)
            if last_backtick > 0:
                safe_split = max(
                    candidate.rfind(" ", 0, last_backtick),
                    candidate.rfind("\n", 0, last_backtick),
                )
                if safe_split > codepoint_limit // 4:
                    split_at = safe_split

        body = remaining[:split_at]
        remaining = remaining[split_at:].lstrip()
        whole = prefix + body
        in_code = carry_language is not None
        language = carry_language or ""
        for line in body.split("\n"):
            stripped = line.strip()
            if not stripped.startswith("```"):
                continue
            if in_code:
                in_code = False
                language = ""
            else:
                in_code = True
                tag = stripped[3:].strip()
                language = tag.split()[0] if tag else ""
        if in_code:
            whole += fence_close
            carry_language = language
        else:
            carry_language = None
        chunks.append(whole)

    if len(chunks) > 1:
        total = len(chunks)
        chunks = [
            _separate_chunk_indicator_from_fence(
                re.sub(
                    r" \((\d+)/(\d+)\)$",
                    r" \\(\1/\2\\)",
                    f"{chunk} ({index + 1}/{total})",
                )
            )
            for index, chunk in enumerate(chunks)
        ]
    return chunks


__all__ = [
    "split_telegram_message",
    "strip_telegram_markdown",
    "to_telegram",
    "utf16_len",
]

