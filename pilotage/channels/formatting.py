"""Markdown in, WhatsApp marks out.

The model writes markdown whether or not we ask it to — it is what every model
was trained on, and an instruction saying otherwise only wins some of the time.
WhatsApp does not read markdown: it has its own marks, and anything else
arrives as literal asterisks and hashes. So the last thing we do before handing
a reply to the bridge is translate it.

The rule set is deliberately short. Every rule is also a chance to mangle text
nobody meant as markdown — an asterisk in a formula, a bracket in a citation —
so we convert the marks a chat reply actually uses and leave everything else
alone. Code is set aside entirely: inside backticks a character is quoted, not
written, and rewriting it would change what it says.
"""

from __future__ import annotations

import re

# Zero-width characters some models sprinkle around emphasis. WhatsApp renders
# them as a stray blob rather than hiding them. NUL goes too, so nothing in the
# text can imitate the placeholders used below.
_INVISIBLE = re.compile(r"[\x00\u200b\u2060\u2063\ufeff]")
# Exotic spaces, same story: they read as mojibake in a chat bubble.
_ODD_SPACE = re.compile(r"[\u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]")

_FENCED_CODE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
# One asterisk pair — but not half of a **bold** one, and not a list bullet
# ("* item"), which is why the delimiters must hug a non-space.
_ITALIC = re.compile(r"(?<!\*)\*(?!\s|\*)([^*\n]*?\S[^*\n]*?)\*(?!\*)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_STRIKETHROUGH = re.compile(r"~~(.+?)~~")
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# __bold__ is markdown too, and is deliberately not converted: models emit it
# almost never, while ``__init__`` and other double-underscore names are
# ordinary words in a technical answer. The rule would cost more than it pays.

_MARK = "\x00"


def _heading_to_bold(match: "re.Match[str]") -> str:
    """A heading becomes a bold line — WhatsApp offers nothing else."""
    title = match.group(1).strip()
    # "## **Title**" has already been through the bold rule and arrived here as
    # "*Title*". Wrapping it again would show the asterisks instead of hiding
    # them.
    while len(title) > 1 and title.startswith("*") and title.endswith("*"):
        title = title[1:-1].strip()
    return f"*{title}*"


def to_whatsapp(text: str) -> str:
    """Rewrite markdown emphasis into the marks WhatsApp actually renders."""
    if not text:
        return text

    result = _ODD_SPACE.sub(" ", _INVISIBLE.sub("", text))

    quoted: list[str] = []

    def _park(match: "re.Match[str]") -> str:
        quoted.append(match.group(0))
        return f"{_MARK}{len(quoted) - 1}{_MARK}"

    result = _FENCED_CODE.sub(_park, result)
    result = _INLINE_CODE.sub(_park, result)

    # Italic first: *one* means italic in markdown but bold in WhatsApp, so it
    # has to move out of the way before **two** collapses down to one.
    result = _ITALIC.sub(r"_\1_", result)
    result = _BOLD.sub(r"*\1*", result)
    result = _STRIKETHROUGH.sub(r"~\1~", result)
    result = _HEADING.sub(_heading_to_bold, result)
    result = _LINK.sub(r"\1 (\2)", result)

    for index, snippet in enumerate(quoted):
        result = result.replace(f"{_MARK}{index}{_MARK}", snippet)
    return result
