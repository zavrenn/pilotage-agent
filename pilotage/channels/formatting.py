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
# ***both at once*** has to be recognised before either rule for one of them:
# read as bold it keeps a stray asterisk, read as italic it keeps two.
_COMBINED = re.compile(r"\*\*\*(?=\S)(.+?)(?<=\S)\*\*\*")
# The delimiters hug a non-space, so "2 ** 3 ** 4" stays arithmetic.
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
# One asterisk pair — but not half of a **bold** one, and not a list bullet
# ("* item"), which is why the delimiters must hug a non-space.
_ITALIC = re.compile(r"(?<!\*)\*(?!\s|\*)([^*\n]*?\S[^*\n]*?)\*(?!\*)")
_STRIKETHROUGH = re.compile(r"~~(.+?)~~")
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# __bold__ is markdown too, and is deliberately not converted: models emit it
# almost never, while ``__init__`` and other double-underscore names are
# ordinary words in a technical answer. The rule would cost more than it pays.

_MARK = "\x00"
_PLACEHOLDER = re.compile(rf"{_MARK}(\d+){_MARK}")


def to_whatsapp(text: str) -> str:
    """Rewrite markdown emphasis into the marks WhatsApp actually renders."""
    if not text:
        return text

    result = _ODD_SPACE.sub(" ", _INVISIBLE.sub("", text))

    # Text that must not be read again — quoted code, and emphasis already
    # converted — waits here and is put back at the end.
    parked: list[str] = []

    def park(snippet: str) -> str:
        parked.append(snippet)
        return f"{_MARK}{len(parked) - 1}{_MARK}"

    def park_match(match: "re.Match[str]") -> str:
        return park(match.group(0))

    def combined_to_whatsapp(match: "re.Match[str]") -> str:
        return park(f"_*{match.group(1)}*_")

    def bold_to_whatsapp(match: "re.Match[str]") -> str:
        # The italic inside a bold span is converted here, while the span is
        # still whole text. A moment later it is one placeholder, and the
        # italic rule can no longer see into it.
        inner = _ITALIC.sub(r"_\1_", match.group(1))
        return park(f"*{inner}*")

    def unbold(title: str) -> str:
        # A heading line is made bold as a whole. Bold inside it would close
        # that emphasis halfway through and show the marks instead of hiding
        # them, so the inner pair comes off. Code in a heading stays as it is.
        def bare(match: "re.Match[str]") -> str:
            snippet = parked[int(match.group(1))]
            if snippet.startswith("_*") and snippet.endswith("*_"):
                return f"_{snippet[2:-2]}_"
            if snippet.startswith("*") and snippet.endswith("*"):
                return snippet[1:-1]
            return match.group(0)

        return _PLACEHOLDER.sub(bare, title)

    def heading_to_bold(match: "re.Match[str]") -> str:
        return f"*{unbold(match.group(1).strip())}*"

    result = _FENCED_CODE.sub(park_match, result)
    result = _INLINE_CODE.sub(park_match, result)

    # Struck-through text and links go first, while the reply is still plain
    # text throughout. They are the two rules that have to reach inside an
    # emphasis span — "**~~dropped~~**", "**[the report](...)**" — and a span
    # stops being readable the moment it is parked.
    result = _STRIKETHROUGH.sub(r"~\1~", result)
    result = _LINK.sub(r"\1 (\2)", result)

    # Emphasis is converted from the outside in, and each span is parked as
    # soon as it is converted. WhatsApp writes bold with the single asterisk
    # markdown uses for italic, so a converted span left in place would be read
    # a second time by the next rule and turned back into markup.
    result = _COMBINED.sub(combined_to_whatsapp, result)
    result = _BOLD.sub(bold_to_whatsapp, result)
    result = _ITALIC.sub(r"_\1_", result)

    result = _HEADING.sub(heading_to_bold, result)

    # Newest first: a parked span can contain a placeholder made before it,
    # never one made after it.
    for index in reversed(range(len(parked))):
        result = result.replace(f"{_MARK}{index}{_MARK}", parked[index])
    return result
