"""Recognize obsolete runtime replies without filtering business answers.

These signatures are limited to notices emitted by the previous runtime. Whole
notices can be retired; a trailing attachment notice can be replaced within its
original chunk. Business text and accepted delivery evidence stay intact.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from .i18n import SUPPORTED_LANGUAGES, t


LEGACY_ATTACHMENT_NOTICE = (
    "[File delivery blocked: restricted sessions can only deliver files "
    "from the current session's exports directory.]"
)

_FIXED_NOTICES = frozenset({
    "Codex response remained incomplete after 3 continuation attempts",
    LEGACY_ATTACHMENT_NOTICE,
    "I stopped because conversation state could not be saved safely. "
    "Check storage first. If an earlier request may have acted, verify "
    "it before using /new.",
    "Je me suis arrêté car l'état de la conversation n'a pas pu être "
    "enregistré en toute sécurité. Vérifiez d'abord le stockage. Si une "
    "demande précédente a pu agir, vérifiez-la avant d'utiliser /new.",
    "توقفت لأن حالة المحادثة لم تُحفظ بأمان. تحقّق من التخزين أولاً. "
    "إذا كان الطلب السابق ربما نفّذ إجراءً، فتحقّق منه قبل استخدام /new.",
})


_DIAGNOSTIC_PATTERNS = (
    re.compile(r"Scheduled job (?:'[^\n]*'|\"[^\n]*\") failed\. Check the agent logs\."),
)
_LEGACY_HELP_HEADERS = {
    "en": "Management commands:",
    "fr": "Commandes de gestion :",
    "ar": "أوامر الإدارة:",
}
_LEGACY_COMMANDS = (
    ("help", "Show the available management commands", "commands"),
    ("new", "Start a fresh conversation", "reset"),
    ("stop", "Stop the active request", ""),
    ("approve", "Allow the oldest pending change once", ""),
    ("deny", "Refuse the oldest pending change", ""),
    ("status", "Show the running agent's essential status", ""),
)
_LEGACY_HELP = frozenset(
    "\n".join([
        _LEGACY_HELP_HEADERS[language],
        *(
            f"/{name} — {description}"
            + (f" ({t('commands.alias', language, alias=alias)})" if alias else "")
            for name, description, alias in _LEGACY_COMMANDS
        ),
    ])
    for language in SUPPORTED_LANGUAGES
)


def is_legacy_technical_reply(content: str) -> bool:
    """Only a complete, recognized old runtime notice may be retired."""

    return (
        content in _FIXED_NOTICES
        or content in _LEGACY_HELP
        or any(pattern.fullmatch(content) for pattern in _DIAGNOSTIC_PATTERNS)
    )


def has_legacy_attachment_notice(content: str) -> bool:
    return content == LEGACY_ATTACHMENT_NOTICE or content.endswith(
        "\n\n" + LEGACY_ATTACHMENT_NOTICE
    )


def attachment_notice_replacements(
    content: str,
    chunks: Sequence[str],
    format_text: Callable[[str], str],
    language: str,
) -> dict[int, tuple[str, ...]]:
    """Replace the reserved trailing notice inside its original text chunk.

    Keep business text, chunk boundaries and numbering intact. All supported
    translations are included so a retry can retain its already recorded wording
    even after the operator changes the display language.
    """
    if not has_legacy_attachment_notice(content):
        return {}
    notice = format_text(LEGACY_ATTACHMENT_NOTICE).strip()
    # Both production splitters keep this short trailing paragraph together.
    # Refuse an unexpected plan rather than leaking a fragment or dropping text.
    if not chunks or sum(chunk.count(notice) for chunk in chunks) != 1 or notice not in chunks[-1]:
        raise ValueError("The legacy attachment notice is not intact in its delivery plan")
    languages = dict.fromkeys((language, *SUPPORTED_LANGUAGES))
    return {len(chunks) - 1: tuple(dict.fromkeys(
        chunks[-1].replace(notice, format_text(t("media.delivery_unavailable", locale)).strip(), 1)
        for locale in languages
    ))}
