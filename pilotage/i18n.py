"""Small catalogs for the static messages Pilotage owns.

The model's language and register belong in ``SOUL.md``.  This module covers
only messages emitted without the model: management commands, approvals,
connection notices, resets, and generic failures.  It is the thin i18n slice
used by current Hermes, reduced to Pilotage's production languages.
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: Final = ("en", "fr", "ar")
DEFAULT_LANGUAGE: Final = "en"
# Preserve the existing Pilotage runtime's French static replies when an older
# profile has not yet gained ``display.language``.  New installs write this
# choice explicitly in config.yaml.
DEFAULT_PROFILE_LANGUAGE: Final = "fr"

_CATALOGS: Final[dict[str, dict[str, str]]] = {
    "en": {
        "runtime.failure": "I couldn't answer just now. Please try again.",
        "runtime.storage_failure": (
            "I stopped because conversation state could not be saved safely. "
            "Check storage first. If an earlier request may have acted, verify "
            "it before using /new."
        ),
        "runtime.reset": "Starting fresh. I forgot our conversation.",
        "runtime.reconnect": "Still nothing back from the model. Reconnecting…",
        "runtime.working": "Still working.",
        "session.auto_reset_idle": "Session automatically reset after inactivity.",
        "session.auto_reset_daily": "Session automatically reset on the daily schedule.",
        "approval.required": "Approval required — {category}",
        "approval.default_summary": "Persistent change requested.",
        "approval.instructions": "Reply /approve to allow this once, or /deny to refuse it.",
        "commands.header": "Management commands:",
        "commands.alias": "alias: /{alias}",
        "commands.usage": "Usage: /{command}",
        "commands.approved": "Approved. I’ll continue.",
        "commands.denied": "Denied. Nothing will be changed.",
        "commands.no_approval": "No approval is waiting.",
        "commands.unknown": "Unknown command: /{command}",
        "commands.profile": "Profile: {profile}",
        "commands.state": "State: {state}",
        "commands.auth": "ChatGPT auth: {scope}",
        "commands.model": "Model: {model}",
        "commands.channel": "Channel: {channel}",
        "commands.tools": "Tools: {tools}",
        "commands.cron": "Cron: {state} ({timezone})",
        "commands.enabled": "enabled",
        "commands.disabled": "disabled",
        "commands.none": "none",
        "commands.system_local": "system local",
        "commands.auth_profile": "this profile",
        "commands.auth_shared": "shared from default profile",
        "commands.auth_missing": "not signed in",
    },
    "fr": {
        "runtime.failure": "Je n'ai pas pu répondre pour le moment. Réessayez.",
        "runtime.storage_failure": (
            "Je me suis arrêté car l'état de la conversation n'a pas pu être "
            "enregistré en toute sécurité. Vérifiez d'abord le stockage. Si une "
            "demande précédente a pu agir, vérifiez-la avant d'utiliser /new."
        ),
        "runtime.reset": "On repart de zéro. J'ai oublié notre conversation.",
        "runtime.reconnect": "Toujours aucune réponse du modèle. Je me reconnecte…",
        "runtime.working": "Je continue.",
        "session.auto_reset_idle": "La session a été réinitialisée après une période d'inactivité.",
        "session.auto_reset_daily": "La session a été réinitialisée selon l'horaire quotidien.",
        "approval.required": "Approbation requise — {category}",
        "approval.default_summary": "Une modification persistante est demandée.",
        "approval.instructions": "Répondez /approve pour l'autoriser une fois, ou /deny pour la refuser.",
        "commands.header": "Commandes de gestion :",
        "commands.alias": "alias : /{alias}",
        "commands.usage": "Utilisation : /{command}",
        "commands.approved": "Approuvé. Je continue.",
        "commands.denied": "Refusé. Rien ne sera modifié.",
        "commands.no_approval": "Aucune approbation n'est en attente.",
        "commands.unknown": "Commande inconnue : /{command}",
        "commands.profile": "Profil : {profile}",
        "commands.state": "État : {state}",
        "commands.auth": "Authentification ChatGPT : {scope}",
        "commands.model": "Modèle : {model}",
        "commands.channel": "Canal : {channel}",
        "commands.tools": "Outils : {tools}",
        "commands.cron": "Cron : {state} ({timezone})",
        "commands.enabled": "activé",
        "commands.disabled": "désactivé",
        "commands.none": "aucun",
        "commands.system_local": "heure locale du système",
        "commands.auth_profile": "ce profil",
        "commands.auth_shared": "partagée depuis le profil par défaut",
        "commands.auth_missing": "non connecté",
    },
    "ar": {
        "runtime.failure": "تعذّر عليّ الرد الآن. حاول مرة أخرى.",
        "runtime.storage_failure": (
            "توقفت لأن حالة المحادثة لم تُحفظ بأمان. تحقّق من التخزين أولاً. "
            "إذا كان الطلب السابق ربما نفّذ إجراءً، فتحقّق منه قبل استخدام /new."
        ),
        "runtime.reset": "سنبدأ من جديد. لقد نسيت محادثتنا السابقة.",
        "runtime.reconnect": "لم يصل رد من النموذج بعد. أعيد الاتصال…",
        "runtime.working": "ما زلت أعمل.",
        "session.auto_reset_idle": "أُعيد ضبط الجلسة تلقائيًا بعد فترة من عدم النشاط.",
        "session.auto_reset_daily": "أُعيد ضبط الجلسة تلقائيًا وفق الجدول اليومي.",
        "approval.required": "الموافقة مطلوبة — {category}",
        "approval.default_summary": "طُلب تغيير دائم.",
        "approval.instructions": "أرسل /approve للسماح بهذه المرة، أو /deny للرفض.",
        "commands.header": "أوامر الإدارة:",
        "commands.alias": "اسم بديل: /{alias}",
        "commands.usage": "الاستخدام: /{command}",
        "commands.approved": "تمت الموافقة. سأتابع.",
        "commands.denied": "تم الرفض. لن يتغير شيء.",
        "commands.no_approval": "لا توجد موافقة معلّقة.",
        "commands.unknown": "أمر غير معروف: /{command}",
        "commands.profile": "الملف الشخصي: {profile}",
        "commands.state": "الحالة: {state}",
        "commands.auth": "مصادقة ChatGPT: {scope}",
        "commands.model": "النموذج: {model}",
        "commands.channel": "القناة: {channel}",
        "commands.tools": "الأدوات: {tools}",
        "commands.cron": "Cron: {state} ({timezone})",
        "commands.enabled": "مُفعّل",
        "commands.disabled": "مُعطّل",
        "commands.none": "لا شيء",
        "commands.system_local": "توقيت النظام المحلي",
        "commands.auth_profile": "هذا الملف الشخصي",
        "commands.auth_shared": "مشتركة من الملف الشخصي الافتراضي",
        "commands.auth_missing": "غير مسجّل الدخول",
    },
}


def normalize_language(value: str) -> str:
    """Return one supported catalog name or reject an operator typo."""

    written = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "english": "en",
        "french": "fr",
        "français": "fr",
        "francais": "fr",
        "arabic": "ar",
        "العربية": "ar",
    }
    written = aliases.get(written, written)
    base = written.split("-", 1)[0]
    if base in SUPPORTED_LANGUAGES:
        return base
    raise ValueError(
        "display.language must be en, fr, or ar, "
        f"not {value!r}"
    )


def t(key: str, language: str = DEFAULT_LANGUAGE, **values: object) -> str:
    """Resolve and safely format one static runtime message."""

    try:
        selected = normalize_language(language)
    except ValueError:
        selected = DEFAULT_LANGUAGE
    value = _CATALOGS.get(selected, {}).get(key)
    if value is None:
        value = _CATALOGS[DEFAULT_LANGUAGE].get(key, key)
    if not values:
        return value
    try:
        return value.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("i18n format failed for %s: %s", key, exc)
        return value


__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_PROFILE_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "normalize_language",
    "t",
]
