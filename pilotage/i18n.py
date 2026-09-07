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
        "runtime.failure": "I couldn't answer just now.",
        "runtime.incomplete_response": (
            "I couldn't finish my reply. Some actions may already be complete."
        ),
        "runtime.storage_failure": (
            "I couldn't continue, and I can't confirm whether your request was completed. "
            "Check the result, then start a new conversation with /new."
        ),
        "runtime.interrupted_unknown": (
            "I couldn't confirm whether the previous request was completed. "
            "Check the result, then start a new conversation with /new."
        ),
        "media.delivery_unavailable": "I couldn't send one or more attachments.",
        "document.too_large": "This document is too large to read. Please send a smaller file.",
        "document.unreadable": "I couldn't read this document. Please save a new copy and send it again.",
        "cron.failure": "I couldn't confirm that the scheduled request was completed.",
        "cron.unavailable": "Scheduling automatic messages is not available.",
        "capability.unavailable": "This feature is not available.",
        "runtime.reset": "Starting fresh. I forgot our conversation.",
        "runtime.working": "Still working.",
        "runtime.working_elapsed_under_minute": "{text} (<1 min)",
        "runtime.working_elapsed_minutes": "{text} ({minutes} min)",
        "runtime.stopped": "Stopped by your request.",
        "runtime.stopped_after_actions": (
            "Stopped by your request. Actions already completed have not been undone."
        ),
        "session.auto_reset_idle": "We're starting a new conversation after a break.",
        "session.auto_reset_daily": "We're starting a new conversation for today.",
        "approval.required": "Approval required — {category}",
        "approval.default_summary": "Persistent change requested.",
        "approval.instructions": "Reply /approve to allow this once, or /deny to refuse it.",
        "commands.header": "Available actions:",
        "commands.ready": "I'm available.",
        "commands.description_help": "Show available actions",
        "commands.description_new": "Start a new conversation",
        "commands.description_stop": "Stop the current request",
        "commands.description_status": "Check availability",
        "commands.alias": "alias: /{alias}",
        "commands.usage": "Usage: /{command}",
        "commands.approved": "Approved. I’ll continue.",
        "commands.denied": "Denied. Nothing will be changed.",
        "commands.no_approval": "No approval is waiting.",
        "commands.stopped": "Stopped.",
        "commands.stopped_after_actions": (
            "Stopped. Actions already completed have not been undone."
        ),
        "commands.stop_unknown": (
            "Stopped. I can't confirm whether the request was completed. "
            "Check the result, then start a new conversation with /new."
        ),
        "commands.stop_too_late": (
            "The answer was already complete and is being delivered."
        ),
        "commands.nothing_to_stop": "Nothing is running.",
        "commands.reset_running": (
            "I'm still working on a request. Send /stop to stop it, "
            "then /new to start a new conversation."
        ),
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
        "runtime.failure": "Je n'ai pas pu répondre pour le moment.",
        "runtime.incomplete_response": (
            "Je n'ai pas pu terminer ma réponse. Certaines actions ont peut-être déjà été effectuées."
        ),
        "runtime.storage_failure": (
            "Je n'ai pas pu continuer et je ne peux pas confirmer si votre demande a été exécutée. "
            "Vérifiez le résultat, puis commencez une nouvelle conversation avec /new."
        ),
        "runtime.interrupted_unknown": (
            "Je n'ai pas pu confirmer si la demande précédente a été exécutée. "
            "Vérifiez le résultat, puis commencez une nouvelle conversation avec /new."
        ),
        "media.delivery_unavailable": "Je n'ai pas pu envoyer une ou plusieurs pièces jointes.",
        "document.too_large": "Ce document est trop volumineux pour être lu. Veuillez envoyer un fichier plus petit.",
        "document.unreadable": "Je n'ai pas pu lire ce document. Veuillez en enregistrer une nouvelle copie et la renvoyer.",
        "cron.failure": "Je n'ai pas pu confirmer que la demande programmée a été exécutée.",
        "cron.unavailable": "La programmation des envois automatiques n’est pas disponible.",
        "capability.unavailable": "Cette fonctionnalité n’est pas disponible.",
        "runtime.reset": "On repart de zéro. J'ai oublié notre conversation.",
        "runtime.working": "Je continue.",
        "runtime.working_elapsed_under_minute": "{text} (<1 min)",
        "runtime.working_elapsed_minutes": "{text} ({minutes} min)",
        "runtime.stopped": "Arrêté à votre demande.",
        "runtime.stopped_after_actions": (
            "Arrêté à votre demande. Les actions déjà terminées n'ont pas été annulées."
        ),
        "session.auto_reset_idle": "Nous reprenons avec une nouvelle conversation après cette pause.",
        "session.auto_reset_daily": "Nous commençons une nouvelle conversation pour aujourd’hui.",
        "approval.required": "Approbation requise — {category}",
        "approval.default_summary": "Une modification persistante est demandée.",
        "approval.instructions": "Répondez /approve pour l'autoriser une fois, ou /deny pour la refuser.",
        "commands.header": "Actions disponibles :",
        "commands.ready": "Je suis disponible.",
        "commands.description_help": "Afficher les actions disponibles",
        "commands.description_new": "Commencer une nouvelle conversation",
        "commands.description_stop": "Arrêter la demande en cours",
        "commands.description_status": "Vérifier la disponibilité",
        "commands.alias": "alias : /{alias}",
        "commands.usage": "Utilisation : /{command}",
        "commands.approved": "Approuvé. Je continue.",
        "commands.denied": "Refusé. Rien ne sera modifié.",
        "commands.no_approval": "Aucune approbation n'est en attente.",
        "commands.stopped": "Arrêté.",
        "commands.stopped_after_actions": (
            "Arrêté. Les actions terminées avant l'arrêt n'ont pas été annulées."
        ),
        "commands.stop_unknown": (
            "Arrêté. Je ne peux pas confirmer si la demande a été exécutée. "
            "Vérifiez le résultat, puis commencez une nouvelle conversation avec /new."
        ),
        "commands.stop_too_late": (
            "La réponse était déjà terminée et est en cours d'envoi."
        ),
        "commands.nothing_to_stop": "Aucun travail n'est en cours.",
        "commands.reset_running": (
            "Je traite encore une demande. Envoyez /stop pour l'arrêter, "
            "puis /new pour commencer une nouvelle conversation."
        ),
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
        "runtime.failure": "تعذّر عليّ الرد الآن.",
        "runtime.incomplete_response": "تعذّر عليّ إكمال ردي. قد تكون بعض الإجراءات قد نُفّذت بالفعل.",
        "runtime.storage_failure": (
            "تعذّر عليّ المتابعة، ولا أستطيع التأكد مما إذا كان طلبك قد نُفّذ. "
            "تحقّق من النتيجة، ثم ابدأ محادثة جديدة باستخدام /new."
        ),
        "runtime.interrupted_unknown": (
            "تعذّر عليّ التأكد مما إذا كان الطلب السابق قد نُفّذ. "
            "تحقّق من النتيجة، ثم ابدأ محادثة جديدة باستخدام /new."
        ),
        "media.delivery_unavailable": "تعذّر عليّ إرسال مرفق واحد أو أكثر.",
        "document.too_large": "هذا المستند كبير جدًا لقراءته. يرجى إرسال ملف أصغر.",
        "document.unreadable": "تعذّر عليّ قراءة هذا المستند. يرجى حفظ نسخة جديدة وإرسالها مجددًا.",
        "cron.failure": "تعذّر عليّ التأكد من تنفيذ الطلب المجدول.",
        "cron.unavailable": "جدولة الرسائل التلقائية غير متاحة.",
        "capability.unavailable": "هذه الميزة غير متاحة.",
        "runtime.reset": "سنبدأ من جديد. لقد نسيت محادثتنا السابقة.",
        "runtime.working": "ما زلت أعمل.",
        "runtime.working_elapsed_under_minute": "{text} (أقل من دقيقة)",
        "runtime.working_elapsed_minutes": "{text} ({minutes} د)",
        "runtime.stopped": "توقّفت بناءً على طلبك.",
        "runtime.stopped_after_actions": (
            "توقّفت بناءً على طلبك. لم يتم التراجع عن الإجراءات المكتملة قبل الإيقاف."
        ),
        "session.auto_reset_idle": "سنبدأ محادثة جديدة بعد هذه الاستراحة.",
        "session.auto_reset_daily": "سنبدأ محادثة جديدة لهذا اليوم.",
        "approval.required": "الموافقة مطلوبة — {category}",
        "approval.default_summary": "طُلب تغيير دائم.",
        "approval.instructions": "أرسل /approve للسماح بهذه المرة، أو /deny للرفض.",
        "commands.header": "الإجراءات المتاحة:",
        "commands.ready": "أنا متاح.",
        "commands.description_help": "عرض الإجراءات المتاحة",
        "commands.description_new": "بدء محادثة جديدة",
        "commands.description_stop": "إيقاف الطلب الحالي",
        "commands.description_status": "التحقّق من التوفّر",
        "commands.alias": "اسم بديل: /{alias}",
        "commands.usage": "الاستخدام: /{command}",
        "commands.approved": "تمت الموافقة. سأتابع.",
        "commands.denied": "تم الرفض. لن يتغير شيء.",
        "commands.no_approval": "لا توجد موافقة معلّقة.",
        "commands.stopped": "تم الإيقاف.",
        "commands.stopped_after_actions": (
            "تم الإيقاف. لم يتم التراجع عن الإجراءات التي اكتملت قبل الإيقاف."
        ),
        "commands.stop_unknown": (
            "تم الإيقاف. لا أستطيع التأكد مما إذا كان الطلب قد نُفّذ. "
            "تحقّق من النتيجة، ثم ابدأ محادثة جديدة باستخدام /new."
        ),
        "commands.stop_too_late": "كانت الإجابة مكتملة بالفعل ويجري إرسالها.",
        "commands.nothing_to_stop": "لا يوجد عمل قيد التنفيذ.",
        "commands.reset_running": (
            "ما زلت أعمل على طلب. أرسل /stop لإيقافه، "
            "ثم /new لبدء محادثة جديدة."
        ),
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
