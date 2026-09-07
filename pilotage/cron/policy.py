"""The configuration check shared by scheduling requests and execution."""

from __future__ import annotations

from typing import Any


def scheduling_enabled(config: Any, origin: Any) -> bool:
    channel = ""
    if origin is not None:
        if not isinstance(origin, dict):
            return False
        channel = str(origin.get("channel") or "").strip().lower()
        if channel not in {"whatsapp", "telegram"} or not origin.get("chat_id"):
            return False
    settings = config.settings.for_channel(channel)
    if channel and not settings.flag(f"{channel}.enabled", False):
        return False
    return settings.flag("cron.enabled", getattr(config, "cron_enabled", True))
