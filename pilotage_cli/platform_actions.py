"""Capability-gated platform action facade for plugins (, action half).

``ctx.platform_actions`` gives a plugin a *minimal*, versioned verb set for
acting on connected chat platforms through the live gateway adapter registry —
no adapter handles, bot clients, or raw SDK objects are ever exposed.

Gating (fail closed, default OFF)
---------------------------------
Every verb checks ``plugin_capability_granted(plugin_id,
"gateway.platform_actions")`` at call time. The capability maps to the
``plugins.entries.<id>.allow_platform_actions`` legacy key and the
consent registry (``granted_capabilities``). No grant → structured
``capability_not_granted`` error, never an exception.

v1 verb set
-----------
* ``add_reaction(platform, chat_id, message_id, emoji)``
* ``set_thread_title(platform, chat_id, thread_id, title)``

Both return a structured result dict — ``{"ok": True, ...}`` on success,
``{"ok": False, "error": <code>, "detail": <str>}`` on failure — and never
raise into hook dispatch. Error codes are part of the v1 contract:
``capability_not_granted``, ``invalid_argument``, ``gateway_unavailable``,
``unknown_platform``, ``adapter_not_registered``, ``adapter_disconnected``,
``unsupported_platform_action``, ``action_failed``.

Raw SDK payload/handle access is deliberately NOT part of this surface; per
the round-2 correction it requires its own capability
(``gateway.raw_events``,) and design.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ACTIONS_CONTRACT_VERSION = 1

CAPABILITY_ID = "gateway.platform_actions"


def _err(code: str, detail: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "error": code}
    if detail:
        result["detail"] = detail
    return result


def _ok(**fields: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": True}
    result.update(fields)
    return result


class PlatformActions:
    """Per-plugin facade over the live gateway adapter registry.

    Instances are cheap and hold only the owning plugin id; the gateway
    runner and adapters are resolved at call time so a facade created
    before the gateway starts (plugin ``register()`` runs first) still
    works once adapters connect.
    """

    def __init__(self, plugin_id: str):
        self._plugin_id = plugin_id

    # -- shared plumbing ----------------------------------------------------

    def _capability_granted(self) -> bool:
        try:
            from pilotage_cli.plugin_capabilities import plugin_capability_granted

            return plugin_capability_granted(self._plugin_id, CAPABILITY_ID)
        except Exception:
            # Ground rule: failure to read consent state = not granted.
            logger.debug(
                "platform_actions capability check failed for %s",
                self._plugin_id, exc_info=True,
            )
            return False

    def _resolve_adapter(self, platform: str):
        """Return ``(adapter, error_dict)``; exactly one is non-None."""
        try:
            from gateway.run import _gateway_runner_ref

            runner = _gateway_runner_ref()
        except Exception:
            runner = None
        if runner is None:
            return None, _err(
                "gateway_unavailable", "no gateway runner is active in this process"
            )
        try:
            from gateway.config import Platform

            platform_enum = Platform(str(platform).strip().lower())
        except Exception:
            return None, _err("unknown_platform", f"unknown platform {platform!r}")
        adapter = getattr(runner, "adapters", {}).get(platform_enum)
        if adapter is None:
            return None, _err(
                "adapter_not_registered",
                f"no {platform_enum.value} adapter is registered",
            )
        try:
            connected = bool(adapter.is_connected)
        except Exception:
            connected = False
        if not connected:
            return None, _err(
                "adapter_disconnected",
                f"the {platform_enum.value} adapter is not connected",
            )
        return adapter, None

    def _gate(self, platform: str, **required: Any):
        """Run the shared gate chain. Returns ``(adapter, error_dict)``."""
        if not self._capability_granted():
            return None, _err(
                "capability_not_granted",
                f"plugin {self._plugin_id!r} lacks the {CAPABILITY_ID!r} "
                "capability (grant via consent flow or "
                f"plugins.entries.{self._plugin_id}.allow_platform_actions)",
            )
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                return None, _err(
                    "invalid_argument", f"{name} must be a non-empty string"
                )
        return self._resolve_adapter(platform)

    # -- v1 verbs -----------------------------------------------------------

    async def add_reaction(
        self, platform: str, chat_id: str, message_id: str, emoji: str
    ) -> Dict[str, Any]:
        """Add/set an emoji reaction on a platform message.

        Telegram note: the Bot API *sets* the bot's reaction (replacing a
        previous one) rather than stacking, per ``set_message_reaction``.
        """
        adapter, error = self._gate(
            platform, chat_id=chat_id, message_id=message_id, emoji=emoji
        )
        if error is not None or adapter is None:
            self._audit("add_reaction", platform, error or _err("gateway_unavailable"))
            return error or _err("gateway_unavailable")
        try:
            if getattr(adapter.platform, "value", None) == "telegram":
                done = await adapter._set_reaction(chat_id, message_id, emoji)
                result = (
                    _ok(action="add_reaction")
                    if done
                    else _err("action_failed", "telegram set_message_reaction failed")
                )
            else:
                result = _err(
                    "unsupported_platform_action",
                    f"add_reaction is not implemented for {platform}",
                )
        except Exception as exc:
            result = _err("action_failed", str(exc)[:512])
        self._audit("add_reaction", platform, result)
        return result

    async def set_thread_title(
        self, platform: str, chat_id: str, thread_id: str, title: str
    ) -> Dict[str, Any]:
        """Rename a thread / forum topic.

        Telegram requires ``chat_id`` (``edit_forum_topic`` is chat-scoped).
        """
        adapter, error = self._gate(
            platform, chat_id=chat_id, thread_id=thread_id, title=title
        )
        if error is not None or adapter is None:
            self._audit("set_thread_title", platform, error or _err("gateway_unavailable"))
            return error or _err("gateway_unavailable")
        try:
            if getattr(adapter.platform, "value", None) == "telegram":
                await adapter.rename_dm_topic(chat_id, int(thread_id), title)
                result = _ok(action="set_thread_title")
            else:
                result = _err(
                    "unsupported_platform_action",
                    f"set_thread_title is not implemented for {platform}",
                )
        except Exception as exc:
            result = _err("action_failed", str(exc)[:512])
        self._audit("set_thread_title", platform, result)
        return result

    def _audit(self, verb: str, platform: str, result: Dict[str, Any]) -> None:
        """Every platform action is logged (the 'all actions logged' rule)."""
        logger.info(
            "platform_action plugin=%s verb=%s platform=%s ok=%s%s",
            self._plugin_id,
            verb,
            platform,
            result.get("ok"),
            "" if result.get("ok") else f" error={result.get('error')}",
        )
