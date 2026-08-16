"""
Channel directory -- cached map of reachable channels/contacts per platform.

Built on gateway startup, refreshed periodically (every 5 min), and saved to
~/.pilotage/channel_directory.json.  The send_message tool reads this file for
action="list" and for resolving human-friendly channel names to numeric IDs.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pilotage_cli.config import get_pilotage_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

DIRECTORY_PATH = get_pilotage_home() / "channel_directory.json"
# User-maintained friendly-name overlay. The directory is fully regenerated
# from live adapters + session data on a timer, so hand-edits to
# channel_directory.json don't survive. Aliases declared here are re-applied
# on every build AND every load, giving durable human-friendly names (and
# letting you pre-name a chat before it has produced any traffic).
# Format: {"<platform>": {"<chat_id>": "<friendly name>", ...}, ...}
CHANNEL_ALIASES_PATH = get_pilotage_home() / "channel_aliases.json"


def _load_channel_aliases() -> Dict[str, Dict[str, str]]:
    if not CHANNEL_ALIASES_PATH.exists():
        return {}
    try:
        with open(CHANNEL_ALIASES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _apply_channel_aliases(platforms: Dict[str, Any]) -> None:
    """Overlay friendly names onto directory entries by chat_id.

    Renames matching entries in place; injects a placeholder entry for an
    aliased id that hasn't been discovered yet (so a freshly-created group is
    addressable by name before its first message). Mutates *platforms*.
    """
    aliases = _load_channel_aliases()
    for plat_name, id_map in aliases.items():
        if not isinstance(id_map, dict):
            continue
        entries = platforms.setdefault(plat_name, [])
        if not isinstance(entries, list):
            continue
        for chat_id, friendly in id_map.items():
            if not isinstance(friendly, str) or not friendly.strip():
                continue
            chat_id = str(chat_id)
            friendly = friendly.strip()
            matched = False
            for e in entries:
                if isinstance(e, dict) and e.get("id") == chat_id:
                    e["name"] = friendly
                    matched = True
            if not matched:
                entries.append({
                    "id": chat_id,
                    "name": friendly,
                    "type": "group" if str(chat_id).endswith("@g.us") else "dm",
                    "thread_id": None,
                })


def _normalize_channel_query(value: str) -> str:
    return value.lstrip("#").strip().lower()


def _channel_target_name(platform_name: str, channel: Dict[str, Any]) -> str:
    """Return the human-facing target label shown to users for a channel entry."""
    name = channel["name"]
    if channel.get("type"):
        return f"{name} ({channel['type']})"
    return name


def _session_entry_id(origin: Dict[str, Any]) -> Optional[str]:
    chat_id = origin.get("chat_id")
    if not chat_id:
        return None
    thread_id = origin.get("thread_id")
    if thread_id:
        return f"{chat_id}:{thread_id}"
    return str(chat_id)


def _session_entry_name(origin: Dict[str, Any]) -> str:
    base_name = origin.get("chat_name") or origin.get("user_name") or str(origin.get("chat_id"))
    thread_id = origin.get("thread_id")
    if not thread_id:
        return base_name

    topic_label = origin.get("chat_topic") or f"topic {thread_id}"
    return f"{base_name} / {topic_label}"


async def build_channel_directory(adapters: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Build a channel directory from connected platform adapters and session data.

    Returns the directory dict and writes it to DIRECTORY_PATH.
    """
    from gateway.config import Platform

    platforms: Dict[str, List[Dict[str, str]]] = {}

    for platform, adapter in adapters.items():
        try:
            list_channels = getattr(adapter, "list_channels", None)
            if callable(list_channels):
                platform_channels = await list_channels()
                if platform_channels is not None:
                    platforms[platform.value] = _normalize_adapter_channels(platform_channels)
                    continue
        except Exception as e:
            logger.warning("Channel directory: failed to build %s: %s", platform.value, e)

    # Platforms that don't support direct channel enumeration get session-based
    # discovery automatically, but only for platforms connected in THIS gateway
    # process. Historical session origins for disabled/decommissioned platforms
    # must not be resurrected into the active send-target directory (stale
    # targets make send_message route to platforms that can no longer deliver).
    _SKIP_SESSION_DISCOVERY = frozenset({"local", "api_server", "webhook"})
    adapter_platform_names = {getattr(p, "value", str(p)) for p in adapters}
    for plat in Platform:
        plat_name = plat.value
        if (
            plat_name in _SKIP_SESSION_DISCOVERY
            or plat_name in platforms
            or plat_name not in adapter_platform_names
        ):
            continue
        platforms[plat_name] = await asyncio.to_thread(_build_from_sessions, plat_name)

    # Include plugin-registered platforms (dynamic enum members aren't in
    # Platform.__members__, so the loop above misses them). Same
    # connected-only rule: don't expose stale session targets for plugins
    # that are not loaded.
    try:
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            if (
                entry.name not in _SKIP_SESSION_DISCOVERY
                and entry.name not in platforms
                and entry.name in adapter_platform_names
            ):
                platforms[entry.name] = await asyncio.to_thread(_build_from_sessions, entry.name)
    except Exception:
        pass

    # Overlay user-maintained friendly names before persisting.
    _apply_channel_aliases(platforms)

    directory = {
        "updated_at": datetime.now().isoformat(),
        "platforms": platforms,
    }

    try:
        await asyncio.to_thread(atomic_json_write, DIRECTORY_PATH, directory)
    except Exception as e:
        logger.warning("Channel directory: failed to write: %s", e)

    return directory


def _normalize_adapter_channels(raw_channels: Any) -> List[Dict[str, Any]]:
    """Validate and dedupe channel entries returned by an adapter's
    ``list_channels()`` hook (see ``build_channel_directory``)."""
    channels: List[Dict[str, Any]] = []
    seen_ids = set()
    if not isinstance(raw_channels, list):
        return channels
    for raw in raw_channels:
        if not isinstance(raw, dict):
            continue
        channel_id = str(raw.get("id") or "").strip()
        name = str(raw.get("name") or channel_id).strip()
        if not channel_id or not name or channel_id in seen_ids:
            continue
        entry: Dict[str, Any] = {
            "id": channel_id,
            "name": name,
            "type": str(raw.get("type") or "dm"),
        }
        if raw.get("thread_id"):
            entry["thread_id"] = str(raw.get("thread_id"))
        if raw.get("guild"):
            entry["guild"] = str(raw.get("guild"))
        channels.append(entry)
        seen_ids.add(channel_id)
    return channels


def _build_from_sessions(platform_name: str) -> List[Dict[str, str]]:
    """Pull known channels/contacts from gateway session origin data.

    state.db is the primary source: gateway session rows persist
    origin_json.  Falls back to sessions.json for pre-migration databases.
    """
    entries = _build_from_sessions_db(platform_name)
    if entries:
        return entries
    return _build_from_sessions_json(platform_name)


def _build_from_sessions_db(platform_name: str) -> List[Dict[str, str]]:
    """Pull channels/contacts from state.db gateway session rows."""
    entries: List[Dict[str, str]] = []
    try:
        from pilotage_state import SessionDB
        db = SessionDB()
        try:
            lister = getattr(db, "list_gateway_sessions", None)
            if not callable(lister):
                return []
            rows = lister(platform=platform_name, active_only=False)
        finally:
            db.close()

        seen_ids = set()
        for row in rows:
            origin: Dict[str, Any] = {}
            if row.get("origin_json"):
                try:
                    parsed = json.loads(row["origin_json"])
                    if isinstance(parsed, dict):
                        origin = parsed
                except (TypeError, ValueError):
                    pass
            if not origin:
                origin = {
                    "chat_id": row.get("chat_id"),
                    "thread_id": row.get("thread_id"),
                    "chat_name": row.get("display_name"),
                }
            entry_id = _session_entry_id(origin)
            if not entry_id or entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            entries.append({
                "id": entry_id,
                "name": _session_entry_name(origin),
                "type": row.get("chat_type") or "dm",
                "thread_id": origin.get("thread_id"),
            })
    except Exception as e:
        logger.debug(
            "Channel directory: state.db session read failed for %s: %s",
            platform_name, e,
        )
    return entries


def _build_from_sessions_json(platform_name: str) -> List[Dict[str, str]]:
    """Legacy fallback: pull channels/contacts from sessions.json origin data."""
    sessions_path = get_pilotage_home() / "sessions" / "sessions.json"
    if not sessions_path.exists():
        return []

    entries = []
    try:
        with open(sessions_path, encoding="utf-8") as f:
            data = json.load(f)

        seen_ids = set()
        for _key, session in data.items():
            # Skip documentation/metadata sentinels (keys starting with "_",
            # e.g. the gateway's "_README" note) — not session entries.
            if str(_key).startswith("_") or not isinstance(session, dict):
                continue
            origin = session.get("origin") or {}
            if origin.get("platform") != platform_name:
                continue
            entry_id = _session_entry_id(origin)
            if not entry_id or entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            entries.append({
                "id": entry_id,
                "name": _session_entry_name(origin),
                "type": session.get("chat_type", "dm"),
                "thread_id": origin.get("thread_id"),
            })
    except Exception as e:
        logger.debug("Channel directory: failed to read sessions for %s: %s", platform_name, e)

    return entries


# ---------------------------------------------------------------------------
# Read / resolve
# ---------------------------------------------------------------------------

def load_directory() -> Dict[str, Any]:
    """Load the cached channel directory from disk."""
    if not DIRECTORY_PATH.exists():
        base = {"updated_at": None, "platforms": {}}
        _apply_channel_aliases(base["platforms"])
        return base
    try:
        with open(DIRECTORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # Re-apply aliases on read so friendly names take effect immediately,
        # even between timed rebuilds and for brand-new alias entries.
        _apply_channel_aliases(data.setdefault("platforms", {}))
        return data
    except Exception:
        base = {"updated_at": None, "platforms": {}}
        _apply_channel_aliases(base["platforms"])
        return base


def lookup_channel_type(platform_name: str, chat_id: str) -> Optional[str]:
    """Return the channel ``type`` string (e.g. ``"channel"``, ``"forum"``) for *chat_id*, or *None* if unknown."""
    directory = load_directory()
    for ch in directory.get("platforms", {}).get(platform_name, []):
        if ch.get("id") == chat_id:
            return ch.get("type")
    return None


def resolve_channel_name(platform_name: str, name: str) -> Optional[str]:
    """
    Resolve a human-friendly channel name to a numeric ID.

    Matching strategy (case-insensitive, first match wins):
    - Telegram: display name or group name
    - WhatsApp: contact or group name
    """
    directory = load_directory()
    channels = directory.get("platforms", {}).get(platform_name, [])
    if not channels:
        return None

    # 0. Exact ID match — case-sensitive, no normalization. Lets callers pass
    # raw platform IDs even when the format guard in _parse_target_ref hasn't
    # recognized them as explicit.
    raw = name.strip()
    for ch in channels:
        if ch.get("id") == raw:
            return ch["id"]

    query = _normalize_channel_query(name)

    # 1. Exact name match, including the display labels shown by send_message(action="list")
    for ch in channels:
        if _normalize_channel_query(ch["name"]) == query:
            return ch["id"]
        if _normalize_channel_query(_channel_target_name(platform_name, ch)) == query:
            return ch["id"]

    # 3. Partial prefix match (only if unambiguous)
    matches = [ch for ch in channels if _normalize_channel_query(ch["name"]).startswith(query)]
    if len(matches) == 1:
        return matches[0]["id"]

    return None


def format_directory_for_display(platforms: Optional[Dict[str, Any]] = None) -> str:
    """Format the channel directory as a human-readable list for the model.

    ``platforms`` overrides the on-disk directory when provided (used by
    ``pilotage send --list`` to merge in configured-but-undiscovered
    platforms). Platforms present with an empty channel list are rendered
    with a "(no channels discovered yet)" hint instead of being hidden —
    a configured platform is a valid send target even before discovery.
    """
    if platforms is None:
        directory = load_directory()
        platforms = directory.get("platforms", {})

    if not platforms:
        return "No messaging platforms connected or no channels discovered yet."

    lines = ["Available messaging targets:\n"]

    for plat_name, channels in sorted(platforms.items()):
        if not channels:
            lines.append(f"{plat_name.title()}:")
            lines.append(
                f"  (no channels discovered yet — send directly with "
                f"{plat_name}:<chat_id>, or bare '{plat_name}' for the home channel)"
            )
            lines.append("")
            continue

        lines.append(f"{plat_name.title()}:")
        for ch in channels:
            lines.append(f"  {plat_name}:{_channel_target_name(plat_name, ch)}")
        lines.append("")

    lines.append('Use these as the "target" parameter when sending.')
    lines.append('Bare platform name (e.g. "telegram") sends to home channel.')

    return "\n".join(lines)
