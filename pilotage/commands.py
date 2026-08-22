"""Small management-command registry.

The definition/alias lookup is the proven Hermes CommandDef pattern, reduced to
commands this Genesis runtime can actually execute. A slash command is handled
outside the model loop, so operating the agent never depends on the model being
available or behaving correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import __version__


@dataclass(frozen=True)
class CommandDef:
    """Definition of one command, without its leading slash."""

    name: str
    description: str
    category: str
    aliases: tuple[str, ...] = ()


COMMAND_REGISTRY: tuple[CommandDef, ...] = (
    CommandDef("help", "Show the available management commands", "Info", aliases=("commands",)),
    CommandDef("new", "Start a fresh conversation", "Session", aliases=("reset",)),
    CommandDef("status", "Show the running agent's essential status", "Info"),
    CommandDef("profile", "Show the active profile and state directory", "Info"),
)


def _build_command_lookup() -> dict[str, CommandDef]:
    lookup: dict[str, CommandDef] = {}
    for command in COMMAND_REGISTRY:
        lookup[command.name] = command
        for alias in command.aliases:
            lookup[alias] = command
    return lookup


_COMMAND_LOOKUP = _build_command_lookup()


def resolve_command(name: str) -> Optional[CommandDef]:
    """Resolve a canonical name or alias, with or without its slash."""

    return _COMMAND_LOOKUP.get(str(name).lower().lstrip("/"))


@dataclass(frozen=True)
class CommandInvocation:
    command: CommandDef
    arguments: str = ""


def parse_command(text: str) -> Optional[CommandInvocation]:
    """Parse a whole-message slash command; ordinary prose remains model input."""

    written = str(text or "").strip()
    if not written.startswith("/"):
        return None
    parts = written.split(maxsplit=1)
    command = resolve_command(parts[0])
    if command is None:
        return None
    arguments = parts[1].strip() if len(parts) == 2 else ""
    return CommandInvocation(command, arguments)


def help_text() -> str:
    lines = ["Management commands:"]
    for command in COMMAND_REGISTRY:
        aliases = "".join(f" (alias: /{alias})" for alias in command.aliases)
        lines.append(f"/{command.name} — {command.description}{aliases}")
    return "\n".join(lines)


def configured_tool_names(config: Any) -> tuple[str, ...]:
    """Return exactly the tools enabled by this profile/channel view."""

    from .tools import build_registry, enabled_groups

    registry = build_registry()
    groups = enabled_groups(config.settings, registry)
    return tuple(registry.names(groups))


def _auth_scope(config: Any) -> str:
    primary = Path(config.credentials_path)
    fallback = Path(config.main_credentials_path)
    if primary.exists():
        return "this profile"
    try:
        distinct = primary.resolve(strict=False) != fallback.resolve(strict=False)
    except OSError:
        distinct = primary != fallback
    if distinct and fallback.exists():
        return "shared from default profile"
    return "not signed in"


def profile_text(config: Any, profile_name: str) -> str:
    return (
        f"Profile: {profile_name}\n"
        f"State: {config.state_dir}\n"
        f"ChatGPT auth: {_auth_scope(config)}"
    )


def status_text(config: Any, profile_name: str) -> str:
    tools = configured_tool_names(config)
    channel = str(getattr(config.settings, "channel", "") or "local")
    cron_state = "enabled" if getattr(config, "cron_enabled", True) else "disabled"
    cron_timezone = str(getattr(config, "cron_timezone", "") or "system local")
    return (
        f"Pilotage {__version__}\n"
        f"Profile: {profile_name}\n"
        f"Model: {config.model}\n"
        f"Channel: {channel}\n"
        f"Tools: {', '.join(tools) if tools else 'none'}\n"
        f"Cron: {cron_state} ({cron_timezone})\n"
        f"ChatGPT auth: {_auth_scope(config)}"
    )


async def execute_command(
    invocation: CommandInvocation,
    *,
    agent: Any,
    config: Any,
    profile_name: str,
    session_id: str,
    reset_reply: str,
) -> str:
    """Execute one recognized command and return its channel-neutral text."""

    name = invocation.command.name
    if invocation.arguments:
        return f"Usage: /{name}"
    if name == "help":
        return help_text()
    if name == "new":
        await agent.forget(session_id)
        return reset_reply
    if name == "status":
        return status_text(config, profile_name)
    if name == "profile":
        return profile_text(config, profile_name)
    return f"Unknown command: /{name}"

