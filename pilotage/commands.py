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
from .i18n import DEFAULT_LANGUAGE, t


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
    CommandDef("approve", "Allow the oldest pending change once", "Approval"),
    CommandDef("deny", "Refuse the oldest pending change", "Approval"),
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


def help_text(language: str = DEFAULT_LANGUAGE) -> str:
    lines = [t("commands.header", language)]
    for command in COMMAND_REGISTRY:
        aliases = "".join(
            f" ({t('commands.alias', language, alias=alias)})"
            for alias in command.aliases
        )
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
        return "profile"
    try:
        distinct = primary.resolve(strict=False) != fallback.resolve(strict=False)
    except OSError:
        distinct = primary != fallback
    if distinct and fallback.exists():
        return "shared"
    return "missing"


def profile_text(config: Any, profile_name: str) -> str:
    language = str(getattr(config, "language", DEFAULT_LANGUAGE))
    auth_scope = t(f"commands.auth_{_auth_scope(config)}", language)
    return (
        f"{t('commands.profile', language, profile=profile_name)}\n"
        f"{t('commands.state', language, state=config.state_dir)}\n"
        f"{t('commands.auth', language, scope=auth_scope)}"
    )


def status_text(config: Any, profile_name: str) -> str:
    language = str(getattr(config, "language", DEFAULT_LANGUAGE))
    tools = configured_tool_names(config)
    channel = str(getattr(config.settings, "channel", "") or "local")
    cron_state = t(
        "commands.enabled"
        if getattr(config, "cron_enabled", True)
        else "commands.disabled",
        language,
    )
    cron_timezone = str(
        getattr(config, "cron_timezone", "")
        or t("commands.system_local", language)
    )
    tool_text = ", ".join(tools) if tools else t("commands.none", language)
    auth_scope = t(f"commands.auth_{_auth_scope(config)}", language)
    return (
        f"Pilotage {__version__}\n"
        f"{t('commands.profile', language, profile=profile_name)}\n"
        f"{t('commands.model', language, model=config.model)}\n"
        f"{t('commands.channel', language, channel=channel)}\n"
        f"{t('commands.tools', language, tools=tool_text)}\n"
        f"{t('commands.cron', language, state=cron_state, timezone=cron_timezone)}\n"
        f"{t('commands.auth', language, scope=auth_scope)}"
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
    language = str(getattr(config, "language", DEFAULT_LANGUAGE))
    if name == "approve":
        if invocation.arguments:
            return t("commands.usage", language, command="approve")
        resolved = agent.resolve_approval(session_id, approved=True)
        return (
            t("commands.approved", language)
            if resolved
            else t("commands.no_approval", language)
        )
    if name == "deny":
        resolved = agent.resolve_approval(
            session_id,
            approved=False,
            reason=invocation.arguments,
        )
        return (
            t("commands.denied", language)
            if resolved
            else t("commands.no_approval", language)
        )
    if invocation.arguments:
        return t("commands.usage", language, command=name)
    if name == "help":
        return help_text(language)
    if name == "new":
        await agent.forget(session_id)
        return reset_reply
    if name == "status":
        return status_text(config, profile_name)
    if name == "profile":
        return profile_text(config, profile_name)
    return t("commands.unknown", language, command=name)

