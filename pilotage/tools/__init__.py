"""The tools this build of the agent has, and the ones a given chat may use.

Every tool the runtime knows about is registered here, once, at import. What
an agent may actually do is decided separately, by the configuration file:

    tools:
      enabled: [todo, web, file]     # absent means every group
      disabled: [terminal]           # always wins

    channels:
      whatsapp:
        tools:
          disabled: [terminal, code_execution]

`disabled` beating `enabled` is on purpose. Switching something off is the
safety-side edit, so it must not be possible to lose that edit by adding a
group somewhere else in the file.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from .registry import (
    Registry,
    Tool,
    ToolContext,
    cap_result,
    run_calls,
    tool_error,
    tool_result,
)
from .files import FILE_TOOLS
from .terminal import TERMINAL_TOOL
from .todo import TODO_TOOL

__all__ = [
    "Registry",
    "Tool",
    "ToolContext",
    "build_registry",
    "cap_result",
    "enabled_groups",
    "run_calls",
    "tool_error",
    "tool_result",
]

# Every tool the runtime has. Slices add to this list; nothing else changes.
ALL_TOOLS: Sequence[Tool] = (*FILE_TOOLS, TERMINAL_TOOL, TODO_TOOL)


def build_registry() -> Registry:
    registry = Registry()
    for tool in ALL_TOOLS:
        registry.register(tool)
    return registry


def enabled_groups(settings: Any, registry: Registry) -> List[str]:
    """Which tool groups this agent may use, after the file has had its say."""
    known = registry.groups()
    enabled = settings.names("tools.enabled", known)
    disabled = set(settings.names("tools.disabled"))
    return [group for group in known if group in set(enabled) and group not in disabled]
