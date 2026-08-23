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

from ..settings import ConfigError

from .registry import (
    Registry,
    Tool,
    ToolContext,
    cap_result,
    is_multimodal_tool_result,
    multimodal_text_summary,
    responses_tool_output,
    run_calls,
    tool_error,
    tool_result,
)
from .cron import CRONJOB_TOOL
from .files import FILE_TOOLS
from .image import IMAGE_GENERATE_TOOL
from .memory import MEMORY_TOOL
from .session_search import SESSION_SEARCH_TOOL
from .skills import SKILLS_TOOLS, build_skills_prompt
from .terminal import TERMINAL_TOOL
from .todo import TODO_TOOL
from .web import WEB_SEARCH_TOOL
from .vision import VISION_ANALYZE_TOOL

__all__ = [
    "Registry",
    "Tool",
    "ToolContext",
    "build_registry",
    "build_skills_prompt",
    "cap_result",
    "enabled_groups",
    "is_multimodal_tool_result",
    "multimodal_text_summary",
    "responses_tool_output",
    "run_calls",
    "tool_error",
    "tool_result",
]

# Every tool the runtime has. Slices add to this list; nothing else changes.
ALL_TOOLS: Sequence[Tool] = (
    CRONJOB_TOOL,
    *FILE_TOOLS,
    IMAGE_GENERATE_TOOL,
    *SKILLS_TOOLS,
    MEMORY_TOOL,
    SESSION_SEARCH_TOOL,
    TERMINAL_TOOL,
    TODO_TOOL,
    WEB_SEARCH_TOOL,
    VISION_ANALYZE_TOOL,
)


def build_registry() -> Registry:
    registry = Registry()
    for tool in ALL_TOOLS:
        registry.register(tool)
    return registry


def enabled_groups(settings: Any, registry: Registry) -> List[str]:
    """Which tool groups this agent may use, after the file has had its say."""
    known = registry.groups()
    known_set = set(known)
    enabled = settings.names("tools.enabled", known)
    disabled = settings.names("tools.disabled")
    for setting_name, written in (
        ("tools.enabled", enabled),
        ("tools.disabled", disabled),
    ):
        unknown = sorted(set(written) - known_set)
        if unknown:
            raise ConfigError(
                f"{setting_name} contains unknown tool groups: {', '.join(unknown)}"
            )
    disabled_set = set(disabled)
    return [group for group in known if group in set(enabled) and group not in disabled_set]
