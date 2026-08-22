"""What the model can do, and how a request to do it is carried out.

A tool is a name, a schema the model reads, and a handler. Tools belong to a
group — `todo`, `web`, `file` — and the operator switches whole groups on and
off in the configuration file, per profile and per channel. Nothing is
discovered, imported dynamically or overridden at runtime: the set of tools an
agent has is decided when it starts and does not change under it.

Two rules from the Hermes registry are kept because both were paid for:

* **A handler never raises into the loop.** Every failure comes back as a JSON
  error the model can read and act on. A tool that throws would otherwise end
  the turn with nothing for the person waiting.
* **A result is bounded.** An unbounded result is replayed on every following
  request for the rest of the conversation, so one `cat` of a large file would
  cost the chat its context and the operator its money.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# A tool error body, capped. Only runaway interpolated exceptions get near it.
MAX_ERROR_CHARS = 2048
TRUNCATION_MARKER = "… [truncated]"

# What one tool may hand back, and what all the tools in one step may hand back
# between them. Hermes' numbers. Past the per-result cap the head is kept: the
# start of a command's output or a file is where the answer usually is.
DEFAULT_MAX_RESULT_CHARS = 100_000
DEFAULT_STEP_BUDGET_CHARS = 200_000


def tool_error(message: Any, **extra: Any) -> str:
    """The JSON a handler returns when it cannot do what was asked."""
    result: Dict[str, Any] = {"error": _bound(str(message))}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)


def tool_result(data: Any = None, **kwargs: Any) -> str:
    """The JSON a handler returns when it did."""
    if data is not None:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)


def _bound(text: str) -> str:
    if len(text) <= MAX_ERROR_CHARS:
        return text
    logger.debug("Tool error body truncated (%d chars): %s", len(text), text[:8192])
    return text[:MAX_ERROR_CHARS] + TRUNCATION_MARKER


@dataclass
class ToolContext:
    """Everything a handler is allowed to know about the turn it serves.

    ``state`` is the chat's own scratch space, kept for as long as the process
    runs. A tool that needs to remember something between calls of one
    conversation — the task list, an open terminal — keeps it there under its
    own key, so nothing has to be threaded through the agent for each new tool.
    """

    chat_id: str
    config: Any
    state: Dict[str, Any] = field(default_factory=dict)
    # Profile-wide curated memory. Explicit because it is shared by chats;
    # putting it in per-chat state would create divergent stores.
    memory_store: Any = None
    # Profile-wide durable scheduling state. It is explicit for the same
    # isolation reason as memory; no tool discovers another profile's root.
    cron_store: Any = None
    # The delivery location that created a job, never a model-selected target.
    origin: Optional[Dict[str, str]] = None
    cron_wake: Optional[Callable[[], None]] = None


Handler = Callable[..., Any]


@dataclass(frozen=True)
class Tool:
    name: str
    group: str
    # OpenAI function schema: name, description, parameters.
    schema: Dict[str, Any]
    handler: Handler
    # An icon for logs and for the line the person waiting sees.
    emoji: str = "⚡"
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS


class Registry:
    """The tools this build of the agent has."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"The tool {tool.name!r} is registered twice.")
        self._tools[tool.name] = tool

    def groups(self) -> List[str]:
        return sorted({tool.group for tool in self._tools.values()})

    def names(self, groups: Sequence[str] = ()) -> List[str]:
        allowed = set(groups)
        return sorted(
            tool.name for tool in self._tools.values() if not allowed or tool.group in allowed
        )

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def emoji(self, name: str, default: str = "⚡") -> str:
        tool = self._tools.get(name)
        return tool.emoji if tool else default

    def definitions(self, groups: Sequence[str]) -> List[Dict[str, Any]]:
        """The enabled tools, in the shape the Responses API asks for.

        Sorted by name so the list — and therefore the cached request prefix —
        does not depend on the order the modules happened to be imported in.
        """
        allowed = set(groups)
        definitions: List[Dict[str, Any]] = []
        for name in sorted(self._tools):
            tool = self._tools[name]
            if tool.group not in allowed:
                continue
            schema = tool.schema
            definitions.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": schema.get("description", ""),
                    # Strict mode requires every property to be required and
                    # additionalProperties false; our schemas have optional
                    # arguments on purpose, so it stays off — as in Hermes.
                    "strict": False,
                    "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return definitions

    async def dispatch(
        self,
        name: str,
        arguments: str,
        context: ToolContext,
        *,
        allowed_groups: Optional[Sequence[str]] = None,
        max_result_chars: Optional[int] = None,
    ) -> str:
        """Run one tool call and return what the model should read.

        Never raises. A bad name, unreadable arguments and a handler that
        throws all come back as an error the model can recover from.
        """
        tool = self._tools.get(name)
        if tool is None:
            return tool_error(f"Unknown tool: {name}")
        # The schema sent to the model is not an authorization boundary. A
        # stale, replayed or fabricated call can still name any registered
        # tool, so enforce the agent's allowlist again at execution time.
        if allowed_groups is not None and tool.group not in set(allowed_groups):
            return tool_error(f"Tool is disabled: {name}", tool=name)

        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except ValueError as exc:
            return tool_error(f"The arguments were not valid JSON: {exc}", tool=name)
        if not isinstance(args, dict):
            return tool_error(f"The arguments must be an object, not {type(args).__name__}.", tool=name)

        try:
            result = tool.handler(args, context)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001 - the model is told, the turn goes on
            logger.exception("The tool %s failed", name)
            return tool_error(f"Tool execution failed: {type(exc).__name__}: {exc}", tool=name)

        if not isinstance(result, str):
            logger.error("The tool %s returned %s, not text", name, type(result).__name__)
            return tool_error(
                f"Tool handler returned unsupported result type: {type(result).__name__}",
                tool=name,
            )
        limit = tool.max_result_chars if max_result_chars is None else max_result_chars
        return cap_result(result, limit)


def cap_result(result: str, limit: int) -> str:
    """Trim a result to what a conversation can afford to carry."""
    if limit <= 0 or len(result) <= limit:
        return result

    # Tool handlers normally return JSON. Raw slicing would leave an invalid
    # fragment exactly when the model most needs a readable recovery path.
    try:
        json.loads(result)
    except (TypeError, ValueError):
        pass
    else:
        if limit <= 1:
            return "0"
        if limit == 2:
            return "{}"
        envelope: Dict[str, Any] = {
            "truncated": True,
            "original_chars": len(result),
            "prefix": "",
        }
        base = json.dumps(envelope, ensure_ascii=False)
        if len(base) > limit:
            return "{}"
        low, high = 0, len(result)
        while low < high:
            middle = (low + high + 1) // 2
            envelope["prefix"] = result[:middle]
            if len(json.dumps(envelope, ensure_ascii=False)) <= limit:
                low = middle
            else:
                high = middle - 1
        envelope["prefix"] = result[:low]
        return json.dumps(envelope, ensure_ascii=False)

    dropped = len(result) - limit
    return result[:limit] + f"\n{TRUNCATION_MARKER} {dropped} characters cut."


async def run_calls(
    registry: Registry,
    calls: Sequence[Dict[str, Any]],
    context: ToolContext,
    *,
    allowed_groups: Optional[Sequence[str]] = None,
    max_result_chars: Optional[int] = None,
    step_budget_chars: int = DEFAULT_STEP_BUDGET_CHARS,
) -> List[str]:
    """Run the calls the model asked for in one step, and return their results.

    They run at the same time — the model asks for several precisely when they
    do not depend on each other — but the results come back in the order they
    were asked for, so the request built from them is the same every time.
    """
    if not calls:
        return []
    results = await asyncio.gather(
        *(
            registry.dispatch(
                str(call.get("name") or ""),
                str(call.get("arguments") or "{}"),
                context,
                allowed_groups=allowed_groups,
                max_result_chars=max_result_chars,
            )
            for call in calls
        )
    )

    # One step's results together are also bounded: eight tools each just
    # inside the per-result cap would otherwise still blow the context apart.
    if step_budget_chars <= 0:
        return list(results)
    remaining = step_budget_chars
    bounded: List[str] = []
    for result in results:
        if remaining <= 0:
            bounded.append(tool_error("The result was dropped: this step's output budget is spent."))
            continue
        bounded.append(cap_result(result, remaining))
        remaining -= len(bounded[-1])
    return bounded
