"""The terminal tool the model sees.

``shell.py`` owns the difficult process mechanics.  This module is the small
tool boundary around it: one persistent shell per chat, a stable schema, input
validation, and the JSON result the Responses loop hands back to the model.

The production agent uses a persistent Linux terminal rooted at its workspace.
We keep only that contract here.  Hermes' remote backends, PTY, approvals,
background-process manager and sudo prompting solve requirements this runtime
does not have.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ..redact import redact_sensitive_text, redact_terminal_output
from .registry import Tool, ToolContext, tool_error
from .shell import DEFAULT_TIMEOUT_SECONDS, Shell

STATE_KEY = "terminal"


@dataclass
class TerminalSession:
    """One chat's shell and the lock that keeps its state ordered."""

    shell: Optional[Shell] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def get_terminal_session(context: ToolContext) -> TerminalSession:
    """Return the chat's shared terminal state.

    File tools use the same lock while resolving the terminal cwd, so a
    parallel ``cd`` cannot move the base directory halfway through a file
    operation.
    """
    session = context.state.get(STATE_KEY)
    if not isinstance(session, TerminalSession):
        session = TerminalSession()
        context.state[STATE_KEY] = session
    return session


def _setting(context: ToolContext, name: str, default: Any) -> Any:
    settings = getattr(context.config, "settings", None)
    if settings is None:
        return default
    if isinstance(default, int):
        return settings.count(name, default)
    return settings.text(name, default)


def shell_cwd(context: ToolContext) -> str:
    """Resolve the terminal root without letting profiles share process cwd."""
    scoped = getattr(context, "working_directory", None)
    if scoped is not None:
        return str(Path(scoped).expanduser())
    configured = str(_setting(context, "terminal.cwd", "")).strip()
    if configured:
        return os.path.expanduser(configured)
    state_dir = getattr(context.config, "state_dir", None)
    if state_dir is None:
        return ""
    workspace = Path(state_dir) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return str(workspace)


def shell_env(context: ToolContext) -> Dict[str, str]:
    """Expose the active profile/session to reused Hermes skill scripts."""
    env = {"HERMES_SESSION_ID": str(context.chat_id)}
    state_dir = getattr(context.config, "state_dir", None)
    if state_dir is not None:
        home = str(Path(state_dir))
        env.update({"PILOTAGE_HOME": home, "HERMES_HOME": home})
    return env


def _capture_limit(context: ToolContext) -> Optional[int]:
    value = getattr(context.config, "max_tool_result_chars", None)
    if value is None:
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def _timeout(value: Any) -> tuple[Optional[int], Optional[str]]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, "timeout must be a positive whole number of seconds"
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, "timeout must be a positive whole number of seconds"
    if parsed <= 0 or (isinstance(value, float) and not value.is_integer()):
        return None, "timeout must be a positive whole number of seconds"
    return parsed, None


async def handle(args: Dict[str, Any], context: ToolContext) -> str:
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return tool_error("command must be a non-empty string")

    timeout, timeout_error = _timeout(args.get("timeout"))
    if timeout_error:
        return tool_error(timeout_error)

    workdir = args.get("workdir", "")
    if workdir is None:
        workdir = ""
    if not isinstance(workdir, str):
        return tool_error("workdir must be a path string")

    session = get_terminal_session(context)
    async with session.lock:
        if session.shell is None:
            try:
                default_timeout = int(
                    _setting(context, "terminal.timeout", DEFAULT_TIMEOUT_SECONDS)
                )
            except (TypeError, ValueError) as exc:
                return tool_error(f"terminal.timeout is invalid: {exc}")
            if default_timeout <= 0:
                return tool_error("terminal.timeout must be greater than zero")
            cwd = shell_cwd(context)
            session.shell = await asyncio.to_thread(
                Shell, cwd=cwd, timeout=default_timeout, env=shell_env(context)
            )

        result = await asyncio.to_thread(
            session.shell.execute,
            command,
            workdir,
            timeout=timeout,
            capture_limit=_capture_limit(context),
        )

        response: Dict[str, Any] = {
            "output": redact_terminal_output(result.get("output") or "", command),
            "exit_code": int(result.get("returncode", -1)),
        }
        # A per-command workdir is a scope, not a move. Shell only sets this
        # flag when the command itself changed the persistent session cwd.
        if result.get("cwd_observed"):
            response["cwd"] = session.shell.cwd
        if result.get("stdin_error"):
            response["stdin_error"] = redact_sensitive_text(result["stdin_error"])
        return json.dumps(response, ensure_ascii=False)


TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": (
        "Execute shell commands in the agent's Linux workspace. The current "
        "working directory, exported variables, functions and aliases persist "
        "between calls in this chat. Commands run in the foreground and return "
        "as soon as they finish. Use workdir for one command without moving the "
        "session; when a command itself changes directory, the result reports cwd. "
        "Use write_file/patch for file edits and the memory/cronjob tools for their "
        "stores; those paths carry the profile's required approval workflow."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Maximum seconds to wait. Omit to use the configured terminal timeout."
                ),
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Working directory for this command only. Omit to use the session cwd."
                ),
            },
        },
        "required": ["command"],
    },
}


TERMINAL_TOOL = Tool(
    name="terminal",
    group="terminal",
    schema=TERMINAL_SCHEMA,
    handler=handle,
    emoji="💻",
)


__all__ = [
    "TERMINAL_SCHEMA",
    "TERMINAL_TOOL",
    "TerminalSession",
    "get_terminal_session",
    "handle",
]
