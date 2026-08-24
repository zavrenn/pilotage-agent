"""Run Python code in Pilotage's prepared skill environments.

Hermes's production process boundary is kept: a temporary script, a scrubbed
child environment, the session working directory, a hard timeout, bounded
head/tail output, ANSI removal, and secret redaction.  Pilotage does not need
Hermes's tool-RPC or remote-backend layers; the designated runtime instead
chooses one of four deployment-prepared Python environments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..redact import redact_sensitive_text
from ..settings import ConfigError
from .ansi_strip import strip_ansi
from .command_guard import find_blocked_python_source, profile_name_for_state_dir
from .registry import Tool, ToolContext, tool_error
from .subprocess_env import build_subprocess_env
from .terminal import get_terminal_session, shell_cwd

logger = logging.getLogger(__name__)

PREPARED_ENVIRONMENTS = ("chart", "docs", "excel", "pdf")
DEFAULT_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 3600
MAX_STDOUT_BYTES = 50_000
MAX_STDERR_BYTES = 10_000
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_ROOT = REPO_ROOT / ".pilotage-envs"

_SAFE_ENV_NAMES = frozenset({
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "TERM",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SHELL",
    "LOGNAME",
    "TZ",
})
_SAFE_ENV_PREFIXES = ("LC_", "XDG_")
_WINDOWS_ESSENTIAL_ENV_NAMES = frozenset({
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "OS",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "APPDATA",
    "LOCALAPPDATA",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
})
_SECRET_NAME_PARTS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "PASSWD",
    "AUTH",
    "DSN",
    "WEBHOOK",
    "CREDS",
    "BEARER",
    "APIKEY",
)


def validate_settings(settings: Any) -> None:
    """Validate code-execution settings without requiring a built deployment."""
    settings.text("code_execution.root", str(DEFAULT_ENV_ROOT))
    timeout = settings.count(
        "code_execution.timeout", DEFAULT_TIMEOUT_SECONDS
    )
    if timeout < 1 or timeout > MAX_TIMEOUT_SECONDS:
        raise ConfigError(
            "code_execution.timeout must be between 1 and "
            f"{MAX_TIMEOUT_SECONDS}, not {timeout!r}"
        )


def _setting(context: ToolContext, name: str, default: Any) -> Any:
    settings = getattr(context.config, "settings", None)
    if settings is None:
        return default
    if isinstance(default, int):
        return settings.count(name, default)
    return settings.text(name, default)


def environment_root(context: ToolContext) -> Path:
    configured = str(
        _setting(context, "code_execution.root", str(DEFAULT_ENV_ROOT))
    ).strip()
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root.resolve(strict=False)


def interpreter_path(context: ToolContext, environment: str) -> Path:
    root = environment_root(context)
    environment_root_path = root / environment
    if os.name == "nt":
        return environment_root_path / "Scripts" / "python.exe"
    return environment_root_path / "bin" / "python"


def _workspace(context: ToolContext) -> Path:
    session = get_terminal_session(context)
    shell = session.shell
    if shell is not None and getattr(shell, "cwd", None):
        return Path(str(shell.cwd)).expanduser().resolve(strict=False)
    return Path(shell_cwd(context)).expanduser().resolve(strict=False)


def _build_child_env(
    interpreter: Path,
    workspace: Path,
    source_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Use Hermes's strict code-child allowlist and select the target venv."""
    sanitized = build_subprocess_env(
        os.environ if source_env is None else source_env
    )
    child: Dict[str, str] = {}
    for name, value in sanitized.items():
        upper = name.upper()
        if any(part in upper for part in _SECRET_NAME_PARTS):
            continue
        if (
            upper in _SAFE_ENV_NAMES
            or any(upper.startswith(prefix) for prefix in _SAFE_ENV_PREFIXES)
            or (os.name == "nt" and upper in _WINDOWS_ESSENTIAL_ENV_NAMES)
        ):
            child[name] = value

    environment_dir = interpreter.parent.parent
    old_path = child.get("PATH", "")
    child["PATH"] = (
        str(interpreter.parent)
        + (os.pathsep + old_path if old_path else "")
    )
    child["VIRTUAL_ENV"] = str(environment_dir)
    child["PYTHONPATH"] = str(workspace)
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    child["PYTHONIOENCODING"] = "utf-8"
    child["PYTHONUTF8"] = "1"
    child["PYTHONNOUSERSITE"] = "1"
    return child


class _HeadTailCapture:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.head_limit = int(maximum * 0.4)
        self.tail_limit = maximum - self.head_limit
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if len(self.head) < self.head_limit:
            count = min(self.head_limit - len(self.head), len(chunk))
            self.head.extend(chunk[:count])
            chunk = chunk[count:]
        if chunk:
            self.tail.extend(chunk)
            if len(self.tail) > self.tail_limit:
                del self.tail[:len(self.tail) - self.tail_limit]

    def result(self) -> tuple[str, Dict[str, Any]]:
        captured = bytes(self.head + self.tail)
        truncated = self.total > len(captured)
        omitted = max(0, self.total - len(captured))
        if truncated:
            head = bytes(self.head).decode("utf-8", errors="replace")
            tail = bytes(self.tail).decode("utf-8", errors="replace")
            text = (
                head
                + f"\n\n... [OUTPUT TRUNCATED - {omitted:,} bytes omitted "
                + f"out of {self.total:,} total] ...\n\n"
                + tail
            )
        else:
            text = captured.decode("utf-8", errors="replace")
        return text, {
            "stdout_truncated": truncated,
            "stdout_bytes_captured": len(captured),
            "stdout_bytes_total": self.total,
            "stdout_bytes_omitted": omitted,
        }


class _HeadCapture:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.data = bytearray()
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if len(self.data) < self.maximum:
            self.data.extend(chunk[:self.maximum - len(self.data)])

    def result(self) -> tuple[str, bool]:
        return (
            bytes(self.data).decode("utf-8", errors="replace"),
            self.total > len(self.data),
        )


def _drain(pipe: Any, capture: Any) -> None:
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                return
            capture.feed(chunk)
    except (OSError, ValueError):
        logger.debug("Could not drain code-execution output", exc_info=True)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    else:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    else:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        logger.error(
            "Code-execution process %s did not exit after termination",
            process.pid,
        )


def _execute(
    code: str,
    environment: str,
    context: ToolContext,
    *,
    workspace: Optional[Path] = None,
) -> str:
    working_dir = workspace or _workspace(context)
    finding = find_blocked_python_source(
        code,
        cwd=str(working_dir),
        current_profile=profile_name_for_state_dir(
            getattr(context.config, "state_dir", None)
        ),
    )
    if finding:
        return tool_error(finding.message)

    interpreter = interpreter_path(context, environment)
    if not interpreter.is_file():
        return tool_error(
            f"Prepared environment '{environment}' is unavailable: "
            f"missing interpreter {interpreter}. Run pilotage doctor."
        )
    if os.name != "nt" and not os.access(interpreter, os.X_OK):
        return tool_error(
            f"Prepared environment '{environment}' is unavailable: "
            f"interpreter is not executable: {interpreter}. Run pilotage doctor."
        )

    if not working_dir.is_dir():
        return tool_error(
            f"Session working directory does not exist: {working_dir}"
        )
    timeout = int(
        _setting(
            context,
            "code_execution.timeout",
            DEFAULT_TIMEOUT_SECONDS,
        )
    )
    if timeout < 1 or timeout > MAX_TIMEOUT_SECONDS:
        return tool_error(
            "code_execution.timeout must be between 1 and "
            f"{MAX_TIMEOUT_SECONDS}"
        )

    started = time.monotonic()
    stdout_capture = _HeadTailCapture(MAX_STDOUT_BYTES)
    stderr_capture = _HeadCapture(MAX_STDERR_BYTES)
    status = "success"
    process: Optional[subprocess.Popen[bytes]] = None

    with tempfile.TemporaryDirectory(prefix="pilotage_code_") as directory:
        script = Path(directory) / "script.py"
        script.write_text(code, encoding="utf-8")
        creation_flags = 0
        if os.name == "nt":
            creation_flags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        try:
            process = subprocess.Popen(
                [str(interpreter), str(script)],
                cwd=str(working_dir),
                env=_build_child_env(interpreter, working_dir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name != "nt",
                creationflags=creation_flags,
            )
        except OSError as exc:
            return tool_error(
                f"Could not start prepared environment '{environment}': {exc}"
            )

        assert process.stdout is not None
        assert process.stderr is not None
        stdout_reader = threading.Thread(
            target=_drain,
            args=(process.stdout, stdout_capture),
            daemon=True,
        )
        stderr_reader = threading.Thread(
            target=_drain,
            args=(process.stderr, stderr_capture),
            daemon=True,
        )
        stdout_reader.start()
        stderr_reader.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = "timeout"
            _stop_process(process)
        finally:
            stdout_reader.join(timeout=3)
            stderr_reader.join(timeout=3)

    output, metadata = stdout_capture.result()
    stderr, stderr_truncated = stderr_capture.result()
    output = redact_sensitive_text(strip_ansi(output))
    stderr = redact_sensitive_text(strip_ansi(stderr))
    exit_code = (
        process.returncode
        if process is not None and process.returncode is not None
        else -1
    )
    duration = round(time.monotonic() - started, 2)

    result: Dict[str, Any] = {
        "status": status,
        "environment": environment,
        "output": output,
        "exit_code": exit_code,
        "duration_seconds": duration,
        **metadata,
    }
    if stderr:
        result["stderr"] = stderr
    if stderr_truncated:
        result["stderr_truncated"] = True
    if status == "timeout":
        result["error"] = f"Script timed out after {timeout}s and was killed."
    elif exit_code != 0:
        result["status"] = "error"
        result["error"] = f"Script exited with code {exit_code}."
    if metadata["stdout_truncated"]:
        result["warning"] = (
            "Output was truncated; rerun with narrower output only if the "
            "omitted data is required."
        )
    return json.dumps(result, ensure_ascii=False)


async def handle(args: Dict[str, Any], context: ToolContext) -> str:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        if "command" in args and "code" not in args:
            return tool_error(
                "execute_code requires Python source in 'code'. "
                "Use terminal for shell commands."
            )
        return tool_error("code must be a non-empty Python source string")
    environment = args.get("environment")
    if not isinstance(environment, str):
        return tool_error(
            "environment must be one of: "
            + ", ".join(PREPARED_ENVIRONMENTS)
        )
    environment = environment.strip().lower()
    if environment not in PREPARED_ENVIRONMENTS:
        return tool_error(
            "environment must be one of: "
            + ", ".join(PREPARED_ENVIRONMENTS)
        )

    session = get_terminal_session(context)
    async with session.lock:
        workspace = _workspace(context)
        return await asyncio.to_thread(
            _execute,
            code,
            environment,
            context,
            workspace=workspace,
        )


EXECUTE_CODE_SCHEMA = {
    "name": "execute_code",
    "description": (
        "Run Python in one deployment-prepared environment: chart for plots "
        "and images, docs for parsing/OCR/conversion, excel for workbooks, or "
        "pdf for bilingual/RTL PDF rendering. The script runs in the session "
        "working directory. Print only the result needed. Use terminal for "
        "shell commands."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            },
            "environment": {
                "type": "string",
                "enum": list(PREPARED_ENVIRONMENTS),
                "description": "The prepared dependency environment.",
            },
        },
        "required": ["code", "environment"],
    },
}


EXECUTE_CODE_TOOL = Tool(
    name="execute_code",
    group="code_execution",
    schema=EXECUTE_CODE_SCHEMA,
    handler=handle,
    emoji="🐍",
)


__all__ = [
    "DEFAULT_ENV_ROOT",
    "EXECUTE_CODE_SCHEMA",
    "EXECUTE_CODE_TOOL",
    "PREPARED_ENVIRONMENTS",
    "environment_root",
    "handle",
    "interpreter_path",
    "validate_settings",
]
