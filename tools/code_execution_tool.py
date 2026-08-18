#!/usr/bin/env python3
"""
Code Execution Tool -- Programmatic Tool Calling (PTC)

Lets the LLM write a Python script that calls Pilotage tools via RPC,
collapsing multi-step tool chains into a single inference turn.

Architecture:

  1. Parent generates a `pilotage_tools.py` stub module with RPC functions
  2. Parent opens a socket and starts an RPC listener thread
  3. Parent spawns a child process that runs the LLM's script
  4. Tool calls travel over the socket back to the parent for dispatch

Only the script's stdout is returned to the LLM; intermediate tool results
never enter the context window.

Transport: a Unix domain socket on Linux/macOS, loopback TCP on Windows.
"""

import json
import logging
import os
import platform
import re
import secrets
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

_IS_WINDOWS = platform.system() == "Windows"
from typing import Any, Dict, List, Optional, Tuple

from tools.thread_context import propagate_context_to_thread
from agent.thread_scoped_output import thread_scoped_silence

# Availability gate.  On Windows we fall back to loopback TCP for the
# sandbox RPC transport (AF_UNIX is unreliable on Windows Python) — see
# ``_use_tcp_rpc`` in ``_execute_local`` below.  That makes execute_code
# available on every platform Pilotage itself runs on.
logger = logging.getLogger(__name__)

SANDBOX_AVAILABLE = True

# The 7 tools allowed inside the sandbox. The intersection of this list
# and the session's enabled tools determines which stubs are generated.
SANDBOX_ALLOWED_TOOLS = frozenset([
    "web_search",
    "web_extract",
    "read_file",
    "write_file",
    "search_files",
    "patch",
    "terminal",
])

# Resource limit defaults (overridable via config.yaml → code_execution.*)
DEFAULT_TIMEOUT = 300        # 5 minutes
DEFAULT_MAX_TOOL_CALLS = 50
MAX_STDOUT_BYTES = 50_000    # 50 KB
MAX_STDERR_BYTES = 10_000    # 10 KB


def _assemble_stdout_result(
    head: bytes,
    tail: bytes = b"",
    *,
    total_bytes: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build display stdout plus explicit truncation metadata.

    The agent receives execute_code results as JSON. A textual truncation
    marker can be missed or later re-truncated by a client layer, so keep the
    marker for humans and also expose byte counts for deterministic handling.
    """
    captured = head + tail
    total = len(captured) if total_bytes is None else max(total_bytes, len(captured))
    truncated = total > len(captured)
    omitted = max(0, total - len(captured))

    if truncated:
        stdout_text = (
            head.decode("utf-8", errors="replace")
            + f"\n\n... [OUTPUT TRUNCATED - {omitted:,} bytes omitted "
            f"out of {total:,} total] ...\n\n"
            + tail.decode("utf-8", errors="replace")
        )
    else:
        stdout_text = captured.decode("utf-8", errors="replace")

    metadata: Dict[str, Any] = {
        "stdout_truncated": truncated,
        "stdout_bytes_captured": len(captured),
        "stdout_bytes_total": total,
        "stdout_bytes_omitted": omitted,
    }
    if truncated:
        metadata["warning"] = (
            "execute_code stdout was truncated; the script did run, but only "
            "the captured head/tail output is included. Re-run only with "
            "narrower output if the omitted data is required."
        )
    return stdout_text, metadata


def _truncate_stdout_text(stdout_text: str) -> Tuple[str, Dict[str, Any]]:
    """Cap a complete stdout string by bytes using the same head/tail policy."""
    stdout_bytes = stdout_text.encode("utf-8", errors="replace")
    if len(stdout_bytes) <= MAX_STDOUT_BYTES:
        return _assemble_stdout_result(stdout_bytes)

    head_bytes = int(MAX_STDOUT_BYTES * 0.4)
    tail_bytes = MAX_STDOUT_BYTES - head_bytes
    return _assemble_stdout_result(
        stdout_bytes[:head_bytes],
        stdout_bytes[-tail_bytes:],
        total_bytes=len(stdout_bytes),
    )

# Environment variable scrubbing rules (shared between the local + remote
# backends).  Secret-substring block is applied first; anything left must
# match a safe prefix, the operational PILOTAGE_ allowlist, or (on Windows) an
# OS-essential name.  Delegate-task child context is also an exact-name
# operational marker: without it, a sandbox script that spawns/imports Pilotage
# code can lose the DB-layer mutation guard while still inheriting
# PILOTAGE_HOME.
#
# NB: the broad "PILOTAGE_" prefix was deliberately removed — it leaked
# PILOTAGE_*-named config that lacks a secret substring (e.g. PILOTAGE_BASE_URL,
# PILOTAGE_*_WEBHOOK).  The child only needs the few
# location/profile vars in _PILOTAGE_CHILD_ALLOWED below; PILOTAGE_RPC_SOCKET /
# PILOTAGE_RPC_DIR / TZ / HOME are injected explicitly after scrubbing.
_SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                      "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                      "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA")
_SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
                      "PASSWD", "AUTH", "DSN", "WEBHOOK",
                      # Abbreviations that appear in real-world credential
                      # variable names but were previously undetected:
                      # CREDS (CREDENTIALS abbreviated), BEARER
                      # (Authorization: Bearer tokens), APIKEY (written
                      # without an underscore). "PASS" is intentionally NOT
                      # added — it false-positives on legitimate non-secret
                      # vars (BYPASS_CACHE, COMPASS_DIR, PASSENGER_HOST) while
                      # PASSWORD/PASSWD already cover the credential cases.
                      "CREDS", "BEARER", "APIKEY")

# Operational PILOTAGE_* vars the child legitimately needs by exact name — these
# are non-secret runtime-location flags (the same set pilotage_cli treats as the
# runtime location) that repo-root modules a sandbox script imports may read at
# import time.  None match _SECRET_SUBSTRINGS.
_PILOTAGE_CHILD_ALLOWED = frozenset({
    "PILOTAGE_HOME",
    "PILOTAGE_PROFILE",
    "PILOTAGE_CONFIG",
    "PILOTAGE_ENV",
    "PILOTAGE_DELEGATED_CHILD_CONTEXT",
})

# Windows-only: a handful of variables are required by the OS/CRT itself.
# Without them, even stdlib calls like ``socket.socket()`` fail with
# WinError 10106 (Winsock can't locate mswsock.dll) and ``subprocess``
# can't resolve cmd.exe.  These are well-known OS paths, not secrets, so
# we allow them through by exact name.  The _SECRET_SUBSTRINGS block
# still runs as a safety net (none of these names match those substrings).
_WINDOWS_ESSENTIAL_ENV_VARS = frozenset({
    "SYSTEMROOT",       # %SYSTEMROOT%\System32 — Winsock needs this
    "SYSTEMDRIVE",      # C: (or wherever Windows lives)
    "WINDIR",           # usually same as SYSTEMROOT
    "COMSPEC",          # cmd.exe path — subprocess shell=True needs it
    "PATHEXT",          # .COM;.EXE;.BAT;... — shell lookup
    "OS",               # "Windows_NT" — some tools gate on this
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    "PUBLIC",           # C:\Users\Public
    "ALLUSERSPROFILE",  # C:\ProgramData — some stdlib paths use it
    "PROGRAMDATA",      # C:\ProgramData
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "APPDATA",          # %USERPROFILE%\AppData\Roaming — Python uses it
    "LOCALAPPDATA",     # %USERPROFILE%\AppData\Local
    "USERPROFILE",      # C:\Users\<name> — Python's expanduser uses it
    "USERDOMAIN",
    "USERNAME",
    "HOMEDRIVE",        # C:
    "HOMEPATH",         # \Users\<name>
    "COMPUTERNAME",
})


def _scrub_child_env(source_env, is_passthrough=None, is_windows=None):
    """Produce the scrubbed child-process env for execute_code.

    Rules (order matters):
      1. Passthrough vars (skill- or config-declared) pass through the active
         profile secret scope; an absent scoped value is omitted and an
         unscoped multiplex read fails closed.
      2. Secret-substring names (KEY/TOKEN/DSN/WEBHOOK/etc.) are blocked.
      3. Names matching a safe prefix pass.
      4. Operational PILOTAGE_* vars (_PILOTAGE_CHILD_ALLOWED) pass by exact name.
      5. On Windows, a small OS-essential allowlist passes by exact name
         — without these the child can't even create a socket or spawn a
         subprocess.

    Extracted into a helper so tests can exercise the logic without
    spawning a subprocess.
    """
    resolve_passthrough_value = None
    if is_passthrough is None:
        try:
            from tools.env_passthrough import (
                is_env_passthrough as _ep,
                resolve_passthrough_value,
            )
        except Exception:
            _ep = lambda _: False  # noqa: E731
            resolve_passthrough_value = lambda _name, _fallback: None  # noqa: E731
        is_passthrough = _ep
    else:
        try:
            from tools.env_passthrough import resolve_passthrough_value
        except Exception:
            resolve_passthrough_value = lambda _name, _fallback: None  # noqa: E731
    if is_windows is None:
        is_windows = _IS_WINDOWS

    scrubbed = {}
    # Non-secret PILOTAGE_* vars dropped by the tightened allowlist. The
    # broad "PILOTAGE_" prefix used to pass these through; now only the
    # operational set does. The drop is intentional (those vars can carry
    # config like PILOTAGE_BASE_URL), but a sandbox script
    # that imports a repo module reading one at import time would otherwise see
    # it silently unset. Surface the drop once so the behavior change is
    # diagnosable and points at the env_passthrough opt-in escape hatch.
    _dropped_pilotage = []
    for k, v in source_env.items():
        if is_passthrough(k):
            resolved = resolve_passthrough_value(k, v)
            if resolved is not None:
                scrubbed[k] = resolved
            continue
        if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
            continue
        if any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
            scrubbed[k] = v
            continue
        if k in _PILOTAGE_CHILD_ALLOWED:
            scrubbed[k] = v
            continue
        if is_windows and k.upper() in _WINDOWS_ESSENTIAL_ENV_VARS:
            scrubbed[k] = v
            continue
        if k.startswith("PILOTAGE_"):
            # Non-secret (secrets were already dropped above) and not in any
            # allowlist — a deliberately-dropped PILOTAGE_* var.
            _dropped_pilotage.append(k)
    if _dropped_pilotage:
        logger.debug(
            "execute_code: dropped %d non-allowlisted PILOTAGE_* var(s) from the "
            "sandbox child env (%s). This is intentional hardening; if "
            "a sandbox script legitimately needs one, declare it via "
            "env_passthrough in the skill/config so it passes by explicit opt-in.",
            len(_dropped_pilotage),
            ", ".join(sorted(_dropped_pilotage)),
        )

    # delegate_task children are marked with a ContextVar, not os.environ, while
    # the execute_code sandbox crosses a process boundary. Bridge that context
    # into the child env and strip dispatcher-owned variables after the
    # normal secret/passthrough scrub so an explicit passthrough cannot re-grant
    # a delegated child the parent's mutation capability.
    try:
        from agent.delegation_context import (
            is_delegated_child_process_context,
            scrub_delegated_child_env,
        )

        if is_delegated_child_process_context():
            scrubbed = scrub_delegated_child_env(scrubbed)
    except Exception:
        pass
    return scrubbed


def check_sandbox_requirements() -> bool:
    """Availability gate for execute_code."""
    return SANDBOX_AVAILABLE


# ---------------------------------------------------------------------------
# pilotage_tools.py code generator
# ---------------------------------------------------------------------------

# Per-tool stub templates: (function_name, signature, docstring, args_dict_expr)
# The args_dict_expr builds the JSON payload sent over the RPC socket.
_TOOL_STUBS = {
    "web_search": (
        "web_search",
        "query: str, limit: int = 5",
        '"""Search the web. Returns dict with data.web list of {url, title, description}."""',
        '{"query": query, "limit": limit}',
    ),
    "web_extract": (
        "web_extract",
        "urls: list, char_limit: int = None",
        '"""Extract content from URLs (no LLM summarization). Returns dict with results list of {url, title, content, error}. Pages over char_limit (default 15000) are head+tail truncated with the full text stored on disk; the content footer gives the path. content is markdown."""',
        '{"urls": urls, "char_limit": char_limit}',
    ),
    "read_file": (
        "read_file",
        "path: str, offset: int = 1, limit: int = 2000",
        '"""Read a file (1-indexed lines). Returns dict with "content" and "total_lines"."""',
        '{"path": path, "offset": offset, "limit": limit}',
    ),
    "write_file": (
        "write_file",
        "path: str, content: str, cross_profile: bool = False",
        '"""Write content to a file (always overwrites). Returns dict with status. cross_profile=True opts out of the cross-Pilotage-profile soft guard."""',
        '{"path": path, "content": content, "cross_profile": cross_profile}',
    ),
    "search_files": (
        "search_files",
        'pattern: str, target: str = "content", path: str = ".", file_glob: str = None, limit: int = 50, offset: int = 0, output_mode: str = "content", context: int = 0',
        '"""Search file contents (target="content") or find files by name (target="files"). Returns dict with "matches"."""',
        '{"pattern": pattern, "target": target, "path": path, "file_glob": file_glob, "limit": limit, "offset": offset, "output_mode": output_mode, "context": context}',
    ),
    "patch": (
        "patch",
        'path: str = None, old_string: str = None, new_string: str = None, replace_all: bool = False, mode: str = "replace", patch: str = None, cross_profile: bool = False',
        '"""Targeted find-and-replace (mode="replace") or V4A multi-file patches (mode="patch"). Returns dict with status. cross_profile=True opts out of the cross-Pilotage-profile soft guard."""',
        '{"path": path, "old_string": old_string, "new_string": new_string, "replace_all": replace_all, "mode": mode, "patch": patch, "cross_profile": cross_profile}',
    ),
    "terminal": (
        "terminal",
        "command: str, timeout: int = None, workdir: str = None",
        '"""Run a shell command (foreground only). Returns dict with "output" and "exit_code"."""',
        '{"command": command, "timeout": timeout, "workdir": workdir}',
    ),
}


def _sandbox_failure_hint(stderr_text: str, enabled_tools=None) -> Optional[str]:
    """Map well-known sandbox script failures to one actionable recovery hint.

    Production mining (state.db): the top execute_code failure classes are
    pilotage_tools import misuse (importing tools that aren't in the sandbox,
    23x in one window), calling the built-in helpers via import, treating
    tool results as strings instead of dicts, and importing third-party
    packages that don't exist in the sandbox interpreter. Bounded scan,
    first match wins, never raises.
    """
    if not stderr_text:
        return None
    window = stderr_text[:4000]
    try:
        m = re.search(
            r"cannot import name '(\w+)' from 'pilotage_tools'", window
        )
        if m:
            missing = m.group(1)
            available = sorted(SANDBOX_ALLOWED_TOOLS & set(enabled_tools or SANDBOX_ALLOWED_TOOLS))
            builtin = {"json_parse", "shell_quote", "retry"}
            if missing in builtin:
                return (
                    f"{missing} is a BUILT-IN helper in the sandbox — no import "
                    f"needed. Remove it from the import line and call {missing}(...) directly."
                )
            return (
                f"'{missing}' is not available inside the execute_code sandbox. "
                f"Importable tools here: {', '.join(available)}. For anything "
                "else, use the normal tool call instead of execute_code."
            )
        m = re.search(r"NameError: name '(json_parse|shell_quote|retry)' is not defined", window)
        if m:
            return (
                f"{m.group(1)} is built into the generated sandbox module — "
                "call it directly at module scope without importing it."
            )
        m = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", window)
        if m:
            return (
                f"'{m.group(1)}' is not installed in the sandbox interpreter. "
                "Use Python stdlib inside execute_code, or run the code via "
                "terminal() with the project venv's python instead."
            )
        if re.search(r"TypeError: string indices must be integers|AttributeError: 'str' object has no attribute 'get'", window):
            return (
                "Tool functions in the sandbox return DICTS (already parsed) — "
                "do not json.loads() them or index them like strings. "
                "Example: read_file(path)['content']."
            )
    except Exception:
        return None
    return None


def generate_pilotage_tools_module(enabled_tools: List[str]) -> str:
    """
    Build the source code for the pilotage_tools.py stub module.

    Only tools in both SANDBOX_ALLOWED_TOOLS and enabled_tools get stubs.

    Args:
        enabled_tools: Tool names enabled in the current session.
    """
    tools_to_generate = sorted(SANDBOX_ALLOWED_TOOLS & set(enabled_tools))

    stub_functions = []
    export_names = []
    for tool_name in tools_to_generate:
        if tool_name not in _TOOL_STUBS:
            continue
        func_name, sig, doc, args_expr = _TOOL_STUBS[tool_name]
        stub_functions.append(
            f"def {func_name}({sig}):\n"
            f"    {doc}\n"
            f"    return _call({func_name!r}, {args_expr})\n"
        )
        export_names.append(func_name)

    return _UDS_TRANSPORT_HEADER + "\n".join(stub_functions)


# ---- Shared helpers section (embedded in the transport header) ------------

_COMMON_HELPERS = '''\

# ---------------------------------------------------------------------------
# Convenience helpers (avoid common scripting pitfalls)
# ---------------------------------------------------------------------------

def json_parse(text: str):
    """Parse JSON tolerant of control characters and UTF-8 BOM (strict=False).
    Use this instead of json.loads() when parsing output from terminal()
    or web_extract() that may contain raw tabs/newlines in strings,
    or from tools/files that prepend a UTF-8 BOM (salvage, credit @woxinwuhen713-bit)."""
    if isinstance(text, str) and text.startswith("﻿"):
        text = text[1:]
    return json.loads(text, strict=False)


def shell_quote(s: str) -> str:
    """Shell-escape a string for safe interpolation into commands.
    Use this when inserting dynamic content into terminal() commands:
        terminal(f"echo {shell_quote(user_input)}")
    """
    return shlex.quote(s)


def retry(fn, max_attempts=3, delay=2):
    """Retry a function up to max_attempts times with exponential backoff.
    Use for transient failures (network errors, API rate limits):
        result = retry(lambda: terminal("gh issue list ..."))
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (2 ** attempt))
    raise last_err

'''

# ---- UDS transport -------------------------------------------------------

_UDS_TRANSPORT_HEADER = '''\
"""Auto-generated Pilotage tools RPC stubs."""
import json, os, socket, shlex, threading, time

_sock = None
# The RPC server handles a single client connection serially and has no
# request-id in the protocol, so concurrent _call() invocations from multiple
# threads (e.g. ThreadPoolExecutor) would race on the shared socket and get
# each other's responses. Serialize the entire send+recv round-trip.
_call_lock = threading.Lock()
''' + _COMMON_HELPERS + '''\

def _connect():
    """Connect to the parent's RPC server via the transport it picked.

    PILOTAGE_RPC_SOCKET can be either:
      - a filesystem path (POSIX Unix domain socket — the default on
        Linux and macOS)
      - a string of the form ``tcp://127.0.0.1:<port>`` (Windows, where
        AF_UNIX is unreliable — the parent falls back to loopback TCP)
    """
    global _sock
    if _sock is None:
        endpoint = os.environ["PILOTAGE_RPC_SOCKET"]
        if endpoint.startswith("tcp://"):
            # tcp://host:port  (host is always 127.0.0.1 in practice — we
            # only bind loopback server-side)
            _host_port = endpoint[len("tcp://"):]
            _host, _, _port = _host_port.rpartition(":")
            _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _sock.connect((_host or "127.0.0.1", int(_port)))
        else:
            _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            _sock.connect(endpoint)
        _sock.settimeout(300)
    return _sock

def _call(tool_name, args):
    """Send a tool call to the parent process and return the parsed result."""
    request = json.dumps({
        "tool": tool_name,
        "args": args,
        "token": os.environ.get("PILOTAGE_RPC_TOKEN", ""),
    }) + "\\n"
    with _call_lock:
        conn = _connect()
        conn.sendall(request.encode())
        buf = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                raise RuntimeError("Agent process disconnected")
            buf += chunk
            if buf.endswith(b"\\n"):
                break
    raw = buf.decode().strip()
    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result

'''

# ---------------------------------------------------------------------------
# RPC server (runs in a thread inside the parent process)
# ---------------------------------------------------------------------------

# Terminal parameters that must not be used from ephemeral sandbox scripts
_TERMINAL_BLOCKED_PARAMS = {"background", "pty", "notify_on_complete", "watch_patterns"}


def _rpc_server_loop(
    server_sock: socket.socket,
    task_id: str,
    tool_call_log: list,
    tool_call_counter: list,   # mutable [int] so the thread can increment
    max_tool_calls: int,
    allowed_tools: frozenset,
    stop_event: threading.Event,
    rpc_token: str,
):
    """
    Accept one client connection and dispatch tool-call requests until
    the client disconnects or the call limit is reached.
    """
    from model_tools import handle_function_call

    conn = None
    try:
        server_sock.settimeout(0.05)
        while not stop_event.is_set():
            try:
                conn, _ = server_sock.accept()
                break
            except socket.timeout:
                continue
        if conn is None:
            return
        conn.settimeout(300)

        buf = b""
        while True:
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk

            # Process all complete newline-delimited messages in the buffer
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue

                call_start = time.monotonic()
                try:
                    request = json.loads(line.decode())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    resp = tool_error(f"Invalid RPC request: {exc}")
                    conn.sendall((resp + "\n").encode())
                    continue

                if not rpc_token or not secrets.compare_digest(
                    # Compare as bytes: compare_digest raises TypeError on a
                    # str with non-ASCII characters, and the token comes from
                    # sandbox-script-supplied JSON.
                    str(request.get("token") or "").encode(), rpc_token.encode()
                ):
                    resp = tool_error("Unauthorized RPC request")
                    conn.sendall((resp + "\n").encode())
                    continue

                tool_name = request.get("tool", "")
                tool_args = request.get("args", {})

                # Enforce the allow-list
                if tool_name not in allowed_tools:
                    available = ", ".join(sorted(allowed_tools))
                    resp = tool_error(
                        f"Tool '{tool_name}' is not available in execute_code. "
                        f"Available: {available}"
                    )
                    conn.sendall((resp + "\n").encode())
                    continue

                # Enforce tool call limit
                if tool_call_counter[0] >= max_tool_calls:
                    resp = tool_error(
                        f"Tool call limit reached ({max_tool_calls}). "
                        "No more tool calls allowed in this execution."
                    )
                    conn.sendall((resp + "\n").encode())
                    continue

                # Strip forbidden terminal parameters
                if tool_name == "terminal" and isinstance(tool_args, dict):
                    for param in _TERMINAL_BLOCKED_PARAMS:
                        tool_args.pop(param, None)

                # Dispatch through the standard tool handler.
                # Suppress stdout/stderr from internal tool handlers so
                # their status prints don't leak into the CLI spinner.
                try:
                    with thread_scoped_silence():
                        result = handle_function_call(
                            tool_name, tool_args, task_id=task_id
                        )
                except Exception as exc:
                    logger.error("Tool call failed in sandbox: %s", exc, exc_info=True)
                    result = tool_error(str(exc))

                tool_call_counter[0] += 1
                call_duration = time.monotonic() - call_start

                # Log for observability
                args_preview = str(tool_args)[:80]
                tool_call_log.append({
                    "tool": tool_name,
                    "args_preview": args_preview,
                    "duration": round(call_duration, 2),
                })

                conn.sendall((result + "\n").encode())

    except socket.timeout:
        logger.debug("RPC listener socket timeout")
    except OSError as e:
        logger.debug("RPC listener socket error: %s", e, exc_info=True)
    finally:
        if conn:
            try:
                conn.close()
            except OSError as e:
                logger.debug("RPC conn close error: %s", e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def execute_code(
    code: str,
    task_id: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
) -> str:
    """
    Run a Python script in a sandboxed child process with RPC access
    to a subset of Pilotage tools.

    Args:
        code:          Python source code to execute.
        task_id:       Session task ID for tool isolation (terminal env, etc.).
        enabled_tools: Tool names enabled in the current session. The sandbox
                       gets the intersection with SANDBOX_ALLOWED_TOOLS.

    Returns:
        JSON string with execution results.
    """
    if not SANDBOX_AVAILABLE:
        return tool_error(
            "execute_code sandbox is unavailable in this environment. "
            "Use normal tool calls (terminal, read_file, write_file, ...) instead."
        )

    if not code or not code.strip():
        return tool_error(
            "No code provided. execute_code requires a non-empty 'code' "
            "parameter containing Python source. To run shell commands, use "
            "terminal(command=...) instead."
        )

    from tools.terminal_tool import _get_env_config
    env_type = _get_env_config()["env_type"]

    # execute_code runs arbitrary Python (subprocess/os.system/...) that never
    # passes through terminal()/DANGEROUS_PATTERNS, so guard the whole script
    # here before either dispatch path spawns it. Runs synchronously in the
    # caller (tool-executor) thread, which holds the session context.
    from tools.approval import check_execute_code_guard
    _guard = check_execute_code_guard(code, env_type)
    if not _guard.get("approved", False):
        return json.dumps({
            "status": "error",
            "error": _guard.get("message") or "execute_code blocked by approval guard.",
            "tool_calls_made": 0,
            "duration_seconds": 0,
        }, ensure_ascii=False)

    # Clean interrupt slate for a user-approved script before it is spawned:
    # drop a stale bit that landed on this thread during the blocking
    # approval-wait so it can't kill the just-approved run on the first poll of
    # the _wait_for_process loop.  A genuine post-clear interrupt re-sets the
    # bit and is still caught downstream.
    if _guard.get("user_approved"):
        from tools.interrupt import clear_current_thread_interrupt
        clear_current_thread_interrupt()

    # Import per-thread interrupt check (cooperative cancellation)
    from tools.interrupt import is_interrupted as _is_interrupted

    # Resolve config
    _cfg = _load_config()
    timeout = _cfg.get("timeout", DEFAULT_TIMEOUT)
    max_tool_calls = _cfg.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)

    # Determine which tools the sandbox can call
    session_tools = set(enabled_tools) if enabled_tools else set()
    sandbox_tools = frozenset(SANDBOX_ALLOWED_TOOLS & session_tools)

    if not sandbox_tools:
        sandbox_tools = SANDBOX_ALLOWED_TOOLS

    # --- Set up temp directory with pilotage_tools.py and script.py ---
    tmpdir = tempfile.mkdtemp(prefix="pilotage_sandbox_")
    # Use /tmp on macOS to avoid the long /var/folders/... path that pushes
    # Unix domain socket paths past the 104-byte macOS AF_UNIX limit.
    # On Linux, tempfile.gettempdir() already returns /tmp.
    #
    # Windows: Python 3.9+ added partial AF_UNIX support but the file-backed
    # variant is flaky across Windows builds (requires Windows 10 1803+,
    # still fails under some configurations, and the socket file can't live
    # on the same temp drive as the script).  Fall back to loopback TCP —
    # same ephemeral port, same 1-connection listen queue, same serialized
    # request/response framing.  The generated client reads the transport
    # selector from PILOTAGE_RPC_SOCKET (path vs. ``tcp://host:port``).
    _sock_tmpdir = "/tmp" if sys.platform == "darwin" else tempfile.gettempdir()
    _use_tcp_rpc = _IS_WINDOWS
    if _use_tcp_rpc:
        sock_path = None  # not used on Windows; TCP endpoint stored below
        rpc_endpoint = None  # set after bind()
    else:
        sock_path = os.path.join(_sock_tmpdir, f"pilotage_rpc_{uuid.uuid4().hex}.sock")
        rpc_endpoint = sock_path

    tool_call_log: list = []
    tool_call_counter = [0]  # mutable so the RPC thread can increment
    exec_start = time.monotonic()
    server_sock = None
    stop_event = threading.Event()

    try:
        # Write the auto-generated pilotage_tools module.
        # encoding="utf-8" is required on Windows — the stub and user code
        # both contain non-ASCII characters (em-dashes in docstrings, plus
        # whatever the user script carries).  Python's default open() uses
        # the system locale on Windows (cp1252 typically), which corrupts
        # those bytes; the child then fails to import with a SyntaxError
        # ("'utf-8' codec can't decode byte 0x97 in position ...") because
        # Python source files are decoded as UTF-8 by default (PEP 3120).
        # sandbox_tools is already the correct set (intersection with session
        # tools, or SANDBOX_ALLOWED_TOOLS as fallback — see lines above).
        tools_src = generate_pilotage_tools_module(list(sandbox_tools))
        with open(os.path.join(tmpdir, "pilotage_tools.py"), "w", encoding="utf-8") as f:
            f.write(tools_src)

        # Write the user's script
        with open(os.path.join(tmpdir, "script.py"), "w", encoding="utf-8") as f:
            f.write(code)

        # --- Start RPC server ---
        rpc_token = secrets.token_urlsafe(32)
        # Two transports:
        #   POSIX: AF_UNIX stream socket on sock_path, chmod 0600 for
        #   owner-only access.  Filesystem permissions gate the socket.
        #   Windows: AF_INET stream socket on 127.0.0.1 with an ephemeral
        #   port.  No filesystem permission story, but loopback-only bind
        #   means only the current user's processes (not remote) can
        #   connect.  PILOTAGE_RPC_SOCKET is set to ``tcp://127.0.0.1:<port>``
        #   which the generated client parses to pick AF_INET.
        if _use_tcp_rpc:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.bind(("127.0.0.1", 0))  # ephemeral port
            _host, _port = server_sock.getsockname()[:2]
            rpc_endpoint = f"tcp://{_host}:{_port}"
        else:
            server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server_sock.bind(sock_path)
            os.chmod(sock_path, 0o600)
        server_sock.listen(1)

        # Wrapped so the thread inherits the turn's approval context + callbacks
        # (see tools.thread_context) — else gateway sandbox tool calls silently
        # auto-approve dangerous commands.
        rpc_thread = threading.Thread(
            target=propagate_context_to_thread(_rpc_server_loop),
            args=(
                server_sock, task_id, tool_call_log,
                tool_call_counter, max_tool_calls, sandbox_tools, stop_event, rpc_token,
            ),
            daemon=True,
        )
        rpc_thread.start()

        # --- Spawn child process ---
        # Build a minimal environment for the child. We intentionally exclude
        # API keys and tokens to prevent credential exfiltration from LLM-
        # generated scripts. The child accesses tools via RPC, not direct API.
        # Exception: env vars declared by loaded skills (via env_passthrough
        # registry) or explicitly allowed by the user in config.yaml
        # (terminal.env_passthrough) are passed through.  On Windows, a small
        # OS-essential allowlist (SYSTEMROOT, WINDIR, COMSPEC, ...) is also
        # passed through — without those, the child can't create a socket
        # or spawn a subprocess.  See ``_scrub_child_env`` for the rules.
        child_env = _scrub_child_env(os.environ)
        child_env["PILOTAGE_RPC_SOCKET"] = rpc_endpoint
        child_env["PILOTAGE_RPC_TOKEN"] = rpc_token
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Force UTF-8 for the child's stdio and default file encoding.
        #
        # Without this, on Windows sys.stdout is bound to the console code
        # page (cp1252 on US-locale installs), and any script that does
        # ``print("café")`` or ``print("→")`` crashes with:
        #
        #   UnicodeEncodeError: 'charmap' codec can't encode character
        #   '\u2192' in position N: character maps to <undefined>
        #
        # PYTHONIOENCODING fixes sys.stdin/stdout/stderr.
        # PYTHONUTF8=1 enables "UTF-8 mode" (PEP 540) which additionally
        # makes ``open()``'s default encoding UTF-8, so user scripts that
        # write files without specifying encoding= also work correctly.
        #
        # On POSIX both values usually match the locale default already,
        # so setting them is harmless belt-and-suspenders for environments
        # with a C/POSIX locale (containers, minimal base images).
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        # Inject user's configured timezone so datetime.now() in sandboxed
        # code reflects the correct wall-clock time.  Only TZ is set —
        # PILOTAGE_TIMEZONE is an internal Pilotage setting and must not leak
        # into child processes.
        _tz_name = os.getenv("PILOTAGE_TIMEZONE", "").strip()
        if _tz_name:
            child_env["TZ"] = _tz_name
        child_env.pop("PILOTAGE_TIMEZONE", None)

        from pilotage_constants import apply_subprocess_home_env
        apply_subprocess_home_env(child_env)

        # Resolve interpreter + CWD based on execute_code mode.
        #   - strict : today's behavior (sys.executable + tmpdir CWD).
        #   - project: user's venv python + session's working directory, so
        #              project deps like pandas and user files resolve.
        # Env scrubbing and tool whitelist apply identically in both modes.
        _mode = _get_execution_mode()
        _child_python = _resolve_child_python(_mode)
        _child_cwd = _resolve_child_cwd(_mode, tmpdir, task_id=task_id or "")
        _script_path = os.path.join(tmpdir, "script.py")

        # ``pilotage_tools.py`` always lives in the staging directory, so that
        # directory must be importable even when project mode changes CWD.
        # Pilotage's own package root is useful too, but only when the child
        # uses the same Python environment. Project mode can select an
        # external venv; exposing Pilotage's site-packages to that interpreter
        # can mix incompatible compiled extensions (for example, Python 3.12
        # NumPy with a Python 3.9 project interpreter).
        _pilotage_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _existing_pp = child_env.get("PYTHONPATH", "")
        _pp_parts = [tmpdir]
        if _uses_pilotage_python_environment(_child_python):
            _pp_parts.append(_pilotage_root)
        elif _child_python not in _external_env_logged:
            # Import behavior changes silently otherwise — surface it (once
            # per interpreter path) so "import pilotage_constants suddenly
            # fails" reports are diagnosable without log spam.
            _external_env_logged.add(_child_python)
            logger.info(
                "execute_code: child interpreter %s is outside the Pilotage "
                "environment; pilotage root omitted from PYTHONPATH",
                _child_python,
            )
        if _existing_pp:
            _pp_parts.append(_existing_pp)
        child_env["PYTHONPATH"] = os.pathsep.join(_pp_parts)

        proc = subprocess.Popen(
            [_child_python, _script_path],
            cwd=_child_cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
        )

        # --- Poll loop: watch for exit, timeout, and interrupt ---
        deadline = time.monotonic() + timeout
        stderr_chunks: list = []

        # Background readers to avoid pipe buffer deadlocks.
        # For stdout we use a head+tail strategy: keep the first HEAD_BYTES
        # and a rolling window of the last TAIL_BYTES so the final print()
        # output is never lost.  Stderr keeps head-only (errors appear early).
        _STDOUT_HEAD_BYTES = int(MAX_STDOUT_BYTES * 0.4)   # 40% head
        _STDOUT_TAIL_BYTES = MAX_STDOUT_BYTES - _STDOUT_HEAD_BYTES  # 60% tail

        def _drain(pipe, chunks, max_bytes):
            """Simple head-only drain (used for stderr)."""
            total = 0
            try:
                while True:
                    data = pipe.read(4096)
                    if not data:
                        break
                    if total < max_bytes:
                        keep = max_bytes - total
                        chunks.append(data[:keep])
                    total += len(data)
            except (ValueError, OSError) as e:
                logger.debug("Error reading process output: %s", e, exc_info=True)

        stdout_total_bytes = [0]  # mutable ref for total bytes seen

        def _drain_head_tail(pipe, head_chunks, tail_chunks, head_bytes, tail_bytes, total_ref):
            """Drain stdout keeping both head and tail data."""
            head_collected = 0
            from collections import deque
            tail_buf = deque()
            tail_collected = 0
            try:
                while True:
                    data = pipe.read(4096)
                    if not data:
                        break
                    total_ref[0] += len(data)
                    # Fill head buffer first
                    if head_collected < head_bytes:
                        keep = min(len(data), head_bytes - head_collected)
                        head_chunks.append(data[:keep])
                        head_collected += keep
                        data = data[keep:]  # remaining goes to tail
                        if not data:
                            continue
                    # Everything past head goes into rolling tail buffer
                    tail_buf.append(data)
                    tail_collected += len(data)
                    # Evict old tail data to stay within tail_bytes budget
                    while tail_collected > tail_bytes and tail_buf:
                        oldest = tail_buf.popleft()
                        tail_collected -= len(oldest)
            except (ValueError, OSError):
                pass
            # Transfer final tail to output list
            tail_chunks.extend(tail_buf)

        stdout_head_chunks: list = []
        stdout_tail_chunks: list = []

        stdout_reader = threading.Thread(
            target=_drain_head_tail,
            args=(proc.stdout, stdout_head_chunks, stdout_tail_chunks,
                  _STDOUT_HEAD_BYTES, _STDOUT_TAIL_BYTES, stdout_total_bytes),
            daemon=True
        )
        stderr_reader = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_chunks, MAX_STDERR_BYTES), daemon=True
        )
        stdout_reader.start()
        stderr_reader.start()

        status = "success"
        _activity_state = {
            "last_touch": time.monotonic(),
            "start": exec_start,
        }
        try:
            from tools.environments.base import touch_activity_if_due
        except Exception:
            touch_activity_if_due = None
        poll_interval = 0.005
        while proc.poll() is None:
            if _is_interrupted():
                _kill_process_group(proc)
                status = "interrupted"
                break
            now = time.monotonic()
            if now > deadline:
                _kill_process_group(proc, escalate=True)
                status = "timeout"
                break
            # Periodic activity touch so the gateway's inactivity timeout
            # doesn't kill the agent during long code execution.
            if touch_activity_if_due is not None:
                try:
                    touch_activity_if_due(_activity_state, "execute_code running")
                except Exception:
                    pass
            try:
                proc.wait(timeout=min(poll_interval, max(0.0, deadline - now)))
            except subprocess.TimeoutExpired:
                pass
            poll_interval = min(0.2, poll_interval * 1.5)

        # Wait for readers to finish draining
        stdout_reader.join(timeout=3)
        stderr_reader.join(timeout=3)

        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        stdout_text, stdout_metadata = _assemble_stdout_result(
            b"".join(stdout_head_chunks),
            b"".join(stdout_tail_chunks),
            total_bytes=stdout_total_bytes[0],
        )

        exit_code = proc.returncode if proc.returncode is not None else -1
        duration = round(time.monotonic() - exec_start, 2)

        # Wait for RPC thread to finish
        stop_event.set()
        server_sock.close()  # break accept() so thread exits promptly
        server_sock = None  # prevent double close in finally
        rpc_thread.join(timeout=3)

        # Strip ANSI escape sequences so the model never sees terminal
        # formatting — prevents it from copying escapes into file writes.
        from tools.ansi_strip import strip_ansi
        stdout_text = strip_ansi(stdout_text)
        stderr_text = strip_ansi(stderr_text)

        # Redact secrets (API keys, tokens, etc.) from sandbox output.
        # The sandbox env-var filter (lines 434-454) blocks os.environ access,
        # but scripts can still read secrets from disk (e.g. open('~/.pilotage/.env')).
        # This ensures leaked secrets never enter the model context.
        # code_file=True: this is code-execution output — skip false-positive
        # ENV/JSON/f-string-template redaction; real credentials still masked.
        from agent.redact import redact_sensitive_text
        stdout_text = redact_sensitive_text(stdout_text, code_file=True)
        stderr_text = redact_sensitive_text(stderr_text, code_file=True)

        # Build response
        result: Dict[str, Any] = {
            "status": status,
            "output": stdout_text,
            "exit_code": exit_code,
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }
        result.update(stdout_metadata)

        if status == "timeout":
            timeout_msg = f"Script timed out after {timeout}s and was killed."
            result["error"] = timeout_msg
            # Include timeout message in output so the LLM always surfaces it
            # to the user.  When output is empty, models often treat the result
            # as "nothing happened" and produce an empty response, which the
            # gateway stream consumer silently drops.
            if stdout_text:
                result["output"] = stdout_text + f"\n\n⏰ {timeout_msg}"
            else:
                result["output"] = f"⏰ {timeout_msg}"
            logger.warning(
                "execute_code timed out after %ss (limit %ss) with %d tool calls",
                duration, timeout, tool_call_counter[0],
            )
        elif status == "interrupted":
            result["output"] = stdout_text + "\n[execution interrupted — user sent a new message]"
        elif exit_code != 0:
            result["status"] = "error"
            result["error"] = stderr_text or f"Script exited with code {exit_code}"
            # Include stderr in output so the LLM sees the traceback
            if stderr_text:
                result["output"] = stdout_text + "\n--- stderr ---\n" + stderr_text
            # Known-failure-class recovery hint (import misuse, missing
            # module, dict-vs-string result handling) so the model fixes
            # the script on the next attempt instead of re-diagnosing.
            hint = _sandbox_failure_hint(stderr_text, enabled_tools=sandbox_tools)
            if hint:
                result["hint"] = hint

        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        duration = round(time.monotonic() - exec_start, 2)
        logger.error(
            "execute_code failed after %ss with %d tool calls: %s: %s",
            duration,
            tool_call_counter[0],
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return json.dumps({
            "status": "error",
            "error": str(exc),
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }, ensure_ascii=False)

    finally:
        # Cleanup temp dir and socket
        if server_sock is not None:
            try:
                server_sock.close()
            except OSError as e:
                logger.debug("Server socket close error: %s", e)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            # Only UDS has a filesystem socket to unlink; TCP sockets are
            # freed by server_sock.close() above.
            if sock_path:
                os.unlink(sock_path)
        except OSError:
            pass  # already cleaned up or never created


def _kill_process_group(proc, escalate: bool = False):
    """Kill the child and its entire process tree (cross-platform via psutil)."""
    import psutil
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
    except psutil.NoSuchProcess:
        pass
    except (PermissionError, OSError) as e:
        logger.debug("Could not terminate process tree: %s", e, exc_info=True)
        try:
            proc.kill()
        except Exception as e2:
            logger.debug("Could not kill process: %s", e2, exc_info=True)

    if escalate:
        # Give the process 5s to exit after SIGTERM, then SIGKILL
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                parent = psutil.Process(proc.pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
            except psutil.NoSuchProcess:
                pass
            except (PermissionError, OSError) as e:
                logger.debug("Could not kill process tree: %s", e, exc_info=True)
                try:
                    proc.kill()
                except Exception as e2:
                    logger.debug("Could not kill process: %s", e2, exc_info=True)


def _load_config() -> dict:
    """Load code_execution config without importing the interactive CLI.

    This helper is called while building the module-level execute_code schema
    during tool discovery.  Importing ``cli`` here pulls prompt_toolkit/Rich and
    a large chunk of the classic REPL onto every agent startup path, including
    ``pilotage --tui`` where it is never used.  Read the lightweight raw config
    instead; the config layer already caches by (mtime, size), and an absent
    key cleanly falls back to DEFAULT_EXECUTION_MODE.
    """
    try:
        from pilotage_cli.config import read_raw_config

        cfg = read_raw_config().get("code_execution", {})
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Execution mode resolution (strict vs project)
# ---------------------------------------------------------------------------

# Valid values for code_execution.mode. Kept as a module constant so tests
# and the config layer can reference the canonical set.
EXECUTION_MODES = ("project", "strict")
DEFAULT_EXECUTION_MODE = "project"


def _get_execution_mode() -> str:
    """Return the active execute_code mode — 'project' or 'strict'.

    Reads ``code_execution.mode`` from config.yaml; invalid values fall back
    to ``DEFAULT_EXECUTION_MODE`` ('project') with a log warning.

    Mode semantics:
      - ``project`` (default): scripts run in the session's working directory
        with the active virtual environment's python, so project dependencies
        (pandas, torch, project packages) and files resolve naturally.
      - ``strict``: scripts run in an isolated temp directory with
        ``sys.executable`` (pilotage-agent's python). Reproducible and the
        interpreter is guaranteed to work, but project deps and relative paths
        won't resolve.

    Env scrubbing and tool whitelist apply identically in both modes.
    """
    cfg_value = str(_load_config().get("mode", DEFAULT_EXECUTION_MODE)).strip().lower()
    if cfg_value in EXECUTION_MODES:
        return cfg_value
    logger.warning(
        "Ignoring code_execution.mode=%r (expected one of %s), falling back to %r",
        cfg_value, EXECUTION_MODES, DEFAULT_EXECUTION_MODE,
    )
    return DEFAULT_EXECUTION_MODE


# Shared budget for the two interpreter-probe caches below. Success-only
# dict caches (FIFO-evicted at the cap) rather than lru_cache: a transient
# probe failure (fork pressure, 5s timeout on a loaded host) must not stick
# for the process lifetime.
_PROBE_CACHE_MAX = 32
_usable_python_cache: dict = {}
_python_prefix_cache: dict = {}

# Interpreter paths already reported as outside the Pilotage environment —
# dedupes the exclusion log to once per path per process.
_external_env_logged: set = set()


def _cache_probe_result(cache: dict, key: str, value):
    """Insert into a bounded probe cache, FIFO-evicting at the cap."""
    if len(cache) >= _PROBE_CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _is_usable_python(python_path: str) -> bool:
    """Check whether a candidate Python interpreter is usable for execute_code.

    Requires Python 3.8+ (f-strings and stdlib modules the RPC stubs need).
    Successful probes are cached per interpreter path; failures are retried
    (a sticky False would silently pin project mode to sys.executable).
    """
    cached = _usable_python_cache.get(python_path)
    if cached is not None:
        return cached
    result = _probe_python(
        python_path,
        "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)",
    )
    if result is None:
        return False
    usable = result.returncode == 0
    _cache_probe_result(_usable_python_cache, python_path, usable)
    return usable


def _probe_python(python_path: str, code: str, *, text: bool = False):
    """Run ``python_path -c code`` with the standard interpreter-probe guards.

    Returns the ``CompletedProcess``, or ``None`` when the interpreter is
    missing, can't be spawned, or hangs past the 5s timeout.
    """
    try:
        from agent.delegation_context import delegated_child_subprocess_env

        return subprocess.run(
            [python_path, "-c", code],
            timeout=5,
            capture_output=True,
            text=text,
            creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
            stdin=subprocess.DEVNULL,
            env=delegated_child_subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return None


def _python_environment_prefix(python_path: str) -> str:
    """Return the resolved ``sys.prefix`` reported by *python_path*, if any.

    Successful probes are cached per interpreter path (bounded, FIFO-evicted).
    Failures are NOT cached: a transient probe failure (fork pressure, 5s
    timeout on a loaded host) must not stick for the process lifetime — a
    sticky empty result would silently drop the pilotage root from every
    subsequent execute_code call's PYTHONPATH.
    """
    cached = _python_prefix_cache.get(python_path)
    if cached is not None:
        return cached
    result = _probe_python(python_path, "import sys; print(sys.prefix)", text=True)
    if result is not None and result.returncode == 0 and result.stdout.strip():
        prefix = os.path.realpath(result.stdout.strip())
        _cache_probe_result(_python_prefix_cache, python_path, prefix)
        return prefix
    return ""


def _uses_pilotage_python_environment(python_path: str) -> bool:
    """Whether *python_path* belongs to Pilotage's active Python environment.

    Short-circuits when *python_path* IS the running interpreter (by path or
    realpath) — no subprocess probe on the default strict-mode path, and no
    way for a flaky probe of ``sys.executable`` itself to break the invariant
    that repo-root modules are importable in strict mode.  The realpath leg
    also covers venvs whose bin/python resolves to the same binary (e.g.
    ``uv run`` setting VIRTUAL_ENV without changing sys.prefix).
    """
    if python_path == sys.executable or (
        os.path.realpath(python_path) == os.path.realpath(sys.executable)
    ):
        return True
    return _python_environment_prefix(python_path) == os.path.realpath(sys.prefix)


def _resolve_child_python(mode: str) -> str:
    """Pick the Python interpreter for the execute_code subprocess.

    In ``strict`` mode, always ``sys.executable`` — guaranteed to work and
    keeps behavior fully reproducible across sessions.

    In ``project`` mode, prefer the user's active virtualenv/conda env's
    python so ``import pandas`` etc. work. Falls back to ``sys.executable``
    if no venv is detected, the candidate binary is missing/not executable,
    or it fails a Python 3.8+ version check.
    """
    if mode != "project":
        return sys.executable

    if _IS_WINDOWS:
        exe_names = ("python.exe", "python3.exe")
        subdirs = ("Scripts",)
    else:
        exe_names = ("python", "python3")
        subdirs = ("bin",)

    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        root = os.environ.get(var, "").strip()
        if not root:
            continue
        for subdir in subdirs:
            for exe in exe_names:
                candidate = os.path.join(root, subdir, exe)
                if not (os.path.isfile(candidate) and os.access(candidate, os.X_OK)):
                    continue
                if _is_usable_python(candidate):
                    return candidate
                # Found the interpreter but it failed the version check —
                # log once and fall through to sys.executable.
                logger.info(
                    "execute_code: skipping %s=%s (Python version < 3.8 or broken). "
                    "Using sys.executable instead.", var, candidate,
                )
                return sys.executable

    return sys.executable


def _resolve_child_cwd(mode: str, staging_dir: str, task_id: str = "") -> str:
    """Resolve the working directory for the execute_code subprocess.

    - ``strict``: the staging tmpdir (today's behavior).
    - ``project``: the session's own cwd — its per-session cwd record
      (written after every completed terminal command), then the raw
      per-session cwd override registered via ``session.cwd.set`` /
      ``register_task_env_overrides``, then the session's TERMINAL_CWD
      (same as the terminal tool), or ``os.getcwd()`` if none points at a
      real dir. Falls back to the staging tmpdir as a last resort so we
      never invoke Popen with a nonexistent cwd.

    This mirrors the resolution ladder file tools and the terminal use
    (record → registered override → TERMINAL_CWD), so all file-writing
    paths within a session agree on the working directory.
    """
    if mode != "project":
        return staging_dir
    if task_id:
        # 1. The session's cwd record — IS the session's `cd` state.
        try:
            from tools.terminal_tool import get_session_cwd

            recorded = get_session_cwd(task_id)
        except Exception:
            recorded = None
        if recorded and os.path.isdir(recorded):
            return recorded
        # 2. Registered workspace override (session.cwd.set → gateway/TUI/ACP).
        try:
            from tools.file_tools import _registered_task_cwd_override

            session_cwd = _registered_task_cwd_override(task_id)
        except Exception:
            session_cwd = None
        if session_cwd and os.path.isdir(session_cwd):
            return session_cwd
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    if raw:
        expanded = os.path.expanduser(raw)
        if os.path.isdir(expanded):
            return expanded
    here = os.getcwd()
    if os.path.isdir(here):
        return here
    return staging_dir


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

# Per-tool documentation lines for the execute_code description.
# Ordered to match the canonical display order.
_TOOL_DOC_LINES = [
    ("web_search",
     "  web_search(query: str, limit: int = 5) -> dict\n"
     "    Returns {\"data\": {\"web\": [{\"url\", \"title\", \"description\"}, ...]}}"),
    ("web_extract",
     "  web_extract(urls: list[str], char_limit: int = None) -> dict\n"
     "    Returns {\"results\": [{\"url\", \"title\", \"content\", \"error\"}, ...]} where content is markdown.\n"
     "    No LLM summarization. Pages over char_limit (default 15000) are head+tail truncated; full text stored on disk (path in the content footer)."),
    ("read_file",
     "  read_file(path: str, offset: int = 1, limit: int = 2000) -> dict\n"
     "    Lines are 1-indexed. Returns {\"content\": \"...\", \"total_lines\": N}"),
    ("write_file",
     "  write_file(path: str, content: str) -> dict\n"
     "    Always overwrites the entire file."),
    ("search_files",
     "  search_files(pattern: str, target=\"content\", path=\".\", file_glob=None, limit=50) -> dict\n"
     "    target: \"content\" (search inside files) or \"files\" (find files by name). Returns {\"matches\": [...]}"),
    ("patch",
     "  patch(path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict\n"
     "    Replaces old_string with new_string in the file."),
    ("terminal",
     "  terminal(command: str, timeout=None, workdir=None) -> dict\n"
     "    Foreground only (no background/pty). Returns {\"output\": \"...\", \"exit_code\": N}"),
]


def build_execute_code_schema(enabled_sandbox_tools: set = None,
                              mode: str = None) -> dict:
    """Build the execute_code schema with description listing only enabled tools.

    When tools are disabled via ``pilotage tools`` (e.g. web is turned off),
    the schema description should NOT mention web_search / web_extract —
    otherwise the model thinks they are available and keeps trying to use them.

    ``mode`` controls the working-directory sentence in the description:
      - ``'strict'``: scripts run in a temp dir (not the session's CWD)
      - ``'project'`` (default): scripts run in the session's CWD with the
        active venv's python
    If ``mode`` is None, the current ``code_execution.mode`` config is read.
    """
    if enabled_sandbox_tools is None:
        enabled_sandbox_tools = SANDBOX_ALLOWED_TOOLS
    if mode is None:
        mode = _get_execution_mode()

    # Build tool documentation lines for only the enabled tools
    tool_lines = "\n".join(
        doc for name, doc in _TOOL_DOC_LINES if name in enabled_sandbox_tools
    )

    # Build example import list from enabled tools
    import_examples = [n for n in ("web_search", "terminal") if n in enabled_sandbox_tools]
    if not import_examples:
        import_examples = sorted(enabled_sandbox_tools)[:2]
    if import_examples:
        import_str = ", ".join(import_examples) + ", ..."
    else:
        import_str = "..."

    # Mode-specific CWD guidance. Project mode is the default and matches
    # terminal()'s filesystem/interpreter; strict mode retains the isolated
    # temp-dir staging and pilotage-agent's own python.
    if mode == "strict":
        cwd_note = (
            "Scripts run in their own temp dir, not the session's CWD — use absolute paths "
            "(os.path.expanduser('~/.pilotage/.env')) or terminal()/read_file() for user files."
        )
    else:
        cwd_note = (
            "Scripts run in the session's working directory with the active venv's python, "
            "so project deps (pandas, etc.) and relative paths work like in terminal()."
        )

    description = (
        "Run a Python script that calls Pilotage tools programmatically. "
        "Use when you need 3+ tool calls with logic between them: "
        "filtering/reducing large outputs before they enter context, "
        "conditional branching, or loops (N pages/files, retry on failure). "
        "Use normal tool calls for single calls, results you must reason "
        "over in full, or anything needing user interaction.\n\n"
        f"Available via `from pilotage_tools import ...`:\n\n"
        f"{tool_lines}\n\n"
        "Limits: 5-minute timeout, 50KB stdout cap, max 50 tool calls per script. "
        "terminal() is foreground-only (no background or pty).\n\n"
        f"{cwd_note}\n\n"
        "Print your final result to stdout; stdlib (json, re, csv, datetime, ...) "
        "is available for processing.\n\n"
        "Built-in helpers (no import): json_parse(text) — tolerant json.loads for "
        "terminal() output; shell_quote(s) — shlex.quote for dynamic shell args; "
        "retry(fn, max_attempts=3, delay=2) — exponential backoff for transient failures."
    )

    return {
        "name": "execute_code",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute. Import tools with "
                        f"`from pilotage_tools import {import_str}` "
                        "and print your final result to stdout."
                    ),
                },
            },
            "required": ["code"],
        },
    }


# Default schema used at registration time (all sandbox tools listed,
# current configured mode).  model_tools.py rebuilds per-session anyway.
EXECUTE_CODE_SCHEMA = build_execute_code_schema()


# --- Registry ---
from tools.registry import registry, tool_error


def _execute_code_handler(args: dict, **kwargs) -> str:
    """Recover misdirected calls before dispatching to ``execute_code``.

    Models sometimes reuse terminal's ``command`` argument or send a
    non-string ``code`` payload; both get an actionable redirect instead
    of a generic failure.
    """
    # Help models recover when they reuse terminal's ``command`` argument.
    if "code" not in args and "command" in args:
        logger.warning(
            "execute_code received 'command' instead of the required 'code' argument"
        )
        return tool_error(
            "execute_code received a 'command' parameter, but it requires "
            "Python source in 'code'. Use terminal(command=...) for shell "
            "commands; for Python, retry as execute_code(code=...)."
        )

    code = args.get("code", "")
    if code is not None and not isinstance(code, str):
        # A non-string payload (int, dict, list) would otherwise surface as
        # a generic AttributeError from code.strip() — redirect instead.
        return tool_error(
            f"execute_code received a {type(code).__name__} in 'code', but it "
            "requires Python source as a string. Retry as "
            "execute_code(code=\"...\")."
        )

    return execute_code(
        code=code or "",
        task_id=kwargs.get("task_id"),
        enabled_tools=kwargs.get("enabled_tools"),
    )


registry.register(
    name="execute_code",
    toolset="code_execution",
    schema=EXECUTE_CODE_SCHEMA,
    handler=_execute_code_handler,
    check_fn=check_sandbox_requirements,
    emoji="🐍",
    max_result_size_chars=100_000,
)
