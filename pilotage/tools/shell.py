"""One bash session, kept alive between commands.

Copied from Hermes (tools/environments/base.py, tools/environments/local.py and
two helpers from tools/terminal_tool.py), reduced to the single backend we run:
bash, on the machine the agent runs on, on Linux. Hermes carries eight backends
and about two thousand lines of Windows/MSYS recovery; none of it is reachable
from an Ubuntu container, so none of it is here.

There is no long-lived shell process. Every command spawns its own bash, and the
session is kept in a *snapshot*: the previous command dumps its exports,
functions and aliases into a file, and the next command sources it before it
starts. That is why ``export FOO=1`` and ``cd /somewhere`` are still true on the
next call. The working directory travels separately, printed inside a
session-unique marker on stdout and parsed back off.

The awkward parts are all load-bearing, and each one is a bug Hermes already
paid for:

* The snapshot is written to a ``mktemp`` file and ``mv``'d over the real path.
  Two commands running at once would otherwise source a half-written file.
* Functions are filtered by *name*, not by line. Filtering ``declare -f`` output
  with grep strips a function's header and leaves its body behind, and the
  orphaned ``{ … }`` breaks every later command with exit 127.
* Output is drained with ``select()`` on a short poll, not by iterating the
  pipe. A backgrounded grandchild inherits the write end and holds the pipe open
  after bash itself has exited; iterating would hang for as long as it lives.
* Killing is by process *group*. We spawn into a new session, so signalling only
  the bash wrapper leaves its children running with PPID=1.
"""

from __future__ import annotations

import codecs
import logging
import os
import select
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from .subprocess_env import build_subprocess_env

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120
SNAPSHOT_TIMEOUT_SECONDS = 30

# What "no limit" means to the output collector. Internal readers — the ones
# feeding a patch engine or a code-execution reply — must keep every byte,
# because truncating them corrupts data rather than shortening a display.
UNBOUNDED_CAPTURE_CHARS = 2**63 - 1


# ---------------------------------------------------------------------------
# Bounded output
# ---------------------------------------------------------------------------


class BoundedOutput:
    """Keep the first 40% and the last 60% of a stream, and nothing else.

    Bounding happens while the stream is drained, not after: a command that
    prints a gigabyte cannot make the agent run out of memory, whatever the
    caller does with the result afterwards.

    Head *and* tail, because both ends carry the answer depending on what was
    run. A directory listing says what it has to say at the top; a build says it
    at the very bottom, after ten thousand lines of compilation.
    """

    def __init__(self, max_chars: int):
        self.max_chars = max(1, int(max_chars))
        self._head_limit = int(self.max_chars * 0.4)
        self._tail_limit = self.max_chars - self._head_limit
        self._head: List[str] = []
        self._tail: deque = deque()
        self._head_chars = 0
        self._tail_chars = 0
        self._total_chars = 0
        self._lock = threading.Lock()

    @property
    def total_chars(self) -> int:
        with self._lock:
            return self._total_chars

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            text_len = len(text)
            self._total_chars += text_len
            start = 0

            if self._head_chars < self._head_limit:
                take = min(self._head_limit - self._head_chars, text_len)
                if take:
                    self._head.append(text[:take])
                    self._head_chars += take
                    start = take

            remaining = text_len - start
            if remaining <= 0 or self._tail_limit <= 0:
                return
            if remaining >= self._tail_limit:
                self._tail.clear()
                self._tail.append(text[-self._tail_limit :])
                self._tail_chars = self._tail_limit
                return

            chunk = text[start:]
            self._tail.append(chunk)
            self._tail_chars += len(chunk)
            while self._tail_chars > self._tail_limit:
                excess = self._tail_chars - self._tail_limit
                first = self._tail[0]
                if len(first) <= excess:
                    self._tail.popleft()
                    self._tail_chars -= len(first)
                else:
                    self._tail[0] = first[excess:]
                    self._tail_chars -= excess

    def render(self, *, suffix: str = "") -> str:
        """The retained text, never longer than ``max_chars``, suffix included.

        The notice that says how much was dropped is itself part of the budget,
        so its length changes the number it has to print. Four passes is enough
        for that to settle; the loop stops as soon as it stops changing.
        """
        with self._lock:
            if len(suffix) >= self.max_chars:
                return suffix[-self.max_chars :]

            head = "".join(self._head)
            tail = "".join(self._tail)
            available = self.max_chars - len(suffix)
            if self._total_chars <= available:
                return head + tail + suffix

            notice = ""
            for _ in range(4):
                content_budget = max(0, available - len(notice))
                head_chars = int(content_budget * 0.4)
                tail_chars = content_budget - head_chars
                omitted = max(0, self._total_chars - head_chars - tail_chars)
                updated = (
                    f"\n\n... [OUTPUT TRUNCATED - {omitted:,} chars omitted "
                    f"out of {self._total_chars:,} total] ...\n\n"
                )
                if updated == notice:
                    break
                notice = updated

            content_budget = max(0, available - len(notice))
            head_chars = int(content_budget * 0.4)
            tail_chars = content_budget - head_chars
            rendered_tail = tail[-tail_chars:] if tail_chars else ""
            return head[:head_chars] + notice[:available] + rendered_tail + suffix


# ---------------------------------------------------------------------------
# Shell parsing
# ---------------------------------------------------------------------------


def read_shell_token(command: str, start: int) -> Tuple[str, int]:
    """Read one shell token from *start*, keeping its quoting intact."""
    i = start
    n = len(command)

    while i < n:
        ch = command[i]
        if ch.isspace() or ch in ";|&()":
            break
        if ch == "'":
            i += 1
            while i < n and command[i] != "'":
                i += 1
            if i < n:
                i += 1
            continue
        if ch == '"':
            i += 1
            while i < n:
                inner = command[i]
                if inner == "\\" and i + 1 < n:
                    i += 2
                    continue
                if inner == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        i += 1

    return command[start:i], i


def rewrite_compound_background(command: str) -> str:
    """Turn ``A && B &`` into ``A && { B & }``.

    Bash binds ``&&`` tighter than ``&``, so it forks a subshell for the whole
    compound and backgrounds *that*. Inside the subshell ``B`` runs in the
    foreground, so the subshell waits for it — forever, when ``B`` is a server.
    The leaked subshell also holds our stdout pipe open. A brace group runs in
    the current shell instead, backgrounds ``B`` properly, and exits.

    Redirections that merely contain an ampersand (``&>``, ``2>&1``) are left
    alone, as is a plain ``cmd &``, which never had the problem. Content inside
    quotes, parentheses and brace groups is skipped, which also makes the
    rewrite idempotent on its own output.
    """
    n = len(command)
    i = 0
    paren_depth = 0
    brace_depth = 0
    # Just after the most recent depth-0 `&&` / `||`; -1 when none is open.
    last_chain_op_end = -1
    rewrites: List[Tuple[int, int]] = []

    while i < n:
        ch = command[i]

        # A newline ends the statement, so check it before skipping whitespace.
        if ch == "\n" and paren_depth == 0 and brace_depth == 0:
            last_chain_op_end = -1
            i += 1
            continue

        if ch.isspace():
            i += 1
            continue

        if ch == "#":
            nl = command.find("\n", i)
            if nl == -1:
                break
            i = nl
            continue

        if ch == "\\" and i + 1 < n:
            i += 2
            continue

        if ch in {"'", '"'}:
            _, next_i = read_shell_token(command, i)
            i = max(next_i, i + 1)
            continue

        if ch == "(":
            paren_depth += 1
            i += 1
            continue

        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue

        # Bash requires whitespace after `{` for a group, which is what tells
        # a group apart from brace expansion here.
        if ch == "{" and i + 1 < n and (command[i + 1].isspace() or command[i + 1] == "\n"):
            brace_depth += 1
            i += 1
            continue
        if ch == "}" and brace_depth > 0:
            brace_depth -= 1
            last_chain_op_end = -1
            i += 1
            continue

        if paren_depth > 0 or brace_depth > 0:
            i += 1
            continue

        if command.startswith("&&", i) or command.startswith("||", i):
            last_chain_op_end = i + 2
            i += 2
            continue

        if ch == ";":
            last_chain_op_end = -1
            i += 1
            continue

        if ch == "|":
            last_chain_op_end = -1
            i += 1
            continue

        if ch == "&":
            if i + 1 < n and command[i + 1] == ">":
                i += 2  # `&>` redirect
                continue
            j = i - 1
            while j >= 0 and command[j].isspace():
                j -= 1
            if j >= 0 and command[j] in "<>":
                i += 1  # `>&` / `<&` fd target
                continue
            if last_chain_op_end >= 0:
                rewrites.append((last_chain_op_end, i))
            last_chain_op_end = -1
            i += 1
            continue

        _, next_i = read_shell_token(command, i)
        i = max(next_i, i + 1)

    if not rewrites:
        return command

    # Back to front, so the indices collected earlier stay valid.
    result = command
    for chain_end, amp_pos in reversed(rewrites):
        insert_pos = chain_end
        while insert_pos < amp_pos and result[insert_pos].isspace():
            insert_pos += 1
        prefix = result[:insert_pos]
        middle = result[insert_pos:amp_pos]
        suffix = result[amp_pos + 1 :]
        result = prefix + "{ " + middle + "& }" + suffix

    return result


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _close_quietly(stream) -> None:
    """Close a pipe we are finished with, whatever state it is in."""
    if stream is None:
        return
    try:
        stream.close()
    except Exception:  # noqa: BLE001
        pass


def _pipe_stdin(proc: subprocess.Popen, data: str) -> None:
    """Feed *data* to the process on a thread, so a full pipe cannot deadlock.

    The write goes through ``proc.stdin.buffer`` rather than the text wrapper,
    encoding here instead. Text mode would translate newlines, which corrupts
    every file the model writes through this path.

    ``surrogateescape`` is the exact inverse of the decode, so bytes that were
    never valid UTF-8 survive the round trip. A string carrying a surrogate
    outside that range cannot be encoded at all; that failure is recorded and
    stdin is still closed, because a child left waiting on EOF never returns.
    """
    errors: List[BaseException] = []
    proc._pilotage_stdin_errors = errors  # type: ignore[attr-defined]

    def _write():
        if proc.stdin is None:
            errors.append(RuntimeError("process stdin unavailable"))
            return
        # Resolve the target before encoding: a failed encode must still reach
        # the finally-close below.
        target = getattr(proc.stdin, "buffer", proc.stdin)
        try:
            raw = data.encode("utf-8", "surrogateescape") if isinstance(data, str) else data
            written = target.write(raw)
            if written != len(raw):
                raise RuntimeError(f"short stdin write: {written} of {len(raw)} bytes")
        except (BrokenPipeError, OSError):
            pass  # the child closed stdin early, which is normal
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            errors.append(exc)
        finally:
            try:
                target.close()
            except Exception:  # noqa: BLE001
                pass

    thread = threading.Thread(target=_write, daemon=True)
    proc._pilotage_stdin_thread = thread  # type: ignore[attr-defined]
    thread.start()


def find_bash() -> str:
    """The bash to run commands with."""
    return (
        shutil.which("bash")
        or ("/usr/bin/bash" if os.path.isfile("/usr/bin/bash") else None)
        or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
        or os.environ.get("SHELL")
        or "/bin/sh"
    )


def _cwd_usable(path: str) -> bool:
    """True when this process can actually start a child in *path*.

    ``isdir`` is not enough. ``stat('/root')`` succeeds for an ordinary user —
    only ``/`` needs search permission for that — but spawning with that
    directory then fails with a permission error. Checking for execute access
    up front lets the caller fall back instead of failing every command.
    """
    return os.path.isdir(path) and os.access(path, os.X_OK)


def resolve_safe_cwd(cwd: str) -> str:
    """*cwd*, or the nearest directory above it we can actually enter.

    A command that deletes its own working directory would otherwise wedge
    every command after it: the failure happens inside the spawn, before bash
    ever starts, so there is nothing to report and nothing to recover from.
    """
    if cwd and _cwd_usable(cwd):
        return cwd
    if cwd and os.path.isdir(cwd):
        logger.warning(
            "Working directory %r exists but this user cannot enter it — "
            "falling back to the nearest one that works.",
            cwd,
        )
    parent = os.path.dirname(cwd) if cwd else ""
    while parent:
        if _cwd_usable(parent):
            return parent
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            break
        parent = next_parent
    return tempfile.gettempdir()


# Sourced before the snapshot is taken. Login shells read the first two, but
# installers that write to .bashrc (nvm, asdf, pyenv) put their PATH line below
# the interactivity guard Debian ships, so a non-interactive login shell alone
# would miss them.
SHELL_INIT_FILES = ("~/.profile", "~/.bash_profile", "~/.bashrc")


def _shell_init_files() -> List[str]:
    resolved: List[str] = []
    for raw in SHELL_INIT_FILES:
        try:
            path = os.path.expandvars(os.path.expanduser(raw))
        except Exception:  # noqa: BLE001
            continue
        if path and os.path.isfile(path):
            resolved.append(path)
    return resolved


def _prepend_shell_init(cmd_string: str, files: List[str]) -> str:
    """Source each file, guarded, so a broken rc file cannot stop the session."""
    if not files:
        return cmd_string
    parts = ["set +e"]
    for path in files:
        safe = path.replace("'", "'\\''")
        parts.append(f"[ -r '{safe}' ] && . '{safe}' 2>/dev/null || true")
    return "\n".join(parts) + "\n" + cmd_string


def _temp_dir() -> str:
    """A writable POSIX directory for the session's snapshot."""
    for name in ("TMPDIR", "TMP", "TEMP"):
        candidate = os.environ.get(name)
        if candidate and candidate.startswith("/"):
            return candidate.rstrip("/") or "/"
    if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK | os.X_OK):
        return "/tmp"
    candidate = tempfile.gettempdir()
    if candidate.startswith("/"):
        return candidate.rstrip("/") or "/"
    return "/tmp"


def _quote_cwd_for_cd(cwd: str) -> str:
    """Quote a ``cd`` target without killing ``~`` expansion."""
    if cwd == "~":
        return cwd
    if cwd == "~/":
        return "$HOME"
    if cwd.startswith("~/"):
        return f"$HOME/{shlex.quote(cwd[2:])}"
    return shlex.quote(cwd)


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class Shell:
    """A bash session on this machine.

    ``execute`` is the whole interface, and the extracted Hermes file stack
    uses it too. This keeps file operations on the same persistent cwd and
    environment as terminal calls.
    """

    def __init__(self, cwd: str = "", timeout: int = DEFAULT_TIMEOUT_SECONDS,
                 env: Optional[Dict[str, str]] = None):
        self.cwd = os.path.abspath(os.path.expanduser(cwd)) if cwd else os.getcwd()
        self.timeout = timeout
        self.env = dict(env or {})

        self._session_id = uuid.uuid4().hex[:12]
        temp_dir = _temp_dir()
        self._snapshot_path = f"{temp_dir}/pilotage-snap-{self._session_id}.sh"
        self._marker = f"__PILOTAGE_CWD_{self._session_id}__"
        self._snapshot_ready = False
        self._closed = False

        self.init_session()

    # -- session snapshot ---------------------------------------------------

    @property
    def _tmp_template(self) -> str:
        return shlex.quote(self._snapshot_path + ".tmp.XXXXXXXXXX")

    def _snapshot_dump(self) -> str:
        """The shell that fills ``$__pilotage_snap_tmp`` with the session.

        Written after *every* command, not only at startup, because the things
        worth carrying are not all exported variables. ``nvm``, ``pyenv`` and
        ``conda activate`` are shell functions; saving exports alone would make
        them work on the first command of a session and vanish on the second.
        """
        tmp = '"$__pilotage_snap_tmp"'
        return "\n".join(
            [
                f"{{ export -p || true; }} > {tmp}",
                # Select the wanted function names first, then dump only those.
                # Filtering the dump itself by line strips a header and leaves
                # its body behind, and the orphaned `{ … }` makes every later
                # command fail with exit 127.
                #
                # The non-empty guard matters: a bare `declare -f` with no names
                # dumps every function there is, so an empty list would leak
                # exactly the private helpers this is meant to drop.
                "__pilotage_fns=$(declare -F | awk '{print $3}' | grep -vE '^_[^_]') || true",
                f'[ -n "$__pilotage_fns" ] && declare -f $__pilotage_fns >> {tmp} 2>/dev/null || true',
                f"alias -p >> {tmp} 2>/dev/null || true",
                # Aliases are not expanded in a non-interactive shell unless
                # this is set, so saving them without it would save nothing.
                f"echo 'shopt -s expand_aliases' >> {tmp}",
                # A profile that set -e or -u must not impose it on the agent's
                # commands: one failing grep would end the whole command.
                f"echo 'set +e' >> {tmp}",
                f"echo 'set +u' >> {tmp}",
            ]
        )

    def init_session(self) -> None:
        """Capture the login shell's environment once, into the snapshot file.

        Until this succeeds every command pays for a login shell of its own, so
        it is worth doing, but it is not worth failing over: a machine with a
        hostile rc file still gets a working agent, only a slower one.
        """
        quoted_snap = shlex.quote(self._snapshot_path)
        tmp = '"$__pilotage_snap_tmp"'
        bootstrap = (
            f"umask 077\n"
            f"__pilotage_snap_tmp=$(mktemp {self._tmp_template}) || exit 1\n"
            f"{self._snapshot_dump()}\n"
            f"mv -f {tmp} {quoted_snap} || rm -f {tmp}\n"
            # Profile scripts are allowed to `cd`; go back to where we were
            # asked to be before reporting the directory.
            f"builtin cd -- {_quote_cwd_for_cd(self.cwd)} 2>/dev/null || true\n"
            f"printf '\\n{self._marker}%s{self._marker}\\n' \"$(pwd -P)\"\n"
        )
        try:
            proc = self._run_bash(bootstrap, login=True, timeout=SNAPSHOT_TIMEOUT_SECONDS)
            result = self._wait_for_process(proc, timeout=SNAPSHOT_TIMEOUT_SECONDS)
            code = int(result.get("returncode") or 0)
            if code != 0:
                raise RuntimeError(f"snapshot bootstrap exited with {code}")
            self._snapshot_ready = True
            self._update_cwd(result)
            logger.info("Shell session ready (id=%s, cwd=%s)", self._session_id, self.cwd)
        except Exception as exc:  # noqa: BLE001 - a slow shell beats no shell
            self._snapshot_ready = False
            logger.warning(
                "Shell session snapshot failed (id=%s): %s — every command will "
                "run in a login shell instead.",
                self._session_id,
                exc,
            )

    # -- command wrapping ---------------------------------------------------

    def _wrap_command(self, command: str, cwd: str) -> str:
        """The script bash actually runs: restore, cd, run, save, report."""
        escaped = command.replace("'", "'\\''")
        quoted_snap = shlex.quote(self._snapshot_path)
        tmp = '"$__pilotage_snap_tmp"'

        parts: List[str] = []

        if self._snapshot_ready:
            # Silenced because some bash builds echo `declare -x` lines while
            # sourcing, which would put sixty lines of environment in front of
            # every single answer.
            parts.append(f"source {quoted_snap} >/dev/null 2>&1 || true")

        # `--` so a directory whose name starts with a hyphen is not read as an
        # option; 126 is the conventional "could not execute" code.
        parts.append(f"builtin cd -- {_quote_cwd_for_cd(cwd)} || exit 126")
        parts.append(f"eval '{escaped}'")
        parts.append("__pilotage_ec=$?")
        # The snapshot can hold anything the environment held, so it is written
        # privately — without changing the umask the command itself ran under.
        parts.append("umask 077")

        if self._snapshot_ready:
            # Assemble, then move over the real path, so a command that runs
            # while another finishes never sources a half-written file. The
            # temp name comes from mktemp rather than $$, which is shared by
            # backgrounded subshells. The redirection is attached to a brace
            # group so the temp variable expands in the same shell that later
            # expands the `mv`.
            parts.append(
                f"__pilotage_snap_tmp=$(mktemp {self._tmp_template}) && {{\n"
                f"{self._snapshot_dump()}\n"
                f"mv -f {tmp} {quoted_snap}\n"
                f"}} 2>/dev/null || rm -f {tmp} 2>/dev/null || true"
            )

        # The leading newline guarantees the marker starts its own line even
        # when the command printed no trailing newline; it is stripped again on
        # the way out.
        parts.append(f"printf '\\n{self._marker}%s{self._marker}\\n' \"$(pwd -P)\"")
        parts.append("exit $__pilotage_ec")

        return "\n".join(parts)

    # -- spawning -----------------------------------------------------------

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = DEFAULT_TIMEOUT_SECONDS,
                  stdin_data: Optional[str] = None) -> subprocess.Popen:
        bash = find_bash()
        if login:
            cmd_string = _prepend_shell_init(cmd_string, _shell_init_files())
        args = [bash, "-l", "-c", cmd_string] if login else [bash, "-c", cmd_string]

        # A terminal command is model-controlled. Keep ordinary operator
        # environment values available, but never inherit Pilotage's own
        # provider, channel, or bridge credentials. This is the same targeted
        # subprocess scrub used by current Hermes.
        run_env = build_subprocess_env(extra=self.env)

        self._ensure_cwd()

        proc = subprocess.Popen(
            args,
            text=True,
            env=run_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            start_new_session=True,
            cwd=self.cwd,
        )
        try:
            proc._pilotage_pgid = os.getpgid(proc.pid)  # type: ignore[attr-defined]
        except (ProcessLookupError, OSError):
            pass

        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)

        return proc

    # -- waiting ------------------------------------------------------------

    def _wait_for_process(self, proc: subprocess.Popen, timeout: int,
                          capture_limit: Optional[int] = None) -> Dict[str, Any]:
        """Wait for the command, draining its output as it arrives."""
        output = BoundedOutput(capture_limit or UNBOUNDED_CAPTURE_CHARS)

        # Bytes are read in fixed chunks, so a multi-byte character can be split
        # across two reads. An incremental decoder holds the partial sequence;
        # `replace` matches how the pipe itself was opened, so binary output is
        # marked rather than lost.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def _drain():
            stream = proc.stdout
            if stream is None:
                return
            try:
                fd = stream.fileno()
            except Exception:  # noqa: BLE001
                return
            idle_after_exit = 0
            try:
                while True:
                    try:
                        ready, _, _ = select.select([fd], [], [], 0.1)
                    except (ValueError, OSError):
                        break  # already closed
                    if ready:
                        try:
                            chunk = os.read(fd, 4096)
                        except (ValueError, OSError):
                            break
                        if not chunk:
                            break  # real end of file: every writer is gone
                        output.append(decoder.decode(chunk))
                        idle_after_exit = 0
                    elif proc.poll() is not None:
                        # Bash has exited and the pipe has been quiet for
                        # ~100ms. Two more turns to catch a buffered tail, then
                        # stop — anything still holding the pipe open is a
                        # backgrounded grandchild we are not waiting for.
                        idle_after_exit += 1
                        if idle_after_exit >= 3:
                            break
            finally:
                try:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        output.append(tail)
                except Exception:  # noqa: BLE001
                    pass

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()
        deadline = time.monotonic() + timeout

        try:
            # Start polling fast so `pwd` comes back in milliseconds, then back
            # off so a half-hour build is not paid for in wakeups.
            poll_sleep = 0.005
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    self._kill_process(proc)
                    drain_thread.join(timeout=2)
                    _close_quietly(proc.stdout)
                    suffix = f"\n[Command timed out after {timeout}s]"
                    rendered = output.render(suffix=suffix)
                    if output.total_chars == 0:
                        rendered = rendered.lstrip()
                    return {"output": rendered, "returncode": 124}
                time.sleep(poll_sleep)
                if poll_sleep < 0.2:
                    poll_sleep = min(poll_sleep * 1.5, 0.2)
        except (KeyboardInterrupt, SystemExit):
            # We spawned into a new session, so letting this propagate would
            # leave the command running with PPID=1 after the agent is gone.
            try:
                self._kill_process(proc)
                drain_thread.join(timeout=2)
                _close_quietly(proc.stdout)
            except Exception:  # noqa: BLE001
                pass
            raise

        drain_thread.join(timeout=2)
        _close_quietly(proc.stdout)

        # Join the writer before reading its errors: a child that exits without
        # reading stdin can otherwise finish first and take a recorded failure
        # with it, turning a broken write into a silent success.
        stdin_thread = getattr(proc, "_pilotage_stdin_thread", None)
        if stdin_thread is not None:
            stdin_thread.join(timeout=5)

        rendered = output.render()
        result: Dict[str, Any] = {"output": rendered, "returncode": proc.returncode}
        stdin_errors = getattr(proc, "_pilotage_stdin_errors", None)
        if stdin_errors:
            err = str(stdin_errors[0])
            result["stdin_error"] = err
            result["output"] = rendered + f"\n[stdin write failed: {err}]"
        return result

    def _kill_process(self, proc: subprocess.Popen) -> None:
        """Kill the whole process group, then make sure it is really gone."""

        def _group_alive(pgid: int) -> bool:
            try:
                os.killpg(pgid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True  # it exists; we simply may not signal it

        def _wait_for_group_exit(pgid: int, seconds: float) -> bool:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                # Reap the leader as we go: a dead but unreaped leader still
                # makes the group look alive.
                try:
                    proc.poll()
                except Exception:  # noqa: BLE001
                    pass
                if not _group_alive(pgid):
                    return True
                time.sleep(0.05)
            try:
                proc.poll()
            except Exception:  # noqa: BLE001
                pass
            return not _group_alive(pgid)

        try:
            try:
                pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                pgid = getattr(proc, "_pilotage_pgid", None)
                if pgid is None:
                    raise

            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return

            # Wait on the group, not the wrapper. Under load bash can exit
            # before its children do, and returning then leaves them behind.
            if _wait_for_group_exit(pgid, 1.0):
                return

            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                return
            _wait_for_group_exit(pgid, 2.0)
            try:
                proc.wait(timeout=0.2)
            except (subprocess.TimeoutExpired, OSError):
                pass
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    # -- working directory --------------------------------------------------

    def _ensure_cwd(self) -> None:
        """Make sure the session's directory is one we can still spawn into.

        A command that deletes the directory it is standing in would otherwise
        break every command after it, not just itself, and the failure happens
        inside the spawn where there is nothing to report.
        """
        safe_cwd = resolve_safe_cwd(self.cwd)
        if safe_cwd != self.cwd:
            logger.warning(
                "Working directory %r is gone; using %r so commands keep working.",
                self.cwd,
                safe_cwd,
            )
            self.cwd = safe_cwd

    def _update_cwd(self, result: Dict[str, Any], *, track: bool = True) -> None:
        """Read the directory back off the marker and strip the marker away.

        A command that was killed or timed out never printed one, so the
        directory is left as it was and ``cwd_observed`` stays absent — the
        caller must not credit this command with a directory it never reported.

        ``track`` is false when the caller named a directory for this one
        command. That is a scope, not a move: a one-command workdir under /srv
        must not relocate the person's terminal there. (Hermes moves the
        session in that case; we deliberately do not.)
        """
        output = result.get("output", "")
        marker = self._marker
        last = output.rfind(marker)
        if last == -1:
            return

        search_start = max(0, last - 4096)  # a path is not four kilobytes long
        first = output.rfind(marker, search_start, last)
        if first == -1 or first == last:
            return

        previous = self.cwd
        cwd_path = output[first + len(marker) : last].strip()
        if cwd_path and track:
            if os.path.isdir(cwd_path):
                self.cwd = cwd_path
                if cwd_path != previous:
                    result["cwd_observed"] = True
            else:
                # The command ended somewhere that no longer exists — it
                # deleted its own directory. Keep the previous one; the spawn
                # would only have to recover from it anyway.
                self.cwd = previous

        # Remove the marker line and the newline we injected in front of it.
        line_start = output.rfind("\n", 0, first)
        if line_start == -1:
            line_start = first
        line_end = output.find("\n", last + len(marker))
        line_end = line_end + 1 if line_end != -1 else len(output)
        result["output"] = output[:line_start] + output[line_end:]

    # -- the interface ------------------------------------------------------

    def execute(self, command: str, cwd: str = "", *,
                timeout: Optional[int] = None,
                stdin_data: Optional[str] = None,
                rewrite_background: bool = True,
                capture_limit: Optional[int] = None) -> Dict[str, Any]:
        """Run *command* and return ``{"output", "returncode"}``.

        ``capture_limit`` bounds what is kept while the output is drained, and
        must be left unset by anything whose result is parsed rather than read:
        a truncated ``cat`` is not a shorter file, it is a corrupted one.
        """
        exec_command = rewrite_compound_background(command) if rewrite_background else command
        effective_timeout = timeout or self.timeout
        # Before wrapping, not after: the `cd` is written into the script, so a
        # directory recovered later would come too late to help this command.
        self._ensure_cwd()
        effective_cwd = cwd or self.cwd

        wrapped = self._wrap_command(exec_command, effective_cwd)
        proc = self._run_bash(
            wrapped,
            login=not self._snapshot_ready,
            timeout=effective_timeout,
            stdin_data=stdin_data,
        )
        result = self._wait_for_process(
            proc, timeout=effective_timeout, capture_limit=capture_limit
        )
        self._update_cwd(result, track=not cwd)
        return result

    def close(self) -> None:
        """Remove the session's files. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            os.unlink(self._snapshot_path)
        except OSError:
            pass
        try:
            import glob

            for leftover in glob.glob(f"{self._snapshot_path}.tmp.*"):
                try:
                    os.unlink(leftover)
                except OSError:
                    pass
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "BoundedOutput",
    "DEFAULT_TIMEOUT_SECONDS",
    "Shell",
    "UNBOUNDED_CAPTURE_CHARS",
    "find_bash",
    "read_shell_token",
    "resolve_safe_cwd",
    "rewrite_compound_background",
]
