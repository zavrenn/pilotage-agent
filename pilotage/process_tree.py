"""Bounded, identity-aware subprocess tree termination for Ubuntu runtimes."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_SIGTERM = getattr(signal, "SIGTERM", 15)
_SIGKILL = getattr(signal, "SIGKILL", 9)


@dataclass(frozen=True)
class ProcessIdentity:
    """A Linux PID plus its kernel start tick, which survives PID reuse."""

    pid: int
    start_time: int


def _read_linux_stat(
    pid: int,
    proc_root: Path = Path("/proc"),
) -> Optional[tuple[int, int, str]]:
    """Return ``(ppid, start_time, state)`` from one Linux proc stat row."""

    try:
        written = proc_root.joinpath(str(int(pid)), "stat").read_text(
            encoding="utf-8"
        )
        # comm is parenthesized and may itself contain spaces or parentheses;
        # fields after its final ')' have stable positions.
        tail = written[written.rfind(")") + 2 :].split()
        if len(tail) < 20:
            return None
        return int(tail[1]), int(tail[19]), tail[0]
    except (OSError, ValueError):
        return None


def snapshot_linux_descendants(
    root_pid: int,
    proc_root: Path = Path("/proc"),
) -> list[ProcessIdentity]:
    """Snapshot all descendants before their parent can be killed/reparented."""

    by_parent: dict[int, list[ProcessIdentity]] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        stat = _read_linux_stat(pid, proc_root)
        if stat is None:
            continue
        ppid, started, state = stat
        if state == "Z":
            continue
        by_parent.setdefault(ppid, []).append(ProcessIdentity(pid, started))

    found: list[ProcessIdentity] = []
    pending = [int(root_pid)]
    while pending:
        parent = pending.pop()
        children = by_parent.get(parent, [])
        found.extend(children)
        pending.extend(child.pid for child in children)
    return found


def _identity_alive(
    identity: ProcessIdentity,
    proc_root: Path = Path("/proc"),
) -> bool:
    stat = _read_linux_stat(identity.pid, proc_root)
    return bool(
        stat is not None
        and stat[1] == identity.start_time
        and stat[2] != "Z"
    )


def _signal_identity(identity: ProcessIdentity, sig: int) -> bool:
    """Signal only when PID still names the snapshotted process."""

    if not _identity_alive(identity):
        return False
    try:
        os.kill(identity.pid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _wait_for_exit(
    process: Any,
    descendants: list[ProcessIdentity],
    seconds: float,
) -> bool:
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        leader_alive = process.poll() is None
        survivors = any(_identity_alive(child) for child in descendants)
        if not leader_alive and not survivors:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def terminate_process_tree(
    process: Any,
    *,
    pgid: Optional[int] = None,
    term_grace_seconds: float = 1.0,
    kill_grace_seconds: float = 2.0,
) -> None:
    """TERM then KILL a child, including descendants that called ``setsid``."""

    if os.name == "nt":
        if process.poll() is not None:
            return
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                timeout=max(1.0, kill_grace_seconds),
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=max(1.0, kill_grace_seconds))
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=0.2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        return

    descendants = snapshot_linux_descendants(process.pid)
    if pgid is None:
        try:
            pgid = os.getpgid(process.pid)
        except (ProcessLookupError, PermissionError, OSError):
            pgid = None

    def signal_leader(sig: int) -> None:
        try:
            # Pilotage children start new sessions.  The equality guard makes
            # a degraded/non-leader path incapable of signalling our own group.
            if pgid is not None and pgid == process.pid:
                os.killpg(pgid, sig)
            elif process.poll() is None:
                os.kill(process.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    signal_leader(_SIGTERM)
    for child in descendants:
        _signal_identity(child, _SIGTERM)
    if _wait_for_exit(process, descendants, term_grace_seconds):
        return

    signal_leader(_SIGKILL)
    for child in descendants:
        _signal_identity(child, _SIGKILL)
    _wait_for_exit(process, descendants, kill_grace_seconds)
    try:
        process.wait(timeout=0.2)
    except (subprocess.TimeoutExpired, OSError):
        pass
