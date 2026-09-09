"""Small operator controls for the installed Pilotage service."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Callable, Sequence

from .deployment import system_command


# The installed unit allows up to 90 seconds for graceful shutdown.
SERVICE_TIMEOUT_SECONDS = 120
_ACTIONS = frozenset({"start", "stop", "restart", "status"})


def unit_name(profile_name: str) -> str:
    return f"pilotage-agent@{profile_name}.service"


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=SERVICE_TIMEOUT_SECONDS,
        check=False,
    )


def run_service_command(
    action: str,
    profile_name: str,
    *,
    run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> int:
    """Control or inspect exactly one installed profile service."""

    action = str(action or "").strip().lower()
    if action not in _ACTIONS:
        print(f"Unknown service action: {action}", file=sys.stderr)
        return 1
    if shutil.which("systemctl") is None:
        print("systemctl is not installed; service control requires Ubuntu systemd.", file=sys.stderr)
        return 1

    unit = unit_name(profile_name)
    if action in {"start", "stop", "restart"}:
        command = [*system_command("systemctl", privileged=True), action, unit]
    else:
        command = [
            *system_command("systemctl"),
            "show",
            unit,
            "--no-pager",
            "--property=LoadState,UnitFileState,ActiveState,SubState,MainPID",
        ]
    try:
        result = run(command)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Could not {action} {unit}: {exc}", file=sys.stderr)
        return 1
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "systemctl failed").strip()
        print(f"{unit}: {detail}", file=sys.stderr)
        return 1

    if action == "status":
        values = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        print(
            f"{unit}: {values.get('ActiveState', 'unknown')} "
            f"({values.get('SubState', 'unknown')}), "
            f"enabled={values.get('UnitFileState', 'unknown')}, "
            f"pid={values.get('MainPID', '0')}"
        )
    else:
        verb = {"start": "Started", "stop": "Stopped", "restart": "Restarted"}[action]
        print(f"{verb} {unit}")
    return 0


__all__ = ["run_service_command", "unit_name"]
