"""Read the selected service's journal, including Python log severity."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime

from .service import unit_name
from .deployment import system_command


LEVELS = {"DEBUG": 7, "INFO": 6, "WARNING": 4, "ERROR": 3, "CRITICAL": 2}
_PYTHON_LEVEL = re.compile(r"^\d{2}:\d{2}:\d{2}\s+(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s")


def journal_since(value: str) -> str:
    """Accept the short relative notation used by the operator CLI."""
    match = re.fullmatch(r"(\d+)([smhd])", value.strip(), re.IGNORECASE)
    if match:
        unit = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[match[2].lower()]
        return f"{match[1]} {unit} ago"
    return value


def entry_priority(entry: dict) -> int | None:
    # Python logs all go to stderr, regardless of their actual severity.
    message = entry.get("MESSAGE", "")
    match = _PYTHON_LEVEL.match(message) if isinstance(message, str) else None
    if match:
        return LEVELS[match[1]]
    # Like Hermes, keep stream lines without an explicit level: they may be
    # traceback continuations or standalone startup failures. In journald,
    # both stdout and stderr use the "stdout" transport.
    if entry.get("_TRANSPORT") == "stdout":
        return None
    try:
        return int(entry["PRIORITY"])
    except (KeyError, TypeError, ValueError):
        return None


def run_logs(*, follow=False, lines=50, level=None, since=None) -> int:
    if shutil.which("journalctl") is None:
        print("journalctl is not installed; logs require Ubuntu systemd.", file=sys.stderr)
        return 1
    command = [
        *system_command("journalctl", privileged=True), "--unit", unit_name(),
        "--no-pager", "--all", "--output=json", "--lines", str(lines),
    ]
    if follow:
        command.append("--follow")
    if since:
        command.extend(["--since", journal_since(since)])
    process = None
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        )
        for raw in process.stdout:
            entry = json.loads(raw)
            priority = entry_priority(entry)
            if level and priority is not None and priority > LEVELS[level]:
                continue
            stamp = datetime.fromtimestamp(
                int(entry["__REALTIME_TIMESTAMP"]) / 1_000_000,
            ).astimezone().isoformat(timespec="microseconds")
            print(f"{stamp} {entry.get('MESSAGE', '')}", flush=True)
        code = process.wait()
        return code if code >= 0 else 1
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Could not read service logs: {exc}", file=sys.stderr)
        return 1
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()
