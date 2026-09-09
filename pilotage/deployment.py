"""The two-account Ubuntu deployment, installed explicitly by the operator.

Permissions on the checkout, state directory and system unit enforce this
boundary. This module only routes CLI operations to the correct account.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

MANIFEST = Path("/etc/pilotage-agent.json")
CHECKOUT = Path("/opt/pilotage-agent")
STATE = Path("/home/agent/.pilotage-agent")


@dataclass(frozen=True)
class Deployment:
    operator: str
    operator_uid: int

    def require_operator(self) -> None:
        if os.geteuid() != self.operator_uid:
            raise PermissionError(f"Run this command as the {self.operator} operator account.")


def load() -> Deployment | None:
    if sys.platform != "linux" or Path(__file__).resolve().parent.parent != CHECKOUT:
        return None
    try:
        info = MANIFEST.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise PermissionError("The deployment manifest must be a root-owned, non-writable regular file.")
    import pwd

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    operator = pwd.getpwnam(data["operator"])
    if operator.pw_uid in {0, pwd.getpwnam("agent").pw_uid}:
        raise ValueError("The operator and runtime accounts must be separate.")
    return Deployment(operator.pw_name, operator.pw_uid)


def system_command(program: str, *, privileged: bool = False) -> list[str]:
    managed = load()
    if managed is None:
        return [program, "--user"]
    managed.require_operator()
    return (["sudo", "--"] if privileged else []) + [program]


def agent_python(arguments: list[str]) -> list[str]:
    managed = load()
    if managed is None:
        raise ValueError("No protected deployment is installed.")
    managed.require_operator()
    # Never run the runtime or read mutable agent state with operator privileges.
    # sudo's normal authentication applies; the agent receives no sudo grant.
    return [
        "sudo", "-H", "-u", "agent", "--", "/usr/bin/env",
        "PILOTAGE_HOME=" + str(STATE),
        "PATH=/opt/pilotage-agent/.venv/bin:/usr/local/bin:/usr/bin:/bin",
        str(CHECKOUT / ".venv/bin/python"), "-I", "-B",
        *arguments,
    ]


def as_agent(arguments: list[str]) -> int:
    return subprocess.call(agent_python(["-m", "pilotage.main", *arguments]))


def validate_policy_paths(state: Path, values: Mapping[str, str]) -> None:
    """Protected installs use exactly the files whose ownership we provision.

    Also called by the installer before the deployment manifest exists. This
    checks values without putting profile-controlled variables into its process.
    """
    for key, filename in (("PILOTAGE_CONFIG", "config.yaml"), ("PILOTAGE_ENV_FILE", ".env")):
        override = values.get(key, "").strip()
        expected = state / filename
        if override and Path(override).expanduser() != expected:
            raise ValueError(f"Protected deployment requires {key} to be unset or exactly {expected}.")


def make_runtime_readable(root: Path) -> None:
    """Keep installed code usable by the agent without exposing Git metadata.

    Called as the checkout owner after every dependency refresh. Do not follow
    symlinks into system Python or the operator's cache.
    """
    paths = [root]
    def unreadable(error: OSError) -> None:
        raise error

    for directory, folders, files in os.walk(root, followlinks=False, onerror=unreadable):
        if Path(directory) == root:
            folders[:] = [name for name in folders if name != ".git"]
            files = [name for name in files if name != ".git"]
        paths.extend(Path(directory) / name for name in [*folders, *files])
    checked = []
    for path in paths:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            continue
        if info.st_uid != os.geteuid() or not (
            stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
        ):
            raise PermissionError(f"Runtime entry must belong to the operator: {path}")
        checked.append((path, info.st_mode))
    for path, mode in checked:
        access = 0o555 if stat.S_ISDIR(mode) or mode & 0o111 else 0o444
        path.chmod((stat.S_IMODE(mode) & ~0o022) | access)


def protected_lock(path: Path) -> bool:
    if load() is None or path.name != ".runtime.lock":
        return False
    return path.parent == STATE or (
        path.parent.parent == STATE / "profiles"
        and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", path.parent.name) is not None
    )
