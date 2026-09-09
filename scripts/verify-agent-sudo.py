"""Verify the Ubuntu sudo policy explicitly denies all access to agent.

A successful root-owned ``sudo -l -U agent`` query can return zero even when
its output says the account is not allowed to use sudo. Require that explicit
result; query failures and unfamiliar policy output must stop installation.
This standalone check runs before the runtime environment is installed.
"""

import os
import re
import subprocess


def verify_agent_sudo():
    try:
        result = subprocess.run(
            ["sudo", "-n", "-l", "-U", "agent"],
            env={**os.environ, "LC_ALL": "C"},
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("could not query agent sudo permissions") from exc
    if result.returncode != 0 or result.stderr.strip():
        raise RuntimeError("sudo policy query failed; inspect LC_ALL=C sudo -n -l -U agent")
    if not re.fullmatch(
        r"User agent is not allowed to run sudo on [^\s]+\.",
        result.stdout.strip(),
    ):
        raise RuntimeError(
            "agent has sudo rules or an unrecognized policy; "
            "inspect LC_ALL=C sudo -n -l -U agent"
        )


if __name__ == "__main__":
    try:
        verify_agent_sudo()
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}")
