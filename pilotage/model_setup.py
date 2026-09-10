"""Interactive operator selection of the agent's Astra effort."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import deployment
from .codex import auth, models
from .settings import ConfigError, Settings, set_agent_model


def _available_efforts(state: Path) -> tuple[str, ...]:
    if deployment.load() is None:
        return models.available_efforts(state / "codex-auth.json")
    # Only the runtime account reads/refreshes its credentials. The operator
    # receives a small list of supported levels, never tokens or mutable code.
    result = subprocess.run(
        deployment.agent_python(["-m", "pilotage.codex.models"]),
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode:
        detail = result.stderr.strip()[:500]
        raise models.ModelError(detail or "Could not confirm Astra access. Try again.")
    try:
        efforts = json.loads(result.stdout)
    except ValueError as exc:
        raise models.ModelError("Invalid ChatGPT model availability result.") from exc
    if not isinstance(efforts, list) or not efforts or any(e not in models.EFFORTS for e in efforts):
        raise models.ModelError("Invalid ChatGPT model availability result.")
    return tuple(e for e in models.EFFORTS if e in efforts)


def run_model_setup(state: Path, path: Path) -> int:
    """Save defaults after an explicit interactive choice; no service changes."""
    try:
        if not sys.stdin.isatty():
            raise models.ModelError("Run pilotage model in an interactive terminal.")
        settings = Settings.load(path)
        for channel in ("whatsapp", "telegram"):
            view = settings.for_channel(channel)
            model = view.text("agent.model", models.MODEL)
            # A common old model can be corrected by this command. A channel
            # override would silently shadow the saved choice, so report it.
            override = settings.get(f"channels.{channel}.agent.model")
            if override is not None and model != models.MODEL:
                raise ConfigError(f"Remove channels.{channel}.agent.model before selecting Astra.")
            effort_override = settings.get(f"channels.{channel}.agent.reasoning_effort")
            if effort_override is not None and effort_override not in models.EFFORTS:
                raise ConfigError(f"Invalid channels.{channel}.agent.reasoning_effort; choose one of {', '.join(models.EFFORTS)}.")
        efforts = _available_efforts(state)
        current = settings.text("agent.reasoning_effort", models.DEFAULT_EFFORT)
        print(f"Model: {models.MODEL_LABEL}")
        print(f"Current default effort: {current}")
        for number, effort in enumerate(efforts, 1):
            print(f"  {number}. {effort}" + (" (current)" if effort == current else ""))
        print("  0. Cancel")
        while True:
            choice = input("Choose the default effort: ").strip()
            if choice == "0":
                print("No changes saved.")
                return 0
            if choice.isascii() and choice.isdigit() and 1 <= int(choice) <= len(efforts):
                break
            print(f"Choose a number from 1 to {len(efforts)}, or 0 to cancel.")
        effort = efforts[int(choice) - 1]
        set_agent_model(path, models.MODEL, effort)
        print(f"Saved: {models.MODEL_LABEL}, effort {effort}.")
        for channel in ("whatsapp", "telegram"):
            override = settings.get(f"channels.{channel}.agent.reasoning_effort")
            if override is not None:
                print(f"{channel} keeps its configured effort override: {override}.")
        print("Run pilotage restart to apply the defaults. Existing session efforts are preserved.")
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nNo changes saved.")
        return 0
    except (OSError, ValueError, auth.AuthError, subprocess.SubprocessError) as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        return 1
