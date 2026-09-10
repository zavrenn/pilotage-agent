"""The one supported model and its ChatGPT-account availability check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

from . import auth
from .client import cloudflare_headers

MODEL = "gpt-6-astra"
MODEL_LABEL = "GPT-6 Astra"
DEFAULT_EFFORT = "high"
# Native Responses efforts; Ultra is a Codex orchestration mode, not an effort.
# https://developers.openai.com/api/docs/models/gpt-6-astra
EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ModelError(ValueError):
    """The account could not confirm the supported model's availability."""


def available_efforts(credentials_path: Path) -> tuple[str, ...]:
    """Ask this agent's own Codex catalog; never invent account access."""

    credentials = auth.resolve_credentials(credentials_path)
    for attempt in range(2):
        headers = cloudflare_headers(credentials.access_token)
        headers["Authorization"] = f"Bearer {credentials.access_token}"
        try:
            response = httpx.get(
                auth.validated_codex_base_url(credentials.base_url).rstrip("/") + "/models",
                params={"client_version": "1.0.0"},
                headers=headers,
                timeout=10,
            )
        except httpx.HTTPError as exc:
            raise ModelError("Could not reach the ChatGPT model catalog. Try again.") from exc
        if response.status_code == 401 and attempt == 0:
            credentials = auth.resolve_credentials(credentials_path, force_refresh=True)
            continue
        if response.status_code != 200:
            raise ModelError(f"ChatGPT model catalog returned HTTP {response.status_code}.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelError("ChatGPT returned an invalid model catalog.") from exc
        entries = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ModelError("ChatGPT returned an invalid model catalog.")
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("slug") != MODEL:
                continue
            if entry.get("visibility") not in (None, "", "list") or entry.get("show_in_picker") is False:
                continue
            levels = entry.get("supported_reasoning_levels")
            if not isinstance(levels, list):
                raise ModelError("ChatGPT did not confirm Astra's supported efforts.")
            advertised = set()
            for level in levels:
                effort = level.get("effort") if isinstance(level, dict) else level
                if isinstance(effort, str):
                    advertised.add(effort.strip())
            available = tuple(effort for effort in EFFORTS if effort in advertised)
            if not available:
                raise ModelError("ChatGPT did not confirm Astra's supported efforts.")
            return available
        raise ModelError("GPT-6 Astra is not available to this agent's ChatGPT account.")
    raise ModelError("Could not confirm ChatGPT model availability.")  # pragma: no cover


def main() -> int:
    """Private unprivileged catalog probe for protected operator setup."""
    from ..config import state_dir

    try:
        print(json.dumps(available_efforts(state_dir() / "codex-auth.json")))
        return 0
    except ModelError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except auth.AuthError:
        print("ChatGPT authentication needs attention. Run pilotage login.", file=sys.stderr)
        return 1
    except OSError:
        print("Could not read this agent's ChatGPT credentials.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
