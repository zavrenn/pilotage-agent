"""Environment policy for model-controlled child processes.

Hermes removes the credentials it manages before terminal and code-execution
children are spawned. Pilotage keeps the same mechanism, narrowed to the
credentials and private routing values this runtime actually owns.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Dict, Optional


# Values owned by Pilotage's main process must not become input to a shell
# command written by the model. General operator credentials (for example AWS
# credentials used by aws/terraform) are deliberately not guessed at here;
# Hermes uses the same targeted, product-owned blocklist policy.
PROTECTED_ENV_VARS = frozenset(
    {
        # Codex / OpenAI credentials and private endpoints.
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "PILOTAGE_CODEX_BASE_URL",
        "VOICE_TOOLS_OPENAI_KEY",
        # Pilotage web extraction.
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        # WhatsApp identities and bridge authentication.
        "PILOTAGE_ALLOWED_SENDERS",
        # Retired group-routing identities are still scrubbed from an upgraded
        # deployment's environment even though the runtime no longer reads them.
        "PILOTAGE_ALLOWED_GROUPS",
        "PILOTAGE_BRIDGE_TOKEN",
        # Telegram authentication, identities, and webhook routing.
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_WEBHOOK_HOST",
        "TELEGRAM_WEBHOOK_PORT",
        "TELEGRAM_WEBHOOK_SECRET",
        "TELEGRAM_WEBHOOK_URL",
        # GitHub credentials are high-value host credentials in Hermes too.
        "GH_TOKEN",
        "GITHUB_TOKEN",
    }
)


# A service-launched Pilotage process normally runs inside its own venv. These
# markers must not make uv/poetry/conda treat that runtime as the active
# environment when the terminal operates on another project. The executable
# remains reachable through PATH, matching Hermes's current behavior.
RUNTIME_ENV_MARKERS = frozenset({"CONDA_PREFIX", "PYTHONHOME", "VIRTUAL_ENV"})


def build_subprocess_env(
    base: Optional[Mapping[str, str]] = None,
    *,
    extra: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Build a child environment with Pilotage-managed secrets removed.

    Explicit extra values use the same filter, so a caller cannot accidentally
    reinsert a protected credential after the base is scrubbed.
    """

    env = dict(os.environ if base is None else base)
    for name in PROTECTED_ENV_VARS | RUNTIME_ENV_MARKERS:
        env.pop(name, None)

    for name, value in (extra or {}).items():
        if name in PROTECTED_ENV_VARS or name in RUNTIME_ENV_MARKERS:
            continue
        env[name] = value

    # Keep child Python output deterministic on Windows and harmless elsewhere.
    env.setdefault("PYTHONUTF8", "1")
    return env


__all__ = ["PROTECTED_ENV_VARS", "RUNTIME_ENV_MARKERS", "build_subprocess_env"]
