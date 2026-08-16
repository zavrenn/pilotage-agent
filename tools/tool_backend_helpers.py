"""Shared helpers for tool backend selection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from utils import is_truthy_value

logger = logging.getLogger(__name__)


_DEFAULT_BROWSER_PROVIDER = "local"


def normalize_browser_cloud_provider(value: object | None) -> str:
    """Return a normalized browser provider key."""
    provider = str(value or _DEFAULT_BROWSER_PROVIDER).strip().lower()
    return provider or _DEFAULT_BROWSER_PROVIDER


def has_direct_modal_credentials() -> bool:
    """Return True when direct Modal credentials/config are available."""
    try:
        modal_file_exists = (Path.home() / ".modal.toml").exists()
    except (PermissionError, OSError):
        modal_file_exists = False
    return bool(
        (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))
        or modal_file_exists
    )


def _scoped_credential(name: str) -> str:
    """Read a credential env var under the active profile secret scope.

    Falls back to a raw read only when ``agent.secret_scope`` cannot be
    imported, so a packaging edge never leaves the caller without a key.
    """
    try:
        from agent.secret_scope import get_secret

        return (get_secret(name, "") or "").strip()
    except Exception:  # pragma: no cover — secret_scope is in-repo
        return (os.getenv(name, "") or "").strip()


def resolve_provider_secret(
    env_var: str,
    provider_id: str,
    config_value: str = "",
    env_getter=None,
) -> str:
    """Resolve a voice-provider API key. Single owner for STT/TTS key lookup.

    Resolution order (fixes — keys added via ``pilotage auth add
    <provider>`` were invisible to the voice tools, which only consulted
    env/.env):

    1. An explicit ``config_value`` from config.yaml, when the caller has one.
    2. The environment / ``~/.pilotage/.env``. Under a multiplexed gateway turn
       this reads the active profile's secret scope (authoritative — a scope
       miss must NOT borrow another profile's ``os.environ``; see
       ``agent/secret_scope.py``). Outside multiplexing it reads
       ``pilotage_cli.config.get_env_value`` (os.environ, then ``.env``),
       matching the tools' historical behaviour exactly.
    3. The credential pool / auth store for ``provider_id`` (``pilotage auth
       add <provider_id>``). Skipped under an active multiplex turn, where
       only the profile scope is authoritative for credentials.

    Never raises — credential resolution must not hard-fail on a pool or
    config read; returns ``""`` when no key is found anywhere.

    ``env_getter`` lets callers supply their module-level ``get_env_value``
    wrapper (transcription_tools / tts_tool expose one that tests patch);
    when omitted, ``pilotage_cli.config.get_env_value`` is used directly.
    """
    value = str(config_value or "").strip()
    if value:
        return value

    # Scope-aware env read: under a multiplexed gateway turn this reads the
    # active profile's secret scope (authoritative); otherwise it reads the
    # scope overlay then os.environ (see ``agent.secret_scope.get_secret``).
    key = _scoped_credential(env_var)
    if key:
        return key

    try:
        from agent.secret_scope import is_multiplex_active

        if is_multiplex_active():
            # Under multiplexing the profile scope is authoritative: do not
            # fall through to the process-global .env or credential pool,
            # which may belong to a different profile than the current turn.
            return ""
    except Exception:  # pragma: no cover — secret_scope is in-repo
        pass

    if env_getter is not None:
        key = str(env_getter(env_var) or "").strip()
    else:
        try:
            from pilotage_cli.config import get_env_value

            key = str(get_env_value(env_var) or "").strip()
        except ImportError:  # pragma: no cover — config is in-repo
            key = ""
    if key:
        return key

    if not provider_id:
        return ""
    try:
        from agent.credential_pool import load_pool

        # `pilotage auth add <provider>` keys a registry provider by its plain
        # id, but a provider declared via config.yaml ``providers.<name>`` /
        # ``custom_providers`` is pooled under ``custom:<name>`` (see
        # agent/credential_pool.py CUSTOM_POOL_PREFIX). Check both.
        for pool_key in (provider_id, f"custom:{provider_id}"):
            pool = load_pool(pool_key)
            if pool is None or not pool.has_credentials():
                continue
            entry = pool.peek()
            if entry is None:
                continue
            key = str(
                getattr(entry, "runtime_api_key", "")
                or getattr(entry, "access_token", "")
                or ""
            ).strip()
            if key:
                return key
    except Exception as exc:
        logger.debug(
            "Could not read %s credential pool for %s: %s",
            provider_id,
            env_var,
            exc,
        )
    return ""


def resolve_openai_audio_api_key() -> str:
    """Prefer the voice-tools key, but fall back to the normal OpenAI key.

    Routed through the profile secret scope rather than reading ``os.environ``
    directly: in a multiplex gateway serving several profiles from one
    process, ``os.environ`` reflects whichever profile's ``.env`` happened to
    load at boot, not the profile the current turn belongs to. A raw read here
    lets one profile's TTS reply / voice-note transcription authenticate as —
    and get billed against — a different profile's OpenAI account. Same
    routing the WeChat send path and ``agent/vertex_adapter`` already use; see
    ``agent/secret_scope.py``.

    Outside a multiplexed turn, ``OPENAI_API_KEY`` additionally falls back to
    the credential pool (``pilotage auth add openai-api``) via
    ``resolve_provider_secret`` — same fix as the other voice
    providers. The dedicated voice-tools override remains env/scope-only.
    """
    return (
        resolve_provider_secret("VOICE_TOOLS_OPENAI_KEY", "")
        or resolve_provider_secret("OPENAI_API_KEY", "openai-api")
    )


def fal_key_is_configured() -> bool:
    """Return True when FAL_KEY is set to a non-whitespace value.

    Consults both ``os.environ`` and ``~/.pilotage/.env`` (via
    ``pilotage_cli.config.get_env_value`` when available) so tool-side
    checks and CLI setup-time checks agree.  A whitespace-only value
    is treated as unset everywhere.
    """
    value = _scoped_credential("FAL_KEY") or None
    if value is None:
        # Fall back to the .env file for CLI paths that may run before
        # dotenv is loaded into os.environ.
        try:
            from pilotage_cli.config import get_env_value

            value = get_env_value("FAL_KEY")
        except Exception:
            value = None
    return bool(value and value.strip())
