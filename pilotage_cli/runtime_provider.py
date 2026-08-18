"""Shared runtime provider resolution for CLI, gateway, cron, and helpers."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from pilotage_cli import auth as auth_mod
from agent.credential_pool import (
    CredentialPool,
    PooledCredential,
    credential_pool_matches_provider,
    get_custom_provider_pool_key,
    load_pool,
)
from agent.secret_scope import get_secret as _get_secret
from pilotage_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    PROVIDER_REGISTRY,
    format_auth_error,
    resolve_provider,
    resolve_codex_runtime_credentials,
    resolve_api_key_provider_credentials,
    has_usable_secret,
)
from pilotage_cli.config import (
    get_compatible_custom_providers,
    load_config,
    normalize_extra_headers,
)
from pilotage_cli.providers import custom_provider_aliases, custom_provider_slug
from pilotage_cli.providers import is_official_openai_host
from utils import base_url_host_matches, base_url_hostname, env_int


def _getenv(name: str, default: str = "") -> str:
    """Profile-scoped replacement for ``os.getenv`` on credential/provider reads.

    Routes through the secret scope (Workstream A): identical to ``os.getenv``
    when multiplexing is off, scope-aware (and fail-closed on an unscoped read)
    when on. Genuinely-global vars are handled inside ``get_secret`` and still
    read ``os.environ``. Keeps the ``(name, default) -> str`` contract every
    call site here already relies on.
    """
    val = _get_secret(name, default)
    return val if val is not None else default


def _normalize_custom_provider_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _loopback_hostname(host: str) -> bool:
    h = (host or "").lower().rstrip(".")
    return h in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _config_base_url_trustworthy_for_bare_custom(cfg_base_url: str, cfg_provider: str) -> bool:
    """Decide whether ``model.base_url`` may back bare ``custom`` runtime resolution.

    GitHub: the model picker can select Custom while ``model.provider`` still reflects a
    previous provider. Reject non-loopback URLs unless the YAML provider is already ``custom``
    (or an alias that resolves to ``custom`` at runtime), so a stale cloud
    base_url cannot hijack local ``custom`` sessions.
    """
    cfg_provider_norm = (cfg_provider or "").strip().lower()
    bu = (cfg_base_url or "").strip()
    if not bu:
        return False
    if cfg_provider_norm == "custom":
        return True
    # GitHub: provider aliases that resolve to "custom" at runtime should be
    # trusted the same way "custom" is, otherwise a legit LAN/WireGuard local
    # endpoint silently falls through to the unconfigured error.
    try:
        from pilotage_cli.auth import resolve_provider as _resolve_provider

        if _resolve_provider(cfg_provider_norm) == "custom":
            return True
    except Exception:
        pass
    return _loopback_hostname(base_url_hostname(bu))


def _detect_api_mode_for_url(base_url: str) -> Optional[str]:
    """Auto-detect api_mode from the resolved base URL.

    - Direct api.openai.com endpoints need the Responses API for GPT-5.x
      tool calls with reasoning (chat/completions returns 400).
    """
    # Official OpenAI host family: canonical api.openai.com plus the
    # data-residency regional hosts (us./eu.api.openai.com). Same API
    # surface, same Responses-API mandate. Shared predicate — see
    # providers.is_official_openai_host for the spoof-rejection contract.
    if is_official_openai_host(base_url):
        return "codex_responses"
    return None


def _fallback_api_mode(provider: str, base_url: str, model: str = "") -> str:
    """Resolve api_mode when no explicit/persisted mode applies.

    Precedence: URL detection (host-mandated wire shapes) first, then the
    transport the provider overlay itself declares via
    ``providers.determine_api_mode`` — which already handles host mandates,
    dual-wire providers, and the registry transport map — and only then the
    ``chat_completions`` default for genuinely unknown providers/endpoints.

    Before this helper the runtime paths consulted URL detection ONLY and
    silently landed reasoning providers on ``chat_completions`` whenever the
    hostname wasn't literally recognized. That is how ``openai-api`` pointed
    at OpenAI's data-residency hosts (``us.api.openai.com``) 400'd on every
    tool-calling turn: the provider declares ``codex_responses`` but the
    declaration was never consulted.
    """
    detected = _detect_api_mode_for_url(base_url)
    if detected:
        return detected
    from pilotage_cli.providers import determine_api_mode

    return determine_api_mode(provider, base_url, model) or "chat_completions"


def _resolve_plain_custom_api_mode(model_cfg: Dict[str, Any], base_url: str) -> str:
    """Resolve api_mode for legacy/plain ``provider: custom`` endpoints.

    Custom endpoints should stay conservative by default. Only direct OpenAI/xAI
    URLs imply Responses API automatically; named custom providers can opt in via
    their own ``api_mode`` field. This also prevents a stale persisted
    ``model.api_mode: codex_responses`` from forcing generic relays onto the
    Responses path after upgrades or /reset.
    """
    configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
    detected_mode = _detect_api_mode_for_url(base_url)

    if configured_mode == "codex_responses" and detected_mode != "codex_responses":
        logger.info(
            "Ignoring persisted custom api_mode=codex_responses for non-OpenAI endpoint %s",
            base_url or "(unknown)",
        )
        configured_mode = None

    return configured_mode or detected_mode or "chat_completions"


def _host_derived_api_key(base_url: str) -> str:
    """Look up `<VENDOR>_API_KEY` in the env, derived from the base URL host.

    Examples:
        https://api.vendor.com/v1 → VENDOR_API_KEY
        https://inference.example.net/v1 → EXAMPLE_API_KEY

    Returns the env value (stripped) or "". Never returns env vars whose names
    are already explicitly checked elsewhere — those are handled by their own
    host-gated paths (OPENAI).

    The vendor label is the *registrable* portion of the hostname: strip
    ``api.`` / ``www.`` prefixes, then take the second-to-last label
    (``api.vendor.com`` → ``vendor``). Falls back to "" for hostnames
    that don't yield a usable vendor label (IPs, loopback, single-label
    hosts).
    """
    hostname = base_url_hostname(base_url)
    if not hostname:
        return ""
    # Reject IPv4 / IPv6 / loopback — no meaningful vendor label.
    if any(ch.isdigit() for ch in hostname.split(".")[-1]):
        # Last label starts with a digit → likely IP. (TLDs are never numeric.)
        return ""
    if hostname in ("localhost",) or ":" in hostname:
        return ""
    labels = [lbl for lbl in hostname.split(".") if lbl]
    # Strip common API/CDN prefixes.
    while labels and labels[0] in ("api", "www"):
        labels.pop(0)
    if len(labels) < 2:
        return ""
    # Take the *registrable* label (second-to-last). For typical provider
    # hosts this is what users intuitively call "the vendor":
    #   vendor.com                    → labels[-2] = "vendor"   ✓
    #   api.vendor.com → vendor.com   → labels[-2] = "vendor"   ✓
    # Crucially, lookalike hosts pick the ATTACKER's label, not the spoofed
    # vendor:
    #   api.vendor.com.attacker.test  → labels[-2] = "attacker"
    # so VENDOR_API_KEY stays put and the chain falls through to
    # no-key-required. This mirrors how `base_url_host_matches` resists the
    # same lookalike attack for explicit hosts.
    vendor = labels[-2]
    # Sanitize to env var charset: A-Z, 0-9, underscore.
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in vendor).upper()
    if not sanitized or not sanitized[0].isalpha():
        return ""
    # Don't re-derive env vars already handled by explicit host-gated paths.
    if sanitized in ("OPENAI",):
        return ""
    env_name = f"{sanitized}_API_KEY"
    return (_getenv(env_name, "") or "").strip()


def _auto_detect_local_model(base_url: str) -> str:
    """Query a local server for its model name when only one model is loaded."""
    if not base_url:
        return ""
    try:
        import requests
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        resp = requests.get(url + "/models", timeout=(2, 3))
        if resp.ok:
            models = resp.json().get("data", [])
            if len(models) == 1:
                model_id = models[0].get("id", "")
                if model_id:
                    return model_id
    except Exception as exc:
        # Log instead of silently swallowing — aids debugging when
        # local model auto-detection fails unexpectedly.
        logger.debug("Auto-detect model from %s failed: %s", base_url, exc)
    return ""


def _get_model_config() -> Dict[str, Any]:
    config = load_config()
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        cfg = dict(model_cfg)
        # Accept "model" as alias for "default" (users intuitively write model.model)
        if not cfg.get("default") and cfg.get("model"):
            cfg["default"] = cfg["model"]
        # Handle model.default being a dict {provider: ..., model: ...} rather than a string
        _default = cfg.get("default")
        if isinstance(_default, dict):
            from pilotage_cli.config import split_model_config_default
            cfg_model, cfg_provider = split_model_config_default(_default)
            cfg_provider = cfg_provider or str(model_cfg.get("provider") or "")
            cfg["default"] = cfg_model
            if cfg_provider and not cfg.get("provider"):
                cfg["provider"] = cfg_provider
            _default = cfg_model
        default = (str(_default or "")).strip()
        base_url = (cfg.get("base_url") or "").strip()
        is_local = base_url_hostname(base_url) in ("localhost", "127.0.0.1")
        is_fallback = not default
        if is_local and is_fallback and base_url:
            detected = _auto_detect_local_model(base_url)
            if detected:
                cfg["default"] = detected
        return cfg
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    return {}


def _provider_supports_explicit_api_mode(provider: Optional[str], configured_provider: Optional[str] = None) -> bool:
    """Check whether a persisted api_mode should be honored for a given provider.

    Prevents stale api_mode from a previous provider leaking into a
    different one after a model/provider switch.  Only applies the
    persisted mode when the config's provider matches the runtime
    provider (or when no configured provider is recorded).
    """
    normalized_provider = (provider or "").strip().lower()
    normalized_configured = (configured_provider or "").strip().lower()
    if not normalized_configured:
        return True
    if normalized_provider == "custom":
        return normalized_configured == "custom" or normalized_configured.startswith("custom:")
    return normalized_configured == normalized_provider


_VALID_API_MODES = {
    "chat_completions",
    "codex_responses",
}


def _parse_api_mode(raw: Any) -> Optional[str]:
    """Validate an api_mode value from config. Returns None if invalid.

    Legacy/alias spellings (``openai``, ``responses``, …) are canonicalized
    via the shared alias map before validation, so configs written against
    older releases keep selecting the transport they named instead of
    silently falling through to hostname-based detection.
    """
    if isinstance(raw, str):
        from pilotage_cli.config import _canonical_api_mode

        normalized = _canonical_api_mode(raw).lower()
        if normalized in _VALID_API_MODES:
            return normalized
    return None


def _resolve_runtime_from_pool_entry(
    *,
    provider: str,
    entry: PooledCredential,
    requested_provider: str,
    model_cfg: Optional[Dict[str, Any]] = None,
    pool: Optional[CredentialPool] = None,
    target_model: Optional[str] = None,
) -> Dict[str, Any]:
    model_cfg = model_cfg or _get_model_config()
    # When the caller is resolving for a specific target model (e.g. a /model
    # mid-session switch), prefer that over the persisted model.default. This
    # prevents api_mode being computed from a stale config default that no
    # longer matches the model actually being used.
    effective_model = (target_model or model_cfg.get("default") or "")
    base_url = (getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or "").rstrip("/")
    api_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    api_mode = "chat_completions"
    if provider == "openai-codex":
        api_mode = "codex_responses"
        base_url = base_url or DEFAULT_CODEX_BASE_URL
    else:
        configured_provider = str(model_cfg.get("provider") or "").strip().lower()
        # Honour model.base_url from config.yaml when the configured provider
        # matches this provider. Only override when the pool entry has no
        # explicit base_url (i.e. it fell back to the hardcoded default).
        # Env var overrides win.
        pconfig = PROVIDER_REGISTRY.get(provider)
        pool_url_is_default = pconfig and base_url.rstrip("/") == pconfig.inference_base_url.rstrip("/")
        if configured_provider == provider and pool_url_is_default:
            cfg_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
            if cfg_base_url:
                base_url = cfg_base_url
        configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
        if configured_mode and _provider_supports_explicit_api_mode(provider, configured_provider):
            api_mode = configured_mode
        else:
            # URL detection first (official OpenAI hosts → codex_responses),
            # then the provider's own declared transport.
            api_mode = _fallback_api_mode(provider, base_url, effective_model)

    return {
        "provider": provider,
        "api_mode": api_mode,
        "base_url": base_url,
        "api_key": api_key,
        "source": getattr(entry, "source", "pool"),
        "credential_pool": pool,
        "requested_provider": requested_provider,
    }


def resolve_requested_provider(requested: Optional[str] = None) -> str:
    """Resolve provider request from explicit arg, config, then env."""
    if requested and requested.strip():
        return requested.strip().lower()

    model_cfg = _get_model_config()
    cfg_provider = model_cfg.get("provider")
    if isinstance(cfg_provider, str) and cfg_provider.strip():
        return cfg_provider.strip().lower()

    # Prefer the persisted config selection over any stale shell/.env
    # provider override so chat uses the endpoint the user last saved.
    env_provider = _getenv("PILOTAGE_INFERENCE_PROVIDER", "").strip().lower()
    if env_provider:
        return env_provider

    return "auto"


def _try_resolve_from_custom_pool(
    base_url: str,
    provider_label: str,
    api_mode_override: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Check if a credential pool exists for a custom endpoint and return a runtime dict if so."""
    pool_key = get_custom_provider_pool_key(base_url, provider_name=provider_name)
    if not pool_key:
        return None
    try:
        pool = load_pool(pool_key)
        if not pool.has_credentials():
            return None
        entry = pool.select()
        if entry is None:
            return None
        pool_api_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        if not pool_api_key:
            return None
        return {
            "provider": provider_label,
            "api_mode": api_mode_override or _detect_api_mode_for_url(base_url) or "chat_completions",
            "base_url": base_url,
            "api_key": pool_api_key,
            "source": f"pool:{pool_key}",
            "credential_pool": pool,
        }
    except Exception:
        return None


def _lift_max_output_tokens(entry: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Propagate a per-provider output cap onto the resolved runtime dict.

    Accepts ``max_output_tokens`` or ``max_tokens`` on a ``custom_providers``
    entry so a provider block can pin its own output limit. Gateway and CLI
    map this onto ``AIAgent.max_tokens`` only when the top-level
    ``model.max_tokens`` isn't set, so the documented global key still wins.
    """
    for _k in ("max_output_tokens", "max_tokens"):
        _v = entry.get(_k)
        if isinstance(_v, int) and _v > 0:
            result["max_output_tokens"] = _v
            return


def _lift_extra_headers(entry: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Copy a validated ``extra_headers`` dict from a provider entry.

    SECURITY: header values routinely carry credentials (Cloudflare Access
    service tokens, proxy auth, custom bearer schemes). Never log them.
    """
    extra_headers = normalize_extra_headers(entry.get("extra_headers"))
    if extra_headers:
        result["extra_headers"] = extra_headers


def _get_named_custom_provider(requested_provider: str) -> Optional[Dict[str, Any]]:
    requested_norm = _normalize_custom_provider_name(requested_provider or "")
    if not requested_norm:
        return None

    # Bare "custom" is normally an incomplete spec — the canonical form is
    # "custom:<name>" — and is otherwise owned by the model.base_url "bare
    # custom" trust path. BUT a user may literally name a ``providers:`` (or
    # legacy ``custom_providers:``) entry "custom" (e.g. ``providers.custom``
    # pointing at cliproxy). We used to return None here *before* scanning
    # config, so such an entry was never matched and resolution fell through to
    # the global default (Codex) — the cause of cron jobs with
    # ``provider: "custom"`` failing with ``auth_unavailable: providers=codex``.
    # Fall through to the config scan instead; if no entry is literally named
    # "custom" it still returns None at the end, preserving the trust path.

    # Raw names should only map to custom providers when they are not already
    # valid built-in providers or aliases. Explicit menu keys like
    # ``custom:local`` always target the saved custom provider. Bare "custom"
    # is exempt from the shadow check — it is not a built-in to defer to.
    if requested_norm == "auto":
        return None
    if requested_norm != "custom" and not requested_norm.startswith("custom:"):
        try:
            canonical = auth_mod.resolve_provider(requested_norm)
        except AuthError:
            pass
        else:
            # A user-declared ``custom_providers`` entry whose name matches
            # only an *alias* of a built-in provider is the
            # user's intended target — alias rewriting would otherwise hijack
            # the request.  We only defer to the built-in when the raw name is
            # the canonical provider itself so accidentally shadowing a
            # canonical provider still resolves to the built-in. See
            # tests/pilotage_cli/test_runtime_provider_resolution.py
            # ``test_named_custom_provider_does_not_shadow_builtin_provider``.
            if (canonical or "").strip().lower() == requested_norm:
                return None

    config = load_config()
    
    # First check providers: dict (new-style user-defined providers)
    providers = config.get("providers")
    if isinstance(providers, dict):
        from pilotage_cli.config import is_provider_enabled
        for ep_name, entry in providers.items():
            if not isinstance(entry, dict):
                continue
            # Skip providers the user explicitly disabled via
            # ``providers.<name>.enabled: false``. They remain in config
            # so re-enabling is a one-line edit, but the resolver pretends
            # they're not configured.
            if not is_provider_enabled(entry):
                continue
            # Resolve the API key from the env var name stored in key_env
            key_env = str(entry.get("key_env", "") or "").strip()
            resolved_api_key = _getenv(key_env, "").strip() if key_env else ""
            # Fall back to inline api_key when key_env is absent or unresolvable
            if not resolved_api_key:
                resolved_api_key = str(entry.get("api_key", "") or "").strip()

            display_name = entry.get("name", "")
            if requested_norm in custom_provider_aliases(
                str(display_name or ep_name),
                str(ep_name),
            ):
                # Found match by provider key
                base_url = entry.get("api") or entry.get("url") or entry.get("base_url") or ""
                if base_url:
                    result = {
                        "name": entry.get("name", ep_name),
                        "base_url": base_url.strip(),
                        "api_key": resolved_api_key,
                        "model": entry.get("default_model", ""),
                    }
                    extra_body = entry.get("extra_body")
                    if isinstance(extra_body, dict):
                        result["extra_body"] = dict(extra_body)
                    _lift_extra_headers(entry, result)
                    # The v11→v12 migration writes the API mode under the new
                    # ``transport`` field, but hand-edited configs may still
                    # use the legacy ``api_mode`` spelling.  Accept both —
                    # the runtime normaliser ``_normalize_custom_provider_entry``
                    # already does, so without this lift every migrated config
                    # silently downgrades codex_responses providers to
                    # chat_completions in the resolved runtime.
                    api_mode = _parse_api_mode(entry.get("api_mode") or entry.get("transport"))
                    if api_mode:
                        result["api_mode"] = api_mode
                    _lift_max_output_tokens(entry, result)
                    return result

    # Fall back to custom_providers: list (legacy format)
    custom_providers = config.get("custom_providers")
    if isinstance(custom_providers, dict):
        logger.warning(
            "custom_providers in config.yaml is a dict, not a list. "
            "Each entry must be prefixed with '-' in YAML."
        )
        return None

    custom_providers = get_compatible_custom_providers(config)
    if not custom_providers:
        return None

    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        base_url = entry.get("base_url")
        if not isinstance(name, str) or not isinstance(base_url, str):
            continue
        provider_key = str(entry.get("provider_key", "") or "").strip()
        if requested_norm not in custom_provider_aliases(name, provider_key):
            continue
        result = {
            "name": name.strip(),
            "base_url": base_url.strip(),
            "api_key": str(entry.get("api_key", "") or "").strip(),
        }
        key_env = str(entry.get("key_env", "") or "").strip()
        if key_env:
            result["key_env"] = key_env
        if provider_key:
            result["provider_key"] = provider_key
        extra_body = entry.get("extra_body")
        if isinstance(extra_body, dict):
            result["extra_body"] = dict(extra_body)
        _lift_extra_headers(entry, result)
        api_mode = _parse_api_mode(entry.get("api_mode"))
        if api_mode:
            result["api_mode"] = api_mode
        model_name = str(entry.get("model", "") or "").strip()
        if model_name:
            result["model"] = model_name
        _lift_max_output_tokens(entry, result)
        return result

    return None


def has_named_custom_provider(requested_provider: str) -> bool:
    """Return True when config defines a custom provider matching the request.

    Thin public wrapper around :func:`_get_named_custom_provider` so other
    modules (e.g. the cronjob tool) can decide whether a provider name will
    actually resolve to a configured ``providers:`` / ``custom_providers:``
    entry — without reaching into a private helper or duplicating the scan.
    """
    try:
        return _get_named_custom_provider(requested_provider) is not None
    except Exception:
        return False


def find_custom_provider_identity(base_url: str) -> Optional[str]:
    """Map an endpoint URL back to its canonical ``custom:<name>`` menu key.

    Returns the ``custom:<normalized-name>`` slug of the first ``providers:``
    / ``custom_providers:`` entry whose base_url matches, or ``None`` when no
    entry owns the URL.

    Session persistence stores the agent's *resolved* provider, and for every
    named custom endpoint that is the literal string ``"custom"`` — the entry
    name is lost, and the api_key is deliberately never persisted. The
    endpoint URL is the one durable fact that survives the round-trip, so
    this reverse lookup lets persist/rebuild paths recover the entry identity
    (and with it key_env/api_key/api_mode resolution via
    :func:`_get_named_custom_provider`) instead of failing with
    ``auth_unavailable`` or silently rebuilding with placeholder credentials.
    """
    target = _normalize_base_url_for_match(base_url)
    if not target:
        return None
    try:
        config = load_config()
    except Exception:
        return None

    providers = config.get("providers")
    if isinstance(providers, dict):
        for ep_name, entry in providers.items():
            if not isinstance(entry, dict):
                continue
            entry_url = (
                entry.get("api") or entry.get("url") or entry.get("base_url") or ""
            )
            if _normalize_base_url_for_match(entry_url) == target:
                return custom_provider_slug(str(ep_name), str(ep_name))

    try:
        custom_providers = get_compatible_custom_providers(config)
    except Exception:
        custom_providers = None
    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        if _normalize_base_url_for_match(entry.get("base_url")) == target:
            return custom_provider_slug(
                name,
                str(entry.get("provider_key", "") or ""),
            )

    return None


def find_custom_provider_identity_by_model(model: str) -> Optional[str]:
    """Map a model id back to the ``custom:<name>`` entry that serves it.

    Returns the ``custom:<normalized-name>`` slug of the first ``providers:``
    / ``custom_providers:`` entry whose ``model`` / ``default_model`` matches,
    or whose ``models`` catalog (dict or list shape) contains the id.
    ``None`` when no entry serves the model.

    Companion to :func:`find_custom_provider_identity` (URL reverse-lookup)
    for the persistence paths where no base_url survived the round-trip: the
    session row always stores the model name, and a custom endpoint's model
    ids (e.g. an in-house SFT checkpoint) virtually never collide with
    catalog models on built-in providers, so the model is the last durable
    fact that can recover the entry identity.
    """
    target = str(model or "").strip().lower()
    if not target:
        return None
    try:
        config = load_config()
    except Exception:
        return None

    def _entry_serves_model(entry: Dict[str, Any]) -> bool:
        for key in ("model", "default_model"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip().lower() == target:
                return True
        models = entry.get("models")
        if isinstance(models, dict):
            return any(
                str(mid).strip().lower() == target for mid in models.keys()
            )
        if isinstance(models, list):
            for item in models:
                if isinstance(item, str) and item.strip().lower() == target:
                    return True
                if isinstance(item, dict):
                    mid = item.get("id") or item.get("name")
                    if isinstance(mid, str) and mid.strip().lower() == target:
                        return True
        return False

    providers = config.get("providers")
    if isinstance(providers, dict):
        for ep_name, entry in providers.items():
            if not isinstance(entry, dict):
                continue
            if _entry_serves_model(entry):
                return custom_provider_slug(str(ep_name), str(ep_name))

    try:
        custom_providers = get_compatible_custom_providers(config)
    except Exception:
        custom_providers = None
    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        if _entry_serves_model(entry):
            return custom_provider_slug(
                name,
                str(entry.get("provider_key", "") or ""),
            )

    return None


def canonical_custom_identity(
    *,
    base_url: Optional[str] = None,
    config_provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Recover a routable ``custom:<name>`` identity for a bare custom provider.

    The bare string ``"custom"`` is the *resolved billing class* shared by
    every named ``providers:`` / ``custom_providers:`` entry — it is NOT a
    routable provider identity (``resolve_runtime_provider("custom")`` with
    no recoverable endpoint surfaces to the user as "No LLM provider
    configured").

    Any code path that persists or restores a session's provider override
    must run the resolved provider through this helper so a bare ``"custom"``
    is upgraded back to its durable ``custom:<name>`` menu key. Three
    recovery sources, in priority order:

    1. ``base_url`` — reverse-lookup the entry that owns the endpoint URL
       (the one fact that always survives the persistence round-trip when a
       URL was recorded).
    2. ``model`` — reverse-lookup the entry that serves the session's model
       (``model``/``default_model``/``models`` catalog). The session row
       always stores the model name, so when no base_url survived (the
       recurring Desktop/TUI regression vector) the model is the last
       session-scoped fact that can recover the entry — and unlike the
       config fallback below it stays correct after the user points their
       global default at a different provider.
    3. ``config_provider`` — the active ``config.model.provider`` (or its
       ``provider``/``PILOTAGE_INFERENCE_PROVIDER`` equivalent). When neither
       a base_url nor a model recovered the entry, the configured provider
       is the only durable identity left, so fall back to it when it names
       a real entry.

    Returns ``custom:<name>`` when a routable identity is recovered, else
    ``None`` (caller keeps whatever it had — bare ``"custom"`` only as a last
    resort, e.g. a genuine ad-hoc endpoint with no config entry).
    """
    # 1. Reverse-lookup by endpoint URL.
    if base_url:
        identity = find_custom_provider_identity(base_url)
        if identity:
            return identity

    # 2. Reverse-lookup by the session's model name.
    if model:
        identity = find_custom_provider_identity_by_model(model)
        if identity:
            return identity

    # 3. Fall back to the configured provider when it names a real entry.
    candidate = str(config_provider or "").strip()
    if not candidate:
        try:
            candidate = str(_get_model_config().get("provider") or "").strip()
        except Exception:
            candidate = ""
    if not candidate:
        candidate = os.environ.get("PILOTAGE_INFERENCE_PROVIDER", "").strip()

    candidate_norm = _normalize_custom_provider_name(candidate)
    # A bare/non-routable candidate cannot heal a bare custom override.
    if not candidate_norm or candidate_norm in {"custom", "auto"}:
        return None
    # Only return it when it actually resolves to a configured custom entry,
    # so we never invent a `custom:<x>` that resolution can't honor.
    try:
        entry = _get_named_custom_provider(candidate)
        if entry is not None:
            # ``candidate`` matched, but it may be the entry's DISPLAY NAME —
            # ``_get_named_custom_provider`` accepts either spelling. For a
            # keyed ``providers:`` entry the display name is not the durable
            # identity, so re-resolve through the endpoint the matched entry
            # owns and return the same config-key slug every other path
            # returns (7b5a18817). Without this, a display name that differs
            # from its key heals to ``custom:<display-name>`` and stops
            # matching the persisted identity.
            identity = find_custom_provider_identity(str(entry.get("base_url") or ""))
            if identity:
                return identity
            if candidate_norm.startswith("custom:"):
                return candidate_norm
            return f"custom:{candidate_norm}"
    except Exception:
        pass
    return None


def _normalize_base_url_for_match(value) -> str:
    return str(value or "").strip().rstrip("/").lower()


def _custom_provider_request_overrides(custom_provider: Dict[str, Any]) -> Dict[str, Any]:
    extra_body = custom_provider.get("extra_body")
    if not isinstance(extra_body, dict) or not extra_body:
        return {}
    return {"extra_body": dict(extra_body)}


def _resolve_named_custom_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    # Bare `provider="custom"` with an explicit base_url (e.g. propagated
    # from a `model_aliases:` direct-alias resolution) — build a runtime
    # directly so the alias's base_url actually takes effect.
    #
    # GitHub: provider aliases that resolve to "custom" at runtime are
    # treated identically here, so a YAML provider alias with a
    # LAN/WireGuard `base_url` doesn't silently fall through to the
    # unconfigured error.
    requested_norm = (requested_provider or "").strip().lower()
    if requested_norm and requested_norm != "custom":
        try:
            from pilotage_cli.auth import resolve_provider as _resolve_provider

            if _resolve_provider(requested_norm) == "custom":
                requested_norm = "custom"
        except Exception:
            pass
    if requested_norm == "custom" and explicit_base_url:
        base_url = explicit_base_url.strip().rstrip("/")
        # Check credential pool first — mirrors the named-custom-provider path
        # so bare `provider: custom` with a configured custom_providers entry
        # also gets its api_key from the pool instead of env var fallbacks.
        pool_result = _try_resolve_from_custom_pool(base_url, "custom", None)
        if pool_result:
            pool_result["source"] = "direct-alias"
            return pool_result
        _da_is_openai_url   = base_url_host_matches(base_url, "openai.com")
        api_key_candidates = [
            (explicit_api_key or "").strip(),
            # Gate env key fallbacks on authoritative hosts
            (_getenv("OPENAI_API_KEY", "").strip()     if _da_is_openai_url else ""),
            # Bonus: derive `<VENDOR>_API_KEY` from the host so users
            # who set a vendor env key get the intuitive match without
            # configuring `custom_providers` first.
            _host_derived_api_key(base_url),
        ]
        api_key = next(
            (c for c in api_key_candidates if has_usable_secret(c)),
            "",
        ) or "no-key-required"
        return {
            "provider": "custom",
            "api_mode": _detect_api_mode_for_url(base_url) or "chat_completions",
            "base_url": base_url,
            "api_key": api_key,
            "source": "direct-alias",
            "requested_provider": requested_provider,
        }

    custom_provider = _get_named_custom_provider(requested_provider)
    if not custom_provider:
        return None

    base_url = (
        (explicit_base_url or "").strip()
        or custom_provider.get("base_url", "")
    ).rstrip("/")
    if not base_url:
        return None

    # Check if a credential pool exists for this custom endpoint
    pool_result = _try_resolve_from_custom_pool(base_url, "custom", custom_provider.get("api_mode"), provider_name=custom_provider.get("name"))
    if pool_result:
        # Propagate the model name even when using pooled credentials —
        # the pool doesn't know about the custom_providers model field.
        model_name = custom_provider.get("model")
        if model_name:
            pool_result["model"] = model_name
        if isinstance(custom_provider.get("max_output_tokens"), int):
            pool_result["max_output_tokens"] = custom_provider["max_output_tokens"]
        request_overrides = _custom_provider_request_overrides(custom_provider)
        if request_overrides:
            pool_result["request_overrides"] = {
                **dict(pool_result.get("request_overrides") or {}),
                **request_overrides,
            }
        # Propagate extra_headers so custom-provider auth headers (e.g.
        # Cloudflare Access service tokens) still apply with pooled
        # credentials. NEVER log the values.
        if custom_provider.get("extra_headers"):
            pool_result["extra_headers"] = dict(custom_provider["extra_headers"])
        return pool_result

    _cp_is_openai_url   = base_url_host_matches(base_url, "openai.com")
    api_key_candidates = [
        (explicit_api_key or "").strip(),
        str(custom_provider.get("api_key", "") or "").strip(),
        _getenv(str(custom_provider.get("key_env", "") or "").strip(), "").strip(),
        # Gate provider env keys on their authoritative hosts — sending
        # OPENAI_API_KEY to a local-llm endpoint leaks credentials.
        (_getenv("OPENAI_API_KEY", "").strip()     if _cp_is_openai_url  else ""),
        # Bonus: derive `<VENDOR>_API_KEY` from the host as a final
        # fallback when key_env wasn't set explicitly.
        _host_derived_api_key(base_url),
    ]
    api_key = next((candidate for candidate in api_key_candidates if has_usable_secret(candidate)), "")

    result = {
        "provider": "custom",
        "api_mode": custom_provider.get("api_mode")
        or _detect_api_mode_for_url(base_url)
        or "chat_completions",
        "base_url": base_url,
        "api_key": api_key or "no-key-required",
        "source": f"custom_provider:{custom_provider.get('name', requested_provider)}",
    }
    # Propagate the model name so callers can override self.model when the
    # provider name differs from the actual model string the API expects.
    if custom_provider.get("model"):
        result["model"] = custom_provider["model"]
    if isinstance(custom_provider.get("max_output_tokens"), int):
        result["max_output_tokens"] = custom_provider["max_output_tokens"]
    # Per-provider extra HTTP headers (proxies, gateways, custom auth).
    # Values may carry credentials — NEVER log them.
    if custom_provider.get("extra_headers"):
        result["extra_headers"] = dict(custom_provider["extra_headers"])
    request_overrides = _custom_provider_request_overrides(custom_provider)
    if request_overrides:
        result["request_overrides"] = request_overrides
    return result


def _resolve_custom_endpoint_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    model_cfg = _get_model_config()
    cfg_base_url = model_cfg.get("base_url") if isinstance(model_cfg.get("base_url"), str) else ""
    cfg_provider = model_cfg.get("provider") if isinstance(model_cfg.get("provider"), str) else ""
    cfg_api_key = ""
    for k in ("api_key", "api"):
        v = model_cfg.get(k)
        if isinstance(v, str) and v.strip():
            cfg_api_key = v.strip()
            break
    requested_norm = (requested_provider or "").strip().lower()
    cfg_provider = cfg_provider.strip().lower()
    # GitHub: provider aliases that resolve to "custom" follow the same
    # base_url trust + routing rules as a bare `provider: custom`.
    # Normalising here keeps every check below — `requested_norm == "custom"`,
    # the trust check, the pool gate up the stack — alias-aware without
    # duplicating the alias map.
    if requested_norm and requested_norm != "custom":
        try:
            from pilotage_cli.auth import resolve_provider as _resolve_provider

            if _resolve_provider(requested_norm) == "custom":
                requested_norm = "custom"
        except Exception:
            pass

    env_custom_base_url = _getenv("CUSTOM_BASE_URL", "").strip()

    # Use config base_url when available and the provider context matches.
    # OPENAI_BASE_URL env var is no longer consulted — config.yaml is
    # the single source of truth for endpoint URLs.
    use_config_base_url = False
    if cfg_base_url.strip() and not explicit_base_url:
        if requested_norm == "auto":
            if not cfg_provider or cfg_provider == "auto":
                use_config_base_url = True
        elif requested_norm == "custom" and _config_base_url_trustworthy_for_bare_custom(
            cfg_base_url, cfg_provider
        ):
            use_config_base_url = True

    base_url = (
        (explicit_base_url or "").strip()
        or env_custom_base_url
        or (cfg_base_url.strip() if use_config_base_url else "")
    ).rstrip("/")
    if not base_url:
        raise AuthError(
            "No LLM provider or custom endpoint configured. Run 'pilotage model' "
            "to choose a provider and model, or set model.base_url / "
            "CUSTOM_BASE_URL in ~/.pilotage/.env.",
            code="no_provider_configured",
        )

    # Custom endpoint: use api_key from config when using config base_url.
    # Gate the provider key on its own host — sending OPENAI_API_KEY to an
    # unrelated custom endpoint leaks credentials and causes 401s.
    # Host-gated matching only, never substring.
    _is_openai_url    = base_url_host_matches(base_url, "openai.com")
    api_key_candidates = [
        explicit_api_key,
        (cfg_api_key if use_config_base_url else ""),
        (_getenv("OPENAI_API_KEY")     if _is_openai_url else ""),
        # Bonus: derive `<VENDOR>_API_KEY` from the host so users
        # who set a vendor env key get the intuitive match. Helper
        # returns "" for IPs/loopback and for env vars already handled by
        # the explicit host-gated paths above.
        _host_derived_api_key(base_url),
    ]
    api_key = next(
        (str(candidate or "").strip() for candidate in api_key_candidates if has_usable_secret(candidate)),
        "",
    )

    source = "explicit" if (explicit_api_key or explicit_base_url) else "env/config"

    # For custom endpoints, check if a credential pool exists
    # Pass requested_provider so pool lookup prefers name match over base_url,
    # fixing credential mix-ups when multiple custom providers share a base_url.
    pool_result = _try_resolve_from_custom_pool(
        base_url, "custom", _parse_api_mode(model_cfg.get("api_mode")),
        provider_name=requested_provider if requested_norm != "custom" else None,
    )
    if pool_result:
        return pool_result

    # Provide a placeholder API key for local servers that don't require
    # authentication — the OpenAI SDK requires a non-empty api_key string.
    if not api_key:
        api_key = "no-key-required"

    return {
        "provider": "custom",
        "api_mode": _resolve_plain_custom_api_mode(model_cfg, base_url),
        "base_url": base_url,
        "api_key": api_key,
        "source": source,
    }


def _resolve_explicit_runtime(
    *,
    provider: str,
    requested_provider: str,
    model_cfg: Dict[str, Any],
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    target_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    explicit_api_key = str(explicit_api_key or "").strip()
    explicit_base_url = str(explicit_base_url or "").strip().rstrip("/")
    if not explicit_api_key and not explicit_base_url:
        return None

    if provider == "openai-codex":
        base_url = explicit_base_url or DEFAULT_CODEX_BASE_URL
        api_key = explicit_api_key
        last_refresh = None
        if not api_key:
            creds = resolve_codex_runtime_credentials()
            api_key = creds.get("api_key", "")
            last_refresh = creds.get("last_refresh")
            if not explicit_base_url:
                base_url = creds.get("base_url", "").rstrip("/") or base_url
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": base_url,
            "api_key": api_key,
            "source": "explicit",
            "last_refresh": last_refresh,
            "requested_provider": requested_provider,
        }

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.auth_type == "api_key":
        env_url = ""
        if pconfig.base_url_env_var:
            env_url = _getenv(pconfig.base_url_env_var, "").strip().rstrip("/")

        base_url = explicit_base_url
        if not base_url:
            base_url = env_url or pconfig.inference_base_url

        api_key = explicit_api_key
        if not api_key:
            creds = resolve_api_key_provider_credentials(provider)
            api_key = creds.get("api_key", "")
            if not base_url:
                base_url = creds.get("base_url", "").rstrip("/")

        configured_provider = str(model_cfg.get("provider") or "").strip().lower()
        configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
        if configured_mode and _provider_supports_explicit_api_mode(provider, configured_provider):
            api_mode = configured_mode
        else:
            # URL detection first, then the provider's declared transport
            # (fixes regional OpenAI hosts and other non-chat overlays).
            api_mode = _fallback_api_mode(
                provider, base_url, target_model or model_cfg.get("default", "")
            )

        return {
            "provider": provider,
            "api_mode": api_mode,
            "base_url": base_url.rstrip("/"),
            "api_key": api_key,
            "source": "explicit",
            "requested_provider": requested_provider,
        }

    return None


def resolve_runtime_provider(
    *,
    requested: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    target_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve runtime provider credentials for agent execution.

    target_model: Optional override for model_cfg.get("default") when
    computing provider-specific api_mode (providers where different models
    route through different API surfaces). Callers performing an explicit
    mid-session model switch should pass the new model here so api_mode is
    derived from the model they are switching TO, not the stale persisted
    default. Other callers can leave it None to preserve existing behavior
    (api_mode derived from config).
    """
    requested_provider = resolve_requested_provider(requested)

    # Honour ``providers.<name>.enabled: false`` for BOTH user-defined
    # custom providers and the built-in ones (openai-codex / openai-api).
    # The earlier ``_get_named_custom_provider`` gate only covers custom
    # blocks — built-in resolution paths (``resolve_provider`` + pool /
    # explicit / generic runtime) walk their own short-circuits and would
    # otherwise return stale config for a provider the user explicitly
    # turned off.
    #
    # Fail fast with a typed error so the fallback chain can advance to
    # the next provider instead of using a disabled one.
    from pilotage_cli.config import is_provider_enabled, load_config
    _full_cfg = load_config()
    _provs_cfg = _full_cfg.get("providers") if isinstance(_full_cfg, dict) else None
    if isinstance(_provs_cfg, dict):
        _block = _provs_cfg.get(requested_provider)
        if isinstance(_block, dict) and not is_provider_enabled(_block):
            raise ValueError(
                f"provider {requested_provider!r} is disabled in config "
                f"(providers.{requested_provider}.enabled: false)"
            )

    custom_runtime = _resolve_named_custom_runtime(
        requested_provider=requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    if custom_runtime:
        custom_runtime["requested_provider"] = requested_provider
        return custom_runtime

    # If provider is "auto" (or unset) but config.yaml has an explicit base_url
    # pointing at a custom/local endpoint, route through the OpenAI-compatible
    # custom resolver instead of letting resolve_provider() pick up an
    # OPENAI_API_KEY from the environment and send the request to a cloud API.
    if not explicit_base_url and not explicit_api_key:
        model_cfg = _get_model_config()
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = str(model_cfg.get("base_url") or "").strip()
        if cfg_base_url and cfg_provider in ("auto", ""):
            # Check that base_url isn't one of the well-known cloud API roots
            # (OpenAI). If it's something else we honour it directly. The full
            # detection logic lives in _resolve_custom_endpoint_runtime; we
            # just skip the resolve_provider() call so env-var credentials
            # don't shadow it. Match on HOST, not substring, so a look-alike
            # base_url (or one whose path merely contains "openai.com")
            # cannot evade the bypass and leak a cloud credential. Mirrors
            # the host-gating used for API-key selection in
            # _resolve_custom_endpoint_runtime.
            _known_cloud_hosts = (
                "openai.com",
            )
            if not any(
                base_url_host_matches(cfg_base_url, host)
                for host in _known_cloud_hosts
            ):
                runtime = _resolve_custom_endpoint_runtime(
                    requested_provider=requested_provider,
                    explicit_api_key=explicit_api_key,
                    explicit_base_url=explicit_base_url,
                )
                runtime["requested_provider"] = requested_provider
                return runtime

    provider = resolve_provider(
        requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    model_cfg = _get_model_config()
    explicit_runtime = _resolve_explicit_runtime(
        provider=provider,
        requested_provider=requested_provider,
        model_cfg=model_cfg,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
        target_model=target_model,
    )
    if explicit_runtime:
        return explicit_runtime

    try:
        pool = load_pool(provider)
    except Exception:
        pool = None
    if pool and pool.has_credentials():
        entry = pool.select()
        pool_api_key = ""
        if entry is not None:
            pool_api_key = (
                getattr(entry, "runtime_api_key", None)
                or getattr(entry, "access_token", "")
            )
        if (
            entry is not None
            and pool_api_key
            and credential_pool_matches_provider(
                pool,
                provider,
                base_url=(
                    getattr(entry, "runtime_base_url", None)
                    or getattr(entry, "base_url", None)
                    or ""
                ),
            )
        ):
            return _resolve_runtime_from_pool_entry(
                provider=provider,
                entry=entry,
                requested_provider=requested_provider,
                model_cfg=model_cfg,
                pool=pool,
                target_model=target_model,
            )

    if provider == "openai-codex":
        try:
            creds = resolve_codex_runtime_credentials()
            return {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": creds.get("base_url", "").rstrip("/"),
                "api_key": creds.get("api_key", ""),
                "source": creds.get("source", "pilotage-auth-store"),
                "last_refresh": creds.get("last_refresh"),
                "requested_provider": requested_provider,
            }
        except AuthError:
            if requested_provider != "auto":
                raise
            # Auto-detected Codex but credentials are stale/revoked —
            # fall through to env-var providers.
            logger.info("Auto-detected Codex provider but credentials failed; "
                        "falling through to next provider.")


    # API-key providers (registry + provider plugins)
    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.auth_type == "api_key":
        creds = resolve_api_key_provider_credentials(provider)
        # An explicitly selected API-key provider is authoritative. Returning
        # a runtime with an empty key defers failure until the first request and
        # can make a later fallback look like a silent provider switch. Fail at
        # resolution so callers surface the missing credential (or consult only
        # an explicitly configured fallback chain).
        if not has_usable_secret(creds.get("api_key")):
            env_names = ", ".join(pconfig.api_key_env_vars)
            hint = f" Set {env_names}." if env_names else ""
            raise AuthError(
                f"No usable credentials found for provider '{provider}'.{hint}",
                provider=provider,
                code="missing_api_key",
            )
        # Honour model.base_url from config.yaml when the configured provider
        # matches this provider (e.g. regional endpoint overrides).
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = ""
        if cfg_provider == provider:
            cfg_base_url = (model_cfg.get("base_url") or "").strip().rstrip("/")
        base_url = cfg_base_url or creds.get("base_url", "").rstrip("/")
        configured_provider = cfg_provider
        # Only honor persisted api_mode when it belongs to the same provider family.
        configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
        if configured_mode and _provider_supports_explicit_api_mode(provider, configured_provider):
            api_mode = configured_mode
        else:
            # URL detection first (official OpenAI hosts → codex_responses),
            # then the provider's declared transport.
            api_mode = _fallback_api_mode(
                provider, base_url, target_model or model_cfg.get("default", "")
            )
        return {
            "provider": provider,
            "api_mode": api_mode,
            "base_url": base_url,
            "api_key": creds.get("api_key", ""),
            "source": creds.get("source", "env"),
            "requested_provider": requested_provider,
        }

    runtime = _resolve_custom_endpoint_runtime(
        requested_provider=requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    runtime["requested_provider"] = requested_provider
    return runtime


def format_runtime_provider_error(error: Exception) -> str:
    if isinstance(error, AuthError):
        return format_auth_error(error)
    return str(error)
