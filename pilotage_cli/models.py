"""
Canonical model catalogs and lightweight validation helpers.

Add, remove, or reorder entries here — both `pilotage setup` and
`pilotage` provider-selection will pick up the change automatically.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import urllib.parse
import urllib.request
import urllib.error
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any, NamedTuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeGuard

from pilotage_cli import __version__ as _PILOTAGE_VERSION
from pilotage_cli.urllib_security import open_credentialed_url

logger = logging.getLogger(__name__)

# Identify ourselves so endpoints fronted by Cloudflare's Browser Integrity
# Check (error 1010) don't reject the default ``Python-urllib/*`` signature.
_PILOTAGE_USER_AGENT = f"pilotage-cli/{_PILOTAGE_VERSION}"

def _urlopen_model_catalog_request(req: urllib.request.Request, *, timeout: float, ssl_context=None):
    """Open catalog requests without forwarding headers across origins."""
    return open_credentialed_url(req, timeout=timeout, ssl_context=ssl_context)


def _custom_provider_ssl_context(base_url: str):
    """Build an ``ssl.SSLContext`` from a custom provider's TLS settings.

    Mirrors the httpx/requests TLS resolution so the urllib ``/models``
    discovery probe honors a provider's ``ssl_ca_cert`` / ``ssl_verify``
    instead of falling back to the process-wide ``SSL_CERT_FILE`` / certifi
    bundle. Returns None when no per-provider TLS override applies, so the
    caller keeps urllib's default policy for public/unconfigured endpoints.
    """
    if not base_url:
        return None
    try:
        from pilotage_cli.config import get_custom_provider_tls_settings

        tls = get_custom_provider_tls_settings(base_url)
        if not tls:
            return None
        import ssl

        if tls.get("ssl_verify") is False:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        ca = tls.get("ssl_ca_cert")
        if isinstance(ca, str) and ca and os.path.isfile(ca):
            return ssl.create_default_context(cafile=ca)
    except Exception:
        return None  # never break discovery on a TLS-config lookup
    return None


def _codex_curated_models() -> list[str]:
    """Derive the openai-codex curated list from codex_models.py.

    Single source of truth: DEFAULT_CODEX_MODELS + forward-compat synthesis.
    This keeps the gateway /model picker in sync with the CLI `pilotage model`
    flow without maintaining a separate static list.
    """
    from pilotage_cli.codex_models import DEFAULT_CODEX_MODELS, _add_forward_compat_models
    return _add_forward_compat_models(list(DEFAULT_CODEX_MODELS))


_PROVIDER_MODELS: dict[str, list[str]] = {
    # Native OpenAI Chat Completions (api.openai.com). Used by /model counts and
    # provider_model_ids fallback when /v1/models is unavailable.
    "openai": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-api": [
        "gpt-5.6-sol",
        "gpt-5.6-sol-pro",
        "gpt-5.6-terra",
        "gpt-5.6-terra-pro",
        "gpt-5.6-luna",
        "gpt-5.6-luna-pro",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-codex": _codex_curated_models(),
}


# ---------------------------------------------------------------------------
# Canonical provider list — single source of truth for provider identity.
# Every code path that lists, displays, or iterates providers derives from
# this list:  pilotage model, /model, list_authenticated_providers.
#
# Fields:
#   slug        — internal provider ID (used in config.yaml, --provider flag)
#   label       — short display name
#   tui_desc    — longer description for the `pilotage model` interactive picker
# ---------------------------------------------------------------------------

class ProviderEntry(NamedTuple):
    slug: str
    label: str
    tui_desc: str   # detailed description for `pilotage model` TUI

CANONICAL_PROVIDERS: list[ProviderEntry] = [
    ProviderEntry("openai-codex",   "ChatGPT or Codex Subscription", "ChatGPT or Codex Subscription (Sign in with your ChatGPT account, uses Codex models)"),
    ProviderEntry("openai-api",     "OpenAI API",               "OpenAI API (api.openai.com, API key)"),
]

# Auto-extend CANONICAL_PROVIDERS with any provider registered in providers/
# that is not already in the list above.  Adding plugins/model-providers/<name>/
# is sufficient to expose a new provider in the model picker, /model, and all
# downstream consumers — no edits to this file needed.
_canonical_slugs = {p.slug for p in CANONICAL_PROVIDERS}
try:
    from providers import list_providers as _list_providers_for_canonical
    for _pp in _list_providers_for_canonical():
        if _pp.name in _canonical_slugs:
            continue
        if _pp.auth_type in {"oauth_device_code", "oauth_external", "external_process", "copilot", "vertex"}:
            continue  # non-api-key flows need bespoke picker UX; skip auto-inject
        _label = _pp.display_name or _pp.name
        _desc = _pp.description or f"{_label} (direct API)"
        CANONICAL_PROVIDERS.append(ProviderEntry(_pp.name, _label, _desc))
        _canonical_slugs.add(_pp.name)
except Exception:
    pass

# Derived dicts — used throughout the codebase
_PROVIDER_LABELS = {p.slug: p.label for p in CANONICAL_PROVIDERS}
_PROVIDER_LABELS["custom"] = "Custom endpoint"  # special case: not a named provider


# ---------------------------------------------------------------------------
# Provider groups — DISPLAY ONLY
#
# Some vendors expose several Pilotage provider slugs (one per endpoint /
# auth method: global API, China API, OAuth coding plan, ...). Listing every
# slug as a top-level row in the interactive `pilotage model` / setup wizard /
# Telegram `/model` pickers makes that list long and noisy.
#
# These groups fold related slugs under one top-level row in INTERACTIVE
# PICKERS only. They do NOT change ``CANONICAL_PROVIDERS``, slug identity,
# the ``--provider`` flag, ``/model <provider:model>``, or any typed path —
# every member slug remains individually addressable. Grouping is a pure
# display affordance; ``group_providers()`` is the single fold used by all
# three picker surfaces so they stay consistent.
#
#   group_id -> (display_label, group_description, [member_slug, ...])
#
# ``group_description`` is a short blurb shown on the collapsed top-level group
# row in the interactive pickers (alongside the label). Member-specific detail
# lives in each member's ``tui_desc`` and shows in the drill-down sub-picker.
# Member order is the order shown inside the group submenu.
# ---------------------------------------------------------------------------
PROVIDER_GROUPS: dict[str, tuple[str, str, list[str]]] = {
    "openai":   ("OpenAI",          "ChatGPT/Codex subscription or direct OpenAI API", ["openai-codex", "openai-api"]),
}

# Reverse index: member slug -> group_id. Built once at import.
_SLUG_TO_GROUP: dict[str, str] = {
    slug: gid for gid, (_label, _desc, members) in PROVIDER_GROUPS.items() for slug in members
}


def provider_group_for_slug(slug: str) -> str:
    """Return the group_id a provider slug belongs to, or "" if ungrouped."""
    return _SLUG_TO_GROUP.get(str(slug or "").strip().lower(), "")


def group_providers(slugs):
    """Fold a flat ordered slug iterable into picker rows by provider group.

    DISPLAY ONLY. Used by every interactive picker (``pilotage model``, the
    setup wizard, the Telegram ``/model`` keyboard) so grouping is identical
    across surfaces.

    Each returned row is a dict::

        {"kind": "single", "slug": <slug>}                       # ungrouped, or
                                                                  # 1-member group
        {"kind": "group", "group_id": <gid>, "label": <label>,
         "description": <desc>, "members": [<slug>, ...]}        # 2+ members

    Rules:
      * A group row appears at the position of its FIRST present member, in
        the input order. Subsequent members fold into that row (and are not
        emitted again).
      * Member order inside a group follows ``PROVIDER_GROUPS`` declaration,
        restricted to the members actually present in ``slugs``.
      * A group reduced to a single present member degrades to a ``single``
        row — no pointless one-item submenu.
      * Slugs not in any group pass through as ``single`` rows, order
        preserved.
      * Duplicate slugs in the input are ignored after first sight.
    """
    seen: set[str] = set()
    # Which present members each group has, in declaration order.
    group_members: dict[str, list[str]] = {}
    for gid, (_label, _desc, members) in PROVIDER_GROUPS.items():
        present = [m for m in members if m in set(slugs)]
        if present:
            group_members[gid] = present

    rows = []
    emitted_groups: set[str] = set()
    for slug in slugs:
        s = str(slug or "").strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        gid = _SLUG_TO_GROUP.get(s, "")
        if not gid:
            rows.append({"kind": "single", "slug": s})
            continue
        if gid in emitted_groups:
            continue  # already folded at the first member's position
        emitted_groups.add(gid)
        members = group_members.get(gid, [s])
        if len(members) <= 1:
            rows.append({"kind": "single", "slug": members[0]})
        else:
            label, desc, _ = PROVIDER_GROUPS[gid]
            rows.append(
                {"kind": "group", "group_id": gid, "label": label,
                 "description": desc, "members": list(members)}
            )
    return rows


_PROVIDER_ALIASES: dict[str, str] = {}


# In-repo fallback for the model Pilotage silently lands on when the user
# never picked one (empty ``model.default``, or provider set with no model).
PREFERRED_SILENT_DEFAULT_MODEL = "gpt-5.4"


def get_preferred_silent_default_model(provider: str = "openai-codex") -> str:
    """Return the silent-default model id — catalog label first, constant second.

    Reads the ``"default": true`` label from the cached remote catalog
    (never hits the network — safe on hot resolution paths), falling back to
    :data:`PREFERRED_SILENT_DEFAULT_MODEL` when no cached manifest exists or
    the provider block carries no label.
    """
    try:
        from pilotage_cli.model_catalog import get_default_model_from_cache
        labeled = get_default_model_from_cache(provider)
        if labeled:
            return labeled
    except Exception:
        pass
    return PREFERRED_SILENT_DEFAULT_MODEL


def pick_silent_default_model(model_ids: list[str], provider: str = "openai-codex") -> str:
    """Pick the silent default from an available-models list.

    Returns the catalog-labeled default (see
    :func:`get_preferred_silent_default_model`) when the list carries it,
    else the first entry, else "". Used by every surface that must choose a
    model on the user's behalf without an interactive picker (GUI onboarding
    recommended-default, empty-model runtime fallback).
    """
    preferred = get_preferred_silent_default_model(provider)
    if preferred in model_ids:
        return preferred
    return model_ids[0] if model_ids else ""


# Providers whose *silent* auto-default must go through the cost-safe
# catalog-labeled default (``get_preferred_silent_default_model``) instead
# of curated-list entry [0], so a missing model never escalates to the
# priciest flagship. None are currently defined.
_SILENT_DEFAULT_PROVIDERS: frozenset[str] = frozenset()


def get_default_model_for_provider(provider: str) -> str:
    """Return a cost-safe default model for a provider, or "" if unknown.

    Used as a NON-INTERACTIVE fallback when a provider is configured but no
    model was ever selected (e.g. ``pilotage auth add openai-codex`` without
    ``pilotage model``, or a profile that sets ``provider`` with no ``model``).

    For most providers this is the first entry in ``_PROVIDER_MODELS`` — the
    same model the ``pilotage model`` picker offers first. For metered aggregators
    whose curated list is ordered most-capable-first, that entry is also the
    most EXPENSIVE one, so silently defaulting to it is a billing footgun.
    Those providers (``_SILENT_DEFAULT_PROVIDERS``) resolve through the
    catalog-labeled default instead; a missing model must never auto-escalate
    to the flagship.
    """
    models = _PROVIDER_MODELS.get(provider, [])
    if provider in _SILENT_DEFAULT_PROVIDERS:
        preferred = get_preferred_silent_default_model(provider)
        # Trust the preferred default even when the provider has no static
        # catalog (OpenRouter's picker list is fetched live; its curated
        # snapshot carries the default).
        if preferred and (preferred in models or not models):
            return preferred
    return models[0] if models else ""


# ---------------------------------------------------------------------------
# Pricing helpers — fetch live pricing from OpenAI-compatible /v1/models
# ---------------------------------------------------------------------------

# Cache: maps model_id → {"prompt": str, "completion": str} per endpoint
_pricing_cache: dict[str, dict[str, dict[str, str]]] = {}

# A failed fetch caches its empty result too, so an unreachable endpoint isn't
# re-dialed on every call — but only until this deadline. Cached forever, one
# bad moment (a blip during startup, a key that hadn't been written yet) turns
# into no live model discovery for the life of the process, and the processes
# that read this most are the ones that run for weeks: the gateway, the desktop
# backend. Every caller falls back to a curated list meanwhile, so the cost of
# the stale entry is silent and invisible.
_FAILED_CATALOG_TTL_SECONDS = 120.0
_pricing_cache_retry_after: dict[str, float] = {}


def _cached_catalog(cache_key: str) -> Optional[dict[str, dict[str, Any]]]:
    """The cached catalog for *cache_key*, or None to go fetch it."""
    cached = _pricing_cache.get(cache_key)
    if cached is None:
        return None
    retry_after = _pricing_cache_retry_after.get(cache_key)
    if retry_after is not None and time.monotonic() >= retry_after:
        _pricing_cache.pop(cache_key, None)
        _pricing_cache_retry_after.pop(cache_key, None)
        return None
    return cached


def _cache_catalog(
    cache_key: str, result: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Cache a catalog result, giving an empty one an expiry."""
    _pricing_cache[cache_key] = result
    if result:
        _pricing_cache_retry_after.pop(cache_key, None)
    else:
        _pricing_cache_retry_after[cache_key] = (
            time.monotonic() + _FAILED_CATALOG_TTL_SECONDS
        )
    return result


def _format_price_per_mtok(per_token_str: str) -> str:
    """Convert a per-token price string to a human-friendly $/Mtok string.

    Always uses 2 decimal places so that prices align vertically when
    right-justified in a column (the decimal point stays in the same position).

    Sub-cent prices (e.g. deep-discount cache-hit promos) extend precision
    instead of collapsing to "$0.00": the smallest decimal place that makes
    the value non-zero is found, then one extra digit is kept and trailing
    zeros trimmed.

    Examples:
        "0.000003"        → "$3.00"      (per million tokens)
        "0.00003"         → "$30.00"
        "0.00000015"      → "$0.15"
        "0.0000001"       → "$0.10"
        "0.00018"         → "$180.00"
        "0.0000000018"    → "$0.0018"    (promo: $0.0018/Mtok)
        "0"               → "free"
    """
    try:
        val = float(per_token_str)
    except (TypeError, ValueError):
        return "?"
    if val == 0:
        return "free"
    per_m = val * 1_000_000
    text = f"{per_m:.2f}"
    if per_m < 0.01:
        # Non-zero price below one cent per Mtok — widen precision until the
        # value shows, keep one extra significant digit, trim trailing zeros.
        prec = 3
        while prec < 12 and round(per_m, prec) == 0:
            prec += 1
        text = f"{per_m:.{min(prec + 1, 12)}f}".rstrip("0").rstrip(".")
    return f"${text}"


def fetch_models_with_pricing(
    api_key: str | None = None,
    base_url: str = "https://api.openai.com",
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fetch ``/v1/models`` and return ``{model_id: {prompt, completion, ...}}``.

    Results are cached per *base_url* so repeated calls are free.
    Works with any OpenAI-compatible endpoint.
    """
    cache_key = (base_url or "").rstrip("/")
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    url = cache_key + "/v1/models"
    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": _PILOTAGE_USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return _cache_catalog(cache_key, {})

    result: dict[str, dict[str, Any]] = {}
    for item in payload.get("data", []):
        mid = item.get("id")
        pricing = item.get("pricing")
        if mid and isinstance(pricing, dict):
            entry: dict[str, Any] = {
                "prompt": str(pricing.get("prompt", "")),
                "completion": str(pricing.get("completion", "")),
            }
            if pricing.get("input_cache_read"):
                entry["input_cache_read"] = str(pricing["input_cache_read"])
            if pricing.get("input_cache_write"):
                entry["input_cache_write"] = str(pricing["input_cache_write"])
            result[mid] = entry

    return _cache_catalog(cache_key, result)


def get_pricing_for_provider(provider: str, *, force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """Return live pricing for providers that support it.

    No built-in provider publishes a per-slug pricing catalog today; the
    generic endpoint-level fetch lives in :func:`fetch_models_with_pricing`
    (used with explicit base URLs by auxiliary surfaces). Kept as a stable
    seam for the picker/pricing call sites, which all handle an empty map.
    """
    return {}


# All provider IDs and aliases that are valid for the provider:model syntax.
_KNOWN_PROVIDER_NAMES: set[str] = (
    set(_PROVIDER_LABELS.keys())
    | set(_PROVIDER_ALIASES.keys())
    | {"custom"}
)


def list_available_providers() -> list[dict[str, str]]:
    """Return info about all providers the user could use with ``provider:model``.

    Each dict has ``id``, ``label``, and ``aliases``.
    Checks which providers have valid credentials configured.

    Derives the provider list from :data:`CANONICAL_PROVIDERS` (single
    source of truth shared with ``pilotage model``, ``/model``, etc.).
    """
    # Derive display order from canonical list + custom
    provider_order = [p.slug for p in CANONICAL_PROVIDERS] + ["custom"]

    # Build reverse alias map
    aliases_for: dict[str, list[str]] = {}
    for alias, canonical in _PROVIDER_ALIASES.items():
        aliases_for.setdefault(canonical, []).append(alias)

    result = []
    for pid in provider_order:
        label = _PROVIDER_LABELS.get(pid, pid)
        alias_list = aliases_for.get(pid, [])
        # Check if this provider has credentials available
        has_creds = False
        try:
            from pilotage_cli.auth import get_auth_status
            if pid == "custom":
                custom_base_url = _get_custom_base_url() or ""
                has_creds = bool(custom_base_url.strip())
            else:
                status = get_auth_status(pid)
                has_creds = bool(status.get("logged_in") or status.get("configured"))
        except Exception:
            pass
        result.append({
            "id": pid,
            "label": label,
            "aliases": alias_list,
            "authenticated": has_creds,
        })
    return result


def parse_model_input(raw: str, current_provider: str) -> tuple[str, str]:
    """Parse ``/model`` input into ``(provider, model)``.

    Supports ``provider:model`` syntax to switch providers at runtime::

        openai-api:gpt-5.4                       →  ("openai-api", "gpt-5.4")
        custom:local:qwen                        →  ("custom:local", "qwen")
        gpt-5.4                                 →  (current_provider, "gpt-5.4")

    The colon is only treated as a provider delimiter if the left side is a
    recognized provider name or alias.  This avoids misinterpreting model names
    that happen to contain colons.

    Returns ``(provider, model)`` where *provider* is either the explicit
    provider from the input or *current_provider* if none was specified.
    """
    stripped = raw.strip()
    colon = stripped.find(":")
    if colon > 0:
        provider_part = stripped[:colon].strip().lower()
        model_part = stripped[colon + 1:].strip()
        if provider_part and model_part and provider_part in _KNOWN_PROVIDER_NAMES:
            # Support custom:name:model triple syntax for named custom
            # providers.  ``custom:local:qwen`` → ("custom:local", "qwen").
            # Single colon ``custom:qwen`` → ("custom", "qwen") as before.
            if provider_part == "custom" and ":" in model_part:
                second_colon = model_part.find(":")
                custom_name = model_part[:second_colon].strip()
                actual_model = model_part[second_colon + 1:].strip()
                if custom_name and actual_model:
                    return (f"custom:{custom_name}", actual_model)
            return (normalize_provider(provider_part), model_part)
    return (current_provider, stripped)


def _get_custom_base_url() -> str:
    """Get the custom endpoint base_url from config.yaml."""
    model_cfg = _get_model_config_dict()
    return str(model_cfg.get("base_url", "")).strip()


def _get_model_config_dict() -> dict[str, Any]:
    """Return the main model config mapping, or an empty dict."""
    try:
        from pilotage_cli.config import load_config
        config = load_config()
        model_cfg = config.get("model", {})
        if isinstance(model_cfg, dict):
            return model_cfg
    except Exception:
        pass
    return {}


def curated_models_for_provider(
    provider: Optional[str],
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return ``(model_id, description)`` tuples for a provider's model list.

    Tries to fetch the live model list from the provider's API first,
    falling back to the static ``_PROVIDER_MODELS`` catalog if the API
    is unreachable.
    """
    normalized = normalize_provider(provider)
    # Try live API first (Codex and OpenAI-compatible endpoints support /models)
    live = provider_model_ids(normalized)
    if live:
        return [(m, "") for m in live]

    # Fallback to static catalog
    models = _PROVIDER_MODELS.get(normalized, [])
    return [(m, "") for m in models]


def _provider_keys(provider: str) -> set[str]:
    key = (provider or "").strip().lower()
    normalized = normalize_provider(provider)
    return {k for k in (key, normalized) if k}


def _provider_catalog_names(provider: str) -> tuple[str, ...]:
    """Active picker models recognized for detection."""
    return tuple(_PROVIDER_MODELS.get(provider, []))


def _model_in_provider_catalog(name_lower: str, providers: set[str]) -> bool:
    return any(
        name_lower == model.lower()
        for provider in providers
        for model in _provider_catalog_names(provider)
    )


_AGGREGATOR_PROVIDERS: frozenset[str] = frozenset()

# Subscription/OAuth providers whose catalogs RE-EXPOSE other vendors' models
# would be listed here (tried only as a last resort for bare short-alias
# resolution, after every native-vendor catalog, so they never hijack an alias
# away from the model's native vendor). None are currently defined.
_BORROWED_MODEL_PROVIDERS: frozenset[str] = frozenset()

# Providers whose live /v1/models endpoint is the authoritative catalog, so
# the curated list is a discovery-only fallback (the picker would merge
# live-first for these). None are currently defined.
_LIVE_FIRST_PICKER_PROVIDERS: frozenset[str] = frozenset()


def _resolve_static_model_alias(
    name_lower: str,
    current_keys: set[str],
) -> Optional[tuple[str, str]]:
    """Resolve short aliases (e.g. sonnet/opus) using static catalogs only."""
    try:
        from pilotage_cli.model_switch import MODEL_ALIASES
    except Exception:
        return None

    identity = MODEL_ALIASES.get(name_lower)
    if identity is None:
        return None

    vendor = identity.vendor
    family = identity.family

    def _match(provider: str) -> Optional[str]:
        models = _PROVIDER_MODELS.get(provider, [])
        if not models:
            return None
        prefix = (
            f"{vendor}/{family}"
            if provider in _AGGREGATOR_PROVIDERS
            else family
        ).lower()
        for model in models:
            if model.lower().startswith(prefix):
                return model
        return None

    for provider in current_keys:
        if matched := _match(provider):
            return provider, matched

    for provider in _PROVIDER_MODELS:
        if (
            provider in current_keys
            or provider in _AGGREGATOR_PROVIDERS
            or provider in _BORROWED_MODEL_PROVIDERS
        ):
            continue
        if matched := _match(provider):
            return provider, matched

    for provider in _AGGREGATOR_PROVIDERS:
        if provider in current_keys and (matched := _match(provider)):
            return provider, matched

    # Last resort: providers that re-expose other vendors' models. Only reached
    # when no native-vendor catalog matched. None are currently defined
    # (_BORROWED_MODEL_PROVIDERS is empty).
    for provider in _BORROWED_MODEL_PROVIDERS:
        if provider in current_keys and (matched := _match(provider)):
            return provider, matched

    return None


def detect_static_provider_for_model(
    model_name: str,
    current_provider: str,
) -> Optional[tuple[str, str]]:
    """Auto-detect a provider from static catalogs only.

    Returns ``(provider_id, model_name)``. The model name may be remapped
    when a static alias or bare provider name resolves to a catalog default.
    Returns ``None`` when no confident match is found.
    """
    name = (model_name or "").strip()
    if not name:
        return None

    name_lower = name.lower()
    current_keys = _provider_keys(current_provider)

    alias_match = _resolve_static_model_alias(name_lower, current_keys)
    if alias_match:
        return alias_match

    # --- Step 0: bare provider name typed as model ---
    # If someone types `/model openai-api`, treat it as a provider switch and
    # pick the first model from that provider's catalog. Skip "custom" — it
    # has no model catalog.
    resolved_provider = _PROVIDER_ALIASES.get(name_lower, name_lower)
    if resolved_provider != "custom":
        default_models = _PROVIDER_MODELS.get(resolved_provider, [])
        if (
            resolved_provider in _PROVIDER_LABELS
            and default_models
            and resolved_provider not in current_keys
        ):
            # Route through the cost-safe default rather than picking
            # ``default_models[0]`` directly. For metered providers whose
            # curated list is ordered most-capable-first, entry [0] is also
            # the priciest flagship, and typing the bare provider name would
            # silently escalate to it — the exact billing footgun the
            # catalog-labeled silent default (``_SILENT_DEFAULT_PROVIDERS``)
            # exists to prevent. For providers outside that set this is
            # unchanged (it returns ``models[0]``).
            return (
                resolved_provider,
                get_default_model_for_provider(resolved_provider) or default_models[0],
            )

    # Aggregators list other providers' models — never auto-switch TO them
    # If the model belongs to the current provider's catalog, don't suggest switching
    if _model_in_provider_catalog(name_lower, current_keys):
        return None

    # --- Step 1: check static provider catalogs for a direct match ---
    # If the current provider is a custom endpoint (custom or custom:*), never
    # auto-switch away from it based on a static catalog match — the user
    # explicitly configured their own endpoint and the same model name may be
    # served there.
    _is_custom_current = (
        current_provider == "custom"
        or current_provider.startswith("custom:")
    )
    for pid in _PROVIDER_MODELS:
        if (
            pid in current_keys
            or pid in _AGGREGATOR_PROVIDERS
            or pid in _BORROWED_MODEL_PROVIDERS
        ):
            continue
        if _is_custom_current:
            continue
        if any(name_lower == m.lower() for m in _provider_catalog_names(pid)):
            return (pid, name)

    # Borrow-list providers (re-expose other vendors' models) only after every
    # native-vendor catalog, and only when one is the current provider.
    for pid in _BORROWED_MODEL_PROVIDERS:
        if pid in current_keys:
            continue
        if any(name_lower == m.lower() for m in _provider_catalog_names(pid)):
            return (pid, name)

    return None


def detect_provider_for_model(
    model_name: str,
    current_provider: str,
) -> Optional[tuple[str, str]]:
    """Auto-detect the best provider for a model name.

    Returns ``(provider_id, model_name)``.
    Returns ``None`` when no confident match is found.

    Priority:
    0. Bare provider name → switch to that provider's default model
    1. Direct provider static catalog match
    """
    name = (model_name or "").strip()
    if not name:
        return None

    static_match = detect_static_provider_for_model(name, current_provider)
    if static_match:
        return static_match
    if _model_in_provider_catalog(name.lower(), _provider_keys(current_provider)):
        return None

    return None


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to Pilotage' canonical provider ids.

    Note: ``"auto"`` passes through unchanged — use
    ``pilotage_cli.auth.resolve_provider()`` to resolve it to a concrete
    provider based on credentials and environment.
    """
    normalized = (provider or "openai-api").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def provider_label(provider: Optional[str]) -> str:
    """Return a human-friendly label for a provider id or alias."""
    original = (provider or "openai-api").strip()
    normalized = original.lower()
    if normalized == "auto":
        return "Auto"
    normalized = normalize_provider(normalized)
    return _PROVIDER_LABELS.get(normalized, original or "OpenAI API")


# Models that support OpenAI Priority Processing (service_tier="priority").
# See https://openai.com/api-priority-processing/ for the canonical list.
#
# Pattern-based matching — any OpenAI flagship model (gpt-*, o1*, o3*, o4*)
# is assumed to support Priority Processing. service_tier=priority is silently
# ignored by non-OpenAI endpoints (OpenAI-compatible proxies strip the field),
# so false positives are harmless. Codex-series models
# (gpt-5-codex, gpt-5.3-codex, etc.) are excluded — they don't expose the
# service_tier parameter through the Codex Responses API.
_OPENAI_FAST_MODE_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "o1",
    "o3",
    "o4",
)


def _is_openai_fast_model(model_id: Optional[str]) -> bool:
    """Return True if the model is an OpenAI flagship eligible for Priority Processing."""
    raw = _strip_vendor_prefix(str(model_id or ""))
    base = raw.split(":")[0]
    if not base:
        return False
    # Exclude Codex-series — they route through the Codex Responses API
    # which doesn't accept service_tier.
    if "codex" in base:
        return False
    return any(base.startswith(prefix) for prefix in _OPENAI_FAST_MODE_PREFIXES)


def _strip_vendor_prefix(model_id: str) -> str:
    """Strip vendor/ prefix from a model ID (e.g. 'openai/gpt-5.4' -> 'gpt-5.4')."""
    raw = str(model_id or "").strip().lower()
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    return raw


def model_supports_fast_mode(model_id: Optional[str]) -> bool:
    """Return whether Pilotage should expose the /fast toggle for this model."""
    return _is_openai_fast_model(model_id)


def resolve_fast_mode_overrides(model_id: Optional[str]) -> dict[str, Any] | None:
    """Return request_overrides for fast/priority mode, or None if unsupported.

    Returns ``{"service_tier": "priority"}`` (OpenAI Priority Processing) for
    eligible OpenAI models. The overrides are injected into the API request
    kwargs by ``_build_api_kwargs`` in run_agent.py — the OpenAI/Codex paths
    handle the service_tier key.
    """
    if not model_supports_fast_mode(model_id):
        return None
    return {"service_tier": "priority"}


# Providers where models.dev is treated as authoritative: curated static
# lists are kept only as an offline fallback and to capture custom additions
# the registry doesn't publish yet. Adding a provider here causes its
# curated list to be merged with fresh models.dev entries (fresh first, any
# curated-only names appended) for both the CLI and the gateway /model picker.
#
# Empty in the OpenAI-only registry: openai-codex / openai-api handle catalog
# freshness through their own live endpoints below, and custom endpoints are
# probed live. Kept as the mechanism so a future provider can opt in with a
# one-line addition.
_MODELS_DEV_PREFERRED: frozenset[str] = frozenset()


def _model_dedup_key(model_id: str) -> str:
    """Case-insensitive dedup key that also folds picker-search aliases.

    Some providers serve the same model under both a curated public slug and
    a bare live wire id. Folding through the search-alias
    table keeps the curated-first merge from emitting both as separate rows.
    The row that survives is the primary list's entry; selection still sends
    whichever id the surviving row carries.
    """
    key = str(model_id).strip().lower()
    try:
        from pilotage_cli.model_search import model_alias_canonical
        return model_alias_canonical(key)
    except Exception:
        return key


def _merge_with_models_dev(provider: str, curated: list[str]) -> list[str]:
    """Merge curated list with fresh models.dev entries for a preferred provider.

    Returns models.dev entries first (in models.dev order), then any
    curated-only entries appended. Preserves case for curated fallbacks
    while trusting models.dev for newer variants.

    If models.dev is unreachable or returns nothing, the curated list is
    returned unchanged — this is the offline/CI fallback path.
    """
    try:
        from agent.models_dev import list_agentic_models
        mdev = list_agentic_models(provider)
    except Exception:
        mdev = []

    if not mdev:
        return list(curated)

    # Case-insensitive dedup while preserving order and curated casing.
    seen_lower: set[str] = set()
    merged: list[str] = []
    for mid in mdev:
        key = str(mid).lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        merged.append(mid)
    for mid in curated:
        key = str(mid).lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        merged.append(mid)
    return merged


def _openai_discovery_base_url(provider: str) -> str:
    """Effective OpenAI endpoint for model discovery.

    Mirrors the runtime precedence so discovery probes the SAME endpoint
    inference uses: ``$OPENAI_BASE_URL`` (explicit env override) →
    ``model.base_url`` from config.yaml when the configured provider matches
    → the canonical default. Previously this read the env var only, so a
    config-set data-residency host (``us.api.openai.com``) was ignored and
    the catalog kept coming from ``api.openai.com``.
    """
    env_raw = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    if env_raw:
        return env_raw
    try:
        model_cfg = _get_model_config_dict()
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        if cfg_provider in ("openai", "openai-api") and normalize_provider(provider) == normalize_provider(cfg_provider):
            cfg_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
            if cfg_url:
                return cfg_url
    except Exception:
        pass
    return "https://api.openai.com/v1"


def provider_model_ids(provider: Optional[str], *, force_refresh: bool = False) -> list[str]:
    """Return the best known model catalog for a provider.

    Tries live API endpoints for providers that support them (Codex, OpenAI,
    custom endpoints), falling back to static lists. For providers in
    ``_MODELS_DEV_PREFERRED``, models.dev entries are merged on top of
    curated so new models released on the platform appear in ``/model``
    without a Pilotage release.
    """
    normalized = normalize_provider(provider)
    if normalized == "openai-codex":
        from pilotage_cli.codex_models import get_codex_model_ids

        # Pass the live OAuth access token so the picker matches whatever
        # ChatGPT lists for this account right now (new models appear without
        # a Pilotage release). Falls back to the hardcoded catalog if no token
        # or the endpoint is unreachable.
        access_token = None
        try:
            from pilotage_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
            access_token = creds.get("api_key")
        except Exception:
            access_token = None
        return get_codex_model_ids(access_token=access_token)
    if normalized in ("openai", "openai-api"):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            base = _openai_discovery_base_url(normalized)
            # Custom OpenAI-compatible endpoints (proxies, gateways, self-hosted)
            # may serve a small curated catalog — use the live list verbatim so
            # discovery works. But the official OpenAI hosts (canonical AND the
            # data-residency regional hosts, which serve the identical dump)
            # return 120+ entries of embeddings, whisper, tts, dall-e,
            # moderation and legacy chat models — none of which belong in the
            # agent model picker. For official hosts, intersect the live list
            # with our curated agentic catalog so ``/model`` matches what
            # ``pilotage model`` shows.
            from pilotage_cli.providers import is_official_openai_host

            is_default_openai = is_official_openai_host(base)
            try:
                live = fetch_api_models(api_key, base)
                if live:
                    if is_default_openai:
                        live_lower = {m.lower() for m in live}
                        curated = list(_PROVIDER_MODELS.get(normalized, []))
                        # Keep curated order; only surface curated models the
                        # account actually has access to.
                        filtered = [m for m in curated if m.lower() in live_lower]
                        if filtered:
                            return filtered
                        # Account serves none of the curated models (rare —
                        # e.g. org without GPT-5 access). Fall back to curated
                        # so the picker still offers sane defaults.
                        return curated or live
                    return live
            except Exception:
                pass
    if normalized == "custom":
        base_url = _get_custom_base_url()
        if base_url:
            model_cfg = _get_model_config_dict()
            # Try common API key env vars for custom endpoints
            api_key = (
                str(model_cfg.get("api_key", "") or "").strip()
                or os.getenv("CUSTOM_API_KEY", "")
                or os.getenv("OPENAI_API_KEY", "")
            )
            live = fetch_api_models(api_key, base_url)
            if live:
                return live
    # ── Profile-based generic live fetch (all simple api-key providers) ──
    # Handles any provider registered in providers/ with auth_type="api_key".
    # Replaces per-provider copy-paste blocks.
    try:
        from providers import get_provider_profile
        from pilotage_cli.auth import resolve_api_key_provider_credentials

        _p = get_provider_profile(normalized)
        if _p and _p.auth_type == "api_key" and _p.base_url:
            try:
                creds = resolve_api_key_provider_credentials(normalized)
                api_key = str(creds.get("api_key") or "").strip()
                base_url = str(creds.get("base_url") or "").strip()
            except Exception:
                api_key, base_url = "", _p.base_url
            if not base_url:
                base_url = _p.base_url
            if api_key:
                live = _p.fetch_models(api_key=api_key, base_url=base_url or None)
                if live:
                    # Merge static curated list with live API results so
                    # models that the live endpoint omits (stale cache,
                    # partial rollout) still appear in the picker.
                    # Curated-first: the curated list leads so the picker's
                    # agentic ordering is preserved; live-only entries are
                    # appended after it.
                    #
                    # Plugin providers with no static _PROVIDER_MODELS entry fall
                    # back to the profile's curated fallback_models so their
                    # agentic picks lead the picker instead of whatever the live
                    # catalog happens to return first.
                    curated = list(_PROVIDER_MODELS.get(normalized, [])) or list(
                        _p.fallback_models or ()
                    )
                    if curated:
                        if normalized in _LIVE_FIRST_PICKER_PROVIDERS:
                            primary, secondary = live, curated
                        else:
                            primary, secondary = curated, live
                        merged = list(primary)
                        merged_lower = {_model_dedup_key(m) for m in primary}
                        for m in secondary:
                            if _model_dedup_key(m) not in merged_lower:
                                merged.append(m)
                                merged_lower.add(_model_dedup_key(m))
                        return merged
                    return live
            # Use profile's fallback_models if defined
            if _p.fallback_models:
                return list(_p.fallback_models)
    except Exception:
        pass

    curated_static = list(_PROVIDER_MODELS.get(normalized, []))
    if normalized in _MODELS_DEV_PREFERRED:
        merged = _merge_with_models_dev(normalized, curated_static)
        return merged
    return curated_static


# ---------------------------------------------------------------------------
# Generic disk cache for provider_model_ids() — keeps /model picker fast.
# ---------------------------------------------------------------------------
#
# Without this layer, every /model picker open re-fetches every authed
# provider's /v1/models endpoint. On a user with several configured
# providers that's multiple seconds of cold HTTP roundtrips just to
# render the provider list.
#
# Cache strategy:
#   - One JSON file at $PILOTAGE_HOME/provider_models_cache.json
#   - Per-provider entries keyed by (provider, credential fingerprint)
#   - Credential fingerprint = sha256 of env-var values that the provider
#     normally reads. Swap your OPENAI_API_KEY and the entry invalidates.
#   - 1h TTL by default. `force_refresh=True` skips the cache entirely
#     and overwrites it on success.
#   - Only NON-EMPTY results are cached. An empty/None response from a
#     transient network error never gets pinned.
#   - Cache file is best-effort. Any read/write error degrades silently
#     to a live fetch — the picker keeps working.

_PROVIDER_MODELS_CACHE_TTL = 3600  # 1h
# Stale-while-revalidate window: an expired-but-same-credentials entry is
# served IMMEDIATELY (picker opens stay instant) while a background daemon
# thread re-fetches the live catalog and rewrites the disk cache for the
# next open. Beyond this bound the entry is considered too old to trust and
# the caller blocks on a live fetch as before. Rationale: the /model picker's
# provider listing runs 8-9 serial /v1/models round-trips (~2-3s) whenever
# the 1h TTL lapses mid-session — model catalogs change on release timescales,
# not hourly, so serving hour-old data while refreshing off-thread is strictly
# better than stalling every picker surface (CLI, TUI, dashboard, gateway).
_PROVIDER_MODELS_STALE_SERVE_MAX = 7 * 24 * 3600  # 7d

# Providers with a background SWR refresh currently in flight — dedupes
# concurrent refreshes so repeated picker opens during one refresh don't
# stack threads or duplicate network calls.
_swr_refresh_inflight: set = set()
_swr_refresh_lock = threading.Lock()


def _spawn_swr_refresh(cache_key: str, refresh_fn=None) -> None:
    """Kick a background refresh of *cache_key*'s model-id cache entry.

    Fire-and-forget daemon thread; at most one in flight per cache key.
    Failures are swallowed — the stale entry stays served until a later
    refresh succeeds (same degradation the blocking path already had).

    ``refresh_fn`` (no-args, returns the fresh cache-entry dict or ``None``)
    lets non-slug keys (``custom:<base_url>`` entries from
    :func:`cached_fetch_api_models`) reuse the same inflight-dedupe and
    thread scaffolding. When omitted, *cache_key* is treated as a
    ``PROVIDER_REGISTRY`` slug and refreshed via :func:`provider_model_ids`
    (the original behavior).
    """
    with _swr_refresh_lock:
        if cache_key in _swr_refresh_inflight:
            return
        _swr_refresh_inflight.add(cache_key)

    def _default_refresh():
        live = provider_model_ids(cache_key, force_refresh=True)
        if not live:
            return None
        return {
            "fp": _credential_fingerprint(cache_key),
            "at": time.time(),
            "models": list(live),
        }

    def _refresh() -> None:
        try:
            entry = (refresh_fn or _default_refresh)()
            if entry:
                cache = _load_provider_models_cache()
                cache[cache_key] = entry
                _save_provider_models_cache(cache)
        except Exception:
            logger.debug("SWR refresh failed for %s", cache_key, exc_info=True)
        finally:
            with _swr_refresh_lock:
                _swr_refresh_inflight.discard(cache_key)

    threading.Thread(
        target=_refresh, daemon=True, name=f"model-cache-swr-{cache_key}"
    ).start()


def _provider_models_cache_path() -> Path:
    from pilotage_constants import get_pilotage_home
    return get_pilotage_home() / "provider_models_cache.json"


def _credential_fingerprint(provider: str) -> str:
    """Return a short hash representing the credentials that
    ``provider_model_ids(provider)`` would see right now.

    Rotating any of the relevant env vars invalidates the cached entry
    for that provider. We hash AT LEAST the api-key + base-url env vars
    declared in ``PROVIDER_REGISTRY``. For OAuth-backed providers
    (codex), the relevant tokens live in ``$PILOTAGE_HOME/auth.json``
    and external credential files. Rather than parse every shape, we
    additionally fold the mtime of those files into the fingerprint so
    refreshes after re-auth bust the cache.
    """
    import hashlib
    import os as _os

    parts: list[str] = []

    # Env vars from PROVIDER_REGISTRY for this slug
    try:
        from pilotage_cli.auth import PROVIDER_REGISTRY
        pcfg = PROVIDER_REGISTRY.get(provider)
        if pcfg is not None:
            for ev in getattr(pcfg, "api_key_env_vars", ()) or ():
                parts.append(f"{ev}={_os.environ.get(ev, '')}")
            bev = getattr(pcfg, "base_url_env_var", "") or ""
            if bev:
                parts.append(f"{bev}={_os.environ.get(bev, '')}")
    except Exception:
        pass

    # Effective configured endpoint: config.yaml's model.base_url changes the
    # endpoint discovery probes (data-residency hosts) without touching any
    # env var, so it must change the fingerprint too or `pilotage config set
    # model.base_url ...` keeps serving the previous endpoint's cached
    # catalog until TTL expiry.
    if provider in ("openai", "openai-api"):
        try:
            parts.append(f"effective_base={_openai_discovery_base_url(provider)}")
        except Exception:
            pass

    # OAuth / external-file mtimes that change on re-auth
    try:
        from pilotage_constants import get_pilotage_home
        for rel in ("auth.json", "credentials.json"):
            p = get_pilotage_home() / rel
            try:
                parts.append(f"{rel}@{p.stat().st_mtime_ns}")
            except FileNotFoundError:
                parts.append(f"{rel}@missing")
            except Exception:
                pass
    except Exception:
        pass

    # External well-known credential file locations
    for path in (
        _os.path.expanduser("~/.codex/auth.json"),
    ):
        try:
            mt = _os.stat(path).st_mtime_ns
            parts.append(f"{path}@{mt}")
        except FileNotFoundError:
            parts.append(f"{path}@missing")
        except Exception:
            pass

    blob = "|".join(parts).encode("utf-8", errors="replace")
    # blake2b for cache-key fingerprinting only — not for credential storage.
    # We never reverse this hash; collisions are harmless (worst case: cache
    # miss → live re-fetch). Use blake2b instead of sha256 here because
    # CodeQL's `py/weak-sensitive-data-hashing` rule flags sha256 over env
    # vars whose names contain "API_KEY" / "TOKEN" even when the hash is
    # used as an identity fingerprint, not for password storage. blake2b
    # is a keyed-hash primitive and isn't flagged.
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def _load_provider_models_cache() -> dict:
    """Return the full cache dict, or {} on any error."""
    try:
        path = _provider_models_cache_path()
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_cache_write_lock = threading.Lock()


def _save_provider_models_cache(data: dict) -> None:
    """Persist the cache dict. Best-effort — silent on any error."""
    try:
        from utils import atomic_json_write
        path = _provider_models_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, data, indent=None)
    except Exception:
        pass


def update_provider_cache_entry(provider: str, models: list[str]) -> None:
    """Thread-safe single-entry update of the provider-models disk cache.

    Used by parallel prefetch workers so concurrent fetches don't clobber
    each other's writes via read-modify-write races on the shared JSON file.
    Each worker loads the latest cache state under the lock, writes its own
    entry, and saves — best-effort, silent on any error.
    """
    try:
        normalized = normalize_provider(provider) or (provider or "")
        if not normalized or not models:
            return
        fp = _credential_fingerprint(normalized)
        with _cache_write_lock:
            cache = _load_provider_models_cache()
            cache[normalized] = {
                "fp": fp,
                "at": time.time(),
                "models": list(models),
            }
            _save_provider_models_cache(cache)
    except Exception:
        pass


def cached_provider_model_ids(
    provider: Optional[str],
    *,
    force_refresh: bool = False,
    ttl_seconds: int = _PROVIDER_MODELS_CACHE_TTL,
) -> list[str]:
    """Disk-cached wrapper around :func:`provider_model_ids`.

    Hits the cache when fresh; otherwise calls the live function and
    persists a non-empty result. Always returns a list (never None).
    """
    normalized = normalize_provider(provider) or (provider or "")
    if not normalized:
        return []

    cache = _load_provider_models_cache()
    fp = _credential_fingerprint(normalized)
    entry = cache.get(normalized)
    now = time.time()

    if not force_refresh and _cache_entry_valid(entry, fp):
        age = now - entry["at"]
        if age < ttl_seconds:
            return list(entry["models"])
        if age < _PROVIDER_MODELS_STALE_SERVE_MAX:
            # Stale-while-revalidate: serve the expired entry immediately so
            # interactive picker opens never block on serial /v1/models
            # round-trips; refresh the cache off-thread for the next open.
            _spawn_swr_refresh(normalized)
            return list(entry["models"])

    # Cache miss / stale / forced refresh — call the live path.
    live = provider_model_ids(normalized, force_refresh=force_refresh)
    if live:
        cache[normalized] = {
            "fp": fp,
            "at": now,
            "models": list(live),
        }
        _save_provider_models_cache(cache)
        return list(live)

    # Live fetch returned nothing. If we have a stale entry with the
    # SAME fingerprint, prefer it over an empty result — stale data
    # beats no data when the network is flaky.
    if _cache_entry_valid(entry, fp):
        return list(entry["models"])
    return list(live or [])


def clear_provider_models_cache(provider: Optional[str] = None) -> None:
    """Drop a single provider's cache entry, or wipe the whole cache.

    ``provider=None`` wipes everything; otherwise only that provider's
    entry is removed. Used by ``/model --refresh`` and
    ``pilotage model --refresh``.
    """
    try:
        if provider is None:
            path = _provider_models_cache_path()
            if path.exists():
                path.unlink()
            return
        cache = _load_provider_models_cache()
        normalized = normalize_provider(provider) or provider or ""
        if normalized in cache:
            del cache[normalized]
            _save_provider_models_cache(cache)
    except Exception:
        pass


def probe_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
    request_headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Probe a ``/models`` endpoint with light URL heuristics.

    Authenticated probes send ``Authorization: Bearer``; the response
    shape (``data[].id``) follows the OpenAI-compatible convention.
    """
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return {
            "models": None,
            "probed_url": None,
            "resolved_base_url": "",
            "suggested_base_url": None,
            "used_fallback": False,
        }

    if normalized.endswith("/v1"):
        alternate_base = normalized[:-3].rstrip("/")
    else:
        alternate_base = normalized + "/v1"

    candidates: list[tuple[str, bool]] = [(normalized, False)]
    if alternate_base and alternate_base != normalized:
        candidates.append((alternate_base, True))

    tried: list[str] = []
    headers: dict[str, str] = {"User-Agent": _PILOTAGE_USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if isinstance(request_headers, dict):
        # Per-provider custom headers can contain auth/proxy secrets. Merge
        # last so endpoint-specific config wins, and never log the values.
        from pilotage_cli.config import normalize_extra_headers

        headers.update(normalize_extra_headers(request_headers))

    _ssl_context = _custom_provider_ssl_context(normalized)
    for candidate_base, is_fallback in candidates:
        url = candidate_base.rstrip("/") + "/models"
        tried.append(url)
        req = urllib.request.Request(url, headers=headers)
        # Only thread ssl_context when a per-provider TLS override actually
        # applies. Public/unconfigured endpoints keep the original 2-arg call,
        # so nothing changes for them (and existing call-seam mocks stay valid).
        _open_kwargs: dict[str, Any] = {"timeout": timeout}
        if _ssl_context is not None:
            _open_kwargs["ssl_context"] = _ssl_context
        try:
            with _urlopen_model_catalog_request(req, **_open_kwargs) as resp:
                data = json.loads(resp.read().decode())
                return {
                    "models": [m.get("id", "") for m in data.get("data", [])],
                    "probed_url": url,
                    "resolved_base_url": candidate_base.rstrip("/"),
                    "suggested_base_url": alternate_base if alternate_base != candidate_base else normalized,
                    "used_fallback": is_fallback,
                }
        except Exception:
            continue

    return {
        "models": None,
        "probed_url": tried[0] if tried else normalized.rstrip("/") + "/models",
        "resolved_base_url": normalized,
        "suggested_base_url": alternate_base if alternate_base != normalized else None,
        "used_fallback": False,
    }


# Legacy filter — used when an item has no surface tag (rolling out
# 2026-05). Once every model returned by the catalog endpoint carries an
# explicit surface tag (``chat``/``embed``/``image-gen``/``tts``/``stt``)
# the regex path becomes unreachable and can be removed.
_DEEPINFRA_EXCLUDE_RE = re.compile(
    r"(?i)(embed|rerank|whisper|stable-diffusion|flux|sdxl|"
    r"tts|bark|speech|image-gen|clip|vit-|dpt-)",
)

# Surface tags announce *what kind of model* this is. When none of these
# are present on a catalog entry, the tags array only carries capability
# tags (``reasoning``, ``vision``, ``prompt_cache``, …) and we have to
# fall back to id-regex inference for the chat surface.
_DEEPINFRA_SURFACE_TAGS: frozenset[str] = frozenset({
    "chat", "embed", "image-gen", "tts", "stt", "video-gen",
})

_DEEPINFRA_DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
_DEEPINFRA_MODELS_QUERY = "filter=true&sort_by=pilotage"

# Module-level cache for the full tagged catalog response, keyed by base URL.
# Each value is the parsed ``data`` list. Surface-specific filters read from
# this cache so a single network round-trip serves chat / image-gen / tts /
# stt callers across the whole process lifetime.
_deepinfra_catalog_cache: dict[str, list[dict]] = {}

# Negative cache: monotonic timestamp of the last failed fetch, keyed by base
# URL. Without this, an unreachable catalog (offline / DNS / firewall) makes
# every surface helper (chat picker, pricing, image/video/tts/stt defaults,
# vision) re-attempt a fresh blocking fetch that eats the full timeout each
# time — several sequential stalls in one user-visible operation. A short TTL
# lets connectivity recover without a process restart.
_deepinfra_catalog_neg_cache: dict[str, float] = {}
_DEEPINFRA_CATALOG_NEG_TTL = 60.0  # seconds


def _deepinfra_catalog_url() -> tuple[str, str]:
    """Return ``(cache_key, full_url)`` for the DeepInfra catalog endpoint."""
    base = os.getenv("DEEPINFRA_BASE_URL", "").strip() or _DEEPINFRA_DEFAULT_BASE_URL
    cache_key = base.rstrip("/")
    return cache_key, f"{cache_key}/models?{_DEEPINFRA_MODELS_QUERY}"


def _fetch_deepinfra_catalog(
    *,
    timeout: float = 5.0,
    force_refresh: bool = False,
) -> Optional[list[dict]]:
    """Fetch the raw DeepInfra catalog list with module-level caching.

    The endpoint serves chat + embed + image-gen + tts + stt models in one
    response. Authentication is optional but Bearer-attached when available
    so user-scoped catalogs (private fine-tunes etc.) are visible.
    """
    cache_key, url = _deepinfra_catalog_url()
    if not force_refresh:
        if cache_key in _deepinfra_catalog_cache:
            return _deepinfra_catalog_cache[cache_key]
        last_fail = _deepinfra_catalog_neg_cache.get(cache_key)
        if last_fail is not None and (time.monotonic() - last_fail) < _DEEPINFRA_CATALOG_NEG_TTL:
            return None

    headers: dict[str, str] = {"User-Agent": _PILOTAGE_USER_AGENT}
    api_key = os.getenv("DEEPINFRA_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _deepinfra_catalog_neg_cache[cache_key] = time.monotonic()
        return None

    data = payload.get("data")
    if not isinstance(data, list):
        _deepinfra_catalog_neg_cache[cache_key] = time.monotonic()
        return None

    _deepinfra_catalog_cache[cache_key] = data
    _deepinfra_catalog_neg_cache.pop(cache_key, None)
    return data


def _fetch_deepinfra_models_by_tag(
    tag: str,
    *,
    timeout: float = 5.0,
    force_refresh: bool = False,
) -> Optional[list[dict]]:
    """Return DeepInfra models whose ``metadata.tags`` includes *tag*.

    Each returned item is ``{"id": str, "metadata": dict}`` so callers can
    inspect context length, pricing, default dimensions (image-gen),
    pricing units (tts ``input_characters``, stt ``input_seconds``), etc.

    For the chat surface, items without any ``tags`` field fall through
    to the legacy name-regex exclusion so this keeps working while the
    tag rollout (mid-2026) is still in flight.

    Returns ``None`` on network failure.
    """
    data = _fetch_deepinfra_catalog(timeout=timeout, force_refresh=force_refresh)
    if data is None:
        return None

    matched: list[dict] = []
    for item in data:
        mid = item.get("id")
        if not mid:
            continue
        # ``metadata is None`` means DeepInfra returns a stub without
        # pricing/context — typically a model that's listed but not
        # served. Skip those for every surface.
        raw_metadata = item.get("metadata")
        if raw_metadata is None:
            continue
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        raw_tags = metadata.get("tags")
        tags = raw_tags if isinstance(raw_tags, list) else []
        has_surface_tag = any(t in _DEEPINFRA_SURFACE_TAGS for t in tags)

        if has_surface_tag:
            if tag in tags:
                matched.append({"id": mid, "metadata": metadata})
            continue
        # Surface-tag rollout incomplete — fall back to id-regex inference.
        # Only meaningful for the chat surface; embed/image-gen/tts/stt
        # cannot be safely inferred from an id alone.
        if tag == "chat" and not _DEEPINFRA_EXCLUDE_RE.search(mid):
            matched.append({"id": mid, "metadata": metadata})

    return matched


def deepinfra_model_ids(tag: str, *, force_refresh: bool = False) -> list[str]:
    """Return DeepInfra model ids carrying surface *tag* (``[]`` on failure).

    Single source of truth for the per-surface model shims (TTS/STT/vision),
    replacing the copy-pasted ``import _fetch_deepinfra_models_by_tag → fetch
    → [item["id"] …]`` wrapper each of them used to carry.
    """
    items = _fetch_deepinfra_models_by_tag(tag, force_refresh=force_refresh)
    return [item["id"] for item in items] if items else []


def deepinfra_base_url(section: Optional[dict] = None) -> str:
    """Resolve the DeepInfra OpenAI-compatible base URL, normalized.

    Precedence: config-section ``base_url`` → ``DEEPINFRA_BASE_URL`` env →
    default. Always stripped with any trailing slash removed. Single source
    of truth for the base-URL chain the TTS/STT/image/video shims each used
    to re-code (with subtly divergent normalization).
    """
    candidate = section.get("base_url") if isinstance(section, dict) else None
    value = candidate or os.getenv("DEEPINFRA_BASE_URL") or _DEEPINFRA_DEFAULT_BASE_URL
    return str(value).strip().rstrip("/")


def fetch_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Fetch the list of available model IDs from the provider's ``/models`` endpoint.

    Returns a list of model ID strings, or ``None`` if the endpoint could not
    be reached (network error, timeout, auth failure, etc.).
    """
    return probe_api_models(
        api_key,
        base_url,
        timeout=timeout,
        api_mode=api_mode,
        request_headers=headers,
    ).get("models")


def _custom_endpoint_fingerprint(
    api_key: Optional[str],
    api_mode: Optional[str],
    headers: Optional[dict[str, str]],
) -> str:
    """Fingerprint the credentials/wire-shape used to probe a custom endpoint.

    Custom OpenAI-compatible endpoints have no ``PROVIDER_REGISTRY`` slug to
    key off (unlike ``_credential_fingerprint``), so this hashes exactly the
    values callers pass to :func:`fetch_api_models`: a rotated ``api_key``, a
    changed ``api_mode``, or an edited ``extra_headers`` block each bust the
    cache entry on their own.
    """
    import hashlib

    blob = "|".join((
        api_key or "",
        api_mode or "",
        json.dumps(headers or {}, sort_keys=True),
    )).encode("utf-8", errors="replace")
    # blake2b for cache-key fingerprinting only, same rationale as
    # _credential_fingerprint (avoids CodeQL's sha256-over-secrets rule).
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def _cache_entry_valid(entry: Any, fp: str) -> "TypeGuard[dict[str, Any]]":
    """True when *entry* is a well-formed cache row for fingerprint *fp*.

    Requires a numeric ``at`` so corrupt disk state (hand-edited JSON with
    ``"at": "yesterday"`` or ``null``) degrades to a cache miss / live fetch
    instead of raising out of the wrapper.
    """
    return (
        isinstance(entry, dict)
        and entry.get("fp") == fp
        and isinstance(entry.get("models"), list)
        and bool(entry["models"])
        and isinstance(entry.get("at"), (int, float))
        and not isinstance(entry.get("at"), bool)
    )


def cached_fetch_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    *,
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    force_refresh: bool = False,
    cache_only: bool = False,
    ttl_seconds: int = _PROVIDER_MODELS_CACHE_TTL,
) -> Optional[list[str]]:
    """Disk-cached wrapper around :func:`fetch_api_models` for custom endpoints.

    Mirrors :func:`cached_provider_model_ids` — including its
    stale-while-revalidate tier — but keys ``provider_models_cache.json``
    off ``custom:<base_url>`` instead of a ``PROVIDER_REGISTRY`` slug, since
    custom endpoints (named ``custom_providers`` rows, bare
    ``provider: custom``, and per-endpoint-map entries) have none. Same
    stale-beats-nothing fallback policy: a live-fetch failure serves the
    last same-fingerprint result rather than an empty list. Returns whatever
    :func:`fetch_api_models` would (a list or ``None``); corrupt cache rows
    degrade to a live fetch instead of raising.

    ``cache_only`` serves a previously-discovered catalog without touching
    the network at all — no live fetch, no background revalidation — and
    returns ``None`` when nothing usable is cached. Callers that deliberately
    skip live probing for latency reasons (GUI picker opens, which must not
    block on a stopped local endpoint) use this so a warm catalog still
    reaches the picker instead of collapsing to the config-declared subset.
    """
    normalized_url = str(base_url or "").strip().rstrip("/").lower()
    if not normalized_url:
        if cache_only:
            return None
        # No base_url means nothing to key the cache on — fall through to a
        # live call so callers keep getting fetch_api_models' own behavior.
        return fetch_api_models(
            api_key, base_url, timeout=timeout, api_mode=api_mode, headers=headers
        )

    cache_key = f"custom:{normalized_url}"
    fp = _custom_endpoint_fingerprint(api_key, api_mode, headers)
    cache = _load_provider_models_cache()
    entry = cache.get(cache_key)
    now = time.time()

    if cache_only:
        # Same trust window as the stale-while-revalidate tier below, minus
        # the revalidation: an entry this side of the bound is good enough to
        # render, and anything older is treated as a miss so the caller falls
        # back to its configured list rather than showing a stale catalog.
        if force_refresh or not _cache_entry_valid(entry, fp):
            return None
        if now - entry["at"] >= _PROVIDER_MODELS_STALE_SERVE_MAX:
            return None
        return list(entry["models"])

    if not force_refresh and _cache_entry_valid(entry, fp):
        age = now - entry["at"]
        if age < ttl_seconds:
            return list(entry["models"])
        if age < _PROVIDER_MODELS_STALE_SERVE_MAX:
            # Stale-while-revalidate: serve the expired entry immediately so
            # picker opens never block on a live /v1/models round-trip
            # 's stall class, which a plain TTL would reintroduce an
            # hour into the session); refresh off-thread for the next open.
            def _refresh_custom():
                live = fetch_api_models(
                    api_key, base_url,
                    timeout=timeout, api_mode=api_mode, headers=headers,
                )
                if not live:
                    return None
                return {"fp": fp, "at": time.time(), "models": list(live)}

            _spawn_swr_refresh(cache_key, _refresh_custom)
            return list(entry["models"])

    live = fetch_api_models(
        api_key, base_url, timeout=timeout, api_mode=api_mode, headers=headers
    )
    if live:
        cache[cache_key] = {"fp": fp, "at": now, "models": list(live)}
        _save_provider_models_cache(cache)
        return list(live)

    # Live fetch returned nothing (offline endpoint, timeout, auth hiccup).
    # A stale same-fingerprint entry beats an empty result.
    if _cache_entry_valid(entry, fp):
        return list(entry["models"])
    return live


def validate_requested_model(
    model_name: str,
    provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> dict[str, Any]:
    """
    Validate a ``/model`` value for the active provider.

    Performs format checks first, then probes the live API to confirm
    the model actually exists.

    Returns a dict with:
      - accepted: whether the CLI should switch to the requested model now
      - persist: whether it is safe to save to config
      - recognized: whether it matched a known provider catalog
      - message: optional warning / guidance for the user
    """
    requested = (model_name or "").strip()
    normalized = normalize_provider(provider)
    requested_for_lookup = requested

    if not requested:
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "Model name cannot be empty.",
        }

    if any(ch.isspace() for ch in requested):
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "Model names cannot contain spaces.",
        }


    if normalized == "custom" or normalized.startswith("custom:"):
        probe = probe_api_models(api_key, base_url)
        api_models = probe.get("models")
        if api_models is not None:
            if requested_for_lookup in set(api_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }

            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, api_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }

            suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)

            message = (
                f"Note: `{requested}` was not found in this custom endpoint's model listing "
                f"({probe.get('probed_url')}). It may still work if the server supports hidden or aliased models."
                f"{suggestion_text}"
            )
            if probe.get("used_fallback"):
                message += (
                    f"\n  Endpoint verification succeeded after trying `{probe.get('resolved_base_url')}`. "
                    f"Consider saving that as your base URL."
                )

            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": message,
            }

        message = (
            f"Note: could not reach this custom endpoint's model listing at `{probe.get('probed_url')}`. "
            f"Pilotage will still save `{requested}`, but the endpoint should expose `/models` for verification."
        )
        if probe.get("suggested_base_url"):
            message += f"\n  If this server expects `/v1`, try base URL: `{probe.get('suggested_base_url')}`"

        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": message,
        }

    # Providers with non-standard catalog validation — /v1/models probing is not the right path.
    if normalized == "openai-codex":
        try:
            catalog_models = provider_model_ids(normalized)
        except Exception:
            catalog_models = []
        if catalog_models:
            if requested_for_lookup in set(catalog_models):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": None,
                }
            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, catalog_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }
            suggestions = get_close_matches(requested_for_lookup, catalog_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)
            provider_label = "OpenAI Codex"
            # Plausibility gate: the soft-accept exists
            # for entitlement-gated *hidden* slugs the curated listing hasn't
            # caught up with — but those are always the provider's own family
            # (openai-codex -> gpt-*). Accepting an
            # unrelated typed name (e.g. `llama-3.1-8b`) here turns
            # what should be an actionable "did you mean --provider <x>?" error
            # into a confusing success that 400s on the next turn. Only soft-
            # accept names that share the provider's family prefix; reject the
            # rest with guidance to pin the right provider.
            _family_prefixes = {
                "openai-codex": ("gpt-", "codex-", "o1", "o3", "o4"),
            }.get(normalized, ())
            _lower = requested_for_lookup.strip().lower()
            _plausible = (not _family_prefixes) or any(
                _lower.startswith(p) for p in _family_prefixes
            )
            if not _plausible:
                return {
                    "accepted": False,
                    "persist": False,
                    "recognized": False,
                    "message": (
                        f"`{requested}` doesn't look like a {provider_label} model "
                        f"and isn't in its listing, so it was not accepted. If it "
                        f"belongs to another configured provider, switch with "
                        f"`--provider <slug>` (or select it from the `/model` "
                        f"picker)."
                        f"{suggestion_text}"
                    ),
                }
            return {
                "accepted": True,
                "persist": True,
                "recognized": False,
                "message": (
                    f"Note: `{requested}` was not found in the {provider_label} model listing. "
                    "It may still work if your account has access to a newer or hidden model ID."
                    f"{suggestion_text}"
                ),
            }

    # Probe the live API to check if the model actually exists
    api_models = fetch_api_models(api_key, base_url)

    if api_models is not None:
        if requested_for_lookup in set(api_models):
            # API confirmed the model exists
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        else:
            # API responded but model is not listed.  Accept anyway —
            # the user may have access to models not shown in the public
            # listing (gated previews, staged rollouts).  Warn but allow.

            # Auto-correct if the top match is very similar (e.g. typo)
            auto = get_close_matches(requested_for_lookup, api_models, n=1, cutoff=0.9)
            if auto:
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "corrected_model": auto[0],
                    "message": f"Auto-corrected `{requested}` → `{auto[0]}`",
                }

            suggestions = get_close_matches(requested, api_models, n=3, cutoff=0.5)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n  Similar models: " + ", ".join(f"`{s}`" for s in suggestions)

            # Model not in live /v1/models — check the curated catalog
            # before rejecting.  Providers may omit models from their live
            # listing that are still valid (stale cache, partial rollout,
            # gated previews).  Use the pure-catalog helper (no extra live
            # fetch) so we only accept models Pilotage actually ships.
            #
            # EXCEPTION: official OpenAI hosts (canonical api.openai.com and
            # the data-residency regional hosts).  Their /v1/models listing is
            # access-scoped and authoritative — a model absent from it is one
            # this key CANNOT serve, so the curated soft-accept would
            # manufacture a selection that 400s at first use.  Custom
            # OpenAI-compatible proxies keep the fallback (incomplete
            # listings are common there).
            _openai_listing_is_authoritative = False
            if normalized in ("openai", "openai-api"):
                from pilotage_cli.providers import is_official_openai_host

                _openai_listing_is_authoritative = is_official_openai_host(base_url)
            if not _openai_listing_is_authoritative and _model_in_provider_catalog(
                requested_for_lookup.lower(), _provider_keys(normalized)
            ):
                return {
                    "accepted": True,
                    "persist": True,
                    "recognized": True,
                    "message": (
                        f"Note: `{requested}` was not found in the live /v1/models listing "
                        f"but exists in the curated catalog — accepted."
                    ),
                }

        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": (
                f"Model `{requested}` was not found in this provider's model listing."
                f"{suggestion_text}"
            ),
        }

    # api_models is None — couldn't reach API.  Accept and persist,
    # but warn so typos don't silently break things.

    # Static-catalog fallback: when the /models probe was unreachable,
    # validate against the curated list from provider_model_ids() — same
    # pattern as the openai-codex branch above.  This keeps
    # /model switches working in the gateway for providers whose /models
    # endpoint is temporarily unreachable or returns a non-JSON payload.
    # Without this block, validate_requested_model would reject every model
    # on such providers, switch_model() would return success=False, and
    # the gateway would never write to _session_model_overrides.
    provider_label = _PROVIDER_LABELS.get(normalized, normalized)
    try:
        catalog_models = provider_model_ids(normalized)
    except Exception:
        catalog_models = []

    if catalog_models:
        catalog_lower = {m.lower(): m for m in catalog_models}
        if requested_for_lookup.lower() in catalog_lower:
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            }
        catalog_lower_list = list(catalog_lower.keys())
        auto = get_close_matches(
            requested_for_lookup.lower(), catalog_lower_list, n=1, cutoff=0.9
        )
        if auto:
            corrected = catalog_lower[auto[0]]
            return {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "corrected_model": corrected,
                "message": f"Auto-corrected `{requested}` → `{corrected}`",
            }
        suggestions = get_close_matches(
            requested_for_lookup.lower(), catalog_lower_list, n=3, cutoff=0.5
        )
        suggestion_text = ""
        if suggestions:
            suggestion_text = "\n  Similar models: " + ", ".join(
                f"`{catalog_lower[s]}`" for s in suggestions
            )
        return {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": (
                f"Note: `{requested}` was not found in the {provider_label} curated catalog "
                f"and the /models endpoint was unreachable.{suggestion_text}"
                f"\n  The model may still work if it exists on the provider."
            ),
        }

    # No catalog available — accept with a warning, matching the comment's
    # stated intent ("Accept and persist, but warn").
    return {
        "accepted": True,
        "persist": True,
        "recognized": False,
        "message": (
            f"Note: could not reach the {provider_label} API to validate `{requested}`. "
            f"If the service isn't down, this model may not be valid."
        ),
    }
