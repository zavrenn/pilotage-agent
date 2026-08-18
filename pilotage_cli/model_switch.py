"""Shared model-switching logic for CLI and gateway /model commands.

Both the CLI (cli.py) and gateway (gateway/run.py) /model handlers
share the same core pipeline:

  parse flags -> alias resolution -> provider resolution ->
  credential resolution -> normalize model name ->
  metadata lookup -> build result

This module ties together the foundation layers:

- ``agent.models_dev``            -- models.dev catalog, ModelInfo, ProviderInfo
- ``pilotage_cli.providers``        -- canonical provider identity + overlays
- ``pilotage_cli.model_normalize``  -- per-provider name formatting

Provider switching uses the ``--provider`` flag exclusively.
No colon-based ``provider:model`` syntax — colons are reserved for
vendor variant suffixes (``:free``, ``:extended``, ``:fast``).
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional

from pilotage_cli.providers import (
    ProviderDef,
    custom_provider_aliases,
    custom_provider_slug,
    determine_api_mode,
    get_label,
    host_mandated_api_mode,
    is_aggregator,
    resolve_provider_full,
)
from pilotage_cli.model_normalize import (
    normalize_model_for_provider,
)
from agent.models_dev import (
    ModelCapabilities,
    ModelInfo,
    get_model_capabilities,
    get_model_info,
    list_provider_models,
)
from utils import base_url_host_matches, base_url_hostname

logger = logging.getLogger(__name__)


def _declared_model_ids(value: Any) -> list[str]:
    """Return configured model IDs from supported config shapes.

    Accepts:
    - ``{"model-id": {...}}``
    - ``["model-a", "model-b"]``
    - ``[{"id": "model-a"}, {"name": "model-b"}]``
    - ``"model-a"``
    """
    ids: list[str] = []
    seen: set[str] = set()

    def _add(candidate: Any) -> None:
        if not isinstance(candidate, str):
            return
        model_id = candidate.strip()
        if not model_id:
            return
        lowered = model_id.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        ids.append(model_id)

    if isinstance(value, str):
        _add(value)
        return ids

    if isinstance(value, dict):
        for model_id in value:
            _add(model_id)
        return ids

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                _add(item)
                continue
            if isinstance(item, dict):
                model_id = item.get("id")
                if not isinstance(model_id, str) or not model_id.strip():
                    model_id = item.get("name")
                _add(model_id)
        return ids

    return ids


def _save_discovered_models_to_config(
    api_url: str, model_ids: list[str]
) -> None:
    """Persist discovered models into ``custom_providers`` in config.yaml.

    Called after a successful ``/v1/models`` probe so that the next read
    with ``discover_models: false`` uses the cached list instead of a stale
    or minimal manually-configured subset.

    Matches entries by ``base_url`` (trailing-slash-normalised).  A failed
    config write is swallowed — the picker still shows the live models for
    this session.
    """
    if not api_url or not model_ids:
        return
    try:
        from pilotage_cli.config import load_config, save_config

        cfg = load_config()
        providers = cfg.get("custom_providers") or []
        if not isinstance(providers, list):
            return

        norm_url = api_url.strip().rstrip("/").lower()
        changed = False
        for entry in providers:
            if not isinstance(entry, dict):
                continue
            entry_url = (entry.get("base_url", "") or entry.get("url", "") or "").strip()
            if entry_url.rstrip("/").lower() != norm_url:
                continue
            existing = entry.get("models")
            # Preserve per-model metadata: when ``models`` is a mapping
            # (e.g. ``{"model-a": {"context_length": 8192}}``) or a list of
            # dicts (e.g. ``[{"id": "model-a", "context_length": 8192}]``),
            # the user has curated metadata per model — do not replace it.
            if isinstance(existing, dict):
                continue
            if isinstance(existing, list) and any(
                isinstance(m, dict) for m in existing
            ):
                continue
            # Only update when models are stale — avoids unnecessary
            # config writes on every picker open.
            if isinstance(existing, list) and existing == model_ids:
                continue
            entry["models"] = model_ids
            changed = True

        if changed:
            cfg["custom_providers"] = providers
            save_config(cfg)
    except Exception:
        pass


def _bare_custom_provider_def(current_base_url: str) -> Optional[ProviderDef]:
    """ProviderDef for a direct ``model.provider: custom`` endpoint."""
    base_url = str(current_base_url or "").strip()
    if not base_url:
        return None
    return ProviderDef(
        id="custom",
        name="Custom endpoint",
        transport="openai_chat",
        api_key_env_vars=(),
        base_url=base_url,
        is_aggregator=False,
        auth_type="api_key",
        source="model-config",
    )


# ---------------------------------------------------------------------------
# Model aliases -- short names -> (vendor, family) with NO version numbers.
# Resolved dynamically against the live models.dev catalog.
# ---------------------------------------------------------------------------

class ModelIdentity(NamedTuple):
    """Vendor slug and family prefix used for catalog resolution."""
    vendor: str
    family: str


MODEL_ALIASES: dict[str, ModelIdentity] = {
    # OpenAI
    "gpt5":      ModelIdentity("openai", "gpt-5"),
    "gpt":       ModelIdentity("openai", "gpt"),
    "codex":     ModelIdentity("openai", "codex"),
    "o3":        ModelIdentity("openai", "o3"),
    "o4":        ModelIdentity("openai", "o4"),
}


# ---------------------------------------------------------------------------
# Direct aliases — exact model+provider+base_url for endpoints that aren't
# in the models.dev catalog (e.g. cloud relays, local servers).
# Checked BEFORE catalog resolution.  Format:
#   alias -> (model_id, provider, base_url)
# These can also be loaded from config.yaml ``model_aliases:`` section.
# ---------------------------------------------------------------------------

class DirectAlias(NamedTuple):
    """Exact model mapping that bypasses catalog resolution."""
    model: str
    provider: str
    base_url: str


# Built-in direct aliases (can be extended via config.yaml model_aliases:)
_BUILTIN_DIRECT_ALIASES: dict[str, DirectAlias] = {}

# Merged dict (builtins + user config); populated by _load_direct_aliases()
DIRECT_ALIASES: dict[str, DirectAlias] = {}


def _load_direct_aliases() -> dict[str, DirectAlias]:
    """Load direct aliases from config.yaml ``model_aliases:`` section.

    Config format::

        model_aliases:
          local:
            model: "my-model:397b"
            provider: custom
            base_url: "https://my-endpoint.example/v1"
          other:
            model: "other-model"
            provider: custom
            base_url: "https://my-endpoint.example/v1"

    Also reads ``model.aliases`` (set by ``pilotage config set model.aliases.xxx``)
    and converts simple string entries (``my-fast: vendor/fast-model-v4``)
    into DirectAlias objects.  The provider is parsed from the ``provider/``
    prefix in the value; if no slash, the current provider is used.
    """
    merged = dict(_BUILTIN_DIRECT_ALIASES)
    try:
        from pilotage_cli.config import load_config
        cfg = load_config()

        # --- model_aliases (dict-based format) ---
        user_aliases = cfg.get("model_aliases")
        if isinstance(user_aliases, dict):
            for name, entry in user_aliases.items():
                if not isinstance(entry, dict):
                    continue
                model = entry.get("model", "")
                provider = entry.get("provider", "custom")
                base_url = entry.get("base_url", "")
                if model:
                    merged[name.strip().lower()] = DirectAlias(
                        model=model, provider=provider, base_url=base_url,
                    )

        # --- model.aliases (string-based format, from config set) ---
        model_section = cfg.get("model", {})
        if isinstance(model_section, dict):
            simple_aliases = model_section.get("aliases")
            if isinstance(simple_aliases, dict):
                current_provider = model_section.get("provider", "")
                for name, value in simple_aliases.items():
                    if not isinstance(value, str) or not value.strip():
                        continue
                    key = name.strip().lower()
                    if key in merged:
                        continue  # don't override explicit model_aliases entries
                    val = value.strip()
                    if "/" in val:
                        provider, model = val.split("/", 1)
                    else:
                        provider = current_provider
                        model = val
                    merged[key] = DirectAlias(
                        model=model.strip(),
                        provider=provider.strip() or current_provider,
                        base_url="",
                    )
    except Exception:
        pass
    return merged


def _ensure_direct_aliases() -> None:
    """Lazy-load direct aliases on first use.

    Mutates the existing DIRECT_ALIASES dict in place rather than rebinding
    the module attribute. This keeps `from pilotage_cli.model_switch import
    DIRECT_ALIASES` references valid in callers — rebinding would leave them
    pointing at a stale empty dict.
    """
    if not DIRECT_ALIASES:
        DIRECT_ALIASES.update(_load_direct_aliases())


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelSwitchResult:
    """Result of a model switch attempt."""

    success: bool
    new_model: str = ""
    target_provider: str = ""
    provider_changed: bool = False
    api_key: str = ""
    base_url: str = ""
    api_mode: str = ""
    error_message: str = ""
    warning_message: str = ""
    provider_label: str = ""
    resolved_via_alias: str = ""
    capabilities: Optional[ModelCapabilities] = None
    model_info: Optional[ModelInfo] = None
    is_global: bool = False


@dataclass(frozen=True)
class ModelFlagParseResult:
    """Parsed flags for a /model command."""

    model_input: str
    explicit_provider: str = ""
    is_global: bool = False
    force_refresh: bool = False
    is_session: bool = False
    is_once: bool = False
# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------

def parse_model_flags_detailed(raw_args: str) -> ModelFlagParseResult:
    """Parse flags from /model command args.

    Returns a :class:`ModelFlagParseResult`. ``--once`` is intentionally
    parsed here but interpreted by each caller because each frontend has its
    own live-session restore hook.

    ``is_global`` and ``is_session`` are independent flag presences; the
    *effective* persistence decision is resolved by
    :func:`resolve_persist_behavior` so the config-gated default
    (``model.persist_switch_by_default``) is applied in one place.

    Examples::

        "gpt5"                           -> ("gpt5", "", False, False, False)
        "gpt5 --global"                  -> ("gpt5", "", True, False, False)
        "gpt5 --session"                 -> ("gpt5", "", False, False, True)
        "gpt5 --once"                    -> is_once=True
        "gpt5 --provider openai-api"     -> ("gpt5", "openai-api", False, False, False)
        "--provider my-endpoint"         -> ("", "my-endpoint", False, False, False)
        "--refresh"                      -> ("", "", False, True, False)
        "gpt5 --provider openai-api --global" -> ("gpt5", "openai-api", True, False, False)
    """
    is_global = False
    explicit_provider = ""
    force_refresh = False
    is_session = False
    is_once = False

    # Normalize Unicode dashes (Telegram/iOS auto-converts -- to em/en dash)
    # A single Unicode dash before a flag keyword becomes "--"
    import re as _re
    raw_args = _re.sub(r'[\u2012\u2013\u2014\u2015](provider|global|session|refresh|once)', r'--\1', raw_args)

    # Keep this hand-rolled because model IDs may contain colons/slashes and
    # the historical parser did not require shell quoting.
    parts = raw_args.split()
    i = 0
    filtered: list[str] = []
    while i < len(parts):
        if parts[i] == "--global":
            is_global = True
            i += 1
        elif parts[i] == "--session":
            is_session = True
            i += 1
        elif parts[i] == "--refresh":
            force_refresh = True
            i += 1
        elif parts[i] == "--once":
            is_once = True
            i += 1
        elif parts[i] == "--provider" and i + 1 < len(parts):
            explicit_provider = parts[i + 1]
            i += 2
        else:
            filtered.append(parts[i])
            i += 1

    model_input = " ".join(filtered).strip()
    return ModelFlagParseResult(
        model_input=model_input,
        explicit_provider=explicit_provider,
        is_global=is_global,
        force_refresh=force_refresh,
        is_session=is_session,
        is_once=is_once,
    )


def parse_model_flags(raw_args: str) -> tuple[str, str, bool, bool, bool]:
    """Parse legacy /model flags and return the historical 5-tuple.

    New call sites that care about ``--once`` should use
    :func:`parse_model_flags_detailed`.
    """
    parsed = parse_model_flags_detailed(raw_args)
    return (
        parsed.model_input,
        parsed.explicit_provider,
        parsed.is_global,
        parsed.force_refresh,
        parsed.is_session,
    )


def resolve_persist_behavior(
    is_global: bool,
    is_session: bool,
    is_once: bool = False,
    explicit_provider: str = "",
) -> bool:
    """Decide whether a ``/model`` switch should persist to ``config.yaml``.

    Resolution order:

    1. ``--once`` explicitly opts out → ``False`` (next turn only).
    2. ``--session`` explicitly opts out → ``False`` (this session only).
    3. ``--global`` explicitly opts in → ``True``.
    4. ``--provider`` given without an explicit persist flag → ``False``
       (session only).  Provider switches are typically exploratory — the
       user is trying a different backend for this conversation, not
       reconfiguring the default.  ``--global`` can still force persist.
    5. Otherwise defer to ``model.persist_switch_by_default`` in
       ``config.yaml`` (defaults to ``False``: a plain ``/model <name>``
       affects only the current session).  Users who want the old
       persist-by-default behavior can set the key to ``true``; a one-off
       ``--global`` always persists.

    The config read is defensive: on a fresh install ``model`` may be a
    flat string rather than a dict, in which case the built-in default
    (``False``) applies.
    """
    if is_once:
        return False
    if is_session:
        return False
    if is_global:
        return True
    if explicit_provider:
        return False
    try:
        from pilotage_cli.config import load_config

        model_cfg = load_config().get("model")
        if isinstance(model_cfg, dict):
            return bool(model_cfg.get("persist_switch_by_default", False))
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Single-owner /model request parsing + effective-model resolution
# ---------------------------------------------------------------------------
#
# Historically each surface (cli.py, gateway/slash_commands.py,
# tui_gateway/server.py) re-implemented flag parsing + conflict checks, and
# each resolution surface (gateway/run.py, gateway/platforms/api_server.py)
# re-implemented the session-override > channel/session > global precedence.
# Commit 7dd00bb47d had to re-fix the api_server discarding session-persisted
# models precisely because the precedence rule lived in two places.  The
# helpers below are the ONE owner; surfaces map error codes to their own
# user-facing copy but never re-derive the semantics.

# Error codes emitted by parse_model_switch_args().
MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL = "once_with_global"
MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET = "once_requires_target"

# Canonical (surface-neutral) error copy.  Surfaces prepend their own
# decoration ("  ✗ " in the CLI, "❌ " in the gateway) but MUST NOT change
# the core sentence — it is shared user-visible copy.
MODEL_SWITCH_ERROR_TEXT = {
    MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL: "/model --once cannot be combined with --global",
    MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET: "/model --once requires a model or provider.",
}


@dataclass(frozen=True)
class ModelSwitchRequest:
    """A fully parsed /model command request.

    ``scope`` is the *requested* persistence scope derived purely from the
    flags: ``"once"`` | ``"session"`` | ``"global"`` | ``"default"`` (no
    explicit scope flag; the effective decision then belongs to
    :func:`resolve_persist_behavior`, which also reads config).

    ``errors`` carries error *codes* (see ``MODEL_SWITCH_ERR_*``); surfaces
    render them via :data:`MODEL_SWITCH_ERROR_TEXT` plus their own prefix.
    """

    raw: str
    target: str
    explicit_provider: str = ""
    is_global: bool = False
    is_session: bool = False
    is_once: bool = False
    force_refresh: bool = False
    scope: str = "default"
    errors: tuple = ()

    # Compat properties so a ModelSwitchRequest can be passed anywhere a
    # ModelFlagParseResult was accepted (e.g. tui_gateway._apply_model_switch).
    @property
    def model_input(self) -> str:
        return self.target

    @property
    def flags(self) -> "ModelFlagParseResult":
        return ModelFlagParseResult(
            model_input=self.target,
            explicit_provider=self.explicit_provider,
            is_global=self.is_global,
            force_refresh=self.force_refresh,
            is_session=self.is_session,
            is_once=self.is_once,
        )

    def error_messages(self) -> list:
        """Canonical (undercorated) error strings for this request."""
        return [MODEL_SWITCH_ERROR_TEXT[code] for code in self.errors]


def parse_model_switch_args(raw: str) -> ModelSwitchRequest:
    """Parse a raw /model argument string into a :class:`ModelSwitchRequest`.

    The ONE parser for every /model surface.  Wraps
    :func:`parse_model_flags_detailed` (tokenization + Unicode-dash
    normalization) and layers on the flag-conflict validation that cli.py,
    gateway/slash_commands.py, and tui_gateway/server.py each used to
    re-implement:

    * ``--once`` + ``--global``  → ``MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL``
    * ``--once`` with no model and no ``--provider``
      → ``MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET``

    Model targets pass through untouched: bare names (``gpt-5.6``),
    aggregator slugs (``vendor/model``), and colon forms (``vendor:model``)
    are all resolved later by :func:`switch_model` (aggregator-aware — bare
    names resolve WITHIN the current aggregator first).
    """
    raw = str(raw or "")
    parsed = parse_model_flags_detailed(raw)

    errors: list = []
    if parsed.is_once and parsed.is_global:
        errors.append(MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL)
    if parsed.is_once and not parsed.model_input and not parsed.explicit_provider:
        errors.append(MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET)

    if parsed.is_once:
        scope = "once"
    elif parsed.is_session:
        scope = "session"
    elif parsed.is_global:
        scope = "global"
    else:
        scope = "default"

    return ModelSwitchRequest(
        raw=raw,
        target=parsed.model_input,
        explicit_provider=parsed.explicit_provider,
        is_global=parsed.is_global,
        is_session=parsed.is_session,
        is_once=parsed.is_once,
        force_refresh=parsed.force_refresh,
        scope=scope,
        errors=tuple(errors),
    )


def _effective_model_candidate(value: Any) -> str:
    """Extract a model-name candidate from a str / dict / attr-object."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("model") or "").strip()
    model_attr = getattr(value, "model", None)
    if model_attr is not None:
        return str(model_attr or "").strip()
    return ""


def resolve_effective_model(
    session_overrides: Any = None,
    channel_config: Any = None,
    global_config: Any = "",
) -> str:
    """Resolve the effective model: session override > channel > global.

    The single owner of the precedence rule that gateway/run.py
    (``_resolve_model_for_channel`` / ``_apply_session_model_override``) and
    gateway/platforms/api_server.py (``_create_agent``'s session-override /
    session-persisted-model branches) each encoded independently — the
    divergence commit 7dd00bb47d had to close.  A user-issued ``/model``
    (session override) always wins over per-channel/session-persisted
    configuration, which wins over the global default.

    Each argument may be a plain model string, a dict with a ``"model"``
    key (a gateway ``_session_model_overrides`` entry), or an object with a
    ``.model`` attribute (a ``ChannelOverride``).  Empty/None entries fall
    through to the next tier.
    """
    for tier in (session_overrides, channel_config, global_config):
        candidate = _effective_model_candidate(tier)
        if candidate:
            return candidate
    return ""


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

def _model_sort_key(model_id: str, prefix: str) -> tuple:
    """Sort key for model version preference.

    Extracts version numbers after the family prefix and returns a sort key
    that prefers higher versions.  Suffix tokens (``pro``, ``omni``, etc.)
    are used as tiebreakers, with common quality indicators ranked.

    Examples (with prefix ``"model"``)::

        model-v2.5-pro  → (-2.5, 0, 'pro')     # highest version wins
        model-v2.5      → (-2.5, 1, '')        # no suffix = lower than pro
        model-v2-pro    → (-2.0, 0, 'pro')
        model-v2-omni   → (-2.0, 1, 'omni')
        model-v2-flash  → (-2.0, 1, 'flash')
    """
    # Strip the prefix (and optional "/" separator for aggregator slugs)
    rest = model_id[len(prefix):]
    if rest.startswith("/"):
        rest = rest[1:]
    rest = rest.lstrip("-").strip()

    # Parse version and suffix from the remainder.
    # "v2.5-pro" → version [2.5], suffix "pro"
    # "-omni"    → version [],    suffix "omni"
    # State machine: start → in_version → between → in_suffix
    nums: list[float] = []
    suffix_buf = ""
    state = "start"
    num_buf = ""

    for ch in rest:
        if state == "start":
            if ch in "vV":
                state = "in_version"
            elif ch.isdigit():
                state = "in_version"
                num_buf += ch
            elif ch in "-_.":
                pass  # skip separators before any content
            else:
                state = "in_suffix"
                suffix_buf += ch
        elif state == "in_version":
            if ch.isdigit():
                num_buf += ch
            elif ch == ".":
                if "." in num_buf:
                    # Second dot — flush current number, start new component
                    try:
                        nums.append(float(num_buf.rstrip(".")))
                    except ValueError:
                        pass
                    num_buf = ""
                else:
                    num_buf += ch
            elif ch in "-_.":
                if num_buf:
                    try:
                        nums.append(float(num_buf.rstrip(".")))
                    except ValueError:
                        pass
                    num_buf = ""
                state = "between"
            else:
                if num_buf:
                    try:
                        nums.append(float(num_buf.rstrip(".")))
                    except ValueError:
                        pass
                    num_buf = ""
                state = "in_suffix"
                suffix_buf += ch
        elif state == "between":
            if ch.isdigit():
                state = "in_version"
                num_buf = ch
            elif ch in "vV":
                state = "in_version"
            elif ch in "-_.":
                pass
            else:
                state = "in_suffix"
                suffix_buf += ch
        elif state == "in_suffix":
            suffix_buf += ch

    # Flush remaining buffer (strip trailing dots — "5.4." → "5.4")
    if num_buf and state == "in_version":
        try:
            nums.append(float(num_buf.rstrip(".")))
        except ValueError:
            pass

    suffix = suffix_buf.lower().strip("-_.")
    suffix = suffix.strip()

    # Split out YYYYMMDD date stamps (e.g. gpt-5-codex-20250514): they are
    # snapshot markers, not version components, and would otherwise dwarf
    # real point versions (20250514 > 8).  Kept as a trailing tiebreaker so
    # bare IDs sort before their dated snapshots, and newer snapshots before
    # older ones.  The 19_000_101 threshold reclassifies only 8-digit stamps,
    # so shorter numeric components (gpt-4-0613) keep
    # their current behavior.
    version_nums: list[float] = []
    date_stamp = 0.0
    for n in nums:
        if n >= 19_000_101:
            date_stamp = max(date_stamp, n)
        else:
            version_nums.append(n)

    # Negate versions so higher → sorts first
    version_key = tuple(-n for n in version_nums)
    date_key = (0.0, 0.0) if date_stamp == 0.0 else (1.0, -date_stamp)

    # Suffix quality ranking: pro/max > (no suffix) > omni/flash/mini/lite
    # Lower number = preferred
    # "sol" is the flagship tier of the GPT-5.6 series (sol > terra > luna);
    # without it, alias resolution would tiebreak alphabetically and pick
    # luna (the cheapest) for `/model gpt`. Unlike pro/max/plus/turbo it is a
    # series codename, not a generic quality word — revisit if another vendor
    # ever ships a "-sol" suffix that isn't a flagship.
    _SUFFIX_RANK = {"pro": 0, "max": 0, "plus": 0, "turbo": 0, "sol": 0}
    suffix_rank = _SUFFIX_RANK.get(suffix, 1)

    return version_key + (suffix_rank, suffix) + date_key


class AmbiguousAliasError(Exception):
    """Alias family-matches multiple catalog models; caller must disambiguate.

    Raised by :func:`resolve_alias` instead of silently picking one candidate
    via version-sort heuristics. ``candidates`` is sorted best-guess-first
    (see :func:`_model_sort_key`) for display purposes only.
    """

    def __init__(self, alias: str, provider: str, candidates: list[str]):
        self.alias = alias
        self.provider = provider
        self.candidates = candidates
        super().__init__(
            f"alias {alias!r} matches {len(candidates)} models on {provider}"
        )


def _ambiguous_alias_message(err: "AmbiguousAliasError") -> str:
    """User-facing disambiguation list for an ambiguous alias."""
    shown = err.candidates[:10]
    lines = "\n".join(f"  {i}. {m}" for i, m in enumerate(shown, 1))
    more = ""
    if len(err.candidates) > len(shown):
        more = f"\n  … and {len(err.candidates) - len(shown)} more"
    return (
        f"'{err.alias}' matches {len(err.candidates)} models on "
        f"{err.provider} — not switching automatically:\n{lines}{more}\n"
        f"Pick one with /model <exact-model-name>."
    )


def resolve_alias(
    raw_input: str,
    current_provider: str,
) -> Optional[tuple[str, str, str]]:
    """Resolve a short alias against the current provider's catalog.

    Looks up *raw_input* in :data:`MODEL_ALIASES`, then searches the
    current provider's models.dev catalog for the model whose ID starts
    with ``vendor/family`` (or just ``family`` for non-aggregator
    providers) and has the **highest version**.

    Returns:
        ``(provider, resolved_model_id, alias_name)`` if a match is
        found on the current provider, or ``None`` if the alias doesn't
        exist or no matching model is available.
    """
    key = raw_input.strip().lower()

    # Check direct aliases first (exact model+provider+base_url mappings)
    _ensure_direct_aliases()
    direct = DIRECT_ALIASES.get(key)
    if direct is not None:
        return (direct.provider, direct.model, key)

    # Reverse lookup: match by model ID so full names (e.g. "my-model-4.7")
    # route through direct aliases instead of falling through
    # to catalog resolution.
    for alias_name, da in DIRECT_ALIASES.items():
        if da.model.lower() == key:
            return (da.provider, da.model, alias_name)

    identity = MODEL_ALIASES.get(key)
    if identity is None:
        return None

    vendor, family = identity

    # Build catalog from models.dev, then merge in static _PROVIDER_MODELS
    # entries that models.dev may be missing (e.g. newly added models not
    # yet synced to the registry).
    catalog = list_provider_models(current_provider)
    try:
        from pilotage_cli.models import _PROVIDER_MODELS
        static = _PROVIDER_MODELS.get(current_provider, [])
        if static:
            seen = {m.lower() for m in catalog}
            for m in static:
                if m.lower() not in seen:
                    catalog.append(m)
    except Exception:
        pass

    # For aggregators, models are vendor/model-name format
    aggregator = is_aggregator(current_provider)

    if aggregator:
        prefix = f"{vendor}/{family}".lower()
        matches = [
            mid for mid in catalog
            if mid.lower().startswith(prefix)
        ]
    else:
        family_lower = family.lower()
        matches = [
            mid for mid in catalog
            if mid.lower().startswith(family_lower)
        ]

    if not matches:
        return None

    # Sort by version descending (best guess first) for display, but NEVER
    # silently pick among multiple candidates: version-sort heuristics have
    # repeatedly guessed wrong (dated snapshots outranking point releases,
    # suffix tiebreaks landing on the cheapest tier). One match = resolve;
    # several = make the user choose.
    prefix_for_sort = f"{vendor}/{family}" if aggregator else family
    matches.sort(key=lambda m: _model_sort_key(m, prefix_for_sort))
    if len(matches) > 1:
        raise AmbiguousAliasError(key, current_provider, matches)
    return (current_provider, matches[0], key)


def get_authenticated_provider_slugs(
    current_provider: str = "",
    user_providers: dict = None,
    custom_providers: list | None = None,
) -> list[str]:
    """Return slugs of providers that have credentials.

    Uses ``list_authenticated_providers()`` which is backed by the models.dev
    in-memory cache (1 hr TTL) — no extra network cost.
    """
    try:
        providers = list_authenticated_providers(
            current_provider=current_provider,
            user_providers=user_providers,
            custom_providers=custom_providers,
            max_models=0,
        )
        return [p["slug"] for p in providers]
    except Exception:
        return []


def _resolve_alias_fallback(
    raw_input: str,
    authenticated_providers: list[str] = (),
) -> Optional[tuple[str, str, str]]:
    """Try to resolve an alias on the user's authenticated providers.

    Without authenticated providers there is nothing to fall back to.
    """
    providers = authenticated_providers or ()
    for provider in providers:
        # AmbiguousAliasError propagates: the alias exists on this provider,
        # the user just has to choose — trying the next provider instead
        # would silently switch them somewhere they didn't ask to go.
        result = resolve_alias(raw_input, provider)
        if result is not None:
            return result
    return None


def resolve_display_context_length(
    model: str,
    provider: str,
    base_url: str = "",
    api_key: str = "",
    model_info: Optional[ModelInfo] = None,
    custom_providers: list | None = None,
    config_context_length: int | None = None,
    configured_model: str | None = None,
    configured_provider: str | None = None,
    configured_base_url: str | None = None,
) -> Optional[int]:
    """Resolve the context length to show in /model output.

    models.dev reports per-vendor context (e.g. gpt-5.5 = 1.05M on openai)
    but provider-enforced limits can be lower (e.g. Codex OAuth caps the
    same slug at 272k). The authoritative source is
    ``agent.model_metadata.get_model_context_length`` which already knows
    about Codex OAuth and falls back to models.dev for the rest.

    When ``custom_providers`` is provided, per-model ``context_length``
    overrides from ``custom_providers[].models.<id>.context_length`` are
    honored — this closes where ``/model`` switch ignored user-set
    overrides.

    Prefer the provider-aware value; fall back to ``model_info.context_window``
    only if the resolver returns nothing.
    """
    if config_context_length is not None and (
        configured_model or configured_provider or configured_base_url
    ):
        try:
            from pilotage_cli.route_identity import should_clear_context_pin

            if should_clear_context_pin(
                configured_model,
                model,
                configured_base_url,
                base_url,
                configured_provider,
                provider,
            ):
                config_context_length = None
        except Exception:
            config_context_length = None

    try:
        from agent.model_metadata import get_model_context_length
        ctx = get_model_context_length(
            model,
            base_url=base_url or "",
            api_key=api_key or "",
            provider=provider or None,
            custom_providers=custom_providers,
            config_context_length=config_context_length,
        )
        if ctx:
            return int(ctx)
    except Exception:
        pass
    if model_info is not None and model_info.context_window:
        return int(model_info.context_window)
    return None


async def resolve_display_context_length_async(
    model: str,
    provider: str,
    base_url: str = "",
    api_key: str = "",
    model_info: Optional[ModelInfo] = None,
    custom_providers: list | None = None,
    config_context_length: int | None = None,
    configured_model: str | None = None,
    configured_provider: str | None = None,
    configured_base_url: str | None = None,
) -> Optional[int]:
    """Async variant of :func:`resolve_display_context_length`.

    The sync version runs two blocking chains: the route comparison in
    ``should_clear_context_pin`` and the full provider probe ladder in
    ``get_model_context_length`` (blocking ``requests`` calls to Codex,
    models.dev and configured endpoints).  Async gateway handlers must not run either on the event
    loop — see ``agent.model_metadata.get_model_context_length_async`` and
    ``pilotage_cli.route_identity.should_clear_context_pin_async``, which
    offload the same chains for the message path.

    Shares all logic with the sync version — no code duplication.
    """
    import asyncio

    return await asyncio.to_thread(
        resolve_display_context_length,
        model,
        provider,
        base_url=base_url,
        api_key=api_key,
        model_info=model_info,
        custom_providers=custom_providers,
        config_context_length=config_context_length,
        configured_model=configured_model,
        configured_provider=configured_provider,
        configured_base_url=configured_base_url,
    )


# ---------------------------------------------------------------------------
# Configured-provider detection for typed model names
# ---------------------------------------------------------------------------


def _configured_provider_matches(
    model_name: str,
    user_providers: Optional[dict],
    custom_providers: Optional[list],
) -> dict[str, str]:
    """Return ``{provider_slug: canonical_model_id}`` for every configured
    provider whose declared models contain an exact (case-insensitive) match
    for ``model_name``.

    Used by :func:`switch_model` to route a *typed* model name to the provider
    that actually declares it in user/custom provider config, instead of
    leaving it on the current provider.  Without this, a model declared under
    ``providers.<slug>`` / ``custom_providers`` but typed while the current
    provider is ``openai-codex`` stays on Codex and is soft-accepted as an
    unknown hidden Codex model.

    Matching is exact (case-insensitive); the configured spelling is returned
    so the downstream validation/override path sees the canonical id.  Only the
    explicitly-declared model collections are scanned (``models``, the singular
    ``model``, and ``default_model``) — never fuzzy/family matching.
    """
    if not model_name or not model_name.strip():
        return {}
    target = model_name.strip().lower()

    def _match(value) -> Optional[str]:
        """Canonical id if ``value`` (a model collection or scalar) declares
        ``target``, else None."""
        for model_id in _declared_model_ids(value):
            if model_id.lower() == target:
                return model_id
        return None

    matches: dict[str, str] = {}

    if isinstance(user_providers, dict):
        for slug, cfg in user_providers.items():
            if not isinstance(slug, str) or not isinstance(cfg, dict):
                continue
            for key in ("models", "model", "default_model"):
                hit = _match(cfg.get(key))
                if hit:
                    matches[slug] = hit
                    break

    if isinstance(custom_providers, list):
        for entry in custom_providers:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            slug = f"custom:{name}"
            if slug in matches:
                continue
            for key in ("models", "model", "default_model"):
                hit = _match(entry.get(key))
                if hit:
                    matches[slug] = hit
                    break

    return matches


def _resolve_named_custom_model_id(
    model_name: str,
    target_provider: str,
    custom_providers: Optional[list],
) -> str:
    """Map a picker-prefixed custom model selection to its configured ID."""
    provider = str(target_provider or "").strip().lower()
    if not provider.startswith("custom:") or "/" not in model_name:
        return model_name

    prefix, candidate = model_name.split("/", 1)
    prefix = prefix.strip().lower()
    candidate = candidate.strip()
    if not prefix or not candidate:
        return model_name

    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        entry_slugs = custom_provider_aliases(
            str(entry.get("name") or ""),
            str(entry.get("provider_key") or ""),
        )
        if provider not in entry_slugs or f"custom:{prefix}" not in entry_slugs:
            continue
        for model_id in _declared_model_ids(entry.get("models")):
            if model_id.lower() == candidate.lower():
                return model_id
    return model_name


# ---------------------------------------------------------------------------
# Core model-switching pipeline
# ---------------------------------------------------------------------------

def switch_model(
    raw_input: str,
    current_provider: str,
    current_model: str,
    current_base_url: str = "",
    current_api_key: str = "",
    is_global: bool = False,
    explicit_provider: str = "",
    user_providers: dict = None,
    custom_providers: list | None = None,
) -> ModelSwitchResult:
    """Core model-switching pipeline shared between CLI and gateway.

    Resolution chain:

      If --provider given:
        a. Resolve provider via resolve_provider_full()
        b. Resolve credentials
        c. If model given, resolve alias on target provider or use as-is
        d. If no model, auto-detect from endpoint

      If no --provider:
        a. Try alias resolution on current provider
        b. If alias exists but not on current provider -> fallback
        c. On aggregator, try vendor/model slug conversion
        d. Aggregator catalog search
        e. detect_provider_for_model() as last resort
        f. Resolve credentials
        g. Normalize model name for target provider

      Finally:
        h. Get full model metadata from models.dev
        i. Build result

    Args:
        raw_input: The model name (after flag parsing).
        current_provider: The currently active provider.
        current_model: The currently active model name.
        current_base_url: The currently active base URL.
        current_api_key: The currently active API key.
        is_global: Whether to persist the switch.
        explicit_provider: From --provider flag (empty = no explicit provider).
        user_providers: The ``providers:`` dict from config.yaml (for user endpoints).
        custom_providers: The ``custom_providers:`` list from config.yaml.

    Returns:
        ModelSwitchResult with all information the caller needs.
    """
    from pilotage_cli.models import (
        detect_provider_for_model,
        validate_requested_model,
    )
    from pilotage_cli.runtime_provider import resolve_runtime_provider

    resolved_alias = ""
    new_model = raw_input.strip()
    target_provider = current_provider

    # =================================================================
    # PATH A: Explicit --provider given
    # =================================================================
    if explicit_provider:
        # Resolve the provider
        pdef = resolve_provider_full(
            explicit_provider,
            user_providers,
            custom_providers,
        )
        if pdef is None and explicit_provider.strip().lower() == "custom":
            pdef = _bare_custom_provider_def(current_base_url)
        if pdef is None:
            _switch_err = (
                f"Unknown provider '{explicit_provider}'. "
                f"Check 'pilotage model' for available providers, or define it "
                f"in config.yaml under 'providers:'."
            )
            # Check for common config issues that cause provider resolution failures
            try:
                from pilotage_cli.config import validate_config_structure
                _cfg_issues = validate_config_structure()
                if _cfg_issues:
                    _switch_err += "\n\nRun 'pilotage doctor' — config issues detected:"
                    for _ci in _cfg_issues[:3]:
                        _switch_err += f"\n  • {_ci.message}"
            except Exception:
                pass
            return ModelSwitchResult(
                success=False,
                is_global=is_global,
                error_message=_switch_err,
            )

        target_provider = pdef.id

        # If no model specified, try auto-detect from endpoint
        if not new_model:
            if pdef.base_url:
                from pilotage_cli.runtime_provider import _auto_detect_local_model
                detected = _auto_detect_local_model(pdef.base_url)
                if detected:
                    new_model = detected
                else:
                    return ModelSwitchResult(
                        success=False,
                        target_provider=target_provider,
                        provider_label=pdef.name,
                        is_global=is_global,
                        error_message=(
                            f"No model detected on {pdef.name} ({pdef.base_url}). "
                            f"Specify the model explicitly: /model <model-name> --provider {explicit_provider}"
                        ),
                    )
            else:
                return ModelSwitchResult(
                    success=False,
                    target_provider=target_provider,
                    provider_label=pdef.name,
                    is_global=is_global,
                    error_message=(
                        f"Provider '{pdef.name}' has no base URL configured. "
                        f"Specify a model: /model <model-name> --provider {explicit_provider}"
                    ),
                )

        # Resolve alias on the TARGET provider
        try:
            alias_result = resolve_alias(new_model, target_provider)
        except AmbiguousAliasError as err:
            return ModelSwitchResult(
                success=False,
                target_provider=target_provider,
                is_global=is_global,
                error_message=_ambiguous_alias_message(err),
            )
        if alias_result is not None:
            _, new_model, resolved_alias = alias_result

    # =================================================================
    # PATH B: No explicit provider — resolve from model input
    # =================================================================
    else:
        try:
            alias_result = resolve_alias(raw_input, current_provider)
        except AmbiguousAliasError as err:
            return ModelSwitchResult(
                success=False,
                is_global=is_global,
                error_message=_ambiguous_alias_message(err),
            )

        # --- Step a: Try alias resolution on current provider ---

        if alias_result is not None:
            target_provider, new_model, resolved_alias = alias_result
            logger.debug(
                "Alias '%s' resolved to %s on %s",
                resolved_alias, new_model, target_provider,
            )
        else:
            # --- Step b: Alias exists but not on current provider -> fallback ---
            key = raw_input.strip().lower()
            if key in MODEL_ALIASES:
                authed = get_authenticated_provider_slugs(
                    current_provider=current_provider,
                    user_providers=user_providers,
                    custom_providers=custom_providers,
                )
                try:
                    fallback_result = _resolve_alias_fallback(raw_input, authed)
                except AmbiguousAliasError as err:
                    return ModelSwitchResult(
                        success=False,
                        is_global=is_global,
                        error_message=_ambiguous_alias_message(err),
                    )
                if fallback_result is not None:
                    target_provider, new_model, resolved_alias = fallback_result
                    logger.debug(
                        "Alias '%s' resolved via fallback to %s on %s",
                        resolved_alias, new_model, target_provider,
                    )
                else:
                    identity = MODEL_ALIASES[key]
                    return ModelSwitchResult(
                        success=False,
                        is_global=is_global,
                        error_message=(
                            f"Alias '{key}' maps to {identity.vendor}/{identity.family} "
                            f"but no matching model was found in any provider catalog. "
                            f"Try specifying the full model name."
                        ),
                    )
            else:
                # --- Step c: On aggregator, convert vendor:model to vendor/model ---
                # Only convert when there's no slash — a slash means the name
                # is already in vendor/model format and the colon is a variant
                # tag (:free, :extended, :fast) that must be preserved.
                colon_pos = raw_input.find(":")
                if colon_pos > 0 and "/" not in raw_input and is_aggregator(current_provider):
                    left = raw_input[:colon_pos].strip().lower()
                    right = raw_input[colon_pos + 1:].strip()
                    if left and right:
                        # Colons become slashes for aggregator slugs
                        new_model = f"{left}/{right}"
                        logger.debug(
                            "Converted vendor:model '%s' to aggregator slug '%s'",
                            raw_input, new_model,
                        )

        # --- Step d: Aggregator catalog search ---
        # Track whether the live catalog of the CURRENT provider resolved the
        # model — if so, step e must not second-guess and switch providers.
        # Critical for flat-namespace resellers whose live /v1/models returns
        # bare IDs that coincidentally match entries in native providers'
        # static catalogs.
        resolved_in_current_catalog = False
        if is_aggregator(target_provider) and not resolved_alias:
            catalog = list_provider_models(target_provider)
            if catalog:
                new_model_lower = new_model.lower()
                for mid in catalog:
                    if mid.lower() == new_model_lower:
                        new_model = mid
                        resolved_in_current_catalog = True
                        break
                else:
                    for mid in catalog:
                        if "/" in mid:
                            _, bare = mid.split("/", 1)
                            if bare.lower() == new_model_lower:
                                new_model = mid
                                resolved_in_current_catalog = True
                                break

        # --- Step d.5: configured-provider exact-match detection ---
        # If the typed model is declared in user/custom provider config, route
        # to that provider BEFORE detect_provider_for_model() guesses from
        # static catalogs and BEFORE the common-path validation can let a
        # soft-accepting current provider (e.g. openai-codex) swallow the name
        # as an unknown hidden model.  Configured matches beat static-catalog
        # detection.  Unlike step e this is deliberately NOT gated on
        # ``not is_custom`` — switching from a local/custom provider A to a
        # configured provider B that declares the typed model is the point.
        config_routed = False
        if (
            not resolved_alias
            and not resolved_in_current_catalog
            and target_provider == current_provider
        ):
            cfg_matches = _configured_provider_matches(
                new_model, user_providers, custom_providers
            )
            if cfg_matches:
                if current_provider in cfg_matches:
                    # The current provider itself declares it — keep current.
                    new_model = cfg_matches[current_provider]
                    config_routed = True
                else:
                    match_slugs = sorted(cfg_matches)
                    if len(match_slugs) > 1:
                        return ModelSwitchResult(
                            success=False,
                            is_global=is_global,
                            error_message=(
                                f"'{new_model}' is declared by multiple configured "
                                f"providers ({', '.join(match_slugs)}). Re-run with "
                                f"--provider <slug> to choose which one to use."
                            ),
                        )
                    target_provider = match_slugs[0]
                    new_model = cfg_matches[target_provider]
                    config_routed = True
                    logger.debug(
                        "Configured-provider detection routed '%s' to %s",
                        new_model, target_provider,
                    )
                    # User-config providers (providers.<slug>) are resolved in
                    # the credential block via resolve_user_provider(), which is
                    # gated on explicit_provider.  Mirror the picker so the
                    # rerouted user provider's base_url/key load from the passed
                    # config rather than a from-scratch runtime re-resolve that
                    # doesn't know user-config slugs.  custom:* slugs resolve via
                    # resolve_runtime_provider() directly and need no hint.
                    if isinstance(user_providers, dict) and target_provider in user_providers:
                        explicit_provider = target_provider

        # --- Step e: detect_provider_for_model() as last resort ---
        _base = current_base_url or ""
        is_custom = (
            current_provider in {"custom", "local"}
            or current_provider.startswith("custom:")
            or base_url_hostname(_base) in ("localhost", "127.0.0.1")
        )

        if (
            target_provider == current_provider
            and not is_custom
            and not resolved_alias
            and not resolved_in_current_catalog
            and not config_routed
        ):
            detected = detect_provider_for_model(new_model, current_provider)
            if detected:
                target_provider, new_model = detected

    # =================================================================
    # COMMON PATH: Resolve credentials, normalize, get metadata
    # =================================================================

    provider_changed = target_provider != current_provider
    provider_label = get_label(target_provider)
    if target_provider == "custom" and current_base_url:
        provider_label = "Custom endpoint"
    if target_provider.startswith("custom:"):
        custom_pdef = resolve_provider_full(
            target_provider,
            user_providers,
            custom_providers,
        )
        if custom_pdef is not None:
            provider_label = custom_pdef.name

    # --- Resolve credentials ---
    api_key = current_api_key
    base_url = current_base_url
    api_mode = ""

    if provider_changed or explicit_provider:
        import os
        # User-config providers (providers.<name> in config.yaml) carry their
        # own base_url + transport + key reference. resolve_runtime_provider()
        # resolves by provider NAME and doesn't know user-config slugs (e.g. a
        # block named "openai"), so it would re-resolve from scratch and fail
        # or hop to an aggregator. Use the pdef's endpoint directly instead.
        _user_pdef = None
        if explicit_provider and user_providers:
            from pilotage_cli.providers import resolve_user_provider as _ruser
            _user_pdef = _ruser(explicit_provider.strip().lower(), user_providers)
            if _user_pdef is None:
                _user_pdef = _ruser(target_provider, user_providers)
        if _user_pdef is not None and _user_pdef.base_url:
            _ucfg = (user_providers or {}).get(explicit_provider.strip().lower()) \
                or (user_providers or {}).get(target_provider) or {}
            _ukey = str(_ucfg.get("api_key", "") or "").strip()
            if _ukey.startswith("${") and _ukey.endswith("}"):
                # Same class as the picker reads below: a raw os.environ read
                # here hands this profile whatever key the process env holds —
                # another profile's, under the multiplexed gateway. Route
                # through the per-profile secret scope (identical to
                # os.getenv when multiplexing is off, fail-closed otherwise).
                _ukey = _scoped_key_env(_ukey[2:-1])
            if not _ukey:
                _kenv = str(_ucfg.get("key_env", "") or "").strip()
                if _kenv:
                    _ukey = _scoped_key_env(_kenv)
            try:
                runtime = resolve_runtime_provider(
                    requested=target_provider,
                    explicit_api_key=_ukey or None,
                    explicit_base_url=_user_pdef.base_url,
                    target_model=new_model,
                )
                api_key = runtime.get("api_key", "") or _ukey
                base_url = runtime.get("base_url", "") or _user_pdef.base_url
                api_mode = runtime.get("api_mode", "")
            except Exception:
                api_key = _ukey
                base_url = _user_pdef.base_url
                api_mode = ""
        elif target_provider == "custom" and current_base_url:
            api_key = current_api_key
            base_url = current_base_url
            api_mode = determine_api_mode(target_provider, base_url)
        else:
            try:
                runtime = resolve_runtime_provider(
                    requested=target_provider,
                    target_model=new_model,
                )
                api_key = runtime.get("api_key", "")
                base_url = runtime.get("base_url", "")
                api_mode = runtime.get("api_mode", "")
            except Exception as e:
                return ModelSwitchResult(
                    success=False,
                    target_provider=target_provider,
                    provider_label=provider_label,
                    is_global=is_global,
                    error_message=(
                        f"Could not resolve credentials for provider "
                        f"'{provider_label}': {e}"
                    ),
                )
    else:
        try:
            runtime = resolve_runtime_provider(
                requested=current_provider,
                target_model=new_model,
            )
            # If resolution fell through to "custom" (e.g. a named custom
            # provider that resolve_runtime_provider doesn't know), keep existing
            # credentials. Otherwise use the resolved values (picks up credential
            # rotation, base_url adjustments, etc.).
            api_key = runtime.get("api_key", "")
            base_url = runtime.get("base_url", "")
            api_mode = runtime.get("api_mode", "")
        except Exception:
            pass

    # --- Direct alias override: use exact base_url from the alias if set ---
    if resolved_alias:
        _ensure_direct_aliases()
        _da = DIRECT_ALIASES.get(resolved_alias)
        if _da is not None and _da.base_url:
            base_url = _da.base_url
            api_mode = ""  # clear so determine_api_mode re-detects from URL
            if not api_key:
                api_key = "no-key-required"

    # --- Resolve api_mode from the final (provider, base_url) before validation ---
    # Two cases this closes, both surfaced when the switched model's reasoning
    # is actually applied (post the reasoning-unification refactor):
    #   1. api_mode empty (e.g. alias cleared it above) → fill from the endpoint.
    #   2. api_mode carried a STALE value from the previous session state
    #      (e.g. a same-provider /model switch that kept the prior
    #      chat_completions mode). A host that mandates
    #      one wire protocol must override the stale value — otherwise the request
    #      goes out on chat_completions and OpenAI 400s on tools+reasoning_effort.
    _mandated_mode = host_mandated_api_mode(base_url)
    if _mandated_mode is not None:
        api_mode = _mandated_mode
    elif not api_mode:
        api_mode = determine_api_mode(target_provider, base_url)

    # --- Normalize model name for target provider ---
    new_model = _resolve_named_custom_model_id(
        new_model, target_provider, custom_providers
    )
    new_model = normalize_model_for_provider(new_model, target_provider)

    # --- Validate ---
    try:
        validation = validate_requested_model(
            new_model,
            target_provider,
            api_key=api_key,
            base_url=base_url,
            api_mode=api_mode or None,
        )
    except Exception as e:
        validation = {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": f"Could not validate `{new_model}`: {e}",
        }

    # Override rejection if model is in the user's saved provider config.
    # API /v1/models may not list cloud/aliased models even though the server supports them.
    if not validation.get("accepted"):
        override = False
        if user_providers:
            from pilotage_cli.config import is_provider_enabled
            # user_providers is a dict: {provider_slug: config_dict}
            for slug, cfg in user_providers.items():
                if not is_provider_enabled(cfg):
                    continue
                if slug == target_provider:
                    if new_model in _declared_model_ids(cfg.get("models", {})):
                        override = True
                        break
        # Also check custom_providers list — models declared there should be accepted
        # even if the remote /v1/models endpoint doesn't list them.
        if not override and custom_providers and isinstance(custom_providers, list):
            for entry in custom_providers:
                if not isinstance(entry, dict):
                    continue
                # Match by provider slug (custom:<name>) or by base_url
                entry_name = entry.get("name", "")
                entry_aliases = custom_provider_aliases(
                    str(entry_name or ""),
                    str(entry.get("provider_key") or ""),
                )
                entry_url = entry.get("base_url", "")
                if target_provider.lower() in entry_aliases or entry_url == base_url:
                    # Check if the requested model matches the entry's model
                    entry_model = entry.get("model", "")
                    entry_models = entry.get("models", {})
                    if new_model == entry_model:
                        override = True
                        break
                    if new_model in _declared_model_ids(entry_models):
                        override = True
                        break
        if override:
            validation = {"accepted": True, "persist": True, "recognized": False, "message": validation.get("message", "")}
        else:
            msg = validation.get("message", "Invalid model")
            return ModelSwitchResult(
                success=False,
                new_model=new_model,
                target_provider=target_provider,
                provider_label=provider_label,
                is_global=is_global,
                error_message=msg,
            )

    # Apply auto-correction if validation found a closer match
    if validation.get("corrected_model"):
        new_model = validation["corrected_model"]

    # --- Determine api_mode if not already set ---
    if not api_mode:
        api_mode = determine_api_mode(
            target_provider, base_url, model=new_model
        )

    # --- Get capabilities (legacy) ---
    capabilities = get_model_capabilities(target_provider, new_model, allow_network=True)

    # --- Get full model info from models.dev ---
    model_info = get_model_info(target_provider, new_model, allow_network=True)

    # --- Collect warnings ---
    warnings: list[str] = []
    if validation.get("message"):
        warnings.append(validation["message"])

    # --- Build result ---
    return ModelSwitchResult(
        success=True,
        new_model=new_model,
        target_provider=target_provider,
        provider_changed=provider_changed,
        api_key=api_key,
        base_url=base_url,
        api_mode=api_mode,
        warning_message=" | ".join(warnings) if warnings else "",
        provider_label=provider_label,
        resolved_via_alias=resolved_alias,
        capabilities=capabilities,
        model_info=model_info,
        is_global=is_global,
    )


# ---------------------------------------------------------------------------
# Authenticated providers listing (for /model no-args display)
# ---------------------------------------------------------------------------

# Process-level guard so the picker prewarm thread is spawned at most once per
# process. Without a guard a long-lived process (or repeated triggers) would
# leak one OS thread per call.
import threading as _threading  # noqa: E402

_picker_prewarm_done = _threading.Event()


def _credential_pool_is_usable(provider: str, *, raw_pool_present: bool = False) -> bool:
    """Return whether *provider* has a credential that can be selected now.

    ``auth.json`` historically allowed opaque token-style pool values that do
    not deserialize into ``PooledCredential`` entries. Preserve visibility for
    those legacy values, but when a real pool exists its availability state is
    authoritative: an all-exhausted/dead pool is not authenticated.
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool(provider)
        if pool.has_credentials():
            return pool.has_available()
    except Exception:
        pass
    return raw_pool_present


def _scoped_key_env(name: str) -> str:
    """Read a provider key env var through the per-profile secret scope.

    The multiplexed gateway installs a secret scope per turn; a raw
    ``os.environ`` read hands the current profile whatever key happens to be
    in the process environment — another profile's, in a multiplexer. That is
    the class swept in 854007d1c for the fallback/aux key reads; the picker's
    ``key_env`` reads were not covered.

    Identical to ``os.getenv`` when multiplexing is off. A fail-closed
    ``UnscopedSecretError`` (multiplexing on, no scope installed) means "no
    credential visible for this profile here", which is exactly how the picker
    already treats a missing key.
    """
    if not name:
        return ""
    try:
        from agent.secret_scope import get_secret

        return (get_secret(name, "") or "").strip()
    except Exception:
        return ""


def list_authenticated_providers(
    current_provider: str = "",
    current_base_url: str = "",
    user_providers: dict = None,
    *,
    max_models: int | None = None,
    current_model: str = "",
    refresh: bool = False,
    for_picker: bool = False,
    excluded_providers: list | None = None,
) -> List[dict]:
    """Detect which providers have credentials and list their curated models.

    Uses the curated model lists from pilotage_cli/models.py
    (_PROVIDER_MODELS) — NOT the full models.dev catalog.  These are hand-picked
    agentic models that work well as agent backends.

    Returns a list of dicts, each with:
      - slug: str — the --provider value to use
      - name: str — display name
      - is_current: bool
      - is_user_defined: bool
      - models: list[str] — curated model IDs (up to max_models)
      - total_models: int — total curated count
      - source: str — "built-in", "models.dev", "user-config"

    Only includes providers that have API keys set or are user-defined endpoints.

    ``refresh`` busts the per-provider model-id disk cache
    (``provider_models_cache.json``) up front so every row re-fetches its
    live catalog. Use for an explicit user-triggered "refresh models" action
    (e.g. the desktop picker's refresh control); leave false for normal picker
    opens so they stay snappy on the 1h cache.

    ``for_picker`` keeps a provider visible when its credential pool exists
    but every key is rate-limited -- limits are per-model, so another model
    under the same provider may still answer.
    """
    import os
    from agent.models_dev import (
        PROVIDER_TO_MODELS_DEV,
        fetch_models_dev,
        get_provider_info as _mdev_pinfo,
    )
    from pilotage_cli.auth import PROVIDER_REGISTRY
    from pilotage_cli.models import (
        _PROVIDER_MODELS,
        cached_provider_model_ids,
        clear_provider_models_cache,
    )

    # Explicit refresh: drop every provider's cached model-id list so the
    # cached_provider_model_ids() calls below all re-fetch live. Without this
    # a stale 1h cache can fall back to the curated static list when its live
    # fetch later fails, silently dropping live-only models the user had
    # seen before.
    if refresh:
        try:
            clear_provider_models_cache()
        except Exception:
            pass

    results: List[dict] = []
    seen_slugs: set = set()  # lowercase-normalized to catch case variants
    _current_provider_norm = str(current_provider or "").strip().lower()
    _current_base_url_norm = str(current_base_url or "").strip().rstrip("/").lower()

    # Normalize the excluded-providers list once for fast membership checks.
    # Compared against both the models.dev id and the Pilotage slug so a single
    # entry hides the provider regardless of which key it surfaces under.
    _excluded: set = {str(p).strip().lower() for p in (excluded_providers or []) if p}
    data = fetch_models_dev()

    # Build curated model lists keyed by pilotage provider ID
    curated: dict[str, list[str]] = dict(_PROVIDER_MODELS)

    # --- Emit one row per provider that has credentials -----------------------
    # PROVIDER_TO_MODELS_DEV holds only openai/openai-codex, both mapping to the
    # models.dev ``openai`` catalog, so PILOTAGE_OVERLAYS is the single source of
    # picker rows: openai-codex (OAuth) and openai-api (key).
    from pilotage_cli.providers import PILOTAGE_OVERLAYS
    from pilotage_cli.auth import PROVIDER_REGISTRY as _auth_registry

    # Build reverse mapping: models.dev ID → Pilotage provider ID.
    # PILOTAGE_OVERLAYS keys may be models.dev IDs while _PROVIDER_MODELS and
    # config.yaml use Pilotage IDs.
    _mdev_to_pilotage = {v: k for k, v in PROVIDER_TO_MODELS_DEV.items()}

    for pid, overlay in PILOTAGE_OVERLAYS.items():
        if pid.lower() in seen_slugs:
            continue

        # Resolve Pilotage slug through the reverse mapping.
        pilotage_slug = _mdev_to_pilotage.get(pid, pid)
        if pilotage_slug.lower() in seen_slugs:
            continue
        if pid.lower() in _excluded or pilotage_slug.lower() in _excluded:
            continue

        # Check if credentials exist
        has_creds = False
        if overlay.extra_env_vars:
            has_creds = any(os.environ.get(ev) for ev in overlay.extra_env_vars)
        # Also check api_key_env_vars from PROVIDER_REGISTRY for api_key auth_type
        if not has_creds and overlay.auth_type == "api_key":
            for _key in (pid, pilotage_slug):
                pcfg = _auth_registry.get(_key)
                if pcfg and pcfg.api_key_env_vars:
                    if any(os.environ.get(ev) for ev in pcfg.api_key_env_vars):
                        has_creds = True
                        break
        # Check auth store and credential pool for non-env-var credentials.
        # This applies to OAuth providers AND api_key providers that also
        # support OAuth via external credential files.
        if not has_creds:
            try:
                from pilotage_cli.auth import _load_auth_store
                store = _load_auth_store()
                providers_store = store.get("providers", {})
                if store and (pid in providers_store or pilotage_slug in providers_store):
                    has_creds = True
            except Exception as exc:
                logger.debug("Auth store check failed for %s: %s", pid, exc)
        # Fallback: check the credential pool with full auto-seeding.
        # This catches credentials that exist in external stores (e.g.
        # Codex CLI ~/.codex/auth.json) which _seed_from_singletons()
        # imports on demand but aren't in the raw auth.json yet.
        if not has_creds:
            try:
                if _credential_pool_is_usable(pilotage_slug):
                    has_creds = True
                elif for_picker:
                    # For the interactive /model picker, also show providers
                    # whose credential pool has entries but all are temporarily
                    # rate-limited.  Rate limits are per-model for many
                    # providers — switching to a different
                    # model under the same provider may work even when all keys
                    # are in cooldown.
                    try:
                        from agent.credential_pool import load_pool
                        _pool = load_pool(pilotage_slug)
                        if _pool.has_credentials():
                            has_creds = True
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Credential pool check failed for %s: %s", pilotage_slug, exc)
        if not has_creds:
            continue

        if pilotage_slug in {"openai-codex"}:
            # Use live OAuth-backed discovery so the gateway /model picker
            # matches what the user's authenticated Codex backend
            # actually serves — including ChatGPT-Pro-only Codex slugs
            # (e.g. gpt-5.3-codex-spark) that aren't in the static curated
            # catalog. ``cached_provider_model_ids()`` falls back to the
            # curated list when the live endpoint is unreachable, so this
            # is safe for unauthenticated and offline cases too.
            model_ids = cached_provider_model_ids(pilotage_slug)
        else:
            # Unified pathway — see Section 1 rationale. Fall back to the
            # curated dict when the live fetcher comes up empty.
            model_ids = cached_provider_model_ids(pilotage_slug)
            if not model_ids:
                model_ids = curated.get(pilotage_slug, []) or curated.get(pid, [])
        total = len(model_ids)
        top = model_ids[:max_models] if max_models is not None else model_ids

        results.append({
            "slug": pilotage_slug,
            "name": get_label(pilotage_slug),
            "is_current": pilotage_slug == current_provider or pid == current_provider,
            "is_user_defined": False,
            "models": top,
            "total_models": total,
            "source": "pilotage",
        })
        seen_slugs.add(pid.lower())
        seen_slugs.add(pilotage_slug.lower())

    # Apply the ``providers.<name>.enabled: false`` post-filter. Indexed by
    # lowercase slug AND by
    # ``provider_id`` so PROVIDER_REGISTRY entries that match user-config
    # blocks are filtered consistently.
    try:
        from pilotage_cli.config import is_provider_enabled
        if isinstance(user_providers, dict):
            _disabled_slugs = {
                str(name).strip().lower()
                for name, cfg in user_providers.items()
                if isinstance(cfg, dict) and not is_provider_enabled(cfg)
            }
            if _disabled_slugs:
                results = [
                    r for r in results
                    if str(r.get("provider_id", "")).strip().lower() not in _disabled_slugs
                    and str(r.get("slug", "")).strip().lower() not in _disabled_slugs
                ]
    except Exception:
        pass

    # Surface a custom / uncurated model the user selected via the CLI.
    # Each row's model list is its curated/live catalog, so a model the user set
    # with `/model <provider>/<uncurated-name>` would otherwise be invisible in
    # every picker. Inject it at the front of the current provider's row
    # (matched by slug) so it is selectable and shown.
    if current_model:
        for _row in results:
            if not _row.get("is_current"):
                continue
            _models = _row.get("models") or []
            if current_model not in _models:
                _row["models"] = [current_model, *_models]
                _row["total_models"] = _row.get("total_models", len(_models)) + 1
            break

    # Sort: current provider first, then by model count descending
    results.sort(key=lambda r: (not r["is_current"], -r["total_models"]))

    return results


def list_picker_providers(
    current_provider: str = "",
    current_base_url: str = "",
    user_providers: dict = None,
    max_models: int | None = None,
    current_model: str = "",
    excluded_providers: list | None = None,
) -> List[dict]:
    """Interactive-picker variant of :func:`list_authenticated_providers`.

    Post-processes the base list so the ``/model`` picker (Telegram/Discord
    inline keyboards) only surfaces models that are actually callable in the
    current install:

    - Provider rows whose model list ends up empty are dropped.

    All other providers and metadata fields are passed through unchanged.
    The typed ``/model <name>`` path is unaffected -- only the interactive
    picker payload is narrowed.
    """
    providers = list_authenticated_providers(
        current_provider=current_provider,
        current_base_url=current_base_url,
        user_providers=user_providers,
        max_models=max_models,
        current_model=current_model,
        for_picker=True,
        excluded_providers=excluded_providers,
    )
    filtered: List[dict] = []
    for p in providers:
        if not p.get("models"):
            continue
        filtered.append(p)

    return filtered
