"""Per-provider model name normalization.

Pilotage talks to OpenAI directly and to Codex, and both accept bare model
ids.  The only translation still needed is repairing a redundant
``provider/`` prefix that users copy into ``config.yaml`` from an aggregator
slug — ``openai/gpt-5.4`` for the ``openai-codex`` provider, ``custom/my-model``
for ``custom``.

Callers write::

    api_model = normalize_model_for_provider(user_input, provider)
"""

from __future__ import annotations

# Providers that want bare names with dots preserved.
_STRIP_VENDOR_ONLY_PROVIDERS: frozenset[str] = frozenset({
    "openai-codex",
})

# Direct providers that accept bare native names but should repair a matching
# provider/ prefix when users copy the aggregator form into config.yaml.
_MATCHING_PREFIX_STRIP_PROVIDERS: frozenset[str] = frozenset({
    "custom",
})


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalize_provider_alias(provider_name: str) -> str:
    """Resolve provider aliases to Pilotage' canonical ids."""
    raw = (provider_name or "").strip().lower()
    if not raw:
        return raw
    try:
        from pilotage_cli.models import normalize_provider

        return normalize_provider(raw)
    except Exception:
        return raw


def _strip_matching_provider_prefix(model_name: str, target_provider: str) -> str:
    """Strip ``provider/`` only when the prefix matches the target provider.

    This prevents arbitrary slash-bearing model IDs from being mangled on
    native providers while still repairing manual config values.

    ``custom`` is a generic bucket for arbitrary user-defined endpoints, not a
    vendor identity. An alias that merely *resolves to* ``custom`` does not
    mean its prefix is redundant -- it may be the actual routing prefix a
    proxy in front of the custom endpoint (e.g. LiteLLM) requires. Only a
    literal ``custom/`` prefix -- the bucket's own name -- is treated as
    redundant here.
    """
    if "/" not in model_name:
        return model_name

    prefix, remainder = model_name.split("/", 1)
    if not prefix.strip() or not remainder.strip():
        return model_name

    normalized_target = _normalize_provider_alias(target_provider)
    if normalized_target == "custom":
        if prefix.strip().lower() == "custom":
            return remainder.strip()
        return model_name

    normalized_prefix = _normalize_provider_alias(prefix)
    if normalized_prefix and normalized_prefix == normalized_target:
        return remainder.strip()
    return model_name


# ---------------------------------------------------------------------------
# Main normalisation entry point
# ---------------------------------------------------------------------------

def normalize_model_for_provider(model_input: str, target_provider: str) -> str:
    """Translate a model name into the format the target provider's API expects.

    Args:
        model_input: The model name as provided by the user or config.
            Can be bare (``"gpt-5.4"``) or vendor-prefixed
            (``"openai/gpt-5.4"``).
        target_provider: The canonical Pilotage provider id, e.g.
            ``"openai"``, ``"openai-codex"``, ``"custom"``.  Should already be
            normalised via ``pilotage_cli.models.normalize_provider()``.

    Returns:
        The model identifier string that the target provider's API expects.
        Always a best-effort string -- never raises.

    Examples::

        >>> normalize_model_for_provider("openai/gpt-5.4", "openai-codex")
        'gpt-5.4'

        >>> normalize_model_for_provider("custom/my-model", "custom")
        'my-model'

        >>> normalize_model_for_provider("my-model", "custom")
        'my-model'
    """
    name = (model_input or "").strip()
    if not name:
        return name

    provider = _normalize_provider_alias(target_provider)

    # --- openai-codex: strip matching provider prefix, keep dots ---
    if provider in _STRIP_VENDOR_ONLY_PROVIDERS:
        stripped = _strip_matching_provider_prefix(name, provider)
        if stripped == name and name.startswith("openai/"):
            # openai-codex maps openai/gpt-5.4 -> gpt-5.4
            return name.split("/", 1)[1]
        return stripped

    # --- Direct providers: repair matching provider prefixes only ---
    if provider in _MATCHING_PREFIX_STRIP_PROVIDERS:
        return _strip_matching_provider_prefix(name, provider)

    # --- Everything else: pass through as-is ---
    return name
