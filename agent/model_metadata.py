"""Model metadata, context lengths, and token estimation utilities.

Pure utility functions with no AIAgent dependency. Used by ContextCompressor
and run_agent.py for pre-flight context checks.
"""

import base64
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

import yaml

if TYPE_CHECKING:  # pragma: no cover — runtime import is lazy (see below)
    import requests

from utils import atomic_json_write, atomic_yaml_write, base_url_host_matches, base_url_hostname

from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS

logger = logging.getLogger(__name__)

# ``requests`` (with urllib3) costs ~27 ms of the `import cli` waterfall and
# is only used inside the fetch functions below. It's resolved lazily:
# ``_ensure_requests()`` populates the module global on the runtime path, and
# the PEP 562 ``__getattr__`` covers external attribute access — notably
# ``patch("agent.model_metadata.requests.get")`` in tests, which resolves the
# attribute at patch time.


def _ensure_requests():
    if "requests" not in globals():
        import requests as _requests
        globals()["requests"] = _requests
    return globals()["requests"]


def __getattr__(name: str):
    if name == "requests":
        return _ensure_requests()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_requests_verify(base_url: str = "") -> bool | str:
    """Resolve SSL verify setting for `requests` calls.

    Priority (mirrors ``agent.ssl_verify.resolve_httpx_verify`` so the
    ``requests``-based ``/models`` probes agree with the httpx chat client):

    1. Per-provider ``ssl_verify: false`` for ``base_url`` — disable verification.
    2. Per-provider ``ssl_ca_cert`` for ``base_url`` — an explicit CA bundle.
       Without this, a custom endpoint whose chain only verifies against the
       provider's configured bundle (not the process ``SSL_CERT_FILE``) logs a
       spurious CERTIFICATE_VERIFY_FAILED on every probe even though the chat
       path succeeds (per-provider ``ssl_ca_cert`` was reaching only httpx).
    3. Env vars ``PILOTAGE_CA_BUNDLE`` / ``REQUESTS_CA_BUNDLE`` / ``SSL_CERT_FILE``
       (a single var covers both ``requests`` and ``httpx`` in-process).
    4. ``True`` — defer to the requests default (certifi).

    ``base_url`` is optional so existing callers keep the
    env-only behavior unchanged; only probes that pass a base_url pick up the
    per-provider override.
    """
    if base_url:
        try:
            from pilotage_cli.config import get_custom_provider_tls_settings
            tls = get_custom_provider_tls_settings(base_url)
            if tls.get("ssl_verify") is False:
                return False
            ca = tls.get("ssl_ca_cert")
            if isinstance(ca, str) and ca and os.path.isfile(ca):
                return ca
        except Exception:
            pass  # fall through to env vars — never break a probe on config lookup
    for env_var in ("PILOTAGE_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        val = os.getenv(env_var)
        if val and os.path.isfile(val):
            return val
    return True

try:
    from providers import list_providers as _list_providers
except Exception:
    def _list_providers():
        return []

def _strip_provider_prefix(model: str) -> str:
    """Strip a recognised provider prefix from a model string.

    Provider names and aliases come from the provider-profile registry, so
    bundled and user plugins are recognised without a core catalog update.

    ``"local:my-model"`` → ``"my-model"``
    ``"model:v2"``      → ``"model:v2"``  (unchanged — not a provider prefix)
    """
    if ":" not in model or model.startswith("http"):
        return model
    prefix, suffix = model.split(":", 1)
    prefix_lower = prefix.strip().lower()
    try:
        from providers import get_provider_profile

        is_provider = get_provider_profile(prefix_lower) is not None
    except Exception:
        is_provider = False
    if is_provider:
        return suffix
    return model

_endpoint_model_metadata_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
_endpoint_model_metadata_cache_time: Dict[str, float] = {}
_ENDPOINT_MODEL_CACHE_TTL = 300
# A configured endpoint that is routable-but-dead — e.g. a corp LAN address
# while off-VPN — blackholes TCP: the SYN draws no SYN-ACK, no RST and no ICMP
# error, so a probe waits out its full timeout instead of failing fast. Startup
# runs a whole waterfall of such probes across several functions here, and the
# stalls stack into a minute-long hang before the banner renders.
#
# Once ANY probe has actually observed a connect timeout for an endpoint, the
# others have nothing to gain by repeating it. Recording that observation and
# short-circuiting on it performs no network I/O of its own — it adds no probe
# for callers or tests to mock, and it can only ever fire after a real timeout
# has already been paid, so it cannot suppress a probe that would have worked.
_ENDPOINT_BLACKHOLE_TTL_SECONDS = 30.0
# Values are monotonic timestamps of the last observed connect timeout.
_endpoint_blackhole_cache: Dict[str, float] = {}


def _endpoint_host_key(base_url: str) -> Optional[str]:
    """Return a ``host:port`` key for ``base_url``, or None if it has no host.

    Keyed on host:port rather than the full URL so every probe path for one
    server — ``/v1``-suffixed or not — shares a single entry.
    """
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return None
    url = normalized if "://" in normalized else f"http://{normalized}"
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except Exception:
        return None
    return f"{host}:{port}" if host else None


def _note_endpoint_blackholed(base_url: str) -> None:
    """Record that a probe to ``base_url`` timed out during TCP connect."""
    key = _endpoint_host_key(base_url)
    if key is None:
        return
    _endpoint_blackhole_cache[key] = time.monotonic()
    logger.debug(
        "Endpoint %s timed out connecting — skipping further probes for %.0fs",
        key, _ENDPOINT_BLACKHOLE_TTL_SECONDS,
    )


def _endpoint_blackholed(base_url: str) -> bool:
    """True if a recent probe to ``base_url`` timed out during TCP connect.

    Pure cache lookup; never touches the network. The entry expires after
    _ENDPOINT_BLACKHOLE_TTL_SECONDS — long enough to collapse one startup's
    burst of probes, short enough that bringing the VPN up mid-session is
    picked up without a restart.
    """
    if _ENDPOINT_BLACKHOLE_TTL_SECONDS <= 0:
        return False
    key = _endpoint_host_key(base_url)
    if key is None:
        return False
    seen = _endpoint_blackhole_cache.get(key)
    if seen is None:
        return False
    if (time.monotonic() - seen) >= _ENDPOINT_BLACKHOLE_TTL_SECONDS:
        del _endpoint_blackhole_cache[key]
        return False
    return True


def _is_connect_timeout(exc: BaseException) -> bool:
    """True for connect-phase timeouts raised by httpx or requests.

    Read timeouts are deliberately excluded: those mean the server accepted
    the connection, which is the opposite of the blackhole this guards.
    """
    try:
        import httpx
        if isinstance(exc, httpx.ConnectTimeout):
            return True
    except Exception:
        pass
    try:
        from requests.exceptions import ConnectTimeout
        if isinstance(exc, ConnectTimeout):
            return True
    except Exception:
        pass
    return False

# Descending tiers for context length probing when the model is unknown.
# We start at 256K (covers GPT-5.x, many current large-context models) and
# step down on context-length errors until one works.  Tier[0] is also the
# default fallback when no detection method succeeds.
CONTEXT_PROBE_TIERS = [
    256_000,
    128_000,
    64_000,
    32_000,
    16_000,
    8_000,
]

# Default context length when no detection method succeeds.
DEFAULT_FALLBACK_CONTEXT = CONTEXT_PROBE_TIERS[0]

# (model, base_url) pairs that already emitted the fallback warning.
# The fallback result itself is deliberately never cached, so without this
# the warning would repeat on every resolution for the same unknown model.
_FALLBACK_WARNED: set = set()


def _warn_context_length_fallback(model: str, base_url: str) -> None:
    """Warn (once per model+endpoint) that context detection failed and the
    hard default is being used, so small-context models (8K, 32K) don't
    silently get 256K and cause hard-to-debug API failures."""
    key = (model, base_url or "")
    if key in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(key)
    logger.warning(
        "Could not determine context length for model %r (base_url=%s) "
        "— falling back to %s tokens. Set model.context_length in "
        "config.yaml to override.",
        model, base_url or "default", f"{DEFAULT_FALLBACK_CONTEXT:,}",
    )

# Minimum context length required to run Pilotage Agent.  Models with fewer
# tokens cannot maintain enough working memory for tool-calling workflows.
# Sessions, model switches, and cron jobs should reject models below this.
MINIMUM_CONTEXT_LENGTH = 64_000

# Thin fallback defaults — only broad model family patterns.
# These fire only when the provider is unknown and models.dev misses.
# For provider-specific context lengths, models.dev is the primary source.
DEFAULT_CONTEXT_LENGTHS = {
    # OpenAI — GPT-5 family (most have 400k; specific overrides first)
    # Source: https://developers.openai.com/api/docs/models
    # GPT-5.5 (launched Apr 23 2026) is 1.05M on the direct OpenAI API and
    # ChatGPT Codex OAuth caps it at 272K; both paths resolve via their own
    # provider-aware branches (_resolve_codex_oauth_context_length + models.dev).
    # This hardcoded value is only reached when every probe misses.
    # GPT-5.6 series (Sol/Terra/Luna, GA 2026-07-09) — 1.05M on the direct
    # OpenAI API (same as gpt-5.5). Codex OAuth caps these at 272K.
    # (Lookups length-sort keys at match time, so dict order is cosmetic.)
    "gpt-5.6-luna": 1050000,
    "gpt-5.6-terra": 1050000,
    "gpt-5.6-sol": 1050000,
    "gpt-5.5": 1050000,
    "gpt-5.4-nano": 400000,           # 400k (not 1.05M like full 5.4)
    "gpt-5.4-mini": 400000,           # 400k (not 1.05M like full 5.4)
    "gpt-5.4": 1050000,               # GPT-5.4, GPT-5.4 Pro (1.05M context)
    # gpt-5.3-codex-spark is Codex-OAuth-only (ChatGPT Pro entitlement) and
    # uses a smaller 128k window than other gpt-5.x slugs. Listed here as
    # a defensive override so the longest-substring fallback doesn't match
    # the generic "gpt-5" entry below (400k) and report the wrong limit if
    # Spark's context ever needs to be resolved through this path. Real
    # usage flows through _CODEX_OAUTH_CONTEXT_FALLBACK at line ~1113.
    "gpt-5.3-codex-spark": 128000,
    "gpt-5.1-chat": 128000,           # Chat variant has 128k context
    "gpt-5": 400000,                  # GPT-5.x base, mini, codex variants (400k)
    "gpt-4.1": 1047576,
    "gpt-4": 128000,
}


_CONTEXT_LENGTH_KEYS = (
    "context_length",
    "context_window",
    "context_size",
    "max_context_length",
    "max_position_embeddings",
    "max_model_len",
    "max_input_tokens",
    "max_sequence_length",
    "max_seq_len",
    "n_ctx_train",
    "n_ctx",
    "ctx_size",
)

_MAX_COMPLETION_KEYS = (
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens",
)

def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _auth_headers(api_key: str = "") -> Dict[str, str]:
    token = str(api_key or "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _is_custom_endpoint(base_url: str) -> bool:
    return bool(_normalize_base_url(base_url))


_URL_TO_PROVIDER: Dict[str, str] = {
    "api.openai.com": "openai",
    "chatgpt.com": "openai",
}

# Auto-extend with hostnames derived from provider profiles.
# Any provider with a base_url not already in the map gets added automatically.
try:
    for _pp in _list_providers():
        _host = _pp.get_hostname()
        if _host and _host not in _URL_TO_PROVIDER:
            _URL_TO_PROVIDER[_host] = _pp.name
except Exception:
    pass


def _infer_provider_from_url(base_url: str) -> Optional[str]:
    """Infer the models.dev provider name from a base URL.

    This allows context length resolution via models.dev for custom
    endpoints without requiring the user to set the provider name in config.
    """
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return None
    parsed = urlparse(normalized if "://" in normalized else f"https://{normalized}")
    host = parsed.netloc.lower() or parsed.path.lower()
    for url_part, provider in _URL_TO_PROVIDER.items():
        if url_part in host:
            return provider
    return None


def _is_known_provider_base_url(base_url: str) -> bool:
    return _infer_provider_from_url(base_url) is not None


def _skip_persistent_context_cache(base_url: str, provider: str) -> bool:
    """Return True when the on-disk context cache must not short-circuit probing.

    Codex OAuth excludes caching because its context window is account- and
    entitlement-specific metadata supplied by the authenticated /models
    endpoint. A fallback value written after a transient probe failure must
    not prevent a later live probe from observing an updated allocation.
    """
    return (provider or "").strip().lower() == "openai-codex"


def _iter_nested_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_nested_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_dicts(item)


def _coerce_reasonable_int(value: Any, minimum: int = 1024, maximum: int = 10_000_000) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        result = int(value)
    except (TypeError, ValueError):
        return None
    if minimum <= result <= maximum:
        return result
    return None


def _extract_first_int(payload: Dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    keyset = {key.lower() for key in keys}
    for mapping in _iter_nested_dicts(payload):
        for key, value in mapping.items():
            if str(key).lower() not in keyset:
                continue
            coerced = _coerce_reasonable_int(value)
            if coerced is not None:
                return coerced
    return None


def _extract_context_length(payload: Dict[str, Any]) -> Optional[int]:
    return _extract_first_int(payload, _CONTEXT_LENGTH_KEYS)


def _extract_max_completion_tokens(payload: Dict[str, Any]) -> Optional[int]:
    return _extract_first_int(payload, _MAX_COMPLETION_KEYS)


def _extract_pricing(payload: Dict[str, Any]) -> Dict[str, Any]:
    alias_map = {
        "prompt": ("prompt", "input", "input_cost_per_token", "prompt_token_cost"),
        "completion": ("completion", "output", "output_cost_per_token", "completion_token_cost"),
        "request": ("request", "request_cost"),
        "cache_read": ("cache_read", "cached_prompt", "input_cache_read", "cache_read_cost_per_token"),
        "cache_write": ("cache_write", "cache_creation", "input_cache_write", "cache_write_cost_per_token"),
    }
    for mapping in _iter_nested_dicts(payload):
        normalized = {str(key).lower(): value for key, value in mapping.items()}
        if not any(any(alias in normalized for alias in aliases) for aliases in alias_map.values()):
            continue
        pricing: Dict[str, Any] = {}
        for target, aliases in alias_map.items():
            for alias in aliases:
                if alias in normalized and normalized[alias] not in {None, ""}:
                    pricing[target] = normalized[alias]
                    break
        if pricing:
            return pricing
    return {}


def _add_model_aliases(cache: Dict[str, Dict[str, Any]], model_id: str, entry: Dict[str, Any]) -> None:
    cache[model_id] = entry
    if "/" in model_id:
        bare_model = model_id.split("/", 1)[1]
        cache.setdefault(bare_model, entry)


def fetch_endpoint_model_metadata(
    base_url: str,
    api_key: str = "",
    force_refresh: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Fetch model metadata from an OpenAI-compatible ``/models`` endpoint.

    This is used for explicit custom endpoints where hardcoded global model-name
    defaults are unreliable. Results are cached in memory per base URL.
    """
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return {}
    _ensure_requests()

    if not force_refresh:
        cached = _endpoint_model_metadata_cache.get(normalized)
        cached_at = _endpoint_model_metadata_cache_time.get(normalized, 0)
        if cached is not None and (time.time() - cached_at) < _ENDPOINT_MODEL_CACHE_TTL:
            return cached

    # Blackholed endpoint: every candidate below would spend its full 5s
    # connect budget. Returned empty rather than cached, so the endpoint is
    # retried as soon as the blackhole entry expires.
    if _endpoint_blackholed(normalized):
        return {}

    candidates = [normalized]
    if normalized.endswith("/v1"):
        alternate = normalized[:-3].rstrip("/")
    else:
        alternate = normalized + "/v1"
    if alternate and alternate not in candidates:
        candidates.append(alternate)

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_error: Optional[Exception] = None

    for candidate in candidates:
        # A connect timeout on one candidate condemns the host, not the path:
        # the remaining candidates differ only by URL suffix, so trying them
        # would repeat the same stall.
        if _endpoint_blackholed(normalized):
            break
        url = candidate.rstrip("/") + "/models"
        response = None
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=(5, 10),
                verify=_resolve_requests_verify(normalized),
                stream=True,
            )
            if response.status_code in (401, 403):
                logger.debug(
                    "Model metadata probe received HTTP %s from %s; stopping candidate probing",
                    response.status_code,
                    url,
                )
                break
            response.raise_for_status()
            payload = response.json()
            cache: Dict[str, Dict[str, Any]] = {}
            for model in payload.get("data", []):
                if not isinstance(model, dict):
                    continue
                model_id = model.get("id")
                if not model_id:
                    continue
                entry: Dict[str, Any] = {"name": model.get("name", model_id)}
                context_length = _extract_context_length(model)
                if context_length is not None:
                    entry["context_length"] = context_length
                max_completion_tokens = _extract_max_completion_tokens(model)
                if max_completion_tokens is not None:
                    entry["max_completion_tokens"] = max_completion_tokens
                pricing = _extract_pricing(model)
                if pricing:
                    entry["pricing"] = pricing
                _add_model_aliases(cache, model_id, entry)

            _endpoint_model_metadata_cache[normalized] = cache
            _endpoint_model_metadata_cache_time[normalized] = time.time()
            return cache
        except Exception as exc:
            last_error = exc
            if _is_connect_timeout(exc):
                _note_endpoint_blackholed(normalized)
        finally:
            if response is not None:
                response.close()

    if last_error:
        logger.debug("Failed to fetch model metadata from %s/models: %s", normalized, last_error)
    _endpoint_model_metadata_cache[normalized] = {}
    _endpoint_model_metadata_cache_time[normalized] = time.time()
    return {}


def _resolve_endpoint_context_length(
    model: str,
    base_url: str,
    api_key: str = "",
) -> Optional[int]:
    """Resolve context length from an endpoint's live ``/models`` metadata."""
    endpoint_metadata = fetch_endpoint_model_metadata(base_url, api_key=api_key)
    matched = endpoint_metadata.get(model)
    if not matched:
        if len(endpoint_metadata) == 1:
            matched = next(iter(endpoint_metadata.values()))
        elif model:
            # Substring fuzzy match — only meaningful with a non-empty model
            # name.  An empty string is a substring of EVERY key, which would
            # "match" whatever model the endpoint happens to list first (e.g.
            # a 32K embedding model) and poison the
            # resolved context length for the whole agent.
            for key, entry in endpoint_metadata.items():
                if model in key or key in model:
                    matched = entry
                    break
    if matched:
        context_length = matched.get("context_length")
        if isinstance(context_length, int):
            return context_length
    return None


def _get_context_cache_path() -> Path:
    """Return path to the persistent context length cache file."""
    from pilotage_constants import get_pilotage_home
    return get_pilotage_home() / "context_length_cache.yaml"


def _load_context_cache() -> Dict[str, int]:
    """Load the model+provider -> context_length cache from disk."""
    path = _get_context_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("context_lengths") or {}
    except Exception as e:
        logger.debug("Failed to load context length cache: %s", e)
        return {}


def _context_cache_key(model: str, base_url: str) -> str:
    """Canonical ``model@base_url`` key for the persistent context cache.

    Trailing slashes are stripped so ``http://host/v1`` and
    ``http://host/v1/`` share one entry instead of creating duplicates
    that can go stale independently.
    """
    return f"{model}@{(base_url or '').rstrip('/')}"


def save_context_length(model: str, base_url: str, length: int) -> None:
    """Persist a discovered context length for a model+provider combo.

    Cache key is ``model@base_url`` so the same model name served from
    different providers can have different limits.
    """
    # Never persist non-positive values — a 0 or negative context length
    # is always a bug and would poison the cache, causing downstream
    # `get_model_context_length()` to return 0 (since `0 is not None`).
    if length <= 0:
        logger.warning(
            "Refusing to cache non-positive context length %s -> %s tokens",
            f"{model}@{base_url}", length,
        )
        return
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    if cache.get(key) == length:
        return  # already stored
    cache[key] = length
    path = _get_context_cache_path()
    try:
        # Atomic write (temp file + fsync + os.replace): a plain truncating
        # ``open(path, "w")`` leaves the file empty/partial if the process is
        # killed mid-dump, and the next _load_context_cache() swallows the
        # resulting YAML error and returns {} — silently wiping EVERY cached
        # context length. It also exposes torn reads to a concurrent process
        # reading between truncate and dump-complete.
        atomic_yaml_write(path, {"context_lengths": cache})
        logger.info("Cached context length %s -> %s tokens", key, f"{length:,}")
    except Exception as e:
        logger.debug("Failed to save context length cache: %s", e)


def get_cached_context_length(model: str, base_url: str) -> Optional[int]:
    """Look up a previously discovered context length for model+provider."""
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    hit = cache.get(key)
    if hit is not None:
        return hit
    # Legacy rows written before key normalization may carry a trailing
    # slash — honor them rather than re-probing. Checked regardless of the
    # caller's slash form: the row's shape and the caller's shape can differ
    # in either direction (old slashed row + new normalized config, or the
    # reverse), so probe the literal form and the slashed canonical form.
    for legacy_key in (f"{model}@{base_url}", f"{key}/"):
        if legacy_key != key:
            hit = cache.get(legacy_key)
            if hit is not None:
                return hit
    return None


def _invalidate_cached_context_length(model: str, base_url: str) -> None:
    """Drop a stale cache entry so it gets re-resolved on the next lookup."""
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    # Clear every key shape for this pair: canonical, the caller's literal
    # form, and the slashed legacy form — same set get_cached_context_length
    # consults, so a lookup can never resurrect a row invalidation missed.
    stale_keys = {key, f"{model}@{base_url}", f"{key}/"}
    if not any(k in cache for k in stale_keys):
        return
    for k in stale_keys:
        cache.pop(k, None)
    path = _get_context_cache_path()
    try:
        # Atomic write — see save_context_length() for why a plain truncating
        # open() here risks wiping the entire cache on an interrupted dump.
        atomic_yaml_write(path, {"context_lengths": cache})
    except Exception as e:
        logger.debug("Failed to invalidate context length cache entry %s: %s", key, e)


def get_next_probe_tier(current_length: int) -> Optional[int]:
    """Return the next lower probe tier, or None if already at minimum."""
    for tier in CONTEXT_PROBE_TIERS:
        if tier < current_length:
            return tier
    return None


def parse_context_limit_from_error(error_msg: str) -> Optional[int]:
    """Try to extract the actual context limit from an API error message.

    Many providers include the limit in their error text, e.g.:
      - "maximum context length is 32768 tokens"
      - "context_length_exceeded: 131072"
      - "Maximum context size 32768 exceeded"
      - "model's max context length is 65536"
    """
    error_lower = error_msg.lower()
    # Pattern: look for numbers near context-related keywords
    patterns = [
        r'max_model_len\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM: "max_model_len 32768", "=32768", ": 32768", "(32768)", "is 32768"
        r'maximum model length\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM alt: "maximum model length 131072", "... is 131072"
        r'(?:max(?:imum)?|limit)\s*(?:context\s*)?(?:length|size|window)?\s*(?:is|of|:)?\s*(\d{4,})',
        r'context\s*(?:length|size|window)\s*(?:is|of|:)?\s*(\d{4,})',
        r'(\d{4,})\s*(?:token)?\s*(?:context|limit)',
        r'>\s*(\d{4,})\s*(?:max|limit|token)',  # "250000 tokens > 200000 maximum"
        r'(\d{4,})\s*(?:max(?:imum)?)\b',  # "200000 maximum"
    ]
    for pattern in patterns:
        match = re.search(pattern, error_lower)
        if match:
            limit = int(match.group(1))
            # Sanity check: must be a reasonable context length
            if 1024 <= limit <= 10_000_000:
                return limit
    return None


def get_context_length_from_provider_error(
    error_msg: str,
    current_context_length: int,
) -> Optional[int]:
    """Return a provider-reported lower context limit, if one is present.

    Context-overflow recovery must not invent a new model window size.  Some
    providers only say that the input exceeds the context window without
    reporting the actual maximum.  In that case callers should keep the
    configured context length and try compression only, rather than stepping
    down through guessed probe tiers (1M → 256K → 128K → ...).
    """
    parsed_limit = parse_context_limit_from_error(error_msg)
    if parsed_limit is None:
        return None
    if parsed_limit < current_context_length:
        return parsed_limit
    return None


def parse_available_output_tokens_from_error(error_msg: str) -> Optional[int]:
    """Detect an "output cap too large" error and return how many output tokens are available.

    Background — two distinct context errors exist:
      1. "Prompt too long"  — the INPUT itself exceeds the context window.
           Fix: compress history, and only reduce context_length if the
           provider explicitly reports the actual lower limit.
      2. "max_tokens too large" — input is fine, but input + requested_output > window.
           Fix: reduce max_tokens (the output cap) for this call.
           Do NOT touch context_length — the window hasn't shrunk.

    Returns the number of output tokens that would fit, or None if the error
    does not look like a max_tokens-too-large error.
    """
    error_lower = error_msg.lower()

    # Must look like an output-cap error, not a prompt-length error.
    #   "This model's maximum context length is 65536 tokens. However, you
    #    requested 65536 output tokens and your prompt contains ..."
    # The "requested N output tokens" phrasing means the OUTPUT cap is the
    # problem (the input itself fits) — reduce max_tokens, don't compress.
    is_output_cap_error = (
        "maximum context length" in error_lower
        and "requested" in error_lower
        and "output tokens" in error_lower
    )
    if not is_output_cap_error:
        return None

    _m_ctx_tok = re.search(r'maximum context length is (\d+)\s*token', error_lower)

    # Both the window and the prompt are reported in TOKENS, e.g.
    #   "This model's maximum context length is 131072 tokens. However, you
    #    requested 65536 output tokens and your prompt contains at least 65537
    #    input tokens, for a total of at least 131073 tokens. Please reduce
    #    the length of the input prompt or the number of requested output
    #    tokens."
    # Available output = window - input. When the input alone is at or over
    # the window this stays None, so the caller correctly falls through to
    # compression instead of futilely shrinking the output cap.
    _m_vllm_input = re.search(
        r'prompt contains (?:at least )?(\d+)\s*input tokens', error_lower
    )
    if _m_ctx_tok and _m_vllm_input:
        _available = int(_m_ctx_tok.group(1)) - int(_m_vllm_input.group(1))
        if _available >= 1:
            return _available

    return None


def is_output_cap_error(error_msg: str) -> bool:
    """Return True if a 400 is about the OUTPUT cap (max_tokens) being too large.

    This is the broader sibling of :func:`parse_available_output_tokens_from_error`:
    that function only returns a number when it can extract the available output
    budget from a known phrasing.  This one answers the cheaper yes/no
    question — "is this an output-cap error at all?" — for wordings we may not
    yet parse a number from.

    Why this matters: an output-cap 400 is deterministic (every retry with the
    same ``max_tokens`` gets the identical rejection).  If such an error is
    misclassified as a context-overflow it gets routed into the compression
    loop, the compressor re-issues the call with the same oversized
    ``max_tokens``, the provider rejects it identically, and the session
    death-loops until "cannot compress further".  Compression cannot help an
    output-cap error — the input already fits.

    The signal: the error talks about ``max_tokens`` (or its aliases) as a
    cap/range/limit, and does NOT talk about the INPUT/prompt/context window
    being too long.  When both are present we defer to the context-overflow
    path (a real input overflow can also mention max_tokens).
    """
    error_lower = error_msg.lower()

    mentions_output_param = (
        "max_tokens" in error_lower
        or "max_output_tokens" in error_lower
        or "max_completion_tokens" in error_lower
    )
    if not mentions_output_param:
        return False

    # Phrasing that signals the OUTPUT cap specifically is the problem.
    output_cap_signal = (
        ("requested" in error_lower
            and "output tokens" in error_lower)
        or "should be" in error_lower                       # generic "max_tokens should be <= N"
        or "less than or equal" in error_lower
        or "must be" in error_lower
    )
    if not output_cap_signal:
        return False

    # If the error ALSO clearly describes an oversized INPUT, it is a genuine
    # context overflow that happens to mention max_tokens — let the
    # context-overflow path handle it (it can compress the input).
    input_overflow_signal = (
        "prompt is too long" in error_lower
        or "prompt too long" in error_lower
        or "input is too long" in error_lower
        or "input token" in error_lower
        or "prompt length" in error_lower
        or "prompt contains" in error_lower
        or "reduce the length" in error_lower
    )
    return not input_overflow_signal


def _model_id_matches(candidate_id: str, lookup_model: str) -> bool:
    """Return True if *candidate_id* (from server) matches *lookup_model* (configured).

    Supports two forms:
    - Exact match:  "gpt-5-codex" == "gpt-5-codex"
    - Slug match:   "openai/gpt-5-codex" matches "gpt-5-codex"
                    (the part after the last "/" equals lookup_model)

    This covers endpoints that list models as "publisher/slug" while users
    configure only the bare slug.
    """
    if candidate_id == lookup_model:
        return True
    # Slug match: basename of candidate equals the lookup name
    if "/" in candidate_id and candidate_id.rsplit("/", 1)[1] == lookup_model:
        return True
    return False


# Known ChatGPT Codex OAuth context windows (observed via live
# chatgpt.com/backend-api/codex/models probe, Apr 2026). These are the
# `context_window` values, which are what Codex actually enforces — the
# direct OpenAI API has larger limits for the same slugs, but Codex OAuth
# caps lower (e.g. gpt-5.5 is 1.05M on the API, 272K on Codex).
#
# Used as a fallback when the live probe fails (no token, network error).
# Longest keys first so substring match picks the most specific entry.
_CODEX_OAUTH_CONTEXT_FALLBACK: Dict[str, int] = {
    "gpt-5.1-codex-max": 272_000,
    "gpt-5.1-codex-mini": 272_000,
    "gpt-5.3-codex": 272_000,
    # Spark runs on specialised low-latency hardware and exposes a smaller
    # 128k window than other Codex OAuth slugs. Listed explicitly so the
    # longest-key-first fallback resolves it correctly — substring match
    # on "gpt-5.3-codex" otherwise wins and reports 272k. Availability is
    # gated by ChatGPT Pro entitlement on the Codex backend.
    "gpt-5.3-codex-spark": 128_000,
    "gpt-5.2-codex": 272_000,
    "gpt-5.4-mini": 272_000,
    "gpt-5.6-sol": 272_000,
    "gpt-5.6-terra": 272_000,
    "gpt-5.6-luna": 272_000,
    "gpt-5.5": 272_000,
    "gpt-5.4": 272_000,
    "gpt-5.2": 272_000,
    "gpt-5": 272_000,
}


_codex_oauth_context_cache: Dict[str, Tuple[Dict[str, int], float]] = {}
_CODEX_OAUTH_CONTEXT_CACHE_TTL = 3600  # 1 hour


def _codex_oauth_token_fingerprint(access_token: str) -> str:
    """Return a non-secret cache key for a Codex OAuth access token."""
    return hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:16]


def _extract_chatgpt_account_id(access_token: str) -> Optional[str]:
    """Extract ``chatgpt_account_id`` from the Codex OAuth JWT.

    The Codex ``/backend-api/codex/models`` endpoint returns the per-account
    catalog only when the ``ChatGPT-Account-Id`` header is present; without
    it, the endpoint returns ``{"models":[]}`` (HTTP 200) and the context
    probe falls back to the hardcoded defaults — which can be stale or
    wrong for the active account's plan. Mirrors the same extraction done
    in ``auxiliary_client.py`` for the request path.

    Returns ``None`` on any parse error rather than raising, so a bad
    token still surfaces as a normal probe failure instead of crashing
    the metadata resolver.
    """
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        if not isinstance(claims, dict):
            return None
        acct_id = claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
        return acct_id if isinstance(acct_id, str) and acct_id else None
    except Exception:
        return None


def _fetch_codex_oauth_context_lengths_with_source(
    access_token: str,
) -> Tuple[Dict[str, int], bool]:
    """Fetch Codex catalogue data and report whether it came from HTTP.

    The in-process cache is scoped by token fingerprint because Codex model
    availability and context windows can vary by account entitlement. The raw
    token is never retained in the cache key. The boolean is false for a
    same-token in-process hit, which must not be treated as a fresh provider
    confirmation when deciding whether to update persistent state.
    """
    global _codex_oauth_context_cache
    now = time.time()
    cache_key = _codex_oauth_token_fingerprint(access_token)
    cached = _codex_oauth_context_cache.get(cache_key)
    if cached is not None:
        cached_models, cached_at = cached
        if now - cached_at < _CODEX_OAUTH_CONTEXT_CACHE_TTL:
            return cached_models, False

    headers = {"Authorization": f"Bearer {access_token}"}
    acct_id = _extract_chatgpt_account_id(access_token)
    if acct_id:
        headers["ChatGPT-Account-Id"] = acct_id

    try:
        _ensure_requests()
        resp = requests.get(
            "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0",
            headers=headers,
            timeout=(5, 10),
            verify=_resolve_requests_verify(),
        )
        if resp.status_code != 200:
            logger.debug(
                "Codex /models probe returned HTTP %s; falling back to hardcoded defaults",
                resp.status_code,
            )
            return {}, False
        data = resp.json()
    except Exception as exc:
        logger.debug("Codex /models probe failed: %s", exc)
        return {}, False

    entries = data.get("models", []) if isinstance(data, dict) else []
    result: Dict[str, int] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        ctx = item.get("context_window")
        if isinstance(slug, str) and isinstance(ctx, int) and ctx > 0:
            result[slug.strip()] = ctx

    if result:
        _codex_oauth_context_cache[cache_key] = (result, now)
    return result, True


def _fetch_codex_oauth_context_lengths(access_token: str) -> Dict[str, int]:
    """Probe the ChatGPT Codex /models endpoint for per-slug context windows.

    Codex OAuth imposes its own context limits that differ from the direct
    OpenAI API (e.g. gpt-5.5 is 1.05M on the API, 272K on Codex). The
    `context_window` field in each model entry is the authoritative source.

    Returns a ``{slug: context_window}`` dict. Empty on failure.
    """
    result, _fresh = _fetch_codex_oauth_context_lengths_with_source(access_token)
    return result


def _resolve_codex_oauth_context_length_with_source(
    model: str, access_token: str = ""
) -> Tuple[Optional[int], str]:
    """Resolve a Codex OAuth model's real context window.

    Prefers a live probe of chatgpt.com/backend-api/codex/models (when we
    have a bearer token), then falls back to ``_CODEX_OAUTH_CONTEXT_FALLBACK``.

    Returns ``(context_length, source)`` where source is ``"live"`` for a
    value returned by a fresh authenticated endpoint probe, ``"memory"`` for
    a same-token in-process catalogue hit, or ``"fallback"`` for the static
    conservative table. Only ``"live"`` is eligible for persistent writes.
    """
    model_bare = _strip_provider_prefix(model).strip()
    if not model_bare:
        return None, ""

    if access_token:
        live, fresh_probe = _fetch_codex_oauth_context_lengths_with_source(access_token)
        live_source = "live" if fresh_probe else "memory"
        if model_bare in live:
            return live[model_bare], live_source
        # Case-insensitive match in case casing drifts
        model_lower = model_bare.lower()
        for slug, ctx in live.items():
            if slug.lower() == model_lower:
                return ctx, live_source

    # Fallback: longest-key-first substring match over hardcoded defaults.
    model_lower = model_bare.lower()
    for slug, ctx in sorted(
        _CODEX_OAUTH_CONTEXT_FALLBACK.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if slug in model_lower:
            return ctx, "fallback"

    return None, ""


def _resolve_codex_oauth_context_length(
    model: str, access_token: str = ""
) -> Optional[int]:
    """Resolve a Codex OAuth model's context length (compatibility wrapper)."""
    context_length, _source = _resolve_codex_oauth_context_length_with_source(
        model, access_token=access_token,
    )
    return context_length



def get_model_context_length(
    model: str,
    base_url: str = "",
    api_key: str = "",
    config_context_length: int | None = None,
    provider: str = "",
    custom_providers: list | None = None,
) -> int:
    """Get the context length for a model.

    Resolution order:
    0. Explicit config override (model.context_length or custom_providers per-model)
    0b. model_overrides config (per-provider+model context_window override)
    0c. Endpoint-scoped metadata for models validated on one multiplexed endpoint
    1. Persistent cache (previously discovered via probing).  Codex OAuth
       bypasses the cache here so its provider metadata can be reconciled
       against the authoritative live source.
    2. Active endpoint metadata (/models for explicit custom endpoints)
    3. Provider-aware lookups:
       a. Codex OAuth /models probe
       b. models.dev registry lookup
    4. Hardcoded defaults (broad family patterns, longest-key-first)
    5. Default fallback (256K)"""
    # 0. Explicit config override — user knows best
    if config_context_length is not None and isinstance(config_context_length, int) and config_context_length > 0:
        return config_context_length

    # 0b. model_overrides config — EXPLICIT per-provider+model context_window
    # override only (fill-gap _default entries are applied later, inside
    # lookup_models_dev_context at step 5f, once the catalog has actually
    # missed — so a _default can never preempt custom_providers or live
    # probes). This is the supported self-unblock path for models with
    # wrong context in models.dev and for custom/local models
    # Config-read only; never blocks on the network.
    if provider and model:
        try:
            from agent.models_dev import _override_context_window
            mo_ctx = _override_context_window(provider, model)
            if mo_ctx is not None and mo_ctx > 0:
                return mo_ctx
        except Exception:
            pass  # fall through to other resolution paths

    # 0c. custom_providers per-model override — check before any probe.
    # This closes the gap where /model switch and display paths used to fall
    # back to 128K despite the user having a per-model context_length set.
    # See.
    if custom_providers and base_url and model:
        try:
            from pilotage_cli.config import get_custom_provider_context_length
            cp_ctx = get_custom_provider_context_length(
                model=model,
                base_url=base_url,
                custom_providers=custom_providers,
            )
            if cp_ctx:
                return cp_ctx
        except Exception:
            pass  # fall through to probing

    # Malformed user-provided URLs (for example an unmatched IPv6 bracket)
    # make urllib.parse raise. Context resolution should treat those as an
    # unknown endpoint rather than crashing before the inference layer can
    # report the configuration error itself.
    if base_url:
        try:
            parsed_base_url = urlparse(_normalize_base_url(base_url))
            _ = parsed_base_url.port
        except ValueError:
            base_url = ""

    # An empty/blank model id can't be meaningfully resolved: every probe
    # below would either miss or — worse — fuzzy-match an arbitrary catalog
    # entry (the endpoint matcher's `model in key` check is vacuously true
    # for ""), returning whatever context length that random entry has and
    # persisting it under a junk "@<base_url>" cache key. Fall back to the
    # default immediately instead.
    if not str(model or "").strip():
        logger.info(
            "No model id provided for context length resolution — defaulting to %s tokens.",
            f"{DEFAULT_FALLBACK_CONTEXT:,}",
        )
        return DEFAULT_FALLBACK_CONTEXT

    # Normalise provider-prefixed model names (e.g. "local:model-name" →
    # "model-name") so cache lookups and server queries use the bare ID that
    # the endpoint actually knows about.
    model = _strip_provider_prefix(model)

    # 1. Check persistent cache (model+provider)
    # Codex OAuth is excluded because the authenticated /models catalogue is
    # account-specific and a fallback must never suppress later revalidation.
    if base_url and not _skip_persistent_context_cache(base_url, provider):
        cached = get_cached_context_length(model, base_url)
        if cached is not None:
            # Reject non-positive cached values — a 0 or negative value
            # is always a bug (corrupted cache, probe failure, or manual
            # edit).  Without this guard, `0 is not None` short-circuits
            # the resolution chain and the compressor gets context_length=0,
            # breaking every status-bar and /usage display downstream.
            if cached <= 0:
                logger.warning(
                    "Dropping non-positive cache entry %s@%s -> %s; re-resolving",
                    model, base_url, cached,
                )
                _invalidate_cached_context_length(model, base_url)
            else:
                return cached

    # 2. Active endpoint metadata for truly custom/unknown endpoints.
    # Known providers skip this — their /models endpoint may report a
    # provider-imposed limit instead of the model's full context. models.dev
    # has the correct per-provider values and is checked later.
    if _is_custom_endpoint(base_url) and not _is_known_provider_base_url(base_url):
        context_length = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
        if context_length is not None:
            return context_length
        if not _is_known_provider_base_url(base_url):
            # 3. Probe-down fallback after endpoint-specific detection failed
            logger.info(
                "Could not detect context length for model %r at %s — "
                "defaulting to %s tokens (probe-down). Set model.context_length "
                "in config.yaml to override.",
                model, base_url, f"{DEFAULT_FALLBACK_CONTEXT:,}",
            )
            # 3b. Before falling back to the hard 256K default, consult the
            # hardcoded catalog as a last resort. A proxied/custom gateway
            # fails the probes above, but the model name may still match an
            # entry in DEFAULT_CONTEXT_LENGTHS. Without this, the early
            # return here short-circuits the catalog lookup below.
            model_lower = model.lower()
            for default_model, length in sorted(
                DEFAULT_CONTEXT_LENGTHS.items(),
                key=lambda x: len(x[0]),
                reverse=True,
            ):
                if default_model in model_lower:
                    logger.info(
                        "Using hardcoded context length %s for model %r "
                        "(custom endpoint, catalog match on %r)",
                        f"{length:,}", model, default_model,
                    )
                    return length
            # Same silent-256K bug class as the step-9 fallback below —
            # warn here too so custom/local endpoints aren't left invisible.
            _warn_context_length_fallback(model, base_url)
            return DEFAULT_FALLBACK_CONTEXT

    # 4. Provider-aware lookups. The same model can have different context
    # limits per provider, so these win over the generic catalog below.
    # If provider is generic (custom/empty), try to infer it from the URL.
    effective_provider = provider
    if not effective_provider or effective_provider == "custom":
        if base_url:
            inferred = _infer_provider_from_url(base_url)
            if inferred:
                effective_provider = inferred

    if effective_provider == "openai-codex":
        # Codex OAuth enforces lower context limits than the direct OpenAI
        # API for the same slug (e.g. gpt-5.5 is 1.05M on the API but 272K
        # on Codex). Authoritative source is Codex's own /models endpoint.
        codex_ctx, codex_source = _resolve_codex_oauth_context_length_with_source(
            model, access_token=api_key or "",
        )
        if codex_ctx:
            # Only a successful authenticated catalogue response is safe to
            # persist. The static fallback is deliberately runtime-only so a
            # transient OAuth/network failure cannot poison future probes.
            if base_url and codex_source == "live":
                save_context_length(model, base_url, codex_ctx)
            return codex_ctx
    if effective_provider:
        from agent.models_dev import lookup_models_dev_context
        ctx = lookup_models_dev_context(effective_provider, model)
        if ctx:
            return ctx

    # 5. Hardcoded defaults (fuzzy match — longest key first for specificity)
    # Only check `default_model in model` (is the key a substring of the input).
    # The reverse (`model in default_model`) would let a shorter name match a
    # longer catalog key and return the wrong window.
    model_lower = model.lower()
    for default_model, length in sorted(
        DEFAULT_CONTEXT_LENGTHS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if default_model in model_lower:
            return length

    # 6. Default fallback — warn (deduped per model+endpoint) so
    #    small-context models don't silently get 256K. See
    #    _warn_context_length_fallback for rationale.
    _warn_context_length_fallback(model, base_url)
    return DEFAULT_FALLBACK_CONTEXT


async def get_model_context_length_async(
    model: str,
    base_url: str = "",
    api_key: str = "",
    config_context_length: int | None = None,
    provider: str = "",
    custom_providers: list | None = None,
) -> int:
    """Async variant of get_model_context_length.

    Offloads the entire synchronous resolution chain (which contains
    blocking HTTP calls via ``requests``) to a background thread so it
    does not freeze the asyncio event loop and cause Discord heartbeat
    timeouts.

    Shares all logic with the sync version — no code duplication.
    """
    import asyncio
    return await asyncio.to_thread(
        get_model_context_length,
        model,
        base_url=base_url,
        api_key=api_key,
        config_context_length=config_context_length,
        provider=provider,
        custom_providers=custom_providers,
    )


def _is_cjk_token_dense_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x1100 <= code <= 0x11FF  # Hangul Jamo
        or 0x2E80 <= code <= 0x9FFF  # CJK radicals/ideographs
        or 0xA960 <= code <= 0xA97F  # Hangul Jamo Extended-A
        or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
        or 0xF900 <= code <= 0xFAFF  # CJK compatibility ideographs
        or 0xFF00 <= code <= 0xFFEF  # Fullwidth forms / halfwidth kana
    )


# Same codepoint ranges as _is_cjk_token_dense_char, as a compiled character
# class so dense-char counting runs in C (``len(text) - len(re.sub(...))``)
# instead of a per-char Python loop.  MUST stay in sync with
# _is_cjk_token_dense_char.
_CJK_DENSE_RE = re.compile(
    "[\u1100-\u11ff"  # Hangul Jamo
    "\u2e80-\u9fff"  # CJK radicals/ideographs
    "\ua960-\ua97f"  # Hangul Jamo Extended-A
    "\uac00-\ud7af"  # Hangul Syllables
    "\uf900-\ufaff"  # CJK compatibility ideographs
    "\uff00-\uffef]"  # Fullwidth forms / halfwidth kana
)


def estimate_tokens_rough(text: str) -> int:
    """Rough token estimate for pre-flight checks.

    Uses ceiling division so short texts (1-3 chars) never estimate as
    0 tokens, which would cause the compressor and pre-flight checks to
    systematically undercount when many short tool results are present.
    CJK/Hangul/Kana text is much denser than English under common LLM
    tokenizers, so count those codepoints as roughly one token each instead
    of applying the English-centric ~4 chars/token rule.

    Perf: this runs on every message in every preflight/compaction walk,
    including MB-scale tool outputs, so the common all-ASCII case must stay
    O(1).  ``str.isascii()`` is a flag check on CPython's compact unicode
    representation (no scan), and the CJK counting itself is a single
    C-level ``re.findall`` rather than a per-character Python loop.
    """
    if not text:
        return 0
    text = str(text)
    if text.isascii():
        # O(1) fast path — ASCII text cannot contain token-dense CJK chars.
        return (len(text) + 3) // 4
    dense = len(text) - len(_CJK_DENSE_RE.sub("", text))
    if not dense:
        # Non-ASCII but no CJK (accents, Cyrillic, emoji, ...): keep the
        # classic ~4 chars/token rule.
        return (len(text) + 3) // 4
    sparse = len(text) - dense
    return dense + ((sparse + 3) // 4)


def estimate_messages_tokens_rough(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate for a message list (pre-flight only).

    Image parts (base64 PNG/JPEG) are counted as a flat ~1500 tokens per
    image — a typical provider pricing model — instead of counting raw base64
    character length. Without this, a single ~1MB screenshot would be
    estimated at ~250K tokens and trigger premature context compression.

    Per-message results are memoized (see ``_estimate_message_tokens_cached``)
    keyed on a deep *identity fingerprint* of the message, so re-walking a
    long history every iteration only pays for messages whose object graph
    actually changed. The memo is exact: equal fingerprints imply identical
    leaf objects and structure, hence an identical estimate.
    """
    _IMAGE_TOKEN_COST = 1500
    total = 0
    for msg in messages:
        total += _estimate_message_tokens_cached(msg, _IMAGE_TOKEN_COST)
    return total


# --- Per-message token-estimate memo -------------------------------------
#
# ``estimate_messages_tokens_rough`` is called on the full history every
# loop iteration (conversation_loop preflight), repeatedly during compaction
# telemetry, and inside an O(n^2) shrink loop in moa_loop. The per-message
# helpers are pure functions of the message's value, so a memo keyed on a
# fingerprint that uniquely determines the value is exactly equivalent.
#
# Fingerprint design (soundness argument):
#   * strings are fingerprinted by ``id()`` AND pinned (a strong reference is
#     stored in the cache entry). While the entry lives, that id cannot be
#     reused by another object, so id-equality implies object-equality —
# strings are immutable, so value-equality too (no-style aliasing).
#   * ints/floats/bools/None are fingerprinted by value.
#   * dicts/lists recurse structurally, preserving key order — ``str(shadow)``
#     depends on insertion order, so order is part of the key.
#   * any other type aborts the memo and falls through to a direct compute.
# Equal fingerprints therefore imply deep-equal messages built from identical
# immutable leaves ⇒ identical ``str(shadow)`` bytes ⇒ identical estimate.
#
# Because the api_messages build shallow-copies history dicts each iteration,
# the copies share the same content strings — so unchanged history messages
# hit the memo even though the outer dicts are fresh objects every turn.
_MSG_TOKENS_CACHE: Dict[Any, Tuple[list, int]] = {}
_MSG_TOKENS_CACHE_MAX = 4096


def _msg_fingerprint(value: Any, pins: list) -> Any:
    if value is None or value is True or value is False:
        return value
    t = type(value)
    if t is str:
        pins.append(value)
        return ("s", id(value))
    if t is int or t is float:
        return ("n", t.__name__, value)
    if t is dict:
        return ("d", tuple(
            (_msg_fingerprint(k, pins), _msg_fingerprint(v, pins))
            for k, v in value.items()
        ))
    if t is list:
        return ("l", tuple(_msg_fingerprint(v, pins) for v in value))
    if t is tuple:
        return ("t", tuple(_msg_fingerprint(v, pins) for v in value))
    raise ValueError("unfingerprintable message value")


def _estimate_message_tokens_cached(msg: Any, image_cost: int) -> int:
    try:
        pins: list = []
        key = _msg_fingerprint(msg, pins)
        hash(key)
    except Exception:
        return (
            _estimate_message_tokens_without_images(msg)
            + _count_image_tokens(msg, image_cost)
        )
    cached = _MSG_TOKENS_CACHE.get(key)
    if cached is not None:
        return cached[1]
    tokens = (
        _estimate_message_tokens_without_images(msg)
        + _count_image_tokens(msg, image_cost)
    )
    _MSG_TOKENS_CACHE[key] = (pins, tokens)
    while len(_MSG_TOKENS_CACHE) > _MSG_TOKENS_CACHE_MAX:
        try:
            _MSG_TOKENS_CACHE.pop(next(iter(_MSG_TOKENS_CACHE)))
        except (StopIteration, KeyError, RuntimeError):
            break
    return tokens


def _count_image_tokens(msg: Dict[str, Any], cost_per_image: int) -> int:
    """Count image-like content parts in a message; return their token cost."""
    count = 0
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in {"image", "image_url", "input_image"}:
                count += 1
    # Multimodal tool results that haven't been converted yet.
    if isinstance(content, dict) and content.get("_multimodal"):
        inner = content.get("content")
        if isinstance(inner, list):
            for part in inner:
                if isinstance(part, dict) and part.get("type") in {"image", "image_url"}:
                    count += 1
    return count * cost_per_image


def _wire_message_shadow(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Shadow of a message holding only what the provider actually receives.

    Two adjustments to the raw persisted dict:

    * ``api_content`` is a SUBSTITUTE for ``content``, not an addition to it.
      ``turn_context.substitute_api_content()`` pops the sidecar and overwrites
      ``content`` at every API-bound build site, so exactly one of the two is
      ever sent. Counting both double-counts any message whose sidecar differs
      from its clean stored content (2.00x on a 40KB sidecar).

      The substitution mirrors that helper's guard exactly: only a non-empty
      STRING sidecar on a ``user``/``assistant`` row displaces ``content``.
      Any other sidecar shape is popped and discarded on the wire without
      touching ``content``, so a shadow that substituted unconditionally
      would UNDERcount those rows — the dangerous direction, since it makes
      compaction fire too late and the turn dies on a hard context error.
    * Base64 image payloads are replaced with a placeholder; they are charged
      separately at a flat rate by ``_count_image_tokens``, and counting their
      raw chars here would massively overestimate usage.
    """
    sidecar = msg.get("api_content")
    sidecar_wins = (
        isinstance(sidecar, str)
        and bool(sidecar)
        and msg.get("role") in ("user", "assistant")
    )
    shadow: Dict[str, Any] = {}
    for k, v in msg.items():
        if k == "reasoning_details" or k in PERSISTENCE_ONLY_MESSAGE_FIELDS:
            continue
        if k == "api_content":
            # Always popped before the request is built; only counted when it
            # actually replaces ``content``.
            if sidecar_wins:
                shadow["content"] = v
            continue
        if k == "content":
            if sidecar_wins:
                # The sidecar wins on the wire; skip the clean copy so the
                # same logical content is not counted twice.
                continue
            if isinstance(v, list):
                cleaned = []
                for part in v:
                    if isinstance(part, dict):
                        if part.get("type") in {"image", "image_url", "input_image"}:
                            cleaned.append({"type": part.get("type"), "image": "[stripped]"})
                        else:
                            cleaned.append(part)
                    else:
                        cleaned.append(part)
                shadow[k] = cleaned
            elif isinstance(v, dict) and v.get("_multimodal"):
                shadow[k] = v.get("text_summary", "")
            else:
                shadow[k] = v
        else:
            shadow[k] = v
    return shadow


def _estimate_message_chars(msg: Dict[str, Any]) -> int:
    """Char count for token estimation, excluding base64 image data.

    Base64 images are counted via `_count_image_tokens` instead; including
    their raw chars here would massively overestimate token usage.
    """
    if not isinstance(msg, dict):
        return len(str(msg))
    return len(str(_wire_message_shadow(msg)))


def _estimate_message_tokens_without_images(msg: Dict[str, Any]) -> int:
    """Token estimate for a message shadow with image payloads stripped."""
    if not isinstance(msg, dict):
        return estimate_tokens_rough(str(msg))
    return estimate_tokens_rough(str(_wire_message_shadow(msg)))


def estimate_request_tokens_rough(
    messages: List[Dict[str, Any]],
    *,
    system_prompt: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Rough token estimate for a full chat-completions request.

    Includes the major payload buckets Pilotage sends to providers:
    system prompt, conversation messages, and tool schemas.  With 50+
    tools enabled, schemas alone can add 20-30K tokens — a significant
    blind spot when only counting messages. Image content is counted
    at a flat per-image cost (see estimate_messages_tokens_rough).
    """
    total = 0
    if system_prompt:
        total += estimate_tokens_rough(system_prompt)
    if messages:
        total += estimate_messages_tokens_rough(messages)
    if tools:
        total += _estimate_tools_tokens_rough(tools)
    return total


# NOTE: tool schemas can be large. Avoid repeated `str(tools)` conversions,
# which are CPU-heavy and can stall GUI event loops under GIL pressure.
#
# Keyed by ``id(tools)``. A long-lived gateway/desktop backend builds many
# transient tool lists over its lifetime, so the cache is bounded and evicts
# oldest-first (insertion-ordered dict) once it exceeds the cap. The cap is
# generous relative to how rarely toolsets are rebuilt within a process.
_TOOLS_TOKENS_CACHE: dict[int, Tuple[int, str, str, int]] = {}
_TOOLS_TOKENS_CACHE_MAX = 256


def _tool_name_for_cache(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if isinstance(name, str):
            return name
    name = tool.get("name")
    return name if isinstance(name, str) else ""


def _estimate_tools_tokens_rough(tools: List[Dict[str, Any]]) -> int:
    if not tools:
        return 0

    # Cache by list identity. Tools are rebuilt rarely (toolset changes),
    # but token estimates are requested frequently (preflight, compaction).
    key = id(tools)
    n = len(tools)
    first = _tool_name_for_cache(tools[0]) if n else ""
    last = _tool_name_for_cache(tools[-1]) if n else ""

    cached = _TOOLS_TOKENS_CACHE.get(key)
    if cached is not None:
        cached_n, cached_first, cached_last, cached_tokens = cached
        if cached_n == n and cached_first == first and cached_last == last:
            return cached_tokens

    # Fast, stable rough estimate: sum lengths of the major schema fields.
    # This avoids the pathological `str(tools)` path while still scaling with
    # schema size (descriptions + parameters dominate).
    total_chars = 0
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict):
            name = fn.get("name") or ""
            desc = fn.get("description") or ""
            params = fn.get("parameters") or {}
        else:
            name = tool.get("name") or ""
            desc = tool.get("description") or ""
            params = tool.get("parameters") or {}

        if isinstance(name, str):
            total_chars += len(name)
        if isinstance(desc, str):
            total_chars += len(desc)
        # Parameters can be nested; JSON is closer to over-the-wire size than repr().
        try:
            total_chars += len(json.dumps(params, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            total_chars += len(str(params))

    tokens = (total_chars + 3) // 4
    # Bound the cache: drop the oldest entry when the cap is exceeded so a
    # long-running process can't accumulate an unbounded number of stale
    # ``id(tools)`` entries (id values are recycled after GC anyway).
    if len(_TOOLS_TOKENS_CACHE) >= _TOOLS_TOKENS_CACHE_MAX:
        _TOOLS_TOKENS_CACHE.pop(next(iter(_TOOLS_TOKENS_CACHE)), None)
    _TOOLS_TOKENS_CACHE[key] = (n, first, last, tokens)
    return tokens
